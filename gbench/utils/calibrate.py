# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hardware calibration.

`gbench --calibrate` inspects the local hardware (GPUs, their occupancy and VRAM, CPU cores, RAM,
Docker and its network address pools) and prints recommended `--num-gpus`, `--sandboxes` and
`--eval-concurrency` values with the reason for each, so a user does not have to guess. Given
the run's `--models` it sizes tensor parallelism for them; given its `--evals` it recommends
`--sandboxes` only when a containerized suite is selected. The same detection backs a pre-run
guard that refuses, with an actionable message, a sandbox concurrency that would over-subscribe
the host's CPU/RAM, or (for suites that create a docker network per task) exceed the free
networks in Docker's address pools, instead of letting the run OOM or fail tasks mid-sweep.

The recommendation is deliberately conservative and heuristic; it is a starting point, not a
guarantee. The knobs it sizes:

* `--sandboxes` (concurrency for containerized evals): each concurrent sandbox runs a heavy Docker
  task container (compilers, test suites, model rollouts), budgeted here at about
  `_CORES_PER_SANDBOX` cores and `_RAM_GB_PER_SANDBOX` GB, and, for PER_TASK_NETWORK_SUITES, one
  docker network from the daemon's address pools.
* `--eval-concurrency` (the eval client's HTTP concurrency to the served model): bound by the
  served model's throughput, not local CPU, so the suggestion is a moderate starting concurrency.
  It is not a sandbox count and the pre-run guard does not check it.
* `--num-gpus`: the idle GPUs (no compute process, at least IDLE_FREE_FRACTION of their memory
  free), and with --models the tensor-parallel size each model needs.
"""

import ipaddress
import json
import os
import shutil
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# A containerized-eval sandbox runs a heavy Docker task; budget cores + RAM per concurrent sandbox.
_CORES_PER_SANDBOX = 2
_RAM_GB_PER_SANDBOX = 8
#: env override to bypass the pre-run over-subscription guard (explicit opt-out).
GUARD_BYPASS_ENV = "GBENCH_SKIP_CALIBRATION_GUARD"

#: Suites that put each concurrent task on a docker network of its own, taken from the daemon's
#: address pools: Harbor's docker compose environment per trial (terminal_bench, swe_bench_pro)
#: and the user-defined network alcatraz creates per task (swe_lancer). N concurrent tasks need N
#: free networks (docs/evals/docker-sandboxes.md).
PER_TASK_NETWORK_SUITES = frozenset({"terminal_bench", "swe_bench_pro", "swe_lancer"})

#: Docker's built-in default-address-pools, used when daemon.json sets none (`docker info`
#: reports DefaultAddressPools as null): 15 /16 networks in 172.17.0.0-172.31.255.255 and 16 /20
#: networks in 192.168.0.0/16, 31 in all (the default bridge docker0 takes one of them).
DOCKER_DEFAULT_ADDRESS_POOLS: Tuple[Tuple[str, int], ...] = (
    ("172.17.0.0/16", 16), ("172.18.0.0/16", 16), ("172.19.0.0/16", 16),
    ("172.20.0.0/14", 16), ("172.24.0.0/14", 16), ("172.28.0.0/14", 16),
    ("192.168.0.0/16", 20),
)

#: The daemon.json remedy printed when the networks run out (docs/evals/docker-sandboxes.md).
_POOL_REMEDY = ('give Docker a larger address pool in /etc/docker/daemon.json, e.g. '
                '{"default-address-pools": [{"base": "172.28.0.0/14", "size": 24}]} (1024 '
                'networks; pick a range unused on this host), restart docker, and remove leftover '
                'networks (see docs/evals/docker-sandboxes.md)')


def _cpu_cores() -> int:
    """Usable CPU cores, affinity-aware (respects cgroup/taskset limits), min 1."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def _ram_gb() -> Tuple[float, float]:
    """(total_gb, available_gb). Reads Linux /proc/meminfo; falls back to sysconf for total and
    assumes ~80% available when MemAvailable is missing."""
    total = avail = 0.0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total = int(line.split()[1]) / (1024 * 1024)
                elif line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        pass
    if total <= 0:
        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
        except (ValueError, OSError):
            total = 0.0
    if avail <= 0:
        avail = total * 0.8
    return round(total, 1), round(avail, 1)


def _run(cmd: List[str], timeout: int = 15) -> Optional[str]:
    """stdout of a read-only command, or None when it fails."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:                                                      # noqa: BLE001
        return None
    return r.stdout if r.returncode == 0 else None


def _docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def parse_address_pools(raw: Optional[str]) -> Optional[Tuple[List[Tuple[str, int]], str]]:
    """(pools as [(base CIDR, subnet size)], source) from `docker info --format
    '{{json .DefaultAddressPools}}'`: null/empty -> Docker's built-in defaults; a list of
    {"Base", "Size"} -> the daemon's configured pools. None when the output cannot be read."""
    if raw is None:
        return None
    text = raw.strip()
    if not text or text in ("null", "[]", "<no value>"):
        return [tuple(p) for p in DOCKER_DEFAULT_ADDRESS_POOLS], "docker default"
    try:
        data = json.loads(text)
    except ValueError:
        return None
    pools = []
    for p in data or []:
        if not isinstance(p, dict):
            continue
        base, size = p.get("Base"), p.get("Size")
        if isinstance(base, str) and isinstance(size, int):
            pools.append((base, size))
    if not pools:
        return [tuple(p) for p in DOCKER_DEFAULT_ADDRESS_POOLS], "docker default"
    return pools, "daemon.json default-address-pools"


def network_capacity(pools: Sequence[Tuple[str, int]], used_subnets: Iterable[str]
                     ) -> Dict[str, int]:
    """{"capacity", "in_use", "free"}: the networks the pools can be carved into (the sum of
    2^(size - prefix) per pool) and how many of them existing networks already hold (the IPv4
    subnets inside a pool)."""
    nets = []
    capacity = 0
    for base, size in pools:
        try:
            net = ipaddress.ip_network(base, strict=False)
        except ValueError:
            continue
        if net.version != 4 or size < net.prefixlen:
            continue
        nets.append(net)
        capacity += 2 ** (size - net.prefixlen)
    in_use = 0
    for s in used_subnets:
        try:
            sub = ipaddress.ip_network(str(s).strip(), strict=False)
        except ValueError:
            continue
        if sub.version == 4 and any(sub.subnet_of(n) for n in nets):
            in_use += 1
    return {"capacity": capacity, "in_use": in_use, "free": max(0, capacity - in_use)}


def _docker_networks() -> Optional[Dict[str, Any]]:
    """The daemon's address-pool capacity and the networks in use (network_capacity), with the
    pools' source; None when docker cannot be queried."""
    parsed = parse_address_pools(_run(["docker", "info", "--format",
                                       "{{json .DefaultAddressPools}}"]))
    if parsed is None:
        return None
    pools, source = parsed
    ids = (_run(["docker", "network", "ls", "-q"]) or "").split()
    subnets: List[str] = []
    if ids:
        out = _run(["docker", "network", "inspect", *ids, "--format",
                    "{{range .IPAM.Config}}{{.Subnet}} {{end}}"], timeout=30)
        if out is None:
            return None
        subnets = out.split()
    info = network_capacity(pools, subnets)
    info["pools_source"] = source
    return info


def _gpu_occupancy() -> Optional[List[Any]]:
    from ..core.config import gpu_inventory
    return gpu_inventory()


def detect_hardware() -> Dict[str, Any]:
    """Inspect local hardware. GPU/VRAM values are 0 when nvidia-smi is absent (fail open).

    ``gpus_idle`` counts the visible GPUs (CUDA_VISIBLE_DEVICES honored) that run no compute
    process and have at least IDLE_FREE_FRACTION of their memory free, the ones a run's automatic
    GPU selection can use. ``docker_networks`` is Docker's address-pool capacity and use (None
    without Docker or when it cannot be read)."""
    from ..core.config import (IDLE_FREE_FRACTION, get_available_gpus, _gpu_total_vram_gb,
                               _gpu_min_free_vram_gb)
    total_ram, avail_ram = _ram_gb()
    inventory = _gpu_occupancy()
    if inventory is not None:
        idle = [g for g in inventory if g.idle and (g.total_gb <= 0
                                                    or g.free_gb >= IDLE_FREE_FRACTION * g.total_gb)]
        gpus = len(inventory)
        gpus_idle = len(idle)
        idle_free = round(min((g.free_gb for g in idle), default=0.0), 1)
        idle_total = round(min((g.total_gb for g in idle), default=0.0), 1)
    else:
        gpus = get_available_gpus()
        gpus_idle = gpus
        idle_free = idle_total = None
    docker = _docker_ok()
    return {
        "gpus": gpus,
        "gpus_idle": gpus_idle,
        "gpu_total_vram_gb": round(_gpu_total_vram_gb(), 1),
        "gpu_free_vram_gb": round(_gpu_min_free_vram_gb(), 1),
        "gpu_idle_free_vram_gb": idle_free,
        "gpu_idle_total_vram_gb": idle_total,
        "cpu_cores": _cpu_cores(),
        "ram_total_gb": total_ram,
        "ram_available_gb": avail_ram,
        "docker": docker,
        "docker_networks": _docker_networks() if docker else None,
    }


def _network_suites(suites: Optional[Iterable[str]]) -> List[str]:
    return sorted({s for s in (suites or []) if s in PER_TASK_NETWORK_SUITES})


def max_safe_sandboxes(hw: Dict[str, Any], suites: Optional[Iterable[str]] = None) -> int:
    """The most concurrent container-eval sandboxes this host should run (0 if Docker is absent):
    min(cores / _CORES_PER_SANDBOX, available RAM / _RAM_GB_PER_SANDBOX), at least 1, and, when
    ``suites`` include a PER_TASK_NETWORK_SUITES suite, at most the free docker networks."""
    if not hw.get("docker"):
        return 0
    cores, ram = hw["cpu_cores"], hw["ram_available_gb"]
    n = max(1, min(cores // _CORES_PER_SANDBOX, int(ram // _RAM_GB_PER_SANDBOX)))
    nets = hw.get("docker_networks")
    if _network_suites(suites) and isinstance(nets, dict) and "free" in nets:
        n = min(n, int(nets["free"]))
    return n


def _largest_pow2(n: int) -> int:
    """Largest power of two <= n (>=1 when n>=1, else 0). Tensor-parallel size must be a
    power of two, and validate_gpu_config hard-rejects other counts, so calibrate must not
    recommend e.g. 6 GPUs on a 6-GPU host - it recommends 4."""
    if n < 1:
        return 0
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _model_tp(model: Any, hw: Dict[str, Any]) -> Dict[str, Any]:
    """The tensor-parallel size one model needs: the larger of its size tier (get_num_gpus) and
    the smallest power of two whose per-GPU weight share (weights_fit_advisory's formula) fits
    the idle GPUs' usable memory, and whether the idle GPUs can host it."""
    from ..core.config import (_PER_RANK_OVERHEAD_GB, get_gpu_memory_utilization,
                               get_num_gpus)
    params = float(getattr(model, "total_params_b", 0.0) or 0.0)
    name = getattr(model, "name", None) or getattr(model, "short_name", "?")
    if params <= 0:
        return {"model": name, "tp": None,
                "reason": "size unknown (config.json not read): pass --num-gpus yourself"}
    tier = get_num_gpus(params)
    util = get_gpu_memory_utilization(params)
    free = hw.get("gpu_idle_free_vram_gb")
    total = hw.get("gpu_idle_total_vram_gb")
    caps = [c for c in (free, (total or 0) * util) if isinstance(c, (int, float)) and c > 0]
    fit = None
    if caps:
        usable = min(caps)
        for p in (1, 2, 4, 8, 16):
            if params * 2.0 / p + _PER_RANK_OVERHEAD_GB <= usable:
                fit = p
                break
    tp = max(tier, fit) if fit else tier
    reason = f"{params:.0f}B: size tier {tier} GPU(s)"
    if fit:
        reason += f", weights fit from {fit} GPU(s) (~{min(caps):.0f} GB usable per idle GPU)"
    elif caps:
        reason += f", weights do not fit even 16 GPUs of ~{min(caps):.0f} GB usable"
    idle = int(hw.get("gpus_idle") or 0)
    if tp > idle:
        reason += f"; needs {tp} but only {idle} GPU(s) are idle"
    return {"model": name, "tp": tp, "reason": reason, "exceeds_idle": tp > idle}


def recommend(hw: Dict[str, Any], models: Optional[Sequence[Any]] = None,
              evals: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Recommended knob values for this hardware, with a one-line reason per value (``reasons``).

    ``models`` (the run's --models) sizes --num-gpus per model; ``evals`` (the selected suites,
    'all' expanded) gives a --sandboxes value only when a containerized suite
    (eval_suites.SANDBOXED_SUITES) is selected, capped by the free docker networks for
    PER_TASK_NETWORK_SUITES."""
    cores = hw["cpu_cores"]
    reasons: Dict[str, str] = {}
    idle = int(hw.get("gpus_idle", hw.get("gpus", 0)) or 0)
    per_model: List[Dict[str, Any]] = []
    if models:
        per_model = [_model_tp(m, hw) for m in models]
        tps = [pm["tp"] for pm in per_model if pm["tp"]]
        num_gpus = max(tps) if tps else None
        reasons["num_gpus"] = "; ".join(f"{pm['model']}: {pm['reason']}" for pm in per_model)
    else:
        num_gpus = _largest_pow2(idle) if idle else None
        reasons["num_gpus"] = (
            f"{idle} of {hw.get('gpus', idle)} GPU(s) idle (no compute process, most memory free); "
            "pass --models to size tensor parallelism for a model" if hw.get("gpus")
            else "no GPU detected")
    sandboxed: List[str] = []
    if evals is not None:
        from ..runners.eval_suites import SANDBOXED_SUITES
        sandboxed = sorted({s for s in evals if s in SANDBOXED_SUITES})
    if evals is not None and not sandboxed:
        sandboxes = None
        reasons["sandboxes"] = "no containerized suite selected (--sandboxes does not apply)"
    else:
        sandboxes = max_safe_sandboxes(hw, evals)
        why = (f"{cores} cores / {_CORES_PER_SANDBOX} and "
               f"{hw['ram_available_gb']:.0f} GB / {_RAM_GB_PER_SANDBOX} GB per sandbox")
        nets = hw.get("docker_networks")
        if _network_suites(evals) and isinstance(nets, dict):
            why += (f", {nets.get('free')} free docker network(s) for "
                    f"{', '.join(_network_suites(evals))}")
        elif (evals is None and hw.get("docker") and isinstance(nets, dict) and "free" in nets
              and int(nets["free"]) < sandboxes):
            # No suite selection to judge by: name the lower limit of the suites that need one
            # docker network per sandbox (the run refuses more than the free networks for them).
            capped = int(nets["free"])
            why += (f"; at most {capped} for {', '.join(sorted(PER_TASK_NETWORK_SUITES))} "
                    f"({capped} free docker network(s), one per sandbox)")
        reasons["sandboxes"] = why if hw.get("docker") else "Docker not available"
    eval_concurrency = max(1, min(64, cores * 4))
    reasons["eval_concurrency"] = ("HTTP concurrency to the served model, endpoint-bound: tune it to "
                                   "the server; not a sandbox count, not guarded")
    network_capped = None
    if (evals is None and hw.get("docker") and isinstance(hw.get("docker_networks"), dict)
            and "free" in hw["docker_networks"] and sandboxes is not None
            and int(hw["docker_networks"]["free"]) < sandboxes):
        network_capped = int(hw["docker_networks"]["free"])
    return {
        "num_gpus": num_gpus,
        "per_model": per_model,
        "sandboxes": sandboxes,
        # without --evals: the most sandboxes for the suites that need a docker network each
        "sandboxes_network_suites": network_capped,
        "sandboxed_suites": sandboxed,
        # HTTP concurrency to the served model; endpoint-throughput-bound, so a moderate default.
        "eval_concurrency": eval_concurrency,
        "reasons": reasons,
    }


def check_oversubscription(requested_sandboxes: Any, hw: Dict[str, Any],
                           suites: Optional[Iterable[str]] = None) -> Tuple[bool, str]:
    """(ok, message). Not ok when a requested sandbox count clearly exceeds CPU/RAM capacity, or,
    when ``suites`` (the selected suites) include a PER_TASK_NETWORK_SUITES suite, exceeds the free
    networks of Docker's address pools (each concurrent task creates one; past the pool, network
    creation fails and the task is lost as a harness error).

    A bypass via GUARD_BYPASS_ENV always returns ok (explicit opt-out). The CPU/RAM cap allows
    headroom over the conservative recommendation so a mild over-shoot is not blocked, only a clear
    one; the network cap is exact."""
    if os.environ.get(GUARD_BYPASS_ENV):
        return True, ""
    try:
        n = int(requested_sandboxes)
    except (TypeError, ValueError):
        return True, ""
    if n <= 0:
        return True, ""
    if not hw.get("docker"):
        # No Docker: container evals hard-error on their own prereq check; nothing to guard here.
        return True, ""
    rec = max_safe_sandboxes(hw)
    hard_cap = max(hw["cpu_cores"], rec * 3)   # headroom: block only a clear over-subscription
    if n > hard_cap:
        return False, (
            f"--sandboxes {n} over-subscribes this host ({hw['cpu_cores']} CPU cores, "
            f"{hw['ram_available_gb']:.0f} GB RAM available): a containerized-eval sandbox needs "
            f"about {_CORES_PER_SANDBOX} cores and {_RAM_GB_PER_SANDBOX} GB each. Recommended "
            f"--sandboxes {rec}. Lower it, run `gbench --calibrate`, or set {GUARD_BYPASS_ENV}=1 "
            f"to override.")
    nets = hw.get("docker_networks")
    net_suites = _network_suites(suites)
    if net_suites and isinstance(nets, dict) and n > int(nets.get("free", n)):
        return False, (
            f"--sandboxes {n} exceeds the {nets.get('free')} free docker network(s) "
            f"({nets.get('in_use')} of {nets.get('capacity')} in use, "
            f"{nets.get('pools_source')}): {', '.join(net_suites)} create(s) one network per "
            f"concurrent task, and the tasks past the pool would fail. Lower --sandboxes to "
            f"{nets.get('free')} or less, or {_POOL_REMEDY}. Set {GUARD_BYPASS_ENV}=1 to "
            "override.")
    return True, ""


def format_report(hw: Dict[str, Any], rec: Dict[str, Any]) -> str:
    idle = hw.get("gpus_idle", hw.get("gpus"))
    gpu_line = (f"{hw['gpus']} GPU(s), {hw['gpu_total_vram_gb']:.0f} GB VRAM each "
                f"({hw['gpu_free_vram_gb']:.0f} GB free on the fullest), {idle} idle"
                if hw["gpus"] else "none detected (nvidia-smi absent or no visible GPU)")
    docker_line = "available" if hw["docker"] else "NOT available (containerized evals will hard-error)"
    nets = hw.get("docker_networks")
    net_line = (f"  Docker networks: {nets.get('free')} free of {nets.get('capacity')} "
                f"({nets.get('pools_source')})" if isinstance(nets, dict) else None)
    reasons = rec.get("reasons") or {}
    lines = [
        "gbench hardware calibration",
        "",
        "Detected:",
        f"  GPU:    {gpu_line}",
        f"  CPU:    {hw['cpu_cores']} usable cores",
        f"  RAM:    {hw['ram_total_gb']:.0f} GB total, {hw['ram_available_gb']:.0f} GB available",
        f"  Docker: {docker_line}",
    ]
    if net_line:
        lines.append(net_line)
    lines += ["", "Recommended flags:"]
    if rec["num_gpus"]:
        lines.append(f"  --num-gpus {rec['num_gpus']}")
    else:
        lines.append("  --num-gpus: n/a (no idle local GPU; free one, or use --remote-endpoint "
                     "against a served model)")
    if reasons.get("num_gpus"):
        lines.append(f"      reason: {reasons['num_gpus']}")
    if rec["sandboxes"]:
        lines.append(f"  --sandboxes {rec['sandboxes']}   (concurrency for containerized evals)")
    elif rec["sandboxes"] is None:
        lines.append("  --sandboxes: n/a")
    else:
        lines.append("  --sandboxes: n/a (Docker not available or no free docker network; "
                     "containerized evals cannot run)")
    if reasons.get("sandboxes"):
        lines.append(f"      reason: {reasons['sandboxes']}")
    if rec.get("sandboxes_network_suites") is not None:
        lines.append(f"  --sandboxes {rec['sandboxes_network_suites']}   (for "
                     f"{', '.join(sorted(PER_TASK_NETWORK_SUITES))}: one docker network per "
                     "sandbox)")
    lines.append(f"  --eval-concurrency {rec['eval_concurrency']}   (eval client HTTP concurrency)")
    if reasons.get("eval_concurrency"):
        lines.append(f"      reason: {reasons['eval_concurrency']}")
    lines += [
        "",
        "Notes:",
        "  - --num-gpus is the tensor-parallel size unless --tensor-parallel splits it into "
        "data-parallel replicas.",
        "  - --sandboxes is a conservative starting point; raise it if the host stays underloaded.",
        f"  - A requested --sandboxes above about {max(hw['cpu_cores'], max_safe_sandboxes(hw) * 3)} "
        f"is refused pre-run unless {GUARD_BYPASS_ENV}=1"
        + (f", and above the free docker networks for {', '.join(sorted(PER_TASK_NETWORK_SUITES))}."
           if isinstance(nets, dict) else "."),
    ]
    return "\n".join(lines)


def calibration_report(models: Optional[Sequence[Any]] = None,
                       evals: Optional[Sequence[str]] = None) -> str:
    """Detect + recommend + format, for `gbench --calibrate` (read-only: no server starts).
    ``models`` and ``evals`` are the run's resolved --models and selected suites when given."""
    hw = detect_hardware()
    return format_report(hw, recommend(hw, models=models, evals=evals))
