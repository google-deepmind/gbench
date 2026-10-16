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

"""Benchmark configuration management.

This module derives batch sizes and timeouts directly from model parameters
(total_params_b and is_moe), rather than hand-tuned lookup tables.

GPU allocation is NOT auto-applied to the served run: the user must pass
``--num-gpus`` for local performance runs (see cli.py's local-run guard).
``get_num_gpus`` below encodes the fair-tier RECOMMENDATION and is used by
``validate_gpu_config`` to warn/abort when the requested count is off-tier or
won't fit - it does not itself set the tensor-parallel degree. The fair tiers:
    <=20B total params  -> 1 GPU
    <=80B total params  -> 2 GPUs
    >80B total params   -> 8 GPUs
"""

import os
import re
import shlex
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple


# -- Param-based benchmark functions --


def get_num_gpus(total_params_b: float) -> int:
    """Derive GPU count from total model parameters.

    Uses discrete tiers for fair benchmarking - all models in the same
    size range get identical GPU allocation regardless of vendor.

    Boundaries chosen so competing models always share the same tier:
      <=20B:  1 GPU  - small/mid dense and edge MoE
      <=80B:  2 GPUs - large dense (23-36B), mid MoE
      >80B:  8 GPUs - all large MoE (100B+) on equal footing
    """
    if total_params_b <= 20:
        return 1
    elif total_params_b <= 80:
        return 2
    else:
        return 8


def get_tensor_parallel(total_params_b: float) -> int:
    """Tensor parallel size = GPU count (1:1 mapping)."""
    return get_num_gpus(total_params_b)


def get_batch_sizes(total_params_b: float, preset: str = "default") -> list[int]:
    """Default concurrency is a single stream (batch size 1) for every model and
    preset. Concurrency sweeping is opt-in via --batch-sizes; the load dimension
    is otherwise covered by the stress test's arrival-rate (QPS) sweep.

    Args:
        total_params_b: Total model parameters in billions (unused; kept for API).
        preset: Benchmark preset (unused; kept for API).
    """
    return [1]


# -- Per-campaign SLOs (capacity-under-SLO) --
# TTFT (first-byte) budget scales with prefill/input size - an 8k-token prompt
# legitimately tolerates a longer first token than a quick chat. ITL/TBT (decode
# cadence) is prompt-independent (~human reading speed, ~10 tok/s). These are
# DEFAULTS: overridable globally (--stress-threshold / --itl-slo-ms) or per
# campaign, and frozen into the run fingerprint so a run's SLOs are reproducible
# and applied IDENTICALLY to every model being compared (tunable across runs,
# fixed within a run).
CAMPAIGN_SLOS: dict[str, dict[str, int]] = {
    "chat-like":     {"ttft_ms": 1000, "itl_ms": 100},
    "decode-heavy":  {"ttft_ms": 1000, "itl_ms": 100},
    "mixed":         {"ttft_ms": 2500, "itl_ms": 100},
    "agentic":       {"ttft_ms": 5000, "itl_ms": 100},
    "prefill-heavy": {"ttft_ms": 5000, "itl_ms": 100},
    "long-decode":   {"ttft_ms": 5000, "itl_ms": 100},
}
_DEFAULT_SLO = {"ttft_ms": 5000, "itl_ms": 200}   # no-campaign fallback


def resolve_campaign_slo(campaign, ttft_override=None, itl_override=None) -> tuple[int, int]:
    """Resolve (ttft_ms, itl_ms) for a campaign.

    Precedence: explicit override (global, applies to all campaigns) > per-campaign
    default > no-campaign fallback. A None override means 'use the per-campaign
    default' - which is why the CLI SLO flags default to None, not a number.
    """
    base = CAMPAIGN_SLOS.get(campaign, _DEFAULT_SLO)
    ttft = ttft_override if ttft_override is not None else base["ttft_ms"]
    itl = itl_override if itl_override is not None else base["itl_ms"]
    return int(ttft), int(itl)


def campaign_ctx(campaign, input_lengths, output_lengths) -> int:
    """Auto-derive the served max_model_len for a campaign.

    Clears the RandomDataset 1.5x length tail (stress samples range_ratio=0.5) with
    slack, rounded up to 256. sharegpt/chat-like keeps the 4096 default (the loader
    filters prompts to <=2048). This removes the 4-of-6-campaign truncation/abort
    risk for direct gbench users who don't pass --max-model-len.
    """
    # sharegpt/chat-like: loader filters prompts to <=2048, so 4096 always fits.
    if campaign == "chat-like" or not input_lengths or not output_lengths:
        return 4096
    in_len = int(input_lengths[0])
    out_len = int(output_lengths[0])
    need = int(1.5 * (in_len + out_len)) + 256
    return max(4096, ((need + 255) // 256) * 256)


def get_server_timeout(total_params_b: float, is_moe: bool) -> int:
    """Derive vLLM server startup timeout from model parameters.

    Generous timeouts for parallel execution - under parallel GPU load,
    weight loading competes for PCIe/NVLink bandwidth and CUDA graph
    compilation is slower. Uses 3x safety margin over solo estimates.

    Args:
        total_params_b: Total model parameters in billions.
        is_moe: Whether the model uses Mixture of Experts.

    Returns:
        Timeout in seconds. Floor 600s, cap 3600s.
    """
    solo = 300 + total_params_b * 10
    parallel = solo * (1.3 if is_moe else 1.0) * 3
    return int(max(600, min(3600, parallel)))


def get_gpu_memory_utilization(total_params_b: float) -> float:
    """GPU memory utilization - uniform 0.90 for fair comparison.

    Leaves ~8GB headroom per 80GB GPU for CUDA graphs + sampler.
    """
    return 0.90


def get_max_model_len(override: Optional[int] = None) -> int:
    """Context length to serve at.

    Returns ``override`` when a positive value is given (the
    ``--max-model-len`` / ``config.max_model_len`` escape hatch for
    long-context campaigns), otherwise the uniform 4096 default that
    matches the MLPerf v4.0 standard context for chat workloads. The
    default prevents OOM from models with large native contexts (10M+,
    128K+) and avoids penalizing small models with excessive KV cache
    reservation; overriding it trades that uniformity for reach.
    """
    if override and override > 0:
        return override
    return 4096


def estimate_required_vram_gb(
    total_params_b: float, bytes_per_param: float = 2.0
) -> float:
    """Estimate the VRAM needed to serve a model, in GB.

    Weights dominate: ``total_params_b * bytes_per_param``. The default of
    2.0 bytes/param is the bf16 weight footprint gbench benchmarks. For
    MoE models this uses TOTAL params, not active - every expert stays
    resident in VRAM regardless of per-token routing, so a "26B-A4B" costs
    ~52GB, not ~8GB.

    Adds a fixed ~10GB for CUDA graphs, activations, and a minimal KV cache
    at get_max_model_len() (4096). This is a floor for a "does it physically
    fit" check, not a capacity-planning estimate.
    """
    weights_gb = total_params_b * bytes_per_param
    runtime_overhead_gb = 10.0
    return weights_gb + runtime_overhead_gb


# The GemmaClaw commit the quality pillar scores against unless
# ``--gemmaclaw-commit`` overrides it.
#
# A sha and not ``main``, because this is the value almost every run uses
# and it is hashed into the quality ``scaffold_id``. A default that tracks
# a branch makes two unflagged runs a week apart two different experiments,
# and the id correctly moves to say so, which means the series breaks for a
# reason nobody chose. Pinning it here is what makes the default
# reproducible: the same bare ``gbench --quality-only`` gives the same
# scaffold_id in six months.
#
# Promoted by hand, which is the point rather than a shortcoming. Pick the
# newest ``gemmaclaw-v*`` tag, put its sha here, update the tag name and
# date below, and say so in the PR. Changing the scorer is then a
# deliberate reviewable commit instead of something that happens to you.
# ``scripts/`` has no automation for this on purpose.
#
# Currently gemmaclaw-v2026.8.3, tagged 2026-06-23.
DEFAULT_GEMMACLAW_COMMIT = "d8bc6989133e0e93535b5ad84e6acda019642839"


# -- Model-specific serve settings for the vLLM servers gbench launches --
# gbench is model-agnostic: nothing here names a model. The settings a model's serving recipe
# needs (its tool-call parser, its reasoning parser, any other `vllm serve` argument) come from
# the operator through --tool-call-parser, --reasoning-parser and --vllm-serve-args, are stored
# on BenchmarkConfig, recorded in the scaffold and metadata.json, and appended to the servers
# gbench launches by serve_args_for().

#: `vllm serve` arguments gbench sets itself, and what sets each one instead. --vllm-serve-args
#: refuses them (parse_vllm_serve_args), so a result never runs under a value gbench did not
#: record. The parser flags have their own gbench flags; the rest are gbench's uniform settings,
#: the port gbench picked, the identity checks or credentials that must not be on a command line.
GBENCH_OWNED_SERVE_ARGS: Dict[str, str] = {
    "--model": "--models",
    "--port": "gbench (GBENCH_SERVER_PORT, else the first free port from 8000)",
    "--host": "GBENCH_SERVER_HOST (a loopback address or all interfaces; unset: performance "
              "servers, and the quality / Golden Set server of a "
              "run that selects no eval suite, bind 127.0.0.1; with eval suites selected the "
              "server binds all interfaces, where task containers on docker's default bridge "
              "reach it at the bridge gateway)",
    "--max-model-len": "--max-model-len",
    "--tensor-parallel-size": "--num-gpus / --tensor-parallel",
    "-tp": "--num-gpus / --tensor-parallel",
    "--data-parallel-size": "--num-gpus / --tensor-parallel",
    "-dp": "--num-gpus / --tensor-parallel",
    "--gpu-memory-utilization": "--gpu-memory-utilization",
    "--max-logprobs": "gbench (the eval server's logprobs cap)",
    "--enable-auto-tool-choice": "--tool-call-parser NAME",
    "--tool-call-parser": "--tool-call-parser NAME",
    "--reasoning-parser": "--reasoning-parser NAME",
    "--max-num-seqs": "gbench (uniform scheduler setting)",
    "--max-num-batched-tokens": "gbench (uniform scheduler setting)",
    "--enable-chunked-prefill": "gbench (uniform scheduler setting)",
    "--no-enable-chunked-prefill": "gbench (uniform scheduler setting)",
    "--enable-prefix-caching": "gbench (off on performance servers, vLLM's default on eval servers)",
    "--no-enable-prefix-caching": "gbench (off on performance servers, vLLM's default on eval "
                                  "servers)",
    "--served-model-name": "gbench (results name the --models id, and the served id is checked "
                           "against it)",
    "--api-key": "VLLM_API_KEY in the environment (a key on a command line reaches logs)",
    "--hf-token": "HF_TOKEN in the environment (a token on a command line reaches logs)",
    "--config": "(a config file can set any of the arguments gbench owns)",
}

#: The environment variable read when --vllm-serve-args is not given (the flag wins).
VLLM_SERVE_ARGS_ENV = "GBENCH_VLLM_SERVE_ARGS"


def _serve_flag_name(token: str) -> str:
    """The flag name of a `vllm serve` token: the part before '=', with a long option's
    underscores read as dashes (vLLM's parser accepts `--max_model_len` for `--max-model-len`)."""
    name = token.split("=", 1)[0]
    if name.startswith("--"):
        return "--" + name[2:].replace("_", "-")
    return name


def parse_vllm_serve_args(raw: Optional[str]) -> List[str]:
    """--vllm-serve-args (or GBENCH_VLLM_SERVE_ARGS) as a token list, shlex-split.

    Raises ValueError, naming the gbench flag to use instead, when a token sets an argument gbench
    owns (GBENCH_OWNED_SERVE_ARGS), including `--flag=value`, underscore spellings and a long
    option abbreviated to a prefix of an owned one (argparse accepts unambiguous prefixes), and
    when the first token is not an option (the model is gbench's positional argument)."""
    if raw is None or not str(raw).strip():
        return []
    try:
        tokens = shlex.split(str(raw))
    except ValueError as e:
        raise ValueError(f"--vllm-serve-args could not be parsed ({e})") from e
    if tokens and not tokens[0].startswith("-"):
        raise ValueError(f"--vllm-serve-args must start with an option, got {tokens[0]!r}: the "
                         "model is gbench's positional argument (--models)")
    for tok in tokens:
        if not tok.startswith("-") or tok == "-" or tok == "--":
            continue
        name = _serve_flag_name(tok)
        owner = GBENCH_OWNED_SERVE_ARGS.get(name)
        if owner is None and name.startswith("--") and len(name) > 3:
            prefixed = [k for k in GBENCH_OWNED_SERVE_ARGS if k.startswith(name)]
            if prefixed:
                owner = GBENCH_OWNED_SERVE_ARGS[prefixed[0]]
                name = f"{name} (an abbreviation of {prefixed[0]})"
        if owner is not None:
            raise ValueError(f"--vllm-serve-args may not set {name}: gbench sets it; use {owner}")
    return tokens


#: The --reasoning-parser value that states the served model has no reasoning parser: accepted for
#: a local --thinking run, recorded, and never passed to vLLM.
NO_REASONING_PARSER = "none"


def serve_args_for(config: "BenchmarkConfig", context: str) -> List[str]:
    """The operator's serve settings to append LAST to a `vllm serve` command gbench builds.

    `context` is ServingBenchmarkRunner._start_server's: "native" (the eval, quality and Golden
    Set server) gets `--enable-auto-tool-choice --tool-call-parser <NAME>` when
    --tool-call-parser is set, `--reasoning-parser <NAME>` when --reasoning-parser names a parser
    (`none`, the operator's statement that the model has none, passes nothing; the scaffold still
    records it), then --vllm-serve-args; "perf" (serving, throughput and stress servers) gets
    --vllm-serve-args only, so the performance pillars keep measuring the raw generation."""
    out: List[str] = []
    if context == "native":
        parser = getattr(config, "tool_call_parser", None)
        if parser:
            out += ["--enable-auto-tool-choice", "--tool-call-parser", str(parser)]
        reasoning = getattr(config, "reasoning_parser", None)
        if reasoning and str(reasoning).strip().lower() != NO_REASONING_PARSER:
            out += ["--reasoning-parser", str(reasoning)]
    out += list(getattr(config, "vllm_serve_args", None) or [])
    return out


def serve_settings_record(config: "BenchmarkConfig") -> Dict[str, object]:
    """The operator's serve settings as the scaffold, metadata.json and each eval's run_shape
    record them. With --remote-endpoint gbench started no server, so it knows none of that
    server's settings: the record says so, and its parsers are only what the operator declared
    with --tool-call-parser / --reasoning-parser (None: not declared, not "none")."""
    record: Dict[str, object] = {
        "tool_call_parser": getattr(config, "tool_call_parser", None),
        "reasoning_parser": getattr(config, "reasoning_parser", None),
        "vllm_serve_args": list(getattr(config, "vllm_serve_args", None) or []),
    }
    if getattr(config, "remote_endpoint", None):
        # --tool-call-parser, --reasoning-parser and --vllm-serve-args are refused with it
        record["server_started_by_gbench"] = False
        record["note"] = ("not started by gbench: the remote endpoint's parsers and serve "
                          "arguments are configured on that server and not known here")
    return record


def serve_arg_value(tokens: Any, flag: str) -> Optional[str]:
    """The value `tokens` (--vllm-serve-args) give the long option `flag`, as `vllm serve` reads
    it: `--flag value` or `--flag=value`, underscore spellings included (_serve_flag_name); the
    last one wins; None when it is not set."""
    args = [str(t) for t in (tokens or [])]
    value = None
    for i, tok in enumerate(args):
        if not tok.startswith("--") or _serve_flag_name(tok) != flag:
            continue
        if "=" in tok:
            value = tok.split("=", 1)[1]
        elif i + 1 < len(args) and not args[i + 1].startswith("--"):
            value = args[i + 1]
    return value


def serve_arg_flag(tokens: Any, flag: str) -> bool:
    """Whether `tokens` (--vllm-serve-args) set the boolean long option `flag` (`--flag`, its
    underscore spelling, or `--flag=true/1/yes`)."""
    on = False
    for tok in (str(t) for t in (tokens or [])):
        if not tok.startswith("--") or _serve_flag_name(tok) != flag:
            continue
        on = ("=" not in tok) or tok.split("=", 1)[1].strip().lower() in ("true", "1", "yes")
    return on


def tokenizer_trust_remote_code(config: Any) -> Tuple[bool, str]:
    """(trust_remote_code for gbench's own loads of the model's tokenizer, where it comes from):
    --tokenizer-trust-remote-code, else --trust-remote-code in --vllm-serve-args (the trust the
    operator gave the model's server), else False."""
    if getattr(config, "tokenizer_trust_remote_code", False):
        return True, "--tokenizer-trust-remote-code"
    if serve_arg_flag(getattr(config, "vllm_serve_args", None), "--trust-remote-code"):
        return True, "--vllm-serve-args --trust-remote-code"
    return False, "not set"


# -- The port of a server gbench launches --
#: Where the automatic port search starts when GBENCH_SERVER_PORT is not set.
DEFAULT_SERVER_PORT = 8000
#: How many ports after DEFAULT_SERVER_PORT the automatic search tries.
SERVER_PORT_SEARCH_SPAN = 200


def port_is_free(port: int) -> bool:
    """True when no process listens on TCP `port`: nothing on 127.0.0.1 accepts a connection,
    and binding all interfaces (as vllm serve binds) succeeds. The bind sets SO_REUSEADDR like
    the server's own socket (so connections left in TIME_WAIT by a server that just stopped do
    not count) and not SO_REUSEPORT (so a listener that set it still counts as busy)."""
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=0.5):
            return False
    except OSError:
        pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("", int(port)))
        except OSError:
            return False
    return True


def resolve_server_port(environ: Optional[Dict[str, str]] = None) -> Tuple[Optional[int], str]:
    """(port, source) for the next server gbench launches, or (None, error).

    GBENCH_SERVER_PORT set: that port when it is free, else the error "port N is in use by
    another process; set GBENCH_SERVER_PORT to a free port" (never another port: the operator
    chose it). Unset: the first free port from DEFAULT_SERVER_PORT (source "auto"). The caller
    must still check, once the server answers, that the process on the port is its own child."""
    env = os.environ if environ is None else environ
    raw = (env.get("GBENCH_SERVER_PORT") or "").strip()
    if raw:
        try:
            port = int(raw)
        except ValueError:
            return None, f"GBENCH_SERVER_PORT={raw!r} is not a port number"
        if not 1 <= port <= 65535:
            return None, f"GBENCH_SERVER_PORT={port} is not a port number (1-65535)"
        if not port_is_free(port):
            return None, (f"port {port} is in use by another process; set GBENCH_SERVER_PORT to a "
                          "free port")
        return port, "GBENCH_SERVER_PORT"
    for port in range(DEFAULT_SERVER_PORT, DEFAULT_SERVER_PORT + SERVER_PORT_SEARCH_SPAN):
        if port_is_free(port):
            return port, "auto"
    return None, (f"no free port in {DEFAULT_SERVER_PORT}-"
                  f"{DEFAULT_SERVER_PORT + SERVER_PORT_SEARCH_SPAN - 1}; set GBENCH_SERVER_PORT "
                  "to a free port")


# -- Endpoints and served model ids --
def normalize_endpoint(url: Optional[str]) -> Optional[str]:
    """An OpenAI-compatible base URL in the one form every pillar uses: no trailing '/', ending
    in '/v1' ('http://h:1' and 'http://h:1/' become 'http://h:1/v1'; 'http://h:1/v1/' becomes
    'http://h:1/v1'). None and '' stay as they are."""
    if not url:
        return url
    base = str(url).strip().rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


#: A Hugging Face repo id: 'org/name', neither a filesystem path nor a bare name.
_HF_ID_RE = re.compile(r"[A-Za-z0-9][\w.-]*/[\w.-]+")


def match_served_model(served_ids: Sequence[str], candidates: Sequence[Optional[str]],
                       override: Optional[str] = None) -> Tuple[Optional[str], str]:
    """(served id, how it was chosen) from an endpoint's /v1/models ids, or (None, error).

    Order: `override` (--served-model-id; it must be served); else the first candidate
    (in order: the model's HF id, its path, its name, its short name) that equals a served id,
    exactly or case-insensitively, or whose last path component does (never another
    organisation's Hugging Face id of the same name); else the single served
    id when there is exactly one AND no candidate was given; else an error listing the served
    ids and naming --served-model-id. Pure: the caller lists /v1/models."""
    served = [str(s) for s in served_ids if s]
    if override:
        if override in served:
            return override, "--served-model-id"
        return None, (f"--served-model-id {override!r} is not served by the endpoint (it serves "
                      f"{served or 'no model'})")
    cands = [str(c) for c in candidates if c]

    def _base(x: str) -> str:
        return x.rstrip("/").rsplit("/", 1)[-1].lower()

    def _org(x: str) -> Optional[str]:
        """The organisation of a Hugging Face id 'org/name' (None for a local path or a bare
        name)."""
        return x.split("/", 1)[0].lower() if _HF_ID_RE.fullmatch(x) else None

    for cand in cands:
        for sid in served:
            if sid == cand:
                return sid, f"matches {cand!r}"
        for sid in served:
            if sid.lower() == cand.lower():
                return sid, f"matches {cand!r} (case-insensitive)"
    # By last path component: a local path (/models/<name>) or a bare name matches, but not
    # another organisation's checkpoint of the same name ('otherorg/<name>' for a model whose
    # Hugging Face id is 'org/<name>'), whichever candidate brought the name.
    orgs = {o for o in (_org(c) for c in cands) if o}
    for cand in cands:
        hits = [sid for sid in served if _base(sid) == _base(cand)
                and not (orgs and _org(sid) and _org(sid) not in orgs)]
        if len(hits) == 1:
            return hits[0], f"matches {cand!r} by its last path component"
    if not cands and len(served) == 1:
        return served[0], "the only model the endpoint serves"
    if not served:
        return None, "the endpoint lists no served model"
    want = ", ".join(repr(c) for c in cands) or "no --models"
    return None, (f"the endpoint serves {served}, none of which is {want}; pass --models with "
                  "the served id, or --served-model-id to name it")


# -- GPU selection --
class GpuInfo(NamedTuple):
    """One GPU as nvidia-smi reports it."""
    index: int
    uuid: str
    total_gb: float
    free_gb: float
    #: PIDs of the compute processes running on it (any process, gbench's own included).
    pids: Tuple[str, ...] = ()

    @property
    def idle(self) -> bool:
        return not self.pids


#: A GPU counts as idle for automatic selection when it runs no compute process and has at least
#: this share of its memory free.
IDLE_FREE_FRACTION = 0.8


def gpu_inventory() -> Optional[List[GpuInfo]]:
    """Every GPU nvidia-smi reports, with its compute processes, or None when nvidia-smi is
    absent or unreadable. CUDA_VISIBLE_DEVICES (indices or UUIDs) limits the list to the GPUs it
    names, in its order."""
    import subprocess
    try:
        gpus = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if gpus.returncode != 0:
        return None
    busy: Dict[str, List[str]] = {}
    if apps.returncode == 0:
        for line in apps.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0]:
                busy.setdefault(parts[0], []).append(parts[1])
    out: List[GpuInfo] = []
    for line in gpus.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            out.append(GpuInfo(int(parts[0]), parts[1], float(parts[2]) / 1024,
                               float(parts[3]) / 1024, tuple(busy.get(parts[1], ()))))
        except ValueError:
            continue
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip():
        wanted = [w.strip() for w in cvd.split(",") if w.strip()]
        picked = []
        for w in wanted:
            for g in out:
                if w == str(g.index) or w == g.uuid or (w.startswith("GPU-") and g.uuid.startswith(w)):
                    picked.append(g)
                    break
        out = picked
    return out


def select_gpus(count: int, requested: Optional[Sequence[str]] = None,
                inventory: Optional[List[GpuInfo]] = None) -> Tuple[Optional[List[GpuInfo]], str]:
    """(the GPUs a local server runs on, message), or (None, error).

    `requested` (--gpus: indices or UUIDs) must name exactly `count` GPUs, each present and
    running no compute process. Without it, the first `count` idle GPUs (no compute process and
    at least IDLE_FREE_FRACTION of their memory free) of the inventory are chosen, and too few
    idle GPUs is an error naming the busy ones. `inventory` defaults to gpu_inventory(); when
    nvidia-smi cannot be read the result is (None, error) for a request and ([], note) otherwise
    (vLLM then picks the devices, as before)."""
    inv = gpu_inventory() if inventory is None else inventory
    if inv is None:
        if requested:
            return None, "--gpus needs nvidia-smi to read the GPUs, and it could not be run"
        return [], "nvidia-smi unavailable: GPU selection left to vLLM"
    if requested:
        chosen: List[GpuInfo] = []
        for r in requested:
            r = str(r).strip()
            hit = next((g for g in inv if r == str(g.index) or r == g.uuid), None)
            if hit is None:
                return None, (f"--gpus names GPU {r!r}, which is not among the visible GPUs "
                              f"({', '.join(str(g.index) for g in inv) or 'none'})")
            if hit in chosen:
                return None, f"--gpus names GPU {r!r} twice"
            chosen.append(hit)
        if len(chosen) != count:
            return None, (f"--gpus names {len(chosen)} GPU(s) but the run needs {count} "
                          "(--num-gpus)")
        busy = [g for g in chosen if not g.idle]
        if busy:
            return None, ("selected GPU(s) run other processes: " + "; ".join(
                f"GPU {g.index} ({g.uuid}) PIDs {', '.join(g.pids)}" for g in busy)
                + ". Free them or choose other GPUs with --gpus")
        return chosen, "GPUs " + ", ".join(f"{g.index} ({g.uuid})" for g in chosen) + " (--gpus)"
    idle = [g for g in inv if g.idle and (g.total_gb <= 0
                                          or g.free_gb >= IDLE_FREE_FRACTION * g.total_gb)]
    if len(idle) < count:
        busy = [g for g in inv if g not in idle]
        return None, (f"the run needs {count} idle GPU(s) (no compute process, at least "
                      f"{IDLE_FREE_FRACTION:.0%} of memory free) and {len(idle)} of {len(inv)} "
                      "are: " + ("; ".join(
                          f"GPU {g.index} busy (PIDs {', '.join(g.pids) or 'none'}, "
                          f"{g.free_gb:.0f}/{g.total_gb:.0f} GB free)" for g in busy) or "none")
                      + ". Free GPUs, or name idle ones with --gpus")
    chosen = idle[:count]
    return chosen, "GPUs " + ", ".join(f"{g.index} ({g.uuid})" for g in chosen) + " (auto)"


def cuda_visible_devices_env(gpu_uuids: Optional[Sequence[str]]) -> Dict[str, str]:
    """The environment a server gbench launches runs with to use exactly `gpu_uuids`:
    {"CUDA_VISIBLE_DEVICES": "<uuid>,...", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}; {} for none."""
    if not gpu_uuids:
        return {}
    return {"CUDA_VISIBLE_DEVICES": ",".join(gpu_uuids), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def parallel_layout(num_gpus: int, tensor_parallel: Optional[int]) -> Tuple[int, int]:
    """(tensor_parallel_size, data_parallel_size) for --num-gpus N and --tensor-parallel M.

    M defaults to N. M must be one of {1, 2, 4, 8, 16} and at most N, and N must be a multiple of
    M (N / M data-parallel replicas); ValueError otherwise."""
    tp = int(tensor_parallel) if tensor_parallel else int(num_gpus)
    if tp not in (1, 2, 4, 8, 16):
        raise ValueError(f"--tensor-parallel {tp} is not supported: use 1, 2, 4, 8 or 16")
    if tp > num_gpus:
        raise ValueError(f"--tensor-parallel {tp} is larger than --num-gpus {num_gpus}")
    if num_gpus % tp:
        raise ValueError(f"--num-gpus {num_gpus} is not a multiple of --tensor-parallel {tp} "
                         "(the GPUs are split into equal data-parallel replicas)")
    return tp, num_gpus // tp


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution.

    Includes optimized vLLM settings for maximum throughput:
    - gpu_memory_utilization: 0.90 (uniform headroom for CUDA graphs)
    - enable_chunked_prefill: True (~20-30% throughput improvement)
    - max_num_batched_tokens: 16384 (optimal scheduler throughput)
    """

    # Iteration control for statistical reliability
    num_iterations: int = 3  # Run each config N times
    warmup_iterations: int = 1  # Warmup runs (not included in stats)
    min_acceptable_cv_percent: float = 5.0  # Max CV% for variance check

    # --personal (device-peak) mode. serving_num_prompts_override forces a fixed
    # small serving sample count (the real lever - _serving_num_prompts, not
    # num_prompts); personal_mode switches the report to the device-peak block.
    serving_num_prompts_override: Optional[int] = None
    personal_mode: bool = False
    # Images per MM request. None -> the runner default (4). --personal sets 1:
    # lighter (280 vs 1120 img tokens) AND universally supported - a single image
    # needs no server-side --limit-mm-per-prompt>1, which a remote endpoint
    # (e.g. Ollama) may not have configured.
    mm_images_per_request: Optional[int] = None
    # Seconds to wait for the remote multimodal capability probe. None -> 10.
    # --personal defaults it higher (cold CPU vision is slow to first token, so
    # 10s falsely reads as "unsupported"). Set via --mm-probe-timeout.
    mm_probe_timeout: Optional[int] = None
    # Per-request serving timeout in SECONDS (set via --timeout). Applied to both
    # remote paths, but the semantics differ by transport and are documented here:
    #   * text (urllib): a per-read STALL timeout - fires when the server sends no
    #     bytes for this long (e.g. a hung prefill), NOT a cap on total decode.
    #   * MM (aiohttp): a TOTAL wall-clock cap on the whole request.
    # Default 600 is generous for personal single-stream CPU runs; raise it for
    # large models / long outputs on slow devices, lower it to fail hangs faster.
    timeout: int = 600
    # Thinking / reasoning mode (set via --thinking). Default False: gbench sends a
    # no-think system prompt on chat requests so a reasoning model (e.g. gemma-4 on
    # Ollama) measures pure answer generation, not an accidental chain-of-thought
    # workload it inherits from the backend's default system prompt. --thinking
    # skips the override and lets the model reason (reasoning tokens are counted).
    thinking: bool = False

    # Batch sizes to test
    batch_sizes: list[int] = field(
        default_factory=lambda: [1]   # single-stream default; override with --batch-sizes
    )

    # Input lengths for throughput benchmarks (tokens)
    input_lengths: list[int] = field(
        default_factory=lambda: [128, 512, 2048]
    )

    # Output lengths for throughput benchmarks (tokens)
    output_lengths: list[int] = field(
        default_factory=lambda: [128, 512, 1024]
    )

    # Number of images per request (multimodal only)
    images_per_request: list[int] = field(
        default_factory=lambda: [1, 2, 4]
    )

    # Base configuration
    num_prompts: int = 200
    num_prompts_throughput: int = 200
    request_rate: str = "inf"
    dataset: str = "random"
    dataset_path: Optional[str] = None
    dataset_multimodal: str = "random-mm"
    remote_endpoint: Optional[str] = None
    tokenizer: Optional[str] = None

    # Output configuration
    results_dir: Path = field(
        default_factory=lambda: Path("./results")
    )
    enable_logging: bool = True
    log_samples: bool = False

    # Resource limits and vLLM optimizations
    gpu_memory_utilization: float = 0.90
    max_num_seqs: Optional[int] = 256
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = 16384
    # Override for the served context window. None keeps the uniform 4096
    # default (get_max_model_len); raise it to run long-context campaigns
    # (agentic/prefill-heavy/mixed/long-decode) whose input+output exceed
    # 4096. Recorded into the scaffold fingerprint, so a non-default value
    # is a distinct experiment.
    max_model_len: Optional[int] = None
    # The --max-model-len the USER passed (None if not passed; -1 = let vLLM fit the window
    # to GPU memory). Kept apart from max_model_len, which a --campaign also sets to its
    # perf window: eval/quality servers serve the model's native window unless the user
    # asked otherwise, and a campaign's perf sizing must not cap them.
    max_model_len_explicit: Optional[int] = None
    # True when the user pinned a workload geometry (a --campaign or explicit
    # --input/--output-lengths). The stress test reads this to decide whether
    # to probe a real campaign shape/dataset or fall back to its neutral
    # 128/128 laptop default.
    workload_shape_explicit: bool = False
    # Active campaign name (set by apply_campaign_to_config). Drives per-campaign
    # SLO resolution and auto max_model_len; recorded in the run fingerprint.
    campaign: Optional[str] = None

    # Stress-test knobs. All four resolve flag > GBENCH_STRESS_* env > runner
    # default. stress_reps: full sweeps per point (None -> runner default 3; the
    # knee is reported as mean+CI over reps since a single sweep varies run-to-run).
    # stress_client_procs: OS processes for the multi-process load generator
    # (None -> runner default 8, capped to cpu_count-1) - keeps the client from
    # becoming the knee.
    stress_reps: Optional[int] = None
    stress_client_procs: Optional[int] = None
    # stress_max_qps: safety-cap on the open-loop arrival sweep (None -> runner
    # default 512). Raise it when a fast model saturates the 512 cap (the knee is
    # then censored at 512, not measured). stress_max_prompts: cap on requests per
    # rate point (None -> runner default 1200); raise it so high-QPS points still
    # span a real steady-state window instead of a sub-second burst. Both also read
    # GBENCH_STRESS_MAX_QPS / GBENCH_STRESS_MAX_PROMPTS (flag wins over env).
    stress_max_qps: Optional[float] = None
    stress_max_prompts: Optional[int] = None
    # stress_min_samples: min steady-state completions to trust a rate point (None ->
    # runner default 15). Lower it for a slow box that meets the SLO but cannot
    # complete 15 per point (else every point is TOO-FEW and the knee is 0); a smaller
    # floor gives a faster, wider-CI, lower-confidence knee.
    stress_min_samples: Optional[int] = None
    # Certification of the reported stress limit (StressTestRunner._certify_knee):
    # "auto" (default: certify within the budget), "always" (ignore the budget), "off";
    # budget in minutes (default 120). Also GBENCH_STRESS_CERTIFY /
    # GBENCH_STRESS_CERTIFY_BUDGET_MIN (flag wins).
    stress_certify: Optional[str] = None
    stress_certify_budget: Optional[float] = None

    # The per-campaign stress SLOs this run resolved (resolve_campaign_slo with --stress-threshold
    # / --itl-slo-ms), set by the CLI before any runner is built, so the scaffold records the SLO
    # that decides the max-sustainable-QPS verdict.
    stress_ttft_slo_ms: Optional[int] = None
    stress_itl_slo_ms: Optional[int] = None

    # Multi-GPU configuration. tensor_parallel_size x data_parallel_size == num_gpus
    # (parallel_layout); a server gbench launches passes --data-parallel-size when it is > 1.
    num_gpus: int = 1
    tensor_parallel_size: Optional[int] = None
    data_parallel_size: int = 1
    # --gpus: the GPUs (indices or UUIDs) the operator asked for; None = choose idle ones.
    gpus: Optional[list[str]] = None
    # The UUIDs of the GPUs select_gpus chose for this run (set by the CLI before any local
    # server starts). A server gbench launches runs with cuda_visible_devices_env(selected_gpus)
    # in its environment; None/[] = the environment's own CUDA_VISIBLE_DEVICES (nvidia-smi could
    # not be read, a remote endpoint, or a dry run).
    selected_gpus: Optional[list[str]] = None

    # Model-specific serve settings of the servers gbench launches (serve_args_for):
    # --tool-call-parser NAME -> `--enable-auto-tool-choice --tool-call-parser NAME` and
    # --reasoning-parser NAME -> `--reasoning-parser NAME` on the eval/quality/Golden Set server;
    # --vllm-serve-args (or GBENCH_VLLM_SERVE_ARGS) -> appended to every server gbench launches.
    # None/[] = not set. Recorded in the scaffold and metadata.json (serve_settings_record).
    tool_call_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    vllm_serve_args: list[str] = field(default_factory=list)

    # --served-model-id (aliases --golden-model-id, --eval-model-id): the id the endpoint serves
    # the model under, when it differs from --models. Every pillar that resolves the served id
    # (match_served_model) takes it first and requires the endpoint to list it.
    served_model_id: Optional[str] = None
    # The base URL (ending in /v1) of the native server the CLI started for this model, set while
    # the quality, Golden Set and eval pillars run and None otherwise. A runner that finds it set
    # uses that server and starts none of its own; with --remote-endpoint it is never set.
    local_endpoint: Optional[str] = None
    # --tokenizer-trust-remote-code: trust_remote_code for gbench's own loads of the model's
    # tokenizer (tokenizer_trust_remote_code), e.g. a remote endpoint's, where --vllm-serve-args
    # is refused.
    tokenizer_trust_remote_code: bool = False
    # Where config.tokenizer came from: "--tokenizer", "--vllm-serve-args --tokenizer" (the
    # server's own tokenizer, when --tokenizer is not given), or None.
    tokenizer_source: Optional[str] = None
    # What the native server records about itself (ServingBenchmarkRunner.server_record: its bind
    # address and why, whether it enforces an endpoint key, GPUs, serve settings, vLLM), set with
    # local_endpoint; the quality, Golden Set and eval results carry it as `server`.
    native_server_record: Optional[Dict[str, Any]] = None

    # Execution control
    dry_run: bool = False
    skip_existing: bool = False

    # Quality benchmark (gemmaclaw) configuration
    gemmaclaw_commit: str = DEFAULT_GEMMACLAW_COMMIT
    gemmaclaw_path: Optional[str] = None
    remote_endpoint: Optional[str] = None
    selected_scenarios: Optional[list[str]] = None

    # Golden Set benchmark configuration
    golden: bool = False
    golden_only: bool = False
    selected_golden_tasks: Optional[list[str]] = None
    # Model name to put in the golden request payload. Leave unset to let
    # the runner resolve it from the endpoint's /models listing.
    golden_model_id: Optional[str] = None

    # Evaluation suites configuration
    evals: Optional[list[str]] = None
    eval_thinking: bool = False
    eval_max_output_tokens: Optional[int] = None
    # --eval-max-soft-tokens: the model processor's max_soft_tokens setting for vision evals.
    # None = the model's own processor setting (nothing is overridden).
    eval_max_soft_tokens: Optional[int] = None
    eval_n_shot: Optional[int] = None  # None -> suite canonical default (mmlu/mmlu_pro: 5-shot); explicit 0 = 0-shot
    #: Per-suite wall-clock budget (s). None -> per-suite defaults in evals.py.
    suite_timeout: Optional[int] = None
    eval_categories: Optional[str] = None
    eval_limit: Optional[int] = None
    # The run's shard "I/N" (--shard, else GBENCH_SHARD from the environment; validated by the
    # CLI), the value the suites read (base.current_shard). None = the full set.
    shard: Optional[str] = None
    sandboxes: Optional[int] = None
    # The eval request concurrency per suite (--eval-concurrency, else --batch-sizes[0]; set by
    # the CLI). None = the runner's own derivation from batch_sizes.
    eval_concurrency: Optional[int] = None
    temperature: Optional[float] = None
    attempt_count: int = 1
    eval_plugins_dir: Optional[list[str]] = None
    eval_custom_jsonl: Optional[str] = None

    # Run metadata tracking
    tags: Optional[list[str]] = None

    def __post_init__(self):
        """Validate configuration. LogManager is deferred to initialize()."""
        self.log_manager = None

    def initialize(self):
        """Create LogManager after all config fields (incl. --results-dir) are set.

        Must be called after CLI argument overrides are applied to this config.
        """
        from gbench.utils import LogManager

        # A dry run writes nothing: the run directory is resolved but not created. An existing
        # run directory is continued only by a run of the same host, shard and gbench commit.
        self.log_manager = LogManager(self.results_dir,
                                      create_dirs=not getattr(self, "dry_run", False),
                                      identity=self.run_identity)
        self.results_dir = self.log_manager.results_dir

    def run_identity(self) -> Dict[str, Any]:
        """{"host", "shard", "gbench"} of this run, as metadata.json records them."""
        try:
            from .scaffold import gbench_commit
            commit: Any = gbench_commit()
        except Exception:                                                 # noqa: BLE001
            commit = None
        return {"host": socket.gethostname(), "shard": getattr(self, "shard", None),
                "gbench": commit}

    def _serving_num_prompts(self, batch: int) -> tuple[int, bool]:
        """Sample count for one serving config, sized to a wall-clock target.

        Problem-B fix: a flat count is the wrong knob - request cost spans 3
        orders of magnitude across campaigns, and single-stream (batch 1) is
        slow regardless of count. So target a fixed wall-clock per config:
        wall_clock ~= n * out_len / (batch * tok/s), solved for n. Samples thus
        scale UP with concurrency and DOWN with output length, keeping each
        campaign's real shape while bounding runtime. Fast/high-concurrency
        configs get many samples (tight P99); slow/low-concurrency ones get few
        (bounded) and are flagged low_confidence so a thinly-sampled P99 is never
        passed off as solid. (Very-long-output campaigns still hit the MIN floor
        and stay expensive at low batch - those are better skipped at the sweep
        level; stress TPOT + offline throughput already cover them.)
        """
        RELIABLE_P99 = 150               # below this, P99 is low-confidence
        # --personal: a fixed small count (device tok/s converges fast; the wall-
        # clock auto-sizer's TPS_PER_STREAM=30 is GPU-calibrated and wrong on CPU).
        if self.serving_num_prompts_override is not None:
            n = max(1, int(self.serving_num_prompts_override))
            return n, (n < RELIABLE_P99)
        TARGET_CONFIG_S = 300            # ~5 min wall-clock target per config
        TPS_PER_STREAM = 30              # rough single-stream decode tok/s
        MIN_PROMPTS, MAX_PROMPTS = 30, 500
        out_len = int(self.output_lengths[0]) if self.output_lengths else 512
        n = int(TARGET_CONFIG_S * max(int(batch), 1) * TPS_PER_STREAM / max(out_len, 64))
        n = max(MIN_PROMPTS, min(n, MAX_PROMPTS))
        return n, (n < RELIABLE_P99)

    def get_serving_configs(self) -> list[dict]:
        """Generate all serving benchmark configurations."""
        configs = []
        for batch in self.batch_sizes:
            n, low_conf = self._serving_num_prompts(batch)
            configs.append({
                "batch_size": batch,
                "num_prompts": n,
                "request_rate": self.request_rate,
                "dataset": self.dataset,
                "low_confidence": low_conf,
            })
        return configs

    def get_throughput_configs(self) -> list[dict]:
        """Generate all throughput benchmark configurations."""
        configs = []
        for input_len in self.input_lengths:
            for output_len in self.output_lengths:
                for batch in self.batch_sizes:
                    configs.append({
                        "input_length": input_len,
                        "output_length": output_len,
                        "batch_size": batch,
                        "num_prompts": self.num_prompts_throughput,
                    })
        return configs

    def get_multimodal_configs(self) -> list[dict]:
        """Generate multimodal-specific configurations."""
        configs = []
        for num_images in self.images_per_request:
            for batch in self.batch_sizes:
                configs.append({
                    "num_images": num_images,
                    "batch_size": batch,
                    "num_prompts": self.num_prompts,
                    "dataset": self.dataset_multimodal,
                })
        return configs


@dataclass
class PerformanceTargets:
    """Target performance metrics for validation."""

    ttft_p50_target_ms: float = 200.0
    ttft_p95_target_ms: float = 500.0
    itl_p50_target_ms: float = 50.0
    itl_p95_target_ms: float = 100.0
    tpot_target_ms: float = 100.0
    throughput_target_tps: float = 1000.0
    memory_target_gb: float = 45.0
    gguf_quality_delta_max: float = 2.0
    repeatability_variance_max: float = 5.0

    def validate_serving_results(
        self, results: dict
    ) -> dict[str, bool]:
        """Validate serving benchmark results against targets."""
        checks = {}
        checks["ttft_p50"] = (
            results.get("ttft_p50", float("inf"))
            <= self.ttft_p50_target_ms
        )
        checks["ttft_p95"] = (
            results.get("ttft_p95", float("inf"))
            <= self.ttft_p95_target_ms
        )
        checks["itl_p50"] = (
            results.get("itl_p50", float("inf"))
            <= self.itl_p50_target_ms
        )
        checks["itl_p95"] = (
            results.get("itl_p95", float("inf"))
            <= self.itl_p95_target_ms
        )
        return checks


# Default configurations
DEFAULT_CONFIG = BenchmarkConfig()
DEFAULT_TARGETS = PerformanceTargets()

# Quick test configuration (smoke test)
QUICK_CONFIG = BenchmarkConfig(
    num_iterations=1,
    warmup_iterations=0,
    batch_sizes=[1],
    input_lengths=[128],
    output_lengths=[512],
    num_prompts=10,
    num_prompts_throughput=10,
    gpu_memory_utilization=0.90,
    enable_chunked_prefill=True,
    max_num_batched_tokens=16384,
)

# Default configuration (production benchmark)
DEFAULT_CONFIG = BenchmarkConfig(
    num_iterations=3,
    warmup_iterations=1,
    batch_sizes=[1],   # single-stream default; override with --batch-sizes
    input_lengths=[128],
    output_lengths=[512],
    num_prompts=1000,
    num_prompts_throughput=1000,
    gpu_memory_utilization=0.90,
    enable_chunked_prefill=True,
    max_num_batched_tokens=16384,
)


def get_available_gpus() -> int:
    """Get the number of available NVIDIA GPUs.

    Respects CUDA_VISIBLE_DEVICES if set.
    """
    import subprocess

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        return len([x.strip() for x in cuda_visible.split(",") if x.strip()])

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return len([l for l in result.stdout.splitlines() if "GPU" in l])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def _gpu_total_vram_gb() -> float:
    """Total VRAM of the smallest visible GPU, in GB (0.0 if unknown).

    Queries nvidia-smi ``memory.total``, respecting CUDA_VISIBLE_DEVICES so
    it reflects the GPUs a run will actually use. Returns 0.0 when
    nvidia-smi is absent or unreadable, so callers can skip the VRAM check
    (fail open) rather than block a run on a missing tool.
    """
    import subprocess

    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits",
    ]
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu_ids:
        cmd.append("--id=" + gpu_ids)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            totals = [
                float(x) / 1024
                for x in result.stdout.split()
                if x.strip()
            ]
            if totals:
                return min(totals)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0.0


def _gpu_min_free_vram_gb() -> float:
    """Free VRAM of the smallest visible GPU, in GB (0.0 if unknown).

    Mirrors ``_gpu_total_vram_gb`` but queries ``memory.free``; respects
    CUDA_VISIBLE_DEVICES. 0.0 (fail-open) when nvidia-smi is absent/unreadable.
    """
    import subprocess

    cmd = ["nvidia-smi", "--query-gpu=memory.free",
           "--format=csv,noheader,nounits"]
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu_ids:
        cmd.append("--id=" + gpu_ids)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            frees = [float(x) / 1024 for x in result.stdout.split() if x.strip()]
            if frees:
                return min(frees)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0.0


# Per-rank overhead (CUDA graph capture, activations, NCCL buffers). Incurred on
# EVERY GPU, so it is NOT shard-divided by TP.
_PER_RANK_OVERHEAD_GB = 3.0


def weights_fit_advisory(
    total_params_b: float,
    tensor_parallel: int,
    gpu_memory_utilization: float,
    min_free_override: Optional[float] = None,
    gpus: Optional[Sequence["GpuInfo"]] = None,
) -> tuple[bool, str]:
    """Best-effort, NON-BLOCKING check that a model's WEIGHTS fit the GPUs the run uses.

    Gates ``weights_per_gpu = total_params_b*2/TP + per-rank overhead`` against
    the usable budget ``min(free, util*total)`` per GPU. Advisory only - vLLM's
    load-time profiler is the hard authority for the full weights+KV fit; this
    gives an early heads-up so a doomed run isn't discovered only after a long
    load. KV is intentionally NOT modeled (per-layer-type: sliding-window / MLA
    make a flat estimate ~5-40x wrong - vLLM sizes it correctly). Returns
    ``(fits, message)``; ``fits=True`` (skip) when the model size or VRAM can't
    be read. ``min_free_override`` (from --min-free-gb) REPLACES the computed
    requirement. ``gpus``: the GPUs the run selected (select_gpus); their smallest total and
    free memory are used. Without it, every visible GPU's (a busy GPU the run will not use can
    then lower the estimate).
    """
    tp = max(1, int(tensor_parallel or 1))
    if not (isinstance(total_params_b, (int, float)) and total_params_b > 0):
        return True, "weights-fit check skipped (unknown model size - deferring to vLLM)."
    if gpus:
        total = min(float(g.total_gb) for g in gpus)
        free = min(float(g.free_gb) for g in gpus)
    else:
        total = _gpu_total_vram_gb()
        free = _gpu_min_free_vram_gb()
    caps = [c for c in (free, total * gpu_memory_utilization) if c > 0]
    if not caps:
        return True, "weights-fit check skipped (could not read VRAM - deferring to vLLM)."
    usable = min(caps)   # bounded by BOTH free memory now AND util*total
    weights_per_gpu = total_params_b * 2.0 / tp + _PER_RANK_OVERHEAD_GB
    if min_free_override is not None:
        # Explicit user floor on FREE memory - the SAME meaning as --min-free-gb in
        # check_gpu_ready (not clamped by util*total), so the flag is consistent.
        free_basis = free if free > 0 else usable
        if free_basis < min_free_override:
            return False, (
                f"weights-fit advisory: free VRAM ~{free_basis:.0f}GB/GPU is below "
                f"the --min-free-gb={min_free_override:.0f}GB floor. Model may not "
                f"fit - vLLM's profiler will decide.")
        return True, (f"weights-fit OK: free ~{free_basis:.0f}GB/GPU >= "
                      f"--min-free-gb {min_free_override:.0f}GB.")
    if usable < weights_per_gpu:
        return False, (
            f"weights-fit advisory: ~{weights_per_gpu:.0f}GB/GPU needed "
            f"(={total_params_b:.0f}B x2 / TP{tp} + {_PER_RANK_OVERHEAD_GB:.0f}GB "
            f"per-rank), but only ~{usable:.0f}GB usable/GPU "
            f"(min of free & {gpu_memory_utilization:.0%}xtotal). Model may not fit "
            f"- vLLM's profiler will decide. Lower --gpu-memory-utilization, raise "
            f"--num-gpus, or set --min-free-gb to override.")
    return True, (f"weights-fit OK: ~{weights_per_gpu:.0f}GB/GPU needed, "
                  f"~{usable:.0f}GB usable/GPU.")


def validate_gpu_config(
    num_gpus: int, total_params_b: float,
    gpu_memory_utilization: Optional[float] = None,
) -> tuple[bool, str]:
    """Validate GPU configuration for a model.

    Args:
        num_gpus: Number of GPUs requested.
        total_params_b: Total model parameters in billions.
        gpu_memory_utilization: Fraction of VRAM vLLM will use (the value the
            server is actually launched with). Defaults to the uniform 0.90 when
            not supplied, so this stays consistent with the launched engine and
            the run fingerprint rather than a separate hardcoded value.

    Returns:
        Tuple of (is_valid, message).
    """
    available = get_available_gpus()
    recommended = get_num_gpus(total_params_b)

    if available == 0:
        return False, "No NVIDIA GPUs detected"

    if num_gpus > available:
        return False, f"Requested {num_gpus} GPUs but only {available} available"

    VALID_TP_SIZES = {1, 2, 4, 8, 16}
    if num_gpus not in VALID_TP_SIZES:
        return False, (
            f"GPU count {num_gpus} not supported for tensor parallel. "
            f"Valid sizes: {sorted(VALID_TP_SIZES)}"
        )

    if num_gpus < recommended:
        # `recommended` (see get_num_gpus) is a FAIRNESS tier: it assumes
        # 80GB GPUs and exists so competing models get identical resources
        # for comparable leaderboard numbers. It is NOT a hard physical
        # limit. Only block a below-tier request when the model genuinely
        # will not fit in the requested GPUs' actual VRAM; otherwise allow
        # it with a warning that the run is off the fair tier.
        util = (gpu_memory_utilization if gpu_memory_utilization is not None
                else get_gpu_memory_utilization(total_params_b))
        per_gpu_gb = _gpu_total_vram_gb()
        required_gb = estimate_required_vram_gb(total_params_b)
        usable_gb = per_gpu_gb * num_gpus * util
        if per_gpu_gb <= 0:
            # VRAM couldn't be read (nvidia-smi absent/unreadable). Do NOT claim
            # the model "fits" - we never checked. Defer the physical-fit decision
            # to vLLM's load-time profiler, which is authoritative.
            return True, (
                f"Warning: using {num_gpus} GPU(s) for a {total_params_b:.0f}B "
                f"model is below the {recommended}-GPU fairness tier. Could not "
                f"read VRAM (nvidia-smi unreadable) - deferring the physical-fit "
                f"check to vLLM's load-time profiler."
            )
        if required_gb > usable_gb:
            return False, (
                f"Model (~{required_gb:.0f}GB bf16) does not fit in "
                f"{num_gpus}x{per_gpu_gb:.0f}GB GPU(s) at {util:.0%} "
                f"utilization. Recommended tier: {recommended} GPU(s)."
            )
        return True, (
            f"Warning: using {num_gpus} GPU(s) for a {total_params_b:.0f}B "
            f"model is below the {recommended}-GPU fairness tier - "
            f"leaderboard numbers may not be comparable across models. "
            f"Model fits (~{required_gb:.0f}GB bf16)."
        )

    if num_gpus > recommended * 2:
        return True, (
            f"Warning: Using {num_gpus} GPUs for {total_params_b:.0f}B model "
            f"may not provide linear scaling. Recommended: {recommended}"
        )

    return True, (
        f"Using {num_gpus} GPU(s) for {total_params_b:.0f}B model"
    )


def check_gpu_ready(min_free_memory_gb: Optional[float] = None,
                    gpu_uuids: Optional[Sequence[str]] = None) -> tuple[bool, str]:
    """Check if GPU is ready for benchmarking (no conflicting processes).

    Uses nvidia-smi to detect other processes and (optionally) a free-memory
    floor. Respects CUDA_VISIBLE_DEVICES to only check assigned GPUs.

    ``gpu_uuids`` (the GPUs select_gpus chose for the run) scopes the check: a compute process on
    one of them is a refusal naming it, and processes on other GPUs are not looked at.

    ``min_free_memory_gb`` defaults to None = NO hard free-memory floor (the old
    fixed 70GB floor blocked every <70GB GPU even for tiny models - the actual
    physical-fit decision is model-aware, done by the weights-fit advisory +
    validate_gpu_config + vLLM's own profiler). Pass a number to re-enable a hard
    floor (e.g. via --min-free-gb).
    """
    import subprocess

    gpu_ids = ",".join(gpu_uuids) if gpu_uuids else os.environ.get("CUDA_VISIBLE_DEVICES", "")

    try:
        proc_cmd = [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(
            proc_cmd, capture_output=True, text=True, timeout=10,
        )

        if result.returncode != 0:
            return True, "nvidia-smi unavailable, skipping GPU check"

        if gpu_uuids:
            selected = {u.strip() for u in gpu_uuids}
            clashes = [line.strip() for line in result.stdout.strip().splitlines()
                       if line.strip() and line.rsplit(",", 1)[-1].strip() in selected]
            if clashes:
                return False, (
                    f"a selected GPU is in use by another process.\n"
                    f"Active processes on the selected GPU(s): {'; '.join(clashes[:3])}\n"
                    f"Free them, or choose idle GPUs with --gpus.")
        elif not gpu_ids:
            processes = [
                line.strip()
                for line in result.stdout.strip().splitlines()
                if line.strip()
            ]
            if processes:
                process_info = "; ".join(processes[:3])
                return False, (
                    f"GPU is currently in use by other processes.\n"
                    f"Active GPU processes: {process_info}\n"
                    f"Please terminate these processes before running benchmarks.\n"
                    f"Tip: Use 'nvidia-smi' to view processes and 'kill <PID>' to stop them."
                )

        mem_cmd = [
            "nvidia-smi",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
        if gpu_ids:
            mem_cmd.extend(["--id=" + gpu_ids])

        mem_result = subprocess.run(
            mem_cmd, capture_output=True, text=True, timeout=10,
        )

        if mem_result.returncode == 0:
            lines = mem_result.stdout.strip().splitlines()
            if lines:
                min_free_gb = float("inf")
                total_gb_first = 0
                for line in lines:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        free_mb = float(parts[0].strip())
                        total_mb = float(parts[1].strip())
                        free_gb = free_mb / 1024
                        if total_gb_first == 0:
                            total_gb_first = total_mb / 1024
                        min_free_gb = min(min_free_gb, free_gb)

                if min_free_memory_gb is not None and min_free_gb < min_free_memory_gb:
                    return False, (
                        f"Insufficient GPU memory.\n"
                        f"Free: {min_free_gb:.1f}GB / {total_gb_first:.1f}GB "
                        f"total (min across {len(lines)} GPU(s))\n"
                        f"Required: {min_free_memory_gb:.1f}GB minimum (--min-free-gb)\n"
                        f"Try waiting 30 seconds or restart the terminal."
                    )

                # Soft busy-card signal when there is no hard floor. Under
                # CUDA_VISIBLE_DEVICES the process-conflict check above is skipped,
                # so this is the only pre-launch hint that an assigned card is
                # already occupied. WARN (never block - a genuinely small GPU can
                # be low in absolute GB yet fine); vLLM's profiler is authoritative.
                if (min_free_memory_gb is None and total_gb_first > 0
                        and min_free_gb < 0.15 * total_gb_first):
                    return True, (
                        f"WARNING: assigned GPU appears busy: only {min_free_gb:.1f}GB / "
                        f"{total_gb_first:.1f}GB free - proceeding (vLLM will confirm; "
                        f"if this is unexpected, free the GPU or wait)."
                    )

                return True, (
                    f"GPU ready: {min_free_gb:.1f}GB / {total_gb_first:.1f}GB "
                    f"free ({len(lines)} GPU(s))"
                )

        return True, "GPU check passed"

    except FileNotFoundError:
        return True, "nvidia-smi not found, skipping GPU check"
    except subprocess.TimeoutExpired:
        return True, "nvidia-smi timed out, skipping GPU check"
    except Exception as e:
        return True, f"GPU check failed: {e}, continuing anyway"
