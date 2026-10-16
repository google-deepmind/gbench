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

"""`gbench --preflight` and `gbench --smoke` (docs/smoke.md).

--preflight checks, before anything is generated, what every selected suite needs: each suite's
own prerequisite check (the check_*_prerequisites its runner calls first, run here exactly as the
runner calls it; PREFLIGHT_CHECKS), and for a suite without one the local images it names, plus
the Gemini key for the suites that grade, search or simulate with gbench's Gemini cascade
(GEMINI_SUITES, presence only). Global checks: free disk under --results-dir
(GBENCH_PREFLIGHT_MIN_DISK_GB, default 100), the Docker daemon, the endpoint's GET /v1/models from
this host and from a container on Docker's default bridge network (an existing local image, never
pulled). The checks run in parallel threads, except that the suites of one pool family
(pool.FAMILIES: a shared harness root, image or ports) are checked one after another (2026-10-11
review: bfcl's and bfcl_v4_agentic's checks wrote and unlinked the same
<project root>/.gbench_write_probe at once, and one failed "not writable" in 2 of 3 real runs);
they call no model under test and build no image; a check still running after
GBENCH_PREFLIGHT_TIMEOUT_S (default 300) is a FAIL. The table names each
failure's reason; the exit status is 2 on any FAIL. Nothing is skipped. On 2026-10-09 five of the
seven hard failures of the 84-suite run (missing keys, credentials, images) were of this kind,
and mcp_bench / mcp_atlas reported theirs only at the end of their lane.

--smoke runs the preflight, then every suite whose checks passed, with --eval-limit 1, as a
--parallel-suites pool (default SMOKE_PARALLEL), with GBENCH_SMOKE=1 in each suite's environment
(read by gbench's agent relays: an agent episode ends after its first reply and the action it
asks for) and each suite's endpoint behind a recording tap (SmokeTap): a transparent proxy that
forwards every request and records, per completions request, the temperature, max_tokens and
thinking switch on the wire and whether a reply came back. --thinking, --max-output-tokens and the
temperature rule are the run's own. A suite PASSes (smoke_verdict) when its preflight passed, it
wrote a result, at least one model reply came back, every model request had the expected shape
(temperature per the rule, chat_template_kwargs.enable_thinking per --thinking, max_tokens at most
--max-output-tokens), and an agent suite sent the follow-up request that shows its action ran
while any other suite had its sample scored by its own scorer. Smoke results go to their own
smoke-<timestamp> directory, carry smoke=true, have their accuracy withheld, and the summary is the
PASS/FAIL table: a smoke run measures nothing.

Because nothing is measured, a smoke run has limits a real run never has: a model request through
the tap that receives no byte for GBENCH_SMOKE_HANG_S (default 300) while the server's token
counters (vLLM /metrics generation and prompt token totals) stood still too is a hang (SmokeWatch;
without readable counters only a streamed reply can show it): the tap closes that request's
upstream connection, the suite's processes are stopped and it FAILs at stage "hang"; and the whole
run stops at GBENCH_SMOKE_DEADLINE_S (default DEFAULT_DEADLINE_S, 3900): the suites still running
FAIL at stage "deadline", those not started at "not started".
"""

from __future__ import annotations

import hashlib
import http.client
import http.server
import importlib
import json
import logging
import os
import shutil
import socketserver
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger("gbench.smoke")

#: Suites that grade, search or simulate users with gbench's Gemini cascade (the modules that
#: call gemini_required_skip / gemini_key_live_valid / gemini_api_keys).
GEMINI_SUITES = frozenset({
    "aa_lcr", "acebench", "beam_128k", "bfcl_v4_agentic", "browsecomp", "charxiv", "cimemories",
    "complexfuncbench", "deepsearch_qa", "frames", "gaia", "gaia2", "gdpval", "healthbench",
    "humanitys_last_exam", "imo_answer_bench", "mcp_atlas", "mcp_bench", "putnam",
    "putnam_formal", "simpleqa", "tau2", "tau3", "toolbench", "wildbench", "wildclawbench"})

#: Suites scored from log-probabilities: temperature 0.0 and no thinking whatever the run's mode
#: (each records it: temperature_note, thinking_not_applied_reason).
LOGPROB_SUITES = frozenset({"mmlu", "wmdp"})

#: Agent suites: their first reply asks for an action, so the request after it (the action's
#: result going back to the model) is the evidence that the action ran.
AGENT_SUITES = frozenset({
    "swe_bench_live", "swe_bench_pro", "swe_bench_multilingual", "multi_swe_bench",
    "copilot_bench_swe", "terminal_bench", "skillsbench", "wildclawbench", "gaia2", "tau2", "tau3",
    "mcp_bench", "mcp_atlas", "mobilegym", "mobilegym_pp", "toolbench", "agent_dojo", "swe_lancer",
    "ui_control_osworld"})

#: Suites a smoke run runs at once. 2026-10-11 smoke run (smoke-10-11-031629, 84 suites, 8 at a
#: time): 42.4 min against the 45-min deadline. Replaying its per-suite durations through the
#: pool's scheduler (family caps included) gives 41.8 min at 8 with the order that run used, and
#: 38.1 min at 8, 10, 12 or 16 with the order the next run reads (the smoke durations it
#: recorded): the wall time is terminal_bench's 2,285 s, which no pool size shortens. 12 is where
#: the replay stops improving (16 gives the same); it does not reach every replay's lower bound
#: (22.2 min against 20.0 with terminal_bench at 300 s and aider_polyglot at 453 s). Over 8, with
#: the order the next run reads: 0 min at the measured durations, 4.5 min (26.7 -> 22.2) with
#: terminal_bench and aider_polyglot short; with a wrong duration table (the order that run used,
#: aider_polyglot estimated at 579 s, ran 1,860 s): 3.7 and 8.6 min. The cost: the replay of the
#: run's 588 tapped model requests has more than 16 in flight for 25.7% of the time at 12 against
#: 11.7% at 8, at most 44 in flight (35 measured at 8), under the pool's admission gate (48); at
#: 16 the replay passes 48.
SMOKE_PARALLEL = 12
DEFAULT_MIN_DISK_GB = 100.0
DEFAULT_CHECK_TIMEOUT_S = 300.0
#: A model request with no byte of its reply for this long while the server generated nothing.
DEFAULT_HANG_S = 300.0
#: The whole smoke run's limit (preflight included). Both 2026-10-11 smoke runs (same command:
#: --thinking, keyless endpoint, --eval-limit 1, same terminal_bench task mcmc-sampling-stan, 8
#: at a time) are bounded by terminal_bench: smoke-10-11-031629 37 s of preflight plus 2,284.7 s;
#: smoke-10-11-022420 37 s plus at least 2,470.2 s (stopped by SIGTERM in Harbor's verifier; its
#: last model request at 02:55:09 plus 031629's 692 s from last request to end projects 2,505 s,
#: a 2,542 s path). 3900 s is 1.53 x the slower path. 2700 s before; 031629 finished 2.6 min
#: short of it.
DEFAULT_DEADLINE_S = 3900.0
#: How often SmokeWatch reads the server's token counters.
COUNTERS_EVERY_S = 10.0
#: How often the tap touches a suite's GBENCH_POOL_ACTIVITY_FILE while its replies flow.
ACTIVITY_EVERY_S = 10.0


# =========================================================================== preflight
@dataclass
class Check:
    name: str
    ok: bool
    reason: str = ""
    notes: List[str] = field(default_factory=list)
    seconds: float = 0.0
    kind: str = "suite"          # "suite" or "global"
    #: a global check whose failure stops every suite (disk, Docker, the endpoint from this host);
    #: the others (bridge reachability, the Gemini key) fail only the suites that need them,
    #: which their own checks name
    fatal: bool = False


def _ok(reason: str = "") -> Tuple[bool, str]:
    return True, reason


def _call_tuple(fn: Callable[[], Any]) -> Tuple[bool, str]:
    """A check returning (ok, reason[, ...]); one that raises fails with its message."""
    out = fn()
    if isinstance(out, tuple):
        return bool(out[0]), str(out[1] if len(out) > 1 else "")
    return bool(out), ""


def _call_raises(fn: Callable[[], Any]) -> Tuple[bool, str]:
    """A check that raises (infra_required) on a missing prerequisite."""
    fn()
    return True, ""


def _mod(suite: str):
    return importlib.import_module(f"gbench.runners.eval_suites.{suite}")


def _image_present(image: str) -> Tuple[bool, str]:
    try:
        r = subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=60)
    except Exception as e:                                                 # noqa: BLE001
        return False, f"cannot inspect image {image!r}: {e}"
    if r.returncode != 0:
        return False, (f"local image {image!r} is not present (gbench never pulls; build it as "
                       "the suite's docs/evals page says)")
    return True, ""


def _agentic(suite: str) -> bool:
    from .runners.eval_suites import swe_agentic
    return swe_agentic.resolve_mode(suite)[0] == swe_agentic.AGENTIC


def _swe_v5(suite: str) -> Tuple[bool, str]:
    m = _mod(suite)
    ok, why = getattr(m, f"check_{suite}_prerequisites")()
    if not ok or not _agentic(suite):
        return ok, why
    from .runners.eval_suites import swe_agentic
    return swe_agentic.check_agent_v2_prerequisites(m.AGENT_VERSION)


def _multi_swe() -> Tuple[bool, str]:
    m = _mod("multi_swe_bench")
    ok, why = m.check_multi_swe_bench_prerequisites()
    if ok and _agentic("multi_swe_bench"):
        ok, why = m.check_mswe_agent_prerequisites()
    return ok, why


def _swe_bench_live() -> Tuple[bool, str, List[str]]:
    m = _mod("swe_bench_live")
    settings = m.resolve_settings(None, None)
    from .runners.eval_suites import swe_agentic
    mode, _ = swe_agentic.resolve_mode("swe_bench_live")
    missing = m.missing_prerequisites(settings, mode)
    windows = {s for s in settings if m.setting_family(s) == "windows"}
    hard = {s: w for s, w in missing.items() if s not in windows or len(missing) == len(settings)}
    notes = [f"{m.TAB_LABELS.get(s, s)} tab: {w} (the suite reports this tab infra_required in "
             "its own entry and runs the others)" for s, w in missing.items() if s not in hard]
    if hard:
        return False, "; ".join(f"{m.TAB_LABELS.get(s, s)}: {w}" for s, w in hard.items()), notes
    return True, "", notes


def _swe_bench_pro() -> Tuple[bool, str]:
    m = _mod("swe_bench_pro")
    if os.getenv("GBENCH_SWEBENCH_PRO_AGENTIC") == "1":
        return m.check_swe_bench_pro_prerequisites()
    return m.check_swe_bench_pro_v2_prerequisites()


def _bfcl(suite: str, base_url: Optional[str]) -> Tuple[bool, str]:
    if not base_url:
        return True, ""
    from .runners.eval_suites import bfcl_v4_agentic as b4
    from .runners.eval_suites.base import gemini_api_keys
    docs = _mod(suite).DOCS_URL
    with b4.suite_context(suite, docs, web_search=bool(gemini_api_keys())):
        ok, why, _cats = b4.check_bfcl_prerequisites(base_url)
    return ok, why


def _tau() -> Tuple[bool, str]:
    from .runners.eval_suites import tau_common as t
    ok, why = t.check_role_providers()
    if ok:
        ok, why = t.check_tau_image()
    return ok, why


def _sandbox(extra: Optional[Callable[[], Optional[str]]] = None) -> Tuple[bool, str]:
    from .runners.eval_suites.sandbox import sandbox_skip_reason
    why = sandbox_skip_reason() or (extra() if extra else None)
    return (not why), (why or "")


def _generic(suite: str) -> Tuple[bool, str]:
    """A suite without a prerequisite check: the local image its module names, if any."""
    try:
        m = _mod(suite)
    except ImportError:
        # a plugin suite: its module is wherever the plugin loader found it
        import sys
        from .runners.eval_suites import SUITES
        fn = SUITES.get(suite)
        m = sys.modules.get(getattr(fn, "__module__", "") or "")
        if m is None:
            return True, ""
    for attr in ("_image", "image", "_scorer_image"):
        fn = getattr(m, attr, None)
        if callable(fn):
            try:
                img = fn()
            except TypeError:
                continue
            if isinstance(img, str) and img:
                return _image_present(img)
    return True, ""


def _registry() -> Dict[str, Callable[[Optional[str]], Any]]:
    """suite -> check(base_url) returning (ok, reason) or (ok, reason, notes): each runner's own
    first-thing prerequisite check, called as the runner calls it."""
    def own(suite):
        return lambda _u: _call_tuple(getattr(_mod(suite), f"check_{suite}_prerequisites"))

    reg: Dict[str, Callable[[Optional[str]], Any]] = {}
    for s in ("bigcodebench", "multipl_e", "ojbench", "spider2", "coco_caption",
              "complexfuncbench", "gaia", "gaia2", "mcp_atlas", "mcp_bench", "mobilegym",
              "mobilegym_pp", "skillsbench",
              "terminal_bench", "toolbench", "wildclawbench", "putnam", "putnam_formal",
              "swe_lancer", "ui_control_osworld"):
        reg[s] = own(s)
    reg.update({
        "aime": lambda _u: _call_tuple(_mod("matharena_common").check_prerequisites),
        "hmmt": lambda _u: _call_tuple(_mod("matharena_common").check_prerequisites),
        "aider_polyglot": lambda _u: _call_raises(
            lambda: _mod("aider_polyglot")._check_prereqs(_mod("aider_polyglot")._image(),
                                                          _mod("aider_polyglot")._bench_dir())),
        "livebench": lambda _u: _call_raises(
            lambda: _mod("livebench")._check_prereqs(_mod("livebench")._image())),
        "omnidocbench": lambda _u: _call_raises(
            lambda: _mod("omnidocbench")._check_prereqs(_mod("omnidocbench")._scorer_image())),
        "gorilla_apibench": lambda _u: _call_raises(_mod("gorilla_apibench").check_prereqs),
        "api_bank": lambda _u: _call_raises(_mod("api_bank")._checkout_dir),
        "codeforces": lambda _u: _sandbox(),
        "cruxeval": lambda _u: _sandbox(),
        "scicode": lambda _u: _sandbox(),
        "lcb": lambda _u: _sandbox(lambda: _mod("lcb")._grader_self_check()),
        "copilot_bench_swe": lambda _u: _swe_v5("copilot_bench_swe"),
        "swe_bench_multilingual": lambda _u: _swe_v5("swe_bench_multilingual"),
        "multi_swe_bench": lambda _u: _multi_swe(),
        "swe_bench_live": lambda _u: _swe_bench_live(),
        "swe_bench_pro": lambda _u: _swe_bench_pro(),
        "bfcl": lambda u: _bfcl("bfcl", u),
        "bfcl_v4_agentic": lambda u: _bfcl("bfcl_v4_agentic", u),
        "tau2": lambda _u: _tau(),
        "tau3": lambda _u: _tau(),
    })
    return reg


#: The suites PREFLIGHT_CHECKS covers with their runner's own check (the rest get _generic).
PREFLIGHT_CHECKS = frozenset(_registry())


def check_suite(suite: str, base_url: Optional[str]) -> Check:
    """One suite's preflight: its own check (or _generic), then the Gemini key when it needs one."""
    t0 = time.monotonic()
    notes: List[str] = []
    try:
        fn = _registry().get(suite)
        out = fn(base_url) if fn else _generic(suite)
        ok, why = bool(out[0]), str(out[1] or "")
        if len(out) > 2:
            notes = list(out[2] or [])
    except Exception as e:                                                 # noqa: BLE001
        ok, why = False, f"{type(e).__name__}: {e}" if "[INFRA REQUIRED]" not in str(e) else str(e)
    if ok and suite in GEMINI_SUITES:
        from .runners.eval_suites.base import gemini_api_keys
        if not gemini_api_keys():
            ok, why = False, ("GEMINI_API_KEY (or GEMINI_API_KEYS) is unset: this suite grades, "
                              "searches or simulates with gbench's Gemini cascade "
                              f"(docs/evals/{suite}.md)")
    from .runners.eval_suites.base import redact_text
    return Check(suite, ok, redact_text(why), notes, round(time.monotonic() - t0, 1))


def _endpoint_key() -> Optional[str]:
    for name in ("VLLM_API_KEY", "OPENAI_API_KEY"):
        v = (os.environ.get(name) or "").strip()
        if v:
            return v
    return None


def check_endpoint_host(base_url: str, timeout: float = 15.0) -> Check:
    t0 = time.monotonic()
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(url)
    key = _endpoint_key()
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:            # noqa: S310
            data = json.loads(r.read().decode("utf-8", "replace"))
        ids = [m.get("id") for m in (data.get("data") or []) if isinstance(m, dict)]
        ok, why = bool(ids), (f"serves {', '.join(map(str, ids[:3]))}" if ids else
                              "GET /v1/models listed no model")
    except Exception as e:                                                 # noqa: BLE001
        ok, why = False, f"GET {url} failed: {type(e).__name__}: {e}"
    return Check("endpoint from this host", ok, why, seconds=round(time.monotonic() - t0, 1),
                 kind="global", fatal=True)


#: Local images (with python3) the bridge check may use, in order; the first present one is run.
def _bridge_images() -> List[str]:
    names: List[str] = []
    for fn in (lambda: _mod("tau_common").tau2_image(), lambda: _mod("matharena_common").image(),
               lambda: _mod("gaia2")._image(), lambda: _mod("toolbench")._image(),
               lambda: _mod("mcp_bench")._image()):
        try:
            names.append(fn())
        except Exception:                                                  # noqa: BLE001
            continue
    return names


_BRIDGE_PROBE = ("import json,sys,urllib.request\n"
                 "r=urllib.request.Request(sys.argv[1])\n"
                 "k=sys.argv[2] if len(sys.argv)>2 else ''\n"
                 "k and r.add_header('Authorization','Bearer '+k)\n"
                 "d=json.load(urllib.request.urlopen(r,timeout=20))\n"
                 "print(len(d.get('data') or []))\n")


def bridge_url(base_url: str) -> str:
    """The endpoint as a container on the default bridge reaches it: a loopback host becomes
    host.docker.internal (mapped to the bridge gateway with --add-host host-gateway)."""
    parts = urllib.parse.urlsplit(base_url)
    host = parts.hostname or ""
    if host in ("127.0.0.1", "localhost", "0.0.0.0", "::1"):
        netloc = "host.docker.internal" + (f":{parts.port}" if parts.port else "")
        parts = parts._replace(netloc=netloc)
    return urllib.parse.urlunsplit(parts)


def check_endpoint_bridge(base_url: str) -> Check:
    t0 = time.monotonic()
    url = bridge_url(base_url).rstrip("/") + "/models"
    tried = []
    for image in _bridge_images():
        if not _image_present(image)[0]:
            continue
        tried.append(image)
        cmd = ["docker", "run", "--rm", "--pull", "never", "--network", "bridge",
               "--add-host", "host.docker.internal:host-gateway", "--entrypoint", "python3",
               image, "-c", _BRIDGE_PROBE, url]
        key = _endpoint_key()
        if key:
            cmd.append(key)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except Exception as e:                                             # noqa: BLE001
            return Check("endpoint from a bridge container", False, f"{image}: {e}",
                         seconds=round(time.monotonic() - t0, 1), kind="global")
        out = (r.stdout or "").strip()
        if r.returncode == 0 and out.isdigit() and int(out) > 0:
            return Check("endpoint from a bridge container", True,
                         f"{url} answers from {image} on --network bridge",
                         seconds=round(time.monotonic() - t0, 1), kind="global")
        err = (r.stderr or out).strip().splitlines()
        if r.returncode in (125, 126, 127) and "python3" in " ".join(err):
            continue                                   # no python3 in that image: next one
        from .runners.eval_suites.base import redact_text
        return Check("endpoint from a bridge container", False,
                     f"GET {url} from {image} on --network bridge failed (exit {r.returncode}): "
                     + redact_text(err[-1] if err else "no output")
                     + "; suites whose containers reach the endpoint directly need the server "
                       "to listen on an address the bridge gateway reaches (e.g. 0.0.0.0)",
                     seconds=round(time.monotonic() - t0, 1), kind="global")
    return Check("endpoint from a bridge container", False,
                 "no local image with python3 to probe from (tried: "
                 + (", ".join(tried) or "none present") + "); build one of the suites' images",
                 seconds=round(time.monotonic() - t0, 1), kind="global")


def check_disk(path: Path, min_gb: float) -> Check:
    p = Path(path).absolute()
    while not p.exists() and p != p.parent:
        p = p.parent
    free = shutil.disk_usage(p).free / 1e9
    return Check(f"disk free at {p}", free >= min_gb,
                 f"{free:.0f} GB free (minimum {min_gb:g}, GBENCH_PREFLIGHT_MIN_DISK_GB)",
                 kind="global", fatal=True)


def check_docker() -> Check:
    t0 = time.monotonic()
    if not shutil.which("docker"):
        return Check("docker daemon", False, "docker CLI not on PATH", kind="global", fatal=True)
    try:
        r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                           capture_output=True, text=True, timeout=30)
        ok = r.returncode == 0
        why = (f"server {r.stdout.strip()}" if ok else
               f"`docker info` failed: {(r.stderr or '').strip()[:200]}")
    except Exception as e:                                                 # noqa: BLE001
        ok, why = False, f"`docker info` failed: {e}"
    return Check("docker daemon", ok, why, seconds=round(time.monotonic() - t0, 1),
                 kind="global", fatal=True)


def check_gemini(suites: Sequence[str]) -> Optional[Check]:
    need = sorted(set(suites) & GEMINI_SUITES)
    if not need:
        return None
    from .runners.eval_suites.base import gemini_api_keys
    ok = bool(gemini_api_keys())
    return Check("Gemini key", ok,
                 (f"set ({len(need)} selected suites use the Gemini cascade)" if ok else
                  f"GEMINI_API_KEY (or GEMINI_API_KEYS) is unset; {len(need)} selected suites "
                  f"need it: {', '.join(need)}"), kind="global")


@dataclass
class PreflightReport:
    checks: List[Check]
    seconds: float

    @property
    def global_failed(self) -> List[Check]:
        return [c for c in self.checks if c.kind == "global" and c.fatal and not c.ok]

    @property
    def suites(self) -> Dict[str, Check]:
        return {c.name: c for c in self.checks if c.kind == "suite"}

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def record(self) -> Dict[str, Any]:
        return {"ok": self.ok, "seconds": self.seconds,
                "checks": [c.__dict__ for c in self.checks]}


def run_preflight(suites: Sequence[str], base_url: Optional[str], results_dir: Path, *,
                  timeout_s: Optional[float] = None,
                  suite_check: Callable[[str, Optional[str]], Check] = check_suite,
                  global_checks: Optional[Sequence[Callable[[], Check]]] = None
                  ) -> PreflightReport:
    """Every global check (default _default_globals) and suite check, at once in daemon threads;
    a check still running after `timeout_s` (GBENCH_PREFLIGHT_TIMEOUT_S) fails, and its thread is
    left behind (the process does not wait for it)."""
    t0 = time.monotonic()
    if timeout_s is None:
        timeout_s = _env_float("GBENCH_PREFLIGHT_TIMEOUT_S", DEFAULT_CHECK_TIMEOUT_S)
    if global_checks is None:
        global_checks = _default_globals(suites, base_url, results_dir)
    jobs: List[Tuple[str, str, Callable[[], Any]]] = [
        ("global", f"global check {i + 1}", fn) for i, fn in enumerate(global_checks)]
    locks = family_locks()

    def _suite_job(s: str) -> Callable[[], Any]:
        lock = locks.get(s)
        if lock is None:
            return lambda: suite_check(s, base_url)

        def _locked() -> Any:
            with lock:
                return suite_check(s, base_url)
        return _locked
    jobs += [("suite", s, _suite_job(s)) for s in suites]
    results: Dict[int, Any] = {}

    def _worker(i: int, fn: Callable[[], Any]) -> None:
        try:
            results[i] = fn()
        except Exception as e:                                             # noqa: BLE001
            results[i] = e
    threads = []
    for i, (_kind, name, fn) in enumerate(jobs):
        th = threading.Thread(target=_worker, args=(i, fn), daemon=True, name=f"preflight-{name}")
        th.start()
        threads.append(th)
    deadline = time.monotonic() + timeout_s
    for th in threads:
        th.join(max(0.0, deadline - time.monotonic()))
    checks: List[Check] = []
    for i, (kind, name, _fn) in enumerate(jobs):
        out = results.get(i)
        if isinstance(out, Check):
            out.kind = kind
            checks.append(out)
        elif isinstance(out, Exception):
            checks.append(Check(name, False, f"{type(out).__name__}: {out}", kind=kind))
        else:
            checks.append(Check(name, False, f"the check did not finish within {timeout_s:.0f} s "
                                "(GBENCH_PREFLIGHT_TIMEOUT_S)", kind=kind))
    return PreflightReport(checks, round(time.monotonic() - t0, 1))


def family_locks() -> Dict[str, threading.Lock]:
    """suite -> the lock its preflight check holds: one per pool family (pool.FAMILIES), so the
    checks of suites that share a harness root, image or ports never run at the same time."""
    from .pool import FAMILIES
    out: Dict[str, threading.Lock] = {}
    for _name, members, _cap in FAMILIES:
        lock = threading.Lock()
        for m in members:
            out.setdefault(m, lock)
    return out


def _default_globals(suites: Sequence[str], base_url: Optional[str],
                     results_dir: Path) -> List[Callable[[], Check]]:
    min_gb = _env_float("GBENCH_PREFLIGHT_MIN_DISK_GB", DEFAULT_MIN_DISK_GB)
    jobs: List[Callable[[], Check]] = [lambda: check_disk(results_dir, min_gb), check_docker]
    if base_url:
        jobs += [lambda: check_endpoint_host(base_url), lambda: check_endpoint_bridge(base_url)]
    gem = check_gemini(suites)
    if gem is not None:
        jobs.append(lambda: gem)
    return jobs


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


def _cell(text: str, width: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= width else text[:width - 3] + "..."


def format_preflight(report: PreflightReport, reason_width: int = 150) -> str:
    rows = [c for c in report.checks if c.kind == "global"] + sorted(
        (c for c in report.checks if c.kind == "suite"), key=lambda c: (c.ok, c.name))
    w = max([len(c.name) for c in rows] + [10])
    nfail = sum(1 for c in rows if not c.ok)
    lines = [f"PREFLIGHT: {len(rows) - nfail} PASS, {nfail} FAIL "
             f"({sum(1 for c in rows if c.kind == 'suite')} suites, "
             f"{sum(1 for c in rows if c.kind == 'global')} global checks, {report.seconds:.0f} s)",
             f"  {'check':<{w}}  result  reason", f"  {'-' * w}  ------  {'-' * 40}"]
    for c in rows:
        lines.append(f"  {c.name:<{w}}  {'PASS' if c.ok else 'FAIL':<6}  "
                     f"{_cell(c.reason, reason_width)}")
        for n in c.notes:
            lines.append(f"  {'':<{w}}  {'note':<6}  {_cell(n, reason_width)}")
    return "\n".join(lines)


# =========================================================================== tap
_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
        "trailers", "transfer-encoding", "upgrade", "host", "content-length"}


def conversation_key(body: Mapping[str, Any]) -> Tuple[Optional[str], int]:
    """(a hash of the request's first two messages, its message count) for a chat request; a
    completions prompt hashes its first 2,000 characters."""
    msgs = body.get("messages")
    if isinstance(msgs, list):
        head = json.dumps(msgs[:2], sort_keys=True, default=str)
        return hashlib.sha256(head.encode("utf-8")).hexdigest()[:16], len(msgs)
    prompt = body.get("prompt")
    if prompt is not None:
        head = json.dumps(prompt, default=str)[:2000]
        return hashlib.sha256(head.encode("utf-8")).hexdigest()[:16], 1
    return None, 0


def request_shape(body: Mapping[str, Any]) -> Dict[str, Any]:
    """What a completions request asked for: temperature, max_tokens (or max_completion_tokens),
    the thinking switch (chat_template_kwargs.enable_thinking), stream."""
    ctk = body.get("chat_template_kwargs")
    mt = body.get("max_tokens", body.get("max_completion_tokens"))
    return {"temperature": body.get("temperature", "absent"),
            "max_tokens": mt,
            "enable_thinking": (ctk.get("enable_thinking") if isinstance(ctk, dict)
                                and "enable_thinking" in ctk else "absent"),
            "stream": bool(body.get("stream"))}


class SmokeTap:
    """A transparent HTTP proxy in front of the served endpoint for one smoke suite: every
    method and path is forwarded unchanged (headers, body, the reply streamed back as it comes),
    and each completions request is recorded as one JSON line in `record_path` (request_shape,
    conversation_key, the reply's status, bytes and timing). It changes nothing: a smoke run's
    evidence of what went over the wire. It adds no timeout of its own; each completions request
    in flight is listed (`inflight`: start, last byte, streamed) for SmokeWatch, which may end a
    hung one (`abort`: its upstream connection is shut down, the record gets `hang`). While
    replies flow it touches `activity_path` (the suite's GBENCH_POOL_ACTIVITY_FILE, read by the
    pool's stall alarm) at most every ACTIVITY_EVERY_S."""

    def __init__(self, upstream: str, record_path: Path, bind_host: str = "0.0.0.0",
                 activity_path: Optional[Path] = None):
        parts = urllib.parse.urlsplit(upstream.strip())
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"the smoke tap needs an http(s) endpoint URL, not {upstream!r}")
        self._scheme, self._host, self._port = parts.scheme, parts.hostname, parts.port
        base = parts.path.rstrip("/")
        self._base = base[:-3] if base.endswith("/v1") else base
        self.record_path = Path(record_path)
        self.bind_host = bind_host
        self._lock = threading.Lock()
        self._server: Optional[socketserver.ThreadingTCPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.requests = 0
        self.activity_path = Path(activity_path) if activity_path else None
        self._activity_at = 0.0
        self._next_id = 0
        #: id -> {"start", "last" (time.monotonic of the last reply byte, else the start),
        #: "stream", "path", "conn", "rec"}
        self.inflight: Dict[int, Dict[str, Any]] = {}

    @property
    def port(self) -> int:
        return self._server.server_address[1] if self._server else 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def _record(self, rec: Dict[str, Any]) -> None:
        with self._lock:
            self.requests += 1
            with open(self.record_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    def _touch(self) -> None:
        if self.activity_path is None:
            return
        now = time.monotonic()
        if now - self._activity_at < ACTIVITY_EVERY_S:
            return
        self._activity_at = now
        try:
            self.activity_path.touch()
        except OSError:
            pass

    def _enter(self, rec: Dict[str, Any], conn: Any) -> int:
        with self._lock:
            self._next_id += 1
            rid = self._next_id
            now = time.monotonic()
            self.inflight[rid] = {"start": now, "last": now, "stream": bool(rec.get("stream")),
                                  "path": rec.get("path"), "conn": conn, "rec": rec}
        return rid

    def _progress(self, rid: Optional[int]) -> None:
        if rid is not None:
            entry = self.inflight.get(rid)
            if entry is not None:
                entry["last"] = time.monotonic()
        self._touch()

    def _leave(self, rid: Optional[int]) -> None:
        if rid is not None:
            with self._lock:
                self.inflight.pop(rid, None)

    def silent(self) -> List[Tuple[float, Dict[str, Any]]]:
        """(seconds since its last reply byte, entry) for each completions request in flight,
        longest silence first."""
        now = time.monotonic()
        with self._lock:
            entries = list(self.inflight.values())
        return sorted(((now - e["last"], e) for e in entries), key=lambda t: -t[0])

    def abort(self, entry: Dict[str, Any], why: str) -> None:
        """End one request in flight: `hang` on its record, its upstream connection shut down
        (vLLM aborts a request whose client left)."""
        entry["rec"]["hang"] = why
        sock = getattr(entry.get("conn"), "sock", None)
        if sock is not None:
            try:
                import socket
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def _connection(self):
        cls = http.client.HTTPSConnection if self._scheme == "https" else http.client.HTTPConnection
        return cls(self._host, self._port, timeout=None)

    def start(self) -> "SmokeTap":
        tap = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):                       # one line per request is noise
                return

            def _body(self) -> bytes:
                if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                    out = b""
                    while True:
                        line = self.rfile.readline().strip()
                        n = int(line.split(b";")[0] or b"0", 16)
                        if n == 0:
                            self.rfile.readline()
                            return out
                        out += self.rfile.read(n)
                        self.rfile.readline()
                n = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(n) if n > 0 else b""

            def _proxy(self):
                body = self._body()
                path = self.path
                rec: Optional[Dict[str, Any]] = None
                if self.command == "POST" and path.split("?")[0].endswith("/completions"):
                    try:
                        data = json.loads(body or b"{}")
                    except ValueError:
                        data = {}
                    if isinstance(data, dict):
                        conv, nmsg = conversation_key(data)
                        rec = {"t": time.time(), "path": path.split("?")[0],
                               **request_shape(data), "conversation": conv,
                               "messages": nmsg, "status": None, "bytes": 0}
                t0 = time.monotonic()
                conn = tap._connection()
                rid = tap._enter(rec, conn) if rec is not None else None
                headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP}
                if body:
                    headers["Content-Length"] = str(len(body))
                try:
                    conn.request(self.command, tap._base + path, body=body or None,
                                 headers=headers)
                    resp = conn.getresponse()
                except Exception as e:                                     # noqa: BLE001
                    tap._leave(rid)
                    if rec is not None:
                        rec.update(status=None, duration_s=round(time.monotonic() - t0, 2))
                        rec.setdefault("error", f"{type(e).__name__}: {e}")
                        tap._record(rec)
                    try:
                        self.send_error(504 if rec is not None and rec.get("hang") else 502,
                                        f"smoke tap: upstream unreachable ({type(e).__name__})")
                    except Exception:                                      # noqa: BLE001
                        pass
                    conn.close()
                    return
                if rec is not None:
                    rec["status"] = resp.status
                try:
                    self.send_response(resp.status, resp.reason)
                    length = resp.getheader("Content-Length")
                    for k, v in resp.getheaders():
                        if k.lower() not in _HOP:
                            self.send_header(k, v)
                    if length is not None and "chunked" not in (
                            resp.getheader("Transfer-Encoding") or "").lower():
                        self.send_header("Content-Length", length)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    if self.command != "HEAD":
                        read = getattr(resp, "read1", None) or resp.read
                        while True:
                            chunk = read(65536)
                            if not chunk:
                                break
                            if rec is not None and rec["bytes"] == 0:
                                rec["first_byte_s"] = round(time.monotonic() - t0, 2)
                            if rec is not None:
                                rec["bytes"] += len(chunk)
                            tap._progress(rid)
                            self.wfile.write(chunk)
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError) as e:
                    if rec is not None:
                        rec["error"] = f"client closed the connection ({type(e).__name__})"
                except Exception as e:                                     # noqa: BLE001
                    # the upstream connection shut down by abort() or stop(), or failed mid-reply
                    if rec is not None:
                        rec.setdefault("error", f"{type(e).__name__}: {e}")
                finally:
                    tap._leave(rid)
                    conn.close()
                    if rec is not None:
                        rec["duration_s"] = round(time.monotonic() - t0, 2)
                        tap._record(rec)

            do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _proxy

        class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        self._server = Server((self.bind_host, 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True,
                                        name="smoke-tap")
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving, and end every request still in flight (its suite's process is gone or
        was stopped: no one reads that reply, and the server stops generating it)."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        with self._lock:
            entries = list(self.inflight.values())
        for entry in entries:
            entry["rec"].setdefault("error", "the suite ended before its reply (smoke tap stopped)")
            sock = getattr(entry.get("conn"), "sock", None)
            if sock is not None:
                try:
                    import socket
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        if self._thread is not None:
            self._thread.join(timeout=10)


def read_tap(path: Path) -> List[Dict[str, Any]]:
    out = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return out


_COUNTER_RE = None


def server_token_total(metrics_text: str) -> Optional[float]:
    """vllm:generation_tokens_total + vllm:prompt_tokens_total over every engine in a /metrics
    page (they advance while the server prefills or decodes anything), or None."""
    import re
    global _COUNTER_RE
    if _COUNTER_RE is None:
        _COUNTER_RE = re.compile(r"^vllm:(?:generation|prompt)_tokens_total(?:\{[^}]*\})?\s+"
                                 r"([0-9.eE+-]+)\s*$")
    total, seen = 0.0, False
    for line in (metrics_text or "").splitlines():
        m = _COUNTER_RE.match(line.strip())
        if m:
            try:
                total += float(m.group(1))
                seen = True
            except ValueError:
                pass
    return total if seen else None


class SmokeWatch:
    """The smoke run's hang detection (module docstring), called by the pool for each running
    suite: a request through the suite's tap with no reply byte for `hang_s` is a hang when the
    server's token counters (server_token_total) have not moved for `hang_s` either; without
    readable counters, only when its reply is streamed (a non-streamed reply shows nothing until
    its end, and the run's deadline bounds it). The hung request is aborted (SmokeTap.abort) and
    the reason returned, so the pool stops the suite."""

    def __init__(self, taps: Mapping[str, "SmokeTap"], metrics_url: Optional[str],
                 hang_s: float = DEFAULT_HANG_S,
                 fetch: Optional[Callable[[str], Optional[str]]] = None,
                 clock: Callable[[], float] = time.monotonic):
        from .pool import fetch_metrics
        self.taps, self.url, self.hang_s = taps, metrics_url, float(hang_s)
        self._fetch = fetch or fetch_metrics
        self._clock = clock
        self._read_at: Optional[float] = None
        self._value: Optional[float] = None
        self._changed_at: Optional[float] = None
        self.readable = False

    def _server_idle_s(self) -> Optional[float]:
        """Seconds since the server's token counters last moved (None: unreadable)."""
        now = self._clock()
        if self._read_at is None or now - self._read_at >= COUNTERS_EVERY_S:
            self._read_at = now
            value = server_token_total(self._fetch(self.url) or "") if self.url else None
            if value is None:
                self.readable = False
                self._value = self._changed_at = None
            else:
                if self._value is None or value != self._value or self._changed_at is None:
                    self._changed_at = now
                self._value, self.readable = value, True
        if not self.readable or self._changed_at is None:
            return None
        return now - self._changed_at

    def __call__(self, child: Any) -> Optional[str]:
        tap = self.taps.get(child.suite)
        if tap is None or self.hang_s <= 0:
            return None
        silent = [(s, e) for s, e in tap.silent() if s >= self.hang_s]
        if not silent:
            return None
        idle = self._server_idle_s()
        for secs, entry in silent:
            if idle is not None:
                if idle < self.hang_s:
                    return None
                cause = (f"while the server's token counters stood still for {idle:.0f} s")
            elif entry["stream"]:
                cause = "on a streamed reply (the server's token counters are unreadable)"
            else:
                continue
            why = (f"hang: a model request ({entry.get('path')}, "
                   f"{'streamed' if entry['stream'] else 'not streamed'}) received no reply byte "
                   f"for {secs:.0f} s {cause} (GBENCH_SMOKE_HANG_S={self.hang_s:g})")
            tap.abort(entry, why)
            return why
        return None


def smoke_child_env(suite: str, args: Any) -> Dict[str, str]:
    """The environment `gbench --smoke` adds to one suite's process: GBENCH_SMOKE=1; for
    AGENT_SUITES, GBENCH_SMOKE_RELAY_REPLIES=1 (their gbench relays serve one model reply per
    conversation, then answer the follow-up with GBENCH_SMOKE_STOP: the follow-up proves the
    agent read the reply and acted); and GBENCH_RELAY_EXPECTED_SHAPE, the request shape the run's
    flags ask for (expected_shape), which every relay checks on what it sends upstream."""
    env = {"GBENCH_SMOKE": "1"}
    if suite in AGENT_SUITES:
        env["GBENCH_SMOKE_RELAY_REPLIES"] = "1"
    exp = expected_shape(suite, thinking=bool(getattr(args, "eval_thinking", False)),
                         temperature=getattr(args, "temperature", None),
                         max_output_tokens=getattr(args, "max_output_tokens", None))
    shape: Dict[str, Any] = {"temperature": exp.temperature}
    if suite not in LOGPROB_SUITES:
        shape["enable_thinking"] = exp.thinking
    # Only where gbench itself sets max_tokens to the run's value (the agent suites,
    # swe_agentic.AGENT_MAX_TOKENS_SUITES); other suites may send their own official cap.
    from .runners.eval_suites.swe_agentic import AGENT_MAX_TOKENS_SUITES
    if exp.max_tokens and suite in AGENT_MAX_TOKENS_SUITES:
        shape["max_tokens_at_most"] = int(exp.max_tokens)
    env["GBENCH_RELAY_EXPECTED_SHAPE"] = json.dumps(shape, sort_keys=True)
    return env


# =========================================================================== verdict
@dataclass
class Expected:
    temperature: float
    thinking: bool
    max_tokens: Optional[int]
    temperature_source: str = ""


def expected_shape(suite: str, *, thinking: bool, temperature: Optional[float],
                   max_output_tokens: Optional[int],
                   env: Optional[Mapping[str, str]] = None) -> Expected:
    """The request shape the run's flags ask for: GBENCH_<SUITE>_TEMPERATURE, else --temperature,
    else 0.0 for the log-probability suites, else 1.0 with --thinking and 0.0 without it
    (base.resolve_temperature); --thinking (not for the log-probability suites); max_tokens at
    most --max-output-tokens."""
    env = os.environ if env is None else env
    from .runners.eval_suites.base import legacy_temperature_env_vars, temperature_env_var
    for var in (temperature_env_var(suite), *legacy_temperature_env_vars(suite)):
        raw = (env.get(var) or "").strip()
        if raw:
            try:
                return Expected(float(raw), thinking and suite not in LOGPROB_SUITES,
                                max_output_tokens, f"env:{var}")
            except ValueError:
                pass
    if temperature is not None:
        # the log-probability suites honour --temperature too (resolve_temperature(...,
        # thinking=False): env, then --temperature, then the rule); they never think
        return Expected(float(temperature), thinking and suite not in LOGPROB_SUITES,
                        max_output_tokens, "--temperature")
    if suite in LOGPROB_SUITES:
        return Expected(0.0, False, max_output_tokens, "log-probability scoring")
    return Expected(1.0 if thinking else 0.0, thinking, max_output_tokens,
                    "thinking" if thinking else "no-think")


def is_probe(rec: Mapping[str, Any]) -> bool:
    """A one-token request (prompt-token counts, multimodal probes, a letter read from
    log-probabilities): not an answer, so not checked for shape or counted as a reply."""
    mt = rec.get("max_tokens")
    return isinstance(mt, int) and not isinstance(mt, bool) and mt <= 1


def shape_mismatches(records: Iterable[Mapping[str, Any]], expected: Expected,
                     suite: str = "", official_caps: Iterable[int] = (),
                     include_probes: bool = False) -> List[str]:
    """Each way a model request differed from `expected` (one line per kind, with its count).
    One-token requests are left out, unless `include_probes` (a suite whose every request is one
    token, e.g. a letter read from log-probabilities): then their temperature is checked."""
    found: Dict[str, int] = {}
    allowed = {c for c in official_caps if isinstance(c, int)}
    for r in records:
        probe = is_probe(r)
        if probe and not include_probes:
            continue
        t = r.get("temperature")
        if t == "absent" or t is None:
            found["no temperature sent (a temperature is always sent)"] = found.get(
                "no temperature sent (a temperature is always sent)", 0) + 1
        else:
            try:
                if abs(float(t) - expected.temperature) > 1e-9:
                    k = (f"temperature {t} sent, expected {expected.temperature:g} "
                         f"({expected.temperature_source})")
                    found[k] = found.get(k, 0) + 1
            except (TypeError, ValueError):
                k = f"temperature {t!r} sent"
                found[k] = found.get(k, 0) + 1
        if (not probe and suite not in LOGPROB_SUITES
                and str(r.get("path", "")).endswith("/chat/completions")):
            et = r.get("enable_thinking")
            if expected.thinking and et is not True:
                k = (f"--thinking is on but chat_template_kwargs.enable_thinking was "
                     f"{'not sent' if et == 'absent' else repr(et)}")
                found[k] = found.get(k, 0) + 1
            elif not expected.thinking and et is True:
                k = "--thinking is off but chat_template_kwargs.enable_thinking=true was sent"
                found[k] = found.get(k, 0) + 1
        mt = r.get("max_tokens")
        if (expected.max_tokens and isinstance(mt, int) and not isinstance(mt, bool)
                and mt > expected.max_tokens and mt not in allowed):
            k = f"max_tokens {mt} sent, above --max-output-tokens {expected.max_tokens}"
            found[k] = found.get(k, 0) + 1
    return [f"{k} ({n} request{'s' if n != 1 else ''})" for k, n in found.items()]


def follow_up_seen(records: Iterable[Mapping[str, Any]]) -> bool:
    """Whether a request continued a conversation after it got a reply: the same first two
    messages, more messages than a request before it that was answered."""
    answered: Dict[str, int] = {}
    for r in sorted((r for r in records if not is_probe(r)), key=lambda r: r.get("t") or 0):
        conv, n = r.get("conversation"), int(r.get("messages") or 0)
        if not conv:
            continue
        if conv in answered and n > answered[conv]:
            return True
        if isinstance(r.get("status"), int) and 200 <= r["status"] < 300 and r.get("bytes"):
            answered[conv] = max(answered.get(conv, 0), n)
    return False


def _relay_records(obj: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        if "requests_relayed" in obj and "relays_to" in obj:
            out.append(obj)
            return out
        for k, v in obj.items():
            if k != "sample_traces":
                out.extend(_relay_records(v))
    elif isinstance(obj, list):
        for v in obj[:200]:
            out.extend(_relay_records(v))
    return out


def model_relays(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """The EndpointRelay records in a result that relay to the model under test (a Gemini
    relay's relays_to is a dict of Gemini URLs)."""
    return [r for r in _relay_records(result) if isinstance(r.get("relays_to"), str)]


def scored_evidence(result: Mapping[str, Any]) -> Tuple[bool, str]:
    """Whether the suite's own scorer scored at least one sample: a numeric accuracy over at least
    one question, or a trace the scorer marked (is_correct True/False, counted in the score)."""
    if result.get("smoke_scored") is not None:
        return bool(result["smoke_scored"]), str(result.get("smoke_scored_reason") or "")
    acc = result.get("accuracy")
    if isinstance(acc, (int, float)) and not isinstance(acc, bool) and \
            (result.get("total_questions") or 0) > 0:
        return True, "accuracy computed over the sample"
    from .runners.eval_suites.base import is_scored
    for t in result.get("sample_traces") or []:
        if isinstance(t, dict) and isinstance(t.get("is_correct"), bool) and is_scored(t):
            return True, "a trace was scored"
    return False, str(result.get("accuracy_withheld_reason") or "no sample was scored")


def _official_caps(result: Mapping[str, Any]) -> List[int]:
    off = result.get("official_sampling")
    caps = []
    if isinstance(off, dict):
        for k in ("max_tokens", "official_max_tokens"):
            if isinstance(off.get(k), int):
                caps.append(off[k])
    return caps


def _relay_follow_up(rec: Mapping[str, Any]) -> bool:
    """Whether an EndpointRelay record shows a request that continued a conversation after a
    served reply (smoke_second_requests under smoke_replies; else two completed replies)."""
    sm = rec.get("smoke") if isinstance(rec.get("smoke"), dict) else {}
    if int(rec.get("smoke_second_requests") or sm.get("smoke_second_requests") or 0) > 0:
        return True
    if sm.get("follow_up_received") or any(
            isinstance(c, dict) and c.get("follow_up_arrived")
            for c in (sm.get("conversations") or {}).values()):
        return True
    if sm:
        # a smoke relay serves one reply per conversation: two replies are two conversations
        return False
    return int(rec.get("completion_replies") or 0) >= 2


def smoke_verdict(suite: str, *, expected: Expected, preflight: Optional[Check] = None,
                  result: Optional[Mapping[str, Any]] = None,
                  tap: Sequence[Mapping[str, Any]] = (), exit_code: Optional[int] = None,
                  log_tail: str = "", stopped: Optional[str] = None) -> Dict[str, Any]:
    """PASS or FAIL for one suite, with the stage that failed and its reason (module docstring)."""
    def out(ok: bool, stage: str, reason: str, **kw: Any) -> Dict[str, Any]:
        return {"suite": suite, "verdict": "PASS" if ok else "FAIL", "stage": stage,
                "reason": reason, **kw}

    if preflight is not None and not preflight.ok:
        return out(False, "pre-check", preflight.reason)
    # every completions request counts for the reply; the one-token ones (probes, letters read
    # from log-probabilities) are shape-checked only when they are all the suite sends
    model = list(tap)
    answers = [r for r in model if not is_probe(r)]
    replies = [r for r in model if isinstance(r.get("status"), int)
               and 200 <= r["status"] < 300 and r.get("bytes")]
    counts = {"requests": len(model), "replies": len(replies)}
    if stopped:
        # the pool stopped the suite: a hung model request (SmokeWatch) or the run's deadline
        return out(False, "hang" if stopped.startswith("hang") else "deadline", stopped,
                   **counts)
    if result is None:
        return out(False, "result", f"no result file written (gbench exited {exit_code})"
                   + (f"; last log line: {log_tail}" if log_tail else ""), **counts)
    relays = model_relays(result)
    if result.get("failed") or result.get("status") in ("error", "failed", "timeout", "skipped"):
        why = str(result.get("error") or result.get("skip_reason")
                  or result.get("accuracy_withheld_reason") or result.get("status"))
        stage = "pre-check" if "[INFRA REQUIRED]" in why else "harness"
        # An agent harness that reports the relay's GBENCH_SMOKE_STOP (a 400 on the follow-up
        # request) as its own error (2026-10-11 smoke run: swe_bench_pro, swe_bench_multilingual
        # "could not run ... BadRequestError"): the stop is the smoke run's own end of the
        # episode, so the follow-up check below decides.
        # terminal_bench reports the same stop as [INFRA REQUIRED] ("5 raised BadRequestError
        # ... GBENCH_SMOKE_STOP", 2026-10-11 rerun): excused only when the error names the stop.
        smoke_stopped = suite in AGENT_SUITES and any(_relay_follow_up(r) for r in relays) and (
            stage == "harness" or "GBENCH_SMOK" in why)   # the error text can be cut short
        if not smoke_stopped:
            return out(False, stage, why, **counts)
    if not replies:
        if not model:
            return out(False, "reply", "the suite sent no model request through the endpoint",
                       **counts)
        statuses = sorted({str(r.get("status")) for r in model})
        errs = sorted({str(r.get("error")) for r in model if r.get("error")})
        return out(False, "reply", f"{len(model)} model request(s), no reply received "
                   f"(statuses {', '.join(statuses)}" + (f"; {errs[0]}" if errs else "") + ")",
                   **counts)
    bad = shape_mismatches(answers or model, expected, suite, _official_caps(result),
                           include_probes=not answers)
    # The agent suites send the run's --max-output-tokens on every model request (the user's
    # decision of 2026-10-11, swe_agentic.AGENT_MAX_TOKENS_SUITES; the tap sees each request after
    # gbench's relay set it). One that reached the endpoint without it ran uncapped: the
    # 2026-10-11 smoke run's terminal_bench and aider_polyglot, whose relay did not start on a
    # keyless endpoint, PASSed with every request at max_tokens null.
    from .runners.eval_suites.swe_agentic import AGENT_MAX_TOKENS_SUITES
    if suite in AGENT_MAX_TOKENS_SUITES and expected.max_tokens:
        uncapped = [r for r in answers if r.get("max_tokens") in (None, "absent")]
        if uncapped:
            bad.append(f"no max_tokens sent on {len(uncapped)} agent request(s) (agent suites "
                       f"send --max-output-tokens {expected.max_tokens})")
    for r in relays:
        # the relay's own check of each request after its body_fields (EndpointRelay
        # shape_check): what the agent's requests asked for, as the tap saw them too
        viol = r.get("shape_violations") or (r.get("shape_check") or {}).get("violations")
        if viol:
            ex = ((r.get("shape_check") or {}).get("examples") or [{}])[0]
            bad.append("relay shape check: " + ", ".join(f"{k} x{v}" for k, v in
                                                          sorted(viol.items()))
                       + (f" (e.g. {ex.get('field')} expected {ex.get('expected')!r}, sent "
                          f"{ex.get('sent')!r})" if ex else ""))
    if bad:
        return out(False, "request shape", "; ".join(bad), **counts)
    if suite in AGENT_SUITES:
        relay_follow = any(_relay_follow_up(r) for r in relays)
        if not (follow_up_seen(answers) or relay_follow):
            return out(False, "tool execution",
                       "no request followed the agent's first reply: its action did not run, or "
                       "the episode ended before the result went back to the model", **counts)
        return out(True, "", "first reply, action and follow-up request seen", **counts)
    scored, why = scored_evidence(result)
    if not scored:
        return out(False, "scoring", why, **counts)
    return out(True, "", "reply received with the expected shape and scored", **counts)


def format_verdicts(verdicts: Sequence[Mapping[str, Any]], seconds: float,
                    reason_width: int = 140) -> str:
    rows = sorted(verdicts, key=lambda v: (v["verdict"] == "PASS", v["suite"]))
    w = max([len(v["suite"]) for v in rows] + [8])
    npass = sum(1 for v in rows if v["verdict"] == "PASS")
    lines = [f"SMOKE: {npass} PASS, {len(rows) - npass} FAIL of {len(rows)} suites in "
             f"{seconds / 60:.1f} min (no scores: a smoke run measures nothing)",
             f"  {'suite':<{w}}  result  {'stage':<14}  reason",
             f"  {'-' * w}  ------  {'-' * 14}  {'-' * 40}"]
    for v in rows:
        lines.append(f"  {v['suite']:<{w}}  {v['verdict']:<6}  {v.get('stage') or '-':<14}  "
                     f"{_cell(v.get('reason', ''), reason_width)}")
    return "\n".join(lines)


def smoke_deadline_s() -> float:
    """GBENCH_SMOKE_DEADLINE_S (default DEFAULT_DEADLINE_S): the whole smoke run's limit."""
    return _env_float("GBENCH_SMOKE_DEADLINE_S", DEFAULT_DEADLINE_S)


def smoke_mode() -> bool:
    return (os.environ.get("GBENCH_SMOKE") or "").strip().lower() in ("1", "true", "yes", "on")


SMOKE_REASON = ("smoke run (gbench --smoke, GBENCH_SMOKE=1): one sample per suite and agent "
                "episodes ended after their first action; a check of the harness, not a "
                "measurement")


def stamp_smoke(res: Dict[str, Any]) -> Dict[str, Any]:
    """A smoke run's result: smoke=true, whether the suite's scorer scored a sample
    (smoke_scored, read by smoke_verdict), and its accuracy withheld and not
    leaderboard-comparable."""
    res["smoke"] = True
    if not res.get("failed"):
        scored, why = scored_evidence({k: v for k, v in res.items()
                                       if k not in ("smoke_scored", "smoke_scored_reason")})
        res["smoke_scored"], res["smoke_scored_reason"] = scored, why
    if res.get("accuracy") is not None:
        res["accuracy"] = None
        res["accuracy_withheld_reason"] = SMOKE_REASON
    try:
        from .runners.eval_suites.base import add_comparability_reason
        add_comparability_reason(res, SMOKE_REASON)
    except Exception:                                                      # noqa: BLE001
        res["leaderboard_comparable"] = False
    return res


# =========================================================================== CLI entry points
def _base_url(args: Any) -> Optional[str]:
    remote = getattr(args, "remote_endpoint", None)
    if not remote:
        return None
    from .core.config import normalize_endpoint
    return normalize_endpoint(remote)


def run_preflight_cli(args: Any, suites: Sequence[str]) -> Tuple[int, PreflightReport]:
    report = run_preflight(suites, _base_url(args), Path(args.results_dir))
    print("\n" + format_preflight(report) + "\n")
    return (0 if report.ok else 2), report


def _newest_result(directory: Path, suite: str) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    from .pool import suite_result_files
    best: Tuple[float, Optional[Path]] = (-1.0, None)
    for p in suite_result_files(directory, suite):
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m > best[0]:
            best = (m, p)
    if best[1] is None:
        return None, None
    try:
        return json.loads(best[1].read_text(encoding="utf-8")), best[1]
    except (OSError, ValueError):
        return None, best[1]


def _log_tail(path: Path) -> str:
    try:
        lines = [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines()
                 if l.strip()]
    except OSError:
        return ""
    return lines[-1].strip()[:300] if lines else ""


def run_smoke_cli(args: Any, argv: Sequence[str], parser: Any, suites: Sequence[str]) -> int:
    """`gbench --smoke`: preflight, the pool with --eval-limit 1 and GBENCH_SMOKE=1, the verdicts."""
    from . import pool as pool_mod
    from .utils.log_manager import create_unique_dir, write_redacted_json
    t0 = time.monotonic()
    base_url = _base_url(args)
    root = create_unique_dir(Path(args.results_dir), f"smoke-{datetime.now():%m-%d-%H%M%S}")
    report = run_preflight(suites, base_url, root)
    print("\n" + format_preflight(report) + "\n")
    write_redacted_json(root / "preflight.json", report.record())
    pre = report.suites
    verdicts: List[Dict[str, Any]] = []
    if report.global_failed:
        logger.error("smoke: a global check failed (%s); no suite is started",
                     ", ".join(c.name for c in report.global_failed))
        why = "; ".join(f"{c.name}: {c.reason}" for c in report.global_failed)
        verdicts = [{"suite": s, "verdict": "FAIL", "stage": "pre-check",
                     "reason": pre[s].reason if not pre[s].ok else f"not started: {why}"}
                    for s in suites]
        return _finish_smoke(root, verdicts, report, None, time.monotonic() - t0)
    runnable = [s for s in suites if pre[s].ok]
    parallel = getattr(args, "parallel_suites", None) or SMOKE_PARALLEL
    plan = pool_mod.build_plan(runnable, parallel, pool_mod.load_durations(pool_mod.SMOKE_KEY))
    base_argv = pool_mod.suite_argv(parser, argv)
    taps: Dict[str, SmokeTap] = {}

    def child_extra(suite: str, d: Path) -> List[str]:
        tap = SmokeTap(base_url, d / "smoke_tap.jsonl", activity_path=d / ".activity").start()
        taps[suite] = tap
        return ["--eval-limit", "1", "--remote-endpoint", tap.url]

    def on_exit(c) -> None:
        tap = taps.get(c.suite)
        if tap is not None:
            tap.stop()

    opts = pool_mod.pool_options(parallel)
    # the deadline counts from the start of the smoke run (preflight included)
    opts.deadline_s = max(0.0, smoke_deadline_s() - (time.monotonic() - t0))
    opts.deadline_desc = (f"the smoke run's {smoke_deadline_s():.0f} s limit "
                          "(GBENCH_SMOKE_DEADLINE_S, preflight included)")
    metrics = pool_mod.metrics_url(base_url)
    gate = pool_mod.AdmissionGate(metrics, opts.max_inflight)
    watch = SmokeWatch(taps, metrics, _env_float("GBENCH_SMOKE_HANG_S", DEFAULT_HANG_S))
    try:
        manifest = pool_mod.run_pool(plan, base_argv, root, opts, gate=gate,
                                     child_env=lambda s, _o: smoke_child_env(s, args),
                                     child_extra=child_extra, on_exit=on_exit, watch=watch)
    finally:
        for tap in taps.values():
            tap.stop()
    clean, lower = pool_mod.duration_updates(manifest, root)
    pool_mod.record_durations(clean, pool_mod.SMOKE_KEY, source=str(root), raise_only=lower)
    thinking = bool(getattr(args, "eval_thinking", False))
    for s in suites:
        if not pre[s].ok:
            verdicts.append(smoke_verdict(s, expected=expected_shape(
                s, thinking=thinking, temperature=args.temperature,
                max_output_tokens=args.max_output_tokens), preflight=pre[s]))
            continue
        rec = (manifest.get("suites") or {}).get(s) or {}
        result, path = _newest_result(root / s, s)
        v = smoke_verdict(
            s, expected=expected_shape(s, thinking=thinking, temperature=args.temperature,
                                       max_output_tokens=args.max_output_tokens),
            preflight=pre[s], result=result, tap=read_tap(root / s / "smoke_tap.jsonl"),
            exit_code=rec.get("exit_code"), log_tail=_log_tail(root / s / "gbench.log"),
            stopped=rec.get("stop_reason"))
        v.update(duration_s=rec.get("duration_s"), log=rec.get("log"),
                 result_file=os.path.relpath(path, root) if path else None)
        if rec.get("not_started"):
            v.update(verdict="FAIL", stage="not started",
                     reason=f"the smoke run was stopped ({rec.get('not_started_by') or 'signal'})"
                            " before this suite started")
        verdicts.append(v)
    return _finish_smoke(root, verdicts, report, manifest, time.monotonic() - t0)


def _finish_smoke(root: Path, verdicts: List[Dict[str, Any]], report: PreflightReport,
                  manifest: Optional[Mapping[str, Any]], seconds: float) -> int:
    from .utils.log_manager import write_redacted_json, write_redacted_text
    table = format_verdicts(verdicts, seconds)
    print("\n" + table + "\n")
    npass = sum(1 for v in verdicts if v["verdict"] == "PASS")
    write_redacted_json(root / "summary.json", {
        "smoke": True, "results_dir": str(root.absolute()),
        "suites_total": len(verdicts), "passed": npass, "failed": len(verdicts) - npass,
        "seconds": round(seconds, 1), "preflight_ok": report.ok,
        "stopped_by_signal": (manifest or {}).get("stopped_by_signal"),
        "deadline_s": (manifest or {}).get("deadline_s"),
        "deadline_reached": bool((manifest or {}).get("deadline_reached")),
        "verdicts": verdicts})
    write_redacted_text(root / "smoke_table.txt", table + "\n")
    print(f"Smoke results: {root} (summary.json, smoke_table.txt; each suite's log and result "
          "under its own directory)")
    if manifest and manifest.get("stopped_by_signal"):
        return 128 + int(manifest["stopped_by_signal"])
    return 0 if npass == len(verdicts) else 2
