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

"""Run an evals-only sweep as a pool of suite processes (`gbench --parallel-suites N ...`).

A serial sweep runs one suite at a time and waits for each suite's slowest reply, so the served
model idles through every suite's tail: the 2026-10-09 --eval-limit 5 sweep of the 84 built-in
suites took 783 min serially, and the three-lane run of 2026-10-10 still had one lane at 0 of 28
suites after 3 h 29 min. The pool runs each selected suite as its own
`gbench --evals-only --evals <suite> <the run's other flags>` process (its own interpreter, so
no suite shares the working directory, os.environ or a crash with another), up to N at once:

* longest first, from a table of earlier wall times (pool_durations.json in this package, seeded
  from the 2026-10-09 serial run and the 2026-10-10 lanes; each pool run adds its own durations to
  an overlay file under the gbench cache, keyed by --eval-limit, smoke runs under their own key:
  a clean result (status success, exit 0, not partial, no trial errors) replaces the suite's entry,
  any other result only raises it, since a suite that failed or was stopped after N s says only
  that it takes at least N s); a suite without an entry counts as the median;
* never two suites of one family at once (FAMILIES: shared files, ports, images or the same
  server-heavy protocol), except the SWE family and the long-context suites, 2 at a time;
* the suites that call the Gemini search tool share its rate: with k of them allowed at once each
  child gets GBENCH_SEARCH_MAX_RPS = (the run's value, default 8) / k (search_tool.py's limiter
  is per process);
* an admission gate: a suite starts only while the served vLLM's num_requests_running +
  num_requests_waiting (GET <endpoint root>/metrics) is below GBENCH_POOL_MAX_INFLIGHT (default
  48); a suite always starts when none of the pool's is running, so external load cannot stall
  the pool forever;
* a stall alarm (alert only, never a kill): a WARNING when a running suite has written no file
  under its directory (its log, traces, its relays' activity file) for GBENCH_POOL_STALL_S
  (default 600 s), repeated every such interval while it stays silent;
* SIGINT/SIGTERM/SIGHUP to the pool send SIGTERM once to every child's process group (each
  child is a session leader), no further suite starts, and a second signal kills the groups; a
  pool `deadline_s` (smoke runs only, GBENCH_SMOKE_DEADLINE_S) or a `watch` callback that names a
  reason (the smoke tap's hang detection) does the same for the children it concerns, with
  SIGKILL KILL_GRACE_S later; the containers each child started are removed when it ends: the
  `docker run --name` processes seen in its process group, and gbench-<suite>-* names that were
  not present when the child started (2026-10-11 review: a name prefix alone also matched an
  unrelated run's `gbench-gsm8k-...` container and putnam_formal's `gbench-putnam-<hex>`
  checkers); any other container created while the pool ran is listed at the end (not removed);
* --skip-existing: each child also searches the run's own --results-dir (GBENCH_REUSE_ROOT), as a
  serial run does, so a restarted pool reuses the results an earlier pool finished;
* one merged summary.json and eval_summary.csv over every suite (merge.union_runs), with each
  suite's eval_*.json linked into the pool directory unchanged.

Every suite records the concurrency it ran under (result `pool`: the pool size, the suites running
when it started, its search rate) and pool.json records the most suites that ran beside it.

Layout: <--results-dir>/pool-<timestamp>/<suite>/ is each child's --results-dir (gbench writes its
<timestamp> run directory inside it; the child's output is <suite>/gbench.log), and the pool
directory holds pool.json, the merged summary.json / eval_summary.csv / metadata.json and links to
each suite's eval_*.json.
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import stat
import statistics
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

logger = logging.getLogger("gbench.pool")

SWE_FAMILY = frozenset({"swe_bench_live", "swe_bench_pro", "swe_bench_multilingual",
                        "multi_swe_bench", "copilot_bench_swe"})
LONG_CONTEXT_SUITES = frozenset({"mrcr", "beam_128k", "aa_lcr", "ruler", "longbench_v2_code"})

#: (family, members, how many of them may run at once). A suite is in at most one family.
FAMILIES: Tuple[Tuple[str, FrozenSet[str], int], ...] = (
    # instance images, the docker socket and hours-long agent rollouts
    ("swe", SWE_FAMILY, 2),
    # one BFCL harness image and project root
    ("bfcl", frozenset({"bfcl", "bfcl_live", "bfcl_v4_agentic", "gorilla_apibench"}), 1),
    # the MathArena grader image and 65K-token replies in a row
    ("matharena", frozenset({"aime", "hmmt", "amc_aime", "imo_answer_bench"}), 1),
    # one tau2-bench image and cache directory
    ("tau", frozenset({"tau2", "tau3"}), 1),
    ("mmlu", frozenset({"mmlu", "mmlu_pro", "mmlu_redux", "multilingual_mmlu"}), 1),
    ("gpqa", frozenset({"gpqa", "gpqa_diamond"}), 1),
    ("putnam", frozenset({"putnam", "putnam_formal"}), 1),
    # gateway/relay ports 18789-18795 (PROVEN conflict, 2026-10-10)
    ("gaia_ports", frozenset({"gaia", "gaia2", "wildclawbench"}), 1),
    # code judges under per-test time limits on this host's CPUs
    ("code_judge", frozenset({"lcb", "codeforces", "ojbench"}), 1),
    ("vqa", frozenset({"docvqa", "infographicvqa", "chartqa", "textvqa", "charxiv"}), 1),
    # MCP server containers
    ("mcp", frozenset({"mcp_bench", "mcp_atlas"}), 1),
    # one gbench-mobilegym image and Playwright Chromium worker pool
    ("mobilegym", frozenset({"mobilegym", "mobilegym_pp"}), 1),
    # 128K-token prompts: the served KV cache holds about 26 full-window sequences
    ("long_context", LONG_CONTEXT_SUITES, 2),
)

#: Suites whose runs call the Gemini search tool (search_tool.py, its token bucket
#: GBENCH_SEARCH_MAX_RPS, default 8, is per process).
SEARCH_SUITES = frozenset({"browsecomp", "deepsearch_qa", "gaia", "bfcl", "bfcl_v4_agentic",
                           "mcp_atlas", "wildclawbench"})
DEFAULT_SEARCH_MAX_RPS = 8.0
#: The most search suites the pool runs at once (each then gets 1/k of the rate).
SEARCH_MAX_CONCURRENT = 4

DEFAULT_MAX_INFLIGHT = 48
DEFAULT_STALL_S = 600.0
#: How often the stall alarm and the container scan look at the running children.
SCAN_INTERVAL_S = 30.0
#: How long the admission gate trusts one /metrics reading.
METRICS_TTL_S = 5.0
#: Seconds between the SIGTERM a deadline or a watch reason sends a child and its SIGKILL.
KILL_GRACE_S = 60.0

DURATIONS_FILE = Path(__file__).with_name("pool_durations.json")

#: The command that runs one suite's gbench (its arguments follow). -P: the working directory is
#: not put on sys.path (2026-10-12: started from /projects/gemma4-vllm, `python -m gbench` imported
#: the repository directory gbench/ as a namespace package and all 84 suites failed on import; from
#: /tmp a stray inspect.py replaced the standard library's).
GBENCH_CMD: List[str] = [sys.executable, "-P", "-m", "gbench"]

#: gbench options the pool sets itself for every child (stripped from the run's arguments).
POOL_OWNED_DESTS = ("parallel_suites", "smoke", "preflight", "evals", "results_dir", "evals_only",
                    "dry_run")


# --------------------------------------------------------------------------- durations
def _overlay_path() -> Path:
    raw = (os.environ.get("GBENCH_POOL_DURATIONS_FILE") or "").strip()
    if raw:
        return Path(raw).expanduser()
    from .runners.eval_suites.base import gbench_cache_dir
    return Path(gbench_cache_dir()) / "pool_durations.json"


#: The overlay key of smoke-run durations (episodes stopped after one action): never read by a
#: real run's ordering.
SMOKE_KEY = "smoke"


def _limit_key(eval_limit: Any) -> str:
    if isinstance(eval_limit, str):
        return eval_limit
    return f"limit={eval_limit}" if eval_limit else "limit=full"


def load_durations(eval_limit: Any = None,
                   seed_path: Optional[Path] = None,
                   overlay_path: Optional[Path] = None) -> Dict[str, float]:
    """{suite: seconds}: the package seed table, overridden by the overlay's entries for this
    --eval-limit (record_durations). Unreadable files count as empty."""
    out: Dict[str, float] = {}
    for path, key in ((seed_path or DURATIONS_FILE, None),
                      (overlay_path or _overlay_path(), _limit_key(eval_limit))):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        table = data.get("suites") if key is None else (data.get(key) or {})
        for name, entry in (table or {}).items():
            dur = entry.get("duration_s") if isinstance(entry, dict) else entry
            if isinstance(dur, (int, float)) and not isinstance(dur, bool) and dur >= 0:
                out[name] = float(dur)
    return out


def record_durations(durations: Mapping[str, float], eval_limit: Any,
                     overlay_path: Optional[Path] = None, source: str = "",
                     raise_only: Mapping[str, float] = (),
                     seed_path: Optional[Path] = None) -> Optional[Path]:
    """Add `durations` (suites with a clean result) to the overlay under this --eval-limit (or
    SMOKE_KEY), and `raise_only` (suites whose result failed, was partial or was stopped) only
    where it is longer than the suite's current estimate (seed or overlay): a run cut short says
    only that the suite takes at least that long (2026-10-11 review: a swe_bench_live record
    failed after 9 s would otherwise have put the longest suite last). Never raises: a cache that
    cannot be written only costs the next run its ordering."""
    path = overlay_path or _overlay_path()
    try:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        raise_only = dict(raise_only or {})
        current = (load_durations(eval_limit, seed_path=seed_path, overlay_path=path)
                   if raise_only else {})
        table = data.setdefault(_limit_key(eval_limit), {})
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for name, dur in durations.items():
            table[name] = {"duration_s": round(float(dur), 1), "updated": stamp,
                           "source": source}
        for name, dur in raise_only.items():
            if name in durations or float(dur) <= current.get(name, -1.0):
                continue
            table[name] = {"duration_s": round(float(dur), 1), "updated": stamp,
                           "source": source, "lower_bound": True}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)
        return path
    except Exception as e:                                                 # noqa: BLE001
        logger.warning("could not update the pool's duration table %s: %s", path, e)
        return None


# --------------------------------------------------------------------------- scheduling
@dataclass
class Plan:
    """The pool's schedule: `order` is longest first; caps per family (FAMILIES plus the search
    family sized from the selection) and each search suite's GBENCH_SEARCH_MAX_RPS."""
    order: List[str]
    estimates: Dict[str, float]
    parallel: int
    families: List[Tuple[str, FrozenSet[str], int]] = field(default_factory=list)
    search_rps: Dict[str, float] = field(default_factory=dict)


def order_longest_first(suites: Sequence[str], durations: Mapping[str, float]
                        ) -> Tuple[List[str], Dict[str, float]]:
    """`suites` longest estimated duration first (ties keep the given order); a suite without a
    duration counts as the median of the known ones (60 s when none is known)."""
    known = [durations[s] for s in suites if s in durations]
    default = statistics.median(known) if known else 60.0
    est = {s: float(durations.get(s, default)) for s in suites}
    pos = {s: i for i, s in enumerate(suites)}
    return sorted(suites, key=lambda s: (-est[s], pos[s])), est


def build_plan(suites: Sequence[str], parallel: int, durations: Mapping[str, float],
               env: Optional[Mapping[str, str]] = None) -> Plan:
    if parallel < 1:
        raise ValueError("--parallel-suites must be at least 1")
    env = os.environ if env is None else env
    order, est = order_longest_first(list(dict.fromkeys(suites)), durations)
    families = [(n, m, c) for n, m, c in FAMILIES]
    search = [s for s in order if s in SEARCH_SUITES]
    rps: Dict[str, float] = {}
    if search:
        k = max(1, min(len(search), parallel, SEARCH_MAX_CONCURRENT))
        families.append(("search", frozenset(search), k))
        try:
            base = float((env.get("GBENCH_SEARCH_MAX_RPS") or "").strip() or DEFAULT_SEARCH_MAX_RPS)
        except ValueError:
            base = DEFAULT_SEARCH_MAX_RPS
        rps = {s: round(base / k, 3) for s in search}
    return Plan(order, est, parallel, families, rps)


def blocked_by(suite: str, running: Iterable[str], plan: Plan) -> Optional[str]:
    """Why `suite` may not start next to `running` (a family at its cap), else None."""
    running = set(running)
    for name, members, cap in plan.families:
        if suite in members and len(running & members) >= cap:
            return f"family {name} at its cap of {cap}"
    return None


def next_suite(pending: Sequence[str], running: Iterable[str], plan: Plan) -> Optional[str]:
    """The first suite of `pending` (in plan order) that may start next to `running`."""
    running = set(running)
    if len(running) >= plan.parallel:
        return None
    for s in pending:
        if blocked_by(s, running, plan) is None:
            return s
    return None


# --------------------------------------------------------------------------- admission gate
_METRIC_RE = re.compile(r"^vllm:num_requests_(running|waiting)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$")


def parse_inflight(metrics_text: str) -> Optional[int]:
    """num_requests_running + num_requests_waiting summed over every engine/model label set in a
    vLLM /metrics page; None when the page has neither."""
    total, seen = 0.0, False
    for line in (metrics_text or "").splitlines():
        m = _METRIC_RE.match(line.strip())
        if m:
            seen = True
            try:
                total += float(m.group(2))
            except ValueError:
                pass
    return int(round(total)) if seen else None


def metrics_url(endpoint: str) -> str:
    base = (endpoint or "").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base + "/metrics"


def fetch_metrics(url: str, timeout: float = 5.0) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:            # noqa: S310
            return r.read().decode("utf-8", "replace")
    except Exception:                                                      # noqa: BLE001
        return None


class AdmissionGate:
    """open() says whether a new suite may start: the served model's in-flight requests are below
    `max_inflight`. An unreadable /metrics page opens the gate (logged once)."""

    def __init__(self, url: Optional[str], max_inflight: int,
                 fetch: Callable[[str], Optional[str]] = fetch_metrics,
                 clock: Callable[[], float] = time.monotonic):
        self.url, self.max_inflight, self._fetch, self._clock = url, max_inflight, fetch, clock
        self._cached: Optional[Tuple[float, Optional[int]]] = None
        self._warned = False
        self.last_inflight: Optional[int] = None

    def inflight(self) -> Optional[int]:
        now = self._clock()
        if self._cached is not None and now - self._cached[0] < METRICS_TTL_S:
            return self._cached[1]
        value = parse_inflight(self._fetch(self.url) or "") if self.url else None
        self._cached = (now, value)
        self.last_inflight = value
        return value

    def open(self) -> bool:
        if not self.url or self.max_inflight <= 0:
            return True
        n = self.inflight()
        if n is None:
            if not self._warned:
                logger.warning("[pool] %s gave no vLLM request counts; the admission gate is "
                               "open (suites start as slots free up)", self.url)
                self._warned = True
            return True
        return n < self.max_inflight


# --------------------------------------------------------------------------- child arguments
def strip_dests(parser: Any, argv: Sequence[str], dests: Iterable[str]) -> List[str]:
    """`argv` without the options whose argparse dest is in `dests` (and their values), matched
    as argparse matches them: full option strings, `--opt=value`, and unique abbreviations."""
    dests = set(dests)
    by_string = {s: a for a in parser._actions for s in a.option_strings}   # noqa: SLF001
    longs = [s for s in by_string if s.startswith("--")]

    def _action(tok: str):
        head = tok.split("=", 1)[0]
        if head in by_string:
            return by_string[head]
        if head.startswith("--"):
            hits = {by_string[s] for s in longs if s.startswith(head)}
            if len(hits) == 1:
                return hits.pop()
        return None

    out: List[str] = []
    i = 0
    toks = list(argv)
    while i < len(toks):
        tok = toks[i]
        act = _action(tok) if tok.startswith("-") and tok != "-" else None
        if act is None or act.dest not in dests:
            out.append(tok)
            i += 1
            continue
        i += 1
        if "=" in tok:
            continue
        nargs = act.nargs
        if nargs == 0:
            continue
        if nargs in (None, 1):
            i += 1
        elif nargs == "?":
            if i < len(toks) and not toks[i].startswith("-"):
                i += 1
        elif nargs in ("+", "*"):
            while i < len(toks) and not toks[i].startswith("-"):
                i += 1
        elif isinstance(nargs, int):
            i += nargs
    return out


def suite_argv(parser: Any, argv: Sequence[str]) -> Callable[[str], List[str]]:
    """suite -> the run's arguments for that suite's process: without POOL_OWNED_DESTS, and
    without --eval-custom-jsonl except for custom_jsonl (gbench adds the custom_jsonl suite to any
    run that has the flag)."""
    plain = strip_dests(parser, argv, POOL_OWNED_DESTS)
    no_custom = strip_dests(parser, plain, ("eval_custom_jsonl",))
    return lambda suite: plain if suite == "custom_jsonl" else no_custom


def child_argv(base_argv: Sequence[str], suite: str, suite_dir: Path,
               extra: Sequence[str] = ()) -> List[str]:
    """One child's gbench arguments: the run's own (POOL_OWNED_DESTS already stripped), then
    --evals-only --evals <suite> --results-dir <suite_dir> and `extra` (argparse keeps the last
    value of a repeated option)."""
    return [*base_argv, "--evals-only", "--evals", suite, "--results-dir", str(suite_dir),
            *extra]


# --------------------------------------------------------------------------- containers
def _docker_run_argvs(pgid: int) -> List[List[str]]:
    """The command line of every `docker run` process in process group `pgid` (from /proc)."""
    out: List[List[str]] = []
    try:
        pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except OSError:
        return out
    for pid in pids:
        try:
            if os.getpgid(pid) != pgid:
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = [a.decode("utf-8", "replace") for a in f.read().split(b"\0") if a]
        except (OSError, ProcessLookupError):
            continue
        if argv and os.path.basename(argv[0]) == "docker" and "run" in argv[1:3]:
            out.append(argv)
    return out


def _docker_run_names(argvs: Iterable[Sequence[str]]) -> Set[str]:
    """The --name of each `docker run` command line of `argvs`."""
    names: Set[str] = set()
    for argv in argvs:
        for j, a in enumerate(argv):
            if a == "--name" and j + 1 < len(argv):
                names.add(argv[j + 1])
            elif a.startswith("--name="):
                names.add(a.split("=", 1)[1])
    return names


def _docker_names_in_group(pgid: int) -> Set[str]:
    """The --name of every `docker run` process in process group `pgid` (from /proc)."""
    return _docker_run_names(_docker_run_argvs(pgid))


def scratch_mounts(argvs: Iterable[Sequence[str]], tmp_root: Optional[str] = None) -> Set[str]:
    """The bind-mount sources (`-v SRC:DST[:opts]`, `--volume`, `--mount type=bind,source=SRC`) of
    the `docker run` command lines `argvs` that are a directory directly under the system temp
    directory (`tmp_root`, default tempfile.gettempdir()): a tempfile.mkdtemp directory of the
    suite's own, such as terminal_bench's /tmp/gbench_tb_* jobs directory, which its harness
    container mounts and the task containers Harbor starts through the docker socket mount parts
    of: mode 0700 and owned by this user, as mkdtemp makes it. Shared mounts (the docker socket,
    caches under $HOME, /tmp/.X11-unix) are not."""
    import tempfile
    root = os.path.realpath(tmp_root or tempfile.gettempdir())
    out: Set[str] = set()
    for argv in argvs:
        argv = list(argv)
        for j, a in enumerate(argv):
            src = None
            if a in ("-v", "--volume") and j + 1 < len(argv):
                src = argv[j + 1].split(":", 1)[0]
            elif a.startswith("--volume="):
                src = a.split("=", 1)[1].split(":", 1)[0]
            elif (a == "--mount" and j + 1 < len(argv)) or a.startswith("--mount="):
                spec = argv[j + 1] if a == "--mount" else a.split("=", 1)[1]
                kv = dict(p.split("=", 1) for p in spec.split(",") if "=" in p)
                if kv.get("type", "volume") == "bind":
                    src = kv.get("source") or kv.get("src")
            if not src or not os.path.isabs(src):
                continue
            src = os.path.realpath(src)
            if os.path.dirname(src) != root:
                continue
            try:
                st = os.stat(src)
            except OSError:
                continue
            # mkdtemp's kind: a directory of this user's, mode 0700 (not /tmp/.X11-unix and such)
            if (stat.S_ISDIR(st.st_mode) and stat.S_IMODE(st.st_mode) == 0o700
                    and st.st_uid == os.getuid()):
                out.add(src)
    return out


def containers_mounting(names: Iterable[str], roots: Iterable[str]
                        ) -> Tuple[Set[str], Set[str]]:
    """The containers among `names` with a bind mount whose source is one of `roots` or under
    one, and the compose projects (label com.docker.compose.project) of those containers."""
    names = sorted(names)
    roots = [r.rstrip("/") for r in roots if r and r.rstrip("/")]
    if not names or not roots:
        return set(), set()
    # a container removed meanwhile makes docker inspect exit 1; the others are still printed
    _code, out = _docker(["inspect", "--format",
                          "{{json .Name}}\t{{json .Mounts}}\t{{json .Config.Labels}}", *names])
    found: Set[str] = set()
    projects: Set[str] = set()
    for line in out.splitlines():
        try:
            name, mounts, labels = (json.loads(x) for x in line.split("\t"))
        except ValueError:
            continue
        for m in mounts or []:
            src = (m or {}).get("Source") or ""
            if (m or {}).get("Type") == "bind" and any(src == r or src.startswith(r + "/")
                                                       for r in roots):
                found.add(str(name).lstrip("/"))
                project = (labels or {}).get("com.docker.compose.project")
                if project:
                    projects.add(project)
                break
    return found, projects


def _docker(args: Sequence[str], timeout: float = 60.0) -> Tuple[int, str]:
    try:
        r = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout or ""
    except Exception as e:                                                 # noqa: BLE001
        return -1, str(e)


def container_names() -> Set[str]:
    code, out = _docker(["ps", "-a", "--format", "{{.Names}}"])
    return {n.strip() for n in out.splitlines() if n.strip()} if code == 0 else set()


def suite_name_prefixes(suite: str) -> Tuple[str, ...]:
    dash = suite.replace("_", "-")
    return tuple(dict.fromkeys((f"gbench-{dash}-", f"gbench-{suite}-", f"gbench_{suite}_")))


def attribute_names(names: Iterable[str], suite: str, all_suites: Iterable[str]) -> Set[str]:
    """The names among `names` that belong to `suite` by prefix (suite_name_prefixes), with the
    longest matching suite winning (gaia vs gaia2, bfcl vs bfcl_live)."""
    others = [s for s in all_suites if s != suite and len(s) > len(suite)]
    out = set()
    for n in names:
        if not n.startswith(suite_name_prefixes(suite)):
            continue
        if any(n.startswith(suite_name_prefixes(o)) for o in others):
            continue
        out.add(n)
    return out


# --------------------------------------------------------------------------- children
@dataclass
class Child:
    suite: str
    proc: Any
    directory: Path
    log_path: Path
    started: float
    started_at: str
    argv: List[str]
    log_handle: Any = None
    running_at_start: List[str] = field(default_factory=list)
    max_alongside: int = 0
    last_activity: float = 0.0
    last_alert: float = 0.0
    stall_alerts: int = 0
    docker_names: Set[str] = field(default_factory=set)
    exit_code: Optional[int] = None
    duration_s: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    #: container names present when the child started (never reaped by name prefix)
    containers_before: Set[str] = field(default_factory=set)
    #: the suite's own temp directories its `docker run` processes mounted (scratch_mounts): a
    #: container created after the child started that bind-mounts one of them is the child's
    scratch_dirs: Set[str] = field(default_factory=set)
    #: why the pool stopped this child (deadline, watch), and when its SIGTERM went out
    stop_reason: Optional[str] = None
    term_sent: Optional[float] = None
    kill_sent: bool = False


def latest_mtime(root: Path, cap: int = 20000) -> float:
    """The newest modification time of `root` and the files under it (at most `cap` entries)."""
    best = 0.0
    stack, seen = [root], 0
    while stack and seen < cap:
        d = stack.pop()
        try:
            st = d.stat()
            best = max(best, st.st_mtime)
            with os.scandir(d) as it:
                for e in it:
                    seen += 1
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                        else:
                            best = max(best, e.stat(follow_symlinks=False).st_mtime)
                    except OSError:
                        continue
        except OSError:
            continue
    return best


class _Signals:
    """The first SIGINT/SIGTERM/SIGHUP sends SIGTERM once to every running child's process group
    (the gbench CLI turns it into a clean stop) and stops new starts; a second one SIGKILLs the
    groups. A SIGHUP the pool was started ignoring
    (`nohup`) stays ignored."""

    def __init__(self, children: Callable[[], List[Child]]):
        self.stop: Optional[int] = None
        self.killed = False
        self._children = children
        self._old: Dict[int, Any] = {}

    def _send(self, sig: int) -> None:
        for c in self._children():
            if c.proc.poll() is None:
                try:
                    os.killpg(c.proc.pid, sig)
                except (OSError, ProcessLookupError):
                    try:
                        c.proc.send_signal(sig)
                    except OSError:
                        pass

    def _handler(self, signum, _frame) -> None:
        if self.stop is None:
            self.stop = signum
            logger.warning("[pool] signal %d: sending SIGTERM to every running suite's process "
                           "group (gbench writes the stopped suite's failed record); starting "
                           "no more suites", signum)
            self._send(signal.SIGTERM)
        elif not self.killed:
            self.killed = True
            logger.warning("[pool] second signal %d: killing every running suite's process "
                           "group", signum)
            self._send(signal.SIGKILL)

    def __enter__(self) -> "_Signals":
        for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)):
            if sig is None:
                continue
            try:
                if sig == getattr(signal, "SIGHUP", None) and \
                        signal.getsignal(sig) == signal.SIG_IGN:
                    continue
                self._old[sig] = signal.signal(sig, self._handler)
            except (ValueError, OSError):
                pass
        return self

    def __exit__(self, *exc) -> None:
        for sig, old in self._old.items():
            try:
                signal.signal(sig, old)
            except (ValueError, OSError):
                pass


@dataclass
class PoolOptions:
    parallel: int
    max_inflight: int = DEFAULT_MAX_INFLIGHT
    stall_s: float = DEFAULT_STALL_S
    poll_s: float = 2.0
    scan_s: float = SCAN_INTERVAL_S
    reap_containers: bool = True
    #: seconds after which every running child is stopped and no further suite starts (None:
    #: no deadline; set by --smoke only, where nothing is measured)
    deadline_s: Optional[float] = None
    #: how the deadline is named in its stop reason (default: "the run's <deadline_s> s limit")
    deadline_desc: Optional[str] = None


def pool_options(parallel: int, env: Optional[Mapping[str, str]] = None) -> PoolOptions:
    env = os.environ if env is None else env

    def _num(name: str, default: float) -> float:
        raw = (env.get(name) or "").strip()
        try:
            return float(raw) if raw else default
        except ValueError:
            logger.warning("%s=%r is not a number; using %s", name, raw, default)
            return default
    return PoolOptions(parallel=parallel,
                       max_inflight=int(_num("GBENCH_POOL_MAX_INFLIGHT", DEFAULT_MAX_INFLIGHT)),
                       stall_s=_num("GBENCH_POOL_STALL_S", DEFAULT_STALL_S))


def run_pool(plan: Plan, base_argv: Any, root: Path, opts: PoolOptions, *,
             gate: Optional[AdmissionGate] = None,
             child_env: Optional[Callable[[str, List[str]], Dict[str, str]]] = None,
             child_extra: Optional[Callable[[str, Path], List[str]]] = None,
             on_start: Optional[Callable[[Child], None]] = None,
             on_exit: Optional[Callable[[Child], None]] = None,
             spawn: Optional[Callable[..., Any]] = None,
             watch: Optional[Callable[[Child], Optional[str]]] = None,
             clock: Callable[[], float] = time.monotonic,
             sleep: Callable[[float], None] = time.sleep,
             install_signals: bool = True) -> Dict[str, Any]:
    """Run every suite of `plan` as a child gbench process under `root` and return the manifest
    (also written to root/pool.json). `base_argv` is the run's arguments without
    POOL_OWNED_DESTS, or a callable suite -> those arguments. `child_env(suite, running)` gives a
    child's extra
    environment, `child_extra(suite, dir)` extra arguments; `on_start` / `on_exit` see each child
    (smoke mode starts and stops its tap there); `watch(child)`, called every poll for each
    running child, returns a reason to stop it (smoke hang detection) or None. `spawn` replaces
    subprocess.Popen (tests)."""
    root.mkdir(parents=True, exist_ok=True)
    spawn = spawn or subprocess.Popen
    gate = gate or AdmissionGate(None, 0)
    pending: List[str] = list(plan.order)
    running: List[Child] = []
    done: List[Child] = []
    baseline = container_names() if opts.reap_containers else set()
    started_wall = datetime.now(timezone.utc)
    started_mono = clock()
    deadline_hit = False
    last_scan = clock()
    gate_note = False
    sig = _Signals(lambda: list(running))
    ctx = sig if install_signals else _NullCtx()
    with ctx:
        while pending or running:
            for c in list(running):
                code = c.proc.poll()
                if code is None:
                    continue
                running.remove(c)
                c.exit_code = code
                c.duration_s = round(clock() - c.started, 1)
                if c.log_handle is not None:
                    try:
                        c.log_handle.close()
                    except OSError:
                        pass
                if opts.reap_containers:
                    _reap(c, plan.order, baseline)
                if on_exit is not None:
                    try:
                        on_exit(c)
                    except Exception as e:                                 # noqa: BLE001
                        logger.error("[pool] %s: finishing it failed: %s", c.suite, e)
                done.append(c)
                logger.info("[pool] %s finished: exit %s after %.0f s (%d running, %d waiting)",
                            c.suite, code, c.duration_s, len(running), len(pending))
            now = clock()
            if (opts.deadline_s is not None and not deadline_hit
                    and now - started_mono >= opts.deadline_s):
                deadline_hit = True
                why = ("deadline: " + (opts.deadline_desc or
                                       f"the run's {opts.deadline_s:.0f} s limit")
                       + " was reached")
                logger.error("[pool] %s; stopping %d running suite(s), starting no more",
                             why, len(running))
                for c in running:
                    _stop_child(c, why, now)
            for c in running:
                if watch is not None and c.stop_reason is None:
                    try:
                        why = watch(c)
                    except Exception as e:                                 # noqa: BLE001
                        logger.error("[pool] %s: its watch failed: %s", c.suite, e)
                        why = None
                    if why:
                        logger.error("[pool] %s: %s; stopping it", c.suite, why)
                        _stop_child(c, why, now)
                if (c.term_sent is not None and not c.kill_sent
                        and now - c.term_sent >= KILL_GRACE_S):
                    c.kill_sent = True
                    _signal_group(c, signal.SIGKILL)
            stopping = (sig.stop is not None if install_signals else False) or deadline_hit
            while not stopping and pending and len(running) < opts.parallel:
                s = next_suite(pending, [c.suite for c in running], plan)
                if s is None:
                    break
                if running and not gate.open():
                    if not gate_note:
                        logger.info("[pool] admission gate closed: %s requests in flight at the "
                                    "server (GBENCH_POOL_MAX_INFLIGHT=%d); %s waits",
                                    gate.last_inflight, opts.max_inflight, s)
                        gate_note = True
                    break
                gate_note = False
                pending.remove(s)
                before = container_names() if opts.reap_containers else set()
                running.append(_start(s, plan, base_argv, root, running, child_env, child_extra,
                                      spawn, clock))
                c = running[-1]
                c.containers_before = before
                for other in running:
                    other.max_alongside = max(other.max_alongside, len(running) - 1)
                if on_start is not None:
                    on_start(c)
            if stopping and pending:
                cause = ("the deadline" if deadline_hit and not (install_signals and sig.stop)
                         else f"signal {sig.stop}")
                logger.warning("[pool] stopped by %s: %d suite(s) not started: %s",
                               cause, len(pending), ", ".join(pending))
                for s in pending:
                    done.append(Child(s, None, root / s, root / s / "gbench.log", clock(), "",
                                      [], exit_code=None,
                                      extra={"not_started": True, "not_started_by": cause}))
                pending = []
            now = clock()
            if running and now - last_scan >= opts.scan_s:
                last_scan = now
                for c in running:
                    _scan(c, now, opts)
            if running:
                sleep(opts.poll_s)
    leftovers = sorted(container_names() - baseline) if opts.reap_containers else []
    if leftovers:
        # not removed: a container another run started meanwhile looks the same from here
        logger.warning("[pool] %d container(s) created while the pool ran are still present: "
                       "%s (if they are this run's, remove them with: docker rm -f %s)",
                       len(leftovers), ", ".join(leftovers[:20]), " ".join(leftovers[:20]))
    manifest = {
        "created": started_wall.isoformat(timespec="seconds"),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parallel_suites": opts.parallel,
        "max_inflight": opts.max_inflight,
        "stall_alarm_s": opts.stall_s,
        "families": {n: {"members": sorted(m), "max_concurrent": c}
                     for n, m, c in plan.families},
        "order": plan.order,
        "estimates_s": {s: round(v, 1) for s, v in plan.estimates.items()},
        "stopped_by_signal": sig.stop if install_signals else None,
        "deadline_s": opts.deadline_s,
        "deadline_reached": deadline_hit,
        "suites": {c.suite: _child_record(c, root, plan) for c in done},
    }
    try:
        (root / "pool.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n",
                                        encoding="utf-8")
    except OSError as e:
        logger.error("[pool] could not write %s: %s", root / "pool.json", e)
    return manifest


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


def _start(suite: str, plan: Plan, base_argv: Sequence[str], root: Path, running: List[Child],
           child_env, child_extra, spawn, clock) -> Child:
    d = root / suite
    d.mkdir(parents=True, exist_ok=True)
    extra = list(child_extra(suite, d)) if child_extra else []
    own = base_argv(suite) if callable(base_argv) else base_argv
    argv = [*GBENCH_CMD, *child_argv(own, suite, d, extra)]
    others = [c.suite for c in running]
    env = dict(os.environ)
    record = {"parallel_suites": plan.parallel, "running_at_start": others,
              "pool_dir": str(root)}
    if suite in plan.search_rps:
        env["GBENCH_SEARCH_MAX_RPS"] = str(plan.search_rps[suite])
        record["search_max_rps"] = plan.search_rps[suite]
    env["GBENCH_POOL_RECORD"] = json.dumps(record, sort_keys=True)
    # A relay or harness that sees traffic touches this file (the stall alarm reads its time).
    env["GBENCH_POOL_ACTIVITY_FILE"] = str(d / ".activity")
    if child_env:
        env.update(child_env(suite, others))
    log_path = d / "gbench.log"
    log = open(log_path, "ab")
    proc = spawn(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    now = clock()
    logger.info("[pool] %s started (pid %s, %d running beside it): log %s", suite,
                getattr(proc, "pid", "?"), len(others), log_path)
    return Child(suite, proc, d, log_path, now,
                 datetime.now(timezone.utc).isoformat(timespec="seconds"), argv, log,
                 running_at_start=others, max_alongside=len(others), last_activity=time.time(),
                 extra={"search_max_rps": plan.search_rps.get(suite)})


def _signal_group(c: Child, sig: int) -> None:
    try:
        os.killpg(c.proc.pid, sig)
    except (OSError, ProcessLookupError, AttributeError, TypeError):
        try:
            c.proc.send_signal(sig)
        except (OSError, AttributeError):
            pass


def _stop_child(c: Child, why: str, now: float) -> None:
    """SIGTERM child `c`'s process group once (the gbench CLI writes the stopped suite's failed
    record), remembering why; run_pool sends SIGKILL KILL_GRACE_S later if it is still there."""
    if c.stop_reason is not None:
        return
    c.stop_reason, c.term_sent = why, now
    _signal_group(c, signal.SIGTERM)


def _scan(c: Child, now: float, opts: PoolOptions) -> None:
    """The stall alarm and the container-name scan for one running child."""
    if opts.reap_containers and getattr(c.proc, "pid", None):
        argvs = _docker_run_argvs(c.proc.pid)
        c.docker_names |= _docker_run_names(argvs)
        c.scratch_dirs |= scratch_mounts(argvs)
    if opts.stall_s <= 0:
        return
    mtime = latest_mtime(c.directory)
    if mtime > c.last_activity:
        c.last_activity = mtime
    silent = time.time() - c.last_activity
    if silent >= opts.stall_s and time.time() - c.last_alert >= opts.stall_s:
        c.last_alert = time.time()
        c.stall_alerts += 1
        logger.warning("[pool] STALL? %s has written nothing under %s for %.0f min (no log "
                       "output, result file or relay activity); it keeps running - check its "
                       "log %s", c.suite, c.directory, silent / 60, c.log_path)


def _reap(c: Child, suites: Sequence[str], baseline: Iterable[str] = ()) -> None:
    """Remove the containers child `c` started that are still present: the `docker run --name`
    processes seen in its process group, and gbench-<suite>-* names created after it started
    (neither present when the pool started, `baseline`, nor when the child started,
    c.containers_before)."""
    present = container_names()
    if not present:
        return
    new = present - set(baseline) - c.containers_before
    mine = (c.docker_names & present) | attribute_names(new, c.suite, suites)
    # containers a harness started through the docker socket under names of its own (2026-10-11:
    # terminal_bench's Harbor, stopped by SIGTERM, left its compose project
    # mcmc-sampling-stan__5flw3gp up): the ones that bind-mount the suite's own temp directory
    projects: Set[str] = set()
    if c.scratch_dirs:
        mounted, projects = containers_mounting(new - mine, c.scratch_dirs)
        mine |= mounted
    if not mine:
        return
    names = sorted(mine)
    code, out = _docker(["rm", "-f", *names], timeout=120)
    c.extra["containers_reaped"] = names if code == 0 else []
    if code == 0:
        logger.warning("[pool] %s: removed %d container(s) it left running: %s", c.suite,
                       len(names), ", ".join(names))
    else:
        logger.error("[pool] %s: could not remove its containers %s: %s", c.suite,
                     ", ".join(names), out.strip()[:300])
    for project in sorted(projects):
        # the compose network of a removed project (docker refuses one still in use)
        ncode, ids = _docker(["network", "ls", "-q", "--filter",
                              f"label=com.docker.compose.project={project}"])
        if ncode == 0 and ids.split() and _docker(["network", "rm", *ids.split()])[0] == 0:
            c.extra.setdefault("networks_reaped", []).append(project)


def suite_result_files(directory: Path, suite: str) -> List[Path]:
    """The eval result files a child wrote for `suite` (backups excluded)."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob(f"eval_{suite}_*.json")
                  if ".bak" not in p.name and p.is_file())


def _child_record(c: Child, root: Path, plan: Plan) -> Dict[str, Any]:
    files = suite_result_files(c.directory, c.suite)
    rel = lambda p: os.path.relpath(p, root)                                # noqa: E731
    return {
        "exit_code": c.exit_code,
        "duration_s": c.duration_s,
        "started": c.started_at or None,
        "log": rel(c.log_path),
        "result_files": [rel(p) for p in files],
        "concurrency": {"parallel_suites": plan.parallel,
                        "running_at_start": c.running_at_start,
                        "max_suites_alongside": c.max_alongside,
                        "search_max_rps": plan.search_rps.get(c.suite)},
        "stall_alerts": c.stall_alerts,
        "stop_reason": c.stop_reason,
        "command": c.argv,
        **{k: v for k, v in c.extra.items() if k != "search_max_rps"},
    }


def _clean_result(path: Path) -> bool:
    """A result that ran to its end: status success, not failed, not partial, no trial errors."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(data, dict) and data.get("status") == "success"
            and not data.get("failed") and not data.get("partial")
            and not data.get("trial_errors"))


def duration_updates(manifest: Mapping[str, Any], root: Path
                     ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """(clean, lower_bound) durations of a pool's suites for record_durations: clean when the
    process exited 0, the pool did not stop it and every result file is clean (_clean_result);
    every other suite that ran is a lower bound. A suite that never started gives neither."""
    clean: Dict[str, float] = {}
    lower: Dict[str, float] = {}
    for suite, rec in (manifest.get("suites") or {}).items():
        dur = rec.get("duration_s")
        if rec.get("not_started") or not isinstance(dur, (int, float)) or isinstance(dur, bool):
            continue
        files = [root / f for f in rec.get("result_files") or []]
        if (rec.get("exit_code") == 0 and not rec.get("stop_reason") and files
                and all(_clean_result(f) for f in files)):
            clean[suite] = float(dur)
        else:
            lower[suite] = float(dur)
    return clean, lower


def pool_exit_code(manifest: Mapping[str, Any]) -> int:
    """2 when a suite was not started, its process failed or wrote no result; else the highest
    child exit code (0 when every suite exited 0)."""
    codes = []
    for rec in (manifest.get("suites") or {}).values():
        code = rec.get("exit_code")
        if rec.get("not_started") or code is None or not rec.get("result_files"):
            codes.append(2)
        else:
            codes.append(code if code >= 0 else 2)
    if manifest.get("stopped_by_signal"):
        return 128 + int(manifest["stopped_by_signal"])
    return max(codes or [0])


def print_plan(plan: Plan, base_argv: Sequence[str], root: Path) -> None:
    total = sum(plan.estimates.values())
    print(f"{len(plan.order)} suites, {plan.parallel} at a time, longest first "
          f"(serial estimate ~{total / 60:.0f} min):")
    for s in plan.order:
        fam = next((n for n, m, _c in plan.families if s in m and n != "search"), "-")
        rps = f", search {plan.search_rps[s]} rps" if s in plan.search_rps else ""
        print(f"  {s:<24} ~{plan.estimates[s] / 60:6.1f} min  family {fam}{rps}")
    caps = ", ".join(f"{n}<={c}" for n, _m, c in plan.families)
    print(f"  family caps: {caps}")
    if plan.order:
        print("  first command: " + " ".join(
            [*GBENCH_CMD, *child_argv(base_argv, plan.order[0], root / plan.order[0])]))


# --------------------------------------------------------------------------- CLI entry point
def _missing_result_record(suite: str, rec: Mapping[str, Any]) -> Dict[str, Any]:
    """The failed record of a suite whose process wrote no result file (summary and CSV only)."""
    why = (f"the pool was stopped ({rec.get('not_started_by') or 'signal'}) before this suite "
           "started" if rec.get("not_started") else
           f"the suite's gbench process exited {rec.get('exit_code')} without writing a result "
           f"(see {rec.get('log')})")
    return {"benchmark_type": "eval", "eval_name": suite, "model_name": "-", "model_short": "-",
            "format": "-", "status": "error", "failed": True, "failure_kind": "error",
            "error": why, "total_questions": 0, "correct_answers": 0, "accuracy": None,
            "accuracy_withheld_reason": "the suite produced no result",
            "duration_s": rec.get("duration_s")}


def format_pool_table(manifest: Mapping[str, Any], root: Path) -> str:
    rows = []
    for suite, rec in sorted((manifest.get("suites") or {}).items()):
        status, acc = "no result", "-"
        for rel in rec.get("result_files") or []:
            try:
                data = json.loads((root / rel).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            status = str(data.get("status"))
            a = data.get("accuracy")
            acc = f"{a:.2f}" if isinstance(a, (int, float)) and not isinstance(a, bool) else "-"
        conc = rec.get("concurrency") or {}
        rows.append((suite, str(rec.get("exit_code")), status, acc,
                     f"{(rec.get('duration_s') or 0) / 60:.1f}",
                     str(conc.get("max_suites_alongside", "-"))))
    w = max([len(r[0]) for r in rows] + [8])
    out = [f"  {'suite':<{w}}  exit  {'status':<22}  {'accuracy':>8}  {'min':>6}  beside",
           f"  {'-' * w}  ----  {'-' * 22}  {'-' * 8}  {'-' * 6}  ------"]
    out += [f"  {s:<{w}}  {e:<4}  {st:<22}  {a:>8}  {m:>6}  {b}" for s, e, st, a, m, b in rows]
    return "\n".join(out)


def run_pool_cli(args: Any, argv: Sequence[str], parser: Any, suites: Sequence[str]) -> int:
    """`gbench --parallel-suites N`: the pool, the merged summary, the duration table update."""
    from .core.config import normalize_endpoint
    from .merge import MergeError, union_runs
    from .utils.log_manager import create_unique_dir
    parallel = int(args.parallel_suites)
    plan = build_plan(suites, parallel, load_durations(getattr(args, "eval_limit", None)))
    base_argv = suite_argv(parser, argv)
    if getattr(args, "dry_run", False):
        print_plan(plan, base_argv(plan.order[0]) if plan.order else [],
                   Path(args.results_dir) / "pool-<timestamp>")
        return 0
    root = create_unique_dir(Path(args.results_dir), f"pool-{datetime.now():%m-%d-%H%M%S}")
    opts = pool_options(parallel)
    gate = AdmissionGate(metrics_url(normalize_endpoint(args.remote_endpoint)),
                         opts.max_inflight)
    child_env = None
    if getattr(args, "skip_existing", False):
        # each child's --results-dir is a new pool-*/<suite>/ directory: --skip-existing also
        # searches the run's own --results-dir, as a serial run does (earlier pools included)
        reuse_root = str(Path(args.results_dir).absolute())
        child_env = lambda _s, _o: {"GBENCH_REUSE_ROOT": reuse_root}       # noqa: E731
    logger.info("[pool] %d suites, %d at a time, into %s", len(plan.order), parallel, root)
    manifest = run_pool(plan, base_argv, root, opts, gate=gate, child_env=child_env)
    recs = manifest.get("suites") or {}
    with_result = [s for s in suites if (recs.get(s) or {}).get("result_files")]
    missing = [_missing_result_record(s, recs.get(s) or {}) for s in suites if s not in with_result]
    try:
        union_runs([root / s for s in with_result], root, link=True, extra_failures=missing,
                   extra_metadata={"pool": {k: manifest.get(k) for k in
                                            ("parallel_suites", "max_inflight", "created",
                                             "finished", "stopped_by_signal")}})
    except MergeError as e:
        logger.error("[pool] the merged summary was not written: %s", e)
    clean, lower = duration_updates(manifest, root)
    record_durations(clean, getattr(args, "eval_limit", None), source=str(root),
                     raise_only=lower)
    print("\nPOOL RESULTS (each suite's own eval_*.json is linked in the pool directory)")
    print(format_pool_table(manifest, root))
    print(f"\nPool directory: {root} (summary.json, eval_summary.csv, pool.json; each suite's "
          "log and run directory under <suite>/)\n")
    return pool_exit_code(manifest)
