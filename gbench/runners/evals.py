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

"""The evals pillar: runs the selected eval suites against one served model.

EvalsBenchmarkRunner.run resolves the suites (resolve_eval_names), the endpoint (--remote-endpoint,
else the native server the CLI started, config.local_endpoint), and the id the endpoint serves the
model under (served_model_for), runs each suite through its runner in eval_suites.SUITES, and
stamps every result the same way whatever the suite did: the served id, the canonical_sync
record, the run's shape (run_shape), its shard (base.finalize_central_stamps), the evals scaffold
and gbench commit, and the run-level flags a suite did not record applying.
"""

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from ..core.config import BenchmarkConfig, normalize_endpoint
from ..core.models import ModelConfig, ModelFormat
from ..utils import endpoint as _endpoint

logger = logging.getLogger(__name__)


def _fmt_accuracy_pct(v) -> str:
    """Format an accuracy as `NN.NN%`, degrading a present-but-None (or otherwise non-numeric)
    value to "-". A suite whose composite is undefined for the sampled data (e.g. omnidocbench when
    the sampled pages carry no scorable annotation) returns accuracy=None; formatting that with
    `{v:.2f}` raises "unsupported format string passed to NoneType.__format__". Because the per-eval
    result print runs at the MODEL level, an unguarded format would abort the whole model run and
    lose every suite after it."""
    return f"{v:.2f}%" if isinstance(v, (int, float)) else "-"


def eval_category_names() -> Dict[str, set]:
    """The `--eval-categories` names each category-aware suite understands, read from the
    suites themselves: bfcl's categories (all 22 BFCL scoring categories, its group names and
    the pre-v4 aliases), bfcl_v4_agentic's agentic categories and swe_bench_live's leaderboard
    tabs (swe_bench_live.SETTINGS: lite, full, verified, ccpp, csharp, java, tsjs, go, rust,
    windows)."""
    from .eval_suites.bfcl import (SUPPORTED_BFCL_CATEGORIES, _CATEGORY_GROUPS,
                                   _RENAMED_CATEGORIES)
    from .eval_suites.bfcl_v4_agentic import MEMORY_CATEGORIES, WEB_SEARCH_CATEGORIES
    from .eval_suites.swe_bench_live import SETTINGS as SWE_BENCH_LIVE_SETTINGS
    return {
        "bfcl": set(SUPPORTED_BFCL_CATEGORIES) | set(_CATEGORY_GROUPS) | set(_RENAMED_CATEGORIES),
        "bfcl_v4_agentic": set(WEB_SEARCH_CATEGORIES) | set(MEMORY_CATEGORIES),
        "swe_bench_live": set(SWE_BENCH_LIVE_SETTINGS),
    }


def route_eval_categories(spec: Optional[str], eval_name: str) -> Optional[str]:
    """The part of `--eval-categories` that `eval_name` understands, or None.

    One flag serves several suites (`--evals all --eval-categories simple_python`), so each
    suite receives only the names it knows; a suite that receives none runs its full default
    track. The CLI has already rejected names no selected suite knows.
    """
    if not spec:
        return None
    known = eval_category_names().get(eval_name)
    if not known:
        return None
    names = [c.strip() for c in str(spec).split(",")]
    return ",".join(c for c in names if c in known) or None


BUILTIN_PILLARS = [
    (
        "1. GENERAL KNOWLEDGE & SCIENTIFIC REASONING",
        [
            "arc_agi",
            "causalbench",
            "cimemories",
            "cyberseceval",
            "frames",
            "gpqa",
            "gpqa_diamond",
            "healthbench",
            "humanitys_last_exam",
            "i18n_translate",
            "ifeval",
            "lab_bench",
            "livebench",
            "medxpertqa",
            "mmlu",
            "mmlu_pro",
            "mmlu_redux",
            "multilingual_mmlu",
            "simpleqa",
            "wildbench",
            "wmdp",
        ],
    ),
    (
        "2. MATHEMATICS & PROOFS",
        [
            "aime",
            "gsm8k",
            "hmmt",
            "imo_answer_bench",
            "amc_aime",
            "putnam",
            "putnam_formal",
        ],
    ),
    (
        "3. CODING & ALGORITHMIC DESIGN",
        [
            "aider_polyglot",
            "bigcodebench",
            "codeforces",
            "copilot_bench_swe",
            "cruxeval",
            "lcb",
            "multi_swe_bench",
            "multipl_e",
            "ojbench",
            "scicode",
            "swe_bench_live",
            "swe_bench_multilingual",
            "swe_bench_pro",
            "swe_lancer",
        ],
    ),
    (
        "4. LONG CONTEXT & RETRIEVAL",
        [
            "aa_lcr",
            "beam_128k",
            "loft_scifact",
            "longbench_v2_code",
            "mrcr",
            "ruler",
        ],
    ),
    (
        "5. TOOL USE & AGENTIC WORKFLOWS",
        [
            "acebench",
            "agent_dojo",
            "api_bank",
            "bfcl",
            "bfcl_live",
            "bfcl_v4_agentic",
            "browsecomp",
            "complexfuncbench",
            "deepsearch_qa",
            "gaia",
            "gaia2",
            "gdpval",
            "gorilla_apibench",
            "mcp_atlas",
            "mcp_bench",
            "mobilegym",
            "mobilegym_pp",
            "nestful",
            "nexusraven_api_eval",
            "seal_tools",
            "skillsbench",
            "spider2",
            "t_eval",
            "tau2",
            "tau3",
            "terminal_bench",
            "toolbench",
            "ui_control_osworld",
            "wildclawbench",
        ],
    ),
    (
        "6. MULTIMODAL VISION & GROUNDING",
        [
            "bundled_detection",
            "chartqa",
            "charxiv",
            "coco_caption",
            "docvqa",
            "infographicvqa",
            "mmmu_pro",
            "omnidocbench",
            "screenspot",
            "screenspot_pointing",
            "textvqa",
        ],
    ),
]

#: Built-in suites that `--evals all` leaves out. Each is implemented but needs a prerequisite most
#: hosts cannot provide (ui_control_osworld: `/dev/kvm` for OSWorld's VM; swe_lancer: a harness
#: that edits the Docker host's DOCKER-USER iptables rules while it runs (upstream recommends an
#: ephemeral VM) and pulls about 1.3 TB of task images), so under `all` it could only hard-error
#: or change the host. Naming one (`--evals ui_control_osworld`) runs it, and it
#: hard-errors infra_required naming whatever is missing. They stay in BUILTIN_PILLARS for --list
#: and the summary tables.
NAMED_ONLY_SUITES = ("swe_lancer", "ui_control_osworld")


def builtin_all_suites() -> List[str]:
    """The suites `--evals all` runs: every BUILTIN_PILLARS suite except NAMED_ONLY_SUITES."""
    names: List[str] = []
    for _, suites in BUILTIN_PILLARS:
        for s in suites:
            if s not in names and s not in NAMED_ONLY_SUITES:
                names.append(s)
    return names


def resolve_eval_names(config: Any, discover: bool = True) -> List[str]:
    """The suites `config` runs, in order, each once (the first occurrence wins).

    Each --evals entry may be a comma list; each name goes through resolve_suite_name (a renamed
    suite's old name runs the new suite). 'all' is builtin_all_suites() followed by the plugins
    discovered in --eval-plugins-dir; 'plugins' is the discovered plugins; custom_jsonl is appended
    when --eval-custom-jsonl is set. `discover` False skips plugin discovery (then 'all' and
    'plugins' name no plugin). The runner and the evals scaffold both use this list, so the
    scaffold hashes exactly the suites that run. Raises ValueError for a suite that moved to the
    roadmap (resolve_suite_name)."""
    from .eval_suites import discover_and_register_plugins, resolve_suite_name
    plugins: List[str] = []
    if discover:
        plugins = list(discover_and_register_plugins(getattr(config, "eval_plugins_dir", None)))
    raw: List[str] = []
    for item in (getattr(config, "evals", None) or []):
        for sub in str(item).split(","):
            name = resolve_suite_name(sub.strip()) if sub.strip() else ""
            if name and name not in raw:
                raw.append(name)
    if getattr(config, "eval_custom_jsonl", None) and "custom_jsonl" not in raw:
        raw.append("custom_jsonl")
    names: List[str] = []
    for entry in raw:
        if entry == "all":
            expansion = builtin_all_suites() + plugins
        elif entry == "plugins":
            expansion = plugins
        else:
            expansion = [entry]
        for name in expansion:
            if name not in names:
                names.append(name)
    return names


def suite_scoring_version(name: str, config: Any = None) -> Dict[str, Any]:
    """One suite's view of what the evals scaffold folds into its scoring version (the scaffold
    builds it for all the run's suites at once, core.scaffold.build_evals_scaffold, from the same
    two functions): {"synced", "version", "signal"} from the suite's canonical_sync entry, where
    "signal" is canonical_sync.scoring_signal([name]) (the entry's anchors and those of every
    suite and shared module it imports); for a suite with no entry that is registered from a
    module (a discovered plugin), "sha256:<hex>" of that module file's bytes
    (core.scaffold._plugin_version); for custom_jsonl, also "data_sha256" of the
    --eval-custom-jsonl file. A reconciled suite, a moved shared-module or dataset anchor, an
    edited plugin or another JSONL file therefore moves it."""
    from ..core.scaffold import _plugin_version
    from .eval_suites.canonical_sync import CANONICAL_SYNC, canonical_sync_for, scoring_signal
    if name in CANONICAL_SYNC:
        cs = canonical_sync_for(name)
        out: Dict[str, Any] = {"synced": cs.get("synced"),
                               "version": cs.get("revision") or cs.get("checksum")
                               or cs.get("status"),
                               "signal": scoring_signal([name])}
    else:
        digest = _plugin_version(name)
        out = {"synced": None, "version": f"sha256:{digest}" if digest else "unverified"}
    if name == "custom_jsonl":
        out["data_sha256"] = _file_sha256(getattr(config, "eval_custom_jsonl", None))
    return out


def eval_concurrency(config: Any) -> int:
    """The request concurrency of each suite: config.eval_concurrency (--eval-concurrency, else
    --batch-sizes[0], set by the CLI), else --batch-sizes[0], else 1; never below 1. A suite in
    eval_suites.SANDBOXED_SUITES runs --sandboxes containers at once instead, when it is set
    (else --batch-sizes[0]); see EvalsBenchmarkRunner._run_single_eval."""
    value = getattr(config, "eval_concurrency", None)
    if value is None:
        sizes = getattr(config, "batch_sizes", None) or []
        value = sizes[0] if sizes else 1
    return max(1, int(value))


#: The summary label of a failed eval configuration, by failure kind.
FAILURE_LABELS = {"infra_required": "INFRA REQUIRED", "timeout": "TIMEOUT", "error": "ERROR"}
#: Result statuses that mean a suite produced no score (base.run_eval_suite: "error" for no
#: samples, "failed" when nothing was scored; a suite's own "error"; the wall-clock "timeout";
#: "skipped", which a suite or plugin may still return through swebench_common.skipped_result:
#: a missing prerequisite, never a 0.00% row).
UNSCORED_STATUSES = ("error", "failed", "timeout", "skipped")
#: The statuses of a result --skip-existing may reuse.
REUSABLE_STATUSES = ("success", "completed_with_errors")


def eval_result_filename(eval_name: str, model_short: str, fmt: str,
                         shard: Any = None) -> str:
    """The file a suite's result is saved to under the run's results directory:
    eval_<suite>_<model>_<format>[.shard-IofN].json (the suffix for a partial shard: an object
    with index/count/partial, or the result's shard_index/shard_count)."""
    suffix = ""
    if shard is not None:
        if isinstance(shard, dict):
            index, count = shard.get("shard_index"), shard.get("shard_count")
            partial = bool(index and count and int(count) > 1)
        else:
            index, count = getattr(shard, "index", None), getattr(shard, "count", None)
            partial = bool(getattr(shard, "partial", False))
        if partial:
            suffix = f".shard-{index}of{count}"
    return f"eval_{eval_name}_{model_short}_{fmt}{suffix}.json"


def _tokenizer_trust(config: Any) -> bool:
    """trust_remote_code for gbench's own tokenizer loads (core.config.tokenizer_trust_remote_code)."""
    from ..core.config import tokenizer_trust_remote_code
    return tokenizer_trust_remote_code(config)[0]


def _suite_attempt_count(config: Any, eval_name: str) -> int:
    """The attempts per sample a suite runs: --attempt-count, or more for a suite that always
    runs more (matharena_common.SUITE_MIN_ATTEMPTS)."""
    k = int(getattr(config, "attempt_count", 1) or 1)
    from .eval_suites.matharena_common import SUITE_MIN_ATTEMPTS
    return max(k, SUITE_MIN_ATTEMPTS.get(eval_name, 1))


def failure_kind(message: str) -> str:
    """"infra_required" for an [INFRA REQUIRED] error (a missing prerequisite), else "error"."""
    return "infra_required" if "[INFRA REQUIRED]" in str(message or "") else "error"


def failure_kind_of(result: Dict[str, Any]) -> str:
    """The failure kind of a failed eval result: its `failure_kind`, else derived from its
    status and error text."""
    if result.get("failure_kind") in FAILURE_LABELS:
        return result["failure_kind"]
    if result.get("status") == "timeout":
        return "timeout"
    if result.get("status") == "skipped":
        # a skip says a prerequisite is missing (docs/evals/plugins.md: raise instead)
        return "infra_required"
    return failure_kind(result.get("error") or result.get("skip_reason") or "")


def failure_label(result: Dict[str, Any]) -> str:
    """How the summary prints a failed eval configuration: INFRA REQUIRED, TIMEOUT or ERROR."""
    return FAILURE_LABELS[failure_kind_of(result)]


def mark_unscored(result: Dict[str, Any]) -> Dict[str, Any]:
    """A result whose status says the suite produced no score (UNSCORED_STATUSES) becomes a
    FAILED configuration: `failed` True, its `failure_kind`, and its accuracy withheld (None,
    with `accuracy_withheld_reason`), so the summary lists it as INFRA REQUIRED / ERROR /
    TIMEOUT, counts it as failed, and the run exits non-zero. Other results are returned as
    they are."""
    if result.get("status") not in UNSCORED_STATUSES:
        return result
    result["failed"] = True
    result["failure_kind"] = failure_kind_of(result)
    if result.get("accuracy") is not None or not result.get("accuracy_withheld_reason"):
        result["accuracy_withheld_reason"] = (
            result.get("accuracy_withheld_reason")
            or f"the suite produced no score (status {result.get('status')}"
               + (f": {result['error']}" if result.get("error") else "") + ")")
    result["accuracy"] = None
    return result


def _run_with_timeout(eval_name, runner_fn, kwargs, configured, model_name):
    """Run one suite, optionally under a wall-clock budget. OFF unless --suite-timeout is set.

    Why it exists: one wedged suite stalls the whole sweep. In the 2026-08-15 run `tau3`
    sat ~45 minutes on a single conversation while its Gemini user-simulator returned
    PREFILL_QUEUE_OVERLOADED, and nothing behind it could start.

    Why it is OPT-IN rather than default-on - a total-runtime cap cannot tell "wedged"
    from "legitimately slow", and cutting a suite is destructive:

      * partial results are LOST. A suite 90% finished at the cap reports `timeout` and
        contributes nothing, which is worse than letting it run.
      * child processes are NOT reaped. The worker thread is abandoned, so the suite's
        Docker containers and harness subprocesses keep running and must be cleaned up by
        hand (terminal_bench leaves one container per task).
      * legitimately long harnesses (terminal_bench ~40 min, the SWE-bench family hours)
        would need per-suite budgets that go stale as hardware changes.

    So: unset -> no budget, exactly the previous behaviour. Set it when you want a sweep
    to make progress unattended and accept losing a suite to get that.
    """
    import concurrent.futures
    import threading

    from ..utils import process_groups

    budget = configured or 0
    if budget <= 0:
        return runner_fn(**kwargs)

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix=f"suite-{eval_name}")
    worker: Dict[str, int] = {}

    def _call():
        worker["thread"] = threading.get_ident()
        try:
            return runner_fn(**kwargs)
        finally:
            process_groups.clear_stop(worker["thread"])
    try:
        future = executor.submit(_call)
        return future.result(timeout=budget)
    except SystemExit:
        # A SIGTERM/SIGHUP (the CLI's handler) reaches this thread only: hand it to the suite's
        # worker thread, which checks it before it starts more work (swe_bench_live).
        stop = process_groups.stop_requested()
        if stop:
            for _ in range(50):
                if worker.get("thread") is not None:
                    break
                import time as _time
                _time.sleep(0.001)
            if worker.get("thread") is not None:
                process_groups.request_stop(stop[0], thread=worker["thread"])
        raise
    except concurrent.futures.TimeoutError:
        logger.error("[%s] exceeded its %ds wall-clock budget; marking it timed out and "
                     "continuing. Raise --suite-timeout if this suite legitimately needs "
                     "longer.", eval_name, budget)
        # A FAILED configuration with no score: listed as TIMEOUT in the summary, counted as
        # failed, and the run exits non-zero (never a 0.00% row).
        return {
            "benchmark_type": "eval",
            "eval_name": eval_name,
            "model_name": model_name,
            "status": "timeout",
            "failed": True,
            "failure_kind": "timeout",
            "total_questions": 0,
            "correct_answers": 0,
            "accuracy": None,
            "accuracy_withheld_reason": "the suite was abandoned at its wall-clock budget",
            "skip_reason": (f"exceeded the {budget}s per-suite wall-clock budget and was "
                            f"abandoned so the sweep could continue "
                            f"(raise --suite-timeout to allow longer)"),
        }
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


# --------------------------------------------------------------------------- #
# The served model: which id the endpoint serves the model under test as, and its window.
# --------------------------------------------------------------------------- #
#: The endpoint does not serve the model under test under an id gbench can match (or a local
#: server's models cannot be listed): a harness error, never a result of the model. The shared
#: resolver (utils.endpoint.resolve_served_model) raises it.
ServedModelError = _endpoint.EndpointError



def _relay_max_tokens_records(obj: Any, out: List[Dict[str, Any]], depth: int = 0) -> None:
    """Every `max_tokens_from_run` record (EndpointRelay.record) nested in a result."""
    if depth > 6:
        return
    if isinstance(obj, dict):
        rec = obj.get("max_tokens_from_run")
        if isinstance(rec, dict) and "requests_filled" in rec:
            out.append(rec)
        for value in obj.values():
            if isinstance(value, (dict, list)):
                _relay_max_tokens_records(value, out, depth + 1)
    elif isinstance(obj, list):
        for value in obj:
            _relay_max_tokens_records(value, out, depth + 1)


def _stamp_agent_max_tokens(res: Dict[str, Any]) -> None:
    """An agent suite whose relays put the run's --max-output-tokens on requests that carried no
    max_tokens (swe_agentic.agent_max_tokens_for_suite, the user's decision of 2026-10-11)
    records it: `agent_max_tokens_from_run`, `max_tokens_sent` when the suite recorded none, and
    a leaderboard_comparable reason (the upstream scaffold sends none)."""
    found: List[Dict[str, Any]] = []
    _relay_max_tokens_records(res, found)
    filled = sum(int(r.get("requests_filled") or 0) for r in found)
    if not found or not filled:
        return
    value = max(int(r.get("value") or 0) for r in found)
    res["agent_max_tokens_from_run"] = {
        "max_tokens": value, "requests_filled": filled,
        "source": "the run's --max-output-tokens, set by gbench's relay on every agent request "
                  "that carried no max_tokens (GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0 turns it off)"}
    if res.get("max_tokens_sent") is None:
        res["max_tokens_sent"] = value
        res["max_output_tokens_note"] = (
            f"sent: max_tokens={value} (the run's --max-output-tokens) on the {filled} agent "
            "request(s) that carried none; the upstream harness sends none")
    reason = (f"max_tokens={value} (the run's --max-output-tokens) sent with {filled} agent "
              "request(s) whose upstream harness sends none")
    prev = res.get("leaderboard_comparable_reason")
    if str(prev or "").strip() == "full canonical run":
        prev = None    # a suite's own "full canonical run" no longer holds with the cap sent
    if reason not in str(prev or ""):
        res["leaderboard_comparable_reason"] = f"{prev}; {reason}" if prev else reason
        res["leaderboard_comparable"] = False

class ServedModelQuery(NamedTuple):
    """What served_model_for matches: the model's candidate ids (served_model_candidates),
    --served-model-id, and whether the endpoint is a server gbench started (`local`)."""
    candidates: Tuple[str, ...]
    override: Optional[str] = None
    local: bool = False


def served_model_candidates(model: Any, format: Any = None) -> Tuple[str, ...]:
    """match_served_model's candidates for `model`, in order: its HuggingFace id, the path gbench
    serves it from (get_model_path(format)), its name and its short name; de-duplicated."""
    path = None
    get_path = getattr(model, "get_model_path", None)
    if callable(get_path) and format is not None:
        try:
            path = get_path(format)
        except Exception:                                                  # noqa: BLE001
            path = None
    out: List[str] = []
    for cand in (getattr(model, "hf_model_id", None), path, getattr(model, "name", None),
                 getattr(model, "short_name", None)):
        if isinstance(cand, str) and cand and cand not in out:
            out.append(cand)
    return tuple(out)


#: The last resolution per endpoint URL: {"id", "how", "max_model_len"}.
_SERVED_MODEL: Dict[str, Dict[str, Any]] = {}
#: The context window of the model resolved at each endpoint URL (None when not advertised).
_SERVER_MAX_LEN: Dict[str, Any] = {}


def served_model_for(base_url: str, query: ServedModelQuery) -> Dict[str, Any]:
    """The id the endpoint at `base_url` serves the model under test as, through the shared
    resolver (utils.endpoint.resolve_served_model: --served-model-id first, it must be listed;
    else the candidate ids, exactly, case-insensitively or by the last path component; never the
    first listed id by position): {"id", "how", "max_model_len" (that entry's window, or None)}.

    Raises ServedModelError when nothing listed matches (the message names the served ids and
    --served-model-id), and for a server gbench started (`local`) also when its listing cannot be
    read. A remote endpoint whose listing cannot be read is asked for the configured id, recorded
    as "configured (listing failed: ...)". Records the result for _server_max_model_len."""
    served = _endpoint.resolve_served_model(base_url, list(query.candidates), query.override,
                                            local=query.local)
    out = {"id": served.id, "how": served.source, "max_model_len": served.max_model_len}
    _SERVED_MODEL[base_url] = out
    _SERVER_MAX_LEN[base_url] = out["max_model_len"]
    return out


def _resolve_remote_model_id(base_url: str, query: Any) -> str:
    """The id to send in requests to `base_url` (served_model_for). `query` is a
    ServedModelQuery, or a plain id (a remote endpoint's configured id as the only candidate)."""
    if not isinstance(query, ServedModelQuery):
        query = ServedModelQuery(candidates=(str(query),) if query else ())
    return served_model_for(base_url, query)["id"]


def _server_max_model_len(base_url: str) -> Optional[int]:
    """The context window of the model under test at `base_url`: the window of the entry
    served_model_for matched, else the one entry the endpoint lists, else None (never the window
    of another model the endpoint serves). Read once per URL."""
    if base_url in _SERVER_MAX_LEN:
        return _SERVER_MAX_LEN[base_url]
    value = None
    try:
        entries = _endpoint.list_served_models(base_url)
        wanted = (_SERVED_MODEL.get(base_url) or {}).get("id")
        hits = ([e for e in entries if str(e["id"]) == wanted] if wanted
                else (entries if len(entries) == 1 else []))
        if hits and hits[0].get("max_model_len"):
            value = int(hits[0]["max_model_len"])
    except Exception as e:                                                 # noqa: BLE001
        logger.debug("could not read max_model_len from %s (%s)", base_url, e)
    _SERVER_MAX_LEN[base_url] = value
    return value


def _file_sha256(path: Optional[str]) -> Optional[str]:
    """sha256 of a file's bytes, or None when it cannot be read."""
    if not path:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _jsonable(obj: Any) -> Any:
    """`obj` as it reads back from a result file (tuples as lists, keys as strings)."""
    return json.loads(json.dumps(obj, sort_keys=True, default=str))


def _result_time(result: Dict[str, Any], path: Path) -> float:
    """When a result was written: its finished_at stamp, else the file's modification time."""
    stamp = result.get("finished_at")
    if isinstance(stamp, str):
        try:
            return datetime.fromisoformat(stamp).timestamp()
        except ValueError:
            pass
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _pool_record() -> Optional[Dict[str, Any]]:
    """The --parallel-suites pool's record for this suite's process (GBENCH_POOL_RECORD, set by
    gbench.pool for each child), else None."""
    raw = (os.environ.get("GBENCH_POOL_RECORD") or "").strip()
    if not raw:
        return None
    try:
        rec = json.loads(raw)
    except ValueError:
        return None
    return rec if isinstance(rec, dict) else None


class EvalsBenchmarkRunner:
    """Runner for self-contained evaluation suites in gbench."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    # ------------------------------------------------------------------ helpers
    def _failed_record(self, eval_name: str, model: Any, format: Any, error: str,
                       served_id: Optional[str] = None) -> Dict[str, Any]:
        """A FAILED configuration for a suite that did not run (a harness error, never a score)."""
        from .eval_suites.base import redact_secrets
        from .eval_suites.canonical_sync import canonical_sync_for
        rec = {"benchmark_type": "eval", "eval_name": eval_name, "model_name": model.name,
               "model_short": model.short_name, "format": format.value, "status": "error",
               "failed": True, "failure_kind": failure_kind(error), "error": error,
               "total_questions": 0, "correct_answers": 0, "accuracy": None,
               "accuracy_withheld_reason": "the suite did not run",
               "served_model_id": served_id,
               "eval_limit": getattr(self.config, "eval_limit", None),
               "canonical_sync": canonical_sync_for(eval_name)}
        return redact_secrets(rec)

    def _stopped_record(self, eval_name: str, model: Any, format: Any, ctx: Dict[str, Any],
                        elapsed_s: float) -> Optional[Dict[str, Any]]:
        """The failed record of a suite a SIGTERM/SIGHUP to the run cut short (the CLI's handler
        recorded the signal: utils.process_groups.stop_requested), else None: a harness error
        with no score, whose error names the signal and how long the suite had run."""
        import signal as _signal
        from ..utils.process_groups import stop_requested
        stop = stop_requested()
        if stop is None:
            return None
        try:
            name = _signal.Signals(stop[0]).name
        except ValueError:
            name = f"signal {stop[0]}"
        res = {"status": "error", "failed": True, "failure_kind": "error",
               "error": (f"terminated by signal {stop[0]} ({name}) after {int(elapsed_s)} s while "
                         "this suite ran (the run was stopped from outside, e.g. an outer "
                         "timeout or a scheduler); no score. Files the suite already wrote stay "
                         "under the results directory"),
               "duration_s": round(elapsed_s, 2), "total_questions": 0, "correct_answers": 0,
               "accuracy": None,
               "accuracy_withheld_reason": "the run was stopped by a signal while this suite ran"}
        try:
            return self._finish(res, eval_name, model, format, ctx)
        except Exception:                                                  # noqa: BLE001
            return res

    def _provenance(self, model: Any, format: Any) -> Dict[str, Any]:
        """The evals condition this run belongs to, as metadata.json records it: scaffold_id,
        subject_id and the gbench commit (all None when they cannot be built)."""
        out: Dict[str, Any] = {"scaffold_id": None, "subject_id": None, "gbench": None}
        try:
            from ..core.scaffold import Pillar, build_condition, gbench_commit
            cond = build_condition(Pillar.EVALS, model, format, config=self.config)
            out["scaffold_id"] = cond.scaffold.scaffold_id
            out["subject_id"] = cond.subject.subject_id
            out["gbench"] = gbench_commit()
        except Exception as e:                                             # noqa: BLE001
            logger.debug("could not build the evals condition for the results (%s)", e)
        return _jsonable(out)

    def _run_shape(self, eval_name: str, shard_label: Optional[str], served_id: Optional[str],
                   window: Optional[int]) -> Dict[str, Any]:
        """What a result was measured under, beyond the suite's own code: the run-level flags as
        this suite received them, the served id and window, the Gemini cascades, the serve
        settings and the run-shaping environment knobs (base.run_shaping_env). --skip-existing
        reuses a result only when this is unchanged."""
        from ..core.config import serve_settings_record
        from .eval_suites.base import judge_cascade, run_shaping_env, search_cascade
        cfg = self.config
        shape: Dict[str, Any] = {
            "temperature": getattr(cfg, "temperature", None),
            "max_output_tokens": getattr(cfg, "eval_max_output_tokens", None),
            "thinking": bool(getattr(cfg, "eval_thinking", False)),
            "eval_limit": getattr(cfg, "eval_limit", None),
            "shard": shard_label,
            "n_shot": getattr(cfg, "eval_n_shot", None),
            "max_soft_tokens": getattr(cfg, "eval_max_soft_tokens", None),
            "attempt_count": _suite_attempt_count(cfg, eval_name),
            "eval_categories": route_eval_categories(getattr(cfg, "eval_categories", None),
                                                     eval_name),
            "served_model_id": served_id,
            "served_max_model_len": window,
            "judge_cascade": judge_cascade(),
            "search_cascade": search_cascade(),
            "serve_settings": serve_settings_record(cfg),
            "env": run_shaping_env([eval_name]),
        }
        if eval_name == "custom_jsonl":
            shape["custom_jsonl_sha256"] = _file_sha256(getattr(cfg, "eval_custom_jsonl", None))
        return _jsonable(shape)

    def _stamp_central(self, res: Dict[str, Any], served_id: Optional[str]) -> None:
        """The shard and judge-provenance stamps (base.finalize_central_stamps). A failed result
        without `shard` ran no item: it records the shard it was asked for (shard_requested) and
        its judge provenance, not a shard it did or did not apply."""
        from .eval_suites import base as base_mod
        shard = base_mod.current_shard()
        if not (res.get("failed") and not res.get("shard")):
            base_mod.finalize_central_stamps(res, served_id)
            return
        if shard is not None and shard.partial:
            res["shard_requested"] = shard.label
        name = res.get("eval_name")
        if res.get("gemini_models_answered") or name in getattr(base_mod, "_GEMINI_GATED", {}):
            if "self_judge" not in res:
                res.update(base_mod.judge_provenance(served_id))
            if res.get("self_judge"):
                base_mod.add_comparability_reason(res, base_mod.SELF_JUDGE_REASON)

    def _stamp_unapplied_knobs(self, res: Dict[str, Any], eval_name: str) -> None:
        """Run-level flags a scored result does not say it applied get a note on the result and a
        WARNING, so no suite (a plugin included) drops one silently: --max-output-tokens (no
        max_tokens_sent / max_output_tokens_note key), --attempt-count > 1 (no attempts record),
        --temperature (none of temperature / temperature_source / sampling) and --eval-thinking
        (no thinking key)."""
        if res.get("failed"):
            return
        cfg = self.config
        mot = getattr(cfg, "eval_max_output_tokens", None)
        if mot is not None and "max_tokens_sent" not in res and "max_output_tokens_note" not in res:
            res["max_output_tokens_note"] = (f"--max-output-tokens {mot}: this suite did not "
                                             "record applying it")
            logger.warning("[%s] %s", eval_name, res["max_output_tokens_note"])
        k = getattr(cfg, "attempt_count", 1) or 1
        if k > 1 and not isinstance(res.get("attempts"), dict):
            per = res.get("attempts")
            res["attempts"] = {
                "attempts_requested": k,
                "attempts_per_sample": per if isinstance(per, int) and per > 0 else 1,
                "not_applied": (f"{eval_name}: its harness runs a fixed number of trials per "
                                f"task; --attempt-count {k} was not applied"),
            }
            logger.warning("[%s] %s", eval_name, res["attempts"]["not_applied"])
        temp = getattr(cfg, "temperature", None)
        if temp is not None and not any(key in res for key in
                                        ("temperature", "temperature_source", "sampling")):
            res["temperature_applied"] = False
            res["temperature_note"] = (f"--temperature {temp}: this suite did not record "
                                       "applying it")
            logger.warning("[%s] %s", eval_name, res["temperature_note"])
        if getattr(cfg, "eval_thinking", False) and "thinking" not in res:
            res["thinking_applied"] = False
            res["thinking_note"] = "--eval-thinking: this suite did not record applying it"
            logger.warning("[%s] %s", eval_name, res["thinking_note"])

    def _finish(self, res: Dict[str, Any], eval_name: str, model: Any, format: Any,
                ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Every stamp a result carries, whatever path produced it."""
        from .eval_suites.base import redact_secrets
        from .eval_suites.canonical_sync import canonical_sync_for
        res["benchmark_type"] = "eval"
        res["eval_name"] = eval_name
        res["model_name"] = model.name
        res["model_short"] = model.short_name
        res["format"] = format.value
        res["served_model_id"] = ctx["served_id"]
        res["served_model_id_source"] = ctx["served_how"]
        res["canonical_sync"] = canonical_sync_for(eval_name)
        res["eval_limit"] = getattr(self.config, "eval_limit", None)
        res["run_shape"] = ctx["shape"]
        res.update(ctx["provenance"])
        res["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        # The native server the CLI started (its bind address, why, whether it enforces a key).
        record = getattr(self.config, "native_server_record", None)
        if record and not getattr(self.config, "remote_endpoint", None):
            res["server"] = dict(record)
        # A suite that returned without a score (status error / failed / timeout) is a failed
        # configuration with its accuracy withheld, like one that raised.
        res = mark_unscored(res)
        _stamp_agent_max_tokens(res)
        self._stamp_central(res, ctx["served_id"])
        self._stamp_unapplied_knobs(res, eval_name)
        # A suite run by a --parallel-suites pool records the concurrency it ran under (the pool
        # size, the suites running when it started, its share of the Gemini search rate).
        pool_record = _pool_record()
        if pool_record is not None:
            res["pool"] = pool_record
        # gbench --smoke (GBENCH_SMOKE=1): smoke=true, whether the suite's scorer scored its
        # sample (smoke_scored), the accuracy withheld: a smoke result is never a score.
        from ..smoke import smoke_mode, stamp_smoke
        if smoke_mode():
            stamp_smoke(res)
        # Suites outside run_eval_suite build their own result: no credential of the
        # environment reaches the results list, the result file or the summaries.
        return redact_secrets(res)

    def _reuse_mismatch(self, eval_name: str, cached: Dict[str, Any],
                        shape: Dict[str, Any], provenance: Dict[str, Any]) -> Optional[str]:
        """Why a cached result may not be reused for this run (None when it may)."""
        from .eval_suites.base import redact_secrets
        from .eval_suites.canonical_sync import canonical_sync_for
        from ..smoke import smoke_mode
        if bool(cached.get("smoke")) != smoke_mode():
            return ("it is a smoke result (gbench --smoke), never reused by a real run"
                    if cached.get("smoke") else "this is a smoke run, which reuses nothing")
        status = cached.get("status")
        if cached.get("failed") or status not in REUSABLE_STATUSES:
            return (f"its status is {status!r} (only {' or '.join(REUSABLE_STATUSES)} is "
                    "reused)")
        if not isinstance(cached.get("run_shape"), dict):
            return "it records no run_shape (written by an older gbench)"
        if _jsonable(cached.get("canonical_sync")) != _jsonable(
                redact_secrets(canonical_sync_for(eval_name))):
            return "its canonical_sync stamp differs (the suite was reconciled since)"
        want = _jsonable(redact_secrets(dict(shape)))
        have = cached["run_shape"]
        diff = sorted(k for k in set(want) | set(have) if want.get(k) != have.get(k))
        if diff:
            return "run_shape differs in " + ", ".join(diff)
        if cached.get("scaffold_id") != provenance.get("scaffold_id"):
            return (f"its evals scaffold_id {cached.get('scaffold_id')!r} differs from this "
                    f"run's {provenance.get('scaffold_id')!r}")
        return None

    def _reusable_result(self, eval_name: str, filename: str, output_file: Path,
                         shape: Dict[str, Any], provenance: Dict[str, Any]
                         ) -> Optional[Tuple[Path, Dict[str, Any]]]:
        """--skip-existing: the newest earlier result for this suite (this run's file, then any
        under --results-dir, and under GBENCH_REUSE_ROOT when a --parallel-suites pool set it: the
        run's own --results-dir, since each pool child writes into a new pool-*/<suite>/
        directory; newest by its finished_at stamp or file time) that ran under this run's
        configuration (_reuse_mismatch), or None. One file reached by two paths (a pool's
        top-level link to a suite's result) is one candidate. A file that cannot be read is
        logged and passed over, never taken as the suite's result."""
        lm = self.config.log_manager
        candidates: List[Path] = [output_file] if output_file.exists() else []
        seen = {output_file.resolve()} if candidates else set()
        roots = [getattr(lm, "base_dir", None),
                 (os.environ.get("GBENCH_REUSE_ROOT") or "").strip() or None]
        for base_dir in dict.fromkeys(r for r in roots if r is not None):
            try:
                for p in Path(base_dir).rglob(filename):
                    real = p.resolve()
                    if real not in seen:
                        seen.add(real)
                        candidates.append(p)
            except OSError as e:
                logger.warning("--skip-existing: could not search %s (%s)", base_dir, e)
        def _read(path: Path) -> Optional[Dict[str, Any]]:
            try:
                if not path.is_file() or path.stat().st_size == 0:
                    return None
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                logger.warning("--skip-existing: cannot read %s (%s); not reusing it", path, e)
                return None
            if not isinstance(data, dict):
                logger.warning("--skip-existing: %s holds no result object; not reusing it", path)
                return None
            return data

        # Two passes, so a large results tree (each file carries its sample traces) is never
        # held in memory at once: first each candidate's time only, then the files newest first,
        # one at a time, until one may be reused.
        ordered: List[Tuple[float, str, Path]] = []
        for path in candidates:
            data = _read(path)
            if data is not None:
                ordered.append((_result_time(data, path), str(path), path))
            del data
        ordered.sort(key=lambda t: (t[0], t[1]), reverse=True)
        for _when, _key, path in ordered:
            data = _read(path)
            if data is None:
                continue
            why = self._reuse_mismatch(eval_name, data, shape, provenance)
            if why:
                logger.info("--skip-existing: not reusing %s: %s", path, why)
                continue
            return path, data
        return None

    # ------------------------------------------------------------------ run
    def run(self, model: ModelConfig, format: ModelFormat) -> List[Dict[str, Any]]:
        """Run the selected eval suites for one model.

        The endpoint is --remote-endpoint, else config.local_endpoint (the native server the CLI
        started for this model's quality, Golden Set and eval pillars). A direct caller that sets
        neither gets a native server started (and stopped) here.

        Returns one result per suite (a failed record, never a score, for a suite that could not
        run: a malformed GBENCH_SHARD, a server that did not start, an endpoint that does not
        serve the model).
        """
        if not getattr(self.config, "evals", None) and not getattr(self.config,
                                                                   "eval_custom_jsonl", None):
            return []

        from .eval_suites import base as base_mod
        from .eval_suites.base import redact_secrets

        evals_to_run = resolve_eval_names(self.config)

        # GBENCH_SHARD is read by every suite: a malformed value fails every suite here, before
        # a server starts or a suite runs, rather than differently in each suite.
        try:
            shard = base_mod.current_shard()
        except ValueError as e:
            msg = f"GBENCH_SHARD is malformed ({e}); set it to I/N or unset it"
            logger.error(msg)
            return [self._failed_record(name, model, format, msg) for name in evals_to_run]
        shard_label = shard.label if shard is not None else None

        num_threads = eval_concurrency(self.config)
        results: List[Dict[str, Any]] = []
        # The results so far, for a caller whose run a signal cut short (the CLI's summary).
        self.partial_results = results

        remote = getattr(self.config, "remote_endpoint", None)
        local = getattr(self.config, "local_endpoint", None)
        serving_runner = None
        if remote:
            base_url = normalize_endpoint(remote)
        elif local:
            base_url = normalize_endpoint(local)
        else:
            from .serving import ServingBenchmarkRunner
            serving_runner = ServingBenchmarkRunner(self.config)
            logger.info("No endpoint configured. Launching local vLLM server for evals...")
            if not serving_runner._start_server(model, format, context="native"):
                logger.error("Failed to start local vLLM server for evals.")
                return [self._failed_record(
                    name, model, format,
                    "the local vLLM server for evals did not start (see its log)")
                    for name in evals_to_run]
            base_url = f"http://127.0.0.1:{serving_runner.server_port}/v1"

        try:
            # For local prompt counting when the server has no /tokenize endpoint: the tokenizer
            # the server itself uses (its HF id, not a GGUF file path or an Ollama tag).
            self._counting_tokenizer = (getattr(self.config, "tokenizer", None)
                                        or model.hf_model_id or model.name)
            query = ServedModelQuery(served_model_candidates(model, format),
                                     getattr(self.config, "served_model_id", None),
                                     local=not remote)
            # Each run resolves again: another model (or a new server) may sit at the same URL.
            _SERVED_MODEL.pop(base_url, None)
            _SERVER_MAX_LEN.pop(base_url, None)
            try:
                target_model_id = _resolve_remote_model_id(base_url, query)
            except ServedModelError as e:
                msg = f"the endpoint does not serve the model under test: {e}"
                logger.error(msg)
                return [self._failed_record(name, model, format, msg) for name in evals_to_run]
            served_how = (_SERVED_MODEL.get(base_url) or {}).get("how")
            logger.info(f"Using server model ID for evals: '{target_model_id}'"
                        + (f" ({served_how})" if served_how else ""))
            window, mot = _server_max_model_len(base_url), getattr(
                self.config, "eval_max_output_tokens", None)
            if window and mot and mot > window // 2:
                logger.warning(
                    "The server's context window is %s tokens, so a request can use at most "
                    "%s tokens minus its prompt for the answer; --max-output-tokens %s will be "
                    "clamped per request (each trace records max_tokens_sent). Serve a larger "
                    "window (--max-model-len, or the server's own setting) to avoid it.",
                    f"{window:,}", f"{window:,}", mot)
            provenance = self._provenance(model, format)
            lm = self.config.log_manager
            skip_existing = bool(getattr(self.config, "skip_existing", False))

            for eval_name in evals_to_run:
                try:
                    ctx = {"served_id": target_model_id, "served_how": served_how,
                           "shape": self._run_shape(eval_name, shard_label, target_model_id,
                                                    window),
                           "provenance": provenance}
                    filename = eval_result_filename(eval_name, model.short_name, format.value,
                                                    shard)
                    output_file = lm.results_dir / filename

                    if skip_existing:
                        hit = self._reusable_result(eval_name, filename, output_file,
                                                    ctx["shape"], provenance)
                        if hit is not None:
                            prev_file, cached_res = hit
                            logger.info(f"Skipping {eval_name.upper()} (reusing {prev_file}: "
                                        "same run configuration)")
                            cached_res["reused_from"] = str(prev_file)
                            self._stamp_central(cached_res, target_model_id)
                            cached_res = redact_secrets(cached_res)
                            results.append(cached_res)
                            if prev_file != output_file:
                                lm.save_result(output_file, cached_res)
                            continue

                    logger.info(f"\n{'='*60}\nRunning eval: {eval_name.upper()} against "
                                f"{target_model_id}\n{'='*60}")
                    start_time = time.time()
                    from .eval_suites.swebench_common import relay_count
                    relay_mark = relay_count()
                    try:
                        res = self._run_single_eval(
                            eval_name=eval_name,
                            model_name=target_model_id,
                            base_url=base_url,
                            num_threads=num_threads,
                        )
                    except SystemExit as stop_exc:
                        # The CLI turns SIGTERM/SIGHUP into SystemExit: the suite it cut short
                        # gets a failed record naming the signal (a harness error, no score),
                        # saved like any other, before the exit goes on.
                        stopped = self._stopped_record(eval_name, model, format, ctx,
                                                       time.time() - start_time)
                        # A suite that names what the stop halted (swe_bench_live's
                        # TabScheduleStopped.record: the tabs cut short, not started, completed)
                        # adds it to the saved record.
                        own = getattr(stop_exc, "record", None)
                        if stopped is not None and isinstance(own, dict):
                            for key, value in own.items():
                                if key.startswith("tabs_") or key not in stopped:
                                    stopped[key] = value
                        if stopped is not None:
                            results.append(stopped)
                            try:
                                lm.save_result(output_file, stopped)
                            except Exception:                              # noqa: BLE001
                                pass
                        raise
                    except Exception as e:
                        duration_s = time.time() - start_time
                        if type(e).__name__ == "DatasetUnavailableError":
                            # A plugin whose private dataset the operator has not provided: a
                            # missing prerequisite, so a failed configuration (infra_required)
                            # with no score, never a 0% row.
                            msg = (f"[INFRA REQUIRED] {eval_name}: {e} (see "
                                   "docs/evals/plugins.md)")
                        else:
                            msg = str(e)
                        # Isolate per-suite failures: one raising suite must never abort the
                        # rest of the sweep. A suite that raised (an [INFRA REQUIRED]
                        # prerequisite or any other error) produced no score: a FAILED
                        # configuration with its accuracy withheld, listed as INFRA REQUIRED /
                        # ERROR in the summary (never 0.00%), counted as failed, and the run
                        # exits non-zero.
                        logger.error(f"Eval '{eval_name}' failed: {e}", exc_info=True)
                        err_res = {
                            "status": "error",
                            "failed": True,
                            "failure_kind": failure_kind(msg),
                            "error": msg,
                            "duration_s": round(duration_s, 2),
                            "total_questions": 0,
                            "correct_answers": 0,
                            "accuracy": None,
                            "accuracy_withheld_reason": "the suite raised before producing a "
                                                        "score (see error)",
                        }
                        # An exception message can carry a command line or a URL with a key
                        # in it: _finish redacts the record.
                        # the records of the relays the suite started before it raised (the
                        # smoke verdict reads their follow-up evidence)
                        from .eval_suites.swebench_common import relays_started_since
                        started = relays_started_since(relay_mark)
                        if started:
                            err_res["relays_at_failure"] = started
                        err_res = self._finish(err_res, eval_name, model, format, ctx)
                        results.append(err_res)
                        try:
                            lm.save_result(output_file, err_res)
                        except Exception:                                  # noqa: BLE001
                            pass
                        continue
                    duration_s = time.time() - start_time
                    res["duration_s"] = round(duration_s, 2)
                    res = self._finish(res, eval_name, model, format, ctx)
                    results.append(res)

                    lm.save_result(output_file, res)
                    logger.info(
                        f"Saved {eval_name.upper()} evaluation traces to {output_file}"
                    )

                    from ..cli import _attempt_split, _category_cells, _correct_cell
                    acc = res.get("accuracy", 0.0)
                    tot = res.get("total_questions", 0)
                    # a protocol that answers each question several times (gpqa's 10, lcb's
                    # 10, ...): total_questions counts generations, so say both
                    split = _attempt_split(res, tot)
                    questions_line = (f"{split[0]} x {split[1]} attempts = {tot} generations"
                                      if split else f"{tot}")
                    # '-' when the headline is not correct/total (a mean score, a corpus metric)
                    corr = _correct_cell(res, tot)
                    status = res.get("status", "ok")
                    cat_acc = res.get("category_accuracy", {})
                    cat_lines = ""
                    if 1 < len(cat_acc) <= 25:
                        cat_lines = ("\n--- By Category (all attempts) ---\n" if split
                                     else "\n--- By Category ---\n") + "\n".join(
                            f"  {c:<22} {_category_cells(s)[0]}/{s.get('total', 0)} "
                            f"({_fmt_accuracy_pct(s.get('accuracy'))})"
                            for c, s in sorted(cat_acc.items())
                        )
                    logger.info(
                        f"\n============ {eval_name.upper()} Eval Result ============\n"
                        f"Model:                   {model.short_name} ({format.value})\n"
                        f"Served model id:         {target_model_id}\n"
                        f"Status:                  {status}\n"
                        f"Thinking enabled:        {res.get('thinking', False)}\n"
                        f"Duration (s):            {res['duration_s']}\n"
                        f"Total questions:         {questions_line}\n"
                        f"Correct answers:         {corr}"
                        f"{f' (of {tot} generations)' if split and corr != '-' else ''}\n"
                        f"Accuracy:                {_fmt_accuracy_pct(acc)}"
                        f"{cat_lines}\n"
                        f"=================================================="
                    )
                finally:
                    # Files a suite, a harness or a subprocess wrote under the results
                    # directory themselves (agent trajectories, submissions, harness
                    # logs) never went through gbench's writers: redact them now.
                    self._redact_results_dir()
        finally:
            if serving_runner is not None:
                logger.info("Cleaning up local vLLM server after evals...")
                serving_runner._cleanup_server()
            self._redact_results_dir()

        return results

    def _redact_results_dir(self) -> None:
        """Replace credential values in every file under the run's results directory
        (redact_tree): the backstop for what suites and subprocesses write there themselves.
        Never raises."""
        lm = getattr(self.config, "log_manager", None)
        root = getattr(lm, "results_dir", None)
        if root is None:
            return
        try:
            from ..utils.redaction import redact_tree
            redact_tree(root)
        except Exception as e:                                             # noqa: BLE001
            logger.error("Could not check %s for credential values: %s", root, e)

    def _run_single_eval(
        self,
        eval_name: str,
        model_name: str,
        base_url: str,
        num_threads: int,
    ) -> Dict[str, Any]:
        """Run single eval suite natively via internal runners."""
        from .eval_suites import SANDBOXED_SUITES, SUITES

        if eval_name not in SUITES:
            logger.error(f"Unsupported eval suite: {eval_name}")
            return {"status": "error", "failed": True, "failure_kind": "error",
                    "error": f"Unsupported eval suite: {eval_name}", "accuracy": None,
                    "accuracy_withheld_reason": "the suite did not run",
                    "total_questions": 0, "correct_answers": 0}

        # Suites that execute model-written code require bubblewrap isolation by default.
        # Missing/blocked isolation is a real EXTERNAL prerequisite gap, so it HARD-ERRORS
        # (infra_required) rather than skipping or running unsandboxed - the no-skip policy.
        # Set GBENCH_SANDBOX=none to opt into unsandboxed execution, or =bwrap to fail at the
        # exec call. Keep this set in sync with the suites that call sandbox.run_sandboxed;
        # multipl_e is deliberately NOT here - it executes via Docker (not bubblewrap) and
        # carries its own Docker infra_required.
        CODE_EXEC_EVALS = {"codeforces", "cruxeval", "lcb", "scicode"}
        if eval_name in CODE_EXEC_EVALS:
            from .eval_suites.sandbox import sandbox_skip_reason
            from .eval_suites.swebench_common import infra_required
            _blocked = sandbox_skip_reason()
            if _blocked:
                raise infra_required(eval_name, _blocked, f"docs/evals/{eval_name}.md")

        # A containerized suite (SANDBOXED_SUITES) runs --sandboxes containers at once when it is
        # set, else --batch-sizes[0] (the value the CLI's host over-subscription guard checks);
        # every other suite sends num_threads concurrent requests. Never below 1.
        concurrency = num_threads
        if eval_name in SANDBOXED_SUITES:
            sandboxes = getattr(self.config, "sandboxes", None)
            sizes = getattr(self.config, "batch_sizes", None) or []
            if sandboxes is not None:
                concurrency = sandboxes
            elif sizes:
                concurrency = sizes[0]
        concurrency = max(1, int(concurrency or 1))

        runner_fn = SUITES[eval_name]
        kwargs: Dict[str, Any] = {
            "model_name": model_name,
            "base_url": base_url,
            "concurrency": concurrency,
            "enable_thinking": getattr(self.config, "eval_thinking", False),
            "max_output_tokens": getattr(self.config, "eval_max_output_tokens", None),
            "eval_max_soft_tokens": getattr(self.config, "eval_max_soft_tokens", None),
            "eval_n_shot": getattr(self.config, "eval_n_shot", None),
            # Each category-aware suite gets only the --eval-categories names it knows.
            "eval_categories": route_eval_categories(getattr(self.config, "eval_categories", None),
                                                     eval_name),
            "limit": getattr(self.config, "eval_limit", None),
            # ruler needs the target tokenizer
            "tokenizer": getattr(self.config, "tokenizer", None),
            "temperature": getattr(self.config, "temperature", None),
            "attempt_count": getattr(self.config, "attempt_count", 1),
            # Expose --sandboxes to every runner (not just SANDBOXED_SUITES) so containerized
            # *plugin* evals can honor it without their name being baked into core.
            "sandboxes": getattr(self.config, "sandboxes", None),
        }
        if eval_name == "custom_jsonl" and getattr(self.config, "eval_custom_jsonl", None):
            kwargs["jsonl_path"] = self.config.eval_custom_jsonl

        # Run-level generation knobs resolved centrally: a suite that does not forward
        # `temperature`/`max_output_tokens` does not silently drop them.
        from .eval_suites import base as base_mod
        from .eval_suites.base import set_run_knobs
        from ..smoke import smoke_mode
        set_run_knobs(temperature=kwargs.get("temperature"),
                      attempt_count=kwargs.get("attempt_count"),
                      # gbench --smoke: the agent relays end an episode after its first action
                      # (also in the environment, GBENCH_SMOKE=1, for harness subprocesses).
                      smoke=smoke_mode(),
                      max_output_tokens=kwargs.get("max_output_tokens"),
                      # Lets base clamp `prompt + max_tokens` to the window BEFORE sending,
                      # instead of paying a 400-and-retry (or failing silently when the
                      # prompt alone is over the limit).
                      max_model_len=_server_max_model_len(base_url),
                      # Exact prompt counts for that clamp: the server's /tokenize first
                      # (chat template and tools included), else the tokenizer the server
                      # was started with. A character estimate refused prompts that fit.
                      tokenize_url=re.sub(r"/v1/?$", "", base_url.rstrip("/")),
                      served_model=model_name,
                      tokenizer=(kwargs.get("tokenizer")
                                 or getattr(self, "_counting_tokenizer", None) or model_name),
                      tokenizer_trust_remote_code=_tokenizer_trust(self.config))
        # set_run_knobs ignores None; the window must be RESET when this server reports none,
        # or the previous server's window would size this one's requests.
        base_mod._RUN_KNOBS["max_model_len"] = _server_max_model_len(base_url)
        # The suite that runs: agent-suite relays read it (swe_agentic.agent_max_tokens_for_suite).
        base_mod._RUN_KNOBS["suite"] = eval_name

        # A plan logged by an earlier run of this suite in the same process (a sweep) is not
        # this run's (limit_plan.planned_for).
        from .eval_suites.limit_plan import clear_plan
        clear_plan(eval_name)
        return _run_with_timeout(eval_name, runner_fn, kwargs,
                                 getattr(self.config, "suite_timeout", None), model_name)
