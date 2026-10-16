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

"""Shared SWE-bench execution-harness helpers.

Every SWE-bench-style suite (swe_bench_live, swe_bench_multilingual,
copilot_bench_swe, ...) resolves a GitHub issue with one unified-diff patch per
instance, then scores execution-based resolved-rate via the swebench Docker harness
(apply patch + run FAIL_TO_PASS/PASS_TO_PASS). By default the patch comes from an
agentic mini-swe-agent rollout inside the instance image (`execute_swebench_agentic`,
see swe_agentic); the one-shot opt-in generates it from the issue text alone
(`execute_swebench`). This module holds the common prereq check, dataset loaders,
patch extractor, and harness scorer so each suite is a thin, consistent wrapper.
A suite that passes `dataset_revision` gets its dataset pinned on both sides: the rows and
the harness's `--dataset_name` come from one local snapshot of that commit
(`pinned_swe_snapshot`), in the agentic and the one-shot mode alike. The snapshot is checked
against the suite's required columns, row count and content fingerprint before it is written
and before every reuse.

`harness_image` runs the harness inside a local image instead of the host interpreter: the
SWE-bench-Live fork (swe_bench_live) or the v5 harness (`harness_v5=True`, copilot_bench_swe and
swe_bench_multilingual, whose current datasets carry the v5 columns).

`host_pull_images` pulls instance images with the host's Docker CLI before a container that has
no Docker client configuration (a scaffold or a harness image) starts task containers from them,
so the pulls carry the operator's registry login and no credential enters any container.

`require_task_network_blocked` is the cloud metadata server gate of the suites whose task
containers run model-controlled code with network access (the SWE family, terminal_bench,
skillsbench, swe_lancer, ui_control_osworld): container_guard.require_metadata_blocked on docker's
default bridge and on a user-defined bridge network (TASK_NETWORKS), before those containers
start, in an image already on the host (nothing is pulled for it), optionally checking that the
host names the task containers need resolve there. `check_windows_task_network` is the same check
for Windows containers on a Windows-containers engine (a PowerShell probe, over IPv4 and IPv6).
`verifier_dns_failure` tells a task verifier that could not resolve its package hosts before its
tests ran from one whose tests ran; `corroborated_dns_failures` makes such a failure a harness
error only when something outside that task container corroborates it.

Harness errors (the SWE family): an instance that failed outside the model's control (its agent
could not run, the harness gave it no verdict, the harness wrote no report) is marked on its trace
(`mark_harness_error`: `scoring_excluded`, `harness_error` naming the kind) and left out of both
the numerator and the denominator of the resolved rate; `scored_headline` computes the resolved
rate over the rest, counts the harness errors by kind and withholds the rate (None) when nothing
was scored. An errored swebench instance whose log shows a patch that would not apply, or tests
over the applied patch that ran past the harness's timeout, is the model's outcome
(`classify_error_ids`): unresolved and scored.
"""

import contextlib
import glob
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from . import swe_thread_cap
from .base import is_scored, run_eval_suite, tally_batch_scores
from .sampling import stratified_sample

logger = logging.getLogger(__name__)


def check_swebench_prereq(
    dataset: str, split: str, namespace: Optional[str], requires_fork: bool = False
) -> Tuple[bool, str]:
    """datasets + swebench + reachable Docker, and the harness can build a TestSpec."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        return False, "Python package 'datasets' is not installed."
    try:
        from swebench.harness.run_evaluation import load_swebench_dataset  # noqa: F401
        try:
            from swebench.harness.test_spec.test_spec import make_test_spec  # noqa: F401  (swebench < 5)
        except ImportError:
            from swebench.harness.run_evaluation import make_test_spec  # noqa: F401  (swebench >= 5 re-export)
    except ImportError:
        return False, ("Python package 'swebench' is not installed (pip install -e \".[evals]\" "
                       "from the gbench checkout).")
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=20)
        if r.returncode != 0:
            return False, "Docker daemon is not reachable."
    except Exception:
        return False, "Docker CLI/daemon is not available."
    try:
        from swebench.harness.run_evaluation import load_swebench_dataset
        try:
            from swebench.harness.test_spec.test_spec import make_test_spec  # swebench < 5
        except ImportError:
            from swebench.harness.run_evaluation import make_test_spec       # swebench >= 5 re-export
        ds = load_swebench_dataset(dataset, split)
        try:
            make_test_spec(ds[0], namespace=namespace)
        except TypeError:
            make_test_spec(ds[0])
    except Exception as e:
        hint = ""
        if requires_fork:
            # The fork and upstream swebench are the SAME package name at different
            # versions, so only one can be installed at a time - and they support
            # different suites. Measured 2026-08-15 on this box:
            #     swebench 4.1.0 (upstream)   swe_bench_multilingual OK, swe_bench_live FAIL
            #     swebench 4.0.3 (Live fork)  swe_bench_live OK, swe_bench_multilingual FAIL
            #                                 (KeyError: 'parse_log_maven')
            # So the fork runs in its own local image, never in the main environment.
            hint = (" This suite needs the SWE-bench-Live harness fork "
                    "(github.com/SWE-bench-Live/SWE-bench-Live), which is the same "
                    "`swebench` package at an older version than the one the main "
                    "environment carries - installing it here would replace it. gbench runs "
                    "the fork in the local image built from "
                    "gbench/docker/swe_bench_live.Dockerfile; see docs/evals/swe_bench_live.md.")
        return False, (
            f"installed 'swebench' cannot build {dataset} test specs.{hint} "
            f"({type(e).__name__}: {str(e)[:100]})"
        )
    return True, ""


#: Local image of the SWE-bench v5 harness (gbench/docker/swebench_v5.Dockerfile) and the
#: release it must carry.
SWEBENCH_V5_IMAGE_DEFAULT = "gbench-swebench-v5:5.0.2"
SWEBENCH_V5_IMAGE_ENV = "GBENCH_SWEBENCH_V5_IMAGE"
SWEBENCH_V5_VERSION = "5.0.2"
#: Columns a v5 TestSpec is built from (swebench/harness/utils.make_test_spec).
SWEBENCH_V5_COLUMNS = ("instance_id", "image", "repo", "version", "FAIL_TO_PASS",
                       "PASS_TO_PASS", "log_parser", "eval_type", "eval_script")


#: Re-runs of the harness's final report step when the harness scored instances but wrote no
#: run report (see the scorer), as the wait in seconds before each: the first at once, then a
#: short backoff. Each re-run starts no container and grades nothing, so it is cheap.
_REPORT_RETRY_BACKOFF_S = (0, 5, 15, 30, 60)

#: The harness's own final step, run alone: `make_run_report` over the predictions and the
#: dataset rows `swebench.harness.run_evaluation.main` hands it, which reads the per-instance
#: report.json files the scoring pass wrote. It grades nothing and starts no container. It takes
#: the scoring pass's arguments (unknown ones such as --max_workers are ignored) and serves every
#: harness release gbench runs: swebench 4.1 (host), the SWE-bench-Live fork (4.0.3) and 5.0.2
#: all load the predictions with utils.get_predictions_from_file, the rows with
#: utils.load_swebench_dataset(name, split, instance_ids) and call
#: reporting.make_run_report(predictions, full_dataset, run_id, client, ...), passing
#: `namespace` where the release takes it (4.1) and 5.x's own docker client constructor.
_REPORT_STEP = (
    "import argparse, inspect\n"
    "import docker\n"
    "from swebench.harness import run_evaluation as _run\n"
    "from swebench.harness.reporting import make_run_report\n"
    "from swebench.harness.utils import get_predictions_from_file, load_swebench_dataset\n"
    "p = argparse.ArgumentParser()\n"
    "for flag in ('--dataset_name', '--split', '--predictions_path', '--run_id', '--namespace'):\n"
    "    p.add_argument(flag)\n"
    "p.add_argument('--instance_ids', nargs='+')\n"
    "a, _ = p.parse_known_args()\n"
    "preds = {x['instance_id']: x for x in get_predictions_from_file(a.predictions_path, "
    "a.dataset_name, a.split)}\n"
    "full = load_swebench_dataset(a.dataset_name, a.split, a.instance_ids)\n"
    "client = getattr(_run, '_docker_client', docker.from_env)()\n"
    "kw = {'namespace': a.namespace} if a.namespace and 'namespace' in "
    "inspect.signature(make_run_report).parameters else {}\n"
    "make_run_report(preds, full, a.run_id, client, **kw)\n"
)


def swebench_v5_image() -> str:
    return (os.environ.get(SWEBENCH_V5_IMAGE_ENV) or "").strip() or SWEBENCH_V5_IMAGE_DEFAULT


def check_swebench_v5_prereq(snapshot: str, split: str) -> Tuple[bool, str]:
    """Docker + the local v5 harness image at SWEBENCH_V5_VERSION with the docker CLI in it, and
    the harness inside it can build a TestSpec from the pinned snapshot's rows (network off,
    snapshot mounted read-only)."""
    image = swebench_v5_image()
    build = (f"Build the SWE-bench v5 harness image LOCALLY (gbench never pulls it): "
             f"docker build -t {image} -f gbench/docker/swebench_v5.Dockerfile gbench/docker")
    try:
        rows = read_swe_snapshot(snapshot, "swebench v5 prerequisite")
    except (OSError, ValueError, RuntimeError) as e:
        return False, f"cannot read the pinned dataset snapshot {snapshot}: {e}"
    missing = [c for c in SWEBENCH_V5_COLUMNS if c not in rows[0]]
    if missing:
        return False, (f"the pinned dataset {snapshot} lacks the v5 harness columns {missing}; "
                       "the v5 harness cannot grade it.")
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode != 0:
            return False, "Docker daemon is not reachable."
    except Exception:
        return False, "Docker CLI/daemon is not available."
    try:
        r = subprocess.run(["docker", "image", "inspect", "--format", "{{json .Config.Labels}}",
                            image], capture_output=True, text=True, timeout=60)
    except Exception as e:
        return False, f"cannot inspect the image {image!r}: {e}"
    if r.returncode != 0:
        return False, f"the SWE-bench v5 harness image {image!r} is not built. " + build
    try:
        labels = json.loads(r.stdout or "null") or {}
    except ValueError:
        labels = {}
    if labels.get("org.gbench.swebench") != SWEBENCH_V5_VERSION:
        return False, (f"the image {image!r} carries swebench {labels.get('org.gbench.swebench')!r}"
                       f", not {SWEBENCH_V5_VERSION}; rebuild it. " + build)
    snap_dir = os.path.dirname(os.path.abspath(snapshot))
    # swebench 5.x stops and removes every task container with the docker CLI
    # (harness/docker_utils.cleanup_container): an image without it leaves them all running.
    probe = ("import shutil, sys; shutil.which('docker') or sys.exit('this image has no docker "
             "CLI, which swebench 5.x needs to stop and remove task containers; rebuild it'); "
             "from swebench.harness.utils import load_swebench_dataset, "
             "make_test_spec; ds = load_swebench_dataset(sys.argv[1], sys.argv[2]); "
             "[make_test_spec(i) for i in ds]; print(len(ds))")
    try:
        r = subprocess.run(["docker", "run", "--rm", "--network", "none",
                            "-v", f"{snap_dir}:{snap_dir}:ro", image, "python", "-c", probe,
                            os.path.abspath(snapshot), split],
                           capture_output=True, text=True, timeout=300)
    except Exception as e:
        return False, f"the v5 harness probe could not run in {image!r}: {e}"
    if r.returncode != 0:
        return False, (f"the swebench {SWEBENCH_V5_VERSION} harness in {image!r} cannot build "
                       f"test specs from {snapshot}: {(r.stderr or r.stdout)[-300:]}")
    return True, ""


def extract_patch(text: str) -> str:
    """Extract a unified-diff patch from a model response.

    The old version looked only at the FIRST fenced block. When the model showed the
    current code first, or labelled the fence anything other than `diff`/`patch`, the
    `diff --git` check failed and it fell through to "everything from `diff --git` to the
    end of the response" - which swept up the closing fence and the prose after it. The
    harness then rejected the patch:

        patch: **** malformed patch at line 20: ```
        >>>>> Patch Apply Failed

    Measured on the 2026-08-17 run: EVERY non-empty patch across copilot_bench_swe (12),
    swe_bench_multilingual (8) and multi_swe_bench (6) carried a stray fence, so the
    resolved rate could only ever be ~0 regardless of the model.
    """
    if not text:
        return ""
    blocks = re.findall(r"```[^\n`]*\n([\s\S]*?)```", text)
    # 1) a block with the full git header
    for body in blocks:
        if "diff --git" in body:
            return _trim_after_diff(body)
    # 2) a block that is a unified diff WITHOUT the `diff --git` line. The model emits this
    #    constantly - `--- a/pylint/utils.py` / `+++ b/pylint/utils.py` / `@@ -115,7 +115,7 @@`
    #    - and it applies fine with `git apply -p1`, but requiring the header threw it away.
    #    Measured on 2026-08-17: 40 of the 47 "no extractable patch" samples across the four
    #    SWE suites were this, i.e. most of what looked like a model failure was extraction.
    for body in blocks:
        if _is_unified_diff(body):
            return _trim_after_diff(body)
    # 3) unfenced
    idx = text.find("diff --git")
    if idx != -1:
        return _trim_after_diff(text[idx:])
    m = _UNIFIED_START.search(text)
    if m and _HUNK.search(text[m.start():]):
        return _trim_after_diff(text[m.start():])
    return ""


#: A unified diff needs a source header and at least one hunk. `/dev/null` covers new files.
_UNIFIED_START = re.compile(r"^--- (?:a/|/dev/null)", re.M)
_HUNK = re.compile(r"^@@ -\d+", re.M)


def _is_unified_diff(body: str) -> bool:
    return bool(_UNIFIED_START.search(body) and _HUNK.search(body))


def _trim_after_diff(body: str) -> str:
    """Drop anything after the diff ends: a bare fence, or trailing prose."""
    lines = body.splitlines()
    out: List[str] = []
    for line in lines:
        if line.strip().startswith("```"):
            break
        out.append(line)
    # trailing prose does not start with a diff marker; cut back to the last patch line
    while out and not re.match(r"^(diff --git|index |--- |\+\+\+ |@@|[+\- ]|new file|"
                               r"deleted file|old mode|new mode|similarity index|"
                               r"rename |Binary files|\\ No newline)", out[-1]):
        out.pop()
    return ("\n".join(out).strip() + "\n") if out else ""


def trace_patch(trace: Dict[str, Any], extract: Optional[Any] = None) -> str:
    """The patch a trace submits to a harness.

    An agentic rollout (swe_agentic) records its submission verbatim under `agent_patch`: it
    is already a `git diff`, and running it through the prose extractor would cut it at the
    first context line that looks like a fence. Every other trace is a model response, so
    the diff is extracted from it as before (`extract`, default `extract_patch`).
    """
    if "agent_patch" in trace:
        return trace.get("agent_patch") or ""
    return (extract or extract_patch)(trace.get("response_text") or "")


def load_swe_samples(
    dataset: str, split: str, limit: Optional[int], eval_name: str,
    revision: Optional[str] = None, *, columns: Sequence[str] = (),
    rows: Optional[int] = None, sha256: Optional[str] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load a SWE-bench-schema dataset into (messages, gold='', extra) samples; raises on failure.

    A diagnostic loader (the built-in suites' `_load_*` helpers): the scored one-shot path
    (`execute_swebench`) hands every row to run_eval_suite, which applies --shard and then the
    --eval-limit draw. Here `limit` draws from the rows given, with no shard.

    `revision` pins the Hub commit: the rows come from `pinned_swe_snapshot` (the same file the
    harness reads, checked against `columns` / `rows` / `sha256` and reused offline once
    written). Every built-in suite passes one; None (a caller outside the built-in suites) reads
    the dataset by name at the Hub's current head, with a warning (UNPINNED_DATASET_REASON)."""
    if revision:
        pinned = _pinned_snapshot_rows(dataset, split, revision, eval_name, columns, rows,
                                       sha256)[1]
        return swe_samples_from_rows(pinned, split, limit, eval_name, f"{dataset}@{revision}")
    logger.warning("[%s] %s", eval_name, UNPINNED_DATASET_REASON)
    try:
        from datasets import load_dataset
        rows = list(load_dataset(dataset, split=split))
    except Exception as e:
        logger.error(f"Failed to load dataset for {eval_name}: {e}")
        raise RuntimeError(f"Could not load dataset for {eval_name}: {e}") from e
    return swe_samples_from_rows(rows, split, limit, eval_name, dataset)


def presharded(samples: List[Any]) -> List[Any]:
    """`samples`, already this node's shard (the loader applied --shard itself, before its
    --eval-limit draw), laid out so base.run_eval_suite keeps exactly them: base applies
    GBENCH_SHARD "I/N" to the list it is given (items[I-1::N]), so each sample is placed at a
    position that selection keeps and the slots between stay None (dropped by the same
    selection). Without a shard the list is returned as it is. A malformed GBENCH_SHARD raises
    ValueError."""
    from .sampling import parse_shard, shard_select
    shard = parse_shard(os.environ.get("GBENCH_SHARD"))
    if not shard:
        return list(samples)
    index, count = shard
    laid_out: List[Any] = [None] * (len(samples) * count)
    laid_out[index - 1::count] = samples
    if shard_select(laid_out, shard) != list(samples):
        raise RuntimeError("presharded samples do not survive the GBENCH_SHARD selection")
    return laid_out


#: The comparability reason of a SWE-bench-schema run whose caller passed no dataset revision:
#: the rows were read by name at the Hub's current head. Every built-in suite pins a revision;
#: the by-name path is kept for callers outside them, and it says so on their result.
UNPINNED_DATASET_REASON = ("dataset read by name at the Hub's current head (no revision pinned), "
                           "so its rows can differ from run to run")

#: The --eval-limit seed of the SWE-bench-schema one-shot samples (a stratified draw by repo):
#: the seed the loader used before run_eval_suite took over the draw, so an unsharded limited run
#: keeps the same instances.
SWE_LIMIT_SEED = "swebench_common"


def swe_samples_from_rows(
    rows: List[Dict[str, Any]], split: str, limit: Optional[int], eval_name: str,
    source: str = "",
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """The one-shot (messages, gold='', extra) samples for SWE-bench-schema rows, one per
    instance id (`unique_instance_rows`), each with its repo as `category`.

    `limit` draws a subset of these rows (stratified by repo, seed SWE_LIMIT_SEED). The scored
    one-shot paths pass limit=None and give the samples to run_eval_suite with
    `limit_seed=SWE_LIMIT_SEED`: base applies --shard to the full ordered set first and then draws
    the same stratified subset (the samples' `category` is the repo), so a sharded limited run
    caps within its shard and an unsharded one keeps the instances it always had."""
    if not rows:
        raise RuntimeError(f"{eval_name} returned empty rows")
    rows = unique_instance_rows(rows)
    # Stratified, not a contiguous head.
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("repo"), seed=SWE_LIMIT_SEED)

    samples = []
    for item in rows:
        repo = item.get("repo")
        instance_id = item.get("instance_id")
        problem = item.get("problem_statement")
        if not instance_id or not problem:
            raise RuntimeError(
                f"{eval_name}: unexpected dataset schema (instance_id/problem_statement); "
                "refusing to fabricate sample data"
            )
        prompt = (
            f"You are an expert software engineer resolving a GitHub issue in {repo}.\n\n"
            f"Issue ({instance_id}):\n{problem}\n\n"
            "Produce a single unified-diff patch in git format (`diff --git a/... b/...`) "
            "that resolves the issue. Output only the patch."
        )
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, "", {"category": str(repo), "instance_id": instance_id, "split": split}))

    logger.info(f"Loaded {len(samples)} {eval_name} samples ('{source}' {split}).")
    return samples


#: Substrings in a swebench `run_instance.log` that mean the MODEL's patch would not apply (the
#: harness DID run - it checked out the repo and tried to apply the diff). These are legitimate
#: unresolved outcomes (the model produced a bad diff), NOT infrastructure failures.
_PATCH_FAIL_SIGNATURES = (
    "patch apply failed",
    "malformed patch",
    "no file to patch",
    "can't find file to patch",
    "patch failed",
    "hunk failed",
    "hunk #",
    "hunks failed",
    "hunk ignored",
    "while trying to apply the patch",
    "saving rejects",
    "corrupt patch",
    "patch does not apply",
    "does not match index",
)


#: The prefix swebench's per-instance logger puts on each record of run_instance.log
#: ("%(asctime)s - %(levelname)s - %(message)s", the same in swebench 4.1 on the host, the
#: SWE-bench-Live fork 4.0.3 and swebench 5.0.2). The markers below are matched only at the start
#: of such a record: the rest of a record can quote text the model wrote (the APPLY_PATCH_FAIL
#: record holds the output of the last apply command, `patch`, which echoes lines of the diff;
#: the "Git diff before:" / "Git diff after:" records hold the diff), and none of it can start a
#: line with this prefix (patch prefixes each line it echoes with "|" and puts a malformed line
#: after "malformed patch at line N: ", git diff prefixes each line of a diff, and a path with a
#: newline is quoted). A line is what follows a "\n" (re.M's "^"): the log is read without
#: newline translation (_instance_log), so a "\r" in the model's text starts no line.
_LOG_RECORD = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - [A-Z]+ - "
#: APPLY_PATCH_PASS: the model's patch applied (git apply or patch --fuzz).
_APPLY_PATCH_PASS_RE = re.compile(_LOG_RECORD + re.escape(">>>>> Applied Patch"), re.M)
#: APPLY_PATCH_FAIL: none of the harness's apply commands took the model's patch.
_APPLY_PATCH_FAIL_RE = re.compile(_LOG_RECORD + re.escape(">>>>> Patch Apply Failed"), re.M)
#: The record logged right after the eval script (the tests over the applied patch) returned or
#: was stopped ("Test runtime: <s> seconds").
_TEST_RUNTIME_RE = re.compile(_LOG_RECORD + r"Test runtime: ", re.M)
#: The record logged after the eval script when it did NOT time out (the diff after the tests).
_GIT_DIFF_AFTER_RE = re.compile(_LOG_RECORD + r"Git diff after:", re.M)
#: The EvaluationError run_instance raises (and logs with its traceback) when the eval script
#: outran the harness's per-instance timeout ("Test timed out after <N> seconds."; test_output.txt
#: then ends with "Timeout error: <N> seconds exceeded."). It is looked for only between the
#: "Test runtime:" record and a "Git diff after:" record, where nothing the model wrote is
#: logged (the only record there is "Test output for <id> written to <path>").
_TEST_TIMEOUT_RE = re.compile(r"Test timed out after \d+ seconds")


def _instance_log(workdir: str, iid: str) -> str:
    """The run_instance.log text of one instance of a swebench run ("" when there is none).

    Read without newline translation (`newline=""`), so a line starts only after a real "\\n"
    written by the logger's FileHandler: a "\\r" inside text the model wrote (a line of its diff
    that `patch` echoes in the APPLY_PATCH_FAIL record, e.g. "junk\\r<timestamp> - INFO - >>>>>
    Applied Patch") stays inside its line, so the _LOG_RECORD markers cannot be forged with it."""
    text = ""
    for lp in glob.glob(os.path.join(workdir, "logs", "run_evaluation", "*", "*", iid,
                                     "run_instance.log")):
        try:
            with open(lp, encoding="utf-8", errors="ignore", newline="") as f:
                text += f.read()
        except OSError:
            pass
    return text


def _test_timed_out(text: str) -> bool:
    """Whether a run_instance.log shows the eval script outrunning the harness's timeout: the
    timeout error after the last "Test runtime:" record and before any "Git diff after:"
    record that follows it."""
    runtimes = list(_TEST_RUNTIME_RE.finditer(text))
    if not runtimes:
        return False
    at = runtimes[-1].end()
    after = _GIT_DIFF_AFTER_RE.search(text, at)
    return bool(_TEST_TIMEOUT_RE.search(text, at, after.start() if after else len(text)))


def _patch_apply_failed(text: str) -> bool:
    """Whether a run_instance.log shows the model's patch not applying: the harness's
    APPLY_PATCH_FAIL record, or (for a log without the harness's record prefix) a
    _PATCH_FAIL_SIGNATURES line; never when the harness logged APPLY_PATCH_PASS (after it the
    log holds the model's diff and `patch --fuzz` output such as "Hunk #1 succeeded at")."""
    if _APPLY_PATCH_PASS_RE.search(text):
        return False
    if _APPLY_PATCH_FAIL_RE.search(text):
        return True
    low = text.lower()
    return any(sig in low for sig in _PATCH_FAIL_SIGNATURES)


def classify_error_ids(workdir: str, error_ids: Sequence[str]) -> Dict[str, set]:
    """swebench `error_ids` by what each instance's `run_instance.log` shows:

    * `patch_apply_failed`: the harness checked out the repository and the model's diff would
      not apply (_patch_apply_failed): the model's outcome, unresolved;
    * `test_timeout`: the patch applied and the tests over it ran past the harness's timeout
      (_test_timed_out). The gold patches finish within it, so a hang is what the model's
      patch did: the model's outcome, unresolved, as upstream counts it (every error_ids
      instance stays in its denominator) and as terminal_bench (VerifierTimeoutError) and
      swe_bench_pro (a test hang past the verifier budget) score it;
    * `harness_error`: anything else (docker, image build, container start, a crash of the
      harness, a missing log): the harness gave the instance no verdict."""
    out: Dict[str, set] = {"patch_apply_failed": set(), "test_timeout": set(),
                           "harness_error": set()}
    for iid in error_ids:
        text = _instance_log(workdir, iid)
        if text and _test_timed_out(text):
            out["test_timeout"].add(iid)
        elif text and _patch_apply_failed(text):
            out["patch_apply_failed"].add(iid)
        else:
            out["harness_error"].add(iid)   # missing log or another error: no verdict
    return out


def _classify_swebench_errors(workdir: str, error_ids: List[str]) -> Tuple[set, set]:
    """Split swebench `error_ids` into (the model's outcome, harness error) (classify_error_ids).

    The model's outcome is a patch that would not apply or tests over the applied patch that
    ran past the harness's timeout: unresolved (0), the canonical outcome. A harness error
    (docker, build, container, a missing log we cannot inspect) means the instance could not be
    scored at all."""
    classes = classify_error_ids(workdir, error_ids)
    return classes["patch_apply_failed"] | classes["test_timeout"], classes["harness_error"]


def record_error_classes(report: Dict[str, Any], workdir: str) -> Dict[str, set]:
    """Put the counts of classify_error_ids on a harness report (`patch_apply_failed_instances`,
    `test_timeout_instances` with `test_timeout_ids`, `harness_error_instances`) and return the
    classes."""
    classes = classify_error_ids(workdir, list(report.get("error_ids") or []))
    report["patch_apply_failed_instances"] = len(classes["patch_apply_failed"])
    report["test_timeout_instances"] = len(classes["test_timeout"])
    if classes["test_timeout"]:
        report["test_timeout_ids"] = sorted(classes["test_timeout"])
    report["harness_error_instances"] = len(classes["harness_error"])
    return classes


def _split_report(report: Dict[str, Any], workdir: str,
                  classes: Optional[Dict[str, set]] = None) -> Tuple[set, set, int]:
    """(errored instances that are the model's outcome, true harness errors, model-scoreable
    count) of a harness report (`classes`: classify_error_ids of its error_ids, when the caller
    has them already).

    Only `error_ids` can be harness errors, split by their run_instance.log
    (`classify_error_ids`: a patch-apply failure or a test timeout is the model's outcome and is
    scored, unresolved). swebench 5.x also lists `infra_failure_ids`,
    `ambiguous_failure_ids` and `failure_reasons`: regex signatures (out of memory, docker daemon
    errors, name resolution, ...) matched in test_output.txt, which the patched code writes, and
    in run_instance.log. Upstream keeps them advisory (harness/infra_failure.py: "A flagged
    instance stays in unresolved_ids / error_ids exactly as before, so the scoring denominator is
    unchanged"), so they stay recorded fields of the report here and change neither the split,
    the status nor the comparability."""
    if classes is None:
        classes = classify_error_ids(workdir, list(report.get("error_ids") or []))
    model_errored = classes["patch_apply_failed"] | classes["test_timeout"]
    harness_err = classes["harness_error"]
    model_scoreable = ((report.get("completed_instances") or 0)
                       + (report.get("empty_patch_instances") or 0) + len(model_errored))
    return model_errored, harness_err, model_scoreable


#: Harness-error kinds of the SWE family's instances (the `harness_error` a trace carries).
AGENT_INFRA = "agent_infra"            # the agent could not run (container, image, endpoint, ...)
NO_VERDICT = "harness_no_verdict"      # the harness ran the instance and gave it no verdict
NO_REPORT = "harness_no_report"        # the harness wrote no report at all

#: Model-endpoint failures by exception class: LiteLLM's exception classes (litellm.exceptions),
#: the OpenAI SDK names they subclass, and an installed agent's ApiRateLimitError. The served
#: endpoint is gbench's side of the run, not the model's (terminal_bench.trial_exception_verdict,
#: scaffold_error_harness_reason).
ENDPOINT_EXCEPTIONS = ("APIConnectionError", "APIError", "APIStatusError", "APITimeoutError",
                       "APIResponseValidationError", "AuthenticationError", "BadGatewayError",
                       "BadRequestError", "InternalServerError", "NotFoundError",
                       "PermissionDeniedError", "RateLimitError", "ServiceUnavailableError",
                       "Timeout", "UnprocessableEntityError", "ApiRateLimitError")
#: What the docker CLI prints when it cannot reach its daemon. Nothing in a task container can
#: stop the host's docker daemon.
DOCKER_DAEMON_MARKERS = ("cannot connect to the docker daemon", "error during connect",
                         "is the docker daemon running")
#: The same from the docker SDK for Python ("Error while fetching server API version: ...") and
#: from the scaffolds built on it (Win-Agent, RepoLaunch: "Docker is not installed or not
#: running.").
DOCKER_SDK_DAEMON_MARKERS = ("error while fetching server api version",
                             "docker is not installed or not running")
#: An exception text that starts with a LiteLLM or OpenAI SDK endpoint exception: str() of a
#: litellm exception is "litellm.<Class>: ..." (e.g. "litellm.InternalServerError: ...").
_ENDPOINT_EXCEPTION_RE = re.compile(
    r"(?:litellm|openai)\.(?:exceptions\.)?(?:%s)\b" % "|".join(ENDPOINT_EXCEPTIONS))


def scaffold_error_harness_reason(exception_text: Any) -> Optional[str]:
    """Why an exception a scaffold (SWE-agent, MSWE-agent, Win-Agent) recorded for a rollout,
    after the model's first turn, is a harness error ("a model endpoint failure", "the docker
    daemon unreachable"); None when it is not known to be one.

    It is a harness error only when the exception text itself STARTS with a model-endpoint
    failure (a LiteLLM or OpenAI SDK exception class of ENDPOINT_EXCEPTIONS) or with the docker
    daemon's unreachable message (DOCKER_DAEMON_MARKERS, DOCKER_SDK_DAEMON_MARKERS). The text is
    matched at its start because the rest can carry the model's own command output (e.g. a
    SWE-ReX NoExitCodeError quotes it), and a repository's tests can print either kind of
    message. Anything else (the model's commands ended its shell or its container, broke the
    scaffold's state command, ...) stays the model's outcome, as upstream counts it (every
    instance stays in the leaderboard's denominator) and as terminal_bench scores an exception
    raised while its agent ran."""
    text = " ".join(str(exception_text or "").split())
    if not text:
        return None
    if _ENDPOINT_EXCEPTION_RE.match(text):
        return "a model endpoint failure"
    low = text.lower()
    if any(low.startswith(m) for m in DOCKER_DAEMON_MARKERS + DOCKER_SDK_DAEMON_MARKERS):
        return "the docker daemon unreachable"
    return None


def mark_harness_error(trace: Dict[str, Any], kind: str, reason: Any = None,
                       status: Optional[str] = None) -> None:
    """Mark one instance's trace a harness error: not the model's outcome, left out of the
    numerator and the denominator (`scoring_excluded`), `is_correct` False, the kind in
    `harness_error` (a kind already recorded, e.g. base's endpoint failure, is kept) and the
    detail in `harness_error_reason`."""
    trace["is_correct"] = False
    trace["scoring_excluded"] = True
    if not trace.get("harness_error"):
        trace["harness_error"] = kind
    if reason and not trace.get("harness_error_reason"):
        trace["harness_error_reason"] = " ".join(str(reason).split())[:300]
    if status:
        trace["status"] = status


def harness_error_kind(trace: Dict[str, Any]) -> str:
    """The kind of a trace's harness error: the `harness_error` string, the `kind` of base's
    {"kind", "detail"} record, or `scoring_excluded` for a trace left out without one."""
    he = trace.get("harness_error")
    if isinstance(he, dict):
        return str(he.get("kind") or "harness_error")
    return str(he) if he else "scoring_excluded"


def scored_headline(traces: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The resolved rate over the instances that were scored (base.is_scored: not left out of
    the score), for a runner that builds its result from one trace per instance.

    Returns `correct_answers`, `scored_instances` (the denominator), `scoring_excluded`,
    `harness_errors` ({kind: count}), `harness_error_ids` (instance ids), `accuracy` (None when
    nothing was scored: the rate is withheld, never 0) and `category_accuracy` (over scored
    instances)."""
    correct, cats = tally_batch_scores(list(traces))
    excluded = [t for t in traces if not is_scored(t)]
    scored = len(traces) - len(excluded)
    kinds: Dict[str, int] = {}
    for t in excluded:
        k = harness_error_kind(t)
        kinds[k] = kinds.get(k, 0) + 1
    return {
        "correct_answers": correct,
        "scored_instances": scored,
        "scoring_excluded": len(excluded),
        "harness_errors": dict(sorted(kinds.items())),
        "harness_error_ids": sorted(str((t.get("extra_payload") or {}).get("instance_id")
                                        or t.get("sample_idx")) for t in excluded),
        "accuracy": round(correct / scored * 100.0, 2) if scored else None,
        "category_accuracy": {
            c: {"correct": st["correct"], "total": st["total"],
                "accuracy": round(st["correct"] / st["total"] * 100.0, 2) if st["total"] else None}
            for c, st in cats.items()},
    }


def record_harness_errors(result: Dict[str, Any], head: Dict[str, Any]) -> None:
    """Put a scored_headline's harness errors on `result` (`harness_errors` by kind and
    `harness_error_ids`, when any) and its denominator as `effective_n` (the summary table's
    Effective N), unless the result already records one (e.g. rows, row_total_headline)."""
    if head["harness_errors"]:
        result["harness_errors"] = head["harness_errors"]
        result["harness_error_ids"] = head["harness_error_ids"]
    result.setdefault("effective_n", result.get("scored_instances", head["scored_instances"]))


def harness_errors_text(harness_errors: Dict[str, int]) -> str:
    """`{kind: n}` as "kind n, kind n"."""
    return ", ".join(f"{k} {v}" for k, v in sorted(harness_errors.items()))


def withhold_on_error(result: Dict[str, Any]) -> None:
    """A result whose status is `error` reports no score: `accuracy` None, with the error as
    `accuracy_withheld_reason`."""
    if result.get("status") == "error":
        result["accuracy"] = None
        result.setdefault("accuracy_withheld_reason",
                          str(result.get("error") or result.get("status_reason")
                              or "nothing was scored"))
    elif result.get("accuracy") is None and "accuracy_withheld_reason" not in result:
        result["accuracy_withheld_reason"] = str(result.get("status_reason")
                                                 or "nothing was scored")


def make_swebench_scorer(
    eval_name: str, model_name: str, dataset: str, split: str,
    namespace: Optional[str], max_workers: int, metrics: Dict[str, Any],
    harness_image: Optional[str] = None, harness_v5: bool = False,
    instance_image: Optional[Callable[[str], str]] = None,
):
    """Async scorer: run the swebench harness over the model patches, mark resolved.

    `harness_image`: run `swebench.harness.run_evaluation` inside that LOCAL image instead of the
    host interpreter - the SWE-bench-Live fork (swe_bench_live) or the v5 harness - so neither
    replaces the swebench 4.1 the main env uses (same package name, incompatible versions).
    None => the host interpreter.

    `harness_v5`: the image runs swebench 5.x, whose run_evaluation takes neither
    `--cache_level` nor `--namespace` (the instance image comes from each row's `image` column)
    and whose report adds the advisory failure classification (`infra_failure_ids`,
    `ambiguous_failure_ids`, `failure_reasons`, recorded as reported). It runs as the caller's
    uid, so the logs it writes stay the caller's.

    `instance_image`: instance id -> the image the harness runs it in. Given, each submitted
    instance's image is pulled on the host first (host_pull_images, recorded in
    `harness_image_pulls`), so the harness image, which has no Docker client configuration,
    finds it present instead of pulling it anonymously.

    Before the harness starts any task container (it runs the tests over each model patch there,
    on docker's default bridge), `require_harness_network_blocked` checks that the cloud metadata
    server is blocked (infra_required otherwise)."""
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        model_tag = "gbench__" + re.sub(r"[^A-Za-z0-9_.-]", "_", model_name)[:48]
        run_id = "gbench_" + re.sub(r"[^A-Za-z0-9_.-]", "_", f"{eval_name}_{model_name}")[:40]

        preds: Dict[str, Dict[str, str]] = {}
        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("instance_id")
            if not iid:
                continue
            preds[iid] = {
                "instance_id": iid,
                "model_name_or_path": model_tag,
                "model_patch": trace_patch(tr),
            }
        if not preds:
            # Nothing to submit (no trace names an instance): the harness is not started, since
            # `--instance_ids` with no id is an argparse error, not a verdict. Each such trace is
            # a harness error, never an unresolved instance.
            for tr in sample_traces:
                mark_harness_error(tr, NO_REPORT, "no instance id to submit to the harness")
            logger.warning("[%s] no prediction names an instance; the swebench harness was not "
                           "started", eval_name)
            return

        workdir = tempfile.mkdtemp(prefix="gbench_swe_")
        metrics["swebench_workdir"] = workdir   # so execute_swebench can classify errored instances
        preds_path = os.path.join(workdir, "preds.jsonl")
        with open(preds_path, "w", encoding="utf-8") as f:
            for p in preds.values():
                f.write(json.dumps(p) + "\n")

        harness_args = [
            "-m", "swebench.harness.run_evaluation",
            "--dataset_name", dataset, "--split", split,
            "--predictions_path", preds_path, "--run_id", run_id,
            "--max_workers", str(max(1, max_workers)),
        ]
        if not harness_v5:
            harness_args += ["--cache_level", "env"]
        harness_args += ["--instance_ids", *list(preds.keys())]
        if namespace and not harness_v5:
            harness_args += ["--namespace", namespace]

        def _command(args: List[str]) -> Tuple[List[str], int]:
            """(argv, per-container thread cap) running the harness with `args`."""
            if not harness_image:
                # Without the thread cap every task container sees all host cores and
                # oversubscribes itself into a spin-wait; see swe_thread_cap for the measurement.
                return swe_thread_cap.apply([sys.executable, *args], max_workers, eval_name)
            # Run the harness INSIDE the isolated image. It spawns per-instance task containers on
            # the HOST daemon via the mounted socket (docker-out-of-docker); the workdir is
            # identity-mounted so the report path it writes matches what we read here, and the HF
            # cache is mounted so it loads the dataset without a re-download.
            hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
            threads = swe_thread_cap.resolve_threads(max_workers)
            cmd = ["docker", "run", "--rm", "--network", "host",
                   "-v", "/var/run/docker.sock:/var/run/docker.sock",
                   "-v", f"{workdir}:{workdir}:rw", "-w", workdir]
            if harness_v5 and hasattr(os, "getuid"):
                cmd += ["--user", f"{os.getuid()}:{os.getgid()}", "-e", "HOME=/tmp"]
                try:
                    cmd += ["--group-add", str(os.stat("/var/run/docker.sock").st_gid)]
                except OSError:
                    pass
            if os.path.isdir(hf_home):
                cmd += ["-v", f"{hf_home}:{hf_home}:rw", "-e", f"HF_HOME={hf_home}"]
            if os.environ.get("HF_TOKEN"):
                # Name only: docker copies the value from its own environment (inherited from
                # this process), so the token never appears on a command line.
                cmd += ["-e", "HF_TOKEN"]
            if dataset.endswith((".json", ".jsonl")) and os.path.isfile(dataset):
                # A pinned local snapshot (pinned_swe_snapshot): mount it at the same path.
                snap_dir = os.path.dirname(os.path.abspath(dataset))
                cmd += ["-v", f"{snap_dir}:{snap_dir}:ro"]
            harness_cmd = ["python", *args]
            if threads and threads > 0:
                # The cap must reach the TASK containers the harness creates, not only the harness
                # container: mount the standalone shim at its own path and let its bootstrap patch
                # the docker SDK inside the harness process (swe_thread_cap.wrap_command).
                shim = os.path.abspath(swe_thread_cap.__file__)
                cmd += ["-v", f"{shim}:{shim}:ro"]
                for _tvar in swe_thread_cap.THREAD_VARS:
                    cmd += ["-e", f"{_tvar}={threads}"]
                harness_cmd = swe_thread_cap.wrap_command(harness_cmd, threads)
            return cmd + [harness_image, *harness_cmd], threads

        if preds:
            await asyncio.to_thread(require_harness_network_blocked, eval_name, harness_image,
                                    list(preds), instance_image, namespace)
        if instance_image is not None and preds:
            pulls = await asyncio.to_thread(
                host_pull_images, [instance_image(i) for i in preds], max_workers)
            metrics["harness_image_pulls"] = image_pull_summary(pulls)
        cmd, threads = _command(harness_args)
        if harness_image:
            metrics["harness_image"] = harness_image
        metrics["docker_thread_cap"] = threads or None

        def _run(argv: List[str]):
            return subprocess.run(argv, cwd=workdir, capture_output=True, text=True)
        proc = await asyncio.to_thread(_run, cmd)

        report_path = os.path.join(workdir, f"{model_tag}.{run_id}.json")
        run_logs = os.path.join(workdir, "logs", "run_evaluation", run_id)
        # Same arguments, report step only (_REPORT_STEP).
        report_args = ["-c", _REPORT_STEP, *harness_args[2:]]
        retries = 0
        while (not os.path.exists(report_path) and os.path.isdir(run_logs)
               and retries < len(_REPORT_RETRY_BACKOFF_S)):
            # The harness ran instances but its final report step died: make_run_report lists
            # containers, which races with containers other processes remove on a shared
            # daemon (docker SDK NotFound). Only that step is run again, on the per-instance
            # reports the scoring pass wrote, so every instance keeps the verdict that pass
            # gave it: one with no report.json (a test timeout, a crash after the tests) stays
            # in error_ids, where classify_error_ids reads its log. (`--rewrite_reports true`
            # would instead re-grade every instance that has test output, including a
            # timed-out one.)
            delay = _REPORT_RETRY_BACKOFF_S[retries]
            retries += 1
            logger.warning("%s: the harness wrote no run report (stderr tail: %s); running its "
                           "report step again in %d s (attempt %d of %d)", eval_name,
                           (proc.stderr or "")[-300:], delay, retries,
                           len(_REPORT_RETRY_BACKOFF_S))
            if delay:
                await asyncio.sleep(delay)
            proc = await asyncio.to_thread(_run, _command(report_args)[0])
        metrics["harness_report_retries"] = retries
        resolved: set = set()
        verdictless: set = set()
        timed_out: set = set()
        report_ok = os.path.exists(report_path)
        if report_ok:
            with open(report_path, encoding="utf-8") as f:
                rep = json.load(f)
            resolved = set(rep.get("resolved_ids", []))
            metrics["swebench_report"] = {
                k: rep.get(k) for k in (
                    "total_instances", "submitted_instances", "completed_instances",
                    "resolved_instances", "unresolved_instances", "error_instances",
                    "empty_patch_instances",
                    # error_ids MUST be carried through: execute_swebench feeds it to
                    # classify_error_ids to split patch-apply failures and test timeouts (the
                    # model's outcome, unresolved) from true harness/infra errors. Without it
                    # the split silently saw [] and the infra guard could never fire, so an
                    # all-infra-errored run reported a fake "success, 0%" instead of
                    # status=error.
                    "error_ids",
                    # swebench 5.x: its post-hoc, advisory classification of failures, recorded
                    # as reported (it changes neither the score nor the status; _split_report).
                    "infra_failure_ids", "ambiguous_failure_ids", "failure_reasons",
                    # task containers the harness left behind (5.x: reported when it has a
                    # docker client); nonzero means its cleanup failed.
                    "unstopped_instances",
                )
                if k in rep or k not in ("infra_failure_ids", "ambiguous_failure_ids",
                                         "failure_reasons", "unstopped_instances")
            }
            # An errored instance whose run_instance.log shows neither a patch-apply failure
            # nor a test timeout got no verdict from the harness (classify_error_ids): a harness
            # error, left out of the score. The other two are the model's outcome, unresolved.
            classes = classify_error_ids(workdir, list(rep.get("error_ids") or []))
            verdictless = classes["harness_error"]
            timed_out = classes["test_timeout"]
        else:
            logger.error(
                "%s: harness report not found (%s). stderr tail: %s",
                eval_name, report_path, (proc.stderr or "")[-800:],
            )
            metrics["swebench_report"] = {"error": "harness report not produced"}

        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("instance_id")
            tr["is_correct"] = iid in resolved
            tr["status"] = "OK"
            if not report_ok and (preds.get(iid) or {}).get("model_patch", "").strip():
                # No report, so no verdict for any submitted patch (an empty patch is the
                # model's outcome, unresolved, with or without the harness).
                mark_harness_error(tr, NO_REPORT, "the swebench harness wrote no run report",
                                   status="HARNESS_ERROR")
            elif iid in verdictless:
                mark_harness_error(tr, NO_VERDICT, "the swebench harness gave this instance no "
                                   "verdict (error_ids; its run_instance.log shows neither a "
                                   "patch-apply failure nor a test timeout: docker, image build, "
                                   "container or a missing log)", status="HARNESS_ERROR")
            elif iid in timed_out:
                # The patch applied and its tests ran past the harness's timeout: the model's
                # outcome, unresolved and scored (classify_error_ids).
                tr["test_timed_out"] = True
    return _score


def execute_swebench(
    eval_name: str, model_name: str, base_url: str, concurrency: int,
    dataset: str, split: str, namespace: Optional[str], **kwargs,
) -> Dict[str, Any]:
    """Load samples, generate patches, and score via the harness. Assumes prereqs pass.

    `dataset_revision` (kwarg) pins the dataset: the samples and the harness both read that Hub
    commit's rows from one `pinned_swe_snapshot` file (written once, then reused offline),
    checked against the `snapshot_columns` / `snapshot_rows` / `snapshot_sha256` kwargs.

    `instance_image` (kwarg) goes to make_swebench_scorer. For the v5 harness in its image it
    defaults to each pinned row's `image` column, the image that harness runs the instance in,
    so the harness image (no Docker client configuration) finds it pulled on the host.

    With a `harness_image`, the cloud metadata server check of the harness's task containers
    (require_harness_network_blocked) runs before any model call, so a host that fails it spends
    no generation; the scorer checks again before the harness starts (cached)."""
    if kwargs.get("harness_image"):
        require_harness_network_blocked(eval_name, kwargs["harness_image"])
    revision = kwargs.get("dataset_revision")
    columns = tuple(kwargs.get("snapshot_columns") or ())
    expected_rows = kwargs.get("snapshot_rows")
    metrics: Dict[str, Any] = {}
    if revision:
        harness_dataset, rows = _pinned_snapshot_rows(dataset, split, revision, eval_name,
                                                      columns, expected_rows,
                                                      kwargs.get("snapshot_sha256"))
        metrics.update(repeated_ids_record(rows))
        # Every row: run_eval_suite applies --shard, then the --eval-limit draw (SWE_LIMIT_SEED).
        samples = swe_samples_from_rows(rows, split, None, eval_name, f"{dataset}@{revision}")
        if (kwargs.get("instance_image") is None and kwargs.get("harness_image")
                and kwargs.get("harness_v5") and any(r.get("image") for r in rows)):
            kwargs = dict(kwargs, instance_image={
                r["instance_id"]: r["image"] for r in rows if r.get("image")}.get)
    else:
        harness_dataset = dataset
        samples = load_swe_samples(dataset, split, None, eval_name)
    result = run_eval_suite(
        eval_name=eval_name,
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=make_swebench_scorer(
            eval_name, model_name, harness_dataset, split, namespace, concurrency, metrics,
            harness_image=kwargs.get("harness_image"),
            harness_v5=bool(kwargs.get("harness_v5")),
            **({"instance_image": kwargs["instance_image"]}
               if kwargs.get("instance_image") is not None else {})),
        thinking=kwargs.get("enable_thinking", False),
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        limit_seed=SWE_LIMIT_SEED,
        max_output_tokens=kwargs.get("max_output_tokens", 8192),
        temperature=kwargs.get("temperature"),
    )
    result.update(metrics)
    if revision:
        result["dataset"] = f"{dataset}@{revision} ({split})"
        row_total_headline(result, rows)
    else:
        result["dataset"] = f"{dataset} ({split}, no revision pinned)"
    # leaderboard_comparable honesty (shared by swe_bench_multilingual / copilot_bench_swe /
    # swe_bench_live). The published SWE-bench resolved rate is the FULL test split at greedy
    # decoding; a --eval-limit subset, a non-greedy (e.g. --thinking) run, or a partially
    # evaluated harness (below) is not that number. Gate on full-dataset + greedy only - NOT on the
    # harness flavour, since swe_bench_live legitimately runs the SWE-bench-Live fork and is still
    # canonical for its own leaderboard.
    noncanon: list = []
    # This shared runner prompts the model single-turn with the issue text and no
    # repository access; the published SWE-bench numbers use an agentic/retrieval
    # scaffold, so the resolved rate here is not directly leaderboard-comparable.
    noncanon.append("single-turn generation with no repository access (not the "
                    "agentic/retrieval SWE-bench protocol)")
    if not revision:
        noncanon.append(UNPINNED_DATASET_REASON)
    if kwargs.get("limit"):
        noncanon.append(kwargs.get("limit_note")
                        or f"subset run (--eval-limit {kwargs.get('limit')})")
    if result.get("temperature") not in (0.0, 0):
        noncanon.append(
            f"non-greedy temperature={result.get('temperature')} (leaderboard is greedy 0.0)")
    # A harness that did not actually evaluate is an error, not a 0% score.
    #
    # The original guard only fired when the report carried an `error` key. On the
    # 2026-08-17 run the reports were well-formed and said the work never happened:
    #   swe_bench_multilingual  completed 0/20   (9 errored, 11 empty patches)
    #   copilot_bench_swe       completed 1/20   (11 errored, 8 empty patches)
    # Both were published as ordinary 0% scores, which is exactly the fake number this
    # check exists to prevent - "the model resolved nothing" and "the harness never
    # ran" are indistinguishable to a reader.
    report = metrics.get('swebench_report')
    if isinstance(report, dict):
        if report.get('error'):
            result['status'] = 'error'
            result['error'] = report['error']
            noncanon.append("harness error (no valid resolved rate)")
        else:
            completed = report.get('completed_instances') or 0
            total = report.get('total_instances') or result.get('total_questions') or 0
            empty = report.get('empty_patch_instances') or 0
            # Split swebench's error_instances into MODEL outcomes (the diff would not apply, or
            # the tests over the applied patch ran past the harness's timeout -> unresolved/0, the
            # canonical outcome) vs TRUE harness errors (docker/build/container, or a missing log
            # -> the instance could not be scored). Only the latter is an "infrastructure
            # failure"; a non-applying or hanging patch is an honest 0, not an error, so it must
            # not hide the model's real resolved rate. Instances with a model-attributable
            # verdict: resolved/unresolved (completed) + empty patch (model emitted none) +
            # patch-apply failure + test timeout. All count as scored.
            classes = record_error_classes(report, metrics.get('swebench_workdir') or "")
            _model_err_ids, harness_err_ids, model_scoreable = _split_report(
                report, metrics.get('swebench_workdir') or "", classes)
            patch_fail_n = len(classes["patch_apply_failed"])
            timeout_n = len(classes["test_timeout"])
            if total:
                if harness_err_ids and model_scoreable == 0:
                    # nothing was model-scoreable AND real infra errors -> genuine infra failure
                    result['status'] = 'error'
                    result['error'] = (
                        "the harness could not score any of {} instances ({} true harness errors "
                        "docker/build/container, {} empty patches); this is a "
                        "harness/infrastructure failure, not a 0% resolved rate"
                    ).format(total, len(harness_err_ids), empty)
                    noncanon.append("harness infrastructure failure (no scoreable instances)")
                elif harness_err_ids:
                    # Some instances hit TRUE infra errors: the scorer marked them harness errors
                    # (scoring_excluded), so the resolved rate is over the rest.
                    result['partial_evaluation'] = {
                        "scoreable_instances": model_scoreable, "total_instances": total,
                        "harness_error_instances": len(harness_err_ids),
                        "patch_apply_failed_instances": patch_fail_n,
                        "test_timeout_instances": timeout_n,
                        "empty_patch_instances": empty,
                    }
                    noncanon.append(
                        f"partial: {len(harness_err_ids)} of {total} instance(s) hit harness errors "
                        f"(not model-scoreable)")
                    logger.warning(
                        "[%s] %d of %d instances hit TRUE harness errors (left out of the score); "
                        "%d patch-apply failures + %d test timeouts + %d empty patches count as "
                        "unresolved (model).", eval_name, len(harness_err_ids), total,
                        patch_fail_n, timeout_n, empty)
                elif completed < total:
                    # every non-completed instance is the MODEL's outcome (patch-apply failure,
                    # test timeout, empty patch), so the resolved rate over `total` is the honest
                    # number - nothing to flag.
                    logger.info(
                        "[%s] %d/%d resolved; %d patch-apply failures + %d test timeouts + %d "
                        "empty patches count as unresolved (the model's outcome, not "
                        "infrastructure).", eval_name, report.get('resolved_instances', 0), total,
                        patch_fail_n, timeout_n, empty)
    # Harness errors by kind (the scorer's, and base's endpoint failures), all left out of the
    # score; an error row reports no score.
    record_harness_errors(result, scored_headline(result.get("sample_traces") or []))
    withhold_on_error(result)
    # Set the honest comparability flag last (after the partial/error checks). Error rows keep
    # whatever status was set above; the flag simply records that the number is not comparable.
    # A reason run_eval_suite already put on the result (an empty shard, ...) is kept.
    prior = [r for r in str(result.get("leaderboard_comparable_reason") or "").split("; ") if r]
    reasons = prior + [r for r in noncanon if r not in prior]
    result["leaderboard_comparable"] = not reasons
    if reasons:
        result["leaderboard_comparable_reason"] = "; ".join(reasons)
    return result


def load_swe_rows(dataset: str, split: str, eval_name: str) -> List[Dict[str, Any]]:
    """Every row of a SWE-bench-schema split, unsampled (the agentic runner applies --shard and
    --eval-limit itself). Raises on a load or schema failure."""
    try:
        from datasets import load_dataset
        rows = [dict(r) for r in load_dataset(dataset, split=split)]
    except Exception as e:
        logger.error(f"Failed to load dataset for {eval_name}: {e}")
        raise RuntimeError(f"Could not load dataset for {eval_name}: {e}") from e
    if not rows:
        raise RuntimeError(f"{eval_name} returned empty rows")
    for r in rows:
        if not r.get("instance_id") or not r.get("problem_statement"):
            raise RuntimeError(
                f"{eval_name}: unexpected dataset schema (instance_id/problem_statement); "
                "refusing to fabricate sample data")
    return rows


def swebench_report_reasons(eval_name: str, result: Dict[str, Any],
                            metrics: Dict[str, Any]) -> List[str]:
    """The harness-report checks of `execute_swebench`, for the agentic path.

    Same rules: a missing report is an error; an errored instance whose log shows the patch
    would not apply, or the tests over the applied patch running past the harness's timeout, is
    the model's outcome (unresolved, scored); any other error is a true harness error (the scorer
    left it out of the score), which makes the run partial, or an error when nothing was
    model-scoreable. Mutates `result` (status / error / partial_evaluation) and returns the
    non-comparability reasons.
    """
    reasons: List[str] = []
    report = metrics.get("swebench_report")
    if not isinstance(report, dict):
        return reasons
    if report.get("error"):
        result["status"] = "error"
        result["error"] = report["error"]
        return ["harness error (no valid resolved rate)"]
    total = report.get("total_instances") or 0
    empty = report.get("empty_patch_instances") or 0
    classes = record_error_classes(report, metrics.get("swebench_workdir") or "")
    _model_err_ids, harness_err_ids, model_scoreable = _split_report(
        report, metrics.get("swebench_workdir") or "", classes)
    if total and harness_err_ids:
        if model_scoreable == 0:
            result["status"] = "error"
            result["error"] = (
                "the harness could not score any of {} submitted instances ({} true harness "
                "errors docker/build/container, {} empty patches); this is a "
                "harness/infrastructure failure, not a 0% resolved rate"
            ).format(total, len(harness_err_ids), empty)
            reasons.append("harness infrastructure failure (no scoreable instances)")
        else:
            partial = result.setdefault("partial_evaluation", {})
            partial.update({
                "harness_scoreable_instances": model_scoreable,
                "harness_submitted_instances": total,
                "harness_error_instances": len(harness_err_ids),
                "patch_apply_failed_instances": len(classes["patch_apply_failed"]),
                "test_timeout_instances": len(classes["test_timeout"]),
                "empty_patch_instances": empty,
            })
            if result.get("status") == "success":
                result["status"] = "completed_with_errors"
            reasons.append(f"partial: {len(harness_err_ids)} of {total} submitted instance(s) "
                           "hit harness errors (not model-scoreable)")
    return reasons


def _snapshot_root() -> str:
    return os.path.join(os.environ.get("GBENCH_CACHE_DIR")
                        or os.path.join(os.path.expanduser("~"), ".cache", "gbench"),
                        "swebench_snapshots")


def pinned_swe_snapshot(dataset: str, split: str, revision: str, eval_name: str, *,
                        columns: Sequence[str] = (), rows: Optional[int] = None,
                        sha256: Optional[str] = None) -> str:
    """The `split` of `dataset` at Hub commit `revision`, as a local JSONL file.

    The swebench harness (and the SWE-bench-Live fork) take `--dataset_name` as either a Hub id,
    which they load at whatever the Hub serves today, or a local .json/.jsonl file
    (harness/utils.load_swebench_dataset). Writing the pinned rows to a file once and handing
    that file to both the rollout and the harness pins both sides to the same commit. The path
    carries the revision and the file is written atomically.

    `columns` are the columns every row must carry (the ones the suite's harness grades from),
    `rows` the split's row count at that commit and `sha256` its content fingerprint
    (`snapshot_sha256`), when the suite knows them. The fetched rows are checked against all
    three before they are written, and an existing file is checked the same way before it is
    reused (then offline): a file that fails (unreadable, empty, a row without
    instance_id/problem_statement or a required column, a different row count, different
    content) is moved aside to `<file>.invalid-<unix time>` and the rows are fetched again.
    An instance id the published split repeats is kept as published (SWE-bench-Live's
    `verified` and `full` repeat one); the harness reads every row, and the rollout and the
    one-shot prompts run once per instance id (`unique_instance_rows`), since the harness keys
    predictions by instance id. Raises RuntimeError with the reason when the rows cannot be
    loaded or do not pass.
    """
    return _pinned_snapshot_rows(dataset, split, revision, eval_name, columns, rows, sha256)[0]


def _pinned_snapshot_rows(dataset: str, split: str, revision: str, eval_name: str,
                          columns: Sequence[str] = (), rows: Optional[int] = None,
                          sha256: Optional[str] = None,
                          ) -> Tuple[str, List[Dict[str, Any]]]:
    """(path, rows) of `pinned_swe_snapshot`, so a caller that needs the rows reads them once."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "__", dataset)
    path = os.path.join(_snapshot_root(), f"{safe}@{revision}", f"{split}.jsonl")
    label = f"{dataset}@{revision} ({split})"
    if os.path.isfile(path):
        try:
            cached = _read_jsonl(path)
            problem = _snapshot_problem(cached, columns, rows, sha256)
        except (OSError, ValueError) as e:
            cached, problem = [], f"unreadable: {type(e).__name__}: {e}"
        if not problem:
            return path, cached
        aside = f"{path}.invalid-{int(time.time())}"
        try:
            os.replace(path, aside)
        except OSError as e:
            raise RuntimeError(
                f"{eval_name}: the cached snapshot {path} is not {label} ({problem}) and could "
                f"not be moved aside ({e}); move it away or set GBENCH_CACHE_DIR to a writable "
                "directory") from e
        logger.warning("%s: the cached snapshot %s is not %s (%s); moved it to %s and fetching "
                       "the pinned rows again", eval_name, path, label, problem, aside)
    try:
        from datasets import load_dataset
        fetched = [dict(r) for r in load_dataset(dataset, split=split, revision=revision)]
    except Exception as e:
        logger.error(f"Failed to load dataset for {eval_name}: {e}")
        raise RuntimeError(
            f"Could not load dataset for {eval_name} ({dataset}@{revision}, {split}): {e}") from e
    problem = _snapshot_problem(fetched, columns, rows, sha256)
    if problem:
        raise RuntimeError(f"{eval_name}: the rows loaded for {label} are not the pinned "
                           f"dataset ({problem}); refusing to write or use them")
    _write_snapshot(path, fetched, eval_name)
    logger.info("%s: pinned %s (%d rows) to %s", eval_name, label, len(fetched), path)
    return path, fetched


def _snapshot_problem(rows: List[Dict[str, Any]], columns: Sequence[str] = (),
                      expected: Optional[int] = None, sha256: Optional[str] = None) -> str:
    """Why `rows` are not a usable pinned snapshot ('' when they are)."""
    if not rows:
        return "no rows"
    for r in rows:
        if not isinstance(r, dict) or not r.get("instance_id") or not r.get("problem_statement"):
            iid = r.get("instance_id") if isinstance(r, dict) else None
            return f"a row without instance_id/problem_statement ({iid!r})"
        missing = [c for c in columns if c not in r]
        if missing:
            return f"missing the harness columns {missing} (row {r['instance_id']!r})"
    if expected is not None and len(rows) != expected:
        return f"{len(rows)} rows, not the {expected} of the pinned split"
    if sha256:
        got = snapshot_sha256(rows)
        if got != sha256:
            return (f"content sha256 {got[:16]}..., not the pinned split's {sha256[:16]}... "
                    "(snapshot_sha256)")
    return ""


def snapshot_sha256(rows: Sequence[Dict[str, Any]]) -> str:
    """Content fingerprint of a split: sha256 over its rows in the published order, each as
    canonical JSON (sorted keys, compact separators, UTF-8, a value JSON cannot encode such as a
    timestamp written as str(), which is how the snapshot file stores it) plus a newline. The
    rows `datasets` returns and the snapshot file written from them give the same value."""
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r, default=str, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":")).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def repeated_instance_ids(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """{instance_id: row count} for every instance id that appears on more than one row."""
    counts: Dict[str, int] = {}
    for r in rows:
        iid = r.get("instance_id")
        counts[iid] = counts.get(iid, 0) + 1
    return {iid: n for iid, n in counts.items() if n > 1}


def unique_instance_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The first row of each instance id, in order. The swebench harnesses (upstream and the
    SWE-bench-Live fork) key predictions by instance id, so an instance id gets one rollout and
    one patch however many rows the published split gives it."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("instance_id") not in seen:
            seen.add(r.get("instance_id"))
            out.append(r)
    return out


def repeated_ids_record(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Result fields recording the instance ids a published split repeats ({} when none)."""
    repeated = repeated_instance_ids(rows)
    if not repeated:
        return {}
    return {
        "dataset_repeated_instance_ids": repeated,
        "dataset_repeated_instance_ids_note": (
            "the published split repeats these instance ids; each ran once (the harness keys "
            "predictions by instance id) and the harness read every row as published, so its "
            "report's total_instances counts rows; total_questions and accuracy count rows too "
            "(resolved instance ids / rows, as the SWE-bench-Live leaderboard computes resolved "
            "/ total), total_instance_ids counts ids"),
    }


def row_total_headline(result: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> None:
    """Put the headline over rows when the evaluated instances include an id the published
    split repeats.

    The harness reads every row and reports `total_instances` = rows, with resolved ids unique
    (make_run_report); SWE-bench-Live's leaderboard ranks resolved / total
    (swe-bench-live.github.io leaderboard.js: resolved_percentage = resolved / report.total;
    its Verified rows have total 500 on the 500-row split that holds 499 ids). So
    `total_questions` becomes the rows of the evaluated instance ids, `accuracy` the resolved
    ids over the rows of the ids that were scored (a harness error's rows are left out, as the
    instance is: `scored_instances` counts those rows), and `total_instance_ids` keeps the id
    count. Nothing changes when no evaluated id repeats, or when the result does not hold one
    trace per id (several attempts)."""
    traces = result.get("sample_traces") or []
    ids = {(t.get("extra_payload") or {}).get("instance_id") for t in traces}
    ids.discard(None)
    if not ids or result.get("total_questions") != len(ids):
        return
    n_rows = sum(1 for r in rows if r.get("instance_id") in ids)
    if n_rows == len(ids):
        return
    scored_ids = {(t.get("extra_payload") or {}).get("instance_id") for t in traces
                  if is_scored(t)}
    n_scored_rows = sum(1 for r in rows if r.get("instance_id") in scored_ids)
    correct = int(result.get("correct_answers") or 0)
    result["total_instance_ids"] = len(ids)
    result["total_questions"] = n_rows
    if "scored_instances" in result or n_scored_rows != n_rows:
        result["scored_instances"] = n_scored_rows
    result["effective_n"] = n_scored_rows
    result["accuracy"] = round(correct / n_scored_rows * 100.0, 2) if n_scored_rows else None


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_snapshot(path: str, rows: List[Dict[str, Any]], eval_name: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                # default=str: SWE-bench-Live's created_at is a timestamp column; the harness
                # types it as a string and never reads it.
                f.write(json.dumps(r, default=str) + "\n")
        os.replace(tmp, path)
    except OSError as e:
        raise RuntimeError(
            f"{eval_name}: cannot write the pinned dataset snapshot {path} ({e}); set "
            "GBENCH_CACHE_DIR to a writable directory") from e


def _check_swe_rows(rows: List[Dict[str, Any]], eval_name: str) -> None:
    if not rows:
        raise RuntimeError(f"{eval_name} returned empty rows")
    for r in rows:
        if not r.get("instance_id") or not r.get("problem_statement"):
            raise RuntimeError(
                f"{eval_name}: unexpected dataset schema (instance_id/problem_statement); "
                "refusing to fabricate sample data")


def read_swe_snapshot(path: str, eval_name: str) -> List[Dict[str, Any]]:
    """Every row of a `pinned_swe_snapshot` file (the exact rows the harness reads)."""
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    _check_swe_rows(rows, eval_name)
    return rows


def execute_swebench_agentic(
    eval_name: str, model_name: str, base_url: str, concurrency: int,
    dataset: str, split: str, namespace: Optional[str], *,
    leaderboard_note: Optional[str] = None, dataset_revision: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """The agentic default for SWE-bench-schema suites: mini-swe-agent rollouts in each
    instance's image (swe_agentic), then the same swebench harness scorer as the one-shot
    path. Assumes the suite's own prerequisites and `swe_agentic.require_agent` (or
    `require_agent_v2` when the suite passes `agent_v2=<release>`) passed.

    With `dataset_revision` the rows come from `pinned_swe_snapshot` (checked against the
    `snapshot_columns` / `snapshot_rows` / `snapshot_sha256` kwargs) and the harness reads the
    same file, so the rollout and the scoring see one dataset commit. Each instance id is rolled
    out once (`unique_instance_rows`); when an evaluated id repeats, the headline counts rows
    (`row_total_headline`)."""
    from . import swe_agentic
    columns = tuple(kwargs.pop("snapshot_columns", None) or ())
    expected_rows = kwargs.pop("snapshot_rows", None)
    expected_sha = kwargs.pop("snapshot_sha256", None)
    if dataset_revision:
        harness_dataset, rows = _pinned_snapshot_rows(dataset, split, dataset_revision, eval_name,
                                                      columns, expected_rows, expected_sha)
    else:
        # Every built-in suite pins a revision; a caller outside them gets the by-name read,
        # named on its result (UNPINNED_DATASET_REASON).
        logger.warning("[%s] %s", eval_name, UNPINNED_DATASET_REASON)
        harness_dataset = dataset
        rows = load_swe_rows(dataset, split, eval_name)
    tasks = [swe_agentic.swebench_task(r, namespace or "swebench", split=split)
             for r in unique_instance_rows(rows)]
    metrics: Dict[str, Any] = {}
    if dataset_revision:
        metrics["dataset"] = f"{dataset}@{dataset_revision} ({split})"
    else:
        metrics["dataset"] = f"{dataset} ({split}, no revision pinned)"
    metrics.update(repeated_ids_record(rows))
    scorer = make_swebench_scorer(
        eval_name, model_name, harness_dataset, split, namespace, concurrency, metrics,
        harness_image=kwargs.get("harness_image"), harness_v5=bool(kwargs.get("harness_v5")))

    def _finalize(res: Dict[str, Any], m: Dict[str, Any]) -> List[str]:
        reasons = swebench_report_reasons(eval_name, res, m)
        row_total_headline(res, rows)
        if not dataset_revision:
            reasons.append(UNPINNED_DATASET_REASON)
        return reasons
    return swe_agentic.run_agentic(
        eval_name, model_name, base_url, concurrency, tasks, scorer, metrics,
        leaderboard_note=leaderboard_note, finalize=_finalize, **kwargs)


def skipped_result(eval_name: str, model_name: str, reason: str, docs_url: str) -> Dict[str, Any]:
    """A result for a suite whose prerequisite is missing, kept for plugins that still return
    one (docs/evals/plugins.md asks them to raise infra_required instead). It carries no score:
    the central backstop (evals.mark_unscored) makes a "skipped" status a failed configuration
    of kind infra_required with its accuracy withheld, and the run exits non-zero."""
    msg = (f"[INFRA REQUIRED] {eval_name} did not run: {reason} See '{docs_url}' for setup "
           "instructions.")
    logger.error(msg)
    return {
        "benchmark_type": "eval",
        "eval_name": eval_name,
        "model_name": model_name,
        "status": "skipped",
        "total_questions": 0,
        "correct_answers": 0,
        "accuracy": None,
        "accuracy_withheld_reason": "a prerequisite is missing, so the suite did not run",
        "skip_reason": f"[INFRA REQUIRED] {reason} (See {docs_url})",
    }


def infra_required(eval_name: str, reason: str, docs_url: str) -> RuntimeError:
    """Return a hard error for a canonical suite whose required infra is absent.

    The no-skip policy (2026-09): a faithful eval that cannot run because an
    out-of-process prerequisite is missing - Docker, an execution harness, a live
    external API/DB/VM, the official agent-environment, or a required judge/API key -
    must NOT silently skip nor emit a fabricated/partial number. It raises instead.
    `run_eval_suites` (evals.py) isolates the raise into a per-suite `status:"error"`
    row carrying this message and continues the sweep, so one un-provisioned suite
    never aborts the rest. Idiom at the call site: `raise infra_required(...)`.
    """
    msg = (f"[INFRA REQUIRED] {eval_name} cannot run and must not be skipped: {reason} "
           f"Provision the prerequisites and re-run - see '{docs_url}'.")
    logger.error(msg)
    return RuntimeError(msg)


def isolated_git_env(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """`env` (default: this process's environment) for a benchmark harness that runs git on this
    host (clones, checkouts): the operator's own git configuration is not read
    (GIT_CONFIG_GLOBAL=/dev/null, GIT_CONFIG_NOSYSTEM=1) and git never prompts
    (GIT_TERMINAL_PROMPT=0). A personal setting such as
    `url.ssh://git@github.com/.insteadOf https://github.com/` otherwise turns the harness's
    public HTTPS clone into an SSH one that fails without the operator's key (multi_swe_bench,
    2026-10-08: no report was written)."""
    out = dict(os.environ if env is None else env)
    # git configuration passed in the environment (GIT_CONFIG_COUNT/KEY_n/VALUE_n,
    # GIT_CONFIG_PARAMETERS) is the operator's too
    for name in [k for k in out if k in ("GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS")
                 or k.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))]:
        out.pop(name)
    out.update({"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_TERMINAL_PROMPT": "0"})
    return out


def prereqs_path(subdir: str, explicit: Optional[str] = None) -> Optional[str]:
    """Resolve a suite's local prerequisite (a checkout / data dir / file), with a shared fallback.

    Precedence:
      1. `explicit` - the suite's own env var (already stripped by the caller), if set;
      2. `$GBENCH_PREREQS_DIR/<subdir>` when GBENCH_PREREQS_DIR is set AND that path exists on disk;
      3. None.

    This is purely ADDITIVE: it returns None (never a fabricated path) when nothing resolves, so the
    caller's existing hard-error / infra_required check fires unchanged. The `os.path.exists` guard
    means a wrong or empty prereqs tree still hard-errors exactly as before - the fallback only kicks
    in when the operator has actually provisioned the standard tree. `<subdir>` is the suite-specific
    path under the prereqs root (may be nested, e.g. "PurpleLlama/CodeShield").
    """
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    root = os.environ.get("GBENCH_PREREQS_DIR")
    if not root:
        return None
    cand = os.path.join(os.path.expanduser(root), subdir)
    return cand if os.path.exists(cand) else None


def partial_run(eval_name: str, reason: str, docs_url: str,
                **detail: Any) -> Dict[str, Any]:
    """Record a run that PROCEEDED but measured less than the benchmark defines.

    The sibling of `skipped_result`, for the case where a missing prerequisite removes
    part of the benchmark instead of all of it. A bare `logger.warning` is not enough:
    the log is ephemeral and the result JSON is what anyone reads later, so a partial
    run was indistinguishable from a complete one. On 2026-08-19 `aider_polyglot` dropped
    all its Java exercises for want of `gradle` and published a 5-language score with
    nothing in the JSON marking it partial.

    Returns fields to merge into the suite's result. Always sets
    `leaderboard_comparable=False`, because a benchmark measured over a subset of what it
    defines is not the benchmark.
    """
    msg = (f"[PARTIAL] {eval_name}: {reason} "
           f"See '{docs_url}' for the prerequisites and how to install them.")
    logger.warning(msg)
    out: Dict[str, Any] = {
        "partial_run": True,
        "partial_reason": f"{reason} (See {docs_url})",
        "leaderboard_comparable": False,
    }
    out.update(detail)
    return out


# --------------------------------------------------------------------------------------------
# endpoint relay
# --------------------------------------------------------------------------------------------
#: A path segment the relay forwards: RFC 3986 unreserved characters only (no percent escape,
#: no '@', ':' or '\\'), so a request can name an endpoint path and nothing else.
_RELAY_SEGMENT = re.compile(r"^[A-Za-z0-9._~-]+$")
#: A query string the relay forwards (the same characters plus '=' and '&').
_RELAY_QUERY = re.compile(r"^[A-Za-z0-9._~=&-]*$")


class _NoRedirect:
    """urllib redirect handler that follows nothing: a 3xx from the endpoint is relayed as it
    is (status and body, without its Location), so the Authorization header the relay adds is
    never re-sent to another URL."""

    @staticmethod
    def build():
        import urllib.request

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None
        return NoRedirect


#: A 400 reply whose message says the request does not fit the served context window (vLLM:
#: "This model's maximum context length is ..." or "... is longer than the maximum model length
#: of ..."). Every relayed scaffold treats that as the conversation outgrowing the window, the
#: model's outcome (SWE-agent exit_context, MSWE-agent exit_cost), so the relay counts it apart
#: from the endpoint's failures.
CONTEXT_OVERFLOW_RE = re.compile(
    r"maximum context length|maximum model length|context length|context window|"
    r"context_length_exceeded", re.IGNORECASE)


def _reply_outcome(finish: Any, content: Any, reasoning: Any,
                   tool_calls: Any) -> Tuple[bool, bool]:
    """(cut off at a limit, empty after reasoning) of one reply choice: `finish_reason` is
    "length", or the choice carries reasoning but neither content nor a tool call."""
    text = content if isinstance(content, str) else ("" if content is None else str(content))
    return finish == "length", bool(reasoning) and not text.strip() and not tool_calls


#: What a reply that stopped on its own with reasoning but no content and no tool call is,
#: told from its generated token ids (classify_empty_reply):
#: * EMPTY_ANSWER_IN_THINKING: the generation holds no tool-call token; the model ended its turn
#:   with its answer inside the thinking channel, the model's outcome (base's
#:   ANSWER_IN_THINKING_POLICY);
#: * EMPTY_TOOL_CALL_DROPPED: the generation holds the served tokenizer's tool-call token (Gemma
#:   4: `<|tool_call>`, id 48): the model wrote a native tool call and the server's tool-call
#:   parser took it out of the reply (a scaffold that declares no tools gets nothing back), a
#:   server-parser artifact, never the model's answer (base's server-parser policy);
#: * EMPTY_UNCLASSIFIED: no token ids (a streamed reply, the first reply before the relay knows
#:   the endpoint is vLLM, a server that is not vLLM) or no single tool-call token in the served
#:   tokenizer, so the two cannot be told apart: never counted as the model's answer.
EMPTY_ANSWER_IN_THINKING = "answer_in_thinking"
EMPTY_TOOL_CALL_DROPPED = "tool_call_dropped"
EMPTY_UNCLASSIFIED = "unclassified"

#: The text of the tool-call opening token looked up in the served tokenizer (served_token_id).
TOOL_CALL_TOKEN_TEXT = "<|tool_call>"

_SERVED_TOKEN_IDS: Dict[Tuple[str, str], Optional[int]] = {}
_SERVED_TOKEN_LOCK = threading.Lock()


def served_token_id(server_root: str, text: str, key: Optional[str] = None,
                    timeout: float = 30.0) -> Optional[int]:
    """The id of `text` in the served model's tokenizer, asked once per (server, text) of the
    endpoint's /tokenize (vLLM; `server_root` is the URL without /v1): the id when `text` is
    exactly one token, else None (another tokenizer, or no /tokenize endpoint). Sampling is not
    involved; a lookup that fails is remembered as None."""
    import urllib.request
    root = (server_root or "").rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    cache_key = (root, text)
    with _SERVED_TOKEN_LOCK:
        if cache_key in _SERVED_TOKEN_IDS:
            return _SERVED_TOKEN_IDS[cache_key]
    found: Optional[int] = None
    try:
        req = urllib.request.Request(
            root + "/tokenize", method="POST",
            data=json.dumps({"prompt": text, "add_special_tokens": False}).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {key}"} if key else {})})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        tokens = data.get("tokens") if isinstance(data, dict) else None
        if (isinstance(tokens, list) and len(tokens) == 1 and isinstance(tokens[0], int)
                and not isinstance(tokens[0], bool)):
            found = tokens[0]
    except Exception as e:                                                  # noqa: BLE001
        logger.info("the served tokenizer's id of %r could not be read from %s/tokenize "
                    "(%s): replies empty after reasoning stay unclassified", text, root,
                    type(e).__name__)
    with _SERVED_TOKEN_LOCK:
        _SERVED_TOKEN_IDS[cache_key] = found
    return found


def classify_empty_reply(token_ids: Any, tool_call_id: Optional[int]) -> str:
    """EMPTY_ANSWER_IN_THINKING, EMPTY_TOOL_CALL_DROPPED or EMPTY_UNCLASSIFIED (see above) for a
    reply that stopped on its own with reasoning but no content and no tool call, from its
    generated `token_ids` and the served tokenizer's tool-call token id."""
    if tool_call_id is None or not isinstance(token_ids, (list, tuple)) or not token_ids:
        return EMPTY_UNCLASSIFIED
    return EMPTY_TOOL_CALL_DROPPED if tool_call_id in token_ids else EMPTY_ANSWER_IN_THINKING


#: The EndpointRelay failure kind of a request the endpoint stopped making progress on
#: (EndpointRelay._ProgressClock).
STALLED_NO_TOKENS = "stalled_no_tokens"


#: The endpoint failures the ENDPOINT produced, as EndpointRelay counts them: no reply
#: (`connection`), a 5xx reply (`http_5xx`), an error event inside a 2xx stream (`stream_error`)
#: and a 2xx completions reply that is not JSON (`unparsable_reply`), and a request the endpoint
#: stopped making progress on (`stalled_no_tokens`: no token of its reply for
#: EndpointRelay.IDLE_TIMEOUT_S while the server's engines stood still too). A working
#: endpoint answers a well-formed request with none of them, whoever sends it. A 4xx is the
#: endpoint rejecting the request it was sent, which a sender can provoke at will (a wrong model
#: name, a malformed body).
def endpoint_side_failure(kind: str) -> bool:
    """True when an EndpointRelay failure kind is one the endpoint itself produced (see above)."""
    return (kind in ("connection", "stream_error", "unparsable_reply", STALLED_NO_TOKENS)
            or kind.startswith("http_5"))


def _message_text(content: Any) -> Optional[str]:
    """The text of a chat message's content: the string itself, or the text parts of a list of
    content parts, concatenated."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [p.get("text") for p in content
                 if isinstance(p, dict) and isinstance(p.get("text"), str)]
        return "".join(parts) if parts else None
    return None


def conversation_fingerprint(messages: Any) -> Optional[str]:
    """"sha256:<16 hex>" of the text of the first user message of a chat conversation, or None
    when there is none. For an agent rollout that is the task's instance prompt, the same in
    every model request of the rollout (their `messages`) and in its trajectory, and different
    from any other task's: EndpointRelay tags each endpoint failure with the fingerprint of the
    request it failed, so a failure can be attributed to one rollout."""
    if not isinstance(messages, list):
        return None
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            text = _message_text(msg.get("content"))
            if text:
                return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            return None
    return None


class _StreamTally:
    """The choices of one streamed (text/event-stream) completion reply, read from its `data:`
    lines as they pass through the relay."""

    def __init__(self) -> None:
        self._buf = b""
        self._choices: Dict[int, Dict[str, Any]] = {}
        self.errors = 0
        #: the `data:` events read so far (an SSE comment such as vLLM's ": keep-alive" is not
        #: one): the reply's progress, for EndpointRelay's idle timeout
        self.events = 0

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            self._line(line.strip())

    def _line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        payload = line[5:].strip()
        if payload:
            self.events += 1
        if not payload or payload == b"[DONE]":
            return
        try:
            data = json.loads(payload)
        except ValueError:
            return
        if not isinstance(data, dict):
            return
        if data.get("error"):
            self.errors += 1
            return
        for ch in data.get("choices") or []:
            if not isinstance(ch, dict):
                continue
            st = self._choices.setdefault(int(ch.get("index") or 0), {
                "content": False, "reasoning": False, "tools": False, "finish": None})
            delta = ch.get("delta") if isinstance(ch.get("delta"), dict) else {}
            text = delta.get("content") if delta else ch.get("text")
            st["content"] = st["content"] or bool(str(text or "").strip())
            st["reasoning"] = st["reasoning"] or bool(delta.get("reasoning_content")
                                                      or delta.get("reasoning"))
            st["tools"] = st["tools"] or bool(delta.get("tool_calls"))
            if ch.get("finish_reason"):
                st["finish"] = ch["finish_reason"]

    def outcomes(self) -> List[Tuple[bool, bool]]:
        if self._buf.strip():
            self._line(self._buf.strip())
            self._buf = b""
        return [_reply_outcome(st["finish"], "x" if st["content"] else "", st["reasoning"],
                               st["tools"]) for st in self._choices.values()]


#: The _ClientWatch of the request the current relay handler thread is forwarding: the relay's
#: endpoint connections (_tracked_handlers) hand it their socket when they connect.
_RELAY_WATCH = threading.local()


class _ClientWatch:
    """Watches the client connection of one relayed request while the relay waits for, reads or
    sends the endpoint's reply. When the client closes its connection (end of file, or a reset,
    seen on the socket by a peek that never consumes a byte) the watch shuts down the endpoint
    connection of the request (`attach`), so the served endpoint sees its client leave and stops
    generating the abandoned reply (vLLM aborts a request whose connection closes), and records
    it in `aborted`. A client that sends more bytes before the reply (a pipelined request) ends
    the watch without an abort. The watch starts after the request body was read and ends with
    `stop` (its thread exits within POLL_S)."""

    POLL_S = 0.25

    def __init__(self, conn: Any):
        self._conn = conn
        self._lock = threading.Lock()
        self._sock: Any = None
        self._done = threading.Event()
        self.aborted = False
        #: the relay gave up on the endpoint's reply itself (cut_upstream: no progress)
        self._cut = False
        self._thread = threading.Thread(target=self._run, name="relay-client-watch", daemon=True)
        self._thread.start()

    def attach(self, sock: Any) -> None:
        """The endpoint connection of the request (shut down at once when the client already
        left, or the relay already gave up on the reply)."""
        with self._lock:
            self._sock = sock
            aborted = self.aborted or self._cut
        if aborted:
            self._shutdown(sock)

    def cut_upstream(self) -> None:
        """Shut down the endpoint connection of the request (now, or when it connects), so the
        served endpoint stops generating a reply the relay gave up on (EndpointRelay's idle
        timeout). The client connection is left to the relay, which still answers it."""
        with self._lock:
            self._cut = True
            sock = self._sock
        if sock is not None:
            self._shutdown(sock)

    @staticmethod
    def _shutdown(sock: Any) -> None:
        import socket
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except (OSError, ValueError, AttributeError):
            pass

    def _client_left(self) -> Optional[bool]:
        """True: the client closed or reset its connection; False: it sent bytes; None: nothing
        to read yet."""
        import select
        import socket
        # poll, not select: select() refuses any fd >= 1024 (FD_SETSIZE), which a long gbench
        # process reaches, and the watch then stopped silently on its first look.
        try:
            fd = self._conn.fileno()
            if fd < 0:                      # the connection was closed: the request is over
                return False
            poller = select.poll()
            poller.register(fd, select.POLLIN | select.POLLERR | select.POLLHUP)
            ready = poller.poll(int(self.POLL_S * 1000))
        except (OSError, ValueError):      # the connection was closed: the request is over
            return False
        if not ready:
            return None
        try:
            return not self._conn.recv(1, socket.MSG_PEEK)
        except (BlockingIOError, InterruptedError):
            return None
        except OSError:                     # reset by the client
            return True

    def _run(self) -> None:
        while not self._done.is_set():
            left = self._client_left()
            if left is None:
                continue
            if left and not self._done.is_set():
                with self._lock:
                    self.aborted = True
                    sock = self._sock
                if sock is not None:
                    self._shutdown(sock)
            return

    def stop(self) -> None:
        self._done.set()


def _tracked_handlers() -> Tuple[Any, Any]:
    """urllib HTTP and HTTPS handlers whose connections hand their socket, once connected, to the
    _ClientWatch of the relay handler thread that opened them (_RELAY_WATCH)."""
    import http.client
    import urllib.request

    def _attach(conn: Any) -> None:
        watch = getattr(_RELAY_WATCH, "watch", None)
        if watch is not None and getattr(conn, "sock", None) is not None:
            watch.attach(conn.sock)

    class _HTTPConnection(http.client.HTTPConnection):
        def connect(self):
            super().connect()
            _attach(self)

    class _HTTPSConnection(http.client.HTTPSConnection):
        def connect(self):
            super().connect()
            _attach(self)

    class _HTTPHandler(urllib.request.HTTPHandler):
        def http_open(self, req):
            return self.do_open(_HTTPConnection, req)

    class _HTTPSHandler(urllib.request.HTTPSHandler):
        def https_open(self, req):
            return self.do_open(_HTTPSConnection, req, context=self._context)
    return _HTTPHandler, _HTTPSHandler


class _StreamAssembler:
    """The non-streamed chat completion reply of a request the relay sent to the endpoint as a
    stream (EndpointRelay.stream_body: `stream` true, `stream_options.include_usage`), rebuilt
    from the stream's `data:` events with the fields and layout of vLLM's non-streamed reply
    (vllm/entrypoints/openai/chat_completion/serving.py chat_completion_full_generator and
    protocol.ChatMessage, vLLM 0.26.1): per choice the message's role, content (the content
    deltas joined; null when they join to nothing, as vLLM's parser returns it), reasoning (the
    `reasoning` or `reasoning_content` deltas joined, under the key(s) the stream used; null when
    empty), tool_calls (only when there is one: per tool index its id, type and function name,
    and its argument deltas joined), refusal / annotations / audio / function_call (null),
    logprobs (null: a request asking for logprobs is never streamed for the client,
    EndpointRelay.stream_body), finish_reason and stop_reason (of the choice's last event),
    token_ids (each event's ids joined, when the request asked for them; null otherwise);
    the reply's id, created, model, prompt_token_ids and prompt_text (of the first event that
    carries them), usage, system_fingerprint and metrics (of the usage event, the last before
    [DONE]). An error event becomes the error reply vLLM sends for a non-streamed request (its
    JSON body, with the status its `code` names, else 500).

    Not the same reply in every case: vLLM's streaming path is not its non-streamed path. In the
    2026-10-11 live check against vLLM 0.26.1rc1 (gemma-4-26B-A4B-it, temperature 0, 13 requests:
    thinking on and off, with and without tools, cut off at max_tokens, after a tool result),
    every request generated the same token ids both ways, 11 rebuilt replies equalled the
    non-streamed reply in every field but the per-request ids (id, created, tool call ids), and 2
    did not: content written before a tool call kept its trailing "\n\n" in the stream (stripped
    in the non-streamed reply), and with tool_choice "required" the stream finished "stop"
    where the non-streamed reply says "tool_calls". So EndpointRelay only streams a non-streamed
    request when `stream_upstream` is on (off by default, docs/relay.md)."""

    def __init__(self) -> None:
        self._buf = b""
        self.events = 0
        self.done = False
        self.unparsable = 0
        self.error: Optional[Dict[str, Any]] = None
        self._head: Dict[str, Any] = {}
        self._choices: Dict[int, Dict[str, Any]] = {}
        self._usage_event: Optional[Dict[str, Any]] = None
        self._reasoning_keys: List[str] = []
        self._fingerprint: Any = None

    def feed(self, chunk: bytes) -> None:
        self._buf += chunk
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            self._line(line.strip())

    def finish(self) -> None:
        if self._buf.strip():
            self._line(self._buf.strip())
        self._buf = b""

    def _line(self, line: bytes) -> None:
        if not line.startswith(b"data:"):
            return
        payload = line[5:].strip()
        if not payload:
            return
        self.events += 1
        if payload == b"[DONE]":
            self.done = True
            return
        try:
            data = json.loads(payload)
        except ValueError:
            self.unparsable += 1
            return
        if not isinstance(data, dict):
            self.unparsable += 1
            return
        if data.get("error") is not None:
            if self.error is None:
                self.error = data
            return
        for key in ("id", "created", "model", "prompt_token_ids", "prompt_text"):
            if key in data and key not in self._head:
                self._head[key] = data[key]
        if data.get("system_fingerprint") is not None:
            self._fingerprint = data["system_fingerprint"]
        if data.get("usage") is not None and not data.get("choices"):
            self._usage_event = data
        for ch in data.get("choices") or []:
            if isinstance(ch, dict):
                self._choice(ch)

    def _choice(self, ch: Dict[str, Any]) -> None:
        st = self._choices.setdefault(int(ch.get("index") or 0), {
            "role": None, "content": [], "reasoning": [], "tools": {}, "finish": None,
            "stop_reason": None, "token_ids": None, "finished": False})
        delta = ch.get("delta") if isinstance(ch.get("delta"), dict) else {}
        if delta.get("role") and st["role"] is None:
            st["role"] = delta["role"]
        if isinstance(delta.get("content"), str):
            st["content"].append(delta["content"])
        # a server that sends both names sends the same text under each: read it once
        for key in ("reasoning", "reasoning_content"):
            if key in delta and key not in self._reasoning_keys:
                self._reasoning_keys.append(key)
        for key in ("reasoning", "reasoning_content"):
            if isinstance(delta.get(key), str):
                st["reasoning"].append(delta[key])
                break
        for tc in delta.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            slot = st["tools"].setdefault(int(tc.get("index") or 0), {
                "id": None, "type": None, "name": [], "arguments": []})
            if tc.get("id") and slot["id"] is None:
                slot["id"] = tc["id"]
            if tc.get("type") and slot["type"] is None:
                slot["type"] = tc["type"]
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            if isinstance(fn.get("name"), str):
                slot["name"].append(fn["name"])
            if isinstance(fn.get("arguments"), str):
                slot["arguments"].append(fn["arguments"])
        if isinstance(ch.get("token_ids"), list):
            st["token_ids"] = (st["token_ids"] or []) + list(ch["token_ids"])
        if ch.get("finish_reason") is not None:
            st["finish"] = ch["finish_reason"]
            st["stop_reason"] = ch.get("stop_reason")
            st["finished"] = True

    def complete(self) -> bool:
        """Whether the stream ended as a whole reply: an error event, or every choice finished
        and the usage event or [DONE] arrived."""
        if self.error is not None:
            return True
        return (bool(self._choices) and all(st["finished"] for st in self._choices.values())
                and (self._usage_event is not None or self.done))

    def reply(self) -> Tuple[int, bytes]:
        """(status, JSON body) of the non-streamed reply (see the class docstring)."""
        if self.error is not None:
            err = self.error.get("error")
            code = err.get("code") if isinstance(err, dict) else None
            status = code if isinstance(code, int) and 400 <= code <= 599 else 500
            return status, json.dumps(self.error).encode("utf-8")
        choices = []
        for index in sorted(self._choices):
            st = self._choices[index]
            msg: Dict[str, Any] = {"role": st["role"] or "assistant",
                                   "content": "".join(st["content"]) or None,
                                   "refusal": None, "annotations": None, "audio": None,
                                   "function_call": None}
            tools = [{"id": slot["id"], "type": slot["type"] or "function",
                      "function": {"name": "".join(slot["name"]),
                                   "arguments": "".join(slot["arguments"])}}
                     for _, slot in sorted(st["tools"].items())]
            if tools:
                msg["tool_calls"] = tools
            for key in self._reasoning_keys or ["reasoning"]:
                msg[key] = "".join(st["reasoning"]) or None
            choices.append({"index": index, "message": msg, "logprobs": None,
                            "finish_reason": st["finish"], "stop_reason": st["stop_reason"],
                            "token_ids": st["token_ids"], "routed_experts": None})
        usage_event = self._usage_event or {}
        raw = usage_event.get("usage") if isinstance(usage_event.get("usage"), dict) else None
        usage = None
        if raw is not None:
            usage = {key: raw.get(key) for key in (
                "prompt_tokens", "total_tokens", "completion_tokens", "prompt_tokens_details",
                "completion_tokens_details")}
            usage.update({k: v for k, v in raw.items() if k not in usage})
        body = {"id": self._head.get("id"), "object": "chat.completion",
                "created": self._head.get("created"), "model": self._head.get("model"),
                "choices": choices, "service_tier": None,
                "system_fingerprint": self._fingerprint, "usage": usage,
                "prompt_logprobs": None, "prompt_token_ids": self._head.get("prompt_token_ids"),
                "prompt_text": self._head.get("prompt_text"), "kv_transfer_params": None,
                "ec_transfer_params": None, "metrics": usage_event.get("metrics")}
        return 200, json.dumps(body).encode("utf-8")


class _UpstreamPump:
    """Opens one relayed request to the endpoint and reads its reply on a thread of its own,
    putting what happens on `events` as it happens: ("open", response) (an HTTPError response
    too), ("error", exception) when no reply came, then ("chunk", bytes) for each read and
    ("eof", None) at its end, or ("read_error", exception). The relay handler waits on `events`
    with its _ProgressClock, so a reply that stops making progress is noticed while the read is
    still blocked; the read itself only ends at the end of the reply, at `timeout_s` (the
    request's total safety cap) or when the endpoint connection is shut down (_ClientWatch:
    the client left, or cut_upstream)."""

    def __init__(self, opener: Any, req: Any, watch: Any, timeout_s: float):
        import queue
        self.events: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, args=(opener, req, watch, timeout_s),
                                        name="relay-upstream", daemon=True)
        self._thread.start()

    def _run(self, opener: Any, req: Any, watch: Any, timeout_s: float) -> None:
        import urllib.error
        _RELAY_WATCH.watch = watch          # the endpoint connection reports to the watch
        try:
            try:
                resp = opener.open(req, timeout=timeout_s)
            except urllib.error.HTTPError as e:
                resp = e
            except Exception as e:                                          # noqa: BLE001
                self.events.put(("error", e))
                return
            self.events.put(("open", resp))
            try:
                while True:
                    chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(65536)
                    if not chunk:
                        self.events.put(("eof", None))
                        return
                    self.events.put(("chunk", chunk))
            except Exception as e:                                          # noqa: BLE001
                self.events.put(("read_error", e))
        finally:
            _RELAY_WATCH.watch = None


class RelayEnvError(RuntimeError):
    """A malformed relay environment knob (GBENCH_RELAY_IDLE_TIMEOUT_S,
    GBENCH_RELAY_EXPECTED_SHAPE, GBENCH_SMOKE_RELAY_REPLIES). Not a ValueError on purpose: the
    suites turn a ValueError of the EndpointRelay constructor into "the endpoint URL cannot be
    relayed" (tau_common) or "it could not listen on" (skillsbench), which would hide a typo in
    a variable behind an endpoint or bind failure (2026-10-11 relay review)."""


class _ProgressClock:
    """When one relayed request stops making progress (EndpointRelay's idle timeout).

    Progress is (a) an event of the request's own reply stream (`progress`, called by the relay
    for each `data:` event; an SSE comment such as vLLM's keep-alive is not one), or (b) the
    served endpoint's engines moving (EndpointRelay.server_progress, per engine: vLLM's /metrics
    token counters, which advance when a request is scheduled and at each output token, or its
    KV-cache usage gauge, which changes at every chunk of a prefill: the token counters alone
    stand still through a whole prefill, 12.5 s for a 254K-token prompt on 26B-A4B, 2026-10-11)
    with no engine that holds requests standing still (EndpointRelay.server_progressed). After `idle_s` without (a), the clock reads (b) twice,
    EndpointRelay.PROGRESS_CONFIRM_S apart: counters that advanced mean the server is working
    (the request waits in its queue under load, or was preempted) and the request waits another
    `idle_s` (counted, EndpointRelay.note_idle_tolerated); counters that stood still mean the
    request STALLED. Without readable counters, a request whose reply streams (`chunked`) stalls
    after `idle_s` without an event; one whose reply arrives whole (nothing to see before its
    end) waits for its total cap. The total cap (`deadline`, time.monotonic()) ends the wait
    either way (TIMEOUT)."""

    STALLED = STALLED_NO_TOKENS
    TIMEOUT = "timeout"

    def __init__(self, relay: Any, idle_s: float, deadline: float, chunked: bool):
        self._relay = relay
        self.idle_s = float(idle_s)
        self.deadline = deadline
        self.chunked = chunked
        self._last = time.monotonic()
        #: the time of the request's last own progress (a), for the idle length recorded
        self._own = self._last
        self._sample: Any = None
        self._confirm_at: Optional[float] = None

    def progress(self) -> None:
        self._last = self._own = time.monotonic()
        self._sample = None
        self._confirm_at = None

    def wait_s(self) -> float:
        """Seconds until the clock must look again."""
        due = self._confirm_at if self._confirm_at is not None else self._last + self.idle_s
        return max(0.0, min(self.deadline, due) - time.monotonic())

    def expired(self) -> Optional[str]:
        """None (keep waiting), STALLED or TIMEOUT, when the wait of wait_s() ran out."""
        now = time.monotonic()
        if now >= self.deadline:
            return self.TIMEOUT
        if self._confirm_at is None:
            if now < self._last + self.idle_s:
                return None
            sample = self._relay.server_progress()
            if sample is None:
                if self.chunked:
                    return self.STALLED
                self._last = now            # nothing to tell from: look again in idle_s
                return None
            self._sample = sample
            self._confirm_at = now + self._relay.PROGRESS_CONFIRM_S
            return None
        if now < self._confirm_at:
            return None
        sample = self._relay.server_progress()
        if (sample is not None and self._sample is not None
                and self._relay.server_progressed(self._sample, sample)):
            if self.chunked:            # a whole reply has nothing to show before its end
                self._relay.note_idle_tolerated(now - self._own)
            self._last = now
            self._sample = None
            self._confirm_at = None
            return None
        if sample is None and not self.chunked:
            self._last = now
            self._sample = None
            self._confirm_at = None
            return None
        return self.STALLED


def _reply_incomplete(resp: Any) -> bool:
    """True when a reply whose reading ended without an error still had body bytes due (its
    Content-Length not reached: http.client ends such a read with b"" when the connection
    closes)."""
    fp = getattr(resp, "fp", None)
    for r in (resp, fp):
        length = getattr(r, "length", None)
        if isinstance(length, int) and not isinstance(length, bool):
            return length > 0
    return False


def _is_timeout(error: Optional[BaseException]) -> bool:
    """Whether `error` (from urllib) is a socket timeout, also when wrapped in a URLError."""
    import socket
    reason = getattr(error, "reason", None)
    return isinstance(error, (socket.timeout, TimeoutError)) or isinstance(
        reason, (socket.timeout, TimeoutError))


#: Every EndpointRelay started in this process, in order (relays_started_since): a suite that
#: raises before it returns a result still gets its relays' records attached by the eval runner
#: (2026-10-11: terminal_bench raised on the smoke stop and its result kept no relay evidence).
_STARTED_RELAYS: List["EndpointRelay"] = []


def relay_count() -> int:
    """How many EndpointRelays this process has started (a mark for relays_started_since)."""
    return len(_STARTED_RELAYS)


def relays_started_since(mark: int) -> List[Dict[str, Any]]:
    """The records of the EndpointRelays started after `mark` (relay_count)."""
    out = []
    for relay in _STARTED_RELAYS[mark:]:
        try:
            out.append(relay.record())
        except Exception:                                               # noqa: BLE001
            pass
    return out


class EndpointRelay:
    """Relays an agent's model requests to one fixed served endpoint.

    For agents gbench runs where the endpoint's key must not go (a task sandbox whose shell the
    model drives, a harness that logs its client configuration) or whose own client cannot send
    a field gbench must set (the served model's reasoning switch). The relay is a thread of the
    gbench process listening on `bind_host`; the agent gets its address instead of the
    endpoint's.

    The endpoint is fixed when the relay is made: its scheme, host and port come from
    `upstream` (urllib.parse.urlsplit) and every target URL is built with urlunsplit from those
    parts and the checked request path, never by concatenating strings, so no request can reach
    another host or port. A request is forwarded only when
      * its method is GET or POST, and
      * its path starts with /v1/ and every segment is made of RFC 3986 unreserved characters
        (letters, digits, '-', '.', '_', '~'), with no empty, '.' or '..' segment, no percent
        escape and no fragment; its query uses the same characters plus '=' and '&'.
      * its path (without the query) is one of `allowed_paths`, by default ALLOWED_PATHS: the
        endpoints the relayed agents use (chat completions, completions, the model list), so
        no other route of the served endpoint (e.g. vLLM's adapter-loading routes) can be
        reached with the key's authority;
      * or, instead of the two conditions above, its path (without the query) is one of
        `root_paths` (none by default): routes a caller names that the server serves outside
        /v1, at its root (vLLM's POST /tokenize), with the same segment and query checks, and
        for POST only. A root path may not lie under /v1/ or end in a completions suffix
        (PATCHED_SUFFIXES), so its body is never patched and its reply never counted as a
        completions reply.
    Anything else is answered 400 (405 for another method) and never forwarded. The request's
    path (/v1/... or a root path) is appended to the endpoint's base path (the part before its
    /v1).

    Redirects are not followed (`_NoRedirect`). The client's own Authorization header is
    dropped; with `key` the relay sends `Authorization: Bearer <key>`, and the key appears
    nowhere else: not in a reply, a log line or an error message. With `body_fields`, a JSON
    POST body to .../chat/completions or .../completions gets those fields set (a dict value is
    merged into the body's dict of the same name), e.g. {"chat_template_kwargs":
    {"enable_thinking": True}}. The reply's status, Content-Type and body go back unchanged,
    streamed as they arrive (a completions reply that is not an event stream, and any reply that
    is not 2xx, is read in full first, then sent on unchanged).

    The relay also counts what it sees (`record`): endpoint failures in `upstream_errors` (a
    request the endpoint did not answer, `connection`; a reply that is not 2xx, `http_<status>`,
    except a 400 that names the context window, counted in `context_length_rejections`; a 2xx
    completions reply that is not JSON, `unparsable_reply`; an error event inside a stream,
    `stream_error`). The relay cannot tell an agent's requests from requests the model's own
    commands send from the same sandbox, so a failure is attributed to a rollout only when the
    endpoint itself produced it (endpoint_side_failure: no reply, a 5xx, a stream error, an
    unparsable 2xx; never a 4xx, which the sender can provoke) on a request carrying that
    rollout's conversation (conversation_fingerprint of the request's `messages`):
    `failure_times` / `failures_between` count those, and `succeeded_after` says whether a later
    request of the same conversation got a reply. A root-path request (e.g. a /tokenize count
    before a model request) is not a model request: its failures are counted apart, in
    `root_path_errors` (`connection`, `http_<status>`), never among `upstream_errors`,
    `context_length_rejections` or the failure times. And, over the choices of the 2xx
    completions replies (`completion_replies`),
    those that stopped at a limit (`finish_reason` "length", `replies_cut_off`) and those that
    carry reasoning but neither content nor a tool call (`replies_empty_after_reasoning`). Those of
    them that stopped on their own are split by their generated token ids (classify_empty_reply,
    the served tokenizer's tool-call token looked up once with served_token_id): no tool-call
    token, the answer inside the thinking channel (`replies_answer_in_thinking`, the model's
    outcome); a tool-call token, a native tool call the server's tool-call parser removed
    (`replies_tool_call_dropped`, a server-parser artifact); no token ids or no tool-call token
    to look for (`replies_empty_unclassified`), never counted as the model's answer. A
    reply whose client closed the connection before the relay had sent all of it (a streamed
    reply the client stops reading, e.g. at a stop sequence it applies itself, or a request the
    client gives up on while the relay still waits for the endpoint, e.g. an OpenAI SDK at its
    read timeout, which then resends it) ends there: the relay closes the endpoint's connection
    of that request, so the served endpoint stops generating, and counts it in
    `client_closed_streams`, not an endpoint failure (a reply cut off this way is not one of the
    completion_replies either; a reply read in full first was already counted). The relay
    notices the client leaving while it waits as well as while it sends (_ClientWatch: the
    client's end of file or reset, seen without reading the connection), so an abandoned
    non-streamed reply, whose endpoint answers only when it has generated all of it, is stopped
    when the client leaves, not generated to its end next to the client's retry. (A client that
    half-closes its sending side after the request counts as leaving; the agents relayed here
    do not.) The age of each request at the moment its client left is kept (`client_close_ages`,
    a histogram); closes within CLIENT_TIMEOUT_WINDOW_S of a known client-side timeout
    (`client_timeout_suspect_s`, by default 600 s, litellm's and the OpenAI SDK's fallback, and
    1800 s, that times the SDK's 3 tries) are counted in `client_timeouts_suspected`: a client
    that gives up on a long reply at its own timeout and asks again is a hidden cap on the
    measurement (2026-10-10 smoke run: SWE-agent's litellm closed trimesh turns at 600 s at
    least 12 times, each resampled).

    Progress, not duration (docs/relay.md). A model request (a completions path) fails only when
    it stops making progress: no event of its reply for IDLE_TIMEOUT_S (300 s,
    GBENCH_RELAY_IDLE_TIMEOUT_S, or `idle_timeout_s`) while the server's engines stood still
    too (server_progress: token counters and KV-cache usage; _ProgressClock); it is then counted as
    `stalled_no_tokens` (an endpoint-side failure), the endpoint's connection is shut down and
    the client gets a 504. Its total duration is bounded only by a safety cap far above any
    reply (request_total_s: request_timeout_s of the request's max_tokens, or of the served
    window when it sends none, and never below TIMEOUT_S). A streamed reply shows its own
    progress event by event; a non-streamed reply arrives whole at its end, so for it the
    server's counters are the only progress there is, and the request is sent as the client
    sent it: the reply stays the endpoint's own non-streamed reply, byte for byte. With
    `stream_upstream` (GBENCH_RELAY_STREAM_UPSTREAM=1; off by default because vLLM's streaming
    path does not always rebuild the same reply, _StreamAssembler) the relay sends a
    non-streamed chat completions request to a vLLM endpoint as a stream (stream_body) and
    rebuilds the reply from its events. Counted in `upstream_transport`.

    Request shape (`expected_shape`, else the JSON of GBENCH_RELAY_EXPECTED_SHAPE; docs/relay.md):
    each completions request's body as it goes to the endpoint (after `body_fields`) is compared
    with the expected temperature, thinking switch and token budget; violations are counted per
    field in `shape_check`, with the first examples, and never block a request.

    Smoke episodes (`smoke_replies`=N, else GBENCH_SMOKE_RELAY_REPLIES=N, which gbench --smoke
    sets for its agent suites, unless `smoke_from_env` is off; docs/relay.md): per conversation (conversation_fingerprint of the request's
    messages) the relay serves the first N completed model replies and answers the next model
    request of that conversation with HTTP 400 GBENCH_SMOKE_STOP (never sent to the endpoint,
    never an endpoint failure). A next request whose messages grew past those of the last served
    request is the proof the agent read the reply, ran its action and came back
    (`smoke_second_requests`). Outside a smoke run, without smoke_replies, nothing changes.
    """

    #: Seconds one relayed request may take (one model turn) at the least: a model request's
    #: safety cap is the larger of this and request_total_s's size-derived cap.
    TIMEOUT_S = 3600
    #: Seconds a model request may go without an event of its reply (and without the server's
    #: token counters advancing) before it counts as stalled (_ProgressClock).
    IDLE_TIMEOUT_S = 300.0
    #: Seconds between the two reads of the server's token counters that tell a busy server
    #: from a stalled one (_ProgressClock).
    PROGRESS_CONFIRM_S = 30.0
    #: Seconds added to request_timeout_s in a model request's safety cap (request_total_s).
    TOTAL_CAP_MARGIN_S = 120
    #: Client-side per-request timeouts a client close near which is suspect (client_close_ages):
    #: litellm's 600 s fallback (also the OpenAI SDK's default), 1800 s (3 tries of it), and
    #: 3660 s, the TIMEOUT_S + 60 that multi_swe_bench (MSWE_AGENT_CLIENT_TIMEOUT_S) sizes its
    #: agent's client timeout to, now below the relay's own safety cap (request_total_s).
    CLIENT_TIMEOUT_SUSPECT_S = (600.0, 1800.0, 3660.0)
    #: Seconds either side of a suspect timeout that a client close counts as near it.
    CLIENT_TIMEOUT_WINDOW_S = 10.0
    #: Upper edges (seconds) of the client_close_ages histogram buckets (the last is open).
    CLOSE_AGE_EDGES = (60, 300, 590, 610, 1790, 1810, 3600)
    #: How many examples shape_check and client_timeout_suspect_ages keep.
    EXAMPLE_LIMIT = 5
    #: The request fields expected_shape can check (check_shape).
    SHAPE_FIELDS = ("temperature", "enable_thinking", "max_tokens", "max_tokens_at_most")
    #: The error message and type of the smoke stop reply (smoke_replies).
    SMOKE_STOP_MESSAGE = "GBENCH_SMOKE_STOP"
    SMOKE_STOP_TYPE = "gbench_smoke_stop"
    #: Request paths whose JSON bodies get `body_fields`.
    PATCHED_SUFFIXES = ("/chat/completions", "/completions")
    #: The endpoint paths the relay forwards unless a caller names others: what the agents
    #: gbench relays call (litellm and the OpenAI SDK: chat completions, completions, models).
    ALLOWED_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/models")

    def __init__(self, upstream: str, bind_host: str, key: Optional[str] = None,
                 body_fields: Optional[Dict[str, Any]] = None, name: str = "endpoint-relay",
                 allowed_paths: Optional[Sequence[str]] = None,
                 root_paths: Sequence[str] = (),
                 reask: Optional[Callable[[Any], bool]] = None, reask_limit: int = 0,
                 timeout_s: Optional[float] = None,
                 expected_shape: Optional[Dict[str, Any]] = None,
                 smoke_replies: Optional[int] = None,
                 stream_upstream: Optional[bool] = None,
                 idle_timeout_s: Optional[float] = None,
                 client_timeout_suspect_s: Optional[Sequence[float]] = None,
                 smoke_from_env: bool = True):
        import urllib.parse
        parts = urllib.parse.urlsplit((upstream or "").strip())
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError("the relay needs an http(s) endpoint URL with a host")
        host = parts.hostname
        if ":" in host:
            host = f"[{host}]"
        self._scheme = parts.scheme
        self._netloc = f"{host}:{parts.port}" if parts.port else host
        base = parts.path.rstrip("/")
        self._base_path = base[:-3] if base.endswith("/v1") else base
        self.upstream = urllib.parse.urlunsplit((self._scheme, self._netloc,
                                                 self._base_path + "/v1", "", ""))
        self.bind_host = bind_host
        self._key = key
        self._body_fields = dict(body_fields or {})
        # An agent suite's request that carries no max_tokens gets the run's --max-output-tokens
        # (swe_agentic.agent_max_tokens_for_suite; the user's decision of 2026-10-11).
        from .swe_agentic import agent_max_tokens_for_suite
        self._fill_max_tokens = agent_max_tokens_for_suite()
        self.max_tokens_filled = 0
        self._name = name
        self.allowed_paths = tuple(allowed_paths if allowed_paths is not None
                                   else self.ALLOWED_PATHS)
        if isinstance(root_paths, str):
            raise ValueError(f"root_paths is a sequence of paths, not the string {root_paths!r}")
        self.root_paths = tuple(root_paths or ())
        for path in self.root_paths:
            segments = path.split("/")[1:]
            if (not path.startswith("/") or path.startswith("/v1/") or path == "/v1"
                    or path.endswith(self.PATCHED_SUFFIXES)
                    or any(not s or s in (".", "..") or not _RELAY_SEGMENT.match(s)
                           for s in segments)):
                raise ValueError(f"a root path is a route outside /v1 that is not a completions "
                                 f"route, not {path!r}")
        self._server = None
        self._thread = None
        self._port = 0
        self.requests_relayed = 0
        self.requests_refused = 0
        #: the root-path requests among requests_relayed, and their failures by kind
        self.root_requests_relayed = 0
        self.root_path_errors: Dict[str, int] = {}
        import threading
        self._lock = threading.Lock()
        self.upstream_errors: Dict[str, int] = {}
        #: (time, conversation fingerprint or None) of each endpoint-side failure
        #: (endpoint_side_failure), for failure_times / failures_between.
        self._failure_times: List[Tuple[float, Optional[str]]] = []
        #: conversation fingerprint -> time of the last 2xx completions reply to a request of it.
        self._last_success: Dict[str, float] = {}
        self.context_length_rejections = 0
        self.completion_replies = 0
        self.replies_cut_off = 0
        self.replies_empty_after_reasoning = 0
        #: the replies empty after reasoning that stopped on their own (not at a length limit),
        #: by classify_empty_reply: the answer inside the thinking channel (base's
        #: ANSWER_IN_THINKING_POLICY), a native tool call the server's parser removed, or not
        #: told apart (no token ids)
        self.replies_answer_in_thinking = 0
        self.replies_tool_call_dropped = 0
        self.replies_empty_unclassified = 0
        #: the served tokenizer's tool-call token id (served_token_id), looked up on the first
        #: reply that needs it; False until then
        self._tool_call_id: Any = False
        #: evidence for the first EMPTY_SAMPLE_LIMIT replies that stopped (not at a length limit)
        #: with reasoning and no content (keep_empty_sample)
        self.empty_after_reasoning_samples: List[Dict[str, Any]] = []
        #: whether the endpoint is vLLM (note_server), None until a completions reply shows it
        self._is_vllm: Optional[bool] = None
        self._probed = False
        self._probe_lock = threading.Lock()
        self.client_closed_streams = 0
        #: the upstream harness's re-ask rule (reask_reply): a non-streamed 2xx completions reply
        #: it matches is asked again, the same request, up to reask_limit times
        self._reask = reask
        self._reask_limit = max(0, int(reask_limit or 0))
        self.requests_reasked = 0
        self.reasks_sent = 0
        self.reasks_exhausted = 0
        self.reasks_cut_short = 0
        self.cut_off_not_reasked = 0
        if timeout_s:
            self.TIMEOUT_S = timeout_s
        # -- progress (IDLE_TIMEOUT_S, _ProgressClock) and the upstream transport (stream_body)
        if idle_timeout_s is None and (os.environ.get("GBENCH_RELAY_IDLE_TIMEOUT_S") or "").strip():
            raw = os.environ["GBENCH_RELAY_IDLE_TIMEOUT_S"]
            try:
                idle_timeout_s = float(raw)
            except ValueError:
                idle_timeout_s = None
            if idle_timeout_s is None or not idle_timeout_s > 0:
                raise RelayEnvError(f"GBENCH_RELAY_IDLE_TIMEOUT_S is a positive number of "
                                    f"seconds, not {raw!r}")
        if idle_timeout_s is not None:
            if not float(idle_timeout_s) > 0:
                raise ValueError(f"the relay's idle timeout is a positive number of seconds, "
                                 f"not {idle_timeout_s!r}")
            self.IDLE_TIMEOUT_S = float(idle_timeout_s)
        if stream_upstream is None:
            stream_upstream = os.environ.get("GBENCH_RELAY_STREAM_UPSTREAM", "").strip().lower() in (
                "1", "true", "yes", "on")
        self._stream_upstream = bool(stream_upstream)
        self.requests_streamed_upstream = 0
        self.replies_assembled = 0
        self.idle_waits_tolerated = 0
        self.longest_tolerated_idle_s = 0.0
        self.largest_request_cap_s = 0.0
        self._progress_lock = threading.Lock()
        self._progress_cache: Optional[Tuple[float, Any]] = None
        # -- client closes (count_client_closed)
        self._suspect_s = tuple(float(x) for x in (
            client_timeout_suspect_s if client_timeout_suspect_s is not None
            else self.CLIENT_TIMEOUT_SUSPECT_S))
        self._close_ages: List[float] = []
        self.client_timeouts_suspected = 0
        # -- request shape (check_shape); GBENCH_RELAY_EXPECTED_SHAPE (JSON) when none is given
        self._shape_source = "caller" if expected_shape is not None else None
        if expected_shape is None and (os.environ.get("GBENCH_RELAY_EXPECTED_SHAPE") or "").strip():
            self._shape_source = "GBENCH_RELAY_EXPECTED_SHAPE"
            try:
                expected_shape = json.loads(os.environ["GBENCH_RELAY_EXPECTED_SHAPE"])
            except ValueError as e:
                raise RelayEnvError(f"GBENCH_RELAY_EXPECTED_SHAPE is not JSON: {e}") from None
            if not isinstance(expected_shape, dict):
                raise RelayEnvError(f"GBENCH_RELAY_EXPECTED_SHAPE must be a JSON object, not "
                                    f"{type(expected_shape).__name__}")
        if expected_shape is not None:
            unknown = set(expected_shape) - set(self.SHAPE_FIELDS)
            if unknown:
                raise (RelayEnvError if self._shape_source == "GBENCH_RELAY_EXPECTED_SHAPE"
                       else ValueError)(
                    f"{self._shape_source if self._shape_source != 'caller' else 'expected_shape'}"
                    f" checks {', '.join(self.SHAPE_FIELDS)}, not {sorted(unknown)}")
        self._expected_shape = dict(expected_shape) if expected_shape is not None else None
        self.shape_requests_checked = 0
        self.shape_requests_violating = 0
        self._shape_violations: Dict[str, int] = {}
        self._shape_examples: List[Dict[str, Any]] = []
        # -- smoke episodes (smoke_stop / note_served): GBENCH_SMOKE_RELAY_REPLIES=N, which
        # gbench --smoke sets for its agent suites only (smoke.AGENT_SUITES), unless the caller
        # says otherwise. GBENCH_SMOKE=1 alone (every suite of a smoke run) changes no relay: a
        # non-agent relay (bfcl's agentic part, aider_polyglot, livebench) runs its sample to its
        # own end under --smoke (docs/smoke.md, Limits; 2026-10-11 relay review).
        self._smoke_source = "caller" if smoke_replies is not None else None
        if (smoke_replies is None and smoke_from_env
                and (os.environ.get("GBENCH_SMOKE_RELAY_REPLIES") or "").strip()):
            raw = os.environ["GBENCH_SMOKE_RELAY_REPLIES"].strip()
            if not raw.isdigit() or int(raw) < 1:
                raise RelayEnvError(f"GBENCH_SMOKE_RELAY_REPLIES is a number of replies "
                                    f"(1 or more), not {raw!r}")
            smoke_replies = int(raw)
            self._smoke_source = "GBENCH_SMOKE_RELAY_REPLIES"
        if smoke_replies is not None and (isinstance(smoke_replies, bool)
                                          or not isinstance(smoke_replies, int)
                                          or smoke_replies < 1):
            raise ValueError(f"smoke_replies is a number of replies (1 or more), "
                             f"not {smoke_replies!r}")
        self._smoke_replies = smoke_replies
        self._smoke: Dict[str, Dict[str, Any]] = {}
        self.smoke_stops = 0
        self.smoke_second_requests = 0
        self.smoke_unkeyed_requests = 0

    @property
    def url(self) -> str:
        """The /v1 endpoint the agent uses (the address it listened on, after `stop` too)."""
        port = self._server.server_address[1] if self._server else self._port
        return f"http://{self.bind_host}:{port}/v1"

    def target(self, raw_path: str) -> Optional[str]:
        """The endpoint URL a request for `raw_path` goes to, or None when it is refused."""
        import urllib.parse
        raw = raw_path or ""
        if "#" in raw:
            return None
        path, _, query = raw.partition("?")
        if not raw.startswith("/v1/") and path not in self.root_paths:
            return None
        segments = path.split("/")[1:]
        if any(not s or s in (".", "..") or not _RELAY_SEGMENT.match(s) for s in segments):
            return None
        if not _RELAY_QUERY.match(query):
            return None
        if path not in self.allowed_paths and path not in self.root_paths:
            return None
        return urllib.parse.urlunsplit((self._scheme, self._netloc, self._base_path + path,
                                        query, ""))

    def refusal_reason(self) -> str:
        """The reason a refused path is answered with (400): the paths the relay forwards."""
        if self.root_paths:
            return ("only the allowed /v1/ endpoint paths and the root paths "
                    + ", ".join(self.root_paths) + " are relayed")
        return "only the allowed /v1/ endpoint paths are relayed"

    def patch_body(self, path: str, body: Optional[bytes], content_type: str) -> Optional[bytes]:
        """`body` with `body_fields` set, for a JSON POST to a completions path, and the run's
        --max-output-tokens as max_tokens when an agent suite's request carries none
        (`_fill_max_tokens`, counted in max_tokens_filled)."""
        if not (self._body_fields or self._fill_max_tokens) or body is None \
                or "json" not in (content_type or "").lower():
            return body
        if not path.partition("?")[0].endswith(self.PATCHED_SUFFIXES):
            return body
        try:
            data = json.loads(body)
        except ValueError:
            return body
        if not isinstance(data, dict):
            return body
        for name, value in self._body_fields.items():
            if isinstance(value, dict) and isinstance(data.get(name), dict):
                data[name] = {**data[name], **value}
            else:
                data[name] = value
        if self._fill_max_tokens and data.get("max_tokens") is None \
                and data.get("max_completion_tokens") is None:
            data["max_tokens"] = int(self._fill_max_tokens)
            with self._lock:
                self.max_tokens_filled += 1
        return json.dumps(data).encode("utf-8")

    #: How many replies empty after reasoning keep their evidence (empty_after_reasoning_samples).
    EMPTY_SAMPLE_LIMIT = 20
    #: How many of a sampled reply's last token ids it keeps.
    TOKEN_ID_TAIL = 256

    def add_token_ids(self, path: str, body: Optional[bytes]) -> Tuple[Optional[bytes], bool]:
        """(`body`, whether the relay asked the endpoint for token ids): a non-streamed JSON
        completions request that does not ask for them itself gets `return_token_ids` (vLLM), so
        a reply that comes back with reasoning and no content keeps the token ids it was decoded
        from (empty_sample). The relay takes them out of the reply again before the client reads
        it (strip_token_ids), so the client gets the reply it asked for. Sampling is unchanged."""
        if body is None or not path.partition("?")[0].endswith(self.PATCHED_SUFFIXES):
            return body, False
        try:
            data = json.loads(body)
        except ValueError:
            return body, False
        if (not isinstance(data, dict) or data.get("stream") or data.get("echo")
                or "return_token_ids" in data or not self._upstream_is_vllm()):
            return body, False
        data["return_token_ids"] = True
        return json.dumps(data).encode("utf-8"), True

    def _upstream_is_vllm(self) -> bool:
        """Whether the endpoint is vLLM: its model list says so (`owned_by` "vllm", asked once
        before the first completions request, _probe_vllm), or its replies show it (note_server:
        a vLLM completions reply carries `prompt_token_ids`, null when not asked for).
        return_token_ids is a vLLM request field another OpenAI-compatible server may refuse,
        so it is only sent once the server is known to be vLLM. Asking the model list first
        lets the first reply carry its token ids too (an empty first reply can then be
        classified, classify_empty_reply)."""
        if self._is_vllm is None and not self._probed:
            with self._probe_lock:          # concurrent first requests wait for the one probe
                if not self._probed:
                    if self._probe_vllm():
                        self._is_vllm = True
                    self._probed = True
        return bool(self._is_vllm)

    def _probe_vllm(self) -> bool:
        """True when the endpoint's GET /v1/models lists a model `owned_by` "vllm" (what vLLM's
        OpenAI server reports); False on any other answer or failure."""
        import urllib.request
        try:
            req = urllib.request.Request(
                self.upstream.rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {self._key}"} if self._key else {})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except Exception:                                                   # noqa: BLE001
            return False
        models = data.get("data") if isinstance(data, dict) else None
        return any(isinstance(m, dict) and m.get("owned_by") == "vllm" for m in models or [])

    def note_server(self, reply: Any) -> None:
        """Learn from the first completions reply whether the endpoint is vLLM."""
        if self._is_vllm is None and isinstance(reply, dict) and reply.get("choices"):
            self._is_vllm = "prompt_token_ids" in reply or any(
                isinstance(ch, dict) and "token_ids" in ch for ch in reply["choices"])

    @staticmethod
    def strip_token_ids(reply: bytes) -> bytes:
        """`reply` with the token-id fields add_token_ids asked for (`prompt_token_ids` and each
        choice's `token_ids`) set back to null, as the server sends them when they are not asked
        for (vLLM includes the keys either way); `reply` itself when it is not a JSON object."""
        try:
            data = json.loads(reply)
        except ValueError:
            return reply
        if not isinstance(data, dict):
            return reply
        if "prompt_token_ids" in data:
            data["prompt_token_ids"] = None
        for ch in data.get("choices") or []:
            if isinstance(ch, dict):
                for key in ("token_ids", "prompt_token_ids"):
                    if key in ch:
                        ch[key] = None
        return json.dumps(data).encode("utf-8")

    def request_fingerprint(self, path: str, body: Optional[bytes]) -> Optional[str]:
        """conversation_fingerprint of a completions request's JSON body (None for another
        path, a body that is not JSON or one without a user message)."""
        if body is None or not path.partition("?")[0].endswith(self.PATCHED_SUFFIXES):
            return None
        try:
            data = json.loads(body)
        except ValueError:
            return None
        return conversation_fingerprint(data.get("messages")) if isinstance(data, dict) else None

    def count_failure(self, kind: str, fingerprint: Optional[str] = None) -> None:
        """Count one endpoint failure under `kind` (see the class docstring). An endpoint-side
        failure (endpoint_side_failure) also keeps its time and the fingerprint of the request
        it failed (`failure_times`)."""
        import time
        with self._lock:
            self.upstream_errors[kind] = self.upstream_errors.get(kind, 0) + 1
            if endpoint_side_failure(kind):
                self._failure_times.append((time.time(), fingerprint))

    def count_root_error(self, kind: str) -> None:
        """Count one failure of a root-path request (a request outside /v1, `root_paths`) under
        `kind`: in root_path_errors, never an endpoint failure of a model request."""
        with self._lock:
            self.root_path_errors[kind] = self.root_path_errors.get(kind, 0) + 1

    def count_success(self, fingerprint: Optional[str]) -> None:
        """Note a 2xx completions reply to a request of conversation `fingerprint`."""
        import time
        if fingerprint:
            with self._lock:
                self._last_success[fingerprint] = time.time()

    def failure_times(self, start: float, end: float,
                      fingerprint: Optional[str] = None) -> List[float]:
        """The times of the endpoint-side failures (endpoint_side_failure) the relay counted
        between the epoch times `start` and `end` (inclusive), only those of requests of
        conversation `fingerprint` when it is given. A sender cannot add one of these without
        the endpoint failing a request, and a fingerprint names one rollout's conversation (a
        concurrent rollout's failure has another)."""
        with self._lock:
            return [t for t, fp in self._failure_times
                    if start <= t <= end and (fingerprint is None or fp == fingerprint)]

    def failures_between(self, start: float, end: float,
                         fingerprint: Optional[str] = None) -> int:
        """How many endpoint-side failures `failure_times` finds in the window."""
        return len(self.failure_times(start, end, fingerprint))

    def succeeded_after(self, fingerprint: Optional[str], when: float) -> bool:
        """True when a request of conversation `fingerprint` got a 2xx completions reply after
        the epoch time `when` (the rollout went on after a failure)."""
        with self._lock:
            return bool(fingerprint) and self._last_success.get(fingerprint, 0.0) > when

    def count_client_closed(self, age_s: Optional[float] = None) -> None:
        """Count one reply whose client closed the connection before the relay had sent all of
        it (`client_closed_streams`): the client's choice, not an endpoint failure. `age_s`, the
        request's age when its client left, goes into client_close_ages; one within
        CLIENT_TIMEOUT_WINDOW_S of a client_timeout_suspect_s timeout counts in
        client_timeouts_suspected."""
        with self._lock:
            self.client_closed_streams += 1
            if age_s is not None:
                self._close_ages.append(float(age_s))
                if any(abs(age_s - t) <= self.CLIENT_TIMEOUT_WINDOW_S for t in self._suspect_s):
                    self.client_timeouts_suspected += 1

    def client_close_record(self) -> Dict[str, Any]:
        """client_close_ages: how many client closes had an age, a histogram of those ages by
        CLOSE_AGE_EDGES ("<60", "60-300", ..., ">=3600"), the largest, and the suspect closes
        (client_timeouts_suspected, with their first ages)."""
        with self._lock:
            ages = list(self._close_ages)
            suspected = self.client_timeouts_suspected
        buckets: Dict[str, int] = {}
        edges = list(self.CLOSE_AGE_EDGES)
        names = ([f"<{edges[0]}"] + [f"{a}-{b}" for a, b in zip(edges, edges[1:])]
                 + [f">={edges[-1]}"])
        for name in names:
            buckets[name] = 0
        for age in ages:
            i = next((k for k, edge in enumerate(edges) if age < edge), len(edges))
            buckets[names[i]] += 1
        suspects = [round(a, 1) for a in ages
                    if any(abs(a - t) <= self.CLIENT_TIMEOUT_WINDOW_S for t in self._suspect_s)]
        return {"closes_with_age": len(ages), "histogram_s": buckets,
                "max_s": round(max(ages), 1) if ages else None,
                "suspect_timeouts_s": list(self._suspect_s),
                "suspect_window_s": self.CLIENT_TIMEOUT_WINDOW_S,
                "client_timeouts_suspected": suspected,
                "client_timeout_suspect_ages": suspects[:self.EXAMPLE_LIMIT]}

    # -- progress and transport -----------------------------------------------------------

    @staticmethod
    def _json_body(body: Optional[bytes]) -> Optional[Dict[str, Any]]:
        if not body:
            return None
        try:
            data = json.loads(body)
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def request_total_s(self, path: str, body: Optional[bytes]) -> float:
        """The safety cap of one relayed request: TIMEOUT_S for a path that is not a completions
        path; for a model request the larger of TIMEOUT_S and base.request_timeout_s of its
        max_completion_tokens / max_tokens (of the served window, get_run_knob("max_model_len")
        or UNCAPPED_TIMEOUT_TOKENS, when it sends none) plus TOTAL_CAP_MARGIN_S. Far above any
        reply: the relay ends a model request that stops making progress (_ProgressClock), not
        one that takes long."""
        if not path.partition("?")[0].endswith(self.PATCHED_SUFFIXES):
            return float(self.TIMEOUT_S)
        from .base import UNCAPPED_TIMEOUT_TOKENS, get_run_knob, request_timeout_s
        data = self._json_body(body) or {}
        cap = None
        for key in ("max_completion_tokens", "max_tokens"):
            value = data.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                cap = value
                break
        if cap is None:
            try:
                cap = int(get_run_knob("max_model_len") or 0) or UNCAPPED_TIMEOUT_TOKENS
            except (TypeError, ValueError):
                cap = UNCAPPED_TIMEOUT_TOKENS
        total = float(max(self.TIMEOUT_S, request_timeout_s(cap) + self.TOTAL_CAP_MARGIN_S))
        with self._lock:
            self.largest_request_cap_s = max(self.largest_request_cap_s, total)
        return total

    def stream_body(self, path: str, body: Optional[bytes]) -> Tuple[Optional[bytes], bool]:
        """(`body`, whether the relay sends it as a stream and rebuilds the non-streamed reply,
        _StreamAssembler): a non-streamed JSON chat completions request to a vLLM endpoint gets
        `stream` true and `stream_options.include_usage`, so its reply shows progress event by
        event (_ProgressClock) instead of arriving whole at its end. Left as it is: with
        `stream_upstream` off; a request that streams already, sends stream_options, or asks for
        logprobs / top_logprobs / prompt_logprobs, echo or n > 1 (fields the rebuilt reply would
        have to merge from per-event pieces); /v1/completions; a server that is not vLLM (the
        rebuilt layout is vLLM's)."""
        if not self._stream_upstream or body is None:
            return body, False
        if not path.partition("?")[0].endswith("/chat/completions"):
            return body, False
        data = self._json_body(body)
        if data is None or data.get("stream") or "stream_options" in data:
            return body, False
        if (data.get("logprobs") or data.get("top_logprobs") is not None
                or data.get("prompt_logprobs") is not None or data.get("echo")
                or (data.get("n") not in (None, 1))):
            return body, False
        if not self._upstream_is_vllm():
            return body, False
        data["stream"] = True
        data["stream_options"] = {"include_usage": True}
        with self._lock:
            self.requests_streamed_upstream += 1
        return json.dumps(data).encode("utf-8"), True

    #: The vLLM /metrics counters whose sum advances when an engine schedules a request
    #: (prefix_cache_queries_total) and at each output token (the others). None of them moves
    #: during a prefill before its first output token (vLLM v1 IterationStats counts prompt
    #: tokens at a request's first output; 2026-10-11 live, 26B-A4B DP=2: all three stood still
    #: 7.3 s through a 182K-token prefill and 12.5 s through a 254K-token one).
    PROGRESS_METRICS = ("vllm:generation_tokens_total", "vllm:prompt_tokens_total",
                        "vllm:iteration_tokens_total_sum", "vllm:prefix_cache_queries_total")
    #: The gauge that moves during a prefill: each chunk allocates KV blocks (2026-10-11 live:
    #: vllm:kv_cache_usage_perc changed at every 0.5 s read through the 254K-token prefill). A
    #: change either way is progress (blocks allocated, or freed by a finished request).
    STATE_METRICS = ("vllm:kv_cache_usage_perc",)
    #: The gauges of the requests an engine holds (server_progress).
    LOAD_METRICS = ("vllm:num_requests_running", "vllm:num_requests_waiting")
    _ENGINE_LABEL = re.compile(r'engine="([^"]*)"')

    def server_progress(self) -> Optional[Dict[str, Tuple[float, float, float]]]:
        """{engine: (sum of its PROGRESS_METRICS, requests it runs or queues, its STATE_METRICS
        sum)} from the
        endpoint's /metrics (vLLM, at the server root; one engine per data-parallel rank), read
        at most once a second (a quarter of PROGRESS_CONFIRM_S when that is shorter) however many
        requests wait; None when it cannot be read or its counters are all 0 (no such counters,
        or a server that keeps no stats: nothing to tell progress from)."""
        import urllib.request
        now = time.monotonic()
        with self._progress_lock:
            fresh = min(1.0, self.PROGRESS_CONFIRM_S / 4)
            if self._progress_cache is not None and now - self._progress_cache[0] < fresh:
                return self._progress_cache[1]
            snapshot: Optional[Dict[str, Tuple[float, float]]] = None
            try:
                req = urllib.request.Request(
                    f"{self._scheme}://{self._netloc}{self._base_path}/metrics",
                    headers={"Authorization": f"Bearer {self._key}"} if self._key else {})
                with urllib.request.build_opener(_NoRedirect.build()).open(req, timeout=10) as r:
                    text = r.read().decode("utf-8", "replace")
                found: Dict[str, List[float]] = {}
                for line in text.splitlines():
                    name = line.split("{", 1)[0].split(" ", 1)[0]
                    if name in self.PROGRESS_METRICS:
                        index = 0
                    elif name in self.LOAD_METRICS:
                        index = 1
                    elif name in self.STATE_METRICS:
                        index = 2
                    else:
                        continue
                    try:
                        value = float(line.rsplit(" ", 1)[1])
                    except (IndexError, ValueError):
                        continue
                    label = self._ENGINE_LABEL.search(line)
                    slot = found.setdefault(label.group(1) if label else "", [0.0, 0.0, 0.0])
                    slot[index] += value
                if any(v[0] > 0 for v in found.values()):
                    snapshot = {engine: (v[0], v[1], v[2]) for engine, v in found.items()}
            except Exception:                                               # noqa: BLE001
                snapshot = None
            self._progress_cache = (time.monotonic(), snapshot)
            return snapshot

    @staticmethod
    def server_progressed(before: Dict[str, Tuple[float, ...]],
                          after: Dict[str, Tuple[float, ...]]) -> bool:
        """Whether the server made progress between two server_progress snapshots: some engine
        moved (its token counters advanced, or its KV-cache usage changed: a prefill), and no
        engine that holds requests (running or queued) stood still (one stuck data-parallel
        engine stalls the requests it holds while the others go on)."""
        advanced = False
        for engine, now in after.items():
            was = before.get(engine)
            if was is None:
                moved = False
            else:
                moved = now[0] > was[0] or (len(now) > 2 and len(was) > 2 and now[2] != was[2])
            advanced = advanced or moved
            if now[1] > 0 and not moved:
                return False
        return advanced

    def note_idle_tolerated(self, idle_s: float) -> None:
        """Count one idle window of a streaming reply the relay waited out because the server's
        counters advanced (_ProgressClock), keeping the longest such idle."""
        with self._lock:
            self.idle_waits_tolerated += 1
            self.longest_tolerated_idle_s = max(self.longest_tolerated_idle_s, float(idle_s))

    def count_assembled(self) -> None:
        with self._lock:
            self.replies_assembled += 1

    def next_event(self, pump: "_UpstreamPump", clock: "_ProgressClock") -> Tuple[str, Any]:
        """The next event of `pump`, or (_ProgressClock.STALLED / TIMEOUT, None) when the clock
        says the request stopped making progress or ran into its safety cap first."""
        import queue
        while True:
            try:
                return pump.events.get(timeout=max(0.05, clock.wait_s()))
            except queue.Empty:
                verdict = clock.expired()
                if verdict is not None:
                    return verdict, None

    def fetch(self, opener: Any, req: Any, watch: Any, timeout_s: float,
              assemble: bool = False) -> Tuple[Optional[int], Optional[bytes], Optional[str]]:
        """(status, body, None) of `req`'s reply read in full (rebuilt from its stream with
        `assemble`, _StreamAssembler), or (None, None, failure): "connection" (no reply, a broken
        or incomplete one), "timeout" (`timeout_s` ran out) or STALLED_NO_TOKENS
        (_ProgressClock). The endpoint connection is shut down on a timeout or a stall."""
        pump = _UpstreamPump(opener, req, watch, timeout_s)
        clock = _ProgressClock(self, self.IDLE_TIMEOUT_S, time.monotonic() + timeout_s,
                               chunked=assemble)
        kind, value = self.next_event(pump, clock)
        if kind in (_ProgressClock.STALLED, _ProgressClock.TIMEOUT):
            if watch is not None:
                watch.cut_upstream()
            return None, None, kind
        if kind == "error":
            return None, None, "timeout" if _is_timeout(value) else "connection"
        resp = value
        status = int(resp.status if hasattr(resp, "status") else resp.code)
        streamed = "event-stream" in (resp.headers.get("Content-Type") or "").lower()
        asm = _StreamAssembler() if assemble and streamed and 200 <= status < 300 else None
        if asm is None:
            clock.chunked = False
        parts: List[bytes] = []
        with resp:
            while True:
                kind, chunk = self.next_event(pump, clock)
                if kind in (_ProgressClock.STALLED, _ProgressClock.TIMEOUT):
                    if watch is not None:
                        watch.cut_upstream()
                    return None, None, kind
                if kind == "read_error":
                    return None, None, "timeout" if _is_timeout(chunk) else "connection"
                if kind == "eof":
                    break
                if asm is not None:
                    seen = asm.events
                    asm.feed(chunk)
                    if asm.events > seen:
                        clock.progress()
                else:
                    parts.append(chunk)
                    clock.progress()
            if _reply_incomplete(resp):
                return None, None, "connection"
        if asm is None:
            return status, b"".join(parts), None
        asm.finish()
        if not asm.complete():
            return None, None, "connection"
        self.count_assembled()
        status, body = asm.reply()
        return status, body, None

    # -- request shape ----------------------------------------------------------------------

    _ABSENT = "<absent>"

    def check_shape(self, path: str, body: Optional[bytes],
                    fingerprint: Optional[str] = None) -> List[str]:
        """The fields of `expected_shape` a completions request's JSON body (as it goes to the
        endpoint, after `body_fields`) does not match, counted in shape_check (never blocking):
          * "temperature": x - the body's `temperature` equals x (absent is a violation: the
            server's default would apply); None - no temperature is sent;
          * "enable_thinking": b - chat_template_kwargs.enable_thinking, absent read as False
            (the chat template's default), is b (chat requests only: a /v1/completions prompt is
            already rendered);
          * max_tokens n - the body's max_completion_tokens (else max_tokens) is n; None - no
            token budget is sent (both absent or null: the served window bounds the reply);
          * "max_tokens_at_most": n - no budget, or a budget of at most n (gbench --smoke's rule:
            a harness may send a smaller official or clamped budget, or none).
        A field not in expected_shape is not checked."""
        exp = self._expected_shape
        if exp is None or not path.partition("?")[0].endswith(self.PATCHED_SUFFIXES):
            return []
        data = self._json_body(body)
        if data is None:
            return []
        found: List[Tuple[str, Any, Any]] = []
        if "temperature" in exp:
            want, sent = exp["temperature"], data.get("temperature", self._ABSENT)
            if want is None:
                ok = sent in (self._ABSENT, None)
            else:
                ok = (isinstance(sent, (int, float)) and not isinstance(sent, bool)
                      and abs(float(sent) - float(want)) < 1e-9)
            if not ok:
                found.append(("temperature", want, sent))
        if "enable_thinking" in exp and path.partition("?")[0].endswith("/chat/completions"):
            kwargs = data.get("chat_template_kwargs")
            sent = (kwargs.get("enable_thinking", self._ABSENT) if isinstance(kwargs, dict)
                    else self._ABSENT)
            # the chat template reads the switch as a truth value (Jinja `if enable_thinking`)
            effective = False if sent in (self._ABSENT, None) else bool(sent)
            if effective is not bool(exp["enable_thinking"]):
                found.append(("enable_thinking", bool(exp["enable_thinking"]), sent))
        if "max_tokens" in exp:
            want = exp["max_tokens"]
            sent = self._ABSENT
            for key in ("max_completion_tokens", "max_tokens"):
                if data.get(key) is not None:
                    sent = data[key]
                    break
            ok = sent == self._ABSENT if want is None else (
                isinstance(sent, int) and not isinstance(sent, bool) and sent == want)
            if not ok:
                found.append(("max_tokens", want, sent))
        if exp.get("max_tokens_at_most") is not None:
            most = exp["max_tokens_at_most"]
            sent = self._ABSENT
            for key in ("max_completion_tokens", "max_tokens"):
                if data.get(key) is not None:
                    sent = data[key]
                    break
            if not (sent == self._ABSENT or (isinstance(sent, int) and not isinstance(sent, bool)
                                             and sent <= most)):
                found.append(("max_tokens_at_most", most, sent))
        with self._lock:
            self.shape_requests_checked += 1
            if found:
                self.shape_requests_violating += 1
            for name, want, sent in found:
                self._shape_violations[name] = self._shape_violations.get(name, 0) + 1
                if len(self._shape_examples) < self.EXAMPLE_LIMIT:
                    self._shape_examples.append({"field": name, "expected": want, "sent": sent,
                                                 "conversation": fingerprint})
            if found and self._smoke_replies is not None and fingerprint:
                self._smoke_conv(fingerprint)["shape_violations"] += 1
        return [name for name, _, _ in found]

    def shape_record(self) -> Dict[str, Any]:
        with self._lock:
            return {"expected": dict(self._expected_shape or {}), "source": self._shape_source,
                    "requests_checked": self.shape_requests_checked,
                    "requests_with_violations": self.shape_requests_violating,
                    "violations": dict(sorted(self._shape_violations.items())),
                    "examples": list(self._shape_examples)}

    # -- smoke episodes ---------------------------------------------------------------------

    def _smoke_conv(self, fingerprint: str) -> Dict[str, Any]:
        """The smoke state of one conversation (called with self._lock held)."""
        return self._smoke.setdefault(fingerprint, {
            "requests": 0, "replies_served": 0, "served_messages": 0, "shape_violations": 0,
            "stopped_requests": 0, "follow_up_arrived": False, "stopped_without_growth": 0})

    @staticmethod
    def message_count(data: Optional[Dict[str, Any]]) -> int:
        messages = data.get("messages") if isinstance(data, dict) else None
        return len(messages) if isinstance(messages, list) else 0

    def smoke_stop(self, path: str, body: Optional[bytes], fingerprint: Optional[str]) -> bool:
        """With `smoke_replies`=N, whether this completions request is answered with the smoke
        stop (HTTP 400 GBENCH_SMOKE_STOP) instead of being sent to the endpoint: its
        conversation already got N replies (note_served). A stopped request whose messages grew
        past those of the conversation's last served request is its follow-up (the agent read
        the reply, ran its action and came back): `follow_up_arrived`, counted once per
        conversation in smoke_second_requests; one that did not grow (a retry, or another
        rollout of the same task prompt) counts in `stopped_without_growth`. A request without a
        conversation fingerprint (no user message) is never stopped (smoke_unkeyed_requests).
        Always False without smoke_replies."""
        if self._smoke_replies is None or not path.partition("?")[0].endswith(
                self.PATCHED_SUFFIXES):
            return False
        if not fingerprint:
            with self._lock:
                self.smoke_unkeyed_requests += 1
            return False
        count = self.message_count(self._json_body(body))
        with self._lock:
            conv = self._smoke_conv(fingerprint)
            conv["requests"] += 1
            if conv["replies_served"] < self._smoke_replies:
                return False
            conv["stopped_requests"] += 1
            self.smoke_stops += 1
            if count > conv["served_messages"]:
                if not conv["follow_up_arrived"]:
                    conv["follow_up_arrived"] = True
                    self.smoke_second_requests += 1
            else:
                conv["stopped_without_growth"] += 1
            return True

    def smoke_stop_reply(self) -> bytes:
        """The JSON body of the smoke stop reply (HTTP 400), an OpenAI-style error."""
        return json.dumps({"error": {"message": self.SMOKE_STOP_MESSAGE,
                                     "type": self.SMOKE_STOP_TYPE, "param": None,
                                     "code": 400}}).encode("utf-8")

    def note_served(self, fingerprint: Optional[str], messages: int) -> None:
        """With `smoke_replies`, count one 2xx completions reply sent in full to a request of
        conversation `fingerprint` with `messages` messages."""
        if self._smoke_replies is None or not fingerprint:
            return
        with self._lock:
            conv = self._smoke_conv(fingerprint)
            conv["replies_served"] += 1
            conv["served_messages"] = max(conv["served_messages"], int(messages))

    def smoke_record(self) -> Dict[str, Any]:
        with self._lock:
            return {"replies_per_conversation": self._smoke_replies, "source": self._smoke_source,
                    "conversations": {fp: dict(c) for fp, c in sorted(self._smoke.items())},
                    "conversations_served": sum(1 for c in self._smoke.values()
                                                if c["replies_served"]),
                    "smoke_stops": self.smoke_stops,
                    "smoke_second_requests": self.smoke_second_requests,
                    "unkeyed_requests": self.smoke_unkeyed_requests}

    #: Of the client's per-request timeout (reask_deadline), the share the relay's re-asks may
    #: use, and the seconds kept back besides, so the reply the client gets always arrives
    #: before the client's own timeout fires (base would retry it and finally exclude it as
    #: "no reply from the served endpoint").
    REASK_DEADLINE_SHARE = 0.9
    REASK_DEADLINE_MARGIN_S = 15.0
    #: A re-ask is not started with less time than this left before the deadline.
    REASK_MIN_REMAINING_S = 5.0

    def reask_deadline(self, body: Optional[bytes], started: float) -> Optional[float]:
        """The time.monotonic() by which a re-ask chain of the request `body` (received at
        `started`) must have ended: the client's per-request timeout (base.request_timeout_s of
        the request's max_tokens, which is never longer than the timeout base gives that request;
        REQUEST_TIMEOUT_FLOOR_S, base's shortest, when the request sends none), times
        REASK_DEADLINE_SHARE, less REASK_DEADLINE_MARGIN_S. None without a re-ask rule."""
        if self._reask is None or not self._reask_limit:
            return None
        from .base import REQUEST_TIMEOUT_FLOOR_S, request_timeout_s
        cap = None
        try:
            data = json.loads(body) if body else None
        except ValueError:
            data = None
        if isinstance(data, dict):
            for key in ("max_completion_tokens", "max_tokens"):
                if isinstance(data.get(key), int) and not isinstance(data.get(key), bool):
                    cap = data[key]
                    break
        limit = request_timeout_s(cap) if cap else REQUEST_TIMEOUT_FLOOR_S
        return started + limit * self.REASK_DEADLINE_SHARE - self.REASK_DEADLINE_MARGIN_S

    def reask_reply(self, req: Any, path: str, buffered: bytes, fingerprint: Optional[str],
                    watch: Any, opener: Any, deadline: Optional[float] = None,
                    assemble: bool = False) -> Optional[bytes]:
        """The upstream harness's re-ask of a reply (the `reask` rule given to the relay, e.g.
        simple-evals' sampler asking a null `content` again, only for a reply the model ended
        itself): while the 2xx completions reply `buffered` matches the rule, the same request
        `req` is sent again, up to `reask_limit` times; each reply is counted (observe).

        Returns the reply the client gets: the first that does not match, or else the last reply
        the endpoint gave, which the client scores as upstream would score it. A chain that ends
        on a matching reply is counted in `reasks_exhausted`: after `reask_limit` re-asks, or cut
        short (also `reasks_cut_short`) because the next re-ask would run past `deadline`
        (reask_deadline: each re-ask is sent with the time left as its timeout, and none is
        started with less than REASK_MIN_REMAINING_S left), because a re-ask timed out or got no
        2xx reply (also counted as an endpoint failure). So a re-ask chain never turns a reply
        the client could score into a failed request. None only when the client left. With
        `assemble`, `req` goes to the endpoint as a stream (stream_body) and each reply is
        rebuilt from it (fetch, _StreamAssembler); a re-ask that stops making progress is
        counted as stalled_no_tokens."""
        import time
        if self._reask is None or not self._reask_limit:
            return buffered
        tries = 0
        while True:
            try:
                data = json.loads(buffered)
            except ValueError:
                break
            if not self._reask(data):
                # a null reply cut off at the budget, on the first try or on a re-ask
                if self._cut_off_match(data):
                    with self._lock:
                        self.cut_off_not_reasked += 1
                break
            if tries >= self._reask_limit:
                with self._lock:
                    self.reasks_exhausted += 1
                break
            if watch is not None and watch.aborted:
                return None
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining < self.REASK_MIN_REMAINING_S:
                self._reask_cut_short()
                break
            tries += 1
            with self._lock:
                self.reasks_sent += 1
                if tries == 1:
                    self.requests_reasked += 1
            timeout = self.TIMEOUT_S if remaining is None else min(self.TIMEOUT_S, remaining)
            status, body, failure = self.fetch(opener, req, watch, timeout, assemble)
            if watch is not None and watch.aborted:
                return None
            if body is None:
                # The re-ask ran into the deadline (the relay's own timeout, not an endpoint
                # failure; the endpoint's connection is closed, so it stops generating), or the
                # endpoint did not answer or stopped making progress (counted). Either way the
                # client gets the last reply the endpoint gave, which it scores as upstream would.
                if failure == STALLED_NO_TOKENS:
                    self.count_failure(STALLED_NO_TOKENS, fingerprint)
                elif remaining is None or failure != "timeout":
                    self.count_failure("connection", fingerprint)
                self._reask_cut_short()
                break
            self.observe(path, int(status), body, fingerprint)
            if not 200 <= int(status) < 300:
                self._reask_cut_short()
                break
            buffered = body
        return buffered

    def _cut_off_match(self, data: Any) -> bool:
        """Whether reply `data` was cut off at the token budget (`finish_reason` "length") with a
        content the re-ask rule would have asked again had the model ended it itself: not
        re-asked (the model's own outcome under the budget), counted in
        `cut_off_not_reasked`."""
        choices = data.get("choices") if isinstance(data, dict) else None
        if not (isinstance(choices, list) and choices and isinstance(choices[0], dict)
                and choices[0].get("finish_reason") == "length"):
            return False
        as_stopped = dict(data, choices=[dict(choices[0], finish_reason="stop"), *choices[1:]])
        try:
            return bool(self._reask(as_stopped))
        except Exception:                                                   # noqa: BLE001
            return False

    def _reask_cut_short(self) -> None:
        """Count a re-ask chain that ended before `reask_limit` on a reply still matching the
        rule (reask_reply): in `reasks_exhausted` and `reasks_cut_short`."""
        with self._lock:
            self.reasks_exhausted += 1
            self.reasks_cut_short += 1

    def reask_record(self) -> Dict[str, Any]:
        """What the relay's re-ask rule did: requests whose reply was asked again
        (`requests_reasked`), the re-asks sent (`reasks_sent`), and the requests whose last reply
        still matched the rule (`reasks_exhausted`, passed on as they were and scored as upstream
        would score them), of which `reasks_cut_short` ended before `reask_limit` re-asks (the
        client's deadline, or a re-ask the endpoint did not answer), and the replies cut off at the
        token budget whose content matched the rule (`cut_off_not_reasked`: the model's own
        outcome, never re-asked)."""
        with self._lock:
            return {"reask_limit": self._reask_limit, "requests_reasked": self.requests_reasked,
                    "reasks_sent": self.reasks_sent, "reasks_exhausted": self.reasks_exhausted,
                    "reasks_cut_short": self.reasks_cut_short,
                    "cut_off_not_reasked": self.cut_off_not_reasked}

    def tool_call_token_id(self) -> Optional[int]:
        """The served tokenizer's tool-call token id (served_token_id of TOOL_CALL_TOKEN_TEXT on
        the upstream server), None when it has none."""
        if self._tool_call_id is False:
            self._tool_call_id = served_token_id(
                f"{self._scheme}://{self._netloc}{self._base_path}", TOOL_CALL_TOKEN_TEXT,
                self._key)
        return self._tool_call_id

    def count_replies(self, outcomes: Sequence[Tuple[Any, ...]]) -> None:
        """Count the choices of one 2xx completions reply: (cut off, empty after reasoning[,
        classify_empty_reply class]); an empty reply that stopped on its own without a class (a
        streamed reply: no token ids) is EMPTY_UNCLASSIFIED."""
        with self._lock:
            for outcome in outcomes:
                cut, empty = outcome[0], outcome[1]
                kind = outcome[2] if len(outcome) > 2 else EMPTY_UNCLASSIFIED
                self.completion_replies += 1
                self.replies_cut_off += int(bool(cut))
                self.replies_empty_after_reasoning += int(bool(empty))
                if empty and not cut:
                    if kind == EMPTY_ANSWER_IN_THINKING:
                        self.replies_answer_in_thinking += 1
                    elif kind == EMPTY_TOOL_CALL_DROPPED:
                        self.replies_tool_call_dropped += 1
                    else:
                        self.replies_empty_unclassified += 1

    def observe(self, path: str, status: int, body: Optional[bytes],
                fingerprint: Optional[str] = None) -> None:
        """Count one reply the relay read in full: a status that is not 2xx is an endpoint
        failure (a 400 naming the context window is counted apart, CONTEXT_OVERFLOW_RE); a 2xx
        completions reply adds its choices to the reply counts. `fingerprint` is the request's
        conversation_fingerprint. A root-path reply that is not 2xx counts in root_path_errors
        (count_root_error); a 2xx one is not counted."""
        if path.partition("?")[0] in self.root_paths:
            if not 200 <= int(status) < 300:
                self.count_root_error(f"http_{int(status)}")
            return
        if not 200 <= int(status) < 300:
            text = (body or b"").decode("utf-8", "replace")
            if int(status) == 400 and CONTEXT_OVERFLOW_RE.search(text):
                with self._lock:
                    self.context_length_rejections += 1
            else:
                self.count_failure(f"http_{int(status)}", fingerprint)
            return
        if not path.partition("?")[0].endswith(self.PATCHED_SUFFIXES):
            return
        try:
            data = json.loads(body or b"null")
        except ValueError:
            self.count_failure("unparsable_reply", fingerprint)
            return
        self.count_success(fingerprint)
        self.note_server(data)
        outcomes = []
        for ch in (data.get("choices") if isinstance(data, dict) else None) or []:
            if not isinstance(ch, dict):
                continue
            msg = ch.get("message") if isinstance(ch.get("message"), dict) else None
            if msg is None:        # /v1/completions
                outcomes.append(_reply_outcome(ch.get("finish_reason"), ch.get("text"), None,
                                               None))
            else:
                reasoning = msg.get("reasoning_content") or msg.get("reasoning")
                outcome = _reply_outcome(ch.get("finish_reason"), msg.get("content"), reasoning,
                                         msg.get("tool_calls"))
                kind = None
                if outcome[1] and ch.get("finish_reason") != "length":
                    # a reply cut off at a limit is counted in replies_cut_off; the evidence is
                    # for replies the model ended itself
                    ids = ch.get("token_ids")
                    kind = classify_empty_reply(
                        ids, self.tool_call_token_id() if isinstance(ids, list) and ids
                        else None)
                    self.keep_empty_sample(ch, msg, reasoning, data, fingerprint, kind)
                outcomes.append((*outcome, kind))
        self.count_replies(outcomes)

    def keep_empty_sample(self, choice: Dict[str, Any], msg: Dict[str, Any], reasoning: Any,
                          reply: Dict[str, Any], fingerprint: Optional[str],
                          kind: Optional[str] = None) -> None:
        """Keep the evidence of one reply that came back with reasoning and no content or tool
        call (up to EMPTY_SAMPLE_LIMIT): whether the model ended its turn inside its reasoning
        or the server's reasoning parser put the answer there can only be told from the tokens
        (`token_ids`, decoded later with special tokens kept, e.g. the server's /detokenize)."""
        from .base import redact_text as _redact
        text = reasoning if isinstance(reasoning, str) else str(reasoning or "")
        ids = choice.get("token_ids")
        ids = list(ids) if isinstance(ids, list) else None
        sample = {"finish_reason": choice.get("finish_reason"),
                  "stop_reason": choice.get("stop_reason"),
                  "content": msg.get("content"),
                  "reasoning_chars": len(text),
                  "reasoning_tail": _redact(text[-1500:]),
                  "completion_tokens": (reply.get("usage") or {}).get("completion_tokens"),
                  # the last TOKEN_ID_TAIL ids: where the turn ended (a stop token after an
                  # unclosed reasoning channel, or a channel close the parser missed)
                  "token_ids_tail": ids[-self.TOKEN_ID_TAIL:] if ids is not None else None,
                  "token_ids_total": len(ids) if ids is not None else None,
                  "conversation": fingerprint,
                  # classify_empty_reply of the reply's token ids
                  "classified_as": kind}
        with self._lock:
            if len(self.empty_after_reasoning_samples) < self.EMPTY_SAMPLE_LIMIT:
                self.empty_after_reasoning_samples.append(sample)

    def start(self) -> "EndpointRelay":
        _STARTED_RELAYS.append(self)
        import http.server
        import threading
        import urllib.error
        import urllib.request
        relay = self
        opener = urllib.request.build_opener(_NoRedirect.build(), *_tracked_handlers())

        class _Handler(http.server.BaseHTTPRequestHandler):
            def _refuse(self, code: int, why: str) -> None:
                relay.requests_refused += 1
                logger.warning("[%s] refused a %s request (%s)", relay._name, self.command, why)
                self.send_error(code, why)

            def _relay(self) -> None:
                if self.command not in ("GET", "POST"):
                    self._refuse(405, "only GET and POST are relayed")
                    return
                target = relay.target(self.path)
                if target is None:
                    self._refuse(400, relay.refusal_reason())
                    return
                if self.command != "POST" and self.path.partition("?")[0] in relay.root_paths:
                    self._refuse(405, "a root path is relayed for POST only")
                    return
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else None
                received = time.monotonic()
                deadline = relay.reask_deadline(body, received)
                fingerprint = relay.request_fingerprint(self.path, body)
                body = relay.patch_body(self.path, body, self.headers.get("Content-Type") or "")
                sent = relay._json_body(body)
                relay.check_shape(self.path, body, fingerprint)
                if relay.smoke_stop(self.path, body, fingerprint):
                    self._send_json(400, relay.smoke_stop_reply())
                    return
                total_s = relay.request_total_s(self.path, body)
                body, asked_ids = relay.add_token_ids(self.path, body)
                body, assemble = relay.stream_body(self.path, body)
                req = urllib.request.Request(target, data=body, method=self.command)
                for name in ("Content-Type", "Accept"):
                    if self.headers.get(name):
                        req.add_header(name, self.headers[name])
                if relay._key:
                    req.add_header("Authorization", f"Bearer {relay._key}")
                # From here until the reply is sent, a client that closes its connection (e.g.
                # an SDK that gives up on a long turn at its read timeout and resends it) ends
                # the endpoint's request too (_ClientWatch), so the served endpoint does not go
                # on generating a reply nobody reads; counted in client_closed_streams.
                watch = _ClientWatch(self.connection)
                try:
                    self._forward(req, fingerprint, watch, asked_ids, deadline,
                                  received=received, total_s=total_s, assemble=assemble,
                                  streamed_request=bool(sent and sent.get("stream")),
                                  messages=relay.message_count(sent))
                finally:
                    watch.stop()

            def _send_json(self, status: int, payload: bytes) -> None:
                """Answer the client with a JSON body the relay made itself."""
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(payload)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _no_progress(self, verdict: str, fingerprint: Optional[str], watch: _ClientWatch,
                             received: float, headers_sent: bool) -> None:
                """The request stopped making progress (STALLED: counted as stalled_no_tokens)
                or ran into its safety cap (counted as `connection`, as a reply the endpoint
                never finished): the endpoint's connection is shut down, so it stops generating,
                and the client gets a 504 / 502 (an error event when its stream had started)."""
                watch.cut_upstream()
                if watch.aborted:
                    relay.count_client_closed(time.monotonic() - received)
                    return
                stalled = verdict == _ProgressClock.STALLED
                kind = STALLED_NO_TOKENS if stalled else "connection"
                if self.path.partition("?")[0] in relay.root_paths:
                    relay.count_root_error(kind)
                else:
                    relay.count_failure(kind, fingerprint)
                why = (f"the served endpoint sent no token for {relay.IDLE_TIMEOUT_S:g} s while "
                       f"its engines stood still: token counters and KV-cache usage "
                       f"({STALLED_NO_TOKENS})" if stalled else
                       "the served endpoint did not finish the reply within the relay's safety "
                       "cap")
                logger.warning("[%s] %s", relay._name, why)
                code = 504 if stalled else 502
                error = {"error": {"message": f"gbench relay: {why}", "type": kind,
                                   "param": None, "code": code}}
                if headers_sent:
                    try:
                        self.wfile.write(b"data: " + json.dumps(error).encode("utf-8") + b"\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self._send_json(code, json.dumps(error).encode("utf-8"))

            def _forward(self, req: Any, fingerprint: Optional[str], watch: _ClientWatch,
                         asked_ids: bool = False, deadline: Optional[float] = None,
                         received: Optional[float] = None, total_s: Optional[float] = None,
                         assemble: bool = False, streamed_request: bool = False,
                         messages: int = 0) -> None:
                path = self.path.partition("?")[0]
                completion = path.endswith(relay.PATCHED_SUFFIXES)
                received = time.monotonic() if received is None else received
                total_s = float(total_s or relay.TIMEOUT_S)
                pump = _UpstreamPump(opener, req, watch, total_s)
                # A model request is watched for progress (_ProgressClock); another request
                # (the model list, a root path) only for its total time.
                clock = _ProgressClock(relay, relay.IDLE_TIMEOUT_S if completion else total_s,
                                       received + total_s,
                                       chunked=completion and (assemble or streamed_request))
                kind, value = relay.next_event(pump, clock)
                if kind in (_ProgressClock.STALLED, _ProgressClock.TIMEOUT):
                    self._no_progress(kind, fingerprint, watch, received, False)
                    return
                if kind == "error":
                    if watch.aborted:
                        # the client left while it waited
                        relay.count_client_closed(time.monotonic() - received)
                        return
                    if path in relay.root_paths:
                        relay.count_root_error("connection")
                    else:
                        relay.count_failure("connection", fingerprint)
                    logger.warning("[%s] the endpoint did not answer (%s)", relay._name,
                                   type(value).__name__)
                    self.send_error(502, "the served endpoint did not answer")
                    return
                resp = value
                with relay._lock:
                    relay.requests_relayed += 1
                    relay.root_requests_relayed += int(path in relay.root_paths)
                status = resp.status if hasattr(resp, "status") else resp.code
                ctype = resp.headers.get("Content-Type") or ""
                ok = 200 <= int(status) < 300
                streamed = ok and "event-stream" in ctype.lower()
                # the reply the relay rebuilds from the stream it asked for (stream_body); a
                # server that answered with a whole reply anyway is relayed as it is
                assembling = assemble and completion and streamed
                if not streamed:
                    clock.chunked = False

                def _gone() -> None:
                    relay.count_client_closed(time.monotonic() - received)

                with resp:
                    buffered = None
                    if not ok or (completion and not streamed) or assembling:
                        asm = _StreamAssembler() if assembling else None
                        parts: List[bytes] = []
                        while True:
                            kind, chunk = relay.next_event(pump, clock)
                            if kind in (_ProgressClock.STALLED, _ProgressClock.TIMEOUT):
                                self._no_progress(kind, fingerprint, watch, received, False)
                                return
                            if kind == "read_error":
                                if watch.aborted:
                                    _gone()             # the client left while it waited
                                    return
                                relay.count_failure("connection", fingerprint)
                                self.send_error(502, "the served endpoint's reply broke off")
                                return
                            if kind == "eof":
                                break
                            if asm is not None:
                                seen = asm.events
                                asm.feed(chunk)
                                if asm.events > seen:
                                    clock.progress()
                            else:
                                parts.append(chunk)
                                clock.progress()
                        if watch.aborted and _reply_incomplete(resp):
                            _gone()
                            return
                        if asm is not None:
                            asm.finish()
                            if not asm.complete():
                                if watch.aborted:
                                    _gone()
                                    return
                                relay.count_failure("connection", fingerprint)
                                self.send_error(502, "the served endpoint's reply broke off")
                                return
                            relay.count_assembled()
                            status, buffered = asm.reply()
                            ctype = "application/json"
                            ok = 200 <= int(status) < 300
                        else:
                            buffered = b"".join(parts)
                        relay.observe(path, int(status), buffered, fingerprint)
                        if ok and completion and relay._reask is not None:
                            again = relay.reask_reply(req, path, buffered, fingerprint, watch,
                                                      opener, deadline, assemble=assemble)
                            if again is None:           # the client left
                                _gone()
                                return
                            buffered = again
                        if asked_ids and ok:
                            buffered = relay.strip_token_ids(buffered)
                        if watch.aborted:
                            _gone()     # the client left before the reply
                            return
                    try:
                        self.send_response(status)
                        if ctype:
                            self.send_header("Content-Type", ctype)
                        self.send_header("Connection", "close")
                        self.end_headers()
                        if buffered is not None:
                            self.wfile.write(buffered)
                            self.wfile.flush()
                            if ok and completion:
                                relay.note_served(fingerprint, messages)
                            return
                    except (BrokenPipeError, ConnectionResetError):
                        _gone()         # the client left before the reply
                        return
                    tally = _StreamTally() if completion else None
                    left = False
                    while True:
                        kind, chunk = relay.next_event(pump, clock)
                        if kind in (_ProgressClock.STALLED, _ProgressClock.TIMEOUT):
                            self._no_progress(kind, fingerprint, watch, received, True)
                            if tally is not None:
                                for _ in range(tally.errors):
                                    relay.count_failure("stream_error", fingerprint)
                            return
                        if kind == "read_error":
                            if not watch.aborted:
                                relay.count_failure("connection", fingerprint)
                                return
                            left = True
                            break
                        if kind == "eof":
                            left = watch.aborted and _reply_incomplete(resp)
                            break
                        if tally is not None:
                            seen = tally.events
                            tally.feed(chunk)
                            if tally.events > seen:
                                clock.progress()
                        else:
                            clock.progress()
                        try:
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            left = True
                            break
                    if left:
                        # The client closed the stream (e.g. at a stop sequence it applies
                        # itself). Leaving `with resp` closes the endpoint's response (the watch
                        # may have shut it down already), so the served endpoint stops the
                        # generation. Not an endpoint failure, and the unfinished reply is not
                        # counted among completion_replies.
                        watch.cut_upstream()
                        _gone()
                        if tally is not None:
                            for _ in range(tally.errors):
                                relay.count_failure("stream_error", fingerprint)
                        return
                    if tally is not None:
                        relay.count_replies(tally.outcomes())
                        for _ in range(tally.errors):
                            relay.count_failure("stream_error", fingerprint)
                        if not tally.errors:
                            relay.count_success(fingerprint)
                            if ok:
                                relay.note_served(fingerprint, messages)

            do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _relay

            def log_message(self, *args: Any) -> None:     # one line per request is noise
                return

        class _Server(http.server.ThreadingHTTPServer):
            # A listen backlog of 128 (socketserver's default is 5), as gemini_relay._Server:
            # in the 2026-10-10 review of the toolbench /tokenize fix, 32 client threads through
            # a relay with the default backlog got 3-5 connection resets per 400 requests, made
            # by the kernel before the relay saw them, so no counter here showed them (toolbench
            # opens two connections per model request: /tokenize, then the completion).
            daemon_threads = True
            request_queue_size = 128

        server = _Server((self.bind_host, 0), _Handler)
        self._server = server
        self._port = server.server_address[1]
        self._thread = threading.Thread(target=server.serve_forever, name=self._name,
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def record(self) -> Dict[str, Any]:
        """What a result records about the relay (never the key)."""
        rec: Dict[str, Any] = {"listens": self.url, "relays_to": self.upstream,
                               "allowed_paths": list(self.allowed_paths),
                               "requests_relayed": self.requests_relayed,
                               "requests_refused": self.requests_refused}
        if self.root_paths:
            rec.update({"root_paths": list(self.root_paths),
                        "root_requests_relayed": self.root_requests_relayed,
                        "root_path_errors": dict(sorted(self.root_path_errors.items()))})
        with self._lock:
            rec.update({
                "upstream_errors": dict(sorted(self.upstream_errors.items())),
                "endpoint_side_failures": len(self._failure_times),
                "context_length_rejections": self.context_length_rejections,
                "completion_replies": self.completion_replies,
                "replies_cut_off": self.replies_cut_off,
                "replies_empty_after_reasoning": self.replies_empty_after_reasoning,
                "replies_answer_in_thinking": self.replies_answer_in_thinking,
                "replies_tool_call_dropped": self.replies_tool_call_dropped,
                "replies_empty_unclassified": self.replies_empty_unclassified,
                "tool_call_token_id": (None if self._tool_call_id is False
                                       else self._tool_call_id),
                "client_closed_streams": self.client_closed_streams,
                "client_timeouts_suspected": self.client_timeouts_suspected,
                # how the relay reached the endpoint and when it gives up (docs/relay.md)
                "upstream_transport": {
                    "streams_non_streamed_chat_requests": self._stream_upstream,
                    "requests_streamed_upstream": self.requests_streamed_upstream,
                    "replies_assembled": self.replies_assembled,
                    "idle_timeout_s": self.IDLE_TIMEOUT_S,
                    "progress_confirm_s": self.PROGRESS_CONFIRM_S,
                    "idle_waits_tolerated": self.idle_waits_tolerated,
                    "longest_tolerated_idle_s": round(self.longest_tolerated_idle_s, 1),
                    "timeout_floor_s": self.TIMEOUT_S,
                    "largest_request_cap_s": self.largest_request_cap_s or None,
                },
            })
            if self.empty_after_reasoning_samples:
                rec["empty_after_reasoning_samples"] = list(self.empty_after_reasoning_samples)
        rec["client_close_ages"] = self.client_close_record()
        if self._expected_shape is not None:
            rec["shape_check"] = self.shape_record()
            rec["shape_violations"] = rec["shape_check"]["violations"]
        if self._smoke_replies is not None:
            rec["smoke"] = self.smoke_record()
            rec["smoke_second_requests"] = rec["smoke"]["smoke_second_requests"]
        if self._reask is not None:
            rec["reask"] = self.reask_record()
        if self._key:
            rec["adds"] = "Authorization: Bearer <the endpoint key>"
        if self._body_fields:
            rec["sets_body_fields"] = self._body_fields
        if self._fill_max_tokens:
            rec["max_tokens_from_run"] = {"value": int(self._fill_max_tokens),
                                          "requests_filled": self.max_tokens_filled}
        return rec


def relay_failure_count(relay_record: Optional[Dict[str, Any]]) -> Optional[int]:
    """The endpoint failures an EndpointRelay's `record()` counts (its upstream_errors plus the
    requests it refused); None without a record."""
    if relay_record is None:
        return None
    return (sum(int(v) for v in (relay_record.get("upstream_errors") or {}).values())
            + int(relay_record.get("requests_refused") or 0))


def relay_failure_reasons(result: Dict[str, Any], relay_record: Dict[str, Any]) -> List[str]:
    """The endpoint failures an EndpointRelay counted (its `record()`), applied to `result`:
    harness errors, never the model's outcome, listed in `endpoint_failures`, a `success` status
    becomes `completed_with_errors`, and the returned reason makes the run not comparable. The
    scaffold retries some failures itself, and the relay cannot tell an agent's request from one
    the model's own commands sent from the same sandbox, so the result says how many there were
    (a run-level flag that never changes a score); an instance whose rollout one of them ended is
    classified by the caller, on evidence the model cannot produce (EndpointRelay.failure_times:
    endpoint-side failures of requests carrying that rollout's conversation)."""
    errors = dict(relay_record.get("upstream_errors") or {})
    failed = sum(int(v) for v in errors.values())
    refused = int(relay_record.get("requests_refused") or 0)
    if not (failed or refused):
        return []
    result["endpoint_failures"] = {"upstream_errors": errors, "requests_refused": refused}
    if result.get("status") == "success":
        result["status"] = "completed_with_errors"
    parts = []
    if failed:
        parts.append(f"the served endpoint failed {failed} relayed model request(s) ("
                     + ", ".join(f"{k}: {v}" for k, v in sorted(errors.items())) + ")")
    if refused:
        parts.append(f"the relay refused {refused} request(s) to a path it does not forward")
    return ["harness errors during the rollouts: " + "; ".join(parts)]


def relay_empty_reply_classes(relay_record: Dict[str, Any]) -> Dict[str, int]:
    """The relay's replies that stopped on their own with reasoning but no content and no tool
    call, by classify_empty_reply: {"answer_in_thinking", "tool_call_dropped", "unclassified"}.
    A record written before the split existed (no `replies_tool_call_dropped`) cannot tell them
    apart: all of them are unclassified."""
    if "replies_tool_call_dropped" in relay_record:
        return {"answer_in_thinking": int(relay_record.get("replies_answer_in_thinking") or 0),
                "tool_call_dropped": int(relay_record.get("replies_tool_call_dropped") or 0),
                "unclassified": int(relay_record.get("replies_empty_unclassified") or 0)}
    old = relay_record.get("replies_answer_in_thinking",
                           relay_record.get("replies_empty_after_reasoning"))
    return {"answer_in_thinking": 0, "tool_call_dropped": 0, "unclassified": int(old or 0)}


def empty_reply_class_reasons(classes: Dict[str, int], scaffold: str) -> List[str]:
    """The non-comparability reasons of the empty replies that are not the model's answer
    (relay_empty_reply_classes): a native tool call the server's tool-call parser removed (a
    server-parser artifact the scaffold saw as an empty reply) and replies that could not be
    told apart from an answer inside the thinking channel. Both stay out of answer_in_thinking."""
    reasons = []
    if classes.get("tool_call_dropped"):
        reasons.append(
            f"{classes['tool_call_dropped']} model turn(s) whose generation was a native tool "
            f"call the served endpoint's tool-call parser removed ({scaffold} reads the reply's "
            "content and declares no tools, so it got an empty reply): a server-parser artifact, "
            "not the model's answer (tool_call_dropped_turns)")
    if classes.get("unclassified"):
        reasons.append(
            f"{classes['unclassified']} model turn(s) that came back with reasoning and no "
            "content or tool call without token ids to tell an answer inside the thinking "
            "channel from a native tool call the server's parser removed (unclassified, not "
            "counted in answer_in_thinking; unclassified_empty_turns)")
    return reasons


def relay_result_reasons(result: Dict[str, Any], relay_record: Dict[str, Any],
                         scaffold: str) -> List[str]:
    """Apply what an EndpointRelay counted (its `record()`) to an agentic result whose scaffold
    reached the model only through that relay; returns the non-comparability reasons.

    * Endpoint failures (`upstream_errors`) and requests the relay refused are harness errors,
      never the model's outcome: listed in `endpoint_failures`, and a `success` status becomes
      `completed_with_errors`. The scaffold retries some failures itself, and the relay cannot
      tell which instance a request belonged to, so the result says how many there were; an
      instance whose rollout one of them ended is classified by the caller.
    * Model turns that stopped at a limit (`finish_reason` "length") are `truncated_turns`,
      which make the run not comparable; a turn cut off at a limit is counted there only. The
      reason names the limit (relay_cutoff_reason): the max_tokens the relay sent (the run's
      --max-output-tokens), else the end of the served context window.
    * Turns that stopped on their own with reasoning but no content and no tool call are
      `empty_after_reasoning_turns`, split by their token ids (relay_empty_reply_classes):
      the answer inside the thinking channel, the model's outcome as the scaffold (which reads
      `content`) handles it, never a harness error and not a comparability reason (base's
      ANSWER_IN_THINKING_POLICY), counted with their share of the model turns
      (`answer_in_thinking`, `answer_in_thinking_rate`); a native tool call the server's
      tool-call parser removed (`tool_call_dropped_turns`) and turns without token ids to tell
      (`unclassified_empty_turns`) are not the model's answer: each makes the run not comparable
      (empty_reply_class_reasons).
      All are per-result counts: each trace records `truncated_turns: null`.
    """
    from .base import answer_in_thinking_fields
    cut = int(relay_record.get("replies_cut_off") or 0)
    replies = int(relay_record.get("completion_replies") or 0)
    classes = relay_empty_reply_classes(relay_record)
    result["truncated_turns"] = cut
    result["empty_after_reasoning_turns"] = sum(classes.values())
    result["tool_call_dropped_turns"] = classes["tool_call_dropped"]
    result["unclassified_empty_turns"] = classes["unclassified"]
    result.update(answer_in_thinking_fields(classes["answer_in_thinking"], replies))
    result["model_replies"] = replies
    reasons: List[str] = relay_failure_reasons(result, relay_record)
    if cut:
        reasons.append(relay_cutoff_reason(cut, relay_record, scaffold))
    reasons.extend(empty_reply_class_reasons(classes, scaffold))
    return reasons


def relay_cutoff_reason(cut: int, relay_record: Dict[str, Any], scaffold: str) -> str:
    """The non-comparability reason for `cut` model turns that ended with finish_reason "length",
    naming the limit from the relay's `record()`:

    * `sets_body_fields.max_tokens` N: the relay set max_tokens=N on every completions request
      (swe_agentic.with_run_max_tokens), so a cut-off turn stopped at N.
    * else `max_tokens_from_run` {value N, requests_filled k > 0}: the relay filled max_tokens=N
      into the k requests that carried none (EndpointRelay.patch_body).
    * else `max_tokens_from_run` {value N, requests_filled 0}: the fill was on and filled
      nothing, which patch_body does only when every completions request already carried its own
      max_tokens or max_completion_tokens, so a cut-off turn stopped at the scaffold's own limit.
    * else no cap was sent by the relay: the end of the served context window, as the scaffold
      sends no max_tokens.

    The capped wordings hold no "; ": the reasons are joined with "; " and split on it again
    (swe_bench_live._grouped_reasons), which cuts the uncapped wording (kept as it was) in two.
    2026-10-11 validation run (pool-10-11-132525): multi_swe_bench's relay set max_tokens=65536
    on every request (sets_body_fields, requests_filled 0) and its 2 cut-off turns were still
    worded "cut off at the end of the served context window (finish_reason length; MSWE-agent
    sends no max_tokens)"; swe_bench_live's tabs the same."""
    body_cap = (relay_record.get("sets_body_fields") or {}).get("max_tokens")
    if body_cap:
        return (f"{cut} model turn(s) cut off at max_tokens={int(body_cap)} (finish_reason "
                f"length: the run's --max-output-tokens, sent by gbench's relay with every "
                f"request, as {scaffold} sends none)")
    fill = relay_record.get("max_tokens_from_run") or {}
    filled = int(fill.get("requests_filled") or 0)
    if fill.get("value") and filled:
        return (f"{cut} model turn(s) cut off at a max_tokens limit (finish_reason length: "
                f"gbench's relay sent max_tokens={int(fill['value'])}, the run's "
                f"--max-output-tokens, with the {filled} request(s) that carried none, as "
                f"{scaffold} sends none)")
    if fill.get("value"):
        return (f"{cut} model turn(s) cut off at a max_tokens limit (finish_reason length: the "
                f"max_tokens or max_completion_tokens {scaffold} sent with every request, so "
                f"gbench's relay filled in none)")
    return (f"{cut} model turn(s) cut off at the end of the served context window "
            f"(finish_reason length; {scaffold} sends no max_tokens)")


def _first_message_content(reply: Any) -> Tuple[bool, Any]:
    """(whether the reply has a first choice with a message that the model ended itself, that
    message's `content`). A reply the model ended itself has `finish_reason` "stop"; one cut off
    at the token budget ("length") is the model's outcome under that budget and is never re-asked
    (base scores it as a truncated reply)."""
    choices = reply.get("choices") if isinstance(reply, dict) else None
    if not (isinstance(choices, list) and choices and isinstance(choices[0], dict)
            and isinstance(choices[0].get("message"), dict)):
        return False, None
    if choices[0].get("finish_reason") != "stop":
        return False, None
    return True, choices[0]["message"].get("content")


def reply_content_is_null(reply: Any) -> bool:
    """simple-evals' ChatCompletionSampler and Corr2Cause's Chatbot re-ask rule, for a reply the
    model ended itself (`finish_reason` "stop"): the chat reply's `choices[0].message.content` is
    None (the sampler raises "OpenAI API returned empty response; retrying"; the Chatbot's
    `.strip()` on None raises and `ask` retries). An empty string is a reply. A reply cut off at
    the token budget (`finish_reason` "length") is not re-asked: it is the model's own outcome
    under the run's budget, scored as base scores a truncated reply."""
    has, content = _first_message_content(reply)
    return has and content is None


def reply_content_is_null_or_empty(reply: Any) -> bool:
    """LongBench's pred.py rule, for a reply the model ended itself (`finish_reason` "stop"):
    query_llm returns `message.content`, and get_pred writes no prediction for a question whose
    output is '' (`if output == '': continue`) or None (`output.strip()` raises and ends the
    worker), so a resumed pred.py run (it skips the questions already in its output file) asks it
    again. A reply cut off at the token budget is not re-asked (the model's own outcome)."""
    has, content = _first_message_content(reply)
    return has and (content is None or content == "")


#: Seconds the re-asking relay waits for one upstream reply: the client (base) applies its own
#: per-request timeout and closes the connection, which stops the upstream generation.
REASK_RELAY_TIMEOUT_S = 24 * 3600


@contextlib.contextmanager
def upstream_reask_relay(eval_name: str, base_url: str, rule: Callable[[Any], bool],
                         limit: int):
    """A loopback EndpointRelay in front of the served endpoint that applies the suite's upstream
    re-ask rule (`rule`, e.g. reply_content_is_null) to each chat reply before base reads it: a
    matching reply the model ended itself (`finish_reason` "stop"; a reply cut off at the token
    budget is the model's own outcome and is never re-asked) is asked again, the same request,
    up to `limit` times (upstream re-asks without a limit; gbench stops there so a run ends), and
    never past the client's per-request deadline (EndpointRelay.reask_deadline): a chain that
    ends on a matching reply passes that last reply on, to be scored as upstream would score it
    (`reasks_exhausted`), never a failed request. Sampling, prompts and token budgets are
    the request's own; the relay adds the endpoint's key (base.endpoint_api_key). Yields the
    relay: base's requests go to `relay.url`, and `relay.reask_record()` says what was
    re-asked. Never a smoke relay (smoke_from_env off): base's single-turn requests are not
    agent episodes, and a request base sends again for the same question must reach the
    endpoint."""
    from .base import endpoint_api_key
    relay = EndpointRelay(base_url, "127.0.0.1", key=endpoint_api_key(),
                          name=f"{eval_name}-reask-relay", reask=rule, reask_limit=limit,
                          timeout_s=REASK_RELAY_TIMEOUT_S, smoke_from_env=False)
    relay.start()
    try:
        yield relay
    finally:
        relay.stop()


#: The counts of reask_summary that a merge of several records adds.
REASK_SUMMARY_COUNTS = ("requests_reasked", "reasks_sent", "reasks_exhausted", "reasks_cut_short",
                        "cut_off_not_reasked",
                        "replies_seen", "replies_cut_off_seen",
                        "replies_empty_after_reasoning_seen", "answer_in_thinking_seen",
                        "tool_call_dropped_seen", "unclassified_empty_seen")


def merge_reask_summaries(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Several reask_summary records (shards) as one: the counts added, the rate recomputed."""
    out: Dict[str, Any] = {"rule": records[0].get("rule"),
                           "reask_limit": records[0].get("reask_limit"),
                           **{k: sum(int(r.get(k) or 0) for r in records)
                              for k in REASK_SUMMARY_COUNTS}}
    seen = out["replies_seen"]
    out["answer_in_thinking_rate_seen"] = (round(out["answer_in_thinking_seen"] / seen, 6)
                                           if seen else None)
    failures: Dict[str, int] = {}
    for r in records:
        for k, v in (r.get("endpoint_failures") or {}).items():
            failures[k] = failures.get(k, 0) + int(v or 0)
    out["endpoint_failures"] = failures
    return out


def reask_divergence_reasons(reask: Optional[Dict[str, Any]], content: str, scored: str,
                             upstream: str, exhausted_reason: bool = True) -> List[str]:
    """The comparability reasons of an upstream_reask record (reask_summary, or several added):
    requests whose last reply still had `content` (e.g. "null content") when gbench stopped
    re-asking (`reasks_exhausted`: after reask_limit re-asks, or `reasks_cut_short` before the
    client's request deadline or on a re-ask the endpoint did not answer), scored as `scored`;
    and replies cut off at the token budget with that content, which gbench never re-asks
    (`cut_off_not_reasked`, the model's own outcome under the run's budget). `upstream` names the
    upstream harness's rule, e.g. "simple-evals' sampler asks a null reply again without
    limit"; without `exhausted_reason` only the cut-off one (the caller names the exhausted
    ones itself)."""
    if not isinstance(reask, dict):
        return []
    out = []
    exhausted = int(reask.get("reasks_exhausted") or 0) if exhausted_reason else 0
    short = int(reask.get("reasks_cut_short") or 0)
    if exhausted:
        how = f"after up to {reask.get('reask_limit')} re-asks"
        if short:
            how += (f"; {short} of them cut short before the limit, at the client's request "
                    "deadline or on a re-ask the endpoint did not answer")
        out.append(f"{exhausted} request(s) still came back with {content} when gbench stopped "
                   f"re-asking ({how}) and were scored {scored} ({upstream})")
    cut = int(reask.get("cut_off_not_reasked") or 0)
    if cut:
        out.append(f"{cut} reply(ies) cut off at the token budget (finish_reason length) with "
                   f"{content} were not asked again: the model's own outcome under the run's "
                   f"budget, scored {scored} ({upstream}, whatever the finish reason)")
    return out


def reask_summary(relay: "EndpointRelay", rule_text: str) -> Dict[str, Any]:
    """The result's `upstream_reask` record of an upstream_reask_relay: the rule, what it did
    (reask_record) and the relay's counts of the replies it saw, re-asked ones included: those
    cut off at the budget (never re-asked) and those that stopped on their own with reasoning and
    no content by class (answer_in_thinking_seen, tool_call_dropped_seen,
    unclassified_empty_seen; classify_empty_reply), with answer_in_thinking_rate_seen over
    replies_seen."""
    rec = relay.record()
    seen = int(rec.get("completion_replies") or 0)
    in_thinking = int(rec.get("replies_answer_in_thinking") or 0)
    return {"rule": rule_text, **relay.reask_record(),
            "replies_seen": seen,
            "replies_cut_off_seen": int(rec.get("replies_cut_off") or 0),
            "replies_empty_after_reasoning_seen": rec.get("replies_empty_after_reasoning"),
            # the replies that stopped on their own with reasoning and no content, by
            # classify_empty_reply, over every reply the relay saw (first replies and re-asks):
            # the suite's answer-in-thinking measurement before the re-asks, which the result's
            # `answer_in_thinking` (the replies still null after them) does not show
            "answer_in_thinking_seen": in_thinking,
            "tool_call_dropped_seen": int(rec.get("replies_tool_call_dropped") or 0),
            "unclassified_empty_seen": int(rec.get("replies_empty_unclassified") or 0),
            "answer_in_thinking_rate_seen": (round(in_thinking / seen, 6) if seen else None),
            "endpoint_failures": dict(rec.get("upstream_errors") or {})}


def host_pull_images(images: Sequence[str], workers: int = 1) -> Dict[str, Dict[str, Any]]:
    """Make each image present on the host's Docker daemon before a container with no Docker
    client configuration (a scaffold image such as SWE-agent-for-eval or MSWE-agent, or a
    harness image) starts task containers from it.

    `docker image inspect` first, and `docker pull` only for an image that is not there (the
    pull-when-missing rule of SWE-ReX's DockerDeployment, MSWE-agent's driver and mini-swe-agent),
    run by THIS process with its own environment, so the Docker CLI reads the operator's client
    configuration (`docker login`, credential helpers, DOCKER_CONFIG). Those containers get no
    Docker client configuration (HOME=/tmp), so no registry credential enters them, nor any task
    container they start, and a pull made there instead would be anonymous (Docker Hub's pull
    limit). Up to `workers` pulls run at once.

    Returns image -> {"state": "present" | "pulled" | "failed"}, with "error" (the CLI's
    message, credential values redacted) for a failed one."""
    import concurrent.futures
    from .base import redact_text, secret_values
    wanted = list(dict.fromkeys(i for i in images if i))

    def _one(image: str) -> Dict[str, Any]:
        try:
            if subprocess.run(["docker", "image", "inspect", image], capture_output=True,
                              text=True, timeout=120).returncode == 0:
                return {"state": "present"}
            r = subprocess.run(["docker", "pull", image], capture_output=True, text=True)
        except Exception as e:
            return {"state": "failed",
                    "error": redact_text(f"{type(e).__name__}: {e}", secret_values())}
        if r.returncode == 0:
            return {"state": "pulled"}
        msg = ((r.stderr or "") + (r.stdout or "")).strip()[-400:]
        return {"state": "failed",
                "error": redact_text(f"docker pull {image} failed: {msg}", secret_values())}
    if not wanted:
        return {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers or 1))) as ex:
        return dict(zip(wanted, ex.map(_one, wanted)))


def image_pull_summary(pulls: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The record of a host_pull_images call: counts, and each failed image with its error."""
    states = [p.get("state") for p in pulls.values()]
    return {"images": len(pulls), "present": states.count("present"),
            "pulled": states.count("pulled"),
            "failed": {i: p.get("error") for i, p in pulls.items() if p.get("state") == "failed"}}


# --------------------------------------------------------------------------------------------
# The cloud metadata server guard for the containers where the model under test runs code
# --------------------------------------------------------------------------------------------
#: The symbolic network name for a user-defined docker bridge network (see TASK_NETWORKS).
USER_DEFINED_NETWORK = "user-defined"
#: The two kinds of docker network the task containers of these suites use, both with network
#: access: docker's default bridge, where a container started with no `--network` lands (the
#: swebench harnesses, mini-swe-agent, SWE-ReX, MSWE-agent and the multi-swe-bench harness start
#: theirs that way, and OSWorld's docker provider too), and a user-defined bridge network, the
#: kind docker compose, Harbor, BenchFlow and alcatraz create for theirs at run time.
TASK_NETWORKS = ("bridge", USER_DEFINED_NETWORK)
#: The package hosts the task containers of terminal_bench (Harbor) and skillsbench (BenchFlow)
#: reach at run time. Each task's verifier script installs its test tools in the task container:
#: `apt-get update` and `apt-get install` from the Ubuntu or Debian archive (the task images are
#: ubuntu:24.04, python:*-slim-bookworm and debian:*-slim), uv from astral.sh's installer, which
#: downloads it from GitHub, and `uvx` or `pip`, which fetch Python from GitHub and the test
#: packages from PyPI; BenchFlow's agent install does the same. The gate checks that each name
#: resolves on the task networks (container_guard's `resolve`) before the run.
TASK_INSTALL_HOSTS = ("archive.ubuntu.com", "deb.debian.org", "astral.sh", "github.com",
                      "pypi.org", "files.pythonhosted.org")
#: The domains those installs fetch from (the hosts of TASK_INSTALL_HOSTS and the ones they send
#: downloads to). A name under one of them that failed to resolve in a verifier's output before
#: its tests ran is a harness DNS failure (verifier_dns_failure).
INSTALL_DOMAINS = ("ubuntu.com", "debian.org", "astral.sh", "github.com", "githubusercontent.com",
                   "pypi.org", "pythonhosted.org", "python.org", "pytorch.org", "nodejs.org",
                   "npmjs.org", "npmjs.com")

_TASK_NETWORK_LOCK = threading.Lock()
_USER_NETWORK_VERDICT: Dict[str, str] = {}
#: The names found to resolve on a user-defined bridge network. Each check runs on a network
#: gbench creates and removes, so container_guard's per-network cache does not carry them.
_USER_NETWORK_RESOLVED: set = set()
#: The Windows verdict (check_windows_task_network), cached like the others.
_WINDOWS_VERDICT: Dict[str, str] = {}


def reset_task_network_verdicts() -> None:
    """Forget the user-defined network verdict, the names resolved there and the Windows verdict
    (container_guard.reset_verdicts forgets the others)."""
    with _TASK_NETWORK_LOCK:
        _USER_NETWORK_VERDICT.clear()
        _USER_NETWORK_RESOLVED.clear()
        _WINDOWS_VERDICT.clear()


def _task_network_pending(networks: Sequence[str], resolve: Sequence[str] = ()) -> List[str]:
    """The networks among `networks` still to check: no cached `unreachable` verdict, or a name
    of `resolve` not yet found to resolve there (`none` needs nothing)."""
    from . import container_guard
    names = list(resolve or ())
    cached = container_guard.cached_verdicts()
    resolved = set(container_guard.cached_resolutions())
    pending: List[str] = []
    for net in dict.fromkeys(networks):
        if net == USER_DEFINED_NETWORK:
            with _TASK_NETWORK_LOCK:
                done = (_USER_NETWORK_VERDICT.get(net) == container_guard.UNREACHABLE
                        and all(n in _USER_NETWORK_RESOLVED for n in names))
        else:
            norm = (net or "").strip()
            norm = "bridge" if norm in ("", "default") else norm
            if norm == "none":
                continue
            done = (cached.get(norm) == container_guard.UNREACHABLE
                    and all((norm, n) in resolved for n in names))
        if not done:
            pending.append(net)
    return pending


def local_probe_image(images: Sequence[str] = (),
                      references: Sequence[str] = ()) -> Optional[str]:
    """The first of `images` present on the host's Docker daemon (`docker image inspect`), else
    the first tagged local image matching one of `references` (`docker images --filter
    reference=<pattern>`), else None. Nothing is pulled."""
    for image in dict.fromkeys(i for i in images if i):
        try:
            if subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                              capture_output=True, text=True, timeout=60).returncode == 0:
                return image
        except (OSError, subprocess.SubprocessError):
            continue
    for ref in dict.fromkeys(r for r in references if r):
        try:
            r = subprocess.run(["docker", "images", "--filter", f"reference={ref}", "--format",
                                "{{.Repository}}:{{.Tag}}"], capture_output=True, text=True,
                               timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode != 0:
            continue
        for line in (r.stdout or "").splitlines():
            line = line.strip()
            if line and not line.endswith(":<none>") and not line.startswith("<none>"):
                return line
    return None


@contextlib.contextmanager
def _probe_network(eval_name: str, docs_url: str):
    """A user-defined bridge network for one probe container (labelled
    container_guard.PROBE_LABEL), removed on exit; yields its name. infra_required when docker
    cannot create it."""
    import uuid
    from . import container_guard
    name = f"gbench-metadata-probe-{uuid.uuid4().hex[:12]}"
    why = ""
    try:
        r = subprocess.run(["docker", "network", "create", "--driver", "bridge", "--label",
                            container_guard.PROBE_LABEL, name],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            why = f"docker exit {r.returncode}: {(r.stderr or r.stdout or '').strip()[-300:]}"
    except (OSError, subprocess.SubprocessError) as e:
        why = f"{type(e).__name__}: {e}"
    if why:
        raise infra_required(
            eval_name, "gbench could not create the user-defined docker network it checks the "
            "cloud metadata server (169.254.169.254) from, the kind of network the suite's "
            f"harness starts its task containers on ({why}), so it could not check that the "
            "metadata server is blocked for them.", docs_url)
    try:
        yield name
    finally:
        try:
            subprocess.run(["docker", "network", "rm", name], capture_output=True, text=True,
                           timeout=120)
        except (OSError, subprocess.SubprocessError):
            logger.warning("[%s] could not remove the probe network %s", eval_name, name)


def _probe_user_defined_network(eval_name: str, image: str, docs_url: str,
                                resolve: Sequence[str] = ()) -> None:
    """container_guard's probe on a user-defined bridge network gbench creates for it (labelled
    container_guard.PROBE_LABEL, removed right after), resolving the names of `resolve` not yet
    found to resolve there; the verdict and the names are cached."""
    from . import container_guard
    names = list(resolve or ())
    with _TASK_NETWORK_LOCK:
        pending = [n for n in names if n not in _USER_NETWORK_RESOLVED]
        if (_USER_NETWORK_VERDICT.get(USER_DEFINED_NETWORK) == container_guard.UNREACHABLE
                and not pending):
            return
        with _probe_network(eval_name, docs_url) as name:
            container_guard.require_metadata_blocked(eval_name, image, name, docs_url,
                                                     resolve=pending)
        _USER_NETWORK_VERDICT[USER_DEFINED_NETWORK] = container_guard.UNREACHABLE
        _USER_NETWORK_RESOLVED.update(pending)


def require_task_network_blocked(eval_name: str, images: Sequence[str] = (), *,
                                 docs_url: Optional[str] = None,
                                 references: Sequence[str] = (), pull: Optional[str] = None,
                                 networks: Sequence[str] = TASK_NETWORKS,
                                 hint: str = "", resolve: Sequence[str] = ()) -> None:
    """Hard-error (infra_required) unless the cloud metadata server (169.254.169.254) is
    unreachable from the docker networks the suite's task containers use, where the model under
    test runs code with network access (its commands, or the tests over its patch).

    `networks` are docker network modes as `docker run --network` takes them, or
    USER_DEFINED_NETWORK; the default TASK_NETWORKS covers docker's default bridge and a
    user-defined bridge network (the kind a harness creates for its task containers at run time,
    which cannot be probed before it exists: gbench creates one of its own for the probe). Each
    is checked with container_guard.require_metadata_blocked, whose verdicts are cached for the
    rest of the process, so a later call only probes what no earlier call settled.

    `resolve` names the hosts the task containers need at run time (the package hosts a task's
    verifier installs from, for example): the same probe container resolves each of them on every
    network, and a name that does not resolve is infra_required as well (container_guard's
    unresolved_message: on GCE the VPC DNS resolver is the metadata server's address).

    The probe runs in the first of `images` present on the host, else in `pull` if that is
    present, else in a local image matching one of `references` (local_probe_image). Nothing is
    pulled, by the probe or for it. `pull` names an image the run itself starts (its first task
    image, for example): without any usable image the call is infra_required, naming the images
    (the first five when there are more), the `docker pull <pull>` to run first, and `hint` (the
    suite's build command, for example).
    gbench never runs sudo or changes the host's firewall: the fix it names is
    container_guard.FIREWALL_COMMAND."""
    from . import container_guard
    docs_url = docs_url or f"docs/evals/{eval_name}.md"
    names = list(dict.fromkeys(n for n in (resolve or ()) if n))
    pending = _task_network_pending(networks, names)
    if not pending:
        return
    image = local_probe_image([*images, *([pull] if pull else [])], references)
    if image is None:
        wanted = ([i for i in dict.fromkeys(images) if i and i != pull]
                  + [f"{r} (any tag)" for r in references] + ([pull] if pull else []))
        if len(wanted) > 6:
            wanted = wanted[:5] + [f"{len(wanted) - 5} more"]
        steps = f" Pull one on this host first: `docker pull {pull}`." if pull else ""
        raise infra_required(
            eval_name, "gbench checks that the cloud metadata server (169.254.169.254) is blocked "
            "for the containers where the model under test runs code, with a short-lived probe "
            "container of an image the suite already has on this host (the probe never pulls "
            f"one), and none of {', '.join(wanted) or 'its images'} is present." + steps
            + (f" {hint}" if hint else ""), docs_url)
    for net in pending:
        if net == USER_DEFINED_NETWORK:
            _probe_user_defined_network(eval_name, image, docs_url, names)
        else:
            container_guard.require_metadata_blocked(eval_name, image, net, docs_url,
                                                     resolve=names)


def require_harness_network_blocked(eval_name: str, harness_image: Optional[str],
                                    instance_ids: Sequence[str] = (),
                                    instance_image: Optional[Callable[[str], str]] = None,
                                    namespace: Optional[str] = None) -> None:
    """require_task_network_blocked for the task containers of a swebench harness pass: the
    probe runs in the harness image when the harness runs in one; for the host interpreter's
    harness, in a local instance image (`instance_image(id)`, else the swebench naming under
    `namespace`). Without one present it is infra_required, naming `docker pull` of the first
    submitted instance's image; nothing is pulled for the probe."""
    if harness_image:
        require_task_network_blocked(
            eval_name, [harness_image],
            hint=f"Build the harness image {harness_image} (see the suite's doc).")
        return
    from .swe_agentic import swebench_image
    images = [instance_image(i) if instance_image else swebench_image(i, namespace or "swebench")
              for i in instance_ids]
    require_task_network_blocked(eval_name, images, pull=images[0] if images else None)


# --------------------------------------------------------------------------------------------
# DNS failures inside a task container: harness errors, not the model's outcome
# --------------------------------------------------------------------------------------------
#: A resolver failure that names the host: apt, apt (older wording), curl (and cargo, which
#: prints curl's error), wget, urllib3 2.x, Maven ("Unknown host oss.sonatype.org"), a Java
#: stack ("java.net.UnknownHostException: repo.maven.apache.org"), Node's getaddrinfo (npm,
#: pnpm, yarn: "getaddrinfo EAI_AGAIN registry.npmjs.org") and Go's resolver ("dial tcp: lookup
#: proxy.golang.org").
_DNS_HOST_RES = (
    re.compile(r"Temporary failure resolving '([^'\s]+)'"),
    re.compile(r"Could not resolve '([^'\s]+)'"),
    re.compile(r"Could not resolve host: ([A-Za-z0-9.-]+)"),
    re.compile(r"unable to resolve host address [\u2018'\"]?([A-Za-z0-9.-]+)"),
    re.compile(r"Failed to resolve '([^'\s]+)'"),
    re.compile(r"Unknown host '?([A-Za-z0-9.-]+)"),
    re.compile(r"UnknownHostException: ([A-Za-z0-9.-]+)"),
    re.compile(r"getaddrinfo (?:EAI_AGAIN|ENOTFOUND|EAI_FAIL|EAI_NONAME) ([A-Za-z0-9.-]+)"),
    re.compile(r"dial tcp: lookup ([A-Za-z0-9.-]+)"),
)
#: A resolver failure that does not name the host on its own line (uv's error chain, pip's retry
#: warnings, getaddrinfo's messages): the host is taken from a URL on that line or one of the ten
#: before it.
_DNS_ERROR_RE = re.compile(
    r"dns error|failed to lookup address|Temporary failure in name resolution|"
    r"Name or service not known|NameResolutionError|getaddrinfo failed|"
    r"No address associated with hostname|nodename nor servname provided", re.I)
_URL_HOST_RE = re.compile(r"(?:https?://|host=')([A-Za-z0-9.-]+)")
#: pip's retry warning names only the index path (`/simple/<package>/`) of its default index.
_PIP_INDEX_RE = re.compile(r"': /simple/")
#: The header pytest prints when its session starts: the tests ran.
_TEST_SESSION_RE = re.compile(r"=+ test session starts =+")
#: The files through which a process in the container changes its own name resolution.
_RESOLVER_FILES_RE = re.compile(r"/etc/(?:resolv\.conf|hosts|nsswitch\.conf)\b")


def _under(host: str, domains: Sequence[str]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in domains)


def dns_failure_hosts(text: str, domains: Sequence[str] = INSTALL_DOMAINS) -> List[str]:
    """The hosts under `domains` that failed to resolve according to `text` (a verifier's or an
    install's output): apt's "Temporary failure resolving 'host'" / "Could not resolve 'host'",
    curl's "Could not resolve host: host", wget's "unable to resolve host address", urllib3's
    "Failed to resolve 'host'", a resolver error (uv's "dns error", "Temporary failure in name
    resolution", ...) with the host's URL on its line or one of the ten before it, and pip's
    retry warning for an index path `/simple/` with a resolver error (pypi.org, pip's default
    index). Sorted, each once."""
    found = set()
    lines = (text or "").splitlines()
    for i, line in enumerate(lines):
        for rx in _DNS_HOST_RES:
            found.update(m.group(1) for m in rx.finditer(line))
        if _DNS_ERROR_RE.search(line):
            for before in lines[max(0, i - 10):i + 1]:
                found.update(m.group(1) for m in _URL_HOST_RE.finditer(before))
            if _PIP_INDEX_RE.search(line):
                found.add("pypi.org")
    return sorted({h.lower().rstrip(".") for h in found if _under(h, domains)})


def verifier_dns_failure(verifier_output: str) -> List[str]:
    """The package hosts (INSTALL_DOMAINS) that failed to resolve in a task verifier's output
    while it installed its test tools, when the tests never started (no pytest session header),
    else []. Such a verifier scored nothing: its reward is a harness DNS failure, not the model's
    outcome. Once the tests have started their verdict stands."""
    if _TEST_SESSION_RE.search(verifier_output or ""):
        return []
    return dns_failure_hosts(verifier_output)


def touched_resolver(agent_text: str) -> bool:
    """True when the agent's transcript names /etc/resolv.conf, /etc/hosts or
    /etc/nsswitch.conf: the model may have changed name resolution in its own container, so a DNS
    failure there stays the model's outcome."""
    return bool(_RESOLVER_FILES_RE.search(agent_text or ""))


#: The evidence a single trial's DNS failure needs before it counts as a harness error.
DNS_CORROBORATION = ("a DNS failure in another task container of the same run, or a fresh probe "
                     "container after the run that cannot resolve one of the same hosts either")


def corroborated_dns_failures(
        candidates: Dict[str, Sequence[str]],
        recheck: Optional[Callable[[List[str]], Optional[List[str]]]] = None,
        what: str = "the verifier",
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Split `candidates` ({trial, task or instance: the package hosts `what` (its verifier, or
    the harness's test run) could not resolve before its tests ran}, each from a task container
    whose transcript names no resolver file) into (harness errors {key: evidence}, the model's
    outcome {key: note}).

    A resolver failure inside one task container can be the model's own doing (it removed a
    resolver library, filled the disk, or broke name resolution some other way without naming
    /etc/resolv.conf, /etc/hosts or /etc/nsswitch.conf), so it counts as a harness error only
    when something outside that container corroborates it (DNS_CORROBORATION): another candidate
    in the same run (independent containers that fail to resolve point at the host's DNS, not at
    one model's commands), or, for a lone candidate, `recheck(hosts)` (a fresh probe container
    after the run; unresolved_after_run) returning one of its hosts. `recheck` returns the hosts
    that still fail, or None when it could not tell. Anything else stays the model's outcome,
    with a note saying why."""
    keys = [k for k, hosts in candidates.items() if hosts]
    if not keys:
        return {}, {}
    if len(keys) >= 2:
        why = (f"{len(keys)} task containers of this run could not resolve package hosts "
               "before their tests ran")
        return {k: why for k in keys}, {}
    key = keys[0]
    hosts = sorted(dict.fromkeys(candidates[key]))
    still: Optional[List[str]] = None
    if recheck is not None:
        try:
            still = recheck(hosts)
        except Exception as e:  # a recheck that cannot run corroborates nothing
            logger.warning("the DNS recheck after the run failed (%s: %s)", type(e).__name__, e)
            still = None
    failing = sorted(set(still or ()) & set(hosts))
    if failing:
        return {key: (f"a fresh probe container after the run could not resolve "
                      f"{', '.join(failing)} either")}, {}
    if still is None:
        after = "no fresh probe after the run could check the hosts"
    else:
        after = f"a fresh probe container after the run resolved {', '.join(hosts)}"
    return {}, {key: (f"{what} could not resolve {', '.join(hosts)} before its tests ran, "
                      f"but nothing outside this task container corroborates a DNS failure (no "
                      f"other task container of the run hit one, and {after}), and the model's "
                      "commands or code ran in that container, so it may be the model's own "
                      "doing: it stays the model's outcome")}


def unresolved_after_run(eval_name: str, images: Sequence[str], names: Sequence[str], *,
                         references: Sequence[str] = (), docs_url: Optional[str] = None,
                         networks: Sequence[str] = TASK_NETWORKS) -> Optional[List[str]]:
    """The names of `names` a fresh probe container cannot resolve now, on each of `networks`
    (docker's default bridge, and a user-defined bridge network gbench creates and removes), with
    container_guard's probe and not its cache (a name the resolver answered does not exist,
    container_guard.NONEXISTENT_NAME_WHYS, is not counted: that is not a DNS outage); [] when
    every name resolves; None when no answer could be had (no image of `images` / `references`
    on the host, a docker error, no clear verdict, a metadata server that answers). The probe
    runs in an image already on the host (local_probe_image); nothing is pulled."""
    from . import container_guard
    docs_url = docs_url or f"docs/evals/{eval_name}.md"
    names = [n for n in dict.fromkeys(names or ()) if n]
    if not names:
        return []
    image = local_probe_image(images, references)
    if image is None:
        logger.warning("[%s] no probe image on the host for the DNS recheck after the run",
                       eval_name)
        return None
    failed: set = set()
    try:
        for net in dict.fromkeys(networks):
            if net == USER_DEFINED_NETWORK:
                with _probe_network(eval_name, docs_url) as name:
                    verdict, _, dns = container_guard._run_probe(eval_name, image, name, docs_url,
                                                                 names)
            else:
                verdict, _, dns = container_guard._run_probe(eval_name, image, net, docs_url,
                                                             names)
            if verdict != container_guard.UNREACHABLE:
                return None
            # Only an outage (a timeout, EAI_AGAIN, no answer) corroborates a DNS failure; a name
            # the resolver answered does not exist (NXDOMAIN) is not an outage.
            failed.update(n for n in names if (dns.get(n) or ("",))[0] == "fail"
                          and (tuple(dns.get(n) or ()) + ("", ""))[1]
                          not in container_guard.NONEXISTENT_NAME_WHYS)
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as e:
        logger.warning("[%s] the DNS recheck after the run gave no answer (%s)", eval_name,
                       _clip_text(str(e)))
        return None
    return sorted(failed)


def read_texts(paths: Sequence[str], limit: int = 8 * 1024 * 1024) -> str:
    """The files at `paths` as text (undecodable bytes replaced; each cut at `limit` bytes; a
    file that cannot be read is skipped), joined by newlines."""
    parts: List[str] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                parts.append(f.read(limit))
        except OSError:
            continue
    return "\n".join(parts)


# --------------------------------------------------------------------------------------------
# The metadata server check for Windows containers (a Docker engine running Windows containers)
# --------------------------------------------------------------------------------------------
#: The network the Windows check covers: docker's default network, where a container started with
#: no network option lands (`nat` on a Windows engine).
WINDOWS_DEFAULT_NETWORK = "default"
#: The programs the Windows probe runs in the image, in order: Windows PowerShell, then PowerShell
#: 7 for an image without it.
WINDOWS_PROBE_PROGRAMS = ("powershell", "pwsh")
_WINDOWS_MARK = "GBENCH_METADATA_PROBE="


def docker_engine_os() -> Tuple[Optional[str], str]:
    """(the Docker engine's OSType from `docker info`, lowercased, or None when it could not be
    read; the reason when it could not)."""
    try:
        r = subprocess.run(["docker", "info", "--format", "{{.OSType}}"], capture_output=True,
                           text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"{type(e).__name__}: {e}"
    if r.returncode != 0:
        return None, f"docker exit {r.returncode}: {(r.stderr or r.stdout or '').strip()[-300:]}"
    os_type = (r.stdout or "").strip().lower()
    return (os_type or None), ("" if os_type else "docker info reported no OSType")


def windows_probe_script(host: Optional[str] = None, port: Optional[int] = None,
                         timeout_s: Optional[float] = None, host6: Optional[str] = None) -> str:
    """The PowerShell probe: open a TCP connection to host:port (default the metadata server,
    169.254.169.254:80) with .NET's TcpClient, and one to [host6]:port (default its IPv6 address,
    container_guard.METADATA_HOST6, fd20:ce::254) with an IPv6 TcpClient, close both, print the
    verdict. Nothing is sent on either connection. Only a connection that opens within
    `timeout_s` (default container_guard.PROBE_CONNECT_TIMEOUT_S) is `reachable`; a timeout, or a
    connection refused, unreachable or forbidden (SocketErrorCode ConnectionRefused,
    HostUnreachable, NetworkUnreachable, TimedOut, AccessDenied) is `unreachable`, and for IPv6
    also no IPv6 support or address (AddressFamilyNotSupported, AddressNotAvailable); any other
    error is `error <what>`, which gives no verdict. As in container_guard's Linux probe, either
    family opening is `reachable ipv4` / `reachable ipv6`; else an error of either is the
    verdict; else `unreachable <ipv4 detail> ipv6:<ipv6 detail>`."""
    from . import container_guard as cg
    host = host or cg.METADATA_HOST
    host6 = host6 or cg.METADATA_HOST6
    port = int(port or cg.METADATA_PORT)
    ms = int(float(timeout_s or cg.PROBE_CONNECT_TIMEOUT_S) * 1000)
    return (
        "$ErrorActionPreference = 'Stop'\n"
        "$gone = @('ConnectionRefused', 'HostUnreachable', 'NetworkUnreachable', 'TimedOut', "
        "'AccessDenied')\n"
        "$gone6 = $gone + @('AddressFamilyNotSupported', 'AddressNotAvailable')\n"
        "function Why($err, $codes) {\n"
        "  $e = $err.Exception\n"
        "  while ($e.InnerException) { $e = $e.InnerException }\n"
        "  if ($e -is [System.Net.Sockets.SocketException]) {\n"
        "    $code = [string]$e.SocketErrorCode\n"
        "    if ($codes -contains $code) { return 'unreachable ' + $code }\n"
        "    return 'error ' + $code\n"
        "  }\n"
        "  return 'error ' + $e.GetType().Name\n"
        "}\n"
        "$c = New-Object System.Net.Sockets.TcpClient\n"
        "try {\n"
        f"  if ($c.ConnectAsync('{host}', {port}).Wait({ms})) {{ $v = 'reachable' }}\n"
        "  else { $v = 'unreachable timeout' }\n"
        "} catch { $v = Why $_ $gone } finally { $c.Close() }\n"
        "$c6 = $null\n"
        "try {\n"
        "  $c6 = New-Object System.Net.Sockets.TcpClient("
        "[System.Net.Sockets.AddressFamily]::InterNetworkV6)\n"
        f"  if ($c6.ConnectAsync([System.Net.IPAddress]::Parse('{host6}'), {port}).Wait({ms})) "
        "{ $v6 = 'reachable' }\n"
        "  else { $v6 = 'unreachable timeout' }\n"
        "} catch { $v6 = Why $_ $gone6 } finally { if ($c6) { $c6.Close() } }\n"
        "if ($v -eq 'reachable') { $verdict = 'reachable ipv4' }\n"
        "elseif ($v6 -eq 'reachable') { $verdict = 'reachable ipv6' }\n"
        "elseif ($v.StartsWith('error')) { $verdict = $v }\n"
        "elseif ($v6.StartsWith('error')) { $verdict = $v6 + ' ipv6' }\n"
        "else { $verdict = $v + ' ipv6:' + ($v6 -split ' ', 2)[-1] }\n"
        f"[Console]::Out.WriteLine('{_WINDOWS_MARK}' + $verdict)\n")


def windows_probe_argv(image: str, name: str, program: str, script: str) -> List[str]:
    """`docker run` of the Windows probe: `program` (powershell or pwsh) in `image` on docker's
    default network (no `--network`, as Win-Agent and RepoLaunch start their task containers),
    `--pull never`, the script passed as -EncodedCommand (base64 of its UTF-16LE text, so no
    quoting reaches the Windows command line)."""
    import base64
    from . import container_guard as cg
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return ["docker", "run", "--rm", "--pull", "never", "--name", name, "--label", cg.PROBE_LABEL,
            "--entrypoint", program, image, "-NoLogo", "-NoProfile", "-NonInteractive",
            "-EncodedCommand", encoded]


def _windows_verdict(stdout: str) -> Optional[Tuple[str, str]]:
    for line in (stdout or "").splitlines():
        line = line.strip()
        if line.startswith(_WINDOWS_MARK):
            word, _, rest = line[len(_WINDOWS_MARK):].partition(" ")
            return word, rest.strip()
    return None


def _clip_text(text: str, limit: int = 300) -> str:
    from .base import redact_text
    text = redact_text(" ".join(str(text or "").split()))
    return text if len(text) <= limit else text[:limit] + "..."


def windows_reachable_message(family: str = "ipv4") -> str:
    """The reason when the metadata server answers a Windows container (over IPv6 when `family`
    is "ipv6")."""
    from . import container_guard as cg
    where = (f"[{cg.METADATA_HOST6}]:{cg.METADATA_PORT}, its IPv6 address" if family == "ipv6"
             else f"{cg.METADATA_HOST}:{cg.METADATA_PORT}")
    return (f"the cloud metadata server ({where}, which hands out this VM's "
            "service-account token) is reachable from a Windows container on docker's default "
            "network (nat on a Windows engine), where Win-Agent and the harness start the task "
            "containers the model under test runs code in (both start them with no network "
            f"option). Block TCP from those containers to {cg.METADATA_HOST} (and to "
            f"{cg.METADATA_HOST6} when their network has IPv6) on every port but 53 "
            "(on GCE the VPC DNS resolver answers at the same address) without cutting the VM's "
            "own access to it, which its guest agent and credentials use: an outbound rule for "
            "the address on the whole host would cut both. gbench names no Windows firewall "
            "command for this, never changes the host's firewall itself, and checks again on the "
            "next run.")


def check_windows_task_network(eval_name: str, images: Sequence[str] = (), *,
                               references: Sequence[str] = (),
                               hint: str = "") -> Tuple[bool, str]:
    """(True, "") when the cloud metadata server is unreachable from a Windows container on
    docker's default network, else (False, the reason). It never passes on a verdict it could
    not establish.

    The engine must run Windows containers (`docker info` OSType windows). The probe runs in the
    first of `images` present on the host, else in a local image matching one of `references`
    (local_probe_image; nothing is pulled): powershell, else pwsh, with windows_probe_script,
    which opens and closes a TCP connection to 169.254.169.254:80 and one to [fd20:ce::254]:80
    (the metadata server's IPv6 address, where a container network with IPv6 reaches it) and
    sends nothing. No image present, no verdict line (neither program starts, a docker error), an
    `error` verdict or a docker timeout is (False, why); `reachable` over either family is
    (False, windows_reachable_message(family)). An `unreachable` verdict (both families) is
    cached for the rest of the process."""
    import uuid
    from . import container_guard as cg
    with _TASK_NETWORK_LOCK:
        if _WINDOWS_VERDICT.get(WINDOWS_DEFAULT_NETWORK) == cg.UNREACHABLE:
            return True, ""
    what = ("so gbench could not check that the cloud metadata server (169.254.169.254) is "
            "blocked for the Windows task containers where the model under test runs code.")
    engine, why = docker_engine_os()
    if engine is None:
        return False, (f"gbench could not read the Docker engine's OSType (`docker info`: "
                       f"{why}), {what}")
    if engine != "windows":
        return False, (f"the Docker engine runs {engine} containers (`docker info` OSType "
                       f"{engine!r}), and the Windows tab's task containers are Windows "
                       "containers: gbench checks that the cloud metadata server "
                       "(169.254.169.254) is blocked for them with a probe container on the "
                       "Windows-containers engine that runs them, which this is not.")
    image = local_probe_image(images, references)
    if image is None:
        first = next((i for i in images if i), None)
        return False, ("gbench checks that the cloud metadata server (169.254.169.254) is "
                       "blocked for the Windows task containers where the model under test runs "
                       "code with a short-lived probe container of one of the suite's own "
                       "Windows images already on this host (the probe never pulls one), and "
                       "none is present"
                       + (f"; pull one on this host first, for example `docker pull {first}`."
                          if first else ".") + (f" {hint}" if hint else ""))
    tried: List[str] = []
    for program in WINDOWS_PROBE_PROGRAMS:
        name = f"gbench-metadata-probe-{uuid.uuid4().hex[:12]}"
        try:
            proc = subprocess.run(windows_probe_argv(image, name, program, windows_probe_script()),
                                  capture_output=True, text=True,
                                  timeout=cg.PROBE_DOCKER_TIMEOUT_S)
        except FileNotFoundError:
            return False, f"the docker CLI is not on PATH, {what}"
        except subprocess.TimeoutExpired:
            try:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True,
                               timeout=cg.PROBE_RM_TIMEOUT_S)
            except (OSError, subprocess.SubprocessError):
                logger.warning("[%s] could not remove the Windows probe container %s",
                               eval_name, name)
            return False, (f"the Windows metadata-server probe container ({program} in {image}) "
                           f"did not finish within {cg.PROBE_DOCKER_TIMEOUT_S} s, {what}")
        except OSError as e:
            return False, f"the Windows metadata-server probe could not start ({e}), {what}"
        found = _windows_verdict(proc.stdout)
        if found is None:
            # No verdict line: the program is not in the image, or docker could not start it.
            tried.append(f"{program}: docker exit {proc.returncode}: "
                         f"{_clip_text(proc.stderr or proc.stdout) or 'no output'}")
            continue
        word, detail = found
        if word == cg.UNREACHABLE:
            with _TASK_NETWORK_LOCK:
                _WINDOWS_VERDICT[WINDOWS_DEFAULT_NETWORK] = cg.UNREACHABLE
            logger.info("[%s] Windows metadata-server probe on docker's default network: %s",
                        eval_name, " ".join((word, detail)).strip())
            return True, ""
        if word == cg.REACHABLE:
            return False, windows_reachable_message(
                "ipv6" if detail.startswith("ipv6") else "ipv4")
        return False, (f"the Windows metadata-server probe ({program} in {image}) gave no "
                       f"verdict ({_clip_text(' '.join((word, detail)))}), {what}")
    return False, (f"the Windows metadata-server probe gave no verdict in {image} (neither "
                   f"powershell nor pwsh answered: {'; '.join(tried)}), {what}")
