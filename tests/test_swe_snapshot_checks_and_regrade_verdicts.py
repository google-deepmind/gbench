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
"""SWE family: pinned snapshot checks, re-grade verdicts, advisory v5 failure ids, report retry,
no cost opt-in.

* A cached dataset snapshot is used only when it holds the pinned split (row count, the harness
  columns, the content fingerprint); anything else is moved aside and fetched again, or
  hard-errors.
* SWE-bench Pro V2 re-grade trials are verdicts on the patch when the verifier ran (a reward,
  or a verifier timeout / missing reward on the patched tree, which scores 0 as in Harbor);
  only failures outside the patch's control are infrastructure (retried, then partial).
* swebench 5.x's infra_failure_ids / ambiguous_failure_ids / failure_reasons stay advisory.
* A harness that scored instances but wrote no report gets its report step run again alone.
* A provisioned machine runs swe_bench_pro with no opt-in variable.

Hermetic: no container, no endpoint, no network; every cache is under tmp_path.
"""

import asyncio
import glob
import json
import os
import sys
import types

import pytest

from gbench.runners.eval_suites import copilot_bench_swe as CB
from gbench.runners.eval_suites import swe_agentic as A
from gbench.runners.eval_suites import swe_bench_live as LV
from gbench.runners.eval_suites import swe_bench_multilingual as ML
from gbench.runners.eval_suites import swe_bench_pro as P
from gbench.runners.eval_suites import swebench_common as C


@pytest.fixture(autouse=True)
def _no_metadata_probe(monkeypatch):
    """The cloud metadata server gate (swebench_common.require_task_network_blocked) starts probe
    containers; tests/test_guard_swe_agents.py covers it, here it is stubbed."""
    monkeypatch.setattr(C, "require_task_network_blocked", lambda *a, **k: None)


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def _scratch(monkeypatch, tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    for var in ("GBENCH_SWE_BENCH_PRO_TASK_ENDPOINT", "GBENCH_SWE_BENCH_PRO_TASK_HOST",
                "GBENCH_SWE_BENCH_PRO_IMAGE", "GBENCH_SWEBENCH_V5_IMAGE", "GBENCH_SWE_OMP_THREADS",
                "GBENCH_SHARD", "GBENCH_SWEBENCH_PRO_AGENTIC", "GBENCH_SWE_BENCH_PRO_MODE",
                "GBENCH_SWE_BENCH_PRO_TEMPERATURE", "GBENCH_SWE_BENCH_PRO_RUN",
                "SWE_BENCH_PRO_RUN", "GBENCH_SWE_BENCH_PRO_HARNESS_DIR",
                "SWE_BENCH_PRO_HARNESS_DIR", "GBENCH_PREREQS_DIR"):
        monkeypatch.delenv(var, raising=False)


def _proc(rc=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


def _fake_datasets(monkeypatch, rows):
    calls = []
    fake = types.ModuleType("datasets")

    def _load(name, *a, **k):
        calls.append((name, a, k))
        if isinstance(rows, Exception):
            raise rows
        return [dict(r) for r in rows]
    fake.load_dataset = _load
    monkeypatch.setitem(sys.modules, "datasets", fake)
    return calls


def _rows(mod, n=None, split="test", **extra):
    n = mod.SPLIT_ROWS[split] if n is None else n
    return [{**{c: "x" for c in mod.SNAPSHOT_COLUMNS}, "repo": "o/r",
             "instance_id": f"o__r-{i}", "problem_statement": f"issue {i}", **extra}
            for i in range(n)]


def _snapshot_path(tmp_path, dataset, revision, split="test"):
    safe = dataset.replace("/", "__")
    return tmp_path / "cache" / "swebench_snapshots" / f"{safe}@{revision}" / f"{split}.jsonl"


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# ============================================================================================
# pinned snapshot: checked before it is written and before it is reused
# ============================================================================================
def test_a_poisoned_cached_snapshot_is_moved_aside_and_fetched_again(monkeypatch, tmp_path):
    """The reported case: a 1-row, 3-column file at the pinned Verified path."""
    path = _snapshot_path(tmp_path, CB._DATASET, CB._REVISION)
    fake = [{"repo": "astropy/astropy", "instance_id": "astropy__astropy-1",
             "problem_statement": "Fix the parser bug."}]
    _write(path, fake)
    # stand-in rows for the pinned split carry their own content fingerprint
    monkeypatch.setitem(CB.SPLIT_SHA256, "test", C.snapshot_sha256(_rows(CB)))
    calls = _fake_datasets(monkeypatch, _rows(CB))
    samples = CB._load_copilot_bench_swe_dataset()
    assert len(samples) == 500 and calls and calls[0][2]["revision"] == CB._REVISION
    aside = glob.glob(f"{path}.invalid-*")
    assert len(aside) == 1 and json.loads(open(aside[0]).read()) == fake[0]
    assert len(C.read_swe_snapshot(str(path), "x")) == 500
    # the rewritten file is reused offline afterwards
    _fake_datasets(monkeypatch, OSError("offline"))
    assert len(CB._load_copilot_bench_swe_dataset(limit=3)) == 3


def test_prereq_with_a_poisoned_snapshot_refetches_before_the_v5_probe(monkeypatch, tmp_path):
    path = _snapshot_path(tmp_path, CB._DATASET, CB._REVISION)
    _write(path, [{"repo": "a/a", "instance_id": "a__a-1", "problem_statement": "p"}])
    monkeypatch.setitem(CB.SPLIT_SHA256, "test", C.snapshot_sha256(_rows(CB)))
    _fake_datasets(monkeypatch, _rows(CB))
    seen = {}
    monkeypatch.setattr(CB, "check_swebench_v5_prereq",
                        lambda snap, split: seen.update(n=len(C.read_swe_snapshot(snap, "x")))
                        or (True, ""))
    assert CB.check_copilot_bench_swe_prerequisites() == (True, "")
    assert seen["n"] == 500


def test_a_poisoned_snapshot_with_the_hub_unreachable_hard_errors(monkeypatch, tmp_path):
    path = _snapshot_path(tmp_path, ML._DATASET, ML._REVISION)
    _write(path, [{"repo": "a/a", "instance_id": "a__a-1", "problem_statement": "p"}])
    _fake_datasets(monkeypatch, OSError("no network"))
    ok, why = ML.check_swe_bench_multilingual_prerequisites()
    assert not ok and "Could not load dataset" in why and "no network" in why
    assert glob.glob(f"{path}.invalid-*")


@pytest.mark.parametrize("bad,problem", [
    (lambda rows: rows[:-1], "(499 rows, not the 500 of the pinned split)"),
    (lambda rows: [{k: v for k, v in r.items() if k != "eval_script"} for r in rows],
     "missing the harness columns ['eval_script']"),
    (lambda rows: rows[:-1] + [dict(rows[-1], eval_script="y")], "content sha256"),
    (lambda rows: rows[:-1] + [dict(rows[0], instance_id="")], "without instance_id"),
    (lambda rows: [], "(no rows)"),
])
def test_fetched_rows_that_are_not_the_pinned_split_are_refused(monkeypatch, tmp_path, bad,
                                                                problem):
    _fake_datasets(monkeypatch, bad(_rows(CB)))
    with pytest.raises(RuntimeError, match="not the pinned dataset") as e:
        C.pinned_swe_snapshot(CB._DATASET, "test", CB._REVISION, "copilot_bench_swe",
                              columns=CB.SNAPSHOT_COLUMNS, rows=500,
                              sha256=C.snapshot_sha256(_rows(CB)))
    assert problem in str(e.value)
    assert not _snapshot_path(tmp_path, CB._DATASET, CB._REVISION).exists()


def test_an_unreadable_cached_snapshot_is_replaced(monkeypatch, tmp_path):
    path = _snapshot_path(tmp_path, LV._DATASET, LV._REVISION, "lite")
    path.parent.mkdir(parents=True)
    path.write_text('{"instance_id": "a__a-1", "problem_state')   # truncated write
    _fake_datasets(monkeypatch, _rows(LV, split="lite"))
    got = C.pinned_swe_snapshot(LV._DATASET, "lite", LV._REVISION, "swe_bench_live",
                                columns=LV.SNAPSHOT_COLUMNS, rows=LV.SPLIT_ROWS["lite"])
    assert got == str(path) and len(C.read_swe_snapshot(got, "x")) == 300
    assert len(glob.glob(f"{path}.invalid-*")) == 1


def test_a_valid_cached_snapshot_is_reused_without_a_fetch(monkeypatch, tmp_path):
    path = _snapshot_path(tmp_path, ML._DATASET, ML._REVISION)
    _write(path, _rows(ML))
    calls = _fake_datasets(monkeypatch, OSError("must not fetch"))
    assert C.pinned_swe_snapshot(ML._DATASET, "test", ML._REVISION, "x",
                                 columns=ML.SNAPSHOT_COLUMNS, rows=300) == str(path)
    assert calls == [] and not glob.glob(f"{path}.invalid-*")


def test_a_bad_snapshot_that_cannot_be_moved_aside_is_a_hard_error(monkeypatch, tmp_path):
    path = _snapshot_path(tmp_path, CB._DATASET, CB._REVISION)
    _write(path, [{"instance_id": "a__a-1", "problem_statement": "p"}])
    _fake_datasets(monkeypatch, _rows(CB))

    def _deny(src, dst):
        raise PermissionError("read-only")
    monkeypatch.setattr(C.os, "replace", _deny)
    with pytest.raises(RuntimeError, match="could not be moved aside") as e:
        C.pinned_swe_snapshot(CB._DATASET, "test", CB._REVISION, "copilot_bench_swe",
                              columns=CB.SNAPSHOT_COLUMNS, rows=500)
    assert "missing the harness columns" in str(e.value)


@pytest.mark.parametrize("mod,run,split", [
    (CB, "run_copilot_bench_swe", "test"),
    (ML, "run_swe_bench_multilingual", "test"),
    (LV, "run_swe_bench_live", "lite"),
])
def test_suites_hand_the_snapshot_checks_to_both_modes(monkeypatch, mod, run, split):
    monkeypatch.setattr(mod, [n for n in vars(mod) if n.startswith("check_")
                              and n.endswith("_prerequisites")][0], lambda *a: (True, ""))
    monkeypatch.setattr(A, "require_agent", lambda *a: None)
    monkeypatch.setattr(A, "require_agent_v2", lambda *a: None)
    # swe_bench_live runs each leaderboard setting in turn, each with its own split's checks
    # (here its Python tabs, the ones these two entry points run)
    if mod is LV:
        monkeypatch.setattr(LV, "SETTINGS", LV.PYTHON_SETTINGS)
        monkeypatch.setattr(LV, "check_sweagent_prerequisites", lambda: (True, ""))
        # the suite's own check by name (the first check_* name is the Windows one)
        monkeypatch.setattr(LV, "check_swe_bench_live_prerequisites", lambda *a: (True, ""))
    want = list(getattr(mod, "SETTINGS", (split,)))
    agentic_entry = ("execute_sweagent_python" if mod is LV       # SWE-agent-for-eval
                     else "execute_swebench_agentic")
    for mode, entry in (("agentic", agentic_entry), ("oneshot", "execute_swebench")):
        calls = []
        monkeypatch.setattr(mod, entry,
                            lambda *a, **k: calls.append((a, k)) or {"status": "success"})
        monkeypatch.setenv(A.mode_env_var(mod.__name__.rsplit(".", 1)[-1]), mode)
        getattr(mod, run)("m", "http://x/v1", 1)
        assert [a[5] for a, _ in calls] == want
        assert all(k["snapshot_columns"] == mod.SNAPSHOT_COLUMNS for _, k in calls)
        assert [k["snapshot_rows"] for _, k in calls] == [mod.SPLIT_ROWS[s] for s in want]


def test_agentic_entry_checks_the_snapshot_and_keeps_the_knobs_out_of_the_runner(monkeypatch,
                                                                                tmp_path):
    _fake_datasets(monkeypatch, _rows(ML, n=2))
    monkeypatch.setattr(C, "make_swebench_scorer", lambda *a, **k: (lambda traces: None))
    with pytest.raises(RuntimeError, match="2 rows, not the 300"):
        C.execute_swebench_agentic("swe_bench_multilingual", "m", "http://x/v1", 1, ML._DATASET,
                                   "test", "swebench", dataset_revision=ML._REVISION,
                                   snapshot_columns=ML.SNAPSHOT_COLUMNS, snapshot_rows=300)
    _fake_datasets(monkeypatch, _rows(ML))
    seen = {}
    monkeypatch.setattr(A, "run_agentic", lambda *a, **k: seen.update(tasks=a[4], kw=k) or {})
    C.execute_swebench_agentic("swe_bench_multilingual", "m", "http://x/v1", 1, ML._DATASET,
                               "test", "swebench", dataset_revision=ML._REVISION,
                               snapshot_columns=ML.SNAPSHOT_COLUMNS, snapshot_rows=300)
    assert len(seen["tasks"]) == 300
    assert "snapshot_columns" not in seen["kw"] and "snapshot_rows" not in seen["kw"]


def test_live_snapshot_columns_are_what_the_fork_test_spec_reads():
    # the fork's make_test_spec reads these keys (instance[...]) or depends on them for the
    # SWE-bench-Live rows (test_cmds, log_parser); the pinned split counts are the Hub's
    assert {"instance_id", "repo", "base_commit", "test_patch", "FAIL_TO_PASS",
            "PASS_TO_PASS", "test_cmds", "log_parser"} <= set(LV.SNAPSHOT_COLUMNS)
    assert LV.SPLIT_ROWS == {"lite": 300, "full": 1319, "verified": 500}
    assert CB.SPLIT_ROWS == {"test": 500} and ML.SPLIT_ROWS == {"test": 300}
    assert CB.SNAPSHOT_COLUMNS == ML.SNAPSHOT_COLUMNS == C.SWEBENCH_V5_COLUMNS


# ============================================================================================
# swebench 5.x failure classification stays advisory
# ============================================================================================
def test_an_advisory_environment_flag_on_a_completed_instance_changes_nothing(tmp_path):
    """The reported case: one completed, unresolved instance that v5 also lists in
    infra_failure_ids (its test output printed 'Killed')."""
    report = {"total_instances": 1, "submitted_instances": 1, "completed_instances": 1,
              "resolved_instances": 0, "unresolved_instances": 1, "error_instances": 0,
              "empty_patch_instances": 0, "error_ids": [], "infra_failure_ids": ["a__a-1"],
              "ambiguous_failure_ids": [], "failure_reasons": {"a__a-1": "out_of_memory"}}
    res = {"status": "success"}
    reasons = C.swebench_report_reasons("copilot_bench_swe", res,
                                        {"swebench_report": report,
                                         "swebench_workdir": str(tmp_path)})
    assert reasons == [] and res == {"status": "success"}
    assert report["harness_error_instances"] == 0
    assert report["infra_failure_ids"] == ["a__a-1"]          # still recorded as reported


def test_ambiguous_flags_on_many_instances_do_not_make_a_run_partial(tmp_path):
    ids = [f"i{n}" for n in range(500)]
    report = {"total_instances": 500, "completed_instances": 500, "empty_patch_instances": 0,
              "error_ids": [], "infra_failure_ids": ids[:3], "ambiguous_failure_ids": ids[3:40]}
    patch_fail, harness_err, scoreable = C._split_report(report, str(tmp_path))
    assert (patch_fail, harness_err, scoreable) == (set(), set(), 500)


def test_oneshot_result_ignores_the_advisory_ids(monkeypatch, tmp_path):
    report = {"total_instances": 2, "completed_instances": 2, "resolved_instances": 1,
              "empty_patch_instances": 0, "error_ids": [], "infra_failure_ids": ["b__b-2"]}

    def _scorer(*a, **k):
        metrics = a[6]

        async def _score(traces):
            metrics["swebench_report"] = dict(report)
            metrics["swebench_workdir"] = str(tmp_path)
        return _score
    _fake_datasets(monkeypatch, _rows(CB))
    monkeypatch.setattr(C, "make_swebench_scorer", _scorer)

    def _suite(**k):
        asyncio.run(k["async_eval_fn"]([]))
        return {"status": "success", "temperature": 0.0}
    monkeypatch.setattr(C, "run_eval_suite", _suite)
    res = C.execute_swebench("copilot_bench_swe", "m", "http://x/v1", 1, CB._DATASET, "test",
                             "swebench", dataset_revision=CB._REVISION,
                             snapshot_columns=CB.SNAPSHOT_COLUMNS, snapshot_rows=500)
    assert "partial_evaluation" not in res and res["status"] == "success"
    assert "harness errors" not in res["leaderboard_comparable_reason"]


# ============================================================================================
# report retry: the harness's report step alone, nothing graded again
# ============================================================================================
@pytest.mark.parametrize("image", [None, "gbench-swebench-v5:5.0.2"])
def test_report_retry_runs_only_the_report_step(monkeypatch, tmp_path, image):
    runs = []

    def _run(cmd, cwd=None, **kw):
        runs.append(list(cmd))
        run_id = cmd[cmd.index("--run_id") + 1]
        os.makedirs(os.path.join(cwd, "logs", "run_evaluation", run_id), exist_ok=True)
        if len(runs) == 3:
            with open(os.path.join(cwd, f"gbench__m.{run_id}.json"), "w") as f:
                json.dump({"total_instances": 1, "completed_instances": 1,
                           "resolved_ids": ["a__a-1"], "error_ids": []}, f)
        return _proc(1, "", "docker.errors.NotFound: No such container")
    monkeypatch.setattr(C.subprocess, "run", _run)
    monkeypatch.setattr(C, "_REPORT_RETRY_BACKOFF_S", (0, 0, 0, 0, 0))
    monkeypatch.setenv("GBENCH_SWE_OMP_THREADS", "0")
    metrics = {}
    tr = [{"extra_payload": {"instance_id": "a__a-1"}, "agent_patch": "d\n"}]
    asyncio.run(C.make_swebench_scorer("x", "m", "d.jsonl", "test", None, 1, metrics,
                                       harness_image=image, harness_v5=bool(image))(tr))
    assert len(runs) == 3 and metrics["harness_report_retries"] == 2 and tr[0]["is_correct"]
    first = runs[0]
    entry = first.index("swebench.harness.run_evaluation")
    assert first[entry - 1] == "-m" and "--rewrite_reports" not in " ".join(map(str, runs))
    for retry in runs[1:]:
        # same argv, with `-m swebench.harness.run_evaluation` replaced by the report step
        assert retry[:entry - 1] == first[:entry - 1]
        assert retry[entry - 1:entry + 1] == ["-c", C._REPORT_STEP]
        assert retry[entry + 1:] == first[entry + 1:]


def test_no_retry_when_the_harness_never_ran(monkeypatch, tmp_path):
    runs = []
    monkeypatch.setattr(C.subprocess, "run", lambda cmd, **kw: runs.append(cmd) or _proc(1))
    metrics = {}
    tr = [{"extra_payload": {"instance_id": "a__a-1"}, "agent_patch": "d\n"}]
    asyncio.run(C.make_swebench_scorer("x", "m", "d.jsonl", "test", None, 1, metrics,
                                       harness_image="img", harness_v5=True)(tr))
    assert len(runs) == 1 and metrics["swebench_report"] == {"error": "harness report not produced"}


# ============================================================================================
# SWE-bench Pro V2: re-grade verdicts
# ============================================================================================
def _trial(job_dir, iid, *, reward=None, exc=None, agent=True, verifier=True, exit_status=None,
           patch="diff --git a/x b/x\n", traj=True):
    tdir = os.path.join(job_dir, f"{iid}__abc")
    os.makedirs(os.path.join(tdir, "agent"), exist_ok=True)
    res = {"task_name": f"swebench-pro/{iid}"}
    if reward is not None:
        res["verifier_result"] = {"rewards": {"reward": reward}}
    if exc:
        res["exception_info"] = {"exception_type": exc, "exception_message": "boom"}
    if agent:
        res["agent_execution"] = {"started_at": "t0"}
    if verifier:
        res["verifier"] = {"started_at": "t1"}
    with open(os.path.join(tdir, "result.json"), "w") as f:
        json.dump(res, f)
    if traj:
        with open(os.path.join(tdir, "agent", "mini-swe-agent.trajectory.json"), "w") as f:
            json.dump({"info": {"exit_status": exit_status, "model_stats": {"api_calls": 3}}}, f)
    if patch is not None:
        with open(os.path.join(tdir, "agent", "model.patch"), "w") as f:
            f.write(patch)


@pytest.mark.parametrize("kw,verdict", [
    (dict(reward=1.0), (1.0, None)),
    (dict(reward=0.0), (0.0, None)),
    # a reward is the verdict whatever else the trial recorded (Harbor's metrics read it)
    (dict(reward=1.0, exc="RuntimeError"), (1.0, None)),
    # the verifier ran on the patched tree and timed out / left no usable reward: a 0
    (dict(exc="VerifierTimeoutError"), (0.0, None)),
    (dict(exc="RewardFileNotFoundError"), (0.0, None)),
    (dict(exc="RewardFileEmptyError"), (0.0, None)),
    (dict(exc="VerifierOutputParseError"), (0.0, None)),
    # outside the patch's control: infrastructure
    (dict(exc="EnvironmentStartTimeoutError", agent=False, verifier=False), (None, "Environment")),
    (dict(exc="ConflictError", agent=False, verifier=False), (None, "ConflictError")),
    (dict(exc="AddTestsDirError"), (None, "AddTestsDirError")),
    (dict(exc="DownloadVerifierDirError"), (None, "DownloadVerifierDirError")),
    (dict(exc="VerifierTimeoutError", verifier=False), (None, "VerifierTimeoutError")),
    (dict(), (None, "no reward")),
])
def test_regrade_verdicts_follow_harbor(tmp_path, kw, verdict):
    _trial(str(tmp_path), "instance_a", traj=False, patch=None, **kw)
    reward, why = P.regrade_verdict(P.job_trials(str(tmp_path))["instance_a"])
    assert reward == verdict[0]
    assert (why is None) == (verdict[1] is None)
    if verdict[1]:
        assert verdict[1] in why
    assert P.regrade_verdict(None) == (None, "no re-grade trial was written")


def test_the_patch_outcome_set_is_harbors_non_retried_verifier_set():
    # Harbor 0.22.0 models/job/config.py RetryConfig.exclude_exceptions (verifier members)
    assert set(P._VERIFIER_OUTCOME_EXCEPTIONS) == {
        "VerifierTimeoutError", "RewardFileNotFoundError", "RewardFileEmptyError",
        "VerifierOutputParseError"}


def _pro_setup(monkeypatch, tmp_path, ids, agent_trials, regrade):
    rows = [{"instance_id": i, "problem_statement": f"P {i}", "repo": "o/r",
             "repo_language": "go", "docker_image": f"ghcr.io/scaleapi/swe-bench_pro-v2:{i}"}
            for i in ids]
    monkeypatch.setattr(P, "load_v2_rows", lambda: rows)
    monkeypatch.setattr(P, "load_hard51_ids", lambda v2=None: set())
    monkeypatch.setattr(P, "read_locked_agent_config",
                        lambda: ("model:\n  model_kwargs:\n    reasoning_effort: high\n",
                                 P.AGENT_CONFIG_SHA256))
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_TRAJ_DIR", str(tmp_path / "runs"))
    jobs = []

    def _harbor(workdir, args, job):
        jobs.append((job, list(args)))
        job_dir = os.path.join(args[args.index("--jobs-dir") + 1], job)
        os.makedirs(job_dir, exist_ok=True)
        include = [a for a in args if str(a).startswith("instance_")]
        if job.startswith("agent"):
            for iid, kw in agent_trials.items():
                _trial(job_dir, iid, **kw)
        else:
            for iid, kw in regrade(job).items():
                if not include or iid in include:
                    _trial(job_dir, iid, traj=False, patch=None, **kw)
        return 0, "harbor done"
    monkeypatch.setattr(P, "_run_harbor", _harbor)
    return jobs


def test_a_regrade_verifier_timeout_scores_zero_without_retry_or_partial(monkeypatch, tmp_path):
    """The reported case: the patch hangs its tests, the re-grade verifier times out."""
    jobs = _pro_setup(
        monkeypatch, tmp_path, ["instance_a", "instance_b"],
        agent_trials={"instance_a": dict(reward=1.0, exit_status="Submitted"),
                      "instance_b": dict(reward=0.0, exit_status="Submitted")},
        regrade=lambda job: {"instance_a": dict(reward=1.0),
                             "instance_b": dict(exc="VerifierTimeoutError")})
    res = P.run_swe_bench_pro_v2("m", "http://h:1/v1", 1, enable_thinking=True)
    assert [j for j, _ in jobs] == ["agent", "regrade"]
    assert res["status"] == "success" and "partial_evaluation" not in res
    assert (res["total_questions"], res["correct_answers"], res["failed_requests"]) == (2, 1, 0)
    # the only deviation is the --thinking rule's temperature 1.0 (the locked config sends none)
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"] == (
        "temperature=1.0 sent by the --thinking rule (every thinking run samples the model under "
        "test at 1.0); the locked config sends none")
    assert res["regrade_verifier_exception_ids"] == {"instance_b": "VerifierTimeoutError"}
    tr = {t["extra_payload"]["instance_id"]: t for t in res["sample_traces"]}
    assert tr["instance_b"]["status"] == "OK" and not tr["instance_b"]["agent_infra_error"]
    assert tr["instance_b"]["regrade_exception"] == "VerifierTimeoutError"
    assert tr["instance_b"]["is_correct"] is False and tr["instance_b"]["agent_error"] is None


def test_a_single_task_whose_verifier_times_out_is_a_zero_not_an_error(monkeypatch, tmp_path):
    _pro_setup(monkeypatch, tmp_path, ["instance_a"],
               agent_trials={"instance_a": dict(reward=0.0, exit_status="Submitted")},
               regrade=lambda job: {"instance_a": dict(exc="RewardFileNotFoundError")})
    res = P.run_swe_bench_pro_v2("m", "http://h:1/v1", 1, enable_thinking=True)
    assert res["status"] == "success" and res["accuracy"] == 0.0 and "error" not in res


def test_a_regrade_infrastructure_failure_is_retried_then_partial(monkeypatch, tmp_path):
    jobs = _pro_setup(
        monkeypatch, tmp_path, ["instance_a", "instance_b"],
        agent_trials={"instance_a": dict(reward=1.0, exit_status="Submitted"),
                      "instance_b": dict(reward=1.0, exit_status="Submitted")},
        regrade=lambda job: {"instance_a": dict(reward=1.0),
                             "instance_b": dict(exc="EnvironmentStartTimeoutError",
                                                agent=False, verifier=False)})
    res = P.run_swe_bench_pro_v2("m", "http://h:1/v1", 1, enable_thinking=True)
    assert [j for j, _ in jobs] == ["agent", "regrade", "regrade_retry1"]
    assert [a for a in jobs[2][1] if a.startswith("instance_")] == ["instance_b"]
    assert res["status"] == "completed_with_errors"
    assert res["partial_evaluation"]["infra_error_ids"] == ["instance_b"]
    assert "EnvironmentStartTimeoutError" in {t["extra_payload"]["instance_id"]: t for t in
                                              res["sample_traces"]}["instance_b"]["agent_error"]
    assert "regrade_verifier_exception_ids" not in res


def test_oneshot_regrade_uses_the_same_verdicts_and_retry(monkeypatch, tmp_path):
    jobs = []

    def _harbor(workdir, args, job):
        jobs.append((job, [a for a in args if str(a).startswith("instance_")]))
        job_dir = os.path.join(args[args.index("--jobs-dir") + 1], job)
        os.makedirs(job_dir, exist_ok=True)
        _trial(job_dir, "instance_a", reward=1.0, traj=False, patch=None)
        _trial(job_dir, "instance_b", exc="VerifierTimeoutError", traj=False, patch=None)
        _trial(job_dir, "instance_c", exc="EnvironmentStartTimeoutError", agent=False,
               verifier=False, traj=False, patch=None)
        return 0, "tail"
    monkeypatch.setattr(P, "_run_harbor", _harbor)
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_TRAJ_DIR", str(tmp_path / "runs"))
    metrics = {}
    traces = [{"extra_payload": {"instance_id": i}, "response_text": "```diff\ndiff --git a/x "
               "b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n```"}
              for i in ("instance_a", "instance_b", "instance_c")]
    asyncio.run(P._make_v2_oneshot_scorer(1, metrics)(traces))
    assert [j for j, _ in jobs] == ["regrade", "regrade_retry1"]
    assert jobs[1][1] == ["instance_c"]
    rep = metrics["swe_bench_pro_report"]
    assert rep["not_regraded_ids"] == ["instance_c"] and rep["regraded_instances"] == 2
    assert [t["is_correct"] for t in traces] == [True, False, False]
    assert "swe_bench_pro_error" not in metrics


# ============================================================================================
# no cost opt-in: a provisioned machine runs swe_bench_pro
# ============================================================================================
def _provisioned(monkeypatch):
    labels = {"org.gbench.harbor": P.HARBOR_VERSION,
              "org.gbench.swe_bench_pro_os": P.HARNESS_COMMIT,
              "org.gbench.egress_sidecar_read_timeout": P.EGRESS_SIDECAR_READ_TIMEOUT}
    monkeypatch.setattr(P.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(P.os.path, "exists", lambda p: True)
    monkeypatch.setattr(P.subprocess, "run", lambda cmd, **kw: _proc(0, json.dumps(labels)))


def test_v2_prerequisites_pass_without_any_opt_in(monkeypatch):
    _provisioned(monkeypatch)
    assert P.check_swe_bench_pro_v2_prerequisites() == (True, "")


def test_a_provisioned_machine_reaches_the_v2_run_without_an_opt_in(monkeypatch):
    _provisioned(monkeypatch)
    seen = {}
    monkeypatch.setattr(P, "run_swe_bench_pro_v2", lambda *a, **k: seen.update(a=a) or {"ok": 1})
    assert P.run_swe_bench_pro("m", "http://x/v1", concurrency=2) == {"ok": 1}
    assert seen["a"][:3] == ("m", "http://x/v1", 2)


def test_missing_image_is_still_a_hard_error(monkeypatch):
    monkeypatch.setattr(P.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(P.os.path, "exists", lambda p: True)
    monkeypatch.setattr(P.subprocess, "run",
                        lambda cmd, **kw: _proc(1 if "inspect" in cmd else 0))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*is not built"):
        P.run_swe_bench_pro("m", "http://x/v1", concurrency=1)


def test_the_opt_in_variable_is_gone_from_code_and_doc():
    src = open(os.path.join(REPO, "gbench", "runners", "eval_suites", "swe_bench_pro.py")).read()
    doc = open(os.path.join(REPO, "docs", "evals", "swe_bench_pro.md")).read()
    assert "SWE_BENCH_PRO_RUN" not in src and "SWE_BENCH_PRO_RUN" not in doc
    assert "No opt-in variable is needed" in doc


# ============================================================================================
# docs name the submission the comparability refers to
# ============================================================================================
@pytest.mark.parametrize("name,folder,commit", [
    ("copilot_bench_swe", "evaluation/verified/20260901_mini-v2.4.2_gemini-3-5-flash", "69f10990"),
    ("swe_bench_multilingual", "evaluation/multilingual/20260902_mini-v2.4.6_gemini-3-5-flash",
     "8fa0575c"),
])
def test_docs_word_comparability_as_the_experiments_submission(name, folder, commit):
    doc = open(os.path.join(REPO, "docs", "evals", f"{name}.md")).read()
    flat = " ".join(doc.split())
    assert folder in doc and commit in doc and "no 2.4.x row" in flat
    assert "leaderboard rows run with mini-swe-agent 2.4" not in flat
    assert "infra_failure_ids" in doc and "advisory" in doc
    assert "make_run_report" in doc and "--rewrite_reports" not in doc


# ============================================================================================
# v5 harness image: the docker CLI its container cleanup needs
# ============================================================================================
def test_v5_prereq_probe_requires_the_docker_cli(monkeypatch, tmp_path):
    snap = tmp_path / "test.jsonl"
    _write(snap, _rows(CB, n=1))
    calls = []

    def _run(cmd, **kw):
        calls.append(list(cmd))
        if cmd[:3] == ["docker", "image", "inspect"]:
            return _proc(0, json.dumps({"org.gbench.swebench": C.SWEBENCH_V5_VERSION}))
        if cmd[:2] == ["docker", "run"]:
            return _proc(1, "", "this image has no docker CLI, which swebench 5.x needs to stop "
                                "and remove task containers; rebuild it")
        return _proc(0)
    monkeypatch.setattr(C.subprocess, "run", _run)
    ok, why = C.check_swebench_v5_prereq(str(snap), "test")
    probe = calls[-1][calls[-1].index("-c") + 1]
    assert "shutil.which('docker')" in probe and "make_test_spec" in probe
    assert not ok and "no docker CLI" in why


def test_v5_dockerfile_installs_the_docker_cli():
    text = open(os.path.join(REPO, "gbench", "docker", "swebench_v5.Dockerfile")).read()
    assert "docker-ce-cli" in text and "docker --version" in text


def test_unstopped_task_containers_are_recorded(monkeypatch, tmp_path):
    def _run(cmd, cwd=None, **kw):
        run_id = cmd[cmd.index("--run_id") + 1]
        with open(os.path.join(cwd, f"gbench__m.{run_id}.json"), "w") as f:
            json.dump({"total_instances": 1, "completed_instances": 1, "resolved_ids": [],
                       "error_ids": [], "unstopped_instances": 1}, f)
        return _proc()
    monkeypatch.setattr(C.subprocess, "run", _run)
    metrics = {}
    tr = [{"extra_payload": {"instance_id": "a__a-1"}, "agent_patch": "d\n"}]
    asyncio.run(C.make_swebench_scorer("x", "m", "d.jsonl", "test", None, 1, metrics,
                                       harness_image="img", harness_v5=True)(tr))
    assert metrics["swebench_report"]["unstopped_instances"] == 1
