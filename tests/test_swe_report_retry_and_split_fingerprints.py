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
"""SWE family: report-step retry, pinned split fingerprints, repeated instance ids, no per-turn
max_tokens on the mini-swe-agent paths, and the v5 image pins.

* When the harness scored instances but its final report step crashed, only that step
  (swebench's own make_run_report) runs again, up to 5 times with a short backoff. An instance
  the scoring pass left without report.json (a test timeout) stays in error_ids; it is never
  re-graded from its truncated test output.
* A pinned snapshot is checked against the split's content fingerprint, so a stand-in file with
  the right row count and columns is refused, and SWE-bench-Live's published splits with a
  repeated instance id are accepted as published. Each instance id gets one rollout / prompt.
* The swe_bench_live prerequisite check fetches and checks the pinned snapshot.
* run_agentic sends no per-turn max_tokens (the bundled mini-swe-agent configs send none) and
  records the --max-output-tokens value as not applied.

Hermetic: no container, no endpoint, no network, no docker daemon; every cache is under
tmp_path.
"""

import asyncio
import datetime
import glob
import json
import os
import re
import sys
import types

import pytest

from gbench.runners.eval_suites import copilot_bench_swe as CB
from gbench.runners.eval_suites import swe_agentic as A
from gbench.runners.eval_suites import swe_bench_live as LV
from gbench.runners.eval_suites import swe_bench_multilingual as ML
from gbench.runners.eval_suites import swebench_common as C


@pytest.fixture(autouse=True)
def _upstream_max_tokens(monkeypatch):
    """These tests check the scaffolds' upstream protocol, which sends no max_tokens: that is
    the GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0 path since the run's --max-output-tokens became the
    default for agent suites (2026-10-11; tests/test_agent_max_tokens_from_run.py)."""
    monkeypatch.setenv("GBENCH_AGENT_MAX_TOKENS_FROM_RUN", "0")



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
    for var in ("GBENCH_SWEBENCH_V5_IMAGE", "GBENCH_SWE_OMP_THREADS", "GBENCH_SHARD",
                "GBENCH_SWE_BENCH_LIVE_MODE", "GBENCH_SWE_BENCH_LIVE_IMAGE",
                "GBENCH_COPILOT_BENCH_SWE_MODE", "GBENCH_SWE_BENCH_MULTILINGUAL_MODE",
                "GBENCH_COPILOT_BENCH_SWE_TEMPERATURE", "GBENCH_SWE_BENCH_LIVE_TEMPERATURE",
                "GBENCH_SWE_BENCH_MULTILINGUAL_TEMPERATURE", "GBENCH_REPETITION_PENALTY",
                "GBENCH_FREQUENCY_PENALTY", "GBENCH_PRESENCE_PENALTY"):
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


def _rows(mod, n, **extra):
    return [{**{c: "x" for c in mod.SNAPSHOT_COLUMNS}, "repo": "o/r",
             "instance_id": f"o__r-{i}", "problem_statement": f"issue {i}", **extra}
            for i in range(n)]


def _snapshot_path(tmp_path, dataset, revision, split):
    safe = dataset.replace("/", "__")
    return tmp_path / "cache" / "swebench_snapshots" / f"{safe}@{revision}" / f"{split}.jsonl"


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# ============================================================================================
# report retry: make_run_report alone, never a re-grade
# ============================================================================================
class _FakeDockerClient:
    """What make_run_report asks a docker client for, without a daemon."""

    class _Images:
        def list(self, *a, **k):
            return []

    class _Containers:
        def list(self, *a, **k):
            return []

    def __init__(self):
        self.images = self._Images()
        self.containers = self._Containers()


def _harness_workdir(workdir, run_id, tag, *, timed_out="b__b-2"):
    """What swebench leaves after scoring a__a-1 (resolved) and `timed_out` (its tests hit the
    timeout: test_output.txt ends with the harness's timeout line, no report.json; 4.x/5.x
    run_instance writes the output, then raises EvaluationError)."""
    base = os.path.join(workdir, "logs", "run_evaluation", run_id, tag)
    a = os.path.join(base, "a__a-1")
    os.makedirs(a)
    with open(os.path.join(a, "test_output.txt"), "w") as f:
        f.write("PASSED tests/test_a.py::test_a\n")
    with open(os.path.join(a, "report.json"), "w") as f:
        json.dump({"a__a-1": {"resolved": True}}, f)
    b = os.path.join(base, timed_out)
    os.makedirs(b)
    with open(os.path.join(b, "test_output.txt"), "w") as f:
        # everything the grader needs to call it resolved, then the timeout line
        f.write("PASSED tests/test_b.py::test_b\n\nTimeout error: 1800 seconds exceeded.")
    with open(os.path.join(b, "run_instance.log"), "w") as f:
        # what run_instance logs for it: the patch applied, the tests ran past the timeout
        f.write("2026-09-27 10:00:00,000 - INFO - >>>>> Applied Patch:\n"
                "Checking patch x.py...\nApplied patch x.py cleanly.\n"
                "2026-09-27 10:30:00,000 - INFO - Test runtime: 1_800.00 seconds\n"
                f"2026-09-27 10:30:00,001 - INFO - Test output for {timed_out} written to "
                "test_output.txt\n"
                "2026-09-27 10:30:00,002 - INFO - Traceback (most recent call last):\n"
                f"swebench.harness.utils.EvaluationError: {timed_out}: Test timed out after "
                "1800 seconds.\n")


def _exec_report_step(monkeypatch, argv, cwd):
    """Run swebench_common._REPORT_STEP in-process against the installed swebench with a fake
    docker client (what the harness container runs, minus the daemon)."""
    import docker
    from swebench.harness import reporting
    monkeypatch.setattr(docker, "from_env", lambda *a, **k: _FakeDockerClient())
    # 4.1's make_run_report builds TestSpecs of the rows to name unremoved images; the rows
    # here are stand-ins, so name the image directly
    monkeypatch.setattr(reporting, "make_test_spec",
                        lambda row, **k: types.SimpleNamespace(
                            instance_image_key=f"img:{row['instance_id']}"), raising=False)
    old = os.getcwd()
    monkeypatch.setattr(sys, "argv", ["-c", *argv])
    try:
        os.chdir(cwd)
        exec(compile(C._REPORT_STEP, "<report step>", "exec"), {"__name__": "__main__"})
    finally:
        os.chdir(old)


def _dataset_file(tmp_path):
    path = tmp_path / "rows.jsonl"
    _write(path, [{"instance_id": i, "problem_statement": "p", "repo": "o/r"}
                  for i in ("a__a-1", "b__b-2")])
    return str(path)


def test_the_report_step_keeps_an_instance_without_report_json_an_error(monkeypatch, tmp_path):
    """The reported case, on the real make_run_report: an instance whose tests timed out has
    test output but no report.json. `--rewrite_reports true` graded it from that output (a
    resolved verdict); the report step alone keeps it in error_ids, as a run whose report step
    did not crash does."""
    dataset = _dataset_file(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    preds = work / "preds.jsonl"
    _write(preds, [{"instance_id": i, "model_name_or_path": "gbench__m", "model_patch": "d\n"}
                   for i in ("a__a-1", "b__b-2")])
    _harness_workdir(str(work), "r1", "gbench__m")
    _exec_report_step(monkeypatch, [
        "--dataset_name", dataset, "--split", "test", "--predictions_path", str(preds),
        "--run_id", "r1", "--max_workers", "2", "--cache_level", "env",
        "--instance_ids", "a__a-1", "b__b-2", "--namespace", "swebench"], str(work))
    rep = json.load(open(work / "gbench__m.r1.json"))
    assert rep["resolved_ids"] == ["a__a-1"] and rep["error_ids"] == ["b__b-2"]
    assert rep["completed_instances"] == 1 and rep["total_instances"] == 2
    # nothing was graded: the timed-out instance still has no report.json
    assert not glob.glob(str(work / "logs" / "run_evaluation" / "r1" / "*" / "b__b-2" /
                             "report.json"))


def test_scorer_retry_after_a_report_crash_matches_an_uncrashed_run(monkeypatch, tmp_path):
    """End to end through the scorer: the scoring pass leaves a timed-out instance and no run
    report; the retry (the report step, run here in-process in place of the harness container)
    keeps it in error_ids, not a clean resolve: its log shows the tests over the applied patch
    running past the timeout, the model's outcome (unresolved, scored)."""
    runs = []

    def _run(cmd, cwd=None, **kw):
        runs.append(list(cmd))
        run_id = cmd[cmd.index("--run_id") + 1]
        if C._REPORT_STEP not in cmd:                      # the scoring pass
            tag = json.loads(open(cmd[cmd.index("--predictions_path") + 1]).readline())[
                "model_name_or_path"]
            _harness_workdir(cwd, run_id, tag)
            return _proc(1, "", "docker.errors.NotFound: 404 Client Error: No such container")
        at = cmd.index("-c")
        assert cmd[at + 1] == C._REPORT_STEP
        _exec_report_step(monkeypatch, cmd[at + 2:], cwd)
        return _proc(0)
    monkeypatch.setattr(C.subprocess, "run", _run)
    metrics = {}
    traces = [{"extra_payload": {"instance_id": i}, "agent_patch": "d\n"}
              for i in ("a__a-1", "b__b-2")]
    asyncio.run(C.make_swebench_scorer("copilot_bench_swe", "m", _dataset_file(tmp_path), "test",
                                       None, 2, metrics, harness_image="gbench-swebench-v5:5.0.2",
                                       harness_v5=True)(traces))
    assert len(runs) == 2 and metrics["harness_report_retries"] == 1
    assert "--rewrite_reports" not in " ".join(map(str, runs[1]))
    assert [t["is_correct"] for t in traces] == [True, False]
    assert metrics["swebench_report"]["error_ids"] == ["b__b-2"]
    assert traces[1].get("test_timed_out") is True and not traces[1].get("scoring_excluded")
    res = {"status": "success"}
    reasons = C.swebench_report_reasons("copilot_bench_swe", res, metrics)
    assert res["status"] == "success" and "partial_evaluation" not in res and not reasons
    assert metrics["swebench_report"]["test_timeout_instances"] == 1
    assert metrics["swebench_report"]["test_timeout_ids"] == ["b__b-2"]
    assert metrics["swebench_report"]["harness_error_instances"] == 0


def test_report_step_retries_up_to_five_times_with_backoff(monkeypatch, tmp_path):
    runs, waits = [], []

    def _run(cmd, cwd=None, **kw):
        runs.append(list(cmd))
        run_id = cmd[cmd.index("--run_id") + 1]
        os.makedirs(os.path.join(cwd, "logs", "run_evaluation", run_id), exist_ok=True)
        return _proc(1, "", "docker.errors.NotFound: No such container")

    async def _sleep(delay):
        waits.append(delay)
    monkeypatch.setattr(C.subprocess, "run", _run)
    monkeypatch.setattr(asyncio, "sleep", _sleep)
    metrics = {}
    tr = [{"extra_payload": {"instance_id": "a__a-1"}, "agent_patch": "d\n"}]
    asyncio.run(C.make_swebench_scorer("x", "m", "d.jsonl", "test", None, 1, metrics,
                                       harness_image="img", harness_v5=True)(tr))
    assert C._REPORT_RETRY_BACKOFF_S == (0, 5, 15, 30, 60)
    assert metrics["harness_report_retries"] == 5 and len(runs) == 6
    assert waits == [5, 15, 30, 60]                     # the first re-run is immediate
    assert C._REPORT_STEP not in runs[0] and all(C._REPORT_STEP in r for r in runs[1:])
    assert metrics["swebench_report"] == {"error": "harness report not produced"}
    res = {"status": "success"}
    assert C.swebench_report_reasons("x", res, metrics) and res["status"] == "error"


def test_report_step_parses_the_scoring_pass_arguments():
    """The program reads the scoring pass's own flags and passes `namespace` only to a release
    whose make_run_report takes it."""
    src = C._REPORT_STEP
    for flag in ("--dataset_name", "--split", "--predictions_path", "--run_id", "--namespace",
                 "--instance_ids"):
        assert flag in src
    assert "parse_known_args" in src and "make_run_report(preds, full, a.run_id, client" in src
    assert "load_swebench_dataset(a.dataset_name, a.split, a.instance_ids)" in src
    assert "rewrite_reports" not in src


# ============================================================================================
# pinned split fingerprints and repeated instance ids
# ============================================================================================
def test_every_pinned_split_has_a_content_fingerprint():
    for mod in (CB, ML, LV):
        assert set(mod.SPLIT_SHA256) == set(mod.SPLIT_ROWS)
        assert all(re.fullmatch(r"[0-9a-f]{64}", v) for v in mod.SPLIT_SHA256.values())
    # measured on the rows `datasets` loads at each pinned commit (and on the snapshot files
    # written from them)
    assert CB.SPLIT_SHA256["test"].startswith("1939f0b7e4fca575")
    assert ML.SPLIT_SHA256["test"].startswith("83441ebb91ffa12c")
    assert LV.SPLIT_SHA256 == {
        "lite": "b9252aeb085eb949f01b8464bee76240d474146c9a4f6aed1bbb693c2b94755e",
        "full": "b231d1932ce614df07b6b4237db2a6f4eb6020213d0d9e919d95a62b976cf95e",
        "verified": "d3ae023598f62e0c24a025a5091ec6ef97224b62dd8aeb6224d7b14154e55974"}


def test_fingerprint_is_the_same_for_loaded_rows_and_the_file_written_from_them(tmp_path):
    """A timestamp column (SWE-bench-Live's created_at) is a datetime when `datasets` loads it
    and a string in the snapshot file; both give one fingerprint, so a cached file is reused."""
    rows = [{"instance_id": "a__a-1", "problem_statement": "p\u00e9", "FAIL_TO_PASS": ["t1"],
             "created_at": datetime.datetime(2025, 4, 1, 12, 30, tzinfo=datetime.timezone.utc),
             "n": 1.5, "hints": None}]
    path = tmp_path / "s.jsonl"
    C._write_snapshot(str(path), rows, "x")
    assert C.snapshot_sha256(C._read_jsonl(str(path))) == C.snapshot_sha256(rows)
    assert C.snapshot_sha256(rows) != C.snapshot_sha256([dict(rows[0], n=2.5)])
    # row order is part of the content
    two = rows + [dict(rows[0], instance_id="b__b-2")]
    assert C.snapshot_sha256(two) != C.snapshot_sha256(two[::-1])


def test_a_stand_in_split_with_the_right_shape_is_refused(monkeypatch, tmp_path):
    """500 placeholder rows carrying every v5 column pass a structural check; the content
    fingerprint refuses them, cached or fetched."""
    stand_in = _rows(CB, 500)
    path = _snapshot_path(tmp_path, CB._DATASET, CB._REVISION, "test")
    _write(path, stand_in)
    _fake_datasets(monkeypatch, stand_in)
    with pytest.raises(RuntimeError, match="not the pinned dataset") as e:
        CB._load_copilot_bench_swe_dataset(limit=1)
    assert "content sha256" in str(e.value)
    assert len(glob.glob(f"{path}.invalid-*")) == 1 and not path.exists()
    ok, why = CB.check_copilot_bench_swe_prerequisites()
    assert not ok and "content sha256" in why


def test_the_published_split_with_a_repeated_id_is_accepted_as_published(monkeypatch, tmp_path):
    """SWE-bench-Live@b51a8642 `verified` / `full` carry conan-io__conan-18153 twice (rows that
    differ in test_cmds and all_hints_text). The snapshot keeps both rows, fetched and reused."""
    rows = _rows(LV, 6)
    dup = {**rows[4], "instance_id": "conan-io__conan-18153", "test_cmds": "['pytest -rA']",
           "all_hints_text": "short"}
    rows[4] = dup
    rows[5] = {**dup, "test_cmds": "['python -m pytest -rA .']", "all_hints_text": "long"}
    sha = C.snapshot_sha256(rows)
    _fake_datasets(monkeypatch, rows)
    path, got = C._pinned_snapshot_rows(LV._DATASET, "verified", LV._REVISION, "swe_bench_live",
                                        LV.SNAPSHOT_COLUMNS, 6, sha)
    assert got == rows and len(C._read_jsonl(path)) == 6
    _fake_datasets(monkeypatch, OSError("offline"))            # reused, not refetched
    assert C.pinned_swe_snapshot(LV._DATASET, "verified", LV._REVISION, "x",
                                 columns=LV.SNAPSHOT_COLUMNS, rows=6, sha256=sha) == path
    assert not glob.glob(f"{path}.invalid-*")
    assert C.repeated_instance_ids(rows) == {"conan-io__conan-18153": 2}
    assert [r["instance_id"] for r in C.unique_instance_rows(rows)] == \
        [r["instance_id"] for r in rows[:5]]
    assert C.unique_instance_rows(rows)[4]["test_cmds"] == "['pytest -rA']"   # first row kept


def _live_dup_rows():
    rows = _rows(LV, 5)
    rows[3] = {**rows[3], "instance_id": "conan-io__conan-18153", "test_cmds": "a"}
    rows[4] = {**rows[3], "test_cmds": "b"}
    return rows


def test_agentic_run_rolls_out_each_instance_id_once_and_records_the_repeat(monkeypatch,
                                                                           tmp_path):
    rows = _live_dup_rows()
    _fake_datasets(monkeypatch, rows)
    seen = {}

    def _run_agentic(eval_name, model_name, base_url, concurrency, tasks, scorer, metrics, **k):
        seen.update(tasks=tasks, metrics=dict(metrics), scorer=scorer)
        return {}
    monkeypatch.setattr(A, "run_agentic", _run_agentic)
    captured = {}
    monkeypatch.setattr(C, "make_swebench_scorer",
                        lambda *a, **k: captured.update(dataset=a[2]) or (lambda traces: None))
    C.execute_swebench_agentic("swe_bench_live", "m", "http://x/v1", 1, LV._DATASET, "verified",
                               "starryzhang", dataset_revision=LV._REVISION,
                               snapshot_columns=LV.SNAPSHOT_COLUMNS, snapshot_rows=5,
                               snapshot_sha256=C.snapshot_sha256(rows))
    ids = [t.instance_id for t in seen["tasks"]]
    assert ids == ["o__r-0", "o__r-1", "o__r-2", "conan-io__conan-18153"]
    assert seen["metrics"]["dataset_repeated_instance_ids"] == {"conan-io__conan-18153": 2}
    assert "each ran once" in seen["metrics"]["dataset_repeated_instance_ids_note"]
    # the harness still reads every published row
    assert len(C._read_jsonl(captured["dataset"])) == 5


def test_oneshot_prompts_each_instance_id_once_and_records_the_repeat(monkeypatch, tmp_path):
    rows = _live_dup_rows()
    _fake_datasets(monkeypatch, rows)
    seen = {}

    def _suite(**k):
        seen["samples"] = k["samples"]
        return {"status": "success", "temperature": 0.0, "total_questions": len(k["samples"])}
    monkeypatch.setattr(C, "run_eval_suite", _suite)
    monkeypatch.setattr(C, "make_swebench_scorer", lambda *a, **k: (lambda traces: None))
    res = C.execute_swebench("swe_bench_live", "m", "http://x/v1", 1, LV._DATASET, "verified",
                             "starryzhang", dataset_revision=LV._REVISION,
                             snapshot_columns=LV.SNAPSHOT_COLUMNS, snapshot_rows=5,
                             snapshot_sha256=C.snapshot_sha256(rows))
    ids = sorted(s[2]["instance_id"] for s in seen["samples"])
    assert ids == sorted(["o__r-0", "o__r-1", "o__r-2", "conan-io__conan-18153"])
    assert res["dataset_repeated_instance_ids"] == {"conan-io__conan-18153": 2}


def test_suites_hand_the_fingerprint_to_both_modes(monkeypatch):
    for mod, run, split in ((CB, "run_copilot_bench_swe", "test"),
                            (ML, "run_swe_bench_multilingual", "test"),
                            (LV, "run_swe_bench_live", "lite")):
        monkeypatch.setattr(mod, [n for n in vars(mod) if n.startswith("check_")
                                  and n.endswith("_prerequisites")][0], lambda *a: (True, ""))
        monkeypatch.setattr(A, "require_agent", lambda *a: None)
        monkeypatch.setattr(A, "require_agent_v2", lambda *a: None)
        # swe_bench_live runs each leaderboard setting with its own split's fingerprint (here
        # its Python tabs, the ones these two entry points run)
        if mod is LV:
            monkeypatch.setattr(LV, "SETTINGS", LV.PYTHON_SETTINGS)
            monkeypatch.setattr(LV, "check_sweagent_prerequisites", lambda: (True, ""))
            # the suite's own check by name (the first check_* name is the Windows one)
            monkeypatch.setattr(LV, "check_swe_bench_live_prerequisites", lambda *a: (True, ""))
        want = list(getattr(mod, "SETTINGS", (split,)))
        agentic_entry = ("execute_sweagent_python" if mod is LV   # SWE-agent-for-eval
                         else "execute_swebench_agentic")
        for mode, entry in (("agentic", agentic_entry),
                            ("oneshot", "execute_swebench")):
            calls = []
            monkeypatch.setattr(mod, entry, lambda *a, **k: calls.append(k) or {"status": "ok"})
            monkeypatch.setenv(A.mode_env_var(mod.__name__.rsplit(".", 1)[-1]), mode)
            getattr(mod, run)("m", "http://x/v1", 1)
            assert [k["snapshot_sha256"] for k in calls] == [mod.SPLIT_SHA256[s] for s in want]


# ============================================================================================
# swe_bench_live: the prerequisite check fetches and checks the snapshot
# ============================================================================================
def _docker_ok(monkeypatch):
    # the Python tabs, whose fork image and snapshots these tests are about
    monkeypatch.setattr(LV, "SETTINGS", LV.PYTHON_SETTINGS)
    monkeypatch.setattr(LV.shutil, "which", lambda _x: "/usr/bin/docker")
    labels = json.dumps({"org.gbench.swe_bench_live_fork": LV.FORK_COMMIT})
    monkeypatch.setattr(LV.subprocess, "run", lambda cmd, **kw: _proc(0, labels))


def test_live_prereq_hard_errors_when_the_snapshot_cannot_be_fetched(monkeypatch, tmp_path):
    _docker_ok(monkeypatch)
    _fake_datasets(monkeypatch, OSError("no network"))
    ok, why = LV.check_swe_bench_live_prerequisites()
    assert not ok and "Could not load dataset" in why and "no network" in why
    monkeypatch.setattr(A, "require_agent", lambda *a: None)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*Could not load dataset"):
        LV.run_swe_bench_live("m", "http://x/v1", 1)


def test_live_prereq_replaces_a_bad_cached_snapshot(monkeypatch, tmp_path):
    _docker_ok(monkeypatch)
    rows = _rows(LV, 300)
    path = _snapshot_path(tmp_path, LV._DATASET, LV._REVISION, "lite")
    _write(path, rows[:1])
    _fake_datasets(monkeypatch, rows)
    monkeypatch.setitem(LV.SPLIT_SHA256, "lite", C.snapshot_sha256(rows))
    assert LV.check_swe_bench_live_prerequisites(["lite"]) == (True, "")
    assert len(C._read_jsonl(str(path))) == 300 and len(glob.glob(f"{path}.invalid-*")) == 1


def test_live_prereq_refuses_rows_that_are_not_the_pinned_content(monkeypatch, tmp_path):
    _docker_ok(monkeypatch)
    _fake_datasets(monkeypatch, _rows(LV, 300))
    ok, why = LV.check_swe_bench_live_prerequisites()
    assert not ok and "content sha256" in why
    assert not _snapshot_path(tmp_path, LV._DATASET, LV._REVISION, "lite").exists()


def test_live_prereq_checks_docker_and_the_image_before_any_fetch(monkeypatch):
    monkeypatch.setattr(LV, "SETTINGS", LV.PYTHON_SETTINGS)
    monkeypatch.setattr(LV.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(LV.subprocess, "run",
                        lambda cmd, **kw: _proc(1 if "inspect" in cmd else 0))
    calls = _fake_datasets(monkeypatch, OSError("must not fetch"))
    ok, why = LV.check_swe_bench_live_prerequisites()
    assert not ok and "is not built" in why and calls == []


# ============================================================================================
# no per-turn max_tokens on the mini-swe-agent paths
# ============================================================================================
def _v1_base():
    return ({"agent": {"step_limit": 250, "cost_limit": 3.0, "instance_template": "/testbed"},
             "environment": {"timeout": 60, "cwd": "/testbed"},
             "model": {"model_name": "anthropic/x",
                       "model_kwargs": {"drop_params": True, "temperature": 0.0}}},
            A.LEADERBOARD_CONFIG_SHA256, "1.17.5")


def _v2_desc():
    return {"version": "2.4.6", "config_name": A.AGENT_V2_CONFIG,
            "config_sha256": A.AGENT_V2_CONFIG_SHA256["2.4.6"],
            "config": {"agent": {"step_limit": 250, "cost_limit": 3.0},
                       "environment": {"timeout": 60},
                       "model": {"model_kwargs": {"drop_params": True}}}}


def _capture_rollouts(monkeypatch, truncated=1):
    seen = {}

    def _rollouts(tasks, config, concurrency, out_dir, eval_name="swe", rollout=None):
        seen["config"] = config
        return [{"sample_idx": i, "category": t.category, "agent_patch": "d\n",
                 "extra_payload": {"instance_id": t.instance_id}, "agent_exit_status": "Submitted",
                 "agent_infra_error": False, "agent_steps": 2, "truncated_turns": truncated,
                 "context_retry_turns": 0, "is_correct": False, "status": "OK"}
                for i, t in enumerate(tasks)]
    monkeypatch.setattr(A, "run_rollouts", _rollouts)
    return seen


async def _resolve_all(traces):
    for t in traces:
        t["is_correct"] = True


def _tasks(*ids):
    return [A.AgentTask(i, f"issue {i}", A.swebench_image(i), "org/repo") for i in ids]


@pytest.mark.parametrize("eval_name", ["swe_bench_live", "multi_swe_bench"])
def test_v1_path_sends_no_per_turn_max_tokens(monkeypatch, eval_name):
    monkeypatch.setattr(A, "load_agent_config", _v1_base)
    seen = _capture_rollouts(monkeypatch)
    res = A.run_agentic(eval_name, "m", "http://h:1/v1", 1, _tasks("a__a-1", "b__b-2"),
                        _resolve_all, {}, max_output_tokens=32768)
    kw = seen["config"]["model"]["model_kwargs"]
    assert "max_tokens" not in kw and kw["drop_params"] is True
    assert res["max_output_tokens_per_turn"] is None
    assert res["max_output_tokens_note"].startswith("--max-output-tokens=32768 is not applied")
    # a length-truncated turn filled the context window, as upstream's request would: recorded
    assert res["truncated_turns"] == 2
    assert "per-turn" not in (res.get("leaderboard_comparable_reason") or "")


@pytest.mark.parametrize("eval_name", ["copilot_bench_swe", "swe_bench_multilingual"])
def test_v2_path_sends_no_per_turn_max_tokens(monkeypatch, eval_name):
    monkeypatch.setattr(A, "describe_agent_v2", lambda v: _v2_desc())
    monkeypatch.setattr(A, "bridge_gateway", lambda: ("127.0.0.1", ""))   # the relay's host
    monkeypatch.setattr(A, "require_relay_reachable", lambda *a, **k: None)   # no docker
    # no turn cut off (a cut-off turn makes a run not comparable; see the token-limit tests)
    seen = _capture_rollouts(monkeypatch, truncated=0)
    # the run-wide knob, not only the kwarg, is recorded as not applied
    monkeypatch.setattr(A, "get_run_knob",
                        lambda name, default=None: 16384 if name == "max_output_tokens"
                        else default)
    res = A.run_agentic(eval_name, "m", "http://h:1/v1", 1, _tasks("a__a-1"), _resolve_all, {},
                        agent_v2="2.4.6")
    assert "max_tokens" not in seen["config"]["model"]["model_kwargs"]
    assert res["max_output_tokens_per_turn"] is None
    assert "--max-output-tokens=16384 is not applied" in res["max_output_tokens_note"]
    assert "mini-swe-agent 2.4.6" in res["max_output_tokens_note"]
    # the only deviation: the temperature gbench always sends (no-think: 0.0)
    assert seen["config"]["model"]["model_kwargs"]["temperature"] == 0.0
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"] == ("temperature=0.0 sent by the no-think rule (every no-think run samples the model under test at 0.0); the bundled mini-swe-agent 2.4.6 config sends none")


def test_the_builders_still_honour_an_explicit_cap():
    """A caller that asks for a per-turn cap still gets one; run_agentic never asks."""
    cfg = A.build_agent_config(_v1_base()[0], model_name="m", base_url="http://h:1",
                               temperature=0.0, thinking=False, max_output_tokens=64)
    assert cfg["model"]["model_kwargs"]["max_tokens"] == 64
    ov = A.build_agent_v2_overrides({}, model_name="m", base_url="http://h:1", temperature=None,
                                    thinking=False, max_output_tokens=None)
    assert "max_tokens" not in ov["model"]["model_kwargs"]


# ============================================================================================
# docs and the v5 image
# ============================================================================================
def _doc(name):
    return open(os.path.join(REPO, "docs", "evals", f"{name}.md"), encoding="utf-8").read()


@pytest.mark.parametrize("name", ["copilot_bench_swe", "swe_bench_multilingual", "swe_bench_live",
                                  "multi_swe_bench"])
def test_docs_say_max_output_tokens_is_sent_to_the_agent(name):
    # since 2026-10-11 the run's --max-output-tokens goes out on every agent turn by default
    # (swe_agentic.AGENT_MAX_TOKENS_FROM_RUN_ENV); =0 restores upstream's none
    flat = " ".join(_doc(name).split())
    assert "| `--max-output-tokens` | sent as every agent turn's `max_tokens`" in flat
    assert "`GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0` restores upstream's none" in flat
    assert "max_output_tokens_note" in flat and "not applied in the agentic mode" not in flat
    assert "context_retry_turns" not in flat and "truncated turns," not in flat
    assert "--max-output-tokens 32768" in flat          # the run command keeps the flag


@pytest.mark.parametrize("name,mod", [("copilot_bench_swe", CB), ("swe_bench_multilingual", ML),
                                      ("swe_bench_live", LV)])
def test_docs_name_the_fingerprints_and_the_report_step(name, mod):
    doc = _doc(name)
    flat = " ".join(doc.split())
    for sha in mod.SPLIT_SHA256.values():
        assert sha in doc
    assert "unique instance ids" not in flat and "--rewrite_reports" not in flat
    assert "make_run_report" in flat and "at most 5" in flat and "stays in `error_ids`" in flat


def test_live_doc_describes_the_repeated_id_and_the_prereq_fetch():
    flat = " ".join(_doc("swe_bench_live").split())
    assert "conan-io__conan-18153" in flat and "dataset_repeated_instance_ids" in flat
    assert "prerequisite check fetches and checks the snapshot of every selected setting" in flat


def test_copilot_doc_lists_every_mini_version_tag_of_the_verified_board():
    flat = " ".join(_doc("copilot_bench_swe").split())
    assert "tagged 2.0.0 (whose config still set temperature 0.0), 1.x and 0.0.0" in flat


def test_v5_image_pins_the_docker_cli_and_the_python_dependencies():
    docker_dir = os.path.join(REPO, "gbench", "docker")
    text = open(os.path.join(docker_dir, "swebench_v5.Dockerfile"), encoding="utf-8").read()
    assert "ARG DOCKER_CLI_VERSION=5:29.8.1-1~ubuntu.24.04~noble" in text
    assert '"docker-ce-cli=${DOCKER_CLI_VERSION}"' in text
    assert "/opt/swebench-dpkg-versions.txt" in text and "org.gbench.docker_cli" in text
    assert "-c /opt/swebench-constraints.txt" in text
    pins = [line.strip() for line in open(os.path.join(docker_dir, "swebench_v5.constraints.txt"),
                                          encoding="utf-8")
            if line.strip() and not line.startswith("#")]
    assert f"swebench=={C.SWEBENCH_V5_VERSION}" in pins
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+-]+", p) for p in pins)
    assert any(p.startswith("docker==") for p in pins)
