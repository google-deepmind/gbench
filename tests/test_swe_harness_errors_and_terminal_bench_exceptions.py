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

"""SWE-family harness errors leave the resolved rate; terminal_bench's exception classifier.

Docker, Harbor, the harnesses, the agents and the network are faked everywhere: no container
starts, nothing is pulled and no model endpoint or Gemini model is contacted.

* terminal_bench reads a Harbor exception by the phase the trial recorded and its type: a
  RuntimeError the model's own keystrokes cause (tmux send-keys after the session ended) stays
  the model's outcome with a note; exceptions before the agent started, endpoint failures,
  cancelled trials and verifier transfer failures on a healthy container are harness errors. An
  unreadable trial result and a sharded run's missing task are harness errors too.
* An instance whose agent could not run, that the harness gave no verdict, or whose harness pass
  wrote no report is a harness error in every SWE-family runner: out of the numerator and the
  denominator, counted by kind, and the rate withheld (None) when nothing was scored.
* swe_lancer leaves a system-error task out of pass@1.
* The DNS recheck after a run does not count a name the resolver says does not exist.
"""

import asyncio
import json
import os
import re
import subprocess
import tempfile
import types

import pytest

from gbench.runners.eval_suites import container_guard as CG
from gbench.runners.eval_suites import multi_swe_bench as M
from gbench.runners.eval_suites import swe_agentic as A
from gbench.runners.eval_suites import swe_bench_live as LV
from gbench.runners.eval_suites import swe_bench_pro as P
from gbench.runners.eval_suites import swe_lancer as SL
from gbench.runners.eval_suites import swebench_common as C
from gbench.runners.eval_suites import terminal_bench as TB

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMUX_MSG = ("t__abc: failed to send non-blocking keys: command='tmux send-keys -t t__abc ...', "
            "return_code=1, stderr='no server running on /tmp/tmux-0/default\\n', stdout=''")
PYTEST_RAN = ("============================= test session starts ==============================\n"
              "FAILED ../tests/test_outputs.py::test_output - AssertionError\n")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    for var in ("GBENCH_SHARD", "GBENCH_TB_KEEP_JOBS_DIR", "OPENAI_API_KEY",
                "GBENCH_COPILOT_BENCH_SWE_TEMPERATURE"):
        monkeypatch.delenv(var, raising=False)
    CG.reset_verdicts()
    C.reset_task_network_verdicts()
    yield
    CG.reset_verdicts()
    C.reset_task_network_verdicts()


def _cp(argv, rc=0, out="", err=""):
    return subprocess.CompletedProcess(argv, rc, out, err)


# ============================================================================================
# terminal_bench: the Harbor exception classifier
# ============================================================================================
def _result(exc, message="boom", traceback="", agent=True, verifier=False):
    data = {"trial_name": "t", "exception_info": {"exception_type": exc,
                                                  "exception_message": message,
                                                  "exception_traceback": traceback}}
    if agent:
        data["agent_execution"] = {"started_at": "2026-09-27T00:00:00"}
    if verifier:
        data["verifier"] = {"started_at": "2026-09-27T00:10:00"}
    return data


def test_the_models_own_tmux_failure_stays_the_models_outcome():
    """The reported case: the model's keystrokes ended its tmux session mid-batch; Terminus-2's
    next send-keys raises RuntimeError, no verifier runs, no reward."""
    why, note = TB.trial_exception_verdict(_result("RuntimeError", TMUX_MSG), None)
    assert why is None
    assert "not a known harness failure" in note and "model's outcome" in note
    assert TB.trial_exception_harness_error(_result("RuntimeError", TMUX_MSG), None) is None


@pytest.mark.parametrize("exc,message,traceback,agent,verifier", [
    # before the agent started: environment, healthcheck, agent setup
    ("EnvironmentStartTimeoutError", "timed out", "", False, False),
    ("RuntimeError", "Docker compose command failed", "", False, False),
    ("AgentSetupTimeoutError", "timed out", "", False, False),
    # a trial Harbor cancelled, at any phase
    ("CancelledError", "", "", True, False),
    # model-endpoint failures during the agent phase
    ("APIConnectionError", "Connection refused", "", True, False),
    ("InternalServerError", "500", "", True, False),
    ("BadRequestError", "invalid request", "", True, False),
    ("ApiRateLimitError", "rate limit", "", True, False),
    # a docker daemon nothing in the task container can stop
    ("RuntimeError", "failed to send keys: stderr='Cannot connect to the Docker daemon at "
     "unix:///var/run/docker.sock. Is the docker daemon running?'", "", True, False),
    # a verifier transfer failure on a container that was still there
    ("AddTestsDirError", "Failed to add tests directory to environment.",
     "RuntimeError: Docker compose command failed ... Stdout: Error response from daemon: "
     "i/o timeout", True, True),
    ("DownloadVerifierDirError", "Failed to download verifier directory", "", True, True),
])
def test_known_harness_failures_are_harness_errors(exc, message, traceback, agent, verifier):
    why, note = TB.trial_exception_verdict(
        _result(exc, message, traceback, agent=agent, verifier=verifier), None)
    assert why and why.startswith(exc) and note is None


@pytest.mark.parametrize("exc,verifier", [
    ("AgentTimeoutError", False), ("NonZeroAgentExitCodeError", False),
    ("ContextLengthExceededError", False), ("OutputLengthExceededError", False),
    ("VerifierTimeoutError", True), ("RewardFileNotFoundError", True),
    ("RewardFileEmptyError", True), ("VerifierOutputParseError", True),
])
def test_the_models_outcomes_carry_no_note(exc, verifier):
    assert TB.trial_exception_verdict(_result(exc, verifier=verifier), None) == (None, None)


@pytest.mark.parametrize("marker", ["service \"main\" is not running",
                                    "Error response from daemon: container abc is not running",
                                    "extraction point is not a directory",
                                    "container rootfs is marked read-only file system"])
def test_a_verifier_upload_to_a_container_the_model_broke_stays_the_models(marker):
    why, note = TB.trial_exception_verdict(_result(
        "AddTestsDirError", "Failed to add tests directory to environment.",
        f"RuntimeError: Docker compose command failed. Stdout: {marker}", verifier=True), None)
    assert why is None and "had stopped or its target path was unusable" in note


def test_a_reward_is_the_verdict_whatever_the_exception():
    assert TB.trial_exception_verdict(_result("RuntimeError", TMUX_MSG), 0.0) == (None, None)
    assert TB.trial_exception_verdict(_result("APIConnectionError"), 1.0) == (None, None)


#: Harbor 0.20.0's default RetryConfig.exclude_exceptions (harbor/models/job/config.py, the
#: field's default_factory), the release terminal_bench runs in its harness image. Harbor runs only
#: in that image, so the host's Harbor (another release, or none) is never what is checked.
HARBOR_0_20_0_RETRY_EXCLUDED = frozenset({
    "AgentTimeoutError", "VerifierTimeoutError", "RewardFileNotFoundError",
    "RewardFileEmptyError", "VerifierOutputParseError", "ApiUsageLimitError",
    "AgentSafetyRefusalError", "AgentAuthenticationError", "ModelNotFoundError"})


def test_the_docstrings_attribute_only_harbors_own_exclusions_to_harbor():
    """Harbor's default RetryConfig.exclude_exceptions holds AgentTimeoutError and the four
    verifier exceptions; the three other model outcomes are gbench's attribution."""
    assert TB.HARBOR_VERSION == "0.20.0", "record the new release's RetryConfig defaults above"
    excluded = HARBOR_0_20_0_RETRY_EXCLUDED
    assert {"AgentTimeoutError", *TB.VERIFIER_OUTCOME_EXCEPTIONS} <= excluded
    for exc in ("NonZeroAgentExitCodeError", "ContextLengthExceededError",
                "OutputLengthExceededError"):
        assert exc not in excluded
    doc = open(os.path.join(REPO, "docs", "evals", "terminal_bench.md"), encoding="utf-8").read()
    text = " ".join(doc.split())
    assert "gbench's own attribution" in text and "exceptions_kept_as_model_outcome" in text
    assert "as Harbor's own default retry policy and metrics treat it" not in text
    src = " ".join(TB.__doc__.split())
    assert "gbench's own attribution" in src


# ------------------------------------------------------------------ terminal_bench end to end
def _trial(root, name, reward, *, exc=None, message=None, verifier="", agent_started=True,
           verifier_started=True, broken=False):
    trial = os.path.join(root, "job", name)
    os.makedirs(os.path.join(trial, "verifier"))
    os.makedirs(os.path.join(trial, "agent"))
    with open(os.path.join(trial, "verifier", "test-stdout.txt"), "w") as f:
        f.write(verifier)
    with open(os.path.join(trial, "agent", "trajectory.json"), "w") as f:
        json.dump({"steps": [{"message": "ls"}]}, f)
    if broken:
        with open(os.path.join(trial, "result.json"), "w") as f:
            f.write("{not json")
        return
    res = {"trial_name": name}
    if reward is not None:
        res["verifier_result"] = {"rewards": {"reward": reward}}
    if exc:
        res["exception_info"] = {"exception_type": exc,
                                 "exception_message": message or f"{exc} happened"}
    if agent_started:
        res["agent_execution"] = {"started_at": "t0"}
    if verifier_started:
        res["verifier"] = {"started_at": "t1"}
    with open(os.path.join(trial, "result.json"), "w") as f:
        json.dump(res, f)


def _run_harbor(monkeypatch, trials, shard_tasks=None):
    monkeypatch.setattr(TB, "check_terminal_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(TB, "require_task_metadata_blocked", lambda task_names=None: None)
    monkeypatch.setattr(TB, "prepare_task_images",
                        lambda task_names=None: ({"cached": 0}, {"present": 0}))
    monkeypatch.setattr(TB, "dns_recheck", lambda task_names, hosts: list(hosts))
    if shard_tasks is not None:
        monkeypatch.setenv("GBENCH_SHARD", "1/1")
        monkeypatch.setattr(TB, "_dataset_task_names", lambda dataset: list(shard_tasks))
    else:
        # A complete run of a dataset of exactly these tasks, one trial each: the run counts every
        # expected trial (FULL_TASK_COUNT x the trials per task) without a result as a harness error.
        monkeypatch.setattr(TB, "FULL_TASK_COUNT", len(trials))
        monkeypatch.setattr(TB, "TERMINAL_BENCH_TRIALS", 1)

    class _Proc:
        returncode = 0
        stdout = iter(())

        def poll(self):
            return 0

        def wait(self):
            return 0

    def _popen(cmd, **kw):
        jobs = cmd[cmd.index("--jobs-dir") + 1]
        for name, reward, kwargs in trials:
            _trial(jobs, name, reward, **kwargs)
        return _Proc()
    monkeypatch.setattr(TB.subprocess, "Popen", _popen)
    return TB.run_terminal_bench("m", "http://h:1/v1", 1)


def test_a_models_tmux_failure_counts_in_the_denominator(monkeypatch):
    res = _run_harbor(monkeypatch, [
        ("t-pass", 1.0, {"verifier": PYTEST_RAN.replace("FAILED", "PASSED")}),
        ("t-tmux", None, {"exc": "RuntimeError", "message": TMUX_MSG,
                          "verifier_started": False}),
        ("t-api", None, {"exc": "APIConnectionError", "verifier_started": False}),
    ])
    # t-tmux is the model's 0; t-api is a harness error left out
    assert res["total_questions"] == 2 and res["correct_answers"] == 1
    assert res["accuracy"] == 50.0
    assert res["harness_errors"] == {"exception": 1}
    assert res["harness_error_trials"] == ["t-api"]
    assert res["exceptions_kept_as_model_outcome"] == {"RuntimeError": 1}
    by = {t["trial_name"]: t for t in res["sample_traces"]}
    assert "harness_error" not in by["t-tmux"] and "model's outcome" in by["t-tmux"][
        "exception_note"]
    assert res["status"] == "completed_with_errors"


def test_an_unreadable_trial_result_is_a_harness_error(monkeypatch):
    res = _run_harbor(monkeypatch, [
        ("t-pass", 1.0, {"verifier": PYTEST_RAN.replace("FAILED", "PASSED")}),
        ("t-fail", 0.0, {"verifier": PYTEST_RAN}),
        ("t-broken", None, {"broken": True}),
    ])
    assert res["parse_failures"] == 1 and res["harness_errors"] == {"parse": 1}
    assert res["failed_requests"] == 1 and res["partial"] is True
    assert res["total_questions"] == 2 and res["accuracy"] == 50.0
    assert res["status"] == "completed_with_errors"
    assert "1 trial result(s) could not be read (harness errors), left out of the score" in res[
        "leaderboard_comparable_reason"]


def test_a_sharded_runs_missing_task_is_a_harness_error_not_a_zero(monkeypatch):
    res = _run_harbor(monkeypatch, [
        ("t-pass", 1.0, {"verifier": PYTEST_RAN.replace("FAILED", "PASSED")}),
        ("t-fail", 0.0, {"verifier": PYTEST_RAN}),
    ], shard_tasks=["org/a", "org/b", "org/c"])
    # 3 selected tasks x the leaderboard's 5 trials each, 2 trials found
    assert TB.TERMINAL_BENCH_TRIALS == 5
    assert res["harness_errors"] == {"missing": 13} and res["failed_requests"] == 13
    assert res["total_questions"] == 2 and res["accuracy"] == 50.0
    assert res["status"] == "completed_with_errors"
    assert ("13 trial(s) of the selected tasks without a result (3 task(s) x 5 trials"
            in res["leaderboard_comparable_reason"])


# ============================================================================================
# the shared helpers
# ============================================================================================
def _trace(iid, correct=False, category="c", **extra):
    return {"extra_payload": {"instance_id": iid}, "category": category, "is_correct": correct,
            **extra}


def test_scored_headline_leaves_harness_errors_out():
    traces = [_trace(f"i{n}", correct=n < 5) for n in range(7)]
    for n in range(7, 10):
        t = _trace(f"i{n}")
        C.mark_harness_error(t, C.AGENT_INFRA, "ImagePullFailed")
        traces.append(t)
    head = C.scored_headline(traces)
    # the example of the report: 10 instances, 3 harness errors, 5 of 7 resolved -> 71.43
    assert head["accuracy"] == 71.43 and head["scored_instances"] == 7
    assert head["harness_errors"] == {"agent_infra": 3}
    assert head["harness_error_ids"] == ["i7", "i8", "i9"]
    assert head["category_accuracy"] == {"c": {"correct": 5, "total": 7, "accuracy": 71.43}}
    assert C.scored_headline(traces[7:])["accuracy"] is None


def test_mark_harness_error_keeps_a_kind_already_recorded():
    t = _trace("a", correct=True, harness_error={"kind": "endpoint_failure", "detail": "x"})
    C.mark_harness_error(t, C.NO_VERDICT, "no verdict")
    assert t["is_correct"] is False and t["scoring_excluded"] is True
    assert C.harness_error_kind(t) == "endpoint_failure"


def test_the_row_headline_leaves_a_harness_errors_rows_out():
    rows = [{"instance_id": "a"}, {"instance_id": "a"}, {"instance_id": "b"},
            {"instance_id": "c"}]
    traces = [_trace("a", True), _trace("b"), _trace("c")]
    C.mark_harness_error(traces[2], C.NO_VERDICT)
    res = {"sample_traces": traces, "total_questions": 3, "correct_answers": 1,
           "scored_instances": 2}
    C.row_total_headline(res, rows)
    assert res["total_questions"] == 4 and res["total_instance_ids"] == 3
    assert res["scored_instances"] == 3 and res["accuracy"] == 33.33


def test_the_dns_recheck_does_not_count_a_name_that_does_not_exist(monkeypatch):
    monkeypatch.setattr(C, "local_probe_image", lambda images, references=(): "img")

    @__import__("contextlib").contextmanager
    def _net(eval_name, docs_url):
        yield "gbench-net"
    monkeypatch.setattr(C, "_probe_network", _net)
    monkeypatch.setattr(CG, "_run_probe", lambda *a, **k: (CG.UNREACHABLE, "", {
        "mirror-typo.ubuntu.com": ("fail", "gaierror:-2"),
        "archive.ubuntu.com": ("fail", "gaierror:-3")}))
    assert C.unresolved_after_run("terminal_bench", ["img"],
                                  ["mirror-typo.ubuntu.com", "archive.ubuntu.com"]) == [
        "archive.ubuntu.com"]
    assert C.unresolved_after_run("terminal_bench", ["img"], ["mirror-typo.ubuntu.com"]) == []


# ============================================================================================
# the swebench scorer (copilot_bench_swe, swe_bench_multilingual, swe_bench_live Python tabs)
# ============================================================================================
def _swebench_run(monkeypatch, report):
    monkeypatch.setattr(C, "require_harness_network_blocked", lambda *a, **k: None)
    monkeypatch.setattr(C, "_REPORT_RETRY_BACKOFF_S", (0, 0, 0, 0, 0))
    monkeypatch.setenv("GBENCH_SWE_OMP_THREADS", "0")

    def _run(cmd, cwd=None, **kw):
        run_id = cmd[cmd.index("--run_id") + 1]
        os.makedirs(os.path.join(cwd, "logs", "run_evaluation", run_id), exist_ok=True)
        if report is not None:
            with open(os.path.join(cwd, f"gbench__m.{run_id}.json"), "w") as f:
                json.dump(report, f)
        return _cp(cmd, 0)
    monkeypatch.setattr(C.subprocess, "run", _run)


def test_the_swebench_scorer_marks_instances_without_a_verdict(monkeypatch):
    _swebench_run(monkeypatch, {"total_instances": 3, "completed_instances": 1,
                                "resolved_ids": ["a"], "error_ids": ["b"],
                                "empty_patch_instances": 1})
    traces = [_trace("a", agent_patch="d\n"), _trace("b", agent_patch="d\n"),
              _trace("c", agent_patch="")]
    metrics = {}
    asyncio.run(C.make_swebench_scorer("x", "m", "d.jsonl", "test", None, 1, metrics)(traces))
    a, b, c = traces
    assert a["is_correct"] and not a.get("scoring_excluded")
    assert b["scoring_excluded"] and b["harness_error"] == C.NO_VERDICT
    assert b["status"] == "HARNESS_ERROR"
    assert not c.get("scoring_excluded") and not c["is_correct"]      # an empty patch: model's


def test_a_harness_pass_without_a_report_leaves_every_submitted_patch_out(monkeypatch):
    _swebench_run(monkeypatch, None)
    traces = [_trace("a", agent_patch="d\n"), _trace("c", agent_patch="")]
    metrics = {}
    asyncio.run(C.make_swebench_scorer("x", "m", "d.jsonl", "test", None, 1, metrics)(traces))
    assert traces[0]["harness_error"] == C.NO_REPORT and traces[0]["scoring_excluded"]
    assert not traces[1].get("scoring_excluded")


# ============================================================================================
# swe_agentic.run_agentic (copilot_bench_swe, swe_bench_multilingual)
# ============================================================================================
def _agent_trace(i, iid, infra=False):
    return {"sample_idx": i, "source_sample_idx": i, "attempt_idx": 0, "category": "r",
            "messages": [], "extra_payload": {"instance_id": iid}, "gold_answer": "",
            "response_text": "" if infra else "diff --git a/x b/x\n",
            "agent_patch": "" if infra else "diff --git a/x b/x\n",
            "agent_exit_status": "ImagePullFailed" if infra else "Submitted",
            "agent_error": "instance image: pull failed" if infra else None,
            "agent_infra_error": infra, "agent_steps": 0 if infra else 3, "trajectory": [],
            "truncated_turns": 0, "empty_after_reasoning_turns": 0, "context_retry_turns": 0,
            "is_correct": False, "status": "AGENT_INFRA_ERROR" if infra else "OK"}


def _run_agentic(monkeypatch, n, infra, resolved, verdictless=()):
    tasks = [A.AgentTask(instance_id=f"o__r-{k}", problem_statement="p", image="img",
                         category="r") for k in range(n)]
    monkeypatch.setattr(A, "require_rollout_network_blocked", lambda *a, **k: None)
    monkeypatch.setattr(A, "run_rollouts", lambda tasks, *a, **k: [
        _agent_trace(i, t.instance_id, infra=t.instance_id in infra)
        for i, t in enumerate(tasks)])
    seen = []

    async def _score(traces):
        seen.extend(t["extra_payload"]["instance_id"] for t in traces)
        for t in traces:
            iid = t["extra_payload"]["instance_id"]
            t["is_correct"] = iid in resolved
            if iid in verdictless:
                C.mark_harness_error(t, C.NO_VERDICT, "no verdict")
    res = A.run_agentic("copilot_bench_swe", "m", "http://127.0.0.1:9/v1", 1, tasks, _score, {})
    return res, seen


def test_agent_infrastructure_failures_leave_the_resolved_rate(monkeypatch):
    infra = {f"o__r-{k}" for k in (7, 8, 9)}
    resolved = {f"o__r-{k}" for k in range(5)}
    res, seen = _run_agentic(monkeypatch, 10, infra, resolved)
    assert not set(seen) & infra                          # the scorer never saw them
    assert res["total_questions"] == 10 and res["correct_answers"] == 5
    assert res["scored_instances"] == 7 and res["accuracy"] == 71.43
    assert res["harness_errors"] == {"agent_infra": 3} and res["failed_requests"] == 3
    assert res["status"] == "completed_with_errors"
    assert res["category_accuracy"]["r"] == {"correct": 5, "total": 7, "accuracy": 71.43}
    by = {t["extra_payload"]["instance_id"]: t for t in res["sample_traces"]}
    assert by["o__r-7"]["scoring_excluded"] and by["o__r-7"]["harness_error"] == "agent_infra"


def test_a_run_of_nothing_but_harness_errors_withholds_its_rate(monkeypatch):
    res, _ = _run_agentic(monkeypatch, 3, {"o__r-0"}, set(), verdictless={"o__r-1", "o__r-2"})
    assert res["accuracy"] is None and res["status"] == "error"
    assert "no instance was scored" in res["error"]
    assert res["accuracy_withheld_reason"] == res["error"]
    assert res["harness_errors"] == {"agent_infra": 1, "harness_no_verdict": 2}
    res, _ = _run_agentic(monkeypatch, 2, {"o__r-0", "o__r-1"}, set())
    assert res["accuracy"] is None and res["status"] == "error"
    assert "could not run on any of 2" in res["accuracy_withheld_reason"]


# ============================================================================================
# multi_swe_bench
# ============================================================================================
class _MsweHarness:
    """The multi-swe-bench harness stand-in: final_report.json with `errors` in error_ids."""

    def __init__(self, resolved, errors=(), report=True):
        self.resolved, self.errors, self.report = set(resolved), set(errors), report

    def __call__(self, argv, **kw):
        cfg = json.load(open(argv[argv.index("--config") + 1]))
        rows = [json.loads(line) for line in open(cfg["patch_files"][0])]
        resolved, unresolved, errors = [], [], []
        for r in rows:
            iid = f"{r['org']}/{r['repo']}:pr-{r['number']}"
            if r["number"] in self.errors:
                errors.append(iid)
            else:
                (resolved if r["number"] in self.resolved else unresolved).append(iid)
        if self.report:
            os.makedirs(cfg["output_dir"], exist_ok=True)
            with open(os.path.join(cfg["output_dir"], "final_report.json"), "w") as f:
                json.dump({"resolved_ids": resolved, "unresolved_ids": unresolved,
                           "completed_ids": resolved + unresolved, "error_ids": errors,
                           "incomplete_ids": [], "total_instances": len(rows)}, f)
        return _cp(argv, 0)


def _mswe_traces(numbers, empty=()):
    return [{"sample_idx": i, "category": "java",
             "extra_payload": {"instance_id": f"o__r-{n}", "org": "o", "repo": "r",
                               "number": n},
             "agent_patch": "" if n in empty else "diff --git a/x b/x\n"}
            for i, n in enumerate(numbers)]


def test_mswe_submissions_without_a_verdict_leave_the_score(monkeypatch):
    monkeypatch.setattr(M.subprocess, "run", _MsweHarness(resolved={1}, errors={3}))
    traces = _mswe_traces([1, 2, 3])
    metrics = {}
    asyncio.run(M._make_scorer("m", 1, metrics)(traces))
    by = {t["extra_payload"]["number"]: t for t in traces}
    assert by[3]["scoring_excluded"] and by[3]["harness_error"] == C.NO_VERDICT
    assert not by[2].get("scoring_excluded") and not by[2]["is_correct"]
    result = {"status": "success"}
    M._report_reasons(result, metrics)
    assert result["status"] == "completed_with_errors"
    assert result["harness_errors"] == {"harness_no_verdict": 1}
    head = C.scored_headline(traces)
    assert head["accuracy"] == 50.0 and head["scored_instances"] == 2


def test_mswe_without_a_report_leaves_every_patch_out(monkeypatch):
    monkeypatch.setattr(M.subprocess, "run", _MsweHarness(resolved=(), report=False))
    traces = _mswe_traces([1, 2], empty={2})
    asyncio.run(M._make_scorer("m", 1, {})(traces))
    assert traces[0]["harness_error"] == C.NO_REPORT and traces[0]["scoring_excluded"]
    assert not traces[1].get("scoring_excluded")


def test_mswe_agentic_leaves_agent_infrastructure_failures_out(monkeypatch):
    tasks = [A.AgentTask(instance_id=f"o__r-{n}", problem_statement="p",
                         image=f"mswebench/o_m_r:pr-{n}", category="java",
                         meta={"org": "o", "repo": "r", "number": n}) for n in (1, 2, 3, 4)]
    monkeypatch.setattr(M, "_load_multi_swe_bench_tasks", lambda: (tasks, {}))

    class _Relay:
        url = "http://127.0.0.1:9/v1"

        def __init__(self, *a, **k):
            pass

        def start(self):
            return self

        def stop(self):
            pass

        def record(self):
            return {"failures": {}, "requests": 0}
    monkeypatch.setattr(M, "EndpointRelay", _Relay)
    monkeypatch.setattr(M, "relay_result_reasons", lambda *a, **k: [])
    summary = {"instances": {f"o__r-{n}": {"completed": True, "exit_status": "submitted",
                                           "patch": "diff --git a/x b/x\n"} for n in (1, 2)},
               "images": {"o__r-3": {"error": "pull failed"}, "o__r-4": {"error": "pull failed"}}}
    monkeypatch.setattr(M, "run_mswe_agent", lambda tasks, *a, **k: [
        M.mswe_agent_trace(i, t, summary, "/tmp", 0.0) for i, t in enumerate(tasks)])
    monkeypatch.setattr(M.subprocess, "run", _MsweHarness(resolved={1}))
    res = M._run_agentic("m", "http://127.0.0.1:1/v1", 1, False, {})
    assert res["total_questions"] == 4 and res["correct_answers"] == 1
    assert res["scored_instances"] == 2 and res["accuracy"] == 50.0
    assert res["harness_errors"] == {"agent_infra": 2} and res["failed_requests"] == 2
    assert res["status"] == "completed_with_errors"


# ============================================================================================
# swe_bench_live
# ============================================================================================
def test_live_launch_scorer_marks_error_ids_and_the_tab_leaves_them_out(monkeypatch):
    rows = [{"instance_id": f"go-{i}", "docker_image": f"img{i}", "repo": "o/r"}
            for i in range(3)]
    monkeypatch.setattr(LV, "_harness_rows", lambda rows: rows)
    monkeypatch.setattr(LV, "host_pull_images", lambda images, workers=1: {
        i: {"state": "present"} for i in images})

    def _run(cmd, **kw):
        out = cmd[cmd.index("--output_dir") + 1]
        os.makedirs(out, exist_ok=True)
        json.dump({"submitted": 3, "success_ids": ["go-0"], "failure_ids": ["go-1"],
                   "empty_patch_ids": [], "error_ids": ["go-2"], "incomplete_ids": []},
                  open(os.path.join(out, "results.json"), "w"))
        return _cp(cmd, 0)
    monkeypatch.setattr(LV.subprocess, "run", _run)
    metrics = {}
    traces = [_trace(r["instance_id"], agent_patch="d\n") for r in rows]
    asyncio.run(LV.make_launch_scorer("go", rows, "linux", 1, metrics)(traces))
    assert [t["is_correct"] for t in traces] == [True, False, False]
    assert traces[2]["harness_error"] == C.NO_VERDICT and traces[2]["scoring_excluded"]
    assert C.scored_headline(traces)["accuracy"] == 50.0


def test_live_combined_category_totals_are_the_scored_instances():
    res = {"status": "completed_with_errors", "total_questions": 10, "scored_instances": 8,
           "correct_answers": 4, "accuracy": 50.0, "leaderboard_comparable": False,
           "leaderboard_comparable_reason": "partial", "sample_traces": []}
    combined = LV.combine_settings({"lite": res})
    assert combined["category_accuracy"]["lite"] == {"correct": 4, "total": 8, "accuracy": 50.0}
    assert combined["scored_instances"] == 8
    err = {"status": "error", "error": "boom", "total_questions": 2, "accuracy": None,
           "leaderboard_comparable": False, "leaderboard_comparable_reason": "x",
           "sample_traces": []}
    combined = LV.combine_settings({"lite": err})
    assert combined["accuracy"] is None and "boom" in combined["accuracy_withheld_reason"]


# ============================================================================================
# swe_bench_pro V2
# ============================================================================================
def test_pro_rates_leave_harness_errors_out():
    tasks = [types.SimpleNamespace(instance_id=i, problem_statement="p", image="img",
                                   category="go") for i in ("a", "b", "c", "d")]
    views = {"a": {"model_outcome": True, "patch": "d", "reward": 1.0, "exit_status": "Submitted"},
             "b": {"model_outcome": True, "patch": "d", "reward": 0.0, "exit_status": "Submitted"},
             "c": {"model_outcome": True, "patch": "d", "reward": None,
                   "exit_status": "Submitted"},
             "d": {"model_outcome": False, "patch": "", "reward": None, "why": "no trial"}}
    regrade = {"a": {"reward": 1.0}, "b": {"reward": 0.0},
               "c": {"reward": None, "exception": "EnvironmentStartTimeoutError",
                     "exception_message": "x"}}
    res = P._assemble_v2({}, tasks, views, regrade, {"a", "d"}, [], False, 0.0)
    assert res["accuracy"] == 50.0 and res["scored_instances"] == 2
    assert res["harness_errors"] == {"agent_infra": 1, "no_regrade_verdict": 1}
    assert res["in_sandbox"]["total"] == 3 and res["in_sandbox"]["resolved"] == 1
    assert res["hard51"]["regraded"]["total"] == 1
    assert res["status"] == "completed_with_errors"
    none = P._assemble_v2({}, tasks[3:], {"d": views["d"]}, {}, set(), [], False, 0.0)
    assert none["accuracy"] is None and none["status"] == "error"
    assert none["accuracy_withheld_reason"] == none["error"]


def test_pro_one_shot_leaves_unregraded_tasks_out(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_TRAJ_DIR", str(tmp_path / "runs"))

    def _harbor(workdir, args, job):
        jobs_dir = args[args.index("--jobs-dir") + 1]
        tdir = os.path.join(jobs_dir, job, "a__x")
        os.makedirs(tdir, exist_ok=True)
        json.dump({"task_name": "swebench-pro/a", "verifier_result": {"rewards": {"reward": 1}},
                   "agent_execution": {"started_at": "t"}, "verifier": {"started_at": "t"}},
                  open(os.path.join(tdir, "result.json"), "w"))
        return 0, ""
    monkeypatch.setattr(P, "_run_harbor", _harbor)
    traces = [_trace("a", response_text="```diff\nd\n```"), _trace("b", response_text="x")]
    metrics = {}
    asyncio.run(P._make_v2_oneshot_scorer(1, metrics)(traces))
    assert traces[0]["is_correct"] and not traces[0].get("scoring_excluded")
    assert traces[1]["harness_error"] == P.NO_REGRADE_VERDICT and traces[1]["scoring_excluded"]


# ============================================================================================
# swe_lancer
# ============================================================================================
def test_lancer_leaves_system_errors_out_of_pass_at_1(monkeypatch, tmp_path):
    rows = [{"question_id": "q1", "variant": "ic_swe", "price": 100.0, "correct": True,
             "earned": 100.0, "available": 100.0, "system_error": None},
            {"question_id": "q2", "variant": "ic_swe", "price": 100.0, "correct": False,
             "earned": 0.0, "available": 100.0, "system_error": None},
            {"question_id": "q3", "variant": "ic_swe", "price": 50.0, "correct": False,
             "earned": 0.0, "available": 50.0, "system_error": "RolloutSystemError: setup"}]

    def _run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            report = [a.split("=", 1)[1] for a in cmd if a.startswith("swelancer.report_path=")]
            json.dump({"summary": {}, "tasks": rows}, open(report[0], "w"))
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(SL.subprocess, "run", _run)
    res = SL._run_variant("ic_swe", ["q1", "q2", "q3"], True, "m", "http://r/v1", 1,
                          str(tmp_path))
    assert res["accuracy"] == 50.0 and res["scored_tasks"] == 2
    assert res["available_usd"] == 200.0 and res["harness_errors"] == {"system_error": 1}
    assert res["status"] == "completed_with_errors"
    assert res["sample_traces"][2]["scoring_excluded"]
    rows[:2] = [dict(r, system_error="RolloutSystemError: x") for r in rows[:2]]
    res = SL._run_variant("ic_swe", ["q1", "q2", "q3"], True, "m", "http://r/v1", 1,
                          str(tmp_path))
    assert res["accuracy"] is None and res["status"] == "error"


# ============================================================================================
# the docs say the same
# ============================================================================================
@pytest.mark.parametrize("doc,gone", [
    ("multi_swe_bench.md", "still count as unresolved in `accuracy`"),
    ("swe_bench_pro.md", "the denominator stays the selected tasks"),
    ("swe_bench_live.md", "count as unresolved and are listed under the setting's"),
    ("copilot_bench_swe.md", "counted as unresolved and listed in"),
    ("swe_bench_multilingual.md", "count as unresolved and are listed"),
    ("swe_lancer.md", "counts as not solved and makes"),
])
def test_the_swe_docs_no_longer_count_harness_errors_as_unresolved(doc, gone):
    text = " ".join(open(os.path.join(REPO, "docs", "evals", doc),
                         encoding="utf-8").read().split())
    assert gone not in text
    assert "left out of" in text


def test_the_swe_code_no_longer_says_harness_errors_count_as_unresolved():
    for mod in (A, M, C, LV, P, SL):
        src = open(mod.__file__, encoding="utf-8").read()
        assert not re.search(r"harness errors?[^.\n]{0,80}count as unresolved", src), mod
    src = open(A.__file__, encoding="utf-8").read()
    assert "endpoint failure) and count as unresolved" not in src
