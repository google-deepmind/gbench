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

"""Reporting helpers of the agentic suites: skillsbench's summary counts reconcile with its mean
reward and its rollout evidence is kept (redacted, capped) before the work directory goes;
swe_bench_pro leaves an in-sandbox verifier's infrastructure failure out of the in-sandbox rate;
wildbench names the task groups absent from a subset; the MSWE driver names the setup step that
failed; toolbench's withheld headline names the harness-error groups apart from judge outages and
flags a request log that counts more failures than the per-step records. Docker, the model and
the network are not used; files go to tmp only."""

import importlib.util
import json
from pathlib import Path

from gbench.runners.eval_suites import skillsbench as SB
from gbench.runners.eval_suites import swe_bench_pro as SBP
from gbench.runners.eval_suites import toolbench as TB
from gbench.runners.eval_suites import wildbench as WB

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# skillsbench
# --------------------------------------------------------------------------- #
def test_skillsbench_counts_reconcile_with_the_mean_reward():
    scoring = {"conditions": {"with_skills": {
        "n_tasks": 3, "n_scored": 2, "per_task": {"a": 1.0, "b": 0.5, "c": None},
        "harness_errors": {"c": {"kind": "agent_timeout"}}}}}
    counts = SB.headline_counts(scoring, "with_skills")
    assert counts["total_questions"] == 3 and counts["effective_n"] == 2
    assert counts["correct_answers"] == 1.5 and "solved-equivalent" in counts["correct_answers_note"]
    assert SB.headline_counts({}, None)["total_questions"] == 0


def test_skillsbench_keeps_rollout_evidence_before_the_workdir_goes(tmp_path, monkeypatch):
    monkeypatch.setenv("GBENCH_TEST_SECRET_FOR_ARTIFACTS", "sk-artifact-secret-123456789")
    monkeypatch.setattr(SB, "redact_file", lambda p: _redact(p, "sk-artifact-secret-123456789"))
    work = tmp_path / "work"
    rollout = work / "jobs" / "with_skills" / "task1" / "2026-09-29" / "task1__abc"
    (rollout / "verifier").mkdir(parents=True)
    (rollout / "trajectory").mkdir()
    (rollout / "result.json").write_text(json.dumps({"reward": 1.0}))
    (rollout / "verifier" / "reward.txt").write_text("1")
    (rollout / "verifier" / "test-stdout.txt").write_text("token sk-artifact-secret-123456789\n")
    (rollout / "trajectory" / "llm_trajectory.jsonl").write_text("big\n")
    (work / "skillsbench_summary.json").write_text("{}")
    dest = tmp_path / "results" / "skillsbench_artifacts" / "run1"
    assert SB.keep_artifacts(str(work), str(dest)) == str(dest)
    kept = dest / "with_skills" / "task1" / "task1__abc"
    assert (kept / "result.json").is_file() and (kept / "verifier" / "reward.txt").is_file()
    assert "sk-artifact-secret" not in (kept / "verifier" / "test-stdout.txt").read_text()
    assert not (kept / "trajectory" / "llm_trajectory.jsonl").exists()      # not kept
    assert (dest / "skillsbench_summary.json").is_file()
    assert SB.remove_workdir(str(work)) == [] and not work.exists()
    assert SB.keep_artifacts(str(work), None) is None


def _redact(path, secret):
    text = Path(path).read_text()
    Path(path).write_text(text.replace(secret, "[REDACTED]"))
    return True


# --------------------------------------------------------------------------- #
# swe_bench_pro
# --------------------------------------------------------------------------- #
def test_an_in_sandbox_verifier_infrastructure_failure_is_no_verdict():
    assert SBP.in_sandbox_verdict({"reward": 1}) == (1.0, None)
    reward, why = SBP.in_sandbox_verdict({"verifier_started": True,
                                          "exception": "DockerDaemonError"})
    assert reward is None and "in-sandbox verifier DockerDaemonError" in why
    # a verifier outcome exception and a trial that never reached its verifier score 0
    assert SBP.in_sandbox_verdict({"verifier_started": True,
                                   "exception": "VerifierTimeoutError"}) == (0.0, None)
    assert SBP.in_sandbox_verdict({"exception": "AgentTimeoutError"}) == (0.0, None)


# --------------------------------------------------------------------------- #
# wildbench
# --------------------------------------------------------------------------- #
def test_wildbench_names_the_task_groups_a_subset_lacks():
    present = {g: 1.0 for g in list(WB.TASK_MACRO_WEIGHTS)[:2]}
    absent = WB.task_groups_absent(present)
    assert absent == sorted(set(WB.TASK_MACRO_WEIGHTS) - set(present))
    assert WB.task_groups_absent({g: 1 for g in WB.TASK_MACRO_WEIGHTS}) == []
    assert WB.task_groups_absent(None) == sorted(WB.TASK_MACRO_WEIGHTS)


# --------------------------------------------------------------------------- #
# the MSWE driver (loaded from its file: it runs inside its image)
# --------------------------------------------------------------------------- #
def _mswe_driver():
    spec = importlib.util.spec_from_file_location(
        "mswe_agent_run_for_tests", REPO / "gbench" / "docker" / "mswe_agent_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_mswe_driver_names_the_setup_step_that_failed():
    D = _mswe_driver()
    assert D.setup_step(RuntimeError("apt-get update failed: 404")) == "apt"
    assert D.setup_step(RuntimeError("anything"), {"archive": "archive.debian.org"}) == "apt"
    assert D.setup_step(RuntimeError("git checkout abc failed")) == "repo_reset"
    assert D.setup_step(RuntimeError("docker: container exited")) == "container_start"
    assert D.setup_step(ValueError("odd")) == "other"


# --------------------------------------------------------------------------- #
# toolbench
# --------------------------------------------------------------------------- #
def test_the_withheld_headline_names_harness_groups_apart_from_judge_outages():
    scores = {"by_group": {"G1": {"sopr": 50.0, "sowr": 40.0}, "G2": {"sopr": None, "sowr": None}}}
    why = TB.withheld_reason(scores, {"G2": {"q1": {}, "q2": {}}}, {"G2": ["q1", "q2"]},
                             "2 the simulator gave no valid JSON")
    assert "every query of ['G2'] is a harness error (2 the simulator gave no valid JSON)" in why
    assert "judge outages" not in why.split("no SoWR")[0]
    assert why.endswith("re-run to replace the harness errors")
    why = TB.withheld_reason(scores, {}, {"G2": ["q1", "q2"]}, "")
    assert "['G2'] had no usable ToolEval verdict (judge outages or invalid replies)" in why
    full = {"by_group": {"G1": {"sopr": 50.0, "sowr": 40.0}}, "sopr": 50.0, "sowr": 40.0}
    assert TB.withheld_reason(full, {}, {"G1": ["q"]}, "") is None


def test_a_request_log_with_more_failures_than_the_step_records_is_flagged():
    assert TB.failure_records_inconsistency({"counts": {"failed_requests": 3}},
                                            {"failed_requests": 1}) == {
        "request_log_failed_requests": 3, "step_records_failed_requests": 1}
    assert TB.failure_records_inconsistency({"counts": {"failed_requests": 1}},
                                            {"failed_requests": 1}) is None
    assert TB.failure_records_inconsistency(None, {}) is None


def test_an_unfinished_mswe_instance_names_its_failed_setup_step(tmp_path):
    from gbench.runners.eval_suites import multi_swe_bench as M
    from gbench.runners.eval_suites import swe_agentic as A
    task = A.AgentTask("o__r-1", "task", "mswebench/o_m_r:pr-1", "go",
                       meta={"org": "o", "repo": "r", "number": 1})
    summary = {"images": {}, "instances": {"o__r-1": {
        "completed": False,
        "setup_error": {"step": "repo_reset", "error": "RuntimeError: git checkout failed"}}}}
    trace = M.mswe_agent_trace(0, task, summary, str(tmp_path), 0.0)
    assert "its setup failed at repo_reset: RuntimeError: git checkout failed" in \
        str(trace.get("agent_error") or trace.get("error"))
    summary["instances"]["o__r-1"].pop("setup_error")
    trace = M.mswe_agent_trace(0, task, summary, str(tmp_path), 0.0)
    # no recorded setup error: never called a setup failure (driver 4 records the rollout's
    # exception when it raised; an instance with neither record says so)
    text = str(trace.get("agent_error") or trace.get("error"))
    assert "recorded no setup or rollout error" in text and "setup failed" not in text
    assert trace["agent_infra_error"] is True
