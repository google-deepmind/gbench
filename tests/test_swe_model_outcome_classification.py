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
"""SWE family: what the model caused stays the model's outcome, and the mini-swe-agent 2.x rollout
container holds no credential.

* A swebench instance whose tests over the applied patch ran past the harness's timeout is the
  model's outcome (unresolved, scored), read only from the records swebench itself logs, so text
  of the model's diff in run_instance.log cannot move an instance either way.
* A SWE-agent exit_error / exit_environment_error, an MSWE-agent exit_error or early_exit and a
  Win-Agent "error" after the model's first turn are the model's outcome unless they name a
  model-endpoint failure the relay saw (the text alone can be the model's own).
* The 2.x rollout container runs on docker's default bridge with the placeholder key and reaches
  the endpoint through gbench's relay on that network's gateway, checked from that network first.

Hermetic: docker, the endpoint and the network are stubbed; every file is under tmp_path.
"""

import asyncio
import json
import os
import types

import pytest

from gbench.runners.eval_suites import multi_swe_bench as M
from gbench.runners.eval_suites import swe_agentic as A
from gbench.runners.eval_suites import swe_bench_live as LV
from gbench.runners.eval_suites import swebench_common as C

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(REPO, "docs", "evals")
SECRET = "sk-w2r-swe-secret-0123456789"


@pytest.fixture(autouse=True)
def _scratch(monkeypatch, tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("GBENCH_MINI_SWE_AGENT_V2_IMAGE", raising=False)
    monkeypatch.setattr(C, "require_task_network_blocked", lambda *a, **k: None)


def _proc(rc=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


# ============================================================================================
# swebench run_instance.log: test timeouts are the model's outcome
# ============================================================================================
def _rec(t, msg):
    """One record of swebench's per-instance logger ("%(asctime)s - %(levelname)s - ...")."""
    return f"2026-09-27 10:{t},000 - INFO - {msg}\n"


_APPLIED = (_rec("00:00", ">>>>> Applied Patch:") + "Checking patch x.py...\n"
            "Applied patch x.py cleanly.\n"
            + _rec("00:01", "Git diff before:") + "diff --git a/x.py b/x.py\n+print(1)\n")
_TIMED_OUT = (_APPLIED + _rec("30:00", "Test runtime: 1_800.00 seconds")
              + _rec("30:00", "Test output for i written to test_output.txt")
              + _rec("30:01", "Traceback (most recent call last):")
              + "swebench.harness.utils.EvaluationError: i: Test timed out after 1800 seconds.\n"
              "Check (run_instance.log) for more information.\n")


def _write_log(workdir, iid, text, run_id="r1", tag="gbench__m"):
    d = os.path.join(workdir, "logs", "run_evaluation", run_id, tag, iid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "run_instance.log"), "w", encoding="utf-8") as f:
        f.write(text)


def test_a_test_timeout_after_the_patch_applied_is_the_models_outcome(tmp_path):
    wd = str(tmp_path)
    _write_log(wd, "django__django-11099", _TIMED_OUT)
    classes = C.classify_error_ids(wd, ["django__django-11099"])
    assert classes["test_timeout"] == {"django__django-11099"}
    assert classes["harness_error"] == set() and classes["patch_apply_failed"] == set()
    model_errored, harness_err, scoreable = C._split_report(
        {"error_ids": ["django__django-11099"], "completed_instances": 1}, wd)
    assert model_errored == {"django__django-11099"} and harness_err == set() and scoreable == 2
    # the reproduction: the timed-out instance and one resolved one -> 50.0, not 100.0
    traces = [{"extra_payload": {"instance_id": "a__a-1"}, "is_correct": True},
              {"extra_payload": {"instance_id": "django__django-11099"}, "is_correct": False}]
    head = C.scored_headline(traces)
    assert head["accuracy"] == 50.0 and head["scored_instances"] == 2


@pytest.mark.parametrize("log", [
    "",                                                                  # no log at all
    _rec("00:00", "Container for i started: abc") + "docker.errors.APIError: 500 Server Error\n",
    _rec("00:00", "Error building image i: Container build timed out after 3600s"),
    # the patch applied (so "Hunk #1 succeeded at" is patch --fuzz output), then the harness
    # crashed before the tests: no verdict
    _rec("00:00", ">>>>> Applied Patch:") + "patching file x.py\nHunk #1 succeeded at 7.\n"
    + _rec("00:01", "Error in evaluating model for i: 404 Client Error: No such container"),
])
def test_other_errored_instances_stay_harness_errors(tmp_path, log):
    wd = str(tmp_path)
    if log:
        _write_log(wd, "i", log)
    assert C.classify_error_ids(wd, ["i"])["harness_error"] == {"i"}


def test_the_models_diff_cannot_fake_a_marker(tmp_path):
    wd = str(tmp_path)
    fake_pass = _rec("00:00", ">>>>> Applied Patch:").strip()
    fake_timeout = "Test timed out after 5 seconds."
    # a patch that did not apply, whose lines patch echoes (prefixed "|") name the pass marker
    _write_log(wd, "p", _rec("00:00", "Failed to apply patch to container: git apply --verbose")
               + _rec("00:01", ">>>>> Patch Apply Failed:")
               + "can't find file to patch at input line 3\nThe text leading up to this was:\n"
               f"|{fake_pass}\n|--- a/y.py\n"
               + f"patch: **** malformed patch at line 4: {fake_pass}\n")
    # a diff naming the timeout, logged before the tests ran (Git diff before), then a crash
    _write_log(wd, "q", _rec("00:00", ">>>>> Applied Patch:") + _rec("00:01", "Git diff before:")
               + f"+# {fake_timeout}\n"
               + _rec("00:02", "Error in evaluating model for q: container died"))
    # tests that finished in time: the diff after them names the timeout, then a crash
    _write_log(wd, "r", _APPLIED + _rec("10:00", "Test runtime: 600.00 seconds")
               + _rec("10:00", "Git diff after:") + f"+# {fake_timeout}\n"
               + _rec("10:01", "Error in evaluating model for r: grading crashed"))
    classes = C.classify_error_ids(wd, ["p", "q", "r"])
    assert classes["patch_apply_failed"] == {"p"}
    assert classes["test_timeout"] == set() and classes["harness_error"] == {"q", "r"}


def test_scorer_keeps_a_timed_out_instance_scored_and_records_it(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "require_harness_network_blocked", lambda *a, **k: None)

    def _run(cmd, cwd=None, **kw):
        run_id = cmd[cmd.index("--run_id") + 1]
        tag = json.loads(open(cmd[cmd.index("--predictions_path") + 1]).readline())[
            "model_name_or_path"]
        _write_log(cwd, "b__b-2", _TIMED_OUT, run_id, tag)
        _write_log(cwd, "c__c-3", _rec("00:00", "docker.errors.APIError: 500"), run_id, tag)
        with open(os.path.join(cwd, f"{tag}.{run_id}.json"), "w") as f:
            json.dump({"total_instances": 3, "submitted_instances": 3, "completed_instances": 1,
                       "resolved_instances": 1, "unresolved_instances": 0, "error_instances": 2,
                       "empty_patch_instances": 0, "resolved_ids": ["a__a-1"],
                       "error_ids": ["b__b-2", "c__c-3"]}, f)
        return _proc(0)
    monkeypatch.setattr(C.subprocess, "run", _run)
    metrics = {}
    traces = [{"extra_payload": {"instance_id": i}, "agent_patch": "d\n"}
              for i in ("a__a-1", "b__b-2", "c__c-3")]
    asyncio.run(C.make_swebench_scorer("copilot_bench_swe", "m", "ds.jsonl", "test", None, 2,
                                       metrics, harness_image="gbench-swebench-v5:5.0.2",
                                       harness_v5=True)(traces))
    a, b, c = traces
    assert a["is_correct"] is True
    assert b["test_timed_out"] is True and not b.get("scoring_excluded") and not b["is_correct"]
    assert c["scoring_excluded"] is True and c["harness_error"] == C.NO_VERDICT
    assert "neither a patch-apply failure nor a test timeout" in c["harness_error_reason"]
    head = C.scored_headline(traces)
    assert head["accuracy"] == 50.0 and head["harness_errors"] == {C.NO_VERDICT: 1}
    res = {"status": "success"}
    reasons = C.swebench_report_reasons("copilot_bench_swe", res, metrics)
    rep = metrics["swebench_report"]
    assert (rep["test_timeout_instances"], rep["test_timeout_ids"]) == (1, ["b__b-2"])
    assert rep["harness_error_instances"] == 1 and rep["patch_apply_failed_instances"] == 0
    assert res["partial_evaluation"]["test_timeout_instances"] == 1
    assert res["partial_evaluation"]["harness_error_instances"] == 1
    assert res["status"] == "completed_with_errors" and reasons


# ============================================================================================
# SWE-agent (swe_bench_live): error exits by evidence
# ============================================================================================
_EOF = ("Exit due to environment error: End Of File (EOF). Exception style platform. "
        "| Exited (autosubmitted)")
_LITELLM = ("Exit due to unknown error: litellm.InternalServerError: OpenAIException - "
            "Connection error. | Exited (autosubmitted)")


def test_sweagent_eof_of_a_killed_shell_is_the_models_outcome():
    error, note = LV.sweagent_exit_verdict("submitted (exit_environment_error)", _EOF, 0)
    assert error is None and "the model's outcome" in note and "End Of File" in note
    # a SwerexException never comes from the endpoint, whatever its text says
    error, note = LV.sweagent_exit_verdict(
        "exit_environment_error",
        "Exit due to environment error: litellm.InternalServerError: fake", 7)
    assert error is None and note


def test_sweagent_endpoint_exit_is_a_harness_error_only_when_the_relay_saw_one():
    error, note = LV.sweagent_exit_verdict("exit_error", _LITELLM, 12)
    assert error and "a model endpoint failure" in error and note is None
    assert LV.sweagent_exit_verdict("exit_error", _LITELLM, None)[0]       # no relay to ask
    error, note = LV.sweagent_exit_verdict("exit_error", _LITELLM, 0)
    assert error is None and "the relay did not see" in note
    # the docker daemon's message cannot come from a SWE-agent step: the model's outcome
    error, note = LV.sweagent_exit_verdict(
        "exit_error", "Exit due to runtime error: Cannot connect to the Docker daemon at "
        "unix:///var/run/docker.sock. Is the docker daemon running?", 3)
    assert error is None and note
    # exit_api (a RetryError the fork's model calls never raise): a harness error only when the
    # tab's relay saw an endpoint failure (or no relay carried the run), else the model's outcome
    assert LV.sweagent_exit_verdict("submitted (exit_api)", "Exit due to retry error: x", 2)[0]
    assert LV.sweagent_exit_verdict("submitted (exit_api)", "Exit due to retry error: x", None)[0]
    error, note = LV.sweagent_exit_verdict("submitted (exit_api)", "Exit due to retry error: x", 0)
    assert error is None and "the model's outcome" in note and "RetryError" in note
    assert LV.sweagent_exit_verdict(
        "exit_error", "Exit due to unknown error: ... is longer than the maximum model length",
        5) == (None, None)
    assert LV.sweagent_exit_verdict("submitted", "", 0) == (None, None)


def _sweagent_out(tmp_path, iid, status, thought):
    out = tmp_path / "run"
    inst = out / iid
    inst.mkdir(parents=True)
    (inst / f"{iid}.pred").write_text(json.dumps({"model_patch": "diff --git a/x b/x\n"}))
    (inst / f"{iid}.traj").write_text(json.dumps({
        "info": {"exit_status": status, "model_stats": {"api_calls": 9}},
        "trajectory": [{"thought": "t", "response": "r", "observation": "o"},
                       {"thought": thought, "response": thought,
                        "observation": "Exited (autosubmitted)"}]}))
    return str(out)


def test_read_sweagent_run_keeps_a_killed_shell_and_counts_it(tmp_path):
    task = A.AgentTask("o__r-1", "p", "img", "o/r", meta={"split": "lite"})
    out = _sweagent_out(tmp_path, "o__r-1", "submitted (exit_environment_error)",
                        "Exit due to environment error: End Of File (EOF).")
    tr = LV.read_sweagent_run([task], out, 0.0, relay_failures=0)[0]
    assert tr["agent_infra_error"] is False and tr["status"] == "OK"
    assert "the model's outcome" in tr["agent_exit_note"]
    out2 = _sweagent_out(tmp_path / "b", "o__r-1", "submitted (exit_error)",
                         "Exit due to unknown error: litellm.APIConnectionError: refused")
    tr2 = LV.read_sweagent_run([task], out2, 0.0, relay_failures=5)[0]
    assert tr2["agent_infra_error"] is True and "agent_exit_note" not in tr2


# ============================================================================================
# Win-Agent (swe_bench_live Windows tab)
# ============================================================================================
def _entry(body):
    return f"\n[2026-09-27 10:00:00]: \n{body}\n"


_WIN_TURN = _entry('[\n {"role": "system"}\n]') + _entry("Step 0...") + _entry('{"content": "x"}')


def test_win_agent_error_before_the_first_turn_is_a_harness_error():
    pre = _entry("Docker is not installed or not running.\nTraceback ...\nPatch not saved.")
    error, note = LV.win_agent_error_verdict(pre, 0)
    assert error and "before the model's first turn" in error and note is None
    assert LV.win_agent_error_verdict(None, 0)[0]


def test_win_agent_error_after_a_turn_is_the_models_outcome_unless_the_relay_saw_it():
    killed = _WIN_TURN + _entry("Status inside container: \nPS C:\\testbed> ") + _entry(
        "Connection aborted.\nTraceback (most recent call last):\n  ...\n"
        "Patch might not be saved.")
    error, note = LV.win_agent_error_verdict(killed, 0)
    assert error is None and "the model's outcome" in note and "Connection aborted" in note
    endpoint = _WIN_TURN + _entry("litellm.InternalServerError: OpenAIException - 502\n"
                                  "Traceback ...\nPatch might not be saved.")
    assert LV.win_agent_error_verdict(endpoint, 5)[0]
    error, note = LV.win_agent_error_verdict(endpoint, 0)
    assert error is None and "the relay did not see" in note
    # the model's command output (logged raw) forging an endpoint entry after the real one
    forged = killed + _entry("Status inside container: \n") + _entry(
        "litellm.InternalServerError: fake\nTraceback ...\nPatch not saved.")
    assert LV.win_agent_error_verdict(forged, 0)[0] is None
    # a docker daemon message after the first turn is not one Win-Agent's rollout writes
    docker = _WIN_TURN + _entry("Docker is not installed or not running.\nPatch not saved.")
    assert LV.win_agent_error_verdict(docker, 9)[0] is None


def test_run_win_agent_asks_the_relay(monkeypatch, tmp_path):
    agent = tmp_path / "win-agent-src"
    (agent / "config").mkdir(parents=True)
    (agent / "main.py").write_text("# stand-in\n")
    (agent / "config" / "default.yaml").write_text("platform: windows\nmax_steps: 100\n")
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_WIN_AGENT_DIR", str(agent))
    out = tmp_path / "out"
    out.mkdir()
    log = _WIN_TURN + _entry("litellm.InternalServerError: x\nTraceback\nPatch might not be "
                             "saved.")

    def _run(cmd, **kw):
        import yaml
        cfg = yaml.safe_load(open(cmd[cmd.index("--config_path") + 1]))
        d = os.path.join(kw["cwd"], "output", LV._win_agent_run_id(cfg))
        os.makedirs(os.path.join(d, "logs"))
        json.dump({}, open(os.path.join(d, "preds.json"), "w"))
        json.dump({"w-1": "error"}, open(os.path.join(d, "exit_status.json"), "w"))
        open(os.path.join(d, "logs", "w-1.txt"), "w").write(log)
        return _proc()
    monkeypatch.setattr(LV.subprocess, "run", _run)
    rows = [{"instance_id": "w-1", "docker_image": "img", "_split": "test"}]
    task = A.AgentTask("w-1", "p", "img", "o/r", meta={"split": "test"})

    class _Relay:
        def __init__(self, failures):
            self.failures = failures

        def record(self):
            return {"upstream_errors": {"http_502": self.failures} if self.failures else {},
                    "requests_refused": 0}
    kept = LV.run_win_agent([task], rows, "m", "http://127.0.0.1:1/v1", 1, str(out),
                            relay=_Relay(0))[0]
    assert kept["agent_infra_error"] is False and kept["agent_exit_note"]
    out2 = tmp_path / "out2"
    out2.mkdir()
    infra = LV.run_win_agent([task], rows, "m", "http://127.0.0.1:1/v1", 1, str(out2),
                             relay=_Relay(5))[0]
    assert infra["agent_infra_error"] is True


def test_relay_failure_count():
    assert C.relay_failure_count(None) is None
    assert C.relay_failure_count({"upstream_errors": {"connection": 2, "http_500": 3},
                                  "requests_refused": 1}) == 6
    assert C.relay_failure_count({"upstream_errors": {}, "requests_refused": 0}) == 0


# ============================================================================================
# MSWE-agent (multi_swe_bench)
# ============================================================================================
def _mswe(tmp_path, status, last_step, step_limit=False):
    traj_rel = "trajectories/gbench/run/o__r-1.traj"
    path = tmp_path / traj_rel
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"trajectory": [{"response": "r0", "observation": "o0"},
                                               last_step]}))
    summary = {"instances": {"o__r-1": {
        "completed": True, "patch": "", "exit_status": status, "api_calls": 7, "steps": 7,
        "step_limit_reached": step_limit, "trajectory_file": "/mswe/" + traj_rel}},
        "images": {}}
    task = A.AgentTask("o__r-1", "task", "mswebench/o_m_r:pr-1", "go",
                       meta={"org": "o", "repo": "r", "number": 1})
    return M.mswe_agent_trace(0, task, summary, str(tmp_path), 0.0)


def test_mswe_early_exit_is_the_models_outcome(tmp_path):
    tr = _mswe(tmp_path, "early_exit", {"response": "kill -9 $$",
                                        "observation": "\nCOMMAND FAILED TO EXECUTE."})
    assert tr["agent_infra_error"] is False and tr["status"] == "OK"
    assert "early_exit" in tr["agent_exit_note"] and "COMMAND FAILED TO EXECUTE" in \
        tr["agent_exit_note"]


def test_mswe_exit_error_by_evidence(tmp_path):
    kept = _mswe(tmp_path / "a", "submitted (exit_error)",
                 {"response": "exit due to runtime error: 'NoneType' object is not iterable"})
    assert kept["agent_infra_error"] is False and kept["agent_exit_note"]
    daemon = _mswe(tmp_path / "b", "exit_error",
                   {"response": "exit due to runtime error: Cannot connect to the Docker daemon "
                                "at unix:///var/run/docker.sock"})
    assert daemon["agent_infra_error"] is True
    limit = _mswe(tmp_path / "c", "submitted (exit_error)",
                  {"response": "exit due to runtime error: [o__r-1] Exceed maximum number of "
                               "steps: 50"}, step_limit=True)
    assert limit["agent_infra_error"] is False and "agent_exit_note" not in limit


# ============================================================================================
# mini-swe-agent 2.x: the rollout container holds no credential
# ============================================================================================
def test_v2_container_is_on_the_bridge_with_the_placeholder_key():
    cmd = A._v2_container_cmd("img", "2.4.6", "/out", "/out/x/spec.json", "n1")
    assert cmd[cmd.index("--network") + 1] == A.AGENT_V2_NETWORK == "bridge"
    assert "host" not in cmd
    assert [c for c in cmd if "OPENAI_API_KEY" in c] == ["OPENAI_API_KEY=EMPTY"]


def test_relay_probe_command_and_verdicts(monkeypatch):
    calls = []
    verdict = {"out": "GBENCH_METADATA_PROBE=reachable\n"}

    def _run(cmd, **kw):
        calls.append((list(cmd), kw.get("env") or {}))
        return _proc(0, verdict["out"])
    monkeypatch.setattr(A.subprocess, "run", _run)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    A.require_relay_reachable("copilot_bench_swe", "172.17.0.1", 40001)
    cmd, env = calls[0]
    assert cmd[:3] == ["docker", "run", "--rm"] and "--pull" in cmd
    assert cmd[cmd.index("--pull") + 1] == "never"
    assert cmd[cmd.index("--network") + 1] == "bridge"
    assert "--read-only" in cmd and cmd[cmd.index("--cap-drop") + 1] == "ALL"
    assert cmd[cmd.index("--entrypoint") + 1] == "python3" and A.agent_v2_image() in cmd
    assert "('172.17.0.1', 40001)" in cmd[-1]
    assert SECRET not in json.dumps(cmd) and SECRET not in json.dumps(env)
    verdict["out"] = "GBENCH_METADATA_PROBE=unreachable 111\n"
    with pytest.raises(RuntimeError, match="cannot open a TCP connection to 172.17.0.1:40001"):
        A.require_relay_reachable("copilot_bench_swe", "172.17.0.1", 40001)
    verdict["out"] = "no verdict"
    with pytest.raises(RuntimeError, match="gave no verdict"):
        A.require_relay_reachable("copilot_bench_swe", "172.17.0.1", 40001)


def _desc():
    return {"version": "2.4.6", "config_name": A.AGENT_V2_CONFIG,
            "config_sha256": A.AGENT_V2_CONFIG_SHA256["2.4.6"],
            "config": {"agent": {"step_limit": 250, "cost_limit": 3.0},
                       "environment": {"timeout": 60, "env": {"PAGER": "cat"}},
                       "model": {"model_kwargs": {"drop_params": True}}}}


def _task(iid="a__a-1"):
    return A.AgentTask(iid, f"issue {iid}", A.swebench_image(iid), "org/repo",
                       meta={"split": "test"})


async def _all_resolved(traces):
    for t in traces:
        t["is_correct"] = True


def test_run_agentic_v2_points_the_agent_at_the_relay_on_the_gateway(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setattr(A, "describe_agent_v2", lambda v: _desc())
    monkeypatch.setattr(A, "bridge_gateway", lambda: ("127.0.0.1", ""))
    probes, relays = [], []
    monkeypatch.setattr(A, "require_relay_reachable",
                        lambda name, host, port, docs=None: probes.append((host, port)))
    real = A.EndpointRelay

    class _Relay(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            relays.append(self)
    monkeypatch.setattr(A, "EndpointRelay", _Relay)
    seen = []

    def _rollout(i, task, overrides, out_dir, *, image, version):
        seen.append(overrides)
        return {"sample_idx": i, "category": task.category,
                "extra_payload": {"instance_id": task.instance_id}, "agent_patch": "d\n",
                "agent_exit_status": "Submitted", "agent_infra_error": False, "agent_steps": 3,
                "truncated_turns": 0, "context_retry_turns": 0, "is_correct": False,
                "status": "OK"}
    monkeypatch.setattr(A, "_rollout_v2", _rollout)
    res = A.run_agentic("swe_bench_multilingual", "m", "http://h:1/v1", 1, [_task()],
                        _all_resolved, {}, agent_v2="2.4.6")
    relay = relays[0]
    assert relay.bind_host == "127.0.0.1" and relay._key == SECRET
    assert relay._server is None                                  # stopped after the rollouts
    api_base = seen[0]["model"]["model_kwargs"]["api_base"]
    assert api_base == relay.url and probes == [("127.0.0.1", int(relay.url.split(":")[2]
                                                                  .split("/")[0]))]
    assert res["agent"]["api_base"] == relay.url and res["relay"]["requests_relayed"] == 0
    assert SECRET not in json.dumps(res, default=str)


def test_run_agentic_v2_stops_the_relay_when_the_bridge_cannot_reach_it(monkeypatch):
    monkeypatch.setattr(A, "describe_agent_v2", lambda v: _desc())
    monkeypatch.setattr(A, "bridge_gateway", lambda: ("127.0.0.1", ""))
    relays = []
    real = A.EndpointRelay

    class _Relay(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            relays.append(self)
    monkeypatch.setattr(A, "EndpointRelay", _Relay)

    def _unreachable(name, host, port, docs=None):
        raise C.infra_required(name, "unreachable", docs)
    monkeypatch.setattr(A, "require_relay_reachable", _unreachable)
    monkeypatch.setattr(A, "_rollout_v2", lambda *a, **k: pytest.fail("no rollout"))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        A.run_agentic("swe_bench_multilingual", "m", "http://h:1/v1", 1, [_task()],
                      _all_resolved, {}, agent_v2="2.4.6")
    assert relays and relays[0]._server is None


def test_run_agentic_v2_relay_endpoint_failures_are_harness_errors(monkeypatch):
    monkeypatch.setattr(A, "describe_agent_v2", lambda v: _desc())
    monkeypatch.setattr(A, "bridge_gateway", lambda: ("127.0.0.1", ""))
    monkeypatch.setattr(A, "require_relay_reachable", lambda *a, **k: None)

    class _Relay(A.EndpointRelay):
        def record(self):
            rec = super().record()
            rec["upstream_errors"] = {"http_503": 2}
            return rec
    monkeypatch.setattr(A, "EndpointRelay", _Relay)
    monkeypatch.setattr(A, "_rollout_v2", lambda i, task, o, d, *, image, version: {
        "sample_idx": i, "category": task.category,
        "extra_payload": {"instance_id": task.instance_id}, "agent_patch": "d\n",
        "agent_exit_status": "Submitted", "agent_infra_error": False, "agent_steps": 3,
        "truncated_turns": 0, "context_retry_turns": 0, "is_correct": False, "status": "OK"})
    res = A.run_agentic("swe_bench_multilingual", "m", "http://h:1/v1", 1, [_task()],
                        _all_resolved, {}, agent_v2="2.4.6")
    assert res["endpoint_failures"]["upstream_errors"] == {"http_503": 2}
    assert res["status"] == "completed_with_errors" and res["leaderboard_comparable"] is False
    assert "the served endpoint failed 2 relayed model request(s)" in \
        res["leaderboard_comparable_reason"]


# ============================================================================================
# docs match the code
# ============================================================================================
def test_docs_describe_the_classification():
    def _doc(name):
        return " ".join(open(os.path.join(DOCS, f"{name}.md"), encoding="utf-8").read().split())
    for name in ("copilot_bench_swe", "swe_bench_multilingual"):
        text = _doc(name)
        assert "docker's default bridge network with the placeholder key" in text
        assert "root-equivalent access to the host's docker daemon" in text
        assert "test_timeout_instances" in text and "neither a patch-apply failure nor a test" \
            " timeout" in text
        assert "the agent container uses the host network" not in text.lower()
    live = _doc("swe_bench_live")
    assert "exits_kept_as_model_outcome" in live and "win_agent_error_verdict" in live
    assert "test_timeout_instances" in live and "the relay saw an endpoint failure" in live
    mswe = _doc("multi_swe_bench")
    assert "early_exit" in mswe and "exits_kept_as_model_outcome" in mswe
    assert "is `early_exit` (the container stopped answering), names `exit_api`" not in mswe
