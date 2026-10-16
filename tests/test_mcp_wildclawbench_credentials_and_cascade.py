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

"""mcp_bench, mcp_atlas and wildclawbench: server keys out of the model's container, the Gemini
cascade handed in from gbench, and harness errors attributed by their cause.

* mcp_bench: the five third-party server keys go only to the credentialed-servers container; the
  model container reaches those servers through the bridge and gbench's relay, whose session
  protocol, key masking and failure accounting are checked here without docker. The in-container
  judge makes one attempt per model (max_retries=0), skips a model the API does not serve, and
  reports the model that answered; a partial judge outage makes the run completed_with_errors; an
  endpoint URL with user info is refused; the harness heuristics attribute per call and scan
  error text only.
* mcp_atlas: its metadata-guard calls resolve the servers' host names; an environment gbench did
  not start is checked for credential variables through its own code tool.
* wildclawbench: a model the Gemini API does not serve is never a content refusal; a stop of the
  agent's leftover processes that fails for a reason the agent cannot cause turns a Gemini
  service failure or a cap hit of grade() into a harness error; a log-only name lookup is
  confirmed only by a resolver failure of the monitor; the answering models are recorded per task.

Docker, Gemini and the network are stubbed; nothing is written outside tmp.
"""

import asyncio
import itertools
import json
import os
import socket
import subprocess
import sys
import threading
import time
import types
import urllib.error

import pytest

from gbench.runners.eval_suites import base as B
from gbench.runners.eval_suites import container_guard as CG
from gbench.runners.eval_suites import mcp_atlas as M
from gbench.runners.eval_suites import mcp_bench as MB
from gbench.runners.eval_suites import wildclawbench as W

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOCKER_DIR = os.path.join(_REPO, "gbench", "docker")
sys.path.insert(0, _DOCKER_DIR)
import mcp_bench_bridge as BRIDGE  # noqa: E402

_REAL_POPEN = subprocess.Popen
import mcp_bench_cascade_judge as CJ  # noqa: E402
import mcp_bench_credentialed as CRED  # noqa: E402
import mcp_bench_run as MRUN  # noqa: E402
import wildclawbench_cascade_judge as J  # noqa: E402
import wildclawbench_run as RUN  # noqa: E402


def _read(rel):
    with open(os.path.join(_REPO, rel), encoding="utf-8") as f:
        return f.read()


def _flat(text):
    return " ".join(text.split())


def _jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


# =========================================================================== #
# The Gemini cascade: one source, handed into every container
# =========================================================================== #
def test_the_in_container_defaults_equal_the_base_cascade(monkeypatch):
    assert J._DEFAULT_JUDGE_CASCADE == list(B.GEMINI_CASCADE)
    assert CJ._DEFAULT_JUDGE_CASCADE == list(B.GEMINI_CASCADE)
    for mod in (J, CJ):
        monkeypatch.delenv("GBENCH_JUDGE_MODELS", raising=False)
        monkeypatch.delenv("GBENCH_JUDGE_MODEL", raising=False)
        assert mod._cascade() == list(B.GEMINI_CASCADE)
        monkeypatch.setenv("GBENCH_JUDGE_MODEL", "gemini-one")
        assert mod._cascade() == ["gemini-one"]
        # the list gbench passes wins over everything
        monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-a, gemini-b")
        assert mod._cascade() == ["gemini-a", "gemini-b"]


def test_mcp_bench_passes_the_runs_cascade_into_the_container(monkeypatch):
    monkeypatch.setenv("GBENCH_JUDGE_MODEL", "gemini-single")
    monkeypatch.delenv("GBENCH_JUDGE_MODELS", raising=False)
    src = _read("gbench/runners/eval_suites/mcp_bench.py")
    assert "gemini_cascade_env()['GBENCH_JUDGE_MODELS']" in src
    # GBENCH_JUDGE_MODEL is resolved by gbench, not forwarded as a second knob
    assert '("GBENCH_JUDGE_MODEL", "GBENCH_JUDGE_MODEL")' not in src
    assert B.gemini_cascade_env()["GBENCH_JUDGE_MODELS"] == "gemini-single"


# =========================================================================== #
# wildclawbench proxy: a model that is not served is not a content refusal
# =========================================================================== #
def _http_error(code, body="x"):
    import io
    return urllib.error.HTTPError("u", code, "e", {}, io.BytesIO(body.encode()))


@pytest.fixture
def cascade_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "m-gone,m-a,m-b")
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", "3")
    monkeypatch.setenv("GBENCH_JUDGE_BACKOFF", "0")
    monkeypatch.setenv("WILDCLAW_WORKDIR", str(tmp_path))
    monkeypatch.setattr(J.time, "sleep", lambda s: None)
    return tmp_path


def _script(outcomes):
    calls = []

    def one(model):
        calls.append(model)
        out = outcomes[model]
        if isinstance(out, Exception):
            raise out
        return out
    return one, calls


def test_a_not_served_model_does_not_hide_a_content_refusal(cascade_env):
    one, calls = _script({"m-gone": _http_error(404), "m-a": _http_error(400),
                          "m-b": _http_error(422)})
    with pytest.raises(J.RequestRejected):
        J._run_cascade(one, "judge", record=False)
    # the 404 model is asked once, and a pass of refusals ends the call
    assert calls == ["m-gone", "m-a", "m-b"]


def test_no_served_model_is_an_outage_that_names_the_models(cascade_env):
    one, calls = _script({m: _http_error(404) for m in ("m-gone", "m-a", "m-b")})
    with pytest.raises(RuntimeError) as ei:
        J._run_cascade(one, "judge", record=True)
    assert not isinstance(ei.value, J.RequestRejected)
    assert "JUDGE_OUTAGE" in str(ei.value)
    assert "not served (HTTP 404): m-a, m-b, m-gone" in str(ei.value)
    assert calls == ["m-gone", "m-a", "m-b"]            # one pass: nothing else to try
    [ev] = [json.loads(x) for x in (cascade_env / J.HARNESS_EVENTS).read_text().splitlines()]
    assert ev["kind"] == "judge" and ev["service_failure"] is True


def test_a_service_failure_with_a_refusal_is_an_outage(cascade_env):
    one, calls = _script({"m-gone": _http_error(404), "m-a": _http_error(503),
                          "m-b": _http_error(400)})
    with pytest.raises(RuntimeError, match="JUDGE_OUTAGE") as ei:
        J._run_cascade(one, "judge", record=False)
    assert not isinstance(ei.value, J.RequestRejected)
    assert calls == ["m-gone", "m-a", "m-b", "m-a", "m-b", "m-a", "m-b"]


def test_a_refusal_is_recorded_as_not_a_service_failure(cascade_env):
    one, _calls = _script({"m-gone": _http_error(404), "m-a": _http_error(400),
                           "m-b": _http_error(400)})
    with pytest.raises(J.RequestRejected):
        J._run_cascade(one, "judge", record=True)
    [ev] = [json.loads(x) for x in (cascade_env / J.HARNESS_EVENTS).read_text().splitlines()]
    assert ev["service_failure"] is False


def test_the_answering_model_is_known_to_the_request(cascade_env):
    one, _calls = _script({"m-gone": _http_error(404), "m-a": {"ok": 1}, "m-b": {"ok": 2}})
    assert J._run_cascade(one, "judge", record=False) == {"ok": 1}
    assert J.last_answering_model() == "m-a"
    J.record_answer("judge", J.last_answering_model(), "172.17.0.9", start=5.0, grader=True)
    [ans] = [json.loads(x) for x in (cascade_env / J.GEMINI_ANSWERS).read_text().splitlines()]
    assert ans["model"] == "m-a" and ans["client"] == "172.17.0.9" and ans["grader"] is True
    assert ans["start"] == 5.0 and ans["kind"] == "judge"


def test_the_judge_request_keeps_the_tasks_temperature_or_none(cascade_env, monkeypatch):
    sent = []

    class _Resp:
        def __init__(self, payload):
            self._b = json.dumps(payload).encode()

        def read(self):
            return self._b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        sent.append(json.loads(req.data.decode()))
        return _Resp({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})
    monkeypatch.setattr(J.urllib.request, "urlopen", fake_urlopen)
    J.cascade_chat_completion({"model": "openai/gpt-5.4", "max_tokens": 300,
                               "response_format": {"type": "json_object"},
                               "messages": [{"role": "user", "content": "grade"}]})
    J.cascade_chat_completion({"model": "x", "temperature": 0,
                               "extra": 1, "thinking": {"type": "disabled"},
                               "messages": [{"role": "user", "content": "grade"}]})
    assert "temperature" not in sent[0]                    # the task sent none: none is sent
    assert sent[0]["max_tokens"] == 300 + J.GEMINI_THINKING_ALLOWANCE and sent[0]["response_format"] == {"type": "json_object"}
    assert sent[1]["temperature"] == 0 and "thinking" not in sent[1]


def test_the_gaia2_proxy_keeps_its_hooks():
    # gaia2_cascade_judge replaces _gemini_base and calls _run_cascade(one, "judge")
    import inspect
    sig = inspect.signature(J._run_cascade)
    assert list(sig.parameters)[:3] == ["call_one", "section", "record"]
    assert sig.parameters["record"].default is True
    assert "(base_check or _gemini_base)()" in _read("gbench/docker/wildclawbench_cascade_judge.py")


# =========================================================================== #
# wildclawbench launcher: why a stop failed, and what it means for grading
# =========================================================================== #
def _fake_docker(stop=None, running="true", python_ok=True):
    """A docker stand-in for stop_leftover_processes: `stop` is the stop exec's result
    (CompletedProcess or an exception); `docker top` lists only the processes from before the
    agent (PID 1)."""
    def run(cmd, **kw):
        if cmd[:2] == ["docker", "top"]:
            return subprocess.CompletedProcess(cmd, 0, "PID COMMAND\n4242 sleep infinity\n", "")
        if cmd[:2] == ["docker", "inspect"]:
            if running is None:
                return subprocess.CompletedProcess(cmd, 1, "", "Error: No such container")
            return subprocess.CompletedProcess(cmd, 0, running + "\n", "")
        if cmd[-3:] == ["python3", "-c", "pass"]:
            return subprocess.CompletedProcess(cmd, 0 if python_ok else 1, "", "")
        if isinstance(stop, BaseException):
            raise stop
        return stop
    return run


@pytest.mark.parametrize("stop, running, python_ok, cause", [
    (subprocess.TimeoutExpired(["docker"], 120), "true", True, "infra"),
    (OSError("docker gone"), "true", True, "infra"),
    (subprocess.CompletedProcess([], 0, json.dumps({"stopped": 1, "left": [42]}), ""), "true",
     True, "model"),
    (subprocess.CompletedProcess([], 127, "", "python3: not found"), "true", True, "model"),
    (subprocess.CompletedProcess([], 126, "", "permission denied"), "true", True, "model"),
    (subprocess.CompletedProcess([], 1, "", "Error response from daemon"), None, True,
     "container_down"),
    (subprocess.CompletedProcess([], 1, "", "Error response from daemon"), "true", True, "infra"),
    (subprocess.CompletedProcess([], 1, "", "boom"), "true", False, "model"),
    (subprocess.CompletedProcess([], 0, "hello", ""), "true", True, "model"),
])
def test_a_failed_stop_records_whether_the_agent_can_have_caused_it(stop, running, python_ok,
                                                                    cause):
    res = RUN.stop_leftover_processes("t1", [[1, 1.0]], run=_fake_docker(stop, running, python_ok))
    assert res["ok"] is False and res["cause"] == cause
    assert cause in RUN.STOP_FAILURE_CAUSES


def test_no_process_list_is_an_infra_cause():
    res = RUN.stop_leftover_processes("t1", None, run=_fake_docker())
    assert res["ok"] is False and res["cause"] == "infra"


@pytest.fixture(autouse=True)
def _fresh_launcher_state():
    RUN._BASELINES.clear()
    RUN._ISOLATION.clear()
    yield
    RUN._BASELINES.clear()
    RUN._ISOLATION.clear()


def _harness(tmp_path, tids):
    root = tmp_path / "h"
    for tid in tids:
        cat = tid.split("_task_")[0]
        d = root / "tasks" / cat
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{tid}_x.md").write_text(
            f"---\nid: {tid}\nname: X\ncategory: {cat}\ntimeout_seconds: 60\n"
            f"modality: pure-text\n---\n\n## Prompt\n\nDo it.\n", encoding="utf-8")
    return str(root)


def _score(out_root, tid, score):
    cat = tid.split("_task_")[0]
    d = os.path.join(out_root, "openclaw", cat, tid, "m_20260101_0000_abc123")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "score.json"), "w") as f:
        json.dump(score, f)


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    wd = tmp_path / "wd"
    (wd / RUN.NETWORK_MONITOR_DIR).mkdir(parents=True)
    monkeypatch.setattr(RUN, "WORKDIR", str(wd))
    for k in ("WILDCLAW_TASK", "WILDCLAW_CATEGORIES", "WILDCLAW_LIMIT", "GBENCH_SHARD"):
        monkeypatch.delenv(k, raising=False)
    return wd


def _ok_round(t):
    return {"t": t, "dns": {h: "ok" for h in W.TASK_HOSTS},
            "tcp": {h: "ok" for h in W.EGRESS_HOSTS}}


def test_a_stop_the_agent_cannot_fail_makes_service_failures_harness_errors(launcher,
                                                                            monkeypatch,
                                                                            tmp_path):
    tids = ["04_Search_Retrieval_task_%d" % i for i in (1, 2, 3, 4)]
    monkeypatch.setattr(RUN, "HARNESS_DIR", _harness(tmp_path, tids))
    out_root = str(tmp_path / "out")
    for t in tids:
        _score(out_root, t, {"overall_score": 0.0})
    infra = {"ok": False, "stopped": 0, "left": [], "error": "the stop did not finish",
             "cause": "infra"}
    model = {"ok": False, "stopped": 1, "left": [9], "error": "exit 3, 1 process(es) left",
             "cause": "model"}

    def rec(i, isolation, stop_cause, timed_out=False):
        out = {"task_id": tids[i], "start": 0.0, "end": 400.0, "ips": [f"172.17.0.{i + 2}"],
               "isolation": isolation,
               "grading": [{"start": 300.0, "end": 350.0, "timed_out": timed_out,
                            "isolated": False, "stop_cause": stop_cause}]}
        if timed_out:
            out["grading_error"] = "Command '['docker', 'exec']' timed out after 120 seconds"
        return out
    _jsonl(launcher / RUN.TASK_WINDOWS, [
        rec(0, infra, "infra"),                  # Gemini service failure in the exec: counted
        rec(1, infra, "infra"),                  # the request itself was refused: not counted
        rec(2, model, "model"),                  # the agent can have failed the stop: scored
        rec(3, infra, "infra", timed_out=True)])  # cap hit after an infra stop failure: counted
    _jsonl(launcher / RUN.NETWORK_MONITOR_DIR / RUN.NETWORK_MONITOR,
           [_ok_round(t) for t in range(0, 420, 15)])
    _jsonl(launcher / RUN.HARNESS_EVENTS, [
        {"time": 330.0, "kind": "judge", "client": "172.17.0.2", "start": 310.0,
         "service_failure": True},
        {"time": 331.0, "kind": "judge", "client": "172.17.0.3", "start": 311.0,
         "service_failure": False},
        {"time": 332.0, "kind": "judge", "client": "172.17.0.4", "start": 312.0,
         "service_failure": True}])
    summary = RUN._summarize(out_root, RUN._task_index())
    rows = {r["task_id"]: r for r in summary["per_task"]}
    assert rows[tids[0]]["harness_errors"] == ["grading_infra"]
    assert rows[tids[1]]["harness_errors"] == []
    assert rows[tids[2]]["harness_errors"] == []
    assert rows[tids[3]]["harness_errors"] == ["grading_infra"]
    assert rows[tids[3]]["grading_timeout"] == "not_isolated_infra"
    assert summary["grading_timeouts"]["not_isolated_infra"] == 1
    assert summary["grader_events_in_unisolated_grading"] == 3
    assert summary["grader_events_counted_infra"] == 1
    assert summary["grader_events_not_counted"] == 2
    assert summary["grading_isolation"]["tasks_not_isolated"] == 4
    assert summary["grading_isolation"]["stop_failure_causes"] == {"infra": 3, "model": 1}
    assert rows[tids[2]]["grading_isolation_cause"] == "model"


def test_an_unverified_grading_environment_makes_the_run_completed_with_errors(monkeypatch,
                                                                               tmp_path):
    monkeypatch.setattr(CG, "require_metadata_blocked", lambda *a, **k: CG.UNREACHABLE)
    harness = tmp_path / "harness"
    harness.mkdir()
    wd = tmp_path / "wd"
    wd.mkdir()
    monkeypatch.setattr(W, "check_wildclawbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(W, "_harness_dir", lambda: str(harness))
    monkeypatch.setattr(W, "free_port", lambda preferred=None: preferred)
    monkeypatch.setattr(W.tempfile, "mkdtemp", lambda prefix="": str(wd))
    for k in ("GBENCH_WILDCLAWBENCH_TASK_NETWORK", "GBENCH_WILDCLAWBENCH_TASK_ENDPOINT",
              "GBENCH_WILDCLAWBENCH_JUDGE_PORT", "GBENCH_WILDCLAWBENCH_TASK_HOST", "GBENCH_SHARD",
              "GBENCH_WILDCLAWBENCH_PARALLEL", "GEMINI_OPENAI_BASE_URL", "GBENCH_JUDGE_MODELS",
              "GBENCH_JUDGE_MODEL", "GBENCH_SEARCH_MODELS", "GBENCH_SEARCH_MODEL"):
        monkeypatch.delenv(k, raising=False)
    runs = []
    summary = {"n_tasks": 60, "n_scored": 60, "global_mean": 0.5, "missing_inputs": {},
               "grading_isolation": {"tasks_isolated": 59, "tasks_not_isolated": 1,
                                     "stop_failure_causes": {"model": 1},
                                     "processes_stopped": 3},
               "gemini_models_answered": {"judge": {"gemini-3.8-flash": 40},
                                          "agent_web_search": {"gemini-3.7-flash": 2}}}

    def fake_run(cmd, *a, **k):
        if cmd[:3] == ["docker", "run", "--rm"]:
            runs.append(list(cmd))
            with open(os.path.join(str(wd), "wildclawbench_summary.json"), "w") as f:
                json.dump(summary, f)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(W.subprocess, "run", fake_run)
    result = W.run_wildclawbench("m", "http://127.0.0.1:9123/v1", concurrency=4)
    assert result["status"] == "completed_with_errors"
    assert "(1 model): an unverified grading environment" in result["leaderboard_comparable_reason"]
    # the run's Gemini lists always go in, and the answering models come back
    [cmd] = runs
    assert f"GBENCH_JUDGE_MODELS={','.join(B.GEMINI_CASCADE)}" in cmd
    assert f"GBENCH_SEARCH_MODELS={','.join(B.GEMINI_CASCADE)}" in cmd
    assert result["gemini_models_answered"] == {"gemini-3.7-flash": 2, "gemini-3.8-flash": 40}
    assert result["gemini_cascade"] == list(B.GEMINI_CASCADE)


def test_a_tcp_only_failure_does_not_confirm_a_logged_lookup(launcher, monkeypatch, tmp_path):
    tids = ["01_Productivity_Flow_task_1", "01_Productivity_Flow_task_2"]
    monkeypatch.setattr(RUN, "HARNESS_DIR", _harness(tmp_path, tids))
    out_root = str(tmp_path / "out")
    for t in tids:
        _score(out_root, t, {"overall_score": 0.0})
    tcp_only = _ok_round(150.0)
    tcp_only["tcp"]["registry.npmjs.org"] = "fail timeout"
    degraded = _ok_round(450.0)
    for h in list(degraded["dns"])[: (len(degraded["dns"]) + 1) // 2]:
        degraded["dns"][h] = "fail gaierror:-3"
    rounds = [_ok_round(t) for t in range(0, 600, 15) if t not in (150, 450)]
    _jsonl(launcher / RUN.NETWORK_MONITOR_DIR / RUN.NETWORK_MONITOR,
           rounds + [tcp_only, degraded])
    _jsonl(launcher / RUN.TASK_WINDOWS, [
        {"task_id": tids[0], "start": 100.0, "end": 200.0,
         "name_resolution_errors": {"agent.log": 1}},
        {"task_id": tids[1], "start": 400.0, "end": 500.0,
         "name_resolution_errors": {"agent.log": 1}}])
    summary = RUN._summarize(out_root, RUN._task_index())
    rows = {r["task_id"]: r for r in summary["per_task"]}
    assert rows[tids[0]]["harness_errors"] == []            # scored, named
    assert summary["name_resolution_errors_unconfirmed"] == {tids[0]: {"agent.log": 1}}
    assert rows[tids[1]]["harness_errors"] == ["network"]   # the resolver failed: confirmed
    assert RUN.classify_round(tcp_only)["dns_degraded"] is False
    assert RUN.classify_round(degraded)["dns_degraded"] is True


def test_the_answering_models_are_recorded_per_task(launcher, monkeypatch, tmp_path):
    tids = ["03_Social_Interaction_task_1", "03_Social_Interaction_task_2"]
    monkeypatch.setattr(RUN, "HARNESS_DIR", _harness(tmp_path, tids))
    out_root = str(tmp_path / "out")
    for t in tids:
        _score(out_root, t, {"overall_score": 1.0})
    _jsonl(launcher / RUN.TASK_WINDOWS, [
        {"task_id": tids[0], "start": 0.0, "end": 100.0, "ips": ["172.17.0.2"]},
        {"task_id": tids[1], "start": 0.0, "end": 100.0, "ips": ["172.17.0.3"]}])
    _jsonl(launcher / RUN.NETWORK_MONITOR_DIR / RUN.NETWORK_MONITOR,
           [_ok_round(t) for t in range(0, 120, 15)])
    _jsonl(launcher / RUN.GEMINI_ANSWERS, [
        {"time": 50.0, "start": 49.0, "kind": "judge", "model": "gemini-3.8-flash",
         "client": "172.17.0.2", "grader": True},
        {"time": 60.0, "start": 59.0, "kind": "judge", "model": "gemini-3.7-flash",
         "client": "172.17.0.2", "grader": True},
        {"time": 70.0, "start": 69.0, "kind": "agent_helper", "model": "gemini-3.8-flash",
         "client": "172.17.0.3", "grader": False},
        {"time": 500.0, "start": 499.0, "kind": "judge", "model": "gemini-3.8-flash",
         "client": "172.17.0.9", "grader": True}])
    summary = RUN._summarize(out_root, RUN._task_index())
    rows = {r["task_id"]: r for r in summary["per_task"]}
    assert rows[tids[0]]["gemini_models"] == {"judge": {"gemini-3.7-flash": 1,
                                                        "gemini-3.8-flash": 1}}
    assert rows[tids[1]]["gemini_models"] == {"agent_helper": {"gemini-3.8-flash": 1}}
    assert summary["gemini_models_answered"]["judge"] == {"gemini-3.7-flash": 1,
                                                          "gemini-3.8-flash": 2}
    assert summary["gemini_answers_unattributed"] == 1


def test_the_helper_model_tasks_and_the_audio_judge_are_named():
    assert len(W.HELPER_MODEL_TASKS) == 25 == len(set(W.HELPER_MODEL_TASKS))
    assert W.AUDIO_JUDGE_TASK in W.HELPER_MODEL_TASKS
    comps = W.gemini_substitutions()
    helper = next(c for c in comps if "helper model" in c["component"])
    assert helper["tasks"] == list(W.HELPER_MODEL_TASKS)
    assert any(W.AUDIO_JUDGE_MODEL in c["official"] for c in comps)
    assert not any("pinned to 0.0" in c["served_by"] for c in comps)


def test_the_wildclawbench_guide_follows_the_code():
    doc = _flat(_read("docs/evals/wildclawbench.md"))
    assert f"`org.gbench.wildclawbench-harness-errors={W.HARNESS_ERRORS_VERSION}`" in doc
    assert f'LABEL {W.HARNESS_ERRORS_LABEL}="{W.HARNESS_ERRORS_VERSION}"' in \
        _read("gbench/docker/wildclawbench.Dockerfile")
    for word in ("`not_isolated_infra`", "`grader_events_counted_infra`",
                 "`grading_isolation_cause`", "stop_failure_causes`",
                 "`grader_events_in_unisolated_grading`", "`gemini_models_answered`",
                 "`gemini_answers.jsonl`", "at least half of the names", "`service_failure`"):
        assert word in doc, word
    assert "temperature is pinned to 0.0" not in doc


# =========================================================================== #
# mcp_bench: the credentialed-servers container and the relay to it
# =========================================================================== #
def test_the_credentialed_servers_are_the_ones_whose_commands_list_a_key():
    table = {"Google Maps": {"cmd": "node dist/cli.js --port 3001",
                             "env": ["GOOGLE_MAPS_API_KEY"], "cwd": "../mcp-google-map",
                             "transport": "http", "port": 3001, "endpoint": "/mcp"},
             "BioMCP": {"cmd": "uv run biomcp run", "env": ["NCI_API_KEY"], "cwd": "../biomcp"},
             "Hugging Face": {"cmd": "python s.py", "env": ["HF_TOKEN"], "cwd": "../hf"},
             "National Parks": {"cmd": "node build/index.js", "env": ["NPS_API_KEY"],
                                "cwd": "../nps"},
             "NASA Data": {"cmd": "uv run python main.py", "env": ["NASA_API_KEY"],
                           "cwd": "../nasa"},
             "Math MCP": {"cmd": "node build/index.js", "env": [], "cwd": "../math"}}
    got = CRED.credentialed_servers(table)
    assert sorted(n for n, _s in got.values()) == sorted(MB.CREDENTIALED_SERVERS)
    assert set(got) == set(MB.credentialed_sessions())
    for name, (key, transport) in MB.CREDENTIALED_SERVERS.items():
        assert MRUN.CREDENTIALED_SERVERS[name] == key
        assert CRED.slug(name) == MRUN.slug(name)
        assert (table[name].get("transport") or "stdio") == transport
    assert tuple(MB._SERVER_KEYS) == MRUN.CREDENTIAL_KEYS == CRED.CREDENTIAL_KEYS


def test_a_server_process_gets_only_the_default_variables_and_its_key(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-secret-1")
    monkeypatch.setenv("NASA_API_KEY", "nasa-secret-2")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("SOMETHING_ELSE", "x")
    env = CRED.server_env({"env": ["HF_TOKEN"]}, {"MCP_SERVER_PORT": "4000"})
    assert env["HF_TOKEN"] == "hf-secret-1" and env["MCP_SERVER_PORT"] == "4000"
    assert "NASA_API_KEY" not in env and "SOMETHING_ELSE" not in env
    assert set(env) <= set(CRED.DEFAULT_ENV_VARS) | {"HF_TOKEN", "MCP_SERVER_PORT"}
    assert CRED.server_cwd({"cwd": "../biomcp"}) == os.path.join(CRED.MCPBENCH_DIR,
                                                                 "mcp_servers", "biomcp")
    assert CRED.with_port(["node", "dist/cli.js", "--port", "3001"], 5555) == \
        ["node", "dist/cli.js", "--port", "5555"]
    assert CRED.with_port(["node", "x.js"], 5555) == ["node", "x.js", "--port", "5555"]
    assert CRED.with_port(["node", "--port=1"], 7) == ["node", "--port=7"]


def test_no_key_value_leaves_the_credentialed_container(monkeypatch):
    key = "AIzaSy-A_very+secret/key=="
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", key)
    for k in ("NCI_API_KEY", "HF_TOKEN", "NPS_API_KEY", "NASA_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    forms = CRED.key_forms()
    import urllib.parse
    for form in (key, json.dumps(key)[1:-1], urllib.parse.quote(key, safe=""),
                 urllib.parse.quote_plus(key, safe="")):
        out = CRED.mask(f'{{"url": "https://x/?key={form}"}}'.encode(), forms)
        assert form.encode() not in out and len(out) == len(f'{{"url": "https://x/?key={form}"}}')
    # a key split across the pieces of a stream is still masked, and lengths are kept
    data = b"HTTP/1.1 200 OK\r\nContent-Length: 40\r\n\r\nkey=" + key.encode() + b" and more"
    masker = CRED.StreamMasker(forms)
    out = b"".join(masker.feed(data[i:i + 3]) for i in range(0, len(data), 3)) + masker.flush()
    assert key.encode() not in out and len(out) == len(data)
    assert out.startswith(b"HTTP/1.1 200 OK")


class _EchoPopen:
    """Stands in for `docker exec -i <container> python3 CREDENTIALED_SCRIPT <mode> <arg>`: says
    the session is up, as the credentialed script does, then runs a local echo (stdio / connect),
    or prints a ready line with the port and waits (http)."""
    calls = []

    def __new__(cls, argv, **kw):
        cls.calls.append(list(argv))
        mode = argv[-2]
        if mode == "http":
            code = ("import sys, json; print(json.dumps({'ready': True, 'port': 45678}), "
                    "flush=True); sys.stdin.read()")
        else:
            code = ("import sys, json\nprint(json.dumps({'ready': True}), flush=True)\n"
                    "while True:\n    b = sys.stdin.buffer.read1(65536)\n"
                    "    if not b:\n        break\n    sys.stdout.buffer.write(b)\n"
                    "    sys.stdout.buffer.flush()\n")
        return _REAL_POPEN([sys.executable, "-c", code], **kw)


def _unframe(raw):
    """(the data of the relay's frames in `raw`, the end frame's dict or None)."""
    data, end = b"", None
    while raw:
        head, sep, rest = raw.partition(b"\n")
        if not sep:
            break
        if head.startswith(b"D"):
            n = int(head[1:], 16)
            if len(rest) < n:
                break
            data, raw = data + rest[:n], rest[n:]
        elif head.startswith(b"E"):
            end = json.loads(head[1:])
            break
        else:
            raise AssertionError(f"not a frame: {head!r}")
    return data, end


def _session(address, header, send=b"", read_until=None, timeout=10):
    """Open a relay session. `got` is the http-start ready line as sent, or the data of the
    frames that came back (framed sessions)."""
    host, port = address.rsplit(":", 1)
    s = socket.create_connection((host, int(port)), timeout=timeout)
    s.sendall(json.dumps(header).encode() + b"\n")
    buf = b""
    while not buf.endswith(b"\n"):
        c = s.recv(1)
        if not c:
            break
        buf += c
    answer = buf.decode().strip()
    got = b""
    if answer == "OK" and header.get("op") == "http-start" and read_until:
        while not got.endswith(b"\n"):
            c = s.recv(1)
            if not c:
                break
            got += c
        return s, answer, got
    if answer == "OK" and (send or read_until):
        if send:
            s.sendall(send)
        raw = b""
        deadline = time.time() + timeout
        while time.time() < deadline and (read_until is None
                                          or read_until not in _unframe(raw)[0]):
            c = s.recv(65536)
            if not c:
                break
            raw += c
        got = _unframe(raw)[0]
    return s, answer, got


@pytest.fixture
def relay(monkeypatch):
    _EchoPopen.calls = []
    monkeypatch.setattr(MB.subprocess, "Popen", _EchoPopen)
    r = MB.CredentialedServersRelay("gbench_mcp_bench_x-cred", "127.0.0.1").start()
    yield r
    r.stop()


def test_the_relay_opens_only_known_sessions_with_the_token(relay):
    addr, token = relay.address, relay.token
    for header, why in (({"token": "nope", "op": "stdio", "server": "biomcp"}, "token"),
                        ({"token": token, "op": "shell", "server": "biomcp"}, "op or server"),
                        ({"token": token, "op": "stdio", "server": "math-mcp"}, "op or server"),
                        ({"token": token, "op": "stdio", "server": "google-maps"}, "transport"),
                        ({"token": token, "op": "http-start", "server": "biomcp"}, "transport"),
                        ({"token": token, "op": "http-connect", "server": "google-maps",
                          "port": 1234}, "port")):
        s, answer, _ = _session(addr, header)
        s.close()
        assert answer == f"ERR {why}", header
    assert _EchoPopen.calls == []                        # nothing ran for a refused session
    s, answer, got = _session(addr, {"token": token, "op": "stdio", "server": "nasa-data"},
                              send=b'{"jsonrpc": "2.0"}\n', read_until=b"\n")
    s.close()
    assert answer == "OK" and got == b'{"jsonrpc": "2.0"}\n'
    [argv] = _EchoPopen.calls
    assert argv == ["docker", "exec", "-i", "gbench_mcp_bench_x-cred", "python3",
                    MB.CREDENTIALED_SCRIPT, "stdio", "nasa-data"]
    rec = relay.record()
    assert rec["sessions"] == {"stdio": 1} and rec["refused"]["token"] == 1
    assert token not in json.dumps(rec)


def test_an_http_session_opens_connections_to_its_port_only(relay):
    addr, token = relay.address, relay.token
    ctrl, answer, got = _session(addr, {"token": token, "op": "http-start",
                                        "server": "google-maps"}, read_until=b"\n")
    assert answer == "OK" and json.loads(got.decode()) == {"ready": True, "port": 45678}
    s, answer, got = _session(addr, {"token": token, "op": "http-connect",
                                     "server": "google-maps", "port": 45678},
                              send=b"GET /mcp HTTP/1.1\r\n\r\n", read_until=b"\r\n\r\n")
    s.close()
    assert answer == "OK" and got == b"GET /mcp HTTP/1.1\r\n\r\n"
    assert _EchoPopen.calls[-1][-2:] == ["connect", "45678"]
    ctrl.close()
    deadline = time.time() + 10
    while time.time() < deadline and relay._ports:
        time.sleep(0.05)
    assert relay._ports == {}                            # the port closes with its session
    s, answer, _ = _session(addr, {"token": token, "op": "http-connect", "server": "google-maps",
                                   "port": 45678})
    s.close()
    assert answer == "ERR port"


def test_the_bridge_records_a_session_the_relay_did_not_open(monkeypatch, tmp_path, relay):
    cfg, fails = tmp_path / "cfg.json", tmp_path / "fails.jsonl"
    monkeypatch.setattr(BRIDGE, "BRIDGE_CONFIG", str(cfg))
    monkeypatch.setattr(BRIDGE, "BRIDGE_FAILURES", str(fails))
    cfg.write_text(json.dumps({"relay": relay.address, "token": "wrong"}))
    assert BRIDGE.run_stdio("biomcp") == 1
    [rec] = MRUN.read_bridge_failures(str(fails))
    assert rec["server"] == "biomcp" and rec["op"] == "stdio" and "token" in rec["why"]
    cfg.write_text(json.dumps({"relay": relay.address, "token": relay.token}))
    sock = BRIDGE.open_session("biomcp", "stdio")
    sock.sendall(b"ping\n")
    sock.shutdown(socket.SHUT_WR)
    got = []
    with sock.makefile("rb") as stream:
        how, code, answered = BRIDGE.read_frames(stream, got.append)
    sock.close()
    assert b"".join(got) == b"ping\n" and (how, code, answered) == ("end", 0, True)


def test_the_launcher_points_the_credentialed_entries_at_the_bridge(tmp_path):
    servers = tmp_path / "mcp_servers"
    servers.mkdir()
    table = {"Google Maps": {"cmd": "node dist/cli.js --port 3001",
                             "env": ["GOOGLE_MAPS_API_KEY"], "cwd": "../mcp-google-map",
                             "transport": "http", "port": 3001, "endpoint": "/mcp"},
             "BioMCP": {"cmd": "uv run biomcp run", "env": ["NCI_API_KEY"], "cwd": "../biomcp"},
             "Math MCP": {"cmd": "node build/index.js", "env": [], "cwd": "../math"}}
    (servers / "commands.json").write_text(json.dumps(table))
    assert MRUN.install_credentialed_bridges(str(tmp_path), environ={}) == {}
    old_cfg = MRUN.BRIDGE_CONFIG
    MRUN.BRIDGE_CONFIG = str(tmp_path / "bridge.json")
    try:
        got = MRUN.install_credentialed_bridges(
            str(tmp_path), environ={"MCP_CREDENTIALED_RELAY": "172.17.0.1:5",
                                    "MCP_CREDENTIALED_TOKEN": "t"})
        cfg = json.loads((tmp_path / "bridge.json").read_text())
    finally:
        MRUN.BRIDGE_CONFIG = old_cfg
        MRUN.BRIDGED.clear()
    assert got == {"Google Maps": "google-maps", "BioMCP": "biomcp"}
    assert cfg == {"relay": "172.17.0.1:5", "token": "t"}
    live = json.loads((servers / "commands.json").read_text())
    assert json.loads((servers / "commands.upstream.json").read_text()) == table
    assert live["Math MCP"] == table["Math MCP"]
    assert live["BioMCP"]["env"] == [] and "mcp_bench_bridge.py stdio biomcp" in \
        live["BioMCP"]["cmd"]
    gm = live["Google Maps"]
    assert gm["env"] == [] and gm["transport"] == "http" and gm["port"] == 3001
    assert gm["endpoint"] == "/mcp" and gm["cmd"].endswith("http google-maps --port 3001")


def test_the_credentialed_container_gets_the_keys_by_name_on_its_own_network(monkeypatch):
    for k in MB._SERVER_KEYS:
        monkeypatch.setenv(k, f"value-of-{k}-123456")
    calls, guards = [], []

    def fake_run(cmd, **kw):
        calls.append((list(cmd), kw))
        return subprocess.CompletedProcess(cmd, 0, "id\n", "")
    monkeypatch.setattr(MB.subprocess, "run", fake_run)
    monkeypatch.setattr(MB.container_guard, "require_metadata_blocked",
                        lambda *a, **k: guards.append((a, k)) or CG.UNREACHABLE)
    cred = MB.CredentialedServers("img", "gbench_mcp_bench_t", "host").start()
    try:
        assert [c[0][:3] for c in calls] == [["docker", "network", "create"],
                                             ["docker", "run", "-d"]]
        [(args, kw)] = guards
        assert args[:3] == ("mcp_bench", "img", "gbench_mcp_bench_t-cred-net")
        assert kw["resolve"] == MB.credentialed_dns_names()
        assert {"pypi.org", "api.nasa.gov", "maps.googleapis.com"} <= set(kw["resolve"])
        run_cmd, run_kw = calls[1]
        assert run_cmd[run_cmd.index("--network") + 1] == "gbench_mcp_bench_t-cred-net"
        for k in MB._SERVER_KEYS:
            assert k in run_cmd and f"value-of-{k}-123456" not in " ".join(run_cmd)
            assert run_kw["env"][k] == f"value-of-{k}-123456"
        assert "-p" not in run_cmd and "-v" not in run_cmd
        assert run_cmd[-4:] == ["--entrypoint", "sleep", "img", "infinity"]
        env = cred.client_env()
        assert set(env) == {"MCP_CREDENTIALED_RELAY", "MCP_CREDENTIALED_TOKEN"}
        assert env["MCP_CREDENTIALED_RELAY"].startswith("127.0.0.1:")
        rec = json.dumps(cred.record())
        assert env["MCP_CREDENTIALED_TOKEN"] not in rec and "value-of-" not in rec
    finally:
        cred.stop()
    assert ["docker", "rm", "-f", "gbench_mcp_bench_t-cred"] in [c[0] for c in calls]
    assert ["docker", "network", "rm", "gbench_mcp_bench_t-cred-net"] in [c[0] for c in calls]


def test_the_probe_passes_no_server_key(monkeypatch):
    for k in MB._SERVER_KEYS:
        monkeypatch.setenv(k, f"value-of-{k}-123456")
    seen = []

    def fake_run(cmd, **kw):
        seen.append((list(cmd), kw))
        return subprocess.CompletedProcess(cmd, 0, "MCP_BENCH_PROBE " + json.dumps(_PROBE_OK),
                                           "")
    monkeypatch.setattr(MB.subprocess, "run", fake_run)
    MB.provisioning_probe("img", {"MCP_CREDENTIALED_RELAY": "1.2.3.4:5",
                                  "MCP_CREDENTIALED_TOKEN": "tok"})
    [(cmd, kw)] = seen
    assert not any(str(x).split("=")[0] in MB._SERVER_KEYS for x in cmd)
    assert not any(k in kw["env"] for k in MB._SERVER_KEYS)
    assert "MCP_CREDENTIALED_TOKEN" in cmd and "tok" not in cmd
    assert kw["env"]["MCP_CREDENTIALED_TOKEN"] == "tok"
    assert "api_key" in MB._PROBE and "install_credentialed_bridges" in MB._PROBE
    assert "os.environ[k]" not in MB._PROBE              # the probe writes no key file


def test_the_model_container_entrypoint_refuses_a_server_key(tmp_path):
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "MCPBENCH_DIR": str(tmp_path),
           "MCP_ENDPOINT": "http://x/v1", "MCP_MODEL": "m", "HF_TOKEN": "hf_xxx",
           "GEMINI_API_KEY": "gbench-gemini-relay-" + "0" * 48}
    proc = subprocess.run(["bash", os.path.join(_DOCKER_DIR, "mcp_bench_entrypoint.sh")],
                          capture_output=True, text=True, env=env, timeout=30)
    assert proc.returncode == 2
    assert "HF_TOKEN is set in the model container" in proc.stderr
    assert "hf_xxx" not in proc.stderr + proc.stdout


def test_an_endpoint_url_with_user_info_is_refused():
    assert MB.endpoint_userinfo_refusal("http://127.0.0.1:8000/v1") is None
    assert MB.endpoint_userinfo_refusal("https://api.example.com/v1") is None
    why = MB.endpoint_userinfo_refusal("https://user:tok3n@api.example.com/v1")
    assert why and "user name or password" in why and "tok3n" not in why
    assert MB.endpoint_userinfo_refusal("http://tok3n@10.0.0.1:8000") is not None


# =========================================================================== #
# mcp_bench judge: one attempt per model, 404 skipped, the answering model reported
# =========================================================================== #
def test_the_judge_client_makes_one_attempt_per_request(monkeypatch):
    made = []

    class _AsyncOpenAI:
        def __init__(self, **kw):
            made.append(kw)
    fake = types.ModuleType("openai")
    fake.AsyncOpenAI = _AsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake)
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv(CJ.RELAY_URL_ENV, raising=False)
    CJ.CascadeGeminiJudge()._next_client()
    assert made and all(kw["max_retries"] == 0 for kw in made)


def test_the_judge_skips_a_model_not_served_and_reports_the_answer(monkeypatch):
    class _Err(Exception):
        def __init__(self, status):
            super().__init__(f"HTTP {status}")
            self.status_code = status
    asked = []
    state = {"n": 0}

    class _Completions:
        async def create(self, model, messages, max_tokens, **kw):
            asked.append(model)
            assert "temperature" not in kw                # upstream sends none
            if model == "gone":
                raise _Err(404)
            state["n"] += 1
            if state["n"] < 3:
                raise _Err(503)
            return types.SimpleNamespace(choices=[types.SimpleNamespace(
                message=types.SimpleNamespace(content='{"score": 7}'))], usage=None)
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))
    answered = []
    monkeypatch.setattr(CJ.CascadeGeminiJudge, "answer_hook", answered.append)
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gone,live")
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", "3")
    monkeypatch.setenv("GBENCH_JUDGE_BACKOFF", "0")

    async def _no_sleep(s):
        return None
    monkeypatch.setattr(CJ.asyncio, "sleep", _no_sleep)
    judge = CJ.CascadeGeminiJudge()
    judge._async_clients = [client]
    judge._client_cycle = itertools.cycle([client])
    assert asyncio.run(judge.get_completion("s", "u", 10)) == '{"score": 7}'
    assert asked == ["gone", "live", "live", "live"]      # the 404 model is asked once
    assert answered == ["live"]


def test_the_harness_records_the_answering_models_per_task(tmp_path):
    h = MRUN.HarnessAccounting(["api.nasa.gov"], resolvers=["10.0.0.2"],
                               bridge_failures=str(tmp_path / "none.jsonl"))
    rec = h.begin("t1", "/x/single.json")
    h.judge_answered("gemini-3.8-flash")
    h.judge_answered("gemini-3.8-flash")
    h.judge_answered("gemini-3.7-flash")
    h.end(rec, {"status": "completed", "evaluation": {"x": 1}})
    log = h.record()
    assert log["version"] == MB.HARNESS_LOG_VERSION == MRUN.HARNESS_LOG_VERSION
    assert log["tasks"][0]["judge_models"] == {"gemini-3.7-flash": 1, "gemini-3.8-flash": 2}
    acct = MB.harness_accounting(log)
    assert acct["judge_models"] == {"gemini-3.7-flash": 1, "gemini-3.8-flash": 2}
    assert acct["judge_models_per_task"] == {"t1": {"gemini-3.7-flash": 1,
                                                    "gemini-3.8-flash": 2}}


def test_a_bridge_failure_in_a_tasks_window_is_a_harness_error(tmp_path):
    fails = tmp_path / "fails.jsonl"
    h = MRUN.HarnessAccounting([], resolvers=[], bridge_failures=str(fails))
    rec = h.begin("t1", "/x/single.json")
    fails.write_text(json.dumps({"t": time.time(), "server": "biomcp", "op": "stdio",
                                 "why": "the relay did not answer"}) + "\n")
    h.end(rec, {"status": "completed", "evaluation": {"x": 1}})
    assert rec["kinds"] == ["credentialed_relay"]
    rec2 = h.begin("t2", "/x/single.json")
    h.end(rec2, {"status": "completed", "evaluation": {"x": 1}})
    assert rec2["kinds"] == []                            # an earlier failure is not t2's


# =========================================================================== #
# mcp_bench harness heuristics: per-call endpoint attribution, error text only
# =========================================================================== #
def test_the_endpoint_flag_is_decided_by_the_calls_last_attempt(monkeypatch, tmp_path):
    class _EndpointDown(Exception):
        pass

    class _Completions:
        def __init__(self, script):
            self.script = list(script)

        async def create(self, **kw):
            step = self.script.pop(0)
            if isinstance(step, Exception):
                raise step
            return types.SimpleNamespace(choices=[types.SimpleNamespace(
                finish_reason="stop", message=types.SimpleNamespace(content=step,
                                                                    tool_calls=None))])

    class _Provider:
        def __init__(self, script):
            self.client = types.SimpleNamespace(
                api_key="dummy", chat=types.SimpleNamespace(completions=_Completions(script)))

        async def get_completion(self, system_prompt, user_prompt, max_tokens):
            # upstream's 3 attempts: an empty reply or an error moves to the next attempt
            last = None
            for _attempt in range(3):
                try:
                    r = await self.client.chat.completions.create(
                        model="m", messages=[], max_tokens=max_tokens)
                    content = r.choices[0].message.content
                    if not content:
                        raise ValueError("Empty content received from LLM")
                    return content
                except Exception as e:  # noqa: BLE001
                    last = e
            raise last

    scripts = []

    class _Factory:
        @staticmethod
        async def create_llm_provider(model_config):
            return _Provider(scripts.pop(0))
    llm = types.ModuleType("llm")
    factory = types.ModuleType("llm.factory")
    factory.LLMFactory = _Factory
    monkeypatch.setitem(sys.modules, "llm", llm)
    monkeypatch.setitem(sys.modules, "llm.factory", factory)
    monkeypatch.setenv("MCP_ENABLE_THINKING", "false")
    monkeypatch.setenv("MCP_REQUEST_LOG", str(tmp_path / "requests.jsonl"))
    monkeypatch.setenv("MCP_MODEL_CONFIG_NAME", "slot")
    monkeypatch.delenv("MCP_MAX_TOKENS", raising=False)
    h = MRUN.HarnessAccounting([], resolvers=[], bridge_failures=str(tmp_path / "none"))
    policy = MRUN.install_request_policy(h)
    cfg = types.SimpleNamespace(name="slot")

    async def run(script):
        scripts.append(script)
        provider = await _Factory.create_llm_provider(cfg)
        rec = h.begin("t", "/x/f.json")
        try:
            await provider.get_completion("s", "u", 10)
        except Exception:  # noqa: BLE001
            pass
        h.end(rec, {"status": "completed", "evaluation": {"x": 1}})
        return rec["kinds"]
    try:
        # an endpoint error, then two empty replies: the model's outcome
        assert asyncio.run(run([_EndpointDown("503"), "", ""])) == []
        # the last attempt failed on the endpoint: a harness error
        assert asyncio.run(run(["", "", _EndpointDown("503")])) == ["model_endpoint"]
        # an endpoint error that a later attempt got past: nothing
        assert asyncio.run(run([_EndpointDown("503"), "fine"])) == []
    finally:
        policy.dump()
    assert policy.counts["failed_requests"] == 3


def test_resolver_text_counts_only_in_error_text_naming_a_server_host_or_the_resolver(tmp_path):
    h = MRUN.HarnessAccounting(["api.nasa.gov"], resolvers=["10.0.0.2"],
                               bridge_failures=str(tmp_path / "none"))

    def kinds(result):
        rec = h.begin("t", "/x/f.json")
        h.end(rec, result)
        return rec["kinds"]
    ok = {"status": "completed", "evaluation": {"x": 1}}
    # a fetched page that quotes the error text: a successful tool result, not scanned
    assert kinds(dict(ok, execution_results=[
        {"result": "Temporary failure in name resolution api.nasa.gov", "success": True}])) == []
    # a model-chosen name whose lookup failed (SERVFAIL reads as EAI_AGAIN)
    assert kinds(dict(ok, execution_results=[
        {"error": "getaddrinfo EAI_AGAIN my-typo.example"}])) == []
    # dig at a server the model chose
    assert kinds(dict(ok, execution_results=[
        {"error": ";; connection timed out; no servers could be reached (8.8.4.4)"}])) == []
    assert kinds(dict(ok, execution_results=[
        {"error": "getaddrinfo EAI_AGAIN api.nasa.gov"}])) == ["dns_failure"]
    assert kinds(dict(ok, execution_results=[
        {"error": ";; connection timed out; no servers could be reached (10.0.0.2#53)"}])) == \
        ["dns_failure"]


def test_the_judge_outage_reason_matches_upstreams_aggregator():
    kind = MB.HARNESS_ERROR_KINDS["judge_outage"]
    assert "skip" in kind and "stop updating" not in kind
    src = _read("gbench/docker/mcp_bench_run.py")
    assert "would then stop updating the metrics" not in src
    doc = _flat(_read("docs/evals/mcp_bench.md"))
    assert "stop updating the metrics" not in doc
    assert "upstream's aggregator skips such a result" in doc


# =========================================================================== #
# mcp_bench run: a partial judge outage sets the status
# =========================================================================== #
_PROBE_OK = {"servers": [], "unavailable": [], "unbuilt": [], "network_up": True,
             "connections": {f"S{i}": {"status": "success", "tools": 3, "error": ""}
                             for i in range(28)}}


def _run_mcp_bench(monkeypatch, harness_tasks):
    for k in ("GBENCH_MCP_BENCH_NETWORK", "GBENCH_MCP_BENCH_DISTRACTION", "GBENCH_SHARD",
              "GBENCH_MCP_BENCH_DISABLE_STABILITY", "GBENCH_MCP_BENCH_ASSUME_NETWORK",
              "GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL"):
        monkeypatch.delenv(k, raising=False)
    # Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "TESTKEY0000000000000000000000000000")
    for k in MB._REQUIRED_KEYS:
        monkeypatch.setenv(k, "server-key-value-1234")
    monkeypatch.setattr(MB, "check_mcp_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(MB, "provisioning_probe", lambda image, relay_env=None: _PROBE_OK)
    monkeypatch.setattr(MB.container_guard, "require_metadata_blocked",
                        lambda *a, **k: CG.UNREACHABLE)
    relay = types.SimpleNamespace(
        client_env=lambda: {"GEMINI_API_KEY": "gbench-gemini-relay-" + "0" * 48,
                            "GEMINI_API_KEYS": "gbench-gemini-relay-" + "0" * 48,
                            "GEMINI_OPENAI_BASE_URL": "http://172.17.0.1:9/v1beta/openai",
                            "GBENCH_GEMINI_RELAY_URL": "http://172.17.0.1:9"},
        stop=lambda: None, record=lambda: {})
    monkeypatch.setattr(MB, "_start_judge_relay", lambda network: relay)
    cred = types.SimpleNamespace(
        client_env=lambda: {"MCP_CREDENTIALED_RELAY": "172.17.0.1:10",
                            "MCP_CREDENTIALED_TOKEN": "t0k"},
        stop=lambda: None, record=lambda: {"container": "c"})
    monkeypatch.setattr(MB, "start_credentialed_servers", lambda image, base, network: cred)
    runs = []

    def fake_run(cmd, **kw):
        if cmd[:3] == ["docker", "network", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0, "[]", "")
        if cmd[:2] == ["docker", "run"]:
            runs.append((list(cmd), kw))
            out = cmd[cmd.index("-v") + 1].split(":")[0]
            metrics = {"input_schema_compliance": 1.0, "valid_tool_name_rate": 1.0,
                       "tool_call_success_rate": 1.0, "task_completion_score": 5.0,
                       "tool_selection_score": 5.0,
                       "planning_effectiveness_and_efficiency_score": 5.0}
            with open(os.path.join(out, "mcp_bench_results.json"), "w") as fh:
                json.dump(metrics, fh)
            with open(os.path.join(out, "subset_manifest.json"), "w") as fh:
                json.dump({"filtered": True, "unavailable_servers": [], "tasks_kept": 104}, fh)
            with open(os.path.join(out, MB.HARNESS_LOG), "w") as fh:
                json.dump({"version": MB.HARNESS_LOG_VERSION, "tasks": harness_tasks}, fh)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(MB.subprocess, "run", fake_run)
    return MB.run_mcp_bench("m", "http://127.0.0.1:8123/v1"), runs


def test_a_partial_judge_outage_makes_the_run_completed_with_errors(monkeypatch):
    res, runs = _run_mcp_bench(monkeypatch, [
        {"task_id": "t1", "file": "f.json", "status": "completed", "kinds": [],
         "judge_calls_failed": 2, "judge_models": {"gemini-3.8-flash": 3}},
        {"task_id": "t2", "file": "f.json", "status": "completed", "kinds": [],
         "judge_calls_failed": 0, "judge_models": {"gemini-3.7-flash": 5}}])
    assert res["status"] == "completed_with_errors"
    [entry] = [e for e in res["harness_errors"] if e.get("kind") == "judge_outage_partial"]
    assert entry["judge_calls_failed"] == 2 and entry["task_ids"] == ["t1"]
    assert res["judge_calls_failed_tasks"] == ["t1"]
    assert res["gemini_models_answered"] == {"gemini-3.7-flash": 5, "gemini-3.8-flash": 3}
    assert res["judge_models_per_task"] == {"t1": {"gemini-3.8-flash": 3},
                                            "t2": {"gemini-3.7-flash": 5}}
    assert res["gemini_cascade"] == list(B.GEMINI_CASCADE)
    assert res["credentialed_servers"] == {"container": "c"}
    [(cmd, kw)] = runs
    assert f"GBENCH_JUDGE_MODELS={','.join(B.GEMINI_CASCADE)}" in cmd
    assert not any(k in kw["env"] for k in MB._SERVER_KEYS)


def test_a_run_without_judge_failures_is_a_success(monkeypatch):
    res, _runs = _run_mcp_bench(monkeypatch, [
        {"task_id": "t1", "file": "f.json", "status": "completed", "kinds": [],
         "judge_calls_failed": 0, "judge_models": {"gemini-3.8-flash": 30}}])
    assert res["status"] == "success" and res["harness_errors"] == []


def test_a_user_info_endpoint_starts_nothing(monkeypatch):
    monkeypatch.setattr(MB, "check_mcp_bench_prerequisites", lambda: (True, ""))
    started = []
    monkeypatch.setattr(MB, "start_credentialed_servers", lambda *a: started.append(a))
    monkeypatch.setattr(MB.container_guard, "require_metadata_blocked",
                        lambda *a, **k: started.append("guard"))
    with pytest.raises(RuntimeError, match="user name or password"):
        MB.run_mcp_bench("m", "https://u:secret@api.example.com/v1")
    assert started == []


def test_the_mcp_bench_label_and_docs_follow_the_code():
    df = _read("gbench/docker/mcp_bench.Dockerfile")
    assert f'LABEL {MB.HARNESS_LABEL}="{MB.HARNESS_VERSION}"' in df
    assert "COPY mcp_bench_bridge.py" in df and "COPY mcp_bench_credentialed.py" in df
    doc = _flat(_read("docs/evals/mcp_bench.md"))
    for word in (f"`{MB.HARNESS_LABEL}={MB.HARNESS_VERSION}`", "credentialed-servers container",
                 "`credentialed_relay`", "`judge_outage_partial`", "`max_retries=0`",
                 "`judge_models_per_task`", "`gemini_models_answered`", "user name or password",
                 "`credentialed_servers`"):
        assert word in doc, word
    assert "which the image's entrypoint writes from the variables gbench passes" not in doc


# =========================================================================== #
# mcp_atlas: the guard resolves the servers' names; an external environment holds no key
# =========================================================================== #
def test_both_mcp_atlas_networks_are_checked_with_their_host_names(monkeypatch):
    guards = []
    monkeypatch.setattr(M.container_guard, "require_metadata_blocked",
                        lambda *a, **k: guards.append((a[2], tuple(k.get("resolve") or ())))
                        or CG.UNREACHABLE)
    monkeypatch.setattr(M, "_require_credentials", lambda: None)
    monkeypatch.setattr(M, "_free_port", lambda: 4555)
    monkeypatch.setattr(M, "_create_network", lambda name: name)
    monkeypatch.setattr(M, "_local_mongo_wanted", lambda: False)
    monkeypatch.setattr(M, "_run_credentialed", lambda image, base, network, mongo: "cred")
    monkeypatch.setattr(M.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "id", ""))
    sb = M._start_container("img")
    assert guards == [(sb.network, M.ENVIRONMENT_HOSTS),
                      (sb.cred_network, M.CREDENTIALED_HOSTS)]
    assert set(M.PACKAGE_HOSTS) <= set(M.ENVIRONMENT_HOSTS)
    assert set(M.PACKAGE_HOSTS) <= set(M.CREDENTIALED_HOSTS)
    for host in ("api.notion.com", "slack.com", "api.github.com", "maps.googleapis.com"):
        assert host in M.CREDENTIALED_HOSTS
    for host in ("en.wikipedia.org", "export.arxiv.org", "httpbin.org"):
        assert host in M.ENVIRONMENT_HOSTS


def test_the_credential_probe_prints_names_never_values(tmp_path):
    script = M.credential_probe_script()
    env = {"PATH": os.environ.get("PATH", "/usr/bin"), "NOTION_TOKEN": "ntn-SECRET-value",
           "GEMINI_API_KEY": ""}
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, timeout=60)
    found = M._CREDENTIAL_VERDICT.search(proc.stdout)
    # other processes of this user on the host may hold such variables too: the probe reads
    # every process environment it can, so only this one's name is certain to be listed
    assert found and "NOTION_TOKEN" in found.group(1).split(",")
    assert int(found.group(3)) >= 1
    # the value, never a name: a name such as LARA_ACCESS_KEY_SECRET may be listed when another
    # process of this user holds it
    assert "ntn-SECRET-value" not in proc.stdout
    assert "GEMINI_API_KEY" in M.EXTERNAL_CREDENTIAL_NAMES
    assert set(v for vs in M.RELEASE_SERVERS.values() for v in vs) <= \
        set(M.EXTERNAL_CREDENTIAL_NAMES)


def _external(replies):
    """A stand-in external environment whose /call-tool answers the next of `replies`."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    calls = []
    queue = list(replies)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            calls.append(json.loads(self.rfile.read(n) or b"{}"))
            status, text = queue.pop(0)
            data = text.encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}", calls


def _code_reply(stdout):
    return 200, json.dumps([{"type": "text", "text": json.dumps(
        {"status": "success", "output": stdout})}])


@pytest.mark.parametrize("replies, ok, needle", [
    ([_code_reply("GBENCH_CREDENTIAL_PROBE=none proc1:read procs:7\n")], True,
     "7 process environment(s) read"),
    ([_code_reply("GBENCH_CREDENTIAL_PROBE=GITHUB_TOKEN,NOTION_TOKEN proc1:read procs:7\n")],
     False, "holds credential variables (GITHUB_TOKEN, NOTION_TOKEN"),
    ([(500, "boom"), (200, "no verdict here")], False, "gave no verdict"),
])
def test_an_external_environment_is_checked_for_credentials(replies, ok, needle):
    srv, url, calls = _external(replies)
    try:
        sb = M._Sandbox(url)
        if ok:
            assert needle in M.probe_external_credentials(sb)
        else:
            with pytest.raises(RuntimeError) as ei:
                M.probe_external_credentials(sb)
            assert needle in str(ei.value)
    finally:
        srv.shutdown()
        srv.server_close()
    assert calls[0]["tool_name"] == "mcp-code-executor_execute_code"
    assert calls[0]["use_cache"] is False
    assert calls[0]["tool_args"] == {"code": M.credential_probe_script()}


def test_the_mcp_atlas_guide_follows_the_code():
    doc = _flat(_read("docs/evals/mcp_atlas.md"))
    assert "`agent_environment.external_credential_check`" in doc
    assert "`agent_environment.dns_names_checked`" in doc
    assert "runs a local shell, code or file tool" in doc
    assert "fetched unpinned from `github.com/geobio`" in doc
    for host in M.CREDENTIALED_HOSTS + M.ENVIRONMENT_HOSTS:
        if host not in M.PACKAGE_HOSTS:
            assert f"`{host}`" in doc, host


# =========================================================================== #
# Plain ASCII, no internal labels
# =========================================================================== #
@pytest.mark.parametrize("rel", [
    "gbench/runners/eval_suites/mcp_bench.py", "gbench/runners/eval_suites/mcp_atlas.py",
    "gbench/runners/eval_suites/wildclawbench.py", "gbench/docker/mcp_bench_run.py",
    "gbench/docker/mcp_bench_bridge.py", "gbench/docker/mcp_bench_credentialed.py",
    "gbench/docker/mcp_bench_cascade_judge.py", "gbench/docker/mcp_bench_entrypoint.sh",
    "gbench/docker/mcp_bench.Dockerfile", "gbench/docker/mcp_bench_request_policy.py",
    "gbench/docker/wildclawbench_run.py", "gbench/docker/wildclawbench_cascade_judge.py",
    "gbench/docker/wildclawbench.Dockerfile", "docs/evals/mcp_bench.md",
    "docs/evals/mcp_atlas.md", "docs/evals/wildclawbench.md",
    "tests/test_mcp_wildclawbench_credentials_and_cascade.py"])
def test_the_changed_files_are_plain_ascii_without_internal_labels(rel):
    text = _read(rel)
    assert all(ord(c) < 128 for c in text), rel
    low = text.lower()
    # built from parts, so this file does not hold them
    for label in ("wave" + " 2", "wave" + "2", "the " + "reviewer", "skep" + "tic"):
        assert label not in low, (rel, label)
