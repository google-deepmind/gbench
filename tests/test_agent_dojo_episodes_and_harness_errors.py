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

"""agent_dojo runs AgentDojo one episode at a time and never scores a harness failure.

* An episode an endpoint failure (or any exception) ends is a harness error: left out of benign
  utility, utility under attack and attack success, counted by kind, named; the run is
  completed_with_errors, and a run with no scored benign episode withholds its score.
* An episode that outgrew the context window is scored as AgentDojo scores it, and counted.
* The endpoint's replies are counted: stopped at the end of the context window, empty, empty
  after reasoning.
* Every episode (benign, under attack, injection task as a user task) is a trace with the
  messages AgentDojo logged; the log directory is removed when the run ends.
* The result names only the suites that ran, reports its co-headlines, and a model-id override
  is checked against the endpoint.

No network, no model server: AgentDojo and the endpoint are faked.
"""

import json
import os
import types
from pathlib import Path

import pytest

from gbench.runners.eval_suites import agent_dojo as AD


class _Task:
    def __init__(self, task_id):
        self.ID = task_id


class _Suite:
    def __init__(self, name, users, injs):
        self.name = name
        self.user_tasks = {u: _Task(u) for u in users}
        self.injection_tasks = {j: _Task(j) for j in injs}

    def get_user_task_by_id(self, t):
        return self.user_tasks[t]

    def get_injection_task_by_id(self, t):
        return self.injection_tasks[t]


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class APIConnectionError(Exception):
    """Named as openai's, so the kind reads model_endpoint_unreachable."""


def _log(logdir, suite, task, attack, inj, text):
    path = Path(logdir) / "vllm_parsed" / suite / task / attack / f"{inj or 'none'}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"messages": [{"role": "assistant", "content": text}],
                                "error": None}))


def _fake(fail=(), client_calls=None, logdirs=None):
    """AgentDojo with suites banking (u0, u1 x i0) and slack (s0 x i0). `fail` holds the
    (task, injection) episodes that raise an endpoint error."""
    suites = {"banking": _Suite("banking", ["u0", "u1"], ["i0"]),
              "slack": _Suite("slack", ["s0"], ["i0"])}

    def run_without(suite, pipe, task, ld, force, version=None):
        if logdirs is not None:
            logdirs.append(ld)
        if (task.ID, "") in fail:
            raise APIConnectionError("connection refused")
        _log(ld, suite.name, task.ID, "none", None, f"done {task.ID}")
        return task.ID != "u1", True

    def run_with(suite, pipe, task, attack, ld, force, injs=None, version=None):
        (inj,) = injs
        if (task.ID, inj) in fail:
            raise APIConnectionError("connection refused")
        _log(ld, suite.name, task.ID, "important_instructions", inj, f"attacked {task.ID}")
        return {(task.ID, inj): True}, {(task.ID, inj): task.ID == "u0"}

    return {
        "version": AD.AGENTDOJO_VERSION,
        "openai": types.SimpleNamespace(OpenAI=lambda **k: object()),
        "OpenAILLM": lambda *a, **k: object(),
        "AgentPipeline": lambda els: types.SimpleNamespace(name=None),
        "SystemMessage": lambda *a: None, "InitQuery": lambda: None,
        "ToolsExecutionLoop": lambda *a: None, "ToolsExecutor": lambda: None,
        "load_system_message": lambda *a: "sys",
        "load_attack": lambda *a: types.SimpleNamespace(name="important_instructions",
                                                        is_dos_attack=False),
        "OutputLogger": lambda *a: _Ctx(), "get_suites": lambda v: suites,
        "run_without": run_without, "run_with": run_with}


def test_an_endpoint_failure_ends_one_episode_as_a_harness_error(monkeypatch):
    monkeypatch.setattr(AD, "_load_agentdojo", lambda: _fake(fail={("u1", "")}))
    r = AD.run_agent_dojo("m", "http://h/v1", concurrency=1)
    # u1's benign episode is left out: benign utility over u0 and s0
    assert r["total_questions"] == 2 and r["accuracy"] == 100.0
    assert r["harness_errors"] == {"model_endpoint_unreachable": 1}
    assert r["status"] == "completed_with_errors"
    assert "1 episode(s) ended by a harness error" in r["status_reason"]
    assert r["leaderboard_comparable"] is False
    assert "harness error" in r["leaderboard_comparable_reason"]
    # its attacked episode still ran and counts
    assert r["utility_under_attack"] == 100.0 and r["attack_success_rate"] == pytest.approx(33.33)
    lost = [t for t in r["sample_traces"] if t["status"] == "HARNESS_ERROR"]
    assert len(lost) == 1 and lost[0]["scoring_excluded"] is True
    assert "connection refused" in lost[0]["harness_error"]["detail"]


def test_a_run_with_no_scored_benign_episode_withholds_its_score(monkeypatch):
    fail = {("u0", ""), ("u1", ""), ("s0", "")}
    monkeypatch.setattr(AD, "_load_agentdojo", lambda: _fake(fail=fail))
    r = AD.run_agent_dojo("m", "http://h/v1", concurrency=1)
    assert r["accuracy"] is None and r["status"] == "failed"
    assert r["accuracy_withheld_reason"] == "every benign episode ended by a harness error"
    assert "nothing was scored" in r["error"]


def test_every_episode_is_a_trace_with_its_log_and_the_log_dir_is_removed(monkeypatch):
    logdirs = []
    monkeypatch.setattr(AD, "_load_agentdojo", lambda: _fake(logdirs=logdirs))
    r = AD.run_agent_dojo("m", "http://h/v1", concurrency=1)
    kinds = sorted(t["kind"] for t in r["sample_traces"])
    # 3 benign, 3 attacked, 2 injection tasks run as user tasks (one per suite)
    assert kinds.count("benign") == 3 and kinds.count("under_attack") == 3
    assert kinds.count("injection_task_as_user") == 2
    benign = next(t for t in r["sample_traces"] if t["sample_key"] == "task_id:banking/u0/")
    assert benign["messages"][0]["content"] == "done u0"
    attacked = next(t for t in r["sample_traces"] if t["kind"] == "under_attack"
                    and t["sample_key"].endswith("banking/u0/i0"))
    assert attacked["attack_succeeded"] is True and attacked["messages"][0]["content"] == \
        "attacked u0"
    assert logdirs and not os.path.exists(logdirs[0])
    assert r["injection_tasks_as_user_tasks"] == {"banking/i0": True, "slack/i0": True}


def test_the_result_names_the_suites_run_and_its_co_headlines(monkeypatch):
    monkeypatch.setattr(AD, "_load_agentdojo", lambda: _fake())
    r = AD.run_agent_dojo("m", "http://h/v1", limit=1, concurrency=1)
    # --eval-limit 1 is one user task for the run, in the suite the seeded deal gives it to
    (suite,) = r["suites"]
    assert r["suites_available"] == ["banking", "slack"]
    ((picked, task),) = AD.select_user_tasks(
        [("banking", "u0"), ("banking", "u1"), ("slack", "s0")], 1)
    assert picked == suite and task != "u1"       # the fake fails u1's benign episode
    assert f"across the suites run ({suite})" in r["metric"]
    assert r["category_accuracy"] == {suite: {"correct": 1, "total": 1, "accuracy": 100.0}}
    # the fake's attack succeeds on u0 only
    assert r["headline_metrics"] == {"benign_utility": 100.0, "utility_under_attack": 100.0,
                                     "attack_success_rate": 100.0 if task == "u0" else 0.0}


class _Choice:
    def __init__(self, finish, content, tool_calls=None, reasoning=None):
        self.finish_reason = finish
        self.message = types.SimpleNamespace(content=content, tool_calls=tool_calls,
                                             reasoning_content=reasoning)


class BadRequestError(Exception):
    code = "context_length_exceeded"
    param = None


def test_the_endpoint_watch_counts_replies_and_context_length_errors():
    replies = [_Choice("length", "partial"), _Choice("stop", "", reasoning="thinking"),
               _Choice("stop", None, tool_calls=[{"id": 1}]), BadRequestError("too long")]

    def create(**kw):
        nxt = replies.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return types.SimpleNamespace(choices=[nxt])
    client = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=create)))
    watch = AD._EndpointWatch()
    watch.install(client)

    def episode():
        for _ in range(3):
            client.chat.completions.create(model="m")
        try:
            client.chat.completions.create(model="m")
        except BadRequestError:
            return False, True              # AgentDojo's own rule for a context-length error
    rec = watch.run(episode)
    assert rec["outcome"] == (False, True) and rec["requests"] == 4
    assert rec["replies_at_context_window"] == 1
    assert rec["empty_responses"] == 1 and rec["empty_after_reasoning"] == 1
    assert rec["context_length_error"] is True


def test_context_length_episodes_are_scored_as_agentdojo_does_and_named(monkeypatch):
    fake = _fake()
    inner = fake["run_without"]

    def run_without(suite, pipe, task, ld, force, version=None):
        if task.ID == "s0":
            AD_watch = run_without.watch
            AD_watch.episode["context_length_error"] = True
            return False, True
        return inner(suite, pipe, task, ld, force, version)
    fake["run_without"] = run_without
    real_install = AD._EndpointWatch.install

    def install(self, client):
        run_without.watch = self
        return real_install(self, client)
    monkeypatch.setattr(AD._EndpointWatch, "install", install)
    monkeypatch.setattr(AD, "_load_agentdojo", lambda: fake)
    r = AD.run_agent_dojo("m", "http://h/v1", concurrency=1)
    assert r["context_length_exceeded"] == 1 and r["status"] == "success"
    assert r["item_results"]["benign"]["slack/s0/"] is False
    assert "1 episode(s) outgrew the served context window" in r["leaderboard_comparable_reason"]


def test_a_model_id_override_is_checked_against_the_endpoint(monkeypatch):
    from gbench.utils import endpoint as EP
    monkeypatch.setattr(AD, "_load_agentdojo", lambda: _fake())
    monkeypatch.setenv("GBENCH_AGENTDOJO_MODEL", "other")

    def refuse(url, candidates, override=None, **kw):
        raise EP.EndpointError("serves only m")
    monkeypatch.setattr(EP, "resolve_served_model", refuse)
    with pytest.raises(RuntimeError, match=r"GBENCH_AGENTDOJO_MODEL='other' is not a model"):
        AD.run_agent_dojo("m", "http://h/v1", concurrency=1)
    monkeypatch.setattr(EP, "resolve_served_model",
                        lambda url, c, o=None, **kw: EP.ServedModel("other", "listed", None))
    r = AD.run_agent_dojo("m", "http://h/v1", concurrency=1)
    assert r["request_model_id"] == "other"
    assert r["request_model_id_source"] == "GBENCH_AGENTDOJO_MODEL (listed)"


def test_the_merge_carries_harness_errors_and_traces(monkeypatch):
    results = []
    for i in (1, 2):
        monkeypatch.setattr(AD, "_load_agentdojo", lambda: _fake(fail={("u1", "")}))
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/2")
        results.append(AD.run_agent_dojo("m", "http://h/v1", concurrency=1))
    monkeypatch.delenv("GBENCH_SHARD")
    monkeypatch.setattr(AD, "_load_agentdojo", lambda: _fake(fail={("u1", "")}))
    full = AD.run_agent_dojo("m", "http://h/v1", concurrency=1)
    merged = AD.merge_shards(results)
    for k in ("accuracy", "utility_under_attack", "attack_success_rate", "total_questions",
              "harness_errors", "status", "category_accuracy", "suites"):
        assert merged[k] == full[k], k
    assert len(merged["sample_traces"]) == len(full["sample_traces"])
