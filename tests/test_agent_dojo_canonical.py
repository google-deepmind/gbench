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

"""Canonical AgentDojo: delegation to the agentdojo package (utility + attack-success-rate)."""
import warnings

warnings.filterwarnings("ignore")

import pytest
from gbench.runners.eval_suites import agent_dojo as AD


def test_agent_dojo_pct():
    assert AD._pct([True, True]) == 100.0
    assert AD._pct([True, False]) == 50.0
    assert AD._pct([]) == 0.0


class _Task:
    def __init__(self, task_id, index):
        self.ID, self.index = task_id, index


class _Suite:
    def __init__(self, users, injs):
        self.user_tasks = {u: _Task(u, k) for k, u in enumerate(users)}
        self.injection_tasks = {j: _Task(j, k) for k, j in enumerate(injs)}

    def get_user_task_by_id(self, task_id):
        return self.user_tasks[task_id]

    def get_injection_task_by_id(self, task_id):
        return self.injection_tasks[task_id]


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Pipe:
    def __init__(self, _elements):
        self.name = None


def _fake_agentdojo():
    """A stand-in for the agentdojo package: 2 user tasks, both benign-pass; the attack succeeds
    on the first user task and fails on the second."""
    def fake_without(suite, pipe, task, ld, force, benchmark_version=None):
        return True, True

    def fake_with(suite, pipe, task, attack, ld, force, injection_tasks=None,
                  benchmark_version=None):
        util, sec = {}, {}
        for j in injection_tasks:
            util[(task.ID, j)] = (task.index != 0)  # task breaks under attack on the first user task
            sec[(task.ID, j)] = (task.index == 0)   # attack SUCCEEDS on the first user task
        return util, sec

    return {
        "openai": type("O", (), {"OpenAI": staticmethod(lambda **k: object())}),
        "OpenAILLM": lambda *a, **k: object(),
        "AgentPipeline": _Pipe,
        "SystemMessage": lambda *a, **k: object(),
        "InitQuery": lambda *a, **k: object(),
        "ToolsExecutionLoop": lambda *a, **k: object(),
        "ToolsExecutor": lambda *a, **k: object(),
        "load_system_message": lambda *a, **k: "sys",
        "get_suites": lambda v: {"banking": _Suite(["user_task_0", "user_task_1"],
                                                   ["injection_task_0"])},
        "load_attack": lambda *a, **k: object(),
        "OutputLogger": lambda *a, **k: _Ctx(),
        "run_without": fake_without,
        "run_with": fake_with,
    }


def test_agent_dojo_delegates_utility_and_asr(monkeypatch):
    """Headline accuracy = benign utility; attack_success_rate = mean(security_results) where
    True means the attack SUCCEEDED (NOT inverted into accuracy)."""
    monkeypatch.setattr(AD, "_load_agentdojo", _fake_agentdojo)
    r = AD.run_agent_dojo("m", "http://x", limit=None, concurrency=1)
    assert r["eval_name"] == "agent_dojo" and r["status"] == "success"
    assert r["accuracy"] == 100.0                 # both benign tasks pass
    assert r["attack_success_rate"] == 50.0       # 1 of 2 injections succeeded
    assert r["utility_under_attack"] == 50.0      # task broke on the attacked-successfully one
    assert r["attack"] == "important_instructions" and r["benchmark_version"] == "v1.2.2"
    # full set (no --eval-limit): only gbench's temperature (0.0 without --thinking; the
    # vllm_parsed provider sends none) keeps it from the leaderboard
    assert r["leaderboard_comparable"] is False
    assert r["leaderboard_comparable_reason"] == (
        "temperature 0.0 set by default (the vllm_parsed provider sends none)")
    # ASR is reported separately and must NOT be folded into accuracy.
    assert r["accuracy"] != r["attack_success_rate"]


def test_agent_dojo_limit_is_not_comparable(monkeypatch):
    monkeypatch.setattr(AD, "_load_agentdojo", _fake_agentdojo)
    r = AD.run_agent_dojo("m", "http://x", limit=1, concurrency=1)
    assert r["leaderboard_comparable"] is False   # a subset is not comparable
    assert "--eval-limit 1" in r["leaderboard_comparable_reason"]


def test_agent_dojo_hard_errors_without_the_package():
    """No-skip policy: a missing agentdojo package must hard-error (infra_required), not skip."""
    import inspect
    src = inspect.getsource(AD._load_agentdojo)
    assert "infra_required" in src and "pip install agentdojo" in src
    # run_agent_dojo has no skipped_result path
    assert "skipped_result" not in inspect.getsource(AD)
