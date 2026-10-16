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
"""The SWE family runs the agentic protocol by default.

copilot_bench_swe, swe_bench_multilingual, swe_bench_live, multi_swe_bench and swe_bench_pro
roll out mini-swe-agent 1.x inside each instance image (swe_agentic) and score the submitted
diffs with their existing harness; GBENCH_<SUITE>_MODE=oneshot is the only way back to the
single-request path, and it is never leaderboard-comparable. Everything here is stubbed: no
container, no model endpoint, no harness run.
"""

import asyncio
import json
import sys
import types

import pytest

from gbench.runners.eval_suites import swe_agentic as A
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


SUITES = ("copilot_bench_swe", "swe_bench_multilingual", "swe_bench_live",
          "multi_swe_bench", "swe_bench_pro")


@pytest.fixture(autouse=True)
def _scratch_tmp(monkeypatch, tmp_path):
    """The runner and the scorers mkdtemp their work dirs, the trajectories follow
    GBENCH_RESULTS_DIR and the dataset snapshots GBENCH_CACHE_DIR; keep all of them inside the
    test's tmp."""
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))


# --------------------------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------------------------
class _StubEnv:
    def __init__(self):
        self.cleaned = False
        self.config = {"image": "stub"}

    def execute(self, command, cwd=""):
        return {"output": "", "returncode": 0}

    def get_template_vars(self):
        return {}

    def cleanup(self):
        self.cleaned = True


class _StubModel:
    def __init__(self, **config):
        self.config = dict(config)
        self.n_calls = 3
        self.cost = 0.0


class _StubAgent:
    """Stands in for DefaultAgent: records the task and returns a fixed outcome."""

    def __init__(self, model, env, outcome=("Submitted", "diff --git a/x b/x\n"), finish="length"):
        self.model, self.env, self.outcome, self.finish = model, env, outcome, finish
        self.config = {"step_limit": 250}
        self.messages = []

    def run(self, task, **kw):
        self.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": task},
            {"role": "assistant", "content": "THOUGHT\n```bash\nls\n```",
             "extra": {"response": {"choices": [{"finish_reason": self.finish, "message": {
                 "content": "x", "reasoning_content": "thinking about it"}}]}}},
            {"role": "user", "content": "<returncode>0</returncode>"},
        ]
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.fixture
def stub_rollout(monkeypatch):
    """Stub the container, model and agent; record what the runner handed them."""
    seen = {"model_cfg": [], "tasks": [], "env_cfg": []}
    envs = []

    def _env(cfg, task):
        seen["env_cfg"].append(cfg)
        seen["tasks"].append(task)
        if task.instance_id.startswith("broken"):
            raise RuntimeError("docker: image not found")
        env = _StubEnv()
        envs.append(env)
        return env

    def _model(cfg):
        seen["model_cfg"].append(cfg)
        return _StubModel(**cfg)

    outcomes = {}

    def _agent(model, env, agent_cfg):
        seen.setdefault("agent_cfg", []).append(agent_cfg)
        task = seen["tasks"][-1]
        return _StubAgent(model, env, outcomes.get(task.instance_id,
                                                   ("Submitted", "diff --git a/x b/x\n")),
                          finish=seen["finish"])

    monkeypatch.setattr(A, "_start_environment", _env)
    monkeypatch.setattr(A, "_make_model", _model)
    monkeypatch.setattr(A, "_make_agent", _agent)
    seen["outcomes"] = outcomes
    seen["envs"] = envs
    seen["finish"] = "length"
    return seen


def _scorer(resolved=()):
    calls = {}

    async def _score(traces):
        calls["traces"] = traces
        for t in traces:
            t["is_correct"] = t["extra_payload"]["instance_id"] in resolved
            t["status"] = "OK"
    return _score, calls


def _tasks(*ids, category="org/repo"):
    return [A.AgentTask(i, f"issue {i}", A.swebench_image(i), category) for i in ids]


# --------------------------------------------------------------------------------------------
# mode selection
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("suite", SUITES)
def test_default_mode_is_agentic_and_oneshot_is_opt_in(monkeypatch, suite):
    var = A.mode_env_var(suite)
    assert var == f"GBENCH_{suite.upper()}_MODE"
    monkeypatch.delenv(var, raising=False)
    assert A.resolve_mode(suite) == (A.AGENTIC, "default")
    monkeypatch.setenv(var, "oneshot")
    assert A.resolve_mode(suite) == (A.ONESHOT, f"env:{var}")
    monkeypatch.setenv(var, "sometimes")
    with pytest.raises(ValueError, match=var):
        A.resolve_mode(suite)


def test_mark_oneshot_is_never_comparable():
    res = A.mark_oneshot({"leaderboard_comparable": True}, "copilot_bench_swe")
    assert res["mode"] == A.ONESHOT and res["leaderboard_comparable"] is False
    assert "GBENCH_COPILOT_BENCH_SWE_MODE=oneshot" in res["leaderboard_comparable_reason"]
    again = A.mark_oneshot(dict(res), "copilot_bench_swe")
    assert again["leaderboard_comparable_reason"] == res["leaderboard_comparable_reason"]


def _stub_agentic_entry(monkeypatch, module, name):
    seen = {}

    def _fake(*a, **k):
        seen.update(args=a, kwargs=k)
        return {"status": "success", "mode": A.AGENTIC}
    monkeypatch.setattr(module, name, _fake)
    return seen


@pytest.mark.parametrize("suite,prereq", [
    ("copilot_bench_swe", "check_copilot_bench_swe_prerequisites"),
    ("swe_bench_multilingual", "check_swe_bench_multilingual_prerequisites"),
    ("swe_bench_live", "check_swe_bench_live_prerequisites"),
])
def test_swebench_schema_suites_route_by_mode(monkeypatch, suite, prereq):
    import importlib
    mod = importlib.import_module(f"gbench.runners.eval_suites.{suite}")
    run = getattr(mod, f"run_{suite}")
    monkeypatch.setattr(mod, prereq, lambda *a: (True, ""))
    monkeypatch.setattr(A, "require_agent", lambda *a: None)
    monkeypatch.setattr(A, "require_agent_v2", lambda *a: None)
    if suite == "swe_bench_live":
        # its Python tabs, the ones these two entry points run (the multi-language and Windows
        # tabs route through their own runners;
        # tests/test_swe_endpoint_relay_live_tabs_and_lancer.py); the agentic entry
        # runs SWE-agent-for-eval
        monkeypatch.setattr(mod, "SETTINGS", mod.PYTHON_SETTINGS)
        monkeypatch.setattr(mod, "check_sweagent_prerequisites", lambda: (True, ""))
    agentic = _stub_agentic_entry(monkeypatch, mod, "execute_sweagent_python"
                                  if suite == "swe_bench_live" else "execute_swebench_agentic")
    oneshot = _stub_agentic_entry(monkeypatch, mod, "execute_swebench")
    monkeypatch.delenv(A.mode_env_var(suite), raising=False)
    out = run("gemma", "http://127.0.0.1:9/v1", 2, enable_thinking=True, limit=3)
    assert agentic and not oneshot
    # --eval-limit 3 is 3 instances for the suite: swe_bench_live deals them over its 3 Python
    # tabs (1 each), the one-split suites run all 3
    want = 1 if suite == "swe_bench_live" else 3
    assert agentic["kwargs"]["enable_thinking"] is True and agentic["kwargs"]["limit"] == want
    assert out["mode"] == A.AGENTIC
    if suite == "swe_bench_live":
        # the fork image still scores the agentic patches
        assert agentic["kwargs"]["harness_image"] == mod._image()
        assert agentic["args"][6] == "starryzhang"
    if suite == "swe_bench_live":
        # SWE-agent-for-eval is one of the leaderboard's reference scaffolds: no scaffold note
        assert "leaderboard_note" not in agentic["kwargs"]
    else:
        # mini-swe-agent 2.x in its image, scored by the v5 harness image
        assert agentic["kwargs"]["agent_v2"] == mod.AGENT_VERSION
        assert agentic["kwargs"]["harness_v5"] is True

    agentic.clear()
    monkeypatch.setenv(A.mode_env_var(suite), "oneshot")
    out = run("gemma", "http://127.0.0.1:9/v1", 2)
    assert oneshot and not agentic
    assert out["mode"] == A.ONESHOT and out["leaderboard_comparable"] is False


def test_multi_swe_bench_routes_by_mode(monkeypatch):
    from gbench.runners.eval_suites import multi_swe_bench as M
    monkeypatch.setattr(M, "check_multi_swe_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M, "check_mswe_agent_prerequisites", lambda: (True, ""))
    seen = _stub_agentic_entry(monkeypatch, M, "_run_agentic")     # MSWE-agent
    monkeypatch.delenv("GBENCH_MULTI_SWE_BENCH_MODE", raising=False)
    out = M.run_multi_swe_bench("gemma", "http://127.0.0.1:9/v1", 2, enable_thinking=True)
    assert seen["args"][:4] == ("gemma", "http://127.0.0.1:9/v1", 2, True)
    assert out["mode"] == A.AGENTIC

    monkeypatch.setenv("GBENCH_MULTI_SWE_BENCH_MODE", "oneshot")
    monkeypatch.setattr(M, "_load_multi_swe_bench_samples", lambda limit=None: [])
    monkeypatch.setattr(M, "run_eval_suite", lambda **k: {"status": "success"})
    out = M.run_multi_swe_bench("gemma", "http://127.0.0.1:9/v1", 2)
    assert out["mode"] == A.ONESHOT and out["leaderboard_comparable"] is False
    assert "single-turn" in out["leaderboard_comparable_reason"]


def test_swe_bench_pro_default_is_the_public_agentic_runner(monkeypatch):
    from gbench.runners.eval_suites import swe_bench_pro as P
    monkeypatch.setattr(P, "check_swe_bench_pro_v2_prerequisites", lambda: (True, ""))
    monkeypatch.delenv("GBENCH_SWEBENCH_PRO_AGENTIC", raising=False)
    monkeypatch.delenv("GBENCH_SWE_BENCH_PRO_MODE", raising=False)
    seen = _stub_agentic_entry(monkeypatch, P, "run_swe_bench_pro_v2")
    P.run_swe_bench_pro("gemma", "http://127.0.0.1:9/v1", concurrency=2, limit=4)
    assert seen["args"][:3] == ("gemma", "http://127.0.0.1:9/v1", 2)
    assert seen["kwargs"]["limit"] == 4


def test_swe_bench_pro_flag_still_routes_to_the_maintained_engine(monkeypatch):
    """GBENCH_SWEBENCH_PRO_AGENTIC=1 keeps its old routing (a fake module stands in)."""
    from gbench.runners.eval_suites import swe_bench_pro as P
    monkeypatch.setattr(P, "check_swe_bench_pro_prerequisites", lambda: (True, ""))
    fake = types.ModuleType("gbench.runners.eval_suites.swe_bench_pro_agentic")
    got = {}
    fake.execute_agentic = lambda *a, **k: got.update(kwargs=k) or {"engine": "maintained"}
    monkeypatch.setitem(sys.modules, "gbench.runners.eval_suites.swe_bench_pro_agentic", fake)
    public = _stub_agentic_entry(monkeypatch, P, "run_swe_bench_pro_v2")
    monkeypatch.setenv("GBENCH_SWEBENCH_PRO_AGENTIC", "1")
    monkeypatch.delenv("GBENCH_SWE_BENCH_PRO_MODE", raising=False)
    res = P.run_swe_bench_pro("m", "http://x", concurrency=1, limit=3)
    assert res["engine"] == "maintained" and res["leaderboard_comparable"] is False
    assert P.ENGINE_V1_NOTE in res["leaderboard_comparable_reason"]
    assert got["kwargs"]["eval_name"] == "swe_bench_pro" and not public
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_MODE", "oneshot")
    with pytest.raises(ValueError, match="different protocols"):
        P.run_swe_bench_pro("m", "http://x", concurrency=1)


def test_swe_bench_pro_oneshot_is_flagged(monkeypatch):
    from gbench.runners.eval_suites import swe_bench_pro as P
    monkeypatch.setattr(P, "check_swe_bench_pro_v2_prerequisites", lambda: (True, ""))
    monkeypatch.delenv("GBENCH_SWEBENCH_PRO_AGENTIC", raising=False)
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_MODE", "oneshot")
    monkeypatch.setattr(P, "_load_swe_bench_pro_samples", lambda limit=None: [])
    monkeypatch.setattr(P, "run_eval_suite", lambda **k: {"status": "success", "temperature": 0.0})
    out = P.run_swe_bench_pro("gemma", "http://127.0.0.1:9/v1", concurrency=1)
    assert out["mode"] == A.ONESHOT and out["leaderboard_comparable"] is False
    assert out["leaderboard_comparable_reason"].startswith("one-shot mode")


# --------------------------------------------------------------------------------------------
# prerequisites: missing Docker / agent hard-errors
# --------------------------------------------------------------------------------------------
def test_missing_docker_hard_errors(monkeypatch):
    from gbench.runners.eval_suites import copilot_bench_swe as CB
    monkeypatch.setattr(CB, "check_copilot_bench_swe_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(A.shutil, "which", lambda _x: None)
    monkeypatch.delenv("GBENCH_COPILOT_BENCH_SWE_MODE", raising=False)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*docs/evals/copilot_bench_swe.md"):
        CB.run_copilot_bench_swe("gemma", "http://127.0.0.1:9/v1")


def test_unreachable_docker_daemon_fails_the_prereq(monkeypatch):
    monkeypatch.setattr(A.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(A.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=1))
    ok, reason = A.check_agent_prerequisites()
    assert not ok and "daemon" in reason


def test_missing_or_v2_mini_swe_agent_fails_the_prereq(monkeypatch):
    def _absent():
        raise ImportError("no minisweagent")
    monkeypatch.setattr(A, "_import_minisweagent", _absent)
    ok, reason = A.check_agent_prerequisites()
    assert not ok and "mini-swe-agent" in reason
    monkeypatch.setattr(A, "_import_minisweagent", lambda: types.SimpleNamespace(__version__="2.4.0"))
    ok, reason = A.check_agent_prerequisites()
    assert not ok and "1.x" in reason


def test_multi_swe_bench_agentic_hard_errors_without_the_agent(monkeypatch):
    from gbench.runners.eval_suites import multi_swe_bench as M
    monkeypatch.setattr(M, "check_multi_swe_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M, "check_mswe_agent_prerequisites",
                        lambda: (False, "the MSWE-agent image 'gbench-mswe-agent:88217624' is not built."))
    monkeypatch.delenv("GBENCH_MULTI_SWE_BENCH_MODE", raising=False)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*GBENCH_MULTI_SWE_BENCH_MODE=oneshot"):
        M.run_multi_swe_bench("gemma", "http://127.0.0.1:9/v1", 1)


# --------------------------------------------------------------------------------------------
# the agent config: bundled swebench.yaml + run knobs
# --------------------------------------------------------------------------------------------
def test_bundled_config_is_the_swebench_one():
    cfg, sha, version = A.load_agent_config()
    assert version.startswith("1.")
    assert cfg["agent"]["step_limit"] == 250 and cfg["environment"]["timeout"] == 60
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached" in \
        cfg["agent"]["instance_template"]
    assert len(sha) == 64


def test_knobs_reach_the_agent_config(monkeypatch, stub_rollout, tmp_path):
    monkeypatch.delenv("GBENCH_COPILOT_BENCH_SWE_TEMPERATURE", raising=False)
    monkeypatch.setenv("GBENCH_REPETITION_PENALTY", "1.05")
    score, _ = _scorer()
    res = A.run_agentic("copilot_bench_swe", "google/gemma-4-E4B-it", "http://127.0.0.1:9", 3,
                        _tasks("a__a-1", "b__b-2"), score, {}, enable_thinking=True,
                        max_output_tokens=32768)
    cfg = stub_rollout["model_cfg"][0]
    kw = cfg["model_kwargs"]
    assert cfg["model_name"] == "openai/google/gemma-4-E4B-it"
    assert cfg["cost_tracking"] == "ignore_errors"
    assert kw["api_base"] == "http://127.0.0.1:9/v1"
    assert kw["temperature"] == 1.0                      # the --thinking rule
    assert kw["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert kw["extra_body"]["repetition_penalty"] == 1.05
    # the bundled config sends no per-turn max_tokens, so --max-output-tokens is not applied
    assert "max_tokens" not in kw
    assert res["temperature"] == 1.0 and res["max_output_tokens_per_turn"] is None
    assert "--max-output-tokens=32768 is not applied" in res["max_output_tokens_note"]
    reason = res["leaderboard_comparable_reason"]
    assert "non-greedy temperature=1.0" in reason and "decoding penalties" in reason
    assert res["temperature_source"] == "thinking"
    assert "(--thinking samples the model under test at 1.0)" in reason
    env = stub_rollout["env_cfg"][0]["environment"]
    assert env["environment_class"] == "docker" and env["container_timeout"] == A.CONTAINER_TIMEOUT
    assert env["pull_timeout"] == A.PULL_TIMEOUT_S
    assert res["agent"]["scaffold"] == "mini-swe-agent" and res["agent"]["step_limit"] == 250


def test_suite_temperature_override_and_greedy_default(monkeypatch, stub_rollout):
    score, _ = _scorer()
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_TEMPERATURE", "0.3")
    res = A.run_agentic("swe_bench_live", "m", "http://127.0.0.1:9/v1", 1, _tasks("a__a-1"),
                        score, {})
    assert stub_rollout["model_cfg"][-1]["model_kwargs"]["temperature"] == 0.3
    assert res["temperature_source"] == "env:GBENCH_SWE_BENCH_LIVE_TEMPERATURE"
    monkeypatch.delenv("GBENCH_SWE_BENCH_LIVE_TEMPERATURE")
    A.run_agentic("swe_bench_live", "m", "http://127.0.0.1:9/v1", 1, _tasks("a__a-1"), score, {})
    assert stub_rollout["model_cfg"][-1]["model_kwargs"]["temperature"] == 0.0
    assert stub_rollout["model_cfg"][-1]["model_kwargs"]["extra_body"]["chat_template_kwargs"] == \
        {"enable_thinking": False}


def test_v2_agent_temperature_is_one_with_thinking_and_zero_without(monkeypatch):
    # the 2.x configs send no temperature; gbench sends its rule's: 1.0 with --thinking, 0.0 without
    monkeypatch.delenv("GBENCH_SWE_BENCH_MULTILINGUAL_TEMPERATURE", raising=False)
    base_cfg = {"model": {"model_kwargs": {}}, "environment": {}}
    t, src = A.agent_temperature("swe_bench_multilingual", None, thinking=True, agent_v2="2.4.6")
    assert (t, src) == (1.0, "thinking")
    kw = A.build_agent_v2_overrides(base_cfg, model_name="m", base_url="http://r:1", temperature=t,
                                    thinking=True, max_output_tokens=None)["model"]["model_kwargs"]
    assert kw["temperature"] == 1.0
    assert kw["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    t, src = A.agent_temperature("swe_bench_multilingual", None, thinking=False, agent_v2="2.4.6")
    assert (t, src) == (0.0, "default")
    kw = A.build_agent_v2_overrides(base_cfg, model_name="m", base_url="http://r:1", temperature=t,
                                    thinking=False, max_output_tokens=None)["model"]["model_kwargs"]
    assert kw["temperature"] == 0.0
    monkeypatch.setenv("GBENCH_SWE_BENCH_MULTILINGUAL_TEMPERATURE", "0.2")
    assert A.agent_temperature("swe_bench_multilingual", None, thinking=True, agent_v2="2.4.6") \
        == (0.2, "env:GBENCH_SWE_BENCH_MULTILINGUAL_TEMPERATURE")


def test_max_tokens_is_only_clamped_to_the_context_window(monkeypatch):
    """The per-turn budget drops only when prompt + budget would not fit the window."""
    cls = A._endpoint_model_class()
    captured = {}

    def _fake_query(self, messages, **kwargs):
        captured.update(kwargs)
        return {"content": "", "extra": {}}
    monkeypatch.setattr(cls.__mro__[1], "query", _fake_query)
    m = cls(model_name="openai/m", model_kwargs={"max_tokens": 8192}, cost_tracking="ignore_errors")
    msgs = [{"role": "user", "content": "x", "timestamp": 1}]
    monkeypatch.setattr(A, "clamp_to_context", lambda plain, budget: (budget, None))
    m.query(msgs)
    assert "max_tokens" not in captured                  # fits: the configured value stands
    monkeypatch.setattr(A, "clamp_to_context", lambda plain, budget: (3000, None))
    m.query(msgs)
    assert captured["max_tokens"] == 3000
    captured.clear()
    monkeypatch.setattr(A, "clamp_to_context", lambda plain, budget: (budget, "does not fit"))
    m.query(msgs)                                        # the server reports the overflow
    assert "max_tokens" not in captured


# --------------------------------------------------------------------------------------------
# traces, scoring, failure accounting
# --------------------------------------------------------------------------------------------
def test_traces_carry_trajectory_exit_status_and_patch(stub_rollout):
    score, calls = _scorer(resolved={"a__a-1"})
    res = A.run_agentic("copilot_bench_swe", "m", "http://127.0.0.1:9/v1", 2,
                        _tasks("a__a-1", "b__b-2"), score, {})
    tr = {t["extra_payload"]["instance_id"]: t for t in res["sample_traces"]}
    t = tr["a__a-1"]
    assert t["agent_exit_status"] == "Submitted" and t["agent_patch"] == "diff --git a/x b/x\n"
    assert t["response_text"] == t["agent_patch"] and t["agent_steps"] == 3
    roles = [m["role"] for m in t["trajectory"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert t["trajectory"][2]["reasoning"] == "thinking about it"
    assert t["trajectory"][2]["finish_reason"] == "length" and t["truncated_turns"] == 1
    assert t["trajectory_file"] and t["trajectory_file"].startswith(res["agent_output_dir"])
    assert t["agent_image"] == A.swebench_image("a__a-1")
    assert res["correct_answers"] == 1 and res["accuracy"] == 50.0
    assert res["agent_exit_statuses"] == {"Submitted": 2}
    assert all(e.cleaned for e in stub_rollout["envs"])
    # no per-turn cap is sent, so a length-truncated turn filled the context window, as it
    # would upstream: recorded, not a deviation
    assert res["truncated_turns"] == 2
    assert "per-turn" not in (res.get("leaderboard_comparable_reason") or "")


def test_trajectory_file_is_mini_swe_agent_format(stub_rollout):
    score, _ = _scorer()
    res = A.run_agentic("copilot_bench_swe", "m", "http://127.0.0.1:9/v1", 1, _tasks("a__a-1"),
                        score, {})
    data = json.load(open(res["sample_traces"][0]["trajectory_file"]))
    assert data["trajectory_format"] == "mini-swe-agent-1"
    assert data["info"]["exit_status"] == "Submitted"
    assert data["info"]["submission"] == "diff --git a/x b/x\n"


def test_infra_failures_are_partial_not_model_zeros(stub_rollout):
    score, calls = _scorer(resolved={"a__a-1"})
    res = A.run_agentic("copilot_bench_swe", "m", "http://127.0.0.1:9/v1", 2,
                        _tasks("a__a-1", "broken__x-1"), score, {})
    broken = [t for t in res["sample_traces"] if t["agent_infra_error"]]
    assert len(broken) == 1 and broken[0]["agent_failed_phase"] == "environment"
    assert broken[0]["status"] == "AGENT_INFRA_ERROR"
    # the harness only sees rollouts that ran
    assert [t["extra_payload"]["instance_id"] for t in calls["traces"]] == ["a__a-1"]
    assert res["status"] == "completed_with_errors" and res["failed_requests"] == 1
    assert res["partial_evaluation"]["agent_infra_error_instances"] == 1
    assert res["leaderboard_comparable"] is False
    assert "partial: 1 of 2" in res["leaderboard_comparable_reason"]


def test_all_infra_failures_are_an_error_not_a_zero(stub_rollout):
    score, calls = _scorer()
    res = A.run_agentic("copilot_bench_swe", "m", "http://127.0.0.1:9/v1", 2,
                        _tasks("broken__a-1", "broken__b-2"), score, {})
    assert res["status"] == "error" and "not a 0% resolved rate" in res["error"]
    assert "traces" not in calls                          # nothing to score


def test_context_window_exhaustion_is_the_models_outcome(stub_rollout):
    class ContextWindowExceededError(Exception):
        pass
    stub_rollout["outcomes"]["a__a-1"] = ContextWindowExceededError("too long")
    stub_rollout["outcomes"]["b__b-2"] = ConnectionError("endpoint down")
    score, calls = _scorer()
    res = A.run_agentic("copilot_bench_swe", "m", "http://127.0.0.1:9/v1", 2,
                        _tasks("a__a-1", "b__b-2"), score, {})
    tr = {t["extra_payload"]["instance_id"]: t for t in res["sample_traces"]}
    assert tr["a__a-1"]["agent_infra_error"] is False and tr["a__a-1"]["agent_patch"] == ""
    assert tr["b__b-2"]["agent_infra_error"] is True and tr["b__b-2"]["agent_failed_phase"] == "agent"
    assert [t["extra_payload"]["instance_id"] for t in calls["traces"]] == ["a__a-1"]


def test_clean_full_greedy_verified_run_is_comparable(stub_rollout):
    """Verified is the bash-only leaderboard protocol: a full greedy run with every instance
    scored and no truncated turn is comparable; any deviation is named."""
    score, _ = _scorer(resolved={"a__a-1"})
    stub_rollout["finish"] = "stop"
    res = A.run_agentic("copilot_bench_swe", "m", "http://127.0.0.1:9/v1", 2,
                        _tasks("a__a-1", "b__b-2"), score, {})
    assert res["leaderboard_comparable"] is True and "leaderboard_comparable_reason" not in res
    assert res["status"] == "success" and res["mode"] == A.AGENTIC
    limited = A.run_agentic("copilot_bench_swe", "m", "http://127.0.0.1:9/v1", 2,
                            _tasks("a__a-1", "b__b-2"), score, {}, limit=1,
                            leaderboard_note="scaffold differs")
    assert limited["leaderboard_comparable"] is False
    assert limited["total_questions"] == 1
    assert "scaffold differs" in limited["leaderboard_comparable_reason"]
    assert "subset run (--eval-limit 1)" in limited["leaderboard_comparable_reason"]


def test_attempts_are_not_fabricated(stub_rollout):
    score, _ = _scorer()
    res = A.run_agentic("copilot_bench_swe", "m", "http://127.0.0.1:9/v1", 1, _tasks("a__a-1"),
                        score, {}, attempt_count=4)
    assert res["attempts"]["attempts_per_sample"] == 1 and res["total_questions"] == 1


def test_shard_is_applied_and_an_empty_shard_is_honest(monkeypatch, stub_rollout):
    score, _ = _scorer()
    monkeypatch.setenv("GBENCH_SHARD", "2/2")
    res = A.run_agentic("copilot_bench_swe", "m", "http://127.0.0.1:9/v1", 1,
                        _tasks("a__a-1", "b__b-2", "c__c-3"), score, {})
    assert [t["extra_payload"]["instance_id"] for t in res["sample_traces"]] == ["b__b-2"]
    assert res["shard"] == "2/2" and res["leaderboard_comparable"] is False
    monkeypatch.setenv("GBENCH_SHARD", "3/3")
    empty = A.run_agentic("copilot_bench_swe", "m", "http://127.0.0.1:9/v1", 1,
                          _tasks("a__a-1"), score, {})
    assert empty["status"] == "success" and empty["total_questions"] == 0


def test_the_swebench_scorer_submits_the_agent_patch_verbatim(monkeypatch, tmp_path):
    """extract_patch cuts a diff at a context line that looks like a fence; the agent's
    submission is already a diff and must reach the harness untouched."""
    patch_text = ("diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n"
                  "@@ -1,3 +1,3 @@\n ```\n-old\n+new\n ```\n")
    assert C.extract_patch(patch_text) != patch_text            # why verbatim matters
    captured = {}

    def _run(cmd, **kw):
        path = cmd[cmd.index("--predictions_path") + 1]
        captured["preds"] = [json.loads(l) for l in open(path)]
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(C.subprocess, "run", _run)
    scorer = C.make_swebench_scorer("copilot_bench_swe", "m", "d", "test", "swebench", 1, {})
    traces = [{"extra_payload": {"instance_id": "a__a-1"}, "response_text": patch_text,
               "agent_patch": patch_text},
              {"extra_payload": {"instance_id": "b__b-2"},
               "response_text": "Here:\n```diff\n" + patch_text.replace(" ```\n", "") + "```"}]
    asyncio.run(scorer(traces))
    preds = {p["instance_id"]: p["model_patch"] for p in captured["preds"]}
    assert preds["a__a-1"] == patch_text
    assert preds["b__b-2"].startswith("diff --git")             # one-shot still extracts


def test_swebench_report_reasons_partial_and_error(tmp_path):
    res = {"status": "success"}
    reasons = C.swebench_report_reasons("x", res, {"swebench_report": {"error": "no report"}})
    assert res["status"] == "error" and reasons == ["harness error (no valid resolved rate)"]
    res = {"status": "success"}
    metrics = {"swebench_workdir": str(tmp_path), "swebench_report": {
        "completed_instances": 2, "total_instances": 3, "error_ids": ["c__c-3"]}}
    reasons = C.swebench_report_reasons("x", res, metrics)
    assert res["status"] == "completed_with_errors"
    assert res["partial_evaluation"]["harness_error_instances"] == 1
    assert reasons and reasons[0].startswith("partial: 1 of 3")


def test_execute_swebench_agentic_wires_rows_scorer_and_report(monkeypatch):
    rows = [{"instance_id": "a__a-1", "problem_statement": "p", "repo": "a/a"}]
    monkeypatch.setattr(C, "load_swe_rows", lambda d, s, e: rows)
    seen = {}

    def _run_agentic(eval_name, model_name, base_url, conc, tasks, scorer, metrics, **kw):
        seen.update(tasks=tasks, kw=kw)
        return {"status": "success"}
    monkeypatch.setattr(A, "run_agentic", _run_agentic)
    C.execute_swebench_agentic("swe_bench_live", "m", "http://x/v1", 1, "SWE-bench-Live/x",
                               "lite", "starryzhang", leaderboard_note="n",
                               harness_image="img", limit=2)
    t = seen["tasks"][0]
    assert t.image == "docker.io/starryzhang/sweb.eval.x86_64.a_1776_a-1:latest"
    assert t.cwd == "/testbed" and t.meta == {"split": "lite"}
    assert seen["kw"]["leaderboard_note"] == "n" and seen["kw"]["limit"] == 2
    assert callable(seen["kw"]["finalize"])


# --------------------------------------------------------------------------------------------
# instance layout per upstream
# --------------------------------------------------------------------------------------------
def test_swebench_image_matches_mini_swe_agent():
    from minisweagent.run.extra.swebench import get_swebench_docker_image_name
    for iid in ("django__django-11099", "apache__druid-13704", "Kong__insomnia-1"):
        assert A.swebench_image(iid) == get_swebench_docker_image_name({"instance_id": iid})


def test_task_config_moves_the_workdir_only_where_the_image_differs():
    base, _, _ = A.load_agent_config()
    cfg = A.build_agent_config(base, model_name="m", base_url="http://x/v1", temperature=0.0,
                               thinking=False, max_output_tokens=None)
    plain = A.task_config(cfg, _tasks("a__a-1")[0])
    assert plain["agent"]["instance_template"] == base["agent"]["instance_template"]
    assert plain["environment"]["cwd"] == "/testbed"
    app = A.AgentTask("instance_x", "P", "img:x", "go", cwd="/app",
                      run_args=["--rm", "--entrypoint", ""])
    moved = A.task_config(cfg, app)
    assert moved["environment"]["cwd"] == "/app"
    assert moved["environment"]["run_args"] == ["--rm", "--entrypoint", ""]
    assert "/testbed" not in moved["agent"]["instance_template"]
    assert "files in /app (this is the working directory" in moved["agent"]["instance_template"]
    assert "max_tokens" not in cfg["model"]["model_kwargs"]


def test_swe_bench_pro_task_refuses_to_guess_the_image(monkeypatch):
    from gbench.runners.eval_suites import swe_bench_pro as P
    rows = [{"instance_id": f"instance_{n}", "problem_statement": "P",
             "docker_image": f"ghcr.io/scaleapi/swe-bench_pro-v2:instance_{n}"}
            for n in range(P.V2_TASK_COUNT)]
    rows[5]["docker_image"] = None
    monkeypatch.setattr(P, "_load_config_rows", lambda config: rows)
    with pytest.raises(RuntimeError, match="not a V2 task"):
        P.load_v2_rows()


def test_pro_agentic_refuses_v2_rows_and_names_the_v1_pin(monkeypatch):
    """The v1 rows behind the maintained engine's raw-sample table refuse V2 rows (docker_image,
    /tests/test_patch.patch), and the engine's result says it is a v1 number."""
    from gbench.runners.eval_suites import swe_bench_pro as P
    row = {"instance_id": "instance_x", "dockerhub_tag": "o.r-x", "problem_statement": "P",
           "requirements": "R", "interface": "I", "repo_language": "go",
           "before_repo_set_cmd": "git reset --hard x\ngit apply --verbose /tests/test_patch.patch",
           "docker_image": "ghcr.io/scaleapi/swe-bench_pro-v2:instance_x"}
    fake_ds = types.ModuleType("datasets")
    fake_ds.load_dataset = lambda name, config, split, revision: [row]
    monkeypatch.setitem(sys.modules, "datasets", fake_ds)
    with pytest.raises(RuntimeError, match="V2"):
        P.load_pinned_rows()
    v1 = {k: v for k, v in row.items() if k != "docker_image"}
    v1["before_repo_set_cmd"] = "git reset --hard x\ngit checkout x -- tests/a.py"
    fake_ds.load_dataset = lambda name, config, split, revision: [v1]
    assert P.load_pinned_rows() == [v1]
    assert P.V1_PIN_NOTE in P.ENGINE_V1_NOTE and "V2" in P.V1_PIN_NOTE
    assert "307bd006" in P.V1_PIN_NOTE and P.DATASET_LABEL == P.V1_DATASET_LABEL
    # the Pro scorer reads the agent's patch verbatim
    assert "trace_patch(tr, _extract_patch)" in open(P.__file__).read()


# --------------------------------------------------------------------------------------------
# multi_swe_bench: canonical 7-language set, pinned, MSWE-agent task text
# --------------------------------------------------------------------------------------------
def _fake_hub(monkeypatch, tmp_path, rows_by_file):
    seen = {"revisions": set()}

    class _Api:
        def list_repo_files(self, repo, repo_type=None, revision=None):
            seen["revisions"].add(revision)
            return list(rows_by_file) + ["README.md", "python/multi_swe_bench_python.jsonl"]

    def _download(repo_id, filename, repo_type=None, revision=None):
        seen["revisions"].add(revision)
        p = tmp_path / filename.replace("/", "_")
        p.write_text("".join(json.dumps(r) + "\n" for r in rows_by_file[filename]))
        return str(p)
    monkeypatch.setattr("huggingface_hub.HfApi", _Api)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _download)
    return seen


def _mrow(org, repo, n):
    return {"org": org, "repo": repo, "number": n, "title": "PR title", "body": "PR body",
            "base": {"sha": "abc"}, "fix_patch": "gold",
            "resolved_issues": [{"number": 1, "title": "Issue title", "body": "Issue body"}]}


def test_multi_swe_bench_loads_the_seven_languages_at_the_pinned_revision(monkeypatch, tmp_path):
    from gbench.runners.eval_suites import multi_swe_bench as M
    files = {f"{lang}/o__{lang}_dataset.jsonl": [_mrow("o", lang, 1), _mrow("o", lang, 2)]
             for lang in ("c", "cpp", "go", "java", "js", "kotlin", "rust", "ts")}
    seen = _fake_hub(monkeypatch, tmp_path, files)
    tasks, rows = M._load_multi_swe_bench_tasks()
    langs = sorted({t.category for t in tasks})
    assert langs == sorted(M._LANGUAGES) and "kotlin" not in langs and len(tasks) == 14
    assert seen["revisions"] == {M._REVISION}
    assert M._REVISION == "56ff018c04a38e27ada1e9d0a6d5839a51f88f0d"
    t = next(t for t in tasks if t.category == "go")
    # the published image and MSWE-agent's own task text; MSWE-agent sets up the repository
    # itself (no gbench startup command) and reads the row the task came from
    assert t.image == "mswebench/o_m_go:pr-1" and t.cwd == "/home/go"
    assert t.problem_statement == "TITLE:\nIssue title\n CONTENT:\nIssue body"
    assert t.startup_command is None
    assert t.meta == {"org": "o", "repo": "go", "number": 1}
    assert rows[t.instance_id]["resolved_issues"][0]["title"] == "Issue title"
    assert M._SRC_FILES[t.instance_id]
    samples = M._load_multi_swe_bench_samples()
    assert {s[2]["category"] for s in samples} == set(M._LANGUAGES)


def test_multi_swe_bench_refuses_to_invent_the_task_text(monkeypatch, tmp_path):
    from gbench.runners.eval_suites import multi_swe_bench as M
    row = _mrow("o", "go", 1)
    row["resolved_issues"] = []
    _fake_hub(monkeypatch, tmp_path, {"go/o__go_dataset.jsonl": [row]})
    with pytest.raises(RuntimeError, match="resolved_issues"):
        M._load_multi_swe_bench_tasks()


def test_multi_swe_bench_scorer_submits_the_agent_patch(monkeypatch, tmp_path):
    from gbench.runners.eval_suites import multi_swe_bench as M
    src = tmp_path / "go.jsonl"
    src.write_text(json.dumps({**_mrow("o", "go", 1), "instance_id": "o__go-1"}) + "\n")
    monkeypatch.setattr(M, "_SRC_FILES", {"o__go-1": str(src)})
    monkeypatch.setenv("GBENCH_MULTI_SWE_REPO_DIR", str(tmp_path / "repos"))
    captured = {}

    def _run(cmd, **kw):
        cfg = json.load(open(cmd[cmd.index("--config") + 1]))
        captured["preds"] = [json.loads(l) for l in open(cfg["patch_files"][0])]
        return types.SimpleNamespace(returncode=1, stdout="", stderr="boom")
    monkeypatch.setattr(M.subprocess, "run", _run)
    metrics = {}
    trace = {"extra_payload": {"instance_id": "o__go-1", "org": "o", "repo": "go", "number": 1},
             "response_text": "x", "agent_patch": "diff --git a/f b/f\n ```\n"}
    asyncio.run(M._make_scorer("m", 1, metrics)([trace]))
    assert captured["preds"][0]["fix_patch"] == "diff --git a/f b/f\n ```\n"
    assert metrics["multi_swe_bench_report"]["error"]


# --------------------------------------------------------------------------------------------
# the real mini-swe-agent DefaultAgent, with a scripted model and a fake container
# --------------------------------------------------------------------------------------------
class _ScriptedModel:
    def __init__(self, replies):
        self.replies, self.n_calls, self.cost = list(replies), 0, 0.0
        self.config = {"model_name": "scripted"}

    def query(self, messages, **kw):
        self.n_calls += 1
        return {"content": self.replies.pop(0), "extra": {}}

    def get_template_vars(self):
        return {"n_model_calls": self.n_calls, "model_cost": self.cost}


class _GitEnv(_StubEnv):
    def execute(self, command, cwd=""):
        if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in command:
            return {"output": "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\ndiff --git a/f b/f\n+x\n",
                    "returncode": 0}
        return {"output": "f\n", "returncode": 0}


def test_real_default_agent_submission_becomes_the_patch(monkeypatch, tmp_path):
    submit = ("THOUGHT: done\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && "
              "git add -A && git diff --cached\n```")
    monkeypatch.setattr(A, "_start_environment", lambda cfg, task: _GitEnv())
    monkeypatch.setattr(A, "_make_model", lambda cfg: _ScriptedModel(
        ["THOUGHT: look\n\n```bash\nls\n```", submit]))
    base, _, _ = A.load_agent_config()
    cfg = A.build_agent_config(base, model_name="m", base_url="http://x/v1", temperature=0.0,
                               thinking=False, max_output_tokens=None)
    tr = A._rollout(0, _tasks("a__a-1")[0], cfg, str(tmp_path))
    assert tr["agent_exit_status"] == "Submitted" and tr["agent_steps"] == 2
    assert tr["agent_patch"] == "diff --git a/f b/f\n+x\n"
    assert tr["trajectory"][0]["role"] == "system" and "issue a__a-1" in tr["trajectory"][1]["content"]


def test_real_default_agent_step_limit_submits_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "_start_environment", lambda cfg, task: _GitEnv())
    monkeypatch.setattr(A, "_make_model", lambda cfg: _ScriptedModel(
        ["THOUGHT\n```bash\nls\n```"] * 3))
    base, _, _ = A.load_agent_config()
    cfg = A.build_agent_config(base, model_name="m", base_url="http://x/v1", temperature=0.0,
                               thinking=False, max_output_tokens=None)
    cfg["agent"]["step_limit"] = 2
    tr = A._rollout(0, _tasks("a__a-1")[0], cfg, str(tmp_path))
    assert tr["agent_exit_status"] == "LimitsExceeded" and tr["agent_patch"] == ""
    assert tr["agent_infra_error"] is False


# ============================================================================================
# Model id, agent version, leaderboard config, refusals, trajectories and pinned snapshots
# ============================================================================================
# --- model id: litellm always gets the openai/ provider prefix ------------------------------
@pytest.mark.parametrize("served", ["google/gemma-4-31b-it", "openai/gpt-oss-20b", "gemma"])
def test_served_model_id_reaches_the_endpoint_unchanged(served):
    """litellm strips one provider segment, so 'openai/gpt-oss-20b' must be sent as
    'openai/openai/gpt-oss-20b' or vLLM receives 'gpt-oss-20b' and answers 404."""
    import litellm
    base, _, _ = A.load_agent_config()
    cfg = A.build_agent_config(base, model_name=served, base_url="http://x/v1", temperature=0.0,
                               thinking=False, max_output_tokens=None)
    model, provider = litellm.get_llm_provider(cfg["model"]["model_name"])[:2]
    assert (model, provider) == (served, "openai")


# --- prerequisite: mini-swe-agent >=1.15,<2 --------------------------------------------------
def _docker_ok(monkeypatch):
    monkeypatch.setattr(A.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(A.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0))


@pytest.mark.parametrize("version,ok", [("1.9.1", False), ("1.13.3", False), ("1.14.0", False),
                                        ("1.15.0", True), ("1.17.5", True), ("2.0.0", False),
                                        ("2.4.0", False), ("dev", False)])
def test_agent_version_floor(monkeypatch, version, ok):
    """cost_tracking (always passed) first exists in 1.15.0; older 1.x crashed every rollout."""
    _docker_ok(monkeypatch)
    monkeypatch.setattr(A, "_import_minisweagent", lambda: types.SimpleNamespace(__version__=version))
    got, reason = A.check_agent_prerequisites()
    assert got is ok
    if not ok:
        assert "mini-swe-agent>=1.15,<2" in reason and version in reason


def test_leaderboard_comparable_requires_the_leaderboard_config(monkeypatch, stub_rollout):
    """A clean full greedy run is comparable only on the a29173c4 swebench.yaml."""
    real = A.load_agent_config
    stub_rollout["finish"] = "stop"
    score, _ = _scorer()
    base, sha, version = real()
    assert sha == A.LEADERBOARD_CONFIG_SHA256
    monkeypatch.setattr(A, "load_agent_config", lambda: (base, "d8bcea20" + "0" * 56, "1.13.3"))
    res = A.run_agentic("copilot_bench_swe", "m", "http://127.0.0.1:9/v1", 1, _tasks("a__a-1"),
                        score, {})
    assert res["leaderboard_comparable"] is False
    assert "sha256 d8bcea200000..." in res["leaderboard_comparable_reason"]
    monkeypatch.setattr(A, "load_agent_config", real)
    res = A.run_agentic("copilot_bench_swe", "m", "http://127.0.0.1:9/v1", 1, _tasks("a__a-1"),
                        score, {})
    assert res["leaderboard_comparable"] is True


# --- context window refusal caused by gbench's per-turn cap ---------------------------------
class ContextWindowExceededError(Exception):
    pass


def test_refusal_at_the_gbench_cap_is_resent_uncapped(monkeypatch):
    """With the window unknown (no clamp), the server refuses prompt + max_tokens; the turn is
    resent without max_tokens (upstream's request) and counted, not scored as an overflow."""
    cls = A._endpoint_model_class()
    sent = []

    def _parent_query(self, messages, **kwargs):
        mt = {**self.config.model_kwargs, **kwargs}.get("max_tokens")
        sent.append(mt)
        if mt is not None or "too long" in messages[-1]["content"]:
            raise ContextWindowExceededError("maximum context length is 1000 tokens")
        return {"content": "ok", "extra": {}}
    monkeypatch.setattr(cls.__mro__[1], "query", _parent_query)
    monkeypatch.setattr(A, "clamp_to_context", lambda plain, budget: (budget, None))
    m = cls(model_name="openai/m", model_kwargs={"max_tokens": 32768}, cost_tracking="ignore_errors")
    assert m.query([{"role": "user", "content": "hi"}])["content"] == "ok"
    assert sent == [32768, None] and m.context_retry_turns == 1
    sent.clear()
    with pytest.raises(ContextWindowExceededError):       # the prompt alone overflows
        m.query([{"role": "user", "content": "too long"}])
    assert sent == [32768, None] and m.context_retry_turns == 1
    # without a gbench cap there is nothing to resend
    plain = cls(model_name="openai/m", model_kwargs={}, cost_tracking="ignore_errors")
    sent.clear()
    with pytest.raises(ContextWindowExceededError):
        plain.query([{"role": "user", "content": "too long"}])
    assert sent == [None]
    # any other failure propagates untouched
    monkeypatch.setattr(cls.__mro__[1], "query",
                        lambda self, messages, **kw: (_ for _ in ()).throw(ConnectionError("down")))
    with pytest.raises(ConnectionError):
        m.query([{"role": "user", "content": "hi"}])


def test_context_retry_turns_are_recorded(monkeypatch, stub_rollout):
    class _Model(_StubModel):
        def __init__(self, **cfg):
            super().__init__(**cfg)
            self.context_retry_turns = 2
    monkeypatch.setattr(A, "_make_model", lambda cfg: _Model(**cfg))
    score, _ = _scorer()
    res = A.run_agentic("copilot_bench_swe", "m", "http://127.0.0.1:9/v1", 1,
                        _tasks("a__a-1"), score, {})
    assert res["sample_traces"][0]["context_retry_turns"] == 2
    assert res["context_retry_turns"] == 2


# --- trajectories live next to the results ---------------------------------------------------
def test_trajectories_are_written_under_the_run_results_dir(monkeypatch, stub_rollout, tmp_path):
    run_dir = tmp_path / "results" / "09-24-120000"
    monkeypatch.setenv("GBENCH_RESULTS_DIR", str(run_dir))
    score, _ = _scorer()
    res = A.run_agentic("swe_bench_live", "m", "http://127.0.0.1:9/v1", 1, _tasks("a__a-1"),
                        score, {})
    out = res["agent_output_dir"]
    assert out.startswith(str(run_dir / "agent_trajectories" / "swe_bench_live_"))
    assert res["sample_traces"][0]["trajectory_file"].startswith(out)
    again = A.agent_output_dir("swe_bench_live")
    assert again != out                                       # a fresh directory per run
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_TRAJ_DIR", str(tmp_path / "mine"))
    assert A.traj_dir_env_var("swe_bench_live") == "GBENCH_SWE_BENCH_LIVE_TRAJ_DIR"
    assert A.agent_output_dir("swe_bench_live").startswith(str(tmp_path / "mine"))


# --- SWE-bench-schema datasets pinned on both sides -----------------------------------------
_PINNED = {
    "copilot_bench_swe": ("SWE-bench/SWE-bench_Verified", "78f471bf655a3137b2e8a75af1501690ec009ec3"),
    "swe_bench_multilingual": ("SWE-bench/SWE-bench_Multilingual", "846e647b9f33c0b51b739d005d13d85493c9af09"),
    "swe_bench_live": ("SWE-bench-Live/SWE-bench-Live", "b51a86422e10cfd403beb4773e5a2947953e36ec"),
}


def _fake_datasets(monkeypatch, rows):
    calls = []
    fake = types.ModuleType("datasets")

    def _load(name, *a, **k):
        calls.append((name, a, k))
        return list(rows)
    fake.load_dataset = _load
    monkeypatch.setitem(sys.modules, "datasets", fake)
    return calls


def test_pinned_snapshot_is_written_once_and_read_by_the_harness(monkeypatch, tmp_path):
    import datetime
    from swebench.harness.utils import load_swebench_dataset
    rows = [{"instance_id": "a__a-1", "problem_statement": "p", "repo": "a/a",
             "created_at": datetime.datetime(2025, 1, 2)},
            {"instance_id": "b__b-2", "problem_statement": "q", "repo": "b/b", "created_at": None}]
    calls = _fake_datasets(monkeypatch, rows)
    path = C.pinned_swe_snapshot("SWE-bench-Live/SWE-bench-Live", "lite", "b51a8642", "swe_bench_live")
    assert path == str(tmp_path / "cache" / "swebench_snapshots"
                       / "SWE-bench-Live__SWE-bench-Live@b51a8642" / "lite.jsonl")
    assert calls == [("SWE-bench-Live/SWE-bench-Live", (), {"split": "lite", "revision": "b51a8642"})]
    # the swebench harness takes the file as --dataset_name
    got = load_swebench_dataset(path, "lite", ["b__b-2"])
    assert [r["instance_id"] for r in got] == ["b__b-2"]
    assert [r["instance_id"] for r in C.read_swe_snapshot(path, "x")] == ["a__a-1", "b__b-2"]
    # reused offline: no second load
    sys.modules["datasets"].load_dataset = lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
    assert C.pinned_swe_snapshot("SWE-bench-Live/SWE-bench-Live", "lite", "b51a8642", "x") == path
    with pytest.raises(RuntimeError, match="Could not load dataset for x"):
        C.pinned_swe_snapshot("SWE-bench-Live/SWE-bench-Live", "full", "b51a8642", "x")


def test_agentic_rows_and_harness_read_the_same_snapshot(monkeypatch):
    rows = [{"instance_id": "a__a-1", "problem_statement": "p", "repo": "a/a"}]
    _fake_datasets(monkeypatch, rows)
    seen = {}

    def _scorer_factory(eval_name, model_name, dataset, split, namespace, workers, metrics,
                        harness_image=None, harness_v5=False):
        seen["harness_dataset"] = dataset
        return lambda traces: None
    monkeypatch.setattr(C, "make_swebench_scorer", _scorer_factory)
    monkeypatch.setattr(C, "load_swe_rows", lambda *a: pytest.fail("unpinned load"))
    monkeypatch.setattr(A, "run_agentic", lambda *a, **k: seen.update(tasks=a[4], metrics=a[6],
                                                                      kw=k) or {})
    C.execute_swebench_agentic("copilot_bench_swe", "m", "http://x/v1", 1,
                               "princeton-nlp/SWE-bench_Verified", "test", "swebench",
                               dataset_revision="c104f840")
    assert seen["harness_dataset"].endswith("princeton-nlp__SWE-bench_Verified@c104f840/test.jsonl")
    assert [t.instance_id for t in seen["tasks"]] == ["a__a-1"]
    assert seen["metrics"]["dataset"] == "princeton-nlp/SWE-bench_Verified@c104f840 (test)"
    assert "dataset_revision" not in seen["kw"]


def test_oneshot_samples_and_harness_use_the_pinned_revision(monkeypatch):
    rows = [{"instance_id": "a__a-1", "problem_statement": "p", "repo": "a/a"}]
    calls = _fake_datasets(monkeypatch, rows)
    seen = {}

    def _scorer_factory(eval_name, model_name, dataset, split, namespace, workers, metrics,
                        harness_image=None, harness_v5=False):
        seen["harness_dataset"] = dataset
        return lambda traces: None
    monkeypatch.setattr(C, "make_swebench_scorer", _scorer_factory)
    monkeypatch.setattr(C, "run_eval_suite", lambda **k: {"status": "success", "temperature": 0.0})
    res = C.execute_swebench("swe_bench_multilingual", "m", "http://x/v1", 1,
                             "SWE-bench/SWE-bench_Multilingual", "test", "swebench",
                             dataset_revision="846e647b")
    assert all(k.get("revision") == "846e647b" for _, _, k in calls) and calls
    assert seen["harness_dataset"].endswith("SWE-bench__SWE-bench_Multilingual@846e647b/test.jsonl")
    assert res["dataset"] == "SWE-bench/SWE-bench_Multilingual@846e647b (test)"


@pytest.mark.parametrize("suite", sorted(_PINNED))
def test_suites_pass_their_pinned_revision(monkeypatch, suite):
    import importlib
    mod = importlib.import_module(f"gbench.runners.eval_suites.{suite}")
    dataset, revision = _PINNED[suite]
    assert (mod._DATASET, mod._REVISION) == (dataset, revision)
    monkeypatch.setattr(mod, f"check_{suite}_prerequisites", lambda *a: (True, ""))
    monkeypatch.setattr(A, "require_agent", lambda *a: None)
    monkeypatch.setattr(A, "require_agent_v2", lambda *a: None)
    agentic_entry = "execute_swebench_agentic"
    if suite == "swe_bench_live":
        # its Python tabs (the multi-language and Windows tabs pin other datasets); the agentic
        # entry runs SWE-agent-for-eval
        monkeypatch.setattr(mod, "SETTINGS", mod.PYTHON_SETTINGS)
        monkeypatch.setattr(mod, "check_sweagent_prerequisites", lambda: (True, ""))
        agentic_entry = "execute_sweagent_python"
    for mode, entry in (("agentic", agentic_entry), ("oneshot", "execute_swebench")):
        seen = _stub_agentic_entry(monkeypatch, mod, entry)
        monkeypatch.setenv(A.mode_env_var(suite), mode)
        mod.__dict__[f"run_{suite}"]("m", "http://x/v1", 1)
        assert seen["kwargs"]["dataset_revision"] == revision and seen["args"][4] == dataset
    # the sample helper used by tests/one-shot callers loads the same commit (the fake rows
    # look like the pinned split: its row count and the columns its harness reads)
    split = getattr(mod, "_SPLIT", None) or mod._DEFAULT_SPLIT
    fake_rows = [{**{c: "x" for c in mod.SNAPSHOT_COLUMNS}, "instance_id": f"a__a-{i}",
                  "problem_statement": "p", "repo": "a/a"} for i in range(mod.SPLIT_ROWS[split])]
    # stand-in rows are not the pinned content, so they need their own fingerprint
    monkeypatch.setitem(mod.SPLIT_SHA256, split, C.snapshot_sha256(fake_rows))
    calls = _fake_datasets(monkeypatch, fake_rows)
    helper = [f for n, f in vars(mod).items() if n.startswith("_load_") and callable(f)][0]
    helper(limit=1)
    assert calls[0][2]["revision"] == revision


@pytest.mark.parametrize("suite", ["copilot_bench_swe", "swe_bench_multilingual"])
def test_prereq_checks_the_pinned_snapshot(monkeypatch, suite):
    import importlib
    mod = importlib.import_module(f"gbench.runners.eval_suites.{suite}")
    check = getattr(mod, f"check_{suite}_prerequisites")
    monkeypatch.setattr(mod, "pinned_swe_snapshot",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Could not load dataset")))
    ok, reason = check()
    assert not ok and "Could not load dataset" in reason
    seen = {}
    monkeypatch.setattr(mod, "pinned_swe_snapshot",
                        lambda d, s, r, e, **k: seen.update(k=k) or f"/snap/{r}/{s}.jsonl")
    monkeypatch.setattr(mod, "check_swebench_v5_prereq",
                        lambda d, s: seen.update(d=d) or (True, ""))
    assert check() == (True, "")
    assert seen["d"] == f"/snap/{mod._REVISION}/test.jsonl"
    # the snapshot is checked against the v5 columns, the split's row count and its content
    assert seen["k"] == {"columns": C.SWEBENCH_V5_COLUMNS, "rows": mod.SPLIT_ROWS["test"],
                         "sha256": mod.SPLIT_SHA256["test"]}


def test_fork_image_scorer_mounts_the_snapshot(monkeypatch, tmp_path):
    snap = tmp_path / "snap" / "lite.jsonl"
    snap.parent.mkdir()
    snap.write_text(json.dumps({"instance_id": "a__a-1"}) + "\n")
    captured = {}
    monkeypatch.setattr(C.subprocess, "run", lambda cmd, **kw: captured.update(cmd=list(cmd))
                        or types.SimpleNamespace(returncode=0, stdout="", stderr=""))
    scorer = C.make_swebench_scorer("swe_bench_live", "m", str(snap), "lite", "starryzhang", 1,
                                    {}, harness_image="gbench-swe-bench-live:a145aa87")
    asyncio.run(scorer([{"extra_payload": {"instance_id": "a__a-1"}, "agent_patch": "d\n"}]))
    cmd = captured["cmd"]
    assert f"{snap.parent}:{snap.parent}:ro" in cmd
    assert cmd[cmd.index("--dataset_name") + 1] == str(snap)
    assert cmd.index(f"{snap.parent}:{snap.parent}:ro") < cmd.index("gbench-swe-bench-live:a145aa87")


# --- swe_bench_pro: pinned v1 set, tied raw-sample table, correct guards ---------------------
_PRO_COLS = ("instance_id", "base_commit", "before_repo_set_cmd", "selected_test_files_to_run",
             "fail_to_pass", "pass_to_pass")


def _pro_row(n, **extra):
    row = {c: f"{c}-{n}" for c in _PRO_COLS}
    row.update(instance_id=f"instance_o__r-{n}", repo="o/r", problem_statement=f"P{n}",
               requirements="R", interface="I", repo_language="go", dockerhub_tag=f"o.r-{n}",
               fail_to_pass="['t1']", pass_to_pass="['t2', \"t,3\"]",
               before_repo_set_cmd="git reset --hard x\ngit checkout x -- a.py")
    row.update(extra)
    return row


def test_pro_every_loader_reads_the_pinned_v1_revision(monkeypatch, tmp_path):
    """The engine-facing v1 table reads the pinned v1 revision; the default V2 path reads V2."""
    from gbench.runners.eval_suites import swe_bench_pro as P
    assert P._V1_REVISION == "307bd006a0a3807424d3c5525eaf98ca8e63aed3"
    assert P._V1_CONFIG == "default"
    assert P._REVISION == "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50" and P._CONFIG == "default"
    calls = _fake_datasets(monkeypatch, [_pro_row(1), _pro_row(2)])
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_HARNESS_DIR", str(tmp_path))
    monkeypatch.delenv("GBENCH_SWE_BENCH_PRO_RAW_SAMPLE", raising=False)
    monkeypatch.delenv("SWE_BENCH_PRO_RAW_SAMPLE", raising=False)
    P.raw_sample_path()
    assert calls == [(P._DATASET, ("default",), {"split": "test", "revision": P._V1_REVISION})]
    calls.clear()
    with pytest.raises(RuntimeError, match="expected 642"):
        P._load_swe_bench_pro_samples()
    assert calls == [(P._DATASET, ("default",), {"split": "test", "revision": P._REVISION})]


def test_pro_raw_sample_table_is_tied_to_the_pinned_revision(monkeypatch, tmp_path):
    import pandas as pd
    from gbench.runners.eval_suites import swe_bench_pro as P
    rows = [_pro_row(1), _pro_row(2)]
    _fake_datasets(monkeypatch, rows)
    hd = tmp_path / "harness"
    hd.mkdir()
    legacy = hd / "swe_bench_pro_full.csv"        # a pre-pin (or V2) table
    legacy.write_text("instance_id,base_commit\nv2-only,abc\n")
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_HARNESS_DIR", str(hd))
    monkeypatch.delenv("GBENCH_SWE_BENCH_PRO_RAW_SAMPLE", raising=False)
    monkeypatch.delenv("SWE_BENCH_PRO_RAW_SAMPLE", raising=False)
    path = P.raw_sample_path()
    assert path == str(hd / "swe_bench_pro_full.v1.0-307bd006.csv") == str(hd / P.RAW_SAMPLE_NAME)
    assert "v2-only" in legacy.read_text()        # never used, never touched
    # what the harness reads (pd.read_csv + fillna, then eval) is the pinned rows
    df = pd.read_csv(path).fillna("").set_index("instance_id", drop=False)
    for r in rows:
        for c in _PRO_COLS:
            assert df.loc[r["instance_id"]][c] == r[c]
    assert eval(df.loc["instance_o__r-1"]["pass_to_pass"]) == ["t2", "t,3"]
    # a table under the pinned name that differs from the pinned rows is rewritten
    pd.DataFrame([_pro_row(1)]).to_csv(path, index=False)
    assert P.raw_sample_path() == path and len(pd.read_csv(path)) == 2
    # the operator's own table is checked, never rewritten
    mine = tmp_path / "mine.csv"
    pd.DataFrame([_pro_row(1), _pro_row(3)]).to_csv(mine, index=False)
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_RAW_SAMPLE", str(mine))
    got, why = P._resolve_raw_sample()
    assert got is None and "different instance set" in why and "307bd006" in why
    pd.DataFrame(rows).to_csv(mine, index=False)
    assert P.raw_sample_path() == str(mine)


def test_pro_prereq_needs_no_opt_in_and_checks_the_table(monkeypatch, tmp_path):
    from gbench.runners.eval_suites import swe_bench_pro as P
    (tmp_path / "swe_bench_pro_eval.py").write_text("")
    (tmp_path / "run_scripts").mkdir()
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_HARNESS_DIR", str(tmp_path))
    monkeypatch.delenv("GBENCH_SWE_BENCH_PRO_RUN", raising=False)
    monkeypatch.delenv("SWE_BENCH_PRO_RUN", raising=False)
    monkeypatch.setattr(P.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(P.subprocess, "run", lambda *a, **k: types.SimpleNamespace(returncode=0))
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))
    monkeypatch.setattr(P, "load_pinned_rows",
                        lambda: (_ for _ in ()).throw(RuntimeError("offline")))
    ok, reason = P.check_swe_bench_pro_prerequisites()
    assert not ok and "raw-sample table" in reason and "RUN" not in reason
    monkeypatch.setattr(P, "load_pinned_rows", lambda: [_pro_row(1)])
    assert P.check_swe_bench_pro_prerequisites() == (True, "")


def _pro_harness(monkeypatch, tmp_path, P, results, stderr=""):
    def _run(cmd, **kw):
        out = [a.split("=", 1)[1] for a in cmd if str(a).startswith("--output_dir=")][0]
        with open(f"{out}/eval_results.json", "w") as f:
            json.dump(results, f)
        return types.SimpleNamespace(returncode=0, stdout="", stderr=stderr)
    monkeypatch.setattr(P, "_harness_dir", lambda: str(tmp_path))
    monkeypatch.setattr(P, "raw_sample_path", lambda: str(tmp_path / "t.csv"))
    monkeypatch.setattr(P.subprocess, "run", _run)
    monkeypatch.setattr(P.swe_thread_cap, "apply", lambda c, n, e: (c, 0))


def _pro_traces(n, agentic, patch=""):
    out = []
    for k in range(n):
        tr = {"extra_payload": {"instance_id": f"i{k}"}, "response_text": patch}
        if agentic:
            tr["agent_patch"] = patch
        out.append(tr)
    return out


def test_pro_agentic_empty_submissions_are_a_zero_not_an_error(monkeypatch, tmp_path):
    """All-LimitsExceeded is the model's recorded outcome (0%), as in the swebench suites; the
    'no extractable patch' error stays a one-shot guard."""
    from gbench.runners.eval_suites import swe_bench_pro as P
    _pro_harness(monkeypatch, tmp_path, P, {f"i{k}": False for k in range(3)})
    metrics = {}
    asyncio.run(P._make_scorer("m", 1, metrics)(_pro_traces(3, agentic=True)))
    assert "swe_bench_pro_error" not in metrics
    res = {"status": "success"}
    assert P.report_reasons(res, metrics) == [] and res["status"] == "success"
    metrics = {}
    asyncio.run(P._make_scorer("m", 1, metrics)(_pro_traces(3, agentic=False)))
    assert "no extractable patch" in metrics["swe_bench_pro_error"]


def test_pro_instances_the_harness_did_not_evaluate_are_not_zeros(monkeypatch, tmp_path):
    from gbench.runners.eval_suites import swe_bench_pro as P
    patch = "diff --git a/x b/x\n"
    _pro_harness(monkeypatch, tmp_path, P, {"i0": True, "i1": False})
    metrics = {}
    asyncio.run(P._make_scorer("m", 1, metrics)(_pro_traces(3, agentic=True, patch=patch)))
    assert metrics["swe_bench_pro_report"]["not_evaluated_ids"] == ["i2"]
    res = {"status": "success"}
    reasons = P.report_reasons(res, metrics)
    assert res["status"] == "completed_with_errors"
    assert res["partial_evaluation"]["harness_not_evaluated_instances"] == 1
    assert reasons == ["partial: 1 of 3 submitted instance(s) were not evaluated by the harness"]
    _pro_harness(monkeypatch, tmp_path, P, {})
    metrics = {}
    asyncio.run(P._make_scorer("m", 1, metrics)(_pro_traces(2, agentic=True, patch=patch)))
    res = {"status": "success"}
    assert P.report_reasons(res, metrics) == ["harness error (no valid resolved rate)"]
    assert res["status"] == "error" and "evaluated none of the 2" in res["error"]


def test_pro_scorer_refuses_to_run_without_the_table(monkeypatch, tmp_path):
    from gbench.runners.eval_suites import swe_bench_pro as P
    monkeypatch.setattr(P, "_harness_dir", lambda: str(tmp_path))
    monkeypatch.setattr(P, "raw_sample_path", lambda: None)
    monkeypatch.setattr(P.subprocess, "run", lambda *a, **k: pytest.fail("harness ran"))
    metrics = {}
    asyncio.run(P._make_scorer("m", 1, metrics)(_pro_traces(1, agentic=True, patch="d\n")))
    assert "raw-sample table unavailable" in metrics["swe_bench_pro_error"]


def test_pro_oneshot_reason_names_the_v1_pin_and_harness_verdict(monkeypatch):
    """One-shot is re-graded by the V2 patch_replay job; tasks without a verdict are partial."""
    from gbench.runners.eval_suites import swe_bench_pro as P
    monkeypatch.setattr(P, "check_swe_bench_pro_v2_prerequisites", lambda: (True, ""))
    monkeypatch.delenv("GBENCH_SWEBENCH_PRO_AGENTIC", raising=False)
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_MODE", "oneshot")
    monkeypatch.setattr(P, "_load_swe_bench_pro_samples", lambda limit=None: [])

    monkeypatch.setattr(P, "run_eval_suite", lambda **k: {"status": "success", "temperature": 0.0})
    # the re-grade's verdict: 3 of the 4 submitted tasks got a reward
    monkeypatch.setattr(P, "_make_v2_oneshot_scorer", lambda n, metrics: metrics.update(
        swe_bench_pro_report={"submitted_instances": 4, "not_regraded_ids": ["i3"]}))
    out = P.run_swe_bench_pro("gemma", "http://127.0.0.1:9/v1", concurrency=1)
    reason = out["leaderboard_comparable_reason"]
    assert "V2 locked" in reason and "partial: 1 of 4" in reason
    assert out["status"] == "completed_with_errors" and out["dataset"] == P.V2_DATASET_LABEL


def test_pro_leaderboard_note_names_scales_actual_scaffolds():
    """The default runs V2's reference mini-swe-agent entry and re-grade (v2/README.md)."""
    from gbench.runners.eval_suites import swe_bench_pro as P
    assert P.AGENT_IMPORT == "locked_mini_swe:LockedMiniSwe"
    assert P.AGENT_CONFIG == "v2/tooling/configs/mini_toolcall.yaml"
    assert P.REGRADE_IMPORT == "patch_replay:PatchReplayAgent"


# --- multi_swe_bench: harness errors are never a 0% ------------------------------------------
def _mswe_run(monkeypatch, tmp_path, report, patches=None, mode="agentic"):
    """Run the suite end to end with stubbed rollouts / generation and a fake harness report."""
    from gbench.runners.eval_suites import multi_swe_bench as M
    ids = [1, 2, 3]
    patches = patches or {n: "diff --git a/x b/x\n" for n in ids}
    src = tmp_path / "src.jsonl"
    src.write_text("".join(json.dumps({"org": "o", "repo": "r", "number": n,
                                       "instance_id": f"o__r-{n}"}) + "\n" for n in ids))
    monkeypatch.setattr(M, "_SRC_FILES", {f"o__r-{n}": str(src) for n in ids})
    monkeypatch.setenv("GBENCH_MULTI_SWE_REPO_DIR", str(tmp_path / "repos"))
    monkeypatch.setattr(M, "check_multi_swe_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M, "check_mswe_agent_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M.swe_thread_cap, "apply", lambda c, n, e: (c, 0))

    def _harness(cmd, **kw):
        cfg = json.load(open(cmd[-1]))
        with open(f"{cfg['output_dir']}/final_report.json", "w") as f:
            json.dump(report, f)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(M.subprocess, "run", _harness)
    tasks = [A.AgentTask(f"o__r-{n}", "p", "img", "go", cwd="/home/r",
                         meta={"org": "o", "repo": "r", "number": n}) for n in ids]
    monkeypatch.setenv("GBENCH_MULTI_SWE_BENCH_MODE", mode)
    if mode == "agentic":
        monkeypatch.setattr(M, "_load_multi_swe_bench_tasks",
                            lambda: (tasks, {t.instance_id: {} for t in tasks}))

        def _rollouts(tasks, rows, model_name, relay_url, concurrency, out_dir, temperature,
                      top_p):
            out = []
            for idx, task in enumerate(tasks):
                n = task.meta["number"]
                out.append({"sample_idx": idx, "category": "go",
                            "extra_payload": {"instance_id": task.instance_id, **task.meta},
                            "agent_patch": patches[n], "response_text": patches[n],
                            "agent_exit_status": "submitted" if patches[n] else "exit_cost",
                            "agent_error": None, "agent_infra_error": False, "agent_steps": 1,
                            "trajectory": [], "truncated_turns": 0, "is_correct": False,
                            "status": "OK"})
            return out
        monkeypatch.setattr(M, "run_mswe_agent", _rollouts)
    else:
        monkeypatch.setattr(M, "_load_multi_swe_bench_samples", lambda limit=None: [])

        def _suite(**k):
            traces = [{"extra_payload": {"instance_id": f"o__r-{n}", "org": "o", "repo": "r",
                                         "number": n},
                       "response_text": f"```diff\n{patches[n]}```" if patches[n] else "prose"}
                      for n in ids]
            asyncio.run(k["async_eval_fn"](traces))
            correct = sum(t["is_correct"] for t in traces)
            return {"status": "success", "total_questions": 3, "correct_answers": correct,
                    "accuracy": round(correct / 3 * 100, 2), "sample_traces": traces}
        monkeypatch.setattr(M, "run_eval_suite", _suite)
    return M.run_multi_swe_bench("m", "http://127.0.0.1:9/v1", 1)


def _final_report(resolved=(), unresolved=(), errors=()):
    hid = lambda n: f"o/r:pr-{n}"   # noqa: E731  (the harness's PullRequestBase.id)
    return {"total_instances": len(resolved) + len(unresolved) + len(errors),
            "submitted_instances": len(resolved) + len(unresolved) + len(errors),
            "completed_instances": len(resolved) + len(unresolved),
            "incomplete_instances": len(errors), "resolved_instances": len(resolved),
            "unresolved_instances": len(unresolved), "empty_patch_instances": 0,
            "error_instances": len(errors),
            "resolved_ids": [hid(n) for n in resolved], "unresolved_ids": [hid(n) for n in unresolved],
            "completed_ids": [hid(n) for n in (*resolved, *unresolved)],
            "incomplete_ids": [hid(n) for n in errors], "error_ids": [hid(n) for n in errors],
            "submitted_ids": [], "empty_patch_ids": []}


@pytest.mark.parametrize("mode", ["agentic", "oneshot"])
def test_multi_swe_bench_all_harness_errors_is_an_error(monkeypatch, tmp_path, mode):
    res = _mswe_run(monkeypatch, tmp_path, _final_report(errors=(1, 2, 3)), mode=mode)
    assert res["status"] == "error" and "not a 0% resolved rate" in res["error"]
    assert "3 errored, 0 missing" in res["error"]
    assert "harness infrastructure failure (no scoreable instances)" in \
        res["leaderboard_comparable_reason"]
    rep = res["multi_swe_bench_report"]
    assert rep["error_ids"] == ["o/r:pr-1", "o/r:pr-2", "o/r:pr-3"]
    assert rep["incomplete_ids"] == rep["error_ids"]


@pytest.mark.parametrize("mode", ["agentic", "oneshot"])
def test_multi_swe_bench_some_harness_errors_is_partial(monkeypatch, tmp_path, mode):
    # instance 2 errored, instance 3 is missing from the report altogether (image never built)
    res = _mswe_run(monkeypatch, tmp_path, _final_report(resolved=(1,), errors=(2,)), mode=mode)
    assert res["status"] == "completed_with_errors" and res["correct_answers"] == 1
    assert res["partial_evaluation"]["harness_unscored_instances"] == 2
    assert res["partial_evaluation"]["harness_missing_instances"] == 1
    assert res["multi_swe_bench_report"]["harness_missing_ids"] == ["o__r-3"]
    assert "partial: 2 of 3 submitted instance(s) hit harness errors" in \
        res["leaderboard_comparable_reason"]
    assert res["leaderboard_comparable"] is False
    if mode == "oneshot":
        assert res["leaderboard_comparable_reason"].startswith("one-shot mode")


def test_multi_swe_bench_empty_submissions_are_the_models_outcome(monkeypatch, tmp_path):
    """An empty submission without a verdict counts as unresolved (as swebench counts an empty
    patch), so it neither errors the run nor makes it partial."""
    res = _mswe_run(monkeypatch, tmp_path, _final_report(unresolved=(1,), errors=(2, 3)),
                    patches={1: "diff --git a/x b/x\n", 2: "", 3: ""})
    assert res["status"] == "success" and "partial_evaluation" not in res
    assert res["multi_swe_bench_report"]["empty_submission_instances"] == 2
    all_empty = _mswe_run(monkeypatch, tmp_path, _final_report(errors=(1, 2, 3)),
                          patches={1: "", 2: "", 3: ""})
    assert all_empty["status"] == "success" and all_empty["accuracy"] == 0.0


def test_multi_swe_bench_clean_report_is_success(monkeypatch, tmp_path):
    res = _mswe_run(monkeypatch, tmp_path, _final_report(resolved=(1, 2), unresolved=(3,)))
    assert res["status"] == "success" and res["correct_answers"] == 2
    assert "partial_evaluation" not in res
