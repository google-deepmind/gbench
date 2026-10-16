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

"""tau2 / tau3: a simulation tau2 ends with USER_ERROR (the Gemini user simulator broke the
message protocol) is a harness error, left out of pass^k; a proxy's credentials never enter the
container (the agent's requests then go through gbench's endpoint relay); the roles' source is the
host's cascade source; a merge unites the shards' selected task ids. tau2, Docker, Gemini and the
network are faked; nothing is written outside tmp."""

import os
import sys
import types
from unittest import mock

import pytest

from gbench.runners.eval_suites import gemini_relay as GR
from gbench.runners.eval_suites import tau_common as TC

_TAU_ENV = ("GBENCH_TAU2_TEMPERATURE", "TAU2_TEMPERATURE", "GBENCH_TAU3_TEMPERATURE",
            "GBENCH_TAU2_TOP_P", "TAU2_TOP_P", "GBENCH_TAU2_TOP_K", "TAU2_TOP_K",
            "GBENCH_TAU2_USER_LLM", "TAU2_USER_LLM", "GBENCH_TAU2_EVAL_LLM", "TAU2_EVAL_LLM",
            "GBENCH_TAU2_NUM_TRIALS", "TAU2_NUM_TRIALS", "GBENCH_TAU2_USER_TEMPERATURE",
            "TAU2_USER_TEMPERATURE", "GBENCH_TAU2_SAVE_TRACES", "TAU2_SAVE_TRACES",
            "GBENCH_RESULTS_DIR")


def _sim(reward, reason="agent_stop", task="t1"):
    return types.SimpleNamespace(reward_info=types.SimpleNamespace(reward=reward),
                                 termination_reason=reason, task_id=task)


def _tau_modules(sims):
    class TextRunConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def run_domain(config):
        os.makedirs(config.save_to, exist_ok=True)
        return types.SimpleNamespace(simulations=list(sims), num_trials=config.num_trials)

    def compute_metrics(results):
        scored = [s for s in results.simulations if s.termination_reason != "infrastructure_error"]
        rate = sum(1.0 for s in scored if s.reward_info.reward >= 1.0) / len(scored)
        return types.SimpleNamespace(pass_hat_ks={1: rate})

    mods = {n: types.ModuleType(n) for n in (
        "tau2", "tau2.data_model", "tau2.data_model.simulation", "tau2.run", "tau2.metrics",
        "tau2.metrics.agent_metrics")}
    mods["tau2.data_model.simulation"].TextRunConfig = TextRunConfig
    mods["tau2.run"].run_domain = run_domain
    mods["tau2.metrics.agent_metrics"].compute_metrics = compute_metrics
    return mods


@pytest.fixture
def tau(monkeypatch, tmp_path):
    for k in _TAU_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GBENCH_RESULTS_DIR", str(tmp_path / "results"))
    for name, value in (("check_tau_env_prerequisites", lambda: (True, "")),
                        ("check_banking_prerequisites", lambda: (True, "")),
                        ("_install_robustness_patches", lambda *a, **k: None),
                        ("_install_credential_guards", lambda *a, **k: None),
                        ("_configure_evaluator_llm", lambda *a, **k: None),
                        ("_quiet_tau2_noise", lambda: None),
                        ("_prune_stale_search_sandboxes", lambda: None),
                        ("_refuse_resume_prompts", lambda: None),
                        ("_tau2_version", lambda: "1.0.1")):
        monkeypatch.setattr(TC, name, value)
    sims = [_sim(1.0, task="t1"), _sim(1.0, task="t2"), _sim(0.0, task="t3"),
            _sim(0.0, "user_error", task="t3")]
    with mock.patch.dict(sys.modules, _tau_modules(sims)):
        yield sims


def test_a_user_error_simulation_is_left_out_of_pass_k(tau):
    TC._DOMAIN_USER_ERRORS.clear()
    total, reward, perfect, infra = TC._run_one_domain("airline", "m", "http://x/v1", 2, None,
                                                      False, "tau2")
    assert (total, reward, perfect, infra) == (4, 2.0, 2, 0)
    assert TC._DOMAIN_USER_ERRORS["airline"] == 1
    assert TC._DOMAIN_PASS_HAT_K["airline"]["pass^1"] == pytest.approx(2 / 3)


def test_a_run_counts_and_names_user_simulator_protocol_errors(tau):
    res = TC.run_tau_env("tau2", ["airline"], "m", "http://x/v1", 2, None, "docs/evals/tau2.md")
    assert res["status"] == "completed_with_errors"
    assert res["total_questions"] == 3 and res["accuracy"] == pytest.approx(66.67)
    assert res["category_accuracy"]["airline"]["user_simulator_protocol_errors"] == 1
    assert res["tau2_report"]["user_simulator_protocol_errors"] == 1
    [rec] = [r for r in res["harness_errors"] if r.get("kind") == "user_simulator_protocol"]
    assert rec["domain"] == "airline" and rec["simulations"] == 1
    assert "1 simulation(s) the Gemini user simulator ended with a protocol error" in \
        res["leaderboard_comparable_reason"]


def test_the_result_classifiers():
    assert TC._is_user_error(_sim(0.0, "user_error"))
    assert TC._is_user_error(_sim(0.0, types.SimpleNamespace(value="user_error")))
    assert not TC._is_user_error(_sim(0.0, "agent_stop"))


# --------------------------------------------------------------------------- #
# The proxy's credentials stay on the host.
# --------------------------------------------------------------------------- #
def test_proxy_credentials_never_enter_the_container():
    env = {"HTTPS_PROXY": "http://alice:s3cret@proxy.corp:3128", "http_proxy": "http://127.0.0.1:8"}
    out = TC.container_proxy_env(env)
    assert out["HTTPS_PROXY"] == "http://proxy.corp:3128"
    assert out["http_proxy"] == f"http://{TC.HOST_ALIAS}:8"
    assert "s3cret" not in repr(out) and "alice" not in repr(out)
    assert TC.proxy_credentials(env) == ["HTTPS_PROXY"]
    assert TC.proxy_credentials({"HTTP_PROXY": "http://proxy:1"}) == []


class _Relay:
    def __init__(self, eval_name, bind, **kw):
        self.origin, self.placeholder = f"http://{bind}:4567", "gbench-gemini-relay-" + "ab" * 24

    def start(self):
        return self

    def stop(self):
        pass

    def record(self):
        return {}


class _EndpointRelay:
    made = []

    def __init__(self, upstream, bind, key=None, name="", **kw):
        self.upstream, self.key, self.url = upstream, key, f"http://{bind}:4568/v1"
        _EndpointRelay.made.append(self)

    def start(self):
        return self

    def stop(self):
        pass

    def record(self):
        return {}


def test_a_credentialed_proxy_sends_the_agent_through_the_endpoint_relay(monkeypatch):
    from gbench.runners.eval_suites import swebench_common
    _EndpointRelay.made = []
    monkeypatch.setattr(TC, "_require_gemini_access", lambda *a: None)
    monkeypatch.setattr(GR, "network_gateway", lambda *a: "172.17.0.1")
    monkeypatch.setattr(GR, "GeminiRelay", _Relay)
    monkeypatch.setattr(swebench_common, "EndpointRelay", _EndpointRelay)
    monkeypatch.setattr(TC, "_require_relay_reachable", lambda *a: None)
    for k in ("VLLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "x" * 35)
    monkeypatch.setenv("HTTPS_PROXY", "http://alice:s3cret@proxy.corp:3128")
    with TC._bridge_services("tau2", "img", "http://10.0.0.5:9/v1", "http://10.0.0.5:9/v1",
                             ["airline"], "docs/evals/tau2.md") as bridge:
        [relay] = _EndpointRelay.made
        assert relay.key is None and bridge.agent_base_url == relay.url
        assert "s3cret" not in repr(bridge.env)
        assert bridge.record()["proxy_credentials_withheld"] == ["HTTPS_PROXY"]
    launch = TC._launch_env({"HTTPS_PROXY": "http://alice:s3cret@proxy.corp:3128"},
                            bridge_env=bridge.env)
    assert "s3cret" not in repr(launch)


# --------------------------------------------------------------------------- #
# The roles' source, and the merged task ids.
# --------------------------------------------------------------------------- #
def test_the_roles_source_is_the_hosts_in_the_image(monkeypatch):
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-3.8-flash")
    monkeypatch.setenv(TC.CASCADE_SOURCE_ENV, "default")
    monkeypatch.setattr(TC, "_in_image", lambda: True)
    assert TC._cascade_from(os.environ)[1] == "default"
    monkeypatch.setattr(TC, "_in_image", lambda: False)      # never trusted on the host
    assert TC._cascade_from(os.environ)[1] == "GBENCH_JUDGE_MODELS"
    monkeypatch.setattr(TC, "_in_image", lambda: True)
    monkeypatch.setenv(TC.CASCADE_SOURCE_ENV, "made-up")
    assert TC._cascade_from(os.environ)[1] == "GBENCH_JUDGE_MODELS"


def test_a_merge_unites_the_shards_selected_task_ids():
    def shard(label, ids):
        return {"shard": label, "domains_requested": ["airline"], "selected_task_ids": ids,
                "status": "success", "tau2_report": {
                    "task_outcomes_by_domain": {"airline": {i: [1, 1] for i in ids["airline"]}},
                    "user_simulator_protocol_errors_by_domain": {"airline": 1}}}
    m = TC.merge_tau_shards([shard("1/2", {"airline": ["3", "1"]}),
                             shard("2/2", {"airline": ["2"]})])
    assert m["selected_task_ids"] == {"airline": ["1", "2", "3"]}
    assert m["tau2_report"]["user_simulator_protocol_errors"] == 2
