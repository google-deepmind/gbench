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
"""SWE family on the current upstream protocols.

* copilot_bench_swe / swe_bench_multilingual: the maintainers' current datasets (v5 columns),
  mini-swe-agent 2.4.2 / 2.4.6 in the local 2.x image, the swebench 5.0.2 harness image.
* swe_bench_pro: SWE-bench Pro V2 under Harbor's locked protocol (agent job + patch_replay
  re-grade), with the v1 harness interface kept for the separately maintained engine.

Hermetic: no container, no endpoint, no network. The 2.x driver is loaded against stub
mini-swe-agent modules.
"""

import asyncio
import importlib.util
import json
import os
import sys
import types

import pytest

from gbench.runners.eval_suites import copilot_bench_swe as CB
from gbench.runners.eval_suites import swe_agentic as A
from gbench.runners.eval_suites import swe_bench_multilingual as ML
from gbench.runners.eval_suites import swe_bench_pro as P
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
DOCKER_DIR = os.path.join(REPO, "gbench", "docker")


@pytest.fixture(autouse=True)
def _scratch(monkeypatch, tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    for var in ("GBENCH_SWE_BENCH_PRO_TASK_ENDPOINT", "GBENCH_SWE_BENCH_PRO_TASK_HOST",
                "GBENCH_SWE_BENCH_PRO_IMAGE", "GBENCH_MINI_SWE_AGENT_V2_IMAGE",
                "GBENCH_SWEBENCH_V5_IMAGE", "GBENCH_SWE_OMP_THREADS", "GBENCH_SHARD",
                "GBENCH_SWEBENCH_PRO_AGENTIC", "GBENCH_SWE_BENCH_PRO_MODE",
                "GBENCH_COPILOT_BENCH_SWE_TEMPERATURE", "GBENCH_SWE_BENCH_PRO_TEMPERATURE"):
        monkeypatch.delenv(var, raising=False)


def _proc(rc=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


# ============================================================================================
# pins
# ============================================================================================
def test_verified_and_multilingual_pin_the_current_datasets_and_releases():
    assert (CB._DATASET, CB._REVISION) == ("SWE-bench/SWE-bench_Verified",
                                           "78f471bf655a3137b2e8a75af1501690ec009ec3")
    assert (ML._DATASET, ML._REVISION) == ("SWE-bench/SWE-bench_Multilingual",
                                           "846e647b9f33c0b51b739d005d13d85493c9af09")
    assert CB.AGENT_VERSION == "2.4.2" and ML.AGENT_VERSION == "2.4.6"
    assert A.AGENT_V2_CONFIG_SHA256 == {
        "2.4.2": "229f178a07faa109e3a020c75dfe603b6a24200e1f3aebe257b97098fd3a6fee",
        "2.4.6": "9a9c86ac10428b86b932c972b15fefc2f7b6e92230bac5ebdc47e83232a8315e"}
    assert C.SWEBENCH_V5_VERSION == "5.0.2"


def test_pro_pins_v2_and_keeps_the_v1_table_for_the_engine():
    assert P._REVISION == "2d52cb3df914a3fcf80c7f66738b3a88ae37fc50"
    assert (P._CONFIG, P._HARD_CONFIG) == ("default", "hard")
    assert (P.V2_TASK_COUNT, P.HARD_TASK_COUNT) == (642, 51)
    assert P.HARNESS_COMMIT == "66f92766bba642462d4bbe5479e83f91f9211862"
    assert P.HARBOR_VERSION == "0.22.0" and P.MINI_SWE_AGENT_VERSION == "2.4.6"
    assert P.AGENT_CONFIG_SHA256 == ("7275afa9920a6fb1b9ec66ab74cee44121db9b0276bb9fa90f5ec0"
                                     "597e523a45")
    assert P._V1_REVISION == "307bd006a0a3807424d3c5525eaf98ca8e63aed3"
    assert P.RAW_SAMPLE_NAME == "swe_bench_pro_full.v1.0-307bd006.csv"
    assert P.DATASET_LABEL == P.V1_DATASET_LABEL and "V2" in P.V2_DATASET_LABEL


def test_dockerfiles_carry_the_pinned_versions():
    msa = open(os.path.join(DOCKER_DIR, "mini_swe_agent_v2.Dockerfile")).read()
    assert 'ARG MINI_SWE_AGENT_VERSIONS="2.4.2 2.4.6"' in msa
    assert "org.gbench.mini_swe_agent_versions" in msa
    assert "COPY mini_swe_agent_v2_run.py /opt/gbench/mini_swe_agent_v2_run.py" in msa
    assert {CB.AGENT_VERSION, ML.AGENT_VERSION} <= set("2.4.2 2.4.6".split())
    v5 = open(os.path.join(DOCKER_DIR, "swebench_v5.Dockerfile")).read()
    assert f"ARG SWEBENCH_VERSION={C.SWEBENCH_V5_VERSION}" in v5 and "org.gbench.swebench" in v5
    pro = open(os.path.join(DOCKER_DIR, "swe_bench_pro_v2.Dockerfile")).read()
    assert f"ARG HARBOR_VERSION={P.HARBOR_VERSION}" in pro
    assert f"ARG SWE_BENCH_PRO_OS_COMMIT={P.HARNESS_COMMIT}" in pro
    assert "sha256sum --quiet -c SHA256SUMS" in pro
    assert "docker-compose-plugin" in pro and "docker-buildx-plugin" in pro
    for name in ("mini_swe_agent_v2.Dockerfile", "mini_swe_agent_v2_run.py",
                 "swebench_v5.Dockerfile", "swe_bench_pro_v2.Dockerfile"):
        open(os.path.join(DOCKER_DIR, name), encoding="ascii").read()


# ============================================================================================
# mini-swe-agent 2.x runner (swe_agentic)
# ============================================================================================
def test_v2_prereq_refuses_unknown_release_missing_image_and_wrong_label(monkeypatch):
    ok, why = A.check_agent_v2_prerequisites("9.9.9")
    assert not ok and "9.9.9" in why
    monkeypatch.setattr(A.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(A.os.path, "exists", lambda p: True)
    labels = {"org.gbench.mini_swe_agent_versions": "2.4.2"}

    def _run(cmd, **kw):
        if cmd[:3] == ["docker", "image", "inspect"]:
            return _proc(0 if labels is not None else 1, json.dumps(labels))
        return _proc()
    monkeypatch.setattr(A.subprocess, "run", _run)
    monkeypatch.setattr(A, "bridge_gateway", lambda: ("172.17.0.1", ""))
    ok, why = A.check_agent_v2_prerequisites("2.4.6")
    assert not ok and "not 2.4.6" in why and "docker build -t gbench-mini-swe-agent-v2:2.4.2-2.4.6" in why
    assert "\n" not in why
    labels = {"org.gbench.mini_swe_agent_versions": "2.4.2 2.4.6"}
    assert A.check_agent_v2_prerequisites("2.4.6") == (True, "")
    labels = None
    ok, why = A.check_agent_v2_prerequisites("2.4.6")
    assert not ok and "is not built" in why


def test_v2_overrides_send_only_the_endpoint_and_gbench_knobs():
    base = {"environment": {"env": {"PAGER": "cat", "OMP_NUM_THREADS": "2"}}}
    ov = A.build_agent_v2_overrides(base, model_name="google/gemma", base_url="http://h:1",
                                    temperature=1.0, thinking=True, max_output_tokens=4096,
                                    threads=4)
    kw = ov["model"]["model_kwargs"]
    assert ov["model"]["model_name"] == "openai/google/gemma"
    assert ov["model"]["cost_tracking"] == "ignore_errors"
    assert kw == {"api_base": "http://h:1/v1", "max_tokens": 4096, "temperature": 1.0,
                  "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
    env = ov["environment"]
    assert env["container_timeout"] == A.CONTAINER_TIMEOUT and env["pull_timeout"] == 3600
    assert env["env"]["OMP_NUM_THREADS"] == "2" and env["env"]["RAYON_NUM_THREADS"] == "4"
    assert env["env"]["PAGER"] == "cat"
    ov = A.build_agent_v2_overrides({}, model_name="m", base_url="http://h:1/v1",
                                    temperature=0.3, thinking=False, max_output_tokens=None,
                                    penalties={"repetition_penalty": 1.1}, threads=0)
    kw = ov["model"]["model_kwargs"]
    assert kw["temperature"] == 0.3 and "max_tokens" not in kw and "api_key" not in kw
    assert kw["extra_body"]["repetition_penalty"] == 1.1
    assert "env" not in ov["environment"]
    merged = A._merged({"model": {"model_kwargs": {"drop_params": True}}, "agent": {"a": 1}}, ov)
    assert merged["model"]["model_kwargs"]["drop_params"] is True and merged["agent"] == {"a": 1}


def test_v2_container_command(monkeypatch):
    cmd = A._v2_container_cmd("img", "2.4.6", "/out", "/out/x/spec.json", "n1")
    assert cmd[:7] == ["docker", "run", "--rm", "--name", "n1", "--network", "bridge"]
    assert "host" not in cmd and "/var/run/docker.sock:/var/run/docker.sock" in cmd
    assert "/out:/out:rw" in cmd and "OPENAI_API_KEY=EMPTY" in cmd
    assert [c for c in cmd if c.startswith("OPENAI_API_KEY")] == ["OPENAI_API_KEY=EMPTY"]
    assert cmd[-5:] == ["img", "/opt/mini-swe-agent-2.4.6/bin/python",
                        "/opt/gbench/mini_swe_agent_v2_run.py", "--spec", "/out/x/spec.json"]
    if hasattr(os, "getuid"):
        assert f"{os.getuid()}:{os.getgid()}" in cmd


def _task(iid="a__a-1"):
    return A.AgentTask(iid, f"issue {iid}", A.swebench_image(iid), "org/repo",
                       meta={"split": "test"})


def test_rollout_v2_reads_the_driver_result(monkeypatch, tmp_path):
    calls = []

    def _run(cmd, **kw):
        calls.append(list(cmd))
        if cmd[:2] == ["docker", "run"]:
            spec = json.load(open(cmd[-1]))
            assert spec["expected_version"] == "2.4.6"
            assert spec["instance"]["image_name"] == A.swebench_image("a__a-1")
            assert spec["instance"]["problem_statement"] == "issue a__a-1"
            assert "OPENAI_API_KEY" not in kw["env"] and "OPENAI_API_KEY=EMPTY" in cmd
            d = os.path.join(spec["output_dir"], "a__a-1")
            json.dump({"exit_status": "Submitted", "submission": "diff --git a/x b/x\n",
                       "infra_error": False, "n_calls": 7, "truncated_turns": 1,
                       "context_retry_turns": 2, "trajectory": [{"role": "system"}],
                       "trajectory_file": os.path.join(d, "a__a-1.traj.json"),
                       "config_sha256": "9a9c"}, open(os.path.join(d, "result.json"), "w"))
        return _proc()
    monkeypatch.setattr(A.subprocess, "run", _run)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    tr = A._rollout_v2(0, _task(), {"model": {}}, str(tmp_path), image="img", version="2.4.6")
    assert tr["agent_patch"] == "diff --git a/x b/x\n" and tr["agent_exit_status"] == "Submitted"
    assert (tr["agent_steps"], tr["truncated_turns"], tr["context_retry_turns"]) == (7, 1, 2)
    assert tr["agent_infra_error"] is False and tr["status"] == "OK"
    assert tr["extra_payload"] == {"instance_id": "a__a-1", "split": "test"}
    assert calls[-1][:3] == ["docker", "rm", "-f"]


def test_rollout_v2_without_a_result_is_an_infra_error_and_cleans_up(monkeypatch, tmp_path):
    removed = []

    def _run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            d = os.path.join(json.load(open(cmd[-1]))["output_dir"], "a__a-1")
            open(os.path.join(d, "container_id"), "w").write("cid123")
            return _proc(137, "", "killed")
        removed.append(cmd[-1])
        return _proc()
    monkeypatch.setattr(A.subprocess, "run", _run)
    tr = A._rollout_v2(0, _task(), {}, str(tmp_path), image="img", version="2.4.6")
    assert tr["agent_infra_error"] is True and tr["status"] == "AGENT_INFRA_ERROR"
    assert "wrote no result" in tr["agent_error"] and "killed" in tr["agent_error"]
    assert "cid123" in removed


def _desc(sha=None):
    return {"version": "2.4.6", "config_name": A.AGENT_V2_CONFIG,
            "config_sha256": sha or A.AGENT_V2_CONFIG_SHA256["2.4.6"],
            "config": {"agent": {"step_limit": 250, "cost_limit": 3.0},
                       "environment": {"timeout": 60, "env": {"PAGER": "cat"}},
                       "model": {"model_kwargs": {"drop_params": True}}}}


def _fake_rollout(i, task, overrides, out_dir, *, image, version):
    return {"sample_idx": i, "category": task.category, "extra_payload": {
        "instance_id": task.instance_id}, "agent_patch": "d\n", "agent_exit_status": "Submitted",
        "agent_infra_error": False, "agent_steps": 3, "truncated_turns": 0,
        "context_retry_turns": 0, "is_correct": False, "status": "OK",
        "overrides": overrides, "version": version}


async def _all_resolved(traces):
    for t in traces:
        t["is_correct"] = True


def test_run_agentic_v2_no_think_sends_zero_and_names_only_that(monkeypatch):
    monkeypatch.setattr(A, "describe_agent_v2", lambda v: _desc())
    monkeypatch.setattr(A, "bridge_gateway", lambda: ("127.0.0.1", ""))   # the relay's host
    monkeypatch.setattr(A, "require_relay_reachable", lambda *a, **k: None)   # no docker
    monkeypatch.setattr(A, "_rollout_v2", _fake_rollout)
    res = A.run_agentic("swe_bench_multilingual", "m", "http://h:1/v1", 2,
                        [_task("a__a-1"), _task("b__b-2")], _all_resolved, {},
                        agent_v2="2.4.6", max_output_tokens=2048)
    assert (res["temperature"], res["temperature_source"]) == (0.0, "default")
    # the only deviation: the temperature gbench always sends (the bundled config sends none)
    assert res["leaderboard_comparable"] is False and res["accuracy"] == 100.0
    assert res["leaderboard_comparable_reason"] == ("temperature=0.0 sent by the no-think rule (every no-think run samples the model under test at 0.0); the bundled mini-swe-agent 2.4.6 config sends none")
    agent = res["agent"]
    assert agent["version"] == "2.4.6" and agent["config"] == "benchmarks/swebench.yaml"
    assert agent["step_limit"] == 250 and agent["command_timeout_s"] == 60
    assert "gbench-mini-swe-agent-v2:2.4.2-2.4.6" in agent["runner"]
    ov = res["sample_traces"][0]["overrides"]
    assert ov["model"]["model_kwargs"]["temperature"] == 0.0
    # the bundled 2.x config sends no per-turn max_tokens; --max-output-tokens is not applied
    assert "max_tokens" not in ov["model"]["model_kwargs"]
    assert res["max_output_tokens_per_turn"] is None
    assert "--max-output-tokens=2048 is not applied" in res["max_output_tokens_note"]


def test_run_agentic_v2_explicit_temperature_and_foreign_config_are_not_comparable(monkeypatch):
    monkeypatch.setattr(A, "describe_agent_v2", lambda v: _desc("0" * 64))
    monkeypatch.setattr(A, "bridge_gateway", lambda: ("127.0.0.1", ""))   # the relay's host
    monkeypatch.setattr(A, "require_relay_reachable", lambda *a, **k: None)   # no docker
    monkeypatch.setattr(A, "_rollout_v2", _fake_rollout)
    monkeypatch.setenv("GBENCH_SWE_BENCH_MULTILINGUAL_TEMPERATURE", "0.0")
    res = A.run_agentic("swe_bench_multilingual", "m", "http://h:1/v1", 1, [_task()],
                        _all_resolved, {}, agent_v2="2.4.6")
    reason = res["leaderboard_comparable_reason"]
    assert res["leaderboard_comparable"] is False and res["temperature"] == 0.0
    assert "temperature=0.0 set explicitly" in reason and "sends none" in reason
    assert "not the 9a9c86ac1042... file" in reason
    assert res["sample_traces"][0]["overrides"]["model"]["model_kwargs"]["temperature"] == 0.0


def test_1x_config_reason_names_the_comparable_range(monkeypatch):
    monkeypatch.setattr(A, "load_agent_config", lambda: ({"agent": {}, "environment": {}},
                                                         "f" * 64, "1.17.5"))
    monkeypatch.setattr(A, "_rollout", lambda i, t, c, o: _fake_rollout(i, t, c, o, image="",
                                                                          version=""))
    res = A.run_agentic("swe_bench_live", "m", "http://h:1/v1", 1, [_task()], _all_resolved,
                        {}, temperature=0.0)
    reason = res["leaderboard_comparable_reason"]
    assert "1.14-1.17.x; older 1.x rows used other configs" in reason
    assert "1.15.0-1.17.5" not in reason


# ============================================================================================
# the 2.x driver (gbench/docker/mini_swe_agent_v2_run.py) against stub mini-swe-agent modules
# ============================================================================================
class _CWE(Exception):
    pass


_CWE.__name__ = "ContextWindowExceededError"


class _APIConn(Exception):
    pass


_APIConn.__name__ = "APIConnectionError"


def _load_driver(monkeypatch, tmp_path, *, script=None, env_error=None, agent_run=None):
    """Import the driver with stub mini-swe-agent modules; returns (module, record)."""
    record = {"queries": [], "rm": []}
    cfgdir = tmp_path / "cfg"
    (cfgdir / "benchmarks").mkdir(parents=True, exist_ok=True)
    (cfgdir / "benchmarks" / "swebench.yaml").write_text(
        "agent:\n  system_template: s\n  instance_template: t\n  step_limit: 250\n"
        "environment:\n  cwd: /testbed\n  timeout: 60\n"
        "model:\n  model_kwargs:\n    drop_params: true\n")

    def _merge(*ds):
        out = {}
        for d in ds:
            for k, v in (d or {}).items():
                out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(
                    out.get(k), dict) else v
        return out

    class LitellmModel:
        def __init__(self, **kw):
            self.config = types.SimpleNamespace(**kw)

        def query(self, messages, **kw):
            record["queries"].append(dict(kw))
            step = (script or []).pop(0)
            if isinstance(step, Exception):
                raise step
            return step

    class Env:
        container_id = "cid-1"
        config = types.SimpleNamespace(executable="docker")

    def get_sb_environment(config, instance):
        record["env_config"] = config
        if env_error:
            raise env_error
        return Env()

    class Agent:
        def __init__(self, model, env, **cfg):
            self.model, self.env, self.cfg = model, env, cfg
            self.n_calls, self.messages = 0, []

        def run(self, task):
            return agent_run(self, task)

        def save(self, path, extra):
            record["saved"] = (str(path), extra)
            path.write_text(json.dumps({"info": extra["info"]}))

    mods = {
        "minisweagent": types.ModuleType("minisweagent"),
        "minisweagent.agents": types.ModuleType("minisweagent.agents"),
        "minisweagent.agents.default": types.ModuleType("minisweagent.agents.default"),
        "minisweagent.config": types.ModuleType("minisweagent.config"),
        "minisweagent.models": types.ModuleType("minisweagent.models"),
        "minisweagent.models.litellm_model": types.ModuleType("minisweagent.models.litellm_model"),
        "minisweagent.run": types.ModuleType("minisweagent.run"),
        "minisweagent.run.benchmarks": types.ModuleType("minisweagent.run.benchmarks"),
        "minisweagent.run.benchmarks.swebench":
            types.ModuleType("minisweagent.run.benchmarks.swebench"),
        "minisweagent.utils": types.ModuleType("minisweagent.utils"),
        "minisweagent.utils.serialize": types.ModuleType("minisweagent.utils.serialize"),
    }
    mods["minisweagent"].__version__ = "2.4.6"
    mods["minisweagent.agents.default"].DefaultAgent = Agent
    mods["minisweagent.config"].builtin_config_dir = cfgdir
    mods["minisweagent.models.litellm_model"].LitellmModel = LitellmModel
    mods["minisweagent.run.benchmarks.swebench"].get_sb_environment = get_sb_environment
    mods["minisweagent.utils.serialize"].recursive_merge = _merge
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv("MSWEA_SILENT_STARTUP", "1")
    spec = importlib.util.spec_from_file_location(
        "_msa2_driver_under_test", os.path.join(DOCKER_DIR, "mini_swe_agent_v2_run.py"))
    drv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(drv)
    monkeypatch.setattr(drv.subprocess, "run",
                        lambda cmd, **kw: record["rm"].append(cmd) or _proc())
    return drv, record


def _spec(tmp_path, **model_kwargs):
    return {"instance": {"instance_id": "a__a-1", "problem_statement": "fix it",
                         "image_name": "img:1"},
            "overrides": {"model": {"model_name": "openai/m", "cost_tracking": "ignore_errors",
                                    "model_kwargs": {"api_base": "http://h/v1",
                                                     **model_kwargs}}},
            "output_dir": str(tmp_path / "out")}


def test_driver_resends_a_turn_refused_only_for_the_cap(monkeypatch, tmp_path):
    reply = {"role": "assistant", "content": "ok"}
    drv, rec = _load_driver(monkeypatch, tmp_path, script=[_CWE("too long"), reply])
    m = drv.EndpointModel(model_name="openai/m", model_kwargs={"max_tokens": 100})
    assert m.query([]) == reply and m.context_retry_turns == 1
    assert rec["queries"] == [{}, {"max_tokens": None}]
    drv, rec = _load_driver(monkeypatch, tmp_path, script=[_CWE("too long")])
    with pytest.raises(Exception, match="too long"):
        drv.EndpointModel(model_name="openai/m", model_kwargs={}).query([])
    drv, rec = _load_driver(monkeypatch, tmp_path, script=[_APIConn("down")])
    with pytest.raises(Exception, match="down"):
        drv.EndpointModel(model_name="openai/m", model_kwargs={"max_tokens": 5}).query([])


def _submitting(agent, task):
    agent.n_calls = 2
    agent.messages = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "a", "extra": {
            "actions": [{"command": "ls"}],
            "response": {"choices": [{"message": {"reasoning_content": "hmm"},
                                      "finish_reason": "length"}]}}},
        {"role": "exit", "content": "diff"}]
    return {"exit_status": "Submitted", "submission": "diff --git a/x b/x\n"}


def test_driver_runs_one_instance_like_process_instance(monkeypatch, tmp_path):
    drv, rec = _load_driver(monkeypatch, tmp_path, agent_run=_submitting)
    res = drv.run_instance(_spec(tmp_path, max_tokens=64))
    out = tmp_path / "out" / "a__a-1"
    assert json.load(open(out / "result.json")) == res
    assert res["exit_status"] == "Submitted" and res["submission"] == "diff --git a/x b/x\n"
    assert res["infra_error"] is False and res["failed_phase"] is None
    assert res["n_calls"] == 2 and res["truncated_turns"] == 1
    assert res["trajectory"][1] == {"role": "assistant", "content": "a", "reasoning": "hmm",
                                    "finish_reason": "length", "actions": ["ls"]}
    assert res["mini_version"] == "2.4.6" and res["config_name"] == "benchmarks/swebench.yaml"
    # the bundled config with the spec's overrides merged in, as mini-swe-agent merges them
    cfg = rec["env_config"]
    assert cfg["model"]["model_kwargs"] == {"drop_params": True, "api_base": "http://h/v1",
                                            "max_tokens": 64}
    assert cfg["agent"]["step_limit"] == 250
    # the trajectory is saved in upstream's shape and the task container removed synchronously
    assert rec["saved"][1]["info"]["exit_status"] == "Submitted"
    assert rec["rm"] == [["docker", "rm", "-f", "cid-1"]]
    assert not (out / "container_id").exists()


@pytest.mark.parametrize("exc,phase,infra", [
    (_CWE("ctx"), "agent", False),
    (_APIConn("down"), "agent", True),
])
def test_driver_classifies_mid_rollout_exceptions(monkeypatch, tmp_path, exc, phase, infra):
    def _raise(agent, task):
        raise exc
    drv, rec = _load_driver(monkeypatch, tmp_path, agent_run=_raise)
    res = drv.run_instance(_spec(tmp_path))
    assert res["exit_status"] == type(exc).__name__ and res["submission"] == ""
    assert res["failed_phase"] == phase and res["infra_error"] is infra
    assert res["traceback"]


def test_driver_environment_failure_is_infrastructure(monkeypatch, tmp_path):
    drv, rec = _load_driver(monkeypatch, tmp_path, env_error=RuntimeError("pull failed"))
    res = drv.run_instance(_spec(tmp_path))
    assert res["infra_error"] is True and res["failed_phase"] == "environment"
    assert res["error"] == "pull failed" and res["trajectory_file"] is None
    assert rec["rm"] == []


def test_driver_describe_and_version_guard(monkeypatch, tmp_path, capsys):
    drv, _ = _load_driver(monkeypatch, tmp_path)
    assert drv.main(["--describe"]) == 0
    desc = json.loads(capsys.readouterr().out)
    assert desc["version"] == "2.4.6" and desc["config"]["agent"]["step_limit"] == 250
    spec = _spec(tmp_path)
    spec["expected_version"] = "2.4.2"
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    with pytest.raises(SystemExit, match="asks for 2.4.2"):
        drv.main(["--spec", str(p)])


# ============================================================================================
# swebench v5 harness (swebench_common)
# ============================================================================================
def test_v5_scorer_command_and_report(monkeypatch, tmp_path):
    snap = tmp_path / "snap" / "test.jsonl"
    snap.parent.mkdir()
    snap.write_text(json.dumps({"instance_id": "a__a-1"}) + "\n")
    seen = {}

    def _run(cmd, cwd=None, **kw):
        seen["cmd"] = list(cmd)
        run_id = cmd[cmd.index("--run_id") + 1] if "--run_id" in cmd else None
        rep = {"total_instances": 2, "submitted_instances": 2, "completed_instances": 2,
               "resolved_instances": 1, "unresolved_instances": 1, "error_instances": 0,
               "empty_patch_instances": 0, "resolved_ids": ["a__a-1"], "error_ids": [],
               "infra_failure_ids": ["b__b-2"], "ambiguous_failure_ids": [],
               "failure_reasons": {"b__b-2": "out_of_memory"}}
        with open(os.path.join(cwd, f"gbench__m.{run_id}.json"), "w") as f:
            json.dump(rep, f)
        return _proc()
    monkeypatch.setattr(C.subprocess, "run", _run)
    monkeypatch.setenv("GBENCH_SWE_OMP_THREADS", "3")
    metrics = {}
    scorer = C.make_swebench_scorer("copilot_bench_swe", "m", str(snap), "test", "swebench", 2,
                                    metrics, harness_image="gbench-swebench-v5:5.0.2",
                                    harness_v5=True)
    traces = [{"extra_payload": {"instance_id": "a__a-1"}, "agent_patch": "d\n"},
              {"extra_payload": {"instance_id": "b__b-2"}, "agent_patch": "e\n"}]
    asyncio.run(scorer(traces))
    cmd = seen["cmd"]
    img = cmd.index("gbench-swebench-v5:5.0.2")
    assert "--cache_level" not in cmd and "--namespace" not in cmd
    assert f"{snap.parent}:{snap.parent}:ro" in cmd and "--user" in cmd
    shim = os.path.abspath(C.swe_thread_cap.__file__)
    assert f"{shim}:{shim}:ro" in cmd and cmd[img + 1:img + 3] == ["python", "-c"]
    assert "swebench.harness.run_evaluation" in cmd[img + 3] and "install(3)" in cmd[img + 3]
    assert [t["is_correct"] for t in traces] == [True, False]
    rep = metrics["swebench_report"]
    assert rep["infra_failure_ids"] == ["b__b-2"] and metrics["harness_report_retries"] == 0
    assert rep["failure_reasons"] == {"b__b-2": "out_of_memory"}
    # v5's advisory classification is recorded, not promoted: b__b-2 stays an unresolved
    # instance and the run stays complete and comparable.
    res = {"status": "success"}
    reasons = C.swebench_report_reasons("copilot_bench_swe", res, metrics)
    assert res["status"] == "success" and "partial_evaluation" not in res
    assert reasons == []


def test_scorer_rewrites_a_report_the_harness_did_not_write(monkeypatch, tmp_path):
    runs = []

    def _run(cmd, cwd=None, **kw):
        runs.append(1)
        run_id = cmd[cmd.index("--run_id") + 1]
        os.makedirs(os.path.join(cwd, "logs", "run_evaluation", run_id), exist_ok=True)
        if len(runs) == 2:
            with open(os.path.join(cwd, f"gbench__m.{run_id}.json"), "w") as f:
                json.dump({"total_instances": 1, "completed_instances": 1,
                           "resolved_ids": ["a__a-1"], "error_ids": []}, f)
            return _proc()
        return _proc(1, "", "docker.errors.NotFound: No such container")
    monkeypatch.setattr(C.subprocess, "run", _run)
    metrics = {}
    tr = [{"extra_payload": {"instance_id": "a__a-1"}, "agent_patch": "d\n"}]
    asyncio.run(C.make_swebench_scorer("x", "m", "d.jsonl", "test", None, 1, metrics,
                                       harness_image="img", harness_v5=True)(tr))
    assert len(runs) == 2 and metrics["harness_report_retries"] == 1 and tr[0]["is_correct"]


def test_split_report_keeps_v5_failure_ids_advisory(tmp_path):
    report = {"completed_instances": 3, "empty_patch_instances": 1,
              "error_ids": ["e1"], "infra_failure_ids": ["u1", "e1"]}
    patch_fail, harness_err, scoreable = C._split_report(report, str(tmp_path))
    # e1 has no run_instance.log, so it is a harness error by its own log; u1 (completed) is not
    assert patch_fail == set() and harness_err == {"e1"}
    assert scoreable == 3 + 1 + 0
    assert C._split_report({"completed_instances": 2, "error_ids": []}, "")[2] == 2


def test_v5_prereq_checks_columns_image_label_and_specs(monkeypatch, tmp_path):
    snap = tmp_path / "test.jsonl"
    row = {c: "x" for c in C.SWEBENCH_V5_COLUMNS}
    row["problem_statement"] = "p"
    snap.write_text(json.dumps({k: v for k, v in row.items() if k != "eval_script"}) + "\n")
    monkeypatch.setattr(C.subprocess, "run", lambda *a, **k: pytest.fail("docker called"))
    ok, why = C.check_swebench_v5_prereq(str(snap), "test")
    assert not ok and "eval_script" in why
    snap.write_text(json.dumps(row) + "\n")
    state = {"label": "5.0.1", "probe": 0}
    calls = []

    def _run(cmd, **kw):
        calls.append(cmd)
        if cmd[:3] == ["docker", "image", "inspect"]:
            return _proc(0, json.dumps({"org.gbench.swebench": state["label"]}))
        if cmd[:2] == ["docker", "run"]:
            return _proc(state["probe"], "1", "KeyError: 'image'")
        return _proc()
    monkeypatch.setattr(C.subprocess, "run", _run)
    ok, why = C.check_swebench_v5_prereq(str(snap), "test")
    assert not ok and "not 5.0.2" in why and "\n" not in why
    state["label"] = "5.0.2"
    assert C.check_swebench_v5_prereq(str(snap), "test") == (True, "")
    probe = calls[-1]
    assert probe[:5] == ["docker", "run", "--rm", "--network", "none"]
    assert f"{tmp_path}:{tmp_path}:ro" in probe and "make_test_spec" in probe[-3]
    state["probe"] = 1
    ok, why = C.check_swebench_v5_prereq(str(snap), "test")
    assert not ok and "KeyError" in why


def test_oneshot_samples_come_from_the_snapshot_offline(monkeypatch, tmp_path):
    rows = [{"instance_id": "a__a-1", "problem_statement": "p", "repo": "a/a"}]
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **k: list(rows)
    monkeypatch.setitem(sys.modules, "datasets", fake)
    first = C.load_swe_samples("o/d", "test", None, "x", revision="abc")
    fake.load_dataset = lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
    again = C.load_swe_samples("o/d", "test", None, "x", revision="abc")
    assert first == again and again[0][2]["instance_id"] == "a__a-1"


def test_suites_route_to_the_2x_agent_and_the_v5_harness(monkeypatch):
    for mod, run, prereq in ((CB, CB.run_copilot_bench_swe,
                              "check_copilot_bench_swe_prerequisites"),
                             (ML, ML.run_swe_bench_multilingual,
                              "check_swe_bench_multilingual_prerequisites")):
        monkeypatch.setattr(mod, prereq, lambda: (True, ""))
        need = []
        monkeypatch.setattr(A, "require_agent_v2", lambda n, d, v: need.append(v))
        monkeypatch.setattr(A, "require_agent", lambda *a: pytest.fail("1.x runner used"))
        seen = {}
        monkeypatch.setattr(mod, "execute_swebench_agentic",
                            lambda *a, **k: seen.update(a=a, k=k) or {"mode": "agentic"})
        monkeypatch.setattr(mod, "execute_swebench",
                            lambda *a, **k: seen.update(one=k) or {"status": "success"})
        monkeypatch.delenv(A.mode_env_var(mod.__name__.split(".")[-1]), raising=False)
        run("m", "http://x/v1", 1)
        assert need == [mod.AGENT_VERSION] and seen["k"]["agent_v2"] == mod.AGENT_VERSION
        assert seen["k"]["harness_v5"] is True
        assert seen["k"]["harness_image"] == "gbench-swebench-v5:5.0.2"
        assert seen["k"]["dataset_revision"] == mod._REVISION
        monkeypatch.setenv(A.mode_env_var(mod.__name__.split(".")[-1]), "oneshot")
        run("m", "http://x/v1", 1)
        assert seen["one"]["harness_v5"] is True


# ============================================================================================
# SWE-bench Pro V2 (Harbor locked protocol)
# ============================================================================================
def test_task_endpoint_and_allowlist_host(monkeypatch):
    assert P.task_endpoint("http://127.0.0.1:8000") == "http://172.17.0.1:8000/v1"
    assert P.task_endpoint("http://localhost:8000/v1") == "http://172.17.0.1:8000/v1"
    assert P.task_endpoint("http://10.0.0.5:8000/v1") == "http://10.0.0.5:8000/v1"
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_TASK_HOST", "192.168.9.1")
    assert P.task_endpoint("http://127.0.0.1:8000/v1") == "http://192.168.9.1:8000/v1"
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_TASK_ENDPOINT", "http://model.lan:9/v1")
    assert P.task_endpoint("http://127.0.0.1:8000/v1") == "http://model.lan:9/v1"
    assert P.endpoint_host("http://model.lan:9/v1") == "model.lan"
    assert P.endpoint_host("http://[fd00::1]:9/v1") == "fd00::1"


def test_harbor_job_arguments_follow_the_v2_readme():
    args = P.agent_job_args(model_name="google/g", endpoint="http://172.17.0.1:8000/v1",
                            config_path="/w/cfg.yaml", jobs_dir="/w/jobs", concurrency=3,
                            include=["instance_a"], threads=2)
    flags = list(zip(args, args[1:]))
    assert ("-p", "/opt/SWE-bench_Pro-os/v2/tasks") in flags and ("-e", "docker") in flags
    assert ("-a", "locked_mini_swe:LockedMiniSwe") in flags and ("-m", "openai/google/g") in flags
    assert ("--ak", "config_file=/w/cfg.yaml") in flags and ("--ak", "version=2.4.6") in flags
    assert ("--ae", "OPENAI_API_BASE=http://172.17.0.1:8000/v1") in flags
    assert ("--allow-agent-host", "172.17.0.1") in flags and ("-n", "3") in flags
    assert ("--job-name", "agent") in flags and ("-i", "instance_a") in flags and "-y" in args
    assert ("--ae", "OMP_NUM_THREADS=2") in flags and ("--ve", "OMP_NUM_THREADS=2") in flags
    assert not any("OPENAI_API_KEY" in a for a in args)
    r = P.regrade_job_args(source_job="/w/jobs/agent", jobs_dir="/w/jobs", concurrency=3,
                           include=None, threads=0)
    rflags = list(zip(r, r[1:]))
    assert ("-a", "patch_replay:PatchReplayAgent") in rflags and ("-m", "replay") in rflags
    assert ("--ak", "source_job=/w/jobs/agent") in rflags and ("--job-name", "regrade") in rflags
    assert "-i" not in r and "--ve" not in r


def test_orchestrator_command_keeps_the_key_off_the_host_command_line(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    cmd = P._orchestrator_cmd(str(tmp_path), "n1", ["-p", "x"])
    assert "sk-secret" not in " ".join(cmd)
    assert f"{tmp_path}:{tmp_path}:rw" in cmd and "/var/run/docker.sock:/var/run/docker.sock" in cmd
    assert cmd[cmd.index("-w") + 1] == "/opt/SWE-bench_Pro-os"
    assert "PYTHONPATH=/opt/SWE-bench_Pro-os/v2/tooling" in cmd
    i = cmd.index(P.V2_IMAGE_DEFAULT)
    # no shell: the agent gets the literal placeholder, and no key enters the container
    assert cmd[i + 1:] == ["harbor", "run", "-p", "x", "--ae", "OPENAI_API_KEY=EMPTY"]
    assert "OPENAI_API_KEY" not in cmd[:i]


def test_locked_config_gets_only_the_reasoning_switch_and_the_temperature():
    text = ("agent:\n  step_limit: 0\n  instance_template: x\nmodel:\n  model_kwargs:\n"
            "    drop_params: true\n    reasoning_effort: high\n")
    cfg = P.build_locked_agent_config(text, thinking=False, temperature=0.0)
    assert cfg["agent"] == {"step_limit": 0, "instance_template": "x"}
    assert cfg["model"]["model_kwargs"] == {
        "drop_params": True, "reasoning_effort": "high", "temperature": 0.0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    cfg = P.build_locked_agent_config(text, thinking=True, temperature=0.7)
    assert cfg["model"]["model_kwargs"]["temperature"] == 0.7


def _trial(job_dir, iid, *, reward=None, exc=None, agent=True, verifier=True, exit_status=None,
           calls=5, patch="diff --git a/x b/x\n", traj=True):
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
            json.dump({"info": {"exit_status": exit_status,
                                "model_stats": {"api_calls": calls}}}, f)
    if patch is not None:
        with open(os.path.join(tdir, "agent", "model.patch"), "w") as f:
            f.write(patch)


@pytest.mark.parametrize("kw,outcome", [
    (dict(reward=1.0, exit_status="Submitted"), True),
    (dict(reward=0.0, exc="AgentTimeoutError", exit_status=""), True),
    (dict(reward=0.0, exc="NonZeroAgentExitCodeError", exit_status="LimitsExceeded"), True),
    (dict(exc="VerifierTimeoutError", exit_status="Submitted"), True),
    (dict(exc="EnvironmentStartTimeoutError", agent=False, verifier=False, traj=False,
          patch=None), False),
    (dict(reward=0.0, exc="NonZeroAgentExitCodeError", exit_status="APIConnectionError",
          calls=0), False),
    (dict(reward=0.0, exit_status="Submitted", patch=None), False),
    (dict(reward=0.0, traj=False), False),
])
def test_agent_trials_are_model_outcomes_or_infrastructure(tmp_path, kw, outcome):
    _trial(str(tmp_path), "instance_a", **kw)
    view = P.agent_trial_view(P.job_trials(str(tmp_path)).get("instance_a"))
    assert view["model_outcome"] is outcome, view
    assert P.agent_trial_view(None)["model_outcome"] is False


def _pro_setup(monkeypatch, tmp_path, ids, hard, agent_trials, regrade_rewards,
               retry_trials=None):
    rows = [{"instance_id": i, "problem_statement": f"P {i}", "repo": "o/r",
             "repo_language": "go" if n % 2 else "python",
             "docker_image": f"ghcr.io/scaleapi/swe-bench_pro-v2:{i}"}
            for n, i in enumerate(ids)]
    monkeypatch.setattr(P, "load_v2_rows", lambda: rows)
    monkeypatch.setattr(P, "load_hard51_ids", lambda v2=None: set(hard))
    monkeypatch.setattr(P, "read_locked_agent_config",
                        lambda: ("model:\n  model_kwargs:\n    reasoning_effort: high\n",
                                 P.AGENT_CONFIG_SHA256))
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_TRAJ_DIR", str(tmp_path / "runs"))
    jobs = []

    def _harbor(workdir, args, job):
        jobs.append((job, list(args)))
        jobs_dir = args[args.index("--jobs-dir") + 1]
        job_dir = os.path.join(jobs_dir, job)
        os.makedirs(job_dir, exist_ok=True)
        include = [a for a in args if a.startswith("instance_")]
        if job == "agent":
            for iid, kw in agent_trials.items():
                _trial(job_dir, iid, **kw)
        elif job.startswith("agent_retry"):
            for iid, kw in (retry_trials or {}).items():
                _trial(job_dir, iid, **kw)
        else:
            for iid, reward in regrade_rewards.items():
                if not include or iid in include:
                    _trial(job_dir, iid, reward=reward, traj=False, patch=None)
        return 0, "harbor done"
    monkeypatch.setattr(P, "_run_harbor", _harbor)
    return jobs


def test_v2_run_reports_regraded_headline_in_sandbox_and_hard51(monkeypatch, tmp_path):
    ids = ["instance_a", "instance_b", "instance_c", "instance_d"]
    jobs = _pro_setup(
        monkeypatch, tmp_path, ids, hard={"instance_a", "instance_b"},
        agent_trials={"instance_a": dict(reward=1.0, exit_status="Submitted"),
                      "instance_b": dict(reward=1.0, exit_status="Submitted"),
                      "instance_c": dict(reward=0.0, exit_status="LimitsExceeded"),
                      "instance_d": dict(exc="EnvironmentStartTimeoutError", agent=False,
                                         verifier=False, traj=False, patch=None)},
        regrade_rewards={"instance_a": 1.0, "instance_b": 0.0, "instance_c": 0.0})
    res = P.run_swe_bench_pro_v2("google/g", "http://127.0.0.1:8000/v1", 2,
                                 enable_thinking=True, max_output_tokens=32768)
    # the task that never reached the model is tried once more; it fails again
    assert [j for j, _ in jobs] == ["agent", "agent_retry1", "regrade"]
    assert [a for a in jobs[1][1] if a.startswith("instance_")] == ["instance_d"]
    assert res["infra_retried_ids"] == ["instance_d"]
    agent_args, regrade_args = jobs[0][1], jobs[2][1]
    assert set(a for a in agent_args if a.startswith("instance_")) == set(ids)
    assert set(a for a in regrade_args if a.startswith("instance_")) == {
        "instance_a", "instance_b", "instance_c"}
    assert (res["total_questions"], res["correct_answers"], res["failed_requests"]) == (4, 1, 1)
    # the task that never reached the model is a harness error, left out of every rate
    assert res["scored_instances"] == 3 and res["harness_errors"] == {"agent_infra": 1}
    assert res["accuracy"] == 33.33 and res["in_sandbox"] == {
        "total": 3, "resolved": 2, "accuracy": 66.67, "harness_errors": 1}
    assert res["hard51"]["regraded"] == {"total": 2, "resolved": 1, "accuracy": 50.0}
    assert res["hard51"]["in_sandbox"]["resolved"] == 2 and res["hard51"]["complete"] is False
    assert res["status"] == "completed_with_errors"
    assert res["partial_evaluation"]["infra_error_ids"] == ["instance_d"]
    reason = res["leaderboard_comparable_reason"]
    assert res["leaderboard_comparable"] is False and "partial: 1 of 4" in reason
    # --thinking: the locked config sends no temperature; the thinking rule's 1.0 is added
    assert (res["temperature"], res["temperature_source"]) == (1.0, "thinking")
    assert res["agent"]["gbench_config_additions"]["model.model_kwargs.temperature"] == 1.0
    assert "temperature=1.0 sent by the --thinking rule" in reason
    assert res["max_output_tokens_per_turn"] is None
    assert "32768 is not applied" in res["max_output_tokens_note"]
    tr = {t["extra_payload"]["instance_id"]: t for t in res["sample_traces"]}
    assert tr["instance_a"]["is_correct"] and tr["instance_a"]["in_sandbox_reward"] == 1.0
    assert tr["instance_b"]["regrade_reward"] == 0.0 and not tr["instance_b"]["is_correct"]
    assert tr["instance_d"]["agent_infra_error"] and "EnvironmentStart" in \
        tr["instance_d"]["agent_error"]
    cfg = open(os.path.join(res["agent_output_dir"], "mini_toolcall.gbench.yaml")).read()
    assert "enable_thinking: true" in cfg and "reasoning_effort: high" in cfg
    import yaml
    assert yaml.safe_load(cfg)["model"]["model_kwargs"]["temperature"] == 1.0


def test_v2_run_without_thinking_or_with_a_temperature_is_not_comparable(monkeypatch, tmp_path):
    _pro_setup(monkeypatch, tmp_path, ["instance_a"], hard=set(),
               agent_trials={"instance_a": dict(reward=1.0, exit_status="Submitted")},
               regrade_rewards={"instance_a": 1.0})
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_TEMPERATURE", "0.2")
    res = P.run_swe_bench_pro_v2("m", "http://h:1/v1", 1, enable_thinking=False)
    reason = res["leaderboard_comparable_reason"]
    assert res["status"] == "success" and res["accuracy"] == 100.0
    assert "reasoning_effort=high" in reason and "temperature=0.2 set explicitly" in reason
    assert res["agent"]["gbench_config_additions"]["model.model_kwargs.temperature"] == 0.2


@pytest.mark.parametrize("thinking,want,source", [(False, 0.0, "default"),
                                                   (True, 1.0, "thinking")])
def test_v2_run_writes_the_rule_temperature_into_the_locked_config(monkeypatch, tmp_path,
                                                                    thinking, want, source):
    # gbench's rule: 0.0 without --thinking, 1.0 with it, in the derived config the agent reads
    _pro_setup(monkeypatch, tmp_path, ["instance_a"], hard=set(),
               agent_trials={"instance_a": dict(reward=1.0, exit_status="Submitted")},
               regrade_rewards={"instance_a": 1.0})
    monkeypatch.delenv("GBENCH_SWE_BENCH_PRO_TEMPERATURE", raising=False)
    res = P.run_swe_bench_pro_v2("m", "http://h:1/v1", 1, enable_thinking=thinking)
    assert (res["temperature"], res["temperature_source"]) == (want, source)
    assert res["agent"]["gbench_config_additions"]["model.model_kwargs.temperature"] == want
    import yaml
    cfg = yaml.safe_load(open(os.path.join(res["agent_output_dir"], "mini_toolcall.gbench.yaml")))
    assert cfg["model"]["model_kwargs"]["temperature"] == want
    assert f"temperature={want} sent by the" in res["leaderboard_comparable_reason"]


def test_v2_run_with_no_scoreable_task_is_an_error_not_a_zero(monkeypatch, tmp_path):
    jobs = _pro_setup(monkeypatch, tmp_path, ["instance_a", "instance_b"], hard=set(),
                      agent_trials={}, regrade_rewards={})
    res = P.run_swe_bench_pro_v2("m", "http://h:1/v1", 1, enable_thinking=True)
    assert res["status"] == "error" and "not a 0% resolved rate" in res["error"]
    assert [j for j, _ in jobs] == ["agent", "agent_retry1"]


def test_v2_infra_retries_regrade_against_their_own_agent_job(monkeypatch, tmp_path):
    jobs = _pro_setup(
        monkeypatch, tmp_path, ["instance_a", "instance_b"], hard=set(),
        agent_trials={"instance_a": dict(reward=1.0, exit_status="Submitted"),
                      "instance_b": dict(exc="EnvironmentStartTimeoutError", agent=False,
                                         verifier=False, traj=False, patch=None)},
        retry_trials={"instance_b": dict(reward=0.0, exit_status="LimitsExceeded")},
        regrade_rewards={"instance_a": 1.0, "instance_b": 1.0})
    res = P.run_swe_bench_pro_v2("m", "http://h:1/v1", 1, enable_thinking=True)
    names = [j for j, _ in jobs]
    assert names == ["agent", "agent_retry1", "regrade", "regrade_agent_retry1"]
    src = {j: a[a.index("--ak") + 1] for j, a in jobs if j.startswith("regrade")}
    assert src["regrade"].endswith("/jobs/agent")
    assert src["regrade_agent_retry1"].endswith("/jobs/agent_retry1")
    assert res["status"] == "success" and res["accuracy"] == 100.0
    # the only deviation is the --thinking rule's temperature 1.0 (the locked config sends none)
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"] == (
        "temperature=1.0 sent by the --thinking rule (every thinking run samples the model under "
        "test at 1.0); the locked config sends none")


def test_v2_regrade_without_a_reward_is_retried_once(monkeypatch, tmp_path):
    calls = {"n": 0}
    jobs = _pro_setup(monkeypatch, tmp_path, ["instance_a"], hard=set(),
                      agent_trials={"instance_a": dict(reward=1.0, exit_status="Submitted")},
                      regrade_rewards={})
    real = P._run_harbor

    def _flaky(workdir, args, job):
        rc, tail = real(workdir, args, job)
        if job == "regrade_retry1":
            calls["n"] += 1
            _trial(os.path.join(args[args.index("--jobs-dir") + 1], job), "instance_a",
                   reward=1.0, traj=False, patch=None)
        return rc, tail
    monkeypatch.setattr(P, "_run_harbor", _flaky)
    res = P.run_swe_bench_pro_v2("m", "http://h:1/v1", 1, enable_thinking=True)
    assert [j for j, _ in jobs] == ["agent", "regrade", "regrade_retry1"] and calls["n"] == 1
    assert res["correct_answers"] == 1 and res["status"] == "success"


def test_v2_prereq_checks_image_labels_without_an_opt_in(monkeypatch):
    monkeypatch.setattr(P.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(P.os.path, "exists", lambda p: True)
    labels = {"org.gbench.harbor": "0.23.0", "org.gbench.swe_bench_pro_os": P.HARNESS_COMMIT,
              "org.gbench.egress_sidecar_read_timeout": P.EGRESS_SIDECAR_READ_TIMEOUT}
    monkeypatch.setattr(P.subprocess, "run", lambda cmd, **kw: _proc(0, json.dumps(labels)))
    ok, why = P.check_swe_bench_pro_v2_prerequisites()
    assert not ok and "0.23.0" in why and "\n" not in why
    labels["org.gbench.harbor"] = "0.22.0"
    monkeypatch.delenv("GBENCH_SWE_BENCH_PRO_RUN", raising=False)
    monkeypatch.delenv("SWE_BENCH_PRO_RUN", raising=False)
    assert P.check_swe_bench_pro_v2_prerequisites() == (True, "")


def test_oneshot_replay_source_is_what_patch_replay_reads(tmp_path):
    """patch_replay._find_patch globs <source_job>/instance_*/result.json, matches
    task_name.split('/')[-1] against the task directory name and reads agent/model.patch."""
    import glob
    src = str(tmp_path / "oneshot")
    P.write_replay_source(src, {"instance_a": "diff\n", "instance_b": ""})
    found = {}
    for rj in glob.glob(os.path.join(src, "instance_*", "result.json")):
        name = json.load(open(rj))["task_name"].split("/")[-1]
        found[name] = open(os.path.join(os.path.dirname(rj), "agent", "model.patch")).read()
    assert found == {"instance_a": "diff\n", "instance_b": ""}


def test_engine_route_keeps_the_v1_interface_and_labels_its_result(monkeypatch):
    seen = {}
    fake = types.ModuleType("gbench.runners.eval_suites.swe_bench_pro_agentic")
    fake.execute_agentic = lambda *a, **k: seen.update(k) or {"status": "success"}
    monkeypatch.setitem(sys.modules, "gbench.runners.eval_suites.swe_bench_pro_agentic", fake)
    monkeypatch.setattr(P, "check_swe_bench_pro_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(P, "check_swe_bench_pro_v2_prerequisites",
                        lambda: pytest.fail("V2 gate on the engine route"))
    monkeypatch.setenv("GBENCH_SWEBENCH_PRO_AGENTIC", "1")
    res = P.run_swe_bench_pro("m", "http://x", concurrency=2, limit=3, temperature=0.5)
    assert seen == {"concurrency": 2, "limit": 3, "only_instance_ids": None,
                    "eval_name": "swe_bench_pro", "temperature": 0.5, "max_output_tokens": None}
    assert res["leaderboard_comparable"] is False
    assert "v1 harness" in res["leaderboard_comparable_reason"]
    # the v1 prerequisites the engine and plugin variants gate on still ask for the v1 harness
    monkeypatch.undo()
    monkeypatch.setattr(P.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(P.subprocess, "run", lambda *a, **k: _proc())
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))
    monkeypatch.delenv("GBENCH_SWE_BENCH_PRO_HARNESS_DIR", raising=False)
    monkeypatch.delenv("SWE_BENCH_PRO_HARNESS_DIR", raising=False)
    monkeypatch.delenv("GBENCH_PREREQS_DIR", raising=False)
    ok, why = P.check_swe_bench_pro_prerequisites()
    assert not ok and "swe_bench_pro_eval.py" in why


def test_v1_raw_sample_table_may_be_jsonl(tmp_path):
    rows = [{"instance_id": f"i{n}", "base_commit": f"c{n}", "before_repo_set_cmd": "git x",
             "selected_test_files_to_run": "['t.py']", "fail_to_pass": "['a']",
             "pass_to_pass": "[]"} for n in range(2)]
    good = tmp_path / "pinned.jsonl"
    good.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert P._raw_sample_mismatch(str(good), rows) == ""
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n")
    assert "JSON lines" in P._raw_sample_mismatch(str(bad), rows)
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps(dict(rows[0], base_commit="zz")) + "\n" + json.dumps(rows[1]))
    assert "base_commit" in P._raw_sample_mismatch(str(other), rows)
