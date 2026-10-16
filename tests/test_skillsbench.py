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

"""Deterministic unit tests for the skillsbench canonical harness (no Docker / model / judge)."""

import json
import os
import sys

import pytest

from gbench.runners.eval_suites import skillsbench as S

_DOCKER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gbench", "docker")
sys.path.insert(0, _DOCKER_DIR)
import skillsbench_run as RUN  # noqa: E402


# --------------------------------------------------------------------------- #
# Headline scoring.
# --------------------------------------------------------------------------- #
def test_score_single_condition():
    summary = {"conditions": {"without-skills": {"mean_reward": 0.42, "n_tasks": 87, "n_scored": 87}}}
    s = S.compute_skillsbench_score(summary)
    assert s["primary_condition"] == "without-skills"
    assert s["mean_reward"] == pytest.approx(0.42)


def test_score_both_conditions_primary_and_lift():
    summary = {
        "conditions": {
            "with-skills": {"mean_reward": 0.55, "n_tasks": 87},
            "without-skills": {"mean_reward": 0.40, "n_tasks": 87},
        },
        "with_skills_lift": 0.15,
    }
    s = S.compute_skillsbench_score(summary)
    assert s["primary_condition"] == "without-skills"   # raw capability is the headline
    assert s["mean_reward"] == pytest.approx(0.40)
    assert s["with_skills_lift"] == pytest.approx(0.15)
    assert set(s["condition_means"]) == {"with-skills", "without-skills"}


def test_score_none_when_no_means():
    assert S.compute_skillsbench_score({"conditions": {}}) is None
    assert S.compute_skillsbench_score({"conditions": {"without-skills": {"mean_reward": None}}}) is None


# --------------------------------------------------------------------------- #
# Task-reachable endpoint translation.
# --------------------------------------------------------------------------- #
def test_task_endpoint_rewrites_localhost(monkeypatch):
    monkeypatch.delenv("GBENCH_SKILLSBENCH_TASK_ENDPOINT", raising=False)
    monkeypatch.delenv("GBENCH_SKILLSBENCH_TASK_HOST", raising=False)
    assert S._task_reachable_endpoint("http://127.0.0.1:8000/v1") == "http://172.17.0.1:8000/v1"
    assert S._task_reachable_endpoint("http://localhost:8000") == "http://172.17.0.1:8000/v1"


def test_task_endpoint_custom_host_and_override(monkeypatch):
    monkeypatch.delenv("GBENCH_SKILLSBENCH_TASK_ENDPOINT", raising=False)
    monkeypatch.setenv("GBENCH_SKILLSBENCH_TASK_HOST", "host.docker.internal")
    assert S._task_reachable_endpoint("http://127.0.0.1:9000/v1") == "http://host.docker.internal:9000/v1"
    monkeypatch.setenv("GBENCH_SKILLSBENCH_TASK_ENDPOINT", "http://gw:1/v1")
    assert S._task_reachable_endpoint("http://127.0.0.1:8000/v1") == "http://gw:1/v1"


def test_task_endpoint_remote_host_unchanged(monkeypatch):
    monkeypatch.delenv("GBENCH_SKILLSBENCH_TASK_ENDPOINT", raising=False)
    # a non-localhost host is left alone (only /v1 is ensured)
    assert S._task_reachable_endpoint("http://10.0.0.5:8000/v1") == "http://10.0.0.5:8000/v1"
    assert S._task_reachable_endpoint("http://10.0.0.5:8000") == "http://10.0.0.5:8000/v1"


# --------------------------------------------------------------------------- #
# Launcher reward parsing from a rollout's result.json.
# --------------------------------------------------------------------------- #
def test_reward_from_jobs_rewards_dict(tmp_path):
    jobs = tmp_path / "jobs"
    (jobs / "r1").mkdir(parents=True)
    (jobs / "r1" / "result.json").write_text(json.dumps({"rewards": {"reward": 1.0}}))
    assert RUN._reward_from_jobs(str(jobs)) == pytest.approx(1.0)


def test_reward_from_jobs_scalar_and_flat_reward(tmp_path):
    jobs = tmp_path / "jobs"
    (jobs / "r").mkdir(parents=True)
    (jobs / "r" / "result.json").write_text(json.dumps({"reward": 0.0}))
    assert RUN._reward_from_jobs(str(jobs)) == pytest.approx(0.0)


def test_reward_from_jobs_none_cases(tmp_path):
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    assert RUN._reward_from_jobs(str(jobs)) is None            # no result.json at all
    (jobs / "r").mkdir()
    (jobs / "r" / "result.json").write_text(json.dumps({"rewards": None, "verifier_error": "x"}))
    assert RUN._reward_from_jobs(str(jobs)) is None            # verifier crashed -> no reward


# --------------------------------------------------------------------------- #
# Sampling: a --thinking run samples the model at 1.0 (the maintainer's rule), set on the agent's
# requests by gbench's relay; a no-think run keeps the agent's own default (no relay, no value).
# --------------------------------------------------------------------------- #
def _fake_upstream():
    """A local HTTP server standing in for the model endpoint: records each request body and
    answers a minimal chat completion."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    bodies = []

    class _H(BaseHTTPRequestHandler):
        def do_POST(self):
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            out = json.dumps({"choices": [{"index": 0, "finish_reason": "stop",
                                           "message": {"role": "assistant", "content": "ok"}}]})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out.encode())

        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, bodies


def _run_with_agent_request(monkeypatch, thinking, env=None, sent_temperature=0.0):
    """run_skillsbench with docker faked: the 'orchestrator' sends one chat request, as the agent
    would, to the GBENCH_MODEL_BASE_URL it was given, with the agent's own temperature
    (`sent_temperature`)."""
    import subprocess
    import urllib.request
    srv, bodies = _fake_upstream()
    upstream = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    for k in ("GBENCH_SKILLSBENCH_API_KEY", "VLLM_API_KEY", "OPENAI_API_KEY",
              "GBENCH_SKILLSBENCH_TASK_ENDPOINT", "GBENCH_SKILLSBENCH_TEMPERATURE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GBENCH_SKILLSBENCH_TASK_HOST", "127.0.0.1")
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(S, "check_skillsbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(S.swebench_common, "require_task_network_blocked", lambda *a, **k: None)
    monkeypatch.setattr(S, "dns_harness_errors", lambda workdir, summary: ({}, {}))
    monkeypatch.setattr(S, "keep_artifacts", lambda workdir, dest: None)
    monkeypatch.setattr(S, "remove_workdir", lambda workdir, image=None: [])
    seen = {}

    def run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            seen["cmd"] = cmd
            model_url = next(a.split("=", 1)[1] for a in cmd
                             if a.startswith("GBENCH_MODEL_BASE_URL="))
            seen["model_url"] = model_url
            req = urllib.request.Request(
                model_url.replace("172.17.0.1", "127.0.0.1") + "/chat/completions",
                data=json.dumps({"model": "m", "temperature": sent_temperature,
                                 "messages": [{"role": "user", "content": "hi"}]}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=30).read()
            workdir = cmd[cmd.index("-v", cmd.index("-v") + 1) + 1].split(":")[0]
            with open(os.path.join(workdir, "skillsbench_summary.json"), "w") as f:
                json.dump({"conditions": {"without-skills": {
                    "per_task": {"a": 1.0}, "mean_reward": 1.0, "n_tasks": 1, "n_scored": 1}},
                    "n_tasks_selected": 1, "tasks_selected": ["a"]}, f)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(S.subprocess, "run", run)
    try:
        res = S.run_skillsbench("m", upstream, 1, enable_thinking=thinking)
    finally:
        srv.shutdown()
        srv.server_close()
    return res, seen, bodies, upstream


def test_thinking_run_sets_temperature_one_on_the_agents_requests(monkeypatch):
    res, seen, bodies, upstream = _run_with_agent_request(monkeypatch, thinking=True)
    assert seen["model_url"] != upstream                     # through gbench's relay
    assert [b["temperature"] for b in bodies] == [1.0]       # the agent sent 0.0
    assert res["temperature_applied"] is True and "temperature_note" not in res
    assert (res["temperature"], res["temperature_source"]) == (1.0, "thinking")
    assert res["model_relay"]["sets_body_fields"] == {
        "temperature": 1.0, "chat_template_kwargs": {"enable_thinking": True}}
    assert "adds" not in res["model_relay"]
    assert res["leaderboard_comparable"] is False
    assert "temperature 1.0 set by thinking on every model request" in (
        res["leaderboard_comparable_reason"])
    assert res["sampling"].startswith("temperature 1.0 (thinking) set by gbench's relay")


def test_thinking_run_operator_temperature_wins(monkeypatch):
    res, _, bodies, _ = _run_with_agent_request(
        monkeypatch, thinking=True, env={"GBENCH_SKILLSBENCH_TEMPERATURE": "0.6"})
    assert [b["temperature"] for b in bodies] == [0.6]
    assert res["temperature_source"] == "env:GBENCH_SKILLSBENCH_TEMPERATURE"


def test_no_think_run_sets_temperature_zero_on_the_agents_requests(monkeypatch):
    # The maintainer's rule: a no-think run samples at 0.0, whatever the agent sends (0.7 here),
    # through the same relay as a --thinking run.
    res, seen, bodies, upstream = _run_with_agent_request(monkeypatch, thinking=False,
                                                          sent_temperature=0.7)
    assert seen["model_url"] != upstream                     # through gbench's relay
    assert [b["temperature"] for b in bodies] == [0.0]
    assert res["temperature_applied"] is True and "temperature_note" not in res
    assert (res["temperature"], res["temperature_source"]) == (0.0, "default")
    assert res["model_relay"]["sets_body_fields"] == {
        "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    assert "adds" not in res["model_relay"]
    assert res["leaderboard_comparable"] is False
    assert "temperature 0.0 set by default on every model request" in (
        res["leaderboard_comparable_reason"])
    assert res["sampling"].startswith("temperature 0.0 (default) set by gbench's relay")


def test_no_think_run_operator_temperature_wins(monkeypatch):
    res, _, bodies, _ = _run_with_agent_request(
        monkeypatch, thinking=False, env={"GBENCH_SKILLSBENCH_TEMPERATURE": "0.6"})
    assert [b["temperature"] for b in bodies] == [0.6]
    assert res["temperature_source"] == "env:GBENCH_SKILLSBENCH_TEMPERATURE"
