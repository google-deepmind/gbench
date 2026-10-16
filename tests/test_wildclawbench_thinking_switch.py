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

"""wildclawbench sets the chat template's thinking switch on the agent's model requests.

2026-10-11 smoke run (--thinking): the relay shape check found chat_template_kwargs.enable_thinking
absent on 2 of 2 of the OpenClaw agent's requests, so the model ran with thinking off. OpenClaw
2026.3.11 sends no switch for gbench's provider (its models config declares no `reasoning`, so
pi-ai's openai-completions client sends neither enable_thinking nor reasoning_effort). The model
relay now sets enable_thinking (true with --thinking, false without, merged into any
chat_template_kwargs the request carries) next to the temperature, counts the requests that carry
it, and thinking_applied is read from that count.

No docker, no network beyond 127.0.0.1: the model endpoint and the orchestrator are faked.
"""

import json
import subprocess
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from gbench.runners.eval_suites import wildclawbench as W

_RELAY_ENV = ("GBENCH_RELAY_EXPECTED_SHAPE", "GBENCH_SMOKE_RELAY_REPLIES")


@pytest.fixture(autouse=True)
def _no_smoke_env(monkeypatch):
    for k in _RELAY_ENV:
        monkeypatch.delenv(k, raising=False)


def _fake_model_endpoint():
    seen = []

    class _H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
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
    return srv, seen


def _post(url, body):
    req = urllib.request.Request(url + "/chat/completions", method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def _relay(endpoint, thinking):
    # built as run_wildclawbench builds it
    return W.ThinkingSwitchRelay(endpoint, "127.0.0.1", key=None,
                                 body_fields=W.relay_body_fields(thinking, 1.0 if thinking else 0.0),
                                 name="wildclawbench-temperature-relay",
                                 enable_thinking=thinking).start()


def test_body_fields_carry_the_switch_in_both_modes():
    assert W.relay_body_fields(True, 1.0) == {
        "temperature": 1.0, "chat_template_kwargs": {"enable_thinking": True}}
    assert W.relay_body_fields(False, 0.0) == {
        "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.parametrize("thinking", [True, False])
def test_relay_sets_the_switch_merged_and_counts_it(thinking):
    srv, seen = _fake_model_endpoint()
    relay = _relay(f"http://127.0.0.1:{srv.server_address[1]}/v1", thinking)
    try:
        # as OpenClaw sends it: no switch; then one with other template kwargs and the opposite
        # switch (the run's mode wins, the other kwargs stay)
        _post(relay.url, {"model": "m", "temperature": 0.7,
                          "messages": [{"role": "user", "content": "hi"}]})
        _post(relay.url, {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                          "chat_template_kwargs": {"enable_thinking": not thinking, "x": 1}})
    finally:
        relay.stop()
        srv.shutdown()
        srv.server_close()
    assert [b["chat_template_kwargs"] for b in seen] == [
        {"enable_thinking": thinking}, {"enable_thinking": thinking, "x": 1}]
    assert [b["temperature"] for b in seen] == [1.0 if thinking else 0.0] * 2
    rec = relay.record()
    assert rec["thinking_switch"] == {"field": "chat_template_kwargs.enable_thinking",
                                      "value": thinking,
                                      "counted_on": ["/v1/chat/completions"],
                                      "chat_requests": 2,
                                      "chat_requests_with_switch": 2}
    assert W.thinking_record(thinking, rec) == {"thinking_applied": True}


def test_smoke_shape_check_passes_with_the_switch(monkeypatch):
    # the check that failed the 2026-10-11 smoke run: enable_thinking expected true
    monkeypatch.setenv("GBENCH_RELAY_EXPECTED_SHAPE",
                       json.dumps({"enable_thinking": True, "temperature": 1.0}))
    srv, _seen = _fake_model_endpoint()
    relay = _relay(f"http://127.0.0.1:{srv.server_address[1]}/v1", True)
    try:
        _post(relay.url, {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    finally:
        relay.stop()
        srv.shutdown()
        srv.server_close()
    rec = relay.record()
    assert rec["shape_check"]["requests_checked"] == 1
    assert rec["shape_violations"] == {}


def test_thinking_record_is_false_without_evidence():
    out = W.thinking_record(True, None)
    assert out["thinking_applied"] is False and "no chat-completions" in (
        out["thinking_not_applied_reason"])
    out = W.thinking_record(True, {"thinking_switch": {"chat_requests": 3,
                                                       "chat_requests_with_switch": 2}})
    assert out["thinking_applied"] is False
    assert out["thinking_not_applied_reason"].startswith("1 of 3 chat-completions request(s)")
    assert "enable_thinking=true" in out["thinking_not_applied_reason"]
    # a relay without the count (an EndpointRelay record) is no evidence either
    assert W.thinking_record(False, {"completion_replies": 5})["thinking_applied"] is False


def _run(monkeypatch, tmp_path, thinking):
    """run_wildclawbench with docker faked: the 'orchestrator' sends one chat request without the
    switch, as OpenClaw does, to the GBENCH_MODEL_BASE_URL it was given."""
    srv, seen = _fake_model_endpoint()
    endpoint = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    harness = tmp_path / "harness"
    harness.mkdir()
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    real_mkdtemp = W.tempfile.mkdtemp
    monkeypatch.setattr(W.tempfile, "mkdtemp", lambda prefix="", dir=None: str(workdir) if (
        prefix.startswith("gbench_wildclawbench_")) else real_mkdtemp(prefix=prefix, dir=dir))
    monkeypatch.setattr(W, "check_wildclawbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(W, "_harness_dir", lambda: str(harness))
    monkeypatch.setattr(W, "free_port", lambda preferred=None: preferred)
    monkeypatch.setattr(W.container_guard, "require_metadata_blocked", lambda *a, **k: None)
    for k in ("GBENCH_WILDCLAWBENCH_TASK_NETWORK", "GBENCH_WILDCLAWBENCH_JUDGE_PORT",
              "GBENCH_SHARD", "GBENCH_RESULTS_DIR", "GBENCH_WILDCLAWBENCH_TASK_ENDPOINT",
              "GBENCH_WILDCLAWBENCH_TEMPERATURE", "VLLM_API_KEY", "OPENAI_API_KEY",
              "GBENCH_WILDCLAWBENCH_MODEL_API_KEY"):
        monkeypatch.delenv(k, raising=False)

    def fake_run(cmd, *a, **k):
        if cmd[:3] == ["docker", "run", "--rm"] and "--name" in cmd:
            url = next(x.split("=", 1)[1] for x in cmd if x.startswith("GBENCH_MODEL_BASE_URL="))
            _post(url, {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
            (workdir / "wildclawbench_summary.json").write_text(json.dumps({
                "n_tasks": 1, "n_scored": 1, "n_headline_tasks": 1, "global_mean": 0.5,
                "n_missing": 0, "judge_fallback_tasks": 0, "n_search_setup_failures": 0,
                "per_task": [{"task_id": "a", "overall_score": 0.5, "harness_errors": []}]}))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="orchestrator log")
    monkeypatch.setattr(W.subprocess, "run", fake_run)
    try:
        return W.run_wildclawbench("m", endpoint, enable_thinking=thinking), seen
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.mark.parametrize("thinking", [True, False])
def test_run_sends_the_switch_and_records_it(monkeypatch, tmp_path, thinking):
    result, seen = _run(monkeypatch, tmp_path, thinking)
    assert [b["chat_template_kwargs"] for b in seen] == [{"enable_thinking": thinking}]
    assert result["thinking"] is thinking and result["thinking_applied"] is True
    assert "thinking_not_applied_reason" not in result
    relay = result["model_temperature_relay"]
    assert relay["sets_body_fields"] == W.relay_body_fields(thinking, 1.0 if thinking else 0.0)
    assert relay["thinking_switch"]["chat_requests"] == 1
    assert relay["thinking_switch"]["chat_requests_with_switch"] == 1
    assert relay["sets_thinking_switch_on"] == ["/v1/chat/completions", "/v1/completions"]
    assert relay["thinking_switch"]["counted_on"] == ["/v1/chat/completions"]
    assert f"enable_thinking={str(thinking).lower()}" in result["sampling"]
    assert "thinking switch not applied" not in result["leaderboard_comparable_reason"]


def test_merge_is_applied_only_when_every_shard_applied_it():
    base = {"raw_summary": {"per_task": []}, "leaderboard_comparable_reason": ""}
    ok = dict(base, shard="1/2", thinking_applied=True)
    bad = dict(base, shard="2/2", thinking_applied=False,
               thinking_not_applied_reason="the relay counted no chat-completions request")
    assert W.merge_shards([ok, dict(ok, shard="2/2")])["thinking_applied"] is True
    merged = W.merge_shards([ok, bad])
    assert merged["thinking_applied"] is False
    assert merged["thinking_not_applied_reason"] == (
        "shard 2/2: the relay counted no chat-completions request")
    # an empty shard ran nothing and does not count
    empty = dict(base, shard="2/2", empty_shard=True)
    assert W.merge_shards([ok, empty])["thinking_applied"] is True


def test_merge_sums_the_relay_switch_counts_over_the_shards():
    """2026-10-11 review: merge_shards kept shard 1's model_temperature_relay, so the
    thinking_switch counts beside the all-shard thinking_applied were shard 1's alone."""
    def shard(n, k, total):
        return {"raw_summary": {"per_task": []}, "leaderboard_comparable_reason": "",
                "shard": f"{k}/{total}", "thinking_applied": True,
                "model_temperature_relay": {"thinking_switch": {
                    "field": "chat_template_kwargs.enable_thinking", "value": True,
                    "chat_requests": n, "chat_requests_with_switch": n}}}
    empty = {"raw_summary": {"per_task": []}, "leaderboard_comparable_reason": "",
             "shard": "3/3", "empty_shard": True}
    merged = W.merge_shards([shard(4, 1, 3), shard(7, 2, 3), empty])
    switch = merged["model_temperature_relay"]["thinking_switch"]
    assert (switch["chat_requests"], switch["chat_requests_with_switch"]) == (11, 11)
    assert switch["summed_over_shards"] == 2 and merged["thinking_applied"] is True
