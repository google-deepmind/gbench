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

"""The agent clients of terminal_bench, tau2/tau3, agent_dojo, complexfuncbench and skillsbench
carry a request timeout sized from the served window or the turn budget, never a client default
of 600 s, and the Gemini judge client is bounded.

2026-10-10 investigation: LiteLLM (1.91 in the Harbor image, 1.81 in the tau2 image) and the
OpenAI SDK give a request 600 s when none is set, and retry it, so a model turn that needed more
than 600 s of decode was dropped and asked again (a hidden cap on the turn's length);
complexfuncbench fixed its timeout at 600 s; BenchFlow's ACP idle watchdog killed and re-ran a
skillsbench trial after 600 s without an ACP event (the 2026-10-10 smoke run, debug-trl-grpo);
skillsbench and agent_dojo never sent the chat template's thinking switch, so a --thinking run
reached the model with thinking off; and gbench's own Gemini judge client had no timeout.

No network, no docker: containers and endpoints are faked (the in-image and live checks are in
the docs/evals pages of each suite).
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import types
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from gbench.runners.eval_suites import agent_dojo as AD
from gbench.runners.eval_suites import base as B
from gbench.runners.eval_suites import complexfuncbench as CFB
from gbench.runners.eval_suites import skillsbench as SB
from gbench.runners.eval_suites import tau_common as TC
from gbench.runners.eval_suites import terminal_bench as TB

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "gbench", "docker"))
import skillsbench_run as SRUN  # noqa: E402

WINDOW = 262144
#: base.request_timeout_s(262144) at the defaults (1200 s floor + 262144 / 8 tok/s)
T_WINDOW = B.request_timeout_s(WINDOW)


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setitem(B._RUN_KNOBS, "max_model_len", WINDOW)


def test_the_window_timeout_is_far_above_600_s():
    assert T_WINDOW == B.request_timeout_s(B.UNCAPPED_TIMEOUT_TOKENS) > 600
    assert B.request_timeout_s(65536) > 546          # one 65536-token reply at 120 tok/s


# --------------------------------------------------------------------------- #
# terminal_bench: llm_call_kwargs timeout / max_retries reach Harbor's LiteLLM
# --------------------------------------------------------------------------- #
def _tb_argv(monkeypatch, thinking=True, env=None):
    for k in ("GBENCH_TB_TURN_MAX_TOKENS", "GBENCH_TB_TIMEOUT_MULTIPLIER", "GBENCH_TB_PARSER",
              "GBENCH_TB_CONTAINER_THREADS", "GBENCH_TERMINAL_BENCH_TEMPERATURE", "GBENCH_SHARD"):
        monkeypatch.delenv(k, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    seen = {}
    monkeypatch.setattr(TB, "check_terminal_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(TB, "require_task_metadata_blocked", lambda task_names=None: None)
    monkeypatch.setattr(TB, "prepare_task_images",
                        lambda task_names=None: ({"cached": 0}, {"present": 0}))

    def popen(cmd, *a, **k):
        seen["cmd"] = cmd
        raise RuntimeError("argv captured")
    monkeypatch.setattr(TB.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError):
        TB.run_terminal_bench("m", "http://x:8000/v1", 2, enable_thinking=thinking)
    cmd = seen["cmd"]
    kw = dict(cmd[i + 1].split("=", 1) for i, a in enumerate(cmd) if a == "--agent-kwarg")
    return json.loads(kw["llm_call_kwargs"])


def test_terminal_bench_sends_the_window_timeout_and_no_sdk_retries(monkeypatch, window):
    llm = _tb_argv(monkeypatch)
    assert llm["timeout"] == T_WINDOW and llm["max_retries"] == 0
    assert llm["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert "max_tokens" not in llm
    rec = TB.model_client_timeout(0)
    assert rec["timeout_s"] == T_WINDOW and rec["retries"] == 0 and rec["hidden_cap_removed"]
    assert rec["version"] == TB.HARNESS_LITELLM_VERSION == "1.91.0"


def test_terminal_bench_sizes_the_timeout_from_an_opt_in_turn_cap(monkeypatch, window):
    llm = _tb_argv(monkeypatch, env={"GBENCH_TB_TURN_MAX_TOKENS": "32768"})
    assert llm["max_tokens"] == 32768 and llm["timeout"] == B.request_timeout_s(32768)


def test_terminal_bench_without_a_known_window_uses_the_uncapped_size(monkeypatch):
    monkeypatch.delitem(B._RUN_KNOBS, "max_model_len", raising=False)
    assert TB.model_client_timeout(0)["timeout_s"] == B.request_timeout_s(B.UNCAPPED_TIMEOUT_TOKENS)


# --------------------------------------------------------------------------- #
# tau2 / tau3: the agent's llm_args carry the timeout; the wire shape is counted
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("thinking", [False, True])
def test_tau_agent_args_carry_the_window_timeout(monkeypatch, window, thinking):
    monkeypatch.delenv("GBENCH_TAU2_TEMPERATURE", raising=False)
    args = TC._build_agent_args("http://x/v1", thinking)
    assert args["timeout"] == T_WINDOW
    if thinking:
        assert args["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    else:
        assert "extra_body" not in args               # upstream's no-think request
    rec = TC.agent_client_timeout()
    assert rec["timeout_s"] == T_WINDOW and rec["retries"] == 3 and rec["hidden_cap_removed"]
    assert rec["version"] == TC.TAU2_IMAGE_LITELLM_VERSION


def test_tau_counts_the_agent_requests_by_thinking_switch(monkeypatch):
    TC.reset_agent_wire()
    calls = []
    guarded = TC._agent_guard(lambda *a, **k: calls.append(k) or "reply")
    monkeypatch.setattr(TC, "_AGENT_ROUTE", {"model": "openai/m", "api_base": "http://x/v1",
                                             "api_key": "EMPTY"})
    for _ in range(2):
        guarded(model="openai/m", api_base="http://x/v1", timeout=T_WINDOW,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}}, messages=[])
    guarded(model="gemini/g", messages=[])            # a user-simulator call: not counted
    rec = TC.agent_wire_record(True)
    assert rec["thinking_applied"] is True
    assert rec["agent_request_shape"]["requests"] == 2
    assert rec["agent_request_shape"]["enable_thinking"] == {"True": 2}
    assert rec["agent_request_shape"]["timeout_s"] == {str(T_WINDOW): 2}
    assert TC.agent_wire_record(False)["thinking_applied"] is False
    TC.reset_agent_wire()
    guarded(model="openai/m", api_base="http://x/v1", messages=[])
    assert TC.agent_wire_record(False)["thinking_applied"] is True
    assert TC.agent_wire_record(True)["thinking_applied"] is False
    TC.reset_agent_wire()
    assert TC.agent_wire_record(True)["thinking_applied"] is False   # no request seen


def test_tau_passes_the_window_into_the_image_only_when_known(monkeypatch):
    knobs = {"temperature": 0.7, "attempt_count": None}
    assert TC._image_run_knobs(lambda k: knobs.get(k)) == knobs
    knobs["max_model_len"] = WINDOW
    assert TC._image_run_knobs(lambda k: knobs.get(k))["max_model_len"] == WINDOW


# --------------------------------------------------------------------------- #
# agent_dojo: the OpenAI client's timeout / retries and the thinking switch
# --------------------------------------------------------------------------- #
def test_agent_dojo_builds_its_client_with_the_window_timeout(monkeypatch, window):
    seen = {}

    def openai_client(**kw):
        seen.update(kw)
        raise RuntimeError("client captured")
    monkeypatch.setattr(AD, "_load_agentdojo", lambda: {
        "version": AD.AGENTDOJO_VERSION,
        "openai": types.SimpleNamespace(OpenAI=openai_client, __version__="2.31.0")})
    monkeypatch.setattr(AD, "_served_model_id", lambda m, u: (m, "argument"))
    with pytest.raises(RuntimeError):
        AD.run_agent_dojo("m", "http://x/v1", enable_thinking=True)
    # the read/write/pool timeout is T; the TCP connect keeps a short bound of its own (a bare
    # float would have applied T to the connect too: 2026-10-11 review)
    t = seen["timeout"]
    assert (t.read, t.write, t.pool, t.connect) == (T_WINDOW, T_WINDOW, T_WINDOW,
                                                    AD.CONNECT_TIMEOUT_S == 10.0 and 10.0)
    assert seen["max_retries"] == 0
    rec = AD.model_client_timeout("2.31.0")
    assert rec["timeout_s"] == T_WINDOW and rec["retries"] == 0 and rec["outer_attempts"] == 3
    assert rec["connect_timeout_s"] == 10.0


@pytest.mark.parametrize("thinking", [False, True])
def test_agent_dojo_requests_carry_the_thinking_switch(thinking):
    sent_kwargs, counted = [], {}
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
        create=lambda **kw: sent_kwargs.append(kw) or "reply")))
    AD._send_temperature(client, 1.0 if thinking else 0.0, thinking, counted)
    client.chat.completions.create(model="m", messages=[], extra_body={"top_k": 64})
    assert sent_kwargs[0]["extra_body"] == {
        "top_k": 64, "chat_template_kwargs": {"enable_thinking": thinking}}
    assert counted == {str(thinking): 1}


def test_agent_dojo_without_the_switch_argument_sends_only_the_temperature():
    sent_kwargs = []
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
        create=lambda **kw: sent_kwargs.append(kw) or "reply")))
    AD._send_temperature(client, 0.0)
    client.chat.completions.create(model="m", messages=[])
    assert sent_kwargs == [{"model": "m", "messages": [], "temperature": 0.0}]


# --------------------------------------------------------------------------- #
# complexfuncbench: the timeout follows the turn budget
# --------------------------------------------------------------------------- #
def test_complexfuncbench_sizes_the_timeout_from_the_turn_budget(monkeypatch, window):
    monkeypatch.delenv(CFB.REQUEST_TIMEOUT_ENV, raising=False)
    assert CFB.model_client_timeout(2048)["timeout_s"] == B.request_timeout_s(2048) == 1456
    rec = CFB.model_client_timeout(65536)
    assert rec["timeout_s"] == B.request_timeout_s(65536) == 9392
    assert rec["retries"] == CFB._REQUEST_ATTEMPTS - 1 == 4 and rec["hidden_cap_removed"]
    assert CFB.model_client_timeout(None)["timeout_s"] == T_WINDOW


def test_complexfuncbench_operator_timeout_wins_and_is_recorded(monkeypatch):
    monkeypatch.setenv(CFB.REQUEST_TIMEOUT_ENV, "600")
    rec = CFB.model_client_timeout(65536)
    assert rec["timeout_s"] == 600 and rec["hidden_cap_removed"] is False
    assert CFB.REQUEST_TIMEOUT_ENV in rec["source"]


def test_complexfuncbench_driver_uses_the_sized_timeout(monkeypatch, window):
    monkeypatch.delenv(CFB.REQUEST_TIMEOUT_ENV, raising=False)
    import aiohttp
    seen = {}
    real = aiohttp.ClientTimeout

    def timeout(**kw):
        seen.update(kw)
        raise RuntimeError("timeout captured")
    monkeypatch.setattr(aiohttp, "ClientTimeout", timeout)
    monkeypatch.setattr(CFB, "_load_dataset", lambda limit: [{"id": "Car-Rental-1"}])
    monkeypatch.setattr(CFB, "_load_exact_match_dict", lambda: {})
    monkeypatch.setattr(CFB, "_get_embedder", lambda: object())
    monkeypatch.setattr(CFB, "_load_tool_info", lambda: {})
    monkeypatch.setattr(CFB, "_RapidAPICall", lambda *a, **k: object())
    monkeypatch.setattr(CFB, "current_shard", lambda: None)
    with pytest.raises(RuntimeError):
        asyncio.run(CFB._drive_all("m", "http://x/v1", 1, 1.0, None, max_output_tokens=65536,
                                   enable_thinking=True))
    assert seen == {"total": 9392, "sock_connect": CFB.CONNECT_TIMEOUT_S}
    assert CFB.model_client_timeout(65536)["connect_timeout_s"] == CFB.CONNECT_TIMEOUT_S == 10.0
    assert real is not timeout


# --------------------------------------------------------------------------- #
# skillsbench: the relay sets the thinking switch; the launcher gets both timeouts
# --------------------------------------------------------------------------- #
def _upstream():
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


def _skillsbench(monkeypatch, thinking):
    srv, bodies = _upstream()
    for k in ("GBENCH_SKILLSBENCH_API_KEY", "VLLM_API_KEY", "OPENAI_API_KEY",
              "GBENCH_SKILLSBENCH_TASK_ENDPOINT", "GBENCH_SKILLSBENCH_TEMPERATURE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GBENCH_SKILLSBENCH_TASK_HOST", "127.0.0.1")
    monkeypatch.setattr(SB, "check_skillsbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(SB.swebench_common, "require_task_network_blocked", lambda *a, **k: None)
    monkeypatch.setattr(SB, "dns_harness_errors", lambda workdir, summary: ({}, {}))
    monkeypatch.setattr(SB, "keep_artifacts", lambda workdir, dest: None)
    monkeypatch.setattr(SB, "remove_workdir", lambda workdir, image=None: [])
    seen = {}

    def run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            env = dict(a.split("=", 1) for a in cmd if "=" in a and a.split("=", 1)[0].isupper())
            seen["env"] = env
            # the agent's ChatOpenAI request: neither a temperature nor a thinking switch
            req = urllib.request.Request(
                env["GBENCH_MODEL_BASE_URL"] + "/chat/completions",
                data=json.dumps({"model": "m", "messages": [{"role": "user",
                                                             "content": "hi"}]}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            urllib.request.urlopen(req, timeout=30).read()
            with open(os.path.join(env["SKILLSBENCH_WORKDIR"], "skillsbench_summary.json"),
                      "w") as f:
                json.dump({"conditions": {"without-skills": {
                    "per_task": {"a": 1.0}, "mean_reward": 1.0, "n_tasks": 1, "n_scored": 1}},
                    "n_tasks_selected": 1, "tasks_selected": ["a"],
                    "agent_idle_timeout_s": env.get("SKILLSBENCH_AGENT_IDLE_TIMEOUT_S"),
                    "model_request_timeout_s": env.get("GBENCH_MODEL_REQUEST_TIMEOUT_S")}, f)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(SB.subprocess, "run", run)
    try:
        res = SB.run_skillsbench("m", f"http://127.0.0.1:{srv.server_address[1]}/v1", 1,
                                 enable_thinking=thinking)
    finally:
        srv.shutdown()
        srv.server_close()
    return res, seen, bodies


@pytest.mark.parametrize("thinking", [False, True])
def test_skillsbench_relay_puts_the_thinking_switch_on_the_wire(monkeypatch, window, thinking):
    res, seen, bodies = _skillsbench(monkeypatch, thinking)
    assert [b["chat_template_kwargs"] for b in bodies] == [{"enable_thinking": thinking}]
    assert [b["temperature"] for b in bodies] == [1.0 if thinking else 0.0]
    assert res["thinking"] is thinking and res["thinking_applied"] is True
    assert res["model_relay"]["sets_body_fields"]["chat_template_kwargs"] == {
        "enable_thinking": thinking}


def test_skillsbench_passes_both_timeouts_and_records_them(monkeypatch, window):
    res, seen, bodies = _skillsbench(monkeypatch, True)
    assert seen["env"]["SKILLSBENCH_AGENT_IDLE_TIMEOUT_S"] == str(T_WINDOW)
    assert seen["env"]["GBENCH_MODEL_REQUEST_TIMEOUT_S"] == str(T_WINDOW)
    assert res["agent_idle_timeout"]["timeout_s"] == T_WINDOW
    assert res["agent_idle_timeout"]["hidden_cap_removed"] is True
    assert res["model_client_timeout"]["timeout_s"] == T_WINDOW
    assert res["model_client_timeout"]["hidden_cap_removed"] is False   # ChatOpenAI had none
    assert res["launcher_timeouts"] == {"agent_idle_timeout_s": str(T_WINDOW),
                                        "model_request_timeout_s": str(T_WINDOW)}


@pytest.mark.parametrize("agent", ["oracle", "codex"])
def test_skillsbench_thinking_applied_is_false_where_the_switch_cannot_reach_the_wire(agent):
    # 2026-10-11 review: a literal thinking_applied true was wrong for these agents
    rec = SB.thinking_record(agent, {"completion_replies": 3})
    assert rec["thinking_applied"] is False and rec["thinking_not_applied_reason"]


def test_skillsbench_thinking_applied_needs_a_relayed_completions_reply():
    assert SB.thinking_record("deepagents", {"completion_replies": 1}) == {
        "thinking_applied": True}
    for rec in ({"completion_replies": 0}, None):
        out = SB.thinking_record("deepagents", rec)
        assert out["thinking_applied"] is False and "no chat-completions" in (
            out["thinking_not_applied_reason"])


def test_skillsbench_records_only_the_timeouts_the_agent_uses(window):
    t = SB.transport_timeouts()
    assert set(SB.agent_timeouts("deepagents", t)) == {"agent_idle_timeout",
                                                       "model_client_timeout"}
    assert set(SB.agent_timeouts("codex", t)) == {"agent_idle_timeout"}
    assert SB.agent_timeouts("oracle", t) == {}


def test_skillsbench_launcher_passes_the_idle_flag_and_the_agent_timeout(monkeypatch):
    monkeypatch.setenv("GBENCH_MODEL_NAME", "m")
    monkeypatch.setenv("GBENCH_MODEL_BASE_URL", "http://172.17.0.1:5/v1")
    monkeypatch.setenv("SKILLSBENCH_AGENT_IDLE_TIMEOUT_S", str(T_WINDOW))
    monkeypatch.setenv("GBENCH_MODEL_REQUEST_TIMEOUT_S", str(T_WINDOW))
    seen = {}
    monkeypatch.setattr(SRUN.subprocess, "run",
                        lambda cmd, **k: seen.setdefault("cmd", cmd)
                        and subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr(SRUN, "_reward_from_jobs", lambda jobs_dir: 1.0)
    SRUN._run_one("t", "without-skills", "deepagents", "/tmp/j")
    cmd = seen["cmd"]
    assert cmd[cmd.index("--agent-idle-timeout") + 1] == str(T_WINDOW)
    assert f"GBENCH_MODEL_REQUEST_TIMEOUT_S={T_WINDOW}" in cmd
    seen.clear()
    monkeypatch.delenv("SKILLSBENCH_AGENT_IDLE_TIMEOUT_S")
    monkeypatch.delenv("GBENCH_MODEL_REQUEST_TIMEOUT_S")
    SRUN._run_one("t", "without-skills", "deepagents", "/tmp/j")
    assert "--agent-idle-timeout" not in seen["cmd"]      # BenchFlow's default when unset


def test_skillsbench_image_pins_the_patch_and_the_agent():
    df = open(os.path.join(REPO, "gbench", "docker", "skillsbench.Dockerfile")).read()
    patch = open(os.path.join(REPO, "gbench", "docker",
                              "skillsbench_benchflow_transport.patch")).read()
    assert "COPY skillsbench_benchflow_transport.patch" in df
    assert f"ARG DEEPAGENTS_VERSION={SB.DEEPAGENTS_VERSION}" in df
    assert f"ARG LANGCHAIN_OPENAI_VERSION={SB.LANGCHAIN_OPENAI_VERSION}" in df
    assert (f'"deepagents=={SB.DEEPAGENTS_VERSION} '
            f'langchain-openai=={SB.LANGCHAIN_OPENAI_VERSION} && "') in patch
    assert "--- a/benchflow/agents/deepagents_acp_shim.py" in patch
    assert 'kwargs["timeout"] = request_timeout' in patch
    assert f"docker build -t {SB._IMAGE_DEFAULT} " in df
    assert f'LABEL {SB.HARNESS_LABEL}="{SB.HARNESS_VERSION}"' in df


# --------------------------------------------------------------------------- #
# base: the Gemini judge client is bounded; a timeout is retried once on the same model
# --------------------------------------------------------------------------- #
def test_judge_clients_get_a_bounded_timeout():
    opts = B._judge_http_options()
    assert opts["timeout"] == int(B.JUDGE_REQUEST_TIMEOUT_S * 1000) == 600_000
    assert opts["base_url"] == B.GEMINI_API_BASE_URL and "retry_options" not in opts


def _judge(monkeypatch, behaviour):
    calls = []

    class _Models:
        async def generate_content(self, model, contents, config=None):
            calls.append(model)
            outcome = behaviour[model].pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return types.SimpleNamespace(text=outcome, candidates=[], prompt_feedback=None)
    client = types.SimpleNamespace(aio=types.SimpleNamespace(models=_Models()))
    monkeypatch.setattr(B, "gemini_api_keys", lambda: ["k"])
    monkeypatch.setattr(B, "_judge_clients", lambda: [client])
    monkeypatch.setattr(B, "_next_judge_client", lambda: client)
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", ",".join(behaviour))
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", "1")
    monkeypatch.setattr(B, "_JUDGE_FALLBACKS", {})
    return calls


def test_judge_timeout_is_retried_once_on_the_same_model(monkeypatch):
    calls = _judge(monkeypatch, {"a": [asyncio.TimeoutError(), "verdict"], "b": ["unused"]})
    text, model, info = asyncio.run(B.judge_generate_cascade_detailed("q"))
    assert (text, model, calls) == ("verdict", "a", ["a", "a"])
    assert B.judge_fallback_counts() == {"a: same-model retry after timeout": 1}
    assert set(info) == {"finish_reason", "max_tokens_without_text", "refused", "block_reasons"}


def test_judge_falls_back_after_a_second_timeout_and_records_why(monkeypatch):
    calls = _judge(monkeypatch, {"a": [asyncio.TimeoutError(), asyncio.TimeoutError()],
                                 "b": ["verdict"]})
    assert asyncio.run(B.judge_generate_cascade_detailed("q"))[:2] == ("verdict", "b")
    assert calls == ["a", "a", "b"]
    assert B.judge_fallback_counts() == {"a: same-model retry after timeout": 1,
                                         "a: TimeoutError": 1}


def test_judge_other_errors_go_to_the_next_model_at_once(monkeypatch):
    err = RuntimeError("429")
    err.code = 429
    calls = _judge(monkeypatch, {"a": [err], "b": ["verdict"]})
    assert asyncio.run(B.judge_generate_cascade_detailed("q"))[:2] == ("verdict", "b")
    assert calls == ["a", "b"]
    assert B.judge_fallback_counts() == {"a: RuntimeError (HTTP 429)": 1}
