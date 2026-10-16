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

"""mcp_bench, toolbench, gaia, deepsearch_qa and gdpval: the token-limit policy and the thinking
switch in the delegated harnesses, the reference agent's caps, and the toolbench simulator base.

  * request settings (both images) - the run's thinking switch on every model-under-test request,
    the operator's temperature when set, and with --thinking the run's budget in place of
    upstream's cap, cut to the room the window leaves; reply counts appended at exit, no URL or
    key in them. The two images carry byte-identical copies.
  * mcp_bench / toolbench hosts - each mode sends what the policy says, the settings the image
    reports are checked, cut replies of a --thinking run are named, the image must carry the
    label, and URLs reach docker by name.
  * gaia                - the reference agent's max_tokens 4096 without --thinking, and with it the
    run's budget; its o1 settings (reasoning_effort "high", 8192) key on OpenAI's model name, so a
    served model gets neither, in either mode.
  * deepsearch_qa / gdpval - no max_tokens, as the notebook and the paper send none.
  * docs                - the uv interpreters of the mcp_bench image as they are.
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import deepsearch_qa as DSQA
from gbench.runners.eval_suites import gaia as GAIA
from gbench.runners.eval_suites import gdpval as GDP
from gbench.runners.eval_suites import mcp_bench as MB
from gbench.runners.eval_suites import toolbench as T

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "..")
_MB_DOCKER = os.path.join(_REPO, "gbench", "docker")
_TB_DOCKER = os.path.join(_REPO, "docker")
if _MB_DOCKER not in sys.path:
    sys.path.insert(0, _MB_DOCKER)
import mcp_bench_request_policy as POL  # noqa: E402
import mcp_bench_run as RUN  # noqa: E402

GEMINI = "AIzaSyTESTgeminiKEY0000001111"


def _read(*parts):
    with open(os.path.join(_REPO, *parts), encoding="utf-8") as fh:
        return fh.read()


def _doc(name):
    return " ".join(_read("docs", "evals", name).split())


# --------------------------------------------------------------------------- #
# a stand-in /tokenize endpoint (ephemeral port on 127.0.0.1)
# --------------------------------------------------------------------------- #
@pytest.fixture
def tokenize_server():
    seen = []
    state = {"count": 1000, "max_model_len": 32768}

    class _H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "body": body,
                         "auth": self.headers.get("Authorization")})
            out = json.dumps({"count": state["count"], "max_model_len": state["max_model_len"],
                              "tokens": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), _H)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen, state
    finally:
        server.shutdown()
        server.server_close()


def _reply(content="ok", finish="stop", tool_calls=None):
    msg = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason=finish,
                                                                 message=msg)])


def _context_error(window, asked, prompt, at_least=True):
    return RuntimeError(
        "Error code: 400 - {'error': {'message': \"This model's maximum context length is "
        f"{window} tokens. However, you requested {asked} output tokens and your prompt "
        f"contains {'at least ' if at_least else ''}{prompt} input tokens, for a total of "
        f"{prompt + asked} tokens.\"}}}}")


# --------------------------------------------------------------------------- #
# the request settings module (both images)
# --------------------------------------------------------------------------- #
def test_both_images_carry_the_same_request_settings_module():
    assert (_read("gbench", "docker", "mcp_bench_request_policy.py")
            == _read("docker", "toolbench_request_policy.py"))


def test_without_thinking_the_harness_request_keeps_its_own_cap():
    p = POL.RequestPolicy("MCP_", env={"MCP_ENABLE_THINKING": "false"})
    kw = p.prepare({"model": "m", "messages": [], "max_tokens": 12000})
    assert kw["max_tokens"] == 12000 and "temperature" not in kw
    assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    # no budget: a context rejection is passed on to the harness as it is
    assert p.refit(_context_error(32768, 12000, 20769), 12000) is None


def test_the_switch_keeps_other_template_kwargs_and_the_temperature_is_the_operators():
    p = POL.RequestPolicy("TB_", env={"TB_ENABLE_THINKING": "true", "TB_TEMPERATURE": "0.7"})
    kw = p.prepare({"model": "m", "messages": [], "max_tokens": 1024,
                    "extra_body": {"chat_template_kwargs": {"x": 1}, "top_k": 5}})
    assert kw["extra_body"] == {"chat_template_kwargs": {"x": 1, "enable_thinking": True},
                                "top_k": 5}
    assert kw["temperature"] == 0.7 and kw["max_tokens"] == 1024


def test_the_runs_temperature_reaches_the_harness_request_in_both_modes():
    # What run_toolbench / run_mcp_bench pass (TB_/MCP_TEMPERATURE: 1.0 with --thinking, 0.0
    # without, gbench's rule) replaces whatever the harness built, a 0.0 included; without the
    # variable the harness request carries none.
    for prefix in ("TB_", "MCP_"):
        on = POL.RequestPolicy(prefix, env={prefix + "ENABLE_THINKING": "true",
                                            prefix + "TEMPERATURE": "1.0"})
        assert on.prepare({"model": "m", "messages": [], "max_tokens": 1024})["temperature"] == 1.0
        off = POL.RequestPolicy(prefix, env={prefix + "ENABLE_THINKING": "false",
                                             prefix + "TEMPERATURE": "0.0"})
        kw = off.prepare({"model": "m", "messages": [], "max_tokens": 1024, "temperature": 0.7})
        assert kw["temperature"] == 0.0 and off.settings()["temperature"] == 0.0
        unset = POL.RequestPolicy(prefix, env={prefix + "ENABLE_THINKING": "false"})
        assert "temperature" not in unset.prepare({"model": "m", "messages": [],
                                                   "max_tokens": 1024})


def test_thinking_budget_is_fitted_to_the_room_the_window_leaves(tokenize_server):
    url, seen, state = tokenize_server
    env = {"MCP_ENABLE_THINKING": "true", "MCP_MAX_TOKENS": "65536"}
    p = POL.RequestPolicy("MCP_", base_url=url.replace("http://", "http://u:pw@"), env=env)
    assert p.tokenize_url == url[:-len("/v1")] + "/tokenize"        # no user info
    msgs = [{"role": "user", "content": "q"}]
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    kw = p.prepare({"model": "m", "messages": msgs, "tools": tools, "max_tokens": 12000},
                   api_key="real-key-123456")
    assert kw["max_tokens"] == 32768 - 1000 and p.counts["fitted_to_context"] == 1
    body = seen[0]["body"]
    assert seen[0]["path"] == "/tokenize" and body["messages"] == msgs and body["tools"] == tools
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["add_generation_prompt"] is True
    assert seen[0]["auth"] == "Bearer real-key-123456"
    # the whole budget fits: sent as it is; a placeholder key sends no Authorization header
    state["max_model_len"] = 131072
    kw = p.prepare({"model": "m", "messages": msgs, "max_tokens": 1024}, api_key="EMPTY")
    assert kw["max_tokens"] == 65536 and seen[-1]["auth"] is None
    # a prompt that leaves less than MIN_ROOM: sent unfitted, the server's rejection applies
    state["count"] = 131072 - 100
    kw = p.prepare({"model": "m", "messages": msgs, "max_tokens": 1024})
    assert kw["max_tokens"] == 65536


def test_a_server_rejection_is_refitted():
    p = POL.RequestPolicy("TB_", env={"TB_MAX_TOKENS": "65536"})
    # vLLM stops counting at the room it had: a lower bound -> shrink geometrically
    assert p.refit(_context_error(32768, 16384, 16385), 16384) == 4096
    # an exact count -> the room it leaves
    assert p.refit(_context_error(32768, 16384, 20000, at_least=False), 16384) == 12752
    # a max_tokens larger than the whole window
    over = ("max_tokens=65536 cannot be greater than max_model_len=max_total_tokens=32768. "
            "Please request fewer output tokens.")
    assert p.refit(over, 65536) == 32768 - POL.MIN_ROOM
    # nothing smaller keeps room for an answer; not a context error at all
    assert p.refit(_context_error(32768, 1000, 32700, at_least=False), 1000) is None
    assert p.refit(RuntimeError("rate limited"), 65536) is None


def test_sync_call_refits_counts_and_dumps_no_url_or_key(tmp_path):
    log = tmp_path / "log.jsonl"
    env = {"TB_ENABLE_THINKING": "true", "TB_MAX_TOKENS": "65536",
           "TB_REQUEST_LOG": str(log)}
    p = POL.RequestPolicy("TB_", base_url="http://u:secretpw99@127.0.0.1:1/v1", env=env)
    p.tokenize_url = None                              # no /tokenize: the reactive path
    sent = []
    replies = [_context_error(32768, 65536, 1, at_least=True), _reply("", "length"),
               _reply("partial", "length"), _reply("", "tool_calls", tool_calls=[{"id": 1}])]

    def create(**kw):
        sent.append(dict(kw))
        r = replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    for _ in range(3):
        p.call(create, {"model": "m", "messages": [], "max_tokens": 1024}, api_key="EMPTY")
    assert [k["max_tokens"] for k in sent] == [65536, 16384, 65536, 65536]
    assert p.counts == {"replies": 3, "cut_at_max_tokens": 2, "cut_in_reasoning": 1,
                        "fitted_to_context": 0, "refitted_after_rejection": 1,
                        "failed_requests": 0, "timed_out": 0, "rejected_over_context": 0,
                        "cancelled": 0}
    p.dump()
    p.dump()                                           # once per process
    lines = log.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["settings"] == {"enable_thinking": True, "max_tokens": 65536,
                               "temperature": None, "request_timeout_s": None}
    assert rec["max_tokens_sent"] == [16384, 65536]
    assert "secretpw99" not in lines[0] and "127.0.0.1" not in lines[0]


def test_async_call_passes_other_errors_on():
    p = POL.RequestPolicy("MCP_", env={"MCP_MAX_TOKENS": "4096"})

    async def create(**kw):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(p.acall(create, {"model": "m", "messages": [], "max_tokens": 1}))

    async def ok(**kw):
        return _reply("done", "stop")

    assert asyncio.run(p.acall(ok, {"model": "m", "messages": []})).choices[0].message.content \
        == "done"
    assert p.counts["replies"] == 1


# --------------------------------------------------------------------------- #
# mcp_bench launcher: the settings reach the model-under-test's provider only
# --------------------------------------------------------------------------- #
def _fake_llm_factory(monkeypatch, calls):
    class _Completions:
        def __init__(self, name):
            self.name = name

        async def create(self, **kw):
            calls.append((self.name, kw))
            return _reply("fine", "length" if kw.get("max_tokens") == 7 else "stop")

    class _Client:
        def __init__(self, name):
            self.api_key = "EMPTY"
            self.chat = types.SimpleNamespace(completions=_Completions(name))

    class LLMFactory:
        @staticmethod
        async def create_llm_provider(model_config):
            return types.SimpleNamespace(client=_Client(model_config.name))

    factory = types.ModuleType("llm.factory")
    factory.LLMFactory = LLMFactory
    llm = types.ModuleType("llm")
    llm.factory = factory
    monkeypatch.setitem(sys.modules, "llm", llm)
    monkeypatch.setitem(sys.modules, "llm.factory", factory)
    return LLMFactory


def test_launcher_applies_the_settings_to_the_model_slot_only(monkeypatch, tmp_path):
    calls = []
    factory = _fake_llm_factory(monkeypatch, calls)
    monkeypatch.setenv("MCP_ENABLE_THINKING", "true")
    monkeypatch.setenv("MCP_MAX_TOKENS", "7")
    monkeypatch.setenv("MCP_REQUEST_LOG", str(tmp_path / "req.json"))
    monkeypatch.delenv("MCP_TEMPERATURE", raising=False)
    monkeypatch.delenv("LLAMA_3_1_8B_BASE_URL", raising=False)
    monkeypatch.delenv("MCP_ENDPOINT", raising=False)
    monkeypatch.setenv("MCP_MODEL_CONFIG_NAME", "llama-3-1-8b")
    policy = RUN.install_request_policy()
    assert policy is not None

    async def go():
        model = await factory.create_llm_provider(types.SimpleNamespace(name="llama-3-1-8b"))
        other = await factory.create_llm_provider(types.SimpleNamespace(name="qwen-3-32b"))
        await model.client.chat.completions.create(model="m", messages=[], max_tokens=12000)
        await other.client.chat.completions.create(model="q", messages=[], max_tokens=3000)
    asyncio.run(go())
    (slot, kw), (other, okw) = calls
    assert slot == "llama-3-1-8b" and kw["max_tokens"] == 7
    assert kw["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert other == "qwen-3-32b" and okw == {"model": "q", "messages": [], "max_tokens": 3000}
    policy.dump()
    rec = json.loads((tmp_path / "req.json").read_text())
    assert rec["counts"]["replies"] == 1 and rec["counts"]["cut_at_max_tokens"] == 1


def test_launcher_applies_nothing_without_the_switch(monkeypatch):
    monkeypatch.delenv("MCP_ENABLE_THINKING", raising=False)
    assert RUN.install_request_policy() is None


def test_mcp_bench_image_ships_the_settings_and_the_label():
    df = _read("gbench", "docker", "mcp_bench.Dockerfile")
    assert ("COPY mcp_bench_request_policy.py /app/adapter/mcp_bench_request_policy.py" in df)
    assert f'LABEL {MB.REQUEST_POLICY_LABEL}="{MB.REQUEST_POLICY_VERSION}"' in df
    tb = _read("docker", "toolbench.Dockerfile")
    assert "COPY toolbench_request_policy.py /stb/toolbench_request_policy.py" in tb
    assert f'LABEL {MB.REQUEST_POLICY_LABEL}="{MB.REQUEST_POLICY_VERSION}"' in tb


# The passing image carries every label the two suites require (toolbench also refuses an image
# without its harness label, toolbench.HARNESS_LABEL=HARNESS_VERSION).
@pytest.mark.parametrize("labels,ok", [(json.dumps({
                                            MB.REQUEST_POLICY_LABEL: MB.REQUEST_POLICY_VERSION,
                                            MB.HARNESS_LABEL: MB.HARNESS_VERSION,
                                            T.HARNESS_LABEL: T.HARNESS_VERSION}), True),
                                       ('{"org.gbench.model-request-policy": "1"}', False),
                                       ("null", False), ('{"other": "1"}', False)])
def test_prerequisites_refuse_an_image_without_the_settings(monkeypatch, labels, ok):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI)
    monkeypatch.setattr(MB, "gemini_key_live_valid", lambda key: (True, ""))
    monkeypatch.setattr(T, "gemini_key_live_valid", lambda key: (True, ""))
    monkeypatch.setattr(T, "real_api_proxy_reachable", lambda: (True, ""))
    monkeypatch.setattr(MB.shutil, "which", lambda x: "/usr/bin/docker")

    def fake_run(cmd, **kw):
        out = labels if "--format" in cmd else ""
        fmt = cmd[cmd.index("--format") + 1] if "--format" in cmd else ""
        if fmt.startswith("{{index .Config.Labels "):
            # one label's value, as docker answers {{index .Config.Labels "<name>"}}
            name = fmt.split('"')[1]
            out = str((json.loads(labels) or {}).get(name, ""))
        return subprocess.CompletedProcess(cmd, 0, out, "")

    monkeypatch.setattr(MB.subprocess, "run", fake_run)
    monkeypatch.setattr(T.subprocess, "run", fake_run)
    for check in (MB.check_mcp_bench_prerequisites, T.check_toolbench_prerequisites):
        good, why = check()
        assert good is ok, why
        if not ok:
            assert "predates the model request settings" in why and "Rebuild" in why


# --------------------------------------------------------------------------- #
# mcp_bench host: each mode, the settings check, the reason
# --------------------------------------------------------------------------- #
_CONNECTED = {f"S{i}": {"status": "success", "tools": 3, "error": ""} for i in range(28)}


@pytest.fixture
def mb_run(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI)
    for k in MB._REQUIRED_KEYS:
        monkeypatch.setenv(k, "x" * 8 + k.lower())
    for k in ("GBENCH_MCP_BENCH_DISTRACTION", "GBENCH_MCP_BENCH_DISABLE_STABILITY",
              "GBENCH_SHARD", "GBENCH_MCP_BENCH_TEMPERATURE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(MB, "check_mcp_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(MB, "provisioning_probe", lambda image, relay_env=None: {
        "servers": [], "unavailable": [], "unbuilt": [], "network_up": True,
        "connections": _CONNECTED})
    # the judge's Gemini relay (tests/test_mcp_credential_isolation_and_dns.py covers it): a
    # stand-in, docker is stubbed
    monkeypatch.setattr(MB, "_start_judge_relay", lambda network: types.SimpleNamespace(
        client_env=lambda: {"GEMINI_API_KEY": "gbench-gemini-relay-" + "0" * 48},
        stop=lambda: None, record=lambda: {}))
    # the credentialed-servers container (tests/test_mcp_wildclawbench_credentials_and_cascade.py
    # covers it): a stand-in
    monkeypatch.setattr(MB, "start_credentialed_servers",
                        lambda image, base, network: types.SimpleNamespace(
                            client_env=lambda: {"MCP_CREDENTIALED_RELAY": "172.17.0.1:10",
                                                "MCP_CREDENTIALED_TOKEN": "t0k"},
                            stop=lambda: None, record=lambda: {}))
    runs = []

    def go(log=None, **kwargs):
        def fake_run(cmd, **kw):
            if cmd[:2] == ["docker", "run"]:
                runs.append((cmd, kw))
                out = cmd[cmd.index("-v") + 1].split(":")[0]
                metrics = {"input_schema_compliance": 1.0, "valid_tool_name_rate": 1.0,
                           "tool_call_success_rate": 1.0, "task_completion_score": 5.0,
                           "tool_selection_score": 5.0,
                           "planning_effectiveness_and_efficiency_score": 5.0}
                with open(os.path.join(out, "mcp_bench_results.json"), "w") as fh:
                    json.dump(metrics, fh)
                with open(os.path.join(out, "subset_manifest.json"), "w") as fh:
                    json.dump({"unavailable_servers": [], "tasks_kept": 104}, fh)
                with open(os.path.join(out, MB.HARNESS_LOG), "w") as fh:
                    json.dump({"version": MB.HARNESS_LOG_VERSION, "tasks": [
                        {"task_id": "t1", "file": "f.json", "status": "completed",
                         "kinds": []}]}, fh)
                if log is not None:
                    with open(os.path.join(out, MB.REQUEST_LOG), "w") as fh:
                        fh.write("".join(json.dumps(r) + "\n" for r in log(cmd)))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(MB.subprocess, "run", fake_run)
        # an endpoint URL with user info is refused before anything starts (it would sit in the
        # container where model text runs; tests/test_mcp_wildclawbench_credentials_and_cascade.py
        # covers it)
        return MB.run_mcp_bench("m", "http://x/v1", **kwargs)
    return go, runs


def _echo(counts=None, sent=(12000,)):
    """A request log whose settings are the ones the command asked for."""
    def log(cmd):
        env = dict(a.split("=", 1) for a in cmd if "=" in a and a.split("=", 1)[0].isupper())
        prefix = "MCP_" if "MCP_ENABLE_THINKING" in env else "TB_"
        raw_max = env.get(prefix + "MAX_TOKENS")
        raw_temp = env.get(prefix + "TEMPERATURE")
        raw_timeout = env.get(prefix + "REQUEST_TIMEOUT_S")
        settings = {"enable_thinking": env[prefix + "ENABLE_THINKING"] == "true",
                    "max_tokens": int(raw_max) if raw_max else None,
                    "temperature": float(raw_temp) if raw_temp else None,
                    "request_timeout_s": int(raw_timeout) if raw_timeout else None}
        return [{"settings": settings,
                 "counts": counts or {"replies": 10, "cut_at_max_tokens": 0,
                                      "cut_in_reasoning": 0, "fitted_to_context": 0,
                                      "refitted_after_rejection": 0},
                 "max_tokens_sent": list(sent)}]
    return log


def _env_names(cmd):
    return [cmd[i + 1] for i, a in enumerate(cmd) if a == "-e"]


def test_mcp_bench_no_think_sends_upstreams_caps(mb_run):
    go, runs = mb_run
    res = go(log=_echo(), max_output_tokens=65536)
    [(cmd, kw)] = runs
    names = _env_names(cmd)
    assert "MCP_ENABLE_THINKING=false" in names
    assert not any(n.startswith("MCP_MAX_TOKENS") for n in names)
    # gbench's rule: a no-think run sends temperature 0.0 (upstream sends none)
    assert "MCP_TEMPERATURE=0.0" in names
    # the endpoint goes by name
    assert "MCP_ENDPOINT" in names and "http://x/v1" not in " ".join(cmd)
    assert kw["env"]["MCP_ENDPOINT"] == "http://x/v1"
    rec = res["official_sampling"]
    assert rec["max_tokens"] == MB.UPSTREAM_PLANNING_TOKENS and rec["max_tokens_applied"] is True
    assert rec["send_temperature"] is False and rec["temperature_matches_official"] is False
    assert "summaries 10000" in rec["per_call_max_tokens"]
    assert res["max_output_tokens_note"].startswith("--max-output-tokens 65536 not applied")
    assert res["thinking"] is False and res["model_replies"]["replies"] == 10
    assert res["protocol_overrides"] == ["temperature 0.0 set by default (upstream sends none)"]


def test_mcp_bench_thinking_sends_the_run_budget_and_names_cut_replies(mb_run):
    go, runs = mb_run
    cut = {"replies": 40, "cut_at_max_tokens": 3, "cut_in_reasoning": 2,
           "fitted_to_context": 5, "refitted_after_rejection": 0}
    res = go(log=_echo(cut, sent=(30000, 65536)), enable_thinking=True, max_output_tokens=65536)
    names = _env_names(runs[0][0])
    assert "MCP_ENABLE_THINKING=true" in names and "MCP_MAX_TOKENS=65536" in names
    # the maintainer's rule: a --thinking run samples at 1.0 (upstream sends no temperature)
    assert "MCP_TEMPERATURE=1.0" in names
    rec = res["official_sampling"]
    assert rec["temperature_matches_official"] is False
    assert (res["temperature"], res["temperature_source"]) == (1.0, "thinking")
    assert any(o.startswith("temperature 1.0 set by thinking (upstream sends none")
               for o in res["protocol_overrides"])
    assert rec["max_tokens_applied"] is False and rec["max_tokens_sent"] == 65536
    assert rec["max_tokens_note"] == base.REASONING_CAP_NOTE
    assert res["max_output_tokens_note"].startswith(base.REASONING_CAP_NOTE)
    assert "cut to the room the context window leaves" in res["max_output_tokens_note"]
    reason = res["leaderboard_comparable_reason"]
    assert "3 model repl(ies) reached this run's max_tokens (2 inside the reasoning channel" in reason
    assert res["model_max_tokens_sent"] == [30000, 65536]


def test_mcp_bench_thinking_without_a_request_log_is_infra_required(mb_run):
    go, _ = mb_run
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*no model request log"):
        go(log=None, enable_thinking=True, max_output_tokens=65536)
    # without --thinking the gap is named, not fatal
    res = go(log=None)
    assert res["model_replies"] is None
    assert "no model request log" in res["leaderboard_comparable_reason"]


def test_mcp_bench_settings_the_image_did_not_apply_are_infra_required(mb_run):
    go, _ = mb_run

    def wrong(cmd):
        rec = _echo()(cmd)
        rec[0]["settings"]["enable_thinking"] = False
        return rec

    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*applied the model request settings"):
        go(log=wrong, enable_thinking=True, max_output_tokens=65536)


def test_mcp_bench_operator_temperature_is_sent_and_named(mb_run, monkeypatch):
    go, runs = mb_run
    monkeypatch.setenv("GBENCH_MCP_BENCH_TEMPERATURE", "0.3")
    res = go(log=_echo())
    assert "MCP_TEMPERATURE=0.3" in _env_names(runs[0][0])
    assert res["temperature"] == 0.3 and res["temperature_source"].startswith("env:")
    assert any("temperature 0.3" in o for o in res["protocol_overrides"])
    assert "temperature 0.3 set by GBENCH_MCP_BENCH_TEMPERATURE" in (
        res["leaderboard_comparable_reason"])


# --------------------------------------------------------------------------- #
# toolbench host
# --------------------------------------------------------------------------- #
def _tb_answer(solved):
    return {"query": "q", "answer": {"final_answer": "S" if solved else "N",
                                     "answer_details": []}}


@pytest.fixture
def tb_run(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI)
    for k in ("GBENCH_TOOLBENCH_GROUPS", "GBENCH_TOOLBENCH_REFERENCE_DIR",
              "GBENCH_TOOLBENCH_REFERENCE", "GBENCH_TOOLBENCH_EVAL_TIMES", "GBENCH_SHARD",
              "GBENCH_TOOLBENCH_SIMULATOR_MODEL", "GBENCH_TOOLBENCH_SIMULATOR_BASE",
              "GBENCH_TOOLBENCH_SIMULATOR_KEY", "GBENCH_TOOLBENCH_MAX_QUERY_COUNT",
              "GBENCH_TOOLBENCH_TEMPERATURE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(T, "check_toolbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(T, "_served_model_id", lambda endpoint, fallback: fallback)
    # the bridge network's gateway (after the metadata guard) and the image's records of its
    # tool calls: stubbed, no docker (the relays listen on loopback)
    monkeypatch.setattr(T, "_bridge_gateway", lambda image: "127.0.0.1")
    monkeypatch.setattr(T, "read_server_calls", lambda path: [])
    monkeypatch.setattr(T, "read_observations", lambda path: {})

    async def status(query, answer, sem):
        return "Solved"

    async def pref(query, cand, ref, sem):
        return "cand"

    monkeypatch.setattr(T, "_answer_status", status)
    monkeypatch.setattr(T, "_preference", pref)
    ref_root = tmp_path / "ref"
    (ref_root / T._REFERENCE_DEFAULT).mkdir(parents=True)
    for g in T._GROUPS:
        (ref_root / T._REFERENCE_DEFAULT / f"{g}.json").write_text(
            json.dumps({"1": _tb_answer(False), "2": _tb_answer(False)}))
    monkeypatch.setattr(T, "_baseline_reference_root", lambda ref, groups, image: str(ref_root))
    monkeypatch.setattr(T, "_group_test_ids",
                        lambda image, groups, limit: {g: ["1", "2"] for g in groups})
    runs = []

    def go(log=None, **kwargs):
        def fake_run(cmd, *a, **kw):
            if cmd[:2] == ["docker", "run"]:
                runs.append((cmd, kw))
                out = cmd[cmd.index("-v") + 1].split(":")[0]
                os.makedirs(os.path.join(out, "candidate"), exist_ok=True)
                for g in T._GROUPS:
                    with open(os.path.join(out, "candidate", f"{g}.json"), "w") as fh:
                        json.dump({"1": _tb_answer(True), "2": _tb_answer(True)}, fh)
                if log is not None:
                    with open(os.path.join(out, T.REQUEST_LOG), "w") as fh:
                        fh.write("".join(json.dumps(r) + "\n" for r in log(cmd)))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(T.subprocess, "run", fake_run)
        return T.run_toolbench("m", "http://x:8000/v1", 2, **kwargs)
    return go, runs


def test_toolbench_query_count_reaches_docker_by_name_and_no_simulator_base_does(tb_run,
                                                                                monkeypatch):
    go, runs = tb_run
    base_url = "https://user:hunter2pass@llm.example.com:8443/v1/?key=querysecret99"
    monkeypatch.setenv("GBENCH_TOOLBENCH_SIMULATOR_BASE", base_url)
    monkeypatch.setenv("GBENCH_TOOLBENCH_MAX_QUERY_COUNT", "150")
    res = go(log=_echo(sent=(1024,)))
    [(cmd, kw)] = runs
    argv = " ".join(cmd)
    assert "hunter2pass" not in argv and "querysecret99" not in argv
    names = _env_names(cmd)
    assert "TB_MAX_QUERY_COUNT" in names
    assert "TB_SIMULATOR_BASE" not in names and "TB_SIMULATOR_BASE" not in kw["env"]
    assert kw["env"]["TB_MAX_QUERY_COUNT"] == "150"
    # the container reaches the model through gbench's endpoint relay, never the endpoint's URL
    [endpoint] = [n for n in names if n.startswith("TB_ENDPOINT=")]
    assert endpoint.startswith("TB_ENDPOINT=http://127.0.0.1:") and "x:8000" not in endpoint
    text = json.dumps(res)
    assert "hunter2pass" not in text and "querysecret99" not in text
    assert "llm.example.com" not in text
    # the simulator stays on the Gemini API's OpenAI-compatible endpoint, named in the reason
    assert ("simulated by the gbench Gemini cascade instead of gpt-4-turbo (Gemini API "
            f"OpenAI-compatible endpoint {T.SIMULATOR_BASE}: ") in (
        res["leaderboard_comparable_reason"])


def test_toolbench_simulator_base_is_fixed_whatever_the_operator_sets(tb_run, monkeypatch):
    assert not hasattr(T, "_public_url") and not hasattr(T, "SIMULATOR_BASE_DEFAULT")
    go, _ = tb_run
    monkeypatch.setenv("GBENCH_TOOLBENCH_SIMULATOR_BASE", "https://h:abc/v1")
    res = go(log=_echo(sent=(1024,)))
    assert res["simulator"]["base_url"] == T.SIMULATOR_BASE


def test_toolbench_no_think_keeps_upstreams_1024(tb_run):
    go, runs = tb_run
    res = go(log=_echo(sent=(1024,)), max_output_tokens=65536)
    names = _env_names(runs[0][0])
    assert "TB_ENABLE_THINKING=false" in names
    assert not any(n.startswith("TB_MAX_TOKENS") for n in names)
    # gbench's rule: a no-think run sends temperature 0.0 (upstream sends none)
    assert "TB_TEMPERATURE=0.0" in names
    rec = res["official_sampling"]
    assert rec["max_tokens"] == 1024 and rec["max_tokens_applied"] is True
    assert rec["params"] == {"frequency_penalty": 0, "presence_penalty": 0}
    assert res["max_output_tokens_note"].startswith("--max-output-tokens 65536 not applied")
    assert res["model_max_tokens_sent"] == [1024] and res["thinking"] is False


def test_toolbench_thinking_sends_the_budget_and_names_cut_replies(tb_run):
    go, runs = tb_run
    cut = {"replies": 90, "cut_at_max_tokens": 4, "cut_in_reasoning": 4,
           "fitted_to_context": 0, "refitted_after_rejection": 0}
    res = go(log=_echo(cut, sent=(65536,)), enable_thinking=True, max_output_tokens=65536)
    names = _env_names(runs[0][0])
    assert "TB_ENABLE_THINKING=true" in names and "TB_MAX_TOKENS=65536" in names
    # the maintainer's rule: a --thinking run samples at 1.0 (upstream sends no temperature)
    assert "TB_TEMPERATURE=1.0" in names
    assert (res["temperature"], res["temperature_source"]) == (1.0, "thinking")
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert "temperature 1.0 set by thinking (upstream sends none" in (
        res["leaderboard_comparable_reason"])
    assert res["official_sampling"]["max_tokens_applied"] is False
    assert res["max_output_tokens_note"].startswith(base.REASONING_CAP_NOTE)
    assert "4 model repl(ies) reached this run's max_tokens (4 inside the reasoning channel" in (
        res["leaderboard_comparable_reason"])
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*no model request log"):
        go(log=None, enable_thinking=True, max_output_tokens=65536)


# the pinned upstream chat_completion_request (StableToolBench@aa4ed9f
# toolbench/inference/LLM/chatgpt_function_model.py, lines 1-50), which the build patches
_UPSTREAM_CHATGPT = '''import os

from openai import OpenAI
import httpx
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
import time
import json
import traceback

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(key, base_url, messages, tools=None, tool_choice=None, key_pos=None,
                            model="gpt-3.5-turbo", stop=None, process_id=0, **args):
    use_messages = []
    for message in messages:
        if not ("valid" in message.keys() and message["valid"] == False):
            use_messages.append(message)

    for message in use_messages:
        if 'function_call' in message.keys():
            message.pop('function_call')

    json_data = {
        "model": model,
        "messages": use_messages,
        "max_tokens": 1024,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        **args
    }
    if stop is not None:
        json_data.update({"stop": stop})
    if tools is not None:
        json_data.update({"tools": tools})
    if tool_choice is not None:
        json_data.update({"tool_choice": tool_choice})

    try:
        if model.startswith("gpt"):
            client = OpenAI(base_url=base_url, api_key=key) if base_url else OpenAI(api_key=key)
        else:
            raise NotImplementedError("Model not supported")
        openai_response = client.chat.completions.create(**json_data)
        json_data = openai_response.dict()
        return json_data

    except Exception as e:
        print("Unable to generate ChatCompletion response")
        traceback.print_exc()
        import pdb;  pdb.set_trace()
        return {"error": str(e), "total_tokens": 0}
'''


@pytest.mark.parametrize("thinking", [False, True])
def test_toolbench_patched_request_goes_through_the_settings(tmp_path, monkeypatch, thinking):
    target = tmp_path / "chatgpt_function_model.py"
    target.write_text(_UPSTREAM_CHATGPT)
    patch_src = _read("docker", "toolbench_patch_llm.py").replace(
        'PATH = "/stb/toolbench/inference/LLM/chatgpt_function_model.py"', f"PATH = {str(target)!r}")
    exec(compile(patch_src, "toolbench_patch_llm.py", "exec"), {"__name__": "__main__"})
    patched = target.read_text()
    assert "pdb.set_trace" not in patched and '"max_tokens": 1024,' in patched
    monkeypatch.setitem(sys.modules, "termcolor", types.SimpleNamespace(colored=lambda *a, **k: ""))
    monkeypatch.syspath_prepend(_TB_DOCKER)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("TB_ENABLE_THINKING", "true" if thinking else "false")
    monkeypatch.setenv("TB_REQUEST_LOG", str(tmp_path / "log.jsonl"))
    monkeypatch.delenv("TB_ENDPOINT", raising=False)
    monkeypatch.delenv("TB_TEMPERATURE", raising=False)
    if thinking:
        monkeypatch.setenv("TB_MAX_TOKENS", "65536")
    else:
        monkeypatch.delenv("TB_MAX_TOKENS", raising=False)
    monkeypatch.delitem(sys.modules, "chatgpt_function_model", raising=False)
    import importlib
    mod = importlib.import_module("chatgpt_function_model")
    sent = []

    class _Resp:
        choices = [types.SimpleNamespace(finish_reason="stop",
                                         message=types.SimpleNamespace(content="hi",
                                                                       tool_calls=None))]

        def dict(self):
            return {"choices": [{"message": {"role": "assistant", "content": "hi"}}],
                    "usage": {"total_tokens": 3}}

    class _OpenAI:
        def __init__(self, base_url=None, api_key=None):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(
                create=lambda **kw: sent.append(kw) or _Resp()))

    monkeypatch.setattr(mod, "OpenAI", _OpenAI)
    out = mod.chat_completion_request("EMPTY", "http://127.0.0.1:1/v1",
                                      [{"role": "user", "content": "q"}], model="google/gemma")
    assert out["usage"]["total_tokens"] == 3
    [kw] = sent
    assert kw["max_tokens"] == (65536 if thinking else 1024)
    assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": thinking}}
    assert kw["frequency_penalty"] == 0 and kw["presence_penalty"] == 0
    assert "temperature" not in kw
    mod._GBENCH_POLICY.dump()
    rec = json.loads((tmp_path / "log.jsonl").read_text())
    assert rec["counts"]["replies"] == 1 and rec["settings"]["enable_thinking"] is thinking


# --------------------------------------------------------------------------- #
# gaia: the reference agent's max_tokens for a served model, per mode
# --------------------------------------------------------------------------- #
def test_gaia_declares_the_reference_agents_caps():
    s = GAIA.GAIA_SAMPLING
    assert s.max_tokens == 4096 and s.send_temperature is False
    # the o1 branch (reasoning_effort "high", 8192) keys on OpenAI's model name: not declared
    assert s.reasoning is None
    assert "run_gaia.py@227ef5e49ddd" in s.reference and "every model_id but o1" in s.reference
    # without --thinking the 4096 exactly; with it the run's budget (else the thinking floor)
    for thinking, knob, policy, sent in ((False, 65536, "official", 4096),
                                         (True, None, "reasoning_run_budget",
                                          base.THINKING_MIN_OUTPUT_TOKENS),
                                         (True, 4096, "reasoning_run_budget", 4096),
                                         (True, 65536, "reasoning_run_budget", 65536)):
        plan = base.resolve_token_budget("gaia", s, thinking=thinking, max_output_tokens=knob)
        assert (plan.policy, plan.max_tokens) == (policy, sent)


def _run_loop(eval_name, sampling, tool_loop, *, thinking, extra_payload=None, knob=4096,
              replies=None):
    seen = []
    queue = list(replies or [])

    async def send(**k):
        async with k["semaphore"]:
            pass
        seen.append(k)
        if queue:
            return queue.pop(0)
        return base.Reply("FINAL ANSWER: 3", None, "stop", completion_tokens=5)

    async def tool(name, args):
        return "3"

    base._RUN_KNOBS["max_output_tokens"] = knob
    base._RUN_KNOBS["max_model_len"] = 262144
    samples = [([{"role": "user", "content": "q"}], "3", {"category": "Level_1"})]
    with mock.patch.object(base, "_send_single_request", send):
        result = base.run_eval_suite(
            eval_name=eval_name, model_name="m", base_url="http://x", concurrency=1,
            samples=samples, eval_fn=lambda r, g: True, thinking=thinking,
            official_sampling=sampling, tool_loop=tool_loop, tool_executor=tool,
            extra_payload=extra_payload)
    return seen, result


@pytest.mark.parametrize("thinking,knob,sent", [(False, 65536, 4096), (True, 4096, 4096),
                                                (True, 65536, 65536)])
def test_gaia_every_turn_sends_the_reference_budget(thinking, knob, sent):
    call = {"id": "c1", "type": "function",
            "function": {"name": "web_search", "arguments": "{\"query\": \"x\"}"}}
    payload = {"tools": [{"type": "function", "function": {"name": "web_search"}}]}
    seen, res = _run_loop("gaia", GAIA.GAIA_SAMPLING, GAIA.GAIA_TOOL_LOOP, thinking=thinking,
                          knob=knob, extra_payload=payload,
                          replies=[base.Reply("", [call], "tool_calls", completion_tokens=3)])
    assert len(seen) == 2
    # every turn: the 4096 without --thinking, the run's budget with it; temperature 0.0 without
    # --thinking, 1.0 with it (the maintainer's rule; the reference agent sends none)
    assert [k["max_output_tokens"] for k in seen] == [sent, sent]
    assert all(k["temperature"] == (1.0 if thinking else 0.0) for k in seen)
    assert res["official_sampling"]["temperature_matches_official"] is False
    # base could forward reasoning_effort, but the served-model path sends none in either mode
    assert "reasoning_effort" in base.ALLOWED_API_KEYS
    assert all("reasoning_effort" not in (k["extra_payload"] or {}) for k in seen)
    rec = res["official_sampling"]
    assert "reasoning_settings" not in rec and rec["max_tokens"] == 4096
    assert rec["max_tokens_applied"] is (not thinking)
    if thinking:
        assert rec["max_tokens_sent"] == knob and rec["max_tokens_note"] == base.REASONING_CAP_NOTE


def test_run_gaia_sends_no_reasoning_effort_in_either_mode(monkeypatch):
    got = []
    monkeypatch.setattr(GAIA, "gaia_split", lambda: "validation")
    monkeypatch.setattr(GAIA, "check_gaia_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(GAIA, "search_available", lambda: True)
    monkeypatch.setattr(GAIA, "gemini_key_live_valid", lambda key: (True, ""))
    monkeypatch.setattr(GAIA, "_load_gaia_samples", lambda limit=None, split=None: [])

    def fake_suite(**kw):
        got.append(kw)
        # what base reports: without --thinking every request carries the official 4096; with it
        # the run's budget, which is no official cap
        sent = 65536 if kw["thinking"] else 4096
        traces = [{"finish_reason": "length", "response_text": "", "max_tokens_sent": sent},
                  {"finish_reason": "stop", "response_text": "FINAL ANSWER: 1",
                   "max_tokens_sent": sent}]
        return {"sample_traces": traces, "official_sampling": {"max_tokens_sent": sent},
                "official_max_tokens_sent": [] if kw["thinking"] else [4096]}

    monkeypatch.setattr(GAIA, "run_eval_suite", fake_suite)
    res = GAIA.run_gaia("m", "http://x", 1, enable_thinking=False)
    assert "reasoning_effort" not in got[-1]["extra_payload"]
    assert got[-1]["official_sampling"] is GAIA.GAIA_SAMPLING
    # a reply at the reference agent's own 4096 is its outcome, not a budget cut-off
    assert res["replies_cut_off_by_budget"] == 0 and "reasoning_effort" not in res
    assert "cut off by this run's token budget" not in res["leaderboard_comparable_reason"]
    res = GAIA.run_gaia("m", "http://x", 1, enable_thinking=True)
    assert "reasoning_effort" not in got[-1]["extra_payload"]
    assert got[-1]["official_sampling"] is GAIA.GAIA_SAMPLING
    assert "reasoning_effort" not in res
    assert res["replies_cut_off_by_budget"] == 1 and res["replies_cut_off_in_reasoning"] == 1
    assert ("1 repl(ies) cut off by this run's token budget, not by an official cap (1 of them "
            "inside the reasoning channel with no answer)") in res["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# deepsearch_qa / gdpval: no max_tokens, as their references send none
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("thinking", [False, True])
def test_deepsearch_qa_sends_no_max_tokens_on_any_turn(thinking):
    assert DSQA.DSQA_SAMPLING.send_max_tokens is False
    call = {"id": "c1", "type": "function",
            "function": {"name": "web_search", "arguments": "{\"query\": \"x\"}"}}
    seen, res = _run_loop("deepsearch_qa", DSQA.DSQA_SAMPLING, DSQA.DSQA_TOOL_LOOP,
                          thinking=thinking, knob=16384,
                          replies=[base.Reply("", [call], "tool_calls", completion_tokens=3)])
    assert [k["max_output_tokens"] for k in seen] == [None, None]
    assert res["official_sampling"]["send_max_tokens"] is False
    assert "--max-output-tokens 16384 not applied" in res["max_output_tokens_note"]


@pytest.mark.parametrize("thinking", [False, True])
def test_gdpval_sends_no_max_tokens(thinking):
    assert GDP.GDPVAL_SAMPLING.send_max_tokens is False
    plan = base.resolve_token_budget("gdpval", GDP.GDPVAL_SAMPLING, thinking=thinking,
                                     max_output_tokens=65536)
    assert plan.policy == "not_sent" and plan.max_tokens is None


def test_docs_follow_the_new_budgets():
    dsqa, gdp, gaia = _doc("deepsearch_qa.md"), _doc("gdpval.md"), _doc("gaia.md")
    assert "every turn gets the run's full output budget" not in dsqa
    assert "no turn carries a `max_tokens`" in dsqa and "`max_output_tokens_note`" in dsqa
    assert "no `max_tokens` is sent, in either mode" in gdp
    assert "every turn gets the run's full output budget" not in gaia
    assert "`max_tokens=4096`" in gaia and "`reasoning_effort=\"high\"`" in gaia
    for src in ("gaia.py", "deepsearch_qa.py"):
        text = _read("gbench", "runners", "eval_suites", src)
        assert "Each turn gets the run's full budget" not in text
        assert "the run's full budget on every turn" not in text
    for name in ("mcp_bench.md", "toolbench.md"):
        doc = _doc(name)
        assert "does not forward a thinking mode" not in doc
        assert "are therefore not forwarded" not in doc
        assert "`chat_template_kwargs.enable_thinking`" in doc
        assert "official cap not applied to reasoning" in doc
    for mod in ("mcp_bench.py", "toolbench.py"):
        header = _read("gbench", "runners", "eval_suites", mod).split('"""', 2)[1]
        assert "thinking mode is likewise not forwarded" not in header
        assert "`--temperature` / `--thinking` /" not in header
        assert "`chat_template_kwargs.enable_thinking`" in header


def test_mcp_bench_docs_name_the_uv_interpreters_as_they_are():
    doc = _doc("mcp_bench.md")
    assert "except nasa-mcp's (3.13, above)" not in doc
    assert "only for a server that pins a version the image lacks" not in doc
    for needle in ("huggingface-mcp-server (`>=3.13`), mcp-nixos (`>=3.11`) and medcalc "
                   "(`>=3.12`)",
                   "newest managed CPython the pinned uv release (0.12.19) knows, 3.14.7",
                   "`commands.json` starts all three with `python`, the image's 3.10"):
        assert needle in doc, needle
    df = " ".join(ln.lstrip("# ") for ln in
                  _read("gbench", "docker", "mcp_bench.Dockerfile").splitlines()
                  if ln.startswith("#"))
    assert "only for a server that pins a version the image lacks" not in df
    assert "huggingface-mcp-server (>=3.13), mcp-nixos (>=3.11) and medcalc (>=3.12)" in df


def test_toolbench_image_rebuilds_the_checked_package_versions():
    df = _read("docker", "toolbench.Dockerfile")
    constraint = df.index("ENV PIP_CONSTRAINT=/tmp/toolbench_constraints.txt")
    assert constraint < df.index("RUN pip install --no-cache-dir torch==2.4.1+cpu")
    pins = dict(ln.split("==", 1) for ln in _read("docker", "toolbench_constraints.txt").splitlines()
                if ln and not ln.startswith("#"))
    # the Dockerfile's own pins agree with the constraints, and the unpinned client is fixed
    assert pins["torch"] == "2.4.1+cpu" and pins["transformers"] == "4.44.2"
    assert pins["openai"] == "3.9.0" and pins["pydantic"].startswith("1.")
    assert not [c for c in _read("docker", "toolbench_constraints.txt") if ord(c) > 127]


@pytest.fixture(autouse=True)
def _metadata_guard_answers_blocked(monkeypatch):
    """Docker is stubbed in these tests: the cloud metadata guard the suites call before starting a
    container (container_guard.require_metadata_blocked) answers "unreachable" without a probe.
    tests/test_guard_tool_suites.py checks the suites' calls to it."""
    from gbench.runners.eval_suites import container_guard
    monkeypatch.setattr(container_guard, "require_metadata_blocked",
                        lambda *a, **k: container_guard.UNREACHABLE)
