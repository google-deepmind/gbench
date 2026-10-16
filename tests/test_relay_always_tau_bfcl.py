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

"""tau2 / tau3 and bfcl_v4_agentic route the model under test through gbench's endpoint relay on a
keyless endpoint too.

2026-10-13 smoke run smoke-10-12-235114: tau_common._bridge_services started the agent's
EndpointRelay only with a real endpoint key (or a proxy with credentials) and bfcl_v4_agentic's
`bfcl generate` only with a real key, so on the user's keyless vLLM the agents talked to the server
directly: the tap recorded 86 tau2, 133 tau3 and 83 bfcl_v4_agentic agent requests, none with
max_tokens, so the run's --max-output-tokens 65536 (swe_agentic.AGENT_MAX_TOKENS_SUITES, the
user's decision of 2026-10-11) never applied, and no relay request-shape check ran. The relay now
starts either way, adds a key only when there is one, and fills the run's --max-output-tokens into
agent requests that carry none. Only the agent goes through it: tau's user simulator and judge keep
the Gemini relay. tests/conftest.py keeps the keyless shortcut for the unit tests that fake docker;
these tests use the real switches (`.original`).
"""

import json
import os
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import bfcl_v4_agentic as AG
from gbench.runners.eval_suites import container_guard
from gbench.runners.eval_suites import gemini_relay as GR
from gbench.runners.eval_suites import tau_common as TC

PLACEHOLDER = "gbench-relay-placeholder-test"


class _Upstream:
    """A stub OpenAI-compatible server on 127.0.0.1 that records each request's path and JSON body
    and answers a chat completion."""

    def __init__(self):
        seen = self.seen = []
        reply = json.dumps({"id": "x", "object": "chat.completion", "model": "m",
                            "choices": [{"index": 0, "finish_reason": "stop",
                                         "message": {"role": "assistant", "content": "ok"}}],
                            "usage": {"prompt_tokens": 3, "completion_tokens": 1,
                                      "total_tokens": 4}}).encode()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = None
                seen.append({"method": self.command, "path": self.path, "body": body,
                             "authorization": self.headers.get("Authorization")})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)
            do_GET = do_POST = _reply

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def completions(self):
        return [s for s in self.seen if s["path"].endswith("/chat/completions")]

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _chat(relay_url, messages, **fields):
    """POST an agent-shaped chat request (no max_tokens unless given) to the relay."""
    body = {"model": "m", "messages": messages, "temperature": 1.0,
            "chat_template_kwargs": {"enable_thinking": True}, **fields}
    req = urllib.request.Request(relay_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, json.loads(r.read())


@pytest.fixture
def upstream():
    up = _Upstream()
    yield up
    up.close()


@pytest.fixture
def keyless(monkeypatch):
    for name in ("VLLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY",
                 "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(base._RUN_KNOBS, "max_output_tokens", 4321)
    monkeypatch.delenv("GBENCH_AGENT_MAX_TOKENS_FROM_RUN", raising=False)


# --------------------------------------------------------------------------- tau2 / tau3
def test_tau_relays_the_agent_on_every_run():
    assert TC.agent_always_relayed.original() is True


class _GeminiRelay:
    started = []

    def __init__(self, eval_name, bind, **kw):
        self.kw, self.bind = kw, bind
        self.placeholder = PLACEHOLDER
        self.origin = f"http://{bind}:4567"
        _GeminiRelay.started.append(self)

    def start(self):
        return self

    def stop(self):
        pass

    def record(self):
        return {"requests_relayed": 0, "relays_to": {"gemini": "x"}}


@pytest.fixture
def tau_bridge(monkeypatch, keyless):
    monkeypatch.setattr(TC, "agent_always_relayed", TC.agent_always_relayed.original)
    _GeminiRelay.started = []
    monkeypatch.setattr(TC, "_require_gemini_access", lambda *a: None)
    monkeypatch.setattr(GR, "network_gateway", lambda *a, **k: "127.0.0.1")
    monkeypatch.setattr(GR, "GeminiRelay", _GeminiRelay)
    probes = []
    monkeypatch.setattr(TC, "_require_relay_reachable",
                        lambda eval_name, image, host, port, docs, relay:
                        probes.append((relay, host, port)))
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-test-key-not-real")
    return probes


@pytest.mark.parametrize("suite", ["tau2", "tau3"])
def test_keyless_tau_agent_goes_through_the_relay_with_the_run_cap(tau_bridge, upstream,
                                                                  monkeypatch, suite):
    monkeypatch.setitem(base._RUN_KNOBS, "suite", suite)
    # a smoke run sets this for the agent suites: the tau relay is no smoke relay (a tau domain
    # whose every simulation ends at GBENCH_SMOKE_STOP hard-errors, losing the relay record)
    monkeypatch.setenv("GBENCH_SMOKE_RELAY_REPLIES", "1")
    with TC._bridge_services(suite, "img", upstream.url, None, ["retail"],
                             f"docs/evals/{suite}.md") as bridge:
        relay = bridge.endpoint_relay
        assert relay is not None and relay._key is None
        assert bridge.agent_base_url == relay.url
        assert bridge.env[TC._AGENT_KEY_ENV] == "EMPTY"
        # the user simulator and judge keep the Gemini relay, unchanged
        (gem,) = _GeminiRelay.started
        assert bridge.env["GEMINI_API_KEY"] == PLACEHOLDER
        assert bridge.env[GR.RELAY_URL_ENV] == gem.origin != relay.url
        first = [{"role": "user", "content": "hi"}]
        status, reply = _chat(relay.url, first)
        assert status == 200 and reply["choices"][0]["message"]["content"] == "ok"
        # the follow-up of the same conversation is forwarded too (no smoke stop)
        status, _ = _chat(relay.url, first + [{"role": "assistant", "content": "ok"},
                                              {"role": "user", "content": "more"}])
        assert status == 200
        # a request that carries its own max_tokens is left as it is
        _chat(relay.url, [{"role": "user", "content": "x"}], max_tokens=77)
        rec = bridge.record()["endpoint_relay"]
    sent = [c["body"].get("max_tokens") for c in upstream.completions()]
    assert sent == [4321, 4321, 77]
    assert all(c["authorization"] in (None, "") for c in upstream.completions())
    assert rec["max_tokens_from_run"] == {"value": 4321, "requests_filled": 2}
    assert not (rec.get("smoke") or {}).get("stops")
    assert ("endpoint relay", "127.0.0.1", TC._port_of(relay.url)) in tau_bridge


def test_keyless_tau_run_checks_the_metadata_gate_not_the_endpoint(monkeypatch, keyless):
    monkeypatch.setattr(TC, "agent_always_relayed", TC.agent_always_relayed.original)
    monkeypatch.setattr(TC, "check_role_providers", lambda: (True, ""))
    monkeypatch.setattr(base, "require_gemini_openai_base_url", lambda *a, **k: None)
    monkeypatch.setattr(TC, "check_tau_image", lambda: (True, ""))
    gates = []
    monkeypatch.setattr(container_guard, "require_metadata_blocked",
                        lambda *a, **k: gates.append((a, k)))

    def no_direct(*a, **k):
        raise AssertionError("the container no longer reaches the endpoint directly")
    monkeypatch.setattr(TC, "require_endpoint_network", no_direct)
    seen = {}

    class Stop(Exception):
        pass

    def bridge(eval_name, image, base_url, agent_base_url, domains, docs_url):
        seen.update(base_url=base_url, agent_base_url=agent_base_url)
        raise Stop
    monkeypatch.setattr(TC, "_bridge_services", bridge)
    with pytest.raises(Stop):
        TC.run_tau_suite("tau2", ["retail"], "m", "http://127.0.0.1:8000/v1", 1, 1,
                         "docs/evals/tau2.md")
    assert len(gates) == 1 and gates[0][1] == {"resolve": []}
    assert seen == {"base_url": "http://127.0.0.1:8000/v1", "agent_base_url": None}


def test_the_result_records_the_relay_fill_as_a_deviation():
    result = {"leaderboard_comparable": False,
              "leaderboard_comparable_reason": "--eval-limit subset (1 task)"}
    TC.record_agent_max_tokens(
        result, {"max_tokens_from_run": {"value": 65536, "requests_filled": 86}}, 65536)
    assert result["max_tokens_sent"] == 65536
    assert "filled in by gbench's endpoint relay (86 request(s))" in \
        result["max_output_tokens_note"]
    rec = result["official_sampling"]["agent_max_tokens_from_run"]
    assert rec["max_tokens_sent"] == 65536 and rec["requests_filled"] == 86
    assert rec["upstream"] == "tau2-bench's agent sends no max_tokens"
    assert result["leaderboard_comparable"] is False
    assert result["leaderboard_comparable_reason"] == (
        "--eval-limit subset (1 task); max_tokens=65536 (the run's --max-output-tokens) sent "
        "with 86 agent request(s) whose upstream harness sends none")


def _stamped(result):
    """A copy of `result` after the eval runner's stamp (evals._stamp_agent_max_tokens)."""
    from gbench.runners import evals
    out = json.loads(json.dumps(result))
    evals._stamp_agent_max_tokens(out)
    return out


def test_the_cap_reason_is_stated_once_after_the_eval_runner_stamp():
    """2026-10-13 review: record_agent_max_tokens and bfcl_v4_agentic named the deviation in their
    own words, and evals._stamp_agent_max_tokens added its reason for the same relay record, so
    the result carried it twice. Both now use the stamp's words, which it does not add again."""
    relay = {"max_tokens_from_run": {"value": 65536, "requests_filled": 40}}
    result = {"leaderboard_comparable": True, "leaderboard_comparable_reason": None,
              "tau2_report": {"endpoint_relay": relay}}
    TC.record_agent_max_tokens(result, relay, 65536)
    stamped = _stamped(result)
    assert stamped["leaderboard_comparable_reason"] == result["leaderboard_comparable_reason"] \
        == TC.relay_fill_reason(65536, 40)
    assert TC.RELAY_FILL_REASON.fullmatch(stamped["leaderboard_comparable_reason"])
    assert stamped["max_tokens_sent"] == 65536
    assert "; " not in TC.relay_fill_reason(65536, 40)      # merges split reasons on "; "


def test_a_relay_that_filled_no_request_is_not_recorded_as_a_cap():
    """2026-10-13 review: a relay with the fill on that filled 0 requests (no agent request went
    through it; for bfcl GBENCH_BFCL_REUSE_GENERATIONS=1 with complete files generates nothing)
    was recorded as max_tokens on every request and made the result not comparable."""
    result = {"leaderboard_comparable": True, "leaderboard_comparable_reason": None}
    TC.record_agent_max_tokens(
        result, {"max_tokens_from_run": {"value": 65536, "requests_filled": 0}}, 65536)
    assert result["max_tokens_sent"] is None and "official_sampling" not in result
    assert result["max_output_tokens_note"].startswith("not sent") and \
        "no agent request went through it" in result["max_output_tokens_note"]
    assert result["leaderboard_comparable"] is True
    assert AG.agent_max_tokens_fill(
        [{"endpoint_relay": {"max_tokens_from_run": {"value": 65536,
                                                     "requests_filled": 0}}}]) is None


def _tau_shard(shard, filled=None, empty=False):
    r = {"eval_name": "tau2", "shard": shard, "status": "success",
         "domains_requested": ["retail"], "leaderboard_comparable": False,
         "leaderboard_comparable_reason": f"shard {shard}",
         "official_sampling": {"temperature": 0.0},
         "tau2_report": {"task_outcomes_by_domain": {}}}
    if empty:
        r.update(empty_shard=True, accuracy=None)
        TC.record_agent_max_tokens(r, None, 65536)
        return r
    relay = {"max_tokens_from_run": {"value": 65536, "requests_filled": filled}}
    r["tau2_report"] = {"endpoint_relay": relay,
                        "task_outcomes_by_domain": {"retail": {f"t{shard}": [1, 1]}}}
    TC.record_agent_max_tokens(r, relay, 65536)
    return _stamped(r)


def test_a_tau_merge_sums_the_fill_and_ignores_an_empty_first_shard():
    """2026-10-13 review: merge_tau_shards started from shard 1's result, so an empty shard 1
    left the merge saying no agent request was capped, and the fill was never summed."""
    empty, a, b = _tau_shard("1/3", empty=True), _tau_shard("2/3", 10), _tau_shard("3/3", 30)
    a_official = json.dumps(a["official_sampling"])
    merged = TC.merge_tau_shards([empty, a, b])
    assert merged["max_tokens_sent"] == 65536
    assert "(40 request(s))" in merged["max_output_tokens_note"]
    assert merged["official_sampling"]["agent_max_tokens_from_run"]["requests_filled"] == 40
    assert merged["agent_max_tokens_from_run"]["requests_filled"] == 40
    assert merged["leaderboard_comparable_reason"] == TC.relay_fill_reason(65536, 40)
    assert json.dumps(a["official_sampling"]) == a_official        # the shard is untouched
    # the eval runner's stamp, run on the merge, adds nothing
    assert _stamped(merged)["leaderboard_comparable_reason"] == \
        merged["leaderboard_comparable_reason"]


def test_a_bfcl_merge_sums_the_fill_and_ignores_an_empty_first_shard(monkeypatch):
    def shard(label, filled):
        relays = [{"command": "generate", "endpoint_relay": {
            "max_tokens_from_run": {"value": 65536, "requests_filled": filled}}}]
        fill = AG.agent_max_tokens_fill(relays)
        return {"eval_name": "bfcl_v4_agentic", "shard": label, "status": "success",
                "category_accuracy": {f"memory_kv_{label[0]}": {"correct": 1, "total": 2}},
                "official_sampling": {"agent_max_tokens_from_run": {"requests_filled": filled}},
                "max_output_tokens_note": AG.max_tokens_note(fill),
                "leaderboard_comparable": False,
                "leaderboard_comparable_reason": "; ".join(
                    [f"shard {label}", TC.relay_fill_reason(65536, filled)]),
                "bfcl_report": {"relays": relays}}
    empty = {"eval_name": "bfcl_v4_agentic", "shard": "1/3", "empty_shard": True,
             "status": "success", "category_accuracy": {}, "settings": {},
             "leaderboard_comparable": False,
             "leaderboard_comparable_reason": "empty shard 1/3: no items on this node",
             "bfcl_report": {"categories_run": [], "shard": "1/3"}}
    merged = AG.merge_shards([empty, shard("2/3", 40), shard("3/3", 43)])
    assert merged["official_sampling"]["agent_max_tokens_from_run"] == {
        "requests_filled": 83, "max_tokens_sent": 65536}
    assert "(83 request(s))" in merged["max_output_tokens_note"]
    assert TC.relay_fill_reason(65536, 83) in merged["leaderboard_comparable_reason"]
    assert "40 agent request" not in merged["leaderboard_comparable_reason"]
    assert [e["shard"] for e in merged["bfcl_report"]["relays"]] == ["2/3", "3/3"]
    assert _stamped(merged)["leaderboard_comparable_reason"] == \
        merged["leaderboard_comparable_reason"]


def test_without_a_fill_the_result_says_not_sent():
    result = {"leaderboard_comparable": True, "leaderboard_comparable_reason": None}
    TC.record_agent_max_tokens(result, {"requests_relayed": 3}, 65536)
    assert result["max_tokens_sent"] is None
    assert "--max-output-tokens 65536 not applied" in result["max_output_tokens_note"]
    assert result["leaderboard_comparable"] is True and "official_sampling" not in result
    empty = {"empty_shard": True}
    TC.record_agent_max_tokens(empty, {"max_tokens_from_run": {"value": 9}}, 9)
    assert empty["max_tokens_sent"] is None and "official_sampling" not in empty


# --------------------------------------------------------------------------- bfcl_v4_agentic
def test_bfcl_generate_is_relayed_on_every_run():
    assert AG.generate_always_relayed.original() is True


@pytest.fixture
def bfcl_bridge(monkeypatch, keyless):
    monkeypatch.setattr(AG, "generate_always_relayed", AG.generate_always_relayed.original)
    gates = []
    monkeypatch.setattr(AG, "_metadata_gate",
                        lambda eval_name, docs_url, names: gates.append(eval_name))
    monkeypatch.setattr(AG, "_bridge_gateway", lambda eval_name, docs_url: "127.0.0.1")
    monkeypatch.setattr(AG, "_require_relay_reachable", lambda *a, **k: None)

    def no_direct(*a, **k):
        raise AssertionError("`bfcl generate` no longer reaches the endpoint directly")
    monkeypatch.setattr(AG, "endpoint_gate", no_direct)
    monkeypatch.setattr(TC, "require_endpoint_network", no_direct)
    return gates


@pytest.mark.parametrize("suite, cap", [("bfcl_v4_agentic", 4321), ("bfcl", None)])
def test_keyless_bfcl_bridge_relays_and_caps_only_the_agent_suite(bfcl_bridge, upstream,
                                                                 monkeypatch, suite, cap):
    monkeypatch.setitem(base._RUN_KNOBS, "suite", suite)
    with AG.bridge_services(suite, f"docs/evals/{suite}.md", upstream.url, key=None,
                            request_timeout=900,
                            always_relay=AG.generate_always_relayed()) as svc:
        relay = svc.endpoint_relay
        assert relay is not None and relay._key is None and svc.endpoint_url == relay.url
        assert relay.TIMEOUT_S >= 960
        status, _ = _chat(relay.url, [{"role": "user", "content": "hi"}])
        assert status == 200
        rec = svc.record()["endpoint_relay"]
    assert [c["body"].get("max_tokens") for c in upstream.completions()] == [cap]
    if cap:
        assert rec["max_tokens_from_run"] == {"value": cap, "requests_filled": 1}
    else:
        assert "max_tokens_from_run" not in rec
    assert bfcl_bridge == [suite]


def test_keyless_bfcl_generate_gets_the_relay_url(bfcl_bridge, upstream, monkeypatch, tmp_path):
    monkeypatch.setitem(base._RUN_KNOBS, "suite", "bfcl_v4_agentic")
    seen = {}

    def launch(cmd, env, timeout, name, stream=False):
        names = [cmd[i + 1] for i, a in enumerate(cmd[:-1])
                 if a == "-e" and "=" not in cmd[i + 1]]
        seen.update({n: env.get(n) for n in names})
        return 0, "ok"
    monkeypatch.setattr(AG, "_launch", launch)
    relays = []
    env = {"BFCL_PROJECT_ROOT": str(tmp_path), "OPENAI_BASE_URL": upstream.url,
           "REMOTE_OPENAI_BASE_URL": upstream.url, "OPENAI_API_KEY": "EMPTY"}
    with AG.suite_context("bfcl_v4_agentic", "docs/evals/bfcl_v4_agentic.md", relays=relays):
        AG._run_cli(["generate", "--model", "m"], env, None, model="m")
    (entry,) = relays
    relay_rec = entry["endpoint_relay"]
    assert relay_rec["relays_to"] == upstream.url
    assert seen["OPENAI_BASE_URL"] == seen["REMOTE_OPENAI_BASE_URL"] != upstream.url
    assert seen["OPENAI_BASE_URL"].startswith("http://127.0.0.1:")
    assert seen["OPENAI_API_KEY"] == "EMPTY"
    assert relay_rec["max_tokens_from_run"] == {"value": 4321, "requests_filled": 0}


def test_bridge_network_check_needs_the_gateway_not_the_endpoint(monkeypatch, keyless):
    calls = []
    monkeypatch.setattr(AG, "_metadata_gate", lambda *a: calls.append("metadata"))
    monkeypatch.setattr(AG, "_require_gateway_reachable",
                        lambda *a: calls.append("gateway") or "127.0.0.1")

    def no_direct(*a, **k):
        raise AssertionError("not checked: generate goes through the relay")
    monkeypatch.setattr(AG, "endpoint_gate", no_direct)
    assert AG.bridge_network_check("bfcl_v4_agentic", "http://127.0.0.1:8000/v1", "d",
                                   key=None, always_relay=True) is None
    assert calls == ["metadata", "gateway"]


def test_the_fill_is_recorded_and_named():
    relays = [{"command": "generate",
               "endpoint_relay": {"max_tokens_from_run": {"value": 65536,
                                                          "requests_filled": 40}}},
              {"command": "generate",
               "endpoint_relay": {"max_tokens_from_run": {"value": 65536,
                                                          "requests_filled": 43}},
               "gemini_relay": {"requests_relayed": 2}}]
    fill = AG.agent_max_tokens_fill(relays)
    assert fill == {"value": 65536, "requests_filled": 83}
    assert "max_tokens=65536" in AG.max_tokens_note(fill) and "83 request(s)" in \
        AG.max_tokens_note(fill)
    assert AG.agent_max_tokens_fill([{"endpoint_relay": {"requests_relayed": 5}}]) is None
    assert AG.agent_max_tokens_fill([{"endpoint_relay": None}]) is None
    assert AG.max_tokens_note(None).startswith("not sent")
