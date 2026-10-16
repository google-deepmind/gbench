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

"""EndpointRelay ends a model request by progress, not duration; checks request shape; serves
smoke episodes; and makes client-side timeouts visible.

2026-10-10 smoke run: SWE-agent's litellm closed trimesh turns at 600 s at least 12 times (its
fallback timeout) and asked again, so every reply longer than 600 s of decoding was resampled,
while the relay's own 3600 s socket timeout bounded every non-streamed reply. Now:
* a model request fails only when it stops making progress: no event of its own reply stream for
  IDLE_TIMEOUT_S while the server's /metrics token counters and KV-cache usage stand still too (stalled_no_tokens,
  an endpoint-side failure, 504); a slow reply that keeps progressing, a request queued behind
  others on a busy server and a long non-streamed reply all complete; one stuck data-parallel
  engine stalls the requests it holds; a long prefill (token counters flat until its first output
  token, KV-cache usage moving: 2026-10-11 live, 12.5 s for a 254K-token prompt) is progress;
* the safety cap is request_timeout_s of the request's budget (the served window without one),
  never below TIMEOUT_S;
* with stream_upstream the relay streams a non-streamed chat request and rebuilds vLLM's
  non-streamed reply (off by default: the 2026-10-11 live check found vLLM's streaming path
  differs in 2 of 13 requests);
* expected_shape counts what actually goes upstream against the expected temperature, thinking
  switch and token budget, never blocking;
* smoke_replies=N (or GBENCH_SMOKE_RELAY_REPLIES=N, set by --smoke for agent suites only; never
  GBENCH_SMOKE alone) serves N replies per conversation, then answers GBENCH_SMOKE_STOP and
  records whether the agent came back with a grown conversation;
* a malformed relay env knob raises RelayEnvError naming it, not a ValueError a suite rewrites
  into an endpoint-URL or bind failure;
* client_close_ages records the age of each request its client abandoned, and closes near 600 s,
  1800 s or 3660 s (multi_swe_bench's client timeout) count in client_timeouts_suspected.

Time is scaled (IDLE_TIMEOUT_S of about 1 s for 300 s). No network beyond loopback fakes.
"""

import http.client
import http.server
import json
import select
import socket
import socketserver
import threading
import time
import urllib.error
import urllib.request

import pytest

from gbench.runners.eval_suites import base as B
from gbench.runners.eval_suites import swebench_common as C

IDLE = 1.0
CONFIRM = 0.3


class _Upstream:
    """A loopback endpoint. `chat(handler, body)` answers POST /v1/chat/completions; GET /metrics
    serves `engines` ({engine: [tokens, load]}), whose tokens advance by 50 per read of an
    engine named in `advancing`, and a KV-cache usage gauge per engine that changes at each read
    of an engine named in `kv_moving` (a prefill); GET /v1/models says owned_by `owner`."""

    def __init__(self, chat, owner="vllm", metrics=True):
        self.chat = chat
        self.owner = owner
        self.metrics = metrics
        self.engines = {"0": [1000.0, 0.0]}
        self.advancing = set()
        self.kv_moving = set()
        self.kv = {}
        self.seen = []
        self.disconnected = threading.Event()
        up = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, code, payload, ctype="application/json"):
                out = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def do_GET(self):
                if self.path == "/metrics":
                    if not up.metrics:
                        self._send(404, {"detail": "Not Found"})
                        return
                    lines = []
                    for eng, (tokens, load) in up.engines.items():
                        if eng in up.advancing:
                            up.engines[eng][0] = tokens = tokens + 50
                        if eng in up.kv_moving:
                            up.kv[eng] = up.kv.get(eng, 0.1) + 0.001
                        lab = f'{{engine="{eng}",model_name="m"}}'
                        lines += [f"vllm:kv_cache_usage_perc{lab} {up.kv.get(eng, 0.1)}",
                                  f"vllm:generation_tokens_total{lab} {tokens}",
                                  f"vllm:prompt_tokens_total{lab} 0.0",
                                  f"vllm:num_requests_running{lab} {load}",
                                  f"vllm:num_requests_waiting{lab} 0.0"]
                    self._send(200, ("\n".join(lines) + "\n").encode(), "text/plain")
                    return
                self._send(200, {"data": [{"id": "m", "owned_by": up.owner}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                up.seen.append(body)
                try:
                    up.chat(self, body)
                except OSError:
                    up.disconnected.set()

            def log_message(self, *a):
                pass

        self.srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def _reply(content="done", finish="stop", **extra):
    return {"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": finish, **extra}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}


def _client_gone(sock) -> bool:
    """What an ASGI server sees: the connection's end of file (or reset) while it generates."""
    r, _, _ = select.select([sock], [], [], 0)
    if not r:
        return False
    try:
        return not sock.recv(1, socket.MSG_PEEK)
    except OSError:
        return True


def _whole_after(seconds, content="done"):
    """A non-streamed reply that arrives whole after `seconds` (vLLM non-streamed); a relay that
    shuts the connection down meanwhile stops it (up.disconnected, as vLLM aborts the request)."""
    def chat(h, body):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if _client_gone(h.connection):
                raise ConnectionResetError("the relay closed the connection")
            time.sleep(0.05)
        h._send(200, _reply(content))
    return chat


def _sse(h, events, gap):
    h.send_response(200)
    h.send_header("Content-Type", "text/event-stream")
    h.end_headers()
    for ev in events:
        if gap(ev):
            time.sleep(gap(ev))
        line = ev if isinstance(ev, bytes) else b"data: " + json.dumps(ev).encode() + b"\n\n"
        h.wfile.write(line)
        h.wfile.flush()
    h.wfile.write(b"data: [DONE]\n\n")
    h.wfile.flush()
    h.close_connection = True


def _chunk(delta, finish=None, **extra):
    return {"id": "chatcmpl-s", "object": "chat.completion.chunk", "created": 7, "model": "m",
            "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish,
                         **extra}]}


def _relay(up, **kw):
    relay = C.EndpointRelay(up.url, "127.0.0.1", idle_timeout_s=IDLE, **kw)
    relay.PROGRESS_CONFIRM_S = CONFIRM
    return relay.start()


def _post(relay, body, timeout=60):
    req = urllib.request.Request(relay.url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read()


MSG = {"model": "m", "messages": [{"role": "user", "content": "fix the bug"}]}


# ============================================================================================
# progress, not duration
# ============================================================================================
def test_a_request_the_server_stopped_working_on_fails_as_stalled_after_the_idle_timeout():
    up = _Upstream(_whole_after(30))        # counters never advance: the server is stuck
    up.engines["0"][1] = 1.0                 # it holds the request
    relay = _relay(up)
    try:
        t0 = time.time()
        status, _, body = _post(relay, MSG)
        took = time.time() - t0
        rec = relay.record()
    finally:
        relay.stop()
        up.close()
    assert status == 504 and json.loads(body)["error"]["type"] == "stalled_no_tokens"
    # IDLE, then the two counter reads CONFIRM apart: not the 30 s the reply would have taken
    assert IDLE + CONFIRM - 0.1 <= took < IDLE + CONFIRM + 2.0
    # the endpoint's connection was shut down: the served endpoint stops generating
    assert up.disconnected.wait(5)
    assert rec["upstream_errors"] == {"stalled_no_tokens": 1}
    assert rec["endpoint_side_failures"] == 1 and C.endpoint_side_failure("stalled_no_tokens")
    assert C.relay_failure_count(rec) == 1


def test_a_long_prefill_moves_the_kv_cache_not_the_token_counters_and_is_progress():
    """2026-10-11 review, live on 26B-A4B DP=2: a 254K-token prefill left every token counter
    flat for 12.5 s (vLLM counts prompt tokens at a request's first output) while
    vllm:kv_cache_usage_perc changed at every read; the relay failed that healthy request as
    stalled_no_tokens. Here the counters stay flat for 3 confirm windows past the idle timeout
    while the KV cache fills: the request completes, and the request whose engine is prefilling
    does not stall a neighbour on another engine either."""
    up = _Upstream(_whole_after(IDLE + 3 * CONFIRM, "after a long prefill"))
    up.engines = {"0": [1000.0, 1.0], "1": [500.0, 1.0]}
    up.kv_moving = {"0"}                     # engine 0 prefills: only its KV cache moves
    up.advancing = {"1"}                     # engine 1 decodes
    relay = _relay(up)
    try:
        status, _, body = _post(relay, MSG)
        rec = relay.record()
    finally:
        relay.stop()
        up.close()
    assert status == 200, body
    assert json.loads(body)["choices"][0]["message"]["content"] == "after a long prefill"
    assert rec["upstream_errors"] == {}


def test_server_progressed_counts_a_kv_change_and_a_scheduled_request_as_movement():
    relay = C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1")
    assert "vllm:prefix_cache_queries_total" in relay.PROGRESS_METRICS
    assert relay.STATE_METRICS == ("vllm:kv_cache_usage_perc",)
    sp = C.EndpointRelay.server_progressed
    # tokens flat, KV usage changed (either way): moved
    assert sp({"0": (10.0, 1.0, 0.20)}, {"0": (10.0, 1.0, 0.25)})
    assert sp({"0": (10.0, 1.0, 0.25)}, {"0": (10.0, 1.0, 0.20)})
    # all flat with requests held: stalled
    assert not sp({"0": (10.0, 1.0, 0.20)}, {"0": (10.0, 1.0, 0.20)})
    # one engine prefilling, another holding requests flat: stalled (the flat one is stuck)
    assert not sp({"0": (10.0, 1.0, 0.2), "1": (5.0, 1.0, 0.3)},
                  {"0": (10.0, 1.0, 0.4), "1": (5.0, 1.0, 0.3)})
    # an idle flat engine does not count against the others
    assert sp({"0": (10.0, 1.0, 0.2), "1": (5.0, 0.0, 0.0)},
              {"0": (11.0, 1.0, 0.2), "1": (5.0, 0.0, 0.0)})


def test_a_long_non_streamed_reply_on_a_working_server_completes():
    up = _Upstream(_whole_after(6 * IDLE, "long answer"))
    up.engines["0"][1] = 1.0
    up.advancing = {"0"}                     # the server decodes: its counters advance
    relay = _relay(up)
    try:
        status, _, body = _post(relay, MSG)
        rec = relay.record()
    finally:
        relay.stop()
        up.close()
    assert status == 200 and json.loads(body)["choices"][0]["message"]["content"] == "long answer"
    assert rec["upstream_errors"] == {} and rec["completion_replies"] == 1
    # the reply is the endpoint's own: the request went as the client sent it (plus the token
    # ids the relay asks a vLLM endpoint for, add_token_ids)
    assert [{k: v for k, v in r.items() if k != "return_token_ids"} for r in up.seen] == [MSG]
    assert rec["upstream_transport"]["requests_streamed_upstream"] == 0


def test_a_slow_stream_that_keeps_progressing_completes_however_long_it_takes():
    # time-scaled 30-minute reply (IDLE = 300 s): an event every 0.6 IDLE for 6 IDLE, the
    # server's counters never advance
    events = [_chunk({"role": "assistant", "content": ""})]
    events += [_chunk({"content": f"t{i} "}) for i in range(8)]
    events += [_chunk({}, finish="stop", stop_reason=None)]
    up = _Upstream(lambda h, b: _sse(h, events, lambda ev: 0.6 * IDLE))
    relay = _relay(up)
    try:
        status, ctype, body = _post(relay, {**MSG, "stream": True})
        rec = relay.record()
    finally:
        relay.stop()
        up.close()
    assert status == 200 and "event-stream" in ctype
    assert body.count(b"data: ") == len(events) + 1 and body.endswith(b"data: [DONE]\n\n")
    assert rec["upstream_errors"] == {} and rec["completion_replies"] == 1


def test_a_request_queued_on_a_busy_server_waits_and_its_idle_is_recorded():
    events = [_chunk({"role": "assistant", "content": ""}), _chunk({"content": "ok"}),
              _chunk({}, finish="stop")]
    # 3 IDLE before the first event (the request queues), keep-alive comments meanwhile
    def chat(h, body):
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.end_headers()
        for _ in range(6):                   # keep-alive comments are not progress
            h.wfile.write(b": keep-alive\n\n")
            h.wfile.flush()
            time.sleep(0.5 * IDLE)
        _sse_rest(h, events)

    def _sse_rest(h, evs):
        for ev in evs:
            h.wfile.write(b"data: " + json.dumps(ev).encode() + b"\n\n")
        h.wfile.write(b"data: [DONE]\n\n")
        h.wfile.flush()
        h.close_connection = True
    up = _Upstream(chat)
    up.engines = {"0": [1000.0, 4.0], "1": [500.0, 3.0]}
    up.advancing = {"0", "1"}               # every engine holding requests advances
    relay = _relay(up)
    try:
        status, _, body = _post(relay, {**MSG, "stream": True})
        rec = relay.record()
    finally:
        relay.stop()
        up.close()
    assert status == 200 and b'"ok"' in body and b"error" not in body
    tr = rec["upstream_transport"]
    assert tr["idle_waits_tolerated"] >= 1 and tr["longest_tolerated_idle_s"] >= IDLE
    assert rec["upstream_errors"] == {}


def test_one_stuck_data_parallel_engine_stalls_the_request():
    up = _Upstream(_whole_after(30))
    up.engines = {"0": [1000.0, 2.0], "1": [500.0, 1.0]}
    up.advancing = {"0"}                     # engine 1 holds a request and stands still
    relay = _relay(up)
    try:
        status, _, _ = _post(relay, MSG)
        rec = relay.record()
    finally:
        relay.stop()
        up.close()
    assert status == 504 and rec["upstream_errors"] == {"stalled_no_tokens": 1}


def test_a_stream_that_stops_mid_reply_ends_with_an_error_event_and_counts_as_stalled():
    def chat(h, body):
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.end_headers()
        h.wfile.write(b"data: " + json.dumps(_chunk({"content": "half"})).encode() + b"\n\n")
        h.wfile.flush()
        time.sleep(30)
    up = _Upstream(chat)
    up.engines["0"][1] = 1.0
    relay = _relay(up)
    try:
        t0 = time.time()
        status, _, body = _post(relay, {**MSG, "stream": True})
        took = time.time() - t0
        rec = relay.record()
    finally:
        relay.stop()
        up.close()
    assert status == 200 and b'"half"' in body and b"stalled_no_tokens" in body
    assert took < IDLE + CONFIRM + 2.0
    assert rec["upstream_errors"] == {"stalled_no_tokens": 1}
    assert rec["completion_replies"] == 0     # an unfinished reply is not a reply


def test_without_metrics_a_whole_reply_waits_for_its_cap_and_a_silent_stream_stalls():
    up = _Upstream(_whole_after(3 * IDLE, "late"), metrics=False)
    relay = _relay(up)
    try:
        status, _, body = _post(relay, MSG)
    finally:
        relay.stop()
        up.close()
    # nothing tells a long whole reply from a stuck one: it is never cut by the idle timeout
    assert status == 200 and json.loads(body)["choices"][0]["message"]["content"] == "late"

    def silent(h, body):
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.end_headers()
        time.sleep(30)
    up = _Upstream(silent, metrics=False)
    relay = _relay(up)
    try:
        t0 = time.time()
        _post(relay, {**MSG, "stream": True})
        took = time.time() - t0
        rec = relay.record()
    finally:
        relay.stop()
        up.close()
    assert took < IDLE + 2.0 and rec["upstream_errors"] == {"stalled_no_tokens": 1}


def test_the_safety_cap_is_sized_from_the_budget_or_the_window(monkeypatch):
    relay = C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1")
    path = "/v1/chat/completions"
    capped = relay.request_total_s(path, json.dumps({**MSG, "max_tokens": 65536}).encode())
    assert capped == B.request_timeout_s(65536) + relay.TOTAL_CAP_MARGIN_S == 9392 + 120
    monkeypatch.setattr(B, "get_run_knob", lambda name, default=None:
                        131072 if name == "max_model_len" else default)
    window = relay.request_total_s(path, json.dumps(MSG).encode())
    assert window == B.request_timeout_s(131072) + relay.TOTAL_CAP_MARGIN_S
    monkeypatch.setattr(B, "get_run_knob", lambda name, default=None: default)
    assert relay.request_total_s(path, json.dumps(MSG).encode()) == \
        B.request_timeout_s(B.UNCAPPED_TIMEOUT_TOKENS) + relay.TOTAL_CAP_MARGIN_S
    # never below the caller's floor; the model list keeps TIMEOUT_S
    big = C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1", timeout_s=10 ** 6)
    assert big.request_total_s(path, json.dumps({**MSG, "max_tokens": 8}).encode()) == 10 ** 6
    assert relay.request_total_s("/v1/models", None) == relay.TIMEOUT_S
    assert relay.IDLE_TIMEOUT_S == 300.0
    monkeypatch.setenv("GBENCH_RELAY_IDLE_TIMEOUT_S", "45")
    assert C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1").IDLE_TIMEOUT_S == 45.0
    with pytest.raises(ValueError):
        C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1", idle_timeout_s=0)


# ============================================================================================
# stream_upstream: the non-streamed reply rebuilt from a stream (opt-in)
# ============================================================================================
def _vllm_stream(h, body, gap=0.0):
    """vLLM 0.26.1's stream for a reasoning + parallel tool call reply (the layout captured live
    on 2026-10-11)."""
    ids = body.get("return_token_ids")
    events = [
        {**_chunk({"role": "assistant", "content": ""}),
         "prompt_token_ids": [2, 105] if ids else None, "prompt_text": None},
        _chunk({"reasoning": "Weather "}, token_ids=[100, 7] if ids else None),
        _chunk({"reasoning": "for two."}, token_ids=[8, 101] if ids else None),
        _chunk({"tool_calls": [{"id": "call_a", "type": "function", "index": 0,
                                "function": {"name": "get_weather", "arguments": "{\"city\": "}}]},
               token_ids=[48, 9] if ids else None),
        _chunk({"tool_calls": [{"index": 0, "function": {"arguments": "\"Paris\"}"}}]},
               token_ids=[49] if ids else None),
        _chunk({"tool_calls": [{"id": "call_b", "type": "function", "index": 1,
                                "function": {"name": "get_weather",
                                             "arguments": "{\"city\": \"Rome\"}"}}]},
               token_ids=[48, 10, 49] if ids else None),
        _chunk({}, finish="tool_calls", stop_reason=50, token_ids=[50] if ids else None),
        {"id": "chatcmpl-s", "object": "chat.completion.chunk", "created": 7, "model": "m",
         "choices": [], "usage": {"prompt_tokens": 2, "total_tokens": 13,
                                  "completion_tokens": 11,
                                  "completion_tokens_details": {"reasoning_tokens": 4}},
         "system_fingerprint": "fp-1"},
    ]
    _sse(h, events, lambda ev: gap)


def test_with_stream_upstream_the_client_gets_vllms_non_streamed_reply():
    up = _Upstream(_vllm_stream)
    relay = _relay(up, stream_upstream=True)
    try:
        _, _, first = _post(relay, MSG)                          # relay adds return_token_ids
        _, ctype, asked = _post(relay, {**MSG, "return_token_ids": True})
        rec = relay.record()
    finally:
        relay.stop()
        up.close()
    assert up.seen[0]["stream"] is True and up.seen[0]["stream_options"] == {
        "include_usage": True}
    assert ctype == "application/json"
    want_msg = {"role": "assistant", "content": None, "refusal": None, "annotations": None,
                "audio": None, "function_call": None,
                "tool_calls": [
                    {"id": "call_a", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}},
                    {"id": "call_b", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\": \"Rome\"}"}}],
                "reasoning": "Weather for two."}
    for raw in (first, asked):
        reply = json.loads(raw)
        assert list(reply) == ["id", "object", "created", "model", "choices", "service_tier",
                               "system_fingerprint", "usage", "prompt_logprobs",
                               "prompt_token_ids", "prompt_text", "kv_transfer_params",
                               "ec_transfer_params", "metrics"]
        assert reply["object"] == "chat.completion" and reply["system_fingerprint"] == "fp-1"
        ch = reply["choices"][0]
        assert ch["message"] == want_msg
        assert (ch["finish_reason"], ch["stop_reason"]) == ("tool_calls", 50)
        assert reply["usage"] == {"prompt_tokens": 2, "total_tokens": 13,
                                  "completion_tokens": 11, "prompt_tokens_details": None,
                                  "completion_tokens_details": {"reasoning_tokens": 4}}
    # ids the relay asked for are taken out again; ids the client asked for are kept
    assert json.loads(first)["choices"][0]["token_ids"] is None
    assert json.loads(first)["prompt_token_ids"] is None
    assert json.loads(asked)["choices"][0]["token_ids"] == [100, 7, 8, 101, 48, 9, 49, 48, 10,
                                                            49, 50]
    tr = rec["upstream_transport"]
    assert tr["requests_streamed_upstream"] == 2 and tr["replies_assembled"] == 2
    assert rec["completion_replies"] == 2 and rec["upstream_errors"] == {}


def test_stream_upstream_is_off_by_default_and_skips_requests_it_cannot_rebuild(monkeypatch):
    up = _Upstream(_whole_after(0))
    relay = _relay(up)
    try:
        _post(relay, MSG)
    finally:
        relay.stop()
        up.close()
    assert "stream" not in up.seen[0]
    on = C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1", stream_upstream=True)
    on._is_vllm = True
    path = "/v1/chat/completions"
    for body in ({**MSG, "logprobs": True}, {**MSG, "n": 2}, {**MSG, "stream": True},
                 {**MSG, "stream_options": {"include_usage": True}}, {**MSG, "echo": True}):
        assert on.stream_body(path, json.dumps(body).encode())[1] is False
    assert on.stream_body("/v1/completions", json.dumps(MSG).encode())[1] is False
    assert on.stream_body(path, json.dumps(MSG).encode())[1] is True
    on._is_vllm = False
    assert on.stream_body(path, json.dumps(MSG).encode())[1] is False
    monkeypatch.setenv("GBENCH_RELAY_STREAM_UPSTREAM", "1")
    assert C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1")._stream_upstream is True


def test_a_rebuilt_reply_with_an_error_event_or_cut_short_is_an_endpoint_failure():
    def errs(h, body):
        _sse(h, [_chunk({"role": "assistant", "content": ""}),
                 {"error": {"message": "Internal server error", "type": "InternalServerError",
                            "param": None, "code": 500}}], lambda ev: 0)

    def cut(h, body):
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.end_headers()
        h.wfile.write(b"data: " + json.dumps(_chunk({"content": "par"})).encode() + b"\n\n")
        h.wfile.flush()
        h.close_connection = True
    for chat, code, kind in ((errs, 500, "http_500"), (cut, 502, "connection")):
        up = _Upstream(chat)
        relay = _relay(up, stream_upstream=True)
        try:
            status, _, body = _post(relay, MSG)
            rec = relay.record()
        finally:
            relay.stop()
            up.close()
        assert status == code and rec["upstream_errors"] == {kind: 1}
        if code == 500:
            assert json.loads(body)["error"]["type"] == "InternalServerError"


def test_a_rebuilt_reply_that_stalls_is_a_504():
    up = _Upstream(lambda h, b: _vllm_stream(h, b, gap=30))
    relay = _relay(up, stream_upstream=True)
    try:
        status, _, _ = _post(relay, MSG)
        rec = relay.record()
    finally:
        relay.stop()
        up.close()
    assert status == 504 and rec["upstream_errors"] == {"stalled_no_tokens": 1}


# ============================================================================================
# request shape
# ============================================================================================
def test_shape_violations_are_counted_on_what_goes_upstream_and_never_block():
    up = _Upstream(_whole_after(0))
    relay = _relay(up, body_fields={"chat_template_kwargs": {"enable_thinking": True}},
                   expected_shape={"temperature": 1.0, "enable_thinking": True,
                                   "max_tokens": None})
    try:
        ok = _post(relay, {**MSG, "temperature": 1.0})
        cold = _post(relay, {**MSG, "temperature": 0.0, "max_tokens": 512})
        absent = _post(relay, MSG)
        rec = relay.record()
    finally:
        relay.stop()
        up.close()
    assert [r[0] for r in (ok, cold, absent)] == [200, 200, 200] and len(up.seen) == 3
    shape = rec["shape_check"]
    assert shape["requests_checked"] == 3 and shape["requests_with_violations"] == 2
    # thinking was set by body_fields: what went upstream is what is checked
    assert rec["shape_violations"] == {"max_tokens": 1, "temperature": 2}
    assert shape["examples"][0] == {"field": "temperature", "expected": 1.0, "sent": 0.0,
                                    "conversation": C.conversation_fingerprint(MSG["messages"])}
    assert {"field": "temperature", "expected": 1.0, "sent": "<absent>",
            "conversation": C.conversation_fingerprint(MSG["messages"])} in shape["examples"]
    assert rec["upstream_errors"] == {}


def test_shape_rules_for_thinking_and_an_exact_budget():
    relay = C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1",
                            expected_shape={"enable_thinking": False, "max_tokens": 65536})
    path = "/v1/chat/completions"

    def check(body):
        return relay.check_shape(path, json.dumps(body).encode())
    assert check({**MSG, "max_tokens": 65536}) == []                  # absent switch: off
    assert check({**MSG, "max_completion_tokens": 65536,
                  "chat_template_kwargs": {"enable_thinking": True}}) == ["enable_thinking"]
    assert check({**MSG, "max_tokens": 4096}) == ["max_tokens"]
    assert check(MSG) == ["max_tokens"]
    assert relay.check_shape("/v1/models", None) == []
    with pytest.raises(ValueError):
        C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1", expected_shape={"top_p": 1})


# ============================================================================================
# smoke episodes
# ============================================================================================
def _turn(*replies):
    msgs = [{"role": "system", "content": "you are an agent"},
            {"role": "user", "content": "fix the bug"}]
    for r in replies:
        msgs += [{"role": "assistant", "content": r}, {"role": "user", "content": "obs"}]
    return {"model": "m", "messages": msgs}


def test_a_smoke_episode_serves_n_replies_then_stops_and_proves_the_agent_came_back():
    up = _Upstream(_whole_after(0, "ls"))
    relay = _relay(up, smoke_replies=1, expected_shape={"temperature": 1.0})
    try:
        first = _post(relay, {**_turn(), "temperature": 1.0})
        follow = _post(relay, {**_turn("ls"), "temperature": 1.0})
        again = _post(relay, {**_turn(), "temperature": 0.5})      # no growth: a retry
        unkeyed = _post(relay, {"model": "m", "messages": [{"role": "system", "content": "x"}]})
        other = _post(relay, {"model": "m", "messages": [{"role": "user", "content": "task 2"}]})
        rec = relay.record()
    finally:
        relay.stop()
        up.close()
    assert first[0] == 200 and unkeyed[0] == 200 and other[0] == 200
    for status, ctype, body in (follow, again):
        assert status == 400 and ctype == "application/json"
        assert json.loads(body)["error"] == {"message": "GBENCH_SMOKE_STOP",
                                             "type": "gbench_smoke_stop", "param": None,
                                             "code": 400}
    assert len(up.seen) == 3                 # stopped requests never reach the endpoint
    smoke = rec["smoke"]
    conv = smoke["conversations"][C.conversation_fingerprint(_turn()["messages"])]
    assert conv == {"requests": 3, "replies_served": 1, "served_messages": 2,
                    "shape_violations": 1, "stopped_requests": 2, "follow_up_arrived": True,
                    "stopped_without_growth": 1}
    assert rec["smoke_second_requests"] == 1 and smoke["smoke_stops"] == 2
    assert smoke["unkeyed_requests"] == 1 and smoke["conversations_served"] == 2
    # a smoke stop is not an endpoint failure
    assert rec["upstream_errors"] == {} and C.relay_failure_count(rec) == 0


def test_a_failed_reply_is_not_served_and_no_smoke_changes_nothing():
    up = _Upstream(lambda h, b: h._send(503, {"error": {"message": "down"}}))
    relay = _relay(up, smoke_replies=1)
    try:
        assert _post(relay, _turn())[0] == 503
        assert _post(relay, _turn("x"))[0] == 503     # still forwarded: nothing was served
        rec = relay.record()
    finally:
        relay.stop()
        up.close()
    conv = next(iter(rec["smoke"]["conversations"].values()))
    assert conv["replies_served"] == 0 and conv["follow_up_arrived"] is False
    up = _Upstream(_whole_after(0))
    relay = _relay(up)
    try:
        for i in range(3):
            assert _post(relay, _turn(*["r"] * i))[0] == 200
        rec = relay.record()
    finally:
        relay.stop()
        up.close()
    assert "smoke" not in rec and "smoke_second_requests" not in rec and len(up.seen) == 3
    with pytest.raises(ValueError):
        C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1", smoke_replies=0)


def test_a_streamed_smoke_reply_counts_as_served_when_the_stream_ends():
    events = [_chunk({"role": "assistant", "content": ""}), _chunk({"content": "ls"}),
              _chunk({}, finish="stop")]
    up = _Upstream(lambda h, b: _sse(h, events, lambda ev: 0))
    relay = _relay(up, smoke_replies=1)
    try:
        assert _post(relay, {**_turn(), "stream": True})[0] == 200
        assert _post(relay, {**_turn("ls"), "stream": True})[0] == 400
        rec = relay.record()
    finally:
        relay.stop()
        up.close()
    assert rec["smoke_second_requests"] == 1


# ============================================================================================
# client closes
# ============================================================================================
def test_a_client_that_gives_up_at_its_own_timeout_is_recorded_and_suspected():
    up = _Upstream(_whole_after(20))
    up.advancing = {"0"}
    up.engines["0"][1] = 1.0
    relay = _relay(up, client_timeout_suspect_s=(1.5,))
    relay.CLIENT_TIMEOUT_WINDOW_S = 0.4
    try:
        host, port = relay.url.split("//")[1].split("/")[0].split(":")
        conn = http.client.HTTPConnection(host, int(port), timeout=1.5)
        conn.request("POST", "/v1/chat/completions", body=json.dumps(MSG),
                     headers={"Content-Type": "application/json"})
        with pytest.raises(OSError):
            conn.getresponse()
        conn.close()                          # the SDK gives up and closes
        deadline = time.time() + 5
        while relay.record()["client_closed_streams"] < 1 and time.time() < deadline:
            time.sleep(0.05)
        rec = relay.record()
        stopped = up.disconnected.wait(5)    # the endpoint stops the abandoned reply
    finally:
        relay.stop()
        up.close()
    assert stopped
    ages = rec["client_close_ages"]
    assert rec["client_closed_streams"] == 1 and ages["closes_with_age"] == 1
    assert rec["client_timeouts_suspected"] == 1 and ages["client_timeouts_suspected"] == 1
    assert 1.1 <= ages["client_timeout_suspect_ages"][0] <= 1.9
    assert ages["histogram_s"]["<60"] == 1 and rec["upstream_errors"] == {}


def test_the_close_age_histogram_and_the_default_suspect_windows():
    relay = C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1")
    for age in (5, 299, 600.4, 1795, 2500, 4000, None):
        relay.count_client_closed(age)
    ages = relay.record()["client_close_ages"]
    assert ages["histogram_s"] == {"<60": 1, "60-300": 1, "300-590": 0, "590-610": 1,
                                   "610-1790": 0, "1790-1810": 1, "1810-3600": 1, ">=3600": 1}
    assert ages["closes_with_age"] == 6 and ages["max_s"] == 4000
    assert ages["suspect_timeouts_s"] == [600.0, 1800.0, 3660.0]
    assert ages["client_timeouts_suspected"] == 2
    assert ages["client_timeout_suspect_ages"] == [600.4, 1795]
    relay.count_client_closed(3655.0)        # multi_swe_bench's TIMEOUT_S + 60 client timeout
    assert relay.record()["client_close_ages"]["client_timeouts_suspected"] == 3
    assert relay.record()["client_closed_streams"] == 8


def test_only_gbench_smoke_relay_replies_makes_a_relay_a_smoke_relay(monkeypatch):
    """2026-10-11 review: GBENCH_SMOKE=1 is in every suite of a smoke run, and reading it turned
    non-agent relays (bfcl's agentic part, aider_polyglot, livebench) into smoke relays, against
    docs/smoke.md (in-process loops run their sample to its own end). --smoke now sets
    GBENCH_SMOKE_RELAY_REPLIES for its agent suites; GBENCH_SMOKE alone changes no relay."""
    monkeypatch.setenv("GBENCH_SMOKE", "1")
    monkeypatch.delenv("GBENCH_SMOKE_RELAY_REPLIES", raising=False)
    assert "smoke" not in C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1").record()
    monkeypatch.setenv("GBENCH_SMOKE_RELAY_REPLIES", "2")
    rec = C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1").record()
    assert rec["smoke"]["replies_per_conversation"] == 2
    assert rec["smoke"]["source"] == "GBENCH_SMOKE_RELAY_REPLIES"


def test_a_malformed_relay_env_knob_names_itself_and_is_not_a_value_error(monkeypatch):
    """2026-10-11 review: tau_common turns a ValueError of the relay constructor into "the model
    endpoint URL cannot be relayed" and skillsbench into "it could not listen on", so a typo in
    a relay variable must raise something else, naming the variable."""
    for var, value in (("GBENCH_RELAY_IDLE_TIMEOUT_S", "5m"), ("GBENCH_RELAY_IDLE_TIMEOUT_S", "0"),
                       ("GBENCH_RELAY_EXPECTED_SHAPE", "{temperature: 1}"),
                       ("GBENCH_RELAY_EXPECTED_SHAPE", "[1]"),
                       ("GBENCH_RELAY_EXPECTED_SHAPE", '{"temprature": 1.0}'),
                       ("GBENCH_SMOKE_RELAY_REPLIES", "one"), ("GBENCH_SMOKE_RELAY_REPLIES", "0")):
        monkeypatch.setenv(var, value)
        with pytest.raises(C.RelayEnvError, match=var) as err:
            C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1")
        assert not isinstance(err.value, ValueError)
        monkeypatch.delenv(var)
    # a caller's own bad argument stays a ValueError
    with pytest.raises(ValueError):
        C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1", expected_shape={"temprature": 1})


def test_a_smoke_run_makes_every_agent_relay_a_smoke_relay_unless_its_caller_opts_out(
        monkeypatch):
    monkeypatch.setenv("GBENCH_SMOKE_RELAY_REPLIES", "1")
    up = _Upstream(_whole_after(0, "ls"))
    relay = _relay(up)
    try:
        assert _post(relay, _turn())[0] == 200
        assert _post(relay, _turn("ls"))[0] == 400
        rec = relay.record()
    finally:
        relay.stop()
        up.close()
    assert (rec["smoke"]["source"] == "GBENCH_SMOKE_RELAY_REPLIES"
            and rec["smoke"]["replies_per_conversation"] == 1)
    assert rec["smoke_second_requests"] == 1
    off = C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1", smoke_from_env=False)
    assert "smoke" not in off.record()
    # base's single-turn re-ask relay is never a smoke relay
    with C.upstream_reask_relay("gpqa", "http://127.0.0.1:1/v1", C.reply_content_is_null, 2) as r:
        assert "smoke" not in r.record()
    monkeypatch.delenv("GBENCH_SMOKE_RELAY_REPLIES")
    assert "smoke" not in C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1").record()


def test_the_expected_shape_can_come_from_the_environment_with_an_upper_budget(monkeypatch):
    monkeypatch.setenv("GBENCH_RELAY_EXPECTED_SHAPE", json.dumps(
        {"temperature": 1.0, "enable_thinking": True, "max_tokens_at_most": 65536}))
    relay = C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1")
    path = "/v1/chat/completions"
    think = {"chat_template_kwargs": {"enable_thinking": True}, "temperature": 1.0}

    def check(body):
        return relay.check_shape(path, json.dumps(body).encode())
    assert check({**MSG, **think}) == []                               # no budget: the window
    assert check({**MSG, **think, "max_tokens": 8192}) == []            # a smaller budget
    assert check({**MSG, **think, "max_tokens": 70000}) == ["max_tokens_at_most"]
    shape = relay.record()["shape_check"]
    assert shape["source"] == "GBENCH_RELAY_EXPECTED_SHAPE"
    assert shape["violations"] == {"max_tokens_at_most": 1}
    monkeypatch.setenv("GBENCH_RELAY_EXPECTED_SHAPE", "[1]")
    with pytest.raises(C.RelayEnvError, match="GBENCH_RELAY_EXPECTED_SHAPE"):
        C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1")


def test_the_thinking_switch_is_checked_on_chat_requests_only():
    relay = C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1",
                            expected_shape={"enable_thinking": True})
    body = json.dumps({"model": "m", "prompt": "<rendered>"}).encode()
    assert relay.check_shape("/v1/completions", body) == []
    assert relay.check_shape("/v1/chat/completions", json.dumps(MSG).encode()) == [
        "enable_thinking"]
