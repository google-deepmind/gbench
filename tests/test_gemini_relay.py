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

"""The host-side Gemini key relay (gemini_relay.GeminiRelay).

The upstream is a stub HTTP server on 127.0.0.1 (GEMINI_OPENAI_BASE_URL points at it, which
base.check_gemini_openai_base_url accepts for a loopback address); nothing reaches the Gemini API.

* A client holding the per-run placeholder gets its OpenAI-compatible and native calls forwarded
  with the real key; the key never appears in a reply, a log line, an error or the record.
* Wrong or missing placeholders are answered 401, other paths and methods 403, oversized bodies
  413, and none of them is forwarded.
* Keys rotate, and 401/403/429 moves to the next key; service failures (no answer, 5xx, 429 on
  every key, a redirect) are harness errors with their time and client; content rejections are
  harness errors unless the suite says they are the model's.
* Redirects are not followed.
* search_tool sends its grounding requests to the relay when GBENCH_GEMINI_RELAY_URL is set and
  its key is the placeholder, and refuses to send a real key there.
"""

import json
import logging
import socket
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gbench.runners.eval_suites import gemini_relay as GR

# Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
KEY_A = "AIza" + "RelayTestKeyAAAAAAAAAAAAAAAAAAAAAAA1"
KEY_B = "AIza" + "RelayTestKeyBBBBBBBBBBBBBBBBBBBBBBB2"


class Upstream:
    """A stub Gemini API: records each request; `answer(method, path, headers, body)` returns
    (status, content_type, body bytes, extra headers)."""

    def __init__(self):
        self.requests = []
        self.answer = lambda m, p, h, b: (200, "application/json", b'{"ok": true}', {})
        up = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                return

            def _any(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                hdrs = {k.lower(): v for k, v in self.headers.items()}
                up.requests.append({"method": self.command, "path": self.path,
                                    "headers": hdrs, "body": body})
                status, ctype, data, extra = up.answer(self.command, self.path, hdrs, body)
                self.send_response(status)
                if ctype:
                    self.send_header("Content-Type", ctype)
                for k, v in extra.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = do_PUT = do_DELETE = _any

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def upstream(monkeypatch):
    up = Upstream()
    monkeypatch.setenv("GEMINI_API_KEYS", f"{KEY_A},{KEY_B}")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", f"http://127.0.0.1:{up.port}/v1beta/openai/")
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy",
                 "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    yield up
    up.close()


@pytest.fixture
def relay(upstream):
    made = []

    def _make(**kw):
        r = GR.GeminiRelay("test_suite", "127.0.0.1", **kw).start()
        made.append(r)
        return r
    yield _make
    for r in made:
        r.stop()


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def call(url, *, method="POST", body=b"{}", headers=None):
    req = urllib.request.Request(url, data=body if method != "GET" else None, method=method,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with _OPENER.open(req, timeout=30) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def _no_key(*texts):
    for t in texts:
        t = t if isinstance(t, str) else json.dumps(t, default=str)
        assert KEY_A not in t and KEY_B not in t


def test_openai_calls_go_upstream_with_the_real_key(relay, upstream):
    r = relay()
    status, data, _ = call(r.openai_base_url + "/chat/completions",
                           body=b'{"model": "gemini-3.7-flash", "messages": []}',
                           headers=bearer(r.placeholder))
    assert status == 200 and data == b'{"ok": true}'
    [req] = upstream.requests
    assert req["method"] == "POST" and req["path"] == "/v1beta/openai/chat/completions"
    assert req["headers"]["authorization"] == f"Bearer {KEY_A}"
    assert r.placeholder not in json.dumps(req["headers"])
    assert req["body"] == b'{"model": "gemini-3.7-flash", "messages": []}'
    assert call(r.openai_base_url + "/models", method="GET",
                headers=bearer(r.placeholder))[0] == 200
    assert upstream.requests[1]["path"] == "/v1beta/openai/models"
    rec = r.record()
    assert rec["requests_relayed"] == 2 and rec["harness_errors"] == {}
    _no_key(rec)


def test_native_calls_carry_the_key_in_x_goog_api_key_and_only_alt(relay, upstream):
    r = relay()
    url = r.origin + "/v1beta/models/gemini-3.7-flash:generateContent?alt=sse&key=" + \
        r.placeholder + "&x=1"
    status, _, _ = call(url, headers={"x-goog-api-key": r.placeholder,
                                      "x-goog-user-project": "someone-elses",
                                      "Cookie": "a=b"})
    assert status == 200
    [req] = upstream.requests
    assert req["path"] == "/v1beta/models/gemini-3.7-flash:generateContent?alt=sse"
    assert req["headers"]["x-goog-api-key"] == KEY_A
    for dropped in ("authorization", "x-goog-user-project", "cookie"):
        assert dropped not in req["headers"]
    # the placeholder as ?key= opens the relay too, and is never passed on
    assert call(r.origin + "/v1beta/models/gemini-embedding-001:embedContent?key="
                + r.placeholder)[0] == 200
    assert upstream.requests[1]["path"] == "/v1beta/models/gemini-embedding-001:embedContent"


def test_wrong_or_missing_placeholder_is_401_and_never_forwarded(relay, upstream):
    r = relay()
    other = GR.PLACEHOLDER_PREFIX + "0" * 48
    for headers in ({}, bearer(other), bearer(KEY_A), {"x-goog-api-key": other}):
        assert call(r.openai_base_url + "/chat/completions", headers=headers)[0] == 401
    assert upstream.requests == []
    rec = r.record()
    assert rec["requests_rejected"] == 4 and rec["harness_errors"] == {}


@pytest.mark.parametrize("method, path", [
    ("GET", "/v1beta/files"),
    ("POST", "/upload/v1beta/files"),
    ("POST", "/v1beta/openai/images/generations"),
    ("DELETE", "/v1beta/openai/models"),
    ("GET", "/v1beta/openai/chat/completions"),
    ("POST", "/v1beta/models/gemini-3.7-flash:countTokens"),
    ("GET", "/v1beta/models/gemini-3.7-flash:generateContent"),
    ("POST", "/v1beta/models/../tunedModels/x:generateContent"),
    ("POST", "/v1beta/openai/../models/x:generateContent"),
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1beta/cachedContents"),
])
def test_other_paths_and_methods_are_refused(relay, upstream, method, path):
    r = relay()
    assert call(r.origin + path, method=method, headers=bearer(r.placeholder))[0] == 403
    assert upstream.requests == []
    rec = r.record()
    assert sum(x["requests"] for x in rec["requests_refused"]) == 1
    assert rec["harness_errors"] == {}


def test_native_models_can_be_restricted(relay, upstream):
    r = relay(native_models=["gemini-3.7-flash"], native_methods=["generateContent"])
    hdr = {"x-goog-api-key": r.placeholder}
    assert call(r.origin + "/v1beta/models/gemini-3.7-flash:generateContent", headers=hdr)[0] \
        == 200
    assert call(r.origin + "/v1beta/models/gemini-2.5-pro:generateContent", headers=hdr)[0] \
        == 403
    assert call(r.origin + "/v1beta/models/gemini-3.7-flash:embedContent", headers=hdr)[0] \
        == 403
    assert len(upstream.requests) == 1


def test_keys_rotate_and_a_rate_limit_moves_to_the_next_key(relay, upstream):
    r = relay()
    seen = []

    def answer(m, p, h, b):
        seen.append(h.get("authorization"))
        if h.get("authorization") == f"Bearer {KEY_A}":
            return 429, "application/json", b'{"error": "quota"}', {}
        return 200, "application/json", b"{}", {}
    upstream.answer = answer
    assert call(r.openai_base_url + "/chat/completions", headers=bearer(r.placeholder))[0] \
        == 200
    assert seen == [f"Bearer {KEY_A}", f"Bearer {KEY_B}"]
    assert r.record()["harness_errors"] == {}
    # the next request starts with the next key
    assert call(r.openai_base_url + "/chat/completions", headers=bearer(r.placeholder))[0] \
        == 200
    assert seen[2] == f"Bearer {KEY_B}"


def test_a_rate_limit_on_every_key_is_a_harness_error(relay, upstream):
    r = relay()
    upstream.answer = lambda m, p, h, b: (429, "application/json", b'{"error": "quota"}', {})
    before = time.time()
    status, data, _ = call(r.openai_base_url + "/chat/completions",
                           headers=bearer(r.placeholder))
    assert status == 429 and b"quota" in data
    assert len(upstream.requests) == 2
    rec = r.record()
    assert rec["harness_errors"] == {"gemini_http_429": 1}
    [ev] = rec["harness_error_events"]
    assert ev["kind"] == "gemini_http_429" and ev["client"] == "127.0.0.1"
    assert before - 1 <= ev["t"] <= time.time()


def test_a_server_error_is_a_harness_error_and_its_body_has_no_key(relay, upstream):
    r = relay()
    upstream.answer = lambda m, p, h, b: (
        500, "application/json", f'{{"error": "bad key {KEY_A} and {KEY_B}"}}'.encode(), {})
    status, data, _ = call(r.origin + "/v1beta/models/gemini-3.7-flash:generateContent",
                           headers={"x-goog-api-key": r.placeholder})
    assert status == 500
    _no_key(data.decode())
    assert b"[REDACTED]" in data
    assert r.record()["harness_errors"] == {"gemini_http_500": 1}


def test_an_upstream_that_does_not_answer_is_a_harness_error(relay, upstream, monkeypatch,
                                                            caplog):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    closed = s.getsockname()[1]
    s.close()
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", f"http://127.0.0.1:{closed}/v1beta/openai")
    caplog.set_level(logging.DEBUG)
    r = relay()
    status, data, _ = call(r.openai_base_url + "/chat/completions",
                           headers=bearer(r.placeholder))
    assert status == 502 and b"did not answer" in data
    assert r.record()["harness_errors"] == {"gemini_unreachable": 1}
    _no_key(data.decode(), caplog.text, r.record())


def test_a_redirect_is_not_followed_and_loses_its_location(relay, upstream):
    r = relay()
    upstream.answer = lambda m, p, h, b: (
        302, "text/plain", b"moved", {"Location": f"http://127.0.0.1:{upstream.port}/stolen"})
    status, data, headers = call(r.openai_base_url + "/chat/completions",
                                 headers=bearer(r.placeholder))
    assert status == 302 and "Location" not in headers and "location" not in headers
    assert [q["path"] for q in upstream.requests] == ["/v1beta/openai/chat/completions"]
    assert r.record()["harness_errors"] == {"gemini_http_302": 1}


def test_content_rejections_are_harness_errors_unless_the_suite_says_otherwise(relay, upstream):
    upstream.answer = lambda m, p, h, b: (400, "application/json", b'{"error": "bad"}', {})
    judge = relay()
    assert call(judge.openai_base_url + "/chat/completions",
                headers=bearer(judge.placeholder))[0] == 400
    assert judge.record()["harness_errors"] == {"gemini_http_400": 1}
    tool = relay(rejections_are_harness_errors=False)
    assert call(tool.openai_base_url + "/chat/completions",
                headers=bearer(tool.placeholder))[0] == 400
    rec = tool.record()
    assert rec["harness_errors"] == {} and rec["rejections"] == {"gemini_http_400": 1}
    assert rec["rejection_events"][0]["kind"] == "gemini_http_400"
    # a 400 is the request's content: the next key is not tried
    assert len(upstream.requests) == 2


def test_an_oversized_body_is_413_unread(relay, upstream, monkeypatch):
    monkeypatch.setattr(GR, "MAX_REQUEST_BODY", 1024)
    r = relay()
    status, data, _ = call(r.openai_base_url + "/chat/completions", body=b"x" * 4096,
                           headers=bearer(r.placeholder))
    assert status == 413 and upstream.requests == []
    assert r.record()["harness_errors"] == {"request_too_large": 1}


def test_a_streamed_reply_passes_through(relay, upstream):
    r = relay()
    sse = b'data: {"candidates": [{"content": {"parts": [{"text": "hi"}]}}]}\r\n\r\n'
    upstream.answer = lambda m, p, h, b: (200, "text/event-stream", sse, {})
    status, data, headers = call(
        r.origin + "/v1beta/models/gemini-3.7-flash:streamGenerateContent?alt=sse",
        headers={"x-goog-api-key": r.placeholder})
    assert status == 200 and data == sse
    assert headers.get("Content-Type") == "text/event-stream"


def test_the_record_and_the_logs_never_hold_a_key(relay, upstream, caplog):
    caplog.set_level(logging.DEBUG)
    r = relay()
    upstream.answer = lambda m, p, h, b: (503, "application/json", KEY_B.encode(), {})
    call(r.openai_base_url + "/chat/completions", headers=bearer(r.placeholder))
    call(r.openai_base_url + "/chat/completions", headers=bearer("wrong"))
    call(r.origin + "/v1beta/files", method="GET", headers=bearer(r.placeholder))
    rec = r.record()
    _no_key(rec, caplog.text)
    assert "2 key(s)" in rec["adds"]


def test_client_env_holds_only_the_placeholder_and_the_relay_addresses(relay):
    r = relay()
    env = r.client_env()
    assert env["GEMINI_API_KEY"] == env["GEMINI_API_KEYS"] == r.placeholder
    assert GR.is_placeholder(r.placeholder) and not GR.is_placeholder(KEY_A)
    assert env["GEMINI_OPENAI_BASE_URL"] == r.openai_base_url
    assert env[GR.RELAY_URL_ENV] == r.origin and r.origin.startswith("http://127.0.0.1:")
    _no_key(env)
    # a new relay gets a new placeholder
    assert relay().placeholder != r.placeholder


def test_a_setup_it_cannot_serve_is_infra_required(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED\] test_suite.*GEMINI_API_KEY"):
        GR.GeminiRelay("test_suite", "127.0.0.1")
    monkeypatch.setenv("GEMINI_API_KEY", KEY_A)
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", "https://evil.example.com/v1beta/openai")
    with pytest.raises(RuntimeError) as ei:
        GR.GeminiRelay("test_suite", "127.0.0.1")
    assert "GEMINI_OPENAI_BASE_URL is not the Gemini API" in str(ei.value)
    assert "evil.example.com" not in str(ei.value)


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "", "example.com"])
def test_it_never_listens_on_every_interface_or_a_name(monkeypatch, host):
    monkeypatch.setenv("GEMINI_API_KEY", KEY_A)
    monkeypatch.delenv("GEMINI_OPENAI_BASE_URL", raising=False)
    with pytest.raises(ValueError):
        GR.GeminiRelay("test_suite", host)


def test_the_default_upstream_is_the_gemini_api(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY_A)
    monkeypatch.delenv("GEMINI_OPENAI_BASE_URL", raising=False)
    rec = GR.GeminiRelay("test_suite", "127.0.0.1").record()
    assert rec["relays_to"] == {
        "openai_compatible": "https://generativelanguage.googleapis.com/v1beta/openai",
        "native": "https://generativelanguage.googleapis.com/v1beta/models"}


def test_an_address_it_cannot_bind_is_infra_required(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY_A)
    monkeypatch.delenv("GEMINI_OPENAI_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="could not listen on 192.0.2.1"):
        GR.GeminiRelay("test_suite", "192.0.2.1").start()


def test_network_gateway_reads_docker_network_inspect(monkeypatch):
    calls = []

    def fake(argv, **kw):
        calls.append((argv, kw))
        import subprocess
        out = {"bridge": "172.17.0.1 ", "v6net": "fd00::1 172.30.0.1 ", "none": " "}[argv[-1]]
        return subprocess.CompletedProcess(argv, 0, out, "")
    monkeypatch.setattr(GR.subprocess, "run", fake)
    assert GR.network_gateway("x", "bridge") == "172.17.0.1"
    assert GR.network_gateway("x", "v6net") == "172.30.0.1"
    with pytest.raises(RuntimeError, match="has no IPv4 gateway"):
        GR.network_gateway("x", "none")
    assert all(kw.get("timeout") for _, kw in calls)
    assert calls[0][0][:3] == ["docker", "network", "inspect"]


# --------------------------------------------------------------------------- #
# search_tool through the relay
# --------------------------------------------------------------------------- #
def _grounded(query):
    return json.dumps({"candidates": [{"content": {"parts": [{"text": f"about {query}"}]},
                                       "groundingMetadata": {"groundingChunks": [
                                           {"web": {"uri": "https://a.example", "title": "A"}}]}}]
                       }).encode()


def test_search_tool_goes_through_the_relay_with_the_placeholder(relay, upstream, monkeypatch):
    from gbench.runners.eval_suites import search_tool
    r = relay()
    upstream.answer = lambda m, p, h, b: (200, "application/json", _grounded("x"), {})
    for name, value in r.client_env().items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(search_tool, "_KEY_CYCLE", None)
    results = search_tool.gemini_search("who wrote hamlet", model="gemini-3.7-flash")
    assert results[0]["url"] == "https://a.example"
    [req] = upstream.requests
    assert req["path"] == "/v1beta/models/gemini-3.7-flash:generateContent"
    assert req["headers"]["x-goog-api-key"] in (KEY_A, KEY_B)
    assert search_tool.RELAY_PLACEHOLDER_PREFIX == GR.PLACEHOLDER_PREFIX
    assert search_tool.RELAY_URL_ENV == GR.RELAY_URL_ENV


def test_search_tool_never_sends_a_real_key_to_the_relay_address(upstream, monkeypatch):
    from gbench.runners.eval_suites import search_tool
    monkeypatch.setenv(GR.RELAY_URL_ENV, f"http://127.0.0.1:{upstream.port}")
    monkeypatch.setenv("GEMINI_API_KEYS", KEY_A)
    results = search_tool.gemini_search("q", model="gemini-3.7-flash")
    assert results[0]["title"] == "error" and GR.RELAY_URL_ENV in results[0]["snippet"]
    assert upstream.requests == []
    _no_key(results)


def test_search_tool_without_the_relay_variable_is_unchanged(monkeypatch):
    from gbench.runners.eval_suites import search_tool
    monkeypatch.delenv(GR.RELAY_URL_ENV, raising=False)
    assert search_tool._generate_url("gemini-3.7-flash", KEY_A) == (
        search_tool.GEMINI_GENERATE_URL.format(model="gemini-3.7-flash"), "")


def test_the_module_imports_only_the_standard_library_at_import_time():
    import ast
    tree = ast.parse(open(GR.__file__, encoding="utf-8").read())
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert all(getattr(n, "level", 0) == 0 for n in top)
    names = {(a.name if isinstance(n, ast.Import) else n.module or "").split(".")[0]
             for n in top for a in n.names}
    assert "gbench" not in names
