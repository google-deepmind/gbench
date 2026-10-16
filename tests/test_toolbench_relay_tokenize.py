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
"""toolbench's endpoint relay forwards the server's root /tokenize.

2026-10-10 smoke test: toolbench's model relay (swebench_common.EndpointRelay) refused 27 POST
requests, one per model reply in model_requests.jsonl ("[toolbench-model-relay] refused a POST
request (only the allowed /v1/ endpoint paths are relayed)", requests_refused: 27). The image's
request policy (docker/toolbench_request_policy.py) counts each request's prompt with
`<server root>/tokenize` (vLLM serves it outside /v1) to fit max_tokens to the room the context
window leaves; the relay forwarded only /v1 routes, so every count was refused and the policy
fell back to not knowing the room. toolbench's relay now names /tokenize as a root path
(EndpointRelay root_paths, TOKENIZE_ROOT_PATHS); the default relay of every other suite still
refuses it, and a /tokenize request is never counted as a model reply or a model-request failure.
No model or network: a local fake endpoint (the live check against a served vLLM is manual).
"""

import http.server
import importlib.util
import inspect
import json
import os
import socketserver
import threading
import urllib.error
import urllib.request

import pytest

from gbench.runners.eval_suites import swebench_common as C
from gbench.runners.eval_suites import toolbench as T

_POLICY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docker",
                            "toolbench_request_policy.py")
KEY = "sk-endpoint-secret"


def _policy_module():
    spec = importlib.util.spec_from_file_location("_toolbench_request_policy_under_test",
                                                  _POLICY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _upstream(seen, tokenize_status=200):
    """A fake vLLM: /tokenize at the root, chat completions under /v1, the model list."""
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status, obj):
            out = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_GET(self):
            seen.append(("GET", self.path, self.headers.get("Authorization"), None))
            self._send(200, {"data": [{"id": "m", "owned_by": "vllm"}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(("POST", self.path, self.headers.get("Authorization"), req))
            if self.path == "/tokenize":
                if tokenize_status != 200:
                    # the shape of a vLLM 400, naming the context window
                    self._send(tokenize_status, {"error": {
                        "message": "This model's maximum context length is 8192 tokens."}})
                    return
                self._send(200, {"count": 1000, "max_model_len": 8192, "tokens": [1] * 1000})
                return
            ch = {"index": 0, "finish_reason": "stop", "token_ids": None,
                  "message": {"role": "assistant", "content": "ok"}}
            self._send(200, {"choices": [ch], "prompt_token_ids": None})

        def log_message(self, *a):
            pass
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _relay(srv, **kw):
    return C.EndpointRelay(f"http://127.0.0.1:{srv.server_address[1]}/v1", "127.0.0.1",
                           key=KEY, **kw).start()


def _post(url, body, auth="Bearer EMPTY"):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": auth})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read())


TOKENIZE_BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}],
                 "add_generation_prompt": True}
CHAT_BODY = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def test_toolbench_names_tokenize_as_its_relays_root_path():
    assert T.TOKENIZE_ROOT_PATHS == ("/tokenize",)
    src = inspect.getsource(T.run_toolbench)
    assert "root_paths=TOKENIZE_ROOT_PATHS" in src
    assert 'name="toolbench-model-relay"' in src


def test_the_toolbench_relay_forwards_tokenize_to_the_server_root_with_the_key():
    seen = []
    srv = _upstream(seen)
    relay = _relay(srv, root_paths=T.TOKENIZE_ROOT_PATHS)
    try:
        root = relay.url[:-len("/v1")]
        status, reply = _post(root + "/tokenize", TOKENIZE_BODY)
        assert status == 200 and reply["count"] == 1000 and reply["max_model_len"] == 8192
        method, path, auth, body = seen[-1]
        assert (method, path) == ("POST", "/tokenize")
        # the client's placeholder is dropped and the endpoint's key sent; the body is unchanged
        assert auth == f"Bearer {KEY}"
        assert body == TOKENIZE_BODY
        assert KEY not in json.dumps(reply)
    finally:
        relay.stop()
        srv.shutdown()


def test_tokenize_is_not_a_model_reply_and_model_requests_count_as_before():
    seen = []
    srv = _upstream(seen)
    relay = _relay(srv, root_paths=T.TOKENIZE_ROOT_PATHS)
    try:
        root = relay.url[:-len("/v1")]
        for _ in range(3):
            _post(root + "/tokenize", TOKENIZE_BODY)
            _post(relay.url + "/chat/completions", CHAT_BODY)
        rec = relay.record()
        assert rec["requests_relayed"] == 6
        assert rec["root_requests_relayed"] == 3
        assert rec["completion_replies"] == 3
        assert rec["requests_refused"] == 0
        assert rec["upstream_errors"] == {} and rec["root_path_errors"] == {}
        assert rec["root_paths"] == ["/tokenize"]
        # only completions requests get return_token_ids; /tokenize bodies are never touched
        tok = [b for m, p, _, b in seen if p == "/tokenize"]
        assert tok and all("return_token_ids" not in b for b in tok)
        chats = [b for m, p, _, b in seen if p == "/v1/chat/completions"]
        assert chats and all(b.get("return_token_ids") for b in chats)
    finally:
        relay.stop()
        srv.shutdown()


def test_a_failed_tokenize_is_counted_apart_from_model_request_failures():
    seen = []
    srv = _upstream(seen, tokenize_status=400)
    relay = _relay(srv, root_paths=T.TOKENIZE_ROOT_PATHS)
    try:
        root = relay.url[:-len("/v1")]
        with pytest.raises(urllib.error.HTTPError) as err:
            _post(root + "/tokenize", TOKENIZE_BODY)
        assert err.value.code == 400                # the reply goes back unchanged
        rec = relay.record()
        assert rec["root_path_errors"] == {"http_400": 1}
        # not a model request: no endpoint failure, no context rejection, no reply counted
        assert rec["upstream_errors"] == {}
        assert rec["context_length_rejections"] == 0
        assert rec["endpoint_side_failures"] == 0
        assert rec["completion_replies"] == 0
    finally:
        relay.stop()
        srv.shutdown()


def test_the_default_relay_still_refuses_tokenize():
    seen = []
    srv = _upstream(seen)
    relay = _relay(srv)
    try:
        root = relay.url[:-len("/v1")]
        with pytest.raises(urllib.error.HTTPError) as err:
            _post(root + "/tokenize", TOKENIZE_BODY)
        assert err.value.code == 400
        assert not [s for s in seen if s[1] == "/tokenize"]
        rec = relay.record()
        assert rec["requests_refused"] == 1 and rec["requests_relayed"] == 0
        assert "root_paths" not in rec          # the other suites' records are unchanged
    finally:
        relay.stop()
        srv.shutdown()


def test_root_paths_keep_every_path_check():
    relay = C.EndpointRelay("http://upstream:8000/v1", "127.0.0.1",
                            root_paths=T.TOKENIZE_ROOT_PATHS)
    assert relay.target("/tokenize") == "http://upstream:8000/tokenize"
    assert relay.target("/tokenize?a=1") == "http://upstream:8000/tokenize?a=1"
    for bad in ("/tokenize#x", "/tokenize/", "/detokenize", "/tokenize/../v1/load_lora_adapter",
                "/%74okenize", "/tokenize?a=<b>", "/v1/tokenize", "//tokenize", "/health"):
        assert relay.target(bad) is None, bad
    # the /v1 routes are as before
    assert relay.target("/v1/chat/completions") == "http://upstream:8000/v1/chat/completions"
    assert relay.target("/v1/load_lora_adapter") is None
    # a base path before /v1 is kept for the root path too
    sub = C.EndpointRelay("http://upstream:8000/serve/v1", "127.0.0.1", root_paths=("/tokenize",))
    assert sub.target("/tokenize") == "http://upstream:8000/serve/tokenize"
    # the default relay refuses it
    assert C.EndpointRelay("http://upstream:8000/v1", "127.0.0.1").target("/tokenize") is None


@pytest.mark.parametrize("bad", ["/v1/tokenize", "/v1", "tokenize", "/x/chat/completions",
                                 "/completions", "/", "//tokenize", "/tokenize/",
                                 "/a/../tokenize", "/%74okenize"])
def test_a_root_path_cannot_be_a_v1_or_completions_route(bad):
    """Also refused when made: a root path target() could never match ('/', an empty, '.' or
    '..' segment, a percent escape), found in the 2026-10-10 review."""
    with pytest.raises(ValueError):
        C.EndpointRelay("http://upstream:8000/v1", "127.0.0.1", root_paths=(bad,))


def test_root_paths_given_as_one_string_are_refused_by_name():
    with pytest.raises(ValueError, match="sequence of paths"):
        C.EndpointRelay("http://upstream:8000/v1", "127.0.0.1", root_paths="/tokenize")


def test_a_root_path_is_relayed_for_post_only():
    """vLLM serves /tokenize for POST only (a GET is its 405); the relay refuses a GET to a root
    path itself, so it never reaches the endpoint or root_path_errors."""
    seen = []
    srv = _upstream(seen)
    relay = _relay(srv, root_paths=T.TOKENIZE_ROOT_PATHS)
    try:
        root = relay.url[:-len("/v1")]
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(root + "/tokenize", timeout=30)
        assert err.value.code == 405
        assert not seen
        rec = relay.record()
        assert rec["requests_refused"] == 1 and rec["requests_relayed"] == 0
        assert rec["root_path_errors"] == {}
        # GET on the /v1 routes is as before
        with urllib.request.urlopen(relay.url + "/models", timeout=30) as resp:
            assert resp.status == 200
    finally:
        relay.stop()
        srv.shutdown()


def test_the_refusal_reason_names_the_root_paths():
    plain = C.EndpointRelay("http://upstream:8000/v1", "127.0.0.1")
    assert plain.refusal_reason() == "only the allowed /v1/ endpoint paths are relayed"
    tb = C.EndpointRelay("http://upstream:8000/v1", "127.0.0.1",
                         root_paths=T.TOKENIZE_ROOT_PATHS)
    assert tb.refusal_reason() == ("only the allowed /v1/ endpoint paths and the root paths "
                                   "/tokenize are relayed")


def test_the_relay_listens_with_a_backlog_of_128():
    """2026-10-10 review: with socketserver's default backlog of 5, 32 client threads through
    a relay got 3-5 connection resets per 400 requests, which no relay counter shows; toolbench
    now opens two connections per model request (/tokenize, then the completion)."""
    seen = []
    srv = _upstream(seen)
    relay = _relay(srv, root_paths=T.TOKENIZE_ROOT_PATHS)
    try:
        assert relay._server.request_queue_size == 128
        assert relay._server.daemon_threads is True
    finally:
        relay.stop()
        srv.shutdown()


def test_the_image_policy_counts_its_prompt_through_the_relay():
    """End to end: the container's RequestPolicy, given the relay's /v1 URL as its base (what
    TB_ENDPOINT carries), derives <relay root>/tokenize and gets the room from the endpoint."""
    seen = []
    srv = _upstream(seen)
    relay = _relay(srv, root_paths=T.TOKENIZE_ROOT_PATHS)
    try:
        pol = _policy_module().RequestPolicy("TB_", base_url=relay.url, env={})
        assert pol.tokenize_url == relay.url[:-len("/v1")] + "/tokenize"
        tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
        room = pol.room({"model": "m", "messages": CHAT_BODY["messages"], "tools": tools,
                         "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
                        api_key="EMPTY")
        assert room == 8192 - 1000
        method, path, auth, body = seen[-1]
        assert (method, path, auth) == ("POST", "/tokenize", f"Bearer {KEY}")
        assert body["tools"] == tools and body["chat_template_kwargs"] == {"enable_thinking": True}
        rec = relay.record()
        assert rec["requests_refused"] == 0 and rec["root_requests_relayed"] == 1
    finally:
        relay.stop()
        srv.shutdown()


def test_the_image_policy_through_a_default_relay_gets_no_room():
    """What the smoke test saw: the default relay refuses, the policy does not know the room."""
    seen = []
    srv = _upstream(seen)
    relay = _relay(srv)
    try:
        pol = _policy_module().RequestPolicy("TB_", base_url=relay.url, env={})
        assert pol.room({"model": "m", "messages": CHAT_BODY["messages"]}) is None
        assert relay.record()["requests_refused"] == 1
    finally:
        relay.stop()
        srv.shutdown()
