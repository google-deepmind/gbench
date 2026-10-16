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

"""The relay keeps the evidence of replies that come back with reasoning and no content.

On 2026-10-08, 4 of 34 multi_swe_bench turns came back reasoning-only, and whether the model
ended its turn inside its reasoning or the server's reasoning parser put the answer there could
not be told afterwards. The relay now asks a non-streamed completions request for token ids,
keeps them (with the reasoning's tail) for such replies, and gives the client the reply it asked
for, token ids taken out. No model or network: a local fake endpoint.
"""

import http.server
import json
import socketserver
import threading
import urllib.request

from gbench.runners.eval_suites import swebench_common as C


def _upstream(seen, content, owner="vllm", finish="stop"):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            out = json.dumps({"data": [{"id": "m", "owned_by": owner}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                # the served tokenizer: `<|tool_call>` is one token (Gemma 4: 48)
                out = json.dumps({"tokens": [48]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
                return
            seen.append(req)
            ch = {"index": 0, "finish_reason": finish, "stop_reason": 106,
                  "message": {"role": "assistant", "content": content,
                              "reasoning_content": "thinking about it. ```\nls\n```"}}
            body = {"choices": [ch], "usage": {"completion_tokens": 3}}
            if owner == "vllm":
                # vLLM includes the keys either way (null when not asked for)
                ch["token_ids"] = [10, 11, 106] if req.get("return_token_ids") else None
                body["prompt_token_ids"] = [1, 2, 3] if req.get("return_token_ids") else None
            out = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _post(relay, body):
    req = urllib.request.Request(relay.url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def _relay(srv):
    return C.EndpointRelay(f"http://127.0.0.1:{srv.server_address[1]}/v1", "127.0.0.1").start()


MSG = {"model": "m", "messages": [{"role": "user", "content": "fix it"}]}


def test_a_reasoning_only_reply_keeps_its_token_ids_and_the_client_sees_none():
    seen = []
    srv = _upstream(seen, None)
    relay = _relay(srv)
    try:
        first = _post(relay, MSG)       # the model list told the relay the endpoint is vLLM
        reply = _post(relay, MSG)
    finally:
        relay.stop()
        srv.shutdown()
    # asked of the endpoint from the first request on (its GET /v1/models: owned_by vllm)
    assert seen[0]["return_token_ids"] is True and first["choices"][0]["token_ids"] is None
    assert seen[1]["return_token_ids"] is True
    # set back to null, as the server sends them when they are not asked for
    assert reply["choices"][0]["token_ids"] is None and reply["prompt_token_ids"] is None
    assert reply["choices"][0]["message"]["reasoning_content"].startswith("thinking")
    rec = relay.record()
    assert rec["replies_empty_after_reasoning"] == 2
    # both stopped on their own, their token ids without the tool-call token (48): answers
    # inside the thinking channel (ANSWER_IN_THINKING_POLICY)
    assert rec["replies_answer_in_thinking"] == 2 and rec["replies_tool_call_dropped"] == 0
    first_sample, sample = rec["empty_after_reasoning_samples"]
    assert first_sample["token_ids_tail"] == [10, 11, 106]
    assert first_sample["classified_as"] == "answer_in_thinking"
    assert sample["token_ids_tail"] == [10, 11, 106] and sample["token_ids_total"] == 3
    assert sample["stop_reason"] == 106
    assert sample["content"] is None and sample["reasoning_tail"].endswith("```")


def test_a_reply_with_content_keeps_no_sample_and_a_client_asking_for_ids_gets_them():
    seen = []
    srv = _upstream(seen, "DISCUSSION\nok\n```\nls\n```")
    relay = _relay(srv)
    try:
        _post(relay, MSG)
        plain = _post(relay, MSG)
        asked = _post(relay, {**MSG, "return_token_ids": True})
        streamed_body, _ = relay.add_token_ids("/v1/chat/completions",
                                               json.dumps({**MSG, "stream": True}).encode())
    finally:
        relay.stop()
        srv.shutdown()
    assert plain["choices"][0].get("token_ids") is None
    assert asked["choices"][0]["token_ids"] == [10, 11, 106]   # the client's own request
    assert "return_token_ids" not in json.loads(streamed_body)  # streams are left as they are
    assert "empty_after_reasoning_samples" not in relay.record()


def test_no_token_ids_are_asked_of_a_server_that_is_not_vllm_or_for_an_echo():
    seen = []
    srv = _upstream(seen, None, owner="someone-else")
    relay = _relay(srv)
    try:
        _post(relay, MSG)
        _post(relay, MSG)
    finally:
        relay.stop()
        srv.shutdown()
    assert not any("return_token_ids" in r for r in seen)
    vllm = []
    srv = _upstream(vllm, None)
    relay = _relay(srv)
    try:
        _post(relay, MSG)
        body, asked = relay.add_token_ids("/v1/completions",
                                          json.dumps({"prompt": "x", "echo": True}).encode())
    finally:
        relay.stop()
        srv.shutdown()
    assert asked is False and "return_token_ids" not in json.loads(body)


def test_a_reply_cut_off_at_a_limit_is_counted_but_not_sampled():
    seen = []
    srv = _upstream(seen, None, finish="length")
    relay = _relay(srv)
    try:
        _post(relay, MSG)
    finally:
        relay.stop()
        srv.shutdown()
    rec = relay.record()
    assert rec["replies_cut_off"] == 1 and rec["replies_empty_after_reasoning"] == 1
    # cut off at the limit: a truncated turn, not an answer inside the thinking channel
    assert rec["replies_answer_in_thinking"] == 0
    assert "empty_after_reasoning_samples" not in rec
