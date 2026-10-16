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

"""Replies that come back with reasoning and no content, told apart by their token ids, and the
upstream re-ask of a null reply.

* A reply whose generation holds the served tokenizer's tool-call token (Gemma 4 `<|tool_call>`,
  id 48) is a native tool call the server's tool-call parser removed (the agent declares no
  tools): a server-parser artifact, counted apart (`replies_tool_call_dropped`) and a
  comparability reason, never the model's answer_in_thinking. The shape is the one replayed on
  2026-10-10 (ts rep 0): `...<channel|><|tool_call>call:search_dir:...<tool_call|><|tool_response>`.
* A reply whose token ids hold no tool-call token is the answer inside the thinking channel.
* A reply without token ids is unclassified (a reason), never answer_in_thinking.
* gpqa / causalbench (simple-evals sampler, Corr2Cause Chatbot) re-ask a null-content reply and
  longbench_v2_code (pred.py) a null or empty one, through a loopback relay, up to
  NULL_REPLY_REASKS times.
No model or network: local fake endpoints.
"""

import http.server
import json
import socketserver
import threading
import types
import urllib.request

import pytest

from gbench.runners.eval_suites import agent_dojo as AD
from gbench.runners.eval_suites import swe_agentic as A
from gbench.runners.eval_suites import swebench_common as C

TOOL_CALL = 48            # Gemma 4 `<|tool_call>` (tokenizer.json added_tokens)
CHANNEL_CLOSE, TOOL_CALL_END, TOOL_RESPONSE, TURN_END = 101, 49, 50, 106
#: ts rep 0 of the 2026-10-10 replay: the thought closed, then a native tool call, stop 50
DROPPED_CALL_IDS = [100, 5, 6, CHANNEL_CLOSE, TOOL_CALL, 7, 8, TOOL_CALL_END, TOOL_RESPONSE]
#: the answer inside the thinking channel: the thought never closed, stop <turn|>
IN_THINKING_IDS = [100, 5, 6, 7, TURN_END]


@pytest.fixture(autouse=True)
def _fresh_token_cache():
    C._SERVED_TOKEN_IDS.clear()
    yield
    C._SERVED_TOKEN_IDS.clear()


class _Upstream:
    """A vLLM-shaped endpoint: GET /v1/models (owned_by vllm), POST /tokenize (`<|tool_call>` is
    one token, 48), POST /v1/chat/completions answering from `script(request) -> (message,
    finish, token_ids)`."""

    def __init__(self, script, owner="vllm", tokenize=True):
        self.seen = []
        self.tokenize_calls = 0
        up = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, code, obj):
                out = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def do_GET(self):
                self._send(200, {"data": [{"id": "m", "owned_by": owner}]})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    up.tokenize_calls += 1
                    if not tokenize:
                        self._send(404, {"detail": "Not Found"})
                        return
                    toks = [TOOL_CALL] if req.get("prompt") == "<|tool_call>" else [1, 2]
                    self._send(200, {"count": len(toks), "tokens": toks})
                    return
                up.seen.append(req)
                msg, finish, ids = script(req)
                ch = {"index": 0, "finish_reason": finish, "stop_reason": None,
                      "message": dict({"role": "assistant"}, **msg),
                      "token_ids": ids if req.get("return_token_ids") else None}
                self._send(200, {"choices": [ch], "usage": {"completion_tokens": 9},
                                 "prompt_token_ids": [1] if req.get("return_token_ids")
                                 else None})

            def log_message(self, *a):
                pass
        self.srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def _post(url, body):
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def _empty(ids):
    return {"content": None, "reasoning": "thinking about it"}, "stop", ids


# ------------------------------------------------------------------------------------------
# classification
# ------------------------------------------------------------------------------------------
def test_classify_empty_reply_from_token_ids():
    assert C.classify_empty_reply(DROPPED_CALL_IDS, TOOL_CALL) == C.EMPTY_TOOL_CALL_DROPPED
    assert C.classify_empty_reply(IN_THINKING_IDS, TOOL_CALL) == C.EMPTY_ANSWER_IN_THINKING
    # no token ids, or no tool-call token in the served tokenizer: cannot be told apart
    for ids, tc in ((None, TOOL_CALL), ([], TOOL_CALL), (IN_THINKING_IDS, None)):
        assert C.classify_empty_reply(ids, tc) == C.EMPTY_UNCLASSIFIED


def test_the_tool_call_token_is_read_from_the_served_tokenizer_once():
    up = _Upstream(lambda r: _empty(None))
    try:
        assert C.served_token_id(up.url, "<|tool_call>") == TOOL_CALL
        assert C.served_token_id(up.url.rsplit("/v1", 1)[0], "<|tool_call>") == TOOL_CALL
        assert up.tokenize_calls == 1                       # cached per server
        assert C.served_token_id(up.url, "two tokens") is None
    finally:
        up.close()
    no = _Upstream(lambda r: _empty(None), tokenize=False)
    try:
        assert C.served_token_id(no.url, "<|tool_call>") is None
    finally:
        no.close()


# ------------------------------------------------------------------------------------------
# the relay (multi_swe_bench, swe_bench_live, swe_agentic 2.x)
# ------------------------------------------------------------------------------------------
def _script_by_scenario(req):
    kind = req["messages"][0]["content"]
    if kind == "dropped":
        return _empty(DROPPED_CALL_IDS)
    if kind == "in_thinking":
        return _empty(IN_THINKING_IDS)
    if kind == "cut":
        return {"content": None, "reasoning": "long"}, "length", [100, 5]
    return {"content": "DISCUSSION\n```\nls\n```"}, "stop", [9, TURN_END]


def test_the_relay_splits_empty_replies_by_their_token_ids():
    up = _Upstream(_script_by_scenario)
    relay = C.EndpointRelay(up.url, "127.0.0.1").start()
    try:
        replies = [_post(relay.url, {"model": "m", "messages": [{"role": "user",
                                                                 "content": k}]})
                   for k in ("dropped", "in_thinking", "dropped", "cut", "ok")]
    finally:
        relay.stop()
        up.close()
    # the endpoint was known to be vLLM from its model list: the FIRST request asked for ids
    assert all(r.get("return_token_ids") is True for r in up.seen)
    # the client gets the reply it asked for (ids set back to null)
    assert all(r["choices"][0]["token_ids"] is None for r in replies)
    rec = relay.record()
    assert rec["tool_call_token_id"] == TOOL_CALL
    assert (rec["completion_replies"], rec["replies_cut_off"],
            rec["replies_empty_after_reasoning"]) == (5, 1, 4)
    assert (rec["replies_answer_in_thinking"], rec["replies_tool_call_dropped"],
            rec["replies_empty_unclassified"]) == (1, 2, 0)
    assert [s["classified_as"] for s in rec["empty_after_reasoning_samples"]] == [
        "tool_call_dropped", "answer_in_thinking", "tool_call_dropped"]
    res = {"status": "success"}
    reasons = C.relay_result_reasons(res, rec, "MSWE-agent")
    assert (res["answer_in_thinking"], res["answer_in_thinking_of"]) == (1, 5)
    assert (res["tool_call_dropped_turns"], res["unclassified_empty_turns"],
            res["empty_after_reasoning_turns"]) == (2, 0, 3)
    assert res["status"] == "success"
    text = "; ".join(reasons)
    assert ("2 model turn(s) whose generation was a native tool call the served endpoint's "
            "tool-call parser removed (MSWE-agent reads the reply's content and declares no "
            "tools") in text
    assert "1 model turn(s) cut off" in text and "unclassified" not in text


def test_a_relay_without_token_ids_leaves_empty_replies_unclassified():
    # a server that is not vLLM: no token ids are asked for, so nothing can be told apart
    up = _Upstream(_script_by_scenario, owner="someone-else")
    relay = C.EndpointRelay(up.url, "127.0.0.1").start()
    try:
        _post(relay.url, {"model": "m", "messages": [{"role": "user", "content": "dropped"}]})
    finally:
        relay.stop()
        up.close()
    rec = relay.record()
    # (the fake's reply carries prompt_token_ids, so the relay learns vLLM after it)
    assert "return_token_ids" not in up.seen[0]
    assert (rec["replies_answer_in_thinking"], rec["replies_tool_call_dropped"],
            rec["replies_empty_unclassified"]) == (0, 0, 1)
    res = {"status": "success"}
    reasons = C.relay_result_reasons(res, rec, "MSWE-agent")
    assert res["answer_in_thinking"] == 0 and res["unclassified_empty_turns"] == 1
    assert any("unclassified" in r for r in reasons)


def test_a_streamed_empty_reply_is_unclassified():
    relay = C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1")
    relay.count_replies([(False, True)])                 # _StreamTally: no token ids
    rec = relay.record()
    assert (rec["replies_answer_in_thinking"], rec["replies_empty_unclassified"]) == (0, 1)


# ------------------------------------------------------------------------------------------
# swe_agentic 2.x: the replies reach the endpoint through the relay
# ------------------------------------------------------------------------------------------
def test_swe_agentic_classifies_the_empty_turns_from_the_relay(monkeypatch, tmp_path):
    from test_swe_agent_v2_counters import (_desc, _driver_result, _messages, _proc,
                                            _resolve_all, _task, _write_traj)
    monkeypatch.setattr(C, "require_task_network_blocked", lambda *a, **k: None)
    monkeypatch.setattr(C, "host_pull_images",
                        lambda images, workers=1: {i: {"state": "present"} for i in images})
    up = _Upstream(_script_by_scenario)
    msgs = _messages()          # 2 empty turns per instance in its trajectory

    def _run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            spec = json.load(open(cmd[-1]))
            iid = spec["instance"]["instance_id"]
            api = spec["overrides"]["model"]["model_kwargs"]["api_base"]
            # the 2.x agent's replies, through the relay: per instance one native tool call the
            # parser removed and one answer inside the thinking channel, plus a usable reply
            for kind in ("dropped", "in_thinking", "ok"):
                _post(api, {"model": "m", "messages": [{"role": "user", "content": kind}]})
            d = f"{spec['output_dir']}/{iid}"
            _write_traj(f"{d}/{iid}.traj.json", msgs, 6)
            with open(f"{d}/result.json", "w", encoding="utf-8") as f:
                json.dump(dict(_driver_result(msgs, truncated=0),
                               trajectory_file=f"{d}/{iid}.traj.json"), f)
        return _proc()
    monkeypatch.setattr(A.subprocess, "run", _run)
    monkeypatch.setenv("GBENCH_SWE_BENCH_MULTILINGUAL_TRAJ_DIR", str(tmp_path / "trajs"))
    monkeypatch.setattr(A, "describe_agent_v2", lambda v: _desc())
    monkeypatch.setattr(A, "bridge_gateway", lambda: ("127.0.0.1", ""))
    monkeypatch.setattr(A, "require_relay_reachable", lambda *a, **k: None)
    try:
        res = A.run_agentic("swe_bench_multilingual", "m", up.url, 1,
                            [_task("a__a-1"), _task("b__b-2")], _resolve_all, {},
                            agent_v2="2.4.6")
    finally:
        up.close()
    assert res["empty_after_reasoning_turns"] == 4         # from the trajectories
    # numerator and denominator both from the relay's 6 replies
    assert (res["answer_in_thinking"], res["answer_in_thinking_of"]) == (2, 6)
    assert (res["tool_call_dropped_turns"], res["unclassified_empty_turns"]) == (2, 0)
    assert ("2 model turn(s) whose generation was a native tool call the served endpoint's "
            "tool-call parser removed") in res["leaderboard_comparable_reason"]


def test_swe_agentic_without_a_relay_counts_every_empty_turn_unclassified():
    traces = [{"empty_after_reasoning_turns": 2, "agent_steps": 5},
              {"empty_after_reasoning_turns": None, "agent_steps": 7}]      # unmeasured
    res = {}
    reasons = A._empty_turn_classes(res, traces, None)
    # over the measured instance's replies only (the unmeasured one adds no turns)
    assert (res["answer_in_thinking"], res["answer_in_thinking_of"]) == (0, 5)
    assert res["unclassified_empty_turns"] == 2 and len(reasons) == 1


# ------------------------------------------------------------------------------------------
# agent_dojo
# ------------------------------------------------------------------------------------------
def _dojo_client(up_url, replies):
    sent = []

    def create(**kw):
        sent.append(kw)
        msg, finish, ids = replies.pop(0)
        asked = (kw.get("extra_body") or {}).get("return_token_ids")
        choice = types.SimpleNamespace(
            finish_reason=finish, token_ids=ids if asked else None,
            message=types.SimpleNamespace(content=msg.get("content"), tool_calls=None,
                                          reasoning=msg.get("reasoning")))
        return types.SimpleNamespace(choices=[choice], model_extra={"prompt_token_ids": None})
    client = types.SimpleNamespace(base_url=up_url + "/", api_key="EMPTY",
                                   chat=types.SimpleNamespace(
                                       completions=types.SimpleNamespace(create=create)))
    return client, sent


def test_agent_dojo_splits_empty_replies_by_their_token_ids():
    up = _Upstream(lambda r: _empty(None))      # serves /tokenize
    try:
        client, sent = _dojo_client(up.url, [_empty(IN_THINKING_IDS), _empty(DROPPED_CALL_IDS),
                                             _empty(IN_THINKING_IDS)])
        watch = AD._EndpointWatch()
        watch.install(client)

        def episode():
            for _ in range(3):
                client.chat.completions.create(model="m")
            return True, False
        rec = watch.run(episode)
    finally:
        up.close()
    # the first reply showed vLLM (its prompt_token_ids key): later requests ask for token ids
    assert "extra_body" not in sent[0]
    assert sent[1]["extra_body"] == {"return_token_ids": True} and watch.vllm is True
    assert rec["empty_after_reasoning"] == 3
    assert (rec.get("empty_unclassified"), rec.get("tool_call_dropped"),
            rec.get("answer_in_thinking")) == (1, 1, 1)
    result = {"sample_traces": [{"requests": 3}]}
    AD._apply_summary(result, {"benign": {"s/t": True}}, {},
                      {"answer_in_thinking": 1, "tool_call_dropped": 1, "empty_unclassified": 1},
                      [])
    assert (result["answer_in_thinking"], result["answer_in_thinking_of"]) == (1, 3)
    reason = result["leaderboard_comparable_reason"]
    assert "1 repl(ies) whose generation held a native tool call" in reason
    assert "without token ids" in reason and result["status"] == "success"


# ------------------------------------------------------------------------------------------
# the upstream re-ask (gpqa / causalbench / longbench_v2_code)
# ------------------------------------------------------------------------------------------
def test_the_reask_rules_follow_each_upstream():
    def reply(content, finish="stop"):
        return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}
    assert C.reply_content_is_null(reply(None)) and not C.reply_content_is_null(reply(""))
    assert C.reply_content_is_null_or_empty(reply(None))
    assert C.reply_content_is_null_or_empty(reply(""))
    assert not C.reply_content_is_null_or_empty(reply(" x"))
    assert not C.reply_content_is_null({"choices": []})
    # only a reply the model ended itself: one cut off at the budget is the model's outcome
    for finish in ("length", None, "tool_calls"):
        assert not C.reply_content_is_null(reply(None, finish))
        assert not C.reply_content_is_null_or_empty(reply("", finish))
        assert not C.reply_content_is_null_or_empty(reply(None, finish))


@pytest.mark.parametrize("nulls,expect_reasked,expect_exhausted", [(0, 0, 0), (3, 1, 0),
                                                                   (9, 1, 1)])
def test_the_reask_relay_resends_the_same_request_until_content_arrives(nulls, expect_reasked,
                                                                         expect_exhausted):
    left = {"n": nulls}

    def script(req):
        if left["n"]:
            left["n"] -= 1
            return {"content": None, "reasoning": "the answer is (A)"}, "stop", IN_THINKING_IDS
        return {"content": "Answer: A"}, "stop", [9, TURN_END]
    up = _Upstream(script)
    body = {"model": "m", "messages": [{"role": "user", "content": "q"}], "temperature": 1.0,
            "max_tokens": 4096, "return_token_ids": True}
    try:
        with C.upstream_reask_relay("gpqa", up.url, C.reply_content_is_null, 5) as relay:
            got = _post(relay.url, body)
            summary = C.reask_summary(relay, "rule")
    finally:
        up.close()
    sent = 1 + min(nulls, 5)                # the request, then up to 5 re-asks
    assert len(up.seen) == sent
    # every re-ask is the request exactly as the client sent it
    assert all(r == body for r in up.seen)
    assert (summary["requests_reasked"], summary["reasks_sent"],
            summary["reasks_exhausted"]) == (expect_reasked, min(nulls, 5), expect_exhausted)
    assert summary["replies_seen"] == sent
    content = got["choices"][0]["message"]["content"]
    assert content == (None if nulls > 5 else "Answer: A")


def test_an_empty_string_is_a_reply_to_the_sampler_but_not_to_pred_py():
    up = _Upstream(lambda r: ({"content": ""}, "stop", [TURN_END]))
    body = {"model": "m", "messages": [{"role": "user", "content": "q"}]}
    try:
        with C.upstream_reask_relay("gpqa", up.url, C.reply_content_is_null, 5) as relay:
            _post(relay.url, body)
        assert relay.reask_record()["reasks_sent"] == 0 and len(up.seen) == 1
        with C.upstream_reask_relay("longbench_v2_code", up.url,
                                    C.reply_content_is_null_or_empty, 2) as relay:
            _post(relay.url, body)
        assert relay.reask_record() == {"reask_limit": 2, "requests_reasked": 1,
                                        "reasks_sent": 2, "reasks_exhausted": 1,
                                        "reasks_cut_short": 0, "cut_off_not_reasked": 0}
    finally:
        up.close()


def test_gpqa_through_base_scores_the_reply_the_sampler_returns(monkeypatch):
    """The real base request path through the re-ask relay: the first reply to each question is
    null-content (the answer inside the thinking channel), the re-ask answers it; base scores
    the re-asked reply and the run has no answer-in-thinking reply and no re-ask reason."""
    from gbench.runners.eval_suites import gpqa as G
    first = {}

    def script(req):
        q = json.dumps(req["messages"])        # one prompt per (question, permutation)
        if q not in first:
            first[q] = True
            return {"content": None, "reasoning": "Answer: A"}, "stop", IN_THINKING_IDS
        return {"content": "Answer: A"}, "stop", [9, TURN_END]
    up = _Upstream(script)
    rows = [{"Question": f"q{i}?", "Correct Answer": "c", "Incorrect Answer 1": "w1",
             "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3", "Subdomain": "s"}
            for i in range(2)]
    monkeypatch.setattr(G, "_load_gpqa_rows", lambda config: rows)
    try:
        res = G._run_simple_evals_gpqa("gpqa", "gpqa_main", 2, "m", up.url, 2, True,
                                       attempt_count=1, max_output_tokens=4096)
    finally:
        up.close()
    reask = res["upstream_reask"]
    n = 2 * G.N_REPEATS
    assert reask["requests_reasked"] == len(first) and reask["reasks_exhausted"] == 0
    assert len(up.seen) == n + len(first) and res["total_questions"] == n
    # every copy was scored on its re-asked reply ("Answer: A")
    assert all(t["response_text"] == "Answer: A" for t in res["sample_traces"])
    assert int(res.get("answer_in_thinking") or 0) == 0
    assert "null content" not in (res.get("leaderboard_comparable_reason") or "")
    assert "answer inside the thinking channel" not in (res.get("leaderboard_comparable_reason")
                                                        or "")


@pytest.mark.parametrize("always_null,comparable_reason", [
    (False, None),
    (True, "still came back with null content when gbench stopped re-asking (after up to 5 "
           "re-asks)"),
])
def test_causalbench_through_base_asks_a_null_reply_again(monkeypatch, always_null,
                                                          comparable_reason):
    """Corr2Cause's Chatbot retries a null reply (raw_query's .strip() on None raises, ask
    retries): through the real base path and the re-ask relay, the re-asked reply is the one
    read; a reply still null after the 5 re-asks is read as base reads it and named."""
    from gbench.runners.eval_suites import causalbench as CB
    monkeypatch.setattr(CB, "_require_nltk", lambda: None)
    monkeypatch.setattr(CB, "predicted_relation",
                        lambda text: "entailment" if "yes" in str(text or "").lower()
                        else "contradiction")
    first = {}

    def script(req):
        q = json.dumps(req["messages"])
        if always_null or q not in first:
            first[q] = True
            return {"content": None, "reasoning": "Yes."}, "stop", IN_THINKING_IDS
        return {"content": "Yes"}, "stop", [9, TURN_END]
    up = _Upstream(script)
    samples = [([{"role": "system", "content": CB.SYSTEM_PROMPT},
                 {"role": "user", "content": f"Question: p{i}\nCan we deduct the following: "
                                             f"h{i}? Just answer \"Yes\" or \"No.\"\n\nAnswer:"}],
                "entailment", {"category": "3", "num_variables": 3, "relation": "x"})
               for i in range(2)]
    monkeypatch.setattr(CB, "_load_causalbench_samples", lambda **kw: samples)
    try:
        res = CB.run_causalbench("m", up.url, 2, enable_thinking=True, max_output_tokens=4096)
    finally:
        up.close()
    reask = res["upstream_reask"]
    assert reask["requests_reasked"] == 2
    assert reask["reasks_exhausted"] == (2 if always_null else 0)
    assert len(up.seen) == (12 if always_null else 4)
    reason = res.get("leaderboard_comparable_reason") or ""
    if comparable_reason is None:
        assert res["exact_label_matches"] == 2 and "null content" not in reason
        assert int(res.get("answer_in_thinking") or 0) == 0
    else:
        assert "2 request(s) " + comparable_reason in reason
        assert res["answer_in_thinking"] == 2       # the replies that reached base
