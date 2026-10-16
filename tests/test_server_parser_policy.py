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
"""The server-parser policy and the shared provenance and docs that follow the code.

* Server-parser policy (base.recover_server_parser_text): on a request sent with thinking off, text
  the server's reasoning parser moved into `reasoning` is the model's output, recovered in
  generation order (from the reply's token ids, else reasoning then content) and scored, counted
  and named as server_parser_moved_text. A reply the tool-call parser deleted (no content, no
  reasoning, no tool call, completion_tokens > 0) is recovered from its token ids
  (server_parser_dropped_text) or, when it cannot be, is a named harness error
  (server_parser_text_lost), left out of the score.
* Every image label the shared docs and provenance name equals the code constant.
* The docs name the Gemini scope (model roles only) and each non-model backend's prerequisites.
Nothing here reaches a network, docker or Gemini.
"""

import asyncio
import json
import os
import re
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import base

REPO = Path(__file__).resolve().parents[1]


# The README was split: these docs hold sections moved verbatim out of the README, so a
# negative check that used to scan the whole README scans README.md plus all of them.
_README_SPLIT_DOCS = (
    "docs/installation.md", "docs/environment-variables.md", "docs/performance.md",
    "docs/eval-suites.md", "docs/golden-set.md", "docs/remote-endpoints.md",
    "docs/results.md", "docs/deployment.md", "docs/cli-reference.md",
)
_README_AND_SPLIT_DOCS = ("README.md", *_README_SPLIT_DOCS)


def _read(rel):
    return (REPO / rel).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# the pure policy
# --------------------------------------------------------------------------- #
def test_signal_names_moved_text_only_on_a_no_think_request_without_bound_calls():
    sig = base.server_parser_signal
    assert sig(content="x", reasoning="thought text", tool_calls=None, thinking=False,
               completion_tokens=9) == base.SERVER_PARSER_MOVED_TEXT
    assert sig(content=None, reasoning="t", tool_calls=None, thinking=False,
               completion_tokens=9) == base.SERVER_PARSER_MOVED_TEXT
    # --thinking: the reasoning channel is the model's reasoning
    assert sig(content="x", reasoning="t", tool_calls=None, thinking=True,
               completion_tokens=9) is None
    # a reply the server bound tool calls for is left as it is
    assert sig(content="", reasoning="t", tool_calls=[{"id": "1"}], thinking=False,
               completion_tokens=9) is None
    assert sig(content="plain answer", reasoning=None, tool_calls=None, thinking=False,
               completion_tokens=9) is None
    assert sig(content="   ", reasoning="  ", tool_calls=None, thinking=False,
               completion_tokens=0) is None


def test_signal_names_dropped_text_in_either_mode():
    for thinking in (False, True):
        assert base.server_parser_signal(
            content=None, reasoning=None, tool_calls=None, thinking=thinking,
            completion_tokens=14) == base.SERVER_PARSER_DROPPED_TEXT
        assert base.server_parser_signal(
            content="", reasoning="", tool_calls=[], thinking=thinking,
            completion_tokens=0) is None


def test_generation_text_leaves_out_the_stop_token_and_cuts_at_a_stop_string():
    vocab = {1: "<eos>", 2: "a", 3: "b", 4: "Observation:", 5: "c", 50: "<|tool_response>"}

    def decode(ids):
        return "".join(vocab[i] for i in ids)

    gen = base.generation_text_from_token_ids
    # end of sequence / a stop token id: vLLM appends it to token_ids, not to the text
    assert gen([2, 3, 1], decode, "stop", None) == "ab"
    assert gen([2, 3, 50], decode, "stop", 50) == "ab"
    # a stop string cuts the text where it starts, as the server cuts it
    assert gen([2, 4, 5], decode, "stop", "Observation:") == "a"
    # cut off at the budget: every id is text
    assert gen([2, 3, 5], decode, "length", None) == "abc"
    assert gen([], decode, "stop", None) is None
    assert gen(None, decode, "stop", None) is None
    assert gen([2], None, "length", None) is None
    assert gen([2, 3], lambda ids: None, "length", None) is None

    def boom(ids):
        raise RuntimeError("no tokenizer")
    assert gen([2, 3], boom, "length", None) is None
    assert gen([1], decode, "stop", None) == ""


def test_moved_text_is_recovered_in_generation_order():
    rec = base.recover_server_parser_text
    # from the token ids: the whole generation, special tokens as the tokenizer writes them
    r = rec(content="print(1)\n", reasoning="def f():", tool_calls=None, thinking=False,
            completion_tokens=12, generation_text="def f():<channel|>print(1)\n")
    assert r == base.ParserRecovery(base.SERVER_PARSER_MOVED_TEXT,
                                    "def f():<channel|>print(1)\n", "token_ids")
    # without token ids: the reasoning field, then the content
    r = rec(content="print(1)\n", reasoning="def f():", tool_calls=None, thinking=False,
            completion_tokens=12, generation_text=None)
    assert r == base.ParserRecovery(base.SERVER_PARSER_MOVED_TEXT, "def f():print(1)\n",
                                    "reasoning_then_content")
    r = rec(content=None, reasoning="Thought: list the mail", tool_calls=None, thinking=False,
            completion_tokens=12)
    assert r.text == "Thought: list the mail" and r.recovered_from == "reasoning_then_content"


def test_dropped_text_is_recovered_from_token_ids_or_named_a_harness_error():
    rec = base.recover_server_parser_text
    call = "<|tool_call>call:EmailClientV2__list_emails{}<tool_call|>"
    r = rec(content=None, reasoning=None, tool_calls=None, thinking=False,
            completion_tokens=14, generation_text=call)
    assert r == base.ParserRecovery(base.SERVER_PARSER_DROPPED_TEXT, call, "token_ids")
    r = rec(content=None, reasoning=None, tool_calls=None, thinking=True,
            completion_tokens=14, generation_text=None)
    assert r == base.ParserRecovery(base.SERVER_PARSER_TEXT_LOST, None, None)
    # a generation that decodes to nothing visible is an empty reply: the model's, no event
    r = rec(content=None, reasoning=None, tool_calls=None, thinking=False,
            completion_tokens=2, generation_text="\n ")
    assert r == base.ParserRecovery(None, None, None)
    # nothing to recover
    assert rec(content="answer", reasoning=None, tool_calls=None, thinking=False,
               completion_tokens=3, generation_text="answer").event is None


# --------------------------------------------------------------------------- #
# the request path
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, body, status=200):
        self.status = status
        self._body = body

    async def json(self):
        return self._body

    async def text(self):
        return json.dumps(self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    def __init__(self, body):
        self.body = body
        self.sent = []

    def post(self, url, json=None, **kw):
        self.sent.append(json)
        return _Resp(self.body)


def _send(body, thinking=False, extra=None):
    session = _Session(body)

    async def go():
        return await base._send_single_request(
            session=session, api_url="http://stub/v1/chat/completions", model_name="m",
            messages=[{"role": "user", "content": "q"}], extra_payload=dict(extra or {}),
            semaphore=asyncio.Semaphore(1), pbar=None, max_output_tokens=64,
            temperature=0.0, thinking=thinking)
    return asyncio.run(go()), session.sent


def _body(content, reasoning=None, token_ids=None, tokens=10, finish="stop", stop_reason=None,
          tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning"] = reasoning
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    choice = {"message": msg, "finish_reason": finish, "stop_reason": stop_reason}
    if token_ids is not None:
        choice["token_ids"] = token_ids
    return {"choices": [choice], "usage": {"completion_tokens": tokens}}


def test_every_request_asks_for_the_generated_token_ids():
    reply, sent = _send(_body("plain"))
    assert sent[0]["return_token_ids"] is True
    assert reply.text == "plain" and reply.parser_event is None


def test_a_no_think_reply_with_moved_text_is_scored_as_reasoning_then_content():
    reply, _ = _send(_body("print(1)\n", reasoning="def f():"))
    assert reply.text == "def f():print(1)\n"
    assert reply.parser_event == base.SERVER_PARSER_MOVED_TEXT
    assert reply.parser_recovered_from == "reasoning_then_content"
    assert reply.reasoning == "def f():"          # the server's field is kept for the trace


def test_a_no_think_reply_with_token_ids_is_rebuilt_from_them(monkeypatch):
    vocab = {7: "def f():", 8: "<channel|>", 9: "print(1)\n", 106: "<turn|>"}
    monkeypatch.setattr(base, "decode_generated_token_ids",
                        lambda ids: "".join(vocab[i] for i in ids))
    reply, _ = _send(_body("print(1)\n", reasoning="def f():", token_ids=[7, 8, 9, 106]))
    assert reply.text == "def f():<channel|>print(1)\n"
    assert reply.parser_recovered_from == "token_ids"


def test_a_thinking_reply_keeps_its_reasoning_apart():
    reply, _ = _send(_body("B", reasoning="long thought"), thinking=True)
    assert reply.text == "B" and reply.parser_event is None


def test_a_suite_that_turns_thinking_off_in_its_payload_gets_the_no_think_rule():
    reply, _ = _send(_body("x", reasoning="moved"), thinking=True,
                     extra={"chat_template_kwargs": {"enable_thinking": False}})
    assert reply.parser_event == base.SERVER_PARSER_MOVED_TEXT and reply.text == "movedx"


def test_a_deleted_tool_call_is_recovered_from_token_ids(monkeypatch):
    vocab = {48: "<|tool_call>", 49: "call:X{}", 51: "<tool_call|>", 50: "<|tool_response>"}
    monkeypatch.setattr(base, "decode_generated_token_ids",
                        lambda ids: "".join(vocab[i] for i in ids))
    reply, _ = _send(_body(None, token_ids=[48, 49, 51, 50], tokens=4, stop_reason=50))
    assert reply.text == "<|tool_call>call:X{}<tool_call|>"
    assert reply.parser_event == base.SERVER_PARSER_DROPPED_TEXT
    assert base.classify_reply(reply) == "ok"


def test_a_deleted_tool_call_without_token_ids_is_lost_not_empty():
    reply, _ = _send(_body(None, tokens=14, stop_reason=50))
    assert reply.parser_event == base.SERVER_PARSER_TEXT_LOST
    assert reply.text == ""


def test_undecodable_token_ids_are_lost(monkeypatch):
    monkeypatch.setattr(base, "decode_generated_token_ids", lambda ids: None)
    reply, _ = _send(_body(None, token_ids=[48, 49, 50], tokens=3))
    assert reply.parser_event == base.SERVER_PARSER_TEXT_LOST


def test_a_bound_tool_call_is_left_as_it_is():
    calls = [{"id": "1", "type": "function",
              "function": {"name": "f", "arguments": "{\"a\": 1}"}}]
    reply, _ = _send(_body("", reasoning="t", tool_calls=calls))
    assert reply.parser_event is None and reply.tool_calls == calls


def test_the_server_detokenize_endpoint_is_asked_first(monkeypatch):
    seen = {}

    class _R:
        def __init__(self, data):
            self._d = data

        def read(self):
            return self._d

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data)
        return _R(json.dumps({"prompt": "<|tool_call>x<tool_call|>"}).encode())

    monkeypatch.setattr(base.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setitem(base._RUN_KNOBS, "tokenize_url", "http://stub:8000")
    monkeypatch.setitem(base._RUN_KNOBS, "served_model", "m")
    base._NO_DETOKENIZE_ENDPOINT.discard("http://stub:8000")
    assert base.decode_generated_token_ids([1, 2]) == "<|tool_call>x<tool_call|>"
    assert seen["url"] == "http://stub:8000/detokenize"
    assert seen["body"] == {"model": "m", "tokens": [1, 2]}


def test_a_missing_detokenize_endpoint_falls_back_to_the_run_tokenizer(monkeypatch):
    import urllib.error

    def fake_urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", {}, None)

    class _Tok:
        def decode(self, ids, skip_special_tokens=True):
            assert skip_special_tokens is False
            return "|".join(str(i) for i in ids)

    monkeypatch.setattr(base.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setitem(base._RUN_KNOBS, "tokenize_url", "http://stub2:8000")
    monkeypatch.setitem(base._RUN_KNOBS, "served_model", "m")
    monkeypatch.setattr(base, "_run_tokenizer", lambda: _Tok())
    base._NO_DETOKENIZE_ENDPOINT.discard("http://stub2:8000")
    assert base.decode_generated_token_ids([3, 4]) == "3|4"
    assert "http://stub2:8000" in base._NO_DETOKENIZE_ENDPOINT
    base._NO_DETOKENIZE_ENDPOINT.discard("http://stub2:8000")


# --------------------------------------------------------------------------- #
# the run: counted, named, never a model failure
# --------------------------------------------------------------------------- #
def _run(replies, **kw):
    """Sample i gets replies[i] (keyed on its prompt, whatever order the requests run in)."""
    async def fake_send(**k):
        return replies[int(k["messages"][-1]["content"][1:])]

    samples = [([{"role": "user", "content": f"q{i}"}], "42", {}) for i in range(len(replies))]
    with mock.patch.object(base, "_send_single_request", fake_send):
        return base.run_eval_suite(eval_name="t", model_name="m", base_url="http://x",
                                   concurrency=1, samples=samples,
                                   eval_fn=lambda r, g: "42" in r, **kw)


def test_recovered_text_is_scored_counted_and_named():
    moved = base.Reply("the answer is 42", None, "stop", "the answer is 42", None, 9, None,
                       None, base.SERVER_PARSER_MOVED_TEXT, "reasoning_then_content")
    ok = base.Reply("42", None, "stop", None, None, 3)
    res = _run([moved, ok])
    assert res["accuracy"] == 100.0 and res["status"] == "success"
    assert res["server_parser_events"] == {base.SERVER_PARSER_MOVED_TEXT: 1}
    assert res["server_parser_samples"] == {base.SERVER_PARSER_MOVED_TEXT: [0]}
    assert "server_parser_moved_text 1" in res["server_parser_note"]
    assert "reasoning_then_content 1" in res["server_parser_note"]
    t0 = res["sample_traces"][0]
    assert t0["server_parser"] == {"event": base.SERVER_PARSER_MOVED_TEXT,
                                   "recovered_from": "reasoning_then_content"}
    assert "server_parser" not in res["sample_traces"][1]
    assert res["empty_responses"] == 0 and res["failed_requests"] == 0


def test_a_lost_reply_is_a_named_harness_error_left_out_of_the_score():
    lost = base.Reply("", None, "stop", None, None, 14, 50, None,
                      base.SERVER_PARSER_TEXT_LOST, None)
    ok = base.Reply("42", None, "stop", None, None, 3)
    res = _run([lost, ok])
    assert res["accuracy"] == 100.0
    assert res["scoring_excluded"] == 1
    assert res["status"] == "completed_with_errors"
    assert "server_parser_text_lost 1" in res["status_reason"]
    assert "got no reply" not in res["status_reason"]
    assert res["request_harness_errors"] == {base.SERVER_PARSER_TEXT_LOST: 1}
    assert res["empty_responses"] == 0 and res["failed_requests"] == 0
    t0 = res["sample_traces"][0]
    assert t0["status"] == "SERVER_PARSER_TEXT_LOST" and t0["scoring_excluded"] is True
    assert t0["harness_error"]["kind"] == base.SERVER_PARSER_TEXT_LOST
    assert "14 token(s)" in t0["harness_error"]["detail"]


def test_a_run_whose_every_reply_is_lost_withholds_its_score():
    lost = base.Reply("", None, "stop", None, None, 14, 50, None,
                      base.SERVER_PARSER_TEXT_LOST, None)
    res = _run([lost, lost])
    assert res["accuracy"] is None and res["status"] == "failed"
    assert "server_parser_text_lost 2" in res["accuracy_withheld_reason"]


def test_a_forced_final_whose_call_was_deleted_is_still_asked_again(monkeypatch):
    """The recovered text of a deleted tool call is the call, not an answer: the forced-final
    retry fires as it did for output_discarded."""
    dropped = base.Reply("<|tool_call>call:x{}<tool_call|>", None, "stop", None, None, 9, 50,
                         None, base.SERVER_PARSER_DROPPED_TEXT, "token_ids")
    answer = base.Reply("final: 42", None, "stop", None, None, 5)
    calls = []

    async def fake_send(**k):
        calls.append(k)
        return answer

    monkeypatch.setattr(base, "_send_single_request", fake_send)

    async def go():
        return await base._retry_discarded_final(
            reply=dropped, convo=[{"role": "user", "content": "q"}], session=None,
            api_url="http://x", model_name="m", payload={}, semaphore=asyncio.Semaphore(1),
            budget=None, temperature=0.0)
    assert asyncio.run(go()) is answer and len(calls) == 1


# --------------------------------------------------------------------------- #
# shared docs and provenance follow the code
# --------------------------------------------------------------------------- #
def _flat(text):
    return re.sub(r"\s+", " ", text)


def test_the_server_parser_policy_is_documented():
    knobs = _flat(_read("docs/evals/env-knobs.md"))
    for name in (*base.SERVER_PARSER_EVENTS, "base.recover_server_parser_text",
                 "`return_token_ids`", "`/detokenize`", "`server_parser_events`",
                 "`server_parser_samples`", "`server_parser_note`", "reasoning_then_content",
                 "--reasoning-parser gemma4 --enable-auto-tool-choice --tool-call-parser gemma4"):
        assert name in knobs, name
    readme = _flat(_read("docs/remote-endpoints.md"))   # moved out of README.md
    assert "`return_token_ids`" in readme and "`/detokenize`" in readme
    assert "--reasoning-parser gemma4 --enable-auto-tool-choice --tool-call-parser gemma4" in readme


def test_the_gemini_scope_is_model_roles_only():
    knobs = _flat(_read("docs/evals/env-knobs.md"))
    assert "Gemini scope: model roles only." in knobs
    assert "No suite reads a SerpAPI, Brave, RapidAPI" not in knobs
    assert "(judge, user simulator, web search, URL fetching, embeddings)" not in knobs
    scope = knobs.split("Gemini scope: model roles only.", 1)[1].split("- Credential values", 1)[0]
    for suite, needle in (("toolbench", "http://8.130.32.149:8080/rapidapi"),
                          ("api_bank", "`googletrans`"),
                          ("complexfuncbench", "`RAPID_API_KEY`"),
                          ("mcp_bench", "`NCI_API_KEY`"),
                          ("mcp_atlas", "`E2B_API_KEY`")):
        assert f"- `{suite}`:" in scope and needle in scope, suite
    assert "never a Gemini substitution" in scope
    for suite in ("toolbench", "api_bank", "complexfuncbench", "mcp_bench", "mcp_atlas"):
        assert f"- {suite}:" not in scope, suite
    readme = _flat(_read("docs/environment-variables.md"))   # moved out of README.md
    assert "(judge, user simulator, web search, URL fetching, embeddings)" not in readme
    for rel in _README_AND_SPLIT_DOCS:   # README.md and every doc split out of it
        assert ("(judge, user simulator, web search, URL fetching, embeddings)"
                not in _flat(_read(rel))), rel
    assert "Every model role a harness has" in readme
    assert "never a Gemini substitution" in readme


def test_the_endpoint_rejection_split_is_recorded_not_claimed_uniform():
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as CS
    for suite in ("gaia2", "toolbench"):
        assert "counts model endpoint replies everywhere" not in CS[suite]["method"], suite
    knobs = _flat(_read("docs/evals/env-knobs.md"))
    assert "HTTP 400/422 and context-window rejections follow one rule" in knobs
    from gbench.runners.eval_suites import base as _base
    for suite in ("gaia2", "toolbench", "quality", "mcp_bench", "terminal_bench"):
        assert suite in _base.ENDPOINT_REJECTION_ATTRIBUTION, suite
    assert "`_MODEL_CONTENT_REJECTION_STATUSES`" in knobs
    from gbench.runners import quality
    assert tuple(quality._MODEL_CONTENT_REJECTION_STATUSES) == (400, 422)
    from gbench.runners.eval_suites import swebench_common
    assert {"BadRequestError", "UnprocessableEntityError"} <= set(
        swebench_common.ENDPOINT_EXCEPTIONS)


def test_the_swe_provenance_names_the_test_timeout_and_exit_rules_the_code_keeps():
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as CS
    common = _read("gbench/runners/eval_suites/swebench_common.py")
    for key in ("test_timed_out", "test_timeout_instances", "scaffold_error_harness_reason"):
        assert key in common, key
    for suite in ("copilot_bench_swe", "swe_bench_multilingual"):
        m = CS[suite]["method"]
        assert "shows neither a patch-apply failure nor a test timeout (harness_no_verdict)" in m
        assert "(test_timed_out, test_timeout_instances)" in m
        assert "OPENAI_API_KEY=EMPTY" in m and "no patch-apply failure (harness_no_verdict)" \
            not in m
    live = CS["swe_bench_live"]["method"]
    assert "test_timed_out, test_timeout_instances" in live
    assert "exits_kept_as_model_outcome" in live
    assert "exit_environment_error or exit_error is an infrastructure failure" not in live
    mswe = CS["multi_swe_bench"]["method"]
    assert "swebench_common.scaffold_error_harness_reason" in mswe
    assert "and early_exit (SWEEnv.step" in mswe
    assert "early_exit, exit_api, an exit_error other than the step limit, or an unfinished" \
        not in mswe
    for suite in ("swe_bench_live", "multi_swe_bench"):
        src = _read(f"gbench/runners/eval_suites/{suite}.py")
        assert "exits_kept_as_model_outcome" in src and "agent_exit_note" in src, suite


def test_the_provenance_records_where_the_data_really_comes_from():
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as CS
    sp = CS["spider2"]
    assert sp["revision"] == "cafb867313aab4e674652054198f383cf4018943"
    assert "ce2e8366" not in json.dumps(sp)
    assert "not the stale HF xlangai/spider2-lite snapshot" in sp["dataset"]
    assert "SCORING_ERROR" in sp["method"]
    cy = CS["cyberseceval"]
    assert cy["upstream"].startswith("github.com/meta-llama/PurpleLlama")
    assert (cy["revision"] == "114d02388213ae791650b10fa61c8785661c2602"
            and "instruct-v2.json" in cy["dataset"] and "trailing newline kept" in cy["method"]
            and "df19f7c51911d46d0ae5c069b7093728db8e682890a44317a211ac61cb02c181"
            in cy["checksum"])
    tb = CS["terminal_bench"]["method"]
    assert "sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a" in tb
    assert ("harbor_version" in tb and "'harbor==0.20.0'" in tb
            and "neither checked nor recorded" not in tb)


def test_the_shared_docs_carry_the_requested_sandbox_and_container_notes():
    tc = _flat(_read("docs/evals/toolchains.md"))
    assert "a Win Rate call that does not win after a failed use of the network is a harness " \
           "error, left out of the Win Rate and counted in `win_rate_network_blocked_calls`" in tc
    assert "The worker reports what it can see and gbench refuses it unless it is isolated." in tc
    assert "shared by the run's `--threads`; the OOM kills in it are recorded as " \
           "`container_oom_kills`" in tc
    assert "each rollout container runs on docker's default bridge network with " \
           "OPENAI_API_KEY=EMPTY" in tc
    knobs = _flat(_read("docs/evals/env-knobs.md"))
    assert "(`vqa_common.mark_no_reply`, `vqa_common.own_loop_status`)" in knobs
    assert "marked nestful_win_blocked_by_sandbox and left out of the Win Rate" in knobs
    assert "the Win Rate runs the model's calls in a bubblewrap sandbox of its own whatever " \
           "GBENCH_SANDBOX says" in knobs
    readme = _flat(_read("docs/cli-reference.md"))   # moved out of README.md
    assert "For `aider_polyglot` the value is `benchmark.py --threads`" in readme
    from gbench.runners.eval_suites import aider_polyglot, nestful
    assert hasattr(aider_polyglot, "container_oom_kills")
    assert "win_rate_network_blocked_calls" in _read("gbench/runners/eval_suites/nestful.py")


def test_the_no_reply_kinds_leave_out_the_parser_harness_error():
    assert base.SERVER_PARSER_TEXT_LOST not in base.NO_REPLY_HARNESS_KINDS
    assert set(base.ENDPOINT_FAILURE_KINDS.values()) < base.NO_REPLY_HARNESS_KINDS
    assert base.MID_LOOP_FAILURE_KIND in base.NO_REPLY_HARNESS_KINDS
