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

"""complexfuncbench harness errors are never the model's outcome.

A sample whose model endpoint request failed on every attempt, whose Gemini call-equivalence
judge gave no verdict (an outage of every cascade model, or a reply without an is_equal verdict),
or whose driving raised in the harness is not scored: it is left out of the numerator and the
denominator of Success Rate and Call Accuracy on full and subset runs alike, counted in
harness_errors, named in leaderboard_comparable_reason, and the run is completed_with_errors. A
completeness / correctness judge that gave no score is left out of its mean and counted the same
way. A model reply with neither content nor tool calls is the model's outcome: a scored failure.

No network, no model server, no Gemini call: the endpoint, the judge and aiohttp are stubbed.
"""

import asyncio
import json
import logging
from collections import Counter
from pathlib import Path

import pytest

from gbench.runners.eval_suites import complexfuncbench as C

ROOT = Path(__file__).resolve().parent.parent
SECRET = "cfb-harness-secret-0123456789abcdef"
_DOMAINS = {"Car-Rental": 150, "Hotels": 150, "Attraction": 150, "Flights": 150, "Cross": 400}


class _LiveAPI:
    """A stand-in for the booking-com15 API of the response-based tier: each call answers its
    own response, so two calls that differ get different responses (the judge decides)."""

    def __init__(self):
        self.stats = Counter()

    async def call(self, func_call):
        return {"response": {"echo": json.dumps(func_call, sort_keys=True)}, "raised": False,
                "labels": ["ok"]}


def _cmp(usage=None):
    return C._CompareFC(C._load_exact_match_dict(), C._get_embedder(), usage, _LiveAPI())


def _tool_call(cid, name, args):
    return {"id": cid, "function": {"name": name, "arguments": json.dumps(args)}}


def _item(sample_id, query, golden_args):
    return {
        "id": sample_id,
        "functions": [{"name": "Search_Foo",
                       "parameters": {"required": ["q"], "properties": {"q": {"type": "string"}}}}],
        "conversations": [
            {"role": "user", "content": query},
            {"role": "assistant", "function_call": [{"name": "Search_Foo",
                                                     "arguments": golden_args}]},
            {"role": "observation", "content": [{"status": True, "data": "obs"}]},
            {"role": "assistant", "content": "final"},
        ],
    }


def _scored(sample_id, success, correct=1, total=1, resp=None):
    return {"id": sample_id, "request_failed": False,
            "message": "Success." if success else {"error_type": "stop_early"},
            "count_dict": {"success_turn_num": 1, "total_turn_num": 1, "correct_call_num": correct,
                           "total_call_num": total, "real_turn_num": 1},
            "resp_eval": resp}


def _upstream_basic_metric(results):
    """print_results.py basic_metric at the pinned upstream revision (success over /150, Cross
    /400 and /1000; call accuracy pooled; the 0/1/2 scores averaged)."""
    succ, calls, comp, corr = Counter(), [0, 0], [0, 0], [0, 0]
    for r in results:
        dom = r["id"].rsplit("-", 1)[0]
        succ[dom] += r["message"] == "Success."
        calls[0] += r["count_dict"]["correct_call_num"]
        calls[1] += r["count_dict"]["total_call_num"]
        if r["resp_eval"] is None:
            continue
        if r["resp_eval"]["complete"]["score"] in {0, 1, 2}:
            comp[0] += r["resp_eval"]["complete"]["score"]
            comp[1] += 1
        if r["resp_eval"]["correct"]["score"] in {0, 1, 2}:
            corr[0] += r["resp_eval"]["correct"]["score"]
            corr[1] += 1
    return {"domain": {k: v / 150 * 100 if k != "Cross" else v / 400 * 100
                       for k, v in succ.items()},
            "overall": sum(succ.values()) / 1000 * 100, "call": calls[0] / calls[1] * 100,
            "complete": comp[0] / comp[1], "correct": corr[0] / corr[1]}


# --------------------------------------------------------------------------- #
# _aggregate: evaluable denominator on every run shape
# --------------------------------------------------------------------------- #
def test_endpoint_failures_leave_the_denominator_on_full_and_subset_runs():
    ok = [_scored(f"{d}-{i}", True) for d, n in _DOMAINS.items() for i in range(n // 2)]
    bad = [{"id": f"{d}-{i}", "request_failed": True, "harness_error": "endpoint_failure",
            "endpoint_error": "HTTP 503", "message": {"error_type": "unknown_error"}}
           for d, n in _DOMAINS.items() for i in range(n // 2, n)]
    assert len(ok) == len(bad) == 500
    full = C._aggregate(ok + bad, 1000, True)
    subset = C._aggregate(ok + bad, 1000, False)
    for agg in (full, subset):
        assert agg["success_rate"] == 100.0          # was 50.0 on a full run (/1000)
        assert agg["call_accuracy"] == 100.0
        assert agg["n_scored"] == 500 and agg["n_request_failed"] == 500
        assert agg["n_harness_errors"] == 500
        assert agg["harness_error_samples"] == {"endpoint_failure": 500}
        assert agg["endpoint_errors"] == {"HTTP 503": 500}
    assert full["per_domain_success_rate"] == {d: 100.0 for d in _DOMAINS}
    assert full["per_domain_evaluated"] == {d: n // 2 for d, n in _DOMAINS.items()}
    assert subset["per_domain_success_rate"] is None


def test_a_clean_full_run_is_upstreams_normalisation():
    results = []
    for d, n in _DOMAINS.items():
        for i in range(n):
            resp = {"complete": {"score": i % 3}, "correct": {"score": (i + 1) % 3}}
            if i % 7 == 0:
                resp = {"complete": {"score": -2}, "correct": {"score": -2}}
            results.append(_scored(f"{d}-{i}", i % 4 == 0, correct=i % 3, total=2,
                                   resp=None if i % 11 == 0 else resp))
    up = _upstream_basic_metric(results)
    agg = C._aggregate(results, 1000, True)
    assert agg["n_harness_errors"] == 0
    assert agg["success_rate"] == round(up["overall"], 2)
    assert agg["per_domain_success_rate"] == {d: round(v, 2) for d, v in up["domain"].items()}
    assert agg["per_domain_evaluated"] == _DOMAINS
    assert agg["call_accuracy"] == round(up["call"], 2)
    assert agg["completeness"] == round(up["complete"], 4)
    assert agg["correctness"] == round(up["correct"], 4)


def test_every_harness_kind_is_left_out_and_counted():
    results = [
        _scored("Hotels-0", True, resp={"complete": {"score": 2}, "correct": {"score": 1}}),
        _scored("Hotels-1", False, correct=0, total=3),
        {"id": "Hotels-2", "request_failed": False, "harness_error": "call_judge_outage"},
        {"id": "Hotels-3", "request_failed": False, "harness_error": "call_judge_unparsed_reply"},
        {"id": "Hotels-4", "request_failed": False, "harness_error": "exception",
         "exception": "KeyError"},
        {"id": "Hotels-5", "request_failed": True, "message": {"error_type": "unknown_error"}},
    ]
    agg = C._aggregate(results, 6, False)
    assert agg["n_scored"] == 2 and agg["success_rate"] == 50.0
    assert agg["call_accuracy"] == 25.0                          # 1 / (1 + 3)
    assert agg["harness_error_samples"] == {"call_judge_outage": 1,
                                            "call_judge_unparsed_reply": 1,
                                            "endpoint_failure": 1, "exception": 1}
    assert agg["n_request_failed"] == 1 and agg["exception_types"] == {"KeyError": 1}
    assert agg["endpoint_errors"] == {"unrecorded": 1}
    assert set(agg["harness_error_samples"]) <= set(C.HARNESS_ERROR_KINDS)


# --------------------------------------------------------------------------- #
# per-sample classification
# --------------------------------------------------------------------------- #
def _evaluate(item, script, usage=None):
    state = {"i": 0}

    async def chat(oai_messages, tools):
        msg = script[state["i"]]
        state["i"] += 1
        return msg
    return asyncio.run(C._evaluate_one(item, chat, C._load_exact_match_dict(),
                                       C._get_embedder(), usage, _LiveAPI()))


def test_an_endpoint_failure_is_a_harness_error():
    r = _evaluate(_item("Hotels-0", "q", {"q": "x"}), [None])
    assert r["harness_error"] == "endpoint_failure" and r["request_failed"] is True
    assert "count_dict" not in r


def test_an_empty_model_message_is_the_models_failed_sample():
    r = _evaluate(_item("Hotels-0", "q", {"q": "x"}), [{"content": None, "tool_calls": []}])
    assert "harness_error" not in r and r["request_failed"] is False
    assert r["message"]["error_type"] == "empty_model_message"
    assert r["count_dict"]["total_call_num"] == 1 and r["count_dict"]["correct_call_num"] == 0
    agg = C._aggregate([r, _scored("Hotels-1", True)], 2, False)
    assert agg["n_scored"] == 2 and agg["success_rate"] == 50.0
    assert agg["n_empty_model_message"] == 1 and agg["n_harness_errors"] == 0


@pytest.mark.parametrize("reply, kind", [((None, "JUDGE_OUTAGE"), "call_judge_outage"),
                                         (("they look alike", "gemini-a"),
                                          "call_judge_unparsed_reply")])
def test_a_call_judge_without_a_verdict_is_a_harness_error(monkeypatch, reply, kind):
    async def fake_cascade(contents, *, config=None):
        # upstream's split: the system prompt as the system instruction, the template as content
        assert config.system_instruction == C._COMPARE_SYSTEM
        assert contents.startswith("Function list:\n```JSON\n")
        return reply
    monkeypatch.setattr(C, "judge_generate_cascade", fake_cascade)
    usage = Counter()
    # rule_based fails, value_checker passes (Search_Foo has no critical parameter) -> the judge
    item = _item("Cross-0", "q", {"q": "NY"})
    r = _evaluate(item, [{"content": None,
                          "tool_calls": [_tool_call("c1", "Search_Foo", {"q": "New York"})]}],
                  usage)
    assert {k: r[k] for k in ("id", "request_failed", "harness_error")} == {
        "id": "Cross-0", "request_failed": False, "harness_error": kind}
    assert "count_dict" not in r and r["resp_judge_calls"] == []
    assert [c["judge_model"] for c in r["call_judge_calls"]] == [reply[1]]
    assert r["gen_convs"][0] == {"role": "user", "content": "q"}
    assert usage == Counter({reply[1]: 1})
    # llm_based itself still returns None, as upstream's does
    cmp = _cmp(Counter())
    a, b = {"name": "Search_Foo", "arguments": {"q": "a"}}, {"name": "Search_Foo",
                                                              "arguments": {"q": "b"}}
    assert asyncio.run(cmp.llm_based([], [], a, b)) is None and cmp.judge_failure == kind
    with pytest.raises(C._CallJudgeNoVerdict) as ei:
        asyncio.run(cmp.compare_single_call([], [], a, b))
    assert ei.value.kind == kind


def test_a_call_judge_verdict_still_decides_the_call(monkeypatch):
    replies = iter([('{"is_equal": true, "reason": "same"}', "gemini-a"),
                    ('```JSON\n{"is_equal": false, "reason": "differ"}\n```', "gemini-a")])

    async def fake_cascade(contents, *, config=None):
        return next(replies)
    monkeypatch.setattr(C, "judge_generate_cascade", fake_cascade)
    cmp = _cmp(Counter())
    a, b = {"name": "Search_Foo", "arguments": {"q": "a"}}, {"name": "Search_Foo",
                                                              "arguments": {"q": "b"}}
    assert asyncio.run(cmp.compare_single_call([], [], a, b)) == (True, None)
    assert asyncio.run(cmp.compare_single_call([], [], a, b)) == (False, None)
    assert cmp.judge_failure is None


def test_a_harness_exception_records_only_its_class(monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)

    async def chat(oai_messages, tools):
        raise RuntimeError(f"upstream said key={SECRET}")
    caplog.set_level(logging.WARNING, logger=C.__name__)
    r = asyncio.run(C._evaluate_one(_item("Hotels-0", "q", {"q": "x"}), chat,
                                    C._load_exact_match_dict(), C._get_embedder()))
    assert {k: r[k] for k in ("id", "request_failed", "harness_error", "exception")} == {
        "id": "Hotels-0", "request_failed": False, "harness_error": "exception",
        "exception": "RuntimeError"}
    assert SECRET not in json.dumps(r)
    assert "RuntimeError" in caplog.text and SECRET not in caplog.text


def test_response_judge_failures_are_named_and_left_out(monkeypatch):
    _no_sleep(monkeypatch)
    replies = {C._COMPLETE_SYSTEM: (None, "JUDGE_OUTAGE"),
               C._CORRECT_SYSTEM: ("a fine answer overall", "gemini-a")}

    async def fake_cascade(contents, *, config=None):
        return replies[config.system_instruction]
    monkeypatch.setattr(C, "judge_generate_cascade", fake_cascade)
    usage = Counter()
    ev = asyncio.run(C._resp_eval([{"role": "user", "content": "q"}], "answer", usage))
    # the outage ends at once; the unparsed reply is asked again, 10 requests in all (upstream's
    # retry(max_attempts=10))
    assert usage == Counter({"JUDGE_OUTAGE": 1, "gemini-a": C._RESP_JUDGE_ATTEMPTS})
    usage = Counter({"JUDGE_OUTAGE": 1, "gemini-a": 1})
    assert ev["complete"] == {"score": -1, "reason": C._RESP_JUDGE_OUTAGE}
    assert ev["correct"] == {"score": -1, "reason": C._RESP_JUDGE_UNPARSED}
    results = [_scored("Hotels-0", True, resp=ev),
               _scored("Hotels-1", False, resp={"complete": {"score": 2},
                                                "correct": {"score": 0}})]
    r = _run(monkeypatch, C._aggregate(results, 2, False), usage)
    assert r["status"] == "completed_with_errors"
    assert r["success_rate"] == 50.0 and r["n_scored"] == 2      # the samples stay scored
    assert r["completeness"] == 2.0 and r["correctness"] == 0.0
    assert r["response_judge_unscored"] == {"completeness": 1, "correctness": 1}
    assert r["harness_errors"] == [{
        "error": "the Gemini completeness / correctness judge gave no score",
        "scores_left_out": {"completeness": 1, "correctness": 1}, "outages": 1,
        "unparsed_replies": 1}]
    assert ("the Gemini completeness / correctness judge gave no score for 1 / 1 final answer(s) "
            "(1 failed on every cascade model, 1 replied without a 0/1/2 score)"
            in r["leaderboard_comparable_reason"])
    assert r["judge_outages"] == 1


# --------------------------------------------------------------------------- #
# run_complexfuncbench: status, reason, note
# --------------------------------------------------------------------------- #
def _run(monkeypatch, agg, usage, n_items=None, is_full=False):
    monkeypatch.setattr(C, "check_complexfuncbench_prerequisites", lambda: (True, ""))

    async def fake_drive(*a, **k):
        return dict(agg, _is_full=is_full, _n_items=n_items or agg["n_scored"],
                    _embed_backend="transformers", _embed_dtype=None,
                    _judge_usage=Counter(usage))
    monkeypatch.setattr(C, "_drive_all", fake_drive)
    return C.run_complexfuncbench("m", "http://x/v1", 1, limit=n_items)


def test_a_clean_run_is_success_with_no_harness_errors(monkeypatch):
    results = [_scored("Hotels-0", True, resp={"complete": {"score": 2}, "correct": {"score": 2}})]
    r = _run(monkeypatch, C._aggregate(results, 1, False), {"gemini-a": 2})
    assert r["status"] == "success" and r["harness_errors"] == []
    assert r["n_harness_errors"] == 0 and "request_failure_note" not in r
    assert "harness errors" not in r["leaderboard_comparable_reason"]


def test_any_judge_outage_sets_the_status_and_is_named(monkeypatch):
    # judge outages the per-sample records do not account for are still a harness error
    results = [_scored("Hotels-0", True)]
    r = _run(monkeypatch, C._aggregate(results, 1, False), {"gemini-a": 3, "JUDGE_OUTAGE": 2})
    assert r["status"] == "completed_with_errors" and r["judge_outages"] == 2
    assert r["harness_errors"] == [{"error": "Gemini judge calls that failed on every cascade "
                                             "model", "judge_calls": 2}]
    assert ("harness errors, not the model's outcome and left out of the metrics: 2 Gemini "
            "judge call(s) failed on every cascade model") in r["leaderboard_comparable_reason"]


def test_all_samples_in_harness_errors_is_a_hard_error_naming_them(monkeypatch):
    results = [{"id": "Hotels-0", "request_failed": True, "harness_error": "endpoint_failure"},
               {"id": "Hotels-1", "request_failed": False, "harness_error": "call_judge_outage"}]
    with pytest.raises(RuntimeError) as ei:
        _run(monkeypatch, C._aggregate(results, 2, False), {"JUDGE_OUTAGE": 1}, n_items=2)
    msg = str(ei.value)
    assert msg.startswith("[INFRA REQUIRED] complexfuncbench")
    assert "call_judge_outage x1, endpoint_failure x1" in msg


class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status, self._payload, self._text = status, payload, text

    async def json(self):
        return self._payload

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    def __init__(self, reply):
        self.reply, self.payloads = reply, []

    def post(self, url, json=None):
        self.payloads.append(json)
        return self.reply(json)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _no_sleep(monkeypatch):
    async def instant(_s):
        return None
    monkeypatch.setattr(C.asyncio, "sleep", instant)


def test_post_chat_labels_each_failed_attempt(monkeypatch, caplog):
    _no_sleep(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", SECRET)
    caplog.set_level(logging.WARNING, logger=C.__name__)

    def run(reply):
        failures = []
        msg = asyncio.run(C._post_chat(_Session(reply), "http://x/v1/chat/completions", {},
                                       failures))
        return msg, failures
    over = ("This model's maximum context length is 8192 tokens. However, you requested 9000 "
            f"tokens. {SECRET}")
    n = C._REQUEST_ATTEMPTS
    assert n == 5                                  # models/qwen.py retry(max_attempts=5)
    assert run(lambda p: _Resp(400, text=over)) == (None, ["HTTP 400 (context window)"] * n)
    assert SECRET not in caplog.text
    assert run(lambda p: _Resp(503, text="busy")) == (None, ["HTTP 503"] * n)
    assert run(lambda p: _Resp(200, {"choices": []})) == (None, ["malformed reply"] * n)

    def boom(p):
        raise ConnectionResetError("reset")
    assert run(boom) == (None, ["ConnectionResetError"] * n)
    replies = iter([_Resp(502), _Resp(200, {"choices": [{"message": {"content": "hi"}}]})])
    assert run(lambda p: next(replies)) == ({"content": "hi"}, ["HTTP 502"])


def test_a_driven_run_with_harness_errors_reports_them(monkeypatch):
    """The whole driver: A succeeds, B's endpoint fails, C's call judge is out, D replies empty."""
    import aiohttp
    _no_sleep(monkeypatch)
    items = [_item("Hotels-0", "query A", {"q": "x"}), _item("Flights-0", "query B", {"q": "x"}),
             _item("Cross-0", "query C", {"q": "NY"}),
             _item("Attraction-0", "query D", {"q": "x"})]

    def reply(payload):
        query, turns = payload["messages"][0]["content"], len(payload["messages"])
        if query == "query A":
            if turns == 1:
                return _Resp(200, {"choices": [{"message": {
                    "content": None,
                    "tool_calls": [_tool_call("c1", "Search_Foo", {"q": "x"})]}}]})
            return _Resp(200, {"choices": [{"message": {"content": "final"}}]})
        if query == "query B":
            return _Resp(503, text="unavailable")
        if query == "query C":
            return _Resp(200, {"choices": [{"message": {
                "content": None,
                "tool_calls": [_tool_call("c1", "Search_Foo", {"q": "New York"})]}}]})
        return _Resp(200, {"choices": [{"message": {"content": None, "tool_calls": []}}]})
    session = _Session(reply)
    monkeypatch.setattr(aiohttp, "ClientSession", lambda **kw: session)
    monkeypatch.setattr(aiohttp, "TCPConnector", lambda **kw: None)
    monkeypatch.setattr(C, "_load_dataset", lambda limit: items)
    monkeypatch.setattr(C, "check_complexfuncbench_prerequisites", lambda: (True, ""))

    async def fake_cascade(contents, *, config=None):
        if config.system_instruction == C._COMPARE_SYSTEM:
            return None, "JUDGE_OUTAGE"
        return '{"score": 2, "reason": "fine"}', "gemini-x"
    monkeypatch.setattr(C, "judge_generate_cascade", fake_cascade)

    r = C.run_complexfuncbench("m", "http://x/v1", 2, limit=4)
    assert r["status"] == "completed_with_errors"
    assert r["n_samples"] == 4 and r["n_scored"] == 2 and r["n_success"] == 1
    assert r["accuracy"] == r["success_rate"] == 50.0             # A over A and D
    assert r["call_accuracy"] == 50.0
    assert r["n_request_failed"] == 1 and r["n_harness_errors"] == 2
    assert r["harness_error_samples"] == {"call_judge_outage": 1, "endpoint_failure": 1}
    assert r["n_empty_model_message"] == 1
    assert r["judge_models"] == {"gemini-x": 2} and r["judge_outages"] == 1
    assert r["harness_errors"] == [
        {"error": C.HARNESS_ERROR_KINDS["endpoint_failure"], "samples_unscored": 1,
         "by_error": {"HTTP 503": 1}},
        {"error": "the Gemini call-equivalence judge gave no verdict", "samples_unscored": 1,
         "outages": 1, "unparsed_replies": 0}]
    reason = r["leaderboard_comparable_reason"]
    assert r["leaderboard_comparable"] is False
    assert ("harness errors, not the model's outcome and left out of the metrics: 1 sample(s) "
            "not scored because the model endpoint request failed on all 5 attempts (last "
            "error: HTTP 503 x1); 1 sample(s) not scored because the Gemini call-equivalence "
            "judge gave no verdict (1 failed on every cascade model, 0 replied without an "
            "is_equal verdict)") in reason
    assert "1 sample(s) ended in a model reply with neither content nor tool calls" in reason
    assert r["request_failure_note"] == (
        "1 sample(s) not scored because the model endpoint request failed on all 5 attempts: "
        "harness errors, left out of Success Rate, Call Accuracy and the judge means, which are "
        "over the 2 evaluable sample(s).")
    # B was tried 5 times; every request carried no credential
    assert sum(p["messages"][0]["content"] == "query B" for p in session.payloads) == 5
    assert all("key" not in json.dumps(p).lower() for p in session.payloads)


# --------------------------------------------------------------------------- #
# docs follow the code
# --------------------------------------------------------------------------- #
def test_doc_and_docstring_describe_the_harness_errors(monkeypatch):
    doc = (ROOT / "docs" / "evals" / "complexfuncbench.md").read_text(encoding="utf-8")
    assert doc.isascii()
    flat = " ".join(doc.split())
    assert "A call-equivalence outage counts as not equivalent" not in flat
    assert "a call-equivalence outage counts as not equivalent" not in " ".join(C.__doc__.split())
    assert "Harness errors are never the model's outcome" in C.__doc__
    for key in C.HARNESS_ERROR_KINDS:
        assert f"`{key}`" in doc, key
    assert "`completed_with_errors`" in doc and "`empty_model_message`" in doc
    assert "`HTTP 400 (context window)`" in doc
    results = [_scored("Hotels-0", True),
               {"id": "Hotels-1", "request_failed": True, "harness_error": "endpoint_failure"}]
    r = _run(monkeypatch, C._aggregate(results, 2, False), {})
    generic = {"benchmark_type", "eval_name", "model_name", "thinking", "accuracy", "judge",
               "metric", "leaderboard_comparable", "leaderboard_comparable_reason",
               "temperature", "temperature_source"}
    for key in set(r) - generic:
        assert f"`{key}`" in doc, key
