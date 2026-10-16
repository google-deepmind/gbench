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

"""The upstream re-ask (swebench_common.upstream_reask_relay, used by gpqa, causalbench and
longbench_v2_code) never changes what is the model's outcome and what is a harness error:

* only a reply the model ended itself (finish_reason "stop") is re-asked; a reply cut off at the
  token budget (finish_reason "length", content null) is the model's own outcome, never re-asked,
  scored as base scores a truncated reply (wrong) and named (`cut_off_not_reasked`);
* a re-ask chain never runs past the client's per-request deadline, so a request whose re-asks
  are exhausted or cut short is scored on its last reply (`reasks_exhausted`,
  `reasks_cut_short`), never "no reply from the served endpoint";
* `upstream_reask` carries the relay's per-class counts of empty replies.
No model or network: local fake endpoints (the vLLM-shaped _Upstream of the re-ask tests).
"""

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from test_empty_reply_classification_and_upstream_reask import (  # noqa: E402
    IN_THINKING_IDS, TURN_END, _post, _Upstream)

from gbench.runners.eval_suites import base as B  # noqa: E402
from gbench.runners.eval_suites import causalbench as CB  # noqa: E402
from gbench.runners.eval_suites import gpqa as G  # noqa: E402
from gbench.runners.eval_suites import swebench_common as C  # noqa: E402

ROWS = [{"Question": "q?", "Correct Answer": "c", "Incorrect Answer 1": "w1",
         "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3", "Subdomain": "s"}]


@pytest.fixture(autouse=True)
def _fresh_token_cache():
    C._SERVED_TOKEN_IDS.clear()
    yield
    C._SERVED_TOKEN_IDS.clear()


def _gpqa(monkeypatch, script):
    up = _Upstream(script)
    monkeypatch.setattr(G, "_load_gpqa_rows", lambda config: ROWS)
    try:
        res = G._run_simple_evals_gpqa("gpqa", "gpqa_main", 1, "m", up.url, 10, True,
                                       attempt_count=1, max_output_tokens=4096)
    finally:
        up.close()
    return res, up


def test_a_length_truncated_null_reply_is_not_reasked_and_stays_a_scored_wrong_answer(
        monkeypatch):
    def script(req):
        # a thinking reply cut off at the budget inside its reasoning
        return {"content": None, "reasoning": "a long thought ..."}, "length", [100, 5, 6]
    res, up = _gpqa(monkeypatch, script)
    n = G.N_REPEATS
    assert len(up.seen) == n                    # one request each, no re-ask
    reask = res["upstream_reask"]
    assert (reask["requests_reasked"], reask["reasks_sent"], reask["reasks_exhausted"]) == \
        (0, 0, 0)
    assert reask["cut_off_not_reasked"] == n and reask["replies_cut_off_seen"] == n
    # scored, and wrong: the model's own outcome, never a harness error
    assert res["accuracy"] == 0.0 and res["total_questions"] == n
    assert int(res.get("truncated_responses") or 0) == n
    assert int(res.get("timed_out_requests") or 0) == 0
    assert not any(t.get("harness_error") for t in res["sample_traces"])
    reason = res["leaderboard_comparable_reason"]
    assert "no reply from the served endpoint" not in reason
    assert f"{n} reply(ies) cut off at the token budget (finish_reason length) with null " \
           "content were not asked again" in reason


def test_a_stop_null_reply_is_reasked_and_its_class_is_recorded():
    left = {"n": 2}

    def script(req):
        if left["n"]:
            left["n"] -= 1
            return {"content": None, "reasoning": "Answer: B"}, "stop", IN_THINKING_IDS
        return {"content": "Answer: B"}, "stop", [9, TURN_END]
    up = _Upstream(script)
    body = {"model": "m", "messages": [{"role": "user", "content": "q"}], "max_tokens": 4096}
    try:
        with C.upstream_reask_relay("gpqa", up.url, C.reply_content_is_null, 5) as relay:
            got = _post(relay.url, body)
            summary = C.reask_summary(relay, "rule")
    finally:
        up.close()
    assert got["choices"][0]["message"]["content"] == "Answer: B"
    assert len(up.seen) == 3
    assert (summary["requests_reasked"], summary["reasks_sent"], summary["reasks_exhausted"],
            summary["reasks_cut_short"], summary["cut_off_not_reasked"]) == (1, 2, 0, 0, 0)
    # the answer-in-thinking measurement before the re-asks stays visible
    assert (summary["answer_in_thinking_seen"], summary["tool_call_dropped_seen"],
            summary["unclassified_empty_seen"], summary["replies_seen"]) == (2, 0, 0, 3)
    assert summary["answer_in_thinking_rate_seen"] == round(2 / 3, 6)


def test_reask_deadline_is_inside_the_clients_timeout():
    relay = C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1",
                            reask=C.reply_content_is_null, reask_limit=5)
    for cap in (128, 4096, 32768, 65536):
        body = json.dumps({"max_tokens": cap}).encode()
        dl = relay.reask_deadline(body, 1000.0)
        assert 1000.0 < dl < 1000.0 + B.request_timeout_s(cap)
    # no max_tokens: base's shortest timeout, the floor
    assert relay.reask_deadline(b"{}", 0.0) < B.REQUEST_TIMEOUT_FLOOR_S
    assert C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1").reask_deadline(b"{}", 0) is None


def _short_deadlines(monkeypatch, client_timeout_s):
    """base's per-request timeout (the client's) and the relay's deadline margins, small enough
    for a test (the relay derives its deadline from base.request_timeout_s)."""
    monkeypatch.setattr(B, "request_timeout_s", lambda m: client_timeout_s)
    monkeypatch.setattr(C.EndpointRelay, "REASK_DEADLINE_MARGIN_S", 0.3)
    monkeypatch.setattr(C.EndpointRelay, "REASK_MIN_REMAINING_S", 0.1)


def test_a_slow_reask_chain_never_yields_an_exclusion(monkeypatch):
    """Every reply is a stop null reply taking 0.8 s; base's per-request timeout is 3 s, which
    1 + NULL_REPLY_REASKS generations (4.8 s) would overrun. The relay stops re-asking before the
    deadline and passes the last reply on: every request is scored (wrong), counted as
    reasks_exhausted / reasks_cut_short, none is a timeout or 'no reply'."""
    _short_deadlines(monkeypatch, 3)

    def script(req):
        time.sleep(0.8)
        return {"content": None, "reasoning": "Answer: A"}, "stop", IN_THINKING_IDS
    res, up = _gpqa(monkeypatch, script)
    n = G.N_REPEATS
    reask = res["upstream_reask"]
    assert reask["reasks_exhausted"] == n and reask["reasks_cut_short"] == n
    assert reask["requests_reasked"] == n
    assert reask["reasks_sent"] < n * G.NULL_REPLY_REASKS
    # the relay's own deadline is not an endpoint failure
    assert reask["endpoint_failures"] == {}
    assert int(res.get("timed_out_requests") or 0) == 0
    assert not any(t.get("harness_error") for t in res["sample_traces"])
    assert res["accuracy"] == 0.0 and res["total_questions"] == n
    reason = res["leaderboard_comparable_reason"]
    assert "no reply from the served endpoint" not in reason
    assert f"{n} request(s) still came back with null content when gbench stopped re-asking " \
           f"(after up to {G.NULL_REPLY_REASKS} re-asks; {n} of them cut short" in reason
    # every client request ended before base's timeout: no base retry (one first request each)
    firsts = [r for r in up.seen]
    assert len(firsts) == n + reask["reasks_sent"]


def test_a_reask_that_hangs_past_the_deadline_returns_the_last_reply(monkeypatch):
    """A re-ask the endpoint does not answer in time is given up at the deadline (its
    connection closed), and the client gets the reply it already had, well before its own
    timeout."""
    _short_deadlines(monkeypatch, 3)
    calls = {"n": 0}

    def script(req):
        calls["n"] += 1
        if calls["n"] > 1:
            time.sleep(6)                       # past the client's 3 s
        return {"content": None, "reasoning": "x"}, "stop", IN_THINKING_IDS
    up = _Upstream(script)
    body = {"model": "m", "messages": [{"role": "user", "content": "q"}], "max_tokens": 64}
    try:
        with C.upstream_reask_relay("causalbench", up.url, C.reply_content_is_null, 5) as relay:
            t0 = time.monotonic()
            got = _post(relay.url, body)
            took = time.monotonic() - t0
            rec = relay.reask_record()
            failures = relay.record()["upstream_errors"]
    finally:
        up.close()
    assert took < 3
    assert got["choices"][0]["message"]["content"] is None
    assert (rec["reasks_sent"], rec["reasks_exhausted"], rec["reasks_cut_short"]) == (1, 1, 1)
    assert failures == {}


def test_a_reask_the_endpoint_fails_returns_the_last_reply_and_counts_the_failure():
    calls = {"n": 0}

    def script(req):
        calls["n"] += 1
        return {"content": None, "reasoning": "x"}, "stop", IN_THINKING_IDS
    up = _Upstream(script)
    body = {"model": "m", "messages": [{"role": "user", "content": "q"}], "max_tokens": 64}
    try:
        with C.upstream_reask_relay("gpqa", up.url, C.reply_content_is_null, 5) as relay:
            orig = up.srv.RequestHandlerClass.do_POST

            def failing(self):
                if calls["n"] >= 1:
                    self._send(503, {"error": "overloaded"})
                    return
                orig(self)
            up.srv.RequestHandlerClass.do_POST = failing
            got = _post(relay.url, body)
            rec = relay.reask_record()
            failures = relay.record()["upstream_errors"]
    finally:
        up.close()
    assert got["choices"][0]["message"]["content"] is None      # the reply it had, not a 502
    assert (rec["reasks_sent"], rec["reasks_exhausted"], rec["reasks_cut_short"]) == (1, 1, 1)
    assert failures == {"http_503": 1}


def test_causalbench_a_length_truncated_null_reply_is_read_as_base_reads_it(monkeypatch):
    monkeypatch.setattr(CB, "_require_nltk", lambda: None)
    monkeypatch.setattr(CB, "predicted_relation",
                        lambda text: "entailment" if "yes" in str(text or "").lower()
                        else "contradiction")
    up = _Upstream(lambda req: ({"content": None, "reasoning": "hmm"}, "length", [100, 5]))
    samples = [([{"role": "user", "content": f"Question: p{i}? Answer:"}], "entailment",
                {"category": "3", "num_variables": 3, "relation": "x"}) for i in range(2)]
    monkeypatch.setattr(CB, "_load_causalbench_samples", lambda **kw: samples)
    try:
        res = CB.run_causalbench("m", up.url, 2, enable_thinking=True, max_output_tokens=4096)
    finally:
        up.close()
    assert len(up.seen) == 2
    assert res["upstream_reask"]["cut_off_not_reasked"] == 2
    assert not any(t.get("harness_error") for t in res["sample_traces"])
    reason = res["leaderboard_comparable_reason"]
    assert "no reply from the served endpoint" not in reason
    assert "2 reply(ies) cut off at the token budget" in reason


def test_merged_reask_summaries_add_the_counts_and_recompute_the_rate():
    a = {"rule": "r", "reask_limit": 5, "replies_seen": 4, "answer_in_thinking_seen": 1,
         "reasks_cut_short": 1, "endpoint_failures": {"connection": 1}}
    b = {"rule": "r", "reask_limit": 5, "replies_seen": 6, "answer_in_thinking_seen": 4,
         "cut_off_not_reasked": 2, "endpoint_failures": {"connection": 2, "http_503": 1}}
    m = C.merge_reask_summaries([a, b])
    assert (m["replies_seen"], m["answer_in_thinking_seen"], m["reasks_cut_short"],
            m["cut_off_not_reasked"]) == (10, 5, 1, 2)
    assert m["answer_in_thinking_rate_seen"] == 0.5
    assert m["endpoint_failures"] == {"connection": 3, "http_503": 1}


# ------------------------------------------------------------------------------------------
# swe_bench_live: the empty-turn classes are added over the tabs, like answer_in_thinking
# ------------------------------------------------------------------------------------------
def test_swe_bench_live_adds_each_tabs_empty_turn_classes_at_the_top():
    from gbench.runners.eval_suites import swe_bench_live as LV
    tabs = [LV.HEADLINE_SETTING, "other_tab"]
    results = {tabs[0]: {"status": "success", "accuracy": 50.0, "total_questions": 2,
                         "correct_answers": 1, "sample_traces": [],
                         "tool_call_dropped_turns": 0, "unclassified_empty_turns": 1},
               tabs[1]: {"status": "success", "accuracy": 0.0, "total_questions": 1,
                         "correct_answers": 0, "sample_traces": [],
                         "tool_call_dropped_turns": 2, "unclassified_empty_turns": 3}}
    combined = LV.combine_settings(results)
    # not the headline tab's values: every tab's, added
    assert (combined["tool_call_dropped_turns"], combined["unclassified_empty_turns"]) == (2, 4)
    assert combined["settings"][tabs[1]]["tool_call_dropped_turns"] == 2
    assert combined["settings"][tabs[0]]["unclassified_empty_turns"] == 1


def test_a_reply_cut_off_on_a_re_ask_is_counted_too():
    """2026-10-10 round-3 review: only a first reply cut off at the budget was counted in
    cut_off_not_reasked; one cut off on a re-ask was scored but missing from the reason."""
    import inspect
    from gbench.runners.eval_suites import swebench_common as C
    src = inspect.getsource(C)
    assert "if tries == 0 and self._cut_off_match(data):" not in src
    assert "if self._cut_off_match(data):" in src
