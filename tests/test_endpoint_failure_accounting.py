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

"""Cross-suite accounting in base.run_eval_suite and the shared helpers around it.

* A request that got no reply from the served endpoint (FAILED, TIMEOUT, OVER_CONTEXT after
  base's retries, or a tool-loop request that failed mid loop) is a harness error: left out of
  the score, counted, named, status completed_with_errors, whatever a batch scorer made of it.
* Nothing scored (harness errors, judge outages, samples left out of the score, in any mix) ->
  status failed, accuracy None with a reason, never 0.
* @k figures count only scored attempts.
* The Gemini judge cascade reports refusals and does not retry them.
* Image labels named in the docs and the provenance equal the code's constants.

Nothing here touches the network, docker or Gemini.
"""

import asyncio
import csv
import functools
import glob
import json
import re
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import base

ROOT = Path(__file__).resolve().parents[1]


def _ok(text, tool_calls=None):
    return base.Reply(text=text, tool_calls=tool_calls, finish_reason="stop", reasoning=None,
                      error=None, completion_tokens=5, stop_reason=None)


def _failed(error):
    return base.Reply(text=None, tool_calls=None, finish_reason=None, reasoning=None, error=error)


def _run(answer_of, samples, **kw):
    """run_eval_suite with a scripted endpoint: answer_of(last user content, messages) -> Reply."""
    async def _fake_send(**k):
        msgs = k["messages"]
        last = next(m["content"] for m in reversed(msgs) if m.get("role") == "user")
        return answer_of(last, msgs)

    with mock.patch.object(base, "_send_single_request", _fake_send):
        return base.run_eval_suite(eval_name="unit_w2q", model_name="m", base_url="http://x",
                                   concurrency=4, samples=samples, max_output_tokens=64, **kw)


def _samples(n, cats=None):
    return [([{"role": "user", "content": f"q{i}"}], "A",
             {"category": cats[i]} if cats else {}) for i in range(n)]


CORRECT = lambda resp, gold: gold in (resp or "")  # noqa: E731


# --------------------------------------------------------------------------- #
# endpoint failures are harness errors
# --------------------------------------------------------------------------- #
def test_an_endpoint_failure_is_left_out_of_the_score_and_named():
    res = _run(lambda q, m: _ok("A") if q == "q0" else _failed("HTTP 500: boom"),
               _samples(2, ["x", "y"]), eval_fn=CORRECT)
    assert res["accuracy"] == 100.0 and res["correct_answers"] == 1
    assert res["total_questions"] == 2 and res["failed_requests"] == 1
    assert res["scoring_excluded"] == 1
    assert res["request_harness_errors"] == {"endpoint_failure": 1}
    assert res["status"] == "completed_with_errors"
    assert "failed_requests 1 (FAILED 1)" in res["status_reason"]
    assert "harness errors left out of the score: 1 request(s)" in res["status_reason"]
    t = res["sample_traces"][1]
    assert t["status"] == "FAILED" and t["error"] == "HTTP 500: boom"
    assert t["scoring_excluded"] is True and t["is_correct"] is False
    assert t["harness_error"] == {"kind": "endpoint_failure", "detail": "HTTP 500: boom"}
    # the failed sample's category is not a 0/1 failure
    assert set(res["category_accuracy"]) == {"x"}


def test_timeouts_and_over_context_prompts_are_harness_errors_too():
    replies = {"q0": _ok("A"), "q1": _failed("timeout after 5s"),
               "q2": _failed(f"{base.OVER_CONTEXT_ERROR}: too long")}
    res = _run(lambda q, m: replies[q], _samples(3), eval_fn=CORRECT)
    assert res["accuracy"] == 100.0
    assert res["timed_out_requests"] == 1 and res["over_context_prompts"] == 1
    assert res["request_harness_errors"] == {"endpoint_timeout": 1, "over_context": 1}
    assert "OVER_CONTEXT 1" in res["status_reason"] and "TIMEOUT 1" in res["status_reason"]


def test_an_empty_reply_stays_the_models_outcome():
    res = _run(lambda q, m: _ok("A") if q == "q0" else _ok(""), _samples(2), eval_fn=CORRECT)
    assert res["accuracy"] == 50.0 and res["scoring_excluded"] == 0
    assert "request_harness_errors" not in res
    assert res["sample_traces"][1]["status"] == "EMPTY_RESPONSE"


def test_a_run_of_only_endpoint_failures_withholds_its_score():
    res = _run(lambda q, m: _failed("HTTP 503: down"), _samples(2), eval_fn=CORRECT)
    assert res["status"] == "failed" and res["accuracy"] is None
    assert res["accuracy_withheld_reason"].startswith("nothing was scored: ")
    assert "endpoint_failure 2" in res["accuracy_withheld_reason"]


def test_a_batch_scorer_cannot_score_an_endpoint_failure():
    async def scorer(traces):
        for t in traces:
            # a scorer that rewrites every status (as ojbench does) and grades a missing reply
            t["status"] = "OK"
            t["is_correct"] = True
    res = _run(lambda q, m: _ok("A") if q == "q0" else _failed("HTTP 502: bad gateway"),
               _samples(2), async_eval_fn=scorer)
    assert res["accuracy"] == 100.0 and res["correct_answers"] == 1
    t = res["sample_traces"][1]
    assert t["is_correct"] is False and t["scoring_excluded"] is True
    assert t["harness_error"]["kind"] == "endpoint_failure"
    assert res["status"] == "completed_with_errors"


# --------------------------------------------------------------------------- #
# nothing scored -> failed, accuracy None
# --------------------------------------------------------------------------- #
def _batch(marks):
    async def scorer(traces):
        for t in traces:
            t["status"] = "OK"
            t["is_correct"] = False
            t.update(marks.get(t["sample_idx"], {}))
    return scorer


@pytest.mark.parametrize("marks", [
    {0: {"scoring_excluded": True}, 1: {"scoring_excluded": True}},
    {0: {"judge_grade": "JUDGE_OUTAGE"}, 1: {"scoring_excluded": True}},
])
def test_nothing_scored_is_failed_with_the_score_withheld(marks):
    res = _run(lambda q, m: _ok("A"), _samples(2), async_eval_fn=_batch(marks))
    assert res["status"] == "failed" and res["accuracy"] is None
    assert res["status_reason"].startswith("nothing was scored: ")
    assert res["accuracy_withheld_reason"] == res["status_reason"]


def test_a_partial_exclusion_is_named_unless_the_protocol_makes_it():
    res = _run(lambda q, m: _ok("A"), _samples(2),
               async_eval_fn=_batch({0: {"is_correct": True}, 1: {"scoring_excluded": True}}))
    assert res["accuracy"] == 100.0 and res["status"] == "completed_with_errors"
    assert "scoring_excluded 1 (left out of the score by the suite)" in res["status_reason"]
    res = _run(lambda q, m: _ok("A"), _samples(2),
               async_eval_fn=_batch({0: {"is_correct": True},
                                     1: {"scoring_excluded": True,
                                         "excluded_by_protocol": "upstream discards it"}}))
    assert res["accuracy"] == 100.0 and res["status"] == "success"
    res = _run(lambda q, m: _ok("A"), _samples(1),
               async_eval_fn=_batch({0: {"scoring_excluded": True,
                                         "excluded_by_protocol": "upstream discards it"}}))
    assert res["status"] == "failed" and res["accuracy"] is None
    assert "all 1 sample(s) left out of the score" in res["status_reason"]


# --------------------------------------------------------------------------- #
# @k over scored attempts only
# --------------------------------------------------------------------------- #
def test_pass_at_k_counts_only_scored_attempts():
    def marks(t):
        src, att = t["source_sample_idx"], t["attempt_idx"]
        if src == 0:
            return {"judge_grade": "JUDGE_OUTAGE"} if att == 0 else {"is_correct": True}
        return {}

    async def scorer(traces):
        for t in traces:
            t["is_correct"] = False
            t["status"] = "OK"
            t.update(marks(t))
    res = _run(lambda q, m: _ok("A"), _samples(2), async_eval_fn=scorer, attempt_count=2,
               supports_attempts=True)
    rep = res["attempts"]
    # sample 0: one scored attempt, correct; sample 1: two wrong attempts. pass^k needs every
    # attempt scored: it is over sample 1 alone, and sample 0 is counted apart.
    assert rep["samples"] == 2 and rep["pass_at_k"] == 50.0 and rep["pass_hat_k"] == 0.0
    assert rep["pass_hat_k_samples"] == 1 and rep["samples_with_fewer_scored_attempts"] == 1
    assert rep["avg_at_k"] == res["accuracy"] == round(100 / 3, 2)


def test_pass_at_k_is_none_without_a_scored_attempt():
    async def scorer(traces):
        for t in traces:
            t["judge_grade"] = "JUDGE_OUTAGE"
    res = _run(lambda q, m: _ok("A"), _samples(1), async_eval_fn=scorer, attempt_count=2,
               supports_attempts=True)
    rep = res["attempts"]
    assert rep["samples"] == 0 and rep["pass_at_k"] is None and rep["pass_hat_k"] is None
    assert rep["avg_at_k"] is None and res["accuracy"] is None and res["status"] == "failed"


# --------------------------------------------------------------------------- #
# tool loops: the transcript on the trace, a failed request mid loop
# --------------------------------------------------------------------------- #
_CALL = [{"id": "c1", "type": "function",
          "function": {"name": "lookup", "arguments": json.dumps({"q": "x"})}}]


def test_a_request_that_fails_mid_loop_is_a_harness_error_and_keeps_its_error():
    def answer(q, msgs):
        if any(m.get("role") == "tool" for m in msgs):
            return _failed("HTTP 500: mid loop")
        return _ok("thinking about it", tool_calls=_CALL)

    async def tool(name, args):
        return f"result for {args['q']}"

    loop = base.ToolLoopProtocol(reference="unit", max_rounds=5, forced_final=False,
                                 house_turn_ceiling=False)
    res = _run(answer, _samples(1), eval_fn=CORRECT, tool_executor=tool, tool_loop=loop)
    t = res["sample_traces"][0]
    assert t["tool_loop_end"] == "request_failed"
    assert t["harness_error"]["kind"] == base.MID_LOOP_FAILURE_KIND
    assert t["error"] == "HTTP 500: mid loop" and t["scoring_excluded"] is True
    assert res["status"] == "failed" and res["accuracy"] is None
    assert "request_failed_mid_loop 1" in res["status_reason"]
    roles = [m["role"] for m in t["tool_loop_transcript"]]
    assert roles == ["assistant", "tool"]
    assert t["tool_loop_transcript"][0]["tool_calls"][0]["function"]["name"] == "lookup"
    assert t["tool_loop_transcript"][1]["content"] == "result for x"


def test_every_tool_loop_trace_keeps_its_transcript():
    def answer(q, msgs):
        if any(m.get("role") == "tool" for m in msgs):
            return _ok("A")
        return _ok("", tool_calls=_CALL)

    async def tool(name, args):
        return "ok"

    res = _run(answer, _samples(2), eval_fn=CORRECT, tool_executor=tool)
    assert res["accuracy"] == 100.0 and res["status"] == "success"
    for t in res["sample_traces"]:
        assert [m["role"] for m in t["tool_loop_transcript"]] == ["assistant", "tool"]


def test_the_executor_arity_is_cached_on_the_function_not_its_id():
    def make(n):
        if n == 3:
            async def ex(name, args, idx):
                return idx
        else:
            async def ex(name, args):
                return None
        return ex

    for n in (3, 2, 3, 2):
        fn = make(n)
        assert base._executor_wants_session(fn) is (n == 3)
        del fn
    assert base._executor_wants_session(functools.partial(lambda a, b, c: 0)) is True
    assert base._executor_wants_session(print) is False     # not weakly referenceable


# --------------------------------------------------------------------------- #
# the judge cascade reports refusals
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, text, reason=None, block=None):
        self.text = text
        self.candidates = [type("C", (), {"finish_reason": reason})()]
        self.prompt_feedback = type("F", (), {"block_reason": block})() if block else None


class _Name:
    def __init__(self, name):
        self.name = name


def _fake_judge(monkeypatch, replies, rounds=3):
    calls = []

    class _Models:
        async def generate_content(self, model, contents, config):
            calls.append((model, config))
            return replies[model]

    client = type("Client", (), {"aio": type("Aio", (), {"models": _Models()})()})()
    monkeypatch.setattr(base, "gemini_api_keys", lambda: ["k"])
    monkeypatch.setattr(base, "_judge_clients", lambda: [client])
    monkeypatch.setattr(base, "_next_judge_client", lambda: client)
    monkeypatch.setattr(base, "_JUDGE_BACKOFF", 0.0)
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", ",".join(replies))
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", str(rounds))
    return calls


def test_a_prompt_every_model_refuses_is_reported_and_not_retried(monkeypatch):
    calls = _fake_judge(monkeypatch, {"m1": _Resp(None, _Name("SAFETY")),
                                      "m2": _Resp(None, None, block=_Name("PROHIBITED_CONTENT"))})
    text, model, info = asyncio.run(base.judge_generate_cascade_detailed("q"))
    assert (text, model) == (None, "JUDGE_OUTAGE")
    assert info["refused"] is True
    assert info["block_reasons"] == ["finish_reason:SAFETY", "prompt_feedback:PROHIBITED_CONTENT"]
    assert [m for m, _c in calls] == ["m1", "m2"]          # one round, no burst retries


def test_an_answer_after_a_refusal_is_returned_with_the_block_recorded(monkeypatch):
    _fake_judge(monkeypatch, {"m1": _Resp(None, _Name("SAFETY")), "m2": _Resp("yes", "STOP")})
    text, model, info = asyncio.run(base.judge_generate_cascade_detailed("q"))
    assert (text, model) == ("yes", "m2")
    assert info["refused"] is False and info["block_reasons"] == ["finish_reason:SAFETY"]


def test_each_cascade_model_can_get_its_own_config(monkeypatch):
    calls = _fake_judge(monkeypatch, {"m1": _Resp(None, _Name("MAX_TOKENS")),
                                      "m2": _Resp("ok", "STOP")})
    text, model, _info = asyncio.run(base.judge_generate_cascade_detailed(
        "q", config=lambda m: {"for": m}))
    assert (text, model) == ("ok", "m2")
    assert calls == [("m1", {"for": "m1"}), ("m2", {"for": "m2"})]


# --------------------------------------------------------------------------- #
# the summary CSV: one category row only when it carries every question
# --------------------------------------------------------------------------- #
def test_the_csv_adds_an_overall_row_when_one_category_is_left(tmp_path, monkeypatch):
    from gbench import cli
    r = {"eval_name": "gdpval", "model_name": "m", "total_questions": 20, "effective_n": 20,
         "correct_answers": 6, "accuracy": 50.0, "status": "completed_with_errors",
         "category_accuracy": {"Jeffery Day": {"correct": 6, "total": 12, "accuracy": 50.0}},
         "sample_traces": []}
    path = cli._save_eval_summary_csv(tmp_path, [r], [], [], {})
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert [row["Subcategory"] for row in rows] == ["OVERALL", "Jeffery Day"]
    assert rows[0]["Questions"] == "20" and rows[1]["Questions"] == "12"
    r["category_accuracy"]["Jeffery Day"]["total"] = 20
    path = cli._save_eval_summary_csv(tmp_path, [r], [], [], {})
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert [row["Subcategory"] for row in rows] == ["Jeffery Day"]


# --------------------------------------------------------------------------- #
# docs and provenance match the code
# --------------------------------------------------------------------------- #
def _label_constants():
    from gbench.runners.eval_suites import gaia2, mcp_bench, toolbench, wildclawbench
    return {gaia2.HARNESS_ERRORS_LABEL: gaia2.HARNESS_ERRORS_VERSION,
            wildclawbench.HARNESS_ERRORS_LABEL: wildclawbench.HARNESS_ERRORS_VERSION,
            mcp_bench.HARNESS_LABEL: mcp_bench.HARNESS_VERSION,
            mcp_bench.REQUEST_POLICY_LABEL: mcp_bench.REQUEST_POLICY_VERSION,
            toolbench.HARNESS_LABEL: toolbench.HARNESS_VERSION}


def test_every_image_label_named_in_docs_and_provenance_equals_the_code():
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC
    texts = {"canonical_sync": json.dumps(CANONICAL_SYNC)}
    for pattern in ("README.md", "NOTICE", "docs/**/*.md", "docker/*", "gbench/docker/*"):
        for f in glob.glob(str(ROOT / pattern), recursive=True):
            if ".bak" in f or not Path(f).is_file():
                continue
            texts[f] = Path(f).read_text(encoding="utf-8", errors="replace")
    seen = set()
    for label, version in _label_constants().items():
        rx = re.compile(re.escape(label) + r'\\?"?\s*[=:]\s*\\?"?(\d+)')
        for where, text in texts.items():
            for m in rx.finditer(text):
                seen.add(label)
                assert m.group(1) == version, (where, label, m.group(1), version)
    assert seen == set(_label_constants())


def test_the_base_accounting_is_documented():
    knobs = re.sub(r"\s+", " ", (ROOT / "docs/evals/env-knobs.md").read_text(encoding="utf-8"))
    for needle in ("Harness errors and withheld scores in `base.run_eval_suite`",
                   "`request_harness_errors`", "`excluded_by_protocol`", "`tool_loop_transcript`",
                   "`accuracy_withheld_reason` names the causes, never 0",
                   "pass@k counts only scored attempts; pass^k counts only the samples with "
                   "every attempt scored (`pass_hat_k_samples`)"):
        assert needle in knobs, needle
    for kind in base.ENDPOINT_FAILURE_KINDS.values():
        assert kind in knobs, kind
    assert base.MID_LOOP_FAILURE_KIND in knobs


def test_the_search_tool_describes_what_it_returns():
    from gbench.runners.eval_suites import gaia, search_tool
    assert search_tool.WEB_SEARCH_TOOL["function"]["description"] == \
        gaia.GROUNDED_SEARCH_DESCRIPTION
    assert "snippet-only" not in search_tool.__doc__


def test_the_provenance_follows_the_endpoint_failure_and_cap_changes():
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as CS
    assert "temperature 1.0" not in CS["cimemories"]["method"]
    assert ("reasoning_effort minimal sent as Gemini thinking_level low"
            in CS["cimemories"]["method"])
    assert "(PutnamBench sets no max_tokens)" not in CS["putnam_formal"]["method"]
    assert "reasoning-model (o1) settings" not in CS["gaia"]["method"]
    assert "snippet-only" not in CS["gaia"]["method"]
    for suite, cap in (("simpleqa", 2048), ("healthbench", 2048), ("wildbench", 1024)):
        assert f"max_tokens={cap} not sent" not in CS[suite]["method"], suite
    assert "max_tokens=1024 sent as max_output_tokens" in CS["wildbench"]["method"]
    assert ("its max_tokens 2048 for the reply, sent as max_output_tokens 2048 + the Gemini "
            "thinking allowance 8192" in CS["simpleqa"]["method"])
    assert CS["browsecomp"]["checksum"].startswith("sha256:7b24471c")
    assert "generativelanguage.googleapis.com (user simulator" not in CS["tau2"]["method"]
    assert "GEMINI_API_KEY is the only provider credential" not in CS["tau3"]["method"]
