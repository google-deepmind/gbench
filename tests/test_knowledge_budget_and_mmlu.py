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

"""Knowledge suites: mmlu's evaluate.py request, wmdp's lm-eval loglikelihood request, replies
cut off by the budget, client errors, and reasoning-model settings.

* mmlu sends hendrycks/test evaluate.py's request (text prompt, max_tokens=1, logprobs=100,
  temperature=0) without `echo`, reads the letters from the top list of the one generated token
  (-100 for a letter not listed, np.argmax's first index on a tie), stops with infra_required
  naming `--max-logprobs 100` when the server rejects the request, flags a reply whose list is
  shorter than requested while a letter is missing, and reports requests that fail mid-run as
  harness errors;
* wmdp sends lm-eval v0.4.2 local-completions' loglikelihood request (one token-id prompt per
  request, echo=True, max_tokens=0, logprobs=10, temperature=0.0, seed=1234) and sums the
  echoed continuation log-probabilities;
* arc_agi counts attempts the server stopped at the max_tokens budget whatever health base
  gives them afterwards (non_convergent included), for both the comparability flag and the
  status rule, through the real base.run_eval_suite.

* mmlu_redux counts thinking replies cut off at the run's budget (with and without answer text)
  and replies that came back empty after the reasoning, as base.run_eval_suite counts them: they
  change the status and the comparability, and the non-letter warning names the right cause in
  each mode. Without thinking the one-token stops are HELM's protocol;
* mmlu_pro stops at once only when the request call itself raises a client error; a malformed
  streamed chunk fails that try, as upstream asks it again; official_sampling.max_tokens_applied
  says whether the upstream's 32768 is what was sent;
* gpqa, gpqa_diamond, multilingual_mmlu, ifeval and arc_agi flag replies cut off before they
  finished (and ifeval flags failed requests);
* the docs state one rule for upstream reasoning-model settings: the path a served model takes
  (HELM VLLMChatClient, lm-eval local-chat-completions) is the reference, and model-name
  branches for OpenAI's hosted models are not applied, simple-evals' o-series samplers
  included (a served model takes its chat sampler in both modes).

No network and no model server: endpoints are local stubs on ephemeral ports, and base's
request function is faked in process.
"""

import asyncio
import inspect
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from gbench.runners.eval_suites import (arc_agi, base, gpqa, gpqa_diamond, gsm8k, ifeval, mmlu,
                                        mmlu_pro, mmlu_redux, wmdp)

REPO = Path(__file__).resolve().parent.parent


def _doc(suite):
    """The suite's doc with whitespace runs folded, so a phrase may wrap across lines."""
    return " ".join((REPO / "docs" / "evals" / f"{suite}.md").read_text(encoding="utf-8").split())


def _folded(text):
    return " ".join((text or "").split())


class _CharTok:
    """Character-level stand-in tokenizer: one token per character, no BOS."""
    bos_token_id = None

    def encode(self, s):
        return [ord(c) for c in s]


def _serve(handler):
    """aiohttp app with POST /v1/completions on an ephemeral port; returns (base_url, stop)."""
    from aiohttp import web
    loop = asyncio.new_event_loop()
    app = web.Application()
    app.router.add_post("/v1/completions", handler)
    runner = web.AppRunner(app)
    ready, port = threading.Event(), {}

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", 0)
        loop.run_until_complete(site.start())
        port["p"] = site._server.sockets[0].getsockname()[1]
        ready.set()
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    ready.wait(10)

    def stop():
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
    return f"http://127.0.0.1:{port['p']}/v1", stop


# --------------------------------------------------------------------------- #
# mmlu
# --------------------------------------------------------------------------- #
def _mmlu_items(n):
    return [{"subject": "anatomy", "header": mmlu.gen_prompt_header("anatomy"), "shots": [],
             "question": f"Q{i}?\nA. a\nB. b\nC. c\nD. d\nAnswer:", "gold": i % 4}
            for i in range(n)]


def _top100(best, missing=(), size=100):
    """A top-log-probability list of `size` entries: `best` highest, `missing` letters absent."""
    top = {f" {c}": (-0.1 if c == best else -2.0 - i) for i, c in enumerate("ABCD")
           if c not in missing}
    filler = 0
    while len(top) < size:
        top[f"tok{filler}"] = -10.0 - filler
        filler += 1
    return top


def _reply(top, decoy=None):
    tops = ([decoy] if decoy else []) + [top]
    return {"choices": [{"index": 0, "text": " A", "logprobs": {
        "tokens": ["x"] * len(tops), "token_logprobs": [-0.1] * len(tops), "top_logprobs": tops}}]}


def _run_mmlu_against(handler, n=4, **kwargs):
    url, stop = _serve(handler)
    try:
        with mock.patch.object(mmlu, "_load_mmlu_items", return_value=_mmlu_items(n)), \
                mock.patch.object(mmlu, "_load_tokenizer", return_value=_CharTok()), \
                mock.patch.object(mmlu, "get_run_knob", return_value=None):
            return mmlu.run_mmlu("m", url, 2, **kwargs)
    finally:
        stop()


def test_mmlu_sends_evaluate_py_request_without_echo_and_reads_the_generated_tokens_list():
    from aiohttp import web
    seen = []

    async def handler(req):
        body = await req.json()
        seen.append(body)
        q = int(body["prompt"].rsplit("Q", 1)[1].split("?", 1)[0])
        # Right on Q0 and Q1 only. An earlier list that favours the gold letter must be ignored:
        # only top_logprobs[-1] (the generated token's list) is read.
        best = "ABCD"[q % 4] if q < 2 else "D" if q % 4 != 3 else "A"
        return web.json_response(_reply(_top100(best), decoy=_top100("ABCD"[q % 4])))
    res = _run_mmlu_against(handler)
    assert len(seen) == 4
    for body in seen:
        assert set(body) == {"model", "prompt", "max_tokens", "logprobs", "temperature"}
        assert body["max_tokens"] == 1 and body["logprobs"] == 100 and body["temperature"] == 0
        assert isinstance(body["prompt"], str) and body["prompt"].endswith("\nAnswer:")
        assert "echo" not in body and "prompt_logprobs" not in body
    assert res["correct_answers"] == 2 and res["accuracy"] == 50.0
    assert res["status"] == "success" and res["failed_requests"] == 0
    assert res["top_logprobs_requested"] == 100 and res["top_logprobs_returned_max"] == 100
    assert res["letters_missing"] == 0 and res["short_top_logprobs"] == 0
    assert res["temperature"] == 0
    trace = res["sample_traces"][0]
    assert trace["choice_loglikelihoods"] == [-0.1, -3.0, -4.0, -5.0]
    assert trace["predicted"] == "A" and trace["top_logprobs_returned"] == 100
    # --eval-limit is not passed and every question was scored.
    assert res["leaderboard_comparable"] is True and "leaderboard_comparable_reason" not in res


def test_mmlu_missing_letter_counts_minus_100_and_a_tie_takes_the_first_letter():
    lps, missing, size = mmlu._letter_logprobs(_reply(_top100("B", missing="ACD"))["choices"][0])
    assert lps == [-100.0, -0.1, -100.0, -100.0] and missing == ["A", "C", "D"] and size == 100
    lps, missing, _ = mmlu._letter_logprobs(_reply(_top100("x", missing="ABCD"))["choices"][0])
    assert lps == [-100.0] * 4 and mmlu._pick(lps) == 0            # np.argmax: A
    with pytest.raises(ValueError, match="no top_logprobs"):
        mmlu._letter_logprobs({"logprobs": {"token_logprobs": [-0.1]}})
    with pytest.raises(ValueError, match="no top_logprobs"):
        mmlu._letter_logprobs({"logprobs": {"top_logprobs": [None]}})


def test_mmlu_missing_letters_in_full_lists_keep_the_run_comparable():
    from aiohttp import web

    async def handler(req):
        body = await req.json()
        q = int(body["prompt"].rsplit("Q", 1)[1].split("?", 1)[0])
        # Q0: no letter at all among the 100 (predicts A, gold A); Q1: only B listed.
        top = (_top100("x", missing="ABCD") if q == 0 else
               _top100("B", missing="ACD") if q == 1 else _top100("ABCD"[q % 4]))
        return web.json_response(_reply(top))
    res = _run_mmlu_against(handler)
    assert res["letters_missing"] == 7 and res["questions_without_a_letter"] == 1
    assert res["short_top_logprobs"] == 0 and res["correct_answers"] == 4
    assert res["sample_traces"][0]["letters_missing"] == ["A", "B", "C", "D"]
    assert res["leaderboard_comparable"] is True


def test_mmlu_a_short_list_missing_a_letter_makes_the_run_non_comparable():
    from aiohttp import web

    async def handler(req):
        body = await req.json()
        q = int(body["prompt"].rsplit("Q", 1)[1].split("?", 1)[0])
        # A server that returns 20 entries: fine while all four letters are in them.
        top = _top100("B", missing="C", size=20) if q == 1 else _top100("ABCD"[q % 4], size=20)
        return web.json_response(_reply(top))
    res = _run_mmlu_against(handler)
    assert res["short_top_logprobs"] == 1 and res["top_logprobs_returned_max"] == 20
    assert res["leaderboard_comparable"] is False
    assert ("1 question(s) lacked a letter in a top-log-probability list shorter than the 100 "
            "requested") in res["leaderboard_comparable_reason"]


def test_mmlu_server_rejecting_logprobs_100_is_infra_required_naming_the_flag(monkeypatch):
    from aiohttp import web
    monkeypatch.setattr(mmlu, "_RETRIES", 1)

    async def handler(req):     # vLLM's reply at its default --max-logprobs 20
        return web.json_response(
            {"error": {"message": "Requested sample logprobs of 100, which is greater than max "
                                  "allowed: 20", "type": "BadRequestError", "code": 400}},
            status=400)
    with pytest.raises(RuntimeError) as err:
        _run_mmlu_against(handler)
    msg = str(err.value)
    assert "INFRA REQUIRED" in msg and "mmlu" in msg and "--max-logprobs 100" in msg
    assert "greater than max allowed: 20" in msg


def test_mmlu_requests_failing_mid_run_are_harness_errors(monkeypatch):
    from aiohttp import web
    monkeypatch.setattr(mmlu, "_RETRIES", 1)

    async def handler(req):
        body = await req.json()
        q = int(body["prompt"].rsplit("Q", 1)[1].split("?", 1)[0])
        if q == 2:
            return web.Response(status=500, text="EngineCore died")
        return web.json_response(_reply(_top100("ABCD"[q % 4])))
    res = _run_mmlu_against(handler)
    assert res["failed_requests"] == 1 and res["status"] == "completed_with_errors"
    assert res["sample_traces"][2]["status"] == "FAILED"
    assert "HTTP 500" in res["sample_traces"][2]["error"]
    assert res["leaderboard_comparable"] is False
    # a harness error, left out of the score (never a wrong answer)
    assert "1 question(s) got no usable reply from the served endpoint (harness errors, left out " \
           "of the score)" in res["leaderboard_comparable_reason"]
    assert res["sample_traces"][2]["scoring_excluded"] is True
    assert res["scoring_excluded"] == 1 and "harness errors left out" in res["status_reason"]


def test_mmlu_crop_rule_fits_the_prompt_and_the_one_generated_token():
    item = {"subject": "anatomy", "header": "H\n\n", "shots": ["S0\n\n", "S1\n\n", "S2\n\n"],
            "question": "Q?\nAnswer:", "gold": 0}
    enc = mmlu._Encoder(_CharTok())
    two = mmlu._prompt(item, 2)
    assert mmlu._fit_to_window(item, enc, len(two) + 1) == (two, 2)
    assert mmlu._fit_to_window(item, enc, len(two)) == (mmlu._prompt(item, 1), 1)
    assert mmlu._fit_to_window(item, enc, 1)[1] == 0                  # never below zero shots


def test_mmlu_payload_and_docs_state_the_request_and_the_flag():
    assert mmlu._payload("m", "P") == {"model": "m", "prompt": "P", "max_tokens": 1,
                                       "logprobs": 100, "temperature": 0}
    doc = _doc("mmlu")
    assert "`--max-logprobs 100`" in doc and "No `echo`" in doc
    assert "`max_tokens=1`, `logprobs=100` and `temperature=0`" in doc
    assert "counts -100" in doc and "`short_top_logprobs`" in doc
    assert "echo=true`, `logprobs=1`" not in doc and "four token-id prompts" not in doc
    source = _folded(mmlu.__doc__)
    assert "--max-logprobs 100" in source and "leaves `echo` out" in source
    assert "_continuation_logprobs" not in Path(mmlu.__file__).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# wmdp
# --------------------------------------------------------------------------- #
def _wmdp_samples(n=3):
    return [{"subset": "wmdp-bio", "context": f"Q{i}?\nA. w\nB. x\nC. y\nD. z\nAnswer:",
             "gold": i % 4} for i in range(n)]


def _run_wmdp_against(handler, n=3):
    url, stop = _serve(handler)
    try:
        with mock.patch.object(wmdp, "_load_wmdp_samples", return_value=_wmdp_samples(n)), \
                mock.patch.object(wmdp, "_load_tokenizer", return_value=_CharTok()):
            return wmdp.run_wmdp("m", url, concurrency=3)
    finally:
        stop()


def test_wmdp_sends_lm_eval_v042_loglikelihood_requests_one_prompt_each():
    from aiohttp import web
    seen = []

    async def handler(req):
        body = await req.json()
        seen.append(body)
        (ids,) = body["prompt"]
        # max_tokens=0: the reply holds the prompt's tokens only, the first without a value.
        # The model prefers C: its continuation token scores highest.
        lps = [None] + [(-0.1 if t == ord("C") and i == len(ids) - 1 else -2.0)
                        for i, t in enumerate(ids) if i > 0]
        return web.json_response({"choices": [{"index": 0, "logprobs": {"token_logprobs": lps}}]})
    res = _run_wmdp_against(handler)
    assert len(seen) == 3 * 4
    for body in seen:
        assert set(body) == {"model", "prompt", "echo", "max_tokens", "temperature", "logprobs",
                             "seed"}
        assert body["echo"] is True and body["max_tokens"] == 0 and body["logprobs"] == 10
        assert body["temperature"] == 0.0 and body["seed"] == 1234
        assert len(body["prompt"]) == 1 and isinstance(body["prompt"][0][0], int)
    # The probe question goes first, alone: its four continuations, one per request.
    endings = sorted("".join(map(chr, b["prompt"][0])) for b in seen[:4])
    assert [e[-2:] for e in endings] == [" A", " B", " C", " D"]
    assert all(e.startswith("Q0?") for e in endings)
    # " C" is 2 tokens (' ', 'C'): -2.0 for the space, -0.1 for the letter.
    trace = res["sample_traces"][0]
    assert trace["choice_loglikelihoods"] == [-4.0, -4.0, -2.1, -4.0]
    assert trace["predicted"] == "C" and res["correct_answers"] == 1
    assert res["status"] == "success" and res["leaderboard_comparable"] is True


def test_wmdp_server_rejecting_max_tokens_zero_is_infra_required(monkeypatch):
    from aiohttp import web
    monkeypatch.setattr(wmdp, "_RETRIES", 1)

    async def handler(req):
        return web.json_response({"error": {"message": "max_tokens must be at least 1, got 0."}},
                                 status=400)
    with pytest.raises(RuntimeError) as err:
        _run_wmdp_against(handler)
    msg = str(err.value)
    assert "INFRA REQUIRED" in msg and "max_tokens=0" in msg and "logprobs=10" in msg
    assert "max_tokens must be at least 1" in msg


def test_wmdp_requests_failing_mid_run_are_harness_errors(monkeypatch):
    from aiohttp import web
    monkeypatch.setattr(wmdp, "_RETRIES", 1)

    async def handler(req):
        body = await req.json()
        text = "".join(map(chr, body["prompt"][0]))
        if text.startswith("Q1?") and text.endswith(" B"):
            return web.Response(status=500, text="CUDA out of memory")
        lps = [None] + [-1.0] * (len(body["prompt"][0]) - 1)
        return web.json_response({"choices": [{"index": 0, "logprobs": {"token_logprobs": lps}}]})
    res = _run_wmdp_against(handler)
    assert res["failed_requests"] == 1 and res["status"] == "completed_with_errors"
    assert res["sample_traces"][1]["status"] == "FAILED"
    assert "HTTP 500" in res["sample_traces"][1]["error"]
    # a harness error, left out of the score (never a wrong answer)
    assert "1 question(s) got no log-probabilities from the served endpoint (harness errors, " \
           "left out of the score)" in res["leaderboard_comparable_reason"]
    assert res["sample_traces"][1]["scoring_excluded"] is True
    assert res["scoring_excluded"] == 1 and "harness errors left out" in res["status_reason"]


def test_wmdp_docs_state_the_request_and_the_memory_prerequisite():
    doc = _doc("wmdp")
    assert "`echo=True`, `max_tokens=0`, `temperature=0.0`, `logprobs=10` and `seed=1234`" in doc
    assert "`batch_size=1`" in doc and "2,774" in doc and "`--gpu-memory-utilization`" in doc
    assert "`logprobs=1`, `max_tokens=1`" not in doc and "some servers reject" not in doc
    source = _folded(wmdp.__doc__)
    assert "max_tokens=0, logprobs=10" in source and "seed=1234" in source
    assert "some servers reject" not in source


# --------------------------------------------------------------------------- #
# arc_agi: cut-off attempts through the real base.run_eval_suite
# --------------------------------------------------------------------------- #
def _arc_samples(n):
    return [([{"role": "user", "content": f"task {i}"}], [[1, 2]],
             {"category": "grid_transformation", "task_id": f"t{i}", "pair_index": 0})
            for i in range(n)]


def _run_arc(replies_for, n):
    """run_arc_agi on n one-pair tasks; replies_for(task_index) gives the Reply to send."""
    async def fake_send(**kw):
        return replies_for(int(kw["messages"][-1]["content"].split()[1]))
    base.set_run_knobs(max_output_tokens=16384)
    with mock.patch.object(arc_agi, "_load_arc_agi_samples", return_value=_arc_samples(n)), \
            mock.patch.object(base, "_send_single_request", fake_send):
        return arc_agi.run_arc_agi("m", "http://127.0.0.1:9/v1", 1)


def _healthy(_):
    return base.Reply("[[1, 2]]", None, "stop", completion_tokens=100)


def _long_budget_hit():
    """A grid, then distinct words (not a loop) until the budget: 16384 tokens, finish length."""
    return base.Reply("[[1, 2]]\n" + " ".join(f"word{j}" for j in range(3000)), None, "length",
                      completion_tokens=16384)


def test_arc_agi_counts_budget_hits_base_relabels_non_convergent():
    res = _run_arc(lambda i: _long_budget_hit() if i == 0 else _healthy(i), 20)
    hits = [t for t in res["sample_traces"] if t.get("source_sample_idx") == 0]
    # base relabelled both (16384 >= 4x the healthy median of 100), keeping status TRUNCATED.
    assert [(t["health"], t["status"], t["finish_reason"]) for t in hits] == [
        ("non_convergent", "TRUNCATED", "length")] * 2
    assert res["attempts_cut_off"] == 2
    assert res["status"] == "success"                     # 2 of 40: within the 10% rule
    assert res["leaderboard_comparable"] is False
    reason = res["leaderboard_comparable_reason"]
    assert ("2 attempt(s) were cut off at the max_tokens budget (max_tokens_sent [16384]) on "
            "their last try") in reason
    assert "2 of them looping or non_convergent" in reason and "may not have let finish" in reason


def test_arc_agi_status_rule_counts_non_convergent_cut_offs():
    res = _run_arc(lambda i: _long_budget_hit() if i < 3 else _healthy(i), 20)
    assert res["attempts_cut_off"] == 6                   # 6 of 40 attempts: over 10%
    assert {t["health"] for t in res["sample_traces"] if t["status"] == "TRUNCATED"} == {
        "non_convergent"}
    assert res["status"] == "completed_with_errors"
    assert res["leaderboard_comparable"] is False


def test_arc_agi_budget_hit_without_answer_text_is_counted():
    cut = base.Reply("", None, "length", reasoning="step " * 50, completion_tokens=16384)
    res = _run_arc(lambda i: cut if i == 0 else _healthy(i), 20)
    assert res["attempts_cut_off"] == 2 and res["failed_requests"] == 0
    assert "2 of them with no answer text" in res["leaderboard_comparable_reason"]
    # The empty budget hits were asked again (no valid Attempt), as ARCTester retries.
    assert res["arc_agi_report"]["retried_attempts"] == 2


def test_arc_agi_clean_run_stays_comparable_through_base():
    res = _run_arc(_healthy, 20)
    assert res["attempts_cut_off"] == 0 and res["status"] == "success"
    # The loader is faked (no --eval-limit): only the temperature gbench sends (0.0 without
    # thinking; arc-agi-benchmarking sends none) makes the run non-comparable.
    assert res["temperature"] == 0.0 and res["temperature_source"] == "default"
    assert res["leaderboard_comparable_reason"] == (
        "temperature 0.0 (default; arc-agi-benchmarking sends none, so the server default "
        "applies)")


def test_arc_agi_doc_counts_every_budget_hit_and_promises_no_finish():
    doc = _doc("arc_agi")
    assert "attempts whose last try was cut off at the max_tokens budget" in doc
    assert "`non_convergent`" in doc and "`attempts_cut_off`" in doc
    assert "may not finish at any budget" in doc
    assert "would have let them finish" not in doc


# --------------------------------------------------------------------------- #
# mmlu_redux
# --------------------------------------------------------------------------- #
def _redux_items(n):
    return [{"subject": "anatomy", "choices": ["a", "b", "c", "d"], "answer": 1,
             "prompt": f"P{i}"} for i in range(n)]


def _run_redux(monkeypatch, replies, thinking, knob=4096):
    """Run mmlu_redux on len(replies) rows; replies[i] answers row i."""
    async def fake_send(**kw):
        idx = int(kw["messages"][0]["content"][1:])
        return replies[idx]
    monkeypatch.setattr(mmlu_redux, "_send_single_request", fake_send)
    base.set_run_knobs(max_output_tokens=knob)
    with mock.patch.object(mmlu_redux, "_load_mmlu_redux_items",
                           return_value=_redux_items(len(replies))):
        return mmlu_redux.run_mmlu_redux("m", "http://127.0.0.1:9/v1", 1,
                                         enable_thinking=thinking)


def _cut_in_reasoning(knob=4096):
    return base.Reply("", None, "length", reasoning="step " * 50, completion_tokens=knob)


def test_redux_thinking_replies_cut_off_in_reasoning_are_counted(monkeypatch, caplog):
    """Every reply ran out of budget while reasoning."""
    with caplog.at_level(logging.WARNING, logger=mmlu_redux.logger.name):
        res = _run_redux(monkeypatch, [_cut_in_reasoning()] * 4, thinking=True)
    assert res["status"] == "completed_with_errors"
    assert res["truncated_responses"] == 4 and res["replies_cut_off_in_reasoning"] == 4
    assert res["empty_responses"] == 0 and res["failed_requests"] == 0
    assert res["non_letter_replies"] == 0            # counted as budget replies instead
    assert res["response_health"] == {"truncated": 4}
    assert [t["status"] for t in res["sample_traces"]] == ["TRUNCATED"] * 4
    assert res["leaderboard_comparable"] is False
    reason = res["leaderboard_comparable_reason"]
    assert "4 of 4 replies were cut off at the max_tokens budget" in reason
    assert "4 of them with no answer text" in reason
    logs = caplog.text
    assert "hit the output-token budget (max_tokens=4096)" in logs
    assert "with thinking off" not in logs
    assert "official_max_tokens_sent" not in res


def test_redux_thinking_few_cut_off_replies_keep_success_but_not_comparability(monkeypatch):
    """At most 10% cut off: base's status rule keeps success; the run is still flagged."""
    replies = [base.Reply("B", None, "stop", reasoning="r", completion_tokens=50)] * 19
    replies = replies + [base.Reply("The answer", None, "length", reasoning="r",
                                    completion_tokens=4096)]
    res = _run_redux(monkeypatch, replies, thinking=True)
    assert res["status"] == "success" and res["accuracy"] == 95.0
    assert res["truncated_responses"] == 1 and res["replies_cut_off_in_reasoning"] == 0
    assert "1 of 20 replies were cut off" in res["leaderboard_comparable_reason"]
    # Three cut off (15%): completed_with_errors.
    replies = replies[:17] + [_cut_in_reasoning()] * 3
    assert _run_redux(monkeypatch, replies, thinking=True)["status"] == "completed_with_errors"


@pytest.mark.parametrize("reply", [
    base.Reply("", None, "stop", reasoning="it is B", completion_tokens=40),
    base.Reply("<thought>it is B</thought>", None, "stop", completion_tokens=40),
])
def test_redux_thinking_answer_in_thinking_is_the_models_outcome(monkeypatch, reply):
    # The answer written inside the thinking channel (reasoning, or a content holding only a
    # thought block): scored wrong, counted in answer_in_thinking, not a failed request (base's
    # ANSWER_IN_THINKING_POLICY).
    ok = base.Reply("B", None, "stop", reasoning="r", completion_tokens=40)
    res = _run_redux(monkeypatch, [ok, reply], thinking=True)
    assert res["empty_responses"] == 0 and res["failed_requests"] == 0
    assert res["answer_in_thinking"] == 1 and res["answer_in_thinking_rate"] == 0.5
    assert res["response_health"].get("empty_reasoned") == 1
    assert res["sample_traces"][1]["status"] == "EMPTY_RESPONSE"
    assert res["sample_traces"][1]["is_correct"] is False
    assert res["status"] == "success" and res["accuracy"] == 50.0
    assert "no answer text" not in res["leaderboard_comparable_reason"]
    assert res["non_letter_replies"] == 0


def test_redux_thinking_a_reply_with_nothing_is_a_failed_request(monkeypatch):
    ok = base.Reply("B", None, "stop", reasoning="r", completion_tokens=40)
    nothing = base.Reply("", None, "stop", completion_tokens=0)
    res = _run_redux(monkeypatch, [ok, nothing], thinking=True)
    assert res["empty_responses"] == 1 and res["failed_requests"] == 1
    assert res["answer_in_thinking"] == 0
    assert res["status"] == "completed_with_errors" and res["accuracy"] == 50.0
    assert "1 repl(ies) came back with no answer text" in res["leaderboard_comparable_reason"]


def test_redux_thinking_all_empty_is_a_failed_run(monkeypatch):
    nothing = base.Reply("", None, "stop", completion_tokens=0)
    assert _run_redux(monkeypatch, [nothing] * 3, thinking=True)["status"] == "failed"
    # every answer inside the thinking channel: scored (0%), the model's outcome
    in_thinking = base.Reply("", None, "stop", reasoning="r", completion_tokens=40)
    res = _run_redux(monkeypatch, [in_thinking] * 3, thinking=True)
    assert res["status"] == "success" and res["accuracy"] == 0.0
    assert res["answer_in_thinking"] == 3


def test_redux_thinking_inline_thought_cut_off_counts_in_reasoning(monkeypatch):
    res = _run_redux(monkeypatch, [base.Reply("<thought>still going", None, "length",
                                              completion_tokens=4096)], thinking=True)
    assert res["truncated_responses"] == 1 and res["replies_cut_off_in_reasoning"] == 1


def test_redux_thinking_non_letter_warning_names_the_thinking_cause(monkeypatch, caplog):
    replies = [base.Reply("The answer is B.", None, "stop", reasoning="r",
                          completion_tokens=40)] * 3
    with caplog.at_level(logging.WARNING, logger=mmlu_redux.logger.name):
        res = _run_redux(monkeypatch, replies, thinking=True)
    assert res["non_letter_replies"] == 3 and res["status"] == "success"
    logs = caplog.text
    assert "3 of 3 complete replies are not a bare option letter" in logs
    assert "after the reasoning" in logs and "reasoning parser" in logs
    assert "with thinking off" not in logs
    reason = res["leaderboard_comparable_reason"]
    assert "the format of the reply after the reasoning" in reason
    assert "served chat template" not in reason


def test_redux_no_think_one_token_stops_are_the_protocol(monkeypatch, caplog):
    replies = [base.Reply("B", None, "length", completion_tokens=1)] * 3
    replies = replies + [base.Reply("**", None, "length", completion_tokens=1)]
    with caplog.at_level(logging.WARNING, logger=mmlu_redux.logger.name):
        res = _run_redux(monkeypatch, replies, thinking=False, knob=65536)
    assert res["status"] == "success"
    assert res["official_max_tokens_sent"] == [1] and res["replies_at_official_max_tokens"] == 4
    assert res["truncated_responses"] == 4 and "replies_cut_off_in_reasoning" not in res
    assert [t["status"] for t in res["sample_traces"]] == ["OK"] * 4
    # 1 of 4 is not a letter (25%): the no-think template warning, as before.
    assert res["non_letter_replies"] == 1
    assert "with thinking off" in caplog.text
    assert "hit the output-token budget" not in caplog.text
    assert "served chat template" in res["leaderboard_comparable_reason"]
    assert "cut off" not in res["leaderboard_comparable_reason"]


def test_redux_no_think_empty_completion_is_scored_not_failed(monkeypatch):
    """HELM scores an empty one-token completion 0 by exact match; it is not a failure."""
    replies = [base.Reply("B", None, "length", completion_tokens=1)] * 19
    replies = replies + [base.Reply("", None, "stop", completion_tokens=1)]
    res = _run_redux(monkeypatch, replies, thinking=False)
    assert res["empty_responses"] == 1 and res["failed_requests"] == 0
    assert res["non_letter_replies"] == 1 and res["status"] == "success"
    assert res["leaderboard_comparable"] is True


def test_redux_docs_state_the_counts_and_the_reasoning_rule():
    src = _folded(mmlu_redux.__doc__)
    doc = _doc("mmlu_redux")
    for text in (src, doc):
        assert "HELM has no reasoning setting" not in text
        assert "VLLMChatClient" in text and "is_reasoning_model" in text
        assert "lines 286-314" in text
        assert "replies_cut_off_in_reasoning" in text and "empty_responses" in text
        assert "replies_at_official_max_tokens" in text
    assert "`completed_with_errors` when more than 10% of the replies were cut off" in doc


# --------------------------------------------------------------------------- #
# mmlu_pro
# --------------------------------------------------------------------------- #
class _WordEncoding:
    @staticmethod
    def encode(text):
        return text.split()


def _chunk(content=None, finish=None):
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(
        delta=SimpleNamespace(content=content), finish_reason=finish)])


_USAGE = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5,
                                               total_tokens=8), choices=[])
_BAD_CHUNK = SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=None,
                                                                  finish_reason=None)])


def _good_reply():
    return [_chunk("The answer is (B)."), _chunk(finish="stop"), _USAGE]


class _Stream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._chunks:
            yield c

    async def close(self):
        self.closed = True


class _Client:
    def __init__(self, script):
        self.script = script
        self.calls = []
        self.streams = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **request):
        self.calls.append(request)
        out = self.script(len(self.calls) - 1)
        if isinstance(out, BaseException):
            raise out
        stream = _Stream(out)
        self.streams.append(stream)
        return stream

    async def close(self):
        pass


@pytest.fixture
def pro(monkeypatch):
    import tiktoken
    monkeypatch.setattr(mmlu_pro, "START_SPACING_S", 0.0)
    monkeypatch.setattr(mmlu_pro, "ERROR_RETRY_WAIT_S", 0.0)
    monkeypatch.setattr(mmlu_pro, "_token_retry_wait", lambda n: 0.0)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: _WordEncoding())


def _run_pro(monkeypatch, client, n=1, **kwargs):
    samples = [([{"role": "user", "content": f"Question {i}"}], "B",
                {"category": "math", "question_id": 100 + i, "row": i}) for i in range(n)]
    monkeypatch.setattr("openai.AsyncOpenAI", lambda **kw: client)
    with mock.patch.object(mmlu_pro, "_load_mmlu_pro_samples", return_value=samples):
        return mmlu_pro.run_mmlu_pro("m", "http://127.0.0.1:9/v1", 1, **kwargs)


def test_pro_malformed_chunk_is_an_ordinary_failed_try(monkeypatch, pro):
    """Upstream's process_one asks again after any exception of the stream (5 s wait)."""
    # upstream's temperature, set explicitly (gbench's no-think 0.0 is not comparable)
    monkeypatch.setenv("GBENCH_MMLU_PRO_TEMPERATURE", "0.1")
    client = _Client(lambda i: [_BAD_CHUNK, _USAGE] if i == 0 else _good_reply())
    res = _run_pro(monkeypatch, client)
    trace = res["sample_traces"][0]
    assert trace["status"] == "OK" and trace["error_tries"] == 1 and trace["is_correct"]
    assert len(client.calls) == 2 and client.streams[0].closed is True
    assert res["status"] == "success" and res["leaderboard_comparable"] is True


def test_pro_malformed_chunks_on_every_try_end_as_failed_not_infra(monkeypatch, pro):
    monkeypatch.setattr(mmlu_pro, "ERROR_TRIES", 3)
    client = _Client(lambda i: [_BAD_CHUNK, _USAGE])
    res = _run_pro(monkeypatch, client)
    trace = res["sample_traces"][0]
    assert trace["status"] == "FAILED" and trace["error"].startswith("AttributeError")
    assert len(client.calls) == 3 and res["leaderboard_comparable"] is False


@pytest.mark.parametrize("exc", [TypeError("unexpected keyword argument 'stream_options'"),
                                 AttributeError("no attribute 'chat'")])
def test_pro_client_error_from_the_request_call_stops_at_once(monkeypatch, pro, exc):
    client = _Client(lambda i: exc)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED") as info:
        _run_pro(monkeypatch, client, n=3)
    message = str(info.value)
    assert "chat.completions.create raised " + type(exc).__name__ in message
    assert len(client.calls) == 1


def test_pro_hard_error_classification():
    inner = TypeError("bad kwarg")
    try:
        raise mmlu_pro._ClientCallError(inner) from inner
    except mmlu_pro._ClientCallError as wrapped:
        assert "TypeError: bad kwarg" in str(mmlu_pro._hard_error(wrapped))
    # The same exception types, raised outside the request call, are not hard errors.
    assert mmlu_pro._hard_error(AttributeError("'NoneType' object has no attribute")) is None
    assert mmlu_pro._hard_error(TypeError("x")) is None


def test_pro_create_is_the_only_call_wrapped():
    src = inspect.getsource(mmlu_pro._stream_once)
    assert "except _CLIENT_ERRORS as e:" in src and "raise _ClientCallError(e) from e" in src
    assert "isinstance(exc, _CLIENT_ERRORS)" not in inspect.getsource(mmlu_pro._hard_error)


@pytest.mark.parametrize("thinking,knob,sent,applied", [
    (False, 65536, 32768, True),
    (True, None, 32768, True),
    (True, 16384, 32768, True),
    (True, 65536, 65536, False),
])
def test_pro_max_tokens_applied_says_whether_32768_was_sent(monkeypatch, pro, thinking, knob,
                                                            sent, applied):
    if knob:
        base.set_run_knobs(max_output_tokens=knob)
    client = _Client(lambda i: _good_reply())
    res = _run_pro(monkeypatch, client, enable_thinking=thinking)
    official = res["official_sampling"]
    assert [c["max_tokens"] for c in client.calls] == [sent]
    assert official["max_tokens"] == 32768 and official["max_tokens_applied"] is applied
    if thinking:
        assert official["max_tokens_sent"] == sent


def test_pro_docs_match():
    doc = _doc("mmlu_pro")
    assert "when it differs from 32768" not in doc
    assert "`max_tokens_sent` (the value sent)" in doc
    assert "false when a thinking run sent a larger `--max-output-tokens`" in doc
    assert "`chat.completions.create`" in doc and "a malformed chunk" in doc
    src = _folded(mmlu_pro.__doc__)
    assert "chat.completions.create" in src and "a malformed chunk" in src


# --------------------------------------------------------------------------- #
# gpqa family: replies cut off before they finished
# --------------------------------------------------------------------------- #
def _gpqa_rows(n):
    return [{"Question": f"Q{i}?", "Correct Answer": f"right{i}", "Incorrect Answer 1": "w1",
             "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3", "Subdomain": "Physics"}
            for i in range(n)]


@pytest.mark.parametrize("module,run", [(gpqa, "run_gpqa"), (gpqa_diamond, "run_gpqa_diamond")])
def test_gpqa_thinking_reply_at_the_run_budget_is_not_comparable(monkeypatch, module, run):
    async def fake_send(**kw):
        return base.Reply("Answer: A", None, "length", reasoning="r", completion_tokens=100)
    monkeypatch.setattr(base, "_send_single_request", fake_send)
    with mock.patch.object(gpqa, "_load_gpqa_rows", return_value=_gpqa_rows(1)):
        res = getattr(module, run)("m", "http://x/v1", 1, enable_thinking=True)
    assert res["truncated_responses"] == 10
    assert res["leaderboard_comparable"] is False
    # The chat sampler's 2048 is not applied to reasoning: the cut is the run's budget.
    assert ("10 repl(ies) stopped at the run's token budget (--max-output-tokens"
            in res["leaderboard_comparable_reason"])


def test_gpqa_no_think_reply_at_2048_stays_comparable(monkeypatch):
    # the sampler's temperature, set explicitly (gbench's no-think 0.0 is not comparable)
    monkeypatch.setenv("GBENCH_GPQA_TEMPERATURE", "0.5")

    async def fake_send(**kw):
        return base.Reply("Answer: A", None, "length", completion_tokens=2048)
    monkeypatch.setattr(base, "_send_single_request", fake_send)
    with mock.patch.object(gpqa, "_load_gpqa_rows", return_value=_gpqa_rows(1)):
        res = gpqa.run_gpqa("m", "http://x/v1", 1)
    assert res["replies_at_official_max_tokens"] == 10 == res["truncated_responses"]
    assert res["leaderboard_comparable"] is True and res["status"] == "success"


def test_gpqa_flag_names_clamped_no_think_cut_offs():
    result = {"truncated_responses": 3, "replies_at_official_max_tokens": 1,
              "sample_traces": [], "official_sampling": {}}
    gpqa._flag_comparability(result, None, 1, 1, thinking=False)
    assert ("2 repl(ies) stopped at a max_tokens the context clamp set below the chat "
            "sampler's 2048" in result["leaderboard_comparable_reason"])


# --------------------------------------------------------------------------- #
# ifeval and arc_agi
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("counts,needle", [
    # a request with no reply is a harness error, left out of the score
    ({"failed_requests": 2}, "2 request(s) got no reply from the served endpoint"),
    ({"truncated_responses": 3, "max_tokens_sent": [8192]},
     "3 repl(ies) were cut off at the max_tokens budget (max_tokens_sent [8192])"),
])
def test_ifeval_flags_failed_and_cut_off_replies(monkeypatch, counts, needle):
    monkeypatch.setattr(ifeval, "_scorer", lambda: None)
    monkeypatch.setattr(ifeval, "_load_ifeval_samples", lambda limit=None: [])
    monkeypatch.setattr(ifeval, "run_eval_suite",
                        lambda **kw: {"sample_traces": [], "temperature": 0.0, **counts})
    monkeypatch.setattr(ifeval, "_ifeval_report",
                        lambda traces: {"prompts_changed_by_whitespace_trim": 0})
    res = ifeval.run_ifeval("m", "http://x/v1", 1)
    assert res["leaderboard_comparable"] is False
    assert needle in res["leaderboard_comparable_reason"]


def test_ifeval_clean_run_stays_comparable(monkeypatch):
    monkeypatch.setattr(ifeval, "_scorer", lambda: None)
    monkeypatch.setattr(ifeval, "_load_ifeval_samples", lambda limit=None: [])
    monkeypatch.setattr(ifeval, "run_eval_suite",
                        lambda **kw: {"sample_traces": [], "temperature": 0.0,
                                      "failed_requests": 0, "truncated_responses": 0})
    monkeypatch.setattr(ifeval, "_ifeval_report",
                        lambda traces: {"prompts_changed_by_whitespace_trim": 0})
    assert ifeval.run_ifeval("m", "http://x/v1", 1)["leaderboard_comparable"] is True


@pytest.mark.parametrize("health,comparable", [("truncated", False), ("ok", True)])
def test_arc_agi_flags_attempts_cut_off_on_their_last_try(health, comparable):
    samples = [([{"role": "user", "content": "p"}], [[1, 2]],
                {"category": "grid_transformation", "task_id": f"t{i}", "pair_index": 0})
               for i in range(400)]

    def fake_run(**kw):
        traces = [{"sample_idx": 2 * i + a, "source_sample_idx": i,
                   "extra_payload": {"task_id": meta["task_id"], "pair_index": 0},
                   "response_text": "[[1, 2]]", "is_correct": True, "status": "OK",
                   "health": health if (i, a) == (0, 1) else "ok"}
                  for i, (_, g, meta) in enumerate(kw["samples"]) for a in range(2)]
        return {"attempts": {"attempts_per_sample": 2}, "sample_traces": traces,
                "max_tokens_sent": [16384]}
    with mock.patch.object(arc_agi, "_load_arc_agi_samples", return_value=samples), \
            mock.patch.object(arc_agi, "run_eval_suite", side_effect=fake_run):
        res = arc_agi.run_arc_agi("m", "http://x/v1")
    assert res["leaderboard_comparable"] is comparable
    if not comparable:
        assert ("1 attempt(s) were cut off at the max_tokens budget (max_tokens_sent [16384])"
                in res["leaderboard_comparable_reason"])


def test_ifeval_and_arc_agi_docs_name_the_new_causes():
    assert "replies cut off at the budget (`truncated_responses`" in _doc("ifeval")
    assert "(`failed_requests`;" in _doc("ifeval")
    assert "attempts whose last try was cut off at the max_tokens budget" in _doc("arc_agi")


# --------------------------------------------------------------------------- #
# one rule for upstream reasoning-model settings
# --------------------------------------------------------------------------- #
def test_gsm8k_docs_explain_why_lm_evals_openai_branch_is_not_applied():
    for text in (_doc("gsm8k"), _folded(gsm8k.__doc__)):
        assert "OpenAIChatCompletion._create_payload" in text
        assert "local-chat-completions" in text and "has no such branch" in text
        # gbench's rule: 0.0 without --thinking (lm-eval's 0), 1.0 with it, seed 1234 in both
        assert "seed 1234 in both" in text


def test_gpqa_family_docs_explain_why_simple_evals_o_series_samplers_are_not_applied():
    for suite in ("gpqa", "gpqa_diamond", "multilingual_mmlu"):
        doc = _doc(suite)
        assert "simple-evals has no path of its own for a served model" not in doc, suite
        assert "keyed on OpenAI model names" in doc, suite
        assert "a served model takes the chat sampler in both modes" in doc, suite
        assert "never taken for a served model" in doc, suite
        # the capped-reply statement is qualified: only replies at the 2048 itself
        assert "they do not make the run `completed_with_errors`" not in doc, suite
        assert "A reply that stops at the 2048 sent, still writing or repeating itself" in doc
        assert "at anything but the 2048" in doc, suite
    header = _folded(gpqa.__doc__)
    assert "simple-evals has no path of its own for a served model" not in header
    assert "a served model takes the chat sampler in both modes" in header


def test_new_text_is_ascii():
    for path in [REPO / "tests" / "test_knowledge_budget_and_mmlu.py",
                 *(REPO / "gbench" / "runners" / "eval_suites" / f"{m}.py"
                   for m in ("mmlu_redux", "mmlu_pro", "gsm8k", "gpqa", "ifeval", "arc_agi")),
                 *(REPO / "docs" / "evals" / f"{s}.md"
                   for s in ("mmlu_redux", "mmlu_pro", "gsm8k", "gpqa", "gpqa_diamond",
                             "ifeval", "arc_agi"))]:
        path.read_text(encoding="utf-8").encode("ascii")
