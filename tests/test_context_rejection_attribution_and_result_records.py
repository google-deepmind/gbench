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

"""Who owns a context-window rejection, and what a result records about it.

* scicode: a context-window rejection is not retried. When the sub-step's prompt carries the
  model's own earlier code (a runaway reply fed forward, as upstream's gencode.py feeds it), the
  step and the steps after it are failed steps and the problem stays in the score, as upstream's
  fixed denominators count it; a prompt built from the dataset alone that the window cannot hold
  is a harness error (over_context). A reply that reached the window repeating itself gets no
  advice to serve a larger window.
* mmlu records why --max-output-tokens does not apply (evaluate.py sends max_tokens=1).
* docvqa, infographicvqa and textvqa: the category rows and the Correct count say the headline's
  mean score, the binary tally kept apart.
* base sends a sample's identity keys (the ones sample_key reads) and its category to nobody.
* MathArena's last-chance record carries the max_tokens its own request sent.

No model server, judge API or dataset hub is contacted: every request goes to a stub.
"""
from unittest import mock

import pytest

from gbench.runners.eval_suites import base, docvqa, infographicvqa, scicode, textvqa

from test_endpoint_failures_in_suite_headlines import _sc_problem


# --------------------------------------------------------------------------- #
# scicode
# --------------------------------------------------------------------------- #
_OVER = ("BadRequestError: Error code: 400 - {'error': {'message': \"This model's maximum context "
         "length is 262144 tokens. However, your request has 262145 input tokens.\"}}")


def _scicode_run(monkeypatch, problems, reply_for, window=None):
    """run_scicode against stubs: `reply_for(prompt)` is what _complete returns."""
    monkeypatch.setattr(scicode, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(scicode, "resolve_test_data", lambda: ("/nonexistent.h5", "test"))
    monkeypatch.setattr(scicode, "_digest", lambda path: scicode._TEST_DATA_SHA256)
    monkeypatch.setattr(scicode, "_load_scicode_problems", lambda limit=None: problems)
    monkeypatch.setenv("GBENCH_SCICODE_TEMPERATURE", "0")
    monkeypatch.delenv("GBENCH_SCICODE_WITH_BACKGROUND", raising=False)
    asked = []

    def fake_complete(client, model, prompt, temperature, max_tokens, thinking, retries=5):
        asked.append(prompt)
        return reply_for(prompt)
    monkeypatch.setattr(scicode, "_complete", fake_complete)
    monkeypatch.setattr(scicode, "_score_step", lambda *a, **k: {
        "correct": True, "returncode": 0, "stdout_tail": "", "stderr_tail": ""})
    if window:
        base.set_run_knobs(max_model_len=window)
    try:
        return scicode.run_scicode("m", "http://h/v1", concurrency=1), asked
    finally:
        if window:
            base.set_run_knobs(max_model_len=None)


def _ok(prompt):
    return {"text": "```python\ndef g():\n    return 1\n```", "reasoning": "",
            "finish_reason": "stop", "completion_tokens": 5}


def test_scicode_over_context_on_the_models_own_code_is_a_failed_step(monkeypatch):
    problems = [_sc_problem("1", 3), _sc_problem("2", 2)]
    problems[0]["sub_steps"][1]["step_description_prompt"] = "OVERFLOW"

    def reply_for(prompt):
        if "OVERFLOW" in prompt:
            return {"text": None, "error": _OVER, "timed_out": False, "over_context": True,
                    "status": 400}
        return _ok(prompt)
    res, asked = _scicode_run(monkeypatch, problems, reply_for)
    assert len(asked) == 4                  # 1.1, 1.2 (refused); 1.3 is not generated; 2.x
    # the problem stays in the score, failed: 1.2 and 1.3 are failed steps, not harness errors
    assert res["total_problems"] == 2 and res["correct_problems"] == 1
    assert res["accuracy"] == 50.0 and res["effective_n"] == 2
    assert res["total_steps"] == 5 and res["correct_steps"] == 3
    assert res["over_context_steps"] == 1 and res["steps_not_generated_after_over_context"] == 1
    assert res["failed_generations"] == 0 and res["harness_error_problems"] == 0
    assert res["scoring_excluded"] == 0 and res["status"] == "success"
    by_step = {t["step_number"]: t for t in res["sample_traces"]}
    assert by_step["1.2"]["over_context"]["attribution"] == base.REJECTION_MODEL_OUTCOME
    assert by_step["1.2"]["over_context"]["kind"] == "over_context"
    assert "gencode.py" in by_step["1.2"]["over_context"]["upstream"]
    assert not by_step["1.2"].get("harness_error") and not by_step["1.3"].get("scoring_excluded")
    assert by_step["1.3"]["is_correct"] is False and "the model's" in by_step["1.3"]["error"]


def test_scicode_over_context_on_a_dataset_prompt_is_a_harness_error(monkeypatch):
    problems = [_sc_problem("1", 2), _sc_problem("2", 1)]
    problems[0]["sub_steps"][0]["step_description_prompt"] = "OVERFLOW"

    def reply_for(prompt):
        if "OVERFLOW" in prompt:
            return {"text": None, "error": _OVER, "timed_out": False, "over_context": True,
                    "status": 400}
        return _ok(prompt)
    res, _ = _scicode_run(monkeypatch, problems, reply_for)
    by_step = {t["step_number"]: t for t in res["sample_traces"]}
    assert by_step["1.1"]["harness_error"]["kind"] == "over_context"
    assert by_step["1.1"]["scoring_excluded"] is True
    assert res["harness_error_problems"] == 1 and res["over_context_steps"] == 0
    assert res["total_problems"] == 1 and res["status"] == "completed_with_errors"


class _Rejected(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


def _client(errors):
    calls = []

    def create(**kw):
        calls.append(kw)
        raise errors[min(len(calls), len(errors)) - 1]
    client = mock.MagicMock()
    client.chat.completions.create.side_effect = create
    return client, calls


def test_scicode_does_not_retry_a_context_window_rejection(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client, calls = _client([_Rejected(_OVER, 400)])
    out = scicode._complete(client, "m", "p", 0.0, None, False, retries=5)
    assert len(calls) == 1
    assert out["text"] is None and out["over_context"] is True and out["status"] == 400
    # any other failure is retried as before
    client, calls = _client([_Rejected("Service Unavailable", 503)])
    out = scicode._complete(client, "m", "p", 0.0, None, False, retries=3)
    assert len(calls) == 3 and out["text"] is None and not out.get("over_context")


def _cut(text):
    return lambda prompt: {"text": text, "reasoning": "", "finish_reason": "length",
                           "completion_tokens": 9}


def test_scicode_gives_no_larger_window_advice_for_a_repetition_loop(monkeypatch):
    loop = "```python\n" + "# keep the same comment line going on and on and on again here\n" * 200
    res, _ = _scicode_run(monkeypatch, [_sc_problem("1", 1)], _cut(loop), window=262144)
    reason = res["leaderboard_comparable_reason"]
    assert res["replies_cut_off_by_budget"] == 1 and res["replies_cut_off_looping"] == 1
    assert "1 repetition loop(s)" in reason and "serve a larger --max-model-len" not in reason
    # a reply cut off while still writing keeps the advice
    prose = "```python\n" + "\n".join(f"x{i} = {i} * compute_value({i})" for i in range(300))
    res, _ = _scicode_run(monkeypatch, [_sc_problem("1", 1)], _cut(prose), window=262144)
    assert res["replies_cut_off_looping"] == 0
    assert "serve a larger --max-model-len" in res["leaderboard_comparable_reason"]


def test_scicode_is_listed_in_the_rejection_table_and_the_docs():
    assert "scicode" in base.ENDPOINT_REJECTION_ATTRIBUTION
    assert "model's outcome" in base.ENDPOINT_REJECTION_ATTRIBUTION["scicode"]


# --------------------------------------------------------------------------- #
# mmlu
# --------------------------------------------------------------------------- #
def test_mmlu_records_why_max_output_tokens_does_not_apply():
    from aiohttp import web

    from gbench.runners.eval_suites import mmlu
    from test_mmlu_wmdp_logprob_paths import (_CharTok, _mmlu_items, _q, _reply, _serve,
                                              _text_keyed)

    async def handler(req):
        body = await req.json()
        return web.json_response(_reply(_text_keyed("ABCD"[_q(body) % 4])))
    url, stop = _serve(handler)
    try:
        with mock.patch.object(mmlu, "_load_mmlu_items", return_value=_mmlu_items(4)), \
                mock.patch.object(mmlu, "_load_tokenizer", return_value=_CharTok()), \
                mock.patch.object(mmlu, "get_run_knob", return_value=None):
            res = mmlu.run_mmlu("m", url, 2, max_output_tokens=16384)
            plain = mmlu.run_mmlu("m", url, 2)
    finally:
        stop()
    assert res["accuracy"] == 100.0
    note = res["max_output_tokens_note"]
    assert note.startswith("--max-output-tokens 16384 not applied") and "max_tokens=1" in note
    assert "max_output_tokens_note" not in plain


# --------------------------------------------------------------------------- #
# docvqa, infographicvqa, textvqa: the category rows say the headline's metric
# --------------------------------------------------------------------------- #
def _vqa_run(module, run, loader, samples, reply_text):
    async def send(**kw):
        return base.Reply(text=reply_text(kw["messages"]), tool_calls=None,
                          finish_reason="stop", completion_tokens=3)
    with mock.patch.object(base, "_send_single_request", side_effect=send), \
            mock.patch.object(module, loader, lambda **k: samples):
        return getattr(module, run)("m", "http://x/v1", 1)


def _q_text(messages):
    content = messages[-1]["content"]
    return content if isinstance(content, str) else " ".join(
        p.get("text", "") for p in content if isinstance(p, dict))


@pytest.mark.parametrize("module,run,loader", [
    (docvqa, "run_docvqa", "_load_docvqa_samples"),
    (infographicvqa, "run_infographicvqa", "_load_infographicvqa_samples"),
])
def test_anls_suites_category_rows_are_mean_anls(module, run, loader):
    samples = [([{"role": "user", "content": [{"type": "text", "text": "qa1"}]}], ["seven"],
                {"category": "a"}),
               ([{"role": "user", "content": [{"type": "text", "text": "qa2"}]}], ["abcdefgh"],
                {"category": "a"}),
               ([{"role": "user", "content": [{"type": "text", "text": "qb1"}]}], ["four"],
                {"category": "b"})]
    # qa1 exact (ANLS 1), qa2 one edit in 8 (ANLS 0.875), qb1 wrong (ANLS 0)
    answers = {"qa1": "seven", "qa2": "abcdefgx", "qb1": "zzzz"}
    res = _vqa_run(module, run, loader, samples,
                   lambda msgs: next(v for k, v in answers.items() if k in _q_text(msgs)))
    cats = res["category_accuracy"]
    assert cats["a"]["accuracy"] == pytest.approx(93.75) and cats["b"]["accuracy"] == 0.0
    assert cats["a"]["pass_count"] == 2 and cats["a"]["pass_rate_at_0.5"] == 100.0
    assert res["accuracy"] == pytest.approx(62.5)
    assert res["correct_answers"] == pytest.approx(1.875) and res["pass_count"] == 2
    assert "sum of the per-sample scores" in res["correct_answers_note"]


def test_textvqa_category_rows_are_the_soft_vqa_accuracy():
    ten = lambda word: [word] * 10                                            # noqa: E731
    samples = [([{"role": "user", "content": [{"type": "text", "text": "qa1"}]}],
                ["four"] * 2 + ["five"] * 8, {"category": "a"}),
               ([{"role": "user", "content": [{"type": "text", "text": "qb1"}]}], ten("red"),
                {"category": "b"})]
    answers = {"qa1": "four", "qb1": "red"}
    res = _vqa_run(textvqa, "run_textvqa", "_load_textvqa_samples", samples,
                   lambda msgs: next(v for k, v in answers.items() if k in _q_text(msgs)))
    cats = res["category_accuracy"]
    soft_a = next(t["vqa_soft_score"] for t in res["sample_traces"] if t["category"] == "a")
    assert 0.0 < soft_a < 1.0
    assert cats["a"]["accuracy"] == pytest.approx(round(soft_a * 100, 2))
    assert cats["b"]["accuracy"] == 100.0
    assert "binary_pass_rate" in cats["a"] and "pass_count" in cats["a"]
    assert res["correct_answers"] == pytest.approx(round(soft_a + 1.0, 4))


# --------------------------------------------------------------------------- #
# base: a sample's identity keys are bookkeeping, not request fields
# --------------------------------------------------------------------------- #
def test_base_does_not_send_a_samples_identity_keys(monkeypatch):
    seen = []

    async def send(**kw):
        seen.append(dict(kw.get("extra_payload") or {}))
        return base.Reply("a", None, "stop", None, None, 3)
    monkeypatch.setattr(base, "_send_single_request", send)
    samples = [([{"role": "user", "content": "q"}], "a",
                {"category": "c", "question_id": "q-1", "task_id": 7, "name": "n",
                 "logprobs": True})]
    res = base.run_eval_suite("unit", "m", "http://stub/v1", 1, samples,
                              eval_fn=lambda p, g: p == g)
    assert seen and all(not (set(p) & base.SAMPLE_META_BOOKKEEPING_KEYS) for p in seen)
    assert seen[0].get("logprobs") is True                  # a request field is still sent
    assert res["sample_traces"][0]["sample_key"] == "task_id:7"     # sample_key's order


# --------------------------------------------------------------------------- #
# MathArena: the last-chance record carries its own max_tokens
# --------------------------------------------------------------------------- #
def test_the_last_chance_record_carries_the_max_tokens_it_sent(monkeypatch):
    from gbench.runners.eval_suites import matharena_common as MA
    from test_matharena_lcb_isolation_and_bcb_splits import _run_hmmt_offline

    res, sent = _run_hmmt_offline(
        monkeypatch, {"one": "\\boxed{5}", "two": "I think it is seven."},
        lambda msgs: base.Reply(text="\\boxed{7}", tool_calls=None, finish_reason="stop",
                                completion_tokens=2))
    lasts = [k for k in sent if k["messages"][-1]["content"] == MA.LAST_CHANCE_PROMPT]
    records = [t["matharena_last_chance"] for t in res["sample_traces"]
               if t.get("matharena_last_chance")]
    assert records and lasts
    assert {r["max_tokens_sent"] for r in records} == {k["max_output_tokens"] for k in lasts}


# --------------------------------------------------------------------------- #
# healthbench: the grader replies its 2048 cap cut off are counted
# --------------------------------------------------------------------------- #
def test_healthbench_counts_grader_replies_at_the_cap(monkeypatch):
    import json

    from gbench.runners.eval_suites import healthbench

    rubric = [{"criterion": "Is kind", "points": 5, "tags": []},
              {"criterion": "Is rude", "points": -3, "tags": []}]
    samples = [([{"role": "user", "content": "QUESTION-ALPHA"}], json.dumps(rubric),
                {"category": "theme:t", "example_tags": ["theme:t"]}),
               ([{"role": "user", "content": "QUESTION-OMEGA"}], json.dumps(rubric),
                {"category": "theme:t", "example_tags": ["theme:t"]})]

    async def detailed(prompt, config=None):
        if "QUESTION-OMEGA" in prompt:           # the second conversation: an outage at the cap
            return None, "JUDGE_OUTAGE", {"finish_reason": None, "max_tokens_without_text": 2}
        cut = "Is kind" in prompt                # parsed as it arrived, stopped at the cap
        return ('{"explanation": "e", "criteria_met": true}', "g",
                {"finish_reason": "MAX_TOKENS" if cut else "STOP", "max_tokens_without_text": 0})

    async def send(**kw):
        return base.Reply("answer", None, "stop", None, None, 3)
    monkeypatch.setattr(healthbench, "gemini_required_skip", lambda *a, **k: None)
    monkeypatch.setattr(healthbench, "_load_healthbench_samples", lambda limit=None: samples)
    monkeypatch.setattr(healthbench, "simple_evals_samples", lambda s, thinking: s)
    monkeypatch.setattr(healthbench, "judge_generate_cascade_detailed", detailed)
    monkeypatch.setattr(base, "_send_single_request", send)
    res = healthbench.run_healthbench("m", "http://x/v1", 1)
    assert res["judge_replies_at_max_output_tokens"] == 1
    assert res["judge_outages_after_max_output_tokens"] == 1
    graded = next(t for t in res["sample_traces"] if t.get("rubric_grades"))
    assert [g["judge_finish_reason"] for g in graded["rubric_grades"]] == ["MAX_TOKENS", "STOP"]
