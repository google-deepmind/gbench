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

"""Judged, vision and long-context suites: thinking runs, budget cut-offs and shared samplers.

  * the lmms-eval suites (chartqa, docvqa, infographicvqa, textvqa, coco_caption) keep the
    task's sampling with --thinking, as lmms-eval's chat client does for a thinking model on a
    vLLM endpoint (enable_thinking_kwarg): temperature 0 is sent and the task's max_new_tokens,
    written for non-reasoning generation, is not applied (base's reasoning_run_budget policy);
  * a reply cut off by the run's token budget (not at an official cap) is counted, in the
    reasoning channel too, and makes the result not comparable, in every suite of this group;
  * simpleqa and healthbench declare simple-evals' 2048 through gpqa.SIMPLE_EVALS_SAMPLING, as
    gpqa does, so the same sampler is recorded the same way;
  * ruler's pre-flight uses the budget base sends, the caller's included;
  * HLE counts judge replies stopped before their JSON object closed;
  * medxpertqa records why upstream's name-keyed branch is not a reasoning-model setting for a
    served model, which takes the chat branch in both modes.

No model server, judge API or dataset hub is contacted; every file is written under tmp dirs.
"""
import json
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import (base, chartqa, coco_caption, docvqa, gpqa, healthbench,
                                        humanitys_last_exam, infographicvqa, medxpertqa, metrics,
                                        ruler, simpleqa, textvqa, vqa_common)

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs" / "evals"
SUITES = REPO / "gbench" / "runners" / "eval_suites"
KNOB = 65536

#: module, runner, loader, gold, the task's max_new_tokens
LMMS = [
    (chartqa, "run_chartqa", "_load_chartqa_samples", "14", 16),
    (docvqa, "run_docvqa", "_load_docvqa_samples", ["14"], 32),
    (infographicvqa, "run_infographicvqa", "_load_infographicvqa_samples", ["14"], 32),
    (textvqa, "run_textvqa", "_load_textvqa_samples", ["14"] * 10, 1024),
]
LMMS_IDS = [m.__name__.rsplit(".", 1)[-1] for m, *_ in LMMS]

#: every suite of this group that sends model requests through base.run_eval_suite
CUTOFF_SUITES = ["chartqa", "docvqa", "infographicvqa", "textvqa", "coco_caption", "charxiv",
                 "mmmu_pro", "medxpertqa", "wildbench", "omnidocbench", "simpleqa", "healthbench",
                 "humanitys_last_exam", "frames", "imo_answer_bench", "aa_lcr", "beam_128k",
                 "loft_scifact", "screenspot", "screenspot_pointing", "ruler", "i18n_translate",
                 "bundled_detection"]


def _simpleqa_detailed(judge):
    """A (text, model) judge stub as simpleqa's detailed cascade call (text, model, info)."""
    async def _detailed(prompt, config=None):
        text, model = await judge(prompt, config=config)
        return text, model, {}
    return _detailed


def _doc(name):
    return " ".join((DOCS / f"{name}.md").read_text(encoding="utf-8").split())


def _src(name):
    return (SUITES / f"{name}.py").read_text(encoding="utf-8")


def _sender(seen, text="14", finish="stop"):
    async def _send(**kw):
        seen.append(kw)
        return base.Reply(text=text, tool_calls=None, finish_reason=finish, completion_tokens=5,
                          reasoning="thinking" if not text else None)
    return _send


def _run_lmms(module, run, loader, gold, *, thinking, text="14", finish="stop"):
    samples = [([{"role": "user", "content": [{"type": "text", "text": "q1"}]}], gold,
                {"category": "c"})]
    seen = []
    with mock.patch.object(base, "_send_single_request", side_effect=_sender(seen, text, finish)), \
            mock.patch.object(module, loader, lambda **k: samples):
        res = getattr(module, run)("m", "http://x/v1", 1, enable_thinking=thinking)
    return res, seen


# --------------------------------------------------------------------------- #
# lmms-eval suites: lmms-eval's own thinking path for a vLLM chat model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module", [chartqa, docvqa, infographicvqa, textvqa, coco_caption],
                         ids=LMMS_IDS + ["coco_caption"])
def test_lmms_eval_suites_declare_no_reasoning_settings(module):
    official = module.OFFICIAL_SAMPLING
    assert official.reasoning is None
    assert official.for_mode(True) is official
    assert (official.temperature, official.send_temperature) == (0.0, True)
    assert official.max_tokens == module.UPSTREAM_MAX_NEW_TOKENS
    assert "o1/o3/o4/gpt-5" not in official.reference
    plan = base.resolve_token_budget("x", official, thinking=True, max_output_tokens=KNOB)
    assert (plan.policy, plan.max_tokens) == ("reasoning_run_budget", KNOB)
    assert plan.record()["max_tokens_applied"] is False
    assert plan.record()["max_tokens_note"] == base.REASONING_CAP_NOTE


@pytest.mark.parametrize("module,run,loader,gold,cap", LMMS, ids=LMMS_IDS)
def test_lmms_thinking_run_samples_at_one_and_lifts_the_cap(module, run, loader, gold, cap,
                                                             monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    base._RUN_KNOBS["max_output_tokens"] = KNOB
    res, seen = _run_lmms(module, run, loader, gold, thinking=True)
    # an official temperature is not applied to reasoning: 1.0, not the task's 0
    assert [(k["max_output_tokens"], k["temperature"]) for k in seen] == [(KNOB, 1.0)]
    official = res["official_sampling"]
    assert official["max_tokens"] == cap and official["max_tokens_applied"] is False
    assert official["max_tokens_sent"] == KNOB and "reasoning_settings" not in official
    assert official["temperature_matches_official"] is False
    assert res["max_output_tokens_note"].startswith(base.REASONING_CAP_NOTE)
    assert res["leaderboard_comparable"] is False
    assert vqa_common.lmms_temperature_reason(res) == (
        "temperature 1.0 (lmms-eval decodes greedily)")
    assert vqa_common.lmms_temperature_reason(res) in res["leaderboard_comparable_reason"]
    assert res["replies_cut_off_by_budget"] == 0


@pytest.mark.parametrize("module,run,loader,gold,cap", LMMS, ids=LMMS_IDS)
def test_lmms_thinking_run_without_a_knob_is_never_sent_the_task_cap(module, run, loader, gold,
                                                                      cap, monkeypatch):
    # the Python API with no --max-output-tokens: the thinking floor, not 16/32/1024
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    base._RUN_KNOBS.pop("max_output_tokens", None)
    res, seen = _run_lmms(module, run, loader, gold, thinking=True)
    assert [k["max_output_tokens"] for k in seen] == [base.THINKING_MIN_OUTPUT_TOKENS]
    assert seen[0]["max_output_tokens"] > cap
    assert res["official_sampling"]["max_tokens_applied"] is False


@pytest.mark.parametrize("module,run,loader,gold,cap", LMMS, ids=LMMS_IDS)
def test_an_explicit_temperature_zero_thinking_run_stays_comparable(module, run, loader, gold,
                                                                     cap, monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    base._RUN_KNOBS["max_output_tokens"] = KNOB
    base._RUN_KNOBS["temperature"] = 0.0
    res, seen = _run_lmms(module, run, loader, gold, thinking=True)
    assert seen[0]["temperature"] == 0.0
    assert vqa_common.lmms_temperature_reason(res) is None
    assert res["leaderboard_comparable"] is True, res.get("leaderboard_comparable_reason")


def test_coco_caption_thinking_keeps_top_p_and_samples_at_one():
    official = coco_caption.OFFICIAL_SAMPLING.for_mode(True)
    assert (official.temperature, official.params) == (0.0, {"top_p": 1.0})
    assert base.resolve_official_sampling("coco_caption", coco_caption.OFFICIAL_SAMPLING,
                                          thinking=True) == (1.0, "thinking")
    assert base.resolve_official_sampling("coco_caption", coco_caption.OFFICIAL_SAMPLING,
                                          thinking=False) == (0.0, "default")


def test_lmms_eval_sampling_cites_the_thinking_path():
    doc = vqa_common.lmms_eval_sampling.__doc__
    for needle in ("enable_thinking_kwarg", "lines", "190-223", "o1, o3, o4 or gpt-5",
                   "not applied", "non-reasoning"):
        assert needle in " ".join(doc.split()), needle


# --------------------------------------------------------------------------- #
# replies cut off by the run's token budget
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module,run,loader,gold,cap", LMMS, ids=LMMS_IDS)
def test_a_thinking_reply_cut_off_by_the_budget_is_counted_and_not_comparable(
        module, run, loader, gold, cap, monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    base._RUN_KNOBS["max_output_tokens"] = KNOB
    res, _ = _run_lmms(module, run, loader, gold, thinking=True, finish="length")
    assert res["replies_cut_off_by_budget"] == 1 and res["replies_cut_off_in_reasoning"] == 0
    assert res["leaderboard_comparable"] is False
    assert "cut off by this run's token budget" in res["leaderboard_comparable_reason"]
    # empty content after reasoning, stopped at the budget: counted in the reasoning channel
    res, _ = _run_lmms(module, run, loader, gold, thinking=True, text="", finish="length")
    assert res["replies_cut_off_by_budget"] == 1 and res["replies_cut_off_in_reasoning"] == 1
    assert res["leaderboard_comparable"] is False


@pytest.mark.parametrize("module,run,loader,gold,cap", LMMS, ids=LMMS_IDS)
def test_a_no_think_reply_at_the_official_cap_is_not_a_cutoff(module, run, loader, gold, cap,
                                                              monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    base._RUN_KNOBS["max_output_tokens"] = KNOB
    res, seen = _run_lmms(module, run, loader, gold, thinking=False, finish="length")
    assert seen[0]["max_output_tokens"] == cap
    assert res["replies_at_official_max_tokens"] == 1
    assert res["replies_cut_off_by_budget"] == 0
    assert res["leaderboard_comparable"] is True, res.get("leaderboard_comparable_reason")


def test_budget_cutoffs_counts_only_length_stops_off_the_official_caps():
    traces = [
        {"finish_reason": "length", "response_text": "a", "max_tokens_sent": 16},    # official
        {"finish_reason": "length", "response_text": "a", "max_tokens_sent": 9},     # lowered
        {"finish_reason": "length", "response_text": "", "max_tokens_sent": 9},      # reasoning
        {"finish_reason": "length", "response_text": "b", "max_tokens_sent": None},  # window
        {"finish_reason": "stop", "response_text": "c", "max_tokens_sent": 16},
        {"finish_reason": None, "response_text": None, "max_tokens_sent": 16},       # failed
    ]
    result = {"sample_traces": traces, "official_max_tokens_sent": [16]}
    assert metrics.budget_cutoffs(result) == (3, 1)
    reason = metrics.record_budget_cutoffs(result)
    assert result["replies_cut_off_by_budget"] == 3
    assert result["replies_cut_off_in_reasoning"] == 1
    assert reason.startswith("3 repl(ies) cut off by this run's token budget") and reason.isascii()
    # with no official cap (a --thinking run) every length stop is a cut-off
    assert metrics.budget_cutoffs({"sample_traces": traces}) == (4, 1)
    clean = {"sample_traces": traces[4:]}
    assert metrics.record_budget_cutoffs(clean) is None
    assert clean["replies_cut_off_by_budget"] == 0


def test_record_budget_cutoffs_sums_the_parts_of_a_multi_turn_suite():
    turn1 = {"sample_traces": [{"finish_reason": "length", "response_text": "",
                                "max_tokens_sent": KNOB}]}
    turn2 = {"sample_traces": [{"finish_reason": "length", "response_text": "B",
                                "max_tokens_sent": 2048}], "official_max_tokens_sent": [2048]}
    combined = {"sample_traces": []}
    reason = metrics.record_budget_cutoffs(combined, parts=[turn1, turn2])
    assert (combined["replies_cut_off_by_budget"], combined["replies_cut_off_in_reasoning"]) == (1, 1)
    assert reason and "1 of them inside the reasoning channel" in reason


@pytest.mark.parametrize("name", CUTOFF_SUITES)
def test_every_suite_of_the_group_flags_budget_cutoffs(name):
    src = _src(name)
    assert "record_budget_cutoffs(" in src, name
    doc = _doc(name)
    assert "replies_cut_off_by_budget" in doc, name
    assert doc.isascii()


def test_medxpertqa_counts_cutoffs_over_both_turns(monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    base._RUN_KNOBS["max_output_tokens"] = KNOB

    def _rows(path, name=None, **k):
        if name != "Text":
            return []
        return [{"id": "Text-0", "question": "Why?\nAnswer Choices: (A) a (B) b", "label": "B",
                 "options": {"A": "a", "B": "b"}, "question_type": "Reasoning", "images": []}]

    async def _send(**kw):
        if len(kw["messages"]) > 2:
            return base.Reply(text="The answer is B.", tool_calls=None, finish_reason="stop",
                              completion_tokens=5)
        # the rationale turn runs out of the run's budget inside the reasoning channel
        return base.Reply(text="", tool_calls=None, finish_reason="length",
                          completion_tokens=kw["max_output_tokens"], reasoning="...")

    with mock.patch("datasets.load_dataset", _rows), \
            mock.patch.object(base, "_send_single_request", side_effect=_send):
        res = medxpertqa.run_medxpertqa("m", "http://x/v1", enable_thinking=True)
    assert res["replies_cut_off_by_budget"] == 1 and res["replies_cut_off_in_reasoning"] == 1
    assert res["leaderboard_comparable"] is False
    assert "cut off by this run's token budget" in res["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# simpleqa / healthbench: the same simple-evals declaration as gpqa
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module", [simpleqa, healthbench], ids=["simpleqa", "healthbench"])
def test_simple_evals_suites_pass_gpqas_declaration_whole(module):
    assert module.SIMPLE_EVALS_SAMPLING is gpqa.SIMPLE_EVALS_SAMPLING
    src = _src(module.__name__.rsplit(".", 1)[-1])
    assert "official_sampling=SIMPLE_EVALS_SAMPLING" in src
    assert "simple_evals_sampling(" not in src and "simple_evals_meta(" not in src


@pytest.mark.parametrize("thinking", [False, True])
def test_simpleqa_records_the_sampler_as_gpqa_does(thinking, monkeypatch):
    samples = [([{"role": "user", "content": "Who?"}], "Ada", {"category": "x"})]

    async def _judge(prompt, config=None):
        return "A", "gemini-x"

    base._RUN_KNOBS["max_output_tokens"] = 16384
    seen = []
    with mock.patch.object(base, "_send_single_request", side_effect=_sender(seen, "Ada")), \
            mock.patch.object(simpleqa, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(simpleqa, "_load_simpleqa_samples", lambda limit=None: samples), \
            mock.patch.object(simpleqa, "judge_generate_cascade_detailed",
                              _simpleqa_detailed(_judge)):
        res = simpleqa.run_simpleqa("m", "http://x/v1", 1, enable_thinking=thinking)
    plan = base.resolve_token_budget("gpqa", gpqa.SIMPLE_EVALS_SAMPLING, thinking=thinking,
                                     max_output_tokens=16384)
    official = res["official_sampling"]
    for key, value in plan.record().items():
        assert official[key] == value, key
    assert official["reference"] == gpqa.SIMPLE_EVALS_SAMPLING.for_mode(thinking).reference
    assert "max_tokens" not in (seen[0]["extra_payload"] or {})
    if thinking:
        # the chat sampler with --thinking too; the 2048 is replaced by the run's budget
        assert seen[0]["max_output_tokens"] == 16384 and "send_max_tokens" not in official
        assert (official["max_tokens_applied"], official["max_tokens_sent"]) == (False, 16384)
        assert res["max_output_tokens_note"] == plan.note
        assert res["sampler_max_tokens"] is None
    else:
        assert seen[0]["max_output_tokens"] == 2048
        assert (official["max_tokens"], official["max_tokens_applied"]) == (2048, True)
        assert res["max_output_tokens_note"] == plan.note
        assert res["sampler_max_tokens"] == 2048


# --------------------------------------------------------------------------- #
# ruler: the pre-flight uses the budget base sends
# --------------------------------------------------------------------------- #
def _preflight_budgets(monkeypatch, thinking, suite_budget, declared=32):
    seen = []

    def _clamp(messages, budget, *a, **k):
        seen.append(budget)
        return budget, None

    monkeypatch.setattr(ruler, "clamp_to_context", _clamp)
    samples = [([{"role": "user", "content": "x"}], "g",
                {"category": "qa_1@4k", "ruler_length": 10, "max_tokens": declared})]
    ruler._require_prompts_fit(samples, thinking, None, suite_budget)
    return seen


def test_ruler_preflight_uses_the_callers_budget_without_a_knob(monkeypatch):
    base._RUN_KNOBS.pop("max_output_tokens", None)
    assert _preflight_budgets(monkeypatch, True, KNOB) == [KNOB]
    assert _preflight_budgets(monkeypatch, True, None) == [base.THINKING_MIN_OUTPUT_TOKENS]
    assert _preflight_budgets(monkeypatch, True, 128) == [base.THINKING_MIN_OUTPUT_TOKENS]
    assert _preflight_budgets(monkeypatch, False, KNOB) == [32]      # the declared cap


def test_ruler_preflight_replaces_a_larger_declared_cap_on_thinking_runs(monkeypatch):
    # base replaces a declared cap with the run's budget, larger or smaller
    base._RUN_KNOBS["max_output_tokens"] = 64
    assert _preflight_budgets(monkeypatch, True, 128, declared=128) == [64]
    seen = []
    samples = [([{"role": "user", "content": "x"}], "g", {"category": "niah@4k", "max_tokens": 128})]
    with mock.patch.object(base, "_send_single_request", side_effect=_sender(seen)):
        base.run_eval_suite(eval_name="ruler", model_name="m", base_url="http://x", concurrency=1,
                            samples=samples, eval_fn=lambda r, g: True, thinking=True,
                            official_sampling=ruler.OFFICIAL_SAMPLING, max_output_tokens=128)
    assert seen[0]["max_output_tokens"] == 64


def test_ruler_run_passes_the_same_budget_to_the_preflight_and_to_base():
    src = _src("ruler")
    assert "_require_prompts_fit(samples, enable_thinking, kwargs.get(\"extra_payload\"), " \
           "suite_budget)" in src
    assert "max_output_tokens=suite_budget" in src
    assert ruler._run_budget(True, None) == base.THINKING_MIN_OUTPUT_TOKENS


# --------------------------------------------------------------------------- #
# HLE: judge replies stopped before the JSON object closed are counted
# --------------------------------------------------------------------------- #
def test_hle_complete_json_check():
    ok = json.dumps({"extracted_final_answer": "42", "reasoning": "r", "correct": "yes",
                     "confidence": 90, "strict": True})
    assert humanitys_last_exam.judge_reply_is_complete_json(ok)
    assert humanitys_last_exam.judge_reply_is_complete_json(f"```json\n{ok}\n```")
    assert not humanitys_last_exam.judge_reply_is_complete_json(ok[:40])
    assert not humanitys_last_exam.judge_reply_is_complete_json("")
    assert not humanitys_last_exam.judge_reply_is_complete_json("[1, 2]")


def test_hle_counts_judge_replies_cut_short(monkeypatch):
    samples = [([{"role": "system", "content": humanitys_last_exam.SYSTEM_PROMPT},
                 {"role": "user", "content": f"Q{i}?"}], "42", {"category": "Math"})
               for i in range(2)]
    full = json.dumps({"extracted_final_answer": "42", "reasoning": "r", "correct": "yes",
                       "confidence": 90, "strict": True})
    replies = iter([full, full[:30]])

    async def _judge(prompt, config=None):
        return next(replies), "gemini-x"

    with mock.patch.object(base, "_send_single_request", side_effect=_sender([], "42")), \
            mock.patch.object(humanitys_last_exam, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(humanitys_last_exam, "_load_humanitys_last_exam_samples",
                              lambda limit=None: samples), \
            mock.patch.object(humanitys_last_exam, "judge_generate_cascade", _judge):
        res = humanitys_last_exam.run_humanitys_last_exam("m", "http://x/v1", 1)
    assert res["judge_replies_incomplete_json"] == 1
    assert res["judge_failures"] == 1
    assert "stopped before the ExtractedAnswer JSON closed" in res["leaderboard_comparable_reason"]
    assert "judge_replies_incomplete_json" in res["judge_sampling"]["max_output_tokens_note"]
    failed = [t for t in res["sample_traces"] if t.get("judge_grade") == "JUDGE_FAILED"]
    assert failed[0]["judge_reply_complete_json"] is False
    assert "judge_replies_incomplete_json" in _doc("humanitys_last_exam")


# --------------------------------------------------------------------------- #
# medxpertqa: why the name-keyed branch is not a reasoning-model setting for a served model
# --------------------------------------------------------------------------- #
def test_medxpertqa_states_why_the_name_keyed_branch_does_not_apply():
    flat_src = " ".join(medxpertqa.__doc__.split())
    flat_doc = _doc("medxpertqa")
    for text in (flat_src, flat_doc):
        # the upstream evidence: which names APIAgent keys on, and why that is not a setting for
        # reasoning models in general
        for name in ("o1", "o3", "qwq", "qvq", "deepseek-reasoner"):
            assert name in text, name
        assert "keys its settings on the model name" in text
        assert "not reasoning-model settings for a served model" in text
        assert "gemma-4-31b-it" in text and "takes the chat branch" in text
        # the rationale that picked deepseek-reasoner's branch for a served model is gone
        assert "the case a thinking model served by vLLM matches" not in text
        assert "only runnable reasoning-model request" not in text
    reference = medxpertqa.OFFICIAL_SAMPLING.reference
    assert "chat branch" in reference and "not used for a served model" in reference
    assert medxpertqa.OFFICIAL_SAMPLING.reasoning is None


# --------------------------------------------------------------------------- #
# docs follow the code
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["chartqa", "docvqa", "infographicvqa", "textvqa", "coco_caption"])
def test_lmms_docs_describe_the_thinking_path(name):
    doc = _doc(name)
    assert "enable_thinking_kwarg" in doc
    assert "(0; a `--thinking` run samples at 1.0, so it is not comparable)" in doc
    assert "**temperature 1.0**" in doc and "(0, with or without `--thinking`)" not in doc
    assert "reasoning_settings" not in doc and "none with it" not in doc
    assert "`official_sampling.max_tokens_applied: false`" in doc
    src = " ".join(_src(name).split())
    assert "temperature 0" in src and "1.0 with `--thinking`, 0.0 without" in src
    assert "temperature 0 with or without `--thinking`" not in src
    assert "reasoning-model settings: no temperature" not in src


def test_mmmu_pro_doc_drops_the_removed_cap_counter():
    doc = _doc("mmmu_pro")
    assert "replies over the 4,096-token cap" not in doc and "no reply past 4,096" not in doc
