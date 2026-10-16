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

"""Judged, vision and long-context suites under the token-limit policy and upstream judge calls.

  * official max-token caps go through base's token-limit policy: a no-think request sends the
    official cap exactly (or none when the harness sends none); a --thinking request uses the
    upstream's reasoning-model settings when it has them and is never capped below
    --max-output-tokens;
  * each judge call sends the parameters its upstream sends (HLE: no temperature and
    max_completion_tokens 4096; CharXiv: top_p 1 and seed 42; the checkers that publish no
    response format are sent none);
  * simpleqa and healthbench share gpqa's simple-evals sampling;
  * the WMT19 cache is locked across processes and a cut-short extraction is redone;
  * the WB-Elo pool is read one model at a time;
  * screenspot reads the official ScreenSpot-v2 release.

No model server, judge API or dataset hub is contacted; every file is written under tmp dirs.
"""
import asyncio
import gc
import hashlib
import io
import json
import os
import re
import tarfile
import threading
import time
import weakref
import zipfile
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import (aa_lcr, base, beam_128k, charxiv, chartqa, coco_caption,
                                        docvqa, frames, gpqa, healthbench, humanitys_last_exam,
                                        i18n_translate, imo_answer_bench, infographicvqa,
                                        loft_scifact, medxpertqa, mmmu_pro, omnidocbench, ruler,
                                        screenspot, screenspot_pointing, simpleqa, textvqa,
                                        wildbench)

def _patch_detailed(module, judge):
    """Patch a suite's judge_generate_cascade_detailed with a (text, model) judge stub; the
    detailed call returns (text, model, info) with nothing cut off at the grader's cap."""
    async def detailed(prompt, config=None):
        text, model = await judge(prompt, config=config)
        return text, model, {"finish_reason": None, "max_tokens_without_text": 0}
    return mock.patch.object(module, "judge_generate_cascade_detailed", detailed)


REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs" / "evals"
SUITES = REPO / "gbench" / "runners" / "eval_suites"
KNOB = 65536


def _simpleqa_detailed(judge):
    """A (text, model) judge stub as simpleqa's detailed cascade call (text, model, info)."""
    async def _detailed(prompt, config=None):
        text, model = await judge(prompt, config=config)
        return text, model, {}
    return _detailed


def _doc(name):
    return (DOCS / f"{name}.md").read_text(encoding="utf-8")


def _src(name):
    return (SUITES / f"{name}.py").read_text(encoding="utf-8")


def _capture(seen, text="ok", finish="stop"):
    """A fake _send_single_request that records every request."""
    async def _send(**kw):
        seen.append(kw)
        return base.Reply(text=text, tool_calls=None, finish_reason=finish, completion_tokens=5)
    return _send


# --------------------------------------------------------------------------- #
# the token-limit policy, per suite and mode
# --------------------------------------------------------------------------- #
#: suite -> (official, no-think cap, --thinking policy, --thinking max_tokens with KNOB)
POLICY = {
    "chartqa": (chartqa.OFFICIAL_SAMPLING, 16, "reasoning_run_budget", KNOB),
    "docvqa": (docvqa.OFFICIAL_SAMPLING, 32, "reasoning_run_budget", KNOB),
    "infographicvqa": (infographicvqa.OFFICIAL_SAMPLING, 32, "reasoning_run_budget", KNOB),
    "textvqa": (textvqa.OFFICIAL_SAMPLING, 1024, "reasoning_run_budget", KNOB),
    "coco_caption": (coco_caption.OFFICIAL_SAMPLING, 64, "reasoning_run_budget", KNOB),
    "charxiv": (charxiv.OFFICIAL_SAMPLING, 1000, "reasoning_run_budget", KNOB),
    "mmmu_pro": (mmmu_pro.OFFICIAL_SAMPLING, 4096, "reasoning_run_budget", KNOB),
    "wildbench": (wildbench.OFFICIAL_SAMPLING, 4096, "reasoning_run_budget", KNOB),
    "omnidocbench": (omnidocbench.OFFICIAL_SAMPLING, 32000, "reasoning_run_budget", KNOB),
    "medxpertqa": (medxpertqa.OFFICIAL_SAMPLING, 2048, "reasoning_run_budget", KNOB),
}


@pytest.mark.parametrize("name", sorted(POLICY))
def test_official_caps_follow_the_token_limit_policy(name):
    official, cap, think_policy, think_sent = POLICY[name]
    base.check_official_sampling_params(name, official)
    plan = base.resolve_token_budget(name, official, thinking=False, max_output_tokens=KNOB)
    assert (plan.policy, plan.max_tokens) == ("official", cap)
    assert plan.note.startswith(f"--max-output-tokens {KNOB} not applied")
    plan = base.resolve_token_budget(name, official, thinking=True, max_output_tokens=KNOB)
    assert (plan.policy, plan.max_tokens) == (think_policy, think_sent)
    # a thinking run is never capped below --max-output-tokens
    assert plan.max_tokens is None or plan.max_tokens >= KNOB
    # the cap is the official one, not counted after the fact
    assert "replies_over_upstream_cap" not in _src(name)
    assert "captions_over_upstream_cap" not in _src(name)


@pytest.mark.parametrize("name,official", [
    ("loft_scifact", loft_scifact.OFFICIAL_SAMPLING),
    ("beam_128k", beam_128k.OFFICIAL_SAMPLING),
    ("screenspot_pointing", screenspot_pointing.OFFICIAL_SAMPLING),
    ("screenspot", screenspot.OFFICIAL_SAMPLING),
])
def test_suites_whose_harness_sends_no_max_tokens_send_none(name, official):
    for thinking in (False, True):
        plan = base.resolve_token_budget(name, official, thinking=thinking, max_output_tokens=KNOB)
        assert plan.policy == "not_sent" and plan.max_tokens is None


def test_aa_lcr_follows_the_methodology_output_caps_per_mode():
    plan = base.resolve_token_budget("aa_lcr", aa_lcr.official_sampling(False), thinking=False,
                                     max_output_tokens=KNOB)
    assert (plan.policy, plan.max_tokens) == ("official", 16384)
    plan = base.resolve_token_budget("aa_lcr", aa_lcr.official_sampling(True), thinking=True,
                                     max_output_tokens=KNOB)
    assert plan.policy == "not_sent"
    assert aa_lcr.official_sampling(True).temperature == aa_lcr.AA_REASONING_TEMPERATURE


def test_the_lmms_eval_suites_keep_the_task_sampling_with_thinking():
    # lmms-eval's enable_thinking path keeps the task's generation_kwargs; the o-series branch
    # matches OpenAI model names only, so no reasoning block is declared
    for module in (chartqa, docvqa, infographicvqa, textvqa, coco_caption):
        official = module.OFFICIAL_SAMPLING
        assert official.reasoning is None and official.for_mode(True) is official
        assert (official.temperature, official.send_temperature) == (0.0, True)
        assert official.max_tokens == module.UPSTREAM_MAX_NEW_TOKENS
        assert "enable_thinking_kwarg" in official.reference and official.reference.isascii()
    assert coco_caption.OFFICIAL_SAMPLING.params == {"top_p": 1.0}


def test_mmmu_pro_no_think_sends_4096_and_thinking_sends_the_flag(monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    samples = [([{"role": "system", "content": mmmu_pro.SYSTEM_MESSAGE},
                 {"role": "user", "content": [{"type": "text", "text": f"t {cat}"}]}],
                {"answer": "A", "options": ["a", "b"], "id": "i"}, {"category": cat})
               for cat in ("standard_10/cot", "standard_10/direct", "vision/cot", "vision/direct")]
    base._RUN_KNOBS["max_output_tokens"] = KNOB
    for thinking, sent_cap in ((False, 4096), (True, KNOB)):
        seen = []
        with mock.patch.object(base, "_send_single_request", side_effect=_capture(seen, "Answer: A")), \
                mock.patch.object(mmmu_pro, "_load_mmmu_pro_samples", lambda *a, **k: samples):
            res = mmmu_pro.run_mmmu_pro("m", "http://x/v1", 1, enable_thinking=thinking)
        assert {k["max_output_tokens"] for k in seen} == {sent_cap}
        # infer_gpt.py sends no temperature; gbench sends its rule's 1.0 / 0.0 and says so
        assert {k["temperature"] for k in seen} == {1.0 if thinking else 0.0}
        assert res["official_sampling"]["max_tokens"] == 4096
        assert res["official_sampling"]["max_tokens_applied"] is (not thinking)
        assert res["leaderboard_comparable"] is False
        assert (f"temperature {1.0 if thinking else 0.0} (infer_gpt.py sends none)"
                in res["leaderboard_comparable_reason"])
        if thinking:
            assert base.REASONING_CAP_NOTE in res["max_output_tokens_note"]
        else:
            assert res["leaderboard_comparable_reason"] == "temperature 0.0 (infer_gpt.py sends none)"


def _charxiv_run(thinking, seen, configs):
    gold = json.dumps({"track": "reasoning", "inst_category": 1, "raw_question": "q?",
                       "answer": "3"})
    samples = [([{"role": "user", "content": [{"type": "text", "text": "q?"}]}], gold,
                {"category": "reasoning"})]

    async def _judge(prompt, config=None):
        configs.append(config)
        return '{"extracted_answer": "3", "score": 1}', "gemini-x"

    with mock.patch.object(base, "_send_single_request", side_effect=_capture(seen, "3")), \
            mock.patch.object(charxiv, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(charxiv, "gemini_api_keys", lambda: ["k"]), \
            mock.patch.object(charxiv, "_load_charxiv_samples", lambda limit=None: samples), \
            mock.patch.object(charxiv, "judge_generate_cascade", _judge):
        return charxiv.run_charxiv("m", "http://x/v1", enable_thinking=thinking)


def test_charxiv_sends_the_chat_client_or_the_reasoning_client_settings():
    base._RUN_KNOBS["max_output_tokens"] = KNOB
    seen, configs = [], []
    res = _charxiv_run(False, seen, configs)
    req = seen[0]
    assert (req["max_output_tokens"], req["temperature"]) == (1000, 0.0)
    assert req["extra_payload"]["top_p"] == 1.0 and req["extra_payload"]["seed"] == 42
    assert res["official_sampling"]["max_tokens_applied"] is True
    seen = []
    res = _charxiv_run(True, seen, configs)
    req = seen[0]
    # gpt.py's settings in both modes (o1.py is keyed on OpenAI model names), except that the
    # temperature is 1.0 and the 1000-token cap is replaced by the run budget on --thinking
    assert (req["max_output_tokens"], req["temperature"]) == (KNOB, 1.0)
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert ("temperature 1.0 (CharXiv's reasoning-model client sends 0.0)"
            in res["leaderboard_comparable_reason"])
    assert req["extra_payload"]["seed"] == 42
    assert "reasoning_settings" not in res["official_sampling"]
    assert "gpt.py" in res["official_sampling"]["reference"]
    # the judge call: JSON, temperature 0, top_p 1, seed 42, no max_output_tokens
    cfg = configs[0]
    assert (cfg.temperature, cfg.top_p, cfg.seed, cfg.response_mime_type) == \
        (0.0, 1.0, 42, "application/json")
    assert cfg.max_output_tokens is None
    assert res["judge_sampling"]["seed"] == 42 and "256" in res["judge_sampling"]["max_output_tokens"]


def _medx_rows(subset):
    if subset != "Text":
        return []
    return [{"id": "Text-0", "question": "Why?\nAnswer Choices: (A) a (B) b", "label": "B",
             "options": {"A": "a", "B": "b"}, "question_type": "Reasoning", "images": []}]


@pytest.mark.parametrize("thinking,knob,cap,temperature", [
    (False, KNOB, 2048, 0.0), (True, KNOB, KNOB, 1.0), (True, 4096, 4096, 1.0)])
def test_medxpertqa_follows_the_api_agent_branch_of_the_mode(thinking, knob, cap, temperature,
                                                             monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    base._RUN_KNOBS["max_output_tokens"] = knob
    seen = []

    async def _send(**kw):
        seen.append(kw)
        text = "The answer is B." if len(kw["messages"]) > 2 else "B looks right."
        return base.Reply(text=text, tool_calls=None, finish_reason="stop", completion_tokens=5)

    with mock.patch("datasets.load_dataset", lambda path, name=None, **k: _medx_rows(name)), \
            mock.patch.object(base, "_send_single_request", side_effect=_send):
        res = medxpertqa.run_medxpertqa("m", "http://x/v1", enable_thinking=thinking)
    assert len(seen) == 2                                     # rationale turn, answer turn
    for req in seen:
        roles = [m["role"] for m in req["messages"]]
        # the chat branch in both modes (the no-system branch is keyed on model names)
        assert "system" in roles
        assert (req["max_output_tokens"], req["temperature"]) == (cap, temperature)
        assert req["extra_payload"]["seed"] == 0
    assert res["accuracy"] == 100.0
    # a --thinking run samples at 1.0, not upstream's 0, and is flagged for it
    assert res["leaderboard_comparable"] is (not thinking), res.get("leaderboard_comparable_reason")
    if thinking:
        assert res["leaderboard_comparable_reason"] == "temperature 1.0 (upstream runs at 0)"


def test_medxpertqa_replies_at_the_official_cap_are_not_truncations(monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    base._RUN_KNOBS["max_output_tokens"] = KNOB

    async def _send(**kw):
        return base.Reply(text="The answer is B.", tool_calls=None, finish_reason="length",
                          completion_tokens=kw["max_output_tokens"])

    with mock.patch("datasets.load_dataset", lambda path, name=None, **k: _medx_rows(name)), \
            mock.patch.object(base, "_send_single_request", side_effect=_send):
        res = medxpertqa.run_medxpertqa("m", "http://x/v1")
    assert res["replies_at_official_max_tokens"] == 2        # one per turn
    assert res["medxpertqa_rationale_turn"]["replies_at_official_max_tokens"] == 1
    assert res["status"] == "success"


def test_aa_lcr_sends_16384_without_thinking_and_the_checker_follows_the_card(monkeypatch):
    samples = [([{"role": "user", "content": "docs + question"}], "42",
                {"category": "Legal", "question": "What?"})]
    configs = []

    async def _judge(prompt, config=None):
        configs.append(config)
        return '```json\n{"verdict": "CORRECT"}\n```', "gemini-x"

    base._RUN_KNOBS["max_output_tokens"] = KNOB
    for thinking, cap in ((False, 16384), (True, None)):
        seen = []
        with mock.patch.object(base, "_send_single_request", side_effect=_capture(seen, "42")), \
                mock.patch.object(aa_lcr, "gemini_required_skip", lambda *a: None), \
                mock.patch.object(aa_lcr, "gemini_api_keys", lambda: ["k"]), \
                mock.patch.object(aa_lcr, "_load_aa_lcr_samples", lambda limit=None: samples), \
                mock.patch.object(aa_lcr, "judge_generate_cascade", _judge), \
                mock.patch("gbench.runners.eval_suites.ojbench.served_model_recommended_sampling",
                           lambda name=None: ({"temperature": 1.0}, "m/generation_config.json")):
            res = aa_lcr.run_aa_lcr("m", "http://x/v1", enable_thinking=thinking)
        assert {k["max_output_tokens"] for k in seen} == {cap}
        assert res["accuracy"] == 100.0
    cfg = configs[0]
    assert cfg.temperature is None and cfg.response_mime_type is None
    assert cfg.system_instruction == aa_lcr.JUDGE_SYSTEM_PROMPT
    assert "not sent" in res["judge_sampling"]["temperature"]


def test_loft_and_screenspot_pointing_send_no_max_tokens(monkeypatch):
    base._RUN_KNOBS["max_output_tokens"] = KNOB
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    seen = []
    loft_samples = [([{"role": "user", "content": "claim"}], {"ids": ["1"], "pids": ["p1"],
                                                               "num_docs": 1}, {"category": "c"})]
    with mock.patch.object(base, "_send_single_request", side_effect=_capture(seen, "[1]")), \
            mock.patch.object(loft_scifact, "_load_loft_scifact_samples",
                              lambda limit=None: loft_samples):
        res = loft_scifact.run_loft_scifact("m", "http://x/v1", 1)
    assert seen[0]["max_output_tokens"] is None
    assert res["official_sampling"]["send_max_tokens"] is False
    assert f"--max-output-tokens {KNOB} not applied" in res["max_output_tokens_note"]
    seen = []
    ss_samples = [([{"role": "user", "content": [{"type": "text", "text": "t"}]}],
                   {"x": 5, "y": 5, "bbox": [0, 0, 0.01, 0.01], "bbox_normalized": True},
                   {"category": "mobile/text"})]
    with mock.patch.object(base, "_send_single_request", side_effect=_capture(seen, "Point: (5, 5)")), \
            mock.patch.object(screenspot_pointing, "_load_screenspot_pointing_samples",
                              lambda **k: ss_samples):
        res = screenspot_pointing.run_screenspot_pointing("m", "http://x/v1", 1)
    # no max_tokens, as the reference sends none; the temperature is gbench's no-think 0.0
    assert (seen[0]["max_output_tokens"], seen[0]["temperature"]) == (None, 0.0)
    assert res["official_sampling"]["send_max_tokens"] is False


def test_beam_reserves_upstreams_answer_room_and_sends_no_max_tokens():
    assert beam_128k._output_reserve(False, KNOB) == beam_128k.UPSTREAM_ANSWER_RESERVE == 2000
    base._RUN_KNOBS["max_output_tokens"] = KNOB
    assert beam_128k._output_reserve(True, None) == KNOB
    base._RUN_KNOBS.pop("max_output_tokens")
    assert beam_128k._output_reserve(True, None) == base.THINKING_MIN_OUTPUT_TOKENS
    assert beam_128k.OFFICIAL_SAMPLING.send_max_tokens is False


def test_beam_judge_sends_temperature_zero_and_no_response_format(monkeypatch):
    configs = []

    async def _judge(prompt, config=None):
        configs.append(config)
        return '{"score": 1.0}', "gemini-x"

    trace = {"response_text": "an answer", "category": "information_extraction",
             "gold_answer": json.dumps({"ideal": "x", "rubric": ["mentions x"]}),
             "status": "PENDING_JUDGE"}
    with mock.patch.object(beam_128k, "gemini_api_keys", lambda: ["k"]), \
            mock.patch.object(beam_128k, "judge_generate_cascade", _judge):
        asyncio.run(beam_128k._make_scorer(__import__("collections").Counter())([trace]))
    assert trace["beam_score"] == 1.0
    cfg = configs[0]
    assert cfg.temperature == 0.0 and cfg.response_mime_type is None
    assert cfg.max_output_tokens is None
    assert "no max_tokens" in beam_128k.judge_sampling_record()["reference"]


def test_ruler_declares_its_caps_in_both_modes_and_preflights_the_policy_budget(monkeypatch):
    seen = []

    def _clamp(messages, budget, *a, **k):
        seen.append(budget)
        return budget, None

    monkeypatch.setattr(ruler, "clamp_to_context", _clamp)
    samples = [([{"role": "user", "content": "x"}], "g",
                {"category": "qa_1@4k", "ruler_length": 10, "max_tokens": 32})]
    base._RUN_KNOBS["max_output_tokens"] = KNOB
    ruler._require_prompts_fit(samples, False, None)
    ruler._require_prompts_fit(samples, True, None)
    assert seen == [32, KNOB]            # the declared cap; with --thinking the run's budget
    src = _src("ruler")
    assert '"max_tokens": cap}' in src and "if not enable_thinking else {}" not in src
    assert "--thinking run: RULER's per-task generation caps" not in src


def test_ruler_thinking_run_records_the_caps_it_did_not_apply():
    seen = []
    samples = [([{"role": "user", "content": "x"}], "g", {"category": "qa_1@4k", "max_tokens": 32})]
    base._RUN_KNOBS["max_output_tokens"] = KNOB
    with mock.patch.object(base, "_send_single_request", side_effect=_capture(seen)):
        res = base.run_eval_suite(eval_name="ruler", model_name="m", base_url="http://x",
                                  concurrency=1, samples=samples, eval_fn=lambda r, g: True,
                                  thinking=True, official_sampling=ruler.OFFICIAL_SAMPLING,
                                  max_output_tokens=128)
    assert seen[0]["max_output_tokens"] == KNOB
    assert res["official_cap_not_applied_to_reasoning"]["declared_max_tokens"] == [32]


# --------------------------------------------------------------------------- #
# judge calls: the upstream's parameters
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("thinking", [False, True])
def test_hle_judge_sends_no_temperature_and_max_completion_tokens(thinking):
    samples = [([{"role": "system", "content": humanitys_last_exam.SYSTEM_PROMPT},
                 {"role": "user", "content": "Q?"}], "42", {"category": "Math"})]
    configs = []

    async def _judge(contents, config=None):
        configs.append(config)
        return json.dumps({"extracted_final_answer": "42", "reasoning": "r", "correct": "yes",
                           "confidence": 90, "strict": True}), "gemini-x"

    with mock.patch.object(base, "_send_single_request", side_effect=_capture([], "Answer: 42")), \
            mock.patch.object(humanitys_last_exam, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(humanitys_last_exam, "_load_humanitys_last_exam_samples",
                              lambda limit=None: samples), \
            mock.patch.object(humanitys_last_exam, "judge_generate_cascade", _judge):
        res = humanitys_last_exam.run_humanitys_last_exam("m", "http://x/v1", 1,
                                                          enable_thinking=thinking)
    cfg = configs[0]
    assert cfg.temperature is None                         # upstream sends none
    assert cfg.max_output_tokens == 4096                   # its max_completion_tokens
    assert cfg.response_mime_type == "application/json" and cfg.response_schema is not None
    assert cfg.system_instruction is None                  # the prompt is the only message
    rec = res["judge_sampling"]
    assert rec["max_output_tokens"] == 4096 and "not sent" in rec["temperature"]
    assert "22ed3074b1e7" in rec["reference"] and res["accuracy"] == 100.0
    assert base.judge_config().temperature == 0.0          # the house default is untouched


def test_hle_docs_state_the_judge_parameters():
    src = _src("humanitys_last_exam")
    assert "pinned at 0.0" not in src
    doc = " ".join(_doc("humanitys_last_exam").split())
    assert "**no temperature** (upstream sends none)" in doc
    assert "`max_output_tokens` 4096" in doc and "`judge_sampling`" in doc


def test_simple_evals_graders_send_no_response_format():
    configs = []
    rubric = json.dumps([{"criterion": "Says hi", "points": 5, "tags": []}])
    samples = [([{"role": "user", "content": "hello"}], rubric,
                {"category": "theme:x", "example_tags": ["theme:x"]})]

    async def _judge(prompt, config=None):
        configs.append(config)
        return '```json\n{"explanation": "ok", "criteria_met": true}\n```', "gemini-x"

    with mock.patch.object(base, "_send_single_request", side_effect=_capture([], "hi")), \
            mock.patch.object(healthbench, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(healthbench, "_load_healthbench_samples", lambda limit=None: samples), \
            _patch_detailed(healthbench, _judge):
        res = healthbench.run_healthbench("m", "http://x/v1", 1)
    cfg = configs[0]
    assert cfg.response_mime_type is None and cfg.temperature == 0.5
    assert res["accuracy"] == 100.0                        # the fenced JSON is still read
    assert "not sent" in res["judge_sampling"]["response_mime_type"]
    doc = _doc("healthbench")
    assert "`response_mime_type=application/json`" not in doc and "no response format" in doc


def test_wildbench_records_its_judge_call():
    rec = wildbench.judge_sampling_record()
    assert rec["temperature"] == 0.0 and rec["response_mime_type"] == "application/json"
    # src/eval.py's max_tokens=1024 is sent as upstream sends it
    # src/eval.py's 1024 for the verdict plus the Gemini judge's thinking allowance
    assert rec["max_output_tokens"] == 1024 + 8192 and "d6b8dcaf377d" in rec["reference"]
    assert 'result["judge_sampling"] = judge_sampling_record()' in _src("wildbench")


def test_judges_without_published_settings_record_the_house_temperature():
    for module in (frames, imo_answer_bench):
        src = _src(module.__name__.rsplit(".", 1)[-1])
        assert 'result["judge_sampling"]' in src and '"temperature": JUDGE_TEMPERATURE' in src
        assert "publishes no" in src


# --------------------------------------------------------------------------- #
# simpleqa / healthbench share gpqa's simple-evals sampling
# --------------------------------------------------------------------------- #
def test_simpleqa_uses_gpqas_samplers():
    assert simpleqa.SIMPLE_EVALS_SAMPLING is gpqa.SIMPLE_EVALS_SAMPLING
    assert healthbench.SIMPLE_EVALS_SAMPLING is gpqa.SIMPLE_EVALS_SAMPLING
    assert simpleqa.simple_evals_messages is gpqa.simple_evals_messages
    src = _src("simpleqa")
    for gone in ("OfficialSampling(", "SIMPLE_EVALS_CHAT_SAMPLER", "SIMPLE_EVALS_REASONING_SAMPLER",
                 "def simple_evals_request", '"max_tokens":'):
        assert gone not in src, gone


def test_simpleqa_no_think_cap_is_the_official_cap_and_is_recorded(monkeypatch):
    samples = [([{"role": "user", "content": "Who?"}], "Ada", {"category": "x"})]

    async def _judge(prompt, config=None):
        return "A", "gemini-x"

    base._RUN_KNOBS["max_output_tokens"] = 16384
    seen = []
    with mock.patch.object(base, "_send_single_request",
                           side_effect=_capture(seen, "Ada", finish="length")), \
            mock.patch.object(simpleqa, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(simpleqa, "_load_simpleqa_samples", lambda limit=None: samples), \
            mock.patch.object(simpleqa, "judge_generate_cascade_detailed",
                              _simpleqa_detailed(_judge)):
        res = simpleqa.run_simpleqa("m", "http://x/v1", 1)
    assert seen[0]["max_output_tokens"] == 2048
    assert res["replies_at_official_max_tokens"] == res["replies_at_sampler_max_tokens"] == 1
    assert res["sampler_max_tokens"] == 2048 and res["status"] == "success"
    assert res["replies_cut_off_by_budget"] == 0          # the official cap is not a cut-off


# --------------------------------------------------------------------------- #
# i18n_translate: one process at a time on the WMT19 cache; redone extractions
# --------------------------------------------------------------------------- #
class _SlowResp(io.BytesIO):
    delay = 0.0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, *a):
        time.sleep(self.delay)
        return super().read(*a)


def _fake_tarball(tmp_path):
    from sacrebleu.dataset import DATASETS
    root = tmp_path / "tgz_src"
    for direction in i18n_translate.DIRECTIONS:
        for rel, kind in zip(DATASETS["wmt19"].langpairs[direction], ("src", "ref")):
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            segs = "".join(f'<seg id="{i + 1}">{kind} {direction} {i}</seg>\n' for i in range(2))
            path.write_text(f'<refset>\n<doc sysid="ref" docid="d" genre="news" '
                            f'origlang="en">\n{segs}</doc>\n</refset>\n', encoding="utf-8")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(root / "sgm", arcname="sgm")
    return buf.getvalue()


@pytest.fixture
def small_wmt19(tmp_path, monkeypatch):
    data = _fake_tarball(tmp_path)
    monkeypatch.setattr(i18n_translate, "TEST_SET_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(i18n_translate, "DIRECTIONS", {d: 2 for d in i18n_translate.DIRECTIONS})
    raw = Path(os.environ["GBENCH_CACHE_DIR"]) / "sacrebleu" / "wmt19" / "raw"
    return data, raw


def test_two_cold_cache_loaders_do_not_race(monkeypatch, small_wmt19):
    data, raw = small_wmt19
    downloads = []

    def _urlopen(url, timeout=None):
        downloads.append(url)
        resp = _SlowResp(data)
        resp.delay = 0.05
        return resp

    monkeypatch.setattr(i18n_translate.urllib.request, "urlopen", _urlopen)
    out = {}

    def _load(tag):
        try:
            out[tag] = len(i18n_translate._load_i18n_translate_samples())
        except Exception as e:                                      # noqa: BLE001
            out[tag] = f"{type(e).__name__}: {e}"

    threads = [threading.Thread(target=_load, args=(t,)) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert out == {"a": 38, "b": 38}
    assert len(downloads) == 1                           # the second waited, then found it
    assert (raw.parent / ".gbench.lock").exists()


def test_an_extraction_cut_short_is_redone(monkeypatch, small_wmt19):
    data, raw = small_wmt19
    monkeypatch.setattr(i18n_translate.urllib.request, "urlopen",
                        lambda *a, **k: _SlowResp(data))
    first = i18n_translate._load_i18n_translate_samples()
    marker = Path(i18n_translate._extracted_marker(str(raw)))
    assert marker.exists()
    # an interrupted extraction: a raw SGML file cut short, no completion marker
    sgm = next(raw.rglob("*.sgm"))
    full = sgm.read_text(encoding="utf-8")
    sgm.write_text(full[:40], encoding="utf-8")
    marker.unlink()
    i18n_translate._wmt19_dataset()                      # extracts again before any text is read
    assert sgm.read_text(encoding="utf-8") == full and marker.exists()
    assert i18n_translate._load_i18n_translate_samples() == first


def test_a_short_sgml_under_a_complete_marker_is_extracted_again(monkeypatch, small_wmt19):
    data, raw = small_wmt19
    monkeypatch.setattr(i18n_translate.urllib.request, "urlopen",
                        lambda *a, **k: _SlowResp(data))
    first = i18n_translate._load_i18n_translate_samples()
    from sacrebleu.dataset import DATASETS
    src_rel = DATASETS["wmt19"].langpairs["de-en"][0]
    sgm = raw / src_rel
    sgm.write_text(sgm.read_text(encoding="utf-8").split("<seg id=\"2\">")[0] + "</doc>\n</refset>\n",
                   encoding="utf-8")
    (raw.parent / "wmt19.de-en.src").write_text("only one line\n", encoding="utf-8")
    assert i18n_translate._load_i18n_translate_samples() == first


def test_i18n_comment_and_doc_describe_the_lock():
    src = _src("i18n_translate")
    assert "sacrebleu itself checks its md5" not in src
    doc = " ".join(_doc("i18n_translate").split())
    assert ".gbench.lock" in doc and "extracted again" in doc and doc.isascii()


# --------------------------------------------------------------------------- #
# WildBench: the WB-Elo pool is read one model at a time before generation
# --------------------------------------------------------------------------- #
def test_pool_outputs_are_read_one_model_at_a_time(monkeypatch):
    reads = []

    def _fetch(config, sha256, purpose):
        reads.append(config)
        return {"s1": f"{config} reply"}

    monkeypatch.setattr(wildbench, "_fetch_model_outputs", _fetch)
    monkeypatch.setattr(wildbench, "WB_ELO_POOL", {"a": "0" * 64, "b": "1" * 64})
    gen = wildbench._load_wb_elo_pool_outputs()
    assert reads == []                                    # nothing is read until asked
    assert next(gen) == ("a", {"s1": "a reply"}) and reads == ["a"]


def test_pool_items_hold_at_most_one_models_replies():
    samples = [([{"role": "user", "content": f"q{i}"}], ["c"], {"session_id": f"s{i}"})
               for i in range(3)]
    refs, alive = {}, []

    class _Replies(dict):
        pass

    def _models():
        for config in ("a", "b", "c", "d"):
            gc.collect()
            alive.append(sum(1 for r in refs.values() if r() is not None))
            replies = _Replies({f"s{i}": f"{config} reply {i}" for i in range(3)})
            refs[config] = weakref.ref(replies)
            yield config, replies
            del replies

    items = wildbench.wb_elo_pool_items(samples, _models())
    gc.collect()
    assert len(items) == 12 and max(alive) == 0          # the previous model was already dropped
    assert all(r() is None for r in refs.values())
    doc = " ".join(_doc("wildbench").split())
    assert "at most one pool model's replies while it builds the items" in doc
    assert "The run never holds the pool's prompts" not in doc


# --------------------------------------------------------------------------- #
# screenspot: the official ScreenSpot-v2 release, the mouse-click tool
# --------------------------------------------------------------------------- #
def _png(fmt="PNG", size=(40, 20)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", size, (0, 0, 255)).save(buf, fmt)
    return buf.getvalue()


@pytest.fixture
def ss_release(tmp_path, monkeypatch):
    files = {"a.png": _png("JPEG"), "b.png": _png(), "c.png": _png()}
    zpath = tmp_path / "screenspotv2_image.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for name, data in files.items():
            zf.writestr(f"screenspotv2_image/{name}", data)
    paths = {"screenspotv2_image.zip": str(zpath)}
    for (platform, (filename, _sha)), name in zip(screenspot_pointing.SPLITS.items(), files):
        p = tmp_path / filename
        p.write_text(json.dumps([{"img_filename": name, "bbox": [4, 2, 20, 10],
                                  "instruction": f"open {name}", "data_type": "icon",
                                  "data_source": platform}]))
        paths[filename] = str(p)
    calls = []

    def _fetch(filename, sha256, **names):
        calls.append(names)
        return paths[filename]

    monkeypatch.setattr(screenspot_pointing, "_fetch", _fetch)
    monkeypatch.setattr(screenspot_pointing, "SPLIT_SIZES", {"mobile": 1, "desktop": 1, "web": 1})
    monkeypatch.delenv("GBENCH_SCREENSPOT_DATA_DIR", raising=False)
    monkeypatch.delenv("SCREENSPOT_PRO_DATA_DIR", raising=False)
    return files, calls


def test_screenspot_reads_the_official_release(ss_release):
    files, calls = ss_release
    samples, label = screenspot._load_screenspot_samples()
    assert label.startswith("OS-Copilot/ScreenSpot-v2@5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c")
    assert calls and all(c == {"eval_name": "screenspot", "docs_url": screenspot.DOCS_URL}
                         for c in calls)
    by_name = {s[2]["img_filename"]: s for s in samples}
    msgs, box, extra = by_name["a.png"]
    text, image = msgs[0]["content"]
    assert text["text"] == 'Click on the UI element: "open a.png".\nUse the computer tool at your disposal.'
    assert image["image_url"]["url"].startswith("data:image/jpeg;base64,")   # the bytes' type
    import base64 as b64
    assert b64.b64decode(image["image_url"]["url"].split(",", 1)[1]) == files["a.png"]
    # [x, y, w, h] pixels -> [x0, y0, x1, y1] / size, then [0, 1000]
    assert box == pytest.approx((100.0, 100.0, 600.0, 600.0))
    assert extra["tools"] == [screenspot.MOUSE_CLICK_TOOL] and extra["category"] == "mobile/icon"
    assert "max_tokens" not in extra and "temperature" not in extra
    assert "HongxinLi" not in _src("screenspot")


def test_screenspot_result_is_scored_by_cell_and_never_comparable(ss_release):
    base._RUN_KNOBS["max_output_tokens"] = KNOB
    seen = []
    reply = 'mouse_click({"button": "left", "x": 300, "y": 300}) mouse_click(button=left, x=300, y=300)'
    with mock.patch.object(base, "_send_single_request", side_effect=_capture(seen, reply)):
        res = screenspot.run_screenspot("m", "http://x/v1", 1)
    assert res["accuracy"] == 100.0
    assert res["screenspot_v2_cells"]["mobile/icon"] == 100.0
    assert res["screenspot_v2_cells"]["mobile/text"] is None
    assert {(k["max_output_tokens"], k["temperature"]) for k in seen} == {(None, 0.0)}
    assert res["leaderboard_comparable"] is False
    assert "gbench mouse-click tool prompt" in res["leaderboard_comparable_reason"]
    assert res["dataset"].startswith("OS-Copilot/ScreenSpot-v2@")


def test_a_named_screenspot_pro_dir_must_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("GBENCH_SCREENSPOT_DATA_DIR", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match=r"^\[INFRA REQUIRED\] screenspot"):
        screenspot._load_screenspot_samples()
    pro = tmp_path / "pro"
    (pro / "annotations").mkdir(parents=True)
    (pro / "images").mkdir()
    (pro / "images" / "s.png").write_bytes(_png(size=(100, 50)))
    (pro / "annotations" / "x.json").write_text(json.dumps(
        [{"instruction": "save", "img_filename": "s.png", "bbox": [10, 5, 20, 10],
          "ui_type": "icon"}]))
    monkeypatch.setenv("GBENCH_SCREENSPOT_DATA_DIR", str(pro))
    samples, label = screenspot._load_screenspot_samples()
    assert label.startswith("ScreenSpot-Pro") and samples[0][1] == (100.0, 100.0, 200.0, 200.0)
    assert samples[0][2]["category"] == "icon"


def test_screenspot_doc_names_the_release():
    doc = " ".join(_doc("screenspot").split())
    for needle in ("OS-Copilot/ScreenSpot-v2", "5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c",
                   "`screenspot_v2_cells`", "`official_sampling`", "`leaderboard_comparable`",
                   "GBENCH_SCREENSPOT_DATA_DIR", "hard error"):
        assert needle in doc, needle
    assert doc.isascii()


# --------------------------------------------------------------------------- #
# docs follow the code
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["chartqa", "docvqa", "infographicvqa", "textvqa", "coco_caption",
                                  "charxiv", "mmmu_pro", "wildbench", "omnidocbench",
                                  "medxpertqa", "aa_lcr"])
def test_docs_describe_the_sent_official_cap(name):
    doc = _doc(name)
    flat = " ".join(doc.split())
    assert doc.isascii()
    assert "`replies_at_official_max_tokens`" in flat, name
    assert "not imposed" not in flat and "`replies_over_upstream_cap`" not in flat, name
    assert re.search(r"`--max-output-tokens`( \(required by the CLI\))? is not applied", flat), name


@pytest.mark.parametrize("name", ["loft_scifact", "beam_128k", "screenspot_pointing",
                                  "screenspot"])
def test_docs_state_that_no_max_tokens_is_sent(name):
    flat = " ".join(_doc(name).split())
    assert "max_output_tokens_note" in flat, name
    assert "no `max_tokens`" in flat or "no request carries a `max_tokens`" in flat, name
