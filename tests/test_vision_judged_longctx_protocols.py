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

"""Upstream protocols of the vision, judged and long-context suites.

coco_caption on the Karpathy test split with the seven lmms-eval metrics; markdown-tolerant
answer extraction for whole bold/italic label lines; lmms-eval prompts, pinned datasets and
comparability reasons for chartqa / textvqa / docvqa / infographicvqa; healthbench no-response
grading failures counted once; imo_answer_bench status on a judge outage; medxpertqa status by
question; WildBench v2 with WB_score.task_macro; RULER pre-flight by token count; CharXiv pinned
revision; toolchains doc.
"""
import asyncio
import json
import shutil
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import (base, chartqa, charxiv, coco_caption, docvqa,
                                        healthbench, imo_answer_bench, infographicvqa,
                                        medxpertqa, ruler, textvqa, wildbench)
from gbench.runners.eval_suites.vqa_common import anls_score, extract_short_answer

def _patch_detailed(module, judge):
    """Patch a suite's judge_generate_cascade_detailed with a (text, model) judge stub; the
    detailed call returns (text, model, info) with nothing cut off at the grader's cap."""
    async def detailed(prompt, config=None):
        text, model = await judge(prompt, config=config)
        return text, model, {"finish_reason": None, "max_tokens_without_text": 0}
    return mock.patch.object(module, "judge_generate_cascade_detailed", detailed)


def _as_detailed(judge):
    """A (text, model) judge stub as the cascade's detailed call, (text, model, info), which
    wildbench uses to tell a refusal from an outage."""
    async def _detailed(prompt, config=None):
        text, model = await judge(prompt, config=config)
        return text, model, {}
    return _detailed

DOCS = Path(__file__).resolve().parent.parent / "docs" / "evals"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _doc(name):
    return (DOCS / name).read_text(encoding="utf-8")


def _run_commands(doc):
    lines = doc.splitlines()
    out = []
    for i, line in enumerate(lines):
        if line.startswith("gbench --evals-only"):
            cmd = [line]
            while cmd[-1].rstrip().endswith("\\") and i + len(cmd) < len(lines):
                cmd.append(lines[i + len(cmd)])
            out.append(" ".join(cmd))
    return out


class _FakeDS(list):
    """The slice of datasets.Dataset the loaders use: iteration, len, column access, cast."""

    def cast_column(self, *_a, **_k):
        return self

    def __getitem__(self, key):
        if isinstance(key, str):
            return [row[key] for row in self]
        return list.__getitem__(self, key)


def _fake_datasets(monkeypatch, loader):
    monkeypatch.setitem(sys.modules, "datasets",
                        types.SimpleNamespace(load_dataset=loader, Image=lambda **k: None))


def _serve(replies, completion_tokens=5):
    """A fake _send_single_request answering by the last user text (None = failed request)."""
    async def _send(**kw):
        content = kw["messages"][-1]["content"]
        text = content if isinstance(content, str) else content[-1]["text"]
        reply = replies(text) if callable(replies) else replies[text]
        if reply is None:
            return base.Reply(text=None, tool_calls=None, error="server error")
        return base.Reply(text=reply, tool_calls=None, finish_reason="stop",
                          completion_tokens=completion_tokens)
    return _send


# --------------------------------------------------------------------------- #
# vqa_common.extract_short_answer: whole bold / italic label lines
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reply,expected", [
    ("**Answer: 14**", "14"),
    ("The chart peaks in 2019.\n\n**Final Answer: 14**", "14"),
    ("*Answer: 14*", "14"),
    ("__Answer: 14__", "14"),
    ("Answer: **14**.", "14"),
    ("**Answer: Yes**", "Yes"),
    ("***Answer: 14***", "14"),
    ("**Total: 7**", "7"),
    ("**Answer:** 14", "14"),
    ("**Answer**: `14`", "14"),
    ("Answer: **14**", "14"),
    ("> **Answer:** 14", "14"),
])
def test_whole_label_line_emphasis_is_removed(reply, expected):
    assert extract_short_answer(reply) == expected


@pytest.mark.parametrize("reply,expected", [
    ("Rabbit**", "Rabbit**"),                  # bare ChartQA gold ending in '**'
    ("LAAP*", "LAAP*"),
    ("Answer: LAAP*", "LAAP*"),                # plain label: the asterisk is part of the answer
    ("Answer: Rabbit**", "Rabbit**"),
    ("**Answer**: Rabbit**", "Rabbit**"),      # the run closed at the label
    ("**Answer:** Rabbit**", "Rabbit**"),      # the run closed right after the colon
    ("Answer: **LAAP***", "LAAP*"),
    ("Answer: U.S.", "U.S."),                  # a bare trailing period stays
    ("Answer: 14.", "14."),
    ("Answer: 2*3", "2*3"),
])
def test_answers_that_end_in_emphasis_characters_are_kept(reply, expected):
    assert extract_short_answer(reply) == expected


def test_the_four_vqa_suites_score_the_reply_as_returned():
    # lmms-eval passes results[0] to the task scorer: a wrapped answer is not unwrapped.
    reply = "Reading the bars.\n\n**Answer: 14**"
    assert chartqa._eval_chartqa(reply, "14") is False
    assert textvqa._eval_textvqa(reply, ["14"] * 10) is False
    assert docvqa._eval_docvqa(reply, ["14"]) is False
    assert infographicvqa._eval_infographicvqa(reply, ["14"]) is False
    for mod, gold in ((chartqa, "14"), (textvqa, ["14"] * 10), (docvqa, ["14"]),
                      (infographicvqa, ["14"])):
        assert getattr(mod, f"_eval_{mod.__name__.rsplit('.', 1)[-1]}")("14", gold) is True
    assert anls_score(extract_short_answer("*Answer: 14*"), ["14"]) == 1.0   # helper unchanged
    assert chartqa._eval_chartqa("Rabbit**", "Rabbit**") is True


# --------------------------------------------------------------------------- #
# chartqa / textvqa / docvqa / infographicvqa: lmms-eval prompts, pins, reasons
# --------------------------------------------------------------------------- #
def test_chartqa_uses_the_lmms_eval_prompt(monkeypatch):
    seen = {}

    def _load(*a, **k):
        seen.update(k)
        return _FakeDS([{"query": "How many bars?", "label": "4", "type": "human",
                         "image": {"bytes": PNG, "path": None}, "imgname": "a.png"}])
    _fake_datasets(monkeypatch, _load)
    (messages, gold, meta), = chartqa._load_chartqa_samples()
    assert seen["revision"] == chartqa.DATASET_REVISION
    assert messages[0]["content"][0]["type"] == "image_url"
    assert messages[0]["content"][1]["text"] == "How many bars?\nAnswer the question with a single word."
    assert gold == "4" and meta == {"category": "human"}


def test_textvqa_uses_the_lmms_eval_prompt_with_capitalize(monkeypatch):
    _fake_datasets(monkeypatch, lambda *a, **k: _FakeDS([{
        "question": "what brand is shown on 'The Hobbit' cover?", "answers": ["x"] * 10,
        "image": {"bytes": PNG, "path": None}, "question_id": 1}]))
    (messages, _gold, _meta), = textvqa._load_textvqa_samples()
    assert messages[0]["content"][1]["text"] == (
        "What brand is shown on 'the hobbit' cover?\n"
        "Answer the question using a single word or phrase.")


@pytest.mark.parametrize("module,loader,config,n", [
    (docvqa, "_load_docvqa_samples", "DocVQA", 5349),
    (infographicvqa, "_load_infographicvqa_samples", "InfographicVQA", 2801),
])
def test_doc_suites_load_the_pinned_lmms_eval_split(module, loader, config, n, monkeypatch):
    calls = []

    def _load(*a, **k):
        calls.append((a, k))
        return _FakeDS([{"questionId": "7", "question": "What is the date?",
                         "answers": ["May 1"], "question_types": ["form"],
                         "answer_type": ["single span"], "image": {"bytes": PNG, "path": None}}])
    _fake_datasets(monkeypatch, _load)
    (messages, gold, meta), = getattr(module, loader)()
    (args, kwargs), = calls
    assert args == ("lmms-lab-encoder/DocVQA", config)
    assert kwargs == {"split": "validation", "revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e"}
    assert messages[0]["content"][1]["text"] == (
        "What is the date?\nAnswer the question using a single word or phrase.")
    assert gold == ["May 1"]
    assert meta["category"] in ("form", "single span")
    assert module.N_QUESTIONS == n


@pytest.mark.parametrize("module,loader", [
    (chartqa, "_load_chartqa_samples"), (textvqa, "_load_textvqa_samples"),
    (docvqa, "_load_docvqa_samples"), (infographicvqa, "_load_infographicvqa_samples"),
])
def test_a_question_without_its_image_stops_the_load(module, loader, monkeypatch):
    _fake_datasets(monkeypatch, lambda *a, **k: _FakeDS([{
        "query": "q", "question": "q", "label": "1", "answers": ["1"], "type": "human",
        "image": None, "imgname": "a.png", "questionId": "1", "question_id": 1}]))
    with pytest.raises(RuntimeError, match="no decodable image"):
        getattr(module, loader)()


@pytest.mark.parametrize("module,run,loader,gold", [
    (chartqa, "run_chartqa", "_load_chartqa_samples", "14"),
    (textvqa, "run_textvqa", "_load_textvqa_samples", ["14"] * 10),
    (docvqa, "run_docvqa", "_load_docvqa_samples", ["14"]),
    (infographicvqa, "run_infographicvqa", "_load_infographicvqa_samples", ["14"]),
])
def test_vqa_results_name_limit_and_shard(module, run, loader, gold, monkeypatch):
    samples = [([{"role": "user", "content": [{"type": "text", "text": q}]}], gold,
                {"category": "c"}) for q in ("q1", "q2")]

    def _run(**kwargs):
        with mock.patch.object(base, "_send_single_request",
                               side_effect=_serve({"q1": "14", "q2": "nine hundred"})), \
             mock.patch.object(module, loader, lambda **k: samples):
            return getattr(module, run)("m", "http://x/v1", 1, **kwargs)
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    res = _run()
    assert res["accuracy"] == 50.0
    assert res["official_sampling"]["temperature"] == 0.0 and res["temperature"] == 0.0
    assert res["leaderboard_comparable"] is True and res["leaderboard_comparable_reason"] is None
    monkeypatch.setenv("GBENCH_SHARD", "1/2")
    res = _run(limit=2)
    reason = res["leaderboard_comparable_reason"]
    assert res["leaderboard_comparable"] is False
    assert "--eval-limit 2 subset" in reason and "sharded run (--shard 1/2)" in reason


# --------------------------------------------------------------------------- #
# coco_caption: Karpathy test split, seven metrics, scorer pre-flight
# --------------------------------------------------------------------------- #
def _coco_fixture(monkeypatch, n=3, missing=False):
    split = _FakeDS([{"filename": f"COCO_val2014_{i:012d}.jpg", "cocoid": i, "imgid": i,
                      "split": "test", "url": "http://images.cocodataset.org/x",
                      "sentences": [f"a photo of thing {i}", f"thing {i} in a photo"]}
                     for i in range(1, n + 1)])
    images = _FakeDS([{"file_name": f"COCO_val2014_{i:012d}.jpg",
                       "image": {"bytes": b"\xff\xd8\xff" + bytes([i]), "path": None},
                       "answer": ["other refs"]} for i in range(1 + int(missing), n + 1)])
    calls = []

    def _load(name, **k):
        calls.append((name, k))
        return split if name == coco_caption._SPLIT_DATASET else images
    _fake_datasets(monkeypatch, _load)
    monkeypatch.setattr(coco_caption, "_N_IMAGES", n)
    return calls


def test_coco_loads_the_karpathy_test_split_with_its_references(monkeypatch):
    calls = _coco_fixture(monkeypatch)
    samples = coco_caption._load_coco_caption_samples()
    assert calls == [("yerevann/coco-karpathy",
                      {"split": "test", "revision": "448fdb1bc7b2d09e46881c4541a14d796a3d41e8"}),
                     ("lmms-lab/COCO-Caption",
                      {"split": "val", "revision": "c3213b54e173ae86517d481de17f31439d6f8c72"})]
    assert len(samples) == 3
    messages, refs, meta = samples[1]
    assert refs == ["a photo of thing 2", "thing 2 in a photo"]      # Karpathy sentences
    assert meta == {"category": "captioning", "image_id": 2}
    assert messages[0]["content"][1] == {"type": "text", "text": "Describe the image briefly."}
    assert messages[0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,/9j/")


def test_coco_refuses_a_missing_image_or_a_wrong_split_size(monkeypatch):
    _coco_fixture(monkeypatch, missing=True)
    with pytest.raises(RuntimeError, match="is not in lmms-lab/COCO-Caption"):
        coco_caption._load_coco_caption_samples()
    _coco_fixture(monkeypatch)
    monkeypatch.setattr(coco_caption, "_N_IMAGES", 5000)
    with pytest.raises(RuntimeError, match="expected the 5,000 Karpathy test images"):
        coco_caption._load_coco_caption_samples()


def test_coco_real_scorers_compute_all_seven_metrics():
    assert shutil.which("java"), "pycocoevalcap's tokenizer and METEOR need a Java runtime"
    assert coco_caption._scorer_smoke_test() is None
    gts = {1: ["a man riding a horse on a beach", "a person on a horse by the sea"],
           2: ["two dogs playing in the snow", "dogs running through snow"]}
    scores, per_image = coco_caption.compute_caption_metrics(
        gts, {1: "a man rides a horse on the beach", 2: ""})       # "" = a failed request
    assert set(scores) == set(coco_caption.METRICS)
    assert all(0.0 <= v for v in scores.values())
    assert set(per_image) == {1, 2} and per_image[2] == 0.0


def test_coco_scorer_failure_stops_before_generation(monkeypatch):
    monkeypatch.setattr(coco_caption.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(coco_caption, "_scorer_smoke_test", lambda: "CalledProcessError: java")
    loaded = []
    monkeypatch.setattr(coco_caption, "_load_coco_caption_samples",
                        lambda **k: loaded.append(k) or [])
    with pytest.raises(RuntimeError, match="INFRA REQUIRED") as err:
        coco_caption.run_coco_caption("m", "http://x/v1")
    assert "two-image fixture (CalledProcessError: java)" in str(err.value)
    assert loaded == []


def _coco_run(monkeypatch, completion_tokens=5, **kwargs):
    samples = [([{"role": "user", "content": [{"type": "text", "text": f"img{i}"}]}],
                [f"ref {i}"], {"category": "captioning", "image_id": i}) for i in (1, 2)]
    monkeypatch.setattr(coco_caption, "check_coco_caption_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(coco_caption, "_load_coco_caption_samples", lambda **k: samples)
    fake_scores = {m: 0.5 for m in coco_caption.METRICS}
    fake_scores["CIDEr"] = 1.1
    monkeypatch.setattr(coco_caption, "compute_caption_metrics",
                        lambda gts, res: (fake_scores, {1: 1.2, 2: 0.2}))
    with mock.patch.object(base, "_send_single_request",
                           side_effect=_serve(lambda t: "a caption", completion_tokens)):
        return coco_caption.run_coco_caption("m", "http://x/v1", **kwargs)


def test_coco_result_headline_metrics_and_comparability(monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    monkeypatch.delenv("GBENCH_COCO_CAPTION_TEMPERATURE", raising=False)
    res = _coco_run(monkeypatch)
    assert res["accuracy"] == 110.0 and res["cider"] == 1.1
    for key in ("bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rouge_l"):
        assert res[key] == 0.5
    assert res["per_image_pass_rate"] == 50.0
    assert "captions_over_upstream_cap" not in res     # the 64-token cap is sent, not counted
    assert res["official_sampling"]["max_tokens"] == 64
    assert res["leaderboard_comparable"] is True and res["leaderboard_comparable_reason"] is None
    assert "yerevann/coco-karpathy@448fdb1b" in res["dataset"]
    statuses = {t["status"] for t in res["sample_traces"]}
    assert statuses == {"OK"}


def test_coco_limit_and_shard_are_not_comparable(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "2/3")
    res = _coco_run(monkeypatch, completion_tokens=65, limit=2)
    reason = res["leaderboard_comparable_reason"]
    assert res["leaderboard_comparable"] is False
    assert "64-token" not in reason                     # the cap is sent, so none runs past it
    assert "--eval-limit 2 subset" in reason and "sharded run (--shard 2/3)" in reason


# --------------------------------------------------------------------------- #
# healthbench: a request with no reply is a harness error, counted once and not graded
# --------------------------------------------------------------------------- #
def test_healthbench_failed_request_is_counted_once_and_not_graded():
    rubric = [{"criterion": "Advises emergency care", "points": 10, "tags": ["axis:accuracy"]}]
    samples = [([{"role": "user", "content": q}], json.dumps(rubric),
                {"category": "theme:t", "example_tags": ["theme:t"]}) for q in ("q1", "q2")]

    prompts = []

    async def _grader(contents, *, config=None):
        prompts.append(contents)
        return json.dumps({"explanation": "e", "criteria_met": True}), "gemini-x"

    with mock.patch.object(base, "_send_single_request",
                           side_effect=_serve({"q1": "Go to the ER.", "q2": None})), \
         mock.patch.object(healthbench, "gemini_required_skip", lambda *a: None), \
         mock.patch.object(healthbench, "_load_healthbench_samples", lambda **k: samples), \
         _patch_detailed(healthbench, _grader):
        res = healthbench.run_healthbench("m", "http://x/v1", 1)
    assert res["failed_requests"] == 1 and res["judge_failures"] == 0
    assert res["status"] == "completed_with_errors"
    assert res["healthbench_scored_conversations"] == 1 and res["accuracy"] == 100.0
    assert len(prompts) == 1 and not any("No response (bad request)." in p for p in prompts)
    failed = next(t for t in res["sample_traces"] if t["response_text"] is None)
    assert failed["judge_grade"] == "not_judged_failed_request"
    assert failed["scoring_excluded"] is True and failed["harness_error"]
    assert "got no reply from the served endpoint" in res["leaderboard_comparable_reason"]


def test_healthbench_failed_request_is_left_out_of_the_mean():
    rubric = [{"criterion": "Advises emergency care", "points": 10, "tags": []}]
    samples = [([{"role": "user", "content": q}], json.dumps(rubric), {"category": "t"})
               for q in ("q1", "q2")]

    async def _grader(contents, *, config=None):
        met = "No response (bad request)." not in contents
        return json.dumps({"explanation": "e", "criteria_met": met}), "gemini-x"

    with mock.patch.object(base, "_send_single_request",
                           side_effect=_serve({"q1": "Go to the ER.", "q2": None})), \
         mock.patch.object(healthbench, "gemini_required_skip", lambda *a: None), \
         mock.patch.object(healthbench, "_load_healthbench_samples", lambda **k: samples), \
         _patch_detailed(healthbench, _grader):
        res = healthbench.run_healthbench("m", "http://x/v1", 1)
    assert res["healthbench_scored_conversations"] == 1
    assert res["accuracy"] == 100.0 and res["failed_requests"] == 1


# --------------------------------------------------------------------------- #
# imo_answer_bench: a judge outage leaves no PENDING_JUDGE status behind
# --------------------------------------------------------------------------- #
def test_imo_judge_outage_trace_status_is_ok(monkeypatch):
    monkeypatch.setattr(imo_answer_bench, "gemini_api_keys", lambda: ["k"])

    async def _outage(prompt, config=None):
        return None, "JUDGE_OUTAGE"
    trace = {"response_text": "The answer is \\boxed{3}.", "gold_answer": "3",
             "status": "PENDING_JUDGE", "extra_payload": {"problem": "p"}}
    with mock.patch.object(imo_answer_bench, "judge_generate_cascade", _outage):
        asyncio.run(imo_answer_bench._async_grade_imo([trace]))
    assert trace["judge_grade"] == "JUDGE_OUTAGE" and trace["status"] == "OK"


# --------------------------------------------------------------------------- #
# medxpertqa: the status follows the questions
# --------------------------------------------------------------------------- #
#: What base puts on the trace of a request that got no reply from the endpoint.
_NO_REPLY = {"harness_error": {"kind": "endpoint_failure", "detail": "HTTP 500"},
             "scoring_excluded": True}


def _medx_turn1(status, text):
    return [{"sample_idx": i, "category": "Text/Reasoning",
             "gold_answer": {"label": "A", "n_options": 10},
             "extra_payload": {"medxpertqa_id": f"T{i}", "subset": "Text"},
             "response_text": text, "status": status, "is_correct": False,
             **(_NO_REPLY if text is None else {})} for i in range(2)]


def _medx_combine(turn1_traces, turn1_failed, turn2_ok):
    turn1 = {"sample_traces": turn1_traces, "failed_requests": turn1_failed,
             "truncated_responses": 0}
    samples, no_rat = medxpertqa._answer_turn_samples(
        turn1_traces, {f"T{i}": [{"role": "user", "content": "q"}] for i in range(2)})
    traces2 = [{"category": "Text/Reasoning", "extra_payload": s[2], "gold_answer": s[1],
                "response_text": "A" if turn2_ok else None, "is_correct": turn2_ok,
                "status": "OK" if turn2_ok else "FAILED",
                **({} if turn2_ok else _NO_REPLY)} for s in samples]
    turn2 = {"sample_traces": traces2, "failed_requests": 0 if turn2_ok else len(traces2),
             "truncated_responses": 0}
    return medxpertqa._combine(turn1, turn2, no_rat, 1)


def test_medxpertqa_empty_rationales_answered_in_turn_two_are_not_failed():
    res = _medx_combine(_medx_turn1("EMPTY_RESPONSE", ""), 2, True)
    assert res["accuracy"] == 100.0 and res["failed_requests"] == 2
    assert res["medxpertqa_unanswered_questions"] == 0
    assert res["status"] == "completed_with_errors"


def test_medxpertqa_is_failed_only_when_no_question_is_answered():
    res = _medx_combine(_medx_turn1("FAILED", None), 2, True)       # no rationale at all
    assert res["medxpertqa_unanswered_questions"] == 2 and res["status"] == "failed"
    # every request got no reply: harness errors, so the score is withheld, never 0
    assert res["accuracy"] is None and res["accuracy_withheld_reason"]
    res = _medx_combine(_medx_turn1("OK", "because"), 0, False)     # every turn 2 failed
    assert res["medxpertqa_unanswered_questions"] == 2 and res["status"] == "failed"
    assert res["accuracy"] is None and res["scoring_excluded"] == 2
    res = _medx_combine(_medx_turn1("OK", "because"), 0, True)
    assert res["status"] == "success"


# --------------------------------------------------------------------------- #
# wildbench: v2 (1,024 tasks) and WB_score.task_macro
# --------------------------------------------------------------------------- #
def test_wildbench_task_groups_follow_create_tables():
    assert wildbench.task_groups("Editing", ["Coding & Debugging"]) == [
        "Coding & Debugging", "Creative Tasks"]
    assert wildbench.task_groups("Math", ["Data Analysis", "Reasoning"]) == [
        "Math & Data Analysis", "Planning & Reasoning"]
    assert wildbench.task_groups("Others", []) == ["Creative Tasks"]
    assert sum(wildbench.TASK_MACRO_WEIGHTS.values()) == 4.75


def test_wildbench_task_macro_matches_the_upstream_formula():
    def _t(raw, groups, text="reply", excluded=False):
        t = {"wb_score_raw": raw, "response_text": text, "extra_payload": {"wb_task_groups": groups}}
        if excluded:
            t["scoring_excluded"] = True
        return t
    traces = [_t(9.0, ["Coding & Debugging"]), _t(7.0, ["Coding & Debugging", "Creative Tasks"]),
              _t(1.0, ["Creative Tasks"], text=""),              # empty reply: out of the groups
              _t(None, ["Math & Data Analysis"], excluded=True),   # grading failure
              _t(4.0, ["Planning & Reasoning"])]
    macro, groups = wildbench.wb_task_macro(traces)
    coding, creative, planning = (8 - 5) * 2, (7 - 5) * 2, (4 - 5) * 2
    expected = (coding * 1.25 + creative * 0.5 + planning * 1.25) / 4.75 * 10
    assert macro == pytest.approx(round(expected, 4))
    assert groups == {"Coding & Debugging": 60.0, "Creative Tasks": 40.0,
                      "Planning & Reasoning": -20.0}
    assert wildbench.wb_task_macro([]) == (None, {})


def test_wildbench_loads_all_v2_tasks_coding_included(monkeypatch):
    rows = [{"session_id": f"s{i}", "conversation_input": [{"role": "user", "content": f"q{i}"}],
             "checklist": ["c"], "primary_tag": tag, "secondary_tags": ["Reasoning"]}
            for i, tag in enumerate(["Coding & Debugging", "Math", "Editing"])]
    seen = {}

    def _load(*a, **k):
        seen["args"], seen["kwargs"] = a, k
        return rows
    _fake_datasets(monkeypatch, _load)
    monkeypatch.setattr(wildbench, "N_TASKS", 3)
    samples = wildbench._load_wildbench_samples()
    assert seen["args"] == ("allenai/WildBench", "v2")
    assert seen["kwargs"] == {"split": "test",
                              "revision": "26c49eb39d7d5ce2099b0bbafed5a88dcce954ec"}
    assert [m["category"] for _, _, m in samples] == ["Coding & Debugging", "Math", "Editing"]
    assert samples[0][2]["wb_task_groups"] == ["Coding & Debugging", "Planning & Reasoning"]
    monkeypatch.setattr(wildbench, "N_TASKS", 1024)
    with pytest.raises(RuntimeError, match="expected 1024"):
        wildbench._load_wildbench_samples()


def test_wildbench_reports_wb_score_and_task_macro(monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    samples = [([{"role": "user", "content": q}], ["c"],
                {"category": "Math", "wb_task_groups": groups, "session_id": q,
                 "wb_pairwise_target_is_a": True})
               for q, groups in (("q1", ["Math & Data Analysis"]), ("q2", ["Creative Tasks"]))]
    scores = {"a": "9", "b": "5"}
    refs = {p: {"q1": "r", "q2": "r"} for p in wildbench.REFERENCE_MODELS}

    async def _judge(prompt, config=None):
        if "<|begin_of_response_A|>" in prompt:      # WB-Reward comparison
            return '{"choice": "A=B"}', "gemini-x"
        out = prompt.split("<|begin_of_response|>\n\n")[1].split("\n\n<|end_of_response|>")[0]
        return '{"score": "%s"}' % scores[out], "gemini-x"

    with mock.patch.object(base, "_send_single_request",
                           side_effect=_serve({"q1": "a", "q2": "b"})), \
         mock.patch.object(wildbench, "gemini_required_skip", lambda *a: None), \
         mock.patch.object(wildbench, "_load_wildbench_samples", lambda **k: samples), \
         mock.patch.object(wildbench, "_load_reference_outputs", lambda: refs), \
         mock.patch.object(wildbench, "_load_wb_elo_pool_outputs", lambda: {}), \
         mock.patch.object(wildbench, "judge_generate_cascade_detailed", _as_detailed(_judge)):
        res = wildbench.run_wildbench("m", "http://x/v1", 1)
    assert res["wb_score"] == pytest.approx((7 - 5) * 20)
    assert res["accuracy"] == res["wb_score_task_macro"]
    assert res["wb_score_task_groups"] == {"Creative Tasks": 0.0, "Math & Data Analysis": 80.0}
    assert res["wb_score_task_macro"] == pytest.approx(round((8 * 1 + 0 * 0.5) / 4.75 * 10, 4))
    assert "gpt-4o-2024-05-13" in res["leaderboard_comparable_reason"]
    assert "non-coding" not in res["leaderboard_comparable_reason"]
    assert res["dataset"] == "allenai/WildBench@26c49eb39d7d5ce2099b0bbafed5a88dcce954ec (v2, test)"


# --------------------------------------------------------------------------- #
# ruler: the pre-flight checks the token-longest prompt of each cell
# --------------------------------------------------------------------------- #
def test_ruler_preflight_picks_the_longest_prompt_by_token_count(monkeypatch):
    checked = []

    def _clamp(messages, budget, *_a):
        checked.append(messages[-1]["content"])
        return budget, None
    monkeypatch.setattr(ruler, "clamp_to_context", _clamp)
    monkeypatch.setattr(ruler, "get_run_knob", lambda *a, **k: None)
    cell = "vt@128k"
    samples = [([{"role": "user", "content": "x" * 900}], "{}",
                {"category": cell, "ruler_length": 100, "max_tokens": 30}),
               ([{"role": "user", "content": "token-dense"}], "{}",
                {"category": cell, "ruler_length": 130, "max_tokens": 30})]
    ruler._require_prompts_fit(samples, False, None)
    assert checked == ["token-dense"]


def test_ruler_rows_carry_their_token_count_only_in_the_meta():
    src = Path(ruler.__file__).read_text(encoding="utf-8")
    assert '"ruler_length": int(r.get("length") or 0)' in src
    allowed = Path(base.__file__).read_text(encoding="utf-8").split(
        "ALLOWED_API_KEYS = {", 1)[1].split("}", 1)[0]
    assert '"max_tokens"' in allowed and "ruler_length" not in allowed


# --------------------------------------------------------------------------- #
# charxiv: pinned revision, charts without images refused
# --------------------------------------------------------------------------- #
_CHARXIV_ROW = {"category": "cs", "figure_path": "images/1.jpg",
                "descriptive_q1": 1, "descriptive_a1": "Title A", "reasoning_q": "What is X?",
                "reasoning_a": "3.14", "reasoning_a_type": 4, "subplot_loc": None,
                "subplot_row": 0, "subplot_col": 0}


def _charxiv_zip(monkeypatch, tmp_path, members):
    """images.zip stand-in (the chart files the official harness sends), no download."""
    import zipfile
    path = tmp_path / "images.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    monkeypatch.setattr(charxiv, "chart_images_zip", lambda: str(path))


def test_charxiv_loads_the_pinned_revision(monkeypatch, tmp_path):
    seen = {}

    def _load(*a, **k):
        seen.update(k)
        return _FakeDS([{**_CHARXIV_ROW, "image": {"bytes": PNG, "path": None}}])
    _fake_datasets(monkeypatch, _load)
    _charxiv_zip(monkeypatch, tmp_path, {"1.jpg": PNG})
    samples = charxiv._load_charxiv_samples(limit=None)
    assert seen == {"split": "validation", "revision": "f441eb632fc62f6f777830a0f47619e6e86459b0"}
    # gpt.py order: the question text, then the image
    assert len(samples) == 2 and all(m[0]["content"][-1]["type"] == "image_url"
                                     and m[0]["content"][0]["type"] == "text"
                                     for m, _, _ in samples)


def test_charxiv_chart_without_an_image_stops_the_load(monkeypatch, tmp_path):
    _fake_datasets(monkeypatch, lambda *a, **k: _FakeDS([{**_CHARXIV_ROW, "image": None}]))
    _charxiv_zip(monkeypatch, tmp_path, {"2.jpg": PNG})      # no 1.jpg: the chart is missing
    with pytest.raises(RuntimeError, match="no decodable image"):
        charxiv._load_charxiv_samples(limit=None)


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["coco_caption", "chartqa", "textvqa", "docvqa", "infographicvqa",
                                  "charxiv", "wildbench", "healthbench", "imo_answer_bench",
                                  "medxpertqa", "ruler"])
def test_changed_docs_are_ascii_and_pass_max_output_tokens(name):
    doc = _doc(f"{name}.md")
    assert all(ord(ch) < 128 for ch in doc), name
    commands = _run_commands(doc)
    assert commands and all("--max-output-tokens" in c for c in commands), name


def test_docs_state_the_protocols_the_code_runs():
    coco = _doc("coco_caption.md")
    assert "Karpathy test split" in coco and "448fdb1b" in coco and "c3213b54" in coco
    assert "`Describe the image briefly.`" in coco and "5,000" in coco
    assert "40,504-image run" not in coco
    for name, config in (("docvqa", "DocVQA"), ("infographicvqa", "InfographicVQA")):
        doc = _doc(f"{name}.md")
        assert "539088ef" in doc and f"config `{config}`" in doc
        assert "`\\nAnswer the question using a single word or phrase.`" in doc
    assert "`\\nAnswer the question with a single word.`" in _doc("chartqa.md")
    wb = _doc("wildbench.md")
    assert "`wb_score_task_macro`" in wb and "1,024" in wb and "WildEval/WildBench" not in wb
    hb = _doc("healthbench.md")
    assert "not_judged_failed_request" in hb and "harness errors" in hb
    assert "`medxpertqa_unanswered_questions`" in _doc("medxpertqa.md")
    assert "f441eb632fc62f6f777830a0f47619e6e86459b0" in _doc("charxiv.md")


def test_toolchains_doc_puts_multipl_e_in_docker_not_bwrap():
    doc = _doc("toolchains.md")
    assert all(ord(ch) < 128 for ch in doc)
    bwrap = doc.split("## Sandboxing", 1)[1].split("\n\n")[1]
    assert "`codeforces`, `cruxeval`, `lcb`, `scicode`" in bwrap
    assert "`multipl_e` executes inside its Docker image" in bwrap
    docker_row = next(l for l in doc.splitlines() if l.startswith("| Docker CLI"))
    assert "`multipl_e`" in docker_row
    assert "SPICE" not in doc
