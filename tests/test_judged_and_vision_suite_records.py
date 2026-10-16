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

"""Judged and vision suites send upstream's inputs, never score a judge failure against the
model, and put their headline metric next to their headline.

* charxiv: reasoning questions and answers as the dataset holds them; a substituted-judge failure
  left out (upstream's -1-as-0 figure kept apart); an empty descriptive reply in its batch.
* gdpval: the judge sees the task's text input files; a task without a reference is the
  protocol's exclusion; clipped input files counted once per file.
* browsecomp / frames / humanitys_last_exam: a request with no reply is a failed request, an
  empty reply an empty response; FRAMES prompts sent verbatim.
* aa_lcr: a bare verdict is read and counted.
* lab_bench, causalbench, i18n_translate, coco_caption, bundled_detection: no 0.0 for nothing,
  the headline metric in the category rows, exact means, upstream's image bytes.

No network, no model server, no Gemini: every judge is faked.
"""

import asyncio
import base64
import io
import json
import zipfile

import pytest

from gbench.runners.eval_suites import aa_lcr, browsecomp, causalbench, charxiv, frames
from gbench.runners.eval_suites import gdpval, humanitys_last_exam, i18n_translate, lab_bench


# --------------------------------------------------------------------------- #
# charxiv                                                                      #
# --------------------------------------------------------------------------- #
def test_charxiv_reasoning_text_is_the_datasets(monkeypatch, tmp_path):
    path = tmp_path / "images.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("20.jpg", b"\xff\xd8\xff\xe0chart")
    monkeypatch.setattr(charxiv, "chart_images_zip", lambda: str(path))
    row = {"category": "cs", "figure_path": "images/20.jpg", "descriptive_q1": 1,
           "descriptive_a1": "T", "reasoning_q": "  How many bars?\n", "reasoning_a": " 3 ",
           "reasoning_a_type": 4, "subplot_loc": None, "subplot_row": 0, "subplot_col": 0}
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: [dict(row)])
    samples = charxiv._load_charxiv_samples()
    reasoning = next(s for s in samples if json.loads(s[1])["track"] == "reasoning")
    gold = json.loads(reasoning[1])
    assert gold["raw_question"] == "  How many bars?\n" and gold["answer"] == " 3 "
    assert "  How many bars?\n" in reasoning[0][0]["content"][0]["text"]


def test_charxiv_judge_failures_are_left_out_and_upstreams_figure_kept():
    batch = {"template": 1, "position": 1, "size": 1, "calls": 10}
    failed, ok = {"category": "reasoning"}, {"category": "reasoning"}
    charxiv._set_verdict(failed, "failed", None, None, "garbage", "gemini-x", batch)
    charxiv._set_verdict(ok, "ok", 1, "3", "{}", "gemini-x", batch)
    assert failed["judge_grade"] == "JUDGE_FAILED" and failed["scoring_excluded"] is True
    assert ok["is_correct"] is True and not ok.get("scoring_excluded")


def test_charxiv_grades_an_empty_descriptive_reply_in_its_batch(monkeypatch):
    prompts = []

    async def grade(prompt, parse, cfg, sem, usage):
        prompts.append(prompt)
        n = prompt.count("<|response|>") or 1
        content = {f"extract_answer_T{i}": "x" for i in range(1, 3)}
        content.update({f"score_T{i}": 0 for i in range(1, 3)})
        return "ok", content, "{}", "gemini-x", 1
    monkeypatch.setattr(charxiv, "_grade_prompt", grade)
    monkeypatch.setattr(charxiv, "gemini_api_keys", lambda: ["k"])
    gold = json.dumps({"track": "descriptive", "qid": 1, "answer": "T"})
    traces = [{"sample_idx": 0, "category": "descriptive", "gold_answer": gold,
               "response_text": ""},
              {"sample_idx": 1, "category": "descriptive", "gold_answer": gold,
               "response_text": "an answer"},
              {"sample_idx": 2, "category": "descriptive", "gold_answer": gold,
               "response_text": None, "harness_error": {"kind": "endpoint_failure"}}]
    asyncio.run(charxiv._async_judge_charxiv(traces))
    assert traces[0]["judge_batch"]["size"] == 2 and traces[0]["judge_batch"]["position"] == 1
    assert traces[1]["judge_batch"]["position"] == 2
    assert traces[2]["judge_grade"] == "FAILED" and "judge_batch" not in traces[2]


# --------------------------------------------------------------------------- #
# gdpval                                                                       #
# --------------------------------------------------------------------------- #
def test_gdpval_judge_sees_the_text_rendering_of_the_inputs():
    trace = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Schedule the interviews."},
        {"type": "text", "text": "\n--- attachment: rooms.docx ---\nRoom A, Room B"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]}
    task = gdpval._task_prompt(trace)
    assert "Schedule the interviews." in task and "Room A, Room B" in task
    assert gdpval._media_inputs(trace) == 1


# --------------------------------------------------------------------------- #
# browsecomp / frames / humanitys_last_exam                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module, fn", [
    (browsecomp, "_async_judge_browsecomp"), (frames, "_async_judge_frames"),
    (humanitys_last_exam, "_async_judge_humanitys_last_exam")])
def test_no_reply_and_an_empty_reply_are_labelled_apart(monkeypatch, module, fn):
    async def never(*a, **k):
        raise AssertionError("the judge is not called for a reply with nothing to grade")
    for name in ("judge_generate_cascade", "judge_generate_cascade_detailed"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, never)
    traces = [{"sample_idx": 0, "response_text": None, "status": "FAILED",
               "harness_error": {"kind": "endpoint_failure"}, "messages": [], "gold_answer": "g"},
              {"sample_idx": 1, "response_text": "", "status": "EMPTY_RESPONSE", "messages": [],
               "gold_answer": "g"}]
    asyncio.run(getattr(module, fn)(traces))
    assert traces[0]["judge_grade"] == "not_judged_failed_request"
    assert traces[0]["scoring_excluded"] is True
    assert traces[1]["judge_grade"] == "not_judged_empty_response"
    assert not traces[1].get("scoring_excluded")


def test_frames_sends_the_question_verbatim(monkeypatch):
    rows = [{"Prompt": "Which trench?   ", "Answer": "28 ", "reasoning_types": "Numerical"}]
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: rows)
    samples = frames._load_frames_samples()
    assert samples[0][0][0]["content"] == "Which trench?   " and samples[0][1] == "28 "


# --------------------------------------------------------------------------- #
# aa_lcr                                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reply, verdict, relaxed", [
    ('{"verdict": "CORRECT"}', "CORRECT", False),
    ('```json\n"INCORRECT"\n```', "INCORRECT", True),
    ("Verdict: **CORRECT**", "CORRECT", True),
    ("CORRECT or INCORRECT", None, False),
    ("The answer is correct.", None, False),
])
def test_aa_lcr_reads_a_bare_verdict(reply, verdict, relaxed):
    assert aa_lcr.parse_verdict_detailed(reply) == (verdict, relaxed)


# --------------------------------------------------------------------------- #
# lab_bench                                                                    #
# --------------------------------------------------------------------------- #
def test_lab_bench_rates_are_null_with_nothing_to_divide_by():
    m = lab_bench._metrics([])
    assert m["accuracy"] is None and m["precision"] is None and m["coverage"] is None
    m = lab_bench._metrics([(False, False)])
    assert m["accuracy"] == 0.0 and m["precision"] is None and m["coverage"] == 0.0


def test_lab_bench_sends_the_image_as_upstreams_encode_image():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 200, 30)).save(buf, format="JPEG", quality=95)
    raw = buf.getvalue()
    url = lab_bench._image_url({"bytes": raw})
    image = Image.open(io.BytesIO(raw))
    again = io.BytesIO()
    image.save(again, format=image.format)         # labbench/utils.py encode_image
    assert url == "data:image/jpeg;base64," + base64.b64encode(again.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# causalbench / i18n_translate                                                 #
# --------------------------------------------------------------------------- #
def test_causalbench_reports_no_figure_for_nothing():
    rep = causalbench.perf_report([], [])
    assert rep["F1"] is None and rep["Acc"] is None and rep["total"] == 0


def test_i18n_means_are_taken_over_the_exact_scores(monkeypatch):
    exact = {"de-en": 20.305150868, "en-de": 63.155523718, "fi-en": 30.405596969,
             "kk-en": 3.673526563}
    per_dir = {d: {"bleu": round(v, 2), "chrf": 50.0, "bleu_exact": v, "chrf_exact": 50.0,
                   "segments": 1, "bleu_signature": "s", "chrf_signature": "s"}
               for d, v in exact.items()}
    monkeypatch.setattr(i18n_translate, "_load_i18n_translate_samples", lambda limit=None: [])
    monkeypatch.setattr(i18n_translate, "_require_sacrebleu",
                        lambda: type("S", (), {"__version__": i18n_translate.SACREBLEU_RELEASE}))
    monkeypatch.setattr(i18n_translate, "wmt_scores", lambda traces: per_dir)
    monkeypatch.setattr(i18n_translate, "chrf_score", lambda h, r: 50.0)
    traces = [{"sample_idx": i, "category": d, "response_text": "x", "gold_answer": "y",
               "is_correct": True, "status": "OK"} for i, d in enumerate(exact)]
    monkeypatch.setattr(i18n_translate, "run_eval_suite", lambda **kw: {
        "sample_traces": traces, "accuracy": 100.0, "correct_answers": 4,
        "category_accuracy": {d: {"correct": 1, "total": 1, "accuracy": 100.0} for d in exact}})
    res = i18n_translate.run_i18n_translate("m", "http://x/v1", 1)
    assert res["accuracy"] == 29.38                  # not 29.39 from the rounded directions
    assert res["headline_metrics"] == {"bleu_mean": 29.38, "chrf_mean": 50.0}
    assert res["category_accuracy"]["de-en"] == {"correct": None, "total": 1, "accuracy": 20.31}
    assert res["correct_answers"] is None and res["segments_chrf40"] == 4


def test_gdpval_protocol_exclusions_and_clip_counts(monkeypatch):
    monkeypatch.setattr(gdpval, "judge_generate_cascade", None)      # never called here
    clips = [{"file": "a.pdf", "cut": "pdf_pages", "full": 30, "kept": 10},
             {"file": "a.pdf", "cut": "chars", "full": 50000, "kept": 20000}]
    trace = {"sample_idx": 0, "response_text": "my deliverable", "status": "PENDING_JUDGE",
             "gold_answer": json.dumps([{"criterion": "c", "points": 1}]),
             "messages": [{"role": "user", "content": "task"}],
             "extra_payload": {"task_id": "t1", "inputs_clipped": clips}}
    metrics = {}
    score = gdpval._make_scorer(None, 1, metrics,
                                {"t1": {"text": None, "note": "none", "n_files": 0}})
    asyncio.run(score([trace]))
    assert trace["status"] == "NO_REFERENCE" and trace["scoring_excluded"] is True
    assert trace["excluded_by_protocol"] == "no_deliverable_shipped"
    assert metrics["gdpval_report"]["input_files_clipped"] == 1     # one file, two cuts


def test_coco_caption_rows_carry_the_corpus_score():
    from gbench.runners.eval_suites import coco_caption
    result = {"accuracy": 12.5, "correct_answers": 1, "total_questions": 8,
              "category_accuracy": {"captioning": {"correct": 1, "total": 8, "accuracy": 12.5}},
              "sample_traces": []}
    coco_caption._headline_and_comparability(result, {"cider": 0.1282}, None)
    assert result["accuracy"] == 12.82 and result["per_image_pass_rate"] == 12.5
    assert result["correct_answers"] is None and result["per_image_passes"] == 1
    assert result["category_accuracy"] == {"captioning": {"total": 8, "correct": None,
                                                          "accuracy": 12.82}}
    assert result["category_cider_pass_rate"]["captioning"]["correct"] == 1


def test_lab_bench_withholds_the_mean_when_a_category_has_nothing_scored(monkeypatch):
    from gbench.runners.eval_suites import base
    monkeypatch.setattr(lab_bench, "_load_lab_bench_samples", lambda limit=None: [])
    monkeypatch.setattr(lab_bench, "_require_image_input", lambda *a: None)
    ok = {"sample_idx": 0, "category": "LitQA2", "is_correct": True, "status": "OK",
          "response_text": "[ANSWER]A[/ANSWER]",
          "gold_answer": {"letter": "A", "letters": "AB", "refuse": "B", "category": "LitQA2",
                          "subtask": "LitQA2"}}
    lost = {"sample_idx": 1, "category": "SeqQA", "is_correct": False, "status": "FAILED",
            "response_text": None, "scoring_excluded": True,
            "harness_error": {"kind": "endpoint_failure"},
            "gold_answer": {"letter": "A", "letters": "AB", "refuse": "B", "category": "SeqQA",
                            "subtask": "SeqQA"}}
    monkeypatch.setattr(lab_bench, "run_eval_suite", lambda **kw: {
        "sample_traces": [ok, lost], "accuracy": 100.0, "status": "completed_with_errors"})
    res = lab_bench.run_lab_bench("m", "http://x/v1")
    assert res["accuracy"] is None
    assert "SeqQA ran but has no question in the score" in res["accuracy_withheld_reason"]
    assert res["lab_bench_report"]["categories_unscored"] == ["SeqQA"]
    assert "not run" in res["metric"] and "of LAB-Bench's 8 categories" in res["metric"]
