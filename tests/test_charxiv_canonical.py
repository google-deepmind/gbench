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

"""Canonical CharXiv: descriptive + reasoning tracks, per-rubric JSON grading."""
import asyncio
import json
import warnings

warnings.filterwarnings("ignore")  # upstream constants carry benign invalid-escape strings

import gbench.runners.eval_suites.charxiv as C


_ROW = {
    "image": {"bytes": b"\x89PNG\r\n\x1a\n", "path": None}, "category": "cs",
    "figure_path": "images/0.jpg",
    "descriptive_q1": 1,  "descriptive_a1": "Title A",       # title rubric
    "descriptive_q2": 18, "descriptive_a2": "2 by 2",        # layout, no subplot prefix
    "descriptive_q3": 8,  "descriptive_a3": "5",             # quant
    "descriptive_q4": 16, "descriptive_a4": "increasing",    # trend
    "reasoning_q": "What is X?", "reasoning_a": "3.14", "reasoning_a_type": 4,
    "subplot_loc": None, "subplot_row": 0, "subplot_col": 0,
}


def _charts_zip(monkeypatch, tmp_path, members=None):
    """images.zip stand-in (the charts the official harness sends), no download."""
    import zipfile
    path = tmp_path / "images.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in (members or {"0.jpg": b"\xff\xd8\xff\xe0chart0"}).items():
            zf.writestr(name, data)
    monkeypatch.setattr(C, "chart_images_zip", lambda: str(path))
    return path


def test_charxiv_loader_fans_out_five_per_chart(monkeypatch, tmp_path):
    import datasets
    _charts_zip(monkeypatch, tmp_path)
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: [dict(_ROW)])
    samples = C._load_charxiv_samples(limit=None)

    assert len(samples) == 5, "1 chart -> 4 descriptive + 1 reasoning"
    tracks = [meta["category"] for _, _, meta in samples]
    assert tracks.count("descriptive") == 4 and tracks.count("reasoning") == 1

    for _msgs, gold, meta in samples:
        # meta dict is category-ONLY (anything else leaks into the request payload)
        assert set(meta.keys()) == {"category"}
        blob = json.loads(gold)
        assert blob["track"] == meta["category"]
        assert "answer" in blob

    rgold = next(json.loads(g) for _, g, m in samples if m["category"] == "reasoning")
    assert rgold["inst_category"] == 4 and rgold["raw_question"] == "What is X?"


def test_charxiv_reasoning_ic4_appends_decimal_instruction(monkeypatch, tmp_path):
    import datasets
    _charts_zip(monkeypatch, tmp_path)
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: [dict(_ROW)])
    samples = C._load_charxiv_samples(limit=None)
    r_msgs = next(m for m, _, meta in samples if meta["category"] == "reasoning")
    text = r_msgs[0]["content"][0]["text"]  # first content part is the question text (gpt.py)
    assert "2 decimal places" in text  # get_number_instruction("3.14")


def _run_judge(traces, cascade):
    import contextlib
    class _P:
        def update(self, n): pass
    # patch the judge + a no-op tqdm bar
    import gbench.runners.eval_suites.charxiv as CC
    orig = CC.judge_generate_cascade
    CC.judge_generate_cascade = cascade
    try:
        asyncio.run(CC._async_judge_charxiv(traces))
    finally:
        CC.judge_generate_cascade = orig


def test_charxiv_judge_scores_both_tracks_and_handles_outage_and_invalid(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    async def cascade(prompt, *, config=None):
        if "Response 1:" in prompt:                      # descriptive
            return ('{"extract_answer_T1":"x","score_T1":1}', "gemini")
        if "OUTAGE" in prompt:
            return (None, "JUDGE_OUTAGE")
        if "INVALID" in prompt:
            return ("not json at all", "gemini")
        return ('{"extracted_answer":"y","score":0}', "gemini")   # reasoning

    traces = [
        {"category": "descriptive", "response_text": "A",
         "gold_answer": json.dumps({"track": "descriptive", "qid": 1, "answer": "A"})},
        {"category": "reasoning", "response_text": "B",
         "gold_answer": json.dumps({"track": "reasoning", "inst_category": 2,
                                    "raw_question": "Q", "answer": "B"})},
        {"category": "reasoning", "response_text": "C",
         "gold_answer": json.dumps({"track": "reasoning", "inst_category": 2,
                                    "raw_question": "OUTAGE", "answer": "C"})},
        {"category": "reasoning", "response_text": "D",
         "gold_answer": json.dumps({"track": "reasoning", "inst_category": 2,
                                    "raw_question": "INVALID", "answer": "D"})},
    ]
    _run_judge(traces, cascade)

    assert traces[0]["is_correct"] is True and traces[0]["judge_grade"] == "CORRECT"
    assert traces[1]["is_correct"] is False and traces[1]["judge_grade"] == "INCORRECT"
    assert traces[2]["judge_grade"] == "JUDGE_OUTAGE" and "is_correct" not in traces[2]
    # an unparseable reply is asked again up to upstream's 10 attempts, then JUDGE_FAILED (0)
    assert traces[3]["judge_grade"] == "JUDGE_FAILED" and traces[3]["is_correct"] is False
    assert traces[3]["judge_batch"]["calls"] == C.JUDGE_MAX_ATTEMPTS == 10
