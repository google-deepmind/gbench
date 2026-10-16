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

"""Canonical-fidelity regressions for the vision suites.

chartqa relaxed accuracy, the TextVQA EvalAI answer processor, MedXpertQA's two-turn
Text + MM protocol, MMMU-Pro's official prompts / parser / settings, coco_caption's
protocol and flags, bundled_detection's mAP denominator, and screenspot_pointing's honest
labelling. No model server is contacted: loaders get fake rows and run_eval_suite is
replaced where a suite's post-processing is under test.
"""

import os
from unittest import mock

import pytest
from PIL import Image

from gbench.runners.eval_suites import (bundled_detection, chartqa, coco_caption, medxpertqa,
                                        mmmu_pro, screenspot_pointing, textvqa)
from gbench.runners.eval_suites.sampling import parse_shard, shard_select
from gbench.runners.eval_suites.vqa_common import (eval_relaxed, normalize_answer,
                                                   relaxed_correctness, vqa_accuracy)


def _patch_hf(fn):
    return mock.patch("datasets.load_dataset", autospec=True, side_effect=fn)


class _FakeDS:
    """Enough of `datasets.Dataset` for the loaders: column reads, row reads, iteration."""

    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, key):
        if isinstance(key, str):
            return [r.get(key) for r in self.rows]
        return self.rows[key]


# --------------------------------------------------------------------------- #
# chartqa: pix2struct relaxed_correctness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pred,gold,expected", [
    ("Between 20 and 30", "20", False),   # was True under the first-number scan
    ("3", "Q3 2019", False),
    ("2019: 45", "2019", False),
    ("0.5", "50%", True),                 # trailing % divides by 100
    ("103", "100", True),                 # within 5%
    ("106", "100", False),
    ("Yes", "yes", True),                 # case-insensitive exact match otherwise
    ("0", "0", True),                     # zero gold falls back to string equality
    ("0.0", "0", False),
    ("1234", "1,234", False),             # no normalization of either side
])
def test_chartqa_relaxed_correctness_matches_pix2struct(pred, gold, expected):
    assert relaxed_correctness(gold, pred) is expected
    assert eval_relaxed(pred, [gold]) is expected


def test_chartqa_relaxed_takes_best_over_golds_and_scores_the_reply_as_returned():
    assert eval_relaxed("42", ["forty-two", "42"]) is True
    assert chartqa._eval_chartqa("41", ["forty-two", "42"]) is True          # within 5%
    # lmms-eval chartqa scores results[0]: no answer extraction
    assert chartqa._eval_chartqa("The chart peaks in 2019.\nAnswer: 41", ["42"]) is False
    assert chartqa._eval_chartqa("The value is 1000 not 100", ["100"]) is False


def test_chartqa_loader_splits_by_half_and_refuses_imageless_rows():
    img = Image.new("RGB", (4, 4))
    rows = [{"query": f"q{i}", "label": "1", "type": ("human" if i % 2 else "augmented"),
             "image": img, "imgname": f"{i}.png"} for i in range(10)]
    calls = []

    def fake(path, *args, **kwargs):
        calls.append((path, kwargs.get("revision")))
        return rows

    with _patch_hf(fake):
        samples = chartqa._load_chartqa_samples(limit=4)
    assert calls == [(chartqa.DATASET, chartqa.DATASET_REVISION)]
    cats = sorted(meta["category"] for _m, _g, meta in samples)
    assert cats == ["augmented", "augmented", "human", "human"]

    bad = rows[:1] + [{"query": "q", "label": "1", "type": "human", "image": None, "imgname": "x"}]
    with _patch_hf(lambda *a, **k: bad):
        with pytest.raises(RuntimeError, match="no decodable image"):
            chartqa._load_chartqa_samples()


# --------------------------------------------------------------------------- #
# textvqa: EvalAI answer processor
# --------------------------------------------------------------------------- #
def test_textvqa_uses_the_evalai_processor():
    assert normalize_answer("coca-cola") == "coca cola"
    assert normalize_answer("10:30") == "10:30"          # ':' is not in the punctuation list
    assert normalize_answer("dont") == "don't"           # contraction map
    assert normalize_answer("The  Two Dogs.") == "2 dogs"
    assert normalize_answer("3.5") == "3.5"              # a period between digits stays
    assert vqa_accuracy("coca cola", ["coca-cola"] * 10) == 1.0
    assert textvqa._eval_textvqa("Coca Cola", ["coca-cola"] * 10) is True


def test_textvqa_accuracy_is_leave_one_out_over_ten():
    golds = ["stop"] * 3 + ["go"] * 7
    # 3 annotators said "stop": each held-out "stop" leaves 2 matches (2/3), the 7 others 3 (1.0)
    assert vqa_accuracy("stop", golds) == pytest.approx((3 * (2 / 3) + 7 * 1.0) / 10)
    assert vqa_accuracy("yield", golds) == 0.0


def test_textvqa_loader_refuses_imageless_rows():
    rows = [{"question": "q", "answers": ["a"] * 10, "image": None, "question_id": 7}]
    with _patch_hf(lambda *a, **k: rows):
        with pytest.raises(RuntimeError, match="no decodable image"):
            textvqa._load_textvqa_samples()


# --------------------------------------------------------------------------- #
# medxpertqa: Text + MM, two-turn zero-shot CoT
# --------------------------------------------------------------------------- #
def _medx_rows(subset):
    if subset == "Text":
        return _FakeDS([{"id": "Text-0", "question": "Which?\nAnswer Choices: (A) a (B) b (C) c (D) d (E) e "
                         "(F) f (G) g (H) h (I) i (J) j", "options": {c: c.lower() for c in "ABCDEFGHIJ"},
                         "label": "J", "question_type": "Reasoning", "medical_task": "x"}])
    return _FakeDS([{"id": "MM-0", "question": "  Look.\nAnswer Choices: (A) a (B) b (C) c (D) d (E) e ",
                     "options": {c: c.lower() for c in "ABCDE"}, "label": "B",
                     "images": ["MM-0-a.jpeg", "MM-0-b.jpeg"], "question_type": "Understanding"}])


@pytest.fixture
def medx_images(tmp_path):
    for name in ("MM-0-a.jpeg", "MM-0-b.jpeg"):
        (tmp_path / name).write_bytes(b"\xff\xd8\xff" + name.encode())
    with mock.patch.object(medxpertqa, "_medxpertqa_image_dir", return_value=str(tmp_path)):
        yield tmp_path


def _fake_medx(path, name=None, **kwargs):
    assert path == medxpertqa.DATASET and kwargs.get("revision") == medxpertqa.DATASET_REVISION
    return _medx_rows(name)


def test_medxpertqa_loads_both_subsets_with_the_upstream_prompt(medx_images):
    with _patch_hf(_fake_medx):
        samples = medxpertqa._load_medxpertqa_samples()
    by_id = {meta["medxpertqa_id"]: (msgs, gold, meta) for msgs, gold, meta in samples}
    assert set(by_id) == {"Text-0", "MM-0"}

    msgs, gold, meta = by_id["MM-0"]
    assert msgs[0] == {"role": "system", "content": "You are a helpful medical assistant."}
    content = msgs[1]["content"]
    # question stripped, options shown once (they are already in the question)
    assert content[0] == {"type": "text", "text": "Q: Look.\nAnswer Choices: (A) a (B) b (C) c (D) d (E) e\n"
                                                  "A: Let's think step by step."}
    assert "Options:" not in content[0]["text"]
    assert [c["type"] for c in content] == ["text", "image_url", "image_url"]   # images after text
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert gold == {"label": "B", "n_options": 5}
    assert meta["category"] == "MM/Understanding"
    assert by_id["Text-0"][1] == {"label": "J", "n_options": 10}


def test_medxpertqa_missing_image_or_zip_is_a_hard_error(tmp_path):
    with mock.patch.object(medxpertqa, "_medxpertqa_image_dir", return_value=str(tmp_path)):
        with _patch_hf(_fake_medx):
            with pytest.raises(RuntimeError, match="INFRA REQUIRED.*missing from images.zip"):
                medxpertqa._load_medxpertqa_samples()
    medxpertqa._medxpertqa_image_dir.cache_clear()
    with mock.patch("huggingface_hub.hf_hub_download", side_effect=OSError("offline")):
        with pytest.raises(RuntimeError, match="INFRA REQUIRED.*images.zip"):
            medxpertqa._medxpertqa_image_dir()
    medxpertqa._medxpertqa_image_dir.cache_clear()


@pytest.mark.parametrize("reply,gold,expected", [
    ("(B)", {"label": "B", "n_options": 5}, True),
    ("Therefore, among A through E, the answer is C, not B.", {"label": "C", "n_options": 5}, True),
    ("B. Although C is tempting.", {"label": "B", "n_options": 5}, True),      # first letter wins
    ("Although C is tempting, B.", {"label": "B", "n_options": 5}, False),
    ("The answer is (F)", {"label": "F", "n_options": 5}, False),              # F is not an MM option
    ("The answer is (F)", {"label": "F", "n_options": 10}, True),
    ("I cannot tell.", {"label": "A", "n_options": 5}, False),                 # no standalone letter
])
def test_medxpertqa_answer_cleansing(reply, gold, expected):
    assert medxpertqa._eval_medxpertqa(reply, gold) is expected


def test_medxpertqa_runs_two_turns_and_counts_rationale_failures(medx_images, monkeypatch):
    calls = []

    def fake_run(**kw):
        calls.append(kw)
        if len(calls) == 1:                      # turn 1: rationales
            assert kw["async_eval_fn"] is medxpertqa._hold_for_answer_turn
            assert kw["supports_attempts"] is True
            traces = []
            for i, (msgs, gold, meta) in enumerate(kw["samples"]):
                ok = meta["medxpertqa_id"] == "MM-0"
                traces.append({"sample_idx": i, "attempt_idx": 0, "category": meta["category"],
                               "gold_answer": gold, "extra_payload": dict(meta),
                               "response_text": "Some reasoning." if ok else None,
                               "status": "OK" if ok else "FAILED", "is_correct": False,
                               # as base marks a request that got no reply: a harness error
                               **({} if ok else {"scoring_excluded": True, "harness_error": {
                                   "kind": "endpoint_failure", "detail": "HTTP 500"}})})
            return {"sample_traces": traces, "failed_requests": 1, "duration_s": 1.0,
                    "attempts": None, "temperature": 0.0}
        # turn 2: answer trigger. GBENCH_SHARD stays set (never popped from the process env),
        # and base's shard selection keeps every turn-2 sample.
        assert os.environ["GBENCH_SHARD"] == "2/3"
        assert kw["attempt_count"] == 1
        (msgs, gold, meta), = shard_select(kw["samples"], parse_shard(os.environ["GBENCH_SHARD"]))
        assert msgs[-2] == {"role": "assistant", "content": "Some reasoning."}
        assert msgs[-1] == {"role": "user", "content": "Therefore, among A through E, the answer is"}
        return {"sample_traces": [{"category": meta["category"], "extra_payload": dict(meta),
                                   "gold_answer": gold, "response_text": "B", "is_correct": True,
                                   "status": "OK"}],
                "failed_requests": 0, "duration_s": 2.0, "temperature": 0.0, "accuracy": 100.0}

    monkeypatch.setattr(medxpertqa, "run_eval_suite", fake_run)
    monkeypatch.setenv("GBENCH_SHARD", "2/3")
    with _patch_hf(_fake_medx):
        res = medxpertqa.run_medxpertqa("m", "http://x/v1")
    assert os.environ["GBENCH_SHARD"] == "2/3"
    assert res["total_questions"] == 2 and res["correct_answers"] == 1
    # the rationale request with no reply is a harness error, left out of every accuracy
    assert res["accuracy"] == 100.0 and res["scoring_excluded"] == 1
    assert res["medxpertqa_mm_accuracy"] == 100.0 and res["medxpertqa_text_accuracy"] is None
    assert res["failed_requests"] == 1 and res["status"] == "completed_with_errors"
    assert res["leaderboard_comparable"] is False and "GBENCH_SHARD" in res["leaderboard_comparable_reason"]


def test_medxpertqa_all_rationales_failed_is_a_failed_run(medx_images, monkeypatch):
    calls = []

    def fake_run(**kw):
        calls.append(kw)
        # as base marks a request that got no reply from the endpoint: a harness error
        traces = [{"sample_idx": i, "category": meta["category"], "gold_answer": gold,
                   "extra_payload": dict(meta), "response_text": None, "status": "FAILED",
                   "is_correct": False, "scoring_excluded": True,
                   "harness_error": {"kind": "endpoint_failure", "detail": "HTTP 500"}}
                  for i, (_m, gold, meta) in enumerate(kw["samples"])]
        return {"sample_traces": traces, "failed_requests": len(traces), "temperature": 0.0}

    monkeypatch.setattr(medxpertqa, "run_eval_suite", fake_run)
    with _patch_hf(_fake_medx):
        res = medxpertqa.run_medxpertqa("m", "http://x/v1")
    assert len(calls) == 1                               # no answer turn without a rationale
    # every question a harness error: the score is withheld, never 0
    assert res["total_questions"] == 2 and res["accuracy"] is None
    assert res["accuracy_withheld_reason"]
    assert res["failed_requests"] == 2 and res["status"] == "failed"


# --------------------------------------------------------------------------- #
# mmmu_pro: official prompts, image order, parser, both settings
# --------------------------------------------------------------------------- #
def _mmmu_ds(name):
    red, blue = Image.new("RGB", (2, 2), "red"), Image.new("RGB", (2, 2), "blue")
    if name == mmmu_pro.STANDARD:
        return _FakeDS([{"id": "test_Art_1", "question": "Which of <image 2> and <image 1>?",
                         "options": "['<image 1>', 'Neither']", "answer": "A", "subject": "Art",
                         "image_1": red, "image_2": blue}])
    return _FakeDS([{"id": "test_Art_1", "image": red, "options": "['<image 1>', 'Neither']",
                     "answer": "A", "subject": "Art"}])


def _fake_mmmu(path, name=None, **kwargs):
    assert path == mmmu_pro.DATASET and kwargs.get("revision") == mmmu_pro.DATASET_REVISION
    return _mmmu_ds(name)


def test_mmmu_pro_builds_the_official_prompts_for_both_settings():
    with _patch_hf(_fake_mmmu):
        samples = mmmu_pro._load_mmmu_pro_samples(modes=("cot", "direct"))
    by_cat = {meta["category"]: msgs[-1]["content"] for msgs, _g, meta in samples}
    # infer_gpt.py request_with_images: its system turn, then the user turn
    assert all(msgs[0] == {"role": "system", "content": mmmu_pro.SYSTEM_MESSAGE}
               and msgs[1]["role"] == "user" and len(msgs) == 2 for msgs, _g, _m in samples)
    assert set(by_cat) == {"standard_10/cot", "standard_10/direct", "vision/cot", "vision/direct"}

    std = by_cat["standard_10/cot"]
    assert std[0]["text"] == ("Which of <image> and <image>?\nA. <image>\nB. Neither\n"
                              + mmmu_pro.PROMPTS["cot"]["standard"])
    # text first, then images in reference order: <image 2>, <image 1>, <image 1> (option A)
    assert [c["type"] for c in std] == ["text", "image_url", "image_url", "image_url"]
    urls = [c["image_url"]["url"] for c in std[1:]]
    assert urls[0] != urls[1] and urls[1] == urls[2]
    assert by_cat["standard_10/direct"][0]["text"].endswith(
        "\nAnswer with the option letter from the given choices directly.")
    vis = by_cat["vision/cot"]
    assert vis[0] == {"type": "text", "text": mmmu_pro.PROMPTS["cot"]["vision"]}
    assert [c["type"] for c in vis] == ["text", "image_url"]


@pytest.mark.parametrize("reply,expected", [
    ("Reasoning...\nAnswer: B", "B"),
    ("Answer: (A) or maybe B", "A"),       # "Answer:" tail names two letters -> the bracketed one
    ("I pick (B) over (A).", "A"),
    ("The painting is Neither of these, clearly.", "B"),   # option-text match
])
def test_mmmu_pro_parse_multi_choice_response(reply, expected):
    index2ans, choices = mmmu_pro.get_multi_choice_info(["<image 1>", "Neither"])
    assert mmmu_pro.parse_multi_choice_response(reply, choices, index2ans) == expected


def test_mmmu_pro_random_fallback_is_reproducible():
    gold = {"answer": "C", "options": [str(i) for i in range(10)], "id": "q1"}
    first = [mmmu_pro._eval_mmmu_pro("no idea", gold) for _ in range(5)]
    assert len(set(first)) == 1
    assert mmmu_pro._eval_mmmu_pro("A careful analysis shows option C is right.", "A") is False
    assert mmmu_pro._eval_mmmu_pro("Answer: A", "A") is True


def test_mmmu_pro_headline_is_mean_of_best_mode_per_setting(monkeypatch):
    def fake_run(**kw):
        assert "temperature" not in kw                     # infer_gpt.py sends none
        assert kw["official_sampling"].send_temperature is False
        assert all("top_p" not in meta for _m, _g, meta in kw["samples"])
        hits = {"standard_10/cot": True, "standard_10/direct": False,
                "vision/cot": False, "vision/direct": True}
        traces = [{"category": meta["category"], "is_correct": hits[meta["category"]]}
                  for _m, _g, meta in kw["samples"]]
        return {"sample_traces": traces, "accuracy": 50.0}

    monkeypatch.setattr(mmmu_pro, "run_eval_suite", fake_run)
    with _patch_hf(_fake_mmmu):
        res = mmmu_pro.run_mmmu_pro("m", "http://x/v1", 1)
    assert res["mmmu_pro_standard_10"] == 100.0 and res["mmmu_pro_vision"] == 100.0
    assert res["accuracy"] == 100.0 and res["pooled_accuracy"] == 50.0
    assert res["leaderboard_comparable"] is True

    monkeypatch.setenv("GBENCH_MMMU_PRO_PROMPT", "cot")
    with _patch_hf(_fake_mmmu):
        res = mmmu_pro.run_mmmu_pro("m", "http://x/v1", 1)
    assert res["accuracy"] == 50.0                         # standard 100 (cot), vision 0 (cot)
    assert res["leaderboard_comparable"] is False
    assert "higher of CoT and Direct" in res["leaderboard_comparable_reason"]

    monkeypatch.setenv("GBENCH_MMMU_PRO_PROMPT", "both-ish")
    with pytest.raises(ValueError):
        mmmu_pro._prompt_modes()


# --------------------------------------------------------------------------- #
# coco_caption: lmms-eval coco2014_cap_val protocol, honest flag, Java checked up front
# --------------------------------------------------------------------------- #
def test_coco_caption_prompt_and_flags(monkeypatch):
    img = Image.new("RGB", (4, 4))
    rows = [{"image": img, "file_name": "COCO_val2014_000000000003.jpg",
             "answer": ["A cat.", "A small cat."], "id": 3}]
    split = [{"filename": "COCO_val2014_000000000003.jpg", "cocoid": 3,
              "sentences": ["A cat.", "A small cat."]}]
    monkeypatch.setattr(coco_caption, "_N_IMAGES", 1)
    with _patch_hf(lambda path, *a, **k: split if path == "yerevann/coco-karpathy" else rows):
        (msgs, refs, meta), = coco_caption._load_coco_caption_samples()
    assert msgs[0]["content"][1] == {"type": "text", "text": "Describe the image briefly."}

    monkeypatch.setattr(coco_caption, "check_coco_caption_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(coco_caption, "_load_coco_caption_samples", lambda limit=None: [])
    monkeypatch.setattr(coco_caption, "run_eval_suite", lambda **kw: {"accuracy": 12.0})
    res = coco_caption.run_coco_caption("m", "http://x/v1")
    # A full, unsharded Karpathy-test run at temperature 0 is the lmms-eval protocol.
    assert res["leaderboard_comparable"] is True
    assert res["leaderboard_comparable_reason"] is None


def test_coco_caption_requires_java_before_generating(monkeypatch):
    real_which = coco_caption.shutil.which
    monkeypatch.setattr(coco_caption.shutil, "which",
                        lambda name: None if name == "java" else real_which(name))
    ok, reason = coco_caption.check_coco_caption_prerequisites()
    assert ok is False and "java" in reason


# --------------------------------------------------------------------------- #
# bundled_detection: failed images keep their ground truth in the mAP
# --------------------------------------------------------------------------- #
def test_bundled_detection_failed_image_stays_in_the_map(monkeypatch):
    box = [100.0, 100.0, 400.0, 400.0]
    good = {"gold_answer": [{"label": "cat", "box_2d": box}],
            "response_text": '[{"label": "cat", "box_2d": [100, 100, 400, 400], "confidence": 0.9}]'}
    failed = {"gold_answer": [{"label": "cat", "box_2d": box}], "response_text": None}
    assert bundled_detection._compute_map([good])["map"] == pytest.approx(1.0)
    # the failed image's cat is still ground truth: recall tops out at 1/2
    assert bundled_detection._compute_map([good, failed])["map"] == pytest.approx(51 / 101)

    monkeypatch.setattr(bundled_detection, "_load_bundled_detection_samples", lambda limit=None: [])
    monkeypatch.setattr(bundled_detection, "run_eval_suite",
                        lambda **kw: {"accuracy": 50.0, "sample_traces": [good, failed]})
    res = bundled_detection.run_bundled_detection("m", "http://x/v1", 1)
    assert res["map_50_95"] == round(51 / 101 * 100, 2) == res["accuracy"]
    assert "failures excluded" not in res["leaderboard_comparable_reason"]
    assert res["leaderboard_comparable"] is False


# --------------------------------------------------------------------------- #
# screenspot_pointing (renamed from semantic_keypoint): honest label, flagged, renameable
# --------------------------------------------------------------------------- #
def test_screenspot_pointing_is_flagged_and_accepts_a_new_name(monkeypatch):
    seen = {}

    def fake_run(**kw):
        seen["eval_name"] = kw["eval_name"]
        return {"accuracy": 40.0}

    monkeypatch.setattr(screenspot_pointing, "_load_screenspot_pointing_samples",
                        lambda enable_thinking=False, limit=None: [])
    monkeypatch.setattr(screenspot_pointing, "run_eval_suite", fake_run)
    res = screenspot_pointing.run_screenspot_pointing("m", "http://x/v1", 1)
    assert seen["eval_name"] == "screenspot_pointing"
    assert res["leaderboard_comparable"] is False and "screenspot" in res["leaderboard_comparable_reason"]
    screenspot_pointing.run_screenspot_pointing("m", "http://x/v1", 1, eval_name="semantic_keypoint")
    assert seen["eval_name"] == "semantic_keypoint"


def test_screenspot_pointing_refuses_rows_without_image_or_bbox(tmp_path, monkeypatch):
    import json as _json
    import zipfile as _zipfile
    rows = [{"instruction": "Click OK", "img_filename": "a.png", "bbox": None, "data_type": "text"}]
    files = {}
    for platform, (name, _sha) in screenspot_pointing.SPLITS.items():
        files[name] = tmp_path / name
        files[name].write_text(_json.dumps(rows if platform == "mobile" else []))
    files[screenspot_pointing.IMAGE_ZIP[0]] = tmp_path / "img.zip"
    with _zipfile.ZipFile(files[screenspot_pointing.IMAGE_ZIP[0]], "w"):
        pass
    monkeypatch.setattr(screenspot_pointing, "_fetch", lambda name, sha: str(files[name]))
    monkeypatch.setattr(screenspot_pointing, "SPLIT_SIZES", {"mobile": 1, "desktop": 0, "web": 0})
    with pytest.raises(RuntimeError, match="no image or bbox"):
        screenspot_pointing._load_screenspot_pointing_samples()


# --------------------------------------------------------------------------- #
# Markdown around the stated answer (vqa_common.extract_short_answer)
# --------------------------------------------------------------------------- #
from gbench.runners.eval_suites import docvqa                       # noqa: E402
from gbench.runners.eval_suites.vqa_common import anls_score, extract_short_answer  # noqa: E402


@pytest.mark.parametrize("reply,expected", [
    ("The chart shows the value.\n\n**Answer:** 14", "14"),   # the label's closing marker
    ("Answer: **14**", "14"),
    ("**Answer**: 14", "14"),
    ("__Answer:__ 14", "14"),
    ("**Final Answer:** `STOP`", "STOP"),
    ("Answer:\n**14**", "14"),
    ("Answer: ** 14 **", "14"),
    ("**14**", "14"),                                         # last-line path
    ("**Total:** 7", "7"),                                    # short label with emphasis
    ("Final value: 42.5%", "42.5%"),
    ("Answer: LAAP*", "LAAP*"),                               # a gold's own footnote asterisk
    ("Answer: **LAAP***", "LAAP*"),                           # one matching pair at a time
    ("*Victor L. Johnson", "*Victor L. Johnson"),             # unpaired: part of the answer
    ("Answer: 3*15", "3*15"),
    ("Yes.", "Yes."),                                         # trailing period kept
    ("Answer: U.S.", "U.S."),
    ("The total is 42.", "The total is 42."),
])
def test_extract_short_answer_accepts_markdown(reply, expected):
    assert extract_short_answer(reply) == expected


@pytest.mark.parametrize("reply", ["**Answer:** 14", "Answer: **14**", "**Answer**: `14`"])
def test_markdown_answers_are_scored_as_returned_on_chartqa_textvqa_and_docvqa(reply):
    # lmms-eval scores results[0] as the endpoint returned it, markdown and label included.
    assert chartqa._eval_chartqa(reply, ["14"]) is False
    assert textvqa._eval_textvqa(reply, ["14"] * 10) is False
    assert docvqa._eval_docvqa(reply, ["14"]) is False
    # extract_short_answer itself is unchanged (kept for other callers)
    assert vqa_accuracy(extract_short_answer(reply), ["14"] * 10) == 1.0
    assert anls_score(extract_short_answer(reply), ["14"]) == 1.0


def test_bare_answers_score_and_wrong_answers_stay_wrong():
    assert chartqa._eval_chartqa("14", ["14"]) is True
    assert textvqa._eval_textvqa("go", ["stop"] * 10) is False
    assert chartqa._eval_chartqa("LAAP", ["LAAP*"]) is False              # no normalization
    assert chartqa._eval_chartqa("LAAP*", ["LAAP*"]) is True
    # relaxed_correctness compares the answer as given: a trailing period is not removed for
    # ChartQA, while TextVQA's EvalAI processor drops it on both sides.
    assert chartqa._eval_chartqa("Yes.", ["Yes"]) is False
    assert textvqa._eval_textvqa("Yes.", ["yes"] * 10) is True


# --------------------------------------------------------------------------- #
# chartqa / textvqa state that the prompt and extraction are gbench's
# --------------------------------------------------------------------------- #
def test_chartqa_and_textvqa_comparability_follows_the_protocol(monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    trace = {"gold_answer": ["14"] * 10, "response_text": "14", "completion_tokens": 2}
    for mod, loader, n in ((chartqa, "_load_chartqa_samples", "2,500"),
                           (textvqa, "_load_textvqa_samples", "5,000")):
        seen = {}

        def fake_run(**kw):
            seen.update(kw)
            return {"accuracy": 100.0, "sample_traces": [dict(trace)], "temperature": 0.0,
                    "official_sampling": {"temperature_matches_official": True}}

        monkeypatch.setattr(mod, loader, lambda limit=None: [])
        monkeypatch.setattr(mod, "run_eval_suite", fake_run)
        run = getattr(mod, f"run_{mod.__name__.rsplit('.', 1)[-1]}")
        res = run("m", "http://x/v1")
        assert res["leaderboard_comparable"] is True and res["leaderboard_comparable_reason"] is None
        assert seen["max_output_tokens"] is None              # base's floor governs when unset
        assert seen["official_sampling"].temperature == 0.0
        res = run("m", "http://x/v1", limit=20)
        assert res["leaderboard_comparable"] is False
        assert f"--eval-limit 20 subset of the {n} questions" in res["leaderboard_comparable_reason"]


def test_chartqa_and_textvqa_docs_match_the_flag():
    for name in ("chartqa", "textvqa"):
        with open(os.path.join(os.path.dirname(__file__), "..", "docs", "evals", f"{name}.md"),
                  encoding="utf-8") as f:
            doc = f.read()
        assert "sets no `leaderboard_comparable` flag" not in doc
        assert "`leaderboard_comparable` is `true` for a full, unsharded run at the official" in doc
        assert "no answer extraction" in doc and "`replies_at_official_max_tokens`" in doc


# --------------------------------------------------------------------------- #
# medxpertqa turn 2 (no env mutation, empty replies, budget)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", [None, "1/1", "1/4", "3/4", "4/4", "2/7"])
def test_medxpertqa_presharded_survives_base_shard_selection(monkeypatch, spec):
    if spec is None:
        monkeypatch.delenv("GBENCH_SHARD", raising=False)
    else:
        monkeypatch.setenv("GBENCH_SHARD", spec)
    items = [("m", {"label": "A"}, {"i": i}) for i in range(5)]
    laid = medxpertqa._presharded(items)
    assert shard_select(laid, parse_shard(os.environ.get("GBENCH_SHARD"))) == items
    assert os.environ.get("GBENCH_SHARD") == spec           # the environment is not touched


def test_medxpertqa_empty_rationale_still_gets_the_trigger(medx_images, monkeypatch):
    calls = []

    def fake_run(**kw):
        calls.append(kw)
        if len(calls) == 1:
            traces = []
            for i, (_m, gold, meta) in enumerate(kw["samples"]):
                # Text-0: 200 with empty content (reasoning used the budget); MM-0: request failed
                empty = meta["medxpertqa_id"] == "Text-0"
                traces.append({"sample_idx": i, "category": meta["category"], "gold_answer": gold,
                               "extra_payload": dict(meta), "response_text": "" if empty else None,
                               "status": "EMPTY_RESPONSE" if empty else "FAILED",
                               "is_correct": False})
            return {"sample_traces": traces, "failed_requests": 2, "empty_responses": 1,
                    "temperature": 0.0}
        (msgs, gold, meta), = kw["samples"]
        assert meta["medxpertqa_id"] == "Text-0"
        assert msgs[-2] == {"role": "assistant", "content": ""}
        assert msgs[-1] == {"role": "user", "content": "Therefore, among A through J, the answer is"}
        return {"sample_traces": [{"category": meta["category"], "extra_payload": dict(meta),
                                   "gold_answer": gold, "response_text": "J", "is_correct": True,
                                   "status": "OK"}], "failed_requests": 0, "temperature": 0.0}

    monkeypatch.setattr(medxpertqa, "run_eval_suite", fake_run)
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    with _patch_hf(_fake_medx):
        res = medxpertqa.run_medxpertqa("m", "http://x/v1")
    assert len(calls) == 2
    assert res["total_questions"] == 2 and res["correct_answers"] == 1 and res["accuracy"] == 50.0
    assert res["medxpertqa_text_accuracy"] == 100.0 and res["medxpertqa_mm_accuracy"] == 0.0
    # the empty turn-1 reply is still reported in the summed counters
    assert res["empty_responses"] == 1 and res["failed_requests"] == 2


def test_medxpertqa_passes_the_flag_and_declares_its_caps_as_official(medx_images, monkeypatch):
    seen, caps = [], []

    def fake_run(**kw):
        seen.append(kw["max_output_tokens"])
        official = kw["official_sampling"]
        caps.append((official.max_tokens, official.reasoning))
        traces = [{"sample_idx": i, "category": meta["category"], "gold_answer": gold,
                   "extra_payload": dict(meta), "response_text": None, "status": "FAILED",
                   "is_correct": False} for i, (_m, gold, meta) in enumerate(kw["samples"])]
        return {"sample_traces": traces, "failed_requests": len(traces), "temperature": 0.0}

    monkeypatch.setattr(medxpertqa, "run_eval_suite", fake_run)
    with _patch_hf(_fake_medx):
        medxpertqa.run_medxpertqa("m", "http://x/v1")
        medxpertqa.run_medxpertqa("m", "http://x/v1", enable_thinking=True)
        medxpertqa.run_medxpertqa("m", "http://x/v1", max_output_tokens=65536)
    assert seen == [None, None, 65536]          # the flag goes to base unchanged
    # api_agent.py's 2048 (the chat branch a served model takes) is the official cap in both
    # modes: base's token-limit policy applies it (not a suite default that the flag overrides).
    # The 8192 upstream gives model names containing o1, o3, qwq, qvq or reasoner is not declared
    # as a reasoning-model setting, so a --thinking run gets the run's budget from the policy.
    assert set(caps) == {(2048, None)}


# --------------------------------------------------------------------------- #
# Comparability reasons describe what was scored (--eval-limit subsets)
# --------------------------------------------------------------------------- #
def test_bundled_detection_reason_counts_the_scored_images(monkeypatch):
    trace = {"gold_answer": [], "response_text": "[]"}
    monkeypatch.setattr(bundled_detection, "_load_bundled_detection_samples", lambda limit=None: [])
    monkeypatch.setattr(bundled_detection, "run_eval_suite",
                        lambda **kw: {"accuracy": 0.0, "sample_traces": [dict(trace)] * 20})
    res = bundled_detection.run_bundled_detection("m", "http://x/v1", 1, limit=20)
    reason = res["leaderboard_comparable_reason"]
    assert "on 20 of the 5,000 val2017 images (--eval-limit 20 subset)" in reason
    monkeypatch.setattr(bundled_detection, "run_eval_suite",
                        lambda **kw: {"accuracy": 0.0, "sample_traces": [dict(trace)] * 5000})
    reason = bundled_detection.run_bundled_detection("m", "http://x/v1", 1)["leaderboard_comparable_reason"]
    assert "on 5,000 of the 5,000 val2017 images" in reason and "--eval-limit" not in reason


def test_coco_caption_reason_counts_the_scored_images(monkeypatch):
    monkeypatch.setattr(coco_caption, "check_coco_caption_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(coco_caption, "_load_coco_caption_samples", lambda limit=None: [])
    monkeypatch.setattr(coco_caption, "run_eval_suite",
                        lambda **kw: {"accuracy": 12.0, "sample_traces": [{}] * 500})
    reason = coco_caption.run_coco_caption("m", "http://x/v1", limit=500)["leaderboard_comparable_reason"]
    assert "--eval-limit 500 subset (500 of the 5,000 Karpathy test images" in reason
    monkeypatch.setattr(coco_caption, "run_eval_suite",
                        lambda **kw: {"accuracy": 12.0, "sample_traces": [{}] * 5000})
    reason = coco_caption.run_coco_caption("m", "http://x/v1")["leaderboard_comparable_reason"]
    assert reason is None


def test_comparability_reasons_count_an_image_once_across_attempts(monkeypatch):
    traces = [{"gold_answer": [], "response_text": "[]", "source_sample_idx": i // 3,
               "extra_payload": {"image_id": i // 3}} for i in range(30)]      # 10 images x 3
    monkeypatch.setattr(bundled_detection, "_load_bundled_detection_samples", lambda limit=None: [])
    monkeypatch.setattr(bundled_detection, "run_eval_suite",
                        lambda **kw: {"accuracy": 0.0, "sample_traces": traces})
    reason = bundled_detection.run_bundled_detection("m", "http://x/v1", 1, limit=10)[
        "leaderboard_comparable_reason"]
    assert "on 10 of the 5,000 val2017 images (--eval-limit 10 subset" in reason
    monkeypatch.setattr(coco_caption, "check_coco_caption_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(coco_caption, "_load_coco_caption_samples", lambda limit=None: [])
    monkeypatch.setattr(coco_caption, "run_eval_suite",
                        lambda **kw: {"accuracy": 1.0, "sample_traces": traces})
    reason = coco_caption.run_coco_caption("m", "http://x/v1", limit=10)["leaderboard_comparable_reason"]
    assert reason.startswith("--eval-limit 10 subset (10 of the 5,000 Karpathy test images")
