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

"""Unit tests for native evaluation suites: their scoring functions and sample loaders."""

import json
from unittest import mock
import pytest
from PIL import Image

from gbench.runners.eval_suites.arc_agi import _eval_arc_agi, _load_arc_agi_samples
from gbench.runners.eval_suites.docvqa import _eval_docvqa, _load_docvqa_samples
from gbench.runners.eval_suites.chartqa import _eval_chartqa, _load_chartqa_samples
from gbench.runners.eval_suites.medxpertqa import _eval_medxpertqa, _load_medxpertqa_samples
from gbench.runners.eval_suites.longbench_v2_code import (
    _eval_longbench_v2_code, _load_longbench_v2_code_samples)
from gbench.runners.eval_suites.imo_answer_bench import _load_imo_answer_bench_samples, parse_grader_verdict
from gbench.runners.eval_suites.multilingual_mmlu import _eval_multilingual_mmlu, _load_multilingual_mmlu_samples
from gbench.runners.eval_suites.coco_caption import _load_coco_caption_samples


def _patch_hf(rows_by_dataset):
    def _fake_load_dataset(path, *args, **kwargs):
        if path not in rows_by_dataset:
            raise AssertionError(f"unexpected dataset: {path}")
        return rows_by_dataset[path]
    return mock.patch("datasets.load_dataset", autospec=True, side_effect=_fake_load_dataset)


def test_arc_agi():
    # ARC-AGI-1 grid puzzles: few-shot train pairs + a test input grid; gold is the
    # test output grid, scored by exact grid match.
    gold_grid = [[1, 2], [3, 4]]
    dummy = [{"id": "t0", "train": [{"input": [[0]], "output": [[1]]}],
              "test": [{"input": [[5, 6], [7, 8]], "output": gold_grid}]}]
    # The loader reads the official fchollet/ARC-AGI files (verified and cached); stand in for them.
    with mock.patch("gbench.runners.eval_suites.arc_agi._load_arc_agi_tasks", return_value=dummy):
        samples = _load_arc_agi_samples(limit=1)
        assert len(samples) == 1
        assert samples[0][1] == gold_grid
    assert _eval_arc_agi(json.dumps(gold_grid), gold_grid) is True
    assert _eval_arc_agi("the answer is [[9, 9], [9, 9]]", gold_grid) is False
    assert _eval_arc_agi("no grid here", gold_grid) is False


def test_docvqa():
    img = Image.new("RGB", (10, 10), color="white")
    dummy = [{"question": "Total amount?", "answers": ["$100.50"], "image": img, "data_type": "invoice"}]
    with _patch_hf({"lmms-lab-encoder/DocVQA": dummy, "hf-internal-testing/fixtures_docvqa": dummy}):
        samples = _load_docvqa_samples(limit=1)
        assert len(samples) == 1
    assert _eval_docvqa("$100.50", ["$100.50"]) is True
    # Canonical DocVQA ANLS = 1 - normalized Levenshtein at threshold 0.5 (no numeric
    # special-case): a clearly different string scores below 0.5 -> False.
    assert _eval_docvqa("no matching value here", ["$100.50"]) is False


def test_chartqa():
    img = Image.new("RGB", (10, 10), color="blue")
    dummy = [{"query": "What is the highest value?", "label": ["45.2%"], "image": img}]
    with _patch_hf({"ahmed-masry/ChartQA": dummy, "HuggingFaceM4/ChartQA": dummy}):
        samples = _load_chartqa_samples(limit=1)
        assert len(samples) == 1
    assert _eval_chartqa("45.2%", ["45.2%"]) is True
    assert _eval_chartqa("45.0%", ["45.2%"]) is True   # both parse as percentages, within 5%
    assert _eval_chartqa("10.0", ["45.2%"]) is False


def test_medxpertqa():
    dummy = [{"question": "Diagnosis?", "options": {"A": "Asthma", "B": "COPD"}, "answer": "A", "specialty": "Pulmonology"}]
    with _patch_hf({"TsinghuaC3I/MedXpertQA": dummy, "wish6424/MedXpertQA-Diagnosis": dummy}):
        samples = _load_medxpertqa_samples(limit=1)
        assert len(samples) == 1
    assert _eval_medxpertqa("Answer: (A)", "A") is True
    assert _eval_medxpertqa("Answer: (B)", "A") is False


def test_longbench_v2_code():
    dummy = [{"context": "class TargetService:\n  def execute(self): pass", "question": "Find service", "choice_A": "ServiceA", "choice_B": "ServiceB", "choice_C": "ServiceC", "choice_D": "ServiceD", "answer": "A", "domain": "code"}]
    with _patch_hf({"zai-org/LongBench-v2": dummy}):
        samples = _load_longbench_v2_code_samples(limit=1)
        assert len(samples) == 1
    # LongBench-v2's extract_answer reads only "The correct answer is (X)" / "... is X".
    assert _eval_longbench_v2_code("The correct answer is (A)", "A") is True
    assert _eval_longbench_v2_code("The correct answer is (B)", "A") is False


def test_imo_answer_bench():
    # The suite loads the real benchmark, not a filtered slice of the NuminaMath-CoT
    # *training* corpus it used to mine for rows mentioning "IMO".
    dummy = [
        {"Problem ID": "imo-bench-algebra-001", "Problem": "Find all integers n.",
         "Short Answer": "3", "Category": "Algebra", "Subcategory": "Operation",
         "Source": "IMO Shortlist 2021"},
    ]
    # v2 (answerbench_v2.csv at a pinned google-deepmind/superhuman commit), not the
    # deprecated v1 HF mirror; the CSV fetch is stubbed here.
    mod = "gbench.runners.eval_suites.imo_answer_bench"
    with mock.patch(f"{mod}._fetch_csv", return_value=""), \
         mock.patch(f"{mod}.parse_answerbench_csv", return_value=dummy):
        samples = _load_imo_answer_bench_samples(limit=5)
    assert len(samples) == 1
    messages, gold, meta = samples[0]
    assert gold == "3" and meta["category"] == "Algebra"     # real per-category breakdown
    assert "Find all integers n." in messages[0]["content"]

    # Correctness is the AnswerAutoGrader's call (it reads the whole solution); the suite only
    # reads its final boxed verdict.
    assert parse_grader_verdict("<thinking>...</thinking>\n\\boxed{Correct}") is True
    assert parse_grader_verdict("<thinking>...</thinking>\n\\boxed{Incorrect}") is False
    assert parse_grader_verdict("no verdict") is None


def test_imo_answer_bench_refuses_an_empty_dataset():
    """An empty load must raise, not silently produce a zero-sample suite."""
    mod = "gbench.runners.eval_suites.imo_answer_bench"
    with mock.patch(f"{mod}._fetch_csv", return_value=""), \
         mock.patch(f"{mod}.parse_answerbench_csv", return_value=[]):
        with pytest.raises(RuntimeError):
            _load_imo_answer_bench_samples(limit=5)


def test_multilingual_mmlu():
    # Canonical human-translated openai/MMMLU: Question + A..D + letter Answer + Subject.
    dummy = [{"Question": "Quelle est la capitale de la France?",
              "A": "Paris", "B": "Lyon", "C": "Nice", "D": "Metz",
              "Answer": "A", "Subject": "geography"}]
    with _patch_hf({"openai/MMMLU": dummy}):
        samples = _load_multilingual_mmlu_samples(limit=1)
        assert len(samples) == 1
        assert samples[0][1] == "A"
    # simple-evals MULTILINGUAL_ANSWER_REGEXES: a localized anchor, a colon, then the letter.
    assert _eval_multilingual_mmlu("Réponse: A", "A") is True
    assert _eval_multilingual_mmlu("Réponse: B", "A") is False
    # non-Latin extraction (Chinese anchor with a full-width colon)
    assert _eval_multilingual_mmlu("答案：A", "A") is True


def test_coco_caption():
    # Real COCO captioning: image sent to the VLM, gold = the reference caption list.
    # The Karpathy test split gives the references; the image comes from COCO-2014 val by name.
    img = Image.new("RGB", (10, 10), color="green")
    split = [{"filename": "COCO_val2014_000000000042.jpg", "cocoid": 42,
              "sentences": ["A dog running in a green park.", "A happy dog in a field."]}]
    dummy = [{"image": img, "file_name": "COCO_val2014_000000000042.jpg",
              "answer": ["A dog running in a green park.", "A happy dog in a field."],
              "id": 7}]
    import gbench.runners.eval_suites.coco_caption as coco
    with _patch_hf({"yerevann/coco-karpathy": split, "lmms-lab/COCO-Caption": dummy}), \
            mock.patch.object(coco, "_N_IMAGES", 1):
        samples = _load_coco_caption_samples(limit=1)
        assert len(samples) == 1
        msgs, refs, extra = samples[0]
        content = msgs[0]["content"]
        assert isinstance(content, list) and content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
        assert extra["image_id"] == 42
        assert len(refs) == 2
