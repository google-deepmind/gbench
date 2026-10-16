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

"""Canonical RULER: per-tokenizer data-gen (prepare.py) + string-match scoring."""
import json
import os
import warnings

warnings.filterwarnings("ignore")

import pytest
import gbench.runners.eval_suites.ruler as R

_RD = os.environ.get("GBENCH_RULER_DIR")
_HAVE = bool(_RD and os.path.isfile(os.path.join(_RD, "scripts", "data", "prepare.py")))
needs_ruler = pytest.mark.skipif(not _HAVE, reason="GBENCH_RULER_DIR checkout not present")


def test_ruler_hard_errors_without_checkout(monkeypatch):
    # No-skip policy: absent checkout -> hard-error with clone/setup instructions, not a skip.
    monkeypatch.delenv("GBENCH_RULER_DIR", raising=False)
    with pytest.raises(RuntimeError, match="docs/evals/ruler.md"):
        R._ruler_dir()


def test_string_match_metrics():
    refs = ["alpha", "bravo", "charlie", "delta"]
    assert R._string_match_all("alpha bravo", refs) == 0.5          # fraction present
    assert R._string_match_all("none", refs) == 0.0
    assert R._string_match_all("ALPHA bravo charlie delta", refs) == 1.0  # case-insensitive
    assert R._string_match_part("alpha only", refs) == 1.0          # any-of
    assert R._string_match_part("none", refs) == 0.0
    assert R._string_match_all("x", []) == 0.0


def test_postprocess_and_item_score_dispatch():
    assert R._postprocess_pred("  a\x00b  ") == "a\nb"
    assert R._item_score("niah_single_1", "alpha", ["alpha", "beta"]) == 0.5   # niah -> all
    assert R._item_score("qa_1", "alpha", ["alpha", "beta"]) == 1.0            # qa -> part


def test_eval_ruler_bool():
    g = json.dumps({"task": "niah_multivalue", "outputs": ["a", "b"]})
    assert R._eval_ruler("a and b", g) is True
    assert R._eval_ruler("only a", g) is False
    assert R._eval_ruler("x", "not json") is False


def test_base_type_and_band_label():
    assert R._base_type("niah_multikey_2") == "niah"
    assert R._base_type("vt") == "variable_tracking"
    assert R._base_type("cwe") == "common_words_extraction"
    assert R._base_type("qa_2") == "qa"
    assert R._band_label(4096) == "4k" and R._band_label(131072) == "128k"


def test_require_context_hard_errors_when_insufficient(monkeypatch):
    monkeypatch.setattr(R, "_get_server_max_model_len", lambda u: 8192)
    with pytest.raises(RuntimeError, match="max_model_len"):
        R._require_context("http://x/v1", 131072)
    monkeypatch.setattr(R, "_get_server_max_model_len", lambda u: 262144)
    R._require_context("http://x/v1", 131072)  # sufficient -> no raise
    monkeypatch.setattr(R, "_get_server_max_model_len", lambda u: None)
    R._require_context("http://x/v1", 131072)  # unknown -> warn, no raise


@needs_ruler
def test_prepare_task_generates_per_tokenizer():
    rows = R._prepare_task(_RD, "google/gemma-4-26B-A4B-it", "niah_single_1", 4096, 2)
    assert len(rows) == 2
    assert rows[0].get("input") and rows[0].get("outputs")
    # a perfect prediction (echo the gold needle) scores 1.0 under the canonical metric
    assert R._item_score("niah_single_1", str(rows[0]["outputs"][0]), rows[0]["outputs"]) == 1.0
