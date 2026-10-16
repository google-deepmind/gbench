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

"""NESTFUL corpus metrics must match IBM-research/NESTFUL src/scorer.py + src/utils.py exactly.

Cross-checks gbench's vendored port (_ibm_macro_f1, _ibm_post_process, _ibm_corpus_scores)
against IBM's own pure helpers, loaded by file path from the NESTFUL checkout when present.
"""
import importlib.util
import os
import warnings

warnings.filterwarnings("ignore")

import pytest
import gbench.runners.eval_suites.nestful as N


def _load_ibm_utils():
    """Import IBM's src/utils.py by path (it is pure: sklearn + json only)."""
    func_dir = os.environ.get("GBENCH_NESTFUL_FUNC_DIR", "")
    # func_dir = <checkout>/data_v2/executable_functions -> src is ../../src
    if not func_dir:
        return None
    src = os.path.normpath(os.path.join(func_dir, "..", "..", "src", "utils.py"))
    if not os.path.exists(src):
        return None
    spec = importlib.util.spec_from_file_location("_ibm_nestful_utils", src)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


IBM = _load_ibm_utils()


def test_macro_f1_matches_ibm_compute_score_sklearn():
    if IBM is None:
        pytest.skip("IBM NESTFUL checkout (GBENCH_NESTFUL_FUNC_DIR) not available")
    gold = [["a", "b"], ["c"], ["a"]]
    pred = [["a"], ["c", "x"], ["a"]]
    _, _, ibm_f1 = IBM.compute_score_sklearn(gold, pred)
    assert N._ibm_macro_f1(gold, pred) == pytest.approx(float(ibm_f1))


def test_post_process_matches_ibm():
    if IBM is None:
        pytest.skip("IBM NESTFUL checkout not available")
    g = ["f(a = 1)", "g(b = 2)", "h(c = 3)"]
    p = ["f(a = 1)", "h(c = 3)"]
    ibm_g, ibm_p = IBM.post_process_api_with_args(list(g), list(p))
    my_g, my_p = N._ibm_post_process(list(g), list(p))
    assert (my_g, my_p) == (ibm_g, ibm_p)


def test_corpus_scores_full_and_partial_and_f1():
    # Two examples. Ex1 pred exactly matches gold (full). Ex2 pred wrong args (partial < 1).
    gold = [
        [{"name": "add", "arguments": {"x": 1, "y": 2}, "label": "$var1"}],
        [{"name": "mul", "arguments": {"x": 3, "y": 4}, "label": "$var1"}],
    ]
    pred = [
        [{"name": "add", "arguments": {"x": 1, "y": 2}, "label": "$var1"}],
        [{"name": "mul", "arguments": {"x": 3, "y": 9}, "label": "$var1"}],  # wrong y
    ]
    s = N._ibm_corpus_scores(pred, gold)
    # Full match: only Ex1 (Ex2's api_with_args differ) -> 0.5
    assert s["full_match_accuracy"] == pytest.approx(0.5)
    # Partial: Ex1 accuracy 1.0; Ex2 accuracy 0.0 (single position, mismatched string) -> mean 0.5
    assert s["partial_match_accuracy"] == pytest.approx(0.5)
    # Intent names identical in both examples -> perfect intent F1
    assert s["f1_intent"] == pytest.approx(1.0)


def test_dollar_ref_repair_matches_upstream():
    # A pred arg missing its closing "$" is repaired to close it (scorer.py slot/api_with_args).
    assert N._ibm_dollar_fix("$var1.out") == "$var1.out$"
    assert N._ibm_dollar_fix("$var1.out$") == "$var1.out$"
    assert N._ibm_dollar_fix("plain") == "plain"
