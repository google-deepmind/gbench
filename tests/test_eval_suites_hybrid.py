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

"""Unit tests for the new hybrid evaluation suites in gbench."""

import pytest
from gbench.runners.eval_suites.amc_aime import _eval_amc_aime, _load_amc_aime_samples
from gbench.runners.eval_suites.bundled_detection import _eval_bundled_detection, _compute_iou, _load_bundled_detection_samples
from gbench.runners.eval_suites.loft_scifact import _eval_loft_scifact, _load_loft_scifact_samples
from gbench.runners.eval_suites.i18n_translate import _eval_i18n_translate, _load_i18n_translate_samples
from gbench.runners.eval_suites.causalbench import _eval_causalbench, _load_causalbench_samples
from gbench.runners.eval_suites.wildbench import build_wb_judge_prompt, _load_wildbench_samples
from gbench.runners.eval_suites import tau2 as tau2_mod, tau3 as tau3_mod
from gbench.runners.eval_suites.tau2 import _eval_tau2
from gbench.runners.eval_suites.tau3 import _eval_tau3


def test_amc_aime():
    assert _eval_amc_aime("The final answer is 142.", "142") is True
    assert _eval_amc_aime("Final Answer: 054", "54") is True
    assert _eval_amc_aime("The final answer is 100", "99") is False
    samples = _load_amc_aime_samples(limit=4)
    assert len(samples) > 0


def test_bundled_detection():
    # Test IoU calculation
    box1 = [100.0, 100.0, 300.0, 300.0]
    box2 = [100.0, 100.0, 300.0, 300.0]
    assert _compute_iou(box1, box2) == 1.0

    gold_objects = [{"label": "cat", "box_2d": [100.0, 100.0, 300.0, 300.0]}]
    pred_json = '[{"label": "cat", "box_2d": [110.0, 110.0, 290.0, 290.0]}]'
    assert _eval_bundled_detection(pred_json, gold_objects) is True

    bad_pred = '[{"label": "dog", "box_2d": [600.0, 600.0, 900.0, 900.0]}]'
    assert _eval_bundled_detection(bad_pred, gold_objects) is False


def test_loft_scifact():
    # Canonical LOFT retrieval: gold is a passage-ID set; Recall@1 on the first
    # predicted ID (extracted from a "Final Answer: [id]" list).
    gold = ["31715818"]
    assert _eval_loft_scifact("Final Answer: [31715818]", gold) is True
    assert _eval_loft_scifact("The passage is ID 31715818.\nFinal Answer: [31715818]", gold) is True
    assert _eval_loft_scifact("Final Answer: [99999]", gold) is False
    assert _eval_loft_scifact("no list here", gold) is False


def test_i18n_translate():
    gold = "Munich 1856: Four maps that will change your view of the city"
    pred = "Munich 1856: Four maps that change your perspective of the city"
    assert _eval_i18n_translate(pred, gold) is True
    bad_pred = "This is a completely unrelated sentence about cooking."
    assert _eval_i18n_translate(bad_pred, gold) is False


def test_causalbench():
    # Corr2Cause's own reading: golds are "entailment" (valid) / "contradiction".
    assert _eval_causalbench("Yes, the hypothesis is valid.", "entailment") is True
    assert _eval_causalbench("No, the relation does not hold.", "contradiction") is True
    assert _eval_causalbench("Yes", "contradiction") is False
    samples = _load_causalbench_samples(limit=2)
    assert len(samples) > 0


def test_wildbench():
    # Scoring is the WildBench WB-Score judge only (the keyword-overlap heuristic is gone); the
    # judge prompt carries the query, the checklist and the response in their template slots.
    checklist = ["list all tube stations in alphabetical order", "numbered list format"]
    good_resp = "Here is a numbered list of all tube stations in alphabetical order:\n1. Acton Central\n2. Acton Town\n3. Aldgate"
    prompt = build_wb_judge_prompt([{"role": "user", "content": "List the tube stations."}],
                                   checklist, good_resp)
    assert "<|begin_of_query|>\n\nList the tube stations.\n\n<|end_of_query|>" in prompt
    assert "- numbered list format\n" in prompt
    assert f"<|begin_of_response|>\n\n{good_resp}\n\n<|end_of_response|>" in prompt


def test_tau2():
    gold = {"tool": "search_flights", "args": {"origin": "AUS", "destination": "SEA", "date": "2026-10-14"}}
    good_call = '{"name": "search_flights", "arguments": {"origin": "AUS", "destination": "SEA", "date": "2026-10-14"}}'
    assert _eval_tau2(good_call, gold) is True
    bad_call = '{"name": "cancel_flight", "arguments": {"confirmation_code": "123"}}'
    assert _eval_tau2(bad_call, gold) is False
    # the diagnostic is offline: no network task loader is left in the module
    assert not hasattr(tau2_mod, "_load_tau2_samples")


def test_tau3():
    gold = {"tool": "lookup_billing_charges", "args": {"phone_number": "555-0144"}}
    good_call = '{"name": "lookup_billing_charges", "arguments": {"phone_number": "555-0144"}}'
    assert _eval_tau3(good_call, gold) is True
    bad_call = '{"name": "cancel_account", "arguments": {"account_id": "123"}}'
    assert _eval_tau3(bad_call, gold) is False
    assert not hasattr(tau3_mod, "_load_tau3_samples")
