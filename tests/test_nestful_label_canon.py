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
"""Regression: NESTFUL full/partial/slot match must be label-spelling agnostic. The gold
is internally inconsistent ('$var_1' in most traces, '$var1' in others) while the prompt MANDATES
'$var_1'. IBM's scorer handles this itself: parse_deepseek_output grounds gold and prediction with
ground_seq_nested_repsonse, which rewrites each `$label.field` reference to the NAME of the call that
produced the label, so the label spelling never reaches the comparison; a real mismatch still differs."""

import json

from gbench.runners.eval_suites.nestful import _ibm_ground, _ibm_parse, _ibm_score_example


def test_grounding_replaces_labels_with_producer_names():
    grounded = _ibm_ground([{"name": "f", "label": "$var_1", "arguments": {"n": 5}},
                            {"name": "g", "label": "$var_2", "arguments": {"data": "$var_1.output_0$"}}])
    assert grounded == [{"name": "f", "arguments": {"n": 5}},
                        {"name": "g", "arguments": {"data": "$f.output_0$"}}]


def test_prompt_compliant_answer_full_matches_inconsistent_gold():
    gold = [{"name": "f", "label": "$var1", "arguments": {"n": 5}},
            {"name": "g", "label": "$var2", "arguments": {"data": "$var1.output_0$"}}]
    pred = [{"name": "f", "label": "$var_1", "arguments": {"n": 5}},
            {"name": "g", "label": "$var_2", "arguments": {"data": "$var_1.output_0$"}}]
    pred_calls, gold_calls, _, err = _ibm_parse(json.dumps(pred), gold)
    assert not err
    assert _ibm_score_example(pred_calls, gold_calls)["accuracy_combined"] == 1


def test_true_negative_preserved():
    gold = [{"name": "f", "label": "$var1", "arguments": {"n": 5}}]
    wrong = [{"name": "f", "label": "$var_1", "arguments": {"n": 99}}]
    pred_calls, gold_calls, _, _ = _ibm_parse(json.dumps(wrong), gold)
    assert _ibm_score_example(pred_calls, gold_calls)["accuracy_combined"] == 0
