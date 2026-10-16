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
"""Regression: medxpertqa scores the reply to the answer trigger with upstream answer_cleansing.

The suite now runs MedXpertQA's two-turn zero-shot CoT: the scored text is the reply to
"Therefore, among A through X, the answer is", read as upstream does (text after the trigger or
" answer is ", then the FIRST option letter). The old loose
regex grabbed stray letters such as the 'C' in '**Answer Choices**'; a word boundary is still
required, so that header can never be read as a letter."""

from gbench.runners.eval_suites.medxpertqa import _eval_medxpertqa


def test_answer_choices_header_does_not_false_match():
    # The stated answer is (A); the '**Answer Choices**' header must NOT be read as 'C'.
    resp = "**Evaluation of the Answer Choices**\nAfter analysis, Answer: (A)"
    assert _eval_medxpertqa(resp, "A") is True
    assert _eval_medxpertqa(resp, "C") is False    # NOT fooled by 'Answer Choices'


def test_last_answer_wins_over_earlier_mentions():
    # Upstream reads the text after " answer is ", so the stated answer beats earlier mentions.
    resp = "Initially I thought (B), but reconsidering, the answer is (D)."
    assert _eval_medxpertqa(resp, "D") is True
    assert _eval_medxpertqa(resp, "B") is False


def test_plain_final_answer():
    assert _eval_medxpertqa("Answer: (E)", "E") is True
    assert _eval_medxpertqa("The answer is F", "F") is True


def test_genuine_wrong_answer_stays_wrong():
    assert _eval_medxpertqa("Answer: (A)", "D") is False


def test_empty_or_missing():
    assert _eval_medxpertqa("", "A") is False
    assert _eval_medxpertqa("Answer: (A)", "") is False
