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

"""mrcr's exact_match_rate counts verbatim reproductions only: a near-copy whose canonical grade
is 0.99 or more (a missing sign-off, added markup) still counts in the mean grade but is not a
verbatim reproduction. Pure functions; no network, no model."""

from gbench.runners.eval_suites.mrcr import _eval_mrcr, grade_mrcr

_BODY = ("Dear friend,\n\n" + "The tide came in and the gulls kept their watch over the harbor. "
         * 40 + "\n\nUntil next time,\n\n[Your Name]")
_GOLD = {"answer": "CANARY42" + _BODY, "canary": "CANARY42"}


def test_a_verbatim_reproduction_passes():
    assert _eval_mrcr("CANARY42" + _BODY, _GOLD) is True


def test_a_near_copy_with_a_high_grade_is_not_a_verbatim_reproduction():
    near = "CANARY42" + _BODY.replace("Dear friend,", "**Dear friend,**").replace(
        "\n\nUntil next time,\n\n[Your Name]", "")
    assert grade_mrcr(near, _GOLD) >= 0.99
    assert _eval_mrcr(near, _GOLD) is False


def test_the_canary_gate_applies():
    assert _eval_mrcr(_BODY, _GOLD) is False
    assert _eval_mrcr("", _GOLD) is False
    assert _eval_mrcr(None, _GOLD) is False


def test_a_gold_without_a_canary_compares_the_whole_reply():
    assert _eval_mrcr("abc", "abc") is True
    assert _eval_mrcr("abc ", "abc") is False
