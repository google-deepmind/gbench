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
"""Regression: the per-eval result print must not crash on a present-but-None accuracy.

omnidocbench returned accuracy=None (composite undefined for the sampled pages). The per-eval
result print did `f"{acc:.2f}%"`, raising "unsupported format string passed to NoneType.__format__".
Because that print runs at the MODEL level (outside the per-eval try/except), it aborted the whole
fast stream at omnidocbench and lost the 12 suites after it. `_fmt_accuracy_pct` degrades None/
non-numeric to "-" so the run continues."""

from gbench.runners.evals import _fmt_accuracy_pct


def test_none_accuracy_degrades_to_dash():
    assert _fmt_accuracy_pct(None) == "-"


def test_numeric_accuracy_formats():
    assert _fmt_accuracy_pct(66.666) == "66.67%"
    assert _fmt_accuracy_pct(0) == "0.00%"
    assert _fmt_accuracy_pct(100.0) == "100.00%"


def test_non_numeric_degrades_to_dash():
    assert _fmt_accuracy_pct("n/a") == "-"
    assert _fmt_accuracy_pct({}) == "-"
