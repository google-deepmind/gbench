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

"""Regression: a present-but-None accuracy must not crash the eval summary.

omnidocbench (and any suite whose composite is undefined for the sampled data) returns
status=success with accuracy=None. `.get('accuracy', 0.0)` does NOT substitute the default for an
existing None, so `f"{None:.2f}"` used to raise "unsupported format string passed to
NoneType.__format__" in _save_eval_summary_csv -- and because that runs at the MODEL level (outside
the per-eval try/except), it aborted the entire run, losing the summary and every later suite."""

from gbench.cli import _save_eval_summary_csv


def test_summary_csv_survives_none_accuracy(tmp_path):
    results = [
        {"eval_name": "omnidocbench", "model_name": "m", "model_short": "m",
         "format": "remote-endpoint", "status": "success", "total_questions": 3,
         "correct_answers": 0, "accuracy": None,
         "category_accuracy": {"tables": {"total": 1, "correct": 0, "accuracy": None}}},
        {"eval_name": "gpqa", "model_name": "m", "model_short": "m", "format": "remote-endpoint",
         "status": "success", "total_questions": 5, "correct_answers": 5, "accuracy": 100.0},
    ]
    p = _save_eval_summary_csv(tmp_path, results, [], [], {})
    assert p is not None and p.exists()
    text = p.read_text().lower()
    # Both rows written (no crash), and the None-accuracy row degraded to "-" not a formatted number.
    assert "omnidocbench" in text and "gpqa" in text
