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

"""Tests for `gbench --list` (discoverability of suites/pillars/presets)."""

import types

from gbench.cli import _handle_list


def _run(capsys, what):
    rc = _handle_list(types.SimpleNamespace(list_what=what, eval_plugins_dir=None))
    return rc, capsys.readouterr().out


def test_list_evals_covers_every_registered_suite(capsys):
    from gbench.runners.eval_suites import SUITES
    rc, out = _run(capsys, "evals")
    assert rc == 0
    for name in SUITES:
        assert name in out, f"{name} missing from --list evals"


def test_list_pillars_matches_builtin_pillars(capsys):
    from gbench.runners.evals import BUILTIN_PILLARS
    rc, out = _run(capsys, "pillars")
    assert rc == 0
    for pillar, evs in BUILTIN_PILLARS:
        assert pillar in out
        for e in evs:
            assert e in out


def test_list_flags_suites_missing_from_evals_all(capsys):
    """A suite registered but absent from BUILTIN_PILLARS never runs under `--evals all`."""
    _rc, out = _run(capsys, "evals")
    from gbench.runners.eval_suites import SUITES
    from gbench.runners.evals import BUILTIN_PILLARS
    orphans = set(SUITES) - {e for _, evs in BUILTIN_PILLARS for e in evs}
    if orphans:
        assert "NOT part of" in out
        for o in orphans:
            assert o in out


def test_list_presets_does_not_crash(capsys):
    rc, out = _run(capsys, "presets")
    assert rc == 0 and "quick" in out and "default" in out


def test_save_eval_summary_csv_includes_effective_n(tmp_path):
    import csv
    from gbench.cli import _save_eval_summary_csv

    eval_results = [
        {
            "eval_name": "custom_suite_a",
            "model_short": "gemma-4",
            "format": "vllm",
            "thinking": False,
            "total_questions": 20,
            "effective_n": 5,
            "correct_answers": 15,
            "accuracy": 75.0,
            "status": "success",
            "duration_s": "12.3",
            "category_accuracy": {
                "cat1": {"total": 10, "correct": 8, "accuracy": 80.0},
                "cat2": {"total": 10, "correct": 7, "accuracy": 70.0},
            },
        }
    ]
    eval_failures = []
    eval_pillars = []
    custom_pillars = {"custom_suite_a": "1. Custom Pillar"}

    csv_path = _save_eval_summary_csv(
        tmp_path, eval_results, eval_failures, eval_pillars, custom_pillars
    )

    assert csv_path is not None
    assert csv_path.exists()

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert "Effective N" in reader[0]
    # Main overall row
    assert reader[0]["Effective N"] == "5"
    assert reader[0]["Questions"] == "20"
    assert reader[0]["Accuracy (%)"] == "75.00"
    # Subcategory rows
    assert reader[1]["Subcategory"] == "cat1"
    assert reader[1]["Effective N"] == "10"

