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

"""livebench regression tests: multi-word categories must not trigger LiveBench's
per-category bench-name truncation (data_analysis -> livebench/data)."""

from gbench.runners.eval_suites.livebench import _build_command


def _inner(monkeypatch, categories=None):
    if categories is None:
        monkeypatch.delenv("GBENCH_LIVEBENCH_CATEGORIES", raising=False)
    else:
        monkeypatch.setenv("GBENCH_LIVEBENCH_CATEGORIES", categories)
    cmd = _build_command("img", "/out", "m", "disp", "http://x/v1", 4, 0.0, "", "")
    return cmd[-1]  # the `bash -lc <inner>` string


def test_full_set_uses_whole_suite_name(monkeypatch):
    """The default (all 6) must run as `--bench-name live_bench`, which uses LiveBench's
    correct full-name code path - NOT the per-category form that truncates data_analysis
    to the non-existent HF dataset livebench/data and crashes the whole run."""
    inner = _inner(monkeypatch)
    assert "run_livebench.py --model" in inner
    # must not enumerate the multi-word categories (that path truncates upstream)
    assert "live_bench/data_analysis" not in inner
    assert "live_bench/instruction_following" not in inner
    # run_livebench.py invoked on the whole suite (bench-name immediately precedes
    # --question-source in the run step; distinct from the show_livebench_result step)
    assert "--bench-name live_bench --question-source" in inner


def test_single_word_subset_enumerates(monkeypatch):
    inner = _inner(monkeypatch, categories="math reasoning")
    assert "live_bench/math" in inner
    assert "live_bench/reasoning" in inner
    assert "live_bench/data_analysis" not in inner
