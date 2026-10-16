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

"""multipl_e: languages_evaluated names the languages a run's problems came from (a subset run
can cover fewer than the configured list, which is kept as languages_requested), and a merge
unites every shard's languages. Pure functions; no container, no network."""

from gbench.runners.eval_suites import multipl_e as M


def test_the_languages_drawn_are_the_ones_present():
    traces = [{"category": "swift"}, {"category": "js"}, {"category": "js"}, {"category": None},
              {}, "not a dict"]
    assert M.languages_drawn(traces) == ["js", "swift"]


def test_a_merge_unites_every_shards_languages(monkeypatch):
    import gbench.merge as merge_mod

    def generic_merge(name, results):
        out = dict(results[0])
        out["sample_traces"] = [t for r in results for t in r.get("sample_traces") or []]
        return out
    monkeypatch.setattr(merge_mod, "generic_merge", generic_merge)
    monkeypatch.setattr(M, "apply_per_problem_pass_at_1", lambda r: r)
    merged = M.merge_shards([
        {"languages_evaluated": ["js", "sh"], "sample_traces": [{"category": "js"}]},
        {"languages_evaluated": ["scala"], "sample_traces": [{"category": "swift"}]}])
    assert merged["languages_evaluated"] == ["js", "scala", "sh", "swift"]
