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

"""spider2's comparability reason names an --eval-limit subset and the replies the run's token
budget cut off (Spider2-lite sends no max_tokens of its own). The run is stubbed; no checkout, no
model, no network."""

from gbench.runners.eval_suites import spider2 as SP


def _run(monkeypatch, traces, **kw):
    monkeypatch.setattr(SP, "check_spider2_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(SP, "_load_spider2_samples", lambda limit=None: [])
    monkeypatch.setattr(SP, "run_eval_suite",
                        lambda **k: {"accuracy": 0.0, "sample_traces": list(traces)})
    monkeypatch.setattr(SP, "repo_revisions",
                        lambda: {"metadata": SP.SPIDER2_COMMIT, "gold": SP.SPIDER2_COMMIT})
    return SP.run_spider2("m", "http://x/v1", 1, **kw)


def test_a_subset_run_names_its_limit(monkeypatch):
    traces = [{"sample_idx": i, "response_text": "SELECT 1", "finish_reason": "stop"}
              for i in range(3)]
    reason = _run(monkeypatch, traces, limit=3)["leaderboard_comparable_reason"]
    assert "subset run (--eval-limit 3: 3 of the local instances" in reason


def test_a_reply_cut_by_the_budget_is_counted_and_named(monkeypatch):
    traces = [{"sample_idx": 0, "response_text": "SELECT", "finish_reason": "length",
               "max_tokens_sent": 16384},
              {"sample_idx": 1, "response_text": "", "finish_reason": "length",
               "max_tokens_sent": 16384}]
    r = _run(monkeypatch, traces)
    assert r["replies_cut_off_by_budget"] == 2 and r["replies_cut_off_in_reasoning"] == 1
    assert "2 repl(ies) cut off by this run's token budget" in r["leaderboard_comparable_reason"]
    assert "--eval-limit" not in r["leaderboard_comparable_reason"]
