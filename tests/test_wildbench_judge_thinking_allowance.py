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

"""The Gemini judge gets upstream's 1024 verdict tokens plus room for its thinking.

src/eval.py sends max_tokens 1024 to judges that do not think (gpt-4o, gpt-4-turbo). Gemini's
thinking counts against max_output_tokens, so at 1024 it stopped mid-verdict: 55 of 315 WB-Elo
pool verdicts (17%) were dropped on the 2026-10-08 smoke run.
"""

from gbench.runners.eval_suites import wildbench as W


def test_the_judge_is_sent_the_verdict_tokens_plus_the_thinking_allowance():
    assert W.JUDGE_MAX_TOKENS == 1024                       # upstream's, for the verdict
    assert W.JUDGE_OUTPUT_LIMIT == W.JUDGE_MAX_TOKENS + W.JUDGE_THINKING_ALLOWANCE
    assert W.JUDGE_THINKING_ALLOWANCE >= 4096
    cfg = W._pool_judge_config()
    if cfg is not None:                                     # google-genai installed
        assert cfg.max_output_tokens == W.JUDGE_OUTPUT_LIMIT
        assert cfg.temperature == 0.0 and cfg.response_mime_type == "application/json"
    rec = W.judge_sampling_record()
    assert rec["max_output_tokens"] == W.JUDGE_OUTPUT_LIMIT
    assert "1024 for the verdict" in rec["max_output_tokens_note"]


def test_verdicts_graded_under_the_old_limit_are_not_reused(monkeypatch):
    new_dir = W.wb_elo_cache_dir()
    monkeypatch.setattr(W, "JUDGE_OUTPUT_LIMIT", W.JUDGE_MAX_TOKENS)
    assert W.wb_elo_cache_dir() != new_dir                  # the cache is keyed by the config


def test_outages_where_the_thinking_took_the_whole_budget_are_counted():
    src = open(W.__file__, encoding="utf-8").read()
    # WB-Score traces, WB-Reward comparisons and pool verdicts all record it, and the result sums it
    assert src.count('info.get("max_tokens_without_text")') >= 3
    assert 'result["judge_outages_after_max_output_tokens"] = outages_at_cap' in src
