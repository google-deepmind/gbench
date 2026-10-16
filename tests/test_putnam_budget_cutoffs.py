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

"""putnam's informal track has no official cap, so every request carries --max-output-tokens; a
reply that budget cut off is counted and named in leaderboard_comparable_reason. The endpoint and
the Gemini judge are stubbed; no network."""

from unittest import mock

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import putnam as P


@pytest.fixture(autouse=True)
def _key(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    for k in ("GEMINI_API_KEYS", "GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL", "GBENCH_SHARD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))


def test_a_reply_the_budget_cut_off_is_counted_and_named(monkeypatch):
    samples = [([{"role": "user", "content": P._PROMPT_HEAD + "Prove that X."
                  + P._INSTRUCTIONS[False]}], "", {"category": "algebra"})]

    async def cascade(prompt, *, config=None):
        return "Grade: INCORRECT", "gemini-3.8-flash"

    async def send(**kw):
        return base.Reply(text="A proof that stops mid", tool_calls=None,
                          finish_reason="length", completion_tokens=1024)
    monkeypatch.setattr(P, "check_putnam_prerequisites", lambda key=None: (True, ""))
    monkeypatch.setattr(P, "_load_putnam_samples", lambda **kw: samples)
    with mock.patch.object(base, "_send_single_request", send), \
            mock.patch.object(P, "judge_generate_cascade", cascade):
        res = P.run_putnam("m", "http://stub/v1", 1, limit=1, max_output_tokens=1024)
    assert res["replies_cut_off_by_budget"] == 1 and res["replies_cut_off_in_reasoning"] == 0
    assert "1 repl(ies) cut off by this run's token budget" in res["leaderboard_comparable_reason"]
