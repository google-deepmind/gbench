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

"""simpleqa sends the grader simple-evals' 2048-token cap and counts what that cap did: grader
replies that stopped at it (parsed as they arrived) and judge outages whose cascade answers
stopped there with no text. The endpoint and the Gemini cascade are stubbed; no network."""

from unittest import mock

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import simpleqa


def test_the_graders_cut_offs_are_counted(monkeypatch):
    samples = [([{"role": "user", "content": q}], "Ada", {"category": "x"})
               for q in ("Who wrote it?", "Who drew it?", "Who sang it?")]

    async def send(**kw):
        return base.Reply(text="Ada", tool_calls=None, finish_reason="stop", completion_tokens=3)

    async def judge(prompt, config=None):
        if "Who wrote it?" in prompt:
            return "A", "gemini-3.8-flash", {"finish_reason": "STOP"}
        if "Who drew it?" in prompt:
            return "A", "gemini-3.8-flash", {"finish_reason": "MAX_TOKENS"}
        return None, "JUDGE_OUTAGE", {"finish_reason": None, "max_tokens_without_text": 5}

    with mock.patch.object(base, "_send_single_request", send), \
            mock.patch.object(simpleqa, "gemini_required_skip", lambda *a, **k: None), \
            mock.patch.object(simpleqa, "_load_simpleqa_samples", lambda limit=None: samples), \
            mock.patch.object(simpleqa, "judge_generate_cascade_detailed", judge):
        res = simpleqa.run_simpleqa("m", "http://x/v1", 1)
    assert res["judge_replies_at_max_output_tokens"] == 1
    assert res["judge_outages_after_max_output_tokens"] == 1
    by_q = {t["messages"][-1]["content"]: t for t in res["sample_traces"]}
    assert by_q["Who drew it?"]["judge_finish_reason"] == "MAX_TOKENS"
    assert by_q["Who drew it?"]["is_correct"] is True          # parsed as it arrived
    assert by_q["Who sang it?"]["judge_grade"] == "JUDGE_OUTAGE"
    assert by_q["Who sang it?"]["judge_max_tokens_without_text"] == 5
