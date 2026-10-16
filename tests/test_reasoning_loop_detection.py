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

"""A reply cut off while looping in its reasoning is `looping`, not `truncated`.

On 2026-10-06, 40 lcb replies repeated one reasoning line up to 2,388 times until the token
budget ended. Their content was empty, and the detector read only the content, so they were
reported as truncated ("raise --max-output-tokens") with repetition_run 0.
"""

import types

from gbench.runners.eval_suites import base

LOOP = "Wait, if person i eats sushi j, then for sushi j+1, person i is still there.\n"
PROSE = " ".join(f"step {i} considers a different case of the input and its bound." for i in range(400))


def _reply(text, reasoning, finish="length"):
    return types.SimpleNamespace(text=text, reasoning=reasoning, finish_reason=finish, error=None,
                                 tool_calls=None, completion_tokens=65536)


def test_a_reply_cut_off_looping_in_its_reasoning_is_looping():
    assert base.classify_reply(_reply("", "Let me think.\n" + LOOP * 500)) == "looping"


def test_a_reply_cut_off_in_long_varied_reasoning_is_still_truncated():
    assert base.classify_reply(_reply("", PROSE)) == "truncated"


def test_content_loops_are_still_found():
    assert base.classify_reply(_reply(LOOP * 500, None)) == "looping"


def test_the_trace_reports_the_channel_that_repeats():
    assert base.looping_channel("", LOOP * 500)[1] == "reasoning"
    assert base.looping_channel(LOOP * 500, LOOP * 500)[1] == "content"
    assert base.looping_channel("short answer", PROSE) == ("short answer", "content")
    text, channel = base.looping_channel("", LOOP * 500)
    assert base.repetition_run(text) >= base.REPETITION_THRESHOLD
    assert base.repetition_onset(text)["onset_word"] == 0
