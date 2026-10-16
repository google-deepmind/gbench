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

"""bigcodebench's stop strings reach an OpenAI-compatible server and give upstream's text.

BigCodeBench's vllm backend runs vLLM in-process with 6 stop strings; an OpenAI-compatible server
(vLLM's included) rejects more than 4 with HTTP 400, which failed every no-think request. gbench
sends 4 and cuts the reply at the earliest of all 6, which is the text vLLM's stop handling returns.
"""

from gbench.runners.eval_suites import bigcodebench as B


def test_at_most_four_stop_strings_are_sent_and_all_six_are_applied():
    sent = B.SAMPLING[True].params["stop"]
    assert len(sent) <= 4 and set(sent) <= set(B.EOS)
    for stop in B.EOS:
        assert B.apply_eos(f"x = 1{stop}rest") == "x = 1", stop


def test_the_earliest_stop_string_wins_and_text_without_one_is_kept():
    assert B.apply_eos("a\nprint(1)\ndef main(): pass") == "a"
    assert B.apply_eos("def f():\n    return 1") == "def f():\n    return 1"


def test_the_prefilled_solution_is_cut_before_tabs_are_replaced():
    trace = {"response_text": "\tx = 1\nif __name__ == '__main__':\n\tf()",
             "extra_payload": {"continue_final_message": True}}
    assert B._solution(trace) == "    x = 1"
    # the openai (thinking) path keeps the message content as it arrived
    assert B._solution({"response_text": "\tx\nprint(1)", "extra_payload": {}}) == "\tx\nprint(1)"
