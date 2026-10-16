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

"""A suite whose protocol answers each question several times shows questions x attempts.

On 2026-10-09 gpqa with --eval-limit 5 printed "Total questions: 50": 5 questions, each answered
10 times as simple-evals does. The result block and the summary table now say 5 x 10.
"""

from gbench import cli


def _res(samples, attempts, total):
    return {"total_questions": total, "correct_answers": total, "accuracy": 100.0,
            "eval_limit_plan": {"limit": 5, "samples": samples, "per_setting": None,
                                "attempts_per_sample": attempts, "generations": total}}


def test_repeated_attempts_show_as_questions_x_attempts():
    r = _res(5, 10, 50)
    assert cli._attempt_split(r, 50) == (5, 10)
    assert cli._questions_cell(r, 50) == "5x10"


def test_one_attempt_or_a_mismatch_shows_the_plain_count():
    assert cli._questions_cell(_res(5, 1, 5), 5) == 5
    # questions x attempts must equal total_questions, else the count is shown as it is
    assert cli._attempt_split(_res(5, 10, 49), 49) is None
    assert cli._questions_cell({"total_questions": 7}, 7) == 7
