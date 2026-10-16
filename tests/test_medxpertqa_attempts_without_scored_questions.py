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

"""medxpertqa: with every attempt a harness error, the @k block reports no rates (None, as base
does), never 0.0; with scored attempts it divides by the questions scored. Pure aggregation; no
model, no network."""

from gbench.runners.eval_suites import medxpertqa as M


def _trace(qid, ok, failed=False):
    t = {"extra_payload": {"medxpertqa_id": qid, "subset": "Text"}, "category": "c",
         "is_correct": ok, "response_text": None if failed else "B", "status": "OK"}
    if failed:
        t.update(status="FAILED", harness_error={"kind": "endpoint_failure"},
                 scoring_excluded=True)
    return t


def test_no_scored_question_gives_no_rates():
    turn2 = {"sample_traces": [_trace("q1", False, True), _trace("q1", False, True)]}
    res = M._combine({}, turn2, [], 2)
    block = res["attempts"]
    assert block["samples"] == 0
    assert block["pass_at_k"] is None and block["pass_hat_k"] is None
    assert res["accuracy"] is None


def test_scored_questions_give_rates_over_them():
    turn2 = {"sample_traces": [_trace("q1", True), _trace("q1", False),
                               _trace("q2", True), _trace("q2", True),
                               _trace("q3", False, True), _trace("q3", False, True)]}
    block = M._combine({}, turn2, [], 2)["attempts"]
    assert block["samples"] == 2
    assert block["pass_at_k"] == 100.0 and block["pass_hat_k"] == 50.0
    assert block["unstable_samples"] == 1
