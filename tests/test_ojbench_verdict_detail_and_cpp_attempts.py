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

"""ojbench keeps the judge's per-generation verdict (WA / TLE / RTE, per-scale verdicts and a capped
detail) in each trace, and its @k block follows the C++ headline instead of pooling both
languages. Pure functions; no judge, no network."""

from gbench.runners.eval_suites import ojbench as O


def test_the_verdict_detail_is_kept_and_capped():
    record = {"verdict": "TLE", "smallverdict": "AC", "largeverdict": "TLE", "uid": "p1#0",
              "detailed_results": [{"case": i, "verdict": "AC", "out": "x" * 500}
                                   for i in range(25)]}
    out = O.judge_execution(record)
    assert out["verdict"] == "TLE"
    assert out["scale_verdicts"] == {"largeverdict": "TLE", "smallverdict": "AC"}
    assert len(out["detailed_results"]) == O._DETAIL_ENTRIES
    assert out["detailed_results_left_out"] == 25 - O._DETAIL_ENTRIES
    assert all(len(d["out"]) == 200 for d in out["detailed_results"])
    assert O.judge_execution({"verdict": "WA", "detailed_results": "boom"}) == {
        "verdict": "WA", "scale_verdicts": {}, "detailed_results": "boom"}


def _trace(pid, lang, ok, excluded=False):
    t = {"extra_payload": {"row": {"id": pid, "language": lang}}, "is_correct": ok}
    if excluded:
        t["scoring_excluded"] = True
    return t


def test_the_attempts_block_counts_cpp_generations_only():
    traces = ([_trace("a", "cpp", False)] * 2 + [_trace("b", "cpp", True), _trace("b", "cpp", False)]
              + [_trace("a", "python", True)] * 2 + [_trace("c", "cpp", True, excluded=True)])
    block = O.cpp_attempts(traces, 2)
    assert block["samples"] == 2 and block["generations"] == 4
    assert block["avg_at_k"] == 25.0          # 1 of 4 C++ generations (python's wins left out)
    assert block["pass_at_k"] == 50.0 and block["pass_hat_k"] == 0.0
    assert block["unstable_samples"] == 1
    assert O.cpp_attempts([_trace("a", "python", True)], 2) is None
