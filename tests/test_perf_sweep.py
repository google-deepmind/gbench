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

"""Pure-function tests for gbench.analysis.perf_sweep (refinement, drain, brackets)."""

import pytest

from gbench.analysis.perf_sweep import (
    drain_gap_s, drained, reseed_start, summarize_brackets,
)


def test_drained_needs_idle_gauges_and_frozen_counters():
    idle = {"running": 0.0, "waiting": 0.0, "request_success": 10.0, "prompt_tokens": 500.0}
    assert drained(idle, dict(idle)) is True
    assert drained(idle, {**idle, "request_success": 11.0}) is False      # finished in between
    assert drained(idle, {**idle, "prompt_tokens": 600.0}) is False       # admitted in between
    assert drained({**idle, "running": 2.0}, idle) is False
    assert drained({**idle, "running": 2.0}, {**idle, "running": 2.0}, own_in_flight=2) is True
    assert drained(None, idle) is None and drained({"running": 0}, idle) is None
    assert drain_gap_s() == 2.0 and drain_gap_s(1.4) == pytest.approx(2.8)


def test_brackets_headline_consistency_and_spread():
    s = summarize_brackets([{"pass": 53.5, "fail": 61.5}, {"pass": 61.5, "fail": 70.7},
                            {"pass": 53.5, "fail": 61.5}])
    assert s["headline"] == 53.5 and s["median_rep"] == 0       # lower median, first index
    assert s["consistent"] is False                              # 61.5 passed and failed
    assert s["highest_pass"] == 61.5 and s["lowest_fail"] == 61.5
    ok = summarize_brackets([{"pass": 53.5, "fail": 107.0}] * 3)
    assert ok["consistent"] is True and ok["spread_below_step"] is True
    none_failed = summarize_brackets([{"pass": 100.0, "fail": None}])
    assert none_failed["consistent"] is True and none_failed["lowest_fail"] is None


def test_reseed_start():
    assert reseed_start(2.0) == pytest.approx(0.25)
    assert reseed_start(100.0) == 0.05                           # floor
    assert reseed_start(None) is None


def test_foreign_requests_and_contamination():
    from gbench.analysis.perf_sweep import contaminated, engine_above_client, foreign_requests
    lo = (10.0, 10.2, {"request_success": 100.0})
    hi = (50.0, 50.3, {"request_success": 150.0})
    ours = [{"done": 20.0 + i * 0.5, "finish_reason": "length"} for i in range(48)]
    ours += [{"done": 10.1, "finish_reason": "length"}, {"done": 50.1, "finish_reason": "stop"}]
    f = foreign_requests(lo, hi, ours)
    assert f == {"server_finished": 50.0, "ours": 48, "ambiguous": 2, "foreign": 2.0}
    assert contaminated(f, False) is False                      # within ambiguity + slack
    f2 = foreign_requests(lo, (50.0, 50.3, {"request_success": 170.0}), ours)
    assert f2["foreign"] == 22.0 and contaminated(f2, False) is True   # ~20 side requests
    assert foreign_requests(lo, (50.0, 50.3, None), ours) is None
    assert contaminated(None, False) is None and contaminated(None, True) is True


def test_engine_above_client_needs_consecutive_samples():
    from gbench.analysis.perf_sweep import engine_above_client
    recs = [{"send": 0.0, "done": 100.0, "ok": True}]                 # 1 in flight throughout
    snaps = [(t, t + 0.1, {"running": 1.0, "waiting": 0.0}) for t in range(10, 20)]
    assert engine_above_client(snaps, recs, 0, 100) is False
    snaps[3] = (13, 13.1, {"running": 3.0, "waiting": 0.0})            # one blip
    assert engine_above_client(snaps, recs, 0, 100) is False
    snaps[4] = (14, 14.1, {"running": 2.0, "waiting": 1.0})            # two in a row
    assert engine_above_client(snaps, recs, 0, 100) is True
