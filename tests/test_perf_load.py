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

"""Pure-function tests for gbench.analysis.perf_load (stress schedule + saturation gate),
over a fixed list of 200 seeds against the simulated servers in tests/perf_sim.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perf_sim import CapacityServer, Latency, UnlimitedServer  # noqa: E402

from gbench.analysis.perf_load import (  # noqa: E402
    INDETERMINATE, NOT_DETECTED, SATURATED, assign_workers, cohort, completion_bound_s,
    completion_fraction, in_flight_at, merged_schedule, offered_rate_ci, queue_growth,
    request_count,
)

SEEDS = range(200)


def test_schedule_is_independent_of_worker_count():
    sched = merged_schedule(5.0, 300, seed=42)
    assert sched[0] > 0 and all(b > a for a, b in zip(sched, sched[1:]))
    for k in (1, 4, 13, 32):
        parts = assign_workers(len(sched), k)
        merged = sorted(i for p in parts for i in p)
        assert merged == list(range(len(sched)))                 # every arrival once
        assert all(i % k == w for w, p in enumerate(parts) for i in p)
    assert merged_schedule(5.0, 300, seed=42) == sched           # reproducible


def test_realized_rate_lies_in_its_poisson_ci():
    inside = 0
    for seed in SEEDS:
        s = merged_schedule(10.0, 600, seed)
        ci = offered_rate_ci(len(s), s[-1])
        inside += ci["low"] <= 10.0 <= ci["high"]
    assert inside / len(SEEDS) > 0.9                             # ~95% nominal


@pytest.mark.parametrize("qps", [0.13, 0.26, 5.0, 50.0, 100.0])
def test_request_count_never_below_the_old_floor(qps):
    for lat in (None, 1.0, 90.0):
        n = request_count(qps, 40, 15, lat)
        assert n >= max(40, round(qps * 60))
        if lat:
            # cohort after the first L of sends still holds >= min_steady requests
            assert n - qps * lat >= 15


def test_request_count_cap_is_honoured():
    assert request_count(100.0, 40, 15, 90.0, cap=500) == 500


def test_cohort_selects_by_planned_send_not_completion():
    recs = [{"planned": t, "send": t, "done": t + (100 if t == 5 else 1), "ok": True}
            for t in range(10)]
    sel = cohort(recs, 3, 7)
    assert [r["planned"] for r in sel] == [3, 4, 5, 6, 7]       # the slow one stays in


def _verdicts(server, rate, l_est, seeds):
    out = []
    for seed in seeds:
        n = request_count(rate, 40, 15, l_est)
        sends = merged_schedule(rate, n, seed)
        recs = server.serve(sends, np.random.default_rng(seed + 7919))
        for r in recs:
            r["planned"] = r["send"]
        l50 = float(np.median([r["done"] - r["send"] for r in recs]))
        coh = cohort(recs, l50, sends[-1])
        out.append(queue_growth([r["planned"] for r in coh], [r["ttft"] for r in coh])["outcome"])
    return out


@pytest.mark.parametrize("slots,smed", [(1, 1.0), (4, 2.0), (256, 20.0)])
@pytest.mark.parametrize("rho", [0.5, 0.8])
def test_false_saturation_at_most_one_percent(slots, smed, rho):
    srv = CapacityServer(slots, Latency(smed, 0.4))
    v = _verdicts(srv, rho * srv.capacity_qps(), smed * np.exp(1.2816 * 0.4), SEEDS)
    assert v.count(SATURATED) / len(v) <= 0.01


@pytest.mark.parametrize("rate,lat", [(0.13, Latency(3.0, 0.6, tail_p=0.02)),
                                      (0.26, Latency(90.0, 0.3)),
                                      (50.0, Latency(3.0, 0.6, tail_p=0.02))])
def test_unlimited_servers_are_never_called_saturated(rate, lat):
    v = _verdicts(UnlimitedServer(lat), rate, lat.median_s * np.exp(1.2816 * lat.sigma), range(100))
    assert v.count(SATURATED) == 0


def test_heavy_overload_is_detected_at_high_concurrency():
    srv = CapacityServer(256, Latency(20.0, 0.4))
    v = _verdicts(srv, 1.25 * srv.capacity_qps(), 20.0 * np.exp(1.2816 * 0.4), range(50))
    assert v.count(SATURATED) / len(v) >= 0.95


def test_overload_is_never_reported_as_keeping_up():
    """There is no 'keeping up' verdict: undetected overload is NOT_DETECTED, and the
    SLO gates decide it."""
    srv = CapacityServer(1, Latency(1.0, 0.4))
    v = _verdicts(srv, 1.25 * srv.capacity_qps(), np.exp(1.2816 * 0.4), range(50))
    assert set(v) <= {SATURATED, NOT_DETECTED, INDETERMINATE}


def test_too_few_requests_is_indeterminate():
    assert queue_growth([1, 2, 3], [1, 1, 1])["outcome"] == INDETERMINATE


def test_completion_fraction_and_bound():
    assert completion_bound_s(5000, 200, 128) == pytest.approx(5.0 + 127 * 0.2)
    recs = [{"ok": True, "done": 10.0}, {"ok": True, "done": 50.0}, {"ok": False, "done": 1.0}]
    assert completion_fraction(recs, t_last_arrival=0.0, bound_s=30.0) == pytest.approx(1 / 3)
    assert completion_fraction([], 0.0, 1.0) is None


def test_in_flight_counts_unfinished_requests_forever():
    recs = [{"send": 0.0, "done": 1.0, "ok": True}, {"send": 0.5, "done": None, "ok": False}]
    assert in_flight_at(recs, [0.25, 0.75, 5.0]) == [1, 2, 1]


def test_percentile_inf_never_returns_nan():
    from gbench.analysis.perf_load import percentile_inf
    assert percentile_inf([1.0] * 98 + [float("inf")] * 2) == float("inf")
    assert percentile_inf(list(range(101))) == pytest.approx(99.0)
    assert percentile_inf([]) == float("inf")
