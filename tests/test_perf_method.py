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


"""Pure-function tests for gbench.analysis.perf_method (the perf pillars' maths)."""

import json
import math
from pathlib import Path

import pytest

from gbench.analysis.perf_method import (
    hist_delta, hist_itl_gate, hist_quantile_bucket, methodology_record,
)
from gbench.runners.stress import StressTestRunner

INF = float("inf")
FIX = Path(__file__).parent / "fixtures" / "perf_itl_hist" / "saved_points.json"


def _snap(buckets):
    return (buckets, buckets[INF])


def _zero(buckets):
    return ({k: 0.0 for k in buckets}, 0.0)


# ── the v1 bug: the deltas are already cumulative ─────────────────────────────

def test_counter_example_that_v1_got_wrong():
    """60 gaps <= 10 ms, 70 <= 20 ms, 100 total: P99 is in the overflow bucket.
    v1 summed the running totals (60+70 >= 99) and returned 20 ms."""
    after = {0.01: 60.0, 0.02: 70.0, INF: 100.0}
    assert StressTestRunner._hist_delta_p99_ms(_zero(after), _snap(after)) == INF
    d, n = hist_delta(_zero(after), _snap(after))
    assert hist_quantile_bucket(d, n) == (0.02, INF)


def test_existing_fixture_still_20ms():
    before = ({0.01: 0.0, 0.02: 0.0, INF: 0.0}, 0.0)
    after = ({0.01: 90.0, 0.02: 99.0, INF: 100.0}, 100.0)
    assert StressTestRunner._hist_delta_p99_ms(before, after) == 20.0
    assert StressTestRunner._hist_delta_p99_ms(None, after) is None
    assert StressTestRunner._hist_delta_p99_ms(before, before) is None


# ── saved histograms from the 2026-10-13 run (docs/perf_evidence) ─────────────

# Hard-coded from the red-team's independent check, NOT recomputed by the code
# under test: 75 req/s sits in (10,25]; the 2 req/s point in (0,10]; every other
# saved point in (50,75].
EXPECTED_MS = {2.0: (0, 10), 75.0: (10, 25)}


def _load_fixture():
    rows = json.loads(FIX.read_text())
    for r in rows:
        r["hist"] = {(INF if k == "+Inf" else float(k)): v for k, v in r["hist_delta_s"].items()}
    return rows


@pytest.mark.parametrize("row", _load_fixture(), ids=lambda r: f"{r['rate_qps']}")
def test_saved_histograms_land_in_the_known_bucket(row):
    h = row["hist"]
    d, n = hist_delta(_zero(h), _snap(h))
    lo, hi = hist_quantile_bucket(d, n)
    want = EXPECTED_MS.get(row["rate_qps"], (50, 75))
    assert (round(lo * 1000), round(hi * 1000)) == want


# ── the gate: PASS / FAIL provable, INDETERMINATE otherwise ───────────────────

EDGES = {0.01: 0.0, 0.025: 0.0, 0.05: 0.0, 0.075: 0.0, 0.1: 0.0, 0.2: 0.0, 0.3: 0.0, INF: 0.0}


def _hist(**counts_at):
    """Cumulative histogram from {edge_ms: cumulative count}; +Inf = 1000."""
    h = dict(EDGES)
    last = 0.0
    for le in sorted(h):
        key = "inf" if math.isinf(le) else str(int(round(le * 1000)))
        last = counts_at.get(key, last)
        h[le] = float(last)
    h[INF] = 1000.0
    return h


def test_limit_on_an_edge_is_exact():
    passing = _hist(**{"10": 500, "200": 990})      # 99.0% <= 200 ms
    failing = _hist(**{"10": 500, "200": 989})      # 98.9%
    for h, want in ((passing, "PASS"), (failing, "FAIL")):
        g = hist_itl_gate(h, 1000.0, 200)
        assert g["outcome"] == want
        assert g["itl_gate_edge_ms"] == 200.0


def test_limit_between_edges_is_indeterminate_when_bucket_straddles():
    # 98% <= 100 ms, 100% <= 200 ms: a 150 ms limit can't be decided from buckets.
    h = _hist(**{"100": 980, "200": 1000})
    g = hist_itl_gate(h, 1000.0, 150)
    assert g["outcome"] == "INDETERMINATE"
    assert g["itl_gate_edge_ms"] == 100.0
    assert g["itl_frac_over_limit"] == pytest.approx(0.02)
    # ...but provable either side
    assert hist_itl_gate(_hist(**{"100": 995}), 1000.0, 150)["outcome"] == "PASS"
    assert hist_itl_gate(_hist(**{"100": 900, "200": 980}), 1000.0, 150)["outcome"] == "FAIL"


def test_limit_below_smallest_edge_is_indeterminate():
    h = _hist(**{"10": 1000})
    g = hist_itl_gate(h, 1000.0, 5)
    assert g["outcome"] == "INDETERMINATE"
    assert g["itl_gate_edge_ms"] is None and g["itl_frac_over_limit"] is None


def test_gate_reports_numeric_upper_edge_and_bucket_label():
    h = _hist(**{"50": 900, "75": 995})
    g = hist_itl_gate(h, 1000.0, 200)
    assert g["p99_itl_server_ms"] == 75.0
    assert g["p99_itl_server_bucket"] == "(50,75]ms"
    assert isinstance(g["p99_itl_server_ms"], float)


def test_overflow_bucket_label():
    h = _hist(**{"300": 900})
    g = hist_itl_gate(h, 1000.0, 200)
    assert g["outcome"] == "FAIL"
    assert g["p99_itl_server_bucket"] == "(300ms,inf]"
    assert math.isinf(g["p99_itl_server_ms"])


def test_methodology_record_shape():
    rec = methodology_record("stress")
    assert rec["methodology_components"]["itl_metric"] >= 2
    assert rec["methodology_components"]["gate"] >= 2
    assert "itl_metric3" in rec["methodology_version"]


def test_synthetic_stall_the_v1_sum_hid():
    """Synthetic histogram shaped like rep 1 of multimodal at 0.63 req/s (client saw a
    1157 ms P99 gap, the server value read 25 ms): 60% <= 10 ms, 70% <= 25 ms, the
    tail beyond 300 ms. v1 summed 600 + 700 >= 990 and reported 25 ms; the corrected
    gate places P99 in the overflow bucket and FAILS the 200 ms SLO."""
    h = _hist(**{"10": 600, "25": 700, "300": 900})
    g = hist_itl_gate(h, 1000.0, 200)
    assert g["outcome"] == "FAIL"
    assert g["p99_itl_server_bucket"] == "(300ms,inf]"


def test_parse_metrics_sums_engines_and_skips_external():
    from gbench.analysis.perf_method import counter_delta, parse_metrics
    text = "\n".join([
        '# HELP x', 'vllm:prefix_cache_hits_total{engine="0",model_name="m"} 10.0',
        'vllm:prefix_cache_hits_total{engine="1",model_name="m"} 5.0',
        'vllm:external_prefix_cache_hits_total{engine="0"} 99.0',
        'vllm:request_success_total{engine="0",finished_reason="stop"} 3.0',
        'vllm:request_success_total{engine="1",finished_reason="length"} 4.0',
        'vllm:request_success_total{engine="1",finished_reason="abort"} 1.0',
        'vllm:num_requests_running{engine="0"} 2.0',
        'vllm:inter_token_latency_seconds_bucket{engine="0",le="0.01"} 9.0',
        'vllm:inter_token_latency_seconds_bucket{engine="0",le="+Inf"} 10.0',
        'vllm:inter_token_latency_seconds_count{engine="0"} 10.0',
        'vllm:cache_config_info{block_size="16",enable_prefix_caching="True"} 1.0',
    ])
    m = parse_metrics(text)
    assert m["prefix_cache_hits"] == 15.0
    assert m["request_success"] == 8.0
    assert m["running"] == 2.0
    assert "mm_cache_hits" not in m                      # not exposed != zero
    assert m["itl"] == ({0.01: 9.0, INF: 10.0}, 10.0)
    assert m["cache_config"]["block_size"] == "16"
    assert counter_delta(m, {**m, "prefix_cache_hits": 20.0}, "prefix_cache_hits") == 5.0
    assert counter_delta(m, m, "mm_cache_hits") is None


@pytest.mark.parametrize("before,after", [
    (({0.025: 100.0, INF: 1000.0}, 1000.0), ({0.025: 1500.0, INF: 1500.0}, 1500.0)),
    (({0.025: 900.0, INF: 1000.0}, 1000.0), ({0.025: 950.0, INF: 2000.0}, 2000.0 - 1500.0 + 1000.0)),
    (({0.025: 10.0, INF: 10.0}, 10.0), ({0.025: 20.0, 0.05: 30.0, INF: 30.0}, 30.0)),
])
def test_impossible_deltas_are_refused(before, after):
    """Counter reset / restart / series that appears mid-point: no verdict."""
    assert hist_delta(before, after) is None


def test_slow_hardware_bucket_label_in_seconds():
    from gbench.analysis.perf_method import bucket_label
    assert bucket_label(1.0, 2.5) == "(1,2.5]s"
    assert bucket_label(0.05, 0.075) == "(50,75]ms"
    assert bucket_label(0.3, INF) == "(300ms,inf]"
    assert bucket_label(20.0, INF) == "(20,inf]s"


def test_per_token_tpot_uses_the_server_token_count():
    from gbench.analysis.perf_method import per_token_tpot_ms
    # 128 tokens over 1270 ms after the first: 10 ms/token, whatever the chunking.
    assert per_token_tpot_ms(1300.0, 30.0, 128) == pytest.approx(10.0)
    assert per_token_tpot_ms(1300.0, 30.0, 1) is None
    assert per_token_tpot_ms(1300.0, 30.0, None) is None


def test_multi_token_chunks_become_per_token_gaps():
    """A chunk carrying 3 tokens after 30 ms is three 10 ms gaps, not one 30 ms gap."""
    from gbench.analysis.perf_method import bucket_index, per_token_gaps
    assert per_token_gaps([30.0, 10.0], [3, 1]) == [10.0, 10.0, 10.0, 10.0]
    assert per_token_gaps([12.0], [0]) == [12.0]                    # never divide by 0
    edges = [0.01, 0.025, 0.05, INF]
    assert [bucket_index(v, edges) for v in (5, 10, 11, 60)] == [0, 0, 1, 3]


def test_one_iteration_cv_is_not_assessed():
    """Issue 8: one iteration has no spread; CV must read 'not assessed', never 0% OK."""
    from gbench.analysis.statistics import (compute_statistics, format_statistics_summary,
                                            validate_repeatability)
    st = compute_statistics([12.5])
    assert st["std"] is None and st["cv_percent"] is None
    ok, msg = validate_repeatability({"x_cv_percent": None}, "x")
    assert ok is None and "not assessed" in msg
    assert "not assessed" in format_statistics_summary({"x_mean": 12.5, "x_std": None}, "x")
    two = compute_statistics([10.0, 12.0])
    assert two["cv_percent"] == pytest.approx(100 * 1.41421356 / 11.0, rel=1e-4)
