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

"""Pure functions behind the stress pillar's open-loop load and its saturation gate.

No network, no clock: the runner owns the I/O and calls these with planned send times
and the records the client workers return. Every threshold here is relative (to the
point's own latency, rate or SLO), never an absolute number tuned to one GPU, so the
same gate is meaningful at 0.1 req/s on a laptop GPU and at 100 req/s on a large host.
"""

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

# ── Schedule ───────────────────────────────────────────────────────────────────

def merged_schedule(rate: float, n: int, seed: int) -> List[float]:
    """n Poisson arrival offsets (seconds after T0) at ``rate`` req/s from ONE seeded
    generator: the whole point's arrival process, independent of how many client
    processes later send it. The first arrival is an exponential gap after T0, never
    at T0 itself."""
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    return [float(x) for x in np.cumsum(rng.exponential(1.0 / float(rate), int(n)))]


def assign_workers(n: int, k: int) -> List[List[int]]:
    """Request indices per worker: arrival i goes to worker i mod k."""
    k = max(1, int(k))
    return [list(range(w, int(n), k)) for w in range(k)]


def request_count(qps: float, min_prompts: int, min_steady: int,
                  latency_est_s: Optional[float], c: float = 3.0,
                  window_s: float = 60.0, cap: Optional[int] = None) -> int:
    """Requests for one point; never fewer than the old floor max(min_prompts, qps·60).

    Also enough to (a) cover c x the loaded latency plus 10 s, so the queue-growth test
    sees a window several latencies long, and (b) leave min_steady requests in the
    cohort after the first L of sends is excluded (planned send in [T0 + L, T_last]).
    ``cap`` (an explicit --stress-max-prompts) is honoured as a recorded cap."""
    n = max(int(min_prompts), int(round(qps * window_s)))
    if latency_est_s and latency_est_s > 0:
        n = max(n, int(math.ceil(qps * (c * latency_est_s + 10.0))))
        n = max(n, int(math.ceil(qps * latency_est_s)) + int(min_steady) + 1)
    else:
        n = max(n, int(min_steady) + 1)
    if cap:
        n = min(n, int(cap))
    return n


def offered_rate_ci(n: int, span_s: float, alpha: float = 0.05) -> Dict[str, float]:
    """Realized offered rate of a Poisson schedule whose n-th arrival is at span_s:
    the unbiased estimate (n - 1) / span and the exact chi-square interval
    [chi2(a/2, 2n), chi2(1 - a/2, 2n)] / (2 span)."""
    if n <= 1 or span_s <= 0:
        return {"rate": 0.0, "low": 0.0, "high": 0.0}
    from scipy.stats import chi2
    return {"rate": (n - 1) / span_s,
            "low": float(chi2.ppf(alpha / 2, 2 * n)) / (2 * span_s),
            "high": float(chi2.ppf(1 - alpha / 2, 2 * n)) / (2 * span_s)}


def percentile_inf(values: Sequence[float], q: float = 99.0) -> float:
    """Linear-interpolated percentile that treats +inf (a request that never finished)
    as larger than everything: inf when the rank lands on or next to one. numpy's
    percentile returns nan there (inf - inf)."""
    v = np.sort(np.asarray(values, dtype=float))
    if len(v) == 0:
        return float("inf")
    rank = q / 100.0 * (len(v) - 1)
    lo, hi = int(math.floor(rank)), int(math.ceil(rank))
    if math.isinf(v[hi]):
        return float("inf")
    return float(v[lo] + (v[hi] - v[lo]) * (rank - lo))


# ── Cohort ─────────────────────────────────────────────────────────────────────

def cohort(records: Sequence[dict], t_lo: float, t_hi: float) -> List[dict]:
    """Requests whose PLANNED send time is in [t_lo, t_hi], in planned order. Selection
    never looks at completion order, so slow requests can't be dropped from it."""
    sel = [r for r in records if r.get("planned") is not None and t_lo <= r["planned"] <= t_hi]
    return sorted(sel, key=lambda r: r["planned"])


# ── Saturation: queue growth ───────────────────────────────────────────────────

SATURATED = "SATURATED"
NOT_DETECTED = "NOT_DETECTED"
INDETERMINATE = "INDETERMINATE"

# Calibration (docs/perf_evidence/step5b/final_calibration_2026-10-15.txt; 2000 seeds per
# healthy point, Clopper-Pearson 95% CIs; 1/4-slot capacity and batching servers, sigma
# 1.0, a fast batching server, 256 slots, unlimited servers at 0.13-100 req/s):
#   certification size (~1300-request cohorts, which decide the headline): false
#   SATURATED 0-0.1% at 0.8x/0.9x (upper CI <= 0.92%), detection 98.75-100% at 1.25x;
#   sweep size (the bracket): false SATURATED up to 1.2% at 0.8x and 1.5-3.6% at 0.9x on
#   1-4-slot servers (0 on large/fast/unlimited ones), detection 10-77% at 1.25x on small
#   servers; a false alarm there only moves the bracket conservatively.
# Live gemma-4-E2B (256 / 4 / 1 slots and a KV-limited server): 0/60 false at <= 0.9x; 32/40
# overloaded points flagged (all 40 also failed the P99 TTFT SLO).
# Undetected overload is NOT_DETECTED, never "keeping up"; the SLO gates then decide.
QG_BATCHES = 10
QG_ALPHA = 1e-4
# Material growth: TTFT rising faster than this many seconds per second of wall time.
# A queue fed at rho x capacity grows its wait by (rho - 1) s/s, so 0.05 means rho >~
# 1.05; a continuous-batching server settling into a larger batch raises TTFT by
# milliseconds over tens of seconds (~0.001 s/s), far below it. Dimensionless, so the
# same on any hardware.
QG_MATERIAL_SLOPE = 0.05


def _batch_means(x: np.ndarray, y: np.ndarray, batches: int, stat: str = "median"):
    idx = np.array_split(np.arange(len(x)), batches)
    f = np.median if stat == "median" else np.mean
    return (np.array([x[i].mean() for i in idx]), np.array([f(y[i]) for i in idx]))


def _t_crit(df: int, one_sided_alpha: float) -> float:
    """Student t critical value (scipy is a core gbench dependency)."""
    from scipy.stats import t as _t
    return float(_t.ppf(1.0 - one_sided_alpha, df))


def queue_growth(times: Sequence[float], values: Sequence[float],
                 batches: int = QG_BATCHES, alpha: float = QG_ALPHA,
                 material_slope: float = QG_MATERIAL_SLOPE, stat: str = "median") -> dict:
    """Is the queue growing over the cohort?

    ``values`` is a queueing-sensitive latency (TTFT, ms) at each request's planned send
    ``times`` (s). The cohort is cut into ``batches`` equal consecutive groups and the
    slope of the batch medians over time is estimated by least squares on those medians
    (batching makes the variance robust to the strong autocorrelation of queueing
    delays, which an i.i.d. test on single requests ignores; medians stop one very slow
    request from faking a trend):

      SATURATED      slope > 0 with one-sided significance ``alpha`` AND TTFT growing
                     faster than ``material_slope`` seconds per second of wall time (a
                     queue fed at rho x capacity grows its wait by rho - 1 s/s)
      NOT_DETECTED   otherwise (not proof of keeping up: the SLO gates decide)
      INDETERMINATE  too few requests for the test
    """
    x = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    out = {"outcome": INDETERMINATE, "n": int(len(x)), "batches": int(batches),
           "alpha": alpha, "material_slope_s_per_s": material_slope}
    if len(x) < 2 * batches or batches < 4:
        out["reason"] = "too few requests for the batch test"
        return out
    order = np.argsort(x)
    x, y = x[order], y[order]
    bx, by = _batch_means(x, y, batches, stat)
    if float(x[-1] - x[0]) <= 0:
        out["reason"] = "degenerate cohort"
        return out
    xc = bx - bx.mean()
    sxx = float((xc ** 2).sum())
    slope = float((xc * (by - by.mean())).sum() / sxx)
    resid = by - (by.mean() + slope * xc)
    df = batches - 2
    se = float(math.sqrt(max(1e-18, (resid ** 2).sum() / df / sxx)))
    tc = _t_crit(df, alpha)
    out.update({"slope_s_per_s": slope / 1000.0, "slope_se_s_per_s": se / 1000.0,
                "t_stat": slope / se if se > 0 else float("inf")})
    out["outcome"] = (SATURATED if (slope / se > tc and slope / 1000.0 > material_slope)
                      else NOT_DETECTED)
    return out


def in_flight_at(records: Sequence[dict], times: Sequence[float]) -> List[int]:
    """Client-side in-flight count (sent and not yet finished, successfully or not) at
    each instant; a request with no end at all counts as in flight forever."""
    sends = np.sort([r["send"] for r in records if r.get("send") is not None])
    dones = np.sort([r["done"] if r.get("done") is not None else math.inf
                     for r in records if r.get("send") is not None])
    return [int(np.searchsorted(sends, t, "right") - np.searchsorted(dones, t, "right"))
            for t in times]


# ── Secondary: completion fraction ─────────────────────────────────────────────

def completion_bound_s(ttft_slo_ms: float, itl_slo_ms: float, max_tokens: int) -> float:
    """The end-to-end time the SLOs imply for one request: TTFT SLO + (max_tokens - 1)
    ITL SLOs. Independent of the measured latency, so the gate is not circular."""
    return (float(ttft_slo_ms) + max(0, int(max_tokens) - 1) * float(itl_slo_ms)) / 1000.0


def completion_fraction(cohort_recs: Sequence[dict], t_last_arrival: float,
                        bound_s: float) -> Optional[float]:
    """Share of the cohort that finished successfully by t_last_arrival + bound_s."""
    if not cohort_recs:
        return None
    deadline = t_last_arrival + bound_s
    done = sum(1 for r in cohort_recs
               if r.get("ok") and r.get("done") is not None and r["done"] <= deadline)
    return done / len(cohort_recs)
