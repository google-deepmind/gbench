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

"""Pure functions behind the stress sweep: judging drain, summarising repeats as
brackets, re-seeding the start, and detecting other traffic on the server.

The sweep finds, per repeat, the highest arrival rate that passed and the lowest that
failed: a bracket [pass, fail]. The headline is the median repeat's pass rate, and the
brackets say how well the limit is known. Nothing here interpolates inside a bracket.
"""

import math
from typing import Dict, List, Optional, Sequence

GRID_STEP = 2.0             # the sweep's ramp: x2 per point


def drained(before: Optional[dict], after: Optional[dict], own_in_flight: int = 0) -> Optional[bool]:
    """Was the server idle between two /metrics scrapes taken some seconds apart?

    Idle means: engine running + waiting minus the requests this client still has in
    flight is 0 in both scrapes, AND no request finished and no prompt was admitted in
    between (request_success / prompt_tokens unchanged). The engine gauges alone miss
    requests still in API-server preprocessing (e.g. multimodal), which the counters
    catch. None when /metrics doesn't expose the series (the caller then waits a fixed
    cool-down instead)."""
    keys = ("running", "waiting", "request_success", "prompt_tokens")
    if not before or not after or any(k not in before or k not in after for k in keys):
        return None
    busy = lambda m: m["running"] + m["waiting"] - own_in_flight          # noqa: E731
    return (busy(before) <= 0 and busy(after) <= 0
            and after["request_success"] == before["request_success"]
            and after["prompt_tokens"] == before["prompt_tokens"])


def drain_gap_s(cold_preprocess_s: Optional[float] = None) -> float:
    """Spacing of the two idle scrapes: max(2 s, 2 x the cold preprocessing time)."""
    return max(2.0, 2.0 * float(cold_preprocess_s or 0.0))


def summarize_brackets(brackets: Sequence[Dict], step: float = GRID_STEP) -> Dict:
    """Headline and honesty fields from per-repeat brackets {"pass": r, "fail": r|None}.

    - headline: the median repeat's pass rate (lower median on ties: deterministic and
      never above what most repeats passed);
    - consistent: the highest pass across repeats is below the lowest fail (every
      repeat agrees where the limit is);
    - spread_below_step: the repeats' pass rates differ by less than one grid step,
      so CV/CI over them would be false precision (reported as not assessed).
    """
    passes = [float(b.get("pass") or 0.0) for b in brackets]
    fails = [float(b["fail"]) for b in brackets if b.get("fail") is not None]
    if not passes:
        return {"headline": 0.0, "reps": 0}
    order = sorted(range(len(passes)), key=lambda i: (passes[i], i))
    med_val = passes[order[(len(order) - 1) // 2]]
    med_idx = min(i for i, p in enumerate(passes) if p == med_val)   # earliest such repeat
    hi_pass, lo_fail = max(passes), (min(fails) if fails else None)
    lo_pass = min(passes)
    spread_below = lo_pass > 0 and hi_pass / lo_pass < step - 1e-9
    return {
        "headline": passes[med_idx],
        "median_rep": med_idx,
        "reps": len(passes),
        "brackets": [{"pass": b.get("pass"), "fail": b.get("fail")} for b in brackets],
        "highest_pass": hi_pass,
        "lowest_fail": lo_fail,
        "consistent": lo_fail is None or hi_pass < lo_fail,
        "range": [lo_pass, hi_pass],
        "spread_below_step": bool(spread_below),
    }


def reseed_start(single_stream_e2e_s: Optional[float], floor: float = 0.05) -> Optional[float]:
    """Start rate re-seeded from a fresh single-stream probe: half the single-stream
    capacity, 0.5 / e2e (used instead of halving again after a second overload)."""
    if not single_stream_e2e_s or single_stream_e2e_s <= 0:
        return None
    return max(floor, 0.5 / float(single_stream_e2e_s))


# ── Foreign traffic (another client on the same server) ───────────────────────

def foreign_requests(lo: tuple, hi: tuple, records: Sequence[dict]) -> Optional[Dict]:
    """Requests the server finished between two scrapes that this client did not send.

    ``lo`` / ``hi`` are (t_request, t_response, snapshot) scrapes bracketing the
    measurement interval. foreign = change in request_success (all finished_reason
    labels) minus every request of ours that got a finish from the server (whatever our
    verdict on it). Each scrape took time, so our finishes inside either scrape's
    [request, response] window are ambiguous (counted on either side of it)."""
    (lo_req, lo_resp, lo_snap), (hi_req, hi_resp, hi_snap) = lo, hi
    if not lo_snap or not hi_snap or "request_success" not in lo_snap \
            or "request_success" not in hi_snap:
        return None
    server = hi_snap["request_success"] - lo_snap["request_success"]
    ours_in = ambiguous = 0
    for r in records:
        t = r.get("done")
        if t is None or not r.get("finish_reason"):
            continue
        if lo_resp < t < hi_req:
            ours_in += 1
        elif lo_req <= t <= lo_resp or hi_req <= t <= hi_resp:
            ambiguous += 1
    return {"server_finished": server, "ours": ours_in, "ambiguous": ambiguous,
            "foreign": server - ours_in}


def engine_above_client(snaps: Sequence[tuple], records: Sequence[dict],
                        t_lo: float, t_hi: float, consecutive: int = 2) -> bool:
    """True when, for ``consecutive`` sampler scrapes in a row inside [t_lo, t_hi], the
    engine held more requests (running + waiting) than this client had in flight: work
    from someone else."""
    from .perf_load import in_flight_at
    run = 0
    for t_req, t_resp, snap in snaps:
        t = (t_req + t_resp) / 2.0
        if not (t_lo <= t <= t_hi) or not snap or "running" not in snap:
            continue
        engine = snap.get("running", 0.0) + snap.get("waiting", 0.0)
        client = in_flight_at(records, [t])[0]
        run = run + 1 if engine > client else 0
        if run >= consecutive:
            return True
    return False


def contaminated(foreign: Optional[Dict], engine_excess: bool, slack: int = 2) -> Optional[bool]:
    """A point is contaminated when foreign finishes exceed the ambiguous count + slack,
    or the engine held more work than the client sent. None without /metrics."""
    if foreign is None:
        return True if engine_excess else None
    return bool(foreign["foreign"] > foreign["ambiguous"] + slack or engine_excess)
