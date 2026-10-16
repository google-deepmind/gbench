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

"""Pure functions behind the performance pillars' measurement method.

Everything here is deterministic and side-effect free (no network, no clock), so
the statistics the stress and serving pillars gate on can be unit-tested over many
seeds without a server. The runners own the I/O; this module owns the maths.
"""

import math
from typing import Dict, Optional, Tuple

# Bumped whenever a change alters a reported number or a pass/fail decision, so old
# and new results are never compared silently. Each component versions one part of
# the method. The serving and stress scaffolds pin methodology_version (as
# dataset_snapshot.workload_version), so a bump changes their scaffold_id and breaks
# the series instead of silently extending it. The offline throughput pillar uses
# none of this and is not pinned.
METHODOLOGY_COMPONENTS: Dict[str, int] = {
    # 2: every request unique by construction (salt + nonce, coded images), one chat
    # layout, the served context window for multimodal text (perf_payload).
    # 3: vLLM requests send skip_special_tokens False, so a reply of special tokens is
    # timed like any other (it streamed as empty chunks and read as never finished).
    "payload": 3,
    # 2: one merged Poisson schedule from the parent, sent at absolute times by any
    # number of workers; request count never below max(min_prompts, qps x 60), sized up
    # for the loaded latency; no prompt cap by default.
    "schedule": 2,
    # 2: the server ITL gate is PASS / FAIL / INDETERMINATE on the bucket counts.
    # 3: every check on a planned-send cohort; saturation = queue growth of TTFT (batch
    # medians, alpha 1e-4, material growth 0.05 s/s) + completion fraction >= 0.90 within
    # the SLO-implied e2e; schedule-based client-bound checks; INCONCLUSIVE points.
    # 4: the headline is the median repeat's bracket pass, certified with
    # percentile-sized probes (bisection to +-9%, budgeted) or labelled UNCERTIFIED; no
    # interpolated limit; drain before every point; contaminated points re-measured.
    "gate": 4,
    # 2: the server ITL P99 reads the CUMULATIVE histogram delta (v1 summed the
    # running totals again and reached 99% one or more buckets too early).
    # 3: client ITL split into inter-chunk (what users see) and per-token (chunk gaps
    # divided by the tokens each chunk carried); the per-token value is the fallback gate.
    "itl_metric": 3,
    # 2: per-token TPOT = (e2e - ttft) / (completion_tokens - 1) from the server's usage,
    # not the mean gap between streamed chunks (a chunk can carry several tokens).
    "tpot_metric": 2,
    # 2: per point / iteration prefix- and multimodal-cache deltas, prompt-length check.
    "cache_telemetry": 2,
    # 2: remote text serving on aiohttp (was a 32-thread pool that capped batch sizes
    # above 32 and opened a connection per request); peak in-flight recorded.
    "serving_transport": 2,
    # 2: serving input_length_actual is the median over iterations (+ template tokens),
    # ignore_eos_honored is observed (n/N requests reached max_tokens), per-request
    # samples saved, honest multimodal banner.
    "reporting": 2,
}


# The components each pillar's numbers depend on. Serving never reads the server ITL
# histogram or runs the sweep, so a schedule/gate/itl_metric bump must not break the
# serving series.
PILLAR_COMPONENTS: Dict[str, Tuple[str, ...]] = {
    "serving": ("payload", "tpot_metric", "cache_telemetry", "serving_transport", "reporting"),
    "stress": tuple(METHODOLOGY_COMPONENTS),
}


def methodology_record(pillar: str) -> dict:
    """The methodology stamp a serving or stress result carries: the components that
    pillar's numbers depend on, and a version string derived from them."""
    comps = {k: METHODOLOGY_COMPONENTS[k] for k in PILLAR_COMPONENTS[pillar]}
    return {
        "methodology_version": ".".join(f"{k}{v}" for k, v in sorted(comps.items())),
        "methodology_components": comps,
    }


# ── Prometheus histogram (vllm:inter_token_latency_seconds) ────────────────────

def hist_delta(before: Optional[tuple], after: Optional[tuple]) -> Optional[Tuple[Dict[float, float], float]]:
    """Per-bucket delta of two cumulative histogram snapshots ({le: count}, total).

    The buckets stay CUMULATIVE: delta[le] = number of observations <= le made
    between the snapshots. None when a snapshot is missing or nothing was observed.
    """
    if not before or not after:
        return None
    b_buckets, b_cnt = before
    a_buckets, a_cnt = after
    d_cnt = float(a_cnt) - float(b_cnt)
    if d_cnt <= 0:
        return None
    # A counter reset, a server restart or a series first exposed mid-point gives an
    # impossible delta; refuse it (the caller falls back to the client value) rather
    # than report a 'provable' verdict from it.
    if set(a_buckets) != set(b_buckets):
        return None
    delta = {}
    prev = 0.0
    for le in sorted(a_buckets):
        v = float(a_buckets[le]) - float(b_buckets[le])
        if v < -1e-9 or v + 1e-9 < prev or v > d_cnt + 1e-9:
            return None
        delta[le] = v
        prev = v
    if math.inf in delta and abs(delta[math.inf] - d_cnt) > 1e-6:
        return None
    return delta, d_cnt


def hist_quantile_bucket(delta: Dict[float, float], total: float,
                         pct: float = 99.0) -> Tuple[float, float]:
    """(lower_edge, upper_edge) in SECONDS of the bucket holding the pct quantile.

    Reads the cumulative counts directly: the first edge whose cumulative count
    reaches pct% of the total. No in-bucket interpolation (a multi-modal ITL
    distribution makes that false precision). lower_edge is 0.0 for the first
    bucket; upper_edge is inf for the overflow bucket.
    """
    les = sorted(delta)
    target = pct / 100.0 * total
    lower = 0.0
    for le in les:
        if delta[le] >= target:
            return lower, le
        lower = le
    return lower, math.inf


def _fmt_edge_ms(sec: float) -> str:
    if math.isinf(sec):
        return "inf"
    ms = sec * 1000.0
    return f"{ms:.0f}" if abs(ms - round(ms)) < 1e-6 else f"{ms:g}"


def bucket_label(lo_s: float, hi_s: float) -> str:
    """'(50,75]ms', or '(1,2.5]s' when the upper edge is 1 s or more (slow hardware)."""
    if not math.isinf(hi_s) and hi_s < 1.0:
        return f"({_fmt_edge_ms(lo_s)},{_fmt_edge_ms(hi_s)}]ms"
    if math.isinf(hi_s) and lo_s < 1.0:
        return f"({_fmt_edge_ms(lo_s)}ms,inf]"
    hi = "inf" if math.isinf(hi_s) else f"{hi_s:g}"
    return f"({lo_s:g},{hi}]s"


def smallest_finite_edge_ms(buckets: Dict[float, float]) -> Optional[float]:
    finite = [le for le in buckets if not math.isinf(le)]
    return min(finite) * 1000.0 if finite else None


def hist_itl_gate(delta: Dict[float, float], total: float, limit_ms: float,
                  pct: float = 99.0) -> dict:
    """Gate a P99 inter-token-latency SLO on a histogram delta, without interpolation.

    Let lo = the largest finite edge <= limit and hi = the smallest edge >= limit.
    The share of gaps <= limit is bounded by [cum(lo), cum(hi)] / total, so:
      PASS           cum(lo) >= pct%  (provably at least pct% within the limit)
      FAIL           cum(hi) <  pct%  (provably fewer than pct% within the limit)
      INDETERMINATE  otherwise (the limit sits inside a bucket that straddles pct%),
                     or the limit is below the smallest finite edge (no lo exists).
    When the limit is itself an edge, lo == hi and the outcome is never
    INDETERMINATE. The caller falls back to the client per-token value on
    INDETERMINATE and records that it did.

    Returns outcome, p99_itl_server_ms (the upper edge of the P99 bucket, numeric),
    p99_itl_server_bucket ("(50,75]ms"), itl_gate_edge_ms (lo, None if none),
    itl_frac_over_limit (1 - cum(lo)/total, an upper bound on the true share over
    the limit; None if no lo), and the bucket deltas in ms.
    """
    limit_s = float(limit_ms) / 1000.0
    les = sorted(delta)
    finite = [le for le in les if not math.isinf(le)]
    lo = max((le for le in finite if le <= limit_s + 1e-12), default=None)
    hi = min((le for le in les if le >= limit_s - 1e-12), default=math.inf)
    target = pct / 100.0 * total
    cum_lo = delta.get(lo, 0.0) if lo is not None else None
    cum_hi = delta.get(hi, total) if not math.isinf(hi) else total

    if lo is not None and cum_lo >= target:
        outcome = "PASS"
    elif cum_hi < target:
        outcome = "FAIL"
    else:
        outcome = "INDETERMINATE"

    b_lo, b_hi = hist_quantile_bucket(delta, total, pct)
    return {
        "outcome": outcome,
        "p99_itl_server_ms": (math.inf if math.isinf(b_hi) else b_hi * 1000.0),
        "p99_itl_server_bucket": bucket_label(b_lo, b_hi),
        "itl_gate_edge_ms": (lo * 1000.0 if lo is not None else None),
        "itl_frac_over_limit": ((1.0 - cum_lo / total) if lo is not None else None),
        "itl_hist_delta_ms": {_fmt_edge_ms(le): delta[le] for le in les},
        "itl_hist_count": total,
    }


# ── /metrics snapshot (vLLM Prometheus text) ────────────────────────────────────

# Counters summed over every engine / API-server label. The "external_" prefix-cache
# series (KV connector) is a different cache and is not summed into these.
_COUNTERS = {
    "vllm:prefix_cache_hits_total": "prefix_cache_hits",
    "vllm:prefix_cache_queries_total": "prefix_cache_queries",
    "vllm:mm_cache_hits_total": "mm_cache_hits",
    "vllm:mm_cache_queries_total": "mm_cache_queries",
    "vllm:prompt_tokens_total": "prompt_tokens",
    "vllm:generation_tokens_total": "generation_tokens",
    "vllm:request_success_total": "request_success",   # all finished_reason labels
    "vllm:spec_decode_num_draft_tokens_total": "spec_draft_tokens",
}
_GAUGES = {
    "vllm:num_requests_running": "running",
    "vllm:num_requests_waiting": "waiting",
}
_ITL_BUCKET = "vllm:inter_token_latency_seconds_bucket"
_ITL_COUNT = "vllm:inter_token_latency_seconds_count"


def parse_metrics(text: str) -> dict:
    """One /metrics scrape as {counter: total, gauge: total, "itl": (buckets, count),
    "cache_config": {label: value}}. Missing series are absent (never 0), so a caller
    can tell 'not exposed' from 'zero'."""
    import re

    out: dict = {}
    buckets: dict = {}
    itl_count = None
    for ln in text.splitlines():
        if not ln or ln.startswith("#"):
            continue
        name, _, rest = ln.partition("{") if "{" in ln.split(" ", 1)[0] else (ln.split(" ", 1)[0], "", "")
        try:
            value = float(ln.rsplit(" ", 1)[-1])
        except ValueError:
            continue
        if name in _COUNTERS:
            k = _COUNTERS[name]
            out[k] = out.get(k, 0.0) + value
        elif name in _GAUGES:
            k = _GAUGES[name]
            out[k] = out.get(k, 0.0) + value
        elif name == _ITL_BUCKET:
            m = re.search(r'le="([^"]+)"', rest)
            if m:
                le = math.inf if m.group(1) == "+Inf" else float(m.group(1))
                buckets[le] = buckets.get(le, 0.0) + value
        elif name == _ITL_COUNT:
            itl_count = (itl_count or 0.0) + value
        elif name == "vllm:cache_config_info" and "cache_config" not in out:
            out["cache_config"] = dict(re.findall(r'(\w+)="([^"]*)"', rest))
    if buckets and itl_count is not None:       # an empty histogram is 0, not absent
        out["itl"] = (buckets, itl_count)
    return out


def counter_delta(before: Optional[dict], after: Optional[dict], key: str) -> Optional[float]:
    """after[key] - before[key], or None when either scrape lacks the series."""
    if not before or not after or key not in before or key not in after:
        return None
    return float(after[key]) - float(before[key])


# ── Client-side per-token timing ───────────────────────────────────────────────

def per_token_tpot_ms(e2e_ms: Optional[float], ttft_ms: Optional[float],
                      completion_tokens: Optional[int]) -> Optional[float]:
    """Per-token TPOT of one request: (e2e - ttft) / (completion_tokens - 1).

    Uses the server's token count (final usage chunk), so a streamed chunk carrying
    several tokens can't inflate it. None without a count or with fewer than 2 tokens."""
    if e2e_ms is None or ttft_ms is None or not completion_tokens or completion_tokens < 2:
        return None
    return max(0.0, (float(e2e_ms) - float(ttft_ms)) / (int(completion_tokens) - 1))


def per_token_gaps(gaps_ms: list, tokens_per_chunk: list) -> list:
    """Per-token gaps: each inter-chunk gap divided by the tokens the later chunk carried
    (one value per token, so a 3-token chunk after 30 ms gives three 10 ms gaps)."""
    out = []
    for g, n in zip(gaps_ms, tokens_per_chunk):
        n = max(1, int(n or 1))
        out.extend([float(g) / n] * n)
    return out


def bucket_index(value_ms: float, edges_s: list) -> int:
    """Index of the histogram bucket (sorted upper edges in seconds, +Inf last) that
    holds value_ms."""
    v = float(value_ms) / 1000.0
    for k, le in enumerate(sorted(edges_s)):
        if v <= le + 1e-12:
            return k
    return len(edges_s) - 1
