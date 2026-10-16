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

"""Statistical analysis utilities for benchmark results."""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_statistics(values: list[float]) -> dict[str, float]:
    """Compute statistical measures for a list of values.
    
    Args:
        values: List of numeric values
        
    Returns:
        Dictionary with statistical measures
    """
    if not values:
        return {}
    
    arr = np.array(values)
    
    stats = {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        # One value has no spread to measure: None, never a "perfect" 0.
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else None,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "count": len(values),
    }
    
    # Coefficient of Variation (CV%); None when it can't be assessed (fewer than 2
    # values, or a zero mean).
    if stats["std"] is not None and stats["mean"] > 0:
        stats["cv_percent"] = (stats["std"] / stats["mean"]) * 100
    else:
        stats["cv_percent"] = None
    
    return stats


def tail_validity(n: int, p: float, min_tail: int = 10) -> tuple[bool, int]:
    """Is a sample of size n large enough for a stable p-th percentile?

    A percentile is only trustworthy if enough samples land in its tail. Rule of
    thumb: need >= min_tail samples beyond the percentile, i.e.
    n >= min_tail / (1 - p/100) for an upper percentile (P99 -> ~1000, P99.9 ->
    ~10000). Below that the percentile is dominated by a handful of points and
    should be reported with its CI, never as a hard number.

    Args:
        n: Number of pooled samples.
        p: Percentile in (0, 100).
        min_tail: Minimum samples required in the tail.

    Returns:
        (is_valid, required_n).
    """
    frac = (1.0 - p / 100.0) if p >= 50 else (p / 100.0)
    if frac <= 0:
        return False, 0
    # Round before ceil so float imprecision (1-0.999 == 0.0009999999999998899,
    # making 10/frac == 10000.0000000011) doesn't inflate P99.9's requirement to
    # 10001 and wrongly flag a correctly-sized 10000-sample run as invalid. Six
    # decimals absorbs the float noise while preserving any genuine fraction.
    required = int(np.ceil(round(min_tail / frac, 6)))
    return n >= required, required


def bootstrap_percentile_ci(
    samples,
    p: float,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 83,
) -> dict[str, Any]:
    """Point estimate + bootstrap CI for the p-th percentile of iid samples.

    Use for per-request metrics (TTFT, E2EL, per-request TPOT) where each
    request is one independent sample. Resamples requests with replacement.

    Args:
        samples: 1-D array of per-request values.
        p: Percentile in (0, 100).
        n_boot: Bootstrap resamples.
        ci: Confidence level (0.95 -> 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        {value, ci_low, ci_high, n} (value None if empty).
    """
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return {"value": None, "ci_low": None, "ci_high": None, "n": 0}
    point = float(np.percentile(arr, p))
    if n < 2:
        return {"value": point, "ci_low": point, "ci_high": point, "n": int(n)}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot = np.percentile(arr[idx], p, axis=1)
    lo = float(np.percentile(boot, (1.0 - ci) / 2.0 * 100.0))
    hi = float(np.percentile(boot, (1.0 + ci) / 2.0 * 100.0))
    return {"value": point, "ci_low": lo, "ci_high": hi, "n": int(n)}


def block_bootstrap_percentile_ci(
    blocks,
    p: float,
    n_boot: int = 500,
    ci: float = 0.95,
    seed: int = 83,
    ci_sample_budget: int = 400_000,
) -> dict[str, Any]:
    """Point estimate + BLOCK-bootstrap CI for a percentile of within-block
    autocorrelated samples (e.g. inter-token latencies within one request).

    Naive per-sample bootstrap underestimates the CI when samples are correlated
    within a request (decode cadence is serially correlated). Resampling whole
    requests (blocks) with replacement respects that structure. The point
    estimate uses the FULL pooled sample; the CI resampling caps total work at
    ci_sample_budget by proportionally trimming each block (CI only - the point
    estimate is exact) so a long-decode config can't blow up wall-clock.

    Args:
        blocks: List of 1-D arrays, one per request (its ITL vector).
        p: Percentile in (0, 100).
        n_boot: Bootstrap resamples (blocks).
        ci: Confidence level.
        seed: RNG seed.
        ci_sample_budget: Max samples used for CI resampling.

    Returns:
        {value, ci_low, ci_high, n, n_blocks}.
    """
    clean = [np.asarray(b, dtype=float) for b in blocks if b is not None and len(b) > 0]
    clean = [b[np.isfinite(b)] for b in clean]
    clean = [b for b in clean if b.size > 0]
    if not clean:
        return {"value": None, "ci_low": None, "ci_high": None, "n": 0, "n_blocks": 0}
    pooled = np.concatenate(clean)
    point = float(np.percentile(pooled, p))
    nb = len(clean)
    total = int(pooled.size)
    if nb < 2:
        return {"value": point, "ci_low": point, "ci_high": point,
                "n": total, "n_blocks": nb}
    # Bound CI cost: if pooled is huge, subsample each block by a COMMON fraction
    # (not a flat per-block cap) for the resampling loop only. A flat cap would
    # keep small blocks whole while shrinking large ones, re-weighting the pooled
    # distribution toward equal-per-block and pushing the CI off the (full-data)
    # point estimate. Proportional subsampling preserves each block's relative
    # weight, so the CI still brackets the point. (Point estimate above already
    # used the full data.)
    ci_blocks = clean
    if total > ci_sample_budget:
        frac = ci_sample_budget / total
        rng0 = np.random.default_rng(seed)
        ci_blocks = []
        for b in clean:
            keep = max(1, int(round(b.size * frac)))
            ci_blocks.append(b if keep >= b.size
                             else b[rng0.integers(0, b.size, size=keep)])
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        pick = rng.integers(0, nb, size=nb)
        cat = np.concatenate([ci_blocks[j] for j in pick])
        boot[i] = np.percentile(cat, p)
    lo = float(np.percentile(boot, (1.0 - ci) / 2.0 * 100.0))
    hi = float(np.percentile(boot, (1.0 + ci) / 2.0 * 100.0))
    return {"value": point, "ci_low": lo, "ci_high": hi,
            "n": total, "n_blocks": nb}


def summarize_latency_metric(
    name: str,
    samples,
    percentiles=(50.0, 95.0, 99.0, 99.9),
    ci_percentile: float = 99.0,
    blocks=None,
    n_boot: int = 1000,
    seed: int = 83,
) -> dict[str, Any]:
    """Summarize one pooled latency metric the honest way.

    Reports mean + each requested percentile as a point estimate, a bootstrap
    95% CI on the headline percentile (block bootstrap when ``blocks`` is given,
    for autocorrelated ITL), and a tail_validity flag per percentile so an
    under-sampled tail is never passed off as a hard number. Percentiles are
    computed ONCE from the pooled sample (never averaged across reps).

    Args:
        name: Metric prefix, e.g. "ttft" / "itl" / "tpot" / "e2el".
        samples: Pooled 1-D per-sample values (used for percentiles + iid CI).
        percentiles: Percentiles to report.
        ci_percentile: Which percentile gets the bootstrap CI.
        blocks: If given (list of per-request arrays), use block bootstrap for
            the CI and flatten them for the pooled percentiles.
        n_boot: Bootstrap resamples.
        seed: RNG seed.

    Returns:
        Flat dict of f"{name}_..." keys.
    """
    if blocks is not None:
        clean = [np.asarray(b, dtype=float) for b in blocks
                 if b is not None and len(b) > 0]
        arr = np.concatenate(clean) if clean else np.array([])
        arr = arr[np.isfinite(arr)]   # same finite-filter as the samples path
    else:
        arr = np.asarray(samples, dtype=float)
        arr = arr[np.isfinite(arr)]
    out: dict[str, Any] = {}
    n = int(arr.size)
    out[f"{name}_n"] = n
    if n == 0:
        return out
    out[f"{name}_mean_ms"] = float(np.mean(arr))
    valid_map = {}
    for p in percentiles:
        val = float(np.percentile(arr, p))
        pw = f"{p:g}".replace(".", "_")
        out[f"{name}_p{pw}_ms"] = val
        ok, req = tail_validity(n, p)
        valid_map[f"p{pw}"] = {"valid": ok, "required_n": req}
    out[f"{name}_tail_validity"] = valid_map
    # Bootstrap CI on the headline percentile.
    if blocks is not None:
        ci = block_bootstrap_percentile_ci(blocks, ci_percentile,
                                           n_boot=min(n_boot, 500), seed=seed)
        out[f"{name}_n_blocks"] = ci.get("n_blocks")
    else:
        ci = bootstrap_percentile_ci(arr, ci_percentile, n_boot=n_boot, seed=seed)
    pw = f"{ci_percentile:g}".replace(".", "_")
    out[f"{name}_p{pw}_ci_low_ms"] = ci["ci_low"]
    out[f"{name}_p{pw}_ci_high_ms"] = ci["ci_high"]
    out[f"{name}_ci_percentile"] = ci_percentile
    return out


def aggregate_benchmark_results(
    results: list[dict[str, Any]],
    metrics: list[str],
) -> dict[str, Any]:
    """Aggregate multiple benchmark results with statistics.
    
    Args:
        results: List of individual benchmark result dictionaries
        metrics: List of metric names to aggregate
        
    Returns:
        Aggregated results with statistics for each metric
    """
    if not results:
        return {}
    
    aggregated = {
        "iterations": results,  # Keep raw data
        "num_iterations": len(results),
    }
    
    # Compute statistics for each metric
    for metric in metrics:
        values = []
        for result in results:
            if metric in result and result[metric] is not None:
                values.append(result[metric])
        
        if values:
            stats = compute_statistics(values)
            # Add metric prefix to stat names
            for stat_name, stat_value in stats.items():
                aggregated[f"{metric}_{stat_name}"] = stat_value
    
    return aggregated


def validate_repeatability(
    aggregated: dict[str, Any],
    metric: str,
    max_cv_percent: float = 5.0,
) -> tuple[bool, str]:
    """Validate benchmark repeatability using CV%.
    
    Args:
        aggregated: Aggregated results dictionary
        metric: Metric name to check
        max_cv_percent: Maximum acceptable CV%
        
    Returns:
        Tuple of (is_valid, message)
    """
    cv_key = f"{metric}_cv_percent"
    
    if cv_key not in aggregated:
        return False, f"No CV% data for {metric}"
    
    cv = aggregated[cv_key]
    if cv is None:
        return None, "not assessed (fewer than 2 iterations)"

    if cv <= max_cv_percent:
        return True, f"OK: CV% = {cv:.2f}% (<= {max_cv_percent}%)"
    elif cv <= max_cv_percent * 2:
        return (
            True,
            f"WARNING: CV% = {cv:.2f}% (acceptable but high)",
        )
    else:
        return (
            False,
            f"FAIL: CV% = {cv:.2f}% (> {max_cv_percent}%, poor repeatability)",
        )


def format_statistics_summary(
    aggregated: dict[str, Any],
    metric: str,
) -> str:
    """Format statistics summary for a metric.
    
    Args:
        aggregated: Aggregated results dictionary
        metric: Metric name
        
    Returns:
        Formatted summary string
    """
    mean = aggregated.get(f"{metric}_mean")
    std = aggregated.get(f"{metric}_std")
    cv = aggregated.get(f"{metric}_cv_percent")
    min_val = aggregated.get(f"{metric}_min")
    max_val = aggregated.get(f"{metric}_max")
    
    if mean is None:
        return f"{metric}: N/A"
    
    summary = (f"{metric}: {mean:.2f} +/- {std:.2f}" if std is not None
               else f"{metric}: {mean:.2f} (one iteration: spread not assessed)")
    
    if cv is not None:
        summary += f" (CV={cv:.1f}%)"
    
    if min_val is not None and max_val is not None:
        summary += f" [min={min_val:.2f}, max={max_val:.2f}]"
    
    return summary
