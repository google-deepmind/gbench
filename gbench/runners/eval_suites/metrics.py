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

"""Shared trace-walk finalizers for canonical (non-binary) eval metrics.

``base.run_eval_suite`` scores every sample to a single boolean and reports
``accuracy = correct / scored``. Many benchmarks are NOT pass-rates: DocVQA is mean-ANLS,
Seal-Tools is micro tool/parameter F1, HealthBench is a mean weighted-rubric score. The
established pattern (ruler.py, healthbench.py, mrcr.py, omnidocbench.py, i18n_translate.py,
...) is to run the suite for the binary pass-rate + traces, then walk ``result['sample_traces']``
and OVERWRITE ``result['accuracy']`` with the canonical metric, keeping the binary number as a
secondary field. That override is copy-pasted boilerplate with one subtle, easily-missed
invariant: judge outages carry no score and must be excluded from the mean (an infra outage
must not depress the metric). These helpers own that boilerplate and that invariant once.

Return the SAME ``result`` dict (mutated) so a suite ends with ``return finalize_mean_metric(...)``.

``record_budget_cutoffs`` is the token-limit policy's trace walk: it counts the replies a token
budget cut off (other than at an official cap) and returns the comparability reason they add.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


def scorable_traces(result: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Traces that count toward a metric: a real response, not a judge outage.

    Mirrors base's ``scored_count`` denominator - JUDGE_OUTAGE and ``scoring_excluded`` traces
    (base marks a request that got no reply from the endpoint so) are excluded from numerator
    AND denominator, and a ``None`` response is not a zero.
    """
    for t in result.get("sample_traces", []) or []:
        if t.get("response_text") is None:
            continue
        if str(t.get("judge_grade") or "") == "JUDGE_OUTAGE" or t.get("scoring_excluded"):
            continue
        yield t


def finalize_mean_metric(
    result: Dict[str, Any],
    *,
    metric_name: str,
    score_key: Optional[str] = None,
    scorer: Optional[Callable[[Dict[str, Any]], Optional[float]]] = None,
    stash_key: Optional[str] = None,
    secondary_key: str = "pass_rate",
    scale: float = 100.0,
    failure_score: Optional[float] = None,
    category_scores: bool = False,
) -> Dict[str, Any]:
    """Overwrite ``result['accuracy']`` with the MEAN of a per-sample float metric.

    Provide EXACTLY ONE of:
      * ``score_key`` - each scorable trace already carries this float (e.g. a judge wrote
        ``trace['healthbench_score']``); or
      * ``scorer`` - ``scorer(trace) -> Optional[float]`` recomputes the score from the
        trace (e.g. ``anls_score(trace['response_text'], trace['gold_answer'])``); ``None``
        skips the sample. The value is stashed on the trace under ``stash_key`` (default
        ``score_key`` or ``'score'``) so per-sample scores survive into the report.

    ``failure_score``: when ``None`` (default) a sample with no response or no score is
    EXCLUDED from the mean (matching healthbench/ruler). Set a float (e.g. ``0.0``) to COUNT
    such a sample at that score instead of dropping it - use it when the canonical metric
    treats a non-answer as wrong (MRCR, ANLS), so a model cannot inflate its mean by having
    its non-answers dropped. ``JUDGE_OUTAGE`` is always excluded regardless, and so is a trace
    marked ``scoring_excluded``: a request that got no reply from the served endpoint (base
    marks it: a harness error, never the model's outcome) or a sample the suite left out (for
    example a grader format failure). It is neither scored nor counted at ``failure_score``, as
    in base's ``scored_count``.

    The prior binary ``accuracy`` is preserved under ``secondary_key`` (default
    ``'pass_rate'``); ``result['metric']`` records what ``accuracy`` now means.

    ``category_scores`` (opt-in; suites that restate their categories themselves, such as mrcr
    and wildbench, leave it off): the per-category rows and the counts say the same metric as
    the headline (replace_category_headline): each category's ``accuracy`` is the mean of its
    samples' scores, base's binary tally moves to the cell's ``secondary_key`` and
    ``pass_count``, and ``correct_answers`` becomes the sum of the scores (score-equivalent,
    ``correct_answers_note``), base's binary count moving to ``pass_count``.
    """
    if (score_key is None) == (scorer is None):
        raise ValueError("finalize_mean_metric: pass exactly one of score_key / scorer")
    stash = stash_key or score_key or "score"

    scored: List[Tuple[Dict[str, Any], float]] = []
    for t in result.get("sample_traces", []) or []:
        if str(t.get("judge_grade") or "") == "JUDGE_OUTAGE" or t.get("scoring_excluded"):
            continue
        if t.get("response_text") is None:            # failed / empty request
            if failure_score is not None:
                scored.append((t, float(failure_score)))
            continue
        if score_key is not None:
            if score_key not in t:
                if failure_score is not None:
                    scored.append((t, float(failure_score)))
                continue
            raw = t[score_key]
        else:
            raw = scorer(t)  # type: ignore[misc]
            if raw is None:
                if failure_score is not None:
                    scored.append((t, float(failure_score)))
                continue
            t[stash] = round(float(raw), 6)
        try:
            scored.append((t, float(raw)))
        except (TypeError, ValueError):
            if failure_score is not None:
                scored.append((t, float(failure_score)))
    scores = [v for _t, v in scored]

    result.setdefault(secondary_key, result.get("accuracy"))
    result["metric"] = metric_name
    if scores:
        result["accuracy"] = round(sum(scores) / len(scores) * scale, 2)
        result["scored_samples"] = len(scores)
    if category_scores:
        replace_category_headline(result, scored, metric_name=metric_name,
                                  binary_key=secondary_key, scale=scale)
    return result


def replace_category_headline(result: Dict[str, Any],
                              scored: List[Tuple[Dict[str, Any], float]], *,
                              metric_name: str, binary_key: str = "pass_rate",
                              scale: float = 100.0) -> Dict[str, Any]:
    """Make the per-category rows and the counts of a mean-metric result say its headline metric.

    ``scored``: (trace, per-sample score in [0, 1]) of every sample the headline counts. Each
    category_accuracy cell gets ``accuracy`` = the mean of its samples' scores x ``scale`` and
    ``correct`` = their sum (score-equivalent), base's binary tally kept as the cell's
    ``binary_key`` (its pass rate) and ``pass_count``; ``correct_answers`` becomes the sum of
    every score (``correct_answers_note`` says so) and base's binary count moves to
    ``pass_count``. The summary's Correct column then reconciles with the Accuracy column (sum
    of scores / scored samples), instead of pairing a pass count with a mean score."""
    by_cat: Dict[Any, List[float]] = {}
    for t, v in scored:
        by_cat.setdefault(t.get("category"), []).append(float(v))
    cats = result.get("category_accuracy")
    if isinstance(cats, dict):
        for cat, cell in cats.items():
            if not isinstance(cell, dict):
                continue
            values = by_cat.get(cat, [])
            cell.setdefault(binary_key, cell.get("accuracy"))
            cell.setdefault("pass_count", cell.get("correct"))
            cell["accuracy"] = round(sum(values) / len(values) * scale, 2) if values else None
            cell["correct"] = round(sum(values), 4)
            cell["total"] = len(values) if values else cell.get("total", 0)
    if scored:
        result.setdefault("pass_count", result.get("correct_answers"))
        result["correct_answers"] = round(sum(v for _t, v in scored), 4)
        result["correct_answers_note"] = (
            f"sum of the per-sample scores of the headline metric ({metric_name}), a "
            "score-equivalent count; pass_count is the binary pass count")
    return result


def pooled_prf(result: Dict[str, Any], count_key: str) -> Dict[str, float]:
    """Corpus-pooled (micro) precision/recall/F1 from per-sample ``(tp, fp, fn)`` counts.

    Each scorable trace carries ``trace[count_key]`` = ``(tp, fp, fn)`` or a dict with those
    keys (produce them with ``fc_common.prf_counts``). P/R/F1 pool counts across the whole
    run - they are NOT averageable per sample, so this must not be a mean of per-sample
    ratios. Returns the prf dict; the caller decides which figure becomes the headline.
    """
    tp = fp = fn = 0
    for t in scorable_traces(result):
        c = t.get(count_key)
        if c is None:
            continue
        if isinstance(c, dict):
            tp += int(c.get("tp", 0)); fp += int(c.get("fp", 0)); fn += int(c.get("fn", 0))
        else:
            a, b, d = c
            tp += int(a); fp += int(b); fn += int(d)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def budget_cutoffs(result: Dict[str, Any]) -> Tuple[int, int]:
    """(replies cut off by the run's token budget, how many of them have no answer text).

    A reply that stopped at max_tokens (finish_reason "length") on a request that carried the
    suite's official cap (one of base's ``official_max_tokens_sent``) is the protocol's outcome,
    scored as the upstream scores it. Any other reply that stopped there was cut off by this
    run's budget: a --thinking run's --max-output-tokens (or the thinking floor), a max_tokens
    the context window lowered, or the end of the window when no max_tokens was sent. One with
    no answer text ran out inside the reasoning channel.
    """
    official = set(result.get("official_max_tokens_sent") or [])
    cut = [t for t in result.get("sample_traces", []) or []
           if t.get("finish_reason") == "length" and t.get("response_text") is not None
           and t.get("max_tokens_sent") not in official]
    return len(cut), sum(1 for t in cut if not str(t.get("response_text") or "").strip())


def record_budget_cutoffs(result: Dict[str, Any],
                          parts: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
    """Count the replies the run's token budget cut off (see budget_cutoffs) on ``result`` and
    return the leaderboard_comparable reason they add, or None.

    ``parts`` are the results whose traces hold the replies when ``result`` does not (a
    multi-turn suite's per-turn results); by default ``result`` itself. Records
    ``replies_cut_off_by_budget`` and ``replies_cut_off_in_reasoning``. Such a reply is scored
    on what arrived, but it measures the budget rather than the model, so the run is not
    comparable.
    """
    cut = in_reasoning = 0
    for part in (parts if parts is not None else [result]):
        c, r = budget_cutoffs(part)
        cut, in_reasoning = cut + c, in_reasoning + r
    result["replies_cut_off_by_budget"] = cut
    result["replies_cut_off_in_reasoning"] = in_reasoning
    if not cut:
        return None
    return (f"{cut} repl(ies) cut off by this run's token budget, not by an official cap "
            f"({in_reasoning} of them inside the reasoning channel with no answer): "
            "--max-output-tokens, a max_tokens the context window lowered, or the end of the "
            "window where no max_tokens was sent")


def finalize_f1_metric(
    result: Dict[str, Any],
    *,
    count_key: str,
    metric_name: str,
    secondary_key: str = "pass_rate",
    scale: float = 100.0,
) -> Dict[str, Any]:
    """Overwrite ``result['accuracy']`` with a corpus-pooled micro-F1 headline.

    Convenience wrapper over :func:`pooled_prf`: also records precision/recall as percentages.
    """
    prf = pooled_prf(result, count_key)
    result.setdefault(secondary_key, result.get("accuracy"))
    result["metric"] = metric_name
    result["precision"] = round(prf["precision"] * scale, 2)
    result["recall"] = round(prf["recall"] * scale, 2)
    result["f1"] = round(prf["f1"] * scale, 2)
    if (prf["tp"] + prf["fp"] + prf["fn"]) > 0:
        result["accuracy"] = round(prf["f1"] * scale, 2)
    return result
