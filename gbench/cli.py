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

"""Command-line interface for the gbench benchmark suite.

This module provides a rich CLI with comprehensive options for running
benchmarks on open models (any HuggingFace id, local checkpoint directory or
OpenAI-compatible endpoint).
"""

import argparse
import logging
import numbers
import contextlib
import os
import re
import shlex
import signal
import socket
import sys
import time
from types import ModuleType
dummy_check = ModuleType("transformers.dependency_versions_check")
def dep_version_check(*args, **kwargs):
    pass
dummy_check.dep_version_check = dep_version_check
# GEMINI_API_KEY / GEMINI_API_KEYS are normalized once, in gbench/__init__.py (imported first).

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import __version__
from .core import (
    BenchmarkConfig,
    DEFAULT_CONFIG,
    DEFAULT_GEMMACLAW_COMMIT,
    QUICK_CONFIG,
    ModelFormat,
    registry,
    check_gpu_ready,
    validate_gpu_config,
    weights_fit_advisory,
    get_available_gpus,
    get_batch_sizes,
    resolve_campaign_slo,
    campaign_ctx,
)
from .core.config import (
    VLLM_SERVE_ARGS_ENV,
    match_served_model,
    normalize_endpoint,
    parallel_layout,
    parse_vllm_serve_args,
    resolve_server_port,
    select_gpus,
    serve_arg_value,
    serve_args_for,
    serve_settings_record,
)

logger = logging.getLogger(__name__)

# Exit codes. A caller pointing gbench at its own endpoint needs to tell
# "the model got something wrong" apart from "the run never happened",
# because only the first is evidence about the model.
EXIT_OK = 0
EXIT_MODEL_FAILURE = 1
EXIT_HARNESS_ERROR = 2

GOLDEN_VERDICT = {"passed": "PASS", "failed": "FAIL", "error": "ERROR",
                  "not_applicable": "N/A"}


def golden_exit_code(results: list[dict]) -> int:
    """Reduce Golden Set results to a process exit code.

    Harness errors outrank model failures: a run that could not reach the
    endpoint has no verdict on the model at all, so it must not be
    reported with the same code as a real regression.
    """
    golden = [r for r in results if r.get("benchmark_type") == "golden"]
    if any(r.get("status") == "error" for r in golden):
        return EXIT_HARNESS_ERROR
    if any(r.get("status") == "failed" for r in golden):
        return EXIT_MODEL_FAILURE
    return EXIT_OK


def golden_category_breakdown(task_results: list[dict]) -> list[dict]:
    """Aggregate per-case Golden Set results into one row per category.

    A bare "9/12" says something regressed but not where, which is the
    first thing you need in order to judge whether it matters. Losing
    both tool_use cases is a different morning from losing one
    translation.

    Args:
        task_results: Per-case result dicts from the Golden runner.

    Returns:
        One row per category, worst first, so problem areas sort to the
        top instead of landing wherever the alphabet puts them. The
        offending task ids are deliberately not repeated here: the
        FAIL and ERROR lines printed below the table already name each
        one, along with the detail you need to reproduce it.
    """
    buckets: dict[str, dict] = {}
    for task in task_results:
        category = task.get("category") or "uncategorized"
        bucket = buckets.setdefault(category, {
            "category": category,
            "total": 0,
            "passed": 0,
            "na": 0,
            "statuses": set(),
        })
        bucket["total"] += 1
        status = task.get("status")
        bucket["statuses"].add(status)
        if status == "passed":
            bucket["passed"] += 1
        elif status == "not_applicable":
            bucket["na"] += 1

    rows = []
    for bucket in buckets.values():
        # Same precedence as golden_exit_code: a category that could not
        # be measured is not a category that passed. A category that is
        # entirely not-applicable (the model lacks that modality) is N/A,
        # not a pass.
        if "error" in bucket["statuses"]:
            bucket["status"] = "error"
        elif "failed" in bucket["statuses"]:
            bucket["status"] = "failed"
        elif bucket["na"] == bucket["total"] and bucket["passed"] == 0:
            bucket["status"] = "not_applicable"
        else:
            bucket["status"] = "passed"
        del bucket["statuses"]
        rows.append(bucket)

    rank = {"error": 0, "failed": 1, "passed": 2, "not_applicable": 3}
    return sorted(rows, key=lambda b: (rank.get(b["status"], 4),
                                       b["category"]))


def _headline_metrics_text(result: dict) -> str:
    """A result's `headline_metrics` as `name=value; name=value`; for a result with no headline
    (accuracy null) its `accuracy_withheld_reason`; else empty."""
    text = "; ".join(f"{k}={v}" for k, v in (result.get("headline_metrics") or {}).items())
    if not text and result.get("accuracy") is None and result.get("accuracy_withheld_reason"):
        reason = " ".join(str(result["accuracy_withheld_reason"]).split())
        text = f"accuracy withheld: {reason[:200]}"
    return text


def _pct_cell(value) -> str:
    """A 10-character accuracy cell: NN.NN% right-aligned, or '-' for a null accuracy (a suite
    whose headline is not scored, for example a category or metric reported infra_required).
    Any real number is a number here, numpy scalars (np.float32, np.int64) included; a bool is
    not."""
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return f"{float(value):>9.2f}%"
    return f"{'-':>9} "


def _evals_only(args) -> bool:
    """True for a run that measures evals only (--evals-only, or --evals without a performance
    campaign or explicit input/output lengths)."""
    return bool(getattr(args, "evals_only", False) or (
        bool(getattr(args, "evals", None))
        and not getattr(args, "campaign", None)
        and not (getattr(args, "input_lengths", None) or getattr(args, "output_lengths", None))
    ))


def _benchmark_mode(args) -> str:
    """The run's benchmark mode, as logged ("Benchmark mode: ...") and recorded in
    metadata.json: evals only, golden set only, quality only, multimodal only, text and
    multimodal (for capable models), or text only."""
    evals_only = _evals_only(args)
    if evals_only:
        return "evals only"
    if getattr(args, "golden_only", False):
        return "golden set only"
    if getattr(args, "quality_only", False):
        return "quality only"
    if getattr(args, "multimodal_only", False):
        return "multimodal only"
    run_multimodal = not getattr(args, "text_only", False)
    return "text and multimodal (for capable models)" if run_multimodal else "text only"


def _print_eval_failure(r: dict, model: str, fmt: str, suite: str, thinking: str) -> None:
    """One console row for an eval configuration that produced no score: its failure label
    (INFRA REQUIRED, ERROR, TIMEOUT) across the Questions/Correct/Accuracy columns, never a
    number, then the reason on its own line."""
    from .runners.evals import failure_label
    print(f"{model:<18} {fmt:<16} {suite:<36} {thinking:<10} {failure_label(r):>30}")
    reason = " ".join(str(r.get("error") or r.get("skip_reason")
                          or r.get("accuracy_withheld_reason") or "").split())
    if reason:
        print(f"  - {reason[:300]}{'...' if len(reason) > 300 else ''}")


def _reconciles(correct, denominators, accuracy) -> bool:
    """Whether `accuracy` (a percent, rounded to 2 decimals) is `correct` over one of
    `denominators`. A Correct count printed next to a headline that is not correct/total (a mean
    score such as ANLS or a sequence-match ratio, a corpus metric, an F1) would read as its
    numerator; those cells print '-' instead."""
    if not isinstance(accuracy, numbers.Real) or isinstance(accuracy, bool):
        return True
    if not isinstance(correct, numbers.Real) or isinstance(correct, bool):
        return False
    dens = [d for d in denominators
            if isinstance(d, numbers.Real) and not isinstance(d, bool) and d > 0]
    if not dens:
        return float(correct) == 0.0
    return any(abs(100.0 * float(correct) / float(d) - float(accuracy)) <= 0.011 for d in dens)


def _correct_cell(result: dict, total_q):
    """The Correct cell of a suite row: `correct_answers` when the headline is correct over the
    questions or the scored items (_reconciles), else '-'."""
    correct = result.get("correct_answers")
    if _reconciles(correct, (_effective_n(result, total_q), total_q), result.get("accuracy")):
        return correct if correct is not None else "-"
    return "-"


def _attempt_split(result: dict, total_q) -> Optional[Tuple[int, int]]:
    """(questions, attempts per question) when a suite's protocol answers each question several
    times (gpqa's 10, lcb's 10, aime's 4, ...), so `total_questions` counts generations; read from
    the run's eval_limit_plan (else `attempts`) and used only when questions x attempts equals
    `total_questions`. None otherwise."""
    plan = result.get("eval_limit_plan") if isinstance(result.get("eval_limit_plan"), dict) else {}
    attempts = result.get("attempts") if isinstance(result.get("attempts"), dict) else {}
    per = plan.get("attempts_per_sample") or attempts.get("attempts_per_sample")
    samples = plan.get("samples")
    try:
        per, samples, total = int(per or 1), int(samples or 0), int(total_q or 0)
    except (TypeError, ValueError):
        return None
    if per > 1 and samples and samples * per == total:
        return samples, per
    return None


def _questions_cell(result: dict, total_q):
    """The Questions cell of a suite row: `5x10` (questions x attempts) when the protocol answers
    each question several times (_attempt_split), else the question count."""
    split = _attempt_split(result, total_q)
    return f"{split[0]}x{split[1]}" if split else total_q


def _category_cells(cstats: dict):
    """(Correct, accuracy) of one category_accuracy cell as the summary prints them: the count
    only when the cell's accuracy is correct/total (_reconciles), else '-'."""
    ctot = cstats.get("total", 0)
    ccorr = cstats.get("correct", 0)
    cacc = cstats.get("accuracy")
    return (ccorr if _reconciles(ccorr, (ctot,), cacc) else "-"), cacc


def _category_effective_n(result: dict, category, ctot):
    """A category row's Effective N: its question count, except for a run with several attempts
    per question, where it is the distinct questions (source_sample_idx) scored in the category,
    as the suite's own effective_n counts them ('-' when the traces are not in the result)."""
    attempts = result.get("attempts")
    per = attempts.get("attempts_per_sample", 1) if isinstance(attempts, dict) else 1
    if not isinstance(per, int) or per <= 1:
        return ctot
    traces = result.get("sample_traces")
    if not isinstance(traces, list) or not traces:
        return "-"
    return len({t.get("source_sample_idx", t.get("sample_idx")) for t in traces
                if isinstance(t, dict) and t.get("category") == category
                and str(t.get("judge_grade") or "") != "JUDGE_OUTAGE"
                and not t.get("scoring_excluded")})


def _effective_n(result: dict, total_q):
    """The summary's Effective N: the suite's own `effective_n` (its headline's denominator) when
    it records one; else, for a run with one trace per question, the traces that count in the
    score (as base.is_scored: not a judge outage and not `scoring_excluded`, so a request that got
    no reply from the served endpoint is not counted); else the question count."""
    if result.get("effective_n") is not None:
        return result["effective_n"]
    traces = result.get("sample_traces")
    if (isinstance(traces, list) and traces and len(traces) == total_q
            and all(isinstance(t, dict) for t in traces)):
        return sum(1 for t in traces
                   if str(t.get("judge_grade") or "") != "JUDGE_OUTAGE"
                   and not t.get("scoring_excluded"))
    return total_q


def _save_eval_summary_csv(
    results_dir: Path,
    eval_results: list,
    eval_failures: list,
    eval_pillars: list,
    custom_pillars: dict,
) -> Optional[Path]:
    """Save clean, structured CSV report for all evaluation results to eval_summary.csv."""
    csv_path = results_dir / "eval_summary.csv"
    suite_to_pillar = {}
    for p_title, keys in eval_pillars:
        clean_title = re.sub(r"^\d+\.\s*", "", p_title).strip()
        for k in keys:
            suite_to_pillar[k.lower()] = clean_title

    fieldnames = [
        "Pillar",
        "Model",
        "Format",
        "Eval Suite",
        "Subcategory",
        "Thinking",
        "Questions",
        "Effective N",
        "Correct",
        "Accuracy (%)",
        "Headline Metrics",
        "Status",
        "Duration (s)",
    ]

    rows = []
    all_runs = [(r, True) for r in eval_results] + [(r, False) for r in eval_failures]

    for r, is_success in all_runs:
        eval_name = r.get("eval_name", "").lower()
        model = r.get("model_short", r.get("model_name", "N/A"))
        fmt = r.get("format", "N/A")
        thinking = "Yes" if r.get("thinking", False) else "No"
        total_q = r.get("total_questions", 0) if is_success else 0
        effective_n = _effective_n(r, total_q) if is_success else "-"
        correct = _correct_cell(r, total_q) if is_success else 0
        # accuracy can be present-but-None (e.g. omnidocbench's composite is undefined when the
        # sampled pages have no tables/formulas); .get(...,0.0) does NOT substitute the default for
        # an existing None, so `f"{None:.2f}"` used to crash the whole model summary (outside the
        # per-eval guard) and abort the run. Treat a non-numeric accuracy as "-".
        _acc = r.get("accuracy")
        acc = f"{_acc:.2f}" if (is_success and isinstance(_acc, (int, float))) else "-"
        status = r.get("status", "success" if is_success else "failed")
        if not is_success:
            # the failure's kind: infra_required, error or timeout (evals.failure_kind)
            from .runners.evals import failure_kind_of
            status = failure_kind_of(r)
        duration = r.get("duration_s", "")
        # A suite whose official headline is more than one number (cimemories: Violation@5,
        # lower is better, with Completeness) reports them all on its suite row.
        headline = _headline_metrics_text(r) if is_success else ""

        if eval_name in suite_to_pillar:
            pillar = suite_to_pillar[eval_name]
        elif eval_name in custom_pillars:
            clean_cp = re.sub(r"^\d+\.\s*", "", custom_pillars[eval_name]).strip()
            pillar = f"[CUSTOM] {clean_cp}"
        else:
            pillar = "[CUSTOM] OTHER"

        # Extract subcategory information
        cat_acc = r.get("category_accuracy", {})
        traces = r.get("sample_traces", [])
        # A suite without category_accuracy: its row is named after its traces' category only
        # when every trace shares it (else the row is the suite's OVERALL, e.g. a multi-category
        # average), never after the first trace's.
        trace_cats = {t.get("category") for t in traces if isinstance(t, dict)} if traces else set()
        trace_cat = (next(iter(trace_cats)) if len(trace_cats) == 1 and None not in trace_cats
                     else ("OVERALL" if trace_cats else None))

        # One category row stands for the suite only when it carries every question: a run
        # whose other categories were left out of the score (harness errors, excluded samples)
        # gets an OVERALL row plus the category row instead.
        only = next(iter(cat_acc.values())) if len(cat_acc) == 1 else None
        if only is not None and (not isinstance(only, dict)
                                 or only.get("total", total_q) == total_q):
            single_cat = list(cat_acc.keys())[0]
            rows.append({
                "Pillar": pillar,
                "Model": model,
                "Format": fmt,
                "Eval Suite": eval_name.upper(),
                "Subcategory": single_cat,
                "Thinking": thinking,
                "Questions": total_q,
                "Effective N": effective_n,
                "Correct": correct,
                "Accuracy (%)": acc,
                "Headline Metrics": headline,
                "Status": status,
                "Duration (s)": duration,
            })
        elif cat_acc:
            # Main overall suite row
            rows.append({
                "Pillar": pillar,
                "Model": model,
                "Format": fmt,
                "Eval Suite": eval_name.upper(),
                "Subcategory": "OVERALL",
                "Thinking": thinking,
                "Questions": total_q,
                "Effective N": effective_n,
                "Correct": correct,
                "Accuracy (%)": acc,
                "Headline Metrics": headline,
                "Status": status,
                "Duration (s)": duration,
            })
            # Individual subcategory rows
            for cat, cstats in sorted(cat_acc.items()):
                ctot = cstats.get("total", 0)
                ccorr, _cacc = _category_cells(cstats)   # _cacc can be None (undefined metric)
                cacc = f"{float(_cacc):.2f}" if isinstance(_cacc, (int, float)) else "-"
                rows.append({
                    "Pillar": pillar,
                    "Model": model,
                    "Format": fmt,
                    "Eval Suite": eval_name.upper(),
                    "Subcategory": cat,
                    "Thinking": thinking,
                    "Questions": ctot,
                    "Effective N": _category_effective_n(r, cat, ctot),
                    "Correct": ccorr,
                    "Accuracy (%)": cacc,
                    "Headline Metrics": "",
                    "Status": status,
                    "Duration (s)": "",
                })
        else:
            # Fallback to trace category if available
            rows.append({
                "Pillar": pillar,
                "Model": model,
                "Format": fmt,
                "Eval Suite": eval_name.upper(),
                "Subcategory": trace_cat or "-",
                "Thinking": thinking,
                "Questions": total_q,
                "Effective N": effective_n,
                "Correct": correct,
                "Accuracy (%)": acc,
                "Headline Metrics": headline,
                "Status": status,
                "Duration (s)": duration,
            })

    if not rows:
        return None

    from .utils.log_manager import write_redacted_csv
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        # The shared results writer: the rendered CSV text is redacted after serialization.
        write_redacted_csv(csv_path, rows, fieldnames)
        logger.info(f"Saved evaluation CSV report to: {csv_path}")
        return csv_path
    except Exception as e:
        logger.error(f"Failed to write CSV summary to {csv_path}: {e}")
        return None


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    for noisy_logger in ["httpx", "httpcore", "datasets", "huggingface_hub", "urllib3", "fsspec", "filelock", "google_genai", "google", "grpc"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    # No API key or token of the environment reaches a log line (an exception that quotes a
    # command line, a streamed harness log).
    from .utils.redaction import install_log_redaction
    install_log_redaction()


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Open Model Performance Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test on one model (a HuggingFace id or a local checkpoint directory)
  gbench --preset quick --models google/gemma-4-E4B-it --num-gpus 1

  # Two models, text and multimodal performance
  gbench --models google/gemma-4-E4B-it google/gemma-4-31B-it --num-gpus 2

  # Multimodal benchmarks only (for models that support images)
  gbench --models google/gemma-4-E4B-it --num-gpus 1 --multimodal-only

  # Tool-calling evals on a server gbench launches: pass the model's parsers
  gbench --models org/model --num-gpus 1 --evals bfcl --max-output-tokens 16384 \\
         --tool-call-parser <parser> --reasoning-parser <parser>

  # Dry run: the plan, the server commands, ports and GPUs, without running anything
  gbench --dry-run --models google/gemma-4-E4B-it --num-gpus 1

Environment knobs: every per-suite knob is GBENCH_<SUITE>_<KNOB> (older bare names like TAU2_* still
work as deprecated aliases). Full list incl. decoding/judge/search knobs: docs/evals/env-knobs.md.
Per-suite temperature: GBENCH_<SUITE>_TEMPERATURE (overrides --temperature).

Exit codes: 0 every pillar produced its result; 1 a model verdict failed (Golden Set failing
cases); 2 a harness error (a server that did not start, a suite without a score, a pillar that
raised, an unreachable endpoint): not a verdict on the model.
        """,
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"gbench {__version__}",
    )

    # Model selection
    model_group = parser.add_argument_group("Model Selection")
    model_group.add_argument(
        "--models",
        nargs="+",
        help="Models to benchmark: HuggingFace ids (e.g. google/gemma-4-31B-it) or local "
             "checkpoint directories. With --remote-endpoint, the id the endpoint serves (e.g. an "
             "Ollama tag), with --tokenizer naming the model's tokenizer.",
    )
    model_group.add_argument(
        "--format",
        choices=["hf", "gguf", "both"],
        default="hf",
        help="Model format to serve locally: hf (the HuggingFace checkpoint, the only one gbench "
             "serves). gguf and both are refused.",
    )
    model_group.add_argument(
        "--served-model-id",
        "--golden-model-id",
        "--eval-model-id",
        dest="served_model_id",
        default=None,
        help="The id the endpoint serves the model under, when it differs from --models (e.g. "
             "an endpoint that serves several models, or a tag other than the HuggingFace id). "
             "It must be listed by the endpoint's /v1/models. Without it the served id must "
             "match the model's HuggingFace id, path or name, or the endpoint must serve exactly "
             "one model when --models is not given. Applies with --remote-endpoint: the server "
             "gbench launches serves the model under its --models id or path, so a local run "
             "refuses any other value. --golden-model-id and --eval-model-id are aliases.",
    )

    # Benchmark type selection
    bench_group = parser.add_argument_group("Benchmark Types")
    bench_group.add_argument(
        "--personal",
        action="store_true",
        help="Personal-device mode: 'what will this model do on my machine'. "
             "Single-stream, text-only, serving-only, few samples, full decode; "
             "reports peak decode tok/s + TTFT. Use with --remote-endpoint (e.g. "
             "Ollama). Incompatible with --campaign.",
    )
    bench_group.add_argument(
        "--serving-only",
        action="store_true",
        help="Run only serving benchmarks",
    )
    bench_group.add_argument(
        "--throughput-only",
        action="store_true",
        help="Run only throughput benchmarks",
    )
    bench_group.add_argument(
        "--text-only",
        action="store_true",
        help="Run only text benchmarks, skip multimodal even for capable models",
    )
    bench_group.add_argument(
        "--multimodal-only",
        action="store_true",
        help="Run only multimodal benchmarks (for models that support it)",
    )
    bench_group.add_argument(
        "--stress-test",
        action="store_true",
        help="Run only stress test (ramp-up to find max sustainable throughput)",
    )
    bench_group.add_argument(
        "--no-stress-test",
        action="store_true",
        help="Skip stress test (included by default in all presets)",
    )
    bench_group.add_argument(
        "--stress-threshold",
        type=int,
        default=None,
        help="P99 TTFT SLO in ms for the stress QPS sweep. GLOBAL override "
             "(applies to every campaign); default None => the per-campaign "
             "SLO map (config.CAMPAIGN_SLOS: 1000 ms chat-like and decode-heavy, 2500 ms "
             "mixed, 5000 ms agentic, prefill-heavy and long-decode); without --campaign: "
             "5000 ms TTFT / 200 ms ITL. Tunable for laptop/H100/GB200.",
    )
    bench_group.add_argument(
        "--itl-slo-ms",
        "--stress-tpot-threshold",
        dest="itl_slo_ms",
        type=int,
        default=None,
        help="P99 ITL/TBT (inter-token, decode-cadence) SLO in ms for the stress "
             "QPS sweep. GLOBAL override; default None => the per-campaign map "
             "(100 ms, about 10 tok/s without speculative decoding, for every campaign); "
             "without --campaign: 5000 ms TTFT / 200 ms ITL. A rate is sustainable only if "
             "BOTH the TTFT and the ITL/TBT SLO hold. The P99 comes from the server ITL "
             "histogram (one gap per engine step) when it decides, else the client per-token "
             "gap, else the client gap between streamed chunks; under speculative decoding the "
             "client per-token gap (or, without a tokenizer, the server histogram against SLO x "
             "tokens per step). "
             "(--stress-tpot-threshold is a back-compat alias.)",
    )
    bench_group.add_argument(
        "--stress-reps",
        type=int,
        default=None,
        help="Sweeps per stress run; each gives a bracket [pass, fail] and the median "
             "repeat's limit is certified (--stress-certify). Env GBENCH_STRESS_REPS (flag "
             "wins). Default None => 3.",
    )
    bench_group.add_argument(
        "--stress-client-procs",
        type=int,
        default=None,
        help="OS processes for the multi-process stress load generator (removes "
             "the single-asyncio-client CPU-contention confound at the knee). "
             "Auto-capped to cpu_count-1, so a large value is safe on small hosts. "
             "Env GBENCH_STRESS_CLIENT_PROCS (flag wins). Default None => min(32, cpu_count-1).",
    )
    bench_group.add_argument(
        "--stress-max-qps",
        type=float,
        default=None,
        help="Safety cap on the open-loop arrival-rate sweep (env GBENCH_STRESS_MAX_QPS; "
             "flag wins). Default None => 8192. Raise it further only if a host passes the SLO "
             "at the cap (the knee is then censored at the cap, not measured); pair with a higher "
             "--stress-client-procs so the client can offer the extra load without becoming "
             "the bottleneck (weak clients hit client_bound first and are excluded).",
    )
    bench_group.add_argument(
        "--stress-max-prompts",
        type=int,
        default=None,
        help="Optional cap on requests per stress rate point (env GBENCH_STRESS_MAX_PROMPTS; "
             "flag wins), recorded when set. Default: no cap - a point sends max(40, rate x "
             "60 s, rate x (3 x latency + 10 s)) requests, never fewer.",
    )
    bench_group.add_argument(
        "--stress-min-samples",
        type=int,
        default=None,
        help="Min requests in a stress point's cohort (those planned after its ramp-up) to "
             "judge it (env GBENCH_STRESS_MIN_SAMPLES; flag wins). Default None => 15. "
             "Lowering it only makes the bracket sweep cheaper; when certification runs, "
             "the reported limit is certified with percentile-sized probes either way.",
    )
    bench_group.add_argument(
        "--stress-certify",
        choices=["auto", "always", "off"],
        default=None,
        help="Certify the reported stress limit with percentile-sized probes (a cohort of "
             ">= 1099 requests for the P99 SLO, about 1400 sent, the same rule on every device) inside the sweep's bracket "
             "(env GBENCH_STRESS_CERTIFY; default auto = within --stress-certify-budget; "
             "always = ignore the budget; off = report the bracket, labelled UNCERTIFIED).",
    )
    bench_group.add_argument(
        "--stress-certify-budget",
        type=float,
        default=None,
        help="Minutes the stress certification may take, per stress pass (text and "
             "multimodal each; env GBENCH_STRESS_CERTIFY_BUDGET_MIN; default 120; 0 = do not "
             "certify in auto mode; always ignores the budget). The budget is checked before each probe. If it runs out after a "
             "probe passed, that rate is reported as a certified lower bound; if none passed, "
             "the bracket is reported UNCERTIFIED with the time certifying would need.",
    )
    bench_group.add_argument(
        "--quality",
        action="store_true",
        help="Run quality (gemmaclaw agentic) benchmarks",
    )
    bench_group.add_argument(
        "--quality-only",
        action="store_true",
        help="Run only quality (gemmaclaw agentic) benchmarks",
    )
    bench_group.add_argument(
        "--scenarios",
        nargs="+",
        help="GemmaClaw QA scenario ids to run (the `id:` of a qa/scenarios/**/*.md at the "
             "pinned commit, e.g. memory-recall mcp-plugin-tools-call; not file paths), each "
             "passed to the suite as --scenario <id>",
    )
    bench_group.add_argument(
        "--golden",
        action="store_true",
        help="Run Golden Set deterministic smoke-test tasks",
    )
    bench_group.add_argument(
        "--golden-only",
        action="store_true",
        help="Run only Golden Set deterministic smoke-test tasks",
    )
    bench_group.add_argument(
        "--golden-tasks",
        nargs="+",
        help="List of specific Golden task IDs or JSON files to run",
    )

    # Evaluation benchmarks
    eval_group = parser.add_argument_group("Evaluation Benchmarks")
    eval_group.add_argument(
        "--evals",
        nargs="+",
        help="Select evaluation benchmark suites to run (space or comma-separated, or 'all', 'plugins')",
    )
    eval_group.add_argument(
        "--eval-thinking",
        "--enable-thinking",
        "--thinking",
        dest="eval_thinking",
        action="store_true",
        help="Let the model think/reason (default OFF everywhere). For SERVING, "
             "gbench otherwise sends a no-think system prompt so a reasoning model "
             "(e.g. gemma-4 on Ollama) measures pure answer generation, not an "
             "accidental chain-of-thought workload (which inflates TTFT and causes "
             "0-content 'empty' replies). For EVALS, enables the model's reasoning mode: "
             "native suites send chat_template_kwargs.enable_thinking on every request, "
             "and delegated harnesses that take the toggle forward it (e.g. livebench, "
             "aider_polyglot). Some suites also change their prompt with it (e.g. "
             "cruxeval, screenspot_pointing).",
    )
    eval_group.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum generation tokens per eval response. REQUIRED for eval runs: it bounds "
             "generation and materially affects scores (truncation reads as a failure), so it "
             "must be explicit for runs to be comparable. e.g. --max-output-tokens 65536.",
    )
    eval_group.add_argument(
        "--eval-max-soft-tokens",
        type=int,
        default=None,
        help="The served model's processor setting max_soft_tokens (the image soft-token budget) "
             "for vision evals on a server gbench launches; a value the model's processor does "
             "not accept is refused by the server. Only for a model whose processor declares a "
             "max_soft_tokens setting (checked before the run: a model without one is an error). "
             "Default: the model's own processor setting.",
    )
    eval_group.add_argument(
        "--suite-timeout",
        type=int,
        default=None,
        help="Per-suite wall-clock budget in seconds (default: OFF). When set, a suite "
             "that exceeds it is reported as 'timeout' and the sweep continues, so one "
             "wedged harness cannot stall the run. Destructive: the suite's partial "
             "results are lost and its child processes/containers are not reaped, so "
             "leave it unset unless running unattended.",
    )
    eval_group.add_argument(
        "--eval-n-shot",
        type=int,
        default=None,
        help="Few-shot exemplar count for MMLU (dev split), MMLU-Pro (validation split), and GSM8K (the fixed Wei et al. 8-shot CoT exemplars, not train rows). Default: canonical 5-shot for mmlu/mmlu_pro and 8-shot for gsm8k; pass 0 for 0-shot. mmlu/mmlu_pro accept 0-5 and gsm8k 0-8; a larger value is an error.",
    )
    eval_group.add_argument(
        "--eval-categories",
        type=str,
        default=None,
        help="Comma-separated category override, routed per suite: bfcl takes any of "
             "BFCL's 22 scoring categories (e.g. 'simple_python,multi_turn_base,memory_kv'; "
             "groups: all, all_scoring, single_turn, non_live, live, multi_turn, agentic, "
             "memory, web_search, python, non_python), bfcl_v4_agentic its agentic ones "
             "(memory_kv, memory_vector, memory_rec_sum, web_search_base, "
             "web_search_no_snippet); swe_bench_live takes its SWE-bench-Live leaderboard "
             "settings (lite, full, verified, ccpp, csharp, java, tsjs, go, rust, windows; "
             "default all ten). Each suite gets only its own "
             "names and runs its full track when it gets none; a name no selected suite knows "
             "is an error. Only with --evals bfcl, bfcl_v4_agentic, swe_bench_live or all.",
    )
    eval_group.add_argument(
        "--eval-limit",
        "--limit",
        dest="eval_limit",
        type=int,
        default=None,
        help="Run N samples per suite in total (e.g. --eval-limit 10). A sample is one distinct "
             "dataset item (question, task, instance, problem, conversation). N is the total for "
             "the whole suite, never N per tab, split, setting, language, domain, scenario, pass "
             "or mode: a suite with several settings spreads the N over them (a setting that "
             "gets 0 is not run). The attempts a protocol makes per sample (lcb's n=10, "
             "arc_agi's 2 attempts, avg@k / pass@k repeats, --attempt-count) are unchanged and "
             "are not extra samples. Each suite logs its planned workload in one line at its "
             "start and records it in its result as eval_limit_plan (limit, samples, "
             "per_setting, attempts_per_sample, and the unit total: generations, rollouts, "
             "episodes or conversations, named for what one attempt is); under --shard the plan "
             "counts that shard's samples. Without it every suite runs its full set.",
    )
    eval_group.add_argument(
        "--shard",
        dest="shard",
        type=str,
        default=None,
        help="Run shard I of N (e.g. --shard 3/8) to split each eval across machines: a "
             "reproducible, non-overlapping subset of each suite that unions back to the full set. "
             "Partitions the full ordered sample set first; --eval-limit then draws up to its "
             "count from this shard (so N shards run up to N x --eval-limit items). Sets "
             "GBENCH_SHARD (an exported GBENCH_SHARD is used when the flag is absent; a malformed "
             "value is an error). Every result records shard, shard_index and shard_count, its "
             "traces a global_idx and a shard-independent sample_key, and a partial shard is not "
             "leaderboard-comparable; an empty shard reports accuracy null (empty_shard). A "
             "suite that cannot shard runs its full set on every node and records "
             "shard_applied=false, and so do the other pillars the run selects (performance, "
             "quality, Golden Set; listed in metadata.json shard_not_applied). Merge the nodes' "
             "results with --merge (a suite's own "
             "merge_shards, else a re-count of its traces; a suite that has neither is refused; "
             "see docs/distributed-gbench.md).",
    )
    eval_group.add_argument(
        "--eval-concurrency",
        type=int,
        default=None,
        help="Concurrent requests per eval suite (the HTTP concurrency of suites that send "
             "their own requests). Default: --batch-sizes[0] (1 unless --batch-sizes is given). "
             "Containerized suites use --sandboxes when it is set. Recorded as "
             "config.eval_concurrency.",
    )
    eval_group.add_argument(
        "--sandboxes",
        type=int,
        default=None,
        help="Concurrency level for containerized/sandboxed evaluations (e.g. terminal_bench, copilot_bench_swe, the swe_bench family, multipl_e, bigcodebench). Defaults to --batch-sizes. Suites whose harness runs strictly one task at a time (e.g. ui_control_osworld, a single-VM loop) ignore it.",
    )
    eval_group.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature for the model under test in the benchmarks. Default, for "
             "EVERY suite: 1.0 with --thinking, 0.0 without it. A benchmark's official or "
             "upstream temperature is never applied, and a temperature is always sent "
             "(harness-owned sampling such as aider_polyglot's and the agentic suites' "
             "included; other official params such as top_p are kept; a suite with official "
             "sampling still records the official values). This flag overrides that rule in "
             "both modes (a suite whose harness cannot take the value records that it was not "
             "applied). Measured 2026-08-17: at no-think, 1.0 halves degenerate repetition "
             "versus greedy (78%%->38%% on looping prompts) - so 0.0 no-think loops more, which "
             "the looping metrics surface. A single eval can be overridden without touching the "
             "rest of the run via GBENCH_<EVAL>_TEMPERATURE, which takes precedence over this "
             "flag.",
    )
    eval_group.add_argument(
        "--attempt-count",
        "--attempts",
        dest="attempt_count",
        type=int,
        default=1,
        help="Generate this many independent attempts per sample and report avg@k, pass@k "
             "and pass^k (default: 1). Most published numbers for small benchmarks are "
             "not single-sample - AIME is avg@4-64, GPQA-Diamond avg@10, ARC-AGI pass@2 "
             "by rule. At --temperature 0 the attempts differ only through server "
             "nondeterminism, which measures reproducibility rather than the benchmark. "
             "Multiplies generation cost by k. Batch-scored suites "
             "that have not opted in run 1 attempt and say so on the result.",
    )
    eval_group.add_argument(
        "--eval-plugins-dir",
        "--eval-plugins-path",
        dest="eval_plugins_dir",
        nargs="+",
        default=None,
        help="Directory or Python file path(s) containing custom/private eval suite plugins (*.py)",
    )
    eval_group.add_argument(
        "--eval-custom-jsonl",
        type=str,
        default=None,
        help="Path to custom JSONL evaluation dataset file for zero-code adhoc evaluation",
    )
    eval_group.add_argument(
        "--evals-only",
        action="store_true",
        help="Run only specific evaluation benchmarks (skip serving, throughput, stress, quality, golden)",
    )
    eval_group.add_argument(
        "--preflight",
        action="store_true",
        help="Check what every selected suite needs before anything is generated, then exit: "
             "each suite's own prerequisite check (images, checkouts, datasets, credentials; a "
             "suite without one: the local image it names), the Gemini key for the suites that "
             "use the Gemini cascade, free disk under --results-dir "
             "(GBENCH_PREFLIGHT_MIN_DISK_GB, default 100), the Docker daemon, and the endpoint's "
             "/v1/models from this host and from a container on Docker's bridge network. All "
             "checks run at once; no model request, no image build. Prints a PASS/FAIL table "
             "with each failure's reason and exits 2 on any FAIL (docs/smoke.md).",
    )
    eval_group.add_argument(
        "--smoke",
        action="store_true",
        help="Check that every selected suite works end to end, without measuring anything: "
             "--preflight, then each suite whose checks passed runs ONE sample (--eval-limit 1) "
             "as a --parallel-suites pool (default 12), its agent episodes ending after the "
             "first action (GBENCH_SMOKE=1), its endpoint behind a recording tap. A suite PASSes "
             "when a model reply came back, every model request carried the run's temperature "
             "rule, --thinking switch and at most --max-output-tokens, an agent's follow-up "
             "request arrived (its action ran) or the sample was scored, and a result was "
             "written. --thinking, --max-output-tokens and the temperature are the run's own. "
             "A model request silent for GBENCH_SMOKE_HANG_S (300) while the server generates "
             "nothing FAILs its suite as a hang, and the whole run stops at "
             "GBENCH_SMOKE_DEADLINE_S (3900). "
             "Needs --remote-endpoint. Writes <--results-dir>/smoke-<timestamp>/ with a "
             "PASS/FAIL table and no scores; exits 2 on any FAIL (docs/smoke.md).",
    )
    eval_group.add_argument(
        "--parallel-suites",
        type=int,
        default=None,
        metavar="N",
        help="Run the selected suites as a pool of up to N gbench processes at once, one per "
             "suite (default 1: the serial sweep). Longest suites first (earlier wall times), "
             "never two suites of one family at once (shared images, files or ports; the SWE "
             "family and the long-context suites 2 at a time), the Gemini search rate split "
             "between the search suites, a new suite started only while the server has fewer "
             "than GBENCH_POOL_MAX_INFLIGHT (default 48) requests in flight, a WARNING when a "
             "suite writes nothing for GBENCH_POOL_STALL_S (default 600 s). Every suite's result "
             "is the same eval_*.json a serial run writes, with the concurrency it ran under "
             "(`pool`); one summary.json and eval_summary.csv cover them all. Needs "
             "--remote-endpoint. Writes <--results-dir>/pool-<timestamp>/ (docs/smoke.md).",
    )
    eval_group.add_argument(
        "--list-plugins",
        nargs="?",
        const=".",
        default=None,
        metavar="PATH",
        help="List all evaluation plugins discovered at PATH (defaults to current directory) and exit",
    )
    eval_group.add_argument(
        "--eval-provenance",
        action="store_true",
        help="Print the canonical-sync provenance table (when each eval was last reconciled against "
             "its upstream, and against what version) and exit. Suites shown as 'unverified' have no "
             "reconcile record yet.",
    )
    eval_group.add_argument(
        "--calibrate",
        action="store_true",
        help="Inspect local hardware (GPUs, their occupancy and VRAM, CPU, RAM, Docker and its "
             "network address pools) and print recommended --num-gpus (the idle GPUs, and the "
             "tensor-parallel size each --models model needs) / --sandboxes / --eval-concurrency, "
             "then exit. A requested --sandboxes that would over-subscribe the machine (CPU/RAM, "
             "or the free Docker networks when terminal_bench, swe_bench_pro or swe_lancer is "
             "selected) is refused before a real run (override with "
             "GBENCH_SKIP_CALIBRATION_GUARD=1).",
    )
    eval_group.add_argument(
        "--list",
        dest="list_what",
        nargs="?",
        const="evals",
        default=None,
        choices=["evals", "pillars", "presets", "golden", "models", "plugins", "all"],
        help="List what gbench can run (evals, pillars, presets, golden, models, plugins, all) and exit",
    )

    # Configuration options
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--preset",
        choices=["quick", "default"],
        default="default",
        help="Configuration preset (default: default)",
    )
    config_group.add_argument(
        "--campaign",
        nargs="+",
        choices=["chat-like", "agentic", "prefill-heavy", "decode-heavy", "mixed", "long-decode"],
        help="Specific performance campaign scenario(s) to run",
    )
    config_group.add_argument(
        "--dataset",
        choices=["random", "custom", "hf"],
        help="Override the serving/throughput data source (random, custom, or hf). ShareGPT is not "
             "selectable here - run it via `--campaign chat-like`, which pairs it with the correct "
             "SLOs and context.",
    )
    config_group.add_argument(
        "--dataset-path",
        type=str,
        help="Path or HF ID for custom dataset",
    )
    config_group.add_argument(
        "--num-iterations",
        type=int,
        help="Number of iterations per config (default: 3)",
    )
    config_group.add_argument(
        "--warmup-iterations",
        type=int,
        help="Number of warmup iterations (default: 1)",
    )
    config_group.add_argument(
        "--max-cv-percent",
        type=float,
        help="Max acceptable CV%% for validation (default: 5.0)",
    )
    config_group.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        help="Custom batch sizes to test",
    )
    config_group.add_argument(
        "--input-lengths",
        nargs="+",
        type=int,
        help="Custom input lengths for throughput tests",
    )
    config_group.add_argument(
        "--output-lengths",
        nargs="+",
        type=int,
        help="Custom output lengths for throughput tests",
    )
    config_group.add_argument(
        "--num-prompts",
        type=int,
        help="(advanced) Fallback prompt count. The SERVING pillar auto-sizes its own sample count "
             "per config, so this does not control it (and is rejected with --serving-only); tune "
             "serving via --batch-sizes / --output-lengths or --personal. Throughput uses its own "
             "fixed count.",
    )
    config_group.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (validates against available GPUs). Required "
             "for any local run, which launches a vLLM server (performance, evals, quality, "
             "Golden Set), so tensor-parallel matches the model size; omit only with "
             "--remote-endpoint. --dry-run validates it as the real run would.",
    )
    config_group.add_argument(
        "--tensor-parallel",
        type=int,
        help="Tensor parallel size: 1, 2, 4, 8 or 16, at most --num-gpus (default: --num-gpus). "
             "When smaller, --num-gpus must be a multiple of it and the server runs "
             "--num-gpus / --tensor-parallel data-parallel replicas (--data-parallel-size).",
    )
    config_group.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU indices or UUIDs a local server runs on (as many as "
             "--num-gpus). Each must run no other process. Default: the first idle GPUs (no "
             "compute process, at least 80%% of memory free) among those CUDA_VISIBLE_DEVICES "
             "exposes; too few idle GPUs is an error. The server gets CUDA_VISIBLE_DEVICES set "
             "to the chosen UUIDs, and the run records them.",
    )
    config_group.add_argument(
        "--tool-call-parser",
        type=str,
        default=None,
        metavar="NAME",
        help="The served model's vLLM tool-call parser, for the eval, quality and Golden Set "
             "server gbench launches: adds `--enable-auto-tool-choice --tool-call-parser NAME`. "
             "Required for a local run of a suite that sends native tools (see --list evals; e.g. "
             "bfcl, tau2, the agentic SWE suites), which otherwise gets HTTP 400 on every "
             "request. Recorded in the scaffold. Not used with --remote-endpoint (configure that "
             "server yourself).",
    )
    config_group.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        metavar="NAME",
        help="The served model's vLLM reasoning parser, for the eval, quality and Golden Set "
             "server gbench launches: adds `--reasoning-parser NAME`, so a --thinking reply "
             "separates its reasoning from its answer. Required for a local --thinking run of "
             "the evals or the quality pillar; `none` states that the model has none (nothing is "
             "passed to vLLM). Recorded in the scaffold. Not used with --remote-endpoint.",
    )
    config_group.add_argument(
        "--vllm-serve-args",
        type=str,
        default=None,
        metavar="'ARGS'",
        help="Extra `vllm serve` arguments for every server gbench launches (one quoted string, "
             "shell-split and appended last; write it with '=' since it starts with '--', e.g. "
             "--vllm-serve-args='--trust-remote-code --chat-template t.jinja'). "
             f"Env {VLLM_SERVE_ARGS_ENV} when the flag is absent. Arguments gbench sets itself "
             "(--port, --host, --max-model-len, --tensor-parallel-size, --data-parallel-size, "
             "--gpu-memory-utilization, --max-logprobs, the parser flags, the scheduler and "
             "prefix-caching flags, --served-model-name, --api-key, --hf-token, --config) are "
             "refused, naming the gbench flag to use. Recorded in the scaffold.",
    )
    config_group.add_argument(
        "--max-model-len",
        type=int,
        help="Context window of the vLLM server gbench launches. Performance "
             "runs default to a uniform 4096, or to the campaign's size with "
             "--campaign (the larger KV cache lowers max concurrency). Eval and "
             "quality runs default to the model's native window, whatever the "
             "campaign; set this to cap it (e.g. when its KV cache does not fit), "
             "or pass -1 to let vLLM fit the window to GPU memory. A positive value "
             "must exceed --max-output-tokens.",
    )
    config_group.add_argument(
        "--gemmaclaw-commit",
        type=str,
        default=DEFAULT_GEMMACLAW_COMMIT,
        help=(
            "Target gemmaclaw git commit, branch or tag (default: the pinned "
            f"release {DEFAULT_GEMMACLAW_COMMIT[:7]}). A branch or tag is "
            "resolved to a commit sha before the run, so the quality "
            "scaffold_id moves when the scorer does. Pass 'main' to score "
            "against the current development tip instead"
        ),
    )
    config_group.add_argument(
        "--gemmaclaw-path",
        type=str,
        help="Path to local gemmaclaw repository checkout (optional)",
    )
    config_group.add_argument(
        "--remote-endpoint",
        type=str,
        help="Remote OpenAI-compatible endpoint to benchmark instead of starting a local vLLM "
             "server (e.g. https://host:8000/v1). Normalized once to end in /v1 (http://h:1 "
             "becomes http://h:1/v1). Its key: VLLM_API_KEY (or OPENAI_API_KEY).",
    )
    config_group.add_argument(
        "--tokenizer",
        type=str,
        help="HuggingFace model ID or local directory path for tokenization when benchmarking custom remote endpoints or Ollama tags. "
             "On a local run whose --vllm-serve-args give the server a --tokenizer, that one is used when this flag is not given (and this flag, when given, must name the same one)",
    )
    config_group.add_argument(
        "--tokenizer-trust-remote-code",
        action="store_true",
        help="Load the model's tokenizer with trust_remote_code=True in gbench's own tokenizer loads (the Golden Set chat-template leak check, prompt counting, the performance pillars' tokenizers), e.g. for a --remote-endpoint whose tokenizer ships custom code. "
             "On a local run, --trust-remote-code in --vllm-serve-args already sets it",
    )

    # Output options
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results"),
        help="Directory for benchmark results (default: ./results). Each run writes its own "
             "<timestamp> directory under it. A directory that already is a run directory "
             "(metadata.json or an eval_*.json in it) is continued in place, by one run at a "
             "time: a run that finds another run still writing there (swarm nodes given one "
             "shared directory) writes its own <timestamp> directory inside it instead.",
    )
    output_group.add_argument(
        "--tags",
        nargs="+",
        help="List of arbitrary key:value tags to assign to this run (e.g., family:gemma-1b stage:prod)",
    )
    output_group.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Reuse a previous result for a suite instead of running it. OFF by default: a "
             "reused result was measured by whatever the code was then, so opt in only when "
             "resuming an interrupted run against unchanged code. A result under --results-dir "
             "(newest by its own finished_at stamp) is reused only if it holds all of: the same "
             "canonical_sync stamp; the same run_shape (temperature, --max-output-tokens, "
             "--eval-thinking, --eval-limit, shard, --eval-n-shot, soft tokens, --attempt-count, "
             "routed categories, served id and window, judge and search cascades, serve "
             "settings, the suite's run-shaping env knobs, the custom JSONL's sha256); the same "
             "evals scaffold id; status success or completed_with_errors. A result that cannot "
             "be read, or has no run_shape (an older gbench), is never reused and the suite "
             "re-runs.",
    )
    output_group.add_argument(
        "--merge",
        nargs="+",
        type=Path,
        default=None,
        metavar="DIR",
        help="Merge the eval results of a sharded run: each DIR is (or contains) one node's "
             "results directory. The shards must be 1..N of one run (the same N, --eval-limit, "
             "gbench commit and evals scaffold), each with status success or "
             "completed_with_errors; the merged result per suite is written under "
             "a new --results-dir/merged-<timestamp>/ of its own. Suites with a corpus or macro "
             "headline merge through their own merge_shards; others are re-counted from the "
             "union of their traces (an item found in two shards is refused). With --dry-run "
             "the shards are checked and merged and nothing is written. Runs nothing else, then "
             "exits (2 when the shards are incomplete or inconsistent).",
    )
    output_group.add_argument(
        "--no-skip-existing",
        action="store_false",
        dest="skip_existing",
        help="Explicitly re-run every benchmark (this is the default).",
    )

    # Execution options
    exec_group = parser.add_argument_group("Execution")
    exec_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running: the planned conditions, and per "
             "model the vllm serve command of each server a local run launches, its port, GPUs "
             "and base URL. Validates --num-gpus / --tensor-parallel / --gpus and the sandbox "
             "concurrency as the real run would (an invalid value exits non-zero).",
    )
    exec_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    exec_group.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (default: 0.9)",
    )
    exec_group.add_argument(
        "--mm-probe-timeout",
        type=int,
        default=None,
        help="Seconds to wait for the remote multimodal capability probe "
             "(default 10; --personal defaults to 120 for slow CPU vision). Raise "
             "it if a capable-but-slow endpoint is wrongly detected as text-only.",
    )
    exec_group.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Per-request serving timeout in seconds (default 600). Applied to "
             "both remote paths: text (urllib) uses it as a per-read STALL timeout "
             "(hung prefill), MM (aiohttp) as a TOTAL wall-clock cap. Raise it for "
             "large models / long outputs on slow devices.",
    )
    exec_group.add_argument(
        "--min-free-gb",
        type=float,
        default=None,
        help="Explicit minimum free VRAM per GPU (GB). Overrides the model-aware "
             "weights-fit advisory and re-enables a hard free-memory pre-check. "
             "Use to unblock/limit on non-datacenter GPUs (default: model-aware, "
             "advisory-only).",
    )

    # Staging utilities
    staging_group = parser.add_argument_group("Staging Utilities")
    staging_group.add_argument(
        "--stage-to-gcs",
        type=str,
        help="Stage models locally (downloading from HF if needed) and upload to the specified GCS destination (e.g. gs://my-bucket/path/). Benchmarks will not be run.",
    )

    return parser


def apply_campaign_to_config(config, campaign: str, args=None) -> None:
    """Apply specific campaign scenario parameters to config. The lengths are set from the
    run's own (pre-campaign) values, so a campaign never inherits the previous one's."""
    if not hasattr(config, "_pre_campaign_lengths"):
        config._pre_campaign_lengths = (list(config.input_lengths or []),
                                        list(config.output_lengths or []))
    # (--campaign and explicit --input/--output-lengths are mutually exclusive)
    config.input_lengths, config.output_lengths = (list(config._pre_campaign_lengths[0]),
                                                   list(config._pre_campaign_lengths[1]))
    config.campaign = campaign
    if campaign == "chat-like":
        config.dataset = "sharegpt"
        # (serving sample count is now auto-sized by output length in
        # get_serving_configs; throughput uses num_prompts_throughput.)
    elif campaign == "agentic":
        config.dataset = "random"
        config.input_lengths = [8000]
        config.output_lengths = [400]
    elif campaign == "prefill-heavy":
        config.dataset = "random"
        config.input_lengths = [8192]
        config.output_lengths = [128]
    elif campaign == "decode-heavy":
        config.dataset = "random"
        config.input_lengths = [128]
        config.output_lengths = [2048]
    elif campaign == "mixed":
        config.dataset = "random"
        config.input_lengths = [4096]
        config.output_lengths = [1024]
    elif campaign == "long-decode":
        config.dataset = "random"
        config.input_lengths = [8192]
        config.output_lengths = [8192]
    if args and getattr(args, "dataset", None):
        config.dataset = args.dataset
    if args and getattr(args, "dataset_path", None):
        config.dataset_path = args.dataset_path
    # Auto-size the served context to this campaign's shape unless the user
    # pinned --max-model-len. RandomDataset (range_ratio=0.5) samples up to
    # 1.5x(in+out), so the fixed 4096 default would truncate/abort the four
    # long-context campaigns; campaign_ctx() clears that tail with slack.
    # Gated on the explicit override so it stays correct at the per-campaign
    # loop call sites (where args.max_model_len was already applied) too.
    if not (args and (getattr(args, "max_model_len", None) or 0) > 0):
        config.max_model_len = campaign_ctx(
            campaign, config.input_lengths, config.output_lengths
        )


def _split_golden_tasks(raw: Optional[list[str]]) -> Optional[list[str]]:
    """Normalise --golden-tasks into a flat list of task ids.

    argparse nargs="+" only splits on spaces, so a comma separated list
    arrives as a single token that matches nothing. Accept either form,
    and both mixed together.

    Args:
        raw: Values as argparse produced them, or None if the flag was
            not passed.

    Returns:
        The flattened task ids, or None when the flag was not passed. An
        explicitly empty selection stays empty rather than becoming None,
        so it is reported as "matched nothing" instead of silently
        running the whole dataset.
    """
    if raw is None:
        return None
    return [part.strip() for item in raw for part in item.split(",")
            if part.strip()]


def get_config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    """Create BenchmarkConfig from CLI arguments."""
    # Start with preset
    if args.preset == "quick":
        config = QUICK_CONFIG
    else:
        config = DEFAULT_CONFIG

    # Override with custom iteration values
    if args.num_iterations:
        config.num_iterations = args.num_iterations
    if args.warmup_iterations is not None:
        config.warmup_iterations = args.warmup_iterations
    if args.max_cv_percent:
        config.min_acceptable_cv_percent = args.max_cv_percent

    # Override with custom configuration values
    if hasattr(args, "campaign") and args.campaign:
        # Concurrency defaults to a single stream (batch size 1); pass
        # --batch-sizes to sweep concurrency. Stress ignores batch_sizes anyway
        # (it runs its own arrival-rate sweep).
        campaigns = args.campaign if isinstance(args.campaign, list) else [args.campaign]
        config.__dict__.pop("_pre_campaign_lengths", None)   # a fresh snapshot per run
        apply_campaign_to_config(config, campaigns[0], args)

    # Record whether the user pinned a workload geometry (campaign or explicit
    # lengths); the stress test uses this to pick a real shape vs its 128/128
    # neutral default.
    config.workload_shape_explicit = bool(
        getattr(args, "campaign", None)
        or args.input_lengths
        or args.output_lengths
    )

    if args.batch_sizes:
        config.batch_sizes = args.batch_sizes
    if args.input_lengths:
        config.input_lengths = args.input_lengths
    if args.output_lengths:
        config.output_lengths = args.output_lengths
    if args.num_prompts:
        config.num_prompts = args.num_prompts
    if getattr(args, "dataset", None):
        config.dataset = args.dataset
    if getattr(args, "dataset_path", None):
        config.dataset_path = args.dataset_path
    # Set GPU configuration. None (unset) leaves the dataclass default (1) so
    # exempt modes (remote/quality/golden/dry-run) keep a valid TP + scaffold
    # fingerprint; local performance runs are required to pass it (guarded below).
    if args.num_gpus is not None:
        config.num_gpus = args.num_gpus
    # --num-gpus N = --tensor-parallel M x N/M data-parallel replicas (parallel_layout; main()
    # refuses an invalid combination before this runs).
    try:
        config.tensor_parallel_size, config.data_parallel_size = parallel_layout(
            config.num_gpus, getattr(args, "tensor_parallel", None))
    except ValueError:
        config.tensor_parallel_size = getattr(args, "tensor_parallel", None) or config.num_gpus
        config.data_parallel_size = 1
    config.gpus = ([g.strip() for g in str(args.gpus).split(",") if g.strip()]
                   if getattr(args, "gpus", None) else None)
    # The model's serve settings (parsers, extra vllm serve arguments), validated by main().
    config.tool_call_parser = getattr(args, "tool_call_parser", None) or None
    config.reasoning_parser = getattr(args, "reasoning_parser", None) or None
    raw_serve = getattr(args, "vllm_serve_args", None)
    if raw_serve is None:
        raw_serve = os.environ.get(VLLM_SERVE_ARGS_ENV)
    try:
        config.vllm_serve_args = parse_vllm_serve_args(raw_serve)
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    # -1 (let vLLM fit the window to GPU memory) applies only to eval/quality servers, which
    # read max_model_len_explicit; perf servers keep their uniform or campaign window.
    if (getattr(args, "max_model_len", None) or 0) > 0:
        config.max_model_len = args.max_model_len
    config.max_model_len_explicit = getattr(args, "max_model_len", None)

    # Stress-test knobs (None => keep config defaults).
    if getattr(args, "stress_reps", None) is not None:
        config.stress_reps = args.stress_reps
    if getattr(args, "stress_client_procs", None) is not None:
        config.stress_client_procs = args.stress_client_procs
    if getattr(args, "stress_max_qps", None) is not None:
        config.stress_max_qps = args.stress_max_qps
    if getattr(args, "stress_max_prompts", None) is not None:
        config.stress_max_prompts = args.stress_max_prompts
    if getattr(args, "stress_min_samples", None) is not None:
        config.stress_min_samples = args.stress_min_samples
    if getattr(args, "stress_certify", None) is not None:
        config.stress_certify = args.stress_certify
    if getattr(args, "stress_certify_budget", None) is not None:
        if args.stress_certify_budget < 0:
            raise SystemExit("--stress-certify-budget must be >= 0 minutes (0 = do not certify)")
        config.stress_certify_budget = args.stress_certify_budget

    # Set output options
    config.results_dir = args.results_dir
    config.skip_existing = args.skip_existing
    config.dry_run = args.dry_run
    config.gpu_memory_utilization = args.gpu_memory_utilization
    if abs(float(args.gpu_memory_utilization) - 0.90) > 1e-9:
        logger.warning(
            f"--gpu-memory-utilization={args.gpu_memory_utilization} differs from the "
            f"0.90 default; KV-cache capacity (and thus stress/throughput numbers) is "
            f"NOT comparable to runs at 0.90. It is recorded in the run fingerprint.")

    # Quality benchmark parameters
    config.gemmaclaw_commit = args.gemmaclaw_commit
    config.gemmaclaw_path = args.gemmaclaw_path
    # One form for every pillar: no trailing '/', ending in /v1.
    config.remote_endpoint = normalize_endpoint(args.remote_endpoint)
    config.tokenizer = getattr(args, "tokenizer", None)
    config.tokenizer_source = "--tokenizer" if config.tokenizer else None
    if not config.tokenizer:
        # The server's own tokenizer (--vllm-serve-args --tokenizer): what every pillar counts
        # and checks with, so gbench and the server agree.
        _serve_tok = serve_arg_value(config.vllm_serve_args, "--tokenizer")
        if _serve_tok:
            config.tokenizer, config.tokenizer_source = _serve_tok, "--vllm-serve-args --tokenizer"
    config.tokenizer_trust_remote_code = bool(getattr(args, "tokenizer_trust_remote_code", False))
    config.selected_scenarios = args.scenarios
    config.tags = args.tags

    # Evaluation parameters
    config.evals = getattr(args, "evals", None)
    if not config.evals:
        if getattr(args, "eval_custom_jsonl", None):
            # A bare --eval-custom-jsonl (no explicit --evals) dispatches the custom_jsonl
            # suite; do NOT let --evals-only expand it to the full 'all' set.
            config.evals = ["custom_jsonl"]
        elif getattr(args, "evals_only", False):
            config.evals = ["plugins"] if getattr(args, "eval_plugins_dir", None) else ["all"]
    config.eval_thinking = getattr(args, "eval_thinking", False)
    config.eval_max_output_tokens = getattr(args, "max_output_tokens", None)
    # None = the model's own processor setting (nothing is overridden).
    config.eval_max_soft_tokens = getattr(args, "eval_max_soft_tokens", None)
    config.eval_n_shot = getattr(args, "eval_n_shot", 0)
    config.suite_timeout = getattr(args, "suite_timeout", None)
    config.eval_categories = getattr(args, "eval_categories", None)
    config.eval_limit = getattr(args, "eval_limit", None)
    # The run's shard: --shard, else an exported GBENCH_SHARD (an operator may set it per swarm
    # node without the flag). Validated here (a malformed value fails fast instead of silently
    # running the whole set) and recorded on the config, so the scaffold pins the value the
    # suites read.
    config.shard = getattr(args, "shard", None) or (os.environ.get("GBENCH_SHARD") or "").strip() \
        or None
    if config.shard:
        from .runners.eval_suites.sampling import parse_shard
        try:
            parse_shard(config.shard)
        except ValueError as e:
            raise SystemExit(f"error: {e}")
        # Transport via env so it reaches both the native path (run_eval_suite reads GBENCH_SHARD)
        # and container evals (forwarded into the orchestrator).
        os.environ["GBENCH_SHARD"] = config.shard
    config.sandboxes = getattr(args, "sandboxes", None)
    # The eval request concurrency: --eval-concurrency, else --batch-sizes[0] (the evals runner
    # reads config.eval_concurrency; containerized suites use --sandboxes when it is set).
    config.eval_concurrency = (getattr(args, "eval_concurrency", None)
                               or (config.batch_sizes[0] if config.batch_sizes else 1))
    config.temperature = getattr(args, "temperature", None)
    config.attempt_count = getattr(args, "attempt_count", 1)
    config.eval_plugins_dir = getattr(args, "eval_plugins_dir", None)
    config.eval_custom_jsonl = getattr(args, "eval_custom_jsonl", None)

    # Golden Set parameters
    config.golden = args.golden or args.golden_only
    config.golden_only = args.golden_only
    # nargs="+" splits on spaces, but a comma separated list is the more
    # natural thing to type and silently matches no tasks. Accept both.
    config.selected_golden_tasks = _split_golden_tasks(args.golden_tasks)
    # --served-model-id (aliases --golden-model-id / --eval-model-id): one value for every pillar;
    # golden_model_id is kept for the Golden Set runner.
    config.served_model_id = getattr(args, "served_model_id", None)
    config.golden_model_id = config.served_model_id

    # Initialize LogManager NOW - after all config fields are set
    # (must be after results_dir is assigned from CLI args)
    config.initialize()

    return config


def _summary_record(result: Dict[str, Any]) -> Dict[str, Any]:
    """A result as summary.json holds it: an eval result without its sample_traces, which stay
    in the suite's own eval_*.json (named under `sample_traces_omitted`), so the summary does not
    repeat every trace (a trace can carry a problem's whole test suite); anything else as is."""
    traces = result.get("sample_traces")
    if result.get("benchmark_type") != "eval" or not isinstance(traces, list):
        return result
    from .runners.evals import eval_result_filename
    out = {k: v for k, v in result.items() if k != "sample_traces"}
    out["sample_traces_omitted"] = {
        "count": len(traces),
        "file": eval_result_filename(str(result.get("eval_name") or ""),
                                     str(result.get("model_short") or ""),
                                     str(result.get("format") or ""), result),
    }
    return out


def _served_model_id_record(config, args, models) -> Dict[str, Any]:
    """metadata.json's served_model_id and where it came from: --served-model-id; the id
    auto-detected at the remote endpoint (no --models: the one model it serves); else None,
    since each pillar resolves the served id per model and each result records its own."""
    if getattr(config, "served_model_id", None):
        return {"served_model_id": config.served_model_id,
                "served_model_id_source": "--served-model-id"}
    if getattr(config, "remote_endpoint", None) and not getattr(args, "models", None) and models:
        return {"served_model_id": models[0].hf_model_id,
                "served_model_id_source": "auto-detected: the only model the endpoint serves"}
    return {"served_model_id": None,
            "served_model_id_source": "resolved per pillar (each result records its "
                                      "served_model_id)"}


class ModelSelectionError(ValueError):
    """--models / --served-model-id cannot name a model the run can serve or reach."""


def _served_model_ids(base_url: str, timeout: float = 5.0) -> List[str]:
    """The ids an OpenAI-compatible endpoint lists at <base_url>/models (the endpoint key sent,
    base.endpoint_auth_headers). Raises ModelSelectionError when the listing cannot be read."""
    import json
    import urllib.request
    from .runners.eval_suites.base import endpoint_auth_headers
    url = f"{normalize_endpoint(base_url)}/models"
    try:
        req = urllib.request.Request(url, headers=endpoint_auth_headers())
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:                                                  # noqa: BLE001
        raise ModelSelectionError(f"could not list the served models at {url} ({e})") from e
    return [str(m.get("id")) for m in (data.get("data") or []) if isinstance(m, dict)
            and m.get("id")]


def _enrich_from_tokenizer(model, tokenizer: Optional[str]) -> None:
    """Borrow real metadata (parameters, MoE, modality) from the --tokenizer HF id for a model
    whose id cannot be introspected (an endpoint tag such as an Ollama name); its identity stays
    its own id."""
    if not tokenizer or model.total_params_b:
        return
    from .core.models import enrich_metadata_from
    if enrich_metadata_from(model, tokenizer):
        logger.info(
            f"Enriched metadata from --tokenizer '{tokenizer}': {model.total_params_b:.1f}B, "
            f"{'MoE' if model.is_moe else 'dense'}, "
            f"{'multimodal' if model.supports_multimodal else 'text'}")


def get_models_from_args(args: argparse.Namespace) -> list:
    """The models to benchmark.

    --models: each value registered (a HuggingFace id or a local checkpoint directory; with
    --remote-endpoint, the endpoint's id, enriched from --tokenizer when it cannot be read).
    Without --models and with --remote-endpoint: the one model the endpoint serves
    (match_served_model), or --served-model-id when it serves several; several served ids and
    no --served-model-id is a ModelSelectionError listing them. Nothing selected: []."""
    models = []
    remote = getattr(args, "remote_endpoint", None)

    if args.models:
        for model_name in args.models:
            model = registry.get(model_name)
            if not model:
                model = registry.register_hf_model(model_name)
            if remote:
                _enrich_from_tokenizer(model, getattr(args, "tokenizer", None))
            models.append(model)
        return models

    if remote:
        served = _served_model_ids(remote)
        served_id, how = match_served_model(served, [], getattr(args, "served_model_id", None))
        if not served_id:
            raise ModelSelectionError(
                f"cannot tell which model to benchmark at {normalize_endpoint(remote)}: {how}")
        model = registry.get(served_id) or registry.register_hf_model(served_id)
        logger.info(f"Auto-detected model '{served_id}' served at {remote} ({how})")
        _enrich_from_tokenizer(model, getattr(args, "tokenizer", None))
        return [model]

    return models


def get_formats_from_args(args: argparse.Namespace) -> list:
    """The formats to test: [REMOTE] with --remote-endpoint, else [HF] (the HuggingFace
    checkpoint, the only format gbench serves; main() refuses --format gguf/both)."""
    if getattr(args, "remote_endpoint", None):
        return [ModelFormat.REMOTE]
    return [ModelFormat.HF]



def _handle_list(args) -> int:
    """`--list <what>`: show what gbench can run, then exit.

    The suites, pillars and presets were previously only discoverable by reading the
    source or scraping --help, which made it hard to know what a sweep would actually
    execute (or which suites would skip).
    """
    what = args.list_what
    show = (lambda k: what in (k, "all"))

    if show("pillars") or show("evals"):
        from .runners.evals import BUILTIN_PILLARS, NAMED_ONLY_SUITES
        from .runners.eval_suites import SUITES
        in_pillars = {e for _, evs in BUILTIN_PILLARS for e in evs}
        if show("pillars"):
            print(f"\nBuilt-in pillars ({len(BUILTIN_PILLARS)}):\n")
            for pillar, evs in BUILTIN_PILLARS:
                named = [e for e in evs if e in NAMED_ONLY_SUITES]
                count = (f"{len(evs)}; {len(evs) - len(named)} in --evals all" if named
                         else f"{len(evs)}")
                print(f"  {pillar}  ({count})")
                print("      " + "  ".join(e + ("*" if e in NAMED_ONLY_SUITES else "")
                                           for e in sorted(evs)))
            if any(e in NAMED_ONLY_SUITES for _, evs in BUILTIN_PILLARS for e in evs):
                print("\n  * runs only when named; not in --evals all")
        if show("evals"):
            from .runners.eval_suites import suite_requires_tool_calling
            from .runners.eval_suites.canonical_sync import canonical_sync_for
            print(f"\nBuilt-in eval suites ({len(SUITES)}):\n")
            print("  columns: suite, provenance (the date it was last reconciled with its "
                  "upstream; pinned = a pinned revision or checksum, date-only otherwise; "
                  "details: --eval-provenance), [tools] = sends native tools (a local run needs "
                  "--tool-call-parser), pillar\n")
            pillar_of = {e: p for p, evs in BUILTIN_PILLARS for e in evs}
            width = max((len(n) for n in SUITES), default=10) + 2
            for name in sorted(SUITES):
                tag = pillar_of.get(name, "(not in --evals all)")
                if name in NAMED_ONLY_SUITES:
                    tag += "  (runs only when named; not in --evals all)"
                cs = canonical_sync_for(name)
                if cs.get("status") == "reconciled":
                    prov = f"{cs.get('synced', '')} {'pinned' if provenance_pinned(cs) else 'date-only'}"
                else:
                    prov = "unverified"
                tools = "[tools]" if suite_requires_tool_calling(name) else ""
                print(f"  {name:<{width}} {prov:<21} {tools:<8} {tag}")
            orphan = sorted((set(SUITES) - in_pillars) | (set(SUITES) & set(NAMED_ONLY_SUITES)))
            if orphan:
                print(f"\n  NB {len(orphan)} suite(s) are registered but NOT part of "
                      f"`--evals all`: {', '.join(orphan)}")

    if show("presets"):
        print("\nPresets (--preset):\n  quick    reduced scenario set\n  default  full scenario set")

    if show("golden"):
        # Golden tasks are JSON files under gbench/golden_dataset/, each with an "id";
        # there is no module-level constant to import.
        import json as _json
        from pathlib import Path as _Path
        dataset_dir = _Path(__file__).parent / "golden_dataset"
        if not dataset_dir.is_dir():
            print(f"\nGolden Set tasks: dataset directory not found at {dataset_dir}")
        else:
            rows = []
            for fp in sorted(dataset_dir.glob("*.json")):
                try:
                    data = _json.loads(fp.read_text(encoding="utf-8"))
                except Exception as e:
                    rows.append((fp.stem, f"<unreadable: {type(e).__name__}>", ""))
                    continue
                rows.append((str(data.get("id", fp.stem)),
                             str(data.get("category", "uncategorized")),
                             str(data.get("description", data.get("name", "")))[:52]))
            print(f"\nGolden Set tasks ({len(rows)}) - use with --golden / --golden-tasks:\n")
            width = max((len(r[0]) for r in rows), default=10) + 2
            cat_w = max((len(r[1]) for r in rows), default=8) + 2
            for tid, cat, desc in rows:
                print(f"  {tid:<{width}} {cat:<{cat_w}} {desc}")

    if show("models"):
        from .core.models import registry
        print("\nModels (--models): gbench has no built-in model list. Each --models id is its own "
              "model: a HuggingFace repo id, a local checkpoint directory, a gs:// path, or an "
              "endpoint's served id. Size, modalities and context come from the model's own "
              "config.json (recorded in metadata.json as model_metadata).")
        registered = registry.list_all()
        if registered:
            print(f"\n  Registered in this process ({len(registered)}):")
            for m in registered:
                print(f"    {m.short_name:<40} {m.hf_model_id}  (metadata: {m.metadata_source})")

    if show("plugins"):
        from .runners.eval_suites.loader import discover_and_register_plugins, CUSTOM_PILLARS
        paths = getattr(args, "eval_plugins_dir", None)
        if not paths:
            env_paths = os.environ.get("GBENCH_EVAL_PLUGINS_PATH", "").strip()
            if env_paths:
                paths = [p.strip() for p in env_paths.split(os.pathsep) if p.strip()]
            elif Path("plugins").is_dir():
                paths = ["plugins"]
            elif Path("gbench/plugins").is_dir():
                paths = ["gbench/plugins"]
        if not paths:
            print("\nPlugins (--list plugins): no plugin directory specified.\n  Pass --eval-plugins-dir <path> or set GBENCH_EVAL_PLUGINS_PATH to discover custom suites.")
        else:
            discovered = discover_and_register_plugins(paths if isinstance(paths, list) else [paths])
            print(f"\nPlugins discovered in {paths} ({len(discovered)}):")
            for name in sorted(discovered):
                print(f"  {name:<52} [{CUSTOM_PILLARS.get(name, 'Custom Plugin')}]")

    print()
    return 0


def provenance_pinned(entry: Dict[str, Any]) -> bool:
    """Whether a canonical_sync entry is version-pinned rather than dated only: the registry's own
    predicate (canonical_sync.entry_pinned), so --list and --eval-provenance say what the gate
    checks."""
    from .runners.eval_suites.canonical_sync import entry_pinned
    return entry_pinned(entry)


def _flag_range_errors(args) -> List[str]:
    """The numeric flags a run cannot use: a count below 1 or a negative temperature."""
    errors = []
    for flag, value in (("--sandboxes", getattr(args, "sandboxes", None)),
                        ("--eval-concurrency", getattr(args, "eval_concurrency", None)),
                        ("--attempt-count", getattr(args, "attempt_count", None)),
                        ("--eval-limit", getattr(args, "eval_limit", None)),
                        ("--max-output-tokens", getattr(args, "max_output_tokens", None)),
                        ("--num-gpus", getattr(args, "num_gpus", None)),
                        ("--eval-max-soft-tokens", getattr(args, "eval_max_soft_tokens", None))):
        if value is not None and value < 1:
            errors.append(f"{flag} must be at least 1, got {value}")
    for value in (getattr(args, "batch_sizes", None) or []):
        if value < 1:
            errors.append(f"--batch-sizes values must be at least 1, got {value}")
            break
    temperature = getattr(args, "temperature", None)
    if temperature is not None and temperature < 0:
        errors.append(f"--temperature must not be negative, got {temperature}")
    return errors


def _expand_selected_suites(evals_selected, plugin_dirs=None) -> List[str]:
    """The suite names a selection runs: 'all' expanded to the built-in suites (plus discovered
    plugins), 'plugins' to the discovered plugins."""
    from .runners.eval_suites import discover_and_register_plugins
    names: List[str] = []
    for e in evals_selected or []:
        if e == "all":
            from .runners.evals import builtin_all_suites
            names += builtin_all_suites()
            names += list(discover_and_register_plugins(plugin_dirs).keys())
        elif e == "plugins":
            names += list(discover_and_register_plugins(plugin_dirs).keys())
        else:
            names.append(e)
    return list(dict.fromkeys(names))


def _golden_tasks_send_tools(selected: Optional[List[str]]) -> bool:
    """True when a selected Golden Set task (all when `selected` is None) sends native tools."""
    import json as _json
    dataset_dir = Path(__file__).parent / "golden_dataset"
    for fp in sorted(dataset_dir.glob("*.json")):
        try:
            data = _json.loads(fp.read_text(encoding="utf-8"))
        except Exception:                                                  # noqa: BLE001
            continue
        tid = str(data.get("id", fp.stem))
        if selected is not None and tid not in selected and fp.name not in selected \
                and fp.stem not in selected:
            continue
        if '"tools"' in _json.dumps(data):
            return True
    return False


def _processor_declares(model_id: str, key: str) -> Optional[bool]:
    """Whether the model's processor configuration (processor_config.json or
    preprocessor_config.json, in a local checkpoint directory or on the Hugging Face Hub) declares
    `key` at any depth: True / False, or None when neither file can be read."""
    import json as _json

    def _has(obj) -> bool:
        if isinstance(obj, dict):
            return key in obj or any(_has(v) for v in obj.values())
        if isinstance(obj, list):
            return any(_has(v) for v in obj)
        return False

    read_any = False
    for name in ("processor_config.json", "preprocessor_config.json"):
        path = None
        if os.path.isdir(model_id):
            candidate = os.path.join(model_id, name)
            path = candidate if os.path.isfile(candidate) else None
        else:
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(model_id, name)
            except Exception:                                              # noqa: BLE001
                path = None
        if not path:
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = _json.load(f)
        except Exception:                                                  # noqa: BLE001
            continue
        read_any = True
        if _has(data):
            return True
    return False if read_any else None


def _termination_handlers():
    """A context manager that turns SIGTERM and SIGHUP into SystemExit while a run executes (a
    SIGHUP the process was started ignoring, as `nohup` sets it, stays ignored), so
    every finally block and atexit hook runs (the local vLLM servers are cleaned up), and puts
    the previous handlers back afterwards. A no-op outside the main thread. The signal is also
    recorded while the run unwinds (utils.process_groups.request_stop): the run writes a failed
    record naming the signal for the suite it cut short, and a suite running in a worker thread
    (--suite-timeout) gets the signal handed over and stops before it starts more work."""
    import contextlib
    from .utils import process_groups

    @contextlib.contextmanager
    def _ctx():
        previous = {}
        process_groups.clear_stop()

        def _handler(signum, _frame):
            process_groups.request_stop(signum)
            raise SystemExit(128 + signum)

        for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
            if sig is None:
                continue
            try:
                # A signal the run was started ignoring stays ignored: `nohup gbench ...` sets
                # SIGHUP to SIG_IGN so a closing terminal does not stop the run, and replacing it
                # stopped such a run on 2026-10-06.
                if sig != signal.SIGTERM and signal.getsignal(sig) == signal.SIG_IGN:
                    continue
                previous[sig] = signal.signal(sig, _handler)
            except (ValueError, OSError):                                  # not the main thread
                pass
        try:
            yield
        finally:
            for sig, old in previous.items():
                try:
                    signal.signal(sig, old)
                except (ValueError, OSError):
                    pass
            # A worker thread the signal left running got it handed over
            # (evals._run_with_timeout); the run-wide record ends with the run.
            process_groups.clear_stop()

    return _ctx()


def _pillar_failure(benchmark_type: str, model, fmt, error: Any,
                    campaign: Optional[str] = None, **extra: Any) -> Dict[str, Any]:
    """A failed record for a pillar that raised or could not run: a harness error (or
    infra_required when the error says so), never a verdict on the model."""
    from .runners.eval_suites.base import redact_text
    msg = redact_text(str(error))
    kind = "infra_required" if "INFRA REQUIRED" in msg.upper() or "infra_required" in msg \
        else "harness"
    rec = {"benchmark_type": benchmark_type, "failed": True, "failure_kind": kind,
           "status": "error", "error": msg[:2000], "model": model.short_name,
           "model_name": model.name, "model_short": model.short_name,
           "format": getattr(fmt, "value", str(fmt))}
    if campaign:
        rec["campaign"] = campaign
    rec.update(extra)
    return rec


def run_exit_code(all_results: List[Dict[str, Any]]) -> int:
    """The run's exit code from its records: EXIT_HARNESS_ERROR (2) when any pillar produced a
    harness or infra failure (a perf pillar, text or multimodal, that failed; a quality, Golden Set
    or eval pillar that raised or could not start its server; an eval suite without a score; a
    Golden Set error), each named in the log; else EXIT_MODEL_FAILURE (1) when the Golden Set has
    failing cases (a model verdict); else EXIT_OK."""
    harness = [r for r in all_results
               if r.get("failed") is True and r.get("benchmark_type") != "eval"]
    eval_code = eval_failures_exit_code(all_results)
    golden_code = golden_exit_code(all_results)
    if harness:
        logger.error(
            "%d pillar configuration(s) produced no result: %s. Exiting %d (harness error). "
            "This is not a verdict on the model.", len(harness),
            "; ".join(sorted({f"{r.get('benchmark_type')} {r.get('model_short', '?')}"
                              + (f" [{r.get('campaign')}]" if r.get("campaign") else "")
                              + f" ({r.get('failure_kind') or 'harness'})" for r in harness})),
            EXIT_HARNESS_ERROR)
    if golden_code == EXIT_HARNESS_ERROR:
        logger.error("Golden Set could not complete (harness error); this is not a verdict on "
                     "the model.")
    elif golden_code == EXIT_MODEL_FAILURE:
        logger.error("Golden Set has failing cases (a model verdict).")
    if harness or eval_code == EXIT_HARNESS_ERROR or golden_code == EXIT_HARNESS_ERROR:
        return EXIT_HARNESS_ERROR
    return golden_code


def _print_dry_run_plan(config, models, formats, serving_runner, *, perf_planned: bool,
                        native_planned: bool, mm_planned) -> None:
    """Per model and format: each server a local run launches (its vllm serve command, port, GPUs
    and base URL), or the remote endpoint every pillar targets."""
    print("\nDRY RUN: servers and endpoints")
    print("-" * 80)
    if config.remote_endpoint:
        print(f"  every pillar targets the remote endpoint {config.remote_endpoint} "
              "(no server is launched)")
        print()
        return
    port, source = resolve_server_port()
    port_s = f"{port} ({source})" if port else f"unavailable: {source}"
    gpus = ", ".join(config.selected_gpus or []) or "(not selected: nvidia-smi unreadable)"
    print(f"  port: {port_s}")
    print(f"  GPUs: {gpus}; tensor-parallel {config.tensor_parallel_size}, data-parallel "
          f"{config.data_parallel_size}")
    build = getattr(serving_runner, "build_server_cmd", None)
    for model in models:
        for fmt in formats:
            contexts = ([("perf", "serving / throughput / stress")] if perf_planned else []) \
                + ([("native", "quality / Golden Set / evals")] if native_planned else [])
            for ctx, label in contexts:
                if callable(build):
                    try:
                        cmd = build(model=model, format=fmt, context=ctx, port=port)
                        shown = " ".join(shlex.quote(str(c)) for c in cmd)
                    except Exception as e:                                 # noqa: BLE001
                        shown = f"(could not build the command: {e})"
                else:
                    shown = (f"vllm serve {model.get_model_path(fmt)} --port {port} ... "
                             + " ".join(shlex.quote(a) for a in serve_args_for(config, ctx)))
                print(f"  {model.short_name} ({fmt.value}) {label}:")
                print(f"    {shown}")
            if native_planned and port:
                print(f"    base_url for quality / Golden Set / evals: http://127.0.0.1:{port}/v1")
            if mm_planned(model):
                print("    multimodal pillars launch their own performance server "
                      "(--limit-mm-per-prompt set per configuration)")
    print()


def main(argv: Optional[list[str]] = None):
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args(argv)

    # Show help if no arguments provided
    import sys
    # The arguments as typed: --parallel-suites / --smoke hand them to each suite's process.
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if (argv is None and len(sys.argv) == 1) or (argv is not None and len(argv) == 0):
        parser.print_help()
        return 0

    # List plugins mode (independent of any benchmark execution)
    if getattr(args, "list_what", None):
        return _handle_list(args)

    if getattr(args, "eval_provenance", False):
        from .runners.eval_suites import SUITES
        from .runners.eval_suites.canonical_sync import canonical_sync_for
        rows = []
        for name in sorted(SUITES):
            cs = canonical_sync_for(name)
            if cs.get("status") == "reconciled":
                ver = cs.get("checksum") or cs.get("revision") or cs.get("dataset") or cs.get("upstream") or ""
                pinned = "pinned" if provenance_pinned(cs) else "date-only"
                rows.append((name, cs.get("synced", ""), pinned, str(ver)[:44]))
            else:
                rows.append((name, "unverified", "", ""))
        w = max((len(r[0]) for r in rows), default=8)
        recon = sum(1 for r in rows if r[1] != "unverified")
        pinned_n = sum(1 for r in rows if r[2] == "pinned")
        print(f"\nCanonical-sync provenance ({recon}/{len(rows)} reconciled, "
              f"{pinned_n} version-pinned, {recon - pinned_n} date-only):\n")
        # 13 wide: a same-day serial (2026-09-28.2) is 12 characters.
        print(f"  {'suite':<{w}}  {'synced':<13}  {'status':<9}  upstream/checksum ref")
        print(f"  {'-'*w}  {'-'*13}  {'-'*9}  {'-'*24}")
        for n, s, st, v in rows:
            print(f"  {n:<{w}}  {s:<13}  {st:<9}  {v}")
        print()
        return 0

    if getattr(args, "merge", None):
        setup_logging(args.verbose)
        from .merge import MergeError, merge_runs
        out_dir = Path(args.results_dir) / f"merged-{time.strftime('%m-%d-%H%M%S')}"
        dry = bool(getattr(args, "dry_run", False))
        try:
            # A new directory of its own (merged-<stamp>, else -2, ...); a dry run writes nothing.
            written = merge_runs(list(args.merge), out_dir, dry_run=dry, unique=True)
        except MergeError as e:
            logger.error(f"merge refused: {e}")
            return EXIT_HARNESS_ERROR
        if written:
            out_dir = written[0].parent
        if dry:
            print(f"\nDRY RUN: the results merge (shards of one run) or join (unsharded runs of "
                  f"different suites); {len(written)} suite result(s) would be written "
                  f"into a new {out_dir} (nothing was written):")
        else:
            print(f"\nMerged {len(written)} suite result(s) into {out_dir} (shards merged; "
                  "unsharded runs joined, with one summary.json and eval_summary.csv):")
        for path in written:
            print(f"  - {path.name}")
        print()
        return 0

    if getattr(args, "calibrate", False):
        # Read-only (no server). calibration_report takes the selected models and suites when it
        # accepts them (models= for the tensor-parallel recommendation, evals= so a --sandboxes
        # value is given only for containerized suites).
        import inspect
        from .utils.calibrate import calibration_report
        params = inspect.signature(calibration_report).parameters
        kwargs: Dict[str, Any] = {}
        if "models" in params and args.models:
            try:
                kwargs["models"] = get_models_from_args(args)
            except Exception as e:                                         # noqa: BLE001
                print(f"(models not resolved for calibration: {e})")
        if "evals" in params and args.evals:
            flat = [s.strip().lower() for item in args.evals for s in str(item).split(",")
                    if s.strip()]
            kwargs["evals"] = _expand_selected_suites(flat, getattr(args, "eval_plugins_dir", None))
        print("\n" + calibration_report(**kwargs) + "\n")
        return 0

    if getattr(args, "list_plugins", None) is not None:
        target_path = args.list_plugins
        from .runners.eval_suites.loader import discover_and_register_plugins, CUSTOM_PILLARS
        discovered = discover_and_register_plugins([target_path])
        if not discovered:
            print(f"No plugins found in '{target_path}'.")
            return 0
        print(f"\nDiscovered {len(discovered)} evaluation plugin(s) in '{target_path}':\n")
        max_name_len = max(len(name) for name in discovered.keys())
        for name, fn in sorted(discovered.items()):
            pillar = CUSTOM_PILLARS.get(name, "Custom Plugin")
            print(f"  - {name:<{max_name_len + 2}} [{pillar}]")
        print()
        return 0

    # Validation constraints
    remote = getattr(args, "remote_endpoint", None)
    for err in _flag_range_errors(args):
        parser.error(err)
    if getattr(args, "format", "hf") != "hf":
        parser.error(f"--format {args.format} is not supported: gbench serves the HuggingFace "
                     "checkpoint (--format hf, the default); GGUF serving was removed.")
    if remote and not getattr(args, "tokenizer", None):
        parser.error(
            "--tokenizer <hf_repo_or_local_path> is required when using --remote-endpoint (e.g. --tokenizer google/gemma-4-E4B-it)."
        )
    if getattr(args, "campaign", None) and (getattr(args, "input_lengths", None) or getattr(args, "output_lengths", None)):
        parser.error("--campaign and --input-lengths/--output-lengths are mutually exclusive.")
    if getattr(args, "num_prompts", None) and getattr(args, "serving_only", False):
        parser.error(
            "--num-prompts cannot be used with --serving-only: the serving pillar auto-sizes its own "
            "sample count per config (a ~5-minute wall-clock target that scales up with --batch-sizes "
            "and down with --output-lengths). Change the count via --batch-sizes / --output-lengths, "
            "or use --personal for a fixed small count.")
    if not getattr(args, "campaign", None) and not (getattr(args, "input_lengths", None) or getattr(args, "output_lengths", None)) and not getattr(args, "preset", None):
        # --stage-to-gcs only copies model artifacts to a bucket. It never
        # runs a benchmark, so it has no workload geometry to describe and
        # belongs with the other modes exempted here.
        if not getattr(args, "evals", None) and not getattr(args, "evals_only", False) and not getattr(args, "golden_only", False) and not getattr(args, "quality_only", False) and not getattr(args, "stress_test", False) and not getattr(args, "stage_to_gcs", None):
            parser.error("Either --preset, --campaign, or --input-lengths/--output-lengths must be provided.")
    # A shard spec from the flag or the environment: malformed is an error, before anything runs.
    _shard_spec = getattr(args, "shard", None) or (os.environ.get("GBENCH_SHARD") or "").strip()
    if _shard_spec:
        from .runners.eval_suites.sampling import parse_shard
        try:
            parse_shard(_shard_spec)
        except ValueError as e:
            parser.error(f"{e}{'' if getattr(args, 'shard', None) else ' (from GBENCH_SHARD)'}")
    # Serve settings: the extras must not set what gbench owns; none applies to a remote endpoint.
    _raw_serve = getattr(args, "vllm_serve_args", None)
    if _raw_serve is None:
        _raw_serve = os.environ.get(VLLM_SERVE_ARGS_ENV)
    try:
        _serve_tokens = parse_vllm_serve_args(_raw_serve)
    except ValueError as e:
        parser.error(str(e))
    _serve_tok = serve_arg_value(_serve_tokens, "--tokenizer")
    if _serve_tok and getattr(args, "tokenizer", None) and args.tokenizer != _serve_tok:
        parser.error(f"--tokenizer {args.tokenizer!r} differs from the --tokenizer {_serve_tok!r} "
                     "in --vllm-serve-args (the server's): gbench counts prompts and checks the "
                     "chat template with its --tokenizer, so both must name the same one (or "
                     "omit --tokenizer to use the server's)")
    if remote:
        _local_only = [flag for flag, value in (
            ("--tool-call-parser", getattr(args, "tool_call_parser", None)),
            ("--reasoning-parser", getattr(args, "reasoning_parser", None)),
            ("--vllm-serve-args", getattr(args, "vllm_serve_args", None)),
            ("--gpus", getattr(args, "gpus", None))) if value]
        if _local_only:
            parser.error(f"{', '.join(_local_only)} configure(s) the server gbench launches; with "
                         "--remote-endpoint gbench launches none (configure that server itself).")
    elif getattr(args, "num_gpus", None) is not None or getattr(args, "tensor_parallel", None):
        try:
            parallel_layout(getattr(args, "num_gpus", None) or 1,
                            getattr(args, "tensor_parallel", None))
        except ValueError as e:
            parser.error(str(e))
        if getattr(args, "gpus", None):
            _gpus = [g for g in str(args.gpus).split(",") if g.strip()]
            if len(_gpus) != (getattr(args, "num_gpus", None) or 1):
                parser.error(f"--gpus names {len(_gpus)} GPU(s) but --num-gpus is "
                             f"{getattr(args, 'num_gpus', None) or 1}")

    # Strict Eval Argument Compatibility Validation
    from .runners.eval_suites import discover_and_register_plugins, SUITES
    discover_and_register_plugins(getattr(args, "eval_plugins_dir", None))

    raw_evals = getattr(args, "evals", None) or []
    flat_evals = []
    for item in raw_evals:
        for sub in str(item).split(","):
            # Suite names are registered in lower case (plugins included): --evals GPQA is gpqa.
            sub_clean = sub.strip().lower()
            if sub_clean:
                flat_evals.append(sub_clean)
    # A renamed suite's old name runs the new suite, with a one-line deprecation warning; a
    # deregistered suite planned in docs/evals_roadmap.md is an error naming its roadmap entry.
    from .runners.eval_suites import resolve_suite_name
    try:
        flat_evals = list(dict.fromkeys(resolve_suite_name(e) for e in flat_evals))
    except ValueError as e:
        parser.error(str(e))
    args.evals = flat_evals
    evals_selected = set(flat_evals)
    if evals_selected:
        for e in evals_selected:
            if e not in {"all", "plugins"} and e not in SUITES:
                parser.error(f"Unknown eval suite '{e}'. Available: {', '.join(sorted(SUITES.keys()))}")

    if getattr(args, "eval_categories", None):
        # Routed per suite (evals.route_eval_categories): each category-aware suite gets only
        # the names it knows, so every name must be known to at least one selected suite.
        from .runners.evals import eval_category_names
        by_suite = eval_category_names()
        if not evals_selected or any(e not in {"all", *by_suite} for e in evals_selected):
            parser.error("--eval-categories is only supported when running 'bfcl', "
                         "'bfcl_v4_agentic' and/or 'swe_bench_live' (or 'all'). Cannot combine "
                         "with other evals.")
        targets = [e for e in by_suite if "all" in evals_selected or e in evals_selected]
        known = set().union(*(by_suite[e] for e in targets))
        names = [c.strip() for c in args.eval_categories.split(",") if c.strip()]
        unknown = [c for c in names if c not in known]
        if unknown or not names:
            parser.error(f"Unknown --eval-categories name(s) {unknown or [args.eval_categories]} "
                         f"for {', '.join(targets)}. Known: {', '.join(sorted(known))}.")
    if remote and getattr(args, "throughput_only", False):
        # The throughput pillar drives a local vLLM engine (offline peak batching); against a
        # remote endpoint it has nothing to run, and --throughput-only used to exit 0 with "No
        # benchmark results to summarize" (2026-10-14 smoke run).
        parser.error("--throughput-only cannot be used with --remote-endpoint: the throughput "
                     "pillar runs a local vLLM engine (offline peak batching, it needs a local "
                     "GPU). Against a remote endpoint use --serving-only or --stress-test.")
    if getattr(args, "eval_max_soft_tokens", None) is not None:
        if remote:
            parser.error("--eval-max-soft-tokens cannot be used with --remote-endpoint (soft tokens are configured server-side).")
        if not evals_selected or any(e not in {"mmmu_pro", "screenspot", "screenspot_pointing", "textvqa", "infographicvqa", "bundled_detection", "all"} for e in evals_selected):
            parser.error("--eval-max-soft-tokens is only supported by vision evals ('mmmu_pro', 'screenspot', 'screenspot_pointing', 'textvqa', 'infographicvqa', 'bundled_detection').")
    if getattr(args, "eval_n_shot", None) is not None:
        if not evals_selected or any(e not in {"mmlu", "mmlu_pro", "gsm8k", "all"} for e in evals_selected):
            parser.error("--eval-n-shot is only supported by 'mmlu', 'mmlu_pro' and 'gsm8k'.")

    # --max-output-tokens is required for eval runs: it bounds generation and materially
    # affects scores (truncation reads as a failure), so it must be explicit rather than
    # silently taking a per-suite default - otherwise two runs are quietly incomparable.
    if (evals_selected or getattr(args, "evals_only", False)) \
            and getattr(args, "max_output_tokens", None) is None:
        parser.error(
            "--max-output-tokens is required for eval runs: it bounds generation and "
            "materially affects scores (truncation reads as a failure), so it must be "
            "explicit for runs to be comparable. Set it, e.g. --max-output-tokens 65536.")

    # --max-model-len: a positive window, or -1 (vLLM fits the window to GPU memory). When gbench
    # launches the eval server, the answer budget must fit it; refuse here, before any pillar
    # runs, rather than clamp every answer or fail after hours of perf work.
    _mml = getattr(args, "max_model_len", None)
    if _mml is not None and _mml <= 0 and _mml != -1:
        parser.error("--max-model-len must be a positive number of tokens, or -1 to let vLLM "
                     "fit the context window to GPU memory.")
    if ((evals_selected or getattr(args, "evals_only", False)) and _mml and _mml > 0
            and not remote
            and (getattr(args, "max_output_tokens", None) or 0) > _mml - 256):
        parser.error(
            f"--max-output-tokens {args.max_output_tokens} does not fit --max-model-len {_mml}: "
            f"it must leave room for the prompt (at least 256 tokens), or every answer is "
            f"clamped. Raise --max-model-len, or omit it to serve the model's native window.")

    # A local run launches the server itself: a selected suite (or pillar) that sends native
    # tools needs the model's tool-call parser on it, or every such request gets HTTP 400.
    # The suites the run will execute (as get_config_from_args and the evals runner resolve them).
    _default_evals: List[str] = []
    if getattr(args, "eval_custom_jsonl", None):
        _default_evals = ["custom_jsonl"]
    elif getattr(args, "evals_only", False):
        _default_evals = ["plugins"] if getattr(args, "eval_plugins_dir", None) else ["all"]
    _planned_suites = _expand_selected_suites(flat_evals or _default_evals,
                                              getattr(args, "eval_plugins_dir", None))
    if flat_evals and getattr(args, "eval_custom_jsonl", None) and "custom_jsonl" not in _planned_suites:
        _planned_suites.append("custom_jsonl")
    if not remote and not getattr(args, "stage_to_gcs", None) \
            and not getattr(args, "tool_call_parser", None):
        from .runners.eval_suites import suite_requires_tool_calling
        _needs = [s for s in _planned_suites if suite_requires_tool_calling(s)]
        if getattr(args, "quality", False) or getattr(args, "quality_only", False):
            _needs.append("the quality pillar (GemmaClaw's tools)")
        if (getattr(args, "golden", False) or getattr(args, "golden_only", False)) \
                and _golden_tasks_send_tools(_split_golden_tasks(getattr(args, "golden_tasks", None))):
            _needs.append("the Golden Set's tool cases")
        if _needs:
            parser.error(
                f"--tool-call-parser NAME is required: {', '.join(_needs)} send(s) native tools, "
                "and the vLLM server gbench launches returns tool calls only with "
                "--enable-auto-tool-choice and the model's tool-call parser (without them every "
                "such request gets HTTP 400). Pass the parser the model's serving recipe names, "
                "or run against --remote-endpoint.")

    # A local --thinking run needs the model's reasoning parser: without one the server gbench
    # launches returns the reasoning inside the answer text (vLLM's default reasoning parser is
    # none), which the suites and the quality pillar would score as the answer.
    # `--reasoning-parser none` states that the model has none.
    if not remote and not getattr(args, "stage_to_gcs", None) \
            and getattr(args, "eval_thinking", False) \
            and not getattr(args, "reasoning_parser", None) \
            and (_planned_suites or getattr(args, "quality", False)
                 or getattr(args, "quality_only", False)):
        parser.error(
            "--thinking on a server gbench launches needs --reasoning-parser NAME: without the "
            "model's reasoning parser the server returns the reasoning inside the answer text, "
            "which would be scored as the answer. Pass the parser the model's serving recipe "
            "names, or --reasoning-parser none if the model has none, or run against "
            "--remote-endpoint.")

    # Setup logging
    setup_logging(args.verbose)

    # --preflight, --smoke and --parallel-suites (docs/smoke.md): run before (instead of) the
    # in-process sweep; each suite of a pool is its own `gbench --evals-only --evals <suite>`.
    _parallel = getattr(args, "parallel_suites", None)
    _smoke = bool(getattr(args, "smoke", False))
    _preflight = bool(getattr(args, "preflight", False))
    if _parallel is not None and _parallel < 1:
        parser.error("--parallel-suites must be at least 1")
    if _smoke or _preflight or (_parallel or 1) > 1:
        if _smoke and _preflight:
            parser.error("--smoke runs --preflight itself; pass one of them")
        if not flat_evals and not getattr(args, "evals_only", False):
            parser.error("--preflight, --smoke and --parallel-suites need the suites to check or "
                         "run: pass --evals (e.g. --evals all)")
        if not _planned_suites:
            parser.error("no eval suite is selected")
        if (_smoke or (_parallel or 1) > 1) and not remote:
            parser.error("--smoke and --parallel-suites need --remote-endpoint: every suite's "
                         "process uses the same served model (without it each would start its "
                         "own vLLM server on the same GPUs)")
        _non_eval = [flag for flag, on in (
            ("--golden/--golden-only", getattr(args, "golden", False)
             or getattr(args, "golden_only", False)),
            ("--quality/--quality-only", getattr(args, "quality", False)
             or getattr(args, "quality_only", False))) if on]
        if _non_eval and (_smoke or (_parallel or 1) > 1):
            parser.error(f"{', '.join(_non_eval)}: --smoke and --parallel-suites run eval suites "
                         "only; run the other pillars on their own")
        _shard_on = getattr(args, "shard", None) or (os.environ.get("GBENCH_SHARD") or "").strip()
        if _shard_on and (_smoke or (_parallel or 1) > 1):
            parser.error("--shard / GBENCH_SHARD cannot be combined with --smoke or "
                         "--parallel-suites: the pool joins one unsharded result per suite "
                         "(merge.union_runs); run the shards as separate runs and join them "
                         "with --merge")
        if _smoke and getattr(args, "skip_existing", False):
            parser.error("--skip-existing cannot be combined with --smoke: a smoke run checks "
                         "the harness as it is now and reuses no result")
        if (_smoke or (_parallel or 1) > 1) and not _evals_only(args):
            parser.error("--smoke and --parallel-suites run eval suites only, and this run also "
                         "selects performance pillars (--campaign / --input-lengths / "
                         "--output-lengths): pass --evals-only, or run the performance pillars "
                         "on their own")
        if _preflight:
            from .smoke import run_preflight_cli
            code, _report = run_preflight_cli(args, _planned_suites)
            return code
        if _smoke:
            if getattr(args, "dry_run", False):
                from . import pool as _pool
                _plan = _pool.build_plan(_planned_suites, _parallel or 8,
                                         _pool.load_durations(_pool.SMOKE_KEY))
                print("DRY RUN: --smoke would run --preflight, then this pool with "
                      "--eval-limit 1 and GBENCH_SMOKE=1:")
                _pool.print_plan(_plan, _pool.suite_argv(parser, raw_argv)(
                    _plan.order[0] if _plan.order else ""),
                                 Path(args.results_dir) / "smoke-<timestamp>")
                return 0
            from .smoke import run_smoke_cli
            return _shared_endpoint(args, lambda: run_smoke_cli(
                args, raw_argv, parser, _planned_suites))
        from .pool import run_pool_cli
        return _shared_endpoint(args, lambda: run_pool_cli(
            args, raw_argv, parser, _planned_suites))

    with _termination_handlers():
        return _run(args, parser, flat_evals, _planned_suites)


def _export_endpoint_key(remote: Optional[str]) -> Optional[str]:
    """The endpoint lock key (utils.endpoint_lock) of the server this run measures:
    computed once from the real upstream and exported, so every child (pool children,
    smoke taps, relays reached through another URL) locks the same server."""
    from .utils import endpoint_lock
    if not remote:
        return os.environ.get(endpoint_lock.KEY_ENV)
    if not os.environ.get(endpoint_lock.KEY_ENV):
        os.environ[endpoint_lock.KEY_ENV] = endpoint_lock.endpoint_key(remote)
    return os.environ[endpoint_lock.KEY_ENV]


def _shared_endpoint(args, run) -> int:
    """Run a pool or smoke run holding the server's lock SHARED for its whole life, so no
    performance pillar measures the server meanwhile (and none is running now)."""
    from .utils import endpoint_lock
    key = _export_endpoint_key(getattr(args, "remote_endpoint", None))
    try:
        with endpoint_lock.hold(key, exclusive=False, log=logger.warning):
            return run()
    except endpoint_lock.EndpointBusy as e:
        logger.error(f"{e}. Wait for it to finish, then re-run.")
        return EXIT_HARNESS_ERROR


def _run(args, parser, flat_evals, planned_suites) -> int:
    """The run itself, after argument validation (main())."""
    remote = getattr(args, "remote_endpoint", None)
    _export_endpoint_key(remote)
    # --personal (device-peak) mode: a single flag for "what will this model do
    # on my machine". Implemented as a thin convenience over existing gates -
    # serving-only + text-only (so stress/throughput/MM are skipped by the usual
    # guards) + few full-decode samples + the device-peak report. Incompatible
    # with --campaign (it uses a fixed single-stream shape).
    if getattr(args, "personal", False):
        # Reject flags that would silently produce a zero-benchmark no-op or fight
        # the single-stream shape. (--personal forces serving-only; combining it
        # with --stress-test/--throughput-only cancels every pillar.)
        _conflict = [name for name, flag in (
            ("--campaign", getattr(args, "campaign", None)),
            ("--stress-test", getattr(args, "stress_test", False)),
            ("--throughput-only", getattr(args, "throughput_only", False)),
        ) if flag]
        if _conflict:
            logger.error(f"--personal is incompatible with {'/'.join(_conflict)} "
                         f"(it runs single-stream text serving only). Drop one.")
            return EXIT_HARNESS_ERROR
        if not remote:
            logger.warning("--personal is intended for --remote-endpoint (e.g. Ollama at "
                           "http://localhost:11434/v1); without it a local vLLM + GPU is "
                           "required and the personal-device framing doesn't apply.")
        # Serving-only (single-stream) is the personal-device metric. MM is NOT
        # forced off: it is probe-gated (runs only if the endpoint accepts images),
        # so a vision model on this device still gets measured. Pass --text-only to
        # skip MM (it's slow single-stream on CPU).
        args.serving_only = True
        if remote and not args.text_only:
            logger.info("--personal: multimodal will run IF the endpoint accepts images "
                        "(probe-gated); note MM single-stream is slow on CPU. Add "
                        "--text-only to skip it.")

    # Get configuration
    config = get_config_from_args(args)
    # MM capability-probe timeout: honor the flag for any run; --personal defaults
    # it higher (cold CPU vision is slow to first token, so 10s falsely reads as
    # "unsupported" and skips MM).
    config.mm_probe_timeout = getattr(args, "mm_probe_timeout", None)
    # Per-request serving timeout: override the config default (600) only when the
    # flag is given, so unset runs keep the documented default.
    if getattr(args, "timeout", None) is not None:
        config.timeout = args.timeout
    # Thinking/reasoning mode: default OFF (no-think system prompt on chat
    # serving); the existing --thinking/--enable-thinking flag (dest eval_thinking)
    # enables it for serving too, so one flag governs "let the model think".
    config.thinking = bool(getattr(args, "eval_thinking", False))
    if getattr(args, "personal", False):
        config.personal_mode = True
        config.num_iterations = 1
        config.warmup_iterations = 1
        config.serving_num_prompts_override = 12
        config.batch_sizes = [1]
        # 1 image per request - lighter than 4 AND needs no server-side
        # --limit-mm-per-prompt>1, which a remote endpoint (Ollama) may not have. MM
        # still probe-gated; skipped if the endpoint refuses images.
        config.mm_images_per_request = 1
        if config.mm_probe_timeout is None:
            config.mm_probe_timeout = 120   # slow CPU vision needs a longer probe
        if not args.output_lengths:
            config.output_lengths = [512]
        if not args.input_lengths:
            config.input_lengths = [512]
    # The stress SLOs this run uses (a global override, else the campaign's defaults), resolved
    # once and stored on the config so the scaffold records the SLO behind the stress verdict.
    _ttft_slo, _itl_slo = resolve_campaign_slo(
        config.campaign, args.stress_threshold, args.itl_slo_ms
    )
    config.stress_ttft_slo_ms, config.stress_itl_slo_ms = _ttft_slo, _itl_slo
    evals_only_mode = _evals_only(args)
    logger.info(f"Using configuration: {args.preset}")
    if evals_only_mode:
        logger.info(f"  Mode: Evaluation Benchmarks Only")
        logger.info(f"  Evals: {config.evals}")
        logger.info(f"  Concurrency: {config.eval_concurrency} per suite "
                    f"({'--eval-concurrency' if getattr(args, 'eval_concurrency', None) else '--batch-sizes[0]'}"
                    f"; containerized suites: "
                    f"{config.sandboxes if config.sandboxes else 'the same'} via --sandboxes)")
        logger.info(f"  Thinking enabled: {config.eval_thinking}")
        logger.info(f"  Max output tokens: {config.eval_max_output_tokens}")
    else:
        logger.info(f"  Iterations: {config.num_iterations} (+{config.warmup_iterations} warmup)")
        logger.info(f"  Batch sizes: {config.batch_sizes}")
        logger.info(f"  Input/Output lengths: {config.input_lengths}/{config.output_lengths}")
        # The serving sample count is NOT config.num_prompts (dead for serving) -
        # it's computed per config by _serving_num_prompts (honors the --personal
        # override). Show the effective value so the summary isn't misleading.
        try:
            _serv_n = config._serving_num_prompts(
                config.batch_sizes[0] if config.batch_sizes else 1)[0]
        except Exception:
            _serv_n = config.num_prompts
        logger.info(f"  Num prompts: serving={_serv_n} (auto-sized per config; ~5-min wall-clock "
                    f"target, scales with batch size and output length - NOT --num-prompts), "
                    f"throughput={config.num_prompts_throughput}")
        if getattr(args, "num_prompts", None):
            logger.warning(
                "  --num-prompts=%s does NOT control the serving pillar (it auto-sizes per config, "
                "shown above). Use --batch-sizes / --output-lengths to change the serving count, or "
                "--personal for a fixed count. Throughput uses its own count (%s).",
                args.num_prompts, config.num_prompts_throughput)
        stress_disabled = args.no_stress_test or args.serving_only or args.throughput_only
        logger.info(f"  Stress test: {'disabled' if stress_disabled else 'enabled (ramp-up)'}")
    logger.info(f"Results directory: {config.results_dir}"
                + (" (dry run: not created)" if config.dry_run else ""))
    if config.tool_call_parser or config.reasoning_parser or config.vllm_serve_args:
        logger.info(f"  Serve settings: {serve_settings_record(config)}")

    # Pre-flight over-subscription guard: a container/sandbox concurrency beyond this host's
    # CPU/RAM would exhaust it mid-sweep. Guarded: an explicit --sandboxes always (plugins may
    # honor it too), and --batch-sizes[0] when it is the sandbox concurrency of a selected
    # containerized suite (SANDBOXED_SUITES; --sandboxes unset). --batch-sizes for HTTP suites is
    # request concurrency and is not guarded. Checked in a dry run too (bypass with
    # GBENCH_SKIP_CALIBRATION_GUARD=1).
    from .runners.eval_suites import SANDBOXED_SUITES
    _sandboxed = [s for s in planned_suites if s in SANDBOXED_SUITES]
    _eff_conc, _conc_flag = None, None
    if config.sandboxes is not None:
        _eff_conc, _conc_flag = config.sandboxes, f"--sandboxes {config.sandboxes}"
    elif _sandboxed and config.batch_sizes:
        _eff_conc = config.batch_sizes[0]
        _conc_flag = (f"--batch-sizes {_eff_conc} (used as sandbox concurrency of "
                      f"{', '.join(_sandboxed[:4])}{'...' if len(_sandboxed) > 4 else ''} because "
                      "--sandboxes is unset)")
    if _eff_conc:
        from .utils.calibrate import detect_hardware, check_oversubscription
        # The selected suites, so the Docker network cap applies when a suite that creates one
        # network per concurrent task (terminal_bench, swe_bench_pro, swe_lancer) runs.
        _ok, _msg = check_oversubscription(_eff_conc, detect_hardware(), suites=planned_suites)
        if not _ok:
            logger.error(f"{_conc_flag}: {_msg}"
                         + (" (the real run would fail)" if config.dry_run else ""))
            return EXIT_HARNESS_ERROR

    # Get models to benchmark
    if args.models and not remote and not args.stage_to_gcs:
        for name in args.models:
            if "/" not in name and not os.path.isdir(name):
                logger.error(
                    f"--models '{name}' is neither a HuggingFace id (org/name) nor a local "
                    "checkpoint directory, so gbench cannot serve it. Pass the HuggingFace id "
                    "(or a local path), or --remote-endpoint with the id that endpoint serves.")
                return EXIT_HARNESS_ERROR
    try:
        models = get_models_from_args(args)
    except ModelSelectionError as e:
        logger.error(str(e))
        return EXIT_HARNESS_ERROR
    if not models:
        logger.error("No models selected: pass --models (or --remote-endpoint, whose served "
                     "model is then used).")
        return EXIT_HARNESS_ERROR

    if args.stage_to_gcs:
        failed_stage = stage_models_to_gcs(models, args.stage_to_gcs)
        if isinstance(failed_stage, list) and failed_stage:
            logger.error("GCS staging failed for: %s", ", ".join(failed_stage))
            return EXIT_HARNESS_ERROR
        return 0

    if not remote:
        from .core.models import METADATA_DEFAULTS
        # metadata_source "defaults" with no size: no config.json was read (from the Hugging Face
        # Hub at the commit the id resolves to, or a local directory's own), so the values are
        # placeholders. A local directory is no longer exempt: its own config.json is read.
        unread = [m.name for m in models
                  if getattr(m, "metadata_source", METADATA_DEFAULTS) == METADATA_DEFAULTS
                  and not m.total_params_b]
        if unread:
            logger.error(
                f"Could not read the configuration of {', '.join(unread)} (its config.json: from "
                "the Hugging Face Hub, or a local checkpoint directory's own): its size and "
                "modalities are unknown, so the run cannot size the server. Check the id or "
                "directory, HF_TOKEN for a gated model, or network access (a cached model works "
                "offline with HF_HUB_OFFLINE=1).")
            return EXIT_HARNESS_ERROR
        if config.eval_max_soft_tokens is not None:
            for m in models:
                declared = _processor_declares(m.hf_model_id or m.name, "max_soft_tokens")
                if declared is not True:
                    logger.error(
                        f"--eval-max-soft-tokens: the processor of {m.name} "
                        + ("declares no max_soft_tokens setting" if declared is False else
                           "configuration could not be read (processor_config.json / "
                           "preprocessor_config.json)")
                        + ", so the value would not reach the model. Omit the flag.")
                    return EXIT_HARNESS_ERROR

    logger.info(f"Selected {len(models)} model(s) for benchmarking")

    # Save run metadata
    metadata = {
        "timestamp": config.log_manager.timestamp if config.log_manager else None,
        "models": [m.name for m in models],
        "tags": config.tags or [],
        "preset": args.preset,
        "benchmark_mode": _benchmark_mode(args),
        # the node that ran it (a later run on another node does not continue this directory)
        "host": socket.gethostname(),
        "shard": config.shard,
        "eval_limit": config.eval_limit,
        # why --results-dir, an existing run directory, was not continued in place (another run
        # was writing there): this run's directory is inside it
        **({"results_dir_not_shared": config.log_manager.shared_dir_note}
           if getattr(config.log_manager, "shared_dir_note", None) else {}),
        "serve_settings": serve_settings_record(config),
        **_served_model_id_record(config, args, models),
        # identity, derived values and where they came from (metadata_source / metadata_origin,
        # revision / revision_source, vision_tokens_source)
        "model_metadata": [m.metadata_record() for m in models if hasattr(m, "metadata_record")],
    }
    if config.log_manager and not config.dry_run:
        config.log_manager.save_metadata(metadata)
    for model in models:
        moe_tag = f", MoE {model.num_experts}x{model.num_active_experts}" if model.is_moe else ""
        logger.info(f"  - {model.name} ({model.total_params_b:.0f}B{moe_tag})")

    # Validate GPU configuration against first model (skip if remote endpoint is used)
    first_model = models[0]
    if remote:
        if config.dry_run:
            logger.info(f"Remote endpoint specified ({config.remote_endpoint}); dry run: not "
                        "contacting it.")
        else:
            logger.info(f"Remote endpoint specified ({config.remote_endpoint}); checking if endpoint is functional...")
            from .utils import verify_endpoint_functional
            # The window logged is that of the served id this run uses (--served-model-id, else
            # the model's id), not another listed model's.
            is_up, err_msg, max_model_len = verify_endpoint_functional(
                config.remote_endpoint,
                model_id=getattr(config, "served_model_id", None) or first_model.hf_model_id)
            if not is_up:
                logger.error(f"Remote endpoint '{config.remote_endpoint}' is unreachable or not functional: {err_msg}")
                logger.error("Cancelling benchmark run.")
                return EXIT_HARNESS_ERROR
            logger.info(f"Remote endpoint '{config.remote_endpoint}' is functional and answering requests.")
            if max_model_len:
                logger.info(f"   Server reported max_model_len: {max_model_len} tokens")
    else:
        # A local run launches a vLLM server for every pillar (performance, quality, Golden Set,
        # evals); its TP = --tensor-parallel (default --num-gpus), so omitting --num-gpus would
        # silently serve a large model at TP=1 and OOM. A dry run validates as the real run would.
        # Post-dispatch guard (NOT arg required=True) so --list/--help/--remote are never blocked.
        from .runners.serving import server_host_env_error
        _host_error = server_host_env_error()
        if _host_error:
            logger.error(_host_error + (" (the real run would fail)" if config.dry_run else ""))
            return EXIT_HARNESS_ERROR
        _served = getattr(config, "served_model_id", None)
        if _served and _served not in {m.hf_model_id for m in models}:
            # The server gbench launches serves each model under its model argument (vllm
            # serve's id; --served-model-name is refused), so any other id would fail the Golden
            # Set and every eval only after the server and the performance pillars ran.
            logger.error(
                f"--served-model-id {_served!r} names the id a --remote-endpoint serves the model "
                "under; the server gbench launches serves each model under its --models id or "
                f"path ({', '.join(sorted({m.hf_model_id for m in models}))}). Drop it, or run "
                "against --remote-endpoint." + (" (the real run would fail)" if config.dry_run
                                                 else ""))
            return EXIT_HARNESS_ERROR
        if args.num_gpus is None:
            logger.error(
                "--num-gpus is required for a local run (it launches a vLLM server) so "
                "tensor-parallel matches the model size (e.g. --num-gpus 1). Omit it only with "
                "--remote-endpoint." + (" (the real run would fail)" if config.dry_run else ""))
            return EXIT_HARNESS_ERROR
        is_valid, gpu_msg = validate_gpu_config(
            config.num_gpus, first_model.total_params_b,
            gpu_memory_utilization=config.gpu_memory_utilization)
        if not is_valid:
            logger.error(f"GPU configuration error: {gpu_msg}"
                         + (" (the real run would fail)" if config.dry_run else ""))
            return EXIT_HARNESS_ERROR
        (logger.warning if "Warning" in gpu_msg else logger.info)(gpu_msg)
        # GPU selection: --gpus, else idle GPUs; the server runs with CUDA_VISIBLE_DEVICES set to
        # the chosen UUIDs (cuda_visible_devices_env). In a dry run a busy GPU is advisory.
        _chosen, _sel_msg = select_gpus(config.num_gpus, config.gpus)
        if _chosen is None:
            if config.dry_run:
                logger.warning(f"GPU selection (advisory in a dry run): {_sel_msg}")
            else:
                logger.error(f"GPU selection failed: {_sel_msg}")
                return EXIT_HARNESS_ERROR
        else:
            config.selected_gpus = [g.uuid for g in _chosen] or None
            logger.info(f"GPU selection: {_sel_msg}")
        # Model-aware weights-fit ADVISORY (WARN only; vLLM's profiler is the hard
        # authority). Sized on the largest selected model, at the resolved TP, over the GPUs the
        # run selected (every visible GPU when none could be selected: a dry run).
        _max_params = max((m.total_params_b for m in models), default=first_model.total_params_b)
        _fits, _fit_msg = weights_fit_advisory(
            _max_params, config.tensor_parallel_size or config.num_gpus or 1,
            config.gpu_memory_utilization, min_free_override=getattr(args, "min_free_gb", None),
            gpus=_chosen or None)
        (logger.warning if not _fits else logger.info)(_fit_msg)
        if not config.dry_run:
            from gbench.utils import require_vllm_engine
            require_vllm_engine("Local GPU benchmarking")
        # Process/idle check on the selected GPUs (no hard free-memory floor by default;
        # --min-free-gb re-enables one). Advisory in a dry run.
        gpu_ready, gpu_msg = check_gpu_ready(min_free_memory_gb=getattr(args, "min_free_gb", None),
                                             gpu_uuids=config.selected_gpus)
        if not gpu_ready:
            if config.dry_run:
                logger.warning(f"GPU pre-flight (advisory in a dry run): {gpu_msg}")
            else:
                logger.error(
                    f"GPU pre-flight check failed:\n{gpu_msg}\n"
                    "No local CUDA GPU? Run against a remote vLLM server with "
                    "--remote-endpoint <url> (serving + stress; offline throughput "
                    "needs a local GPU).")
                return EXIT_HARNESS_ERROR
        elif gpu_msg.startswith("WARNING"):
            logger.warning(gpu_msg)
        else:
            logger.info(gpu_msg)
    metadata["selected_gpus"] = config.selected_gpus
    if remote:
        # gbench started no server: the endpoint's parallelism is not known here (the CLI's
        # defaults would read as a measured TP=1).
        metadata["parallel"] = {"tensor_parallel_size": None, "data_parallel_size": None,
                                "note": "not started by gbench: the remote endpoint's "
                                        "parallelism is unknown"}
    else:
        metadata["parallel"] = {"tensor_parallel_size": config.tensor_parallel_size,
                                "data_parallel_size": config.data_parallel_size}

    # Get formats
    formats = get_formats_from_args(args)
    logger.info(f"Testing formats: {[f.value for f in formats]}")

    # Determine benchmark modes
    # Default: run text + multimodal for multimodal-capable models
    # --text-only: skip multimodal benchmarks entirely
    # --multimodal-only: skip text benchmarks, only run multimodal
    # --serving-only / --throughput-only: control benchmark type, not text vs multimodal
    evals_only = _evals_only(args)
    run_golden = args.golden or args.golden_only
    run_quality = args.quality or args.quality_only
    run_text = not args.multimodal_only and not args.quality_only and not args.golden_only and not evals_only
    run_multimodal = not args.text_only and not args.quality_only and not args.golden_only and not evals_only

    if args.multimodal_only:
        run_multimodal = True
        run_text = False

    # Display benchmark mode (the same string metadata.json records)
    benchmark_mode = _benchmark_mode(args)
    logger.info(f"Benchmark mode: {benchmark_mode}")

    # Create runners
    from .runners import ServingBenchmarkRunner, ThroughputBenchmarkRunner, StressTestRunner, QualityBenchmarkRunner, GoldenBenchmarkRunner, EvalsBenchmarkRunner
    serving_runner = ServingBenchmarkRunner(config)
    throughput_runner = ThroughputBenchmarkRunner(config)
    # The SLO for the active campaign was resolved above (a CLI global override, else the
    # per-campaign default map), frozen for this run and recorded in the fingerprint.
    stress_runner = StressTestRunner(
        config,
        ttft_threshold_ms=_ttft_slo,
        tpot_threshold_ms=_itl_slo,
    )
    quality_runner = QualityBenchmarkRunner(config)
    golden_runner = GoldenBenchmarkRunner(config)
    evals_runner = EvalsBenchmarkRunner(config)

    # Determine if stress test should run
    # --stress-test means run stress test only (stress_test_only)
    # --no-stress-test means skip stress test entirely
    # --serving-only or --throughput-only also skip stress test
    stress_test_only = args.stress_test  # --stress-test means ONLY run stress test
    run_stress_test = (
        (args.stress_test or not args.no_stress_test)
        and not args.serving_only
        and not args.throughput_only
        and not args.quality_only
        and not args.golden_only
        and not evals_only
    )

    # Scaffold contract: record every (pillar, model, format) condition this
    # run plans to produce. See docs/scaffold-versioning.md.
    #
    # `conditions` is the planned matrix, not the outcome. Multimodal pillars
    # are listed from the model's declared capability, and a remote endpoint
    # probe can still skip one at run time. The results say what completed.
    from .core.scaffold import (
        Pillar,
        build_condition,
        gbench_commit,
        render_conditions,
        resolve_gemmaclaw_sha,
    )

    # A branch name is not a pin. The default is already a sha, so this is
    # a no-op that touches nothing for an unflagged run, but anyone passing
    # `--gemmaclaw-commit main` or a tag needs it resolved. Resolve once,
    # here, so the scaffold built below and the checkout the quality runner
    # performs later are the same commit by construction. Resolving in both
    # places instead would let `main` advance in between and pin a scorer
    # the run never used.
    #
    # Gated on the pillar actually being planned, because a non-sha ref
    # reaches the network and a serving-only run has no business doing that.
    if run_quality:
        resolved = resolve_gemmaclaw_sha(config.gemmaclaw_commit, config.gemmaclaw_path)
        if resolved:
            if resolved != config.gemmaclaw_commit:
                logger.info(
                    f"Resolved gemmaclaw ref '{config.gemmaclaw_commit}' to {resolved}"
                )
            config.gemmaclaw_commit = resolved
        else:
            logger.warning(
                f"Could not resolve gemmaclaw ref '{config.gemmaclaw_commit}' to a "
                "commit. The quality scaffold_id will pin the ref itself, so it "
                "will not move when that ref does."
            )

    conditions = []
    is_remote = bool(remote)
    for model in models:
        mm = run_multimodal and model.supports_multimodal
        pillars = [
            (Pillar.SERVING, run_text and not args.throughput_only and not stress_test_only),
            (Pillar.THROUGHPUT, run_text and not args.serving_only and not stress_test_only and not is_remote),
            (Pillar.SERVING_MULTIMODAL,
             mm and not args.throughput_only and not stress_test_only),
            (Pillar.THROUGHPUT_MULTIMODAL,
             mm and not args.serving_only and not stress_test_only and not is_remote),
            (Pillar.STRESS_TEST, run_stress_test),
            (Pillar.STRESS_TEST_MULTIMODAL, mm and run_stress_test),
            (Pillar.QUALITY, run_quality),
            (Pillar.GOLDEN, run_golden),
            (Pillar.EVALS, bool(config.evals)),
        ]
        for pillar, planned in pillars:
            if not planned:
                continue
            for fmt in formats:
                is_golden = pillar is Pillar.GOLDEN
                conditions.append(build_condition(
                    pillar,
                    model,
                    fmt,
                    dataset_dir=golden_runner.dataset_dir if is_golden else None,
                    selected_tasks=getattr(args, "golden_tasks", None) if is_golden else None,
                    config=config,
                ))

    # Printed before the persist, and outside its condition, because the
    # console and metadata.json are two views of the same objects rather
    # than one depending on the other. A run with no log manager, a dry
    # run, and a GCS staging run are all still entitled to know what
    # scaffold they are about to measure against.
    if conditions:
        print("\n" + render_conditions(conditions))

    # --shard splits the eval suites only: every other planned pillar runs in full on each node,
    # which its records say (shard_requested, shard_applied=false), like a suite that cannot
    # shard, so a node's perf, quality or Golden Set result is never read as a shard of one.
    shard_unapplied = (sorted({c.pillar.value for c in conditions if c.pillar is not Pillar.EVALS})
                       if config.shard else [])
    if shard_unapplied:
        logger.warning(
            f"--shard {config.shard} splits the eval suites only: {', '.join(shard_unapplied)} "
            "run in full on this node and record shard_applied=false (--merge does not merge "
            "them). Run them on one node, and the shard nodes with --evals-only.")
        metadata["shard_not_applied"] = {"shard_requested": config.shard,
                                         "pillars": shard_unapplied}

    # A SECOND save_metadata rather than a move of the first one. The early
    # write happens before the plan exists, and removing it would leave an
    # aborted run as a directory with no metadata.json at all, so nothing in
    # it would say which models it ran.
    if config.log_manager and not config.dry_run:
        metadata["conditions"] = [c.to_dict() for c in conditions]
        # Beside the conditions rather than inside any id. Results are
        # never uploaded anywhere central by a CLI run, so a run
        # directory has to be self-describing: there is no index to
        # reconstruct later which harness produced it.
        metadata["gbench"] = gbench_commit()
        config.log_manager.save_metadata(metadata)

    native_planned = bool(run_quality or run_golden or config.evals)
    if config.dry_run:
        _print_dry_run_plan(
            config, models, formats, serving_runner,
            perf_planned=bool(run_text and not remote) or bool(run_stress_test and not remote),
            native_planned=native_planned,
            mm_planned=lambda m: bool(run_multimodal and m.supports_multimodal and not remote))
        logger.info("Dry-run complete. Exiting without executing benchmarks.")
        return 0

    # Track all results for final summary
    all_results = []

    def _stamp(result: Dict[str, Any], btype: str, model, fmt, camp=None) -> Dict[str, Any]:
        result['benchmark_type'] = btype
        result['model'] = model.short_name
        result['model_name'] = model.name
        result['model_short'] = model.short_name
        result['format'] = fmt.value
        if camp:
            result['campaign'] = camp
        if config.shard and btype != 'eval':
            # A non-eval pillar runs in full on every node (shard_unapplied above).
            result['shard_requested'] = config.shard
            result['shard_applied'] = False
        return result

    from .utils import endpoint_lock as _endpoint_lock

    def _endpoint_hold(exclusive: bool):
        """The server lock for a pillar against the remote endpoint (a server gbench
        launches is its own): exclusive for performance pillars, shared otherwise."""
        if not remote:
            return contextlib.nullcontext()
        return _endpoint_lock.hold(_endpoint_lock.inherited_or(remote), exclusive=exclusive,
                                   log=logger.warning)

    def _perf_pillar(btype: str, model, fmt, run, camp=None) -> None:
        """One perf configuration: its results, or a failed record when it raised (an
        endpoint held by other gbench runs is such a failure: nothing is measured)."""
        try:
            with _endpoint_hold(exclusive=True):
                results = list(run())
            for result in results:
                all_results.append(_stamp(result, btype, model, fmt, camp))
        except Exception as e:                                             # noqa: BLE001
            logger.error(f"{btype} for {model.name} ({fmt.value}) failed: {e}")
            all_results.append(_pillar_failure(btype, model, fmt, e, camp))

    def _eval_failures_for_all_suites(model, fmt, error: str) -> List[Dict[str, Any]]:
        from .runners.eval_suites.canonical_sync import canonical_sync_for
        names = planned_suites or ["evals"]
        return [{"benchmark_type": "eval", "eval_name": name, "model_name": model.name,
                 "model_short": model.short_name, "format": fmt.value, "status": "error",
                 "failed": True, "failure_kind": "error", "error": error,
                 "total_questions": 0, "correct_answers": 0, "accuracy": None,
                 "accuracy_withheld_reason": "the suite did not run",
                 "canonical_sync": canonical_sync_for(name)} for name in names]

    campaigns_to_run = ((args.campaign if isinstance(args.campaign, list) else [args.campaign])
                        if getattr(args, "campaign", None) else [None])

    # Execute benchmarks
    try:
        for model in models:
            # Check multimodal compatibility (dynamic HTTP probe for remote endpoints)
            if remote and run_multimodal:
                from gbench.utils import probe_multimodal
                _probe_to = int(getattr(config, "mm_probe_timeout", None) or 10)
                _probe_id = (getattr(config, "served_model_id", None) or model.hf_model_id
                             or model.name)
                logger.info(f"Probing remote endpoint multimodal capability for {model.name} "
                            f"(served id {_probe_id}, timeout {_probe_to}s)...")
                _probe = probe_multimodal(config.remote_endpoint, _probe_id, timeout=_probe_to)
                if _probe.verdict == "supported":
                    model_supports_multimodal = True
                    logger.info(f"Remote model {model.name} ({_probe.model_id}) supports "
                                "multimodal requests.")
                elif _probe.verdict == "unsupported":
                    model_supports_multimodal = False
                    logger.info(f"Remote model {model.name} ({_probe.model_id}): the endpoint "
                                f"rejected the image payload ({_probe.reason}); text-only, "
                                "skipping MM. A multimodal model accepts the payload even when "
                                "a cold vision load makes it slow to respond.")
                else:
                    # An endpoint failure is no capability verdict: the MM pillars run, and
                    # their failures are harness errors, not a text-only skip.
                    model_supports_multimodal = True
                    logger.warning(f"Remote model {model.name}: the multimodal probe was "
                                   f"inconclusive ({_probe.reason}); the multimodal pillars run, "
                                   "and a failure there counts as a harness error.")
            else:
                # Image-based MM benchmarks require VISION, so key off
                # supports_multimodal (has_vision) - NOT category==MULTIMODAL, which
                # is vision OR audio; an audio-only model would otherwise be handed
                # image benchmarks it cannot serve.
                model_supports_multimodal = model.supports_multimodal

            if run_multimodal and not model_supports_multimodal:
                logger.warning(
                    f"{model.name} does not support multimodal, skipping multimodal benchmarks"
                )

            for format in formats:
                logger.info(
                    f"\n{'='*60}\n"
                    f"Benchmarking: {model.name} ({format.value})\n"
                    f"{'='*60}"
                )

                # Apply param-based batch sizes (unless user specified --batch-sizes)
                if not args.batch_sizes:
                    model_batch_sizes = get_batch_sizes(model.total_params_b, args.preset)
                    config.batch_sizes = model_batch_sizes
                    logger.info(f"  Batch sizes for {model.total_params_b:.0f}B: {model_batch_sizes}")

                # Each pillar is its own unit: one that raises becomes a failed record (a harness
                # error) and the next pillar still runs.
                if run_text and not args.throughput_only and not stress_test_only:
                    for camp in campaigns_to_run:
                        if camp:
                            logger.info(f"Running serving benchmarks (text) - campaign: {camp}...")
                            apply_campaign_to_config(config, camp, args)
                        else:
                            logger.info("Running serving benchmarks (text)...")
                        _perf_pillar('serving', model, format,
                                     lambda: serving_runner.run_all(model, format), camp)

                if run_text and not args.serving_only and not stress_test_only and not remote:
                    for camp in campaigns_to_run:
                        if camp:
                            logger.info(f"Running throughput benchmarks (text) - campaign: {camp}...")
                            apply_campaign_to_config(config, camp, args)
                        else:
                            logger.info("Running throughput benchmarks (text)...")
                        _perf_pillar('throughput', model, format,
                                     lambda: throughput_runner.run_all(model, format), camp)

                # Multimodal serving: real image+text requests. A multimodal failure on a model
                # whose image support is confirmed (config, or the endpoint probe) is a harness
                # error like any other pillar's.
                if run_multimodal and model_supports_multimodal and not args.throughput_only and not stress_test_only:
                    for camp in campaigns_to_run:
                        if camp:
                            logger.info(f"Running serving benchmarks (multimodal) - campaign: {camp}...")
                            apply_campaign_to_config(config, camp, args)
                        else:
                            logger.info("Running serving benchmarks (multimodal)...")
                        _perf_pillar('serving_multimodal', model, format,
                                     lambda: serving_runner.run_all(model, format, multimodal=True),
                                     camp)

                if run_multimodal and model_supports_multimodal and not args.serving_only and not stress_test_only and not remote:
                    for camp in campaigns_to_run:
                        if camp:
                            logger.info(f"Running throughput benchmarks (multimodal) - campaign: {camp}...")
                            apply_campaign_to_config(config, camp, args)
                        else:
                            logger.info("Running throughput benchmarks (multimodal)...")
                        _perf_pillar('throughput_multimodal', model, format,
                                     lambda: throughput_runner.run_all(model, format, multimodal=True),
                                     camp)

                # Run stress test (ramp-up to find max sustainable throughput)
                if run_stress_test:
                    # One stress pass per campaign (its shape, dataset and SLOs), like serving.
                    for camp in campaigns_to_run:
                        if camp:
                            logger.info(f"Running stress test (finding max sustainable throughput) "
                                        f"- campaign: {camp}...")
                            apply_campaign_to_config(config, camp, args)
                            _t, _i = resolve_campaign_slo(camp, args.stress_threshold,
                                                          args.itl_slo_ms)
                            config.stress_ttft_slo_ms, config.stress_itl_slo_ms = _t, _i
                            stress_runner.ttft_threshold_ms, stress_runner.tpot_threshold_ms = _t, _i
                        else:
                            logger.info("Running stress test (finding max sustainable throughput)...")
                        try:
                            # Honor --text-only / --multimodal-only for stress too: the text pass
                            # is gated by run_text, the multimodal pass by run_multimodal and the
                            # model's image support (a text-only model, or a text-only remote
                            # endpoint, never builds images).
                            with _endpoint_hold(exclusive=True):
                                stress_results = stress_runner.run_all(
                                    model, format,
                                    include_text=run_text,
                                    include_multimodal=run_multimodal and model_supports_multimodal,
                                )
                            for result in stress_results:
                                all_results.append(_stamp(
                                    result, 'stress_test_multimodal' if result.get('multimodal')
                                    else 'stress_test', model, format, camp))
                        except Exception as e:                             # noqa: BLE001
                            logger.error(f"stress test for {model.name} failed: {e}")
                            all_results.append(_pillar_failure('stress_test', model, format, e,
                                                               camp))

                # Quality, Golden Set and evals share ONE native server per model on a local run:
                # started here (after the perf servers are gone), handed to the runners as
                # config.local_endpoint, and stopped when they are done.
                native_server = None
                if native_planned and not remote:
                    native_server = ServingBenchmarkRunner(config)
                    logger.info("Starting the local vLLM server for quality / Golden Set / evals...")
                    try:
                        started = native_server._start_server(model, format, context="native")
                    except Exception as e:                                 # noqa: BLE001
                        logger.error(f"the local vLLM server raised while starting: {e}")
                        started = False
                    if not started:
                        err = ("the local vLLM server for quality / Golden Set / evals did not "
                               "start (see its log)")
                        if run_quality:
                            all_results.append(_pillar_failure('quality', model, format, err))
                        if run_golden:
                            all_results.append(_pillar_failure('golden', model, format, err))
                        if config.evals:
                            all_results.extend(_eval_failures_for_all_suites(model, format, err))
                        try:
                            native_server._cleanup_server()
                        except Exception:                                  # noqa: BLE001
                            pass
                        continue
                    config.local_endpoint = f"http://127.0.0.1:{native_server.server_port}/v1"
                    logger.info(f"Local server for quality / Golden Set / evals: "
                                f"{config.local_endpoint}")
                    # Its bind address, why, and whether it enforces an endpoint key: on every
                    # quality, Golden Set and eval result (`server`) and in metadata.json.
                    try:
                        config.native_server_record = native_server.server_record()
                    except Exception as e:                                 # noqa: BLE001
                        logger.warning(f"could not record the local server's settings: {e}")
                        config.native_server_record = None
                    if config.native_server_record:
                        metadata.setdefault("native_servers", []).append(
                            {"model": model.name, "format": format.value,
                             **config.native_server_record})
                        if config.log_manager and not config.dry_run:
                            config.log_manager.save_metadata(metadata)
                _native_hold = _endpoint_hold(exclusive=False)
                try:
                    _native_hold.__enter__()
                except _endpoint_lock.EndpointBusy as e:
                    logger.error(f"{e}")
                    if run_quality:
                        all_results.append(_pillar_failure('quality', model, format, e))
                    if run_golden:
                        all_results.append(_pillar_failure('golden', model, format, e))
                    if config.evals:
                        all_results.extend(_eval_failures_for_all_suites(model, format, str(e)))
                    continue
                try:
                    if run_quality:
                        logger.info("Running quality benchmarks (gemmaclaw)...")
                        try:
                            all_results.append(_stamp(quality_runner.run(model, format),
                                                      'quality', model, format))
                        except Exception as e:                             # noqa: BLE001
                            logger.error(f"quality for {model.name} failed: {e}")
                            all_results.append(_pillar_failure('quality', model, format, e))

                    if run_golden:
                        logger.info("Running Golden Set benchmarks...")
                        try:
                            golden_res = _stamp(golden_runner.run(model), 'golden', model,
                                                format)
                            if config.native_server_record and not remote:
                                golden_res.setdefault("server",
                                                      dict(config.native_server_record))
                            all_results.append(golden_res)
                        except Exception as e:                             # noqa: BLE001
                            logger.error(f"Golden Set for {model.name} failed: {e}")
                            all_results.append(_pillar_failure('golden', model, format, e))

                    if config.evals:
                        logger.info("Running Evaluation benchmarks...")
                        try:
                            all_results.extend(evals_runner.run(model, format))
                        except Exception as e:                             # noqa: BLE001
                            logger.error(f"evals for {model.name} failed: {e}")
                            all_results.extend(_eval_failures_for_all_suites(
                                model, format, f"the evals pillar raised: {e}"))
                finally:
                    _native_hold.__exit__(None, None, None)
                    if native_server is not None:
                        logger.info("Stopping the local vLLM server for quality / Golden Set / "
                                    "evals...")
                        config.local_endpoint = None
                        config.native_server_record = None
                        try:
                            native_server._cleanup_server()
                        except Exception as e:                             # noqa: BLE001
                            logger.error(f"could not stop the local vLLM server: {e}")

        # Print comprehensive results summary
        print_results_summary(all_results, config)
    except SystemExit:
        # A SIGTERM/SIGHUP (turned into SystemExit by _termination_handlers): summary.json still
        # records what finished and the failed record of the suite the signal cut short
        # (evals_runner._stopped_record), then the exit goes on.
        from .utils.process_groups import stop_requested
        if stop_requested():
            for rec in getattr(evals_runner, "partial_results", None) or []:
                if not any(rec is r for r in all_results):
                    all_results.append(rec)
            try:
                print_results_summary(all_results, config)
            except Exception as e:                                         # noqa: BLE001
                logger.error(f"could not write the summary of the stopped run: {e}")
        raise
    finally:
        # Every file the run left under its results directory - including what the serving,
        # stress and quality tools, a harness or the local server wrote there themselves -
        # carries no credential value, however the run ended (an exception or Ctrl-C included).
        _redact_run_results(config)

    return run_exit_code(all_results)


def eval_failures_exit_code(all_results: list) -> Optional[int]:
    """EXIT_HARNESS_ERROR (logged, naming each suite and its failure label) when an eval
    configuration produced no score, else None.

    Evals that produced no score (a suite that raised, for example [INFRA REQUIRED]; one that
    returned status error / failed / timeout; the local eval server did not start) are a harness
    error, not a verdict on the model, and must not let the run exit 0."""
    eval_harness_failures = [
        r for r in all_results
        if r.get("failed") is True and r.get("benchmark_type") == "eval"
    ]
    if not eval_harness_failures:
        return None
    from .runners.evals import failure_label
    logger.error(
        "%d eval suite(s) produced no score: %s. Exiting %d (harness error). "
        "This is not a verdict on the model.", len(eval_harness_failures),
        ", ".join(sorted({f"{r.get('eval_name', '?')} ({failure_label(r)})"
                          for r in eval_harness_failures})),
        EXIT_HARNESS_ERROR)
    return EXIT_HARNESS_ERROR


def _redact_run_results(config) -> None:
    """redact_tree over the run's results directory (never raises)."""
    lm = getattr(config, "log_manager", None)
    root = getattr(lm, "results_dir", None)
    if root is None:
        return
    try:
        from .utils.redaction import redact_tree
        redact_tree(root)
    except Exception as e:                                                 # noqa: BLE001
        logger.error("Could not check %s for credential values: %s", root, e)


def print_results_summary(results: list[dict], config: BenchmarkConfig):
    """Print comprehensive summary of all benchmark results."""
    if not results:
        logger.warning("No benchmark results to summarize")
        return

    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY".center(80))
    print("="*80 + "\n")

    # Separate successful and failed results
    successful_results = [r for r in results if not r.get('failed', False)]
    failed_results = [r for r in results if r.get('failed', False)]

    # Group results by benchmark type (text and multimodal separately)
    serving_results = [r for r in successful_results if r.get('benchmark_type') == 'serving']
    serving_mm_results = [r for r in successful_results if r.get('benchmark_type') == 'serving_multimodal']
    throughput_results = [r for r in successful_results if r.get('benchmark_type') == 'throughput']
    throughput_mm_results = [r for r in successful_results if r.get('benchmark_type') == 'throughput_multimodal']
    # Include BOTH text and multimodal stress (the table prints the mode per row);
    # keying only on 'stress_test' silently dropped every MM stress result.
    stress_results = [r for r in successful_results if r.get('benchmark_type') in ('stress_test', 'stress_test_multimodal')]
    quality_results = [r for r in successful_results if r.get('benchmark_type') == 'quality']
    golden_results = [r for r in successful_results
                      if r.get('benchmark_type') == 'golden']
    serving_failures = [r for r in failed_results if r.get('benchmark_type') == 'serving']
    serving_mm_failures = [r for r in failed_results if r.get('benchmark_type') == 'serving_multimodal']
    throughput_failures = [r for r in failed_results if r.get('benchmark_type') in ('throughput', 'throughput_multimodal')]
    stress_failures = [r for r in failed_results if r.get('benchmark_type') in ('stress_test', 'stress_test_multimodal')]
    quality_failures = [r for r in failed_results if r.get('benchmark_type') == 'quality']

    # -- PERSONAL-DEVICE PEAK (--personal): single-stream tok/s + TTFT headline --
    if getattr(config, "personal_mode", False) and (serving_results or serving_mm_results):
        def _md(r, m):
            # Cover the pooled keys AND the legacy/remote aggregate's _mean-suffixed
            # keys (remote goes through aggregate_benchmark_results -> mean_{m}_ms_mean).
            for k in (f'{m}_p50_ms', f'median_{m}_ms', f'mean_{m}_ms',
                      f'median_{m}_ms_mean', f'mean_{m}_ms_mean'):
                v = r.get(k)
                if isinstance(v, (int, float)):
                    return v
            return None
        print("=" * 80)
        print("PERSONAL-DEVICE PERFORMANCE".center(80))
        print("=" * 80)
        for r, mode in ([(x, 'text') for x in serving_results]
                        + [(x, 'image+text') for x in serving_mm_results]):
            model = r.get('model_short', r.get('model', 'N/A'))
            served = r.get('format', 'N/A')
            tpot = _md(r, 'tpot'); ttft = _md(r, 'ttft')
            toks = (1000.0 / tpot) if isinstance(tpot, (int, float)) and tpot > 0 else None
            # Support count = requests that actually BACK the percentiles, not the
            # offered prompt count (they differ when requests are empty/error).
            n = (r.get('completed_requests') or r.get('pooled_sample_count')
                 or r.get('num_prompts'))
            emptyc = int(r.get('empty_requests') or 0)
            errc = int(r.get('failed_requests') or 0)
            offered = int(r.get('offered_requests') or 0) or ((n or 0) + emptyc + errc)
            realized = r.get('output_length_realized')
            cv = next((r.get(k) for k in ('mean_tpot_ms_cv_percent',
                       'median_tpot_ms_cv_percent', 'tpot_cv_percent')
                       if isinstance(r.get(k), (int, float))), None)
            print(f"  Model: {model}   |   served via: {served}   |   {mode}, single stream (batch=1)")
            print("  " + "-" * 66)
            print(f"    Decode speed         {(f'~{toks:.1f} tok/s' if toks else 'n/a'):<16} how fast the reply streams")
            print(f"    Time to first token  {(f'~{ttft/1000:.1f} s' if isinstance(ttft,(int,float)) else 'n/a'):<16} wait before text appears")
            print(f"    Per-token latency    {(f'~{tpot:.0f} ms' if isinstance(tpot,(int,float)) else 'n/a'):<16} (median TPOT)")
            # Reliability: how many offered requests produced a usable reply, vs
            # empty (model returned 0 tokens) vs error (transport/HTTP failure).
            rel = (f"{n}/{offered} ok" if offered else f"{n} ok")
            if emptyc:
                rel += f", {emptyc} empty"
            if errc:
                rel += f", {errc} error"
            _rl = (f"median reply {int(realized)} tok"
                   if isinstance(realized, (int, float)) else "")
            print(f"    Reliability          {rel:<16} {_rl}")
            warn = []
            if emptyc:
                warn.append(f"{emptyc} empty (model returned nothing)")
            if errc:
                reasons = r.get('error_reasons') or {}
                warn.append(f"{errc} errored" + (f" {reasons}" if reasons else ""))
            if r.get('low_confidence') or (isinstance(n, (int, float)) and n < 30):
                warn.append(f"n={n}")
            if isinstance(cv, (int, float)) and cv > 20:
                warn.append(f"TPOT CV {cv:.0f}%")
            if warn:
                print(f"    WARNING: indicative only ({', '.join(warn)})")
        print("  " + "-" * 66)
        print("  Median single-stream (batch=1) on THIS device - a rough LOCAL figure,")
        print("  NOT comparable to datacenter/GPU runs. If the backend ignores ignore_eos")
        print("  (e.g. Ollama), replies stop at natural EOS, so tok/s reflects natural-")
        print("  length outputs, not a fixed decode length.")
        print("=" * 80 + "\n")

    # -- SERVING - single-stream latency (batch=1), text + multimodal together --
    def _p50(r, m):
        # Prefer the true median across every path's key spelling; include the
        # legacy/remote `_mean`-suffixed median BEFORE the mean, so a remote/
        # legacy row never shows its MEAN under a "p50" header (mirror `_md`).
        for k in (f'{m}_p50_ms', f'median_{m}_ms', f'median_{m}_ms_mean',
                  f'mean_{m}_ms', f'mean_{m}_ms_mean'):
            v = r.get(k)
            if isinstance(v, (int, float)):
                return v
        return 'N/A'
    serv_all = ([(x, 'text') for x in serving_results]
                + [(x, 'MM') for x in serving_mm_results])
    serv_fail = ([(x, 'text') for x in serving_failures]
                 + [(x, 'MM') for x in serving_mm_failures])
    if serv_all or serv_fail:
        print("SERVING - single-stream latency, batch=1 (one request at a time; latency IS the metric)")
        print("-" * 92)
        print(f"{'Model':<20} {'Campaign':<14} {'Mode':<5} {'Input':<8} {'TTFT p50':<10} {'TPOT p50':<10} {'ITL p50':<10}")
        print("-" * 92)
        def _fmt1(v): return f"{v:.1f}" if isinstance(v, (int, float)) else str(v)
        has_mm_serv = False
        for r, mode in serv_all:
            model = r.get('model_short', 'N/A')[:20]
            camp = str(r.get('campaign', '-'))[:14]
            inp = r.get('input_length')
            inp_s = (str(r['workload_dataset'])[:8] if r.get('workload_dataset')   # real prompts
                     else f"{int(inp)}" if isinstance(inp, (int, float)) else "-")
            if mode == 'MM':
                has_mm_serv = True
            # Repeatability: requests/s, or tokens/s for a real dataset (different prompts
            # in each iteration); older results carry only the requests/s verdict.
            ok = (r['repeatability_valid'] if 'repeatability_valid' in r
                  else r.get('request_throughput_repeatability_valid', True))
            ok_s = 'n/a' if ok is None else ('OK' if ok else 'FAIL')     # n/a: 1 iteration
            print(f"{model:<20} {camp:<14} {mode:<5} {inp_s:<8} {_fmt1(_p50(r,'ttft')):<10} "
                  f"{_fmt1(_p50(r,'tpot')):<10} {_fmt1(_p50(r,'itl')):<10} {ok_s}")
        for r, mode in serv_fail:
            model = r.get('model', 'N/A')[:20]; camp = str(r.get('campaign', '-'))[:14]
            print(f"{model:<20} {camp:<14} {mode:<5} {'-':<8} {'FAILED':<10} {'-':<10} {'-':<10} FAIL")
        print("  note: pooled across iterations, shown as P50 (median). batch=1 = single-stream,")
        print("        so P99~P50 (no queueing) - use --batch-sizes for tail-under-load latency.")
        print("        TPOT = per token (server token count); ITL = gap between streamed chunks")
        print("        (a chunk can carry several tokens).")
        _kinds = sorted({str(r.get('tpot_kind')) for r, _m in serv_all
                         if r.get('tpot_kind') not in (None, 'per_token')})
        if _kinds:
            print(f"        TPOT fell back to the chunk mean where the server sent no usage: {', '.join(_kinds)}.")
        _short = [r for r, _m in serv_all
                  if (r.get('iterations_ok') or 0) < (r.get('iterations_requested') or 0)]
        for r in _short:
            print(f"        {r.get('model_short', '?')}: {r.get('iterations_ok')}/"
                  f"{r.get('iterations_requested')} iterations succeeded (pooled over those only).")
        if has_mm_serv:
            _mm = next((r for r, m in serv_all if m == 'MM'), {})
            _ni = int(_mm.get('mm_images') or 1)
            _act = _mm.get('input_length_actual')
            _actnote = (f"; server actually prefilled ~{int(_act)} tok"
                        if isinstance(_act, (int, float)) and _act > 0 else "")
            _pit = _mm.get('per_image_tokens')
            _pit_s = (f"{int(_pit)} tok/img, {_mm.get('per_image_tokens_source') or 'recorded'}"
                      if isinstance(_pit, (int, float)) else "per-image tokens unknown")
            print(f"  note: MM 'Input' = full campaign text + {_ni} image(s) "
                  f"({_pit_s}){_actnote} ON TOP,")
            print("        so MM Input >= text Input and MM TTFT >= text at the same campaign")
            print("        (identical text prefill + extra vision-tower work) - directly comparable.")
        _rel_bits = []
        for r, mode in serv_all:
            e = int(r.get('empty_requests') or 0); f = int(r.get('failed_requests') or 0)
            if e or f:
                seg = f"{mode} {r.get('completed_requests')}/{r.get('offered_requests')} ok"
                if e: seg += f", {e} empty"
                if f: seg += f", {f} error"
                _rel_bits.append(seg)
        if _rel_bits:
            print("  reliability: " + "; ".join(_rel_bits)
                  + "  (empty = model returned 0 tokens on a nonsense prompt, NOT an error)")
        print()

    # -- THROUGHPUT - offline peak (max batch, latency ignored), text + MM --
    thr_all = ([(x, 'text') for x in throughput_results]
               + [(x, 'MM') for x in throughput_mm_results])
    thr_fail = [(x, 'text') for x in throughput_failures]
    if thr_all or thr_fail:
        print("THROUGHPUT - offline peak: max batching (<= max_num_seqs, KV-bound), LATENCY IGNORED")
        print("-" * 92)
        print(f"{'Model':<20} {'Campaign':<14} {'Mode':<5} {'In/Out':<12} {'out tok/s':<11} {'CV%':<7}")
        print("-" * 92)
        has_mm = False
        for r, mode in thr_all:
            model = r.get('model_short', 'N/A')[:20]
            camp = str(r.get('campaign', '-'))[:14]
            in_out = f"{r.get('input_length','?')}/{r.get('output_length','?')}"
            tp = r.get('output_tokens_per_second_mean', r.get('output_tokens_per_second', 'N/A'))
            cv = r.get('output_tokens_per_second_cv_percent', 'N/A')
            tp_s = f"{tp:.0f}" if isinstance(tp, (int, float)) else str(tp)
            cv_s = f"{cv:.1f}%" if isinstance(cv, (int, float)) else ("n/a" if cv is None else str(cv))
            if mode == 'MM':
                has_mm = True
            print(f"{model:<20} {camp:<14} {mode:<5} {in_out:<12} {tp_s:<11} {cv_s:<7} OK")
        for r, mode in thr_fail:
            model = r.get('model', 'N/A')[:20]; camp = str(r.get('campaign', '-'))[:14]
            in_out = f"{r.get('input_length','?')}/{r.get('output_length','?')}"
            print(f"{model:<20} {camp:<14} {mode:<5} {in_out:<12} {'FAILED':<11} {'-':<7} FAIL")
        if has_mm:
            _mmt = next((r for r, m in thr_all if m == 'MM'), {})
            _nit = int(_mmt.get('mm_images') or 4)
            _pit = _mmt.get('per_image_tokens')
            _pit_s = (f"{int(_pit)} tok/img, {_mmt.get('per_image_tokens_source') or 'recorded'}"
                      if isinstance(_pit, (int, float)) else "per-image tokens unknown")
            print(f"  note: MM rows = full campaign text + {_nit} image(s) ON TOP ({_pit_s}).")
            print("        'In' = text + image tokens, so it exceeds the text row's In.")
            print("        MM does the SAME text prefill as text plus the images, so MM out tok/s")
            print("        <= text - directly comparable (cost of adding images).")
        print()

    # (Multimodal serving + throughput are now merged into the unified SERVING
    # and THROUGHPUT tables above via the Mode column.)

    # Print stress test results - open-loop capacity-under-SLO (goodput) format.
    if stress_results or stress_failures:
        print("STRESS TEST BENCHMARKS (open-loop capacity under SLO)")
        print("=" * 84)

        for r in stress_results:
            model = r.get('model', r.get('model_short', 'N/A'))
            fmt = r.get('format', 'N/A')
            mode = 'Multimodal' if r.get('multimodal', False) else 'Text'
            camp = r.get('campaign')
            ttft_thr = r.get('ttft_threshold_ms', 5000)
            itl_thr = r.get('itl_threshold_ms', r.get('tpot_threshold_ms', 200))
            reps = r.get('reps', 1)

            hdr = f"Model: {model} ({fmt}) - {mode}"
            if camp:
                hdr += f" - campaign: {camp}"
            print(hdr)
            print(f"SLO: P99 TTFT <= {ttft_thr}ms  AND  P99 ITL <= {itl_thr}ms "
                  f"| {reps} reps | {r.get('num_client_procs', '?')} client procs")
            print("-" * 84)

            # Per-rate sweep frontier (from rep 0): arrival -> achieved, tail latencies.
            sweep = r.get('sweep_points', [])
            if sweep:
                print(f"{'Arrival':>9} {'Achieved':>9} {'P99 TTFT':>10} {'P99 ITL':>14} "
                      f"{'%SLO':>5} {'Status':>16}")
                used_server_itl = used_client_itl = used_chunk_itl = used_tpot_itl = used_step_itl = False
                for pt in sweep:
                    arr = pt.get('arrival_qps', pt.get('request_rate_qps', 0))
                    ach = pt.get('achieved_qps', 0)
                    pttft = pt.get('p99_ttft_ms', 0)
                    # Show the ITL the knee actually GATED on: the server's own P99
                    # ITL (vllm /metrics) when available, else the client-measured
                    # ITL, else per-request TPOT. Showing the client ITL next to a
                    # server-ITL-gated PASS reads as self-contradictory (e.g. 1181ms
                    # ITL with a PASS under a 200ms SLO).
                    if str(pt.get('itl_slo_source', '')) == 'server' and pt.get('p99_itl_server_bucket'):
                        pitl = str(pt.get('p99_itl_server_bucket'))
                        used_server_itl = True
                    elif str(pt.get('itl_slo_source', '')).startswith('server (per-step') \
                            and pt.get('p99_itl_server_bucket'):
                        pitl = f"{pt.get('p99_itl_server_bucket')} st"
                        used_step_itl = True
                    elif str(pt.get('itl_slo_source', '')).startswith('client_per_token') \
                            and pt.get('p99_itl_per_token_ms'):
                        pitl = f"{pt.get('p99_itl_per_token_ms'):.0f}ms c"
                        used_client_itl = True
                    elif pt.get('p99_itl_ms'):
                        pitl = f"{pt.get('p99_itl_ms'):.0f}ms ch"
                        used_chunk_itl = True
                    else:
                        pitl = f"{(pt.get('p99_tpot_ms') or 0):.0f}ms t"
                        used_tpot_itl = True
                    slo = pt.get('slo_attainment', 0) * 100
                    _v = pt.get('verdict') or ("pass" if pt.get('passed') else "fail")
                    st = {"pass": "PASS", "inconclusive": "INCONCLUSIVE"}.get(_v, "fail")
                    if pt.get('client_bound'):
                        st += " CLIENT-BOUND"
                    print(f"{arr:>9.3g} {ach:>9.3g} {pttft:>9.0f}ms {pitl:>14} "
                          f"{slo:>4.0f}% {st:>16}")
                print("-" * 84)
                print("%SLO = share of the point's requests within the TTFT SLO with a mean "
                      "per-token time within the ITL SLO; a point also fails on the P99 gates, a "
                      "growing queue or a backlog.")
                if any(pt.get('verdict') == 'inconclusive' for pt in sweep):
                    print("INCONCLUSIVE = the point could not be judged (short window or cohort, "
                          "no data, or a client that fell behind); it is never a pass. Its flags "
                          "are in the JSON.")
                if used_server_itl:
                    print("P99 ITL (a,b] = the server histogram bucket (vllm /metrics, one gap per "
                          "engine step) holding the P99;\nthe gate reads the bucket counts, no "
                          "interpolation.")
                if used_step_itl:
                    print("P99 ITL '(a,b] st' = server bucket per engine step, judged against the SLO "
                          "x tokens per step\n(speculative decoding, no tokenizer for a per-token "
                          "client value).")
                if used_client_itl:
                    print("P99 ITL '<n>ms c' = client P99 per-token gap (the server bucket straddled "
                          "the SLO, no /metrics, or speculative decoding).")
                if used_chunk_itl:
                    print("P99 ITL '<n>ms ch' = client P99 gap between streamed chunks (no tokenizer "
                          "to count tokens per chunk; a chunk can carry several tokens).")
                if used_tpot_itl:
                    print("P99 ITL '<n>ms t' = P99 of per-request TPOT (per token from the server "
                          "count; chunk mean when the server sent no usage).")

            if r.get('itl_disagree_points'):
                print(f"Note: at {r['itl_disagree_points']} point(s) the client per-token ITL and the "
                      f"server ITL differ by more than one histogram bucket.")
            if r.get('spec_decode_points'):
                print("Note: speculative decoding was active; the server ITL counts one gap per "
                      "engine step, so the client per-token gap decided the ITL SLO (or, with no "
                      "tokenizer, the server histogram against SLO x tokens per step).")
            pc = r.get('payload_check')
            if pc is None and 'payload_check' in r:
                if r.get('workload_dataset') in ('sharegpt', 'custom', 'hf'):
                    print("Payload check: not judged for real-dataset prompts (they may share "
                          "prefixes); synthetic workloads are checked.")
                else:
                    print("Payload check: not observable (no /metrics or no calibrated overhead).")
            elif pc:
                if pc.get('points_cache_checked'):
                    bad = pc.get('points_suspect') or []
                    cache_txt = (f"cache reuse suspected at {bad} req/s" if bad else
                                 f"no cache reuse in {pc['points_cache_checked']} points")
                else:
                    cache_txt = "cache reuse not observable (no /metrics)"
                len_txt = (f"{pc.get('len_mismatch_total', 0)} prompt-length mismatches"
                           if pc.get('points_len_checked') else "prompt lengths not checked")
                print(f"Payload check: {cache_txt}; {len_txt}.")

            # Headline capacity: the median repeat's limit, CERTIFIED by percentile-sized
            # probes or labelled UNCERTIFIED, with every repeat's [pass, fail] bracket.
            max_qps = r.get('max_sustainable_qps', 0)
            n_users = r.get('little_law_n_users', 0)
            br = r.get('brackets') or {}
            cert = r.get('certification') or {}
            if max_qps and max_qps > 0:
                label = "CERTIFIED" if r.get('max_sustainable_qps_certified') else "UNCERTIFIED"
                pairs = ", ".join(f"[{b.get('pass')}, {b.get('fail')}]"
                                  for b in br.get('brackets', []))
                print(f"Sustainable load: {max_qps:.3g} req/s {label}  "
                      f"(repeat brackets {pairs}{'' if br.get('consistent', True) else ', NOT consistent'})")
                print(f"   = max arrival rate holding P99 TTFT <= {ttft_thr}ms AND P99 ITL <= {itl_thr}ms")
                if r.get('max_sustainable_qps_certified'):
                    res = cert.get('resolution_pct')
                    print(f"   certified with probes of >= {cert.get('n_per_probe')} cohort requests"
                          + (f", +-{res:g}% resolution" if res is not None else "")
                          + (f"; {cert['note']}" if cert.get('note') else ""))
                elif cert.get('needed_min'):
                    print(f"   not certified ({cert.get('reason') or 'budget'}); certifying needs ~"
                          f"{cert['needed_min']:.0f} min (--stress-certify-budget "
                          f"{int(cert['needed_min']) + 1})")
                elif cert.get('reason'):
                    print(f"   not certified: {cert['reason']}")
                if r.get('knee_saturation_note'):
                    print(f"   note: {r['knee_saturation_note']}")
                if n_users and n_users > 0:
                    e2e_s = r.get('mean_e2e_ms', 0) / 1000.0
                    print(f"   ~ {n_users:.0f} concurrent in-flight requests "
                          f"(Little's Law: {max_qps:.3g} req/s x {e2e_s:.1f}s mean latency at this load)")
            elif (r.get('certification') or {}).get('outcome') == 'refuted':
                c = r['certification']
                print(f"WARNING: no limit reported - {c.get('reason')}"
                      + (f"; certifying lower needs ~{c['needed_min']:.0f} min" if c.get('needed_min') else ""))
            elif r.get('below_measurable_floor'):
                print(f"WARNING: below the measurable floor ({r['below_measurable_floor']:.3g} req/s): "
                      f"no rate down to it met the SLOs.")
            elif r.get('preflight_no_stream'):
                print(f"WARNING: Stress skipped - the endpoint produced no output tokens for any "
                      f"single-stream probe, so the sweep cannot measure capacity.")
                print(f"   Check that the endpoint streams chat/completions for this model, "
                      f"or use --no-stress-test.")
            elif r.get('below_stress_floor'):
                if r.get('below_stress_floor_reason') == 'ITL':
                    tok = float(r.get('preflight_single_stream_per_token_ms', 0) or 0)
                    why = f"decodes at {tok:.0f}ms per token (> {itl_thr}ms ITL SLO)"
                    knob = "--itl-slo-ms"
                else:
                    ttft = float(r.get('preflight_single_stream_ttft_ms', 0) or 0)
                    why = f"takes {ttft:.0f}ms to first token (> {ttft_thr}ms SLO)"
                    knob = "--stress-threshold"
                print(f"WARNING: Below the stress floor - a single request already {why}, so "
                      f"no open-loop rate can meet the SLOs on this hardware.")
                print(f"   Use --no-stress-test, or raise {knob} to measure capacity at this "
                      f"latency.")
            else:
                why = r.get('no_rate_reason') or 'see the point flags'
                if why.startswith('request failures'):
                    print(f"WARNING: No sustainable rate found - too many requests failed even at "
                          f"the lowest rate tried: {why}.")
                else:
                    print(f"WARNING: No sustainable rate found - the lowest rate tried could not "
                          f"be judged: {why}.")
                print(f"   The per-point verdicts and flags are in the JSON (sweep_points).")
            print("=" * 84)
            print()
        
        # Print failures
        for r in stress_failures:
            model = r.get('model', 'N/A')
            fmt = r.get('format', 'N/A')
            mode = 'Multimodal' if r.get('multimodal', False) else 'Text'
            error = r.get('error', 'Unknown error')
            print(f"Model: {model} ({fmt}) - {mode} Mode")
            print("-" * 80)
            if r.get('failure_kind') == 'contaminated':
                print("Status: CONTAMINATED - other traffic on the server; no limit is reported")
                print(f"Reason: {error}")
                print(f"        {len(r.get('point_log') or [])} measurement(s) kept in the JSON; "
                      f"re-run with the server to yourself.")
                print()
                continue
            print("Status: FAILED")
            print(f"Error:  {error}")
            print("=" * 80)
            print()

    # Print quality benchmarks results table
    if quality_results or quality_failures:
        print("QUALITY BENCHMARKS (AGENTIC)")
        print("-" * 80)
        print(f"{'Model':<20} {'Format':<8} {'Commit':<12} {'Scenarios':<12} {'Pass Rate':<10}")
        print("-" * 80)
        
        for r in quality_results:
            model = r.get('model_short', 'N/A')[:20]
            fmt = r.get('format', 'N/A')
            commit = r.get('gemmaclaw_commit', 'N/A')[:7]
            passed = r.get('passed_scenarios', 0)
            total = r.get('total_scenarios', 0)
            scenarios_str = f"{passed}/{total}"
            pass_rate = r.get('pass_rate', 0.0)
            # null when the quality runner withheld it (pass_rate_withheld_reason; docs/quality.md)
            rate = "withheld" if pass_rate is None else f"{pass_rate:.1f}%"

            print(f"{model:<20} {fmt:<8} {commit:<12} {scenarios_str:<12} {rate}")
        
        # Print failures
        for r in quality_failures:
            model = r.get('model', 'N/A')[:20]
            fmt = r.get('format', 'N/A')
            print(f"{model:<20} {fmt:<8} {'-':<12} {'FAILED':<12} -")
        
        print()

    # Print evaluation benchmarks results table
    eval_results = [r for r in successful_results if r.get('benchmark_type') == 'eval']
    eval_failures = [r for r in failed_results if r.get('benchmark_type') == 'eval']
    if eval_results or eval_failures:
        from .runners.evals import BUILTIN_PILLARS
        eval_pillars = BUILTIN_PILLARS

        builtin_suite_keys = set()
        for _, keys in eval_pillars:
            builtin_suite_keys.update(keys)

        builtin_results = [r for r in eval_results if r.get('eval_name', '').lower() in builtin_suite_keys]
        builtin_fails = [r for r in eval_failures if r.get('eval_name', '').lower() in builtin_suite_keys]

        custom_results = [r for r in eval_results if r.get('eval_name', '').lower() not in builtin_suite_keys]
        custom_fails = [r for r in eval_failures if r.get('eval_name', '').lower() not in builtin_suite_keys]

        from .runners.eval_suites import CUSTOM_PILLARS

        table_width = 113
        label_width = 83
        grand_total_q = 0
        grand_correct_q = 0

        # -------------------------------------------------------------------------
        # SECTION 1: BUILT-IN EVALUATION BENCHMARKS (GBENCH STANDARD)
        # -------------------------------------------------------------------------
        if builtin_results or builtin_fails:
            print("BUILT-IN EVALUATION BENCHMARKS (GBENCH STANDARD)")
            print("=" * table_width)
            print(f"{'Model':<18} {'Format':<16} {'Eval Suite':<36} {'Thinking':<10} {'Questions':>9} {'Correct':>9} {'Accuracy':>10}")
            print("=" * table_width)

            builtin_tot = 0
            builtin_corr = 0

            for pillar_title, suite_keys in eval_pillars:
                p_res = [r for r in builtin_results if r.get('eval_name', '').lower() in suite_keys]
                p_fails = [r for r in builtin_fails if r.get('eval_name', '').lower() in suite_keys]
                if not p_res and not p_fails:
                    continue

                print(f"\n>> {pillar_title}")
                print("-" * table_width)
                p_tot = 0
                p_corr = 0

                for r in p_res:
                    model = r.get('model_short', r.get('model_name', 'N/A'))[:18]
                    fmt = r.get('format', 'N/A')[:16]
                    suite = r.get('eval_name', 'N/A').upper()[:36]
                    thinking = "Yes" if r.get('thinking', False) else "No"
                    total_q = r.get('total_questions', 0) or 0
                    # '-' when the headline is not correct/total: left out of the subtotal
                    correct = _correct_cell(r, total_q)
                    acc = r.get('accuracy', 0.0)
                    if correct != "-":
                        p_tot += total_q
                        p_corr += correct
                    print(f"{model:<18} {fmt:<16} {suite:<36} {thinking:<10} {_questions_cell(r, total_q):>9} {correct:>9} {_pct_cell(acc)}")
                    if _headline_metrics_text(r):
                        print(f"  - headline: {_headline_metrics_text(r)}")

                    if r.get("low_diversity"):
                        eff_n = r.get("effective_n", total_q)
                        print(f"  - WARNING: low diversity (effective n={eff_n} / {total_q} questions)")

                    cat_acc = r.get("category_accuracy", {})
                    if 1 < len(cat_acc) <= 25:
                        for cat, cstats in sorted(cat_acc.items()):
                            cat_label = f"  - {cat}"[:label_width]
                            ctot = cstats.get("total", 0)
                            ccorr, cacc = _category_cells(cstats)
                            print(f"{cat_label:<{label_width}} {ctot:>9} {ccorr:>9} {_pct_cell(cacc)}")

                for r in p_fails:
                    model = r.get('model_short', r.get('model_name', 'N/A'))[:18]
                    fmt = r.get('format', 'N/A')[:16]
                    suite = r.get('eval_name', 'N/A').upper()[:36]
                    thinking = "Yes" if r.get('thinking', False) else "No"
                    _print_eval_failure(r, model, fmt, suite, thinking)

                if p_tot > 0:
                    p_acc = (p_corr / p_tot * 100.0)
                    print(f"{'  - Pillar Subtotal':<{label_width}} {p_tot:>9} {p_corr:>9} {p_acc:>9.2f}%")
                    builtin_tot += p_tot
                    builtin_corr += p_corr

            if builtin_tot > 0:
                b_acc = (builtin_corr / builtin_tot * 100.0)
                print("-" * table_width)
                print(f"{'BUILT-IN EVALS SUBTOTAL':<{label_width}} {builtin_tot:>9} {builtin_corr:>9} {b_acc:>9.2f}%")
                print("  WARNING: Subtotal % is a micro-average of raw correct/total across suites with")
                print("    DIFFERENT metrics (multiple-choice, code pass@k, ANLS, judge-mean)")
                print("    - it is NOT a comparable score. Use the per-suite Accuracy column. Suites")
                print("    whose headline is not correct/total (e.g. HealthBench mean-rubric, DocVQA")
                print("    ANLS) print '-' as Correct and are left out of the subtotals.")
                print("=" * table_width)
                grand_total_q += builtin_tot
                grand_correct_q += builtin_corr
            print()

        # -------------------------------------------------------------------------
        # SECTION 2: CUSTOM / PLUGIN EVALUATION BENCHMARKS
        # -------------------------------------------------------------------------
        if custom_results or custom_fails:
            print("CUSTOM / PLUGIN EVALUATION BENCHMARKS")
            print("=" * table_width)
            print(f"{'Model':<18} {'Format':<16} {'Eval Suite':<36} {'Thinking':<10} {'Questions':>9} {'Correct':>9} {'Accuracy':>10}")
            print("=" * table_width)

            custom_pillar_map: Dict[str, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
            truly_uncat_res = []
            truly_uncat_fails = []

            for r in custom_results:
                ename = r.get('eval_name', '').lower()
                if ename in CUSTOM_PILLARS:
                    p_name = CUSTOM_PILLARS[ename]
                    custom_pillar_map.setdefault(p_name, ([], []))[0].append(r)
                else:
                    truly_uncat_res.append(r)

            for r in custom_fails:
                ename = r.get('eval_name', '').lower()
                if ename in CUSTOM_PILLARS:
                    p_name = CUSTOM_PILLARS[ename]
                    custom_pillar_map.setdefault(p_name, ([], []))[1].append(r)
                else:
                    truly_uncat_fails.append(r)

            custom_tot = 0
            custom_corr = 0

            for cp_title, (cp_res, cp_fails) in sorted(custom_pillar_map.items()):
                clean_cp_title = re.sub(r"^\d+\.\s*", "", cp_title).strip()
                print(f"\n>> [CUSTOM] {clean_cp_title.upper()}")
                print("-" * table_width)
                cp_tot = 0
                cp_corr = 0
                for r in cp_res:
                    model = r.get('model_short', r.get('model_name', 'N/A'))[:18]
                    fmt = r.get('format', 'N/A')[:16]
                    suite = r.get('eval_name', 'N/A').upper()[:36]
                    thinking = "Yes" if r.get('thinking', False) else "No"
                    total_q = r.get('total_questions', 0) or 0
                    # '-' when the headline is not correct/total: left out of the subtotal
                    correct = _correct_cell(r, total_q)
                    acc = r.get('accuracy', 0.0)
                    if correct != "-":
                        cp_tot += total_q
                        cp_corr += correct
                    print(f"{model:<18} {fmt:<16} {suite:<36} {thinking:<10} {_questions_cell(r, total_q):>9} {correct:>9} {_pct_cell(acc)}")
                    if _headline_metrics_text(r):
                        print(f"  - headline: {_headline_metrics_text(r)}")

                    if r.get("low_diversity"):
                        eff_n = r.get("effective_n", total_q)
                        print(f"  - WARNING: low diversity (effective n={eff_n} / {total_q} questions)")

                    cat_acc = r.get("category_accuracy", {})
                    if 1 < len(cat_acc) <= 25:
                        for cat, cstats in sorted(cat_acc.items()):
                            cat_label = f"  - {cat}"[:label_width]
                            ctot = cstats.get("total", 0)
                            ccorr, cacc = _category_cells(cstats)
                            print(f"{cat_label:<{label_width}} {ctot:>9} {ccorr:>9} {_pct_cell(cacc)}")

                for r in cp_fails:
                    model = r.get('model_short', r.get('model_name', 'N/A'))[:18]
                    fmt = r.get('format', 'N/A')[:16]
                    suite = r.get('eval_name', 'N/A').upper()[:36]
                    thinking = "Yes" if r.get('thinking', False) else "No"
                    _print_eval_failure(r, model, fmt, suite, thinking)

                if cp_tot > 0:
                    cp_acc = (cp_corr / cp_tot * 100.0)
                    print(f"{'  - Pillar Subtotal':<{label_width}} {cp_tot:>9} {cp_corr:>9} {cp_acc:>9.2f}%")
                    custom_tot += cp_tot
                    custom_corr += cp_corr

            if truly_uncat_res or truly_uncat_fails:
                print("\n>> [CUSTOM] OTHER EVALUATIONS")
                print("-" * table_width)
                uncat_tot = 0
                uncat_corr = 0
                for r in truly_uncat_res:
                    model = r.get('model_short', r.get('model_name', 'N/A'))[:18]
                    fmt = r.get('format', 'N/A')[:16]
                    suite = r.get('eval_name', 'N/A').upper()[:36]
                    thinking = "Yes" if r.get('thinking', False) else "No"
                    total_q = r.get('total_questions', 0) or 0
                    # '-' when the headline is not correct/total: left out of the subtotal
                    correct = _correct_cell(r, total_q)
                    acc = r.get('accuracy', 0.0)
                    if correct != "-":
                        uncat_tot += total_q
                        uncat_corr += correct
                    print(f"{model:<18} {fmt:<16} {suite:<36} {thinking:<10} {_questions_cell(r, total_q):>9} {correct:>9} {_pct_cell(acc)}")

                    if r.get("low_diversity"):
                        eff_n = r.get("effective_n", total_q)
                        print(f"  - WARNING: low diversity (effective n={eff_n} / {total_q} questions)")

                for r in truly_uncat_fails:
                    model = r.get('model_short', r.get('model_name', 'N/A'))[:18]
                    fmt = r.get('format', 'N/A')[:16]
                    suite = r.get('eval_name', 'N/A').upper()[:36]
                    thinking = "Yes" if r.get('thinking', False) else "No"
                    _print_eval_failure(r, model, fmt, suite, thinking)

                if uncat_tot > 0:
                    u_acc = (uncat_corr / uncat_tot * 100.0)
                    print(f"{'  - Pillar Subtotal':<{label_width}} {uncat_tot:>9} {uncat_corr:>9} {u_acc:>9.2f}%")
                    custom_tot += uncat_tot
                    custom_corr += uncat_corr

            if custom_tot > 0:
                c_acc = (custom_corr / custom_tot * 100.0)
                print("-" * table_width)
                print(f"{'CUSTOM PLUGINS SUBTOTAL':<{label_width}} {custom_tot:>9} {custom_corr:>9} {c_acc:>9.2f}%")
                print("=" * table_width)
                grand_total_q += custom_tot
                grand_correct_q += custom_corr
            print()

        # Print overall total if both built-in and custom were executed
        if (builtin_results or builtin_fails) and (custom_results or custom_fails) and grand_total_q > 0:
            grand_acc = (grand_correct_q / grand_total_q * 100.0)
            print("=" * table_width)
            print(f"{'OVERALL EVALS TOTAL':<{label_width}} {grand_total_q:>9} {grand_correct_q:>9} {grand_acc:>9.2f}%")
            print("  WARNING: micro-average of correct/total across EVERY suite (built-in + custom) and")
            print("    every metric - the least comparable number in this report. It is a coverage")
            print("    tally, NOT a benchmark score; compare models per-suite, never on this %.")
            print("=" * table_width)
            print()

        print()

        # Generate CSV summary report
        target_results_dir = config.log_manager.results_dir if getattr(config, "log_manager", None) else Path(getattr(config, "results_dir", "results"))
        csv_file = _save_eval_summary_csv(
            target_results_dir,
            eval_results,
            eval_failures,
            eval_pillars,
            CUSTOM_PILLARS,
        )

    # Print Golden Set results table
    if golden_results:
        print("GOLDEN SET (DETERMINISTIC SMOKE TEST)")
        print("-" * 80)
        print(f"{'Model':<20} {'Requested':<24} {'Cases':<10} {'Verdict':<10}")
        print("-" * 80)

        for r in golden_results:
            model = r.get('model_short', r.get('model', 'N/A'))[:20]
            requested = (r.get('requested_model') or '(endpoint default)')[:24]
            passed = r.get('passed_cases', 0)
            # Denominator is the cases APPLICABLE to this model's modalities, so
            # a case the model architecturally cannot run (e.g. audio on a
            # no-audio-tower model) does not count against it.
            applicable = r.get('applicable_tasks', r.get('total_tasks', 0))
            na = r.get('not_applicable_cases', 0)
            cases = f"{passed}/{applicable}"
            verdict = GOLDEN_VERDICT.get(r.get('status'), 'UNKNOWN')
            print(f"{model:<20} {requested:<24} {cases:<10} {verdict:<10}")

            rows = golden_category_breakdown(r.get('task_results', []))
            if len(rows) > 1:
                for row in rows:
                    cat_label = f"  - {row['category']}"[:45]
                    cat_cases = f"{row['passed']}/{row['total']}"
                    cat_verdict = GOLDEN_VERDICT.get(row['status'], 'UNKNOWN')
                    print(f"{cat_label:<45} {cat_cases:<10} "
                          f"{cat_verdict:<10}".rstrip())

            if na:
                na_ids = [t.get('task_id') for t in r.get('task_results', [])
                          if t.get('status') == 'not_applicable']
                print(f"  - {na} case(s) N/A (model lacks the required "
                      f"modality): {', '.join(na_ids)}")

        print()

        for r in golden_results:
            for err in r.get('harness_errors', []):
                print(f"  ERROR  {err}")
            for t in r.get('task_results', []):
                if t.get('status') == 'failed':
                    print(f"  FAIL   {t.get('task_id')}: {t.get('details')}")
        print()

    # Pillars that produced no result and have no table of their own above (a Golden Set pillar
    # that raised or whose server did not start): named, never silently dropped.
    other_failures = [r for r in failed_results if r.get('benchmark_type') == 'golden']
    if other_failures:
        print("PILLAR FAILURES (harness errors, not verdicts on the model)")
        print("-" * 80)
        for r in other_failures:
            print(f"  {r.get('benchmark_type')}  {r.get('model_short', r.get('model', 'N/A'))} "
                  f"({r.get('format', 'N/A')}): {str(r.get('error') or '')[:300]}")
        print()

    # Print file locations using LogManager
    print("="*80)
    print("RESULTS & LOGS")
    print("="*80)
    
    lm = getattr(config, "log_manager", None)
    if lm:
        # Save top-level aggregated summary.json for programmatic consumption
        summary_payload = {
            "timestamp": lm.timestamp,
            "results_dir": str(lm.results_dir.absolute()),
            "total_configurations": len(results),
            "successful_configurations": len(successful_results),
            "failed_configurations": len(failed_results),
            "models": [_summary_record(r) for r in successful_results],
            "failures": [_summary_record(r) for r in failed_results],
        }
        lm.save_summary(summary_payload)

        summary = lm.get_summary()
        print(f"\nResults Directory: {summary['results_dir']}")
        
        # CSV summary file
        csv_path = lm.results_dir / "eval_summary.csv"
        if csv_path.exists():
            print(f"\nSummary Report (CSV):")
            print(f"  - {csv_path.name} (ready for Google Sheets / Excel)")

        # List result files
        result_files = summary['result_files']
        if result_files:
            print(f"\nResult Files ({len(result_files)} total):")
            for f in result_files[:10]:
                print(f"  - {f}")
            if len(result_files) > 10:
                print(f"  ... and {len(result_files) - 10} more files")
        
        # List log files
        log_files = summary['log_files']
        if log_files:
            print(f"\nLog Files ({len(log_files)} total):")
            for f in log_files[:5]:
                print(f"  - {f}")
            if len(log_files) > 5:
                print(f"  ... and {len(log_files) - 5} more files")
    else:
        results_dir = getattr(config, "results_dir", "results")
        print(f"\nResults Directory: {results_dir}")

    # Statistical validation summary
    print("\n" + "="*80)
    print("STATISTICAL VALIDATION")
    print("="*80)
    
    total_configs = len(results)
    num_failed = len(failed_results)
    num_passed = len(successful_results)
    
    print(f"\nTotal Configurations: {total_configs}")
    print(f"Successful: {num_passed}")
    if num_failed > 0:
        print(f"Failed: {num_failed}")
    
    print("\n" + "="*80)
    if num_failed > 0:
        print(f"Benchmark suite complete! {num_passed} passed, {num_failed} failed.")
    else:
        print(f"Benchmark suite complete! All {total_configs} configurations passed.")

    # A configuration "passed" here only means its runner returned. A
    # Golden Set run that returned a FAIL or ERROR verdict is one of
    # those, so without this line the footer reads "All 1 configurations
    # passed" directly above a non-zero exit code.
    golden_bad = [r for r in golden_results
                  if r.get('status') in ('failed', 'error')]
    if golden_bad:
        verdicts = ", ".join(
            f"{r.get('model_short', r.get('model', 'N/A'))} "
            f"{GOLDEN_VERDICT.get(r.get('status'), 'UNKNOWN')}"
            for r in golden_bad
        )
        print(f"Golden Set did not pass: {verdicts}. See the table above.")
    print("="*80 + "\n")


def stage_models_to_gcs(models: list, gcs_destination: str) -> List[str]:
    """Upload each model's checkpoint to `gcs_destination`/<short_name>/ with `gcloud storage cp`.

    The checkpoint is the model's local directory when one exists (a --models value that is a
    directory), else a snapshot of its HuggingFace repo downloaded into a temporary directory
    (huggingface_hub.snapshot_download; HF_TOKEN for a gated repo). Each model is staged on its
    own: a failure is logged and the next model still runs. Returns the names of the models that
    failed (main() then exits with EXIT_HARNESS_ERROR); [] when every model was uploaded."""
    import subprocess
    import tempfile
    from gbench.core.models import ModelFormat

    logger.info(f"Starting GCS staging to: {gcs_destination}")
    failed: List[str] = []
    for model in models:
        logger.info(f"Processing model: {model.name}")
        dest_url = gcs_destination.rstrip("/") + "/" + model.short_name + "/"
        tmp = None
        try:
            local_path = None
            for candidate in (model.hf_model_id, model.get_model_path(ModelFormat.HF)):
                if candidate and os.path.isdir(str(candidate)):
                    local_path = str(candidate)
                    break
            if local_path is None:
                from huggingface_hub import snapshot_download
                tmp = tempfile.TemporaryDirectory(prefix="gbench-stage-")
                logger.info(f"Downloading {model.hf_model_id} from the Hugging Face Hub ...")
                local_path = snapshot_download(model.hf_model_id, local_dir=tmp.name)
            src_items = [os.path.join(local_path, item) for item in sorted(os.listdir(local_path))
                         if item != ".cache"]
            if not src_items:
                raise RuntimeError(f"no files in {local_path} to upload")
            cmd = ["gcloud", "storage", "cp", "-r"] + src_items + [dest_url]
            logger.info(f"Uploading {local_path} to {dest_url} ...")
            subprocess.run(cmd, check=True)
            logger.info(f"Successfully uploaded {model.name} to GCS.")
        except Exception as e:                                             # noqa: BLE001
            logger.error(f"Failed to stage {model.name} to GCS: {e}")
            failed.append(model.name)
        finally:
            if tmp is not None:
                tmp.cleanup()
    return failed


if __name__ == "__main__":
    sys.exit(main())

