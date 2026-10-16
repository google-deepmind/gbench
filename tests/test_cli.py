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

"""Unit tests for the CLI module."""

import argparse
from unittest.mock import patch, MagicMock
import pytest
from gbench.cli import create_parser, main, _split_golden_tasks


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, None),
        ([], []),
        (["math_canonical"], ["math_canonical"]),
        (["a", "b"], ["a", "b"]),
        (["a,b"], ["a", "b"]),
        (["a, b", "c"], ["a", "b", "c"]),
        (["a,,b", " "], ["a", "b"]),
    ],
)
def test_split_golden_tasks(raw, expected):
    """Comma and space separated task lists both work.

    nargs="+" alone turns "a,b" into one token that matches no task, so
    the natural thing to type would fail with a harness error.
    """
    assert _split_golden_tasks(raw) == expected


def test_parser_stage_to_gcs():
    """Verify --stage-to-gcs argument is parsed correctly."""
    parser = create_parser()
    args = parser.parse_args(["--models", "gemma-4-E4B-it", "--stage-to-gcs", "gs://my-bucket/path"])

    assert args.models == ["gemma-4-E4B-it"]
    assert args.stage_to_gcs == "gs://my-bucket/path"


@patch("gbench.cli.stage_models_to_gcs")
@patch("gbench.cli.get_models_from_args")
@patch("gbench.cli.check_gpu_ready")
def test_main_stage_to_gcs_flow(mock_gpu_ready, mock_get_models, mock_stage):
    """Verify main() runs the staging flow and bypasses GPU checks."""
    mock_model = MagicMock()
    mock_get_models.return_value = [mock_model]

    # Run main with --stage-to-gcs
    retval = main(["--models", "some-model", "--stage-to-gcs", "gs://my-bucket/path"])

    assert retval == 0
    # check_gpu_ready should NOT be called
    mock_gpu_ready.assert_not_called()
    # stage_models_to_gcs should be called
    mock_stage.assert_called_once_with([mock_model], "gs://my-bucket/path")


def test_parser_max_output_tokens():
    """Verify --max-output-tokens is parsed and populated into config."""
    from gbench.cli import get_config_from_args
    parser = create_parser()
    
    # 1. Custom override
    args = parser.parse_args(["--models", "some-model", "--max-output-tokens", "4096"])
    cfg = get_config_from_args(args)
    assert cfg.eval_max_output_tokens == 4096

    # 2. None default
    args_default = parser.parse_args(["--models", "some-model"])
    cfg_default = get_config_from_args(args_default)
    assert cfg_default.eval_max_output_tokens is None


def test_builtin_subtotal_carries_a_comparability_disclaimer(capsys):
    """The micro-averaged subtotal mixes incomparable metrics (MCQ, code pass@k,
    ANLS, judge-mean, sub-step), so the report must say it is NOT a comparable score."""
    from gbench.cli import print_results_summary
    from gbench.core.config import BenchmarkConfig
    results = [
        {"benchmark_type": "eval", "eval_name": "gpqa_diamond", "model_name": "m",
         "model_short": "m", "format": "remote-endpoint", "status": "success",
         "total_questions": 100, "correct_answers": 50, "accuracy": 50.0, "thinking": False},
        {"benchmark_type": "eval", "eval_name": "scicode", "model_name": "m",
         "model_short": "m", "format": "remote-endpoint", "status": "success",
         "total_questions": 25, "correct_answers": 7, "accuracy": 28.0, "thinking": False},
    ]
    print_results_summary(results, BenchmarkConfig())
    out = capsys.readouterr().out
    assert "BUILT-IN EVALS SUBTOTAL" in out
    assert "NOT a comparable score" in out and "per-suite" in out


def test_every_eval_runner_accepts_the_threaded_cli_kwargs():
    """evals.py._run_single_eval calls runner_fn(**kwargs) with the full threaded param
    set. Every suite runner must accept all of them (via **kwargs or explicit params) or a
    --flag silently crashes that suite. Keep THREADED in sync with evals.py's kwargs dict."""
    import inspect
    from gbench.runners.eval_suites import SUITES, discover_and_register_plugins
    discover_and_register_plugins(None)
    THREADED = {"model_name", "base_url", "concurrency", "enable_thinking",
                "max_output_tokens", "eval_max_soft_tokens", "eval_n_shot",
                "eval_categories", "limit", "temperature", "attempt_count", "sandboxes"}
    offenders = []
    for name, fn in SUITES.items():
        params = inspect.signature(fn).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            continue  # **kwargs absorbs everything
        missing = [k for k in THREADED if k not in params]
        if missing:
            offenders.append((name, missing))
    assert not offenders, f"runners that would crash on threaded kwargs: {offenders}"


def test_max_output_tokens_required_for_eval_runs(capsys):
    """Eval runs must set --max-output-tokens explicitly (it bounds generation and moves
    scores); the command fails fast without it. Parser-level parsing is unaffected."""
    with pytest.raises(SystemExit):
        main(["--evals-only", "--evals", "scicode",
              "--remote-endpoint", "http://127.0.0.1:8000/v1",
              "--tokenizer", "google/gemma-4-E4B-it"])
    assert "max-output-tokens is required" in capsys.readouterr().err

