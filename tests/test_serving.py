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

"""Serving runner regression tests."""

import pytest

from gbench.core.config import BenchmarkConfig
from gbench.runners.serving import ServingBenchmarkRunner


def test_no_think_fields_disables_reasoning_by_default():
    """The perf path (serving/stress) must run no-think by default so a reasoning
    prefix doesn't pollute the controlled fixed-length workload. It sends BOTH
    switches because different backends honor different ones (verified live):
      - reasoning_effort:'none'          -> honored by Ollama's /v1
      - chat_template_kwargs.enable_thinking:False -> honored by vLLM
    Backends that don't support a field ignore it (HTTP 200, no error)."""
    r = ServingBenchmarkRunner(BenchmarkConfig())
    assert r.no_think_fields() == {
        "chat_template_kwargs": {"enable_thinking": False},
        "reasoning_effort": "none",
    }


def test_no_think_fields_empty_under_thinking():
    """--thinking must NOT suppress reasoning (the model reasons on purpose)."""
    cfg = BenchmarkConfig()
    cfg.thinking = True
    assert ServingBenchmarkRunner(cfg).no_think_fields() == {}


@pytest.fixture(autouse=True)
def _no_port_probe(monkeypatch):
    """The launch is stopped at Popen: pick a fixed port instead of probing local ports."""
    from gbench.runners import serving
    monkeypatch.setattr(serving, "resolve_server_port", lambda environ=None: (18123, "auto"))


def _planned_server_cmd(tmp_path, context, explicit=None, campaign_window=None):
    """The `vllm serve` argv _start_server would launch (Popen is stubbed; nothing runs).

    ``explicit`` is the user's --max-model-len; ``campaign_window`` is what a --campaign writes
    into config.max_model_len for the perf server.
    """
    from unittest import mock
    from gbench.core.models import ModelConfig, ModelFormat

    class _LM:
        def get_server_log_path(self, *a):
            return str(tmp_path / "server.log")

    cfg = BenchmarkConfig()
    cfg.max_model_len = explicit or campaign_window
    cfg.max_model_len_explicit = explicit
    cfg.log_manager = _LM()
    runner = ServingBenchmarkRunner(cfg)
    captured = []

    def fake_popen(cmd, *a, **k):
        captured.append(cmd)
        raise RuntimeError("stop before launch")

    model = ModelConfig(short_name="m", hf_model_id="org/m")
    with mock.patch("subprocess.Popen", fake_popen), \
            mock.patch.object(runner, "_cleanup_server", lambda: None):
        runner._start_server(model, ModelFormat.HF, context=context)
    assert captured, "Popen was never reached"
    return captured[0]


def _served_len(cmd):
    return cmd[cmd.index("--max-model-len") + 1] if "--max-model-len" in cmd else None


def test_eval_and_quality_servers_use_the_native_context_window(tmp_path):
    """Regression: the eval server was launched with the perf default
    --max-model-len 4096, silently clamping a 65536-token thinking budget to <4k and
    breaking the long-context suites."""
    assert _served_len(_planned_server_cmd(tmp_path, "native")) is None
    assert _served_len(_planned_server_cmd(tmp_path, "native", explicit=32768)) == "32768"
    assert _served_len(_planned_server_cmd(tmp_path, "native", explicit=-1)) == "-1"


def test_a_campaign_window_does_not_cap_the_eval_server(tmp_path):
    """Review finding: --campaign writes its perf window into config.max_model_len, which the
    first fix mistook for the user's --max-model-len, so campaign+evals still served 4096."""
    assert _served_len(_planned_server_cmd(tmp_path, "native", campaign_window=13056)) is None
    assert _served_len(_planned_server_cmd(tmp_path, "perf", campaign_window=13056)) == "13056"


def test_perf_servers_keep_the_uniform_4096_context(tmp_path):
    assert _served_len(_planned_server_cmd(tmp_path, "perf")) == "4096"
    assert _served_len(_planned_server_cmd(tmp_path, "perf", explicit=13056)) == "13056"


def test_prefix_caching_is_disabled_only_for_perf_servers(tmp_path):
    assert "--no-enable-prefix-caching" in _planned_server_cmd(tmp_path, "perf")
    assert "--no-enable-prefix-caching" not in _planned_server_cmd(tmp_path, "native")


def test_cli_refuses_an_answer_budget_that_cannot_fit_before_anything_runs():
    """Review finding: the refusal came back as a bare {status: error} counted as a passed
    configuration (exit 0). It is now an argument error, raised before any pillar runs."""
    import pytest
    from gbench.cli import main
    with pytest.raises(SystemExit) as exc:
        main(["--models", "org/m", "--evals", "gsm8k", "--max-output-tokens", "16384",
              "--max-model-len", "8192"])
    assert exc.value.code != 0
    with pytest.raises(SystemExit):
        main(["--models", "org/m", "--evals", "gsm8k", "--max-output-tokens", "1024",
              "--max-model-len", "0"])


def test_the_scaffold_records_the_served_context_per_pillar():
    """Review finding: the scaffold still recorded 4096 for eval/quality servers launched at
    their native window, so metadata.json was wrong and scaffold_id did not move."""
    from gbench.core.scaffold import build_serving_environment
    cfg = BenchmarkConfig()
    cfg.remote_endpoint = None
    cfg.max_model_len = 13056            # a campaign's perf window
    cfg.max_model_len_explicit = None
    assert build_serving_environment(cfg).max_model_len == 13056
    assert build_serving_environment(cfg, context="native").max_model_len == "native"
    cfg.max_model_len_explicit = 32768
    assert build_serving_environment(cfg, context="native").max_model_len == 32768


def test_a_local_eval_server_forgets_the_previous_models_window(monkeypatch, tmp_path):
    """Review finding: the window cache is keyed by URL and a local server always has the same
    URL, so model 2 was clamped and MRCR-filtered with model 1's window."""
    from gbench.runners import evals as evals_mod
    from gbench.core.models import ModelConfig, ModelFormat

    url = f"http://127.0.0.1:{ServingBenchmarkRunner(BenchmarkConfig()).server_port}/v1"
    monkeypatch.setitem(evals_mod._SERVER_MAX_LEN, url, 131072)        # model 1's window
    seen = []
    monkeypatch.setattr(ServingBenchmarkRunner, "_start_server", lambda self, *a, **k: True)
    monkeypatch.setattr(ServingBenchmarkRunner, "_cleanup_server", lambda self: None)
    monkeypatch.setattr(evals_mod, "_resolve_remote_model_id",
                        lambda base, name: seen.append(url in evals_mod._SERVER_MAX_LEN) or name)
    monkeypatch.setattr(evals_mod, "_server_max_model_len", lambda base: 262144)
    monkeypatch.setattr(evals_mod.EvalsBenchmarkRunner, "_run_single_eval",
                        lambda self, **k: {"status": "success", "accuracy": 0.0})

    class _LM:
        results_dir = tmp_path
        def save_result(self, *a, **k):
            pass

    cfg = BenchmarkConfig()
    cfg.remote_endpoint = None
    cfg.evals = ["gsm8k"]
    cfg.eval_max_output_tokens = 1024
    cfg.log_manager = _LM()
    evals_mod.EvalsBenchmarkRunner(cfg).run(ModelConfig(short_name="m2", hf_model_id="org/m2"),
                                            ModelFormat.HF)
    assert seen == [False], "model 1's cached window must be dropped when a new server starts"


def test_minus_one_leaves_perf_and_campaign_windows_alone():
    """Review finding: --max-model-len -1 (vLLM auto-fit) was stored as the perf window too,
    dropping perf servers to 4096 and switching off campaign sizing."""
    from gbench.cli import create_parser, get_config_from_args
    args = create_parser().parse_args(["--models", "org/m", "--campaign", "agentic",
                                       "--max-model-len", "-1"])
    cfg = get_config_from_args(args)
    assert cfg.max_model_len_explicit == -1
    assert (cfg.max_model_len or 0) > 4096, "the campaign's perf window must survive -1"


def test_failed_eval_records_make_the_run_exit_non_zero():
    from gbench.cli import EXIT_HARNESS_ERROR
    import gbench.cli as cli
    src = open(cli.__file__, encoding="utf-8").read()
    assert 'r.get("failed") is True and r.get("benchmark_type") == "eval"' in src
    assert EXIT_HARNESS_ERROR != 0
