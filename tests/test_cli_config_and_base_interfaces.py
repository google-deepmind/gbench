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

"""The cross-cutting interfaces of the CLI, the config and base.

* model-agnostic serve settings (--tool-call-parser, --reasoning-parser, --vllm-serve-args) and
  the arguments gbench owns;
* the server port, GPU selection and parallel layout helpers;
* served-model matching and endpoint normalization;
* the shard identity helpers (current_shard, apply_shard_indexed, sample_key,
  finalize_shard_fields), the empty shard, and the trace identity across shards;
* the endpoint key, the uniform 400/422 rule, the run-shaping registry, the judge provenance;
* the server-parser residue rule;
* CLI validation, the tool-parser pre-flight, the dry-run plan, the exit codes, the shared native
  server of the quality / Golden Set / eval pillars, --stage-to-gcs and --merge.
No test starts a server, calls Gemini or reaches the network.
"""

import asyncio
import json
import os
import signal
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

from gbench import cli
from gbench.core import config as C
from gbench.core.models import ModelConfig
from gbench.runners.eval_suites import base

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _restore_presets():
    """get_config_from_args mutates the module-level preset configs: put them back after each
    test, so a --batch-sizes or --shard given here never reaches another test."""
    import copy
    saved = {id(c): (c, copy.copy(c.__dict__)) for c in (cli.DEFAULT_CONFIG, cli.QUICK_CONFIG)}
    env_shard = os.environ.get("GBENCH_SHARD")
    yield
    for cfg, state in saved.values():
        cfg.__dict__.clear()
        cfg.__dict__.update(state)
    if env_shard is None:
        os.environ.pop("GBENCH_SHARD", None)
    else:
        os.environ["GBENCH_SHARD"] = env_shard


# The README was split: these docs hold sections moved verbatim out of the README, so a
# negative check that used to scan the whole README scans README.md plus all of them.
_README_SPLIT_DOCS = (
    "docs/installation.md", "docs/environment-variables.md", "docs/performance.md",
    "docs/eval-suites.md", "docs/golden-set.md", "docs/remote-endpoints.md",
    "docs/results.md", "docs/deployment.md", "docs/cli-reference.md",
)
_README_AND_SPLIT_DOCS = ("README.md", *_README_SPLIT_DOCS)


def _read(rel):
    return (REPO / rel).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# serve settings
# --------------------------------------------------------------------------- #
def test_serve_args_are_shell_split_and_owned_arguments_are_refused():
    assert C.parse_vllm_serve_args("--trust-remote-code --chat-template 't x.jinja'") == [
        "--trust-remote-code", "--chat-template", "t x.jinja"]
    assert C.parse_vllm_serve_args(None) == [] and C.parse_vllm_serve_args("  ") == []
    for bad, flag in (("--port 9", "--port"), ("--port=9", "--port"),
                      ("--max_model_len 9", "--max-model-len"), ("-tp 2", "-tp"),
                      ("--max-model 9", "abbreviation of --max-model-len"),
                      ("--api-key k", "--api-key"), ("--config c.yaml", "--config"),
                      ("--enable-auto-tool-choice", "--tool-call-parser NAME"),
                      ("--reasoning-parser x", "--reasoning-parser NAME"),
                      ("--served-model-name m", "--served-model-name"),
                      ("--gpu-memory-utilization 0.5", "--gpu-memory-utilization")):
        with pytest.raises(ValueError) as exc:
            C.parse_vllm_serve_args(bad)
        assert flag in str(exc.value), bad
    with pytest.raises(ValueError, match="positional"):
        C.parse_vllm_serve_args("some/model --trust-remote-code")


def test_parsers_go_to_the_native_server_and_extras_to_every_server():
    cfg = C.BenchmarkConfig()
    cfg.tool_call_parser, cfg.reasoning_parser = "tp", "rp"
    cfg.vllm_serve_args = ["--trust-remote-code"]
    assert C.serve_args_for(cfg, "native") == [
        "--enable-auto-tool-choice", "--tool-call-parser", "tp", "--reasoning-parser", "rp",
        "--trust-remote-code"]
    assert C.serve_args_for(cfg, "perf") == ["--trust-remote-code"]
    assert C.serve_settings_record(cfg) == {"tool_call_parser": "tp", "reasoning_parser": "rp",
                                            "vllm_serve_args": ["--trust-remote-code"]}
    assert C.serve_args_for(C.BenchmarkConfig(), "native") == []


# --------------------------------------------------------------------------- #
# port, GPUs, layout, endpoints
# --------------------------------------------------------------------------- #
def _a_free_port_the_kernel_never_assigns():
    """A free port from 20000 up that lies outside the ephemeral range (ip_local_port_range; the
    Linux default when it cannot be read). The kernel hands a port of that range to every
    connect() and bind(port 0) on the host, so a port taken from it can go to another process
    between the test's close() and its check, and port_is_free's own probe connect() can pick it
    as its source port and connect to itself. Outside the range only a process that asks for the
    exact port gets it. Free means bindable on every interface, as port_is_free binds it.
    Model-server ports (8000-8010, 8100-8101) are far below."""
    try:
        with open("/proc/sys/net/ipv4/ip_local_port_range") as f:
            low, high = (int(x) for x in f.read().split())
    except (OSError, ValueError):
        low, high = 32768, 60999
    for port in range(20000, 65536):
        if low <= port <= high:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("", port))
            except OSError:
                continue
        return port
    pytest.fail(f"no free port from 20000 up outside the ephemeral range {low}-{high}")


def test_a_listening_port_is_busy_and_a_closed_one_is_free():
    port = _a_free_port_the_kernel_never_assigns()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", port))
    s.listen(1)
    try:
        assert C.port_is_free(port) is False
    finally:
        s.close()
    assert C.port_is_free(port) is True


def test_an_explicit_busy_port_is_an_error_and_the_automatic_search_skips_busy_ports(monkeypatch):
    busy = {8000, 8001}
    monkeypatch.setattr(C, "port_is_free", lambda p: p not in busy)
    assert C.resolve_server_port({"GBENCH_SERVER_PORT": "8000"}) == (
        None, "port 8000 is in use by another process; set GBENCH_SERVER_PORT to a free port")
    assert C.resolve_server_port({"GBENCH_SERVER_PORT": "8005"}) == (8005, "GBENCH_SERVER_PORT")
    assert C.resolve_server_port({}) == (8002, "auto")
    assert C.resolve_server_port({"GBENCH_SERVER_PORT": "x"})[0] is None


def test_endpoints_are_normalized_to_one_form():
    for raw, want in (("http://h:1", "http://h:1/v1"), ("http://h:1/", "http://h:1/v1"),
                      ("http://h:1/v1/", "http://h:1/v1"), ("http://h:1/v1", "http://h:1/v1")):
        assert C.normalize_endpoint(raw) == want
    assert C.normalize_endpoint(None) is None


def test_the_served_model_is_matched_never_taken_blindly():
    assert C.match_served_model(["org/m"], ["org/m"])[0] == "org/m"
    assert C.match_served_model(["ORG/M", "x"], ["org/m"])[0] == "ORG/M"
    # by the last path component: a local path or a bare name, never another organisation's
    # Hugging Face id of the same name
    assert C.match_served_model(["a", "/models/m"], ["google/m"])[0] == "/models/m"
    assert C.match_served_model(["a", "org/m"], ["google/m"])[0] is None
    served, why = C.match_served_model(["someone-elses/model"], ["org/m", "m"])
    assert served is None and "--served-model-id" in why
    assert C.match_served_model(["x", "y"], [], override="y") == ("y", "--served-model-id")
    assert C.match_served_model(["x"], [], override="y")[0] is None
    assert C.match_served_model(["only"], []) == ("only", "the only model the endpoint serves")
    assert C.match_served_model(["a", "b"], [])[0] is None


def _gpus(*specs):
    return [C.GpuInfo(i, f"GPU-{i}", 80.0, free, tuple(pids)) for i, free, pids in specs]


def test_gpu_selection_takes_idle_gpus_and_refuses_busy_ones():
    inv = _gpus((0, 4.0, ["11"]), (1, 79.0, []), (2, 79.0, []), (3, 10.0, []))
    chosen, msg = C.select_gpus(2, inventory=inv)
    assert [g.uuid for g in chosen] == ["GPU-1", "GPU-2"] and "auto" in msg
    chosen, msg = C.select_gpus(3, inventory=inv)
    assert chosen is None and "GPU 0 busy" in msg and "GPU 3 busy" in msg
    chosen, msg = C.select_gpus(1, ["0"], inventory=inv)
    assert chosen is None and "PIDs 11" in msg
    chosen, msg = C.select_gpus(1, ["GPU-2"], inventory=inv)
    assert [g.index for g in chosen] == [2]
    assert C.select_gpus(2, ["1"], inventory=inv)[0] is None          # count mismatch
    assert C.select_gpus(1, ["9"], inventory=inv)[0] is None          # not visible
    assert C.select_gpus(1, inventory=[])[0] is None                 # no GPU at all
    assert C.cuda_visible_devices_env(["GPU-1", "GPU-2"]) == {
        "CUDA_VISIBLE_DEVICES": "GPU-1,GPU-2", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    assert C.cuda_visible_devices_env(None) == {}


def test_gpu_selection_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(C, "gpu_inventory", lambda: None)
    assert C.select_gpus(1) == ([], "nvidia-smi unavailable: GPU selection left to vLLM")
    assert C.select_gpus(1, ["0"])[0] is None


def test_the_gpu_process_check_is_scoped_to_the_selected_gpus(monkeypatch):
    def fake_run(cmd, **kw):
        if "--query-compute-apps=pid,process_name,used_memory,gpu_uuid" in cmd:
            return SimpleNamespace(returncode=0, stdout="11, python, 500, GPU-0\n")
        return SimpleNamespace(returncode=0, stdout="70000, 81920\n")
    import subprocess
    monkeypatch.setattr(subprocess, "run", fake_run)
    ok, msg = C.check_gpu_ready(gpu_uuids=["GPU-0"])
    assert not ok and "selected GPU" in msg
    ok, msg = C.check_gpu_ready(gpu_uuids=["GPU-1"])
    assert ok


def test_the_parallel_layout_is_validated():
    assert C.parallel_layout(4, None) == (4, 1)
    assert C.parallel_layout(4, 2) == (2, 2)
    for n, tp in ((3, None), (2, 4), (6, 4), (8, 3)):
        with pytest.raises(ValueError):
            C.parallel_layout(n, tp)


# --------------------------------------------------------------------------- #
# CLI validation
# --------------------------------------------------------------------------- #
class _Stop(Exception):
    pass


def _cli(monkeypatch, *argv):
    """Run main() up to argument validation (setup_logging is the first thing after it) and
    return the parsed args."""
    captured = {}
    real = cli.create_parser

    def _parser():
        p = real()
        orig = p.parse_args
        p.parse_args = lambda a=None: captured.setdefault("args", orig(a))
        return p

    def _stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(cli, "create_parser", _parser)
    monkeypatch.setattr(cli, "setup_logging", _stop)
    with pytest.raises(_Stop):
        cli.main(list(argv))
    return captured["args"]


REMOTE = ("--remote-endpoint", "http://127.0.0.1:1/v1", "--tokenizer", "t",
          "--max-output-tokens", "1024")


@pytest.mark.parametrize("flag,value,name", [
    ("--sandboxes", "0", "--sandboxes"), ("--batch-sizes", "0", "--batch-sizes"),
    ("--attempt-count", "0", "--attempt-count"), ("--eval-limit", "0", "--eval-limit"),
    ("--eval-limit", "-3", "--eval-limit"), ("--temperature", "-0.5", "--temperature")])
def test_counts_below_one_and_negative_temperatures_are_refused(capsys, flag, value, name):
    with pytest.raises(SystemExit):
        cli.main([*REMOTE, "--evals", "gsm8k", flag, value])
    assert name in capsys.readouterr().err


def test_zero_max_output_tokens_is_refused(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--remote-endpoint", "http://127.0.0.1:1/v1", "--tokenizer", "t",
                  "--max-output-tokens", "0", "--evals", "gsm8k"])
    assert "--max-output-tokens" in capsys.readouterr().err


def test_eval_names_are_case_insensitive(monkeypatch):
    args = _cli(monkeypatch, *REMOTE, "--evals", "GPQA,Gsm8K")
    assert args.evals == ["gpqa", "gsm8k"]


def test_gguf_is_refused(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--models", "org/m", "--format", "gguf", "--preset", "quick"])
    assert "GGUF serving was removed" in capsys.readouterr().err


def test_a_local_tool_calling_suite_needs_the_tool_call_parser(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        cli.main(["--models", "org/m", "--evals", "bfcl", "--max-output-tokens", "1024",
                  "--num-gpus", "1"])
    err = capsys.readouterr().err
    assert "--tool-call-parser NAME is required" in err and "bfcl" in err
    args = _cli(monkeypatch, "--models", "org/m", "--evals", "bfcl", "--max-output-tokens",
                "1024", "--num-gpus", "1", "--tool-call-parser", "p")
    assert args.tool_call_parser == "p"
    # a suite without tools needs none; a remote run is not pre-flighted
    _cli(monkeypatch, "--models", "org/m", "--evals", "gsm8k", "--max-output-tokens", "1024",
         "--num-gpus", "1")
    _cli(monkeypatch, *REMOTE, "--evals", "bfcl")


def test_the_golden_set_and_quality_need_the_parser_on_a_local_run(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--models", "org/m", "--golden-only", "--num-gpus", "1"])
    assert "Golden Set's tool cases" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["--models", "org/m", "--quality-only", "--num-gpus", "1"])
    assert "quality pillar" in capsys.readouterr().err


def test_a_plugin_declares_tool_calling_with_an_attribute(monkeypatch):
    from gbench.runners import eval_suites
    def run_x(**kw):
        return {}
    run_x.requires_tool_calling = True
    monkeypatch.setitem(eval_suites.SUITES, "x_plugin", run_x)
    assert eval_suites.suite_requires_tool_calling("x_plugin")
    assert not eval_suites.suite_requires_tool_calling("gsm8k")
    assert eval_suites.TOOL_CALLING_SUITES <= set(eval_suites.SUITES)
    assert eval_suites.SANDBOXED_SUITES <= set(eval_suites.SUITES)


def test_server_flags_are_refused_with_a_remote_endpoint(capsys):
    for extra in (["--tool-call-parser", "p"], ["--reasoning-parser", "p"],
                  ["--vllm-serve-args=--trust-remote-code"], ["--gpus", "0"]):
        with pytest.raises(SystemExit):
            cli.main([*REMOTE, "--evals", "gsm8k", *extra])
        assert "gbench launches none" in capsys.readouterr().err


def test_owned_serve_arguments_are_refused_from_the_flag_and_the_environment(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        cli.main(["--models", "org/m", "--preset", "quick", "--num-gpus", "1",
                  "--vllm-serve-args=--port 9"])
    assert "--vllm-serve-args may not set --port" in capsys.readouterr().err
    monkeypatch.setenv("GBENCH_VLLM_SERVE_ARGS", "--max-model-len 9")
    with pytest.raises(SystemExit):
        cli.main(["--models", "org/m", "--preset", "quick", "--num-gpus", "1"])
    assert "--max-model-len" in capsys.readouterr().err


def test_a_malformed_shard_from_the_environment_is_an_error(monkeypatch, capsys):
    monkeypatch.setenv("GBENCH_SHARD", "3/2")
    with pytest.raises(SystemExit):
        cli.main([*REMOTE, "--evals", "gsm8k"])
    assert "GBENCH_SHARD" in capsys.readouterr().err


def test_an_invalid_parallel_layout_is_refused(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--models", "org/m", "--preset", "quick", "--num-gpus", "3"])
    assert "--tensor-parallel 3 is not supported" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["--models", "org/m", "--preset", "quick", "--num-gpus", "2", "--gpus", "0"])
    assert "--gpus names 1 GPU(s)" in capsys.readouterr().err


def test_the_config_carries_the_interfaces(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "2/4")
    parser = cli.create_parser()
    args = parser.parse_args(["--models", "org/m", "--num-gpus", "4", "--tensor-parallel", "2",
                              "--remote-endpoint", "http://h:9/", "--golden-model-id", "srv",
                              "--vllm-serve-args=--trust-remote-code", "--gpus", "1,2,3,4",
                              "--tool-call-parser", "tp"])
    cfg = cli.get_config_from_args(args)
    assert cfg.remote_endpoint == "http://h:9/v1"
    assert cfg.shard == "2/4" and os.environ["GBENCH_SHARD"] == "2/4"
    assert cfg.served_model_id == "srv" == cfg.golden_model_id
    assert (cfg.tensor_parallel_size, cfg.data_parallel_size) == (2, 2)
    assert cfg.vllm_serve_args == ["--trust-remote-code"] and cfg.tool_call_parser == "tp"
    assert cfg.gpus == ["1", "2", "3", "4"]
    assert cfg.eval_max_soft_tokens is None
    args = parser.parse_args(["--models", "org/m", "--eval-model-id", "e"])
    assert cli.get_config_from_args(args).served_model_id == "e"
    args = parser.parse_args(["--models", "org/m", "--batch-sizes", "16"])
    assert cli.get_config_from_args(args).eval_concurrency == 16
    args = parser.parse_args(["--models", "org/m", "--batch-sizes", "16", "--eval-concurrency", "4"])
    assert cli.get_config_from_args(args).eval_concurrency == 4


def test_the_help_names_no_bare_registry_model_and_no_fixed_soft_token_choices():
    text = cli.create_parser().format_help()
    assert "registry names" not in text and "gemma-4-E4B-it gemma-4-31B-it" not in text
    assert "--category" not in text and "--priority" not in text
    assert "Default: the model's own processor setting" in " ".join(text.split())
    assert "[70, 140, 280, 560, 1120]" not in text and "{70,140,280,560,1120}" not in text


# --------------------------------------------------------------------------- #
# the run: dry-run plan, exit codes, shared native server
# --------------------------------------------------------------------------- #
def _model(name="org/m", params=4.0, mm=False):
    m = ModelConfig(short_name=name.rsplit("/", 1)[-1], hf_model_id=name, total_params_b=params)
    m.supports_multimodal = mm
    return m


class _FakeRunners:
    """Stand-ins for the runners: record calls and what config.local_endpoint was meanwhile."""

    def __init__(self, cfg_box, start_ok=True, golden_raises=False):
        self.calls = []
        self.box = cfg_box
        self.start_ok = start_ok
        self.golden_raises = golden_raises
        outer = self

        class Serving:
            def __init__(self, config):
                outer.box["config"] = config
                self.config = config
                self.server_port = 8123
                self.server_process = None

            def _start_server(self, model, fmt, context="perf", **kw):
                outer.calls.append(("start", context))
                return outer.start_ok

            def _cleanup_server(self):
                outer.calls.append(("cleanup",))

            def run_all(self, model, fmt, multimodal=False):
                outer.calls.append(("serving", multimodal))
                if multimodal:
                    raise RuntimeError("mm server did not start")
                return [{"ok": True}]

            def build_server_cmd(self, model, format, context="perf", port=None):
                return ["vllm", "serve", model.hf_model_id, "--port", str(port), "--ctx", context]

        class Throughput:
            def __init__(self, config):
                pass

            def run_all(self, model, fmt, multimodal=False):
                outer.calls.append(("throughput", multimodal))
                return []

        class Stress:
            def __init__(self, config, **kw):
                pass

            def run_all(self, model, fmt, include_text=True, include_multimodal=False):
                return []

        class Quality:
            def __init__(self, config):
                pass

            def run(self, model, fmt):
                return {"status": "success"}

        class Golden:
            dataset_dir = REPO / "gbench" / "golden_dataset"

            def __init__(self, config):
                self.config = config

            def run(self, model):
                outer.calls.append(("golden", self.config.local_endpoint))
                if outer.golden_raises:
                    raise RuntimeError("golden broke")
                return {"status": "passed", "task_results": []}

        class Evals:
            def __init__(self, config):
                self.config = config

            def run(self, model, fmt):
                outer.calls.append(("evals", self.config.local_endpoint))
                return [{"benchmark_type": "eval", "eval_name": "gsm8k", "status": "success",
                         "accuracy": 50.0, "total_questions": 2, "correct_answers": 1}]

        self.Serving, self.Throughput, self.Stress = Serving, Throughput, Stress
        self.Quality, self.Golden, self.Evals = Quality, Golden, Evals


def _patch_local_run(monkeypatch, fakes, model=None):
    import gbench.runners as runners
    import gbench.utils as utils
    monkeypatch.setattr(runners, "ServingBenchmarkRunner", fakes.Serving)
    monkeypatch.setattr(runners, "ThroughputBenchmarkRunner", fakes.Throughput)
    monkeypatch.setattr(runners, "StressTestRunner", fakes.Stress)
    monkeypatch.setattr(runners, "QualityBenchmarkRunner", fakes.Quality)
    monkeypatch.setattr(runners, "GoldenBenchmarkRunner", fakes.Golden)
    monkeypatch.setattr(runners, "EvalsBenchmarkRunner", fakes.Evals)
    monkeypatch.setattr(cli, "get_models_from_args", lambda args: [model or _model()])
    monkeypatch.setattr(cli, "validate_gpu_config", lambda *a, **k: (True, "Using 1 GPU(s)"))
    monkeypatch.setattr(cli, "weights_fit_advisory", lambda *a, **k: (True, "fits"))
    monkeypatch.setattr(cli, "select_gpus", lambda n, req=None: (
        [C.GpuInfo(0, "GPU-0", 80.0, 79.0)], "GPUs 0 (GPU-0) (auto)"))
    monkeypatch.setattr(cli, "check_gpu_ready", lambda **k: (True, "GPU ready"))
    monkeypatch.setattr(utils, "require_vllm_engine", lambda *a, **k: None)
    monkeypatch.setattr(cli, "print_results_summary", lambda results, config: None)
    monkeypatch.setattr(cli, "_redact_run_results", lambda config: None)


def test_golden_and_evals_share_one_native_server_per_model(monkeypatch, tmp_path):
    box = {}
    fakes = _FakeRunners(box)
    _patch_local_run(monkeypatch, fakes)
    rc = cli.main(["--models", "org/m", "--num-gpus", "1", "--golden", "--evals", "gsm8k",
                   "--max-output-tokens", "1024", "--tool-call-parser", "p",
                   "--results-dir", str(tmp_path)])
    native = [c for c in fakes.calls if c[0] == "start"]
    assert native == [("start", "native")]
    assert ("golden", "http://127.0.0.1:8123/v1") in fakes.calls
    assert ("evals", "http://127.0.0.1:8123/v1") in fakes.calls
    assert fakes.calls[-1] == ("cleanup",) and box["config"].local_endpoint is None
    assert rc == cli.EXIT_OK
    assert box["config"].selected_gpus == ["GPU-0"]


def test_a_native_server_that_does_not_start_fails_its_pillars_as_harness_errors(monkeypatch,
                                                                                 tmp_path):
    box = {}
    fakes = _FakeRunners(box, start_ok=False)
    _patch_local_run(monkeypatch, fakes)
    rc = cli.main(["--models", "org/m", "--num-gpus", "1", "--golden", "--evals", "gsm8k",
                   "--max-output-tokens", "1024", "--tool-call-parser", "p",
                   "--results-dir", str(tmp_path)])
    assert rc == cli.EXIT_HARNESS_ERROR
    assert not any(c[0] in ("golden", "evals") for c in fakes.calls)


def test_a_failing_pillar_is_recorded_and_the_next_pillar_still_runs(monkeypatch, tmp_path):
    box = {}
    fakes = _FakeRunners(box, golden_raises=True)
    _patch_local_run(monkeypatch, fakes, model=_model(mm=True))
    rc = cli.main(["--models", "org/m", "--num-gpus", "1", "--golden", "--evals", "gsm8k",
                   "--max-output-tokens", "1024", "--tool-call-parser", "p",
                   "--input-lengths", "128", "--output-lengths", "64", "--serving-only",
                   "--results-dir", str(tmp_path)])
    # the multimodal serving pass raised and the Golden Set raised: both recorded, evals still ran
    assert ("serving", True) in fakes.calls and any(c[0] == "evals" for c in fakes.calls)
    assert rc == cli.EXIT_HARNESS_ERROR


def test_the_exit_code_ranks_harness_errors_above_model_verdicts():
    assert cli.run_exit_code([{"benchmark_type": "serving_multimodal", "failed": True}]) == 2
    assert cli.run_exit_code([{"benchmark_type": "golden", "status": "failed"}]) == 1
    assert cli.run_exit_code([{"benchmark_type": "golden", "status": "failed"},
                              {"benchmark_type": "eval", "failed": True, "eval_name": "x"}]) == 2
    assert cli.run_exit_code([{"benchmark_type": "golden", "status": "passed"}]) == 0


def test_the_dry_run_prints_the_server_plan_and_validates_the_gpus(monkeypatch, tmp_path, capsys):
    box = {}
    fakes = _FakeRunners(box)
    _patch_local_run(monkeypatch, fakes)
    rc = cli.main(["--models", "org/m", "--num-gpus", "1", "--evals", "gsm8k",
                   "--max-output-tokens", "1024", "--dry-run", "--results-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0 and "DRY RUN: servers and endpoints" in out
    assert "vllm serve org/m --port" in out and "--ctx native" in out
    assert "base_url for quality / Golden Set / evals: http://127.0.0.1:" in out
    assert not any(c[0] == "start" for c in fakes.calls)
    monkeypatch.setattr(cli, "validate_gpu_config",
                        lambda *a, **k: (False, "Requested 2 GPUs but only 1 available"))
    rc = cli.main(["--models", "org/m", "--num-gpus", "2", "--evals", "gsm8k",
                   "--max-output-tokens", "1024", "--dry-run", "--results-dir", str(tmp_path)])
    assert rc == cli.EXIT_HARNESS_ERROR


def test_a_bare_model_name_is_refused_for_a_local_run(monkeypatch, tmp_path):
    fakes = _FakeRunners({})
    _patch_local_run(monkeypatch, fakes)
    monkeypatch.setattr(cli, "get_models_from_args", lambda args: pytest.fail("not reached"))
    assert cli.main(["--models", "gemma-x", "--num-gpus", "1", "--preset", "quick",
                     "--results-dir", str(tmp_path)]) == cli.EXIT_HARNESS_ERROR


def test_the_sandbox_guard_checks_only_sandbox_concurrency(monkeypatch, tmp_path):
    seen = []
    import gbench.utils.calibrate as cal
    monkeypatch.setattr(cal, "detect_hardware", lambda: {})
    monkeypatch.setattr(cal, "check_oversubscription",
                        lambda conc, hw, suites=None: (seen.append(conc) or False, "over"))

    def _stop(args):
        raise _Stop
    monkeypatch.setattr(cli, "get_models_from_args", _stop)
    with pytest.raises(_Stop):
        cli.main([*REMOTE, "--evals", "gpqa_diamond", "--batch-sizes", "32",
                  "--results-dir", str(tmp_path)])
    assert seen == []
    assert cli.main([*REMOTE, "--evals", "terminal_bench", "--batch-sizes", "32",
                     "--results-dir", str(tmp_path)]) == cli.EXIT_HARNESS_ERROR
    assert cli.main([*REMOTE, "--evals", "gpqa_diamond", "--sandboxes", "4",
                     "--results-dir", str(tmp_path)]) == cli.EXIT_HARNESS_ERROR
    assert seen == [32, 4]


def test_termination_signals_become_system_exit_and_handlers_are_restored():
    before = signal.getsignal(signal.SIGTERM)
    with cli._termination_handlers():
        with pytest.raises(SystemExit) as exc:
            os.kill(os.getpid(), signal.SIGTERM)
            import time
            time.sleep(1)
        assert exc.value.code == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) == before


def test_staging_a_model_that_cannot_be_fetched_fails_without_a_traceback(monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    model = _model("org/not-local")
    failed = cli.stage_models_to_gcs([model], "gs://b/p")
    assert failed == [model.name]


def test_provenance_pinning_needs_a_revision_or_a_checksum():
    assert cli.provenance_pinned({"revision": "a" * 40})
    assert cli.provenance_pinned({"revision": "1.2.0"})
    assert cli.provenance_pinned({"revision": "harbor==0.20.0"})
    assert cli.provenance_pinned({"revision": "0.1.35 (benchmark v1.2.2)"})
    # the registry's own predicate (canonical_sync.entry_pinned): a checksum is a full digest
    assert cli.provenance_pinned({"checksum": "sha256:" + "a" * 64})
    assert not cli.provenance_pinned({"checksum": "sha256:abc"})
    assert not cli.provenance_pinned({"revision": "pycocotools 2.0.11 (as verified)"})
    assert not cli.provenance_pinned({"revision": "2026-09-01"})
    assert not cli.provenance_pinned({})


# --------------------------------------------------------------------------- #
# shard identity
# --------------------------------------------------------------------------- #
def test_a_malformed_shard_is_an_error_and_a_valid_one_is_parsed(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "0/3")
    with pytest.raises(ValueError):
        base.current_shard()
    monkeypatch.setenv("GBENCH_SHARD", "2/3")
    shard = base.current_shard()
    assert (shard.index, shard.count, shard.label, shard.partial) == (2, 3, "2/3", True)
    assert base.shard_identity(shard) == {"shard": "2/3", "shard_index": 2, "shard_count": 3,
                                          "shard_applied": True}
    monkeypatch.delenv("GBENCH_SHARD")
    assert base.current_shard() is None and base.shard_identity(None) == {}


def test_indexed_shards_are_disjoint_and_union_to_the_full_list(monkeypatch):
    items = list("abcdefghij")
    seen = []
    for i in (1, 2, 3):
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/3")
        sel, label, pos = base.apply_shard_indexed(items, "x")
        assert label == f"{i}/3" and [items[p] for p in pos] == sel
        seen += pos
    assert sorted(seen) == list(range(10))


def test_sample_keys_are_the_items_own_ids_else_a_stable_hash():
    assert base.sample_key(([], "g", {"instance_id": "django-1"})) == "instance_id:django-1"
    assert base.sample_key({"task_id": 7}) == "task_id:7"
    a = base.sample_key(([{"role": "user", "content": "q"}], "g", {"category": "c"}))
    b = base.sample_key(([{"role": "user", "content": "q"}], "g", {"category": "c"}))
    c = base.sample_key(([{"role": "user", "content": "q2"}], "g", {"category": "c"}))
    assert a == b != c and a.startswith("sha256:")


def test_the_central_shard_stamp(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "1/2")
    r = base.finalize_shard_fields({"eval_name": "x", "shard": "1/2"})
    assert r["shard_applied"] and r["shard_index"] == 1 and r["shard_count"] == 2
    assert r["leaderboard_comparable"] is False and "sharded run (--shard 1/2)" in \
        r["leaderboard_comparable_reason"]
    r = base.finalize_shard_fields({"eval_name": "x", "leaderboard_comparable": True})
    assert r["shard_applied"] is False and r["shard_requested"] == "1/2"
    assert r["leaderboard_comparable"] is True
    monkeypatch.setenv("GBENCH_SHARD", "1/1")
    assert base.finalize_shard_fields({"eval_name": "x"}) == {"eval_name": "x"}


def _samples(n):
    return [([{"role": "user", "content": f"q{i}"}], f"a{i}", {"category": f"c{i % 2}"})
            for i in range(n)]


def _fake_send(**kw):
    text = kw["messages"][-1]["content"].replace("q", "a")

    async def go():
        return base.Reply(text, None, "stop", None, None, 3)
    return go()


def test_an_empty_shard_is_an_honest_empty_row(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "5/5")
    called = []

    async def scorer(traces):
        called.append(traces)
    res = base.run_eval_suite("unit", "m", "http://stub/v1", 1, _samples(3),
                              async_eval_fn=scorer)
    assert res["status"] == "success" and res["accuracy"] is None and not called
    assert "empty shard 5/5" in res["accuracy_withheld_reason"]
    assert res["shard"] == "5/5" and res["leaderboard_comparable"] is False
    monkeypatch.delenv("GBENCH_SHARD")
    res = base.run_eval_suite("unit", "m", "http://stub/v1", 1, [], eval_fn=lambda p, g: True)
    assert res["status"] == "error"


def test_traces_carry_their_global_index_and_key_across_shards(monkeypatch):
    monkeypatch.setattr(base, "_send_single_request", _fake_send)
    keys, idxs = [], []
    for i in (1, 2):
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/2")
        res = base.run_eval_suite("unit", "m", "http://stub/v1", 2, _samples(10),
                                  eval_fn=lambda p, g: p == g)
        assert res["shard"] == f"{i}/2" and res["shard_count"] == 2
        keys += [t["sample_key"] for t in res["sample_traces"]]
        idxs += [t["global_idx"] for t in res["sample_traces"]]
        assert res["accuracy"] == 100.0
    assert sorted(idxs) == list(range(10)) and len(set(keys)) == 10


def test_the_limit_is_drawn_from_the_shard(monkeypatch):
    monkeypatch.setattr(base, "_send_single_request", _fake_send)
    monkeypatch.setenv("GBENCH_SHARD", "1/2")
    res = base.run_eval_suite("unit", "m", "http://stub/v1", 2, _samples(10),
                              eval_fn=lambda p, g: True, limit=3)
    assert res["total_questions"] == 3
    assert all(t["global_idx"] % 2 == 0 for t in res["sample_traces"])


def test_concurrency_below_one_is_clamped(monkeypatch):
    monkeypatch.setattr(base, "_send_single_request", _fake_send)
    res = base.run_eval_suite("unit", "m", "http://stub/v1", 0, _samples(2),
                              eval_fn=lambda p, g: True)
    assert res["total_questions"] == 2


# --------------------------------------------------------------------------- #
# endpoint key, 400/422 rule, run-shaping knobs, judge provenance
# --------------------------------------------------------------------------- #
def test_the_endpoint_key_prefers_vllm_api_key_and_ignores_placeholders():
    assert base.endpoint_api_key({"VLLM_API_KEY": "k1", "OPENAI_API_KEY": "k2"}) == "k1"
    assert base.endpoint_api_key({"VLLM_API_KEY": "EMPTY", "OPENAI_API_KEY": "k2"}) == "k2"
    assert base.endpoint_api_key({"OPENAI_API_KEY": "dummy"}) is None
    assert base.endpoint_auth_headers({"VLLM_API_KEY": "k1"}) == {"Authorization": "Bearer k1"}
    assert base.endpoint_auth_headers({}) == {}


def test_base_sends_the_endpoint_key_on_its_requests(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "secret-key-123456")
    seen = {}
    real = base.aiohttp.ClientSession

    def Session(*a, **kw):
        seen.update(kw.get("headers") or {})
        return real(*a, **kw)
    monkeypatch.setattr(base.aiohttp, "ClientSession", Session)
    monkeypatch.setattr(base, "_send_single_request", _fake_send)
    base.run_eval_suite("unit", "m", "http://stub/v1", 1, _samples(1), eval_fn=lambda p, g: True)
    assert seen.get("Authorization") == "Bearer secret-key-123456"


def test_the_uniform_rejection_rule():
    r = base.attribute_endpoint_rejection(400, "maximum context length is 8192 tokens")
    assert r["kind"] == "over_context" and r["attribution"] == "harness_error"
    r = base.attribute_endpoint_rejection(422, "bad", upstream_outcome="task failed",
                                          upstream_reference="ARE")
    assert r["attribution"] == "model_outcome" and r["upstream"] == "ARE"
    assert not base.is_context_window_rejection(500, "maximum context length")
    for suite in ("gaia2", "toolbench", "quality", "mcp_bench", "swe_bench_live"):
        assert suite in base.ENDPOINT_REJECTION_ATTRIBUTION
    knobs = " ".join(_read("docs/evals/env-knobs.md").split())
    assert "follow one rule (`base.attribute_endpoint_rejection`" in knobs


def test_run_shaping_knobs_are_snapshotted_by_class(monkeypatch):
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_MODE", "oneshot")
    monkeypatch.setenv("GBENCH_LCB_VERSION", "release_v6")
    monkeypatch.setenv("GBENCH_GPQA_TEMPERATURE", "0.3")
    monkeypatch.setenv("GBENCH_REPETITION_PENALTY", "1.1")
    snap = base.run_shaping_env(["swe_bench_live", "lcb", "gpqa"])
    assert snap["prompt"] == {"GBENCH_SWE_BENCH_LIVE_MODE": "oneshot"}
    assert snap["dataset"] == {"GBENCH_LCB_VERSION": "release_v6"}
    assert snap["decode"] == {"GBENCH_GPQA_TEMPERATURE": "0.3", "GBENCH_REPETITION_PENALTY": "1.1"}
    assert "GBENCH_LCB_VERSION" not in json.dumps(base.run_shaping_env(["gpqa"]))


#: Removed knobs a suite names only to refuse them.
_REMOVED_KNOBS = {"GBENCH_TAU2_RETRIEVAL_CONFIG"}


def test_every_run_shaping_suite_knob_is_declared():
    import re
    pattern = re.compile(r"GBENCH_[A-Z0-9_]+_(?:MODE|SPLIT|CONFIG|VERSION|RELEASE|SUBSET|LANG[A-Z_]*"
                         r"|CATEGORIES|PROMPT|NUM_TRIALS|MAX_STEPS|SAMPLES|NUM_SAMPLES|WINDOW"
                         r"|ALL_DOMAINS|DOMAIN|TASKS|EDIT_FORMAT|SETTINGS|MODEL_CONFIG)\b")
    declared = set(base.RUN_SHAPING_GLOBAL_ENV)
    for knobs in base.RUN_SHAPING_SUITE_ENV.values():
        declared |= set(knobs)
    # a deprecated alias is snapshotted under its canonical knob (RUN_SHAPING_ALIASES)
    for aliases in base.RUN_SHAPING_ALIASES.values():
        declared |= set(aliases)
    skip = {"swe_bench_pro_agentic.py", "jetski_agent.py", "slashbin_agent.py", "base.py",
            "canonical_sync.py"}
    missing = set()
    for path in (REPO / "gbench" / "runners" / "eval_suites").glob("*.py"):
        if path.name in skip:
            continue
        for name in pattern.findall(path.read_text(encoding="utf-8")):
            if name not in declared and name not in _REMOVED_KNOBS:
                missing.add(f"{path.name}: {name}")
    assert not missing, sorted(missing)


def test_a_judge_cascade_holding_the_model_under_test_is_flagged(monkeypatch):
    for k in ("GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL"):
        monkeypatch.delenv(k, raising=False)
    prov = base.judge_provenance("gemini-3.7-flash")
    assert prov["self_judge"] is True and "gemini-3.7-flash" in prov["self_judge_models"]
    assert base.judge_provenance("org/other-model")["self_judge"] is False
    r = base.apply_self_judge({}, "gemini-3.7-flash")
    assert r["leaderboard_comparable"] is False
    assert base.SELF_JUDGE_REASON in r["leaderboard_comparable_reason"]


def test_run_eval_suite_stamps_the_cascade_and_flags_a_self_judged_gated_suite(monkeypatch):
    for k in ("GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL", "GEMINI_API_KEY", "GEMINI_API_KEYS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(base, "_send_single_request", _fake_send)
    res = base.run_eval_suite("unit_plain", "gemini-3.7-flash", "http://stub/v1", 1,
                              _samples(1), eval_fn=lambda p, g: True)
    assert res["gemini_cascade"] == list(base.GEMINI_CASCADE) and "self_judge" not in res
    with pytest.raises(Exception):
        base.gemini_required_skip("unit_judged", "gemini-3.7-flash")
    assert base._GEMINI_GATED.get("unit_judged") == "gemini-3.7-flash"
    res = base.run_eval_suite("unit_judged", "gemini-3.7-flash", "http://stub/v1", 1,
                              _samples(1), eval_fn=lambda p, g: True)
    assert res["self_judge"] is True and res["leaderboard_comparable"] is False
    # a bypass suite: the central stamp adds it after the suite returned
    out = base.finalize_central_stamps({"eval_name": "unit_judged",
                                        "leaderboard_comparable": True}, "gemini-3.7-flash")
    assert out["self_judge"] is True and out["leaderboard_comparable"] is False
    base._GEMINI_GATED.pop("unit_judged", None)


def test_a_tool_loop_turn_ceiling_is_recorded(monkeypatch):
    monkeypatch.setattr(base, "_send_single_request", _fake_send)
    monkeypatch.delenv("GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS", raising=False)
    base.set_run_knobs(max_output_tokens=65536)

    async def tool(name, args):
        return "{}"
    res = base.run_eval_suite("unit_loop", "m", "http://stub/v1", 1, _samples(1),
                              eval_fn=lambda p, g: True, tool_executor=tool)
    assert res["tool_loop_turn_ceiling"] == {"applied": base.TOOL_LOOP_MAX_OUTPUT_TOKENS,
                                             "requested": 65536}
    assert res["leaderboard_comparable"] is False


# --------------------------------------------------------------------------- #
# server-parser residue
# --------------------------------------------------------------------------- #
def _res(**kw):
    kw.setdefault("tool_calls", None)
    return base.server_parser_residue(**kw)


def test_a_call_deleted_after_no_think_content_is_recovered():
    gen = 'Let me look that up.<|tool_call>call:search{query:<|"|>weather<|"|>}<tool_call|>'
    r = _res(content="Let me look that up.", reasoning=None, thinking=False, generation_text=gen)
    assert r.event == base.SERVER_PARSER_DROPPED_TEXT and r.text == gen


def test_a_call_deleted_after_the_reasoning_of_a_thinking_reply_is_recovered():
    gen = "<|channel>thought\nI will search.<channel|><|tool_call>call:search{q:1}<tool_call|>"
    r = _res(content=None, reasoning="I will search.", thinking=True, generation_text=gen)
    assert r.event == base.SERVER_PARSER_DROPPED_TEXT
    assert r.text == "<channel|><|tool_call>call:search{q:1}<tool_call|>"


def test_ordinary_replies_have_no_residue():
    cases = [
        ("<|channel>thought\nR<channel|>The answer is 4", "The answer is 4", "R", True),
        ("<think>\nR\n</think>\n\nThe answer is 4", "The answer is 4", "R", True),
        ("The answer is 4", "The answer is 4", None, False),
        ("The answer is 4<end_of_turn>", "The answer is 4", None, False),
        ("<|channel>thought\nR<channel|>", None, "R", True),
    ]
    for gen, content, reasoning, thinking in cases:
        r = _res(content=content, reasoning=reasoning, thinking=thinking, generation_text=gen)
        assert r.event is None, gen
    # unlocatable content, no generation, bound tool calls: no event
    assert _res(content="x", reasoning=None, thinking=False, generation_text="y z").event is None
    assert _res(content="x", reasoning=None, thinking=False, generation_text=None).event is None
    assert _res(content="x", reasoning=None, thinking=False, tool_calls=[{"id": "1"}],
                generation_text="x call{}").event is None


def test_the_residue_rule_runs_on_replies_base_receives(monkeypatch):
    gen = "Let me check.<|tool_call>call:f{a:1}<tool_call|>"
    monkeypatch.setattr(base, "decode_generated_token_ids", lambda ids: gen + "<eos>")

    class Resp:
        status = 200

        async def json(self):
            return {"choices": [{"message": {"role": "assistant", "content": "Let me check."},
                                 "finish_reason": "stop", "stop_reason": None,
                                 "token_ids": [1, 2, 3]}],
                    "usage": {"completion_tokens": 3}}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *e):
            return False

    class Session:
        def post(self, url, json=None, **kw):
            return Resp()

    async def go():
        return await base._send_single_request(
            session=Session(), api_url="http://stub/v1/chat/completions", model_name="m",
            messages=[{"role": "user", "content": "q"}], extra_payload={},
            semaphore=asyncio.Semaphore(1), pbar=None, max_output_tokens=64, temperature=0.0)
    reply = asyncio.run(go())
    assert reply.parser_event == base.SERVER_PARSER_DROPPED_TEXT
    assert reply.parser_recovered_from == "token_ids" and "call:f{a:1}" in reply.text


# --------------------------------------------------------------------------- #
# --merge
# --------------------------------------------------------------------------- #
def _write_shard(root, i, n, traces, accuracy=None, scaffold="p1", status="success"):
    d = Path(root) / f"node-{i}" / "09-28-000000"
    d.mkdir(parents=True)
    correct = sum(1 for t in traces if t["is_correct"])
    res = {"benchmark_type": "eval", "eval_name": "gsm8k", "model_short": "m", "format": "hf",
           "shard": f"{i}/{n}", "status": status, "sample_traces": traces,
           "accuracy": accuracy if accuracy is not None else round(correct / len(traces) * 100, 2),
           "leaderboard_comparable": False,
           "leaderboard_comparable_reason": base.SHARD_REASON.format(label=f"{i}/{n}", index=i,
                                                                     count=n)}
    (d / "eval_gsm8k_m_hf.json").write_text(json.dumps(res))
    meta = {"gbench": {"commit": "abc"}, "eval_limit": None,
            "conditions": [{"pillar": "evals", "model": "m", "format": "hf",
                            "scaffold": {"pinned": {"x": scaffold, "shard": f"{i}/{n}"}},
                            "subject": {"subject_id": "sb_1"}}]}
    (d / "metadata.json").write_text(json.dumps(meta))
    return d.parent


def _traces(start, n, correct):
    return [{"sample_key": f"id:{k}", "global_idx": k, "attempt_idx": 0, "is_correct": k in correct,
             "category": "c"} for k in range(start, start + n)]


def test_merged_shards_equal_the_unsharded_count(tmp_path):
    good = {0, 1, 4, 7}
    dirs = [_write_shard(tmp_path, 1, 3, _traces(0, 3, good)),
            _write_shard(tmp_path, 2, 3, _traces(3, 3, good)),
            _write_shard(tmp_path, 3, 3, _traces(6, 4, good))]
    from gbench.merge import merge_runs
    written = merge_runs(dirs, tmp_path / "out")
    merged = json.loads(written[0].read_text())
    assert merged["accuracy"] == 40.0 and merged["total_questions"] == 10
    assert merged["merged_shards"] == 3 and merged["leaderboard_comparable"] is True
    assert "shard" not in merged


def test_incomplete_or_inconsistent_shards_are_refused(tmp_path):
    from gbench.merge import MergeError, merge_runs
    d1 = _write_shard(tmp_path / "a", 1, 2, _traces(0, 2, {0}))
    with pytest.raises(MergeError, match="missing shard"):
        merge_runs([d1], tmp_path / "out")
    d2 = _write_shard(tmp_path / "a", 2, 2, _traces(2, 2, {2}), scaffold="p2")
    with pytest.raises(MergeError, match="different evals scaffolds"):
        merge_runs([d1, d2], tmp_path / "out")
    d3 = _write_shard(tmp_path / "b", 1, 2, _traces(0, 2, {0}), accuracy=77.0)
    d4 = _write_shard(tmp_path / "b", 2, 2, _traces(2, 2, {2}))
    with pytest.raises(MergeError, match="merge_shards"):
        merge_runs([d3, d4], tmp_path / "out")
    d5 = _write_shard(tmp_path / "c", 1, 2, _traces(0, 2, {0}))
    d6 = _write_shard(tmp_path / "c", 2, 2, _traces(2, 2, {2}), status="error")
    with pytest.raises(MergeError, match="re-run it"):
        merge_runs([d5, d6], tmp_path / "out")


def test_the_cli_merge_exits_two_on_a_refusal(tmp_path):
    d1 = _write_shard(tmp_path / "a", 1, 2, _traces(0, 2, {0}))
    assert cli.main(["--merge", str(d1), "--results-dir", str(tmp_path / "r")]) == 2


# --------------------------------------------------------------------------- #
# docs follow the code
# --------------------------------------------------------------------------- #
def test_the_docs_name_the_new_interfaces():
    # the CLI tables and the env-var section moved from README.md into these docs
    readme = "\n".join(_read(rel) for rel in
                       ("README.md", "docs/cli-reference.md", "docs/environment-variables.md"))
    for flag in ("--tool-call-parser", "--reasoning-parser", "--vllm-serve-args",
                 "--served-model-id", "--gpus", "--merge", "--shard", "--eval-provenance",
                 "GBENCH_SERVER_PORT", "GBENCH_VLLM_SERVE_ARGS"):
        assert flag in readme, flag
    # the exit-code link and its target section (moved from README.md into docs/results.md)
    assert "](results.md#exit-codes)" in _read("docs/environment-variables.md")
    assert "\n### Exit codes" in _read("docs/results.md")
    # README.md and every doc split out of it
    for rel in _README_AND_SPLIT_DOCS:
        text = _read(rel)
        assert "GEMMA_MODELS_DIR" not in text and "--format gguf" not in text, rel
    knobs = " ".join(_read("docs/evals/env-knobs.md").split())
    for needle in ("config.resolve_server_port", "config.serve_args_for",
                   "base.endpoint_api_key", "base.server_parser_residue",
                   "base.RUN_SHAPING_SUITE_ENV", "self_judge"):
        assert needle in knobs, needle
    dist = _read("docs/distributed-gbench.md")
    assert "sample_key" in dist and "--merge" in dist and "shard_applied: false" in dist
