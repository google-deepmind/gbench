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

"""The servers gbench launches, the served model it measures, the Golden Set's endpoint, and
the calibration guard.

No vLLM runs: server launches are stopped at (or replaced by stubs of) subprocess.Popen, the
endpoints are in-process HTTP servers on OS-assigned ports, docker and nvidia-smi are faked, and
local serving ports are never probed.
"""

import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import pytest

import gbench.core.config as C
from gbench.core.config import BenchmarkConfig
from gbench.core.models import ModelConfig, ModelFormat
from gbench.runners import golden as golden_mod
from gbench.runners import serving
from gbench.runners.serving import ServingBenchmarkRunner
from gbench.utils import calibrate as cal
from gbench.utils import endpoint as ep
from gbench.utils import process_groups

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _LM:
    def __init__(self, tmp_path):
        self.tmp = tmp_path

    def get_server_log_path(self, *a):
        return str(self.tmp / "server.log")


def _model(name="org/m", mm=False, vtoks=None):
    m = ModelConfig(short_name=name.rsplit("/", 1)[-1], hf_model_id=name, total_params_b=4.0)
    m.supports_multimodal = mm
    m.vision_tokens_per_image = vtoks
    return m


def _runner(tmp_path, **cfg_kw):
    cfg = BenchmarkConfig()
    cfg.max_model_len_explicit = None
    cfg.log_manager = _LM(tmp_path)
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    return ServingBenchmarkRunner(cfg)


class _Endpoint:
    """An in-process OpenAI-compatible endpoint: /health, /v1/models and chat completions,
    recording every path and Authorization header it saw."""

    def __init__(self, models=("org/m",), key=None, chat_status=200, max_len=None):
        self.models, self.key, self.chat_status, self.max_len = list(models), key, chat_status, max_len
        self.paths, self.auth = [], []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, body):
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def _authorized(self):
                outer.auth.append(self.headers.get("Authorization"))
                return outer.key is None or self.headers.get("Authorization") == f"Bearer {outer.key}"

            def do_GET(self):  # noqa: N802
                outer.paths.append(self.path)
                if self.path == "/health":
                    return self._send(200, {})
                if not self._authorized():
                    return self._send(401, {"error": "unauthorized"})
                if self.path == "/v1/models":
                    data = [{"id": m, **({"max_model_len": outer.max_len.get(m)}
                                         if outer.max_len else {})} for m in outer.models]
                    return self._send(200, {"data": data})
                return self._send(404, {"error": "not found"})

            def do_POST(self):  # noqa: N802
                outer.paths.append(self.path)
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if not self._authorized():
                    return self._send(401, {"error": "unauthorized"})
                if outer.chat_status != 200:
                    return self._send(outer.chat_status, {"error": "scripted"})
                return self._send(200, {"model": body.get("model"), "choices": [
                    {"message": {"content": "ok"}}], "usage": {"prompt_tokens": 5}})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def port(self):
        return self.server.server_address[1]

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def endpoint_factory():
    made = []

    def _make(**kw):
        e = _Endpoint(**kw)
        made.append(e)
        return e
    yield _make
    for e in made:
        e.close()


@pytest.fixture(autouse=True)
def _no_endpoint_key(monkeypatch):
    for name in ("VLLM_API_KEY", "OPENAI_API_KEY", "GBENCH_SERVER_PORT", "GBENCH_SERVER_HOST",
                 "GBENCH_VLLM_BIN", "GBENCH_VLLM_SERVE_ARGS"):
        monkeypatch.delenv(name, raising=False)


def _capture_popen(runner, model, **start_kwargs):
    """(cmd, popen kwargs) _start_server hands to Popen (the launch is stopped there)."""
    seen = []

    def fake_popen(cmd, *a, **k):
        seen.append((list(cmd), k))
        raise RuntimeError("stop before launch")

    with mock.patch("subprocess.Popen", fake_popen):
        assert runner._start_server(model, ModelFormat.HF, **start_kwargs) is False
    assert seen, "Popen was never reached"
    return seen[0]


# --------------------------------------------------------------------------- #
# the port, and a server that is ours
# --------------------------------------------------------------------------- #
def test_an_explicit_busy_port_fails_before_anything_is_launched(tmp_path, monkeypatch):
    monkeypatch.setenv("GBENCH_SERVER_PORT", "8000")
    monkeypatch.setattr(C, "port_is_free", lambda p: False)
    runner = _runner(tmp_path)
    with mock.patch("subprocess.Popen", side_effect=AssertionError("launched")):
        assert runner._start_server(_model(), ModelFormat.HF) is False


def test_an_automatic_port_skips_busy_ones_and_every_consumer_follows_it(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "port_is_free", lambda p: p not in (8000, 8001))
    runner = _runner(tmp_path)
    cmd, _ = _capture_popen(runner, _model())
    assert cmd[cmd.index("--port") + 1] == "8002"
    assert runner.server_port == 8002 and runner.server_port_source == "auto"


def test_a_dead_child_is_never_ready_whatever_answers_the_port(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    runner.server_port = 18999
    runner.server_process = mock.Mock(pid=12345, poll=mock.Mock(return_value=1))
    monkeypatch.setattr(serving.requests, "get",
                        lambda *a, **k: mock.Mock(status_code=200))
    assert runner._wait_for_server_ready(timeout=5) is False


def test_a_foreign_listener_on_the_port_is_reported_as_taken(tmp_path, monkeypatch):
    runner = _runner(tmp_path)
    runner.server_port = 18999
    runner.server_process = mock.Mock(pid=12345, poll=mock.Mock(return_value=None))
    monkeypatch.setattr(serving.requests, "get", lambda *a, **k: mock.Mock(status_code=200))
    monkeypatch.setattr(serving, "listener_owned_by_session", lambda port, sid: False)
    assert runner._wait_for_server_ready(timeout=5, expected_model_id="org/m") == "port_taken"


def test_a_server_that_serves_another_model_is_not_ready(tmp_path, endpoint_factory):
    e = endpoint_factory(models=["someone-elses/model"])
    runner = _runner(tmp_path)
    runner.server_port = e.port
    # The listener belongs to this test process's session: pass that as the child's session.
    runner.server_process = mock.Mock(pid=os.getsid(0), poll=mock.Mock(return_value=None))
    assert runner._wait_for_server_ready(timeout=10, expected_model_id="org/m") is False
    e.models = ["org/m"]
    assert runner._wait_for_server_ready(timeout=10, expected_model_id="org/m") is True


def test_listener_ownership_is_read_from_proc():
    """Linux: the listener's socket inode is found among the session's file descriptors.
    Without /proc the answer is None (unknown), and readiness falls back to the model listing."""
    has_proc = Path("/proc/net/tcp").exists()
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        assert serving.listener_owned_by_session(port, os.getsid(0)) is (True if has_proc
                                                                         else None)
        assert serving.listener_owned_by_session(port, 2 ** 22 + 7) is (False if has_proc
                                                                        else None)
    finally:
        s.close()
    assert serving.listener_owned_by_session(port, os.getsid(0)) is None


def test_an_automatic_port_taken_during_startup_is_retried_but_an_explicit_one_is_not(
        tmp_path, monkeypatch):
    ports = iter([(8000, "auto"), (8001, "auto")])
    monkeypatch.setattr(serving, "resolve_server_port", lambda environ=None: next(ports))
    outcomes = iter(["port_taken", "ready"])
    runner = _runner(tmp_path)
    monkeypatch.setattr(runner, "_launch_and_wait", lambda *a, **k: next(outcomes))
    assert runner._start_server(_model(), ModelFormat.HF) is True
    assert runner.server_port == 8001

    monkeypatch.setattr(serving, "resolve_server_port",
                        lambda environ=None: (8005, "GBENCH_SERVER_PORT"))
    calls = []
    monkeypatch.setattr(runner, "_launch_and_wait",
                        lambda *a, **k: calls.append(1) or "port_taken")
    assert runner._start_server(_model(), ModelFormat.HF) is False
    assert calls == [1]


# --------------------------------------------------------------------------- #
# the command: vLLM from gbench's interpreter, parsers, extras, DP, host
# --------------------------------------------------------------------------- #
def test_the_server_runs_the_vllm_of_gbenchs_interpreter(tmp_path, monkeypatch):
    cmd = _runner(tmp_path).build_server_cmd(_model(), ModelFormat.HF, port=18001)
    assert cmd[:4] == [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve"]
    monkeypatch.setenv("GBENCH_VLLM_BIN", "/opt/vllm-env/bin/vllm")
    cmd = _runner(tmp_path).build_server_cmd(_model(), ModelFormat.HF, port=18001)
    assert cmd[:2] == ["/opt/vllm-env/bin/vllm", "serve"]
    from gbench.utils import vllm_engine_record
    assert vllm_engine_record()["cli_source"] == "GBENCH_VLLM_BIN"


def test_the_builder_is_pure_and_the_parsers_go_only_on_the_eval_server(tmp_path):
    runner = _runner(tmp_path, tool_call_parser="toolp", reasoning_parser="reasp",
                     vllm_serve_args=["--trust-remote-code", "--chat-template", "t.jinja"])
    with mock.patch("subprocess.Popen", side_effect=AssertionError("launched")), \
            mock.patch.object(C, "port_is_free", side_effect=AssertionError("probed")):
        native = runner.build_server_cmd(_model(), ModelFormat.HF, context="native", port=18001)
        perf = runner.build_server_cmd(_model(), ModelFormat.HF, context="perf", port=18001)
    assert native[-8:] == ["--enable-auto-tool-choice", "--tool-call-parser", "toolp",
                           "--reasoning-parser", "reasp", "--trust-remote-code",
                           "--chat-template", "t.jinja"]
    assert "--tool-call-parser" not in perf and "--reasoning-parser" not in perf
    assert perf[-3:] == ["--trust-remote-code", "--chat-template", "t.jinja"]
    assert "<port>" in runner.build_server_cmd(_model(), ModelFormat.HF, port=None)


def test_data_parallel_replicas_are_passed_to_the_server(tmp_path):
    runner = _runner(tmp_path, tensor_parallel_size=2, data_parallel_size=2)
    cmd = runner.build_server_cmd(_model(), ModelFormat.HF, port=18001)
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"
    assert cmd[cmd.index("--data-parallel-size") + 1] == "2"
    assert "--data-parallel-size" not in _runner(tmp_path).build_server_cmd(
        _model(), ModelFormat.HF, port=18001)


def test_perf_servers_bind_loopback_and_the_eval_server_binds_what_bridge_suites_reach(
        tmp_path, monkeypatch):
    r = _runner(tmp_path, evals=["gsm8k"])
    perf = r.build_server_cmd(_model(), ModelFormat.HF, context="perf", port=18001)
    native = r.build_server_cmd(_model(), ModelFormat.HF, context="native", port=18001)
    assert perf[perf.index("--host") + 1] == "127.0.0.1"
    assert "--host" not in native
    monkeypatch.setenv("GBENCH_SERVER_HOST", "127.0.0.1")
    native = r.build_server_cmd(_model(), ModelFormat.HF, context="native", port=18001)
    assert native[native.index("--host") + 1] == "127.0.0.1"


def test_the_soft_token_setting_reaches_the_processor_and_only_the_eval_server(tmp_path):
    assert BenchmarkConfig().eval_max_soft_tokens is None
    plain = _runner(tmp_path).build_server_cmd(_model(), ModelFormat.HF, context="native",
                                               port=18001)
    assert "--hf-overrides" not in plain and "--mm-processor-kwargs" not in plain
    r = _runner(tmp_path, eval_max_soft_tokens=560)
    native = r.build_server_cmd(_model(), ModelFormat.HF, context="native", port=18001)
    assert json.loads(native[native.index("--mm-processor-kwargs") + 1]) == {"max_soft_tokens": 560}
    assert "--hf-overrides" not in native
    assert "--mm-processor-kwargs" not in r.build_server_cmd(_model(), ModelFormat.HF,
                                                             context="perf", port=18001)
    r.config.vllm_serve_args = ["--mm-processor-kwargs", "{}"]
    with pytest.raises(ValueError, match="max_soft_tokens"):
        r.build_server_cmd(_model(), ModelFormat.HF, context="native", port=18001)


def test_the_server_runs_on_the_selected_gpus(tmp_path, monkeypatch):
    monkeypatch.setattr(serving, "resolve_server_port", lambda environ=None: (18001, "auto"))
    runner = _runner(tmp_path, selected_gpus=["GPU-a", "GPU-b"])
    _cmd, kwargs = _capture_popen(runner, _model())
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"
    assert kwargs["env"]["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert kwargs["start_new_session"] is True


def test_the_gpu_memory_wait_looks_only_at_the_selected_gpus(tmp_path, monkeypatch):
    runner = _runner(tmp_path, selected_gpus=["GPU-a"])
    calls = []

    def fake_run(cmd, *a, **k):
        calls.append(cmd)
        out = "81920" if "--query-gpu=memory.total" in cmd else "80000"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
    monkeypatch.setattr(serving.subprocess, "run", fake_run)
    runner._wait_for_gpu_memory(max_wait_seconds=5)
    assert calls and all("--id=GPU-a" in c for c in calls)


# --------------------------------------------------------------------------- #
# the served model
# --------------------------------------------------------------------------- #
def test_a_remote_endpoint_serving_another_model_is_an_error_not_its_first_id(
        tmp_path, endpoint_factory):
    e = endpoint_factory(models=["served-name"])
    runner = _runner(tmp_path, remote_endpoint=f"{e.base}/v1")
    with pytest.raises(ep.EndpointError, match="--served-model-id"):
        runner._resolve_model_id(_model("org/m"), ModelFormat.REMOTE)
    assert "/v1/v1/models" not in e.paths and "/v1/models" in e.paths


def test_the_served_id_matching_the_model_is_used_and_recorded(tmp_path, endpoint_factory):
    e = endpoint_factory(models=["other/x", "org/m"])
    runner = _runner(tmp_path, remote_endpoint=f"{e.base}/v1")
    assert runner._resolve_model_id(_model("org/m"), ModelFormat.REMOTE) == "org/m"
    rec = runner.server_record(_model("org/m"), ModelFormat.REMOTE)
    assert rec["served_model_id"] == "org/m" and rec["served_model_id_source"].startswith("matches")
    runner2 = _runner(tmp_path, remote_endpoint=f"{e.base}/v1", served_model_id="other/x")
    assert runner2._resolve_model_id(_model("org/m"), ModelFormat.REMOTE) == "other/x"


def test_the_context_window_comes_from_the_matched_entry(endpoint_factory):
    e = endpoint_factory(models=["a/first", "org/m"], max_len={"a/first": 999, "org/m": 131072})
    served = ep.resolve_served_model(f"{e.base}/v1", ["org/m"])
    assert served == ep.ServedModel("org/m", served.source, 131072)
    ok, _msg, max_len = ep.verify_endpoint_functional(e.base, model_id="org/m")
    assert ok and max_len == 131072


def test_a_listing_failure_is_fatal_locally_and_recorded_remotely():
    dead = "http://127.0.0.1:1/v1"
    with pytest.raises(ep.EndpointError):
        ep.resolve_served_model(dead, ["org/m"], local=True)
    served = ep.resolve_served_model(dead, ["org/m"], local=False)
    assert served.id == "org/m" and served.source.startswith("configured (listing failed")


# --------------------------------------------------------------------------- #
# the endpoint key
# --------------------------------------------------------------------------- #
def test_the_preflight_sends_the_key_and_a_refused_key_is_a_hard_failure(
        endpoint_factory, monkeypatch):
    e = endpoint_factory(key="sekrit-key-1234")
    ok, msg, _ = ep.verify_endpoint_functional(e.base)
    assert ok is False and "401" in msg and "VLLM_API_KEY" in msg   # no /health fallback
    monkeypatch.setenv("OPENAI_API_KEY", "sekrit-key-1234")          # the documented alias
    assert ep.verify_endpoint_functional(e.base)[0] is True
    monkeypatch.setenv("VLLM_API_KEY", "EMPTY")                      # a placeholder is skipped
    assert ep.verify_endpoint_functional(e.base)[0] is True


def test_the_remote_perf_requests_send_the_key(tmp_path, endpoint_factory, monkeypatch):
    e = endpoint_factory(key="sekrit-key-1234")
    monkeypatch.setenv("VLLM_API_KEY", "sekrit-key-1234")
    runner = _runner(tmp_path, remote_endpoint=f"{e.base}/v1")
    assert runner._resolve_model_id(_model("org/m"), ModelFormat.REMOTE) == "org/m"
    assert runner._request_headers() == {"Authorization": "Bearer sekrit-key-1234"}
    from gbench.runners.stress import StressTestRunner
    assert StressTestRunner(runner.config)._request_headers() == runner._request_headers()


def test_the_golden_set_sends_the_openai_alias_key(tmp_path, endpoint_factory, monkeypatch):
    e = endpoint_factory(models=["org/m"], key="sekrit-key-1234")
    monkeypatch.setenv("OPENAI_API_KEY", "sekrit-key-1234")
    cfg = BenchmarkConfig(results_dir=tmp_path / "r")
    cfg.remote_endpoint = f"{e.base}/v1"
    g = golden_mod.GoldenBenchmarkRunner(cfg)
    assert g._resolve_model_id(cfg.remote_endpoint, _model("org/m")) == "org/m"
    assert "Bearer sekrit-key-1234" in e.auth


# --------------------------------------------------------------------------- #
# the Golden Set's endpoint
# --------------------------------------------------------------------------- #
def _golden_run(cfg, model):
    g = golden_mod.GoldenBenchmarkRunner(cfg)
    task = {"id": "t", "category": "x",
            "conversation": {"messages": [{"role": "user", "content": "say ok"}]},
            "golden_truth": {"match_type": "contains_all", "expected_outputs": ["ok"]}}
    return g.run(model, tasks=[task])


def test_the_golden_set_uses_the_server_the_cli_launched(tmp_path, endpoint_factory):
    e = endpoint_factory(models=["org/m"])
    cfg = BenchmarkConfig(results_dir=tmp_path / "r")
    cfg.remote_endpoint = None
    cfg.local_endpoint = f"{e.base}/v1"
    summary = _golden_run(cfg, _model("org/m"))
    assert summary["status"] == golden_mod.STATUS_PASSED
    assert summary["endpoint"] == cfg.local_endpoint
    assert "/v1/chat/completions" in e.paths


def test_the_golden_set_without_an_endpoint_is_a_harness_error(tmp_path):
    cfg = BenchmarkConfig(results_dir=tmp_path / "r")
    cfg.remote_endpoint = None
    cfg.local_endpoint = None
    summary = _golden_run(cfg, _model("org/m"))
    assert summary["status"] == golden_mod.STATUS_ERROR and summary["accuracy_percent"] is None
    assert "--remote-endpoint" in summary["harness_errors"][0]
    assert not hasattr(golden_mod, "DEFAULT_ENDPOINT")


def test_a_local_golden_server_serving_another_model_is_a_harness_error(tmp_path, endpoint_factory):
    e = endpoint_factory(models=["someone-elses/model"])
    cfg = BenchmarkConfig(results_dir=tmp_path / "r")
    cfg.remote_endpoint = None
    cfg.local_endpoint = f"{e.base}/v1"
    summary = _golden_run(cfg, _model("org/m"))
    assert summary["status"] == golden_mod.STATUS_ERROR
    assert not any(p.endswith("/chat/completions") for p in e.paths)


def test_a_remote_endpoint_without_v1_is_normalized_for_the_golden_set(tmp_path, endpoint_factory):
    e = endpoint_factory(models=["org/m"])
    cfg = BenchmarkConfig(results_dir=tmp_path / "r")
    cfg.remote_endpoint = e.base                      # a programmatic caller's raw value
    summary = _golden_run(cfg, _model("org/m"))
    assert summary["status"] == golden_mod.STATUS_PASSED
    assert summary["endpoint"] == f"{e.base}/v1"


# --------------------------------------------------------------------------- #
# the Golden Set's scoring version gate
# --------------------------------------------------------------------------- #
def test_the_golden_scoring_code_matches_the_reviewed_fingerprint():
    assert golden_mod.scoring_fingerprint() == golden_mod.GOLDEN_SCORING_FP, (
        "the Golden Set scoring code changed: bump SCORING_VERSION if verdicts can change, "
        "then set GOLDEN_SCORING_FP to the new scoring_fingerprint()")
    assert golden_mod.SCORING_VERSION == 4


def test_the_fingerprint_moves_with_code_and_not_with_docstrings(tmp_path):
    root = tmp_path / "runners"
    (root / "eval_suites").mkdir(parents=True)
    src = (REPO / "gbench/runners/golden.py").read_text(encoding="utf-8")
    (root / "eval_suites" / "sandbox.py").write_text(
        (REPO / "gbench/runners/eval_suites/sandbox.py").read_text(encoding="utf-8"))
    (root / "golden.py").write_text(src)
    base = golden_mod.scoring_fingerprint(root)
    assert base == golden_mod.GOLDEN_SCORING_FP
    (root / "golden.py").write_text(src.replace(
        '    """Normalize an answer before comparison.', '    """Normalise an answer first.'))
    assert golden_mod.scoring_fingerprint(root) == base
    (root / "golden.py").write_text(src.replace(
        'return " ".join(text.split()).lower()', 'return " ".join(text.split()).upper()'))
    assert golden_mod.scoring_fingerprint(root) != base


def test_the_verdict_excludes_not_applicable_cases():
    counts = {"passed": 3, "failed": 1, "error": 0, "not_applicable": 2}
    assert golden_mod._verdict(counts, 6) == ("failed", 75.0, 4)
    assert golden_mod._verdict({"passed": 1, "error": 1}, 2)[:2] == ("error", None)
    assert golden_mod._verdict({"not_applicable": 2}, 2) == ("passed", None, 0)


# --------------------------------------------------------------------------- #
# multimodal: the per-image count is measured, never a family constant
# --------------------------------------------------------------------------- #
def test_serving_sizes_the_mm_workload_with_the_count_the_server_measured(tmp_path):
    r = _runner(tmp_path)
    assert not hasattr(ServingBenchmarkRunner, "PER_IMAGE_SOFT_TOKENS")
    with pytest.raises(RuntimeError, match="per-image token count"):
        r._per_image_tokens(_model(mm=True))
    assert r._per_image_tokens(_model(mm=True, vtoks=256)) == (256, "config")
    r._vision_tokens_per_image, r._vision_tokens_source = 1100, "measured"
    assert r._per_image_tokens(_model(mm=True, vtoks=256)) == (1100, "measured")


def test_the_mm_serving_path_adopts_the_preflight_measurement(tmp_path, monkeypatch):
    r = _runner(tmp_path, remote_endpoint="http://h:1/v1", warmup_iterations=0, num_iterations=1)
    monkeypatch.setattr(r, "_resolve_model_id", lambda m, f: "org/m")
    monkeypatch.setattr(r, "_assert_mm_images_processed", lambda *a, **k: 777)
    seen = {}

    async def fake_bench(model, format, max_concurrency, num_prompts, phase="serving", rep=0):
        seen["per_img"] = r._per_image_tokens(model)
        raise RuntimeError("stop")
    monkeypatch.setattr(r, "_run_mm_serving_benchmark", fake_bench)
    with pytest.raises(RuntimeError):
        r._run_mm_with_iterations(_model(mm=True), ModelFormat.REMOTE, 1, 1, manage_server=False)
    assert seen["per_img"] == (777, "measured")


def test_throughput_counts_image_tokens_with_the_models_processor(tmp_path, monkeypatch):
    from gbench.runners.throughput import ThroughputBenchmarkRunner
    t = ThroughputBenchmarkRunner(BenchmarkConfig())
    monkeypatch.setattr(t, "_count_image_tokens", lambda path: 64)
    assert t._mm_image_tokens(_model(mm=True, vtoks=280), ModelFormat.HF) == (64, "processor")
    t2 = ThroughputBenchmarkRunner(BenchmarkConfig())
    monkeypatch.setattr(t2, "_count_image_tokens", lambda path: None)
    assert t2._mm_image_tokens(_model(mm=True, vtoks=280), ModelFormat.HF) == (280, "config")
    t3 = ThroughputBenchmarkRunner(BenchmarkConfig())
    monkeypatch.setattr(t3, "_count_image_tokens", lambda path: None)
    assert t3._mm_image_tokens(_model(mm=True), ModelFormat.HF) == (None, "unknown")


def test_no_model_family_constants_or_gguf_paths_remain_in_the_runners():
    for rel in ("gbench/runners/serving.py", "gbench/runners/stress.py",
                "gbench/runners/throughput.py", "gbench/runners/golden.py"):
        src = (REPO / rel).read_text(encoding="utf-8")
        assert "PER_IMAGE_SOFT_TOKENS" not in src, rel
        assert "ModelFormat.GGUF" not in src, rel
        assert "or 280" not in src, rel
        assert "Gemma benchmark suite" not in src, rel


# --------------------------------------------------------------------------- #
# offline throughput: same vLLM, selected GPUs, engine-only extras
# --------------------------------------------------------------------------- #
def test_throughput_runs_gbenchs_vllm_with_engine_extras_only(tmp_path, monkeypatch):
    from gbench.runners import throughput as T
    cfg = BenchmarkConfig()
    cfg.vllm_serve_args = ["--trust-remote-code", "--chat-template", "t.jinja",
                           "--hf-overrides", '{"a": 1}']
    t = T.ThroughputBenchmarkRunner(cfg)
    monkeypatch.setattr(T, "_engine_option_names",
                        lambda: frozenset({"--trust-remote-code", "--hf-overrides"}))
    cmd = t._build_command(_model(), ModelFormat.HF, 128, 128, 1, 10, tmp_path / "o.json")
    assert cmd[:5] == [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "throughput"]
    assert cmd[-3:] == ["--trust-remote-code", "--hf-overrides", '{"a": 1}']
    assert "--chat-template" not in cmd
    fp = t._run_fingerprint(_model(), 10, 4096, False)
    assert fp["vllm_serve_args_not_applied"] == ["--chat-template", "t.jinja"]


def test_split_engine_args_keeps_each_options_values():
    from gbench.runners.throughput import split_engine_args
    kept, dropped = split_engine_args(
        ["--trust-remote-code", "--tool-call-parser", "x", "--max_num_seqs=8", "--no-foo"],
        frozenset({"--trust-remote-code", "--max-num-seqs"}))
    assert kept == ["--trust-remote-code", "--max_num_seqs=8"]
    assert dropped == ["--tool-call-parser", "x", "--no-foo"]
    assert split_engine_args(["--a", "1"], None) == (["--a", "1"], [])


def test_offline_throughput_runs_on_one_replica_of_the_selected_gpus(monkeypatch):
    from gbench.runners import throughput as T
    cfg = BenchmarkConfig()
    cfg.selected_gpus = ["GPU-a", "GPU-b", "GPU-c", "GPU-d"]
    cfg.tensor_parallel_size, cfg.data_parallel_size = 2, 2
    t = T.ThroughputBenchmarkRunner(cfg)
    seen = {}

    class P:
        pid = 424242
        returncode = 0

        def communicate(self, timeout=None):
            return "out", "err"

    def fake_popen(cmd, **k):
        seen.update(k)
        return P()
    monkeypatch.setattr(T.subprocess, "Popen", fake_popen)
    assert t._run_offline_subprocess(["x"], timeout=5) == (0, "out", "err")
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-b"
    assert 424242 not in process_groups.tracked_pids()
    fp = t._run_fingerprint(_model(), 10, 4096, False)
    assert fp["data_parallel_size"] == 1 and fp["run_data_parallel_size"] == 2


# --------------------------------------------------------------------------- #
# a terminated run leaves no server behind
# --------------------------------------------------------------------------- #
def test_stop_group_reaches_every_process_of_the_group():
    proc = subprocess.Popen(["sh", "-c", "sleep 60 & sleep 60; wait"], start_new_session=True)
    process_groups.track(proc)
    try:
        time.sleep(0.3)
        process_groups.stop_group(proc, grace_s=5)
        with pytest.raises(ProcessLookupError):
            os.killpg(proc.pid, 0)
        assert proc.pid not in process_groups.tracked_pids()
    finally:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_sigterm_to_gbench_stops_the_server_group(tmp_path):
    pid_file = tmp_path / "server.pid"
    script = tmp_path / "run.py"
    script.write_text(textwrap.dedent(f"""
        import subprocess, sys, time
        sys.path.insert(0, {str(REPO)!r})
        from gbench.utils import process_groups
        from gbench.cli import _termination_handlers
        proc = subprocess.Popen(["sh", "-c", "sleep 120 & sleep 120; wait"],
                                start_new_session=True)
        process_groups.track(proc)
        open({str(pid_file)!r}, "w").write(str(proc.pid))
        with _termination_handlers():
            time.sleep(120)
    """))
    child = subprocess.Popen([sys.executable, str(script)], cwd=str(tmp_path))
    pgid = None
    try:
        deadline = time.time() + 90
        while time.time() < deadline and not (pid_file.exists() and pid_file.read_text()):
            assert child.poll() is None, "the gbench stand-in exited early"
            time.sleep(0.2)
        pgid = int(pid_file.read_text())
        os.killpg(pgid, 0)                         # the server group is up
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=60)
        deadline = time.time() + 15
        while time.time() < deadline:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.2)
        with pytest.raises(ProcessLookupError):
            os.killpg(pgid, 0)
    finally:
        if child.poll() is None:
            child.kill()
        if pgid:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_the_dead_atexit_line_is_gone():
    src = (REPO / "gbench/runners/serving.py").read_text(encoding="utf-8")
    assert "atexit.register(self._cleanup_server)" not in src


# --------------------------------------------------------------------------- #
# run directories of concurrent runs
# --------------------------------------------------------------------------- #
def test_runs_started_in_the_same_second_get_their_own_directory(tmp_path, monkeypatch):
    from datetime import datetime
    from gbench.utils import log_manager

    class _Fixed(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 28, 12, 0, 0)
    monkeypatch.setattr(log_manager, "datetime", _Fixed)
    a = log_manager.LogManager(tmp_path)
    b = log_manager.LogManager(tmp_path)
    c = log_manager.LogManager(tmp_path)
    assert a.results_dir.name == "09-28-120000"
    assert b.results_dir.name == "09-28-120000-2" and c.results_dir.name == "09-28-120000-3"
    assert b.timestamp == b.results_dir.name
    (a.results_dir / "metadata.json").write_text("{}")
    a.release_run_lock()                   # run a has ended; a later run continues its directory
    assert log_manager.LogManager(a.results_dir).results_dir == a.results_dir   # reuse path kept


# --------------------------------------------------------------------------- #
# calibration: docker networks, occupancy, models, suites
# --------------------------------------------------------------------------- #
def test_docker_address_pools_default_to_31_networks():
    pools, source = cal.parse_address_pools("null")
    assert source == "docker default"
    assert cal.network_capacity(pools, [])["capacity"] == 31
    # the default bridge and two compose networks already hold three of them
    used = cal.network_capacity(pools, ["172.17.0.0/16", "172.18.0.0/16", "192.168.0.0/20",
                                        "10.99.0.0/24", "fd00::/64"])
    assert used == {"capacity": 31, "in_use": 3, "free": 28}
    pools, source = cal.parse_address_pools('[{"Base":"172.28.0.0/14","Size":24}]')
    assert source.startswith("daemon.json") and cal.network_capacity(pools, [])["capacity"] == 1024
    assert cal.parse_address_pools(None) is None


def _hw(**kw):
    hw = {"gpus": 8, "gpus_idle": 8, "gpu_total_vram_gb": 80.0, "gpu_free_vram_gb": 79.0,
          "gpu_idle_free_vram_gb": 79.0, "gpu_idle_total_vram_gb": 80.0, "cpu_cores": 96,
          "ram_total_gb": 700.0, "ram_available_gb": 600.0, "docker": True,
          "docker_networks": {"capacity": 31, "in_use": 11, "free": 20,
                              "pools_source": "docker default"}}
    hw.update(kw)
    return hw


def test_the_guard_caps_network_suites_at_the_free_networks(monkeypatch):
    monkeypatch.delenv(cal.GUARD_BYPASS_ENV, raising=False)
    hw = _hw()
    assert cal.check_oversubscription(32, hw)[0]                          # CPU/RAM only
    assert cal.check_oversubscription(32, hw, suites=["gpqa", "lcb"])[0]  # no per-task network
    ok, msg = cal.check_oversubscription(32, hw, suites=["terminal_bench"])
    assert not ok and "20 free docker network" in msg and "default-address-pools" in msg
    assert cal.check_oversubscription(20, hw, suites=["terminal_bench"])[0]
    assert cal.max_safe_sandboxes(hw, ["swe_bench_pro"]) == 20
    assert cal.max_safe_sandboxes(hw) == 48


def test_calibrates_own_recommendation_passes_the_guard_on_a_small_host(monkeypatch):
    monkeypatch.delenv(cal.GUARD_BYPASS_ENV, raising=False)
    hw = _hw(gpus=0, gpus_idle=0, cpu_cores=8, ram_available_gb=28.0)
    rec = cal.recommend(hw, evals=["terminal_bench", "gpqa"])
    assert cal.check_oversubscription(rec["sandboxes"], hw, suites=["terminal_bench"])[0]
    assert rec["eval_concurrency"] == 32
    report = cal.format_report(hw, rec)
    assert "--eval-concurrency 32" in report and "--batch-sizes" not in report
    assert "not guarded" in report


def test_calibration_counts_idle_gpus_and_sizes_the_selected_models():
    hw = _hw(gpus_idle=1)
    rec = cal.recommend(hw)
    assert rec["num_gpus"] == 1 and "1 of 8 GPU(s) idle" in rec["reasons"]["num_gpus"]
    big = ModelConfig(short_name="big", hf_model_id="org/big", total_params_b=60.0)
    rec = cal.recommend(hw, models=[big])
    assert rec["num_gpus"] == 2
    assert rec["per_model"][0]["exceeds_idle"] is True
    assert "only 1 GPU(s) are idle" in rec["reasons"]["num_gpus"]
    rec = cal.recommend(hw, evals=["gpqa_diamond"])
    assert rec["sandboxes"] is None and "no containerized suite" in rec["reasons"]["sandboxes"]


def test_detect_hardware_reads_occupancy_and_networks_from_fakes(monkeypatch):
    inv = [C.GpuInfo(0, "GPU-0", 80.0, 79.0), C.GpuInfo(1, "GPU-1", 80.0, 4.0, ("123",))]
    monkeypatch.setattr(cal, "_gpu_occupancy", lambda: inv)
    monkeypatch.setattr(cal, "_docker_ok", lambda: True)
    outputs = {"info": "null", "ls": "n1 n2\n",
               "inspect": "172.17.0.0/16 \n172.18.0.0/16 \n"}

    def fake_run(cmd, timeout=15):
        return outputs["info" if "info" in cmd else "ls" if "ls" in cmd else "inspect"]
    monkeypatch.setattr(cal, "_run", fake_run)
    monkeypatch.setattr(C, "_gpu_total_vram_gb", lambda: 80.0)
    monkeypatch.setattr(C, "_gpu_min_free_vram_gb", lambda: 4.0)
    hw = cal.detect_hardware()
    assert hw["gpus"] == 2 and hw["gpus_idle"] == 1
    assert hw["docker_networks"]["free"] == 29 and hw["docker_networks"]["capacity"] == 31


# --------------------------------------------------------------------------- #
# the remote multimodal probe
# --------------------------------------------------------------------------- #
def test_an_unknown_model_id_is_inconclusive_not_missing_vision(endpoint_factory):
    e = endpoint_factory(models=["org/other"], chat_status=404)
    probe = ep.probe_multimodal(f"{e.base}/v1", "org/m", timeout=5)
    assert probe.verdict == "inconclusive" and "does not serve" in probe.reason
    assert ep.probe_multimodal_support(f"{e.base}/v1", "org/m", timeout=5) is True


def test_the_probe_targets_the_served_id_and_a_rejection_is_unsupported(endpoint_factory):
    e = endpoint_factory(models=["local/dir/m"], chat_status=400)
    probe = ep.probe_multimodal(f"{e.base}/v1", "org/m", timeout=5)
    assert probe.model_id == "local/dir/m" and probe.verdict == "unsupported"
    assert ep.probe_multimodal_support(f"{e.base}/v1", "org/m", timeout=5) is False


# --------------------------------------------------------------------------- #
# failed perf records are harness errors
# --------------------------------------------------------------------------- #
def test_a_failed_serving_configuration_is_a_named_harness_record(tmp_path, monkeypatch):
    r = _runner(tmp_path, remote_endpoint="http://h:1/v1")
    monkeypatch.setattr(r.config, "get_serving_configs",
                        lambda: [{"batch_size": 1, "num_prompts": 2}])
    monkeypatch.setattr(r, "run_with_iterations",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("endpoint 503")))
    out = r.run_all(_model(), ModelFormat.REMOTE)
    assert out == [{"failed": True, "failure_kind": "harness", "error": "endpoint 503",
                    "model": "m", "format": "remote-endpoint", "batch_size": 1,
                    "multimodal": False}]
    from gbench.cli import run_exit_code
    assert run_exit_code([{**out[0], "benchmark_type": "serving"}]) == 2


def test_a_vllm_bin_override_is_not_held_to_the_process_tree_check(tmp_path, monkeypatch,
                                                                   endpoint_factory):
    """GBENCH_VLLM_BIN may start the server outside gbench's process tree (a container, a
    detaching wrapper): its listener is not the child's, so only the model listing is checked."""
    e = endpoint_factory(models=["org/m"])
    monkeypatch.setenv("GBENCH_VLLM_BIN", "/opt/wrapper/vllm")
    monkeypatch.setattr(serving, "listener_owned_by_session",
                        lambda *a: pytest.fail("ownership checked for an override"))
    runner = _runner(tmp_path)
    runner.server_port = e.port
    runner.server_process = mock.Mock(pid=2 ** 22 + 7, poll=mock.Mock(return_value=None))
    assert runner._wait_for_server_ready(timeout=10, expected_model_id="org/m") is True


def test_the_dry_run_prints_the_real_server_commands(tmp_path, monkeypatch, capsys):
    from gbench import cli
    monkeypatch.setattr(cli, "resolve_server_port", lambda environ=None: (18555, "auto"))
    cfg = BenchmarkConfig()
    cfg.remote_endpoint = None
    cfg.selected_gpus = ["GPU-a"]
    cfg.tool_call_parser = "toolp"
    runner = ServingBenchmarkRunner(cfg)
    cli._print_dry_run_plan(cfg, [_model()], [ModelFormat.HF], runner, perf_planned=True,
                            native_planned=True, mm_planned=lambda m: False)
    out = capsys.readouterr().out
    assert "vllm.entrypoints.cli.main serve org/m --port 18555 --host 127.0.0.1" in out
    assert "--enable-auto-tool-choice --tool-call-parser toolp" in out
    assert "http://127.0.0.1:18555/v1" in out


def test_the_offline_gpu_memory_wait_looks_only_at_its_gpus(monkeypatch):
    from gbench.runners import throughput as T
    cfg = BenchmarkConfig()
    cfg.selected_gpus = ["GPU-a", "GPU-b"]
    cfg.tensor_parallel_size, cfg.data_parallel_size = 1, 2
    t = T.ThroughputBenchmarkRunner(cfg)
    t._first_run = False
    calls = []

    def fake_run(cmd, *a, **k):
        calls.append(cmd)
        out = "81920" if "--query-gpu=memory.total" in cmd else "80000"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
    monkeypatch.setattr(subprocess, "run", fake_run)
    t._wait_for_gpu_memory(max_wait_seconds=5)
    assert calls and all("--id=GPU-a" in c for c in calls)


def test_one_golden_runner_resolves_each_models_own_server(tmp_path, endpoint_factory):
    """The CLI builds one Golden Set runner for every model of a run, each model on its own
    server: the second model must not be sent the first model's id."""
    e1 = endpoint_factory(models=["org/one"])
    e2 = endpoint_factory(models=["org/two"])
    cfg = BenchmarkConfig(results_dir=tmp_path / "r")
    cfg.remote_endpoint = None
    g = golden_mod.GoldenBenchmarkRunner(cfg)
    task = {"id": "t", "category": "x",
            "conversation": {"messages": [{"role": "user", "content": "say ok"}]},
            "golden_truth": {"match_type": "contains_all", "expected_outputs": ["ok"]}}
    cfg.local_endpoint = f"{e1.base}/v1"
    s1 = g.run(_model("org/one"), tasks=[task])
    cfg.local_endpoint = f"{e2.base}/v1"
    s2 = g.run(_model("org/two"), tasks=[task])
    assert (s1["requested_model"], s2["requested_model"]) == ("org/one", "org/two")
    assert s1["status"] == s2["status"] == golden_mod.STATUS_PASSED


def test_serving_aggregate_reports_observed_ignore_eos_and_median_actual_input(tmp_path):
    """Issue 10: ignore_eos is observed (n/N requests reached max_tokens), never stamped;
    input_length_actual is the median over iterations; per-request samples are saved."""
    r = _runner(tmp_path, input_lengths=[128], output_lengths=[16], num_iterations=3)
    r.config.log_manager.get_serving_result_path = lambda *a, **k: tmp_path / "serving.json"
    r._is_vllm = False
    it = lambda act, lens: {"raw_ttfts_ms": [50.0] * len(lens),                # noqa: E731
                            "raw_itl_blocks_ms": [[10.0] * 15] * len(lens),
                            "raw_e2els_ms": [200.0] * len(lens), "output_lens": lens,
                            "output_lens_usage": lens, "input_length_actual": act,
                            "completed": len(lens), "failed": 0, "empty": 0,
                            "total": len(lens), "duration": 1.0,
                            "request_throughput": 3.0, "output_throughput": 48.0}
    agg = r._aggregate_serving_results(
        [it(150, [16, 16]), it(170, [16, 9]), it(160, [16, 16])], _model(), ModelFormat.REMOTE,
        1, multimodal=False)
    assert agg["ignore_eos_honored"] is False
    assert agg["ignore_eos_observed"] == {"n": 5, "of": 6}
    assert agg["input_length_actual"] == 160 and agg["input_length_template_tokens"] == 32
    assert agg["iterations_ok"] == 3 and agg["iterations_requested"] == 3
    assert len(agg["samples"]["ttft_ms"]) == 6
    # a failed iteration (no samples) is not in the pooled numbers or their labels
    agg2 = r._aggregate_serving_results(
        [it(150, [16, 16]), {"error": "boom"}, it(160, [16, 16])], _model(), ModelFormat.REMOTE,
        1, multimodal=False)
    assert agg2["iterations_ok"] == 2 and agg2["tpot_kind"] == "per_token"


def test_serving_aggregate_carries_peak_in_flight(tmp_path):
    r = _runner(tmp_path, input_lengths=[128], output_lengths=[16], num_iterations=2)
    r.config.log_manager.get_serving_result_path = lambda *a, **k: tmp_path / "serving.json"
    r._is_vllm = False
    it = lambda peak: {"raw_ttfts_ms": [50.0], "raw_itl_blocks_ms": [[10.0] * 15],  # noqa: E731
                       "raw_e2els_ms": [200.0], "output_lens": [16], "output_lens_usage": [16],
                       "completed": 1, "failed": 0, "empty": 0, "total": 1, "duration": 1.0,
                       "request_throughput": 1.0, "output_throughput": 16.0,
                       "peak_in_flight": peak, "client_loop_lag_max_ms": 3.0}
    agg = r._aggregate_serving_results([it(64), it(60)], _model(), ModelFormat.REMOTE, 64,
                                       multimodal=False)
    assert agg["peak_in_flight"] == 64 and agg["peak_in_flight_per_iteration"] == [64, 60]
    assert agg["batch_size_reached"] is True and agg["client_loop_lag_max_ms"] == 3.0


def test_real_dataset_repeatability_is_judged_on_tokens_per_second(tmp_path):
    """Live finding (2026-10-15): chat-like replays different ShareGPT prompts in each
    iteration, so requests/s moved 10% with the reply lengths while tokens/s held at 0.5%;
    the summary printed FAIL. A real dataset is judged on tokens/s and labelled by name."""
    r = _runner(tmp_path, input_lengths=[128], output_lengths=[16], num_iterations=3)
    r.config.log_manager.get_serving_result_path = lambda *a, **k: tmp_path / "serving.json"
    r._is_vllm = False
    r.config.dataset = "sharegpt"
    it = lambda rps, tps: {"raw_ttfts_ms": [50.0], "raw_itl_blocks_ms": [[10.0] * 15],   # noqa: E731
                           "raw_e2els_ms": [200.0], "output_lens": [16], "output_lens_usage": [16],
                           "completed": 1, "failed": 0, "empty": 0, "total": 1, "duration": 1.0,
                           "request_throughput": rps, "output_throughput": tps}
    agg = r._aggregate_serving_results([it(1.0, 260.0), it(1.25, 261.0), it(1.05, 262.5)],
                                       _model(), ModelFormat.REMOTE, 1, multimodal=False)
    assert agg["request_throughput_repeatability_valid"] is None
    assert agg["repeatability_metric"] == "output_throughput" and agg["repeatability_valid"] is True
    assert agg["workload_dataset"] == "sharegpt"
    r.config.dataset = "random"
    agg2 = r._aggregate_serving_results([it(1.0, 260.0), it(1.25, 261.0), it(1.05, 262.5)],
                                        _model(), ModelFormat.REMOTE, 1, multimodal=False)
    assert agg2["repeatability_valid"] is False and "workload_dataset" not in agg2
