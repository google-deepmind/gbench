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

"""The endpoint key between gbench and the server it launches, the bind address of that server,
and a results manager that creates nothing.

No vLLM runs: server launches stop at a subprocess.Popen stub, vLLM's benchmark() is replaced by
a stub module, and the endpoints are in-process HTTP servers on OS-assigned ports.
"""

import asyncio
import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import pytest

import gbench.utils as gbench_utils
from gbench.core.config import BenchmarkConfig
from gbench.core.models import ModelConfig, ModelFormat
from gbench.runners import serving
from gbench.runners.serving import ServingBenchmarkRunner
from gbench.utils import endpoint as ep
from gbench.utils import log_manager

KEY = "sekrit-key-1234"
OTHER = "other-key-5678"


class _LM:
    def __init__(self, tmp_path):
        self.tmp = tmp_path

    def get_server_log_path(self, *a):
        return str(self.tmp / "server.log")


def _model(name="org/m"):
    return ModelConfig(short_name=name.rsplit("/", 1)[-1], hf_model_id=name, total_params_b=4.0)


def _runner(tmp_path, **cfg_kw):
    cfg = BenchmarkConfig()
    cfg.max_model_len_explicit = None
    cfg.log_manager = _LM(tmp_path)
    for k, v in cfg_kw.items():
        setattr(cfg, k, v)
    return ServingBenchmarkRunner(cfg)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("VLLM_API_KEY", "OPENAI_API_KEY", "GBENCH_SERVER_PORT", "GBENCH_SERVER_HOST",
                 "GBENCH_VLLM_BIN", "GBENCH_VLLM_SERVE_ARGS"):
        monkeypatch.delenv(name, raising=False)
    # LogManager exports the run directory; keep the test process's value as it was.
    monkeypatch.setenv("GBENCH_RESULTS_DIR", "unchanged-by-the-test")


def _popen_env(runner, context="native"):
    """The environment _start_server hands to Popen (the launch is stopped there)."""
    seen = []

    def fake_popen(cmd, *a, **k):
        seen.append((list(cmd), k))
        raise RuntimeError("stop before launch")

    with mock.patch.object(serving, "resolve_server_port", lambda environ=None: (18001, "auto")), \
            mock.patch("subprocess.Popen", fake_popen):
        assert runner._start_server(_model(), ModelFormat.HF, context=context) is False
    assert seen, "Popen was never reached"
    return seen[0]


class _KeyedEndpoint:
    """/v1/models answered only with `Authorization: Bearer <key>` when `key` is set, the way
    vllm serve's middleware enforces a non-empty VLLM_API_KEY; /health is never protected."""

    def __init__(self, key):
        self.key, self.auth = key, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):  # noqa: N802
                if self.path == "/health":
                    status, body = 200, {}
                else:
                    got = self.headers.get("Authorization")
                    outer.auth.append(got)
                    ok = not outer.key or got == f"Bearer {outer.key}"
                    status, body = (200, {"data": [{"id": "org/m"}]}) if ok else (401, {})
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


# --------------------------------------------------------------------------- #
# the local text serving benchmark sends the endpoint key
# --------------------------------------------------------------------------- #
def _benchmark_kwargs(tmp_path):
    """The keyword arguments the local text path hands vLLM's benchmark() (stubbed)."""
    seen = {}

    async def fake_benchmark(**kw):
        seen.update(kw)
        return {"stub": True}

    fake = types.ModuleType("vllm.benchmarks.serve")
    fake.benchmark = fake_benchmark
    fake.TaskType = types.SimpleNamespace(GENERATION="generation")
    r = _runner(tmp_path)
    r.server_port = 18001
    with mock.patch.dict(sys.modules, {"vllm.benchmarks.serve": fake}), \
            mock.patch.object(gbench_utils, "check_vllm_available", lambda: True):
        asyncio.run(r._run_benchmark_async(_model(), ModelFormat.HF, 1, 1, [], None))
    assert seen["api_url"] == "http://127.0.0.1:18001/v1/completions"
    return seen


def test_the_local_text_benchmark_sends_vllm_api_key_without_openai_api_key(
        tmp_path, monkeypatch):
    # vLLM's request functions send only OPENAI_API_KEY; the documented VLLM_API_KEY, which the
    # launched server enforces, must reach it through extra_headers.
    monkeypatch.setenv("VLLM_API_KEY", KEY)
    assert _benchmark_kwargs(tmp_path)["extra_headers"] == {"Authorization": f"Bearer {KEY}"}


def test_the_local_text_benchmark_header_replaces_a_different_openai_api_key(
        tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", KEY)
    monkeypatch.setenv("OPENAI_API_KEY", OTHER)
    assert _benchmark_kwargs(tmp_path)["extra_headers"] == {"Authorization": f"Bearer {KEY}"}


def test_the_local_text_benchmark_sends_no_header_without_a_key(tmp_path, monkeypatch):
    assert _benchmark_kwargs(tmp_path)["extra_headers"] is None
    monkeypatch.setenv("VLLM_API_KEY", "EMPTY")
    assert _benchmark_kwargs(tmp_path)["extra_headers"] is None


# --------------------------------------------------------------------------- #
# the launched server enforces exactly the key gbench sends
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("environ, enforced", [
    ({}, None),
    ({"VLLM_API_KEY": "EMPTY"}, None),                   # a placeholder: no key either side
    ({"VLLM_API_KEY": "dummy", "OPENAI_API_KEY": OTHER}, None),
    ({"VLLM_API_KEY": f"  {KEY}\n"}, KEY),               # stripped, as gbench sends it
    ({"VLLM_API_KEY": KEY, "OPENAI_API_KEY": OTHER}, KEY),
    ({"OPENAI_API_KEY": OTHER}, None),                   # the alias is never made the server's key
])
def test_the_launched_server_enforces_exactly_the_key_gbench_sends(
        tmp_path, monkeypatch, environ, enforced):
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    runner = _runner(tmp_path)
    _cmd, kwargs = _popen_env(runner)
    server_key = kwargs["env"].get("VLLM_API_KEY")
    assert server_key == enforced
    assert runner.server_key_enforced is bool(enforced)
    assert "--api-key" not in _cmd
    # vllm serve enforces a non-empty VLLM_API_KEY; gbench's preflight must pass against it.
    e = _KeyedEndpoint(server_key)
    try:
        ok, msg, _ = ep.verify_endpoint_functional(e.base, model_id="org/m")
        assert ok is True, msg
    finally:
        e.close()


def test_a_placeholder_key_used_to_lock_gbench_out_of_its_own_server(tmp_path, monkeypatch):
    # What the server would have enforced before: the raw VLLM_API_KEY it inherited.
    monkeypatch.setenv("VLLM_API_KEY", "EMPTY")
    e = _KeyedEndpoint("EMPTY")
    try:
        ok, msg, _ = ep.verify_endpoint_functional(e.base)
        assert ok is False and "401" in msg
    finally:
        e.close()
    _cmd, kwargs = _popen_env(_runner(tmp_path))
    assert "VLLM_API_KEY" not in kwargs["env"]


def test_the_server_record_names_the_bind_reason_and_key_enforcement_but_never_the_key(
        tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", KEY)
    runner = _runner(tmp_path)
    _popen_env(runner)
    rec = runner.server_record()
    assert rec["host"] == "127.0.0.1"
    assert "no eval suite" in rec["host_source"]
    assert rec["endpoint_key_enforced"] is True
    assert KEY not in json.dumps(rec, default=str)


# --------------------------------------------------------------------------- #
# the bind address of the quality / Golden Set / eval server
# --------------------------------------------------------------------------- #
def test_the_quality_and_golden_server_binds_loopback_when_no_eval_suite_is_selected(tmp_path):
    for evals in (None, []):
        r = _runner(tmp_path, evals=evals)
        cmd = r.build_server_cmd(_model(), ModelFormat.HF, context="native", port=18001)
        assert cmd[cmd.index("--host") + 1] == "127.0.0.1"


def test_the_eval_server_still_binds_all_interfaces_for_bridge_suites(tmp_path):
    r = _runner(tmp_path, evals=["all"])
    cmd = r.build_server_cmd(_model(), ModelFormat.HF, context="native", port=18001)
    assert "--host" not in cmd
    host, source = r._server_host_choice("native")
    assert host is None and "bridge" in source


def test_the_host_override_wins_for_every_server(tmp_path, monkeypatch):
    monkeypatch.setenv("GBENCH_SERVER_HOST", "0.0.0.0")
    for evals in (None, ["gsm8k"]):
        r = _runner(tmp_path, evals=evals)
        for ctx in ("perf", "native"):
            assert r._server_host_choice(ctx) == ("0.0.0.0", "GBENCH_SERVER_HOST")


def test_a_launched_server_records_its_bind_address(tmp_path):
    runner = _runner(tmp_path, evals=["gsm8k"])
    _popen_env(runner)
    assert runner.server_host is None and "bridge" in runner.server_host_source
    perf = _runner(tmp_path, evals=["gsm8k"])
    cmd, _kw = _popen_env(perf, context="perf")
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1" == perf.server_host


# --------------------------------------------------------------------------- #
# a results manager that creates nothing (a dry run)
# --------------------------------------------------------------------------- #
def test_a_results_manager_without_directories_creates_nothing(tmp_path):
    import os
    base = tmp_path / "results"
    lm = log_manager.LogManager(base, create_dirs=False)
    assert not base.exists()
    assert lm.results_dir.parent == base and lm.timestamp == lm.results_dir.name
    assert lm.creates_dirs is False
    assert os.environ["GBENCH_RESULTS_DIR"] == "unchanged-by-the-test"


def test_a_results_manager_without_directories_keeps_the_reuse_rule(tmp_path):
    run = tmp_path / "09-28-120000"
    run.mkdir()
    (run / "metadata.json").write_text("{}")
    lm = log_manager.LogManager(run, create_dirs=False)
    assert lm.results_dir == run
    assert sorted(p.name for p in run.iterdir()) == ["metadata.json"]


def test_the_default_results_manager_still_creates_its_run_directory(tmp_path):
    import os
    lm = log_manager.LogManager(tmp_path / "results")
    assert lm.creates_dirs is True
    for sub in ("performance", "quality", "logs"):
        assert (lm.results_dir / sub).is_dir()
    assert os.environ["GBENCH_RESULTS_DIR"] == str(lm.results_dir)
