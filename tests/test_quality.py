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

"""Unit tests for QualityBenchmarkRunner."""

import json
import os
import subprocess
from pathlib import Path
from unittest import mock
import pytest

from gbench.core.config import BenchmarkConfig
from gbench.core.models import ModelConfig, ModelFormat
from gbench.runners import quality as Q
from gbench.runners.quality import QualityBenchmarkRunner


@pytest.fixture(autouse=True)
def _served_listing(monkeypatch):
    """The model endpoint's /v1/models, stubbed (no network): it serves the model under test
    under its HuggingFace id, as gbench's own server and these endpoints do."""
    monkeypatch.setattr("gbench.utils.endpoint.list_served_models",
                        lambda base_url, timeout=5.0, headers=None:
                        [{"id": "google/gemma-4-E2B-it"}])


@pytest.fixture
def base_config(tmp_path):
    config = BenchmarkConfig()
    config.results_dir = tmp_path / "results"
    config.results_dir.mkdir()
    config.gemmaclaw_commit = "main"
    config.initialize()
    return config


@pytest.fixture
def mock_model():
    model = ModelConfig(
        short_name="gemma-4-E2B-it",
        hf_model_id="google/gemma-4-E2B-it",
        total_params_b=5.2,
        is_moe=False,
    )
    return model


#: A Gemini key in the environment of every run(): the runner starts its key relay with it (a
#: real run's preflight check requires one).
GEMINI_ENV = {"GEMINI_API_KEY": "AIzaSyTESTqualityRunner00000000001"}

SUMMARY = {"counts": {"total": 10, "passed": 8, "failed": 2, "blocked": 0}}


def _stub_container(runner, summary=SUMMARY):
    """The suite's container stubbed: the image's metadata, the outputs read back from it, and
    its removal (docker run itself is the mocked subprocess.Popen)."""
    runner._image = "gbench-quality:test"

    def meta(image, dest):
        for pid in ("qa-lab", "openai"):
            (dest / "extensions" / pid).mkdir(parents=True, exist_ok=True)
            (dest / "extensions" / pid / "openclaw.plugin.json").write_text(
                json.dumps({"id": pid}))
        return {"patches": [], "commit": "mocked-commit"}

    runner._image_metadata = meta
    runner._collect_container_outputs = lambda name: {
        "summary": summary, "scenario_summaries": {},
        "entrypoint": {"command_exit": 0, "forwarder_alive_at_end": True},
        "forwarder_errors": []}
    runner._remove_container = mock.MagicMock()


def _fake_docker(seen, wait_side_effect=None):
    """A mocked `docker run`: records its command line, the container's --env and the
    openclaw.json it mounts, and exits 0 (or as `wait_side_effect` says)."""
    def popen(argv, stdin=None, stdout=None, stderr=None, env=None, **kw):
        seen["argv"] = list(argv)
        seen["docker_env"] = dict(env or {})
        seen["env"] = dict(argv[i + 1].split("=", 1) for i, a in enumerate(argv)
                           if a == "--env")
        for i, a in enumerate(argv):
            if a == "--volume" and argv[i + 1].endswith(":/etc/gbench/openclaw.json:ro"):
                seen["config"] = json.loads(Path(argv[i + 1].split(":")[0]).read_text())
        proc = mock.MagicMock()
        if wait_side_effect is not None:
            proc.wait.side_effect = wait_side_effect
        else:
            proc.poll.return_value = 0
            proc.wait.return_value = 0
            proc.returncode = 0
        seen["proc"] = proc
        return proc
    return popen


@mock.patch.dict("os.environ", GEMINI_ENV)
def test_run_local_server(base_config, mock_model, monkeypatch):
    """Test standard execution starting local vLLM."""
    runner = QualityBenchmarkRunner(base_config)

    # Mock preflight and repo setup
    runner._preflight_check = mock.MagicMock(return_value=True)
    runner._setup_gemmaclaw_repo = mock.MagicMock(return_value="mocked-commit")
    runner._start_server = mock.MagicMock(return_value=True)
    runner._cleanup_server = mock.MagicMock()
    _stub_container(runner)
    seen = {}
    monkeypatch.setattr(Q.subprocess, "Popen", _fake_docker(seen))

    # The window lookup is stubbed (no live HTTP): GemmaClaw must be told the SERVED window.
    with mock.patch("gbench.runners.evals._server_max_model_len", return_value=32768):
        result = runner.run(mock_model, ModelFormat.HF)

    # Quality serves the model's native context window, not the perf default of 4096.
    runner._start_server.assert_called_once_with(mock_model, ModelFormat.HF, context="native")
    runner._cleanup_server.assert_called_once()
    runner._remove_container.assert_called_once()

    # openclaw.json: gbench's own server is behind the relay too (the forwarder's URL), with a
    # placeholder where a key would go.
    config_json = seen["config"]
    assert config_json["models"]["providers"]["vllm"]["baseUrl"] == Q.CONTAINER_MODEL_BASE_URL
    assert config_json["models"]["providers"]["vllm"]["apiKey"] == Q.KEY_PLACEHOLDER
    assert config_json["models"]["providers"]["vllm"]["models"][0]["contextWindow"] == 32768
    assert config_json["plugins"]["entries"]["qa-lab"]["enabled"] is True
    assert config_json["plugins"]["entries"]["openai"]["enabled"] is False
    # The google plugin gets the forwarder's base URL, never the Gemini key.
    google = config_json["models"]["providers"]["google"]
    assert google["baseUrl"] == Q.CONTAINER_GEMINI_BASE_URL
    assert google["apiKey"] == Q.KEY_PLACEHOLDER
    assert GEMINI_ENV["GEMINI_API_KEY"] not in json.dumps(config_json)

    cmd = seen["argv"]
    env = seen["env"]
    assert "openclaw" in cmd
    assert "qa" in cmd
    assert "suite" in cmd
    assert cmd[:2] == ["docker", "run"] and cmd[cmd.index("--network") + 1] == "none"
    assert env["OPENCLAW_ENABLE_PRIVATE_QA_CLI"] == "1"
    assert env["VLLM_API_KEY"] == Q.KEY_PLACEHOLDER
    assert "GEMINI_API_KEY" not in env and "GEMINI_API_KEYS" not in env
    assert GEMINI_ENV["GEMINI_API_KEY"] not in json.dumps(cmd + [seen["docker_env"]])
    # the relay token goes on the container's standard input
    token = seen["proc"].stdin.write.call_args.args[0].decode().strip()
    assert token.startswith("gbench-relay-") and token not in json.dumps(cmd)

    assert result["passed_scenarios"] == 8
    assert result["total_scenarios"] == 10
    assert result["pass_rate"] == 80.0
    assert result["model_endpoint_relayed"] is True
    assert result["model_endpoint_key_at_relay"] is False


@mock.patch("urllib.request.urlopen", side_effect=OSError("Metadata unavailable"))
@mock.patch("gbench.runners.quality.subprocess.check_output", autospec=True)
@mock.patch.dict("os.environ", {"VLLM_API_KEY": "", **GEMINI_ENV}, clear=False)
def test_run_remote_endpoint_google(
    mock_check_output,
    mock_urlopen,
    base_config,
    mock_model,
    monkeypatch,
):
    """Test remote endpoint targeting Cloud Run (fetches OIDC token)."""
    base_config.remote_endpoint = "https://gbench-vllm-abc123-uc.a.run.app/v1"
    runner = QualityBenchmarkRunner(base_config)

    runner._preflight_check = mock.MagicMock(return_value=True)
    runner._setup_gemmaclaw_repo = mock.MagicMock(return_value="mocked-commit")
    runner._start_server = mock.MagicMock()
    runner._cleanup_server = mock.MagicMock()
    _stub_container(runner)
    seen = {}
    monkeypatch.setattr(Q.subprocess, "Popen", _fake_docker(seen))

    mock_check_output.return_value = "mocked-oidc-token\n"

    # GCP identity-token auto-auth is opt-in (GBENCH_GCP_AUTH=1); this test exercises that path.
    with mock.patch.dict("os.environ", {"GBENCH_GCP_AUTH": "1"}):
        result = runner.run(mock_model, ModelFormat.HF)

    runner._start_server.assert_not_called()
    runner._cleanup_server.assert_called_once() # Called in finally block

    mock_check_output.assert_called_once_with(
        ["gcloud", "auth", "print-identity-token"], text=True
    )

    # The identity token stays with the key relay: the suite gets the forwarder's URL and a
    # placeholder.
    config_json = seen["config"]
    vllm = config_json["models"]["providers"]["vllm"]
    assert vllm["baseUrl"] == Q.CONTAINER_MODEL_BASE_URL
    assert vllm["apiKey"] == Q.KEY_PLACEHOLDER
    assert "mocked-oidc-token" not in json.dumps(config_json)

    env = seen["env"]
    assert env["VLLM_API_KEY"] == Q.KEY_PLACEHOLDER
    assert all("mocked-oidc-token" not in v for v in env.values())
    assert "mocked-oidc-token" not in json.dumps(seen["argv"] + [seen["docker_env"]])
    assert result["model_endpoint_key_at_relay"] is True

    assert result["passed_scenarios"] == 8
    assert result["total_scenarios"] == 10
    assert result["pass_rate"] == 80.0


@mock.patch("gbench.runners.quality.subprocess.check_output", autospec=True)
def test_run_remote_endpoint_google_with_key_override(
    mock_check_output,
    base_config,
    mock_model,
    monkeypatch,
):
    """Test remote endpoint but VLLM_API_KEY is already set (bypasses gcloud)."""
    base_config.remote_endpoint = "https://gbench-vllm-abc123-uc.a.run.app/v1"
    runner = QualityBenchmarkRunner(base_config)

    runner._preflight_check = mock.MagicMock(return_value=True)
    runner._setup_gemmaclaw_repo = mock.MagicMock(return_value="mocked-commit")
    runner._start_server = mock.MagicMock()
    runner._cleanup_server = mock.MagicMock()
    _stub_container(runner)
    seen = {}
    monkeypatch.setattr(Q.subprocess, "Popen", _fake_docker(seen))

    monkeypatch.setenv("VLLM_API_KEY", "user-override-key")
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_ENV["GEMINI_API_KEY"])

    result = runner.run(mock_model, ModelFormat.HF)

    mock_check_output.assert_not_called()
    runner._cleanup_server.assert_called_once()

    # The endpoint's own key stays with the key relay.
    config_json = seen["config"]
    assert config_json["models"]["providers"]["vllm"]["apiKey"] == Q.KEY_PLACEHOLDER
    # A *.run.app endpoint is asked for the model's HF id, like any other endpoint, not its
    # registry short name.
    assert [m["id"] for m in config_json["models"]["providers"]["vllm"]["models"]] == \
        [mock_model.hf_model_id]
    assert "user-override-key" not in json.dumps(config_json)
    assert all("user-override-key" not in v for v in seen["env"].values())
    assert "user-override-key" not in json.dumps(seen["argv"] + [seen["docker_env"]])

    assert result["passed_scenarios"] == 8
    assert result["total_scenarios"] == 10
    assert result["pass_rate"] == 80.0


@mock.patch("gbench.runners.quality.subprocess.check_output", autospec=True)
def test_run_remote_endpoint_google_no_auto_auth_without_optin(
    mock_check_output,
    base_config,
    mock_model,
    monkeypatch,
):
    """A GCP-looking endpoint with no key and GBENCH_GCP_AUTH unset must NOT shell out to gcloud
    or the metadata server (auto-auth is opt-in); the relay then forwards without a key."""
    base_config.remote_endpoint = "https://gbench-vllm-abc123-uc.a.run.app/v1"
    runner = QualityBenchmarkRunner(base_config)

    runner._preflight_check = mock.MagicMock(return_value=True)
    runner._setup_gemmaclaw_repo = mock.MagicMock(return_value="mocked-commit")
    runner._start_server = mock.MagicMock()
    runner._cleanup_server = mock.MagicMock()
    _stub_container(runner)
    seen = {}
    monkeypatch.setattr(Q.subprocess, "Popen", _fake_docker(seen))

    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.delenv("GBENCH_GCP_AUTH", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_ENV["GEMINI_API_KEY"])

    result = runner.run(mock_model, ModelFormat.HF)

    mock_check_output.assert_not_called()   # no gcloud subprocess when opt-in is off
    assert seen["config"]["models"]["providers"]["vllm"]["apiKey"] == Q.KEY_PLACEHOLDER
    assert result["model_endpoint_key_at_relay"] is False
    assert result["pass_rate"] == 80.0


@mock.patch.dict("os.environ", GEMINI_ENV)
def test_run_timeout_kills_process(base_config, mock_model, monkeypatch):
    """Test that timeout kills the running process and raises RuntimeError."""
    runner = QualityBenchmarkRunner(base_config)

    runner._preflight_check = mock.MagicMock(return_value=True)
    runner._setup_gemmaclaw_repo = mock.MagicMock(return_value="mocked-commit")
    runner._start_server = mock.MagicMock(return_value=True)
    runner._cleanup_server = mock.MagicMock()
    _stub_container(runner)
    seen = {}
    monkeypatch.setattr(Q.subprocess, "Popen", _fake_docker(
        seen, wait_side_effect=[subprocess.TimeoutExpired(cmd=["openclaw"], timeout=3600), -9]))
    docker_calls = []
    monkeypatch.setattr(Q.subprocess, "run", lambda cmd, **kw: docker_calls.append(cmd))

    with pytest.raises(RuntimeError, match="gemmaclaw suite timed out"):
        runner.run(mock_model, ModelFormat.HF)

    mock_process = seen["proc"]
    mock_process.kill.assert_called_once()
    assert mock_process.wait.call_count == 2
    name = seen["argv"][seen["argv"].index("--name") + 1]
    assert ["docker", "kill", name] in docker_calls      # the container is stopped too
    runner._remove_container.assert_called_once_with(name)
    runner._cleanup_server.assert_called_once()


def test_node_version_parses_and_handles_missing():
    """_node_version returns (0,0) for a missing/None binary and a parsed
    (major, minor) for a real one, so the nvm-upgrade gate can compare it."""
    import shutil
    from gbench.runners.quality import _node_version
    assert _node_version(None) == (0, 0)
    assert _node_version("/nonexistent/definitely-not-node") == (0, 0)
    real = shutil.which("node")
    if real:
        mj, mn = _node_version(real)
        assert isinstance(mj, int) and isinstance(mn, int)


def test_git_rewrite_safe_env_blanks_global_config(base_config, monkeypatch):
    """The quality git env neutralizes the host github HTTPS->SSH rewrite by
    default (mirrors the swe_lancer fix) and forbids interactive git prompts."""
    monkeypatch.delenv("GBENCH_QUALITY_GIT_CONFIG_GLOBAL", raising=False)
    monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)
    runner = QualityBenchmarkRunner(base_config)
    env = runner._git_rewrite_safe_env()
    assert env["GIT_CONFIG_GLOBAL"] == os.devnull
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert "BatchMode=yes" in env["GIT_SSH_COMMAND"]


def test_git_rewrite_safe_env_override(base_config, monkeypatch):
    """A host that legitimately needs its rewrites can point the knob at a real
    gitconfig instead of /dev/null."""
    monkeypatch.setenv("GBENCH_QUALITY_GIT_CONFIG_GLOBAL", "/custom/gitconfig")
    runner = QualityBenchmarkRunner(base_config)
    env = runner._git_rewrite_safe_env()
    assert env["GIT_CONFIG_GLOBAL"] == "/custom/gitconfig"
