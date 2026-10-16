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

"""Deterministic unit tests for the gaia2 canonical Meta-ARE harness (no Docker / model / real judge)."""

import json
import os
import sys

import pytest

from gbench.runners.eval_suites import gaia2 as G

_DOCKER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gbench", "docker")
sys.path.insert(0, _DOCKER_DIR)
import gaia2_run as RUN  # noqa: E402


# --------------------------------------------------------------------------- #
# Headline scoring (macro preferred, micro fallback).
# --------------------------------------------------------------------------- #
def test_compute_score_macro_micro_none():
    s = G.compute_gaia2_score({"macro_success_rate": 0.37, "micro_success_rate": 0.41})
    assert s["headline"] == pytest.approx(0.37)         # macro is the GAIA2 Overall
    s2 = G.compute_gaia2_score({"macro_success_rate": None, "micro_success_rate": 0.41})
    assert s2["headline"] == pytest.approx(0.41)        # micro fallback
    assert G.compute_gaia2_score({"macro_success_rate": None, "micro_success_rate": None}) is None
    assert G.compute_gaia2_score({}) is None


# --------------------------------------------------------------------------- #
# Model endpoint (--network host -> 127.0.0.1, /v1 ensured).
# --------------------------------------------------------------------------- #
def test_model_endpoint(monkeypatch):
    monkeypatch.delenv("GBENCH_GAIA2_MODEL_ENDPOINT", raising=False)
    assert G._task_reachable_endpoint("http://127.0.0.1:8000") == "http://127.0.0.1:8000/v1"
    assert G._task_reachable_endpoint("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"
    monkeypatch.setenv("GBENCH_GAIA2_MODEL_ENDPOINT", "http://x:1/v1")
    assert G._task_reachable_endpoint("http://127.0.0.1:8000") == "http://x:1/v1"


# --------------------------------------------------------------------------- #
# Prerequisite gate (raises infra_required, never skips).
# --------------------------------------------------------------------------- #
def test_gate_missing_docker():
    import unittest.mock as mock
    with mock.patch.object(G.shutil, "which", return_value=None):
        ok, reason = G.check_gaia2_prerequisites()
    assert not ok and "docker CLI not found" in reason


def _gate_ready(monkeypatch):
    monkeypatch.setattr(G.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(G.subprocess, "run",
                        lambda *a, **k: type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    monkeypatch.setattr(G, "_docker_image_exists", lambda _i: True)
    monkeypatch.setattr(G, "_image_accounts_harness_errors", lambda _i: True)


def test_gate_requires_gemini_then_validates(monkeypatch):
    _gate_ready(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    ok, reason = G.check_gaia2_prerequisites()
    assert not ok and "GEMINI_API_KEY" in reason
    # present but rejected -> hard-error
    monkeypatch.setenv("GEMINI_API_KEY", "bad")
    monkeypatch.delenv("GBENCH_GAIA2_SKIP_KEY_VALIDATION", raising=False)
    monkeypatch.setattr(G, "_gemini_key_valid", lambda _k: (False, "HTTP 400"))
    ok, reason = G.check_gaia2_prerequisites()
    assert not ok and "rejected by the Gemini API" in reason
    # valid -> pass
    monkeypatch.setattr(G, "_gemini_key_valid", lambda _k: (True, ""))
    ok, reason = G.check_gaia2_prerequisites()
    assert ok and reason == ""
    # skip-validation env bypasses the live ping
    monkeypatch.setenv("GBENCH_GAIA2_SKIP_KEY_VALIDATION", "1")

    def _boom(_k):
        raise AssertionError("validation must be skipped")
    monkeypatch.setattr(G, "_gemini_key_valid", _boom)
    ok, reason = G.check_gaia2_prerequisites()
    assert ok and reason == ""


def test_gemini_key_valid_classification(monkeypatch):
    import urllib.error

    def _mk(status=None, http_code=None, body=b""):
        def _open(req, timeout=0):
            if http_code is not None:
                raise urllib.error.HTTPError(req.full_url, http_code, "e", {}, __import__("io").BytesIO(body))
            return type("R", (), {"status": status, "__enter__": lambda s: s, "__exit__": lambda s, *a: False})()
        return _open

    monkeypatch.setattr(G.urllib.request, "urlopen", _mk(status=200))
    assert G._gemini_key_valid("k")[0] is True
    monkeypatch.setattr(G.urllib.request, "urlopen", _mk(http_code=401))
    assert G._gemini_key_valid("k")[0] is False
    monkeypatch.setattr(G.urllib.request, "urlopen",
                        _mk(http_code=400, body=b'{"error":{"message":"Please pass a valid API key"}}'))
    assert G._gemini_key_valid("k")[0] is False           # real Gemini bad-key shape
    monkeypatch.setattr(G.urllib.request, "urlopen", _mk(http_code=500))
    assert G._gemini_key_valid("k")[0] is True            # inconclusive -> pass


# --------------------------------------------------------------------------- #
# Launcher: are-benchmark command construction + stats parsing.
# --------------------------------------------------------------------------- #
def test_model_arg_openai_prefix(monkeypatch):
    monkeypatch.setenv("GBENCH_MODEL_NAME", "google/gemma-4-26B-A4B-it")
    assert RUN._model_arg() == "openai/google/gemma-4-26B-A4B-it"
    monkeypatch.setenv("GBENCH_MODEL_NAME", "openai/already")
    assert RUN._model_arg() == "openai/already"           # not double-prefixed


def test_base_cmd_wires_local_provider_and_judge(monkeypatch, tmp_path):
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_MODEL_NAME", "m")
    monkeypatch.setenv("GBENCH_MODEL_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.delenv("GAIA2_JUDGE_ENDPOINT", raising=False)
    monkeypatch.delenv("GAIA2_JUDGE_MODEL", raising=False)
    monkeypatch.setenv("GAIA2_JUDGE_PORT", "18790")
    cmd = RUN._base_cmd("run")
    assert cmd[:2] == ["are-benchmark", "run"]
    assert "--provider" in cmd and cmd[cmd.index("--provider") + 1] == "local"
    assert cmd[cmd.index("--model") + 1] == "openai/m"
    assert cmd[cmd.index("--endpoint") + 1] == "http://127.0.0.1:8000/v1"
    assert cmd[cmd.index("--judge_provider") + 1] == "local"
    assert cmd[cmd.index("--judge_endpoint") + 1] == "http://127.0.0.1:18790/v1"
    assert cmd[cmd.index("--judge_model") + 1] == "openai/gbench-cascade"
    assert cmd[cmd.index("--output_dir") + 1] == str(tmp_path)


def test_summarize_parses_benchmark_stats(tmp_path, monkeypatch):
    # ARE's benchmark_stats.json: success rates in percent; one success_rate per capability.
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path))
    stats = {
        "metadata": {"model": "openai/m"},
        "statistics": {
            "global": {"macro_success_rate": 42.0, "micro_success_rate": 40.0,
                       "total_scenarios": 800, "total_runs": 800, "validated_runs": 780,
                       "success_runs": 320, "no_validation_runs": 20, "exception_runs": 0,
                       "pass_at_k_percent": 42.0, "pass_k_percent": 30.0},
            "per_capability": {
                "search": {"success_rate": 50.0, "validated_runs": 160, "total_scenarios": 160},
                "time": {"success_rate": 34.0, "validated_runs": 160, "total_scenarios": 160},
            },
        },
    }
    (tmp_path / "benchmark_stats.json").write_text(json.dumps(stats), encoding="utf-8")
    s = RUN._summarize()
    assert s["macro_success_rate"] == pytest.approx(42.0)
    assert s["micro_success_rate"] == pytest.approx(40.0)
    assert s["total_scenarios"] == 800 and s["no_validation_runs"] == 20
    assert set(s["per_capability"]) == {"search", "time"}
    assert s["per_capability"]["search"]["success_rate"] == pytest.approx(50.0)


def test_summarize_none_when_no_stats(tmp_path, monkeypatch):
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path))
    assert RUN._summarize() is None
