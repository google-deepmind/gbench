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

"""Deterministic unit tests for the ui_control_osworld canonical OSWorld harness (no VM / KVM / model)."""

import json
from pathlib import Path
import os
import sys
import types

import pytest

from gbench.runners.eval_suites import ui_control_osworld as U

_DOCKER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gbench", "docker")
sys.path.insert(0, _DOCKER_DIR)
import ui_control_osworld_run as RUN  # noqa: E402


# --------------------------------------------------------------------------- #
# Headline scoring + endpoint.
# --------------------------------------------------------------------------- #
def test_compute_score():
    assert U.compute_ui_control_osworld_score({"success_rate": 0.12})["success_rate"] == pytest.approx(0.12)
    assert U.compute_ui_control_osworld_score({}) is None
    assert U.compute_ui_control_osworld_score({"success_rate": None}) is None


def test_model_endpoint(monkeypatch):
    monkeypatch.delenv("GBENCH_OSWORLD_MODEL_ENDPOINT", raising=False)
    assert U._model_endpoint("http://127.0.0.1:8000") == "http://127.0.0.1:8000/v1"
    assert U._model_endpoint("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"
    monkeypatch.setenv("GBENCH_OSWORLD_MODEL_ENDPOINT", "http://x:1/v1")
    assert U._model_endpoint("http://127.0.0.1:8000") == "http://x:1/v1"


# --------------------------------------------------------------------------- #
# Gate (raises infra_required; the binding blocker is /dev/kvm).
# --------------------------------------------------------------------------- #
def test_gate_missing_docker():
    import unittest.mock as mock
    with mock.patch.object(U.shutil, "which", return_value=None):
        ok, reason = U.check_ui_control_osworld_prerequisites()
    assert not ok and "docker CLI not found" in reason


def test_gate_kvm_then_qcow2_then_image(tmp_path, monkeypatch):
    monkeypatch.setattr(U.shutil, "which", lambda _x: "/usr/bin/docker")
    # docker answers; the image's launcher label is the current one (image_launcher_version)
    monkeypatch.setattr(U.subprocess, "run",
                        lambda *a, **k: type("P", (), {
                            "returncode": 0, "stderr": "",
                            "stdout": U.LAUNCHER_VERSION if "--format" in a[0] else ""})())
    monkeypatch.setattr(U, "_docker_image_exists", lambda _i: True)
    real_exists = os.path.exists
    kvm = {"present": False}
    monkeypatch.setattr(U.os.path, "exists",
                        lambda p: True if p == "/dev/kvm" and kvm["present"] else (False if p == "/dev/kvm" else real_exists(p)))
    # /dev/kvm missing -> hard block
    ok, reason = U.check_ui_control_osworld_prerequisites()
    assert not ok and "/dev/kvm" in reason
    # kvm present, but no VM dir
    kvm["present"] = True
    monkeypatch.delenv("GBENCH_OSWORLD_VM_DIR", raising=False)
    ok, reason = U.check_ui_control_osworld_prerequisites()
    assert not ok and "GBENCH_OSWORLD_VM_DIR" in reason
    # VM dir set but no qcow2
    monkeypatch.setenv("GBENCH_OSWORLD_VM_DIR", str(tmp_path))
    ok, reason = U.check_ui_control_osworld_prerequisites()
    assert not ok and "Ubuntu.qcow2" in reason
    # provide qcow2 -> passes (image mocked present, no GEMINI needed)
    (tmp_path / "docker_vm_data").mkdir()
    (tmp_path / "docker_vm_data" / "Ubuntu.qcow2").write_text("x")
    ok, reason = U.check_ui_control_osworld_prerequisites()
    assert ok and reason == ""


# --------------------------------------------------------------------------- #
# Launcher: model-name rewrite (the riskiest adapter - vLLM rejects the routing alias).
# --------------------------------------------------------------------------- #
def test_install_model_rewrite(monkeypatch):
    # Fake mm_agents.agent with a recording requests.post, injected so RUN can monkeypatch it.
    captured = {}

    def fake_post(url, *a, **k):
        captured["url"] = url
        captured["json"] = k.get("json")
        return "resp"

    fake_requests = types.SimpleNamespace(post=fake_post)
    fake_agent = types.ModuleType("mm_agents.agent")
    fake_agent.requests = fake_requests
    fake_pkg = types.ModuleType("mm_agents")
    monkeypatch.setitem(sys.modules, "mm_agents", fake_pkg)
    monkeypatch.setitem(sys.modules, "mm_agents.agent", fake_agent)

    RUN._install_model_rewrite("google/gemma-4-26B-A4B-it")
    # A chat/completions POST with the routing alias -> model rewritten to the served name.
    fake_agent.requests.post("http://127.0.0.1:8000/v1/chat/completions",
                             json={"model": "gpt-4o", "messages": []})
    assert captured["json"]["model"] == "google/gemma-4-26B-A4B-it"
    # A non-alias / non-chat POST is untouched.
    fake_agent.requests.post("http://127.0.0.1:8000/v1/embeddings", json={"model": "other"})
    assert captured["json"]["model"] == "other"


# --------------------------------------------------------------------------- #
# Launcher: subset meta build + results parsing.
# --------------------------------------------------------------------------- #
def _make_osworld_dir(tmp_path, meta):
    d = tmp_path / "osw"
    (d / "evaluation_examples").mkdir(parents=True)
    (d / "evaluation_examples" / "test_all.json").write_text(json.dumps(meta))
    return str(d)


def test_build_subset_meta_domain_and_limit(tmp_path, monkeypatch):
    osw = _make_osworld_dir(tmp_path, {"os": ["a", "b", "c"], "calc": ["d", "e"]})
    monkeypatch.setattr(RUN, "OSWORLD_DIR", osw)
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path))
    # domain filter
    monkeypatch.setenv("OSWORLD_DOMAIN", "calc")
    monkeypatch.delenv("OSWORLD_LIMIT", raising=False)
    _p, sel, _selection = RUN._build_subset_meta()
    assert sel == {"calc": ["d", "e"]}
    # limit across domains
    monkeypatch.setenv("OSWORLD_DOMAIN", "all")
    monkeypatch.setenv("OSWORLD_LIMIT", "2")
    _p, sel, _selection = RUN._build_subset_meta()
    assert sum(len(v) for v in sel.values()) == 2


def test_summarize_results_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path))
    results = str(tmp_path / "results")
    base = os.path.join(results, "pyautogui", "screenshot", "gpt-4o")
    # os: a=1.0, b=0.0 ; calc: d scored 1.0, e MISSING (counts as 0)
    for domain, ex, val in [("os", "a", "1.0"), ("os", "b", "0.0"), ("calc", "d", "1.0")]:
        p = os.path.join(base, domain, ex)
        os.makedirs(p)
        with open(os.path.join(p, "result.txt"), "w") as f:
            f.write(val)
    selected = {"os": ["a", "b"], "calc": ["d", "e"]}
    s = RUN._summarize("pyautogui", "gpt-4o", "screenshot", results, selected)
    assert s["n_selected"] == 4 and s["n_scored"] == 3 and s["n_missing"] == 1
    # mean over ALL selected (1.0 + 0.0 + 1.0 + 0[missing]) / 4
    assert s["success_rate"] == pytest.approx(0.5)
    assert s["per_domain"]["calc"]["n"] == 2
