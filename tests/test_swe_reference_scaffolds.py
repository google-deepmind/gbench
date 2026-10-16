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
"""SWE family: the leaderboards' reference scaffolds, the per-setting prerequisite rule, the
relay's path allow-list and the credential-free agent environments.

* swe_bench_live: the Python and MultiLang tabs run SWE-agent-for-eval (the fork SWE-bench-Live's
  evaluation README names) from its local image, its README's rollout command; a Windows tab
  whose prerequisite is missing is reported infra_required in its own entry while the other
  tabs run; Win-Agent and the harnesses get no credential variable.
* multi_swe_bench: MSWE-agent from its local image, one language at a time, with the published
  instance images, the leaderboard runs' 50-step limit and MSWE-agent's own sampling.
* swebench_common.EndpointRelay: only the allowed endpoint paths are forwarded.
* swe_lancer: the module header states its sampling contract.

Hermetic: no container, no docker daemon, no network beyond loopback test servers; every cache
and output is under tmp_path.
"""

import ast
import http.server
import importlib.util
import json
import os
import socket
import sys
import threading
import types

import pytest

from gbench.runners.eval_suites import multi_swe_bench as M
from gbench.runners.eval_suites import swe_agentic as A
from gbench.runners.eval_suites import swe_bench_live as LV
from gbench.runners.eval_suites import swe_bench_pro as P
from gbench.runners.eval_suites import swe_lancer as SL
from gbench.runners.eval_suites import swebench_common as C


@pytest.fixture(autouse=True)
def _no_metadata_probe(monkeypatch):
    """The cloud metadata server gate (swebench_common.require_task_network_blocked) starts probe
    containers; tests/test_guard_swe_agents.py covers it, here it is stubbed."""
    monkeypatch.setattr(C, "require_task_network_blocked", lambda *a, **k: None)


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECRET = "sk-test-endpoint-key-5566778899"


@pytest.fixture(autouse=True)
def _scratch(monkeypatch, tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    for var in ("OPENAI_API_KEY", "HF_TOKEN", "GBENCH_SHARD", "GBENCH_SWE_BENCH_LIVE_MODE",
                "GBENCH_SWE_BENCH_LIVE_IMAGE", "GBENCH_SWE_BENCH_LIVE_AGENT_IMAGE",
                "GBENCH_SWE_BENCH_LIVE_TEMPERATURE", "GBENCH_SWE_BENCH_LIVE_WIN_AGENT_DIR",
                "GBENCH_SWE_BENCH_LIVE_HARNESS_DIR", "GBENCH_MULTI_SWE_BENCH_MODE",
                "GBENCH_MULTI_SWE_BENCH_TEMPERATURE", "GBENCH_MULTI_SWE_BENCH_AGENT_IMAGE",
                "GBENCH_TEMPERATURE", "GBENCH_REPETITION_PENALTY", "GBENCH_FREQUENCY_PENALTY",
                "GBENCH_PRESENCE_PENALTY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GBENCH_SWE_OMP_THREADS", "0")
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_TRAJ_DIR", str(tmp_path / "traj"))
    monkeypatch.setenv("GBENCH_MULTI_SWE_BENCH_TRAJ_DIR", str(tmp_path / "mswe"))
    monkeypatch.setenv("GBENCH_MULTI_SWE_REPO_DIR", str(tmp_path / "repos"))


def _doc(name):
    with open(os.path.join(REPO, "docs", "evals", f"{name}.md"), encoding="utf-8") as f:
        return f.read()


def _read(*parts):
    with open(os.path.join(REPO, *parts), encoding="utf-8") as f:
        return f.read()


# ============================================================================================
# EndpointRelay: the path allow-list
# ============================================================================================
class _Recorder(http.server.BaseHTTPRequestHandler):
    seen: list = []

    def _answer(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        type(self).seen.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                "body": json.loads(body) if body else None})
        out = json.dumps({"choices": [{"message": {"content": "hello"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    do_GET = do_POST = _answer

    def log_message(self, *a):
        return


@pytest.fixture
def upstream():
    handler = type("Up", (_Recorder,), {"seen": []})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server, handler
    server.shutdown()
    server.server_close()


def _raw(port, request_line, body=b""):
    s = socket.create_connection(("127.0.0.1", port), timeout=30)
    head = f"{request_line}\r\nHost: x\r\nConnection: close\r\n"
    if body:
        head += f"Content-Length: {len(body)}\r\nContent-Type: application/json\r\n"
    s.sendall(head.encode() + b"\r\n" + body)
    data = b""
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        data += chunk
    s.close()
    return int(data.split(b"\r\n", 1)[0].split()[1])


@pytest.mark.parametrize("path", ["/v1/load_lora_adapter", "/v1/unload_lora_adapter",
                                  "/v1/tokenize", "/v1/embeddings", "/v1/responses",
                                  "/v1/chat/completions/x", "/v1/models/x"])
def test_relay_refuses_paths_outside_its_allow_list(upstream, path):
    up = f"http://127.0.0.1:{upstream[0].server_address[1]}/v1"
    relay = C.EndpointRelay(up, "127.0.0.1", key=SECRET).start()
    try:
        status = _raw(relay._server.server_address[1], f"POST {path} HTTP/1.1", b"{}")
    finally:
        relay.stop()
    assert status == 400 and upstream[1].seen == [] and relay.requests_refused == 1


def test_relay_forwards_the_allowed_paths_with_the_key(upstream):
    port = upstream[0].server_address[1]
    relay = C.EndpointRelay(f"http://127.0.0.1:{port}/v1", "127.0.0.1", key=SECRET).start()
    try:
        for line in ("POST /v1/chat/completions HTTP/1.1", "POST /v1/completions HTTP/1.1",
                     "GET /v1/models HTTP/1.1"):
            assert _raw(relay._server.server_address[1], line, b"{}" if "POST" in line else b"") \
                == 200
    finally:
        relay.stop()
    # the first is the relay's own GET /v1/models, read once before the first completions
    # request to learn whether the endpoint is vLLM (EndpointRelay._probe_vllm)
    assert [s["path"] for s in upstream[1].seen] == ["/v1/models"] + list(
        C.EndpointRelay.ALLOWED_PATHS)
    assert all(s["auth"] == f"Bearer {SECRET}" for s in upstream[1].seen)
    rec = relay.record()
    assert rec["allowed_paths"] == list(C.EndpointRelay.ALLOWED_PATHS)
    # the address it listened on stays recorded after stop()
    assert rec["listens"].startswith("http://127.0.0.1:") and ":0/" not in rec["listens"]
    assert SECRET not in json.dumps(rec)


def test_relay_allow_list_can_be_narrowed_or_widened():
    relay = C.EndpointRelay("http://h:1/v1", "127.0.0.1", allowed_paths=("/v1/responses",))
    assert relay.target("/v1/responses") == "http://h:1/v1/responses"
    assert relay.target("/v1/chat/completions") is None


def test_pro_forwarder_record_names_the_allowed_paths():
    fwd = P.KeyForwarder("http://h:1/v1", SECRET, "127.0.0.1")
    assert fwd.allowed_paths == C.EndpointRelay.ALLOWED_PATHS
    src = open(P.__file__, encoding="utf-8").read()
    assert "', '.join(forwarder.allowed_paths)" in src


# ============================================================================================
# swe_lancer: the sampling contract is in the module header
# ============================================================================================
def test_lancer_header_states_its_sampling_contract():
    doc = ast.get_docstring(ast.parse(open(SL.__file__, encoding="utf-8").read()))
    para = doc[doc.index("Sampling:"):].split("\n\n")[0]
    for want in ("no temperature and no max_tokens", "--temperature",
                 "GBENCH_SWE_LANCER_TEMPERATURE", "not leaderboard-comparable",
                 "--max-output-tokens"):
        assert want in para, want
    assert SL.OFFICIAL_SAMPLING.send_temperature is False
    assert SL.OFFICIAL_SAMPLING.send_max_tokens is False


def test_unused_lancer_adapter_is_marked():
    for rel in (("gbench", "runners", "eval_suites", "_swelancer_adapter",
                 "predictions_solver.py"),
                ("gbench", "runners", "eval_suites", "_swelancer_adapter",
                 "run_swelancer_eval.py"),
                ("docker", "swe_lancer.Dockerfile")):
        assert "no longer used" in _read(*rel), rel
    assert "_swelancer_adapter" not in open(SL.__file__, encoding="utf-8").read()


# ============================================================================================
# swe_bench_live: per-setting prerequisites (the Windows tab)
# ============================================================================================
def _stub_settings(monkeypatch, windows_ok):
    checked, ran = [], []
    monkeypatch.setattr(LV, "check_windows_prerequisites", lambda: (
        (True, "") if windows_ok else (False, "The SWE-bench-Live Windows tab needs a Windows "
                                              "host whose Docker engine runs Windows containers.")))
    monkeypatch.setattr(LV, "check_swe_bench_live_prerequisites",
                        lambda settings: checked.append(list(settings)) or (True, ""))
    monkeypatch.setattr(LV, "check_sweagent_prerequisites", lambda: (True, ""))
    if windows_ok:
        monkeypatch.setattr(LV, "check_windows_metadata_guard", lambda: (True, ""))

    def _setting(setting, mode, model_name, base_url, concurrency, kwargs):
        ran.append(setting)
        return {"status": "success", "accuracy": 50.0, "total_questions": 2,
                "correct_answers": 1, "leaderboard_comparable": True, "tab": LV.TAB_LABELS[setting],
                "sample_traces": [{"sample_idx": 0}, {"sample_idx": 1}]}
    monkeypatch.setattr(LV, "_run_setting", _setting)
    return checked, ran


def test_live_default_run_off_windows_runs_the_linux_tabs_and_reports_windows(monkeypatch):
    checked, ran = _stub_settings(monkeypatch, windows_ok=False)
    res = LV.run_swe_bench_live("m", "http://h:1/v1", 1)
    linux = [s for s in LV.SETTINGS if s != "windows"]
    assert ran == linux and checked == [linux]
    w = res["settings"]["windows"]
    assert w["status"] == "infra_required" and "Windows host" in w["infra_required"]
    assert w["error"].startswith("[INFRA REQUIRED] swe_bench_live Windows tab")
    assert w["accuracy"] is None and w["leaderboard_comparable"] is False
    assert res["status"] == "completed_with_errors"
    assert res["settings_infra_required"] == ["windows"] and "settings_with_errors" not in res
    assert res["headline_metrics"]["windows_resolved_rate"] is None
    assert res["headline_metrics"]["lite_resolved_rate"] == 50.0
    assert "windows" not in res["category_accuracy"] and res["accuracy"] == 50.0
    assert res["leaderboard_comparable"] is False
    assert "Windows tab was not run" in res["leaderboard_comparable_reason"]
    assert res["leaderboard_settings"] == list(LV.SETTINGS)


def test_live_windows_alone_off_windows_hard_errors(monkeypatch):
    monkeypatch.setattr(LV._platform, "system", lambda: "Linux")
    monkeypatch.setattr(LV, "_run_setting", lambda *a, **k: pytest.fail("nothing may run"))
    with pytest.raises(RuntimeError, match=r"^\[INFRA REQUIRED\] swe_bench_live.*Windows host"):
        LV.run_swe_bench_live("m", "http://h:1/v1", 1, eval_categories="windows")


def test_live_windows_runs_where_its_prerequisite_exists(monkeypatch):
    checked, ran = _stub_settings(monkeypatch, windows_ok=True)
    res = LV.run_swe_bench_live("m", "http://h:1/v1", 1, eval_categories="go,windows")
    assert ran == ["go", "windows"] and checked == [["go", "windows"]]
    assert res["status"] == "success" and "settings_infra_required" not in res


def test_live_all_settings_missing_or_errored_is_an_error():
    res = LV.combine_settings({
        "go": {"status": "error", "error": "go broke", "leaderboard_comparable": False},
        "windows": LV.infra_required_setting("windows", "needs a Windows host")})
    assert res["status"] == "error" and res["settings_infra_required"] == ["windows"]
    assert "windows: [INFRA REQUIRED]" in res["error"]


def test_live_linux_prerequisites_still_hard_error(monkeypatch):
    _stub_settings(monkeypatch, windows_ok=False)
    monkeypatch.setattr(LV, "check_swe_bench_live_prerequisites",
                        lambda settings: (False, "docker daemon not reachable."))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*docker daemon not reachable"):
        LV.run_swe_bench_live("m", "http://h:1/v1", 1)


def test_live_agentic_needs_the_sweagent_image(monkeypatch):
    _stub_settings(monkeypatch, windows_ok=False)
    monkeypatch.setattr(LV, "check_sweagent_prerequisites",
                        lambda: (False, "the SWE-agent-for-eval image 'x' is not built."))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*SWE-agent-for-eval.*oneshot"):
        LV.run_swe_bench_live("m", "http://h:1/v1", 1, eval_categories="lite")
    # the one-shot mode does not need it
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_MODE", "oneshot")
    res = LV.run_swe_bench_live("m", "http://h:1/v1", 1, eval_categories="lite")
    assert res["mode"] == A.ONESHOT


# ============================================================================================
# swe_bench_live: credential-free environments for Win-Agent and the harnesses
# ============================================================================================
def _win_agent_checkout(tmp_path):
    agent = tmp_path / "win-agent-src"
    (agent / "config").mkdir(parents=True)
    (agent / "main.py").write_text("# stand-in\n")
    (agent / "config" / "default.yaml").write_text(
        "dataset: SWE-bench-Live/Windows\nplatform: windows\nmodel: x\nbase_url: y\n"
        "max_steps: 100\nworkers: 2\ninstance_timeout: 480\n")
    return agent


def test_win_agent_gets_no_credential_variable(monkeypatch, tmp_path):
    agent = _win_agent_checkout(tmp_path)
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_WIN_AGENT_DIR", str(agent))
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("HF_TOKEN", "hf_" + "a" * 30)
    monkeypatch.setenv("SOME_SERVICE_API_KEY", "value-of-another-key-123")
    seen = {}

    def _run(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw["env"]
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(LV.subprocess, "run", _run)
    task = A.AgentTask("i-1", "p", "img", "o/r", meta={"split": "test"})
    row = {"instance_id": "i-1", "_split": "test", "problem_statement": "p"}
    LV.run_win_agent([task], [row], "m", "http://127.0.0.1:9/v1", 1, str(tmp_path / "out"))
    env = seen["env"]
    assert "OPENAI_API_KEY" not in env and "HF_TOKEN" not in env
    assert "SOME_SERVICE_API_KEY" not in env
    assert env["PYTHONUTF8"] == "1" and "PATH" in env
    assert SECRET not in " ".join(seen["cmd"]) + json.dumps(env)
    assert seen["cmd"][-2:] == ["--api_key", "EMPTY"]


@pytest.mark.parametrize("platform", ["linux", "windows"])
def test_launch_harness_gets_no_credential_variable(monkeypatch, tmp_path, platform):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("HF_TOKEN", "hf_" + "b" * 30)
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_HARNESS_DIR", str(tmp_path / "sbl"))
    seen = {}

    def _run(cmd, **kw):
        seen["env"] = kw["env"]
        return types.SimpleNamespace(returncode=1, stdout="", stderr="")
    monkeypatch.setattr(LV.subprocess, "run", _run)
    import asyncio
    row = {"instance_id": "i-1", "_split": "go", "problem_statement": "p"}
    scorer = LV.make_launch_scorer("go", [row], platform, 1, {})
    asyncio.run(scorer([{"extra_payload": {"instance_id": "i-1"}, "agent_patch": "d\n"}]))
    assert "OPENAI_API_KEY" not in seen["env"] and "HF_TOKEN" not in seen["env"]
    if platform == "windows":
        assert seen["env"]["PYTHONUTF8"] == "1"


# ============================================================================================
# swe_bench_live: SWE-agent-for-eval
# ============================================================================================
def test_sweagent_pins_and_image_agree():
    text = _read("gbench", "docker", "sweagent_for_eval.Dockerfile")
    assert f"ARG SWE_AGENT_FOR_EVAL_REF={LV.SWEAGENT_COMMIT}" in text
    assert f"ARG SWE_AGENT_FOR_EVAL_CONFIG_SHA256={LV.SWEAGENT_CONFIG_SHA256}" in text
    assert 'org.gbench.sweagent_for_eval="${SWE_AGENT_FOR_EVAL_REF}"' in text
    assert "FROM ubuntu:24.04@sha256:" in text and "snapshot.ubuntu.com" in text
    assert "pip install --no-cache-dir --editable ." in text
    assert "cp -r swerex venv/lib/python3.12/site-packages/swerex" in text
    assert "PIP_CONSTRAINT=/opt/sweagent-for-eval-constraints.txt" in text
    assert f"ARG SWEREX_RUNTIME_SHA256={LV.SWEREX_RUNTIME_SHA256}" in text
    assert f"ARG SWEREX_RUNTIME_UPSTREAM_SHA256={LV.SWEREX_RUNTIME_UPSTREAM_SHA256}" in text
    assert 'org.gbench.sweagent_for_eval_swerex_runtime="${SWEREX_RUNTIME_SHA256}"' in text
    assert "COPY sweagent_for_eval_swerex_heredoc.patch " in text and "git apply" in text
    assert f"docker build -t {LV.SWEAGENT_IMAGE_DEFAULT} -f" in text
    pins = [c for c in _read("gbench", "docker", "sweagent_for_eval.constraints.txt").splitlines()
            if c and not c.startswith("#")]
    assert pins and all("==" in c for c in pins) and "litellm==1.87.0" in pins
    assert LV.SWEAGENT_CALL_LIMIT == 100 and LV.SWEAGENT_CONFIG == "config/default.yaml"
    assert LV.SWEAGENT_SAMPLING.send_temperature is False
    assert LV.SWEAGENT_SAMPLING.send_max_tokens is False


def _labels(monkeypatch, labels, rc=0):
    monkeypatch.setattr(LV.os.path, "exists", lambda p: True)
    monkeypatch.setattr(LV.subprocess, "run", lambda cmd, **kw: types.SimpleNamespace(
        returncode=rc, stdout=json.dumps(labels), stderr=""))


def test_sweagent_prerequisites_check_the_image_labels(monkeypatch):
    _labels(monkeypatch, {}, rc=1)
    ok, why = LV.check_sweagent_prerequisites()
    assert not ok and "not built" in why and "sweagent_for_eval.Dockerfile" in why
    _labels(monkeypatch, {"org.gbench.sweagent_for_eval": "0" * 40})
    ok, why = LV.check_sweagent_prerequisites()
    assert not ok and LV.SWEAGENT_COMMIT in why
    _labels(monkeypatch, {"org.gbench.sweagent_for_eval": LV.SWEAGENT_COMMIT,
                          "org.gbench.sweagent_for_eval_config_sha256": "x"})
    ok, why = LV.check_sweagent_prerequisites()
    assert not ok and LV.SWEAGENT_CONFIG_SHA256 in why
    # the image before gbench's fix of the task containers' runtime (2026-10-08): refused, with
    # the reason and the rebuild command
    _labels(monkeypatch, {"org.gbench.sweagent_for_eval": LV.SWEAGENT_COMMIT,
                          "org.gbench.sweagent_for_eval_config_sha256": LV.SWEAGENT_CONFIG_SHA256})
    ok, why = LV.check_sweagent_prerequisites()
    assert not ok and LV.SWEREX_RUNTIME_SHA256 in why and "heredoc" in why
    assert ("docker build -t gbench-sweagent-for-eval:82422422-rex3 -f "
            "gbench/docker/sweagent_for_eval.Dockerfile gbench/docker") in why
    _labels(monkeypatch, {"org.gbench.sweagent_for_eval": LV.SWEAGENT_COMMIT,
                          "org.gbench.sweagent_for_eval_config_sha256": LV.SWEAGENT_CONFIG_SHA256,
                          "org.gbench.sweagent_for_eval_swerex_runtime": "0" * 64})
    ok, why = LV.check_sweagent_prerequisites()
    assert not ok and LV.SWEREX_RUNTIME_SHA256 in why
    # an image with the runtime fix but not the submit-time read fix (rex1) is refused
    _labels(monkeypatch, {"org.gbench.sweagent_for_eval": LV.SWEAGENT_COMMIT,
                          "org.gbench.sweagent_for_eval_config_sha256": LV.SWEAGENT_CONFIG_SHA256,
                          "org.gbench.sweagent_for_eval_swerex_runtime": LV.SWEREX_RUNTIME_SHA256})
    ok, why = LV.check_sweagent_prerequisites()
    assert not ok and LV.SWEAGENT_AGENTS_SHA256 in why and "UTF-8" in why
    # the rex2 image (2026-10-09: the read decodes tolerantly but still opens a submodule's
    # directory, duckdb__ducklake-1224) is refused too, with the rebuild command
    _labels(monkeypatch, {"org.gbench.sweagent_for_eval": LV.SWEAGENT_COMMIT,
                          "org.gbench.sweagent_for_eval_config_sha256": LV.SWEAGENT_CONFIG_SHA256,
                          "org.gbench.sweagent_for_eval_swerex_runtime": LV.SWEREX_RUNTIME_SHA256,
                          "org.gbench.sweagent_for_eval_agents":
                              "8dcc901016a3a2f59168e8671bcdabdc54afbbdbdad8fef02f583268326497a3"})
    ok, why = LV.check_sweagent_prerequisites()
    assert not ok and LV.SWEAGENT_AGENTS_SHA256 in why and "directory" in why
    assert "docker build -t gbench-sweagent-for-eval:82422422-rex3 " in why
    _labels(monkeypatch, {"org.gbench.sweagent_for_eval": LV.SWEAGENT_COMMIT,
                          "org.gbench.sweagent_for_eval_config_sha256": LV.SWEAGENT_CONFIG_SHA256,
                          "org.gbench.sweagent_for_eval_swerex_runtime": LV.SWEREX_RUNTIME_SHA256,
                          "org.gbench.sweagent_for_eval_agents": LV.SWEAGENT_AGENTS_SHA256})
    assert LV.check_sweagent_prerequisites() == (True, "")


def test_sweagent_instance_line_carries_only_what_its_loader_reads():
    py = {"instance_id": "a__b-1", "problem_statement": "p", "base_commit": "c",
          "FAIL_TO_PASS": ["t"], "test_patch": "secret tests", "repo": "a/b"}
    assert LV.sweagent_instance(py) == {
        "instance_id": "a__b-1", "problem_statement": "p", "base_commit": "c",
        "image_name": "starryzhang/sweb.eval.x86_64.a_1776_b-1"}
    ml = dict(py, docker_image="starryzhang/sweb.eval.x86_64.x")
    assert LV.sweagent_instance(ml)["image_name"] == "starryzhang/sweb.eval.x86_64.x"


def test_sweagent_command_is_the_readme_rollout_command(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    out = str(tmp_path)
    cmd = LV.sweagent_command(f"{out}/instances.jsonl", out, "org/model",
                              "http://127.0.0.1:4242/v1", 3, "gbench-sweagent-x")
    assert cmd[:4] == ["docker", "run", "--rm", "--name"] and "--network" in cmd
    assert cmd[cmd.index("--network") + 1] == "host" and f"{out}:{out}:rw" in cmd
    assert "OPENAI_API_KEY=EMPTY" in cmd and SECRET not in " ".join(cmd)
    tail = cmd[cmd.index(LV.sweagent_image()) + 1:]
    assert tail == ["/opt/SWE-agent-for-eval/venv/bin/sweagent", "run-batch",
                    "--config", "/opt/SWE-agent-for-eval/config/default.yaml",
                    "--agent.model.name", "openai/org/model",
                    "--agent.model.api_base", "http://127.0.0.1:4242/v1",
                    "--instances.type", "swe_bench", "--instances.subset",
                    f"{out}/instances.jsonl",
                    "--instances.deployment.docker_args="
                    "--label=org.gbench.sweagent_batch=gbench-sweagent-x",
                    "--num_workers", "3", "--output_dir", out,
                    "--progress_bar", "False"]


def _write_sweagent_outputs(out, iid, status, patch, pred=True):
    d = os.path.join(out, iid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{iid}.traj"), "w") as f:
        json.dump({"info": {"exit_status": status, "submission": patch,
                            "model_stats": {"api_calls": 7}}, "trajectory": []}, f)
    if pred:
        with open(os.path.join(d, f"{iid}.pred"), "w") as f:
            json.dump({"instance_id": iid, "model_patch": patch}, f)


def test_read_sweagent_run_classifies_the_exit_statuses(tmp_path):
    out = str(tmp_path / "run")
    tasks = [A.AgentTask(f"i-{n}", "p", "img", "o/r", meta={"split": "go"}) for n in range(5)]
    _write_sweagent_outputs(out, "i-0", "submitted", "diff --git a/x b/x\n")
    _write_sweagent_outputs(out, "i-1", "submitted (exit_cost)", "diff --git a/y b/y\n")
    _write_sweagent_outputs(out, "i-2", "exit_context", None)
    _write_sweagent_outputs(out, "i-3", "submitted (exit_api)", "diff --git a/z b/z\n")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "run_batch_exit_statuses.yaml"), "w") as f:
        f.write("instances_by_exit_status:\n  Uncaught DockerPullError:\n  - i-4\n")
    tr = {t["extra_payload"]["instance_id"]: t for t in LV.read_sweagent_run(tasks, out, 0.0)}
    assert (tr["i-0"]["agent_infra_error"], tr["i-0"]["agent_patch"]) == (
        False, "diff --git a/x b/x\n")
    assert tr["i-0"]["agent_steps"] == 7 and tr["i-0"]["trajectory_file"].endswith("i-0.traj")
    assert tr["i-1"]["agent_infra_error"] is False          # the call limit: the model's outcome
    assert tr["i-2"]["agent_infra_error"] is False and tr["i-2"]["agent_patch"] == ""
    assert tr["i-3"]["agent_infra_error"] is True            # the endpoint kept failing
    assert tr["i-4"]["agent_infra_error"] is True and "Uncaught DockerPullError" in \
        tr["i-4"]["agent_error"]
    assert tr["i-0"]["extra_payload"] == {"instance_id": "i-0", "split": "go"}


def _python_rows(n):
    return [{**{c: "x" for c in LV.SNAPSHOT_COLUMNS}, "repo": "o/r",
             "instance_id": f"lite__r-{i}", "problem_statement": f"issue {i}",
             "base_commit": "c"} for i in range(n)]


@pytest.mark.parametrize("thinking", [True, False])
def test_python_tab_runs_sweagent_through_the_relay_then_the_fork(monkeypatch, tmp_path, thinking):
    want = 1.0 if thinking else 0.0
    rows = _python_rows(3)
    monkeypatch.setitem(LV.SPLIT_ROWS, "lite", 3)
    monkeypatch.setitem(LV.SPLIT_SHA256, "lite", C.snapshot_sha256(rows))
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda name, *a, split=None, revision=None, **k: [dict(r) for r in rows]
    monkeypatch.setitem(sys.modules, "datasets", fake)
    monkeypatch.setattr(LV, "check_swe_bench_live_prerequisites", lambda settings: (True, ""))
    monkeypatch.setattr(LV, "check_sweagent_prerequisites", lambda: (True, ""))
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    relays, runs, harness = [], [], []
    real_relay = LV.EndpointRelay

    class _Relay(real_relay):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            relays.append(self)
    monkeypatch.setattr(LV, "EndpointRelay", _Relay)

    def _docker(cmd, **kw):
        if "run-batch" in cmd:
            runs.append({"cmd": cmd, "env": kw.get("env")})
            out = cmd[cmd.index("--output_dir") + 1]
            lines = [json.loads(x) for x in open(cmd[cmd.index("--instances.subset") + 1])]
            for line in lines:
                _write_sweagent_outputs(out, line["instance_id"], "submitted",
                                        f"diff --git a/{line['instance_id']} b/x\n")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(LV.subprocess, "run", _docker)

    def _scorer(eval_name, model_name, dataset, split, namespace, workers, metrics, **k):
        harness.append({"dataset": dataset, "split": split, "namespace": namespace, **k})

        async def _score(traces):
            metrics["swebench_report"] = {"total_instances": len(traces), "error_ids": [],
                                          "submitted_instances": len(traces)}
            for t in traces:
                t["is_correct"] = t["extra_payload"]["instance_id"] == "lite__r-1"
        return _score
    monkeypatch.setattr(C, "make_swebench_scorer", _scorer)
    res = LV.run_swe_bench_live("m", "http://127.0.0.1:1/v1", 2, eval_categories="lite",
                                enable_thinking=thinking)
    s = res["settings"]["lite"]
    assert (s["correct_answers"], s["total_questions"]) == (1, 3)
    assert s["agent"]["scaffold"] == "SWE-agent-for-eval"
    assert s["agent"]["commit"] == LV.SWEAGENT_COMMIT
    assert s["agent"]["per_instance_call_limit"] == 100
    # gbench's rule, although SWE-agent-for-eval sends none: the relay sends 1.0 with --thinking,
    # 0.0 without
    assert (s["temperature"], s["temperature_source"]) == (
        (1.0, "thinking") if thinking else (0.0, "default"))
    assert s["official_sampling"]["send_temperature"] is False
    assert s["official_sampling"]["temperature_matches_official"] is False
    assert ("--thinking rule" if thinking else "no-think rule") in s["leaderboard_comparable_reason"]
    assert "not sent" in s["max_output_tokens_note"]
    assert harness[0]["split"] == "lite" and harness[0]["namespace"] == "starryzhang"
    assert harness[0]["harness_image"] == LV._image()
    # the instance file the agent read, and the relay it talked to
    lines = [json.loads(x) for x in open(runs[0]["cmd"][runs[0]["cmd"].index(
        "--instances.subset") + 1])]
    assert all(set(x) == {"instance_id", "problem_statement", "base_commit", "image_name"}
               for x in lines)
    relay = relays[0]
    assert relay._key == SECRET and relay._server is None
    assert relay._body_fields == {"chat_template_kwargs": {"enable_thinking": thinking},
                                  "temperature": want}
    assert "OPENAI_API_KEY" not in runs[0]["env"]
    assert res["leaderboard_comparable"] is False       # stand-in rows: no board total of 3
    assert "reference scaffolds" not in res["leaderboard_comparable_reason"]
    assert SECRET not in json.dumps(res, default=str)


def test_python_tab_with_an_operator_temperature_is_not_comparable(monkeypatch):
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_TEMPERATURE", "0.3")
    temperature, source = LV.resolve_official_sampling("swe_bench_live", LV.SWEAGENT_SAMPLING)
    assert temperature == 0.3 and source.startswith("env:")


# ============================================================================================
# multi_swe_bench: MSWE-agent
# ============================================================================================
def test_mswe_pins_and_image_agree():
    text = _read("gbench", "docker", "mswe_agent.Dockerfile")
    assert f"ARG MSWE_AGENT_REF={M.MSWE_AGENT_COMMIT}" in text
    assert f"ARG MSWE_AGENT_CONFIG_SHA256={M.MSWE_AGENT_CONFIG_SHA256}" in text
    assert f"ARG MSWE_HARNESS_JAVA_REF={M.MSWE_AGENT_JAVA_REF}" in text
    assert f'ARG MSWE_JAVA_ORGS="{M.MSWE_AGENT_JAVA_ORGS}"' in text
    assert "FROM python:3.9.21-slim-bookworm@sha256:" in text
    assert "requirements.txt dataclasses-json" in text
    assert "from __future__ import annotations" in text
    pins = [c for c in _read("gbench", "docker", "mswe_agent.constraints.txt").splitlines()
            if c and not c.startswith("#")]
    assert pins and all("==" in c for c in pins)
    assert "openai==1.70.0" in pins and "dataclasses-json==0.6.7" in pins
    assert M.MSWE_AGENT_STEP_LIMIT == 50
    assert M.MSWE_AGENT_SAMPLING.temperature == 0.0
    assert M.MSWE_AGENT_SAMPLING.params == {"top_p": 0.95}
    assert M.MSWE_AGENT_SAMPLING.send_max_tokens is False


def test_mswe_prerequisites_check_the_image_labels(monkeypatch):
    monkeypatch.setattr(M.os.path, "exists", lambda p: True)
    good = {"org.gbench.mswe_agent": M.MSWE_AGENT_COMMIT,
            "org.gbench.mswe_agent_config_sha256": M.MSWE_AGENT_CONFIG_SHA256,
            "org.gbench.mswe_agent_java_repos":
                f"multi-swe-bench@{M.MSWE_AGENT_JAVA_REF}: {M.MSWE_AGENT_JAVA_ORGS}",
            M.MSWE_AGENT_DRIVER_LABEL: M.MSWE_AGENT_DRIVER_VERSION}
    stale = {k: v for k, v in good.items() if k != M.MSWE_AGENT_DRIVER_LABEL}
    for labels, rc, want in (({}, 1, "not built"),
                             (stale, 0, M.MSWE_AGENT_DRIVER_LABEL),
                             (dict(good, **{"org.gbench.mswe_agent": "x"}), 0,
                              M.MSWE_AGENT_COMMIT),
                             (dict(good, **{"org.gbench.mswe_agent_java_repos": "x"}), 0,
                              M.MSWE_AGENT_JAVA_REF)):
        monkeypatch.setattr(M.subprocess, "run", lambda cmd, labels=labels, rc=rc, **kw:
                            types.SimpleNamespace(returncode=rc, stdout=json.dumps(labels)))
        ok, why = M.check_mswe_agent_prerequisites()
        assert not ok and want in why
    monkeypatch.setattr(M.subprocess, "run", lambda cmd, **kw: types.SimpleNamespace(
        returncode=0, stdout=json.dumps(good)))
    assert M.check_mswe_agent_prerequisites() == (True, "")


def test_mswe_command_mounts_a_fixed_path_and_passes_no_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    cmd = M.mswe_agent_command(str(tmp_path), "n", "http://127.0.0.1:77/v1")
    assert f"{tmp_path}:/mswe:rw" in cmd and cmd[cmd.index("-w") + 1] == "/mswe"
    assert "OPENAI_API_KEY=EMPTY" in cmd and "OPENAI_API_BASE_URL=http://127.0.0.1:77/v1" in cmd
    assert cmd[-4:] == ["/opt/MSWE-agent/venv/bin/python", "/opt/gbench/mswe_agent_run.py",
                        "--spec", "/mswe/spec.json"]
    assert SECRET not in " ".join(cmd)


def _summary(instances, images=None, **extra):
    return {"language": "go", "instances": instances,
            "images": images or {i: {"error": None} for i in instances}, **extra}


def test_mswe_trace_classification(tmp_path):
    w = str(tmp_path)
    t = A.AgentTask("o__r-1", "task", "mswebench/o_m_r:pr-1", "go",
                    meta={"org": "o", "repo": "r", "number": 1})

    def trace(**inst):
        base = {"completed": True, "patch": "\ndiff --git a/x b/x\n", "exit_status": "submitted",
                "api_calls": 12, "steps": 12, "step_limit_reached": False,
                "trajectory_file": "/mswe/trajectories/gbench/run/o__r-1.traj"}
        base.update(inst)
        return M.mswe_agent_trace(0, t, _summary({"o__r-1": base}), w, 0.0)
    ok = trace()
    assert ok["agent_infra_error"] is False and ok["agent_patch"].startswith("\ndiff")
    assert ok["trajectory_file"] == os.path.join(w, "trajectories/gbench/run/o__r-1.traj")
    limit = trace(exit_status="submitted (exit_error)", step_limit_reached=True, api_calls=50)
    assert limit["agent_infra_error"] is False and limit["agent_step_limit_reached"] is True
    assert trace(exit_status="submitted (exit_cost)")["agent_infra_error"] is False
    assert trace(exit_status="exit_format")["agent_infra_error"] is False
    # an exit_error other than the step limit that names no endpoint or docker daemon failure
    # (here: no trajectory to read) is the model's outcome, with a note
    kept = trace(exit_status="submitted (exit_error)")
    assert kept["agent_infra_error"] is False and "model's outcome" in kept["agent_exit_note"]
    # early_exit: the container stopped answering while it ran the model's command, the
    # model's outcome of that run (unresolved), with a note
    early = trace(exit_status="early_exit")
    assert early["agent_infra_error"] is False and "model's outcome" in early["agent_exit_note"]
    assert trace(exit_status="exit_api")["agent_infra_error"] is True
    assert trace(completed=False)["agent_infra_error"] is True
    img = M.mswe_agent_trace(0, t, _summary({}, {"o__r-1": {"error": "docker pull failed"}}),
                             w, 0.0)
    assert img["agent_infra_error"] is True and "docker pull failed" in img["agent_error"]
    none = M.mswe_agent_trace(0, t, None, w, 0.0)
    assert none["agent_infra_error"] is True and "no summary" in none["agent_error"]


def _mrow(org, repo, n):
    return {"org": org, "repo": repo, "number": n, "instance_id": f"{org}__{repo}-{n}",
            "title": "PR", "body": "B", "base": {"sha": "abc"}, "fix_patch": "gold",
            "resolved_issues": [{"number": 1, "title": "Issue", "body": "Body"}]}


def test_mswe_runs_one_language_at_a_time_with_its_sampling(monkeypatch, tmp_path):
    rows = {r["instance_id"]: r for r in (_mrow("o", "go", 1), _mrow("o", "go", 2),
                                          _mrow("o", "java", 3))}
    lang = {"o__go-1": "go", "o__go-2": "go", "o__java-3": "java"}
    tasks = [A.AgentTask(i, M._issue_text(r), M.published_image(r["org"], r["repo"],
                         r["number"]), lang[i], meta={"org": r["org"], "repo": r["repo"],
                                                      "number": r["number"]})
             for i, r in rows.items()]
    specs = []

    def _run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            work = cmd[cmd.index("-v", cmd.index("-v") + 1) + 1].split(":")[0]
            spec = json.load(open(os.path.join(work, "spec.json")))
            got = [json.loads(x) for x in open(os.path.join(work, "rows.jsonl"))]
            specs.append((spec, [g["instance_id"] for g in got], kw.get("env")))
            inst = {g["instance_id"]: {"completed": True, "patch": "d\n",
                                       "exit_status": "submitted", "api_calls": 3, "steps": 3,
                                       "step_limit_reached": False, "trajectory_file": None}
                    for g in got}
            json.dump(_summary(inst), open(os.path.join(work, "summary.json"), "w"))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(M.subprocess, "run", _run)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    traces = M.run_mswe_agent(tasks, rows, "served/model", "http://127.0.0.1:5/v1", 4,
                              str(tmp_path / "out"), 0.0, 0.95)
    assert [s[0]["language"] for s in specs] == ["go", "java"]
    assert specs[0][1] == ["o__go-1", "o__go-2"] and specs[1][1] == ["o__java-3"]
    spec = specs[0][0]
    assert (spec["temperature"], spec["top_p"], spec["step_limit"], spec["workers"]) == (
        0.0, 0.95, 50, 4)
    assert spec["model_name"] == "served/model" and spec["rows_file"] == "/mswe/rows.jsonl"
    assert all("OPENAI_API_KEY" not in s[2] for s in specs)
    assert [t["extra_payload"]["instance_id"] for t in traces] == list(rows)
    assert all(not t["agent_infra_error"] for t in traces)


def _agentic_setup(monkeypatch, tmp_path, report):
    rows = [_mrow("o", "r", n) for n in (1, 2)]
    src = tmp_path / "src.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(M, "_read_instances", lambda: [("go", r) for r in rows])
    monkeypatch.setattr(M, "_SRC_FILES", {r["instance_id"]: str(src) for r in rows})
    monkeypatch.setattr(M, "check_multi_swe_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M, "check_mswe_agent_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M.swe_thread_cap, "apply", lambda c, n, e: (c, 0))
    seen = {}

    def _rollouts(tasks, rws, model_name, relay_url, concurrency, out_dir, temperature, top_p):
        seen.update(temperature=temperature, top_p=top_p, relay_url=relay_url)
        return [{"sample_idx": i, "category": "go",
                 "extra_payload": {"instance_id": t.instance_id, **t.meta},
                 "agent_patch": "diff --git a/x b/x\n", "agent_exit_status": "submitted",
                 "agent_error": None, "agent_infra_error": False, "agent_steps": 5,
                 "agent_step_limit_reached": False, "is_correct": False, "status": "OK"}
                for i, t in enumerate(tasks)]
    monkeypatch.setattr(M, "run_mswe_agent", _rollouts)

    def _harness(cmd, **kw):
        cfg = json.load(open(cmd[-1]))
        with open(f"{cfg['output_dir']}/final_report.json", "w") as f:
            json.dump(report, f)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(M.subprocess, "run", _harness)
    return seen


def _report(resolved, unresolved):
    hid = lambda n: f"o/r:pr-{n}"   # noqa: E731
    return {"total_instances": 2, "submitted_instances": 2, "completed_instances": 2,
            "resolved_instances": len(resolved), "unresolved_instances": len(unresolved),
            "error_instances": 0, "empty_patch_instances": 0,
            "resolved_ids": [hid(n) for n in resolved],
            "unresolved_ids": [hid(n) for n in unresolved],
            "completed_ids": [hid(n) for n in (*resolved, *unresolved)], "error_ids": [],
            "incomplete_ids": []}


def test_mswe_clean_full_run_is_comparable(monkeypatch, tmp_path):
    seen = _agentic_setup(monkeypatch, tmp_path, _report((1,), (2,)))
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    res = M.run_multi_swe_bench("m", "http://127.0.0.1:1/v1", 1, enable_thinking=False)
    assert (res["correct_answers"], res["total_questions"], res["status"]) == (1, 2, "success")
    assert res["leaderboard_comparable"] is True and "leaderboard_comparable_reason" not in res
    assert (seen["temperature"], seen["top_p"]) == (0.0, 0.95)
    assert res["agent"]["scaffold"] == "MSWE-agent" and res["agent"]["step_limit"] == 50
    assert res["official_sampling"]["temperature_matches_official"] is True
    assert res["relay"]["sets_body_fields"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert res["relay"]["relays_to"] == "http://127.0.0.1:1/v1"
    assert SECRET not in json.dumps(res, default=str)


def test_mswe_thinking_run_passes_temperature_one_to_the_agent(monkeypatch, tmp_path):
    # --thinking: MSWE-agent's official 0.0 does not apply; its --temperature is 1.0, top_p stays
    seen = _agentic_setup(monkeypatch, tmp_path, _report((1,), (2,)))
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    res = M.run_multi_swe_bench("m", "http://127.0.0.1:1/v1", 1, enable_thinking=True)
    assert (seen["temperature"], seen["top_p"]) == (1.0, 0.95)
    assert (res["temperature"], res["temperature_source"]) == (1.0, "thinking")
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert res["leaderboard_comparable"] is False
    assert "MSWE-agent runs at 0.0" in res["leaderboard_comparable_reason"]
    assert res["relay"]["sets_body_fields"] == {"chat_template_kwargs": {"enable_thinking": True}}


def test_mswe_operator_temperature_and_subset_are_not_comparable(monkeypatch, tmp_path):
    seen = _agentic_setup(monkeypatch, tmp_path, _report((), (1,)))
    monkeypatch.setenv("GBENCH_MULTI_SWE_BENCH_TEMPERATURE", "0.7")
    res = M.run_multi_swe_bench("m", "http://127.0.0.1:1/v1", 1, limit=1)
    assert seen["temperature"] == 0.7
    reason = res["leaderboard_comparable_reason"]
    assert "temperature=0.7" in reason and "MSWE-agent runs at 0.0" in reason
    assert "--eval-limit subset (1 tasks)" in reason


def _driver():
    spec = importlib.util.spec_from_file_location(
        "mswe_agent_run", os.path.join(REPO, "gbench", "docker", "mswe_agent_run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: MSWE-agent@88217624 sweagent/environment/utils.py LANGUAGE_MAP and specify_languages: the
#: first language one of whose markers occurs as `<marker>_` or `<marker>-` in the data path.
_MSWE_LANGUAGE_MAP = {"java": ["java"], "javascript": ["javascript", "js", "nodejs"],
                      "cpp": ["cpp", "c++"], "c": ["c"], "typescript": ["typescript", "ts"],
                      "go": ["go"], "rust": ["rust"]}


def _specify_languages(path):
    for lang, markers in _MSWE_LANGUAGE_MAP.items():
        if any(f"{x}_" in path or f"{x}-" in path for x in markers):
            return lang
    return None


def test_driver_data_paths_name_their_language_only():
    D = _driver()
    for lang in M._LANGUAGES:
        assert _specify_languages(D.instances_path("/mswe", lang)) == D.MSWE_LANGUAGE[lang]
    assert D.hub_image("Kong", "insomnia", 7) == "mswebench/kong_m_insomnia:pr-7"


def test_driver_run_args_are_the_readme_arguments():
    D = _driver()
    args = D.run_args({"model_name": "served", "temperature": 0.0, "top_p": 0.95},
                      "/mswe/data/go_verified.jsonl")
    joined = " ".join(args)
    for want in ("--model_name served", "--pr_file /mswe/data/go_verified.jsonl",
                 "--config_file /opt/MSWE-agent/config/default.yaml",
                 "--per_instance_cost_limit 5.00", "--temperature 0.0", "--top_p 0.95",
                 "--pre_build_all_images False", "--cache_task_images True"):
        assert want in joined, want
    assert "per_instance_api_calls_limit" not in joined


def test_driver_step_limit_raises_on_the_51st_model_step(monkeypatch):
    D = _driver()
    calls = []

    class Agent:
        def run(self, setup_args, env, *a, **k):
            return "ran"

        def forward_model(self, observation, state, lang):
            calls.append(observation)
            return "ok"
    agents = types.ModuleType("sweagent.agent.agents")
    agents.Agent = Agent
    pkg = types.ModuleType("sweagent")
    sub = types.ModuleType("sweagent.agent")
    sub.agents = agents
    pkg.agent = sub
    monkeypatch.setitem(sys.modules, "sweagent", pkg)
    monkeypatch.setitem(sys.modules, "sweagent.agent", sub)
    monkeypatch.setitem(sys.modules, "sweagent.agent.agents", agents)
    D.install_step_limit(50)
    agent = Agent()
    env = types.SimpleNamespace(record=types.SimpleNamespace(data={"instance_id": "o__r-9"}))
    assert agent.run({}, env) == "ran"
    for i in range(50):
        assert agent.forward_model(str(i), "{}", "go") == "ok"
    with pytest.raises(RuntimeError, match=r"^\[o__r-9\] Exceed maximum number of steps: 50$"):
        agent.forward_model("x", "{}", "go")
    agent.run({}, env)                     # a new instance starts from zero
    assert agent.forward_model("y", "{}", "go") == "ok" and len(calls) == 51


def test_driver_collects_mswe_agents_own_files(tmp_path):
    D = _driver()
    run = tmp_path / "trajectories" / "gbench" / "served" / "run"
    run.mkdir(parents=True)
    (run / "o__r-1.traj").write_text(json.dumps({
        "info": {"exit_status": "submitted (exit_error)", "model_stats": {"api_calls": 50}},
        "trajectory": [{"action": "ls"}] * 50 + [{
            "action": "exit_error", "response":
                "exit due to runtime error: [o__r-1] Exceed maximum number of steps: 50"}]}))
    (run / "all_preds.jsonl").write_text(json.dumps(
        {"instance_id": "o__r-1", "model_patch": "\ndiff --git a/x b/x\n"}) + "\n")
    out = D.collect(str(tmp_path), ["o__r-1", "o__r-2"])
    assert out["o__r-1"]["completed"] is True and out["o__r-1"]["step_limit_reached"] is True
    assert out["o__r-1"]["api_calls"] == 50 and out["o__r-1"]["steps"] == 51
    assert out["o__r-2"] == {"completed": False, "patch": "", "exit_status": None,
                             "api_calls": None, "steps": 0, "step_limit_reached": False,
                             "trajectory_file": None}


def test_mswe_image_carries_the_driver():
    assert "COPY mswe_agent_run.py /opt/gbench/mswe_agent_run.py" in _read(
        "gbench", "docker", "mswe_agent.Dockerfile")


# ============================================================================================
# docs match the code
# ============================================================================================
def test_live_doc_describes_the_reference_scaffold_and_the_per_setting_rule():
    doc = _doc("swe_bench_live")
    for want in (LV.SWEAGENT_COMMIT, LV.SWEAGENT_CONFIG_SHA256, "SWE-agent-for-eval",
                 "gbench-sweagent-for-eval:82422422-rex3", LV.SWEREX_RUNTIME_SHA256,
                 "sweagent_for_eval_review_decode.patch", "org.gbench.sweagent_for_eval_agents",
                 "sweagent_for_eval_swerex_heredoc.patch", "settings_infra_required",
                 '`status: "infra_required"`', "GBENCH_SWE_BENCH_LIVE_AGENT_IMAGE",
                 "`/v1/chat/completions`, `/v1/completions` and `/v1/models`",
                 "--instances.type swe_bench", "base_commit"):
        assert want in doc, want
    for gone in ("mini-swe-agent", "zz-image-path", "Win-Agent only ever sees the placeholder"):
        assert gone not in doc, gone


def test_mswe_doc_describes_the_scaffold():
    doc = _doc("multi_swe_bench")
    for want in (M.MSWE_AGENT_COMMIT, M.MSWE_AGENT_CONFIG_SHA256, M.MSWE_AGENT_JAVA_REF,
                 "gbench-mswe-agent:88217624", "Exceed maximum number of steps: 50", "top_p 0.95",
                 "GBENCH_MULTI_SWE_BENCH_AGENT_IMAGE", "mswebench/<org>_m_<repo>:pr-<number>"):
        assert want in doc, want
    assert "mini-swe-agent" not in doc


def test_relay_docs_name_the_allowed_paths():
    for name in ("swe_bench_pro", "swe_lancer"):
        doc = _doc(name)
        assert "/v1/chat/completions" in doc and "/v1/models" in doc, name
        assert "plain `/v1/...`" not in doc, name


def test_new_files_are_ascii():
    for rel in (("gbench", "docker", "sweagent_for_eval.Dockerfile"),
                ("gbench", "docker", "sweagent_for_eval.constraints.txt"),
                ("gbench", "docker", "mswe_agent.Dockerfile"),
                ("gbench", "docker", "mswe_agent.constraints.txt"),
                ("gbench", "docker", "mswe_agent_run.py")):
        _read(*rel).encode("ascii")
