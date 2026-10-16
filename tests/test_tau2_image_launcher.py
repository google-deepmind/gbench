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

"""tau2 / tau3 run tau2-bench in the local gbench-tau2 image.

  * image     - docker/tau2.Dockerfile pins the revision gbench checks, installs tau2-bench from
                its own uv.lock, and locks sandbox-runtime 0.0.23; the suites hard-error with the
                build command when Docker or the image is missing or built at another revision.
  * launcher  - run_tau_suite starts one container per run (bridge network, the operator's uid,
                no capabilities, read-only gbench source, credentials passed by name only), maps
                the trace paths back to the host and passes errors through without re-wrapping.
  * driver    - the program the image runs imports tau_common under an empty package, so the
                rest of gbench never loads there, and writes result.json.
  * guards    - inside the image the installed packages must match tau2-bench's uv.lock, the
                banking_knowledge shell tool must really execute a command without seeing any
                credential, and the agent's endpoint key never lands in llm_args.
  * docs      - build command, pins and the sandbox-runtime requirement are stated.

No Docker, no network, no model server: subprocess calls and tau2 are stubbed.
"""

import contextlib
import json
import os
import subprocess
import sys
import types
from unittest import mock

import pytest

from gbench.runners.eval_suites import tau_common as TC

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SUITES_DIR = os.path.join(_REPO, "gbench", "runners", "eval_suites")


def _read(*parts):
    with open(os.path.join(_REPO, *parts), encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------- #
# the image definition
# --------------------------------------------------------------------------- #
def test_dockerfile_pins_the_revision_the_suite_checks():
    dockerfile = _read("docker", "tau2.Dockerfile")
    assert f"ARG TAU2_REF={TC.TAU2_BENCH_REVISION}" in dockerfile
    assert f'{TC.TAU2_IMAGE_LABEL}="${{TAU2_REF}}"' in dockerfile
    assert TC.TAU2_IMAGE_DEFAULT == "gbench-tau2:" + TC.TAU2_BENCH_REVISION[:8]
    assert TC.image_build_command() in dockerfile
    assert TC.image_build_command() == (
        f"docker build -t gbench-tau2:{TC.TAU2_BENCH_REVISION[:8]} -f docker/tau2.Dockerfile "
        "docker/tau2_srt")


def test_dockerfile_installs_upstreams_lock_and_sandbox_runtime():
    dockerfile = _read("docker", "tau2.Dockerfile")
    assert "FROM python:3.12-slim" in dockerfile            # tau2-bench: >=3.12,<3.14
    assert "uv sync --frozen --inexact --no-dev --extra knowledge" in dockerfile
    assert '"websockets==15.0.1"' in dockerfile
    assert "npm ci --omit=dev" in dockerfile
    for tool in ("ripgrep", "bubblewrap", "socat", "git"):
        assert tool in dockerfile
    assert "docker pull" not in dockerfile and "python3-pysrt" not in dockerfile
    assert "RUN python - <<" not in dockerfile, "heredocs do not run under the legacy builder"
    pkg = json.loads(_read("docker", "tau2_srt", "package.json"))
    assert pkg["dependencies"] == {"@anthropic-ai/sandbox-runtime": "0.0.23"}
    lock = json.loads(_read("docker", "tau2_srt", "package-lock.json"))
    packages = {k: v for k, v in lock["packages"].items() if k}
    assert packages["node_modules/@anthropic-ai/sandbox-runtime"]["version"] == "0.0.23"
    assert all(v.get("integrity", "").startswith("sha512-") for v in packages.values())


# --------------------------------------------------------------------------- #
# image prerequisite check
# --------------------------------------------------------------------------- #
def _fake_run(label=TC.TAU2_BENCH_REVISION, info_rc=0, inspect_rc=0):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        if cmd[:2] == ["docker", "info"]:
            return subprocess.CompletedProcess(cmd, info_rc, b"", b"")
        if cmd[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(cmd, inspect_rc, label + "\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return run, calls


@pytest.mark.parametrize("which, run_kw, fragment", [
    (None, {}, "docker CLI is not on PATH"),
    ("/usr/bin/docker", {"info_rc": 1}, "daemon is not reachable"),
    ("/usr/bin/docker", {"inspect_rc": 1}, "is not built"),
    ("/usr/bin/docker", {"label": "5ebebbe8"}, "rebuild it"),
    ("/usr/bin/docker", {}, None),
])
def test_check_tau_image(monkeypatch, which, run_kw, fragment):
    monkeypatch.delenv("GBENCH_TAU2_IMAGE", raising=False)
    run, calls = _fake_run(**run_kw)
    monkeypatch.setattr(TC.shutil, "which", lambda name: which)
    monkeypatch.setattr(TC.subprocess, "run", run)
    ok, why = TC.check_tau_image()
    if fragment is None:
        assert ok and why == ""
        assert calls[-1][-1] == TC.TAU2_IMAGE_DEFAULT
    else:
        assert not ok and fragment in why
        assert TC.image_build_command() in why and "never pulls" in why


def test_image_override(monkeypatch):
    monkeypatch.setenv("GBENCH_TAU2_IMAGE", "my-tau2:dev")
    assert TC.tau2_image() == "my-tau2:dev"
    assert TC.image_build_command().startswith("docker build -t my-tau2:dev ")


@pytest.mark.parametrize("suite", ["tau2", "tau3"])
def test_missing_image_is_a_hard_error_naming_the_build(monkeypatch, suite):
    import importlib
    mod = importlib.import_module(f"gbench.runners.eval_suites.{suite}")
    monkeypatch.setattr(TC.shutil, "which", lambda name: None)
    launched = []
    monkeypatch.setattr(TC, "_launch_image", lambda cmd: launched.append(cmd) or (0, ""))
    with pytest.raises(RuntimeError) as ei:
        getattr(mod, f"run_{suite}")("m", "http://x/v1", 2)
    msg = str(ei.value)
    assert msg.startswith("[INFRA REQUIRED]") and f"docs/evals/{suite}.md" in msg
    assert "docker build -t gbench-tau2:" in msg and launched == []
    if suite == "tau3":
        assert "banking_knowledge" in msg


@pytest.mark.parametrize("kwargs, domains", [({}, ["retail", "airline", "telecom"]),
                                              # not a gbench knob: the track's three domains run
                                              ({"domain": "Airline"},
                                               ["retail", "airline", "telecom"])])
def test_tau2_always_runs_the_suite_in_the_image(monkeypatch, kwargs, domains):
    from gbench.runners.eval_suites import tau2 as T2
    monkeypatch.delenv("GBENCH_TAU2_ENV_RUN", raising=False)
    monkeypatch.delenv("TAU2_ENV_RUN", raising=False)
    seen = {}
    monkeypatch.setattr(T2, "run_tau_suite",
                        lambda *a, **k: seen.update(args=a, kwargs=k) or {"status": "success"})
    assert T2.run_tau2("m", "http://x/v1", 3, enable_thinking=True, limit=6, **kwargs) == {
        "status": "success"}
    assert seen["args"] == ("tau2", domains, "m", "http://x/v1", 3, 6, "docs/evals/tau2.md")
    assert seen["kwargs"] == {"enable_thinking": True}


# --------------------------------------------------------------------------- #
# the docker command and the environment passed in
# --------------------------------------------------------------------------- #
def test_forwarded_env_passes_knobs_and_credentials_by_name_only():
    env = {"GEMINI_API_KEY": "g", "GEMINI_API_KEYS": "g,h", "OPENAI_API_KEY": "o",
           "GBENCH_TAU2_NUM_TRIALS": "4", "TAU2_SEED": "301", "GBENCH_TAU3_TEMPERATURE": "0.5",
           "GBENCH_SEARCH_MODELS": "a,b", "GBENCH_JUDGE_MODEL": "j", "HTTPS_PROXY": "p",
           "GBENCH_TAU2_SAVE_TRACES": "/host/traces", "TAU2_SAVE_TRACES": "/host/t",
           "GBENCH_TAU2_BENCH_SRC": "/host/src", "TAU2_BENCH_SRC": "/host/src",
           "TAU2_DATA_DIR": "/host/data", "GBENCH_TAU2_IMAGE": "x", "GBENCH_TAU2_ENV_RUN": "1",
           "ANTHROPIC_API_KEY": "a", "OPENROUTER_API_KEY": "r",
           "HF_TOKEN": "secret", "PATH": "/usr/bin", "HOME": "/home/u"}
    names = TC._forwarded_env(env, domains=["airline"])
    # No credential of the host goes in: the Gemini key stays with gbench's Gemini relay, the
    # host's OPENAI_API_KEY (the endpoint key) travels only as GBENCH_TAU_AGENT_KEY, and the
    # cascade is handed over by the launcher (GBENCH_JUDGE_MODELS), not passed through.
    # a proxy set: NO_PROXY / no_proxy carry host.docker.internal on the bridge network
    assert names == sorted(["GBENCH_TAU_AGENT_KEY",
                            "GBENCH_TAU2_NUM_TRIALS", "TAU2_SEED", "GBENCH_TAU3_TEMPERATURE",
                            "HTTPS_PROXY", "NO_PROXY", "no_proxy"])
    bridge = {"GEMINI_API_KEY": "gbench-gemini-relay-" + "0" * 48,
              "GBENCH_GEMINI_RELAY_URL": "http://172.17.0.1:4000",
              "GBENCH_JUDGE_MODELS": "gemini-a,gemini-b"}
    assert TC._forwarded_env(env, domains=["airline"], bridge_env=bridge) == sorted(
        names + list(bridge))


@pytest.mark.parametrize("domains, traces", [(["retail", "airline", "telecom"], True),
                                             (["banking_knowledge"], False)])
def test_docker_command(tmp_path, domains, traces):
    trace_root = str(tmp_path / "traces") if traces else None
    cmd = TC._docker_run_command("gbench-tau2-abc", "gbench-tau2:x", domains, "/h/io", "/h/cache",
                                 trace_root, ["GEMINI_API_KEY", "GBENCH_TAU2_SEED"],
                                 suites_dir="/h/suites")
    joined = " ".join(cmd)
    assert cmd[:5] == ["docker", "run", "--rm", "--name", "gbench-tau2-abc"]
    # docker's bridge network, never the host network (see test_guard_code_images.py)
    assert "--network bridge --add-host host.docker.internal:host-gateway" in joined
    assert "--network host" not in joined
    assert f"--user {os.getuid()}:{os.getgid()}" in joined
    assert "--cap-drop ALL" in joined and "--security-opt no-new-privileges" in joined
    assert "-e GEMINI_API_KEY -e GBENCH_TAU2_SEED" in joined, "values never on the command line"
    assert "-v /h/suites:/opt/gbench/eval_suites:ro" in joined
    assert "-v /h/io:/gbench_io:rw" in joined
    assert "-v /h/cache:/gbench_cache:rw -w /gbench_cache" in joined
    shell_opts = ("seccomp=unconfined", "apparmor=unconfined", "systempaths=unconfined")
    if traces:
        assert f"-v {trace_root}:/gbench_traces:rw" in joined
        assert "GBENCH_TAU2_SAVE_TRACES=/gbench_traces" in cmd
        assert not any(o in joined for o in shell_opts), "no shell tool outside banking"
    else:
        assert "/gbench_traces:rw" not in joined and "GBENCH_TAU2_SAVE_TRACES=" in cmd
        assert all(o in joined for o in shell_opts)
    assert cmd[-7:-1] == ["gbench-tau2:x", "python", "-I", "-B", "-u", "-c"]
    assert cmd[-1] == TC._image_bootstrap()


def test_host_path_maps_the_trace_mount_only(tmp_path):
    root = str(tmp_path)
    assert TC._host_path("/gbench_traces/run/airline/results.json", root) == os.path.join(
        root, "run", "airline", "results.json")
    assert TC._host_path("/gbench_traces", root) == os.path.join(root, ".")
    assert TC._host_path("/gbench_tracesX/a", root) == "/gbench_tracesX/a"
    assert TC._host_path(None, root) is None
    assert TC._host_path("/gbench_traces/x", None) == "/gbench_traces/x"


# --------------------------------------------------------------------------- #
# run_tau_suite round trip (the container is simulated)
# --------------------------------------------------------------------------- #
def _mount(cmd, target):
    for i, arg in enumerate(cmd):
        if arg == "-v" and cmd[i + 1].split(":")[1] == target:
            return cmd[i + 1].split(":")[0]
    return None


@contextlib.contextmanager
def fake_bridge_services(eval_name, image, base_url, agent_base_url, domains, docs_url):
    """Stands in for tau_common._bridge_services: no Gemini gate, no relay."""
    bridge = TC._TauBridge(agent_base_url or base_url)
    bridge.env = {"GEMINI_API_KEY": "gbench-gemini-relay-" + "0" * 48,
                  "GBENCH_GEMINI_RELAY_URL": "http://172.17.0.1:4000",
                  "GBENCH_JUDGE_MODELS": "gemini-a"}
    yield bridge


@pytest.fixture
def launcher(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.delenv("GBENCH_TAU2_SAVE_TRACES", raising=False)
    monkeypatch.delenv("TAU2_SAVE_TRACES", raising=False)
    monkeypatch.setattr(TC, "check_tau_image", lambda: (True, ""))
    # the network checks before the container are docker calls of their own
    # (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC.container_url(url))
    monkeypatch.setattr(TC, "_image_id", lambda image: "sha256:feed")
    # the Gemini gate and the relays on the bridge gateway
    # (tests/test_tau_gemini_cascade_and_relays.py)
    monkeypatch.setattr(TC, "_bridge_services", fake_bridge_services)
    removed = []
    monkeypatch.setattr(TC.subprocess, "run",
                        lambda cmd, **kw: removed.append(cmd) or
                        subprocess.CompletedProcess(cmd, 0, b"", b""))
    state = {"removed": removed, "payload": None, "code": 0}

    def launch(cmd, env=None):
        io_dir = _mount(cmd, "/gbench_io")
        state["cmd"] = cmd
        with open(os.path.join(io_dir, "job.json"), encoding="utf-8") as fh:
            state["job"] = json.load(fh)
        state["io_dir"] = io_dir
        if state["payload"] is not None:
            with open(os.path.join(io_dir, "result.json"), "w", encoding="utf-8") as fh:
                json.dump(state["payload"], fh)
        return state["code"], "last lines of output\nKilled"

    monkeypatch.setattr(TC, "_launch_image", launch)
    return state


def test_round_trip_maps_paths_and_records_the_image(launcher, tmp_path):
    from gbench.runners.eval_suites.base import _RUN_KNOBS, set_run_knobs
    set_run_knobs(temperature=0.7)
    try:
        launcher["payload"] = {"ok": True, "result": {
            "accuracy": 50.0, "status": "success",
            "tau2_report": {
                "run_dir": "/gbench_traces/tau2_m_nothink_x",
                "traces": {"airline": "/gbench_traces/tau2_m_nothink_x/airline/results.json"}}}}
        r = TC.run_tau_suite("tau2", ["airline"], "m", "http://127.0.0.1:9/v1", 3, 2,
                             "docs/evals/tau2.md", enable_thinking=True)
    finally:
        _RUN_KNOBS.pop("temperature", None)
    root = str(tmp_path / "results" / "tau_traces")
    rep = r["tau2_report"]
    assert rep["run_dir"] == os.path.join(root, "tau2_m_nothink_x")
    assert rep["traces"]["airline"] == os.path.join(root, "tau2_m_nothink_x", "airline",
                                                    "results.json")
    assert rep["image"] == {"name": TC.tau2_image(), "id": "sha256:feed",
                            "dockerfile": "docker/tau2.Dockerfile",
                            "shell_sandbox_docker_opts": False}
    # the agent reaches an endpoint on this host at host.docker.internal from the bridge network
    assert launcher["job"] == {"eval_name": "tau2", "domains": ["airline"], "model_name": "m",
                               "base_url": "http://host.docker.internal:9/v1", "concurrency": 3,
                               "limit": 2,
                               "docs_url": "docs/evals/tau2.md", "enable_thinking": True,
                               "shard": None,
                               "run_knobs": {"temperature": 0.7, "attempt_count": None}}
    assert _mount(launcher["cmd"], "/gbench_traces") == root and os.path.isdir(root)
    cache = _mount(launcher["cmd"], "/gbench_cache")
    assert cache == os.path.join(os.environ["GBENCH_CACHE_DIR"], "tau2") and os.path.isdir(cache)
    assert not os.path.exists(launcher["io_dir"]), "the exchange directory is removed"
    assert launcher["removed"][-1][:3] == ["docker", "rm", "-f"]


def test_infra_required_from_the_image_is_passed_through_verbatim(launcher):
    msg = ("[INFRA REQUIRED] tau2 cannot run and must not be skipped: tau2 environment needs "
           "GEMINI_API_KEY ... see 'docs/evals/tau2.md'.")
    launcher["payload"] = {"ok": False, "error_type": "RuntimeError", "error": msg}
    launcher["code"] = 1
    with pytest.raises(RuntimeError) as ei:
        TC.run_tau_suite("tau2", ["airline"], "m", "http://x/v1", 1, None, "docs/evals/tau2.md")
    assert str(ei.value) == msg


def test_other_image_errors_become_infra_required(launcher):
    launcher["payload"] = {"ok": False, "error_type": "ValueError", "error": "boom"}
    with pytest.raises(RuntimeError, match=r"\[INFRA REQUIRED\] tau3 .*ValueError: boom"):
        TC.run_tau_suite("tau3", ["banking_knowledge"], "m", "http://x/v1", 1, None,
                         "docs/evals/tau3.md")
    assert "systempaths=unconfined" in launcher["cmd"]


def test_a_container_without_a_result_is_a_hard_error(launcher):
    launcher["code"] = 137
    with pytest.raises(RuntimeError, match="exited 137 without a result.*Killed"):
        TC.run_tau_suite("tau2", ["airline"], "m", "http://x/v1", 1, None, "docs/evals/tau2.md")
    assert launcher["removed"][-1][:3] == ["docker", "rm", "-f"]


def test_disabled_traces_mount_nothing(launcher, monkeypatch):
    monkeypatch.setenv("GBENCH_TAU2_SAVE_TRACES", "")
    launcher["payload"] = {"ok": True, "result": {"tau2_report": {"run_dir": None, "traces": None}}}
    r = TC.run_tau_suite("tau2", ["airline"], "m", "http://x/v1", 1, None, "docs/evals/tau2.md")
    assert _mount(launcher["cmd"], "/gbench_traces") is None
    assert "GBENCH_TAU2_SAVE_TRACES=" in launcher["cmd"]
    assert r["tau2_report"]["run_dir"] is None and r["tau2_report"]["traces"] is None


# --------------------------------------------------------------------------- #
# the driver the image runs
# --------------------------------------------------------------------------- #
def _job(io_dir, **kw):
    job = {"eval_name": "tau2", "domains": ["airline"], "model_name": "m",
           "base_url": "http://x/v1", "concurrency": 2, "limit": None,
           "docs_url": "docs/evals/tau2.md", "enable_thinking": False,
           "run_knobs": {"temperature": None}}
    job.update(kw)
    with open(os.path.join(io_dir, "job.json"), "w", encoding="utf-8") as fh:
        json.dump(job, fh)


def test_image_main_writes_the_result(monkeypatch, tmp_path):
    from gbench.runners.eval_suites.base import _RUN_KNOBS, get_run_knob
    _job(str(tmp_path), run_knobs={"temperature": 0.3}, enable_thinking=True, limit=4)
    seen = {}

    def fake_env(*a, **k):
        seen.update(args=a, kwargs=k, temperature=get_run_knob("temperature"))
        return {"accuracy": 12.5}

    monkeypatch.setattr(TC, "run_tau_env", fake_env)
    try:
        assert TC._image_main(str(tmp_path)) == 0
    finally:
        _RUN_KNOBS.pop("temperature", None)
    assert seen["args"] == ("tau2", ["airline"], "m", "http://x/v1", 2, 4, "docs/evals/tau2.md")
    assert seen["kwargs"] == {"enable_thinking": True, "shard": None} \
        and seen["temperature"] == 0.3
    out = json.loads((tmp_path / "result.json").read_text())
    assert out == {"ok": True, "result": {"accuracy": 12.5}}


def test_image_main_records_the_error(monkeypatch, tmp_path):
    _job(str(tmp_path))

    def boom(*a, **k):
        raise RuntimeError("[INFRA REQUIRED] tau2 cannot run: x")

    monkeypatch.setattr(TC, "run_tau_env", boom)
    assert TC._image_main(str(tmp_path)) == 1
    out = json.loads((tmp_path / "result.json").read_text())
    assert out == {"ok": False, "error_type": "RuntimeError",
                   "error": "[INFRA REQUIRED] tau2 cannot run: x"}


def test_bootstrap_loads_only_tau_common_and_its_helpers(tmp_path):
    """Run the image's program with this interpreter against the repository's suite directory:
    it must import tau_common under the empty package (never gbench or the suite registry) and
    write result.json; here the run stops at a prerequisite check before any network use."""
    io_dir = tmp_path / "io"
    io_dir.mkdir()
    _job(str(io_dir))
    env = {"PATH": os.environ.get("PATH", "/usr/bin"), "HOME": str(tmp_path),
           "TMPDIR": str(tmp_path), "GBENCH_TAU_SANDBOX_MAX_AGE_H": "0",
           "GBENCH_TAU2_SAVE_TRACES": str(tmp_path / "traces")}
    proc = subprocess.run([sys.executable, "-I", "-B", "-X", "importtime", "-c",
                           TC._image_bootstrap(_SUITES_DIR, str(io_dir))],
                          env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 1, proc.stderr[-2000:]
    out = json.loads((io_dir / "result.json").read_text())
    assert out["ok"] is False and out["error"].startswith("[INFRA REQUIRED] tau2")
    # tau_common itself is loaded through importlib (not traced by -X importtime); its logger
    # name shows the package it was loaded under.
    assert "gbench_tau_suites.tau_common - ERROR" in proc.stderr
    imported = [ln.rsplit("|", 1)[-1].strip() for ln in proc.stderr.splitlines()
                if ln.startswith("import time:")]
    ours = {m for m in imported if m.startswith("gbench")}
    assert "gbench_tau_suites.base" in ours
    assert ours <= {"gbench_tau_suites.base", "gbench_tau_suites.sampling",
                    "gbench_tau_suites.swebench_common", "gbench_tau_suites.swe_thread_cap",
                    "gbench_tau_suites.search_tool", "gbench_tau_suites.gemini_relay"}, ours


def test_modules_the_image_loads_import_only_the_stdlib_at_top_level():
    """The image has tau2-bench's packages, not gbench's: tau_common and every sibling module it
    imports must import only the standard library (or each other) at module level."""
    import ast
    # search_tool and gemini_relay: imported in functions
    seen, todo = set(), ["tau_common", "search_tool", "gemini_relay"]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        tree = ast.parse(open(os.path.join(_SUITES_DIR, name + ".py"), encoding="utf-8").read())
        for node in tree.body:
            if isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                roots = [node.module.split(".")[0]]
            elif isinstance(node, ast.ImportFrom):
                todo.extend([node.module] if node.module else [a.name for a in node.names])
                continue
            else:
                continue
            bad = [r for r in roots if r not in sys.stdlib_module_names and r != "__future__"]
            assert not bad, f"{name}.py imports {bad} at module level"
    assert {"tau_common", "base", "swebench_common", "sampling", "swe_thread_cap"} <= seen


# --------------------------------------------------------------------------- #
# guards inside the image
# --------------------------------------------------------------------------- #
def _checkout(tmp_path, litellm="1.81.11"):
    root = tmp_path / "tau2-bench"
    root.mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nname = "tau2"\nversion = "1.0.1"\n'
        'dependencies = ["litellm>=1.80.15,<1.82.7", "rich", "PyYAML>=6.0.2"]\n'
        '[project.optional-dependencies]\nknowledge = ["rank-bm25>=0.2.2"]\n'
        'voice = ["websockets>=13.0"]\n')
    (root / "uv.lock").write_text(
        'version = 1\n'
        '[[package]]\nname = "litellm"\nversion = "1.81.11"\n'
        '[[package]]\nname = "rich"\nversion = "14.2.0"\n'
        '[[package]]\nname = "pyyaml"\nversion = "6.0.3"\n'
        '[[package]]\nname = "rank-bm25"\nversion = "0.2.2"\n'
        '[[package]]\nname = "websockets"\nversion = "15.0.1"\n')
    installed = {"litellm": litellm, "rich": "14.2.0", "pyyaml": "6.0.3", "rank-bm25": "0.2.2"}
    return str(root), installed


def _versions(installed):
    import importlib.metadata as md

    def version(name):
        key = TC._normalize_dist(name)
        if key not in installed:
            raise md.PackageNotFoundError(name)
        return installed[key]
    return version


def test_dependency_check_accepts_the_lock(monkeypatch, tmp_path):
    root, installed = _checkout(tmp_path)
    monkeypatch.setattr("importlib.metadata.version", _versions(installed))
    assert TC.check_tau2_dependencies(root) == (True, "")


def test_dependency_check_rejects_another_litellm(monkeypatch, tmp_path):
    root, installed = _checkout(tmp_path, litellm="1.100.0")
    del installed["rank-bm25"]
    monkeypatch.setattr("importlib.metadata.version", _versions(installed))
    ok, why = TC.check_tau2_dependencies(root)
    assert not ok
    assert "litellm 1.100.0 (uv.lock: 1.81.11)" in why
    assert "rank-bm25 not installed (uv.lock: 0.2.2)" in why
    assert "websockets" not in why, "only the base dependencies and the knowledge extra"
    assert TC.image_build_command() in why


def test_prerequisites_include_the_dependency_check(monkeypatch):
    monkeypatch.setenv("GBENCH_TAU2_USER_LLM", "openai/gpt-4.1-2025-04-14")
    monkeypatch.setenv("GBENCH_TAU2_EVAL_LLM", "openai/gpt-4.1-2025-04-14")
    monkeypatch.setattr(TC, "_import_tau2", lambda: (True, ""))
    monkeypatch.setattr(TC, "check_tau2_revision", lambda: (True, ""))
    monkeypatch.setattr(TC, "check_tau2_dependencies", lambda: (False, "litellm differs"))
    assert TC.check_tau_env_prerequisites() == (False, "litellm differs")


def test_missing_tau2_points_at_the_image(monkeypatch):
    monkeypatch.setattr(TC, "_import_tau2",
                        lambda: (False, "ModuleNotFoundError: No module named 'tau2'"))
    ok, why = TC.check_tau_env_prerequisites()
    assert not ok and TC.image_build_command() in why and "pip install" not in why


class _Sandbox:
    result = (0, "Title: gbench probe\n====\n\ngbench sandbox probe", "")
    raise_on_init = None

    def __init__(self, allow_writes=False, **kw):
        if self.raise_on_init:
            raise self.raise_on_init
        self.exported = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def export_documents(self, documents, file_format="txt"):
        self.exported = documents

    def run_command(self, command, timeout=30):
        assert command == TC._SHELL_PROBE_COMMAND
        assert command.startswith("cat gbench_probe.txt && printenv && cd / && ")
        return self.result


def _fake_tau2(sandbox_cls, variants=None):
    sm = types.ModuleType("tau2.knowledge.sandbox_manager")
    sm.SandboxManager = sandbox_cls
    retrieval = types.ModuleType("tau2.domains.banking_knowledge.retrieval")
    retrieval.RETRIEVAL_VARIANTS = variants or {}
    retrieval.RETRIEVAL_VARIANT_ALIASES = {"AllTools": "alltools"}
    names = ("tau2", "tau2.knowledge", "tau2.domains", "tau2.domains.banking_knowledge")
    mods = {n: types.ModuleType(n) for n in names}
    mods["tau2.knowledge.sandbox_manager"] = sm
    mods["tau2.domains.banking_knowledge.retrieval"] = retrieval
    mods["tau2.domains.banking_knowledge"].retrieval = retrieval
    return mods


@pytest.mark.parametrize("result, raise_on_init, fragment", [
    ((0, "Title: gbench probe\n\ngbench sandbox probe", ""), None, None),
    ((2, "", "usage: srt [-h] {shift,rate,split,break} ... invalid choice"), None,
     "exited 2 (usage: srt"),
    ((1, "", "bwrap: No permissions to create a new namespace"), None, "No permissions"),
    (None, RuntimeError("required binaries are not installed: srt"), "cannot start its sandbox"),
])
def test_shell_probe(result, raise_on_init, fragment):
    cls = type("Sandbox", (_Sandbox,), {"result": result, "raise_on_init": raise_on_init})
    with mock.patch.dict(sys.modules, _fake_tau2(cls)):
        ok, why = TC.check_shell_sandbox()
    if fragment is None:
        assert ok and why == ""
    else:
        assert not ok and fragment in why and "docs/evals/tau3.md" in why


def test_banking_prerequisites_always_probe_the_shell(monkeypatch):
    # alltools (the one config) has the shell tool; the removed retrieval-config knob changes
    # nothing, so every run probes the shell sandbox.
    dense = types.SimpleNamespace(type="embedding", embedder_type="openai",
                                  embedder_model="text-embedding-3-large")
    shell = types.SimpleNamespace(shell=object(), kb_search_dense=dense)
    plain = types.SimpleNamespace(shell=None)
    variants = {"terminal_use": shell, "alltools": shell, "bm25": plain}
    probes = []
    monkeypatch.setattr(TC, "check_shell_sandbox", lambda: probes.append(1) or (False, "probe"))
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    with mock.patch.dict(sys.modules, _fake_tau2(_Sandbox, variants)):
        sys.modules["rank_bm25"] = types.ModuleType("rank_bm25")
        try:
            for override in (None, "AllTools", "terminal_use", "bm25"):
                if override is None:
                    monkeypatch.delenv("GBENCH_TAU2_RETRIEVAL_CONFIG", raising=False)
                else:
                    monkeypatch.setenv("GBENCH_TAU2_RETRIEVAL_CONFIG", override)
                monkeypatch.delenv("TAU2_RETRIEVAL_CONFIG", raising=False)
                assert TC.check_banking_prerequisites() == (False, "probe"), override
        finally:
            sys.modules.pop("rank_bm25", None)
    assert len(probes) == 4


class _FakeTau2Run:
    """Just enough of tau2 for run_tau_env: TextRunConfig, run_domain, compute_metrics."""

    def __init__(self):
        self.configs = []

    def modules(self):
        fake = self

        class TextRunConfig:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        def run_domain(config):
            fake.configs.append(config)
            sims = [types.SimpleNamespace(reward_info=types.SimpleNamespace(reward=1.0),
                                          termination_reason="agent_stop")]
            return types.SimpleNamespace(simulations=sims, num_trials=1)

        def compute_metrics(results):
            return types.SimpleNamespace(pass_hat_ks={1: 1.0})

        mods = {n: types.ModuleType(n) for n in (
            "tau2", "tau2.data_model", "tau2.data_model.simulation", "tau2.run", "tau2.metrics",
            "tau2.metrics.agent_metrics")}
        mods["tau2.data_model.simulation"].TextRunConfig = TextRunConfig
        mods["tau2.run"].run_domain = run_domain
        mods["tau2.metrics.agent_metrics"].compute_metrics = compute_metrics
        return mods


@pytest.mark.parametrize("endpoint_key, expected", [(None, "EMPTY"),
                                                    ("endpoint-key-123", "endpoint-key-123")])
def test_agent_endpoint_key_stays_out_of_llm_args(monkeypatch, tmp_path, endpoint_key, expected):
    """The agent's endpoint key comes from GBENCH_TAU_AGENT_KEY (the host's OPENAI_API_KEY), never
    from an OPENAI_API_KEY in the image's environment (the launcher passes none; one set there is
    left alone and unused). It is removed from the environment and kept out of llm_args, which
    tau2 writes into every results.json."""
    if endpoint_key is None:
        monkeypatch.delenv(TC._AGENT_KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(TC._AGENT_KEY_ENV, endpoint_key)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-provider-key")
    monkeypatch.setenv("GBENCH_TAU2_SAVE_TRACES", str(tmp_path / "traces"))
    monkeypatch.setenv("GBENCH_TAU2_NUM_TRIALS", "1")
    for name, value in (("check_tau_env_prerequisites", lambda: (True, "")),
                        ("check_banking_prerequisites", lambda: (True, "")),
                        ("_install_robustness_patches", lambda *a, **k: None),
                        ("_install_credential_guards", lambda *a, **k: None),
                        ("_configure_evaluator_llm", lambda *a, **k: None),
                        ("_quiet_tau2_noise", lambda: None),
                        ("_prune_stale_search_sandboxes", lambda: None),
                        ("_refuse_resume_prompts", lambda: None),
                        ("_tau2_version", lambda: "1.0.1"),
                        ("_setup_banking_retrieval", lambda: ("alltools-gemini", None))):
        monkeypatch.setattr(TC, name, value)
    fake = _FakeTau2Run()
    routes = []
    real_run_domain = fake.modules()["tau2.run"].run_domain
    mods = fake.modules()
    mods["tau2.run"].run_domain = lambda config: (routes.append(dict(TC._AGENT_ROUTE)),
                                                  real_run_domain(config))[1]
    with mock.patch.dict(sys.modules, mods):
        TC.run_tau_env("tau3", ["banking_knowledge"], "m", "http://x/v1", 1, None,
                       "docs/evals/tau3.md")
    assert "api_key" not in fake.configs[0].llm_args_agent
    assert routes == [{"model": "openai/m", "api_base": "http://x/v1", "api_key": expected}]
    assert TC._AGENT_KEY_ENV not in os.environ, "removed before tau2 runs"
    assert TC._AGENT_ROUTE == {} and TC._SECRETS == [], "no key kept after the run"
    assert os.environ["OPENAI_API_KEY"] == "openai-provider-key"


# --------------------------------------------------------------------------- #
# comparability text
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("user, gap", [("gemini/gemini-3.6-flash", True), ("openai/gpt-5.2", False),
                                       ("gpt-4.1-2025-04-14", False), ("openai/gpt-4o", True)])
def test_user_simulator_gap_names_the_leaderboards_standard(monkeypatch, user, gap):
    monkeypatch.setenv("GBENCH_TAU2_USER_LLM", user)
    monkeypatch.setenv("GBENCH_TAU2_EVAL_LLM", "gpt-4.1-2025-04-14")
    monkeypatch.delenv("GBENCH_TAU2_USER_TEMPERATURE", raising=False)
    monkeypatch.delenv("TAU2_USER_TEMPERATURE", raising=False)
    monkeypatch.setattr(TC, "_LAST_AGENT_TEMPERATURE", {"value": 0.0})
    gaps = TC._comparability_gaps(["retail"], 4, None, 0)
    assert bool(gaps) is gap
    if gap:
        assert f"user simulator {user}" in gaps[0]
        assert "instead of gpt-5.2 (recommended by docs/leaderboard-submission.md and used by " \
               "Sierra's 2026 standard text entries)" in gaps[0]
        assert "custom submission" in gaps[0] and "gpt-4.1-2025-04-14" in gaps[0]
        assert "leaderboard's standard" not in gaps[0]


@pytest.mark.parametrize("eval_name", ["tau2", "tau3"])
def test_agent_temperature_is_one_with_thinking_and_upstreams_zero_without(monkeypatch, eval_name):
    # the agent (the model under test) samples at 1.0 with --thinking (upstream's 0.0 does not
    # apply to a reasoning run); a no-think run keeps upstream's DEFAULT_LLM_ARGS_AGENT
    from gbench.runners.eval_suites import base
    for var in ("GBENCH_TAU2_TEMPERATURE", "GBENCH_TAU3_TEMPERATURE", "TAU2_TEMPERATURE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delitem(base._RUN_KNOBS, "temperature", raising=False)
    monkeypatch.setattr(TC, "_LAST_AGENT_TEMPERATURE", {})
    think = TC._build_agent_args("http://x/v1", True, eval_name)
    assert think["temperature"] == 1.0
    assert think["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert TC._LAST_AGENT_TEMPERATURE["source"] == "thinking"
    gaps = TC._comparability_gaps(["retail"], 4, None, 0)
    assert any("agent sampled at temperature 1.0 (--thinking samples the model under test at "
               "1.0)" in g and "upstream sends only temperature 0.0" in g for g in gaps)
    cfg = TC._run_config(eval_name, ["retail"], "m", True, None, 4)
    assert (cfg["agent_temperature"], cfg["agent_temperature_source"]) == (1.0, "thinking")
    plain = TC._build_agent_args("http://x/v1", False, eval_name)
    assert plain["temperature"] == 0.0 and "extra_body" not in plain
    assert TC._LAST_AGENT_TEMPERATURE["source"] == "default"
    assert not any("agent sampled" in g for g in TC._comparability_gaps(["retail"], 4, None, 0))
    cfg = TC._run_config(eval_name, ["retail"], "m", False, None, 4)
    assert (cfg["agent_temperature"], cfg["agent_temperature_source"]) == (0.0, "default")
    # the operator's choice still wins in a thinking run
    monkeypatch.setenv(f"GBENCH_{eval_name.upper()}_TEMPERATURE", "0.4")
    assert TC._build_agent_args("http://x/v1", True, eval_name)["temperature"] == 0.4


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #
def test_docs_state_the_image_and_its_pins():
    tau2, tau3 = _read("docs", "evals", "tau2.md"), _read("docs", "evals", "tau3.md")
    for doc in (tau2, tau3):
        assert TC.image_build_command() in doc
        assert TC.TAU2_BENCH_REVISION in doc and "--max-output-tokens" in doc
        assert "GBENCH_TAU2_ENV_RUN=1 gbench" not in doc
        assert "pip install -e ./tau2-bench" not in doc
    flat2 = " ".join(tau2.split())
    assert "`litellm` 1.81.11" in flat2 and "`websockets` 15.0.1" in flat2
    assert "GBENCH_TAU2_IMAGE" in tau2 and "No provider key (Gemini, OpenAI, ...) is ever sent to " \
        "your endpoint" in flat2
    assert "`gpt-5.2` is recommended by `docs/leaderboard-submission.md` and used by Sierra's " \
        "2026 standard text entries" in flat2
    assert "leaderboard's standard text entries use" not in flat2
    flat3 = " ".join(tau3.split())
    assert "@anthropic-ai/sandbox-runtime@0.0.23" in flat3
    assert "sudo apt install ripgrep socat bubblewrap python3-pysrt" not in tau3
    assert "systempaths=unconfined" in flat3 and "Shell probe" in tau3
