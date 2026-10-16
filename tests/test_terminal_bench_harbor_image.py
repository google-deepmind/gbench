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

"""The local harness images: terminal_bench runs Harbor 0.20.0 from its own pinned image, and
every Dockerfile under gbench/docker pins its base image by digest.

terminal_bench
  * the image holds Harbor HARBOR_VERSION and the TB2.1 leaderboard lock's package versions
    (constraints file, build-time check), the Docker CLI and plugins by sha256, on a base pinned
    by digest; its labels are what the prerequisite check requires;
  * Harbor runs in that image (docker-out-of-docker): the socket, the jobs directory at the same
    path, gbench's Harbor cache and the salvage patch are the only mounts; the container gets
    only the variables gbench names, never a credential; an interrupted run stops the container;
  * the task cache gbench reads (metadata probe, task categories) is the one the container uses.

Docker, Harbor's registry and the model endpoint are stubbed throughout.
"""

import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from gbench.runners.eval_suites import base as B
from gbench.runners.eval_suites import terminal_bench as TB

REPO = Path(__file__).resolve().parents[1]
DOCKER_DIR = REPO / "gbench" / "docker"
DOCKERFILE = DOCKER_DIR / "terminal_bench.Dockerfile"
CONSTRAINTS = DOCKER_DIR / "terminal_bench.constraints.txt"
SECRET = "sk-test-endpoint-key-0123456789abcdef"
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")


def _args(text):
    """{name: default} of the ARG lines of a Dockerfile."""
    return dict(m.groups() for m in re.finditer(r"^ARG ([A-Z0-9_]+)=(\S+)$", text, re.M))


def _verify_env_module():
    spec = importlib.util.spec_from_file_location(
        "terminal_bench_verify_env", DOCKER_DIR / "terminal_bench_verify_env.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =========================================================================== #
# The image
# =========================================================================== #
def test_the_image_pins_its_base_its_tools_and_harbor():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert text.isascii()
    froms = re.findall(r"^FROM (\S+)", text, re.M)
    assert froms == ["python:3.12-slim@sha256:"
                     "78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"]
    args = _args(text)
    assert args["HARBOR_VERSION"] == TB.HARBOR_VERSION == "0.20.0"
    assert args["TB21_LOCK_COMMIT"] == TB.TB21_LOCK_COMMIT
    for tool in ("DOCKER", "COMPOSE", "BUILDX"):
        assert re.fullmatch(r"[0-9a-f]{64}", args[f"{tool}_SHA256"]), tool
        assert f'echo "${{{tool}_SHA256}}  ' in text and "sha256sum -c -" in text
    # wheels only, the lock's versions, and the check that nothing else got installed
    assert ('pip install --only-binary=:all: -c /opt/harbor/constraints.txt' in
            " ".join(text.split()))
    assert "python -m pip check" in text and "/opt/harbor/verify_env.py" in text
    assert "harbor --version" in text
    # the labels the prerequisite check requires come from the same ARGs
    assert 'org.gbench.harbor="${HARBOR_VERSION}"' in text
    assert 'org.gbench.terminal_bench_lock="${TB21_LOCK_COMMIT}"' in text
    assert set(TB.HARBOR_IMAGE_LABELS) == {"org.gbench.harbor", "org.gbench.terminal_bench_lock"}
    # the documented build command is the one gbench names, and produces the tag gbench runs
    assert "#   " + TB.harbor_image_build_command() in text.splitlines()
    assert TB.harbor_image_build_command() == (
        "docker build -t gbench-terminal-bench:harbor-0.20.0 "
        "-f gbench/docker/terminal_bench.Dockerfile gbench/docker")
    assert TB.harbor_image() == TB.HARBOR_IMAGE_DEFAULT == "gbench-terminal-bench:harbor-0.20.0"
    # any uid can run it: HOME is world-writable in the image
    assert "chmod 1777 /harbor-home" in text and "ENV HOME=/harbor-home" in text


def test_the_constraints_are_the_leaderboard_locks_versions():
    lines = [ln for ln in CONSTRAINTS.read_text(encoding="utf-8").splitlines()]
    assert all(ln.isascii() for ln in lines)
    header = " ".join(ln for ln in lines if ln.startswith("#"))
    assert TB.TB21_LOCK_COMMIT in header and "leaderboard/uv.lock" in header
    pins = [ln for ln in lines if ln and not ln.startswith("#")]
    assert all(re.fullmatch(r"[a-z0-9][a-z0-9._-]*==[0-9][0-9A-Za-z.+-]*", p) for p in pins), pins
    names = [p.split("==")[0] for p in pins]
    assert len(names) == len(set(names)) == 83
    got = dict(p.split("==") for p in pins)
    assert got["harbor"] == TB.HARBOR_VERSION
    assert got["litellm"] == "1.91.0" and got["openai"] == "2.44.0"


def test_the_build_check_refuses_anything_the_constraints_do_not_pin(tmp_path, monkeypatch):
    mod = _verify_env_module()
    pins = {"harbor": "0.20.0", "litellm": "1.91.0"}
    assert mod.problems(pins, {"harbor": "0.20.0", "litellm": "1.91.0", "pip": "25.0.1"},
                        "0.20.0") == []
    bad = mod.problems(pins, {"harbor": "0.22.0", "litellm": "1.92.0", "extra-pkg": "1"},
                       "0.20.0")
    assert any("harbor is '0.22.0', not 0.20.0" in b for b in bad)
    assert any("litellm is 1.92.0, the constraints pin 1.91.0" in b for b in bad)
    assert any("extra-pkg==1 is installed but not pinned" in b for b in bad)
    assert mod.canonical("Typing_Extensions") == "typing-extensions"

    constraints = tmp_path / "c.txt"
    constraints.write_text("# header\nharbor==0.20.0\nLiteLLM==1.91.0  # inline\n")
    assert mod.read_constraints(str(constraints)) == pins
    freeze = tmp_path / "freeze.txt"
    monkeypatch.setattr(mod, "installed", lambda: {"harbor": "0.20.0", "litellm": "1.91.0"})
    assert mod.main(["x", str(constraints), "0.20.0", str(freeze)]) == 0
    assert freeze.read_text() == "harbor==0.20.0\nlitellm==1.91.0\n"
    monkeypatch.setattr(mod, "installed", lambda: {"harbor": "0.20.0", "litellm": "1.90.0"})
    assert mod.main(["x", str(constraints), "0.20.0", str(tmp_path / "f2.txt")]) == 1
    assert not (tmp_path / "f2.txt").exists()


def _header_build_commands(text):
    """The `docker build ...` commands of a Dockerfile's header comments, continuation lines
    joined."""
    out, cur = [], None
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        body = line.lstrip("#").strip()
        if cur is not None:
            cur += " " + body.rstrip("\\").strip()
            if not body.endswith("\\"):
                out.append(cur)
                cur = None
        elif body.startswith("docker build"):
            if body.endswith("\\"):
                cur = body.rstrip("\\").strip()
            else:
                out.append(body)
    return out


def test_every_gbench_docker_image_pins_its_base_by_digest_and_names_its_own_file():
    files = sorted(DOCKER_DIR.glob("*.Dockerfile"))
    assert DOCKERFILE in files
    for path in files:
        text = path.read_text(encoding="utf-8")
        args = _args(text)
        for ref in re.findall(r"^FROM (\S+)", text, re.M):
            m = re.fullmatch(r"\$\{?([A-Z0-9_]+)\}?", ref)
            ref = args.get(m.group(1), "") if m else ref
            assert DIGEST.search(ref), f"{path.name}: FROM {ref} is not pinned by digest"
        builds = _header_build_commands(text)
        assert builds, f"{path.name}: no documented build command"
        for cmd in builds:
            f = cmd.split(" -f ", 1)[1].split()[0]
            assert f == f"gbench/docker/{path.name}", (path.name, cmd)


# =========================================================================== #
# The harness container
# =========================================================================== #
def test_harbor_runs_in_the_image_with_only_the_named_mounts_and_variables(monkeypatch,
                                                                          tmp_path):
    monkeypatch.delenv("GBENCH_TB_SALVAGE_PATCH", raising=False)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    jobs = tmp_path / "jobs"
    cmd = TB.harbor_command(["run", "-d", TB.DATASET_SPEC], jobs_dir=str(jobs), name="c1",
                            env_names=["OPENAI_API_KEY", "OPENAI_BASE_URL", "HARBOR_TELEMETRY"])
    at = cmd.index(TB.HARBOR_IMAGE_DEFAULT)
    opts = cmd[:at]
    assert opts[:2] == ["docker", "run"] and "--rm" in opts and "--init" in opts
    assert opts[opts.index("--network") + 1] == "host"
    assert opts[opts.index("--cap-drop") + 1] == "ALL"
    assert opts[opts.index("--security-opt") + 1] == "no-new-privileges"
    assert opts[opts.index("--name") + 1] == "c1"
    assert "--privileged" not in opts
    if hasattr(os, "getuid"):
        assert opts[opts.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    mounts = [opts[i + 1] for i, a in enumerate(opts) if a == "-v"]
    patch = os.path.abspath(TB.terminus_salvage_patch.__file__)
    assert mounts == [
        "/var/run/docker.sock:/var/run/docker.sock",
        f"{jobs}:{jobs}:rw",                                  # the same path on both sides
        f"{tmp_path / 'cache' / 'harbor'}:/harbor-home/.cache/harbor:rw",
        f"{patch}:/opt/gbench/terminus_salvage_patch.py:ro"]
    envs = [opts[i + 1] for i, a in enumerate(opts) if a == "-e"]
    # by name only (docker copies the value from the client's environment), plus HOME and
    # Harbor's telemetry off, whatever the host's HARBOR_TELEMETRY is
    assert envs == ["HOME=/harbor-home", "HARBOR_TELEMETRY=0", "OPENAI_API_KEY", "OPENAI_BASE_URL"]
    assert cmd[at + 1:at + 3] == ["python", "-c"]
    assert cmd[at + 4:] == ["run", "-d", TB.DATASET_SPEC]
    compile(cmd[at + 3], "<bootstrap>", "exec")


def test_the_container_gets_only_the_openai_harbor_and_proxy_variables():
    env = {"OPENAI_API_KEY": "EMPTY", "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
           "HARBOR_TELEMETRY": "0", "HARBOR_REGISTRY_WEBSITE_URL": "https://h",
           "HTTPS_PROXY": "http://proxy:3128", "no_proxy": "localhost", "PATH": "/usr/bin",
           "HOME": "/home/u", "HF_TOKEN_PATH": "/x", "GBENCH_TB_PARSER": "xml"}
    # HARBOR_TELEMETRY is never passed from the host: every harness container gets it as 0
    assert TB.container_env_names(env) == [
        "OPENAI_API_KEY", "OPENAI_BASE_URL", "HARBOR_REGISTRY_WEBSITE_URL",
        "HTTPS_PROXY", "no_proxy"]


def test_the_shard_enumeration_container_has_no_docker_socket(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HARBOR_TELEMETRY", "0")
    monkeypatch.setenv("VLLM_API_KEY", SECRET)
    seen = {}

    def run(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw["env"]
        return subprocess.CompletedProcess(cmd, 0, stdout='["terminal-bench/a"]\n', stderr="")
    monkeypatch.setattr(TB.subprocess, "run", run)
    assert TB._dataset_task_names(TB.DATASET_SPEC) == ["terminal-bench/a"]
    cmd = seen["cmd"]
    assert "--group-add" not in cmd and not any("docker.sock" in a for a in cmd)
    assert "HARBOR_TELEMETRY=0" in cmd and SECRET not in " ".join(cmd)
    assert SECRET not in json.dumps(seen["env"])
    assert (tmp_path / "cache" / "harbor").is_dir()        # created, so docker never makes it


def _trial(root, name, reward):
    trial = os.path.join(root, "job", name)
    os.makedirs(os.path.join(trial, "agent"))
    with open(os.path.join(trial, "result.json"), "w") as f:
        json.dump({"trial_name": name, "task_name": "terminal-bench/t0",
                   "task_id": {"org": "terminal-bench", "name": "t0", "ref": "sha256:" + "a" * 64},
                   "verifier_result": {"rewards": {"reward": reward}}}, f)


class _Proc:
    returncode = 0
    stdout = iter(())

    def poll(self):
        return 0

    def wait(self):
        return 0


def _stub_run(monkeypatch, tmp_path, popen):
    for k in ("GBENCH_TB_TURN_MAX_TOKENS", "GBENCH_TB_TIMEOUT_MULTIPLIER", "GBENCH_TB_PARSER",
              "GBENCH_TB_CONTAINER_THREADS", "GBENCH_TERMINAL_BENCH_TEMPERATURE", "GBENCH_SHARD",
              "GBENCH_TB_KEEP_JOBS_DIR", "VLLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    B._RUN_KNOBS.pop("temperature", None)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    # Harbor's jobs directory under the test's own directory (kept there, never in /tmp)
    (tmp_path / "jobs").mkdir()
    monkeypatch.setenv("GBENCH_TB_KEEP_JOBS_DIR", str(tmp_path / "jobs"))

    def check():
        TB._VERIFIED_IMAGE.clear()
        TB._VERIFIED_IMAGE.update({"image": TB.HARBOR_IMAGE_DEFAULT, "id": "sha256:feed",
                                   "labels": dict(TB.HARBOR_IMAGE_LABELS)})
        return True, ""
    monkeypatch.setattr(TB, "check_terminal_bench_prerequisites", check)
    monkeypatch.setattr(TB, "require_task_metadata_blocked", lambda task_names=None: None)
    monkeypatch.setattr(TB, "prepare_task_images",
                        lambda task_names=None: ({"cached": 0}, {"present": 0}))
    monkeypatch.setattr(TB, "dns_recheck", lambda task_names, hosts: list(hosts))
    monkeypatch.setattr(TB, "FULL_TASK_COUNT", 1)
    monkeypatch.setattr(TB, "TERMINAL_BENCH_TRIALS", 1)
    monkeypatch.setattr(TB.subprocess, "Popen", popen)


def test_a_run_hands_the_harness_container_no_credential(monkeypatch, tmp_path):
    made, seen = [], {}

    class Relay:
        TIMEOUT_S = 10

        def __init__(self, upstream, bind_host, key=None, name=""):
            self.upstream, self.bind_host, self.key = upstream, bind_host, key
            made.append(self)

        def start(self):
            return self

        def stop(self):
            pass

        url = "http://127.0.0.1:45678/v1"

        def record(self):
            return {"listens": self.url}
    monkeypatch.setattr(TB.swebench_common, "EndpointRelay", Relay)

    def popen(cmd, **kw):
        seen["cmd"], seen["env"] = cmd, kw["env"]
        jobs = cmd[cmd.index("--jobs-dir") + 1]
        _trial(jobs, "t0__1", 1.0)
        return _Proc()
    _stub_run(monkeypatch, tmp_path, popen)
    monkeypatch.setenv("VLLM_API_KEY", SECRET)
    # Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "TESTfixtureKEY000000000000000000000")
    monkeypatch.setenv("HARBOR_TELEMETRY", "1")          # the host's value never reaches Harbor
    cache = tmp_path / "cache" / "harbor" / "tasks" / "packages" / "terminal-bench" / "t0"
    (cache / ("a" * 64)).mkdir(parents=True)
    (cache / ("a" * 64) / "task.toml").write_text('[metadata]\ncategory = "games"\n')
    res = TB.run_terminal_bench("m", "http://h:1/v1", 1)

    cmd, env = seen["cmd"], seen["env"]
    at = cmd.index(TB.HARBOR_IMAGE_DEFAULT)
    envs = [cmd[i + 1] for i, a in enumerate(cmd[:at]) if a == "-e"]
    assert envs == ["HOME=/harbor-home", "HARBOR_TELEMETRY=0", "OPENAI_API_KEY", "OPENAI_BASE_URL"]
    assert env["OPENAI_API_KEY"] == "EMPTY" and env["OPENAI_BASE_URL"] == Relay.url
    assert made[0].key == SECRET and made[0].bind_host == "127.0.0.1"
    assert SECRET not in " ".join(cmd) and SECRET not in json.dumps(env)
    assert "GEMINI_API_KEY" not in env and "VLLM_API_KEY" not in env
    # the jobs directory is mounted at the path Harbor is told to write to
    jobs = cmd[cmd.index("--jobs-dir") + 1]
    assert os.path.isabs(jobs) and f"{jobs}:{jobs}:rw" in cmd
    # the task cache the container uses is the one gbench read the category from
    assert res["sample_traces"][0]["category"] == "games"
    assert res["harbor_version"] == TB.HARBOR_VERSION
    assert res["harness_image"] == {"image": TB.HARBOR_IMAGE_DEFAULT, "id": "sha256:feed",
                                    "harbor": TB.HARBOR_VERSION, "lock": TB.TB21_LOCK_COMMIT}
    assert res["harness_telemetry"] == {"harbor_telemetry": "disabled", "env": "HARBOR_TELEMETRY=0"}
    assert res["status"] == "success" and res["accuracy"] == 100.0


def test_an_interrupted_run_stops_the_harness_container(monkeypatch, tmp_path):
    stopped = []

    class Running(_Proc):
        returncode = None

        def poll(self):
            return None

        def terminate(self):
            stopped.append("terminate")

    def popen(cmd, **kw):
        stopped.append(cmd[cmd.index("--name") + 1])
        return Running()
    _stub_run(monkeypatch, tmp_path, popen)

    def interrupted(_s):
        raise KeyboardInterrupt
    monkeypatch.setattr(TB.time, "sleep", interrupted)
    monkeypatch.setattr(TB.subprocess, "run",
                        lambda argv, **kw: stopped.append(argv[:3]) or
                        subprocess.CompletedProcess(argv, 0))
    with pytest.raises(KeyboardInterrupt):
        TB.run_terminal_bench("m", "http://h:1/v1", 1)
    name = stopped[0]
    assert name.startswith("gbench-terminal-bench-")
    assert stopped[1:] == ["terminate", ["docker", "stop", "--time"], ["docker", "rm", "-f"]]


def test_a_finished_run_leaves_the_container_to_docker(monkeypatch, tmp_path):
    calls = []

    def popen(cmd, **kw):
        _trial(cmd[cmd.index("--jobs-dir") + 1], "t0__1", 0.0)
        return _Proc()
    _stub_run(monkeypatch, tmp_path, popen)
    monkeypatch.setattr(TB, "_stop_container", lambda *a, **k: calls.append(a))
    res = TB.run_terminal_bench("m", "http://h:1/v1", 1)
    assert res["accuracy"] == 0.0 and calls == []          # --rm removes it


def test_without_cached_tasks_the_hint_downloads_with_the_image(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    seen = {}

    def gate(eval_name, images, **kw):
        seen.update(kw, images=images)
    monkeypatch.setattr(TB.swebench_common, "require_task_network_blocked", gate)
    TB.require_task_metadata_blocked()
    cache = tmp_path / "cache" / "harbor"
    # no cached task: the harness image is the probe's image, and the hint names the empty cache
    assert seen.get("pull") is None and seen["images"] == [TB.harbor_image()]
    assert "holds no task of the dataset" in seen["hint"]
    assert TB.harbor_download_command() == (
        f"mkdir -p {cache} && docker run --rm --network host --user \"$(id -u):$(id -g)\" "
        f"-v {cache}:/harbor-home/.cache/harbor -e HARBOR_TELEMETRY=0 "
        f"gbench-terminal-bench:harbor-0.20.0 harbor dataset download {TB.DATASET_SPEC} --cache")


def test_the_guide_documents_the_image():
    doc = " ".join((REPO / "docs" / "evals" / "terminal_bench.md").read_text(
        encoding="utf-8").split())
    for needle in (TB.harbor_image_build_command(), "harbor --version", "`harness_image`",
                   "docker-out-of-docker", "/var/run/docker.sock", "no-new-privileges",
                   "$GBENCH_CACHE_DIR/harbor", "/harbor-home/.cache/harbor",
                   f"harbor dataset download {TB.DATASET_SPEC} --cache", "HARBOR_TELEMETRY",
                   "org.gbench.harbor", "org.gbench.terminal_bench_lock",
                   "terminal_bench.constraints.txt"):
        assert needle in doc, needle
    assert "install 'harbor" not in doc and "uv tool install" not in doc
