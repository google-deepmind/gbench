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

"""The cluster scripts (scripts/eval_cluster.sh, async_proxy.py, auto_cleanup_after_gbench.sh,
deploy_a3_vms.sh), the GKE manifest (vllm-gke.yaml) and the CI helper scripts.

* The terminal_bench unit tests run without Harbor (Harbor 0.20.0 needs Python 3.12, and CI
  also runs 3.11) and without Harbor's task registry (a network service): run in a child pytest
  whose imports of `harbor` fail, they still pass.
* scripts/ci_skip_gate.py counts a requirement whose environment marker is false on the running
  interpreter as undeclared there, so a skip it causes is noted, not failed.
* scripts/eval_cluster.sh starts the server it evaluates: a thinking run (its default) needs the
  model's reasoning parser, as a local gbench --thinking run does; `none` states the model has
  none and reaches vLLM as nothing (the manifest's launcher and deploy_a3_vms.sh drop it).

The cluster tools (kubectl, gcloud, gbench, docker) are fakes on PATH; a fake `kubectl
port-forward` stands up a tiny OpenAI-style server on the local port it is given, answering
/v1/models with FAKE_SERVED_MODEL. Nothing here contacts a real cluster, a model server or the
network: every port is a free ephemeral port."""

import asyncio
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))
import async_proxy  # noqa: E402
import ci_skip_gate as G  # noqa: E402

_FAKE_KUBECTL = r'''#!PYTHON
import http.server, json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write("kubectl " + json.dumps(args) + "\n")
if args[:1] == ["get"] and "pods" in args:
    print(os.environ.get("FAKE_PODS", "pod-a pod-b")); sys.exit(0)
if args[:1] == ["get"] and "deployment" in args:
    print("vllm/vllm-openai:v9@sha256:" + "0" * 64); sys.exit(0)
if args[:1] == ["port-forward"]:
    if os.environ.get("FAKE_PF_FAIL"):
        print("error: unable to listen on any of the requested ports", flush=True); sys.exit(1)
    port = int(args[-1].split(":")[0])
    served = os.environ.get("FAKE_SERVED_MODEL", "")

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": [{"id": served}]} if self.path == "/v1/models"
                              else {"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"Forwarding from 127.0.0.1:{port} -> 8000", flush=True)
    srv.serve_forever()
sys.exit(0)
'''

_FAKE_GCLOUD = r'''#!PYTHON
import json, os, subprocess, sys
args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write("gcloud " + json.dumps(args) + "\n")
if args[:3] == ["compute", "instances", "list"]:
    print("10.0.0.1\n10.0.0.2\n10.0.0.3\n10.0.0.4"); sys.exit(0)
if args[:2] == ["compute", "ssh"]:
    cmd = next(a for a in args if a.startswith("--command="))[len("--command="):]
    sys.exit(subprocess.run(["bash", "-c", cmd]).returncode)
sys.exit(0)
'''

_FAKE_GBENCH = r'''#!/usr/bin/env bash
ep=""; prev=""
for a in "$@"; do [ "$prev" = "--remote-endpoint" ] && ep="$a"; prev="$a"; done
PYTHON -c 'import json, os, sys; json.dump(sys.argv[1:], open(os.environ["FAKE_GBENCH_ARGS"], "w"))' "$@"
curl -sf "$ep/models" > "$FAKE_GBENCH_MODELS" || echo "unreachable" > "$FAKE_GBENCH_MODELS"
exit "${FAKE_GBENCH_RC:-0}"
'''

_FAKE_DOCKER = r'''#!PYTHON
import json, os, sys
with open(os.environ["FAKE_DOCKER_LOG"], "a") as f:
    f.write(json.dumps(sys.argv[1:]) + "\n")
'''


def _write_exe(path: Path, text: str) -> None:
    path.write_text(text.replace("#!PYTHON", "#!" + sys.executable)
                    .replace("PYTHON -c", sys.executable + " -c"))
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@pytest.fixture
def fakes(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_exe(bin_dir / "kubectl", _FAKE_KUBECTL)
    _write_exe(bin_dir / "gcloud", _FAKE_GCLOUD)
    _write_exe(bin_dir / "gbench", _FAKE_GBENCH)
    _write_exe(bin_dir / "docker", _FAKE_DOCKER)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GBENCH_", "HF_TOKEN"))}
    env.update({
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "PYTHON": sys.executable,
        "FAKE_LOG": str(tmp_path / "calls.log"),
        "FAKE_GBENCH_ARGS": str(tmp_path / "gbench_args.json"),
        "FAKE_GBENCH_MODELS": str(tmp_path / "gbench_models.json"),
        "FAKE_DOCKER_LOG": str(tmp_path / "docker.log"),
        "FAKE_SERVED_MODEL": "org/model-x",
        "HOME": str(tmp_path / "home"),
    })
    return tmp_path, env


def _cluster_args(tmp_path, *extra):
    # a thinking run (the script's default) needs the model's reasoning parser
    return [str(SCRIPTS / "eval_cluster.sh"), "org/model-x", "gpqa_diamond mmlu_pro",
            "--gpu", "h100", "--num-gpus", "8", "--tp", "2", "--pool", "p1", "--nodes", "2",
            "--project", "proj", "--cluster", "c1", "--region", "r1",
            "--reasoning-parser", "rp1",
            "--results-dir", str(tmp_path / "results"), *extra]


def _run(cmd, env, timeout=90):
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)


def _calls(tmp_path):
    out = []
    for line in (tmp_path / "calls.log").read_text().splitlines():
        tool, _, rest = line.partition(" ")
        out.append((tool, json.loads(rest)))
    return out


def _listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


# -- eval_cluster.sh ---------------------------------------------

def test_eval_cluster_runs_gbench_through_its_own_proxy_and_cleans_up(fakes):
    tmp_path, env = fakes
    r = _run(_cluster_args(tmp_path, "--tool-call-parser", "tp1",
                           "--vllm-serve-args", "--trust-remote-code --limit-mm-per-prompt "
                                                "'{\"image\": 8}'"), env)
    assert r.returncode == 0, r.stdout + r.stderr
    gargs = json.loads((tmp_path / "gbench_args.json").read_text())
    ep = gargs[gargs.index("--remote-endpoint") + 1]
    m = re.fullmatch(r"http://127\.0\.0\.1:(\d+)/v1", ep)
    assert m and int(m.group(1)) not in range(8000, 8011)
    assert gargs[gargs.index("--models") + 1] == "org/model-x"
    evals = gargs[gargs.index("--evals") + 1:gargs.index("--models")]
    assert evals == ["gpqa_diamond", "mmlu_pro"]
    # gbench reached the model through the proxy
    served = json.loads((tmp_path / "gbench_models.json").read_text())
    assert [d["id"] for d in served["data"]] == ["org/model-x"]
    calls = _calls(tmp_path)
    set_env = next(a for t, a in calls if t == "kubectl" and a[:2] == ["set", "env"])
    assert "MODEL_NAME=org/model-x" in set_env and "TENSOR_PARALLEL_SIZE=2" in set_env
    assert "DATA_PARALLEL_SIZE=4" in set_env and "TOOL_CALL_PARSER=tp1" in set_env
    assert "MAX_MODEL_LEN=" in set_env          # empty: the model's native context
    assert ("VLLM_SERVE_ARGS=--trust-remote-code --limit-mm-per-prompt '{\"image\": 8}'"
            in set_env)
    pf = [a for t, a in calls if t == "kubectl" and a[:1] == ["port-forward"]]
    assert len(pf) == 2 * 4                    # 2 pods x DP tunnels
    assert all(a[1:3] == ["--address", "127.0.0.1"] for a in pf)
    ports = [int(a[-1].split(":")[0]) for a in pf]
    assert len(set(ports)) == len(ports) and not set(ports) & set(range(8000, 8011))
    # the record of what served the run
    serving = json.loads((tmp_path / "results" / "cluster" / "serving.json").read_text())
    assert serving["tensor_parallel_size"] == 2 and serving["max_model_len"] is None
    assert serving["vllm_serve_args"] == ["--trust-remote-code", "--limit-mm-per-prompt",
                                          '{"image": 8}']
    # cleanup: its own tunnels and proxy are gone, the deployment and pool scaled to 0
    time.sleep(0.5)
    assert not any(_listening(p) for p in ports + [int(m.group(1))])
    assert any(t == "kubectl" and a[:3] == ["scale", "deployment/vllm-deployment",
                                            "--replicas=0"] for t, a in calls)
    assert any(t == "gcloud" and "resize" in a and "--num-nodes=0" in a for t, a in calls)


def test_eval_cluster_propagates_the_gbench_exit_code(fakes):
    tmp_path, env = fakes
    env["FAKE_GBENCH_RC"] = "2"
    r = _run(_cluster_args(tmp_path), env)
    assert r.returncode == 2, r.stdout + r.stderr


def test_eval_cluster_refuses_a_pod_that_serves_another_model(fakes):
    tmp_path, env = fakes
    env["FAKE_SERVED_MODEL"] = "someone-elses/model"
    r = _run(_cluster_args(tmp_path), env)
    assert r.returncode == 1
    assert "does not serve org/model-x" in r.stderr and "someone-elses/model" in r.stderr
    assert not (tmp_path / "gbench_args.json").exists()


def test_eval_cluster_fails_when_a_tunnel_cannot_bind(fakes):
    tmp_path, env = fakes
    env["FAKE_PF_FAIL"] = "1"
    t0 = time.time()
    r = _run(_cluster_args(tmp_path), env)
    assert r.returncode == 1 and "did not come up" in r.stderr
    assert time.time() - t0 < 60                 # no silent retry loop
    assert not (tmp_path / "gbench_args.json").exists()


def test_eval_cluster_refuses_a_busy_proxy_port(fakes):
    tmp_path, env = fakes
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        busy = s.getsockname()[1]
        r = _run(_cluster_args(tmp_path, "--proxy-port", str(busy)), env)
    assert r.returncode == 1 and f"--proxy-port {busy} is in use" in r.stderr
    assert not (tmp_path / "gbench_args.json").exists()


@pytest.mark.parametrize("extra,message", [
    (["--image", "vllm/vllm-openai:nightly"], "moving tag"),
    (["--image", "vllm/vllm-openai:latest"], "moving tag"),
    (["--image", "vllm/vllm-openai"], "no tag or digest"),
    (["--vllm-serve-args", "--max-model-len 4096"], "may not set --max-model-len"),
    (["--vllm-serve-args", "--tensor_parallel_size=4"], "may not set --tensor-parallel-size"),
    (["--vllm-serve-args", "--port=9"], "may not set --port"),
    (["--vllm-serve-args", "org/other --x"], "must start with an option"),
    (["--dp", "3"], "must equal --num-gpus"),
    (["--proxy-script", "/nonexistent/proxy.py"], "does not exist"),
    (["--manifest-path", "/nonexistent/m.yaml"], "does not exist"),
])
def test_eval_cluster_validates_before_touching_the_cluster(fakes, extra, message):
    tmp_path, env = fakes
    r = _run(_cluster_args(tmp_path, *extra), env)
    assert r.returncode == 1 and message in r.stderr, r.stderr
    assert not (tmp_path / "calls.log").exists()


def test_eval_cluster_requires_tp_and_has_no_model_defaults(fakes):
    tmp_path, env = fakes
    args = _cluster_args(tmp_path)
    i = args.index("--tp")
    del args[i:i + 2]
    r = _run(args, env)
    assert r.returncode == 1 and "--tp is required" in r.stderr
    text = (SCRIPTS / "eval_cluster.sh").read_text()
    assert 'PROXY_SCRIPT="$SCRIPT_DIR/async_proxy.py"' in text
    assert 'MANIFEST_PATH="$REPO_ROOT/vllm-gke.yaml"' in text
    assert "pkill" not in text and ".venv" not in text


def test_eval_cluster_sets_a_pinned_image_when_given(fakes):
    tmp_path, env = fakes
    r = _run(_cluster_args(tmp_path, "--image", "vllm/vllm-openai:v0.30.0"), env)
    assert r.returncode == 0, r.stderr
    assert ["set", "image", "deployment/vllm-deployment", "vllm=vllm/vllm-openai:v0.30.0"] in [
        a for t, a in _calls(tmp_path) if t == "kubectl"]


def test_eval_cluster_setup_only_leaves_its_proxy_and_tunnels_running(fakes):
    tmp_path, env = fakes
    r = _run(_cluster_args(tmp_path, "--setup-only", "--tunnels-per-pod", "1"), env)
    assert r.returncode == 0, r.stdout + r.stderr
    m = re.search(r"SETUP COMPLETE - proxy running on http://127\.0\.0\.1:(\d+)/v1", r.stdout)
    pids = [int(p) for p in re.search(r"^  kill ([\d ]+)$", r.stdout, re.M).group(1).split()]
    try:
        assert m and _listening(int(m.group(1)))
        assert len(pids) == 1 + 2                 # the proxy and one tunnel per pod
        for pid in pids:
            os.kill(pid, 0)                       # alive
        assert not (tmp_path / "gbench_args.json").exists()
        assert not any(t == "kubectl" and a[:2] == ["scale", "deployment/vllm-deployment"]
                       and a[-1] == "--replicas=0" for t, a in _calls(tmp_path))
        assert "auto_cleanup_after_gbench.sh --pid <pid> --kill-pids" in r.stdout
    finally:
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + 10
    while time.time() < deadline and _listening(int(m.group(1))):
        time.sleep(0.2)
    assert not _listening(int(m.group(1)))


# -- async_proxy.py ----------------------------------------------

async def _backend(name, served="org/m", fail_status=None):
    from aiohttp import web

    async def models(request):
        return web.json_response({"data": [{"id": served}]})

    async def chat(request):
        body = await request.json()
        resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream",
                                                       "X-Backend": name})
        await resp.prepare(request)
        for i in range(3):
            await resp.write(f"data: {name}-{i}-{body['n']}\n\n".encode())
            await asyncio.sleep(0.01)
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_get("/v1/models", models)
    app.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_proxy_round_robins_streams_and_skips_a_dead_backend():
    import aiohttp
    from aiohttp import web

    async def main():
        r1, p1 = await _backend("a")
        r2, p2 = await _backend("b")
        dead = _free_port()                      # nothing listens there
        app = async_proxy.make_app(async_proxy.parse_backends(f"{p1},{dead},{p2}"),
                                   down_seconds=60)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        seen = []
        try:
            async with aiohttp.ClientSession() as s:
                for n in range(6):
                    async with s.post(f"http://127.0.0.1:{port}/v1/chat/completions?x=1",
                                      json={"n": n}) as resp:
                        assert resp.status == 200
                        assert resp.headers["Content-Type"].startswith("text/event-stream")
                        text = await resp.text()
                        seen.append(resp.headers["X-Backend"])
                        assert text.count("data: ") == 3 and f"-{n}\n" in text
        finally:
            await runner.cleanup()
            await r1.cleanup()
            await r2.cleanup()
        return seen

    seen = asyncio.run(main())
    assert set(seen) == {"a", "b"}
    assert abs(seen.count("a") - seen.count("b")) <= 1


def test_proxy_answers_502_when_every_backend_refuses():
    import aiohttp
    from aiohttp import web

    async def main():
        app = async_proxy.make_app([f"http://127.0.0.1:{_free_port()}"])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"http://127.0.0.1:{port}/v1/models") as resp:
                    return resp.status, await resp.json()
        finally:
            await runner.cleanup()

    status, body = asyncio.run(main())
    assert status == 502 and body["error"]["type"] == "proxy_error"


def test_proxy_expect_model_refuses_a_foreign_backend():
    import aiohttp

    async def main():
        r1, p1 = await _backend("a", served="org/m")
        r2, p2 = await _backend("b", served="someone-elses/model")
        try:
            async with aiohttp.ClientSession() as s:
                return await async_proxy.check_models(
                    s, async_proxy.parse_backends(f"{p1},{p2}"), "org/m")
        finally:
            await r1.cleanup()
            await r2.cleanup()

    problems = asyncio.run(main())
    assert len(problems) == 1 and "someone-elses/model" in problems[0]


def test_proxy_refuses_a_busy_port_and_bad_backends(capsys):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        busy = s.getsockname()[1]
        rc = async_proxy.main(["--port", str(busy), "--backends", str(_free_port())])
    assert rc == 1 and "cannot listen" in capsys.readouterr().err
    assert async_proxy.main(["--port", "0", "--backends", "not-a-backend"]) == 2
    with pytest.raises(ValueError):
        async_proxy.parse_backends("")


def test_proxy_binds_localhost_by_default():
    assert async_proxy.build_parser().parse_args(["--port", "0", "--backends", "1"]).host == \
        "127.0.0.1"


# -- auto_cleanup_after_gbench.sh --------------------------------

def test_auto_cleanup_waits_for_its_pid_and_stops_only_the_given_pids(fakes):
    tmp_path, env = fakes
    env.update({"AUTO_CLEANUP_POLL_S": "0.2", "AUTO_CLEANUP_FLUSH_S": "0"})
    # the watched process is not this test's child (a child left unreaped stays a zombie that
    # `kill -0` still finds); gbench's own parent reaps it in a real run
    gbench_pid = int(subprocess.run(["bash", "-c", "sleep 1 >/dev/null 2>&1 & echo $!"],
                                    capture_output=True, text=True).stdout)
    target = subprocess.Popen(["sleep", "60"])
    bystander = subprocess.Popen(["bash", "-c", "exec -a async_proxy.py sleep 60"])
    try:
        r = _run([str(SCRIPTS / "auto_cleanup_after_gbench.sh"), "--pid", str(gbench_pid),
                  "--kill-pids", str(target.pid), "--cluster", "c", "--node-pool", "p",
                  "--region", "r", "--project", "x", "--log", str(tmp_path / "cleanup.log")],
                 env, timeout=30)
        assert r.returncode == 0, (tmp_path / "cleanup.log").read_text()
        assert target.wait(timeout=5) == -signal.SIGTERM
        assert bystander.poll() is None          # a same-named process is left alone
        calls = _calls(tmp_path)
        assert any(t == "gcloud" and "resize" in a for t, a in calls)
    finally:
        for p in (target, bystander):
            if p.poll() is None:
                p.kill()
                p.wait()
    assert "pkill" not in (SCRIPTS / "auto_cleanup_after_gbench.sh").read_text()


def test_auto_cleanup_rejects_a_non_numeric_pid(fakes):
    tmp_path, env = fakes
    r = _run([str(SCRIPTS / "auto_cleanup_after_gbench.sh"), "--pid", "1", "--kill-pids",
              "gbench", "--cluster", "c", "--node-pool", "p", "--region", "r", "--project",
              "x", "--log", str(tmp_path / "l.log")], env)
    assert r.returncode == 1 and "not a process id" in r.stderr


# -- deploy_a3_vms.sh --------------------------------------------

def test_deploy_a3_vms_passes_the_model_recipe_through_quoted(fakes):
    tmp_path, env = fakes
    r = _run([str(SCRIPTS / "deploy_a3_vms.sh"), "--project", "p", "--model", "org/m",
              "--vm-prefix", "vm", "--tp", "2", "--reasoning-parser", "rp",
              "--vllm-serve-args", "--trust-remote-code --limit-mm-per-prompt '{\"image\": 8}'"],
             env)
    assert r.returncode == 0, r.stdout + r.stderr
    runs = [json.loads(line) for line in (tmp_path / "docker.log").read_text().splitlines()
            if json.loads(line)[:1] == ["run"]]
    assert len(runs) == 4
    argv = runs[0]
    image_at = next(i for i, a in enumerate(argv) if a.startswith("vllm/vllm-openai:"))
    assert "@sha256:" in argv[image_at]
    serve = argv[image_at + 1:]
    assert serve[:5] == ["org/m", "--tensor-parallel-size", "2", "--data-parallel-size", "4"]
    assert "--max-model-len" not in serve        # the model's native context
    assert serve[serve.index("--reasoning-parser") + 1] == "rp"
    assert "--tool-call-parser" not in serve
    assert serve[-3:] == ["--trust-remote-code", "--limit-mm-per-prompt", '{"image": 8}']


@pytest.mark.parametrize("extra,message", [
    ([], "--tp is required"),
    (["--tp", "3"], "does not divide"),
    (["--tp", "2", "--image", "vllm/vllm-openai:nightly"], "moving tag"),
    (["--tp", "2", "--vllm-serve-args", "--port 1"], "may not set --port"),
])
def test_deploy_a3_vms_validates_before_creating_vms(fakes, extra, message):
    tmp_path, env = fakes
    r = _run([str(SCRIPTS / "deploy_a3_vms.sh"), "--project", "p", "--model", "org/m",
              "--vm-prefix", "vm", *extra], env)
    assert r.returncode == 1 and message in r.stderr, r.stderr
    assert not (tmp_path / "calls.log").exists()


# -- Manifest and scripts carry no model identity ----------------

def _code_lines(path: Path):
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            yield line


@pytest.mark.parametrize("rel", ["vllm-gke.yaml", "scripts/eval_cluster.sh",
                                 "scripts/deploy_a3_vms.sh"])
def test_deploy_artifacts_name_no_model_recipe(rel):
    """No parser name, context length or soft-token count of any model outside comments, and a
    pinned image (never a moving tag)."""
    code = "\n".join(_code_lines(REPO / rel))
    for literal in ("gemma", "262144", "1120", "max_soft_tokens", "default_output_length",
                    "--trust-remote-code\"", "limit-mm-per-prompt\"", ":nightly\n",
                    "vllm-openai:latest"):
        assert literal not in code, (rel, literal)
    images = re.findall(r"vllm/vllm-openai:[^\s\"']+", code)
    assert all("@sha256:" in i for i in images), images
    if rel != "scripts/eval_cluster.sh":      # eval_cluster.sh uses the manifest's image
        assert images, rel


def test_manifest_starts_at_zero_replicas_and_reads_its_settings_from_env():
    docs = list(yaml.safe_load_all((REPO / "vllm-gke.yaml").read_text()))
    dep = docs[1]
    assert dep["spec"]["replicas"] == 0
    container = dep["spec"]["template"]["spec"]["containers"][0]
    names = {e["name"] for e in container["env"]}
    assert {"MODEL_NAME", "TENSOR_PARALLEL_SIZE", "DATA_PARALLEL_SIZE", "MAX_MODEL_LEN",
            "TOOL_CALL_PARSER", "REASONING_PARSER", "VLLM_SERVE_ARGS"} <= names


def _launch(env_over, tmp_path):
    container = list(yaml.safe_load_all((REPO / "vllm-gke.yaml").read_text()))[1][
        "spec"]["template"]["spec"]["containers"][0]
    fake = tmp_path / "vllm"
    fake.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    fake.chmod(0o755)
    env = {"PATH": f"{tmp_path}:{os.environ['PATH']}", "MODEL_NAME": "org/m",
           "TENSOR_PARALLEL_SIZE": "2", "DATA_PARALLEL_SIZE": "2"}
    env.update(env_over)
    return subprocess.run([sys.executable, "-c", container["args"][0]], env=env,
                          capture_output=True, text=True, timeout=30)


def test_manifest_launcher_builds_the_serve_command(tmp_path):
    r = _launch({"MAX_MODEL_LEN": "", "TOOL_CALL_PARSER": "t", "REASONING_PARSER": "",
                 "VLLM_SERVE_ARGS": "--foo '{\"a\": 1}'"}, tmp_path)
    assert r.returncode == 0, r.stderr
    argv = json.loads(r.stdout.strip().splitlines()[-1])
    assert argv[:2] == ["serve", "org/m"]
    assert "--max-model-len" not in argv
    assert argv[argv.index("--tool-call-parser") + 1] == "t"
    assert argv[-2:] == ["--foo", '{"a": 1}']
    r = _launch({"MAX_MODEL_LEN": "8192"}, tmp_path)
    argv = json.loads(r.stdout.strip().splitlines()[-1])
    assert argv[argv.index("--max-model-len") + 1] == "8192"


@pytest.mark.parametrize("env_over,message", [
    ({"MODEL_NAME": ""}, "MODEL_NAME is not set"),
    ({"VLLM_SERVE_ARGS": "--max_model_len=4"}, "may not set --max-model-len"),
    ({"VLLM_SERVE_ARGS": "--served-model-name x"}, "may not set --served-model-name"),
])
def test_manifest_launcher_refuses_what_it_owns(tmp_path, env_over, message):
    r = _launch(env_over, tmp_path)
    assert r.returncode != 0 and message in r.stderr


def test_the_removed_leftovers_are_gone():
    """The Gemma-only lineup script and the model-table checker (which parsed a script that no
    longer exists) are not shipped."""
    assert not (SCRIPTS / "run_benchmarks.sh").exists()
    assert not (REPO / "utils" / "verify_model_defs.py").exists()


# -- CI helpers --------------------------------------------------

def _gate_run(tmp_path, test_body, declared):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0"\ndependencies = []\n'
        f'[project.optional-dependencies]\ndev = {json.dumps(declared)}\n')
    (tmp_path / "test_gate_probe.py").write_text(textwrap.dedent(test_body))
    env = dict(os.environ, PYTHONPATH=str(SCRIPTS))
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           "-p", "ci_skip_gate", "--skip-gate-pyproject",
                           str(tmp_path / "pyproject.toml"), "--rootdir", str(tmp_path),
                           str(tmp_path / "test_gate_probe.py")],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)


def test_ci_skip_gate_fails_a_skip_caused_by_a_declared_dependency(tmp_path):
    r = _gate_run(tmp_path, """
        import pytest
        def test_x():
            pytest.importorskip("fakepkg_xyz_declared.sub")
    """, ["fakepkg-xyz-declared>=1"])
    assert r.returncode == 1, r.stdout
    assert "fakepkg_xyz_declared.sub" in r.stdout and "fakepkg-xyz-declared" in r.stdout


def test_ci_skip_gate_fails_a_module_level_skip_too(tmp_path):
    r = _gate_run(tmp_path, """
        import pytest
        pytest.importorskip("fakepkg_xyz_declared", reason="harness not installed")
        def test_x():
            pass
    """, ["fakepkg_xyz_declared"])
    assert r.returncode == 1, r.stdout


def test_ci_skip_gate_only_notes_an_undeclared_module(tmp_path):
    r = _gate_run(tmp_path, """
        import pytest
        def test_x():
            pytest.importorskip("fakepkg_nobody_declares")
        def test_y():
            assert True
    """, ["something-else"])
    assert r.returncode == 0, r.stdout
    assert "fakepkg_nobody_declares" in r.stdout


def test_ci_skip_gate_fails_a_not_installed_skip_of_a_declared_package(tmp_path):
    r = _gate_run(tmp_path, """
        import pytest
        def test_x():
            pytest.skip("Python package 'fakepkg_xyz_declared' is not installed")
    """, ["fakepkg-xyz-declared"])
    assert r.returncode == 1, r.stdout
    assert "fakepkg-xyz-declared" in r.stdout


def test_ci_skip_gate_only_notes_a_not_installed_tool_nothing_declares(tmp_path):
    r = _gate_run(tmp_path, """
        import pytest
        def test_x():
            pytest.skip("bubblewrap is not installed")
        def test_y():
            assert True
    """, ["something-else"])
    assert r.returncode == 0, r.stdout
    assert "bubblewrap" in r.stdout


def test_ci_skip_gate_maps_import_names_to_distributions():
    import ci_skip_gate as G
    declared = G.declared_modules(REPO / "pyproject.toml", ["dev", "evals"])
    assert G.owning_distribution("bfcl_eval.constants.category_mapping", declared) == "bfcl-eval"
    assert G.owning_distribution("google.genai", declared) == "google-genai"
    assert G.owning_distribution("minisweagent", declared) == "mini-swe-agent"
    assert G.owning_distribution("sklearn.metrics", declared) == "scikit-learn"
    assert G.owning_distribution("torch", declared) == ""


def _workflow(name):
    return yaml.safe_load((REPO / ".github" / "workflows" / name).read_text())


def test_the_unit_test_workflow_installs_the_declared_dependencies():
    """The package with its dependencies and the dev, evals and omnidocbench extras, never
    --no-deps or a hand-written list; the skip gate runs with the tests."""
    steps = _workflow("test.yml")["jobs"]["pytest"]["steps"]
    runs = "\n".join(s.get("run", "") for s in steps)
    assert '-e ".[dev,evals,omnidocbench]"' in runs
    assert "--no-deps" not in runs
    assert "-p ci_skip_gate" in runs
    assert "nltk.downloader" in runs
    run_step = next(s for s in steps if "-p ci_skip_gate" in s.get("run", ""))
    assert run_step["env"]["PYTHONPATH"] == "scripts"
    assert (SCRIPTS / "ci_skip_gate.py").is_file()


def test_no_test_skips_a_missing_package_with_a_message_of_its_own():
    """ci_skip_gate recognizes pytest.importorskip and "No module named" skips; a missing package
    skipped with pytest.skip("... not installed") would pass CI unseen, so no test does that."""
    import ast
    bad = []
    for path in sorted((REPO / "tests").glob("test_*.py")):
        if path.name == "test_swe_bench_pro_agentic.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "skip" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and "not installed" in node.args[0].value.lower()):
                bad.append(f"{path.name}:{node.lineno}: {node.args[0].value}")
    assert not bad, "use pytest.importorskip(<module>) instead:\n" + "\n".join(bad)


def test_the_canonical_sync_gate_calls_the_modules_own_check():
    runs = "\n".join(s.get("run", "") for s in
                     _workflow("canonical-sync-gate.yml")["jobs"]["canonical-sync-gate"]["steps"])
    assert runs.count("gbench/runners/eval_suites/canonical_sync.py --check-bump") == 2
    r = subprocess.run([sys.executable, str(REPO / "gbench/runners/eval_suites/canonical_sync.py"),
                        "--help"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "--check-bump" in r.stdout and "--base" in r.stdout


def test_the_leak_gate_refuses_tracked_backup_files():
    step = next(s for s in _workflow("leak-gate.yml")["jobs"]["leak-gate"]["steps"]
                if "No backup" in s.get("name", ""))
    pattern = re.search(r"grep -E '([^']+)'", step["run"]).group(1)
    for name in ("a.py.bak", "a.py.bak_20260927_x", "b.bak-1", "c.orig"):
        assert re.search(pattern, name), name
    for name in ("backup.py", "bake.py", "origin.md", "scripts/check_no_leaks.sh"):
        assert not re.search(pattern, name), name


# --------------------------------------------------------------------------- #
# The CI install on both Python legs, and the reasoning parser of the cluster scripts
# --------------------------------------------------------------------------- #
# -- terminal_bench's unit tests need neither Harbor nor its registry ---------------------------

#: Every test that imported Harbor or queried Harbor's task registry.
HARBOR_FREE = [
    "tests/test_turn_budget.py::test_thinking_run_resolves_temperature_to_one",
    "tests/test_turn_budget.py::test_extra_env_is_never_passed_as_an_agent_kwarg",
    "tests/test_turn_budget.py::test_thread_env_goes_to_both_agent_env_and_verifier_env",
    "tests/test_turn_budget.py::test_no_agent_kwarg_collides_with_a_reserved_factory_parameter",
    "tests/test_guard_swe_dns_and_windows_probe.py::test_harbor_task_images_reads_both_cache_layouts_and_the_runs_tasks",
    "tests/test_guard_swe_dns_and_windows_probe.py::test_terminal_bench_probes_with_the_runs_tasks_first_and_checks_the_hosts",
    "tests/test_guard_swe_dns_and_windows_probe.py::test_terminal_bench_without_a_local_task_image_names_the_pull",
    "tests/test_guard_swe_agents.py::test_terminal_bench_reads_the_task_images_from_harbors_cache",
    "tests/test_guard_swe_agents.py::test_terminal_bench_without_cached_tasks_asks_for_the_dataset",
    "tests/test_swe_harness_errors_and_terminal_bench_exceptions.py::test_the_docstrings_attribute_only_harbors_own_exclusions_to_harbor",
]

_BLOCK_HARBOR = '''
import importlib.abc, sys

class _NoHarbor(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "harbor" or name.startswith("harbor."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None

sys.meta_path.insert(0, _NoHarbor())
for _m in [m for m in sys.modules if m == "harbor" or m.startswith("harbor.")]:
    del sys.modules[_m]
'''


def test_the_terminal_bench_tests_pass_where_harbor_cannot_be_imported(tmp_path):
    (tmp_path / "no_harbor.py").write_text(_BLOCK_HARBOR)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [str(tmp_path), str(SCRIPTS), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "-p", "no_harbor", *HARBOR_FREE],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
    passed = re.search(r"(\d+) passed", r.stdout)
    assert passed and int(passed.group(1)) >= len(HARBOR_FREE), r.stdout[-2000:]
    assert "skipped" not in r.stdout and "deselected" not in r.stdout, r.stdout[-2000:]


# -- ci_skip_gate and environment markers --------------------------------------------------------

def test_a_requirement_whose_marker_is_false_here_is_not_declared_here(tmp_path):
    here = f"{sys.version_info.major}.{sys.version_info.minor}"
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "x"\nversion = "0"\ndependencies = []\n'
        '[project.optional-dependencies]\n'
        f'tb = ["pkg-elsewhere==1; python_version > \'{here}\'", '
        f'"pkg-here==1; python_version == \'{here}\'", "pkg-plain"]\n')
    declared = G.declared_modules(pyproject, ["tb"])
    assert "pkg_elsewhere" not in declared
    assert declared["pkg_here"] == "pkg-here" and declared["pkg_plain"] == "pkg-plain"


def test_an_unparsable_marker_still_counts_as_declared():
    assert G.applies_here("x==1; not a marker (((") is True
    assert G.applies_here("x==1") is True
    assert G.applies_here("x==1; python_version < '3'") is False
    assert G.applies_here("x; extra == 'dev'", "dev") is True
    assert G.applies_here("x; extra == 'dev'", "evals") is False


def _marker_gate_run(tmp_path, requirement, extras="dev"):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\nversion = "0"\ndependencies = []\n'
        f'[project.optional-dependencies]\ndev = {json.dumps([requirement])}\n')
    (tmp_path / "test_gate_probe.py").write_text(textwrap.dedent("""
        import pytest
        def test_x():
            pytest.importorskip("fakepkg_marker_probe")
    """))
    env = dict(os.environ, PYTHONPATH=str(SCRIPTS))
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                           "-p", "ci_skip_gate", "--skip-gate-pyproject",
                           str(tmp_path / "pyproject.toml"), "--skip-gate-extras", extras,
                           "--rootdir", str(tmp_path), str(tmp_path / "test_gate_probe.py")],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)


def test_the_gate_notes_a_skip_from_a_requirement_this_interpreter_does_not_install(tmp_path):
    r = _marker_gate_run(tmp_path, "fakepkg-marker-probe==1; python_version < '3'")
    assert r.returncode == 0, r.stdout
    assert "nothing declares for this interpreter: fakepkg_marker_probe" in r.stdout


def test_the_gate_fails_a_skip_from_a_requirement_this_interpreter_installs(tmp_path):
    r = _marker_gate_run(tmp_path, "fakepkg-marker-probe==1; python_version >= '3'")
    assert r.returncode == 1, r.stdout
    assert 'pip install -e ".[dev]"' in r.stdout


# -- the reasoning parser of the cluster scripts -------------------------------------------------

def _without_parser(args):
    i = args.index("--reasoning-parser")
    return args[:i] + args[i + 2:]


def test_eval_cluster_refuses_a_thinking_run_without_a_reasoning_parser(fakes):
    tmp_path, env = fakes
    r = _run(_without_parser(_cluster_args(tmp_path)), env)
    assert r.returncode == 1
    assert "needs --reasoning-parser NAME" in r.stderr and "--no-thinking" in r.stderr
    assert not (tmp_path / "calls.log").exists()        # nothing touched the cluster


def test_eval_cluster_runs_without_a_parser_when_thinking_is_off(fakes):
    tmp_path, env = fakes
    r = _run(_without_parser(_cluster_args(tmp_path, "--no-thinking")), env)
    assert r.returncode == 0, r.stdout + r.stderr
    gargs = json.loads((tmp_path / "gbench_args.json").read_text())
    assert "--thinking" not in gargs
    set_env = next(a for t, a in _calls(tmp_path) if t == "kubectl" and a[:2] == ["set", "env"])
    assert "REASONING_PARSER=" in set_env


def test_eval_cluster_passes_the_parser_and_records_it(fakes):
    tmp_path, env = fakes
    r = _run(_cluster_args(tmp_path), env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--thinking" in json.loads((tmp_path / "gbench_args.json").read_text())
    set_env = next(a for t, a in _calls(tmp_path) if t == "kubectl" and a[:2] == ["set", "env"])
    assert "REASONING_PARSER=rp1" in set_env
    serving = json.loads((tmp_path / "results" / "cluster" / "serving.json").read_text())
    assert serving["reasoning_parser"] == "rp1"


def test_eval_cluster_accepts_none_for_a_model_without_a_parser(fakes):
    tmp_path, env = fakes
    args = _without_parser(_cluster_args(tmp_path)) + ["--reasoning-parser", "none"]
    r = _run(args, env)
    assert r.returncode == 0, r.stdout + r.stderr
    set_env = next(a for t, a in _calls(tmp_path) if t == "kubectl" and a[:2] == ["set", "env"])
    assert "REASONING_PARSER=none" in set_env
    serving = json.loads((tmp_path / "results" / "cluster" / "serving.json").read_text())
    assert serving["reasoning_parser"] == "none"


@pytest.mark.parametrize("value", ["none", "None", " NONE "])
def test_the_manifest_launcher_passes_no_parser_for_none(tmp_path, value):
    r = _launch({"REASONING_PARSER": value}, tmp_path)
    assert r.returncode == 0, r.stderr
    argv = json.loads(r.stdout.strip().splitlines()[-1])
    assert "--reasoning-parser" not in argv


def test_the_manifest_launcher_passes_a_named_parser(tmp_path):
    r = _launch({"REASONING_PARSER": "rp"}, tmp_path)
    argv = json.loads(r.stdout.strip().splitlines()[-1])
    assert argv[argv.index("--reasoning-parser") + 1] == "rp"


def _deploy(fakes_, *extra):
    tmp_path, env = fakes_
    r = _run([str(SCRIPTS / "deploy_a3_vms.sh"), "--project", "p", "--model", "org/m",
                "--vm-prefix", "vm", "--tp", "8", *extra], env)
    runs = []
    if (tmp_path / "docker.log").exists():
        runs = [json.loads(line) for line in (tmp_path / "docker.log").read_text().splitlines()
                if json.loads(line)[:1] == ["run"]]
    return r, runs


def test_deploy_a3_vms_passes_no_parser_for_none(fakes):
    r, runs = _deploy(fakes, "--reasoning-parser", "none")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(runs) == 4 and all("--reasoning-parser" not in argv for argv in runs)
    assert "WARNING" not in r.stderr


def test_deploy_a3_vms_warns_when_no_parser_is_given(fakes):
    r, runs = _deploy(fakes)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "WARNING: no --reasoning-parser" in r.stderr
    assert len(runs) == 4 and all("--reasoning-parser" not in argv for argv in runs)


def test_the_scripts_document_the_parser_rule():
    cluster = (SCRIPTS / "eval_cluster.sh").read_text()
    assert "required for a thinking run" in cluster and "`none` states that the model" in cluster
    manifest = (REPO / "vllm-gke.yaml").read_text()
    assert "`none`" in manifest




# -- Argument checks of the cluster scripts ----------------------

_DIGEST = "@sha256:" + "0123456789abcdef" * 4


@pytest.mark.parametrize("serve_args,owned", [
    ("--max-model 4096", "--max-model-len"),         # vLLM's parser reads it as --max-model-len
    ("--served-model x", "--served-model-name"),
    ("--ho 0.0.0.0", "--host"),                      # two letters are enough for vLLM
    ("--tensor_parallel 2", "--tensor-parallel-size"),
    ("--tool-call=t", "--tool-call-parser"),
])
def test_eval_cluster_refuses_an_abbreviated_owned_argument(fakes, serve_args, owned):
    tmp_path, env = fakes
    r = _run(_cluster_args(tmp_path, "--vllm-serve-args", serve_args), env)
    assert r.returncode == 1, r.stdout + r.stderr
    assert f"(an abbreviation of {owned})" in r.stderr, r.stderr
    assert not (tmp_path / "calls.log").exists()


def test_eval_cluster_passes_options_that_only_share_a_prefix(fakes):
    """--model-impl and --max-num-seqs start like owned options but are not prefixes of one."""
    tmp_path, env = fakes
    r = _run(_cluster_args(tmp_path, "--vllm-serve-args",
                           "--model-impl transformers --max-num-seqs 8"), env)
    assert r.returncode == 0, r.stdout + r.stderr
    set_env = next(a for t, a in _calls(tmp_path) if t == "kubectl" and a[:2] == ["set", "env"])
    assert "VLLM_SERVE_ARGS=--model-impl transformers --max-num-seqs 8" in set_env


def test_eval_cluster_refuses_an_extra_positional_argument(fakes):
    tmp_path, env = fakes
    r = _run(_cluster_args(tmp_path, "results/old_run"), env)
    assert r.returncode == 1
    assert "unexpected argument(s): results/old_run" in r.stderr, r.stderr
    assert not (tmp_path / "calls.log").exists()


@pytest.mark.parametrize("image,message", [
    ("registry:5000/vllm", "has no tag or digest"),      # a registry port is not a tag
    ("vllm/vllm-openai:dev", "is not a release version"),
    ("vllm/vllm-openai:latest-cu12", "moving tag"),
    ("vllm/vllm-openai:v0.30.0-nightly", "moving tag"),
    ("vllm/vllm-openai:", "has no tag or digest"),
    ("vllm/vllm-openai@sha256:abc", "malformed digest"),
])
def test_eval_cluster_refuses_an_unpinned_image(fakes, image, message):
    tmp_path, env = fakes
    r = _run(_cluster_args(tmp_path, "--image", image), env)
    assert r.returncode == 1 and message in r.stderr, r.stderr
    assert not (tmp_path / "calls.log").exists()


@pytest.mark.parametrize("image", ["registry:5000/vllm/vllm-openai:v0.30.0",
                                   "vllm/vllm-openai:v0.10.1.1",
                                   "vllm/vllm-openai:v0.6.3.post1",
                                   "vllm/vllm-openai" + _DIGEST])
def test_eval_cluster_accepts_a_release_tag_or_a_digest(fakes, image):
    tmp_path, env = fakes
    r = _run(_cluster_args(tmp_path, "--image", image), env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert ["set", "image", "deployment/vllm-deployment", f"vllm={image}"] in [
        a for t, a in _calls(tmp_path) if t == "kubectl"]


def test_eval_cluster_reads_parser_names_without_surrounding_whitespace(fakes):
    tmp_path, env = fakes
    r = _run(_without_parser(_cluster_args(tmp_path)) + ["--reasoning-parser", "   "], env)
    assert r.returncode == 1 and "needs --reasoning-parser NAME" in r.stderr, r.stderr
    assert not (tmp_path / "calls.log").exists()
    r = _run(_without_parser(_cluster_args(tmp_path)) + ["--reasoning-parser", " rp1 ",
                                                          "--tool-call-parser", "\tt1 "], env)
    assert r.returncode == 0, r.stdout + r.stderr
    set_env = next(a for t, a in _calls(tmp_path) if t == "kubectl" and a[:2] == ["set", "env"])
    assert "REASONING_PARSER=rp1" in set_env and "TOOL_CALL_PARSER=t1" in set_env


@pytest.mark.parametrize("extra,message", [
    (["--vllm-serve-args", "--served-model x"], "(an abbreviation of --served-model-name)"),
    (["--vllm-serve-args", "--max-model 4096"], "(an abbreviation of --max-model-len)"),
    (["--image", "registry:5000/img"], "has no tag or digest"),
    (["--image", "img:dev"], "is not a release version"),
    (["--image", "img:latest-cu12"], "moving tag"),
])
def test_deploy_a3_vms_refuses_abbreviations_and_unpinned_images(fakes, extra, message):
    tmp_path, _ = fakes
    r, runs = _deploy(fakes, *extra)
    assert r.returncode == 1 and message in r.stderr, r.stderr
    assert not (tmp_path / "calls.log").exists() and runs == []


@pytest.mark.parametrize("image", ["img:v0.30.0", "registry:5000/img:v0.30.0"])
def test_deploy_a3_vms_runs_a_release_tag(fakes, image):
    r, runs = _deploy(fakes, "--image", image, "--reasoning-parser", "rp")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(runs) == 4 and all(image in argv for argv in runs)


def test_deploy_a3_vms_reads_parser_names_without_surrounding_whitespace(fakes):
    r, runs = _deploy(fakes, "--reasoning-parser", " none", "--tool-call-parser", " t ")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(runs) == 4 and all("--reasoning-parser" not in argv for argv in runs)
    assert all(argv[argv.index("--tool-call-parser") + 1] == "t" for argv in runs)
    assert "WARNING" not in r.stderr
    tmp_path, _ = fakes
    (tmp_path / "docker.log").unlink()
    r, runs = _deploy(fakes, "--reasoning-parser", "  ")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "WARNING: no --reasoning-parser" in r.stderr
    assert len(runs) == 4 and all("--reasoning-parser" not in argv for argv in runs)


# -- async_proxy.py connection cap and open-file limit ------------

def test_proxy_does_not_cap_the_connections_to_the_backends():
    from aiohttp import web

    async def main(**kw):
        app = async_proxy.make_app(["http://127.0.0.1:1"], **kw)
        runner = web.AppRunner(app)
        await runner.setup()                      # runs the startup hook that opens the session
        try:
            c = app[async_proxy._SESSION].connector
            return c.limit, c.limit_per_host
        finally:
            await runner.cleanup()

    assert asyncio.run(main()) == (0, 0)
    assert asyncio.run(main(max_connections=5)) == (5, 0)
    assert async_proxy.build_parser().parse_args(
        ["--port", "0", "--backends", "1"]).max_connections == 0
    assert async_proxy.main(["--port", "0", "--backends", "1", "--max-connections", "-1"]) == 2


def test_proxy_forwards_more_than_a_hundred_requests_at_once():
    """aiohttp's default connector would hold every request past the 100th until one finished;
    the backend here answers only once all of them are in flight."""
    import aiohttp
    from aiohttp import web

    n = 130

    async def main():
        state = {"now": 0, "peak": 0}
        everyone = asyncio.Event()

        async def slow(request):
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            if state["now"] >= n:
                everyone.set()
            try:
                await asyncio.wait_for(everyone.wait(), timeout=20)
                return web.json_response({"ok": True})
            except asyncio.TimeoutError:
                return web.json_response({"ok": False}, status=504)
            finally:
                state["now"] -= 1

        backend = web.Application()
        backend.router.add_post("/v1/completions", slow)
        b_runner = web.AppRunner(backend)
        await b_runner.setup()
        b_site = web.TCPSite(b_runner, "127.0.0.1", 0)
        await b_site.start()
        b_port = b_site._server.sockets[0].getsockname()[1]
        p_runner = web.AppRunner(async_proxy.make_app([f"http://127.0.0.1:{b_port}"]))
        await p_runner.setup()
        p_site = web.TCPSite(p_runner, "127.0.0.1", 0)
        await p_site.start()
        p_port = p_site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession(
                    connector=aiohttp.TCPConnector(limit=0)) as s:
                async def one():
                    async with s.post(f"http://127.0.0.1:{p_port}/v1/completions",
                                      json={}) as resp:
                        return resp.status
                statuses = await asyncio.gather(*[one() for _ in range(n)])
        finally:
            await p_runner.cleanup()
            await b_runner.cleanup()
        return statuses, state["peak"]

    statuses, peak = asyncio.run(main())
    assert statuses == [200] * n and peak == n


def test_proxy_raises_its_open_file_soft_limit(monkeypatch):
    calls = []
    monkeypatch.setattr(async_proxy.resource, "getrlimit", lambda which: (1024, 65536))
    monkeypatch.setattr(async_proxy.resource, "setrlimit",
                        lambda which, limits: calls.append((which, limits)))
    assert async_proxy.raise_open_file_limit() == 65536
    assert calls == [(async_proxy.resource.RLIMIT_NOFILE, (65536, 65536))]

    def refuse(which, limits):
        raise ValueError("not allowed")
    monkeypatch.setattr(async_proxy.resource, "setrlimit", refuse)
    assert async_proxy.raise_open_file_limit() == 1024


# -- The Unit Tests workflow's collection step -------------------

def test_the_collect_step_prints_why_collection_failed():
    """pytest reports collection errors on stdout: the step must not discard it."""
    steps = _workflow("test.yml")["jobs"]["pytest"]["steps"]
    collect = next(s["run"] for s in steps if "--collect-only" in s.get("run", ""))
    assert "/dev/null" not in collect
    assert 'cat "$log"' in collect and "exit 1" in collect
