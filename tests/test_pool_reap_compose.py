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

"""pool._reap removes the compose containers a harness started through the docker socket.

2026-10-11: a SIGTERM to the smoke pool left terminal_bench's Harbor task container
(mcmc-sampling-stan__5flw3gp, a compose project) and its network running: they were neither in
the suite's process group nor named gbench-<suite>-*. The pool now also removes containers
created after the suite started that bind-mount one of the suite's own temp directories."""
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import pytest

from gbench import pool


def test_scratch_mounts_keep_only_the_suites_own_mkdtemp_dirs(tmp_path):
    own = tempfile.mkdtemp(prefix="gbench_tb_")
    try:
        argv = ["docker", "run", "--rm", "-v", "/var/run/docker.sock:/var/run/docker.sock",
                "-v", f"{own}:{own}:rw", "-v", f"{Path.home()}/.cache:/c:rw",
                "--mount", f"type=bind,source={own},target=/m", "-v", "/tmp/.X11-unix:/x",
                "--volume=named:/n", "img"]
        assert pool.scratch_mounts([argv]) == {os.path.realpath(own)}
    finally:
        os.rmdir(own)


def test_reap_removes_compose_containers_that_mount_the_suites_scratch_dir(monkeypatch):
    calls = []
    inspect_out = "\n".join([
        json.dumps("/mcmc-sampling-stan__5flw3gp-main-1") + "\t" + json.dumps(
            [{"Type": "bind", "Source": "/tmp/gbench_tb_x/job/trial/verifier"}]) + "\t" + json.dumps(
            {"com.docker.compose.project": "mcmc-sampling-stan__5flw3gp"}),
        json.dumps("/other-run-main-1") + "\t" + json.dumps(
            [{"Type": "bind", "Source": "/tmp/gbench_tb_y/job/trial/verifier"}]) + "\tnull"])

    def fake(args, timeout=60.0):
        calls.append(list(args))
        if args[0] == "inspect":
            return 0, inspect_out
        if args[:2] == ["network", "ls"]:
            return 0, "abc123\n"
        return 0, ""
    monkeypatch.setattr(pool, "container_names", lambda: {
        "mcmc-sampling-stan__5flw3gp-main-1", "other-run-main-1", "before"})
    monkeypatch.setattr(pool, "_docker", fake)
    c = pool.Child("terminal_bench", None, Path("/x"), Path("/x/log"), 0.0, "", [],
                   containers_before={"before"}, scratch_dirs={"/tmp/gbench_tb_x"})
    pool._reap(c, ["terminal_bench"], baseline=set())
    assert ["rm", "-f", "mcmc-sampling-stan__5flw3gp-main-1"] in calls
    assert ["network", "rm", "abc123"] in calls
    assert c.extra["networks_reaped"] == ["mcmc-sampling-stan__5flw3gp"]
    assert not any("other-run-main-1" in a for a in calls if a[0] == "rm")


def _docker_ok():
    return subprocess.run(["docker", "image", "inspect", "alpine/socat:latest"],
                          capture_output=True).returncode == 0


@pytest.mark.skipif(not _docker_ok(), reason="needs docker and the local alpine/socat image")
def test_live_compose_style_container_left_by_a_stopped_harness_is_reaped(tmp_path):
    scratch = tempfile.mkdtemp(prefix="gbench_tb_")
    trial = Path(scratch) / "job" / "trial" / "verifier"
    trial.mkdir(parents=True)
    project = f"gbenchprobe-{uuid.uuid4().hex[:8]}"
    # the harness: a process whose command line reads `docker run -v <scratch>:<scratch> ...`
    (tmp_path / "run").write_text("import time\ntime.sleep(60)\n")
    harness = subprocess.Popen(["docker", "run", "--name", f"gbench-terminal-bench-{project}",
                                "-v", f"{scratch}:{scratch}:rw", "img"],
                               executable=sys.executable, cwd=tmp_path, start_new_session=True)
    net = f"{project}_default"
    subprocess.run(["docker", "network", "create", "--label",
                    f"com.docker.compose.project={project}", net], check=True,
                   capture_output=True)
    cname = f"{project}-main-1"
    try:
        c = pool.Child("terminal_bench", harness, tmp_path, tmp_path / "log", 0.0, "", [],
                       containers_before=pool.container_names())
        opts = pool.PoolOptions(parallel=1, stall_s=0)
        for _ in range(100):
            pool._scan(c, 0.0, opts)
            if c.scratch_dirs:
                break
            time.sleep(0.05)
        assert c.scratch_dirs == {os.path.realpath(scratch)}
        # what Harbor's `docker compose up` creates through the socket
        subprocess.run(["docker", "run", "-d", "--name", cname, "--network", net,
                        "--label", f"com.docker.compose.project={project}",
                        "-v", f"{trial}:/logs/verifier", "--entrypoint", "sleep",
                        "alpine/socat:latest", "600"], check=True, capture_output=True)
        harness.kill()
        harness.wait()                                   # the harness was stopped
        pool._reap(c, ["terminal_bench"], baseline=set())
        assert cname not in pool.container_names()
        assert c.extra["containers_reaped"] == [cname]
        assert c.extra["networks_reaped"] == [project]
        nets = subprocess.run(["docker", "network", "ls", "-q", "--filter", f"name={net}"],
                              capture_output=True, text=True).stdout.split()
        assert nets == []
    finally:
        if harness.poll() is None:
            harness.kill()
        subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
        subprocess.run(["docker", "network", "rm", net], capture_output=True)
        subprocess.run(["rm", "-rf", scratch])
