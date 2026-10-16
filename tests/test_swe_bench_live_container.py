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

"""swe_bench_live isolates the SWE-bench-Live `swebench` fork in a LOCAL image (no host venv).

The fork is the same package name as upstream `swebench` at an incompatible version, so scoring runs
inside the fork image while the host env keeps upstream for swe_bench_multilingual / copilot_bench_swe.
"""

import asyncio
import sys

import pytest

from gbench.runners.eval_suites import swebench_common as C
from gbench.runners.eval_suites import swe_bench_live as S


@pytest.fixture(autouse=True)
def _no_metadata_probe(monkeypatch):
    """The cloud metadata server gate (swebench_common.require_task_network_blocked) starts probe
    containers; tests/test_guard_swe_agents.py covers it, here it is stubbed."""
    monkeypatch.setattr(C, "require_task_network_blocked", lambda *a, **k: None)


class _Proc:
    returncode = 0
    stdout = ""
    stderr = ""


def _run_scorer(harness_image, monkeypatch):
    captured = {}

    def _fake_run(cmd, **kw):
        captured["cmd"] = list(cmd)
        return _Proc()

    monkeypatch.setattr(C.subprocess, "run", _fake_run)
    metrics = {}
    scorer = C.make_swebench_scorer(
        "swe_bench_live", "gemma", "SWE-bench-Live/SWE-bench-Live", "lite",
        "starryzhang", 1, metrics, harness_image=harness_image)
    traces = [{"extra_payload": {"instance_id": "repo__1"}, "response_text": "diff --git a b\n+x\n"}]
    asyncio.run(scorer(traces))
    return captured["cmd"], metrics


@pytest.mark.parametrize("threads", ["0", "2"])
def test_scoring_runs_inside_the_fork_image_not_the_host(monkeypatch, threads):
    monkeypatch.setenv("GBENCH_SWE_OMP_THREADS", threads)
    cmd, metrics = _run_scorer("gbench-swe-bench-live:a145aa87", monkeypatch)
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "gbench-swe-bench-live:a145aa87" in cmd
    # inside the image the fork's harness runs as a module: `python -m <module>` when the thread
    # cap is off, or the thread-cap bootstrap (`python -c`), which runs that same module as
    # __main__ after installing the cap; the harness arguments follow unchanged.
    inner = cmd[cmd.index("gbench-swe-bench-live:a145aa87") + 1:]
    if threads == "0":
        assert inner[:3] == ["python", "-m", "swebench.harness.run_evaluation"]
        rest = inner[3:]
    else:
        assert inner[:2] == ["python", "-c"]
        assert ("runpy.run_module('swebench.harness.run_evaluation', run_name='__main__'"
                in inner[2]) and "install(2)" in inner[2]
        rest = inner[3:]
    assert rest[:2] == ["--dataset_name", "SWE-bench-Live/SWE-bench-Live"]
    # docker-out-of-docker: task containers spawn on the host daemon
    assert "/var/run/docker.sock:/var/run/docker.sock" in cmd
    # the fork must NOT run on the host interpreter (that is upstream swebench)
    assert sys.executable not in cmd
    assert metrics.get("harness_image") == "gbench-swe-bench-live:a145aa87"


def test_host_path_unchanged_when_no_image(monkeypatch):
    # harness_image=None (swe_bench_multilingual / copilot / pro / multi): host interpreter, no docker run
    cmd, metrics = _run_scorer(None, monkeypatch)
    assert cmd[0] != "docker"
    assert sys.executable in cmd
    # the host path may wrap the module in a thread-cap `-c` shim, so match anywhere in the argv
    assert any("swebench.harness.run_evaluation" in str(c) for c in cmd)
    assert "harness_image" not in metrics


def test_hard_errors_without_the_fork_image(monkeypatch):
    monkeypatch.setattr(S.shutil, "which", lambda _x: "/usr/bin/docker")

    def _fake_run(cmd, **kw):
        class R:
            # docker info OK, but `docker image inspect <fork image>` fails (image not built)
            returncode = 0 if cmd[:2] == ["docker", "info"] else 1
        return R()

    monkeypatch.setattr(S.subprocess, "run", _fake_run)
    ok, reason = S.check_swe_bench_live_prerequisites(S.PYTHON_SETTINGS)
    assert not ok
    assert "swe_bench_live.Dockerfile" in reason and "docker build" in reason


def test_run_forwards_the_image_as_harness_image(monkeypatch):
    seen = {}

    def _fake_execute(eval_name, model_name, base_url, concurrency, dataset, split, namespace, **kwargs):
        seen.update(kwargs)
        return {"status": "success"}

    monkeypatch.setattr(S, "check_swe_bench_live_prerequisites", lambda *a: (True, ""))
    monkeypatch.setattr(S, "execute_swebench", _fake_execute)
    # execute_swebench is the one-shot path; agentic (the default) is covered in
    # tests/test_swe_agentic_default_protocol.py.
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_MODE", "oneshot")
    S.run_swe_bench_live("gemma", "http://x/v1", 1, eval_categories="lite,full,verified")
    assert seen.get("harness_image") == S._image()
