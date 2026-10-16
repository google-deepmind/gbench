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

"""terminal_bench: Harbor runs with the leaderboard's trial retries; the harness container gets
the daemon socket the docker CLI uses; the run's tasks are cached and their images pulled on the
host before Harbor starts; exceptions kept as the model's outcome are named in status_reason; a
merge records each shard's harness image. Docker and Harbor are stubbed; nothing is written
outside tmp."""

import subprocess
from unittest import mock

import pytest

from gbench.runners.eval_suites import terminal_bench as TB


def _captured_harbor_argv():
    seen = {}

    def _popen(cmd, *a, **k):
        seen["cmd"] = cmd
        raise RuntimeError("argv captured")
    with mock.patch.object(TB, "check_terminal_bench_prerequisites", return_value=(True, "")), \
         mock.patch.object(TB, "require_task_metadata_blocked", return_value=None), \
         mock.patch.object(TB, "prepare_task_images", return_value=({}, {})), \
         mock.patch.object(TB.subprocess, "Popen", _popen):
        with pytest.raises(RuntimeError):
            TB.run_terminal_bench("m", "http://x:8000/v1", 1)
    return seen["cmd"]


def test_harbor_runs_with_the_leaderboards_trial_retries():
    cmd = _captured_harbor_argv()
    i = cmd.index("--max-retries")
    assert cmd[i + 1] == str(TB.TRIAL_MAX_RETRIES) == "3"
    assert cmd[cmd.index("--n-attempts") + 1] == str(TB.TERMINAL_BENCH_TRIALS)


def test_the_dead_image_version_helper_is_gone():
    assert not hasattr(TB, "installed_harbor_version")


# --------------------------------------------------------------------------- #
# The daemon socket the docker CLI uses.
# --------------------------------------------------------------------------- #
def _context(monkeypatch, endpoint, rc=0):
    monkeypatch.setattr(TB.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, rc, stdout=endpoint + "\n", stderr=""))


@pytest.mark.parametrize("docker_host, context, expected", [
    ("", "unix:///var/run/docker.sock", TB._DOCKER_SOCKET),
    ("", "", TB._DOCKER_SOCKET),
    ("unix:///run/user/1000/docker.sock", "", "/run/user/1000/docker.sock"),
    ("", "unix:///home/u/.docker/run/docker.sock", "/home/u/.docker/run/docker.sock"),
])
def test_the_socket_follows_docker_host_and_the_context(monkeypatch, docker_host, context,
                                                        expected):
    if docker_host:
        monkeypatch.setenv("DOCKER_HOST", docker_host)
    else:
        monkeypatch.delenv("DOCKER_HOST", raising=False)
    _context(monkeypatch, context)
    assert TB.docker_socket_path() == (expected, "")


@pytest.mark.parametrize("endpoint", ["tcp://10.0.0.5:2376", "ssh://builder@10.0.0.5"])
def test_a_non_unix_endpoint_cannot_be_driven_from_the_harness_container(monkeypatch, endpoint):
    monkeypatch.setenv("DOCKER_HOST", endpoint)
    path, why = TB.docker_socket_path()
    assert path is None and "only a unix:// endpoint" in why
    assert "10.0.0.5" not in why and "builder" not in why        # the value is never repeated
    monkeypatch.setattr(TB.shutil, "which", lambda _x: "/usr/bin/docker")
    _context(monkeypatch, "")
    ok, reason = TB.check_terminal_bench_prerequisites()
    assert not ok and "only a unix:// endpoint" in reason


def test_the_container_mounts_the_resolved_socket(monkeypatch, tmp_path):
    sock = tmp_path / "docker.sock"
    sock.write_text("")
    monkeypatch.setitem(TB._VERIFIED_SOCKET, "path", str(sock))
    cmd = TB.harbor_command(["run"], jobs_dir=str(tmp_path / "j"), name="c")
    assert f"{sock}:/var/run/docker.sock" in cmd
    assert cmd[cmd.index("--group-add") + 1] == str(sock.stat().st_gid)


# --------------------------------------------------------------------------- #
# The run's tasks cached and their images pulled on the host.
# --------------------------------------------------------------------------- #
def _cache(tmp_path, monkeypatch, tasks):
    root = tmp_path / "packages"
    for name, image in tasks.items():
        d = root / "terminal-bench" / name
        d.mkdir(parents=True)
        (d / "task.toml").write_text(f'[environment]\ndocker_image = "{image}"\n')
    monkeypatch.setattr(TB, "harbor_package_cache", lambda: str(root))
    monkeypatch.setattr(TB, "harbor_cache_dir", lambda: str(tmp_path / "harbor"))
    return root


def test_a_cached_run_downloads_nothing(monkeypatch, tmp_path):
    _cache(tmp_path, monkeypatch, {"a": "x/a:1"})
    rec = TB.ensure_dataset_cached(["terminal-bench/a"],
                                   run=lambda *a, **k: pytest.fail("downloaded"))
    assert rec == {"cached": 1, "of": 1, "download": None}


def test_a_missing_task_is_downloaded_with_the_harness_image(monkeypatch, tmp_path):
    root = _cache(tmp_path, monkeypatch, {"a": "x/a:1"})
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        d = root / "terminal-bench" / "b"
        d.mkdir(parents=True)
        (d / "task.toml").write_text('[environment]\ndocker_image = "x/b:1"\n')
        return subprocess.CompletedProcess(cmd, 0, "", "")
    rec = TB.ensure_dataset_cached(["terminal-bench/a", "terminal-bench/b"], run=run)
    assert rec == {"cached": 1, "of": 2, "download": {"returncode": 0, "cached_after": 2}}
    [cmd] = calls
    assert cmd[-4:] == ["dataset", "download", TB.DATASET_SPEC, "--cache"]
    assert TB.harbor_image() in cmd and not any("docker.sock" in a for a in cmd)


def test_a_failed_download_is_recorded_not_fatal(monkeypatch, tmp_path):
    _cache(tmp_path, monkeypatch, {})
    rec = TB.ensure_dataset_cached(None, run=lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, "", "registry down"))
    assert rec["download"] == {"returncode": 1, "cached_after": 0}
    assert rec["of"] == TB.FULL_TASK_COUNT


def test_the_prepull_pulls_only_what_is_missing_and_records_failures(monkeypatch):
    present = {"x/a:1"}
    pulls = []

    def run(cmd, **kw):
        if cmd[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0 if cmd[-1] in present else 1, "", "")
        if cmd[:2] == ["docker", "pull"]:
            pulls.append(cmd[2])
            ok = cmd[2] != "x/c:1"
            return subprocess.CompletedProcess(cmd, 0 if ok else 1, "",
                                               "" if ok else "toomanyrequests: rate limit")
        raise AssertionError(cmd)
    monkeypatch.setattr(TB.swebench_common.subprocess, "run", run)
    rec = TB.prepull_task_images(["x/a:1", "x/b:1", "x/c:1", "x/b:1"], concurrency=2)
    assert sorted(pulls) == ["x/b:1", "x/c:1"]
    assert rec["images"] == 3 and rec["present"] == 1 and rec["pulled"] == 1
    assert list(rec["failed"]) == ["x/c:1"] and "toomanyrequests" in rec["failed"]["x/c:1"]


def test_the_gate_runs_before_the_prepull(monkeypatch):
    order = []
    monkeypatch.setattr(TB, "check_terminal_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(TB, "require_task_metadata_blocked",
                        lambda task_names=None: order.append("gate"))
    monkeypatch.setattr(TB, "prepare_task_images",
                        lambda task_names=None: order.append("prepare") or ({}, {}))
    monkeypatch.setattr(TB.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("stop")))
    with pytest.raises(RuntimeError):
        TB.run_terminal_bench("m", "http://x:8000/v1", 1)
    assert order == ["gate", "prepare"]


# --------------------------------------------------------------------------- #
# Reporting: exceptions kept as the model's outcome, the shards' harness images.
# --------------------------------------------------------------------------- #
def test_exceptions_kept_as_the_models_outcome_are_named():
    assert TB.model_outcome_exceptions_note({}) == ""
    note = TB.model_outcome_exceptions_note({"RuntimeError": 2})
    assert note.startswith("2 trial(s) raised an exception not known as a harness failure")
    assert "RuntimeError x2" in note
    assert TB._with_note(None, note) == note
    assert TB._with_note("harness errors: 1", note) == f"harness errors: 1; {note}"


def _shard(label, lock, kept=None):
    return {"shard": label, "sample_traces": [], "attempts": {"attempts_per_sample": 5},
            "harness_image": {"image": TB.harbor_image(), "id": f"sha256:{label}",
                              "harbor": TB.HARBOR_VERSION, "lock": lock},
            "exceptions_kept_as_model_outcome": kept or {}, "status": "success",
            "task_image_prepull": {"present": 1}, "dataset_cache": {"cached": 1}}


def test_merge_records_each_shards_harness_image():
    same = TB.merge_shards([_shard("1/2", TB.TB21_LOCK_COMMIT, {"RuntimeError": 1}),
                            _shard("2/2", TB.TB21_LOCK_COMMIT)])
    assert set(same["harness_image_by_shard"]) == {"1/2", "2/2"}
    assert same["harness_image"]["lock"] == TB.TB21_LOCK_COMMIT
    assert same["harness_image"]["id"] is None and same["harness_image_note"] is None
    assert "RuntimeError x1" in same["status_reason"]
    assert same["task_image_prepull_by_shard"] == {"1/2": {"present": 1}, "2/2": {"present": 1}}
    differ = TB.merge_shards([_shard("1/2", TB.TB21_LOCK_COMMIT), _shard("2/2", "0" * 40)])
    assert differ["harness_image"] is None and "harness_image_by_shard" in \
        differ["harness_image_note"]


def test_the_guide_describes_retries_socket_and_prepull():
    doc = " ".join(open("docs/evals/terminal_bench.md", encoding="utf-8").read().split())
    for needle in ("`harbor run --max-retries 3`", "`trial_max_retries`", "`DOCKER_HOST`",
                   "`task_image_prepull`", "`dataset_cache`", "`harness_image_by_shard`",
                   "`status_reason` names them"):
        assert needle in doc, needle
