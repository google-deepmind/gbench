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

"""The host-wide endpoint lock: one key per upstream server, shared for traffic,
exclusive for perf pillars, never a silent wait (real processes contend)."""

import multiprocessing as mp
import os
import time

import pytest

from gbench.utils import endpoint_lock as el


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv(el.KEY_ENV, raising=False)
    monkeypatch.delenv(el.WAIT_ENV, raising=False)


def test_local_addresses_share_one_key():
    keys = {el.endpoint_key(u) for u in ("http://localhost:8000/v1", "http://127.0.0.1:8000",
                                         "http://0.0.0.0:8000/v1", "127.0.0.1:8000")}
    assert keys == {"local:8000"}
    assert el.endpoint_key("http://127.0.0.1:9000") == "local:9000"
    assert el.endpoint_key("https://api.example.invalid/v1") == "api.example.invalid:443"


def test_children_inherit_the_parent_key(monkeypatch):
    monkeypatch.setenv(el.KEY_ENV, "local:8000")
    # a smoke tap / relay URL must not derive its own key
    assert el.inherited_or("http://127.0.0.1:43123/v1") == "local:8000"


def _hold(cache, key, exclusive, ready, release):
    os.environ["GBENCH_CACHE_DIR"] = cache
    with el.hold(key, exclusive=exclusive):
        ready.set()
        release.wait(30)


def _spawn(tmp_path, exclusive):
    ctx = mp.get_context("spawn")
    ready, release = ctx.Event(), ctx.Event()
    p = ctx.Process(target=_hold, args=(str(tmp_path), "local:8000", exclusive, ready, release))
    p.start()
    assert ready.wait(30)
    return p, release


def test_shared_holders_coexist_and_exclusive_fails_fast_naming_them(tmp_path):
    p, release = _spawn(tmp_path, exclusive=False)
    try:
        with el.hold("local:8000", exclusive=False):     # a second shared holder is fine
            pass
        with pytest.raises(el.EndpointBusy) as e:
            with el.hold("local:8000", exclusive=True, wait_s=0):
                pass
        assert f"pid {p.pid}" in str(e.value) and "shared" in str(e.value)
    finally:
        release.set()
        p.join(10)


def test_shared_fails_at_once_under_a_perf_pillar(tmp_path):
    p, release = _spawn(tmp_path, exclusive=True)
    try:
        t = time.time()
        with pytest.raises(el.EndpointBusy) as e:
            with el.hold("local:8000", exclusive=False):
                pass
        assert time.time() - t < 2.0
        assert "performance pillar" in str(e.value) and f"pid {p.pid}" in str(e.value)
    finally:
        release.set()
        p.join(10)
    with el.hold("local:8000", exclusive=True):           # free again once it exits
        assert el.holders("local:8000")[0]["pid"] == os.getpid()
    assert el.holders("local:8000") == []


def test_exclusive_waits_up_to_the_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(el, "POLL_S", 0.2)
    p, release = _spawn(tmp_path, exclusive=False)
    t = time.time()
    with pytest.raises(el.EndpointBusy):
        with el.hold("local:8000", exclusive=True, wait_s=1.0):
            pass
    assert 0.9 <= time.time() - t < 5.0
    release.set()
    p.join(10)


def test_no_key_locks_nothing():
    with el.hold(None, exclusive=True) as k:
        assert k is None


def test_pool_and_smoke_refuse_to_start_under_a_perf_pillar(tmp_path):
    """The CLI wrapper of --parallel-suites / --smoke fails fast, naming the perf pillar,
    without running anything."""
    from gbench import cli
    p, release = _spawn(tmp_path, exclusive=True)
    ran = []
    try:
        args = type("A", (), {"remote_endpoint": "http://127.0.0.1:8000/v1"})()
        code = cli._shared_endpoint(args, lambda: ran.append(1) or 0)
    finally:
        release.set()
        p.join(10)
    assert code == cli.EXIT_HARNESS_ERROR and ran == []
    assert os.environ[el.KEY_ENV] == "local:8000"            # exported for the children


def test_a_server_gbench_launched_is_locked_while_it_runs(tmp_path):
    """Live finding (2026-10-15): an eval run pointed at the port of a perf server gbench
    launched itself sent traffic into the stress sweep. The runner now holds the lock on
    local:<port> for its server's lifetime: exclusive for perf, shared for native."""
    from gbench.core.config import BenchmarkConfig
    from gbench.runners.serving import ServingBenchmarkRunner
    r = ServingBenchmarkRunner(BenchmarkConfig())
    r.server_port = 8123
    assert r._hold_server_lock("perf") is True
    with pytest.raises(el.EndpointBusy):                   # an eval run against that port
        with el.hold(el.endpoint_key("http://localhost:8123/v1"), exclusive=False):
            pass
    r._cleanup_server()                                    # releases it
    with el.hold("local:8123", exclusive=True):
        pass
    assert r._hold_server_lock("native") is True           # native: shared
    with el.hold("local:8123", exclusive=False):
        pass
    r._release_server_lock()


def test_launch_refuses_a_port_another_gbench_run_holds(tmp_path):
    from gbench.core.config import BenchmarkConfig
    from gbench.runners.serving import ServingBenchmarkRunner
    p, release = _spawn(tmp_path, exclusive=False)         # holds local:8000 shared
    try:
        r = ServingBenchmarkRunner(BenchmarkConfig())
        r.server_port = 8000
        assert r._hold_server_lock("perf") is False
    finally:
        release.set()
        p.join(10)
