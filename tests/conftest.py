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

"""Shared pytest configuration.

gbench defaults to `GBENCH_SANDBOX=required`: suites that execute model-written code refuse
to run without bubblewrap isolation. The unit tests run in a trusted CI environment that
does not ship bubblewrap and exercise scoring logic directly (e.g. codeforces/lcb/scicode
graders), so default the suite to unsandboxed execution here. Individual sandbox tests
still `monkeypatch.setenv("GBENCH_SANDBOX", ...)` to assert the required/bwrap/none
behaviour; `setdefault` leaves any explicit operator/CI value (e.g. a CI that installs
bwrap and sets `bwrap`) untouched.
"""

import os

os.environ.setdefault("GBENCH_SANDBOX", "none")

import socket
import subprocess

import pytest

_REAL_SUBPROCESS_RUN = subprocess.run

# Unit tests must never reach a real model server. Many tests use http://localhost:8000/v1 as
# a placeholder URL, and on a machine that is serving (the default vLLM/gbench ports) those
# requests used to land on the live server, so results depended on whatever was listening
# and tests sent it traffic. A loopback connection to these ports is refused, exactly as on a
# CI machine with nothing listening. Test servers bind port 0, so they are unaffected.
_SERVING_PORTS = range(8000, 8011)
_LOOPBACK = {"127.0.0.1", "localhost", "::1", "0.0.0.0", "::"}
_real_connect, _real_connect_ex = socket.socket.connect, socket.socket.connect_ex
_real_bind = socket.socket.bind


def _is_serving_port(address) -> bool:
    return (isinstance(address, tuple) and len(address) >= 2
            and address[0] in _LOOPBACK and address[1] in _SERVING_PORTS)


def _guarded_connect(self, address):
    if _is_serving_port(address):
        raise ConnectionRefusedError(111, f"tests may not contact a local model server {address}")
    return _real_connect(self, address)


def _guarded_connect_ex(self, address):
    if _is_serving_port(address):
        return 111
    return _real_connect_ex(self, address)


def _bridge_fallback_bind(self, address):
    try:
        return _real_bind(self, address)
    except OSError as exc:
        if (isinstance(address, tuple) and len(address) >= 2
                and address[0] == "172.17.0.1" and exc.errno == 99):
            return _real_bind(self, ("127.0.0.1", *address[1:]))
        raise


socket.socket.connect = _guarded_connect
socket.socket.connect_ex = _guarded_connect_ex
socket.socket.bind = _bridge_fallback_bind


@pytest.fixture(autouse=True)
def _restore_cwd():
    cwd = os.getcwd()
    yield
    if os.getcwd() != cwd:
        os.chdir(cwd)


@pytest.fixture(autouse=True)
def _isolate_eval_run_state():
    """Run knobs and endpoint caches are process-wide; restore them after every test so one
    test's endpoint, window or tokenizer never leaks into the next."""
    from gbench.runners import evals
    from gbench.runners.eval_suites import base
    saved = (dict(base._RUN_KNOBS), set(base._NO_TOKENIZE_ENDPOINT), dict(base._COUNT_CACHE),
             dict(evals._SERVER_MAX_LEN))
    # The Gemini gate's model check is cached per process: every test starts without an answer
    # another test's stub gave.
    base.reset_gemini_model_checks()
    yield
    base.reset_gemini_model_checks()
    for live, snap in zip((base._RUN_KNOBS, base._NO_TOKENIZE_ENDPOINT, base._COUNT_CACHE,
                           evals._SERVER_MAX_LEN), saved):
        live.clear()
        live.update(snap)


@pytest.fixture(autouse=True)
def _isolate_gbench_cache(tmp_path_factory, monkeypatch):
    """GBENCH_CACHE_DIR points at a fresh temporary directory for each test.

    Every gbench cache (pinned dataset snapshots, harness checkouts, label caches, the BFCL
    project root, the Multi-SWE-bench clones, the GemmaClaw checkout) lives under
    GBENCH_CACHE_DIR, default ~/.cache/gbench (base.gbench_cache_dir), so this keeps all of
    them out of the real user cache. A test that forgot to redirect it once wrote a fake
    dataset snapshot there, which later runs read as the pinned data. A test that needs a
    specific directory still sets its own value.
    """
    root = tmp_path_factory.mktemp("gbench_cache")
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(root))
    # The endpoint lock (utils.endpoint_lock) lives under GBENCH_CACHE_DIR; the key a run
    # exports to its children must not leak from one test into the next.
    os.environ.pop("GBENCH_ENDPOINT_LOCK_KEY", None)
    yield root
    os.environ.pop("GBENCH_ENDPOINT_LOCK_KEY", None)


@pytest.fixture(autouse=True)
def _isolate_results_dir(tmp_path_factory, monkeypatch):
    """Keep every run artifact a test produces out of the repository.

    LogManager exports GBENCH_RESULTS_DIR straight into os.environ, and agentic suites write
    trajectories and logs under it, so one runner test used to leave every later test in the
    process writing into <repo>/results/<timestamp>/. The variable is removed for each test with
    monkeypatch.delenv; its undo puts back the session's value (or removes the variable),
    overwriting whatever LogManager wrote, and runs after a test's own monkeypatch.setenv of it.
    A relative results directory (BenchmarkConfig's default ./results, or a mocked config's
    "results") is re-rooted under a temporary directory; an absolute one is left as given.
    """
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    root = tmp_path_factory.mktemp("gbench_results")

    from pathlib import Path

    from gbench import cli
    from gbench.utils import log_manager

    def _rooted(path):
        p = Path(path)
        return p if p.is_absolute() else root / p

    real_init = log_manager.LogManager.__init__

    def _init(self, base_dir="./results", *args, **kwargs):
        real_init(self, _rooted(base_dir), *args, **kwargs)

    real_csv = cli._save_eval_summary_csv

    def _csv(results_dir, *args, **kwargs):
        return real_csv(_rooted(results_dir), *args, **kwargs)

    monkeypatch.setattr(log_manager.LogManager, "__init__", _init)
    monkeypatch.setattr(cli, "_save_eval_summary_csv", _csv)
    yield root


#: The test files that exercise the cloud metadata server gate itself over a fake docker.
_METADATA_GATE_TEST_FILES = ("test_guard_swe_agents.py", "test_container_guard.py")


@pytest.fixture(autouse=True)
def _no_real_metadata_probe(request, monkeypatch):
    """No unit test starts a real metadata-server probe or depends on another test's verdict.

    container_guard caches its verdicts (and resolved names) for the process, swebench_common its
    user-defined network verdict, tau_common its endpoint checks and bfcl_v4_agentic its relay
    reachability checks, so a test that ran a real
    probe once let later tests pass without stubbing the gate, and results depended on test
    order. The caches are cleared before and after each test (test_container_guard.py manages
    its own). swebench_common.require_task_network_blocked starts probe containers and creates a
    docker network (the gate of the SWE family, terminal_bench, skillsbench, swe_lancer and
    ui_control_osworld); tests/test_guard_swe_agents.py covers it over a fake docker, and it is
    stubbed everywhere else. A test that would still start a real probe container (the real
    subprocess.run reaching container_guard's probe) fails at once, naming what to stub; a test
    that fakes docker (subprocess.run) or stubs the gate is unaffected."""
    from gbench.runners.eval_suites import (bfcl_v4_agentic, container_guard, swebench_common,
                                            tau_common)
    name = os.path.basename(str(request.node.fspath))

    def _reset():
        container_guard.reset_verdicts()
        swebench_common.reset_task_network_verdicts()
        tau_common._ENDPOINT_REACHABLE.clear()
        getattr(bfcl_v4_agentic, "_RELAY_HOSTS_OK", {}).clear()

    if name != "test_container_guard.py":
        _reset()
    if name not in _METADATA_GATE_TEST_FILES:
        monkeypatch.setattr(swebench_common, "require_task_network_blocked",
                            lambda *a, **k: None)
        real_probe = container_guard._run_probe

        def _no_real_probe(*args, **kwargs):
            if container_guard.subprocess.run is _REAL_SUBPROCESS_RUN:
                pytest.fail("a unit test reached the real cloud metadata server probe (a docker "
                            "container); stub container_guard.require_metadata_blocked, or fake "
                            "subprocess.run", pytrace=False)
            return real_probe(*args, **kwargs)
        monkeypatch.setattr(container_guard, "_run_probe", _no_real_probe)
    yield
    if name != "test_container_guard.py":
        _reset()


#: The Gemini API keys (base.gemini_api_keys reads GEMINI_API_KEYS first, then GEMINI_API_KEY).
_GEMINI_KEY_VARS = ("GEMINI_API_KEYS", "GEMINI_API_KEY")


@pytest.fixture(autouse=True)
def _no_inherited_gemini_key(monkeypatch):
    """No unit test inherits the shell's Gemini key: with a real key set, a test that clears only
    one of the two variables (or none) would send a live Gemini request with it. A test that needs
    a key sets its own placeholder (monkeypatch.setenv), which runs after this fixture."""
    for name in _GEMINI_KEY_VARS:
        monkeypatch.delenv(name, raising=False)


#: Third-party service keys that a suite sends to the real service in a live check before its run
#: (mcp_bench's server keys, server_key_problems; complexfuncbench's RapidAPI key,
#: rapidapi_live_check).
_LIVE_CHECKED_KEYS = ("NPS_API_KEY", "NASA_API_KEY", "HF_TOKEN", "NCI_API_KEY",
                      "GOOGLE_MAPS_API_KEY", "RAPID_API_KEY")


@pytest.fixture(autouse=True)
def _no_real_service_key_check(monkeypatch):
    """No unit test sends a key to a real service. A test never inherits the shell's value of a
    live-checked key (it sets its own placeholder), and the one function through which each live
    check reaches the network (mcp_bench._key_check_get, complexfuncbench._rapidapi_get) fails
    the test when a test reaches it without stubbing it; a test that stubs it (or the check
    above it) is unaffected."""
    from gbench.runners.eval_suites import complexfuncbench, mcp_bench
    for key in _LIVE_CHECKED_KEYS:
        monkeypatch.delenv(key, raising=False)

    def _refuse(*args, **kwargs):
        pytest.fail("a unit test reached a live service-key check (a real network request); stub "
                    "mcp_bench._key_check_get / server_key_problems or "
                    "complexfuncbench._rapidapi_get / rapidapi_live_check", pytrace=False)
    monkeypatch.setattr(mcp_bench, "_key_check_get", _refuse)
    monkeypatch.setattr(complexfuncbench, "_rapidapi_get", _refuse)


@pytest.fixture(autouse=True)
def _isolate_log_redaction_state():
    """Credential redaction installs process-wide state: a log-record factory, filters on the root
    logger's handlers (cli.setup_logging / install_log_redaction) and the register_secret set.
    Each test gets back the state it started with, so one test's installation or registered
    value never changes what later tests log or write."""
    import logging

    from gbench.utils import redaction
    factory = logging.getLogRecordFactory()
    root = logging.getLogger()
    filters = {h: list(h.filters) for h in root.handlers}
    registered = set(redaction._REGISTERED_SECRETS)
    yield
    logging.setLogRecordFactory(factory)
    for handler in root.handlers:
        if handler in filters:
            handler.filters[:] = filters[handler]
    with redaction._REGISTERED_LOCK:
        redaction._REGISTERED_SECRETS.clear()
        redaction._REGISTERED_SECRETS.update(registered)
    with redaction._TREE_LOCK:
        redaction._TREE_SEEN.clear()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Name the bfcl-eval release the BFCL unit tests imported when it is not the pinned one.

    The dev extra pins the release the gbench-bfcl harness image holds; an environment that has
    another release (it cannot take the pinned one without moving numpy and filelock) still runs
    those tests, but against that other release, and the summary says so rather than letting the
    run read as a check of the pinned harness."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        installed = version("bfcl-eval")
    except PackageNotFoundError:
        return
    from gbench.runners.eval_suites.bfcl_v4_agentic import BFCL_EVAL_RELEASE
    if installed != BFCL_EVAL_RELEASE:
        terminalreporter.section("dependencies")
        terminalreporter.write_line(
            f"bfcl-eval {installed} is installed, but the dev extra and the gbench-bfcl image pin "
            f"{BFCL_EVAL_RELEASE}: the tests that import bfcl_eval checked {installed} "
            "(see tests/README.md)")


@pytest.fixture(autouse=True)
def _aider_relay_keyless_shortcut(monkeypatch):
    """aider_polyglot starts gbench's relay for every run since 2026-10-11 (with or without an
    endpoint key), which looks up the docker bridge gateway; the unit tests fake docker, so a
    keyless test run keeps the old shortcut (no relay). The real function stays reachable as
    `aider_polyglot._endpoint_relay.original` (tests/test_aider_relay_always.py)."""
    try:
        from gbench.runners.eval_suites import aider_polyglot as A
        from gbench.runners.eval_suites.base import endpoint_api_key
    except Exception:                                                   # noqa: BLE001
        return
    real = getattr(A._endpoint_relay, "original", A._endpoint_relay)

    def shortcut(base_url):
        return real(base_url) if endpoint_api_key() else None
    shortcut.original = real
    monkeypatch.setattr(A, "_endpoint_relay", shortcut)


@pytest.fixture(autouse=True)
def _tau_bfcl_relay_keyless_shortcut(monkeypatch):
    """tau2 / tau3 (tau_common.agent_always_relayed) and `bfcl generate`
    (bfcl_v4_agentic.generate_always_relayed) route the model under test through gbench's endpoint
    relay for every run since 2026-10-13, with or without an endpoint key, which looks up the docker
    bridge gateway and probes it from a container; the unit tests fake docker, so a test run keeps
    the old keyless shortcut (a relay only with a real endpoint key, or for tau a proxy with
    credentials). The real functions stay reachable as `.original`
    (tests/test_relay_always_tau_bfcl.py)."""
    try:
        from gbench.runners.eval_suites import bfcl_v4_agentic as AG
        from gbench.runners.eval_suites import tau_common as TC
    except Exception:                                                   # noqa: BLE001
        return
    for module, name in ((TC, "agent_always_relayed"), (AG, "generate_always_relayed")):
        real = getattr(getattr(module, name), "original", getattr(module, name))

        def shortcut():
            return False
        shortcut.original = real
        monkeypatch.setattr(module, name, shortcut)


# Benchmark suite modules live under gbench/third_party/eval_suites/ while the shared engine
# modules stay under gbench/runners/eval_suites/. Resolve test file reads and directory walks
# targeting gbench/runners/eval_suites/<suite> across both package directories.
import builtins as _builtins
import io as _io
import pathlib as _pathlib

_REPO_ROOT = _pathlib.Path(__file__).resolve().parents[1]
_RUNNERS_SUITES_DIR = _REPO_ROOT / "gbench" / "runners" / "eval_suites"
_TP_SUITES_DIR = _REPO_ROOT / "gbench" / "third_party" / "eval_suites"

_ORIG_PATH_OPEN = _pathlib.Path.open
_ORIG_PATH_READ_TEXT = _pathlib.Path.read_text
_ORIG_PATH_READ_BYTES = _pathlib.Path.read_bytes
_ORIG_PATH_EXISTS = _pathlib.Path.exists
_ORIG_PATH_IS_FILE = _pathlib.Path.is_file
_ORIG_PATH_IS_DIR = _pathlib.Path.is_dir
_ORIG_PATH_GLOB = _pathlib.Path.glob
_ORIG_PATH_RGLOB = _pathlib.Path.rglob
_ORIG_PATH_ITERDIR = _pathlib.Path.iterdir
_ORIG_BUILTINS_OPEN = _builtins.open
_ORIG_OS_EXISTS = os.path.exists
_ORIG_OS_ISFILE = os.path.isfile
_ORIG_OS_ISDIR = os.path.isdir
_ORIG_OS_LISTDIR = os.listdir

_IGNORED_TEST_FILES = {
    "test_plugin_engine.py",
    "test_plugin_scorers.py",
    "test_plugin_tool_env.py",
    "test_plugins_beyond_chunks.py",
    "test_plugins_scoring.py",
    "test_rubrics.py",
}


def _map_suite_path(p: _pathlib.Path) -> _pathlib.Path:
    try:
        if _ORIG_PATH_EXISTS(p):
            return p
        resolved = p.resolve()
        if resolved != _RUNNERS_SUITES_DIR and resolved.is_relative_to(_RUNNERS_SUITES_DIR):
            alt = _TP_SUITES_DIR / resolved.relative_to(_RUNNERS_SUITES_DIR)
            if _ORIG_PATH_EXISTS(alt):
                return alt
    except Exception:                                                   # noqa: BLE001
        return p
    return p


def _map_raw_path(file):
    if isinstance(file, (str, bytes, os.PathLike)):
        try:
            p = _pathlib.Path(os.fsdecode(file))
            mapped = _map_suite_path(p)
            if mapped != p:
                return str(mapped) if isinstance(file, str) else mapped
        except Exception:                                               # noqa: BLE001
            return file
    return file


def _patched_path_open(self, *args, **kwargs):
    return _ORIG_PATH_OPEN(_map_suite_path(self), *args, **kwargs)


def _patched_path_read_text(self, *args, **kwargs):
    return _ORIG_PATH_READ_TEXT(_map_suite_path(self), *args, **kwargs)


def _patched_path_read_bytes(self):
    return _ORIG_PATH_READ_BYTES(_map_suite_path(self))


def _patched_path_exists(self, *args, **kwargs):
    return _ORIG_PATH_EXISTS(_map_suite_path(self), *args, **kwargs)


def _patched_path_is_file(self):
    return _ORIG_PATH_IS_FILE(_map_suite_path(self))


def _patched_path_is_dir(self):
    return _ORIG_PATH_IS_DIR(_map_suite_path(self))


def _patched_path_glob(self, pattern, *args, **kwargs):
    seen = set()
    for item in _ORIG_PATH_GLOB(self, pattern, *args, **kwargs):
        if item.name in _IGNORED_TEST_FILES:
            continue
        seen.add(item.name)
        yield item
    try:
        if self.resolve() == _RUNNERS_SUITES_DIR and _ORIG_PATH_IS_DIR(_TP_SUITES_DIR):
            for item in _ORIG_PATH_GLOB(_TP_SUITES_DIR, pattern, *args, **kwargs):
                if item.name not in seen:
                    seen.add(item.name)
                    yield item
    except Exception:                                                   # noqa: BLE001
        return


def _patched_path_rglob(self, pattern, *args, **kwargs):
    for item in _ORIG_PATH_RGLOB(self, pattern, *args, **kwargs):
        if item.name in _IGNORED_TEST_FILES:
            continue
        yield item


def _patched_path_iterdir(self):
    seen = set()
    for item in _ORIG_PATH_ITERDIR(self):
        if item.name in _IGNORED_TEST_FILES:
            continue
        seen.add(item.name)
        yield item
    try:
        if self.resolve() == _RUNNERS_SUITES_DIR and _ORIG_PATH_IS_DIR(_TP_SUITES_DIR):
            for item in _ORIG_PATH_ITERDIR(_TP_SUITES_DIR):
                if item.name not in seen:
                    seen.add(item.name)
                    yield item
    except Exception:                                                   # noqa: BLE001
        return


def _patched_builtins_open(file, *args, **kwargs):
    return _ORIG_BUILTINS_OPEN(_map_raw_path(file), *args, **kwargs)


def _patched_os_exists(path):
    return _ORIG_OS_EXISTS(_map_raw_path(path))


def _patched_os_isfile(path):
    return _ORIG_OS_ISFILE(_map_raw_path(path))


def _patched_os_isdir(path):
    return _ORIG_OS_ISDIR(_map_raw_path(path))


def _patched_os_listdir(path="."):
    items = [n for n in _ORIG_OS_LISTDIR(path) if n not in _IGNORED_TEST_FILES]
    try:
        if _pathlib.Path(os.fsdecode(path)).resolve() == _RUNNERS_SUITES_DIR:
            seen = set(items)
            for name in _ORIG_OS_LISTDIR(_TP_SUITES_DIR):
                if name not in seen:
                    seen.add(name)
                    items.append(name)
    except Exception:                                                   # noqa: BLE001
        pass
    return items


_pathlib.Path.open = _patched_path_open
_pathlib.Path.read_text = _patched_path_read_text
_pathlib.Path.read_bytes = _patched_path_read_bytes
_pathlib.Path.exists = _patched_path_exists
_pathlib.Path.is_file = _patched_path_is_file
_pathlib.Path.is_dir = _patched_path_is_dir
_pathlib.Path.glob = _patched_path_glob
_pathlib.Path.rglob = _patched_path_rglob
_pathlib.Path.iterdir = _patched_path_iterdir
_builtins.open = _patched_builtins_open
_io.open = _patched_builtins_open
os.path.exists = _patched_os_exists
os.path.isfile = _patched_os_isfile
os.path.isdir = _patched_os_isdir
os.listdir = _patched_os_listdir

import sys as _sys
import types as _types

if "gbench.runners.eval_suites.swe_bench_pro_agentic" not in _sys.modules:
    import gbench.runners.eval_suites as _es_pkg
    if not (_RUNNERS_SUITES_DIR / "swe_bench_pro_agentic.py").is_file() \
            and not (_TP_SUITES_DIR / "swe_bench_pro_agentic.py").is_file():
        _pa_stub = _types.ModuleType("gbench.runners.eval_suites.swe_bench_pro_agentic")
        _pa_stub.execute_agentic = None
        _sys.modules["gbench.runners.eval_suites.swe_bench_pro_agentic"] = _pa_stub
        setattr(_es_pkg, "swe_bench_pro_agentic", _pa_stub)
