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

"""What gbench starts for evals: the vLLM server's flags, and the code sandbox's view of paths.

mmlu sends hendrycks/test evaluate.py's request with logprobs=100; vLLM rejects any request
above its --max-logprobs (default 20), so a gbench-managed mmlu run needs the eval server
started with --max-logprobs 100. Popen is stubbed in those tests: no server is launched.

The code sandbox (GBENCH_SANDBOX) empties the home directory and binds back what a program
reads. A path reached through a symlink inside the home (an interpreter run as
~/venv/bin/python, a hub cache under a symlinked ~/.cache, also when the home itself is reached
through a symlink) must still resolve in the jail, or the sandbox probe fails with the wrong
diagnosis and a data file a scorer opens is missing. Those tests build a fake home under
/var/tmp (the jail empties /tmp) and run bubblewrap where it works; no model server is used.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from gbench.core.config import BenchmarkConfig
from gbench.core.models import ModelConfig, ModelFormat
from gbench.runners import serving
from gbench.runners.serving import EVAL_SERVER_MAX_LOGPROBS, ServingBenchmarkRunner

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_port_probe(monkeypatch):
    """The launch is stopped at Popen: pick a fixed port instead of probing local ports."""
    monkeypatch.setattr(serving, "resolve_server_port", lambda environ=None: (18123, "auto"))


def _planned_cmd(tmp_path, **start_kwargs):
    """The argv _start_server would hand to Popen (the launch is stopped there)."""

    class _LM:
        def get_server_log_path(self, *a):
            return str(tmp_path / "server.log")

    cfg = BenchmarkConfig()
    cfg.max_model_len_explicit = None
    cfg.log_manager = _LM()
    runner = ServingBenchmarkRunner(cfg)
    captured = []

    def fake_popen(cmd, *a, **k):
        captured.append(list(cmd))
        raise RuntimeError("stop before launch")

    model = ModelConfig(short_name="m", hf_model_id="org/m")
    with mock.patch("subprocess.Popen", fake_popen), \
            mock.patch.object(runner, "_cleanup_server", lambda: None):
        assert runner._start_server(model, ModelFormat.HF, **start_kwargs) is False
    assert captured, "Popen was never reached"
    return captured[0]


def _flag_values(cmd, flag):
    return [cmd[i + 1] for i, a in enumerate(cmd) if a == flag]


def test_the_eval_server_accepts_mmlus_top_100_logprobs(tmp_path):
    cmd = _planned_cmd(tmp_path, context="native")
    assert _flag_values(cmd, "--max-logprobs") == ["100"]


def test_the_cap_is_mmlus_own_request_value():
    from gbench.runners.eval_suites import mmlu
    assert EVAL_SERVER_MAX_LOGPROBS == mmlu.TOP_LOGPROBS == 100


@pytest.mark.parametrize("kwargs", [
    {},                                                    # serving pillar
    {"context": "perf"},
    {"context": "perf", "mm_limit_images": 4, "allowed_local_media_path": "/tmp"},  # stress MM
])
def test_performance_servers_keep_vllms_default(tmp_path, kwargs):
    assert "--max-logprobs" not in _planned_cmd(tmp_path, **kwargs)


def test_the_eval_launchers_ask_for_the_native_server():
    """The CLI starts the one eval / quality / Golden Set server per model; the flag rides on
    the native context, so it must ask for it. Any runner that still starts a server of its own
    (a programmatic caller without config.local_endpoint) must ask for the same context."""
    import re
    cli_src = (REPO / "gbench/cli.py").read_text(encoding="utf-8")
    assert '_start_server(model, format, context="native")' in cli_src
    for rel in ("gbench/cli.py", "gbench/runners/evals.py", "gbench/runners/quality.py"):
        src = (REPO / rel).read_text(encoding="utf-8")
        for call in re.findall(r"_start_server\(model, format[^)]*\)", src):
            assert 'context="native"' in call, (rel, call)


def test_the_flag_is_added_only_where_the_context_is_native():
    import inspect
    src = inspect.getsource(serving.ServingBenchmarkRunner.build_server_cmd)
    block = src[src.index('"--max-logprobs"') - 200:src.index('"--max-logprobs"')]
    assert 'if context == "native":' in block


def test_the_docs_state_the_flag():
    readme = (REPO / "docs" / "performance.md").read_text(encoding="utf-8")
    assert "`--max-logprobs 100`" in readme
    assert ("Prefix caching stays on for these servers; it is off only on the servers gBench "
            "launches for performance runs, as defence in depth (performance requests are "
            "already unique). They are also started with `--max-logprobs 100`") in readme
    knobs = (REPO / "docs" / "evals" / "env-knobs.md").read_text(encoding="utf-8")
    assert "--max-logprobs 100" in knobs


# --------------------------------------------------------------------------- #
# the code sandbox: paths reached through a symlink inside the emptied home
# --------------------------------------------------------------------------- #
from gbench.runners.eval_suites import sandbox  # noqa: E402

_HF_VARS = ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME",
            "HF_TOKEN_PATH", "SCICODE_TEST_DATA", "GBENCH_SCICODE_TEST_DATA")


def _var_tmp_root(prefix):
    base = "/var/tmp"
    if not (os.path.isdir(base) and os.access(base, os.W_OK)):
        pytest.skip("needs a writable temporary directory outside /tmp (/var/tmp)")
    return tempfile.TemporaryDirectory(dir=base, prefix=prefix)


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


@pytest.fixture
def jail(monkeypatch):
    monkeypatch.setenv("GBENCH_SANDBOX", "required")
    sandbox.sandbox_available.cache_clear()
    sandbox._probe_bwrap.cache_clear()
    try:
        if not sandbox.sandbox_available():
            pytest.skip("bubblewrap is unavailable or blocked on this host")
        yield
    finally:
        sandbox.sandbox_available.cache_clear()
        sandbox._probe_bwrap.cache_clear()


@pytest.fixture
def linked_home(monkeypatch):
    """HOME reached through a symlinked parent (<root>/homes -> <root>/realhomes), with
    ~/.cache a symlink to another disk holding the hub cache and the token next to it."""
    with _var_tmp_root("gbench-sbx-home-") as root:
        root = os.path.realpath(root)
        real_home = os.path.join(root, "realhomes", "user")
        os.makedirs(real_home)
        os.symlink(os.path.join(root, "realhomes"), os.path.join(root, "homes"))
        disk = os.path.join(root, "disk")
        _write(f"{disk}/cache/huggingface/hub/datasets--z/snapshots/r/test_data.h5", "hub-h5")
        _write(f"{disk}/cache/huggingface/token", "hf_FAKEhomelinked0001")
        os.symlink(f"{disk}/cache", f"{real_home}/.cache")
        for name in _HF_VARS:
            monkeypatch.delenv(name, raising=False)
        home = os.path.join(root, "homes", "user")
        monkeypatch.setenv("HOME", home)
        yield home, real_home, disk


def test_a_link_in_a_home_reached_through_a_link_is_bound_where_it_lands(linked_home):
    home, real_home, disk = linked_home
    binds = sandbox._home_binds(sandbox.masked_homes())
    assert (f"{disk}/cache/huggingface/hub", f"{real_home}/.cache/huggingface/hub") in binds
    assert sandbox._through_home_link(f"{home}/.cache/huggingface/hub", [real_home]) == (
        f"{real_home}/.cache/huggingface/hub")
    assert sandbox._through_home_link(f"{home}/plain/file", [real_home]) is None
    assert sandbox._through_home_link(f"{disk}/cache", [real_home]) is None


def test_the_hub_file_resolves_in_the_jail_and_the_token_stays_hidden(linked_home, jail):
    home, _, disk = linked_home
    targets = {"hub": f"{home}/.cache/huggingface/hub/datasets--z/snapshots/r/test_data.h5",
               "token_as_written": f"{home}/.cache/huggingface/token",
               "token_real": f"{disk}/cache/huggingface/token"}
    prog = ("import json, sys\nout = {}\n"
            "for k, p in json.loads(sys.argv[1]).items():\n"
            "    try:\n        out[k] = open(p).read()\n"
            "    except OSError as e:\n        out[k] = type(e).__name__\n"
            "print(json.dumps(out))\n")
    r = sandbox.run_sandboxed([sys.executable, "-c", prog, json.dumps(targets)],
                              capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    seen = json.loads(r.stdout.strip().splitlines()[-1])
    assert seen["hub"] == "hub-h5"
    for name in ("token_as_written", "token_real"):
        assert seen[name] in ("", "PermissionError", "FileNotFoundError"), (name, seen[name])
    assert "hf_FAKEhomelinked0001" not in r.stdout


_PROBE_CHILD = r"""
import os, sys
sys.path.insert(0, sys.argv[1])
from gbench.runners.eval_suites import sandbox
print("exe", sys.executable)
print("probe", int(sandbox._probe_bwrap()))
"""


def test_an_interpreter_reached_through_a_home_symlink_passes_the_probe(jail):
    """The sandbox probe runs [sys.executable, "-c", "print(1)"] in the jail. With the
    interpreter reached as ~/venv/bin/python (~/venv a symlink to the real environment), that
    path must resolve in the emptied home, or every code-executing suite hard-errors with a
    'bubblewrap is BLOCKED' message that names the wrong fix."""
    with _var_tmp_root("gbench-sbx-venv-") as root:
        root = os.path.realpath(root)
        home = os.path.join(root, "home")
        os.makedirs(home)
        os.symlink(os.path.realpath(sys.prefix), os.path.join(home, "venv"))
        exe = os.path.join(home, "venv", "bin", os.path.basename(sys.executable))
        if not os.path.exists(exe):
            pytest.skip(f"this interpreter is not reachable as <prefix>/bin/{exe}")
        env = sandbox.credential_free_env({k: v for k, v in os.environ.items()
                                           if k not in _HF_VARS})
        env.update(HOME=home, GBENCH_SANDBOX="required")
        r = subprocess.run([exe, "-c", _PROBE_CHILD, str(REPO)], capture_output=True,
                           text=True, timeout=120, env=env, cwd=str(REPO))
        assert r.returncode == 0, r.stderr[-2000:]
        assert f"exe {exe}" in r.stdout, r.stdout
        assert "probe 1" in r.stdout, (r.stdout, r.stderr[-2000:])
