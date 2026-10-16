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

"""nestful: the executable Win Rate runs model-controlled code, and its worker is sandboxed; a run
whose requests got no reply withholds its metrics and names the harness errors.

The Win Rate executes the reply's calls of NESTFUL's reference functions with the reply's
arguments, and some of those functions eval their arguments. The worker therefore runs in a
bubblewrap container of its own: no network, an empty environment, no host files beyond the
interpreter and the reference functions. It reports what it can see when it starts, and gbench
refuses a worker that is not isolated. Without a working sandbox the Win Rate alone is
infra_required; nothing ever runs unsandboxed.

The sandbox tests start the real bubblewrap container (the worker cannot run without it); the
model endpoint is stubbed and nothing leaves the machine: the one connection attempted inside the
sandbox targets 192.0.2.1 (TEST-NET-1, never routed) and fails there for lack of a network.
"""
import errno
import json
import os
import subprocess
import types
import sys
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import nestful as N

REPO = Path(__file__).resolve().parents[1]
SENTINEL = "GBENCH_W2Q_FC_SENTINEL"

_BASIC = (
    "def evaluate(expression):\n    return eval(expression)\n\n"
    "def add(a, b):\n    return a + b\n\n"
    "def argv():\n    import sys\n    return ' '.join(sys.argv)\n")
_TOOLS = [{"name": n, "description": "", "parameters": {}, "output_parameters": {"result": {}}}
          for n in ("evaluate", "add", "argv")]


def _func_dir(tmp_path) -> Path:
    fd = tmp_path / "NESTFUL" / "data_v2" / "executable_functions"
    fd.mkdir(parents=True)
    (fd / "basic_functions.py").write_text(_BASIC)
    (fd / "func_file_map.json").write_text("{}")
    return fd


def _call(expression):
    return [{"name": "evaluate", "label": "$var_1", "arguments": {"expression": expression}}]


class _PopenLog:
    """Records every process nestful itself starts (its own subprocess.Popen), then starts it.
    Only nestful's reference to the subprocess module is replaced (a namespace carrying the
    module's names, with Popen recorded), so a library's own processes (joblib's lscpu, say)
    never enter the log, whatever the test order."""

    def __init__(self, monkeypatch):
        self.started = []
        real = subprocess.Popen
        log = self

        class _Recorded(real):
            def __init__(self, argv, *a, **k):
                log.started.append(list(argv))
                super().__init__(argv, *a, **k)
                log.procs.append(self)
        self.procs = []
        proxy = types.SimpleNamespace(**{k: getattr(subprocess, k) for k in dir(subprocess)
                                         if not k.startswith("__")})
        proxy.Popen = _Recorded
        monkeypatch.setattr(N, "subprocess", proxy)


# --------------------------------------------------------------------------- #
# The worker's sandbox
# --------------------------------------------------------------------------- #
def test_worker_sees_no_host_variable_file_or_network(tmp_path, monkeypatch):
    fd = _func_dir(tmp_path)
    monkeypatch.setenv(SENTINEL, "must-not-leak")
    assert os.environ[SENTINEL] == "must-not-leak"            # the host has it
    log = _PopenLog(monkeypatch)
    w = N._WinRateWorker(str(fd))
    try:
        # the worker was started through bubblewrap, never as a bare interpreter
        assert log.started and all(os.path.basename(a[0]) == "bwrap" for a in log.started)
        argv = next(a for a in log.started if sys.executable in a)
        assert "--clearenv" in argv and "--unshare-all" in argv and "--unshare-user" in argv
        assert "--proc" not in argv and "--share-net" not in argv

        def ev(expression, value):
            return w.score(_call(expression), json.dumps(value), _TOOLS)

        # The call runs (control). Each probe returns a string, a list or a non-zero number:
        # a failed call yields False, which upstream's comparison takes as equal to a gold 0.
        assert ev("1 + 1", 2) is True and ev("1 + 1", 3) is False
        # the host variable is not there (and the reply cannot read it)
        assert ev(f"str({SENTINEL!r} in __import__('os').environ)", "False") is True
        assert ev(f"str({SENTINEL!r} in __import__('os').environ)", "True") is False
        allowed = sorted(set(N._SANDBOX_ENV) | N._SANDBOX_ENV_EXTRA)
        assert ev(f"str(sorted(set(__import__('os').environ) - set({allowed!r})))", "[]") is True
        # no network: a connection fails for lack of a route, and loopback is the only interface
        assert ev("__import__('socket').socket().connect_ex(('192.0.2.1', 9))",
                  errno.ENETUNREACH) is True
        assert ev("str([n for _i, n in __import__('socket').if_nameindex()])", "['lo']") is True
        # no host files: no /proc, no gbench tree, no gbench package
        assert ev("str(__import__('os').path.exists('/proc'))", "False") is True
        assert ev(f"str(__import__('os').path.exists({str(REPO)!r}))", "False") is True
        assert ev("str(__import__('importlib').util.find_spec('gbench') is None)", "True") is True
        # the reference functions are read-only
        target = fd / "planted.py"
        assert ev(f"open({str(target)!r}, 'w').write('x')", 1) is False
        assert not target.exists()
        # sys.argv is upstream's scoring command line
        assert w.score([{"name": "argv", "label": "$v", "arguments": {}}],
                       json.dumps(" ".join(N._UPSTREAM_SCORER_ARGV)), _TOOLS) is True
        assert w.kills == 0
        facts = w.facts
        assert facts["ifaces"] == ["lo"] and facts["proc"] is False and facts["run"] is False
        assert SENTINEL not in facts["env"] and facts["functions"] is True
        assert facts["functions_writable"] is False
    finally:
        w.close()
    assert w.proc is None


def test_a_worker_that_is_not_isolated_is_refused_before_any_call(tmp_path, monkeypatch):
    fd = _func_dir(tmp_path)
    # A command without bubblewrap: the worker starts on the host and reports so.
    monkeypatch.setattr(N, "_sandbox_argv", lambda d, **_k: [
        sys.executable, "-I", "-c", N._worker_source() + f"\n_win_worker_main({d!r})\n"])
    log = _PopenLog(monkeypatch)
    with pytest.raises(N.WinRateUnavailable) as e:
        N._WinRateWorker(str(fd))
    assert e.value.prerequisite == N._SANDBOX_PREREQ
    assert "is not isolated as it must be" in e.value.reason
    assert "network interfaces" in e.value.reason and "/proc or /run" in e.value.reason
    assert len(log.procs) == 1 and log.procs[0].poll() is not None     # stopped, never used


def test_isolation_check_names_what_is_wrong():
    fd = "/data/NESTFUL/data_v2/executable_functions"
    good = {"pid": 2, "env": ["HOME", "PATH", "PWD"], "root": ["data", "dev", "tmp", "usr"],
            "proc": False, "run": False, "ifaces": ["lo"], "functions": True,
            "functions_writable": False}
    assert N._isolation_problem(good, fd) is None
    for change, words in (({"env": ["HOME", "PATH", "HF_TOKEN"]}, "environment holds HF_TOKEN"),
                          ({"ifaces": ["ens4", "lo"]}, "network interfaces"),
                          ({"pid": 4242}, "own PID namespace"),
                          ({"proc": True}, "/proc or /run"),
                          ({"root": ["data", "home", "usr"]}, "root holds home"),
                          ({"functions_writable": True}, "writable"),
                          ({"functions": False}, "not visible")):
        why = N._isolation_problem({**good, **change}, fd)
        assert why and words in why, (change, why)
    assert N._isolation_problem(None, fd)


def test_worker_program_imports_nothing_from_gbench():
    src = N._worker_source()
    compile(src, "<nestful win rate worker>", "exec")
    assert "import gbench" not in src and "from gbench" not in src and "from ." not in src
    assert repr(N._UPSTREAM_SCORER_ARGV) in src
    for name in N._WORKER_FUNCTIONS:
        assert f"def {name}(" in src


def test_sandbox_argv_binds_only_the_interpreter_and_the_functions(tmp_path):
    fd = str(_func_dir(tmp_path))
    argv = N._sandbox_argv(fd)
    binds = [argv[i + 1] for i, a in enumerate(argv) if a in ("--ro-bind", "--bind")]
    assert "--bind" not in argv                                  # nothing writable from the host
    allowed = {"/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32", fd}
    allowed |= set(N._interpreter_dirs())
    assert set(b for b in binds if not b.startswith("/etc/ld.so")) <= allowed
    assert argv.index("--tmpfs") < argv.index(fd)                # functions bound after /tmp
    env = {argv[i + 1]: argv[i + 2] for i, a in enumerate(argv) if a == "--setenv"}
    assert env == N._SANDBOX_ENV


# --------------------------------------------------------------------------- #
# run_nestful with the sandbox
# --------------------------------------------------------------------------- #
_OUT = [{"name": "add", "label": "$var_1", "arguments": {"a": 1, "b": 2}}]
_ROWS = [
    {"sample_id": "s1", "input": "1+2?", "tools": _TOOLS, "output": _OUT, "gold_answer": 3},
    {"sample_id": "s2", "input": "2+2?", "tools": _TOOLS,
     "output": [{"name": "add", "label": "$var_1", "arguments": {"a": 2, "b": 2}}],
     "gold_answer": 4},
]


def _checkout(tmp_path, monkeypatch) -> Path:
    fd = _func_dir(tmp_path)
    data = fd.parent / "nestful_data.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in _ROWS))
    monkeypatch.setitem(N._PINNED_BLOBS, "nestful_data.jsonl", N._git_blob_sha1(str(data)))
    monkeypatch.setattr(N, "_func_dir", lambda: str(fd))
    return fd


def _endpoint(answer):
    async def _reply(**kw):
        return answer(kw["messages"][-1]["content"])
    return _reply


def _ok(text):
    return base.Reply(text, None, "stop", completion_tokens=5)


_FAILED = base.Reply(None, None, None, None, "HTTP 500")


def _s1_right_s2_wrong(question):
    return _ok(json.dumps(_OUT) if question == "1+2?" else "[]")


class _InProcWorker:
    def __init__(self, func_dir, *a, **k):
        self.func_dir, self.kills = func_dir, 0

    def score(self, pred, gold, tools):
        return N._win_score(pred, gold, tools, self.func_dir)

    def close(self):
        pass


def test_run_scores_the_win_rate_in_the_sandbox_and_records_it(tmp_path, monkeypatch):
    _checkout(tmp_path, monkeypatch)
    log = _PopenLog(monkeypatch)
    with mock.patch.object(base, "_send_single_request", _endpoint(_s1_right_s2_wrong)):
        r = N.run_nestful("m", "http://x", 1)
    assert r["win_rate"] == 50.0 and r["win_rate_killed_calls"] == 0
    assert "win_rate_metric" not in r and r["status"] == "success"
    assert all(os.path.basename(a[0]) == "bwrap" for a in log.started) and log.started
    box = r["win_rate_sandbox"]
    assert box["sandbox"] == "bubblewrap" and box["network"].startswith("none")
    assert box["worker_reported"]["ifaces"] == ["lo"] and "get_url_content" in box["note"]


def test_without_bubblewrap_only_the_win_rate_is_infra_required(tmp_path, monkeypatch):
    _checkout(tmp_path, monkeypatch)
    real_which = N.shutil.which
    monkeypatch.setattr(N.shutil, "which", lambda name, *a, **k: None if name == "bwrap"
                        else real_which(name, *a, **k))
    log = _PopenLog(monkeypatch)
    with mock.patch.object(base, "_send_single_request", _endpoint(_s1_right_s2_wrong)):
        r = N.run_nestful("m", "http://x", 1)
    assert log.started == []                                     # nothing ran, sandboxed or not
    entry = r["win_rate_metric"]
    assert r["win_rate"] is None and r["metrics_infra_required"] == ["win_rate"]
    assert entry["status"] == "infra_required" and entry["prerequisite"] == N._SANDBOX_PREREQ
    assert "bubblewrap (bwrap) is not installed" in entry["infra_required"]
    assert r["full_match_accuracy"] == 50.0 and r["accuracy"] == 50.0
    assert r["status"] == "completed_with_errors" and r["leaderboard_comparable"] is False
    assert "bubblewrap" in r["leaderboard_comparable_reason"]
    assert all("nestful_win" not in t for t in r["sample_traces"])


def test_a_sandbox_lost_mid_run_withholds_the_whole_win_rate(tmp_path, monkeypatch):
    _checkout(tmp_path, monkeypatch)

    class _LosesItsSandbox(_InProcWorker):
        calls = 0

        def score(self, pred, gold, tools):
            type(self).calls += 1
            if type(self).calls > 1:
                raise N.WinRateUnavailable("the sandbox did not come back", N._SANDBOX_PREREQ)
            return super().score(pred, gold, tools)
    monkeypatch.setattr(N, "_WinRateWorker", _LosesItsSandbox)
    both_right = lambda q: _ok(json.dumps(_OUT if q == "1+2?" else _ROWS[1]["output"]))
    with mock.patch.object(base, "_send_single_request", _endpoint(both_right)):
        r = N.run_nestful("m", "http://x", 1)
    assert _LosesItsSandbox.calls == 2
    assert r["win_rate"] is None and r["win_rate_killed_calls"] is None
    assert r["win_rate_metric"]["prerequisite"] == N._SANDBOX_PREREQ
    assert all("nestful_win" not in t for t in r["sample_traces"])
    assert r["full_match_accuracy"] == 100.0 and r["leaderboard_comparable"] is False


# --------------------------------------------------------------------------- #
# Requests that got no reply
# --------------------------------------------------------------------------- #
_METRICS = ("full_match_accuracy", "partial_match_accuracy", "f1_intent", "f1_slot", "p_intent",
            "r_intent", "p_slot", "r_slot", "parsing_errors", "win_rate", "accuracy")


def test_a_run_with_no_reply_withholds_every_metric(tmp_path, monkeypatch):
    _checkout(tmp_path, monkeypatch)
    monkeypatch.setattr(N, "_WinRateWorker", _InProcWorker)
    with mock.patch.object(base, "_send_single_request", _endpoint(lambda q: _FAILED)):
        r = N.run_nestful("m", "http://x", 1)
    for key in _METRICS:
        assert r[key] is None, key                                # withheld, never 0
    assert r["status"] == "failed"
    assert r["accuracy_withheld_reason"].startswith("nothing was scored")
    assert r["request_harness_errors"] == {"endpoint_failure": 2}
    assert r["leaderboard_comparable"] is False
    reason = r["leaderboard_comparable_reason"]
    assert "nothing was scored" in reason and "endpoint_failure 2" in reason


def test_requests_left_out_are_named_and_make_the_run_not_comparable(tmp_path, monkeypatch):
    _checkout(tmp_path, monkeypatch)
    monkeypatch.setattr(N, "_WinRateWorker", _InProcWorker)
    s2_fails = lambda q: _ok(json.dumps(_OUT)) if q == "1+2?" else _FAILED
    with mock.patch.object(base, "_send_single_request", _endpoint(s2_fails)):
        r = N.run_nestful("m", "http://x", 1)
    # every metric covers the one answered row
    assert r["full_match_accuracy"] == 100.0 and r["accuracy"] == 100.0
    assert r["win_rate"] == 100.0 and r["parsing_errors"] == 0
    assert r["status"] == "completed_with_errors"
    assert r["leaderboard_comparable"] is False
    reason = r["leaderboard_comparable_reason"]
    assert "1 of 2 requests got no reply" in reason and "endpoint_failure 1" in reason


def test_a_clean_run_names_no_harness_error(tmp_path, monkeypatch):
    _checkout(tmp_path, monkeypatch)
    monkeypatch.setattr(N, "_WinRateWorker", _InProcWorker)
    with mock.patch.object(base, "_send_single_request", _endpoint(_s1_right_s2_wrong)):
        r = N.run_nestful("m", "http://x", 1)
    assert r["leaderboard_comparable"] is True and r["leaderboard_comparable_reason"] is None


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #
def test_nestful_doc_describes_the_sandbox_and_the_withholding():
    doc = " ".join((REPO / "docs" / "evals" / "nestful.md").read_text().split())
    assert "the model cannot inject code" not in doc
    assert "bubblewrap container of its own, whatever `GBENCH_SANDBOX` says" in doc
    assert "`GBENCH_SANDBOX` does not apply" in doc
    assert "fa57c28a-e9d8-44fb-a358-a9404a557d5b" in doc and "1857 wins" in doc
    assert "never 0" in doc and "`win_rate_sandbox`" in doc
    assert "requests got no reply and are left out of every metric" in doc
