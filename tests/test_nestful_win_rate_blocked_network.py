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

"""nestful: a Win Rate call that fails a use of the network, which the worker's sandbox withholds,
is a harness error: marked, counted, named, left out of the Win Rate, never scored as a loss.

The sandbox tests start the real bubblewrap container, as the other nestful sandbox tests do; the
model endpoint is stubbed and nothing leaves the machine: the container has loopback alone and no
resolver, the names looked up end in `.invalid` (never resolvable) and the one address connected
to is 192.0.2.1 (TEST-NET-1, never routed).
"""
import errno
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import nestful as N

REPO = Path(__file__).resolve().parents[1]

_BASIC = (
    "def add(a, b):\n    return a + b\n\n"
    "def fetch(url):\n"
    "    import urllib.request\n"
    "    with urllib.request.urlopen(url, timeout=5) as response:\n"
    "        return response.read().decode()\n\n"
    "def fetch_or(url, fallback):\n"
    "    import urllib.request\n"
    "    try:\n"
    "        return urllib.request.urlopen(url, timeout=5).read().decode()\n"
    "    except Exception:\n"
    "        return fallback\n\n"
    "def reach(host, port):\n"
    "    import socket\n"
    "    s = socket.socket()\n"
    "    try:\n"
    "        return s.connect_ex((host, port))\n"
    "    finally:\n"
    "        s.close()\n\n"
    "def localhost_address():\n"
    "    import socket\n"
    "    return socket.getaddrinfo('localhost', 80)[0][4][0]\n\n"
    "def bind_localhost():\n"
    "    import socket\n"
    "    s = socket.socket()\n"
    "    try:\n"
    "        s.bind(('localhost', 0))\n"
    "        return 'bound'\n"
    "    finally:\n"
    "        s.close()\n")
_NAMES = ("add", "fetch", "fetch_or", "reach", "localhost_address", "bind_localhost")
_TOOLS = [{"name": n, "description": "", "parameters": {}, "output_parameters": {"result": {}}}
          for n in _NAMES]
_URL = "https://gbench-test.invalid/data"


def _func_dir(tmp_path) -> Path:
    fd = tmp_path / "NESTFUL" / "data_v2" / "executable_functions"
    fd.mkdir(parents=True)
    (fd / "basic_functions.py").write_text(_BASIC)
    (fd / "func_file_map.json").write_text("{}")
    return fd


def _one(name, **arguments):
    return [{"name": name, "label": "$var_1", "arguments": arguments}]


# --------------------------------------------------------------------------- #
# The worker notices a failed use of the network (real bubblewrap container)
# --------------------------------------------------------------------------- #
def test_worker_reports_a_losing_call_that_needed_the_network(tmp_path, monkeypatch):
    fd = _func_dir(tmp_path)
    started = []
    real = subprocess.Popen

    class _Recorded(real):
        def __init__(self, argv, *a, **k):
            started.append(list(argv))
            super().__init__(argv, *a, **k)
    monkeypatch.setattr(N.subprocess, "Popen", _Recorded)
    w = N._WinRateWorker(str(fd))
    try:
        argv = started[-1]
        assert os.path.basename(argv[0]) == "bwrap" and "--share-net" not in argv
        if "--ro-bind-data" in N._bwrap_options():
            i = argv.index("--ro-bind-data")
            assert argv[i + 2] == "/etc/hosts"

        def score(calls, gold):
            return w.score(calls, json.dumps(gold), _TOOLS)

        # a call that loses after a failed name lookup: the sandbox decided it
        assert score(_one("fetch", url=_URL), "page") is False
        assert w.last_blocked and any("getaddrinfo('gbench-test.invalid')" in u
                                      for u in w.last_blocked), w.last_blocked
        assert all(len(u) <= 300 for u in w.last_blocked) and len(w.last_blocked) <= 5
        # the next calls start clean: a win and a plain loss are the reply's own
        assert score(_one("add", a=1, b=2), 3) is True and w.last_blocked is None
        assert score(_one("add", a=1, b=2), 4) is False and w.last_blocked is None
        # a failed connection counts; the call still returns what it returned before
        assert score(_one("reach", host="192.0.2.1", port=9), 0) is False
        assert w.last_blocked and "connect_ex(('192.0.2.1', 9))" in w.last_blocked[0]
        assert score(_one("reach", host="192.0.2.1", port=9), errno.ENETUNREACH) is True
        assert w.last_blocked is None                    # a win keeps its win
        # a function that swallows the failure and wins keeps its win too
        assert score(_one("fetch_or", url=_URL, fallback="none"), "none") is True
        assert w.last_blocked is None
        # localhost resolves (the container's /etc/hosts), as on any host
        if "--ro-bind-data" in N._bwrap_options():
            assert score(_one("localhost_address"), "127.0.0.1") is True
            assert score(_one("bind_localhost"), "bound") is True
            assert w.last_blocked is None
        # an empty reply never reaches the worker and is a plain loss
        assert w.score([], json.dumps(3), _TOOLS) is False and w.last_blocked is None
        assert w.kills == 0
    finally:
        w.close()


def test_worker_program_carries_the_probe():
    src = N._worker_source()
    compile(src, "<nestful win rate worker>", "exec")
    assert "def _install_network_probe(" in src
    assert "_install_network_probe" in N._WORKER_FUNCTIONS
    main = src[src.index("def _win_worker_main("):]
    # installed before anything else runs in the worker, emptied before every request
    assert main.index("_install_network_probe()") < main.index("_sandbox_facts(")
    assert "del refused[:]" in main


# --------------------------------------------------------------------------- #
# The parent side of the verdict protocol (a scripted child, no sandbox needed)
# --------------------------------------------------------------------------- #
_SCRIPTED_CHILD = r'''
import json, sys
facts = {"pid": 2, "env": ["HOME", "PATH"], "root": ["dev", "tmp", "usr"], "proc": False,
         "run": False, "ifaces": ["lo"], "functions": True, "functions_writable": False}
sys.stdout.write("ready " + json.dumps(facts) + "\n"); sys.stdout.flush()
answers = iter(%r)
for _line in sys.stdin:
    sys.stdout.write(next(answers)); sys.stdout.flush()
'''


def test_parent_reads_the_blocked_verdict(tmp_path, monkeypatch):
    fd = str(_func_dir(tmp_path))
    answers = ['n ["getaddrinfo(\'x.invalid\'): gaierror: no name"]\n', "1\n", "n junk\n",
               "0\n", "garbage\n"]
    monkeypatch.setattr(N, "_sandbox_argv", lambda d, **_k: [
        sys.executable, "-I", "-c", _SCRIPTED_CHILD % (answers,)])
    monkeypatch.setattr(N, "_interpreter_dirs", lambda: [])
    w = N._WinRateWorker(fd)
    try:
        call = _one("add", a=1, b=2)
        assert w.score(call, "3", _TOOLS) is False
        assert w.last_blocked == ["getaddrinfo('x.invalid'): gaierror: no name"]
        assert w.score(call, "3", _TOOLS) is True and w.last_blocked is None
        assert w.score(call, "3", _TOOLS) is False
        assert w.last_blocked == ["a use of the network (the worker named none)"]
        assert w.score(call, "3", _TOOLS) is False and w.last_blocked is None
        assert w.kills == 0
        # anything else is a dead or stuck worker: killed, counted, never a harness error
        assert w.score(call, "3", _TOOLS) is False and w.last_blocked is None
        assert w.kills == 1
    finally:
        w.close()


def test_blocked_uses_is_bounded():
    uses = N._blocked_uses(json.dumps(["x" * 1000] * 9).encode())
    assert len(uses) == 5 and all(len(u) == 300 for u in uses)
    assert N._blocked_uses(b"[]") == ["a use of the network (the worker named none)"]
    assert N._blocked_uses(b'{"a": 1}') == ["a use of the network (the worker named none)"]


# --------------------------------------------------------------------------- #
# run_nestful's accounting
# --------------------------------------------------------------------------- #
def _row(sid, question, calls, gold):
    return {"sample_id": sid, "input": question, "tools": _TOOLS, "output": calls,
            "gold_answer": gold}


_ADD = _one("add", a=1, b=2)
_FETCH = _one("fetch", url=_URL)
_ROWS = [_row("s1", "1+2?", _ADD, 3), _row("s2", "fetch?", _FETCH, "page"),
         _row("s3", "2+2?", _one("add", a=2, b=2), 4)]


def _checkout(tmp_path, monkeypatch, rows=_ROWS) -> Path:
    fd = _func_dir(tmp_path)
    data = fd.parent / "nestful_data.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setitem(N._PINNED_BLOBS, "nestful_data.jsonl", N._git_blob_sha1(str(data)))
    monkeypatch.setattr(N, "_func_dir", lambda: str(fd))
    # the fetch rows' gold needs the network, as the pinned data's fa57c28a does
    monkeypatch.setattr(N, "_GOLD_NEEDS_NETWORK", frozenset({"s2", "f1"}))
    return fd


def _endpoint(answer):
    async def _reply(**kw):
        return base.Reply(answer(kw["messages"][-1]["content"]), None, "stop",
                          completion_tokens=5)
    return _reply


def _gold_calls(question):
    """Replies with each row's gold calls, except 2+2 answered with 1+2 (a loss)."""
    return json.dumps({"1+2?": _ADD, "fetch?": _FETCH, "2+2?": _ADD}[question])


class _StubWorker:
    """In-process Win Rate; a call of `fetch` is reported as a failed use of the network."""

    def __init__(self, func_dir, *a, **k):
        self.func_dir, self.kills, self.last_blocked = func_dir, 0, None

    def score(self, pred, gold, tools):
        self.last_blocked = None
        if pred and pred[0]["name"] == "fetch":
            self.last_blocked = ["getaddrinfo('gbench-test.invalid'): gaierror: no name"]
            return False
        return N._win_score(pred, gold, tools, self.func_dir)

    def close(self):
        pass


def test_blocked_rows_leave_the_win_rate_and_are_named(tmp_path, monkeypatch):
    _checkout(tmp_path, monkeypatch)
    monkeypatch.setattr(N, "_WinRateWorker", _StubWorker)
    with mock.patch.object(base, "_send_single_request", _endpoint(_gold_calls)):
        r = N.run_nestful("m", "http://x", 1)
    # s1 wins, s3 loses, s2 is a harness error: 1 of 2 scored calls
    assert r["win_rate"] == 50.0
    assert r["win_rate_network_blocked_calls"] == 1
    assert r["win_rate_network_blocked_samples"] == ["s2"]
    by_id = {json.loads(t["gold_answer"])["sample_id"]: t for t in r["sample_traces"]}
    assert "nestful_win" not in by_id["s2"]
    assert by_id["s2"]["nestful_win_blocked_by_sandbox"] == [
        "getaddrinfo('gbench-test.invalid'): gaierror: no name"]
    assert by_id["s1"]["nestful_win"] is True and by_id["s3"]["nestful_win"] is False
    assert "nestful_win_blocked_by_sandbox" not in by_id["s1"]
    # the other metrics execute nothing and keep every row
    assert r["full_match_accuracy"] == round(2 / 3 * 100, 2) and r["accuracy"] is not None
    assert "win_rate_withheld_reason" not in r
    assert r["status"] == "completed_with_errors"
    assert "1 Win Rate call(s) failed a use of the network" in r["status_reason"]
    assert r["leaderboard_comparable"] is False
    reason = r["leaderboard_comparable_reason"]
    assert "failed a use of the network" in reason and "s2" in reason
    assert "harness errors, left out of the Win Rate" in reason


def test_every_call_blocked_withholds_the_win_rate(tmp_path, monkeypatch):
    rows = [_row("f1", "fetch?", _FETCH, "page")]
    _checkout(tmp_path, monkeypatch, rows)
    monkeypatch.setattr(N, "_WinRateWorker", _StubWorker)
    with mock.patch.object(base, "_send_single_request", _endpoint(_gold_calls)):
        r = N.run_nestful("m", "http://x", 1)
    assert r["win_rate"] is None                                  # withheld, never 0
    assert r["win_rate_withheld_reason"].startswith("no Win Rate call was scored")
    assert r["win_rate_network_blocked_calls"] == 1
    assert r["full_match_accuracy"] == 100.0                      # still scored
    assert r["status"] == "completed_with_errors" and r["leaderboard_comparable"] is False


def test_a_clean_run_counts_no_blocked_call(tmp_path, monkeypatch):
    rows = [_row("s1", "1+2?", _ADD, 3), _row("s3", "2+2?", _one("add", a=2, b=2), 4)]
    _checkout(tmp_path, monkeypatch, rows)
    monkeypatch.setattr(N, "_WinRateWorker", _StubWorker)
    with mock.patch.object(base, "_send_single_request", _endpoint(_gold_calls)):
        r = N.run_nestful("m", "http://x", 1)
    assert r["win_rate"] == 50.0 and r["win_rate_network_blocked_calls"] == 0
    assert "win_rate_network_blocked_samples" not in r
    assert r["status"] == "success" and r["leaderboard_comparable"] is True
    assert all("nestful_win_blocked_by_sandbox" not in t for t in r["sample_traces"])


def test_a_sandbox_lost_after_a_blocked_row_clears_the_marks(tmp_path, monkeypatch):
    _checkout(tmp_path, monkeypatch)

    class _LosesItsSandbox(_StubWorker):
        calls = 0

        def score(self, pred, gold, tools):
            type(self).calls += 1
            if type(self).calls > 2:
                raise N.WinRateUnavailable("the sandbox did not come back", N._SANDBOX_PREREQ)
            return super().score(pred, gold, tools)
    monkeypatch.setattr(N, "_WinRateWorker", _LosesItsSandbox)
    with mock.patch.object(base, "_send_single_request", _endpoint(_gold_calls)):
        r = N.run_nestful("m", "http://x", 1)
    assert r["win_rate"] is None and r["win_rate_network_blocked_calls"] is None
    assert "win_rate_network_blocked_samples" not in r
    assert all("nestful_win_blocked_by_sandbox" not in t and "nestful_win" not in t
               for t in r["sample_traces"])
    assert r["win_rate_metric"]["prerequisite"] == N._SANDBOX_PREREQ


def test_run_with_the_real_sandbox_marks_the_network_call(tmp_path, monkeypatch):
    _checkout(tmp_path, monkeypatch)
    with mock.patch.object(base, "_send_single_request", _endpoint(_gold_calls)):
        r = N.run_nestful("m", "http://x", 1)
    assert r["win_rate"] == 50.0 and r["win_rate_network_blocked_calls"] == 1
    assert r["win_rate_network_blocked_samples"] == ["s2"]
    by_id = {json.loads(t["gold_answer"])["sample_id"]: t for t in r["sample_traces"]}
    assert any("gbench-test.invalid" in u
               for u in by_id["s2"]["nestful_win_blocked_by_sandbox"])
    assert r["win_rate_killed_calls"] == 0 and r["status"] == "completed_with_errors"
    assert "nestful_win_blocked_by_sandbox" in r["win_rate_sandbox"]["note"]


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #
def test_nestful_doc_describes_the_blocked_calls():
    doc = " ".join((REPO / "docs" / "evals" / "nestful.md").read_text().split())
    for phrase in ("`nestful_win_blocked_by_sandbox`", "`win_rate_network_blocked_calls`",
                   "`win_rate_network_blocked_samples`", "`win_rate_withheld_reason`",
                   "never scored as a loss", "names `localhost` alone",
                   "the worker cannot tell which", "A refused connection to loopback counts too",
                   "1856 wins, 4 losses, and that one sequence reported as a failed use of the "
                   "network"):
        assert phrase in doc, phrase
    assert "not by itself make a run not comparable" not in doc
    assert "got its verdict from the sandbox, not from the reply" not in doc
    # the module says the same, and the per-run note names the marks
    assert "the worker cannot tell which" in " ".join((N.__doc__ or "").split())
    for key in ("nestful_win_blocked_by_sandbox", "win_rate_network_blocked_calls",
                "--ro-bind-data"):
        assert key in N._SANDBOX_NETWORK_NOTE, key
