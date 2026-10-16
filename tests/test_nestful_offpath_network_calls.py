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

"""nestful: a losing Win Rate call that failed a use of the network is a harness error only on a
row whose pinned gold sequence needs the network; on any other row the reply went off the gold
path by its own choice, so the loss is scored and counted. The Win Rate worker is stubbed in
process and the model endpoint is faked; nothing leaves the machine and files go to tmp only."""

import json
from pathlib import Path
from unittest import mock

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import nestful as N

_TOOLS = [{"name": n, "description": "", "parameters": {}, "output_parameters": {"result": {}}}
          for n in ("add", "check_url_status")]
_BASIC = ("def add(a, b):\n    return a + b\n\n"
          "def check_url_status(url):\n    return False\n")


def _one(name, **arguments):
    return [{"name": name, "label": "$var_1", "arguments": arguments}]


_ADD = _one("add", a=1, b=2)
_CHECK = _one("check_url_status", url="https://gbench-test.invalid/")
_ROWS = [
    {"sample_id": "math-1", "input": "1+2?", "tools": _TOOLS, "output": _ADD, "gold_answer": 3},
    {"sample_id": "math-2", "input": "2+2?", "tools": _TOOLS,
     "output": _one("add", a=2, b=2), "gold_answer": 4},
    {"sample_id": "net-1", "input": "fetch?", "tools": _TOOLS, "output": _CHECK,
     "gold_answer": True},
]


def _checkout(tmp_path, monkeypatch) -> Path:
    fd = tmp_path / "NESTFUL" / "data_v2" / "executable_functions"
    fd.mkdir(parents=True)
    (fd / "basic_functions.py").write_text(_BASIC)
    (fd / "func_file_map.json").write_text("{}")
    data = fd.parent / "nestful_data.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in _ROWS))
    monkeypatch.setitem(N._PINNED_BLOBS, "nestful_data.jsonl", N._git_blob_sha1(str(data)))
    monkeypatch.setattr(N, "_func_dir", lambda: str(fd))
    monkeypatch.setattr(N, "_GOLD_NEEDS_NETWORK", frozenset({"net-1"}))
    return fd


class _StubWorker:
    """In-process Win Rate; any call of check_url_status is reported as a failed use of the
    network (as the sandbox, which has none, reports it)."""

    def __init__(self, func_dir, *a, **k):
        self.func_dir, self.kills, self.last_blocked = func_dir, 0, None

    def score(self, pred, gold, tools):
        self.last_blocked = None
        win = N._win_score(pred, gold, tools, self.func_dir)
        if not win and any(c.get("name") == "check_url_status" for c in pred or []):
            self.last_blocked = ["getaddrinfo('gbench-test.invalid'): gaierror: no name"]
        return win

    def close(self):
        pass


def _endpoint(replies):
    async def _reply(**kw):
        return base.Reply(json.dumps(replies[kw["messages"][-1]["content"]]), None, "stop",
                          completion_tokens=5)
    return _reply


def _run(tmp_path, monkeypatch, replies):
    _checkout(tmp_path, monkeypatch)
    monkeypatch.setattr(N, "_WinRateWorker", _StubWorker)
    with mock.patch.object(base, "_send_single_request", _endpoint(replies)):
        return N.run_nestful("m", "http://x", 1)


def test_an_off_path_network_call_is_the_replys_loss(tmp_path, monkeypatch):
    # math-2 answered with a network call its gold does not need: scored, a loss, counted
    r = _run(tmp_path, monkeypatch, {"1+2?": _ADD, "2+2?": _CHECK, "fetch?": _CHECK})
    by_id = {json.loads(t["gold_answer"])["sample_id"]: t for t in r["sample_traces"]}
    assert by_id["math-2"]["nestful_win"] is False
    assert by_id["math-2"]["nestful_win_network_blocked_offpath"] == [
        "getaddrinfo('gbench-test.invalid'): gaierror: no name"]
    assert "nestful_win_blocked_by_sandbox" not in by_id["math-2"]
    assert r["win_rate_network_blocked_offpath"] == 1
    assert r["win_rate_network_blocked_offpath_samples"] == ["math-2"]
    # net-1's gold needs the network: its blocked call stays a harness error
    assert "nestful_win" not in by_id["net-1"]
    assert r["win_rate_network_blocked_calls"] == 1
    assert r["win_rate_network_blocked_samples"] == ["net-1"]
    # math-1 wins, math-2 loses: 1 of 2 scored calls
    assert r["win_rate"] == 50.0
    assert r["status"] == "completed_with_errors"


def test_a_run_with_only_off_path_network_calls_stays_clean(tmp_path, monkeypatch):
    rows_replies = {"1+2?": _ADD, "2+2?": _CHECK, "fetch?": [{"name": "add", "label": "$v",
                                                                "arguments": {"a": 0, "b": 0}}]}
    r = _run(tmp_path, monkeypatch, rows_replies)
    assert r["win_rate_network_blocked_calls"] == 0
    assert r["win_rate_network_blocked_offpath"] == 1
    assert r["win_rate"] == round(1 / 3 * 100, 2)
    assert "failed a use of the network" not in (r.get("status_reason") or "")
    assert "failed a use of the network" not in (r.get("leaderboard_comparable_reason") or "")


def test_the_pinned_set_names_the_one_network_gold_sequence():
    assert N._GOLD_NEEDS_NETWORK == frozenset({"fa57c28a-e9d8-44fb-a358-a9404a557d5b"})
    assert "nestful_win_network_blocked_offpath" in N._SANDBOX_NETWORK_NOTE
