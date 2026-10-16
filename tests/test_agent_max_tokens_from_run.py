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

"""Agent suites send the run's --max-output-tokens on every request that carries none.

The user's decision of 2026-10-11: without a per-reply bound one runaway agent turn runs to the
served window (about 36 min at 120 tok/s, and an instance can have several), which no 4-hour
smoke and validation run can absorb. The value is the run's --max-output-tokens, never a fixed
number ("don't hardcode on 64k"), so these tests run at 12,345. GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0
turns it off; non-agent suites and requests that carry their own budget are unchanged.

No server: the relay's body patching and the result stamp are exercised directly.
"""

import json
import types

import pytest

from gbench.runners import evals as E
from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import swe_agentic as A
from gbench.runners.eval_suites.swebench_common import EndpointRelay

PATH = "/v1/chat/completions"
CT = "application/json"


@pytest.fixture
def run(monkeypatch):
    monkeypatch.delenv(A.AGENT_MAX_TOKENS_FROM_RUN_ENV, raising=False)
    monkeypatch.setitem(base._RUN_KNOBS, "max_output_tokens", 12345)

    def suite(name):
        monkeypatch.setitem(base._RUN_KNOBS, "suite", name)
    return suite


def _sent(relay, body):
    return json.loads(relay.patch_body(PATH, json.dumps(body).encode(), CT))


def test_an_agent_suite_request_without_a_budget_gets_the_runs_value(run):
    run("swe_bench_live")
    relay = EndpointRelay("http://127.0.0.1:9/v1", "127.0.0.1")
    assert _sent(relay, {"messages": []})["max_tokens"] == 12345
    assert relay.record()["max_tokens_from_run"] == {"value": 12345, "requests_filled": 1}


def test_a_request_with_its_own_budget_keeps_it(run):
    run("terminal_bench")
    relay = EndpointRelay("http://127.0.0.1:9/v1", "127.0.0.1")
    assert _sent(relay, {"max_tokens": 77})["max_tokens"] == 77
    assert _sent(relay, {"max_completion_tokens": 88}).get("max_tokens") is None
    assert relay.record()["max_tokens_from_run"]["requests_filled"] == 0


def test_a_non_agent_suite_is_unchanged(run):
    run("gsm8k")
    relay = EndpointRelay("http://127.0.0.1:9/v1", "127.0.0.1")
    assert "max_tokens" not in _sent(relay, {"messages": [], "temperature": 1.0})
    assert "max_tokens_from_run" not in relay.record()


def test_the_switch_turns_it_off(run, monkeypatch):
    run("swe_bench_pro")
    monkeypatch.setenv(A.AGENT_MAX_TOKENS_FROM_RUN_ENV, "0")
    relay = EndpointRelay("http://127.0.0.1:9/v1", "127.0.0.1")
    assert "max_tokens" not in _sent(relay, {"messages": []})


def test_the_value_follows_the_run_knob(run, monkeypatch):
    run("tau2")
    monkeypatch.setitem(base._RUN_KNOBS, "max_output_tokens", 4096)
    assert A.agent_max_tokens_for_suite() == 4096
    monkeypatch.setitem(base._RUN_KNOBS, "max_output_tokens", None)
    assert A.agent_max_tokens_for_suite() is None


def test_the_result_records_the_deviation():
    res = {"max_tokens_sent": None, "leaderboard_comparable": True,
           "relay": {"max_tokens_from_run": {"value": 12345, "requests_filled": 7}}}
    E._stamp_agent_max_tokens(res)
    assert res["max_tokens_sent"] == 12345
    assert res["agent_max_tokens_from_run"]["requests_filled"] == 7
    assert res["leaderboard_comparable"] is False
    assert "max_tokens=12345 (the run's --max-output-tokens)" in res["leaderboard_comparable_reason"]
    untouched = {"max_tokens_sent": None, "relay": {"max_tokens_from_run": {
        "value": 12345, "requests_filled": 0}}}
    E._stamp_agent_max_tokens(untouched)
    assert "agent_max_tokens_from_run" not in untouched


def test_agent_dojo_puts_the_value_on_requests_without_one():
    from gbench.runners.eval_suites import agent_dojo as D
    seen = []
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
        create=lambda *a, **k: seen.append(k))))
    D._send_temperature(client, 1.0, True, None, max_tokens=12345)
    client.chat.completions.create(messages=[])
    client.chat.completions.create(messages=[], max_tokens=5)
    assert [k.get("max_tokens") for k in seen] == [12345, 5]


def test_a_suites_full_canonical_run_reason_is_replaced_when_the_cap_was_sent():
    """2026-10-14 docs audit: terminal_bench's default run read "full canonical run; max_tokens=..."
    while being marked not comparable; the suite's own verdict no longer holds with the cap."""
    res = {"max_tokens_sent": None, "leaderboard_comparable": True,
           "leaderboard_comparable_reason": "full canonical run",
           "relay": {"max_tokens_from_run": {"value": 4096, "requests_filled": 3}}}
    E._stamp_agent_max_tokens(res)
    assert "full canonical run" not in res["leaderboard_comparable_reason"]
    assert res["leaderboard_comparable_reason"].startswith("max_tokens=4096")
