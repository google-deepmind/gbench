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

"""Every SWE agent's model client waits for the relay, not for a hidden 600 s.

2026-10-10 smoke run: SWE-agent-for-eval's litellm 1.87 fell back to a 600 s client timeout (its
request_timeout is the package default 6000, which completion_timeout.py replaces by
COMPLETION_HTTP_FALLBACK_SECONDS) and the OpenAI SDK made 2 more attempts, so a turn needing more
than 600 s of decode was abandoned and sampled again (12 times on one trimesh instance, 1800 s per
SWE-agent attempt, x20 attempts). mini-swe-agent 2.x (litellm 1.102), the swe_bench_pro sandbox
agent and MSWE-agent (openai 1.70) had the same 600 s. Each path now gets the relay's limit
(base.request_timeout_s of the max_tokens sent, else of the served window) plus 60 s, with no
transport retries where the client lets gbench set them, and the result records it
(`model_client_timeout`). GBENCH_AGENT_MAX_TOKENS_FROM_RUN (default off) sends the run's
--max-output-tokens with every agent request, recorded as a deviation.

No network, no docker: containers, relays and rollouts are faked.
"""

import json
import os
import re
import types

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import multi_swe_bench as M
from gbench.runners.eval_suites import swe_agentic as A
from gbench.runners.eval_suites import swe_bench_live as LV
from gbench.runners.eval_suites import swe_bench_pro as P
from gbench.runners.eval_suites import swebench_common as C


@pytest.fixture(autouse=True)
def _upstream_max_tokens(monkeypatch):
    """These tests check the scaffolds' upstream protocol, which sends no max_tokens: that is
    the GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0 path since the run's --max-output-tokens became the
    default for agent suites (2026-10-11; tests/test_agent_max_tokens_from_run.py)."""
    monkeypatch.setenv("GBENCH_AGENT_MAX_TOKENS_FROM_RUN", "0")


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCKER_DIR = os.path.join(REPO, "gbench", "docker")


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    monkeypatch.delenv(A.AGENT_MAX_TOKENS_FROM_RUN_ENV, raising=False)
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    monkeypatch.setattr(C, "require_task_network_blocked", lambda *a, **k: None)
    for suite in ("SWE_BENCH_LIVE", "SWE_BENCH_PRO", "MULTI_SWE_BENCH", "COPILOT_BENCH_SWE"):
        monkeypatch.setenv(f"GBENCH_{suite}_TRAJ_DIR", str(tmp_path / suite.lower()))
        monkeypatch.delenv(f"GBENCH_{suite}_TEMPERATURE", raising=False)


def _proc(rc=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


class _FakeRelay:
    """Records how it was made; answers like an EndpointRelay that saw nothing."""
    made = []

    def __init__(self, upstream, bind_host, key=None, body_fields=None, name="r",
                 timeout_s=None, **kw):
        self.body_fields, self.TIMEOUT_S = dict(body_fields or {}), timeout_s or 3600
        self.kw = dict(kw)
        self.allowed_paths, self.upstream = ("/v1/chat/completions",), upstream
        _FakeRelay.made.append(self)

    def start(self):
        return self

    def stop(self):
        pass

    @property
    def url(self):
        return "http://127.0.0.1:4242/v1"

    def record(self):
        return {"requests_relayed": 0, "upstream_errors": {}, "requests_refused": 0}


# --------------------------------------------------------------------------------------------
# sizing
# --------------------------------------------------------------------------------------------
def test_the_timeout_is_sized_like_every_base_request():
    t = A.agent_request_timeouts()
    assert t.tokens == base.UNCAPPED_TIMEOUT_TOKENS
    assert t.relay_s == base.request_timeout_s(base.UNCAPPED_TIMEOUT_TOKENS)
    # the client outlasts the relay's own safety cap on a model request (limit + 120 s)
    assert t.relay_cap_s == t.relay_s + C.EndpointRelay.TOTAL_CAP_MARGIN_S
    assert t.client_s == t.relay_cap_s + 60 == t.relay_s + 180 > 600
    assert A.client_timeout_for_relay_limit(t.relay_s) == t.client_s
    base._RUN_KNOBS["max_model_len"] = 131072
    t = A.agent_request_timeouts()
    assert (t.tokens, t.relay_s) == (131072, base.request_timeout_s(131072))
    assert "served context window (131072 tokens)" in t.source
    t = A.agent_request_timeouts(65536)
    assert (t.tokens, t.relay_s, t.client_s) == (65536, base.request_timeout_s(65536),
                                                 base.request_timeout_s(65536) + 180)
    assert A.litellm_client_kwargs(t) == {"timeout": t.client_s, "max_retries": 0}


def test_the_max_tokens_option_is_on_unless_turned_off(monkeypatch):
    base._RUN_KNOBS["max_output_tokens"] = 65536
    monkeypatch.delenv(A.AGENT_MAX_TOKENS_FROM_RUN_ENV, raising=False)
    assert A.agent_max_tokens_from_run() == 65536     # the default (the user's decision)
    monkeypatch.setenv(A.AGENT_MAX_TOKENS_FROM_RUN_ENV, "1")
    assert A.agent_max_tokens_from_run() == 65536
    assert A.agent_max_tokens_from_run(32768) == 32768
    assert A.with_run_max_tokens({"temperature": 1.0}, 65536) == {"temperature": 1.0,
                                                                  "max_tokens": 65536}
    assert A.with_run_max_tokens({"temperature": 1.0}, None) == {"temperature": 1.0}
    monkeypatch.setenv(A.AGENT_MAX_TOKENS_FROM_RUN_ENV, "0")
    assert A.agent_max_tokens_from_run() is None


def test_the_recorded_client_versions_are_the_pinned_ones():
    def pin(name, pkg):
        text = open(os.path.join(DOCKER_DIR, name), encoding="utf-8").read()
        return re.search(rf"^{pkg}==(\S+)$", text, re.M).group(1)
    assert pin("sweagent_for_eval.constraints.txt", "litellm") == LV.SWEAGENT_LITELLM_VERSION
    assert pin("mini_swe_agent_v2.constraints.txt", "litellm") == A.AGENT_V2_LITELLM_VERSION
    assert pin("mswe_agent.constraints.txt", "openai") == M.MSWE_AGENT_OPENAI_VERSION


# --------------------------------------------------------------------------------------------
# swe_bench_live: SWE-agent-for-eval (litellm 1.87) and Win-Agent
# --------------------------------------------------------------------------------------------
def test_the_run_batch_command_carries_the_timeout_and_no_retries():
    cmd = LV.sweagent_command("/o/i.jsonl", "/o", "m", "http://127.0.0.1:1/v1", 2, "n",
                              client_timeout_s=9452)
    i = cmd.index("--agent.model.completion_kwargs")
    assert json.loads(cmd[i + 1]) == {"timeout": 9452, "max_retries": 0}
    assert cmd.index("--agent.model.api_base") < i < cmd.index("--instances.type")


def test_the_batch_takes_its_timeout_and_stall_limit_from_the_relay(monkeypatch, tmp_path):
    out = str(tmp_path / "out")
    os.makedirs(out)
    monkeypatch.setattr(LV, "host_pull_images",
                        lambda images, workers=1: {i: {"state": "present"} for i in images})
    seen = {}

    def _run(cmd, **k):
        if "run-batch" in cmd:
            seen["cmd"] = list(cmd)
        return _proc()
    monkeypatch.setattr(LV.subprocess, "run", _run)
    watches = []
    real = LV.SweagentWatch

    def _watch(*a, **k):
        w = real(*a, **k)
        watches.append(w)
        return w
    monkeypatch.setattr(LV, "SweagentWatch", _watch)
    relay = _FakeRelay("http://h/v1", "127.0.0.1", timeout_s=33968)
    iid = "o__r-1"
    rows = [{"instance_id": iid, "problem_statement": "p", "base_commit": "c", "repo": "o/r"}]
    LV.run_sweagent([A.AgentTask(iid, "p", LV.python_instance_image(iid), "o/r")], rows, "m",
                    relay.url, 1, out, relay=relay)
    cmd = seen["cmd"]
    kw = json.loads(cmd[cmd.index("--agent.model.completion_kwargs") + 1])
    assert kw == {"timeout": 33968 + 180, "max_retries": 0}
    # the watchdog never ends a rollout that waits for one honest model request, and keeps the
    # command-sized limit for every other silence
    assert watches[0].model_stall_s == LV.sweagent_stall_s(33968 + 180) == 33968 + 180 + 15 * 60
    assert watches[0].stall_s == LV.SWEAGENT_STALL_S == 6300
    assert watches[0].record()["model_request_limit_s"] == 33968 + 180
    rec = json.load(open(os.path.join(out, "watchdog.json")))
    assert (rec["stall_limit_s"], rec["model_request_stall_limit_s"]) == (6300, 33968 + 180 + 900)


def _sweagent_tab(monkeypatch, kwargs=None):
    _FakeRelay.made.clear()
    monkeypatch.setattr(LV, "EndpointRelay", _FakeRelay)
    got = {}

    def _rollouts(tasks, rows, model_name, relay_url, concurrency, out_dir, relay=None):
        got["relay"] = relay
        return [{"sample_idx": 0, "source_sample_idx": 0, "category": tasks[0].category,
                 "agent_patch": "d", "response_text": "d",
                 "extra_payload": {"instance_id": tasks[0].instance_id},
                 "agent_exit_status": "submitted", "agent_infra_error": False,
                 "agent_error": None, "agent_steps": 3, "truncated_turns": None,
                 "is_correct": False, "status": "OK"}]
    monkeypatch.setattr(LV, "run_sweagent", _rollouts)

    async def _scorer(traces):
        for t in traces:
            t["is_correct"] = True
    rows = [{"instance_id": "o__r-0", "problem_statement": "p", "base_commit": "c",
             "repo": "o/r"}]
    res = LV._sweagent_tab("lite", rows, _scorer, {}, lambda r, m: [], "m",
                           "http://127.0.0.1:9/v1", 1, dict(kwargs or {}, enable_thinking=True))
    return res, _FakeRelay.made[-1], got


def test_the_sweagent_tab_sizes_the_relay_and_records_the_client(monkeypatch):
    base._RUN_KNOBS["max_model_len"] = 262144
    base._RUN_KNOBS["max_output_tokens"] = 65536
    res, relay, got = _sweagent_tab(monkeypatch, {"max_output_tokens": 65536})
    t = A.agent_request_timeouts()
    assert relay.TIMEOUT_S == t.relay_s and got["relay"] is relay
    assert "max_tokens" not in relay.body_fields         # the fork sends none; neither does gbench
    rec = res["model_client_timeout"]
    assert rec["timeout_s"] == t.client_s and rec["retries"] == 0
    assert rec["version"] == "1.87.0" and "completion_kwargs" in rec["client"]
    assert rec["hidden_cap_removed"] is True and "600 s" in rec["replaces"]
    assert res["agent"]["relay_timeout_s"] == t.relay_s
    assert res["max_output_tokens_per_turn"] is None
    assert "agent_max_tokens_from_run" not in res["official_sampling"]


def test_the_max_tokens_option_reaches_the_relay_and_is_a_deviation(monkeypatch):
    monkeypatch.setenv(A.AGENT_MAX_TOKENS_FROM_RUN_ENV, "1")
    base._RUN_KNOBS["max_output_tokens"] = 65536
    res, relay, _ = _sweagent_tab(monkeypatch, {"max_output_tokens": 65536})
    assert relay.body_fields["max_tokens"] == 65536
    assert relay.body_fields["chat_template_kwargs"] == {"enable_thinking": True}
    assert relay.TIMEOUT_S == base.request_timeout_s(65536)
    assert res["model_client_timeout"]["timeout_s"] == base.request_timeout_s(65536) + 180
    assert res["max_output_tokens_per_turn"] == 65536
    assert res["official_sampling"]["agent_max_tokens_from_run"]["max_tokens_sent"] == 65536
    assert res["leaderboard_comparable"] is False
    assert "max_tokens=65536 sent with every agent request" in res["leaderboard_comparable_reason"]


def test_win_agent_gets_request_timeout_in_its_own_environment(monkeypatch):
    monkeypatch.delenv("REQUEST_TIMEOUT", raising=False)
    env = LV.win_agent_env(9452)
    assert env["REQUEST_TIMEOUT"] == "9452" and env["PYTHONUTF8"] == "1"
    assert "REQUEST_TIMEOUT" not in os.environ
    # litellm reads exactly 6000 as "unset" and falls back to 600 s
    assert LV.win_agent_request_timeout(6000) == "6001"


# --------------------------------------------------------------------------------------------
# mini-swe-agent: 2.x container (copilot_bench_swe, swe_bench_multilingual) and 1.x in-process
# --------------------------------------------------------------------------------------------
def test_the_config_builders_put_the_client_kwargs_in_model_kwargs():
    ck = {"timeout": 34028, "max_retries": 0}
    ov = A.build_agent_v2_overrides({}, model_name="m", base_url="http://h:1", temperature=1.0,
                                    thinking=True, max_output_tokens=None, client_kwargs=ck)
    assert {k: ov["model"]["model_kwargs"][k] for k in ck} == ck
    cfg = A.build_agent_config({"model": {}}, model_name="m", base_url="http://h:1",
                               temperature=1.0, thinking=True, max_output_tokens=None,
                               client_kwargs=ck)
    assert {k: cfg["model"]["model_kwargs"][k] for k in ck} == ck


def _v2_run(monkeypatch, **kwargs):
    _FakeRelay.made.clear()
    monkeypatch.setattr(A, "EndpointRelay", _FakeRelay)
    monkeypatch.setattr(A, "describe_agent_v2", lambda v: {
        "config": {"model": {"model_kwargs": {"drop_params": True}}, "agent": {}},
        "config_sha256": A.AGENT_V2_CONFIG_SHA256["2.4.6"], "version": "2.4.6"})
    monkeypatch.setattr(A, "require_rollout_network_blocked", lambda *a, **k: None)
    monkeypatch.setattr(A, "bridge_gateway", lambda: ("172.17.0.1", ""))
    monkeypatch.setattr(A, "require_relay_reachable", lambda *a, **k: None)
    seen = {}

    def _rollouts(tasks, cfg, concurrency, out_dir, eval_name="swe", rollout=None):
        seen["cfg"] = cfg
        return [{"extra_payload": {"instance_id": t.instance_id}, "agent_infra_error": False,
                 "agent_exit_status": "Submitted", "agent_steps": 1, "category": t.category,
                 "agent_patch": "d", "is_correct": False, "status": "OK", "sample_idx": i}
                for i, t in enumerate(tasks)]
    monkeypatch.setattr(A, "run_rollouts", _rollouts)

    async def _score(traces):
        for t in traces:
            t["is_correct"] = True
    tasks = [A.AgentTask("a__a-1", "p", "img", "a/a")]
    res = A.run_agentic("copilot_bench_swe", "m", "http://127.0.0.1:9/v1", 1, tasks, _score, {},
                        enable_thinking=True, agent_v2="2.4.6", **kwargs)
    return res, _FakeRelay.made[-1], seen["cfg"]["model"]["model_kwargs"]


def test_the_v2_rollouts_wait_for_the_relay(monkeypatch):
    base._RUN_KNOBS["max_model_len"] = 262144
    res, relay, kw = _v2_run(monkeypatch, max_output_tokens=65536)
    t = A.agent_request_timeouts()
    assert relay.TIMEOUT_S == t.relay_s
    assert (kw["timeout"], kw["max_retries"]) == (t.client_s, 0)
    assert kw["api_base"] == relay.url and "max_tokens" not in kw
    rec = res["model_client_timeout"]
    assert (rec["timeout_s"], rec["retries"], rec["version"]) == (t.client_s, 0, "1.102.1")
    assert res["agent"]["relay_timeout_s"] == t.relay_s
    assert res["max_output_tokens_per_turn"] is None


def test_the_v2_max_tokens_option_uses_the_drivers_capped_turn(monkeypatch):
    monkeypatch.setenv(A.AGENT_MAX_TOKENS_FROM_RUN_ENV, "1")
    res, relay, kw = _v2_run(monkeypatch, max_output_tokens=65536)
    # model_kwargs, where the driver's EndpointModel resends a turn refused only for the cap,
    # with max_tokens=None: so the timeouts are sized for the window, not the cap
    t = A.agent_request_timeouts(None)
    assert kw["max_tokens"] == 65536 and kw["timeout"] == t.client_s
    assert relay.TIMEOUT_S == t.relay_s
    assert "sized for the window, not the per-turn cap" in res["model_client_timeout"]["source"]
    assert res["official_sampling"]["agent_max_tokens_from_run"]["max_tokens_sent"] == 65536
    assert "max_tokens=65536 sent with every agent request" in res["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------------------------
# swe_bench_pro: mini-swe-agent 2.4.6 in each Harbor sandbox
# --------------------------------------------------------------------------------------------
def test_the_locked_config_carries_the_client_kwargs():
    text = "model:\n  model_kwargs:\n    drop_params: true\n    reasoning_effort: high\n"
    cfg = P.build_locked_agent_config(text, thinking=True, temperature=1.0,
                                      client_kwargs={"timeout": 34028, "max_retries": 0})
    kw = cfg["model"]["model_kwargs"]
    assert (kw["timeout"], kw["max_retries"], kw["reasoning_effort"]) == (34028, 0, "high")


def test_the_pro_run_sizes_relay_and_sandbox_client(monkeypatch, tmp_path):
    base._RUN_KNOBS["max_model_len"] = 262144
    _FakeRelay.made.clear()
    monkeypatch.setattr(P, "EndpointRelay", _FakeRelay)
    monkeypatch.setattr(P, "endpoint_api_key", lambda: None)
    monkeypatch.setattr(P, "require_sandbox_network_blocked", lambda: None)
    monkeypatch.setattr(P, "load_v2_rows", lambda: [])
    monkeypatch.setattr(P, "load_hard51_ids", lambda v2=None: set())
    monkeypatch.setattr(P, "read_locked_agent_config",
                        lambda: ("model:\n  model_kwargs:\n    reasoning_effort: high\n",
                                 P.AGENT_CONFIG_SHA256))
    res = P.run_swe_bench_pro_v2("m", "http://h:1/v1", 1, enable_thinking=True)
    t = A.agent_request_timeouts()
    assert _FakeRelay.made[-1].TIMEOUT_S == t.relay_s
    import yaml
    written = yaml.safe_load(open(os.path.join(res["agent_output_dir"],
                                               "mini_toolcall.gbench.yaml")))
    kw = written["model"]["model_kwargs"]
    assert (kw["timeout"], kw["max_retries"]) == (t.client_s, 0)
    assert res["agent"]["gbench_config_additions"]["model.model_kwargs.timeout"] == t.client_s
    assert res["model_client_timeout"]["timeout_s"] == t.client_s
    assert res["model_client_timeout"]["retries"] == 0


def test_the_key_forwarder_takes_the_timeout_and_body_fields():
    fwd = P.KeyForwarder("http://h:1/v1", "k", "127.0.0.1", body_fields={"max_tokens": 7},
                         timeout_s=9392)
    assert fwd.TIMEOUT_S == 9392 and fwd._body_fields == {"max_tokens": 7}


# --------------------------------------------------------------------------------------------
# multi_swe_bench: MSWE-agent's OpenAI client (driver 4)
# --------------------------------------------------------------------------------------------
def test_mswe_spec_and_relay_use_the_runs_timeout(monkeypatch, tmp_path):
    base._RUN_KNOBS["max_model_len"] = 262144
    t = M.mswe_agent_timeouts()
    assert t == A.agent_request_timeouts()
    rows = {"o__r-1": {"org": "o", "repo": "r", "number": 1, "instance_id": "o__r-1",
                       "resolved_issues": [{"title": "t", "body": "b"}]}}
    tasks = [A.AgentTask("o__r-1", "t", M.published_image("o", "r", 1), "go",
                         meta={"org": "o", "repo": "r", "number": 1})]
    monkeypatch.setattr(M, "host_pull_images", lambda images, workers=1: {
        i: {"state": "present"} for i in images})
    monkeypatch.setattr(M.swebench_common, "require_task_network_blocked", lambda *a, **k: None)
    specs = []

    def _run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            work = cmd[cmd.index("-v", cmd.index("-v") + 1) + 1].split(":")[0]
            specs.append(json.load(open(os.path.join(work, "spec.json"))))
        return _proc()
    monkeypatch.setattr(M.subprocess, "run", _run)
    M.run_mswe_agent(tasks, rows, "m", "http://127.0.0.1:5/v1", 1, str(tmp_path / "o"), 1.0, 0.95)
    assert specs[0]["client_timeout_s"] == t.client_s == t.relay_s + 180


# --------------------------------------------------------------------------------------------
# 2026-10-11 review: the client outlasts the relay's safety cap; the relays flag a close at the
# client timeout; the watchdog keeps the command-sized limit outside a model request
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("window", [None, 262144, 131072, 8192])
def test_the_client_outlasts_the_real_relays_safety_cap(window):
    if window:
        base._RUN_KNOBS["max_model_len"] = window
    for sent in (None, 4096, 65536):
        t = A.agent_request_timeouts(sent)
        relay = C.EndpointRelay("http://127.0.0.1:9/v1", "127.0.0.1", timeout_s=t.relay_s)
        body = json.dumps({"messages": [], **({"max_tokens": sent} if sent else {})}).encode()
        cap = relay.request_total_s("/v1/chat/completions", body)
        assert cap == t.relay_cap_s < t.client_s, (window, sent)
        assert relay.request_total_s("/v1/models", None) < t.client_s


def test_a_request_the_relay_still_waits_for_is_answered_by_the_relay(monkeypatch):
    """Time-scaled: margins of 1 s (relay cap = limit + 1, client = cap + 1), an upstream that
    never answers and has no /metrics. The relay ends the request at its cap and answers; the
    client, waiting client_s, gets that answer (a counted endpoint failure), not a timeout of
    its own (a silent client close the scaffold would resample)."""
    import http.server
    import socket
    import threading
    import time
    import urllib.error
    import urllib.request
    monkeypatch.setattr(C.EndpointRelay, "TOTAL_CAP_MARGIN_S", 1)
    monkeypatch.setattr(A, "AGENT_CLIENT_MARGIN_S", 1)
    monkeypatch.setattr(base, "request_timeout_s", lambda tokens: 1)
    release = threading.Event()

    class _Silent(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            release.wait(30)

        def do_GET(self):
            self.send_error(404)

        def log_message(self, *a):
            pass
    up = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Silent)
    up.daemon_threads = True
    threading.Thread(target=up.serve_forever, daemon=True).start()
    t = A.agent_request_timeouts()
    assert (t.relay_s, t.relay_cap_s, t.client_s) == (1, 2, 3)
    relay = C.EndpointRelay(f"http://127.0.0.1:{up.server_address[1]}/v1", "127.0.0.1",
                            timeout_s=t.relay_s, client_timeout_suspect_s=t.suspect_s()).start()
    try:
        req = urllib.request.Request(relay.url + "/chat/completions", method="POST",
                                     data=json.dumps({"messages": []}).encode(),
                                     headers={"Content-Type": "application/json"})
        started = time.monotonic()
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req, timeout=t.client_s)
        took = time.monotonic() - started
        assert err.value.code in (502, 504) and took < t.client_s
        assert not isinstance(err.value.reason, socket.timeout)
        time.sleep(0.2)
        assert sum(relay.upstream_errors.values()) == 1
        assert relay.client_closed_streams == 0
    finally:
        release.set()
        relay.stop()
        up.shutdown()


def test_a_relay_flags_a_close_at_the_client_timeout():
    t = A.agent_request_timeouts()
    relay = C.EndpointRelay("http://127.0.0.1:9/v1", "127.0.0.1", timeout_s=t.relay_s,
                            client_timeout_suspect_s=t.suspect_s())
    assert set(C.EndpointRelay.CLIENT_TIMEOUT_SUSPECT_S) < set(t.suspect_s())
    relay.count_client_closed(float(t.client_s) - 2)
    assert relay.client_timeouts_suspected == 1
    assert float(t.client_s) in relay.client_close_record()["suspect_timeouts_s"]


def test_every_relay_flags_its_clients_timeout(monkeypatch, tmp_path):
    base._RUN_KNOBS["max_model_len"] = 262144
    t = A.agent_request_timeouts()
    _, relay, _ = _sweagent_tab(monkeypatch)
    assert relay.kw["client_timeout_suspect_s"] == t.suspect_s()
    _, relay, _ = _v2_run(monkeypatch)
    assert relay.kw["client_timeout_suspect_s"] == t.suspect_s()
    fwd = P.KeyForwarder("http://h:1/v1", "k", "127.0.0.1", timeout_s=t.relay_s,
                         client_timeout_suspect_s=t.suspect_s())
    assert float(t.client_s) in fwd._suspect_s
    test_the_pro_run_sizes_relay_and_sandbox_client(monkeypatch, tmp_path)
    assert _FakeRelay.made[-1].kw["client_timeout_suspect_s"] == t.suspect_s()


def test_the_mswe_relay_flags_its_clients_timeout(monkeypatch, tmp_path):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_swe_reference_scaffolds as H
    H._agentic_setup(monkeypatch, tmp_path, H._report((), (1, 2)))
    made = []

    class _Rec(C.EndpointRelay):
        def __init__(self, *a, **k):
            made.append(k)
            super().__init__(*a, **k)
    monkeypatch.setattr(M, "EndpointRelay", _Rec)
    res = M.run_multi_swe_bench("m", "http://127.0.0.1:1/v1", 1, enable_thinking=True)
    t = M.mswe_agent_timeouts()
    assert made[-1]["client_timeout_suspect_s"] == t.suspect_s()
    assert res["agent"]["openai_client_timeout_s"] == t.client_s == t.relay_cap_s + 60


_TRACE_HEAD = (
    "2026-10-10 19:33:26,898 - DEBUG - swea-tools - Retrieved state from environment: {}\n"
    "2026-10-10 19:33:26,900 - INFO - swea-agent - \U0001f916 MODEL INPUT\n"
    "2026-10-10 19:33:26,906 - INFO - swea-agent - ===== STEP 36 =====\n")
_TRACE_QUERY = "2026-10-10 19:36:52,562 - DEBUG - swea-lm - n_cache_control: 2\n"
_TRACE_RETRY = ("2026-10-10 20:06:54,282 - WARNING - swea-lm - Retrying LM query: attempt 1 "
                "(slept for 10.00s) due to Timeout: litellm.Timeout\n")
_TRACE_REPLY = ("2026-10-10 19:33:28,303 - DEBUG - swea-lm - Response: ModelResponse(id='x',\n"
                "  choices=[...])\n"
                "2026-10-10 19:33:28,310 - DEBUG - swea-lm - Error calculating cost: ...\n")
_TRACE_COMMAND = ("2026-10-10 19:33:28,315 - TRACE - swea-env - Input:\nsleep 99999\n")


@pytest.mark.parametrize("text,inflight", [
    (_TRACE_HEAD, False),
    (_TRACE_HEAD + _TRACE_QUERY, True),
    (_TRACE_HEAD + _TRACE_QUERY + "2026-10-10 19:36:52,600 - WARNING - swea-lm - No max input "
     "tokens found for model\n", True),
    (_TRACE_HEAD + _TRACE_QUERY + _TRACE_REPLY, False),
    (_TRACE_HEAD + _TRACE_QUERY + _TRACE_RETRY, True),
    (_TRACE_HEAD + _TRACE_QUERY + _TRACE_REPLY + _TRACE_COMMAND, False),
    (_TRACE_HEAD + _TRACE_QUERY.replace("swea-lm", "swea-lm-o__r-1"), True),
    (_TRACE_HEAD + _TRACE_QUERY + "2026-10-10 19:36:53,000 - ERROR - swea-agent - Exiting\n",
     False),
])
def test_the_watchdog_reads_a_model_request_in_flight(tmp_path, text, inflight):
    path = tmp_path / "i.trace.log"
    path.write_text(text, encoding="utf-8")
    assert LV.sweagent_model_request_inflight(str(path)) is inflight
    assert LV.sweagent_model_request_inflight(str(tmp_path / "missing.log")) is False


def _watched(tmp_path, monkeypatch, trace):
    out = str(tmp_path / "out")
    iid = "o__r-1"
    os.makedirs(os.path.join(out, iid))
    with open(os.path.join(out, iid, f"{iid}.trace.log"), "w", encoding="utf-8") as f:
        f.write(trace)
    flagged = []
    monkeypatch.setattr(LV, "remove_batch_task_containers",
                        lambda container, image=None: flagged.append(image) or [])
    stopped = []
    monkeypatch.setattr(LV.SweagentWatch, "stop", lambda self, reason: stopped.append(reason))
    client = A.client_timeout_for_relay_limit(33968)
    watch = LV.SweagentWatch(out, {iid: LV.python_instance_image(iid)}, "b",
                             request_timeout_s=client, model_stall_s=LV.sweagent_stall_s(client))
    last = os.stat(os.path.join(out, iid)).st_mtime
    for root, dirs, files in os.walk(out):
        for name in dirs + files:
            os.utime(os.path.join(root, name), (last, last))
    os.utime(out, (last, last))
    return watch, iid, last, stopped


def test_a_silent_command_or_runtime_is_still_found_after_6300_s(tmp_path, monkeypatch):
    for trace in (_TRACE_HEAD + _TRACE_QUERY + _TRACE_REPLY + _TRACE_COMMAND, _TRACE_HEAD):
        sub = tmp_path / str(len(trace))
        watch, iid, last, stopped = _watched(sub, monkeypatch, trace)
        assert watch.stall_s == 6300 and watch.model_stall_s == 33968 + 180 + 900
        watch.check(now=last + 6300 - 5)
        assert iid not in watch.stalled
        watch.check(now=last + 6300 + 5)
        assert iid in watch.stalled and "a command ends at 5400 s" in watch.stalled[iid]
        watch.check(now=last + 6300 + 600 + 5)
        assert stopped and "6300 s, plus 600 s" in stopped[0]


def test_a_rollout_waiting_for_its_model_request_gets_the_long_limit(tmp_path, monkeypatch):
    watch, iid, last, stopped = _watched(tmp_path, monkeypatch, _TRACE_HEAD + _TRACE_QUERY)
    watch.check(now=last + 6300 + 600 + 5)
    assert iid not in watch.stalled and not stopped
    watch.check(now=last + watch.model_stall_s - 5)
    assert iid not in watch.stalled
    watch.check(now=last + watch.model_stall_s + 5)
    assert iid in watch.stalled and "its model request may take" in watch.stalled[iid]
    rec = watch.record()
    assert (rec["stall_limit_s"], rec["model_request_stall_limit_s"]) == (6300, 35048)
    assert rec["model_request_batch_stop_limit_s"] == 35048 + 600


def test_the_watchdog_defaults_are_unchanged(tmp_path):
    watch = LV.SweagentWatch(str(tmp_path), {}, "b")
    assert (watch.stall_s, watch.model_stall_s) == (6300, 6300)

