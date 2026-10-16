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
"""Sharding, empty shards, the endpoint key, attempts and the harness-error evidence of the
agentic and harness suites (tau2/tau3, bfcl_v4_agentic, agent_dojo, complexfuncbench, the SWE
family, gaia2, wildclawbench, mcp_bench, mcp_atlas).

* --shard is applied to each suite's full, ordered item list before the --eval-limit draw, the
  result records it, a shard with no item reports no score (accuracy null, never 0 or an error),
  and the suites whose headline is not a correct / scored fraction merge their shards
  themselves (merge_shards).
* swe_bench_pro: a budget cut is a harness error only on evidence the model cannot produce from
  its sandbox (endpoint-side failures of requests carrying its own conversation, no reply after).
* gaia2: a --thinking reply whose native tool call the server's parser deleted after the thought
  is recovered from its token ids and handed to ARE (the model's output), and every reply's
  generation text is kept.
* The endpoint key is base.endpoint_api_key (VLLM_API_KEY, else OPENAI_API_KEY) everywhere.

Hermetic: docker, the endpoint, Gemini and the network are stubbed (loopback stubs only); files
go under tmp_path.
"""

import asyncio
import http.server
import json
import re
import socket
import socketserver
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from gbench.runners.eval_suites import agent_dojo as AD
from gbench.runners.eval_suites import base as B
from gbench.runners.eval_suites import complexfuncbench as CFB
from gbench.runners.eval_suites import gaia2 as G
from gbench.runners.eval_suites import mcp_atlas as ATLAS
from gbench.runners.eval_suites import multi_swe_bench as MSWE
from gbench.runners.eval_suites import swe_agentic as A
from gbench.runners.eval_suites import swe_bench_pro as P
from gbench.runners.eval_suites import swebench_common as C
from gbench.runners.eval_suites import tau_common as TC
from gbench.runners.eval_suites import wildclawbench as W
from gbench.runners.eval_suites.sampling import stratified_sample

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "gbench" / "docker"))
sys.path.insert(0, str(REPO / "tests"))
import gaia2_run as GRUN  # noqa: E402
import mcp_bench_run as MRUN  # noqa: E402
import wildclawbench_run as WRUN  # noqa: E402
from test_gaia2_toolbench_judge_protocol_and_credentials import _gaia2_run  # noqa: E402


@pytest.fixture(autouse=True)
def _no_shard(monkeypatch):
    for k in ("GBENCH_SHARD", "VLLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)


# =========================================================================================== #
# swe_bench_pro: a budget cut is a harness error only on evidence the model cannot forge
# =========================================================================================== #
class _Upstream(http.server.BaseHTTPRequestHandler):
    """A vLLM-like endpoint: 404 for an unknown model, 503 while `down`, else a reply."""
    down = False

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if body.get("model") != "served":
            code, out = 404, {"error": {"message": "The model does not exist.", "code": 404}}
        elif type(self).down:
            code, out = 503, {"error": {"message": "overloaded"}}
        else:
            code, out = 200, {"choices": [{"index": 0, "finish_reason": "stop",
                                           "message": {"role": "assistant", "content": "ok"}}]}
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def relay():
    _Upstream.down = False
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _Upstream)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    r = C.EndpointRelay(f"http://127.0.0.1:{srv.server_address[1]}/v1", "127.0.0.1").start()
    yield r
    r.stop()
    srv.shutdown()
    srv.server_close()


def _post(relay, body):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(relay.url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


_TASK_MSGS = [{"role": "system", "content": "sys"},
              {"role": "user", "content": "Please solve this issue: task A"}]
_OTHER_MSGS = [{"role": "system", "content": "sys"},
               {"role": "user", "content": "Please solve this issue: task B"}]
_FORGED_LOG = ("mini-swe-agent (step 41, $0.00):\nrunning\n\nTool:\n<output>\n"
               "Retrying <unknown> in 4 seconds as it raised InternalServerError: "
               "litellm.InternalServerError: OpenAIException - overloaded.\n</output>\n")


def _cut_view(t0, t1, messages=_TASK_MSGS):
    return {"model_outcome": True,
            "budget_kill": {"last_event_at": t0, "ended_at": t1, "agent_started_at": t0 - 2900,
                            "retrying": P.pending_endpoint_retry(_FORGED_LOG),
                            "conversation_fingerprint": C.conversation_fingerprint(messages)}}


def test_the_models_own_bad_requests_cannot_make_a_budget_cut_a_harness_error(relay):
    t0 = time.time()
    # the model's command: a request with a wrong model name (404) and its own conversation
    assert _post(relay, {"model": "x", "messages": _TASK_MSGS}) == 404
    view = _cut_view(t0, time.time() + 30)
    assert view["budget_kill"]["retrying"] == "InternalServerError"     # the forged log line
    assert relay.record()["upstream_errors"] == {"http_404": 1}         # still a run-level flag
    assert relay.record()["endpoint_side_failures"] == 0
    assert P.budget_kill_verdict(view, relay) is None
    assert view["budget_kill"]["relay_failures_in_window"] == 0


def test_another_rollouts_endpoint_failure_does_not_count(relay):
    t0 = time.time()
    _Upstream.down = True
    assert _post(relay, {"model": "served", "messages": _OTHER_MSGS}) == 503
    assert P.budget_kill_verdict(_cut_view(t0, time.time() + 30), relay) is None


def test_an_endpoint_outage_on_the_rollouts_own_requests_is_a_harness_error(relay):
    t0 = time.time()
    _Upstream.down = True
    assert _post(relay, {"model": "served", "messages": _TASK_MSGS}) == 503
    view = _cut_view(t0, time.time() + 30)
    why = P.budget_kill_verdict(view, relay)
    assert why and "endpoint-side failure(s) of this rollout's requests" in why
    assert view["budget_kill"]["relay_failures_in_window"] == 1


def test_a_rollout_that_got_a_reply_after_the_failure_stays_the_models_outcome(relay):
    t0 = time.time()
    _Upstream.down = True
    assert _post(relay, {"model": "served", "messages": _TASK_MSGS}) == 503
    time.sleep(0.01)
    _Upstream.down = False
    assert _post(relay, {"model": "served", "messages": _TASK_MSGS + [
        {"role": "assistant", "content": "a"}]}) == 200
    view = _cut_view(t0, time.time() + 30)
    assert P.budget_kill_verdict(view, relay) is None
    assert view["budget_kill"]["recovered_after_failure"] is True


def test_conversation_fingerprint_reads_the_first_user_message():
    fp = C.conversation_fingerprint(_TASK_MSGS)
    assert fp == C.conversation_fingerprint(_TASK_MSGS + [{"role": "user", "content": "later"}])
    assert fp == C.conversation_fingerprint(
        [{"role": "user", "content": [{"type": "text", "text": "Please solve this issue: "},
                                      {"type": "text", "text": "task A"}]}])
    assert fp != C.conversation_fingerprint(_OTHER_MSGS)
    assert C.conversation_fingerprint([{"role": "system", "content": "x"}]) is None
    assert C.endpoint_side_failure("http_503") and C.endpoint_side_failure("connection")
    assert not C.endpoint_side_failure("http_404") and not C.endpoint_side_failure("http_400")


# =========================================================================================== #
# --shard before --eval-limit in the SWE one-shot paths and mcp_atlas
# =========================================================================================== #
def _swe_rows(n=12):
    return [{"instance_id": f"o__r{i % 3}-{i}", "repo": f"o/r{i % 3}",
             "problem_statement": f"issue {i}"} for i in range(n)]


def _fake_send(**kw):
    async def go():
        return B.Reply("diff --git a/x b/x\n", None, "stop", None, None, 3)
    return go()


def _execute_swebench(monkeypatch, limit):
    rows = _swe_rows()
    monkeypatch.setattr(B, "_send_single_request", _fake_send)
    monkeypatch.setattr(C, "_pinned_snapshot_rows", lambda *a, **k: ("snap", rows))

    def scorer(*a, **k):
        async def _score(traces):
            for t in traces:
                t["is_correct"] = False
                t["status"] = "OK"
        return _score
    monkeypatch.setattr(C, "make_swebench_scorer", scorer)
    return C.execute_swebench("swe_bench_multilingual", "m", "http://stub/v1", 2, "d/d", "test",
                              None, dataset_revision="abc", limit=limit)


def test_the_swe_one_shot_limit_is_drawn_from_the_shard(monkeypatch):
    rows = _swe_rows()
    monkeypatch.setenv("GBENCH_SHARD", "2/3")
    res = _execute_swebench(monkeypatch, 2)
    shard_ids = {r["instance_id"] for r in rows[1::3]}
    got = {t["extra_payload"]["instance_id"] for t in res["sample_traces"]}
    assert len(got) == 2 and got <= shard_ids and res["shard"] == "2/3"


def test_an_unsharded_limited_swe_run_keeps_the_instances_it_always_drew(monkeypatch):
    rows = _swe_rows()
    res = _execute_swebench(monkeypatch, 4)
    want = {r["instance_id"] for r in stratified_sample(
        C.unique_instance_rows(rows), 4, lambda r: r.get("repo"), seed="swebench_common")}
    assert {t["extra_payload"]["instance_id"] for t in res["sample_traces"]} == want


def test_an_empty_swe_shard_is_an_honest_empty_row(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "13/13")
    res = _execute_swebench(monkeypatch, None)
    assert res["status"] == "success" and res["accuracy"] is None and res["empty_shard"]
    assert "empty shard 13/13" in res["accuracy_withheld_reason"]
    assert res["leaderboard_comparable"] is False


def test_the_swe_scorer_does_not_start_the_harness_without_an_instance(monkeypatch):
    started = []
    monkeypatch.setattr(C.subprocess, "run", lambda *a, **k: started.append(a))
    metrics = {}
    score = C.make_swebench_scorer("x", "m", "d", "test", None, 1, metrics)
    traces = [{"extra_payload": {}, "response_text": "diff"}]
    asyncio.run(score(traces))
    assert started == [] and traces[0]["harness_error"] == C.NO_REPORT
    assert traces[0]["scoring_excluded"] and "swebench_report" not in metrics


def test_swe_bench_pro_one_shot_shards_before_its_limit(monkeypatch):
    rows = [dict(r, repo_language="go") for r in _swe_rows()]
    monkeypatch.setattr(P, "load_v2_rows", lambda: rows)
    monkeypatch.setenv("GBENCH_SHARD", "1/4")
    samples = P._load_swe_bench_pro_samples(limit=2)
    ids = {s[2]["instance_id"] for s in samples}
    assert len(ids) == 2 and ids <= {r["instance_id"] for r in rows[0::4]}
    laid = C.presharded(samples)
    assert B.apply_shard(laid, "x")[0] == samples       # base keeps exactly them


def test_multi_swe_bench_hands_every_instance_to_base(monkeypatch):
    seen = {}
    monkeypatch.setattr(MSWE, "check_multi_swe_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(MSWE.swe_agentic, "resolve_mode", lambda n: (A.ONESHOT, "env"))
    monkeypatch.setattr(MSWE, "_load_multi_swe_bench_samples",
                        lambda limit=None: seen.update(loader_limit=limit) or [])
    monkeypatch.setattr(MSWE, "run_eval_suite", lambda **kw: seen.update(base=kw) or {
        "sample_traces": [], "total_questions": 0, "status": "success"})
    MSWE.run_multi_swe_bench("m", "http://stub/v1", 1, limit=5)
    # the loader draws nothing; base applies --shard, then --eval-limit 5 with the loader's seed
    assert seen["loader_limit"] is None and seen["base"]["limit"] == 5
    assert seen["base"].get("limit_seed") in (None, "multi_swe_bench")


def test_mcp_atlas_draws_its_limit_within_the_shard(monkeypatch):
    rows = [{"TASK": f"t{i}", "PROMPT": f"p{i}", "ENABLED_TOOLS": "[]",
             "GTFA_CLAIMS": json.dumps([f"c{i}"])} for i in range(10)]
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **k: rows
    monkeypatch.setitem(sys.modules, "datasets", fake)
    monkeypatch.setattr(ATLAS, "extract_claims", lambda claims: [claims])
    monkeypatch.setenv("GBENCH_SHARD", "2/5")
    samples = ATLAS._load_mcp_atlas_samples(limit=1, catalog=[])
    assert [s[2]["task_id"] for s in samples] in (["t1"], ["t6"])


# =========================================================================================== #
# swe agentic: an empty shard reports no score
# =========================================================================================== #
def test_an_empty_agentic_swe_shard_reports_no_score(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "agent_output_dir", lambda name: str(tmp_path))
    monkeypatch.setenv("GBENCH_SHARD", "2/3")
    tasks = [A.AgentTask("a__a-1", "issue", A.swebench_image("a__a-1"), "org/repo")]

    async def scorer(traces):
        raise AssertionError("nothing to score")
    res = A.run_agentic("copilot_bench_swe", "m", "http://127.0.0.1:9/v1", 1, tasks, scorer, {})
    assert res["status"] == "success" and res["accuracy"] is None and res["empty_shard"]
    assert res["accuracy_withheld_reason"].startswith("empty shard 2/3: no items on this node")
    assert res["full_set_items"] == 1 and res["shard"] == "2/3"


# =========================================================================================== #
# tau2 / tau3
# =========================================================================================== #
def _fake_tau2(monkeypatch, ids_by_domain):
    sim = types.ModuleType("tau2.data_model.simulation")
    sim.TextRunConfig = types.SimpleNamespace(
        model_fields={"task_split_name": types.SimpleNamespace(default="base")})
    helpers = types.ModuleType("tau2.runner.helpers")
    helpers.get_tasks = lambda task_set_name, task_split_name: [
        types.SimpleNamespace(id=i) for i in ids_by_domain[task_set_name]]
    for name in ("tau2", "tau2.data_model", "tau2.runner"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "tau2.data_model.simulation", sim)
    monkeypatch.setitem(sys.modules, "tau2.runner.helpers", helpers)


def test_tau_shards_each_domain_in_tau2s_order_before_its_limit(monkeypatch):
    from gbench.runners.eval_suites.sampling import stratified_sample
    ids = [str(i) for i in range(10)]
    _fake_tau2(monkeypatch, {"retail": ids})
    got = [TC._shard_task_ids("retail", f"{i}/3", None)[0] for i in (1, 2, 3)]
    assert sorted(sum(got, [])) == sorted(ids) and not set(got[0]) & set(got[1])
    # the limit is a seeded draw from the shard (seed: the suite name), not tau2's head
    assert TC._shard_task_ids("retail", "2/3", 2) == (
        stratified_sample(["1", "4", "7"], 2, seed="tau2"), 10)


def test_a_tau_shard_skips_an_empty_domain_and_an_empty_shard_has_no_score(monkeypatch,
                                                                           tmp_path):
    ran = []

    def one(domain, *a, task_ids=None, **k):
        ran.append((domain, list(task_ids)))
        TC._DOMAIN_PASS_HAT_K[domain] = {"pass^1": 0.5}
        TC._DOMAIN_TASK_OUTCOMES[domain] = {t: [1, 1] for t in task_ids}
        return len(task_ids), 1.0, 1, 0
    monkeypatch.setattr(TC, "_run_one_domain", one)
    monkeypatch.setattr(TC, "_shard_task_ids", lambda d, s, lim, **kw: (
        (["x1"], 5) if d == "retail" else ([], 1)))
    res = TC._run_domains("tau2", ["retail", "airline"], "m", "http://h/v1", 1, None, "d",
                          False, 4, str(tmp_path), {}, "h", False, shard="2/5")
    assert ran == [("retail", ["x1"])] and res["empty_shard_domains"] == ["airline"]
    assert res["accuracy"] == 50.0 and res["shard"] == "2/5"
    assert "holds no task of airline" in res["leaderboard_comparable_reason"]
    monkeypatch.setattr(TC, "_shard_task_ids", lambda d, s, lim, **kw: ([], 3))
    res = TC._run_domains("tau2", ["retail", "airline"], "m", "http://h/v1", 1, None, "d",
                          False, 4, str(tmp_path), {}, "h", False, shard="5/5")
    assert res["accuracy"] is None and res["empty_shard"] and res["status"] == "success"
    assert res["accuracy_withheld_reason"].startswith("empty shard 5/5")


def test_tau_merge_recomputes_pass_k_as_tau2(monkeypatch):
    def shard(label, outcomes):
        return {"eval_name": "tau2", "shard": label, "status": "success",
                "domains_requested": ["retail", "airline"],
                "leaderboard_comparable_reason": B.SHARD_REASON.format(
                    label=label, index=int(label[0]), count=2) + "; user simulator x",
                "tau2_report": {"task_outcomes_by_domain": outcomes}}
    a = shard("1/2", {"retail": {"1": [4, 4], "2": [4, 1]}, "airline": {"a": [4, 2]}})
    b = shard("2/2", {"retail": {"3": [4, 0]}, "airline": {"b": [4, 4]}})
    merged = TC.merge_tau_shards([a, b])
    import math
    retail = [(4, 4), (4, 1), (4, 0)]
    airline = [(4, 2), (4, 4)]

    def p(rows, k):
        return sum(math.comb(c, k) / math.comb(n, k) for n, c in rows) / len(rows)
    assert merged["pass_k_by_domain"]["retail"]["pass^2"] == round(100 * p(retail, 2), 2)
    want = round((100 * p(retail, 1) + 100 * p(airline, 1)) / 2, 2)
    assert merged["accuracy"] == want and merged["pass_k"]["pass^1"] == want
    assert merged["leaderboard_comparable_reason"] == "user simulator x"


def test_tau_trials_follow_attempt_count_never_below_the_leaderboards_4(monkeypatch):
    monkeypatch.delenv("GBENCH_TAU2_NUM_TRIALS", raising=False)
    monkeypatch.delenv("TAU2_NUM_TRIALS", raising=False)
    B.set_run_knobs(attempt_count=2)
    try:
        trials, rec = TC._resolve_trials()
        assert trials == 4 and rec["attempts_per_sample"] == 4
        B.set_run_knobs(attempt_count=6)
        assert TC._resolve_trials()[0] == 6
        monkeypatch.setenv("GBENCH_TAU2_NUM_TRIALS", "3")
        assert TC._resolve_trials()[0] == 3
        B.set_run_knobs(attempt_count=1)
        assert TC._resolve_trials()[1] is None
    finally:
        B.set_run_knobs(attempt_count=None)


def test_a_malformed_shard_stops_tau_before_anything_starts(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "0/4")
    monkeypatch.setattr(TC, "check_role_providers",
                        lambda: pytest.fail("started before the shard was validated"))
    with pytest.raises(ValueError):
        TC.run_tau_suite("tau2", ["retail"], "m", "http://h/v1", 1, None, "d")


def test_the_tau_endpoint_key_is_the_one_every_pillar_sends(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "vllm-key-000011112222")
    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")
    assert TC._real_endpoint_key() == "vllm-key-000011112222"
    env = TC._launch_env()
    assert env[TC._AGENT_KEY_ENV] == "EMPTY" and "VLLM_API_KEY" not in env


# =========================================================================================== #
# agent_dojo
# =========================================================================================== #
def _dojo():
    class Suite:
        def __init__(self, users, injs):
            self.user_tasks = {u: types.SimpleNamespace(ID=u) for u in users}
            self.injection_tasks = {j: types.SimpleNamespace(ID=j) for j in injs}

        def get_user_task_by_id(self, task_id):
            return self.user_tasks[task_id]

        def get_injection_task_by_id(self, task_id):
            return self.injection_tasks[task_id]

    class Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    runs = []

    def without(suite, pipe, task, ld, force, benchmark_version=None):
        runs.append(("without", task.ID))
        return task.ID.endswith("0"), True

    def with_(suite, pipe, task, attack, ld, force, injection_tasks=None,
              benchmark_version=None):
        util = {(task.ID, j): True for j in injection_tasks}
        sec = {(task.ID, j): j.endswith("0") for j in injection_tasks}
        return util, sec
    clients = []

    class Pipe:
        def __init__(self, elements):
            self.name = None
    return runs, clients, {
        "openai": types.SimpleNamespace(OpenAI=lambda **k: clients.append(k) or object()),
        "OpenAILLM": lambda client, model, **k: clients.append({"model": model}) or object(),
        "AgentPipeline": Pipe, "SystemMessage": lambda *a: None, "InitQuery": lambda: None,
        "ToolsExecutionLoop": lambda *a: None, "ToolsExecutor": lambda: None,
        "load_system_message": lambda *a: "sys", "load_attack": lambda *a: None,
        "OutputLogger": lambda *a: Ctx(), "run_without": without, "run_with": with_,
        "get_suites": lambda v: {"banking": Suite(["u0", "u1", "u2"], ["i0", "i1"]),
                                 "slack": Suite(["s0", "s1"], ["i0"])},
        "version": AD.AGENTDOJO_VERSION}


def test_agent_dojo_shards_user_tasks_and_merges_them_back(monkeypatch):
    results = []
    for i in (1, 2):
        runs, clients, fake = _dojo()
        monkeypatch.setattr(AD, "_load_agentdojo", lambda f=fake: f)
        monkeypatch.setenv("GBENCH_SHARD", f"{i}/2")
        results.append(AD.run_agent_dojo("served-id", "http://h/v1", concurrency=1))
    a, b = results
    assert a["shard"] == "1/2" and b["shard"] == "2/2"
    keys_a, keys_b = set(a["item_results"]["benign"]), set(b["item_results"]["benign"])
    assert not keys_a & keys_b and len(keys_a | keys_b) == 5
    merged = AD.merge_shards(results)
    monkeypatch.delenv("GBENCH_SHARD")
    runs, clients, fake = _dojo()
    monkeypatch.setattr(AD, "_load_agentdojo", lambda: fake)
    full = AD.run_agent_dojo("served-id", "http://h/v1", concurrency=1)
    for k in ("accuracy", "utility_under_attack", "attack_success_rate", "total_questions"):
        assert merged[k] == full[k], k


def test_an_empty_agent_dojo_shard_and_the_served_id_and_key(monkeypatch):
    runs, clients, fake = _dojo()
    monkeypatch.setattr(AD, "_load_agentdojo", lambda: fake)
    monkeypatch.setenv("VLLM_API_KEY", "vllm-key-000011112222")
    monkeypatch.setenv("GBENCH_SHARD", "7/7")
    res = AD.run_agent_dojo("served-id", "http://h/v1", concurrency=1, attempt_count=3)
    assert res["accuracy"] is None and res["empty_shard"] and runs == []
    assert res["request_model_id"] == "served-id"
    assert res["attempts"]["attempts_per_sample"] == 1
    assert {"model": "served-id"} in clients
    assert any(c.get("api_key") == "vllm-key-000011112222" for c in clients)
    assert "sequential" in res["execution"]


# =========================================================================================== #
# complexfuncbench
# =========================================================================================== #
def _cfb_rows(n=20):
    doms = ["Car-Rental", "Hotels", "Attraction", "Flights", "Cross"]
    return [{"id": f"{doms[i % 5]}-{i}"} for i in range(n)]


def test_complexfuncbench_shards_before_its_limit(monkeypatch, tmp_path):
    path = tmp_path / "cfb.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in _cfb_rows()))
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda **k: str(path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(CFB, "_sha256", lambda p: CFB.HF_SHA256)
    monkeypatch.setenv("GBENCH_SHARD", "3/4")
    got = CFB._load_dataset(3)
    shard_ids = {r["id"] for r in _cfb_rows()[2::4]}
    assert len(got) == 3 and {r["id"] for r in got} <= shard_ids


def test_complexfuncbench_merge_recomputes_complexeval(monkeypatch):
    def trace(i, ok, calls=(1, 1), harness=None):
        return {"id": f"Hotels-{i}", "sample_key": f"id:Hotels-{i}",
                "message": "Success." if ok else "fail", "harness_error": harness,
                "count_dict": {"correct_call_num": calls[0], "total_call_num": calls[1]},
                "resp_eval": {"complete": {"score": 2}, "correct": {"score": 1}}}
    a = {"status": "success", "sample_traces": [trace(1, True), trace(2, False, (0, 2))],
         "leaderboard_comparable_reason": "the two judge roles run on the Gemini API"}
    b = {"status": "success", "sample_traces": [trace(3, True), trace(4, False, harness="exception")]}
    m = CFB.merge_shards([a, b])
    assert m["n_scored"] == 3 and m["correct_answers"] == 2
    assert m["accuracy"] == round(2 / 3 * 100, 2) and m["call_accuracy"] == 50.0
    assert m["n_harness_errors"] == 1 and m["leaderboard_comparable"] is False


def test_an_empty_complexfuncbench_shard(monkeypatch):
    res = CFB._empty_shard_result("m", False, "9/9")
    assert res["accuracy"] is None and res["empty_shard"] and res["shard"] == "9/9"


# =========================================================================================== #
# bfcl_v4_agentic
# =========================================================================================== #
def test_bfcl_registers_a_served_model_with_no_vendor_identity_and_its_own_handler():
    pytest.importorskip("bfcl_eval", reason="canonical BFCL harness not installed")
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from gbench.runners.eval_suites import bfcl_v4_agentic as BF
    name = "someorg/some-model-7b"
    try:
        BF._register_model(name)
        cfg = MODEL_CONFIG_MAPPING[name]
        text = json.dumps({k: str(getattr(cfg, k)) for k in ("url", "org", "license",
                                                            "display_name")}).lower()
        assert "gemma" not in text and "google" not in text
        assert cfg.org == "someorg" and cfg.license == "unknown" and cfg.url == ""
    finally:
        MODEL_CONFIG_MAPPING.pop(name, None)
    upstream_key = next(iter(MODEL_CONFIG_MAPPING))
    saved = MODEL_CONFIG_MAPPING[upstream_key]
    try:
        BF._register_model(upstream_key)
        assert MODEL_CONFIG_MAPPING[upstream_key].model_handler.__name__ == \
            "_GbenchOpenAICompletionsHandler"
    finally:
        MODEL_CONFIG_MAPPING[upstream_key] = saved


def test_bfcl_id_selection_shards_each_category_before_the_limit(monkeypatch):
    pytest.importorskip("bfcl_eval", reason="canonical BFCL harness not installed")
    from gbench.runners.eval_suites import bfcl_v4_agentic as BF
    cats = ["web_search_base"]
    full = BF._select_limited_ids(cats, 0, shard="1/1")["web_search_base"]
    parts = [BF._select_limited_ids(cats, 0, shard=f"{i}/3")["web_search_base"]
             for i in (1, 2, 3)]
    assert sorted(sum(parts, [])) == sorted(full)
    assert not set(parts[0]) & set(parts[1])
    limited = BF._select_limited_ids(cats, 2, shard="2/3")["web_search_base"]
    assert len(limited) == 2 and set(limited) <= set(parts[1])


def test_the_bfcl_endpoint_key_is_the_one_every_pillar_sends(monkeypatch):
    from gbench.runners.eval_suites import bfcl_v4_agentic as BF
    monkeypatch.setenv("VLLM_API_KEY", "vllm-key-000011112222")
    assert BF.endpoint_key() == "vllm-key-000011112222"
    assert BF._without_keys({"OPENAI_API_KEY": "short1"})["OPENAI_API_KEY"] == \
        BF.ENDPOINT_KEY_PLACEHOLDER


# =========================================================================================== #
# gaia2
# =========================================================================================== #
_RESIDUE_CASES = [
    dict(content=None, reasoning="I should list contacts", thinking=True,
         generation_text="I should list contacts<|end|><|tool_call>call:list{}<tool_call|>"),
    dict(content="Let me look.", reasoning=None, thinking=False,
         generation_text="Let me look.<|tool_call>call:f{}<tool_call|>"),
    dict(content="answer", reasoning=None, thinking=False, generation_text="answer<|end|>"),
    dict(content=None, reasoning="r", thinking=True, generation_text="x r"),
    dict(content="c", reasoning="r", thinking=True, generation_text=None),
]


@pytest.mark.parametrize("case", _RESIDUE_CASES)
def test_the_launchers_copy_of_the_residue_check_equals_base(case):
    got = GRUN.server_parser_residue(tool_calls=None, **case)
    want = B.server_parser_residue(tool_calls=None, **case)
    assert tuple(got) == tuple(want)


def test_a_thinking_reply_whose_tool_call_the_server_deleted_goes_to_are():
    # reproduction: a streamed --thinking reply, reasoning then a native tool call
    info = {"completion_tokens": 30, "finish_reason": "stop", "stop_reason": 50,
            "content_chars": 0, "reasoning_chars": 22}
    ids = [1, 2, 3, 4, 50]
    table = {(1, 2, 3, 4): "I should list contacts<|channel>"
                           "<|tool_call>call:list_contacts{}<tool_call|>"}
    text, event, source, generation = GRUN.apply_server_parser_policy(
        None, "I should list contacts", info, True, ids, lambda xs: table.get(tuple(xs)))
    assert event == GRUN.SERVER_PARSER_DROPPED_TEXT and source == "token_ids"
    assert text == "<|channel><|tool_call>call:list_contacts{}<tool_call|>"
    assert generation.startswith("I should list contacts")
    # nothing after the thought: the model's outcome, but the generation is still kept
    table = {(1, 2, 3, 4): "I should list contacts"}
    text, event, source, generation = GRUN.apply_server_parser_policy(
        None, "I should list contacts", info, True, ids, lambda xs: table.get(tuple(xs)))
    assert event is None and text is None and generation == "I should list contacts"


def test_gaia2_shard_ids_are_disjoint_and_complete():
    ids = [f"s{i}" for i in range(11)] + ["s3"]
    parts = [GRUN.shard_scenario_ids(ids, (i, 4)) for i in (1, 2, 3, 4)]
    assert sorted(sum(parts, [])) == sorted(set(ids))
    assert all(not set(a) & set(b) for a in parts for b in parts if a is not b)
    with pytest.raises(ValueError):
        GRUN.parse_shard("3/2")
    assert GRUN.parse_shard("") is None


def test_the_gaia2_shard_hook_filters_the_stream_ARE_reads(monkeypatch, tmp_path):
    rows = [{"scenario_id": f"s{i}", "data": "{}"} for i in range(6)]

    class Split(list):
        def filter(self, fn):
            return Split(r for r in self if fn(r))

    class IDD(dict):
        pass
    fake = types.ModuleType("datasets")
    fake.IterableDatasetDict = IDD
    fake.load_dataset = lambda path, *a, **k: IDD(validation=Split(rows))
    monkeypatch.setitem(sys.modules, "datasets", fake)
    monkeypatch.setattr(GRUN, "WORKDIR", str(tmp_path))
    monkeypatch.setattr(GRUN, "DATASET", "/opt/gaia2/gaia2")
    GRUN.install_shard_hook((2, 3))
    got = [r["scenario_id"] for r in fake.load_dataset("/opt/gaia2/gaia2", name="time")[
        "validation"]]
    assert got == ["s1", "s4"]
    other = fake.load_dataset("/elsewhere", name="time")["validation"]
    assert len(other) == 6
    summary = GRUN.shard_summary(str(tmp_path))
    assert summary["spec"] == "2/3" and summary["per_config"][0]["selected"] == 2


def test_gaia2_forwards_the_shard_attempts_and_concurrency(monkeypatch, tmp_path):
    summ = {"shard": {"spec": "2/3", "per_config": [{"config": "time", "selected": 3}]}}
    res, [(cmd, kw)], _ = _gaia2_run(monkeypatch, tmp_path, summary=summ,
                                     env={"GBENCH_SHARD": "2/3"}, concurrency=5,
                                     attempt_count=4)
    envs = [cmd[i + 1] for i, x in enumerate(cmd) if x == "-e"]
    assert "GAIA2_SHARD=2/3" in envs and "GAIA2_MAX_CONCURRENT=5" in envs
    assert "GAIA2_NUM_RUNS=4" in envs
    assert res["shard"] == "2/3" and res["shard_selection"][0]["selected"] == 3
    assert res["attempts"]["attempts_per_sample"] == 4
    assert res["max_concurrent_scenarios"] == 5


def test_gaia2_refuses_a_shard_the_image_did_not_apply(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="did not apply it"):
        _gaia2_run(monkeypatch, tmp_path, env={"GBENCH_SHARD": "2/3"})


def test_an_empty_gaia2_shard(monkeypatch, tmp_path):
    res, _, _ = _gaia2_run(monkeypatch, tmp_path, env={"GBENCH_SHARD": "9/9"},
                           summary={"empty_shard": True, "shard": {"spec": "9/9"}})
    assert res["accuracy"] is None and res["empty_shard"] and res["shard"] == "9/9"


def test_gaia2_merge_recomputes_ares_statistics():
    def run(sid, rn, status, config="time", harness=()):
        return {"scenario_id": sid, "run_number": rn, "config": config,
                "phase_name": "standard", "status": status, "harness_errors": list(harness)}
    a = {"status": "success", "sample_traces": [run("a", 1, "success"), run("a", 2, "failed"),
                                                run("b", 1, "no_validation")]}
    b = {"status": "success", "sample_traces": [
        run("c", 1, "success", config="search"), run("c", 2, "success", config="search"),
        run("d", 1, "success", config="search", harness=["judge"])]}
    m = G.merge_shards([a, b])
    stats = m["merged_statistics"]
    assert stats["per_capability"]["time"]["success_rate"] == 50.0
    assert stats["per_capability"]["search"]["success_rate"] == 100.0
    assert m["accuracy"] == 75.0 and round(stats["micro_success_rate"], 2) == round(300 / 4, 2)
    assert m["n_harness_error_runs"] == 1 and m["status"] == "completed_with_errors"


# =========================================================================================== #
# wildclawbench and mcp_bench
# =========================================================================================== #
def test_the_openclaw_port_probe_finds_a_held_port():
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        port = held.getsockname()[1]
        assert W.busy_openclaw_ports([port]) == [port]
    assert W.busy_openclaw_ports([]) == []


def test_host_network_mode_refuses_busy_openclaw_ports(monkeypatch):
    monkeypatch.setattr(W, "check_wildclawbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(W, "_harness_dir", lambda: "/tmp/h")
    monkeypatch.setattr(W, "busy_openclaw_ports", lambda: [18789])
    monkeypatch.setenv("GBENCH_WILDCLAWBENCH_TASK_NETWORK", "host")
    with pytest.raises(RuntimeError, match="18789.*default bridge network"):
        W.run_wildclawbench("m", "http://127.0.0.1:9/v1")


@pytest.mark.parametrize("launcher", [WRUN, MRUN])
@pytest.mark.parametrize("spec", ["3/2", "x", "0/4", "1-2"])
def test_the_launchers_stop_on_a_malformed_shard(monkeypatch, launcher, spec):
    monkeypatch.setenv("GBENCH_SHARD", spec)
    with pytest.raises(SystemExit) as ei:
        launcher._shard()
    assert "GBENCH_SHARD" in str(ei.value.code)
    monkeypatch.setenv("GBENCH_SHARD", "")
    assert launcher._shard() is None
    monkeypatch.setenv("GBENCH_SHARD", "2/3")
    assert launcher._shard() == (2, 3)


def test_wildclawbench_merge_recomputes_the_harness_means():
    def row(tid, score, modality="pure-text", harness=()):
        return {"task_id": tid, "category": "c", "modality": modality, "overall_score": score,
                "harness_errors": list(harness)}
    a = {"status": "success", "raw_summary": {"per_task": [row("t1", 1.0), row("t2", None)]},
         "leaderboard_comparable_reason": "not the full 60-task set"}
    b = {"status": "success", "raw_summary": {"per_task": [
        row("t3", 0.5, "multimodal"), row("t4", 0.0, harness=["grading_infra"])]}}
    m = W.merge_shards([a, b])
    assert m["global_mean"] == 0.5 and m["accuracy"] == 50.0
    assert m["multimodal_mean"] == 0.5 and m["pure_text_mean"] == 0.5
    assert m["weighted_overall"] == 0.5 and m["n_harness_errors"] == 1


def test_the_wildclawbench_launcher_writes_an_empty_shard_summary(monkeypatch):
    monkeypatch.setattr(WRUN, "_task_index", lambda: {"a": {"category": "01_x"}})
    monkeypatch.setattr(WRUN, "ALL_CATEGORIES", ["01_x"])
    monkeypatch.delenv("WILDCLAW_TASK", raising=False)
    monkeypatch.delenv("WILDCLAW_LIMIT", raising=False)
    monkeypatch.delenv("WILDCLAW_CATEGORIES", raising=False)
    monkeypatch.setenv("GBENCH_SHARD", "2/2")
    assert WRUN._empty_shard_summary()["empty_shard"] is True
    monkeypatch.setenv("GBENCH_SHARD", "1/2")
    assert WRUN._empty_shard_summary() is None


def test_wildclawbench_notes_and_empty_shard():
    assert "--max-output-tokens 4096 not applied" in W._max_tokens_note(4096)
    res = W._empty_shard_result("m", False, "3/3", {"full_set_items": 2})
    assert res["accuracy"] is None and res["empty_shard"] and res["shard"] == "3/3"


# =========================================================================================== #
# the endpoint key and the documented run commands
# =========================================================================================== #
def test_the_swe_endpoint_key_is_the_one_every_pillar_sends(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "vllm-key-000011112222")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key-000011112222")
    assert P.endpoint_api_key() == "vllm-key-000011112222"
    assert A.endpoint_api_key() == "vllm-key-000011112222"
    monkeypatch.delenv("VLLM_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")
    assert P.endpoint_api_key() is None and A.endpoint_api_key() == "EMPTY"


_OWNED_DOCS = ("tau2", "tau3", "bfcl_v4_agentic", "agent_dojo", "complexfuncbench",
               "swe_bench_live", "swe_bench_multilingual", "swe_bench_pro", "multi_swe_bench",
               "copilot_bench_swe", "gaia2", "wildclawbench", "mcp_bench", "mcp_atlas")


def _commands(text):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.search(r"(^|\s|`)gbench\s", line):
            cmd = line
            while cmd.rstrip().endswith("\\") and i + 1 < len(lines):
                i += 1
                cmd = cmd.rstrip()[:-1] + " " + lines[i].strip()
            yield cmd
        i += 1


@pytest.mark.parametrize("name", _OWNED_DOCS)
def test_every_documented_eval_command_passes_max_output_tokens(name):
    text = (REPO / "docs" / "evals" / f"{name}.md").read_text(encoding="utf-8")
    for cmd in _commands(text):
        if re.search(r"--evals(-only)?\b", cmd) and "--help" not in cmd and "--dry-run" not in cmd:
            assert "--max-output-tokens" in cmd, cmd


@pytest.mark.parametrize("name", ["tau2", "tau3", "agent_dojo", "complexfuncbench",
                                  "bfcl_v4_agentic", "mcp_atlas", "swe_bench_pro",
                                  "wildclawbench"])
def test_the_tool_calling_docs_name_the_parser_flag_for_a_local_server(name):
    text = " ".join((REPO / "docs" / "evals" / f"{name}.md").read_text().split())
    assert "--tool-call-parser NAME" in text


def test_bfcl_merge_sums_category_counts_then_aggregates_as_bfcl():
    from gbench.runners.eval_suites import bfcl_v4_agentic as BF
    a = {"status": "success", "category_accuracy": {
        "memory_kv": {"correct": 3, "total": 5}, "web_search_base": {"correct": 1, "total": 4}},
        "total_questions": 9}
    b = {"status": "success", "category_accuracy": {
        "memory_kv": {"correct": 2, "total": 5}, "web_search_base": {"correct": 3, "total": 4}},
        "total_questions": 9}
    m = BF.merge_shards([a, b])
    want = BF.agentic_scores({"memory_kv": {"correct": 5, "total": 10},
                              "web_search_base": {"correct": 4, "total": 8}})
    # two of the five categories: no Agentic score, the pooled accuracy is the headline
    assert want["overall_accuracy_agentic"] is None and m["overall_accuracy_agentic"] is None
    assert m["accuracy"] == m["accuracy_pooled"] == 50.0
    assert m["category_accuracy"]["memory_kv"] == {"correct": 5, "total": 10, "accuracy": 50.0}
    assert m["total_questions"] == 18 and m["effective_n"] == 18
