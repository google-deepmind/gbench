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
"""SWE scaffolds (SWE-agent-for-eval, Win-Agent, MSWE-agent, mini-swe-agent): what is the
model's outcome and what is the harness's.

* An endpoint that keeps failing ends a SWE-agent-for-eval rollout with exit_error (its litellm
  retries re-raise the last exception, so exit_api never happens): an infrastructure failure,
  except a context-window overflow the fork does not recognise.
* The endpoint relay counts endpoint failures, replies cut off at a limit and replies with
  reasoning but no content or tool call; an agentic result applies them (harness errors;
  not comparable), and the traces never claim a measured zero.
* Instance images are pulled with the host's Docker CLI (the operator's login) before a
  container without Docker client configuration starts task containers from them; a failed
  pull is that instance's infrastructure failure.
* SWE-bench-Live's per-setting prerequisite rule works in both directions: on a Windows host
  the Windows tab runs and the Linux tabs are infra_required.
Hermetic: no container, no docker daemon, no network beyond loopback test servers; every cache
and output is under tmp_path.
"""
import asyncio
import http.server
import json
import os
import threading
import types

import pytest

from gbench.runners.eval_suites import multi_swe_bench as M
from gbench.runners.eval_suites import swe_agentic as A
from gbench.runners.eval_suites import swe_bench_live as LV
from gbench.runners.eval_suites import swebench_common as C


@pytest.fixture(autouse=True)
def _no_metadata_probe(monkeypatch):
    """The cloud metadata server gate (swebench_common.require_task_network_blocked) starts probe
    containers; tests/test_guard_swe_agents.py covers it, here it is stubbed."""
    monkeypatch.setattr(C, "require_task_network_blocked", lambda *a, **k: None)


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECRET = "sk-test-endpoint-key-5566778899"


@pytest.fixture(autouse=True)
def _scratch(monkeypatch, tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    for var in ("OPENAI_API_KEY", "HF_TOKEN", "GBENCH_SHARD", "GBENCH_SWE_BENCH_LIVE_MODE",
                "GBENCH_MULTI_SWE_BENCH_MODE", "GBENCH_SWE_BENCH_LIVE_TEMPERATURE",
                "GBENCH_MULTI_SWE_BENCH_TEMPERATURE", "GBENCH_TEMPERATURE",
                "GBENCH_REPETITION_PENALTY", "GBENCH_FREQUENCY_PENALTY",
                "GBENCH_PRESENCE_PENALTY", "GBENCH_SWE_BENCH_LIVE_IMAGE",
                "GBENCH_SWE_BENCH_LIVE_AGENT_IMAGE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GBENCH_SWE_OMP_THREADS", "0")
    for suite in ("SWE_BENCH_LIVE", "MULTI_SWE_BENCH", "COPILOT_BENCH_SWE"):
        monkeypatch.setenv(f"GBENCH_{suite}_TRAJ_DIR", str(tmp_path / "traj" / suite.lower()))


def _proc(rc=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


def _all_present(images, workers=1):
    return {i: {"state": "present"} for i in images}


# ============================================================================================
# SWE-agent-for-eval: exit statuses
# ============================================================================================
def _write_run(out, iid, status, message=None, patch="diff --git a/x b/x\n", api_calls=4):
    d = os.path.join(out, iid)
    os.makedirs(d, exist_ok=True)
    steps = [{"thought": "look", "response": "look", "observation": "ok"}]
    if message:
        steps.append({"thought": message, "response": message,
                      "observation": "Exited (autosubmitted)"})
    with open(os.path.join(d, f"{iid}.traj"), "w") as f:
        json.dump({"info": {"exit_status": status, "submission": patch,
                            "model_stats": {"api_calls": api_calls}}, "trajectory": steps}, f)
    with open(os.path.join(d, f"{iid}.pred"), "w") as f:
        json.dump({"instance_id": iid, "model_patch": patch}, f)


def _task(iid):
    return A.AgentTask(iid, f"issue {iid}", f"starryzhang/sweb.eval.x86_64.{iid}", "o/r",
                       meta={"split": "lite"})


@pytest.mark.parametrize("status", ["submitted (exit_error)", "exit_error"])
@pytest.mark.parametrize("message", [
    "Exit due to unknown error: litellm.InternalServerError: InternalServerError: "
    "OpenAIException - the served endpoint did not answer",
    "Exit due to unknown error: litellm.NotFoundError: NotFoundError: OpenAIException - "
    "The model `openai/wrong` does not exist.",
    "Exit due to unknown error: litellm.AuthenticationError: AuthenticationError: "
    "OpenAIException - Unauthorized",
    "Exit due to unknown error: litellm.APIConnectionError: OpenAIException - Connection "
    "error.",
])
def test_an_exit_error_rollout_is_an_infrastructure_failure(tmp_path, status, message):
    out = str(tmp_path / "run")
    _write_run(out, "o__r-1", status, message)
    tr = LV.read_sweagent_run([_task("o__r-1")], out, 0.0)[0]
    assert tr["agent_infra_error"] is True and tr["status"] == "AGENT_INFRA_ERROR"
    assert "exit_error" in tr["agent_error"] and "not the model's outcome" in tr["agent_error"]
    assert message[:60] in tr["agent_error"] and tr["agent_exit_message"].startswith(message)


def test_a_context_overflow_exit_error_is_the_models_outcome(tmp_path):
    # vLLM's wording is not the fork's own "is longer than the model's context length" check,
    # so the BadRequestError reaches forward_with_handling's catch-all as exit_error
    out = str(tmp_path / "run")
    _write_run(out, "o__r-1", "submitted (exit_error)",
               "Exit due to unknown error: litellm.BadRequestError: OpenAIException - The "
               "decoder prompt (length 140000) is longer than the maximum model length of "
               "131072.")
    tr = LV.read_sweagent_run([_task("o__r-1")], out, 0.0)[0]
    assert tr["agent_infra_error"] is False and tr["status"] == "OK"
    assert tr["agent_patch"] == "diff --git a/x b/x\n"


@pytest.mark.parametrize("status,infra", [
    ("submitted", False), ("submitted (exit_cost)", False), ("exit_context", False),
    ("submitted (exit_format)", False), ("exit_forfeit", False),
    ("submitted (exit_command_timeout)", False), ("exit_total_execution_time", False),
    ("submitted (exit_environment_error)", False), ("exit_api", True),
])
def test_the_other_exit_statuses(tmp_path, status, infra):
    out = str(tmp_path / "run")
    _write_run(out, "o__r-1", status, "Exit due to something")
    assert LV.read_sweagent_run([_task("o__r-1")], out, 0.0)[0]["agent_infra_error"] is infra


def test_sweagent_traces_do_not_claim_a_measured_zero(tmp_path):
    out = str(tmp_path / "run")
    _write_run(out, "o__r-1", "submitted")
    tr = LV.read_sweagent_run([_task("o__r-1")], out, 0.0)[0]
    assert tr["truncated_turns"] is None and "context_retry_turns" not in tr
    assert tr["agent_steps"] == 4


def test_an_instance_whose_image_did_not_pull_is_an_infrastructure_failure(tmp_path):
    out = str(tmp_path / "run")
    _write_run(out, "o__r-2", "submitted")
    tasks = [_task("o__r-1"), _task("o__r-2")]
    tr = LV.read_sweagent_run(tasks, out, 0.0,
                              {"o__r-1": "docker pull x failed: toomanyrequests"})
    assert tr[0]["agent_infra_error"] is True and tr[0]["agent_exit_status"] == \
        "image_pull_failed"
    assert "instance image: docker pull x failed: toomanyrequests" == tr[0]["agent_error"]
    assert tr[1]["agent_infra_error"] is False


# ============================================================================================
# the relay's counts
# ============================================================================================
_REPLIES = {
    "tool": (200, "application/json", {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": "", "reasoning_content": "r", "tool_calls": [{"id": "1"}]}}]}),
    "length": (200, "application/json", {"choices": [{"finish_reason": "length", "message": {
        "content": "partial", "reasoning_content": "r"}}]}),
    "empty": (200, "application/json", {"choices": [{"finish_reason": "stop", "message": {
        "content": "", "reasoning": "a long thought"}}]}),
    "ctx400": (400, "application/json", {"object": "error", "message": (
        "This model's maximum context length is 131072 tokens. However, your request has "
        "140000 input tokens."), "code": 400}),
    "bad400": (400, "application/json", {"object": "error", "message": "invalid tool_choice",
                                         "code": 400}),
    "http500": (500, "application/json", {"object": "error", "message": "engine dead"}),
    "http404": (404, "application/json", {"object": "error", "message": "no such model"}),
    "http401": (401, "application/json", {"object": "error", "message": "unauthorized"}),
}
_SSE = (b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"thinking"}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}\n\n'
        b"data: [DONE]\n\n")


class _Upstream(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        scenario = body.get("scenario")
        if scenario == "sse":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for part in _SSE.split(b"\n\n")[:-1]:
                self.wfile.write(part + b"\n\n")
                self.wfile.flush()
            return
        status, ctype, payload = _REPLIES[scenario]
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST

    def log_message(self, *a):
        return


@pytest.fixture
def upstream():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()
    server.server_close()


def _post(url, scenario):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url + "/chat/completions", method="POST",
                                 data=json.dumps({"model": "m", "messages": [],
                                                  "scenario": scenario}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_the_relay_counts_failures_and_cut_off_replies_and_relays_them_unchanged(upstream):
    relay = C.EndpointRelay(upstream, "127.0.0.1", key=SECRET).start()
    try:
        got = {s: _post(relay.url, s) for s in list(_REPLIES) + ["sse"]}
    finally:
        relay.stop()
    for scenario, (status, _, payload) in _REPLIES.items():
        assert got[scenario] == (status, json.dumps(payload).encode()), scenario
    assert got["sse"] == (200, _SSE)
    rec = relay.record()
    assert rec["upstream_errors"] == {"http_400": 1, "http_401": 1, "http_404": 1,
                                      "http_500": 1}
    assert rec["context_length_rejections"] == 1
    # tool, length, empty and the streamed reply
    assert rec["completion_replies"] == 4
    assert rec["replies_cut_off"] == 2                  # "length" and the streamed one
    # "empty", and the streamed reply, cut off inside its reasoning; a tool call is not empty
    assert rec["replies_empty_after_reasoning"] == 2
    assert SECRET not in json.dumps(rec)


def test_the_relay_counts_an_endpoint_that_does_not_answer():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                                           # nothing listens there
    relay = C.EndpointRelay(f"http://127.0.0.1:{port}/v1", "127.0.0.1").start()
    try:
        status, _ = _post(relay.url, "tool")
    finally:
        relay.stop()
    assert status == 502 and relay.record()["upstream_errors"] == {"connection": 1}


def test_relay_result_reasons_applies_the_counts():
    clean = {"status": "success"}
    assert C.relay_result_reasons(clean, {"upstream_errors": {}, "requests_refused": 0,
                                          "replies_cut_off": 0,
                                          "replies_empty_after_reasoning": 0,
                                          "completion_replies": 9}, "X") == []
    assert clean == {"status": "success", "truncated_turns": 0,
                     "empty_after_reasoning_turns": 0, "model_replies": 9,
                     "tool_call_dropped_turns": 0, "unclassified_empty_turns": 0,
                     "answer_in_thinking": 0, "answer_in_thinking_of": 9,
                     "answer_in_thinking_rate": 0.0}
    res = {"status": "success"}
    reasons = C.relay_result_reasons(res, {"upstream_errors": {"http_500": 2, "connection": 1},
                                           "requests_refused": 1, "replies_cut_off": 3,
                                           "replies_empty_after_reasoning": 1}, "MSWE-agent")
    assert res["status"] == "completed_with_errors"
    assert res["endpoint_failures"] == {"upstream_errors": {"http_500": 2, "connection": 1},
                                        "requests_refused": 1}
    assert (res["truncated_turns"], res["empty_after_reasoning_turns"]) == (3, 1)
    # a record without the token-id split cannot tell the empty turn apart: unclassified, never
    # the model's answer, and a comparability reason
    assert (res["answer_in_thinking"], res["unclassified_empty_turns"]) == (0, 1)
    text = "; ".join(reasons)
    assert "the served endpoint failed 3 relayed model request(s) (connection: 1, http_500: 2)" \
        in text and "the relay refused 1 request(s)" in text
    assert "3 model turn(s) cut off at the end of the served context window" in text
    assert "MSWE-agent sends no max_tokens" in text
    assert "1 model turn(s) that came back with reasoning and no content or tool call " \
        "without token ids" in text


def test_relay_answer_in_thinking_leaves_out_the_turns_cut_off_at_a_limit():
    relay = C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1")
    # (cut off, empty after reasoning, class): a reasoning-only turn cut off at the length limit
    # is a truncated turn, not one that ended with its answer inside the thinking channel
    relay.count_replies([(False, True, C.EMPTY_ANSWER_IN_THINKING), (True, True, None),
                         (False, False, None), (False, True, C.EMPTY_ANSWER_IN_THINKING)])
    rec = relay.record()
    assert (rec["completion_replies"], rec["replies_cut_off"],
            rec["replies_empty_after_reasoning"], rec["replies_answer_in_thinking"]) == (4, 1, 3, 2)
    res = {"status": "success"}
    reasons = C.relay_result_reasons(res, rec, "MSWE-agent")
    assert res["status"] == "success"
    assert (res["empty_after_reasoning_turns"], res["answer_in_thinking"],
            res["answer_in_thinking_of"], res["answer_in_thinking_rate"]) == (2, 2, 4, 0.5)
    assert reasons == ["1 model turn(s) cut off at the end of the served context window "
                       "(finish_reason length; MSWE-agent sends no max_tokens)"]
    errored = {"status": "error"}
    C.relay_result_reasons(errored, {"upstream_errors": {"http_500": 1}}, "X")
    assert errored["status"] == "error"


# ============================================================================================
# the agentic results apply the relay's counts
# ============================================================================================
def _counting_relay(counts):
    class _Relay(C.EndpointRelay):
        def record(self):
            rec = super().record()
            rec.update(counts)
            return rec
    return _Relay


def _python_rows(n):
    return [{**{c: "x" for c in LV.SNAPSHOT_COLUMNS}, "repo": "o/r",
             "instance_id": f"lite__r-{i}", "problem_statement": f"issue {i}",
             "base_commit": "c"} for i in range(n)]


def _sweagent_tab(monkeypatch, counts):
    rows = _python_rows(2)
    monkeypatch.setattr(LV, "EndpointRelay", _counting_relay(counts))

    def _rollouts(tasks, rws, model_name, relay_url, concurrency, out_dir, relay=None):
        return [{"sample_idx": i, "source_sample_idx": i, "category": t.category,
                 "agent_patch": "d\n", "extra_payload": {"instance_id": t.instance_id, **t.meta},
                 "agent_exit_status": "submitted", "agent_infra_error": False,
                 "agent_steps": 3, "truncated_turns": None, "is_correct": False,
                 "status": "OK"} for i, t in enumerate(tasks)]
    monkeypatch.setattr(LV, "run_sweagent", _rollouts)

    async def _score(traces):
        for t in traces:
            t["is_correct"] = True
    return LV._sweagent_tab("lite", rows, _score, {}, lambda res, m: [], "m",
                            "http://127.0.0.1:1/v1", 1, {})


def test_a_sweagent_tab_with_cut_off_turns_is_not_comparable(monkeypatch):
    res = _sweagent_tab(monkeypatch, {"replies_cut_off": 2, "completion_replies": 40})
    assert res["status"] == "success" and res["truncated_turns"] == 2
    assert res["model_replies"] == 40 and res["leaderboard_comparable"] is False
    assert "2 model turn(s) cut off at the end of the served context window " \
        "(finish_reason length; SWE-agent-for-eval sends no max_tokens)" in \
        res["leaderboard_comparable_reason"]
    assert all(t["truncated_turns"] is None for t in res["sample_traces"])


def test_a_sweagent_tab_with_endpoint_failures_is_a_harness_error(monkeypatch):
    res = _sweagent_tab(monkeypatch, {"upstream_errors": {"http_503": 4}})
    assert res["status"] == "completed_with_errors"
    assert res["endpoint_failures"]["upstream_errors"] == {"http_503": 4}
    assert "harness errors during the rollouts" in res["leaderboard_comparable_reason"]


def test_the_mswe_run_applies_the_relays_counts(monkeypatch, tmp_path):
    rows = [{"org": "o", "repo": "r", "number": n, "instance_id": f"o__r-{n}", "title": "T",
             "body": "B", "base": {"sha": "abc"}, "fix_patch": "gold",
             "resolved_issues": [{"number": 1, "title": "Issue", "body": "Body"}]}
            for n in (1, 2)]
    src = tmp_path / "src.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(M, "_read_instances", lambda: [("go", r) for r in rows])
    monkeypatch.setattr(M, "_SRC_FILES", {r["instance_id"]: str(src) for r in rows})
    monkeypatch.setattr(M, "check_multi_swe_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M, "check_mswe_agent_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M.swe_thread_cap, "apply", lambda c, n, e: (c, 0))
    monkeypatch.setattr(M, "EndpointRelay", _counting_relay({
        "replies_empty_after_reasoning": 1, "replies_answer_in_thinking": 1,
        "upstream_errors": {"http_400": 1}}))
    monkeypatch.setattr(M, "run_mswe_agent", lambda tasks, *a: [
        M.mswe_agent_trace(i, t, {"images": {t.instance_id: {"error": None}}, "instances": {
            t.instance_id: {"completed": True, "patch": "diff --git a/x b/x\n",
                            "exit_status": "submitted", "api_calls": 3}}}, str(tmp_path), 0.0)
        for i, t in enumerate(tasks)])

    def _harness(cmd, **kw):
        cfg = json.load(open(cmd[-1]))
        with open(f"{cfg['output_dir']}/final_report.json", "w") as f:
            json.dump({"resolved_ids": ["o/r:pr-1"], "unresolved_ids": ["o/r:pr-2"],
                       "completed_ids": ["o/r:pr-1", "o/r:pr-2"], "error_ids": [],
                       "incomplete_ids": []}, f)
        return _proc()
    monkeypatch.setattr(M.subprocess, "run", _harness)
    res = M.run_multi_swe_bench("m", "http://127.0.0.1:1/v1", 1)
    assert res["correct_answers"] == 1 and res["status"] == "completed_with_errors"
    assert res["empty_after_reasoning_turns"] == 1 and res["truncated_turns"] == 0
    assert res["answer_in_thinking"] == 1
    reason = res["leaderboard_comparable_reason"]
    assert "http_400: 1" in reason and "returned reasoning with no content" not in reason
    assert all(t["truncated_turns"] is None for t in res["sample_traces"])
    assert all("context_retry_turns" not in t for t in res["sample_traces"])


@pytest.mark.parametrize("truncated,empty,relay_counts,in_thinking,reason", [
    (1, 0, {"completion_replies": 2}, 0,
     "1 model turn(s) cut off at the end of the served context window"),
    # answer-in-thinking turns (the relay's token ids show no tool call) are the model's
    # outcome: no reason of their own (the run is not comparable here only because of the
    # no-think temperature the test runs at)
    (0, 2, {"completion_replies": 2, "replies_answer_in_thinking": 2}, 2, None),
    # a native tool call the server's parser removed is a server-parser artifact
    (0, 2, {"completion_replies": 2, "replies_answer_in_thinking": 1,
            "replies_tool_call_dropped": 1}, 1,
     "1 model turn(s) whose generation was a native tool call the served endpoint's "
     "tool-call parser removed"),
    # empty turns the relay did not classify (no token ids): unclassified, a reason
    (0, 2, {"completion_replies": 2}, 0,
     "2 model turn(s) that came back with reasoning and no content or tool call without "
     "token ids"),
])
def test_mini_swe_agent_runs_count_cut_off_and_empty_turns(monkeypatch, truncated, empty,
                                                           relay_counts, in_thinking, reason):
    monkeypatch.setattr(A, "EndpointRelay", _counting_relay(relay_counts))
    monkeypatch.setattr(A, "bridge_gateway", lambda: ("127.0.0.1", ""))   # the relay's host
    monkeypatch.setattr(A, "require_relay_reachable", lambda *a, **k: None)   # no docker
    monkeypatch.setattr(A, "describe_agent_v2", lambda v: {
        "version": "2.4.6", "config_name": A.AGENT_V2_CONFIG,
        "config_sha256": A.AGENT_V2_CONFIG_SHA256["2.4.6"],
        "config": {"agent": {"step_limit": 250, "cost_limit": 3.0},
                   "environment": {"timeout": 60},
                   "model": {"model_kwargs": {"drop_params": True}}}})
    monkeypatch.setattr(A, "run_rollouts", lambda tasks, *a, **k: [
        {"sample_idx": i, "category": t.category, "agent_patch": "d\n",
         "extra_payload": {"instance_id": t.instance_id}, "agent_exit_status": "Submitted",
         "agent_infra_error": False, "agent_steps": 2, "truncated_turns": truncated,
         "empty_after_reasoning_turns": empty, "is_correct": False, "status": "OK"}
        for i, t in enumerate(tasks)])

    async def _resolve(traces):
        for t in traces:
            t["is_correct"] = True
    res = A.run_agentic("copilot_bench_swe", "m", "http://h:1/v1", 1,
                        [A.AgentTask("a__a-1", "issue", A.swebench_image("a__a-1"), "o/r")],
                        _resolve, {}, agent_v2="2.4.6")
    assert (res["truncated_turns"], res["empty_after_reasoning_turns"]) == (truncated, empty)
    # the rate's numerator and denominator both come from the relay's replies
    assert (res["answer_in_thinking"], res["answer_in_thinking_of"]) == (in_thinking, 2)
    if reason is None:
        assert "model turn(s)" not in (res.get("leaderboard_comparable_reason") or "")
    else:
        assert res["leaderboard_comparable"] is False and reason in \
            res["leaderboard_comparable_reason"]


def test_empty_after_reasoning_reads_the_condensed_trajectory():
    msgs = [{"role": "assistant", "content": "", "reasoning": "r"},
            {"role": "assistant", "content": "", "reasoning": "r", "actions": ["ls"]},
            {"role": "assistant", "content": "THOUGHT", "reasoning": "r"},
            {"role": "user", "content": "", "reasoning": "r"},
            {"role": "assistant", "content": "  ", "reasoning": ""}]
    assert A.empty_after_reasoning(msgs) == 1


def test_win_agent_traces_leave_unmeasured_fields_empty(monkeypatch, tmp_path):
    agent = tmp_path / "win-agent-src"
    (agent / "config").mkdir(parents=True)
    (agent / "main.py").write_text("# stand-in\n")
    (agent / "config" / "default.yaml").write_text("platform: windows\nmax_steps: 100\n")
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_WIN_AGENT_DIR", str(agent))
    out = tmp_path / "out"
    out.mkdir()

    def _run(cmd, **kw):
        import yaml
        cfg = yaml.safe_load(open(cmd[cmd.index("--config_path") + 1]))
        d = os.path.join(kw["cwd"], "output", LV._win_agent_run_id(cfg))
        os.makedirs(d)
        json.dump({"w-1": {"model_patch": "d\n"}}, open(os.path.join(d, "preds.json"), "w"))
        json.dump({"w-1": "submitted"}, open(os.path.join(d, "exit_status.json"), "w"))
        return _proc()
    monkeypatch.setattr(LV.subprocess, "run", _run)
    rows = [{"instance_id": "w-1", "docker_image": "img", "_split": "test"}]
    tr = LV.run_win_agent([A.AgentTask("w-1", "p", "img", "o/r", meta={"split": "test"})],
                          rows, "m", "http://127.0.0.1:1/v1", 1, str(out))[0]
    assert tr["agent_steps"] is None and tr["truncated_turns"] is None
    assert "context_retry_turns" not in tr and tr["agent_infra_error"] is False


# ============================================================================================
# instance images: pulled on the host
# ============================================================================================
def test_host_pull_images_pulls_only_missing_images_with_the_hosts_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    calls = []

    def _run(cmd, **kw):
        calls.append((list(cmd), kw.get("env")))
        image = cmd[-1]
        if cmd[:3] == ["docker", "image", "inspect"]:
            return _proc(0 if image == "present:1" else 1)
        if image == "broken:1":
            return _proc(1, "", f"denied: token {SECRET} rejected")
        return _proc(0)
    monkeypatch.setattr(C.subprocess, "run", _run)
    got = C.host_pull_images(["present:1", "missing:1", "broken:1", "missing:1", ""], 2)
    assert got["present:1"] == {"state": "present"}
    assert got["missing:1"] == {"state": "pulled"}
    assert got["broken:1"]["state"] == "failed" and SECRET not in got["broken:1"]["error"]
    assert "docker pull broken:1 failed" in got["broken:1"]["error"]
    pulls = [c for c, _ in calls if c[:2] == ["docker", "pull"]]
    assert sorted(pulls) == [["docker", "pull", "broken:1"], ["docker", "pull", "missing:1"]]
    # this process's own environment: the CLI reads the operator's login and DOCKER_CONFIG
    assert all(env is None for _, env in calls)
    assert C.image_pull_summary(got) == {"images": 3, "present": 1, "pulled": 1,
                                         "failed": {"broken:1": got["broken:1"]["error"]}}


def test_sweagent_images_are_pulled_on_the_host_before_the_batch(monkeypatch, tmp_path):
    out = str(tmp_path / "out")
    os.makedirs(out)
    rows = _python_rows(3)
    tasks = [A.AgentTask(r["instance_id"], r["problem_statement"],
                         LV.sweagent_instance(r)["image_name"], "o/r") for r in rows]
    bad = tasks[1].image
    pulled = []

    def _pulls(images, workers=1):
        pulled.append(list(images))
        return {i: ({"state": "failed", "error": "docker pull failed: toomanyrequests"}
                    if i == bad else {"state": "pulled"}) for i in images}
    monkeypatch.setattr(LV, "host_pull_images", _pulls)
    runs = []

    def _run(cmd, **kw):
        if "run-batch" in cmd:
            lines = [json.loads(x) for x in open(cmd[cmd.index("--instances.subset") + 1])]
            runs.append((cmd, lines))
            for line in lines:
                _write_run(out, line["instance_id"], "submitted")
        return _proc()
    monkeypatch.setattr(LV.subprocess, "run", _run)
    traces = LV.run_sweagent(tasks, rows, "m", "http://127.0.0.1:9/v1", 2, out)
    assert pulled == [[t.image for t in tasks] + [LV.SWEREX_BUILDER_BASE]]
    assert LV.SWEREX_BUILDER_BASE == "python:3.11.9-slim-bookworm"
    cmd, lines = runs[0]
    assert [x["instance_id"] for x in lines] == ["lite__r-0", "lite__r-2"]
    # the task-container label is SWE-agent's option `instances.deployment.docker_args`, the
    # one token whose name contains ".docker"; every other token is checked
    label = ("--instances.deployment.docker_args=--label=org.gbench.sweagent_batch="
             + cmd[cmd.index("--name") + 1])
    assert label in cmd
    joined = " ".join(a for a in cmd if a != label)
    assert "DOCKER_CONFIG" not in joined and ".docker" not in joined and "HOME=/tmp" in cmd
    by_id = {t["extra_payload"]["instance_id"]: t for t in traces}
    assert by_id["lite__r-1"]["agent_infra_error"] is True
    assert "toomanyrequests" in by_id["lite__r-1"]["agent_error"]
    assert not by_id["lite__r-0"]["agent_infra_error"]
    record = json.load(open(os.path.join(out, "image_pulls.json")))
    assert record["instance_images"]["pulled"] == 2 and bad in record["instance_images"]["failed"]
    assert record["builder_base"] == {"image": LV.SWEREX_BUILDER_BASE, "state": "pulled"}


def test_mswe_images_are_pulled_on_the_host_before_each_language(monkeypatch, tmp_path):
    rows = {f"o__r-{n}": {"org": "o", "repo": "r", "number": n, "instance_id": f"o__r-{n}",
                          "resolved_issues": [{"title": "t", "body": "b"}]} for n in (1, 2)}
    tasks = [A.AgentTask(i, M._issue_text(r), M.published_image("o", "r", r["number"]), "go",
                         meta={"org": "o", "repo": "r", "number": r["number"]})
             for i, r in rows.items()]
    monkeypatch.setattr(M, "host_pull_images", lambda images, workers=1: {
        i: ({"state": "failed", "error": "docker pull failed: denied"}
            if i.endswith("pr-2") else {"state": "present"}) for i in images})
    seen = []

    def _run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            seen.append(cmd)
            work = cmd[cmd.index("-v", cmd.index("-v") + 1) + 1].split(":")[0]
            got = [json.loads(x)["instance_id"] for x in open(os.path.join(work, "rows.jsonl"))]
            json.dump({"language": "go", "images": {g: {"error": None} for g in got},
                       "instances": {g: {"completed": True, "patch": "d\n",
                                         "exit_status": "submitted", "api_calls": 2}
                                     for g in got}},
                      open(os.path.join(work, "summary.json"), "w"))
        return _proc()
    monkeypatch.setattr(M.subprocess, "run", _run)
    traces = M.run_mswe_agent(tasks, rows, "m", "http://127.0.0.1:5/v1", 2,
                              str(tmp_path / "out"), 0.0, 0.95)
    assert len(seen) == 1
    joined = " ".join(seen[0])
    assert "DOCKER_CONFIG" not in joined and ".docker" not in joined and "HOME=/tmp" in seen[0]
    assert [t["agent_infra_error"] for t in traces] == [False, True]
    assert "instance image: docker pull failed: denied" in traces[1]["agent_error"]
    record = json.load(open(tmp_path / "out" / "go" / "image_pulls.json"))
    assert record["present"] == 1 and list(record["failed"]) == ["mswebench/o_m_r:pr-2"]


def test_a_mini_swe_agent_2_image_that_does_not_pull_starts_no_container(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "host_pull_images", lambda images, workers=1: {
        i: {"state": "failed", "error": "docker pull x failed: denied"} for i in images})
    monkeypatch.setattr(A.subprocess, "run", lambda *a, **k: pytest.fail("no container"))
    tr = A._rollout_v2(0, A.AgentTask("a__a-1", "p", A.swebench_image("a__a-1"), "o/r"), {},
                       str(tmp_path), image="img", version="2.4.6")
    assert tr["agent_infra_error"] is True and tr["agent_exit_status"] == "ImagePullFailed"
    assert tr["agent_error"] == "instance image: docker pull x failed: denied"
    assert tr["agent_failed_phase"] == "environment"


def test_the_multilang_harness_pass_pulls_the_submitted_images_first(monkeypatch):
    rows = [{"instance_id": f"go-{i}", "docker_image": f"starryzhang/img-{i}", "_split": "go",
             "log_parser": "x"} for i in range(3)]
    pulled = []
    monkeypatch.setattr(LV, "host_pull_images", lambda images, workers=1: pulled.append(
        list(images)) or {i: {"state": "pulled"} for i in images})

    def _run(cmd, **kw):
        out = cmd[cmd.index("--output_dir") + 1]
        os.makedirs(out, exist_ok=True)
        json.dump({"submitted": 2, "success_ids": ["go-0"], "failure_ids": ["go-2"],
                   "empty_patch_ids": [], "error_ids": [], "incomplete_ids": []},
                  open(os.path.join(out, "results.json"), "w"))
        return _proc()
    monkeypatch.setattr(LV.subprocess, "run", _run)
    metrics = {}
    traces = [{"extra_payload": {"instance_id": i}, "agent_patch": "d\n"} for i in ("go-0",
                                                                                    "go-2")]
    asyncio.run(LV.make_launch_scorer("go", rows, "linux", 2, metrics)(traces))
    assert pulled == [["starryzhang/img-0", "starryzhang/img-2"]]
    assert metrics["harness_image_pulls"]["pulled"] == 2
    pulled.clear()
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_HARNESS_DIR", "/nonexistent")
    asyncio.run(LV.make_launch_scorer("windows", rows, "windows", 2, {})(traces))
    assert pulled == []                                  # the Windows host pulls for itself


def test_the_python_harness_pass_pulls_the_starryzhang_images_first(monkeypatch, tmp_path):
    pulled = []
    monkeypatch.setattr(C, "host_pull_images", lambda images, workers=1: pulled.append(
        list(images)) or {i: {"state": "present"} for i in images})
    monkeypatch.setattr(C.subprocess, "run", lambda cmd, **kw: _proc())
    metrics = {}
    scorer = C.make_swebench_scorer("swe_bench_live", "m", str(tmp_path / "rows.jsonl"),
                                    "lite", "starryzhang", 1, metrics,
                                    harness_image="gbench-swe-bench-live:a145aa87",
                                    instance_image=LV.python_instance_image)
    asyncio.run(scorer([{"extra_payload": {"instance_id": "psf__requests-1"},
                         "agent_patch": "d\n"}]))
    assert pulled == [["starryzhang/sweb.eval.x86_64.psf_1776_requests-1"]]
    assert metrics["harness_image_pulls"] == {"images": 1, "present": 1, "pulled": 0,
                                              "failed": {}}
    assert LV.sweagent_instance({"instance_id": "psf__requests-1", "problem_statement": "p",
                                 "base_commit": "c"})["image_name"] == pulled[0][0]


def test_the_v5_one_shot_harness_pulls_each_rows_image(monkeypatch):
    rows = [{"instance_id": "a__a-1", "image": "swebench/sweb.eval.x86_64.a_1776_a-1:latest",
             "problem_statement": "p", "repo": "a/a"}]
    monkeypatch.setattr(C, "_pinned_snapshot_rows", lambda *a: ("/snap/test.jsonl", rows))
    seen = {}

    def _scorer(*a, **k):
        seen.update(k)
        return lambda traces: None
    monkeypatch.setattr(C, "make_swebench_scorer", _scorer)
    monkeypatch.setattr(C, "run_eval_suite", lambda **k: {"status": "success",
                                                          "temperature": 0.0})
    C.execute_swebench("copilot_bench_swe", "m", "http://x/v1", 1, "d/d", "test", "swebench",
                       dataset_revision="abc", harness_image="gbench-swebench-v5:5.0.2",
                       harness_v5=True)
    assert seen["instance_image"]("a__a-1") == rows[0]["image"]


# ============================================================================================
# SWE-bench-Live: the per-setting prerequisite rule in both directions
# ============================================================================================
def _record_runs(monkeypatch):
    ran = []

    def _setting(setting, mode, model_name, base_url, concurrency, kwargs):
        ran.append(setting)
        return {"status": "success", "accuracy": 50.0, "total_questions": 2,
                "correct_answers": 1, "leaderboard_comparable": True,
                "tab": LV.TAB_LABELS[setting], "sample_traces": []}
    monkeypatch.setattr(LV, "_run_setting", _setting)
    return ran


def test_on_a_windows_host_the_windows_tab_runs_and_the_linux_tabs_are_reported(monkeypatch,
                                                                              tmp_path):
    monkeypatch.setattr(LV, "check_windows_metadata_guard", lambda: (True, ""))
    monkeypatch.setattr(LV._platform, "system", lambda: "Windows")
    monkeypatch.setattr(LV, "check_windows_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(LV.shutil, "which", lambda name: "docker")
    monkeypatch.setattr(LV, "pinned_swe_snapshot", lambda *a, **k: str(tmp_path / "s.jsonl"))
    monkeypatch.setattr(LV, "_check_launch_harness", lambda path: (True, ""))

    def _run(cmd, **kw):
        if cmd[:2] == ["docker", "info"]:
            return _proc(0)
        return _proc(1, "", "No such image")          # a Windows engine holds no Linux image
    monkeypatch.setattr(LV.subprocess, "run", _run)
    ran = _record_runs(monkeypatch)
    res = LV.run_swe_bench_live("m", "http://h:1/v1", 1)
    linux = [s for s in LV.SETTINGS if s != "windows"]
    assert ran == ["windows"]
    assert res["status"] == "completed_with_errors" and res["settings_infra_required"] == linux
    for s in linux:
        entry = res["settings"][s]
        assert entry["status"] == "infra_required" and entry["accuracy"] is None
        assert "is not built" in entry["infra_required"]
        assert "This is a Windows host" in entry["infra_required"]
        assert res["headline_metrics"][f"{s}_resolved_rate"] is None
    assert res["headline_metrics"]["windows_resolved_rate"] == 50.0
    assert res["headline_setting"] == "windows" and res["accuracy"] == 50.0


def test_a_setting_whose_snapshot_cannot_be_fetched_is_reported_alone(monkeypatch, tmp_path):
    monkeypatch.setattr(LV.shutil, "which", lambda name: "docker")
    labels = json.dumps({"org.gbench.swe_bench_live_fork": LV.FORK_COMMIT,
                         "org.gbench.swe_bench_live_main": LV.MAIN_COMMIT,
                         "org.gbench.repolaunch": LV.REPOLAUNCH_COMMIT})
    monkeypatch.setattr(LV.subprocess, "run", lambda cmd, **kw: _proc(0, labels))
    monkeypatch.setattr(LV, "_check_launch_harness", lambda path: (True, ""))

    def _snapshot(dataset, split, revision, eval_name, **kw):
        if split == "full":
            raise RuntimeError("Could not load dataset SWE-bench-Live/SWE-bench-Live (full)")
        return str(tmp_path / f"{split}.jsonl")
    monkeypatch.setattr(LV, "pinned_swe_snapshot", _snapshot)
    monkeypatch.setattr(LV, "check_sweagent_prerequisites", lambda: (True, ""))
    ran = _record_runs(monkeypatch)
    res = LV.run_swe_bench_live("m", "http://h:1/v1", 1, eval_categories="lite,full,go")
    assert ran == ["lite", "go"] and res["settings_infra_required"] == ["full"]
    assert "Could not load dataset" in res["settings"]["full"]["infra_required"]
    assert res["status"] == "completed_with_errors"


def test_without_the_sweagent_image_the_linux_tabs_are_reported_and_windows_runs(monkeypatch):
    monkeypatch.setattr(LV, "check_windows_metadata_guard", lambda: (True, ""))
    monkeypatch.setattr(LV, "check_windows_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(LV, "check_swe_bench_live_prerequisites", lambda settings: (True, ""))
    monkeypatch.setattr(LV, "check_sweagent_prerequisites",
                        lambda: (False, "the SWE-agent-for-eval image 'x' is not built."))
    ran = _record_runs(monkeypatch)
    res = LV.run_swe_bench_live("m", "http://h:1/v1", 1, eval_categories="lite,go,windows")
    assert ran == ["windows"] and res["settings_infra_required"] == ["lite", "go"]
    why = res["settings"]["lite"]["infra_required"]
    assert "SWE-agent-for-eval" in why and "GBENCH_SWE_BENCH_LIVE_MODE=oneshot" in why


def test_when_no_setting_can_run_the_error_names_each_missing_prerequisite(monkeypatch):
    monkeypatch.setattr(LV, "check_windows_prerequisites",
                        lambda: (False, "needs a Windows host."))
    monkeypatch.setattr(LV, "check_swe_bench_live_prerequisites",
                        lambda settings: (False, "the harness image is not built."))
    monkeypatch.setattr(LV, "_run_setting", lambda *a, **k: pytest.fail("nothing may run"))
    with pytest.raises(RuntimeError) as err:
        LV.run_swe_bench_live("m", "http://h:1/v1", 1, eval_categories="lite,go,windows")
    text = str(err.value)
    assert text.startswith("[INFRA REQUIRED] swe_bench_live")
    assert "no selected setting can run here" in text
    assert "Lite, Go: the harness image is not built." in text
    assert "Windows: needs a Windows host." in text


def test_missing_prerequisites_checks_the_selection_once_when_it_is_complete(monkeypatch):
    monkeypatch.setattr(LV, "check_windows_metadata_guard", lambda: (True, ""))
    checked = []
    monkeypatch.setattr(LV, "check_windows_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(LV, "check_swe_bench_live_prerequisites",
                        lambda settings: checked.append(list(settings)) or (True, ""))
    monkeypatch.setattr(LV, "check_sweagent_prerequisites", lambda: (True, ""))
    assert LV.missing_prerequisites(list(LV.SETTINGS), A.AGENTIC) == {}
    assert checked == [list(LV.SETTINGS)]


# ============================================================================================
# the docs say what the code does
# ============================================================================================
def _doc(name):
    return " ".join(open(os.path.join(REPO, "docs", "evals", f"{name}.md"),
                         encoding="utf-8").read().split())


def test_the_docs_describe_the_exit_statuses_pulls_and_counts():
    live = _doc("swe_bench_live")
    assert "`exit_api` (the endpoint kept failing)" not in live
    for phrase in ("retries with tenacity `reraise=True`", "`agent_exit_message`",
                   "`python:3.11.9-slim-bookworm`", "`swebench_common.host_pull_images`",
                   "`empty_after_reasoning_turns`", "`endpoint_failures`",
                   "`harness_image_pulls`", '"image_pull_failed"', "Binary files ... differ",
                   "in both directions", "each naming the image it lacks"):
        assert phrase in live, phrase
    mswe = _doc("multi_swe_bench")
    for phrase in ("`swebench_common.host_pull_images`", "`empty_after_reasoning_turns`",
                   "`image_pulls.json`", "`endpoint_failures`"):
        assert phrase in mswe, phrase
    for name in ("copilot_bench_swe", "swe_bench_multilingual"):
        doc = _doc(name)
        assert "not a deviation" not in doc and '"ImagePullFailed"' in doc, name
        assert "`empty_after_reasoning_turns`" in doc and "`harness_image_pulls`" in doc, name
