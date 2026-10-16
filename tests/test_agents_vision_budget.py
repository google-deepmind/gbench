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

"""mcp_bench and toolbench: the request timeout of a --thinking run and failed model requests;
medxpertqa's record of the official cap a --thinking run does not apply.

  * request settings (both images) - with <prefix>REQUEST_TIMEOUT_S every request carries that
    timeout (read, write and pool; connect 5 s as in the openai client's default), and the real
    openai client honours it; a request that ends in an endpoint or connection error is counted
    (failed_requests, timed_out, failed_by_error by class and HTTP status, no message text), a
    context-window rejection the harness handles as upstream would is rejected_over_context, and
    an asyncio cancellation is cancelled.
  * hosts - a --thinking run passes the timeout base.request_timeout_s sizes for its budget and
    checks the image applied it; failed requests are harness errors (status
    completed_with_errors, harness_errors, named in the reason); the image label is version 2.
  * medxpertqa - with --thinking the chat branch's 2048 (a served model's path in both modes) is
    recorded as not applied to reasoning, and the record names what was sent: the run's
    --max-output-tokens, else the thinking floor.
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import mcp_bench as MB
from gbench.runners.eval_suites import medxpertqa
from gbench.runners.eval_suites import toolbench as T

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.join(_HERE, "..")
_MB_DOCKER = os.path.join(_REPO, "gbench", "docker")
_TB_DOCKER = os.path.join(_REPO, "docker")
if _MB_DOCKER not in sys.path:
    sys.path.insert(0, _MB_DOCKER)
import mcp_bench_request_policy as POL  # noqa: E402

GEMINI = "AIzaSyTESTgeminiKEY0000002222"


def _read(*parts):
    with open(os.path.join(_REPO, *parts), encoding="utf-8") as fh:
        return fh.read()


def _reply(content="ok", finish="stop"):
    msg = types.SimpleNamespace(content=content, tool_calls=None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason=finish,
                                                                 message=msg)])


class APITimeoutError(Exception):
    """Stands in for openai.APITimeoutError (the class name is what the count reads)."""


class InternalServerError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class BadRequestError(Exception):
    status_code = 400


def _window_error(text):
    return BadRequestError("Error code: 400 - {'error': {'message': \"" + text + "\"}}")


# --------------------------------------------------------------------------- #
# the request settings module (both images)
# --------------------------------------------------------------------------- #
def test_both_images_carry_the_same_request_settings_module():
    assert (_read("gbench", "docker", "mcp_bench_request_policy.py")
            == _read("docker", "toolbench_request_policy.py"))


def test_the_timeout_is_sent_only_when_the_host_sets_it():
    p = POL.RequestPolicy("TB_", env={"TB_ENABLE_THINKING": "true", "TB_MAX_TOKENS": "65536",
                                      "TB_REQUEST_TIMEOUT_S": "9392"})
    kw = p.prepare({"model": "m", "messages": [], "max_tokens": 1024})
    assert kw["max_tokens"] == 65536
    t = kw["timeout"]
    assert (t.read, t.write, t.pool, t.connect) == (9392.0, 9392.0, 9392.0, POL.CONNECT_TIMEOUT_S)
    assert POL.CONNECT_TIMEOUT_S == 5.0
    assert p.settings() == {"enable_thinking": True, "max_tokens": 65536, "temperature": None,
                            "request_timeout_s": 9392}
    # without --thinking the host sets no timeout: the harness client's own applies
    q = POL.RequestPolicy("MCP_", env={"MCP_ENABLE_THINKING": "false"})
    kw = q.prepare({"model": "m", "messages": [], "max_tokens": 12000})
    assert "timeout" not in kw and kw["max_tokens"] == 12000
    assert q.settings()["request_timeout_s"] is None


def test_timed_out_requests_are_counted_and_dumped_without_text(tmp_path):
    log = tmp_path / "log.jsonl"
    p = POL.RequestPolicy("TB_", base_url="http://u:secretpw77@127.0.0.1:1/v1",
                          env={"TB_ENABLE_THINKING": "true", "TB_MAX_TOKENS": "65536",
                               "TB_REQUEST_TIMEOUT_S": "9392", "TB_REQUEST_LOG": str(log)})
    p.tokenize_url = None
    sent = []

    def create(**kw):
        sent.append(kw)
        raise APITimeoutError("Request timed out. secret-message-text")

    for _ in range(5):
        with pytest.raises(APITimeoutError):
            p.call(create, {"model": "m", "messages": [], "max_tokens": 1024})
    assert all(k["timeout"].read == 9392.0 for k in sent)
    assert p.counts["failed_requests"] == 5 and p.counts["timed_out"] == 5
    assert p.counts["replies"] == 0 and p.failed_by_error == {"APITimeoutError": 5}
    p.dump()
    line = log.read_text()
    rec = json.loads(line)
    assert rec["counts"]["failed_requests"] == 5 and rec["counts"]["timed_out"] == 5
    assert rec["failed_by_error"] == {"APITimeoutError": 5}
    assert rec["settings"]["request_timeout_s"] == 9392
    for leak in ("secret-message-text", "secretpw77", "127.0.0.1", "timed out"):
        assert leak not in line


def test_endpoint_errors_are_counted_by_class_and_status():
    p = POL.RequestPolicy("MCP_", env={"MCP_ENABLE_THINKING": "false"})

    async def create(**kw):
        raise InternalServerError("Error code: 503", 503)

    async def refused(**kw):
        raise ConnectionRefusedError("refused")

    for fn in (create, create, refused):
        with pytest.raises((InternalServerError, ConnectionRefusedError)):
            asyncio.run(p.acall(fn, {"model": "m", "messages": [], "max_tokens": 12000}))
    assert p.counts["failed_requests"] == 3 and p.counts["timed_out"] == 0
    assert p.failed_by_error == {"InternalServerError 503": 2, "ConnectionRefusedError": 1}


def test_a_window_rejection_of_upstreams_own_cap_is_left_to_the_harness():
    p = POL.RequestPolicy("MCP_", env={"MCP_ENABLE_THINKING": "false"})
    err = _window_error("This model's maximum context length is 32768 tokens. However, you "
                        "requested 12000 output tokens and your prompt contains 25000 input "
                        "tokens, for a total of 37000 tokens.")

    def create(**kw):
        raise err

    with pytest.raises(BadRequestError):
        p.call(create, {"model": "m", "messages": [], "max_tokens": 12000})
    assert p.counts["rejected_over_context"] == 1 and p.counts["failed_requests"] == 0


def test_a_window_rejection_of_the_budget_is_a_failure_unless_the_prompt_fills_the_window():
    env = {"TB_ENABLE_THINKING": "true", "TB_MAX_TOKENS": "65536"}
    # the prompt alone leaves less than MIN_ROOM (known from /tokenize): the harness's own case
    p = POL.RequestPolicy("TB_", env=env)
    p.room = lambda kw, api_key=None: 100
    full = _window_error("This model's maximum context length is 32768 tokens. However, you "
                         "requested 65536 output tokens and your prompt contains 32668 input "
                         "tokens, for a total of 98204 tokens.")

    def create_full(**kw):
        raise full

    with pytest.raises(BadRequestError):
        p.call(create_full, {"model": "m", "messages": [], "max_tokens": 1024})
    assert p.counts["rejected_over_context"] == 1 and p.counts["failed_requests"] == 0
    # no /tokenize and vLLM's character-count rejection, which the refit cannot read although
    # the window had room: the budget could not be fitted, a failed request
    q = POL.RequestPolicy("TB_", env=env)
    q.tokenize_url = None
    chars = _window_error("This model's maximum context length is 32768 tokens. However, you "
                          "requested 32512 output tokens and your prompt contains 80000 "
                          "characters (more than 7936 characters, which is the upper bound for "
                          "256 input tokens).")
    over = _window_error("max_tokens=65536 cannot be greater than "
                         "max_model_len=max_total_tokens=32768. Please request fewer output "
                         "tokens.")
    errors = [over, chars]
    sent = []

    def create_chars(**kw):
        sent.append(kw["max_tokens"])
        raise errors.pop(0)

    with pytest.raises(BadRequestError):
        q.call(create_chars, {"model": "m", "messages": [], "max_tokens": 1024})
    assert sent == [65536, 32768 - POL.MIN_ROOM]
    assert q.counts["refitted_after_rejection"] == 1
    assert q.counts["failed_requests"] == 1 and q.counts["rejected_over_context"] == 0
    assert q.failed_by_error == {"BadRequestError 400": 1}


def test_a_cancelled_request_is_counted_apart_from_failures():
    p = POL.RequestPolicy("MCP_", env={"MCP_ENABLE_THINKING": "true", "MCP_MAX_TOKENS": "4096",
                                       "MCP_REQUEST_TIMEOUT_S": "1712"})
    state = {}

    async def slow(**kw):
        state["started"].set()
        await asyncio.sleep(30)
        return _reply()

    async def go():
        state["started"] = started = asyncio.Event()
        task = asyncio.ensure_future(p.acall(slow, {"model": "m", "messages": []}))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert p.counts["cancelled"] == 1 and p.counts["failed_requests"] == 0


# --------------------------------------------------------------------------- #
# the real openai client honours the per-request timeout
# --------------------------------------------------------------------------- #
#: How long the fake endpoint waits before it answers, by the request's model name: "warm"
#: answers at once (the client's warm-up call), "slow" long after the 1 s timeout under test
#: (so the timeout, and nothing else, ends that call), and any other model after
#: _REPLY_AFTER_S, well inside the 10 s timeout that must let its reply through.
_REPLY_AFTER_S = 1.5
_REPLY_DELAY_S = {"warm": 0.0, "slow": 6.0}


@pytest.fixture
def slow_server():
    released = threading.Event()

    class _H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            # Event.wait instead of sleep: the fixture's teardown releases a reply nobody waits
            # for any more, so server_close() (which joins the handler threads) returns at once.
            released.wait(_REPLY_DELAY_S.get(body.get("model"), _REPLY_AFTER_S))
            out = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                              "choices": [{"index": 0, "finish_reason": "stop",
                                           "message": {"role": "assistant", "content": "hi"}}]}
                             ).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
            except OSError:
                pass

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        released.set()
        server.shutdown()
        server.server_close()


def test_the_openai_client_applies_the_sent_timeout(slow_server):
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=slow_server, api_key="EMPTY", max_retries=0)
    # The client's first request pays its one-time start-up cost (lazy imports, model classes,
    # the connection pool), which can take half a second on a cold interpreter. One call to the
    # endpoint's instant reply first keeps that cost out of the timed call below.
    warm = client.chat.completions.create(model="warm", messages=[
        {"role": "user", "content": "q"}])
    assert warm.choices[0].message.content == "hi"
    short = POL.RequestPolicy("TB_", env={"TB_ENABLE_THINKING": "true", "TB_MAX_TOKENS": "64",
                                          "TB_REQUEST_TIMEOUT_S": "1"})
    short.tokenize_url = None
    t0 = time.monotonic()
    with pytest.raises(openai.APITimeoutError):
        short.call(client.chat.completions.create, {"model": "slow", "messages": [
            {"role": "user", "content": "q"}]})
    elapsed = time.monotonic() - t0
    # The 1 s timeout ended the call: not before it (a shorter timeout, or another failure), and
    # not at twice it or later (a doubled or mis-sent timeout; no timeout, or openai's own 600 s
    # default, would have waited for the reply due at 6 s and returned it). About 1 s of margin
    # over the warm ~1.0 s call.
    assert 0.9 <= elapsed < 2.0, elapsed
    assert short.counts["failed_requests"] == 1 and short.counts["timed_out"] == 1
    assert short.failed_by_error == {"APITimeoutError": 1}
    long = POL.RequestPolicy("TB_", env={"TB_ENABLE_THINKING": "true", "TB_MAX_TOKENS": "64",
                                         "TB_REQUEST_TIMEOUT_S": "10"})
    long.tokenize_url = None
    # A reply due after 1.5 s gets through a 10 s timeout (so the 1 s above was the timeout sent,
    # not a fixed short one).
    reply = long.call(client.chat.completions.create, {"model": "m", "messages": [
        {"role": "user", "content": "q"}]})
    assert reply.choices[0].message.content == "hi" and long.counts["replies"] == 1
    assert long.counts["failed_requests"] == 0


# --------------------------------------------------------------------------- #
# hosts
# --------------------------------------------------------------------------- #
def test_both_images_carry_label_version_2():
    assert MB.REQUEST_POLICY_VERSION == "2"
    label = f'LABEL {MB.REQUEST_POLICY_LABEL}="2"'
    assert label in _read("gbench", "docker", "mcp_bench.Dockerfile")
    assert label in _read("docker", "toolbench.Dockerfile")


def test_the_request_log_merges_failures_over_processes(tmp_path):
    path = tmp_path / "log.jsonl"
    settings = {"enable_thinking": True, "max_tokens": 65536, "temperature": None,
                "request_timeout_s": 9392}
    recs = [{"settings": settings, "counts": {"replies": 3, "failed_requests": 2, "timed_out": 1},
             "failed_by_error": {"APITimeoutError": 1, "InternalServerError 503": 1},
             "max_tokens_sent": [65536]},
            {"settings": settings, "counts": {"replies": 1, "failed_requests": 1, "timed_out": 1},
             "failed_by_error": {"APITimeoutError": 1}, "max_tokens_sent": [30000]}]
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    log = MB.read_model_request_log(str(path))
    assert log["counts"] == {"replies": 4, "failed_requests": 3, "timed_out": 2}
    assert log["failed_by_error"] == {"APITimeoutError": 2, "InternalServerError 503": 1}
    assert log["settings"] == settings and log["processes"] == 2


def _log_for(counts=None, by_error=None):
    """A request log whose settings are the ones the docker command asked for."""
    def log(cmd):
        env = dict(a.split("=", 1) for a in cmd if "=" in a and a.split("=", 1)[0].isupper())
        prefix = "MCP_" if "MCP_ENABLE_THINKING" in env else "TB_"
        raw_max, raw_temp = env.get(prefix + "MAX_TOKENS"), env.get(prefix + "TEMPERATURE")
        raw_timeout = env.get(prefix + "REQUEST_TIMEOUT_S")
        settings = {"enable_thinking": env[prefix + "ENABLE_THINKING"] == "true",
                    "max_tokens": int(raw_max) if raw_max else None,
                    "temperature": float(raw_temp) if raw_temp else None,
                    "request_timeout_s": int(raw_timeout) if raw_timeout else None}
        base_counts = {"replies": 10, "cut_at_max_tokens": 0, "cut_in_reasoning": 0,
                       "fitted_to_context": 0, "refitted_after_rejection": 0,
                       "failed_requests": 0, "timed_out": 0, "rejected_over_context": 0,
                       "cancelled": 0}
        base_counts.update(counts or {})
        return [{"settings": settings, "counts": base_counts,
                 "failed_by_error": by_error or {}, "max_tokens_sent": [1024]}]
    return log


_CONNECTED = {f"S{i}": {"status": "success", "tools": 3, "error": ""} for i in range(28)}


@pytest.fixture
def mb_run(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI)
    for k in MB._REQUIRED_KEYS:
        monkeypatch.setenv(k, "y" * 8 + k.lower())
    for k in ("GBENCH_MCP_BENCH_DISTRACTION", "GBENCH_MCP_BENCH_DISABLE_STABILITY",
              "GBENCH_SHARD", "GBENCH_MCP_BENCH_TEMPERATURE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(MB, "check_mcp_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(MB, "provisioning_probe", lambda image, relay_env=None: {
        "servers": [], "unavailable": [], "unbuilt": [], "network_up": True,
        "connections": _CONNECTED})
    # the judge's Gemini relay (tests/test_mcp_credential_isolation_and_dns.py covers it): a
    # stand-in, docker is stubbed
    monkeypatch.setattr(MB, "_start_judge_relay", lambda network: types.SimpleNamespace(
        client_env=lambda: {"GEMINI_API_KEY": "gbench-gemini-relay-" + "0" * 48},
        stop=lambda: None, record=lambda: {}))
    # the credentialed-servers container (tests/test_mcp_wildclawbench_credentials_and_cascade.py
    # covers it): a stand-in
    monkeypatch.setattr(MB, "start_credentialed_servers",
                        lambda image, base, network: types.SimpleNamespace(
                            client_env=lambda: {"MCP_CREDENTIALED_RELAY": "172.17.0.1:10",
                                                "MCP_CREDENTIALED_TOKEN": "t0k"},
                            stop=lambda: None, record=lambda: {}))
    runs = []

    def go(log, **kwargs):
        def fake_run(cmd, **kw):
            if cmd[:2] == ["docker", "run"]:
                runs.append(cmd)
                out = cmd[cmd.index("-v") + 1].split(":")[0]
                metrics = {"input_schema_compliance": 1.0, "valid_tool_name_rate": 1.0,
                           "tool_call_success_rate": 1.0, "task_completion_score": 5.0,
                           "tool_selection_score": 5.0,
                           "planning_effectiveness_and_efficiency_score": 5.0}
                with open(os.path.join(out, "mcp_bench_results.json"), "w") as fh:
                    json.dump(metrics, fh)
                with open(os.path.join(out, "subset_manifest.json"), "w") as fh:
                    json.dump({"unavailable_servers": [], "tasks_kept": 104}, fh)
                with open(os.path.join(out, MB.HARNESS_LOG), "w") as fh:
                    json.dump({"version": MB.HARNESS_LOG_VERSION, "tasks": [
                        {"task_id": "t1", "file": "f.json", "status": "completed",
                         "kinds": []}]}, fh)
                with open(os.path.join(out, MB.REQUEST_LOG), "w") as fh:
                    fh.write("".join(json.dumps(r) + "\n" for r in log(cmd)))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(MB.subprocess, "run", fake_run)
        return MB.run_mcp_bench("m", "http://x/v1", **kwargs)
    return go, runs


def _env_names(cmd):
    return [cmd[i + 1] for i, a in enumerate(cmd) if a == "-e"]


def test_mcp_bench_thinking_sends_a_timeout_sized_to_the_budget(mb_run):
    go, runs = mb_run
    res = go(_log_for(), enable_thinking=True, max_output_tokens=65536)
    want = base.request_timeout_s(65536)
    assert want == int(base.REQUEST_TIMEOUT_FLOOR_S + 65536 / max(1.0, base.MIN_DECODE_TOK_S))
    assert f"MCP_REQUEST_TIMEOUT_S={want}" in _env_names(runs[0])
    assert res["model_request_timeout_s"] == want
    assert res["model_request_settings"]["request_timeout_s"] == want
    assert f"({want} s to read the reply)" in res["max_output_tokens_note"]
    assert res["status"] == "success" and res["harness_errors"] == []
    # without --thinking: no timeout sent, the harness client's own applies
    res = go(_log_for(), max_output_tokens=65536)
    assert not any(n.startswith("MCP_REQUEST_TIMEOUT_S") for n in _env_names(runs[1]))
    assert res["model_request_timeout_s"] is None


def test_mcp_bench_image_that_ignored_the_timeout_is_infra_required(mb_run):
    go, _ = mb_run

    def stale(cmd):
        rec = _log_for()(cmd)
        rec[0]["settings"]["request_timeout_s"] = None
        return rec

    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*applied the model request settings"):
        go(stale, enable_thinking=True, max_output_tokens=65536)


def test_mcp_bench_failed_requests_are_harness_errors(mb_run):
    go, _ = mb_run
    counts = {"failed_requests": 3, "timed_out": 2, "rejected_over_context": 1, "cancelled": 1}
    by_error = {"APITimeoutError": 2, "InternalServerError 503": 1}
    res = go(_log_for(counts, by_error), enable_thinking=True, max_output_tokens=65536)
    timeout = base.request_timeout_s(65536)
    assert res["status"] == "completed_with_errors"
    [err] = res["harness_errors"]
    assert (err["failed_requests"], err["timed_out"], err["by_error"],
            err["request_timeout_s"]) == (3, 2, by_error, timeout)
    assert res["model_replies"]["failed_requests"] == 3
    assert res["model_failed_by_error"] == by_error
    reason = res["leaderboard_comparable_reason"]
    assert ("3 model request(s) ended in an error of the served endpoint or of the connection "
            f"(2 timed out after the {timeout} s request timeout sized to the --thinking budget; "
            "by error: APITimeoutError x2, InternalServerError 503 x1): harness errors, not the "
            "model's outcome") in reason
    assert "1 model request(s) rejected by the served context window" in reason
    assert ("1 model request(s) were cancelled in flight by MCP-Bench (for example at "
            "MCP-Bench's task_timeout of 5000 s") in reason
    assert res["leaderboard_comparable"] is False


def test_mcp_bench_no_think_failures_name_the_clients_own_timeout(mb_run):
    go, _ = mb_run
    res = go(_log_for({"failed_requests": 1, "timed_out": 1}, {"APITimeoutError": 1}))
    assert res["status"] == "completed_with_errors"
    assert ("1 timed out after the harness client's own timeout (the openai client's default: "
            "600 s to read the reply)") in res["leaderboard_comparable_reason"]
    # only requests left to upstream's own handling: counted and named, no harness error
    res = go(_log_for({"rejected_over_context": 2}))
    assert res["status"] == "success" and res["harness_errors"] == []
    assert "2 model request(s) rejected by the served context window" in (
        res["leaderboard_comparable_reason"])


def _tb_answer(solved):
    return {"query": "q", "answer": {"final_answer": "S" if solved else "N",
                                     "answer_details": []}}


@pytest.fixture
def tb_run(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI)
    for k in ("GBENCH_TOOLBENCH_GROUPS", "GBENCH_TOOLBENCH_REFERENCE_DIR",
              "GBENCH_TOOLBENCH_REFERENCE", "GBENCH_TOOLBENCH_EVAL_TIMES", "GBENCH_SHARD",
              "GBENCH_TOOLBENCH_SIMULATOR_MODEL", "GBENCH_TOOLBENCH_SIMULATOR_BASE",
              "GBENCH_TOOLBENCH_SIMULATOR_KEY", "GBENCH_TOOLBENCH_MAX_QUERY_COUNT",
              "GBENCH_TOOLBENCH_TEMPERATURE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(T, "check_toolbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(T, "_served_model_id", lambda endpoint, fallback: fallback)
    # the bridge network's gateway (after the metadata guard) and the image's records of its
    # tool calls: stubbed, no docker (the relays listen on loopback)
    monkeypatch.setattr(T, "_bridge_gateway", lambda image: "127.0.0.1")
    monkeypatch.setattr(T, "read_server_calls", lambda path: [])
    monkeypatch.setattr(T, "read_observations", lambda path: {})

    async def status(query, answer, sem):
        return "Solved"

    async def pref(query, cand, ref, sem):
        return "cand"

    monkeypatch.setattr(T, "_answer_status", status)
    monkeypatch.setattr(T, "_preference", pref)
    ref_root = tmp_path / "ref"
    (ref_root / T._REFERENCE_DEFAULT).mkdir(parents=True)
    for g in T._GROUPS:
        (ref_root / T._REFERENCE_DEFAULT / f"{g}.json").write_text(
            json.dumps({"1": _tb_answer(False), "2": _tb_answer(False)}))
    monkeypatch.setattr(T, "_baseline_reference_root", lambda ref, groups, image: str(ref_root))
    monkeypatch.setattr(T, "_group_test_ids",
                        lambda image, groups, limit: {g: ["1", "2"] for g in groups})
    runs = []

    def go(log, **kwargs):
        def fake_run(cmd, *a, **kw):
            if cmd[:2] == ["docker", "run"]:
                runs.append(cmd)
                out = cmd[cmd.index("-v") + 1].split(":")[0]
                os.makedirs(os.path.join(out, "candidate"), exist_ok=True)
                for g in T._GROUPS:
                    with open(os.path.join(out, "candidate", f"{g}.json"), "w") as fh:
                        json.dump({"1": _tb_answer(True), "2": _tb_answer(True)}, fh)
                with open(os.path.join(out, T.REQUEST_LOG), "w") as fh:
                    fh.write("".join(json.dumps(r) + "\n" for r in log(cmd)))
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(T.subprocess, "run", fake_run)
        return T.run_toolbench("m", "http://x:1/v1", 2, **kwargs)
    return go, runs


def test_toolbench_thinking_timeout_and_failed_requests(tb_run):
    go, runs = tb_run
    timeout = base.request_timeout_s(65536)
    res = go(_log_for({"failed_requests": 4, "timed_out": 4}, {"APITimeoutError": 4}),
             enable_thinking=True, max_output_tokens=65536)
    assert f"TB_REQUEST_TIMEOUT_S={timeout}" in _env_names(runs[0])
    assert res["model_request_timeout_s"] == timeout
    assert res["status"] == "completed_with_errors"
    assert res["harness_errors"][0]["failed_requests"] == 4
    assert ("4 model request(s) ended in an error of the served endpoint or of the connection "
            f"(4 timed out after the {timeout} s request timeout sized to the --thinking budget; "
            "by error: APITimeoutError x4): harness errors, not the model's outcome "
            "(StableToolBench received an error") in res["leaderboard_comparable_reason"]
    # a clean no-think run: no timeout sent, success
    res = go(_log_for(), max_output_tokens=65536)
    assert not any(n.startswith("TB_REQUEST_TIMEOUT_S") for n in _env_names(runs[1]))
    assert res["status"] == "success" and res["model_request_timeout_s"] is None


def test_toolbench_patched_request_carries_the_timeout_and_counts_a_failure(tmp_path,
                                                                           monkeypatch):
    sys.path.insert(0, _HERE)
    try:
        from test_delegated_harness_request_settings import _UPSTREAM_CHATGPT
    finally:
        sys.path.remove(_HERE)
    target = tmp_path / "chatgpt_function_model.py"
    target.write_text(_UPSTREAM_CHATGPT)
    patch_src = _read("docker", "toolbench_patch_llm.py").replace(
        'PATH = "/stb/toolbench/inference/LLM/chatgpt_function_model.py"', f"PATH = {str(target)!r}")
    exec(compile(patch_src, "toolbench_patch_llm.py", "exec"), {"__name__": "__main__"})
    monkeypatch.setitem(sys.modules, "termcolor", types.SimpleNamespace(colored=lambda *a, **k: ""))
    monkeypatch.syspath_prepend(_TB_DOCKER)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("TB_ENABLE_THINKING", "true")
    monkeypatch.setenv("TB_MAX_TOKENS", "65536")
    monkeypatch.setenv("TB_REQUEST_TIMEOUT_S", "9392")
    monkeypatch.setenv("TB_REQUEST_LOG", str(tmp_path / "log.jsonl"))
    monkeypatch.delenv("TB_ENDPOINT", raising=False)
    monkeypatch.delenv("TB_TEMPERATURE", raising=False)
    monkeypatch.delitem(sys.modules, "chatgpt_function_model", raising=False)
    import importlib
    mod = importlib.import_module("chatgpt_function_model")
    sent = []

    def create(**kw):
        sent.append(kw)
        raise APITimeoutError("Request timed out.")

    class _OpenAI:
        def __init__(self, base_url=None, api_key=None):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    monkeypatch.setattr(mod, "OpenAI", _OpenAI)
    out = mod.chat_completion_request("EMPTY", "http://127.0.0.1:1/v1",
                                      [{"role": "user", "content": "q"}], model="google/gemma")
    # upstream turns the error into its error reply; the count sees the failed request
    assert out == {"error": "Request timed out.", "total_tokens": 0}
    [kw] = sent
    assert kw["timeout"].read == 9392.0 and kw["max_tokens"] == 65536
    mod._GBENCH_POLICY.dump()
    rec = json.loads((tmp_path / "log.jsonl").read_text())
    assert rec["counts"]["failed_requests"] == 1 and rec["counts"]["timed_out"] == 1
    assert rec["settings"]["request_timeout_s"] == 9392


def test_docs_describe_the_timeout_and_the_failure_counts():
    for name in ("mcp_bench.md", "toolbench.md"):
        doc = " ".join(_read("docs", "evals", name).split())
        assert "`org.gbench.model-request-policy=2`" in doc
        assert "`model_request_timeout_s`" in doc and "`model_failed_by_error`" in doc
        assert "`failed_requests`" in doc and "`rejected_over_context`" in doc
        assert "`completed_with_errors`" in doc
        assert "is not refitted and is counted as a failed request" in doc
    for src in (MB.__doc__, T.__doc__):
        flat = " ".join(src.split())
        assert "model_request_timeout_s" in flat and "completed_with_errors" in flat


# --------------------------------------------------------------------------- #
# medxpertqa: the official cap a --thinking run does not apply is recorded as not applied
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("knob,sent", [(None, base.THINKING_MIN_OUTPUT_TOKENS), (4096, 4096),
                                       (65536, 65536)])
def test_medxpertqa_thinking_records_what_was_sent(monkeypatch, tmp_path, knob, sent):
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    saved = dict(base._RUN_KNOBS)
    base._RUN_KNOBS.clear()
    if knob is not None:
        base._RUN_KNOBS["max_output_tokens"] = knob

    def _rows(path, name=None, **k):
        if name != "Text":
            return []
        return [{"id": "Text-0", "question": "Why?\nAnswer Choices: (A) a (B) b", "label": "B",
                 "options": {"A": "a", "B": "b"}, "question_type": "Reasoning", "images": []}]

    seen = []

    async def _send(**kw):
        seen.append(kw["max_output_tokens"])
        return base.Reply(text="The answer is B.", tool_calls=None, finish_reason="stop",
                          completion_tokens=5)

    try:
        with mock.patch("datasets.load_dataset", _rows), \
                mock.patch.object(base, "_send_single_request", side_effect=_send):
            res = medxpertqa.run_medxpertqa("m", "http://x/v1", enable_thinking=True)
    finally:
        base._RUN_KNOBS.clear()
        base._RUN_KNOBS.update(saved)
    # both turns carry the run's budget (else the thinking floor), never the 2048 or an 8192
    assert seen == [sent, sent]
    rec = res["official_sampling"]
    assert rec["max_tokens"] == 2048 and rec["max_tokens_applied"] is False
    assert rec["max_tokens_sent"] == sent
    assert rec["max_tokens_note"] == base.REASONING_CAP_NOTE
    assert "reasoning_settings" not in rec


@pytest.fixture(autouse=True)
def _metadata_guard_answers_blocked(monkeypatch):
    """Docker is stubbed in these tests: the cloud metadata guard the suites call before starting a
    container (container_guard.require_metadata_blocked) answers "unreachable" without a probe.
    tests/test_guard_tool_suites.py checks the suites' calls to it."""
    from gbench.runners.eval_suites import container_guard
    monkeypatch.setattr(container_guard, "require_metadata_blocked",
                        lambda *a, **k: container_guard.UNREACHABLE)
