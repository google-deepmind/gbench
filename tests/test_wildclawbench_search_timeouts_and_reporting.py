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

"""wildclawbench: an agent web search that does not reach OpenClaw within its web_search timeout
(60 s since 2026-10-13; OpenClaw's default is 30 s) is a harness error (the proxy's deadline, its delivery record, the launcher's count of
aborted calls in the transcript); the model relay records the max_tokens OpenClaw sent, the
replies cut off at it and the endpoint's token usage; a timed-out stop of the leftover processes
is classified by what still runs; the result's summary row reconciles with the headline; the
per-task artifacts are kept and the root-owned workdir removed. Docker, Gemini and the network are
stubbed; files go to tmp only."""

import json
import os
import subprocess
import sys
import threading
import time
import types
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gbench.runners.eval_suites import wildclawbench as W

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "gbench", "docker"))
import wildclawbench_cascade_judge as J  # noqa: E402
import wildclawbench_run as RUN  # noqa: E402


def _lines(path):
    return [json.loads(x) for x in open(path).read().splitlines()] if os.path.exists(path) else []


@pytest.fixture
def wd(monkeypatch, tmp_path):
    monkeypatch.setenv("WILDCLAW_WORKDIR", str(tmp_path))
    J.reset_usage()
    J._SEARCH_TOOL["module"] = None
    yield tmp_path
    J._SEARCH_TOOL["module"] = None


def _search_stub(backend="gemini:gemini-3.8-flash", delay=0.0, seen=None):
    def search_with_fallback(query, max_results=8, deadline=None):
        if seen is not None:
            seen.append({"query": query, "deadline": deadline, "at": time.monotonic()})
        time.sleep(delay)
        if backend.startswith("gemini:"):
            return [{"title": "T", "url": "https://e.org", "snippet": "S"}], backend
        return [{"title": "error", "url": "", "snippet": "no answer before the deadline"}], backend
    return types.SimpleNamespace(search_with_fallback=search_with_fallback)


# --------------------------------------------------------------------------- #
# The proxy: a deadline inside OpenClaw's timeout, and an answer only when delivered in time.
# --------------------------------------------------------------------------- #
def test_the_deadline_is_inside_openclaws_web_search_timeout():
    # 2026-10-13: 60 s, raised from OpenClaw's 30 s default (tests/test_wildclawbench_search_timeout.py)
    assert J.AGENT_WEB_SEARCH_TIMEOUT_S == RUN.AGENT_WEB_SEARCH_TIMEOUT_S == 60
    assert 0 < J.WEB_SEARCH_DEADLINE_S < J.AGENT_WEB_SEARCH_TIMEOUT_S


def test_a_lookup_gets_the_deadline_from_when_the_request_arrived(wd):
    seen = []
    J._SEARCH_TOOL["module"] = _search_stub(seen=seen)
    code, _body = J.brave_search_reply("q=x", "172.17.0.4", started=1000.0)
    assert code == 200 and seen[0]["deadline"] == 1000.0 + J.WEB_SEARCH_DEADLINE_S
    assert J.last_search_model() == "gemini-3.8-flash"
    # the answer is not recorded before the reply is delivered
    assert _lines(wd / J.GEMINI_ANSWERS) == []


def test_a_lookup_that_missed_the_deadline_is_a_502_and_a_harness_event(wd):
    J._SEARCH_TOOL["module"] = _search_stub(backend="deadline-passed")
    code, body = J.brave_search_reply("q=x", "172.17.0.4")
    assert code == 502 and body["error"]["code"] == "SEARCH_FAILED"
    [ev] = _lines(wd / J.HARNESS_EVENTS)
    assert ev["kind"] == "agent_web_search" and "deadline-passed" in ev["detail"]
    assert J.last_search_model() is None


@pytest.mark.parametrize("delivered, elapsed, answered", [
    (True, 3.0, True),                       # delivered in time: an answer
    (True, 31.0, True),                      # past the old 30 s default, inside 60 s (2026-10-13)
    (True, 61.0, False),                     # after OpenClaw's timeout: it had aborted the call
    (False, 3.0, False),                     # the connection was gone
])
def test_settle_records_an_answer_only_when_the_agent_got_it(wd, delivered, elapsed, answered):
    J._SEARCH_TOOL["module"] = _search_stub()
    J.brave_search_reply("q=x", "172.17.0.4", started=100.0)
    J.settle_web_search("172.17.0.4", 5000.0, 100.0, 200, delivered, now=100.0 + elapsed)
    answers, events = _lines(wd / J.GEMINI_ANSWERS), _lines(wd / J.HARNESS_EVENTS)
    if answered:
        assert [(a["kind"], a["model"], a["start"]) for a in answers] == [
            ("agent_web_search", "gemini-3.8-flash", 5000.0)]
        assert events == []
    else:
        assert answers == []
        [ev] = events
        assert ev["kind"] == "agent_web_search" and ev["service_failure"] is True
        assert ev["start"] == 5000.0 and "aborted" in ev["detail"]


def test_a_failed_lookup_is_not_settled_twice(wd):
    J._SEARCH_TOOL["module"] = _search_stub(backend="all-grounding-models-failed")
    code, _ = J.brave_search_reply("q=x", "c")
    J.settle_web_search("c", 1.0, 0.0, code, True, now=1.0)
    assert len(_lines(wd / J.HARNESS_EVENTS)) == 1 and _lines(wd / J.GEMINI_ANSWERS) == []


def test_the_proxy_records_a_late_answer_as_a_harness_event(wd, monkeypatch):
    monkeypatch.setattr(J, "AGENT_WEB_SEARCH_TIMEOUT_S", 0.2)
    J._SEARCH_TOOL["module"] = _search_stub(delay=0.4)
    server = ThreadingHTTPServer(("127.0.0.1", 0), J._Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/res/v1/web/search?q=slow"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=10) as r:
            assert r.status == 200
        deadline = time.time() + 5
        while not _lines(wd / J.HARNESS_EVENTS) and time.time() < deadline:
            time.sleep(0.05)
    finally:
        server.shutdown()
        server.server_close()
    [ev] = _lines(wd / J.HARNESS_EVENTS)
    assert ev["kind"] == "agent_web_search" and "past OpenClaw" in ev["detail"]
    assert _lines(wd / J.GEMINI_ANSWERS) == []


# --------------------------------------------------------------------------- #
# The launcher: aborted web_search calls in the transcript.
# --------------------------------------------------------------------------- #
def _msg(ts, message):
    return json.dumps({"type": "message", "timestamp": ts, "message": message})


def _call(ts, call_id):
    return _msg(ts, {"role": "assistant", "content": [
        {"type": "toolCall", "id": call_id, "name": "web_search", "arguments": {"query": "q"}}]})


def _result(ts, call_id, text, error=True, tool="web_search"):
    return _msg(ts, {"role": "toolResult", "toolCallId": call_id, "toolName": tool,
                     "content": [{"type": "text", "text": text}], "isError": error})


def test_aborted_web_search_calls_are_counted_from_the_transcript(tmp_path):
    chat = tmp_path / "chat.jsonl"
    chat.write_text("\n".join([
        _call("2026-09-28T22:15:24.072Z", "a"),
        _result("2026-09-28T22:16:24.079Z", "a", RUN.WEB_SEARCH_ABORTED),     # 60 s: counted
        _call("2026-09-28T22:16:00.000Z", "b"),
        _result("2026-09-28T22:16:05.000Z", "b", RUN.WEB_SEARCH_ABORTED),     # 5 s: not
        _result("2026-09-28T22:17:00.000Z", "zz", RUN.WEB_SEARCH_ABORTED),    # call unknown
        _call("2026-09-28T22:18:00.000Z", "c"),
        _result("2026-09-28T22:18:40.000Z", "c", "results"),                   # not an error
        _call("2026-09-28T22:19:00.000Z", "d"),
        _result("2026-09-28T22:19:40.000Z", "d", RUN.WEB_SEARCH_ABORTED, tool="web_fetch"),
        "not json",
    ]) + "\n")
    assert RUN.count_web_search_aborts(str(chat)) == 2
    assert RUN.count_web_search_aborts(str(tmp_path / "missing.jsonl")) == 0


def _index(monkeypatch, tmp_path, tids):
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path))
    index = {t: {"category": "04_Search_Retrieval", "modality": "pure-text", "path": None}
             for t in tids}
    monkeypatch.setattr(RUN, "_selected_task_ids", lambda _i: list(tids))
    monkeypatch.setattr(RUN, "network_rounds", lambda: [])
    return index


def _score(tmp_path, tid, score, usage=None):
    d = tmp_path / "output" / "openclaw" / "04_Search_Retrieval" / tid / "m_1"
    d.mkdir(parents=True)
    (d / "score.json").write_text(json.dumps({"overall_score": score}))
    (d / "usage.json").write_text(json.dumps(usage or {
        "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "request_count": 3,
        "elapsed_time": 60.0}))
    return d


def test_a_task_with_an_aborted_search_is_left_out_of_the_headline(monkeypatch, tmp_path):
    tids = ["t_a", "t_b"]
    index = _index(monkeypatch, tmp_path, tids)
    for tid, sc in zip(tids, (0.0, 0.8)):
        _score(tmp_path, tid, sc)
    windows = [{"task_id": "t_a", "start": 0.0, "end": 100.0, "raised": False, "ips": [],
                "grading": [], "web_search_aborted": 9},
               {"task_id": "t_b", "start": 0.0, "end": 100.0, "raised": False, "ips": [],
                "grading": []}]
    (tmp_path / RUN.TASK_WINDOWS).write_text("\n".join(json.dumps(w) for w in windows) + "\n")
    summary = RUN._summarize(str(tmp_path / "output"), index)
    rows = {r["task_id"]: r for r in summary["per_task"]}
    assert rows["t_a"]["harness_errors"] == ["agent_web_search"]
    assert rows["t_a"]["web_search_aborted"] == 9
    assert summary["web_search_aborted_calls"] == 9
    assert summary["global_mean"] == pytest.approx(0.8) and summary["n_headline_tasks"] == 1


# --------------------------------------------------------------------------- #
# The model relay: max_tokens sent, replies cut off at it, token usage.
# --------------------------------------------------------------------------- #
def test_reply_scan_reads_a_stream_that_ends_at_the_cap_while_reasoning():
    scan = J.ReplyScan(streamed=True)
    stream = (b'data: {"choices": [{"delta": {"reasoning_content": "hm"}}]}\n\n'
              b'data: {"choices": [{"delta": {}, "finish_reason": "length"}]}\n\n'
              b'data: [DONE]\n\n')
    for i in range(0, len(stream), 7):          # split mid-line, as a socket may deliver it
        scan.feed(stream[i:i + 7])
    assert scan.done() == {"finish_reason": "length", "content": False, "reasoning": True,
                           "usage": None}


def test_reply_scan_reads_a_json_reply_with_usage():
    scan = J.ReplyScan(streamed=False)
    scan.feed(json.dumps({"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 5, "completion_tokens": 2,
                                    "total_tokens": 7}}).encode())
    assert scan.done() == {"finish_reason": "stop", "content": True, "reasoning": False,
                           "usage": {"prompt_tokens": 5, "completion_tokens": 2,
                                     "total_tokens": 7}}


def test_request_shape_reads_the_cap_and_the_stream_flag():
    assert J.request_shape(b'{"max_tokens": 4096, "stream": true}') == (4096, True)
    assert J.request_shape(b'{"max_completion_tokens": 512}') == (512, False)
    assert J.request_shape(b'{"messages": []}') == (None, False)
    assert J.request_shape(b"not json") == (None, False)
    assert J.request_shape(None) == (None, False)


class _Upstream(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b'data: {"choices": [{"delta": {"content": "par"}}]}\n\n'
                         b'data: {"choices": [{"delta": {}, "finish_reason": "length"}]}\n\n'
                         b'data: [DONE]\n\n')
        self.wfile.flush()


def test_the_relay_records_each_chat_request(wd, monkeypatch):
    up = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), J._Handler)
    for s in (up, proxy):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    monkeypatch.setenv("GBENCH_MODEL_BASE_URL", f"http://127.0.0.1:{up.server_address[1]}/v1")
    monkeypatch.setenv("GBENCH_MODEL_API_KEY", "sk-endpoint")
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy.server_address[1]}/model/v1/chat/completions",
            method="POST", data=json.dumps({"max_tokens": 8192, "stream": True}).encode(),
            headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=10) as r:
            body = r.read()
        assert b"finish_reason" in body                  # passed on unchanged
    finally:
        for s in (proxy, up):
            s.shutdown()
            s.server_close()
    [rec] = _lines(wd / J.MODEL_REQUESTS)
    assert rec["max_tokens"] == 8192 and rec["stream"] is True and rec["status"] == 200
    assert rec["finish_reason"] == "length" and rec["content"] is True and rec["usage"] is None
    assert "sk-endpoint" not in json.dumps(rec)


def test_model_request_fields_and_usage():
    reqs = [{"max_tokens": 8192, "finish_reason": "length", "content": False,
             "usage": {"prompt_tokens": 10, "completion_tokens": 8192}},
            {"max_tokens": None, "finish_reason": "length", "content": True,
             "usage": {"prompt_tokens": 3, "completion_tokens": 40}}]
    f = RUN.model_request_fields(reqs)
    assert f["model_requests"] == 2 and f["max_tokens_sent"] == [8192, None]
    assert f["replies_cut_at_max_tokens"] == {"in_reasoning": 1, "in_content": 1}
    assert f["relay_output_tokens"] == 8232 and f["relay_input_tokens"] == 13
    row = {"output_tokens": 0, "cost_usd": 0.0, "agent_request_count": 2}
    assert RUN.usage_fields(row, f) == (8232, None, "model relay (endpoint usage)")
    # no usage from the endpoint and OpenClaw's tally 0: not reported, never 0
    none = RUN.model_request_fields([{"max_tokens": None, "usage": None}])
    assert RUN.usage_fields(row, none) == (None, None, None)
    assert RUN.usage_fields({"output_tokens": 55, "cost_usd": 0.12}, none) == (
        55, 0.12, "agent (usage.json)")
    # no model traffic at all: 0 is the truth
    assert RUN.usage_fields({"output_tokens": 0, "agent_request_count": 0},
                            RUN.model_request_fields([])) == (0, None, None)


def test_the_summary_records_caps_cut_replies_and_missing_usage(monkeypatch, tmp_path):
    index = _index(monkeypatch, tmp_path, ["t_a"])
    _score(tmp_path, "t_a", 1.0)
    (tmp_path / RUN.TASK_WINDOWS).write_text(json.dumps(
        {"task_id": "t_a", "start": 0.0, "end": 100.0, "raised": False, "ips": [],
         "grading": []}) + "\n")
    (tmp_path / RUN.MODEL_REQUESTS).write_text("\n".join(json.dumps(r) for r in [
        {"time": 10.0, "client": None, "max_tokens": 16384, "finish_reason": "length",
         "content": False, "usage": None},
        {"time": 20.0, "client": None, "max_tokens": 16384, "finish_reason": "stop",
         "content": True, "usage": None},
        {"time": 500.0, "client": None, "max_tokens": 16384}]) + "\n")
    summary = RUN._summarize(str(tmp_path / "output"), index)
    [row] = summary["per_task"]
    assert row["model_requests"] == 2 and row["max_tokens_sent"] == [16384]
    assert row["replies_cut_at_max_tokens"] == {"in_reasoning": 1, "in_content": 0}
    assert row["output_tokens"] is None and row["cost_usd"] is None
    assert summary["model_requests"] == 3 and summary["model_requests_unattributed"] == 1
    assert summary["max_tokens_sent"] == [16384]
    assert summary["output_tokens_total"] is None and summary["tasks_without_token_usage"] == 1
    assert summary["avg_cost_usd"] is None


# --------------------------------------------------------------------------- #
# The launcher: a stop exec that times out or fails, classified by what still runs.
# --------------------------------------------------------------------------- #
def _docker(stop, top):
    def run(cmd, **kw):
        if cmd[:2] == ["docker", "top"]:
            if isinstance(top, BaseException):
                raise top
            return top
        if cmd[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(cmd, 0, "true\n", "")
        if cmd[-3:] == ["python3", "-c", "pass"]:
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if isinstance(stop, BaseException):
            raise stop
        return stop
    return run


def _top(*lines):
    return subprocess.CompletedProcess([], 0, "PID COMMAND\n" + "".join(f"{x}\n" for x in lines),
                                       "")


_TIMEOUT = subprocess.TimeoutExpired(["docker"], 120)
_DAEMON = subprocess.CompletedProcess([], 1, "", "Error response from daemon")


@pytest.mark.parametrize("stop, top, cause", [
    # only the pre-agent processes, the OpenClaw gateway and gbench's own helper: not the agent's
    (_TIMEOUT, _top("1 sleep infinity", "20 node openclaw gateway",
                    f"30 python3 -c # {RUN.HELPER_MARK} import json"), "infra"),
    (_DAEMON, _top("1 sleep infinity"), "infra"),
    # a fork storm the agent left: the agent's doing
    (_TIMEOUT, _top("1 sleep infinity", *[f"{100 + i} bash -c :" for i in range(50)]), "model"),
    (_DAEMON, _top("1 sleep infinity", "20 node openclaw gateway", "21 python3 loop.py"), "model"),
    # docker top cannot list them either: ambiguous, left as the model's
    (_TIMEOUT, subprocess.TimeoutExpired(["docker", "top"], 60), "model"),
    (_TIMEOUT, subprocess.CompletedProcess([], 1, "", "Error"), "model"),
])
def test_a_timed_out_or_failed_stop_is_classified_by_what_still_runs(stop, top, cause):
    res = RUN.stop_leftover_processes("t1", [[1, 5]], run=_docker(stop, top))
    assert res["ok"] is False and res["cause"] == cause
    assert res["stop_note"]


def test_the_helper_scripts_carry_the_mark():
    assert RUN.HELPER_MARK in RUN._PROCESS_SNAPSHOT_SCRIPT
    assert RUN.HELPER_MARK in RUN._STOP_LEFTOVER_SCRIPT
    assert RUN.processes_beyond_snapshot("c", [[1, 5]], run=_docker(None, _top(
        "1 sleep", "2 node gw", "3 x", f"4 python3 {RUN.HELPER_MARK}"))) == 2


# --------------------------------------------------------------------------- #
# The runner: the summary row, the cap reason, tokens, artifacts, the workdir.
# --------------------------------------------------------------------------- #
def test_headline_counts_reconcile_with_the_headline():
    summary = {"n_tasks": 3, "n_headline_tasks": 2, "per_task": [
        {"task_id": "a", "overall_score": 0.5, "harness_errors": []},
        {"task_id": "b", "overall_score": 0.25, "harness_errors": []},
        {"task_id": "c", "overall_score": 0.0, "harness_errors": ["agent_web_search"]}]}
    counts = W.headline_counts(summary)
    assert counts["total_questions"] == 3 and counts["effective_n"] == 2
    assert counts["correct_answers"] == 0.75 and "solved-equivalent" in \
        counts["correct_answers_note"]


def _run(monkeypatch, tmp_path, summary_extra=None, thinking=False, workdir_files=None,
         results_dir=None):
    harness = tmp_path / "harness"
    harness.mkdir()
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    real_mkdtemp = W.tempfile.mkdtemp

    def mkdtemp(prefix="", dir=None):
        return str(workdir) if prefix.startswith("gbench_wildclawbench_") else \
            real_mkdtemp(prefix=prefix, dir=dir)
    monkeypatch.setattr(W, "check_wildclawbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(W, "_harness_dir", lambda: str(harness))
    monkeypatch.setattr(W, "free_port", lambda preferred=None: preferred)
    monkeypatch.setattr(W.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(W.container_guard, "require_metadata_blocked", lambda *a, **k: None)
    for k in ("GBENCH_WILDCLAWBENCH_TASK_NETWORK", "GBENCH_WILDCLAWBENCH_JUDGE_PORT",
              "GBENCH_SHARD"):
        monkeypatch.delenv(k, raising=False)
    if results_dir:
        monkeypatch.setenv("GBENCH_RESULTS_DIR", str(results_dir))
    else:
        monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    calls = []

    def fake_run(cmd, *a, **k):
        calls.append(list(cmd))
        if cmd[:3] == ["docker", "run", "--rm"] and "--name" in cmd:
            summary = {"n_tasks": 2, "n_scored": 2, "n_headline_tasks": 2, "global_mean": 0.4,
                       "n_missing": 0, "judge_fallback_tasks": 0, "n_search_setup_failures": 0,
                       "per_task": [{"task_id": "a", "overall_score": 0.8, "harness_errors": []},
                                    {"task_id": "b", "overall_score": 0.0,
                                     "harness_errors": []}]}
            summary.update(summary_extra or {})
            (workdir / "wildclawbench_summary.json").write_text(json.dumps(summary))
            for rel, text in (workdir_files or {}).items():
                path = workdir / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="orchestrator log")
    monkeypatch.setattr(W.subprocess, "run", fake_run)
    result = W.run_wildclawbench("m", "http://127.0.0.1:9123/v1", enable_thinking=thinking)
    return result, calls, workdir


def test_the_result_row_and_the_cap_reason(monkeypatch, tmp_path):
    result, calls, workdir = _run(monkeypatch, tmp_path, thinking=True, summary_extra={
        "max_tokens_sent": [16384], "model_requests": 7,
        "replies_cut_at_max_tokens": {"in_reasoning": 2, "in_content": 1},
        "output_tokens_total": None, "tasks_without_token_usage": 2,
        "web_search_aborted_calls": 0})
    assert result["total_questions"] == 2 and result["effective_n"] == 2
    assert result["correct_answers"] == 0.8 and result["accuracy"] == 40.0
    assert result["max_tokens_sent"] == [16384] and result["model_requests"] == 7
    assert result["replies_cut_at_max_tokens"] == {"in_reasoning": 2, "in_content": 1}
    assert result["output_tokens_total"] is None and "never 0" in result["token_usage_note"]
    assert ("3 model repl(ies) ended at OpenClaw's own per-turn max_tokens (max_tokens 16384) on "
            "a --thinking run (2 while reasoning, 1 in the answer text)") in \
        result["leaderboard_comparable_reason"]
    run_cmd = next(c for c in calls if c[:3] == ["docker", "run", "--rm"] and "--name" in c)
    assert f"WILDCLAW_HOST_OWNER={os.getuid()}:{os.getgid()}" in run_cmd
    assert not workdir.exists()


def test_a_no_think_run_records_cut_replies_without_a_reason(monkeypatch, tmp_path):
    result, _calls, _wd = _run(monkeypatch, tmp_path, summary_extra={
        "replies_cut_at_max_tokens": {"in_reasoning": 0, "in_content": 1}})
    assert result["replies_cut_at_max_tokens"] == {"in_reasoning": 0, "in_content": 1}
    assert "per-turn max_tokens" not in result["leaderboard_comparable_reason"]


def test_the_artifacts_are_kept_capped_and_redacted(monkeypatch, tmp_path):
    secret = "sk-test-endpoint-key-0123456789abcdef"
    monkeypatch.setenv("VLLM_API_KEY", secret)
    run_dir = "output/openclaw/04_Search_Retrieval/t_a/m_1"
    big = "x" * (W.ARTIFACT_TASK_FILES["chat.jsonl"] + 1000)
    result, _calls, _wd = _run(monkeypatch, tmp_path, results_dir=tmp_path / "results",
                               workdir_files={
                                   f"{run_dir}/score.json": '{"overall_score": 0.8}',
                                   f"{run_dir}/usage.json": "{}",
                                   f"{run_dir}/chat.jsonl": f"key {secret}\n" + big,
                                   "harness_events.jsonl": "{}\n"})
    keep = result["artifacts_dir"]
    assert keep and keep.startswith(str(tmp_path / "results" / "wildclawbench_artifacts"))
    chat = open(os.path.join(keep, "tasks", "04_Search_Retrieval", "t_a", "m_1",
                             "chat.jsonl")).read()
    assert secret not in chat and "left out]" in chat
    assert len(chat) < len(big)
    assert os.path.isfile(os.path.join(keep, "tasks", "04_Search_Retrieval", "t_a", "m_1",
                                       "score.json"))
    assert os.path.isfile(os.path.join(keep, "harness_events.jsonl"))
    assert open(os.path.join(keep, "wildclawbench_orchestrator.log")).read() == "orchestrator log"


def test_remove_workdir_uses_a_cleanup_container_for_root_owned_files(monkeypatch, tmp_path):
    wd = tmp_path / "wd"
    (wd / "output").mkdir(parents=True)
    calls = []
    state = {"n": 0}
    real_rmtree = W.shutil.rmtree

    def rmtree(path, onerror=None):
        state["n"] += 1
        if state["n"] == 1:                   # the first pass meets a root-owned file
            onerror(os.remove, str(wd / "output" / "root_owned"), None)
            return
        real_rmtree(path)
    monkeypatch.setattr(W.shutil, "rmtree", rmtree)
    monkeypatch.setattr(W.subprocess, "run",
                        lambda cmd, **k: calls.append(list(cmd)) or
                        subprocess.CompletedProcess(cmd, 0, "", ""))
    assert W.remove_workdir(str(wd), "gbench-wildclawbench:316334cc") == []
    [cmd] = calls
    assert cmd[:6] == ["docker", "run", "--rm", "--network", "none", "--entrypoint"]
    assert f"{wd}:/w" in cmd and "gbench-wildclawbench:316334cc" in cmd
    assert not wd.exists()


def test_merge_keeps_the_new_fields():
    a = {"shard": "1/2", "raw_summary": {"per_task": [
        {"task_id": "a", "overall_score": 1.0, "harness_errors": []}]},
        "max_tokens_sent": [16384], "replies_cut_at_max_tokens": {"in_reasoning": 1,
                                                                  "in_content": 0},
        "model_requests": 3, "output_tokens_total": 100, "web_search_aborted_calls": 2,
        "artifacts_dir": "/r/a", "status": "success"}
    b = {"shard": "2/2", "raw_summary": {"per_task": [
        {"task_id": "b", "overall_score": 0.5, "harness_errors": ["agent_web_search"]}]},
        "max_tokens_sent": [None], "replies_cut_at_max_tokens": {"in_reasoning": 0,
                                                                 "in_content": 2},
        "model_requests": 4, "output_tokens_total": None, "artifacts_dir": "/r/b",
        "status": "completed_with_errors"}
    m = W.merge_shards([a, b])
    assert m["total_questions"] == 2 and m["effective_n"] == 1 and m["correct_answers"] == 1.0
    assert m["max_tokens_sent"] == [16384, None]
    assert m["replies_cut_at_max_tokens"] == {"in_reasoning": 1, "in_content": 2}
    assert m["model_requests"] == 7 and m["output_tokens_total"] is None
    assert m["web_search_aborted_calls"] == 2
    assert m["artifacts_dir"] is None and m["artifacts_dir_by_shard"] == {"1/2": "/r/a",
                                                                         "2/2": "/r/b"}


def test_the_guide_describes_the_timeout_the_relay_record_and_the_artifacts():
    doc = " ".join(open(os.path.join(_REPO, "docs", "evals", "wildclawbench.md"),
                        encoding="utf-8").read().split())
    for needle in ("`WEB_SEARCH_DEADLINE_S`", "`web_search_aborted_calls`",
                   "`replies_cut_at_max_tokens`", "`max_tokens_sent`", "`output_tokens_total`",
                   "### Artifacts", "`WILDCLAW_HOST_OWNER`", "`total_questions`",
                   "`model_requests.jsonl`", "`stop_note`"):
        assert needle in doc, needle
    entry = open(os.path.join(_REPO, "gbench", "docker", "wildclawbench_entrypoint.sh"),
                 encoding="utf-8").read()
    assert 'chown -R "$WILDCLAW_HOST_OWNER" "$WILDCLAW_WORKDIR"' in entry
