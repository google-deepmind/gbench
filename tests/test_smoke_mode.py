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

"""gbench --preflight and gbench --smoke (gbench.smoke).

2026-10-09 smoke run: five of the seven hard failures of the 84-suite --eval-limit 5 run were
missing keys, credentials and images that a check before generation would have named, and
mcp_bench / mcp_atlas reported theirs only at the end of their lane. 2026-10-10: skillsbench
recorded thinking_applied=False under --thinking (the switch never reached the server), seen only
by reading its result. The preflight table names each missing prerequisite in seconds; the smoke
verdict checks, per suite, that a reply came back, that every model request carried the run's
temperature, thinking switch and token budget on the wire (a recording tap in front of the
endpoint), that an agent's action ran (a follow-up request) or the sample was scored.

No docker, no model: checks are stubbed and the tap talks to a local fake server.
"""

import http.server
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from gbench import smoke
from gbench.smoke import Check, Expected


# --------------------------------------------------------------------------- expected shape
def test_expected_shape_follows_the_temperature_rule():
    assert smoke.expected_shape("gsm8k", thinking=True, temperature=None, max_output_tokens=65536,
                                env={}) == Expected(1.0, True, 65536, "thinking")
    assert smoke.expected_shape("gsm8k", thinking=False, temperature=None, max_output_tokens=10,
                                env={}).temperature == 0.0
    assert smoke.expected_shape("gsm8k", thinking=True, temperature=0.6, max_output_tokens=10,
                                env={}).temperature == 0.6
    e = smoke.expected_shape("gsm8k", thinking=True, temperature=0.6, max_output_tokens=10,
                             env={"GBENCH_GSM8K_TEMPERATURE": "0.3"})
    assert e.temperature == 0.3 and e.temperature_source == "env:GBENCH_GSM8K_TEMPERATURE"
    m = smoke.expected_shape("mmlu", thinking=True, temperature=None, max_output_tokens=10,
                             env={})
    assert m.temperature == 0.0 and m.thinking is False


# --------------------------------------------------------------------------- request shape
def _req(**kw):
    rec = {"t": kw.pop("t", 1.0), "path": "/v1/chat/completions", "temperature": 1.0,
           "max_tokens": 65536, "enable_thinking": True, "conversation": "c1", "messages": 2,
           "status": 200, "bytes": 100}
    rec.update(kw)
    return rec


EXP = Expected(1.0, True, 65536, "thinking")


def test_shape_mismatches_name_each_kind_with_its_count():
    recs = [_req(), _req(temperature="absent"), _req(temperature=0.0),
            _req(enable_thinking="absent"), _req(enable_thinking="absent"),
            _req(max_tokens=131072), _req(max_tokens=1, temperature="absent")]
    bad = smoke.shape_mismatches(recs, EXP, "gsm8k")
    assert "no temperature sent (a temperature is always sent) (1 request)" in bad
    assert "temperature 0.0 sent, expected 1 (thinking) (1 request)" in bad
    assert ("--thinking is on but chat_template_kwargs.enable_thinking was not sent "
            "(2 requests)") in bad
    assert "max_tokens 131072 sent, above --max-output-tokens 65536 (1 request)" in bad
    assert len(bad) == 4                       # the one-token probe is not checked


def test_shape_allows_smaller_budgets_official_caps_and_no_think_defaults():
    assert smoke.shape_mismatches([_req(max_tokens=8192), _req(max_tokens=None)], EXP) == []
    assert smoke.shape_mismatches([_req(max_tokens=70000)], EXP, "x", [70000]) == []
    nothink = Expected(0.0, False, 65536, "no-think")
    assert smoke.shape_mismatches([_req(temperature=0.0, enable_thinking="absent"),
                                   _req(temperature=0.0, enable_thinking=False)], nothink) == []
    assert smoke.shape_mismatches([_req(temperature=0.0, enable_thinking=True)], nothink) == \
        ["--thinking is off but chat_template_kwargs.enable_thinking=true was sent (1 request)"]
    # /v1/completions carries no chat template switch; log-probability suites skip it too
    assert smoke.shape_mismatches([_req(path="/v1/completions", enable_thinking="absent")],
                                  EXP) == []


def test_follow_up_needs_an_answered_request_then_a_longer_conversation():
    assert smoke.follow_up_seen([_req(t=1, messages=2), _req(t=2, messages=4)])
    assert not smoke.follow_up_seen([_req(t=1, messages=2, status=500), _req(t=2, messages=4)])
    assert not smoke.follow_up_seen([_req(t=1, messages=2), _req(t=2, messages=2,
                                                                  conversation="c2")])


# --------------------------------------------------------------------------- verdict
def _result(**kw):
    res = {"eval_name": "gsm8k", "status": "success", "accuracy": 100.0, "total_questions": 1,
           "sample_traces": [{"is_correct": True}]}
    res.update(kw)
    return res


def test_verdict_stages_in_order():
    pre_fail = Check("gsm8k", False, "image missing")
    v = smoke.smoke_verdict("gsm8k", expected=EXP, preflight=pre_fail)
    assert (v["verdict"], v["stage"], v["reason"]) == ("FAIL", "pre-check", "image missing")
    v = smoke.smoke_verdict("gsm8k", expected=EXP, result=None, exit_code=1,
                            log_tail="Traceback ...")
    assert v["stage"] == "result" and "exited 1" in v["reason"] and "Traceback" in v["reason"]
    v = smoke.smoke_verdict("gsm8k", expected=EXP,
                            result=_result(status="error", failed=True,
                                           error="[INFRA REQUIRED] gsm8k: no dataset"))
    assert v["stage"] == "pre-check"
    v = smoke.smoke_verdict("gsm8k", expected=EXP, result=_result(), tap=[])
    assert v["stage"] == "reply" and "no model request" in v["reason"]
    v = smoke.smoke_verdict("gsm8k", expected=EXP, result=_result(),
                            tap=[_req(status=None, bytes=0, error="ConnectionRefusedError")])
    assert v["stage"] == "reply" and "ConnectionRefusedError" in v["reason"]
    v = smoke.smoke_verdict("gsm8k", expected=EXP, result=_result(),
                            tap=[_req(temperature=0.0)])
    assert v["stage"] == "request shape"
    v = smoke.smoke_verdict("gsm8k", expected=EXP, result=_result(accuracy=None,
                                                                  sample_traces=[]),
                            tap=[_req()])
    assert v["stage"] == "scoring"
    v = smoke.smoke_verdict("gsm8k", expected=EXP, result=_result(), tap=[_req()])
    assert v["verdict"] == "PASS" and v["replies"] == 1


def test_agent_verdict_needs_the_follow_up_request():
    res = _result(eval_name="terminal_bench")
    v = smoke.smoke_verdict("terminal_bench", expected=EXP, result=res, tap=[_req()])
    assert (v["verdict"], v["stage"]) == ("FAIL", "tool execution")
    v = smoke.smoke_verdict("terminal_bench", expected=EXP, result=res,
                            tap=[_req(t=1), _req(t=2, messages=4)])
    assert v["verdict"] == "PASS"
    # the relay ended the episode at the follow-up (it did not forward it): its record says so
    relayed = _result(relay={"relays_to": "http://127.0.0.1:1/v1", "requests_relayed": 2,
                             "completion_replies": 1, "smoke_second_requests": 1,
                             "smoke": {"replies_per_conversation": 1, "smoke_stops": 1,
                                       "smoke_second_requests": 1}})
    v = smoke.smoke_verdict("swe_bench_pro", expected=EXP, result=relayed, tap=[_req()])
    assert v["verdict"] == "PASS"
    # the relay's own shape check (after its body_fields) also fails the request shape
    relayed["relay"]["shape_violations"] = {"enable_thinking": 3}
    relayed["relay"]["shape_check"] = {"violations": {"enable_thinking": 3}, "examples": [
        {"field": "enable_thinking", "expected": True, "sent": "<absent>"}]}
    v = smoke.smoke_verdict("swe_bench_pro", expected=EXP, result=relayed, tap=[_req()])
    assert v["stage"] == "request shape" and "enable_thinking x3" in v["reason"]


def test_stamp_smoke_withholds_the_accuracy_and_keeps_the_scored_evidence():
    res = smoke.stamp_smoke(_result())
    assert res["smoke"] is True and res["accuracy"] is None and res["smoke_scored"] is True
    assert res["leaderboard_comparable"] is False
    assert smoke.scored_evidence(res) == (True, "accuracy computed over the sample")
    unscored = smoke.stamp_smoke(_result(accuracy=None, sample_traces=[{"is_correct": None}]))
    assert unscored["smoke_scored"] is False


def test_model_relays_leave_out_gemini_relays():
    res = {"gemini_relay": {"relays_to": {"native": "https://g"}, "requests_relayed": 3},
           "x": [{"model_relay": {"relays_to": "http://h/v1", "requests_relayed": 1}}]}
    assert [r["relays_to"] for r in smoke.model_relays(res)] == ["http://h/v1"]


# --------------------------------------------------------------------------- preflight
def test_preflight_table_and_status(tmp_path):
    def suite_check(s, _url):
        if s == "slow":
            time.sleep(5)
        return Check(s, s != "bad", "image 'x' not found" if s == "bad" else "")
    report = smoke.run_preflight(
        ["good", "bad", "slow"], "http://h/v1", tmp_path, timeout_s=0.5, suite_check=suite_check,
        global_checks=[lambda: Check("docker daemon", True, "server 29", fatal=True),
                       lambda: Check("Gemini key", False, "unset")])
    by = {c.name: c for c in report.checks}
    assert by["good"].ok and not by["bad"].ok and not by["slow"].ok
    assert "did not finish within 0 s" in by["slow"].reason or "within 1 s" in by["slow"].reason
    assert not report.ok and report.global_failed == []   # the Gemini key is not fatal
    table = smoke.format_preflight(report)
    assert "FAIL    image 'x' not found" in table and table.startswith("PREFLIGHT: 2 PASS, 3 FAIL")


def test_check_suite_reports_a_raising_check_and_a_missing_gemini_key(monkeypatch):
    monkeypatch.setattr(smoke, "_registry", lambda: {
        "gsm8k": lambda _u: (_ for _ in ()).throw(RuntimeError("boom")),
        "simpleqa": lambda _u: (True, "")})
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    c = smoke.check_suite("gsm8k", None)
    assert not c.ok and c.reason == "RuntimeError: boom"
    c = smoke.check_suite("simpleqa", None)
    assert not c.ok and "GEMINI_API_KEY" in c.reason


def test_every_registered_check_names_a_builtin_suite():
    from gbench.runners.eval_suites import SUITES
    assert smoke.PREFLIGHT_CHECKS <= set(SUITES)
    assert smoke.GEMINI_SUITES <= set(SUITES) and smoke.AGENT_SUITES <= set(SUITES)


def test_bridge_url_maps_loopback_to_the_docker_host_alias():
    assert smoke.bridge_url("http://127.0.0.1:8000/v1") == "http://host.docker.internal:8000/v1"
    assert smoke.bridge_url("http://10.0.0.5:8000/v1") == "http://10.0.0.5:8000/v1"


# --------------------------------------------------------------------------- tap
class _Upstream(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        return

    def do_GET(self):
        body = json.dumps({"data": [{"id": "m"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.path == "/v1/chat/completions"
        assert self.headers.get("Authorization") == "Bearer k"
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(3):
                self.wfile.write(f"data: {{\"i\": {i}}}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            return
        body = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_tap_forwards_unchanged_and_records_the_request_shape(tmp_path):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    tap = smoke.SmokeTap(f"http://127.0.0.1:{srv.server_address[1]}/v1",
                         tmp_path / "tap.jsonl", bind_host="127.0.0.1").start()
    try:
        models = json.loads(urllib.request.urlopen(tap.url + "/models", timeout=10).read())
        assert models["data"][0]["id"] == "m"
        for stream in (False, True):
            payload = {"model": "m", "temperature": 1.0, "max_tokens": 64, "stream": stream,
                       "chat_template_kwargs": {"enable_thinking": True},
                       "messages": [{"role": "user", "content": "x"}]}
            req = urllib.request.Request(tap.url + "/chat/completions",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": "Bearer k"})
            text = urllib.request.urlopen(req, timeout=10).read().decode()
            assert ("[DONE]" in text) if stream else (json.loads(text)["choices"])
    finally:
        tap.stop()
        srv.shutdown()
    recs = smoke.read_tap(tmp_path / "tap.jsonl")
    assert len(recs) == 2                          # GET /v1/models is forwarded, not recorded
    assert all(r["temperature"] == 1.0 and r["enable_thinking"] is True and r["max_tokens"] == 64
               and r["status"] == 200 and r["bytes"] > 0 for r in recs)
    assert recs[1]["stream"] is True and recs[0]["conversation"] == recs[1]["conversation"]
    assert smoke.shape_mismatches(recs, Expected(1.0, True, 65536)) == []


# --------------------------------------------------------------------------- CLI
def test_cli_refuses_smoke_without_a_remote_endpoint(capsys):
    from gbench.cli import main
    with pytest.raises(SystemExit) as e:
        main(["--smoke", "--evals", "gsm8k", "--max-output-tokens", "100"])
    assert e.value.code == 2
    assert "need --remote-endpoint" in capsys.readouterr().err


def test_cli_preflight_returns_the_report_status(monkeypatch):
    from gbench import cli
    seen = {}

    def fake(args, suites):
        seen["suites"] = list(suites)
        return 2, None
    monkeypatch.setattr(smoke, "run_preflight_cli", fake)
    code = cli.main(["--preflight", "--evals", "gsm8k,ifeval", "--max-output-tokens", "100"])
    assert code == 2 and seen["suites"] == ["gsm8k", "ifeval"]


def test_a_log_probability_suite_passes_on_one_token_requests_with_the_right_temperature():
    exp = smoke.expected_shape("mmlu", thinking=True, temperature=None, max_output_tokens=65536,
                               env={})
    letters = [_req(path="/v1/completions", max_tokens=1, temperature=0.0,
                    enable_thinking="absent", t=i) for i in range(3)]
    res = _result(eval_name="mmlu")
    assert smoke.smoke_verdict("mmlu", expected=exp, result=res, tap=letters)["verdict"] == "PASS"
    letters[0]["temperature"] = 1.0
    v = smoke.smoke_verdict("mmlu", expected=exp, result=res, tap=letters)
    assert v["stage"] == "request shape" and "temperature 1.0 sent, expected 0" in v["reason"]
    # a one-token probe next to real answers is neither shape-checked nor needed as the reply
    probe = _req(max_tokens=1, temperature="absent", enable_thinking=False, t=0)
    assert smoke.smoke_verdict("gsm8k", expected=EXP, result=_result(),
                               tap=[probe, _req(t=1)])["verdict"] == "PASS"


# --------------------------------------------------------------------------- 2026-10-11 review
def test_preflight_checks_the_suites_of_one_family_one_after_another(tmp_path):
    # 2026-10-11 review: bfcl's and bfcl_v4_agentic's checks wrote and unlinked the same
    # <project root>/.gbench_write_probe at once; bfcl_v4_agentic failed "not writable" in 2 of
    # 3 real `gbench --preflight --evals all` runs
    lock = threading.Lock()
    active, peak = {"bfcl": 0, "other": 0}, {"bfcl": 0, "other": 0}

    def check(suite, _url):
        fam = "bfcl" if suite in ("bfcl", "bfcl_v4_agentic", "bfcl_live") else "other"
        with lock:
            active[fam] += 1
            peak[fam] = max(peak[fam], active[fam])
        time.sleep(0.2)
        with lock:
            active[fam] -= 1
        return Check(suite, True)
    report = smoke.run_preflight(["bfcl", "bfcl_v4_agentic", "bfcl_live", "gsm8k", "ifeval"],
                                 None, tmp_path, suite_check=check, global_checks=[])
    assert report.ok
    assert peak["bfcl"] == 1 and peak["other"] == 2


def test_a_log_probability_suite_takes_the_cli_temperature():
    # mmlu.py / wmdp.py call resolve_temperature(..., thinking=False): env, then --temperature
    e = smoke.expected_shape("mmlu", thinking=True, temperature=0.7, max_output_tokens=10, env={})
    assert e.temperature == 0.7 and e.thinking is False
    assert smoke.shape_mismatches([_req(temperature=0.7, max_tokens=1)], e, "mmlu",
                                  include_probes=True) == []


def test_server_token_total_sums_generation_and_prompt_counters():
    text = ('vllm:prompt_tokens_total{engine="0",model_name="m"} 1.5e+02\n'
            'vllm:prompt_tokens_total{engine="1",model_name="m"} 50\n'
            'vllm:generation_tokens_total{engine="0",model_name="m"} 7\n'
            'vllm:prompt_tokens_cached_total{engine="0",model_name="m"} 999\n')
    assert smoke.server_token_total(text) == 207.0
    assert smoke.server_token_total("# nothing") is None


class _FakeTap:
    def __init__(self, entries):
        self.entries, self.aborted = entries, []

    def silent(self):
        return self.entries

    def abort(self, entry, why):
        self.aborted.append(why)


class _Kid:
    suite = "gsm8k"


def _watch(tap, pages, hang_s=300.0):
    t = {"now": 0.0}
    feed = iter(pages)
    w = smoke.SmokeWatch({"gsm8k": tap}, "http://x/metrics", hang_s,
                         fetch=lambda _u: next(feed), clock=lambda: t["now"])
    return w, t


def test_watch_ends_a_silent_request_only_while_the_server_generates_nothing():
    stream = {"stream": True, "path": "/v1/chat/completions"}
    whole = {"stream": False, "path": "/v1/chat/completions"}
    page = lambda n: f'vllm:generation_tokens_total{{engine="0"}} {n}\n'    # noqa: E731
    # the server keeps generating (other suites): a silent request waits
    tap = _FakeTap([(400.0, whole)])
    w, t = _watch(tap, [page(n) for n in range(100)])
    for step in range(60):
        t["now"] = step * 11.0
        assert w(_Kid()) is None
    assert tap.aborted == []
    # the counters stand still for hang_s: a hang, the request aborted
    tap = _FakeTap([(400.0, whole)])
    w, t = _watch(tap, [page(5)] * 100)
    reasons = []
    for step in range(40):
        t["now"] = step * 11.0
        reasons.append(w(_Kid()))
    hit = [r for r in reasons if r]
    assert hit and hit[0].startswith("hang:") and "not streamed" in hit[0]
    assert tap.aborted and reasons.index(hit[0]) * 11.0 >= 300.0
    # no readable counters: only a streamed reply can show a hang
    tap = _FakeTap([(400.0, whole)])
    w, _t = _watch(tap, [""] * 10)
    assert w(_Kid()) is None
    tap = _FakeTap([(400.0, stream)])
    w, _t = _watch(tap, [""] * 10)
    assert "streamed reply" in w(_Kid())
    # a request silent for less than hang_s is never looked at
    tap = _FakeTap([(100.0, stream)])
    w, _t = _watch(tap, [""] * 10)
    assert w(_Kid()) is None


class _Hanging(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        return

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        time.sleep(20)                              # never answers within the test


def test_tap_abort_ends_a_hung_request_and_records_the_hang(tmp_path):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Hanging)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    activity = tmp_path / ".activity"
    tap = smoke.SmokeTap(f"http://127.0.0.1:{srv.server_address[1]}/v1",
                         tmp_path / "tap.jsonl", bind_host="127.0.0.1",
                         activity_path=activity).start()
    out = {}

    def client():
        req = urllib.request.Request(tap.url + "/chat/completions", data=json.dumps(
            {"model": "m", "temperature": 1.0, "max_tokens": 8,
             "messages": [{"role": "user", "content": "x"}]}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=15).read()
            out["code"] = 200
        except urllib.error.HTTPError as e:
            out["code"] = e.code
        except Exception as e:                                           # noqa: BLE001
            out["code"] = repr(e)
    th = threading.Thread(target=client)
    th.start()
    try:
        for _ in range(100):
            if tap.silent():
                break
            time.sleep(0.05)
        (secs, entry), = tap.silent()
        assert entry["stream"] is False and secs >= 0
        tap.abort(entry, "hang: test")
        th.join(10)
    finally:
        tap.stop()
        srv.shutdown()
    assert out.get("code") == 504
    rec, = smoke.read_tap(tmp_path / "tap.jsonl")
    assert rec["hang"] == "hang: test" and rec["status"] is None
    assert tap.silent() == []


def test_tap_touches_the_activity_file_while_replies_flow(tmp_path):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    activity = tmp_path / ".activity"
    tap = smoke.SmokeTap(f"http://127.0.0.1:{srv.server_address[1]}/v1",
                         tmp_path / "tap.jsonl", bind_host="127.0.0.1",
                         activity_path=activity).start()
    try:
        req = urllib.request.Request(tap.url + "/chat/completions", data=json.dumps(
            {"model": "m", "messages": [{"role": "user", "content": "x"}]}).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer k"})
        urllib.request.urlopen(req, timeout=10).read()
    finally:
        tap.stop()
        srv.shutdown()
    assert activity.exists()


def test_verdict_names_a_hang_or_the_deadline():
    v = smoke.smoke_verdict("gsm8k", expected=EXP, result=None, tap=[_req()],
                            stopped="hang: a model request ... (GBENCH_SMOKE_HANG_S=300)")
    assert (v["verdict"], v["stage"]) == ("FAIL", "hang")
    v = smoke.smoke_verdict("gsm8k", expected=EXP, result=_result(), tap=[_req()],
                            stopped="deadline: the run's 2700 s limit")
    assert (v["verdict"], v["stage"]) == ("FAIL", "deadline")
    assert smoke.smoke_verdict("gsm8k", expected=EXP, preflight=Check("gsm8k", False, "x"),
                               stopped="deadline")["stage"] == "pre-check"


def test_stopping_the_tap_ends_the_requests_still_in_flight(tmp_path):
    # a suite stopped by the deadline or a hang leaves its request at the server; the tap ends it
    # when the suite's process is gone (vLLM aborts a request whose client left)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Hanging)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    tap = smoke.SmokeTap(f"http://127.0.0.1:{srv.server_address[1]}/v1",
                         tmp_path / "tap.jsonl", bind_host="127.0.0.1").start()

    def client():
        req = urllib.request.Request(tap.url + "/chat/completions", data=json.dumps(
            {"model": "m", "temperature": 1.0, "messages": []}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=15).read()
        except Exception:                                                # noqa: BLE001
            pass
    th = threading.Thread(target=client, daemon=True)
    th.start()
    try:
        for _ in range(100):
            if tap.silent():
                break
            time.sleep(0.05)
        assert tap.silent()
        t0 = time.monotonic()
        tap.stop()
        for _ in range(100):
            if smoke.read_tap(tmp_path / "tap.jsonl"):
                break
            time.sleep(0.05)
    finally:
        srv.shutdown()
    rec, = smoke.read_tap(tmp_path / "tap.jsonl")
    assert "suite ended" in rec["error"] and time.monotonic() - t0 < 10


def test_an_agent_harness_error_caused_by_the_smoke_stop_is_not_a_failure():
    """2026-10-11 smoke run: swe_bench_pro and swe_bench_multilingual reported the relay's
    GBENCH_SMOKE_STOP (a 400 on the follow-up request) as their own error, and the verdict failed
    them at "harness" before reading the relay's follow-up evidence."""
    relay = {"relays_to": "http://127.0.0.1:8000/v1", "requests_relayed": 1,
             "smoke": {"smoke_second_requests": 1},
             "smoke_second_requests": 1, "shape_violations": {}}
    tap = [_req(status=200, bytes=100)]
    errored = _result(eval_name="swe_bench_multilingual", status="error", accuracy=None,
                      error="the agent could not run on any of 1 instance(s) (BadRequestError: "
                            "GBENCH_SMOKE_STOP)", relay=relay)
    v = smoke.smoke_verdict("swe_bench_multilingual", expected=EXP, result=errored, tap=tap)
    assert v["verdict"] == "PASS", v
    # without the follow-up evidence the harness error still fails
    no_follow = dict(errored, relay={"relays_to": "http://127.0.0.1:8000/v1",
                                     "requests_relayed": 1,
                                     "smoke": {"smoke_second_requests": 0}})
    v = smoke.smoke_verdict("swe_bench_multilingual", expected=EXP, result=no_follow, tap=tap)
    assert (v["verdict"], v["stage"]) == ("FAIL", "harness")
    # an INFRA REQUIRED error is never excused
    infra = dict(errored, error="[INFRA REQUIRED] image missing")
    v = smoke.smoke_verdict("swe_bench_multilingual", expected=EXP, result=infra, tap=tap)
    assert (v["verdict"], v["stage"]) == ("FAIL", "pre-check")


# --------------------------------------------------------------------------- run-size defaults
def test_smoke_defaults_cover_the_measured_critical_path(monkeypatch):
    """2026-10-11 smoke runs, same command (--thinking, keyless endpoint, --eval-limit 1, same
    terminal_bench task): their wall time is terminal_bench's, which no pool size shortens
    (docs/smoke.md, Wall time). smoke-10-11-031629: 37 s of preflight plus 2,284.7 s.
    smoke-10-11-022420: 37 s plus at least 2,470.2 s (SIGTERM in Harbor's verifier), projected
    2,505 s from its last model request. The deadline is at least 1.5 x the slower path, and the
    pool size is 12."""
    assert smoke.SMOKE_PARALLEL == 12
    assert smoke.DEFAULT_DEADLINE_S == 3900.0
    assert smoke.DEFAULT_DEADLINE_S >= 1.5 * (37.0 + 2505.0)
    monkeypatch.delenv("GBENCH_SMOKE_DEADLINE_S", raising=False)
    assert smoke.smoke_deadline_s() == 3900.0
    monkeypatch.setenv("GBENCH_SMOKE_DEADLINE_S", "900")
    assert smoke.smoke_deadline_s() == 900.0


def test_smoke_cli_runs_the_default_pool_size_under_the_default_deadline(monkeypatch, tmp_path):
    """Without --parallel-suites a smoke run's pool runs SMOKE_PARALLEL suites at once, and the
    pool's deadline is GBENCH_SMOKE_DEADLINE_S less the time the preflight took."""
    from types import SimpleNamespace
    from gbench import pool as pool_mod
    seen = {}
    monkeypatch.delenv("GBENCH_SMOKE_DEADLINE_S", raising=False)
    monkeypatch.setattr(smoke, "run_preflight", lambda suites, base_url, root: smoke.PreflightReport(
        [Check(s, True) for s in suites], 0.1))
    monkeypatch.setattr(pool_mod, "load_durations", lambda *_a, **_k: {})
    monkeypatch.setattr(pool_mod, "record_durations", lambda *_a, **_k: None)
    monkeypatch.setattr(pool_mod, "suite_argv", lambda _parser, _argv: (lambda _s: []))

    def fake_run_pool(plan, base_argv, root, opts, **_kw):
        seen.update(parallel=plan.parallel, opts_parallel=opts.parallel,
                    deadline_s=opts.deadline_s)
        return {"suites": {}}
    monkeypatch.setattr(pool_mod, "run_pool", fake_run_pool)
    args = SimpleNamespace(remote_endpoint="http://127.0.0.1:9/v1", results_dir=str(tmp_path),
                           parallel_suites=None, eval_thinking=True, temperature=None,
                           max_output_tokens=100)
    smoke.run_smoke_cli(args, [], None, ["gsm8k"])
    assert seen["parallel"] == seen["opts_parallel"] == smoke.SMOKE_PARALLEL == 12
    assert smoke.DEFAULT_DEADLINE_S - 60 < seen["deadline_s"] <= smoke.DEFAULT_DEADLINE_S


def test_an_agent_suite_request_without_max_tokens_fails_the_shape_check():
    """2026-10-11: terminal_bench and aider_polyglot PASSed with every request at max_tokens
    null (their relay did not start on a keyless endpoint, so the run's cap never applied)."""
    relay = {"relays_to": "http://127.0.0.1:8000/v1", "requests_relayed": 2,
             "smoke": {"smoke_second_requests": 1}, "smoke_second_requests": 1}
    res = _result(eval_name="terminal_bench", relay=relay)
    v = smoke.smoke_verdict("terminal_bench", expected=EXP, result=res,
                            tap=[_req(max_tokens=None), _req(max_tokens=None, messages=4)])
    assert (v["verdict"], v["stage"]) == ("FAIL", "request shape")
    assert "no max_tokens sent on 2 agent request(s)" in v["reason"]
    v = smoke.smoke_verdict("terminal_bench", expected=EXP, result=res,
                            tap=[_req(), _req(messages=4)])
    assert v["verdict"] == "PASS", v


def test_an_infra_required_error_naming_the_smoke_stop_is_not_a_failure():
    """2026-10-11 rerun: terminal_bench raised [INFRA REQUIRED] because every trial ended on the
    relay's GBENCH_SMOKE_STOP; the relay saw the follow-up request, so the wiring worked."""
    relay = {"relays_to": "http://127.0.0.1:8000/v1", "requests_relayed": 2,
             "smoke": {"smoke_second_requests": 1}, "smoke_second_requests": 1}
    res = _result(eval_name="terminal_bench", status="error", accuracy=None, relay=relay,
                  error="[INFRA REQUIRED] terminal_bench cannot run: 5 raised BadRequestError "
                        "(e.g. BadRequestError: OpenAIException - GBENCH_SMOKE_STOP)")
    v = smoke.smoke_verdict("terminal_bench", expected=EXP, result=res,
                            tap=[_req(), _req(messages=4)])
    assert v["verdict"] == "PASS", v


def test_a_raised_agent_suite_passes_on_the_relays_attached_at_failure():
    """2026-10-11 rerun: terminal_bench raised [INFRA REQUIRED] on the smoke stop before it built
    its result, so the result kept no relay record and its error text was cut at "GBENCH_SMOK".
    The eval runner now attaches the records of the relays the suite started
    (relays_at_failure, swebench_common.relays_started_since)."""
    res = _result(eval_name="terminal_bench", status="error", failed=True, accuracy=None,
                  error="[INFRA REQUIRED] terminal_bench cannot run: every one of the 5 trial(s) "
                        "failed (e.g. BadRequestError: OpenAIException - GBENCH_SMOK",
                  relays_at_failure=[{"relays_to": "http://127.0.0.1:8000/v1",
                                      "requests_relayed": 5, "smoke_second_requests": 5,
                                      "smoke": {"smoke_second_requests": 5}}])
    v = smoke.smoke_verdict("terminal_bench", expected=EXP, result=res,
                            tap=[_req(), _req(messages=4)])
    assert v["verdict"] == "PASS", v


def test_relays_started_since_returns_the_new_relays_records():
    from gbench.runners.eval_suites import swebench_common as C
    mark = C.relay_count()
    relay = C.EndpointRelay("http://127.0.0.1:9/v1", "127.0.0.1").start()
    try:
        recs = C.relays_started_since(mark)
        assert len(recs) == 1 and recs[0]["relays_to"] == "http://127.0.0.1:9/v1"
    finally:
        relay.stop()
