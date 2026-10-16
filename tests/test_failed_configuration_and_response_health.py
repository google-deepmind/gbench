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

"""Run accounting and the shared helpers.

* A suite that raises ([INFRA REQUIRED] or any error), times out, or returns without a score is
  a FAILED configuration: accuracy withheld (null), printed INFRA REQUIRED / ERROR / TIMEOUT
  (never 0.00%), counted as failed in summary.json, `-` in the CSV, and the run exits non-zero.
* metadata.json records the benchmark mode the CLI logs (evals only for --evals-only runs).
* base.run_eval_suite: response_health reads each trace's final health (non_convergent, not
  truncated); looping and non-convergent replies, and replies that reached the end of the context
  window when no max_tokens is sent, do not make the run completed_with_errors; status_reason
  names what did.
* container_guard tells a name that does not exist from a DNS outage.
* The Gemini relay caps chunk-size lines and the request bodies it holds at once.
No network, no docker, no model server.
"""

import csv
import json
import time
import types
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import pytest

from gbench import cli
from gbench.runners import evals
from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import container_guard as cg
from gbench.runners.eval_suites import gemini_relay as GR
from gbench.runners.eval_suites.swebench_common import infra_required

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# a raising suite is a failed configuration
# --------------------------------------------------------------------------- #
def _run_suites(tmp_path, suites):
    from gbench.core.config import BenchmarkConfig
    from gbench.core.models import ModelFormat
    from gbench.runners.eval_suites import SUITES
    config = BenchmarkConfig()
    config.evals = list(suites)
    config.remote_endpoint = "http://127.0.0.1:1/v1"
    config.batch_sizes = [1]
    config.skip_existing = False
    config.eval_max_output_tokens = 256
    config.results_dir = tmp_path / "results"
    config.initialize()
    model = mock.MagicMock()
    model.name = model.short_name = model.hf_model_id = "m"
    with mock.patch.dict(SUITES, suites), \
         mock.patch("gbench.runners.eval_suites.discover_and_register_plugins", return_value={}), \
         mock.patch("gbench.runners.evals._resolve_remote_model_id", return_value="m"), \
         mock.patch("gbench.runners.evals._server_max_model_len", return_value=None):
        results = evals.EvalsBenchmarkRunner(config).run(model, ModelFormat.HF)
    return config, results


def _ok_suite(**kw):
    return {"status": "success", "total_questions": 2, "correct_answers": 1, "accuracy": 50.0,
            "category_accuracy": {}}


def _infra_suite(**kw):
    raise infra_required("api_bank", "the API-Bank checkout is missing.", "docs/evals/api_bank.md")


def _boom_suite(**kw):
    raise ValueError("unexpected harness crash")


def _unscored_suite(**kw):
    return {"status": "failed", "total_questions": 3, "correct_answers": 0, "accuracy": 0.0,
            "category_accuracy": {}}


def test_a_raising_suite_is_a_failed_configuration_with_no_score(tmp_path, capsys):
    config, results = _run_suites(tmp_path, {"mmlu": _ok_suite, "api_bank": _infra_suite,
                                             "nestful": _boom_suite, "gsm8k": _unscored_suite})
    by = {r["eval_name"]: r for r in results}
    assert by["mmlu"].get("failed") is None and by["mmlu"]["accuracy"] == 50.0
    for name, kind in (("api_bank", "infra_required"), ("nestful", "error"),
                       ("gsm8k", "error")):
        r = by[name]
        assert r["failed"] is True and r["accuracy"] is None, name
        assert r["failure_kind"] == kind and r["accuracy_withheld_reason"], name
    # the saved result file says the same
    saved = json.loads((config.log_manager.results_dir / "eval_api_bank_m_hf.json")
                       .read_text(encoding="utf-8"))
    assert saved["failed"] is True and saved["accuracy"] is None

    capsys.readouterr()
    cli.print_results_summary(results, config)
    out = capsys.readouterr().out
    rows = {line.split()[2]: line for line in out.splitlines()
            if line.startswith("m ") and len(line.split()) > 3}
    assert "INFRA REQUIRED" in rows["API_BANK"] and "%" not in rows["API_BANK"]
    assert rows["NESTFUL"].rstrip().endswith("ERROR") and "%" not in rows["NESTFUL"]
    assert rows["GSM8K"].rstrip().endswith("ERROR") and "0.00%" not in rows["GSM8K"]
    assert "the API-Bank checkout is missing" in out
    assert "1 passed, 3 failed" in out
    summary = json.loads(next(config.log_manager.results_dir.rglob("summary.json"))
                         .read_text(encoding="utf-8"))
    assert summary["failed_configurations"] == 3 and summary["successful_configurations"] == 1

    ok = [r for r in results if not r.get("failed")]
    bad = [r for r in results if r.get("failed")]
    path = cli._save_eval_summary_csv(tmp_path, ok, bad, evals.BUILTIN_PILLARS, {})
    with open(path, newline="", encoding="utf-8") as fh:
        csv_rows = {r["Eval Suite"]: r for r in csv.DictReader(fh)}
    assert csv_rows["API_BANK"]["Accuracy (%)"] == "-"
    assert csv_rows["API_BANK"]["Status"] == "infra_required"
    assert csv_rows["NESTFUL"]["Status"] == "error" and csv_rows["MMLU"]["Accuracy (%)"] == "50.00"

    assert cli.eval_failures_exit_code(results) == cli.EXIT_HARNESS_ERROR
    assert cli.eval_failures_exit_code([r for r in results if not r.get("failed")]) is None


def test_a_suite_timeout_is_a_failed_configuration():
    def slow(**kw):
        time.sleep(2)
        return {"status": "success", "accuracy": 1.0}
    res = evals._run_with_timeout("gsm8k", slow, {}, 0.1, "m")
    assert res["status"] == "timeout" and res["failed"] is True and res["accuracy"] is None
    assert evals.failure_label(res) == "TIMEOUT"


def test_mark_unscored_leaves_a_scored_result_alone():
    scored = {"status": "completed_with_errors", "accuracy": 40.0}
    assert evals.mark_unscored(dict(scored)) == scored
    res = evals.mark_unscored({"status": "error", "accuracy": None,
                               "error": "[INFRA REQUIRED] x cannot run"})
    assert res["failed"] is True and evals.failure_label(res) == "INFRA REQUIRED"


def test_metadata_records_the_benchmark_mode_the_cli_logs():
    ns = types.SimpleNamespace
    base_args = dict(evals_only=False, evals=None, campaign=None, input_lengths=None,
                     output_lengths=None, golden_only=False, quality_only=False,
                     multimodal_only=False, text_only=False)
    assert cli._benchmark_mode(ns(**{**base_args, "evals_only": True})) == "evals only"
    assert cli._benchmark_mode(ns(**{**base_args, "evals": ["gsm8k"]})) == "evals only"
    assert cli._benchmark_mode(ns(**{**base_args, "evals": ["gsm8k"], "campaign": "x"})) == \
        "text and multimodal (for capable models)"
    assert cli._benchmark_mode(ns(**{**base_args, "quality_only": True})) == "quality only"
    assert cli._benchmark_mode(ns(**{**base_args, "golden_only": True})) == "golden set only"
    assert cli._benchmark_mode(ns(**{**base_args, "text_only": True})) == "text only"
    src = (REPO / "gbench/cli.py").read_text(encoding="utf-8")
    assert '"benchmark_mode": _benchmark_mode(args)' in src
    assert 'args.quality_only and "quality" or "performance"' not in src


# --------------------------------------------------------------------------- #
# base.run_eval_suite: response_health and the status rule
# --------------------------------------------------------------------------- #
_LOOP = " ".join(["step again we check the same thing and the value stays exactly as before now"
                  " so we"] * 12)
_LONG = " ".join(f"w{i}" for i in range(400))


def _suite(replies, **kw):
    """Run base.run_eval_suite over len(replies) samples; replies[i] is the Reply for sample i."""
    samples = [([{"role": "user", "content": f"q{i}"}], "gold", {}) for i in range(len(replies))]

    async def send(**k):
        i = int(k["messages"][0]["content"][1:])
        return replies[i]
    with mock.patch.object(base, "_send_single_request", send):
        return base.run_eval_suite(eval_name="unit_status", model_name="m",
                                   base_url="http://127.0.0.1:1/v1", concurrency=1,
                                   samples=samples, eval_fn=lambda r, g: False, **kw)


def _ok(tokens=100):
    return base.Reply("fine answer", None, "stop", completion_tokens=tokens)


def test_looping_replies_are_the_models_outcome_not_a_measurement_problem():
    replies = [_ok() for _ in range(6)] + [base.Reply(_LOOP, None, "length",
                                                      completion_tokens=5000)] * 4
    res = _suite(replies)
    assert res["looping_responses"] == 4 and res["truncated_responses"] == 4
    assert res["status"] == "success" and res["status_reason"] is None


def test_replies_cut_off_while_writing_name_the_cause():
    replies = [_ok() for _ in range(7)] + [base.Reply(_LONG, None, "length",
                                                      completion_tokens=300)] * 3
    res = _suite(replies)
    assert res["status"] == "completed_with_errors"
    assert "replies_cut_off_by_non_official_budget 3 of 10" in res["status_reason"]


def test_failed_requests_and_outages_are_named_in_the_status_reason():
    replies = [_ok() for _ in range(9)] + [base.Reply(None, None, None, error="HTTP 503")]
    res = _suite(replies)
    assert res["status"] == "completed_with_errors"
    assert res["status_reason"].startswith("failed_requests 1 (FAILED 1)")


def test_response_health_reads_the_non_convergent_relabel():
    replies = [_ok(100) for _ in range(9)] + [base.Reply(_LONG, None, "length",
                                                         completion_tokens=5000)]
    res = _suite(replies)
    assert res["non_convergent_responses"] == 1
    assert res["response_health"] == {"ok": 9, "non_convergent": 1}
    assert res["status"] == "success"


def test_a_reply_at_the_context_window_with_no_max_tokens_is_the_protocols_outcome():
    official = base.OfficialSampling(reference="unit", send_max_tokens=False)
    replies = [_ok() for _ in range(7)] + [base.Reply(_LONG, None, "length",
                                                      completion_tokens=300)] * 3
    res = _suite(replies, official_sampling=official)
    assert res["max_tokens_sent"] is None
    assert res["replies_at_context_window"] == 3
    assert res["status"] == "success" and res["status_reason"] is None


# --------------------------------------------------------------------------- #
# container_guard: a name that does not exist is not a DNS outage
# --------------------------------------------------------------------------- #
def test_a_nonexistent_name_asks_for_the_url_not_the_firewall():
    msg = cg.unresolved_message("bridge", {"no-such-host.invalid": "gaierror:-2"})
    assert "do not exist" in msg and "no-such-host.invalid" in msg
    assert "check the url or host name" in msg.lower()
    assert cg.FIREWALL_COMMAND not in msg and "cannot resolve host names" not in msg
    outage = cg.unresolved_message("bridge", {"pypi.org": "gaierror:-3"})
    assert "cannot resolve host names" in outage and cg.FIREWALL_COMMAND in outage
    both = cg.unresolved_message("bridge", {"pypi.org": "timeout",
                                            "typo.example": "nslookup:NXDOMAIN"})
    assert cg.FIREWALL_COMMAND in both and "typo.example" in both.split("Also:")[1]
    assert "pypi.org" not in both.split("Also:")[1]


def test_the_bash_probe_reports_nxdomain():
    script = cg.bash_probe_script(resolve=["example.org"])
    assert "*NXDOMAIN*" in script and "nslookup:NXDOMAIN" in script


def test_the_guard_raises_the_nonexistent_message(monkeypatch):
    cg.reset_verdicts()

    def fake_run(argv, **kw):
        out = ("GBENCH_METADATA_PROBE=unreachable timeout\n"
               "GBENCH_DNS_PROBE=fail api.typo.invalid gaierror:-2\n")
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")
    monkeypatch.setattr(cg.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as ei:
        cg.require_metadata_blocked("tau2", "img:1", "bridge", resolve=["api.typo.invalid"])
    assert "do not exist" in str(ei.value) and cg.FIREWALL_COMMAND not in str(ei.value)


# --------------------------------------------------------------------------- #
# the Gemini relay's limits
# --------------------------------------------------------------------------- #
@pytest.fixture
def relay(monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Up(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

    up = ThreadingHTTPServer(("127.0.0.1", 0), Up)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    # Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "RelayLimitKey0000000000000000000001")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL",
                       f"http://127.0.0.1:{up.server_address[1]}/v1beta/openai/")
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy",
                 "ALL_PROXY"):
        monkeypatch.delenv(name, raising=False)
    r = GR.GeminiRelay("unit", "127.0.0.1").start()
    yield r
    r.stop()
    up.shutdown()


def _raw(relay, request: bytes) -> bytes:
    import socket
    host, port = relay.origin.split("//")[1].split(":")
    with socket.create_connection((host, int(port)), timeout=10) as s:
        s.sendall(request)
        s.shutdown(socket.SHUT_WR)
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                return data
            data += chunk


def test_an_overlong_chunk_line_is_400_and_not_a_harness_error(relay):
    head = (f"POST /v1beta/openai/chat/completions HTTP/1.1\r\nHost: x\r\n"
            f"Authorization: Bearer {relay.placeholder}\r\nContent-Type: application/json\r\n"
            "Transfer-Encoding: chunked\r\n\r\n").encode()
    reply = _raw(relay, head + b"1" * (GR.MAX_CHUNK_LINE + 50))
    assert reply.split(b" ", 2)[1] == b"400"
    rec = relay.record()
    assert rec["requests_over_limits"] == {"chunk_line_too_long": 1}
    assert rec["harness_errors"] == {} and relay.harness_error_count() == 0


def test_bodies_over_the_in_flight_budget_are_503(relay, monkeypatch):
    monkeypatch.setattr(GR, "MAX_INFLIGHT_BODY_BYTES", 10)
    req = urllib.request.Request(relay.openai_base_url + "/chat/completions",
                                 data=b'{"model": "m", "messages": []}', method="POST",
                                 headers={"Authorization": f"Bearer {relay.placeholder}",
                                          "Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with pytest.raises(urllib.error.HTTPError) as ei:
        opener.open(req, timeout=10)
    assert ei.value.code == 503
    assert relay.record()["requests_over_limits"] == {"relay_busy": 1}
    monkeypatch.setattr(GR, "MAX_INFLIGHT_BODY_BYTES", 1 << 20)
    with opener.open(req, timeout=10) as r:                  # the budget was given back
        assert r.status == 200
    deadline = time.monotonic() + 5            # the handler thread gives it back after replying
    while relay._inflight and time.monotonic() < deadline:
        time.sleep(0.01)
    assert relay._inflight == 0 and relay.harness_error_count() == 0
