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
"""SWE family: harness errors are named from evidence the model cannot forge, and every rollout
runs as upstream ran it.

* run_instance.log is read without newline translation, so a "\\r" in the model's diff (echoed by
  `patch` in the APPLY_PATCH_FAIL record) cannot forge a swebench record.
* EndpointRelay: a client that closes a streamed reply ends it quietly and is counted apart from
  endpoint failures; failures carry their times (failures_between).
* multi_swe_bench: MSWE-agent's reset `apt-get update` on an archived Debian release falls back to
  archive.debian.org (as SWE-bench_Pro-os's LockedMiniSwe does), recorded per trace and per run;
  the image must carry the driver that does it.
* swe_bench_pro: the sandboxes always reach the endpoint through gbench's relay, whose failures
  are harness errors; a budget cut while mini-swe-agent was retrying a failed request is a harness
  error only when the relay saw an endpoint-side failure of that rollout's own requests in that
  window, with no reply after it.
* ui_control_osworld: a real endpoint key stays with gbench's relay; the orchestrator (docker
  socket, host network) gets the placeholder.
* Every SWE doc says why the rollout container keeps the docker socket and what that leaves open.

Hermetic: docker, the endpoint and the network are stubbed (loopback stubs only); every file is
under tmp_path.
"""

import contextlib
import http.client
import http.server
import importlib.util
import io
import json
from pathlib import Path
import os
import re
import shutil
import socketserver
import subprocess
import sys
import threading
import time
import types

import pytest

from gbench.runners.eval_suites import multi_swe_bench as M
from gbench.runners.eval_suites import swe_bench_pro as P
from gbench.runners.eval_suites import swebench_common as C
from gbench.runners.eval_suites import ui_control_osworld as U

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(REPO, "docs", "evals")
SECRET = "sk-w2s-swe-secret-9876543210"


@pytest.fixture(autouse=True)
def _scratch(monkeypatch, tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    for var in ("OPENAI_API_KEY", "GBENCH_SHARD", "GBENCH_SWE_BENCH_PRO_TASK_ENDPOINT",
                "GBENCH_OSWORLD_MODEL_API_KEY", "GBENCH_OSWORLD_MODEL_ENDPOINT",
                "GBENCH_MULTI_SWE_BENCH_MODE", "GBENCH_MULTI_SWE_BENCH_TEMPERATURE",
                "GBENCH_TEMPERATURE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GBENCH_SWE_OMP_THREADS", "0")
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_TASK_HOST", "127.0.0.1")
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_TRAJ_DIR", str(tmp_path / "pro"))
    monkeypatch.setenv("GBENCH_MULTI_SWE_BENCH_TRAJ_DIR", str(tmp_path / "mswe"))
    monkeypatch.setenv("GBENCH_MULTI_SWE_REPO_DIR", str(tmp_path / "repos"))
    monkeypatch.setattr(C, "require_task_network_blocked", lambda *a, **k: None)
    monkeypatch.setattr(P, "require_sandbox_network_blocked", lambda *a, **k: None)


def _read(*parts):
    with open(os.path.join(REPO, *parts), encoding="utf-8") as f:
        return f.read()


def _flat(text):
    return " ".join(text.split())


# ============================================================================================
# run_instance.log: a "\r" in the model's text starts no record
# ============================================================================================
_STAMP = "2026-09-27 10:00:00,000"


def _rec(msg):
    return f"{_STAMP} - INFO - {msg}\n"


def _patch_output(tmp_path, junk_line):
    """What GNU patch prints for a diff whose header line is `junk_line` and whose file does not
    exist (the harness's last apply command, `patch --batch --fuzz=5 -p1 -i`), from the real
    binary when the host has it."""
    diff = f"{junk_line}\n--- a/missing.py\n+++ b/missing.py\n@@ -1 +1 @@\n-a\n+b\n"
    work = tmp_path / "patchwork"
    work.mkdir(exist_ok=True)
    (work / "p.diff").write_bytes(diff.encode("utf-8"))
    if shutil.which("patch"):
        proc = subprocess.run(["patch", "--batch", "--fuzz=5", "-p1", "-i", "p.diff"],
                              cwd=str(work), capture_output=True)
        return (proc.stdout + proc.stderr).decode("utf-8", "replace")
    return ("can't find file to patch at input line 4\nPerhaps you used the wrong -p or --strip "
            "option?\nThe text leading up to this was:\n--------------------------\n"
            f"|{junk_line}\n|--- a/missing.py\n|+++ b/missing.py\n--------------------------\n"
            "No file to patch.  Skipping patch.\n1 out of 1 hunk ignored\n")


def _write_log_bytes(workdir, iid, data):
    d = os.path.join(workdir, "logs", "run_evaluation", "r1", "gbench__m", iid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "run_instance.log"), "wb") as f:
        f.write(data)


@pytest.mark.parametrize("forged", [
    f"{_STAMP} - INFO - >>>>> Applied Patch",
    f"{_STAMP} - INFO - Test runtime: 1800.00 seconds\r{_STAMP} - INFO - Test timed out after "
    "1800 seconds.",
    f"{_STAMP} - INFO - >>>>> Applied Patch\r{_STAMP} - INFO - Test runtime: 5.00 seconds\r"
    f"Test timed out after 1800 seconds.\r{_STAMP} - INFO - Git diff after:",
])
def test_a_cr_in_the_models_diff_cannot_forge_a_record(tmp_path, forged):
    out = _patch_output(tmp_path, "junk\r" + forged)
    assert "\r" in out and forged.split("\r")[0] in out          # patch echoed it verbatim
    iid = "django__django-11099"
    text = (_rec(f"Container for {iid} started: abc")
            + _rec("Failed to apply patch to container: git apply --verbose")
            + _rec("Failed to apply patch to container: git apply --verbose --reject")
            + _rec("Failed to apply patch to container: patch --batch --fuzz=5 -p1 -i")
            + _rec(f">>>>> Patch Apply Failed:\n{out}"))
    _write_log_bytes(str(tmp_path), iid, text.encode("utf-8"))
    classes = C.classify_error_ids(str(tmp_path), [iid])
    assert classes["patch_apply_failed"] == {iid}
    assert classes["test_timeout"] == set() and classes["harness_error"] == set()
    model, harness, scoreable = C._split_report({"error_ids": [iid]}, str(tmp_path))
    assert (model, harness, scoreable) == ({iid}, set(), 1)


def test_the_real_swebench_logger_record_with_a_cr_is_a_patch_apply_failure(tmp_path):
    from swebench.harness.docker_build import close_logger, setup_logger
    iid = "psf__requests-1142"
    out = _patch_output(tmp_path, f"junk\r{_STAMP} - INFO - >>>>> Applied Patch")
    d = tmp_path / "logs" / "run_evaluation" / "r1" / "gbench__m" / iid
    d.mkdir(parents=True)
    lg = setup_logger(iid, d / "run_instance.log")
    lg.info(f"Container for {iid} started: abc")
    lg.info(f">>>>> Patch Apply Failed:\n{out}")
    close_logger(lg)
    assert b"\r" in (d / "run_instance.log").read_bytes()
    assert C.classify_error_ids(str(tmp_path), [iid])["patch_apply_failed"] == {iid}


def test_a_log_written_with_crlf_line_ends_still_reads_its_records(tmp_path):
    iid = "i1"
    text = (_rec(">>>>> Applied Patch:") + _rec("Git diff before:") + "diff --git a/x b/x\n"
            + _rec("Test runtime: 1800.00 seconds") + _rec("Test output for i1 written to t")
            + _rec("Traceback (most recent call last):")
            + "swebench.harness.utils.EvaluationError: i1: Test timed out after 1800 seconds.\n")
    _write_log_bytes(str(tmp_path), iid, text.replace("\n", "\r\n").encode("utf-8"))
    assert C.classify_error_ids(str(tmp_path), [iid])["test_timeout"] == {iid}
    _write_log_bytes(str(tmp_path), "i2", _rec(">>>>> Patch Apply Failed:\nerror: corrupt patch")
                     .replace("\n", "\r\n").encode("utf-8"))
    assert C.classify_error_ids(str(tmp_path), ["i2"])["patch_apply_failed"] == {"i2"}


# ============================================================================================
# EndpointRelay: a client that closes a streamed reply
# ============================================================================================
def _sse_upstream(state):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                for i in range(400):
                    data = ("data: " + json.dumps({"choices": [{"index": 0, "delta": {
                        "content": f" t{i}"}, "finish_reason": None}]}) + "\n\n").encode()
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
                    self.wfile.flush()
                    state["sent"] = i + 1
                    time.sleep(0.01)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                state["disconnected"] = True

        def log_message(self, *a):
            pass
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_a_client_that_closes_a_stream_ends_it_quietly_and_is_counted_apart():
    state = {"sent": 0, "disconnected": False}
    srv = _sse_upstream(state)
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            relay = C.EndpointRelay(f"http://127.0.0.1:{srv.server_address[1]}/v1",
                                    "127.0.0.1", name="w2s-relay").start()
            host, port = relay.url.split("//")[1].split("/")[0].split(":")
            conn = http.client.HTTPConnection(host, int(port), timeout=10)
            conn.request("POST", "/v1/chat/completions",
                         body=json.dumps({"model": "m", "stream": True, "messages": []}),
                         headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            got = resp.read1(200)
            resp.close()
            conn.close()
            deadline = time.time() + 10
            while relay.record()["client_closed_streams"] < 1 and time.time() < deadline:
                time.sleep(0.05)
            relay.stop()
    finally:
        srv.shutdown()
        srv.server_close()
    rec = relay.record()
    assert got and rec["client_closed_streams"] == 1
    assert rec["upstream_errors"] == {} and rec["completion_replies"] == 0
    assert C.relay_failure_count(rec) == 0
    assert state["disconnected"] and state["sent"] < 400      # the endpoint stopped generating
    assert err.getvalue() == ""                                 # no traceback on stderr


def test_relay_failures_carry_their_times():
    relay = C.EndpointRelay("http://127.0.0.1:1/v1", "127.0.0.1")
    before = time.time()
    relay.count_failure("http_503")
    relay.count_failure("connection")
    after = time.time()
    assert relay.failures_between(before - 1, after + 1) == 2
    assert relay.failures_between(after + 5, after + 10) == 0
    assert relay.record()["upstream_errors"] == {"connection": 1, "http_503": 1}


# ============================================================================================
# multi_swe_bench: the archive.debian.org fallback of MSWE-agent's reset
# ============================================================================================
def _driver():
    spec = importlib.util.spec_from_file_location(
        "mswe_agent_run_w2s", os.path.join(REPO, "gbench", "docker", "mswe_agent_run.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_BUSTER_404 = (
    "Err:4 http://security.debian.org/debian-security buster/updates Release\n"
    "  404  Not Found [IP: 151.101.2.132 80]\n"
    "E: The repository 'http://deb.debian.org/debian buster Release' no longer has a Release "
    "file.\n")


def _fake_env_module(monkeypatch, script):
    """A stand-in for sweagent.environment.swe_env.SWEEnv: communicate() answers from `script`
    (a list of (output, returncode) per call, recorded), communicate_with_handling is upstream's
    (close + RuntimeError on a non-zero return code)."""
    calls = []

    class SWEEnv:
        def __init__(self):
            self.returncode = None
            self.closed = False

        def communicate(self, input, timeout_duration=25, no_output_timeout_duration=25):
            calls.append(input)
            out, rc = script.pop(0)
            self.returncode = rc
            return out

        def communicate_with_handling(self, input, error_msg, timeout_duration=25,
                                      no_output_timeout_duration=25, except_error_msgs=[]):
            logs = self.communicate(input, timeout_duration=timeout_duration,
                                    no_output_timeout_duration=no_output_timeout_duration)
            if self.returncode != 0:
                self.closed = True
                raise RuntimeError(f"{error_msg}: {logs}")
            return logs

        def reset(self, instance_id, apply_test_patch=False):
            for cmd in ("apt-get update", "apt-get install -y jq"):
                self.communicate_with_handling(cmd, error_msg="Failed to install")
            return None, {}
    mod = types.ModuleType("sweagent.environment.swe_env")
    mod.SWEEnv = SWEEnv
    env_pkg = types.ModuleType("sweagent.environment")
    env_pkg.swe_env = mod
    pkg = types.ModuleType("sweagent")
    pkg.environment = env_pkg
    monkeypatch.setitem(sys.modules, "sweagent", pkg)
    monkeypatch.setitem(sys.modules, "sweagent.environment", env_pkg)
    monkeypatch.setitem(sys.modules, "sweagent.environment.swe_env", mod)
    return SWEEnv, calls


def test_a_working_apt_get_update_is_left_alone(monkeypatch):
    D = _driver()
    SWEEnv, calls = _fake_env_module(monkeypatch, [("Reading package lists...", 0),
                                                   ("jq installed", 0)])
    D.install_apt_archive_fallback()
    SWEEnv().reset("o__r-1")
    assert calls == ["apt-get update", "apt-get install -y jq"]
    assert D.APT_ARCHIVE == {}


def test_an_archived_release_gets_the_archive_and_upstreams_own_apt_get_update(monkeypatch):
    D = _driver()
    SWEEnv, calls = _fake_env_module(monkeypatch, [
        (_BUSTER_404, 100), ("apt archive fallback: applied (buster)", 0),
        ("Reading package lists...", 0), ("jq installed", 0)])
    D.install_apt_archive_fallback()
    env = SWEEnv()
    env.reset("simdjson__simdjson-949")
    assert calls == ["apt-get update", D.APT_ARCHIVE_REPAIR, "apt-get update",
                     "apt-get install -y jq"]
    note = D.APT_ARCHIVE["simdjson__simdjson-949"]
    assert note["applied"] is True and note["release"] == "buster"
    assert note["apt_update_after"] == "ok" and "no longer has a Release file" in note["apt_error"]
    assert not env.closed


def test_apt_still_failing_after_the_archive_stays_upstreams_failure(monkeypatch):
    D = _driver()
    SWEEnv, calls = _fake_env_module(monkeypatch, [
        (_BUSTER_404, 100), ("apt archive fallback: applied (buster)", 0),
        ("E: Release file expired", 100)])
    D.install_apt_archive_fallback()
    env = SWEEnv()
    with pytest.raises(RuntimeError, match="^Failed to install: E: Release file expired"):
        env.reset("o__r-2")
    assert env.closed and D.APT_ARCHIVE["o__r-2"]["apt_update_after"].startswith("failed: ")


def test_another_apt_failure_takes_upstreams_path_untouched(monkeypatch):
    D = _driver()
    dns = "Temporary failure resolving 'deb.debian.org'"
    SWEEnv, calls = _fake_env_module(monkeypatch, [(dns, 100), (dns, 100)])
    D.install_apt_archive_fallback()
    with pytest.raises(RuntimeError, match="Failed to install"):
        SWEEnv().reset("o__r-3")
    assert calls == ["apt-get update", "apt-get update"]        # no repair ran
    assert D.APT_ARCHIVE == {}


def _run_repair(tmp_path, os_release, sources, extra=None):
    """APT_ARCHIVE_REPAIR as the container runs it, with /etc/os-release and /etc/apt moved under
    tmp_path (the script is otherwise unchanged)."""
    D = _driver()
    etc = tmp_path / "etc"
    (etc / "apt" / "apt.conf.d").mkdir(parents=True)
    (etc / "apt" / "sources.list.d").mkdir()
    (etc / "os-release").write_text(os_release)
    (etc / "apt" / "sources.list").write_text(sources)
    for name, text in (extra or {}).items():
        (etc / "apt" / "sources.list.d" / name).write_text(text)
    script = D.APT_ARCHIVE_REPAIR.replace("/etc/", f"{etc}/")
    assert script.count(str(etc)) == 4       # os-release, sources.list, sources.list.d, apt.conf.d
    out = subprocess.run(["sh", "-c", script[len("sh -c '"):-1]], capture_output=True, text=True)
    conf = etc / "apt" / "apt.conf.d" / "99archive-debian-valid-until"
    return (out.returncode, out.stdout.strip(), (etc / "apt" / "sources.list").read_text(),
            conf.read_text() if conf.exists() else None, etc)


def test_the_repair_on_the_buster_sources_of_the_published_images(tmp_path):
    # /etc/apt/sources.list of mswebench/simdjson_m_simdjson:pr-949 (Debian 10)
    rc, out, sources, conf, _ = _run_repair(
        tmp_path, 'ID=debian\nVERSION_ID="10"\nVERSION_CODENAME=buster\n',
        "deb http://deb.debian.org/debian buster main\n"
        "deb http://security.debian.org/debian-security buster/updates main\n"
        "deb http://deb.debian.org/debian buster-updates main\n")
    assert (rc, out) == (0, "apt archive fallback: applied (buster)")
    assert sources == ("deb http://archive.debian.org/debian buster main\n"
                       "deb http://archive.debian.org/debian-security buster/updates main\n"
                       "deb http://archive.debian.org/debian buster-updates main\n")
    assert conf.strip() == 'Acquire::Check-Valid-Until "false";'


def test_the_repair_on_jessie_drops_the_updates_suite_the_archive_lacks(tmp_path):
    rc, out, sources, conf, etc = _run_repair(
        tmp_path, 'ID=debian\nVERSION="8 (jessie)"\n',
        "deb http://httpredir.debian.org/debian jessie main\n"
        "deb http://deb.debian.org/debian jessie-updates main\n"
        "deb http://security.debian.org jessie/updates main\n",
        {"backports.list": "deb http://ftp.us.debian.org/debian/ jessie-backports main\n"})
    assert (rc, out) == (0, "apt archive fallback: applied (jessie)")
    assert sources == ("deb http://archive.debian.org/debian jessie main\n"
                       "deb http://archive.debian.org/debian-security jessie/updates main\n")
    assert (etc / "apt" / "sources.list.d" / "backports.list").read_text() == \
        "deb http://archive.debian.org/debian/ jessie-backports main\n"
    assert conf is not None


@pytest.mark.parametrize("os_release,label", [
    ("ID=ubuntu\nVERSION_CODENAME=jammy\n", "ubuntu jammy"),
    ("ID=debian\nVERSION_CODENAME=bookworm\n", "debian bookworm"),
])
def test_the_repair_leaves_other_systems_as_they_are(tmp_path, os_release, label):
    original = "deb http://archive.ubuntu.com/ubuntu/ jammy main\n"
    rc, out, sources, conf, _ = _run_repair(tmp_path, os_release, original)
    assert (rc, out) == (0, f"apt archive fallback: not applied ({label})")
    assert sources == original and conf is None


def _task(iid="simdjson__simdjson-949"):
    from gbench.runners.eval_suites import swe_agentic
    return swe_agentic.AgentTask(instance_id=iid, problem_statement="p",
                                 image="mswebench/simdjson_m_simdjson:pr-949", category="cpp",
                                 cwd="/home/simdjson",
                                 meta={"org": "simdjson", "repo": "simdjson", "number": 949})


_APT = {"applied": True, "release": "buster", "apt_error": "E: ... no longer has a Release file.",
        "apt_update_after": "ok"}


def test_the_trace_records_what_the_fallback_did(tmp_path):
    summary = {"images": {}, "instances": {"simdjson__simdjson-949": {
        "completed": True, "patch": "diff", "exit_status": "submitted", "api_calls": 3,
        "steps": 3, "step_limit_reached": False, "trajectory_file": None,
        "apt_archive": dict(_APT)}}}
    tr = M.mswe_agent_trace(0, _task(), summary, str(tmp_path), time.time())
    assert tr["agent_apt_archive"]["release"] == "buster" and not tr["agent_infra_error"]
    failed = {"images": {}, "instances": {"simdjson__simdjson-949": {"completed": False}},
              "apt_archive": {"simdjson__simdjson-949": dict(
                  _APT, apt_update_after="failed: Failed to install: E: 403")}}
    tr = M.mswe_agent_trace(0, _task(), failed, str(tmp_path), time.time())
    assert tr["agent_infra_error"] is True
    assert "pointed at archive.debian.org and apt-get update then failed" in tr["agent_error"]
    not_applied = {"images": {}, "instances": {"simdjson__simdjson-949": {"completed": False}},
                   "apt_archive": {"simdjson__simdjson-949": {
                       "applied": False, "release": "ubuntu noble", "apt_error": "E: 404"}}}
    tr = M.mswe_agent_trace(0, _task(), not_applied, str(tmp_path), time.time())
    assert "did not apply (ubuntu noble)" in tr["agent_error"]


def test_the_run_records_the_fallback(monkeypatch, tmp_path):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_swe_reference_scaffolds as H
    H._agentic_setup(monkeypatch, tmp_path, H._report((1,), (2,)))
    rollouts = M.run_mswe_agent

    def _with_apt(*a, **k):
        traces = rollouts(*a, **k)
        traces[0]["agent_apt_archive"] = dict(_APT)
        return traces
    monkeypatch.setattr(M, "run_mswe_agent", _with_apt)
    res = M.run_multi_swe_bench("m", "http://127.0.0.1:1/v1", 1, enable_thinking=False)
    fb = res["apt_archive_fallback"]
    assert (fb["instances"], fb["applied"], fb["releases"]) == (1, 1, {"buster": 1})
    assert fb["apt_update_failed_after"] == [] and len(fb["ids"]) == 1
    assert res["status"] == "success" and res["leaderboard_comparable"] is True
    assert res["agent"]["driver"].endswith(f"({M.MSWE_AGENT_DRIVER_LABEL}={M.MSWE_AGENT_DRIVER_VERSION})")


def test_the_image_carries_the_driver_gbench_checks():
    dockerfile = _read("gbench", "docker", "mswe_agent.Dockerfile")
    assert f'{M.MSWE_AGENT_DRIVER_LABEL}="{M.MSWE_AGENT_DRIVER_VERSION}"' in dockerfile
    assert 'grep -q "^def install_apt_archive_fallback"' in dockerfile
    driver = _read("gbench", "docker", "mswe_agent_run.py")
    assert "install_apt_archive_fallback()" in driver.split("def main(")[1]
    doc = _flat(_read("docs", "evals", "multi_swe_bench.md"))
    assert f"`org.gbench.mswe_agent_driver`, version `{M.MSWE_AGENT_DRIVER_VERSION}`" in doc
    assert "archive.debian.org" in doc and "`agent_apt_archive`" in doc


# ============================================================================================
# swe_bench_pro: the relay for every run, and the budget cut during an endpoint failure
# ============================================================================================
#: mini-swe-agent 2.4.6's console log around a model request the endpoint failed (captured from
#: the real agent against a stub that answered 500: tenacity's before_sleep_log record).
_RETRY = ("Retrying <unknown> in 4 seconds as it raised InternalServerError: "
          "litellm.InternalServerError: InternalServerError: OpenAIException - stub overloaded.\n")
_REPLY = "\nmini-swe-agent (step 7, $0.00):\nSubmitting.\n\n```\nls\n```\n\nTool:\n{}\n"


def test_the_pending_retry_is_read_after_the_last_reply():
    assert P.pending_endpoint_retry(_REPLY + _RETRY) == "InternalServerError"
    assert P.pending_endpoint_retry(_RETRY + _REPLY) is None           # a reply came after
    wrapped = _REPLY + "Retrying <unknown> in 8\nseconds as it raised APIConnectionError:\n" \
        "litellm.APIConnectionError: Connection refused\n"
    assert P.pending_endpoint_retry(wrapped) == "APIConnectionError"
    assert P.pending_endpoint_retry(
        _REPLY + "Retrying <unknown> in 4 seconds as it raised ValueError: bad\n") is None
    assert P.pending_endpoint_retry("") is None


class _Relay:
    def __init__(self, n, recovered=False):
        self.n, self.recovered, self.windows = n, recovered, []

    def failure_times(self, start, end, fingerprint=None):
        self.windows.append((start, end, fingerprint))
        return [1005.0] * self.n

    def succeeded_after(self, fingerprint, when):
        return self.recovered


_FP = "sha256:0123456789abcdef"


def _cut_view(retrying="InternalServerError", fingerprint=_FP):
    return {"model_outcome": True, "exit_status": "", "why": None,
            "budget_kill": {"last_event_at": 1000.0, "ended_at": 1011.0,
                            "agent_started_at": 10.0, "last_event_age_s": 11.0,
                            "retrying": retrying, "conversation_fingerprint": fingerprint}}


def test_a_budget_cut_is_a_harness_error_only_with_the_relays_count():
    relay = _Relay(0)
    assert P.budget_kill_verdict(_cut_view(), relay) is None
    assert relay.windows == [(999.0, 1012.0, _FP)]          # only this rollout's failures
    view = _cut_view()
    why = P.budget_kill_verdict(view, _Relay(3))
    assert "retrying a model request the endpoint failed (InternalServerError)" in why
    assert view["budget_kill"]["relay_failures_in_window"] == 3
    assert P.budget_kill_verdict(_cut_view(retrying=None), _Relay(3)) is None
    assert P.budget_kill_verdict(_cut_view(), None) is None
    # a reply to this conversation after the failure: the rollout went on, the model's outcome
    view = _cut_view()
    assert P.budget_kill_verdict(view, _Relay(3, recovered=True)) is None
    assert view["budget_kill"]["recovered_after_failure"] is True
    # no fingerprint to attribute a failure with: the model's outcome
    assert P.budget_kill_verdict(_cut_view(fingerprint=None), _Relay(3)) is None


#: The start of a rollout's conversation: its task prompt is what the relay's failure records
#: are attributed by (conversation_fingerprint).
_TRAJ_HEAD = [{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content": "Please solve this issue: instance_a"}]
_TASK_FP = C.conversation_fingerprint(_TRAJ_HEAD)


def _trial(root, iid, *, exit_status, log, last_ts=1790529392.76, finished="2026-09-27T17:16:43Z"):
    tdir = os.path.join(root, f"{iid}__abc")
    os.makedirs(os.path.join(tdir, "agent"), exist_ok=True)
    with open(os.path.join(tdir, "result.json"), "w") as f:
        json.dump({"task_name": iid,
                   "agent_execution": {"started_at": "2026-09-27T16:27:37Z",
                                       "finished_at": finished},
                   "verifier": {"started_at": "2026-09-27T17:16:44Z"},
                   "verifier_result": {"rewards": {"reward": 0.0}},
                   "exception_info": {"exception_type": "ApiOverloadedError",
                                      "exception_message": "Command failed (exit 124)"}}, f)
    with open(os.path.join(tdir, "agent", "mini-swe-agent.trajectory.json"), "w") as f:
        json.dump({"info": {"exit_status": exit_status, "model_stats": {"api_calls": 7}},
                   "messages": _TRAJ_HEAD + [
                       {"role": "assistant", "extra": {"timestamp": str(last_ts - 1)}},
                       {"role": "tool", "extra": {"timestamp": last_ts}}]}, f)
    with open(os.path.join(tdir, "agent", "mini-swe-agent.txt"), "w") as f:
        f.write(log)
    with open(os.path.join(tdir, "agent", "model.patch"), "w") as f:
        f.write("diff --git a/x b/x\n")
    return tdir


def test_the_trial_view_records_the_budget_cut(tmp_path):
    _trial(str(tmp_path), "instance_a", exit_status="", log=_REPLY + _RETRY)
    view = P.agent_trial_view(P.job_trials(str(tmp_path))["instance_a"])
    kill = view["budget_kill"]
    assert view["model_outcome"] is True and kill["retrying"] == "InternalServerError"
    assert kill["ended_at"] - kill["last_event_at"] == pytest.approx(kill["last_event_age_s"],
                                                                     abs=0.1)
    assert kill["conversation_fingerprint"] == _TASK_FP


def _pro_run(monkeypatch, tmp_path, *, failures, log, exit_status="", fingerprint=_TASK_FP):
    rows = [{"instance_id": "instance_a", "problem_statement": "p", "docker_image": "img",
             "repo_language": "go", "repo": "o/r"}]
    monkeypatch.setattr(P, "load_v2_rows", lambda: rows)
    monkeypatch.setattr(P, "load_hard51_ids", lambda ids: set())
    monkeypatch.setattr(P, "read_locked_agent_config",
                        lambda: ("model:\n  model_kwargs: {}\n", P.AGENT_CONFIG_SHA256))
    made = []

    class Relay(C.EndpointRelay):
        def start(self):
            made.append(self)
            return super().start()
    monkeypatch.setattr(P, "EndpointRelay", Relay)
    jobs = []

    def _harbor(workdir, args, job):
        jobs.append((job, list(args)))
        jobs_dir = args[args.index("--jobs-dir") + 1]
        root = os.path.join(jobs_dir, job)
        now = time.time()
        if job == "agent":
            for _ in range(failures):
                made[0].count_failure("http_503", fingerprint)
            _trial(root, "instance_a", exit_status=exit_status, log=log, last_ts=now - 5,
                   finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now + 1)))
        else:
            os.makedirs(os.path.join(root, "instance_a__r"), exist_ok=True)
            with open(os.path.join(root, "instance_a__r", "result.json"), "w") as f:
                json.dump({"task_name": "instance_a",
                           "verifier_result": {"rewards": {"reward": 1.0}}}, f)
        return 0, ""
    monkeypatch.setattr(P, "_run_harbor", _harbor)
    res = P.run_swe_bench_pro_v2("m", "http://127.0.0.1:8000/v1", 1, enable_thinking=True)
    return res, made, jobs


def test_pro_a_cut_during_an_endpoint_outage_is_a_harness_error(monkeypatch, tmp_path):
    res, made, jobs = _pro_run(monkeypatch, tmp_path, failures=2, log=_REPLY + _RETRY)
    tr = res["sample_traces"][0]
    assert tr["harness_error"] == C.AGENT_INFRA and tr["agent_infra_error"] is True
    assert "the endpoint failed (InternalServerError)" in tr["harness_error_reason"]
    assert tr["budget_kill"]["relay_failures_in_window"] == 2
    assert tr["agent_harbor_exception"] == "ApiOverloadedError"
    assert [j for j, _ in jobs] == ["agent"]                   # nothing to re-grade
    assert res["accuracy"] is None and res["status"] == "error"
    assert res["endpoint_failures"]["upstream_errors"] == {"http_503": 2}
    assert made and made[0]._server is None                    # stopped when the run ended


def test_pro_a_cut_without_an_endpoint_failure_stays_the_models_outcome(monkeypatch, tmp_path):
    # The same log, but the relay saw nothing: the retry record alone (text the model's commands
    # can also print) does not make the task a harness error.
    res, _, jobs = _pro_run(monkeypatch, tmp_path, failures=0, log=_REPLY + _RETRY)
    tr = res["sample_traces"][0]
    assert not tr.get("harness_error") and tr["is_correct"] is True
    assert tr["budget_kill"]["relay_failures_in_window"] == 0
    assert res["status"] == "success" and "endpoint_failures" not in res
    assert res["relay"]["upstream_errors"] == {}
    assert [j for j, _ in jobs] == ["agent", "regrade"]


def test_pro_endpoint_failures_make_the_run_completed_with_errors(monkeypatch, tmp_path):
    res, _, _ = _pro_run(monkeypatch, tmp_path, failures=1, log=_REPLY,
                         exit_status="Submitted")
    assert res["sample_traces"][0]["is_correct"] is True and res["accuracy"] == 100.0
    assert res["status"] == "completed_with_errors"
    assert "the served endpoint failed 1 relayed model request(s)" in \
        res["leaderboard_comparable_reason"]
    assert res["leaderboard_comparable"] is False


# ============================================================================================
# ui_control_osworld: the endpoint key stays with gbench's relay
# ============================================================================================
def _osworld(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_OSWORLD_VM_DIR", str(tmp_path))
    monkeypatch.setattr(U, "check_ui_control_osworld_prerequisites", lambda: (True, ""))
    seen = {}

    def fake_run(cmd, *a, **k):
        if cmd[:2] == ["docker", "run"]:
            seen["cmd"], seen["env"] = list(cmd), dict(k.get("env") or {})
            # the orchestrator writes its summary into this run's own directory
            run_dir = next(c.split("=", 1)[1] for c in cmd
                           if c.startswith("OSWORLD_RUN_DIR="))
            (Path(run_dir) / "ui_control_osworld_summary.json").write_text(json.dumps(
                {"success_rate": 0.5, "per_domain": {}, "n_selected": 2, "n_scored": 2,
                 "n_missing": 0}), encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(U.subprocess, "run", fake_run)
    return seen


def test_osworld_with_a_key_relays_it_and_the_orchestrator_gets_the_placeholder(monkeypatch,
                                                                              tmp_path):
    monkeypatch.setenv("GBENCH_OSWORLD_MODEL_API_KEY", SECRET)
    seen = _osworld(monkeypatch, tmp_path)
    res = U.run_ui_control_osworld(model_name="m", base_url="http://127.0.0.1:8000/v1")
    relay = res["model_endpoint_relay"]
    assert relay["relays_to"] == "http://127.0.0.1:8000/v1"
    assert relay["listens"].startswith("http://127.0.0.1:")
    assert f"GBENCH_MODEL_BASE_URL={relay['listens']}" in seen["cmd"]
    assert seen["env"]["GBENCH_MODEL_API_KEY"] == U.MODEL_KEY_PLACEHOLDER == "dummy"
    assert "GBENCH_OSWORLD_MODEL_API_KEY" not in seen["env"]
    assert SECRET not in json.dumps(seen) and SECRET not in json.dumps(res, default=str)
    assert relay["adds"] == "Authorization: Bearer <GBENCH_OSWORLD_MODEL_API_KEY>"
    assert res["accuracy"] == 50.0 and res["status"] == "success"


def test_osworld_without_a_key_still_goes_through_the_relay(monkeypatch, tmp_path):
    # the relay counts the endpoint's failures with or without a key; it adds none here
    seen = _osworld(monkeypatch, tmp_path)
    res = U.run_ui_control_osworld(model_name="m", base_url="http://127.0.0.1:8000/v1")
    relay = res["model_endpoint_relay"]
    assert relay["relays_to"] == "http://127.0.0.1:8000/v1"
    assert f"GBENCH_MODEL_BASE_URL={relay['listens']}" in seen["cmd"]
    assert "GBENCH_MODEL_BASE_URL=http://127.0.0.1:8000/v1" not in seen["cmd"]
    assert seen["env"]["GBENCH_MODEL_API_KEY"] == "dummy"
    assert relay["adds"] == "nothing (the endpoint has no key)"


def test_osworld_relay_stops_when_the_orchestrator_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_OSWORLD_MODEL_API_KEY", SECRET)
    monkeypatch.setenv("GBENCH_OSWORLD_VM_DIR", str(tmp_path))
    monkeypatch.setattr(U, "check_ui_control_osworld_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(U.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        returncode=1, stdout="", stderr="boom"))           # no summary written
    made = []

    class Relay(C.EndpointRelay):
        def start(self):
            made.append(self)
            return super().start()
    monkeypatch.setattr(U, "EndpointRelay", Relay)
    with pytest.raises(Exception, match="produced no summary") as info:
        U.run_ui_control_osworld(model_name="m", base_url="http://127.0.0.1:8000/v1")
    assert SECRET not in str(info.value)
    assert len(made) == 1 and made[0]._server is None        # stopped by the finally


# ============================================================================================
# every SWE doc says why the rollout container keeps the docker socket
# ============================================================================================
@pytest.mark.parametrize("name", ["multi_swe_bench", "swe_bench_pro", "swe_bench_multilingual",
                                  "copilot_bench_swe", "swe_bench_live", "swe_lancer",
                                  "ui_control_osworld"])
def test_every_swe_doc_names_the_docker_sockets_residual_risk(name):
    doc = _flat(_read("docs", "evals", f"{name}.md"))
    assert "root-equivalent" in doc and "docker daemon" in doc
    assert "gbench keeps upstream's design" in doc
    assert "maintainer decision this doc records" not in doc
