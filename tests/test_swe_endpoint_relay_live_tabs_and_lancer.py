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
"""SWE family: the endpoint relay, SWE-bench-Live's ten leaderboard tabs, and SWE-Lancer with its
own agent and harness.

* swebench_common.EndpointRelay (swe_bench_pro's KeyForwarder, swe_lancer's and the Windows
  tab's relay): only GET/POST requests for plain /v1/ paths reach the fixed endpoint; a request
  path cannot move it to another host or port, redirects are not followed, and the key never
  comes back to the client.
* swe_bench_live: every tab of the leaderboard (Lite, Full, Verified; C/C++, C#, Java, TS/JS,
  Go, Rust from SWE-bench-Live/MultiLang; Windows), each with its own pinned splits, harness and
  headline; --eval-categories routes the tab names; the Windows tab is infra_required off a
  Windows Docker host; an infra_required error inside a setting stops the run.
* swe_lancer: SimpleAgentSolver through run_swelancer.py's main in the local harness image,
  both task types, the official task table and images; the relay adds the key and the
  reasoning switch; the result's headlines are pass@1 and dollars earned per task type.

Hermetic: no container, no docker daemon, no network beyond loopback test servers; every cache
and output is under tmp_path.
"""

import asyncio
import http.server
import json
import os
import re
import socket
import sys
import threading
import types
from unittest import mock

import pytest

from gbench.runners.eval_suites import swe_agentic as A
from gbench.runners.eval_suites import swe_bench_live as LV
from gbench.runners.eval_suites import swe_bench_pro as P
from gbench.runners.eval_suites import swe_lancer as SL
from gbench.runners.eval_suites import swebench_common as C


@pytest.fixture(autouse=True)
def _no_metadata_probe(monkeypatch):
    """The cloud metadata server gate (swebench_common.require_task_network_blocked) starts probe
    containers; tests/test_guard_swe_agents.py covers it, here it is stubbed."""
    monkeypatch.setattr(C, "require_task_network_blocked", lambda *a, **k: None)


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SECRET = "sk-test-endpoint-key-9876543210"


@pytest.fixture(autouse=True)
def _scratch(monkeypatch, tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    for var in ("OPENAI_API_KEY", "HF_TOKEN", "GBENCH_SHARD", "GBENCH_SWE_BENCH_LIVE_MODE",
                "GBENCH_SWE_BENCH_LIVE_IMAGE", "GBENCH_SWE_BENCH_LIVE_TEMPERATURE",
                "GBENCH_SWE_BENCH_LIVE_WIN_AGENT_DIR", "GBENCH_SWE_BENCH_LIVE_HARNESS_DIR",
                "GBENCH_SWE_LANCER_IMAGE", "GBENCH_SWE_LANCER_TEMPERATURE", "GBENCH_TEMPERATURE",
                "GBENCH_REPETITION_PENALTY", "GBENCH_FREQUENCY_PENALTY",
                "GBENCH_PRESENCE_PENALTY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GBENCH_SWE_OMP_THREADS", "0")
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_TRAJ_DIR", str(tmp_path / "traj"))
    monkeypatch.setenv("GBENCH_SWE_LANCER_TRAJ_DIR", str(tmp_path / "lancer"))


# ============================================================================================
# loopback servers
# ============================================================================================
class _Recorder(http.server.BaseHTTPRequestHandler):
    """Records every request (path, Authorization, body) and answers with a small JSON."""
    seen: list = []
    redirect_to = None

    def _answer(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        type(self).seen.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                "body": json.loads(body) if body else None,
                                "method": self.command})
        if type(self).redirect_to:
            self.send_response(302)
            self.send_header("Location", type(self).redirect_to)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"moved")
            return
        out = json.dumps({"choices": [{"message": {"content": "hello"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    do_GET = do_POST = _answer

    def log_message(self, *a):
        return


def _server(handler):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def upstream():
    handler = type("Up", (_Recorder,), {"seen": [], "redirect_to": None})
    server = _server(handler)
    yield server, handler
    server.shutdown()
    server.server_close()


@pytest.fixture
def other():
    handler = type("Other", (_Recorder,), {"seen": [], "redirect_to": None})
    server = _server(handler)
    yield server, handler
    server.shutdown()
    server.server_close()


def _raw(port, request_line, body=b""):
    """Send one raw HTTP request (any request-target) and return (status, headers, body)."""
    s = socket.create_connection(("127.0.0.1", port), timeout=30)
    head = f"{request_line}\r\nHost: x\r\nConnection: close\r\n"
    if body:
        head += f"Content-Length: {len(body)}\r\nContent-Type: application/json\r\n"
    s.sendall(head.encode() + b"\r\n" + body)
    data = b""
    while True:
        chunk = s.recv(65536)
        if not chunk:
            break
        data += chunk
    s.close()
    head, _, rest = data.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = int(lines[0].split()[1])
    headers = {k.lower(): v.strip() for k, _, v in (x.partition(":") for x in lines[1:])}
    return status, headers, rest


# ============================================================================================
# EndpointRelay / KeyForwarder
# ============================================================================================
@pytest.mark.parametrize("path_tmpl", [
    ":{p}/anything", ".attacker.example/x", "@127.0.0.1:{p}/x", "//127.0.0.1:{p}/v1/x",
    "http://127.0.0.1:{p}/v1/x", "/v1/../x", "/v1/./x", "/v1/%2e%2e/x", "/v1//x",
    "/v1/a%2fb", "/v2/x", "/v1", "/v1/", "/v1/x#frag", "/v1/x?y=<z>", "/v1/a\\b", "/v1/a@b",
    "/v1/a:1",
])
def test_relay_refuses_a_path_that_could_retarget_it(upstream, other, path_tmpl):
    # The endpoint has no explicit port, the case where ':<port>' or '.domain' appended to the
    # endpoint's root would name another server; the relay must never reach `other`.
    port = other[0].server_address[1]
    relay = C.EndpointRelay("http://127.0.0.1/v1", "127.0.0.1", key=SECRET).start()
    try:
        status, _, body = _raw(relay._server.server_address[1],
                               f"GET {path_tmpl.format(p=port)} HTTP/1.1")
    finally:
        relay.stop()
    assert status == 400
    assert other[1].seen == [] and upstream[1].seen == []
    assert SECRET.encode() not in body and relay.requests_refused == 1


def test_key_forwarder_is_an_endpoint_relay_and_refuses_the_port_probe(other):
    port = other[0].server_address[1]
    fwd = P.KeyForwarder("http://127.0.0.1/v1", SECRET, "127.0.0.1")
    assert isinstance(fwd, C.EndpointRelay)
    fwd.start()
    try:
        status, _, body = _raw(fwd._server.server_address[1], f"GET :{port}/anything HTTP/1.1")
    finally:
        fwd.stop()
    assert status == 400 and other[1].seen == [] and SECRET.encode() not in body


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH", "OPTIONS"])
def test_relay_forwards_only_get_and_post(upstream, method):
    up = f"http://127.0.0.1:{upstream[0].server_address[1]}/v1"
    relay = C.EndpointRelay(up, "127.0.0.1", key=SECRET).start()
    try:
        status, _, _ = _raw(relay._server.server_address[1], f"{method} /v1/models HTTP/1.1")
    finally:
        relay.stop()
    assert status == 405 and upstream[1].seen == []


def test_relay_forwards_plain_v1_paths_to_the_fixed_endpoint_with_the_key(upstream):
    port = upstream[0].server_address[1]
    relay = C.EndpointRelay(f"http://127.0.0.1:{port}/v1/", "127.0.0.1", key=SECRET).start()
    try:
        body = json.dumps({"model": "m", "messages": []}).encode()
        status, headers, reply = _raw(relay._server.server_address[1],
                                      "POST /v1/chat/completions?stream=false HTTP/1.1", body)
    finally:
        relay.stop()
    assert status == 200 and headers["content-type"] == "application/json"
    assert json.loads(reply)["choices"][0]["message"]["content"] == "hello"
    seen = upstream[1].seen[-1]
    assert seen["path"] == "/v1/chat/completions?stream=false"
    assert seen["auth"] == f"Bearer {SECRET}" and seen["body"] == {"model": "m", "messages": []}
    assert relay.target("/v1/models") == f"http://127.0.0.1:{port}/v1/models"
    rec = relay.record()
    assert rec["relays_to"] == f"http://127.0.0.1:{port}/v1" and SECRET not in json.dumps(rec)


def test_relay_keeps_the_endpoint_base_path(upstream):
    relay = C.EndpointRelay("https://gw.example:8443/serving/v1", "127.0.0.1")
    assert relay.target("/v1/chat/completions") == (
        "https://gw.example:8443/serving/v1/chat/completions")
    assert relay.target("/v1/../../x") is None
    with pytest.raises(ValueError):
        C.EndpointRelay("file:///etc/passwd", "127.0.0.1")


def test_relay_drops_the_clients_authorization(upstream):
    port = upstream[0].server_address[1]
    relay = C.EndpointRelay(f"http://127.0.0.1:{port}/v1", "127.0.0.1").start()
    try:
        s = socket.create_connection(("127.0.0.1", relay._server.server_address[1]))
        s.sendall(b"GET /v1/models HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer from-sandbox\r\n"
                  b"Connection: close\r\n\r\n")
        while s.recv(65536):
            pass
        s.close()
    finally:
        relay.stop()
    assert upstream[1].seen[-1]["auth"] is None


def test_relay_does_not_follow_a_redirect(upstream, other):
    upstream[1].redirect_to = f"http://127.0.0.1:{other[0].server_address[1]}/v1/steal"
    port = upstream[0].server_address[1]
    relay = C.EndpointRelay(f"http://127.0.0.1:{port}/v1", "127.0.0.1", key=SECRET).start()
    try:
        status, headers, body = _raw(relay._server.server_address[1], "GET /v1/models HTTP/1.1")
    finally:
        relay.stop()
    assert status == 302 and "location" not in headers and body == b"moved"
    assert other[1].seen == []          # the key was never sent to the redirect target


def test_relay_sets_body_fields_on_completion_requests_only(upstream):
    port = upstream[0].server_address[1]
    fields = {"chat_template_kwargs": {"enable_thinking": True}, "temperature": 0.3}
    relay = C.EndpointRelay(f"http://127.0.0.1:{port}/v1", "127.0.0.1",
                            body_fields=fields).start()
    try:
        body = json.dumps({"model": "m", "chat_template_kwargs": {"x": 1}}).encode()
        _raw(relay._server.server_address[1], "POST /v1/chat/completions HTTP/1.1", body)
        # an allowed path that is not a completions path keeps its body as sent
        _raw(relay._server.server_address[1], "POST /v1/models HTTP/1.1", body)
    finally:
        relay.stop()
    first, second = upstream[1].seen[-2:]
    assert first["body"] == {"model": "m", "temperature": 0.3,
                             "chat_template_kwargs": {"x": 1, "enable_thinking": True}}
    assert second["body"] == {"model": "m", "chat_template_kwargs": {"x": 1}}


def test_relay_failure_reveals_no_key(caplog):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    relay = C.EndpointRelay(f"http://127.0.0.1:{dead}/v1", "127.0.0.1", key=SECRET).start()
    try:
        status, _, body = _raw(relay._server.server_address[1], "GET /v1/models HTTP/1.1")
    finally:
        relay.stop()
    assert status == 502 and SECRET.encode() not in body and SECRET not in caplog.text


def test_pro_result_records_what_the_forwarder_forwards(monkeypatch, tmp_path, upstream):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_TASK_HOST", "127.0.0.1")
    monkeypatch.setenv("GBENCH_SWE_BENCH_PRO_TRAJ_DIR", str(tmp_path / "runs"))
    rows = [{"instance_id": "instance_a", "problem_statement": "p", "docker_image": "img",
             "repo_language": "go", "repo": "o/r"}]
    monkeypatch.setattr(P, "load_v2_rows", lambda: rows)
    monkeypatch.setattr(P, "load_hard51_ids", lambda ids: set())
    monkeypatch.setattr(P, "read_locked_agent_config",
                        lambda: ("model:\n  model_kwargs: {}\n", P.AGENT_CONFIG_SHA256))
    monkeypatch.setattr(P, "_run_harbor", lambda workdir, args, job: (0, ""))
    base = f"http://127.0.0.1:{upstream[0].server_address[1]}/v1"
    res = P.run_swe_bench_pro_v2("m", base, 1, enable_thinking=True)
    fwd = res["harness"]["key_forwarder"]
    assert "/v1/" in fwd["forwards"] and base in fwd["forwards"] and "no redirects" in fwd["forwards"]
    assert SECRET not in json.dumps(res, default=str)


def test_pro_doc_describes_the_path_check():
    doc = " ".join(open(os.path.join(REPO, "docs", "evals", "swe_bench_pro.md"),
                        encoding="utf-8").read().split())
    for phrase in ("urlunsplit", "GET :8080/x", "`405`", "Redirects from the endpoint are not "
                   "followed", "`forwards`"):
        assert phrase in doc


# ============================================================================================
# SWE-bench-Live: the ten tabs
# ============================================================================================
def test_live_settings_are_the_leaderboards_ten_tabs_in_its_order():
    js = ("const validSets = ['lite', 'full', 'verified', 'ccpp', 'csharp', 'java', 'tsjs', "
          "'go', 'rust', 'windows'];")
    names = tuple(re.findall(r"'([a-z]+)'", js))
    assert LV.SETTINGS == names
    assert LV.PYTHON_SETTINGS + LV.MULTILANG_SETTINGS + LV.WINDOWS_SETTINGS == LV.SETTINGS
    assert [LV.TAB_LABELS[s] for s in LV.SETTINGS] == [
        "Lite", "Full", "Verified", "C/C++", "C#", "Java", "TS/JS", "Go", "Rust", "Windows"]
    assert {LV.setting_family(s) for s in LV.MULTILANG_SETTINGS} == {"multilang"}
    assert LV.setting_family("windows") == "windows"


def test_live_multilang_tabs_pin_their_splits():
    assert LV.MULTILANG_REVISION == "3638632e8153a10ca422c1022bed79023084b5c9"
    assert LV.WINDOWS_REVISION == "b9fa25f2732dbc2c93b9738b610d053d8b7bd7a9"
    assert LV.MULTILANG_TAB_SPLITS == {"ccpp": ("c", "cpp"), "csharp": ("cs",),
                                       "java": ("java",), "tsjs": ("js", "ts"), "go": ("go",),
                                       "rust": ("rust",)}
    assert {s: LV.setting_rows(s) for s in LV.MULTILANG_SETTINGS + ("windows",)} == {
        "ccpp": 253, "csharp": 136, "java": 160, "tsjs": 219, "go": 138, "rust": 171,
        "windows": 66}
    assert sum(n for n, _ in LV.MULTILANG_SPLITS.values()) == 1077
    for part in LV.setting_splits("ccpp") + LV.setting_splits("windows"):
        assert re.fullmatch(r"[0-9a-f]{64}", part["sha256"])
        assert set(LV.LAUNCH_COLUMNS) <= set(part["columns"])
    assert "language" in LV.setting_splits("windows")[0]["columns"]
    assert LV.setting_dataset_label("tsjs") == (
        "SWE-bench-Live/MultiLang@3638632e8153a10ca422c1022bed79023084b5c9 (js + ts)")


def test_live_leaderboard_totals_and_the_reason_when_no_row_matches():
    res = {}
    assert LV._leaderboard_totals("java", res) == [] and res["leaderboard_totals"][-1] == 160
    res = {}
    reasons = LV._leaderboard_totals("ccpp", res)
    assert reasons == ["no C/C++ leaderboard row scores the pinned 253 instances (board totals "
                       "48, 66)"]
    assert "253" in res["leaderboard_totals_note"]
    assert LV._leaderboard_totals("full", {}) == [] and LV._leaderboard_totals("verified", {}) == []


def test_live_eval_categories_route_every_tab():
    from gbench import cli
    from gbench.runners import evals as E
    assert E.eval_category_names()["swe_bench_live"] == set(LV.SETTINGS)
    assert E.route_eval_categories("java,simple_python,tsjs", "swe_bench_live") == "java,tsjs"
    assert LV.resolve_settings(None, "windows,go,lite") == ["lite", "go", "windows"]
    with pytest.raises(ValueError, match="not leaderboard settings"):
        LV.resolve_settings(None, "cpp")

    class Passed(Exception):
        pass
    with mock.patch.object(cli, "setup_logging", side_effect=Passed):
        with pytest.raises(Passed):
            cli.main(["--remote-endpoint", "http://127.0.0.1:1/v1", "--tokenizer", "t",
                      "--max-output-tokens", "1024", "--evals", "swe_bench_live",
                      "--eval-categories", "java,rust"])


def test_live_eval_categories_reach_the_suite_through_the_runner(monkeypatch):
    from gbench.core.config import BenchmarkConfig
    from gbench.runners.evals import EvalsBenchmarkRunner
    config = BenchmarkConfig()
    config.eval_categories = "go,simple_python"
    runner = EvalsBenchmarkRunner(config)
    seen = {}
    with mock.patch.dict("gbench.runners.eval_suites.SUITES",
                         {"swe_bench_live": lambda **k: seen.update(k) or {"status": "success"}}):
        runner._run_single_eval("swe_bench_live", "m", "http://127.0.0.1:1/v1", num_threads=2)
    assert seen.get("eval_categories") == "go"


def test_live_windows_tab_off_a_windows_host_is_infra_required(monkeypatch):
    monkeypatch.setattr(LV.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(LV.subprocess, "run", lambda cmd, **kw: types.SimpleNamespace(
        returncode=0, stdout="{}"))
    monkeypatch.setattr(LV._platform, "system", lambda: "Linux")
    monkeypatch.setattr(LV, "pinned_swe_snapshot", lambda *a, **k: pytest.fail("no fetch"))
    with pytest.raises(RuntimeError) as err:
        LV.run_swe_bench_live("m", "http://h:1/v1", 1, split="windows")
    text = str(err.value)
    assert text.startswith("[INFRA REQUIRED] swe_bench_live")
    assert "Windows host" in text and "Windows containers" in text and "Linux" in text
    assert "--eval-categories" in text


def test_live_windows_prerequisites_check_engine_and_checkouts(monkeypatch, tmp_path):
    monkeypatch.setattr(LV._platform, "system", lambda: "Windows")
    engine = {"os": "linux"}
    monkeypatch.setattr(LV.subprocess, "run", lambda cmd, **kw: types.SimpleNamespace(
        returncode=0, stdout=engine["os"] + "\n"))
    ok, why = LV.check_windows_prerequisites()
    assert not ok and "'linux'" in why and "switch it to Windows containers" in why
    engine["os"] = "windows"
    ok, why = LV.check_windows_prerequisites()
    assert not ok and "GBENCH_SWE_BENCH_LIVE_WIN_AGENT_DIR" in why and LV.WIN_AGENT_COMMIT in why
    agent = tmp_path / "win-agent"
    agent.mkdir()
    (agent / "main.py").write_text("")
    harness = tmp_path / "sbl"
    (harness / "evaluation").mkdir(parents=True)
    (harness / "evaluation" / "evaluation.py").write_text("")
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_WIN_AGENT_DIR", str(agent))
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_HARNESS_DIR", str(harness))
    heads = {str(agent): "0" * 40}
    monkeypatch.setattr(LV, "_git_head", lambda p: heads.get(p))
    ok, why = LV.check_windows_prerequisites()
    assert not ok and "check out that commit" in why
    heads[str(agent)] = LV.WIN_AGENT_COMMIT
    heads[str(harness)] = LV.MAIN_COMMIT
    ok, why = LV.check_windows_prerequisites()
    assert not ok and "git submodule update --init" in why
    heads[os.path.join(str(harness), "launch")] = LV.REPOLAUNCH_COMMIT
    assert LV.check_windows_prerequisites() == (True, "")


def test_live_multilang_needs_the_image_labels_of_upstreams_harness(monkeypatch):
    labels = {"org.gbench.swe_bench_live_fork": LV.FORK_COMMIT}
    monkeypatch.setattr(LV.subprocess, "run", lambda cmd, **kw: types.SimpleNamespace(
        returncode=0, stdout=json.dumps(labels)))
    ok, why = LV._check_image(["python", "multilang"])
    assert not ok and LV.MAIN_COMMIT in why and "docker build" in why
    labels.update({"org.gbench.swe_bench_live_main": LV.MAIN_COMMIT,
                   "org.gbench.repolaunch": LV.REPOLAUNCH_COMMIT})
    assert LV._check_image(["python", "multilang"]) == (True, "")
    labels["org.gbench.repolaunch"] = "f" * 40
    ok, why = LV._check_image(["multilang"])
    assert not ok and LV.REPOLAUNCH_COMMIT in why


def test_live_prereq_probes_upstreams_harness_for_each_multilang_split(monkeypatch):
    monkeypatch.setattr(LV.shutil, "which", lambda _x: "/usr/bin/docker")
    labels = json.dumps({"org.gbench.swe_bench_live_main": LV.MAIN_COMMIT,
                         "org.gbench.repolaunch": LV.REPOLAUNCH_COMMIT})
    runs = []

    def _run(cmd, **kw):
        runs.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout=labels, stderr="")
    monkeypatch.setattr(LV.subprocess, "run", _run)
    fetched = []
    monkeypatch.setattr(LV, "pinned_swe_snapshot", lambda d, split, rev, name, **k: fetched.append(
        (d, split, rev, k["rows"])) or f"/snap/{split}.jsonl")
    assert LV.check_swe_bench_live_prerequisites(["ccpp"]) == (True, "")
    assert fetched == [(LV.MULTILANG_DATASET, "c", LV.MULTILANG_REVISION, 111),
                       (LV.MULTILANG_DATASET, "cpp", LV.MULTILANG_REVISION, 142)]
    probes = [c for c in runs if c[:2] == ["docker", "run"]]
    assert len(probes) == 2 and all("none" in c and f"PYTHONPATH={LV.MAIN_PYTHONPATH}" in c
                                    for c in probes)
    assert "compile(r['log_parser']" in probes[0][probes[0].index("-c") + 1]


def test_live_launch_harness_command_runs_upstreams_evaluation(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    w = str(tmp_path)
    cmd, env = LV.launch_harness_command(w, f"{w}/rows.jsonl", f"{w}/preds.json", f"{w}/eval",
                                         3, 0, "linux")
    assert cmd[:6] == ["docker", "run", "--rm", "--network", "host", "-v"]
    assert f"{w}:{w}:rw" in cmd and cmd[cmd.index("-w") + 1] == w
    assert f"PYTHONPATH={LV.MAIN_PYTHONPATH}" in cmd and "--user" in cmd and env == {}
    tail = cmd[cmd.index(LV._image()) + 1:]
    assert tail == ["python", "-m", "evaluation.evaluation", "--dataset", f"{w}/rows.jsonl",
                    "--patch_dir", f"{w}/preds.json", "--platform", "linux", "--workers", "3",
                    "--output_dir", f"{w}/eval", "--overwrite", "1"]
    assert SECRET not in " ".join(cmd)
    capped, _ = LV.launch_harness_command(w, "r", "p", "o", 3, 4, "linux")
    assert "OMP_NUM_THREADS=4" in capped and "evaluation.evaluation" in " ".join(capped)
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_HARNESS_DIR", str(tmp_path / "sbl"))
    wcmd, wenv = LV.launch_harness_command(w, "r", "p", "o", 2, 4, "windows")
    assert wcmd[0] == sys.executable and "--platform" in wcmd and "windows" in wcmd
    assert wenv["PYTHONUTF8"] == "1"
    assert wenv["PYTHONPATH"].split(os.pathsep) == [str(tmp_path / "sbl" / "launch"),
                                                    str(tmp_path / "sbl")]


def _ml_row(split, i, **extra):
    row = {c: "x" for c in LV.LAUNCH_COLUMNS}
    row.update({"instance_id": f"{split}-org__repo-{i}", "repo": f"org/{split}",
                "problem_statement": f"issue {split} {i}", "FAIL_TO_PASS": ["t"],
                "PASS_TO_PASS": [], "test_cmds": ["make test"], "print_cmds": ["cat log"],
                "rebuild_cmds": [], "docker_image": f"starryzhang/sweb.eval.x86_64.{split}_{i}",
                "log_parser": "def parser(log):\n    return {}\n",
                "created_at": "2026-01-02T00:00:00Z"})
    row.update(extra)
    return row


def _fake_harness(monkeypatch, success, errors=()):
    """subprocess.run stand-in for upstream's harness: writes its results.json. The instance
    images count as present on the host (no pull)."""
    calls = []
    monkeypatch.setattr(LV, "host_pull_images", lambda images, workers=1: {
        i: {"state": "present"} for i in images})

    def _run(cmd, **kw):
        calls.append({"cmd": cmd, "cwd": kw.get("cwd"), "env": kw.get("env")})
        out = cmd[cmd.index("--output_dir") + 1]
        preds = json.load(open(cmd[cmd.index("--patch_dir") + 1]))
        rows = [json.loads(line) for line in open(cmd[cmd.index("--dataset") + 1])]
        assert all("_split" not in r for r in rows)
        os.makedirs(out, exist_ok=True)
        ids = list(preds)
        empty = [i for i in ids if not preds[i]["model_patch"].strip()]
        err = [i for i in ids if i in errors]
        ok = [i for i in ids if i in success and i not in empty and i not in err]
        fail = [i for i in ids if i not in ok and i not in empty and i not in err]
        json.dump({"submitted": len(ids), "empty_patch": len(empty), "empty_patch_ids": empty,
                   "success_ids": ok, "failure_ids": fail, "error_ids": err,
                   "incomplete_ids": [], "success": len(ok), "failure": len(fail),
                   "error": len(err), "incomplete": 0}, open(os.path.join(out, "results.json"),
                                                            "w"))
        return types.SimpleNamespace(returncode=0, stdout="Evaluation ended", stderr="")
    monkeypatch.setattr(LV.subprocess, "run", _run)
    return calls


def test_live_launch_scorer_reads_results_json(monkeypatch):
    rows = [dict(_ml_row("go", i), _split="go") for i in range(3)]
    calls = _fake_harness(monkeypatch, success={"go-org__repo-0"}, errors={"go-org__repo-2"})
    metrics = {}
    traces = [{"extra_payload": {"instance_id": r["instance_id"]}, "agent_patch": "d\n"}
              for r in rows]
    asyncio.run(LV.make_launch_scorer("go", rows, "linux", 2, metrics)(traces))
    assert [t["is_correct"] for t in traces] == [True, False, False]
    rep = metrics["harness_report"]
    assert rep["success_ids"] == ["go-org__repo-0"] and rep["error_ids"] == ["go-org__repo-2"]
    assert metrics["harness"]["tree"].startswith(f"microsoft/SWE-bench-Live@{LV.MAIN_COMMIT}")
    assert calls[0]["cwd"] == metrics["harness_workdir"]
    res = {"status": "success"}
    assert LV.launch_report_reasons(res, metrics) == [
        "partial: 1 of 3 submitted instance(s) hit harness errors (not model-scoreable)"]
    assert res["status"] == "completed_with_errors"
    assert res["partial_evaluation"]["harness_error_instances"] == 1


def test_live_launch_report_errors():
    res = {"status": "success"}
    assert LV.launch_report_reasons(res, {"harness_report": {
        "report_error": "harness report not produced"}})
    assert res["status"] == "error"
    res = {"status": "success"}
    reasons = LV.launch_report_reasons(res, {"harness_report": {
        "submitted": 2, "success_ids": [], "failure_ids": [], "empty_patch_ids": [],
        "error_ids": ["a", "b"], "incomplete_ids": []}})
    assert res["status"] == "error" and "infrastructure failure" in res["error"]
    assert reasons == ["harness infrastructure failure (no scoreable instances)"]
    res = {"status": "success"}
    assert LV.launch_report_reasons(res, {"harness_report": {
        "submitted": 2, "success_ids": ["a"], "failure_ids": ["b"], "empty_patch_ids": [],
        "error_ids": [], "incomplete_ids": []}}) == [] and res["status"] == "success"


def _multilang_setup(monkeypatch):
    by_split = {"c": [_ml_row("c", i) for i in range(2)],
                "cpp": [_ml_row("cpp", i) for i in range(3)]}
    for split, rows in by_split.items():
        monkeypatch.setitem(LV.MULTILANG_SPLITS, split, (len(rows), C.snapshot_sha256(rows)))
    fake = types.ModuleType("datasets")
    loads = []

    def _load(name, *a, split=None, revision=None, **k):
        loads.append((name, split, revision))
        return [dict(r) for r in by_split[split]]
    fake.load_dataset = _load
    monkeypatch.setitem(sys.modules, "datasets", fake)
    monkeypatch.setattr(LV, "check_swe_bench_live_prerequisites", lambda settings: (True, ""))
    monkeypatch.setattr(A, "require_agent", lambda *a: None)
    monkeypatch.setattr(A, "load_agent_config", lambda: (
        {"agent": {"step_limit": 250, "instance_template": "/testbed"},
         "environment": {"timeout": 60, "cwd": "/testbed"},
         "model": {"model_name": "x", "model_kwargs": {"drop_params": True,
                                                      "temperature": 0.0}}},
        A.LEADERBOARD_CONFIG_SHA256, "1.17.5"))
    return by_split, loads


def test_live_multilang_tab_end_to_end(monkeypatch):
    by_split, loads = _multilang_setup(monkeypatch)
    monkeypatch.setattr(LV, "check_sweagent_prerequisites", lambda: (True, ""))
    seen_tasks = []

    def _rollouts(tasks, rows, model_name, relay_url, concurrency, out_dir, relay=None):
        seen_tasks.extend(tasks)
        return [{"sample_idx": i, "source_sample_idx": i, "category": t.category,
                 "agent_patch": "d\n", "extra_payload": {"instance_id": t.instance_id, **t.meta},
                 "agent_exit_status": "submitted", "agent_infra_error": False,
                 "agent_steps": 1, "truncated_turns": 0, "context_retry_turns": 0,
                 "is_correct": False, "status": "OK"} for i, t in enumerate(tasks)]
    monkeypatch.setattr(LV, "run_sweagent", _rollouts)
    calls = _fake_harness(monkeypatch, success={"c-org__repo-0", "cpp-org__repo-1",
                                                "cpp-org__repo-2"})
    res = LV.run_swe_bench_live("m", "http://h:1/v1", 2, eval_categories="ccpp")
    assert loads == [(LV.MULTILANG_DATASET, "c", LV.MULTILANG_REVISION),
                     (LV.MULTILANG_DATASET, "cpp", LV.MULTILANG_REVISION)]
    # SWE-agent-for-eval works in each row's own docker_image
    assert [t.image for t in seen_tasks] == [r["docker_image"] for r in by_split["c"]
                                              + by_split["cpp"]]
    assert len(calls) == 1 and "--platform" in calls[0]["cmd"]
    s = res["settings"]["ccpp"]
    assert (s["correct_answers"], s["total_questions"], s["accuracy"]) == (3, 5, 60.0)
    assert s["split_accuracy"] == {"c": {"correct": 1, "total": 2, "accuracy": 50.0},
                                   "cpp": {"correct": 2, "total": 3, "accuracy": 66.67}}
    assert s["tab"] == "C/C++" and s["dataset"] == LV.setting_dataset_label("ccpp")
    assert res["headline_metrics"] == {"ccpp_resolved_rate": 60.0}
    assert res["headline_setting"] == "ccpp" and res["accuracy"] == 60.0
    reason = res["leaderboard_comparable_reason"]
    assert "reference scaffolds" not in reason and "no C/C++ leaderboard row scores" in reason
    assert s["agent"]["scaffold"] == "SWE-agent-for-eval"
    assert s["leaderboard_totals"] == [48, 66]


def test_live_multilang_oneshot_uses_the_same_harness(monkeypatch):
    _multilang_setup(monkeypatch)
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_MODE", "oneshot")
    seen = {}

    def _suite(**k):
        seen["samples"] = k["samples"]
        traces = [{"extra_payload": s[2], "response_text": "```diff\ndiff --git a/x b/x\n"
                   "--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n```", "category": s[2]["category"]}
                  for s in k["samples"]]
        asyncio.run(k["async_eval_fn"](traces))
        correct = sum(t["is_correct"] for t in traces)
        return {"status": "success", "temperature": 0.0, "total_questions": len(traces),
                "correct_answers": correct, "accuracy": round(correct / len(traces) * 100, 2),
                "sample_traces": traces, "category_accuracy": {}}
    monkeypatch.setattr(LV, "run_eval_suite", _suite)
    _fake_harness(monkeypatch, success={"c-org__repo-1"})
    res = LV.run_swe_bench_live("m", "http://h:1/v1", 1, eval_categories="ccpp")
    assert {s[2]["split"] for s in seen["samples"]} == {"c", "cpp"}
    assert res["settings"]["ccpp"]["correct_answers"] == 1
    assert res["mode"] == A.ONESHOT and "one-shot mode" in res["leaderboard_comparable_reason"]


def test_live_infra_required_inside_a_setting_stops_the_run(monkeypatch):
    monkeypatch.setattr(LV, "check_swe_bench_live_prerequisites", lambda settings: (True, ""))
    monkeypatch.setattr(LV, "check_sweagent_prerequisites", lambda: (True, ""))
    ran = []

    def _setting(setting, *a, **k):
        ran.append(setting)
        if setting == "full":
            raise C.infra_required("swe_bench_live", "the snapshot changed", LV.DOCS_URL)
        return {"status": "success", "accuracy": 1.0, "total_questions": 1,
                "correct_answers": 0, "leaderboard_comparable": False}
    monkeypatch.setattr(LV, "_run_setting", _setting)
    with pytest.raises(RuntimeError, match=r"^\[INFRA REQUIRED\].*snapshot changed"):
        LV.run_swe_bench_live("m", "http://h:1/v1", 1, eval_categories="lite,full,verified")
    assert ran == ["lite", "full"]


def test_live_multilang_instance_line_is_the_problem_statement_and_the_image():
    # SWE-agent-for-eval reads each instance from a swe_bench-type file; the MultiLang row's
    # own docker_image is the image, and nothing but the fields its loader needs is written
    row = dict(_ml_row("go", 0), _split="go", base_commit="abc")
    line = LV.sweagent_instance(row)
    assert line == {"instance_id": row["instance_id"], "problem_statement":
                    row["problem_statement"], "base_commit": "abc",
                    "image_name": row["docker_image"]}


def _win_agent_checkout(tmp_path):
    agent = tmp_path / "win-agent-src"
    (agent / "config").mkdir(parents=True)
    (agent / "main.py").write_text("# Win-Agent main.py stand-in\n")
    (agent / "config" / "default.yaml").write_text(
        "dataset: SWE-bench-Live/Windows\nsplit: test\nplatform: windows\n"
        "model: openrouter/deepseek/deepseek-v3.1-terminus\n"
        "base_url: https://openrouter.ai/api/v1\ntools:\n  - shell\n  - read\n  - create\n"
        "  - string_replace\n  - submit\nmax_steps: 100\nworkers: 2\ninstance_timeout: 480\n"
        "prompt: |\n  You are given a code repository at C:\\testbed\n  _PROBLEM_STATEMENT_\n")
    return agent


@pytest.mark.parametrize("thinking", [True, False])
def test_live_windows_tab_runs_win_agent_then_the_harness(monkeypatch, tmp_path, thinking):
    want = 1.0 if thinking else 0.0
    monkeypatch.setattr(LV, "check_windows_metadata_guard", lambda: (True, ""))
    rows = [_ml_row("test", i, language="Go") for i in range(3)]
    monkeypatch.setitem(LV.WINDOWS_SPLITS, "test", (3, C.snapshot_sha256(rows)))
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda name, *a, split=None, revision=None, **k: [dict(r) for r in rows]
    monkeypatch.setitem(sys.modules, "datasets", fake)
    monkeypatch.setattr(LV, "check_swe_bench_live_prerequisites", lambda settings: (True, ""))
    monkeypatch.setattr(A, "require_agent", lambda *a: pytest.fail("no mini-swe-agent here"))
    agent = _win_agent_checkout(tmp_path)
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_WIN_AGENT_DIR", str(agent))
    monkeypatch.setenv("GBENCH_SWE_BENCH_LIVE_HARNESS_DIR", str(tmp_path / "sbl"))
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    relays = []
    real_relay = LV.EndpointRelay

    class _Relay(real_relay):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            relays.append(self)
    monkeypatch.setattr(LV, "EndpointRelay", _Relay)
    harness_calls = _fake_harness(monkeypatch, success={"test-org__repo-0"})
    harness_run = LV.subprocess.run
    agent_calls = []

    def _run(cmd, **kw):
        if "main.py" in cmd:
            agent_calls.append({"cmd": cmd, "cwd": kw.get("cwd")})
            import yaml
            cfg = yaml.safe_load(open(cmd[cmd.index("--config_path") + 1]))
            out = os.path.join(kw["cwd"], "output", LV._win_agent_run_id(cfg))
            os.makedirs(out)
            json.dump({"test-org__repo-0": {"model_patch": "diff --git a/x b/x\n"},
                       "test-org__repo-1": {"model_patch": ""}},
                      open(os.path.join(out, "preds.json"), "w"))
            json.dump({"test-org__repo-0": "submitted", "test-org__repo-1": "exit_cost",
                       "test-org__repo-2": "error"}, open(os.path.join(out, "exit_status.json"),
                                                          "w"))
            return types.SimpleNamespace(returncode=0)
        return harness_run(cmd, **kw)
    monkeypatch.setattr(LV.subprocess, "run", _run)
    res = LV.run_swe_bench_live("m", "http://127.0.0.1:1/v1", 2, enable_thinking=thinking,
                                split="windows")
    cmd = agent_calls[0]["cmd"]
    assert cmd[0] == sys.executable and cmd[1] == "main.py"
    assert cmd[-2:] == ["--api_key", "EMPTY"] and SECRET not in " ".join(cmd)
    import yaml
    cfg = yaml.safe_load(open(cmd[cmd.index("--config_path") + 1]))
    assert cfg["model"] == "openai/m" and cfg["platform"] == "windows"
    assert cfg["max_steps"] == 100 and cfg["instance_timeout"] == 480 and cfg["workers"] == 2
    assert cfg["base_url"].startswith("http://127.0.0.1:") and cfg["base_url"].endswith("/v1")
    assert agent_calls[0]["cwd"] != str(agent)                # a fresh copy of the checkout
    assert cfg["dataset"] == "instances.jsonl"                 # relative: no drive colon
    assert os.path.isfile(os.path.join(agent_calls[0]["cwd"], "instances.jsonl"))
    relay = relays[0]
    # Win-Agent sends no temperature; the relay adds gbench's rule: 1.0 with --thinking, 0.0 without
    assert relay._key == SECRET and relay._body_fields == {
        "chat_template_kwargs": {"enable_thinking": thinking}, "temperature": want}
    assert relay._server is None                               # stopped after the rollouts
    # the harness scored the two instances whose rollout ran; Win-Agent's "error" is infra
    preds = json.load(open(harness_calls[0]["cmd"][harness_calls[0]["cmd"].index(
        "--patch_dir") + 1]))
    assert sorted(preds) == ["test-org__repo-0", "test-org__repo-1"]
    assert "windows" in harness_calls[0]["cmd"]
    s = res["settings"]["windows"]
    assert (s["correct_answers"], s["total_questions"]) == (1, 3)
    assert s["status"] == "completed_with_errors"
    assert s["partial_evaluation"]["agent_infra_error_instances"] == 1
    assert s["agent"]["scaffold"] == "Win-Agent" and s["agent"]["commit"] == LV.WIN_AGENT_COMMIT
    assert (s["temperature"], s["temperature_source"]) == (
        (1.0, "thinking") if thinking else (0.0, "default"))
    assert s["official_sampling"]["send_max_tokens"] is False
    assert "not sent" in s["max_output_tokens_note"]
    # Win-Agent's "error" rollout is a harness error, left out of the rate: 1 of 2 scored
    assert s["scored_instances"] == 2 and s["harness_errors"] == {"agent_infra": 1}
    assert res["headline_metrics"] == {"windows_resolved_rate": 50.0}
    assert SECRET not in json.dumps(res, default=str)


def test_live_dockerfile_carries_upstreams_multilang_harness():
    text = open(os.path.join(REPO, "gbench", "docker", "swe_bench_live.Dockerfile"),
                encoding="utf-8").read()
    assert f"ARG SWE_BENCH_LIVE_MAIN_REF={LV.MAIN_COMMIT}" in text
    assert f"ARG REPOLAUNCH_REF={LV.REPOLAUNCH_COMMIT}" in text
    assert 'org.gbench.swe_bench_live_main="${SWE_BENCH_LIVE_MAIN_REF}"' in text
    assert 'org.gbench.repolaunch="${REPOLAUNCH_REF}"' in text
    assert "ls-tree HEAD launch" in text               # submodule commit checked at build time
    assert f"ARG SWE_BENCH_LIVE_REF={LV.FORK_COMMIT}" in text


def _doc(name):
    return " ".join(open(os.path.join(REPO, "docs", "evals", f"{name}.md"),
                         encoding="utf-8").read().split())


def test_live_doc_matches_the_ten_tabs():
    doc = _doc("swe_bench_live")
    for split, (rows, sha) in LV.MULTILANG_SPLITS.items():
        assert sha in doc, split
    assert LV.WINDOWS_SPLITS["test"][1] in doc
    for s in LV.SETTINGS:
        assert f"`{s}_resolved_rate`" in doc and f"`{s}`" in doc
    for label in LV.TAB_LABELS.values():
        assert label in doc
    for value in (LV.MULTILANG_REVISION, LV.WINDOWS_REVISION, LV.MAIN_COMMIT,
                  LV.REPOLAUNCH_COMMIT, LV.WIN_AGENT_COMMIT, "GBENCH_SWE_BENCH_LIVE_WIN_AGENT_DIR",
                  "GBENCH_SWE_BENCH_LIVE_HARNESS_DIR", "`12b98891`, 2025-07-02",
                  "`leaderboard_totals`", "`split_accuracy`", LV.SWEAGENT_COMMIT,
                  "SWE-agent-for-eval", "whose result is `status: \"error\"`"):
        assert value in doc, value
    assert "d6d7a837" not in doc and "did not run" not in doc
    src = open(LV.__file__, encoding="utf-8").read()
    assert "d6d7a837" not in src and "12b98891" in src


# ============================================================================================
# SWE-Lancer: SimpleAgentSolver, the official task table and images
# ============================================================================================
def test_lancer_pins_the_current_upstream():
    assert SL.HARNESS_REPO == "openai/frontier-evals"
    assert SL.HARNESS_COMMIT == "51052cede8cc608f95bb00346635e03759013e5a"
    assert SL.TASKS_SHA256 == "5c3a6d4570b49be0d9fced98f5b32487420b16f25c98d6658830e31fa03f049a"
    assert SL.VARIANTS == ("ic_swe", "swe_manager")
    assert SL.VARIANT_TASKS == {"ic_swe": 198, "swe_manager": 265}
    assert SL.SOLVER == "swelancer.solvers.swelancer_agent.solver:SimpleAgentSolver"
    assert (SL.TASK_IMAGE_PREFIX, SL.TASK_IMAGE_TAG) == ("swelancer/swelancer_x86", "releasev1")
    assert SL.OFFICIAL_SAMPLING.send_temperature is False
    assert SL.OFFICIAL_SAMPLING.send_max_tokens is False


def test_lancer_prerequisites_name_what_is_missing(monkeypatch):
    monkeypatch.setattr(SL.os.path, "exists", lambda p: True)
    monkeypatch.setattr(__import__("shutil"), "which", lambda n: "/usr/bin/docker")
    state = {"labels": None, "probe": 0}

    def _run(cmd, **kw):
        if cmd[:3] == ["docker", "image", "inspect"]:
            if state["labels"] is None:
                return types.SimpleNamespace(returncode=1, stdout="", stderr="")
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(state["labels"]),
                                         stderr="")
        if cmd[-1] == "--probe":
            return types.SimpleNamespace(returncode=state["probe"], stdout="",
                                         stderr="no iptables backend sees the DOCKER-USER chain")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(SL.subprocess, "run", _run)
    ok, why = SL.check_swe_lancer_prerequisites()
    assert not ok and "is not built" in why and "swe_lancer_harness.Dockerfile" in why
    state["labels"] = {"org.gbench.frontier_evals": "0" * 40}
    ok, why = SL.check_swe_lancer_prerequisites()
    assert not ok and SL.HARNESS_COMMIT in why and "rebuild it" in why
    state["labels"] = {"org.gbench.frontier_evals": SL.HARNESS_COMMIT,
                       "org.gbench.swelancer_tasks_sha256": SL.TASKS_SHA256}
    state["probe"] = 1
    ok, why = SL.check_swe_lancer_prerequisites()
    assert not ok and "DOCKER-USER" in why and "NET_ADMIN" in why
    state["probe"] = 0
    assert SL.check_swe_lancer_prerequisites() == (True, "")


def test_lancer_harness_command_is_upstreams_run_command(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    w = str(tmp_path)
    cmd = SL.harness_command("n", w, "ic_swe", None, "google/gemma", "http://127.0.0.1:7/v1", 4,
                             f"{w}/r.json")
    assert cmd[:4] == ["docker", "run", "--rm", "--name"]
    for flag in ("--network", "host", "--cap-add", "NET_ADMIN", "NET_RAW",
                 "/var/run/docker.sock:/var/run/docker.sock", f"{w}:{w}:rw",
                 "OPENAI_BASE_URL=http://127.0.0.1:7/v1", "OPENAI_API_KEY=EMPTY",
                 "USE_WEB_PROXY=false", "EXPENSIFY_URL=https://www.expensify.com/",
                 "NEW_EXPENSIFY_URL=https://new.expensify.com/", f"GBENCH_WORKDIR={w}"):
        assert flag in cmd, flag
    args = cmd[cmd.index("-c") + 2:]
    assert cmd[cmd.index("-c") + 1] == "import swe_lancer_eval; swe_lancer_eval.entry()"
    assert args == [
        "swelancer.split=diamond", "swelancer.task_type=ic_swe",
        f"swelancer.solver={SL.SOLVER}", "swelancer.solver.model=openai/google/gemma",
        "swelancer.solver.computer_runtime="
        "nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime",
        "swelancer.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig",
        "swelancer.solver.computer_runtime.env.pull_from_registry=True",
        "swelancer.docker_image_prefix=swelancer/swelancer_x86",
        "swelancer.docker_image_tag=releasev1", f"swelancer.runs_dir={w}/runs",
        f"swelancer.report_path={w}/r.json", "runner.concurrency=4",
        "runner.experimental_use_multiprocessing=False", "runner.enable_slackbot=False",
        "runner.recorder=nanoeval.recorder:dummy_recorder", "runner.max_retries=2"]
    assert SECRET not in " ".join(cmd)
    mgr = SL.harness_command("n", w, "swe_manager", ["1-manager-0"], "m", "u", 1, "r")
    assert "swelancer.use_single_image=True" in mgr
    assert 'swelancer.taskset=["1-manager-0"]' in mgr


def _lancer_stubs(monkeypatch, rows_by_variant, tasks):
    monkeypatch.setattr(SL, "check_swe_lancer_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(SL, "load_tasks", lambda: tasks)
    runs = []

    def _run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            runs.append(cmd)
            variant = [a.split("=", 1)[1] for a in cmd if a.startswith("swelancer.task_type=")][0]
            report = [a.split("=", 1)[1] for a in cmd if a.startswith("swelancer.report_path=")][0]
            if rows_by_variant.get(variant) is not None:
                json.dump({"summary": {"total_earned": 1}, "tasks": rows_by_variant[variant]},
                          open(report, "w"))
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(SL.subprocess, "run", _run)
    return runs


def _tasks():
    return ([{"question_id": f"ic{i}", "variant": "ic_swe", "set": "diamond", "price": 100.0}
             for i in range(3)]
            + [{"question_id": f"m{i}", "variant": "swe_manager", "set": "diamond",
                "price": 1000.0} for i in range(2)])


def _row(qid, variant, price, correct, err=None):
    return {"question_id": qid, "variant": variant, "price": price, "correct": correct,
            "earned": price if correct else 0.0, "available": price, "system_error": err,
            "run_dir": f"/runs/{qid}", "input_tokens": 10, "output_tokens": 5}


def test_lancer_runs_both_task_types_with_their_headlines(monkeypatch):
    monkeypatch.setitem(SL.VARIANT_TASKS, "ic_swe", 3)
    monkeypatch.setitem(SL.VARIANT_TASKS, "swe_manager", 2)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    rows = {"ic_swe": [_row("ic0", "ic_swe", 100.0, True), _row("ic1", "ic_swe", 100.0, False),
                       _row("ic2", "ic_swe", 100.0, False, err="RolloutSystemError: setup")],
            "swe_manager": [_row("m0", "swe_manager", 1000.0, True),
                            _row("m1", "swe_manager", 1000.0, True)]}
    runs = _lancer_stubs(monkeypatch, rows, _tasks())
    relays = []
    real = SL.EndpointRelay

    class _Relay(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            relays.append(self)
    monkeypatch.setattr(SL, "EndpointRelay", _Relay)
    res = SL.run_swe_lancer("m", "http://127.0.0.1:1/v1", 2, enable_thinking=True)
    # two upstream runs, full task sets: no taskset argument
    assert [next(a for a in c if a.startswith("swelancer.task_type=")) for c in runs] == [
        "swelancer.task_type=ic_swe", "swelancer.task_type=swe_manager"]
    assert not any(a.startswith("swelancer.taskset=") for c in runs for a in c)
    assert relays[0]._key == SECRET and relays[0]._server is None
    # --thinking: SimpleAgentSolver sends no temperature, the relay adds the thinking rule's 1.0
    assert relays[0]._body_fields == {"chat_template_kwargs": {"enable_thinking": True},
                                      "temperature": 1.0}
    assert all(f"OPENAI_BASE_URL={relays[0].url}".rsplit(":", 1)[0] in " ".join(c) for c in runs)
    h = res["headline_metrics"]
    # the system-error task is a harness error, left out of pass@1: 1 of 2, 3 of 4 overall
    assert (h["ic_swe_pass_at_1"], h["ic_swe_earned_usd"]) == (50.0, 100.0)
    assert (h["swe_manager_pass_at_1"], h["swe_manager_earned_usd"]) == (100.0, 2000.0)
    assert (h["diamond_pass_at_1"], h["diamond_earned_usd"]) == (75.0, 2100.0)
    assert (res["accuracy"], res["total_questions"], res["correct_answers"]) == (75.0, 5, 3)
    assert res["scored_tasks"] == 4 and res["harness_errors"] == {"system_error": 1}
    assert res["settings"]["ic_swe"]["status"] == "completed_with_errors"
    assert res["status"] == "completed_with_errors"
    assert (res["temperature"], res["temperature_source"]) == (1.0, "thinking")
    assert res["official_sampling"]["send_temperature"] is False
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert "not sent" in res["max_output_tokens_note"]
    assert res["leaderboard_comparable"] is False
    assert "partial: 1 IC SWE task(s) ended in a system error" in res[
        "leaderboard_comparable_reason"]
    assert ("temperature=1.0 sent by the --thinking rule" in res["leaderboard_comparable_reason"]
            and "SimpleAgentSolver sends none" in res["leaderboard_comparable_reason"])
    assert [t["category"] for t in res["sample_traces"]] == ["ic_swe"] * 3 + ["swe_manager"] * 2
    assert [t["sample_idx"] for t in res["sample_traces"]] == list(range(5))
    assert SECRET not in json.dumps(res, default=str)
    assert not any(SECRET in a for c in runs for a in c)



def test_lancer_no_think_run_sends_temperature_zero(monkeypatch):
    # no-think: gbench's rule, 0.0 on every request (SimpleAgentSolver itself sends none)
    monkeypatch.setitem(SL.VARIANT_TASKS, "ic_swe", 3)
    monkeypatch.setitem(SL.VARIANT_TASKS, "swe_manager", 2)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    rows = {"ic_swe": [_row(f"ic{i}", "ic_swe", 100.0, True) for i in range(3)],
            "swe_manager": [_row(f"m{i}", "swe_manager", 1000.0, True) for i in range(2)]}
    _lancer_stubs(monkeypatch, rows, _tasks())
    relays = []
    real = SL.EndpointRelay

    class _Relay(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            relays.append(self)
    monkeypatch.setattr(SL, "EndpointRelay", _Relay)
    res = SL.run_swe_lancer("m", "http://127.0.0.1:1/v1", 2, enable_thinking=False)
    assert relays[0]._body_fields == {"chat_template_kwargs": {"enable_thinking": False},
                                      "temperature": 0.0}
    sent = json.loads(relays[0].patch_body("/v1/chat/completions", b'{"model": "m"}',
                                           "application/json"))
    assert sent["temperature"] == 0.0
    assert (res["temperature"], res["temperature_source"]) == (0.0, "default")
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert ("temperature=0.0 sent by the no-think rule"
            in res["leaderboard_comparable_reason"])

def test_lancer_clean_full_run_names_only_the_temperature(monkeypatch):
    monkeypatch.setitem(SL.VARIANT_TASKS, "ic_swe", 3)
    monkeypatch.setitem(SL.VARIANT_TASKS, "swe_manager", 2)
    rows = {"ic_swe": [_row(f"ic{i}", "ic_swe", 100.0, i == 0) for i in range(3)],
            "swe_manager": [_row(f"m{i}", "swe_manager", 1000.0, False) for i in range(2)]}
    _lancer_stubs(monkeypatch, rows, _tasks())
    res = SL.run_swe_lancer("m", "http://127.0.0.1:1/v1", 1)
    # the only deviation: the temperature gbench always sends (SimpleAgentSolver sends none)
    assert res["status"] == "success" and res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"] == (
        "temperature=0.0 sent by the no-think rule (every no-think run samples the model under "
        "test at 0.0); SimpleAgentSolver sends none")
    assert "237" in res["reference_note"]


def test_lancer_subset_temperature_and_a_missing_report(monkeypatch):
    monkeypatch.setenv("GBENCH_SWE_LANCER_TEMPERATURE", "0.2")
    rows = {"ic_swe": [_row("ic0", "ic_swe", 100.0, True)], "swe_manager": None}
    runs = _lancer_stubs(monkeypatch, rows, _tasks())
    # --eval-limit 2 is 2 tasks for the suite: one of each task type
    res = SL.run_swe_lancer("m", "http://127.0.0.1:1/v1", 1, limit=2)
    assert any(a.startswith("swelancer.taskset=") for a in runs[0])
    assert res["temperature"] == 0.2 and res["relay"]["sets_body_fields"]["temperature"] == 0.2
    reason = res["leaderboard_comparable_reason"]
    assert "subset run (--eval-limit 2 for the suite: 1 IC SWE, 1 SWE Manager task(s))" in reason
    assert "temperature=0.2 set explicitly" in reason
    assert res["eval_limit_plan"]["samples"] == 2
    assert res["settings"]["swe_manager"]["status"] == "error"
    assert "wrote no report" in res["settings"]["swe_manager"]["error"]
    assert res["settings_with_errors"] == ["swe_manager"]
    assert res["headline_metrics"]["swe_manager_pass_at_1"] is None


def test_lancer_variant_selection():
    assert SL.resolve_variants() == ["ic_swe", "swe_manager"]
    assert SL.resolve_variants("swe_manager") == ["swe_manager"]
    assert SL.resolve_variants(None, "swe_manager,java") == ["swe_manager"]
    with pytest.raises(ValueError, match="not SWE-Lancer task types"):
        SL.resolve_variants("all")


def test_lancer_harness_files_ship_and_agree():
    import py_compile
    here = os.path.join(REPO, "gbench", "docker")
    py_compile.compile(os.path.join(here, "swe_lancer_eval.py"), doraise=True)
    src = open(os.path.join(here, "swe_lancer_eval.py"), encoding="utf-8").read()
    assert "class GbenchSWELancerEval(SWELancerEval)" in src
    assert "await super().get_full_summary(results)" in src
    assert "from swelancer.run_swelancer import main as swelancer_main" in src
    docker = open(os.path.join(here, "swe_lancer_harness.Dockerfile"), encoding="utf-8").read()
    assert f"ARG FRONTIER_EVALS_REF={SL.HARNESS_COMMIT}" in docker
    assert f"ARG SWELANCER_TASKS_SHA256={SL.TASKS_SHA256}" in docker
    assert "uv sync --frozen --no-dev" in docker and "FROM ubuntu:24.04@sha256:" in docker
    entry = open(os.path.join(here, "swe_lancer_entrypoint.sh"), encoding="utf-8").read()
    assert "--probe" in entry and "DOCKER-USER" in entry and 'chown -R' in entry


def test_lancer_doc_matches_the_code():
    doc = _doc("swe_lancer")
    for value in (SL.HARNESS_COMMIT, SL.TASKS_SHA256, "SimpleAgentSolver", "198 IC SWE",
                  "265 SWE Manager", "swelancer/swelancer_x86_monolith:releasev1",
                  "gbench-swe-lancer-harness:51052ced", "`ic_swe_pass_at_1`", "`diamond_earned_usd`",
                  "runner.max_retries=2", "Status: registered, not in `--evals all`",
                  "CRITICAL", "OPENAI_BASE_URL", "DOCKER-USER", "disable_internet=True"):
        assert value in doc, value
    assert not [c for c in doc if ord(c) > 127]
    assert "SWELANCER_HARNESS_DIR" not in doc and "one-shot" not in doc.lower()


# ============================================================================================
# the harness images install from a pinned Ubuntu archive snapshot
# ============================================================================================
@pytest.mark.parametrize("name", ["swebench_v5", "swe_bench_live", "swe_lancer_harness"])
def test_harness_images_install_from_a_pinned_archive_snapshot(name):
    text = open(os.path.join(REPO, "gbench", "docker", f"{name}.Dockerfile"),
                encoding="utf-8").read()
    assert ("FROM ubuntu:24.04@sha256:"
            "33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517") in text
    assert "ARG UBUNTU_SNAPSHOT=20260925T120000Z" in text
    assert "https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" in text
    assert "--allow-downgrades" in text and '= 2 \\' in text   # both archive URIs rewritten
