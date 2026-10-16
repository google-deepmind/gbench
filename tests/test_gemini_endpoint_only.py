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

"""gaia2, wildclawbench, mcp_bench and tau: the Gemini API only, keys off argv, harness errors apart.

* gaia2 passes its keys into the container by name (never on the docker argv); its judge is always
  the in-container Gemini proxy (the judge endpoint and model knobs are gone); it records the judge
  substitution; ARE's percent success rates are reported as percent; harness errors set the status.
* GEMINI_OPENAI_BASE_URL reaches a container or proxy only as base.check_gemini_openai_base_url
  accepts it (the Gemini API over https or a loopback address), whatever a skip-key-validation knob
  says; the proxies of wildclawbench, gaia2 and mcp_bench check it again and send no key elsewhere.
* wildclawbench: no knob points the task containers' judge anywhere but the proxy, which listens on
  loopback or a gateway of the task network only; the model relay never follows a redirect; a
  search-setup failure, an endpoint failure, a Gemini failure, a missing score, a grading error or
  a judge fallback is a harness error: counted, named, status completed_with_errors, left out of the
  headline.

Deterministic: no Docker, no model, no Gemini; only local stub servers on OS-assigned ports.
"""

import ast
import inspect
import json
import logging
import os
import subprocess
import sys
import threading
import types
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import gaia2 as G
from gbench.runners.eval_suites import tau_common as TC
from gbench.runners.eval_suites import wildclawbench as W

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOCKER_DIR = os.path.join(_REPO, "gbench", "docker")
sys.path.insert(0, _DOCKER_DIR)
import gaia2_cascade_judge as GJ          # noqa: E402
import gaia2_run as GRUN                  # noqa: E402
import mcp_bench_cascade_judge as MJ      # noqa: E402
import wildclawbench_cascade_judge as J   # noqa: E402
import wildclawbench_run as RUN           # noqa: E402

# Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
GEMINI_KEY = "AIza" + "EndpointOnlyFAKEkey0000000000000001"
MODEL_KEY = "sk-endpoint-only-model-key-000000000001"
HF_SECRET = "hf_EndpointOnlyFakeToken000000000000001"
HUB_SECRET = "hf_EndpointOnlyFakeHubToken00000000002"
REFUSED_URLS = ["https://api.example.com/v1", "http://generativelanguage.googleapis.com/v1beta/openai/",
                "https://user:pw@generativelanguage.googleapis.com/v1", "http://172.17.0.1:8100/v1",
                "http://169.254.169.254/computeMetadata/v1", "http://0.0.0.0:5555/v1"]
ACCEPTED_URLS = [None, "", "https://generativelanguage.googleapis.com/v1beta/openai/",
                 "http://127.0.0.1:5555/v1beta/openai/", "http://localhost:9/v1", "http://[::1]:7/v1"]


def _read(rel):
    with open(os.path.join(_REPO, rel), encoding="utf-8") as f:
        return f.read()


def _flat(text):
    return " ".join(text.split())


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    for k in ("GEMINI_API_KEYS", "GEMINI_OPENAI_BASE_URL", "GBENCH_GAIA2_JUDGE_ENDPOINT",
              "GBENCH_GAIA2_JUDGE_MODEL", "GBENCH_GAIA2_MODEL_API_KEY", "HF_TOKEN",
              "HUGGING_FACE_HUB_TOKEN", "GBENCH_SHARD", "GBENCH_GAIA2_CONFIG", "GBENCH_GAIA2_LIMIT",
              "GBENCH_GAIA2_MODE", "GBENCH_GAIA2_JUDGE_PORT", "GBENCH_GAIA2_MODEL_ENDPOINT",
              "GBENCH_GAIA2_SKIP_KEY_VALIDATION", "GBENCH_GAIA2_KEEP_WORKDIR",
              "GBENCH_WILDCLAWBENCH_TASK_NETWORK", "GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT",
              "GBENCH_WILDCLAWBENCH_TASK_ENDPOINT", "GBENCH_WILDCLAWBENCH_JUDGE_PORT",
              "GBENCH_WILDCLAWBENCH_TASK_HOST", "GBENCH_WILDCLAWBENCH_MODEL_API_KEY",
              "GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION", "GBENCH_JUDGE_MODELS",
              "GBENCH_JUDGE_MODEL", "GAIA2_JUDGE_ENDPOINT", "GAIA2_JUDGE_MODEL", "http_proxy",
              "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    workdir = tmp_path / "proxy_workdir"
    workdir.mkdir()
    monkeypatch.setenv("WILDCLAW_WORKDIR", str(workdir))
    J.reset_usage()
    yield


def _serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _stop(*servers):
    for s in servers:
        s.shutdown()
        s.server_close()


def _open(req, timeout=15):
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout)


def _events(tmp_path):
    path = tmp_path / "proxy_workdir" / "harness_events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# The container copies of the endpoint check are the shared check, verbatim.
# --------------------------------------------------------------------------- #
def _body_without_docstring(fn):
    tree = ast.parse(inspect.getsource(fn).lstrip())
    node = tree.body[0]
    body = node.body[1:] if (node.body and isinstance(node.body[0], ast.Expr)
                             and isinstance(node.body[0].value, ast.Constant)) else node.body
    return [ast.dump(b) for b in body]


@pytest.mark.parametrize("module", [J, MJ])
def test_container_copies_match_the_shared_check(module):
    assert _body_without_docstring(module.check_gemini_openai_base_url) == \
        _body_without_docstring(base.check_gemini_openai_base_url)
    assert module.GEMINI_OPENAI_BASE_URL == base.GEMINI_OPENAI_BASE_URL
    assert module._GEMINI_API_HOST == base._GEMINI_API_HOST
    for value in REFUSED_URLS + ACCEPTED_URLS:
        assert module.check_gemini_openai_base_url(value) == \
            base.check_gemini_openai_base_url(value), value


# --------------------------------------------------------------------------- #
# gaia2: no key in the container (relays), a fixed judge, the checked endpoint, percent,
# harness errors.
# --------------------------------------------------------------------------- #
def _gaia2_summary(**extra):
    summary = {"harness_accounting": True,
               "macro_success_rate": 42.0, "micro_success_rate": 40.0, "total_scenarios": 5,
               "validated_runs": 5, "no_validation_runs": 0, "exception_runs": 0,
               "agent_invoked": True, "per_capability": {}, "harness_events": {},
               "gemini_usage": {"judge": {"gemini-3.7-flash": 7}}}
    summary.update(extra)
    return summary


def _run_gaia2(monkeypatch, tmp_path, env=None, summary=None):
    monkeypatch.setattr(G, "check_gaia2_prerequisites", lambda: (True, ""))
    # the guarded bridge network's gateway: the relays listen on loopback here, no docker
    monkeypatch.setattr(G.container_guard, "require_metadata_blocked",
                        lambda *a, **k: "unreachable")
    monkeypatch.setattr(G.gemini_relay, "network_gateway", lambda *a, **k: "127.0.0.1")
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    wd = tmp_path / "gaia2_wd"
    wd.mkdir(exist_ok=True)
    monkeypatch.setattr(G.tempfile, "mkdtemp", lambda prefix="", dir=None: str(wd))
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    cap = {"calls": []}

    def fake_run(cmd, *a, **k):
        cap["calls"].append(list(cmd))
        if cmd[:3] == ["docker", "run", "--rm"]:
            cap["cmd"], cap["env"] = list(cmd), dict(k.get("env") or {})
            with open(wd / "gaia2_summary.json", "w") as f:
                json.dump(summary or _gaia2_summary(), f)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(G.subprocess, "run", fake_run)
    result = G.run_gaia2("google/gemma-4-E4B-it", "http://127.0.0.1:9123/v1")
    return result, cap


def test_gaia2_container_gets_no_key_only_the_relay_placeholder_by_name(monkeypatch, tmp_path):
    result, cap = _run_gaia2(monkeypatch, tmp_path, env={
        "GBENCH_GAIA2_MODEL_API_KEY": MODEL_KEY, "HF_TOKEN": HF_SECRET,
        "HUGGING_FACE_HUB_TOKEN": HUB_SECRET})
    cmd = cap["cmd"]
    joined = "\n".join(map(str, cmd))
    env = json.dumps(cap["env"])
    for secret in (GEMINI_KEY, MODEL_KEY, HF_SECRET, HUB_SECRET):
        assert secret not in joined and secret not in env
    # the Gemini relay's placeholder and addresses go by name; no other credential goes at all
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GEMINI_OPENAI_BASE_URL",
                 "GBENCH_GEMINI_RELAY_URL"):
        assert cmd[cmd.index(name) - 1] == "-e"
        assert not any(str(x).startswith(name + "=") for x in cmd)
    assert cap["env"]["GEMINI_API_KEY"].startswith("gbench-gemini-relay-")
    for name in ("GBENCH_MODEL_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        assert name not in cmd and name not in cap["env"]
    assert cmd[cmd.index("--network") + 1] == "bridge"
    assert result["status"] == "success"


def test_gaia2_without_hf_tokens_forwards_no_hf_name(monkeypatch, tmp_path):
    _result, cap = _run_gaia2(monkeypatch, tmp_path)
    assert "HF_TOKEN" not in cap["cmd"] and "HUGGING_FACE_HUB_TOKEN" not in cap["cmd"]
    assert "GBENCH_MODEL_API_KEY" not in cap["cmd"]    # the endpoint relay adds the key


def test_gaia2_judge_endpoint_and_model_knobs_change_nothing(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        result, cap = _run_gaia2(monkeypatch, tmp_path, env={
            "GBENCH_GAIA2_JUDGE_ENDPOINT": "https://judge.example/v1",
            "GBENCH_GAIA2_JUDGE_MODEL": "openai/gpt-x"})
    joined = "\n".join(map(str, cap["cmd"]))
    assert "judge.example" not in joined and "gpt-x" not in joined
    assert "GAIA2_JUDGE_ENDPOINT" not in joined and "GAIA2_JUDGE_MODEL" not in joined
    assert "GBENCH_GAIA2_JUDGE_ENDPOINT no longer changes anything" in caplog.text
    assert "GBENCH_GAIA2_JUDGE_MODEL no longer changes anything" in caplog.text
    assert "GAIA2_JUDGE_PORT=18790" in cap["cmd"]


def test_gaia2_launcher_judge_is_always_the_local_proxy(monkeypatch):
    monkeypatch.setenv("GAIA2_JUDGE_ENDPOINT", "https://judge.example/v1")
    monkeypatch.setenv("GAIA2_JUDGE_MODEL", "openai/gpt-x")
    monkeypatch.setenv("GAIA2_JUDGE_PORT", "18797")
    assert GRUN._judge_args() == ["--judge_provider", "local", "--judge_endpoint",
                                  "http://127.0.0.1:18797/v1", "--judge_model",
                                  "openai/gbench-cascade"]
    src = _read("gbench/docker/gaia2_run.py") + _read("gbench/docker/gaia2_entrypoint.sh")
    assert "GAIA2_JUDGE_ENDPOINT" not in src and "GAIA2_JUDGE_MODEL" not in src


def test_gaia2_relays_to_the_accepted_endpoint_and_gives_the_container_the_relay(monkeypatch,
                                                                                  tmp_path):
    result, cap = _run_gaia2(monkeypatch, tmp_path, env={
        "GEMINI_OPENAI_BASE_URL": "http://127.0.0.1:5555/v1beta/openai/"})
    cmd = cap["cmd"]
    assert cmd[cmd.index("GEMINI_OPENAI_BASE_URL") - 1] == "-e"
    assert not any(str(x).startswith("GEMINI_OPENAI_BASE_URL=") for x in cmd)
    # the container gets the relay's URL; the relay sends the key to the accepted endpoint
    assert cap["env"]["GEMINI_OPENAI_BASE_URL"].startswith("http://127.0.0.1:")
    assert ":5555" not in cap["env"]["GEMINI_OPENAI_BASE_URL"]
    assert result["gemini_relay"]["relays_to"]["openai_compatible"] == \
        "http://127.0.0.1:5555/v1beta/openai"


@pytest.mark.parametrize("url", REFUSED_URLS)
def test_gaia2_refuses_another_endpoint_even_when_the_gate_passes(monkeypatch, tmp_path, url):
    with pytest.raises(RuntimeError) as ei:
        _run_gaia2(monkeypatch, tmp_path, env={"GEMINI_OPENAI_BASE_URL": url})
    assert "GEMINI_OPENAI_BASE_URL" in str(ei.value) and url not in str(ei.value)


def _gaia2_gate_ready(monkeypatch):
    monkeypatch.setattr(G.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(G.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0))
    monkeypatch.setattr(G, "_docker_image_exists", lambda _i: True)
    monkeypatch.setattr(G, "_image_accounts_harness_errors", lambda _i: True)


def test_gaia2_gate_refuses_another_endpoint_even_with_skip_validation(monkeypatch):
    _gaia2_gate_ready(monkeypatch)
    monkeypatch.setenv("GBENCH_GAIA2_SKIP_KEY_VALIDATION", "1")
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setattr(G, "_gemini_key_valid", lambda _k: pytest.fail("no ping expected"))
    ok, why = G.check_gaia2_prerequisites()
    assert not ok and "GEMINI_OPENAI_BASE_URL is not the Gemini API" in why
    assert "api.example.com" not in why


def test_gaia2_key_rejection_does_not_suggest_a_bypass(monkeypatch):
    _gaia2_gate_ready(monkeypatch)
    monkeypatch.setattr(G, "_gemini_key_valid", lambda _k: (False, "HTTP 401"))
    ok, why = G.check_gaia2_prerequisites()
    assert not ok and "rejected by the Gemini API (HTTP 401)" in why and "Check the key." in why
    assert "bypass" not in why and "SKIP_KEY_VALIDATION" not in why


def test_gaia2_headline_is_ares_percent_value(monkeypatch, tmp_path):
    result, _cap = _run_gaia2(monkeypatch, tmp_path)
    assert result["accuracy"] == pytest.approx(42.0)
    assert result["macro_success_rate"] == pytest.approx(0.42)
    assert result["micro_success_rate"] == pytest.approx(0.40)


def test_gaia2_records_the_judge_substitution(monkeypatch, tmp_path):
    result, _cap = _run_gaia2(monkeypatch, tmp_path)
    sub = result["gemini_substitutions"]
    assert len(sub) == 1 and "meta-llama/Meta-Llama-3.3-70B-Instruct" in sub[0]["official"]
    assert "gbench Gemini judge cascade" in sub[0]["served_by"]
    assert sub[0]["models"] == list(base.judge_cascade())
    reason = result["leaderboard_comparable_reason"]
    assert "the judge is the gbench Gemini cascade, not meta-llama/Meta-Llama-3.3-70B" in reason
    assert result["leaderboard_comparable"] is False
    assert result["gemini_usage"] == {"judge": {"gemini-3.7-flash": 7}}


def test_gaia2_harness_errors_set_the_status(monkeypatch, tmp_path):
    result, _cap = _run_gaia2(monkeypatch, tmp_path, summary=_gaia2_summary(
        no_validation_runs=3, harness_events={"judge": 2}))
    assert result["status"] == "completed_with_errors"
    assert result["judge_failures"] == 2 and result["harness_events"] == {"judge": 2}
    reason = result["leaderboard_comparable_reason"]
    assert "3 run(s) could not be validated" in reason
    assert "2 judge call(s) were served by no Gemini cascade model" in reason


def test_gaia2_launcher_reads_the_proxy_records(monkeypatch, tmp_path):
    monkeypatch.setattr(GRUN, "WORKDIR", str(tmp_path))
    stats = {"statistics": {"global": {"macro_success_rate": 50.0}, "per_capability": {
        "time": {"success_rate": 50.0, "validated_runs": 2, "no_validation_runs": 1}}}}
    (tmp_path / "benchmark_stats.json").write_text(json.dumps(stats))
    (tmp_path / "gemini_usage.json").write_text(json.dumps({"judge": {"failed": 1}}))
    (tmp_path / "harness_events.jsonl").write_text(
        json.dumps({"time": 1.0, "kind": "judge", "detail": "x"}) + "\nnot json\n")
    s = GRUN._summarize()
    assert s["harness_events"] == {"judge": 1} and s["gemini_usage"] == {"judge": {"failed": 1}}
    assert s["per_capability"]["time"]["success_rate"] == 50.0
    assert s["per_capability"]["time"]["no_validation_runs"] == 1


class _StubGemini(BaseHTTPRequestHandler):
    seen = []
    reply = {"choices": [{"message": {"role": "assistant", "content": "[[Success]]"}}]}
    status = 200

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        _StubGemini.seen.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                 "body": json.loads(body or b"{}")})
        payload = json.dumps(_StubGemini.reply).encode()
        self.send_response(_StubGemini.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST


RELAY_PLACEHOLDER = "gbench-gemini-relay-" + "0f" * 24


@pytest.fixture
def gaia2_proxy(monkeypatch):
    """The proxy with the Gemini relay's placeholder; the stub stands in for the relay."""
    _StubGemini.seen.clear()
    _StubGemini.status = 200
    gemini, gemini_url = _serve(_StubGemini)
    monkeypatch.setenv("GEMINI_API_KEY", RELAY_PLACEHOLDER)
    monkeypatch.setenv("GBENCH_GEMINI_RELAY_URL", gemini_url)
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", gemini_url + "/v1beta/openai")
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-a,gemini-b")
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", "1")
    proxy, proxy_url = _serve(GJ._Handler)
    try:
        yield proxy_url
    finally:
        _stop(proxy, gemini)


def _post(url, payload):
    return urllib.request.Request(url, method="POST", data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json",
                                           "Authorization": f"Bearer {MODEL_KEY}"})


def test_gaia2_proxy_answers_the_judge_with_the_gemini_cascade(gaia2_proxy):
    with _open(_post(gaia2_proxy + "/v1/chat/completions",
                     {"model": "gbench-cascade", "temperature": 0.7,
                      "messages": [{"role": "user", "content": "judge this"}]})) as r:
        reply = json.loads(r.read())
    assert reply["choices"][0]["message"]["content"] == "[[Success]]"
    seen = _StubGemini.seen[-1]
    assert seen["path"] == "/v1beta/openai/chat/completions"
    assert seen["auth"] == f"Bearer {RELAY_PLACEHOLDER}"    # the relay's placeholder, not ARE's
    assert seen["body"]["model"] == "gemini-a" and seen["body"]["temperature"] == 0.0


def test_gaia2_proxy_has_no_relay_and_no_search(gaia2_proxy):
    with _open(urllib.request.Request(gaia2_proxy + "/model/v1/chat/completions?x=1")) as r:
        assert json.loads(r.read()) == {"status": "ok"}   # a health reply, nothing relayed
    with pytest.raises(urllib.error.HTTPError) as ei:
        _open(_post(gaia2_proxy + "/model/v1/other", {}))
    assert ei.value.code == 404
    assert _StubGemini.seen == []


def test_gaia2_proxy_outage_is_503_and_a_harness_event(gaia2_proxy, tmp_path, monkeypatch):
    _StubGemini.status = 500
    monkeypatch.setenv("GBENCH_JUDGE_BACKOFF", "0")
    with pytest.raises(urllib.error.HTTPError) as ei:
        _open(_post(gaia2_proxy + "/v1/chat/completions",
                    {"messages": [{"role": "user", "content": "x"}]}))
    assert ei.value.code == 503
    body = ei.value.read().decode()
    assert "JUDGE_OUTAGE" in body and RELAY_PLACEHOLDER not in body
    assert [e["kind"] for e in _events(tmp_path)] == ["judge"]


def test_gaia2_proxy_does_not_start_on_another_endpoint(monkeypatch):
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setattr(GJ, "ThreadingHTTPServer", lambda *a, **k: pytest.fail("bound a port"))
    assert GJ.main() == 7
    monkeypatch.delenv("GEMINI_API_KEY")
    assert GJ.main() == 2


def test_gaia2_image_entrypoint_and_dockerfile():
    entry = _read("gbench/docker/gaia2_entrypoint.sh")
    assert ("env -u GEMINI_API_KEYS -u GEMINI_API_KEY -u GBENCH_GEMINI_RELAY_URL "
            "-u GEMINI_OPENAI_BASE_URL") in entry
    assert "exit 6" in entry and "exit 5" in entry
    assert 'WILDCLAW_WORKDIR="$GAIA2_WORKDIR"' in entry
    docker = _read("gbench/docker/gaia2.Dockerfile")
    assert "COPY gaia2_cascade_judge.py /app/adapter/gaia2_cascade_judge.py" in docker
    assert "COPY wildclawbench_cascade_judge.py /app/adapter/wildclawbench_cascade_judge.py" in docker
    assert GJ.LISTEN_HOST == "127.0.0.1"


def test_gaia2_docs_follow_the_code():
    doc = _read("docs/evals/gaia2.md")
    src = _read("gbench/runners/eval_suites/gaia2.py")
    import re
    for knob in sorted(set(re.findall(r"GBENCH_GAIA2_[A-Z_]+", src))):
        assert knob in doc, knob
    flat = _flat(doc)
    assert "`GBENCH_GAIA2_JUDGE_ENDPOINT` and `GBENCH_GAIA2_JUDGE_MODEL` no longer change anything" \
        in flat
    assert "| `GBENCH_GAIA2_JUDGE_ENDPOINT`" not in doc
    assert "go into the container **by name**" in flat
    assert "**`--network bridge`**" in flat
    assert f"`{G.HARNESS_ERRORS_LABEL}={G.HARNESS_ERRORS_VERSION}`" in flat
    assert "ARE writes its success rates in percent" in flat
    assert "completed_with_errors" in flat


# --------------------------------------------------------------------------- #
# wildclawbench: the proxy's endpoint, forwarding, proxy host, ports, /v1.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("url", REFUSED_URLS)
def test_wildclaw_proxy_sends_no_key_to_another_endpoint(monkeypatch, tmp_path, url):
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", url)
    monkeypatch.setattr(J.urllib.request, "urlopen", lambda *a, **k: pytest.fail("a key was sent"))
    with pytest.raises(J.GeminiEndpointRefused) as ei:
        J.cascade_chat_completion({"messages": [{"role": "user", "content": "x"}]})
    assert url not in str(ei.value)
    with pytest.raises(J.GeminiEndpointRefused):
        J.cascade_chat_completion({"tools": [{"type": "openrouter:web_search"}],
                                   "messages": [{"role": "user", "content": "x"}]})
    assert [e["kind"] for e in _events(tmp_path)] == ["judge", "judge_web_search"]


def test_wildclaw_proxy_does_not_start_on_another_endpoint(monkeypatch):
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("GBENCH_MODEL_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setattr(J, "search_tool", lambda: object())
    monkeypatch.setattr(J, "ThreadingHTTPServer", lambda *a, **k: pytest.fail("bound a port"))
    assert J.main() == 7


def test_wildclaw_proxy_uses_an_accepted_loopback_endpoint(monkeypatch):
    _StubGemini.seen.clear()
    _StubGemini.status = 200
    gemini, gemini_url = _serve(_StubGemini)
    try:
        monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", gemini_url + "/v1beta/openai/")
        monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-a")
        out = J.cascade_chat_completion({"messages": [{"role": "user", "content": "x"}]})
    finally:
        _stop(gemini)
    assert out["choices"][0]["message"]["content"] == "[[Success]]"
    assert _StubGemini.seen[-1]["path"] == "/v1beta/openai/chat/completions"


def _run_wildclaw(monkeypatch, tmp_path, env=None, summary_extra=None, gateways=None):
    harness = tmp_path / "harness"
    harness.mkdir(exist_ok=True)
    monkeypatch.setattr(W, "check_wildclawbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(W, "_harness_dir", lambda: str(harness))
    monkeypatch.setattr(W, "free_port", lambda preferred=None: preferred)
    wd = tmp_path / "wc_wd"
    wd.mkdir(exist_ok=True)
    monkeypatch.setattr(W.tempfile, "mkdtemp", lambda prefix="": str(wd))
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    cap = {"calls": []}

    def fake_run(cmd, *a, **k):
        cap["calls"].append(list(cmd))
        if cmd[:3] == ["docker", "network", "inspect"]:
            cfg = [{"Subnet": "x", "Gateway": g} for g in (gateways or [])]
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(cfg), stderr="")
        if cmd[:3] == ["docker", "run", "--rm"]:
            cap["cmd"], cap["env"] = list(cmd), dict(k.get("env") or {})
            summary = {"n_tasks": 60, "n_scored": 60, "n_missing": 0, "global_mean": 0.25,
                       "judge_fallback_tasks": 0, "n_search_setup_failures": 0,
                       "n_harness_errors": 0, "harness_error_kinds": {}, "harness_events": {},
                       "n_unattributed_harness_events": 0}
            summary.update(summary_extra or {})
            with open(wd / "wildclawbench_summary.json", "w") as f:
                json.dump(summary, f)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(W.subprocess, "run", fake_run)
    result = W.run_wildclawbench("google/gemma-4-E4B-it", "http://127.0.0.1:9123/v1")
    return result, cap


def test_wildclaw_forwards_only_the_accepted_endpoint_by_name(monkeypatch, tmp_path):
    _result, cap = _run_wildclaw(monkeypatch, tmp_path, env={
        "GEMINI_OPENAI_BASE_URL": "http://127.0.0.1:5555/v1beta/openai/"})
    cmd = cap["cmd"]
    assert cmd[cmd.index("GEMINI_OPENAI_BASE_URL") - 1] == "-e"
    assert not any(str(x).startswith("GEMINI_OPENAI_BASE_URL=") for x in cmd)
    assert cap["env"]["GEMINI_OPENAI_BASE_URL"] == "http://127.0.0.1:5555/v1beta/openai"
    assert "GEMINI_OPENAI_BASE_URL" not in W._FORWARDED_KNOBS


def test_wildclaw_unset_endpoint_is_not_forwarded(monkeypatch, tmp_path):
    _result, cap = _run_wildclaw(monkeypatch, tmp_path)
    assert "GEMINI_OPENAI_BASE_URL" not in cap["cmd"]
    assert "GEMINI_OPENAI_BASE_URL" not in cap["env"]


@pytest.mark.parametrize("url", REFUSED_URLS)
def test_wildclaw_refuses_another_endpoint_even_when_the_gate_passes(monkeypatch, tmp_path, url):
    with pytest.raises(RuntimeError) as ei:
        _run_wildclaw(monkeypatch, tmp_path, env={"GEMINI_OPENAI_BASE_URL": url})
    assert "GEMINI_OPENAI_BASE_URL" in str(ei.value) and url not in str(ei.value)


def _wildclaw_gate_ready(tmp_path, monkeypatch):
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "run_batch.py").write_text("x")
    (tmp_path / "workspace").mkdir()
    monkeypatch.setattr(W.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(W.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="", stderr=""))
    monkeypatch.setattr(W, "_docker_image_exists", lambda _i: True)
    monkeypatch.setenv("GBENCH_WILDCLAWBENCH_HOST_DIR", str(tmp_path))


def test_wildclaw_key_rejection_does_not_suggest_a_bypass(tmp_path, monkeypatch):
    _wildclaw_gate_ready(tmp_path, monkeypatch)
    monkeypatch.setattr(W, "_gemini_key_valid", lambda _k: (False, "HTTP 401"))
    ok, why = W.check_wildclawbench_prerequisites()
    assert not ok and "was rejected by the Gemini API (HTTP 401)" in why and "Check the key." in why
    assert "bypass" not in why and "SKIP_KEY_VALIDATION" not in why


def test_wildclaw_gate_refuses_an_image_without_the_harness_error_label(tmp_path, monkeypatch):
    _wildclaw_gate_ready(tmp_path, monkeypatch)
    monkeypatch.setattr(W, "_gemini_key_valid", lambda _k: (True, ""))
    ok, why = W.check_wildclawbench_prerequisites()      # docker inspect prints no label
    assert not ok and "predates the harness-error accounting" in why
    assert f"{W.HARNESS_ERRORS_LABEL}={W.HARNESS_ERRORS_VERSION}" in why
    seen = []

    def inspect(cmd, *a, **k):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=W.HARNESS_ERRORS_VERSION + "\n",
                                           stderr="")
    monkeypatch.setattr(W.subprocess, "run", inspect)
    assert W.check_wildclawbench_prerequisites() == (True, "")
    assert any(W.HARNESS_ERRORS_LABEL in " ".join(c) for c in seen)
    docker = _read("gbench/docker/wildclawbench.Dockerfile")
    assert f'LABEL {W.HARNESS_ERRORS_LABEL}="{W.HARNESS_ERRORS_VERSION}"' in docker


def test_wildclaw_removed_judge_endpoint_knob_moves_nothing(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        _result, cap = _run_wildclaw(monkeypatch, tmp_path, env={
            "GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT": "http://10.1.2.3:18795/v1"})
    cmd = cap["cmd"]
    # the default task network is the docker bridge: the proxy is at the bridge gateway
    assert "WILDCLAW_JUDGE_HOST=172.17.0.1" in cmd
    assert "WILDCLAW_JUDGE_ENDPOINT=http://172.17.0.1:18795/v1" in cmd
    assert "WILDCLAW_SEARCH_ORIGIN=http://172.17.0.1:18795" in cmd
    assert not any("10.1.2.3" in str(x) for x in cmd)
    assert "GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT no longer changes anything" in caplog.text


def test_wildclaw_bridge_default_is_the_bridge_gateway_without_inspect(monkeypatch, tmp_path):
    _result, cap = _run_wildclaw(monkeypatch, tmp_path, env={
        "GBENCH_WILDCLAWBENCH_TASK_NETWORK": "bridge"})
    assert "WILDCLAW_JUDGE_HOST=172.17.0.1" in cap["cmd"]
    assert not any(c[:3] == ["docker", "network", "inspect"] for c in cap["calls"])


def test_wildclaw_task_host_must_be_a_gateway_of_the_task_network(monkeypatch, tmp_path):
    _result, cap = _run_wildclaw(monkeypatch, tmp_path, env={
        "GBENCH_WILDCLAWBENCH_TASK_NETWORK": "mynet", "GBENCH_WILDCLAWBENCH_TASK_HOST": "172.18.0.1"},
        gateways=["172.18.0.1"])
    cmd = cap["cmd"]
    assert "WILDCLAW_JUDGE_HOST=172.18.0.1" in cmd
    assert "WILDCLAW_MODEL_RELAY_URL=http://172.18.0.1:18795/model/v1" in cmd
    assert ["docker", "network", "inspect", "mynet", "--format", "{{json .IPAM.Config}}"] in \
        cap["calls"]


@pytest.mark.parametrize("host", ["10.0.0.5", "host.docker.internal", "0.0.0.0", "127.0.0.1"])
def test_wildclaw_task_host_elsewhere_is_a_hard_error(monkeypatch, tmp_path, host):
    with pytest.raises(RuntimeError) as ei:
        _run_wildclaw(monkeypatch, tmp_path, env={
            "GBENCH_WILDCLAWBENCH_TASK_NETWORK": "bridge", "GBENCH_WILDCLAWBENCH_TASK_HOST": host},
            gateways=["172.17.0.1"])
    assert "is not a gateway address of the task network 'bridge'" in str(ei.value)


def test_wildclaw_ipv6_gateway_is_bracketed(monkeypatch, tmp_path):
    _result, cap = _run_wildclaw(monkeypatch, tmp_path, env={
        "GBENCH_WILDCLAWBENCH_TASK_NETWORK": "v6net", "GBENCH_WILDCLAWBENCH_TASK_HOST": "fd00::1"},
        gateways=["fd00:0::1"])
    assert "WILDCLAW_JUDGE_HOST=fd00::1" in cap["cmd"]
    assert "WILDCLAW_JUDGE_ENDPOINT=http://[fd00::1]:18795/v1" in cap["cmd"]


@pytest.mark.parametrize("port", ["18790", "18792", "18793"])
def test_wildclaw_refuses_every_openclaw_port_with_the_host_network(monkeypatch, tmp_path, port):
    with pytest.raises(RuntimeError) as ei:
        _run_wildclaw(monkeypatch, tmp_path, env={"GBENCH_WILDCLAWBENCH_TASK_NETWORK": "host",
                                                  "GBENCH_WILDCLAWBENCH_JUDGE_PORT": port})
    assert f"GBENCH_WILDCLAWBENCH_JUDGE_PORT={port}" in str(ei.value)
    assert "18789-18793, 18800-18899" in str(ei.value)


def test_wildclaw_default_port_is_outside_the_openclaw_ports():
    assert W._JUDGE_PORT_DEFAULT == 18795 and W._JUDGE_PORT_DEFAULT not in W._OPENCLAW_PORTS
    assert W._OPENCLAW_PORTS == frozenset({*range(18789, 18794), *range(18800, 18900)})


@pytest.mark.parametrize("network,override,expected", [
    ("host", "http://127.0.0.1:8000", "http://127.0.0.1:8000/v1"),
    ("host", "http://127.0.0.1:8000/v1/", "http://127.0.0.1:8000/v1"),
    ("bridge", "http://172.17.0.1:8000", "http://172.17.0.1:8000/v1"),
])
def test_wildclaw_model_url_override_gets_v1(monkeypatch, tmp_path, network, override, expected):
    _result, cap = _run_wildclaw(monkeypatch, tmp_path, env={
        "GBENCH_WILDCLAWBENCH_TASK_NETWORK": network,
        "GBENCH_WILDCLAWBENCH_TASK_ENDPOINT": override})
    # the orchestrator's proxy reaches the model through gbench's temperature relay (every run
    # sets the temperature), which relays to the overridden model URL
    url = next(x.split("=", 1)[1] for x in cap["cmd"] if x.startswith("GBENCH_MODEL_BASE_URL="))
    assert url.startswith("http://127.0.0.1:") and url.endswith("/v1")
    assert _result["model_temperature_relay"]["relays_to"] == expected


# --------------------------------------------------------------------------- #
# wildclawbench: the model relay never follows a redirect; endpoint failures are harness events.
# --------------------------------------------------------------------------- #
class _Other(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        _Other.seen.append(self.headers.get("Authorization"))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")


class _Upstream(BaseHTTPRequestHandler):
    mode = "ok"
    location = ""

    def log_message(self, *a):
        pass

    def do_GET(self):
        if _Upstream.mode == "redirect":
            self.send_response(302)
            self.send_header("Location", _Upstream.location)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if _Upstream.mode == "cut":
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b"partial")
            self.wfile.flush()
            self.connection.close()
            return
        code = {"ok": 200, "bad_request": 400, "overloaded": 503, "rate": 429}[_Upstream.mode]
        payload = b'{"x": 1}'
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def relay(monkeypatch):
    _Other.seen.clear()
    _Upstream.mode = "ok"
    other, other_url = _serve(_Other)
    upstream, up_url = _serve(_Upstream)
    proxy, proxy_url = _serve(J._Handler)
    _Upstream.location = other_url + "/steal"
    monkeypatch.setenv("GBENCH_MODEL_BASE_URL", up_url + "/v1")
    monkeypatch.setenv("GBENCH_MODEL_API_KEY", MODEL_KEY)
    try:
        yield proxy_url
    finally:
        _stop(proxy, upstream, other)


def test_relay_passes_a_redirect_back_and_never_follows_it(relay, tmp_path):
    _Upstream.mode = "redirect"
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}), J._NoRedirect())
    with pytest.raises(urllib.error.HTTPError) as ei:
        client.open(urllib.request.Request(relay + "/model/v1/models"), timeout=15)
    assert ei.value.code == 302 and ei.value.headers["Location"].endswith("/steal")
    assert _Other.seen == []                       # the second host is never contacted
    assert _events(tmp_path) == []


@pytest.mark.parametrize("mode,event", [("ok", False), ("bad_request", False),
                                        ("overloaded", True), ("rate", True)])
def test_relay_endpoint_failure_status_is_a_harness_event(relay, tmp_path, mode, event):
    _Upstream.mode = mode
    try:
        with _open(urllib.request.Request(relay + "/model/v1/models")) as r:
            r.read()
    except urllib.error.HTTPError:
        pass
    kinds = [e["kind"] for e in _events(tmp_path)]
    assert kinds == (["model_endpoint"] if event else [])


def test_relay_unreachable_and_cut_off_replies_are_harness_events(relay, tmp_path, monkeypatch):
    _Upstream.mode = "cut"
    try:
        with _open(urllib.request.Request(relay + "/model/v1/models")) as r:
            r.read()
    except Exception:
        pass
    monkeypatch.setenv("GBENCH_MODEL_BASE_URL", f"http://127.0.0.1:{base.free_port()}/v1")
    with pytest.raises(urllib.error.HTTPError) as ei:
        _open(urllib.request.Request(relay + "/model/v1/models"))
    assert ei.value.code == 502
    events = _events(tmp_path)
    assert [e["kind"] for e in events] == ["model_endpoint", "model_endpoint"]
    assert "cut off" in events[0]["detail"] and "unreachable" in events[1]["detail"]
    assert all(MODEL_KEY not in e["detail"] for e in events)


def test_agent_web_search_failure_is_a_harness_event(monkeypatch, tmp_path):
    fake = types.SimpleNamespace(search_with_fallback=lambda q, max_results=5, deadline=None: (
        [{"title": "error", "url": "", "snippet": f"boom {GEMINI_KEY}"}],
        "all-grounding-models-failed"))
    monkeypatch.setattr(J, "search_tool", lambda: fake)
    code, body = J.brave_search_reply("q=hello")
    assert code == 502
    events = _events(tmp_path)
    assert [e["kind"] for e in events] == ["agent_web_search"]
    assert GEMINI_KEY not in json.dumps(events) and GEMINI_KEY not in json.dumps(body)


# --------------------------------------------------------------------------- #
# wildclawbench launcher: harness errors per task, left out of the headline.
# --------------------------------------------------------------------------- #
def _harness(tmp_path, tasks):
    for tid, mod in tasks.items():
        cat = tid.split("_task_")[0]
        d = tmp_path / "h" / "tasks" / cat
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{tid}.md").write_text(
            f"---\nid: {tid}\nname: X\ncategory: {cat}\ntimeout_seconds: 60\nmodality: {mod}\n---\n",
            encoding="utf-8")
    return str(tmp_path / "h")


def _score(out_root, tid, score):
    cat = tid.split("_task_")[0]
    d = os.path.join(out_root, "openclaw", cat, tid, "m_20260101_0000_abc123")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "score.json"), "w") as f:
        json.dump(score, f)


def test_search_setup_failure_is_recorded_with_its_task_and_left_out(monkeypatch, tmp_path):
    wd = tmp_path / "wd"
    wd.mkdir()
    monkeypatch.setattr(RUN, "WORKDIR", str(wd))
    monkeypatch.setattr(RUN, "HARNESS_DIR", _harness(tmp_path, {
        "01_Productivity_Flow_task_1": "pure-text", "01_Productivity_Flow_task_2": "pure-text"}))
    monkeypatch.delenv("WILDCLAW_TASK", raising=False)
    monkeypatch.delenv("WILDCLAW_CATEGORIES", raising=False)
    monkeypatch.setenv("WILDCLAW_SEARCH_ORIGIN", "http://127.0.0.1:18795")
    monkeypatch.setenv("WILDCLAW_TASK_NETWORK", "host")

    def fake_subprocess_run(cmd, *a, **k):
        if cmd[:3] == ["docker", "run", "-d"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="cid\n", stderr="")
        if cmd[:2] == ["docker", "exec"]:      # the search setup fails in every container
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="cannot delete skill")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    docker_utils = types.SimpleNamespace(
        subprocess=types.SimpleNamespace(run=fake_subprocess_run))
    RUN._patch_task_containers(docker_utils)
    out_root = str(tmp_path / "out")

    def run_single_task(task, *a, **k):
        if task["task_id"] == "01_Productivity_Flow_task_2":
            r = docker_utils.subprocess.run(["docker", "run", "-d", "--name", "c2", "img"])
            assert r.returncode == 1               # start_container would raise here
            return {"task_id": "c2", "error": "Container startup failed"}
        _score(out_root, task["task_id"], {"overall_score": 0.8})
        return {"task_id": "c1", "error": None}
    rb = types.SimpleNamespace(run_single_task=run_single_task)
    RUN._track_task_windows(rb)
    rb.run_single_task({"task_id": "01_Productivity_Flow_task_1"})
    rb.run_single_task({"task_id": "01_Productivity_Flow_task_2"})

    records = [json.loads(x) for x in (wd / RUN.SEARCH_SETUP_FAILURES).read_text().splitlines()]
    assert [r["task_id"] for r in records] == ["01_Productivity_Flow_task_2"]
    windows = [json.loads(x) for x in (wd / RUN.TASK_WINDOWS).read_text().splitlines()]
    assert [w["task_id"] for w in windows] == ["01_Productivity_Flow_task_1",
                                               "01_Productivity_Flow_task_2"]
    assert windows[1]["backend_error"] is True and windows[0]["backend_error"] is False
    summary = RUN._summarize(out_root, RUN._task_index())
    kinds = {r["task_id"]: r["harness_errors"] for r in summary["per_task"]}
    assert kinds["01_Productivity_Flow_task_2"] == ["search_setup", "task_start"]
    assert kinds["01_Productivity_Flow_task_1"] == []
    assert summary["global_mean"] == pytest.approx(0.8)      # not (0.8 + 0) / 2
    assert summary["n_scored"] == 1 and summary["n_harness_errors"] == 1
    assert summary["harness_error_kinds"] == {"search_setup": 1, "task_start": 1}


def test_proxy_events_are_attributed_to_the_tasks_running_then(monkeypatch, tmp_path):
    wd = tmp_path / "wd"
    wd.mkdir()
    monkeypatch.setattr(RUN, "WORKDIR", str(wd))
    tids = ["01_Productivity_Flow_task_1", "01_Productivity_Flow_task_2",
            "05_Creative_Synthesis_task_3"]
    monkeypatch.setattr(RUN, "HARNESS_DIR", _harness(tmp_path, {
        tids[0]: "pure-text", tids[1]: "pure-text", tids[2]: "multimodal"}))
    monkeypatch.delenv("WILDCLAW_TASK", raising=False)
    monkeypatch.delenv("WILDCLAW_CATEGORIES", raising=False)
    out_root = str(tmp_path / "out")
    for tid, sc in zip(tids, (1.0, 0.0, 0.5)):
        _score(out_root, tid, {"overall_score": sc})
    # task_2's grade() exec (run after its leftover processes were stopped) holds the judge
    # failure at 260: a judge failure counts only in such an exec of the task
    lines = [{"task_id": tids[0], "start": 100.0, "end": 200.0},
             {"task_id": tids[1], "start": 200.5, "end": 300.0,
              "grading": [{"start": 255.0, "end": 280.0, "timed_out": False, "isolated": True}]},
             {"task_id": tids[2], "start": 300.5, "end": 400.0}]
    (wd / RUN.TASK_WINDOWS).write_text("".join(json.dumps(x) + "\n" for x in lines))
    events = [{"time": 250.0, "kind": "model_endpoint"}, {"time": 260.0, "kind": "judge"},
              {"time": 350.0, "kind": "agent_web_search"}, {"time": 500.0, "kind": "judge"}]
    (wd / RUN.HARNESS_EVENTS).write_text("".join(json.dumps(x) + "\n" for x in events))
    summary = RUN._summarize(out_root, RUN._task_index())
    kinds = {r["task_id"]: r["harness_errors"] for r in summary["per_task"]}
    assert kinds == {tids[0]: [], tids[1]: ["model_endpoint", "judge"],
                     tids[2]: ["agent_web_search"]}
    assert summary["global_mean"] == pytest.approx(1.0)       # only task_1 is clean
    assert summary["multimodal_mean"] is None and summary["pure_text_mean"] == pytest.approx(1.0)
    assert summary["harness_events"] == {"model_endpoint": 1, "judge": 2, "agent_web_search": 1}
    assert summary["n_unattributed_harness_events"] == 1
    assert summary["n_harness_errors"] == 2


def _runs(wd, records):
    (wd / RUN.TASK_WINDOWS).write_text("".join(json.dumps(x) + "\n" for x in records))


def test_the_models_own_failures_stay_in_the_headline(monkeypatch, tmp_path):
    """A grade() reporting the model's missing output next to numeric scores, and a grade() that
    raised on its own, are the model's outcome (scored 0, as run_batch does); grading that could
    not run (docker copy failure, the 120 s cap) is a harness error."""
    wd = tmp_path / "wd"
    wd.mkdir()
    monkeypatch.setattr(RUN, "WORKDIR", str(wd))
    tids = [f"03_Social_Interaction_task_{i}" for i in range(1, 6)]
    monkeypatch.setattr(RUN, "HARNESS_DIR", _harness(tmp_path, {t: "pure-text" for t in tids}))
    monkeypatch.delenv("WILDCLAW_TASK", raising=False)
    monkeypatch.delenv("WILDCLAW_CATEGORIES", raising=False)
    out_root = str(tmp_path / "out")
    _score(out_root, tids[0], {"overall_score": 1.0})
    _score(out_root, tids[1], {"overall_score": 0.0, "error": "results.md not found"})
    # tids[2]: grade() raised on its own -> no score.json, run_single_task returned the error
    _score(out_root, tids[3], {"overall_score": 0.0, "error":
                               "Command '['docker', 'exec']' timed out after 120 seconds"})
    # tids[4]: upstream's docker cp of the grade runner failed -> no score.json
    base_rec = {"start": 1.0, "end": 2.0, "raised": False, "backend_error": False}
    _runs(wd, [dict(base_rec, task_id=tids[0], grading_error=None),
               dict(base_rec, task_id=tids[1], grading_error="results.md not found"),
               dict(base_rec, task_id=tids[2], grading_error="grade script failed: KeyError"),
               dict(base_rec, task_id=tids[3], grading_error="Command timed out after 120 seconds"),
               dict(base_rec, task_id=tids[4], grading_error="docker cp failed: no such container")])
    summary = RUN._summarize(out_root, RUN._task_index())
    kinds = {r["task_id"]: r["harness_errors"] for r in summary["per_task"]}
    assert kinds == {tids[0]: [], tids[1]: [], tids[2]: [], tids[3]: ["grading_infra"],
                     tids[4]: ["grading_infra"]}
    assert summary["n_headline_tasks"] == 3 and summary["n_scored"] == 2
    assert summary["global_mean"] == pytest.approx(1.0 / 3)   # 1.0 + 0.0 + 0 (raised) over 3
    assert RUN.grading_infra_failure("transcript loader module not found: /x")
    assert not RUN.grading_infra_failure("json parse failed: no valid JSON in stdout")


def test_a_task_whose_run_raised_is_a_harness_error(monkeypatch, tmp_path):
    wd = tmp_path / "wd"
    wd.mkdir()
    monkeypatch.setattr(RUN, "WORKDIR", str(wd))

    def boom(task, *a, **k):
        raise RuntimeError("harness crashed")
    rb = types.SimpleNamespace(run_single_task=boom)
    RUN._track_task_windows(rb)
    with pytest.raises(RuntimeError):
        rb.run_single_task({"task_id": "01_Productivity_Flow_task_1"})
    rec = json.loads((wd / RUN.TASK_WINDOWS).read_text())
    assert rec["raised"] is True and rec["task_id"] == "01_Productivity_Flow_task_1"
    assert rec["end"] >= rec["start"]


def test_every_task_harness_errored_leaves_no_headline(monkeypatch, tmp_path):
    wd = tmp_path / "wd"
    wd.mkdir()
    monkeypatch.setattr(RUN, "WORKDIR", str(wd))
    monkeypatch.setattr(RUN, "HARNESS_DIR", _harness(tmp_path, {
        "01_Productivity_Flow_task_1": "pure-text"}))
    monkeypatch.delenv("WILDCLAW_TASK", raising=False)
    monkeypatch.delenv("WILDCLAW_CATEGORIES", raising=False)
    summary = RUN._summarize(str(tmp_path / "out"), RUN._task_index())
    assert summary["global_mean"] is None and summary["n_scored"] == 0


# --------------------------------------------------------------------------- #
# wildclawbench runner: harness errors -> completed_with_errors, named, not scored.
# --------------------------------------------------------------------------- #
def test_runner_search_setup_failure_is_a_harness_error(monkeypatch, tmp_path):
    result, _cap = _run_wildclaw(monkeypatch, tmp_path, summary_extra={
        "n_search_setup_failures": 1, "n_harness_errors": 1, "n_scored": 59, "n_missing": 1,
        "n_headline_tasks": 59, "global_mean": 0.5,
        "harness_error_kinds": {"search_setup": 1, "task_start": 1}})
    assert result["status"] == "completed_with_errors"
    assert result["accuracy"] == pytest.approx(50.0)
    assert result["n_harness_errors"] == 1 and result["n_scored"] == 59
    assert result["n_headline_tasks"] == 59
    assert result["harness_error_kinds"] == {"search_setup": 1, "task_start": 1}
    reason = result["leaderboard_comparable_reason"]
    assert "1 task container(s) could not be pointed at the Gemini web search and did not run " \
        "(harness errors, left out of the headline)" in reason
    assert "1 of 60 task(s) hit a harness error and are left out of the headline " \
        "(1 web-search setup failed, did not run, 1 task container or agent backend failed " \
        "before the agent ran)" in reason
    assert "counted as 0" not in reason


def test_runner_endpoint_and_gemini_failures_are_named(monkeypatch, tmp_path):
    result, _cap = _run_wildclaw(monkeypatch, tmp_path, summary_extra={
        "n_harness_errors": 2, "n_scored": 58,
        "harness_error_kinds": {"model_endpoint": 1, "judge": 1, "agent_web_search": 1},
        "harness_events": {"model_endpoint": 3, "judge": 1, "agent_web_search": 1},
        "n_unattributed_harness_events": 1})
    assert result["status"] == "completed_with_errors"
    reason = result["leaderboard_comparable_reason"]
    assert "1 model endpoint failure during the task" in reason
    assert "1 Gemini judge outage" in reason and "1 Gemini web search failure" in reason
    assert "1 harness event(s) outside every task run" in reason
    assert result["harness_events"] == {"model_endpoint": 3, "judge": 1, "agent_web_search": 1}


def test_runner_without_harness_errors_is_a_success(monkeypatch, tmp_path):
    result, _cap = _run_wildclaw(monkeypatch, tmp_path)
    assert result["status"] == "success" and result["n_harness_errors"] == 0
    assert "harness error" not in result["leaderboard_comparable_reason"]


def test_runner_with_no_task_outside_harness_errors_hard_errors(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError) as ei:
        _run_wildclaw(monkeypatch, tmp_path, summary_extra={
            "n_scored": 0, "global_mean": None, "n_harness_errors": 60,
            "harness_error_kinds": {"not_run": 60}})
    assert "no task produced a score outside harness errors" in str(ei.value)
    assert "60 never run" in str(ei.value)


def test_wildclaw_docs_follow_the_code():
    doc = _read("docs/evals/wildclawbench.md")
    flat = _flat(doc)
    for kind, _label in W.HARNESS_ERROR_LABELS:
        assert f"`{kind}`" in doc, kind
    assert [k for k, _ in W.HARNESS_ERROR_LABELS] == list(RUN.HARNESS_ERROR_KINDS)
    assert "`GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT` no longer changes anything" in flat
    assert "| `GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT`" not in doc
    assert "it scores 0" not in doc and "counted as 0" not in doc
    assert "It never follows a redirect" in flat
    assert "must be a gateway address of the task network" in flat
    assert "`18789`-`18793`" in doc
    assert f"org.gbench.wildclawbench-harness-errors={W.HARNESS_ERRORS_VERSION}" in doc


# --------------------------------------------------------------------------- #
# mcp_bench: the in-container judge sends no key to another endpoint.
# --------------------------------------------------------------------------- #
def _fake_openai(monkeypatch):
    built = []

    class FakeAsyncOpenAI:
        def __init__(self, api_key, base_url, max_retries=None):
            # one attempt per request: the cascade moves on to the next model itself
            assert max_retries == 0
            built.append(base_url)
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI))
    return built


@pytest.mark.parametrize("url", REFUSED_URLS)
def test_mcp_bench_judge_builds_no_client_for_another_endpoint(monkeypatch, url):
    built = _fake_openai(monkeypatch)
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", url)
    judge = MJ.CascadeGeminiJudge()
    with pytest.raises(RuntimeError) as ei:
        judge._next_client()
    assert "GEMINI_OPENAI_BASE_URL is not the Gemini API" in str(ei.value)
    assert url not in str(ei.value) and built == []


@pytest.mark.parametrize("url,expected", [
    (None, "https://generativelanguage.googleapis.com/v1beta/openai/"),
    ("http://127.0.0.1:5555/v1beta/openai", "http://127.0.0.1:5555/v1beta/openai/")])
def test_mcp_bench_judge_uses_the_accepted_endpoint(monkeypatch, url, expected):
    built = _fake_openai(monkeypatch)
    if url:
        monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", url)
    MJ.CascadeGeminiJudge()._next_client()
    assert built == [expected]


# --------------------------------------------------------------------------- #
# tau: GEMINI_OPENAI_BASE_URL enters the image only as the check accepts it.
# --------------------------------------------------------------------------- #
def test_tau_forwards_only_the_accepted_endpoint():
    """GEMINI_OPENAI_BASE_URL never enters the tau2 container, accepted or not: gbench's Gemini
    relay on the host forwards to it, and the container holds only the relay's placeholder."""
    host = {"GEMINI_API_KEY": GEMINI_KEY, "OPENAI_API_KEY": MODEL_KEY}
    for url in ["http://127.0.0.1:5555/v1beta/openai/"] + list(REFUSED_URLS):
        host["GEMINI_OPENAI_BASE_URL"] = url
        assert "GEMINI_OPENAI_BASE_URL" not in TC._launch_env(host)
        assert "GEMINI_OPENAI_BASE_URL" not in TC._forwarded_env(host)
        assert GEMINI_KEY not in TC._launch_env(host).values()


def test_tau_run_refuses_another_endpoint_before_the_image(monkeypatch):
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setattr(TC, "check_role_providers", lambda: (True, ""))
    monkeypatch.setattr(TC, "check_tau_image", lambda: pytest.fail("reached the image check"))
    with pytest.raises(RuntimeError) as ei:
        TC.run_tau_suite("tau2", ["retail"], "m", "http://127.0.0.1:9/v1", 1, None,
                         "docs/evals/tau2.md")
    assert "GEMINI_OPENAI_BASE_URL" in str(ei.value) and "api.example.com" not in str(ei.value)


@pytest.fixture(autouse=True)
def _metadata_guard_answers_blocked(monkeypatch):
    """Docker is stubbed in these tests: the cloud metadata guard the suites call before starting a
    container (container_guard.require_metadata_blocked) answers "unreachable" without a probe.
    tests/test_guard_tool_suites.py checks the suites' calls to it."""
    from gbench.runners.eval_suites import container_guard
    monkeypatch.setattr(container_guard, "require_metadata_blocked",
                        lambda *a, **k: container_guard.UNREACHABLE)
