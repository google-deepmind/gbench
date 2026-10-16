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

"""wildclawbench: web_search is the agent's only search tool, and no key reaches a task container.

* The task image's other search skills (local-web-search and web-search on DuckDuckGo,
  google-search on Google Custom Search) are deleted from every task container, and recorded.
* The agent calls the model under test through the proxy's model relay: the task containers get a
  placeholder apiKey, the proxy adds the model endpoint's key.
* The proxy listens on the task containers' address (never 0.0.0.0), on a default port outside the
  OpenClaw gateway's own ports.

Deterministic: no Docker, no model, no network beyond local stub servers on OS-assigned ports.
"""

import http.client
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from gbench.runners.eval_suites import wildclawbench as W

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOCKER_DIR = os.path.join(_REPO, "gbench", "docker")
sys.path.insert(0, _DOCKER_DIR)
import wildclawbench_run as RUN            # noqa: E402
import wildclawbench_cascade_judge as J    # noqa: E402

FAKE_KEY = "AIzaFAKE-wildclaw-single-path-0001"
MODEL_KEY = "sk-model-endpoint-secret-single-path"
SEARCH_SKILLS = ("local-web-search", "web-search", "google-search")


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.delenv("GEMINI_OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("WILDCLAW_WORKDIR", str(tmp_path / "proxy_workdir"))
    os.makedirs(tmp_path / "proxy_workdir", exist_ok=True)
    J.reset_usage()
    J._SEARCH_TOOL["module"] = None
    yield
    J._SEARCH_TOOL["module"] = None


# --------------------------------------------------------------------------- #
# The search skills: deleted in every task container (the setup script), recorded.
# --------------------------------------------------------------------------- #
def _skill(root, dirname, name, body="Use it.\n"):
    d = root / dirname
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\ndescription: x\n---\n\n{body}")
    (d / "scripts").mkdir()
    (d / "scripts" / "run.py").write_text("print('x')\n")
    return d


def _image(tmp_path, with_skills=True):
    """A fake task image: an OpenClaw package (bundle + bundled skills), a workspace with skills,
    and the config, laid out like wildclawbench-ubuntu:v1.3."""
    pkg = tmp_path / "pkg" / "openclaw"
    (pkg / "dist").mkdir(parents=True)
    (pkg / "dist" / "reply.js").write_text(
        'const BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search";\n')
    (pkg / "openclaw.mjs").write_text("// entry\n")
    os.chmod(pkg / "openclaw.mjs", 0o755)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    os.symlink(pkg / "openclaw.mjs", bindir / "openclaw")
    home = tmp_path / "root"
    ws_skills = home / "skills"
    ws_skills.mkdir(parents=True)
    (home / ".openclaw").mkdir()
    kept = {}
    if with_skills:
        _skill(ws_skills, "local-web-search-skill", "local-web-search")
        _skill(ws_skills, "web-search", "web-search")
        _skill(pkg / "skills", "google-search", "google-search")
        # a copy in the managed skills dir (~/.openclaw/skills) goes too
        _skill(home / ".openclaw" / "skills", "ddg", "web-search")
    kept["scrapling"] = _skill(ws_skills, "scrapling-web-scraper", "Scrapling Web Scraping")
    kept["weather"] = _skill(pkg / "skills", "weather", "weather")
    # "name: web-search" only in the body, not the frontmatter: not a search skill, kept
    kept["notes"] = _skill(ws_skills, "notes", "notes", body="Example:\nname: web-search\n")
    cfg = home / ".openclaw" / "openclaw.json"
    cfg.write_text(json.dumps({
        "agents": {"defaults": {"workspace": str(home)}},
        "tools": {"profile": "coding", "web": {"search": {
            "enabled": True, "provider": "brave", "apiKey": "${BRAVE_API_KEY}"}}},
        "gateway": {"port": 18789}}))
    return SimpleNamespace(pkg=pkg, bindir=bindir, home=home, cfg=cfg, kept=kept)


def _run_script(img, skills=",".join(SEARCH_SKILLS)):
    env = {**os.environ, "PATH": f"{img.bindir}{os.pathsep}{os.environ.get('PATH', '')}"}
    return subprocess.run([sys.executable, "-c", RUN._SEARCH_SETUP_SCRIPT,
                           "http://127.0.0.1:18795", RUN.BRAVE_API_ORIGIN,
                           RUN.SEARCH_KEY_PLACEHOLDER, str(img.cfg),
                           str(RUN.AGENT_WEB_SEARCH_TIMEOUT_S), skills],
                          capture_output=True, text=True, env=env)


def test_removed_skill_names_match_between_runner_and_launcher():
    assert tuple(s["skill"] for s in W.REMOVED_SEARCH_SKILLS) == RUN.SEARCH_SKILLS_REMOVED
    assert RUN.SEARCH_SKILLS_REMOVED == SEARCH_SKILLS


def test_setup_script_deletes_every_search_skill_and_keeps_the_rest(tmp_path):
    img = _image(tmp_path)
    r = _run_script(img)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["patched_files"] == 1 and out["provider"] == "brave"
    removed = sorted((s["name"], os.path.relpath(s["dir"], tmp_path))
                     for s in out["removed_search_skills"])
    assert removed == sorted([
        ("google-search", os.path.join("pkg", "openclaw", "skills", "google-search")),
        ("local-web-search", os.path.join("root", "skills", "local-web-search-skill")),
        ("web-search", os.path.join("root", ".openclaw", "skills", "ddg")),
        ("web-search", os.path.join("root", "skills", "web-search")),
    ])
    for _name, rel in removed:
        assert not (tmp_path / rel).exists()
    for d in img.kept.values():
        assert (d / "SKILL.md").is_file()
    # the web_search setup itself is unchanged
    assert json.loads(img.cfg.read_text())["tools"]["web"]["search"]["apiKey"] == \
        "gbench-gemini-grounding"


def test_setup_script_without_search_skills_reports_none(tmp_path):
    img = _image(tmp_path, with_skills=False)
    r = _run_script(img)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["removed_search_skills"] == []


def _exec_script(img, monkeypatch):
    """Run the setup script in this process (so shutil can be patched); returns the exit message,
    or None when it finished."""
    monkeypatch.setenv("PATH", f"{img.bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setattr(sys, "argv", ["-c", "http://127.0.0.1:18795", RUN.BRAVE_API_ORIGIN,
                                      RUN.SEARCH_KEY_PLACEHOLDER, str(img.cfg),
                                      str(RUN.AGENT_WEB_SEARCH_TIMEOUT_S),
                                      ",".join(SEARCH_SKILLS)])
    try:
        exec(compile(RUN._SEARCH_SETUP_SCRIPT, "<setup>", "exec"), {"__name__": "__main__"})
    except SystemExit as e:
        return str(e.code)
    return None


def test_setup_script_fails_when_a_search_skill_cannot_be_deleted(tmp_path, monkeypatch, capsys):
    img = _image(tmp_path)

    def refuse(path, *a, **k):
        raise PermissionError(13, "Permission denied", str(path))
    monkeypatch.setattr(shutil, "rmtree", refuse)
    msg = _exec_script(img, monkeypatch)
    assert msg and "cannot delete the" in msg and "skill at" in msg


def test_setup_script_fails_when_a_search_skill_is_left(tmp_path, monkeypatch, capsys):
    img = _image(tmp_path)
    monkeypatch.setattr(shutil, "rmtree", lambda *a, **k: None)     # "deletes" nothing
    msg = _exec_script(img, monkeypatch)
    assert msg and "search skills left" in msg and "local-web-search" in msg


def test_setup_command_and_preflight_pass_the_skill_list(monkeypatch):
    monkeypatch.setenv("WILDCLAW_SEARCH_ORIGIN", "http://127.0.0.1:18795")
    monkeypatch.setenv("WILDCLAW_TASK_IMAGE", "wildclawbench-ubuntu:v1.3")
    cmd = RUN.search_setup_command("task_c")
    assert cmd[:3] == ["docker", "exec", "task_c"]
    assert cmd[-1] == "local-web-search,web-search,google-search"
    seen = []

    def run(c, **k):
        seen.append(c)
        return subprocess.CompletedProcess(c, 0, stdout="{}", stderr="")
    assert RUN.preflight_search_setup(run=run)[0]
    assert seen[0][-1] == "local-web-search,web-search,google-search"


def test_launcher_summary_records_the_removed_skills(monkeypatch, tmp_path):
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("WILDCLAW_SEARCH_ORIGIN", "http://127.0.0.1:18795")
    monkeypatch.setenv("WILDCLAW_MODEL_RELAY_URL", "http://127.0.0.1:18795/model/v1")
    setup = {"patched_files": 10, "provider": "brave", "enabled": True,
             "removed_search_skills": [
                 {"name": "local-web-search", "dir": "/root/skills/local-web-search-skill"},
                 {"name": "web-search", "dir": "/root/skills/web-search"},
                 {"name": "google-search",
                  "dir": "/usr/lib/node_modules/openclaw/skills/google-search"}]}
    monkeypatch.setattr(RUN, "preflight_search_setup", lambda: (True, json.dumps(setup)))
    # the network monitor of the task network
    # (tests/test_wildclawbench_grading_isolation_and_network.py covers it): stubbed
    first = {"t": 1.0, "dns": {"github.com": "ok"}, "tcp": {"github.com": "ok"}}
    monkeypatch.setattr(RUN, "start_network_monitor", lambda: ("netmon", None))
    monkeypatch.setattr(RUN, "wait_first_round", lambda name: (first, None))
    monkeypatch.setattr(RUN, "stop_network_monitor", lambda name: None)
    monkeypatch.setattr(RUN, "_run_batch", lambda cfg, model: str(tmp_path / "output"))
    monkeypatch.setattr(RUN, "_task_index", lambda: {})
    monkeypatch.setattr(RUN, "_summarize", lambda out, idx: {
        "n_tasks": 1, "n_scored": 1, "global_mean": 1.0, "judge_fallback_tasks": 0})
    assert RUN.main() == 0
    summary = json.loads((tmp_path / "wildclawbench_summary.json").read_text())
    assert summary["removed_search_skills"] == ["local-web-search", "web-search", "google-search"]
    assert summary["web_search_setup"] == setup
    my_api = json.loads((tmp_path / "my_api.json").read_text())
    assert my_api["providers"]["gbench-model"]["baseUrl"] == "http://127.0.0.1:18795/model/v1"


def test_launcher_refuses_to_run_without_the_model_relay(monkeypatch, tmp_path):
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path))
    monkeypatch.delenv("WILDCLAW_MODEL_RELAY_URL", raising=False)

    def never():
        raise AssertionError("the preflight must not run")
    monkeypatch.setattr(RUN, "preflight_search_setup", never)
    assert RUN.main() == 5


def test_models_config_never_holds_the_model_key(monkeypatch, tmp_path):
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_MODEL_API_KEY", MODEL_KEY)      # even if it leaked into the launcher
    monkeypatch.setenv("WILDCLAW_MODEL_RELAY_URL", "http://172.17.0.1:18795/model/v1")
    path, _ = RUN._write_models_config()
    text = open(path, encoding="utf-8").read()
    assert MODEL_KEY not in text
    prov = json.loads(text)["providers"]["gbench-model"]
    assert prov["apiKey"] == RUN.MODEL_KEY_PLACEHOLDER == "gbench-model-relay"


# --------------------------------------------------------------------------- #
# The runner: records, the relay URL, the bind address, the port.
# --------------------------------------------------------------------------- #
def _run_runner(monkeypatch, tmp_path, env=None, summary_extra=None):
    harness = tmp_path / "harness"
    harness.mkdir(exist_ok=True)
    monkeypatch.setattr(W, "check_wildclawbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(W, "_harness_dir", lambda: str(harness))
    asked = []

    def free_port(preferred=None):
        asked.append(preferred)
        return preferred
    monkeypatch.setattr(W, "free_port", free_port)
    monkeypatch.setattr(W.tempfile, "mkdtemp", lambda prefix="": str(tmp_path / "workdir"))
    os.makedirs(tmp_path / "workdir", exist_ok=True)
    for k in ("GBENCH_WILDCLAWBENCH_TASK_NETWORK", "GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT",
              "GBENCH_WILDCLAWBENCH_TASK_ENDPOINT", "GBENCH_WILDCLAWBENCH_JUDGE_PORT",
              "GBENCH_WILDCLAWBENCH_TASK_HOST", "GBENCH_SHARD",
              "GBENCH_WILDCLAWBENCH_MODEL_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    captured = {"free_port": asked}

    def fake_run(cmd, *a, **k):
        if cmd[:3] == ["docker", "run", "--rm"]:
            captured["cmd"], captured["env"] = list(cmd), dict(k.get("env") or {})
            workdir = next(x.split("=", 1)[1] for x in cmd
                           if str(x).startswith("WILDCLAW_WORKDIR="))
            summary = {"n_tasks": 60, "n_scored": 60, "n_missing": 0, "global_mean": 0.25,
                       "judge_fallback_tasks": 0, "n_search_setup_failures": 0,
                       "removed_search_skills": list(SEARCH_SKILLS)}
            summary.update(summary_extra or {})
            with open(os.path.join(workdir, "wildclawbench_summary.json"), "w") as f:
                json.dump(summary, f)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(W.subprocess, "run", fake_run)
    result = W.run_wildclawbench("google/gemma-4-E4B-it", "http://127.0.0.1:9123/v1")
    return result, captured


def test_runner_records_the_removed_search_skills(monkeypatch, tmp_path):
    result, _cap = _run_runner(monkeypatch, tmp_path)
    assert result["removed_search_skills"] == list(SEARCH_SKILLS)
    sub = next(s for s in result["gemini_substitutions"] if "web_search tool" in s["component"])
    assert [s["skill"] for s in sub["removed_search_skills"]] == list(SEARCH_SKILLS)
    services = " ".join(s["service"] for s in sub["removed_search_skills"])
    assert "DuckDuckGo" in services and "Google Custom Search" in services
    reason = result["leaderboard_comparable_reason"]
    assert "local-web-search and web-search on DuckDuckGo" in reason
    assert "google-search on Google Custom Search" in reason
    assert "web_search is the agent's only search tool" in reason
    assert result["leaderboard_comparable"] is False


def test_runner_host_network_relay_bind_and_port(monkeypatch, tmp_path):
    _result, cap = _run_runner(monkeypatch, tmp_path, env={
        "GBENCH_WILDCLAWBENCH_TASK_NETWORK": "host",
        "GBENCH_WILDCLAWBENCH_MODEL_API_KEY": MODEL_KEY})
    cmd = cap["cmd"]
    assert cap["free_port"] == [18795]
    assert "WILDCLAW_JUDGE_PORT=18795" in cmd
    assert "WILDCLAW_JUDGE_HOST=127.0.0.1" in cmd
    assert "WILDCLAW_MODEL_RELAY_URL=http://127.0.0.1:18795/model/v1" in cmd
    # the proxy reaches the model through gbench's temperature relay on 127.0.0.1
    url = next(x.split("=", 1)[1] for x in cmd if x.startswith("GBENCH_MODEL_BASE_URL="))
    assert url.startswith("http://127.0.0.1:") and url != "http://127.0.0.1:9123/v1"
    assert _result["model_temperature_relay"]["relays_to"] == "http://127.0.0.1:9123/v1"
    assert not any("0.0.0.0" in str(x) for x in cmd)
    # the model key travels by name, to the orchestrator (for its proxy) only
    assert MODEL_KEY not in "\n".join(map(str, cmd))
    assert cmd[cmd.index("GBENCH_MODEL_API_KEY") - 1] == "-e"
    assert cap["env"]["GBENCH_MODEL_API_KEY"] == MODEL_KEY


def test_runner_bridge_network_relay_and_bind(monkeypatch, tmp_path):
    _result, cap = _run_runner(monkeypatch, tmp_path, env={
        "GBENCH_WILDCLAWBENCH_TASK_NETWORK": "bridge"})
    cmd = cap["cmd"]
    assert "WILDCLAW_JUDGE_HOST=172.17.0.1" in cmd
    assert "WILDCLAW_MODEL_RELAY_URL=http://172.17.0.1:18795/model/v1" in cmd
    # the proxy runs in the orchestrator on the host network: the model URL is not rewritten
    # (it reaches the model through gbench's temperature relay on 127.0.0.1)
    url = next(x.split("=", 1)[1] for x in cmd if x.startswith("GBENCH_MODEL_BASE_URL="))
    assert url.startswith("http://127.0.0.1:")
    assert _result["model_temperature_relay"]["relays_to"] == "http://127.0.0.1:9123/v1"


@pytest.mark.parametrize("port", ["18789", "18791", "18800", "18899"])
def test_runner_refuses_openclaw_ports_with_the_host_network(monkeypatch, tmp_path, port):
    with pytest.raises(Exception) as ei:
        _run_runner(monkeypatch, tmp_path, env={"GBENCH_WILDCLAWBENCH_TASK_NETWORK": "host",
                                                "GBENCH_WILDCLAWBENCH_JUDGE_PORT": port})
    assert f"GBENCH_WILDCLAWBENCH_JUDGE_PORT={port}" in str(ei.value)


def test_runner_allows_an_openclaw_port_number_with_bridge(monkeypatch, tmp_path):
    _result, cap = _run_runner(monkeypatch, tmp_path, env={
        "GBENCH_WILDCLAWBENCH_TASK_NETWORK": "bridge", "GBENCH_WILDCLAWBENCH_JUDGE_PORT": "18791"})
    assert "WILDCLAW_JUDGE_PORT=18791" in cap["cmd"]


def test_runner_ignores_the_removed_judge_endpoint_knob(monkeypatch, tmp_path, caplog):
    # GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT no longer moves the judge: it only logs a warning.
    with caplog.at_level("WARNING"):
        _result, cap = _run_runner(monkeypatch, tmp_path, env={
            "GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT": "http://judge.example:9/v1"})
    assert "WILDCLAW_JUDGE_ENDPOINT=http://172.17.0.1:18795/v1" in cap["cmd"]
    assert not any("judge.example" in str(x) for x in cap["cmd"])
    assert "GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT no longer changes anything" in caplog.text


def test_default_port_is_the_same_everywhere_and_outside_openclaw_ports():
    assert W._JUDGE_PORT_DEFAULT == 18795
    assert W._JUDGE_PORT_DEFAULT not in W._OPENCLAW_PORTS
    assert {18789, 18791, 18800, 18899} <= W._OPENCLAW_PORTS
    entry = open(os.path.join(_DOCKER_DIR, "wildclawbench_entrypoint.sh"), encoding="utf-8").read()
    assert 'JUDGE_PORT="${WILDCLAW_JUDGE_PORT:-18795}"' in entry
    assert "18791" not in entry


def test_gate_refuses_a_non_gemini_base_url(tmp_path, monkeypatch):
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "run_batch.py").write_text("x")
    (tmp_path / "workspace").mkdir()
    monkeypatch.setattr(W.shutil, "which", lambda _x: "/usr/bin/docker")
    ok_proc = type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})
    monkeypatch.setattr(W.subprocess, "run", lambda *a, **k: ok_proc())
    monkeypatch.setattr(W, "_docker_image_exists", lambda _i: True)
    monkeypatch.setattr(W, "_image_accounts_harness_errors", lambda _i: True)
    monkeypatch.setenv("GBENCH_WILDCLAWBENCH_HOST_DIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION", "1")
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", "https://relay.example.com/v1beta/openai/")
    ok, reason = W.check_wildclawbench_prerequisites()
    assert not ok and "GEMINI_OPENAI_BASE_URL is not the Gemini API" in reason
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", "http://127.0.0.1:18799/v1beta/openai/")
    assert W.check_wildclawbench_prerequisites() == (True, "")


# --------------------------------------------------------------------------- #
# The proxy: model relay, bind address, redaction.
# --------------------------------------------------------------------------- #
class _Upstream(BaseHTTPRequestHandler):
    """A stub model endpoint: records each request; streams two SSE events for stream=true, the
    second only after the test has received the first through the relay."""
    seen = []
    proceed = threading.Event()
    first_seen_in_time = []

    def log_message(self, *a):
        pass

    def _record(self, body):
        _Upstream.seen.append({"method": self.command, "path": self.path,
                               "auth": self.headers.get("Authorization"),
                               "x_client": self.headers.get("X-Client"), "body": body})

    def do_GET(self):
        self._record(b"")
        payload = json.dumps({"object": "list", "data": [{"id": "m"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._record(body)
        req = json.loads(body or b"{}")
        if req.get("fail"):
            payload = b'{"error": "unauthorized"}'
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"delta": "one"}\n\n')
            self.wfile.flush()
            _Upstream.first_seen_in_time.append(_Upstream.proceed.wait(timeout=10))
            self.wfile.write(b'data: {"delta": "two"}\n\ndata: [DONE]\n\n')
            self.wfile.flush()
            return
        payload = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture
def relay(monkeypatch):
    _Upstream.seen.clear()
    _Upstream.first_seen_in_time.clear()
    _Upstream.proceed.clear()
    upstream, up_url = _serve(_Upstream)
    proxy, proxy_url = _serve(J._Handler)
    monkeypatch.setenv("GBENCH_MODEL_BASE_URL", up_url + "/v1")
    monkeypatch.setenv("GBENCH_MODEL_API_KEY", MODEL_KEY)
    # an http proxy in the environment must not be used for the model endpoint
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    try:
        yield proxy_url
    finally:
        _Upstream.proceed.set()
        for s in (proxy, upstream):
            s.shutdown()
            s.server_close()


def _no_proxy_open(req, timeout=15):
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout)


def _post(url, payload, headers=None):
    return urllib.request.Request(url, method="POST", data=json.dumps(payload).encode(),
                                  headers={"Content-Type": "application/json", **(headers or {})})


def test_model_relay_adds_the_key_and_passes_the_reply(relay):
    req = _post(relay + "/model/v1/chat/completions", {"model": "m", "messages": []},
                {"Authorization": "Bearer gbench-model-relay", "X-Client": "openclaw"})
    with _no_proxy_open(req) as r:
        assert r.status == 200
        assert json.loads(r.read()) == {"choices": [{"message": {"content": "hi"}}]}
    seen = _Upstream.seen[-1]
    assert seen["path"] == "/v1/chat/completions" and seen["method"] == "POST"
    assert seen["auth"] == f"Bearer {MODEL_KEY}"                  # placeholder replaced
    assert seen["x_client"] == "openclaw"                         # other headers kept
    assert json.loads(seen["body"]) == {"model": "m", "messages": []}


def test_model_relay_streams_without_buffering(relay):
    req = _post(relay + "/model/v1/chat/completions", {"stream": True})
    with _no_proxy_open(req) as r:
        assert r.headers.get("Content-Type") == "text/event-stream"
        first = r.readline() + r.readline()
        assert first == b'data: {"delta": "one"}\n\n'
        _Upstream.proceed.set()          # only now does the stub send the rest
        rest = r.read()
    assert rest == b'data: {"delta": "two"}\n\ndata: [DONE]\n\n'
    assert _Upstream.first_seen_in_time == [True]


def test_model_relay_get_keeps_the_query_and_passes_error_status(relay):
    with _no_proxy_open(urllib.request.Request(relay + "/model/v1/models?limit=1")) as r:
        assert json.loads(r.read())["data"] == [{"id": "m"}]
    assert _Upstream.seen[-1]["path"] == "/v1/models?limit=1"
    with pytest.raises(urllib.error.HTTPError) as ei:
        _no_proxy_open(_post(relay + "/model/v1/chat/completions", {"fail": True}))
    assert ei.value.code == 401 and json.loads(ei.value.read()) == {"error": "unauthorized"}


def test_model_relay_decodes_a_chunked_request_body(relay):
    host, port = relay.split("//", 1)[1].split(":")
    conn = http.client.HTTPConnection(host, int(port), timeout=15)
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": "x"}]}).encode()
    conn.putrequest("POST", "/model/v1/chat/completions")
    conn.putheader("Content-Type", "application/json")
    conn.putheader("Transfer-Encoding", "chunked")
    conn.endheaders()
    for part in (body[:10], body[10:]):
        conn.send(b"%x\r\n%s\r\n" % (len(part), part))
    conn.send(b"0\r\n\r\n")
    resp = conn.getresponse()
    assert resp.status == 200
    resp.read()
    conn.close()
    assert json.loads(_Upstream.seen[-1]["body"])["messages"][0]["content"] == "x"


def test_model_relay_unreachable_endpoint_is_502(relay, monkeypatch):
    monkeypatch.setenv("GBENCH_MODEL_BASE_URL", "http://127.0.0.1:9/v1")
    with pytest.raises(urllib.error.HTTPError) as ei:
        _no_proxy_open(_post(relay + "/model/v1/chat/completions", {}))
    assert ei.value.code == 502
    err = json.loads(ei.value.read())["error"]
    assert err["type"] == "model_relay_error" and MODEL_KEY not in err["message"]


def test_judge_path_is_not_relayed(relay, monkeypatch):
    monkeypatch.setattr(J, "cascade_chat_completion",
                        lambda body, **kw: {"choices": [{"message": {"content": "judge"}}]})
    with _no_proxy_open(_post(relay + "/v1/chat/completions", {"messages": []})) as r:
        assert json.loads(r.read())["choices"][0]["message"]["content"] == "judge"
    assert _Upstream.seen == []


def test_model_relay_target_mapping(monkeypatch):
    monkeypatch.setenv("GBENCH_MODEL_BASE_URL", "http://127.0.0.1:9123/v1/")
    assert J.model_relay_target("/model/v1/chat/completions") == \
        "http://127.0.0.1:9123/v1/chat/completions"
    assert J.model_relay_target("/model/v1?x=1") == "http://127.0.0.1:9123/v1?x=1"
    assert J.model_relay_target("/v1/chat/completions") is None
    assert J.model_relay_target("/model/v10/chat") is None
    assert J.model_relay_target("/res/v1/web/search?q=x") is None


def test_redact_covers_the_model_key(monkeypatch):
    monkeypatch.setenv("GBENCH_MODEL_API_KEY", MODEL_KEY)
    out = J.redact(f"failed with {MODEL_KEY} and {FAKE_KEY}")
    assert MODEL_KEY not in out and FAKE_KEY not in out


_NO_SEARCH = SimpleNamespace(search_with_fallback=lambda q, max_results=5: ([], ""))


def _fake_server(bound):
    class _Server:
        def __init__(self, addr, handler):
            bound.append(addr)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass
    return _Server


def test_proxy_listens_on_loopback_by_default(monkeypatch):
    for k in ("WILDCLAW_JUDGE_HOST", "WILDCLAW_JUDGE_PORT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GBENCH_MODEL_BASE_URL", "http://127.0.0.1:9123/v1")
    J._SEARCH_TOOL["module"] = _NO_SEARCH
    bound = []
    monkeypatch.setattr(J, "ThreadingHTTPServer", _fake_server(bound))
    assert J.main() == 0
    assert bound == [("127.0.0.1", 18795)]
    monkeypatch.setenv("WILDCLAW_JUDGE_HOST", "172.17.0.1")
    monkeypatch.setenv("WILDCLAW_JUDGE_PORT", "18796")
    assert J.main() == 0
    assert bound[-1] == ("172.17.0.1", 18796)


def test_proxy_does_not_start_without_the_model_url(monkeypatch):
    monkeypatch.delenv("GBENCH_MODEL_BASE_URL", raising=False)
    J._SEARCH_TOOL["module"] = _NO_SEARCH
    monkeypatch.setattr(J, "ThreadingHTTPServer", _fake_server([]))
    assert J.main() == 4


def test_entrypoint_binds_the_given_host_and_keeps_the_model_key_from_the_launcher():
    text = open(os.path.join(_DOCKER_DIR, "wildclawbench_entrypoint.sh"), encoding="utf-8").read()
    assert 'JUDGE_HOST="${WILDCLAW_JUDGE_HOST:-127.0.0.1}"' in text
    assert 'WILDCLAW_JUDGE_HOST="$JUDGE_HOST"' in text
    assert "http://${JUDGE_HOST}:${JUDGE_PORT}/health" in text
    assert "0.0.0.0" not in text
    assert "env -u GEMINI_API_KEYS -u GEMINI_API_KEY -u GBENCH_MODEL_API_KEY" in text


# --------------------------------------------------------------------------- #
# Docs match the code.
# --------------------------------------------------------------------------- #
def test_docs_describe_the_single_search_path_and_the_relay():
    doc = open(os.path.join(_REPO, "docs", "evals", "wildclawbench.md"), encoding="utf-8").read()
    for skill in W.REMOVED_SEARCH_SKILLS:
        assert f"`{skill['skill']}`" in doc
    for needle in ("/root/skills/local-web-search-skill", "/root/skills/web-search",
                   "`removed_search_skills`", "`gbench-model-relay`", "`<proxy origin>/model/v1`",
                   "`18795` when free", "`127.0.0.1` with the host network",
                   "`172.17.0.1` with the default bridge network",
                   "`18789`-`18793` or `18800`-`18899`"):
        assert needle in doc, needle
    assert "Every external model or search call of a run goes through" not in doc
    assert "0.0.0.0" not in doc
    assert "18791` when free" not in doc


@pytest.fixture(autouse=True)
def _metadata_guard_answers_blocked(monkeypatch):
    """Docker is stubbed in these tests: the cloud metadata guard the suites call before starting a
    container (container_guard.require_metadata_blocked) answers "unreachable" without a probe.
    tests/test_guard_tool_suites.py checks the suites' calls to it."""
    from gbench.runners.eval_suites import container_guard
    monkeypatch.setattr(container_guard, "require_metadata_blocked",
                        lambda *a, **k: container_guard.UNREACHABLE)
