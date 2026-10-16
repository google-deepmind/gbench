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

"""wildclawbench on the Gemini API only: the OpenClaw web_search served by Gemini Google Search
grounding, the web-search judge of 04_Search_Retrieval_task_1 served by Gemini grounding, no Brave
key or path, and no Gemini key on a command line, in a task container, a reply or a log line.

Deterministic: no Docker, no model, no network (Gemini replies and the search backend are stubs).
"""

import io
import json
import os
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from gbench.runners.eval_suites import wildclawbench as W

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOCKER_DIR = os.path.join(_REPO, "gbench", "docker")
sys.path.insert(0, _DOCKER_DIR)
import wildclawbench_run as RUN            # noqa: E402
import wildclawbench_cascade_judge as J    # noqa: E402

FAKE_KEY = "AIzaFAKE-wildclaw-test-key-0001"
FAKE_KEY_2 = "AIzaFAKE-wildclaw-test-key-0002"


@pytest.fixture(autouse=True)
def _gemini_env(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.setenv("WILDCLAW_WORKDIR", str(tmp_path / "proxy_workdir"))
    os.makedirs(tmp_path / "proxy_workdir", exist_ok=True)
    for k in ("GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL", "GEMINI_OPENAI_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    J.reset_usage()
    J._SEARCH_TOOL["module"] = None
    yield
    J._SEARCH_TOOL["module"] = None


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stub_search_tool(results=None, backend="gemini:gemini-3.7-flash", raises=None, seen=None):
    def search_with_fallback(query, max_results=8, deadline=None):
        if seen is not None:
            seen.append((query, max_results))
        if raises is not None:
            raise raises
        return list(results or []), backend
    return SimpleNamespace(search_with_fallback=search_with_fallback)


# --------------------------------------------------------------------------- #
# Agent web_search: Brave response shape, answered by Gemini grounding.
# --------------------------------------------------------------------------- #
def test_brave_search_reply_maps_grounding_results():
    seen = []
    J._SEARCH_TOOL["module"] = _stub_search_tool([
        {"title": "Paper page", "url": "https://vertexaisearch.example/redirect/1",
         "snippet": "Ziyu Liu and Dahua Lin co-authored X."},
        {"title": "grounded-answer", "url": "", "snippet": "an answer without a source"},
    ], seen=seen)
    code, body = J.brave_search_reply("q=ziyu+liu+coauthors&count=3")
    assert code == 200
    assert seen == [("ziyu liu coauthors", 3)]              # count -> max_results
    assert body["query"]["original"] == "ziyu liu coauthors"
    assert body["web"]["results"] == [
        {"title": "Paper page", "url": "https://vertexaisearch.example/redirect/1",
         "description": "Ziyu Liu and Dahua Lin co-authored X."},
        {"title": "grounded-answer", "url": "", "description": "an answer without a source"},
    ]
    assert J.usage_snapshot()["agent_web_search"] == {"gemini:gemini-3.7-flash": 1}


def test_brave_search_reply_no_results_is_empty_list():
    J._SEARCH_TOOL["module"] = _stub_search_tool(
        [{"title": "no results", "url": "", "snippet": "The search returned nothing."}])
    code, body = J.brave_search_reply("q=nothing+here")
    assert code == 200 and body["web"]["results"] == []


def test_brave_search_reply_failure_is_502_never_empty_results():
    J._SEARCH_TOOL["module"] = _stub_search_tool(
        [{"title": "error", "url": "", "snippet": "all grounding models failed"}],
        backend="all-grounding-models-failed")
    code, body = J.brave_search_reply("q=x")
    assert code == 502
    assert "Gemini Google Search grounding could not serve" in body["error"]["detail"]
    assert "web" not in body
    assert J.usage_snapshot()["agent_web_search"] == {"all-grounding-models-failed": 1}


def test_brave_search_reply_redacts_key_in_backend_error():
    J._SEARCH_TOOL["module"] = _stub_search_tool(
        raises=RuntimeError(f"boom https://x/y?key={FAKE_KEY} {FAKE_KEY}"))
    code, body = J.brave_search_reply("q=x")
    assert code == 502
    assert FAKE_KEY not in json.dumps(body)
    assert "[REDACTED]" in body["error"]["detail"]


def test_brave_search_reply_counts_unapplied_filters_and_rejects_empty_query():
    J._SEARCH_TOOL["module"] = _stub_search_tool([{"title": "t", "url": "u", "snippet": "s"}])
    J.brave_search_reply("q=x&country=DE&search_lang=de&freshness=pw")
    assert J.usage_snapshot()["agent_web_search"]["filters_not_applied"] == 1
    code, _ = J.brave_search_reply("count=3")
    assert code == 422


def test_search_tool_loads_from_mount_dir(monkeypatch):
    monkeypatch.setenv("WILDCLAW_SEARCH_TOOL_DIR",
                       os.path.join(_REPO, "gbench", "runners", "eval_suites"))
    mod = J.search_tool()
    assert callable(mod.search_with_fallback) and callable(mod.get_gemini_api_key)
    monkeypatch.setenv("WILDCLAW_SEARCH_TOOL_DIR", "/nonexistent-dir")
    J._SEARCH_TOOL["module"] = None
    with pytest.raises(RuntimeError, match="search_tool.py not found"):
        J.search_tool()


def _serve():
    server = ThreadingHTTPServer(("127.0.0.1", 0), J._Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_proxy_http_routes_brave_path_and_rejects_others():
    J._SEARCH_TOOL["module"] = _stub_search_tool([{"title": "T", "url": "https://e.org", "snippet": "S"}])
    server, origin = _serve()
    try:
        with urllib.request.urlopen(origin + "/res/v1/web/search?q=hello&count=2", timeout=10) as r:
            body = json.loads(r.read())
        assert body["web"]["results"][0]["description"] == "S"
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(origin + "/res/v1/llm/context?q=hello", timeout=10)
        assert ei.value.code == 404
        with urllib.request.urlopen(origin + "/health", timeout=10) as r:
            assert json.loads(r.read()) == {"status": "ok"}
        # A web-search judge request mixed with function tools -> 400 (not silently dropped).
        req = urllib.request.Request(
            origin + "/v1/chat/completions", method="POST",
            data=json.dumps({"tools": [{"type": "openrouter:web_search"},
                                       {"type": "function", "function": {"name": "f"}}],
                             "messages": [{"role": "user", "content": "x"}]}).encode(),
            headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(req, timeout=10)
        assert ei.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


# --------------------------------------------------------------------------- #
# Judge: web search (04_Search_Retrieval_task_1) via Gemini grounding; keys; redaction.
# --------------------------------------------------------------------------- #
def _grounded_reply(text, uris=()):
    return {"candidates": [{"content": {"parts": [{"text": text}]},
                            "groundingMetadata": {"groundingChunks": [
                                {"web": {"uri": u, "title": f"title {i}"}}
                                for i, u in enumerate(uris)]}}],
            "usageMetadata": {"promptTokenCount": 11, "candidatesTokenCount": 7,
                              "totalTokenCount": 18}}


def test_web_search_judge_goes_to_google_search_grounding(monkeypatch):
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-a,gemini-b")
    calls = []

    def fake_urlopen(req, timeout=0):
        calls.append({"url": req.full_url, "headers": dict(req.header_items()),
                      "body": json.loads(req.data.decode("utf-8"))})
        if "gemini-a" in req.full_url:                  # empty content -> next cascade model
            return _Resp(_grounded_reply(""))
        return _Resp(_grounded_reply('{"score": 1}', uris=["https://src.example/p"]))

    monkeypatch.setattr(J.urllib.request, "urlopen", fake_urlopen)
    # The exact request shape of 04_Search_Retrieval_task_1's verification judge.
    out = J.cascade_chat_completion({
        "model": "openai/gpt-5.4", "temperature": 0,
        "messages": [{"role": "system", "content": "You are a strict coauthorship verifier."},
                     {"role": "user", "content": "Verify these paths."}],
        "tools": [{"type": "openrouter:web_search",
                   "parameters": {"engine": "auto", "max_results": 5, "max_total_results": 30}}],
    })
    assert [c["url"].rsplit("/", 1)[-1] for c in calls] == [
        "gemini-a:generateContent", "gemini-b:generateContent"]
    first = calls[0]
    assert first["url"].startswith("https://generativelanguage.googleapis.com/v1beta/models/")
    assert FAKE_KEY not in first["url"]                                  # key in a header only
    assert first["headers"].get("X-goog-api-key") == FAKE_KEY
    assert first["body"]["tools"] == [{"google_search": {}}]
    assert first["body"]["systemInstruction"] == {
        "parts": [{"text": "You are a strict coauthorship verifier."}]}
    assert first["body"]["contents"] == [{"role": "user", "parts": [{"text": "Verify these paths."}]}]
    assert first["body"]["generationConfig"]["temperature"] == 0.0
    msg = out["choices"][0]["message"]
    assert msg["content"] == '{"score": 1}'
    assert msg["annotations"] == [{"type": "url_citation",
                                   "url_citation": {"url": "https://src.example/p",
                                                    "title": "title 0"}}]
    assert out["model"] == "gemini-b" and out["usage"]["total_tokens"] == 18
    assert J.usage_snapshot()["judge_web_search"] == {"gemini-b": 1}
    written = json.load(open(os.path.join(os.environ["WILDCLAW_WORKDIR"], "gemini_usage.json")))
    assert written["judge_web_search"] == {"gemini-b": 1}


def test_grounded_request_maps_max_tokens_and_rejects_images():
    req = J.gemini_grounded_request({"max_tokens": 256, "messages": [
        {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        {"role": "assistant", "content": "c"}]})
    # the request's own sampling only: no temperature sent here, so none is set; the task's cap
    # plus the Gemini thinking allowance
    assert req["generationConfig"] == {"maxOutputTokens": 256 + J.GEMINI_THINKING_ALLOWANCE}
    req0 = J.gemini_grounded_request({"temperature": 0, "messages": [
        {"role": "user", "content": "a"}]})
    assert req0["generationConfig"] == {"temperature": 0}
    assert req["contents"] == [{"role": "user", "parts": [{"text": "a"}, {"text": "b"}]},
                               {"role": "model", "parts": [{"text": "c"}]}]
    with pytest.raises(J.UnsupportedRequest):
        J.gemini_grounded_request({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]})


def test_unknown_server_tool_type_is_rejected():
    with pytest.raises(J.UnsupportedRequest, match="unsupported judge tool type"):
        J.cascade_chat_completion({"tools": [{"type": "openrouter:image_generation"}],
                                   "messages": [{"role": "user", "content": "x"}]})


def test_gemini_api_keys_wins_and_rotates(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", f"{FAKE_KEY};{FAKE_KEY_2}")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaFAKE-single-key-ignored")
    assert J.gemini_keys() == [FAKE_KEY, FAKE_KEY_2]
    got = {J.next_gemini_key() for _ in range(4)}
    assert got == {FAKE_KEY, FAKE_KEY_2}


def test_judge_error_text_is_redacted(monkeypatch):
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "m1")
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", "1")

    def fail(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 400, "bad", {},
                                     io.BytesIO(f'{{"error": "API key {FAKE_KEY} invalid"}}'.encode()))
    monkeypatch.setattr(J.urllib.request, "urlopen", fail)
    with pytest.raises(RuntimeError) as ei:
        J.cascade_chat_completion({"messages": [{"role": "user", "content": "x"}]})
    assert FAKE_KEY not in str(ei.value) and "[REDACTED]" in str(ei.value)
    assert J.usage_snapshot()["judge"] == {"failed": 1}
    assert J.redact(f"u?key={FAKE_KEY}&alt=json") == "u?key=[REDACTED]&alt=json"


# --------------------------------------------------------------------------- #
# Launcher: task containers get the Gemini web search, never a key.
# --------------------------------------------------------------------------- #
def test_strip_brave_env():
    cmd = ["docker", "run", "-d", "--name", "t", "-e", "http_proxy=", "-e", "BRAVE_API_KEY=abc",
           "-e", "OPENROUTER_BASE_URL=http://127.0.0.1:1/v1", "img"]
    assert RUN._strip_brave_env(cmd) == ["docker", "run", "-d", "--name", "t", "-e", "http_proxy=",
                                         "-e", "OPENROUTER_BASE_URL=http://127.0.0.1:1/v1", "img"]


def _patched(monkeypatch, tmp_path, setup_rc=0):
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("WILDCLAW_SEARCH_ORIGIN", "http://127.0.0.1:18791")
    monkeypatch.setenv("WILDCLAW_TASK_NETWORK", "host")
    monkeypatch.setenv("WILDCLAW_TASK_LABEL", "gbench_wildclaw_run=x")
    calls = []

    def orig_run(cmd, *a, **k):
        calls.append(list(cmd))
        if cmd[:2] == ["docker", "exec"]:
            return subprocess.CompletedProcess(cmd, setup_rc, stdout="",
                                               stderr="" if setup_rc == 0 else "no bundle")
        return subprocess.CompletedProcess(cmd, 0, stdout="cid\n", stderr="")
    fake = SimpleNamespace(subprocess=SimpleNamespace(run=orig_run))
    RUN._patch_task_containers(fake)
    return fake, calls


def test_task_container_start_gets_gemini_search_setup(monkeypatch, tmp_path):
    fake, calls = _patched(monkeypatch, tmp_path)
    start = ["docker", "run", "-d", "--name", "04_task_1_m_x", "-e", "BRAVE_API_KEY=",
             "-e", "OPENROUTER_API_KEY=gbench-cascade", "-v", "/w:/app:ro",
             "wildclawbench-ubuntu:v1.3", "/bin/bash", "-c", "tail -f /dev/null"]
    r = fake.subprocess.run(start, capture_output=True, text=True)
    assert r.returncode == 0
    run_cmd, exec_cmd = calls
    assert run_cmd[:7] == ["docker", "run", "-d", "--network", "host",
                           "--label", "gbench_wildclaw_run=x"]
    assert not any(str(x).startswith("BRAVE_API_KEY") for x in run_cmd)
    assert exec_cmd[:5] == ["docker", "exec", "04_task_1_m_x", "python3", "-c"]
    assert exec_cmd[-6:] == ["http://127.0.0.1:18791", "https://api.search.brave.com",
                             "gbench-gemini-grounding", "/root/.openclaw/openclaw.json", "60",
                             "local-web-search,web-search,google-search"]
    # nothing a task container receives carries a Gemini key
    assert all(FAKE_KEY not in str(x) for c in calls for x in c)
    # other docker calls pass through untouched
    fake.subprocess.run(["docker", "exec", "c", "ls"])
    assert calls[-1] == ["docker", "exec", "c", "ls"]


def test_task_container_search_setup_failure_fails_the_start(monkeypatch, tmp_path):
    fake, calls = _patched(monkeypatch, tmp_path, setup_rc=1)
    r = fake.subprocess.run(["docker", "run", "-d", "--name", "t1", "img"],
                            capture_output=True, text=True)
    assert r.returncode == 1 and "Gemini proxy failed" in r.stderr
    assert RUN._count_search_setup_failures() == 1


def _fake_openclaw(tmp_path, provider="brave", brave_mode=None):
    pkg = tmp_path / "pkg" / "openclaw"
    (pkg / "dist" / "plugin-sdk").mkdir(parents=True)
    src = ('const BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search";\n'
           'const BRAVE_LLM_CONTEXT_ENDPOINT = "https://api.search.brave.com/res/v1/llm/context";\n')
    (pkg / "dist" / "pi-embedded.js").write_text(src)
    (pkg / "dist" / "plugin-sdk" / "reply.js").write_text(src)
    (pkg / "dist" / "other.js").write_text("const x = 1;\n")
    (pkg / "openclaw.mjs").write_text("// entry\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    os.symlink(pkg / "openclaw.mjs", bindir / "openclaw")
    os.chmod(pkg / "openclaw.mjs", 0o755)
    search = {"enabled": True, "provider": provider, "apiKey": "${BRAVE_API_KEY}"}
    if brave_mode:
        search["brave"] = {"mode": brave_mode}
    cfg = tmp_path / "openclaw.json"
    cfg.write_text(json.dumps({"tools": {"profile": "coding", "web": {"search": search}},
                               "gateway": {"port": 18789}}))
    return pkg, bindir, cfg


def _run_setup_script(bindir, cfg, origin="http://127.0.0.1:18791"):
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"}
    return subprocess.run([sys.executable, "-c", RUN._SEARCH_SETUP_SCRIPT, origin,
                           RUN.BRAVE_API_ORIGIN, RUN.SEARCH_KEY_PLACEHOLDER, str(cfg),
                           str(RUN.AGENT_WEB_SEARCH_TIMEOUT_S),
                           ",".join(RUN.SEARCH_SKILLS_REMOVED)],
                          capture_output=True, text=True, env=env)


def test_setup_script_rewrites_bundle_and_config(tmp_path):
    pkg, bindir, cfg = _fake_openclaw(tmp_path)
    r = _run_setup_script(bindir, cfg)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["patched_files"] == 2
    for f in ("dist/pi-embedded.js", "dist/plugin-sdk/reply.js"):
        text = (pkg / f).read_text()
        assert "api.search.brave.com" not in text
        assert '"http://127.0.0.1:18791/res/v1/web/search"' in text
    conf = json.loads(cfg.read_text())
    assert conf["tools"]["web"]["search"] == {"enabled": True, "provider": "brave",
                                              "apiKey": "gbench-gemini-grounding",
                                              "timeoutSeconds": 60}     # 2026-10-13: was 30 s
    assert conf["gateway"] == {"port": 18789}                      # everything else untouched
    # a second pass finds nothing to rewrite -> refuses (each task container starts fresh)
    assert _run_setup_script(bindir, cfg).returncode != 0


@pytest.mark.parametrize("provider,mode", [("gemini", None), ("brave", "llm-context")])
def test_setup_script_refuses_an_image_it_was_not_built_for(tmp_path, provider, mode):
    _pkg, bindir, cfg = _fake_openclaw(tmp_path, provider=provider, brave_mode=mode)
    r = _run_setup_script(bindir, cfg)
    assert r.returncode != 0 and "wildclawbench web-search setup" in r.stderr


def test_preflight(monkeypatch):
    monkeypatch.setenv("WILDCLAW_TASK_IMAGE", "wildclawbench-ubuntu:v1.3")
    seen = []

    def ok_run(cmd, **k):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout='{"patched_files": 10}', stderr="")
    monkeypatch.setenv("WILDCLAW_SEARCH_ORIGIN", "http://127.0.0.1:18791")
    ok, detail = RUN.preflight_search_setup(run=ok_run)
    assert ok and "patched_files" in detail
    assert seen[0][:7] == ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python3"]
    assert seen[0][7] == "wildclawbench-ubuntu:v1.3"

    def bad_run(cmd, **k):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="provider is not brave")
    ok, detail = RUN.preflight_search_setup(run=bad_run)
    assert not ok and "provider is not brave" in detail
    monkeypatch.setenv("WILDCLAW_SEARCH_ORIGIN", "")
    ok, detail = RUN.preflight_search_setup(run=ok_run)
    assert not ok and "WILDCLAW_SEARCH_ORIGIN" in detail


def test_gemini_usage_file_is_read(monkeypatch, tmp_path):
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path))
    assert RUN._gemini_usage() == {}
    (tmp_path / "gemini_usage.json").write_text(json.dumps({"judge": {"gemini-3.7-flash": 4}}))
    assert RUN._gemini_usage() == {"judge": {"gemini-3.7-flash": 4}}


def test_entrypoint_keeps_keys_away_from_the_launcher():
    text = open(os.path.join(_DOCKER_DIR, "wildclawbench_entrypoint.sh"), encoding="utf-8").read()
    assert ('env -u GEMINI_API_KEYS -u GEMINI_API_KEY -u GBENCH_MODEL_API_KEY \\\n'
            '    python3 "$ADAPTER_DIR/wildclawbench_run.py"') in text
    assert "BRAVE" not in text


# --------------------------------------------------------------------------- #
# Runner: no key on argv, the search tool mounted, substitutions recorded.
# --------------------------------------------------------------------------- #
def _run_runner(monkeypatch, tmp_path, env=None, summary_extra=None):
    harness = tmp_path / "harness"
    harness.mkdir()
    monkeypatch.setattr(W, "check_wildclawbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(W, "_harness_dir", lambda: str(harness))
    monkeypatch.setattr(W, "free_port", lambda preferred=None: preferred)
    monkeypatch.setattr(W.tempfile, "mkdtemp", lambda prefix="": str(tmp_path / "workdir"))
    os.makedirs(tmp_path / "workdir", exist_ok=True)
    for k in ("GBENCH_WILDCLAWBENCH_TASK_NETWORK", "GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT",
              "GBENCH_WILDCLAWBENCH_TASK_ENDPOINT", "GBENCH_WILDCLAWBENCH_JUDGE_PORT",
              "GBENCH_SHARD", "GBENCH_WILDCLAWBENCH_MODEL_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    captured = {}

    def fake_run(cmd, *a, **k):
        if cmd[:3] == ["docker", "run", "--rm"]:
            captured["cmd"], captured["env"] = list(cmd), dict(k.get("env") or {})
            workdir = next(x.split("=", 1)[1] for x in cmd if str(x).startswith("WILDCLAW_WORKDIR="))
            summary = {"n_tasks": 60, "n_scored": 60, "n_missing": 0, "global_mean": 0.25,
                       "judge_fallback_tasks": 0, "n_search_setup_failures": 0,
                       "gemini_usage": {"agent_web_search": {"gemini:gemini-3.7-flash": 12}}}
            summary.update(summary_extra or {})
            with open(os.path.join(workdir, "wildclawbench_summary.json"), "w") as f:
                json.dump(summary, f)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(W.subprocess, "run", fake_run)
    result = W.run_wildclawbench("google/gemma-4-E4B-it", "http://127.0.0.1:9123/v1")
    return result, captured


def test_runner_argv_carries_no_key_and_mounts_search_tool(monkeypatch, tmp_path):
    result, cap = _run_runner(monkeypatch, tmp_path, env={
        "GEMINI_API_KEYS": f"{FAKE_KEY},{FAKE_KEY_2}", "BRAVE_API_KEY": "brave-should-not-travel",
        "GBENCH_WILDCLAWBENCH_MODEL_API_KEY": "sk-model-endpoint-secret-1"})
    cmd = cap["cmd"]
    joined = "\n".join(map(str, cmd))
    for secret in (FAKE_KEY, FAKE_KEY_2, "sk-model-endpoint-secret-1", "brave-should-not-travel"):
        assert secret not in joined
    assert "BRAVE" not in joined
    for name in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "GBENCH_MODEL_API_KEY"):
        i = cmd.index(name)
        assert cmd[i - 1] == "-e"                                   # passed by name only
    assert cap["env"]["GEMINI_API_KEYS"] == f"{FAKE_KEY},{FAKE_KEY_2}"
    assert cap["env"]["GEMINI_API_KEY"] == FAKE_KEY
    assert cap["env"]["GBENCH_MODEL_API_KEY"] == "sk-model-endpoint-secret-1"
    assert f"{W._SEARCH_TOOL_PATH}:/app/gbench/search_tool.py:ro" in cmd
    # the default task network is the docker bridge: the proxy is at the bridge gateway
    assert "WILDCLAW_SEARCH_ORIGIN=http://172.17.0.1:18795" in cmd
    assert "WILDCLAW_JUDGE_ENDPOINT=http://172.17.0.1:18795/v1" in cmd

    assert result["status"] == "success" and result["leaderboard_comparable"] is False
    comps = [s["component"] for s in result["gemini_substitutions"]]
    assert len(comps) == 5
    assert any(W.AUDIO_JUDGE_MODEL in s["official"] for s in result["gemini_substitutions"])
    assert any("web_search tool" in c for c in comps)
    assert any("04_Search_Retrieval_task_1" in c for c in comps)
    search_sub = next(s for s in result["gemini_substitutions"] if "web_search tool" in s["component"])
    assert search_sub["official"].startswith("Brave Search API")
    assert "Gemini Google Search grounding" in search_sub["served_by"]
    reason = result["leaderboard_comparable_reason"]
    assert "not the Brave Search API" in reason and "not openai/gpt-5.4" in reason
    assert "OpenRouter's web_search tool" in reason
    assert result["gemini_usage"] == {"agent_web_search": {"gemini:gemini-3.7-flash": 12}}
    assert result["n_search_setup_failures"] == 0


def test_runner_bridge_mode_search_origin(monkeypatch, tmp_path):
    _result, cap = _run_runner(monkeypatch, tmp_path,
                               env={"GBENCH_WILDCLAWBENCH_TASK_NETWORK": "bridge"})
    assert "WILDCLAW_SEARCH_ORIGIN=http://172.17.0.1:18795" in cap["cmd"]
    assert W._search_origin("http://gw.example:9000/v1") == "http://gw.example:9000"
    assert W._search_origin("") == ""


def test_runner_records_search_setup_failures(monkeypatch, tmp_path):
    result, _cap = _run_runner(monkeypatch, tmp_path, summary_extra={"n_search_setup_failures": 2})
    assert result["n_search_setup_failures"] == 2
    assert "2 task container(s) could not be pointed at the Gemini web search" in \
        result["leaderboard_comparable_reason"]


def test_runner_no_summary_error_is_redacted(monkeypatch, tmp_path):
    harness = tmp_path / "harness"
    harness.mkdir()
    monkeypatch.setattr(W, "check_wildclawbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(W, "_harness_dir", lambda: str(harness))
    monkeypatch.setattr(W.tempfile, "mkdtemp", lambda prefix="": str(tmp_path / "wd"))
    os.makedirs(tmp_path / "wd", exist_ok=True)
    monkeypatch.setattr(W.subprocess, "run", lambda cmd, *a, **k: subprocess.CompletedProcess(
        cmd, 5, stdout="", stderr=f"[wildclawbench] PREFLIGHT FAILED: {FAKE_KEY}"))
    with pytest.raises(Exception) as ei:
        W.run_wildclawbench("m", "http://127.0.0.1:9123/v1")
    assert "PREFLIGHT FAILED" in str(ei.value) and FAKE_KEY not in str(ei.value)


def test_docs_name_every_suite_knob_and_no_brave_requirement():
    src = open(W.__file__, encoding="utf-8").read()
    doc = open(os.path.join(_REPO, "docs", "evals", "wildclawbench.md"), encoding="utf-8").read()
    for knob in sorted(set(re.findall(r"GBENCH_WILDCLAWBENCH_[A-Z_]+", src))):
        assert knob in doc, knob
    # BRAVE_API_KEY is named only in the record of the official service, never read or forwarded.
    assert not re.search(r"environ[^\n]*BRAVE", src)
    assert not any("BRAVE" in k for k in W._FORWARDED_KNOBS)
    assert "export BRAVE_API_KEY" not in doc


@pytest.fixture(autouse=True)
def _metadata_guard_answers_blocked(monkeypatch):
    """Docker is stubbed in these tests: the cloud metadata guard the suites call before starting a
    container (container_guard.require_metadata_blocked) answers "unreachable" without a probe.
    tests/test_guard_tool_suites.py checks the suites' calls to it."""
    from gbench.runners.eval_suites import container_guard
    monkeypatch.setattr(container_guard, "require_metadata_blocked",
                        lambda *a, **k: container_guard.UNREACHABLE)
