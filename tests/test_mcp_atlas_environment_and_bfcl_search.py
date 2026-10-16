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

"""Canonical tool suites: mcp_atlas in the release's agent environment, bfcl_v4_agentic on the
harness' own search, gaia's attachment packages.

Offline. The agent environment is a local stand-in HTTP server that answers the way the
release's FastAPI app does (`/`, `/enabled-servers`, `/list-tools`, `/call-tool`); the model
endpoint and the judge are replaced at their seams.
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import mcp_atlas as M

REPO = Path(__file__).resolve().parents[1]
DOC = (REPO / "docs" / "evals" / "mcp_atlas.md").read_text(encoding="utf-8")
DOCKERFILE = (REPO / "docker" / "mcp_atlas.Dockerfile").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# a stand-in agent environment
# --------------------------------------------------------------------------- #
class _Env:
    """What the fake environment serves and what it was asked."""

    def __init__(self, catalog=None, servers=None, call=None):
        self.catalog = catalog or []
        self.servers = servers if servers is not None else [[s, "OK"] for s in M.RELEASE_SERVERS]
        self.call = call or (lambda name, args: (200, [{"type": "text", "text": "ok"}]))
        self.calls = []


def _serve(env):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status, body):
            data = (body if isinstance(body, str) else json.dumps(body)).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                return self._send(200, {"message": "MCP Agent Environment API"})
            if self.path == "/enabled-servers":
                online = sum(1 for _, st in env.servers if st == "OK")
                return self._send(200, {"servers": env.servers, "total": len(env.servers),
                                        "online": online, "offline": len(env.servers) - online})
            return self._send(404, {"detail": "Not Found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/list-tools":
                return self._send(200, env.catalog)
            if self.path == "/call-tool":
                env.calls.append(body)
                status, out = env.call(body["tool_name"], body["tool_args"])
                return self._send(status, out)
            return self._send(404, {"detail": "Not Found"})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


@pytest.fixture
def fake_env():
    servers = []

    def _start(**kw):
        env = _Env(**kw)
        srv, url = _serve(env)
        servers.append(srv)
        return env, url

    yield _start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def _run(coro_fn, sandbox):
    token = M._ACTIVE_SANDBOX.set(sandbox)
    try:
        return asyncio.run(coro_fn())
    finally:
        M._ACTIVE_SANDBOX.reset(token)


# --------------------------------------------------------------------------- #
# mcp_atlas: the release's server set and pre-flight check
# --------------------------------------------------------------------------- #
def test_release_server_table_is_the_36_template_servers():
    assert len(M.RELEASE_SERVERS) == 36
    assert sum(1 for v in M.RELEASE_SERVERS.values() if not v) == 20, "20 need no key"
    assert set(M.HEALTH_CHECK_CALLS) == set(M.RELEASE_SERVERS)
    assert set(M.DATA_SETUP_SERVERS) <= set(M.RELEASE_SERVERS)
    assert M.RELEASE_SERVERS["google-workspace"] == (
        "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN")
    assert M.HEALTH_CHECK_CALLS["fetch"] == ("fetch_fetch", {"url": "https://httpbin.org/get"})


@pytest.mark.parametrize("server,status,body,ok", [
    ("x", 200, '[{"type": "text", "text": "fine"}]', True),
    ("x", 500, '{"detail": "boom"}', False),
    ("x", 200, '[{"type": "text", "text": "Error: quota"}]', False),
    ("x", 200, '{"message": "an error happened"}', False),
    ("x", 200, "[]", False),
    ("memory", 200, "[]", True),               # memory/e2b/calendar may be empty
    ("e2b-server", 200, "{}", True),
    ("x", 200, '{"results": []}', False),
    ("x", 200, "No results were found", False),
    ("x", 200, "not json but content", True),
])
def test_health_call_pass_rule_is_test_servers_py(server, status, body, ok):
    assert M.health_call_ok(server, status, body) is ok


def _grounded(monkeypatch):
    """Stand-in for Gemini Google Search grounding (the search engines' backend); no network."""
    queries = []

    def ground(query, max_results):
        queries.append(query)
        return [{"title": "T", "url": "https://example.org/", "snippet": "S"}], "gemini:test"

    monkeypatch.setattr(M, "_ground", ground)
    return queries


def test_preflight_passes_when_every_server_answers(fake_env, monkeypatch):
    grounded = _grounded(monkeypatch)
    env, url = fake_env()
    got = M._preflight(M._Sandbox(url))
    assert got["servers_online"] == 36 and "36/36" in got["health_check"]
    tools = {c["tool_name"] for c in env.calls}
    assert tools == {tool for s, (tool, _) in M.HEALTH_CHECK_CALLS.items()
                     if s not in M.GEMINI_PREFLIGHT_SERVERS}
    # brave-search's and ddg-search's checks go to Gemini grounding; exa's and oxylabs' reach
    # their own servers (the keys their page tools use), oxylabs with a fresh random query
    assert len(grounded) == 2
    oxy = next(c for c in env.calls if c["tool_name"] == "oxylabs_google_search_scraper")
    assert oxy["tool_args"]["query"] != "python", "as test_servers.py"


def test_preflight_names_unset_keys_and_data_servers(fake_env, monkeypatch):
    _grounded(monkeypatch)
    for v in {v for vs in M.RELEASE_SERVERS.values() for v in vs}:
        monkeypatch.delenv(v, raising=False)
    keyed = {s for s, v in M.RELEASE_SERVERS.items() if v}

    def call(name, args):
        server = next(s for s, (t, _) in M.HEALTH_CHECK_CALLS.items() if t == name)
        if server in keyed:
            return 500, {"detail": f"Failed to call tool '{name}': Unknown tool: {name}"}
        if server == "fetch":
            return 500, {"detail": "Failed to call tool 'fetch_fetch': upstream 503"}
        return 200, [{"type": "text", "text": "ok"}]

    _, url = fake_env(servers=[[s, "OK"] for s in M.RELEASE_SERVERS if s not in keyed],
                      call=call)
    sb = M._Sandbox(url, container="c")          # started by gbench: its env is checked
    # 15 keyed environment servers (exa and oxylabs among them) plus fetch; brave-search's
    # check answers through Gemini
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*16 of 36") as e:
        M._preflight(sb)
    msg = str(e.value)
    assert "Not running in the agent environment" in msg
    assert "notion (needs NOTION_TOKEN, unset here: NOTION_TOKEN; plus the release's sample data" in msg
    assert "Running but failing their check: fetch" in msg and "upstream 503" in msg


def test_preflight_aborts_on_an_offline_configured_server(fake_env):
    _, url = fake_env(servers=[["github", "ERROR_NOT_ONLINE"], ["fetch", "OK"]])
    with pytest.raises(RuntimeError, match="offline in the agent environment.*github"):
        M._preflight(M._Sandbox(url))


def test_preflight_note_does_not_guess_an_external_environment():
    results = {s: (s != "github", "Unknown tool") for s in M.RELEASE_SERVERS}
    external = M.preflight_failure_reason(results, local_env=False)
    assert "github (needs GITHUB_TOKEN)" in external and "unset here" not in external
    assert M.preflight_failure_reason({s: (True, "") for s in M.RELEASE_SERVERS}, True) == ""


# --------------------------------------------------------------------------- #
# mcp_atlas: tools and tool calls as the harness makes them
# --------------------------------------------------------------------------- #
def test_enabled_tools_use_the_harness_parser():
    assert M.parse_enabled_tools('["fetch_fetch", {"name": "git_git_log"}]') == [
        "fetch_fetch", "git_git_log"]
    # not JSON: run_eval.py falls back to a comma split, and so does gbench
    assert M.parse_enabled_tools("fetch_fetch, whois_whois_domain") == [
        "fetch_fetch", "whois_whois_domain"]
    assert M.parse_enabled_tools("") == []


def test_declared_tools_follow_transform_tool_calls():
    catalog = [
        {"name": "b_tool", "description": None, "inputSchema": {"type": "object",
                                                                 "properties": {"q": {}}}},
        {"name": "a_tool", "inputSchema": None},
        {"name": "unused", "description": "x", "inputSchema": {}},
    ]
    tools = M.declared_tools(["a_tool", "b_tool", "gone_tool"], catalog)
    assert [t["function"]["name"] for t in tools] == ["b_tool", "a_tool"], "catalog order"
    b, a = (t["function"] for t in tools)
    assert b == {"name": "b_tool", "description": None,
                 "parameters": {"type": "object", "properties": {"q": {}}}, "strict": False}
    assert a == {"name": "a_tool", "parameters": {}, "strict": False}
    assert catalog[0]["inputSchema"] is not b["parameters"], "a copy, as {...inputSchema}"


@pytest.mark.parametrize("status,body,expected,kind", [
    (500, '{"detail":"Failed to call tool \'x\': Unknown tool: x"}',
     '{"detail":"Failed to call tool \'x\': Unknown tool: x"}', "error_reply"),
    (422, '{"detail":[{"msg":"Input should be a valid dictionary"}]}',
     '{"detail":[{"msg":"Input should be a valid dictionary"}]}', "error_reply"),
    (200, "[]", "success", "ok"),
    (200, '[{"type": "text", "text": "   "}]', "success", "ok"),
    (200, '[{"type": "image", "data": "AAA", "mimeType": "image/png"}]', "success", "ok"),
    (200, '[{"type": "text", "text": "a"}, {"type": "image", "data": "AAA", "mimeType": "x"}]',
     "Error: [", "schema_error"),
    (200, '[{"type": "text", "text": "a"}, {"type": "resource", "resource": {}}]',
     "Error: [", "schema_error"),
    (200, '[{"type": "text", "text": "one", "annotations": null}, {"type": "text", "text": "two"}]',
     "one\ntwo", "ok"),
])
def test_tool_reply_becomes_the_harness_tool_message(status, body, expected, kind):
    assert M.harness_tool_output(status, body, "x") == (expected, kind)


def test_non_json_reply_is_the_harness_error_line():
    out, kind = M.harness_tool_output(200, "<html>", "fetch_fetch")
    assert out.startswith("Error: Failed to call tool fetch_fetch: ") and kind == "bad_reply"


def test_js_truthiness_of_tool_args():
    assert M._js_truthy([]) and M._js_truthy({}) and M._js_truthy("x") and M._js_truthy(1)
    assert not any(M._js_truthy(v) for v in (None, False, 0, 0.0, ""))


def test_tool_call_posts_the_harness_body_and_returns_its_reply(fake_env):
    def call(name, args):
        if name == "calculator_calculate":
            return 200, [{"type": "text", "text": "42", "annotations": None, "_meta": None}]
        return 500, {"detail": f"Failed to call tool '{name}': Unknown tool: {name}"}

    env, url = fake_env(call=call)
    sb = M._Sandbox(url)

    async def go():
        return [await M.execute_mcp_tool("calculator_calculate", {"expression": "6 * 7"}),
                await M.execute_mcp_tool("nosuch_tool", {}),
                await M.execute_mcp_tool("met-museum_get-museum-object", {"objectId": 1}),
                await M.execute_mcp_tool("calculator_calculate", [])]

    outs = _run(go, sb)
    assert outs[0] == "42"
    assert outs[1] == '{"detail": "Failed to call tool \'nosuch_tool\': Unknown tool: nosuch_tool"}'
    assert env.calls[0] == {"tool_name": "calculator_calculate", "tool_args": {"expression": "6 * 7"}}
    assert env.calls[1]["tool_args"] == {}
    assert env.calls[2]["tool_args"] == {"objectId": 1, "returnImage": False}
    assert env.calls[3]["tool_args"] == [], "[] is truthy in `args || {}`"
    assert (sb.calls, sb.error_replies, sb.timeouts, sb.transport_errors) == (4, 2, 0, 0)


def test_tool_call_timeout_and_unreachable_environment(fake_env, monkeypatch):
    import time as _time

    def slow(name, args):
        _time.sleep(1.0)
        return 200, [{"type": "text", "text": "late"}]

    _, url = fake_env(call=slow)
    monkeypatch.setattr(M, "TOOL_CALL_TIMEOUT_S", 0.2)
    sb = M._Sandbox(url)
    out = _run(lambda: M.execute_mcp_tool("fetch_fetch", {"url": "u"}), sb)
    assert out == "Tool call timed out after 0.2s" and sb.timeouts == 1

    dead = M._Sandbox("http://127.0.0.1:9")
    out = _run(lambda: M.execute_mcp_tool("fetch_fetch", {"url": "u"}), dead)
    assert out.startswith("Error: Failed to call tool fetch_fetch: ") and "\n" not in out
    assert dead.transport_errors == 1


def test_default_tool_call_timeout_is_the_harness_default():
    assert M.TOOL_CALL_TIMEOUT_S == 60.0 and M.LIST_TOOLS_TIMEOUT_S == 180.0
    assert M.HEALTH_TIMEOUT_S == 30.0 and M.HEALTH_CONCURRENCY == 8


# --------------------------------------------------------------------------- #
# mcp_atlas: provisioning
# --------------------------------------------------------------------------- #
def test_prerequisites_name_the_local_build(monkeypatch):
    monkeypatch.delenv("GBENCH_MCP_ATLAS_SANDBOX_URL", raising=False)
    monkeypatch.setattr(M.shutil, "which", lambda name: None)
    ok, reason = M.check_mcp_atlas_prerequisites()
    assert not ok and "docker build -t gbench-mcp-atlas:f24ba3fb -f docker/mcp_atlas.Dockerfile docker" in reason


def test_prerequisites_refuse_an_image_from_another_commit(monkeypatch):
    monkeypatch.delenv("GBENCH_MCP_ATLAS_SANDBOX_URL", raising=False)
    monkeypatch.setattr(M.shutil, "which", lambda name: "/usr/bin/docker")

    class _P:
        def __init__(self, rc=0, out=""):
            self.returncode, self.stdout = rc, out

    labels = {"value": "0" * 40}

    def label(cmd):
        # the search-engine label is right; the revision labels read labels["value"]
        return (M.IMAGE_SEARCH_LABEL_VALUE if M.IMAGE_SEARCH_LABEL in cmd[4]
                else labels["value"])

    monkeypatch.setattr(M.subprocess, "run", lambda cmd, **kw: (
        _P(0, label(cmd) + "\n") if cmd[:3] == ["docker", "image", "inspect"] else _P()))
    ok, reason = M.check_mcp_atlas_prerequisites()
    assert not ok and M.UPSTREAM_REVISION in reason
    labels["value"] = M.UPSTREAM_REVISION
    assert M.check_mcp_atlas_prerequisites() == (True, "")


def test_an_external_environment_still_needs_docker_for_the_credentialed_servers(monkeypatch):
    # gbench runs the servers that need credentials in its own container even when the
    # environment the model's tools run in is external
    monkeypatch.setenv("GBENCH_MCP_ATLAS_SANDBOX_URL", "http://127.0.0.1:1984")
    monkeypatch.setattr(M.shutil, "which", lambda name: None)
    ok, reason = M.check_mcp_atlas_prerequisites()
    assert not ok and "docker CLI not found" in reason
    seen = []
    monkeypatch.setattr(M, "_external_environment_setup", lambda sandbox: seen.append(sandbox))
    sb = M._open_sandbox()
    assert sb.url == "http://127.0.0.1:1984" and sb.container is None and seen == [sb]


def test_container_gets_the_set_server_variables_without_their_values(monkeypatch):
    for v in {v for vs in M.RELEASE_SERVERS.values() for v in vs}:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    monkeypatch.setenv("BRAVE_API_KEY", " ")
    args = M._container_env_args()
    assert args == ["-e", "GITHUB_TOKEN"] and "secret-token" not in " ".join(args)
    assert not 8000 <= M._free_port() <= 8010


def test_a_failed_docker_run_removes_what_it_created(monkeypatch):
    # an operator MongoDB, so no local MongoDB container is started first
    monkeypatch.setenv("MONGODB_CONNECTION_STRING", "mongodb://db.example:27017/")
    for server in M.CREDENTIALED_SERVERS:
        for var in M.ENVIRONMENT_SERVERS[server]:
            monkeypatch.setenv(var, "v" * 12)
    monkeypatch.setattr(M, "_require_metadata_blocked", lambda image, network, resolve=(): None)
    cmds = []

    class _P:
        returncode, stdout, stderr = 0, "id\n", ""

    def run(cmd, **kw):
        cmds.append(cmd)
        p = _P()
        if cmd[:3] == ["docker", "run", "-d"] and "-p" in cmd:     # the environment container
            p.returncode, p.stderr = 125, "port is already allocated"
        return p

    monkeypatch.setattr(M.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*port is already allocated"):
        M._start_container("gbench-mcp-atlas:f24ba3fb")
    [started] = [c for c in cmds if c[:3] == ["docker", "run", "-d"] and "-p" in c]
    assert started[:4] == ["docker", "run", "-d", "--name"]
    assert started[started.index("-p") + 1].startswith("127.0.0.1:")
    assert started[started.index("-p") + 1].endswith(":1984")
    after = cmds[cmds.index(started) + 1:]
    assert after[0] == ["docker", "rm", "-f", started[4]]
    # then the credentialed-servers container and both networks
    assert after[1] == ["docker", "rm", "-f", started[4] + "-cred"]
    assert [c[:3] for c in after[2:]] == [["docker", "network", "rm"]] * 2


def test_the_container_is_removed_even_when_the_preflight_fails(monkeypatch):
    removed = []
    monkeypatch.setattr(M, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(M, "check_mcp_atlas_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(M, "_open_sandbox", lambda: M._Sandbox("http://env", container="c1"))
    monkeypatch.setattr(M, "_wait_ready", lambda sb, t: None)
    monkeypatch.setattr(M, "_stop_container", lambda sb: removed.append(sb.container))

    def _fail(sb):
        raise RuntimeError("[INFRA REQUIRED] mcp_atlas: pre-flight failed")

    monkeypatch.setattr(M, "_preflight", _fail)
    with pytest.raises(RuntimeError, match="pre-flight failed"):
        M.run_mcp_atlas("m", "http://x", concurrency=1)
    assert removed == ["c1"] and M._ACTIVE_SANDBOX.get() is None


def test_missing_docker_is_infra_required_before_anything_starts(monkeypatch):
    monkeypatch.setattr(M, "gemini_required_skip", lambda *a: None)
    monkeypatch.setattr(M, "check_mcp_atlas_prerequisites",
                        lambda: (False, "docker CLI not found. Build ..."))
    monkeypatch.setattr(M, "_open_sandbox", lambda: pytest.fail("no container may start"))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*docker CLI not found"):
        M.run_mcp_atlas("m", "http://x", concurrency=1)


def test_loader_needs_the_environment_catalog():
    with pytest.raises(RuntimeError, match="list-tools"):
        M._load_mcp_atlas_samples(limit=1)


# --------------------------------------------------------------------------- #
# mcp_atlas: a whole run against the stand-in environment
# --------------------------------------------------------------------------- #
def test_full_run_uses_the_environment_catalog_and_tool_replies(fake_env, monkeypatch):
    catalog = [{"name": "calculator_calculate", "description": "Calculates.",
                "inputSchema": {"type": "object", "properties": {"expression": {}},
                                "required": ["expression"]}},
               {"name": "fetch_fetch", "description": "Fetches.", "inputSchema": {}}]

    def call(name, args):
        if name == "calculator_calculate" and args.get("expression") == "6 * 7":
            return 200, [{"type": "text", "text": "42"}]
        return 200, [{"type": "text", "text": "ok"}]      # the pre-flight calls

    env, url = fake_env(catalog=catalog, call=call)
    _grounded(monkeypatch)
    monkeypatch.setenv("GBENCH_MCP_ATLAS_SANDBOX_URL", url)
    # the external environment's checks, gbench's credentialed-servers container and the
    # image checks they need (tests/test_mcp_credential_isolation_and_dns.py covers them): no
    # docker here
    monkeypatch.setattr(M, "_external_environment_setup", lambda sandbox: None)
    monkeypatch.setattr(M, "check_mcp_atlas_prerequisites", lambda: (True, ""))
    rows = [{"TASK": "t1", "PROMPT": "What is 6 times 7?",
             "ENABLED_TOOLS": '["calculator_calculate", "rijksmuseum-server_search_artwork"]',
             "GTFA_CLAIMS": '["The answer is 42."]', "TRAJECTORY": "[]"}]
    monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: rows)
    monkeypatch.setattr(M, "gemini_required_skip", lambda *a: None)

    seen = []

    async def fake_send(**k):
        seen.append(dict(k, messages=list(k["messages"])))    # base keeps appending to it
        if k["messages"][-1]["role"] == "user":
            return base.Reply(text="", tool_calls=[{"id": "c1", "type": "function", "function": {
                "name": "calculator_calculate", "arguments": '{"expression": "6 * 7"}'}}],
                finish_reason="tool_calls")
        return base.Reply(text=f"It is {k['messages'][-1]['content']}.", tool_calls=None,
                          finish_reason="stop")

    async def fake_judge(prompt, config=None):
        return json.dumps({"claim_text": "c", "coverage_outcome": "fulfilled",
                           "justification": "j", "confidence_level": 1.0}), "gemini-test"

    monkeypatch.setattr(base, "_send_single_request", fake_send)
    monkeypatch.setattr(M, "judge_generate_cascade", fake_judge)
    res = M.run_mcp_atlas("m", "http://127.0.0.1:1/v1", concurrency=1, max_output_tokens=512)

    first = seen[0]
    assert first["messages"] == [{"role": "user", "content": "What is 6 times 7?"}]
    assert first["extra_payload"]["tools"] == [{"type": "function", "function": {
        "name": "calculator_calculate", "description": "Calculates.",
        "parameters": catalog[0]["inputSchema"], "strict": False}}]
    tool_msg = seen[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["content"] == "42"
    assert env.calls[-1] == {"tool_name": "calculator_calculate",
                             "tool_args": {"expression": "6 * 7"}}
    tr = res["sample_traces"][0]
    assert tr["response_text"] == "It is 42." and tr["claim_recall"] == 1.0
    assert res["accuracy"] == res["pass_rate_0.75"] == 100.0
    assert res["leaderboard_column"] == "Pass Rate % (Public 500)"
    envr = res["agent_environment"]
    assert envr["provisioned_by"] == "GBENCH_MCP_ATLAS_SANDBOX_URL"
    assert envr["catalog_tools"] == 2 + len(M.GEMINI_SEARCH_TOOLS)
    assert envr["health_check"] == "passed (36/36 servers)"
    assert envr["enabled_tools_not_in_environment"] == ["rijksmuseum-server_search_artwork"]
    assert res["tool_calls"] == {"executed": 1, "error_replies": 0, "timeouts": 0,
                                 "transport_errors": 0, "refused_by_gbench": {},
                                 "account_error_suspects": {}}
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"].startswith("judge is the gbench Gemini cascade")
    assert res["tool_loop_protocol"]["max_tool_calls"] == 100
    assert M._ACTIVE_SANDBOX.get() is None


# --------------------------------------------------------------------------- #
# mcp_atlas: the Dockerfile is the release Dockerfile
# --------------------------------------------------------------------------- #
#: The steps of services/agent-environment/Dockerfile at the pinned commit that must appear
#: verbatim (the COPY steps are replaced by copies out of the pinned clone).
_RELEASE_STEPS = [
    "RUN apt-get update && apt-get install -y npm bash git procps curl gettext-base && rm -rf /var/lib/apt/lists/*",
    "RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get update",
    "RUN apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*",
    "WORKDIR /agent-environment",
    "RUN uv venv --python 3.12",
    "RUN mkdir -p /data",
    "RUN mv /agent-environment/data/* /data/ 2>/dev/null || true",
    "RUN rm -rf /agent-environment/data",
    "done < /data/repos/git_submodule_info.csv",
    "WORKDIR /data/repos/mcp_code_executor_workspace",
    "RUN rm -rf /agent-environment/dev_scripts",
    "RUN uv sync\n",
    "EXPOSE 1984",
    'ENTRYPOINT ["/agent-environment/entrypoint.sh"]',
    'CMD ["uv", "run", "python", "-m", "uvicorn", "agent_environment.main:app", "--host", "0.0.0.0", "--port", "1984"]',
]


def test_dockerfile_keeps_every_release_step_in_order():
    pos = [DOCKERFILE.index(step) for step in _RELEASE_STEPS]
    assert pos == sorted(pos)
    assert "/install_mcp_packages.sh" in DOCKERFILE


def test_dockerfile_pins_the_release_base_commit_and_cutoff():
    assert f"ARG MCP_ATLAS_REF={M.UPSTREAM_REVISION}" in DOCKERFILE
    assert f'LABEL {M.IMAGE_REVISION_LABEL}="${{MCP_ATLAS_REF}}"' in DOCKERFILE
    assert ("ghcr.io/astral-sh/uv:0.9.30-python3.12-bookworm-slim@sha256:"
            "e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58") in DOCKERFILE
    assert "ARG RESOLVE_CUTOFF=2026-07-17T19:29:31Z" in DOCKERFILE
    assert 'UV_EXCLUDE_NEWER="${RESOLVE_CUTOFF}" NPM_CONFIG_BEFORE="${RESOLVE_CUTOFF}"' in DOCKERFILE
    # the environment's own `uv run` keeps its committed lock
    assert "ENV UV_NO_CONFIG=1" in DOCKERFILE
    assert DOCKERFILE.index("ENV UV_NO_CONFIG=1") > DOCKERFILE.index("RUN uv sync\n")


def test_dockerfile_and_doc_are_plain_ascii():
    for text in (DOCKERFILE, DOC):
        bad = [ln for ln in text.splitlines() if any(ord(c) > 127 for c in ln)]
        assert not bad, bad[:3]


# --------------------------------------------------------------------------- #
# mcp_atlas: the doc states what the code does
# --------------------------------------------------------------------------- #
def test_doc_matches_the_code():
    flat = " ".join(DOC.split())
    for needle in (M.DATASET_REVISION, M.UPSTREAM_REVISION, "Pass Rate % (Public 500)",
                   "docker build -t gbench-mcp-atlas:f24ba3fb -f docker/mcp_atlas.Dockerfile docker",
                   "GBENCH_MCP_ATLAS_IMAGE", "GBENCH_MCP_ATLAS_SANDBOX_URL",
                   "GBENCH_MCP_ATLAS_TEMPERATURE", "--max-output-tokens", "2026-07-17T19:29:31Z",
                   "Tool call timed out after 60s", "Error: [", M.IMAGE_REVISION_LABEL):
        assert needle in flat, needle
    for v in {v for vs in M.RELEASE_SERVERS.values() for v in vs}:
        assert f"`{v}`" in DOC, v
    assert "servable_tool_fraction" not in DOC and "lower bound" not in DOC


# --------------------------------------------------------------------------- #
# bfcl_v4_agentic: web search served by Gemini Google Search grounding only
# --------------------------------------------------------------------------- #
def test_bfcl_web_search_needs_the_gemini_key(monkeypatch):
    from gbench.runners.eval_suites import bfcl_v4_agentic as B
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEYS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-looking-key-0123")
    monkeypatch.setenv("GBENCH_BFCL_SEARCH_BACKEND", "serpapi")
    backend, why = B.search_backend()
    assert backend is None and "GEMINI_API_KEY" in why


def test_bfcl_missing_gemini_key_leaves_only_web_search_infra_required(monkeypatch, tmp_path):
    from gbench.runners.eval_suites import bfcl_v4_agentic as B
    calls = []
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEYS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(B, "check_bfcl_prerequisites",
                        lambda url: (True, "", list(B.AGENTIC_CATEGORIES)))
    monkeypatch.setattr(B, "_run_cli", lambda *a, **k: (calls.append(a), (0, ""))[1])
    monkeypatch.setattr(B, "_parse_scores", lambda root, model, cats: {
        c: {"correct": 1, "total": 2, "accuracy": 50.0} for c in cats})
    # the suite's own run: a hard error before anything is generated (its score weighs the Web
    # Search setting); bfcl's agentic part reports the setting and runs Memory
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*GEMINI_API_KEY"):
        B.run_bfcl_v4_agentic("m", "http://x", concurrency=1)
    assert not calls
    with B.suite_context("bfcl", "docs/evals/bfcl.md"):
        res = B.run_bfcl_v4_agentic("m", "http://x", concurrency=1)
    assert "GEMINI_API_KEY" in res["settings"]["web_search"]["infra_required"]
    [gen] = [a for a in calls if a[0][0] == "generate"]
    assert "web_search" not in " ".join(gen[0]) and "GEMINI_API_KEY" not in gen[1]
    calls.clear()
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*GEMINI_API_KEY"):
        B.run_bfcl_v4_agentic("m", "http://x", concurrency=1, eval_categories="web_search_base")
    assert not calls


def test_bfcl_handler_is_the_harness_handler_plus_the_thinking_toggle():
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
    from gbench.runners.eval_suites import bfcl_v4_agentic as B
    cls = B._patched_openai_handler()
    assert cls.__mro__[1] is OpenAICompletionsHandler
    assert set(vars(cls)) - {"__module__", "__qualname__", "__doc__", "__firstlineno__",
                             "__static_attributes__"} == {"generate_with_backoff"}
    code = B._CHILD_BOOTSTRAP.format(model="m", args=["generate"])
    # the one search reference: the web_search tool served by Gemini grounding
    assert "serpapi" not in code.lower()
    assert code.count("search") == code.count("_serve_web_search_with_gemini") == 2


def test_bfcl_doc_describes_gemini_search_only():
    doc = (REPO / "docs" / "evals" / "bfcl_v4_agentic.md").read_text(encoding="utf-8")
    assert "Gemini Google Search grounding" in doc and "GEMINI_API_KEY" in doc
    assert "SERPAPI_API_KEY" not in doc and "GBENCH_BFCL_SEARCH_BACKEND" not in doc


# --------------------------------------------------------------------------- #
# gaia: the attachment packages are importable in the gbench environment
# --------------------------------------------------------------------------- #
def test_gaia_attachment_renderers_have_their_packages():
    import importlib
    for mod in ("xlrd", "openpyxl", "pymupdf"):
        importlib.import_module(mod)


@pytest.fixture(autouse=True)
def _metadata_guard_answers_blocked(monkeypatch):
    """Docker is stubbed in these tests: the cloud metadata guard the suites call before starting a
    container (container_guard.require_metadata_blocked) answers "unreachable" without a probe.
    tests/test_guard_tool_suites.py checks the suites' calls to it."""
    from gbench.runners.eval_suites import container_guard
    monkeypatch.setattr(container_guard, "require_metadata_blocked",
                        lambda *a, **k: container_guard.UNREACHABLE)
