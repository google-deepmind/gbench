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

"""No key in a BFCL harness container, and the execution guard in `bfcl generate`.

* `bfcl generate` (the agentic memory and web_search categories) runs the model's calls with
  eval() in bfcl_eval's multi_turn_utils. The harness child installs the bfcl suite's execution
  guard before the CLI dispatches: the environment functions upstream intends (a public method
  of an involved backend, literal arguments) run exactly as upstream; anything else gets
  upstream's own "Function call X is not allowed." Refusals are logged, counted and make the
  result not leaderboard-comparable.
* No process that eval()s the model's calls holds a key: a real endpoint key (OPENAI_API_KEY)
  is added by swebench_common.EndpointRelay on the bridge gateway and the container holds the
  placeholder EMPTY; the Gemini key for the web searches is added by gemini_relay.GeminiRelay and
  the container holds its per-run placeholder. A placeholder endpoint key is passed as it is. A
  launch that would still pass a credential stops (infra_required).
* The web_search `bfcl generate` resolves the grounding links' host from the bridge network; the
  metadata guard checks it (container_guard resolve=).
* A gate failure is labeled with the suite that runs: bfcl and docs/evals/bfcl.md when bfcl runs
  the agentic track.

Docker, the metadata probe and the relay probe are stubbed; relays listen on 127.0.0.1 and talk
to stub servers on 127.0.0.1 (never a serving port). Nothing contacts a model or Gemini.
"""

import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from gbench.runners.eval_suites import base, bfcl
from gbench.runners.eval_suites import bfcl_v4_agentic as AG
from gbench.runners.eval_suites import container_guard as CG
from gbench.runners.eval_suites import gemini_relay as GR
from gbench.runners.eval_suites import search_tool
from gbench.runners.eval_suites import tau_common as TC

REPO = Path(__file__).resolve().parents[1]
# Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
ENDPOINT_KEY = "sk-" + "realENDPOINTkey0123456789"
GEMINI_KEY = "AIzaREALgeminiKEY0123456789abcdef"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GEMINI_API_KEYS", "GEMINI_OPENAI_BASE_URL",
                 "GBENCH_GEMINI_RELAY_URL", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy",
                 "https_proxy", "NO_PROXY", "no_proxy", "GBENCH_SEARCH_MODELS",
                 "GBENCH_SEARCH_MODEL", "GBENCH_BFCL_TIMEOUT_S", "BFCL_TIMEOUT_S",
                 "GBENCH_SHARD", AG.SEARCH_LOG_ENV, AG.GUARD_LOG_ENV):
        monkeypatch.delenv(name, raising=False)
    AG._RELAY_HOSTS_OK.clear()
    yield
    AG._RELAY_HOSTS_OK.clear()


def _stub_bridge(monkeypatch, gate_calls=None):
    """Relays on 127.0.0.1; the metadata gate recorded, the relay probe a no-op."""
    calls = [] if gate_calls is None else gate_calls
    monkeypatch.setattr(AG, "_metadata_gate",
                        lambda eval_name, docs_url, names: calls.append(
                            (eval_name, docs_url, list(names))))
    monkeypatch.setattr(AG, "_bridge_gateway", lambda eval_name, docs_url: "127.0.0.1")
    monkeypatch.setattr(AG, "_require_relay_reachable", lambda *a, **k: None)
    return calls


def _gate_recorder(monkeypatch, fail=None):
    calls = []

    def gate(eval_name, image, url, docs_url):
        calls.append((eval_name, image, url, docs_url))
        if fail is not None:
            raise fail(eval_name, docs_url)
        return TC.container_url(url)
    monkeypatch.setattr(TC, "require_endpoint_network", gate)
    return calls


class _Upstream:
    """A stub server on 127.0.0.1 that records each request's path and headers (names in lower
    case)."""

    def __init__(self, body=None):
        seen = self.seen = []
        payload = json.dumps(body or {"ok": True}).encode()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self):
                length = int(self.headers.get("Content-Length") or 0)
                seen.append({"method": self.command, "path": self.path,
                             "headers": {k.lower(): v for k, v in self.headers.items()},
                             "body": self.rfile.read(length) if length else b""})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            do_GET = do_POST = _reply

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _capture_launch(monkeypatch, during=None):
    seen = {}

    def launch(cmd, env, timeout, name, stream=False):
        seen.update(cmd=list(cmd), env=dict(env))
        if during is not None:
            during(cmd, env)
        return 0, "ok"
    monkeypatch.setattr(AG, "_launch", launch)
    return seen


def _forwarded(cmd, env):
    """{name: value} of the variables the container receives (`-e NAME`)."""
    names = [cmd[i + 1] for i, a in enumerate(cmd[:-1]) if a == "-e" and "=" not in cmd[i + 1]]
    return {n: env.get(n) for n in names}


# --------------------------------------------------------------------------- #
# the execution guard in `bfcl generate`'s harness child
# --------------------------------------------------------------------------- #
def test_the_child_installs_the_guard_before_the_cli_dispatches():
    code = AG._CHILD_BOOTSTRAP.format(model="m", args=["generate"])
    assert "_install_execution_guard(os.environ.get(\"GBENCH_BFCL_GUARD_LOG\"))" in code
    assert code.index("_install_execution_guard(os.environ") < code.index("cli()")
    assert code.index("_serve_web_search_with_gemini(os.environ") \
        < code.index("_install_execution_guard(os.environ")
    assert AG.GUARD_LOG_ENV == "GBENCH_BFCL_GUARD_LOG" and AG.GUARD_LOG_ENV in AG._CHILD_ENV_NAMES


@pytest.fixture
def mtu(monkeypatch, tmp_path):
    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path / "project"))
    import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils as module
    yield module
    vars(module).pop("eval", None)


def _run_calls(mtu, calls, model, classes, config, entry):
    try:
        out, _ = mtu.execute_multi_turn_func_call(calls, config, classes, model, entry)
    finally:
        bfcl._drop_instances(model, entry)
    return out


def _memory_config(tmp_path, entry):
    return {"MemoryAPI_kv": {"model_result_dir": tmp_path / "res", "test_id": entry,
                             "scenario": "customer"}}


BENIGN_MEMORY = ["core_memory_add(key='name', value='Ada')", "core_memory_retrieve(key='name')",
                 "core_memory_list_keys()", "archival_memory_add(key='k', value='v')",
                 "archival_memory_key_search(query='k', k=1)", "no_such_tool(x=1)",
                 "core_memory_add(key='n', value=[1, {'a': (2, 3)}])"]
INJECTED = ["len(obj=1)", "__import__('os').getcwd()",
            "core_memory_add(key=__import__('os').getcwd(), value='x')",
            "open(file='/nonexistent')", "exec(source='1')", "globals()"]


def test_memory_calls_run_as_upstream_runs_them_and_injected_code_is_refused(mtu, tmp_path):
    """The agentic memory backend through the same execute_multi_turn_func_call `bfcl
    generate`'s loop calls: every environment call gives upstream's result, every injected call
    string upstream's own refusal text, and each refusal is logged."""
    entry = "memory_kv_prereq_0-customer-0"
    classes = ["MemoryAPI_kv"]
    plain = _run_calls(mtu, BENIGN_MEMORY, "unguarded", classes, _memory_config(tmp_path, entry),
                       entry)
    log = tmp_path / "guard.jsonl"
    AG._install_execution_guard(str(log))
    guarded = _run_calls(mtu, BENIGN_MEMORY + INJECTED, "guarded", classes,
                         _memory_config(tmp_path, entry), entry)
    assert guarded[:len(BENIGN_MEMORY)] == plain
    assert all(r.startswith("Error during execution: Function call ") and
               r.endswith(" is not allowed.") for r in guarded[len(BENIGN_MEMORY):]), guarded
    # each refused call string as the guard saw it (a backend method with its instance prefix)
    rows = [json.loads(line)["call"] for line in log.read_text().splitlines()]
    assert len(rows) == len(INJECTED) and all(r.endswith(c) for r, c in zip(rows, INJECTED))
    assert AG._read_guard_log(log) == rows


def test_web_search_calls_are_guarded_the_same_way(mtu, tmp_path):
    entry = "web_search_base_0"
    config = {"WebSearchAPI": {"show_snippet": True}}
    calls = ["fetch_url_content(url='file:///nonexistent')",
             "fetch_url_content(url=__import__('os').getcwd())",
             "search_engine_query(keywords=str(len('x')))"]
    plain = _run_calls(mtu, calls[:1], "unguarded", ["WebSearchAPI"], config, entry)
    AG._install_execution_guard(None)
    out = _run_calls(mtu, calls, "guarded", ["WebSearchAPI"], config, entry)
    assert out[0] == plain[0]
    assert out[1:] == ["Error during execution: Function call fetch_url_content is not allowed.",
                       "Error during execution: Function call search_engine_query is not "
                       "allowed."]


def test_generate_loop_reaches_the_guarded_eval(mtu):
    """BaseHandler.inference_multi_turn_FC calls base_handler.execute_multi_turn_func_call, the
    same function object, whose globals are multi_turn_utils': the guard there covers it."""
    import bfcl_eval.model_handler.base_handler as base_handler
    assert base_handler.execute_multi_turn_func_call.__globals__ is vars(mtu)
    AG._install_execution_guard(None)
    assert isinstance(vars(mtu)["eval"], bfcl._ExecutionGuard)


def test_a_failing_refusal_log_never_changes_the_answer(mtu, tmp_path):
    guard = bfcl._ExecutionGuard(mtu, on_refused=lambda s: (_ for _ in ()).throw(OSError("x")))
    with pytest.raises(Exception, match="Function call open is not allowed."):
        guard("open(file='x')")
    assert guard.refused == ["open(file='x')"]


@pytest.fixture
def agentic_run(monkeypatch, tmp_path):
    """run_bfcl_v4_agentic with the harness CLI stubbed; `state["refusals"]` lands in the
    guard log during `bfcl generate`."""
    root = tmp_path / "bfcl"
    state = {"cli": [], "refusals": []}
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(root))
    monkeypatch.setattr(AG, "check_bfcl_prerequisites",
                        lambda url: (True, "", list(AG.MEMORY_CATEGORIES)))

    def cli(args, env, timeout, model="", backend=None):
        state["cli"].append({"args": list(args), "env": dict(env)})
        if args[0] == "generate":
            with open(env[AG.GUARD_LOG_ENV], "a", encoding="utf-8") as f:
                for call in state["refusals"]:
                    f.write(json.dumps({"call": call}) + "\n")
        return 0, ""
    monkeypatch.setattr(AG, "_run_cli", cli)
    monkeypatch.setattr(AG, "_clear_stale_scores", lambda *a: 0)
    monkeypatch.setattr(AG, "_parse_scores", lambda r, m, cats: {
        c: {"correct": 1, "total": 2, "accuracy": 50.0} for c in cats})
    monkeypatch.setattr(AG, "_parse_score_traces", lambda *a: [])
    monkeypatch.setattr(AG, "_score_error_types", lambda *a: {})
    return state


def test_refusals_during_generation_are_counted_named_and_not_comparable(agentic_run):
    agentic_run["refusals"] = ["len(obj=1)", "exec(source='1')"]
    res = AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1)
    report = res["bfcl_report"]
    assert report["execution_guard_refusals"] == 2
    assert report["execution_guard_refused_calls"] == agentic_run["refusals"]
    assert res["leaderboard_comparable"] is False
    assert ("2 agentic calls refused by the execution guard (upstream would have run them)"
            in res["leaderboard_comparable_reason"])
    gen = next(c for c in agentic_run["cli"] if c["args"][0] == "generate")
    assert gen["env"][AG.GUARD_LOG_ENV].endswith(AG.GUARD_LOG_NAME)


def test_a_run_without_refusals_records_none(agentic_run):
    res = AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1)
    assert res["bfcl_report"]["execution_guard_refusals"] == 0
    assert "execution guard" not in (res["leaderboard_comparable_reason"] or "")
    assert res["bfcl_report"]["credentials"]["endpoint_key"].startswith("none")
    assert res["bfcl_report"]["relays"] == []


def test_the_guard_log_is_emptied_at_the_start_of_each_run(agentic_run, tmp_path):
    agentic_run["refusals"] = ["exec(source='1')"]
    AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1)
    agentic_run["refusals"] = []
    res = AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1)
    assert res["bfcl_report"]["execution_guard_refusals"] == 0


# --------------------------------------------------------------------------- #
# the endpoint key: placeholder in the container, EndpointRelay on the gateway
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value, real", [("EMPTY", False), ("dummy", False), ("not-needed", False),
                                         ("", False), (ENDPOINT_KEY, True)])
def test_endpoint_key_follows_the_placeholder_rule(value, real):
    assert AG.endpoint_key({"OPENAI_API_KEY": value}) == (value if real else None)


def test_generate_with_a_real_key_goes_through_the_endpoint_relay(monkeypatch, tmp_path):
    upstream = _Upstream({"choices": []})
    gate_calls = _stub_bridge(monkeypatch)
    endpoint_gate = _gate_recorder(monkeypatch)
    sent = {}

    def during(cmd, env):
        req = urllib.request.Request(env["OPENAI_BASE_URL"] + "/chat/completions",
                                     data=b'{"model": "m"}', method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {env['OPENAI_API_KEY']}"})
        with urllib.request.urlopen(req, timeout=10) as r:
            sent["status"] = r.status
    seen = _capture_launch(monkeypatch, during)
    relays = []
    env = {"BFCL_PROJECT_ROOT": str(tmp_path), "OPENAI_BASE_URL": upstream.url + "/v1",
           "REMOTE_OPENAI_BASE_URL": upstream.url + "/v1", "OPENAI_API_KEY": ENDPOINT_KEY,
           "REMOTE_OPENAI_API_KEY": ENDPOINT_KEY, AG.REQUEST_TIMEOUT_ENV: "7200"}
    try:
        with AG.suite_context("bfcl_v4_agentic", AG.DOCS_URL, relays=relays):
            AG._run_cli(["generate", "--model", "m"], env, None, model="m")
    finally:
        upstream.close()
    held = _forwarded(seen["cmd"], seen["env"])
    assert held["OPENAI_API_KEY"] == held["REMOTE_OPENAI_API_KEY"] == AG.ENDPOINT_KEY_PLACEHOLDER
    assert ENDPOINT_KEY not in json.dumps(held) and ENDPOINT_KEY not in " ".join(seen["cmd"])
    assert held["OPENAI_BASE_URL"].startswith("http://127.0.0.1:") and \
        held["OPENAI_BASE_URL"] != upstream.url + "/v1"
    # the relay added the real key on the way out
    assert sent["status"] == 200
    # (the relay first reads GET /v1/models once to learn whether the endpoint is vLLM,
    # swebench_common.EndpointRelay._probe_vllm, with the key too)
    assert all(s["headers"]["authorization"] == f"Bearer {ENDPOINT_KEY}" for s in upstream.seen)
    posts = [s for s in upstream.seen if s["path"] != "/v1/models"]
    assert posts[0]["path"] == "/v1/chat/completions"
    # the metadata guard ran for the bridge network; the direct-endpoint check did not
    assert gate_calls == [("bfcl_v4_agentic", AG.DOCS_URL, [])] and endpoint_gate == []
    [record] = relays
    assert record["endpoint_relay"]["requests_relayed"] == 1 and record["gemini_relay"] is None
    assert ENDPOINT_KEY not in json.dumps(record)
    assert env["OPENAI_API_KEY"] == ENDPOINT_KEY, "the caller's env is untouched"


def test_the_endpoint_relay_outlasts_the_clients_own_timeout(monkeypatch):
    _stub_bridge(monkeypatch)
    with AG.bridge_services("bfcl", "d.md", "http://127.0.0.1:9/v1", key=ENDPOINT_KEY,
                            request_timeout=100000) as svc:
        assert svc.endpoint_relay.TIMEOUT_S == 100060
        assert svc.endpoint_url.startswith("http://127.0.0.1:")
    assert svc.endpoint_relay._server is None, "stopped at the end of the block"


def test_a_placeholder_key_is_passed_as_it_is_with_no_relay(monkeypatch, tmp_path):
    gate_calls = _stub_bridge(monkeypatch)
    endpoint_gate = _gate_recorder(monkeypatch)
    seen = _capture_launch(monkeypatch)
    env = {"BFCL_PROJECT_ROOT": str(tmp_path), "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
           "OPENAI_API_KEY": "not-needed"}
    AG._run_cli(["generate", "--model", "m"], env, None, model="m")
    held = _forwarded(seen["cmd"], seen["env"])
    assert held["OPENAI_API_KEY"] == "not-needed"
    assert held["OPENAI_BASE_URL"] == f"http://{TC.HOST_ALIAS}:8000/v1"
    assert gate_calls == [] and endpoint_gate == [
        ("bfcl_v4_agentic", AG.bfcl_image(), "http://127.0.0.1:8000/v1", AG.DOCS_URL)]


def test_evaluate_gets_no_key_and_no_network(monkeypatch, tmp_path):
    _gate_recorder(monkeypatch, fail=lambda n, d: AssertionError("evaluate needs no gate"))
    seen = _capture_launch(monkeypatch)
    env = {"BFCL_PROJECT_ROOT": str(tmp_path), "OPENAI_API_KEY": ENDPOINT_KEY,
           "GEMINI_API_KEY": GEMINI_KEY, "OPENAI_DEFAULT_HEADERS": '{"Authorization": "x"}'}
    AG._run_cli(["evaluate", "--model", "m"], env, None, model="m")
    held = _forwarded(seen["cmd"], seen["env"])
    assert held["OPENAI_API_KEY"] == AG.ENDPOINT_KEY_PLACEHOLDER
    assert "GEMINI_API_KEY" not in held and "OPENAI_DEFAULT_HEADERS" not in held
    assert seen["cmd"][seen["cmd"].index("--network") + 1] == "none"


def test_default_headers_never_reach_a_container():
    assert "OPENAI_DEFAULT_HEADERS" not in AG._CHILD_ENV_NAMES


def test_a_credential_in_a_forwarded_variable_stops_the_launch(monkeypatch, tmp_path):
    seen = _capture_launch(monkeypatch)
    env = {"BFCL_PROJECT_ROOT": str(tmp_path),
           "HTTPS_PROXY": "http://user:proxyPASSWORD123@proxy.example:3128"}
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED\] bfcl_v4_agentic.*HTTPS_PROXY "
                                           r"would carry a credential"):
        AG._run_cli(["evaluate", "--model", "m"], env, None, model="m")
    assert seen == {}


def test_run_in_image_refuses_a_key_in_the_job_environment(monkeypatch):
    seen = _capture_launch(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", ENDPOINT_KEY)
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED\] bfcl.*OPENAI_API_KEY"):
        AG.run_in_image("bfcl", "bfcl", "image_probe", {}, env_names=["OPENAI_API_KEY"])
    assert seen == {}


def test_the_multi_turn_part_uses_the_relay_with_a_real_key(monkeypatch):
    _stub_bridge(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", ENDPOINT_KEY)
    seen = {}

    def run_in_image(eval_name, module, func, kwargs, **kw):
        seen.update(kwargs=kwargs, kw=kw)
        return {"traces": [], "guard_refusals": []}
    monkeypatch.setattr(bfcl._agentic, "run_in_image", run_in_image)
    part = bfcl.run_multi_turn_in_image("m", "http://127.0.0.1:8000/v1", 1,
                                        [{"id": "multi_turn_base_0",
                                          "category": "multi_turn_base"}],
                                        temperature=0.001, thinking=False)
    assert seen["kwargs"]["base_url"].startswith("http://127.0.0.1:")
    assert seen["kwargs"]["base_url"] != "http://127.0.0.1:8000/v1"
    assert seen["kw"]["env"]["OPENAI_API_KEY"] == AG.ENDPOINT_KEY_PLACEHOLDER
    assert "OPENAI_API_KEY" in seen["kw"]["env_names"] and seen["kw"]["network"] == "bridge"
    assert part["endpoint_relay"]["relays_to"] == "http://127.0.0.1:8000/v1"
    assert ENDPOINT_KEY not in json.dumps(part)


def test_the_multi_turn_part_without_a_key_reaches_the_endpoint_directly(monkeypatch):
    seen = {}

    def run_in_image(eval_name, module, func, kwargs, **kw):
        seen.update(kwargs=kwargs, kw=kw)
        return {"traces": [], "guard_refusals": []}
    monkeypatch.setattr(bfcl._agentic, "run_in_image", run_in_image)
    part = bfcl.run_multi_turn_in_image("m", "http://127.0.0.1:8000/v1", 1, [],
                                        temperature=0.001, thinking=False)
    assert seen["kwargs"]["base_url"] == "http://127.0.0.1:8000/v1"
    assert part["endpoint_relay"] is None


def test_the_multi_turn_report_names_the_relay(monkeypatch):
    monkeypatch.setattr(bfcl, "_harness_probe", lambda name, multi, single: (
        bfcl.BFCL_EVAL_RELEASE, [{"id": "multi_turn_base_0", "category": "multi_turn_base"}]))
    monkeypatch.setattr(bfcl, "endpoint_network", lambda url: None)
    monkeypatch.setattr(bfcl, "run_multi_turn_in_image", lambda *a, **k: {
        "traces": [{"sample_idx": 0, "source_sample_idx": 0, "attempt_idx": 0,
                    "category": "multi_turn_base", "question_id": "multi_turn_base_0",
                    "is_correct": True, "status": "OK", "health": "ok", "steps": 1}],
        "guard_refusals": [], "harness_error": None,
        "endpoint_relay": {"listens": "http://172.17.0.1:1/v1", "requests_relayed": 1}})
    res = bfcl.run_bfcl("m", "http://127.0.0.1:8000/v1", 1, eval_categories="multi_turn_base")
    assert res["multi_turn_report"]["endpoint_relay"]["requests_relayed"] == 1


def test_endpoint_network_with_a_real_key_checks_the_gateway_not_the_endpoint(monkeypatch):
    gate_calls = _stub_bridge(monkeypatch)
    reached = []
    monkeypatch.setattr(AG, "_require_gateway_reachable",
                        lambda eval_name, docs_url: reached.append((eval_name, docs_url))
                        or "172.17.0.1")
    endpoint_gate = _gate_recorder(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", ENDPOINT_KEY)
    assert bfcl.endpoint_network("http://127.0.0.1:8000/v1") is None
    assert gate_calls == [("bfcl", bfcl.DOCS_URL, [])] and reached == [("bfcl", bfcl.DOCS_URL)]
    assert endpoint_gate == []


def test_the_prerequisite_check_asks_the_endpoint_with_its_key(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", ENDPOINT_KEY)
    asked = []

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(request, timeout=None):
        asked.append(request.get_header("Authorization"))
        return _Resp()
    monkeypatch.setattr(AG.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(AG, "image_problem", lambda: None)
    monkeypatch.setattr(AG, "run_in_image", lambda *a, **k: {
        "release": AG.BFCL_EVAL_RELEASE, "agentic_categories": list(AG.AGENTIC_CATEGORIES),
        "agentic_mismatch": None})
    checks = []
    monkeypatch.setattr(AG, "bridge_network_check", lambda *a, **k: checks.append((a, k)))
    ok, _why, _cats = AG.check_bfcl_prerequisites("http://127.0.0.1:8000/v1")
    assert ok and asked == [f"Bearer {ENDPOINT_KEY}"]
    assert checks[0][1]["key"] == ENDPOINT_KEY


# --------------------------------------------------------------------------- #
# the Gemini key: GeminiRelay on the gateway, a placeholder in the web_search generate
# --------------------------------------------------------------------------- #
GROUNDED = {"candidates": [{"content": {"parts": [{"text": "Paris is the capital."}]},
                            "groundingMetadata": {
                                "groundingChunks": [{"web": {
                                    "uri": "https://vertexaisearch.cloud.google.com/grounding-"
                                           "api-redirect/abc", "title": "example.org"}}],
                                "groundingSupports": [{"segment": {"text": "Paris"},
                                                       "groundingChunkIndices": [0]}]}}]}


def test_the_web_search_generate_holds_a_placeholder_and_searches_through_the_relay(
        monkeypatch, tmp_path):
    gemini = _Upstream(GROUNDED)
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", gemini.url + "/v1beta/openai")
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "gemini-3.7-flash")
    gate_calls = _stub_bridge(monkeypatch)
    _gate_recorder(monkeypatch)
    found = {}

    def during(cmd, env):
        # search_tool in the container, with only what the container receives
        held = _forwarded(cmd, env)
        saved = dict(os.environ)
        try:
            for name in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GBENCH_GEMINI_RELAY_URL"):
                os.environ[name] = held[name]
            found["results"] = search_tool.gemini_search("capital of France",
                                                         model="gemini-3.7-flash")
        finally:
            os.environ.clear()
            os.environ.update(saved)
    seen = _capture_launch(monkeypatch, during)
    relays = []
    env = {"BFCL_PROJECT_ROOT": str(tmp_path), "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
           "OPENAI_API_KEY": "EMPTY", "GEMINI_API_KEY": GEMINI_KEY,
           "GEMINI_API_KEYS": GEMINI_KEY, AG.SEARCH_LOG_ENV: str(tmp_path / "s.jsonl"),
           "GBENCH_SEARCH_MODELS": "gemini-3.7-flash"}
    try:
        with AG.suite_context("bfcl_v4_agentic", AG.DOCS_URL, relays=relays):
            AG._run_cli(["generate", "--model", "m"], env, None, model="m")
    finally:
        gemini.close()
    held = _forwarded(seen["cmd"], seen["env"])
    assert GR.is_placeholder(held["GEMINI_API_KEY"]) and GR.is_placeholder(held["GEMINI_API_KEYS"])
    assert held["GBENCH_GEMINI_RELAY_URL"].startswith("http://127.0.0.1:")
    assert GEMINI_KEY not in json.dumps(held) and GEMINI_KEY not in " ".join(seen["cmd"])
    # the relay added the real key; the search came back in search_tool's shape
    assert gemini.seen[0]["headers"]["x-goog-api-key"] == GEMINI_KEY
    assert gemini.seen[0]["path"] == "/v1beta/models/gemini-3.7-flash:generateContent"
    assert found["results"][0]["url"].startswith("https://vertexaisearch.cloud.google.com/")
    # the grounding links' host is checked from the bridge network (container_guard resolve=)
    assert gate_calls == [("bfcl_v4_agentic", AG.DOCS_URL, [AG.GROUNDING_REDIRECT_HOST])]
    [record] = relays
    rec = record["gemini_relay"]
    assert rec["native_methods"] == ["generateContent"] and rec["openai_calls"] == []
    assert rec["native_models"] == ["gemini-3.7-flash"] and rec["requests_relayed"] == 1
    assert GEMINI_KEY not in json.dumps(record)


def test_the_gemini_relay_forwards_only_the_grounding_call(monkeypatch):
    gemini = _Upstream(GROUNDED)
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", gemini.url + "/v1beta/openai")
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "gemini-3.7-flash")
    _stub_bridge(monkeypatch)
    try:
        with AG.bridge_services("bfcl_v4_agentic", "d.md", "http://127.0.0.1:9/v1",
                                gemini=True) as svc:
            relay = svc.gemini_relay
            codes = []
            for path, method in (("/v1beta/openai/chat/completions", "POST"),
                                 ("/v1beta/models/gemini-9-pro:generateContent", "POST"),
                                 ("/v1beta/models/gemini-3.7-flash:embedContent", "POST")):
                req = urllib.request.Request(relay.origin + path, data=b"{}", method=method,
                                             headers={"x-goog-api-key": relay.placeholder})
                try:
                    urllib.request.urlopen(req, timeout=10)
                    codes.append(200)
                except urllib.error.HTTPError as e:
                    codes.append(e.code)
    finally:
        gemini.close()
    assert codes == [403, 403, 403] and gemini.seen == []


def test_a_proxy_moves_the_name_check_to_the_proxy_and_exempts_the_relay(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    gate_calls = _stub_bridge(monkeypatch)
    _gate_recorder(monkeypatch)
    seen = _capture_launch(monkeypatch)
    env = {"BFCL_PROJECT_ROOT": str(tmp_path), "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
           "GEMINI_API_KEY": GEMINI_KEY, "HTTPS_PROXY": "http://proxy.corp.example:3128",
           AG.SEARCH_LOG_ENV: str(tmp_path / "s.jsonl")}
    AG._run_cli(["generate", "--model", "m"], env, None, model="m")
    assert gate_calls == [("bfcl_v4_agentic", AG.DOCS_URL, ["proxy.corp.example"])]
    held = _forwarded(seen["cmd"], seen["env"])
    for name in ("NO_PROXY", "no_proxy"):
        assert "127.0.0.1" in held[name].split(",") and TC.HOST_ALIAS in held[name].split(",")


def test_bridge_network_check_asks_for_the_grounding_host_only_for_web_search(monkeypatch):
    gate_calls = _stub_bridge(monkeypatch)
    reached = []
    monkeypatch.setattr(AG, "_require_gateway_reachable",
                        lambda eval_name, docs_url: reached.append(eval_name) or "172.17.0.1")
    endpoint_gate = _gate_recorder(monkeypatch)
    AG.bridge_network_check("bfcl", "http://127.0.0.1:8000/v1", "docs/evals/bfcl.md", key=None)
    assert gate_calls == [] and reached == [] and len(endpoint_gate) == 1
    AG.bridge_network_check("bfcl", "http://127.0.0.1:8000/v1", "docs/evals/bfcl.md",
                            web_search=True, key=None)
    assert gate_calls == [("bfcl", "docs/evals/bfcl.md", [AG.GROUNDING_REDIRECT_HOST])]
    assert reached == ["bfcl"] and len(endpoint_gate) == 2


def test_the_guard_resolves_the_names_bfcl_passes(monkeypatch):
    """_metadata_gate is container_guard.require_metadata_blocked on the bridge with resolve=."""
    got = []
    monkeypatch.setattr(CG, "require_metadata_blocked",
                        lambda eval_name, image, network, docs_url=None, *, resolve=(): got.append(
                            (eval_name, image, network, docs_url, list(resolve))))
    AG._metadata_gate("bfcl", "docs/evals/bfcl.md", [AG.GROUNDING_REDIRECT_HOST])
    assert got == [("bfcl", AG.bfcl_image(), "bridge", "docs/evals/bfcl.md",
                    [AG.GROUNDING_REDIRECT_HOST])]


# --------------------------------------------------------------------------- #
# the relay probe (a container of the image on the bridge network)
# --------------------------------------------------------------------------- #
class _FakeDocker:
    def __init__(self, stdout):
        self.stdout, self.calls = stdout, []

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, self.stdout, "")


def test_the_relay_probe_opens_a_tcp_connection_from_the_bridge_and_is_cached(monkeypatch):
    fake = _FakeDocker("GBENCH_RELAY_PROBE=open\n")
    monkeypatch.setattr(AG.subprocess, "run", fake)
    AG._require_relay_reachable("bfcl", "172.17.0.1", 4321, "docs/evals/bfcl.md")
    AG._require_relay_reachable("bfcl", "172.17.0.1", 4322, "docs/evals/bfcl.md")
    [argv] = fake.calls
    assert argv[:5] == ["docker", "run", "--rm", "--pull", "never"]
    i = argv.index("--network")
    assert argv[i:i + 4] == list(TC.ENDPOINT_NETWORK_ARGS)
    assert {"--read-only", "--cap-drop", "no-new-privileges"} <= set(argv)
    assert "('172.17.0.1', 4321)" in argv[-1] and AG.bfcl_image() in argv


def test_an_unreachable_relay_is_infra_required(monkeypatch):
    monkeypatch.setattr(AG.subprocess, "run",
                        _FakeDocker("GBENCH_RELAY_PROBE=closed ConnectionRefusedError\n"))
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED\] bfcl .*172\.17\.0\.1:4321.*"
                                           r"docs/evals/bfcl\.md"):
        AG._require_relay_reachable("bfcl", "172.17.0.1", 4321, "docs/evals/bfcl.md")
    assert AG._RELAY_HOSTS_OK == {}


def test_a_probe_with_no_verdict_is_infra_required(monkeypatch):
    monkeypatch.setattr(AG.subprocess, "run", _FakeDocker(""))
    with pytest.raises(RuntimeError, match="gave no verdict"):
        AG._require_relay_reachable("bfcl", "172.17.0.1", 4321, "d.md")


def test_the_gateway_check_probes_a_live_listener(monkeypatch):
    monkeypatch.setattr(AG, "_bridge_gateway", lambda eval_name, docs_url: "127.0.0.1")
    probed = []

    def probe(eval_name, host, port, docs_url):
        import socket
        socket.create_connection((host, port), 5).close()     # the listener accepts
        probed.append((host, port))
    monkeypatch.setattr(AG, "_require_relay_reachable", probe)
    assert AG._require_gateway_reachable("bfcl", "d.md") == "127.0.0.1"
    assert probed and probed[0][0] == "127.0.0.1"


# --------------------------------------------------------------------------- #
# a gate failure is labeled with the suite that runs
# --------------------------------------------------------------------------- #
def _prereq_stubs(monkeypatch, tmp_path, image_calls):
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(tmp_path))

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(AG.urllib.request, "urlopen", lambda *a, **k: _Resp())
    monkeypatch.setattr(AG, "image_problem", lambda: None)

    def run_in_image(eval_name, *a, **k):
        image_calls.append(eval_name)
        return {"release": AG.BFCL_EVAL_RELEASE,
                "agentic_categories": list(AG.AGENTIC_CATEGORIES), "agentic_mismatch": None}
    monkeypatch.setattr(AG, "run_in_image", run_in_image)


def _metadata_reachable(eval_name, docs_url):
    return base_infra(eval_name, "the cloud metadata server is reachable (DOCKER-USER)", docs_url)


def base_infra(eval_name, reason, docs_url):
    from gbench.runners.eval_suites.swebench_common import infra_required
    return infra_required(eval_name, reason, docs_url)


def test_the_bfcl_suite_labels_an_agentic_gate_failure_as_its_own(monkeypatch, tmp_path):
    image_calls = []
    _prereq_stubs(monkeypatch, tmp_path, image_calls)
    gate = _gate_recorder(monkeypatch, fail=_metadata_reachable)
    with pytest.raises(RuntimeError) as err:
        bfcl._check_agentic("http://127.0.0.1:8000/v1", list(AG.MEMORY_CATEGORIES))
    message = str(err.value)
    assert message.startswith("[INFRA REQUIRED] bfcl cannot run")
    assert "'docs/evals/bfcl.md'" in message and "bfcl_v4_agentic" not in message
    assert gate == [("bfcl", AG.bfcl_image(), "http://127.0.0.1:8000/v1", "docs/evals/bfcl.md")]
    assert image_calls == ["bfcl"]


def test_bfcl_v4_agentic_alone_keeps_its_own_label(monkeypatch, tmp_path):
    image_calls = []
    _prereq_stubs(monkeypatch, tmp_path, image_calls)
    _gate_recorder(monkeypatch, fail=_metadata_reachable)
    with pytest.raises(RuntimeError) as err:
        AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:8000/v1", concurrency=1)
    assert str(err.value).startswith("[INFRA REQUIRED] bfcl_v4_agentic cannot run")
    assert "'docs/evals/bfcl_v4_agentic.md'" in str(err.value)
    assert image_calls == ["bfcl_v4_agentic"]


def test_the_agentic_part_runs_under_the_bfcl_label(monkeypatch):
    """bfcl hands the track its label: a failure inside run_bfcl_v4_agentic (its own prerequisite
    check, `bfcl generate`'s network checks) names bfcl and its doc."""
    monkeypatch.setattr(bfcl, "_check_agentic", lambda url, cats: None)
    seen = {}

    def agentic(*a, **k):
        ctx = AG._suite()
        seen.update(eval_name=ctx.eval_name, docs_url=ctx.docs_url)
        ok, reason = False, "the harness image is gone"
        raise AG.infra_required(ctx.eval_name, reason, ctx.docs_url)
    monkeypatch.setattr(AG, "run_bfcl_v4_agentic", agentic)
    with pytest.raises(RuntimeError, match=r"^\[INFRA REQUIRED\] bfcl cannot run.*"
                                           r"docs/evals/bfcl\.md"):
        bfcl.run_bfcl("m", "http://127.0.0.1:8000/v1", 1, eval_categories="memory_kv")
    assert seen == {"eval_name": "bfcl", "docs_url": "docs/evals/bfcl.md"}
    assert AG._suite().eval_name == "bfcl_v4_agentic", "the label is reset afterwards"


def test_bfcl_checks_the_web_search_needs_before_anything_is_generated(monkeypatch):
    monkeypatch.setattr(AG, "search_backend", lambda: (AG.SEARCH_BACKEND, ""))
    seen = []
    monkeypatch.setattr(AG, "check_bfcl_prerequisites",
                        lambda url: (seen.append((AG._suite().eval_name, AG._suite().docs_url,
                                                  AG._suite().web_search)), (True, "", []))[1])
    assert bfcl._check_agentic("http://x/v1", list(AG.AGENTIC_CATEGORIES)) is None
    assert seen == [("bfcl", "docs/evals/bfcl.md", True)]
    seen.clear()
    monkeypatch.setattr(AG, "search_backend", lambda: (None, "GEMINI_API_KEY is unset"))
    assert "GEMINI_API_KEY is unset" in bfcl._check_agentic("http://x/v1",
                                                             list(AG.AGENTIC_CATEGORIES))
    assert seen == [("bfcl", "docs/evals/bfcl.md", False)]
    seen.clear()
    bfcl._check_agentic("http://x/v1", list(AG.MEMORY_CATEGORIES))
    assert seen == [("bfcl", "docs/evals/bfcl.md", False)]


def test_the_prerequisite_check_includes_the_web_search_needs_when_asked(monkeypatch, tmp_path):
    image_calls = []
    _prereq_stubs(monkeypatch, tmp_path, image_calls)
    checks = []
    monkeypatch.setattr(AG, "bridge_network_check",
                        lambda eval_name, url, docs_url, **k: checks.append(
                            (eval_name, docs_url, k["web_search"])))
    with AG.suite_context("bfcl", "docs/evals/bfcl.md", web_search=True):
        assert AG.check_bfcl_prerequisites("http://127.0.0.1:8000/v1")[0]
    AG.check_bfcl_prerequisites("http://127.0.0.1:8000/v1")
    assert checks == [("bfcl", "docs/evals/bfcl.md", True),
                      ("bfcl_v4_agentic", AG.DOCS_URL, False)]


# --------------------------------------------------------------------------- #
# docs match the code
# --------------------------------------------------------------------------- #
def _flat(rel):
    return " ".join((REPO / rel).read_text(encoding="utf-8").split())


@pytest.mark.parametrize("rel", ["docs/evals/bfcl_v4_agentic.md", "docs/evals/bfcl.md"])
def test_the_guides_describe_the_guard_and_the_relays(rel):
    text = _flat(rel)
    assert text.isascii()
    for needle in ("execution guard", "EndpointRelay", "placeholder", "`EMPTY`",
                   "vertexaisearch.cloud.google.com", "Function call X is not allowed."):
        assert needle in text, (rel, needle)


def test_the_agentic_guide_names_the_new_variables_and_records():
    text = _flat("docs/evals/bfcl_v4_agentic.md")
    for needle in ("GBENCH_BFCL_GUARD_LOG", "GBENCH_GEMINI_RELAY_URL", "OPENAI_DEFAULT_HEADERS",
                   "bfcl_report.execution_guard_refusals", "bfcl_report.relays",
                   "bfcl_report.credentials", "GeminiRelay", "generateContent"):
        assert needle in text, needle


def test_the_dockerfile_header_says_no_key_enters_the_image():
    text = _flat("gbench/docker/bfcl.Dockerfile")
    assert "The Gemini key is passed through the environment" not in text
    assert "no key enters a container" in text


def test_a_leftover_relay_address_is_never_forwarded(monkeypatch, tmp_path):
    seen = _capture_launch(monkeypatch)
    env = {"BFCL_PROJECT_ROOT": str(tmp_path), "GBENCH_GEMINI_RELAY_URL": "http://10.0.0.9:1"}
    AG._run_cli(["evaluate", "--model", "m"], env, None, model="m")
    assert "GBENCH_GEMINI_RELAY_URL" not in _forwarded(seen["cmd"], seen["env"])
