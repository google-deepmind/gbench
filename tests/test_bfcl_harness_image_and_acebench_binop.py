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

"""The BFCL suites on their local harness image, the failed Gemini-grounded web searches the agentic
track now stops on (and the key it never lets through), and ACEBench's BinOp arguments evaluated as
upstream does.

The unit tests run offline, without Docker. The tests marked `image` run the real harness image
(gbench-bfcl:2026.3.23, built from gbench/docker/bfcl.Dockerfile) against local stub servers,
never a model server or the Gemini API; set GBENCH_TEST_BFCL_IMAGE=1 to run them."""

import ast
import asyncio
import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from gbench.runners.eval_suites import acebench as AB
from gbench.runners.eval_suites import base, bfcl
from gbench.runners.eval_suites import bfcl_v4_agentic as AG
from gbench.runners.eval_suites import tau_common as TC_GUARD
from gbench.runners.eval_suites.fc_common import BFCL_OFFICIAL_SAMPLING

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "evals"
KEY = "gemKEY0123456789abcdef0123456789abcdef"

image = pytest.mark.skipif(not os.environ.get("GBENCH_TEST_BFCL_IMAGE"),
                           reason="set GBENCH_TEST_BFCL_IMAGE=1 to run the BFCL harness image")


# --------------------------------------------------------------------------- #
# Gemini grounding: a search no cascade model served is a harness failure
# --------------------------------------------------------------------------- #

def test_the_search_log_separates_failures_from_answered_searches():
    rows = [{"backend": "gemini:gemini-3.7-flash", "results": 5, "error": None},
            {"backend": "all-grounding-models-failed", "results": 0,
             "error": "search backend failed: HTTP Error 429: Too Many Requests"},
            {"backend": "gemini:gemini-3.6-flash", "results": 0, "error": None},
            {"backend": "none", "results": 0, "error": "web search requires GEMINI_API_KEY"},
            {"backend": "error", "results": 0, "error": "RuntimeError: boom"},
            {"backend": "gemini:gemini-3.7-flash", "results": 2, "error": None}]
    s = AG.search_log_summary(rows)
    assert (s["requests"], s["failed"], s["empty_results"]) == (6, 3, 1)
    assert s["by_backend"]["gemini:gemini-3.7-flash"] == 2
    assert s["failures"] == {
        "all-grounding-models-failed: search backend failed: HTTP Error 429: Too Many Requests": 1,
        "none: web search requires GEMINI_API_KEY": 1, "error: RuntimeError: boom": 1}
    text = AG._failure_text(s)
    assert "for 3 of the 6 web searches" in text and "Gemini Google Search grounding" in text


@pytest.fixture
def agentic(monkeypatch, tmp_path):
    """run_bfcl_v4_agentic with the image's commands stubbed; `seen["log"]` rows land in the
    web search log during the stubbed generate, as the harness child writes them."""
    seen = {"cli": [], "log": [], "env": None, "out": ""}
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(tmp_path / "bfcl"))
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.delenv("GBENCH_BFCL_V4_AGENTIC_TEMPERATURE", raising=False)
    monkeypatch.setattr(AG, "check_bfcl_prerequisites",
                        lambda url: (True, "", list(AG.AGENTIC_CATEGORIES)))
    monkeypatch.setattr(AG, "search_backend", lambda: (AG.SEARCH_BACKEND, ""))

    def cli(args, env, timeout, model="", backend=None):
        seen["cli"].append(list(args))
        if args[0] == "generate":
            seen["env"] = dict(env)
            log = env.get(AG.SEARCH_LOG_ENV)
            if log:
                with open(log, "a", encoding="utf-8") as f:
                    for row in seen["log"]:
                        f.write(json.dumps(row) + "\n")
        return 0, seen["out"]
    monkeypatch.setattr(AG, "_run_cli", cli)
    monkeypatch.setattr(AG, "_clear_stale_scores", lambda *a: 0)
    monkeypatch.setattr(AG, "_parse_scores", lambda root, m, cats: {
        c: {"correct": 1, "total": 2, "accuracy": 50.0} for c in cats})
    monkeypatch.setattr(AG, "_parse_score_traces", lambda *a: [])
    monkeypatch.setattr(AG, "_score_error_types", lambda *a: {})
    return seen


@pytest.mark.parametrize("row,needle", [
    ({"backend": "all-grounding-models-failed", "results": 0,
      "error": "search backend failed: <urlopen error timed out>"},
     "all-grounding-models-failed: search backend failed"),
    ({"backend": "none", "results": 0, "error": "no key"}, "none: no key"),
    ({"backend": "error", "results": 0, "error": "TypeError: x"}, "error: TypeError: x"),
])
def test_a_failed_search_after_generation_leaves_web_search_infra_required(agentic, row,
                                                                          needle):
    agentic["log"] = [{"backend": "gemini:gemini-3.7-flash", "results": 3, "error": None}, row]
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    web = res["settings"]["web_search"]
    assert re.search(r"1 of the 2 web searches made during generation.*" + re.escape(needle),
                     web["infra_required"])
    assert web["error"].startswith("[INFRA REQUIRED] bfcl_v4_agentic")
    # the web_search categories are not evaluated; the memory ones are
    assert [c[0] for c in agentic["cli"]] == ["generate", "evaluate"]
    assert set(res["category_accuracy"]) == set(AG.MEMORY_CATEGORIES)


def test_an_answered_search_without_results_is_not_a_failure(agentic):
    agentic["log"] = [{"backend": "gemini:gemini-3.6-flash", "results": 0, "error": None},
                      {"backend": "gemini:gemini-3.7-flash", "results": 4, "error": None}]
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    assert res["bfcl_report"]["web_search_calls"]["failed"] == 0
    assert res["bfcl_report"]["web_search_calls"]["empty_results"] == 1
    assert "settings_infra_required" not in res


def test_the_key_stays_out_of_memory_only_runs(agentic):
    AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1,
                           eval_categories=",".join(AG.MEMORY_CATEGORIES))
    assert "GEMINI_API_KEY" not in agentic["env"] and "GEMINI_API_KEYS" not in agentic["env"]
    agentic["cli"].clear()
    AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    assert agentic["env"]["GEMINI_API_KEY"] == KEY


def test_a_key_in_the_harness_output_or_files_is_masked(agentic, tmp_path, caplog):
    root = tmp_path / "bfcl"
    (root / "result" / "m").mkdir(parents=True)
    leaked = root / "result" / "m" / "BFCL_v4_web_search_base_result.json"
    leaked.write_text(json.dumps({"inference_log": f"api_key={KEY}"}) + "\n")
    agentic["out"] = f"Traceback ... api_key={KEY}"
    AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    assert KEY not in leaked.read_text() and base.REDACTED in leaked.read_text()


def _gemini_search_tool(monkeypatch, tmp_path):
    """BFCL's web_search backend (a stand-in with upstream's class shape) served by Gemini
    grounding, with the grounding endpoint refusing connections."""
    import types
    import urllib.error

    from gbench.runners.eval_suites import search_tool

    class WebSearchAPI:
        def __init__(self):
            self.show_snippet = True

        def search_engine_query(self, keywords, max_results=10, region="wt-wt"):
            raise AssertionError("the harness' own search must not run")
    ws = types.ModuleType("bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search")
    ws.WebSearchAPI = WebSearchAPI
    bh = types.ModuleType("bfcl_eval.model_handler.base_handler")
    bh.execute_multi_turn_func_call = lambda calls, *a, **k: ([str(c) for c in calls], {})
    fsc = types.ModuleType("bfcl_eval.eval_checker.multi_turn_eval.func_source_code")
    fsc.web_search = ws
    mh = types.ModuleType("bfcl_eval.model_handler")
    mh.base_handler = bh
    for name, mod in (("bfcl_eval.eval_checker.multi_turn_eval.func_source_code", fsc),
                      (ws.__name__, ws), ("bfcl_eval.model_handler", mh), (bh.__name__, bh)):
        monkeypatch.setitem(sys.modules, name, mod)

    def refuse(req, timeout=None):
        # the key travels in the x-goog-api-key header, never in the URL; the refusal text
        # echoes both, as a transport error's text can
        assert "key=" not in req.full_url and KEY not in req.full_url
        assert req.get_header("X-goog-api-key") == KEY
        raise urllib.error.URLError(f"connection refused for {req.full_url} "
                                    f"(x-goog-api-key: {req.get_header('X-goog-api-key')})")
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(search_tool, "_SEARCH_ATTEMPTS", 1)
    monkeypatch.setenv("GBENCH_SEARCH_CASCADE_ROUNDS", "1")
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "gemini-3.7-flash")
    monkeypatch.setattr(search_tool, "_RATE_LIMITER", search_tool._RateLimiter(0, 1))
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    log = tmp_path / "search.jsonl"
    AG._serve_web_search_with_gemini(str(log))
    return WebSearchAPI, log


def test_a_transport_error_reaches_the_harness_without_the_key(monkeypatch, tmp_path):
    """The grounding request carries the key in its x-goog-api-key header, not in its URL; a
    transport error's text that echoes the header is masked; the model gets BFCL's own error
    text, and the log holds no key."""
    WebSearchAPI, log = _gemini_search_tool(monkeypatch, tmp_path)
    result = WebSearchAPI().search_engine_query("tea")
    assert result == {"error": AG.SEARCH_FAILED_TEXT}
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert rows[0]["backend"] == "all-grounding-models-failed" and KEY not in log.read_text()
    assert base.REDACTED in rows[0]["error"] and "connection refused" in rows[0]["error"]
    assert AG.search_log_summary(rows)["failed"] == 1


def test_the_search_tool_keeps_bfcls_method_shape(monkeypatch, tmp_path):
    import inspect
    WebSearchAPI, _log = _gemini_search_tool(monkeypatch, tmp_path)
    inst = WebSearchAPI()
    assert inspect.ismethod(inst.search_engine_query)
    assert "search_engine_query" in dict(inspect.getmembers(inst, inspect.ismethod))
    assert list(inspect.signature(inst.search_engine_query).parameters) == [
        "keywords", "max_results", "region"]
    assert WebSearchAPI.search_engine_query.gbench_backend == AG.SEARCH_BACKEND


# --------------------------------------------------------------------------- #
# The harness image: launch, job protocol, credentials
# --------------------------------------------------------------------------- #

def test_cli_commands_run_in_the_image_with_no_secret_on_the_command_line(monkeypatch, tmp_path):
    seen = {}

    def launch(cmd, env, timeout, name, stream=False):
        seen.update(cmd=cmd, env=env, name=name)
        return 0, "ok"
    monkeypatch.setattr(AG, "_launch", launch)
    # `bfcl generate` runs on the bridge network after the network checks, docker calls of their
    # own (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC_GUARD, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC_GUARD.container_url(url))
    # the keys stay in this process: gbench's endpoint and Gemini relays (here on 127.0.0.1,
    # their docker and metadata checks stubbed: tests/test_bfcl_execution_guard_and_relays.py) add
    # them
    monkeypatch.setattr(AG, "_metadata_gate", lambda *a, **k: None)
    monkeypatch.setattr(AG, "_bridge_gateway", lambda *a, **k: "127.0.0.1")
    monkeypatch.setattr(AG, "_require_relay_reachable", lambda *a, **k: None)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    env = {"BFCL_PROJECT_ROOT": str(tmp_path), "OPENAI_API_KEY": "sk-endpoint-0123456789",
           "GEMINI_API_KEY": KEY, AG.SEARCH_LOG_ENV: str(tmp_path / "s.jsonl"),
           "OPENAI_BASE_URL": "http://127.0.0.1:9/v1", "UNRELATED_SECRET_TOKEN": "zzzzzzzzzzzz"}
    AG._run_cli(["generate", "--model", "m"], env, 60, model="m")
    cmd = seen["cmd"]
    joined = " ".join(cmd)
    assert cmd[:3] == ["docker", "run", "--rm"] and AG.bfcl_image() in cmd
    assert KEY not in joined and "sk-endpoint" not in joined and "zzzz" not in joined
    for name in ("GEMINI_API_KEY", "OPENAI_API_KEY", AG.SEARCH_LOG_ENV, "BFCL_PROJECT_ROOT"):
        assert ["-e", name] == cmd[cmd.index(name) - 1:cmd.index(name) + 1]
    assert "UNRELATED_SECRET_TOKEN" not in cmd
    # the container holds placeholders, never a key
    held = json.dumps({n: seen["env"].get(n) for n in cmd if n in seen["env"]})
    assert KEY not in held and "sk-endpoint" not in held
    assert seen["env"]["OPENAI_API_KEY"] == AG.ENDPOINT_KEY_PLACEHOLDER
    assert f"{tmp_path}:{tmp_path}:rw" in cmd and "--network" in cmd
    program = cmd[-1]
    assert "gbench.runners.eval_suites" in program and "_register_model('m')" in program


def test_the_launcher_masks_every_credential_in_the_output():
    env = dict(os.environ, GEMINI_API_KEY=KEY)
    rc, out = AG._launch(["sh", "-c", 'echo "search api_key=$GEMINI_API_KEY done"'], env, 30,
                         "unused-name")
    assert rc == 0 and KEY not in out and f"api_key={base.REDACTED} done" in out


def test_image_jobs_travel_in_a_file_and_come_back_masked(monkeypatch):
    """run_in_image with the container simulated: _image_main runs the job from the mounted
    exchange directory, as it does in the image."""
    seen = {}

    def launch(cmd, env, timeout, name, stream=False):
        io_dir = next(m.split(":")[0] for m in cmd if m.endswith(f":{AG._IMAGE_IO_DIR}:rw"))
        seen["argv"] = " ".join(cmd)
        seen["project_root"] = env["BFCL_PROJECT_ROOT"]
        monkeypatch.setenv("GEMINI_API_KEY", KEY)
        return AG._image_main(io_dir), ""
    monkeypatch.setattr(AG, "_launch", launch)
    got = AG.run_in_image("t", "bfcl_v4_agentic", "search_log_summary",
                          {"rows": [{"backend": "error", "error": f"x {KEY} y"}]})
    assert got["failures"] == {f"error: x {base.REDACTED} y": 1}   # the image masks its result too
    assert KEY not in seen["argv"] and "search_log_summary" not in seen["argv"]
    assert seen["project_root"] == AG._IMAGE_SCRATCH_ROOT
    with pytest.raises(RuntimeError, match=r"TypeError"):
        AG.run_in_image("t", "bfcl_v4_agentic", "search_log_summary", {"text": "x"})


def test_a_missing_or_foreign_image_is_named_with_the_build_command(monkeypatch):
    monkeypatch.setattr(AG.shutil, "which", lambda name: None)
    assert "docker CLI is not on PATH" in AG.image_problem()
    assert AG.image_build_command() in AG.image_problem()
    monkeypatch.setattr(AG.shutil, "which", lambda name: "/usr/bin/docker")

    class Done:
        def __init__(self, rc, out=""):
            self.returncode, self.stdout = rc, out
    labels = json.dumps({"gbench.bfcl.release": "2025.12.17.1",
                         "gbench.bfcl.commit": AG.BFCL_EVAL_COMMIT})
    monkeypatch.setattr(AG.subprocess, "run", lambda cmd, **k: Done(0, labels if "inspect" in cmd else ""))
    why = AG.image_problem()
    assert "gbench.bfcl.release=2025.12.17.1 (wanted 2026.3.23)" in why
    monkeypatch.setattr(AG.subprocess, "run", lambda cmd, **k: Done(1 if "inspect" in cmd else 0))
    assert "is not built" in AG.image_problem()


def test_the_suites_stop_before_generating_without_the_image(monkeypatch):
    monkeypatch.setattr(AG, "image_problem", lambda: "the BFCL harness image 'x' is not built.")
    for call in (lambda: bfcl._harness_probe("bfcl", [], True),
                 lambda: bfcl._harness_probe("bfcl", ["multi_turn_base"], False)):
        with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*is not built"):
            call()


def test_gbench_never_imports_bfcl_eval_and_writes_nothing_next_to_it(tmp_path):
    """The parent process used to import bfcl_eval's eval_config (creating result/, score/ and
    .file_locks/ beside the package); now only the image imports bfcl_eval."""
    code = ("import sys, json, os\n"
            "from unittest import mock\n"
            "from gbench.runners.eval_suites import bfcl, bfcl_live, bfcl_v4_agentic as AG\n"
            "AG.check_bfcl_prerequisites = lambda url: (True, '', list(AG.AGENTIC_CATEGORIES))\n"
            "AG._run_cli = lambda *a, **k: (0, '')\n"
            "AG._parse_scores = lambda root, m, cats: {c: {'correct': 1, 'total': 2} for c in cats}\n"
            "AG.run_bfcl_v4_agentic('m', 'http://x/v1', 1, eval_categories='memory_kv')\n"
            "print(json.dumps(sorted(m for m in sys.modules if m.startswith('bfcl_eval'))))\n")
    import importlib.util
    spec = importlib.util.find_spec("bfcl_eval")
    pkg_parent = Path(spec.origin).resolve().parents[1] if spec else None
    before = {p.name for p in pkg_parent.iterdir()} if pkg_parent else set()
    env = {k: v for k, v in os.environ.items() if k != "BFCL_PROJECT_ROOT"}
    env["GBENCH_BFCL_PROJECT_ROOT"] = str(tmp_path / "bfcl")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []
    if pkg_parent:
        assert {p.name for p in pkg_parent.iterdir()} - before == set()


# --------------------------------------------------------------------------- #
# Single-turn: the reply becomes the model result BFCL's handler records
# --------------------------------------------------------------------------- #

def test_the_model_result_is_what_the_fc_handler_records():
    calls = [{"id": "a", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}},
             {"id": "b", "type": "function", "function": {"name": "g", "arguments": "{bad"}}]
    assert bfcl.upstream_model_result("ignored", calls) == [{"f": '{"x": 1}'}, {"g": "{bad"}]
    assert bfcl.upstream_model_result("some prose", None) == "some prose"
    assert bfcl.upstream_model_result("", []) == ""


def test_the_batch_scorer_hands_bfcl_the_entry_and_the_model_result():
    seen = []

    def checker(items):
        seen.extend(items)
        return [{"valid": i == 0, "error_type": None if i == 0 else "ast_decoder:decoder_failed",
                 "error": [] if i == 0 else ["x"]} for i in range(len(items))]
    doc = [{"name": "f", "description": "F.", "parameters": {"type": "dict", "properties": {}}}]
    traces = [
        {"gold_answer": {"id": "simple_python_0", "category": "simple_python",
                         "ground_truth": [{"f": {}}]}, "response_text": "",
         "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
         "status": "PENDING_JUDGE", "health": "ok"},
        {"gold_answer": {"id": "simple_python_1", "category": "simple_python",
                         "ground_truth": [{"f": {}}]}, "response_text": "f()", "tool_calls": None,
         "status": "PENDING_JUDGE", "health": "ok"},
        {"gold_answer": {"id": "simple_python_2", "category": "simple_python", "ground_truth": []},
         "response_text": None, "tool_calls": None, "status": "FAILED"}]
    asyncio.run(bfcl.bfcl_batch_scorer({f"simple_python_{i}": doc for i in range(3)},
                                       checker=checker)(traces))
    assert [it["model_result"] for it in seen] == [[{"f": "{}"}], "f()"]
    assert seen[0]["function"] == doc and seen[0]["ground_truth"] == [{"f": {}}]
    assert [t.get("is_correct") for t in traces] == [True, False, None]
    assert traces[1]["bfcl_error_type"] == "ast_decoder:decoder_failed"


def test_bfcl_requests_carry_no_max_tokens():
    rec = BFCL_OFFICIAL_SAMPLING.record(0.001, "official")
    assert BFCL_OFFICIAL_SAMPLING.send_max_tokens is False and rec["send_max_tokens"] is False
    base.set_run_knobs(max_output_tokens=4096)
    try:
        assert AG.max_tokens_note().endswith("--max-output-tokens 4096 not applied")
    finally:
        base._RUN_KNOBS.pop("max_output_tokens", None)


# --------------------------------------------------------------------------- #
# ACEBench: BinOp arguments are upstream's eval
# --------------------------------------------------------------------------- #

# (argument, what upstream's CommonAgent.decode_function_list / utils.ast_parse give for it at
# ACEBench@56dd66cf, run on the upstream files)
_UPSTREAM_BINOPS = [
    ("'code ' + str(42)", "code 42"), ("[1] + [2]", [1, 2]), ("(1, 2) * 2", (1, 2, 1, 2)),
    ("1 << 3", 8), ("6 | 1", 7), ("len('abc') + 1", 4), ("sorted([3, 1]) + [0]", [1, 3, 0]),
    ("'%s-%d' % ('a', 5)", "a-5"), ("{1, 2} | {3}", {1, 2, 3}), ("2 ** 0.5", 2 ** 0.5),
]


@pytest.mark.parametrize("arg,value", _UPSTREAM_BINOPS)
def test_binop_arguments_evaluate_as_upstreams_eval(arg, value):
    assert AB.ast_parse(f"[F(x={arg})]") == [{"F": {"x": value}}]
    call = AB.decode_function_list(f"[add_reminder(description={arg})]", "fail")
    assert call == [f"add_reminder(description={value!r})"]


def test_binop_globals_and_errors_follow_the_upstream_module():
    # utils.py imports json, common_agent.py imports re: each role sees its own module's names
    assert AB.ast_parse("[F(x=json.dumps({'a': 1}) + '')]") == [{"F": {"x": '{"a": 1}'}}]
    with pytest.raises(NameError, match="name 'json' is not defined"):
        AB.decode_function_list("[F(x=json.dumps({}) + '')]", "fail")
    assert AB.decode_function_list("[F(x=__name__ + '')]", "skip") == [
        "F(x='model_inference.multi_turn.execution_role')"]
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        AB.ast_parse("[F(x=1 / 0 + 1)]")
    with pytest.raises(NameError, match="name 'nope' is not defined"):
        AB.ast_parse("[F(x=nope + 1)]")


def test_what_the_isolated_eval_cannot_reproduce_is_refused_and_counted():
    sink = []
    token = AB._REFUSALS.set(sink)
    try:
        for expr in ("self.functions + []",      # a local of upstream's method
                     "value + 1",                # ditto
                     "__file__ + ''",            # upstream's checkout path
                     "[print] + []",             # a value that is not a literal
                     "10**10**10 + 0"):          # too large to compute
            with pytest.raises(AB._GuardRefusal):
                AB.decode_function_list(f"[F(x={expr})]", "fail")
    finally:
        AB._REFUSALS.reset(token)
    assert sink == ["self.functions + []", "value + 1", "__file__ + ''", "[print] + []",
                    "10 ** 10 ** 10 + 0"]
    assert AB.normal_row_valid("[Api(q=value + 1)]", [], {"Api": {"q": "x"}}, "",
                               "normal_atom_bool") is False


def test_normal_rows_with_refused_binops_break_comparability(monkeypatch):
    monkeypatch.setattr(AB, "_load_acebench_samples", lambda **k: [])
    monkeypatch.setattr(AB, "_load_agent_items", lambda language="en": [])
    gold = json.dumps({"id": "normal_atom_bool_7", "sub": "normal_atom_bool", "language": "en",
                       "ground_truth": {"Api": {"q": "x"}}, "functions": [], "question": ""})

    def fake_run(**kw):
        # the rows are scored after generation by the suite's batch scorer (off the event loop)
        trace = {"gold_answer": gold, "response_text": "[Api(q=value + 1)]",
                 "status": "PENDING_JUDGE", "health": "ok"}
        asyncio.run(kw["async_eval_fn"]([trace]))
        ok = trace["is_correct"]
        return {"sample_traces": [], "accuracy": 0.0, "temperature": 0.0,
                "temperature_source": "official", "category_accuracy": {}, "attempts": None,
                "official_sampling": AB.ACEBENCH_SAMPLING.record(0.0, "official"),
                "total_questions": 1, "failed_requests": 0, "row_ok": ok}
    monkeypatch.setattr(AB, "run_eval_suite", fake_run)
    res = AB.run_acebench("m", "http://x", 1)
    assert res["acebench_report"]["single_turn_guard_refusals"] == 1
    assert res["acebench_report"]["single_turn_guard_refused"] == [
        {"id": "normal_atom_bool_7", "expression": "value + 1"}]
    assert res["leaderboard_comparable"] is False
    assert "1 BinOp arguments in Normal replies" in res["leaderboard_comparable_reason"]


def test_acebench_needs_the_isolated_evaluator_before_generating(monkeypatch):
    # The evaluator's container is its own (GBENCH_SANDBOX no longer gates it): no bubblewrap
    # is a hard error.
    monkeypatch.setattr(AB.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*acebench.*BinOp.*bubblewrap"):
        AB.check_binop_evaluator()


def test_the_isolated_eval_sees_no_credential_and_no_home(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", KEY)
    assert AB._isolated_eval("__import__('os').environ.get('SERPAPI_API_KEY', '') + ''",
                             "normal") == ""
    home = os.path.expanduser("~")
    if (os.path.isdir(home) and not (home + os.sep).startswith("/usr/")
            and not any(d == home or d.startswith(home + os.sep)
                        for d in AB._interpreter_dirs())):
        # the container's filesystem has no home directory at all
        with pytest.raises(FileNotFoundError):
            AB._isolated_eval(f"__import__('os').listdir({home!r}) + []", "normal")


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #

def test_docs_describe_the_image_and_the_fixes():
    bfcl_doc = (DOCS / "bfcl.md").read_text(encoding="utf-8")
    agentic_doc = (DOCS / "bfcl_v4_agentic.md").read_text(encoding="utf-8")
    live_doc = (DOCS / "bfcl_live.md").read_text(encoding="utf-8")
    for doc in (bfcl_doc, agentic_doc, live_doc):
        assert (AG.image_build_command(AG.BFCL_IMAGE_DEFAULT) in doc and "gemma4-env" not in doc
                and doc.isascii())
    assert "pip install bfcl-eval" not in bfcl_doc + agentic_doc + live_doc
    assert "no `max_tokens`" in bfcl_doc
    assert AG.SEARCH_FAILED_TEXT in " ".join(agentic_doc.split())
    ace = (DOCS / "acebench.md").read_text(encoding="utf-8")
    assert "isolated process" in ace and "as undefined names" not in ace
    gor = (DOCS / "gorilla_apibench.md").read_text(encoding="utf-8")
    assert "get_llm_responses_retriever.py" in gor
    teval = (DOCS / "t_eval.md").read_text(encoding="utf-8")
    bullets = teval[teval.index("- **Reason"):teval.index("- **Overall")]
    assert "\n\n" not in bullets.strip()
    dockerfile = (REPO / "gbench" / "docker" / "bfcl.Dockerfile").read_text(encoding="utf-8")
    assert AG.BFCL_EVAL_RELEASE in dockerfile and AG.BFCL_EVAL_COMMIT in dockerfile
    assert "3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a" in dockerfile
    constraints = (REPO / "gbench" / "docker" / "bfcl.constraints.txt").read_text(encoding="utf-8")
    for pin in ("numpy==1.26.4", "filelock==3.20.0", "tree-sitter==0.21.3", "faiss-cpu==1.11.0"):
        assert pin in constraints.splitlines()


# --------------------------------------------------------------------------- #
# the real image (GBENCH_TEST_BFCL_IMAGE=1)
# --------------------------------------------------------------------------- #

def _server(handler_cls):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


def _chat_handler(answer, log):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, data):
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"object": "list", "data": [{"id": "stub", "object": "model"}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            log.append(req)
            msg = answer(req)
            calls = msg.get("tool_calls")
            self._send({"id": "x", "object": "chat.completion", "created": 0, "model": req["model"],
                        "choices": [{"index": 0, "message": msg,
                                     "finish_reason": "tool_calls" if calls else "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                                  "total_tokens": 15}})
    return Handler


@image
def test_image_probe_holds_the_pinned_release():
    assert AG.image_problem() is None
    found = AG.run_in_image("t", "bfcl", "image_probe",
                            {"multi_turn_categories": bfcl.MULTI_TURN_CATEGORIES,
                             "single_turn": True}, timeout=900)
    assert found["release"] == "2026.3.23" and len(found["multi_turn_index"]) == 800
    assert found["single_turn_mismatch"] is None and found["multi_turn_mismatch"] is None
    agentic = AG.run_in_image("t", "bfcl_v4_agentic", "image_probe", {}, timeout=900)
    assert agentic["agentic_mismatch"] is None
    assert set(agentic["agentic_categories"]) == set(AG.AGENTIC_CATEGORIES)


@image
def test_image_single_turn_checker_scores_as_eval_runner():
    doc = [{"name": "math.add", "description": "Add.", "parameters": {
        "type": "dict", "required": ["a"], "properties": {"a": {"type": "integer", "description": "A"}}}}]
    gt = [{"math.add": {"a": [1]}}]
    items = [{"id": "simple_python_0", "category": "simple_python", "function": doc,
              "ground_truth": gt, "model_result": [{"math_add": '{"a": 1}'}]},
             {"id": "simple_python_1", "category": "simple_python", "function": doc,
              "ground_truth": gt, "model_result": [{"math_add": '{"a": "1"}'}]},
             {"id": "simple_python_2", "category": "simple_python", "function": doc,
              "ground_truth": gt, "model_result": "math.add(a=1)"},
             {"id": "irrelevance_0", "category": "irrelevance", "function": doc,
              "ground_truth": [], "model_result": "I can't."},
             {"id": "live_relevance_0", "category": "live_relevance", "function": doc,
              "ground_truth": [], "model_result": [{"math_add": '{"a": 1}'}]}]
    got = bfcl.check_single_turn(items)
    assert [v["valid"] for v in got] == [True, False, False, True, True]
    assert got[1]["error_type"] == "type_error:simple"
    assert got[2]["error_type"] == "ast_decoder:decoder_failed"


@image
def test_image_multi_turn_replays_the_ground_truth(tmp_path):
    items = AG.run_in_image("t", "bfcl", "load_multi_turn_entries",
                            {"categories": ["multi_turn_base", "multi_turn_miss_func"]}, timeout=900)
    # entries whose ground-truth calls pass every argument by keyword
    def keyword_only(it):
        for turn in it["ground_truth"]:
            for c in turn:
                if ast.parse(c, mode="eval").body.args:
                    return False
        return True
    chosen = [it for it in items if keyword_only(it)][:6]
    table = {}
    held_out = "I have updated some more functions you can choose from. What about now?"
    for it in chosen:
        convo = []
        for t, turn in enumerate(it["entry"]["question"]):
            if str(t) in it["entry"].get("missed_function", {}):
                turn = [{"role": "user", "content": held_out}]
            convo += turn
            key = json.dumps([(m.get("role"), m.get("content")) for m in convo
                              if m.get("role") in ("user", "system")])
            table[key] = it["ground_truth"][t]

    def answer(req):
        msgs = req["messages"]
        key = json.dumps([(m.get("role"), m.get("content")) for m in msgs
                          if m.get("role") in ("user", "system")])
        calls = [] if msgs[-1].get("role") == "tool" else table.get(key, [])
        if not calls:
            return {"role": "assistant", "content": "Done."}
        out = []
        for i, c in enumerate(calls):
            node = ast.parse(c, mode="eval").body
            out.append({"id": f"c{i}", "type": "function", "function": {
                "name": node.func.id,
                "arguments": json.dumps({k.arg: ast.literal_eval(k.value) for k in node.keywords})}})
        return {"role": "assistant", "content": None, "tool_calls": out}
    log = []
    srv, url = _server(_chat_handler(answer, log))
    try:
        res = bfcl.run_multi_turn_in_image("stub", url, 4,
                                           [{"id": it["id"], "category": it["category"]}
                                            for it in chosen], temperature=0.001, thinking=False)
    finally:
        srv.shutdown()
    assert [t["is_correct"] for t in res["traces"]] == [True] * len(chosen), [
        (t["question_id"], t.get("bfcl_error_type")) for t in res["traces"]]
    assert log and all("max_tokens" not in r and r["temperature"] == 0.001 for r in log)
    assert not res["guard_refusals"]


@image
def test_image_agentic_memory_runs_end_to_end(tmp_path, monkeypatch):
    log = []
    srv, url = _server(_chat_handler(lambda req: {"role": "assistant", "content": "I do not know."},
                                     log))
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(tmp_path / "bfcl"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    try:
        res = AG.run_bfcl_v4_agentic("stub", url, 2, eval_categories="memory_kv,memory_vector",
                                     limit=2)
    finally:
        srv.shutdown()
    assert set(res["category_accuracy"]) == {"memory_kv", "memory_vector"}
    assert res["total_questions"] == 2 and res["bfcl_report"]["eval_limit"] == 2
    assert log and all("max_tokens" not in r and r["temperature"] == 0.001 for r in log)
    assert list((tmp_path / "bfcl").glob("score_limit2/stub/**/BFCL_v4_memory_kv_score.json"))


@image
def test_image_transport_error_reaches_the_model_without_the_key(tmp_path):
    """The pinned release's WebSearchAPI.search_engine_query served by Gemini grounding, with the
    grounding endpoint unreachable (the container has no network): the error the model would get
    carries no key."""
    program = (AG._package_preamble()
               + "import json, os\n"
               "from gbench.runners.eval_suites.bfcl_v4_agentic import "
               "_serve_web_search_with_gemini\n"
               "_serve_web_search_with_gemini(os.environ['GBENCH_BFCL_SEARCH_LOG'])\n"
               "from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search "
               "import WebSearchAPI\n"
               "print('RESULT', json.dumps(WebSearchAPI().search_engine_query('tea')))\n")
    root = str(tmp_path)
    env = dict(os.environ, GEMINI_API_KEY=KEY, BFCL_PROJECT_ROOT=root,
               GBENCH_BFCL_SEARCH_LOG=str(tmp_path / "search.jsonl"), GBENCH_SEARCH_ATTEMPTS="1",
               GBENCH_SEARCH_CASCADE_ROUNDS="1", GBENCH_SEARCH_MODELS="gemini-3.7-flash")
    name = AG._container_name("search")
    cmd = AG._docker_run_command(name, ["GEMINI_API_KEY", "BFCL_PROJECT_ROOT",
                                        "GBENCH_BFCL_SEARCH_LOG", "GBENCH_SEARCH_ATTEMPTS",
                                        "GBENCH_SEARCH_CASCADE_ROUNDS", "GBENCH_SEARCH_MODELS"],
                                 [(root, root, "rw")], program)
    cmd[cmd.index("--network") + 1] = "none"
    # the output is not masked here, so a leaked key would show
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout.split("RESULT", 1)[1])
    assert result == {"error": AG.SEARCH_FAILED_TEXT}
    assert KEY not in proc.stdout + proc.stderr
    log = tmp_path / "search.jsonl"
    assert KEY not in log.read_text()
    assert AG.search_log_summary(AG._read_search_log(log))["failed"] == 1
