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

"""BFCL and the function-calling suites: per-attempt backend state, BFCL's temperature in every
part of BFCL Overall, the agentic harness (FC decoding, harness pin, Gemini key check), the
official sampling of every function-calling suite, NexusRaven's published domain numbers,
T-Eval's post-processing and Chinese setting, and API-Bank's message layout.

Everything runs against local stubs (a local OpenAI-compatible stub server on a free port for
the multi-turn loop, a local HTTP server for the Gemini API's models endpoint); no model server
and no network."""

import ast
import collections
import importlib
import importlib.metadata
import inspect
import json
import logging
import re
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import pytest

from gbench.runners.eval_suites import base, bfcl
from gbench.runners.eval_suites import bfcl_v4_agentic as AG
from gbench.runners.eval_suites.fc_common import BFCL_OFFICIAL_SAMPLING

#: The real key check, kept before any fixture replaces it.
_REAL_SEARCH_BACKEND = AG.search_backend


@pytest.fixture(autouse=True)
def _isolated(tmp_path_factory, monkeypatch):
    # bfcl_eval creates result/score/lock directories under BFCL_PROJECT_ROOT on first import.
    if "bfcl_eval.constants.eval_config" not in sys.modules:
        monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path_factory.mktemp("bfcl_project")))
    monkeypatch.setenv("TQDM_DISABLE", "1")
    for var in ("GBENCH_BFCL_TEMPERATURE", "GBENCH_BFCL_V4_AGENTIC_TEMPERATURE",
                "GBENCH_BFCL_LIVE_TEMPERATURE", "GBENCH_SHARD", "GEMINI_API_KEY",
                "GEMINI_API_KEYS"):
        monkeypatch.delenv(var, raising=False)
    saved = dict(base._RUN_KNOBS)
    yield
    base._RUN_KNOBS.clear()
    base._RUN_KNOBS.update(saved)


# --------------------------------------------------------------------------- #
# multi-turn: one attempt's cleanup never touches another attempt's backends
# --------------------------------------------------------------------------- #

def _instance_names(key, entry, classes=("GorillaFileSystem", "TwitterAPI")):
    names = []
    for suffix in bfcl._REPLAY_SUFFIXES:
        for cls in classes:
            names.append(re.sub(r"[-./:]", "_", f"{key}{suffix}_{entry}_{cls}_instance"))
    return names


def test_cleanup_removes_only_its_own_attempt_and_entry():
    import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils as mtu
    ns = vars(mtu)
    mine = _instance_names("gbench_bfcl_r_a1", "multi_turn_base_1")
    others = (_instance_names("gbench_bfcl_r_a10", "multi_turn_base_1")
              + _instance_names("gbench_bfcl_r_a11", "multi_turn_base_1")
              + _instance_names("gbench_bfcl_r_a1", "multi_turn_base_10")
              + _instance_names("gbench_bfcl_r_a1", "multi_turn_miss_func_1"))
    try:
        for n in mine + others:
            ns[n] = object()
        bfcl._drop_instances("gbench_bfcl_r_a1", "multi_turn_base_1")
        assert not [n for n in mine if n in ns]
        assert all(n in ns for n in others)
    finally:
        for n in mine + others:
            ns.pop(n, None)


def _backend_params():
    from bfcl_eval.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING
    params = {}
    for cls, mod in CLASS_FILE_PATH_MAPPING.items():
        if cls.startswith(("Memory", "WebSearch")):
            continue
        c = getattr(importlib.import_module(mod), cls)
        for name, fn in inspect.getmembers(c, inspect.isfunction):
            params[name] = [p for p in inspect.signature(fn).parameters if p != "self"]
    return params


def _truth_server(items):
    """A local OpenAI chat server answering every multi-turn turn with its ground-truth calls."""
    from bfcl_eval.constants.default_prompts import DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC
    params, table = _backend_params(), {}

    def key(msgs):
        return json.dumps([(m.get("role"), m.get("content")) for m in msgs
                           if m.get("role") in ("user", "system")])
    for it in items:
        convo = []
        for t, turn in enumerate(it["entry"]["question"]):
            if str(t) in it["entry"].get("missed_function", {}):
                turn = [{"role": "user", "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC}]
            convo += turn
            table[key(convo)] = it["ground_truth"][t]

    def tool_call(call, i):
        node = ast.parse(call, mode="eval").body
        args = {params[node.func.id][k]: ast.literal_eval(a) for k, a in enumerate(node.args)}
        args.update({k.arg: ast.literal_eval(k.value) for k in node.keywords})
        return {"id": f"c{i}", "type": "function",
                "function": {"name": node.func.id, "arguments": json.dumps(args)}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            msgs = body["messages"]
            calls = [] if msgs[-1].get("role") == "tool" else table.get(key(msgs), [])
            msg = ({"role": "assistant", "content": None,
                    "tool_calls": [tool_call(c, i) for i, c in enumerate(calls)]} if calls
                   else {"role": "assistant", "content": "Done."})
            data = json.dumps({"id": "x", "object": "chat.completion", "created": 0,
                               "model": body["model"],
                               "choices": [{"index": 0, "message": msg,
                                            "finish_reason": "tool_calls" if calls else "stop"}],
                               "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                                         "total_tokens": 15}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


def test_twelve_concurrent_attempts_keep_their_backend_state(monkeypatch):
    """With attempts >= 11 every attempt of an entry runs at once; no backend instance may
    disappear (and be rebuilt from the initial state) while its entry is still running."""
    import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker as mtc
    import bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils as mtu
    import bfcl_eval.model_handler.base_handler as bh
    orig = mtu.execute_multi_turn_func_call
    seen, recreated, lock = set(), collections.Counter(), threading.Lock()

    def counting(func_call_list, initial_config, involved_classes, model_name, test_entry_id,
                 long_context=False, is_evaL_run=False):
        mn = model_name + ("_eval" if is_evaL_run else "")
        names = [re.sub(r"[-./:]", "_", f"{mn}_{test_entry_id}_{c}_instance")
                 for c in involved_classes]
        with lock:
            for n in names:
                if n in seen and n not in vars(mtu):
                    recreated[n] += 1
                seen.add(n)
        time.sleep(0.005)
        return orig(func_call_list, initial_config, involved_classes, model_name, test_entry_id,
                    long_context, is_evaL_run)

    monkeypatch.setattr(bh, "execute_multi_turn_func_call", counting)
    monkeypatch.setattr(mtc, "execute_multi_turn_func_call", counting)
    items = bfcl.load_multi_turn_entries(["multi_turn_base"])[:4]
    srv, url = _truth_server(items)
    try:
        res = bfcl.run_multi_turn("stub", url, 48, items, temperature=0.001,
                                  thinking=False, attempts=12)
    finally:
        srv.shutdown()
    assert len(res["traces"]) == 48
    assert not recreated, dict(recreated)
    assert all(t["is_correct"] for t in res["traces"]), [
        (t["question_id"], t["attempt_idx"], t.get("bfcl_error_type")) for t in res["traces"]
        if not t["is_correct"]]
    assert not [k for k in vars(mtu) if k.startswith("gbench_bfcl_")]


def test_guard_refusal_names_the_call_as_upstream_does():
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils as mtu
    it = bfcl.load_multi_turn_entries(["multi_turn_base"])[0]
    key = f"guardname_{time.time_ns()}"
    calls = ["cd(folder=__import__('os').getcwd())", "open(file='/tmp/x', mode='w')"]
    try:
        with bfcl._harness_patches():
            out, _ = mtu.execute_multi_turn_func_call(
                calls, it["eval_entry"]["initial_config"], it["eval_entry"]["involved_classes"],
                key, it["id"])
    finally:
        bfcl._drop_instances(key, it["id"])
    assert out == ["Error during execution: Function call cd is not allowed.",
                   "Error during execution: Function call open is not allowed."]
    assert key not in " ".join(out)


# --------------------------------------------------------------------------- #
# BFCL's temperature in every part of BFCL Overall
# --------------------------------------------------------------------------- #

def _single_result(**_kw):
    ca = {c: {"correct": 3, "total": 4, "accuracy": 75.0} for c in bfcl.SINGLE_TURN_CATEGORIES}
    return {"benchmark_type": "eval", "eval_name": "bfcl", "model_name": "m", "thinking": False,
            "total_questions": 52, "correct_answers": 39, "failed_requests": 0,
            "empty_responses": 0, "truncated_responses": 0, "over_context_prompts": 0,
            "timed_out_requests": 0, "judge_outages": 0, "response_health": {"ok": 52},
            "category_accuracy": ca, "accuracy": 75.0, "status": "success",
            "sample_traces": []}


@pytest.fixture
def parts(monkeypatch, tmp_path):
    """run_bfcl with the single-turn and multi-turn parts stubbed; the agentic part is the real
    run_bfcl_v4_agentic with the harness CLI stubbed, so its generate command can be read."""
    seen = {"cli": []}
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(tmp_path / "bfcl"))
    monkeypatch.setattr(bfcl, "_load_bfcl_samples", lambda cats: ([], {}))
    # the harness image has its own tests: the entries come from the bfcl-eval installed here
    monkeypatch.setattr(bfcl, "_harness_probe", lambda name, multi, single: (
        bfcl.BFCL_EVAL_RELEASE, [{"id": it["id"], "category": it["category"]}
                                 for it in (bfcl.load_multi_turn_entries(multi) if multi else [])]))

    def single(**kw):
        seen["single"] = kw
        return _single_result()
    monkeypatch.setattr(bfcl, "run_eval_suite", single)

    def multi(model, url, conc, items, **kw):
        seen["multi"] = kw
        return {"traces": [{"sample_idx": i, "source_sample_idx": i, "attempt_idx": 0,
                            "category": it["category"], "question_id": it["id"],
                            "is_correct": True, "status": "OK", "health": "ok", "steps": 1}
                           for i, it in enumerate(items)], "guard_refusals": []}
    monkeypatch.setattr(bfcl, "run_multi_turn_in_image", multi)
    # the multi-turn part is stubbed, so the network checks before its container
    # (tests/test_guard_code_images.py) do not apply
    monkeypatch.setattr(bfcl, "endpoint_network", lambda base_url: base_url)
    monkeypatch.setattr(AG, "check_bfcl_prerequisites",
                        lambda url: (True, "", list(AG.AGENTIC_CATEGORIES)))
    monkeypatch.setattr(AG, "search_backend", lambda: (AG.SEARCH_BACKEND, ""))
    monkeypatch.setattr(AG, "_register_model", lambda m: m)

    def cli(args, env, timeout, model="", backend=None):
        seen["cli"].append(list(args))
        return 0, ""
    monkeypatch.setattr(AG, "_run_cli", cli)
    monkeypatch.setattr(AG, "_clear_stale_scores", lambda *a: 0)
    monkeypatch.setattr(AG, "_parse_scores", lambda root, m, cats: {
        c: {"correct": 1, "total": 2, "accuracy": 50.0} for c in cats})
    monkeypatch.setattr(AG, "_parse_score_traces", lambda *a: [])
    monkeypatch.setattr(AG, "_score_error_types", lambda *a: {})
    return seen


def _generate_temperature(seen):
    gen = next(c for c in seen["cli"] if c[0] == "generate")
    return float(gen[gen.index("--temperature") + 1])


def test_every_part_generates_at_zero_without_thinking(parts):
    # The maintainer's rule: a no-think run samples every part at 0.0, not BFCL's 0.001, which is
    # only recorded.
    res = bfcl.run_bfcl("m", "http://x/v1", 4, enable_thinking=False)
    assert res["temperature"] == 0.0 and res["temperature_source"] == "default"
    assert res["official_sampling"]["temperature"] == 0.001
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert parts["single"]["official_sampling"] is BFCL_OFFICIAL_SAMPLING
    assert parts["multi"]["temperature"] == 0.0
    assert _generate_temperature(parts) == 0.0
    assert res["agentic_report"]["temperature"] == 0.0
    assert res["leaderboard_comparable"] is False
    assert f"agentic: {AG.SEARCH_SUBSTITUTION_REASON}" in res["leaderboard_comparable_reason"]
    assert "temperature 0.0 (BFCL generates at 0.001)" in res["leaderboard_comparable_reason"]
    assert "the agentic part generated at" not in res["leaderboard_comparable_reason"]


def test_a_thinking_run_generates_every_part_at_one(parts):
    # The maintainer's rule: with --thinking every request samples at 1.0, BFCL's 0.001 is not
    # applied, and the result says so.
    res = bfcl.run_bfcl("m", "http://x/v1", 4, enable_thinking=True)
    assert (res["temperature"], res["temperature_source"]) == (1.0, "thinking")
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert parts["multi"]["temperature"] == 1.0
    assert _generate_temperature(parts) == 1.0
    assert res["agentic_report"]["temperature"] == 1.0
    assert res["leaderboard_comparable"] is False
    assert "temperature 1.0 (BFCL generates at 0.001)" in res["leaderboard_comparable_reason"]
    assert "the agentic part generated at" not in res["leaderboard_comparable_reason"]


def test_the_agentic_suites_own_knob_cannot_move_the_agentic_part(parts, monkeypatch, caplog):
    monkeypatch.setenv("GBENCH_BFCL_V4_AGENTIC_TEMPERATURE", "1.0")
    with caplog.at_level(logging.WARNING):
        res = bfcl.run_bfcl("m", "http://x/v1", 4)
    assert _generate_temperature(parts) == 0.0
    assert "the agentic part generated at" not in res["leaderboard_comparable_reason"]
    assert any("GBENCH_BFCL_V4_AGENTIC_TEMPERATURE" in r.getMessage() for r in caplog.records)


def test_an_operator_temperature_reaches_all_parts_and_is_flagged(parts):
    base.set_run_knobs(temperature=0.7)
    res = bfcl.run_bfcl("m", "http://x/v1", 4)
    assert res["temperature"] == 0.7 and res["temperature_source"] == "cli:--temperature"
    assert parts["multi"]["temperature"] == 0.7 and _generate_temperature(parts) == 0.7
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert res["leaderboard_comparable"] is False
    assert "temperature 0.7 (BFCL generates at 0.001)" in res["leaderboard_comparable_reason"]


def test_an_agentic_part_at_another_temperature_is_not_comparable(parts, monkeypatch):
    real = AG.run_bfcl_v4_agentic

    def drifted(*a, **kw):
        out = real(*a, **{**kw, "temperature": 1.0})
        return out
    monkeypatch.setattr(AG, "run_bfcl_v4_agentic", drifted)
    res = bfcl.run_bfcl("m", "http://x/v1", 4)
    assert res["leaderboard_comparable"] is False
    assert "the agentic part generated at temperature 1.0" in res["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# bfcl_v4_agentic on its own
# --------------------------------------------------------------------------- #

@pytest.fixture
def agentic(parts):
    return parts


def test_standalone_agentic_generates_at_zero_without_thinking(agentic):
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1, enable_thinking=False)
    assert (res["temperature"], res["temperature_source"]) == (0.0, "default")
    # the value bfcl generate --temperature gets, which the harness' handler sends on every request
    assert _generate_temperature(agentic) == 0.0
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"] == (
        f"{AG.SEARCH_SUBSTITUTION_REASON}; temperature 0.0 set by default (BFCL generates at "
        "0.001)")


@pytest.mark.parametrize("thinking,expected", [(False, 0.0), (True, 1.0)])
def test_the_harness_handler_sends_the_generate_temperature(agentic, thinking, expected):
    """The value handed to `bfcl generate --temperature` is what the harness' handler puts on every
    request (OpenAICompletionsHandler._query_FC), a 0.0 included."""
    AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1, enable_thinking=thinking)
    temp = _generate_temperature(agentic)
    assert temp == expected
    handler_cls = AG._patched_openai_handler()
    h = handler_cls.__new__(handler_cls)
    h.model_name, h.temperature = "m", temp
    h.generate_with_backoff = lambda **kw: kw
    sent = h._query_FC({"message": [{"role": "user", "content": "q"}], "tools": []})
    assert sent["temperature"] == expected


def test_standalone_agentic_thinking_run_generates_at_one(agentic):
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1, enable_thinking=True)
    assert (res["temperature"], res["temperature_source"]) == (1.0, "thinking")
    assert _generate_temperature(agentic) == 1.0
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert res["leaderboard_comparable"] is False
    assert "temperature 1.0 set by thinking (BFCL generates at 0.001)" in (
        res["leaderboard_comparable_reason"])


def test_standalone_agentic_temperature_override_is_flagged(agentic, monkeypatch):
    monkeypatch.setenv("GBENCH_BFCL_V4_AGENTIC_TEMPERATURE", "1.0")
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    assert res["temperature"] == 1.0 and _generate_temperature(agentic) == 1.0
    assert res["leaderboard_comparable"] is False
    assert "(BFCL generates at 0.001)" in res["leaderboard_comparable_reason"]


def test_every_question_without_a_final_reply_is_flagged(agentic, monkeypatch):
    monkeypatch.setattr(AG, "_score_error_types",
                        lambda *a: {"agentic:no_last_message": 10})
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    assert res["leaderboard_comparable"] is False
    assert "agentic:no_last_message" in res["leaderboard_comparable_reason"]
    assert res["bfcl_report"]["error_types"] == {"agentic:no_last_message": 10}


def test_the_handler_adds_only_the_thinking_toggle():
    cls = AG._patched_openai_handler()
    assert set(vars(cls)) - {"__module__", "__qualname__", "__doc__", "__firstlineno__",
                             "__static_attributes__"} == {"generate_with_backoff"}


def test_a_prose_reply_ends_the_turn_and_is_what_the_checker_grades(monkeypatch):
    """A reply with no tool_calls field (OpenAI, current vLLM): upstream keeps its text, the FC
    decoder raises on it (the turn ends), and the agentic checker grades exactly that text."""
    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:9/v1")
    from bfcl_eval.eval_checker.eval_runner import _evaluate_single_agentic_entry
    cls = AG._patched_openai_handler()
    handler = cls(model_name="m", temperature=0.001, registry_name="m", is_fc_model=True)
    msg = types.SimpleNamespace(content="The founder is Zhang Yiming.", tool_calls=None)
    resp = types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)],
                                 usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1))
    parsed = handler._parse_query_response_FC(resp)
    assert parsed["model_responses"] == "The founder is Zhang Yiming."
    with pytest.raises(Exception):
        handler.decode_execute(parsed["model_responses"], has_tool_call_tag=False)
    steps = [[{"search_engine_query": json.dumps({"keywords": "tea"})}], parsed["model_responses"]]
    ok = _evaluate_single_agentic_entry(handler, "web_search_0", [steps], ["Zhang Yiming"],
                                        {"question": []}, "m", "web_search_base")
    assert ok["valid"] is True
    bad = _evaluate_single_agentic_entry(handler, "web_search_0", [steps], ["Someone Else"],
                                         {"question": []}, "m", "web_search_base")
    assert bad["valid"] is False and bad["error"]["error_type"] == "agentic:answer_not_found"


# --------------------------------------------------------------------------- #
# Gemini key: checked with the Gemini API before generating (the judged suites' gate)
# --------------------------------------------------------------------------- #

@pytest.fixture
def gemini(monkeypatch):
    """A local stand-in for the Gemini API's OpenAI-compatible /models endpoint
    (GEMINI_OPENAI_BASE_URL); `state` picks the reply."""
    state = {"mode": "live", "hits": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            state["hits"] += 1
            if state["mode"] == "invalid":
                code, body = 401, {"error": {"code": 401, "status": "UNAUTHENTICATED"}}
            elif state["mode"] == "bad-key":
                code, body = 400, {"error": {"message": "API key not valid. Please pass a "
                                                        "valid API key.",
                                             "status": "INVALID_ARGUMENT"}}
            else:
                code, body = 200, {"object": "list", "data": [{"id": "models/gemini"}]}
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}/v1")
    monkeypatch.setenv("GEMINI_API_KEY", "secret-key-123")
    yield state
    srv.shutdown()


def test_a_live_key_is_accepted(gemini):
    assert AG.search_backend() == (AG.SEARCH_BACKEND, "")
    # the key ping, then one model check per model of the Gemini cascade (base's gate)
    from gbench.runners.eval_suites import base
    assert gemini["hits"] == 1 + len(base.gemini_cascade_models())


def test_a_rejected_key_is_refused_without_echoing_it(gemini):
    gemini["mode"] = "invalid"
    backend, why = AG.search_backend()
    assert backend is None and "HTTP 401" in why and "rejected the key" in why
    assert "secret-key-123" not in why


def test_a_key_the_api_calls_invalid_is_refused(gemini):
    gemini["mode"] = "bad-key"
    backend, why = AG.search_backend()
    assert backend is None and "HTTP 400" in why and "API key not valid" in why


def test_an_unreachable_gemini_endpoint_does_not_block_the_key(monkeypatch):
    # As for the judged suites: a check that gets no answer is inconclusive, never a refusal.
    monkeypatch.setenv("GEMINI_API_KEY", "k-0123456789")
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", f"http://127.0.0.1:{base.free_port()}/v1")
    assert AG.search_backend() == (AG.SEARCH_BACKEND, "")


def test_a_dead_key_leaves_web_search_infra_required_before_generating(gemini, agentic,
                                                                      monkeypatch):
    gemini["mode"] = "invalid"
    monkeypatch.setattr(AG, "search_backend", _REAL_SEARCH_BACKEND)
    # the suite's own run: a hard error before generating
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*bfcl_v4_agentic.*rejected the key"):
        AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    assert not [c for c in agentic["cli"] if c[0] == "generate"]
    # bfcl's agentic part: the setting is infra_required and Memory runs
    with AG.suite_context("bfcl", "docs/evals/bfcl.md"):
        res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    web = res["settings"]["web_search"]
    assert web["status"] == "infra_required" and web["when"] == "before generation"
    assert "rejected the key" in web["infra_required"] and "secret-key-123" not in json.dumps(res)
    gen = next(c for c in agentic["cli"] if c[0] == "generate")
    assert gen[gen.index("--test-category") + 1] == ",".join(AG.MEMORY_CATEGORIES)
    agentic["cli"].clear()
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*bfcl_v4_agentic.*rejected the key"):
        AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1,
                               eval_categories=",".join(AG.WEB_SEARCH_CATEGORIES))
    assert agentic["cli"] == []
    why = bfcl._check_agentic("http://x/v1", list(AG.WEB_SEARCH_CATEGORIES))
    assert "rejected the key" in why and "secret-key-123" not in why


def test_the_key_is_not_checked_again_after_generation(agentic, monkeypatch):
    """After generation the search log decides, not another key check."""
    answers = iter([(AG.SEARCH_BACKEND, ""), (None, "the Gemini API rejected the key")])
    monkeypatch.setattr(AG, "search_backend", lambda: next(answers))
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    assert res["leaderboard_comparable_reason"].startswith(AG.SEARCH_SUBSTITUTION_REASON)
    assert "rejected" not in res["leaderboard_comparable_reason"]
    assert res["bfcl_report"]["web_search_calls"]["failed"] == 0


# --------------------------------------------------------------------------- #
# harness pin: bfcl-eval 2026.3.23 (= the pinned gorilla commit)
# --------------------------------------------------------------------------- #

def test_harness_pins_name_one_release_and_agree_on_shared_files():
    assert AG.BFCL_EVAL_RELEASE == bfcl.BFCL_EVAL_RELEASE == "2026.3.23"
    assert all(len(v) == 1 for v in bfcl._MT_PINNED_BLOBS.values())
    shared = set(AG._AGENTIC_PINNED_BLOBS) & set(bfcl._MT_PINNED_BLOBS)
    assert shared and all(AG._AGENTIC_PINNED_BLOBS[f] == bfcl._MT_PINNED_BLOBS[f][0] for f in shared)
    assert len(bfcl._MT_PINNED_BLOBS) == 38 and len(AG._AGENTIC_PINNED_BLOBS) == 35


def test_installed_harness_is_accepted_only_at_the_pinned_release():
    installed = importlib.metadata.version("bfcl-eval")
    why = AG.harness_mismatch()
    if installed == AG.BFCL_EVAL_RELEASE:
        assert why is None
    else:
        assert "bfcl-eval==2026.3.23" in why and installed in why
        assert AG.image_build_command() in why


def test_a_changed_agentic_file_is_refused_by_the_prerequisite_check(monkeypatch, tmp_path):
    import bfcl_eval
    from pathlib import Path
    root = Path(bfcl_eval.__file__).resolve().parent
    pins = {rel: AG._git_blob_sha1((root / rel).read_bytes())
            for rel in ("utils.py", "eval_checker/agentic_eval/agentic_checker.py")}
    monkeypatch.setattr(AG, "_AGENTIC_PINNED_BLOBS", pins)
    assert AG.harness_mismatch() is None
    monkeypatch.setitem(AG._AGENTIC_PINNED_BLOBS, "utils.py", "0" * 40)
    why = AG.harness_mismatch()
    assert "utils.py" in why and "bfcl-eval==2026.3.23" in why
    # the prerequisite check reads the mismatch from the image's probe (run here in-process)
    monkeypatch.setattr(AG, "image_problem", lambda: None)
    monkeypatch.setattr(AG, "run_in_image", lambda *a, **k: {
        **AG.image_probe(), "release": AG.BFCL_EVAL_RELEASE})

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(AG.urllib.request, "urlopen", lambda *a, **k: _Resp())
    ok, reason, cats = AG.check_bfcl_prerequisites("http://x/v1")
    assert ok is False and cats == [] and "utils.py" in reason


def test_child_bootstrap_registers_models_under_the_run_project_root():
    code = AG._CHILD_BOOTSTRAP.format(model="m", args=["evaluate"])
    assert 'os.environ["BFCL_PROJECT_ROOT"]' in code and "~/.cache" not in code


# --------------------------------------------------------------------------- #
# official sampling of the other function-calling suites
# --------------------------------------------------------------------------- #

def _capture_run(monkeypatch, module):
    """Replace module.run_eval_suite with a recorder that returns an empty result."""
    seen = {}

    def fake(**kw):
        seen.update(kw)
        official = kw["official_sampling"]
        temp, src = base.resolve_official_sampling(kw["eval_name"], official)
        return {"sample_traces": [], "accuracy": 0.0, "temperature": temp,
                "temperature_source": src, "attempts": None, "category_accuracy": {},
                "official_sampling": official.record(temp, src)}
    monkeypatch.setattr(module, "run_eval_suite", fake)
    return seen


def _stub_loaders(monkeypatch, name, mod, tmp_path):
    if name == "gorilla_apibench":
        monkeypatch.setattr(mod, "check_prereqs", lambda: None)
        monkeypatch.setattr(mod, "_load_gorilla_apibench_samples", lambda limit=None: [])
    elif name == "seal_tools":
        monkeypatch.setattr(mod, "_load_seal_tools_samples", lambda limit=None: [])
    elif name == "acebench":
        monkeypatch.setattr(mod, "_load_acebench_samples", lambda **k: [])
        monkeypatch.setattr(mod, "_load_agent_items", lambda language="en": [])
    elif name == "bfcl_live":
        monkeypatch.setattr(mod, "_load_bfcl_live_samples", lambda: ([], {}))
        monkeypatch.setattr(bfcl, "_harness_probe",
                            lambda name, multi, single: (bfcl.BFCL_EVAL_RELEASE, []))
    elif name == "nestful":
        monkeypatch.setattr(mod, "_func_dir", lambda: str(tmp_path))
        monkeypatch.setattr(mod, "_data_file", lambda d: str(tmp_path / "data.jsonl"))
        monkeypatch.setattr(mod, "_load_nestful_samples", lambda limit=None, data_path=None: [])

        class _NoWorker:
            kills = 0

            def __init__(self, *a):
                pass

            def close(self):
                pass
        monkeypatch.setattr(mod, "_WinRateWorker", _NoWorker)
    elif name == "api_bank":
        monkeypatch.setattr(mod, "_checkout_dir", lambda: str(tmp_path))
        monkeypatch.setattr(mod, "_load_l12_samples", lambda d: [([{"role": "user", "content": "q"}],
                                                                  "{}", {"category": "c"})])
        monkeypatch.setattr(mod, "_load_l3_samples", lambda: [])
        monkeypatch.setattr(mod, "_score_traces", lambda d, traces: set())


@pytest.mark.parametrize("name,constant", [
    ("gorilla_apibench", "GORILLA_SAMPLING"), ("seal_tools", "SEAL_TOOLS_SAMPLING"),
    ("acebench", "ACEBENCH_SAMPLING"), ("bfcl_live", "BFCL_OFFICIAL_SAMPLING"),
    ("nestful", "NESTFUL_SAMPLING"), ("api_bank", "API_BANK_SAMPLING"),
])
def test_each_suite_hands_its_official_sampling_to_the_request_path(name, constant, monkeypatch,
                                                                     tmp_path):
    mod = importlib.import_module(f"gbench.runners.eval_suites.{name}")
    _stub_loaders(monkeypatch, name, mod, tmp_path)
    seen = _capture_run(monkeypatch, mod)
    run = getattr(mod, f"run_{name}")
    res = run("m", "http://x", 1, enable_thinking=True)
    official = getattr(mod, constant)
    assert seen["official_sampling"] is official
    # the stub resolves a no-think request: the rule's 0.0, which matches only an official 0.0
    assert res["temperature"] == 0.0
    matches = official.send_temperature and official.temperature == 0.0
    assert res["official_sampling"]["temperature_matches_official"] is matches
    assert ("temperature" in (res.get("leaderboard_comparable_reason") or "")) is (not matches)


@pytest.mark.parametrize("name,temperature,send,params", [
    ("gorilla_apibench", 0.0, True, None),
    ("nestful", 0.0, True, None),
    ("nexusraven_api_eval", 0.0, True, None),
    ("acebench", 0.0, True, {"top_p": 0.9}),
    ("t_eval", 0.0, True, None),
    ("seal_tools", None, False, None),
    ("api_bank", None, False, None),
    ("agent_dojo", None, False, None),
    ("bfcl_live", 0.001, True, None),
])
def test_each_suite_declares_its_upstream_sampling(name, temperature, send, params):
    mod = importlib.import_module(f"gbench.runners.eval_suites.{name}")
    official = [v for v in vars(mod).values() if isinstance(v, base.OfficialSampling)]
    assert len(official) == 1, name
    o = official[0]
    assert (o.temperature, o.send_temperature, o.params) == (temperature, send, params)
    assert o.reference and o.reference.isascii()


def test_a_no_think_run_samples_at_zero_whatever_the_official_sampling():
    from gbench.runners.eval_suites import acebench, seal_tools
    assert base.resolve_official_sampling("acebench", acebench.ACEBENCH_SAMPLING,
                                          thinking=False) == (0.0, "default")
    assert base.resolve_official_sampling("seal_tools", seal_tools.SEAL_TOOLS_SAMPLING,
                                          thinking=False) == (0.0, "default")
    assert seal_tools.SEAL_TOOLS_SAMPLING.record(0.0, "default")[
        "temperature_matches_official"] is False


def test_a_thinking_run_samples_at_one_whatever_the_official_sampling():
    from gbench.runners.eval_suites import acebench, seal_tools
    assert base.resolve_official_sampling("acebench", acebench.ACEBENCH_SAMPLING,
                                          thinking=True) == (1.0, "thinking")
    assert base.resolve_official_sampling("seal_tools", seal_tools.SEAL_TOOLS_SAMPLING,
                                          thinking=True) == (1.0, "thinking")
    rec = seal_tools.SEAL_TOOLS_SAMPLING.record(1.0, "thinking")
    assert rec["temperature_matches_official"] is False


def test_seal_tools_flags_a_temperature_upstream_does_not_send(monkeypatch):
    from gbench.runners.eval_suites import seal_tools as S
    monkeypatch.setattr(S, "_load_seal_tools_samples", lambda limit=None: [])

    def fake(**kw):
        return {"sample_traces": [], "accuracy": 0.0, "temperature": 0.3,
                "temperature_source": "cli:--temperature", "attempts": None,
                "official_sampling": kw["official_sampling"].record(0.3, "cli:--temperature")}
    monkeypatch.setattr(S, "run_eval_suite", fake)
    r = S.run_seal_tools("m", "http://x", 1)
    assert r["leaderboard_comparable"] is False
    assert "upstream sends none" in r["leaderboard_comparable_reason"]


@pytest.mark.parametrize("thinking,expected", [(False, 0.0), (True, 1.0)])
def test_agent_dojo_requests_carry_the_rules_temperature_zero_included(thinking, expected):
    """AgentDojo's OpenAILLM sends `temperature or NOT_GIVEN`, which drops a 0.0: gbench's wrapper
    on the client's chat.completions.create puts the run's value on every request."""
    from gbench.runners.eval_suites import agent_dojo as AD
    temp, src = base.resolve_official_sampling("agent_dojo", AD.AGENTDOJO_SAMPLING,
                                               thinking=thinking)
    assert (temp, src) == (expected, "thinking" if thinking else "default")
    sent = []
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(
        create=lambda **kw: sent.append(kw) or "reply")))
    AD._send_temperature(client, temp)
    # what agentdojo's chat_completion_request passes: `temperature or NOT_GIVEN`
    from openai import NOT_GIVEN
    assert client.chat.completions.create(model="m", messages=[],
                                          temperature=temp or NOT_GIVEN) == "reply"
    assert sent[0]["temperature"] == expected and isinstance(sent[0]["temperature"], float)
    assert AD.AGENTDOJO_SAMPLING.record(temp, src)["temperature_matches_official"] is False


# --------------------------------------------------------------------------- #
# NexusRaven: the two domain numbers the repo publishes
# --------------------------------------------------------------------------- #

def test_nexusraven_domains_are_unweighted_task_means():
    from gbench.runners.eval_suites import nexusraven_api_eval as NX
    # GPT-4 zero-shot per task (README figures): cve_cpe 50/78, virustotal 122/151, emailrep 34/38
    per_task = {"cve_cpe": [True] * 50 + [False] * 28, "virustotal": [True] * 122 + [False] * 29,
                "emailrep": [True] * 34 + [False] * 4}
    dom = NX.domain_accuracy(per_task)
    assert dom["Cybersecurity Domain"] == 78.12 and dom["Generic Domain"] is None
    assert sorted(t for ts in NX.DOMAINS.values() for t in ts) == sorted(NX.TASKS)


def test_nexusraven_headline_is_the_mean_of_the_two_domains(monkeypatch):
    from gbench.runners.eval_suites import nexusraven_api_eval as NX
    monkeypatch.setattr(NX, "_load_nexusraven_api_eval_samples", lambda limit=None: [])
    traces = []
    for task, (ok, n) in {"toolalpaca": (1, 2), "toolllm": (1, 1), "cve_cpe": (1, 1),
                          "emailrep": (0, 1), "virustotal": (1, 1)}.items():
        for k in range(n):
            traces.append({"gold_answer": json.dumps({"task": task, "key": k}),
                           "response_text": "x", "_ok": k < ok})
    monkeypatch.setattr(NX, "score", lambda task, key, text: (
        next(t["_ok"] for t in traces if json.loads(t["gold_answer"]) == {"task": task, "key": key}),
        False))
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return {"sample_traces": traces, "accuracy": 0.0, "temperature": 0.0,
                "temperature_source": "official", "attempts": None,
                "official_sampling": kw["official_sampling"].record(0.0, "official")}
    monkeypatch.setattr(NX, "run_eval_suite", fake)
    r = NX.run_nexusraven_api_eval("m", "http://x", 1)
    assert r["domain_accuracy"] == {"Generic Domain": 75.0, "Cybersecurity Domain": 66.67}
    assert r["accuracy"] == round((75.0 + 66.67) / 2, 2)
    assert seen["official_sampling"] is NX.NEXUSRAVEN_SAMPLING
    assert "Nexus_Function_Calling" not in json.dumps(r)


def test_nexusraven_doc_cites_only_the_benchmark_it_runs():
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1] / "docs" / "evals" /
           "nexusraven_api_eval.md").read_text(encoding="utf-8")
    assert "Nexus_Function_Calling_Leaderboard" not in doc and "calculate_averages" not in doc
    assert "Zero-shot_Evaluation.png" in doc and "Generic Domain" in doc
    src = (Path(__file__).resolve().parents[1] / "gbench" / "runners" / "eval_suites" /
           "nexusraven_api_eval.py").read_text(encoding="utf-8")
    assert "Nexus_Function_Calling_Leaderboard" not in src and "calculate_averages" not in src


# --------------------------------------------------------------------------- #
# T-Eval: upstream post-processing and the Chinese setting
# --------------------------------------------------------------------------- #

def test_t_eval_applies_test_pys_split_special_tokens():
    from gbench.runners.eval_suites import t_eval as TE
    raw = '```json\n{"name": "f", "args": {}}\n```'
    assert TE.split_special_tokens(raw) == '{"name": "f", "args": {}}'
    assert TE.split_special_tokens("answer<|im_end|>junk") == "answer"
    assert TE.split_special_tokens("answer\nuser: next") == "answer"
    d = TE._datum({"gold_answer": json.dumps({"sample": {"gt": 1}}),
                   "response_text": "<thought>x</thought>```json\n{\"a\": 1}\n```"})
    assert d["prediction"] == '{"a": 1}'


def test_t_eval_runs_both_published_settings(monkeypatch):
    from gbench.runners.eval_suites import t_eval as TE
    stems = []

    def load(dim, stem, limit=None):
        stems.append(stem)
        return [([{"role": "user", "content": stem}], json.dumps({"dimension": dim, "sample": {}}),
                 {"category": f"t_eval::{dim}"})]
    monkeypatch.setattr(TE, "_load_file_samples", load)
    monkeypatch.setattr(TE, "_score_file", lambda dim, traces, language="en": {})

    async def reply(**kw):
        return base.Reply("ok", None, "stop", completion_tokens=3)
    with mock.patch.object(base, "_send_single_request", reply):
        r = TE.run_t_eval("m", "http://x", 1)
    assert stems == [s for _, s in TE._FILES] + [s + "_zh" for _, s in TE._FILES]
    assert "overall_zh" in r and set(r["dimension_scores_zh"]) == set(r["dimension_scores"])
    assert r["temperature"] == 0.0 and r["official_sampling"]["temperature_matches_official"]
    assert r["leaderboard_comparable"] is True


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #

def test_docs_match_the_new_behaviour():
    from pathlib import Path
    d = Path(__file__).resolve().parents[1] / "docs" / "evals"
    bfcl_md = (d / "bfcl.md").read_text(encoding="utf-8")
    ag_md = (d / "bfcl_v4_agentic.md").read_text(encoding="utf-8")
    for needle in ("bfcl-eval==2026.3.23", "0.001", "official_sampling", "GEMINI_API_KEYS",
                   "--max-output-tokens"):
        assert needle in bfcl_md, needle
    for needle in ("bfcl-eval==2026.3.23", "0.001", "base.gemini_key_live_valid",
                   "web_search_requests.jsonl", "_AGENTIC_PINNED_BLOBS", "agentic:no_last_message",
                   "--max-output-tokens"):
        assert needle in ag_md, needle
    for name in ("bfcl.md", "bfcl_live.md", "bfcl_v4_agentic.md", "gorilla_apibench.md",
                 "nestful.md", "seal_tools.md", "acebench.md", "api_bank.md", "t_eval.md",
                 "agent_dojo.md", "nexusraven_api_eval.md"):
        text = (d / name).read_text(encoding="utf-8")
        assert text.isascii(), name
        assert "0.0 greedy, 1.0 with" not in text and "0.0 greedy / 1.0" not in text, name
        assert "--max-output-tokens" in text, name
