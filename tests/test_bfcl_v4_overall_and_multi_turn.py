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

"""BFCL v4 in full (single-turn, multi-turn in bfcl-eval's own harness, agentic) with BFCL's
Overall Acc, plus the function-calling suite fixes: t_eval's merged counters, reasoning tags
stripped before api_bank / t_eval scoring, acebench's bounded products and nestful's
comparability flag for killed Win Rate calls.

The multi-turn tests drive bfcl_eval's real inference loop, executable backends and checker
through a local OpenAI-compatible stub server (never a model server)."""

import ast
import csv
import importlib
import inspect
import json
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import base, bfcl
from gbench.runners.eval_suites.bfcl_v4_agentic import search_backend as _REAL_SEARCH_BACKEND
from gbench.runners.eval_suites.fc_common import bfcl_score

REPO = Path(__file__).resolve().parents[1]
SUITES = REPO / "gbench" / "runners" / "eval_suites"


@pytest.fixture(autouse=True)
def _project_root(tmp_path_factory, monkeypatch):
    # bfcl_eval creates result/score/lock directories under BFCL_PROJECT_ROOT on first import.
    if "bfcl_eval.constants.eval_config" not in sys.modules:
        monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path_factory.mktemp("bfcl_project")))
    monkeypatch.setenv("TQDM_DISABLE", "1")


# --------------------------------------------------------------------------- #
# categories and aggregation, against the installed bfcl_eval
# --------------------------------------------------------------------------- #

def test_categories_and_groups_are_bfcl_evals():
    from bfcl_eval.constants.category_mapping import ALL_SCORING_CATEGORIES, TEST_COLLECTION_MAPPING
    assert bfcl.ALL_SCORING_CATEGORIES == list(ALL_SCORING_CATEGORIES)
    assert bfcl.SUPPORTED_BFCL_CATEGORIES == list(ALL_SCORING_CATEGORIES)
    for group, cats in TEST_COLLECTION_MAPPING.items():
        want = [c for c in cats if c != "format_sensitivity"]
        assert bfcl.resolve_categories(group) == want, group
    assert bfcl.resolve_categories(None) == list(ALL_SCORING_CATEGORIES)


@pytest.mark.parametrize("seed", range(6))
def test_every_column_matches_bfcls_leaderboard_csvs(seed, tmp_path):
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from bfcl_eval.eval_checker.eval_runner_helper import generate_leaderboard_csv
    name = next(k for k in MODEL_CONFIG_MAPPING if "/" not in k)
    rng = random.Random(seed)
    table, ca = {name: {}}, {}
    for cat in bfcl.ALL_SCORING_CATEGORIES:
        n = rng.randint(1, 60)
        k = rng.randint(0, n)
        table[name][cat] = {"accuracy": k / n, "total_count": n}
        ca[cat] = {"correct": k, "total": n, "accuracy": round(k / n * 100, 2)}
    generate_leaderboard_csv(table, tmp_path)
    summary = bfcl.bfcl_summary(ca)
    for fname, key in (("data_overall.csv", "overall"), ("data_non_live.csv", "non_live"),
                       ("data_live.csv", "live"), ("data_multi_turn.csv", "multi_turn"),
                       ("data_agentic.csv", "agentic")):
        header, row = list(csv.reader(open(tmp_path / fname)))[:2]
        cols = summary[key]
        compared = [h for h in header if h in cols]
        assert len(compared) == len(cols), (fname, set(cols) - set(compared))
        for i, h in enumerate(header):
            if h in cols:
                assert f"{cols[h]:.2f}%" == row[i], (fname, h)
    assert bfcl.overall_score(ca) == summary["overall"]["Overall Acc"]


def test_overall_needs_every_weighed_category_but_not_live_relevance():
    ca = {c: {"correct": 1, "total": 2, "accuracy": 50.0} for c in bfcl.ALL_SCORING_CATEGORIES}
    assert bfcl.overall_score(ca) == 50.0
    assert bfcl.overall_score({k: v for k, v in ca.items() if k != "live_relevance"}) == 50.0
    assert bfcl.overall_score({k: v for k, v in ca.items() if k != "memory_kv"}) is None
    assert sum(w for _, _, w in bfcl.OVERALL_SEGMENTS) == 100
    assert sum(len(c) for _, c, _ in bfcl.OVERALL_SEGMENTS) == 21


def test_reply_text_is_never_a_call_for_any_single_turn_category():
    doc = [{"name": "f", "description": "F.", "parameters": {
        "type": "dict", "required": ["x"], "properties": {"x": {"type": "integer", "description": "X."}}}}]
    txt = '{"name": "f", "arguments": {"x": 1}}'
    assert not bfcl_score("simple_python", doc, [{"f": {"x": [1]}}], response_text=txt)["valid"]
    assert not bfcl_score("live_relevance", doc, [], response_text=txt)["valid"]
    assert bfcl_score("irrelevance", doc, [], response_text=txt)["valid"]
    call = [{"type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}]
    assert bfcl_score("simple_python", doc, [{"f": {"x": [1]}}], tool_calls=call)["valid"]
    src = (SUITES / "fc_common.py").read_text(encoding="utf-8")
    assert "text_fallback" not in src and "bfcl_text_json_calls" not in src


# --------------------------------------------------------------------------- #
# multi-turn harness pinning
# --------------------------------------------------------------------------- #

def test_installed_bfcl_eval_matches_the_pinned_harness():
    # Only the release byte-identical to the pinned commit is accepted; any other installed
    # release stops the suite with the install command.
    import importlib.metadata
    installed = importlib.metadata.version("bfcl-eval")
    if installed == bfcl.BFCL_EVAL_RELEASE:
        assert bfcl.check_multi_turn_harness() == installed
    else:
        with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*bfcl-eval==2026\.3\.23.*docker build"):
            bfcl.check_multi_turn_harness()


def test_a_changed_harness_file_is_a_hard_infra_error(monkeypatch):
    rel = "eval_checker/multi_turn_eval/multi_turn_checker.py"
    monkeypatch.setitem(bfcl._MT_PINNED_BLOBS, rel, ("0" * 40,))
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*bfcl.*multi_turn_checker\.py"):
        bfcl.check_multi_turn_harness()


def test_a_missing_bfcl_eval_is_a_hard_infra_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "bfcl_eval", None)
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*bfcl-eval==2026\.3\.23.*docker build"):
        bfcl.check_multi_turn_harness()


def test_multi_turn_entries_load_from_the_package():
    items = bfcl.load_multi_turn_entries(bfcl.MULTI_TURN_CATEGORIES)
    assert len(items) == 800
    assert {it["category"] for it in items} == set(bfcl.MULTI_TURN_CATEGORIES)
    miss = next(it for it in items if it["category"] == "multi_turn_miss_func")
    assert miss["entry"]["missed_function"]                    # held-out docs, added on their turn
    assert miss["entry"]["function"][0]["description"].endswith("Python 3 syntax.")
    assert not miss["eval_entry"]["function"][0]["description"].endswith("Python 3 syntax.")


# --------------------------------------------------------------------------- #
# execution guard
# --------------------------------------------------------------------------- #

def _execute(calls, entry_id, guarded):
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils as mtu
    items = {it["id"]: it for it in bfcl.load_multi_turn_entries(["multi_turn_base"])}
    it = items[entry_id]
    key = f"guardtest_{'g' if guarded else 'u'}_{time.time_ns()}"
    ctx = bfcl._harness_patches() if guarded else _nullguard()
    with ctx:
        out, _ = mtu.execute_multi_turn_func_call(
            calls, it["eval_entry"]["initial_config"], it["eval_entry"]["involved_classes"],
            key, entry_id)
    bfcl._drop_instances(key, entry_id)
    return out


class _nullguard:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def test_guard_leaves_every_benign_call_exactly_as_upstream_runs_it():
    items = {it["id"]: it for it in bfcl.load_multi_turn_entries(["multi_turn_base"])}
    for entry_id in ("multi_turn_base_0", "multi_turn_base_24", "multi_turn_base_77"):
        calls = [c for turn in items[entry_id]["ground_truth"] for c in turn]
        calls += ["no_such_tool(x=1)", "cd(folder=nan)", "cd(folder=", "cd(folder='a', folder='b')",
                  "cd(folder=-1)", "cd(folder=[1, {'a': (2, 3)}])"]
        assert _execute(calls, entry_id, True) == _execute(calls, entry_id, False), entry_id


def test_guard_refuses_calls_that_would_run_host_code():
    calls = ["open(file='/tmp/x', mode='w')", "print(x=1)", "copy.deepcopy(x=1)",
             "cd(folder=__import__('os').getcwd())", "exec(source='1')", "cd(folder=float('nan'))"]
    out = _execute(calls, "multi_turn_base_0", True)
    assert all(o.startswith("Error during execution: Function call ") and o.endswith("is not allowed.")
               for o in out), out
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils as mtu
    assert "eval" not in vars(mtu)                            # the patch is removed afterwards


def test_multi_turn_requests_carry_no_max_tokens(monkeypatch):
    # BFCL's handler sends no max_tokens, so gbench's handler sends none either (and has nothing
    # to shrink on a context-length rejection): only extra_body is added to upstream's request.
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
    sent = {}

    def capture(self, **kwargs):
        sent.update(kwargs)
        raise RuntimeError("stop here")
    monkeypatch.setattr(OpenAICompletionsHandler, "generate_with_backoff", capture)
    handler = bfcl._fc_handler("m", "http://127.0.0.1:9/v1", 0.001, True, "k")
    with pytest.raises(RuntimeError, match="stop here"):
        handler.generate_with_backoff(messages=[], model="m", temperature=0.001, store=False)
    assert "max_tokens" not in sent and sent["temperature"] == 0.001
    assert sent["extra_body"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert not hasattr(bfcl, "_clamped_max_tokens")


# --------------------------------------------------------------------------- #
# multi-turn end to end through a local stub server
# --------------------------------------------------------------------------- #

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


def _key(messages):
    return json.dumps([(m.get("role"), m.get("content")) for m in messages
                       if m.get("role") in ("user", "system")])


def _stub(items, mode="truth", log=None):
    """An OpenAI chat server. mode=truth answers each turn with its ground-truth calls (one step)
    then ends the turn; mode=prose never calls a tool."""
    from bfcl_eval.constants.default_prompts import DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC
    params, table = _backend_params(), {}
    for it in items:
        convo = []
        for t, turn in enumerate(it["entry"]["question"]):
            if str(t) in it["entry"].get("missed_function", {}):
                turn = [{"role": "user", "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC}]
            convo += turn
            table[_key(convo)] = it["ground_truth"][t]

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
            if log is not None:
                log.append(body)
            msgs = body["messages"]
            calls = [] if mode == "prose" or msgs[-1].get("role") == "tool" else table.get(_key(msgs), [])
            msg = ({"role": "assistant", "content": None,
                    "tool_calls": [tool_call(c, i) for i, c in enumerate(calls)]} if calls
                   else {"role": "assistant", "content": "Done.", "tool_calls": []})
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


def _pick(n_per_category=3):
    items = []
    for cat in bfcl.MULTI_TURN_CATEGORIES:
        items += bfcl.load_multi_turn_entries([cat])[:n_per_category]
    return items


@pytest.mark.parametrize("category", bfcl.MULTI_TURN_CATEGORIES)
def test_ground_truth_play_scores_every_entry_correct(category):
    items = bfcl.load_multi_turn_entries([category])[:4]
    log = []
    srv, url = _stub(items, log=log)
    try:
        res = bfcl.run_multi_turn("stub", url, 4, items, temperature=0.0,
                                  thinking=True)
    finally:
        srv.shutdown()
    assert [t["is_correct"] for t in res["traces"]] == [True] * 4, [
        (t["question_id"], t.get("bfcl_error_type")) for t in res["traces"]]
    assert not res["guard_refusals"]
    req = log[0]
    assert "max_tokens" not in req and req["temperature"] == 0.0
    assert req["chat_template_kwargs"] == {"enable_thinking": True}
    assert req["tools"][0]["function"]["description"].endswith("Python 3 syntax.")
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils as mtu
    assert not [k for k in vars(mtu) if k.startswith("gbench_bfcl_")]   # instances dropped


def test_held_out_functions_arrive_on_their_turn():
    items = [it for it in bfcl.load_multi_turn_entries(["multi_turn_miss_func"])][:1]
    log = []
    srv, url = _stub(items, log=log)
    try:
        bfcl.run_multi_turn("stub", url, 1, items, temperature=0.0, thinking=False)
    finally:
        srv.shutdown()
    counts = [len(r["tools"]) for r in log]
    held = sum(len(v) for v in items[0]["entry"]["missed_function"].values())
    assert counts[-1] == counts[0] + held
    assert any(m.get("content") == "I have updated some more functions you can choose from. What about now?"
               for m in log[-1]["messages"])


def test_a_model_that_never_calls_fails_every_entry_on_bfclss_checker():
    items = _pick(1)
    srv, url = _stub(items, mode="prose")
    try:
        res = bfcl.run_multi_turn("stub", url, 2, items, temperature=0.0,
                                  thinking=False)
    finally:
        srv.shutdown()
    assert [t["bfcl_error_type"] for t in res["traces"]] == ["multi_turn:empty_turn_model_response"] * 4
    assert all(t["status"] == "OK" for t in res["traces"])


def test_an_unreachable_server_is_a_failed_request_not_a_model_answer():
    items = _pick(1)[:2]
    port = base.free_port()
    res = bfcl.run_multi_turn("stub", f"http://127.0.0.1:{port}/v1", 2, items,
                              temperature=0.0, thinking=False)
    counts = bfcl._multi_turn_counts(res["traces"])
    assert counts["failed_requests"] == 2 and counts["correct_answers"] == 0
    assert all(t["bfcl_error_type"] == "multi_turn:inference_error" for t in res["traces"])


def test_attempts_do_not_share_backend_state():
    # one category: the stub routes by conversation, and base_0 / miss_func_0 open alike
    items = bfcl.load_multi_turn_entries(["multi_turn_base"])[:2]
    srv, url = _stub(items)
    try:
        res = bfcl.run_multi_turn("stub", url, 2, items, temperature=0.0,
                                  thinking=False, attempts=3)
    finally:
        srv.shutdown()
    assert len(res["traces"]) == 6 and all(t["is_correct"] for t in res["traces"])
    assert sorted((t["source_sample_idx"], t["attempt_idx"]) for t in res["traces"]) == [
        (i, a) for i in range(2) for a in range(3)]


# --------------------------------------------------------------------------- #
# the suite: all three parts, BFCL Overall Acc
# --------------------------------------------------------------------------- #

def _single_turn_result(**kw):
    ca = {c: {"correct": 3, "total": 4, "accuracy": 75.0} for c in bfcl.SINGLE_TURN_CATEGORIES}
    return {"benchmark_type": "eval", "eval_name": "bfcl", "model_name": "m", "thinking": False,
            "total_questions": 52, "correct_answers": 39, "failed_requests": 0,
            "empty_responses": 0, "truncated_responses": 0, "over_context_prompts": 0,
            "timed_out_requests": 0, "judge_outages": 0, "response_health": {"ok": 52},
            "category_accuracy": ca, "accuracy": 75.0, "status": "success", "temperature": 0.0,
            "temperature_source": "default",
            "sample_traces": [{"sample_idx": i, "source_sample_idx": i, "attempt_idx": 0,
                               "category": c, "is_correct": True}
                              for i, c in enumerate(bfcl.SINGLE_TURN_CATEGORIES)]}


def _agentic_result(extra_reason=None, temperature=0.001):
    """The agentic part's result: its web search is served by Gemini grounding, so it is never
    comparable; `extra_reason` adds another deviation."""
    from gbench.runners.eval_suites import bfcl_v4_agentic
    ca = {c: {"correct": 1, "total": 2, "accuracy": 50.0} for c in bfcl.AGENTIC_CATEGORIES}
    reasons = [bfcl_v4_agentic.SEARCH_SUBSTITUTION_REASON] + ([extra_reason] if extra_reason else [])
    return {"category_accuracy": ca, "total_questions": 10, "correct_answers": 5,
            "leaderboard_comparable": False, "leaderboard_comparable_reason": "; ".join(reasons),
            "temperature": temperature, "temperature_source": "official",
            "bfcl_report": {"search_backend": bfcl_v4_agentic.SEARCH_BACKEND},
            "sample_traces": [{"category": "memory_kv"}]}


@pytest.fixture
def full_run(monkeypatch):
    from gbench.runners.eval_suites import bfcl_v4_agentic
    calls = {}
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    monkeypatch.setattr(bfcl, "_load_bfcl_samples", lambda cats: ([], {}))
    # the harness image and its pin have their own tests; these ones cover the assembly around it
    monkeypatch.setattr(bfcl, "_harness_probe", lambda name, multi, single: (
        bfcl.BFCL_EVAL_RELEASE, [{"id": it["id"], "category": it["category"]}
                                 for it in (bfcl.load_multi_turn_entries(multi) if multi else [])]))
    monkeypatch.setattr(bfcl, "run_eval_suite", lambda **kw: calls.setdefault("single", kw) and _single_turn_result())
    monkeypatch.setattr(bfcl_v4_agentic, "check_bfcl_prerequisites", lambda url: (True, "", []))
    monkeypatch.setattr(bfcl_v4_agentic, "search_backend",
                        lambda: (bfcl_v4_agentic.SEARCH_BACKEND, ""))

    def agentic(*a, **kw):
        calls["agentic"] = kw
        # the agentic part generates at the temperature bfcl hands it
        return _agentic_result(calls.get("agentic_extra"), kw["temperature"])
    monkeypatch.setattr(bfcl_v4_agentic, "run_bfcl_v4_agentic", agentic)

    def multi(model, url, conc, items, **kw):
        calls["multi"] = (items, kw)
        traces = [{"sample_idx": i, "source_sample_idx": i, "attempt_idx": 0,
                   "category": it["category"], "question_id": it["id"], "is_correct": i % 2 == 0,
                   "status": "OK", "health": "ok", "steps": 3}
                  for i, it in enumerate(items)]
        return {"traces": traces, "guard_refusals": []}
    monkeypatch.setattr(bfcl, "run_multi_turn_in_image", multi)
    # the multi-turn part is stubbed, so the network checks before its container
    # (tests/test_guard_code_images.py) do not apply
    monkeypatch.setattr(bfcl, "endpoint_network", lambda base_url: base_url)
    return calls


def test_full_run_headlines_bfcl_overall_acc(full_run):
    res = bfcl.run_bfcl("m", "http://x", 4)
    assert set(res["category_accuracy"]) == set(bfcl.ALL_SCORING_CATEGORIES)
    # single-turn 75% everywhere, multi-turn 50%, agentic 50% -> .1*.75*3 + .3*.5 + .4*.5
    assert res["accuracy"] == pytest.approx(57.5) and res["accuracy"] == res["bfcl_summary"]["overall"]["Overall Acc"]
    assert res["metric"].startswith("BFCL Overall Acc")
    assert res["total_questions"] == 52 + 800 + 10
    # the deviations: the no-think 0.0 (gbench's rule, BFCL's is 0.001) and the agentic part's
    # web search served by Gemini grounding
    from gbench.runners.eval_suites.bfcl_v4_agentic import SEARCH_SUBSTITUTION_REASON
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"].startswith(
        f"temperature 0.0 (BFCL generates at 0.001); agentic: {SEARCH_SUBSTITUTION_REASON}")
    assert res["status"] == "success"
    assert full_run["agentic"]["eval_categories"] is None and full_run["agentic"]["limit"] is None
    assert res["multi_turn_report"]["entries"] == 800


def test_non_comparable_agentic_part_is_carried_into_the_reason(full_run):
    full_run["agentic_extra"] = "re-scored cached generations"
    res = bfcl.run_bfcl("m", "http://x", 4)
    assert res["leaderboard_comparable"] is False
    assert "agentic: web search served by Gemini Google Search grounding" in \
        res["leaderboard_comparable_reason"]
    assert "; re-scored cached generations" in res["leaderboard_comparable_reason"]


def test_eval_limit_is_split_over_the_parts(full_run):
    res = bfcl.run_bfcl("m", "http://x", 4, limit=22)
    assert full_run["single"]["limit"] == 13 and full_run["agentic"]["limit"] == 5
    assert len(full_run["multi"][0]) == 4
    assert {it["category"] for it in full_run["multi"][0]} == set(bfcl.MULTI_TURN_CATEGORIES)
    assert "--eval-limit 22 subset" in res["leaderboard_comparable_reason"]
    q3 = bfcl._part_limits(3, {"single": ["a"] * 13, "multi": ["b"] * 4, "agentic": []})
    assert set(q3) == {"single", "multi"} and sum(q3.values()) == 3
    assert res["eval_limit_plan"]["samples"] == 22
    assert res["eval_limit_plan"]["per_setting"] == {"single": 13, "multi": 4, "agentic": 5}


@pytest.mark.parametrize("limit", [1, 2, 5, 9])
def test_eval_limit_is_n_entries_for_the_whole_suite(full_run, limit):
    """--eval-limit N runs N entries over the three parts in total, never one per part at least;
    a part that gets 0 is not run."""
    res = bfcl.run_bfcl("m", "http://x", 4, limit=limit)
    quotas = bfcl._part_limits(limit, {"single": list(bfcl.SINGLE_TURN_CATEGORIES),
                                       "multi": list(bfcl.MULTI_TURN_CATEGORIES),
                                       "agentic": list(bfcl.AGENTIC_CATEGORIES)})
    assert sum(quotas.values()) == limit
    plan = dict(res["eval_limit_plan"])
    # the memory prerequisite entries the drawn agentic questions depend on are not samples
    assert ("memory prerequisite" in plan.pop("note", "")) == bool(quotas["agentic"])
    assert plan == {"limit": limit, "samples": limit, "per_setting": quotas,
                    "attempts_per_sample": 1, "generations": limit}
    assert (full_run.get("single") or {}).get("limit") == (quotas["single"] or None)
    assert ("single" in full_run) == bool(quotas["single"])
    assert len((full_run.get("multi") or [[]])[0]) == quotas["multi"]
    assert ("agentic" in full_run) == bool(quotas["agentic"])
    if quotas["agentic"]:
        assert full_run["agentic"]["limit"] == quotas["agentic"]


def test_no_eval_limit_runs_every_part_whole(full_run):
    res = bfcl.run_bfcl("m", "http://x", 4)
    assert full_run["single"]["limit"] is None and full_run["agentic"]["limit"] is None
    assert len(full_run["multi"][0]) == 800
    assert res["eval_limit_plan"]["limit"] is None
    assert res["eval_limit_plan"]["per_setting"] == {"single": None, "multi": None,
                                                     "agentic": None}


def test_agentic_runs_in_shard_one_only(full_run, monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "2/2")
    res = bfcl.run_bfcl("m", "http://x", 4)
    assert "agentic" not in full_run and len(full_run["multi"][0]) == 400
    assert "the agentic categories run in shard 1 only" in res["leaderboard_comparable_reason"]
    assert res["bfcl_summary"]["overall"]["Overall Acc"] is None


def test_web_search_without_a_gemini_key_is_reported_while_the_rest_runs(full_run, monkeypatch):
    from gbench.runners.eval_suites import bfcl_v4_agentic
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEYS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(bfcl_v4_agentic, "search_backend", _REAL_SEARCH_BACKEND)
    # web_search is the only agentic category: the agentic part is not called
    res = bfcl.run_bfcl("m", "http://x", 4, eval_categories="single_turn,multi_turn,web_search")
    assert "single" in full_run and "multi" in full_run and "agentic" not in full_run
    assert res["settings_infra_required"] == ["web_search"]
    assert "GEMINI_API_KEY (or GEMINI_API_KEYS) is unset" in \
        res["web_search_setting"]["infra_required"]
    assert res["web_search_setting"]["prerequisite"] == "GEMINI_API_KEY"
    assert res["status"] == "completed_with_errors"
    assert "Web Search setting" in res["leaderboard_comparable_reason"]
    # a run of web_search alone has nothing else: a hard error before anything is generated
    full_run.clear()
    with pytest.raises(RuntimeError,
                       match=r"INFRA REQUIRED.*GEMINI_API_KEY \(or GEMINI_API_KEYS\) is unset"):
        bfcl.run_bfcl("m", "http://x", 4, eval_categories="web_search")
    assert "single" not in full_run and "multi" not in full_run
    res = bfcl.run_bfcl("m", "http://x", 4, eval_categories="single_turn,multi_turn,memory")
    assert full_run["agentic"]["eval_categories"] == "memory_kv,memory_vector,memory_rec_sum"
    assert "category subset" in res["leaderboard_comparable_reason"]


def test_thinking_temperature_is_flagged(full_run, monkeypatch):
    monkeypatch.setenv("GBENCH_BFCL_TEMPERATURE", "1.0")
    res = bfcl.run_bfcl("m", "http://x", 4)
    assert "temperature 1.0 (BFCL generates at 0.001)" in res["leaderboard_comparable_reason"]
    assert full_run["multi"][1]["temperature"] == 1.0 and full_run["agentic"]["temperature"] == 1.0


def test_bfcl_live_reports_no_strict_duplicates():
    src = (SUITES / "bfcl_live.py").read_text(encoding="utf-8")
    assert "strict_" not in src and "TEXT_FALLBACK" not in src


# --------------------------------------------------------------------------- #
# t_eval: one result from all 8 files
# --------------------------------------------------------------------------- #

def test_t_eval_sums_counters_over_all_files_and_reports_no_placeholder_rows(monkeypatch):
    from gbench.runners.eval_suites import t_eval as TE
    monkeypatch.setattr(TE, "_load_file_samples", lambda dim, stem, limit=None: [
        ([{"role": "user", "content": dim}], json.dumps({"dimension": dim, "sample": {}}),
         {"category": f"t_eval::{dim}"})])
    monkeypatch.setattr(TE, "_score_file", lambda dim, traces, language="en": {})

    async def reply(**kw):
        if kw["messages"][0]["content"] == "instruct":
            return base.Reply("ok", None, "stop", completion_tokens=3)
        return base.Reply(None, None, None, None, "HTTP 500")
    with mock.patch.object(base, "_send_single_request", reply):
        r = TE.run_t_eval("m", "http://x", 1)
    # 8 English files and the same 8 in Chinese
    assert r["total_questions"] == 16 and r["failed_requests"] == 14
    assert r["status"] == "completed_with_errors"
    assert r["category_accuracy"] == {}
    assert r["file_status"]["instruct"] == "success" and r["file_status"]["review_str"] == "failed"
    assert r["file_status"]["instruct_zh"] == "success" and r["file_status"]["review_str_zh"] == "failed"
    assert sum(r["response_health"].values()) == 16


def test_t_eval_scores_the_reply_without_inline_reasoning():
    from gbench.runners.eval_suites import t_eval as TE
    d = TE._datum({"gold_answer": json.dumps({"sample": {"gt": 1}}),
                   "response_text": "<thought>maybe [f(x=1)]</thought>final"})
    assert d == {"gt": 1, "prediction": "final"}


def test_fc_suites_inject_no_output_budget_of_their_own():
    for mod in ("api_bank", "acebench", "gorilla_apibench", "nestful", "seal_tools",
                "nexusraven_api_eval", "t_eval"):
        src = (SUITES / f"{mod}.py").read_text(encoding="utf-8")
        assert 'kwargs.get("max_output_tokens",' not in src, mod


# --------------------------------------------------------------------------- #
# api_bank, acebench, nestful
# --------------------------------------------------------------------------- #

def test_api_bank_scores_the_reply_without_inline_reasoning(tmp_path, monkeypatch):
    from gbench.runners.eval_suites import api_bank as API
    seen = []
    (tmp_path / "evaluator.py").write_text(
        "class Sample:\n    @staticmethod\n    def from_chat_history(h):\n        return h\n"
        "class Evaluator:\n    def __init__(self, s):\n        pass\n"
        "    def evaluate(self, sid, call):\n        return call == '[Good()]', None\n")
    (tmp_path / "tool_manager.py").write_text(
        "class ToolManager:\n    def __init__(self, *a):\n        pass\n")
    (tmp_path / "api_call_extraction.py").write_text(
        "import re\nSEEN = []\n"
        "def get_api_call(text):\n    SEEN.append(text)\n    m = re.search(r'\\[\\w+\\(\\)\\]', text)\n"
        "    return m.group(0) if m else None\n"
        "def parse_api_call(c):\n    return c, {}\n")
    samples = tmp_path / "lv1-lv2-samples" / "level-1-given-desc"
    samples.mkdir(parents=True)
    (samples / "a.jsonl").write_text("{}\n")
    for name in ("evaluator", "tool_manager", "api_call_extraction"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.syspath_prepend(str(tmp_path))
    trace = {"gold_answer": json.dumps({"level": 1, "track": "api", "file": "a.jsonl", "sample_id": 0}),
             "response_text": "<thought>Maybe [Bad()] first.</thought>[Good()]"}
    try:
        API._score_traces(str(tmp_path), [trace])
        seen = sys.modules["api_call_extraction"].SEEN
    finally:
        for name in ("evaluator", "tool_manager", "api_call_extraction"):
            sys.modules.pop(name, None)
    assert seen == ["[Good()]"] and trace["is_correct"] is True


def test_acebench_integer_product_chains_cannot_stall_the_scorer():
    from gbench.runners.eval_suites import acebench as ACE
    chain = ast.parse("*".join(["7**33000"] * 200), mode="eval").body
    t0 = time.time()
    with pytest.raises(ValueError, match="too large"):
        ACE._literal_binop(chain)
    assert time.time() - t0 < 2.0
    assert ACE._literal_binop(ast.parse("3*4*5", mode="eval").body) == 60
    assert ACE._literal_binop(ast.parse("2**64 * 2**64", mode="eval").body) == 2 ** 128


def test_nestful_killed_win_rate_calls_break_comparability():
    from gbench.runners.eval_suites import nestful as NF
    clean = {"temperature": 0.0, "thinking": False, "attempts": None}
    assert NF._noncanonical_reasons(clean, None) == []
    reasons = NF._noncanonical_reasons({**clean, "win_rate_killed_calls": 2}, None)
    assert reasons and "2 Win Rate call(s) killed by the 30 s backstop" in reasons[0]


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #

def test_bfcl_doc_matches_the_code():
    doc = (REPO / "docs" / "evals" / "bfcl.md").read_text(encoding="utf-8")
    from gbench.runners.eval_suites.bfcl_v4_agentic import SEARCH_SUBSTITUTION_REASON
    for needle in ("BFCL Overall Acc", "multi_turn_long_context", "web_search_no_snippet",
                   "GEMINI_API_KEY", SEARCH_SUBSTITUTION_REASON, "bfcl-eval==2026.3.23",
                   bfcl.BFCL_EVAL_COMMIT, "--max-output-tokens", "Execution guard", "shard 1"):
        assert needle in " ".join(doc.split()), needle
    assert "SERPAPI_API_KEY" not in doc
    assert "strict_accuracy" not in doc and "leniency" not in doc
    for f in ("bfcl.md", "bfcl_live.md", "t_eval.md", "acebench.md", "nestful.md",
              "gorilla_apibench.md", "nexusraven_api_eval.md"):
        text = (REPO / "docs" / "evals" / f).read_text(encoding="utf-8")
        assert text.isascii(), f


def test_run_bfcl_merges_real_single_turn_and_multi_turn_parts(tmp_path, monkeypatch):
    """run_bfcl end to end on simple_python (base's request path, fake transport) plus
    multi_turn_base (bfcl_eval's loop against the local stub server)."""
    from gbench.runners.eval_suites.fc_common import BFCL_EVAL_COMMIT
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    root = tmp_path / "bfcl_data" / BFCL_EVAL_COMMIT
    (root / "possible_answer").mkdir(parents=True)
    doc = [{"name": "f", "description": "F.", "parameters": {
        "type": "dict", "required": ["x"], "properties": {"x": {"type": "integer", "description": "X."}}}}]
    rows = [{"id": f"simple_python_{i}", "question": [[{"role": "user", "content": f"q{i}"}]],
             "function": doc} for i in range(3)]
    (root / "BFCL_v4_simple_python.json").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (root / "possible_answer" / "BFCL_v4_simple_python.json").write_text("".join(
        json.dumps({"id": r["id"], "ground_truth": [{"f": {"x": [1]}}]}) + "\n" for r in rows))
    for f in root.rglob("*.json"):
        monkeypatch.setitem(bfcl._PINNED_BLOBS, f.relative_to(root).as_posix(),
                            bfcl._git_blob_sha1(f.read_bytes()))

    async def single(**kw):
        call = {"id": "c", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}
        return base.Reply("", [call], "tool_calls", completion_tokens=4)

    items = bfcl.load_multi_turn_entries(["multi_turn_base"])
    # the harness image and its pin have their own tests; this one runs the image's jobs (the
    # real loop and checkers) in this process on the installed package
    monkeypatch.setattr(bfcl, "_harness_probe", lambda name, multi, single: (
        bfcl.BFCL_EVAL_RELEASE, [{"id": it["id"], "category": it["category"]}
                                 for it in (bfcl.load_multi_turn_entries(multi) if multi else [])]))
    monkeypatch.setattr(bfcl, "check_single_turn",
                        lambda items, eval_name="bfcl": bfcl.single_turn_verdicts(items))
    monkeypatch.setattr(bfcl, "run_multi_turn_in_image", lambda model, url, conc, its, **kw: (
        bfcl.multi_turn_job(model, url, conc, [[it["id"], it["category"]] for it in its],
                            kw["temperature"], kw["thinking"], kw.get("attempts", 1))))
    # the multi-turn job runs in this process, not on the bridge network, so the network checks
    # before its container (tests/test_guard_code_images.py) do not apply
    monkeypatch.setattr(bfcl, "endpoint_network", lambda base_url: base_url)
    srv, url = _stub(items)
    try:
        with mock.patch.object(base, "_send_single_request", single):
            res = bfcl.run_bfcl("stub", url, 4, eval_categories="simple_python,multi_turn_base",
                                limit=4, max_output_tokens=256)
    finally:
        srv.shutdown()
    assert res["category_accuracy"]["simple_python"] == {"correct": 2, "total": 2, "accuracy": 100.0}
    assert res["category_accuracy"]["multi_turn_base"] == {"correct": 2, "total": 2, "accuracy": 100.0}
    assert res["total_questions"] == res["correct_answers"] == 4
    assert res["accuracy"] == res["pooled_accuracy"] == 100.0 and res["status"] == "success"
    assert res["bfcl_summary"]["multi_turn"]["Base"] == 100.0
    assert res["bfcl_summary"]["overall"]["Overall Acc"] is None
    assert sorted(t["sample_idx"] for t in res["sample_traces"]) == [0, 1, 2, 3]
    assert len({t["source_sample_idx"] for t in res["sample_traces"]}) == 4
    assert "Multi-Turn" in res["leaderboard_comparable_reason"]
    assert res["multi_turn_report"]["entries"] == 2


def test_bfcl_eval_writes_under_gbenchs_project_root_and_leaves_no_env_behind(tmp_path):
    import subprocess
    code = ("import os\n"
            "from gbench.runners.eval_suites import bfcl\n"
            "bfcl.load_multi_turn_entries(['multi_turn_base'])\n"
            "from bfcl_eval.constants import eval_config\n"
            "print(eval_config.PROJECT_ROOT, 'BFCL_PROJECT_ROOT' in os.environ)\n")
    env = {k: v for k, v in os.environ.items() if k != "BFCL_PROJECT_ROOT"}
    env["GBENCH_BFCL_PROJECT_ROOT"] = str(tmp_path)
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split()[-2:] == [str(tmp_path), "False"]
    assert (tmp_path / "result").is_dir() and (tmp_path / "score").is_dir()
