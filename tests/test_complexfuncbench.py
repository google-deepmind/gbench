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

"""Deterministic unit tests for the complexfuncbench canonical harness.

These exercise the ported pieces that do NOT need the endpoint, the Gemini judge, or the
embedding download: the max-weight assignment, the match cascade's rule/value tiers, the
judge reply parsers (upstream decode_json), and the full golden-driven loop with a scripted `chat` that replays
golden calls (exact replays hit rule_based + the exact-match path in mapping_call, so no
judge/embedder is touched)."""

import asyncio
import json
from itertools import permutations

import numpy as np
import pytest

from gbench.runners.eval_suites import complexfuncbench as C


# --------------------------------------------------------------------------- #
# _assign_max vs brute-force optimum.
# --------------------------------------------------------------------------- #
def _bruteforce_max(m):
    n, k = m.shape
    q = min(n, k)
    best = None
    if n <= k:
        for perm in permutations(range(k), n):
            s = sum(m[i, perm[i]] for i in range(n))
            best = s if best is None else max(best, s)
    else:
        for perm in permutations(range(n), k):
            s = sum(m[perm[j], j] for j in range(k))
            best = s if best is None else max(best, s)
    return best


@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 3), (2, 3), (3, 2), (4, 4), (1, 5), (5, 1)])
def test_assign_max_is_optimal(shape):
    rng = np.random.default_rng(shape[0] * 10 + shape[1])
    m = rng.random(shape)
    r, c = C._assign_max(m)
    assert len(r) == len(c) == min(shape)
    assert len(set(r)) == len(r) and len(set(c)) == len(c)      # injective
    got = sum(m[i, j] for i, j in zip(r, c))
    assert abs(got - _bruteforce_max(m)) < 1e-9


def test_assign_max_empty():
    assert C._assign_max(np.zeros((0, 3))) == ([], [])


# --------------------------------------------------------------------------- #
# Judge parsers.
# --------------------------------------------------------------------------- #
def test_parse_is_equal():
    # upstream utils.decode_json + llm_based: a JSON object's is_equal, fences stripped; an
    # anchored text line is not a verdict upstream can read
    assert C._parse_is_equal("reasoning...\nis_equal: true") is None
    assert C._parse_is_equal('```JSON\n{"is_equal": true, "reason": "x"}\n```') is True
    assert C._parse_is_equal('```json\n{"is_equal": False, "reason": "x"}\n```') is False
    assert C._parse_is_equal('{"is_equal": false}') is False
    assert C._parse_is_equal('{"reason": "no verdict"}') is None
    assert C._parse_is_equal("no verdict here") is None


def test_parse_score():
    assert C._parse_score("blah\nscore: 2") is None
    assert C._parse_score('{"score": 1, "reason": "partial"}') == {"score": 1, "reason": "partial"}
    assert C._parse_score('```JSON\n{"score": 0, "reason": "none"}\n```')["score"] == 0
    assert C._parse_score('{"score": 3}') is None
    assert C._parse_score("garbage") is None


# --------------------------------------------------------------------------- #
# Match cascade: rule_based + value_checker (deterministic tiers).
# --------------------------------------------------------------------------- #
class _LiveAPI:
    """A stand-in for the booking-com15 API of the response-based tier: each call answers its
    own response, so two calls that differ get different responses (the judge decides)."""

    def __init__(self):
        from collections import Counter
        self.stats = Counter()

    async def call(self, func_call):
        return {"response": {"echo": json.dumps(func_call, sort_keys=True)}, "raised": False,
                "labels": ["ok"]}


def _cmp():
    return C._CompareFC(C._load_exact_match_dict(), C._get_embedder(), None, _LiveAPI())


def test_rule_based():
    cmp = _cmp()
    a = {"name": "Search_Foo", "arguments": {"q": "x", "n": 1}}
    b = {"name": "Search_Foo", "arguments": {"n": 1, "q": "x"}}   # order-independent
    assert cmp.rule_based(a, b) is True
    assert cmp.rule_based(a, {"name": "Search_Foo", "arguments": {"q": "y", "n": 1}}) is False
    assert cmp.rule_based(a, {"name": "Other", "arguments": {"q": "x", "n": 1}}) is False


def test_rule_based_categories_filter_setwise():
    cmp = _cmp()
    a = {"name": "F", "arguments": {"categories_filter": "a,b, c"}}
    b = {"name": "F", "arguments": {"categories_filter": "c,a,b"}}
    assert cmp.rule_based(a, b) is True


def test_value_checker_critical_param():
    cmp = _cmp()
    # Search_Car_Rentals has critical params in exact_match_values.json (incl. pick_up_date).
    assert "Search_Car_Rentals" in cmp.exact_match_dict
    pk = cmp.exact_match_dict["Search_Car_Rentals"][0]
    golden = {"name": "Search_Car_Rentals", "arguments": {pk: "GOLD"}}
    ok, _ = cmp.value_checker({"name": "Search_Car_Rentals", "arguments": {pk: "GOLD"}}, golden)
    assert ok is True
    bad, msg = cmp.value_checker({"name": "Search_Car_Rentals", "arguments": {pk: "WRONG"}}, golden)
    assert bad is False and msg["error_type"] == "value_error"
    miss, msg2 = cmp.value_checker({"name": "Search_Car_Rentals", "arguments": {}}, golden)
    assert miss is False and msg2["error_type"] == "param_missing"


def test_format_check():
    cmp = _cmp()
    funcs = [{"name": "F", "parameters": {"required": ["a"], "properties": {"a": {"type": "string"}}}}]
    assert cmp.format_check({"name": "F", "arguments": {"a": "x"}}, funcs) is True
    assert "error" in cmp.format_check({"name": "F", "arguments": {}}, funcs)          # missing req
    assert "error" in cmp.format_check({"name": "G", "arguments": {}}, funcs)          # unknown fn
    assert "error" in cmp.format_check({"name": "F", "arguments": {"a": 3}}, funcs)    # wrong type


# --------------------------------------------------------------------------- #
# mapping_call: exact-match path aligns swapped parallel calls WITHOUT the embedder.
# --------------------------------------------------------------------------- #
def test_mapping_call_exact_swapped():
    cmp = _cmp()
    callA = {"name": "A", "arguments": {"x": 1}}
    callB = {"name": "B", "arguments": {"y": 2}}
    matches = cmp.mapping_call([dict(callB), dict(callA)], [dict(callA), dict(callB)],
                               ["obsA", "obsB"])
    by_pred = {m["idx"]: m for m in matches}
    assert by_pred[0]["golden_obs"] == "obsB"   # predicted[0]==callB -> golden B's obs
    assert by_pred[1]["golden_obs"] == "obsA"


# --------------------------------------------------------------------------- #
# Full golden-driven loop with a scripted chat (exact replay -> Success.).
# --------------------------------------------------------------------------- #
def _tool_call(cid, name, args):
    return {"id": cid, "function": {"name": name, "arguments": json.dumps(args)}}


def _item_two_step():
    return {
        "id": "Cross-0",
        "functions": [
            {"name": "Search_Foo", "parameters": {"required": ["q"], "properties": {"q": {"type": "string"}}}},
            {"name": "Get_Bar", "parameters": {"required": ["id"], "properties": {"id": {"type": "number"}}}},
        ],
        "conversations": [
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "function_call": [{"name": "Search_Foo", "arguments": {"q": "x"}}]},
            {"role": "observation", "content": [{"status": True, "data": "foo-obs"}]},
            {"role": "assistant", "function_call": [{"name": "Get_Bar", "arguments": {"id": 5}}]},
            {"role": "observation", "content": [{"status": True, "data": "bar-obs"}]},
            {"role": "assistant", "content": "here is your final answer"},
        ],
    }


def _run(item, script):
    """Drive a runner with a scripted list of assistant messages (dicts)."""
    runner = C._Runner(_cmp())
    state = {"i": 0}

    async def chat(oai_messages, tools):
        msg = script[state["i"]]
        state["i"] += 1
        return msg

    return asyncio.run(runner.run(item, chat))


def test_full_loop_success_exact_replay():
    item = _item_two_step()
    script = [
        {"content": None, "tool_calls": [_tool_call("c1", "Search_Foo", {"q": "x"})]},
        {"content": None, "tool_calls": [_tool_call("c2", "Get_Bar", {"id": 5})]},
        {"content": "here is your final answer", "tool_calls": None},
    ]
    convs, message, turn_id, correct = _run(item, script)
    assert message == "Success."
    assert turn_id == 2            # both golden steps completed
    assert correct == 2            # both calls matched
    assert convs[-1]["content"] == "here is your final answer"


def test_full_loop_wrong_first_call_fails():
    item = _item_two_step()
    # Patch the judge so a non-rule/value match deterministically returns "not equal".
    async def _no(*a, **k):
        return False
    cmp = _cmp()
    cmp.llm_based = _no  # type: ignore
    runner = C._Runner(cmp)
    script = [
        {"content": None, "tool_calls": [_tool_call("c1", "Search_Foo", {"q": "WRONG"})]},
    ]
    state = {"i": 0}

    async def chat(oai_messages, tools):
        m = script[state["i"]]; state["i"] += 1; return m

    convs, message, turn_id, correct = asyncio.run(runner.run(item, chat))
    assert message != "Success."
    assert correct == 0


def test_full_loop_hallucinated_extra_call_after_done():
    """Model keeps calling after the golden chain is exhausted -> func_hallucination."""
    item = {
        "id": "Cross-1",
        "functions": [
            {"name": "Search_Foo", "parameters": {"required": ["q"], "properties": {"q": {"type": "string"}}}},
        ],
        "conversations": [
            {"role": "user", "content": "one call then stop"},
            {"role": "assistant", "function_call": [{"name": "Search_Foo", "arguments": {"q": "x"}}]},
            {"role": "observation", "content": [{"status": True, "data": "obs"}]},
            {"role": "assistant", "content": "done"},
        ],
    }
    script = [
        {"content": None, "tool_calls": [_tool_call("c1", "Search_Foo", {"q": "x"})]},
        {"content": None, "tool_calls": [_tool_call("c2", "Search_Foo", {"q": "again"})]},
    ]
    convs, message, turn_id, correct = _run(item, script)
    assert isinstance(message, dict) and message["error_type"] == "func_hallucination"
    assert correct == 1   # the first (correct) call still counted


@pytest.mark.slow
def test_embedder_reproduction_and_multicall_alignment():
    """Validate the bge reproduction (normalised CLS embeddings) + a genuine 2x2 parallel-call
    alignment that actually exercises the embedder. Downloads bge-large-en-v1.5 once."""
    emb = C._get_embedder()
    vecs = emb.encode(["book a flight from Chicago to Tokyo", "book a flight from Chicago to Tokyo",
                       "reserve a hotel room in Milan"])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3)                 # L2-normalised
    assert vecs.shape[1] == 1024                              # bge-large hidden size
    sim_same = float(vecs[0] @ vecs[1])
    sim_diff = float(vecs[0] @ vecs[2])
    assert sim_same > 0.999                                   # identical strings
    assert sim_same > sim_diff                                # and more similar than a different one

    # 2x2 alignment: two parallel calls that don't exact-match; the embedder must pair each
    # predicted call with the golden call it most resembles.
    cmp = _cmp()
    g1 = {"name": "Search_Flights", "arguments": {"fromId": "ORD", "toId": "HND"}}
    g2 = {"name": "Search_Hotels", "arguments": {"city": "Milan"}}
    p1 = {"name": "Search_Flights", "arguments": {"fromId": "ORD", "toId": "TYO"}}   # ~ g1
    p2 = {"name": "Search_Hotels", "arguments": {"city": "Milano"}}                  # ~ g2
    matches = cmp.mapping_call([p1, p2], [g1, g2], ["obs_flight", "obs_hotel"])
    by_pred = {m["idx"]: m for m in matches}
    assert by_pred[0]["golden_obs"] == "obs_flight"
    assert by_pred[1]["golden_obs"] == "obs_hotel"


def test_aggregate_metrics():
    results = [
        {"id": "Cross-0", "request_failed": False, "message": "Success.",
         "count_dict": {"success_turn_num": 2, "total_turn_num": 2, "correct_call_num": 2,
                        "total_call_num": 2, "real_turn_num": 2},
         "resp_eval": {"complete": {"score": 2}, "correct": {"score": 1}}},
        {"id": "Hotels-0", "request_failed": False, "message": {"error_type": "stop_early"},
         "count_dict": {"success_turn_num": 1, "total_turn_num": 3, "correct_call_num": 1,
                        "total_call_num": 4, "real_turn_num": 1},
         "resp_eval": {"complete": {"score": 0}, "correct": {"score": 0}}},
        {"id": "Flights-0", "request_failed": True, "message": {"error_type": "unknown_error"}},
    ]
    agg = C._aggregate(results, n_total=2, is_full=False)
    assert agg["n_scored"] == 2
    assert agg["n_request_failed"] == 1
    assert agg["n_success"] == 1
    assert agg["success_rate"] == 50.0                 # 1 success / 2 scored
    assert agg["call_accuracy"] == round(3 / 6 * 100, 2)  # (2+1)/(2+4)
    assert agg["completeness"] == 1.0                  # mean(2, 0)
    assert agg["correctness"] == 0.5                   # mean(1, 0)
