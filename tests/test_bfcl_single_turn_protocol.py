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

"""bfcl / bfcl_live follow BFCL's own single-turn protocol (bfcl_eval, pinned commit).

Request: dataset messages verbatim + convert_to_tool'ed docs with the language hint.
Scoring: the ported ast_checker (call count, types, Java/JS converters, standardize_string,
exact parameter names) and the relevance/irrelevance rule. Headline: BFCL's aggregation.
Expected verdicts below were checked against bfcl_eval's own ast_checker.
"""

import copy
import json
from unittest import mock

import pytest

from gbench.runners.eval_suites import base, bfcl, bfcl_live
from gbench.runners.eval_suites.fc_common import (
    BFCL_EVAL_COMMIT, bfcl_ast_checker, bfcl_convert_to_tools, bfcl_request_functions,
    bfcl_score, bfcl_standardize_string)


def _call(name, **args):
    return {"id": "c", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


WEATHER_DOC = [{
    "name": "get_current_weather",
    "description": "Current weather.",
    "parameters": {"type": "dict", "required": ["location"], "properties": {
        "location": {"type": "string", "description": "City, State."},
        "unit": {"type": "string", "description": "Unit.", "enum": ["celsius", "fahrenheit"]},
    }},
}]
WEATHER_GT = [{"get_current_weather": {"location": ["San Francisco, CA"], "unit": ["fahrenheit", ""]}}]


# --------------------------------------------------------------------------- #
# request format
# --------------------------------------------------------------------------- #

def test_convert_to_tool_matches_bfcl_openai_completions_style():
    docs = [{"name": "math.hypot", "description": "Hypotenuse.", "parameters": {
        "type": "dict", "required": ["x"], "properties": {
            "x": {"type": "float", "description": "Side."},
            "pts": {"type": "tuple", "items": {"type": "integer"}, "description": "Points."},
            "opts": {"type": "dict", "description": "Options.",
                     "properties": {"k": {"type": "float", "description": "K."}}},
        }}}]
    before = copy.deepcopy(docs)
    (tool,) = bfcl_convert_to_tools(docs)
    fn = tool["function"]
    assert tool["type"] == "function" and fn["name"] == "math_hypot"
    assert fn["parameters"]["type"] == "object"
    x = fn["parameters"]["properties"]["x"]
    assert x == {"type": "number", "format": "float", "description": "Side. This is a float type value."}
    assert fn["parameters"]["properties"]["pts"]["type"] == "array"
    assert fn["parameters"]["properties"]["pts"]["items"]["type"] == "integer"
    assert fn["parameters"]["properties"]["opts"]["properties"]["k"]["type"] == "number"
    assert docs == before                              # works on a copy


def test_language_hints_and_java_js_string_retyping():
    py = bfcl_request_functions(copy.deepcopy(WEATHER_DOC), "live_simple")
    assert py[0]["description"].endswith(" Note that the provided function is in Python 3 syntax.")
    assert py[0]["parameters"]["properties"]["location"]["type"] == "string"

    java_doc = [{"name": "Util.run", "description": "Run.", "parameters": {"type": "dict", "required": ["n", "xs"],
                 "properties": {"n": {"type": "long", "description": "N."},
                                "xs": {"type": "ArrayList", "items": {"type": "String"}, "description": "Xs."}}}}]
    java = bfcl_request_functions(java_doc, "simple_java")
    props = java[0]["parameters"]["properties"]
    assert java[0]["description"].endswith("Java 8 SDK syntax.")
    assert props["n"] == {"type": "string",
                          "description": "N. This is Java long type parameter in string representation."}
    assert "items" not in props["xs"] and props["xs"]["type"] == "string"
    assert "The list elements are of type String" in props["xs"]["description"]
    assert java_doc[0]["parameters"]["properties"]["n"]["type"] == "long"   # original untouched

    js_doc = [{"name": "f", "description": "F.", "parameters": {"type": "dict", "required": ["o"],
               "properties": {"o": {"type": "dict", "description": "O.",
                                    "properties": {"a": {"type": "String"}}}}}}]
    js = bfcl_request_functions(js_doc, "simple_javascript")
    o = js[0]["parameters"]["properties"]["o"]
    assert o["type"] == "string" and "properties" not in o
    assert '{"a": {"type": "String"}}' in o["description"]


def test_request_sends_dataset_messages_verbatim_without_injected_prompt():
    q = [[{"role": "system", "content": "Answer in French."}, {"role": "user", "content": "Weather in SF?"}]]
    messages, tools = bfcl.bfcl_request({"category": "live_simple", "question": q, "function": WEATHER_DOC})
    assert messages == q[0]                            # system kept, nothing added
    assert tools[0]["function"]["description"].endswith("Python 3 syntax.")


def test_default_categories_are_bfcl_scoring_categories_without_sql():
    cats = bfcl.resolve_categories(None)
    assert cats == ["simple_python", "simple_java", "simple_javascript", "multiple", "parallel",
                    "parallel_multiple", "irrelevance", "live_simple", "live_multiple",
                    "live_parallel", "live_parallel_multiple", "live_irrelevance", "live_relevance",
                    "multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param",
                    "multi_turn_long_context", "memory_kv", "memory_vector", "memory_rec_sum",
                    "web_search_base", "web_search_no_snippet"]
    assert bfcl.resolve_categories("simple,live") == ["simple_python"] + bfcl.LIVE_CATEGORIES
    with pytest.raises(ValueError, match="unknown category 'sql'"):
        bfcl.resolve_categories("sql")
    assert bfcl.resolve_categories("simple_python,multi_turn") == ["simple_python"] + bfcl.MULTI_TURN_CATEGORIES


# --------------------------------------------------------------------------- #
# ast_checker semantics
# --------------------------------------------------------------------------- #

def test_call_count_is_enforced():
    ok = [_call("get_current_weather", location="San Francisco, CA")]
    assert bfcl_score("simple_python", WEATHER_DOC, WEATHER_GT, tool_calls=ok)["valid"]
    extra = ok + [_call("get_current_weather", location="Boston, MA")]
    v = bfcl_score("simple_python", WEATHER_DOC, WEATHER_GT, tool_calls=extra)
    assert not v["valid"] and v["error_type"] == "simple_function_checker:wrong_count"
    # a duplicate of a correct call is still one call too many
    v = bfcl_score("live_multiple", WEATHER_DOC, WEATHER_GT, tool_calls=ok + ok)
    assert v["error_type"] == "multiple_function_checker:wrong_count"


def test_types_are_checked():
    doc = [{"name": "calc", "description": "", "parameters": {"type": "dict", "required": ["n"],
            "properties": {"n": {"type": "integer", "description": ""},
                           "r": {"type": "float", "description": ""},
                           "b": {"type": "boolean", "description": ""}}}}]
    gt = [{"calc": {"n": [5], "r": [2.0, ""], "b": [True, ""]}}]
    assert not bfcl_score("simple_python", doc, gt, tool_calls=[_call("calc", n="5")])["valid"]
    assert bfcl_score("simple_python", doc, gt, tool_calls=[_call("calc", n=5, r=2)])["valid"]   # int -> float
    assert not bfcl_score("simple_python", doc, gt, tool_calls=[_call("calc", n=5, b=1)])["valid"]
    assert not bfcl_score("simple_python", doc, gt, tool_calls=[_call("calc", n=5.0)])["valid"]


def test_strings_are_standardised_and_names_exact():
    assert bfcl_standardize_string("April 1, 2024") == bfcl_standardize_string("april 1 2024")
    for loc in ("san francisco CA", "San Francisco,CA", "SAN-FRANCISCO, ca"):
        assert bfcl_score("live_simple", WEATHER_DOC, WEATHER_GT,
                          tool_calls=[_call("get_current_weather", location=loc)])["valid"], loc
    assert not bfcl_score("live_simple", WEATHER_DOC, WEATHER_GT,
                          tool_calls=[_call("get_current_weather", location="Boston, MA")])["valid"]
    v = bfcl_score("live_simple", WEATHER_DOC, WEATHER_GT,
                   tool_calls=[_call("get_current_weather", Location="San Francisco, CA")])
    assert v["error_type"] == "simple_function_checker:missing_required"
    v = bfcl_score("live_simple", WEATHER_DOC, WEATHER_GT,
                   tool_calls=[_call("get_current_weather", location="San Francisco, CA", days=3)])
    assert v["error_type"] == "simple_function_checker:unexpected_param"


def test_dotted_function_name_is_expected_in_its_advertised_form():
    doc = [dict(WEATHER_DOC[0], name="weather.get")]
    gt = [{"weather.get": WEATHER_GT[0]["get_current_weather"]}]
    assert bfcl_score("live_simple", doc, gt, tool_calls=[_call("weather_get", location="San Francisco, CA")])["valid"]
    assert not bfcl_score("live_simple", doc, gt, tool_calls=[_call("weather.get", location="San Francisco, CA")])["valid"]


def test_parallel_is_order_free_but_exact_in_count():
    gt = [{"get_current_weather": {"location": ["Boston, MA"], "unit": [""]}},
          {"get_current_weather": {"location": ["San Francisco, CA"], "unit": [""]}}]
    a, b = (_call("get_current_weather", location="San Francisco, CA"),
            _call("get_current_weather", location="Boston, MA"))
    assert bfcl_score("parallel", WEATHER_DOC, gt, tool_calls=[a, b])["valid"]
    assert not bfcl_score("parallel", WEATHER_DOC, gt, tool_calls=[a])["valid"]
    assert not bfcl_score("parallel", WEATHER_DOC, gt, tool_calls=[a, b, b])["valid"]


def test_java_values_must_be_source_strings():
    doc = [{"name": "Svc.put", "description": "", "parameters": {"type": "dict", "required": ["n", "m"],
            "properties": {"n": {"type": "long", "description": ""},
                           "m": {"type": "HashMap", "description": ""},
                           "xs": {"type": "Array", "items": {"type": "integer"}, "description": ""}}}}]
    gt = [{"Svc.put": {"n": [5], "m": [{"k": ["v"]}], "xs": [[1, 2], ""]}}]
    good = _call("Svc_put", n="5L", m='new HashMap<String, String>() {{ put("k", "v"); }}',
                 xs="new int[]{1, 2}")
    assert bfcl_score("simple_java", doc, gt, tool_calls=[good])["valid"]
    v = bfcl_score("simple_java", doc, gt, tool_calls=[_call("Svc_put", n=5, m="new HashMap<>()")])
    assert not v["valid"] and v["error_type"] == "type_error:java"
    # "5" is not a Java long literal: it stays a string and fails the type check
    v = bfcl_score("simple_java", doc, gt, tool_calls=[_call("Svc_put", n="5", m='new HashMap<String, String>() {{ put("k", "v"); }}')])
    assert not v["valid"]


def test_javascript_values_must_be_source_strings():
    doc = [{"name": "sum", "description": "", "parameters": {"type": "dict", "required": ["xs", "flag"],
            "properties": {"xs": {"type": "array", "items": {"type": "integer"}, "description": ""},
                           "flag": {"type": "Boolean", "description": ""}}}}]
    gt = [{"sum": {"xs": [[1, 2, 3]], "flag": [True]}}]
    assert bfcl_score("simple_javascript", doc, gt, tool_calls=[_call("sum", xs="[1, 2, 3]", flag="true")])["valid"]
    assert not bfcl_score("simple_javascript", doc, gt, tool_calls=[_call("sum", xs=[1, 2, 3], flag=True)])["valid"]


def test_checker_matches_upstream_on_a_known_live_row():
    # live_simple_0-0-0 of BFCL_v4 (special is optional: "" among its accepted values)
    doc = [{"name": "get_user_info", "description": "Retrieve details for a specific user by their unique identifier.",
            "parameters": {"type": "dict", "required": ["user_id"], "properties": {
                "user_id": {"type": "integer", "description": "The unique identifier of the user."},
                "special": {"type": "string", "description": "Any special information.", "default": "none"}}}}]
    gt = [{"get_user_info": {"user_id": [7890], "special": ["black"]}}]
    assert bfcl_ast_checker(doc, [{"get_user_info": {"user_id": 7890, "special": "black"}}], gt, "live_simple")["valid"]
    v = bfcl_ast_checker(doc, [{"get_user_info": {"user_id": 7890}}], gt, "live_simple")
    assert v["error_type"] == "simple_function_checker:missing_optional"


# --------------------------------------------------------------------------- #
# relevance / irrelevance and decoding
# --------------------------------------------------------------------------- #

def test_irrelevance_is_correct_iff_no_call():
    assert bfcl_score("irrelevance", WEATHER_DOC, [], tool_calls=None, response_text="")["valid"]
    assert bfcl_score("live_irrelevance", WEATHER_DOC, [], response_text="I can't help with that.")["valid"]
    v = bfcl_score("live_irrelevance", WEATHER_DOC, [], tool_calls=[_call("get_current_weather", location="x")])
    assert not v["valid"] and v["error_type"] == "irrelevance_error:decoder_success"
    # an undecodable call is no call, exactly as BFCL decides it
    bad = [{"type": "function", "function": {"name": "get_current_weather", "arguments": "{not json"}}]
    assert bfcl_score("irrelevance", WEATHER_DOC, [], tool_calls=bad)["valid"]
    assert not bfcl_score("live_simple", WEATHER_DOC, WEATHER_GT, tool_calls=bad)["valid"]


def test_live_relevance_needs_a_call():
    assert bfcl_score("live_relevance", WEATHER_DOC, [], tool_calls=[_call("get_current_weather", location="x")])["valid"]
    v = bfcl_score("live_relevance", WEATHER_DOC, [], response_text="")
    assert not v["valid"] and v["error_type"] == "relevance_error:decoder_failed"


def test_reply_text_is_never_a_call():
    txt = '{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA"}}'
    assert not bfcl_score("live_simple", WEATHER_DOC, WEATHER_GT, response_text=txt)["valid"]
    echoed = '{"name": "get_current_weather", "parameters": {"type": "object"}}'
    assert not bfcl_score("live_simple", WEATHER_DOC, WEATHER_GT, response_text=echoed)["valid"]
    assert bfcl_score("live_irrelevance", WEATHER_DOC, [], response_text=echoed)["valid"]
    assert bfcl_score("live_irrelevance", WEATHER_DOC, [], response_text="I'm sorry (I cannot help).")["valid"]


# --------------------------------------------------------------------------- #
# batch scoring and aggregation
# --------------------------------------------------------------------------- #

def test_batch_scorer_scores_empty_irrelevance_replies_and_skips_failed_requests():
    import asyncio
    docs = {"irrelevance_0": WEATHER_DOC, "live_simple_0": WEATHER_DOC}
    traces = [
        {"gold_answer": {"id": "irrelevance_0", "category": "irrelevance", "ground_truth": []},
         "response_text": "", "tool_calls": None, "status": "EMPTY_RESPONSE", "is_correct": False},
        {"gold_answer": {"id": "live_simple_0", "category": "live_simple", "ground_truth": WEATHER_GT},
         "response_text": "r", "tool_calls": [_call("get_current_weather", location="San Francisco, CA")],
         "status": "PENDING_JUDGE", "health": "ok", "is_correct": False},
        {"gold_answer": {"id": "live_simple_0", "category": "live_simple", "ground_truth": WEATHER_GT},
         "response_text": None, "tool_calls": None, "status": "FAILED", "is_correct": False},
    ]
    asyncio.run(bfcl.bfcl_batch_scorer(docs, checker=bfcl.single_turn_verdicts)(traces))
    assert [t["is_correct"] for t in traces] == [True, True, False]
    assert traces[1]["status"] == "OK"
    assert "bfcl_error_type" not in traces[2]


def _cat(correct, total):
    return {"correct": correct, "total": total, "accuracy": round(correct / total * 100, 2)}


def test_summary_uses_bfcl_weighting():
    ca = {"simple_python": _cat(40, 40), "simple_java": _cat(0, 10), "simple_javascript": _cat(5, 10),
          "multiple": _cat(10, 20), "parallel": _cat(20, 20), "parallel_multiple": _cat(0, 20),
          "irrelevance": _cat(12, 24),
          "live_simple": _cat(10, 20), "live_multiple": _cat(60, 100), "live_parallel": _cat(1, 2),
          "live_parallel_multiple": _cat(0, 2), "live_irrelevance": _cat(80, 80), "live_relevance": _cat(1, 2)}
    s = bfcl.bfcl_summary(ca)
    assert s["non_live"]["Simple AST"] == 50.0                       # mean(100, 0, 50), unweighted
    assert s["non_live"]["Non-Live Overall Acc"] == 50.0             # mean(50, 50, 100, 0)
    assert s["live"]["Live Overall Acc"] == round(71 / 124 * 100, 2)  # weighted by entries
    assert s["overall"]["Irrelevance Detection"] == 75.0             # mean(50, 100)
    assert s["live"]["Relevance Detection"] == 50.0
    assert bfcl.overall_score(ca) is None                            # no multi-turn / agentic
    full = {**ca, "multi_turn_base": _cat(1, 2), "multi_turn_miss_func": _cat(0, 2),
            "multi_turn_miss_param": _cat(2, 2), "multi_turn_long_context": _cat(1, 2),
            "memory_kv": _cat(1, 1), "memory_vector": _cat(0, 1), "memory_rec_sum": _cat(1, 2),
            "web_search_base": _cat(1, 4), "web_search_no_snippet": _cat(3, 4)}
    agentic = (0.5 + 0.5) / 2                                        # mean(web mean, memory mean)
    expected = 0.1 * 0.5 + 0.1 * 71 / 124 + 0.1 * 0.75 + 0.3 * 0.5 + 0.4 * agentic
    assert bfcl.overall_score(full) == round(expected * 100, 2)
    partial = {k: v for k, v in full.items() if k != "simple_java"}
    assert bfcl.bfcl_summary(partial)["non_live"]["Non-Live Overall Acc"] is None
    assert bfcl.overall_score(partial) is None


# --------------------------------------------------------------------------- #
# loading and the end-to-end suite
# --------------------------------------------------------------------------- #

FIXTURE = {
    "simple_python": ([{"id": "simple_python_0", "question": [[{"role": "user", "content": "Weather in SF?"}]],
                        "function": WEATHER_DOC}],
                      [{"id": "simple_python_0", "ground_truth": WEATHER_GT}]),
    "irrelevance": ([{"id": "irrelevance_0", "question": [[{"role": "user", "content": "Area of a circle r=2?"}]],
                      "function": WEATHER_DOC}], None),
    "live_simple": ([{"id": "live_simple_0-0-0", "question": [[{"role": "system", "content": "Be brief."},
                                                               {"role": "user", "content": "SF weather?"}]],
                      "function": WEATHER_DOC}],
                    [{"id": "live_simple_0-0-0", "ground_truth": WEATHER_GT}]),
    "live_irrelevance": ([{"id": "live_irrelevance_0-0-0", "question": [[{"role": "user", "content": "Hi"}]],
                           "function": WEATHER_DOC}], None),
    "live_relevance": ([{"id": "live_relevance_0-0-0", "question": [[{"role": "user", "content": "SF?"}]],
                         "function": WEATHER_DOC}], None),
}


@pytest.fixture
def bfcl_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    root = tmp_path / "bfcl_data" / BFCL_EVAL_COMMIT
    (root / "possible_answer").mkdir(parents=True)
    for cat, (rows, answers) in FIXTURE.items():
        (root / f"BFCL_v4_{cat}.json").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        if answers is not None:
            (root / "possible_answer" / f"BFCL_v4_{cat}.json").write_text(
                "\n".join(json.dumps(a) for a in answers) + "\n")
    # The fixture files stand in for the pinned ones: pin their own blob ids.
    for f in root.rglob("*.json"):
        monkeypatch.setitem(bfcl._PINNED_BLOBS, f.relative_to(root).as_posix(),
                            bfcl._git_blob_sha1(f.read_bytes()))
    # The harness image's single-turn checker, run in this process on the installed bfcl-eval.
    monkeypatch.setattr(bfcl, "_harness_probe",
                        lambda name, multi, single: (bfcl.BFCL_EVAL_RELEASE, []))
    monkeypatch.setattr(bfcl, "check_single_turn",
                        lambda items, eval_name="bfcl": bfcl.single_turn_verdicts(items))
    return root


def test_loader_reads_the_pinned_files_and_joins_answers_by_id(bfcl_cache, monkeypatch):
    entries = bfcl.load_bfcl_entries(["simple_python", "irrelevance"], "bfcl")
    assert [(e["id"], e["category"]) for e in entries] == [("simple_python_0", "simple_python"),
                                                           ("irrelevance_0", "irrelevance")]
    assert entries[0]["ground_truth"] == WEATHER_GT and entries[1]["ground_truth"] == []
    other = bfcl_cache / "possible_answer" / "BFCL_v4_simple_python.json"
    other.write_text(json.dumps({"id": "other", "ground_truth": []}) + "\n")
    monkeypatch.setitem(bfcl._PINNED_BLOBS, "possible_answer/BFCL_v4_simple_python.json",
                        bfcl._git_blob_sha1(other.read_bytes()))
    with pytest.raises(RuntimeError, match="has no possible answer"):
        bfcl.load_bfcl_entries(["simple_python"], "bfcl")


def test_missing_data_is_a_hard_infra_error(tmp_path, monkeypatch):
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))

    def _offline(*a, **k):
        raise OSError("network unreachable")
    monkeypatch.setattr(bfcl.urllib.request, "urlopen", _offline)
    with pytest.raises(RuntimeError, match=r"\[INFRA REQUIRED\] bfcl_live.*docs/evals/bfcl_live\.md"):
        bfcl.load_bfcl_entries(["live_simple"], "bfcl_live")
    assert not list(tmp_path.rglob("*.json"))          # nothing partial was cached
    assert BFCL_EVAL_COMMIT in bfcl.BFCL_DATA_URL


def _fake_endpoint(sent):
    """A server that answers every function-calling prompt perfectly."""
    async def _reply(**kw):
        sent.append(kw)
        user = kw["messages"][-1]["content"]
        if user in ("Weather in SF?", "SF weather?", "SF?"):
            return base.Reply("", [_call("get_current_weather", location="San Francisco, CA")], "stop",
                              completion_tokens=12)
        return base.Reply("", None, "stop", completion_tokens=0)       # empty reply
    return _reply


def test_run_bfcl_end_to_end(bfcl_cache, monkeypatch):
    monkeypatch.setattr(bfcl, "SUPPORTED_BFCL_CATEGORIES", list(FIXTURE))
    sent = []
    with mock.patch.object(base, "_send_single_request", _fake_endpoint(sent)):
        res = bfcl.run_bfcl("m", "http://x", 1, eval_categories=",".join(FIXTURE))
    assert res["correct_answers"] == res["total_questions"] == 5
    live_req = next(k for k in sent if k["messages"][0]["role"] == "system")
    assert live_req["messages"][0]["content"] == "Be brief."            # dataset system kept
    assert live_req["extra_payload"]["tools"][0]["function"]["description"].endswith("Python 3 syntax.")
    assert all(not m["content"].startswith("You are an expert") for k in sent for m in k["messages"])
    assert res["bfcl_summary"]["live"]["Relevance Detection"] == 100.0
    assert res["leaderboard_comparable"] is False
    assert "Multi-Turn" in res["leaderboard_comparable_reason"]
    assert res["dataset_revision"].startswith(f"ShishirPatil/gorilla@{BFCL_EVAL_COMMIT}")


def test_run_bfcl_live_headline_is_live_overall_acc(bfcl_cache, monkeypatch):
    monkeypatch.setattr(bfcl_live, "LIVE_CATEGORIES", ["live_simple", "live_irrelevance", "live_relevance"])
    sent = []
    with mock.patch.object(base, "_send_single_request", _fake_endpoint(sent)):
        res = bfcl_live.run_bfcl_live("m", "http://x", 1)
    live = res["bfcl_summary"]["live"]
    # Live Overall Acc needs all four live AST categories; this fixture has one
    assert live["Live Overall Acc"] is None and live["Python Simple AST"] == 100.0
    assert live["Irrelevance Detection"] == 100.0                      # the empty reply is correct
    assert res["leaderboard_comparable"] is False
    assert "Live Overall Acc" in res["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# BFCL's FC rule end to end, data integrity, report contract
# --------------------------------------------------------------------------- #

def test_strict_rule_reads_reply_text_as_no_call_like_bfcls_fc_handler():
    txt = '{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA"}}'
    strict = {}
    v = bfcl_score("live_simple", WEATHER_DOC, WEATHER_GT, response_text=txt, **strict)
    assert not v["valid"] and v["error_type"] == "ast_decoder:decoder_failed"
    # an empty reply is an empty call list: a wrong call count, not a decode failure
    v = bfcl_score("live_simple", WEATHER_DOC, WEATHER_GT, response_text="", **strict)
    assert not v["valid"] and v["error_type"] == "simple_function_checker:wrong_count"
    # irrelevance: a JSON example inside a refusal is no call under BFCL's rule
    refusal = 'I cannot. Example: {"name": "f", "arguments": {}}'
    assert bfcl_score("live_irrelevance", WEATHER_DOC, [], response_text=refusal, **strict)["valid"]
    assert not bfcl_score("live_relevance", WEATHER_DOC, [], response_text=txt, **strict)["valid"]
    # structured tool calls are scored identically by both rules
    ok = [_call("get_current_weather", location="San Francisco, CA")]
    assert bfcl_score("live_simple", WEATHER_DOC, WEATHER_GT, tool_calls=ok, **strict)["valid"]


def test_batch_scorer_applies_bfcls_fc_rule():
    import asyncio
    txt = '{"name": "get_current_weather", "arguments": {"location": "San Francisco, CA"}}'
    docs = {"live_simple_0": WEATHER_DOC, "live_irrelevance_0": WEATHER_DOC}
    traces = [
        {"gold_answer": {"id": "live_simple_0", "category": "live_simple", "ground_truth": WEATHER_GT},
         "response_text": txt, "tool_calls": None, "status": "PENDING_JUDGE", "health": "ok",
         "is_correct": False, "category": "live_simple"},
        {"gold_answer": {"id": "live_irrelevance_0", "category": "live_irrelevance", "ground_truth": []},
         "response_text": "<thought>x</thought>" + txt, "tool_calls": None, "status": "PENDING_JUDGE",
         "health": "ok", "is_correct": False, "category": "live_irrelevance"},
    ]
    asyncio.run(bfcl.bfcl_batch_scorer(docs, checker=bfcl.single_turn_verdicts)(traces))
    assert [t["is_correct"] for t in traces] == [False, True]
    assert traces[0]["bfcl_error_type"] == "ast_decoder:decoder_failed"


def _text_json_endpoint(sent):
    """Answers the SF prompts with the call as JSON TEXT (no structured tool calls)."""
    async def _reply(**kw):
        sent.append(kw)
        user = kw["messages"][-1]["content"]
        if user in ("Weather in SF?", "SF weather?", "SF?"):
            return base.Reply('{"name": "get_current_weather", "arguments": '
                              '{"location": "San Francisco, CA"}}', None, "stop", completion_tokens=9)
        return base.Reply("", None, "stop", completion_tokens=0)
    return _reply


def test_run_bfcl_headline_reads_text_json_as_no_call(bfcl_cache, monkeypatch):
    monkeypatch.setattr(bfcl, "SUPPORTED_BFCL_CATEGORIES", list(FIXTURE))
    with mock.patch.object(base, "_send_single_request", _text_json_endpoint([])):
        res = bfcl.run_bfcl("m", "http://x", 1, eval_categories=",".join(FIXTURE))
    # BFCL's FC rule: the 3 call entries answered with JSON text fail; the 2 irrelevance pass
    assert res["correct_answers"] == 2 and res["accuracy"] == 40.0   # pooled: 2 of 5
    assert res["bfcl_summary"]["live"]["Relevance Detection"] == 0.0
    assert res["bfcl_summary"]["live"]["Irrelevance Detection"] == 100.0
    assert "strict_accuracy" not in res and "text_json_verdicts" not in res


def test_run_bfcl_live_reads_text_json_as_no_call(bfcl_cache, monkeypatch):
    monkeypatch.setattr(bfcl_live, "LIVE_CATEGORIES", ["live_simple", "live_irrelevance", "live_relevance"])
    seen = {}
    real = base.run_eval_suite

    def spy(**kw):
        seen["max_output_tokens"] = kw.get("max_output_tokens")
        return real(**kw)
    monkeypatch.setattr(bfcl_live, "run_eval_suite", spy)
    with mock.patch.object(base, "_send_single_request", _text_json_endpoint([])):
        res = bfcl_live.run_bfcl_live("m", "http://x", 1)
    assert seen["max_output_tokens"] is None                          # no suite default of its own
    assert res["bfcl_summary"]["live"]["Python Simple AST"] == 0.0
    assert res["accuracy"] == round(1 / 3 * 100, 2)                  # pooled: the irrelevance entry


def test_every_data_file_the_loader_reads_is_blob_pinned():
    wanted = set()
    for cat in bfcl.SINGLE_TURN_CATEGORIES:      # multi-turn files are checked inside bfcl-eval
        wanted.add(f"BFCL_v4_{cat}.json")
        if "relevance" not in cat:
            wanted.add(f"possible_answer/BFCL_v4_{cat}.json")
    assert set(bfcl._PINNED_BLOBS) == wanted and len(wanted) == 23
    assert all(len(b) == 40 and int(b, 16) >= 0 for b in bfcl._PINNED_BLOBS.values())


class _Resp:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self.data


def test_a_tampered_cached_file_is_never_scored(bfcl_cache, monkeypatch):
    target = bfcl_cache / "possible_answer" / "BFCL_v4_simple_python.json"
    good = target.read_bytes()
    target.write_text(json.dumps({"id": "simple_python_0", "ground_truth": [{"x": {}}]}) + "\n")
    calls = []

    def _offline(*a, **k):
        calls.append(a)
        raise OSError("network unreachable")
    monkeypatch.setattr(bfcl.urllib.request, "urlopen", _offline)
    with pytest.raises(RuntimeError, match=r"\[INFRA REQUIRED\] bfcl.*does not match the pinned file"):
        bfcl.load_bfcl_entries(["simple_python"], "bfcl")
    assert calls, "a mismatching cache must be fetched again, not read"
    # online: the pinned bytes replace the tampered copy
    monkeypatch.setattr(bfcl.urllib.request, "urlopen", lambda *a, **k: _Resp(good))
    entries = bfcl.load_bfcl_entries(["simple_python"], "bfcl")
    assert entries[0]["ground_truth"] == WEATHER_GT and target.read_bytes() == good


def test_a_download_that_does_not_match_the_pin_is_a_hard_infra_error(tmp_path, monkeypatch):
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(bfcl.urllib.request, "urlopen",
                        lambda *a, **k: _Resp(b'{"id": "live_relevance_0", "question": [], "function": []}\n'))
    with pytest.raises(RuntimeError, match=r"\[INFRA REQUIRED\] bfcl_live.*does not match"):
        bfcl.load_bfcl_entries(["live_relevance"], "bfcl_live")
    assert not list(tmp_path.rglob("*.json"))


def _through_reports(res, name, tmp_path, monkeypatch):
    """Send a suite result through the real evals.py result block and the cli summaries."""
    from types import SimpleNamespace
    from gbench import cli
    from gbench.runners import evals as E
    monkeypatch.setattr(E, "_resolve_remote_model_id", lambda url, fallback: "m")
    monkeypatch.setattr(E, "_server_max_model_len", lambda url: None)
    monkeypatch.setattr(E.EvalsBenchmarkRunner, "_run_single_eval", lambda self, eval_name, **k: dict(res))
    cfg = SimpleNamespace(evals=[name], eval_custom_jsonl=None, eval_plugins_dir=None, batch_sizes=[1],
                          remote_endpoint="http://127.0.0.1:1/v1", tokenizer="t",
                          eval_max_output_tokens=1024, skip_existing=False,
                          log_manager=SimpleNamespace(results_dir=tmp_path, base_dir=tmp_path,
                                                      save_result=lambda p, r: None))
    model = SimpleNamespace(name="m", short_name="m", hf_model_id="org/m")
    out = E.EvalsBenchmarkRunner(cfg).run(model, SimpleNamespace(value="remote-endpoint"))
    assert [r["eval_name"] for r in out] == [name] and out[0].get("status") != "error"
    cli.print_results_summary(out, SimpleNamespace(results_dir=tmp_path))
    csv = cli._save_eval_summary_csv(tmp_path, out, [], [("5. Tool Use", [name])], {})
    assert csv is not None and name.upper() in csv.read_text()
    return out[0]


def _assert_category_shape(res):
    ca = res["category_accuracy"]
    assert 1 < len(ca) <= 25, "the per-category report block must actually run"
    for cat, st in ca.items():
        assert isinstance(st, dict), cat
        assert isinstance(st["correct"], int) and isinstance(st["total"], int), cat
        assert isinstance(st["accuracy"], (int, float)), cat


def test_bfcl_results_pass_through_the_report_code(bfcl_cache, monkeypatch, tmp_path):
    monkeypatch.setattr(bfcl, "SUPPORTED_BFCL_CATEGORIES", list(FIXTURE))
    monkeypatch.setattr(bfcl_live, "LIVE_CATEGORIES", ["live_simple", "live_irrelevance", "live_relevance"])
    with mock.patch.object(base, "_send_single_request", _fake_endpoint([])):
        r1 = bfcl.run_bfcl("m", "http://x", 1, eval_categories=",".join(FIXTURE))
        r2 = bfcl_live.run_bfcl_live("m", "http://x", 1)
    for res, name in ((r1, "bfcl"), (r2, "bfcl_live")):
        _assert_category_shape(res)
        _through_reports(res, name, tmp_path, monkeypatch)


def test_empty_irrelevance_replies_are_answers_not_failed_requests(bfcl_cache, monkeypatch):
    """End to end: the fake server answers both irrelevance entries with an empty reply (the
    right answer); base must not count them as failed requests."""
    monkeypatch.setattr(bfcl, "SUPPORTED_BFCL_CATEGORIES", list(FIXTURE))
    with mock.patch.object(base, "_send_single_request", _fake_endpoint([])):
        res = bfcl.run_bfcl("m", "http://x", 1, eval_categories=",".join(FIXTURE))
    empties = [t for t in res["sample_traces"] if t.get("empty_is_answer")]
    assert sorted(t["category"] for t in empties) == ["irrelevance", "live_irrelevance"]
    assert res["empty_responses"] == 0 and res["failed_requests"] == 0
    assert res["status"] == "success"
