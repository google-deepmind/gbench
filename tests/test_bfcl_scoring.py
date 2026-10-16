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

"""bfcl regression: score off the STRUCTURED tool calls, not base.py's lossy text render.

Each case is a real sample that the model answered correctly but the old text-parse
scorer marked wrong (0/5) because the `name({json}) name(a=1)` render corrupts values."""

from gbench.runners.eval_suites.bfcl import _eval_bfcl
from gbench.runners.eval_suites.base import _apply_eval_fn, _eval_fn_accepts_tool_calls


def _tc(name, args):
    return [{"type": "function", "function": {"name": name, "arguments": args}}]


def test_unquoted_value_with_space_preserved():
    # text render dropped the space: 'J.K. Rowling' -> 'j.k.rowling'
    gold = [{"filterBooksByAuthor": {"library": [["bookA", "bookB", "bookC"]], "author": ["J.K. Rowling"]}}]
    tc = _tc("filterBooksByAuthor", {"author": "J.K. Rowling", "library": ["bookA", "bookB", "bookC"]})
    assert _eval_bfcl("", gold, tool_calls=tc)


def test_unquoted_string_with_comma_preserved():
    # text render split 'San Jose, CA' into two args
    gold = [{"Services_4_FindProvider": {"city": ["San Jose, CA"], "type": ["Psychologist"],
                                         "insurance_accepted": ["", True]}}]
    tc = _tc("Services_4_FindProvider", {"city": "San Jose, CA", "type": "Psychologist"})
    assert _eval_bfcl("", gold, tool_calls=tc)


def test_namespaced_function_name_and_string_id():
    # text render truncated 'CustomDashboardsApi.get_...' to the attr, and '12345' matched str==int.
    # BFCL's convert_to_tool advertises the tool as 'CustomDashboardsApi_get_...' (OpenAI names
    # allow no dots) and its checker expects that name back.
    gold = [{"CustomDashboardsApi.get_shareable_api_tokens": {"user_id": ["12345"],
                                                              "include_revoked": ["", False]}}]
    tc = _tc("CustomDashboardsApi_get_shareable_api_tokens", {"user_id": "12345"})
    assert _eval_bfcl("", gold, tool_calls=tc)


def test_nested_dict_param():
    # nested possible_answer: profile_data is a {subparam: [accepted]} object; bio is optional ("")
    gold = [{"update_user_profile": {"user_id": [12345],
                                     "profile_data": [{"email": ["john.doe@example.com"], "age": [30], "bio": [""]}],
                                     "notify": ["", True]}}]
    tc = _tc("update_user_profile", {"profile_data": {"age": 30, "email": "john.doe@example.com"}, "user_id": 12345})
    assert _eval_bfcl("", gold, tool_calls=tc)


def test_wrong_function_is_still_wrong():
    # sample 0: the model called a different function -> genuinely incorrect
    gold = [{"linear_regression": {"independent_var": [["interest_rate", "unemployment_rate"]],
                                   "dependent_var": ["house_price"], "forecast_period": [5]}}]
    tc = _tc("random_forest_regression", {"dependent_var": "house_price", "forecast_period": 5,
                                          "independent_var": ["interest_rate", "unemployment_rate"]})
    assert not _eval_bfcl("", gold, tool_calls=tc)


def test_base_apply_eval_fn_gating():
    # _eval_bfcl opts in to structured calls; a 2-arg eval_fn does not and is called unchanged.
    assert _eval_fn_accepts_tool_calls(_eval_bfcl) is True
    assert _eval_fn_accepts_tool_calls(lambda p, g: False) is False
    gold = [{"filterBooksByAuthor": {"library": [["bookA"]], "author": ["J.K. Rowling"]}}]
    tc = _tc("filterBooksByAuthor", {"author": "J.K. Rowling", "library": ["bookA"]})
    assert _apply_eval_fn(_eval_bfcl, "", gold, tc)                        # structured path
    assert _apply_eval_fn(lambda p, g: p == "x", "x", "gold", None) is True  # 2-arg path unaffected


# --- bfcl_live: scored off the structured tool calls ---

def _call(name, **args):
    import json as _json
    return {"type": "function", "function": {"name": name, "arguments": _json.dumps(args)}}


def test_v3_live_structured_call_with_comma_in_argument_passes():
    from gbench.runners.eval_suites.base import _apply_eval_fn
    from gbench.runners.eval_suites.bfcl_live import _eval_bfcl_live
    gold = [{"get_current_weather": {"location": ["San Francisco, CA"], "unit": ["fahrenheit", ""]}}]
    tc = [_call("get_current_weather", location="San Francisco, CA", unit="fahrenheit")]
    text = 'get_current_weather({"location": "San Francisco, CA", "unit": "fahrenheit"})'
    # base.py only passes tool_calls to scorers that declare the parameter.
    assert _apply_eval_fn(_eval_bfcl_live, text, gold, tc) is True


def test_v3_live_wrong_value_and_extra_call_still_fail():
    from gbench.runners.eval_suites.bfcl_live import _eval_bfcl_live
    gold = [{"get_current_weather": {"location": ["San Francisco, CA"]}}]
    assert not _eval_bfcl_live("", gold, tool_calls=[_call("get_current_weather", location="Boston, MA")])
    assert not _eval_bfcl_live("", gold, tool_calls=[
        _call("get_current_weather", location="San Francisco, CA"), _call("send_email", to="x")])


def test_v3_live_irrelevance_refusal_is_correct_and_any_call_is_wrong():
    from gbench.runners.eval_suites.bfcl_live import _eval_bfcl_live
    refusal = "I'm sorry (I cannot help with that request)."
    assert _eval_bfcl_live(refusal, [], tool_calls=None) is True
    assert _eval_bfcl_live("", [], tool_calls=[_call("get_weather", city="Paris")]) is False
    # No structured calls: BFCL's FC handler reads any reply text, JSON call or not, as no call.
    assert _eval_bfcl_live('{"name": "get_weather", "arguments": {"city": "Paris"}}', [],
                              tool_calls=None) is True


def test_v3_live_camelcase_parameter_keys_still_match():
    """Review finding: structured calls keep the model's key case, and the shared checker
    looked them up lower-cased, so every camelCase parameter row failed (36 regressions)."""
    from gbench.runners.eval_suites.bfcl_live import _eval_bfcl_live
    gold = [{"ChaFod": {"TheFod": ["BURGER"]}}]
    assert _eval_bfcl_live("", gold, tool_calls=[_call("ChaFod", TheFod="BURGER")])


def test_v3_live_json_answer_without_arguments_is_not_a_call():
    from gbench.runners.eval_suites.bfcl_live import _eval_bfcl_live
    assert _eval_bfcl_live('{"name": "Ada Lovelace", "born": 1815}', [], tool_calls=None)
    assert _eval_bfcl_live('{"name": "f", "arguments": {"x": 1}}', [], tool_calls=None)   # text: no call


def test_v3_live_structured_keys_match_exactly():
    """Review finding: lower-casing both sides accepted wrong-case parameter names, which
    BFCL's checker rejects. Structured calls now match keys exactly."""
    from gbench.runners.eval_suites.bfcl_live import _eval_bfcl_live
    gold = [{"ChaFod": {"TheFod": ["BURGER"]}}]
    assert _eval_bfcl_live("", gold, tool_calls=[_call("ChaFod", TheFod="BURGER")])
    assert not _eval_bfcl_live("", gold, tool_calls=[_call("ChaFod", thefod="BURGER")])


def test_v3_live_echoed_tool_definition_is_not_a_call_and_empty_irrelevance_is_correct():
    from gbench.runners.eval_suites.bfcl_live import _eval_bfcl_live
    echoed = '{"name": "get_weather", "parameters": {"type": "object", "properties": {}}}'
    assert _eval_bfcl_live(echoed, [], tool_calls=None)
    assert _eval_bfcl_live("", [], tool_calls=None)
