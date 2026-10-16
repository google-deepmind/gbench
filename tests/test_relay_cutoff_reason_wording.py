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
"""The reason relay_result_reasons gives for agent turns cut off at a limit names that limit.

Incident, 2026-10-11 validation run (pool-10-11-132525): since the relay sends the run's
--max-output-tokens as max_tokens with every agent request (swe_agentic.with_run_max_tokens;
relay record `sets_body_fields.max_tokens` 65536, `max_tokens_from_run.requests_filled` 0),
multi_swe_bench's 2 cut-off turns and swe_bench_live's tabs were still worded "cut off at the end
of the served context window (finish_reason length; MSWE-agent sends no max_tokens)". The
wording now says max_tokens=N when the relay set or filled it, and keeps the old wording when the
relay sent no cap. Counts and statuses are unchanged.
"""
import json

from gbench.runners.eval_suites import swebench_common as C


def _reasons(rec):
    res = {"status": "success"}
    reasons = C.relay_result_reasons(res, dict(rec, replies_cut_off=2, completion_replies=46),
                                     "MSWE-agent")
    return res, [r for r in reasons if "cut off" in r]


def test_body_field_cap_names_the_run_max_tokens():
    res, cut = _reasons({"sets_body_fields": {"chat_template_kwargs": {"enable_thinking": True},
                                              "max_tokens": 65536},
                         "max_tokens_from_run": {"value": 65536, "requests_filled": 0}})
    assert cut == ["2 model turn(s) cut off at max_tokens=65536 (finish_reason length: the run's "
                   "--max-output-tokens, sent by gbench's relay with every request, as "
                   "MSWE-agent sends none)"]
    assert "; " not in cut[0] and "context window" not in cut[0]
    assert res["truncated_turns"] == 2 and res["status"] == "success"


def test_filled_cap_names_the_filled_requests():
    _, cut = _reasons({"max_tokens_from_run": {"value": 4096, "requests_filled": 40}})
    assert cut == ["2 model turn(s) cut off at a max_tokens limit (finish_reason length: "
                   "gbench's relay sent max_tokens=4096, the run's --max-output-tokens, with the "
                   "40 request(s) that carried none, as MSWE-agent sends none)"]
    assert "; " not in cut[0]


def test_fill_on_but_nothing_filled_names_the_scaffold_limit():
    """Review finding, 2026-10-11: the fill was on (value 4096) and filled no request, which
    EndpointRelay.patch_body does only when every request carried its own max_tokens or
    max_completion_tokens; the reason had fallen back to "MSWE-agent sends no max_tokens"."""
    res, cut = _reasons({"max_tokens_from_run": {"value": 4096, "requests_filled": 0}})
    assert cut == ["2 model turn(s) cut off at a max_tokens limit (finish_reason length: the "
                   "max_tokens or max_completion_tokens MSWE-agent sent with every request, so "
                   "gbench's relay filled in none)"]
    assert "; " not in cut[0] and "sends no max_tokens" not in cut[0]
    assert res["truncated_turns"] == 2


def test_fill_counts_zero_when_the_request_carries_its_own_cap():
    relay = C.EndpointRelay.__new__(C.EndpointRelay)
    relay._body_fields, relay._fill_max_tokens, relay.max_tokens_filled = {}, 4096, 0
    import threading
    relay._lock = threading.Lock()
    for own in ({"max_tokens": 100}, {"max_completion_tokens": 100}):
        relay.patch_body("/v1/chat/completions", json.dumps(own).encode(), "application/json")
    assert relay.max_tokens_filled == 0
    relay.patch_body("/v1/chat/completions", b"{}", "application/json")
    assert relay.max_tokens_filled == 1


def test_no_cap_keeps_the_context_window_wording():
    for rec in ({}, {"max_tokens_from_run": {"value": None, "requests_filled": 0}},
                {"sets_body_fields": {"temperature": 1.0}}):
        res, cut = _reasons(rec)
        assert cut == ["2 model turn(s) cut off at the end of the served context window "
                       "(finish_reason length; MSWE-agent sends no max_tokens)"]
        assert res["truncated_turns"] == 2


def test_capped_reason_survives_the_swe_bench_live_tab_grouping():
    from gbench.runners.eval_suites import swe_bench_live as L
    reason = C.relay_cutoff_reason(2, {"sets_body_fields": {"max_tokens": 65536}},
                                   "SWE-agent-for-eval")
    grouped = L._grouped_reasons({"lite": reason, "go": "other"}, ["lite", "go"])
    assert grouped == f"{reason} (lite); other (go)"
