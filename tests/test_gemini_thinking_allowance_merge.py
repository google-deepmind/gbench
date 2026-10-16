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

"""The Gemini thinking allowance (gemini_budget), second part: a sharded run's merge sums the
counts of replies still cut at the output limit and names them once over the union
(simple-evals' grader, ACEBench's user simulator, the wildclawbench proxy tally); ToolBench's
in-container cache-miss simulator gets upstream's max_tokens plus the allowance and counts its
cut replies; HealthBench counts a cut grader reply that was retried; wildbench uses the shared
allowance. The endpoint, the Gemini API and docker are stubbed; no network."""

import json
import re
import types
from pathlib import Path
from unittest import mock

import pytest

from gbench import merge
from gbench.runners.eval_suites import (acebench, base, browsecomp, gemini_budget, healthbench,
                                        simpleqa, toolbench, wildbench, wildclawbench)

REPO = Path(__file__).resolve().parent.parent
ALLOW = gemini_budget.GEMINI_THINKING_ALLOWANCE


# --------------------------------------------------------------------------- #
# One value everywhere
# --------------------------------------------------------------------------- #
def test_wildbench_and_the_toolbench_simulator_patch_use_the_shared_allowance():
    assert wildbench.JUDGE_THINKING_ALLOWANCE == ALLOW
    assert wildbench.JUDGE_OUTPUT_LIMIT == gemini_budget.gemini_output_limit(1024) == 1024 + ALLOW
    src = (REPO / "gbench" / "runners" / "eval_suites" / "wildbench.py").read_text()
    assert "JUDGE_THINKING_ALLOWANCE = GEMINI_THINKING_ALLOWANCE" in src
    assert not re.search(r"JUDGE_THINKING_ALLOWANCE\s*=\s*\d", src)
    patch = (REPO / "docker" / "toolbench_patch_simulator.py").read_text()
    m = re.search(r"^GEMINI_THINKING_ALLOWANCE = (\d+)$", patch, re.M)
    assert m and int(m.group(1)) == ALLOW
    assert toolbench.SIMULATOR_OUTPUT_LIMIT == 1024 + ALLOW


# --------------------------------------------------------------------------- #
# simple-evals: a sharded run's grader cut-offs are summed and named once
# --------------------------------------------------------------------------- #
def test_the_merge_sums_the_grader_cut_off_counts():
    assert "judge_replies_at_max_output_tokens" in merge.SUMMED_COUNTS
    assert "judge_outages_after_max_output_tokens" in merge.SUMMED_COUNTS
    for module in (simpleqa, healthbench, browsecomp, acebench):
        assert callable(getattr(module, "merge_shards", None)), module.__name__


_QUESTIONS = ("Who wrote it?", "Who drew it?", "Who sang it?", "Who made it?")


def _simpleqa_shard(monkeypatch, shard):
    monkeypatch.setenv("GBENCH_SHARD", shard)
    samples = [([{"role": "user", "content": q}], "Ada", {"category": "x"}) for q in _QUESTIONS]

    async def send(**kw):
        return base.Reply(text="Ada", tool_calls=None, finish_reason="stop", completion_tokens=3)

    async def judge(prompt, config=None):
        # one grader reply per shard stops at the limit (the shards' counts are equal, so a
        # merge that keeps one shard's count, or de-duplicates the reasons, under-counts)
        cut = "Who wrote it?" in prompt or "Who drew it?" in prompt
        return "A", "gemini-3.8-flash", {"finish_reason": "MAX_TOKENS" if cut else "STOP"}

    with mock.patch.object(base, "_send_single_request", send), \
            mock.patch.object(simpleqa, "gemini_required_skip", lambda *a, **k: None), \
            mock.patch.object(simpleqa, "_load_simpleqa_samples",
                              lambda limit=None: base.apply_shard(samples)[0]), \
            mock.patch.object(simpleqa, "judge_generate_cascade_detailed", judge):
        return simpleqa.run_simpleqa("m", "http://x/v1", 1)


def test_a_sharded_simpleqa_merge_counts_every_shards_cut_grader_replies(monkeypatch):
    parts = [_simpleqa_shard(monkeypatch, "1/2"), _simpleqa_shard(monkeypatch, "2/2")]
    assert [p["judge_replies_at_max_output_tokens"] for p in parts] == [1, 1]
    one = gemini_budget.cutoff_reason("grader", 1, 0, simpleqa.GRADER_MAX_TOKENS)
    assert all(one in p["leaderboard_comparable_reason"] for p in parts)
    merged = merge.merge_suite("simpleqa", parts)
    assert merged["judge_replies_at_max_output_tokens"] == 2
    assert merged["judge_outages_after_max_output_tokens"] == 0
    assert "judge_replies_at_max_output_tokens" not in (merged.get("merge_unreconciled_fields")
                                                        or {})
    reason = merged["leaderboard_comparable_reason"]
    assert gemini_budget.cutoff_reason("grader", 2, 0, simpleqa.GRADER_MAX_TOKENS) in reason
    assert one not in reason
    assert reason.count("grader repl(ies) stopped at max_output_tokens") == 1
    assert merged["total_questions"] == 4 and merged["leaderboard_comparable"] is False


def test_the_cut_off_reason_is_dropped_when_the_union_has_none():
    shard = {"judge_replies_at_max_output_tokens": 1}
    reason = simpleqa.grader_cutoff_reason(shard)
    assert reason and "; " not in reason
    merged = {"leaderboard_comparable_reason": "x; " + reason,
              "judge_replies_at_max_output_tokens": 0}
    with mock.patch("gbench.merge.generic_merge", lambda suite, results: dict(merged)):
        out = simpleqa.merge_shards([shard])
    assert out["leaderboard_comparable_reason"] == "x" and out["leaderboard_comparable"] is False


def test_a_sharded_browsecomp_merge_counts_every_shards_cut_grader_replies(monkeypatch):
    problems = ["p-one", "p-two", "p-three", "p-four"]
    samples = [([{"role": "user", "content": browsecomp.QUERY_TEMPLATE.format(question=p)}],
                f"gold-{i}", {"category": "Art", browsecomp.PROBLEM_KEY: p})
               for i, p in enumerate(problems)]

    async def judge(prompt, config=None):
        return "correct: yes", "gemini-x", {"finish_reason": "MAX_TOKENS",
                                            "max_tokens_without_text": 0}

    async def send(**kw):
        return base.Reply(text="Exact Answer: x", tool_calls=None, finish_reason="stop")

    parts = []
    for shard in ("1/2", "2/2"):
        monkeypatch.setenv("GBENCH_SHARD", shard)
        with mock.patch.object(base, "_send_single_request", side_effect=send), \
                mock.patch.object(browsecomp, "gemini_required_skip", lambda *a: None), \
                mock.patch.object(browsecomp, "_load_browsecomp_samples",
                                  lambda limit=None: base.apply_shard(samples)[0]), \
                mock.patch.object(browsecomp, "judge_generate_cascade_detailed", judge):
            parts.append(browsecomp.run_browsecomp("m", "http://x/v1", 1))
    assert [p["judge_replies_at_max_output_tokens"] for p in parts] == [2, 2]
    merged = merge.merge_suite("browsecomp", parts)
    assert merged["judge_replies_at_max_output_tokens"] == 4
    assert ("4 grader repl(ies) stopped at max_output_tokens 10240"
            in merged["leaderboard_comparable_reason"])
    assert "2 grader repl(ies)" not in merged["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# healthbench: a cut grader reply that was retried still counts
# --------------------------------------------------------------------------- #
def test_healthbench_counts_a_cut_grader_reply_that_a_retry_replaced():
    rubric = [{"criterion": "Advises emergency care", "points": 10, "tags": ["axis:accuracy"]}]
    samples = [([{"role": "user", "content": "q"}], json.dumps(rubric),
                {"category": "theme:t", "example_tags": ["theme:t"]})]
    calls = []

    async def judge(prompt, config=None):
        calls.append(1)
        if len(calls) == 1:     # cut mid-JSON: no criteria_met boolean, so it is asked again
            return '{"explanation": "the response adv', "gemini-x", {"finish_reason": "MAX_TOKENS"}
        return '{"explanation": "e", "criteria_met": true}', "gemini-x", {"finish_reason": "STOP"}

    async def send(**kw):
        return base.Reply(text="answer", tool_calls=None, finish_reason="stop")

    with mock.patch.object(base, "_send_single_request", side_effect=send), \
            mock.patch.object(healthbench, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(healthbench, "_load_healthbench_samples", lambda **k: samples), \
            mock.patch.object(healthbench, "judge_generate_cascade_detailed", judge):
        res = healthbench.run_healthbench("m", "http://x/v1", 1)
    assert len(calls) == 2
    trace = res["sample_traces"][0]
    assert trace["judge_finish_reasons"] == ["STOP"]          # the final attempt's end only
    assert trace["judge_replies_at_max_output_tokens"] == 1   # ... the retried cut one counted
    assert res["judge_replies_at_max_output_tokens"] == 1
    assert "1 grader repl(ies) stopped at max_output_tokens 10240" in \
        res["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# acebench: the simulator reason is one entry; the merge counts it over the union
# --------------------------------------------------------------------------- #
def test_acebench_user_simulator_reason_is_one_entry_and_recounted_over_the_union():
    sim = {"replies_cut_at_max_tokens": 1, "empty_replies_at_max_tokens": 2,
           "outages_after_max_tokens": 1}
    reason = acebench.user_sim_cutoff_reason(sim)
    assert "; " not in reason and "in 1 replies used as they were and 2 with no text" in reason
    assert "1 simulator outage(s)" in reason and str(acebench.USER_SIM_OUTPUT_LIMIT) in reason
    assert acebench.user_sim_cutoff_reason({}) == ""

    def agent(cut, empty, outage, models):
        return {"user_sim_requests": sum(models.values()), "user_sim_models": models,
                "user_sim_replies_cut_at_max_tokens": cut, "user_sim_empty_at_max_tokens": empty,
                "user_sim_outages_after_max_tokens": outage}
    shards = [dict(sim), dict(sim, outages_after_max_tokens=0)]
    merged = {"sample_traces": [agent(1, 2, 1, {"g1": 3}), agent(1, 2, 0, {"g1": 2, "g2": 1}),
                                {"sample_key": "single-turn row"}],
              "merge_unreconciled_fields": {"user_simulator": {"1/2": shards[0],
                                                               "2/2": shards[1]}}}
    acebench._recount_user_simulator(merged)
    got = merged["user_simulator"]
    assert (got["replies_cut_at_max_tokens"], got["empty_replies_at_max_tokens"],
            got["outages_after_max_tokens"]) == (2, 4, 1)
    assert got["requests"] == 6 and got["models_used"] == {"g1": 5, "g2": 1}
    assert "merge_unreconciled_fields" not in merged


# --------------------------------------------------------------------------- #
# wildclawbench: the proxy's tallies are summed over the shards
# --------------------------------------------------------------------------- #
def _wcb_shard(label, usage, task):
    reason = "; ".join(x for x in (
        base.SHARD_REASON.format(label=label, index=int(label[0]), count=2),
        "temperature 0.0 set by x on every model request",
        wildclawbench.output_limit_reason(usage)) if x)
    return {"eval_name": "wildclawbench", "shard": label, "status": "success",
            "gemini_usage": usage, "leaderboard_comparable_reason": reason,
            "gemini_output_limit": wildclawbench.gemini_output_limit(usage),
            "raw_summary": {"per_task": [{"task_id": task, "overall_score": 1.0,
                                          "modality": "text"}]}}


def test_a_sharded_wildclawbench_merge_sums_the_proxy_tallies_and_states_the_reason_once():
    u1 = {"judge": {"gemini-x": 3}, "thinking_allowance": {"judge": 3},
          "max_tokens_cut": {"judge": 1}, "max_tokens_empty": {"judge": 2},
          "outage_after_max_tokens": {}, "thinking_off": {"judge": 1}}
    u2 = {"judge": {"gemini-x": 5}, "thinking_allowance": {"judge": 4, "agent_helper": 1},
          "max_tokens_cut": {"judge": 3}, "max_tokens_empty": {"judge": 2},
          "outage_after_max_tokens": {"judge": 1}, "thinking_off": {}}
    reason1 = wildclawbench.output_limit_reason(u1)
    assert "; " not in reason1          # one entry of the "; "-joined reasons
    merged = wildclawbench.merge_shards([_wcb_shard("1/2", u1, "t1"),
                                         _wcb_shard("2/2", u2, "t2")])
    usage = merged["gemini_usage"]
    assert usage["max_tokens_cut"] == {"judge": 4}
    assert usage["max_tokens_empty"] == {"judge": 4}
    assert usage["outage_after_max_tokens"] == {"judge": 1}
    assert usage["thinking_allowance"] == {"agent_helper": 1, "judge": 7}
    assert usage["judge"] == {"gemini-x": 8}
    assert merged["gemini_output_limit"]["max_tokens_cut"] == {"judge": 4}
    assert merged["gemini_output_limit"]["requests_raised"] == {"agent_helper": 1, "judge": 7}
    assert merged["judge_thinking_off"]["requests"] == {"judge": 1}
    reasons = merged["leaderboard_comparable_reason"].split("; ")
    limit = [r for r in reasons if "raised max_tokens" in r]
    assert limit == [wildclawbench.output_limit_reason(usage)]
    assert "4 repl(ies) returned cut at the raised max_tokens (4 judge)" in limit[0]
    assert "4 repl(ies) stopped at the raised max_tokens with no text (4 judge)" in limit[0]
    assert "1 request(s) no cascade model served after such a reply (1 judge)" in limit[0]
    assert not any(r.startswith("sharded run") for r in reasons)


def test_the_wildclawbench_merge_drops_an_older_reason_split_on_semicolons():
    old_piece = "2 repl(ies) stopped at the raised max_tokens with no text (2 judge)"
    assert wildclawbench._is_output_limit_reason(old_piece)
    assert wildclawbench._is_output_limit_reason("Gemini output limit (the task's ...): x")
    assert not wildclawbench._is_output_limit_reason("not the full 60-task set")


# --------------------------------------------------------------------------- #
# toolbench: the in-container cache-miss simulator
# --------------------------------------------------------------------------- #
def _patched_helper(tmp_path):
    """The patch applied to a stand-in of upstream's server/main.py: (patched source, the
    namespace of its helper functions with a stub OpenAI client)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_tb_upstream_stand_in",
        REPO / "tests" / "test_gaia2_toolbench_judge_protocol_and_credentials.py")
    stand_in = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stand_in)
    _UPSTREAM_MAIN = stand_in._UPSTREAM_MAIN
    main = tmp_path / "main.py"
    main.write_text(_UPSTREAM_MAIN)
    patch = (REPO / "docker" / "toolbench_patch_simulator.py").read_text().replace(
        "/stb/server/main.py", str(main))
    exec(compile(patch, "toolbench_patch_simulator.py", "exec"), {"__name__": "patch"})
    return main.read_text()


class _Choice:
    def __init__(self, content, finish):
        self.message = types.SimpleNamespace(content=content)
        self.finish_reason = finish


def _server(tmp_path, monkeypatch, replies, sent):
    patched = _patched_helper(tmp_path)

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None, max_retries=None):
            class Completions:
                def create(self, **kw):
                    sent.append(kw)
                    content, finish = replies[kw["model"]]
                    return types.SimpleNamespace(choices=[_Choice(content, finish)])
            self.chat = types.SimpleNamespace(completions=Completions())

    class App:
        def post(self, path):
            return lambda fn: fn
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "m1,m2")
    monkeypatch.setenv("TB_SIMULATOR_ROUNDS", "1")
    monkeypatch.setenv("TB_SIMULATOR_BACKOFF", "0")
    import time as _time
    ns = {"app": App(), "Request": object, "Info": object, "OpenAI": FakeOpenAI,
          "OPENAI_API_KEY": "placeholder", "OPENAI_API_BASE": "http://relay/v1beta/openai/",
          "CONFIG": {"model": "m0", "temperature": 0, "toolbench_url": "http://proxy"},
          "LOGS": [], "__name__": "main"}
    exec(compile(patched.replace("import requests\n", ""), "main.py", "exec"), ns)
    ns["time"] = _time
    return patched, ns


def test_the_simulator_is_sent_1024_plus_the_allowance_and_counts_a_cut_reply(tmp_path,
                                                                            monkeypatch):
    sent = []
    patched, ns = _server(tmp_path, monkeypatch,
                          {"m1": ('{"error": "", "response": "cut ...', "length"),
                           "m2": ('{"error": "", "response": "ok"}', "stop")}, sent)
    assert "max_tokens = 1024 + _GBENCH_THINKING_ALLOWANCE," in patched
    assert f"_GBENCH_THINKING_ALLOWANCE = {ALLOW}\n" in patched
    record = {}
    ns["_GBENCH_CALL"].record = record
    out = ns["fake_response_function_chat"]([], {"q": 1}, {})
    # the reply cut with content is used as it arrived, as upstream uses a reply cut at its
    # max_tokens (the next model is not asked), and counted
    assert [(kw["model"], kw["max_tokens"]) for kw in sent] == [("m1", 1024 + ALLOW)]
    assert out == '{"error": "", "response": "cut ...'
    assert record["simulator_max_tokens_cut"] == 1 and "simulator_max_tokens_empty" not in record


def test_an_empty_reply_at_the_limit_asks_the_next_model_and_an_outage_is_marked(tmp_path,
                                                                                monkeypatch):
    sent = []
    _p, ns = _server(tmp_path, monkeypatch, {"m1": ("", "length"),
                                             "m2": ('{"error": "", "response": "ok"}', "stop")},
                     sent)
    record = {}
    ns["_GBENCH_CALL"].record = record
    out = ns["fake_response_function_chat"]([], {"q": 1}, {})
    assert json.loads(out)["response"] == "ok"
    assert record["simulator_max_tokens_empty"] == 1 and record["simulator_models"] == ["m2"]
    assert "simulator_outage_after_max_tokens" not in record

    sent.clear()
    _p, ns = _server(tmp_path, monkeypatch, {"m1": ("", "length"), "m2": (None, "length")}, sent)
    record = {}
    ns["_GBENCH_CALL"].record = record
    with pytest.raises(ValueError):
        ns["_gbench_simulate"](messages=[], max_tokens=1024 + ALLOW)
    assert record["simulator_max_tokens_empty"] == 2
    assert record["simulator_outage_after_max_tokens"] is True


def test_the_host_counts_the_simulators_cut_replies_and_names_them():
    records = [{"type": "fake_response", "group": "G1", "query_id": "1",
                "simulator_models": ["m1"], "simulator_max_tokens_cut": 2},
               {"type": "fake_response", "group": "G1", "query_id": "2",
                "simulator_models": ["m2"], "simulator_max_tokens_empty": 1},
               {"type": "server_error", "server_error": True, "stage": "simulator",
                "group": "G1", "query_id": "3", "simulator_max_tokens_empty": 2,
                "simulator_outage_after_max_tokens": True}]
    usage = toolbench.summarize_server_calls(records)["usage"]
    assert usage["simulator_replies_cut_at_max_tokens"] == 2
    assert usage["simulator_empty_replies_at_max_tokens"] == 3
    assert usage["simulator_outages_after_max_tokens"] == 1
    reason = toolbench.simulator_cutoff_reason(usage)
    assert "; " not in reason
    assert (f"stopped at max_tokens {1024 + ALLOW} (server/main.py's 1024 plus {ALLOW}"
            in reason)
    assert "in 2 replies used as they arrived and 3 with no content" in reason
    assert "1 simulator outage(s)" in reason
    assert toolbench.simulator_cutoff_reason(toolbench.summarize_server_calls([])["usage"]) == ""


def test_the_toolbench_image_check_requires_harness_version_5():
    assert toolbench.HARNESS_VERSION == "5"
    assert toolbench._IMAGE_DEFAULT == "gbench-toolbench:aa4ed9f4-v5"
    df = (REPO / "docker" / "toolbench.Dockerfile").read_text()
    assert f'LABEL {toolbench.HARNESS_LABEL}="5"' in df
    import subprocess
    with mock.patch.object(toolbench.subprocess, "run",
                           return_value=subprocess.CompletedProcess([], 0, stdout="4\n")):
        assert not toolbench.image_records_harness_errors("gbench-toolbench:aa4ed9f4")
    with mock.patch.object(toolbench.subprocess, "run",
                           return_value=subprocess.CompletedProcess([], 0, stdout="5\n")):
        assert toolbench.image_records_harness_errors(toolbench._IMAGE_DEFAULT)
