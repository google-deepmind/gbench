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

"""A Gemini model in a role written for a non-thinking model gets the upstream visible budget
plus a thinking allowance (gemini_budget), and the replies still cut at that limit are counted
and named: simple-evals' grader (simpleqa, healthbench, browsecomp), ToolEval's judge
(toolbench), ACEBench's user simulator and the wildclawbench in-container proxy. The endpoint,
the Gemini API and docker are stubbed; no network."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import (acebench, base, browsecomp, gemini_budget, healthbench,
                                        simpleqa, toolbench, wildclawbench)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "gbench" / "docker"))
import wildclawbench_cascade_judge as J  # noqa: E402

ALLOW = gemini_budget.GEMINI_THINKING_ALLOWANCE


def _doc(name):
    return " ".join((REPO / "docs" / "evals" / f"{name}.md").read_text(encoding="utf-8").split())


# --------------------------------------------------------------------------- #
# The shared rule
# --------------------------------------------------------------------------- #
def test_the_allowance_is_one_value_everywhere():
    assert ALLOW == 8192
    assert gemini_budget.gemini_output_limit(2048) == 2048 + 8192
    # the proxy cannot import gbench: its copy must equal the shared constant
    assert J.GEMINI_THINKING_ALLOWANCE == ALLOW
    rec = gemini_budget.output_limit_record(1000, "x's max_tokens", "gpt-4o")
    assert (rec["max_output_tokens"], rec["upstream_max_tokens"], rec["thinking_allowance"]) == (
        9192, 1000, 8192)
    assert gemini_budget.cutoff_reason("grader", 0, 0, 2048) == ""
    reason = gemini_budget.cutoff_reason("grader", 2, 1, 2048)
    assert "2 grader repl(ies) stopped at max_output_tokens 10240" in reason
    assert "1 grader outage(s)" in reason


# --------------------------------------------------------------------------- #
# simple-evals' grader: simpleqa, healthbench, browsecomp
# --------------------------------------------------------------------------- #
def test_the_simple_evals_grader_is_sent_2048_plus_the_allowance():
    cfg = simpleqa.grader_config()
    assert simpleqa.GRADER_MAX_TOKENS == 2048
    assert cfg.max_output_tokens == simpleqa.GRADER_OUTPUT_LIMIT == 2048 + ALLOW
    assert cfg.temperature == 0.5
    rec = simpleqa.grader_sampling_record()
    assert rec["max_output_tokens"] == 10240 and rec["upstream_max_tokens"] == 2048
    assert rec["thinking_allowance"] == ALLOW and "gpt-4.1" in rec["max_output_tokens_note"]


def _simpleqa_run(judge):
    samples = [([{"role": "user", "content": q}], "Ada", {"category": "x"})
               for q in ("Who wrote it?", "Who drew it?", "Who sang it?")]

    async def send(**kw):
        return base.Reply(text="Ada", tool_calls=None, finish_reason="stop", completion_tokens=3)

    with mock.patch.object(base, "_send_single_request", send), \
            mock.patch.object(simpleqa, "gemini_required_skip", lambda *a, **k: None), \
            mock.patch.object(simpleqa, "_load_simpleqa_samples", lambda limit=None: samples), \
            mock.patch.object(simpleqa, "judge_generate_cascade_detailed", judge):
        return simpleqa.run_simpleqa("m", "http://x/v1", 1)


def test_simpleqa_names_grader_replies_still_cut_at_the_raised_limit():
    configs = []

    async def judge(prompt, config=None):
        configs.append(config)
        if "Who drew it?" in prompt:
            return "A", "gemini-3.8-flash", {"finish_reason": "MAX_TOKENS"}
        if "Who sang it?" in prompt:
            return None, "JUDGE_OUTAGE", {"finish_reason": None, "max_tokens_without_text": 2}
        return "A", "gemini-3.8-flash", {"finish_reason": "STOP"}

    res = _simpleqa_run(judge)
    assert {c.max_output_tokens for c in configs} == {10240}
    assert res["judge_replies_at_max_output_tokens"] == 1
    assert res["judge_outages_after_max_output_tokens"] == 1
    assert ("1 grader repl(ies) stopped at max_output_tokens 10240 (upstream's 2048 plus 8192"
            in res["leaderboard_comparable_reason"])
    assert "1 grader outage(s)" in res["leaderboard_comparable_reason"]
    assert res["judge_sampling"]["max_output_tokens"] == 10240


def test_simpleqa_without_cut_offs_names_none():
    async def judge(prompt, config=None):
        return "A", "gemini-3.8-flash", {"finish_reason": "STOP"}

    res = _simpleqa_run(judge)
    assert res["judge_replies_at_max_output_tokens"] == 0
    assert "stopped at max_output_tokens" not in res["leaderboard_comparable_reason"]


def test_healthbench_grades_with_the_raised_limit_and_names_cut_verdicts():
    rubric = [{"criterion": "Advises emergency care", "points": 10, "tags": ["axis:accuracy"]},
              {"criterion": "Mentions aspirin", "points": 5, "tags": ["axis:completeness"]}]
    samples = [([{"role": "user", "content": "q"}], json.dumps(rubric),
                {"category": "theme:t", "example_tags": ["theme:t"]})]
    configs = []

    async def judge(prompt, config=None):
        configs.append(config)
        finish = "MAX_TOKENS" if "aspirin" in prompt else "STOP"
        return '{"explanation": "e", "criteria_met": true}', "gemini-x", {"finish_reason": finish}

    async def send(**kw):
        return base.Reply(text="answer", tool_calls=None, finish_reason="stop")

    with mock.patch.object(base, "_send_single_request", side_effect=send), \
            mock.patch.object(healthbench, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(healthbench, "_load_healthbench_samples", lambda **k: samples), \
            mock.patch.object(healthbench, "judge_generate_cascade_detailed", judge):
        res = healthbench.run_healthbench("m", "http://x/v1", 1)
    assert configs and {c.max_output_tokens for c in configs} == {10240}
    assert res["judge_replies_at_max_output_tokens"] == 1
    assert "1 grader repl(ies) stopped at max_output_tokens 10240" in \
        res["leaderboard_comparable_reason"]
    assert res["judge_sampling"]["thinking_allowance"] == ALLOW


def test_browsecomp_grades_with_the_raised_limit_and_names_cut_verdicts():
    problems = ["p-one", "p-two"]
    samples = [([{"role": "user", "content": browsecomp.QUERY_TEMPLATE.format(question=p)}],
                f"gold-{i}", {"category": "Art", browsecomp.PROBLEM_KEY: p})
               for i, p in enumerate(problems)]
    configs = []

    async def judge(prompt, config=None):
        configs.append(config)
        finish = "MAX_TOKENS" if "p-two" in prompt else "STOP"
        return "correct: yes", "gemini-x", {"finish_reason": finish, "max_tokens_without_text": 0}

    async def send(**kw):
        return base.Reply(text="Exact Answer: x", tool_calls=None, finish_reason="stop")

    with mock.patch.object(base, "_send_single_request", side_effect=send), \
            mock.patch.object(browsecomp, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(browsecomp, "_load_browsecomp_samples", lambda limit=None: samples), \
            mock.patch.object(browsecomp, "judge_generate_cascade_detailed", judge):
        res = browsecomp.run_browsecomp("m", "http://x/v1", 1)
    assert configs and {c.max_output_tokens for c in configs} == {10240}
    assert res["judge_replies_at_max_output_tokens"] == 1
    assert "1 grader repl(ies) stopped at max_output_tokens 10240" in \
        res["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# toolbench: ToolEval's judge
# --------------------------------------------------------------------------- #
def _gt_reply(finish, call=None):
    from google.genai import types as gt
    parts = ([gt.Part(function_call=gt.FunctionCall(name=call, args={"answer_status": "Solved",
                                                                     "reason": "r"}))]
             if call else [gt.Part(text="thinking...")])
    return gt.GenerateContentResponse(candidates=[gt.Candidate(
        content=gt.Content(role="model", parts=parts), finish_reason=finish)])


def _fake_client(script, seen):
    class Models:
        async def generate_content(self, model, contents, config):
            seen.append(config.max_output_tokens)
            return script[model]
    return type("C", (), {"aio": type("A", (), {"models": Models()})()})()


def test_toolbench_judge_counts_an_outage_after_replies_cut_at_the_raised_limit(monkeypatch):
    from google.genai import types as gt
    seen = []
    client = _fake_client({"m1": _gt_reply(gt.FinishReason.MAX_TOKENS),
                           "m2": _gt_reply(gt.FinishReason.MAX_TOKENS)}, seen)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(toolbench._base, "_judge_clients", lambda: [client])
    monkeypatch.setattr(toolbench._base, "_next_judge_client", lambda: client)
    monkeypatch.setattr(toolbench._base, "judge_cascade", lambda: ["m1", "m2"])
    monkeypatch.setattr(toolbench._base, "_judge_cascade_rounds", lambda: 1)

    async def one():
        verdicts = toolbench._Verdicts()
        toolbench._VERDICTS.set(verdicts)
        cfg = toolbench._judge_generation_config("check_answer_status", True)
        return await toolbench._tooleval_cascade("prompt", config=cfg), verdicts

    (text, model), verdicts = asyncio.run(one())
    assert (text, model) == (None, "JUDGE_OUTAGE")
    assert set(seen) == {1000 + ALLOW}
    assert verdicts.max_tokens_cut == 2 and verdicts.outages_at_cap == 1
    reason = gemini_budget.cutoff_reason("ToolEval judge", verdicts.max_tokens_cut,
                                         verdicts.outages_at_cap, toolbench._JUDGE_MAX_TOKENS)
    assert "max_output_tokens 9192 (upstream's 1000 plus 8192" in reason


def test_toolbench_records_the_limit_on_the_result_record():
    rec = gemini_budget.output_limit_record(toolbench._JUDGE_MAX_TOKENS,
                                            "config.yaml completions_kwargs max_tokens",
                                            toolbench.UPSTREAM_JUDGE_MODEL)
    assert toolbench._JUDGE_OUTPUT_LIMIT == rec["max_output_tokens"] == 9192
    src = (REPO / "gbench" / "runners" / "eval_suites" / "toolbench.py").read_text()
    # the result reports the limit and names the cut counts
    assert '"judge_max_output_tokens": _JUDGE_OUTPUT_LIMIT' in src
    assert '"judge_outages_after_max_output_tokens_by_group"' in src
    assert "judge_cut = cutoff_reason(" in src


# --------------------------------------------------------------------------- #
# acebench: APIUSER's user simulator
# --------------------------------------------------------------------------- #
def test_acebench_user_simulator_counts_an_outage_after_empty_replies_at_the_limit(monkeypatch):
    from google.genai import types as gt
    seen = []
    empty = gt.GenerateContentResponse(candidates=[gt.Candidate(
        content=gt.Content(role="model", parts=[]), finish_reason=gt.FinishReason.MAX_TOKENS)])
    client = _fake_client({"m1": empty, "m2": empty}, seen)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(acebench.base, "_judge_clients", lambda: [client])
    monkeypatch.setattr(acebench.base, "_next_judge_client", lambda: client)
    monkeypatch.setattr(acebench, "judge_cascade", lambda: ["m1", "m2"])
    monkeypatch.setattr(acebench.base, "_judge_cascade_rounds", lambda: 1)
    sim = acebench._UserSimulator(["BaseApi"], "en")
    sim.messages = [{"role": "system", "content": "INSTR"}, {"role": "user", "content": "hi"}]
    with pytest.raises(acebench._UserSimulatorOutage):
        asyncio.run(sim._complete())
    assert set(seen) == {1000 + ALLOW}
    assert sim.cuts == {"cut": 0, "empty": 2, "outage_at_cap": 1}
    assert acebench.USER_SIM_OUTPUT_LIMIT == 9192


def test_acebench_user_simulator_text_reply_is_no_outage(monkeypatch):
    from google.genai import types as gt
    seen = []
    ok = gt.GenerateContentResponse(candidates=[gt.Candidate(
        content=gt.Content(role="model", parts=[gt.Part(text="sure")]),
        finish_reason=gt.FinishReason.MAX_TOKENS)])
    client = _fake_client({"m1": ok}, seen)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(acebench.base, "_judge_clients", lambda: [client])
    monkeypatch.setattr(acebench.base, "_next_judge_client", lambda: client)
    monkeypatch.setattr(acebench, "judge_cascade", lambda: ["m1"])
    sim = acebench._UserSimulator(["BaseApi"], "en")
    sim.messages = [{"role": "system", "content": "INSTR"}, {"role": "user", "content": "hi"}]
    assert asyncio.run(sim._complete()) == "sure"
    assert sim.cuts == {"cut": 1, "empty": 0, "outage_at_cap": 0}


# --------------------------------------------------------------------------- #
# wildclawbench: the in-container proxy and the host's report
# --------------------------------------------------------------------------- #
@pytest.fixture
def proxy_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "m-a,m-b")
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", "1")
    monkeypatch.setenv("GBENCH_JUDGE_BACKOFF", "0")
    monkeypatch.setenv("WILDCLAW_WORKDIR", str(tmp_path))
    monkeypatch.setattr(J.time, "sleep", lambda s: None)
    J.reset_usage()
    yield tmp_path
    J.reset_usage()


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _chat(content, finish):
    return {"choices": [{"message": {"role": "assistant", "content": content},
                         "finish_reason": finish}]}


def test_the_proxy_raises_the_tasks_cap_and_counts_replies_cut_at_it(proxy_env, monkeypatch):
    sent = []
    replies = {"m-a": _chat("", "length"), "m-b": _chat('{"score": 1', "length")}

    def fake_urlopen(req, timeout=0):
        body = json.loads(req.data.decode())
        sent.append(body)
        return _Resp(replies[body["model"]])
    monkeypatch.setattr(J.urllib.request, "urlopen", fake_urlopen)
    out = J.cascade_chat_completion({"model": "openai/gpt-5.4", "max_tokens": 128,
                                     "messages": [{"role": "user", "content": "grade"}]})
    assert [b["max_tokens"] for b in sent] == [128 + ALLOW, 128 + ALLOW]
    assert out["choices"][0]["message"]["content"] == '{"score": 1'   # used as it arrived
    assert J.last_limit_state() == {"raised": True, "cut": True, "empty": 1}
    J._count_limit_state("judge", outage=False)
    usage = json.loads((proxy_env / "gemini_usage.json").read_text())
    assert usage["thinking_allowance"] == {"judge": 1}
    assert usage["max_tokens_cut"] == {"judge": 1} and usage["max_tokens_empty"] == {"judge": 1}
    assert usage["outage_after_max_tokens"] == {}


def test_the_proxy_counts_an_outage_after_empty_replies_at_the_cap(proxy_env, monkeypatch):
    def fake_urlopen(req, timeout=0):
        return _Resp(_chat(None, "length"))
    monkeypatch.setattr(J.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="JUDGE_OUTAGE"):
        J.cascade_chat_completion({"max_completion_tokens": 300,
                                   "messages": [{"role": "user", "content": "grade"}]},
                                  record=False)
    J._count_limit_state("judge", outage=True)
    usage = J.usage_snapshot()
    assert usage["max_tokens_empty"] == {"judge": 2}
    assert usage["outage_after_max_tokens"] == {"judge": 1}
    assert usage["max_tokens_cut"] == {}


def test_a_request_without_a_cap_is_sent_none(proxy_env, monkeypatch):
    sent = []

    def fake_urlopen(req, timeout=0):
        sent.append(json.loads(req.data.decode()))
        return _Resp(_chat("ok", "stop"))
    monkeypatch.setattr(J.urllib.request, "urlopen", fake_urlopen)
    J.cascade_chat_completion({"messages": [{"role": "user", "content": "grade"}]})
    assert "max_tokens" not in sent[0] and "max_completion_tokens" not in sent[0]
    assert J.last_limit_state() == {"raised": False, "cut": False, "empty": 0}


def test_the_grounded_path_raises_the_cap_and_passes_the_real_finish_reason():
    req = J.gemini_grounded_request({"max_tokens": 512, "messages": [
        {"role": "user", "content": "verify"}]})
    assert req["generationConfig"]["maxOutputTokens"] == 512 + ALLOW
    for native, mapped in (("MAX_TOKENS", "length"), ("STOP", "stop"),
                           ("SAFETY", "content_filter"), (None, "stop")):
        cand = {"content": {"parts": [{"text": "t"}]}}
        if native:
            cand["finishReason"] = native
        choice = J.openai_completion_from_gemini({"candidates": [cand]}, "m")["choices"][0]
        assert choice["finish_reason"] == mapped and choice["native_finish_reason"] == native


def test_the_host_reports_the_limit_and_names_cut_replies():
    usage = {"thinking_allowance": {"judge": 4}, "max_tokens_cut": {"judge": 1},
             "max_tokens_empty": {"agent_helper": 2}, "outage_after_max_tokens": {}}
    rec = wildclawbench.gemini_output_limit(usage)
    assert rec["thinking_allowance"] == ALLOW and rec["requests_raised"] == {"judge": 4}
    reason = wildclawbench.output_limit_reason(usage)
    assert "1 repl(ies) returned cut at the raised max_tokens (1 judge)" in reason
    assert "2 repl(ies) stopped at the raised max_tokens with no text (2 agent_helper)" in reason
    assert wildclawbench.output_limit_reason({"thinking_allowance": {"judge": 4}}) is None


def test_the_wildclawbench_image_check_refuses_the_old_image():
    assert wildclawbench.HARNESS_ERRORS_VERSION == "9"
    assert wildclawbench._IMAGE_DEFAULT == "gbench-wildclawbench:316334cc-v9"
    df = (REPO / "gbench" / "docker" / "wildclawbench.Dockerfile").read_text()
    assert f'LABEL {wildclawbench.HARNESS_ERRORS_LABEL}="9"' in df
    assert f"docker build -t {wildclawbench._IMAGE_DEFAULT} " in df
    with mock.patch.object(wildclawbench.subprocess, "run",
                           return_value=subprocess.CompletedProcess([], 0, stdout="8\n")):
        assert not wildclawbench._image_accounts_harness_errors("gbench-wildclawbench:316334cc-v8")
    with mock.patch.object(wildclawbench.subprocess, "run",
                           return_value=subprocess.CompletedProcess([], 0, stdout="9\n")):
        assert wildclawbench._image_accounts_harness_errors(wildclawbench._IMAGE_DEFAULT)


# --------------------------------------------------------------------------- #
# Docs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,needles", [
    ("simpleqa", ["`max_output_tokens` 10240", "thinking allowance of 8192"]),
    ("healthbench", ["`max_output_tokens` 10240", "thinking allowance of 8192"]),
    ("browsecomp", ["`max_output_tokens` 10240", "thinking allowance of 8192"]),
    ("toolbench", ["`max_output_tokens=9192`", "`judge_outages_after_max_output_tokens_by_group`"]),
    ("acebench", ["`max_output_tokens` 9192", "`outages_after_max_tokens`"]),
    ("wildclawbench", ["### Judge output limit", "`gemini_usage.max_tokens_cut`",
                       "`gemini_usage.outage_after_max_tokens`", "`native_finish_reason`",
                       "`org.gbench.wildclawbench-harness-errors=9`",
                       "gbench-wildclawbench:316334cc-v9"]),
])
def test_the_guides_describe_the_limit(name, needles):
    doc = _doc(name)
    for needle in needles:
        assert needle in doc, (name, needle)
