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

"""medxpertqa and gaia: upstream settings keyed on model names are not reasoning settings.

MedXpertQA's APIAgent (eval/model/api_agent.py @5f1db162) sends names containing o1 or o3, and
deepseek-reasoner, down a branch with no system turn, no temperature and 8192 tokens; GAIA's
reference agent (smolagents run_gaia.py @227ef5e49ddd) gives model_id "o1" reasoning_effort
"high" and 8192 tokens. Both key on model names of hosted APIs, so a served model takes the path
every other model name takes, in both modes:

  * medxpertqa - the chat branch: system turn, temperature 0, seed 0, max_tokens 2048 without
    --thinking; with it the same request and the run's budget (the 2048 not applied).
  * gaia       - no temperature, max_tokens 4096 without --thinking; with it the run's budget,
    and no reasoning_effort in either mode. Final replies cut off by the run's budget are
    counted in replies_cut_off_by_budget and named in the reason, in both splits.
  * the request never depends on the served model's name.
  * the docs and docstrings say so.
"""

import os
from unittest import mock

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import gaia as GAIA
from gbench.runners.eval_suites import medxpertqa as MX

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def _doc(name):
    with open(os.path.join(_REPO, "docs", "evals", f"{name}.md"), encoding="utf-8") as fh:
        return " ".join(fh.read().split())


def _flat(text):
    return " ".join((text or "").split())


# --------------------------------------------------------------------------- #
# medxpertqa
# --------------------------------------------------------------------------- #
def test_medxpertqa_declares_no_name_keyed_reasoning_settings():
    s = MX.OFFICIAL_SAMPLING
    assert s.reasoning is None
    assert (s.temperature, s.send_temperature, s.params, s.max_tokens) == (0.0, True, {"seed": 0},
                                                                           2048)
    assert not hasattr(MX, "REASONING_SAMPLING")
    assert not hasattr(MX, "UPSTREAM_REASONING_MAX_TOKENS")
    assert "chat branch" in s.reference and "not used for a served model" in s.reference


@pytest.mark.parametrize("knob", [2048, 4096, 65536])
def test_medxpertqa_token_budget_per_mode(knob):
    plan = base.resolve_token_budget("medxpertqa", MX.OFFICIAL_SAMPLING, thinking=False,
                                     max_output_tokens=knob)
    assert (plan.policy, plan.max_tokens) == ("official", 2048)
    plan = base.resolve_token_budget("medxpertqa", MX.OFFICIAL_SAMPLING, thinking=True,
                                     max_output_tokens=knob)
    # the 2048 is not applied to reasoning, and no 8192 comes in: the run's budget is sent
    assert (plan.policy, plan.max_tokens, plan.official_cap) == ("reasoning_run_budget", knob, 2048)
    rec = plan.record()
    assert rec["max_tokens_applied"] is False and rec["max_tokens_sent"] == knob


def _medx_rows(path, name=None, **k):
    if name != "Text":
        return []
    return [{"id": "Text-0", "question": "Why?\nAnswer Choices: (A) a (B) b", "label": "B",
             "options": {"A": "a", "B": "b"}, "question_type": "Reasoning", "images": []}]


def _run_medx(monkeypatch, *, thinking, knob, model="google/gemma-4-31b-it", reply=None):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    monkeypatch.delenv("GBENCH_MEDXPERTQA_TEMPERATURE", raising=False)
    base._RUN_KNOBS.pop("temperature", None)
    base._RUN_KNOBS["max_output_tokens"] = knob
    seen = []

    async def _send(**kw):
        seen.append(kw)
        if reply is not None:
            return reply(kw)
        text = "The answer is B." if len(kw["messages"]) > 2 else "B looks right."
        return base.Reply(text=text, tool_calls=None, finish_reason="stop", completion_tokens=5)

    with mock.patch("datasets.load_dataset", _medx_rows), \
            mock.patch.object(base, "_send_single_request", side_effect=_send):
        res = MX.run_medxpertqa(model, "http://x/v1", enable_thinking=thinking)
    return seen, res


@pytest.mark.parametrize("thinking,knob,sent", [(False, 65536, 2048), (True, 4096, 4096),
                                                (True, 65536, 65536)])
def test_medxpertqa_sends_the_chat_branch_in_both_modes(monkeypatch, thinking, knob, sent):
    seen, res = _run_medx(monkeypatch, thinking=thinking, knob=knob)
    assert len(seen) == 2                                    # rationale turn, answer turn
    for req in seen:
        assert req["messages"][0] == {"role": "system", "content": MX.SYSTEM_ROLE}
        # upstream's 0 without thinking; a --thinking run samples at 1.0
        assert req["temperature"] == (1.0 if thinking else 0.0)
        assert req["extra_payload"]["seed"] == 0
        assert req["max_output_tokens"] == sent
        assert req["thinking"] is thinking
    rec = res["official_sampling"]
    assert "reasoning_settings" not in rec
    assert rec["temperature_matches_official"] is (not thinking) and rec["max_tokens"] == 2048
    assert rec["max_tokens_applied"] is (not thinking)
    if thinking:
        assert rec["max_tokens_sent"] == knob
        assert rec["max_tokens_note"] == base.REASONING_CAP_NOTE
    assert res["accuracy"] == 100.0
    assert res["leaderboard_comparable"] is (not thinking), res.get("leaderboard_comparable_reason")
    if thinking:
        assert res["leaderboard_comparable_reason"] == "temperature 1.0 (upstream runs at 0)"


@pytest.mark.parametrize("thinking", [False, True])
def test_medxpertqa_request_does_not_depend_on_the_model_name(monkeypatch, thinking):
    def _shape(seen):
        return [([m["role"] for m in k["messages"]], k["temperature"], k["max_output_tokens"],
                 k["extra_payload"]) for k in seen]

    shapes = [_shape(_run_medx(monkeypatch, thinking=thinking, knob=16384, model=name)[0])
              for name in ("google/gemma-4-31b-it", "o1-2024-12-17", "deepseek-reasoner",
                           "qwq-32b-preview")]
    assert all(s == shapes[0] for s in shapes)


def test_medxpertqa_thinking_cut_offs_are_counted(monkeypatch):
    def _reply(kw):
        if len(kw["messages"]) > 2:
            return base.Reply(text="The answer is B.", tool_calls=None, finish_reason="stop",
                              completion_tokens=5)
        # the rationale runs out of the run's budget inside the reasoning channel
        return base.Reply(text="", tool_calls=None, finish_reason="length",
                          completion_tokens=kw["max_output_tokens"], reasoning="...")

    seen, res = _run_medx(monkeypatch, thinking=True, knob=4096, reply=_reply)
    assert [k["max_output_tokens"] for k in seen] == [4096, 4096]
    assert res["replies_cut_off_by_budget"] == 1 and res["replies_cut_off_in_reasoning"] == 1
    assert res["replies_at_official_max_tokens"] == 0
    assert res["leaderboard_comparable"] is False
    assert "cut off by this run's token budget" in res["leaderboard_comparable_reason"]


def test_medxpertqa_thinking_temperature_reason_names_upstreams_0(monkeypatch):
    monkeypatch.delenv("GBENCH_SHARD", raising=False)
    monkeypatch.setenv("GBENCH_MEDXPERTQA_TEMPERATURE", "1.0")
    base._RUN_KNOBS["max_output_tokens"] = 8192

    async def _send(**kw):
        return base.Reply(text="The answer is B.", tool_calls=None, finish_reason="stop",
                          completion_tokens=5)

    with mock.patch("datasets.load_dataset", _medx_rows), \
            mock.patch.object(base, "_send_single_request", side_effect=_send):
        res = MX.run_medxpertqa("m", "http://x/v1", enable_thinking=True)
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert "temperature 1.0 (upstream runs at 0)" in res["leaderboard_comparable_reason"]
    assert "reasoning model" not in res["leaderboard_comparable_reason"]


def test_medxpertqa_loader_keeps_the_system_turn():
    with mock.patch("datasets.load_dataset", _medx_rows):
        [(messages, gold, meta)] = MX._load_medxpertqa_samples()
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == MX.SYSTEM_ROLE
    assert gold == {"label": "B", "n_options": 2}


def test_medxpertqa_docs_and_docstring_describe_the_served_model_path():
    src, doc = _flat(MX.__doc__), _doc("medxpertqa")
    for text in (src, doc):
        assert "not reasoning-model settings for a served model" in text
        assert "deepseek-reasoner" in text and "qwq" in text and "model_info.json" in text
        assert "takes the chat branch and the 2048" in text
        assert "never below" not in text and "reasoning branch's 8192" not in text
    assert "System `You are a helpful medical assistant.` (in both modes)" in doc
    assert "(0; a `--eval-thinking` run samples at 1.0, so it is not comparable)" in doc
    assert "(0, with or without `--eval-thinking`)" not in doc
    assert "`official_sampling.max_tokens_applied: false`" in doc
    assert "none with it" not in doc and "reasoning_settings: true" not in doc


# --------------------------------------------------------------------------- #
# gaia
# --------------------------------------------------------------------------- #
def test_gaia_declares_no_name_keyed_reasoning_settings():
    s = GAIA.GAIA_SAMPLING
    assert s.reasoning is None
    assert (s.send_temperature, s.max_tokens) == (False, 4096)
    assert "run_gaia.py@227ef5e49ddd" in s.reference and "every model_id but o1" in s.reference
    for name in ("REFERENCE_REASONING_EFFORT", "REFERENCE_REASONING_MAX_TOKENS",
                 "record_reference_budget"):
        assert not hasattr(GAIA, name)


@pytest.mark.parametrize("thinking,knob,policy,sent", [
    (False, 65536, "official", 4096), (True, 2048, "reasoning_run_budget", 2048),
    (True, 4096, "reasoning_run_budget", 4096), (True, 65536, "reasoning_run_budget", 65536)])
def test_gaia_token_budget_per_mode(thinking, knob, policy, sent):
    plan = base.resolve_token_budget("gaia", GAIA.GAIA_SAMPLING, thinking=thinking,
                                     max_output_tokens=knob)
    assert (plan.policy, plan.max_tokens) == (policy, sent)


def _run_gaia_loop(*, thinking, knob, extra_payload):
    seen = []
    call = {"id": "c1", "type": "function",
            "function": {"name": "web_search", "arguments": "{\"query\": \"x\"}"}}
    queue = [base.Reply("", [call], "tool_calls", completion_tokens=3)]

    async def send(**k):
        async with k["semaphore"]:
            pass
        seen.append(k)
        return queue.pop(0) if queue else base.Reply("FINAL ANSWER: 3", None, "stop",
                                                     completion_tokens=5)

    async def tool(name, args):
        return "3"

    base._RUN_KNOBS["max_output_tokens"] = knob
    base._RUN_KNOBS["max_model_len"] = 262144
    samples = [([{"role": "user", "content": "q"}], "3", {"category": "Level_1"})]
    with mock.patch.object(base, "_send_single_request", send):
        res = base.run_eval_suite(
            eval_name="gaia", model_name="m", base_url="http://x", concurrency=1,
            samples=samples, eval_fn=lambda r, g: True, thinking=thinking,
            official_sampling=GAIA.GAIA_SAMPLING, tool_loop=GAIA.GAIA_TOOL_LOOP,
            tool_executor=tool, extra_payload=extra_payload)
    return seen, res


@pytest.mark.parametrize("thinking,knob,sent", [(False, 65536, 4096), (True, 2048, 2048),
                                                (True, 65536, 65536)])
def test_gaia_every_turn_sends_the_served_model_path(monkeypatch, thinking, knob, sent):
    got = []
    monkeypatch.setattr(GAIA, "check_gaia_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(GAIA, "search_available", lambda: True)
    monkeypatch.setattr(GAIA, "gemini_key_live_valid", lambda key: (True, ""))
    monkeypatch.setattr(GAIA, "_load_gaia_samples", lambda limit=None, split=None: [])
    monkeypatch.setattr(GAIA, "run_eval_suite", lambda **kw: got.append(kw) or {})
    monkeypatch.delenv("GBENCH_GAIA_SPLIT", raising=False)
    GAIA.run_gaia("o1", "http://x", 1, enable_thinking=thinking)
    payload = got[-1]["extra_payload"]
    assert "reasoning_effort" not in payload and got[-1]["official_sampling"] is GAIA.GAIA_SAMPLING

    seen, res = _run_gaia_loop(thinking=thinking, knob=knob, extra_payload=payload)
    assert len(seen) == 2
    assert [k["max_output_tokens"] for k in seen] == [sent, sent]
    # temperature 0.0 without --thinking, 1.0 with it (gbench's rule; the reference agent sends
    # none)
    assert all(k["temperature"] == (1.0 if thinking else 0.0) for k in seen)
    assert all("reasoning_effort" not in (k["extra_payload"] or {}) for k in seen)
    rec = res["official_sampling"]
    assert "reasoning_settings" not in rec and rec["max_tokens"] == 4096
    assert rec["max_tokens_applied"] is (not thinking)
    if thinking:
        assert rec["max_tokens_sent"] == knob
        assert rec["max_tokens_note"] == base.REASONING_CAP_NOTE


def _fake_suite(traces, official_sent):
    def fake(**kw):
        return {"sample_traces": [dict(t) for t in traces],
                "official_max_tokens_sent": list(official_sent),
                "official_sampling": {}}
    return fake


def _gaia_validation(monkeypatch, traces, official_sent, thinking):
    monkeypatch.setattr(GAIA, "gaia_split", lambda: "validation")
    monkeypatch.setattr(GAIA, "check_gaia_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(GAIA, "search_available", lambda: True)
    monkeypatch.setattr(GAIA, "gemini_key_live_valid", lambda key: (True, ""))
    monkeypatch.setattr(GAIA, "_load_gaia_samples", lambda limit=None, split=None: [])
    monkeypatch.setattr(GAIA, "run_eval_suite", _fake_suite(traces, official_sent))
    return GAIA.run_gaia("m", "http://x", 1, enable_thinking=thinking)


def test_gaia_thinking_run_counts_every_budget_cut_off(monkeypatch):
    traces = [{"finish_reason": "length", "response_text": "", "max_tokens_sent": 4096},
              {"finish_reason": "length", "response_text": "FINAL ANSWER: 1",
               "max_tokens_sent": 4096},
              {"finish_reason": "stop", "response_text": "FINAL ANSWER: 1",
               "max_tokens_sent": 4096}]
    res = _gaia_validation(monkeypatch, traces, [], thinking=True)
    assert res["replies_cut_off_by_budget"] == 2 and res["replies_cut_off_in_reasoning"] == 1
    assert ("2 repl(ies) cut off by this run's token budget, not by an official cap (1 of them "
            "inside the reasoning channel with no answer)") in res["leaderboard_comparable_reason"]
    assert res["leaderboard_comparable"] is False
    for gone in ("reasoning_effort", "replies_cut_at_max_tokens", "budget_notes"):
        assert gone not in res


def test_gaia_no_think_cut_at_the_reference_cap_is_its_outcome(monkeypatch):
    traces = [{"finish_reason": "length", "response_text": "FINAL ANSWER: 1",
               "max_tokens_sent": 4096},
              # the context window lowered this sample's request below the 4096
              {"finish_reason": "length", "response_text": "", "max_tokens_sent": 1000}]
    res = _gaia_validation(monkeypatch, traces, [4096], thinking=False)
    assert res["replies_cut_off_by_budget"] == 1 and res["replies_cut_off_in_reasoning"] == 1
    assert "1 repl(ies) cut off by this run's token budget" in res["leaderboard_comparable_reason"]
    res = _gaia_validation(monkeypatch, traces[:1], [4096], thinking=False)
    assert res["replies_cut_off_by_budget"] == 0
    assert "cut off by this run's token budget" not in res["leaderboard_comparable_reason"]


def test_gaia_test_split_records_budget_cut_offs(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(GAIA, "gaia_split", lambda: "test")
    monkeypatch.setattr(GAIA, "check_gaia_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(GAIA, "search_available", lambda: True)
    monkeypatch.setattr(GAIA, "gemini_key_live_valid", lambda key: (True, ""))
    monkeypatch.setattr(GAIA, "_load_gaia_samples", lambda limit=None, split=None: [])
    traces = [{"sample_idx": 0, "category": "Level_1", "extra_payload": {"task_id": "t0"},
               "finish_reason": "length", "response_text": "", "max_tokens_sent": 65536,
               "status": "TRUNCATED"},
              {"sample_idx": 1, "category": "Level_1", "extra_payload": {"task_id": "t1"},
               "finish_reason": "stop", "response_text": "FINAL ANSWER: 2",
               "max_tokens_sent": 65536, "status": "OK"}]
    got = []

    def fake(**kw):
        got.append(kw)
        return _fake_suite(traces, [])(**kw)

    monkeypatch.setattr(GAIA, "run_eval_suite", fake)
    with pytest.raises(RuntimeError) as err:
        GAIA.run_gaia("m", "http://x", 1, enable_thinking=True)
    assert "reasoning_effort" not in got[-1]["extra_payload"]
    msg = str(err.value)
    assert "[INFRA REQUIRED] gaia" in msg
    assert "1 repl(ies) cut off by this run's token budget" in msg
    out = tmp_path / "gaia_submission"
    [result_file] = [p for p in out.iterdir() if p.name.endswith("_result.json")]
    import json
    rec = json.loads(result_file.read_text())
    assert rec["leaderboard_submission"]["replies_cut_off_by_budget"] == 1
    assert rec["replies_cut_off_by_budget"] == 1 and rec["replies_cut_off_in_reasoning"] == 1


def test_gaia_docs_and_docstring_describe_the_served_model_path():
    src, doc = _flat(GAIA.__doc__), _doc("gaia")
    for text in (src, doc):
        assert "not a reasoning-model setting for a served model" in text
        assert "`replies_cut_off_by_budget`" in text
        assert "reasoning-model settings apply" not in text
        assert "never below" not in text and "above that 8192" not in text
    assert "No reasoning_effort is sent in either mode." in src
    assert "no `reasoning_effort` is sent" in doc
    assert "`leaderboard_submission.replies_cut_off_by_budget`" in doc
    assert "`official_sampling.max_tokens_applied: false`" in doc
    assert "`replies_cut_at_max_tokens`" not in doc and "above the reference agent's 8192" not in doc
    assert "You ran out of room to think" in doc and "counted only when that turn is cut off" in doc
