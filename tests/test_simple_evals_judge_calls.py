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

"""The simple-evals family and the other judged suites (browsecomp, simpleqa, frames, healthbench,
humanitys_last_exam, wildbench, gdpval): the judge call follows upstream verbatim (prompt,
system message, temperature, max_tokens), every verdict records the Gemini model that gave it,
judge-side failures are harness errors (counted, named, never the model's outcome), and a run
with nothing scored withholds its score. Docker, the network and Gemini are stubbed."""

import asyncio
import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import (base, browsecomp, frames, gdpval, healthbench, simpleqa,
                                        wildbench)
from gbench.runners.eval_suites import humanitys_last_exam as hle

def _patch_detailed(module, judge):
    """Patch a suite's judge_generate_cascade_detailed with a (text, model) judge stub; the
    detailed call returns (text, model, info) with nothing cut off at the grader's cap."""
    async def detailed(prompt, config=None):
        text, model = await judge(prompt, config=config)
        return text, model, {"finish_reason": None, "max_tokens_without_text": 0}
    return mock.patch.object(module, "judge_generate_cascade_detailed", detailed)


REPO = Path(__file__).resolve().parent.parent

#: sha256 of openai/simple-evals@652c89d0 browsecomp_eval.py GRADER_TEMPLATE as Python builds it
#: at run time ("""...""".strip(), `\%` kept as backslash-percent), and of its QUERY_TEMPLATE
#: (placeholder `{Question}`).
UPSTREAM_GRADER_SHA256 = "0f0023ee579b8c134f1834ed8952778b9e01460e31d47c242ee3629da9d44835"
UPSTREAM_QUERY_SHA256 = "6990664d8e2ac329133e805515199d65f943c128aba9a3af9727fcf1dd32264f"


def _simpleqa_detailed(judge):
    """A (text, model) judge stub as simpleqa's detailed cascade call (text, model, info)."""
    async def _detailed(prompt, config=None):
        text, model = await judge(prompt, config=config)
        return text, model, {}
    return _detailed


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _doc(name):
    return " ".join((REPO / "docs" / "evals" / f"{name}.md").read_text(encoding="utf-8").split())


def _serve(replies, seen=None, finish_reason="stop"):
    """A fake _send_single_request answering by the last user text (None = failed request)."""
    async def _send(**kw):
        if seen is not None:
            seen.append(kw)
        user = next(m["content"] for m in reversed(kw["messages"]) if m.get("role") == "user")
        reply = replies[user]
        if reply is None:
            return base.Reply(text=None, tool_calls=None, error="HTTP 500 server error")
        return base.Reply(text=reply, tool_calls=None, finish_reason=finish_reason,
                          completion_tokens=5)
    return _send


def _detailed(fn):
    """A judge stub `fn(prompt, config) -> (text, model, info)` as the detailed cascade."""
    async def _call(prompt, config=None):
        return fn(prompt, config)
    return _call


# --------------------------------------------------------------------------- #
# browsecomp
# --------------------------------------------------------------------------- #
def test_browsecomp_templates_are_upstreams_text():
    assert _sha(browsecomp.GRADER_TEMPLATE) == UPSTREAM_GRADER_SHA256
    assert "if there if there is any inconsistency" in browsecomp.GRADER_TEMPLATE
    assert "0|\\%| and 100|\\%|" in browsecomp.GRADER_TEMPLATE
    assert "incorrect.\n\n\nconfidence:" in browsecomp.GRADER_TEMPLATE
    assert _sha(browsecomp.QUERY_TEMPLATE.replace("{question}", "{Question}")) == \
        UPSTREAM_QUERY_SHA256


def test_browsecomp_dataset_is_pinned(monkeypatch):
    seen = {}

    class _DS(list):
        pass

    def _load(*a, **k):
        seen["a"], seen["k"] = a, k
        return _DS()
    import types
    monkeypatch.setitem(__import__("sys").modules, "datasets",
                        types.SimpleNamespace(load_dataset=_load))
    with pytest.raises(RuntimeError, match="empty rows"):
        browsecomp._load_browsecomp_samples()
    assert seen["a"] == ("smolagents/browse_comp",)
    assert seen["k"] == {"split": "test", "revision": "f975f60afc3c811de82951165f5a9a921d024e14"}


def _encrypt(plain, canary):
    import base64
    data = plain.encode()
    key = browsecomp._derive_key(canary, len(data))
    return base64.b64encode(bytes(a ^ b for a, b in zip(data, key))).decode()


def test_browsecomp_loader_keeps_the_raw_problem_for_the_grader(monkeypatch):
    import types
    rows = [{"problem": _encrypt("Which ship?", "pw"), "answer": _encrypt("Argo", "pw"),
             "canary": "pw", "problem_topic": "History"}]
    monkeypatch.setitem(__import__("sys").modules, "datasets",
                        types.SimpleNamespace(load_dataset=lambda *a, **k: rows))
    (messages, gold, meta), = browsecomp._load_browsecomp_samples()
    assert gold == "Argo" and meta[browsecomp.PROBLEM_KEY] == "Which ship?"
    assert messages == [{"role": "user", "content": browsecomp.QUERY_TEMPLATE.format(
        question="Which ship?")}]
    # the meta key is never sent to the model's endpoint
    assert browsecomp.PROBLEM_KEY not in base.ALLOWED_API_KEYS


def _browsecomp(grader_replies, model_replies=None, thinking=False, seen=None, calls=None,
                finish="stop", judge_finish="STOP"):
    problems = list(grader_replies)
    samples = [([{"role": "user", "content": browsecomp.QUERY_TEMPLATE.format(question=p)}],
                f"gold-{i}", {"category": "Art", browsecomp.PROBLEM_KEY: p})
               for i, p in enumerate(problems)]
    model_replies = model_replies or {p: f"Exact Answer: {p}" for p in problems}
    replies = {browsecomp.QUERY_TEMPLATE.format(question=p): model_replies[p] for p in problems}

    def _judge(prompt, config):
        if calls is not None:
            calls.append((prompt, config))
        problem = prompt.split("[question]: ", 1)[1].split("\n\n[response]", 1)[0]
        text = grader_replies[problem]
        if text is None:
            return None, "JUDGE_OUTAGE", {"finish_reason": None, "max_tokens_without_text": 0}
        return text, "gemini-3.8-flash", {"finish_reason": judge_finish,
                                           "max_tokens_without_text": 0}

    with mock.patch.object(base, "_send_single_request",
                           side_effect=_serve(replies, seen, finish)), \
            mock.patch.object(browsecomp, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(browsecomp, "_load_browsecomp_samples", lambda limit=None: samples), \
            mock.patch.object(browsecomp, "judge_generate_cascade_detailed", _detailed(_judge)):
        return browsecomp.run_browsecomp("m", "http://x/v1", 1, enable_thinking=thinking,
                                         max_output_tokens=16384)


def test_browsecomp_samples_as_simple_evals_chat_sampler():
    base.set_run_knobs(max_output_tokens=16384)
    seen = []
    res = _browsecomp({"P1": "correct: yes"}, seen=seen)
    req = seen[0]
    assert req["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    # gbench's rule: a no-think run samples at 0.0; the sampler's 0.5 is recorded, not sent
    assert req["temperature"] == 0.0 and req["max_output_tokens"] == 2048
    assert res["official_sampling"]["temperature"] == 0.5
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert res["sampler_max_tokens"] == 2048 and res["max_tokens_sent"] == [2048]
    seen = []
    res = _browsecomp({"P1": "correct: yes"}, thinking=True, seen=seen)
    # the maintainer's rule: a --thinking run samples at 1.0, not the sampler's 0.5
    assert seen[0]["max_output_tokens"] == 16384 and seen[0]["temperature"] == 1.0
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert res["sampler_max_tokens"] is None


def test_browsecomp_a_reply_at_the_2048_cap_is_the_samplers_outcome():
    res = _browsecomp({"P1": "correct: no"}, finish="length")
    assert res["replies_at_sampler_max_tokens"] == 1 and res["replies_cut_off_by_budget"] == 0
    assert res["status"] == "success"


def test_browsecomp_grader_gets_the_raw_problem_and_grading_sampler_settings():
    calls = []
    _browsecomp({"Who built the Argo?": "correct: yes"}, calls=calls)
    prompt, cfg = calls[0]
    assert prompt == browsecomp.GRADER_TEMPLATE.format(
        question="Who built the Argo?", response="Exact Answer: Who built the Argo?",
        correct_answer="gold-0")
    assert "Your response should be in the following format" not in prompt
    assert cfg.system_instruction == "You are a helpful assistant."
    # grading_sampler's 2048 for the reply plus the Gemini thinking allowance
    assert cfg.temperature == 0.5 and cfg.max_output_tokens == 2048 + 8192
    assert cfg.response_mime_type is None


def test_browsecomp_records_the_answering_model_and_reply_per_verdict():
    res = _browsecomp({"P1": "extracted_final_answer: x\ncorrect: yes", "P2": "correct: no"})
    t1, t2 = sorted(res["sample_traces"], key=lambda t: t["sample_idx"])
    assert t1["judge_model"] == "gemini-3.8-flash" and "correct: yes" in t1["judge_reply"]
    assert t1["is_correct"] is True and t2["is_correct"] is False
    assert res["judge_models"] == {"gemini-3.8-flash": 2}
    assert res["judge_cascade"] == base.judge_cascade()
    assert res["judge_sampling"]["max_output_tokens"] == 2048 + 8192
    assert res["judge_sampling"]["upstream_max_tokens"] == 2048
    assert res["judge"] == "gbench-gemini-cascade" and res["accuracy"] == 50.0
    assert res["metric"].startswith("BrowseComp accuracy")


def test_browsecomp_grader_failure_is_excluded_and_named():
    res = _browsecomp({"P1": "correct: yes", "P2": "I cannot tell."})
    assert res["accuracy"] == 100.0                  # 1 yes / 1 graded
    assert res["judge_failures"] == 1 and res["scoring_excluded"] == 1
    assert res["status"] == "completed_with_errors"
    assert "grader failures" in res["leaderboard_comparable_reason"]
    assert "scores such a reply 'no'" in res["leaderboard_comparable_reason"]


def test_browsecomp_every_verdict_missing_withholds_the_score():
    res = _browsecomp({"P1": "no verdict", "P2": "none either"})
    assert res["status"] == "failed"
    # a grader failure plus a judge outage: nothing scored, so the score is withheld, not 0
    res = _browsecomp({"P1": "no verdict", "P2": None})
    assert res["status"] == "failed" and res["accuracy"] is None
    # base.run_eval_suite withholds it first and names both causes
    why = res["accuracy_withheld_reason"]
    assert why.startswith("nothing was scored: ")
    assert "judge failures 1" in why and "judge_outages 1" in why


def test_browsecomp_relaxed_parse_is_counted_and_named():
    res = _browsecomp({"P1": "Correct:  Yes", "P2": "correct: no"})
    assert res["judge_verdicts_read_only_by_relaxed_parse"] == 1
    assert res["accuracy"] == 50.0
    assert "space-tolerant parse" in res["leaderboard_comparable_reason"]
    assert browsecomp.parse_verdict("**correct:** yes") is None
    assert browsecomp.parse_verdict("correct: no\ncorrect: yes") == "no"


def test_browsecomp_reason_names_every_deviation():
    res = _browsecomp({"P1": "correct: yes"}, judge_finish="MAX_TOKENS")
    reason = res["leaderboard_comparable_reason"]
    assert "gpt-4.1-2025-04-14" in reason
    assert res["judge_replies_at_max_output_tokens"] == 1
    assert "max_output_tokens 10240 (upstream's 2048 plus 8192" in reason
    assert res["leaderboard_comparable"] is False
    with mock.patch.dict("os.environ", {"GBENCH_BROWSECOMP_TEMPERATURE": "0.0"}):
        res = _browsecomp({"P1": "correct: yes"})
    assert "temperature 0.0 (simple-evals' chat sampler samples at 0.5)" in \
        res["leaderboard_comparable_reason"]


def test_browsecomp_judge_outage_is_excluded():
    res = _browsecomp({"P1": "correct: yes", "P2": None})
    assert res["judge_outages"] == 1 and res["accuracy"] == 100.0


def test_browsecomp_doc_states_the_protocol():
    doc = _doc("browsecomp")
    for needle in ("f975f60afc3c811de82951165f5a9a921d024e14", "You are a helpful assistant.",
                   "temperature 0.5", "max_tokens` 2048", "max_output_tokens` 10240",
                   "raw decrypted problem", "if there if there", "JUDGE_FAILED",
                   "judge_verdicts_read_only_by_relaxed_parse", "judge_model"):
        assert needle in doc, needle
    assert "one remaining reason only" not in doc


# --------------------------------------------------------------------------- #
# simpleqa: the grader call and the gold target
# --------------------------------------------------------------------------- #
def test_simpleqa_sends_grading_samplers_max_tokens():
    cfg = simpleqa.grader_config()
    assert cfg.max_output_tokens == 2048 + 8192 and cfg.temperature == 0.5
    assert cfg.system_instruction == "You are a helpful assistant."
    rec = simpleqa.grader_sampling_record()
    assert rec["max_output_tokens"] == 2048 + 8192 and rec["upstream_max_tokens"] == 2048
    assert "not sent" not in json.dumps(rec).replace("not sent (grading_sampler sends no "
                                                     "response_format)", "")


def test_simpleqa_gold_target_keeps_its_whitespace(monkeypatch):
    import types
    rows = [{"problem": "When?", "answer": " April 2 ", "metadata": "{'topic': 'History'}"}]
    monkeypatch.setitem(__import__("sys").modules, "datasets",
                        types.SimpleNamespace(load_dataset=lambda *a, **k: rows))
    (msgs, gold, meta), = simpleqa._load_simpleqa_samples()
    assert gold == " April 2 "
    prompts = []

    async def _judge(prompt, config=None):
        prompts.append(prompt)
        return "A", "gemini-3.8-flash"
    trace = {"messages": msgs, "gold_answer": gold, "response_text": "April 2"}
    with mock.patch.object(simpleqa, "judge_generate_cascade_detailed", _simpleqa_detailed(_judge)):
        asyncio.run(simpleqa._async_judge_simpleqa([trace]))
    assert "Gold target:  April 2 \nPredicted answer: April 2" in prompts[0]
    assert trace["judge_model"] == "gemini-3.8-flash"


def test_simpleqa_and_healthbench_docs_state_the_judge_cap():
    for name in ("simpleqa", "healthbench"):
        doc = _doc(name)
        assert "max_output_tokens` 10240" in doc and "max_tokens` 2048" in doc, name
        assert "max_tokens=2048` is not sent" not in doc and \
            "cap is not sent" not in doc, name


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #
def test_frames_autorater_failure_is_named_in_the_reason():
    samples = [([{"role": "user", "content": q}], "Paris", {"category": "T"}) for q in ("a", "b")]

    async def _judge(prompt, config=None):
        return ("Decision: TRUE" if "Predicted Answer: Paris" in prompt else "unsure"), "g-1"
    with mock.patch.object(base, "_send_single_request",
                           side_effect=_serve({"a": "Paris", "b": "Lyon"})), \
            mock.patch.object(frames, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(frames, "_load_frames_samples", lambda limit=None: samples), \
            mock.patch.object(frames, "judge_generate_cascade", _judge):
        res = frames.run_frames("m", "http://x/v1", 1, max_output_tokens=512)
    assert res["accuracy"] == 100.0 and res["judge_failures"] == 1
    assert "1 autorater repl(ies) without a Decision line" in res["leaderboard_comparable_reason"]
    failed = next(t for t in res["sample_traces"] if t["judge_grade"] == "JUDGE_FAILED")
    assert failed["scoring_excluded"] is True and failed["judge_model"] == "g-1"
    assert "left out of the accuracy" in _doc("frames")


# --------------------------------------------------------------------------- #
# humanitys_last_exam
# --------------------------------------------------------------------------- #
def _hle(replies, judge_text, finish="stop", golds=None):
    samples = [([{"role": "system", "content": hle.SYSTEM_PROMPT},
                 {"role": "user", "content": q}], (golds or {}).get(q, "42"), {"category": "M"})
               for q in replies]
    prompts = []

    async def _judge(prompt, config=None):
        prompts.append(prompt)
        return judge_text(prompt), "gemini-3.7-flash"
    with mock.patch.object(base, "_send_single_request",
                           side_effect=_serve(replies, finish_reason=finish)), \
            mock.patch.object(hle, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(hle, "_load_humanitys_last_exam_samples",
                              lambda limit=None: samples), \
            mock.patch.object(hle, "judge_generate_cascade", _judge):
        return hle.run_humanitys_last_exam("m", "http://x/v1", 1, max_output_tokens=512), prompts


_YES = json.dumps({"extracted_final_answer": "42", "reasoning": "r", "correct": "yes",
                   "confidence": 90, "strict": True})


def test_hle_judge_failure_is_excluded_and_named():
    res, _ = _hle({"q1": "Answer: 42", "q2": "Answer: 7"},
                  lambda p: _YES if "Answer: 42" in p else '{"extracted_final_answer": "7", ')
    assert res["accuracy"] == 100.0 and res["judge_failures"] == 1
    failed = next(t for t in res["sample_traces"] if t["judge_grade"] == "JUDGE_FAILED")
    assert failed["scoring_excluded"] is True and failed["judge_reply_complete_json"] is False
    assert res["judge_replies_incomplete_json"] == 1
    reason = res["leaderboard_comparable_reason"]
    assert "1 judge repl(ies) without a yes/no verdict left out of the score" in reason
    assert "upstream counts an unjudged question as wrong" in reason
    ok = next(t for t in res["sample_traces"] if t["judge_grade"] == "yes")
    assert ok["judge_reply"] == _YES and ok["judge_model"] == "gemini-3.7-flash"


def test_hle_gold_answer_enters_the_judge_prompt_unchanged():
    _res, prompts = _hle({"q1": "Answer: 42"}, lambda p: _YES, golds={"q1": " 4.328"})
    assert "[correct_answer]:  4.328\n" in prompts[0]


def test_hle_reply_at_the_end_of_the_window_is_the_protocol_outcome():
    looping = "loop " * 100
    res, _ = _hle({"q1": looping + "Answer: 42", "q2": "Answer: 42"}, lambda p: _YES,
                  finish="length")
    assert res["replies_cut_off_by_budget"] == 0
    assert "cut off by this run's token budget" not in res["leaderboard_comparable_reason"]
    assert res["status"] == "success" and res["replies_at_context_window"] == 1
    assert res["looping_responses"] == 1
    assert res["official_sampling"]["send_max_tokens"] is False


def test_hle_doc_states_the_judge_failure_rule():
    doc = _doc("humanitys_last_exam")
    assert "left out of the accuracy" in doc and "replies_at_context_window" in doc
    assert "scored wrong, counted in `judge_failures`" not in doc


# --------------------------------------------------------------------------- #
# healthbench
# --------------------------------------------------------------------------- #
def _healthbench(grader, replies=None):
    rubric = json.dumps([{"criterion": "Says hi", "points": 5, "tags": ["axis:x"]},
                         {"criterion": "Is rude", "points": -2, "tags": []}])
    samples = [([{"role": "user", "content": q}], rubric,
                {"category": "theme:x", "example_tags": ["theme:x"]}) for q in ("h1", "h2")]

    async def _judge(prompt, config=None):
        return grader(prompt)
    with mock.patch.object(base, "_send_single_request",
                           side_effect=_serve(replies or {"h1": "hi", "h2": "hey"})), \
            mock.patch.object(healthbench, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(healthbench, "_load_healthbench_samples",
                              lambda limit=None: samples), \
            _patch_detailed(healthbench, _judge):
        return healthbench.run_healthbench("m", "http://x/v1", 1, max_output_tokens=512)


def test_healthbench_every_rubric_verdict_names_its_model():
    res = _healthbench(lambda p: ('{"explanation": "e", "criteria_met": %s}'
                                  % ("true" if "Says hi" in p else "false"),
                                  "gemini-3.6-flash" if "Says hi" in p else "gemini-3.8-flash"))
    t = res["sample_traces"][0]
    models = {g["criterion"]: (g["judge_model"], g["grader_attempts"]) for g in t["rubric_grades"]}
    assert models == {"Says hi": ("gemini-3.6-flash", 1), "Is rude": ("gemini-3.8-flash", 1)}
    assert res["accuracy"] == 100.0


def test_healthbench_grader_failure_keeps_the_replies_for_audit():
    res = _healthbench(lambda p: ("not json", "gemini-3.8-flash") if "Is rude" in p and
                       "hey" in p else ('{"explanation": "e", "criteria_met": true}', "g"))
    failed = next(t for t in res["sample_traces"] if t.get("judge_grade") == "JUDGE_FAILED")
    assert failed["judge_failed_items"] == [{"criterion": "Is rude", "judge_model":
                                             "gemini-3.8-flash", "grader_attempts": 3,
                                             "judge_finish_reason": None,
                                             "judge_reply": "not json"}]


def test_healthbench_with_nothing_scored_withholds_the_score():
    def grader(p):
        if "hi" in p.split("# Rubric item")[0].split("assistant: ")[-1]:
            return None, "JUDGE_OUTAGE"                 # h1: judge outage
        return "never json", "gemini-3.8-flash"          # h2: grader failure
    res = _healthbench(grader)
    assert res["judge_outages"] == 1 and res["judge_failures"] == 1
    assert res["accuracy"] is None and res["status"] == "failed"
    assert "no conversation received a HealthBench score" in res["accuracy_withheld_reason"]


# --------------------------------------------------------------------------- #
# wildbench
# --------------------------------------------------------------------------- #
def _wb(judge, pool=None, pool_replies=None, replies=None):
    samples = [([{"role": "user", "content": q}], ["c"],
                {"category": cat, "wb_task_groups": ["Creative Tasks"], "session_id": q,
                 "wb_pairwise_target_is_a": True})
               for q, cat in (("s1", "Brainstorming"), ("s2", "Editing"), ("s3", "Editing"))]
    refs = {p: {"s1": "r", "s2": "r", "s3": "r"} for p in wildbench.REFERENCE_MODELS}
    with mock.patch.object(base, "_send_single_request",
                           side_effect=_serve(replies or {"s1": "reply-one", "s2": "reply-two",
                                                          "s3": "reply-three"})), \
            mock.patch.object(wildbench, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(wildbench, "_load_wildbench_samples", lambda **k: samples), \
            mock.patch.object(wildbench, "_load_reference_outputs", lambda: refs), \
            mock.patch.object(wildbench, "_load_wb_elo_pool_outputs",
                              lambda: iter(pool or [])), \
            mock.patch.object(wildbench, "_load_wb_elo_pool_replies",
                              lambda c: (pool_replies or {})[c]), \
            mock.patch.object(wildbench, "judge_generate_cascade_detailed", _detailed(judge)):
        return wildbench.run_wildbench("m", "http://x/v1", 1, max_output_tokens=512)


_REFUSED = {"finish_reason": None, "max_tokens_without_text": 0, "refused": True,
            "block_reasons": ["prompt_feedback:PROHIBITED_CONTENT"]}


def _wb_judge(prompt, config):
    if "reply-one" in prompt:                      # every judge call on task s1 is refused
        return None, "JUDGE_OUTAGE", dict(_REFUSED)
    if "<|begin_of_response_A|>" in prompt:
        return '{"choice": "A=B"}', "gemini-3.7-flash", {"finish_reason": "STOP"}
    return '{"score": "8"}', "gemini-3.8-flash", {"finish_reason": "STOP"}


def test_wildbench_judge_calls_send_srcs_max_tokens():
    configs = []

    def judge(prompt, config):
        configs.append(config)
        return _wb_judge(prompt, config)
    _wb(judge)
    # every call site sends src/eval.py's 1024 for the verdict plus the thinking allowance
    assert configs and all(c.max_output_tokens == 1024 + 8192 and c.temperature == 0.0
                           and c.response_mime_type == "application/json" for c in configs)
    rec = wildbench.judge_sampling_record()
    assert rec["max_output_tokens"] == 1024 + 8192 and "1024" in rec["max_output_tokens_note"]


def test_wildbench_refusal_is_not_an_outage_and_is_named():
    res = _wb(_wb_judge)
    t1 = next(t for t in res["sample_traces"] if t["extra_payload"]["session_id"] == "s1")
    assert t1["judge_grade"] == wildbench.JUDGE_REFUSED and t1["scoring_excluded"] is True
    assert t1["judge_block_reasons"] == ["prompt_feedback:PROHIBITED_CONTENT"]
    assert all(r["judge_grade"] == wildbench.JUDGE_REFUSED for r in t1["wb_pairwise"].values())
    assert res["judge_outages"] == 0
    assert res["judge_refusals"]["wb_score"] == 1 and res["judge_refusals"]["wb_reward"] == 3
    assert res["judge_refusals"]["tasks"] == ["s1"]
    assert res["wb_reward_judge_refusals"] == {p: 1 for p in wildbench.REFERENCE_MODELS}
    assert res["status"] == "completed_with_errors" and "judge_refusals" in res["status_reason"]
    reason = res["leaderboard_comparable_reason"]
    assert "the Gemini judge refused the task" in reason and "PROHIBITED_CONTENT" in reason
    assert "tasks the Gemini judge refused: s1" in reason
    assert res["wb_score"] == pytest.approx((8 - 5) * 20)     # s2, s3 only


def test_wildbench_score_mode_outage_is_named():
    def judge(prompt, config):
        if "reply-two" in prompt and "<|begin_of_response_A|>" not in prompt:
            return None, "JUDGE_OUTAGE", {"finish_reason": None}
        return _wb_judge(prompt, config)
    res = _wb(judge)
    assert res["judge_outages"] == 1
    assert "1 WB-Score repl(ies) not graded (judge outage)" in res["leaderboard_comparable_reason"]
    assert "not all 3" in res["leaderboard_comparable_reason"]


def test_wildbench_pool_refusals_are_counted_and_not_cached():
    tasks = {"s1": "pool-1", "s2": "pool-2", "s3": "pool-3"}
    pool = [("cfgA", dict(tasks))]

    def judge(prompt, config):
        if "pool-1" in prompt:
            return None, "JUDGE_OUTAGE", dict(_REFUSED)
        return _wb_judge(prompt, config)
    res = _wb(judge, pool=pool, pool_replies={"cfgA": dict(tasks)})
    detail = res["wb_elo_detail"]
    assert detail["pool_judge_refusals"] == 1 and detail["pool_judge_outages"] == 0
    assert detail["pool_judge_refused_sessions"] == {"s1": 1}
    cached = (Path(detail["pool_cache_dir"]) / "cfgA.jsonl").read_text().splitlines()
    assert len(cached) == 2 and all(json.loads(c)["session_id"] != "s1" for c in cached)
    assert "1 WB-Elo pool repl(ies) not graded (judge refusal)" in \
        res["leaderboard_comparable_reason"]


def test_wildbench_category_rows_show_the_wb_score():
    def judge(prompt, config):
        if "<|begin_of_response_A|>" in prompt:
            return '{"choice": "A=B"}', "g", {}
        score = {"reply-one": "9", "reply-two": "6", "reply-three": "4"}[
            wildbench.score_output_from_prompt(prompt)]
        return '{"score": "%s"}' % score, "g", {}
    res = _wb(judge)
    cells = res["category_accuracy"]
    assert cells["Brainstorming"]["accuracy"] == pytest.approx((9 - 5) * 20)
    assert cells["Brainstorming"]["pass_rate_at_7"] == 100.0
    assert cells["Editing"]["accuracy"] == pytest.approx(((6 + 4) / 2 - 5) * 20)
    assert cells["Editing"]["pass_rate_at_7"] == 0.0


def test_wildbench_doc_states_the_judge_cap_and_refusals():
    doc = _doc("wildbench")
    for needle in ("max_output_tokens` 9216", "JUDGE_REFUSED", "judge_refusals",
                   "pass_rate_at_7", "WB-Score repl"):
        assert needle in doc, needle


# --------------------------------------------------------------------------- #
# gdpval
# --------------------------------------------------------------------------- #
_MARK = "__MODEL__"


def _gd_trace(task_id="t1", response=f"deliverable {_MARK}", status=None):
    tr = {"extra_payload": {"task_id": task_id},
          "gold_answer": json.dumps([{"score": 5, "criterion": "c"}]),
          "response_text": response, "messages": [{"role": "user", "content": "Do it."}]}
    if status is not None:
        tr["status"] = status
    return tr


def _gd_score(traces, refs, winner="A"):
    metrics = {}

    async def fake(prompt):
        return json.dumps({"winner": winner, "reason": "why"}), "gemini-3.6-flash"
    with mock.patch.object(gdpval, "judge_generate_cascade", fake):
        asyncio.run(gdpval._make_scorer(None, 4, metrics, refs)(traces))
    return metrics["gdpval_report"]


_REF = {"t1": {"text": "expert", "note": None, "n_files": 1},
        "t2": {"text": "expert", "note": None, "n_files": 1}}


@pytest.mark.parametrize("status", ["FAILED", "TIMEOUT", "OVER_CONTEXT"])
def test_gdpval_failed_generation_is_a_harness_error_not_a_loss(status):
    traces = [_gd_trace("t1"), _gd_trace("t2", response=None, status=status)]
    rep = _gd_score(traces, _REF)
    failed = traces[1]
    assert failed["status"] == status and failed["scoring_excluded"] is True
    assert failed["generation_failed"] == status and "pairwise_score" not in failed
    assert rep["comparisons_in_win_rate"] == 1 and rep["losses"] == 0
    assert rep["comparisons_generation_failed"] == {status: 1}


def test_gdpval_empty_deliverable_is_still_a_loss_and_keeps_its_status():
    traces = [_gd_trace("t1", response="", status="EMPTY_RESPONSE")]
    rep = _gd_score(traces, _REF)
    assert traces[0]["pairwise_score"] == 0.0 and traces[0]["status"] == "EMPTY_RESPONSE"
    assert rep["losses"] == 1 and rep["win_rate"] == 0.0


def test_gdpval_each_comparison_records_its_judge_models_and_replies():
    traces = [_gd_trace("t1")]
    _gd_score(traces, _REF, winner="tie")
    d = traces[0]["pairwise_detail"]
    assert d["call1_judge_model"] == d["call2_judge_model"] == "gemini-3.6-flash"
    assert json.loads(d["call1_judge_reply"]) == {"winner": "tie", "reason": "why"}
    assert d["consistent"] is True


def _gd_run(traces_replies, refs, winner="A"):
    samples = [([{"role": "user", "content": q}], json.dumps([{"score": 5, "criterion": "c"}]),
                {"category": "Occ", "task_id": tid}) for q, tid in traces_replies]

    async def fake(prompt):
        return json.dumps({"winner": winner}), "gemini-3.6-flash"
    replies = {q: (None if q.startswith("fail") else f"answer {q}") for q, _ in traces_replies}
    with mock.patch.object(base, "_send_single_request", side_effect=_serve(replies)), \
            mock.patch.object(gdpval, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(gdpval, "_load_gdpval_samples",
                              lambda limit=None: (samples, refs)), \
            mock.patch.object(gdpval, "judge_generate_cascade", fake):
        return gdpval.run_gdpval("m", "http://x/v1", 1, max_output_tokens=512)


def test_gdpval_run_with_nothing_scored_withholds_the_score():
    refs = {"t1": {"text": None, "note": "none", "n_files": 0}}
    res = _gd_run([("q1", "t1")], refs)
    assert res["accuracy"] is None and res["status"] == "failed"
    # base.run_eval_suite withholds it first (every comparison left out of the score)
    assert res["accuracy_withheld_reason"].startswith("nothing was scored: ")
    assert "scoring_excluded 3" in res["accuracy_withheld_reason"]
    assert res["status_reason"] == res["accuracy_withheld_reason"]


def test_gdpval_run_names_failed_generations_and_sets_effective_n():
    res = _gd_run([("q1", "t1"), ("fail2", "t2")], _REF)
    assert res["status"] == "completed_with_errors"
    assert res["comparisons_generation_failed"] == 3             # 3 samples per task
    assert "generation(s) whose request failed (FAILED 3)" in res["leaderboard_comparable_reason"]
    assert res["effective_n"] == 1 and res["tasks_scored"] == 1
    assert res["judge_models"] == {"gemini-3.6-flash": 6}
    assert res["judge_cascade"] == base.judge_cascade()


def test_gdpval_doc_states_the_harness_error_rule():
    doc = _doc("gdpval")
    assert "comparisons_generation_failed" in doc and "call1_judge_model" in doc
    assert "An empty/failed model deliverable **is** counted, as a loss." not in doc
