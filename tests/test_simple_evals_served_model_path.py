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

"""The simple-evals suites send a served model simple-evals' chat sampler in both modes.

simple_evals.py (openai/simple-evals @ 652c89d0, lines 145-212) sends OpenAI's o-series models
through ResponsesSampler(reasoning_model=True) and OChatCompletionSampler (no system message, no
temperature, no max_tokens). Those entries are keyed on OpenAI model names, so they are not
reasoning settings for a served model: gpqa, gpqa_diamond, multilingual_mmlu, simpleqa and
healthbench send the `gpt-4o` entry's ChatCompletionSampler request (system message "You are a
helpful assistant.", temperature 0.5) with and without --thinking, except that a --thinking
request is sampled at temperature 1.0 (base.resolve_temperature: an official temperature is not
applied to reasoning; recorded as not matching). The token-limit policy sends its max_tokens=2048
exactly without --thinking and the run's --max-output-tokens with it (the 2048 is recorded as not
applied to reasoning).

No network, model server or judge is contacted: datasets, requests and judges are faked.
"""

import json
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import (base, gpqa, gpqa_diamond, healthbench,
                                        multilingual_mmlu, simpleqa)

def _patch_detailed(module, judge):
    """Patch a suite's judge_generate_cascade_detailed with a (text, model) judge stub; the
    detailed call returns (text, model, info) with nothing cut off at the grader's cap."""
    async def detailed(prompt, config=None):
        text, model = await judge(prompt, config=config)
        return text, model, {"finish_reason": None, "max_tokens_without_text": 0}
    return mock.patch.object(module, "judge_generate_cascade_detailed", detailed)


REPO = Path(__file__).resolve().parent.parent
SUITE_DIR = REPO / "gbench" / "runners" / "eval_suites"
SUITES = ("gpqa", "gpqa_diamond", "multilingual_mmlu", "simpleqa", "healthbench")
SYSTEM = {"role": "system", "content": "You are a helpful assistant."}


def _simpleqa_detailed(judge):
    """A (text, model) judge stub as simpleqa's detailed cascade call (text, model, info)."""
    async def _detailed(prompt, config=None):
        text, model = await judge(prompt, config=config)
        return text, model, {}
    return _detailed


def _doc(suite):
    """The suite's doc with whitespace runs folded, so a phrase may wrap across lines."""
    return " ".join((REPO / "docs" / "evals" / f"{suite}.md").read_text(encoding="utf-8").split())


def _folded(text):
    return " ".join((text or "").split())


def _gpqa_rows(n):
    return [{"Question": f"Q{i}?", "Correct Answer": f"right{i}", "Incorrect Answer 1": "w1",
             "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3", "Subdomain": "Physics"}
            for i in range(n)]


_MMMLU_ROW = {"Question": "Quelle est la capitale?", "A": "Paris", "B": "Lyon", "C": "Nice",
              "D": "Metz", "Answer": "A", "Subject": "geography"}


def _mmmlu_load(path, *args, **kwargs):
    assert kwargs.get("revision"), "openai/MMMLU loaded without a pinned revision"
    return [dict(_MMMLU_ROW)]


def _judge_simpleqa(prompt, config=None):
    async def _run():
        return "A", "gemini-x"
    return _run()


def _judge_healthbench(prompt, config=None):
    async def _run():
        return '{"explanation": "ok", "criteria_met": true}', "gemini-x"
    return _run()


def _run_suite(suite, thinking, reply_text="Answer: A", finish="stop", judge=None, **kwargs):
    """Run `suite` through base with a faked endpoint (and judge, for the judged suites);
    returns (result, requests seen)."""
    seen = []

    async def _send(**kw):
        seen.append(kw)
        return base.Reply(reply_text, None, finish, completion_tokens=7)

    with mock.patch.object(base, "_send_single_request", side_effect=_send):
        if suite in ("gpqa", "gpqa_diamond"):
            module = gpqa if suite == "gpqa" else gpqa_diamond
            with mock.patch.object(gpqa, "_load_gpqa_rows", return_value=_gpqa_rows(1)):
                res = getattr(module, f"run_{suite}")("m", "http://x/v1", 1,
                                                      enable_thinking=thinking, **kwargs)
        elif suite == "multilingual_mmlu":
            with mock.patch("datasets.load_dataset", side_effect=_mmmlu_load):
                res = multilingual_mmlu.run_multilingual_mmlu("m", "http://x/v1", 1,
                                                              enable_thinking=thinking, **kwargs)
        elif suite == "simpleqa":
            samples = [([{"role": "user", "content": "Who?"}], "Ada", {"category": "x"})]
            with mock.patch.object(simpleqa, "gemini_required_skip", lambda *a: None), \
                    mock.patch.object(simpleqa, "_load_simpleqa_samples",
                                      lambda limit=None: samples), \
                    mock.patch.object(simpleqa, "judge_generate_cascade_detailed",
                                      _simpleqa_detailed(judge or _judge_simpleqa)):
                res = simpleqa.run_simpleqa("m", "http://x/v1", 1, enable_thinking=thinking,
                                            **kwargs)
        else:
            rubric = json.dumps([{"criterion": "Says hi", "points": 5, "tags": []}])
            samples = [([{"role": "user", "content": "hello"}], rubric,
                        {"category": "theme:x", "example_tags": ["theme:x"]})]
            with mock.patch.object(healthbench, "gemini_required_skip", lambda *a: None), \
                    mock.patch.object(healthbench, "_load_healthbench_samples",
                                      lambda limit=None: samples), \
                    _patch_detailed(healthbench,
                                      judge or _judge_healthbench):
                res = healthbench.run_healthbench("m", "http://x/v1", 1,
                                                  enable_thinking=thinking, **kwargs)
    return res, seen


# --------------------------------------------------------------------------- #
# the declaration
# --------------------------------------------------------------------------- #
def test_the_declaration_has_no_reasoning_block():
    full = gpqa.SIMPLE_EVALS_SAMPLING
    assert full.reasoning is None and full.for_mode(True) is full
    assert (full.temperature, full.send_temperature, full.send_max_tokens,
            full.max_tokens) == (0.5, True, True, 2048)
    assert not hasattr(gpqa, "SIMPLE_EVALS_REASONING_SAMPLING")
    # gpqa_diamond runs through gpqa._run_simple_evals_gpqa; the others import the declaration
    for module in (multilingual_mmlu, simpleqa, healthbench):
        assert module.SIMPLE_EVALS_SAMPLING is full
    assert "'gpt-4o' entry" in full.reference and "temperature=0.5" in full.reference


@pytest.mark.parametrize("thinking,policy,sent", [(False, "official", 2048),
                                                  (True, "reasoning_run_budget", 65536)])
def test_the_policy_sends_the_cap_without_thinking_and_the_run_budget_with_it(thinking, policy,
                                                                             sent):
    plan = base.resolve_token_budget("gpqa", gpqa.SIMPLE_EVALS_SAMPLING, thinking=thinking,
                                     max_output_tokens=65536)
    assert (plan.policy, plan.max_tokens, plan.official_cap) == (policy, sent, 2048)
    temperature, source = base.resolve_official_sampling("gpqa", gpqa.SIMPLE_EVALS_SAMPLING,
                                                         thinking=thinking)
    assert (temperature, source) == ((1.0, "thinking") if thinking else (0.0, "default"))


def test_the_helpers_are_the_same_in_both_modes():
    user = [{"role": "user", "content": "q"}]
    assert gpqa.simple_evals_messages(user, True) == gpqa.simple_evals_messages(user, False)
    assert gpqa.simple_evals_messages(user, True) == [SYSTEM] + user
    # one sampler declaration, the same in both modes, carrying the chat sampler's 2048
    sampling = gpqa.SIMPLE_EVALS_SAMPLING
    assert sampling.for_mode(True) is sampling.for_mode(False) is sampling
    assert sampling.max_tokens == 2048
    assert not hasattr(gpqa, "simple_evals_sampling") and not hasattr(gpqa, "simple_evals_meta")
    samples = [([{"role": "user", "content": "Who?"}], "Ada", {"category": "x"})]
    assert (simpleqa.simple_evals_samples(samples, True)
            == simpleqa.simple_evals_samples(samples, False))


# --------------------------------------------------------------------------- #
# through base: the request each mode sends
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("suite", SUITES)
def test_a_thinking_run_sends_the_chat_sampler_with_the_run_budget(suite):
    base.set_run_knobs(max_output_tokens=65536)
    res, seen = _run_suite(suite, thinking=True)
    assert seen
    for kw in seen:
        assert kw["messages"][0] == SYSTEM and kw["messages"][-1]["role"] == "user"
        assert kw["temperature"] == 1.0              # not the sampler's 0.5: a thinking run
        assert kw["max_output_tokens"] == 65536
        assert "max_tokens" not in (kw["extra_payload"] or {})
    official = res["official_sampling"]
    assert "reasoning_settings" not in official and "send_max_tokens" not in official
    assert official["temperature"] == 0.5 and official["send_temperature"] is True
    assert official["temperature_matches_official"] is False
    assert res["temperature_source"] == "thinking"
    assert (official["max_tokens"], official["max_tokens_applied"],
            official["max_tokens_sent"]) == (2048, False, 65536)
    assert official["max_tokens_note"] == base.REASONING_CAP_NOTE
    assert res["max_output_tokens_note"].startswith(base.REASONING_CAP_NOTE)
    assert "--max-output-tokens 65536" in res["max_output_tokens_note"]
    assert res["sampler_max_tokens"] is None and res["replies_at_sampler_max_tokens"] == 0
    # the temperature is the only difference from the protocol here, and it is named
    assert res["leaderboard_comparable"] is False
    assert "temperature 1.0 (" in res["leaderboard_comparable_reason"]


@pytest.mark.parametrize("suite", SUITES)
def test_the_two_modes_differ_only_in_max_tokens_and_temperature(suite):
    base.set_run_knobs(max_output_tokens=65536)
    _, nothink = _run_suite(suite, thinking=False)
    _, think = _run_suite(suite, thinking=True)
    # The requests are dispatched concurrently, so compare them as multisets, not in order.
    def _sorted(calls):
        return sorted(json.dumps(kw["messages"], sort_keys=True) for kw in calls)
    assert _sorted(think) == _sorted(nothink) and len(think) == len(nothink)
    assert {kw["temperature"] for kw in nothink} == {0.0}
    assert {kw["temperature"] for kw in think} == {1.0}
    assert {kw["max_output_tokens"] for kw in nothink} == {2048}
    assert {kw["max_output_tokens"] for kw in think} == {65536}


@pytest.mark.parametrize("suite", SUITES)
def test_a_smaller_run_budget_is_sent_too(suite):
    # the 2048 is replaced by the run's budget with --thinking, larger or smaller
    base.set_run_knobs(max_output_tokens=1024)
    res, seen = _run_suite(suite, thinking=True)
    assert {kw["max_output_tokens"] for kw in seen} == {1024}
    assert res["official_sampling"]["max_tokens_sent"] == 1024


def test_a_thinking_reply_cut_at_the_run_budget_is_not_comparable():
    base.set_run_knobs(max_output_tokens=65536)
    res, _ = _run_suite("gpqa", thinking=True, finish="length")
    assert res["truncated_responses"] == 10 and "replies_at_official_max_tokens" not in res
    assert res["replies_at_sampler_max_tokens"] == 0
    assert res["leaderboard_comparable"] is False
    assert ("10 repl(ies) stopped at the run's token budget (--max-output-tokens, or a "
            "max_tokens the context clamp lowered; the chat sampler's 2048 is not applied to "
            "reasoning)" in res["leaderboard_comparable_reason"])


@pytest.mark.parametrize("suite", ["simpleqa", "healthbench"])
def test_judged_suites_count_a_thinking_reply_cut_at_the_run_budget(suite):
    base.set_run_knobs(max_output_tokens=65536)
    res, _ = _run_suite(suite, thinking=True, reply_text="Ada" if suite == "simpleqa" else "hi",
                        finish="length")
    assert res["replies_cut_off_by_budget"] == 1
    assert "cut off by this run's token budget" in res["leaderboard_comparable_reason"]


def test_healthbench_grades_the_system_turn_with_thinking_too():
    graded = []

    async def _judge(prompt, config=None):
        graded.append(prompt)
        return '{"explanation": "ok", "criteria_met": true}', "gemini-x"

    base.set_run_knobs(max_output_tokens=65536)
    res, _ = _run_suite("healthbench", thinking=True, reply_text="hi", judge=_judge)
    # upstream grade_sample reads the chat sampler's actual_queried_message_list
    assert "system: You are a helpful assistant.\n\nuser: hello" in graded[0]
    assert res["accuracy"] == 100.0


@pytest.mark.parametrize("suite,needle", [
    ("gpqa", "temperature 0.0 (cli:--temperature; simple-evals' sampler sends temperature 0.5)"),
    ("simpleqa", "temperature 0.0 (simple-evals' chat sampler samples at 0.5)"),
    ("healthbench", "temperature 0.0 (simple-evals' chat sampler samples at 0.5)"),
])
def test_an_operator_temperature_on_a_thinking_run_is_named(suite, needle):
    base.set_run_knobs(max_output_tokens=65536, temperature=0.0)
    res, seen = _run_suite(suite, thinking=True)
    assert {kw["temperature"] for kw in seen} == {0.0}
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert res["leaderboard_comparable"] is False
    assert needle in res["leaderboard_comparable_reason"]
    assert "reasoning models no temperature" not in res["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# docstrings and docs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("suite", SUITES)
def test_docstrings_no_longer_cite_the_reasoning_samplers_as_settings(suite):
    src = (SUITE_DIR / f"{suite}.py").read_text(encoding="utf-8")
    assert "SIMPLE_EVALS_REASONING_SAMPLING" not in src
    assert "OfficialSampling.reasoning" not in src
    assert "upstream's explicit reasoning-model settings" not in src
    assert "sends reasoning models no temperature" not in src
    header = _folded({"gpqa": gpqa, "gpqa_diamond": gpqa_diamond,
                      "multilingual_mmlu": multilingual_mmlu, "simpleqa": simpleqa,
                      "healthbench": healthbench}[suite].__doc__)
    assert "keyed on OpenAI model names" in header
    assert "not reasoning settings for a served model" in header


@pytest.mark.parametrize("suite", SUITES)
def test_docs_state_the_chat_sampler_in_both_modes(suite):
    doc = _doc(suite)
    assert "keyed on OpenAI model names" in doc
    assert "takes the chat sampler in both modes" in doc
    assert "lines 145-212" in doc and "lines 236-240" in doc
    assert "`max_tokens_applied: false`" in doc and "`max_tokens_sent`" in doc
    for stale in ("OfficialSampling.reasoning", "simple-evals has no path of its own",
                  "with **no temperature**", "a reasoning row", "thinking request sends none",
                  "none with it", "the upstream's explicit reasoning-model settings"):
        assert stale not in doc, (suite, stale)


def test_new_and_changed_text_is_ascii():
    # multilingual_mmlu's code and doc quote simple-evals' non-English answer anchors verbatim,
    # so only its docstring and the doc's sampling section are checked
    paths = [REPO / "tests" / "test_simple_evals_served_model_path.py",
             *(SUITE_DIR / f"{s}.py" for s in SUITES if s != "multilingual_mmlu"),
             *(REPO / "docs" / "evals" / f"{s}.md" for s in SUITES if s != "multilingual_mmlu")]
    for path in paths:
        path.read_text(encoding="utf-8").encode("ascii")
    multilingual_mmlu.__doc__.encode("ascii")
    doc = (REPO / "docs" / "evals" / "multilingual_mmlu.md").read_text(encoding="utf-8")
    doc[doc.index("## Request and sampling"):].encode("ascii")
