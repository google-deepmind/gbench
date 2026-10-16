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

"""The knowledge and reasoning suites follow their upstream request settings and shard order:

* gpqa, gpqa_diamond and multilingual_mmlu send what simple-evals' chat sampler sends, the path
  a served model takes in both modes: every request carries ChatCompletionSampler's system
  message and temperature 0.5, a no-think request its max_tokens=2048, a thinking request the
  run's budget (the 2048 is not applied to reasoning);
* mmlu_pro sends evaluate_from_apiX.py's max_tokens=32768 in both modes and names failed
  questions;
* mmlu_redux is non-comparable when too many replies are not a bare option letter;
* the gsm8k docs attribute the stop-sequence cut to the right lm-eval backend;
* `--shard` is applied before the `--eval-limit` draw in every loader that draws the limit
  itself (cruxeval, lab_bench, gpqa over questions, multilingual_mmlu, mmlu_pro, ifeval,
  longbench_v2_code).

No network and no model server: datasets and endpoints are faked.
"""

from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import (arc_agi, base, cruxeval, gpqa, gpqa_diamond, gsm8k,
                                        ifeval, lab_bench, longbench_v2_code, mmlu_pro,
                                        mmlu_redux, multilingual_mmlu)
from gbench.runners.eval_suites.sampling import (limit_dataset, parse_shard, shard_select,
                                                 stratified_indices, stratified_sample)

REPO = Path(__file__).resolve().parent.parent
SYSTEM = {"role": "system", "content": "You are a helpful assistant."}


def _doc(suite):
    return (REPO / "docs" / "evals" / f"{suite}.md").read_text(encoding="utf-8")


def _fake_hf(rows_by_key):
    """load_dataset stand-in keyed by (path, config, split), (path, config) or path; asserts a
    pinned revision is requested."""
    def _load(path, *args, **kwargs):
        assert kwargs.get("revision"), f"{path} loaded without a pinned revision"
        name = args[0] if args else kwargs.get("name")
        for k in ((path, name, kwargs.get("split")), (path, name), path):
            if k in rows_by_key:
                return list(rows_by_key[k])
        raise AssertionError(f"unexpected dataset request {(path, name, kwargs.get('split'))}")
    return mock.patch("datasets.load_dataset", side_effect=_load)


def _gpqa_rows(n, sub=("Physics", "Chemistry", "Biology")):
    return [{"Question": f"Q{i}?", "Correct Answer": f"right{i}", "Incorrect Answer 1": f"w1{i}",
             "Incorrect Answer 2": f"w2{i}", "Incorrect Answer 3": f"w3{i}",
             "Subdomain": sub[i % len(sub)]} for i in range(n)]


_MMMLU_ROW = {"Question": "Quelle est la capitale?", "A": "Paris", "B": "Lyon", "C": "Nice",
              "D": "Metz", "Answer": "A", "Subject": "geography"}


# =========================================================================== #
# simple-evals samplers: system message, temperature and the chat sampler's max_tokens
# =========================================================================== #
def test_simple_evals_reference_names_the_gpt_4o_entry_precisely():
    ref = gpqa.SIMPLE_EVALS_CHAT_SAMPLING.reference
    assert "'gpt-4o' entry" in ref and "system_message=OPENAI_SYSTEM_MESSAGE_API" in ref
    assert "max_tokens=2048" in ref and "temperature=0.5" in ref
    assert "every non-reasoning chat model" not in ref
    assert gpqa.SIMPLE_EVALS_SYSTEM_MESSAGE == "You are a helpful assistant."
    assert gpqa.SIMPLE_EVALS_CHAT_MAX_TOKENS == 2048
    for module in (gpqa, gpqa_diamond, multilingual_mmlu):
        assert "every non-reasoning chat model" not in (module.__doc__ or "")


def test_simple_evals_messages_follow_the_sampler():
    user = [{"role": "user", "content": "q"}]
    assert gpqa.simple_evals_messages(user, thinking=False) == [SYSTEM] + user
    # A served model takes the chat sampler in both modes (the o-series samplers, keyed on
    # OpenAI model names, are not reasoning settings for it).
    assert gpqa.simple_evals_messages(user, thinking=True) == [SYSTEM] + user


@pytest.mark.parametrize("module,config", [(gpqa, "gpqa_main"), (gpqa_diamond, "gpqa_diamond")])
def test_gpqa_no_think_request_carries_the_chat_sampler_system_message(module, config):
    with _fake_hf({("Idavidrein/gpqa", config, "train"): _gpqa_rows(1)}):
        nothink = module._load_gpqa_samples(False)
        think = module._load_gpqa_samples(True)
    assert len(nothink) == len(think) == 10
    for (m0, g0, meta0), (m1, g1, meta1) in zip(nothink, think):
        assert m0[0] == SYSTEM and [m["role"] for m in m0] == ["system", "user"]
        # The same request in both modes: the chat sampler is a served model's path.
        assert m1 == m0
        # The chat sampler's 2048 is declared once (gpqa.SIMPLE_EVALS_SAMPLING), not per sample.
        assert m0[-1] == m1[-1] and g0 == g1 and meta0 == meta1
        assert "max_tokens" not in meta0 and "max_tokens" not in meta1


def test_multilingual_no_think_request_carries_the_chat_sampler_system_message():
    rows = {(multilingual_mmlu.DATASET, cfg, "test"): [_MMMLU_ROW]
            for cfg in multilingual_mmlu.MMMLU_CONFIGS}
    with _fake_hf(rows):
        nothink = multilingual_mmlu._load_multilingual_mmlu_samples(enable_thinking=False)
        think = multilingual_mmlu._load_multilingual_mmlu_samples(enable_thinking=True)
    assert len(nothink) == 14
    assert all(m[0] == SYSTEM and len(m) == 2 for m, _, _ in nothink)
    # The same request in both modes: the chat sampler is a served model's path.
    assert [m for m, _, _ in think] == [m for m, _, _ in nothink]


def _traces(tokens, finish="stop", sent=2048):
    return [{"sample_idx": i, "completion_tokens": n, "is_correct": True,
             "finish_reason": finish, "max_tokens_sent": sent} for i, n in enumerate(tokens)]


def test_no_think_requests_are_sent_the_chat_sampler_cap():
    res = gpqa._flag_comparability({"sample_traces": _traces([2048, 10], finish="length")},
                                   None, 448, 448, 10)
    # A reply cut at the sampler's max_tokens is the sampler's own outcome: counted, comparable.
    assert res["sampler_max_tokens"] == 2048 and res["replies_at_sampler_max_tokens"] == 2
    assert res["leaderboard_comparable"] is True and "leaderboard_comparable_reason" not in res
    shrunk = gpqa._flag_comparability({"sample_traces": _traces([10], sent=1500)}, None, 448,
                                      448, 10)
    assert shrunk["leaderboard_comparable"] is False
    assert "sent max_tokens [1500]" in shrunk["leaderboard_comparable_reason"]


def test_thinking_runs_have_no_sampler_cap():
    # A thinking request is sent the run's budget; the chat sampler's 2048 is not applied.
    res = gpqa._flag_comparability({"sample_traces": _traces([40000], sent=65536)}, None, 448,
                                   448, 10, thinking=True)
    assert res["sampler_max_tokens"] is None and res["replies_at_sampler_max_tokens"] == 0
    assert res["leaderboard_comparable"] is True
    assert gpqa.SIMPLE_EVALS_SAMPLING.reasoning is None
    assert gpqa.SIMPLE_EVALS_CHAT_SAMPLING.send_max_tokens is True


@pytest.mark.parametrize("module,run", [(gpqa, "run_gpqa"), (gpqa_diamond, "run_gpqa_diamond")])
def test_gpqa_run_sends_the_cap_only_without_thinking(module, run):
    seen = []

    def fake_run(**kw):
        seen.append(kw)
        return {"sample_traces": _traces([3000] * len(kw["samples"]), finish="length")}
    for thinking in (False, True):
        items = gpqa._gpqa_items(_gpqa_rows(2), 10, None, "gpqa", thinking)
        with mock.patch.object(gpqa, "_load_gpqa_items", return_value=items), \
                mock.patch.object(gpqa, "run_eval_suite", side_effect=fake_run):
            res = getattr(module, run)("m", "http://x/v1", 1, enable_thinking=thinking)
        # No per-sample cap: the official sampling carries the chat sampler and its 2048, and
        # base's token-limit policy sends the cap without thinking and the run's budget with it.
        caps = {s[2].get("max_tokens") for s in seen[-1]["samples"]}
        assert caps == {None}
        official = seen[-1]["official_sampling"]
        assert official is gpqa.SIMPLE_EVALS_SAMPLING
        assert official.for_mode(thinking) is official and official.max_tokens == 2048
        plan = base.resolve_token_budget("gpqa", official, thinking=thinking,
                                         max_output_tokens=65536)
        assert (plan.policy, plan.max_tokens) == (("reasoning_run_budget", 65536) if thinking
                                                  else ("official", 2048))
        assert res["replies_at_sampler_max_tokens"] == (0 if thinking else 20)
        assert res["leaderboard_comparable"] is True


def test_multilingual_run_sends_the_cap_only_without_thinking():
    rows = {(multilingual_mmlu.DATASET, cfg, "test"): [_MMMLU_ROW]
            for cfg in multilingual_mmlu.MMMLU_CONFIGS}
    for thinking in (False, True):
        with _fake_hf(rows):
            samples = multilingual_mmlu._load_multilingual_mmlu_samples(enable_thinking=thinking)
        assert {m.get("max_tokens") for _, _, m in samples} == {None}   # declared once, in
        # gpqa.SIMPLE_EVALS_SAMPLING
        with mock.patch.object(multilingual_mmlu, "_load_multilingual_mmlu_samples",
                               return_value=samples), \
                mock.patch.object(multilingual_mmlu, "run_eval_suite",
                                  return_value={"sample_traces": _traces([2048])}):
            res = multilingual_mmlu.run_multilingual_mmlu("m", "http://x/v1", 1,
                                                          enable_thinking=thinking)
        assert res["sampler_max_tokens"] == (None if thinking else 2048)


def test_gpqa_end_to_end_request_and_shard(monkeypatch):
    """Through base: a no-think request is [system, user] at gbench's 0.0; under --shard
    each question keeps all 10 copies and the result records the shard."""
    sent = []

    async def fake_send(**kw):
        sent.append(kw)
        return base.Reply("Answer: A", None, "stop", completion_tokens=12)
    monkeypatch.setattr(base, "_send_single_request", fake_send)
    monkeypatch.setenv("GBENCH_SHARD", "1/2")
    with mock.patch.object(gpqa, "_load_gpqa_rows", return_value=_gpqa_rows(4)):
        res = gpqa.run_gpqa("m", "http://x/v1", 1)
    assert res["shard"] == "1/2"
    assert res["repeats"]["questions"] == 2 and res["repeats"]["generations"] == 20
    assert {t["source_sample_idx"] for t in res["sample_traces"]} == {0, 2}
    assert len(sent) == 20
    assert all(kw["messages"][0] == SYSTEM and kw["temperature"] == 0.0 for kw in sent)
    assert res["official_sampling"]["temperature"] == 0.5
    # The chat sampler's max_tokens reaches the request whatever --max-output-tokens says
    # (base sends it as the request's max_tokens, not as a payload copy that would override
    # the context clamp).
    assert all(kw["max_output_tokens"] == 2048 and "max_tokens" not in kw["extra_payload"]
               for kw in sent)
    assert res["max_tokens_sent"] == [2048] and res["replies_at_sampler_max_tokens"] == 0
    sent.clear()
    monkeypatch.delenv("GBENCH_SHARD")
    with mock.patch.object(gpqa, "_load_gpqa_rows", return_value=_gpqa_rows(2)):
        res = gpqa.run_gpqa("m", "http://x/v1", 1, enable_thinking=True)
    assert "shard" not in res and len(sent) == 20
    # The chat sampler in both modes: [system, user], here at gbench's thinking temperature 1.0
    # and the run's budget instead of the 2048 (no
    # --max-output-tokens here, so the thinking floor).
    assert all(kw["messages"][0] == SYSTEM and len(kw["messages"]) == 2
               and kw["temperature"] == 1.0 for kw in sent)
    assert res["temperature_source"] == "thinking"
    assert all(kw["max_output_tokens"] == base.thinking_run_budget("gpqa")
               and "max_tokens" not in kw["extra_payload"] for kw in sent)
    assert res["official_sampling"]["max_tokens_applied"] is False


# =========================================================================== #
# mmlu_pro: evaluate_from_apiX.py's max_tokens in both modes, failed questions
# =========================================================================== #
def _pro(outcomes, thinking=False, **kwargs):
    samples = [([{"role": "user", "content": f"p{i}"}], "A", {"category": "c", "row": i})
               for i in range(len(outcomes))]

    async def fake_harness(*args, **kw):
        return [{"category": "c", "predicted": None, "response_text": None, "status": "OK",
                 "is_correct": o.get("predicted") == "A", **o} for o in outcomes]
    with mock.patch.object(mmlu_pro, "_load_mmlu_pro_samples", return_value=samples), \
            mock.patch.object(mmlu_pro, "_run_harness", side_effect=fake_harness):
        return mmlu_pro.run_mmlu_pro("m", "http://x/v1", 1, enable_thinking=thinking, **kwargs)


def test_mmlu_pro_sends_evaluate_from_apix_max_tokens_in_both_modes(monkeypatch):
    ok = [{"response_text": "The answer is (A).", "predicted": "A"}]
    for thinking in (False, True):
        res = _pro(ok, thinking=thinking)
        assert res["max_tokens_sent"] == [32768] and res["harness"]["max_tokens"] == 32768
        assert "upstream_max_tokens" not in res and "replies_over_upstream_max_tokens" not in res
        # gbench samples at 1.0 / 0.0, not upstream's 0.1, so neither mode is comparable
        assert res["leaderboard_comparable"] is False
        assert res["leaderboard_comparable_reason"] == (
            "sampled at temperature 1.0 (thinking; evaluate_from_apiX.py sends temperature 0.1)"
            if thinking else
            "sampled at temperature 0.0 (default; evaluate_from_apiX.py sends temperature 0.1)")
    # Below, upstream's temperature is set explicitly, so only the token limits differ.
    monkeypatch.setenv("GBENCH_MMLU_PRO_TEMPERATURE", "0.1")
    # The token-limit policy: a no-think run sends 32768 whatever --max-output-tokens says; a
    # thinking run is never capped below --max-output-tokens (evaluate_from_apiX.py's 32768 is
    # its reasoning-model setting too), which makes it a different measurement.
    res = _pro(ok, thinking=False, max_output_tokens=65536)
    assert res["max_tokens_sent"] == [32768] and res["leaderboard_comparable"] is True
    res = _pro(ok, thinking=True, max_output_tokens=65536)
    assert res["max_tokens_sent"] == [65536] and res["harness"]["max_tokens"] == 65536
    assert res["leaderboard_comparable"] is False
    assert "max_tokens=65536" in res["leaderboard_comparable_reason"]
    # A question whose three tries all passed the token limit scores 0 and stays comparable.
    res = _pro([{"status": "MAX_TOKENS_EXCEEDED"}, ok[0]])
    assert res["max_tokens_exceeded"] == 1 and res["accuracy"] == 50.0
    assert res["leaderboard_comparable"] is True


def test_mmlu_pro_names_failed_questions_apart_from_letterless_replies():
    res = _pro([{"status": "FAILED", "error": "APIConnectionError: Connection error."},
                {"response_text": "no letter here", "predicted": None}])
    reason = res["leaderboard_comparable_reason"]
    # a harness error, left out of the score (never a wrong answer)
    assert ("1 question(s) got no reply from the served endpoint after 10 tries (harness errors, "
            "left out of the score") in reason
    assert "letter" not in reason
    assert res["failed_requests"] == 1 and res["no_answer_replies"] == 1
    assert res["status"] == "completed_with_errors"


# =========================================================================== #
# mmlu_redux: many non-letter replies make the run non-comparable
# =========================================================================== #
def _redux(reply_for):
    rows = {(mmlu_redux.DATASET, s, "test"): [
        {"question": f"{s} q{i}?", "choices": ["a", "b", "c", "d"], "answer": 0,
         "error_type": "ok"} for i in range(2)] for s in mmlu_redux.MMLU_REDUX_SUBJECTS}
    rows[(mmlu_redux.MMLU_DATASET, "all", "dev")] = [
        {"question": f"d{i}", "choices": ["a", "b", "c", "d"], "answer": 0, "subject": s}
        for s in mmlu_redux.MMLU_REDUX_SUBJECTS for i in range(5)]

    async def fake_generate(base_url, model_name, prompts, concurrency, temperature):
        return [base.Reply(reply_for(i), None, "length") for i in range(len(prompts))]
    with _fake_hf(rows), mock.patch.object(mmlu_redux, "_generate_all",
                                           side_effect=fake_generate):
        return mmlu_redux.run_mmlu_redux("m", "http://x/v1", 1)


def test_mmlu_redux_many_non_letter_replies_make_the_run_non_comparable():
    res = _redux(lambda i: "")
    assert res["non_letter_replies"] == 114 and res["accuracy"] == 0.0
    assert res["leaderboard_comparable"] is False
    assert "114 of 114 replies (non_letter_replies) are not a bare option letter" in res[
        "leaderboard_comparable_reason"]


def test_mmlu_redux_a_few_non_letter_replies_stay_comparable():
    # 11 of 114 (9.6%) is under the 10% threshold; 12 (10.5%) is over it.
    res = _redux(lambda i: "The" if i < 11 else "A")
    assert res["non_letter_replies"] == 11 and res["leaderboard_comparable"] is True
    res = _redux(lambda i: "The" if i < 12 else "A")
    assert res["leaderboard_comparable"] is False
    assert "non_letter_replies" in res["leaderboard_comparable_reason"]


# =========================================================================== #
# gsm8k: the stop-sequence cut is attributed to the right lm-eval backend
# =========================================================================== #
def test_gsm8k_docs_attribute_the_post_reasoning_cut_to_the_hf_backend():
    doc = _doc("gsm8k")
    assert "as lm-eval's `postprocess_generated_text` cuts it" not in doc
    assert "`stop=None`" in doc and "HF" in doc and "models/huggingface.py" in doc
    assert "which leaves the same text" not in doc
    for text in (gsm8k.__doc__, gsm8k._load_gsm8k_samples.__doc__):
        assert "HF backend" in text and "which leaves the same text" not in text
    assert "as postprocess_generated_text does" not in gsm8k.__doc__


# =========================================================================== #
# --shard before --eval-limit in the loaders that draw the limit themselves
# =========================================================================== #
def _check_shard_then_limit(full, sharded_limited, spec, limit, key, seed):
    """`sharded_limited` is the stratified `limit` draw from shard `spec` of `full`."""
    shard = shard_select(full, parse_shard(spec))
    assert sharded_limited == stratified_sample(shard, limit, key, seed=seed)
    assert len(sharded_limited) == min(limit, len(shard))


def _crux_rows(n):
    return [{"code": f"def f(x):\n    return x + {i}", "input": str(i), "output": str(2 * i),
             "id": f"sample_{i}"} for i in range(n)]


def test_cruxeval_shards_before_the_limit(monkeypatch):
    with mock.patch("datasets.load_dataset", return_value=_crux_rows(12)):
        full = cruxeval._load_cruxeval_samples()
        assert cruxeval._load_cruxeval_samples(limit=5) == stratified_sample(
            full, 5, lambda s: s[2]["category"], seed="cruxeval")          # unchanged unsharded
        monkeypatch.setenv("GBENCH_SHARD", "2/3")
        limited = cruxeval._load_cruxeval_samples(limit=3)
        sharded = cruxeval._load_cruxeval_samples()
    _check_shard_then_limit(full, limited, "2/3", 3, lambda s: s[2]["category"], "cruxeval")
    assert sharded == shard_select(full, (2, 3))


def _lab_rows(category, n):
    return [{"id": f"{category}{i}", "question": f"{category} q{i}?", "ideal": "yes",
             "distractors": ["no", "maybe"], "protocol": "P: ", "paper-title": "T",
             "source": "doi", "subtask": category} for i in range(n)]


def test_lab_bench_shards_before_the_limit(monkeypatch):
    text_only = {c: _lab_rows(c, 3) for c, _ in lab_bench.CATEGORIES
                 if c not in ("FigQA", "TableQA")}
    with mock.patch.object(lab_bench, "_load_category",
                           side_effect=lambda c: list(text_only.get(c, []))):
        full = lab_bench._load_lab_bench_samples()
        monkeypatch.setenv("GBENCH_SHARD", "1/4")
        limited = lab_bench._load_lab_bench_samples(limit=2)
        sharded = lab_bench._load_lab_bench_samples()
    assert len(full) == 18 and sharded == shard_select(full, (1, 4))
    _check_shard_then_limit(full, limited, "1/4", 2, lambda s: s[2]["category"], "lab_bench")


def test_gpqa_shards_questions_before_the_limit_and_keeps_every_repeat(monkeypatch):
    rows = _gpqa_rows(9)
    everything = {(it["question_idx"], it["repeat_idx"]): it["permutation"]
                  for it in gpqa._gpqa_items(rows, 10, None, "gpqa")}
    assert len(everything) == 90

    def drawn(questions, limit):
        tagged = [dict(rows[q], _gbench_qidx=q) for q in questions]
        return {r["_gbench_qidx"] for r in limit_dataset(tagged, limit, "Subdomain", seed="gpqa")}
    unsharded = gpqa._gpqa_items(rows, 10, 2, "gpqa")
    assert {it["question_idx"] for it in unsharded} == drawn(range(9), 2)   # unchanged unsharded
    monkeypatch.setenv("GBENCH_SHARD", "3/3")
    items = gpqa._gpqa_items(rows, 10, None, "gpqa")
    assert {it["question_idx"] for it in items} == {2, 5, 8} and len(items) == 30
    limited = gpqa._gpqa_items(rows, 10, 2, "gpqa")
    assert {it["question_idx"] for it in limited} == drawn([2, 5, 8], 2) and len(limited) == 20
    # Every copy keeps the permutation a full simple-evals run gives it.
    assert all(everything[(it["question_idx"], it["repeat_idx"])] == it["permutation"]
               for it in items + limited)


def test_multilingual_mmlu_shards_before_the_limit(monkeypatch):
    rows = {(multilingual_mmlu.DATASET, cfg, "test"): [dict(_MMMLU_ROW, Question=f"{cfg} {i}")
                                                        for i in range(2)]
            for cfg in multilingual_mmlu.MMMLU_CONFIGS}
    with _fake_hf(rows):
        full = multilingual_mmlu._load_multilingual_mmlu_samples()
        monkeypatch.setenv("GBENCH_SHARD", "2/5")
        limited = multilingual_mmlu._load_multilingual_mmlu_samples(limit=3)
    assert len(full) == 28
    _check_shard_then_limit(full, limited, "2/5", 3, lambda s: s[2]["category"],
                            "multilingual_mmlu")


def test_mmlu_pro_shards_before_the_limit(monkeypatch):
    from datasets import Dataset
    rows = [{"question": f"q{c}-{i}", "options": ["a", "b", "c", "N/A"], "answer_index": i % 3,
             "category": f"cat{c}"} for c in range(4) for i in range(5)]
    with mock.patch("datasets.load_dataset", return_value=Dataset.from_list(rows)):
        full = mmlu_pro._load_mmlu_pro_samples(n_shot=0)
        keys = [r["category"] for r in rows]
        assert mmlu_pro._load_mmlu_pro_samples(limit=6, n_shot=0) == [
            full[i] for i in stratified_indices(keys, 6, "mmlu_pro")]      # unchanged unsharded
        monkeypatch.setenv("GBENCH_SHARD", "2/3")
        limited = mmlu_pro._load_mmlu_pro_samples(limit=4, n_shot=0)
    _check_shard_then_limit(full, limited, "2/3", 4, lambda s: s[2]["category"], "mmlu_pro")


def _ifeval_rows(n):
    return [{"key": 1000 + i, "prompt": f"Write {i} words.",
             "instruction_id_list": ["length_constraints:number_words"],
             "kwargs": [{"relation": "at least", "num_words": i + 1}]} for i in range(n)]


def test_ifeval_shards_before_the_limit(monkeypatch):
    with mock.patch("datasets.load_dataset", return_value=_ifeval_rows(10)):
        full = ifeval._load_ifeval_samples()
        assert ifeval._load_ifeval_samples(limit=4) == stratified_sample(
            full, 4, None, seed="ifeval")                                   # unchanged unsharded
        monkeypatch.setenv("GBENCH_SHARD", "1/3")
        limited = ifeval._load_ifeval_samples(limit=2)
    _check_shard_then_limit(full, limited, "1/3", 2, None, "ifeval")


def _longbench_rows(n):
    return [{"_id": f"id{i}", "domain": ("Code Repository Understanding", "Single-Document QA")[
        i % 2], "context": f"context {i}", "question": f"q{i}", "choice_A": "a",
        "choice_B": "b", "choice_C": "c", "choice_D": "d", "answer": "A", "difficulty": "easy",
        "length": "short"} for i in range(n)]


def test_longbench_v2_code_shards_before_the_limit(monkeypatch):
    monkeypatch.setenv("GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS", "1")
    with mock.patch("datasets.load_dataset", return_value=_longbench_rows(10)):
        full = longbench_v2_code._load_longbench_v2_code_samples()
        monkeypatch.setenv("GBENCH_SHARD", "2/2")
        limited = longbench_v2_code._load_longbench_v2_code_samples(limit=3)
    assert len(full) == 10
    _check_shard_then_limit(full, limited, "2/2", 3, lambda s: s[2]["category"], "culer")


@pytest.mark.parametrize("module,run,loader,kwargs", [
    (cruxeval, "run_cruxeval", "_load_cruxeval_samples", {}),
    (lab_bench, "run_lab_bench", "_load_lab_bench_samples", {}),
    (multilingual_mmlu, "run_multilingual_mmlu", "_load_multilingual_mmlu_samples", {}),
    (ifeval, "run_ifeval", "_load_ifeval_samples", {}),
])
def test_runs_hand_base_exactly_this_shards_samples(monkeypatch, module, run, loader, kwargs):
    """The loader drew the shard and the limit; base must keep exactly those samples and draw
    no second limit."""
    monkeypatch.setenv("GBENCH_SHARD", "2/3")
    samples = [([{"role": "user", "content": f"p{i}"}], "A", {"category": "c"})
               for i in range(4)]
    seen = []

    def fake_run(**kw):
        seen.append(kw)
        return {"sample_traces": [], "temperature": None, "attempts": {}}
    patches = [mock.patch.object(module, loader, return_value=samples),
               mock.patch.object(module, "run_eval_suite", side_effect=fake_run)]
    if module is cruxeval:
        patches += [mock.patch.object(cruxeval, "sandbox_skip_reason", return_value=None),
                    mock.patch.object(cruxeval, "_grader_self_check", return_value=None)]
    if module is ifeval:
        patches.append(mock.patch.object(ifeval, "_scorer", return_value=None))
    for p in patches:
        p.start()
    try:
        getattr(module, run)("m", "http://x/v1", concurrency=1, limit=2, **kwargs)
    finally:
        for p in reversed(patches):
            p.stop()
    assert seen
    for kw in seen:
        assert kw["limit"] is None
        assert shard_select(kw["samples"], parse_shard("2/3")) == samples


def test_longbench_run_hands_base_exactly_this_shards_samples(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "2/3")
    base.set_run_knobs(max_model_len=262144)
    samples = [([{"role": "user", "content": f"p{i}"}], "A", {"category": "c"})
               for i in range(3)]
    calls = []
    monkeypatch.setattr(longbench_v2_code, "_load_tokenizer", lambda name: object())
    monkeypatch.setattr(longbench_v2_code, "_load_longbench_v2_code_samples", lambda **kw: samples)
    monkeypatch.setattr(longbench_v2_code, "run_eval_suite",
                        lambda **kw: calls.append(kw) or {"sample_traces": [], "temperature": 0.1})
    longbench_v2_code.run_longbench_v2_code("m", "http://x/v1", limit=2)   # conftest restores knobs
    # One call per setting's first query (no CoT reply, so no answer query), each with the
    # samples unchanged; pred.py's max_tokens (128 non-CoT, 1024 CoT query) is each one's cap.
    assert [kw["official_sampling"].max_tokens for kw in calls] == [128, 1024]
    for kw in calls:
        assert kw["limit"] is None
        assert shard_select(kw["samples"], parse_shard("2/3")) == samples


def test_cruxeval_run_uses_the_same_layout_for_both_passes(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "1/2")
    with mock.patch("datasets.load_dataset", return_value=_crux_rows(3)):
        samples = cruxeval._load_cruxeval_samples()
    seen = []

    def fake_run(**kw):
        seen.append(kw)
        return {"sample_traces": [], "temperature": kw["official_sampling"].temperature,
                "attempts": {"attempts_per_sample": 10}, "failed_requests": 0}
    with mock.patch.object(cruxeval, "sandbox_skip_reason", return_value=None), \
            mock.patch.object(cruxeval, "_grader_self_check", return_value=None), \
            mock.patch.object(cruxeval, "_load_cruxeval_samples", return_value=samples), \
            mock.patch.object(cruxeval, "run_eval_suite", side_effect=fake_run):
        cruxeval.run_cruxeval("m", "http://x/v1", 1)
    assert len(seen) == 2
    assert all(shard_select(kw["samples"], (1, 2)) == samples for kw in seen)
    assert arc_agi._for_base(samples) == seen[0]["samples"]


# =========================================================================== #
# docs match the code
# =========================================================================== #
def test_docs_describe_the_request_settings_and_shard_order():
    for suite in ("gpqa", "gpqa_diamond", "multilingual_mmlu"):
        doc = _doc(suite)
        assert "`You are a helpful assistant.`" in doc and "`gpt-4o` entry" in doc
        assert "max_tokens=2048" in doc and "`replies_at_sampler_max_tokens`" in doc
        assert "every non-reasoning chat model" not in doc
        assert "default 0.5" in doc and "--shard I/N" in doc
        assert "The prompt is the same with and without" not in doc
    pro = _doc("mmlu_pro")
    assert "max_tokens=32768" in pro and "evaluate_from_apiX.py" in pro
    assert "max_tokens=4000" not in pro and "`replies_over_upstream_max_tokens`" not in pro
    assert "--shard I/N" in pro
    redux = _doc("mmlu_redux")
    assert "flagged non-comparable" in redux and "`non_letter_replies` above 10%" in redux
    for suite in ("cruxeval", "lab_bench", "ifeval", "longbench_v2_code"):
        assert "--shard I/N" in _doc(suite), suite
    for suite in ("gpqa", "gpqa_diamond", "multilingual_mmlu", "mmlu_pro", "mmlu_redux", "gsm8k",
                  "cruxeval", "lab_bench", "ifeval", "longbench_v2_code"):
        doc = _doc(suite)
        run = doc.split("## Run", 1)[1]
        assert f"--evals {suite}" in run and "--max-output-tokens" in run, suite
        assert doc.isascii() or suite == "multilingual_mmlu", suite


def test_module_headers_describe_the_shard_order():
    for module in (gpqa, gpqa_diamond, multilingual_mmlu, mmlu_pro, cruxeval, lab_bench, ifeval,
                   longbench_v2_code):
        assert "--shard I/N" in module.__doc__ and "before the `--eval-limit` draw" in (
            " ".join(module.__doc__.split())), module.__name__
    assert "flagged `leaderboard_comparable=False`" in " ".join(mmlu_redux.__doc__.split())
