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

"""Canonical-fidelity tests: arc_agi, longbench_v2_code (was culer), ifeval, lab_bench, cruxeval, wmdp,
causalbench and gsm8k. No network, no model server: datasets and endpoints are faked."""

import asyncio
import hashlib
import json
import sys
import threading
import types
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parent.parent


def _fake_load_dataset(rows_by_key):
    """A `datasets.load_dataset` stand-in keyed by (path, config) or path; asserts a pinned
    revision is requested."""
    def _load(path, *args, **kwargs):
        assert kwargs.get("revision"), f"{path} loaded without a pinned revision"
        name = args[0] if args else kwargs.get("name")
        key = (path, name) if (path, name) in rows_by_key else path
        return rows_by_key[key]
    return mock.patch("datasets.load_dataset", side_effect=_load)


# --------------------------------------------------------------------------- #
# arc_agi (#02, #76)
# --------------------------------------------------------------------------- #
_ARC_UPSTREAM_PROMPT = (
    "You are participating in a puzzle solving competition. You are an expert at solving "
    "puzzles.\n\nBelow is a list of input and output pairs with a pattern. Your goal is to "
    "identify the pattern or transformation in the training examples that maps the input to "
    "the output, then apply that pattern to the test input to give a final output.\n\nRespond "
    "in the format of the training output examples\n\n--Training Examples--\n"
    "{training_examples}\n--End of Training Examples--\n\n--Test Input--\n{test_input}\n"
    "--End of Test Input--\n\nYour response:")


def test_arc_agi_prompt_is_the_upstream_system_prompt():
    from gbench.runners.eval_suites import arc_agi
    assert arc_agi._PROMPT_TEMPLATE == _ARC_UPSTREAM_PROMPT


def _arc_rows():
    g = [[1]]
    return [
        {"id": "t1", "train": [{"input": [[0]], "output": [[1]]}],
         "test": [{"input": [[2]], "output": g}]},
        {"id": "t2", "train": [{"input": [[0]], "output": [[1]]}],
         "test": [{"input": [[3]], "output": g}, {"input": [[4]], "output": [[2]]}]},
    ]


def _arc_trace(src, attempt, task_id, pair, correct, text="[[1]]"):
    return {"sample_idx": src * 2 + attempt, "source_sample_idx": src, "attempt_idx": attempt,
            "extra_payload": {"category": "grid_transformation", "task_id": task_id,
                              "pair_index": pair},
            "response_text": text, "is_correct": correct}


def test_arc_agi_headline_is_pass2_averaged_per_task_from_base_result_shape():
    """base.py returns the @k report under result["attempts"]; the headline must be pass@2
    per test pair, averaged per TASK (upstream score_task / score_submission)."""
    from gbench.runners.eval_suites import arc_agi
    traces = [
        _arc_trace(0, 0, "t1", 0, False), _arc_trace(0, 1, "t1", 0, True),     # t1 pair solved
        _arc_trace(1, 0, "t2", 0, True), _arc_trace(1, 1, "t2", 0, False),     # t2 pair 0 solved
        _arc_trace(2, 0, "t2", 1, False), _arc_trace(2, 1, "t2", 1, False),    # t2 pair 1 not
    ]
    base_result = {  # the real base.py shape: avg@k accuracy + "attempts" report
        "accuracy": 33.33, "total_questions": 6, "correct_answers": 2,
        "attempts": {"attempts_requested": 2, "attempts_per_sample": 2, "samples": 3,
                     "generations": 6, "avg_at_k": 33.33, "pass_at_k": 66.67,
                     "pass_hat_k": 0.0, "unstable_samples": 2},
        "category_accuracy": {}, "sample_traces": traces, "status": "success",
    }
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        return dict(base_result)

    with mock.patch.object(arc_agi, "_load_arc_agi_tasks", return_value=_arc_rows()), \
            mock.patch.object(arc_agi, "run_eval_suite", side_effect=fake_run):
        res = arc_agi.run_arc_agi("m", "http://x/v1", attempt_count=1)
    assert seen["attempt_count"] == 2 and seen["limit"] is None
    # t1 = 1/1, t2 = 1/2 -> (1 + 0.5) / 2 tasks = 75%, not 2/3 pairs = 66.67%
    assert res["accuracy"] == 75.0
    assert res["total_questions"] == 2 and res["correct_answers"] == 1.5
    assert res["pass_at_k_per_pair"] == 66.67 and res["avg_at_k_accuracy"] == 33.33
    assert res["arc_agi_report"]["test_pairs"] == 3
    assert res["metric"].startswith("pass@2 averaged per task")
    assert res["leaderboard_comparable"] is True


def test_arc_agi_limit_draws_whole_tasks_and_flags_unparsed_attempts():
    from gbench.runners.eval_suites import arc_agi
    with mock.patch.object(arc_agi, "_load_arc_agi_tasks", return_value=_arc_rows()):
        samples = arc_agi._load_arc_agi_samples(limit=1)
    task_ids = {m["task_id"] for _, _, m in samples}
    assert len(task_ids) == 1
    expected_pairs = {"t1": 1, "t2": 2}[task_ids.pop()]
    assert len(samples) == expected_pairs
    rep = arc_agi._score_per_task([_arc_trace(0, 0, "t1", 0, False, text="no grid here")])
    assert rep["unparsed_attempts"] == 1 and rep["score"] == 0.0


# --------------------------------------------------------------------------- #
# longbench_v2_code, formerly culer (#07, #41)
# --------------------------------------------------------------------------- #
def test_culer_extraction_is_longbench_v2_extract_answer():
    from gbench.runners.eval_suites.longbench_v2_code import _eval_longbench_v2_code as _eval_culer
    assert _eval_culer("The correct answer is (B)", "B")
    assert _eval_culer("**The correct answer is B**", "B")
    # false positives of a looser reader: an ordinary word must never read as an option letter
    for gold in "ABCD":
        assert not _eval_culer("Looking at each option carefully, it must be X. Answer: (B)",
                               gold)
        assert not _eval_culer("The answer depends on the config loader. Answer: (A)", gold)
    assert not _eval_culer("the correct answer is (b)", "B")     # upstream is case-sensitive


def test_culer_prompt_is_0shot_txt_and_truncation_keeps_head_and_tail():
    from gbench.runners.eval_suites import longbench_v2_code as culer
    assert hashlib.sha256(culer._TEMPLATE_0SHOT.encode()).hexdigest() == (
        "68a162252bc9ff71d5d7abca3d69bb31aac3c35f832d657a2866f2018b8a6950")
    item = {"context": "  DOC  ", "question": " Q? ", "choice_A": "a", "choice_B": "b",
            "choice_C": "c", "choice_D": "d"}
    p = culer._fill_prompt(item)
    assert "<text>\nDOC\n</text>" in p and "question: Q?\nChoices:\n(A) a\n" in p

    class CharTok:  # one token per character
        def encode(self, s):
            return [ord(c) for c in s]

        def decode(self, ids, skip_special_tokens=True):
            return "".join(chr(i) for i in ids)

    out, n, cut = culer._middle_truncate("HEAD" + "x" * 100 + "TAIL", CharTok(), 9)
    assert cut and n == 108
    # pred.py: input_ids[:max_len//2] + input_ids[-max_len//2:]  (-9//2 == -5)
    assert out == "HEAD" + "xTAIL"


def test_culer_hard_errors_without_a_window_and_flags_every_deviation(monkeypatch):
    from gbench.runners.eval_suites import base
    from gbench.runners.eval_suites import longbench_v2_code as culer
    monkeypatch.delenv("GBENCH_LONGBENCH_V2_CODE_WINDOW", raising=False)
    monkeypatch.delenv("GBENCH_CULER_WINDOW", raising=False)
    base._RUN_KNOBS.pop("max_model_len", None)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        culer.run_longbench_v2_code("m", "http://x/v1")

    base.set_run_knobs(max_model_len=131072)
    # The w/ CoT setting tokenizes its answer prompt (pred.py truncates every query).
    monkeypatch.setattr(culer, "_load_tokenizer", lambda name: _CharTok())
    monkeypatch.setattr(culer, "_load_longbench_v2_code_samples", lambda cot=False, **kw: [
        ([{"role": "user", "content": "p"}], "A",
         {"category": "Code Repository Understanding", "difficulty": "easy", "length": "short",
          "prompt_truncated": True, **({"cot_answer_prompt": "p $COT$"} if cot else {})})])
    seen = {}

    def fake_run(**kw):
        seen.update(kw)
        return {"accuracy": 100.0, "temperature": kw["official_sampling"].temperature,
                "sample_traces": [
                    {"response_text": "The correct answer is (A)", "is_correct": True,
                     "extra_payload": dict(kw["samples"][0][2])}]}   # the sample's meta, as base

    monkeypatch.setattr(culer, "run_eval_suite", fake_run)
    res = culer.run_longbench_v2_code("m", "http://x/v1")
    assert seen["official_sampling"].temperature == 0.1     # LongBench-v2's setting
    assert "temperature" not in seen
    # A 131,072-token window holds upstream's 120,000 plus the 128 budget: pred.py's max_len.
    assert res["longbench_v2_code_report"]["max_prompt_tokens"] == 120000
    assert res["longbench_v2_code_report"]["no_extractable_answer"] == 0
    # pred.py's non-CoT max_new_tokens=128 is the suite's official cap, applied by base.
    assert culer._output_budget(False) == 128
    assert seen["official_sampling"].max_tokens == 128
    assert all("max_tokens" not in meta for _, _, meta in seen["samples"])
    reason = res["leaderboard_comparable_reason"]
    assert res["leaderboard_comparable"] is False
    assert "Code Repository Understanding" in reason
    assert "120,000" not in reason          # truncated exactly as upstream
    assert "answer budget" not in reason
    res = culer.run_longbench_v2_code("m", "http://x/v1", enable_thinking=True)
    # pred.py samples every query at 0.1, so a thinking run does too (and is flagged).
    assert seen["official_sampling"].temperature == 0.1
    assert "--thinking" in res["leaderboard_comparable_reason"]
    assert "max_tokens=128" in res["leaderboard_comparable_reason"]
    assert "max_tokens" not in seen["samples"][0][2]


def test_culer_tokenizer_failure_is_a_hard_error(monkeypatch):
    from gbench.runners.eval_suites import longbench_v2_code as culer
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        mock.Mock(side_effect=OSError("no such repo")))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        culer._load_tokenizer("not/a-model")


# --------------------------------------------------------------------------- #
# ifeval (#12)
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_langdetect(monkeypatch):
    """langdetect is not in the test env; a stand-in that calls ASCII text English."""
    mod = types.ModuleType("langdetect")

    class LangDetectException(Exception):
        pass

    class DetectorFactory:
        seed = None

    mod.LangDetectException = LangDetectException
    mod.DetectorFactory = DetectorFactory
    mod.detect = lambda text: "en" if all(ord(c) < 128 for c in text) else "fr"
    monkeypatch.setitem(sys.modules, "langdetect", mod)
    from gbench.runners.eval_suites import ifeval
    from gbench.runners.eval_suites._ifeval_vendor import instructions
    monkeypatch.setattr(instructions, "langdetect", mod)
    ifeval._scorer.cache_clear()
    yield mod
    ifeval._scorer.cache_clear()


def _gold(ids, kwargs, prompt="p", key=1):
    return json.dumps({"key": key, "ids": ids, "kwargs": kwargs, "prompt": prompt})


def test_ifeval_missing_langdetect_is_a_hard_error(monkeypatch):
    from gbench.runners.eval_suites import ifeval
    monkeypatch.setitem(sys.modules, "langdetect", None)    # import raises ImportError
    ifeval._scorer.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="INFRA REQUIRED.*langdetect"):
            ifeval.run_ifeval("m", "http://x/v1", 1)
    finally:
        ifeval._scorer.cache_clear()


def test_ifeval_uses_the_upstream_checkers(fake_langdetect):
    """Four cases where the former hand-written checkers diverged from upstream's."""
    from gbench.runners.eval_suites.ifeval import _eval_ifeval
    paras = _gold(["length_constraints:number_paragraphs"], [{"num_paragraphs": 2}])
    assert _eval_ifeval("one\n***\ntwo", paras)
    assert not _eval_ifeval("one\n\ntwo", paras)             # blank lines are not the divider
    constrained = _gold(["detectable_format:constrained_response"], [{}])
    assert _eval_ifeval("Having thought about it: My answer is yes.", constrained)
    sections = _gold(["detectable_format:multiple_sections"],
                     [{"section_spliter": "Section", "num_sections": 2}])
    assert _eval_ifeval("Section 1\nintro\nSection 2\nbody", sections)
    assert not _eval_ifeval("This Section talks about another Section.", sections)
    forbidden = _gold(["keywords:forbidden_words"], [{"forbidden_words": ["cat"]}])
    assert _eval_ifeval("A category of concatenated words.", forbidden)
    assert not _eval_ifeval("The cat sat.", forbidden)
    lang = _gold(["language:response_language"], [{"language": "fr"}])
    assert _eval_ifeval("Tres bien, merci beaucoup. C'est la vie!", lang) is False
    assert _eval_ifeval("Tr\u00e8s bien, merci beaucoup.", lang) is True


def test_ifeval_loader_applies_upstream_fix_and_drops_null_kwargs():
    from gbench.runners.eval_suites import ifeval
    old, new = ifeval._UPSTREAM_PROMPT_FIXES[2785]
    rows = [{"key": 2785, "prompt": "Intro. " + old,
             "instruction_id_list": ["detectable_content:number_placeholders"],
             "kwargs": [{"num_placeholders": 3, "num_words": None, "language": None}]}]
    with _fake_load_dataset({"google/IFEval": rows}):
        samples = ifeval._load_ifeval_samples()
    gold = json.loads(samples[0][1])
    assert samples[0][0][0]["content"] == "Intro. " + new
    assert gold["kwargs"] == [{"num_placeholders": 3}]


def test_ifeval_random_letter_prompts_are_reproducible_and_report_has_four_metrics(
        fake_langdetect):
    from gbench.runners.eval_suites import ifeval
    g = _gold(["keywords:letter_frequency"],
              [{"let_relation": "at least", "letter": "!", "let_frequency": 6}], key=1129)
    text = "Great ice cream! So smooth and creamy, a real treat for everyone around here."
    verdicts = {tuple(ifeval._follow_lists(g, text)[0]) for _ in range(5)}
    assert len(verdicts) == 1
    traces = [{"gold_answer": _gold(["punctuation:no_comma", "startend:quotation"], [{}, {}]),
               "response_text": '"no commas here"'},
              {"gold_answer": _gold(["punctuation:no_comma", "startend:quotation"], [{}, {}]),
               "response_text": "a, b"}]
    rep = ifeval._ifeval_report(traces)
    assert rep["prompt_level_strict"] == 50.0 and rep["instruction_level_strict"] == 50.0
    assert {"prompt_level_loose", "instruction_level_loose", "by_instruction_strict"} <= set(rep)


# --------------------------------------------------------------------------- #
# lab_bench (#49)
# --------------------------------------------------------------------------- #
def test_lab_bench_prompt_has_refusal_choice_and_answer_tag_template():
    from gbench.runners.eval_suites import lab_bench as L
    assert hashlib.sha256(L.MCQ_INSTRUCT_TEMPLATE.encode()).hexdigest() == (
        "e1ec85a5ab68cd14f681f428a7a9736f295d2c53b68158ed0c91b3e840481a0f")
    choices, gold, refuse = L._randomize_choices("seed", "the ideal", ["d1", "d2", "d3"])
    assert len(choices) == 5
    assert f"({gold}) the ideal" in choices
    assert f"({refuse}) {L.REFUSE_CHOICE}" in choices
    prompt, gold2 = L._build_mcq_prompt("PROTOCOL\n", "Why?", "the ideal", ["d1", "d2", "d3"])
    assert "Question: PROTOCOL\nWhy?\n\nOptions:\n(A) " in prompt
    assert "Insufficient information to answer the question" in prompt
    assert prompt.endswith("We require this because we use automatic parsing.")
    assert "\nThink step by step.\n" in prompt


def test_lab_bench_answer_tag_decides_as_upstream_and_letters_are_per_item():
    from gbench.runners.eval_suites.lab_bench import _eval_lab_bench
    g5 = {"letter": "C", "letters": "ABCDE", "refuse": "B"}
    assert _eval_lab_bench("reasoning\n[ANSWER]C[/ANSWER]", g5)
    assert _eval_lab_bench("[ANSWER]C. Use 0.85 g NaCl[/ANSWER]", g5)
    # sorted(set(matches)) then the first offered letter: two tags keep the alphabetical first
    assert _eval_lab_bench("[ANSWER]D[/ANSWER] no wait [ANSWER]C[/ANSWER]", g5)
    assert not _eval_lab_bench("[ANSWER]D[/ANSWER] no wait [ANSWER]E[/ANSWER]",
                               {**g5, "letter": "E"})
    # a tag with a letter that was not offered is no answer, even with a later Final Answer
    assert not _eval_lab_bench("[ANSWER]H[/ANSWER]\nFinal Answer: C", g5)
    g8 = {"letter": "H", "letters": "ABCDEFGH", "refuse": "A"}
    assert _eval_lab_bench("[ANSWER]H[/ANSWER]", g8)
    assert not _eval_lab_bench("Final Answer: H", g8)        # untagged: LAB-Bench reads nothing


def test_lab_bench_report_precision_counts_the_refusal_as_unanswered():
    from gbench.runners.eval_suites.lab_bench import _lab_bench_report
    g = {"letter": "C", "letters": "ABCDE", "refuse": "B"}
    traces = [{"gold_answer": g, "response_text": "[ANSWER]C[/ANSWER]", "is_correct": True},
              {"gold_answer": g, "response_text": "[ANSWER]B[/ANSWER]", "is_correct": False},
              {"gold_answer": g, "response_text": "Final Answer: D", "is_correct": False},
              {"gold_answer": g, "response_text": "unsure", "is_correct": False}]
    rep = _lab_bench_report(traces)
    cat = rep["categories"]["unknown"]
    assert cat["accuracy"] == 25.0 and cat["coverage"] == 75.0 and cat["precision"] == 33.33
    assert rep["chose_refusal"] == 1 and rep["no_letter"] == 2


def test_lab_bench_result_discloses_scope():
    from gbench.runners.eval_suites import lab_bench as L
    sample = ([{"role": "user", "content": "p"}], {"letter": "A", "letters": "ABCDE",
                                                   "refuse": "B"}, {"category": "x"})
    with mock.patch.object(L, "_load_lab_bench_samples", return_value=[sample]), \
            mock.patch.object(L, "run_eval_suite", return_value={"sample_traces": [
                {"gold_answer": sample[1], "response_text": "Final Answer: A",
                 "is_correct": True}]}):
        res = L.run_lab_bench("m", "http://x/v1")
    reason = res["leaderboard_comparable_reason"]
    assert res["leaderboard_comparable"] is False
    assert "public split" in reason and "mean of the 8 category accuracies" in reason
    assert "ProtocolQA task only" not in reason and "no [ANSWER] tag" not in reason


# --------------------------------------------------------------------------- #
# cruxeval (#40)
# --------------------------------------------------------------------------- #
def test_cruxeval_prompts_render_exactly_as_upstream():
    from gbench.runners.eval_suites import cruxeval as C
    s = ("def f(nums):\n    return nums[::-1]", "[1, 2, 3]")
    want = {
        "make_direct_output_prompt": "26de3217393f51e282c22700f1e683db582af4bf10a39233df6c02485932ca9e",
        "make_cot_output_prompt": "6995f3551f26d3bc2ae0ea6274c1c8569eab566065b051bde3ca8d39897350d2",
        "make_direct_input_prompt": "588f5d08344e7f6cc40f30e01b9ea0543dcf6370e09ccb9f5017ff39211c0433",
        "make_cot_input_prompt": "d48f947f7c8d776d3717a0a7fd41f3e70b73a928dc80caba30022559a52e88eb",
    }
    for name, digest in want.items():
        assert hashlib.sha256(getattr(C, name)(s).encode()).hexdigest() == digest, name


def test_cruxeval_loader_builds_both_tasks_with_system_prompt_and_stop():
    from gbench.runners.eval_suites import cruxeval as C
    rows = [{"code": "def f(x):\n    return x + 1", "input": "1", "output": "2", "id": "s0"}]
    with _fake_load_dataset({"cruxeval-org/cruxeval": rows}):
        direct = C._load_cruxeval_samples()
        cot = C._load_cruxeval_samples(enable_thinking=True)
    assert sorted(m["category"] for _, _, m in direct) == ["input_prediction",
                                                           "output_prediction"]
    msgs, gold, meta = direct[0]
    assert msgs[0] == {"role": "system", "content": C._SYSTEM_PROMPT}
    assert meta["stop"] == ["[/ANSWER]"]
    assert all("stop" not in m for _, _, m in cot)          # thinking: no stop string
    assert all(g["cot"] for _, g, _ in cot)


def test_cruxeval_scores_by_execution_like_evaluate_score():
    from gbench.runners.eval_suites.cruxeval import _check
    out = {"code": "def f(x):\n    return len(x)", "input": "[1]", "output": "1",
           "mode": "output", "cot": False}
    assert _check("assert f([1]) == 1\n[/ANSWER]", out) == "passed"
    assert _check("assert f([1]) == 1.0", out) == "passed"       # executed ==, not typed
    assert _check("assert f([1]) == True", out) == "passed"
    assert _check("assert f([1]) == 2", out) == "failed"
    assert _check("assert f([1]) == f([1])", out) == "excluded"  # contains f(input)
    assert _check("Final Answer: 1", out) == "failed"            # no other answer format
    inp = {**out, "mode": "input"}
    assert _check("assert f([7]) == 1", inp) == "passed"          # any input that works
    assert _check("assert f([]) == 1", inp) == "failed"
    assert _check("[7]", inp) == "excluded"                       # no f( call
    cot = {**out, "cot": True}
    assert _check("[THOUGHT]\nlen is 1\n[/THOUGHT]\n[ANSWER]\nassert f([1]) == 1\n[/ANSWER]",
                  cot) == "passed"


def test_cruxeval_report_follows_evaluate_generations():
    from gbench.runners.eval_suites.cruxeval import _cruxeval_report

    def tr(src, mode, check):
        return {"source_sample_idx": src, "gold_answer": {"mode": mode},
                "cruxeval_check": check}
    traces = [tr(0, "output", "passed"), tr(0, "output", "excluded"),    # n shrinks to 1
              tr(1, "output", "failed"), tr(1, "output", "excluded"),    # all-False, n = 2
              tr(2, "input", "passed"), tr(2, "input", "failed")]
    rep = _cruxeval_report(traces)
    assert rep["CRUXEval-O"]["pass_at_1"] == 50.0          # (1/1 + 0/2) / 2
    assert rep["CRUXEval-I"]["pass_at_1"] == 50.0
    assert rep["CRUXEval-O"]["excluded_generations"] == 2


def test_cruxeval_hard_errors_when_the_sandbox_is_blocked(monkeypatch):
    from gbench.runners.eval_suites import cruxeval as C
    monkeypatch.setattr(C, "sandbox_skip_reason", lambda: "bubblewrap is not installed")
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        C.run_cruxeval("m", "http://x/v1", 1)


# --------------------------------------------------------------------------- #
# wmdp (#67)
# --------------------------------------------------------------------------- #
class _CharTok:
    """Character-level stand-in tokenizer (no BOS), so continuations are 2 tokens (' ', 'A')."""
    bos_token_id = None

    def encode(self, s):
        return [ord(c) for c in s]


def test_wmdp_context_is_lm_eval_v042_template():
    from gbench.runners.eval_suites import wmdp
    rows = {("cais/wmdp", sub): [{"question": " What? ", "choices": ["w", "x", "y", "z"],
                                  "answer": 2}] for sub, _ in wmdp._SUBSETS}
    with _fake_load_dataset(rows):
        samples = wmdp._load_wmdp_samples()
    bio = [s for s in samples if s["subset"] == "wmdp-bio"][0]
    assert bio["context"] == ("The following are multiple choice questions (with answers) about "
                              "biology.\n\nWhat?\nA. w\nB. x\nC. y\nD. z\nAnswer:")
    assert bio["gold"] == 2


def test_wmdp_encoder_matches_lm_eval_encode_pair_with_tokenizer_defaults():
    from gbench.runners.eval_suites.wmdp import _Encoder
    ctx, cont = _Encoder(_CharTok()).pair("Answer: ", "A")     # trailing space moves over
    assert "".join(map(chr, ctx)) == "Answer:" and "".join(map(chr, cont)) == " A"

    # A Gemma tokenizer that adds no BOS by default (Gemma 4) gets none, as in lm-eval.
    class GemmaTokenizerFast(_CharTok):
        bos_token_id = 2
    enc = _Encoder(GemmaTokenizerFast())
    ctx, cont = enc.pair("Answer:", " A")
    assert ctx[0] != 2 and "".join(map(chr, cont)) == " A" and enc.bos_prepended is False


def _serve(handler):
    """Run an aiohttp app on an ephemeral port in a background thread; returns (base_url, stop)."""
    from aiohttp import web
    loop = asyncio.new_event_loop()
    app = web.Application()
    app.router.add_post("/v1/completions", handler)
    runner = web.AppRunner(app)
    ready = threading.Event()
    port = {}

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", 0)
        loop.run_until_complete(site.start())
        port["p"] = site._server.sockets[0].getsockname()[1]
        ready.set()
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    ready.wait(10)

    def stop():
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
    return f"http://127.0.0.1:{port['p']}/v1", stop


def _wmdp_rows():
    from gbench.runners.eval_suites import wmdp
    return {("cais/wmdp", sub): [{"question": f"q{i}", "choices": ["w", "x", "y", "z"],
                                  "answer": i % 4} for i in range(4)]
            for sub, _ in wmdp._SUBSETS}


def test_wmdp_end_to_end_reads_echoed_prompt_logprobs(monkeypatch):
    from aiohttp import web
    from gbench.runners.eval_suites import wmdp
    seen = []

    async def handler(req):
        body = await req.json()
        seen.append(body)
        choices = []
        for i, ids in enumerate(body["prompt"]):
            # the model "prefers" the letter C: its continuation token scores highest
            lps = [None] + [(-0.1 if t == ord("C") else -2.0) for t in ids[1:]] + [-0.3]
            choices.append({"index": i, "logprobs": {"token_logprobs": lps}})
        return web.json_response({"choices": choices})

    url, stop = _serve(handler)
    monkeypatch.setattr(wmdp, "_load_tokenizer", lambda name: _CharTok())
    try:
        with _fake_load_dataset(_wmdp_rows()):
            res = wmdp.run_wmdp("m", url, concurrency=4)
    finally:
        stop()
    # lm-eval v0.4.2 local-completions' loglikelihood request, one context + continuation each.
    assert len(seen) == 12 * 4
    assert all(b["echo"] is True and b["logprobs"] == 10 and b["max_tokens"] == 0
               and b["temperature"] == 0.0 and b["seed"] == 1234 and len(b["prompt"]) == 1
               for b in seen)
    assert res["total_questions"] == 12 and res["correct_answers"] == 3   # answer index 2 = C
    assert res["accuracy"] == 25.0 and res["status"] == "success"
    assert res["scoring_mode"] == "loglikelihood" and res["leaderboard_comparable"] is True
    assert res["sample_traces"][0]["predicted"] == "C"


def test_wmdp_endpoint_without_echo_is_a_hard_error(monkeypatch):
    from aiohttp import web
    from gbench.runners.eval_suites import wmdp

    async def handler(req):   # a chat-style server that ignores echo: only the generated token
        body = await req.json()
        return web.json_response({"choices": [
            {"index": i, "logprobs": {"token_logprobs": [-0.5]}}
            for i in range(len(body["prompt"]))]})

    url, stop = _serve(handler)
    monkeypatch.setattr(wmdp, "_load_tokenizer", lambda name: _CharTok())
    try:
        with _fake_load_dataset(_wmdp_rows()), \
                pytest.raises(RuntimeError, match="INFRA REQUIRED.*echo"):
            wmdp.run_wmdp("m", url, concurrency=2)
    finally:
        stop()


# --------------------------------------------------------------------------- #
# gsm8k (#70), causalbench (#68)
# --------------------------------------------------------------------------- #
def test_gsm8k_default_is_lm_eval_gsm8k_cot_and_reason_names_deviations():
    from gbench.runners.eval_suites import gsm8k
    rows = [{"question": "1+1?", "answer": "so #### 2"}]
    with _fake_load_dataset({"openai/gsm8k": rows}):
        samples = gsm8k._load_gsm8k_samples()
    messages, gold, meta = samples[0]
    assert len(messages) == 2 * 8 + 1 and gold == "2" and meta["stop"] == ["Q:", "</s>", "<|im_end|>"]
    assert messages[0] == {"role": "user", "content": "Q: There are 15 trees in the grove. Grove "
                           "workers will plant trees in the grove today. After they are done, "
                           "there will be 21 trees. How many trees did the grove workers plant "
                           "today?\nA:"}
    assert messages[-1] == {"role": "user", "content": "Q: 1+1?\nA:"}
    with _fake_load_dataset({"openai/gsm8k": rows}), \
            mock.patch.object(gsm8k, "run_eval_suite",
                              return_value={"temperature": 1.0, "sample_traces": [],
                                            "official_sampling": {
                                                "temperature_matches_official": False}}):
        res = gsm8k.run_gsm8k("m", "http://x/v1", 1, eval_n_shot=5, limit=1)
    reason = res["leaderboard_comparable_reason"]
    assert res["leaderboard_comparable"] is False
    assert "5-shot" in reason and "temperature 1.0" in reason and "--eval-limit" in reason


def test_causalbench_f1_uses_the_scored_text():
    from gbench.runners.eval_suites import causalbench
    traces = [{"gold_answer": "entailment", "response_text": "<thought>No.</thought>Yes"},
              {"gold_answer": "contradiction", "response_text": "No"}]
    with mock.patch.object(causalbench, "_load_causalbench_samples", return_value=[]), \
            mock.patch.object(causalbench, "run_eval_suite",
                              return_value={"accuracy": 100.0, "sample_traces": traces}):
        res = causalbench.run_causalbench("m", "http://x/v1", 1)
    assert res["accuracy"] == 100.0 and res["causalbench_report"]["tp"] == 1
    assert res["raw_accuracy"] == 100.0 and res["leaderboard_comparable"] is True


# --------------------------------------------------------------------------- #
# docs match the code
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("suite", ["arc_agi", "longbench_v2_code", "ifeval", "lab_bench", "cruxeval",
                                   "wmdp", "causalbench", "gsm8k"])
def test_suite_doc_names_the_pinned_revision_and_run_command(suite):
    import importlib
    mod = importlib.import_module(f"gbench.runners.eval_suites.{suite}")
    doc = (REPO / "docs" / "evals" / f"{suite}.md").read_text(encoding="utf-8")
    assert mod._DATASET_REVISION in doc
    assert f"--evals {suite}" in doc and "## Requirements" in doc
    assert doc.isascii()


# =========================================================================== #
# arc_agi parsed as upstream, documented run commands, wmdp and gsm8k records
# =========================================================================== #
# sha256 of each function's source in arcprize/arc-agi-benchmarking
# src/arc_agi_benchmarking/utils/parsing.py at commit 28e67d54b05df5be10281892243c509a42a874f1.
_ARC_UPSTREAM_PARSER_SHA256 = {
    "backscan_json_parser": "36867864c08084490d77aee58ab7dd9c3fb307f7bf13cd70aade5c693f130ee1",
    "extract_from_boxed": "9d9e9b2fc7e79cbd2ba3c98708605891338422e77473ab84e3d8eb30798b3dac",
    "parse_and_validate_json": "4f92d288e6cb6e73a4c96b03d149733e49399e635799e76a289c6b7486bae3cb",
}


def test_arc_agi_parser_is_the_upstream_parser_verbatim():
    import inspect
    from gbench.runners.eval_suites import arc_agi
    for name, digest in _ARC_UPSTREAM_PARSER_SHA256.items():
        src = inspect.getsource(getattr(arc_agi, name)).rstrip("\n")
        assert hashlib.sha256(src.encode()).hexdigest() == digest, name


def test_arc_agi_first_boxed_grid_wins_like_upstream():
    from gbench.runners.eval_suites import arc_agi
    reply = ("\\boxed{[[1, 1], [1, 1]]}\nDouble-checking against example 2, the alternative "
             "would be [[2, 2], [2, 2]].")
    assert arc_agi._parse_grid(reply) == [[1, 1], [1, 1]]
    assert arc_agi._eval_arc_agi(reply, [[1, 1], [1, 1]]) is True
    assert arc_agi._eval_arc_agi(reply, [[2, 2], [2, 2]]) is False
    # No \boxed: the JSON closing at the reply's last bracket (upstream backscan), code fences
    # need no special handling.
    assert arc_agi._parse_grid("[[9]]\n```json\n[[3, 3], [3, 3]]\n```") == [[3, 3], [3, 3]]
    # The last bracket closes something that is not a list of lists: upstream raises (and
    # retries), gbench has no grid.
    assert arc_agi._parse_grid("Final grid:\n[[3, 3], [3, 3]]\nNote (see [0]).") is None


def test_arc_agi_full_run_of_first_boxed_replies_is_scored_as_upstream():
    """Reproduction: 400 tasks x 2 attempts whose replies box the right grid first and
    mention another later. Upstream scores 100%; so must gbench, and the run is comparable."""
    from gbench.runners.eval_suites import arc_agi as A
    gold = [[1, 1], [1, 1]]
    reply = "\\boxed{[[1, 1], [1, 1]]}\nThe alternative [[2, 2], [2, 2]] does not fit."
    samples = [([{"role": "user", "content": "p"}], gold,
                {"category": "grid_transformation", "task_id": f"t{i}", "pair_index": 0})
               for i in range(400)]

    def fake_run(**kw):
        traces = [{"sample_idx": 2 * i + a, "source_sample_idx": i,
                   "extra_payload": {"task_id": meta["task_id"], "pair_index": 0},
                   "response_text": reply, "is_correct": A._eval_arc_agi(reply, g)}
                  for i, (_, g, meta) in enumerate(kw["samples"]) for a in range(2)]
        return {"attempts": {"attempts_per_sample": 2}, "sample_traces": traces}

    with mock.patch.object(A, "_load_arc_agi_samples", return_value=samples), \
            mock.patch.object(A, "run_eval_suite", side_effect=fake_run):
        res = A.run_arc_agi("m", "http://x/v1")
    assert res["accuracy"] == 100.0 and res["leaderboard_comparable"] is True
    assert "leaderboard_comparable_reason" not in res


def test_arc_agi_counts_grids_upstream_would_coerce_or_retry():
    from gbench.runners.eval_suites import arc_agi as A

    def trace(i, text, idx=0):
        return {"sample_idx": idx, "source_sample_idx": i, "extra_payload": {"task_id": f"t{i}"},
                "response_text": text, "is_correct": A._eval_arc_agi(text, [[1, 2]]),
                "status": "OK"}
    # Upstream's Attempt model coerces [[1.0, 2]] to [[1, 2]] (correct); \boxed{[]} is a
    # valid, wrong answer; "no grid" cannot build an Attempt and is a null attempt.
    assert A._parse_grid("[[1.0, 2]]") == [[1, 2]] and A._parse_grid("\\boxed{[]}") == []
    rep = A._score_per_task([trace(0, "[[1.0, 2]]"), trace(1, "\\boxed{[]}"),
                             trace(2, "no grid"), trace(3, "[[1, 2]]")])
    assert rep["unparsed_attempts"] == 1 and rep["pairs_solved"] == 2

    samples = [([{"role": "user", "content": "p"}], [[1, 2]],
                {"category": "grid_transformation", "task_id": "t0", "pair_index": 0})]
    calls = []

    def fake_run(**kw):
        calls.append(kw)
        if len(calls) == 1:     # first tries: one attempt has no grid
            return {"attempts": {"attempts_per_sample": 2},
                    "sample_traces": [trace(0, "no grid", 0), trace(0, "[[1, 2]]", 1)]}
        return {"sample_traces": [{**trace(0, "[[1.0, 2]]", 0), "sample_idx": 0}]}
    with mock.patch.object(A, "_load_arc_agi_samples", return_value=samples), \
            mock.patch.object(A, "run_eval_suite", side_effect=fake_run):
        res = A.run_arc_agi("m", "http://x/v1")
    assert len(calls) == 2 and calls[1]["attempt_count"] == 1 and len(calls[1]["samples"]) == 1
    assert res["arc_agi_report"]["retried_attempts"] == 1
    assert res["arc_agi_report"]["recovered_by_retry"] == 1
    assert res["arc_agi_report"]["unparsed_attempts"] == 0
    assert res["leaderboard_comparable"] is True


def _run_section(suite):
    doc = (REPO / "docs" / "evals" / f"{suite}.md").read_text(encoding="utf-8")
    return doc.split("## Run", 1)[1]


@pytest.mark.parametrize("suite", ["arc_agi", "longbench_v2_code", "ifeval", "lab_bench", "cruxeval",
                                   "wmdp", "causalbench", "gsm8k"])
def test_every_doc_run_command_passes_the_mandatory_max_output_tokens(suite):
    blocks = _run_section(suite).split("```bash")[1:]
    assert blocks
    for block in blocks:
        cmd = block.split("```", 1)[0]
        assert f"--evals {suite}" in cmd and "--max-output-tokens" in cmd, cmd


def test_wmdp_doc_says_the_output_budget_does_not_affect_the_score():
    run = _run_section("wmdp")
    assert "required by the CLI" in run and "does not affect" in run


def test_culer_doc_quotes_budgets_the_cli_can_produce():
    doc = (REPO / "docs" / "evals" / "longbench_v2_code.md").read_text(encoding="utf-8")
    assert "114,624" not in doc and "98,240" not in doc
    assert "131,072 - 65,536 - 64 = 65,472" in doc and "120,000" in doc


def test_wmdp_thinking_request_is_recorded_as_not_applied(monkeypatch, caplog):
    import logging
    from gbench.runners.eval_suites import wmdp

    async def fake_score_all(base_url, model_name, enc, samples, concurrency,
                             temperature=0.0):
        return [{"loglikelihoods": [-1.0, -2.0, -3.0, -4.0], "pred": 0,
                 "is_correct": s["gold"] == 0, "status": "OK", "error": None} for s in samples]

    monkeypatch.setattr(wmdp, "_load_tokenizer", lambda name: _CharTok())
    monkeypatch.setattr(wmdp, "_score_all", fake_score_all)
    with _fake_load_dataset(_wmdp_rows()), caplog.at_level(logging.WARNING, logger=wmdp.__name__):
        res = wmdp.run_wmdp("m", "http://x/v1", enable_thinking=True)
    assert res["thinking"] is False and res["thinking_requested"] is True
    assert res["thinking_applied"] is False
    assert "not applied" in res["thinking_not_applied_reason"]
    assert any(r.levelno == logging.WARNING and "not applied" in r.getMessage()
               for r in caplog.records)
    # The loglikelihood score itself is still the canonical measurement.
    assert res["leaderboard_comparable"] is True
    with _fake_load_dataset(_wmdp_rows()):
        plain = wmdp.run_wmdp("m", "http://x/v1")
    assert plain["thinking"] is False and plain["thinking_requested"] is False
    assert "thinking_not_applied_reason" not in plain


def test_wmdp_claims_about_lm_eval_match_v042_for_gemma4():
    from gbench.runners.eval_suites import wmdp
    doc = (REPO / "docs" / "evals" / "wmdp.md").read_text(encoding="utf-8")
    for text in (doc, wmdp.__doc__):
        assert "the same request lm-eval" not in text
        assert "gemma4" in text and "max_tokens=0" in text and "logprobs=10" in text

    class GemmaTokenizerFast(_CharTok):
        bos_token_id = 2
    enc = wmdp._Encoder(GemmaTokenizerFast())
    assert enc.bos_prepended is False


def test_gsm8k_n_shot_above_8_is_an_error_and_n_shot_is_recorded():
    from gbench.runners.eval_suites import gsm8k as G
    with mock.patch.object(G, "_load_gsm8k_samples") as load, \
            pytest.raises(ValueError, match="exceeds the 8"):
        G.run_gsm8k("m", "http://x/v1", 1, eval_n_shot=12)
    load.assert_not_called()
    with pytest.raises(ValueError):
        G.fewshot_messages("q", 9)
    for requested in (None, 0, 5):
        with mock.patch.object(G, "_load_gsm8k_samples", return_value=[]) as load, \
                mock.patch.object(G, "run_eval_suite", return_value={"temperature": 0.0}):
            res = G.run_gsm8k("m", "http://x/v1", 1, eval_n_shot=requested)
        applied = 8 if requested is None else requested
        assert load.call_args.kwargs["n_shot"] == applied and res["n_shot"] == applied
        clause = f"{applied}-shot (gsm8k_cot is 8-shot)"
        # a full 8-shot greedy run is gsm8k_cot as published: comparable, no reason
        assert (clause in res.get("leaderboard_comparable_reason", "")) == (applied != 8)
        assert res["leaderboard_comparable"] is (applied == 8)


def test_lab_bench_letters_only_replies_are_parsed_as_upstream_and_not_flagged():
    from gbench.runners.eval_suites import lab_bench as L
    g = {"letter": "B", "letters": "ABCDEF", "refuse": "D"}
    # chembench prepare_mcq_answer's no-tag branch, then the first offered letter.
    assert L._answer_letter("B", "ABCDEF") == ("B", "letters_only")
    assert L._answer_letter("B, D", "ABCDEF") == ("B", "letters_only")
    assert L._answer_letter("A B", "ABCDEF") == ("A", "letters_only")
    assert L._answer_letter("H", "ABCDEF") == (None, "none")          # parsed, not offered
    # Untagged prose is not read at all (LAB-Bench scores it wrong).
    assert L._answer_letter("Final Answer: B", "ABCDEF") == (None, "none")
    assert L._answer_letter("(C).", "ABCDEF") == (None, "none")
    assert L._eval_lab_bench("B, D", g) is True
    traces = [{"gold_answer": g, "response_text": "B", "is_correct": True},
              {"gold_answer": g, "response_text": "Final Answer: B", "is_correct": False}]
    rep = L._lab_bench_report(traces)
    assert rep["no_letter"] == 1 and rep["letters_only_answers"] == 1
    sample = ([{"role": "user", "content": "p"}], g, {"category": "x"})
    with mock.patch.object(L, "_load_lab_bench_samples", return_value=[sample]), \
            mock.patch.object(L, "run_eval_suite", return_value={"sample_traces": traces}):
        res = L.run_lab_bench("m", "http://x/v1")
    reason = res["leaderboard_comparable_reason"]
    assert "no [ANSWER] tag" not in reason and "unanswered" not in reason


def test_cruxeval_runs_score_dict_golds_only():
    from gbench.runners.eval_suites import cruxeval as C
    rows = [{"code": "def f(x):\n    return x", "input": "1", "output": "1", "id": "s0"}]
    with _fake_load_dataset({"cruxeval-org/cruxeval": rows}):
        samples = C._load_cruxeval_samples() + C._load_cruxeval_samples(enable_thinking=True)
    assert all(isinstance(g, dict) and {"code", "input", "output", "mode"} <= set(g)
               for _, g, _ in samples)
    assert "literal-check helper only" in C._check.__doc__
