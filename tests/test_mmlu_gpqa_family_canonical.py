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

"""Canonical fidelity: mmlu, mmlu_pro, mmlu_redux, multilingual_mmlu, gpqa(_diamond).

Each suite now follows one upstream protocol: TIGER-Lab's CoT evaluation (mmlu_pro), the
Hendrycks 5-shot format (mmlu), ZeroEval on MMLU-Redux-2.0 (mmlu_redux) and simple-evals
(multilingual_mmlu, gpqa, gpqa_diamond). No dataset is downloaded: `datasets.load_dataset`
is patched with fixed rows.
"""

import os
from unittest import mock

import pytest

from gbench.runners.eval_suites import (gpqa, gpqa_diamond, mmlu, mmlu_pro, mmlu_redux,
                                        multilingual_mmlu)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fake_hf(rows_by_key):
    """Patch load_dataset; keys are (path, config or None, split). Records every call."""
    calls = []

    def _load(path, *args, **kwargs):
        config = args[0] if args else kwargs.get("name")
        calls.append({"path": path, "config": config, **kwargs})
        for key in ((path, config, kwargs.get("split")), (path, None, kwargs.get("split"))):
            if key in rows_by_key:
                rows = rows_by_key[key]
                if isinstance(rows, Exception):
                    raise rows
                return rows
        raise AssertionError(f"unexpected dataset request: {path} {config} {kwargs}")

    return mock.patch("datasets.load_dataset", side_effect=_load), calls


def _doc(suite):
    with open(os.path.join(_REPO, "docs", "evals", f"{suite}.md"), encoding="utf-8") as f:
        return f.read()


# --------------------------------------------------------------------------- #
# mmlu_pro (#18, #71, #86)
# --------------------------------------------------------------------------- #

_PRO_VAL = [
    {"question": f"Val q{i}?", "options": ["x", "y", "N/A"], "answer": "B", "answer_index": 1,
     "cot_content": f"A: Let's think step by step. Reason {i}. The answer is (B).",
     "category": "math"}
    for i in range(5)
]
_PRO_TEST = [{"question": "What is 2 + 2?", "options": ["3", "4", "5", "N/A"], "answer": "B",
              "answer_index": 1, "category": "math"}]


def test_mmlu_pro_extraction_follows_tiger_lab():
    ev = mmlu_pro._eval_mmlu_pro
    # Reproductions: the old start-of-text tier read "B" from "Based".
    assert ev("Based on the analysis above, the answer is (C).", "C") is True
    assert ev("Given the constraints, the answer is (E).", "E") is True
    assert ev("First, compute the sum. The answer is (H).", "H") is True
    assert ev("**The answer is (D)**", "D") is True
    # evaluate_from_apiX.py extract_answer: re.findall, the LAST "answer is" wins.
    assert ev("the answer is (A) ... no wait, the answer is (C)", "C") is True
    # Stage 2 then stage 3.
    assert ev("Reasoning.\nAnswer: G", "G") is True
    assert ev("I pick J", "J") is True
    # No start-of-text tier: a word's initial is never an answer.
    assert ev("Based on nothing at all", "B") is False
    assert ev("", "A") is False


def test_mmlu_pro_prompt_is_tiger_lab_cot_in_both_modes():
    patch, calls = _fake_hf({("TIGER-Lab/MMLU-Pro", None, "test"): _PRO_TEST,
                             ("TIGER-Lab/MMLU-Pro", None, "validation"): _PRO_VAL})
    with patch:
        think = mmlu_pro._load_mmlu_pro_samples(enable_thinking=True, n_shot=2)
        nothink = mmlu_pro._load_mmlu_pro_samples(enable_thinking=False, n_shot=2)
    assert all(c["revision"] == mmlu_pro.DATASET_REVISION for c in calls)
    prompt = think[0][0][0]["content"]
    assert prompt == nothink[0][0][0]["content"]
    assert prompt == (
        "The following are multiple choice questions (with answers) about math. Think step by"
        " step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n"
        "Question: Val q0?\nOptions: A. x\nB. y\nAnswer: Let's think step by step. Reason 0. "
        "The answer is (B).\n\n"
        "Question: Val q1?\nOptions: A. x\nB. y\nAnswer: Let's think step by step. Reason 1. "
        "The answer is (B).\n\n"
        "Question: What is 2 + 2?\nOptions: A. 3\nB. 4\nC. 5\nAnswer: Let's think step by step."
        "\n\n")
    assert think[0][1] == "B"


def test_mmlu_pro_exemplar_load_failure_is_a_hard_error_not_zero_shot():
    patch, _ = _fake_hf({("TIGER-Lab/MMLU-Pro", None, "test"): _PRO_TEST,
                         ("TIGER-Lab/MMLU-Pro", None, "validation"): OSError("hub down")})
    with patch, pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*mmlu_pro"):
        mmlu_pro._load_mmlu_pro_samples(n_shot=5)
    with pytest.raises(ValueError):
        mmlu_pro._few_shot_exemplars(6)


def test_mmlu_pro_defaults_to_five_shot_and_records_it():
    with mock.patch.object(mmlu_pro, "_load_mmlu_pro_samples", return_value=[]) as load:
        res = mmlu_pro.run_mmlu_pro("m", "http://x/v1", 1, max_output_tokens=100)
        assert load.call_args.kwargs["n_shot"] == 5 and res["n_shot"] == 5
        res0 = mmlu_pro.run_mmlu_pro("m", "http://x/v1", 1, eval_n_shot=0)
        assert load.call_args.kwargs["n_shot"] == 0 and res0["n_shot"] == 0


def test_mmlu_pro_docs_state_the_real_default():
    doc = _doc("mmlu_pro")
    assert "5-shot CoT by default" in doc and "Zero-shot by default" not in doc
    assert mmlu_pro.DATASET_REVISION in doc


# --------------------------------------------------------------------------- #
# mmlu (#53, #86)
# --------------------------------------------------------------------------- #

_MMLU_DEV = [{"question": f"Dev {i}?", "choices": ["w", "x", "y", "z"], "answer": i % 4,
              "subject": "abstract_algebra"} for i in range(5)]
_MMLU_TEST = [{"question": " Test q?", "choices": ["0", "1", "2", "3"], "answer": 2,
               "subject": "abstract_algebra"}]


def test_mmlu_prompt_is_the_hendrycks_format():
    patch, calls = _fake_hf({("cais/mmlu", "all", "test"): _MMLU_TEST,
                             ("cais/mmlu", "all", "dev"): _MMLU_DEV})
    with patch:
        samples = mmlu._load_mmlu_samples(n_shot=2)
        think = mmlu._load_mmlu_samples(n_shot=2, enable_thinking=True)
    assert all(c["revision"] == mmlu.DATASET_REVISION for c in calls)
    prompt = samples[0][0][0]["content"]
    assert prompt == think[0][0][0]["content"]
    # Shots and target share one format; the reference's double space and raw question kept.
    assert prompt == (
        "The following are multiple choice questions (with answers) about  abstract algebra.\n\n"
        "Dev 0?\nA. w\nB. x\nC. y\nD. z\nAnswer: A\n\n"
        "Dev 1?\nA. w\nB. x\nC. y\nD. z\nAnswer: B\n\n"
        " Test q?\nA. 0\nB. 1\nC. 2\nD. 3\nAnswer:")
    assert "Subject:" not in prompt and "Final Answer" not in prompt
    assert samples[0][1] == "C"


def test_mmlu_exemplar_load_failure_is_a_hard_error_not_zero_shot():
    patch, _ = _fake_hf({("cais/mmlu", "all", "test"): _MMLU_TEST,
                         ("cais/mmlu", "all", "dev"): OSError("hub down")})
    with patch, pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*mmlu"):
        mmlu._load_mmlu_samples(n_shot=5)


def test_mmlu_extracts_one_committed_letter():
    # The reference ranks the A-D log-probabilities; one letter per question, the first on a
    # tie (np.argmax). No text is parsed.
    assert mmlu._pick([-2.0, -0.5, -3.0, -0.7]) == 1
    assert mmlu._pick([-1.0, -2.0, -1.0, -3.0]) == 0
    assert not hasattr(mmlu, "_extract_mmlu_letter")


def _mmlu_items(n=1):
    return [{"subject": "abstract_algebra", "header": mmlu.gen_prompt_header("abstract_algebra"),
             "shots": [], "question": "Q?\nA. a\nB. b\nC. c\nD. d\nAnswer:", "gold": 2}] * n


class _CharTok:
    bos_token_id = None

    def encode(self, s):
        return [ord(c) for c in s]


def _run_mmlu(items, scores, **kwargs):
    async def fake_score_all(base_url, model_name, enc, samples, concurrency,
                             temperature=0.0):
        return scores(samples)
    with mock.patch.object(mmlu, "_load_mmlu_items", return_value=items), \
            mock.patch.object(mmlu, "_load_tokenizer", return_value=_CharTok()), \
            mock.patch.object(mmlu, "_score_all", side_effect=fake_score_all):
        return mmlu.run_mmlu("m", "http://x/v1", 1, **kwargs)


def _ok(samples):
    return [{"loglikelihoods": [-3.0, -2.0, -0.1, -4.0], "pred": 2, "is_correct": True,
             "status": "OK", "error": None} for _ in samples]


def test_mmlu_result_is_flagged_not_leaderboard_comparable():
    # Scored by log-likelihood like the reference, a full 5-shot run is now comparable.
    res = _run_mmlu(_mmlu_items(), _ok)
    assert res["leaderboard_comparable"] is True and "leaderboard_comparable_reason" not in res
    assert res["n_shot"] == 5 and res["scoring_mode"] == "loglikelihood"
    assert "leaderboard_comparable` is true for a full 5-shot run" in _doc("mmlu")


# --------------------------------------------------------------------------- #
# mmlu_redux (#19, #87)
# --------------------------------------------------------------------------- #

def _redux_row(err, answer=0, correct=None):
    return {"question": f"Q {err} {correct}", "choices": ["w", "x", "y", "z"], "answer": answer,
            "error_type": err, "correct_answer": correct, "source": "", "potential_reason": ""}


def _mmlu_dev(subjects):
    return [{"question": f"Dev {s} {i}?", "choices": ["p", "q", "r", f"s{i}"], "answer": i % 4,
             "subject": s} for s in subjects for i in range(5)]


def _redux_hf(rows_by_subject_or_all, subjects=None):
    subjects = subjects or mmlu_redux.MMLU_REDUX_SUBJECTS
    rows = ({("edinburgh-dawg/mmlu-redux-2.0", None, "test"): rows_by_subject_or_all}
            if isinstance(rows_by_subject_or_all, list) else dict(rows_by_subject_or_all))
    rows[("cais/mmlu", "all", "dev")] = _mmlu_dev(subjects)
    return _fake_hf(rows)


def test_mmlu_redux_loads_v2_all_57_subjects_pinned():
    assert len(mmlu_redux.MMLU_REDUX_SUBJECTS) == 57
    assert len(set(mmlu_redux.MMLU_REDUX_SUBJECTS)) == 57
    patch, calls = _redux_hf([_redux_row("ok", 1)])
    with patch:
        samples = mmlu_redux._load_mmlu_redux_samples()
    redux = [c for c in calls if c["path"] == "edinburgh-dawg/mmlu-redux-2.0"]
    assert len(redux) == 57 and len(samples) == 57
    assert {c["config"] for c in redux} == set(mmlu_redux.MMLU_REDUX_SUBJECTS)
    assert all(c["revision"] == mmlu_redux.DATASET_REVISION for c in redux)
    assert [c["revision"] for c in calls if c["path"] == "cais/mmlu"] == [
        mmlu_redux.MMLU_REVISION]


def test_mmlu_redux_keeps_only_rows_with_a_single_verified_key():
    # The authors score the "correct instances": error_type == "ok" only, MMLU key.
    rows = [
        _redux_row("ok", 1),                               # -> B
        _redux_row("wrong_groundtruth", 0, "C"),
        _redux_row("wrong_groundtruth", 0, "3"),
        _redux_row("wrong_groundtruth", 0, None),
        _redux_row("bad_question_clarity", 0),
        _redux_row("bad_options_clarity", 0),
        _redux_row("no_correct_answer", 0, "42"),
        _redux_row("multiple_correct_answers", 0, "A, B"),
        _redux_row("expert", 0, "B"),
        _redux_row("ok", 3),                               # -> D
    ]
    with mock.patch.object(mmlu_redux, "MMLU_REDUX_SUBJECTS", ["anatomy"]):
        patch, _ = _redux_hf({("edinburgh-dawg/mmlu-redux-2.0", "anatomy", "test"): rows},
                             ["anatomy"])
        with patch:
            samples = mmlu_redux._load_mmlu_redux_samples()
    assert [s[1] for s in samples] == ["B", "D"]


def test_mmlu_redux_prompt_is_zeroeval_mcqa():
    # HELM MMLU multiple_choice_joint with the instruction-tuned output format line.
    prompt = mmlu_redux.helm_prompt("college_biology", ["Question: d?\nA. a\nB. b\nC. c\nD. d"
                                                        "\nAnswer: B\n"],
                                    "What is 1+1?", ["1", "2", "3", "4"])
    assert prompt == (
        "Answer with only a single letter.\n\nThe following are multiple choice questions "
        "(with answers) about college biology.\n\nQuestion: d?\nA. a\nB. b\nC. c\nD. d\n"
        "Answer: B\n\nQuestion: What is 1+1?\nA. 1\nB. 2\nC. 3\nD. 4\nAnswer:")


def test_mmlu_redux_credits_one_extracted_answer_not_any_tier():
    em = mmlu_redux._exact_match
    choices = ["1", "2", "3", "4"]
    # Only a reply that is exactly one letter (after stripping) can score.
    assert em(" B\n", choices, 1) == (True, "B")
    assert em("B", choices, 0) == (False, "B")
    for reply in ("Answer: B", "**B**", "The answer is B", "", None, "b"):
        assert em(reply, choices, 1)[0] is False, reply
    # HELM compares option TEXT: two options with the gold's text both score.
    assert em("C", ["x", "y", "x", "z"], 0) == (True, "C")


def test_mmlu_redux_result_is_flagged_and_documented():
    doc = _doc("mmlu_redux")
    assert mmlu_redux.DATASET_REVISION in doc and "mmlu-redux-2.0" in doc
    assert "5,330" in doc and "HELM" in doc and "ZeroEval" not in doc


# --------------------------------------------------------------------------- #
# multilingual_mmlu (#55, #72)
# --------------------------------------------------------------------------- #

def test_multilingual_anchor_does_not_read_correcta_as_c():
    ex = multilingual_mmlu._extract_mc_letter
    for reply in ("La respuesta correcta es B.", "A resposta correta é D.",
                  "La risposta corretta è A.", "La réponse correcte est B."):
        assert ex(reply) != "C", reply
    assert ex("Respuesta: B") == "B"
    assert ex("Réponse : C") == "C"
    assert ex("答案：D") == "D"
    assert ex("الإجابة: ب") == "B"                 # Arabic letter normalized
    assert ex("**Answer: A**") == "A"             # markdown bold stripped
    assert ex("Answer: (B)") is None              # simple-evals does not read the parenthesis
    assert multilingual_mmlu._eval_multilingual_mmlu("Antwort: b", "B") is False


def test_multilingual_prompt_is_simple_evals_template_and_pinned():
    row = {"Question": "Quelle est la capitale?", "A": "Paris", "B": "Lyon", "C": "Nice",
           "D": "Metz", "Answer": "A", "Subject": "geography"}
    patch, calls = _fake_hf({("openai/MMMLU", None, "test"): [row]})
    with patch:
        samples = multilingual_mmlu._load_multilingual_mmlu_samples()
    assert len(calls) == 14
    assert {c["config"] for c in calls} == set(multilingual_mmlu.MMMLU_CONFIGS)
    assert all(c["revision"] == multilingual_mmlu.DATASET_REVISION for c in calls)
    # A no-think request carries simple-evals' chat-sampler system message before the prompt.
    assert samples[0][0][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert samples[0][0][-1]["content"] == (
        "Answer the following multiple choice question. The last line of your response should "
        "be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of "
        "ABCD. Think step by step before answering.\n\nQuelle est la capitale?\n\n"
        "A) Paris\nB) Lyon\nC) Nice\nD) Metz")
    assert samples[0][1] == "A"


def test_multilingual_docs_name_the_loaded_dataset():
    doc = _doc("multilingual_mmlu")
    assert "openai/MMMLU" in doc and "alexandrainst" not in doc
    assert multilingual_mmlu.DATASET_REVISION in doc
    for cfg in multilingual_mmlu.MMMLU_CONFIGS:
        assert cfg in doc


# --------------------------------------------------------------------------- #
# gpqa / gpqa_diamond (#45, #84)
# --------------------------------------------------------------------------- #

_GPQA_ROW = {"Question": "Which is right?", "Correct Answer": "right",
             "Incorrect Answer 1": "wrong1", "Incorrect Answer 2": "wrong2",
             "Incorrect Answer 3": "wrong3", "Subdomain": "Physics"}


@pytest.mark.parametrize("module,config", [(gpqa, "gpqa_main"), (gpqa_diamond, "gpqa_diamond")])
def test_gpqa_prompt_is_simple_evals_template(module, config):
    patch, calls = _fake_hf({("Idavidrein/gpqa", config, "train"): [_GPQA_ROW]})
    with patch:
        samples = module._load_gpqa_samples(False)
    assert calls[0]["revision"] == gpqa.DATASET_REVISION
    messages, gold, meta = samples[0]
    # A no-think request carries simple-evals' chat-sampler system message before the prompt.
    assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
    content = messages[-1]["content"]
    assert content.startswith(
        "Answer the following multiple choice question. The last line of your response should "
        "be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of "
        "ABCD. Think step by step before answering.\n\nWhich is right?\n\nA) ")
    assert "Final Answer" not in content and "Try to reason" not in content
    assert f"{gold}) right" in content
    # ...and the chat sampler's max_tokens=2048, declared once as the suite's official cap
    # (gpqa.SIMPLE_EVALS_SAMPLING, applied by base's token-limit policy), not per sample.
    assert meta == {"category": "Physics"}
    assert gpqa.SIMPLE_EVALS_SAMPLING.max_tokens == 2048


@pytest.mark.parametrize("ev", [gpqa._eval_gpqa, gpqa_diamond._eval_gpqa])
def test_gpqa_letter_extraction_is_case_sensitive(ev):
    # Reproductions: a lower-case word start was upper-cased into a letter.
    assert ev("Final answer: after weighing both mechanisms, the product is (C).", "A") is False
    assert ev("The answer: based on the NMR data it is option (D).", "B") is False
    assert ev("Reasoning...\nAnswer: C", "C") is True
    assert ev("Reasoning...\nAnswer: $C$", "C") is True


def test_gpqa_docs_do_not_claim_boxed_parsing():
    for suite in ("gpqa", "gpqa_diamond"):
        doc = _doc(suite)
        assert "(`\\boxed{}`, then an explicit anchor" not in doc
        assert "simple-evals" in doc and gpqa.DATASET_REVISION in doc


# =========================================================================== #
# Verbatim upstream functions, explicit comparability and its docs
# =========================================================================== #
# sha256 of each function's source upstream: hendrycks/test evaluate.py @ 4450500f.
_UPSTREAM_SHA256 = {
    (mmlu, "format_subject"): "63ec725f9683d57c34787e0c89c78086a60f5ded67756d4bb62abaf9bd907314",
}


def test_ported_upstream_functions_are_verbatim():
    import hashlib
    import inspect
    for (module, name), digest in _UPSTREAM_SHA256.items():
        src = inspect.getsource(getattr(module, name)).rstrip("\n")
        assert hashlib.sha256(src.encode()).hexdigest() == digest, name
    assert 'l = subject.split("_")' in inspect.getsource(mmlu.format_subject)
    assert mmlu.gen_prompt_header("abstract_algebra") == (
        "The following are multiple choice questions (with answers) about  abstract algebra.\n\n")


def _pro_run(outcome, **kwargs):
    sample = ([{"role": "user", "content": "p"}], "A", {"category": "math", "row": 0})

    async def fake_harness(*args, **kw):
        return [{"category": "math", "is_correct": outcome.get("predicted") == "A",
                 "predicted": None, "response_text": None, "status": "OK", **outcome}]
    with mock.patch.object(mmlu_pro, "_load_mmlu_pro_samples", return_value=[sample]), \
            mock.patch.object(mmlu_pro, "_run_harness", side_effect=fake_harness):
        return mmlu_pro.run_mmlu_pro("m", "http://x/v1", 1, **kwargs)


def test_mmlu_pro_states_comparability_explicitly(monkeypatch):
    ok = {"response_text": "The answer is (A).", "predicted": "A"}
    # gbench's no-think 0.0 is not upstream's 0.1, and the reason says so
    res = _pro_run(ok)
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"] == (
        "sampled at temperature 0.0 (default; evaluate_from_apiX.py sends temperature 0.1)")
    # at upstream's 0.1, set explicitly, the run is comparable
    monkeypatch.setenv("GBENCH_MMLU_PRO_TEMPERATURE", "0.1")
    res = _pro_run(ok)
    assert res["leaderboard_comparable"] is True and "leaderboard_comparable_reason" not in res
    # evaluate_from_apiX.py scores a letterless reply 0 (no random guess): still comparable.
    res = _pro_run({"response_text": "", "predicted": None})
    assert res["leaderboard_comparable"] is True and res["no_answer_replies"] == 1
    for kwargs, outcome, env, clause in (
            ({"limit": 20}, ok, {}, "--eval-limit subset (1 of 12,032 questions)"),
            ({"eval_n_shot": 0}, ok, {}, "0-shot (evaluate_from_apiX.py puts all 5"),
            ({}, ok, {"GBENCH_MMLU_PRO_TEMPERATURE": "1.0"}, "sampled at temperature 1.0"),
            ({}, {"status": "FAILED", "predicted": None}, {},
             "1 question(s) got no reply from the served endpoint after 10 tries")):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        res = _pro_run(outcome, **kwargs)
        for k in env:
            monkeypatch.delenv(k)
        assert res["leaderboard_comparable"] is False
        assert clause in res["leaderboard_comparable_reason"], clause


@pytest.mark.parametrize("module,run,total", [
    (gpqa, "run_gpqa", 448), (gpqa_diamond, "run_gpqa_diamond", 198),
    (multilingual_mmlu, "run_multilingual_mmlu", 14 * 14042)])
def test_simple_evals_suites_state_comparability_explicitly(module, run, total):
    # gpqa and gpqa_diamond build their (question, repeat) copies in gpqa.py.
    target = gpqa if module is not multilingual_mmlu else multilingual_mmlu
    for limit in (None, 3):
        if target is gpqa:
            items = [{"question_idx": q, "repeat_idx": r, "permutation": [0, 1, 2, 3],
                      "sample": ("m", "A", {})} for r in range(10) for q in range(limit or 1)]
            loader = mock.patch.object(gpqa, "_load_gpqa_items", return_value=items)
        else:
            loader = mock.patch.object(module, "_load_multilingual_mmlu_samples",
                                       return_value=[("m", "A", {})] * (limit or 1))
        with loader, mock.patch.object(target, "run_eval_suite",
                                       return_value={"temperature": 1.0}):
            res = getattr(module, run)("m", "http://x/v1", 1, limit=limit)
        if limit:
            assert res["leaderboard_comparable"] is False
            assert res["leaderboard_comparable_reason"] == (
                f"--eval-limit subset (3 of {total:,} questions)")
        else:
            # simple-evals fixes no temperature, so T=1.0 does not make it non-comparable.
            assert res["leaderboard_comparable"] is True
            assert "leaderboard_comparable_reason" not in res


def test_mmlu_and_mmlu_redux_reasons_name_subsets_and_shot_count():
    res = _run_mmlu(_mmlu_items(), _ok, limit=1, eval_n_shot=0)
    reason = res["leaderboard_comparable_reason"]
    assert res["leaderboard_comparable"] is False
    assert "0-shot (the reference is 5-shot)" in reason
    assert "--eval-limit subset (1 of 14,042 questions)" in reason
    item = {"subject": "anatomy", "choices": ["a", "b", "c", "d"], "answer": 1, "prompt": "p"}
    reply = mock.Mock(text="B", finish_reason="length", error=None)

    async def fake_generate(*args, **kwargs):
        return [reply]
    with mock.patch.object(mmlu_redux, "_load_mmlu_redux_items", return_value=[item]), \
            mock.patch.object(mmlu_redux, "_generate_all", side_effect=fake_generate):
        res = mmlu_redux.run_mmlu_redux("m", "http://x/v1", 1, limit=1)
    assert res["leaderboard_comparable"] is False and res["accuracy"] == 100.0
    assert "--eval-limit subset (1 of the 5,330 scored rows)" in res["leaderboard_comparable_reason"]


def test_docs_describe_the_comparability_flag():
    assert "leaderboard_comparable` is true when the run is upstream's protocol" in _doc("mmlu_pro")
    for suite in ("gpqa", "gpqa_diamond", "multilingual_mmlu"):
        doc = _doc(suite)
        assert "carries no `leaderboard_comparable` flag" not in doc
        assert "leaderboard_comparable` is true for a full run" in doc
