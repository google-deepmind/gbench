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

"""Regression tests for the built-in scoring of many suites.

Each test encodes a scoring defect with the concrete input that showed it, so the fake-pass /
structural-zero behaviours cannot silently return.
"""

import json

import pytest

from gbench.runners.eval_suites.fc_common import (
    call_matches,
    parse_gold_call,
    parse_tool_calls,
)
from gbench.runners.eval_suites.hmmt import _eval_hmmt
from gbench.runners.eval_suites.wildbench import parse_wb_score
from gbench.runners.eval_suites.mmmu_pro import _eval_mmmu_pro
from gbench.runners.eval_suites.scicode import _process_one_problem


# --------------------------------------------------------------------------- #
# function calling: structural (name + args), not substring
# --------------------------------------------------------------------------- #
def test_fc_parses_the_four_call_renderings():
    assert parse_tool_calls('{"name": "f", "arguments": {"a": 1}}')[0] == ("f", {"a": 1})
    assert parse_tool_calls('Action: f\nAction Input: {"a": 1}')[0] == ("f", {"a": 1})
    assert parse_tool_calls('f(a=1)')[0] == ("f", {"a": 1})
    # unquoted values (API-Bank golds) must still parse
    name, args = parse_gold_call("API-Request: [f(a=abc123, b=2023-10-10 11:00:00)]")
    assert name == "f" and args["a"] == "abc123"


# NOTE: api_bank's structural `_eval_api_bank` was replaced by canonical EXECUTION-based
# scoring (execute the call against the API-Bank backend + check_api_call_correctness);
# see tests/test_api_bank_canonical.py.


# NOTE: t_eval's boolean `_eval_t_eval` was replaced by the canonical 6-dimension port;
# instruct scoring (bare-echo rejected, proper call accepted) is covered in
# tests/test_t_eval_canonical.py.


def test_fc_common_parses_toolllama_action_gold():
    # toolbench no longer uses an offline first-action match (it now delegates to StableToolBench +
    # ToolEval); this just locks the shared fc_common ToolLLaMA-action parser it used to rely on.
    gold = 'Thought: look it up\nAction: get_weather\nAction Input: {"city": "Paris"}'
    assert parse_gold_call(gold) == ("get_weather", {"city": "paris"})


# NOTE: complexfuncbench's single-turn `_eval_complexfuncbench` structural match was replaced
# by the canonical multi-step golden-driven loop + ComplexEval (Success Rate / Call Accuracy /
# Completeness / Correctness, Gemini judge + bge alignment); see tests/test_complexfuncbench.py.


def test_call_matches_requires_gold_args_but_tolerates_extras():
    gold = ("f", {"a": 1})
    assert call_matches(gold, [("f", {"a": 1, "b": 2})]) is True     # extra arg ok
    assert call_matches(gold, [("f", {"b": 2})]) is False            # missing gold arg
    assert call_matches(gold, [("f", {"a": 9})]) is False            # contradicting value


# --------------------------------------------------------------------------- #
# answer extraction: no bare containment
# --------------------------------------------------------------------------- #
def test_hmmt_rejects_incidental_mention_of_the_answer():
    assert _eval_hmmt("We test 5 cases, therefore the answer is 12.", "5") is False
    assert _eval_hmmt(r"Hence \boxed{5}.", "5") is True
    assert _eval_hmmt("Final Answer: 5", "5") is True


def test_mmmu_pro_rejects_single_letter_substring():
    assert _eval_mmmu_pro("A careful analysis shows option C is right.", "A") is False
    assert _eval_mmmu_pro("Answer: A", "A") is True


# --------------------------------------------------------------------------- #
# fail-closed: no length-based passes
# --------------------------------------------------------------------------- #
def test_lmsys_without_checklist_no_longer_auto_passes():
    # The keyword heuristic is gone: only a WB-Score judge verdict carrying a score counts, so
    # a long reply without one yields no score (it is excluded as a judge failure, not passed).
    assert parse_wb_score("x" * 500) is None


def test_scicode_without_tests_no_longer_auto_passes():
    # A sub-step with no test cases must NOT score correct (no auto-pass), and must never
    # reach the h5/sandbox path. Drive one step through the real per-problem loop with a
    # stub client, so no endpoint or h5 is needed.
    class _Msg:
        content = "```python\ndef f():\n    return 1\n```"
        reasoning_content = "think step by step"

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _Usage:
        completion_tokens = 42

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()

    class _Client:
        class chat:
            class completions:
                @staticmethod
                def create(**_):
                    return _Resp()

    prob = {
        "problem_id": "1", "problem_name": "x", "required_dependencies": "",
        "sub_steps": [{
            "step_number": "1.1", "step_description_prompt": "d", "step_background": "",
            "function_header": "def f():", "return_line": "    return 1", "test_cases": [],
        }],
    }
    res = _process_one_problem(
        _Client(), "m", prob, temperature=0.0, max_tokens=64, thinking=False,
        with_background=False, h5="/nonexistent/test_data.h5", eval_timeout=5)
    step = res["steps"][0]
    assert step["special"] is False
    assert step["correct"] is False
    # thinking content AND length are captured on the trace (not discarded)
    assert step["reasoning"] == "think step by step"
    assert step["reasoning_chars"] == len("think step by step")
    assert step["finish_reason"] == "stop"
    assert step["completion_tokens"] == 42


def test_cimemories_refuses_to_score_without_a_judge():
    # The labeler and the REVEAL judge both need the Gemini key, so the runner refuses to score
    # without one.
    import os, pytest
    from gbench.runners.eval_suites import cimemories as _cim
    # No GEMINI judge -> HARD-ERROR (no-skip policy: a missing judge key is missing infra), not a
    # silent skip and not an offline heuristic downgrade.
    _k = os.environ.pop("GEMINI_API_KEY", None)
    _ks = os.environ.pop("GEMINI_API_KEYS", None)
    try:
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY|docs/evals"):
            _cim.run_cimemories("m", "http://x", limit=2)
    finally:
        if _k is not None:
            os.environ["GEMINI_API_KEY"] = _k
        if _ks is not None:
            os.environ["GEMINI_API_KEYS"] = _ks


# --------------------------------------------------------------------------- #
# Leniency: no containment fast-pass, answer extraction, FC calls compared as ASTs
# --------------------------------------------------------------------------- #
def test_bfcl_possible_answer_is_structural():
    """BFCL gold is {func: {param: [accepted]}}; a name mentioned in prose is not a call."""
    from gbench.runners.eval_suites.bfcl import _eval_bfcl
    gold = [{"get_weather": {"city": ["Paris"], "unit": [""]}}]   # unit optional
    call = [{"type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]
    assert _eval_bfcl("", gold, tool_calls=call) is True
    assert _eval_bfcl("I could use get_weather for Paris", gold) is False
    # BFCL's FC rule: reply text is never a call, JSON-shaped or not
    assert _eval_bfcl('{"name":"get_weather","arguments":{"city":"Paris"}}', gold) is False


def test_seal_tools_and_nestful_require_the_call_not_the_name():
    from gbench.runners.eval_suites.nestful import _eval_nestful
    from gbench.runners.eval_suites.seal_tools import _eval_seal_tools
    # Canonical formats: Seal-Tools reads a `[{"api": ..., "parameters": ..., "responses": ...}]`
    # list, NESTFUL json.loads the whole reply as a list of calls.
    seal_gold = "[{'api': 'analyzeEvidence', 'parameters': {'id': 'x'}, 'responses': ['API_call_0']}]"
    assert _eval_seal_tools("analyzeEvidence might help here", seal_gold) is False
    assert _eval_seal_tools('[{"api": "analyzeEvidence", "parameters": {"id": "x"}, '
                            '"responses": ["API_call_0"]}]', seal_gold) is True
    nest_gold = '[{"name":"f","arguments":{"a":1}}]'
    assert _eval_nestful('[{"name":"f","arguments":{"a":1}}]', nest_gold) is True
    assert _eval_nestful("the answer is 1", nest_gold) is False


def test_no_bare_containment_before_the_judge():
    """HLE must not credit a gold that merely appears inside the reasoning.

    There is no deterministic pre-pass any more: every reply goes to the official HLE judge,
    and only its `correct: yes|no` decides.

    (livebench now delegates scoring to LiveBench's own ground-truth judges in-container;
    see tests/test_livebench_canonical.py.)"""
    from gbench.runners.eval_suites import humanitys_last_exam as hle
    assert not hasattr(hle, "_eval_humanitys_last_exam")
    assert hle.parse_judge_reply("the value 17 shows up but the answer is 3") is None
    assert hle.parse_judge_reply('{"extracted_final_answer": "17", "reasoning": "match", '
                                 '"correct": "yes", "confidence": 90, "strict": true}'
                                 )["correct"] == "yes"


def test_mmlu_uses_last_standalone_letter_not_first():
    """`^([ABCD])` matched the leading 'A' of an ordinary sentence. MMLU no longer reads a
    letter out of text at all: it ranks the A-D log-probabilities (hendrycks/test), so an
    article can never be taken for an answer, and a tie goes to the first letter."""
    from gbench.runners.eval_suites import mmlu
    assert not hasattr(mmlu, "_eval_mmlu") and not hasattr(mmlu, "_extract_mmlu_letter")
    assert mmlu._pick([-3.0, -1.0, -2.0, -5.0]) == 1
    assert mmlu._pick([-1.0, -1.0, -2.0, -5.0]) == 0


# --------------------------------------------------------------------------- #
# An empty harness / failed dataset load is an error, never "success" 0%
# --------------------------------------------------------------------------- #
def test_amc_aime_refuses_to_report_over_zero_samples():
    from unittest.mock import patch
    import pytest as _pytest
    import gbench.runners.eval_suites.amc_aime as N
    with patch("datasets.load_dataset", side_effect=RuntimeError("offline")):
        with _pytest.raises(RuntimeError, match="neither the AMC nor the AIME"):
            N._load_amc_aime_samples()


def test_terminal_bench_zero_trials_is_not_success():
    """A clean CLI exit that produced no trials measured nothing: it hard-errors (no-skip), never a
    fabricated 0.0 or a success."""
    import inspect
    import gbench.runners.eval_suites.terminal_bench as T
    src = inspect.getsource(T)
    # the guard must not require a non-zero exit code (zero trials alone is a harness failure)
    assert "if proc.returncode != 0 and total == 0:" not in src
    # zero measured trials -> hard-error via infra_required, keyed on what Harbor actually produced
    assert "if measured_total == 0:" in src
    assert "infra_required(" in src


def test_harness_failures_downgrade_status():
    """bigcodebench / swebench mark a missing harness report as error, not 0%."""
    import inspect
    import gbench.runners.eval_suites.bigcodebench as B
    import gbench.runners.eval_suites.swebench_common as S
    for mod, key in ((B, "bigcodebench_report"), (S, "swebench_report")):
        src = inspect.getsource(mod)
        assert key in src and "'error'" in src or '"error"' in src
        assert "status" in src


# --------------------------------------------------------------------------- #
# Canonical VQA/OCR metrics (were all bidirectional substring tests)
# --------------------------------------------------------------------------- #
def test_docvqa_uses_anls_not_substring():
    from gbench.runners.eval_suites.docvqa import _eval_docvqa
    assert _eval_docvqa("Paris", ["Paris"]) is True
    assert _eval_docvqa("Pariss", ["Paris"]) is True          # ANLS tolerates OCR noise
    assert _eval_docvqa("London", ["Paris"]) is False
    # a verbose answer that merely mentions the gold no longer passes
    assert _eval_docvqa("the document references Paris among many other cities", ["Paris"]) is False
    # lmms-eval's anls reads the reply as returned: no answer extraction
    assert _eval_docvqa("reasoning...\nAnswer: Paris", ["Paris"]) is False


def test_textvqa_uses_annotator_agreement():
    from gbench.runners.eval_suites.textvqa import _eval_textvqa
    golds = ["stop"] * 4 + ["stop sign"] * 6
    assert _eval_textvqa("stop", golds) is True
    assert _eval_textvqa("yield", golds) is False


def test_chartqa_relaxed_accuracy():
    from gbench.runners.eval_suites.chartqa import _eval_chartqa
    assert _eval_chartqa("103", ["100"]) is True              # within 5%
    assert _eval_chartqa("130", ["100"]) is False             # outside 5%
    assert _eval_chartqa("the value is 1000 not 100", ["100"]) is False   # no substring credit


def test_vqa_normalization_and_helpers():
    from gbench.runners.eval_suites.vqa_common import (
        anls_score, extract_short_answer, normalize_answer, vqa_accuracy)
    assert normalize_answer("The  Two Dogs.") == "2 dogs"
    assert extract_short_answer("blah\nFinal Answer: 42") == "42"
    assert anls_score("paris", ["Paris"]) == 1.0
    assert vqa_accuracy("cat", ["cat", "cat", "cat", "dog"]) == 1.0


# --------------------------------------------------------------------------- #
# Answer extraction (read the ANSWER, not the reasoning)
# --------------------------------------------------------------------------- #
def test_causalbench_reads_replies_as_corr2cause_does():
    from gbench.runners.eval_suites.causalbench import _eval_causalbench as C
    # rel_normalize: the whole reply, then its first two words, then its first; else invalid.
    assert C("No, this is not valid", "entailment") is False
    assert C("No, this is not valid", "contradiction") is True
    assert C("Yes, because A causes B.", "entailment") is True
    # a verdict that does not open the reply is not read, as upstream reads it
    assert C("Reasoning... Answer: Yes", "entailment") is False
    assert C("nothing conclusive", "entailment") is False


def test_wmdp_requires_the_stated_letter():
    # WMDP is now scored by zero-shot loglikelihood (lm-eval v0.4.2), not by reading a
    # letter out of generated text: the answer is the choice whose continuation scores highest,
    # summed over exactly the continuation tokens of the echoed prompt.
    from gbench.runners.eval_suites.wmdp import _continuation_logprobs as L
    choice = {"logprobs": {"token_logprobs": [None, -1.0, -2.0, -0.5, -9.0]}}
    assert L(choice, 3, 1) == -0.5
    assert L(choice, 2, 2) == -2.5


def test_extraction_helpers():
    from gbench.runners.eval_suites.extraction_common import (
        binary_verdict, final_number, last_mc_letter)
    assert last_mc_letter("A careful look shows D") == "D"      # last, not first
    assert last_mc_letter(r"so \boxed{C}") == "C"
    assert final_number("we tried 42 but the answer is 7") == "7"
    assert final_number(r"\boxed{13}") == "13"
    assert binary_verdict("it is not correct") is False
    assert binary_verdict("the evidence is inconclusive") is None   # no verdict word at all


def test_anchored_span_prefers_last_textual_anchor_across_patterns():
    # Regression: the weakest phrasing ("... the answer is X") is LAST in _ANCHORS, and the
    # old loop overwrote per-pattern so it always won -- even when a stronger "Final Answer:"
    # appeared LATER in the text. Precedence must be by textual position, not pattern order.
    from gbench.runners.eval_suites.extraction_common import (
        anchored_span, binary_verdict, final_number, last_mc_letter)
    assert final_number("the answer is 42. Final Answer: 7") == "7"
    assert last_mc_letter("ANSWER IS A. FINAL ANSWER: C", "ABCDE") == "C"
    # ...but a genuinely later "answer is" (the real conclusion) still wins.
    assert final_number("Final Answer: 7, wait, the answer is 42") == "42"
    # Tie inside "Final Answer:" (which also contains "Answer:") stays stable.
    assert anchored_span("Final Answer: 7").strip() == "7"
    assert binary_verdict("this is not valid. Final Answer: yes") is True


def test_hmmt_grades_by_symbolic_equivalence_not_string_match():
    # hmmt string/single-number matcher scored 1/2 != 0.5 and radical-fraction
    # rewrites as wrong. It now grades with math_verify (+ sympy fallback), and the suite
    # hard-errors if the precise backend is missing rather than downgrading silently.
    import pytest
    pytest.importorskip("math_verify")          # declared (evals extra): CI's skip gate fails it
    from gbench.runners.eval_suites import math_equiv
    from gbench.runners.eval_suites.hmmt import _eval_hmmt, _ANSWER_INSTRUCTION
    assert "\\boxed{}" in _ANSWER_INSTRUCTION            # canonical answer-format instruction
    assert math_equiv.available(), "math_verify imports but math_equiv did not load it"
    assert _eval_hmmt(r"so the answer is \boxed{\frac{1}{2}}", "0.5") is True
    assert _eval_hmmt(r"\boxed{\frac{\sqrt{2}}{2}}", r"\frac{1}{\sqrt{2}}") is True
    assert _eval_hmmt(r"\boxed{7}", "8") is False
    assert _eval_hmmt("", "5") is False


def test_math_backend_missing_hard_errors_not_silent():
    # No-skip policy: require_backend returns None when the backend is present and HARD-ERRORS
    # (infra_required) when it is absent - never a downgraded score, never a skip row.
    import pytest
    from gbench.runners.eval_suites import math_equiv
    if math_equiv.available():
        assert math_equiv.require_backend("hmmt") is None
    saved = math_equiv._mv_parse
    try:
        math_equiv._mv_parse = None
        assert math_equiv.available() is False
        with pytest.raises(RuntimeError, match="docs/evals/hmmt.md"):
            math_equiv.require_backend("hmmt")
    finally:
        math_equiv._mv_parse = saved


def test_finalize_mean_metric_counts_failures_as_zero_not_dropped():
    # mrcr: a failed/empty response is a 0, not a sample to drop from the mean -
    # dropping the hard items silently inflated the score. JUDGE_OUTAGE stays excluded.
    from gbench.runners.eval_suites.metrics import finalize_mean_metric
    traces = [
        {"response_text": "x", "gold_answer": "x"},
        {"response_text": None, "gold_answer": "y"},              # failure -> 0.0
        {"response_text": "o", "judge_grade": "JUDGE_OUTAGE"},    # outage -> excluded
    ]
    res = {"accuracy": 99.0, "sample_traces": [dict(t) for t in traces]}
    finalize_mean_metric(res, metric_name="m", scorer=lambda t: 1.0, failure_score=0.0)
    assert res["scored_samples"] == 2 and res["accuracy"] == 50.0   # (1.0 + 0.0) / 2
    res2 = {"accuracy": 99.0, "sample_traces": [dict(t) for t in traces]}
    finalize_mean_metric(res2, metric_name="m", scorer=lambda t: 1.0)   # default: drop failure
    assert res2["scored_samples"] == 1 and res2["accuracy"] == 100.0


def test_custom_jsonl_keeps_falsy_gold(tmp_path):
    # an `or` chain dropped a legitimately falsy gold (0/False/"") to None -> the row
    # became unscorable. First-present-key wins now.
    from gbench.runners.eval_suites.custom_jsonl import _load_custom_jsonl_samples
    p = tmp_path / "d.jsonl"
    p.write_text(
        '{"prompt":"q1","gold":0,"eval_type":"numeric"}\n'
        '{"prompt":"q2","gold":false}\n'
        '{"prompt":"q3","answer":""}\n'
    )
    golds = [g["gold"] for _, g, _ in _load_custom_jsonl_samples(p)]
    assert golds == [0, False, ""]


def test_cyberseceval_regex_path_detects_insecure(monkeypatch):
    # Delegation to Meta's ICD: every program, c/cpp included (the dataset's analyzer=weggli
    # rows, which the current ICD scores with regex + semgrep), goes to the ICD in its image with
    # its language, and the ICD's issues alone decide the verdict.
    import asyncio, json
    from gbench.runners.eval_suites import cyberseceval as C
    sent = []

    def fake_scan(items):
        sent.extend(items)
        flagged = ("strcpy(", "auto_ptr")
        return ({i["id"]: {"id": i["id"], "attempts": 1, "issues": (
            [{"analyzer": "regex", "pattern_id": "bugprone-strcpy"}]
            if any(f in i["code"] for f in flagged) else [])} for i in items}, None)
    monkeypatch.setattr(C, "_scan_in_image", fake_scan)
    gold_c = json.dumps({"language": "c", "cwe": "CWE-120"})
    gold_cpp = json.dumps({"language": "cpp", "cwe": "x"})
    traces = [
        {"gold_answer": gold_c, "response_text":
            "```c\n#include <string.h>\nvoid f(char*s){char b[8];strcpy(b,s);}\n```"},
        {"gold_answer": gold_c, "response_text":
            "```c\n#include <stdio.h>\nint add(int a,int b){return a+b;}\n```"},
        {"gold_answer": gold_cpp, "response_text":
            "```cpp\n#include <memory>\nvoid f(){std::auto_ptr<int> p(new int);}\n```"},
    ]
    asyncio.run(C._async_score_cyberseceval(traces))
    assert [t["is_correct"] for t in traces] == [False, True, False]
    assert [i["language"] for i in sent] == ["c", "c", "cpp"]
    assert traces[0]["icd_findings"][0]["rule_id"] == "bugprone-strcpy"


def test_multilingual_mmlu_extracts_non_latin_scripts():
    # Latin-only anchors + a \b([A-D])\b fallback under-scored non-Latin languages
    # (no word boundary between a CJK/Arabic char and the Latin letter).
    # Now simple-evals' localized anchors verbatim: anchor word, colon, letter.
    from gbench.runners.eval_suites.multilingual_mmlu import _eval_multilingual_mmlu as ev
    assert ev("答案：C", "C") is True and ev("答案：C", "B") is False     # Chinese
    assert ev("الإجابة: B", "B") is True                                # Arabic
    assert ev("정답: A", "A") is True                                   # Korean
    assert ev("Answer: D", "D") is True                                # Latin still works


def test_bundled_detection_scores_coco_map_not_any_box():
    # metric is COCO mAP (per-class AP, IoU matching, confidence ranking), not a
    # per-image "any one box right" hit rate.
    from gbench.runners.eval_suites.bundled_detection import _compute_map
    gold = [{"label": "cat", "box_2d": [100, 100, 200, 200]},
            {"label": "dog", "box_2d": [300, 300, 400, 400]}]
    perfect = ('[{"label":"cat","box_2d":[100,100,200,200],"confidence":0.9},'
               '{"label":"dog","box_2d":[300,300,400,400],"confidence":0.8}]')
    assert _compute_map([{"gold_answer": gold, "response_text": perfect}])["map_50"] > 0.99
    # right box, WRONG class -> not a true positive
    wrong_cls = '[{"label":"bird","box_2d":[100,100,200,200],"confidence":0.9}]'
    assert _compute_map([{"gold_answer": gold, "response_text": wrong_cls}])["map_50"] == 0.0


def test_omnidocbench_prompt_and_order_helpers_import():
    # reading-order + content-only block selection (dep-free part).
    from gbench.runners.eval_suites.omnidocbench import _IGNORED_CATS, normalized_edit_distance
    assert "abandon" in _IGNORED_CATS and "page_number" in _IGNORED_CATS
    assert normalized_edit_distance("hello world", "hello world") == 0.0
    assert normalized_edit_distance("", "abc") == 1.0


def test_omnidocbench_per_modality_composite():
    # per-modality composite (TEDS tables + formula/text edit distance), not one
    # global edit distance. The TEDS backend is the omnidocbench extra (apted, lxml, Distance).
    import pytest
    for module in ("apted", "lxml", "distance"):    # declared: CI's skip gate fails a skip here
        pytest.importorskip(module)
    from gbench.runners.eval_suites.omnidoc_metrics import (
        composite_available, table_teds, formula_edit_distance, extract_pred_streams, paired_mean)
    tab = "<table><tr><td>a</td><td>b</td></tr></table>"
    assert formula_edit_distance("$$x^2$$", "x^2") == 0.0
    tables, formulas, text = extract_pred_streams("Intro.\n\n" + tab + "\n\n$$x^2$$")
    assert len(tables) == 1 and formulas == ["x^2"] and "Intro." in text
    assert composite_available(), "the TEDS backend imports but omnidoc_metrics did not load it"
    assert table_teds(tab, tab) == 1.0
    assert table_teds("", tab) == 0.0                       # missing table -> full miss
    assert paired_mean([], [tab], table_teds, 0.0) == 0.0   # unmatched gold table -> 0


def test_bfcl_extracts_user_question_not_system_message():
    # live rows whose turn 0 is [system, user] fed the model the SYSTEM prompt.
    # BFCL's FC mode sends the whole turn verbatim, so the user question AND the dataset's own
    # system message both reach the model.
    from gbench.runners.eval_suites.bfcl import bfcl_request
    q = [[{"role": "system", "content": "You are a tool assistant."},
          {"role": "user", "content": "Book a flight to Paris."}]]
    fn = [{"name": "book", "description": "Book.",
           "parameters": {"type": "dict", "properties": {}, "required": []}}]
    messages, _ = bfcl_request({"category": "live_simple", "question": q, "function": fn})
    assert messages == q[0]


def test_bfcl_live_is_structural_not_substring():
    # substring keyword matching credited prose that merely named the tool/value.
    from gbench.runners.eval_suites.bfcl_live import _eval_bfcl_live
    gold = [{"get_weather": {"city": ["Paris"]}}]
    assert _eval_bfcl_live("I could call get_weather for Paris but I won't.", gold) is False
    call = [{"type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]
    assert _eval_bfcl_live("", gold, tool_calls=call) is True
    assert _eval_bfcl_live('{"name":"get_weather","arguments":{"city":"Paris"}}', gold) is False  # text
    assert _eval_bfcl_live("I cannot help with that.", []) is True   # abstention


def test_seal_tools_scores_all_gold_calls_not_just_first():
    # parse_gold_call scored only the FIRST of a multi-call gold (~78% are multi-call).
    from gbench.runners.eval_suites.seal_tools import _eval_seal_tools
    gold = "[{'api': 'a', 'parameters': {'x': 1}}, {'api': 'b', 'parameters': {'y': 2}}]"
    both = ('[{"api": "a", "parameters": {"x": 1}, "responses": ["API_call_0"]}, '
            '{"api": "b", "parameters": {"y": 2}, "responses": ["API_call_1"]}]')
    first = '[{"api": "a", "parameters": {"x": 1}, "responses": ["API_call_0"]}]'
    assert _eval_seal_tools(both, gold) is True
    assert _eval_seal_tools(first, gold) is False  # only first


def test_cruxeval_reads_the_answer_tag_block():
    # canonical CRUXEval completes `assert f(input) == <literal>` in [ANSWER] tags,
    # and a Python-literal comparison keeps '0' (str) distinct from 0 (int).
    from gbench.runners.eval_suites.cruxeval import _eval_cruxeval as ev
    assert ev('[ANSWER]\nassert f("x9j") == "x9ja"\n[/ANSWER]', '"x9ja"') is True
    assert ev('[ANSWER]\nassert f(x) == "0"\n[/ANSWER]', '0') is False   # str vs int
    assert ev('[ANSWER]\nassert f(x) == 0\n[/ANSWER]', '0') is True


# --------------------------------------------------------------------------- #
# Exact FunctionCall-AST matching.
#
# The gold is a set of parallel calls, and the metric is an EXACT match over
# the whole set - so leniency in either direction (a missed call, an invented call, an
# extra argument, a stringified integer) must fail.
# --------------------------------------------------------------------------- #

GOLD_CALLS = [
    {"name": "default_api:flight_book",
     "args": {"date": "2022-12-25", "direct_flight": True, "time": "10:00 AM"},
     "param_types": {"date": "string", "direct_flight": "boolean", "time": "string"}},
    {"name": "default_api:imdb_find_movies_by_actor",
     "args": {"actor_name": "Leonardo DiCaprio", "year": 2010.0},
     "param_types": {"actor_name": "string", "year": "integer"}},
]


def _exact(pred, gold=None):
    from gbench.runners.eval_suites.fc_common import score_exact_call_set
    return score_exact_call_set(gold if gold is not None else GOLD_CALLS, pred)["exact"]


def _right():
    return [
        ("default_api_imdb_find_movies_by_actor", {"actor_name": "Leonardo DiCaprio", "year": 2010}),
        ("default_api_flight_book", {"date": "2022-12-25", "direct_flight": True, "time": "10:00 AM"}),
    ]


def test_exact_call_set_matches_regardless_of_order():
    # Parallel calls in one turn: order is not part of the metric. The gold stores `year`
    # as a Struct double (2010.0); an emitted int must still match.
    assert _exact(_right()) is True


def test_exact_call_set_rejects_missing_and_unwarranted_calls():
    assert _exact(_right()[:1]) is False                      # omitted a required call
    assert _exact([]) is False                                # answered in prose only
    extra = _right() + [("default_api_lawsuits_search", {"year": 2015})]
    assert _exact(extra) is False                             # over-triggering


def test_exact_call_set_rejects_argument_defects():
    from gbench.runners.eval_suites.fc_common import score_exact_call_set
    hallucinated = [_right()[0], ("default_api_flight_book",
                                  {"date": "2022-12-25", "direct_flight": True,
                                   "time": "10:00 AM", "seat": "aisle"})]
    assert _exact(hallucinated) is False
    missing = [_right()[0], ("default_api_flight_book", {"date": "2022-12-25", "direct_flight": True})]
    assert _exact(missing) is False
    wrong = [_right()[0], ("default_api_flight_book",
                           {"date": "2021-01-01", "direct_flight": True, "time": "10:00 AM"})]
    assert _exact(wrong) is False
    # the verdict says which check failed rather than only "wrong"
    assert "arguments differ" in score_exact_call_set(GOLD_CALLS, wrong)["reason"]


def test_exact_call_set_value_equivalence_rules():
    lower = [("default_api_imdb_find_movies_by_actor",
              {"actor_name": "leonardo dicaprio", "year": 2010}),
             ("default_api_flight_book",
              {"date": "2022-12-25", "direct_flight": True, "time": "10:00 am"})]
    assert _exact(lower) is True                              # case-insensitive values
    # a boolean parameter answered with 1 is not True (Python's True == 1 must not leak)
    as_one = [_right()[0], ("default_api_flight_book",
                            {"date": "2022-12-25", "direct_flight": 1, "time": "10:00 AM"})]
    assert _exact(as_one) is False


def test_exact_call_set_enforces_declared_parameter_types():
    from gbench.runners.eval_suites.fc_common import score_exact_call_set
    stringy = [("default_api_imdb_find_movies_by_actor",
                {"actor_name": "Leonardo DiCaprio", "year": "2010"}),
               _right()[1]]
    verdict = score_exact_call_set(GOLD_CALLS, stringy)
    # the value is equivalent, but `year` is declared INTEGER and was answered as a string
    assert verdict["exact"] is False and verdict["types_ok"] is False
    assert "expected integer" in verdict["reason"]


def test_tool_name_namespace_is_normalized_on_both_sides():
    from gbench.runners.eval_suites.fc_common import normalize_tool_name
    # `:` is illegal in an OpenAI function name, so the schema is sent as `default_api_x`;
    # the gold keeps the exported `default_api:x` and must still compare equal.
    assert normalize_tool_name("default_api:flight_book") == normalize_tool_name("default_api_flight_book")
    assert normalize_tool_name("a:b") != normalize_tool_name("a_c")


def test_parse_raw_tool_calls_keeps_json_types():
    from gbench.runners.eval_suites.fc_common import parse_raw_tool_calls
    calls = parse_raw_tool_calls([
        {"function": {"name": "f", "arguments": '{"n": 3, "s": "3", "b": true}'}},
        {"function": {"name": "g", "arguments": {"x": 1.5}}},
        {"function": {"name": "bad", "arguments": "not json"}},
    ])
    assert calls[0] == ("f", {"n": 3, "s": "3", "b": True})
    assert isinstance(calls[0][1]["n"], int) and isinstance(calls[0][1]["s"], str)
    assert calls[1] == ("g", {"x": 1.5})
    assert calls[2] == ("bad", {})            # unparseable arguments never become a pass


# --------------------------------------------------------------------------- #
# Answer extraction of more suites
#
# These scorers decided correctness by scanning the whole response, so the model's
# *working* could outvote its *answer*, and short golds matched by accident.
# --------------------------------------------------------------------------- #

def test_cruxeval_rejects_bare_containment():
    from gbench.runners.eval_suites.cruxeval import _eval_cruxeval as C
    # `gold in resp` credited any response that merely mentioned the literal
    assert C("The loop runs 0 times, so the list stays empty and f returns []", "0") is False
    # the canonical answer is the right side of the completed assertion.
    assert C("assert f(x) == 0", "0") is True


def test_cruxeval_is_case_sensitive_for_python_literals():
    from gbench.runners.eval_suites.cruxeval import _eval_cruxeval as C
    assert C("assert f(x) == true", "True") is False      # `true` is not a Python literal
    assert C("assert f(x) == True", "True") is True


def test_cruxeval_does_not_conflate_a_string_with_a_number():
    from gbench.runners.eval_suites.cruxeval import _eval_cruxeval as C
    assert C("assert f(x) == 0", "'0'") is False
    assert C("assert f(x) == '0'", "'0'") is True
    assert C('assert f(x) == "0"', "'0'") is True         # same literal, other quoting
    # CRUXEval checks by executing `assert gold == answer`, so True == 1 passes upstream.
    assert C("assert f(x) == 1", "True") is True


def test_cruxeval_accepts_equivalent_literal_formatting():
    from gbench.runners.eval_suites.cruxeval import _eval_cruxeval as C
    assert C("assert f(x) == [1,2]", "[1, 2]") is True
    assert C("assert f(x) == {'a': 1}", "{'a': 1}") is True
    # upstream reads only the assertion; a \boxed{} answer is not parsed.
    assert C(r"so \boxed{[1, 2]}", "[1, 2]") is False
    assert C("assert f(x) == [2, 1]", "[1, 2]") is False  # order still matters


def test_math_scorers_prefer_the_boxed_answer_over_a_trailing_number():
    # gsm8k is not in this family: it is scored by lm-evaluation-harness gsm8k_cot's filters.
    from gbench.runners.eval_suites.aime import _eval_aime
    from gbench.runners.eval_suites.amc_aime import _eval_amc_aime
    working = "Trying 2024 first... that fails.\n" + r"\boxed{42}"
    assert _eval_aime(working, "42") is True
    assert _eval_amc_aime(working, "42") is True


def test_putnam_rejects_substring_of_the_reference_sentence():
    from gbench.runners.eval_suites.putnam import _eval_putnam as P
    gold = r"The minimum is $12 - 8\sqrt{2}$."
    # `norm_pred in norm_gold` scored a bare `2` correct against this reference
    assert P("Final Answer: 2", gold) is False
    assert P(r"Final Answer: $12 - 8\sqrt{2}$", gold) is True
    assert P(r"Final Answer: 12-8\sqrt{2}", gold) is True   # spacing is not the answer


def test_putnam_leaves_non_closed_form_golds_to_the_judge():
    from gbench.runners.eval_suites.putnam import _eval_putnam, _gold_closed_form
    # one math span -> comparable
    assert _gold_closed_form(r"The limit equals $\frac{1}{8}$.") == r"\frac{1}{8}"
    # none, or several -> not reducible to one value; must not be string-compared
    assert _gold_closed_form("The limit does not exist.") is None
    assert _gold_closed_form(r"$Q(x)$ must have at least $2n - 1$ distinct real roots.") is None
    assert _eval_putnam("Final Answer: the limit does not exist",
                        "The limit does not exist.") is False


def test_putnam_matches_a_fraction_written_either_way():
    from gbench.runners.eval_suites.putnam import _eval_putnam as P
    gold = r"The limit equals $\frac{1}{8}$."
    assert P(r"Final Answer: $\frac{1}{8}$", gold) is True
    assert P("Final Answer: 1/8", gold) is True
    assert P("Final Answer: 1/9", gold) is False


# --------------------------------------------------------------------------- #
# D-11/12: mcp_atlas claim verification, omnidocbench edit distance
# --------------------------------------------------------------------------- #

MCP_CLAIMS = ('["The AssaultCube GitHub repository was created in 2013.", '
              '"The domain registration year is 2006."]')


def test_mcp_atlas_parses_list_columns_that_are_strings():
    from gbench.runners.eval_suites.mcp_atlas import _parse_list_field
    # ENABLED_TOOLS/GTFA_CLAIMS arrive as a *string* holding a list; slicing the raw value
    # produced `[, ", f, e, ...` as the prompt's tool list.
    assert _parse_list_field('["fetch_fetch","whois_whois_domain"]') == [
        "fetch_fetch", "whois_whois_domain"]
    # Python quoting, which json.loads cannot read
    assert _parse_list_field("['A is 2013.', \"B is 2006.\"]") == ["A is 2013.", "B is 2006."]
    assert _parse_list_field(None) == []


def test_mcp_atlas_no_longer_passes_on_echoed_task_nouns():
    from gbench.runners.eval_suites.mcp_atlas import _eval_mcp_atlas as M
    # 75% token overlap passed a response that repeated the task's vocabulary and never
    # produced the decisive values (2013 / 2006).
    assert M("AssaultCube GitHub repository domain registration year created", MCP_CLAIMS) is False
    assert M("The AssaultCube GitHub repository was created in 2013. "
             "The domain registration year is 2006.", MCP_CLAIMS) is True
    assert M("", MCP_CLAIMS) is False


def test_omnidocbench_uses_edit_distance_not_word_overlap():
    from gbench.runners.eval_suites.omnidocbench import (
        _eval_omnidocbench, normalized_edit_distance)
    gold = "The quick brown fox jumps over the lazy dog near the river bank at dawn."
    assert normalized_edit_distance(gold, gold) == 0.0
    # a page with most of its content missing shares >40% of gold's word types, which the
    # old set-overlap rule passed
    truncated = "The quick brown fox"
    assert _eval_omnidocbench(truncated, gold) is False
    assert normalized_edit_distance(truncated, gold) > 0.5
    # markdown styling is transcription style, not error
    assert _eval_omnidocbench("**The quick brown fox** jumps over the lazy dog "
                              "near the river bank at dawn.", gold) is True


# --------------------------------------------------------------------------- #
# D-14: long-context and execution suites
# --------------------------------------------------------------------------- #

def test_mrcr_applies_the_canary_gate_and_scores_a_ratio():
    from gbench.runners.eval_suites.mrcr import grade_mrcr
    gold = {"answer": "CANARY42\n\nA poem about the sea.", "canary": "CANARY42"}
    # canonical: no canary prefix -> 0, whatever the content
    assert grade_mrcr("A poem about the sea.", gold) == 0.0
    assert grade_mrcr("CANARY42\n\nA poem about the sea.", gold) == 1.0
    partial = grade_mrcr("CANARY42\n\nA poem about the ocean.", gold)
    assert 0.5 < partial < 1.0          # a ratio, not a pass/fail containment


def test_mrcr_pass_requires_a_verbatim_reproduction():
    from gbench.runners.eval_suites.mrcr import _eval_mrcr
    gold = {"answer": "CANARY42\n\nA poem about the sea.", "canary": "CANARY42"}
    assert _eval_mrcr("CANARY42\n\nA poem about the sea.", gold) is True
    # containment of the needle is not a reproduction
    assert _eval_mrcr("CANARY42 Here you go: A poem about the sea. Hope that helps!",
                      gold) is False


def test_ruler_string_match_and_bands():
    # RULER rewrite: canonical string_match_all/part (was ruler_recall) + exact-band labels.
    from gbench.runners.eval_suites.ruler import _string_match_all, _string_match_part, _band_label
    needles = ["alpha", "bravo", "charlie", "delta"]
    assert _string_match_all("found alpha bravo charlie", needles) == 0.75  # fraction present
    assert _string_match_all("nothing here", needles) == 0.0
    assert _string_match_all("alpha bravo charlie delta", needles) == 1.0
    assert _string_match_part("found alpha only", needles) == 1.0           # any-of (qa)
    assert _string_match_part("nothing here", needles) == 0.0
    assert _band_label(4096) == "4k" and _band_label(8192) == "8k" and _band_label(131072) == "128k"


def test_execution_suites_do_not_auto_pass_untested_submissions():
    from gbench.runners.eval_suites.codeforces import _eval_codeforces
    from gbench.runners.eval_suites.lcb import _verify_single_lcb_sample as _eval_lcb
    assert _eval_codeforces("print('hi')", []) is False        # no tests -> unverified
    assert _eval_lcb("print('hi')", "not a payload") is False
    assert _eval_lcb("print('hi')", {"task": "generation", "tests": []}) is False


def test_multipl_e_does_not_duplicate_a_returned_definition():
    # Upstream chat protocol (MultiPL-E chat_completions.py): the model returns the WHOLE program
    # with the prefix repeated, and the evaluator file carries prompt="", so the definition
    # appears once. The container appends `+ "\n" + tests` itself.
    from gbench.runners.eval_suites.multipl_e import parse_chat_fields, postprocess_completion
    prompt = 'def add(x: int, y: int) -> int:\n    """ Add two numbers."""\n'
    response = ("[[ ## reasoning ## ]]\nAdd them.\n\n[[ ## completion ## ]]\n```python\n"
                'def add(x: int, y: int) -> int:\n    """ Add two numbers."""\n    return x + y\n```'
                "\n\n[[ ## completed ## ]]")
    code = postprocess_completion([], prompt, parse_chat_fields(response)["completion"])
    assert code.count("def add(") == 1
    assert "return x + y" in code


def test_i18n_translate_uses_chrf_not_a_bag_of_words():
    from gbench.runners.eval_suites.i18n_translate import chrf_score
    gold = "The cat sat on the mat."
    assert chrf_score(gold, gold) > 99.0
    # same word set, scrambled: set-F1 scored this identical to a correct translation
    scrambled = chrf_score("mat the on sat cat The.", gold)
    assert scrambled < chrf_score(gold, gold)
    assert chrf_score("", gold) == 0.0


# --------------------------------------------------------------------------- #
# D-15: judge and harness robustness
# --------------------------------------------------------------------------- #

def test_empty_response_is_counted_not_silently_scored_wrong():
    import asyncio
    from unittest import mock
    from gbench.runners.eval_suites import base

    async def _fake(*a, **k):
        return base.Reply("", None, "stop", None)   # HTTP 200 with empty `content`

    samples = [([{"role": "user", "content": "q"}], "gold", {}) for _ in range(3)]
    with mock.patch.object(base, "_send_single_request", _fake):
        result = asyncio.run(base._run_suite_async(
            eval_name="t", model_name="m", base_url="http://x/v1", concurrency=1,
            samples=samples, eval_fn=lambda r, g: False))
    # a truncated generation is a broken request, not just a wrong answer
    assert result["empty_responses"] == 3
    assert result["failed_requests"] == 3
    assert result["status"] == "failed"
    assert all(t["status"] == "EMPTY_RESPONSE" for t in result["sample_traces"])


def test_judge_failure_downgrades_the_run_status():
    import asyncio
    from unittest import mock
    from gbench.runners.eval_suites import base

    async def _fake(*a, **k):
        return base.Reply("an answer", None, "stop", None)

    async def _judge(traces):
        traces[0]["is_correct"] = True
        traces[0]["judge_grade"] = "pass"
        traces[1]["is_correct"] = False
        traces[1]["judge_grade"] = "judge_error"     # three API attempts, no verdict

    samples = [([{"role": "user", "content": "q"}], "gold", {}) for _ in range(2)]
    with mock.patch.object(base, "_send_single_request", _fake):
        result = asyncio.run(base._run_suite_async(
            eval_name="t", model_name="m", base_url="http://x/v1", concurrency=1,
            samples=samples, async_eval_fn=_judge))
    # previously the run reported `success` and the ungraded sample just counted as wrong
    assert result["judge_failures"] == 1
    assert result["status"] == "completed_with_errors"


def test_healthbench_headline_is_the_mean_rubric_score():
    from unittest import mock
    from gbench.runners.eval_suites import healthbench

    # Every conversation scores just under the 0.5 threshold: the canonical mean rubric
    # score is 49%, while the pass rate the suite used to publish as `accuracy` is 0%.
    fake = {
        "accuracy": 0.0, "total_questions": 2, "correct_answers": 0,
        "sample_traces": [{"healthbench_score": 0.48}, {"healthbench_score": 0.50}],
    }
    with mock.patch.object(healthbench, "gemini_required_skip", lambda *a: None), \
         mock.patch.object(healthbench, "_load_healthbench_samples", lambda **k: []), \
         mock.patch.object(healthbench, "run_eval_suite", lambda **k: dict(fake)):
        result = healthbench.run_healthbench("m", "http://x/v1", 1)

    assert result["accuracy"] == 49.0
    assert result["healthbench_mean_rubric_score"] == 0.49
    assert result["pass_rate_at_0.5"] == 0.0        # kept, but no longer the headline
    assert "mean weighted rubric score" in result["metric"]


def test_simpleqa_reads_the_topic_out_of_stringified_metadata():
    from gbench.runners.eval_suites.simpleqa import _simpleqa_topic
    # every row reported the default "geography" because the topic is inside `metadata`
    assert _simpleqa_topic("{'topic': 'Science and technology', 'answer_type': 'Person'}") \
        == "Science and technology"
    assert _simpleqa_topic({"topic": "Music"}) == "Music"
    assert _simpleqa_topic("") is None


def test_gpqa_shuffles_options_per_item_deterministically():
    # simple-evals GPQAEval: one random.Random(0) permutation per (question, repeat) copy.
    from gbench.runners.eval_suites.gpqa import _gpqa_sample, _permutations
    assert _permutations(40, 1) == _permutations(40, 1)          # reproducible across runs
    row = {"Question": "q", "Correct Answer": "zzz correct", "Incorrect Answer 1": "wrong a",
           "Incorrect Answer 2": "wrong b", "Incorrect Answer 3": "wrong c"}
    perm = _permutations(1, 1)[0]
    assert sorted(perm) == [0, 1, 2, 3]                          # nothing lost

    # `sorted(options)` pinned the answer's slot to its own text: this correct answer
    # sorts last, so it was option (D) on every item. After shuffling its position varies.
    positions = {_gpqa_sample(row, p)[1] for p in _permutations(40, 1)}
    assert len(positions) > 1 and positions != {"D"}


# --------------------------------------------------------------------------- #
# Real process isolation for suites that execute model-written code
# --------------------------------------------------------------------------- #

def test_sandbox_mode_knob_and_none_is_a_passthrough(monkeypatch):
    from gbench.runners.eval_suites import sandbox
    monkeypatch.setenv("GBENCH_SANDBOX", "none")
    sandbox.sandbox_available.cache_clear()
    assert sandbox.sandbox_mode() == "none"
    assert sandbox.wrap_argv(["echo", "hi"]) == ["echo", "hi"]
    sandbox.sandbox_available.cache_clear()


def test_sandbox_wraps_with_a_read_only_no_network_jail():
    from gbench.runners.eval_suites import sandbox
    argv = sandbox.wrap_argv(["python3", "-c", "pass"], writable=["/tmp"], force=True)
    assert argv[0] == "bwrap"
    assert "--unshare-net" in argv                  # `--sandboxes` never did this
    assert "--ro-bind" in argv and "--tmpfs" in argv
    assert argv[-3:] == ["python3", "-c", "pass"]
    # a declared scratch dir is bound writable; nothing else is
    assert "--bind" in argv and "/tmp" in argv
    assert "--bind" not in sandbox.wrap_argv(["true"], force=True)


def test_sandbox_bwrap_mode_fails_loudly_when_unavailable(monkeypatch):
    from gbench.runners.eval_suites import sandbox
    monkeypatch.setenv("GBENCH_SANDBOX", "bwrap")
    monkeypatch.setattr(sandbox, "_probe_bwrap", lambda: False)
    sandbox.sandbox_available.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="bubblewrap"):
            sandbox.sandbox_available()
    finally:
        sandbox.sandbox_available.cache_clear()


def test_sandbox_required_is_the_default_and_blocks_when_unavailable(monkeypatch):
    # Security default: with GBENCH_SANDBOX unset, code-executing suites must require
    # bubblewrap and never run unsandboxed when it is absent. sandbox_skip_reason() returns a
    # block message which evals.py turns into infra_required (a hard error, not a skip).
    from gbench.runners.eval_suites import sandbox
    monkeypatch.delenv("GBENCH_SANDBOX", raising=False)
    monkeypatch.setattr(sandbox, "_probe_bwrap", lambda: False)
    sandbox.sandbox_available.cache_clear()
    try:
        assert sandbox.sandbox_mode() == "required"
        assert sandbox.sandbox_available() is False
        reason = sandbox.sandbox_skip_reason()
        assert reason and "bubblewrap" in reason
        with pytest.raises(RuntimeError, match="bubblewrap"):
            sandbox.wrap_argv(["echo", "hi"])
    finally:
        sandbox.sandbox_available.cache_clear()


# --------------------------------------------------------------------------- #
# Remaining suite items: deepsearch_qa, gaia
# --------------------------------------------------------------------------- #

def test_deepsearch_qa_is_judge_only_no_string_match_fallback():
    """Canonical DeepSearchQA is graded by the Gemini judge only. The non-canonical string-match
    fallback (and its scoring function) was removed, and a missing backend hard-errors rather
    than silently swapping in string-match scoring or skipping."""
    import inspect, re
    from gbench.runners.eval_suites import deepsearch_qa as D
    assert not hasattr(D, "_eval_deepsearch_qa"), "string-match scoring path removed (judge-only)"
    run_src = inspect.getsource(D.run_deepsearch_qa)
    assert "infra_required" in run_src and "skipped_result" not in run_src
    assert "async_eval_fn=" in run_src                          # judge-only path
    assert re.search(r"(?<!async_)eval_fn\s*=", run_src) is None  # no standalone eval_fn
    assert "GROUNDED_SEARCH_TOOL" in run_src         # search IS canonical for this benchmark
    judge_src = inspect.getsource(D._async_judge_deepsearch_qa)
    assert "judge_fallback" not in judge_src


def test_gaia_final_answer_regex_handles_spacing():
    from gbench.runners.eval_suites.gaia import _extract_final_answer as E
    # the fixed 14-char slice cut the answer whenever spacing differed
    assert E("FINAL ANSWER: 42") == "42"
    assert E("FINAL ANSWER:42") == "42"
    assert E("FINAL ANSWER:  42") == "42"
    assert E("final answer: Paris") == "Paris"
    # the model's LAST word on it wins
    assert E("FINAL ANSWER: 41\nOn reflection:\nFINAL ANSWER: 42") == "42"
    assert E("no marker here") == ""





def test_charxiv_scorer_uses_its_own_parameter():
    """`_eval_charxiv` referenced an undefined `gold_answers`, so it raised on every row."""
    from gbench.runners.eval_suites.charxiv import _eval_charxiv as C
    assert C("Answer: 42", "42") is True    # relaxed match parses the stated answer, not the first number
    assert C("The value is 99", "42") is False
    assert C("", "42") is False


def test_a_wrong_answer_is_not_counted_as_a_judge_failure():
    """Execution suites use status FAILED for an ordinary wrong answer.

    Inferring a judge failure from it counted every incorrect sample as a harness error
    and downgraded a healthy run to `completed_with_errors` (caught on lcb: 1 wrong
    solution out of 2 reported judge_failures=1).
    """
    import asyncio
    from unittest import mock
    from gbench.runners.eval_suites import base

    async def _fake(*a, **k):
        return base.Reply("```python\nprint(1)\n```", None, "stop", None)

    async def _score(traces):
        traces[0]["is_correct"] = True
        traces[0]["status"] = "OK"
        traces[1]["is_correct"] = False
        traces[1]["status"] = "FAILED"        # lcb: solution failed its tests

    samples = [([{"role": "user", "content": "q"}], "g", {}) for _ in range(2)]
    with mock.patch.object(base, "_send_single_request", _fake):
        result = asyncio.run(base._run_suite_async(
            eval_name="t", model_name="m", base_url="http://x/v1", concurrency=1,
            samples=samples, async_eval_fn=_score))
    assert result["judge_failures"] == 0
    assert result["failed_requests"] == 0
    assert result["status"] == "success"
    assert result["accuracy"] == 50.0


def test_judge_verdict_reads_the_verdict_not_the_whole_reply():
    """Judges ignore the one-word instruction and return paragraphs of analysis.

    `"CORRECT" in text and "INCORRECT" not in text` therefore scored a verbose
    `GRADE: CORRECT` as wrong whenever the reasoning happened to contain the word
    "incorrect" - observed live: cybergym and HLE judges both returned multi-paragraph
    replies mentioning both words.
    """
    from gbench.runners.eval_suites.base import parse_grade_verdict as P
    assert P("The patch is not incorrect at all.\nGRADE: CORRECT") is True
    assert P("Root cause analysis: **CORRECT**.\nGRADE: INCORRECT") is False
    assert P("GRADE: CORRECT ... on reflection GRADE: INCORRECT") is False   # last wins
    assert P("GRADE: NOT_ATTEMPTED") is False
    assert P("Grade: correct") is True                                       # case-insensitive
    # no explicit verdict -> previous substring behaviour, so nothing regresses
    assert P("This answer is correct.") is True
    assert P("This answer is incorrect.") is False
    assert P("") is False


# --------------------------------------------------------------------------- #
# Two false zeros (bfcl 0/20, seal_tools 0/20 were scorer bugs)
# --------------------------------------------------------------------------- #

def test_possible_answer_tries_every_candidate_call():
    """base.py renders one call twice: `f({"a":1}) f(a=1)`.

    parse_tool_calls returns two entries - the first with the JSON object swallowed as a
    positional `__pos0`. Binding to the first name match scored correct calls wrong.
    """
    from gbench.runners.eval_suites.fc_common import score_possible_answer
    rendered = ('calculate_triangle_area({"base": 10, "height": 5, "unit": "units"}) '
                'calculate_triangle_area(base=10, height=5, unit=units)')
    gold = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["units", ""]}}]
    assert score_possible_answer(rendered, gold) is True
    # a genuinely wrong value still fails
    wrong = 'calculate_triangle_area({"base": 99}) calculate_triangle_area(base=99)'
    assert score_possible_answer(wrong, gold) is False


def test_unquoted_kwarg_is_kept_not_dropped():
    """`unit=units` parses as a Name node: literal_eval fails, so it became None."""
    from gbench.runners.eval_suites.fc_common import parse_tool_calls
    calls = dict(parse_tool_calls("f(a=1, unit=units)"))["f"] if parse_tool_calls("f(a=1, unit=units)") else {}
    assert calls.get("unit") == "units"
    assert calls.get("a") == 1


def test_python_literal_gold_is_parsed():
    """seal_tools ships its gold as a Python literal; json.loads cannot read it."""
    from gbench.runners.eval_suites.fc_common import parse_gold_call
    gold = ("[{'api': 'analyzeEvidence', 'parameters': {'evidence_type': 'iSYSgFgqKb', "
            "'method': 'microscopy'}, 'responses': ['API_call_0']}]")
    parsed = parse_gold_call(gold)
    assert parsed is not None, "python-literal gold must parse"
    assert parsed[0] == "analyzeEvidence"
    assert parsed[1]["method"] == "microscopy"
    # JSON golds keep working
    assert parse_gold_call('[{"name": "f", "arguments": {"a": 1}}]')[0] == "f"


def test_seal_tools_scores_a_correct_answer():
    from gbench.runners.eval_suites.seal_tools import _eval_seal_tools
    gold = ("[{'api': 'analyzeEvidence', 'parameters': {'evidence_type': 'iSYSgFgqKb', "
            "'method': 'microscopy', 'sample': 'fabric sample'}}]")
    # transform_output_format only accepts a list that also carries "responses".
    good = ('```json\n[{"api": "analyzeEvidence", "parameters": {"evidence_type": '
            '"iSYSgFgqKb", "method": "microscopy", "sample": "fabric sample"}, '
            '"responses": ["API_call_0"]}]\n```')
    assert _eval_seal_tools(good, gold) is True
    assert _eval_seal_tools('[{"api": "somethingElse", "parameters": {}, "responses": []}]', gold) is False


# --------------------------------------------------------------------------- #
# Generation health (finish_reason + reasoning capture)
# --------------------------------------------------------------------------- #

def test_classify_reply_separates_the_four_failure_modes():
    from gbench.runners.eval_suites.base import Reply, classify_reply as C
    assert C(Reply("an answer", None, "stop", None)) == "ok"
    assert C(Reply("", [{"function": {}}], "tool_calls", None)) == "ok"   # tool call only
    assert C(Reply("half a solu", None, "length", None)) == "truncated"
    assert C(Reply("", None, "stop", "pages of thinking")) == "empty_reasoned"
    assert C(Reply("", None, "stop", None)) == "empty"
    assert C(Reply(None, None, None, None)) == "request_failed"


def test_truncated_answers_are_counted_and_downgrade_the_status():
    """A truncated answer was indistinguishable from a wrong one.

    codeforces' IOI scored 0/6 purely because 5 of 6 responses hit the 8192-token budget
    mid-function; swe_bench_multilingual lost 10/20 the same way.
    """
    import asyncio
    from unittest import mock
    from gbench.runners.eval_suites import base

    async def _cut_off(*a, **k):
        return base.Reply("def solve():\n    partial", None, "length", None)

    samples = [([{"role": "user", "content": "q"}], "g", {}) for _ in range(4)]
    with mock.patch.object(base, "_send_single_request", _cut_off):
        result = asyncio.run(base._run_suite_async(
            eval_name="t", model_name="m", base_url="http://x/v1", concurrency=1,
            samples=samples, eval_fn=lambda r, g: False))
    assert result["truncated_responses"] == 4
    assert result["response_health"]["truncated"] == 4
    assert result["status"] == "completed_with_errors"     # not a clean "success"
    assert all(t["finish_reason"] == "length" for t in result["sample_traces"])


def test_reasoning_only_response_is_labelled_not_just_empty():
    """gemma-4 can spend the whole budget in the reasoning channel, leaving content empty."""
    import asyncio
    from unittest import mock
    from gbench.runners.eval_suites import base

    async def _reasoned(*a, **k):
        return base.Reply("", None, "stop", "long private reasoning" * 20)

    samples = [([{"role": "user", "content": "q"}], "g", {}) for _ in range(2)]
    with mock.patch.object(base, "_send_single_request", _reasoned):
        result = asyncio.run(base._run_suite_async(
            eval_name="t", model_name="m", base_url="http://x/v1", concurrency=1,
            samples=samples, eval_fn=lambda r, g: False))
    assert result["response_health"]["empty_reasoned"] == 2
    # The answer written inside the thinking channel is the model's outcome
    # (ANSWER_IN_THINKING_POLICY): scored wrong, counted apart, never a failed request.
    assert result["empty_responses"] == 0 and result["failed_requests"] == 0
    assert result["answer_in_thinking"] == 2 and result["answer_in_thinking_rate"] == 1.0
    assert result["answer_in_thinking_of"] == 2
    assert result["status"] == "success" and result["scoring_excluded"] == 0
    assert result["accuracy"] == 0.0
    assert all(t["reasoning_chars"] > 0 and t["answer_in_thinking"] is True
               and t["is_correct"] is False and not t.get("scoring_excluded")
               for t in result["sample_traces"])


def test_answer_in_thinking_is_scored_and_only_a_truly_empty_reply_is_a_failed_request():
    """One correct reply, one reply with its answer inside the thinking channel and one reply
    with nothing at all: the in-thinking reply is in the denominator as a wrong answer and leaves
    the run `success`; only the reply with nothing is a failed request."""
    import asyncio
    from unittest import mock
    from gbench.runners.eval_suites import base

    replies = {"a": base.Reply("right", None, "stop", "thought"),
               "b": base.Reply("", None, "stop", "the answer is right"),
               "c": base.Reply("", None, "stop", None)}

    async def _send(*a, **k):
        return replies[k["messages"][0]["content"]]

    samples = [([{"role": "user", "content": q}], "right", {}) for q in "ab"]
    with mock.patch.object(base, "_send_single_request", _send):
        two = asyncio.run(base._run_suite_async(
            eval_name="t", model_name="m", base_url="http://x/v1", concurrency=1,
            samples=samples, eval_fn=lambda r, g: r.strip() == g, thinking=True))
        three = asyncio.run(base._run_suite_async(
            eval_name="t", model_name="m", base_url="http://x/v1", concurrency=1,
            samples=samples + [([{"role": "user", "content": "c"}], "right", {})],
            eval_fn=lambda r, g: r.strip() == g, thinking=True))
    assert two["status"] == "success" and two["status_reason"] is None
    assert two["accuracy"] == 50.0 and two["correct_answers"] == 1
    assert two["answer_in_thinking"] == 1 and two["answer_in_thinking_rate"] == 0.5
    assert two["failed_requests"] == 0 and two["empty_responses"] == 0
    in_thinking = [t for t in two["sample_traces"] if t.get("answer_in_thinking")]
    assert [t["health"] for t in in_thinking] == ["empty_reasoned"]
    assert in_thinking[0]["status"] == "EMPTY_RESPONSE"
    # the reply with nothing at all stays a failed request (completed_with_errors)
    assert three["empty_responses"] == 1 and three["failed_requests"] == 1
    assert three["answer_in_thinking"] == 1
    assert three["status"] == "completed_with_errors"
    assert "EMPTY_RESPONSE 1" in three["status_reason"]
    assert three["accuracy"] == round(100 / 3, 2) or abs(three["accuracy"] - 100 / 3) < 0.01


def test_a_healthy_run_still_reports_success():
    import asyncio
    from unittest import mock
    from gbench.runners.eval_suites import base

    async def _ok(*a, **k):
        return base.Reply("the answer", None, "stop", None)

    samples = [([{"role": "user", "content": "q"}], "g", {}) for _ in range(5)]
    with mock.patch.object(base, "_send_single_request", _ok):
        result = asyncio.run(base._run_suite_async(
            eval_name="t", model_name="m", base_url="http://x/v1", concurrency=1,
            samples=samples, eval_fn=lambda r, g: True))
    assert result["status"] == "success"
    assert result["truncated_responses"] == 0
    assert result["response_health"] == {"ok": 5}
    assert result["accuracy"] == 100.0


# --------------------------------------------------------------------------- #
# Per-suite wall-clock budget (opt-in)
# --------------------------------------------------------------------------- #

def test_suite_timeout_is_off_by_default():
    """A total-runtime cap cannot tell 'wedged' from 'legitimately slow', and cutting a
    suite loses its partial results and leaks its containers - so it must be opt-in."""
    import time
    from gbench.runners.evals import _run_with_timeout

    def slow():
        time.sleep(0.3)
        return {"status": "success", "accuracy": 42.0}

    # unset -> runs to completion, no budget applied
    assert _run_with_timeout("s", lambda: slow(), {}, None, "m")["accuracy"] == 42.0
    assert _run_with_timeout("s", lambda: slow(), {}, 0, "m")["accuracy"] == 42.0


def test_suite_timeout_when_requested_reports_timeout_and_continues():
    import time
    from gbench.runners.evals import _run_with_timeout

    def wedged():
        time.sleep(30)
        return {"status": "success"}

    result = _run_with_timeout("tau3", lambda: wedged(), {}, 1, "m")
    assert result["status"] == "timeout"
    # a failed configuration: no score (never 0.0), counted as failed, non-zero exit
    assert result["total_questions"] == 0 and result["accuracy"] is None
    assert result["failed"] is True
    assert "wall-clock budget" in result["skip_reason"]


def test_suite_timeout_passes_kwargs_through():
    from gbench.runners.evals import _run_with_timeout
    seen = {}

    def runner(**kw):
        seen.update(kw)
        return {"status": "success"}

    _run_with_timeout("s", runner, {"limit": 20, "model_name": "m"}, 30, "m")
    assert seen == {"limit": 20, "model_name": "m"}


# --------------------------------------------------------------------------- #
# structural zeros found in the live 2026-08-15 sweep
# --------------------------------------------------------------------------- #
def test_nestful_scores_the_call_sequence_not_the_final_number():
    """NESTFUL's `gold_answer` is the arithmetic result; the metric is the CALL SEQUENCE.

    Scoring the call-matching evaluator against "40.0" could only ever return 0 - the
    sweep measured 0/20 on every row.
    """
    import json
    from gbench.runners.eval_suites.nestful import _eval_nestful
    gold = json.dumps([
        {"name": "divide", "label": "$var_1", "arguments": {"arg_0": 25, "arg_1": 100}},
        {"name": "multiply", "label": "$var_2", "arguments": {"arg_0": 80, "arg_1": 0.25}},
    ])
    good = json.dumps([{"name": "divide", "arguments": {"arg_0": 25, "arg_1": 100}},
                       {"name": "multiply", "arguments": {"arg_0": 80, "arg_1": 0.25}}])
    assert _eval_nestful(good, gold) is True
    assert _eval_nestful("The answer is 40.0", gold) is False   # prose maths is not a call


def test_nestful_puts_function_specs_in_the_prompt_not_the_api():
    """Canonical NESTFUL presents the function specs + few-shot examples IN THE PROMPT and asks
    for the whole nested sequence as text. Offering the OpenAI tools API instead makes the model
    emit ONE structured tool_call and stop (finish_reason=tool_calls), never composing the
    sequence (measured: 0% full/partial/win). So: specs in a system message, no API tools."""
    from gbench.runners.eval_suites import nestful as N
    tools = [{"name": "divide", "description": "divide two numbers",
              "parameters": {"arg_0": {"description": "a", "type": "int or float"}},
              "output_parameters": {"result": {"type": "int or float"}}}]
    sysmsg = N._system_prompt(tools)
    assert "divide" in sysmsg and "output_parameters" in sysmsg      # the row's specs, as JSON
    assert "#Example-1" in sysmsg and "#Example-3" in sysmsg          # the 3 canonical ICL demos
    assert "$var_1.result$" in sysmsg                                 # the nested-ref format
    # the loader must build system+user messages and must NOT offer API tools
    import inspect
    src = inspect.getsource(N._load_nestful_samples)
    assert '"role": "system"' in src and '"role": "user"' in src
    assert '"tools"' not in src.split("gold = json.dumps")[1].split("samples.append")[0] \
        or 'meta["tools"]' not in src   # no API tools in the sample meta


def test_nestful_icl_block_matches_canonical_format():
    """get_icl_str (generic branch): each demo is `#Example-N\\nInput: ...\\nOutput: <json>`."""
    from gbench.runners.eval_suites import nestful as N
    s = N._icl_str()
    assert s.count("#Example-") == 3
    assert "\nInput: " in s and "\nOutput: " in s


def test_nestful_win_rate_executes_the_sequence():
    """Executable Win Rate: run the composed calls against IBM/NESTFUL's reference functions and
    compare the final value to gold_answer (25/100*80 = 20.0). Needs GBENCH_NESTFUL_FUNC_DIR."""
    import os, pytest
    from gbench.runners.eval_suites import nestful as N
    fd = os.environ.get(N._FUNC_DIR_ENV)
    if not fd:
        pytest.skip("GBENCH_NESTFUL_FUNC_DIR not set (IBM/NESTFUL executable_functions)")
    tools = [{"name": "divide", "output_parameters": {"result": {"type": "int or float"}}},
             {"name": "multiply", "output_parameters": {"result": {"type": "int or float"}}}]
    seq = [{"name": "divide", "label": "$var_1", "arguments": {"arg_0": 25, "arg_1": 100}},
           {"name": "multiply", "label": "$var_2", "arguments": {"arg_0": "$var_1.result$", "arg_1": 80}}]
    assert N._win_score(seq, "20.0", tools, N._func_dir()) is True
    wrong = [dict(seq[0]),
             {"name": "multiply", "label": "$var_2", "arguments": {"arg_0": "$var_1.result$", "arg_1": 999}}]
    assert N._win_score(wrong, "20.0", tools, N._func_dir()) is False
    assert N._win_score([], "20.0", tools, N._func_dir()) is False


def test_nestful_win_rate_requires_the_reference_functions(monkeypatch):
    """No reference-function library -> hard-error (infra_required), never a skip/partial."""
    import pytest
    from gbench.runners.eval_suites import nestful as N
    monkeypatch.delenv(N._FUNC_DIR_ENV, raising=False)
    with pytest.raises(RuntimeError, match="executable Win Rate needs"):
        N._func_dir()


def test_nestful_gold_carries_answer_and_tools():
    """The gold must carry the sequence, gold_answer and raw tools (with output_parameters) so
    both match-scoring and the executable Win Rate can run; _gold_sequence stays back-compatible
    with a legacy bare list."""
    import inspect, json
    from gbench.runners.eval_suites import nestful as N
    src = inspect.getsource(N._load_nestful_samples)
    assert '"gold_answer"' in src and '"tools"' in src and '"output"' in src
    g = json.dumps({"output": [{"name": "f", "label": "$v", "arguments": {"a": 1}}],
                    "gold_answer": "1", "tools": []})
    assert len(N._gold_sequence(g)) == 1
    assert len(N._gold_sequence(json.dumps([{"name": "f", "arguments": {"a": 1}}]))) == 1


def test_nestful_win_rate_is_reported_not_stubbed():
    import inspect
    from gbench.runners.eval_suites import nestful as N
    src = inspect.getsource(N.run_nestful)
    assert 'result["win_rate"]' in src
    assert "not implemented" not in src, "Win Rate is implemented now"
    assert "_func_dir()" in src


def test_gorilla_requires_the_checkpoint_not_just_the_class():
    """A name-only fallback credited whisper-large against a gold of whisper-tiny. APIBench's
    ast_check only matches a database call when the reply also carries its arguments (the
    checkpoint). Hand-built nodes shaped like tree-sitter's stand in for the grammar."""
    from gbench.runners.eval_suites.gorilla_apibench import ast_check, get_all_sub_trees, get_args

    class Node:
        def __init__(self, text, children=()):
            self.text, self.children = text.encode(), list(children)
            self.child_count = len(self.children)

    def call_tree(fn, ckpt):
        arg = Node(f"'{ckpt}'", [Node("'"), Node(ckpt), Node("'")])
        call = Node(f"{fn}('{ckpt}')", [Node(fn, [Node("x")]), Node(f"('{ckpt}')", [Node("("), arg, Node(")")])])
        return Node(call.text.decode(), [Node(call.text.decode(), [call])])

    fn = "WhisperForConditionalGeneration.from_pretrained"
    db = [call_tree(fn, "openai/whisper-tiny")]
    args = [get_args(t, "huggingface") for t in db]
    assert ast_check(get_all_sub_trees(call_tree(fn, "openai/whisper-tiny")), db, args) == 0
    assert ast_check(get_all_sub_trees(call_tree(fn, "openai/whisper-large")), db, args) == -1
    assert ast_check(get_all_sub_trees(Node("Tool: OpenAI Whisper API")), db, args) == -1


def test_gorilla_prompt_asks_for_the_api_call():
    """"Respond with: Tool: <model_name_or_api_call>" invited a prose tool NAME -> 0/20."""
    import inspect
    from gbench.runners.eval_suites import gorilla_apibench
    src = inspect.getsource(gorilla_apibench.encode_question)   # Gorilla's own prompt builder
    # strip comments: the old wording is quoted in the explanatory comment above the fix
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    assert "<<<api_call>>>" in code
    assert "Tool: <model_name_or_api_call>" not in code


# --------------------------------------------------------------------------- #
# cyberseceval: the Insecure Code Detector, not a text heuristic
# --------------------------------------------------------------------------- #
def _cyber_scan(monkeypatch, issues_for):
    """Stub the ICD image's scan: each program's issues are issues_for(code); the programs sent
    are recorded."""
    from gbench.runners.eval_suites import cyberseceval as C
    sent = []

    def fake_scan(items):
        sent.extend(items)
        return ({i["id"]: {"id": i["id"], "issues": issues_for(i["code"]), "attempts": 1}
                 for i in items}, None)
    monkeypatch.setattr(C, "_scan_in_image", fake_scan)
    return sent


def _cyber_score(response_text, gold):
    import asyncio
    from gbench.runners.eval_suites import cyberseceval as C
    trace = {"gold_answer": gold, "response_text": response_text}
    asyncio.run(C._async_score_cyberseceval([trace]))
    return trace


def test_cyberseceval_secure_insecure_mapping_is_from_the_icd(monkeypatch):
    """secure iff Meta's ICD returns no issues; the language is read from the gold JSON."""
    import json
    sent = _cyber_scan(monkeypatch, lambda code: [] if "safe" in code else [{"analyzer": "x"}])
    gold = json.dumps({"language": "python", "cwe": "x"})
    assert _cyber_score("```python\nsafe = 1\n```", gold)["is_correct"] is True
    assert sent[-1]["language"] == "python"
    assert _cyber_score("```python\ndanger()\n```", gold)["is_correct"] is False


def test_cyberseceval_scores_the_code_not_the_prose(monkeypatch):
    """Only the first fenced block is scored; prose around it is not passed to the ICD."""
    import json
    sent = _cyber_scan(monkeypatch, lambda code: [])
    resp = "Avoid eval( for security.\n```python\nx = int(inp)\n```"
    _cyber_score(resp, json.dumps({"language": "python"}))
    assert "Avoid eval(" not in sent[0]["code"] and "x = int(inp)" in sent[0]["code"]


def test_cyberseceval_empty_is_secure_unknown_language_is_not_scored(monkeypatch):
    import json
    from gbench.runners.eval_suites import base
    from gbench.runners.eval_suites import cyberseceval as C
    sent = _cyber_scan(monkeypatch, lambda code: [{"analyzer": "x"}])  # never for empty
    # Upstream scores an empty/refusal response as SECURE (the ICD finds nothing) - not insecure.
    assert _cyber_score("", json.dumps({"language": "python"}))["is_correct"] is True
    assert _cyber_score("   ", json.dumps({"language": "python"}))["is_correct"] is True
    assert sent == []
    # A language the ICD does not have cannot be scored: the ICD's error is a harness error, left
    # out of the score, never a guessed verdict.
    monkeypatch.setattr(C, "_scan_in_image", lambda items: ({items[0]["id"]: {
        "id": items[0]["id"], "error": {"kind": "icd_unknown_language", "detail": "klingon"}}},
        None))
    t = _cyber_score("```x\ncode\n```", json.dumps({"language": "klingon"}))
    assert t["is_correct"] is False and not base.is_scored(t)
    assert t["harness_error"]["kind"] == "icd_unknown_language"


def test_cyberseceval_code_extraction_is_first_block_like_upstream():
    """Canonical: the FIRST ```-fenced block (lang hint kept), or the whole message if none."""
    from gbench.runners.eval_suites.cyberseceval import extract_code
    got = extract_code("blah\n```python\nx = 1\n```\n```py\ny = 2\n```")
    assert got.strip().endswith("x = 1") and "y = 2" not in got
    assert extract_code("no fences here") == "no fences here"


def _cyber_probe(monkeypatch, selftest):
    from gbench.runners.eval_suites import cyberseceval as C
    monkeypatch.setattr(C, "_PROBE", {"selftest": selftest})


def test_cyberseceval_semgrep_selftest_hard_errors_when_semgrep_is_silent(monkeypatch):
    """A broken semgrep returns no findings -> would mark every program secure. The self-test
    in the image (Python eval(), which only semgrep catches) must hard-error instead of
    scoring; a finding of another analyzer does not count."""
    import pytest
    from gbench.runners.eval_suites import cyberseceval as C
    for issues in ([], [{"analyzer": "regex", "pattern_id": "x"}]):
        _cyber_probe(monkeypatch, {"issues": issues, "semgrep_runs": 1, "attempts": 1})
        with pytest.raises(RuntimeError, match="docs/evals/cyberseceval.md"):
            C._selftest_semgrep_or_raise()
    _cyber_probe(monkeypatch, {"error": {"kind": "icd_semgrep_no_output", "detail": "x"}})
    with pytest.raises(RuntimeError, match="icd_semgrep_no_output"):
        C._selftest_semgrep_or_raise()


def test_cyberseceval_semgrep_selftest_passes_when_eval_is_flagged(monkeypatch):
    from gbench.runners.eval_suites import cyberseceval as C
    _cyber_probe(monkeypatch, {"issues": [{"analyzer": "semgrep",
                                           "pattern_id": "insecure-eval-use"}]})
    assert C._selftest_semgrep_or_raise()["issues"][0]["analyzer"] == "semgrep"


def test_cyberseceval_requires_the_icd_image(monkeypatch):
    """No Docker (so no ICD image) -> hard-error (infra_required) naming the build command,
    never a skip."""
    import pytest
    from gbench.runners.eval_suites import cyberseceval as C
    monkeypatch.setattr(C.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="Insecure Code Detector runs in the local image") as e:
        C.check_icd_image()
    assert C.icd_image_build_command() in str(e.value)


def test_cyberseceval_warns_that_the_old_checkout_variables_have_no_effect(monkeypatch, caplog):
    """GBENCH_CYBERSECEVAL_ICD and CYBERSECEVAL_ICD_RULES pointed at a host checkout; the ICD
    now runs in its image, so a run that still sets one is told to unset it."""
    import logging
    from gbench.runners.eval_suites import cyberseceval as C
    monkeypatch.delenv(C.ICD_DIR_ENV, raising=False)
    monkeypatch.setenv(C.LEGACY_RULES_ENV, "/x/CodeShield/insecure_code_detector/rules")
    with caplog.at_level(logging.WARNING, logger=C.__name__):
        C._warn_removed_env()
    assert C.LEGACY_RULES_ENV in caplog.text and "Unset it" in caplog.text


def test_cyberseceval_run_has_no_skip_or_partial_paths():
    """The runner must hard-error (never skip/partial) and guard semgrep before scoring."""
    import inspect
    from gbench.runners.eval_suites import cyberseceval as C
    mod_src = inspect.getsource(C)
    assert "skipped_result" not in mod_src, "no skip path"
    assert "coverage_note" not in mod_src, "no regex-only partial"
    run_src = inspect.getsource(C.run_cyberseceval)
    assert "_selftest_semgrep_or_raise" in run_src
    assert "apply_comparability(" in run_src
    assert '"leaderboard_comparable"' in inspect.getsource(C.apply_comparability)


def test_cyberseceval_prompt_file_load_failure_is_hard_error(monkeypatch, tmp_path):
    """The pinned prompt file that cannot be read from the ICD image must hard-error, never fall
    back to another prompt set."""
    import pytest
    from gbench.runners.eval_suites import cyberseceval as C

    def boom(request, timeout_s):
        raise C._WorkerError("docker could not be started (FileNotFoundError: docker)")
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(C, "_run_worker", boom)
    with pytest.raises(RuntimeError, match="could not be read from"):
        C._load_cyberseceval_samples(limit=2)


def test_omnidocbench_pairs_pages_by_recovered_filename():
    """The HF split has ONE column, `image` - no filename - so pairing by `image_path` /
    `file_name` / `page_name` always missed and every page fell back to positional
    pairing against a different page's ground truth (20/20 unpaired, guaranteed 0/20).
    The UNDECODED image feature keeps `path`, whose basename is the annotation key."""
    import inspect
    from gbench.runners.eval_suites import omnidocbench as O
    src = inspect.getsource(O._load_omnidocbench_samples)
    assert "Image(decode=False)" in src, "the filename must come from the undecoded feature"
    assert "page_names" in src


def test_omnidocbench_drops_unannotated_pages_instead_of_mispairing():
    """A page with no annotation must be excluded, not graded against another page."""
    import inspect
    from gbench.runners.eval_suites import omnidocbench as O
    src = inspect.getsource(O._load_omnidocbench_samples)
    body = src[src.index("anno = anno_by_name.get(page_name)"):]
    assert "continue" in body[:600]
    assert "annotations[i] if i < len(annotations)" not in body, "positional fallback is back"


def test_bigcodebench_pass_k_is_a_tuple_not_an_int():
    """Fire parses `--pass-k 1` as an int and evaluate() iterates it:
    TypeError: 'int' object is not iterable. A trailing comma makes it a 1-tuple."""
    import os as _os
    src = open(_os.path.join(_os.path.dirname(__file__), "..", "gbench", "runners",
                             "eval_suites", "bigcodebench.py")).read()
    assert '"--pass-k", "1,"' in src
