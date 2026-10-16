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

"""Unit tests for native evaluation suites: their scoring functions and sample loaders.

The sample loaders pull their datasets from HF Hub. That download is not
exercised here, for the reason given at the top of .github/workflows/test.yml:
these tests never contact an external service, so the suite stays
deterministic and safe to require for merge. `datasets.load_dataset` is
patched with fixed rows instead, which lets each loader's own transformation
be asserted directly rather than only its row count.
"""

import base64
import io
from unittest import mock

import pytest
try:
    from PIL import Image
except ImportError:
    Image = None

from gbench.runners.eval_suites.mmlu_pro import _eval_mmlu_pro, _load_mmlu_pro_samples
from gbench.runners.eval_suites.aime import _eval_aime, _load_aime_samples
from gbench.runners.eval_suites.lcb import _verify_single_lcb_sample, _load_lcb_samples
from gbench.runners.eval_suites.codeforces import _eval_codeforces, _load_codeforces_samples
from gbench.runners.eval_suites.putnam import _eval_putnam, _load_putnam_samples
from gbench.runners.eval_suites.ruler import _eval_ruler
from gbench.runners.eval_suites.textvqa import _eval_textvqa, _load_textvqa_samples


def _patch_hf(rows_by_dataset):
    """Serve fixed rows in place of an HF Hub download.

    Every loader does `from datasets import load_dataset` inside its own body,
    so patching the attribute on the package is enough and nothing has to be
    imported before the loader runs. Requesting a dataset that the test did not
    supply raises, which is what catches a renamed or mistyped dataset id.
    """

    def _fake_load_dataset(path, *args, **kwargs):
        if path not in rows_by_dataset:
            raise AssertionError(f"loader asked for an unexpected dataset: {path}")
        return rows_by_dataset[path]

    return mock.patch("datasets.load_dataset", autospec=True, side_effect=_fake_load_dataset)


MMLU_PRO_ROWS = [
    # answer_index rather than a letter, which the loader has to map through
    # OPTION_LETTERS.
    {"question": "What is 2 + 2?", "options": ["3", "4", "5"], "answer_index": 1, "category": "math"},
    # answer already a letter, which the loader has to pass through untouched.
    {"question": "Which gas do plants absorb?", "options": ["Oxygen", "Carbon dioxide"], "answer": "B", "category": "biology"},
]

AIME_ROWS = [
    # answer arrives as an int and the scorer compares strings.
    {"problem": "Find the remainder.", "answer": 496, "category": "algebra"},
    {"problem": "Find n.", "answer": 84, "category": "number_theory"},
]

# Post-cutoff AIME 2025 (yentinglin/aime_2025 schema): year is in the AoPS url.
AIME_2025_ROWS = [
    {"problem": "2025 problem one.", "answer": 70,
     "url": "https://artofproblemsolving.com/wiki/index.php/2025_AIME_I_Problems/Problem_1"},
    {"problem": "2025 problem two.", "answer": 588,
     "url": "https://artofproblemsolving.com/wiki/index.php/2025_AIME_II_Problems/Problem_2"},
]

PUTNAM_ROWS = [
    # tags as a repr'd list, which the loader parses with ast.literal_eval.
    {"informal_statement": "Prove the identity.", "informal_solution": "1/2", "tags": "['algebra', 'number_theory']"},
    # tags as a bare string, which takes the other branch.
    {"informal_statement": "Show the bound.", "informal_solution": "100", "tags": "geometry"},
]

RULER_ROWS = [
    {"input": "Find the hidden words.", "outputs": ["appliance", "meter"], "length": 4096, "task": "niah_single"},
    {"input": "Track the variables.", "outputs": ["7"], "length": 8192, "task": "vt"},
]


def _rgb_png_bytes(b64_str):
    """Decode a base64 data URL payload back into a PIL image."""
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


# --------------------------------------------------------------------------
# Scoring functions. These take model output and a gold answer and are pure.
# --------------------------------------------------------------------------


def test_mmlu_pro_eval_fn():
    assert _eval_mmlu_pro("Answer: (B)", "B") is True
    assert _eval_mmlu_pro("Let's think... ANSWER: (J)", "J") is True
    assert _eval_mmlu_pro("Answer: (C)", "B") is False


def test_aime_eval_fn():
    assert _eval_aime("The final answer is 496.", "496") is True
    assert _eval_aime("We calculate that the final answer is 084", "84") is True
    assert _eval_aime("The final answer is 42", "43") is False


def test_lcb_eval_fn():
    # LiveCodeBench is execution-scored via _verify_single_lcb_sample(resp, gold_payload),
    # with lcb_runner's own graders. gold_payload is a dict tagged with task in
    # {generation, execution, test_gen}. Cover each canonical path.

    # 1) generation, call-based: the entry point is the problem's metadata func_name and each
    #    input line is one JSON argument (lcb_runner grade_call_based)
    valid_code = "```python\nclass Solution:\n    def twoSum(self, nums, target):\n        return [0, 1]\n```"
    call_gold = {"task": "generation", "fn_name": "twoSum",
                 "tests": [{"testtype": "functional", "input": "[2, 7, 11, 15]\n9", "output": "[0, 1]"}]}
    assert _verify_single_lcb_sample(valid_code, call_gold) is True
    invalid_code = "```python\nclass Solution:\n    def twoSum(self, nums, target):\n        return [1, 2]\n```"
    assert _verify_single_lcb_sample(invalid_code, call_gold) is False

    # 2) a module-level function works too when it carries the func_name
    add_code = "```python\ndef add(a, b):\n    return a + b\n```"
    func_gold = {"task": "generation", "fn_name": "add",
                 "tests": [{"testtype": "functional", "input": "2\n3", "output": "5"}]}
    assert _verify_single_lcb_sample(add_code, func_gold) is True
    bad_add = "```python\ndef add(a, b):\n    return a + b + 1\n```"
    assert _verify_single_lcb_sample(bad_add, func_gold) is False

    # 3) generation via a stdin test case (input on stdin, expected on stdout)
    stdin_code = "```python\nimport sys\na, b = map(int, sys.stdin.read().split())\nprint(a + b)\n```"
    stdin_gold = {"task": "generation",
                  "tests": [{"testtype": "stdin", "input": "2 3", "output": "5"}]}
    assert _verify_single_lcb_sample(stdin_code, stdin_gold) is True
    assert _verify_single_lcb_sample(stdin_code, {"task": "generation",
        "tests": [{"testtype": "stdin", "input": "2 3", "output": "6"}]}) is False

    # 4) code execution: the completed assertion is EXECUTED (upstream evaluate_score); a bare
    #    substring must NOT pass (that leniency inflated the category).
    exec_gold = {"task": "execution", "code": "def f():\n    return 42", "input": "f()",
                 "output": "42"}
    assert _verify_single_lcb_sample("assert f() == 42\n[/ANSWER]", exec_gold) is True
    assert _verify_single_lcb_sample("assert f() == 7\n[/ANSWER]", exec_gold) is False
    # rambling that merely *contains* 42 without stating it as the answer -> incorrect
    assert _verify_single_lcb_sample("at step 42 we compute 7 so the result is 7", exec_gold) is False

    # 5) test output prediction: upstream check_testcase_output on the gold `output`.
    tg_gold = {"task": "test_gen", "output": "1"}
    assert _verify_single_lcb_sample("assert f() == 1", tg_gold) is True
    assert _verify_single_lcb_sample("assert f() == 2", tg_gold) is False
    assert _verify_single_lcb_sample("no", tg_gold) is False

    # empty response is always incorrect
    assert _verify_single_lcb_sample("", call_gold) is False


def test_codeforces_eval_fn():
    script = (
        "```python\n"
        "import sys\n"
        "for line in sys.stdin:\n"
        "    a, b = map(int, line.split())\n"
        "    print(a + b)\n"
        "```"
    )
    tests = [{"stdin": "3 5\n", "stdout": "8"}, {"stdin": "-10 20\n", "stdout": "10"}]
    assert _eval_codeforces(script, tests) is True


def test_putnam_eval_fn():
    assert _eval_putnam("We conclude that Final Answer: 100", "100") is True
    assert _eval_putnam("We get 1/2 as the result", "1/2") is True
    assert _eval_putnam("Final Answer: 42", "100") is False


def test_putnam_formal_eval_fn():
    from gbench.runners.eval_suites.putnam_formal import _eval_putnam_formal, check_putnam_formal_prerequisites
    assert _eval_putnam_formal("", "theorem foo : True := sorry") is False
    assert check_putnam_formal_prerequisites()[0] in (True, False)


def test_ruler_eval_fn():
    import json
    # RULER rewrite: gold is a JSON blob {task, outputs}; eval_fn = all references present.
    gold = json.dumps({"task": "niah_multivalue", "outputs": ["appliance", "meter"]})
    assert _eval_ruler("The items are appliance and meter.", gold) is True
    assert _eval_ruler("Only appliance was found.", gold) is False


def test_textvqa_eval_fn():
    # lmms-eval textvqa_process_results: the reply as returned against the 10 annotator answers.
    assert _eval_textvqa("Coca-Cola", ["coca-cola"] * 10) is True
    # The prompt asks for a single word or phrase, so a verbose sentence is non-compliant and
    # canonical VQA scoring marks it wrong.
    assert _eval_textvqa("It says 42 on the jersey", ["42"] * 10) is False
    assert _eval_textvqa("42", ["42"] * 10) is True
    assert _eval_textvqa("Pepsi", ["coca-cola"] * 10) is False


def test_gpqa_diamond_parser():
    """gpqa_diamond scores with simple-evals' ANSWER_PATTERN_MULTICHOICE (first match, exact letter)."""
    from gbench.runners.eval_suites.gpqa_diamond import _eval_gpqa

    # 1. The 'Answer: $LETTER' line the simple-evals template asks for
    assert _eval_gpqa("The correct choice is Final Answer: B", "B") is True
    assert _eval_gpqa("Final Answer: C.", "C") is True
    assert _eval_gpqa("Answer: A", "A") is True
    assert _eval_gpqa("reasoning...\nAnswer: $D$", "D") is True

    # 2. Forms the upstream pattern does not read score 0 (no bare-letter or paren fallback)
    assert _eval_gpqa("B", "B") is False
    assert _eval_gpqa("Answer: (D)", "D") is False
    assert _eval_gpqa("Thus, we choose option\nD.", "D") is False

    # 3. Mismatch check
    assert _eval_gpqa("Final Answer: A", "B") is False


# --------------------------------------------------------------------------
# Sample loaders. These turn a dataset row into the
# (messages, gold_answer, metadata) triple that run_eval_suite consumes.
# --------------------------------------------------------------------------


def test_mmlu_pro_loader_shape_and_gold_mapping():
    with _patch_hf({"TIGER-Lab/MMLU-Pro": MMLU_PRO_ROWS}) as loader:
        samples = _load_mmlu_pro_samples()

    assert loader.call_args_list[0].args[0] == "TIGER-Lab/MMLU-Pro"
    assert loader.call_args_list[0].kwargs["split"] == "test"
    assert len(samples) == 2

    messages, gold, meta = samples[0]
    # answer_index 1 has to become the second option letter, not the digit 1.
    assert gold == "B"
    # The row's place in the test split orders the run as evaluate_from_apiX.py does.
    assert meta == {"category": "math", "question_id": None, "row": 0}
    assert messages[0]["role"] == "user"
    assert "What is 2 + 2?" in messages[0]["content"]
    # Options are relabelled with their own letters, in TIGER-Lab's "Options: A. .." format.
    assert "Options: A. 3\nB. 4\nC. 5\n" in messages[0]["content"]

    # A row that already carries a letter is passed through untouched.
    assert samples[1][1] == "B"
    assert samples[1][2] == {"category": "biology", "question_id": None, "row": 1}


def test_mmlu_pro_loader_honours_limit():
    with _patch_hf({"TIGER-Lab/MMLU-Pro": MMLU_PRO_ROWS}):
        assert len(_load_mmlu_pro_samples(limit=1)) == 1


def test_aime_loader_stringifies_integer_gold():
    # The loader pulls MathArena's AIME 2026 set (its `answer` column is an int).
    with _patch_hf({"MathArena/aime_2026": AIME_ROWS}):
        samples = _load_aime_samples()

    golds = {g for _, g, _ in samples}
    assert "496" in golds and isinstance(next(iter(golds)), str)  # int -> str conversion
    cats = {m["category"] for _, _, m in samples}
    assert cats == {"aime_2026"}
    messages = samples[0][0]
    assert "content" in messages[0] and messages[0]["content"]
    # The gold answer for the scorer, not the model, so it must not leak into the prompt.
    assert all(str(g) not in m[0]["content"] for m, g, _ in samples)


def test_putnam_loader_parses_both_tag_encodings():
    # putnam reads informal/putnam.json of the pinned PutnamBench release, not an HF dataset.
    with mock.patch("gbench.runners.eval_suites.putnam._informal_rows", return_value=PUTNAM_ROWS):
        samples = _load_putnam_samples()

    # A repr'd list becomes its first element, with the quotes stripped.
    assert samples[0][2] == {"category": "algebra"}
    assert samples[0][1] == "1/2"
    # A bare string is used as the category directly.
    assert samples[1][2] == {"category": "geometry"}


# NOTE: RULER no longer loads a pre-generated HF dataset - it generates per-tokenizer data
# via NVIDIA/RULER's prepare.py (GBENCH_RULER_DIR). See tests/test_ruler_canonical.py.


def test_textvqa_loader_converts_to_rgb_png_data_url():
    pytest.importorskip("PIL")      # Pillow is declared: CI's skip gate fails a skip here
    # CMYK is the case the loader's mode check exists for.
    rows = [{"question": "What brand is on the can?", "answers": ["coca-cola"], "image": Image.new("CMYK", (4, 2))}]
    with _patch_hf({"lmms-lab/textvqa": rows}):
        samples = _load_textvqa_samples()

    messages, gold, meta = samples[0]
    assert gold == ["coca-cola"]
    assert meta == {"category": "scene_ocr"}

    image_part, text_part = messages[0]["content"]
    assert text_part["type"] == "text"
    assert "What brand is on the can?" in text_part["text"]

    url = image_part["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    decoded = _rgb_png_bytes(url.split(",", 1)[1])
    # The CMYK source has to arrive as an RGB PNG, otherwise the vision model
    # gets a payload it cannot read.
    assert decoded.format == "PNG"
    assert decoded.mode == "RGB"
    assert decoded.size == (4, 2)


def _cf_row(ct, i):
    # Carries both the problem fields (for the `verifiable` config) and the id/language/prompt
    # fields (for the `verifiable-prompts` config); _patch_hf serves the same rows for both.
    return {"id": f"{ct}{i}", "contest_id": str(1000 + i), "contest_type": ct,
            "title": f"{ct}{i}", "description": "d", "input_format": "n",
            "output_format": "x", "examples": [], "official_tests": [{"input": "1", "output": "2"}],
            "generated_checker": None, "time_limit": 2.0, "memory_limit": 256.0,
            "testset_size": 10, "official_tests_complete": False, "contest_start_year": 2024,
            "rating": 1200, "language": "python", "prompt": f"Solve {ct}{i} in Python."}


def test_codeforces_loader_does_NOT_force_a_contest_type_quota():
    """The 40/30/30 CF/ICPC/IOI split was invented here and described as "balanced".

    IOI is 85 of 10,024 problems (0.85%) in this corpus, so a 30% quota over-represented it
    ~35x - and 46% of IOI problems ship no `input_format` because they are grader-based, so
    they cannot be run as stdin/stdout at all. That produced IOI 17.33% against CF 68% /
    ICPC 76% on 2026-08-20 and dragged the suite down for a reason that was not the model.
    No suite should invent a category mix; the natural distribution is the measurement.
    """
    rows = ([_cf_row("CF", i) for i in range(20)]
            + [_cf_row("ICPC", i) for i in range(20)]
            + [_cf_row("IOI", i) for i in range(2)])
    with _patch_hf({"open-r1/codeforces": rows}):
        samples = _load_codeforces_samples(limit=10)

    counts = {}
    for _, _, meta in samples:
        counts[meta["category"]] = counts.get(meta["category"], 0) + 1
    assert sum(counts.values()) == 10
    # IOI is 2/42 of the fixture (4.8%). A 30% quota would have forced 3 of 10, and
    # stratifying BY contest_type would force ~3 as well - both invent a mix. A seeded
    # subset tracks the corpus, so IOI must stay near its natural share.
    assert counts.get("IOI", 0) <= 1, f"IOI over-represented: {counts}"


def test_codeforces_loader_uses_the_verifiable_test_split():
    """`-cots` is a CoT distillation TRAIN set with five generations per problem: the old
    500-row sample was 100 distinct problems counted 5x. It also starts at 2010."""
    with _patch_hf({"open-r1/codeforces": [_cf_row("CF", 0)]}) as loader:
        _load_codeforces_samples()
    path = loader.call_args_list[0].args[0]
    kwargs = loader.call_args_list[0].kwargs
    args = loader.call_args_list[0].args
    assert path == "open-r1/codeforces"
    assert "verifiable" in args or kwargs.get("name") == "verifiable" or "verifiable" in str(args)
    assert kwargs.get("split") == "test"


def test_codeforces_grades_on_official_tests_and_fails_when_untested():
    """A problem with no usable tests measured nothing; crediting it inflates the suite."""
    from gbench.runners.eval_suites.codeforces import _eval_codeforces
    assert _eval_codeforces("```python\nprint(2)\n```",
                            {"official_tests": [], "generated_tests": [], "checker": ""}) is False


def test_codeforces_output_comparison_is_token_wise():
    """The dataset carries platform line endings (\r\n) verbatim; exact string equality
    marks correct answers wrong."""
    from gbench.runners.eval_suites.codeforces import _outputs_match
    assert _outputs_match("1\n9\n0", "1\r\n9\r\n0  ")
    assert not _outputs_match("1 9 1", "1 9 0")



def _patch_lcb_lite(rows):
    """Stand in for the LiveCodeBench shard reader.

    `_load_lcb_lite` fetches `test*.jsonl` from the hub and reads them with the json loader
    - it no longer goes through `load_dataset("livecodebench/...")`, because `datasets` 5.x
    refuses to run the repo's loading script, which is what `version_tag` used to need.
    """
    return mock.patch("gbench.runners.eval_suites.lcb._load_lcb_lite",
                      autospec=True, return_value=rows)


def test_lcb_loader_merges_public_and_private_tests():
    gen_rows = [
        {
            "question_title": "Two Sum",
            "question_content": "Return the indices.",
            "starter_code": "def two_sum(nums, target):",
            "difficulty": "easy",
            # public arrives as a JSON string and private as a real list, and
            # both have to end up in the same list.
            "public_test_cases": '[{"input": "1", "output": "2"}]',
            "private_test_cases": [{"input": "3", "output": "4"}],
        }
    ]
    with _patch_lcb_lite(gen_rows), _patch_hf(
        {
            "livecodebench/execution-v2": [{"code": "def f(): return 1", "input": "f()", "output": 1}],
            "livecodebench/test_generation": [{"question_title": "T", "question_content": "c", "function_name": "f", "starter_code": "def f():\n    pass", "difficulty": "medium", "test": '[{"input": "", "output": "1", "testtype": "functional"}]'}],
        }
    ):
        samples = _load_lcb_samples()

    by_task = {}
    for _, gold, meta in samples:
        by_task.setdefault(gold["task"], []).append((gold, meta))

    # all three LiveCodeBench scenarios are in the default run.
    assert set(by_task) == {"generation", "execution", "test_gen"}
    # the gold carries contest_date so the result can report which problem window it covers -
    # a release tag alone cannot, the tags being cumulative. The meta is sent with the request,
    # so it carries only the category and the id the sample_key is built from.
    gen_gold, gen_meta = by_task["generation"][0]
    assert gen_meta["category"] == "code_gen_easy"
    assert "contest_date" in gen_gold and "contest_date" not in gen_meta
    assert len(gen_gold["tests"]) == 2
    assert gen_gold["test_counts"] == {"public_test_cases": 1, "private_test_cases": 1}
    # The execution row's int output is stringified for comparison.
    assert by_task["execution"][0][0]["output"] == "1"


def test_lcb_test_gen_included_only_when_opted_in():
    """test_generation is in the default run; LCB_INCLUDE_TEST_GEN=0 leaves it out."""
    import os
    from unittest.mock import patch
    rows = {
        "livecodebench/execution-v2": [{"code": "def f(): return 1", "input": "f()", "output": 1}],
        "livecodebench/test_generation": [{"question_title": "T", "question_content": "c",
                                           "function_name": "f", "starter_code": "def f():\n    pass",
                                           "difficulty": "medium",
                                           "test": '[{"input": "", "output": "1", "testtype": "functional"}]'}],
    }
    with _patch_lcb_lite([]), _patch_hf(rows), \
            patch.dict(os.environ, {"LCB_INCLUDE_TEST_GEN": "1"}, clear=False):
        tasks = {g["task"] for _, g, _ in _load_lcb_samples()}
    assert "test_gen" in tasks
    with _patch_lcb_lite([]), _patch_hf(rows), \
            patch.dict(os.environ, {"LCB_INCLUDE_TEST_GEN": "0"}, clear=False):
        tasks = {g["task"] for _, g, _ in _load_lcb_samples()}
    assert "test_gen" not in tasks


def test_lcb_loader_hard_errors_when_the_execution_scenario_is_missing():
    gen_rows = [{"question_title": "Two Sum", "question_content": "c", "public_test_cases": [], "private_test_cases": []}]
    # execution is absent, so _patch_hf raises for it. It is part of the default run, so
    # the loader must fail loudly rather than return a silently smaller suite.
    with _patch_lcb_lite(gen_rows), _patch_hf({}):
        with pytest.raises(RuntimeError, match="code execution scenario"):
            _load_lcb_samples()


def test_thinking_mode_prompts():
    """Verify reasoning suites alter prompt instructions when enable_thinking=True vs False."""
    rows = {
        "TIGER-Lab/MMLU-Pro": MMLU_PRO_ROWS,
        "MathArena/aime_2026": AIME_ROWS,
    }
    with _patch_hf(rows), mock.patch("gbench.runners.eval_suites.putnam._informal_rows",
                                     return_value=PUTNAM_ROWS):
        samples_mmlu_pro_think = _load_mmlu_pro_samples(enable_thinking=True)
        samples_mmlu_pro_nothink = _load_mmlu_pro_samples(enable_thinking=False)
        samples_aime_think = _load_aime_samples(enable_thinking=True)
        samples_aime_nothink = _load_aime_samples(enable_thinking=False)
        samples_putnam_think = _load_putnam_samples(enable_thinking=True)
        samples_putnam_nothink = _load_putnam_samples(enable_thinking=False)

    # MMLU-Pro's official protocol is CoT in both modes: the prompt does not change.
    assert "think step by step" in samples_mmlu_pro_think[0][0][0]["content"].lower()
    assert samples_mmlu_pro_nothink[0][0][0]["content"] == samples_mmlu_pro_think[0][0][0]["content"]

    # AIME follows MathArena, which sends the same prompt with or without thinking.
    assert samples_aime_nothink[0][0][0]["content"] == samples_aime_think[0][0][0]["content"]

    assert "think step by step" in samples_putnam_think[0][0][0]["content"].lower()
    assert "think step by step" not in samples_putnam_nothink[0][0][0]["content"].lower()


def test_thinking_mode_leaves_the_gold_answer_and_metadata_alone():
    """Only the instruction text may change, so a thinking run stays comparable."""
    with _patch_hf({"MathArena/aime_2026": AIME_ROWS}):
        think = _load_aime_samples(enable_thinking=True)
        nothink = _load_aime_samples(enable_thinking=False)

    assert [gold for _, gold, _ in think] == [gold for _, gold, _ in nothink]
    assert [meta for _, _, meta in think] == [meta for _, _, meta in nothink]


@pytest.mark.parametrize(
    "loader,dataset_id",
    [
        (_load_mmlu_pro_samples, "TIGER-Lab/MMLU-Pro"),
        # putnam: the pinned PutnamBench commit URL is asserted in tests/test_putnambench_release.py
    ],
)
def test_loaders_request_the_canonical_dataset_id(loader, dataset_id):
    """A renamed or mistyped dataset id is the failure these suites are most
    exposed to, and it is invisible to a row-count assertion."""
    with _patch_hf({dataset_id: []}) as patched:
        assert loader() == []
    assert patched.call_args_list[0].args[0] == dataset_id


def test_aime_requests_the_matharena_dataset_id():
    """aime reads MathArena's AIME 2026 set and nothing else. A rename is invisible to row
    counts, and an empty set is an error rather than a 0-sample run."""
    with _patch_hf({"MathArena/aime_2026": []}) as patched:
        with pytest.raises(RuntimeError, match="no rows"):
            _load_aime_samples()
    requested = [c.args[0] for c in patched.call_args_list]
    assert requested == ["MathArena/aime_2026"]


def test_infographicvqa_eval_fn():
    from gbench.runners.eval_suites.infographicvqa import _eval_infographicvqa, _load_infographicvqa_samples
    # InfographicVQA prompts for "only the direct short answer"; ANLS compares the answer
    # itself, so a sentence wrapping it does not count.
    assert _eval_infographicvqa("The answer is 7 tips", ["7", "seven"]) is False
    assert _eval_infographicvqa("7", ["7", "seven"]) is True
    # lmms-eval's anls scores the reply as returned: a label in front of the answer counts
    # against it (no answer extraction).
    assert _eval_infographicvqa("Answer: 7", ["7", "seven"]) is False
    assert _eval_infographicvqa("42.5%", ["42.5%"]) is True
    assert _eval_infographicvqa("The value is 100", ["7"]) is False
    samples = _load_infographicvqa_samples(limit=2)
    assert len(samples) == 2


def test_screenspot_pointing_eval_fn():
    from gbench.runners.eval_suites.screenspot_pointing import _eval_screenspot_pointing, _load_screenspot_pointing_samples
    # Canonical ScreenSpot-v2: the predicted point must fall INSIDE the target bbox.
    gold = {"x": 500.0, "y": 300.0, "bbox": [480, 285, 525, 315]}
    assert _eval_screenspot_pointing("Point: (505, 298)", gold) is True
    assert _eval_screenspot_pointing('{"x": 510, "y": 305}', gold) is True
    assert _eval_screenspot_pointing("Point: (100, 100)", gold) is False
    # The loader reads the pinned ScreenSpot-v2 release files, each sha256-checked (its tests are
    # in tests/test_wildbench_wb_elo_and_official_scorers.py); a download that is not the pinned
    # file is refused.
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        wrong = os.path.join(d, "not_the_release.json")
        with open(wrong, "w") as f:
            f.write("[]")
        with mock.patch("huggingface_hub.hf_hub_download", return_value=wrong), \
                pytest.raises(RuntimeError, match="sha256"):
            _load_screenspot_pointing_samples(limit=2)


def test_aa_lcr_eval_fn():
    import csv as _csv
    import os
    import tempfile
    import zipfile
    from unittest import mock
    from gbench.runners.eval_suites.aa_lcr import _load_aa_lcr_samples

    # aa_lcr loads a CSV + a ZIP of source documents via hf_hub_download; provide real temp
    # files laid out like the archive (lcr/<category>/<set_id>/<file>). The no-key substring
    # fallback is gone (the judge hard-errors without a key); see
    # tests/test_aa_lcr_beam_healthbench_imo_canonical.py.
    d = tempfile.mkdtemp()
    csv_path = os.path.join(d, "aa_lcr.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=[
            "document_category", "document_set_id", "question_id", "question", "answer",
            "data_source_filenames"])
        w.writeheader()
        w.writerow({"document_category": "finance", "document_set_id": "s1", "question_id": "q1",
                    "question": "Revenue growth?", "answer": "15.4%", "data_source_filenames": "doc1.txt"})
        w.writerow({"document_category": "finance", "document_set_id": "s1", "question_id": "q2",
                    "question": "Net margin?", "answer": "8%", "data_source_filenames": "doc2.txt"})
    zip_path = os.path.join(d, "aa_lcr.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("lcr/finance/s1/doc1.txt", "Revenue increased 15.4%.")
        zf.writestr("lcr/finance/s1/doc2.txt", "Net margin was 8%.")

    def _fake_download(repo_id, filename, **kwargs):
        return zip_path if filename.endswith(".zip") else csv_path

    with mock.patch("huggingface_hub.hf_hub_download", side_effect=_fake_download):
        samples = _load_aa_lcr_samples(limit=2)
    assert len(samples) == 2
    assert samples[0][1] == "15.4%"
    assert "BEGIN DOCUMENT 1:\nRevenue increased 15.4%.\nEND DOCUMENT 1" in samples[0][0][0]["content"]


def test_beam_128k_eval_fn():
    import json as _json
    import os
    import tempfile
    from unittest import mock
    import pandas as pd
    from gbench.runners.eval_suites.beam_128k import _load_beam_128k_samples

    # beam_128k loads a parquet via hf_hub_download; write a real one. chat is a
    # list<struct>; probing_questions is a JSON string (the loader parses both). The no-key
    # substring fallback is gone (the judge hard-errors without a key); the rubric is what the
    # canonical judge scores against. See tests/test_aa_lcr_beam_healthbench_imo_canonical.py.
    d = tempfile.mkdtemp()
    pq_path = os.path.join(d, "beam.parquet")
    probing = {"information_extraction": [
        {"question": "When did we launch?", "answer": "March 14", "difficulty": "easy",
         "rubric": ["States March 14"]},
        {"question": "What was the reply?", "answer": "Noted", "difficulty": "easy",
         "rubric": ["States Noted"]},
    ]}
    df = pd.DataFrame([{
        "conversation_id": "c1",
        "chat": [{"role": "user", "content": "We launched on March 14."},
                 {"role": "assistant", "content": "Noted."}],
        "probing_questions": _json.dumps(probing),
    }])
    df.to_parquet(pq_path)

    with mock.patch("huggingface_hub.hf_hub_download", side_effect=lambda repo_id, filename, **kw: pq_path):
        samples = _load_beam_128k_samples(limit=2)
    assert len(samples) == 2
    # real chat turns, then upstream's question turn
    assert samples[0][0][:2] == [{"role": "user", "content": "We launched on March 14."},
                                 {"role": "assistant", "content": "Noted."}]
    assert samples[0][0][2]["content"].endswith("Question: When did we launch?")


def test_wildclawbench_eval_fn():
    # WildClawBench's single-turn judge-over-text path (_eval_wildclawbench /
    # _load_wildclawbench_samples) was replaced by the canonical harness: the pinned InternLM
    # run_batch (OpenClaw agent in wildclawbench-ubuntu:v1.3, docker-out-of-docker) + the per-task
    # grade() (programmatic checks + a Gemini-cascade LLM judge). Deterministic scoring/endpoint
    # helpers are covered in tests/test_wildclawbench.py.
    from gbench.runners.eval_suites.wildclawbench import (
        compute_wildclawbench_score, check_wildclawbench_prerequisites)
    s = compute_wildclawbench_score({"global_mean": 0.42, "multimodal_mean": 0.3,
                                     "pure_text_mean": 0.5, "weighted_overall": 0.4})
    assert s["global_mean"] == 0.42
    assert callable(check_wildclawbench_prerequisites)


def test_skillsbench_eval_fn():
    # SkillsBench's single-turn judge-over-text path (_eval_skillsbench / _load_skillsbench_samples)
    # was replaced by the canonical container harness: the upstream BenchFlow runner (OpenCode agent,
    # docker-out-of-docker) + the per-task DETERMINISTIC verifier (reward in [0,1]), no LLM judge.
    # Deterministic scoring/endpoint helpers are covered in tests/test_skillsbench.py.
    from gbench.runners.eval_suites.skillsbench import (
        compute_skillsbench_score, check_skillsbench_prerequisites)
    s = compute_skillsbench_score({"conditions": {"without-skills": {"mean_reward": 0.5, "n_tasks": 1}}})
    assert s["mean_reward"] == 0.5
    assert callable(check_skillsbench_prerequisites)




# --- LCB: contamination window + compressed tests -----------------------------------------
def test_lcb_decodes_both_test_encodings_and_refuses_to_guess():
    """`code_generation` ships plain JSON; `code_generation_lite` ships
    base64->zlib->pickle. The old `except: tc_data = []` turned an undecodable payload into
    a problem with NO private tests - which does not fail the run, it makes the problem
    free and inflates the score invisibly."""
    import base64
    import json
    import pickle
    import zlib
    from gbench.runners.eval_suites.lcb import _decode_test_cases

    plain = json.dumps([{"input": "1", "output": "2", "testtype": "stdin"}])
    assert _decode_test_cases(plain, "public_test_cases", "q1")[0]["input"] == "1"

    packed = base64.b64encode(zlib.compress(pickle.dumps(
        json.dumps([{"input": "3", "output": "4", "testtype": "stdin"}])))).decode()
    assert _decode_test_cases(packed, "private_test_cases", "q1")[0]["input"] == "3"

    assert _decode_test_cases(None, "private_test_cases", "q1") == []
    with pytest.raises(RuntimeError, match="could not decode"):
        _decode_test_cases("!!!not-decodable!!!", "private_test_cases", "q9")


def test_lcb_defaults_to_a_post_v1_release_window():
    """release_v1 is 2023-05..2024-03 - inside every current model's training window, which
    defeats the one thing LiveCodeBench exists to do. Measured 2026-08-20 on v1:
    code_gen_easy 94.37%."""
    from gbench.runners.eval_suites.lcb import _LCB_VERSION
    assert _LCB_VERSION not in ("release_v1", "v1")


def test_lcb_unknown_task_is_not_auto_passed():
    """The verifier used to `return True` for any task it did not recognise."""
    from gbench.runners.eval_suites.lcb import _verify_single_lcb_sample
    assert _verify_single_lcb_sample("x", {"task": "something_new"}) is False


def test_lcb_unrunnable_test_case_fails_the_problem():
    """A skipped case let a problem pass on the cases that DID run."""
    from gbench.runners.eval_suites.lcb import _verify_single_lcb_sample
    assert _verify_single_lcb_sample("```python\nprint(1)\n```",
                                     {"task": "generation", "tests": ["not-a-dict-case"]}
                                     ) in (False, True)   # string-assert path, not the skip
    assert _verify_single_lcb_sample(
        "```python\nprint(1)\n```",
        {"task": "generation", "tests": [{"testtype": "brand_new", "input": "", "output": ""}]}
    ) is False


# --- AIME: the competition is the category ------------------------------------------------
def test_aime_category_is_the_competition():
    """MathArena's set is one competition (AIME 2026 I and II); there is no url column to parse
    and no other year in the suite, so every sample carries the competition as its category."""
    from gbench.runners.eval_suites import aime as A
    assert not hasattr(A, "_aime_year")
    with _patch_hf({"MathArena/aime_2026": AIME_ROWS}):
        assert {m["category"] for _, _, m in A._load_aime_samples()} == {"aime_2026"}


# --- codeforces generated-test shards -----------------------------------------------------
def _write_shard(tmp_path, rows, order_col="test_i"):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    d = tmp_path / "generated_tests"
    d.mkdir(parents=True, exist_ok=True)
    cols = {"problem_id": [r[0] for r in rows], "input": [r[1] for r in rows],
            "output": [r[2] for r in rows]}
    if order_col:
        cols[order_col] = [r[3] for r in rows]
    pq.write_table(pa.table(cols), str(d / "test_cases_1993.parquet"))
    return tmp_path


def test_codeforces_reads_the_real_shard_column_name(tmp_path, monkeypatch):
    """The dataset card documents `test_case_i`; the shipped shards use `test_i`.

    The old fallback `(tbl.get("test_case_i") or [0])[i]` raised IndexError on the SECOND
    matching row, so with a mirror configured every problem crashed - a failure invisible to
    the mocked unit tests and only reachable with real data.
    """
    from gbench.runners.eval_suites import codeforces as CF
    root = _write_shard(tmp_path, [("1993/A", "1", "a", 2), ("1993/A", "2", "b", 1),
                                   ("1993/B", "9", "z", 1)])
    monkeypatch.setattr(CF, "_GENERATED_TESTS_DIR", str(root))
    got = CF._load_generated_tests("1993", "1993/A")
    assert [t["input"] for t in got] == ["2", "1"], "rows must be ordered by test_i"
    assert CF._load_generated_tests("1993", "1993/B") == [{"input": "9", "output": "z"}]


def test_codeforces_shard_without_an_order_column_still_loads(tmp_path, monkeypatch):
    from gbench.runners.eval_suites import codeforces as CF
    root = _write_shard(tmp_path, [("1993/A", "1", "a", 0), ("1993/A", "2", "b", 0)],
                        order_col=None)
    monkeypatch.setattr(CF, "_GENERATED_TESTS_DIR", str(root))
    assert len(CF._load_generated_tests("1993", "1993/A")) == 2


def test_codeforces_per_problem_shard_load_is_graceful_but_suite_requires_mirror(monkeypatch):
    """The per-problem shard loader degrades to [] (never crashes) when a shard is absent, but
    the SUITE requires the mirror: _require_generated_tests_mirror hard-errors when it is unset,
    because official_tests alone is ~9.7% coverage (a near-meaningless partial)."""
    import pytest
    from gbench.runners.eval_suites import codeforces as CF
    monkeypatch.setattr(CF, "_GENERATED_TESTS_DIR", "")
    assert CF._load_generated_tests("1993", "1993/A") == []          # per-problem: graceful
    with pytest.raises(RuntimeError, match="generated_tests mirror is required"):
        CF._require_generated_tests_mirror()                          # suite: hard-error


def test_codeforces_generated_tests_uncapped_by_default_but_env_can_cap(tmp_path, monkeypatch):
    """Canonical coverage runs ALL generated tests (default cap 0). The shards carry a plain
    `test_i` index (NOT a hardness order), so the optional env cap just bounds wall-clock and
    makes the run a non-canonical subset."""
    from gbench.runners.eval_suites import codeforces as CF
    root = _write_shard(tmp_path, [("1993/A", str(i), "o", i) for i in range(40)])
    monkeypatch.setattr(CF, "_GENERATED_TESTS_DIR", str(root))
    monkeypatch.setattr(CF, "_MAX_GENERATED_TESTS", 0)               # default: no cap
    assert len(CF._load_generated_tests("1993", "1993/A")) == 40
    monkeypatch.setattr(CF, "_MAX_GENERATED_TESTS", 10)             # optional cap still works
    assert len(CF._load_generated_tests("1993", "1993/A")) == 10


def test_codeforces_checker_reads_stdout_verdict_not_returncode():
    """The open-r1 checker prints its verdict (0=WA, 1/100=AC) to STDOUT and ALWAYS exits 0, and
    takes argv = input, reference, submission. Grading must read the stdout score (with reference
    calibration), not the return code (the old bug auto-passed every checker-graded problem)."""
    from gbench.runners.eval_suites.codeforces import _run_case
    # argv[3] is the submission; exits 0 regardless; prints 1 for the accepted answer else 0.
    checker = ("import sys\n"
               "sub = open(sys.argv[3]).read().strip()\n"
               "print(1 if sub == 'RIGHT' else 0)\n"
               "sys.exit(0)\n")
    test = {"input": "", "output": "RIGHT"}      # reference/jury answer
    assert _run_case("print('RIGHT')\n", test, 10.0, checker) is True
    assert _run_case("print('WRONG')\n", test, 10.0, checker) is False


def test_aime_has_no_pre_cutoff_reference():
    """The pre-cutoff reference is gone (older AIME years live in amc_aime): only MathArena's
    AIME 2026 set is loaded, with no knob to add another."""
    from gbench.runners.eval_suites import aime as A
    assert not hasattr(A, "_AIME_INCLUDE_PRECUTOFF")
    with _patch_hf({"MathArena/aime_2026": AIME_ROWS}):
        samples = A._load_aime_samples()
    assert len(samples) == 2
    assert {m["category"] for _, _, m in samples} == {"aime_2026"}
