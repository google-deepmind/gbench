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

"""The Golden Set chat-template leak check, derived from the served model's tokenizer.

The four chat_template cases used to forbid a fixed list of Gemma delimiters, so a template
leak spelled in another family's delimiters passed. They now forbid the served model's own
chat-template tokens (every special token plus every added token that occurs in its chat
template), read at run time. These tests never load a real tokenizer and never touch a model:
the tokenizer loader is stubbed with small fakes shaped like a Gemma 4, a Gemma 3 and a
Llama 3 tokenizer, and the endpoint is the in-process fake from test_golden.
"""

import copy
import tomllib
from pathlib import Path

import pytest
from test_golden import (
    CORRECT_REPLIES,
    GEMMA4_LIKE_TOKENIZER,
    GEMMA4_SPECIAL,
    FakeEndpoint,
    FakeTokenizer,
    by_id,
)

from gbench.core.config import BenchmarkConfig
from gbench.core.models import ModelCategory, ModelConfig
from gbench.runners import golden as golden_mod
from gbench.runners.golden import (
    FORBID_TEMPLATE_TOKENS,
    STATUS_ERROR,
    STATUS_FAILED,
    STATUS_PASSED,
    GoldenBenchmarkRunner,
    chat_template_tokens,
)

REPO = Path(__file__).resolve().parent.parent

TEMPLATE_CASES = ("template_multi_turn", "template_multi_turn_twin",
                  "template_tool_response", "template_tool_response_twin")

# What the cases forbade before the list was derived: the Gemma 4 delimiters, the Gemma 2 and
# 3 turn delimiters, and (tool cases) two plain strings.
OLD_MULTI_TURN = ["<|turn>", "<turn|>", "<start_of_turn>", "<end_of_turn>"]
OLD_TOOL = ["tool_call_id", "<|tool_response>", "<|tool_call>", "<tool_response>",
            "<|turn>", "<turn|>", "<start_of_turn>", "<end_of_turn>"]
OLD_LISTS = {
    "template_multi_turn": OLD_MULTI_TURN, "template_multi_turn_twin": OLD_MULTI_TURN,
    "template_tool_response": OLD_TOOL, "template_tool_response_twin": OLD_TOOL,
}

# Shaped like a Gemma 2/3 tokenizer: the turn delimiters are added tokens its template uses
# (not flagged special here, to exercise that path), and the added vocabulary also holds
# whitespace runs, HTML tags and short strings, some of which occur in the template text.
GEMMA3_SPECIAL = ["<bos>", "<eos>", "<unk>", "<pad>", "<start_of_image>", "<end_of_image>",
                  "<image_soft_token>"]
GEMMA3_LIKE_TOKENIZER = FakeTokenizer(
    special=GEMMA3_SPECIAL,
    added=GEMMA3_SPECIAL + ["<start_of_turn>", "<end_of_turn>", "\n", "\n\n", "  ",
                            "<table>", "<br>", "b", "model"],
    chat_template=(
        "{{ bos_token }}{% for m in messages %}<start_of_turn>{{ role }}\n"
        "{{ m.content | trim }}<end_of_turn>\n{% endfor %}"
        "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
    ),
)
GEMMA3_TOKENS = sorted(GEMMA3_SPECIAL + ["<start_of_turn>", "<end_of_turn>"])

# Shaped like a Llama 3 instruct tokenizer: two special tokens, the header and turn markers as
# added tokens its template uses, and reserved added tokens the template never uses.
LLAMA3_SPECIAL = ["<|begin_of_text|>", "<|eot_id|>"]
LLAMA3_LIKE_TOKENIZER = FakeTokenizer(
    special=LLAMA3_SPECIAL,
    added=["<|begin_of_text|>", "<|end_of_text|>", "<|reserved_special_token_0|>",
           "<|reserved_special_token_1|>", "<|start_header_id|>", "<|end_header_id|>",
           "<|eom_id|>", "<|eot_id|>", "<|python_tag|>"],
    chat_template=(
        "{{- bos_token }}{%- for message in messages %}"
        "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}"
        "{%- if message.role == 'ipython' %}<|python_tag|>{% endif %}"
        "{{- message['content'] | trim + '<|eot_id|>' }}{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}<|eom_id|>"
    ),
)
LLAMA3_TOKENS = sorted(["<|begin_of_text|>", "<|eot_id|>", "<|start_header_id|>",
                        "<|end_header_id|>", "<|python_tag|>", "<|eom_id|>"])
LLAMA_ID = "example-org/llama-like-8b-instruct"


@pytest.fixture(autouse=True)
def no_gcloud(monkeypatch):
    monkeypatch.setenv("VLLM_API_KEY", "test-token")


@pytest.fixture
def config(tmp_path):
    return BenchmarkConfig(results_dir=tmp_path / "results", num_iterations=1)


@pytest.fixture
def tokenizers(monkeypatch):
    """Stub the tokenizer loader with a name -> fake mapping; records every load. A name the
    mapping lacks raises as an unknown repo does."""
    state = {"by_name": {}, "loads": [], "trust": []}

    def _load(name, trust_remote_code=False):
        state["loads"].append(name)
        state["trust"].append(trust_remote_code)
        tok = state["by_name"].get(name)
        if isinstance(tok, BaseException):
            raise tok
        if tok is None:
            raise OSError(f"{name} is not a local folder and is not a valid model identifier")
        return tok

    monkeypatch.setattr("gbench.runners.golden._load_tokenizer", _load)
    return state


def _gemma_model():
    return ModelConfig(short_name="gemma-4-E2B-it", hf_model_id="google/gemma-4-E2B-it",
                       category=ModelCategory.MULTIMODAL, total_params_b=2.0,
                       supports_multimodal=True, supports_audio=True)


def _llama_model():
    return ModelConfig(short_name="llama-like-8b-instruct", hf_model_id=LLAMA_ID,
                       category=ModelCategory.TEXT, total_params_b=8.0)


def _cases(config):
    return {t["id"]: t for t in GoldenBenchmarkRunner(config).load_tasks(list(TEMPLATE_CASES))}


def _old_case(case):
    """The case as it was: the fixed list, no derived tokens."""
    old = copy.deepcopy(case)
    old["golden_truth"].pop(FORBID_TEMPLATE_TOKENS)
    old["golden_truth"]["expected_not_contains"] = list(OLD_LISTS[case["id"]])
    return old


def _status(runner, case, reply, tokens=None):
    return runner.evaluate_task_response(case, reply, template_tokens=tokens)["status"]


# ---------------------------------------------------------------------------------------- #
# The shipped cases
# ---------------------------------------------------------------------------------------- #
def test_the_chat_template_cases_no_longer_name_any_models_delimiters(config):
    cases = _cases(config)
    assert set(cases) == set(TEMPLATE_CASES)
    for case_id, case in cases.items():
        truth = case["golden_truth"]
        assert truth[FORBID_TEMPLATE_TOKENS] is True, case_id
        literal = truth.get("expected_not_contains")
        if case_id.startswith("template_tool_response"):
            # The plain strings a leak of the tool turn shows, checked for every model whether or
            # not its tokenizer has them as tokens.
            assert literal == ["tool_call_id", "<tool_response>"], case_id
        else:
            assert literal is None, case_id
        for delimiter in set(OLD_MULTI_TURN) | {"<|tool_response>", "<|tool_call>"}:
            assert delimiter not in (literal or []), (case_id, delimiter)
        # 1.1 moved every template case to the derived tokens; 1.2 reworded
        # template_tool_response's description of the two plain strings (no verdict change).
        expected = "1.2" if case_id == "template_tool_response" else "1.1"
        assert case["meta"]["version"] == expected, case_id
        assert case["meta"]["changelog"][-1]["version"] == expected, case_id
        assert [e["version"] for e in case["meta"]["changelog"]][1] == "1.1", case_id


def test_the_scoring_version_moved_and_the_fingerprint_was_reviewed():
    assert golden_mod.SCORING_VERSION == 4
    assert golden_mod.scoring_fingerprint() == golden_mod.GOLDEN_SCORING_FP


# ---------------------------------------------------------------------------------------- #
# Gemma: every old verdict stands, except a leaked special token the old list did not name
# ---------------------------------------------------------------------------------------- #
MULTI_TURN_REPLIES = [
    "128",
    "Your second warehouse, in Lima, holds 128 pallets.",
    "You said 512 first, but the second warehouse holds 128.",
    "512",
    "<|turn>model\n128<turn|>",
    "128<turn|>",
    "<|TURN>model 128",                                   # the check ignores case, as before
    "<|channel>thought\n<channel|>128",                   # special tokens not on the old list
    "128<eos>",
    "<|tool_call>call:lookup{}<tool_call|>128",
]
TOOL_REPLIES = [
    "Order ORD-4417 is delayed. The tracking code is QX7R2M.",
    "Order ORD-4417 is delayed.\n\nTracking code: QX7R2M",
    "Order ORD-4417 is on its way.",
    "<|tool_response>response:get_order_status{value:{\"status\": \"delayed\", "
    "\"tracking_code\": \"QX7R2M\"}}<tool_response|>",
    "<|turn>model\nOrder ORD-4417 is delayed. The tracking code is QX7R2M.<turn|>",
    "The tool_call_id was call_ord_4417: delayed, QX7R2M.",
    "<tool_response>delayed QX7R2M</tool_response>",
    "<|tool_call>call:get_order_status{}<tool_call|> delayed QX7R2M",
    "Order ORD-4417 is delayed.<|\"|>QX7R2M<|\"|>",        # special tokens not on the old list
    "<bos>Order ORD-4417 is delayed. The tracking code is QX7R2M.",
]


def test_a_gemma_model_keeps_every_old_verdict_except_a_leaked_special_token_off_the_old_list(
        config):
    runner = GoldenBenchmarkRunner(config)
    tokens = chat_template_tokens(GEMMA4_LIKE_TOKENIZER)["tokens"]
    assert tokens == sorted(GEMMA4_SPECIAL)
    exceptions = []
    for case_id, case in _cases(config).items():
        old_case = _old_case(case)
        replies = TOOL_REPLIES if case_id.startswith("template_tool") else MULTI_TURN_REPLIES
        for reply in replies:
            old = _status(runner, old_case, reply)
            new = _status(runner, case, reply, tokens)
            if old == new:
                continue
            # The only permitted difference: the old list passed a reply that leaks one of the
            # model's special tokens it did not name, and the derived list fails it.
            leaked = [t for t in tokens
                      if t.lower() in reply.lower() and t not in OLD_LISTS[case_id]]
            assert (old, new) == (STATUS_PASSED, STATUS_FAILED), (case_id, reply, old, new)
            assert leaked, (case_id, reply)
            exceptions.append((case_id, reply))
    # The battery does exercise that difference (otherwise it proves nothing).
    assert ("template_multi_turn", "<|channel>thought\n<channel|>128") in exceptions
    assert ("template_tool_response", "Order ORD-4417 is delayed.<|\"|>QX7R2M<|\"|>") in exceptions


def test_the_gemma_leaks_the_old_list_caught_still_fail_and_correct_answers_still_pass(config):
    runner = GoldenBenchmarkRunner(config)
    tokens = chat_template_tokens(GEMMA4_LIKE_TOKENIZER)["tokens"]
    cases = _cases(config)
    assert _status(runner, cases["template_multi_turn"], "128", tokens) == STATUS_PASSED
    leak = runner.evaluate_task_response(cases["template_multi_turn"],
                                         "<|turn>model\n128<turn|>", template_tokens=tokens)
    assert leak["status"] == STATUS_FAILED
    assert "<|turn>" in leak["details"] and "<turn|>" in leak["details"]
    for reply in ("The tool_call_id was call_ord_4417: delayed, QX7R2M.",
                  "<tool_response>delayed QX7R2M</tool_response>"):
        assert _status(runner, cases["template_tool_response"], reply, tokens) == STATUS_FAILED


def test_a_gemma_2_or_3_model_is_checked_against_its_own_turn_delimiters(config):
    """The Gemma 2/3 spelling the old list carried comes from that model's own tokenizer now;
    the whitespace, one-character and plain-word added tokens its template text happens to
    contain are left out, so a multi-line answer that says 'model' still passes."""
    derived = chat_template_tokens(GEMMA3_LIKE_TOKENIZER)
    assert derived["tokens"] == GEMMA3_TOKENS
    assert {e["token"]: e["reason"] for e in derived["excluded"]} == {
        "\n": "whitespace only", "b": "single character", "model": "plain word"}
    runner = GoldenBenchmarkRunner(config)
    cases = _cases(config)
    tokens = derived["tokens"]
    assert _status(runner, cases["template_multi_turn"],
                   "<start_of_turn>model\n128<end_of_turn>", tokens) == STATUS_FAILED
    assert _status(runner, cases["template_tool_response_twin"],
                   "The model says: Order ORD-4417 is delayed.\n\nThe tracking code is QX7R2M.",
                   tokens) == STATUS_PASSED


def test_a_gemma_4_model_is_checked_against_gemma_4_tokens_only(config):
    """The one Gemma verdict that moves the other way: the Gemma 2/3 spelling is not a Gemma 4
    token, so a Gemma 4 model that prints it as text is no longer failed for it."""
    runner = GoldenBenchmarkRunner(config)
    case = _cases(config)["template_multi_turn"]
    tokens = chat_template_tokens(GEMMA4_LIKE_TOKENIZER)["tokens"]
    reply = "<start_of_turn>model\n128<end_of_turn>"
    assert _status(runner, _old_case(case), reply) == STATUS_FAILED
    assert _status(runner, case, reply, tokens) == STATUS_PASSED


# ---------------------------------------------------------------------------------------- #
# Another family: its own delimiters are caught, which the Gemma list never did
# ---------------------------------------------------------------------------------------- #
def test_a_llama_model_leaking_its_own_delimiters_fails_where_the_gemma_list_passed_it(config):
    runner = GoldenBenchmarkRunner(config)
    derived = chat_template_tokens(LLAMA3_LIKE_TOKENIZER)
    assert derived["tokens"] == LLAMA3_TOKENS
    tokens = derived["tokens"]
    cases = _cases(config)
    leak = "<|start_header_id|>assistant<|end_header_id|>\n\n128<|eot_id|>"
    assert _status(runner, _old_case(cases["template_multi_turn"]), leak) == STATUS_PASSED
    result = runner.evaluate_task_response(cases["template_multi_turn"], leak,
                                           template_tokens=tokens)
    assert result["status"] == STATUS_FAILED
    for tok in ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"):
        assert tok in result["details"]
    tool_leak = "<|python_tag|>get_order_status: delayed, QX7R2M<|eom_id|>"
    assert _status(runner, _old_case(cases["template_tool_response"]), tool_leak) == STATUS_PASSED
    assert _status(runner, cases["template_tool_response"], tool_leak, tokens) == STATUS_FAILED
    # Correct answers pass, and the list is this model's own: another family's delimiters are
    # not on it.
    assert _status(runner, cases["template_multi_turn"], "128", tokens) == STATUS_PASSED
    assert _status(runner, cases["template_multi_turn"], "<|turn>model\n128<turn|>",
                   tokens) == STATUS_PASSED
    # Reserved added tokens the template never uses are not forbidden.
    assert "<|reserved_special_token_0|>" not in tokens


def test_a_llama_run_end_to_end_records_the_tokens_it_checked(config, tokenizers):
    tokenizers["by_name"][LLAMA_ID] = LLAMA3_LIKE_TOKENIZER
    replies = dict(CORRECT_REPLIES)
    replies["second warehouse hold"] = "<|start_header_id|>assistant<|end_header_id|>\n\n128"
    config.selected_golden_tasks = list(TEMPLATE_CASES)
    with FakeEndpoint(replies, served_models=(LLAMA_ID,)) as endpoint:
        config.remote_endpoint = endpoint.base_url
        summary = GoldenBenchmarkRunner(config).run(_llama_model())
    results = by_id(summary)
    assert results["template_multi_turn"]["status"] == STATUS_FAILED
    assert results["template_multi_turn_twin"]["status"] == STATUS_FAILED
    assert "<|start_header_id|>" in results["template_multi_turn"]["details"]
    assert results["template_tool_response"]["status"] == STATUS_PASSED
    assert results["template_tool_response_twin"]["status"] == STATUS_PASSED
    assert summary["status"] == STATUS_FAILED and summary["error_cases"] == 0
    check = summary["chat_template_leak_check"]
    assert check["tokenizer"] == LLAMA_ID and check["tokenizer_source"] == "model id"
    assert check["tokens"] == LLAMA3_TOKENS
    assert check["chat_template_found"] is True and "error" not in check
    assert check["tokenizer_class"] == "FakeTokenizer"
    for case_id in TEMPLATE_CASES:
        assert results[case_id]["chat_template_tokens_checked"] == len(LLAMA3_TOKENS)
    assert tokenizers["loads"] == [LLAMA_ID]


# ---------------------------------------------------------------------------------------- #
# No tokenizer: a named harness error, never a pass
# ---------------------------------------------------------------------------------------- #
@pytest.mark.parametrize("failure", [
    OSError("google/gemma-4-E2B-it does not appear to have a file named tokenizer.json"),
    ImportError("No module named 'transformers'"),
])
def test_an_unloadable_tokenizer_makes_the_leak_cases_harness_errors(config, tokenizers,
                                                                      failure):
    tokenizers["by_name"]["google/gemma-4-E2B-it"] = failure
    config.selected_golden_tasks = list(TEMPLATE_CASES) + ["math_canonical"]
    with FakeEndpoint(CORRECT_REPLIES) as endpoint:
        config.remote_endpoint = endpoint.base_url
        summary = GoldenBenchmarkRunner(config).run(_gemma_model())
        prompts = " ".join(str(body.get("messages")) for body in endpoint.requests)
    results = by_id(summary)
    for case_id in TEMPLATE_CASES:
        res = results[case_id]
        assert res["status"] == STATUS_ERROR and res["passed"] is False, case_id
        assert "google/gemma-4-E2B-it" in res["details"]
        assert "--tokenizer" in res["details"]
        assert type(failure).__name__ in res["details"]
    assert results["math_canonical"]["status"] == STATUS_PASSED
    # The cases whose leak check cannot run are never sent.
    assert "warehouse" not in prompts and "ORD-4417" not in prompts
    assert summary["status"] == STATUS_ERROR and summary["accuracy_percent"] is None
    assert summary["error_cases"] == 4 and summary["passed_cases"] == 1
    assert sorted(e.split(":", 1)[0] for e in summary["harness_errors"]) == sorted(TEMPLATE_CASES)
    check = summary["chat_template_leak_check"]
    assert check["tokens"] == [] and "harness error" in check["error"]
    assert tokenizers["loads"] == ["google/gemma-4-E2B-it"]      # tried once, not per case


def test_a_tokenizer_with_nothing_to_check_is_a_harness_error(config, tokenizers):
    tokenizers["by_name"]["google/gemma-4-E2B-it"] = FakeTokenizer(
        special=[], added=["\n", "x"], chat_template="{% for m in messages %}x\n{% endfor %}")
    config.selected_golden_tasks = ["template_multi_turn"]
    with FakeEndpoint(CORRECT_REPLIES) as endpoint:
        config.remote_endpoint = endpoint.base_url
        summary = GoldenBenchmarkRunner(config).run(_gemma_model())
    res = by_id(summary)["template_multi_turn"]
    assert res["status"] == STATUS_ERROR
    assert "no special token" in res["details"]
    assert summary["accuracy_percent"] is None


def test_evaluating_a_leak_case_without_tokens_is_an_error_never_a_pass(config):
    runner = GoldenBenchmarkRunner(config)
    case = _cases(config)["template_multi_turn"]
    for tokens in (None, []):
        result = runner.evaluate_task_response(case, "128", template_tokens=tokens)
        assert result["status"] == STATUS_ERROR and result["passed"] is False
    assert runner.evaluate_task_response(case, "128", template_tokens=["<|x|>"])["passed"]


# ---------------------------------------------------------------------------------------- #
# Which tokenizer, how often, and when not at all
# ---------------------------------------------------------------------------------------- #
def test_the_tokenizer_flag_names_the_tokenizer(config, tokenizers):
    tokenizers["by_name"]["some-org/its-tokenizer"] = GEMMA4_LIKE_TOKENIZER
    config.tokenizer = "some-org/its-tokenizer"
    config.selected_golden_tasks = ["template_multi_turn"]
    with FakeEndpoint(CORRECT_REPLIES) as endpoint:
        config.remote_endpoint = endpoint.base_url
        summary = GoldenBenchmarkRunner(config).run(_gemma_model())
    assert summary["status"] == STATUS_PASSED
    check = summary["chat_template_leak_check"]
    assert (check["tokenizer"], check["tokenizer_source"]) == ("some-org/its-tokenizer",
                                                               "--tokenizer")
    assert tokenizers["loads"] == ["some-org/its-tokenizer"]


def test_a_run_without_a_leak_case_loads_no_tokenizer(config, tokenizers):
    config.selected_golden_tasks = ["math_canonical"]
    with FakeEndpoint(CORRECT_REPLIES) as endpoint:
        config.remote_endpoint = endpoint.base_url
        summary = GoldenBenchmarkRunner(config).run(_gemma_model())
    assert summary["status"] == STATUS_PASSED
    assert summary["chat_template_leak_check"] is None
    assert tokenizers["loads"] == []


def test_one_runner_checks_each_model_against_its_own_tokenizer(config, tokenizers):
    """The CLI builds one Golden Set runner for every model of a run."""
    tokenizers["by_name"]["google/gemma-4-E2B-it"] = GEMMA4_LIKE_TOKENIZER
    tokenizers["by_name"][LLAMA_ID] = LLAMA3_LIKE_TOKENIZER
    config.selected_golden_tasks = ["template_multi_turn", "template_multi_turn_twin"]
    runner = GoldenBenchmarkRunner(config)
    with FakeEndpoint(CORRECT_REPLIES,
                      served_models=("google/gemma-4-E2B-it", LLAMA_ID)) as endpoint:
        config.remote_endpoint = endpoint.base_url
        gemma = runner.run(_gemma_model())
        llama = runner.run(_llama_model())
    assert gemma["chat_template_leak_check"]["tokens"] == sorted(GEMMA4_SPECIAL)
    assert llama["chat_template_leak_check"]["tokens"] == LLAMA3_TOKENS
    assert gemma["status"] == llama["status"] == STATUS_PASSED
    assert tokenizers["loads"] == ["google/gemma-4-E2B-it", LLAMA_ID]


def test_a_local_server_given_its_own_template_is_checked_against_that_template(
        config, tokenizers, tmp_path):
    """--vllm-serve-args '--chat-template t.jinja' replaces the tokenizer's template on the
    server gbench launches, so the added tokens that template uses are the ones forbidden; the
    operator's --trust-remote-code is the trust the tokenizer is loaded with."""
    tokenizers["by_name"]["google/gemma-4-E2B-it"] = FakeTokenizer(
        special=["<s>"], added=["<s>", "[CUSTOM_TURN]", "[UNUSED]"], chat_template="{{ m }}")
    served = tmp_path / "t.jinja"
    served.write_text("{% for m in messages %}[CUSTOM_TURN]{{ m.content }}{% endfor %}")
    config.vllm_serve_args = ["--trust-remote-code", "--chat-template", str(served)]
    config.selected_golden_tasks = ["template_multi_turn"]
    replies = {"second warehouse hold": "[CUSTOM_TURN]128"}
    with FakeEndpoint(replies) as endpoint:
        config.remote_endpoint = None
        config.local_endpoint = endpoint.base_url
        summary = GoldenBenchmarkRunner(config).run(_gemma_model())
    assert by_id(summary)["template_multi_turn"]["status"] == STATUS_FAILED
    check = summary["chat_template_leak_check"]
    assert check["chat_template_source"] == "--chat-template"
    assert check["tokens"] == ["<s>", "[CUSTOM_TURN]"]
    assert tokenizers["trust"] == [True]


def test_a_remote_endpoints_template_is_read_from_its_tokenizer(config, tokenizers):
    tokenizers["by_name"]["google/gemma-4-E2B-it"] = GEMMA4_LIKE_TOKENIZER
    config.vllm_serve_args = ["--chat-template=[CUSTOM_TURN]{{ m }}"]
    config.selected_golden_tasks = ["template_multi_turn"]
    with FakeEndpoint(CORRECT_REPLIES) as endpoint:
        config.remote_endpoint = endpoint.base_url
        summary = GoldenBenchmarkRunner(config).run(_gemma_model())
    check = summary["chat_template_leak_check"]
    assert check["chat_template_source"] == "tokenizer"
    assert check["tokens"] == sorted(GEMMA4_SPECIAL)
    assert tokenizers["trust"] == [False]


def test_the_served_template_is_read_from_either_flag_spelling(tmp_path):
    served = tmp_path / "t.jinja"
    served.write_text("[FILE_TEMPLATE]")
    assert golden_mod._served_chat_template(["--chat-template", str(served)]) == "[FILE_TEMPLATE]"
    assert golden_mod._served_chat_template([f"--chat-template={served}"]) == "[FILE_TEMPLATE]"
    assert golden_mod._served_chat_template(["--chat-template={{ m }}[X]"]) == "{{ m }}[X]"
    assert golden_mod._served_chat_template(["--enforce-eager"]) is None
    assert golden_mod._served_chat_template(None) is None


# ---------------------------------------------------------------------------------------- #
# chat_template_tokens itself
# ---------------------------------------------------------------------------------------- #
def test_named_templates_and_a_missing_template_are_handled():
    named = FakeTokenizer(special=["<s>"], added=["<s>", "[INST]", "[TOOL]", "[UNUSED]"],
                          chat_template={"default": "[INST] {{ m }}", "tool_use": "[TOOL]"})
    assert chat_template_tokens(named)["tokens"] == ["<s>", "[INST]", "[TOOL]"]
    listed = FakeTokenizer(special=[], added=["[INST]", "[TOOL]"],
                           chat_template=[{"name": "default", "template": "[INST]"},
                                          {"name": "tool_use", "template": "[TOOL]"}])
    assert chat_template_tokens(listed)["tokens"] == ["[INST]", "[TOOL]"]
    bare = FakeTokenizer(special=["</s>", "<unk>"], added=["[INST]"], chat_template=None)
    derived = chat_template_tokens(bare)
    assert derived["tokens"] == ["</s>", "<unk>"]
    assert derived["chat_template_found"] is False


# ---------------------------------------------------------------------------------------- #
# The Gemma 4 template conformance check moved out of the unit tests
# ---------------------------------------------------------------------------------------- #
def test_the_gemma4_template_check_lives_in_tools_not_in_the_unit_tests():
    assert not (REPO / "tests" / "test_chat_template_gemma4.py").exists()
    moved = REPO / "tools" / "template_checks" / "test_chat_template_gemma4.py"
    assert moved.is_file()
    assert (REPO / "tools" / "template_checks" / "README.md").is_file()
    # A bare `pytest` collects tests/ only, so the model-artifact check never joins the gate.
    ini = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    assert ini["tool"]["pytest"]["ini_options"]["testpaths"] == ["tests"]
