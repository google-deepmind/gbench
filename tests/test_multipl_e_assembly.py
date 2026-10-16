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
"""Regression: MultiPL-E program assembly for CHAT completions.

gbench used to repair chat answers with its own heuristics (whitespace-insensitive signature
de-duplication, brace balancing against the tests). The pinned MultiPL-E revision ships its own chat
protocol (dataset_builder/chat_completions.py): the model is told to return the whole program with
the prefix verbatim, `postprocess_completion` only drops a Markdown fence and applies the stop
tokens after the prefix length, and the evaluator file carries prompt="". These tests pin that
behaviour: no gbench-side repair, the model's program is what gets executed."""

from gbench.runners.eval_suites.multipl_e import postprocess_completion

_net = lambda s: s.count("{") - s.count("}")


def test_reformatted_signature_is_kept_as_the_model_wrote_it():
    # No declaration matching: a reformatted signature ('foo(nums) {' vs the prompt's 'foo(nums){')
    # is simply part of the model's program.
    code = "function foo(nums) {\n  return 1;\n}"
    assert postprocess_completion([], "function foo(nums){\n", code) == code


def test_signature_not_emitted_twice_when_reformatted():
    prompt = "function foo(nums){\n"
    resp = "```js\nfunction foo(nums) {\n  return nums.length;\n}\n```"
    asm = postprocess_completion([], prompt, resp)
    assert asm.count("function foo") == 1


def test_self_contained_tests_keep_the_functions_close():
    # The model's complete function keeps its closing brace.
    code = "function foo(nums) {\n  return nums.length;\n}"
    tests = "const assert = require('assert');\nassert.equal(foo([1,2]), 2);"
    out = postprocess_completion([], "function foo(nums) {\n", code)
    assert out == code                        # nothing stripped
    assert _net(out + "\n" + tests) == 0      # program brace-balanced


def test_scope_closing_happens_through_stop_tokens_not_brace_balancing():
    # Upstream cuts the text after the prefix length at the first dataset stop token; there is no
    # brace arithmetic. A stop token only matches AFTER the prefix-length head.
    prompt = "object P {\n  def foo(): Int = {\n"
    code = prompt + "    42\n  }\n\n  def main() = {}\n}"
    out = postprocess_completion(["\n  def "], prompt, code)
    assert out == prompt + "    42\n  }\n"


def test_indentation_language_unaffected():
    code = "def foo(n):\n    return n + 1"
    assert postprocess_completion([], "def foo(n):\n", code) == code


def test_never_adds_braces_for_underclosed_code():
    # An under-closed program is a genuine model error; post-processing must NOT invent closers.
    code = "function foo() {\n  return 1;"        # missing '}'
    assert postprocess_completion([], "function foo() {\n", code) == code
