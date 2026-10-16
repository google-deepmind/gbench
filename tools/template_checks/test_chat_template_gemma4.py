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

"""Gemma 4 chat-template conformance checks.

Ported from the vLLM Gemma4 reproduction suite:
https://gist.github.com/lucianommartins/c449b600647d207408a799a42870a561

Only the template half is ported. The rest of that suite drives vLLM
parser internals (``Gemma4Parser``, the reasoning parser, the tool
parsers). Those belong next to the code they test, and porting them
would put a vLLM install on this repo's test path for no benefit here.

These render the template locally with Jinja, so they are fast and
deterministic and never contact a model. But their pass rate is a fact
about a template artifact rather than about gbench, which is the same
reason the Golden Set eval is kept out of CI. So they do not gate
merges: they live in tools/template_checks/, outside the tests/ directory
a plain ``pytest`` collects, and with no template resolved the whole
module skips. Run them from the repository root:

    curl -L -o /tmp/gemma4.jinja \\
      https://huggingface.co/google/gemma-4-31B-it/resolve/main/chat_template.jinja
    GEMMA4_TEMPLATE_PATH=/tmp/gemma4.jinja pytest tools/template_checks

Point ``$GEMMA4_TEMPLATE_PATH`` at a candidate template to check a fix
before it ships.

Calibration
-----------
The reference is ``google-deepmind/dialog``, the training-time
serialization. Of the 48 checks here, 29 assert behaviour the shipped
template already has. The other 19 assert dialog-correct behaviour it
does not, and are ``xfail(strict=True)``: they record the gap now, and
the moment a template closes one it reports an XPASS, so the marker
gets removed instead of quietly rotting into a lie. Every reason string
quotes what the template actually emitted.

Verified against ``google/gemma-4-31B-it@main`` and
``google/gemma-4-E4B-it@main`` on 2026-07-30, which produce identical
results, so the calibration is not specific to one variant.

The upstream suite describes this calibration in prose but ships
without the markers, so running it unchanged is red out of the box.
"""

import os
from pathlib import Path

import pytest

jinja2 = pytest.importorskip("jinja2", reason="Jinja is needed to render the template")
import jinja2.sandbox  # noqa: E402


def _template_path():
    """Resolve the template under test, or None to skip.

    Deliberately never downloads. A test module that reaches out to
    Hugging Face on import turns an unrelated CI run red whenever that
    host has a bad day, and the fetched artifact changes under you
    without a commit to point at.
    """
    env = os.environ.get("GEMMA4_TEMPLATE_PATH")
    if env and Path(env).is_file():
        return Path(env)
    return None


TEMPLATE_PATH = _template_path()

pytestmark = pytest.mark.skipif(
    TEMPLATE_PATH is None,
    reason=(
        "set $GEMMA4_TEMPLATE_PATH to a Gemma 4 chat_template.jinja "
        "(see this module's docstring)"
    ),
)


@pytest.fixture(scope="module")
def gemma4_template():
    env = jinja2.sandbox.ImmutableSandboxedEnvironment()
    return env.from_string(TEMPLATE_PATH.read_text())


def _render(template, messages, **kwargs):
    kwargs.setdefault("bos_token", "<bos>")
    kwargs.setdefault("add_generation_prompt", False)
    return template.render(messages=messages, **kwargs)


def _weather_tool(**params):
    fn = {"name": "get_weather", "description": "Get the weather."}
    if params:
        fn["parameters"] = params
    return {"type": "function", "function": fn}


def _call(name="get_weather", args=None, content="", cid="c1"):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"id": cid, "function": {"name": name, "arguments": args or {}}}
        ],
    }


class TestBaseline_Template:
    """Template behaviors that work correctly."""

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "shipped template emits <|tool_call> before content, which is what this "
            "test's own docstring says is correct -- the assertion contradicts it. "
            "Same divergence as "
            "Test_Matrix_ContentOrdering::test_precall_preamble_position"
        ),
    )
    def test_tool_calls_before_content(self, gemma4_template):
        """Gemma4 docs: tool_calls render BEFORE content."""
        messages = [
            {"role": "user", "content": "Help"},
            {
                "role": "assistant",
                "content": "Here's what I found.",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "lookup",
                            "arguments": {"q": "test"},
                        },
                    }
                ],
            },
        ]
        result = _render(gemma4_template, messages)
        tool_pos = result.find("<|tool_call>")
        content_pos = result.find("Here's what I found.")
        assert content_pos < tool_pos


# --------------------------------------------------------------------------
# Text and turn structure
# --------------------------------------------------------------------------


class Test_Matrix_TextStructure:

    def test_turn_wrapper(self, gemma4_template):
        """✅ <|turn>role\\n…<turn|> wrapper matches dialog."""
        out = _render(gemma4_template, [{"role": "user", "content": "Hi"}])
        assert "<|turn>user\nHi<turn|>" in out

    def test_bos_token(self, gemma4_template):
        """🔵 Template prepends <bos> (dialog never emits BOS) - intentional,
        the tokenizer/serving layer owns BOS."""
        out = _render(gemma4_template, [{"role": "user", "content": "Hi"}])
        assert out.startswith("<bos>")

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "shipped template strips surrounding whitespace ('  Hi  ' -> 'Hi'); "
            "dialog emits content verbatim"
        ),
    )
    def test_content_trimming(self, gemma4_template):
        """dialog emits content verbatim; template strips leading/trailing ws."""
        out = _render(gemma4_template, [{"role": "user", "content": "  Hi  "}])
        assert "<|turn>user\n  Hi  <turn|>" in out

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "shipped template emits one turn per message (2 here); dialog merges "
            "adjacent same-role turns"
        ),
    )
    def test_merge_consecutive_user_system(self, gemma4_template):
        """dialog merges adjacent same-role turns into one."""
        out = _render(
            gemma4_template,
            [
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
            ],
        )
        assert out.count("<|turn>user") == 1

    def test_merge_consecutive_model(self, gemma4_template):
        """✅ template merges consecutive assistant turns (like dialog)."""
        out = _render(
            gemma4_template,
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
                {"role": "assistant", "content": "b"},
            ],
        )
        assert out.count("<|turn>model") == 1


# --------------------------------------------------------------------------
# Content ordering
# --------------------------------------------------------------------------


class Test_Matrix_ContentOrdering:

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "shipped template puts <|tool_call> before the preamble content; dialog "
            "puts the preamble first"
        ),
    )
    def test_precall_preamble_position(self, gemma4_template):
        """dialog: a {content, tool_calls} preamble renders BEFORE <|tool_call>."""
        out = _render(
            gemma4_template,
            [
                _call(args={"city": "Paris"}, content="Let me check."),
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
            ],
            add_generation_prompt=True,
        )
        assert out.find("Let me check.") < out.find("<|tool_call>")

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "same call-before-preamble ordering as test_precall_preamble_position, "
            "per round"
        ),
    )
    def test_interleave_text_thinking_tools(self, gemma4_template):
        """dialog preserves author order: each round's preamble precedes its call."""
        out = _render(
            gemma4_template,
            [
                _call(args={"city": "NY"}, content="Let me check NY.", cid="c1"),
                {"role": "tool", "tool_call_id": "c1", "content": "cloudy"},
                _call(args={"city": "London"}, content="Now let me check London.", cid="c2"),
                {"role": "tool", "tool_call_id": "c2", "content": "rainy"},
            ],
            add_generation_prompt=True,
        )
        second_call = out.find("<|tool_call>", out.find("<|tool_call>") + 1)
        assert out.find("Now let me check London.") < second_call


# --------------------------------------------------------------------------
# Tool calls
# --------------------------------------------------------------------------


class Test_Matrix_ToolCalls:

    def test_toolcall_shape_and_no_escaping(self, gemma4_template):
        """✅ call:name{…} shape, <|"|> string sentinel, no escaping."""
        out = _render(gemma4_template, [_call(args={"note": 'say "hi"'})])
        assert "<|tool_call>call:get_weather{" in out
        assert 'note:<|"|>say "hi"<|"|>' in out  # internal quote passed through raw

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "shipped template alphabetizes argument keys via dictsort "
            "({apple:2,mango:3,zebra:1}); dialog keeps insertion order"
        ),
    )
    def test_arg_key_order(self, gemma4_template):
        """Tool-call argument keys keep insertion order (dialog), not
        alphabetized by `| dictsort`."""
        out = _render(gemma4_template, [_call(args={"zebra": 1, "apple": 2, "mango": 3})])
        assert "{zebra:1,apple:2,mango:3}" in out

    def test_none_arg_renders_null(self, gemma4_template):
        """🟢 FIXED by PR: None -> null (was the literal 'None')."""
        out = _render(gemma4_template, [_call(args={"k": None})])
        assert "k:null" in out
        assert "k:None" not in out

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "shipped template renders 1.0 as 'x:1.0'; dialog narrows a whole-number "
            "float to an int"
        ),
    )
    def test_whole_number_float(self, gemma4_template):
        """A whole-number float renders as an int (1.0 -> 1), matching dialog."""
        out = _render(gemma4_template, [_call(args={"x": 1.0})])
        assert "call:get_weather{x:1}<tool_call|>" in out

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "shipped template renders 1e20 as 'x:1e+20'; dialog writes the full "
            "digits"
        ),
    )
    def test_large_float(self, gemma4_template):
        """A large whole float renders as full digits (not 1e+20), matching dialog."""
        out = _render(gemma4_template, [_call(args={"x": 1e20})])
        assert "100000000000000000000" in out

    def test_string_valued_arguments_raise(self, gemma4_template):
        """🔵 stricter (intended): string arguments raise (must be a dict)."""
        with pytest.raises(Exception):
            _render(
                gemma4_template,
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "f", "arguments": '{"k": "v"}'}}
                        ],
                    }
                ],
            )


# --------------------------------------------------------------------------
# Tool responses
# --------------------------------------------------------------------------


class Test_Matrix_ToolResponses:

    def test_structured_tool_responses_ext(self, gemma4_template):
        """✅ native tool_responses extension expands structured content."""
        out = _render(
            gemma4_template,
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_responses": [
                        {"name": "get_weather", "response": {"temp": 21}}
                    ],
                }
            ],
        )
        assert "response:get_weather{temp:21}" in out

    def test_standard_openai_tool_string(self, gemma4_template):
        """⚪ format limitation: a standard role:tool STRING result is wrapped
        under a synthetic value: key. dialog would expand structuredContent
        into fields - unreachable from an opaque OpenAI string."""
        out = _render(
            gemma4_template,
            [
                _call(args={"city": "Paris"}),
                {"role": "tool", "tool_call_id": "c1", "content": '{"temp":21}'},
            ],
            add_generation_prompt=True,
        )
        assert 'value:<|"|>' in out  # value-wrapped, not structured

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "shipped template alphabetizes response keys ({apple:2,zebra:1}); "
            "dialog keeps insertion order"
        ),
    )
    def test_response_key_order(self, gemma4_template):
        """Tool-response keys keep insertion order (dialog), not alphabetized."""
        out = _render(
            gemma4_template,
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_responses": [
                        {"name": "get_weather", "response": {"zebra": 1, "apple": 2}}
                    ],
                }
            ],
        )
        assert "{zebra:1,apple:2}" in out


# --------------------------------------------------------------------------
# Tool definitions
# --------------------------------------------------------------------------


class Test_Matrix_ToolDefinitions:

    def test_declaration_alphabetized(self, gemma4_template):
        """✅ declaration:name{…}; keys alphabetized (both dialog & template)."""
        out = _render(
            gemma4_template,
            [{"role": "user", "content": "hi"}],
            tools=[_weather_tool(type="object", properties={"city": {"type": "string"}})],
        )
        assert "<|tool>declaration:get_weather{" in out

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "shipped template emits the Python repr type:<|\"|>['STRING', "
            "'NULL']<|\"|> instead of a token list"
        ),
    )
    def test_union_nullable_type(self, gemma4_template):
        """Union/nullable type ['string','null'] renders as a token list, not a Python repr."""
        out = _render(
            gemma4_template,
            [{"role": "user", "content": "hi"}],
            tools=[_weather_tool(type="object", properties={"x": {"type": ["string", "null"]}})],
        )
        assert 'type:[<|"|>STRING<|"|>,<|"|>NULL<|"|>]' in out

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "shipped template omits the parameters block entirely for an empty "
            "schema and emits a bare empty description"
        ),
    )
    def test_empty_input_schema(self, gemma4_template):
        """Empty inputSchema {} renders as parameters:{} (not a spurious empty description)."""
        out = _render(
            gemma4_template,
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "noop", "parameters": {}}}],
        )
        assert "parameters:{}" in out

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "shipped template drops response schema properties, leaving "
            "response:{type:<|\"|>OBJECT<|\"|>}"
        ),
    )
    def test_output_schema_props_required(self, gemma4_template):
        """outputSchema/response properties + required are rendered (not dropped)."""
        out = _render(
            gemma4_template,
            [{"role": "user", "content": "hi"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                    "response": {
                        "type": "object",
                        "properties": {"temperature": {"type": "number"}},
                        "required": ["temperature"],
                    },
                },
            }],
        )
        assert "temperature" in out

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "shipped template emits response:{description:<|\"|>the text<|\"|>,} -- "
            "type missing and a dangling comma"
        ),
    )
    def test_non_object_response_type(self, gemma4_template):
        """A non-OBJECT response type is rendered with no dangling comma."""
        out = _render(
            gemma4_template,
            [{"role": "user", "content": "hi"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                    "response": {"type": "string", "description": "the text"},
                },
            }],
        )
        # anchored on the response block (not the params' string type)
        assert 'the text<|"|>,type:<|"|>STRING<|"|>}' in out

    def test_string_enum_kept(self, gemma4_template):
        """✅ string enums are preserved."""
        out = _render(
            gemma4_template,
            [{"role": "user", "content": "hi"}],
            tools=[_weather_tool(
                type="object",
                properties={"u": {"type": "string", "enum": ["a", "b"]}})],
        )
        assert 'enum:[<|"|>a<|"|>,<|"|>b<|"|>]' in out

    @pytest.mark.xfail(
        strict=True,
        reason="shipped template drops enum for non-string property types",
    )
    def test_enum_on_non_string(self, gemma4_template):
        """enum is kept for non-string property types (integer/number/etc.)."""
        out = _render(
            gemma4_template,
            [{"role": "user", "content": "hi"}],
            tools=[_weather_tool(
                type="object",
                properties={"n": {"type": "integer", "enum": [1, 2, 3]}})],
        )
        assert "enum:[1,2,3]" in out

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "shipped template drops format/minimum/pattern/default from property "
            "schemas"
        ),
    )
    def test_additional_schema_keys(self, gemma4_template):
        """additionalProperties/format/minimum/pattern/default pass through (not dropped)."""
        out = _render(
            gemma4_template,
            [{"role": "user", "content": "hi"}],
            tools=[_weather_tool(
                type="object",
                properties={"x": {"type": "string", "format": "email"}})],
        )
        assert 'format:<|"|>email<|"|>' in out


# --------------------------------------------------------------------------
# Thinking channel
# --------------------------------------------------------------------------


class Test_Matrix_Thinking:

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "shipped template emits a newline after <|think|>, "
            "dialog has none"
        ),
    )
    def test_think_enable_token(self, gemma4_template):
        """<|think|> is emitted with no trailing newline, matching dialog."""
        out = _render(
            gemma4_template,
            [{"role": "system", "content": "You are helpful"}],
            enable_thinking=True,
        )
        assert "<|think|>\n" not in out  # dialog has no newline after the control token

    def test_reasoning_on_final_answer(self, gemma4_template):
        """🟢 FIXED by PR: reasoning on the final (no-tool_calls) answer is kept."""
        out = _render(
            gemma4_template,
            [
                {"role": "user", "content": "Weather in Tokyo?"},
                _call(args={"city": "Tokyo"}, content=""),
                {"role": "tool", "tool_call_id": "c1", "content": "30C"},
                {"role": "assistant", "content": "It's 30C.",
                 "reasoning_content": "Tool returned 30C"},
            ],
            add_generation_prompt=True,
            enable_thinking=True,
            tools=[_weather_tool(type="object", properties={"city": {"type": "string"}})],
        )
        assert "Tool returned 30C" in out

    def test_gen_prompt_primer(self, gemma4_template):
        """🔵 intentional: gen prompt (thinking off) is one of the two sanctioned
        forms - bare `<|turn>model\\n` (E2B) or with the thought primer (large)."""
        out = _render(
            gemma4_template,
            [{"role": "user", "content": "hi"}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        assert out.endswith("<|turn>model\n") or out.endswith(
            "<|turn>model\n<|channel>thought\n<channel|>"
        )

    def test_post_tool_response_thinking_prime(self, gemma4_template):
        """🔵 intentional (added by PR): after a tool_response with thinking on,
        the gen prompt primes an open thought channel."""
        out = _render(
            gemma4_template,
            [
                _call(args={"city": "Paris"}),
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
            ],
            add_generation_prompt=True,
            enable_thinking=True,
        )
        assert out.endswith("<|channel>thought\n")


class Test_Template_ThinkingChannelWithoutToolCalls:

    def test_reasoning_on_final_answer_preserved(self, gemma4_template):
        """After a tool chain, the final assistant with reasoning but
        no tool_calls should have its reasoning in the rendered output."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "d",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        messages = [
            {"role": "user", "content": "Weather in Tokyo?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I need to check the weather",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Tokyo"},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "30C"},
            {
                "role": "assistant",
                "content": "It's 30C in Tokyo.",
                "reasoning_content": "Tool returned 30C, I can answer",
            },
        ]
        result = _render(
            gemma4_template,
            messages,
            add_generation_prompt=True,
            enable_thinking=True,
            tools=tools,
        )
        assert "I need to check the weather" in result
        assert "Tool returned 30C" in result


# --------------------------------------------------------------------------
# Multimodal
# --------------------------------------------------------------------------


class Test_Matrix_Multimodal:

    def test_image_audio_placeholders(self, gemma4_template):
        """✅ <|image|> / <|audio|> placeholders."""
        img = _render(
            gemma4_template,
            [{"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image"}]}],
        )
        assert "<|image|>" in img
        aud = _render(
            gemma4_template,
            [{"role": "user", "content": [{"type": "audio"}]}],
        )
        assert "<|audio|>" in aud

    def test_image_url_input_audio_aliases(self, gemma4_template):
        """🟢 FIXED by PR: OpenAI content-part aliases image_url / input_audio."""
        out = _render(
            gemma4_template,
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}],
        )
        assert "<|image|>" in out

    def test_video_placeholder(self, gemma4_template):
        """🔴/⚪ edge: template emits <|video|>; dialog has no Video and rejects
        the token. Documented; a normal client only hits this with a video part."""
        out = _render(
            gemma4_template,
            [{"role": "user", "content": [{"type": "video"}]}],
        )
        assert "<|video|>" in out


# --------------------------------------------------------------------------
# History and robustness
# --------------------------------------------------------------------------


class Test_Matrix_HistoryRobustness:

    def test_tool_response_inline(self, gemma4_template):
        """✅ tool response stays inline in the model turn (no extra <|turn>)."""
        out = _render(
            gemma4_template,
            [
                _call(args={"city": "London"}),
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
            ],
            add_generation_prompt=True,
        )
        assert out.count("<|turn>model") == 1

    def test_turn_closure(self, gemma4_template):
        """🟢 turn-closure fix: a conversation ending in an assistant answer
        after a tool chain closes cleanly with <turn|>."""
        out = _render(
            gemma4_template,
            [
                {"role": "user", "content": "Weather?"},
                _call(args={"city": "Paris"}, content="Checking."),
                {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
                {"role": "assistant", "content": "It's sunny in Paris."},
            ],
        )
        assert out.rstrip().endswith("<turn|>")
        assert out.count("<|turn>model") == 1

    def test_continuation_prefill(self, gemma4_template):
        """⚪ format limitation: a trailing assistant turn is CLOSED (and a new
        model turn appended with add_generation_prompt) - dialog would leave it
        open for prefill/continuation. Documented current behavior."""
        out = _render(
            gemma4_template,
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "partial"},
            ],
            add_generation_prompt=True,
        )
        assert "partial<turn|>" in out  # closed, not left open

    def test_get_guards_and_defaults(self, gemma4_template):
        """🟢 robustness: empty messages / missing keys don't crash (PR hardened
        accessors + guarded messages[0])."""
        assert isinstance(_render(gemma4_template, []), str)
        assert isinstance(
            _render(gemma4_template, [{"role": "user"}]), str
        )


# --------------------------------------------------------------------------
# Fixes from the HF template PR
# --------------------------------------------------------------------------


class Test_HF_lucianommartins_PR_NoneRendering:

    def test_none_renders_as_null(self, gemma4_template):
        messages = [
            {"role": "user", "content": "Test"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "set_value",
                            "arguments": {"key": "test", "value": None},
                        },
                    }
                ],
            },
        ]
        result = _render(gemma4_template, messages)
        assert "value:null" in result
        assert "value:None" not in result


class Test_HF_lucianommartins_PR_StringArgumentsValidation:

    def test_string_arguments_raises_error(self, gemma4_template):
        messages = [
            {"role": "user", "content": "Test"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "fn",
                            "arguments": '{"key": "value"}',
                        },
                    }
                ],
            },
        ]
        with pytest.raises(Exception):
            _render(gemma4_template, messages)


class Test_HF_lucianommartins_PR_NoExtraTurnInToolChain:

    def test_no_extra_turn_model_after_tool_response(self, gemma4_template):
        messages = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "London"},
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "sunny, 15C",
            },
        ]
        result = _render(
            gemma4_template, messages, add_generation_prompt=True
        )
        assert result.count("<|turn>model") == 1


# --------------------------------------------------------------------------
# Ported from vLLM CI
# --------------------------------------------------------------------------


class Test_CI_ChatTemplate:

    def test_reasoning_in_tool_chains(self, gemma4_template):
        """reasoning on an assistant WITH tool_calls (after last user) emits
        <|channel>thought\\n...<channel|>."""
        messages = [
            {"role": "user", "content": "Calculate something"},
            {
                "role": "assistant",
                "content": "",
                "reasoning": "Let me think about this...",
                "tool_calls": [
                    {"function": {"name": "calculator", "arguments": {"expr": "2+2"}}}
                ],
            },
        ]
        result = _render(gemma4_template, messages)
        assert "<|channel>thought\n" in result
        assert "Let me think about this..." in result
        assert "<channel|>" in result

    def test_reasoning_not_before_last_user(self, gemma4_template):
        """reasoning on an assistant BEFORE the last user message is dropped
        (thinking_gate)."""
        messages = [
            {"role": "user", "content": "First"},
            {
                "role": "assistant",
                "content": "Response",
                "reasoning": "Old reasoning that should be dropped",
                "tool_calls": [{"function": {"name": "fn", "arguments": {}}}],
            },
            {"role": "user", "content": "Second"},
        ]
        result = _render(gemma4_template, messages, add_generation_prompt=True)
        assert "Old reasoning" not in result

    def test_strip_thinking_in_model_content(self, gemma4_template):
        """<|channel>...<channel|> inline in model content is stripped."""
        messages = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "<|channel>internal thought<channel|>Visible answer",
            },
        ]
        result = _render(gemma4_template, messages)
        assert "internal thought" not in result
        assert "Visible answer" in result

    def test_multi_turn_tool_chain_one_model_turn(self, gemma4_template):
        """assistant->tool->assistant->tool produces exactly one <|turn>model
        (later assistants continue the same turn)."""
        messages = [
            {"role": "user", "content": "Do two things"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c1", "function": {"name": "step1", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "result1"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "c2", "function": {"name": "step2", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "c2", "content": "result2"},
        ]
        result = _render(gemma4_template, messages, add_generation_prompt=True)
        assert result.count("<|turn>model\n") == 1

    def test_format_argument_types(self, gemma4_template):
        """Strings wrapped in <|"|>, booleans as true/false, integers bare."""
        messages = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "test_fn", "arguments": {
                 "name": "Alice", "active": True, "count": 42}}}]},
        ]
        result = _render(gemma4_template, messages)
        assert '<|"|>Alice<|"|>' in result
        assert "active:true" in result
        assert "count:42" in result


# --------------------------------------------------------------------------
# Residual dialog parity
# --------------------------------------------------------------------------


class Test_Residual_DialogParity:

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "shipped template does not resolve $defs refs, collapsing the nested "
            "schema to user:{type:<|\"|><|\"|>}"
        ),
    )
    def test_tooldef_defs_type_not_uppercased(self, gemma4_template):
        """dialog only uppercases `type` within its visited key-set, so `type`
        inside $defs stays lowercase; the shipped template uppercases it."""
        tool = {
            "type": "function",
            "function": {
                "name": "fn",
                "description": "d",
                "parameters": {
                    "type": "object",
                    "properties": {"user": {"$ref": "#/$defs/User"}},
                    "$defs": {
                        "User": {
                            "type": "object",
                            "properties": {"age": {"type": "integer"}},
                        }
                    },
                },
            },
        }
        out = _render(gemma4_template, [{"role": "user", "content": "x"}], tools=[tool])
        assert '{age:{type:<|"|>integer<|"|>}}' in out

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "shipped template emits one turn per leading system message (2 here); "
            "dialog merges them"
        ),
    )
    def test_leading_system_messages_merge(self, gemma4_template):
        """dialog merges consecutive same-role turns; shipped template emits the
        first system in the header block and a second system as its own turn."""
        out = _render(gemma4_template, [{"role": "system", "content": "s1"},
                                        {"role": "system", "content": "s2"}])
        assert out.count("<|turn>system") == 1

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "shipped template trims and appends a trailing space to system content "
            "parts ('hello ')"
        ),
    )
    def test_system_content_parts_verbatim(self, gemma4_template):
        """dialog renders content verbatim; shipped top-block trims + appends a
        trailing space."""
        out = _render(gemma4_template, [{"role": "system",
                      "content": [{"type": "text", "text": "hello"}]}])
        assert "<|turn>system\nhello<turn|>" in out
