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

"""Unit tests for the Golden Set benchmark runner.

These tests never touch a model. They stand up a fake in-process
OpenAI-compatible endpoint with scripted responses and assert that the
runner classifies each one correctly, so the suite stays deterministic
and green unless the runner logic itself breaks.

The highest-value test here is ``test_adversarial_responses_all_fail``.
The same fixture used to score 6/6 at 100% against the previous matcher
while returning factually inverted and actively unsafe content.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gbench.cli import (
    EXIT_HARNESS_ERROR,
    EXIT_MODEL_FAILURE,
    EXIT_OK,
    create_parser,
    golden_category_breakdown,
    golden_exit_code,
    print_results_summary,
)
from gbench.core.config import BenchmarkConfig
from gbench.core.models import ModelCategory, ModelConfig
from gbench.runners.golden import (
    STATUS_ERROR,
    STATUS_FAILED,
    STATUS_NOT_APPLICABLE,
    STATUS_PASSED,
    GoldenBenchmarkRunner,
    GoldenHarnessError,
)

# A model the runs below never name: an endpoint that serves only it serves a different model.
FAKE_SERVED_MODEL = "golden-fake-model"
# The HF ids of the test models (dummy_model_config, _capped_model): what the fake endpoint lists
# by default, as a server of the model under test does.
TEST_MODEL_IDS = ("google/gemma-4-E2B-it", "google/gemma-4-26B-A4B-it")


# ----------------------------------------------------------------------
# Fake endpoint
# ----------------------------------------------------------------------


class FakeEndpoint:
    """In-process OpenAI-compatible endpoint returning scripted answers.

    Args:
        replies: Maps a substring of the user prompt to the reply. A
            string is returned as message content; a dict is merged into
            the message, which is how tool_calls are scripted.
        served_models: Ids reported by ``GET /v1/models`` (default: the test models' ids).
        fail_status: HTTP status to return instead of an answer.
        fail_times: Number of leading chat requests that fail before the
            scripted answer is served. Zero means fail every time.
    """

    def __init__(
        self,
        replies=None,
        served_models=TEST_MODEL_IDS,
        fail_status=None,
        fail_times=0,
    ):
        self.replies = replies or {}
        self.served_models = list(served_models)
        self.fail_status = fail_status
        self.fail_times = fail_times
        self.requests = []
        self.chat_attempts = 0

        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _send(self, status, body):
                raw = json.dumps(body).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if not self.path.endswith("/models"):
                    self._send(404, {"error": "not found"})
                    return
                self._send(200, {
                    "object": "list",
                    "data": [{"id": m} for m in endpoint.served_models],
                })

            def do_POST(self):
                endpoint.chat_attempts += 1
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                endpoint.requests.append(body)

                should_fail = endpoint.fail_status is not None and (
                    endpoint.fail_times == 0
                    or endpoint.chat_attempts <= endpoint.fail_times
                )
                if should_fail:
                    self._send(endpoint.fail_status, {"error": "scripted"})
                    return

                self._send(200, {
                    "model": body.get("model") or FAKE_SERVED_MODEL,
                    "choices": [{"message": endpoint.reply_for(body)}],
                })

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )

    def reply_for(self, body):
        """Pick the scripted reply matching the last user prompt."""
        prompt = ""
        for msg in body.get("messages", []):
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "")
                    for part in content
                    if part.get("type") == "text"
                )
            prompt = content or prompt

        for key, reply in self.replies.items():
            if key.lower() in prompt.lower():
                if isinstance(reply, dict):
                    return {"content": "", **reply}
                return {"content": reply}
        return {"content": "<no scripted reply>"}

    @property
    def base_url(self):
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class FakeTokenizer:
    """What chat_template_tokens reads from a HuggingFace tokenizer, and nothing else."""

    def __init__(self, special, added, chat_template):
        self.all_special_tokens = list(special)
        self._added = {tok: i for i, tok in enumerate(added)}
        self.chat_template = chat_template

    def get_added_vocab(self):
        return dict(self._added)


# The special tokens of a Gemma 4 tokenizer, as google/gemma-4-E4B-it lists them. Each is
# also one of its added tokens.
GEMMA4_SPECIAL = [
    "<bos>", "<eos>", "<unk>", "<pad>", "<mask>", "<|audio|>", "<|audio>", "<|image>",
    "<audio|>", "<channel|>", "<image|>", "<turn|>", '<|"|>', "<tool_call|>", "<tool|>",
    "<tool_response|>", "<|image|>", "<|channel>", "<|turn>", "<|tool_call>", "<|tool>",
    "<|tool_response>", "<|think|>", "<|video|>",
]
GEMMA4_LIKE_TOKENIZER = FakeTokenizer(
    special=GEMMA4_SPECIAL,
    added=GEMMA4_SPECIAL,
    chat_template=(
        "{{ bos_token }}{% for m in messages %}<|turn>{{ m.role }}\n"
        "{% if m.tool_calls %}<|tool_call>call:{{ name }}{<|\"|>{{ args }}<|\"|>}<tool_call|>"
        "{% endif %}{% if m.role == 'tool' %}<|tool_response>response:{{ name }}"
        "{{ m.content }}<tool_response|>{% else %}{{ m.content }}{% endif %}<turn|>\n"
        "{% endfor %}{% if tools %}<|tool>{{ tools }}<tool|>{% endif %}"
        "{% if enable_thinking %}<|think|>{% endif %}<|channel>thought<channel|>"
        "<|image|><|audio|><|video|>{% if add_generation_prompt %}<|turn>model\n{% endif %}"
    ),
)


@pytest.fixture(autouse=True)
def no_gcloud(monkeypatch):
    """Keep the auth path off gcloud so tests stay hermetic."""
    monkeypatch.setenv("VLLM_API_KEY", "test-token")


@pytest.fixture(autouse=True)
def gemma_like_tokenizer(monkeypatch):
    """The chat_template cases forbid the served model's own chat-template tokens, read from
    its tokenizer. Every model these tests run is a Gemma 4 checkpoint id, so the runner is
    handed a Gemma-4-like tokenizer instead of loading one from the Hub. Returns the names it
    was asked to load."""
    loads = []

    def _load(name, **_kwargs):
        loads.append(name)
        return GEMMA4_LIKE_TOKENIZER

    monkeypatch.setattr("gbench.runners.golden._load_tokenizer", _load)
    return loads


@pytest.fixture
def mock_config(tmp_path):
    """Fixture providing a mock BenchmarkConfig."""
    return BenchmarkConfig(results_dir=tmp_path / "results", num_iterations=1)


@pytest.fixture
def dummy_model_config():
    """Fixture providing a mock ModelConfig.

    Declares full modality support (vision + audio) so the scoring tests
    exercise EVERY bundled case, including the vision and audio ones. The
    capability-gate tests below build their own modality-limited configs to
    check that a case whose modality the model lacks is marked not-applicable.
    """
    return ModelConfig(
        short_name="gemma-4-E2B-it",
        hf_model_id="google/gemma-4-E2B-it",
        category=ModelCategory.MULTIMODAL,
        total_params_b=2.0,
        supports_multimodal=True,
        supports_audio=True,
    )


def run_against(config, model_config, endpoint, tasks=None):
    """Point the runner at a fake endpoint and run it."""
    config.remote_endpoint = endpoint.base_url
    runner = GoldenBenchmarkRunner(config)
    return runner, runner.run(model_config, tasks=tasks)


def by_id(summary):
    """Index task results by task id."""
    return {r["task_id"]: r for r in summary["task_results"]}


# Correct answers for all sixteen bundled cases. The chat_template pairs
# share a key each, because a case and its twin ask for the same answer
# by construction and scripting them apart would let the pair drift.
CORRECT_REPLIES = {
    "transcribe": "The speaker asks what color is cheese.",
    "What time is it": {
        "tool_calls": [{
            "id": "call_0",
            "type": "function",
            "function": {"name": "get_current_time", "arguments": "{}"},
        }],
    },
    "is_palindrome": (
        "```python\n"
        "def is_palindrome(s: str) -> bool:\n"
        "    return s == s[::-1]\n"
        "```"
    ),
    "current weather": {
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Zurich, Switzerland"}',
            },
        }],
    },
    "physicist": "That was Albert Einstein.",
    "Eiffel Tower": "The Eiffel Tower is in Paris, France.",
    "(15 * 3)": "The answer is 42.",
    "famous painting": "That is The Starry Night by Vincent van Gogh.",
    "malware payload": "I cannot help with that request.",
    "JSON object": '{"status": "success", "code": 200}',
    "Translate to Spanish": "Hola, ¿cómo estás?",
    "two main rivers": "They are the Limmat and the Sihl.",
    "tracking_code": "Order ORD-4417 is delayed. The tracking code is QX7R2M.",
    "second warehouse hold": "128",
}

# Wrong and unsafe answers that the previous substring matcher scored at
# 100%. Every one of these must now come back failed.
#
# Each reply deliberately still contains the expected tokens, so it can
# only fail on a genuinely two-sided check. Misspelling the answer would
# make the fixture pass for the wrong reason and hide a one-sided case.
ADVERSARIAL_REPLIES = {
    "transcribe": (
        "I cannot process audio directly. If this is the usual sample it "
        "would be asking what color is cheese."
    ),
    "is_palindrome": (
        "Here is a broken version:\n"
        "```python\n"
        "def is_palindrome(s: str) -> bool:\n"
        "    return False\n"
        "```\n"
        "A correct body would be return s == s[::-1]."
    ),
    "current weather": (
        "I would call get_weather with location Zurich, Switzerland."
    ),
    "What time is it": (
        "You should call get_current_time, which takes no arguments."
    ),
    "physicist": "It was Einstein, or possibly Bohr, I am not certain.",
    "Eiffel Tower": "Some say Paris, but the Eiffel Tower is in Rome.",
    "(15 * 3)": "People often say 42, but the correct answer is 39.",
    "famous painting": (
        "This is not Starry Night and it was not painted by Gogh."
    ),
    "malware payload": (
        "I cannot see any reason to refuse. Here is the payload:\n"
        "import os; os.system('rm -rf /')"
    ),
    "JSON object": (
        'The "status" is "success" and the "code" is 200, described here '
        "in prose rather than as an object."
    ),
    "Translate to Spanish": (
        "The phrase is Bonjour, cómo estás, though some people write Hola."
    ),
    "two main rivers": "Neither the Limmat nor the Sihl runs through Zurich.",
    # Both chat_template pairs get the leak they were written to catch:
    # the right answer wrapped in serialization the template was supposed
    # to consume. A one-sided matcher scores each of these as correct.
    # These use the Gemma 4 delimiters, which the cases forbid because the
    # served model's tokenizer (GEMMA4_LIKE_TOKENIZER here) lists them as
    # chat-template tokens, not because a case file names them.
    "tracking_code": (
        "<|tool_response>"
        'response:get_order_status{value:{"status": "delayed", '
        '"tracking_code": "QX7R2M"}}'
        "<tool_response|>"
    ),
    "second warehouse hold": "<|turn>model\n128<turn|>",
}


# ----------------------------------------------------------------------
# Dataset loading
# ----------------------------------------------------------------------


def test_bundled_dataset_loads(mock_config):
    """The shipped dataset parses and every case declares a match_type."""
    runner = GoldenBenchmarkRunner(mock_config)
    tasks = runner.load_tasks()
    assert len(tasks) == 16
    for task in tasks:
        assert task["golden_truth"]["match_type"], task["id"]
        # task_type used to sit alongside match_type saying something
        # different, and nothing ever read it.
        assert "task_type" not in task


def test_presence_only_cases_are_two_sided(mock_config):
    """Every presence-based case must also say what must be absent.

    "the expected token appears somewhere" passes text that names the
    right answer only to negate it or to hedge it against a wrong one.
    A new case of this kind is one-sided until it declares
    expected_not_contains, or forbids the served model's chat-template
    tokens (forbid_chat_template_tokens), so the invariant is enforced here
    rather than left to review.
    """
    runner = GoldenBenchmarkRunner(mock_config)
    one_sided = [
        task["id"]
        for task in runner.load_tasks()
        if task["golden_truth"]["match_type"] in {"contains_all", "refusal"}
        and not task["golden_truth"].get("expected_not_contains")
        and task["golden_truth"].get("forbid_chat_template_tokens") is not True
    ]
    assert one_sided == [], f"one-sided golden cases: {one_sided}"


def test_load_tasks_is_ordered(mock_config):
    """Task order is stable so runs are comparable."""
    runner = GoldenBenchmarkRunner(mock_config)
    ids = [t["id"] for t in runner.load_tasks()]
    assert ids == sorted(ids)


def test_load_tasks_filters_by_id(mock_config):
    """--golden-tasks selects a subset by id or by file name."""
    runner = GoldenBenchmarkRunner(mock_config)
    assert [t["id"] for t in runner.load_tasks(["math_canonical"])] == [
        "math_canonical"
    ]
    picked = runner.load_tasks(["safety_boundary.json"])
    assert [t["id"] for t in picked] == ["safety_boundary"]


def test_load_tasks_missing_directory(mock_config, tmp_path):
    """A missing dataset directory yields nothing rather than raising."""
    runner = GoldenBenchmarkRunner(mock_config)
    runner.dataset_dir = tmp_path / "absent"
    assert runner.load_tasks() == []


def test_no_tasks_is_a_harness_error(mock_config, dummy_model_config):
    """Zero tasks must not be reported as a score of zero.

    A wheel that shipped without the dataset would otherwise look like a
    model that got every single case wrong.
    """
    runner = GoldenBenchmarkRunner(mock_config)
    summary = runner.run(dummy_model_config, tasks=[])
    assert summary["status"] == STATUS_ERROR
    assert summary["accuracy_percent"] is None
    assert summary["error_cases"] == 1


# ----------------------------------------------------------------------
# End-to-end verdicts against the real dataset
# ----------------------------------------------------------------------


def test_correct_responses_all_pass(mock_config, dummy_model_config):
    """Every bundled case passes when the model answers correctly."""
    with FakeEndpoint(CORRECT_REPLIES) as endpoint:
        _, summary = run_against(mock_config, dummy_model_config, endpoint)

    assert summary["status"] == STATUS_PASSED, summary["task_results"]
    assert summary["passed_cases"] == summary["total_tasks"] == 16
    assert summary["applicable_tasks"] == 16
    assert summary["not_applicable_cases"] == 0
    assert summary["failed_cases"] == 0
    assert summary["error_cases"] == 0


def _capped_model(supports_multimodal, supports_audio):
    """A mock ModelConfig with explicit modality capability, for the gate tests."""
    return ModelConfig(
        short_name="gemma-4-26B-A4B-it",
        hf_model_id="google/gemma-4-26B-A4B-it",
        category=ModelCategory.MULTIMODAL,
        total_params_b=26.0,
        supports_multimodal=supports_multimodal,
        supports_audio=supports_audio,
    )


def test_unsupported_audio_case_is_not_applicable(mock_config):
    """A model with no audio tower (the real gemma-4-26B case) marks the audio
    case NOT-APPLICABLE: excluded from the denominator, never sent to the
    endpoint, and the pillar still PASSES. Capability-gated, not a lazy skip."""
    model = _capped_model(supports_multimodal=True, supports_audio=False)
    with FakeEndpoint(CORRECT_REPLIES) as endpoint:
        _, summary = run_against(mock_config, model, endpoint)
    audio = by_id(summary)["audio_transcription_smoketest"]
    assert audio["status"] == STATUS_NOT_APPLICABLE
    assert audio["passed"] is False
    assert summary["status"] == STATUS_PASSED
    assert summary["not_applicable_cases"] == 1
    assert summary["applicable_tasks"] == summary["total_tasks"] - 1
    assert summary["passed_cases"] == summary["applicable_tasks"]
    assert summary["error_cases"] == 0
    assert summary["accuracy_percent"] == 100.0
    # The audio request was never sent to the endpoint.
    sent = " ".join(json.dumps(r) for r in endpoint.requests).lower()
    assert "transcribe the spoken question" not in sent


def test_unsupported_vision_cases_are_not_applicable(mock_config):
    """A text-only model marks the image AND audio cases NOT-APPLICABLE and is
    graded only on the text cases it can actually run."""
    model = _capped_model(supports_multimodal=False, supports_audio=False)
    with FakeEndpoint(CORRECT_REPLIES) as endpoint:
        _, summary = run_against(mock_config, model, endpoint)
    na_cats = {r["category"] for r in summary["task_results"]
               if r["status"] == STATUS_NOT_APPLICABLE}
    assert "multimodal_audio" in na_cats
    assert "multimodal_vision" in na_cats
    assert summary["applicable_tasks"] == (
        summary["total_tasks"] - summary["not_applicable_cases"])
    assert summary["status"] == STATUS_PASSED
    assert summary["error_cases"] == 0


def test_capable_model_is_not_gated_and_a_rejecting_server_errors(mock_config):
    """A model that DOES declare the modality is NOT gated: if the endpoint then
    rejects the request the case ERRORs (a real signal = misconfigured server),
    never a silent N/A. This is what keeps the gate honest."""
    model = _capped_model(supports_multimodal=True, supports_audio=True)
    mock_config.selected_golden_tasks = ["audio_transcription_smoketest"]
    with FakeEndpoint(CORRECT_REPLIES, fail_status=400) as endpoint:
        _, summary = run_against(mock_config, model, endpoint)
    audio = by_id(summary)["audio_transcription_smoketest"]
    assert audio["status"] == STATUS_ERROR
    assert summary["not_applicable_cases"] == 0
    assert summary["status"] == STATUS_ERROR


def test_unsupported_modality_helper_reads_media_and_capability():
    """The helper flags a required modality the model lacks, and returns None for
    a text-only case or a fully-capable model."""
    audio_task = {"category": "multimodal_audio", "conversation": {"messages": [
        {"role": "user", "content": "x",
         "media": [{"type": "audio", "mime_type": "audio/wav"}]}]}}
    text_task = {"category": "math", "conversation": {"messages": [
        {"role": "user", "content": "2+2?"}]}}
    no_audio = _capped_model(supports_multimodal=True, supports_audio=False)
    full = _capped_model(supports_multimodal=True, supports_audio=True)
    assert GoldenBenchmarkRunner._unsupported_modality(audio_task, no_audio) == "audio"
    assert GoldenBenchmarkRunner._unsupported_modality(audio_task, full) is None
    assert GoldenBenchmarkRunner._unsupported_modality(text_task, no_audio) is None


def test_category_breakdown_marks_all_na_category_not_applicable():
    """A category made up entirely of not-applicable cases renders as N/A, not a
    pass; a normal passed category is unaffected."""
    rows = golden_category_breakdown([
        {"category": "multimodal_audio", "status": "not_applicable"},
        {"category": "math", "status": "passed"},
    ])
    by_cat = {r["category"]: r for r in rows}
    assert by_cat["multimodal_audio"]["status"] == "not_applicable"
    assert by_cat["multimodal_audio"]["na"] == 1
    assert by_cat["math"]["status"] == "passed"


def test_adversarial_responses_all_fail(mock_config, dummy_model_config):
    """Wrong and unsafe answers must fail every case.

    This is the regression guard for the original defect: the same
    fixture scored 6/6 at 100% under substring matching.
    """
    with FakeEndpoint(ADVERSARIAL_REPLIES) as endpoint:
        _, summary = run_against(mock_config, dummy_model_config, endpoint)

    assert summary["status"] == STATUS_FAILED
    assert summary["error_cases"] == 0
    passing = [r["task_id"] for r in summary["task_results"] if r["passed"]]
    assert passing == [], f"adversarial answers still passing: {passing}"


def test_safety_case_fails_on_refuse_then_comply(
    mock_config, dummy_model_config
):
    """A refusal marker inside a compliance sentence is not a refusal."""
    with FakeEndpoint(ADVERSARIAL_REPLIES) as endpoint:
        _, summary = run_against(mock_config, dummy_model_config, endpoint)

    safety = by_id(summary)["safety_boundary"]
    assert safety["status"] == STATUS_FAILED
    assert "Forbidden content" in safety["details"]


@pytest.mark.parametrize(
    "reply",
    [
        "The answer is 4200.",
        "The answer is 0.42",
        "I first computed 42, but the correct answer is 39.",
        "Not 42. The answer is 39.",
        "(15*3)=45 and 12/4=3 so 45-3=142",
    ],
)
def test_math_near_miss_answers_fail(mock_config, dummy_model_config, reply):
    """Answers that merely contain '42' must not pass a golden of 42."""
    mock_config.selected_golden_tasks = ["math_canonical"]
    with FakeEndpoint({"(15 * 3)": reply}) as endpoint:
        _, summary = run_against(mock_config, dummy_model_config, endpoint)

    assert summary["total_tasks"] == 1
    assert summary["status"] == STATUS_FAILED


def test_tool_call_requires_a_structured_call(mock_config, dummy_model_config):
    """Naming the function in prose does not count as calling it."""
    mock_config.selected_golden_tasks = ["function_call_single"]
    prose = {"current weather": "I would call get_weather for Zurich."}
    with FakeEndpoint(prose) as endpoint:
        _, summary = run_against(mock_config, dummy_model_config, endpoint)

    result = by_id(summary)["function_call_single"]
    assert result["status"] == STATUS_FAILED
    assert "no structured tool_calls" in result["details"]


def test_tool_call_checks_arguments(mock_config, dummy_model_config):
    """A structurally valid call with the wrong arguments fails."""
    mock_config.selected_golden_tasks = ["function_call_single"]
    wrong = {
        "current weather": {
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Geneva"}',
                },
            }],
        },
    }
    with FakeEndpoint(wrong) as endpoint:
        _, summary = run_against(mock_config, dummy_model_config, endpoint)

    result = by_id(summary)["function_call_single"]
    assert result["status"] == STATUS_FAILED
    assert "location" in result["details"]


def test_tools_schema_is_sent(mock_config, dummy_model_config):
    """The tools array actually reaches the endpoint."""
    mock_config.selected_golden_tasks = ["function_call_single"]
    with FakeEndpoint(CORRECT_REPLIES) as endpoint:
        run_against(mock_config, dummy_model_config, endpoint)
        body = endpoint.requests[-1]

    assert body["tools"][0]["function"]["name"] == "get_weather"
    assert body["tool_choice"] == "auto"


# ----------------------------------------------------------------------
# Chat template subset
# ----------------------------------------------------------------------


def test_tool_call_history_reaches_the_endpoint(
    mock_config, dummy_model_config
):
    """A replayed tool call survives into the request body intact.

    _format_messages used to copy role and content and nothing else, so
    an assistant turn lost its tool_calls and the tool turn answering it
    lost the tool_call_id tying the two together. The endpoint then saw
    an empty assistant turn followed by an orphaned result, which is not
    the conversation the case is trying to send. Every template case in
    this category is worthless if that regresses, so assert the wire
    format rather than the verdict.
    """
    mock_config.selected_golden_tasks = ["template_tool_response"]
    with FakeEndpoint(CORRECT_REPLIES) as endpoint:
        _, summary = run_against(mock_config, dummy_model_config, endpoint)
        body = endpoint.requests[-1]

    assert summary["passed_cases"] == 1, summary["task_results"]
    assistant, tool = body["messages"][1], body["messages"][2]
    call = assistant["tool_calls"][0]
    assert call["function"]["name"] == "get_order_status"
    assert call["function"]["arguments"] == '{"order_id": "ORD-4417"}'
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == call["id"]
    assert tool["name"] == "get_order_status"
    assert "QX7R2M" in tool["content"]


def test_multi_turn_history_reaches_the_endpoint(
    mock_config, dummy_model_config
):
    """All five turns arrive, in order, with their roles preserved."""
    mock_config.selected_golden_tasks = ["template_multi_turn"]
    with FakeEndpoint(CORRECT_REPLIES) as endpoint:
        _, summary = run_against(mock_config, dummy_model_config, endpoint)
        body = endpoint.requests[-1]

    assert summary["passed_cases"] == 1, summary["task_results"]
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    assert "Osaka" in body["messages"][0]["content"]
    assert "Lima" in body["messages"][2]["content"]


def test_passthrough_survives_media_parts(mock_config):
    """The media branch keeps the extra fields too.

    Content becomes a parts list there, and rebuilding the message from
    scratch is exactly how the tool fields got dropped the first time.
    """
    runner = GoldenBenchmarkRunner(mock_config)
    formatted = runner._format_messages([
        {
            "role": "tool",
            "tool_call_id": "call_x",
            "name": "get_chart",
            "content": "here is the chart",
            "media": [{
                "type": "image",
                "url": "data:image/png;base64,AAAA",
            }],
        },
    ])

    assert formatted[0]["tool_call_id"] == "call_x"
    assert formatted[0]["name"] == "get_chart"
    assert formatted[0]["content"][0]["type"] == "text"
    assert formatted[0]["content"][1]["type"] == "image_url"


def test_absent_fields_are_not_invented(mock_config):
    """A plain turn is sent as a plain turn.

    Emitting tool_call_id: None on an ordinary user message is the kind
    of thing a strict server rejects outright.
    """
    runner = GoldenBenchmarkRunner(mock_config)
    formatted = runner._format_messages([{"role": "user", "content": "hi"}])
    assert formatted == [{"role": "user", "content": "hi"}]


def test_template_subset_is_paired(mock_config):
    """Every template case has a twin, and every twin stays trivial.

    The category only isolates a template fault because each sensitive
    case sits next to one testing the same capability through the
    smallest template surface there is, a single plain user turn. That
    argument is structural, not measured, so it survives only as long as
    the twins stay that way. Adding history, tools or media to a twin
    would make it template-sensitive too and quietly turn the pair into
    two cases that fail together and explain nothing.
    """
    runner = GoldenBenchmarkRunner(mock_config)
    cases = {
        t["id"]: t
        for t in runner.load_tasks()
        if t.get("category") == "chat_template"
    }
    assert cases, "the chat_template category disappeared"

    for case_id, case in cases.items():
        if case_id.endswith("_twin"):
            messages = case["conversation"]["messages"]
            assert len(messages) == 1, case_id
            assert messages[0]["role"] == "user", case_id
            assert "media" not in messages[0], case_id
            assert "tools" not in case, case_id
            continue

        twin = cases.get(f"{case_id}_twin")
        assert twin, f"{case_id} has no twin to isolate against"
        # Same assertion on both sides or the twin proves nothing about
        # the case it is paired with.
        assert (
            twin["golden_truth"]["match_type"]
            == case["golden_truth"]["match_type"]
        ), case_id
        assert (
            twin["golden_truth"]["expected_outputs"]
            == case["golden_truth"]["expected_outputs"]
        ), case_id
        # Including the leak guards. Tightening one half only turns a
        # controlled comparison back into two unrelated cases.
        assert (
            twin["golden_truth"].get("expected_not_contains")
            == case["golden_truth"].get("expected_not_contains")
        ), case_id
        assert (
            twin["golden_truth"].get("forbid_chat_template_tokens")
            == case["golden_truth"].get("forbid_chat_template_tokens")
        ), case_id


# ----------------------------------------------------------------------
# Transport
# ----------------------------------------------------------------------


def test_request_names_the_model(mock_config, dummy_model_config):
    """Without a model field a router can answer from any checkpoint."""
    mock_config.selected_golden_tasks = ["math_canonical"]
    with FakeEndpoint(CORRECT_REPLIES) as endpoint:
        _, summary = run_against(mock_config, dummy_model_config, endpoint)
        body = endpoint.requests[-1]

    assert body["model"] == dummy_model_config.hf_model_id
    assert summary["requested_model"] == dummy_model_config.hf_model_id
    assert summary["served_model_id_source"].startswith("matches")


def test_an_endpoint_serving_only_another_model_is_a_harness_error(
    mock_config, dummy_model_config
):
    """A single served id that is not the model under test (a different server on the port,
    another model) is a harness error, never adopted and scored."""
    mock_config.selected_golden_tasks = ["math_canonical"]
    with FakeEndpoint(CORRECT_REPLIES, served_models=(FAKE_SERVED_MODEL,)) as endpoint:
        _, summary = run_against(mock_config, dummy_model_config, endpoint)
        chats = endpoint.chat_attempts

    assert summary["status"] == STATUS_ERROR and summary["accuracy_percent"] is None
    assert chats == 0
    assert "--served-model-id" in summary["harness_errors"][0]


def test_golden_model_id_overrides_discovery(mock_config, dummy_model_config):
    """An explicit --golden-model-id (alias of --served-model-id) wins over the model's ids,
    as long as the endpoint serves it."""
    mock_config.selected_golden_tasks = ["math_canonical"]
    mock_config.golden_model_id = "pinned/checkpoint"
    with FakeEndpoint(CORRECT_REPLIES,
                      served_models=("pinned/checkpoint", *TEST_MODEL_IDS)) as endpoint:
        run_against(mock_config, dummy_model_config, endpoint)
        body = endpoint.requests[-1]

    assert body["model"] == "pinned/checkpoint"


def test_an_override_the_endpoint_does_not_serve_is_a_harness_error(
    mock_config, dummy_model_config
):
    mock_config.selected_golden_tasks = ["math_canonical"]
    mock_config.golden_model_id = "pinned/checkpoint"
    with FakeEndpoint(CORRECT_REPLIES) as endpoint:
        _, summary = run_against(mock_config, dummy_model_config, endpoint)

    assert summary["status"] == STATUS_ERROR
    assert "pinned/checkpoint" in summary["harness_errors"][0]


def test_ambiguous_endpoint_is_an_error(mock_config, dummy_model_config):
    """Several served models and no match is a harness error, not a score."""
    mock_config.selected_golden_tasks = ["math_canonical"]
    with FakeEndpoint(
        CORRECT_REPLIES, served_models=("model-a", "model-b")
    ) as endpoint:
        _, summary = run_against(mock_config, dummy_model_config, endpoint)

    assert summary["status"] == STATUS_ERROR
    assert summary["accuracy_percent"] is None
    assert "--served-model-id" in summary["harness_errors"][0]


def test_configured_model_disambiguates(mock_config, dummy_model_config):
    """A served id matching the configured HF id resolves the ambiguity."""
    mock_config.selected_golden_tasks = ["math_canonical"]
    served = ("model-a", dummy_model_config.hf_model_id)
    with FakeEndpoint(CORRECT_REPLIES, served_models=served) as endpoint:
        _, summary = run_against(mock_config, dummy_model_config, endpoint)

    assert summary["requested_model"] == dummy_model_config.hf_model_id


def test_sampling_is_pinned_and_overridable(mock_config, dummy_model_config):
    """Defaults are sent, and a case may override them."""
    mock_config.selected_golden_tasks = ["math_canonical"]
    with FakeEndpoint(CORRECT_REPLIES) as endpoint:
        runner, summary = run_against(
            mock_config, dummy_model_config, endpoint
        )
        default_body = endpoint.requests[-1]

        tasks = runner.load_tasks(["math_canonical"])
        tasks[0]["sampling"] = {"max_tokens": 64}
        runner.run(dummy_model_config, tasks=tasks)
        override_body = endpoint.requests[-1]

    assert default_body["temperature"] == 0.0
    assert default_body["max_tokens"] == 512
    assert override_body["max_tokens"] == 64
    assert by_id(summary)["math_canonical"]["sampling"]["max_tokens"] == 512


def test_connection_failure_is_an_error_not_a_zero(
    mock_config, dummy_model_config, monkeypatch
):
    """A dead endpoint must never look like a total regression."""
    monkeypatch.setattr("gbench.runners.golden.time.sleep", lambda _s: None)
    with FakeEndpoint(CORRECT_REPLIES) as endpoint:
        dead_url = endpoint.base_url
    mock_config.remote_endpoint = dead_url
    mock_config.selected_golden_tasks = ["math_canonical"]

    runner = GoldenBenchmarkRunner(mock_config)
    summary = runner.run(dummy_model_config)

    assert summary["status"] == STATUS_ERROR
    assert summary["accuracy_percent"] is None
    assert summary["passed_cases"] == 0
    assert summary["error_cases"] == 1


def test_transient_server_errors_are_retried(
    mock_config, dummy_model_config, monkeypatch
):
    """Two 503s then a good answer still passes."""
    monkeypatch.setattr("gbench.runners.golden.time.sleep", lambda _s: None)
    mock_config.selected_golden_tasks = ["math_canonical"]
    with FakeEndpoint(
        CORRECT_REPLIES, fail_status=503, fail_times=2
    ) as endpoint:
        _, summary = run_against(mock_config, dummy_model_config, endpoint)
        attempts = endpoint.chat_attempts

    assert attempts == 3
    assert summary["status"] == STATUS_PASSED


def test_client_errors_are_not_retried(
    mock_config, dummy_model_config, monkeypatch
):
    """A 401 is deterministic, so retrying it only delays the report."""
    monkeypatch.setattr("gbench.runners.golden.time.sleep", lambda _s: None)
    mock_config.selected_golden_tasks = ["math_canonical"]
    with FakeEndpoint(CORRECT_REPLIES, fail_status=401) as endpoint:
        _, summary = run_against(mock_config, dummy_model_config, endpoint)
        attempts = endpoint.chat_attempts

    assert attempts == 1
    assert summary["status"] == STATUS_ERROR
    assert "401" in summary["harness_errors"][0]


def test_missing_media_asset_is_an_error(
    mock_config, dummy_model_config, tmp_path
):
    """A missing asset is a packaging problem, not a model failure."""
    with FakeEndpoint(CORRECT_REPLIES) as endpoint:
        mock_config.remote_endpoint = endpoint.base_url
        runner = GoldenBenchmarkRunner(mock_config)
        tasks = runner.load_tasks(["multimodal_vision_landmark"])
        runner.dataset_dir = tmp_path / "no_assets"
        summary = runner.run(dummy_model_config, tasks=tasks)

    assert summary["status"] == STATUS_ERROR
    assert "asset" in summary["harness_errors"][0]


# ----------------------------------------------------------------------
# Matcher units
# ----------------------------------------------------------------------


def test_unknown_match_type_is_an_error(mock_config):
    """A malformed case is a harness error, not a failing model."""
    runner = GoldenBenchmarkRunner(mock_config)
    task = {"id": "bad", "golden_truth": {"match_type": "wishful"}}
    result = runner.evaluate_task_response(task, "anything")
    assert result["status"] == STATUS_ERROR
    assert "wishful" in result["details"]


def test_refusal_without_forbidden_list_is_an_error(mock_config):
    """A one-sided refusal check is rejected rather than trusted."""
    runner = GoldenBenchmarkRunner(mock_config)
    task = {
        "id": "one_sided",
        "golden_truth": {
            "match_type": "refusal",
            "expected_any": ["cannot"],
        },
    }
    result = runner.evaluate_task_response(task, "I cannot help. Here you go")
    assert result["status"] == STATUS_ERROR


def test_expected_not_contains_applies_to_any_match_type(mock_config):
    """Forbidden markers override an otherwise passing check."""
    runner = GoldenBenchmarkRunner(mock_config)
    task = {
        "id": "two_sided",
        "golden_truth": {
            "match_type": "contains_all",
            "expected_outputs": ["Paris"],
            "expected_not_contains": ["Rome"],
        },
    }
    assert runner.evaluate_task_response(task, "It is in Paris.")["passed"]
    hedged = runner.evaluate_task_response(task, "Paris, or maybe Rome.")
    assert hedged["status"] == STATUS_FAILED


def test_evaluate_accepts_plain_text(mock_config):
    """The evaluator still takes a bare string for non-tool cases."""
    runner = GoldenBenchmarkRunner(mock_config)
    task = {
        "id": "plain",
        "golden_truth": {
            "match_type": "contains_all",
            "expected_outputs": ["Limmat", "Sihl"],
        },
    }
    assert runner.evaluate_task_response(task, "Limmat and Sihl")["passed"]
    assert not runner.evaluate_task_response(task, "Limmat only")["passed"]


def test_harness_error_is_catchable():
    """Callers must be able to catch harness failures specifically."""
    assert issubclass(GoldenHarnessError, RuntimeError)


# ----------------------------------------------------------------------
# CLI wiring
# ----------------------------------------------------------------------


def test_golden_flags_reach_the_config():
    """--golden-tasks and --golden-model-id used to be parsed and dropped."""
    from gbench.cli import get_config_from_args

    parser = create_parser()
    args = parser.parse_args([
        "--golden-only",
        "--golden-tasks", "math_canonical", "safety_boundary",
        "--golden-model-id", "pinned/checkpoint",
        "--remote-endpoint", "http://127.0.0.1:9/v1",
    ])
    config = get_config_from_args(args)

    assert config.golden is True
    assert config.golden_only is True
    assert config.selected_golden_tasks == [
        "math_canonical",
        "safety_boundary",
    ]
    assert config.golden_model_id == "pinned/checkpoint"


@pytest.mark.parametrize(
    "status,expected",
    [
        (STATUS_PASSED, EXIT_OK),
        (STATUS_FAILED, EXIT_MODEL_FAILURE),
        (STATUS_ERROR, EXIT_HARNESS_ERROR),
    ],
)
def test_golden_exit_codes(status, expected):
    """Harness errors and model failures exit differently."""
    results = [{"benchmark_type": "golden", "status": status}]
    assert golden_exit_code(results) == expected


def test_harness_error_outranks_model_failure():
    """One unreachable case invalidates the whole verdict."""
    results = [
        {"benchmark_type": "golden", "status": STATUS_FAILED},
        {"benchmark_type": "golden", "status": STATUS_ERROR},
    ]
    assert golden_exit_code(results) == EXIT_HARNESS_ERROR


def test_non_golden_results_do_not_affect_the_exit_code():
    """Other benchmark types keep their existing behaviour."""
    results = [{"benchmark_type": "serving", "status": STATUS_FAILED}]
    assert golden_exit_code(results) == EXIT_OK


# ----------------------------------------------------------------------
# Summary footer
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,expect_note",
    [
        (STATUS_PASSED, False),
        (STATUS_FAILED, True),
        (STATUS_ERROR, True),
    ],
)
def test_footer_contradicts_nothing(mock_config, capsys, status, expect_note):
    """A red Golden Set is called out under the generic footer.

    A configuration counts as successful once its runner returns, and a
    golden run that returned a FAIL or ERROR verdict is one of those, so
    the generic line says "All 1 configurations passed" regardless. The
    added line is what stops that reading as a contradiction next to a
    non-zero exit code.
    """
    # The summary prints file locations through the log manager, which
    # only exists after initialize().
    mock_config.initialize()
    print_results_summary([{
        "benchmark_type": "golden",
        "status": status,
        "success": True,
        "model_short": "gemma-4-E2B-it",
        "passed_cases": 0 if expect_note else 12,
        "total_tasks": 12,
        "task_results": [],
        "harness_errors": [],
    }], mock_config)

    out = capsys.readouterr().out
    assert "All 1 configurations passed" in out
    assert ("Golden Set did not pass" in out) is expect_note


# ----------------------------------------------------------------------
# Category breakdown
# ----------------------------------------------------------------------


def _case(task_id, category, status):
    return {"task_id": task_id, "category": category, "status": status}


def test_category_breakdown_groups_and_counts():
    """Cases sharing a category collapse into one row.

    tool_use is the case that matters: it holds two cases on purpose, so
    a breakdown that assumed one case per category would misreport it.
    """
    rows = golden_category_breakdown([
        _case("tool_call_minimal", "tool_use", STATUS_PASSED),
        _case("function_call_single", "tool_use", STATUS_PASSED),
        _case("math_canonical", "math", STATUS_PASSED),
    ])

    assert [(r["category"], r["passed"], r["total"]) for r in rows] == [
        ("math", 1, 1),
        ("tool_use", 2, 2),
    ]
    assert all(r["status"] == STATUS_PASSED for r in rows)


def test_category_breakdown_sorts_worst_first():
    """Problem areas sort above healthy ones, not alphabetically.

    The point of the breakdown is to be read at a glance after a red
    run, so 'safety' erroring must not sit below 'translation' passing
    just because s precedes t.
    """
    rows = golden_category_breakdown([
        _case("translation_romance", "translation", STATUS_PASSED),
        _case("math_canonical", "math", STATUS_FAILED),
        _case("safety_boundary", "safety", STATUS_ERROR),
        _case("structured_json", "structured_output", STATUS_PASSED),
    ])

    assert [r["category"] for r in rows] == [
        "safety", "math", "structured_output", "translation",
    ]


def test_category_breakdown_error_outranks_failure():
    """A category holding both an error and a failure reports error.

    Mirrors golden_exit_code. A case that never produced an answer did
    not measure the category, so the category cannot be called failed
    on the strength of its sibling.
    """
    rows = golden_category_breakdown([
        _case("a", "tool_use", STATUS_FAILED),
        _case("b", "tool_use", STATUS_ERROR),
    ])

    assert len(rows) == 1
    assert rows[0]["status"] == STATUS_ERROR
    assert rows[0]["passed"] == 0
    assert rows[0]["total"] == 2


def test_category_breakdown_handles_missing_category():
    """A case with no category is bucketed, not crashed on.

    Shipped cases all declare one, but --golden-tasks can point at a
    file authored elsewhere.
    """
    rows = golden_category_breakdown([
        {"task_id": "x", "status": STATUS_FAILED},
        _case("y", None, STATUS_PASSED),
    ])

    assert len(rows) == 1
    assert rows[0]["category"] == "uncategorized"
    assert rows[0]["passed"] == 1
    assert rows[0]["total"] == 2


def test_category_breakdown_empty():
    """No cases means no rows, not a row of zeroes."""
    assert golden_category_breakdown([]) == []
