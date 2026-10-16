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

"""Knowledge suites send their upstream request, including its max_tokens:

* mmlu_pro runs TIGER-Lab's evaluate_from_apiX.py (the repository's runner for
  OpenAI-compatible endpoints) with its defaults: temperature 0.1, top_p 0.95, max_tokens 32768,
  streamed, a tiktoken cl100k_base count of prompt plus deltas with 3 tries past the limit,
  findall-last extraction on the content, 2 s start spacing and the stall check, categories in
  turn;
* gpqa, gpqa_diamond and multilingual_mmlu send simple-evals' chat sampler max_tokens=2048
  without thinking and the run's budget with it (the 2048 is not applied to reasoning);
* gsm8k (256), causalbench (10), cruxeval (100, direct prompts) and longbench_v2_code (128)
  send the upstream cap on a no-think request, whatever --max-output-tokens says.

No model server: endpoints are faked in process or on a loopback port that is never 8000-8010.
The mmlu_pro tests count tokens with the real tiktoken cl100k_base encoding, so its BPE file must
be in tiktoken's cache (TIKTOKEN_CACHE_DIR, else <tmp>/data-gym-cache) or downloadable from
openaipublic.blob.core.windows.net on the first run.
"""

import asyncio
import hashlib
import inspect
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from gbench.runners.eval_suites import (base, causalbench, cruxeval, gpqa, gsm8k,
                                        longbench_v2_code, mmlu_pro, multilingual_mmlu)

REPO = Path(__file__).resolve().parent.parent


def _doc(suite):
    return (REPO / "docs" / "evals" / f"{suite}.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
def _chunk(content=None, reasoning=None, finish=None):
    delta = SimpleNamespace(content=content)
    if reasoning is not None:
        delta.reasoning = reasoning
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=delta,
                                                                finish_reason=finish)])


def _usage_chunk(completion=5):
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=10, completion_tokens=completion,
                                                 total_tokens=10 + completion), choices=[])


def _reply(text, reasoning=None, usage=True):
    out = [_chunk(content="")]
    if reasoning:
        out.append(_chunk(reasoning=reasoning))
    out += [_chunk(content=w + " ") for w in text.split()]
    out.append(_chunk(finish="stop"))
    if usage:
        out.append(_usage_chunk())
    return out


class _Stream:
    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._chunks:
            yield c

    async def close(self):
        self.closed = True


class _Client:
    """openai.AsyncOpenAI stand-in: `script(call_index, request)` returns chunks or raises."""

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.times = []
        self.kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **request):
        self.calls.append(request)
        self.times.append(time.monotonic())
        out = self.script(len(self.calls) - 1, request)
        if isinstance(out, BaseException):
            raise out
        return _Stream(out)

    async def close(self):
        pass


@pytest.fixture
def fast(monkeypatch):
    """No start spacing or retry waits, so a test runs in milliseconds."""
    monkeypatch.setattr(mmlu_pro, "START_SPACING_S", 0.0)
    monkeypatch.setattr(mmlu_pro, "ERROR_RETRY_WAIT_S", 0.0)
    monkeypatch.setattr(mmlu_pro, "_token_retry_wait", lambda n: 0.0)


def _samples(n, cats=("math",)):
    return [([{"role": "user", "content": f"Question {i}"}], "B",
             {"category": cats[i % len(cats)], "question_id": 100 + i, "row": i})
            for i in range(n)]


def _run_pro(monkeypatch, client, samples, **kwargs):
    made = {}

    def factory(**kw):
        made.update(kw)
        client.kwargs = kw
        return client
    monkeypatch.setattr("openai.AsyncOpenAI", factory)
    with mock.patch.object(mmlu_pro, "_load_mmlu_pro_samples", return_value=samples):
        return mmlu_pro.run_mmlu_pro("m", "http://127.0.0.1:9/v1", kwargs.pop("concurrency", 2),
                                     **kwargs)


# --------------------------------------------------------------------------- #
# mmlu_pro: evaluate_from_apiX.py verbatim
# --------------------------------------------------------------------------- #
# sha256 of each function's source in TIGER-AI-Lab/MMLU-Pro evaluate_from_apiX.py @ f418b116.
_APIX_SHA256 = {
    "format_example": "7c6b73df41591839c67ec484b6a6a2f0f926ce765a822c7e1e8b545fa68ee02f",
    "extract_answer": "b802ce606497c63d67f2fd884eecf239cff9d28cd963c0c67811855112b29e2b",
}


def test_mmlu_pro_ports_evaluate_from_apix_functions_verbatim():
    for name, digest in _APIX_SHA256.items():
        src = inspect.getsource(getattr(mmlu_pro, name)).rstrip("\n")
        assert hashlib.sha256(src.encode()).hexdigest() == digest, name
    assert mmlu_pro._PATTERNS == (r"answer is \(?([A-J])\)?", r"[aA]nswer:\s*([A-J])",
                                  r"\b[A-J]\b")
    assert mmlu_pro._prompt_header("law") == (
        "The following are multiple choice questions (with answers) about law. Think step by "
        "step and then output the answer in the format of \"The answer is (X)\" at the end.\n\n")
    assert mmlu_pro.UPSTREAM == "TIGER-AI-Lab/MMLU-Pro@f418b116 evaluate_from_apiX.py"


def test_mmlu_pro_extraction_takes_the_last_match_of_the_first_pattern_that_matches():
    ex = mmlu_pro.extract_answer
    assert ex("the answer is (A). Hmm, no: the answer is (C).") == "C"
    assert ex("Answer: D ... Answer: E") == "E"                 # no '.*' anchor: last on any line
    assert ex("The answer is **(B)**") == "B"                   # no '**' stripping; stage 3 reads B
    assert ex("I would pick G, or maybe H") == "H"
    assert ex("Based on nothing") is None and ex("") is None


def test_mmlu_pro_samples_like_evaluate_from_apix():
    official = mmlu_pro.OFFICIAL_SAMPLING
    assert official.temperature == 0.1 and official.params == {"top_p": 0.95}
    assert "evaluate_from_apiX.py streaming_request" in official.reference
    assert (mmlu_pro.UPSTREAM_MAX_TOKENS, mmlu_pro.UPSTREAM_TOKEN_TRIES) == (32768, 3)
    assert mmlu_pro.TOKEN_ENCODING == "cl100k_base"
    assert (mmlu_pro.START_SPACING_S, mmlu_pro.STALL_S, mmlu_pro.ERROR_RETRY_WAIT_S) == (
        2.0, 25.0, 5.0)
    assert mmlu_pro.REQUEST_TIMEOUT_S == 3600.0
    assert mmlu_pro._token_retry_wait(1) == 2 and mmlu_pro._token_retry_wait(9) == 30


@pytest.mark.parametrize("thinking", [False, True])
def test_mmlu_pro_request_is_evaluate_from_apix_request(monkeypatch, fast, thinking):
    client = _Client(lambda i, req: _reply("Reasoning. The answer is (B)."))
    res = _run_pro(monkeypatch, client, _samples(2), enable_thinking=thinking,
                   max_output_tokens=65536)
    # a placeholder key (openai 2.54+ refuses none) that the HTTP client's hook never sends
    http_client = client.kwargs.pop("http_client")
    assert mmlu_pro._drop_authorization in http_client.event_hooks["request"]
    assert client.kwargs == {"api_key": mmlu_pro.CLIENT_KEY_PLACEHOLDER,
                             "base_url": "http://127.0.0.1:9/v1", "timeout": 3600.0}
    # The token-limit policy: 32768 exactly without thinking; a thinking run is never capped
    # below --max-output-tokens (and is then a different measurement).
    expected = 65536 if thinking else 32768
    # gbench's rule on the wire, not upstream's 0.1: 0.0 without thinking, 1.0 with it (top_p
    # stays 0.95)
    temperature = 1.0 if thinking else 0.0
    for req in client.calls:
        assert req == {"model": "m", "max_tokens": expected, "top_p": 0.95,
                       "temperature": temperature,
                       "stream": True, "stream_options": {"include_usage": True},
                       "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}},
                       "messages": req["messages"]}
        assert [m["role"] for m in req["messages"]] == ["user"]
    assert res["accuracy"] == 100.0 and res["leaderboard_comparable"] is False
    assert (res["temperature"], res["temperature_source"]) == (
        (1.0, "thinking") if thinking else (0.0, "default"))
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert res["max_tokens_sent"] == [expected]
    assert (f"sampled at temperature {temperature} ({'thinking' if thinking else 'default'}; "
            "evaluate_from_apiX.py sends temperature 0.1)") in res["leaderboard_comparable_reason"]
    if thinking:
        assert "never capped below --max-output-tokens" in res["max_output_tokens_note"]
    else:
        assert res["max_output_tokens_note"].startswith(
            "--max-output-tokens 65536 not applied: the official harness sends max_tokens=32768 "
            "with every request (TIGER-AI-Lab/MMLU-Pro@f418b116 evaluate_from_apiX.py")
    trace = res["sample_traces"][0]
    assert trace["status"] == "OK" and trace["predicted"] == "B"
    assert trace["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_mmlu_pro_token_count_is_cl100k_of_prompt_plus_reasoning_and_content(monkeypatch, fast):
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    chunks = [_chunk(content=""), _chunk(reasoning="Let me think."),
              _chunk(content="The answer"), _chunk(content=" is (B)."), _usage_chunk()]
    client = _Client(lambda i, req: chunks)
    res = _run_pro(monkeypatch, client, _samples(1))
    expected = (len(enc.encode("Question 0")) + len(enc.encode("Let me think."))
                + len(enc.encode("The answer")) + len(enc.encode(" is (B).")))
    trace = res["sample_traces"][0]
    assert trace["client_token_count"] == expected
    # The reasoning is counted but not scored: the reply is the content.
    assert trace["response_text"] == "The answer is (B)." and trace["reasoning"] == "Let me think."


def test_mmlu_pro_reads_the_answer_from_the_content_only(monkeypatch, fast):
    client = _Client(lambda i, req: (_reply("The answer is (B).", reasoning="The answer is (C).")
                                     if i == 0 else _reply("", reasoning="The answer is (B).")))
    monkeypatch.setenv("GBENCH_MMLU_PRO_TEMPERATURE", "0.1")    # upstream's, set explicitly
    res = _run_pro(monkeypatch, client, _samples(2), concurrency=1)
    first, second = res["sample_traces"]
    assert first["predicted"] == "B" and first["is_correct"] is True
    assert second["predicted"] is None and second["is_correct"] is False
    assert res["no_answer_replies"] == 1 and res["empty_responses"] == 1
    assert res["leaderboard_comparable"] is True      # scored 0 as upstream scores it


def test_mmlu_pro_token_limit_asks_again_up_to_three_tries(monkeypatch, fast):
    monkeypatch.setattr(mmlu_pro, "UPSTREAM_MAX_TOKENS", 40)
    # The cap is the official sampling's (both modes), applied by base's token-limit policy.
    official = mmlu_pro.OFFICIAL_SAMPLING
    monkeypatch.setattr(mmlu_pro, "OFFICIAL_SAMPLING", official._replace(
        max_tokens=40, reasoning=official.reasoning._replace(max_tokens=40)))
    long = _reply("word " * 60 + "The answer is (B).")

    def script(i, req):
        return long if i < 4 else _reply("The answer is (B).")
    client = _Client(script)
    monkeypatch.setenv("GBENCH_MMLU_PRO_TEMPERATURE", "0.1")    # upstream's, set explicitly
    res = _run_pro(monkeypatch, client, _samples(2), concurrency=1)
    exceeded, recovered = res["sample_traces"]
    # Question 0: three tries past the limit, then MaxTokensExceeded (scored 0).
    assert exceeded["status"] == "MAX_TOKENS_EXCEEDED"
    assert exceeded["max_tokens_exceeded_tries"] == 3
    assert exceeded["predicted"] is None and exceeded["response_text"] is None
    assert "word" in exceeded["partial_response_tail"]
    # Question 1: one try past the limit, the second fits.
    assert recovered["status"] == "OK" and recovered["max_tokens_exceeded_tries"] == 1
    assert recovered["is_correct"] is True
    assert len(client.calls) == 5 and all(r["max_tokens"] == 40 for r in client.calls)
    assert res["max_tokens_exceeded"] == 1 and res["accuracy"] == 50.0
    assert res["leaderboard_comparable"] is True and res["status"] == "success"


def test_mmlu_pro_missing_usage_and_errors_are_asked_again_then_fail(monkeypatch, fast):
    monkeypatch.setattr(mmlu_pro, "ERROR_TRIES", 3)
    monkeypatch.setenv("GEMINI_API_KEY", "sk-test-secret-value-123")

    def script(i, req):
        if i == 0:
            return _reply("The answer is (B).", usage=False)       # MissingUsageError
        if i == 1:
            return _reply("The answer is (B).")
        return RuntimeError("upstream down sk-test-secret-value-123")
    client = _Client(script)
    res = _run_pro(monkeypatch, client, _samples(2), concurrency=1)
    ok, failed = res["sample_traces"]
    assert ok["status"] == "OK" and ok["error_tries"] == 1
    assert failed["status"] == "FAILED" and failed["error_tries"] == 3
    assert "sk-test-secret-value-123" not in json.dumps(res, default=str)
    assert res["failed_requests"] == 1 and res["status"] == "completed_with_errors"
    assert res["leaderboard_comparable"] is False
    assert "1 question(s) got no reply from the served endpoint after 3 tries" in \
        res["leaderboard_comparable_reason"]


def test_mmlu_pro_endpoint_rejection_is_a_hard_error_and_429_is_retried(monkeypatch, fast):
    class StatusError(Exception):
        def __init__(self, code):
            super().__init__(f"HTTP {code}")
            self.status_code, self.message = code, f"status {code}"

    client = _Client(lambda i, req: StatusError(400))
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*mmlu_pro.*HTTP 400"):
        _run_pro(monkeypatch, client, _samples(3), concurrency=2)
    client = _Client(lambda i, req: StatusError(429) if i == 0 else _reply("The answer is (B)."))
    res = _run_pro(monkeypatch, client, _samples(1))
    assert res["sample_traces"][0]["error_tries"] == 1 and res["accuracy"] == 100.0


def test_mmlu_pro_runs_categories_in_turn_in_dataset_order(monkeypatch, fast):
    samples = [([{"role": "user", "content": f"{cat}-{row}"}], "B",
                {"category": cat, "row": row, "question_id": row})
               for cat, row in (("law", 3), ("math", 0), ("law", 1), ("math", 2))]
    client = _Client(lambda i, req: _reply("The answer is (B)."))
    res = _run_pro(monkeypatch, client, samples, concurrency=1)
    order = [r["messages"][0]["content"] for r in client.calls]
    assert order == ["math-0", "math-2", "law-1", "law-3"]
    # Traces stay in sample order.
    assert [t["row"] for t in res["sample_traces"]] == [3, 0, 1, 2]


def test_mmlu_pro_spaces_request_starts_within_a_category(monkeypatch, fast):
    monkeypatch.setattr(mmlu_pro, "START_SPACING_S", 0.15)
    client = _Client(lambda i, req: _reply("The answer is (B)."))
    _run_pro(monkeypatch, client, _samples(3), concurrency=3)
    gaps = [b - a for a, b in zip(client.times, client.times[1:])]
    assert len(gaps) == 2 and min(gaps) >= 0.14


def test_mmlu_pro_stall_check_holds_back_new_questions(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(mmlu_pro.time, "time", lambda: now[0])
    pacer = mmlu_pro._Pacer()
    pacer.reset(1)
    assert pacer.any_stalled() is False          # no token history yet: not stalled
    pacer.token(1)
    now[0] += 26.0
    assert pacer.any_stalled() is True           # 26 s without a token
    pacer.token(1)
    assert pacer.any_stalled() is False
    now[0] += 26.0
    pacer.done(1)
    assert pacer.any_stalled() is False


def test_mmlu_pro_window_too_small_is_a_hard_error_before_any_request(monkeypatch, fast):
    base.set_run_knobs(max_model_len=32768)
    client = _Client(lambda i, req: _reply("The answer is (B)."))
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*max_tokens=32768"):
        _run_pro(monkeypatch, client, _samples(1))
    assert client.calls == []


def test_mmlu_pro_missing_client_or_encoding_is_a_hard_error(monkeypatch, fast):
    import builtins
    import tiktoken
    client = _Client(lambda i, req: _reply("The answer is (B)."))
    with mock.patch.object(tiktoken, "get_encoding", side_effect=OSError("offline")):
        with pytest.raises(RuntimeError,
                           match=r"INFRA REQUIRED.*cl100k_base.*TIKTOKEN_CACHE_DIR"):
            _run_pro(monkeypatch, client, _samples(1))
    assert client.calls == []
    real_import = builtins.__import__

    def no_openai(name, *args, **kwargs):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, *args, **kwargs)
    with mock.patch.object(builtins, "__import__", side_effect=no_openai), \
            mock.patch.object(mmlu_pro, "_load_mmlu_pro_samples", return_value=_samples(1)):
        with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*`openai` package"):
            mmlu_pro.run_mmlu_pro("m", "http://127.0.0.1:9/v1", 1)


def test_mmlu_pro_never_sends_the_bearer_header_the_client_builds(monkeypatch, fast):
    """The openai client holds a placeholder key (2.54+ refuses none); the HTTP client's request
    hook removes whatever Authorization header the client built from it, empty or not."""
    import httpx
    for value in ("Bearer ", "Bearer " + mmlu_pro.CLIENT_KEY_PLACEHOLDER):
        request = httpx.Request("POST", "http://127.0.0.1:9/v1/chat/completions",
                                headers={"Authorization": value})
        asyncio.run(mmlu_pro._drop_authorization(request))
        assert "authorization" not in request.headers
    client = _Client(lambda i, req: _reply("The answer is (B)."))
    res = _run_pro(monkeypatch, client, _samples(1))
    assert client.calls and res["accuracy"] == 100.0


def test_mmlu_pro_operator_settings_are_sent_and_flagged(monkeypatch, fast):
    monkeypatch.setenv("GBENCH_MMLU_PRO_TEMPERATURE", "0.7")
    monkeypatch.setenv("GBENCH_REPETITION_PENALTY", "1.1")
    client = _Client(lambda i, req: _reply("The answer is (B)."))
    res = _run_pro(monkeypatch, client, _samples(1), attempt_count=3)
    req = client.calls[0]
    assert req["temperature"] == 0.7 and req["extra_body"]["repetition_penalty"] == 1.1
    reason = res["leaderboard_comparable_reason"]
    assert "sampled at temperature 0.7" in reason and "decoding penalties" in reason
    assert res["attempts"]["not_applied"].startswith("evaluate_from_apiX.py asks each question")


def test_mmlu_pro_sends_no_key_to_the_endpoint_over_http(monkeypatch, fast):
    """The real openai client against a loopback SSE server: the body is exactly
    evaluate_from_apiX.py's, and no Authorization header is sent even with OPENAI_API_KEY set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-never-be-sent")
    seen = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body, dict(self.headers)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            base_chunk = {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m"}
            events = [dict(base_chunk, choices=[{"index": 0, "delta": {"content": piece},
                                                 "finish_reason": None}])
                      for piece in ("The answer ", "is (B).")]
            events.append(dict(base_chunk, choices=[], usage={
                "prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}))
            for event in events + ["[DONE]"]:
                data = ("data: " + (event if isinstance(event, str) else json.dumps(event))
                        + "\n\n").encode()
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    assert not 8000 <= port <= 8010
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with mock.patch.object(mmlu_pro, "_load_mmlu_pro_samples", return_value=_samples(1)):
            res = mmlu_pro.run_mmlu_pro("m", f"http://127.0.0.1:{port}/v1", 1)
    finally:
        server.shutdown()
        server.server_close()
    path, body, headers = seen[0]
    assert path == "/v1/chat/completions"
    assert body == {"model": "m", "messages": [{"role": "user", "content": "Question 0"}],
                    "temperature": 0.0, "top_p": 0.95, "max_tokens": 32768, "stream": True,
                    "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": False}}
    assert "Authorization" not in headers and "authorization" not in headers
    assert res["accuracy"] == 100.0 and res["sample_traces"][0]["usage"]["total_tokens"] == 7


def test_mmlu_pro_doc_and_header_describe_evaluate_from_apix():
    doc = " ".join(_doc("mmlu_pro").split())
    for needle in ("evaluate_from_apiX.py", "`temperature=0.1`", "`top_p=0.95`",
                   "`max_tokens=32768`", "`cl100k_base`", "3 tries in all", "2 seconds apart",
                   "25 seconds", "stops after 10 tries", "HTTP 4xx other than 408, 409 and 429",
                   "`max_output_tokens_note`", "openai 1.30.1", "TIKTOKEN_CACHE_DIR",
                   "about 6.7 hours", "`--suite-timeout`"):
        assert needle in doc, needle
    assert "max_tokens=4000" not in doc and "random guess" in doc
    header = " ".join(mmlu_pro.__doc__.split())
    assert "evaluate_from_apiX.py" in header and "10 tries" in header
    assert "gpt-4o" not in header


# --------------------------------------------------------------------------- #
# simple-evals chat sampler: max_tokens=2048 reaches the request
# --------------------------------------------------------------------------- #
def _gpqa_rows(n):
    return [{"Question": f"Q{i}?", "Correct Answer": f"right{i}", "Incorrect Answer 1": "w1",
             "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3", "Subdomain": "Physics"}
            for i in range(n)]


@pytest.mark.parametrize("thinking", [False, True])
def test_gpqa_request_max_tokens_follows_the_sampler(monkeypatch, thinking):
    sent = []

    async def fake_send(**kw):
        sent.append(kw)
        return base.Reply("Answer: A", None, "length" if not thinking else "stop",
                          completion_tokens=2048)
    monkeypatch.setattr(base, "_send_single_request", fake_send)
    base.set_run_knobs(max_output_tokens=65536)
    with mock.patch.object(gpqa, "_load_gpqa_rows", return_value=_gpqa_rows(1)):
        res = gpqa.run_gpqa("m", "http://x/v1", 1, enable_thinking=thinking)
    assert len(sent) == 10
    if thinking:
        # The chat sampler's 2048 is not applied to reasoning: the run's budget is sent.
        assert all(kw["max_output_tokens"] == 65536 for kw in sent)
        assert res["sampler_max_tokens"] is None
        assert res["max_output_tokens_note"].startswith(
            "official cap not applied to reasoning: it was defined for non-reasoning generation "
            "(the official harness sends max_tokens=2048); each request was sent this run's "
            "budget, --max-output-tokens 65536")
        assert (res["official_sampling"]["max_tokens_applied"],
                res["official_sampling"]["max_tokens_sent"]) == (False, 65536)
    else:
        assert all(kw["max_output_tokens"] == 2048 for kw in sent)
        assert res["max_output_tokens_note"].startswith(
            "--max-output-tokens 65536 not applied: the official harness sends max_tokens=2048 "
            "with every request (openai/simple-evals@652c89d0")
        assert res["replies_at_sampler_max_tokens"] == 10
    # gbench samples at 1.0 / 0.0, not the sampler's 0.5, and is flagged for it
    assert all(kw["temperature"] == (1.0 if thinking else 0.0) for kw in sent)
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"] == (
        "temperature 1.0 (thinking; simple-evals' sampler sends temperature 0.5)" if thinking
        else "temperature 0.0 (default; simple-evals' sampler sends temperature 0.5)")


def test_multilingual_mmlu_meta_carries_the_sampler_cap_without_thinking():
    # No per-sample cap helper is left; multilingual_mmlu declares the chat sampler's 2048 once,
    # in gpqa.SIMPLE_EVALS_SAMPLING (base replaces it with the run's budget with --thinking).
    assert not hasattr(gpqa, "simple_evals_meta")
    src = inspect.getsource(multilingual_mmlu._load_multilingual_mmlu_samples)
    assert "simple_evals_meta(" not in src
    assert "SIMPLE_EVALS_SAMPLING" in inspect.getsource(multilingual_mmlu.run_multilingual_mmlu)


# --------------------------------------------------------------------------- #
# single-sampler suites: the upstream cap on a no-think request
# --------------------------------------------------------------------------- #
def _through_base(monkeypatch, run, loader_patch, thinking=False, **kwargs):
    sent = []

    async def fake_send(**kw):
        sent.append(kw)
        return base.Reply("The answer is 3.", None, "stop", completion_tokens=5)
    monkeypatch.setattr(base, "_send_single_request", fake_send)
    base.set_run_knobs(max_output_tokens=65536)
    with loader_patch:
        res = run(thinking, **kwargs)
    return res, sent


def test_gsm8k_no_think_request_carries_max_gen_toks(monkeypatch):
    samples = [(gsm8k.fewshot_messages("q"), "3", {"stop": list(gsm8k.UNTIL),
                                                   "max_tokens": 256})]
    res, sent = _through_base(
        monkeypatch, lambda t: gsm8k.run_gsm8k("m", "http://x/v1", 1, enable_thinking=t),
        mock.patch.object(gsm8k, "_load_gsm8k_samples", return_value=samples))
    assert sent[0]["max_output_tokens"] == 256 and "max_tokens" not in sent[0]["extra_payload"]
    assert res["gsm8k_cot_report"]["max_gen_toks_sent"] == 256
    assert "not applied" in res["max_output_tokens_note"]
    doc = gsm8k._reply_text.__doc__
    assert "as lm-eval's chat backend sees it" not in doc and "stop=None" in doc


def test_gsm8k_thinking_request_keeps_the_run_budget(monkeypatch):
    res, sent = _through_base(
        monkeypatch, lambda t: gsm8k.run_gsm8k("m", "http://x/v1", 1, enable_thinking=t),
        mock.patch.object(gsm8k, "_load_gsm8k_samples",
                          return_value=[(gsm8k.fewshot_messages("q"), "3", {})]), thinking=True)
    assert sent[0]["max_output_tokens"] == 65536 and "max_tokens" not in sent[0]["extra_payload"]
    assert res["gsm8k_cot_report"]["max_gen_toks_sent"] is None
    # The result records that the official 256 was not applied to reasoning.
    assert "official cap not applied to reasoning" in res["max_output_tokens_note"]
    assert res["official_sampling"]["max_tokens_applied"] is False


def test_causalbench_no_think_samples_carry_max_tokens_10(monkeypatch):
    rows = [{"input": "Premise: A correlates with B.\nHypothesis: A causes B.", "label": 1,
             "num_variables": 2, "template": "parent"}]
    with mock.patch("datasets.load_dataset", return_value=rows):
        direct = causalbench._load_causalbench_samples()
        think = causalbench._load_causalbench_samples(enable_thinking=True)
    # Upstream's 10 is the official sampling's cap (applied by base), not a per-sample field.
    assert "max_tokens" not in direct[0][2] and "max_tokens" not in think[0][2]
    assert causalbench.OFFICIAL_SAMPLING.max_tokens == 10
    monkeypatch.setattr(causalbench, "_require_nltk", lambda: None)
    monkeypatch.setattr(causalbench, "predicted_relation", lambda text: "entailment")
    res, sent = _through_base(
        monkeypatch, lambda t: causalbench.run_causalbench("m", "http://x/v1", 1,
                                                           enable_thinking=t),
        mock.patch.object(causalbench, "_load_causalbench_samples", return_value=direct))
    assert sent[0]["max_output_tokens"] == 10
    assert res["causalbench_report"]["max_tokens_sent"] == [10]
    assert "corr2cause" in res["max_output_tokens_note"]


def test_cruxeval_direct_samples_carry_max_tokens_100():
    rows = [{"id": "sample_0", "code": "def f(x):\n    return x", "input": "1", "output": "1"}]
    with mock.patch("datasets.load_dataset", return_value=rows):
        direct = cruxeval._load_cruxeval_samples()
        cot = cruxeval._load_cruxeval_samples(enable_thinking=True)
    assert all(m["stop"] == ["[/ANSWER]"] and "max_tokens" not in m for _, _, m in direct)
    assert all("max_tokens" not in m and "stop" not in m for _, _, m in cot)
    # openai_run.py's caps are each mode's official cap (100 direct, 1000 CoT), applied by base.
    assert cruxeval._official(0.2, "pass@1").max_tokens == 100
    assert cruxeval._official(0.2, "pass@1", cot=True).max_tokens == 1000
    traces = [{"gold_answer": {"cot": False}, "max_tokens_sent": 60}]
    assert cruxeval._direct_caps_changed(traces) == [60]
    assert cruxeval._direct_caps_changed([dict(traces[0], max_tokens_sent=100)]) == []


def test_longbench_v2_code_no_think_budget_is_pred_py_128(monkeypatch):
    base.set_run_knobs(max_output_tokens=65536)
    assert longbench_v2_code._output_budget(False) == 128
    assert longbench_v2_code._output_budget(True) == 65536


def test_capped_suite_docs_name_the_sent_cap():
    for suite, needle in (("gsm8k", "`max_tokens=256`"), ("causalbench", "`max_tokens=10`"),
                          ("cruxeval", "`max_tokens=100`"),
                          ("longbench_v2_code", "`max_tokens=128`")):
        doc = _doc(suite)
        assert needle in doc and "`max_output_tokens_note`" in doc, suite
        assert "never caps" not in doc, suite
    for suite in ("gpqa", "gpqa_diamond", "multilingual_mmlu"):
        doc = _doc(suite)
        assert "sent `max_tokens=2048`" in doc and "never caps" not in doc, suite
    for module in (gsm8k, causalbench, cruxeval, longbench_v2_code, gpqa, multilingual_mmlu):
        assert "never caps" not in (module.__doc__ or ""), module.__name__
