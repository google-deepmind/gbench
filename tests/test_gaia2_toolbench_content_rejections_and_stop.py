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

"""gaia2 and toolbench: model endpoint 400/422 replies are the model's outcome, and a --thinking
gaia2 answer ends at the agent's stop sequences.

gaia2
  * a model endpoint call that failed with any HTTP status but 400 and 422 (or with none) is an
    endpoint failure, as gbench counts model endpoint replies everywhere (runners/quality.py);
  * a --thinking agent request sends no `stop` (the server would match it inside the reasoning):
    it is streamed and closed once the answer holds a stop sequence, so the answer, the server's
    generation and the measured generation time (the Time capability's clock) end there; a
    streamed reply still running at the request timeout is closed and raises LiteLLM's Timeout;
  * answers that still held a stop sequence and --thinking replies cut at max_tokens are named in
    leaderboard_comparable_reason; a run in which no agent request got a reply publishes nothing.
toolbench
  * a model request the endpoint rejected for its content (HTTP 400/422) is not an endpoint
    failure: the image records its step as `rejected`, the query stays scored, and the request is
    taken out of the harness errors and counted in `model_content_rejections`.

Docker, Gemini, the network and the model endpoint are stubbed throughout.
"""

import importlib
import json
import os
import sys
import types

import pytest

from gbench.runners import quality
from gbench.runners.eval_suites import gaia2 as G
from gbench.runners.eval_suites import toolbench as T

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "gbench", "docker"))
sys.path.insert(0, os.path.join(_REPO, "docker"))
sys.path.insert(0, _HERE)
import gaia2_run as GRUN  # noqa: E402
from test_gaia2_toolbench_judge_protocol_and_credentials import (_gaia2_run, _stub_are, _engine, tb,  # noqa: E402,F401
                                      _read, _flat)
from test_gaia2_toolbench_capabilities_and_endpoint_failures import _CHATGPT_FUNCTION  # noqa: E402
from test_delegated_harness_request_settings import _UPSTREAM_CHATGPT  # noqa: E402

STOPS = ["<end_action>", "Observation:"]


class _Exc(Exception):
    def __init__(self, status=None):
        super().__init__("x")
        if status is not None:
            self.status_code = status


# =========================================================================== #
# One rule for model endpoint statuses
# =========================================================================== #
def test_gaia2_counts_endpoint_statuses_as_gbench_does_everywhere():
    for status in range(400, 600):
        assert GRUN.endpoint_failure(_Exc(status)) == quality.model_status_is_harness_error(
            status, True), status
    for status in (405, 409, 413, 415):
        assert GRUN.endpoint_failure(_Exc(status)), status
    assert not GRUN.endpoint_failure(_Exc(400)) and not GRUN.endpoint_failure(_Exc(422))
    assert GRUN.endpoint_failure(_Exc())                       # no status: connection, timeout


def test_the_content_rejection_statuses_are_the_same_in_every_suite():
    assert (set(T.CONTENT_REJECTION_STATUSES) == set(GRUN.REQUEST_REJECTION_STATUSES)
            == set(quality._MODEL_CONTENT_REJECTION_STATUSES) == {400, 422})
    src = _read("docker", "toolbench_patch_llm.py")
    assert "_GBENCH_CONTENT_REJECTION_STATUSES = (400, 422)" in src


# =========================================================================== #
# gaia2: the --thinking answer ends at the stop sequences
# =========================================================================== #
def test_a_thinking_request_is_streamed_without_stop():
    on = GRUN.agent_settings({"GAIA2_ENABLE_THINKING": "true", "GAIA2_AGENT_MAX_TOKENS": "65536"})
    off = GRUN.agent_settings({"GAIA2_ENABLE_THINKING": "false"})
    kw = GRUN.agent_completion_kwargs(on, STOPS)
    assert "stop" not in kw and kw["stream"] is True
    assert kw["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}
    assert GRUN.client_stop(on, STOPS) and not GRUN.client_stop(off, STOPS)
    kw = GRUN.agent_completion_kwargs(off, STOPS)
    assert kw["stop"] == STOPS and "stream" not in kw
    # no stop sequences: nothing to close the stream at, so nothing is streamed
    assert "stream" not in GRUN.agent_completion_kwargs(on, [])
    assert not GRUN.client_stop(on, [])
    assert G.STREAM_OPTIONS == GRUN.STREAM_OPTIONS


def _chunk(finish=None, usage=None, **delta):
    out = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        out["usage"] = usage
    return out


def _usage(n):
    return {"prompt_tokens": 10, "completion_tokens": n, "total_tokens": 10 + n}


def test_the_stream_is_read_until_the_answer_holds_a_stop_sequence():
    chunks = iter([_chunk(role="assistant", content=""),
                   _chunk(reasoning="the next Observation: will tell", usage=_usage(1)),
                   _chunk(content="Thought: t\nAction:\n{}<end_", usage=_usage(2)),
                   _chunk(content="act", usage=_usage(3)),
                   _chunk(content="ion>\nObservation: made up", usage=_usage(4)),
                   _chunk(content=" never read", usage=_usage(5))])
    out = GRUN.read_stream(chunks, STOPS)
    # the stop string inside the reasoning is not matched; the one split over chunks is
    assert out["stop_sequence"] == "<end_action>" and out["finish_reason"] == "stop"
    assert out["content"] == "Thought: t\nAction:\n{}<end_action>\nObservation: made up"
    assert out["reasoning_chars"] == len("the next Observation: will tell")
    assert out["usage"] == _usage(4)
    assert next(chunks) == _chunk(content=" never read", usage=_usage(5))   # not consumed
    info = GRUN.stream_reply_info(out)
    assert info["completion_tokens"] == 4 and info["content_chars"] == len(out["content"])
    assert info["finish_reason"] == "stop" and info["content_null"] is False


def test_a_stream_that_ends_without_a_stop_keeps_the_servers_finish_reason():
    obj = types.SimpleNamespace
    chunks = [obj(usage=None, choices=[obj(index=0, delta=obj(content="Thought: t", reasoning=None),
                                           finish_reason=None)]),
              obj(usage=obj(prompt_tokens=3, completion_tokens=9, total_tokens=12,
                            completion_tokens_details=obj(reasoning_tokens=5)),
                  choices=[obj(index=0, delta=obj(content=None, reasoning=None),
                               finish_reason="length")])]
    out = GRUN.read_stream(chunks, STOPS)
    assert out["stop_sequence"] is None and out["finish_reason"] == "length"
    info = GRUN.stream_reply_info(out)
    assert info["completion_tokens"] == 9 and info["reasoning_tokens"] == 5
    # a stream with reasoning and no answer text: no content, as a reply whose content is null
    out = GRUN.read_stream([_chunk(reasoning="r"), _chunk(finish="stop")], STOPS)
    assert out["content"] is None and out["finish_reason"] == "stop"


def test_a_stream_past_its_deadline_raises():
    ticks = iter([0.0, 1.0, 5.0])
    chunks = [_chunk(reasoning="a"), _chunk(reasoning="b"), _chunk(reasoning="c")]
    with pytest.raises(GRUN.StreamDeadline):
        GRUN.read_stream(chunks, STOPS, deadline=2.0, clock=lambda: next(ticks))


class _FakeStream:
    def __init__(self, chunks, fail_after=None):
        self.chunks, self.fail_after, self.read, self.closed = list(chunks), fail_after, 0, False

    def __iter__(self):
        for c in self.chunks:
            if self.fail_after is not None and self.read >= self.fail_after:
                raise type("APIError", (Exception,), {})("error event in the stream")
            self.read += 1
            yield c

    def close(self):
        self.closed = True


class _FakeTimeout(Exception):
    status_code = 408

    def __init__(self, message, model, llm_provider):
        super().__init__(message)
        self.model, self.llm_provider = model, llm_provider


@pytest.fixture
def streamed(monkeypatch, tmp_path):
    monkeypatch.setattr(GRUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("GAIA2_JUDGE_PORT", "18790")
    monkeypatch.setenv("GAIA2_ENABLE_THINKING", "true")
    monkeypatch.setenv("GAIA2_AGENT_MAX_TOKENS", "65536")
    monkeypatch.setenv("GAIA2_AGENT_TIMEOUT_S", "2400")
    state = {"stream": None}

    def responder(kw, ModelResponse):
        assert kw["stream"] is True and "stop" not in kw
        return types.SimpleNamespace(completion_stream=state["stream"])

    are = _stub_are(monkeypatch, responder)
    monkeypatch.setattr(sys.modules["litellm.exceptions"], "Timeout", _FakeTimeout, raising=False)
    GRUN.install_are_hooks()
    return types.SimpleNamespace(are=are, state=state, workdir=str(tmp_path))


def test_the_hook_closes_the_stream_at_the_stop_and_measures_up_to_it(streamed, monkeypatch):
    stream = _FakeStream([_chunk(reasoning="plan the Observation: step", usage=_usage(1)),
                          _chunk(content='Thought: ok\nAction:\n{"a": True}<end_action>',
                                 usage=_usage(2)),
                          _chunk(content="Observation: made up", usage=_usage(3))])
    streamed.state["stream"] = stream
    clock = iter([100.0, 100.5, 101.5])        # start, after the reasoning chunk, the stop
    monkeypatch.setattr(GRUN.time, "monotonic", lambda: next(clock))
    agent = _engine(streamed.are, "http://172.17.0.1:5678/v1")
    res, meta = agent([{"role": "user", "content": "q"}], stop_sequences=STOPS)
    assert res == 'Thought: ok\nAction:\n{"a": true}'        # ARE's rewrite and split
    assert stream.read == 2 and stream.closed                   # closed at the stop
    assert meta["completion_duration"] == 1.5 and meta["completion_tokens"] == 2
    [call] = GRUN._read_jsonl(GRUN.AGENT_CALLS, streamed.workdir)
    assert call["streamed"] is True and call["stopped_client_side"] is True
    assert call["stop_sent"] is False and call["ran_past_stop"] is False
    assert call["finish_reason"] == "stop" and call["duration_s"] == 1.5
    summary = GRUN.agent_call_summary(streamed.workdir)
    assert summary["streamed"] == 1 and summary["stopped_client_side"] == 1
    assert GRUN._read_jsonl(GRUN.CALL_FAILURES, streamed.workdir) == []


def test_a_failure_inside_the_stream_is_recorded_as_an_endpoint_failure(streamed):
    stream = _FakeStream([_chunk(reasoning="r"), _chunk(reasoning="s")], fail_after=1)
    streamed.state["stream"] = stream
    agent = _engine(streamed.are, "http://172.17.0.1:5678/v1")
    with pytest.raises(Exception, match="error event in the stream"):
        agent([{"role": "user", "content": "q"}], stop_sequences=STOPS)
    assert stream.closed
    [fail] = GRUN._read_jsonl(GRUN.CALL_FAILURES, streamed.workdir)
    assert fail["role"] == "agent" and fail["endpoint_failure"] is True
    assert fail["error"] == "APIError" and fail["scenario_id"] == "s1"


def test_a_stream_past_the_request_timeout_is_closed_and_raises_timeout(streamed, monkeypatch):
    stream = _FakeStream([_chunk(reasoning="r")] * 5)
    streamed.state["stream"] = stream
    clock = iter([0.0, 10.0, 2500.0, 2600.0, 2700.0])
    monkeypatch.setattr(GRUN.time, "monotonic", lambda: next(clock))
    agent = _engine(streamed.are, "http://172.17.0.1:5678/v1")
    with pytest.raises(_FakeTimeout, match="request timeout ran out"):
        agent([{"role": "user", "content": "q"}], stop_sequences=STOPS)
    assert stream.closed and stream.read == 2
    [fail] = GRUN._read_jsonl(GRUN.CALL_FAILURES, streamed.workdir)
    assert fail == {**fail, "role": "agent", "error": "_FakeTimeout", "status": 408,
                    "endpoint_failure": True}
    assert GRUN._read_jsonl(GRUN.AGENT_CALLS, streamed.workdir) == []


def test_a_run_without_thinking_is_not_streamed(monkeypatch, tmp_path):
    monkeypatch.setattr(GRUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("GAIA2_JUDGE_PORT", "18790")
    monkeypatch.setenv("GAIA2_ENABLE_THINKING", "false")
    are = _stub_are(monkeypatch, lambda kw, M: M('Thought: t\nAction:\n{"a": 1}', "served"))
    GRUN.install_are_hooks()
    agent = _engine(are, "http://172.17.0.1:5678/v1")
    agent([{"role": "user", "content": "q"}], stop_sequences=STOPS)
    assert are.sent[-1]["stop"] == STOPS and "stream" not in are.sent[-1]
    [call] = GRUN._read_jsonl(GRUN.AGENT_CALLS, str(tmp_path))
    assert call["streamed"] is False and call["stopped_client_side"] is False


def test_the_launcher_records_how_the_stop_is_applied(monkeypatch, tmp_path):
    monkeypatch.setattr(GRUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("GAIA2_ENABLE_THINKING", "true")
    (tmp_path / "benchmark_stats.json").write_text(json.dumps({"statistics": {}}))
    out = GRUN._summarize()
    assert out["agent_protocol"]["stop_applied_by"] == "client_stream"
    assert out["agent_protocol"]["stream_options"] == GRUN.STREAM_OPTIONS
    monkeypatch.setenv("GAIA2_ENABLE_THINKING", "false")
    out = GRUN._summarize()
    assert out["agent_protocol"]["stop_applied_by"] == "server"
    assert out["agent_protocol"]["stream_options"] is None


def _calls(**kw):
    base = {"calls": 20, "ran_past_stop": 0, "cut_at_max_tokens": 0, "cut_without_content": 0,
            "streamed": 20, "stopped_client_side": 18}
    base.update(kw)
    return base


def test_gaia2_names_answers_past_the_stop_and_thinking_replies_cut_at_max_tokens(
        monkeypatch, tmp_path):
    res, _, _ = _gaia2_run(monkeypatch, tmp_path, thinking=True, max_output_tokens=65536,
                           summary={"agent_calls": _calls(ran_past_stop=2, cut_at_max_tokens=3,
                                                          cut_without_content=1)})
    reason = res["leaderboard_comparable_reason"]
    assert ("2 agent repl(ies) still held a stop sequence in the answer: the generation ran past "
            "the action (ARE's split dropped the rest) and the Time capability's clock counted "
            "that generation") in reason
    assert ("3 agent repl(ies) reached this run's max_tokens (65536 per turn; 1 of them with no "
            "answer), and the Time capability's clock counted the whole generation") in reason
    assert res["status"] == "success"                  # the model's own replies
    assert res["agent_calls"]["stopped_client_side"] == 18
    assert res["agent_protocol"]["stop_applied_by"] == "client_stream"
    assert res["agent_protocol"]["stream_options"] == G.STREAM_OPTIONS
    assert "closed once the answer holds a stop sequence" in res["agent_protocol"]["stop_note"]
    # without --thinking the paper's 16000 cap is the protocol: a cut reply is only counted
    res, _, _ = _gaia2_run(monkeypatch, tmp_path,
                           summary={"agent_calls": _calls(cut_at_max_tokens=3)})
    assert "max_tokens" not in res["leaderboard_comparable_reason"]
    assert res["agent_protocol"]["stop_applied_by"] == "server"
    assert res["agent_protocol"]["stream_options"] is None


def test_gaia2_publishes_nothing_when_no_agent_request_got_a_reply(monkeypatch, tmp_path):
    assert G.agent_never_answered({"agent_calls": {"calls": 0}})
    assert not G.agent_never_answered({"agent_calls": {"calls": 1}})
    assert not G.agent_never_answered({}) and not G.agent_never_answered({"agent_calls": {}})
    with pytest.raises(RuntimeError, match="not one agent request of the run got a reply"):
        _gaia2_run(monkeypatch, tmp_path, summary={"agent_calls": _calls(calls=0)})


def test_gaia2_image_docs_and_label_follow_the_launcher():
    # the image's harness-errors label moves with every change of the launcher's error handling
    assert G.HARNESS_ERRORS_VERSION == "7"
    df = _read("gbench", "docker", "gaia2.Dockerfile")
    assert f'LABEL {G.HARNESS_ERRORS_LABEL}="{G.HARNESS_ERRORS_VERSION}"' in df
    assert "'completion_stream' in inspect.signature(CustomStreamWrapper.__init__)" in df
    doc = _flat(_read("docs", "evals", "gaia2.md"))
    for needle in (f"`{G.HARNESS_ERRORS_LABEL}={G.HARNESS_ERRORS_VERSION}`",
                   "`agent_protocol.stop_applied_by`", "`stopped_client_side`", "`streamed`",
                   '`stream_options: {"include_usage": true, "continuous_usage_stats": true}`',
                   "vLLM aborts a request whose client went away",
                   "any HTTP status other than 400 and 422",
                   "not one agent request got a reply from the model endpoint",
                   "appendix B.5"):
        assert needle in doc, needle
    src = _read("gbench", "docker", "gaia2_run.py")
    assert "ENDPOINT_FAILURE_STATUSES" not in src
    # the paper's appendix B.5, quoted with the models it names (it applied them per model)
    flat = " ".join(src.replace("#", " ").split())
    assert "models that tend to continue past a single tool call (Claude, Kimi, Qwen)" in flat
    assert "ran GPT-5 without them" in flat


# =========================================================================== #
# toolbench: a 400/422 is the model's outcome
# =========================================================================== #
@pytest.fixture
def client(monkeypatch, tmp_path):
    target = tmp_path / "chatgpt_function_model.py"
    target.write_text(_UPSTREAM_CHATGPT + _CHATGPT_FUNCTION)
    patch_src = _read("docker", "toolbench_patch_llm.py").replace(
        'PATH = "/stb/toolbench/inference/LLM/chatgpt_function_model.py"', f"PATH = {str(target)!r}")
    exec(compile(patch_src, "toolbench_patch_llm.py", "exec"), {"__name__": "__main__"})
    monkeypatch.setitem(sys.modules, "termcolor", types.SimpleNamespace(colored=lambda *a, **k: ""))
    monkeypatch.syspath_prepend(os.path.join(_REPO, "docker"))
    monkeypatch.syspath_prepend(str(tmp_path))
    log = tmp_path / "model_request_failures.jsonl"
    monkeypatch.setenv("TB_MODEL_FAILURE_LOG", str(log))
    monkeypatch.setenv("TB_GROUP", "G2_category")
    monkeypatch.setenv("TB_ENABLE_THINKING", "false")
    monkeypatch.setenv("TB_REQUEST_LOG", str(tmp_path / "requests.jsonl"))
    for k in ("TB_ENDPOINT", "TB_TEMPERATURE", "TB_MAX_TOKENS", "TB_REQUEST_TIMEOUT_S"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delitem(sys.modules, "chatgpt_function_model", raising=False)
    mod = importlib.import_module("chatgpt_function_model")
    plan = []

    class BadRequestError(Exception):
        status_code = 400

    class UnprocessableEntityError(Exception):
        status_code = 422

    class InternalServerError(Exception):
        status_code = 503

    class _Resp:
        choices = [types.SimpleNamespace(finish_reason="stop", message=types.SimpleNamespace(
            content="hi", tool_calls=None))]

        def dict(self):
            return {"choices": [{"message": {"role": "assistant", "content": "hi"}}],
                    "usage": {"total_tokens": 3}}

    errors = {"400": BadRequestError("bad request: invalid tool"),
              "422": UnprocessableEntityError("unprocessable"),
              "503": InternalServerError("unavailable")}

    def create(**kw):
        step = plan.pop(0)
        if step in errors:
            raise errors[step]
        return _Resp()

    class _OpenAI:
        def __init__(self, base_url=None, api_key=None):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    monkeypatch.setattr(mod, "OpenAI", _OpenAI)
    llm = mod.ChatGPTFunction(base_url="http://127.0.0.1:1/v1")
    llm._gbench_query_id = 77
    llm.change_messages([{"role": "user", "content": "q"}])
    return types.SimpleNamespace(mod=mod, llm=llm, plan=plan, log=log)


def _records(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def test_a_step_rejected_on_every_try_is_the_models_outcome(client):
    client.plan[:] = ["400"] * 6
    msg, code, _ = client.llm.parse([], 0)
    assert code == -1 and "invalid tool" in msg["content"]     # upstream passes the error on
    [rec] = _records(client.log)
    assert rec == {"group": "G2_category", "query_id": "77", "tries": 6, "endpoint_failures": 0,
                   "content_rejections": 6, "unrecovered": False, "rejected": True,
                   "errors": ["BadRequestError 400"]}
    # the request settings still count the request (the host splits it out of the harness errors)
    assert client.mod._GBENCH_POLICY.failed_by_error == {"BadRequestError 400": 6}


def test_rejections_and_endpoint_failures_in_one_step(client):
    client.plan[:] = ["422", "ok"]                              # a rejection, then a reply
    assert client.llm.parse([], 0)[1] == 0
    client.plan[:] = ["503"] + ["400"] * 5                      # ends on a rejection
    assert client.llm.parse([], 0)[1] == -1
    client.plan[:] = ["400"] + ["503"] * 5                      # ends on an endpoint failure
    assert client.llm.parse([], 0)[1] == -1
    answered, rejected, lost = _records(client.log)
    assert answered["content_rejections"] == 1 and answered["endpoint_failures"] == 0
    assert not answered["rejected"] and not answered["unrecovered"]
    assert rejected["endpoint_failures"] == 1 and rejected["content_rejections"] == 5
    assert rejected["rejected"] is True and rejected["unrecovered"] is False
    assert lost["unrecovered"] is True and lost["rejected"] is False
    assert lost["errors"] == ["BadRequestError 400", "InternalServerError 503"]
    s = T.summarize_model_failures([answered, rejected, lost])
    assert (s["steps_unrecovered"], s["steps_rejected"], s["steps_recovered"]) == (1, 1, 0)
    assert s["failed_requests"] == 6 and s["content_rejections"] == 7
    slot = s["per_query"][("G2_category", "77")]
    assert slot["unrecovered_steps"] == 1 and slot["rejected_steps"] == 1
    assert slot["content_rejections"] == 7


def test_a_rejected_step_leaves_its_query_scored():
    recs = [{"group": "G1_tool", "query_id": "1", "rejected": True, "unrecovered": False,
             "endpoint_failures": 0, "content_rejections": 6, "errors": ["BadRequestError 400"]},
            {"group": "G1_tool", "query_id": "2", "unrecovered": True, "rejected": False,
             "endpoint_failures": 6, "content_rejections": 0},
            {"group": "", "query_id": "", "rejected": True, "content_rejections": 6}]
    s = T.summarize_model_failures(recs)
    assert s["steps_rejected"] == 2 and s["unattributed_unrecovered"] == 0
    assert s["unattributed_recovered"] == 0
    got = T.harness_error_queries({"G1_tool": ["1", "2"]}, {"G1_tool": {"1": {}, "2": {}}},
                                  {}, {}, s["per_query"])
    assert got == {"G1_tool": {"2": ["model_endpoint"]}}


def test_content_rejections_are_split_out_of_the_harness_errors():
    replies = {"counts": {"replies": 40, "failed_requests": 9, "timed_out": 1},
               "failed_by_error": {"APITimeoutError": 1, "BadRequestError 400": 6,
                                   "InternalServerError 503": 1,
                                   "UnprocessableEntityError 422": 1},
               "settings": {"enable_thinking": False}}
    harness, rej = T.split_content_rejections(replies)
    assert rej == {"requests": 7, "by_error": {"BadRequestError 400": 6,
                                               "UnprocessableEntityError 422": 1}}
    assert harness["counts"]["failed_requests"] == 2 and harness["counts"]["replies"] == 40
    assert harness["failed_by_error"] == {"APITimeoutError": 1, "InternalServerError 503": 1}
    assert replies["counts"]["failed_requests"] == 9              # the record is not changed
    again, none = T.split_content_rejections(harness)             # nothing left to split
    assert again is harness and none == {"requests": 0, "by_error": {}}
    assert T.split_content_rejections(None) == (None, {"requests": 0, "by_error": {}})


def _with_request_log(tb, monkeypatch, record):  # noqa: F811
    real = T.subprocess.run

    def run(cmd, *a, **kw):
        out = real(cmd, *a, **kw)
        if cmd[:2] == ["docker", "run"]:
            with open(os.path.join(tb.state["workdir"], T.REQUEST_LOG), "w") as fh:
                fh.write(json.dumps(record) + "\n")
        return out
    monkeypatch.setattr(T.subprocess, "run", run)


# what the image's request settings report for a no-think run: gbench's temperature 0.0
_SETTINGS = {"enable_thinking": False, "max_tokens": None, "temperature": 0.0,
             "request_timeout_s": None}


def test_toolbench_scores_a_query_whose_step_the_endpoint_rejected(tb, monkeypatch):  # noqa: F811
    tb.state["model_failures"] = [
        {"group": "G1_tool", "query_id": "1", "tries": 6, "endpoint_failures": 0,
         "content_rejections": 6, "unrecovered": False, "rejected": True,
         "errors": ["BadRequestError 400"]}]
    _with_request_log(tb, monkeypatch, {
        "settings": _SETTINGS, "counts": {"replies": 30, "failed_requests": 6},
        "failed_by_error": {"BadRequestError 400": 6}, "max_tokens_sent": [1024]})
    res = T.run_toolbench("m", "http://127.0.0.1:8000/v1", 2)
    assert res["status"] == "success" and res["harness_errors"] == []
    assert res["n_harness_error_queries"] == 0 and res["queries_by_group"]["G1_tool"] == 2
    assert res["model_content_rejections"] == {"requests": 6,
                                               "by_error": {"BadRequestError 400": 6}}
    assert res["model_request_failures"]["steps_rejected"] == 1
    reason = res["leaderboard_comparable_reason"]
    assert ("6 model request(s) were rejected by the endpoint for their content "
            "(BadRequestError 400 x6): the model's outcome, not harness errors") in reason
    assert "1 DFSDT step(s) ended on such a rejection, their queries scored" in reason
    assert "ended in an error of the served endpoint" not in reason
    traces = {(t["group"], t["query_id"]): t for t in res["sample_traces"]}
    assert traces[("G1_tool", "1")]["model_request_failures"]["rejected_steps"] == 1
    assert "sopr_passes" in traces[("G1_tool", "1")]


def test_toolbench_still_names_endpoint_failures_next_to_rejections(tb, monkeypatch):  # noqa: F811
    _with_request_log(tb, monkeypatch, {
        "settings": _SETTINGS, "counts": {"replies": 30, "failed_requests": 3},
        "failed_by_error": {"BadRequestError 400": 2, "InternalServerError 503": 1},
        "max_tokens_sent": [1024]})
    res = T.run_toolbench("m", "http://127.0.0.1:8000/v1", 2)
    assert res["status"] == "completed_with_errors"
    [err] = [e for e in res["harness_errors"] if "by_error" in e]
    assert err["failed_requests"] == 1 and err["by_error"] == {"InternalServerError 503": 1}
    # this fixture's per-step records hold no endpoint failure: the request log's 503 is named
    [mismatch] = [e for e in res["harness_errors"] if "request_log_failed_requests" in e]
    assert mismatch == {**mismatch, "request_log_failed_requests": 1,
                        "step_records_failed_requests": 0}
    assert res["model_content_rejections"]["requests"] == 2
    assert res["model_replies"]["failed_requests"] == 3          # the image's own record


def test_toolbench_publishes_nothing_when_every_request_was_rejected(tb, monkeypatch):  # noqa: F811
    _with_request_log(tb, monkeypatch, {
        "settings": _SETTINGS, "counts": {"replies": 0, "failed_requests": 12},
        "failed_by_error": {"BadRequestError 400": 12}, "max_tokens_sent": []})
    with pytest.raises(RuntimeError, match="12 of them rejected for their content "
                                           r"\(BadRequestError 400 x12\), although no request "
                                           "carried text the model had generated"):
        T.run_toolbench("m", "http://127.0.0.1:8000/v1", 2)


def test_toolbench_image_docs_and_label_follow_the_client():
    assert T.HARNESS_VERSION == "5"
    df = _read("docker", "toolbench.Dockerfile")
    assert f'LABEL {T.HARNESS_LABEL}="{T.HARNESS_VERSION}"' in df
    doc = _flat(_read("docs", "evals", "toolbench.md"))
    for needle in (f"`{T.HARNESS_LABEL}={T.HARNESS_VERSION}`", "`model_content_rejections`",
                   "`steps_rejected`", "`content_rejections`", "HTTP 400 or 422",
                   "When not one request of the run got a reply, every outcome is a harness "
                   "error"):
        assert needle in doc, needle
    # the shared request settings stay byte-identical to mcp_bench's copy
    assert _read("docker", "toolbench_request_policy.py") == _read(
        "gbench", "docker", "mcp_bench_request_policy.py")
