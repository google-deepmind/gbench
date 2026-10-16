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

"""Knowledge suites: upstream truncation, Python 3.10 support and the token-limit policy.

* longbench_v2_code truncates exactly as LongBench pred.py does (max_len = the model's context
  window: 120,000 for a window up to 131,072, else the window minus the official cap minus 64,
  whenever the served window holds it plus the output budget; a smaller window only shortens
  it), and its leaderboard_comparable flag follows from what the run actually did;
* mmlu_pro bounds each streamed try with asyncio.wait_for (the same 3600 s bound as upstream's
  asyncio.timeout, which Python 3.10 lacks), fails fast on client errors no try can fix, and
  applies evaluate_from_apiX.py's max_tokens through base's token-limit policy;
* every suite with an official cap declares it in base.OfficialSampling (gpqa family 2048,
  gsm8k 256, causalbench 10, cruxeval 100/1000, longbench_v2_code 128, mmlu_pro 32768,
  mmlu_redux 1) or declares that none is sent (lab_bench): exact without thinking, not applied
  to reasoning with it.

No network and no model server: requests are faked in process, and tiktoken is replaced by a
word-count encoder.
"""

import ast
import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from gbench.runners.eval_suites import (base, causalbench, cruxeval, gpqa, gpqa_diamond,
                                        gsm8k, lab_bench, longbench_v2_code, mmlu_pro,
                                        mmlu_redux, multilingual_mmlu)

REPO = Path(__file__).resolve().parent.parent


def _doc(suite):
    """The suite's doc with whitespace runs folded, so a phrase may wrap across lines."""
    return " ".join((REPO / "docs" / "evals" / f"{suite}.md").read_text(encoding="utf-8").split())


# --------------------------------------------------------------------------- #
# longbench_v2_code: pred.py's max_len and a flag computed from the run
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("window,budget,expected", [
    (131072, 128, 120000),        # 128K window: upstream's 120,000 fits
    (120192, 128, 120000),        # the smallest window that holds 120,000 + 128 + 64
    (120191, 128, 119999),        # one token less: only now shortened
    (1048576, 128, 1047488),      # a larger window: upstream's max_len is the window's, one for
                                  # both settings (window - 1,024 - 64)
    (262144, 65536, 196544),      # a thinking budget on a larger window
    (131072, 65536, 65472),       # a thinking budget the window cannot hold with 120,000
])
def test_longbench_prompt_budget_is_pred_py_max_len_when_it_fits(window, budget, expected):
    assert longbench_v2_code._prompt_budget(window, budget) == expected


def test_longbench_budget_comes_from_base_token_policy():
    assert longbench_v2_code.OFFICIAL_SAMPLING.max_tokens == 128
    assert longbench_v2_code._output_budget(False) == 128
    base.set_run_knobs(max_output_tokens=65536)
    assert longbench_v2_code._output_budget(True) == 65536
    assert longbench_v2_code._output_budget(False) == 128
    base._RUN_KNOBS.pop("max_output_tokens", None)
    assert longbench_v2_code._output_budget(True) == base.THINKING_MIN_OUTPUT_TOKENS


def _lb_samples(tokens_before):
    out = []
    for i, n in enumerate(tokens_before):
        out.append(([{"role": "user", "content": f"doc {i}"}], "A",
                    {"category": "Code Repository Understanding", "difficulty": "easy",
                     "length": "long", "prompt_tokens_before_truncation": n,
                     "prompt_truncated": n is not None and n > 0}))
    return out


class _LbCharTok:
    """One token per character: the w/ CoT answer prompt is tokenized for pred.py's truncation."""
    def encode(self, s):
        return list(s)


def _lb_cot_meta(meta, cot):
    """A w/ CoT sample's meta also carries its 0shot_cot_ans.txt fill, up to $COT$."""
    return {**meta, "cot_answer_prompt": "Q? $COT$"} if cot else meta


def _lb_run(monkeypatch, window, samples, thinking=False, all_domains=True, suite_result=None,
            **kwargs):
    """run_longbench_v2_code with a fake loader and run_eval_suite; returns (result, seen):
    `seen` holds the non-CoT (headline) setting's loader and suite call, `seen["calls"]` every
    suite call (non-CoT, then the w/ CoT setting's CoT query and answer query)."""
    monkeypatch.setenv("GBENCH_LONGBENCH_V2_CODE_WINDOW", str(window))
    if all_domains:
        monkeypatch.setenv("GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS", "1")
    else:
        monkeypatch.delenv("GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS", raising=False)
    seen = {}

    def fake_loader(limit=None, tokenizer=None, max_prompt_tokens=None, cot=False):
        if not cot:
            seen["max_prompt_tokens"] = max_prompt_tokens
        # What the loader records: truncated when longer than this run's max_len.
        return [(m, g, _lb_cot_meta({**meta, "prompt_truncated": bool(
                    meta["prompt_tokens_before_truncation"]
                    and meta["prompt_tokens_before_truncation"] > max_prompt_tokens)}, cot))
                for m, g, meta in samples]

    def fake_suite(**kw):
        seen.setdefault("calls", []).append(kw)
        if len(seen["calls"]) == 1:
            seen.update(kw)
        res = dict(suite_result or {
            "temperature": 0.1, "failed_requests": 0,
            "sample_traces": [{"response_text": "The correct answer is (A)",
                               "is_correct": True, "extra_payload": {},
                               "max_tokens_sent": None if thinking else 128}
                              for _ in samples]})
        # As base does: each trace carries its sample's meta, and a request sent 128 here stands
        # for one sent this query's official cap (1024 for the CoT query).
        metas = [s[2] for s in kw["samples"] if s is not None]
        cap = kw["official_sampling"].max_tokens
        res["sample_traces"] = [
            {**t, "extra_payload": {**metas[i], **(t.get("extra_payload") or {})},
             "max_tokens_sent": (cap if t.get("max_tokens_sent") == 128
                                 else t.get("max_tokens_sent"))}
            for i, t in enumerate(res["sample_traces"])]
        return res
    monkeypatch.setattr(longbench_v2_code, "_load_tokenizer", lambda name: _LbCharTok())
    monkeypatch.setattr(longbench_v2_code, "_load_longbench_v2_code_samples", fake_loader)
    monkeypatch.setattr(longbench_v2_code, "run_eval_suite", fake_suite)
    res = longbench_v2_code.run_longbench_v2_code("m", "http://x/v1", 1,
                                                  enable_thinking=thinking, **kwargs)
    return res, seen


def test_longbench_full_no_think_run_on_a_128k_window_is_comparable(monkeypatch):
    res, seen = _lb_run(monkeypatch, 131072, _lb_samples([300000, 50000]))
    assert seen["max_prompt_tokens"] == 120000
    assert seen["official_sampling"] is longbench_v2_code.OFFICIAL_SAMPLING
    assert all("max_tokens" not in meta for _, _, meta in seen["samples"])
    report = res["longbench_v2_code_report"]
    assert report["max_prompt_tokens"] == 120000 and report["upstream_max_prompt_tokens"] == 120000
    assert report["prompts_truncated"] == 1 and report["prompts_cut_shorter_than_upstream"] == 0
    assert res["leaderboard_comparable"] is True
    assert "leaderboard_comparable_reason" not in res


def test_longbench_flag_counts_only_prompts_actually_cut_shorter(monkeypatch):
    # A window one token short of 120,192: max_len 119,999.
    res, _ = _lb_run(monkeypatch, 120191, _lb_samples([50000, 90000]))
    # No prompt is longer than 119,999, so nothing was cut differently from upstream.
    assert res["longbench_v2_code_report"]["max_prompt_tokens"] == 119999
    assert res["longbench_v2_code_report"]["prompts_cut_shorter_than_upstream"] == 0
    assert res["leaderboard_comparable"] is True
    res, _ = _lb_run(monkeypatch, 120191, _lb_samples([50000, 400000]))
    assert res["longbench_v2_code_report"]["prompts_cut_shorter_than_upstream"] == 1
    assert res["leaderboard_comparable"] is False
    assert "1 prompt(s) middle-truncated to 119,999 tokens" in res["leaderboard_comparable_reason"]


@pytest.mark.parametrize("case", ["code_only", "thinking", "temperature", "failed", "limit",
                                  "clamped", "short_window"])
def test_longbench_non_comparable_always_names_a_reason(monkeypatch, case):
    samples = _lb_samples([400000])
    traces = [{"response_text": "The correct answer is (A)", "is_correct": True,
               "extra_payload": {}, "max_tokens_sent": 128}]
    result = {"temperature": 0.1, "failed_requests": 0, "sample_traces": traces}
    kwargs, window, thinking, all_domains = {}, 131072, False, True
    if case == "code_only":
        all_domains = False
    elif case == "thinking":
        thinking = True
        base.set_run_knobs(max_output_tokens=4096)
    elif case == "temperature":
        result["temperature"] = 0.7
    elif case == "failed":
        result["failed_requests"] = 1
    elif case == "limit":
        kwargs["limit"] = 1
    elif case == "clamped":
        traces[0]["max_tokens_sent"] = 100
    elif case == "short_window":
        window = 65536
    res, _ = _lb_run(monkeypatch, window, samples, thinking=thinking, all_domains=all_domains,
                     suite_result=result, **kwargs)
    assert res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"].strip()


def test_longbench_through_base_sends_128_and_records_the_policy(monkeypatch):
    sent = []

    async def fake_send(**kw):
        sent.append(kw)
        return base.Reply("The correct answer is (A)", None, "length", completion_tokens=128)
    monkeypatch.setattr(base, "_send_single_request", fake_send)
    base.set_run_knobs(max_output_tokens=65536)
    samples = _lb_samples([10])
    monkeypatch.setenv("GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS", "1")
    monkeypatch.setenv("GBENCH_LONGBENCH_V2_CODE_WINDOW", "131072")
    # pred.py's temperature, set explicitly (gbench's no-think 0.0 is not comparable)
    monkeypatch.setenv("GBENCH_LONGBENCH_V2_CODE_TEMPERATURE", "0.1")
    monkeypatch.setattr(longbench_v2_code, "_load_tokenizer", lambda name: _LbCharTok())
    monkeypatch.setattr(longbench_v2_code, "_load_longbench_v2_code_samples",
                        lambda cot=False, **kw: [(m, g, _lb_cot_meta(meta, cot))
                                                 for m, g, meta in samples])
    res = longbench_v2_code.run_longbench_v2_code("m", "http://x/v1", 1)
    assert sent[0]["max_output_tokens"] == 128 and "max_tokens" not in sent[0]["extra_payload"]
    assert res["official_sampling"]["max_tokens"] == 128
    assert res["official_sampling"]["max_tokens_applied"] is True
    assert res["max_output_tokens_note"].startswith("--max-output-tokens 65536 not applied")
    # A reply that stops at pred.py's 128 is the protocol's outcome, not a truncation problem.
    assert res["replies_at_official_max_tokens"] == 1 and res["status"] == "success"
    assert res["leaderboard_comparable"] is True
    sent.clear()
    res = longbench_v2_code.run_longbench_v2_code("m", "http://x/v1", 1, enable_thinking=True)
    assert sent[0]["max_output_tokens"] == 65536
    assert res["official_sampling"]["max_tokens_applied"] is False
    assert base.REASONING_CAP_NOTE in res["max_output_tokens_note"]
    assert res["longbench_v2_code_report"]["max_prompt_tokens"] == 131072 - 65536 - 64
    assert "--thinking run" in res["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# mmlu_pro: fakes
# --------------------------------------------------------------------------- #
class _WordEncoding:
    """tiktoken stand-in: one token per whitespace-separated word."""

    @staticmethod
    def encode(text):
        return text.split()


def _chunk(content=None, reasoning=None, finish=None):
    delta = SimpleNamespace(content=content)
    if reasoning is not None:
        delta.reasoning = reasoning
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(delta=delta,
                                                                finish_reason=finish)])


def _usage():
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=5,
                                                 total_tokens=8), choices=[])


def _reply(text):
    return [_chunk(content=w + " ") for w in text.split()] + [_chunk(finish="stop"), _usage()]


class _Stream:
    def __init__(self, chunks, hang=False):
        self._chunks = list(chunks)
        self._hang = hang
        self.closed = False

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._chunks:
            yield c
        if self._hang:
            await asyncio.sleep(3600)

    async def close(self):
        self.closed = True


class _Client:
    def __init__(self, script):
        self.script = script
        self.calls = []
        self.streams = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **request):
        self.calls.append(request)
        out = self.script(len(self.calls) - 1, request)
        if isinstance(out, BaseException):
            raise out
        stream = out if isinstance(out, _Stream) else _Stream(out)
        self.streams.append(stream)
        return stream

    async def close(self):
        pass


@pytest.fixture
def pro(monkeypatch):
    """No waits, and tiktoken replaced by a word counter (no download)."""
    import tiktoken
    monkeypatch.setattr(mmlu_pro, "START_SPACING_S", 0.0)
    monkeypatch.setattr(mmlu_pro, "ERROR_RETRY_WAIT_S", 0.0)
    monkeypatch.setattr(mmlu_pro, "_token_retry_wait", lambda n: 0.0)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: _WordEncoding())


def _pro_samples(n):
    return [([{"role": "user", "content": f"Question {i}"}], "B",
             {"category": "math", "question_id": 100 + i, "row": i}) for i in range(n)]


def _run_pro(monkeypatch, client, samples, **kwargs):
    monkeypatch.setattr("openai.AsyncOpenAI", lambda **kw: client)
    with mock.patch.object(mmlu_pro, "_load_mmlu_pro_samples", return_value=samples):
        return mmlu_pro.run_mmlu_pro("m", "http://127.0.0.1:9/v1", 1, **kwargs)


# --------------------------------------------------------------------------- #
# mmlu_pro: Python 3.10 and the per-try bound
# --------------------------------------------------------------------------- #
def test_mmlu_pro_code_never_calls_asyncio_timeout():
    """asyncio.timeout is Python 3.11+; gbench declares >=3.10."""
    tree = ast.parse(inspect.getsource(mmlu_pro))
    uses = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "timeout"
            and isinstance(n.value, ast.Name) and n.value.id == "asyncio"]
    assert not uses
    assert "asyncio.wait_for(" in inspect.getsource(mmlu_pro._stream_once)


def test_mmlu_pro_runs_where_asyncio_has_no_timeout(monkeypatch, pro):
    """Python 3.10's asyncio has no timeout(); the run must still answer every question."""
    monkeypatch.delattr(asyncio, "timeout", raising=False)
    client = _Client(lambda i, req: _reply("The answer is (B)."))
    res = _run_pro(monkeypatch, client, _pro_samples(3))
    assert res["status"] == "success" and res["accuracy"] == 100.0
    assert [t["status"] for t in res["sample_traces"]] == ["OK"] * 3
    assert all(t["error_tries"] == 0 for t in res["sample_traces"])
    assert len(client.calls) == 3


@pytest.mark.parametrize("without_asyncio_timeout", [False, True])
def test_mmlu_pro_try_bound_raises_timeout_and_closes_the_stream(monkeypatch, pro,
                                                                 without_asyncio_timeout):
    if without_asyncio_timeout:
        monkeypatch.delattr(asyncio, "timeout", raising=False)
    monkeypatch.setattr(mmlu_pro, "REQUEST_TIMEOUT_S", 0.2)
    stream = _Stream([_chunk(content="thinking ")], hang=True)
    client = _Client(lambda i, req: stream)
    request = {"model": "m", "max_tokens": 100, "messages": [{"role": "user", "content": "q"}]}

    async def go():
        await mmlu_pro._stream_once(client, request, "q", _WordEncoding(), mmlu_pro._Pacer(), 0)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(go())
    assert stream.closed is True


def test_mmlu_pro_a_timed_out_try_is_asked_again_like_any_failed_try(monkeypatch, pro):
    monkeypatch.setattr(mmlu_pro, "REQUEST_TIMEOUT_S", 0.2)
    monkeypatch.setattr(mmlu_pro, "ERROR_TRIES", 3)

    def script(i, req):
        return _Stream([], hang=True) if i == 0 else _reply("The answer is (B).")
    client = _Client(script)
    res = _run_pro(monkeypatch, client, _pro_samples(1))
    trace = res["sample_traces"][0]
    assert trace["status"] == "OK" and trace["error_tries"] == 1 and len(client.calls) == 2
    assert client.streams[0].closed is True
    # Every try times out: FAILED after ERROR_TRIES, as for any other failure.
    client = _Client(lambda i, req: _Stream([], hang=True))
    res = _run_pro(monkeypatch, client, _pro_samples(1))
    trace = res["sample_traces"][0]
    assert trace["status"] == "FAILED" and trace["error"].startswith("TimeoutError")
    assert len(client.calls) == 3


@pytest.mark.parametrize("exc", [TypeError("unexpected keyword argument 'stream_options'"),
                                 AttributeError("module 'x' has no attribute 'y'"),
                                 ImportError("no module"), NameError("name 'z'")])
def test_mmlu_pro_client_errors_stop_the_suite_at_once(monkeypatch, pro, exc):
    client = _Client(lambda i, req: exc)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED") as info:
        _run_pro(monkeypatch, client, _pro_samples(5))
    assert type(exc).__name__ in str(info.value)
    assert len(client.calls) == 1        # not ERROR_TRIES tries per question


# --------------------------------------------------------------------------- #
# mmlu_pro: the token-limit policy
# --------------------------------------------------------------------------- #
def test_mmlu_pro_declares_its_cap_for_both_modes():
    official = mmlu_pro.OFFICIAL_SAMPLING
    assert official.max_tokens == 32768 and official.reasoning.max_tokens == 32768
    assert official.reasoning.temperature == 0.1 and official.reasoning.params == {"top_p": 0.95}


@pytest.mark.parametrize("thinking,knob,sent,comparable", [
    (False, None, 32768, True),
    (False, 65536, 32768, True),       # not applied without thinking
    (True, None, 32768, False),        # the upstream's reasoning-model setting (at 1.0) ...
    (True, 16384, 32768, False),       # never below the upstream's setting ...
    (True, 65536, 65536, False),       # ... nor below --max-output-tokens
])
def test_mmlu_pro_max_tokens_follow_the_token_policy(monkeypatch, pro, thinking, knob, sent,
                                                     comparable):
    if knob:
        base.set_run_knobs(max_output_tokens=knob)
    if not thinking:     # upstream's temperature, set explicitly: only the cap is under test
        monkeypatch.setenv("GBENCH_MMLU_PRO_TEMPERATURE", "0.1")
    client = _Client(lambda i, req: _reply("The answer is (B)."))
    res = _run_pro(monkeypatch, client, _pro_samples(1), enable_thinking=thinking)
    assert [c["max_tokens"] for c in client.calls] == [sent]
    assert res["max_tokens_sent"] == [sent] and res["harness"]["max_tokens"] == sent
    assert res["sample_traces"][0]["max_tokens_sent"] == sent
    assert res["leaderboard_comparable"] is comparable
    assert res["official_sampling"]["max_tokens"] == 32768
    # applied = the upstream's 32768 is what went on the wire
    assert res["official_sampling"]["max_tokens_applied"] is (sent == 32768)
    assert bool(res["official_sampling"].get("reasoning_settings")) is thinking
    if knob:
        assert res["max_output_tokens_note"]
    if sent != 32768:
        assert f"max_tokens={sent}" in res["leaderboard_comparable_reason"]
    # a --thinking run samples at 1.0, not upstream's 0.1 (the only cause at sent == 32768)
    assert [c["temperature"] for c in client.calls] == [1.0 if thinking else 0.1]
    if thinking:
        assert "sampled at temperature 1.0 (thinking;" in res["leaderboard_comparable_reason"]


def test_mmlu_pro_client_count_limit_is_the_max_tokens_sent(monkeypatch, pro):
    """The client's token count stops at the max_tokens actually sent (upstream's --max_tokens)."""
    official = mmlu_pro.OFFICIAL_SAMPLING
    monkeypatch.setattr(mmlu_pro, "OFFICIAL_SAMPLING", official._replace(
        max_tokens=20, reasoning=official.reasoning._replace(max_tokens=20)))
    base.set_run_knobs(max_output_tokens=60)
    thirty_words = _reply("word " * 26 + "The answer is (B).")
    # No-think: capped at 20 by the client count, 3 tries, MaxTokensExceeded.
    client = _Client(lambda i, req: thirty_words)
    res = _run_pro(monkeypatch, client, _pro_samples(1))
    assert res["sample_traces"][0]["status"] == "MAX_TOKENS_EXCEEDED"
    assert all(c["max_tokens"] == 20 for c in client.calls) and len(client.calls) == 3
    # Thinking: the run's 60 is sent and counted against, so the reply fits.
    client = _Client(lambda i, req: thirty_words)
    res = _run_pro(monkeypatch, client, _pro_samples(1), enable_thinking=True)
    assert res["sample_traces"][0]["status"] == "OK" and res["accuracy"] == 100.0
    assert [c["max_tokens"] for c in client.calls] == [60]


def test_mmlu_pro_doc_matches_the_code():
    doc = _doc("mmlu_pro")
    for needle in ("asyncio.wait_for", "3.10", "never capped below", "`evals` extra",
                   "`reasoning` field", "TypeError"):
        assert needle in doc, needle
    assert "counted but not kept" not in doc
    assert "counted but not kept" not in (mmlu_pro.__doc__ or "")


# --------------------------------------------------------------------------- #
# gpqa, gpqa_diamond, multilingual_mmlu: simple-evals' samplers, one declaration
# --------------------------------------------------------------------------- #
def _gpqa_rows(n):
    return [{"Question": f"Q{i}?", "Correct Answer": f"right{i}", "Incorrect Answer 1": "w1",
             "Incorrect Answer 2": "w2", "Incorrect Answer 3": "w3", "Subdomain": "Physics"}
            for i in range(n)]


def test_simple_evals_sampling_declares_both_modes():
    full = gpqa.SIMPLE_EVALS_SAMPLING
    assert full.max_tokens == 2048 and full.temperature == 0.5
    # No reasoning block: a served model takes the chat sampler with --thinking too (the
    # o-series samplers are keyed on OpenAI model names); base does not apply the 2048 there.
    assert full.reasoning is None and full.for_mode(True) is full
    assert not hasattr(gpqa, "SIMPLE_EVALS_REASONING_SAMPLING")
    assert gpqa.SIMPLE_EVALS_CHAT_SAMPLING.max_tokens is None
    assert full == gpqa.SIMPLE_EVALS_CHAT_SAMPLING._replace(max_tokens=2048)
    # the cap is declared once, there: no per-sample helper is left
    assert not hasattr(gpqa, "simple_evals_sampling") and not hasattr(gpqa, "simple_evals_meta")


@pytest.mark.parametrize("module,run", [(gpqa, "run_gpqa"), (gpqa_diamond, "run_gpqa_diamond")])
def test_gpqa_through_base_applies_the_policy_per_mode(monkeypatch, module, run):
    sent = []

    async def fake_send(**kw):
        sent.append(kw)
        return base.Reply("Answer: A", None, "length", completion_tokens=2048)
    monkeypatch.setattr(base, "_send_single_request", fake_send)
    base.set_run_knobs(max_output_tokens=65536)
    # the sampler's temperature, set explicitly: only the cap is under test
    monkeypatch.setenv(base.temperature_env_var(module.__name__.rsplit(".", 1)[-1]), "0.5")
    with mock.patch.object(gpqa, "_load_gpqa_rows", return_value=_gpqa_rows(1)):
        res = getattr(module, run)("m", "http://x/v1", 1)
    assert [kw["max_output_tokens"] for kw in sent] == [2048] * 10
    assert all("max_tokens" not in kw["extra_payload"] for kw in sent)
    assert res["official_sampling"]["max_tokens"] == 2048
    assert res["official_sampling"]["max_tokens_applied"] is True
    # Replies stopping at the sampler's cap are its outcome: counted, not a failed run.
    assert res["replies_at_official_max_tokens"] == 10 and res["status"] == "success"
    assert res["replies_at_sampler_max_tokens"] == 10 and res["leaderboard_comparable"] is True
    sent.clear()
    monkeypatch.delenv(base.temperature_env_var(module.__name__.rsplit(".", 1)[-1]))
    with mock.patch.object(gpqa, "_load_gpqa_rows", return_value=_gpqa_rows(1)):
        res = getattr(module, run)("m", "http://x/v1", 1, enable_thinking=True)
    # The chat sampler with --thinking too, at gbench's temperature 1.0 and the run's budget (the
    # 2048 is not applied to reasoning); a reply cut off at that budget makes the run
    # non-comparable, and so does the temperature.
    assert all(kw["max_output_tokens"] == 65536 and kw["temperature"] == 1.0 for kw in sent)
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert "temperature 1.0 (thinking;" in res["leaderboard_comparable_reason"]
    assert "reasoning_settings" not in res["official_sampling"]
    assert "send_max_tokens" not in res["official_sampling"]
    assert (res["official_sampling"]["max_tokens_applied"],
            res["official_sampling"]["max_tokens_sent"]) == (False, 65536)
    assert res["max_output_tokens_note"].startswith("official cap not applied to reasoning")
    assert res["replies_at_sampler_max_tokens"] == 0 and res["leaderboard_comparable"] is False
    assert "10 repl(ies) stopped at the run's token budget" in res["leaderboard_comparable_reason"]


def test_multilingual_mmlu_passes_the_one_declaration(monkeypatch):
    seen = {}
    rows = [([{"role": "user", "content": "q"}], "A", {"category": "fr"})]
    with mock.patch.object(multilingual_mmlu, "_load_multilingual_mmlu_samples",
                           return_value=rows), \
            mock.patch.object(multilingual_mmlu, "run_eval_suite",
                              side_effect=lambda **kw: seen.update(kw) or {"sample_traces": []}):
        multilingual_mmlu.run_multilingual_mmlu("m", "http://x/v1", 1)
    assert seen["official_sampling"] is gpqa.SIMPLE_EVALS_SAMPLING


# --------------------------------------------------------------------------- #
# single-cap suites through base: exact without thinking, the run budget with it
# --------------------------------------------------------------------------- #
def _through_base(monkeypatch, fn, thinking, finish="stop"):
    sent = []

    async def fake_send(**kw):
        sent.append(kw)
        return base.Reply("The answer is 3.", None, finish, completion_tokens=5)
    monkeypatch.setattr(base, "_send_single_request", fake_send)
    base.set_run_knobs(max_output_tokens=65536)
    return fn(thinking), sent


@pytest.mark.parametrize("thinking", [False, True])
def test_gsm8k_cap_follows_the_token_policy(monkeypatch, thinking):
    samples = [(gsm8k.fewshot_messages("q"), "3", {} if thinking else {"stop": list(gsm8k.UNTIL)})]
    with mock.patch.object(gsm8k, "_load_gsm8k_samples", return_value=samples):
        res, sent = _through_base(
            monkeypatch, lambda t: gsm8k.run_gsm8k("m", "http://x/v1", 1, enable_thinking=t),
            thinking, finish="length")
    assert sent[0]["max_output_tokens"] == (65536 if thinking else 256)
    assert res["official_sampling"]["max_tokens"] == 256
    assert res["official_sampling"]["max_tokens_applied"] is (not thinking)
    if thinking:
        assert base.REASONING_CAP_NOTE in res["max_output_tokens_note"]
    else:
        # A reply that stops at 256 is lm-eval's outcome, not a measurement problem.
        assert res["replies_at_official_max_tokens"] == 1 and res["status"] == "success"
    reason = res.get("leaderboard_comparable_reason") or ""
    assert "is sent no cap" not in reason


@pytest.mark.parametrize("thinking", [False, True])
def test_causalbench_cap_follows_the_token_policy(monkeypatch, thinking):
    samples = [([{"role": "user", "content": "Question: x\n\nAnswer:"}], "entailment",
                {"category": "2_vars", "num_variables": 2, "relation": "parent"})]
    monkeypatch.setattr(causalbench, "_require_nltk", lambda: None)
    monkeypatch.setattr(causalbench, "predicted_relation", lambda text: "entailment")
    with mock.patch.object(causalbench, "_load_causalbench_samples", return_value=samples):
        res, sent = _through_base(
            monkeypatch, lambda t: causalbench.run_causalbench("m", "http://x/v1", 1,
                                                               enable_thinking=t),
            thinking, finish="length")
    assert sent[0]["max_output_tokens"] == (65536 if thinking else 10)
    assert res["causalbench_report"]["max_tokens_sent"] == [65536 if thinking else 10]
    assert res["causalbench_report"]["upstream_max_tokens"] == 10
    assert res["official_sampling"]["max_tokens_applied"] is (not thinking)
    if not thinking:
        assert res["status"] == "success" and res["leaderboard_comparable"] is True
    else:
        assert "not applied to reasoning" in res["leaderboard_comparable_reason"]


def test_cruxeval_declares_each_mode_cap_for_both_passes(monkeypatch):
    seen = []

    def fake_run(**kw):
        seen.append(kw)
        return {"sample_traces": [], "temperature": kw["official_sampling"].temperature}
    monkeypatch.setattr(cruxeval, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(cruxeval, "_grader_self_check", lambda: None)
    monkeypatch.setattr(cruxeval, "_load_cruxeval_samples", lambda **kw: [])
    monkeypatch.setattr(cruxeval, "run_eval_suite", fake_run)
    for thinking, cap in ((False, 100), (True, 1000)):
        seen.clear()
        cruxeval.run_cruxeval("m", "http://x/v1", 1, enable_thinking=thinking)
        assert [kw["official_sampling"].max_tokens for kw in seen] == [cap, cap]
        assert [kw["official_sampling"].temperature for kw in seen] == [0.2, 0.8]
    # base applies them: 100 exactly without thinking; the 1000 is not applied to reasoning.
    base.set_run_knobs(max_output_tokens=65536)
    direct = base.resolve_token_budget("cruxeval", cruxeval._official(0.2, "pass@1"),
                                       thinking=False)
    cot = base.resolve_token_budget("cruxeval", cruxeval._official(0.2, "pass@1", cot=True),
                                    thinking=True)
    assert (direct.policy, direct.max_tokens) == ("official", 100)
    assert (cot.policy, cot.max_tokens, cot.official_cap) == ("reasoning_run_budget", 65536, 1000)


# --------------------------------------------------------------------------- #
# mmlu_redux: HELM's one-token cap is not applied to reasoning
# --------------------------------------------------------------------------- #
def _redux_items(n=2):
    return [{"subject": "anatomy", "choices": ["a", "b", "c", "d"], "answer": 1,
             "prompt": f"P{i}"} for i in range(n)]


def test_mmlu_redux_no_think_request_is_unchanged(monkeypatch):
    sent = []

    async def fake_send(**kw):
        sent.append(kw)
        return base.Reply("B", None, "length", completion_tokens=1)
    monkeypatch.setattr(mmlu_redux, "_send_single_request", fake_send)
    base.set_run_knobs(max_output_tokens=65536)
    with mock.patch.object(mmlu_redux, "_load_mmlu_redux_items", return_value=_redux_items()):
        res = mmlu_redux.run_mmlu_redux("m", "http://x/v1", 1)
    assert [(kw["max_output_tokens"], kw["thinking"]) for kw in sent] == [(1, False)] * 2
    assert res["max_tokens_sent"] == [1] and res["official_sampling"]["max_tokens"] == 1
    assert res["official_sampling"]["max_tokens_applied"] is True
    assert res["max_output_tokens_note"].startswith("--max-output-tokens 65536 not applied")
    assert res["leaderboard_comparable"] is True and res["thinking_applied"] is False


def test_mmlu_redux_thinking_run_thinks_with_the_run_budget(monkeypatch):
    sent = []

    async def fake_send(**kw):
        sent.append(kw)
        return base.Reply("<thought>let me see</thought>B", None, "stop",
                          reasoning="it is B", completion_tokens=900)
    monkeypatch.setattr(mmlu_redux, "_send_single_request", fake_send)
    base.set_run_knobs(max_output_tokens=65536)
    with mock.patch.object(mmlu_redux, "_load_mmlu_redux_items", return_value=_redux_items()):
        res = mmlu_redux.run_mmlu_redux("m", "http://x/v1", 1, enable_thinking=True)
    assert [(kw["max_output_tokens"], kw["thinking"]) for kw in sent] == [(65536, True)] * 2
    assert res["thinking_applied"] is True and res["accuracy"] == 100.0
    assert res["official_sampling"]["max_tokens_applied"] is False
    assert base.REASONING_CAP_NOTE in res["max_output_tokens_note"]
    assert res["sample_traces"][0]["reasoning"] == "it is B"
    assert res["leaderboard_comparable"] is False
    assert "--thinking run" in res["leaderboard_comparable_reason"]
    assert "thinking_not_applied_reason" not in res


def test_mmlu_redux_thinking_budget_is_fitted_to_the_window(monkeypatch):
    sent = []

    async def fake_send(**kw):
        sent.append(kw)
        return base.Reply("B", None, "stop", completion_tokens=5)

    async def fake_clamp(messages, max_tokens, tools=None, ctk=None):
        if messages[0]["content"] == "P1":
            return max_tokens, "prompt is ~9,000 tokens, which does not fit"
        return 4000, None
    monkeypatch.setattr(mmlu_redux, "_send_single_request", fake_send)
    monkeypatch.setattr(mmlu_redux, "clamp_to_context_async", fake_clamp)
    base.set_run_knobs(max_output_tokens=65536)
    with mock.patch.object(mmlu_redux, "_load_mmlu_redux_items", return_value=_redux_items()):
        res = mmlu_redux.run_mmlu_redux("m", "http://x/v1", 1, enable_thinking=True)
    assert [kw["max_output_tokens"] for kw in sent] == [4000]
    statuses = [t["status"] for t in res["sample_traces"]]
    assert statuses == ["OK", "OVER_CONTEXT"] and res["failed_requests"] == 1
    assert res["max_tokens_sent"] == [4000]
    assert res["sample_traces"][1]["max_tokens_sent"] is None


# --------------------------------------------------------------------------- #
# lab_bench: the baseline harness sends no max_tokens
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("thinking", [False, True])
def test_lab_bench_sends_no_max_tokens(monkeypatch, thinking):
    assert lab_bench.OFFICIAL_SAMPLING.send_max_tokens is False
    sent = []

    async def fake_send(**kw):
        sent.append(kw)
        return base.Reply("[ANSWER]A[/ANSWER]", None, "stop", completion_tokens=5)
    monkeypatch.setattr(base, "_send_single_request", fake_send)
    base.set_run_knobs(max_output_tokens=65536)
    samples = [([{"role": "user", "content": "Which? (A) x (B) y"}],
                {"ideal": "A", "letters": ["A", "B"], "refuse": "B"},
                {"category": "LitQA2", "subtask": "LitQA2"})]
    monkeypatch.setattr(lab_bench, "_load_lab_bench_samples", lambda limit=None: samples)
    monkeypatch.setattr(lab_bench, "_require_image_input", lambda *a: None)

    async def no_score(traces):
        for t in traces:
            t["status"] = "OK"
    monkeypatch.setattr(lab_bench, "_async_score_lab_bench", no_score)
    monkeypatch.setattr(lab_bench, "_lab_bench_report",
                        lambda traces: {"mean_category_accuracy": 0.0})
    res = lab_bench.run_lab_bench("m", "http://x/v1", enable_thinking=thinking)
    assert sent and all(kw["max_output_tokens"] is None for kw in sent)
    assert res["official_sampling"]["send_max_tokens"] is False
    assert "not sent" in res["max_output_tokens_note"]


# --------------------------------------------------------------------------- #
# docs match the code
# --------------------------------------------------------------------------- #
def test_docs_describe_the_token_policy():
    lb = _doc("longbench_v2_code")
    assert "`min(upstream_max_prompt_tokens, window - output budget - 64)`" in lb
    assert "120,000 for a window up to 131,072 tokens" in lb
    assert "leaderboard_comparable` is always **false**" not in lb
    assert "prompts_cut_shorter_than_upstream" in lb and "120,192" in lb
    redux = _doc("mmlu_redux")
    assert "cannot be applied" not in redux and "thinking_not_applied_reason" not in redux
    assert "not applied to reasoning" in redux
    assert "no `max_tokens`" in _doc("lab_bench")
    for suite in ("gsm8k", "causalbench", "cruxeval", "longbench_v2_code", "mmlu_redux"):
        assert "`max_tokens_applied: false`" in _doc(suite), suite
    # the simple-evals suites declare no reasoning block (a served model takes the chat sampler
    # in both modes), so --thinking follows the policy's "not applied to reasoning" rule
    for suite in ("gpqa", "gpqa_diamond", "multilingual_mmlu"):
        assert "OfficialSampling.reasoning" not in _doc(suite), suite
        assert "`max_tokens_applied: false`" in _doc(suite), suite
