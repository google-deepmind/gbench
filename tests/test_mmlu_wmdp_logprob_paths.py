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

"""How mmlu and wmdp read log-probabilities from a /v1/completions server.

* mmlu reads the letters by the token text " A" .. " D" (hendrycks/test evaluate.py). A server
  that keys its top list otherwise (token-id placeholders, letters without the leading space)
  is a harness error, never an all-A score: a reply whose keys show another format fails, on
  the first question the suite stops, and a run in which no scored reply lists a letter stops
  with infra_required.
* wmdp reads lm-eval v0.4.2's per-choice loglikelihoods from one next-token request per
  question (logprob_token_ids, token-id keys) when every continuation is one token, and from
  lm-eval's own echoed-prompt requests otherwise. Both give the same values.

Every server here is an aiohttp stub on an ephemeral 127.0.0.1 port; no model runs.
"""

import asyncio
import re
import threading
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import mmlu, wmdp

REPO = Path(__file__).resolve().parent.parent


def _doc(suite):
    return " ".join((REPO / "docs" / "evals" / f"{suite}.md").read_text(encoding="utf-8").split())


def _folded(text):
    return " ".join((text or "").split())


def _serve(handler):
    """aiohttp app with POST /v1/completions on an ephemeral port; returns (base_url, stop)."""
    from aiohttp import web
    loop = asyncio.new_event_loop()
    app = web.Application()
    app.router.add_post("/v1/completions", handler)
    runner = web.AppRunner(app)
    ready, port = threading.Event(), {}

    def run():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", 0)
        loop.run_until_complete(site.start())
        port["p"] = site._server.sockets[0].getsockname()[1]
        ready.set()
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    ready.wait(10)

    def stop():
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
    return f"http://127.0.0.1:{port['p']}/v1", stop


class _CharTok:
    """One token per character, no BOS."""
    bos_token_id = None

    def encode(self, s):
        return [ord(c) for c in s]


# =========================================================================== #
# mmlu: key format
# =========================================================================== #
#: Gemma 4's ids for " A" .. " D" (checked with its tokenizer); any ids would do here.
_LETTER_IDS = {"A": 562, "B": 603, "C": 565, "D": 622}


def _mmlu_items(n):
    return [{"subject": "anatomy", "header": mmlu.gen_prompt_header("anatomy"), "shots": [],
             "question": f"Q{i}?\nA. a\nB. b\nC. c\nD. d\nAnswer:", "gold": i % 4}
            for i in range(n)]


def _q(body):
    return int(body["prompt"].rsplit("Q", 1)[1].split("?", 1)[0])


def _reply(top):
    tok = next(iter(top))
    return {"choices": [{"index": 0, "text": tok, "finish_reason": "length", "logprobs": {
        "text_offset": [0], "tokens": [tok], "token_logprobs": [top[tok]], "top_logprobs": [top]}}]}


def _text_keyed(gold, size=100):
    """vLLM's default keys: token text. The gold letter ranks first."""
    top = {" " + gold: -0.1}
    top.update({" " + c: -3.0 for c in "ABCD" if c != gold})
    top.update({f"tok{j}": -7.0 - j for j in range(size - 4)})
    return top


def _token_id_keyed(gold, size=100):
    """A server started with --return-tokens-as-token-ids: every key is 'token_id:<id>'."""
    top = {f"token_id:{_LETTER_IDS[gold]}": -0.1}
    top.update({f"token_id:{_LETTER_IDS[c]}": -3.0 for c in "ABCD" if c != gold})
    top.update({f"token_id:{1000 + j}": -7.0 - j for j in range(size - 4)})
    return top


def _bare_letter_keyed(gold, size=100):
    """vLLM before v0.26.0 for a tokenizer whose one-token decode drops the leading space."""
    top = {gold: -0.1}
    top.update({c: -3.0 for c in "ABCD" if c != gold})
    top.update({f"tok{j}": -7.0 - j for j in range(size - 4)})
    return top


def _run_mmlu(handler, n=4):
    url, stop = _serve(handler)
    try:
        with mock.patch.object(mmlu, "_load_mmlu_items", return_value=_mmlu_items(n)), \
                mock.patch.object(mmlu, "_load_tokenizer", return_value=_CharTok()), \
                mock.patch.object(mmlu, "get_run_knob", return_value=None):
            return mmlu.run_mmlu("m", url, 2)
    finally:
        stop()


def test_mmlu_token_id_keys_stop_the_suite_naming_the_flag():
    from aiohttp import web
    seen = []

    async def handler(req):
        body = await req.json()
        seen.append(body)
        return web.json_response(_reply(_token_id_keyed("ABCD"[_q(body) % 4])))
    with pytest.raises(RuntimeError) as err:
        _run_mmlu(handler)
    msg = str(err.value)
    assert "INFRA REQUIRED" in msg and "mmlu" in msg
    assert "token_id:<id>" in msg and "--return-tokens-as-token-ids" in msg
    assert len(seen) == 1                       # stopped at the first question, no score


def test_mmlu_letters_without_the_leading_space_stop_the_suite():
    from aiohttp import web

    async def handler(req):
        body = await req.json()
        return web.json_response(_reply(_bare_letter_keyed("ABCD"[_q(body) % 4])))
    with pytest.raises(RuntimeError) as err:
        _run_mmlu(handler)
    msg = str(err.value)
    assert "INFRA REQUIRED" in msg and "without the leading space" in msg
    assert "v0.26.0" in msg and "#48674" in msg


def test_mmlu_no_letter_in_any_scored_reply_is_a_harness_error_not_an_all_a_score():
    from aiohttp import web

    async def handler(req):     # keys that show no known format and never a letter
        return web.json_response(_reply({f"tok{j}": -1.0 - j for j in range(100)}))
    with pytest.raises(RuntimeError) as err:
        _run_mmlu(handler, n=4)
    msg = str(err.value)
    assert "INFRA REQUIRED" in msg and "none of the 4 scored replies" in msg
    assert "every question A" in msg and "not a model result" in msg
    assert "'tok0'" in msg                      # the keys of such a reply are quoted


def test_mmlu_no_letter_in_any_reply_counts_only_the_scored_ones(monkeypatch):
    from aiohttp import web
    monkeypatch.setattr(mmlu, "_RETRIES", 1)

    async def handler(req):
        body = await req.json()
        if _q(body) == 3:
            return web.Response(status=500, text="EngineCore died")
        return web.json_response(_reply({f"tok{j}": -1.0 - j for j in range(100)}))
    with pytest.raises(RuntimeError, match="none of the 3 scored replies"):
        _run_mmlu(handler, n=4)


def test_mmlu_a_foreign_keyed_reply_mid_run_fails_that_question_only():
    from aiohttp import web

    async def handler(req):
        body = await req.json()
        gold = "ABCD"[_q(body) % 4]
        top = _token_id_keyed(gold) if _q(body) == 2 else _text_keyed(gold)
        return web.json_response(_reply(top))
    res = _run_mmlu(handler, n=4)
    assert res["failed_requests"] == 1 and res["status"] == "completed_with_errors"
    trace = res["sample_traces"][2]
    assert trace["status"] == "FAILED" and trace["predicted"] is None
    assert "not keyed by token text" in trace["error"] and "token_id:<id>" in trace["error"]
    assert res["correct_answers"] == 3 and res["leaderboard_comparable"] is False
    # a harness error, left out of the score (never a wrong answer)
    assert "1 question(s) got no usable reply from the served endpoint" in \
        res["leaderboard_comparable_reason"]


def test_mmlu_a_model_reply_without_a_letter_is_scored_as_the_reference_scores_it():
    from aiohttp import web

    async def handler(req):
        body = await req.json()
        q = _q(body)
        top = ({f"tok{j}": -1.0 - j for j in range(100)} if q == 1
               else _text_keyed("ABCD"[q % 4]))
        return web.json_response(_reply(top))
    res = _run_mmlu(handler, n=4)
    assert res["status"] == "success" and res["failed_requests"] == 0
    assert res["questions_without_a_letter"] == 1
    assert res["sample_traces"][1]["predicted"] == "A"        # np.argmax of four -100s
    assert res["correct_answers"] == 3 and res["leaderboard_comparable"] is True


def test_mmlu_foreign_key_format_is_named_only_when_the_keys_show_it():
    f = mmlu._foreign_key_format
    assert "--return-tokens-as-token-ids" in f(_token_id_keyed("B"))
    assert "#48674" in f(_bare_letter_keyed("C"))
    assert f({f"tok{j}": -1.0 for j in range(100)}) is None
    # Some but not all bare letters, or one placeholder among text keys: not shown.
    assert f({"A": -0.1, "B": -0.2, "tok": -1.0}) is None
    assert f({"token_id:562": -0.1, "tok": -1.0}) is None
    # A list that lists a spaced letter is read as evaluate.py reads it, bare letters or not.
    lps, missing, _ = mmlu._letter_logprobs(
        _reply({" B": -0.2, "A": -0.1, "B": -0.3, "C": -0.4, "D": -0.5})["choices"][0])
    assert missing == ["A", "C", "D"] and lps[1] == -0.2
    with pytest.raises(ValueError, match="not keyed by token text"):
        mmlu._letter_logprobs(_reply(_bare_letter_keyed("A"))["choices"][0])


@pytest.mark.parametrize("thinking,env,sent,source", [
    (False, None, 0.0, "default"),
    (True, None, 0.0, "default"),     # --thinking is not applied: the request is a no-think one
    (False, "0.7", 0.7, "env:GBENCH_MMLU_TEMPERATURE"),
    (True, "0.7", 0.7, "env:GBENCH_MMLU_TEMPERATURE"),
])
def test_mmlu_request_carries_gbench_temperature(monkeypatch, thinking, env, sent, source):
    """gbench's rule on the wire: 0.0 for the (always no-think) request; the operator's
    per-suite variable reaches the request in both modes."""
    from aiohttp import web
    if env is None:
        monkeypatch.delenv("GBENCH_MMLU_TEMPERATURE", raising=False)
    else:
        monkeypatch.setenv("GBENCH_MMLU_TEMPERATURE", env)
    seen = []

    async def handler(req):
        body = await req.json()
        seen.append(body)
        return web.json_response(_reply(_text_keyed("ABCD"[_q(body) % 4])))
    url, stop = _serve(handler)
    try:
        with mock.patch.object(mmlu, "_load_mmlu_items", return_value=_mmlu_items(2)), \
                mock.patch.object(mmlu, "_load_tokenizer", return_value=_CharTok()), \
                mock.patch.object(mmlu, "get_run_knob", return_value=None):
            res = mmlu.run_mmlu("m", url, 2, enable_thinking=thinking)
    finally:
        stop()
    assert len(seen) == 2 and all(body["temperature"] == sent for body in seen)
    assert (res["temperature"], res["temperature_source"]) == (sent, source)
    assert res["correct_answers"] == 2


def test_mmlu_docs_state_the_key_format_rule():
    doc = _doc("mmlu")
    assert "**Key format**" in doc and "`--return-tokens-as-token-ids`" in doc
    assert "v0.26.0, PR #48674" in doc and "instead of reporting the all-A accuracy" in doc
    source = _folded(mmlu.__doc__)
    assert "never reported as a score" in source and "--return-tokens-as-token-ids" in source
    assert "infra_required instead of reporting the all-A accuracy" in source


# =========================================================================== #
# wmdp: the next-token request and lm-eval's echoed-prompt request
# =========================================================================== #
class _PieceTok:
    """Each non-space character, with the space before it if any, is one token; no BOS. So
    " A" after "Answer:" is one token, as with the Gemma 4 tokenizer."""
    bos_token_id = None

    def __init__(self):
        self.vocab, self.inverse = {}, {}

    def encode(self, s):
        ids = []
        for piece in re.findall(r" ?[^ ]| ", s):
            if piece not in self.vocab:
                self.vocab[piece] = 1000 + len(self.vocab)
                self.inverse[self.vocab[piece]] = piece
            ids.append(self.vocab[piece])
        return ids

    def decode(self, ids):
        return "".join(self.inverse.get(i, f"<{i}>") for i in ids)


#: Tokens outside the prompts that the stub model also ranks: 50 fillers.
_FILLERS = list(range(9000, 9050))


def _wmdp_samples(n=4):
    return [{"subset": "wmdp-bio", "context": f"Q{i}?\nA. w\nB. x\nC. y\nD. z\nAnswer:",
             "gold": i % 4} for i in range(n)]


def _model(tok, prefix, t):
    """The stub model's log P(t | prefix), shared by every request kind."""
    text, piece = tok.decode(prefix), tok.decode([t])
    if t in _FILLERS:
        return -1.0 - (t - _FILLERS[0]) / 100.0           # -1.00 .. -1.49
    if text.endswith("Answer:") and piece in (" A", " B", " C", " D"):
        q = int(text.rsplit("Q", 1)[1].split("?", 1)[0])
        best = "ABCD"[q % 4] if q != 3 else "A"          # right on Q0-Q2, wrong on Q3
        if q == 1 and piece == " D":
            return -5.0                                   # below every filler: out of a top 20
        return -0.2 if piece == " " + best else -0.5 - 0.1 * "ABCD".index(piece[1])
    return -2.0


def _echo_reply(tok, ids):
    lps = [None] + [_model(tok, ids[:i], ids[i]) for i in range(1, len(ids))]
    return {"choices": [{"index": 0, "text": "", "logprobs": {"token_logprobs": lps}}]}


def _candidates(tok, ctx):
    letters = [tok.vocab[" " + c] for c in "ABCD"]
    return sorted(letters + _FILLERS, key=lambda t: -_model(tok, ctx, t))


def _stub(tok, kind, seen, fail=None):
    """kind: 'ids' (vLLM v0.26.0+: logprob_token_ids honoured), 'top20' (older vLLM: the field
    ignored, top 20 by id) or 'text' (a server that ignores both extension fields: top 20 keyed
    by text). `fail(body)` may return a web.Response to send instead."""
    from aiohttp import web

    async def handler(req):
        body = await req.json()
        seen.append(body)
        if fail is not None:
            resp = fail(body)
            if resp is not None:
                return resp
        (ids,) = body["prompt"]
        if body.get("echo"):
            return web.json_response(_echo_reply(tok, ids))
        ranked = _candidates(tok, ids)
        sampled = ranked[0]
        if kind == "ids":
            listed = [sampled] + list(body["logprob_token_ids"])
        else:
            listed = ranked[:body["logprobs"]]
        if kind == "text":
            top = {tok.decode([t]) if t not in _FILLERS else f"f{t}": _model(tok, ids, t)
                   for t in listed}
            key = tok.decode([sampled])
        else:
            top = {f"token_id:{t}": _model(tok, ids, t) for t in listed}
            key = f"token_id:{sampled}"
        return web.json_response({"choices": [{"index": 0, "text": "", "logprobs": {
            "tokens": [key], "token_logprobs": [top[key]], "top_logprobs": [top]}}]})
    return handler


def _run_wmdp(handler, tok, n=4):
    url, stop = _serve(handler)
    try:
        with mock.patch.object(wmdp, "_load_wmdp_samples", return_value=_wmdp_samples(n)), \
                mock.patch.object(wmdp, "_load_tokenizer", return_value=tok):
            return wmdp.run_wmdp("m", url, concurrency=3)
    finally:
        stop()


def _expected(tok, n=4):
    """lm-eval's value for each question and choice: log P(" X" | context)."""
    out = []
    for s in _wmdp_samples(n):
        ctx = tok.encode(s["context"])
        out.append([round(_model(tok, ctx, tok.vocab[" " + c]), 6) for c in "ABCD"])
    return out


def test_wmdp_scores_each_question_from_one_next_token_request():
    tok, seen = _PieceTok(), []
    res = _run_wmdp(_stub(tok, "ids", seen), tok)
    assert len(seen) == 4                               # one request per question, not four
    letters = sorted(tok.vocab[" " + c] for c in "ABCD")
    for body in seen:
        assert set(body) == {"model", "prompt", "max_tokens", "temperature", "seed", "logprobs",
                             "logprob_token_ids", "return_tokens_as_token_ids"}
        assert body["max_tokens"] == 1 and body["temperature"] == 0.0 and body["seed"] == 1234
        assert body["logprobs"] == 20 and body["return_tokens_as_token_ids"] is True
        assert body["logprob_token_ids"] == letters and "echo" not in body
        (ids,) = body["prompt"]
        assert tok.decode(ids).endswith("\nAnswer:")    # the context only, no continuation
    assert [t["choice_loglikelihoods"] for t in res["sample_traces"]] == _expected(tok)
    assert all(t["request"] == "next_token" for t in res["sample_traces"])
    assert res["questions_scored_by_next_token_request"] == 4
    assert res["questions_scored_by_echo_request"] == 0
    assert [t["predicted"] for t in res["sample_traces"]] == ["A", "B", "C", "A"]
    assert res["correct_answers"] == 3 and res["accuracy"] == 75.0
    assert res["status"] == "success" and res["leaderboard_comparable"] is True


def test_wmdp_next_token_and_lm_eval_requests_give_the_same_loglikelihoods():
    tok = _PieceTok()
    seen_ids, seen_text = [], []
    by_ids = _run_wmdp(_stub(tok, "ids", seen_ids), tok)
    # A server that ignores logprob_token_ids and return_tokens_as_token_ids keys its list by
    # text, which is not read: every question is scored with lm-eval's own four requests.
    by_echo = _run_wmdp(_stub(tok, "text", seen_text), tok)
    assert by_echo["questions_scored_by_echo_request"] == 4
    assert sum(1 for b in seen_text if b.get("echo")) == 16
    for a, b in zip(by_ids["sample_traces"], by_echo["sample_traces"]):
        assert a["choice_loglikelihoods"] == b["choice_loglikelihoods"]
        assert a["predicted"] == b["predicted"]
        assert (a["request"], b["request"]) == ("next_token", "lm_eval_echo")
    assert by_ids["accuracy"] == by_echo["accuracy"]
    for body in (b for b in seen_text if b.get("echo")):
        assert body["max_tokens"] == 0 and body["logprobs"] == 10 and body["seed"] == 1234


def test_wmdp_older_vllm_top20_without_a_letter_uses_lm_eval_requests_for_that_question():
    tok, seen = _PieceTok(), []
    res = _run_wmdp(_stub(tok, "top20", seen), tok)
    requests = [t["request"] for t in res["sample_traces"]]
    assert requests == ["next_token", "lm_eval_echo", "next_token", "next_token"]
    assert res["questions_scored_by_next_token_request"] == 3
    assert res["questions_scored_by_echo_request"] == 1
    assert [t["choice_loglikelihoods"] for t in res["sample_traces"]] == _expected(tok)
    assert res["sample_traces"][1]["choice_loglikelihoods"][3] == -5.0
    echo = [b for b in seen if b.get("echo")]
    assert len(echo) == 4 and all(tok.decode(b["prompt"][0]).startswith("Q1?") for b in echo)
    assert res["status"] == "success" and res["leaderboard_comparable"] is True


def test_wmdp_next_token_request_rejected_with_4xx_goes_to_lm_eval_requests_at_once():
    from aiohttp import web
    tok, seen = _PieceTok(), []

    def fail(body):      # a server that refuses the extension fields
        if not body.get("echo"):
            return web.json_response({"error": {"message": "unknown field"}}, status=400)
        return None
    res = _run_wmdp(_stub(tok, "ids", seen, fail=fail), tok)
    assert sum(1 for b in seen if not b.get("echo")) == 4          # not retried
    assert sum(1 for b in seen if b.get("echo")) == 16
    assert res["questions_scored_by_echo_request"] == 4
    assert [t["choice_loglikelihoods"] for t in res["sample_traces"]] == _expected(tok)
    assert res["status"] == "success"


def test_wmdp_next_token_429_is_retried_not_treated_as_a_rejection(monkeypatch):
    from aiohttp import web
    monkeypatch.setattr(wmdp, "_RETRIES", 2)
    tok, seen, calls = _PieceTok(), [], {"n": 0}

    def fail(body):
        if not body.get("echo") and calls["n"] == 0:
            calls["n"] += 1
            return web.Response(status=429, text="Too Many Requests")
        return None
    res = _run_wmdp(_stub(tok, "ids", seen, fail=fail), tok, n=1)
    assert res["sample_traces"][0]["request"] == "next_token"
    assert not any(b.get("echo") for b in seen) and len(seen) == 2


def test_wmdp_next_token_server_error_is_a_harness_error_not_a_switch(monkeypatch):
    from aiohttp import web
    monkeypatch.setattr(wmdp, "_RETRIES", 1)
    tok, seen = _PieceTok(), []

    def fail(body):
        if not body.get("echo") and tok.decode(body["prompt"][0]).startswith("Q2?"):
            return web.Response(status=500, text="CUDA out of memory")
        return None
    res = _run_wmdp(_stub(tok, "ids", seen, fail=fail), tok)
    trace = res["sample_traces"][2]
    assert trace["status"] == "FAILED" and trace["request"] == "next_token"
    assert "HTTP 500" in trace["error"]
    assert not any(b.get("echo") for b in seen)
    assert res["failed_requests"] == 1 and res["status"] == "completed_with_errors"
    assert res["leaderboard_comparable"] is False
    assert res["questions_scored_by_next_token_request"] == 3


def test_wmdp_first_question_failing_both_requests_is_infra_required(monkeypatch):
    from aiohttp import web
    monkeypatch.setattr(wmdp, "_RETRIES", 1)
    tok = _PieceTok()

    async def handler(req):
        return web.json_response({"error": {"message": "not supported"}}, status=400)
    with pytest.raises(RuntimeError) as err:
        _run_wmdp(handler, tok)
    msg = str(err.value)
    assert "INFRA REQUIRED" in msg and "logprob_token_ids" in msg and "echo=true" in msg


def test_wmdp_reads_only_a_reply_with_exactly_one_position():
    targets = [1, 2, 3, 4]
    one = {f"token_id:{t}": -float(t) for t in targets}
    read = wmdp._next_token_logprobs
    assert read({"logprobs": {"top_logprobs": [one]}}, targets) == [-1.0, -2.0, -3.0, -4.0]
    # A list per prompt position too (an echo the request did not ask for): not read.
    assert read({"logprobs": {"top_logprobs": [one, one]}}, targets) is None
    assert read({"logprobs": {"top_logprobs": [{"token_id:1": -1.0}]}}, targets) is None
    assert read({"logprobs": {"top_logprobs": [{" A": -1.0}]}}, targets) is None
    assert read({"logprobs": None}, targets) is None


def test_wmdp_multi_token_continuations_use_lm_eval_requests_only():
    from aiohttp import web
    seen = []

    async def handler(req):     # one token per character: " A" is two tokens
        body = await req.json()
        seen.append(body)
        (ids,) = body["prompt"]
        lps = [None] + [-1.0] * (len(ids) - 1)
        return web.json_response({"choices": [{"index": 0, "logprobs": {"token_logprobs": lps}}]})
    res = _run_wmdp(handler, _CharTok(), n=2)
    assert len(seen) == 8 and all(b.get("echo") is True and b["max_tokens"] == 0 for b in seen)
    assert res["questions_scored_by_echo_request"] == 2
    assert res["questions_scored_by_next_token_request"] == 0


@pytest.mark.parametrize("thinking,env,sent,source", [
    (False, None, 0.0, "default"),
    (True, None, 0.0, "default"),     # --thinking is not applied: the requests are no-think ones
    (False, "0.7", 0.7, "env:GBENCH_WMDP_TEMPERATURE"),
    (True, "0.7", 0.7, "env:GBENCH_WMDP_TEMPERATURE"),
])
def test_wmdp_requests_carry_gbench_temperature(monkeypatch, thinking, env, sent, source):
    """gbench's rule on the wire, for the next-token and the lm-eval echo requests: 0.0 (the
    requests are never thinking ones); the operator's per-suite variable reaches both."""
    if env is None:
        monkeypatch.delenv("GBENCH_WMDP_TEMPERATURE", raising=False)
    else:
        monkeypatch.setenv("GBENCH_WMDP_TEMPERATURE", env)
    tok, seen = _PieceTok(), []
    url, stop = _serve(_stub(tok, "top20", seen))   # Q1's letter is out of the top 20: echo too
    try:
        with mock.patch.object(wmdp, "_load_wmdp_samples", return_value=_wmdp_samples(4)), \
                mock.patch.object(wmdp, "_load_tokenizer", return_value=tok):
            res = wmdp.run_wmdp("m", url, concurrency=3, enable_thinking=thinking)
    finally:
        stop()
    assert any(body.get("echo") for body in seen) and any(not body.get("echo") for body in seen)
    assert all(body["temperature"] == sent for body in seen)
    assert (res["temperature"], res["temperature_source"]) == (sent, source)


def test_wmdp_payloads():
    assert wmdp._payload("m", [5, 6]) == {"model": "m", "prompt": [[5, 6]], "echo": True,
                                          "max_tokens": 0, "temperature": 0.0, "logprobs": 10,
                                          "seed": 1234}
    assert wmdp._next_token_payload("m", [5, 6], [9, 7, 9, 8]) == {
        "model": "m", "prompt": [[5, 6]], "max_tokens": 1, "temperature": 0.0, "seed": 1234,
        "logprobs": 20, "logprob_token_ids": [7, 8, 9], "return_tokens_as_token_ids": True}


def test_wmdp_docs_state_the_next_token_request_and_the_equivalence():
    doc = _doc("wmdp")
    for needle in ("**Next-token request**", "**Why the numbers are the same**",
                   "`logprob_token_ids`", "v0.26.0 and later (PR #43463)",
                   "`token_logprobs[ctxlen]`", "`return_tokens_as_token_ids=true`",
                   "`questions_scored_by_next_token_request`",
                   "`questions_scored_by_echo_request`", "`lm_eval_echo`",
                   "an HTTP 4xx other than 408 and 429"):
        assert needle in doc, needle
    source = _folded(wmdp.__doc__)
    assert "one request replaces four" in source and "logprob_token_ids" in source
    assert "PR #43463" in source and "raw_logprobs" in source
