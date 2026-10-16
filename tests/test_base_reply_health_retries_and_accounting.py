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

"""base.run_eval_suite and its request path: reply health, retries and accounting.

* A reply that ran to the cap made of one repeated character (a run of spaces) is `looping`,
  not `truncated`.
* A reply stopped at the suite's official max_tokens keeps `truncated` (never `non_convergent`).
* A follow-up request of a suite's protocol that got no reply is counted as base's harness error,
  not as a suite exclusion.
* An endpoint that refuses `return_token_ids` gets requests without it for the rest of the run.
* A context-length 400 in vLLM's exact-count and character-count wordings is retried with a
  lower max_tokens, and the reply records the budget it answered (Reply.max_tokens_sent).
* pass^k counts only the samples with every attempt scored.
* redact_tree records a file under the stat taken before the scan that found it clean.
* gbench_cache_dir is absolute; TAU2_TEMPERATURE and the GBENCH_CULER_* spellings move the
  run-shaping snapshot as their canonical names do; an empty presharded shard names its full set.

Nothing here touches the network, docker or Gemini.
"""

import asyncio
import json
import os
from unittest import mock

import pytest

from gbench.runners.eval_suites import base


@pytest.fixture(autouse=True)
def _fresh_token_ids_state():
    base._TOKEN_IDS_REFUSED.clear()
    yield
    base._TOKEN_IDS_REFUSED.clear()


def _reply(text, finish="stop", tokens=5, error=None):
    return base.Reply(text=text, tool_calls=None, finish_reason=finish, reasoning=None,
                      error=error, completion_tokens=tokens, stop_reason=None)


def _run(answer_of, samples, **kw):
    async def _fake_send(**k):
        msgs = k["messages"]
        last = next(m["content"] for m in reversed(msgs) if m.get("role") == "user")
        return answer_of(last, k)

    with mock.patch.object(base, "_send_single_request", _fake_send):
        return base.run_eval_suite(eval_name="unit_base_health", model_name="m",
                                   base_url="http://x", concurrency=4, samples=samples,
                                   max_output_tokens=kw.pop("max_output_tokens", 64), **kw)


def _samples(n):
    return [([{"role": "user", "content": f"q{i}"}], "A", {}) for i in range(n)]


# --------------------------------------------------------------------------- #
# reply health
# --------------------------------------------------------------------------- #
def test_a_reply_of_spaces_at_the_cap_is_looping_and_names_its_run():
    spaces = "Here is the page " + " " * 50000 + "end"
    assert base.classify_reply(_reply(spaces, finish="length")) == "looping"
    assert base.classify_reply(_reply(" ".join(str(i) for i in range(5000)),
                                      finish="length")) == "truncated"
    rec = base.degenerate_char_run(spaces)
    assert rec["repeated"] == "whitespace" and rec["repeat_count"] >= 50000
    assert base.degenerate_char_run("x = 1\n" * 300) is None
    res = _run(lambda q, k: _reply(spaces, finish="length", tokens=64), _samples(1))
    t = res["sample_traces"][0]
    assert t["health"] == "looping" and res["looping_responses"] == 1
    assert t["repetition_onset"]["kind"] == "character_run"


def test_official_cap_replies_are_never_relabelled_non_convergent():
    """A cruxeval-like run: 100-token official caps against a ~20-token healthy median."""
    traces = [{"sample_idx": i, "health": "ok", "completion_tokens": 20} for i in range(10)]
    traces += [{"sample_idx": 10 + i, "health": "truncated", "completion_tokens": 100}
               for i in range(5)]
    caps = {10 + i: 100 for i in range(5)}
    assert base.reclassify_non_convergent([dict(t) for t in traces], caps) == 0
    assert base.reclassify_non_convergent([dict(t) for t in traces]) == 5


def test_the_looping_warning_does_not_blame_a_cap_that_was_never_sent(caplog):
    spaces = "x " + " " * 5000
    sampling = base.OfficialSampling(reference="unit", send_temperature=False,
                                     send_max_tokens=False)
    with caplog.at_level("WARNING"):
        _run(lambda q, k: _reply(spaces, finish="length"), _samples(1),
             official_sampling=sampling, max_output_tokens=None)
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "ran to the end of the context window" in text
    assert "Raising --max-output-tokens" not in text


# --------------------------------------------------------------------------- #
# a follow-up request that got no reply
# --------------------------------------------------------------------------- #
def test_a_follow_up_request_with_no_reply_is_base_accounted():
    async def scorer(traces):
        for t in traces:
            t["is_correct"], t["status"] = True, "OK"
            if t["sample_idx"] == 1:
                t["harness_error"] = base.endpoint_harness_error("FAILED",
                                                                 "last-chance request: 500")
                t["scoring_excluded"] = True
    res = _run(lambda q, k: _reply("A"), _samples(2), async_eval_fn=scorer)
    assert res["request_harness_errors"] == {"endpoint_failure": 1}
    assert "follow-up request(s) of the suite's protocol" in res["status_reason"]
    assert "scoring_excluded 1 (left out of the score by the suite)" not in res["status_reason"]
    assert res["status"] == "completed_with_errors" and res["accuracy"] == 100.0


# --------------------------------------------------------------------------- #
# the request path: return_token_ids refusal and context-length retries
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, status, body):
        self.status, self._body = status, body

    async def json(self):
        return self._body

    async def text(self):
        return self._body if isinstance(self._body, str) else json.dumps(self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Session:
    """Answers each post with the next scripted (status, body)."""

    def __init__(self, script):
        self.script = list(script)
        self.sent = []

    def post(self, url, json=None, **kw):
        self.sent.append(dict(json))
        return _Resp(*self.script.pop(0))


def _ok_body(text="done", finish="stop"):
    return {"choices": [{"message": {"role": "assistant", "content": text},
                         "finish_reason": finish}],
            "usage": {"completion_tokens": 3}}


def _send(session, max_tokens=4096):
    async def go():
        return await base._send_single_request(
            session=session, api_url="http://stub/v1/chat/completions", model_name="m",
            messages=[{"role": "user", "content": "q"}], extra_payload={},
            semaphore=asyncio.Semaphore(1), pbar=None, max_output_tokens=max_tokens,
            temperature=0.0)
    return asyncio.run(go())


def test_an_endpoint_that_refuses_return_token_ids_gets_requests_without_it():
    session = _Session([
        (400, "Unrecognized request argument supplied: return_token_ids"), (200, _ok_body())])
    reply = _send(session)
    assert reply.text == "done"
    assert "return_token_ids" in session.sent[0] and "return_token_ids" not in session.sent[1]
    rec = base.server_parser_token_ids_record()
    assert rec["available"] is False and "refused return_token_ids" in rec["reason"]
    later = _Session([(200, _ok_body())])
    _send(later)
    assert "return_token_ids" not in later.sent[0]          # the rest of the run


def test_an_exact_prompt_count_gets_an_exact_clamp_and_the_reply_records_it():
    err = ("This model's maximum context length is 8192 tokens. However, you requested 4096 "
           "output tokens and your prompt contains 5000 input tokens, for a total of 9096 tokens.")
    session = _Session([(400, err), (200, _ok_body(finish="length"))])
    reply = _send(session, max_tokens=4096)
    assert session.sent[1]["max_tokens"] == 8192 - 5000 - 16
    assert reply.max_tokens_sent == 8192 - 5000 - 16


def test_a_character_count_rejection_is_retried_with_less_room():
    err = ("This model's maximum context length is 8192 tokens. However, you requested 4096 "
           "output tokens and your prompt contains 30000 characters (more than 16384 characters, "
           "which is the upper bound for 4096 input tokens).")
    session = _Session([(400, err), (200, _ok_body())])
    reply = _send(session, max_tokens=4096)
    assert session.sent[1]["max_tokens"] < 4096 and reply.text == "done"
    # a prompt whose lower bound fills the window is over context, not retried
    err_big = err.replace("30000 characters", "40000 characters")
    over = _send(_Session([(400, err_big)]), max_tokens=4096)
    assert over.text is None and over.error.startswith(base.OVER_CONTEXT_ERROR)


def test_a_reply_at_a_lowered_budget_is_not_counted_at_the_official_cap():
    async def fake_send(**k):
        return base.Reply("partial", None, "length", None, None, 10, None, None, None, None, 100)
    samples = [([{"role": "user", "content": "q"}], "A", {"max_tokens": 1000})]
    with mock.patch.object(base, "_send_single_request", fake_send):
        res = base.run_eval_suite(eval_name="unit_base_budget", model_name="m",
                                  base_url="http://x", concurrency=1, samples=samples,
                                  max_output_tokens=64, eval_fn=lambda r, g: False)
    t = res["sample_traces"][0]
    assert t["max_tokens_sent"] == 100
    assert 100 not in (res.get("official_max_tokens_sent") or [])


# --------------------------------------------------------------------------- #
# pass^k over the samples with every attempt scored
# --------------------------------------------------------------------------- #
def test_pass_hat_k_needs_every_attempt_scored():
    async def scorer(traces):
        for t in traces:
            t["status"] = "OK"
            t["is_correct"] = True
            if t["source_sample_idx"] == 0 and t["attempt_idx"] == 1:
                t["judge_grade"] = "JUDGE_OUTAGE"
    res = _run(lambda q, k: _reply("A"), _samples(2), async_eval_fn=scorer, attempt_count=2,
               supports_attempts=True)
    rep = res["attempts"]
    assert rep["pass_hat_k"] == 100.0 and rep["pass_hat_k_samples"] == 1
    assert rep["samples_with_fewer_scored_attempts"] == 1
    assert "left out of pass^k" in rep["fewer_scored_attempts_note"]


# --------------------------------------------------------------------------- #
# redact_tree
# --------------------------------------------------------------------------- #
def test_bytes_appended_after_the_clean_scan_are_scanned_next_sweep(tmp_path, monkeypatch):
    secret = "sk-unit-secret-value-0123456789"
    path = tmp_path / "log.txt"
    path.write_text("clean line\n", encoding="utf-8")
    real = base.redact_file
    state = {"calls": 0}

    def scan_then_append(p, secrets=None):
        out = real(p, secrets)
        state["calls"] += 1
        if state["calls"] == 1:
            with open(p, "a", encoding="utf-8") as f:
                f.write(f"late {secret}\n")
        return out
    monkeypatch.setattr(base, "redact_file", scan_then_append)
    base._TREE_SEEN.clear()
    base.redact_tree(tmp_path, [secret])
    assert secret in path.read_text(encoding="utf-8")            # appended after the scan
    monkeypatch.setattr(base, "redact_file", real)
    base.redact_tree(tmp_path, [secret])
    assert secret not in path.read_text(encoding="utf-8")        # scanned again


# --------------------------------------------------------------------------- #
# small shared rules
# --------------------------------------------------------------------------- #
def test_the_cache_dir_is_absolute(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GBENCH_CACHE_DIR", "cache")
    assert base.gbench_cache_dir() == str(tmp_path / "cache")
    monkeypatch.setenv("GBENCH_CACHE_DIR", "~/gbc")
    assert base.gbench_cache_dir() == os.path.join(os.path.expanduser("~"), "gbc")


def test_legacy_spellings_move_the_snapshot_under_the_canonical_name():
    for suite in ("tau2", "tau3"):
        canon = base.temperature_env_var(suite)
        assert base.run_shaping_env([suite], environ={"TAU2_TEMPERATURE": "0.3"}) == \
            base.run_shaping_env([suite], environ={canon: "0.3"}) == {"decode": {canon: "0.3"}}
    for legacy, canon in (("GBENCH_CULER_WINDOW", "GBENCH_LONGBENCH_V2_CODE_WINDOW"),
                          ("GBENCH_CULER_ALL_DOMAINS", "GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS"),
                          ("CULER_ALL_DOMAINS", "GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS")):
        assert base.run_shaping_env(["longbench_v2_code"], environ={legacy: "1"}) == \
            base.run_shaping_env(["longbench_v2_code"], environ={canon: "1"})
    assert "GBENCH_CULER_WINDOW" not in base.RUN_SHAPING_SUITE_ENV["longbench_v2_code"]


def test_an_empty_presharded_shard_names_its_full_set(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "3/4")
    res = base.run_eval_suite(eval_name="unit_base_shard", model_name="m", base_url="http://x",
                              concurrency=1, samples=[], max_output_tokens=64,
                              full_set_items=2)
    assert res["empty_shard"] is True and res["full_set_items"] == 2
    assert "(2 in the full set)" in res["status_reason"]


def test_an_empty_presharded_shard_without_its_size_does_not_claim_zero(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "3/4")
    res = base.run_eval_suite(eval_name="unit_base_shard", model_name="m", base_url="http://x",
                              concurrency=1, samples=[], max_output_tokens=64)
    assert res["empty_shard"] is True and res["full_set_items"] is None
    assert "(0 in the full set)" not in res["status_reason"]
