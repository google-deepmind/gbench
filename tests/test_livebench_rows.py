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

"""LiveBench's answer rows read by cause, and bigcodebench's --thinking request for a served model.

* livebench: LiveBench@0baa4aa7 scores every api_error row 0. A reply the model finished with no
  content (finish_reason other than length) makes its litellm path raise "No message returned
  from litellm (finish_reason=stop)", which becomes an api_error row: that is the model's
  outcome (`empty_replies`), not an endpoint failure, and changes neither the status nor
  comparability. Every other api_error row (connection, HTTP status, stream error) is a harness
  failure (`failed_requests`, `completed_with_errors`, not comparable). The rows used here are
  the ones LiveBench wrote in an image built from 0baa4aa7 against a stub streaming endpoint.
* livebench: the official cap of a LiveBench config that sets no cap is gen_api_answer.py's
  default, and the note and reason say so; the answer-token count uses the run's tokenizer
  (--tokenizer, else the served model's id).
* bigcodebench: make_request's reasoning-model branch keys on hosted vendor model names, so a
  --thinking run sends make_request's request for a served model (temperature 0.0, top_p 0.95)
  with base's token-limit policy (the 1280 cap is not applied to reasoning), and no
  reasoning_effort.

Docker and the endpoint are stubbed: the LiveBench run writes its outputs into a temporary
directory, and bigcodebench talks to a local stub HTTP server on an ephemeral port.
"""

import json
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import bigcodebench as B
from gbench.runners.eval_suites import livebench as LB
from gbench.runners.eval_suites import tau_common as TC_GUARD


# --------------------------------------------------------------------------- #
# livebench                                                                    #
# --------------------------------------------------------------------------- #
OK_ROW = {"question_id": "q", "choices": [{"index": 0, "turns": ["ok"]}],
          "total_output_tokens": 10}

#: What LiveBench@0baa4aa7 writes for a reply the model finished (HTTP 200, finish_reason stop)
#: with reasoning and empty content, or with an immediate end of turn.
EMPTY_FINISHED_ROW = {"question_id": "q", "choices": [{"index": 0, "turns": ["$ERROR$"]}],
                      "total_output_tokens": 0, "eval_status": "api_error",
                      "error": "Exception",
                      "error_msg": "No message returned from litellm (finish_reason=stop)"}

#: What it writes for requests that failed at the endpoint (same image and stub: an HTTP 500 and
#: a stream that ended in an error).
HTTP_500_ROW = {"question_id": "q", "choices": [{"index": 0, "turns": ["$ERROR$"]}],
                "total_output_tokens": 0, "eval_status": "api_error",
                "error": "InternalServerError",
                "error_msg": "litellm.InternalServerError: InternalServerError: OpenAIException "
                             "- boom"}
STREAM_ERROR_ROW = {"question_id": "q", "choices": [{"index": 0, "turns": ["$ERROR$"]}],
                    "total_output_tokens": 0, "eval_status": "api_error",
                    "error": "MidStreamFallbackError",
                    "error_msg": "litellm.MidStreamFallbackError: litellm.APIConnectionError: "
                                 "APIConnectionError: OpenAIException - openai raised a "
                                 "streaming error - finish_reason: error"}

#: LiveBench's config for gemma-3-27b-it at 0baa4aa7 (google.yml): no api_kwargs, so no cap.
GEMMA3_27B = {
    "display_name": "gemma-3-27b-it", "api_name": {"google": "gemma-3-27b-it"},
    "aliases": ["gemma-3-27b"], "api_kwargs": None, "prompt_prefix": None,
    "preserve_reasoning": None, "cost_per_million": None, "api_base": None,
    "default_provider": None, "api_keys": None, "agentic_native_tools": None,
}


def _write_rows(tmp_path, rows):
    d = tmp_path / LB._ANSWER_DIR / "data" / "live_bench" / "math" / "x" / "model_answer"
    d.mkdir(parents=True, exist_ok=True)
    (d / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _write_if_rows(tmp_path):
    d = (tmp_path / LB._ANSWER_DIR / "data" / "live_bench" / "instruction_following"
         / "paraphrase" / "model_answer")
    d.mkdir(parents=True, exist_ok=True)
    (d / "m.jsonl").write_text(json.dumps(OK_ROW) + "\n", encoding="utf-8")


def _run(monkeypatch, tmp_path, config=None, rows=(OK_ROW,), **kwargs):
    monkeypatch.setenv("GBENCH_LIVEBENCH_RELEASE", "2025-04-02")
    for name in ("GBENCH_LIVEBENCH_CATEGORIES", "GBENCH_LIVEBENCH_TEMPERATURE",
                 "GBENCH_LIVEBENCH_MODEL", "GBENCH_LIVEBENCH_EXTRA_ARGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(LB, "_check_prereqs", lambda image: None)
    # the network checks before the run container are docker calls of their own
    # (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC_GUARD, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC_GUARD.container_url(url))
    monkeypatch.setattr(LB, "upstream_model_config", lambda image, names: {
        "name": names[-1] if config else None, "config": config,
        "commit": LB.LIVEBENCH_COMMIT})
    monkeypatch.setattr(LB.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))

    def fake_run(cmd, **_kw):
        cols = LB._CATEGORIES
        (tmp_path / "all_groups.csv").write_text(
            "model,average," + ",".join(cols) + "\ngoogle-gemma-4-31b-it,50.0,"
            + ",".join("50.0" for _ in cols) + "\n", encoding="utf-8")
        _write_rows(tmp_path, rows)
        if rows:   # instruction_following is scored: its answer rows are copied out too
            _write_if_rows(tmp_path)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(LB.subprocess, "run", fake_run)
    return LB.run_livebench("google/gemma-4-31B-it", "http://h/v1", 4, **kwargs)


def test_an_empty_reply_the_model_finished_is_its_outcome_not_an_endpoint_failure(tmp_path):
    _write_rows(tmp_path, [OK_ROW, EMPTY_FINISHED_ROW])
    rows = LB.read_answer_rows(str(tmp_path), 4096)
    assert rows == {"rows": 2, "cut": 0, "in_reasoning": 0, "uncounted": 0, "empty": 1,
                    "failed": 0}


def test_endpoint_failures_are_still_counted_as_failed(tmp_path):
    no_detail = {"question_id": "q", "choices": [{"index": 0, "turns": ["$ERROR$"]}],
                 "total_output_tokens": 0, "eval_status": "api_error"}
    # LiveBench's non-litellm path raises another message; gbench always runs --use-litellm.
    openai_path = {**EMPTY_FINISHED_ROW, "error_msg": "No message returned from OpenAI"}
    # the message with another exception class is not the empty-reply raise
    other_class = {**EMPTY_FINISHED_ROW, "error": "APIConnectionError"}
    _write_rows(tmp_path, [HTTP_500_ROW, STREAM_ERROR_ROW, no_detail, openai_path, other_class])
    rows = LB.read_answer_rows(str(tmp_path), 4096)
    assert rows["failed"] == 5 and rows["empty"] == 0


def test_run_livebench_scores_an_empty_reply_as_livebench_does(monkeypatch, tmp_path):
    r = _run(monkeypatch, tmp_path, rows=(OK_ROW, EMPTY_FINISHED_ROW))
    assert r["empty_replies"] == 1 and r["failed_requests"] == 0
    assert r["status"] == "success"
    assert r["leaderboard_comparable"] is True, r.get("leaderboard_comparable_reason")
    note = r["empty_replies_note"]
    assert "the model's outcome" in note and "scores 0" in note
    assert "vLLM's abort" in note


def test_run_livebench_empty_reply_in_a_thinking_run_is_not_a_cutoff(monkeypatch, tmp_path):
    base.set_run_knobs(max_output_tokens=32768)
    r = _run(monkeypatch, tmp_path, rows=(OK_ROW, EMPTY_FINISHED_ROW), enable_thinking=True,
             max_output_tokens=32768)
    assert r["empty_replies"] == 1 and r["failed_requests"] == 0
    assert r["replies_cut_off_by_budget"] == 0 and r["replies_cut_off_uncounted"] == 0
    assert r["status"] == "success"
    # The only departure: a thinking run samples at 1.0 (base.resolve_temperature).
    assert r["leaderboard_comparable"] is False
    assert r["leaderboard_comparable_reason"] == (
        "temperature=1.0 (thinking; sent as the config's temperature) (LiveBench samples at "
        "temperature 0, or the question's required temperature)")


def test_run_livebench_separates_the_two_causes(monkeypatch, tmp_path):
    r = _run(monkeypatch, tmp_path, rows=(OK_ROW, EMPTY_FINISHED_ROW, HTTP_500_ROW))
    assert r["empty_replies"] == 1 and r["failed_requests"] == 1
    assert r["status"] == "completed_with_errors"
    assert r["leaderboard_comparable"] is False
    assert ("1 request(s) failed at the served endpoint (LiveBench's api_error rows other than "
            "empty replies") in r["leaderboard_comparable_reason"]


def test_no_empty_replies_note_without_empty_replies(monkeypatch, tmp_path):
    r = _run(monkeypatch, tmp_path, rows=(OK_ROW,))
    assert r["empty_replies"] == 0 and "empty_replies_note" not in r


def test_a_capless_config_names_gen_api_answers_default_as_its_cap(monkeypatch, tmp_path):
    base.set_run_knobs(max_output_tokens=30000, max_model_len=32768)
    r = _run(monkeypatch, tmp_path, config=GEMMA3_27B, enable_thinking=True,
             max_output_tokens=30000)
    note = r["max_output_tokens_note"]
    assert ("LiveBench's config for gemma-3-27b-it sets no cap, so gen_api_answer.py's "
            "--max-tokens default is 4096") in note
    assert "sets max_tokens=4096" not in note


def test_a_capless_config_lowered_to_fit_names_the_default(monkeypatch, tmp_path):
    base.set_run_knobs(max_model_len=6000)
    r = _run(monkeypatch, tmp_path, config=GEMMA3_27B)
    assert r["max_output_tokens"] == 3000
    assert ("LiveBench's config for gemma-3-27b-it sets no cap, so gen_api_answer.py's "
            "--max-tokens default is 4096, the cap a no-think run gets") in \
        r["max_output_tokens_note"]
    assert ("max_tokens=3000: LiveBench's config for gemma-3-27b-it sets no cap, so "
            "gen_api_answer.py's --max-tokens default sends 4096") in \
        r["leaderboard_comparable_reason"]


def test_uncounted_rows_name_the_tokenizer_fallback(monkeypatch, tmp_path):
    base.set_run_knobs(max_output_tokens=32768)
    monkeypatch.setattr(LB, "_token_count", lambda text: None)
    r = _run(monkeypatch, tmp_path, rows=(OK_ROW, {**OK_ROW, "total_output_tokens": 50000}),
             enable_thinking=True, max_output_tokens=32768)
    assert r["replies_cut_off_uncounted"] == 1
    reason = r["leaderboard_comparable_reason"]
    assert "the run's --tokenizer, else the served model's id, could not be loaded" in reason
    assert "pass --tokenizer" in reason
    assert "else the served model's id" in r["replies_cut_off_note"]


# --------------------------------------------------------------------------- #
# bigcodebench                                                                 #
# --------------------------------------------------------------------------- #
@pytest.fixture
def endpoint():
    """A stub OpenAI-compatible endpoint that records every request body."""
    records = []
    state = {"reply": lambda body: ("ok", None, "stop")}

    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):                                      # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            records.append(body)
            content, reasoning, finish = state["reply"](body)
            msg = {"role": "assistant", "content": content}
            if reasoning is not None:
                msg["reasoning_content"] = reasoning
            data = json.dumps({"id": "x", "object": "chat.completion", "choices": [
                {"index": 0, "message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 7,
                          "total_tokens": 17}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield types.SimpleNamespace(url=f"http://127.0.0.1:{server.server_address[1]}/v1",
                                    records=records, state=state)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _bcb_setup(monkeypatch):
    monkeypatch.delenv("GBENCH_BIGCODEBENCH_TEMPERATURE", raising=False)
    monkeypatch.delenv("GBENCH_BIGCODEBENCH_SUBSET", raising=False)
    monkeypatch.setattr(B, "check_bigcodebench_prerequisites", lambda: (True, ""))
    samples = [([{"role": "user", "content": B.make_raw_chat_prompt("Write task_func.", split)}],
                "BigCodeBench/0", {"category": split, "task_id": "BigCodeBench/0",
                                   "split": split, "hard": False})
               for split in B.SPLITS]
    monkeypatch.setattr(B, "_load_bigcodebench_samples", lambda limit=None, subset="full": samples)

    def fake_scorer(subset, parallel, metrics, image):
        async def _score(traces):
            for t in traces:
                t["is_correct"], t["status"] = True, "OK"
        return _score

    monkeypatch.setattr(B, "_make_scorer", fake_scorer)


def test_bigcodebench_declares_no_reasoning_setting_for_a_served_model():
    for prefill in (True, False):
        assert B.SAMPLING[prefill].reasoning is None
        assert B.SAMPLING[prefill].for_mode(True) is B.SAMPLING[prefill]
    plan = base.resolve_token_budget("bigcodebench", B.SAMPLING[False], thinking=True,
                                     max_output_tokens=16384)
    assert (plan.policy, plan.max_tokens, plan.official_cap) == ("reasoning_run_budget", 16384,
                                                                 1280)
    assert B.VENDOR_REASONING_NAMES == ("o1-", "o3-", "reasoner", "grok-3-mini-beta")


def test_bigcodebench_thinking_sends_make_requests_served_model_request(monkeypatch, endpoint):
    _bcb_setup(monkeypatch)
    base.set_run_knobs(max_output_tokens=16384)
    endpoint.state["reply"] = lambda body: ("```python\ndef task_func():\n    pass\n```",
                                            "reasoning", "stop")
    r = B.run_bigcodebench("m", endpoint.url, 2, enable_thinking=True)
    assert len(endpoint.records) == 2
    for body in endpoint.records:
        # a thinking run samples at 1.0 (base.resolve_temperature); top_p stays make_request's
        assert body["temperature"] == 1.0 and body["top_p"] == 0.95
        assert body["max_tokens"] == 16384
        assert "reasoning_effort" not in body and "stop" not in body
        assert "continue_final_message" not in body
        assert body["chat_template_kwargs"]["enable_thinking"] is True
    rec = r["official_sampling"]
    assert "reasoning_settings" not in rec
    assert rec["max_tokens"] == 1280 and rec["max_tokens_applied"] is False
    assert rec["max_tokens_note"] == base.REASONING_CAP_NOTE and rec["max_tokens_sent"] == 16384
    assert rec["temperature_matches_official"] is False and rec["temperature"] == 0.0
    assert r["temperature_source"] == "thinking"
    assert base.REASONING_CAP_NOTE in r["max_output_tokens_note"]
    assert B._NO_REASONING_SETTING_NOTE in r["max_output_tokens_note"]
    assert "reasoning_effort" not in r
    assert r["prefill"] is False
    assert r["replies_cut_off_by_budget"] == 0
    assert r["leaderboard_comparable"] is False
    assert r["leaderboard_comparable_reason"] == (
        "temperature=1.0 (thinking; the leaderboard is greedy, temperature 0.0)")


def test_bigcodebench_thinking_reply_cut_off_at_the_runs_budget_is_not_comparable(monkeypatch,
                                                                                  endpoint):
    _bcb_setup(monkeypatch)
    base.set_run_knobs(max_output_tokens=16384)
    endpoint.state["reply"] = lambda body: ("", "Let me think ...", "length")
    r = B.run_bigcodebench("m", endpoint.url, 2, enable_thinking=True)
    assert r["replies_cut_off_by_budget"] == 2 and r["replies_cut_off_in_reasoning"] == 2
    assert r["leaderboard_comparable"] is False
    assert ("2 repl(ies) stopped at this run's max_tokens (16384), 2 of them inside the "
            "reasoning channel with no answer (run_codegen's 1280 is not applied to reasoning"
            ) in r["leaderboard_comparable_reason"]
    assert "max_tokens=[16384]" not in r["leaderboard_comparable_reason"]


def test_bigcodebench_thinking_at_another_temperature_is_not_comparable(monkeypatch, endpoint):
    _bcb_setup(monkeypatch)
    base.set_run_knobs(max_output_tokens=16384)
    monkeypatch.setenv("GBENCH_BIGCODEBENCH_TEMPERATURE", "0.6")
    r = B.run_bigcodebench("m", endpoint.url, 2, enable_thinking=True)
    assert {b["temperature"] for b in endpoint.records} == {0.6}
    assert ("temperature=0.6 (env:GBENCH_BIGCODEBENCH_TEMPERATURE; the leaderboard is greedy, "
            "temperature 0.0)") in r["leaderboard_comparable_reason"]


def test_bigcodebench_a_reasoning_effort_in_the_payload_is_not_comparable(monkeypatch, endpoint):
    _bcb_setup(monkeypatch)
    base.set_run_knobs(max_output_tokens=16384)
    monkeypatch.setattr(base, "ALLOWED_API_KEYS", base.ALLOWED_API_KEYS | {"reasoning_effort"})
    r = B.run_bigcodebench("m", endpoint.url, 2, enable_thinking=True,
                           extra_payload={"reasoning_effort": "medium"})
    assert {b.get("reasoning_effort") for b in endpoint.records} == {"medium"}
    assert ("reasoning_effort='medium' (make_request sends none for a served model"
            in r["leaderboard_comparable_reason"])
