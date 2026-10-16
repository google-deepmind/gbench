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

"""The token-limit policy in the maths and code suites, per mode, on the wire.

* lcb, multipl_e and bigcodebench declare their upstream cap in OfficialSampling and base's
  resolve_token_budget applies it: a no-think request carries exactly 2000 / 2048 / 1280; a
  --thinking request is never capped at it and is sent the run's budget, since none of the three
  declares a reasoning-model setting for a served model. bigcodebench's make_request has a
  reasoning-model branch, but it keys on hosted vendor model names, so a --thinking run sends
  make_request's request for a served model (temperature 0.0, top_p 0.95) and no
  reasoning_effort.
* A --thinking reply cut off at the run's own budget inside the reasoning channel makes lcb and
  multipl_e not comparable.
* ojbench: no max_tokens without --thinking (the model's default context length), 64k with it.
* livebench: gen_api_answer.py's 4096 without --thinking, the run's budget with it; a served model
  with a LiveBench config (gemma-4-31b-it) runs that config's settings.
* scicode and aider_polyglot send none in either mode and say so through the same hook.
* MultiPL-E: a whitespace-only reply is a parse failure (completion ""), not an empty reply.
* hmmt and aime send the served model's own recommended settings (MathArena's rule); an
  unreadable generation_config.json is infra_required.

The endpoint is a local stub HTTP server on an ephemeral port, so the tests see what base
actually sends. Docker is stubbed, except one opt-in check against the local pinned LiveBench
image (livebench.LIVEBENCH_IMAGE_DEFAULT), which runs only when that image exists.
"""

import asyncio
import hashlib
import json
import shutil
import subprocess
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from gbench.runners.eval_suites import aider_polyglot as A
from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import bigcodebench as B
from gbench.runners.eval_suites import lcb as L
from gbench.runners.eval_suites import livebench as LB
from gbench.runners.eval_suites import multipl_e as M
from gbench.runners.eval_suites import ojbench as OJ
from gbench.runners.eval_suites import scicode as SC
from gbench.runners.eval_suites import tau_common as TC_GUARD

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# A stub OpenAI-compatible endpoint that records every request body            #
# --------------------------------------------------------------------------- #
@pytest.fixture
def endpoint():
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


def _cut_in_reasoning(body):
    """A thinking model that ran out of budget while reasoning: no content, finish length."""
    return "", "Let me think about this step by step ...", "length"


# --------------------------------------------------------------------------- #
# lcb                                                                          #
# --------------------------------------------------------------------------- #
_STDIN_Q = "Read two integers and print their sum."


def _lcb_setup(monkeypatch):
    monkeypatch.delenv("GBENCH_LCB_INCLUDE_SELF_REPAIR", raising=False)
    monkeypatch.delenv("GBENCH_LCB_TEMPERATURE", raising=False)
    monkeypatch.setattr(L, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(L, "_grader_self_check", lambda: None)
    samples = [([{"role": "system", "content": L._SYSTEM_MESSAGE_GENERIC},
                 {"role": "user", "content": L._generation_prompt(_STDIN_Q, "")}],
                {"task": "generation", "fn_name": None, "question_content": _STDIN_Q,
                 "tests": [{"input": "2 3", "output": "5"}]},
                {"category": "code_gen_easy", "contest_date": "2025-02-01"})]
    monkeypatch.setattr(L, "_load_lcb_samples", lambda *a, **k: samples)


def test_lcb_no_think_sends_lcb_runners_2000_on_every_request(monkeypatch, endpoint):
    _lcb_setup(monkeypatch)
    base.set_run_knobs(max_output_tokens=16384)
    endpoint.state["reply"] = lambda body: ("```python\nprint(6)\n```", None, "stop")
    res = L.run_lcb("m", endpoint.url, 4)
    assert len(endpoint.records) == 20                 # 10 generations + 10 self-repairs
    assert {b["max_tokens"] for b in endpoint.records} == {2000}
    assert res["official_sampling"]["max_tokens"] == 2000
    assert res["official_sampling"]["max_tokens_applied"] is True
    assert "--max-output-tokens 16384 not applied" in res["max_output_tokens_note"]
    assert res["self_repair"]["max_output_tokens"] == 2000


def test_lcb_clamped_budget_on_the_wire_is_the_recorded_one(monkeypatch, endpoint):
    # The cap comes from OfficialSampling, not the sample meta, so the value base clamps to the
    # window is the value the request carries and the value max_tokens_sent records.
    _lcb_setup(monkeypatch)
    monkeypatch.setenv("GBENCH_LCB_INCLUDE_SELF_REPAIR", "0")
    base.set_run_knobs(max_model_len=2100)
    endpoint.state["reply"] = lambda body: ("```python\nprint(6)\n```", None, "stop")
    res = L.run_lcb("m", endpoint.url, 4)
    wire = sorted({b["max_tokens"] for b in endpoint.records})
    assert wire == res["max_tokens_sent"] and max(wire) < 2000
    assert f"max_tokens={wire}; lcb_runner sends 2000" in res["leaderboard_comparable_reason"]


def test_lcb_thinking_is_never_capped_at_the_non_reasoning_2000(monkeypatch, endpoint):
    _lcb_setup(monkeypatch)
    base.set_run_knobs(max_output_tokens=32768)
    endpoint.state["reply"] = lambda body: (
        "```python\nprint(sum(map(int, input().split())))\n```", "reasoning", "stop")
    res = L.run_lcb("m", endpoint.url, 4, enable_thinking=True)
    assert len(endpoint.records) == 10                 # every program accepted: no repairs
    for body in endpoint.records:
        assert body["max_tokens"] == 32768
        assert body["chat_template_kwargs"]["enable_thinking"] is True
        # a thinking run samples at 1.0 (base.resolve_temperature); top_p stays lcb_runner's
        assert body["temperature"] == 1.0 and body["top_p"] == 0.95
    rec = res["official_sampling"]
    assert rec["temperature"] == 0.2 and rec["temperature_matches_official"] is False
    assert res["temperature_source"] == "thinking"
    assert rec["max_tokens"] == 2000 and rec["max_tokens_applied"] is False
    assert rec["max_tokens_note"] == base.REASONING_CAP_NOTE
    assert base.REASONING_CAP_NOTE in res["max_output_tokens_note"]
    assert "--max-output-tokens 32768" in res["max_output_tokens_note"]
    assert "OpenAIChat" in res["max_output_tokens_note"]
    assert res["max_tokens_sent"] == [32768] and res["self_repair"]["max_output_tokens"] == 32768
    assert res["replies_cut_off_in_reasoning"] == 0
    # the only reasons: a thinking run samples at 1.0, and this one-problem fixture has no code
    # execution or test output sample
    assert res["leaderboard_comparable_reason"] == (
        "temperature=1.0 (thinking); lcb_runner samples at 0.2; scenario(s) not run: "
        "code_execution, test_output_prediction (no sample loaded) (LiveCodeBench publishes all "
        "four)")


def test_lcb_thinking_without_a_flag_gets_the_thinking_floor(monkeypatch, endpoint):
    _lcb_setup(monkeypatch)
    endpoint.state["reply"] = lambda body: (
        "```python\nprint(sum(map(int, input().split())))\n```", None, "stop")
    L.run_lcb("m", endpoint.url, 4, enable_thinking=True)
    assert {b["max_tokens"] for b in endpoint.records} == {base.THINKING_MIN_OUTPUT_TOKENS}


def test_lcb_replies_cut_off_in_reasoning_are_not_comparable(monkeypatch, endpoint):
    # The review's probe: a thinking model that runs out of budget inside the reasoning channel.
    _lcb_setup(monkeypatch)
    base.set_run_knobs(max_output_tokens=32768)
    endpoint.state["reply"] = _cut_in_reasoning
    res = L.run_lcb("m", endpoint.url, 4, enable_thinking=True)
    assert len(endpoint.records) == 20
    assert {b["max_tokens"] for b in endpoint.records} == {32768}     # not lcb_runner's 2000
    assert res["accuracy"] == 0.0
    assert res["replies_cut_off_in_reasoning"] == 20
    assert res["self_repair"]["cut_off_in_reasoning"] == 10
    assert res["leaderboard_comparable"] is False
    assert ("20 repl(ies) (10 generation, 10 self-repair) reached this run's max_tokens 32768 "
            "inside the reasoning channel") in res["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# multipl_e                                                                    #
# --------------------------------------------------------------------------- #
def _mpe_setup(monkeypatch):
    monkeypatch.setenv("GBENCH_MULTIPL_E_SAMPLES", "2")
    monkeypatch.delenv("GBENCH_MULTIPL_E_LANGS", raising=False)
    monkeypatch.delenv("GBENCH_MULTIPL_E_TEMPERATURE", raising=False)
    monkeypatch.setattr(M, "check_multipl_e_prerequisites", lambda: (True, ""))
    gold = json.dumps({"name": "HumanEval_0", "language": "jl", "prompt": "p", "tests": "t",
                       "stop_tokens": []})
    monkeypatch.setattr(M, "_load_multipl_e_samples", lambda **k: [
        ([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], gold,
         {"category": "jl", "name": "HumanEval_0"})])

    def fake_scorer(image, workers, metrics):
        async def _score(traces):
            for t in traces:
                t["is_correct"], t["status"] = False, "OK"
            metrics.update(chat_adapter_parse_failures=0, chat_adapter_empty_responses=0,
                           multipl_e_no_response=0)
        return _score

    monkeypatch.setattr(M, "_make_scorer", fake_scorer)


def test_multipl_e_budget_per_mode(monkeypatch, endpoint):
    _mpe_setup(monkeypatch)
    base.set_run_knobs(max_output_tokens=16384)
    r = M.run_multipl_e("m", endpoint.url, 2)
    assert {b["max_tokens"] for b in endpoint.records} == {2048}
    assert {b["temperature"] for b in endpoint.records} == {0.0}    # gbench's no-think 0.0
    assert all("top_p" not in b for b in endpoint.records)
    assert "--max-output-tokens 16384 not applied" in r["max_output_tokens_note"]
    assert r["official_sampling"]["max_tokens"] == 2048
    endpoint.records.clear()
    r = M.run_multipl_e("m", endpoint.url, 2, enable_thinking=True)
    assert {b["max_tokens"] for b in endpoint.records} == {16384}
    # a thinking run samples at 1.0 (base.resolve_temperature), not the recipe's 0.2
    assert {b["temperature"] for b in endpoint.records} == {1.0}
    assert ("temperature=1.0 (thinking; the upstream chat recipe uses 0.2)"
            in r["leaderboard_comparable_reason"])
    assert r["official_sampling"]["max_tokens_applied"] is False
    assert base.REASONING_CAP_NOTE in r["max_output_tokens_note"]
    assert r["replies_cut_off_in_reasoning"] == 0


def test_multipl_e_thinking_reply_cut_off_in_reasoning_is_not_comparable(monkeypatch, endpoint):
    _mpe_setup(monkeypatch)
    base.set_run_knobs(max_output_tokens=16384)
    endpoint.state["reply"] = _cut_in_reasoning
    r = M.run_multipl_e("m", endpoint.url, 2, enable_thinking=True)
    assert r["replies_cut_off_in_reasoning"] == 2
    assert ("2 repl(ies) reached this run's max_tokens 16384 inside the reasoning channel"
            in r["leaderboard_comparable_reason"])


_FIXTURE = b'''"""A stand-in with the shape of chat_completions.py."""
from tqdm.auto import tqdm


class Completion(dspy.Signature):
    """
    Stand-in instructions.
    """

    lang: str = dspy.InputField(description="Language")
    prefix: str = dspy.InputField(description="Prefix")
    completion: str = dspy.OutputField(description="Completion")


def stop_at_stop_token(decoded_string, stop_tokens):
    return decoded_string


def postprocess_completion(lang, problem_filename, stop_tokens, prompt, completion):
    return completion
'''


def test_multipl_e_whitespace_only_reply_is_a_parse_failure_as_in_dspy(monkeypatch):
    # dspy 3.0.0b1 Adapter._call_postprocess tests the raw content (`if text:`): "" is empty
    # (upstream's script then stops), "\n\n" is parsed and rejected (completion "").
    monkeypatch.setattr(M, "_UPSTREAM", None)
    monkeypatch.setattr(M, "CHAT_SCRIPT_SHA256", hashlib.sha256(_FIXTURE).hexdigest())

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return _FIXTURE

    monkeypatch.setattr(M.urllib.request, "urlopen", lambda url, timeout=None: _Resp())

    def fake_run(cmd, **k):
        # the evaluator's verdict on each program (all fail: every one is empty)
        out = Path(next(a.split(":", 1)[0] for a in cmd if a.endswith(":/out:rw")))
        for p in sorted(out.glob("*.json")):
            if not p.name.endswith(".results.json"):
                name = json.loads(p.read_text())["name"]
                (out / f"{name}.results.json").write_text(json.dumps(
                    {"name": name, "results": [{"status": "Exception", "exit_code": 1}]}))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(M.subprocess, "run", fake_run)
    gold = json.dumps({"name": "HumanEval_0", "language": "jl", "prompt": "", "tests": "",
                       "stop_tokens": []})
    traces = [{"gold_answer": gold, "response_text": "\n\n"},
              {"gold_answer": gold, "response_text": "   "},
              {"gold_answer": gold, "response_text": ""}]
    metrics = {}
    asyncio.run(M._make_scorer("img", 1, metrics)(traces))
    assert metrics["chat_adapter_parse_failures"] == 2
    assert metrics["chat_adapter_empty_responses"] == 1
    assert 'completion "" as upstream records it' in traces[0]["scoring_note"]
    assert traces[2]["scoring_note"] == "empty response"


# --------------------------------------------------------------------------- #
# bigcodebench                                                                 #
# --------------------------------------------------------------------------- #
def _bcb_setup(monkeypatch):
    monkeypatch.delenv("GBENCH_BIGCODEBENCH_SUBSET", raising=False)
    monkeypatch.delenv("GBENCH_BIGCODEBENCH_TEMPERATURE", raising=False)
    monkeypatch.setattr(B, "check_bigcodebench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(B, "_load_bigcodebench_samples", lambda limit=None, subset="full": [
        ([{"role": "user", "content": B.make_raw_chat_prompt("Do X.", split)}], "BigCodeBench/0",
         {"category": split, "task_id": "BigCodeBench/0", "split": split, "hard": False})
        for split in B.SPLITS])

    def fake_scorer(subset, parallel, metrics, image):
        async def _score(traces):
            for t in traces:
                t["is_correct"], t["status"] = True, "OK"
        return _score

    monkeypatch.setattr(B, "_make_scorer", fake_scorer)


def test_bigcodebench_no_think_sends_1280_greedy_on_the_prefill_path(monkeypatch, endpoint):
    _bcb_setup(monkeypatch)
    base.set_run_knobs(max_output_tokens=4096)
    endpoint.state["reply"] = lambda body: ("\ndef task_func():\n    pass\n```", None, "stop")
    r = B.run_bigcodebench("m", endpoint.url, 2)
    assert len(endpoint.records) == 2
    for body in endpoint.records:
        assert body["max_tokens"] == 1280 and body["temperature"] == 0.0
        assert body["top_p"] == 1.0 and body["stop"] == B.SERVER_STOP
        assert body["continue_final_message"] is True
    assert "--max-output-tokens 4096 not applied" in r["max_output_tokens_note"]
    assert r["official_sampling"]["max_tokens"] == 1280
    assert r["leaderboard_comparable"] is True, r.get("leaderboard_comparable_reason")


def test_bigcodebench_thinking_follows_make_requests_served_model_request(monkeypatch, endpoint):
    _bcb_setup(monkeypatch)
    base.set_run_knobs(max_output_tokens=32768)
    # even where base forwards reasoning_effort, the served-model request carries none
    monkeypatch.setattr(base, "ALLOWED_API_KEYS", base.ALLOWED_API_KEYS | {"reasoning_effort"})
    endpoint.state["reply"] = lambda body: ("```python\ndef task_func():\n    pass\n```",
                                            "reasoning", "stop")
    r = B.run_bigcodebench("m", endpoint.url, 2, enable_thinking=True)
    assert len(endpoint.records) == 2
    for body in endpoint.records:
        # make_request for a served model: top_p 0.95; the run's budget replaces the 1280, and
        # a thinking run samples at 1.0 (base.resolve_temperature), not greedy
        assert body["temperature"] == 1.0 and body["top_p"] == 0.95
        assert body["max_tokens"] == 32768
        assert "reasoning_effort" not in body and "continue_final_message" not in body
        assert body["chat_template_kwargs"]["enable_thinking"] is True
    rec = r["official_sampling"]
    assert "reasoning_settings" not in rec and "send_max_tokens" not in rec
    assert rec["temperature_matches_official"] is False and "for a served model" in rec["reference"]
    assert rec["max_tokens"] == 1280 and rec["max_tokens_applied"] is False
    assert rec["max_tokens_sent"] == 32768 and rec["max_tokens_note"] == base.REASONING_CAP_NOTE
    assert r["max_output_tokens_note"].startswith(base.REASONING_CAP_NOTE)
    assert "--max-output-tokens 32768" in r["max_output_tokens_note"]
    assert "reasoning_effort" not in r
    assert r["leaderboard_comparable_reason"] == (
        "temperature=1.0 (thinking; the leaderboard is greedy, temperature 0.0)")


def test_bigcodebench_thinking_sends_no_reasoning_effort_and_needs_none(monkeypatch, endpoint):
    # whether or not base forwards the field, the served-model request has no reasoning_effort,
    # and its absence is no comparability gap
    _bcb_setup(monkeypatch)
    monkeypatch.setattr(base, "ALLOWED_API_KEYS", base.ALLOWED_API_KEYS - {"reasoning_effort"})
    endpoint.state["reply"] = lambda body: ("```python\nx = 1\n```", None, "stop")
    r = B.run_bigcodebench("m", endpoint.url, 2, enable_thinking=True)
    assert all("reasoning_effort" not in b for b in endpoint.records)
    assert "reasoning_effort" not in r
    # the thinking run's temperature 1.0 is the only gap
    assert r["leaderboard_comparable_reason"] == (
        "temperature=1.0 (thinking; the leaderboard is greedy, temperature 0.0)")


def test_bigcodebench_thinking_with_an_operator_temperature_is_not_comparable(monkeypatch,
                                                                            endpoint):
    _bcb_setup(monkeypatch)
    # the served-model path is greedy: an operator temperature of 0.0 is the official one
    monkeypatch.setenv("GBENCH_BIGCODEBENCH_TEMPERATURE", "0.0")
    r = B.run_bigcodebench("m", endpoint.url, 2, enable_thinking=True)
    assert {b["temperature"] for b in endpoint.records} == {0.0}
    assert r["leaderboard_comparable"] is True, r.get("leaderboard_comparable_reason")
    endpoint.records.clear()
    monkeypatch.setenv("GBENCH_BIGCODEBENCH_TEMPERATURE", "0.7")
    r = B.run_bigcodebench("m", endpoint.url, 2, enable_thinking=True)
    assert {b["temperature"] for b in endpoint.records} == {0.7}
    assert r["official_sampling"]["temperature_matches_official"] is False
    assert ("temperature=0.7 (env:GBENCH_BIGCODEBENCH_TEMPERATURE; the leaderboard is greedy, "
            "temperature 0.0)") in r["leaderboard_comparable_reason"]


def test_bigcodebench_probes_only_the_prefill_fields(monkeypatch, endpoint):
    # the reasoning_effort probe went with make_request's vendor-name branch; the prefill probe
    # stays, and a --thinking run (no prefill) probes nothing
    assert not hasattr(B, "_reasoning_effort_forwarded") and not hasattr(B, "REASONING_EFFORT")
    assert B._prefill_supported() is True
    _bcb_setup(monkeypatch)
    monkeypatch.setattr(B, "_probe_request", lambda *a, **k: pytest.fail("no probe expected"))
    B.run_bigcodebench("m", endpoint.url, 2, enable_thinking=True)
    assert len(endpoint.records) == 2


def test_upstream_caps_are_official_sampling_not_sample_meta():
    assert L.OFFICIAL_SAMPLING.max_tokens == 2000 and L.OFFICIAL_SAMPLING.reasoning is None
    assert M.OFFICIAL_SAMPLING.max_tokens == 2048 and M.OFFICIAL_SAMPLING.reasoning is None
    assert B.SAMPLING[True].max_tokens == 1280 and B.SAMPLING[True].reasoning is None
    # the openai path declares no reasoning-model setting: make_request's branch for vendor model
    # names is not one for a served model
    assert B.SAMPLING[False].max_tokens == 1280 and B.SAMPLING[False].reasoning is None
    assert (B.SAMPLING[False].temperature, B.SAMPLING[False].params) == (0.0, {"top_p": 0.95})
    for name, sampling in (("lcb", L.OFFICIAL_SAMPLING), ("multipl_e", M.OFFICIAL_SAMPLING),
                           ("bigcodebench", B.SAMPLING[True]),
                           ("bigcodebench", B.SAMPLING[False])):
        assert base.resolve_token_budget(name, sampling, thinking=False).policy == "official"
        assert base.resolve_token_budget(name, sampling,
                                         thinking=True).policy == "reasoning_run_budget"
    for name in ("lcb", "multipl_e", "bigcodebench"):
        src = (REPO / "gbench" / "runners" / "eval_suites" / f"{name}.py").read_text()
        code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
        assert '"max_tokens":' not in code, name


# --------------------------------------------------------------------------- #
# ojbench                                                                      #
# --------------------------------------------------------------------------- #
def _oj_setup(monkeypatch):
    monkeypatch.setenv("GBENCH_OJBENCH_SAMPLES", "1")
    monkeypatch.delenv("GBENCH_OJBENCH_TEMPERATURE", raising=False)
    monkeypatch.setattr(OJ, "check_ojbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(OJ, "served_model_recommended_sampling",
                        lambda model_name=None: (dict(OJ.GEMMA4_RECOMMENDED), "stub"))
    monkeypatch.setattr(OJ, "_load_ojbench_samples", lambda limit=None: [
        ([{"role": "user", "content": "solve"}], 1,
         {"category": lang, "row": {"id": 1, "dataset": "NOI", "language": lang,
                                    "difficulty": "easy"}}) for lang in ("cpp", "python")])

    def fake_scorer(workers):
        async def _score(traces):
            for t in traces:
                t["is_correct"], t["status"] = False, "OK"
        return _score

    monkeypatch.setattr(OJ, "_make_scorer", fake_scorer)


def test_ojbench_no_think_sends_no_max_tokens(monkeypatch, endpoint):
    _oj_setup(monkeypatch)
    base.set_run_knobs(max_output_tokens=8192)
    r = OJ.run_ojbench("m", endpoint.url, 2)
    assert endpoint.records and all("max_tokens" not in b for b in endpoint.records)
    assert {b["temperature"] for b in endpoint.records} == {0.0}    # gbench's no-think 0.0
    assert r["official_sampling"]["send_max_tokens"] is False
    assert "--max-output-tokens 8192 not applied" in r["max_output_tokens_note"]
    assert "max_tokens" not in (r.get("leaderboard_comparable_reason") or "")


@pytest.mark.parametrize("knob,sent", [(32768, 65536), (131072, 131072), (None, 65536)])
def test_ojbench_thinking_gets_the_papers_64k_never_below_the_flag(monkeypatch, endpoint, knob,
                                                                   sent):
    _oj_setup(monkeypatch)
    if knob:
        base.set_run_knobs(max_output_tokens=knob)
    r = OJ.run_ojbench("m", endpoint.url, 2, enable_thinking=True)
    assert {b["max_tokens"] for b in endpoint.records} == {sent}
    rec = r["official_sampling"]
    assert rec["reasoning_settings"] is True and rec["max_tokens"] == 65536
    # the paper's 64k is applied only when it is what is sent (a larger flag replaces it)
    assert rec["max_tokens_applied"] is (sent == 65536) and rec["max_tokens_sent"] == sent
    assert "max_tokens" not in (r.get("leaderboard_comparable_reason") or "")


def test_ojbench_thinking_clamped_below_64k_is_named(monkeypatch, endpoint):
    _oj_setup(monkeypatch)
    base.set_run_knobs(max_model_len=32768)
    r = OJ.run_ojbench("m", endpoint.url, 2, enable_thinking=True)
    assert max(b["max_tokens"] for b in endpoint.records) < 65536
    assert "OJBench gives reasoning models 64k, 65536" in r["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# scicode and aider_polyglot: none sent, recorded through the same hook        #
# --------------------------------------------------------------------------- #
def test_scicode_records_the_policy_and_sizes_the_timeout_for_an_uncapped_reply(monkeypatch):
    monkeypatch.setattr(SC, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(SC, "resolve_test_data", lambda: ("/nonexistent.h5", "test"))
    monkeypatch.setattr(SC, "_digest", lambda path: SC._TEST_DATA_SHA256)
    monkeypatch.setattr(SC, "_load_scicode_problems", lambda limit=None: [
        {"problem_id": "1", "problem_name": "p1", "required_dependencies": "",
         "sub_steps": [{"step_number": "1.1", "test_cases": ["assert True"]}]}])
    monkeypatch.delenv("GBENCH_SCICODE_TEMPERATURE", raising=False)
    monkeypatch.delenv("GBENCH_SCICODE_HTTP_TIMEOUT_S", raising=False)
    seen = {}

    def fake_one(client, model, prob, **k):
        seen.update(k)
        return {"problem_id": "1", "problem_name": "p1", "steps": [
            {"step_number": "1.1", "special": False, "gen_failed": False, "correct": True,
             "response_text": "x", "error": None}]}

    import openai
    clients = []
    monkeypatch.setattr(openai, "OpenAI", lambda **k: clients.append(k) or object())
    monkeypatch.setattr(SC, "_process_one_problem", fake_one)
    base.set_run_knobs(max_output_tokens=16384)
    r = SC.run_scicode("m", "http://h/v1", limit=1, concurrency=1, enable_thinking=True)
    assert seen["max_tokens"] is None
    assert r["max_output_tokens_note"].startswith("not sent: the official harness sends no "
                                                  "max_tokens")
    assert "--max-output-tokens 16384 not applied" in r["max_output_tokens_note"]
    assert r["official_sampling"]["send_max_tokens"] is False
    assert clients[0]["timeout"] == float(base.request_timeout_s(base.UNCAPPED_TIMEOUT_TOKENS))


def test_aider_records_that_it_sends_no_max_tokens(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_AIDER_BENCHMARK_DIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_AIDER_MODEL", "openai/m")
    monkeypatch.delenv("GBENCH_AIDER_LANGUAGES", raising=False)
    monkeypatch.delenv("GBENCH_AIDER_EXTRA_ARGS", raising=False)
    monkeypatch.setattr(A, "_check_prereqs", lambda image, bench: None)
    # the network checks before the run container are docker calls of their own
    # (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC_GUARD, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC_GUARD.container_url(url))

    def fake_run(cmd, **k):
        d = tmp_path / "2026-run--gbench" / "python" / "exercises" / "practice" / "x0"
        d.mkdir(parents=True, exist_ok=True)
        (d / ".aider.results.json").write_text(json.dumps({"language": "python",
                                                           "tests_outcomes": [True]}))
        report = {"files_loaded": ["/gbench-aider/model-settings.yml"], "extra_params": {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            "timeout": A.model_request_timeout_s(), "max_retries": 0, "temperature": 1.0}}
        return types.SimpleNamespace(returncode=0, stderr="",
                                     stdout=A._CHECK_MARKER + json.dumps(report) + "\n")

    monkeypatch.setattr(A.subprocess, "run", fake_run)
    (tmp_path / "polyglot-benchmark" / "python" / "exercises" / "practice" / "x0").mkdir(
        parents=True, exist_ok=True)   # the exercises a limited run selects from
    r = A.run_aider_polyglot("m", "http://127.0.0.1:1/v1", concurrency=1, limit=1,
                             enable_thinking=True, max_output_tokens=32768)
    assert r["max_output_tokens_note"].startswith("not sent: the official harness sends no "
                                                  "max_tokens")
    assert "--max-output-tokens 32768 not applied" in r["max_output_tokens_note"]
    # a model whose aider settings carry a cap gets it from aider, and the note says so
    assert A._max_tokens_note({"max_tokens": 64000}, True, 32768) == (
        "aider sends max_tokens=64000 from the model's aider settings with every request; "
        "--max-output-tokens 32768 not applied")


# --------------------------------------------------------------------------- #
# livebench                                                                    #
# --------------------------------------------------------------------------- #
#: LiveBench's config for gemma-4-31b-it at 0baa4aa7 (livebench/model/model_configs/google.yml),
#: as dataclasses.asdict(ModelConfig) returns it.
GEMMA4_31B = {
    "display_name": "gemma-4-31b-it",
    "api_name": {"google": "gemma-4-31b-it",
                 "https://openrouter.ai/api/v1": "google/gemma-4-31b-it"},
    "default_provider": "https://openrouter.ai/api/v1",
    "api_keys": {"https://openrouter.ai/api/v1": "OPENROUTER_API_KEY"},
    "aliases": ["gemma-4-31b"],
    "api_kwargs": {"default": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 65536},
                   "https://openrouter.ai/api/v1": {"extra_body": {"reasoning": {"enabled": True}}}},
    "prompt_prefix": None, "preserve_reasoning": None, "cost_per_million": None, "api_base": None,
    "agentic_native_tools": None,
}


def test_livebench_config_candidates_include_the_hf_ids_last_component():
    base.set_run_knobs(tokenizer="google/gemma-4-31B-it")
    assert LB._config_candidates("gemma") == ["gemma", "google/gemma-4-31B-it", "gemma-4-31B-it"]
    assert LB._config_candidates("/models/gemma-4-31B-it/")[:2] == ["/models/gemma-4-31B-it",
                                                                     "gemma-4-31B-it"]


def test_livebench_translates_the_gemma_4_config_for_the_local_endpoint():
    sampling, kept, cap_key, gaps = LB.translate_model_config(GEMMA4_31B, True)
    assert (sampling.temperature, sampling.params, sampling.max_tokens) == (
        1.0, {"top_p": 0.95}, 65536)
    assert sampling.reasoning.max_tokens == 65536
    assert kept == {"temperature": 1.0, "top_p": 0.95} and cap_key == "max_tokens"
    assert gaps == []
    _, _, _, gaps = LB.translate_model_config(GEMMA4_31B, False)
    assert gaps == ["LiveBench's config for gemma-4-31b-it runs with reasoning enabled "
                    "(https://openrouter.ai/api/v1 extra_body.reasoning.enabled); this run is "
                    "no-think"]
    odd = dict(GEMMA4_31B, api_kwargs={"default": {"temperature": 0.6, "thinking_config": {}}})
    sampling, kept, cap_key, gaps = LB.translate_model_config(odd, True)
    assert cap_key is None and sampling.max_tokens == LB.LIVEBENCH_MAX_TOKENS
    assert sampling.reasoning is None
    assert gaps and "thinking_config" in gaps[0]


def _livebench(monkeypatch, tmp_path, config=None, commit=LB.LIVEBENCH_COMMIT, **kwargs):
    monkeypatch.setenv("GBENCH_LIVEBENCH_RELEASE", "2025-04-02")
    monkeypatch.delenv("GBENCH_LIVEBENCH_CATEGORIES", raising=False)
    monkeypatch.delenv("GBENCH_LIVEBENCH_EXTRA_ARGS", raising=False)
    monkeypatch.delenv("GBENCH_LIVEBENCH_TEMPERATURE", raising=False)
    monkeypatch.delenv("GBENCH_LIVEBENCH_MODEL", raising=False)
    monkeypatch.setattr(LB, "_check_prereqs", lambda image: None)
    # the network checks before the run container are docker calls of their own
    # (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC_GUARD, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC_GUARD.container_url(url))
    looked_up = {}

    def fake_lookup(image, names):
        looked_up["names"] = names
        return {"name": names[-1] if config else None, "config": config, "commit": commit}

    monkeypatch.setattr(LB, "upstream_model_config", fake_lookup)
    monkeypatch.setattr(LB.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))
    seen = {}

    def fake_run(cmd, **k):
        seen["inner"] = cmd[-1]
        cols = LB._CATEGORIES
        (tmp_path / "all_groups.csv").write_text(
            "model,average," + ",".join(cols) + "\ngoogle-gemma-4-31b-it,50.0,"
            + ",".join("50.0" for _ in cols) + "\n", encoding="utf-8")
        answers = tmp_path / "model_answer" / "data" / "live_bench" / "math" / "x" / "model_answer"
        answers.mkdir(parents=True, exist_ok=True)
        (answers / "google-gemma-4-31b-it.jsonl").write_text(json.dumps(
            {"question_id": "q", "choices": [{"index": 0, "turns": ["ok"]}],
             "total_output_tokens": 10}) + "\n", encoding="utf-8")
        # instruction_following is scored: its answer rows are copied out too
        if_dir = (tmp_path / "model_answer" / "data" / "live_bench" / "instruction_following"
                  / "paraphrase" / "model_answer")
        if_dir.mkdir(parents=True, exist_ok=True)
        (if_dir / "google-gemma-4-31b-it.jsonl").write_text(json.dumps(
            {"question_id": "i1", "choices": [{"index": 0, "turns": ["ok"]}],
             "total_output_tokens": 10}) + "\n", encoding="utf-8")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(LB.subprocess, "run", fake_run)
    r = LB.run_livebench("google/gemma-4-31B-it", "http://h/v1", 4, **kwargs)
    cfg = json.load(open(tmp_path / LB._MODEL_CONFIG_FILE))
    return r, seen["inner"], cfg, looked_up


def test_livebench_model_without_a_config_no_think_sends_4096(monkeypatch, tmp_path):
    base.set_run_knobs(max_output_tokens=32768)
    r, inner, cfg, looked = _livebench(monkeypatch, tmp_path, max_output_tokens=32768)
    assert looked["names"] == ["google/gemma-4-31B-it", "gemma-4-31B-it"]
    assert "--max-tokens 4096 " in inner and "--force-temperature" not in inner
    assert "max_tokens" not in cfg["api_kwargs"]["default"]
    assert r["max_output_tokens"] == 4096 and r["max_output_tokens_source"] == "official"
    assert "--max-output-tokens 32768 not applied" in r["max_output_tokens_note"]
    assert r["official_sampling"]["max_tokens"] == 4096
    assert r["livebench_model_config"] is None
    assert r["leaderboard_comparable"] is True, r.get("leaderboard_comparable_reason")


def test_livebench_model_without_a_config_thinking_gets_the_run_budget(monkeypatch, tmp_path):
    base.set_run_knobs(max_output_tokens=32768)
    r, inner, cfg, _ = _livebench(monkeypatch, tmp_path, enable_thinking=True,
                                  max_output_tokens=32768)
    assert "--max-tokens 32768 " in inner
    assert cfg["api_kwargs"]["default"]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True}
    assert r["official_sampling"]["max_tokens_applied"] is False
    assert base.REASONING_CAP_NOTE in r["max_output_tokens_note"]
    # a thinking run samples at 1.0, written into the config (never --force-temperature)
    assert cfg["api_kwargs"]["default"]["temperature"] == 1.0
    assert "--force-temperature" not in inner
    assert (r["temperature"], r["temperature_source"]) == (1.0, "thinking")
    assert r["official_sampling"]["temperature_matches_official"] is False
    assert r["leaderboard_comparable_reason"] == (
        "temperature=1.0 (thinking; sent as the config's temperature) (LiveBench samples at "
        "temperature 0, or the question's required temperature)")


def test_livebench_thinking_floor_fitted_to_the_window_is_what_the_result_says(monkeypatch,
                                                                             tmp_path):
    base.set_run_knobs(max_model_len=32768)
    r, inner, _, _ = _livebench(monkeypatch, tmp_path, enable_thinking=True)
    fitted = 32768 - LB._PROMPT_RESERVE_TOKENS
    assert f"--max-tokens {fitted} " in inner
    assert r["official_sampling"]["max_tokens_sent"] == fitted
    assert f"lowered to {fitted} to fit the served context window" in r["max_output_tokens_note"]


@pytest.mark.parametrize("knob,sent", [(32768, 65536), (131072, 131072)])
def test_livebench_gemma_4_31b_runs_its_livebench_config(monkeypatch, tmp_path, knob, sent):
    base.set_run_knobs(max_output_tokens=knob)
    r, inner, cfg, _ = _livebench(monkeypatch, tmp_path, config=GEMMA4_31B, enable_thinking=True,
                                  max_output_tokens=knob)
    default = cfg["api_kwargs"]["default"]
    assert default == {"temperature": 1.0, "top_p": 0.95, "max_tokens": sent,
                       "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
    assert cfg["api_name"] == {"local": "google/gemma-4-31B-it"}
    assert "--max-tokens" not in inner and "--force-temperature" not in inner
    assert r["temperature"] == 1.0 and r["max_output_tokens"] == sent
    assert r["official_sampling"]["reasoning_settings"] is True
    assert r["official_sampling"]["temperature_matches_official"] is True
    assert r["livebench_model_config"]["display_name"] == "gemma-4-31b-it"
    assert r["livebench_model_config"]["provider"] == "https://openrouter.ai/api/v1"
    assert r["leaderboard_comparable"] is True, r.get("leaderboard_comparable_reason")


def test_livebench_gemma_4_31b_no_think_is_not_its_leaderboard_row(monkeypatch, tmp_path):
    r, inner, cfg, _ = _livebench(monkeypatch, tmp_path, config=GEMMA4_31B)
    assert cfg["api_kwargs"]["default"]["max_tokens"] == 65536
    assert cfg["api_kwargs"]["default"]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False}
    assert r["leaderboard_comparable"] is False
    assert "runs with reasoning enabled" in r["leaderboard_comparable_reason"]


def test_livebench_operator_temperature_is_written_into_the_config(monkeypatch, tmp_path):
    monkeypatch.setattr(LB, "resolve_official_sampling", lambda name, s, thinking=False: (
        0.3, "cli:--temperature"))
    r, inner, cfg, _ = _livebench(monkeypatch, tmp_path, config=GEMMA4_31B, enable_thinking=True)
    assert cfg["api_kwargs"]["default"]["temperature"] == 0.3
    assert "--force-temperature" not in inner
    assert ("temperature=0.3 (cli:--temperature; LiveBench's config for gemma-4-31b-it samples "
            "at 1.0)") in r["leaderboard_comparable_reason"]


def test_livebench_image_from_another_commit_is_not_comparable(monkeypatch, tmp_path):
    r, _, _, _ = _livebench(monkeypatch, tmp_path, commit="bc5164c1841e3cf099b4fb48c59b49ee0a734bef")
    assert r["livebench_image_commit"].startswith("bc5164c1")
    assert "holds LiveBench bc5164c1841e, not the pinned 0baa4aa7a78d" in \
        r["leaderboard_comparable_reason"]


def test_livebench_lookup_reads_the_images_answer(monkeypatch):
    calls = []

    def fake_run(cmd, **k):
        calls.append(cmd)
        return types.SimpleNamespace(returncode=0, stderr="", stdout="noise\n" + LB._CONFIG_MARKER
                                     + json.dumps({"name": "gemma-4-31B-it", "config": GEMMA4_31B,
                                                   "commit": "abc", "error": None}) + "\n")

    monkeypatch.setattr(LB.subprocess, "run", fake_run)
    out = LB.upstream_model_config("img", ["google/gemma-4-31B-it", "gemma-4-31B-it"])
    assert out == {"name": "gemma-4-31B-it", "config": GEMMA4_31B, "commit": "abc"}
    cmd = calls[0]
    assert cmd[:5] == ["docker", "run", "--rm", "--network", "none"]
    assert cmd[-2:] == ["google/gemma-4-31B-it", "gemma-4-31B-it"]
    monkeypatch.setattr(LB.subprocess, "run", lambda cmd, **k: types.SimpleNamespace(
        returncode=1, stdout="", stderr="ModuleNotFoundError"))
    with pytest.raises(RuntimeError, match="printed no result"):
        LB.upstream_model_config("img", ["x"])
    monkeypatch.setattr(LB.subprocess, "run", lambda cmd, **k: types.SimpleNamespace(
        returncode=0, stderr="", stdout=LB._CONFIG_MARKER + json.dumps(
            {"name": "x", "config": None, "commit": None, "error": "Multiple model configs"})))
    with pytest.raises(RuntimeError, match="Multiple model configs"):
        LB.upstream_model_config("img", ["x"])


def _image_exists(image):
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "image", "inspect", image], capture_output=True,
                              timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


@pytest.mark.skipif(not _image_exists(LB.LIVEBENCH_IMAGE_DEFAULT),
                    reason=f"needs the locally built {LB.LIVEBENCH_IMAGE_DEFAULT} image")
def test_livebench_lookup_in_the_real_image_finds_the_gemma_4_config():
    out = LB.upstream_model_config(LB.LIVEBENCH_IMAGE_DEFAULT,
                                   LB._config_candidates("google/gemma-4-31B-it"))
    assert out["config"]["display_name"] == "gemma-4-31b-it"
    assert out["config"]["api_kwargs"]["default"]["max_tokens"] == 65536
    none = LB.upstream_model_config(LB.LIVEBENCH_IMAGE_DEFAULT, ["not-a-livebench-model-xyz"])
    assert none["config"] is None and none["name"] is None


# --------------------------------------------------------------------------- #
# docs state the per-mode choice                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,needles", [
    ("lcb", ("OpenAIChat", "not applied to reasoning", "replies_cut_off_in_reasoning")),
    ("multipl_e", ("one mode", "not applied to reasoning", "replies_cut_off_in_reasoning",
                   "whitespace")),
    ("bigcodebench", ("reasoning-model branch", "reasoning_effort")),
    ("ojbench", ("64k", "default context length")),
    ("livebench", ("gemma-4-31b-it", "4096", "livebench_model_config")),
    ("scicode", ("UNCAPPED_TIMEOUT_TOKENS",)),
    ("aider_polyglot", ("max_output_tokens_note",)),
])
def test_docs_state_the_per_mode_token_choice(name, needles):
    doc = (REPO / "docs" / "evals" / f"{name}.md").read_text(encoding="utf-8")
    for needle in needles:
        assert needle in doc, (name, needle)
    assert "with and without `--eval-thinking`, and the run's `--max-output-tokens`" not in doc


# --------------------------------------------------------------------------- #
# hmmt and aime: MathArena's rule applied to the served model                  #
# --------------------------------------------------------------------------- #
def _matharena_setup(monkeypatch, recommended):
    from gbench.runners.eval_suites import matharena_common as MA
    monkeypatch.setattr(MA, "check_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(MA, "self_check", lambda config: None)
    monkeypatch.delenv("GBENCH_HMMT_TEMPERATURE", raising=False)
    monkeypatch.delenv("GBENCH_AIME_TEMPERATURE", raising=False)
    if isinstance(recommended, Exception):
        def unreadable(model_name=None):
            raise recommended
        monkeypatch.setattr(OJ, "served_model_recommended_sampling", unreadable)
    else:
        monkeypatch.setattr(OJ, "served_model_recommended_sampling",
                            lambda model_name=None: (dict(recommended),
                                                     "Qwen/Qwen3-8B/generation_config.json"))


@pytest.mark.parametrize("suite", ["hmmt", "aime"])
def test_matharena_sends_a_non_gemma_models_own_recommended_settings(monkeypatch, endpoint,
                                                                     suite):
    from gbench.runners.eval_suites import aime as AI
    from gbench.runners.eval_suites import hmmt as H
    mod = H if suite == "hmmt" else AI
    _matharena_setup(monkeypatch, {"temperature": 0.6, "top_p": 0.95, "top_k": 20})
    loader = "_load_hmmt_samples" if suite == "hmmt" else "_load_aime_samples"
    monkeypatch.setattr(mod, loader, lambda *a, **k: [
        ([{"role": "user", "content": "1+1?"}], "2", {"category": "x", "problem_idx": 1})])

    def fake_scorer(comp, config, stats, **kw):
        stats.update(kw)

        async def _score(traces):
            for t in traces:
                t["is_correct"], t["status"] = True, "OK"
        return _score

    monkeypatch.setattr(mod.MA, "make_scorer", fake_scorer)
    base.set_run_knobs(max_output_tokens=32768)
    endpoint.state["reply"] = lambda body: ("\\boxed{2}", None, "stop")
    run = H.run_hmmt if suite == "hmmt" else AI.run_aime
    r = run("m", endpoint.url, 2)
    # gbench's temperature (0.0 without thinking); top_p and top_k are the model's
    for body in endpoint.records:
        assert (body["temperature"], body["top_p"], body["top_k"]) == (0.0, 0.95, 20)
        assert body["max_tokens"] == 32768
    assert r["recommended_sampling"]["top_k"] == 20
    assert r["official_sampling"]["temperature"] == 0.6
    assert "Qwen/Qwen3-8B/generation_config.json" in r["official_sampling"]["reference"]
    assert ("temperature=0.0 (default); MathArena samples each model at its provider-recommended "
            "temperature (0.6 for the served model)") in r["leaderboard_comparable_reason"]
    # a thinking run samples at 1.0 (base.resolve_temperature); top_p and top_k stay the model's
    endpoint.records.clear()
    r = run("m", endpoint.url, 2, enable_thinking=True)
    for body in endpoint.records:
        assert (body["temperature"], body["top_p"], body["top_k"]) == (1.0, 0.95, 20)
    assert r["official_sampling"]["temperature_matches_official"] is False
    assert ("temperature=1.0 (thinking); MathArena samples each model at its provider-recommended "
            "temperature (0.6 for the served model)") in r["leaderboard_comparable_reason"]


@pytest.mark.parametrize("suite", ["hmmt", "aime"])
def test_matharena_unreadable_generation_config_is_infra_required(monkeypatch, suite):
    from gbench.runners.eval_suites import aime as AI
    from gbench.runners.eval_suites import hmmt as H
    _matharena_setup(monkeypatch, RuntimeError("could not read generation_config.json of 'm'"))
    run = H.run_hmmt if suite == "hmmt" else AI.run_aime
    monkeypatch.setattr(H if suite == "hmmt" else AI, "run_eval_suite",
                        lambda **k: pytest.fail("no generation before the settings are known"))
    with pytest.raises(RuntimeError, match="provider's recommended temperature, top_p and top_k"):
        run("m", "http://127.0.0.1:1/v1", 1)
