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

"""Shared base and CLI behaviour the suites rely on: context clamping, audio budgeting, the judge
key gate, empty replies, round caps, category routing and suite renames.

  * base.clamp_to_context: minimum answer room min(256, max_tokens) and an exact, slack-free
    count through a simulated vLLM /tokenize that receives the request's chat_template_kwargs.
  * base: audio parts counted as soft tokens and truncated in traces; the judge gate accepts
    GEMINI_API_KEYS; an empty reply the batch scorer marks as the answer is not a failed
    request; a protocol round cap counts model turns.
  * --eval-categories routed per suite; cruxeval behind the code-execution gate.
  * renames semantic_keypoint -> screenspot_pointing, lmsys_noncoding_hard -> wildbench,
    bfcl_v3_live -> bfcl_live, loft_x_arxiv -> loft_scifact, with old names resolving to the new
    ones.
  * canonical_sync entries and the evals extra.

No model server, judge API or dataset hub is contacted; the /tokenize server binds port 0.
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from gbench.runners import evals as E
from gbench.runners.eval_suites import base

REPO = Path(__file__).resolve().parents[1]
SUITE_DIR = REPO / "gbench" / "runners" / "eval_suites"
WINDOW = 131072

# The README was split: these docs hold sections moved verbatim out of the README, so a
# negative check that used to scan the whole README scans README.md plus all of them.
_README_SPLIT_DOCS = (
    "docs/installation.md", "docs/environment-variables.md", "docs/performance.md",
    "docs/eval-suites.md", "docs/golden-set.md", "docs/remote-endpoints.md",
    "docs/results.md", "docs/deployment.md", "docs/cli-reference.md",
)
_README_AND_SPLIT_DOCS = ("README.md", *_README_SPLIT_DOCS)


# --------------------------------------------------------------------------- #
# a simulated vLLM /tokenize endpoint
# --------------------------------------------------------------------------- #
class _TokenizeHandler(BaseHTTPRequestHandler):
    def do_POST(self):                                              # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        self.server.bodies.append(body)
        if self.path != "/tokenize":
            self.send_response(404)
            self.end_headers()
            return
        data = json.dumps({"count": self.server.count_fn(body), "max_model_len": WINDOW,
                           "tokens": []}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def tokenize_server(monkeypatch):
    """A /tokenize server wired into the run knobs. `server.count_fn(body)` decides the count;
    like vLLM, enable_thinking changes the rendered template (here: 7 more tokens)."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _TokenizeHandler)
    srv.bodies = []
    srv.count_fn = lambda body: 100
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(base, "_COUNT_CACHE", {})
    monkeypatch.setattr(base, "_NO_TOKENIZE_ENDPOINT", set())
    base.set_run_knobs(tokenize_url=f"http://127.0.0.1:{srv.server_address[1]}",
                       served_model="org/m", max_model_len=WINDOW)
    try:
        yield srv
    finally:
        srv.shutdown()
        srv.server_close()


def _thinking_aware(base_count):
    def count(body):
        ctk = body.get("chat_template_kwargs") or {}
        return base_count + (7 if ctk.get("enable_thinking") else 0)
    return count


# --------------------------------------------------------------------------- #
# Minimum answer room
# --------------------------------------------------------------------------- #
def _bounds(low, high, mode):
    return lambda messages, tools=None, chat_template_kwargs=None: (low, high, mode)


def test_small_cap_needs_only_its_own_room(monkeypatch):
    # tokenizer mode (slack 64): 100 tokens of room minus 64 leaves 36 >= a 32-token cap
    monkeypatch.setattr(base, "_prompt_token_bounds", _bounds(WINDOW - 100, WINDOW - 100, "tokenizer"))
    assert base.clamp_to_context([{"role": "user", "content": "x"}], 32, WINDOW) == (32, None)
    # the same prompt with no cap (or a big one) still needs 256 tokens of room
    _, why = base.clamp_to_context([{"role": "user", "content": "x"}], None, WINDOW)
    assert why and "does not fit" in why
    _, why = base.clamp_to_context([{"role": "user", "content": "x"}], 4096, WINDOW)
    assert why and "does not fit" in why


def test_headroom_fallback_uses_the_same_need(monkeypatch):
    # estimate mode: high says 40 tokens of room, low says 400; a 64-token cap takes the
    # optimistic (low) headroom because the pessimistic one is below its need
    monkeypatch.setattr(base, "_prompt_token_bounds",
                        _bounds(WINDOW - 400 - 64, WINDOW - 40 - 64, "estimate"))
    assert base.clamp_to_context([{"role": "user", "content": "x"}], 64, WINDOW) == (64, None)


# --------------------------------------------------------------------------- #
# Exact server-mode counting through a simulated vLLM /tokenize
# --------------------------------------------------------------------------- #
def test_tokenize_body_carries_chat_template_kwargs_and_the_cache_keys_on_them(tokenize_server):
    tokenize_server.count_fn = _thinking_aware(1000)
    msgs = [{"role": "user", "content": "hello"}]
    assert base._server_prompt_tokens(msgs, None, {"enable_thinking": False}) == 1000
    assert base._server_prompt_tokens(msgs, None, {"enable_thinking": True}) == 1007
    assert [b.get("chat_template_kwargs") for b in tokenize_server.bodies] == [
        {"enable_thinking": False}, {"enable_thinking": True}]
    assert all(b["add_generation_prompt"] is True for b in tokenize_server.bodies)
    base._server_prompt_tokens(msgs, None, {"enable_thinking": True})
    assert len(tokenize_server.bodies) == 2, "a repeated request is answered from the cache"
    # no kwargs given: the body omits them, as before
    base._server_prompt_tokens(msgs)
    assert "chat_template_kwargs" not in tokenize_server.bodies[-1]


def test_ruler_128k_prompt_fits_a_131072_window_exactly(tokenize_server):
    """RULER builds its top band so prompt + the task's cap fits 131072, leaving ~cap + 5."""
    cap = 64
    tokenize_server.count_fn = _thinking_aware(WINDOW - cap - 5)
    msgs = [{"role": "user", "content": "haystack " * 50}]
    ctk = base.effective_chat_template_kwargs(None, False)
    assert base.clamp_to_context(msgs, cap, None, None, ctk) == (cap, None)
    # Without the request's kwargs the count is not exact for the request, so slack stays
    # and the prompt is refused (what happened to every prompt before the exact count).
    _, why = base.clamp_to_context(msgs, cap, None, None, None)
    assert why and "does not fit" in why


def test_media_keeps_the_slack_even_in_server_mode(tokenize_server):
    cap = 64
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 200}}
    msgs = [{"role": "user", "content": [image, {"type": "text", "text": "where?"}]}]
    # the server counts the text; the image adds its soft-token budget (an estimate)
    tokenize_server.count_fn = lambda body: WINDOW - cap - 5 - base.IMAGE_SOFT_TOKENS
    _, why = base.clamp_to_context(msgs, cap, None, None, {"enable_thinking": False})
    assert why and "does not fit" in why
    sent = tokenize_server.bodies[-1]["messages"][0]["content"]
    assert sent == [{"type": "text", "text": "where?"}], "image parts are not sent to /tokenize"


def test_run_suite_sends_the_ruler_cap_through_the_clamp(tokenize_server):
    """End to end: the sample's own max_tokens (RULER's task cap) passes the pre-send check and
    /tokenize sees the same enable_thinking the chat request sends."""
    cap = 64
    tokenize_server.count_fn = _thinking_aware(WINDOW - cap - 5)
    seen = {}

    async def _fake_send(**k):
        seen["max_tokens"] = k["max_output_tokens"]
        seen["thinking"] = k["thinking"]
        return base.Reply(text="The number is 42.", tool_calls=None, finish_reason="stop")

    samples = [([{"role": "user", "content": "niah " * 40}], "42", {"max_tokens": cap})]
    with mock.patch.object(base, "_send_single_request", _fake_send):
        res = asyncio.run(base._run_suite_async(
            eval_name="w2b_ruler", model_name="m", base_url="http://x/v1", concurrency=1,
            samples=samples, eval_fn=lambda r, g: g in r, max_output_tokens=65536))
    tr = res["sample_traces"][0]
    assert tr["status"] == "OK" and res["over_context_prompts"] == 0
    assert tr["max_tokens_sent"] == cap and seen["max_tokens"] == cap
    assert tokenize_server.bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert seen["thinking"] is False


def test_effective_kwargs_match_what_the_request_sends():
    assert base.effective_chat_template_kwargs(None, True) == {"enable_thinking": True}
    assert base.effective_chat_template_kwargs({"x": 1}, False) == {"x": 1, "enable_thinking": False}
    assert base.effective_chat_template_kwargs({"enable_thinking": True}, False) == {"enable_thinking": True}


# --------------------------------------------------------------------------- #
# Audio parts
# --------------------------------------------------------------------------- #
def _audio_msgs(b64):
    return [{"role": "user", "content": [
        {"type": "text", "text": "Transcribe."},
        {"type": "input_audio", "input_audio": {"data": b64, "format": "mp3"}},
        {"type": "audio_url", "audio_url": {"url": "data:audio/mpeg;base64," + b64}}]}]


def test_audio_parts_are_budgeted_not_counted_as_text(monkeypatch):
    msgs = _audio_msgs("Q" * 370_000)
    text, images, n_msgs, audios = base._prompt_text_and_images(msgs)
    assert (images, audios, n_msgs) == (0, 2, 1) and "QQQQ" not in text
    clean, images, audios = base._text_only_messages(msgs)
    assert audios == 2 and clean[0]["content"] == [{"type": "text", "text": "Transcribe."}]
    monkeypatch.setattr(base, "_server_prompt_tokens", lambda *a, **k: None)
    monkeypatch.setitem(base._RUN_KNOBS, "tokenizer", None)
    low, high, mode = base._prompt_token_bounds(msgs)
    assert mode == "estimate" and high < 2 * base.AUDIO_SOFT_TOKENS + 200
    assert low >= 2 * base.AUDIO_SOFT_TOKENS
    assert base.clamp_to_context(msgs, 8192, WINDOW)[1] is None


def test_server_count_adds_the_audio_budget(tokenize_server):
    tokenize_server.count_fn = lambda body: 50
    assert base._server_prompt_tokens(_audio_msgs("Q" * 1000)) == 50 + 2 * base.AUDIO_SOFT_TOKENS


def test_traces_truncate_audio_payloads():
    out = base._sanitize_for_trace({"messages": _audio_msgs("Q" * 5000)})
    parts = out["messages"][0]["content"]
    assert parts[1]["input_audio"]["data"].endswith("...") and len(parts[1]["input_audio"]["data"]) < 60
    assert parts[1]["input_audio"]["format"] == "mp3"
    assert parts[2]["audio_url"]["url"].startswith("data:audio/") and len(parts[2]["audio_url"]["url"]) < 60
    assert base._sanitize_for_trace("data:audio/wav;base64," + "A" * 500).endswith("...")
    assert base._sanitize_for_trace({"input_audio": {"data": "short"}}) == {"input_audio": {"data": "short"}}


# --------------------------------------------------------------------------- #
# The judge gate reads GEMINI_API_KEYS too
# --------------------------------------------------------------------------- #
def test_judge_gate_accepts_a_key_list_set_only_as_gemini_api_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEYS", "k1,k2")
    seen = []
    monkeypatch.setattr(base, "gemini_key_live_valid", lambda key: (seen.append(key) or (True, "")))
    assert base.gemini_required_skip("simpleqa", "m") is None
    assert seen == ["k1,k2"]


def test_judge_gate_still_hard_errors_with_no_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*GEMINI_API_KEY"):
        base.gemini_required_skip("simpleqa", "m")


def test_package_import_gives_direct_readers_the_first_listed_key():
    env = {k: v for k, v in os.environ.items() if k not in ("GEMINI_API_KEY", "GEMINI_API_KEYS")}
    env["GEMINI_API_KEYS"] = "first-key;second-key"
    out = subprocess.run([sys.executable, "-c", "import os, gbench; print(os.environ['GEMINI_API_KEY'])"],
                         cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "first-key"


def test_charxiv_judge_reads_the_key_like_the_gate():
    src = (SUITE_DIR / "charxiv.py").read_text(encoding="utf-8")
    assert "have_key = bool(gemini_api_keys())" in src
    assert 'os.environ.get("GEMINI_API_KEY")' not in src


# --------------------------------------------------------------------------- #
# An empty reply that IS the answer is not a failed request
# --------------------------------------------------------------------------- #
def _run_empty(scorer):
    async def _fake_send(**k):
        return base.Reply(text="", tool_calls=None, finish_reason="stop")

    samples = [([{"role": "user", "content": "q"}], "g", {}) for _ in range(2)]
    with mock.patch.object(base, "_send_single_request", _fake_send):
        return asyncio.run(base._run_suite_async(
            eval_name="w2b_empty", model_name="m", base_url="http://x/v1", concurrency=1,
            samples=samples, async_eval_fn=scorer, declared_scoring_mode="deterministic"))


def test_empty_answer_marked_by_the_scorer_is_not_counted_as_failed():
    async def scorer(traces):
        traces[0].update(is_correct=True, empty_is_answer=True)     # irrelevance: no call
        traces[1].update(is_correct=False)                          # an ordinary empty reply

    res = _run_empty(scorer)
    assert res["empty_responses"] == 1 and res["failed_requests"] == 1
    assert res["correct_answers"] == 1


def test_flag_without_a_correct_verdict_changes_nothing():
    async def scorer(traces):
        for t in traces:
            t.update(is_correct=False, empty_is_answer=True)

    res = _run_empty(scorer)
    assert res["empty_responses"] == 2 and res["failed_requests"] == 2 and res["status"] == "failed"


def test_all_answered_by_empty_replies_is_a_clean_run():
    async def scorer(traces):
        for t in traces:
            t.update(is_correct=True, empty_is_answer=True)

    res = _run_empty(scorer)
    assert (res["empty_responses"], res["failed_requests"], res["status"]) == (0, 0, "success")


def test_bfcl_scorer_marks_an_empty_irrelevance_reply():
    from gbench.runners.eval_suites import bfcl
    doc = [{"name": "get_weather", "description": "Weather.", "parameters": {
        "type": "dict", "required": ["city"], "properties": {"city": {"type": "string"}}}}]
    traces = [
        {"gold_answer": {"id": "irrelevance_0", "category": "irrelevance", "ground_truth": []},
         "response_text": "", "tool_calls": None, "status": "EMPTY_RESPONSE", "is_correct": False},
        {"gold_answer": {"id": "irrelevance_1", "category": "irrelevance", "ground_truth": []},
         "response_text": "I cannot help with that.", "tool_calls": None, "status": "PENDING_JUDGE",
         "health": "ok", "is_correct": False},
    ]
    asyncio.run(bfcl.bfcl_batch_scorer({"irrelevance_0": doc, "irrelevance_1": doc},
                                       checker=bfcl.single_turn_verdicts)(traces))
    assert traces[0]["is_correct"] is True and traces[0].get("empty_is_answer") is True
    assert traces[1]["is_correct"] is True and "empty_is_answer" not in traces[1]


# --------------------------------------------------------------------------- #
# A protocol round cap counts model turns
# --------------------------------------------------------------------------- #
_CALL = base.Reply(text="", tool_calls=[{"id": "c1", "function": {
    "name": "web_search", "arguments": '{"query":"q"}'}}], finish_reason="tool_calls")


def _loop(protocol=None, **kw):
    sent = []

    async def _fake_send(**k):
        sent.append({"tools": "tools" in (k["extra_payload"] or {}),
                     "last": k["messages"][-1].get("content")})
        return (base.Reply(text="Final answer: 3", tool_calls=None, finish_reason="stop")
                if not sent[-1]["tools"] else _CALL)

    async def _tool(name, args):
        return "{}"

    tools = [{"type": "function", "function": {"name": "web_search", "parameters": {}}}]
    with mock.patch.object(base, "_send_single_request", _fake_send):
        res = base.run_eval_suite(
            eval_name="w2b_loop", model_name="m", base_url="http://x", concurrency=1,
            samples=[([{"role": "user", "content": "q"}], "3", {"tools": tools})],
            eval_fn=lambda r, g: g in (r or ""), tool_executor=_tool, tool_loop=protocol, **kw)
    return res, sent


def test_protocol_cap_allows_exactly_max_rounds_tool_enabled_turns():
    proto = base.ToolLoopProtocol(reference="t", max_rounds=2, forced_final=True,
                                  house_turn_ceiling=False)
    res, sent = _loop(proto)
    assert [s["tools"] for s in sent] == [True, True, False]    # 2 tool turns + forced final
    assert "No further tool calls" in sent[-1]["last"]
    tr = res["sample_traces"][0]
    assert tr["tool_calls_executed"] == 2 and tr["tool_loop_end"] == "forced_final"
    assert res["tool_rounds"]["total_rounds"] == 2


def test_protocol_without_forced_final_stops_after_its_last_turn():
    proto = base.ToolLoopProtocol(reference="t", max_rounds=3, forced_final=False,
                                  house_turn_ceiling=False)
    res, sent = _loop(proto)
    assert [s["tools"] for s in sent] == [True, True, True]
    assert res["sample_traces"][0]["tool_loop_end"] == "max_rounds"


def test_house_loop_keeps_its_round_semantics():
    res, sent = _loop(None, max_tool_rounds=2)
    assert [s["tools"] for s in sent] == [True, True, True, False]


# --------------------------------------------------------------------------- #
# --eval-categories routed per suite
# --------------------------------------------------------------------------- #
def test_category_names_come_from_the_suites():
    from gbench.runners.eval_suites import bfcl, bfcl_v4_agentic
    names = E.eval_category_names()
    assert set(bfcl.SUPPORTED_BFCL_CATEGORIES) <= names["bfcl"] and "live" in names["bfcl"]
    assert "simple" in names["bfcl"]                               # pre-v4 alias
    assert names["bfcl_v4_agentic"] == set(bfcl_v4_agentic.WEB_SEARCH_CATEGORIES
                                           + bfcl_v4_agentic.MEMORY_CATEGORIES)


def test_route_gives_each_suite_only_its_own_names():
    spec = "simple_python, memory_kv,live"
    assert E.route_eval_categories(spec, "bfcl") == "simple_python,memory_kv,live"   # bfcl runs all 22
    assert E.route_eval_categories(spec, "bfcl_v4_agentic") == "memory_kv"
    assert E.route_eval_categories("simple_python", "bfcl_v4_agentic") is None
    assert E.route_eval_categories(spec, "mmlu") is None
    assert E.route_eval_categories(None, "bfcl") is None


class _Stop(Exception):
    pass


def _cli(monkeypatch, *argv):
    """Run gbench.cli.main through validation only; returns the parsed args."""
    from gbench import cli
    captured = {}
    real = cli.create_parser

    def _parser():
        p = real()
        orig = p.parse_args
        p.parse_args = lambda a=None: captured.setdefault("args", orig(a))
        return p

    def _stop(*a, **k):
        raise _Stop

    monkeypatch.setattr(cli, "create_parser", _parser)
    monkeypatch.setattr(cli, "setup_logging", _stop)
    with pytest.raises(_Stop):
        cli.main(["--remote-endpoint", "http://127.0.0.1:1/v1", "--tokenizer", "t",
                  "--max-output-tokens", "1024", *argv])
    return captured["args"]


def _cli_error(capsys, *argv):
    from gbench import cli
    with pytest.raises(SystemExit):
        cli.main(["--remote-endpoint", "http://127.0.0.1:1/v1", "--tokenizer", "t",
                  "--max-output-tokens", "1024", *argv])
    return capsys.readouterr().err


def test_all_with_a_bfcl_category_still_validates(monkeypatch):
    args = _cli(monkeypatch, "--evals", "all", "--eval-categories", "simple_python")
    assert args.eval_categories == "simple_python"


def test_agentic_categories_are_accepted_for_bfcl_v4_agentic(monkeypatch):
    _cli(monkeypatch, "--evals", "bfcl_v4_agentic", "--eval-categories",
         "memory_kv,memory_vector,memory_rec_sum")
    _cli(monkeypatch, "--evals", "bfcl,bfcl_v4_agentic", "--eval-categories", "live,memory_kv")


def test_category_typos_and_misrouted_names_error(capsys):
    assert "Unknown --eval-categories" in _cli_error(capsys, "--evals", "all", "--eval-categories",
                                                     "simple_pyhton")
    assert "Unknown --eval-categories" in _cli_error(capsys, "--evals", "bfcl_v4_agentic",
                                                     "--eval-categories", "simple_python")
    assert "only supported" in _cli_error(capsys, "--evals", "mmlu", "--eval-categories",
                                          "simple_python")


def _runner(**overrides):
    cfg = SimpleNamespace(eval_thinking=False, eval_max_output_tokens=1024,
                          eval_max_soft_tokens=None, eval_n_shot=None, eval_categories=None,
                          eval_limit=None, tokenizer=None, temperature=None, attempt_count=1,
                          sandboxes=None, eval_custom_jsonl=None, suite_timeout=None)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return E.EvalsBenchmarkRunner(cfg)


def _capture_suite(monkeypatch, name):
    from gbench.runners.eval_suites import SUITES
    seen = {}

    def fake(**kw):
        seen.update(kw)
        return {"accuracy": 0.0}

    monkeypatch.setitem(SUITES, name, fake)
    return seen


def test_runner_passes_each_suite_its_own_categories(monkeypatch):
    monkeypatch.setattr(E, "_server_max_model_len", lambda url: None)
    runner = _runner(eval_categories="simple_python,memory_kv")
    for name, want in (("bfcl", "simple_python,memory_kv"), ("bfcl_v4_agentic", "memory_kv"),
                       ("mmlu", None)):
        seen = _capture_suite(monkeypatch, name)
        runner._run_single_eval(name, "m", "http://127.0.0.1:1/v1", 4)
        assert seen["eval_categories"] == want, name
    runner = _runner(eval_categories="simple_python")
    seen = _capture_suite(monkeypatch, "bfcl_v4_agentic")
    runner._run_single_eval("bfcl_v4_agentic", "m", "http://127.0.0.1:1/v1", 4)
    assert seen["eval_categories"] is None, "no agentic name -> the full agentic track"


# --------------------------------------------------------------------------- #
# Cruxeval behind the code-execution gate, not the --sandboxes concurrency cap
# --------------------------------------------------------------------------- #
def test_cruxeval_hits_the_sandbox_gate_before_running(monkeypatch):
    from gbench.runners.eval_suites import sandbox
    monkeypatch.setattr(E, "_server_max_model_len", lambda url: None)
    monkeypatch.setattr(sandbox, "sandbox_skip_reason", lambda: "bubblewrap is not installed")
    seen = _capture_suite(monkeypatch, "cruxeval")
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*cruxeval"):
        _runner()._run_single_eval("cruxeval", "m", "http://127.0.0.1:1/v1", 4)
    assert not seen


def test_cruxeval_concurrency_is_not_bounded_by_sandboxes(monkeypatch):
    from gbench.runners.eval_suites import sandbox
    monkeypatch.setattr(E, "_server_max_model_len", lambda url: None)
    monkeypatch.setattr(sandbox, "sandbox_skip_reason", lambda: None)
    seen = _capture_suite(monkeypatch, "cruxeval")
    _runner(sandboxes=2)._run_single_eval("cruxeval", "m", "http://127.0.0.1:1/v1", 16)
    assert seen["concurrency"] == 16


# --------------------------------------------------------------------------- #
# Renames
# --------------------------------------------------------------------------- #
RENAMES = {"semantic_keypoint": "screenspot_pointing", "lmsys_noncoding_hard": "wildbench",
           "bfcl_v3_live": "bfcl_live", "loft_x_arxiv": "loft_scifact"}


def test_new_names_are_registered_and_old_names_are_not_listed():
    from gbench.runners.eval_suites import SUITES, RENAMED_SUITES
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC
    assert RENAMES.items() <= RENAMED_SUITES.items()
    builtin = {s for _, su in E.BUILTIN_PILLARS for s in su}
    for old, new in RENAMES.items():
        assert new in SUITES and new in builtin and new in CANONICAL_SYNC
        assert old not in SUITES and old not in builtin and old not in CANONICAL_SYNC
        assert (REPO / "docs" / "evals" / f"{new}.md").exists()
        assert not (REPO / "docs" / "evals" / f"{old}.md").exists()
        assert not (SUITE_DIR / f"{old}.py").exists()


def test_old_names_resolve_with_one_warning(caplog):
    from gbench.runners.eval_suites import resolve_suite_name, _RENAME_WARNED
    _RENAME_WARNED.discard("lmsys_noncoding_hard")
    with caplog.at_level("WARNING"):
        assert resolve_suite_name("lmsys_noncoding_hard") == "wildbench"
        assert resolve_suite_name("lmsys_noncoding_hard") == "wildbench"
    hits = [r for r in caplog.records if "renamed" in r.getMessage()]
    assert len(hits) == 1 and "wildbench" in hits[0].getMessage()
    assert resolve_suite_name("mmlu") == "mmlu"


def test_cli_maps_old_names_and_accepts_them_in_allow_lists(monkeypatch):
    args = _cli(monkeypatch, "--evals", "bfcl_v3_live,bfcl_live", "semantic_keypoint")
    assert args.evals == ["bfcl_live", "screenspot_pointing"]


def test_runner_resolves_old_names(monkeypatch, tmp_path):
    ran = []
    monkeypatch.setattr(E, "_resolve_remote_model_id", lambda url, fallback: "m")
    monkeypatch.setattr(E, "_server_max_model_len", lambda url: None)
    monkeypatch.setattr(E.EvalsBenchmarkRunner, "_run_single_eval",
                        lambda self, eval_name, **k: ran.append(eval_name) or {"accuracy": 0.0})
    saved = []
    cfg = SimpleNamespace(evals=["loft_x_arxiv", "screenspot_pointing,semantic_keypoint"],
                          eval_custom_jsonl=None, eval_plugins_dir=None, batch_sizes=[1],
                          remote_endpoint="http://127.0.0.1:1/v1", tokenizer="t",
                          eval_max_output_tokens=1024, skip_existing=False,
                          log_manager=SimpleNamespace(results_dir=tmp_path, base_dir=tmp_path,
                                                      save_result=lambda p, r: saved.append(r)))
    model = SimpleNamespace(name="m", short_name="m", hf_model_id="org/m")
    results = E.EvalsBenchmarkRunner(cfg).run(model, SimpleNamespace(value="remote-endpoint"))
    assert ran == ["loft_scifact", "screenspot_pointing"]
    assert [r["eval_name"] for r in results] == ran
    assert results[0]["canonical_sync"]["status"] == "reconciled"


def test_old_temperature_variable_is_a_deprecated_alias(monkeypatch):
    monkeypatch.delenv("GBENCH_WILDBENCH_TEMPERATURE", raising=False)
    monkeypatch.setenv("GBENCH_LMSYS_NONCODING_HARD_TEMPERATURE", "0.3")
    assert base.resolve_temperature("wildbench") == (
        0.3, "env:GBENCH_LMSYS_NONCODING_HARD_TEMPERATURE")
    monkeypatch.setenv("GBENCH_WILDBENCH_TEMPERATURE", "0.7")
    assert base.resolve_temperature("wildbench") == (0.7, "env:GBENCH_WILDBENCH_TEMPERATURE")
    assert base.legacy_temperature_env_vars("mmlu") == ()


def test_limit_seeds_keep_the_old_strings():
    """--eval-limit subsets must stay the same rows across the rename."""
    for module, seed in (("screenspot_pointing", 'seed="semantic_keypoint"'),
                         ("wildbench", 'seed="lmsys_noncoding_hard"'),
                         ("loft_scifact", 'seed="loft_x_arxiv"'),
                         ("bfcl_live", 'limit_seed="bfcl_v3_live"')):
        assert seed in (SUITE_DIR / f"{module}.py").read_text(encoding="utf-8"), module


def test_renamed_modules_carry_their_new_names():
    from gbench.runners.eval_suites import bfcl_live, loft_scifact, screenspot_pointing, wildbench
    assert wildbench.EVAL_NAME == "wildbench"
    assert bfcl_live.EVAL_NAME == "bfcl_live"
    assert loft_scifact.EVAL_NAME == "loft_scifact"
    assert screenspot_pointing.EVAL_NAME == "screenspot_pointing"
    assert wildbench.DOCS_URL == "docs/evals/wildbench.md"


def test_catalog_docs_use_the_new_suite_ids():
    for rel in (*_README_AND_SPLIT_DOCS, "docs/RESOURCES.md",
                "docs/THIRD_PARTY_DATASETS.md", "docs/evals_roadmap.md"):
        text = (REPO / rel).read_text(encoding="utf-8")
        for old, new in RENAMES.items():
            assert f"`{old}`" not in text.replace(f"renamed from `{old}`", ""), (rel, old)
    # the suite catalog moved verbatim from the README into docs/eval-suites.md
    readme = (REPO / "docs" / "eval-suites.md").read_text(encoding="utf-8")
    assert all(f"`{new}`" in readme for new in RENAMES.values())


# --------------------------------------------------------------------------- #
# Canonical_sync entries
# --------------------------------------------------------------------------- #
def test_entries_use_only_the_standard_keys_and_charxiv_has_no_fallback_claim():
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC
    for name, entry in CANONICAL_SYNC.items():
        assert set(entry) <= {"synced", "upstream", "revision", "dataset", "checksum",
                              "method"}, name
    method = CANONICAL_SYNC["charxiv"]["method"]
    assert "no relaxed-match fallback" in method and "hard-errors" in method


def test_swe_family_entries_record_their_pins():
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as C
    from gbench.runners.eval_suites import multi_swe_bench, ojbench, scicode
    assert C["multi_swe_bench"]["revision"].startswith("24f493f8a103e72312ded4f6b9c89f081d69cb09, ")
    assert multi_swe_bench._REVISION in C["multi_swe_bench"]["dataset"]
    assert C["swe_bench_pro"]["revision"] == "66f92766bba642462d4bbe5479e83f91f9211862"
    assert "mini-swe-agent 2.4.2" in C["copilot_bench_swe"]["revision"]
    assert C["scicode"]["revision"] == scicode._DATASET_REVISION
    assert C["ojbench"]["revision"] in C["ojbench"]["dataset"]
    assert "5e94480b" in (REPO / "docker" / "ojbench.Dockerfile").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# The evals extra declares the suites' runtime dependencies
# --------------------------------------------------------------------------- #
def test_evals_extra_declares_the_suite_dependencies():
    with open(REPO / "pyproject.toml", "rb") as f:
        evals = tomllib.load(f)["project"]["optional-dependencies"]["evals"]
    for req in ("tree-sitter==0.21.3", "tree-sitter-python==0.21.0", "scikit-learn", "openpyxl",
                "pymupdf", "xlrd", "python-pptx", "langdetect", "nltk",
                "mini-swe-agent>=1.15,<2"):
        assert req in evals, req
    assert "mini-swe-agent<2" not in evals
