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

"""BFCL output masking, Gemini key checks and request bounds; ACEBench's context window, row
scoring and user-simulator accounting.

  * The harness output is masked before it is cut, so a credential split by a read, or lying on
    the cut of a line longer than 64 KB, is masked whole in the captured and streamed output.
  * A Gemini key check that gets no answer (unreachable, 5xx, 429) never stops a run; only a key
    the Gemini API rejects does. During generation only a search no grounding model served
    stops it.
  * A stopped `bfcl evaluate` after a stopped `bfcl generate` keeps both records; every harness
    request's timeout is sized for the context window (262,144 tokens when none is reported).
  * ACEBench applies LLMInfer's 8192-token window on no-think runs, scores the single-turn rows
    off the event loop, counts replies cut off by a budget, and counts user-simulator replies
    that stopped at APIUSER's max_tokens.
  * The other function-calling suites follow the token-limit policy: NESTFUL sends its 1000 and
    T-Eval its 512 without --thinking; Seal-Tools, API-Bank, NexusRaven and Gorilla send no
    max_tokens, as their harnesses send none.

Everything runs offline; the launcher tests start local Python children."""

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import acebench as AB
from gbench.runners.eval_suites import base, bfcl
from gbench.runners.eval_suites import bfcl_v4_agentic as AG

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "evals"
KEY = "gemKEY0123456789abcdef0123456789abcdef"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("GBENCH_BFCL_TIMEOUT_S", raising=False)
    monkeypatch.delenv("BFCL_TIMEOUT_S", raising=False)
    monkeypatch.delenv(AG.REQUEST_TIMEOUT_ENV, raising=False)


# --------------------------------------------------------------------------- #
# the launcher masks before it cuts
# --------------------------------------------------------------------------- #

def _pieces_program(pieces, pause=0.3):
    """A child that writes `pieces` to stdout one by one, flushed, `pause` seconds apart."""
    return ("import sys, time\n"
            f"pieces = {pieces!r}\n"
            "for i, p in enumerate(pieces):\n"
            f"    if i: time.sleep({pause!r})\n"
            "    sys.stdout.write(p); sys.stdout.flush()\n")


def _launch(pieces, env, stream=False):
    return AG._launch([sys.executable, "-c", _pieces_program(pieces)], env, 60,
                      "gbench-test-unused", stream=stream)


def test_a_credential_on_the_cut_of_a_long_line_is_masked_whole():
    # The case the earlier hold-back leaked: a whole key in the text read, on the place a piece of
    # a line longer than 64 KB is cut (it cut the raw text first and masked each piece).
    env = dict(os.environ, GEMINI_API_KEY=KEY)
    head = "x" * 65536
    for before, after in ((100, 220), (216, 224), (5, 300), (4000, 200)):
        line = "x" * before + KEY + "z" * after
        rc, out = _launch([head, line, "\n"], env)
        assert rc == 0 and KEY not in out
        assert out.replace(base.REDACTED, KEY) == head + line + "\n"


def test_a_credential_split_by_a_read_is_masked_whole():
    # The key's beginning arrives in one read, the rest in the next: it waits in the held tail.
    env = dict(os.environ, GEMINI_API_KEY=KEY)
    for split in (1, 20, len(KEY) - 1):
        pieces = ["x" * 65536, "x" * 100 + KEY[:split], KEY[split:] + "z" * 10 + "\n"]
        rc, out = _launch(pieces, env)
        assert rc == 0 and KEY not in out and out.count(base.REDACTED) == 1
        # and on an ordinary short line
        rc, out = _launch(["start " + KEY[:split], KEY[split:] + " end\n"], env)
        assert out == f"start {base.REDACTED} end\n"


def test_every_form_of_a_credential_is_masked_across_reads():
    key = 'Sk/ab+cd=="ef0123456789'
    env = dict(os.environ, GEMINI_API_KEY=key)
    for form in base.secret_forms(key):
        for split in (1, len(form) // 2, len(form) - 1):
            rc, out = _launch(["x" * 65536, "y" * 300 + form[:split], form[split:] + "\n"], env)
            assert rc == 0 and not any(f in out for f in base.secret_forms(key)), form
            assert out.count(base.REDACTED) == 1


def test_a_multi_line_credential_is_masked_across_a_line_break():
    # Neither half is a credential on its own (under 8 characters), only the whole value.
    value = "abc1234\nxyz5678"
    env = dict(os.environ, MY_SERVICE_SECRET=value)
    rc, out = _launch(["hello abc1234\nxyz", "5678 bye\n"], env)
    assert rc == 0 and "abc1234" not in out and "xyz5678" not in out
    assert out == f"hello {base.REDACTED} bye\n"


def test_the_streamed_output_is_masked_too(capsys):
    env = dict(os.environ, GEMINI_API_KEY=KEY)
    rc, out = _launch(["x" * 65536, "x" * 100 + KEY + "z" * 220, "\n"], env, stream=True)
    err = capsys.readouterr().err
    assert KEY not in err and KEY not in out and base.REDACTED in err
    assert err == out


# --------------------------------------------------------------------------- #
# Gemini key checks: only the Gemini API's rejection stops a run
# --------------------------------------------------------------------------- #

@pytest.fixture
def account(monkeypatch):
    """A local Gemini OpenAI-compatible endpoint (GEMINI_OPENAI_BASE_URL, which
    base.gemini_key_live_valid asks for /models) answering `state["reply"]` = (status, body)."""
    state = {"reply": (200, {"data": []}), "hits": 0, "auth": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["hits"] += 1
            state["auth"].append(self.headers.get("Authorization"))
            status, body = state["reply"]
            data = json.dumps(body).encode() if not isinstance(body, bytes) else body
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}/v1")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    yield state
    srv.shutdown()


@pytest.mark.parametrize("reply,rejected,needle", [
    ((200, {"data": []}), False, ""),
    ((429, {"error": "Too many requests"}), False, ""),
    ((503, {}), False, ""),
    ((400, {"error": {"message": "malformed"}}), False, ""),
    ((401, {"error": "unauthenticated"}), True, "HTTP 401"),
    ((403, {}), True, "HTTP 403"),
    ((400, {"error": {"message": "API key not valid. Please pass a valid API key."}}), True,
     "HTTP 400"),
])
def test_the_key_check_tells_a_rejection_from_no_answer(account, reply, rejected, needle):
    account["reply"] = reply
    backend, why = AG.search_backend()
    # a key the API accepts (200) is followed by one model check per cascade model (base's gate)
    checks = len(base.gemini_cascade_models()) if reply[0] == 200 else 0
    assert account["hits"] == 1 + checks and account["auth"] == [f"Bearer {KEY}"] * (1 + checks)
    if rejected:
        assert backend is None and needle in why and KEY not in why
        assert "rejected" in why and "Gemini Google Search grounding" in why
    else:
        assert (backend, why) == (AG.SEARCH_BACKEND, "")


def test_an_unreachable_endpoint_does_not_block_the_key(monkeypatch):
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", f"http://127.0.0.1:{base.free_port()}/v1")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    assert AG.search_backend() == (AG.SEARCH_BACKEND, "")


def _log_search(log, backend="gemini:gemini-3.7-flash", results=3, error=None, n=1):
    with log.open("a") as f:
        for _ in range(n):
            f.write(json.dumps({"backend": backend, "results": results, "error": error}) + "\n")


def test_the_watch_does_not_stop_generate_while_every_search_is_served(tmp_path):
    log = tmp_path / AG.SEARCH_LOG_NAME
    log.write_text("")
    watch = AG._search_watch(log)
    assert watch() is None                              # no search yet
    _log_search(log, n=3)
    _log_search(log, backend="gemini:gemini-3.6-flash", results=0)   # answered, no result
    assert watch() is None


def test_the_watch_stops_generate_on_a_failed_search(tmp_path):
    log = tmp_path / AG.SEARCH_LOG_NAME
    log.write_text("")
    watch = AG._search_watch(log)
    _log_search(log)
    _log_search(log, backend="all-grounding-models-failed", results=0,
                error="search backend failed: HTTP Error 429: Too Many Requests")
    problem = watch()
    assert isinstance(problem, AG.SearchUnavailable) and "INFRA REQUIRED" in str(problem)
    assert "for 1 of the 2 web searches" in str(problem) and "was stopped" in str(problem)


# --------------------------------------------------------------------------- #
# the agentic part: stops, timeouts
# --------------------------------------------------------------------------- #

@pytest.fixture
def agentic(monkeypatch, tmp_path):
    seen = {"cli": [], "stop": set(), "env": None, "counts_calls": 0, "log": []}
    root = tmp_path / "bfcl"
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(root))
    monkeypatch.setattr(AG, "check_bfcl_prerequisites",
                        lambda url: (True, "", list(AG.AGENTIC_CATEGORIES)))
    monkeypatch.setattr(AG, "search_backend", lambda: (AG.SEARCH_BACKEND, ""))

    def cli(args, env, timeout, model="", backend=None):
        seen["cli"].append((list(args), timeout))
        if args[0] == "generate":
            seen["env"] = dict(env)
            log = env.get(AG.SEARCH_LOG_ENV)
            if log:
                with open(log, "a", encoding="utf-8") as f:
                    for row in seen["log"]:
                        f.write(json.dumps(row) + "\n")
            path = root / "result/m/agentic/BFCL_v4_web_search_base_result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"id": "web_search_base_0", "result": "x"}) + "\n")
        if args[0] in seen["stop"]:
            raise subprocess.TimeoutExpired(cmd=f"bfcl {args[0]}", timeout=timeout)
        return 0, ""
    monkeypatch.setattr(AG, "_run_cli", cli)

    def run_in_image(eval_name, module, func, kwargs, **kw):
        assert func == "scored_entry_counts"
        seen["counts_calls"] += 1
        seen["counts_before_generate"] = not any(c[0][0] == "generate" for c in seen["cli"])
        return {c: 5 for c in kwargs["categories"]}
    monkeypatch.setattr(AG, "run_in_image", run_in_image)
    monkeypatch.setattr(AG, "_clear_stale_scores", lambda *a: 0)
    monkeypatch.setattr(AG, "_parse_scores", lambda r, m, cats: {
        c: {"correct": 1, "total": 2, "accuracy": 50.0} for c in cats})
    monkeypatch.setattr(AG, "_parse_score_traces", lambda *a: [])
    monkeypatch.setattr(AG, "_score_error_types", lambda *a: {})
    return seen


def test_a_stopped_evaluate_after_a_stopped_generate_keeps_both_records(agentic, monkeypatch):
    monkeypatch.setenv("GBENCH_BFCL_TIMEOUT_S", "600")
    agentic["stop"] = {"generate", "evaluate"}
    with pytest.raises(AG.HarnessTimeout) as e:
        AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    assert e.value.part == "`bfcl evaluate`"
    [gen] = e.value.earlier
    assert gen["part"] == "`bfcl generate`" and gen["timeout_s"] == 600.0
    assert gen["entries_planned"] == gen["entries_unscored"] == 25
    assert "`bfcl evaluate` did not finish" in str(e.value)
    assert "before it, bfcl_v4_agentic: `bfcl generate` did not finish" in str(e.value)
    # the evaluate record is its own; the generate stop has its own record
    assert "generate" not in e.value.record()["error"]


def test_a_full_run_with_a_wall_clock_reads_its_counts_before_generating(agentic, monkeypatch):
    monkeypatch.setenv("GBENCH_BFCL_TIMEOUT_S", "600")
    agentic["stop"] = {"generate"}
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    assert agentic["counts_calls"] == 1 and agentic["counts_before_generate"] is True
    assert res["harness_errors"][0]["entries_planned"] == 25


def test_a_run_without_a_wall_clock_runs_no_extra_job(agentic):
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    assert agentic["counts_calls"] == 0 and res["harness_errors"] == []
    assert all(t is None for _a, t in agentic["cli"])


def test_a_stopped_generate_whose_searches_were_served_is_a_plain_timeout(agentic, monkeypatch):
    monkeypatch.setenv("GBENCH_BFCL_TIMEOUT_S", "600")
    agentic["stop"] = {"generate"}
    agentic["log"] = [{"backend": "gemini:gemini-3.7-flash", "results": 4, "error": None}] * 2
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    assert "INFRA REQUIRED" not in json.dumps(res["harness_errors"])
    assert [e["part"] for e in res["harness_errors"]] == ["`bfcl generate`"]
    assert "settings_infra_required" not in res and res["status"] == "completed_with_errors"


def test_a_stopped_generate_with_a_failed_search_names_it(agentic, monkeypatch):
    monkeypatch.setenv("GBENCH_BFCL_TIMEOUT_S", "600")
    agentic["stop"] = {"generate"}
    agentic["log"] = [{"backend": "gemini:gemini-3.7-flash", "results": 4, "error": None},
                      {"backend": "all-grounding-models-failed", "results": 0,
                       "error": "search backend failed: timed out"}]
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    web = res["settings"]["web_search"]
    assert re.search(r"GBENCH_BFCL_TIMEOUT_S=600s.*1 of the 2 web searches.*timed out",
                     web["infra_required"])
    # the memory resume ran past the wall clock too (this stub stops every generate): recorded
    # like a stopped generate, its unfinished entries counted and left out of the score (no
    # entry finished here, so the memory categories are unscored and the score withheld)
    [err] = res["harness_errors"]
    assert err["part"] == "`bfcl generate` (the memory categories, resumed)"
    assert err["entries_planned"] == 15 and res["category_accuracy"] == {}
    assert res["categories_unscored"] == list(AG.MEMORY_CATEGORIES)
    assert res["request_harness_errors"] == {"unfinished": 15}
    assert res["status"] == "failed" and res["accuracy"] is None


def test_the_agentic_requests_get_a_timeout_sized_for_the_window(agentic, monkeypatch):
    monkeypatch.setitem(base._RUN_KNOBS, "max_model_len", 32768)
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    want = base.request_timeout_s(32768)
    assert agentic["env"][AG.REQUEST_TIMEOUT_ENV] == str(want)
    assert res["bfcl_report"]["request_timeout_s"] == want
    assert AG.REQUEST_TIMEOUT_ENV in AG._CHILD_ENV_NAMES
    monkeypatch.setitem(base._RUN_KNOBS, "max_model_len", None)
    assert AG.agentic_request_timeout() == base.request_timeout_s(base.UNCAPPED_TIMEOUT_TOKENS)


def test_the_handler_sends_the_timeout_with_every_request(monkeypatch):
    cls = AG._patched_openai_handler()
    handler = cls.__new__(cls)
    sent = {}

    def create(**kw):
        sent.update(kw)
        return types.SimpleNamespace(choices=[])
    handler.client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    monkeypatch.setenv(AG.REQUEST_TIMEOUT_ENV, "4321")
    handler.generate_with_backoff(model="m", messages=[])
    assert sent["timeout"] == 4321.0
    sent.clear()
    monkeypatch.delenv(AG.REQUEST_TIMEOUT_ENV)
    handler.generate_with_backoff(model="m", messages=[])
    assert "timeout" not in sent                        # the client's own default


# --------------------------------------------------------------------------- #
# the bfcl suite
# --------------------------------------------------------------------------- #

def test_the_multi_turn_timeout_falls_back_to_an_uncapped_window(monkeypatch):
    monkeypatch.setitem(base._RUN_KNOBS, "max_model_len", None)
    assert bfcl.multi_turn_request_timeout() == base.request_timeout_s(base.UNCAPPED_TIMEOUT_TOKENS)
    monkeypatch.setitem(base._RUN_KNOBS, "max_model_len", 65536)
    assert bfcl.multi_turn_request_timeout() == base.request_timeout_s(65536)
    seen = {}

    def run_in_image(eval_name, module, func, kwargs, **kw):
        seen.update(kwargs)
        return {"traces": [], "guard_refusals": []}
    monkeypatch.setattr(AG, "run_in_image", run_in_image)
    monkeypatch.setitem(base._RUN_KNOBS, "max_model_len", None)
    bfcl.run_multi_turn_in_image("m", "http://x/v1", 1, [], temperature=0.001, thinking=False)
    assert seen["request_timeout"] == base.request_timeout_s(base.UNCAPPED_TIMEOUT_TOKENS) > 2224


def _agentic_only(monkeypatch):
    monkeypatch.setattr(AG, "check_bfcl_prerequisites",
                        lambda url: (True, "", list(AG.AGENTIC_CATEGORIES)))
    monkeypatch.setattr(AG, "search_backend", lambda: (AG.SEARCH_BACKEND, ""))


def test_the_bfcl_suite_keeps_the_generate_record_of_a_double_stop(monkeypatch):
    _agentic_only(monkeypatch)
    gen = AG.HarnessTimeout("bfcl_v4_agentic", "`bfcl generate`", 60).record(entries_planned=5,
                                                                           entries_unscored=5)

    def run(*a, **k):
        raise AG.HarnessTimeout("bfcl_v4_agentic", "`bfcl evaluate`", 60, earlier=[gen])
    monkeypatch.setattr(AG, "run_bfcl_v4_agentic", run)
    res = bfcl.run_bfcl("m", "http://x/v1", 1, eval_categories="memory_kv")
    assert [e["part"] for e in res["harness_errors"]] == ["`bfcl generate`", "`bfcl evaluate`"]
    assert res["harness_errors"][1]["categories_unscored"] == ["memory_kv"]
    reason = res["leaderboard_comparable_reason"]
    assert "`bfcl generate` did not finish" in reason and "`bfcl evaluate` did not finish" in reason


def test_another_lost_prerequisite_in_the_bfcl_suite_logs_the_finished_parts(monkeypatch,
                                                                             caplog):
    # (a failed web search leaves only the Web Search setting infra_required instead)
    _agentic_only(monkeypatch)

    def run(*a, **k):
        raise AG.infra_required("bfcl_v4_agentic", "the harness image is gone", AG.DOCS_URL)
    monkeypatch.setattr(AG, "run_bfcl_v4_agentic", run)
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        bfcl.run_bfcl("m", "http://x/v1", 1, eval_categories="memory_kv")
    assert any("the parts already run scored" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# ACEBench: LLMInfer's 8192-token window
# --------------------------------------------------------------------------- #

def _msgs(n_chars):
    return [{"role": "system", "content": "s" * n_chars}, {"role": "user", "content": "u"}]


def test_a_short_prompt_is_not_counted(monkeypatch):
    monkeypatch.setattr(base, "_prompt_token_bounds",
                        lambda *a, **k: pytest.fail("a short prompt needs no count"))
    assert AB.upstream_window_budget(_msgs(1000), 1024) == (1024, None, "short")


@pytest.mark.parametrize("tokens,mode,want", [
    (3000, "server", (1024, 3000, "server")),
    (7500, "server", (693, 7500, "server")),
    (8191, "tokenizer", (1024, None, "uncounted")),
    (7500, "estimate", (1024, None, "uncounted")),
])
def test_the_window_leaves_the_room_after_the_prompt(monkeypatch, tokens, mode, want):
    monkeypatch.setattr(base, "_prompt_token_bounds", lambda *a, **k: (tokens, tokens, mode))
    assert AB.upstream_window_budget(_msgs(8000), 1024) == want


def test_a_prompt_longer_than_the_window_is_not_sent(monkeypatch):
    monkeypatch.setattr(base, "_prompt_token_bounds", lambda *a, **k: (8192, 8192, "server"))
    assert AB.upstream_window_budget(_msgs(9000), 1024) == (1, 8192, "server")
    monkeypatch.setattr(base, "_prompt_token_bounds", lambda *a, **k: (8193, 8193, "server"))
    assert AB.upstream_window_budget(_msgs(9000), 1024) == (0, 8193, "server")


def test_the_single_turn_rows_get_the_room_the_window_leaves(monkeypatch):
    sizes = {"a": 100, "b": 7600, "c": 9000}
    monkeypatch.setattr(base, "_prompt_token_bounds", lambda m, *a, **k: (
        sizes[m[1]["content"]],) * 2 + ("server",))
    samples = [([{"role": "system", "content": "s" * 9000}, {"role": "user", "content": k}],
                f"gold-{k}", {"category": "normal_atom_bool"}) for k in sizes]
    out, over, verdicts = AB._single_turn_window(samples)
    assert [m for _s, _g, m in out] == [
        {"category": "normal_atom_bool"},
        {"category": "normal_atom_bool", "max_tokens": 593}]
    assert [(row[1], tokens) for row, tokens in over] == [("gold-c", 9000)]
    assert verdicts["gold-c"] == (0, 9000, "server")


def _agent_shared(**extra):
    return {"session": None, "api_url": "u", "model_name": "m", "payload": {}, "semaphore": None,
            "budget": 1024, "official_cap": 1024, "temperature": 0.0, "thinking": False,
            "ctk": None, "upstream_window": True, **extra}


def test_an_agent_request_gets_the_room_and_a_reply_there_is_at_the_official_cap(monkeypatch):
    monkeypatch.setattr(base, "_prompt_token_bounds", lambda *a, **k: (7800, 7800, "server"))
    sent = []

    async def clamp(convo, budget, tools, ctk):
        return budget, None

    async def send(**k):
        sent.append(k["max_output_tokens"])
        return base.Reply("[f()]", None, "length", completion_tokens=393)
    monkeypatch.setattr(AB, "clamp_to_context_async", clamp)
    with mock.patch.object(base, "_send_single_request", send):
        model = AB._AgentModel(_agent_shared())
        asyncio.run(model.inference("s" * 9000, "u"))
    assert sent == [393] and (model.at_official_cap, model.truncated) == (1, 0)
    assert model.window_limited == 1 and model.window_uncounted == 0


def test_an_agent_prompt_over_the_window_gets_an_empty_reply_and_goes_on(monkeypatch):
    monkeypatch.setattr(base, "_prompt_token_bounds", lambda *a, **k: (9000, 9000, "server"))

    real = AB._AgentModel.inference

    async def inference(self, system_prompt, user_prompt):
        return await real(self, "s" * 9000, user_prompt)

    async def send(**k):
        raise AssertionError("a prompt longer than the window is not sent")
    item = {"language": "en", "category": "agent_multi_step", "key": "agent_multi_step",
            "row": {"id": "agent_multi_step_0", "function": [], "initial_config": {},
                    "involved_classes": [], "question": "q"},
            "answer": {"ground_truth": [], "mile_stone": []}}
    monkeypatch.setattr(AB, "_Environment", lambda *a, **k: types.SimpleNamespace(refusals=[]))
    monkeypatch.setattr(AB._AgentModel, "inference", inference)
    with mock.patch.object(base, "_send_single_request", send):
        trace = asyncio.run(AB._run_conversation(_agent_shared(), item))
    # upstream's run goes on with the "" (the execution role answers it with its fixed text)
    assert trace["status"] == "OK" and trace["error"] is None
    assert trace["model_requests"] == 0 and trace["window_over_steps"] == AB.MAX_DIALOG_TURNS // 2
    assert trace["agent_dialogue"][1] == {"sender": "agent", "recipient": "user", "message": ""}
    assert trace["agent_dialogue"][2]["message"] == AB._STEP_NOT_A_CALL


# --------------------------------------------------------------------------- #
# ACEBench: row scoring off the event loop; cut replies
# --------------------------------------------------------------------------- #

def test_the_rows_are_scored_in_worker_threads(monkeypatch):
    main = threading.get_ident()
    calls = []

    def eval_fn(text, gold):
        calls.append((threading.get_ident(), text, gold))
        return gold == "right"
    traces = [{"gold_answer": "right", "response_text": "<thought>x</thought>[A()]",
               "status": "PENDING_JUDGE", "health": "ok"},
              {"gold_answer": "wrong", "response_text": "[B()]", "status": "PENDING_JUDGE",
               "health": "truncated"},
              {"gold_answer": "right", "response_text": "", "status": "EMPTY_RESPONSE",
               "is_correct": False}]
    asyncio.run(AB._row_scorer(eval_fn)(traces))
    assert all(tid != main for tid, _t, _g in calls)
    assert sorted((t, g) for _i, t, g in calls) == [("[A()]", "right"), ("[B()]", "wrong")]
    assert [(t["status"], t["is_correct"]) for t in traces] == [
        ("OK", True), ("TRUNCATED", False), ("EMPTY_RESPONSE", False)]


def test_cut_replies_are_counted_by_mode():
    traces = [{"finish_reason": "length", "max_tokens_sent": 1024, "gold_answer": "a"},
              {"finish_reason": "length", "max_tokens_sent": 592, "gold_answer": "b"},
              {"finish_reason": "length", "max_tokens_sent": 300, "gold_answer": "c"},
              {"finish_reason": "stop", "max_tokens_sent": 1024, "gold_answer": "d"},
              {"finish_reason": "length", "max_tokens_sent": 1, "gold_answer": "e",
               "over_upstream_window": True}]
    # no-think: stops at the official cap (1024, or the window's room) are the protocol's outcome
    assert AB._single_turn_cut(traces, {"b": 592}, thinking=False) == 1
    # --thinking: every stop at the budget is a cut
    assert AB._single_turn_cut(traces, {}, thinking=True) == 3


def _gold(row_id, sub="normal_atom_bool"):
    return json.dumps({"id": row_id, "sub": sub, "language": "en", "ground_truth": {"Api": {}},
                       "functions": [], "question": ""})


def _run(monkeypatch, thinking, reply, rows=None, agent_traces=None, bounds=None):
    base.set_run_knobs(max_output_tokens=4096)
    rows = rows or [("normal_atom_bool_1", "u")]
    monkeypatch.setattr(AB, "_load_acebench_samples", lambda **k: [
        ([{"role": "system", "content": "s" * 8000}, {"role": "user", "content": u}], _gold(r),
         {"category": "normal_atom_bool"}) for r, u in rows] if k.get("language") == "en" else [])
    monkeypatch.setattr(AB, "_load_agent_items", lambda language="en": [])
    monkeypatch.setattr(AB, "check_binop_evaluator", lambda: None)
    monkeypatch.setattr(base, "_prompt_token_bounds",
                        bounds or (lambda *a, **k: (3000, 3000, "server")))
    if agent_traces is not None:
        monkeypatch.setattr(AB, "_load_agent_items", lambda language="en": [
            {"category": "agent_multi_turn", "key": "agent_multi_turn", "language": language,
             "row": {"id": "agent_multi_turn_0"}, "answer": {}}] if language == "en" else [])
        monkeypatch.setattr(AB, "check_user_simulator", lambda: None)

        async def run_agents(*a, **k):
            return agent_traces
        monkeypatch.setattr(AB, "_run_agent_items", run_agents)
    sent = []

    async def send(**k):
        sent.append(k["max_output_tokens"])
        return reply
    with mock.patch.object(base, "_send_single_request", send):
        res = AB.run_acebench("m", "http://127.0.0.1:9/v1", 1, enable_thinking=thinking)
    return res, sent


def test_a_no_think_row_near_the_window_is_sent_the_room(monkeypatch):
    res, sent = _run(monkeypatch, False, base.Reply("[Api()]", None, "length",
                                                    completion_tokens=693),
                     bounds=lambda *a, **k: (7500, 7500, "server"))
    assert sent == [693]
    assert res["replies_at_official_max_tokens"] == 1 and res["status"] == "success"
    win = res["acebench_report"]["upstream_context_window"]
    assert win["applied"] is True and win["single_turn_rows_limited"] == 1
    assert res["acebench_report"]["single_turn_replies_cut_off"] == 0
    assert res["leaderboard_comparable"] is True, res["leaderboard_comparable_reason"]


def test_a_no_think_row_over_the_window_is_not_sent_and_scored_on_an_empty_reply(monkeypatch):
    res, sent = _run(monkeypatch, False, base.Reply("[Api()]", None, "length",
                                                    completion_tokens=1),
                     bounds=lambda *a, **k: (8500, 8500, "server"))
    assert sent == []
    [t] = res["sample_traces"]
    assert (t["status"], t["response_text"], t["request_sent"]) == ("OK", "", False)
    assert t["is_correct"] is False and t["over_upstream_window"] is True
    assert res["acebench_report"]["upstream_context_window"]["single_turn_rows_over_window"] == 1
    assert res["leaderboard_comparable"] is True, res["leaderboard_comparable_reason"]


def test_a_thinking_run_applies_no_window_and_counts_cut_replies(monkeypatch):
    res, sent = _run(monkeypatch, True, base.Reply("", None, "length", completion_tokens=4096,
                                                   reasoning="r" * 50),
                     bounds=lambda *a, **k: pytest.fail("no window with --thinking"))
    assert sent == [4096]
    win = res["acebench_report"]["upstream_context_window"]
    assert win["applied"] is False and "not applied to reasoning" in win["note"]
    assert res["acebench_report"]["single_turn_replies_cut_off"] == 1
    assert res["leaderboard_comparable"] is False
    assert "1 single-turn and 0 agent replies were cut off by the run's budget, max_tokens 4096" \
        in res["leaderboard_comparable_reason"]


def test_an_uncounted_prompt_near_the_window_makes_the_run_not_comparable(monkeypatch):
    res, sent = _run(monkeypatch, False, base.Reply("[Api()]", None, "stop", completion_tokens=3),
                     bounds=lambda *a, **k: (7000, 9000, "estimate"))
    assert sent == [1024]
    assert res["acebench_report"]["upstream_context_window"]["single_turn_rows_uncounted"] == 1
    assert "could not be counted exactly" in res["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# ACEBench: user-simulator replies at APIUSER's max_tokens
# --------------------------------------------------------------------------- #

def _resp(text, finish):
    return types.SimpleNamespace(text=text, candidates=[types.SimpleNamespace(
        finish_reason=types.SimpleNamespace(name=finish))])


def _fake_cascade(monkeypatch, replies):
    calls = []

    async def generate_content(model, contents, config):
        calls.append((model, config))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
    client = types.SimpleNamespace(aio=types.SimpleNamespace(
        models=types.SimpleNamespace(generate_content=generate_content)))
    monkeypatch.setattr(AB, "gemini_api_keys", lambda: ["k"])
    monkeypatch.setattr(base, "_judge_clients", lambda: [client])
    monkeypatch.setattr(base, "_next_judge_client", lambda: client)
    monkeypatch.setattr(AB, "judge_cascade", lambda: ["g1", "g2", "g3"])
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", "1")
    return calls


def test_user_simulator_replies_at_max_tokens_are_counted(monkeypatch):
    calls = _fake_cascade(monkeypatch, [_resp(None, "MAX_TOKENS"), RuntimeError("429"),
                                        _resp("I need Frank's number.", "MAX_TOKENS")])
    user = AB._UserSimulator(["BaseApi"], "en")
    text = asyncio.run(user.get_init_prompt("call Frank"))
    assert text == "I need Frank's number." and user.models == ["g3"]
    assert user.cuts == {"cut": 1, "empty": 1, "outage_at_cap": 0}
    cfg = calls[0][1]
    # APIUSER's 1000 for the reply plus the Gemini thinking allowance
    assert cfg is None or (cfg.max_output_tokens == 1000 + 8192 and cfg.temperature == 0.001)
    # outside a simulator nothing is counted, and a plain stop counts as nothing
    _fake_cascade(monkeypatch, [_resp("fine", "STOP")])
    assert asyncio.run(AB._user_sim_generate([{"role": "system", "content": "i"},
                                              {"role": "user", "content": "o"}])) == ("fine", "g1")


def test_a_cascade_that_never_answers_is_an_outage(monkeypatch):
    _fake_cascade(monkeypatch, [_resp(None, "MAX_TOKENS")] * 3)
    user = AB._UserSimulator(["BaseApi"], "en")
    with pytest.raises(AB._UserSimulatorOutage):
        asyncio.run(user.get_init_prompt("q"))
    # the outage followed replies stopped at the limit with no text
    assert user.cuts == {"cut": 0, "empty": 3, "outage_at_cap": 1}


def test_user_simulator_cuts_reach_the_result_and_break_comparability(monkeypatch):
    trace = {"status": "OK", "is_correct": True, "process_accuracy": 1.0,
             "acebench_language": "en", "category": "agent_multi_turn",
             "gold_answer": json.dumps({"sub": "agent_multi_turn", "id": "agent_multi_turn_0"}),
             "execution_guard_refusals": [], "truncated_replies": 0,
             "replies_at_official_max_tokens": 0, "model_requests": 2, "user_sim_requests": 3,
             "user_sim_models": {"g1": 3}, "user_sim_replies_cut_at_max_tokens": 2,
             "user_sim_empty_at_max_tokens": 1, "window_limited_requests": 0,
             "window_uncounted_requests": 0}
    res, _sent = _run(monkeypatch, False, base.Reply("[Api()]", None, "stop", completion_tokens=3),
                      agent_traces=[trace])
    sim = res["user_simulator"]
    assert sim["replies_cut_at_max_tokens"] == 2 and sim["empty_replies_at_max_tokens"] == 1
    assert "stopped at max_output_tokens 9192 (APIUSER's max_tokens 1000 plus 8192 for the " \
           "Gemini model's thinking) in 2 replies used as they were and 1 with no text" \
           in res["leaderboard_comparable_reason"]
    assert sim["max_output_tokens"] == 9192 and sim["outages_after_max_tokens"] == 0


# --------------------------------------------------------------------------- #
# docs match the code
# --------------------------------------------------------------------------- #

def test_the_docs_describe_the_new_behavior():
    ag = (DOCS / "bfcl_v4_agentic.md").read_text(encoding="utf-8")
    flat = " ".join(ag.split())
    assert "a check that gets no answer does not" in flat and "blocks it" in flat
    assert "update_mode" in ag and "count as unfinished" in ag
    assert "GBENCH_BFCL_REQUEST_TIMEOUT_S" in ag and "request_timeout_s" in ag
    assert "entries_unscored" in ag
    bf = (DOCS / "bfcl.md").read_text(encoding="utf-8")
    assert "262,144 tokens" in bf and "600 s read" not in bf
    assert "both stops are recorded, generate first" in bf
    ace = (DOCS / "acebench.md").read_text(encoding="utf-8")
    assert "max_model_len=8192" in ace and "upstream_context_window" in ace
    assert "replies_cut_at_max_tokens" in ace and "empty_replies_at_max_tokens" in ace
    assert "single_turn_replies_cut_off" in ace and "_row_scorer" in ace
    assert "`/home` included (only an interpreter directory" in ace
    for text in (ag, bf, ace):
        assert not [c for c in text if ord(c) > 127]


# --------------------------------------------------------------------------- #
# the token-limit policy in the other function-calling suites
# --------------------------------------------------------------------------- #

def test_nestful_and_t_eval_send_their_official_caps_without_thinking(monkeypatch):
    from gbench.runners.eval_suites import nestful, t_eval
    base.set_run_knobs(max_output_tokens=4096)
    for mod, sampling, cap in ((nestful, nestful.NESTFUL_SAMPLING, 1000),
                               (t_eval, t_eval.T_EVAL_SAMPLING, 512)):
        assert mod.UPSTREAM_MAX_TOKENS == cap and sampling.max_tokens == cap
        budget = base.resolve_token_budget(mod.__name__, sampling, thinking=False)
        assert (budget.policy, budget.max_tokens) == ("official", cap)
        assert "--max-output-tokens 4096 not applied" in budget.note
        budget = base.resolve_token_budget(mod.__name__, sampling, thinking=True)
        assert (budget.policy, budget.max_tokens) == ("reasoning_run_budget", 4096)
        assert budget.record()["max_tokens_applied"] is False


@pytest.mark.parametrize("name,attr", [("seal_tools", "SEAL_TOOLS_SAMPLING"),
                                       ("api_bank", "API_BANK_SAMPLING"),
                                       ("nexusraven_api_eval", "NEXUSRAVEN_SAMPLING"),
                                       ("gorilla_apibench", "GORILLA_SAMPLING")])
def test_suites_whose_harness_sends_no_max_tokens_send_none(name, attr):
    import importlib
    sampling = getattr(importlib.import_module(f"gbench.runners.eval_suites.{name}"), attr)
    assert sampling.send_max_tokens is False
    for thinking in (False, True):
        budget = base.resolve_token_budget(name, sampling, thinking=thinking)
        assert budget.policy == "not_sent" and budget.max_tokens is None


def test_a_no_max_tokens_suite_sends_none_on_the_wire(monkeypatch):
    from gbench.runners.eval_suites import seal_tools
    base.set_run_knobs(max_output_tokens=4096)
    sent = []

    async def send(**k):
        sent.append(k["max_output_tokens"])
        return base.Reply("[]", None, "stop", completion_tokens=2)
    samples = [([{"role": "user", "content": "q"}], "[]", {"category": "c"})]
    with mock.patch.object(base, "_send_single_request", send):
        res = base.run_eval_suite(eval_name="seal_tools", model_name="m",
                                  base_url="http://127.0.0.1:9/v1", concurrency=1,
                                  samples=samples, eval_fn=lambda p, g: True,
                                  official_sampling=seal_tools.SEAL_TOOLS_SAMPLING)
    assert sent == [None] and "not sent" in res["max_output_tokens_note"]


def test_cut_replies_in_nestful_and_t_eval_are_counted_by_mode():
    from gbench.runners.eval_suites import nestful, t_eval
    for mod in (nestful, t_eval):
        cap = mod.UPSTREAM_MAX_TOKENS
        traces = [{"finish_reason": "length", "max_tokens_sent": cap},
                  {"finish_reason": "length", "max_tokens_sent": cap - 7},
                  {"finish_reason": "stop", "max_tokens_sent": cap}]
        assert mod.replies_cut_off(traces, thinking=False) == 1
        assert mod.replies_cut_off(traces, thinking=True) == 2
    reasons = nestful._noncanonical_reasons({"temperature": 0.0, "replies_cut_off": 2,
                                             "official_sampling": {
                                                 "temperature_matches_official": True}}, None)
    assert reasons == ["2 replies cut off by the token budget before they finished (not at "
                       "NESTFUL's max_tokens 1000)"]


def test_t_eval_status_leaves_official_cap_replies_out_of_the_truncations():
    from gbench.runners.eval_suites import t_eval
    files = [{"total_questions": 10, "failed_requests": 0, "truncated_responses": 5,
              "replies_at_official_max_tokens": 5, "judge_outages": 0}] * 2
    merged = t_eval._merge_file_results(files)
    assert merged["replies_at_official_max_tokens"] == 10 and merged["status"] == "success"
    files = [{"total_questions": 10, "failed_requests": 0, "truncated_responses": 5,
              "replies_at_official_max_tokens": 2, "judge_outages": 0}] * 2
    assert t_eval._merge_file_results(files)["status"] == "completed_with_errors"


def test_the_fc_docs_describe_the_budgets():
    for name, needles in (
            ("nestful", ("`max_tokens=1000`", "replies_cut_off", "official cap not applied")),
            ("t_eval", ("`max_tokens=512`", "max_new_tokens=512", "replies_cut_off")),
            ("seal_tools", ("No `max_tokens` is sent",)),
            ("api_bank", ("No request carries `max_tokens`",)),
            ("nexusraven_api_eval", ("No `max_tokens` is sent",)),
            ("gorilla_apibench", ("No `max_tokens` is sent",))):
        text = (DOCS / f"{name}.md").read_text(encoding="utf-8")
        for needle in needles:
            assert needle in text, (name, needle)
        assert "uses your `--max-output-tokens`" not in text
        assert "Every request uses the run's" not in text
        assert not [c for c in text if ord(c) > 127]


def test_agent_dojo_records_that_no_max_tokens_is_sent():
    from gbench.runners.eval_suites import agent_dojo
    note = agent_dojo._max_tokens_note(4096)
    assert note.startswith("not sent: AgentDojo's OpenAILLM sends no max_tokens")
    assert "--max-output-tokens 4096 not applied" in note
    assert "not applied" not in agent_dojo._max_tokens_note(None)
