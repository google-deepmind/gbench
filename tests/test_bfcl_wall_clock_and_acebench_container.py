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

"""BFCL bounds and ACEBench's evaluator container and token budget.

  * BFCL: no harness part has a wall clock by default (upstream bounds none); each request and
    entry is bounded as upstream bounds it. GBENCH_BFCL_TIMEOUT_S bounds each part on its own,
    and a part it stops is a harness error with everything else kept: finished multi-turn and
    agentic entries are scored, unfinished ones count as timed out, unscored single-turn replies
    are left out of every accuracy. A web search no Gemini grounding model could serve stops the
    run at once and leaves the Web Search setting infra_required.
  * ACEBench: the BinOp evaluator runs in a bubblewrap container of its own (own PID namespace,
    no /proc, no /run or Docker socket, only the interpreter on its root), whatever
    GBENCH_SANDBOX says; LLMInfer's max_tokens=1024 is sent exactly without --thinking and not
    applied with it; the user simulator gets APIUSER's own sampling.

The unit tests run offline; the ACEBench container tests start bubblewrap. The tests marked
`image` run the real harness image against local stub servers; set GBENCH_TEST_BFCL_IMAGE=1."""

import asyncio
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import acebench as AB
from gbench.runners.eval_suites import base, bfcl, bfcl_live
from gbench.runners.eval_suites import bfcl_v4_agentic as AG

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "evals"
KEY = "gemKEY0123456789abcdef0123456789abcdef"

image = pytest.mark.skipif(not os.environ.get("GBENCH_TEST_BFCL_IMAGE"),
                           reason="set GBENCH_TEST_BFCL_IMAGE=1 to run the BFCL harness image")


@pytest.fixture(autouse=True)
def _no_part_timeout(monkeypatch):
    monkeypatch.delenv("GBENCH_BFCL_TIMEOUT_S", raising=False)
    monkeypatch.delenv("BFCL_TIMEOUT_S", raising=False)


# --------------------------------------------------------------------------- #
# the part wall clock
# --------------------------------------------------------------------------- #

def test_no_part_has_a_wall_clock_unless_one_is_set(monkeypatch):
    assert AG.part_timeout() is None
    for raw, want in (("", None), ("0", None), ("120", 120.0), ("7.5", 7.5)):
        monkeypatch.setenv("GBENCH_BFCL_TIMEOUT_S", raw)
        assert AG.part_timeout() == want
    monkeypatch.delenv("GBENCH_BFCL_TIMEOUT_S")
    monkeypatch.setenv("BFCL_TIMEOUT_S", "30")
    assert AG.part_timeout() == 30.0
    monkeypatch.setenv("GBENCH_BFCL_TIMEOUT_S", "3h")
    with pytest.raises(ValueError, match="GBENCH_BFCL_TIMEOUT_S"):
        AG.part_timeout()


def test_each_part_gets_its_own_bound(monkeypatch):
    seen = []

    def run(eval_name, module, func, kwargs, **kw):
        seen.append((func, kw.get("timeout"), kw.get("part")))
        return [] if func == "single_turn_verdicts" else {"traces": [], "guard_refusals": []}
    monkeypatch.setattr(AG, "run_in_image", run)
    bfcl.check_single_turn([{"id": "x"}])
    bfcl.run_multi_turn_in_image("m", "http://x/v1", 1, [], temperature=0.001, thinking=False)
    assert seen == [("single_turn_verdicts", None, "the single-turn checker"),
                    ("multi_turn_job", None, "the multi-turn part")]
    monkeypatch.setenv("GBENCH_BFCL_TIMEOUT_S", "90")
    seen.clear()
    bfcl.check_single_turn([{"id": "x"}])
    bfcl.run_multi_turn_in_image("m", "http://x/v1", 1, [], temperature=0.001, thinking=False)
    assert [t for _f, t, _p in seen] == [90.0, 90.0]


def test_the_launcher_stops_a_container_past_its_wall_clock():
    t0 = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        AG._launch([sys.executable, "-c", "import time; time.sleep(60)"], dict(os.environ), 1,
                   "gbench-test-not-a-container")
    assert time.monotonic() - t0 < 30


def test_a_watch_that_reports_a_problem_stops_the_container(monkeypatch):
    monkeypatch.setattr(AG, "_WATCH_EVERY_S", 0.2)
    calls = []

    def watch():
        calls.append(1)
        return RuntimeError("quota gone") if len(calls) >= 2 else None
    t0 = time.monotonic()
    with AG._watching(watch), pytest.raises(RuntimeError, match="quota gone"):
        AG._launch([sys.executable, "-c", "import time; time.sleep(60)"], dict(os.environ), None,
                   "gbench-test-not-a-container")
    assert len(calls) == 2 and time.monotonic() - t0 < 30
    assert AG._LAUNCH_WATCH.get() is None


def test_a_credential_on_the_cut_of_a_long_line_is_still_masked():
    env = dict(os.environ, GEMINI_API_KEY=KEY)
    for head in (65536 - 10, 65536 - 1, 65536 + 3):
        program = (f"import sys; sys.stdout.write('x' * {head} + {KEY!r} + 'y' * 50); "
                   "sys.stdout.flush()")
        rc, out = AG._launch([sys.executable, "-c", program], env, 30, "gbench-test-unused")
        assert rc == 0 and KEY not in out and base.REDACTED in out


def test_a_stopped_job_raises_harness_timeout_with_its_masked_progress(monkeypatch):
    def launch(cmd, env, timeout, name, stream=False):
        io_dir = next(m.split(":")[0] for m in cmd if m.endswith(f":{AG._IMAGE_IO_DIR}:rw"))
        with open(os.path.join(io_dir, AG._PROGRESS_NAME), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"job": 0, "trace": {"text": f"key {KEY}"}}) + "\n")
            fh.write('{"job": 1, "trace": {"cut')              # the stop cut this one off
        raise subprocess.TimeoutExpired(cmd, timeout)
    monkeypatch.setattr(AG, "_launch", launch)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    with pytest.raises(AG.HarnessTimeout) as e:
        AG.run_in_image("bfcl", "bfcl", "multi_turn_job", {}, timeout=5, part="the multi-turn part")
    assert e.value.part == "the multi-turn part" and e.value.timeout == 5.0
    assert "GBENCH_BFCL_TIMEOUT_S=5s" in str(e.value)
    assert e.value.progress == [{"job": 0, "trace": {"text": f"key {base.REDACTED}"}}]
    assert e.value.record(n=1) == {"part": "the multi-turn part", "error": str(e.value),
                                   "timeout_s": 5.0, "n": 1}


def test_a_job_reports_progress_into_the_exchange_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    (tmp_path / "job.json").write_text(json.dumps({
        "module": "bfcl_v4_agentic", "func": "report_progress",
        "kwargs": {"row": {"job": 3, "note": f"k={KEY}"}}}), encoding="utf-8")
    assert AG._image_main(str(tmp_path)) == 0
    [line] = (tmp_path / AG._PROGRESS_NAME).read_text(encoding="utf-8").splitlines()
    assert json.loads(line) == {"job": 3, "note": f"k={base.REDACTED}"}
    assert AG._PROGRESS_PATH is None                 # reset once the job is done
    AG.report_progress({"job": 4})                   # outside a job: a no-op
    assert len((tmp_path / AG._PROGRESS_NAME).read_text().splitlines()) == 1


# --------------------------------------------------------------------------- #
# the multi-turn part
# --------------------------------------------------------------------------- #

def test_the_multi_turn_loop_reports_every_finished_entry(monkeypatch):
    reported = []

    class Guard:
        refused = []

    @contextlib.contextmanager
    def patches():
        yield Guard

    def score(handler, item, key):
        if item["id"] == "b":
            Guard.refused.append("open('x')")
        return {"category": item["category"], "question_id": item["id"], "is_correct": True,
                "status": "OK", "health": "ok", "steps": 1}
    monkeypatch.setattr(bfcl, "_fc_handler", lambda *a, **k: type(
        "H", (), {"model_name_underline_replaced": a[4]})())
    monkeypatch.setattr(bfcl, "_harness_patches", patches)
    monkeypatch.setattr(bfcl, "_score_multi_turn_entry", score)
    monkeypatch.setattr(bfcl, "report_progress", reported.append)
    items = [{"id": i, "category": "multi_turn_base"} for i in ("a", "b", "c")]
    res = bfcl.run_multi_turn("m", "http://x/v1", 1, items, temperature=0.001, thinking=False,
                              attempts=2)
    assert sorted(r["job"] for r in reported) == list(range(6))
    assert all(r["trace"]["question_id"] == items[r["job"] // 2]["id"] for r in reported)
    assert sum(len(r["refused"]) for r in reported) == len(res["guard_refusals"]) == 2


def test_a_stopped_multi_turn_part_keeps_what_it_finished(monkeypatch):
    items = [{"id": f"multi_turn_base_{i}", "category": "multi_turn_base"} for i in range(3)]
    done = {"category": "multi_turn_base", "question_id": "multi_turn_base_1", "is_correct": True,
            "status": "OK", "health": "ok", "steps": 4}

    def run(*a, **kw):
        raise AG.HarnessTimeout("bfcl", kw["part"], 60,
                                [{"job": 2, "trace": done, "refused": ["exec('x')"]}])
    monkeypatch.setattr(AG, "run_in_image", run)
    res = bfcl.run_multi_turn_in_image("m", "http://x/v1", 2, items, temperature=0.001,
                                       thinking=False, attempts=2)
    traces = res["traces"]
    assert [t["sample_idx"] for t in traces] == list(range(6))
    assert traces[2]["status"] == "OK" and traces[2]["attempt_idx"] == 0
    assert traces[2]["source_sample_idx"] == 1
    others = [t for i, t in enumerate(traces) if i != 2]
    assert all(t["status"] == "TIMEOUT" and t["harness_timeout"] and not t["is_correct"]
               for t in others)
    assert res["guard_refusals"] == ["exec('x')"]
    assert res["harness_error"]["entries_finished"] == 1
    assert res["harness_error"]["entries_unfinished"] == 5
    counts = bfcl._multi_turn_counts(traces)
    assert counts["timed_out_requests"] == counts["failed_requests"] == 5
    # the unfinished entries were never measured: left out of the score (harness errors)
    assert counts["category_accuracy"]["multi_turn_base"] == {"correct": 1, "total": 1,
                                                               "accuracy": 100.0}
    assert counts["scoring_excluded"] == 5
    assert counts["request_harness_errors"] == {"unfinished": 5}
    assert all(t["scoring_excluded"] and t["harness_error"]["kind"] == "unfinished"
               for t in others)


# --------------------------------------------------------------------------- #
# the single-turn checker
# --------------------------------------------------------------------------- #

def _trace(i, cat="simple_python"):
    return {"sample_idx": i, "category": cat, "response_text": "", "tool_calls": None,
            "gold_answer": {"id": f"{cat}_{i}", "category": cat, "ground_truth": []},
            "status": "PENDING_JUDGE", "health": "ok"}


def test_a_stopped_checker_leaves_the_replies_unscored_not_wrong():
    def checker(items):
        raise AG.HarnessTimeout("bfcl", "the single-turn checker", 30)
    scorer = bfcl.bfcl_batch_scorer({f"simple_python_{i}": [] for i in range(3)}, checker=checker)
    traces = [_trace(i) for i in range(3)]
    asyncio.run(scorer(traces))
    assert all(t["status"] == "HARNESS_ERROR" and t["scoring_excluded"] and not t["is_correct"]
               for t in traces)
    assert scorer.harness_errors[0]["replies_unscored"] == 3
    assert base.tally_batch_scores(traces) == (0, {})       # out of every accuracy


def _parts(monkeypatch, tmp_path, *, checker_stops=False, multi_stops=False,
           evaluate_stops=False):
    """run_bfcl with every part stubbed at the harness boundary."""
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(tmp_path / "bfcl"))
    monkeypatch.setattr(bfcl, "_harness_probe", lambda name, multi, single: (
        bfcl.BFCL_EVAL_RELEASE, [{"id": f"{c}_0", "category": c} for c in multi]))
    samples = [([{"role": "user", "content": "q"}],
                {"id": f"{c}_0", "category": c, "ground_truth": []}, {"category": c})
               for c in bfcl.SINGLE_TURN_CATEGORIES]
    monkeypatch.setattr(bfcl, "_load_bfcl_samples", lambda cats: (
        [s for s in samples if s[2]["category"] in cats], {s[1]["id"]: [] for s in samples}))

    def check(items, eval_name="bfcl"):
        if checker_stops:
            raise AG.HarnessTimeout(eval_name, "the single-turn checker", 30)
        return [{"valid": True, "error_type": None, "error": []} for _ in items]
    monkeypatch.setattr(bfcl, "check_single_turn", check)

    def run_in_image(*a, **kw):
        if multi_stops:
            raise AG.HarnessTimeout("bfcl", kw["part"], 30)
        entries = a[3]["entries"]
        return {"traces": [{"sample_idx": i, "source_sample_idx": i, "attempt_idx": 0,
                            "category": c, "question_id": e, "is_correct": True, "status": "OK",
                            "health": "ok", "steps": 1} for i, (e, c) in enumerate(entries)],
                "guard_refusals": []}
    monkeypatch.setattr(AG, "run_in_image", run_in_image)
    # the multi-turn container is stubbed, so the network checks before it
    # (tests/test_guard_code_images.py) do not apply
    monkeypatch.setattr(bfcl, "endpoint_network", lambda base_url: base_url)
    monkeypatch.setattr(AG, "check_bfcl_prerequisites",
                        lambda url: (True, "", list(AG.AGENTIC_CATEGORIES)))
    monkeypatch.setattr(AG, "search_backend", lambda: (AG.SEARCH_BACKEND, ""))

    def cli(args, env, timeout, model="", backend=None):
        if args[0] == "evaluate" and evaluate_stops:
            raise subprocess.TimeoutExpired(cmd="bfcl evaluate", timeout=timeout)
        return 0, ""
    monkeypatch.setattr(AG, "_run_cli", cli)
    monkeypatch.setattr(AG, "_clear_stale_scores", lambda *a: 0)
    monkeypatch.setattr(AG, "_parse_scores", lambda root, m, cats: {
        c: {"correct": 1, "total": 1, "accuracy": 100.0} for c in cats})
    monkeypatch.setattr(AG, "_parse_score_traces", lambda *a: [])
    monkeypatch.setattr(AG, "_score_error_types", lambda *a: {})

    async def send(**k):
        return base.Reply("", [{"id": "c", "type": "function",
                                "function": {"name": "f", "arguments": "{}"}}], "tool_calls",
                          completion_tokens=3)
    return send


def test_a_full_run_with_no_stop_is_unchanged(monkeypatch, tmp_path):
    send = _parts(monkeypatch, tmp_path)
    with mock.patch.object(base, "_send_single_request", send):
        res = bfcl.run_bfcl("m", "http://x/v1", 2)
    assert res["harness_errors"] == [] and res["status"] == "success"
    # the deviations: the no-think 0.0 (gbench's rule) and the Gemini-served web search
    assert res["accuracy"] == 100.0 and res["leaderboard_comparable"] is False
    assert res["leaderboard_comparable_reason"].startswith(
        f"temperature 0.0 (BFCL generates at 0.001); agentic: {AG.SEARCH_SUBSTITUTION_REASON}")


def test_a_stopped_part_is_a_harness_error_and_the_rest_is_kept(monkeypatch, tmp_path):
    send = _parts(monkeypatch, tmp_path, multi_stops=True, evaluate_stops=True)
    with mock.patch.object(base, "_send_single_request", send):
        res = bfcl.run_bfcl("m", "http://x/v1", 2)
    parts = [e["part"] for e in res["harness_errors"]]
    assert parts == ["the multi-turn part", "`bfcl evaluate`"]
    ca = res["category_accuracy"]
    assert all(ca[c]["correct"] == ca[c]["total"] == 1 for c in bfcl.SINGLE_TURN_CATEGORIES)
    # the multi-turn part was stopped before any entry finished: nothing of it measured the
    # model, so its categories are unscored (N/A), never 0, and its stop record says so
    assert not any(c in ca for c in bfcl.MULTI_TURN_CATEGORIES)
    assert res["harness_errors"][0]["categories_unscored"] == sorted(bfcl.MULTI_TURN_CATEGORIES)
    assert not any(c in ca for c in bfcl.AGENTIC_CATEGORIES)      # unscored, N/A
    assert res["timed_out_requests"] == 4 and res["status"] == "completed_with_errors"
    assert res["bfcl_summary"]["overall"]["Overall Acc"] is None
    assert res["bfcl_summary"]["non_live"]["Non-Live Overall Acc"] == 100.0
    assert res["leaderboard_comparable"] is False
    assert "harness error: bfcl: the multi-turn part did not finish" in \
        res["leaderboard_comparable_reason"]


def test_a_stopped_checker_keeps_the_other_parts(monkeypatch, tmp_path):
    send = _parts(monkeypatch, tmp_path, checker_stops=True)
    with mock.patch.object(base, "_send_single_request", send):
        res = bfcl.run_bfcl("m", "http://x/v1", 2,
                            eval_categories="simple_python,multi_turn_base")
    assert [e["part"] for e in res["harness_errors"]] == ["the single-turn checker"]
    assert "simple_python" not in res["category_accuracy"]
    assert res["category_accuracy"]["multi_turn_base"]["correct"] == 1
    assert res["status"] == "completed_with_errors" and res["pooled_accuracy"] == 100.0


def test_bfcl_live_reports_a_stopped_checker(monkeypatch):
    monkeypatch.setattr(bfcl, "_harness_probe", lambda name, multi, single: (
        bfcl.BFCL_EVAL_RELEASE, []))
    monkeypatch.setattr(bfcl_live, "_load_bfcl_live_samples", lambda: (
        [([{"role": "user", "content": "q"}],
          {"id": "live_simple_0", "category": "live_simple", "ground_truth": []},
          {"category": "live_simple"})], {"live_simple_0": []}))
    monkeypatch.setattr(bfcl, "check_single_turn", lambda items, eval_name="bfcl": (
        _ for _ in ()).throw(AG.HarnessTimeout(eval_name, "the single-turn checker", 9)))

    async def send(**k):
        return base.Reply("no call", None, "stop", completion_tokens=3)
    with mock.patch.object(base, "_send_single_request", send):
        res = bfcl_live.run_bfcl_live("m", "http://x/v1", 1)
    assert res["status"] == "failed" and len(res["harness_errors"]) == 1
    assert "harness error" in res["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# the agentic part
# --------------------------------------------------------------------------- #

@pytest.fixture
def agentic(monkeypatch, tmp_path):
    seen = {"cli": [], "stop": None, "watch": [], "files": {}}
    root = tmp_path / "bfcl"
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(root))
    monkeypatch.setattr(AG, "check_bfcl_prerequisites",
                        lambda url: (True, "", list(AG.AGENTIC_CATEGORIES)))
    monkeypatch.setattr(AG, "search_backend", lambda: (AG.SEARCH_BACKEND, ""))

    def cli(args, env, timeout, model="", backend=None):
        seen["cli"].append(list(args))
        if args[0] == "generate":
            seen["watch"].append(AG._LAUNCH_WATCH.get())
            for rel, text in seen["files"].items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
        if seen["stop"] == args[0]:
            raise subprocess.TimeoutExpired(cmd=f"bfcl {args[0]}", timeout=timeout)
        return 0, ""
    monkeypatch.setattr(AG, "_run_cli", cli)
    # after a stopped generate, memory_kv has no entry finished and so no score file
    monkeypatch.setattr(AG, "_parse_scores", lambda r, m, cats: {
        c: {"correct": 1, "total": 2, "accuracy": 50.0} for c in cats
        if not (seen["stop"] == "generate" and c == "memory_kv")})
    monkeypatch.setattr(AG, "_parse_score_traces", lambda *a: [])
    monkeypatch.setattr(AG, "_score_error_types", lambda *a: {})
    monkeypatch.setattr(AG, "run_in_image", lambda eval_name, module, func, kwargs, **kw: (
        {c: 5 for c in kwargs["categories"]} if func == "scored_entry_counts"
        else {c: [f"{c}_{i}" for i in range(3)] + [f"{c}_prereq_0"] for c in kwargs["categories"]}))
    seen["root"] = root
    return seen


def test_a_stopped_generate_scores_what_finished_and_counts_the_rest(agentic, monkeypatch):
    monkeypatch.setenv("GBENCH_BFCL_TIMEOUT_S", "600")
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    agentic["stop"] = "generate"
    agentic["files"] = {"result/m/agentic/BFCL_v4_web_search_base_result.json":
                        json.dumps({"id": "web_search_base_0", "result": f"k {KEY}"}) + "\n"
                        + '{"id": "web_search_base_1", "res'}
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    evaluate = next(c for c in agentic["cli"] if c[0] == "evaluate")
    assert "--partial-eval" in evaluate
    ca = res["category_accuracy"]
    # the unfinished entries are harness errors, left out of the score
    assert ca["web_search_base"] == {"correct": 1, "total": 2, "accuracy": 50.0,
                                     "unfinished": 3}
    assert "memory_kv" not in ca and res["categories_unscored"] == ["memory_kv"]
    assert res["bfcl_report"]["harness_errors_left_out"]["memory_kv"] == {"unfinished": 5}
    [err] = res["harness_errors"]
    assert err["part"] == "`bfcl generate`" and err["timeout_s"] == 600.0
    assert err["entries_planned"] == 25 and err["entries_unfinished"] == 17
    assert res["failed_requests"] == res["timed_out_requests"] == 17
    assert res["scoring_excluded"] == 17 and res["request_harness_errors"] == {"unfinished": 17}
    assert res["status"] == "completed_with_errors" and res["leaderboard_comparable"] is False
    # the cut-off line is dropped and the key masked, although generate did not return
    text = (agentic["root"] / "result/m/agentic/BFCL_v4_web_search_base_result.json").read_text()
    assert text.splitlines() == [json.dumps({"id": "web_search_base_0",
                                             "result": f"k {base.REDACTED}"})]


def test_a_stopped_limited_generate_counts_its_own_ids(agentic, monkeypatch):
    agentic["stop"] = "generate"
    stale = agentic["root"] / "result_limit4/m/memory/BFCL_v4_memory_kv_result.json"
    stale.parent.mkdir(parents=True)
    stale.write_text(json.dumps({"id": "memory_kv_0", "result": "an earlier run"}) + "\n")
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2, limit=4)
    assert not stale.exists()           # only this run's entries can be scored
    # prereqs left out of the planned count; nothing finished, so nothing is in the score
    assert res["bfcl_report"]["harness_errors_left_out"]["memory_vector"] == {"unfinished": 3}
    assert res["category_accuracy"] == {} and res["accuracy"] is None
    assert res["harness_errors"][0]["entries_planned"] == 15
    # nothing was written before the stop: nothing to evaluate, every entry unfinished
    assert not any(c[0] == "evaluate" for c in agentic["cli"])
    assert res["harness_errors"][0]["entries_unfinished"] == 15


def test_a_stopped_evaluate_raises_a_harness_timeout(agentic):
    agentic["stop"] = "evaluate"
    with pytest.raises(AG.HarnessTimeout, match="`bfcl evaluate` did not finish"):
        AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)


def test_generate_is_watched_only_when_it_searches(agentic):
    AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2)
    AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=2,
                           eval_categories=",".join(AG.MEMORY_CATEGORIES))
    assert callable(agentic["watch"][0]) and agentic["watch"][1] is None


def _served(results=3, backend="gemini:gemini-3.7-flash"):
    return json.dumps({"backend": backend, "results": results, "error": None}) + "\n"


def test_the_search_watch_stops_a_run_at_its_first_failed_search(tmp_path):
    log = tmp_path / AG.SEARCH_LOG_NAME
    log.write_text(_served())
    watch = AG._search_watch(log)
    assert watch() is None                              # served
    with log.open("a") as f:
        f.write(_served(results=0, backend="gemini:gemini-3.6-flash"))
    assert watch() is None                              # answered without a result
    with log.open("a") as f:
        f.write(json.dumps({"backend": "all-grounding-models-failed", "results": 0,
                            "error": "search backend failed: HTTP Error 503"}) + "\n")
    problem = watch()
    assert isinstance(problem, AG.SearchUnavailable)
    assert "INFRA REQUIRED" in str(problem) and "for 1 of the 3 web searches" in str(problem)
    assert "HTTP Error 503" in str(problem)


def test_the_search_watch_names_each_kind_of_failure(tmp_path):
    for row, needle in (({"backend": "all-grounding-models-failed", "results": 0,
                          "error": "all grounding models failed"},
                         "all-grounding-models-failed: all grounding models failed"),
                        ({"backend": "none", "results": 0, "error": "no key"}, "none: no key"),
                        ({"backend": "error", "results": 0, "error": "KeyError: 'x'"},
                         "error: KeyError")):
        log = tmp_path / f"{row['backend']}.jsonl"
        log.write_text(_served() + json.dumps(row) + "\n")
        problem = AG._search_watch(log)()
        assert "INFRA REQUIRED" in str(problem) and needle in str(problem)
        assert "`bfcl generate` was stopped" in str(problem)


def test_the_search_watch_never_checks_the_key_again(tmp_path, monkeypatch):
    log = tmp_path / AG.SEARCH_LOG_NAME
    monkeypatch.setattr(AG, "search_backend", lambda: pytest.fail("the log alone decides"))
    watch = AG._search_watch(log)
    for _ in range(3):
        with log.open("a") as f:
            f.write(_served())
        assert watch() is None


# --------------------------------------------------------------------------- #
# ACEBench: the evaluator container
# --------------------------------------------------------------------------- #

def test_the_evaluator_runs_in_its_own_container_whatever_gbench_sandbox_says(monkeypatch):
    monkeypatch.setenv("GBENCH_SANDBOX", "none")
    argv = AB._evaluator_argv()
    assert os.path.basename(argv[0]) == "bwrap"
    for flag in ("--unshare-all", "--unshare-user", "--clearenv", "--new-session",
                 "--die-with-parent"):
        assert flag in argv
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    binds = [argv[i + 1] for i, a in enumerate(argv) if a in ("--ro-bind", "--bind")]
    assert "/" not in binds and "--proc" not in argv
    assert not any(b == "/run" or b.startswith(("/run/", "/var", "/home", "/proc")) for b in binds)
    assert argv[-4:-1] == [sys.executable, "-I", "-c"]


def test_the_container_has_its_own_pid_namespace_and_no_host_proc_or_docker_socket(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", KEY)
    probe = ("[__import__('os').getpid(), __import__('os').path.exists('/proc'), "
             "__import__('os').path.exists('/run'), "
             "__import__('os').path.exists('/var/run/docker.sock'), "
             "__import__('os').path.exists('/run/docker.sock'), "
             "sorted(__import__('os').environ), sorted(__import__('os').listdir('/'))] + []")
    pid, has_proc, has_run, var_sock, run_sock, env, root = AB._isolated_eval(probe, "normal")
    assert pid <= 16                                       # one of the first in its namespace
    assert not (has_proc or has_run or var_sock or run_sock)
    assert "SERPAPI_API_KEY" not in env
    allowed = set(AB._CONTAINER_ROOT) | {p.strip("/").split("/")[0]
                                         for p in AB._interpreter_dirs()}
    assert set(root) <= allowed
    assert AB.container_problem() is None
    AB.check_binop_evaluator()


def test_a_container_that_is_not_isolated_stops_the_suite(monkeypatch):
    real = AB._isolated_eval

    def leaky(source, scope):
        if source == AB._CONTAINER_PROBE:
            return [4242, True, True, True, ["home", "proc", "run", "usr"]]
        return real(source, scope)
    monkeypatch.setattr(AB, "_isolated_eval", leaky)
    why = AB.container_problem()
    assert "process 4242" in why and "/proc, /run or the Docker socket" in why
    assert "home, proc, run" in why
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*acebench.*container"):
        AB.check_binop_evaluator()


def test_a_stdlib_exception_keeps_its_name_module_and_text():
    with pytest.raises(Exception) as e:
        AB._isolated_eval("__import__('json').loads('{') + ''", "normal")
    assert type(e.value).__name__ == "JSONDecodeError"
    assert type(e.value).__module__ == "json.decoder"
    assert str(e.value) == "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        AB._isolated_eval("[1 / 0] + []", "normal")


def test_agent_decoding_leaves_the_event_loop_free_and_keeps_its_refusals(monkeypatch):
    threads = []
    real = AB.decode_function_list

    def decode(result, non_call="fail"):
        threads.append(threading.current_thread() is threading.main_thread())
        return real(result, non_call)
    monkeypatch.setattr(AB, "decode_function_list", decode)

    async def conversation():
        sink = []
        token = AB._REFUSALS.set(sink)
        try:
            with pytest.raises(AB._GuardRefusal):
                await AB._decode_off_loop("[F(x=value + 1)]", "fail")
            return sink
        finally:
            AB._REFUSALS.reset(token)
    assert asyncio.run(conversation()) == ["value + 1"]
    assert threads == [False]


# --------------------------------------------------------------------------- #
# ACEBench: the token-limit policy
# --------------------------------------------------------------------------- #

def test_acebench_declares_llminfers_max_tokens():
    assert AB.ACEBENCH_SAMPLING.max_tokens == AB.UPSTREAM_MAX_TOKENS == 1024
    assert "max_tokens=1024" in AB.ACEBENCH_SAMPLING.reference
    assert AB.ACEBENCH_SAMPLING.reasoning is None      # ACEBench has no reasoning-model settings


@pytest.mark.parametrize("thinking,sent,policy", [(False, 1024, "official"),
                                                  (True, 4096, "reasoning_run_budget")])
def test_acebench_requests_follow_the_token_limit_policy(monkeypatch, thinking, sent, policy):
    base.set_run_knobs(max_output_tokens=4096)
    gold = json.dumps({"id": "normal_atom_bool_1", "sub": "normal_atom_bool", "language": "en",
                       "ground_truth": {"Api": {"q": True}}, "functions": [], "question": ""})
    monkeypatch.setattr(AB, "_load_acebench_samples", lambda **k: [
        ([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], gold,
         {"category": "normal_atom_bool"})] if k.get("language") == "en" else [])
    monkeypatch.setattr(AB, "_load_agent_items", lambda language="en": [
        {"category": "agent_multi_step", "key": "agent_multi_step", "language": language,
         "row": {"id": f"agent_multi_step_{language}"}, "answer": {}}])
    agent = {}

    async def run_agents(items, model_name, base_url, concurrency, thinking, temperature, payload,
                         max_tokens, official_cap=None, upstream_window=False):
        agent.update(max_tokens=max_tokens, official_cap=official_cap,
                     upstream_window=upstream_window)
        return []
    monkeypatch.setattr(AB, "_run_agent_items", run_agents)
    seen = []

    async def send(**k):
        seen.append(k["max_output_tokens"])
        return base.Reply("[Api(q=True)]", None, "length", completion_tokens=sent)
    with mock.patch.object(base, "_send_single_request", send):
        res = AB.run_acebench("m", "http://127.0.0.1:9/v1", 1, enable_thinking=thinking)
    assert seen == [sent] and agent["max_tokens"] == sent
    assert agent["official_cap"] == (1024 if not thinking else None)
    assert agent["upstream_window"] is (not thinking)     # LLMInfer's window, no-think only
    rec = res["official_sampling"]
    assert rec["max_tokens"] == 1024 and rec["max_tokens_applied"] is (not thinking)
    assert res["acebench_report"]["token_policy"] == policy
    assert res["acebench_report"]["max_tokens_per_request"] == sent
    if thinking:
        assert rec["max_tokens_sent"] == 4096 and "not applied to reasoning" in rec["max_tokens_note"]
        assert res["status"] == "completed_with_errors"       # a real truncation
    else:
        # a reply that stops at the official 1024 is the protocol's outcome, not a truncation
        assert res["replies_at_official_max_tokens"] == 1 and res["status"] == "success"
        assert "--max-output-tokens 4096 not applied" in res["max_output_tokens_note"]


def test_an_agent_reply_at_the_official_cap_is_not_a_truncation(monkeypatch):
    async def clamp(convo, budget, tools, ctk):
        return budget, None

    async def send(**k):
        return base.Reply("I am done.", None, "length", completion_tokens=k["max_output_tokens"])
    monkeypatch.setattr(AB, "clamp_to_context_async", clamp)
    shared = {"session": None, "api_url": "http://x", "model_name": "m", "payload": {},
              "semaphore": asyncio.Semaphore(1), "budget": 1024, "official_cap": 1024,
              "temperature": 0.0, "thinking": False, "ctk": None}
    with mock.patch.object(base, "_send_single_request", send):
        model = AB._AgentModel(shared)
        asyncio.run(model.inference("s", "u"))
        assert (model.at_official_cap, model.truncated) == (1, 0)
        model = AB._AgentModel({**shared, "official_cap": None, "budget": 4096})
        asyncio.run(model.inference("s", "u"))
        assert (model.at_official_cap, model.truncated) == (0, 1)


def test_the_user_simulator_sends_apiusers_own_sampling(monkeypatch):
    # APIModel_user.APIUSER(temperature=0.001, top_p=1, max_tokens=1000), built with its defaults
    # by common_inference; only the model is replaced.
    assert (AB.USER_SIM_TEMPERATURE, AB.USER_SIM_TOP_P, AB.USER_SIM_MAX_TOKENS) == (0.001, 1.0, 1000)
    seen = {}

    async def cascade(contents, config=None):
        seen["config"] = config
        return "hello", "gemini-x"
    monkeypatch.setattr(AB, "_user_sim_cascade", cascade)
    msgs = [{"role": "system", "content": "INSTR"}, {"role": "user", "content": "open"}]
    assert asyncio.run(AB._user_sim_generate(msgs)) == ("hello", "gemini-x")
    cfg = seen["config"]
    assert (cfg.system_instruction, cfg.temperature, cfg.top_p, cfg.max_output_tokens) == (
        "INSTR", 0.001, 1.0, 1000 + 8192)   # APIUSER's 1000 plus the thinking allowance


def test_acebench_status_leaves_official_cap_replies_out_of_the_truncations():
    res = {"total_questions": 10, "failed_requests": 0, "truncated_responses": 5,
           "replies_at_official_max_tokens": 5}
    assert AB._status(res) == "success"
    res["replies_at_official_max_tokens"] = 3
    assert AB._status(res) == "completed_with_errors"


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #

def test_docs_describe_the_bounds_the_container_and_the_budget():
    texts = [(DOCS / f).read_text(encoding="utf-8")
             for f in ("bfcl.md", "bfcl_v4_agentic.md", "bfcl_live.md", "acebench.md")]
    assert all(t.isascii() for t in texts)
    bfcl_doc, agentic_doc, live_doc, ace = (" ".join(t.split()) for t in texts)
    for doc in (bfcl_doc, agentic_doc, live_doc):
        assert "GBENCH_BFCL_TIMEOUT_S" in doc and "harness_errors" in doc
        assert "10800" not in doc
    for needle in ("timeout=72000", "20-step limit", "--partial-eval", "harness_timeout",
                   "HARNESS_ERROR"):
        assert needle in bfcl_doc, needle
    # (the agentic requests' timeout is sized for the window, no longer the SDK's 600 s read)
    for needle in ("GBENCH_BFCL_REQUEST_TIMEOUT_S", "MAXIMUM_STEP_LIMIT", "entries_unfinished",
                   "reads the log every 30 s"):
        assert needle in agentic_doc, needle
    for needle in ("max_tokens=1024", "reasoning_run_budget", "--unshare-all", "No `/proc`",
                   "Docker daemon's socket", "container_problem", "JSONDecodeError",
                   "not even `none`"):
        assert needle in ace, needle
    assert "gbench uses your `--max-output-tokens`" not in ace


# --------------------------------------------------------------------------- #
# the real image (GBENCH_TEST_BFCL_IMAGE=1)
# --------------------------------------------------------------------------- #

def _slow_server(delay):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, data):
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"object": "list", "data": [{"id": "stub", "object": "model"}]})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            time.sleep(delay)
            self._send({"id": "x", "object": "chat.completion", "created": 0,
                        "model": req["model"],
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": "Done."}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                  "total_tokens": 2}})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


@image
def test_image_a_stopped_multi_turn_part_keeps_its_finished_entries(monkeypatch):
    monkeypatch.setenv("GBENCH_BFCL_TIMEOUT_S", "40")
    srv, url = _slow_server(3)
    items = [{"id": f"multi_turn_base_{i}", "category": "multi_turn_base"} for i in range(8)]
    try:
        res = bfcl.run_multi_turn_in_image("stub", url, 1, items, temperature=0.001,
                                           thinking=False)
    finally:
        srv.shutdown()
    finished = [t for t in res["traces"] if not t.get("harness_timeout")]
    assert 1 <= len(finished) < 8
    assert all(t["status"] == "OK" for t in finished)
    assert res["harness_error"]["entries_finished"] == len(finished)


@image
def test_image_a_stopped_generate_still_produces_a_result(tmp_path, monkeypatch):
    monkeypatch.setenv("GBENCH_BFCL_TIMEOUT_S", "45")
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(tmp_path / "bfcl"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    srv, url = _slow_server(2)
    try:
        res = AG.run_bfcl_v4_agentic("stub", url, 1, eval_categories="memory_kv", limit=3)
    finally:
        srv.shutdown()
    [err] = res["harness_errors"]
    assert err["part"] == "`bfcl generate`" and err["entries_planned"] == 3
    # every planned entry is accounted for: in the score, or left out of it as a harness error
    # (the unfinished ones among them)
    assert res["total_questions"] == 3
    row = res["category_accuracy"]["memory_kv"]
    assert row["total"] + res["scoring_excluded"] == 3
    assert (res.get("request_harness_errors") or {}).get("unfinished", 0) == row["unfinished"]
    assert res["status"] == "completed_with_errors"
