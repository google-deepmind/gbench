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

"""complexfuncbench, api_bank, acebench, gaia, deepsearch_qa, aa_lcr and beam_128k: canonical
prompts and settings, harness-error attribution and per-sample records.

No network, no model server, no Gemini call: the endpoint, the Gemini judge, the search backend,
the API-Bank checkout and the datasets are stubbed; files are written under pytest's tmp_path.
"""

import asyncio
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import aa_lcr, acebench as AB, api_bank as A, base
from gbench.runners.eval_suites import beam_128k, complexfuncbench as C
from gbench.runners.eval_suites import deepsearch_qa as D, gaia as G
from gbench.runners.eval_suites import search_tool

ROOT = Path(__file__).resolve().parent.parent


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _no_sleep(monkeypatch, module):
    slept = []

    async def instant(s):
        slept.append(s)
    monkeypatch.setattr(module.asyncio, "sleep", instant)
    return slept


# =========================================================================== #
# complexfuncbench
# =========================================================================== #
def test_cfb_judge_prompts_are_upstream_verbatim():
    for name, digest in C.PROMPT_SHA256.items():
        assert _sha(getattr(C, name)) == digest, name
    # the details the old port dropped
    assert C._COMPARE_SYSTEM.startswith('"You are an assistant for function call comparison.')
    assert "## Example 1" in C._COMPARE_SYSTEM
    assert "Note: The output must be a valid JSON" in C._COMPARE_SYSTEM
    assert C._COMPLETE_SYSTEM.count("## Example ") == 3
    assert "(e.g. \"all requested information is addressed\"" in C._COMPLETE_SYSTEM
    assert ("You should output the score for each dialogue history and corresponding response"
            in C._CORRECT_SYSTEM) and "- score: the completeness score" in C._CORRECT_SYSTEM
    assert C._COMPLETE_USER_TEMPLATE.endswith("output:\n    \n")
    assert C._CORRECT_USER_TEMPLATE.endswith("output:\n\n")
    assert not hasattr(C, "_COMPARE_ANCHOR") and not hasattr(C, "_SCORE_ANCHOR")


def test_cfb_fill_template_replaces_in_order_as_simple_template_prompt():
    assert C._fill_template("a [args1] b [args2]", "x", 3) == "a x b 3"
    # sequential, as upstream: a value that holds a later placeholder is replaced too
    assert C._fill_template("[args1]|[args2]", "[args2]", "y") == "y|y"


def test_cfb_call_judge_sends_upstreams_system_and_user_split(monkeypatch):
    seen = []
    reply = '```JSON\n{"is_equal": true, "reason": "same"}\n```'

    async def fake(contents, *, config=None):
        seen.append((contents, config))
        return reply, "gemini-a"
    monkeypatch.setattr(C, "judge_generate_cascade", fake)
    usage = Counter()
    cmp = C._CompareFC(C._load_exact_match_dict(), C._get_embedder(), usage)
    fn = [{"name": "F"}]
    hist = [{"role": "user", "content": "q"}]
    a, b = {"name": "F", "arguments": {"x": "1"}}, {"name": "F", "arguments": {"x": "one"}}
    assert asyncio.run(cmp.llm_based(fn, hist, a, b)) is True
    [(contents, config)] = seen
    assert config.system_instruction == C._COMPARE_SYSTEM and config.temperature == 0.0
    assert contents == C._fill_template(C._COMPARE_USER_TEMPLATE, json.dumps(fn),
                                        json.dumps(hist), json.dumps(a), json.dumps(b))
    assert cmp.judge_calls == [{"predict": a, "golden": b, "judge_model": "gemini-a",
                                "reply": reply, "is_equal": True}]
    assert usage == Counter({"gemini-a": 1})


def test_cfb_decode_json_is_upstreams():
    assert C._decode_json('```json\n{"a": True}\n```') == {"a": True}
    assert C._decode_json('prose {"a": 1}') is None           # upstream cannot read it
    assert C._decode_json(None) is None
    # an undecodable verdict is never a verdict
    assert C._parse_is_equal('{"reason": "x"}') is None and C._parse_is_equal("[1]") is None


def test_cfb_response_judge_retries_like_upstream_and_keeps_the_judges_reason(monkeypatch):
    slept = _no_sleep(monkeypatch, C)
    replies = iter(["no json here", "still none", '{"score": 2, "reason": "complete"}',
                    '{"score": 1, "reason": "partly"}'])

    async def fake(contents, *, config=None):
        return next(replies), "gemini-a"
    monkeypatch.setattr(C, "judge_generate_cascade", fake)
    calls = []
    ev = asyncio.run(C._resp_eval([{"role": "user", "content": "q"}], "answer", Counter(), calls))
    assert ev == {"complete": {"score": 2, "reason": "complete"},
                  "correct": {"score": 1, "reason": "partly"}}
    assert [(c["role"], c["attempt"], c["score"]) for c in calls] == [
        ("completeness", 1, None), ("completeness", 2, None), ("completeness", 3, 2),
        ("correctness", 1, 1)]
    assert slept == [C._RESP_JUDGE_RETRY_DELAY_S] * 2


def test_cfb_response_judge_gives_up_after_ten_unparsed_replies(monkeypatch):
    _no_sleep(monkeypatch, C)
    usage = Counter()

    async def fake(contents, *, config=None):
        return "never a score", "gemini-a"
    monkeypatch.setattr(C, "judge_generate_cascade", fake)
    ev = asyncio.run(C._resp_eval([{"role": "user", "content": "q"}], "a", usage))
    assert ev["complete"] == {"score": -1, "reason": C._RESP_JUDGE_UNPARSED}
    assert usage == Counter({"gemini-a": 2 * C._RESP_JUDGE_ATTEMPTS})


def _cfb_run(monkeypatch, thinking, knob, limit=4):
    seen = {}
    monkeypatch.setattr(C, "check_complexfuncbench_prerequisites", lambda: (True, ""))
    monkeypatch.setitem(base._RUN_KNOBS, "max_output_tokens", knob)
    monkeypatch.delitem(base._RUN_KNOBS, "temperature", raising=False)

    async def fake_drive(model_name, base_url, concurrency, temperature, limit,
                         max_output_tokens=None, enable_thinking=False):
        seen.update(temperature=temperature, max_tokens=max_output_tokens)
        rec = {"id": "Hotels-0", "request_failed": False, "message": "Success.",
               "count_dict": {"success_turn_num": 1, "total_turn_num": 1,
                              "correct_call_num": 1, "total_call_num": 1, "real_turn_num": 1},
               "resp_eval": {"complete": {"score": 2, "reason": "r"},
                             "correct": {"score": 2, "reason": "r"}},
               "gen_convs": [{"role": "user", "content": "q"}],
               "call_judge_calls": [], "resp_judge_calls": [{"judge_model": "gemini-a"}]}
        bad = dict(rec, id="Flights-0", message={"error_type": "stop_early"})
        agg = C._aggregate([rec, bad], 2, False)
        turns = {0: [{"finish_reason": "length", "max_tokens_sent": max_output_tokens}], 1: []}
        return dict(agg, _is_full=False, _n_items=2, _embed_backend="transformers",
                    _embed_dtype="float32", _judge_usage=Counter({"gemini-a": 4}),
                    _sample_traces=[C._sample_trace(i, r, turns[i])
                                    for i, r in enumerate([rec, bad])])
    monkeypatch.setattr(C, "_drive_all", fake_drive)
    r = C.run_complexfuncbench("m", "http://x/v1", 1, enable_thinking=thinking, limit=limit)
    return r, seen


def test_cfb_no_think_sends_upstreams_2048_whatever_the_flag_says(monkeypatch):
    r, seen = _cfb_run(monkeypatch, False, 16384)
    assert seen == {"temperature": 0.0, "max_tokens": 2048}
    rec = r["official_sampling"]
    assert rec["max_tokens"] == 2048 and rec["max_tokens_applied"] is True
    assert rec["temperature_matches_official"] is True and "models/qwen.py" in rec["reference"]
    assert r["max_tokens_sent"] == [2048]
    assert "--max-output-tokens 16384 not applied" in r["max_output_tokens_note"]
    assert r["turns_stopped_at_max_tokens"] == 1 and r["model_turns"] == 1
    assert "per-turn max_tokens" not in r["leaderboard_comparable_reason"]


def test_cfb_thinking_run_is_never_capped_below_the_flag(monkeypatch):
    r, seen = _cfb_run(monkeypatch, True, 32768)
    # the maintainer's rule: a --thinking run samples at 1.0, not upstream's 0.0
    assert seen == {"temperature": 1.0, "max_tokens": 32768}
    assert r["official_sampling"]["temperature_matches_official"] is False
    assert "temperature 1.0 set by thinking (upstream sends 0.0)" in r["leaderboard_comparable_reason"]
    rec = r["official_sampling"]
    assert rec["max_tokens_applied"] is False and rec["max_tokens_note"] == base.REASONING_CAP_NOTE
    assert ("per-turn max_tokens 32768 where upstream sends 2048"
            in r["leaderboard_comparable_reason"])


def test_cfb_result_carries_per_sample_records_and_headline_counts(monkeypatch):
    r, _ = _cfb_run(monkeypatch, False, 2048)
    assert (r["total_questions"], r["correct_answers"], r["effective_n"]) == (2, 1, 2)
    assert r["accuracy"] == 50.0
    t0, t1 = r["sample_traces"]
    assert t0["id"] == "Hotels-0" and t0["category"] == "Hotels" and t0["is_correct"] is True
    assert t1["is_correct"] is False and t1["status"] == "OK" and t1["harness_error"] is None
    assert t0["gen_convs"] and t0["resp_judge_calls"] == [{"judge_model": "gemini-a"}]
    assert t0["turns"] == [{"finish_reason": "length", "max_tokens_sent": 2048}]
    assert r["gemini_models_answered"] == {"gemini-a": 4}
    reason = r["leaderboard_comparable_reason"]
    assert "--eval-limit 4: 2 of ComplexFuncBench's 1000 samples" in reason
    assert "ran at float32 where upstream's FlagModel uses fp16" in reason


class _Resp:
    def __init__(self, status, payload=None):
        self.status, self._payload = status, payload

    async def json(self):
        return self._payload

    async def text(self):
        return "busy"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *e):
        return False


class _Session:
    def __init__(self, resp):
        self.resp = resp

    def post(self, url, json=None):
        return self.resp


def test_cfb_endpoint_requests_retry_five_times_twenty_seconds_apart(monkeypatch):
    slept = _no_sleep(monkeypatch, C)
    failures = []
    assert asyncio.run(C._post_chat(_Session(_Resp(503)), "u", {}, failures)) is None
    assert failures == ["HTTP 503"] * 5 and slept == [20.0] * 4


def test_cfb_driver_records_each_turn():
    payload = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
               "usage": {"completion_tokens": 7}}
    turns = []
    msg = asyncio.run(C._post_chat(_Session(_Resp(200, payload)), "u", {"max_tokens": 2048},
                                   [], turns))
    assert msg == {"content": "hi"}
    assert turns == [{"finish_reason": "stop", "completion_tokens": 7, "max_tokens_sent": 2048,
                      "tool_calls": 0, "failed_attempts": 0}]


# =========================================================================== #
# api_bank
# =========================================================================== #
def test_api_bank_prompts_are_evaluator_py_byte_for_byte():
    for name, digest in A.PROMPT_SHA256.items():
        assert _sha(getattr(A, name)) == digest, name
    assert "actual parameters. \nYour output" in A.API_CALL_PROMPT
    assert "in your output. \nThis year" in A.API_CALL_PROMPT
    assert "\nInput: \nUser:" in A.API_CALL_PROMPT and "\nInput: \nUser:" in A.RESPONSE_PROMPT


def test_api_bank_backend_rule():
    def fail():
        return "coroutine error"

    def ok():
        return None
    exc = {"output": None, "exception": "'coroutine' object has no attribute 'text'"}
    err = A._backend_harness_error("Translate", "Translate", exc, fail)
    assert err == {"kind": "backend_unavailable", "api": "Translate", "detail": "coroutine error"}
    # the gold call runs: the model's error
    assert A._backend_harness_error("Translate", "Translate", exc, ok) is None
    # another API, no exception, or an API inside the process: the model's error
    assert A._backend_harness_error("Dictionary", "Translate", exc, fail) is None
    assert A._backend_harness_error("Translate", "Translate",
                                    {"output": "x", "exception": None}, fail) is None
    assert A._backend_harness_error("AddMeeting", "AddMeeting", exc, fail) is None
    # the text Evaluator.evaluate returns when the call raised counts as an exception
    assert A._backend_harness_error("ToolSearcher", "ToolSearcher", "OSError: offline", fail)
    assert A._exec_exception("API Name Mismatch: a vs b") is None


def test_api_bank_proxy_env_is_set_while_scoring_and_restored(monkeypatch):
    monkeypatch.delenv("ALL_PROXY", raising=False)
    with A._proxy_env():
        assert os.environ["ALL_PROXY"] == ""
        os.environ["ALL_PROXY"] = "changed-by-translate"
    assert "ALL_PROXY" not in os.environ
    monkeypatch.setenv("ALL_PROXY", "socks5://p:1")
    with A._proxy_env():
        os.environ["ALL_PROXY"] = ""
    assert os.environ["ALL_PROXY"] == "socks5://p:1"


_FAKE_EVALUATOR = '''
from tool_manager import ToolManager
from api_call_extraction import parse_api_call


class Sample:
    def __init__(self, gt):
        self.ground_truth = gt

    @classmethod
    def from_chat_history(cls, hist):
        return [cls(h) for h in hist]


class Evaluator:
    def __init__(self, dataset):
        self.dataset = dataset

    def evaluate(self, sample_id, model_output):
        tm = ToolManager()
        gt = self.dataset[sample_id].ground_truth
        name, params = parse_api_call(model_output)
        if name != gt["api_name"]:
            return False, "API Name Mismatch: {} vs {}".format(name, gt["api_name"])
        try:
            result = tm.api_call(name, **params)
        except Exception as e:
            return False, str(e)
        return tm.init_tool(name).check_api_call_correctness(result, gt["result"]), result


def calculate_rouge_l_score(ref, hyp):
    return 1.0 if ref == hyp else 0.0
'''

_FAKE_TOOL_MANAGER = '''
import os


class _Tool:
    def check_api_call_correctness(self, res, gt):
        return res["output"] == gt["output"] and res["exception"] == gt["exception"]


class ToolManager:
    def __init__(self, apis_dir="./apis"):
        self.apis_dir = apis_dir

    def api_call(self, name, **params):
        if name in os.environ.get("FAKE_DOWN", "").split(","):
            return {"output": None, "exception": name + " backend unreachable"}
        return {"output": [list(kv) for kv in sorted(params.items())], "exception": None}

    def init_tool(self, name):
        return _Tool()
'''

_FAKE_EXTRACTION = r'''
import re


def get_api_call(text):
    m = re.search(r"\[(\w+)\((.*)\)\]", text)
    return m.group(0) if m else None


def parse_api_call(call):
    m = re.match(r"\[(\w+)\((.*)\)\]", call)
    return m.group(1), dict(re.findall(r"(\w+)='([^']*)'", m.group(2)))
'''


@pytest.fixture
def fake_checkout(tmp_path, monkeypatch):
    d = tmp_path / "api-bank"
    (d / "lv1-lv2-samples" / "level-1-given-desc").mkdir(parents=True)
    (d / "evaluator.py").write_text(_FAKE_EVALUATOR)
    (d / "tool_manager.py").write_text(_FAKE_TOOL_MANAGER)
    (d / "api_call_extraction.py").write_text(_FAKE_EXTRACTION)
    steps = [{"api_name": "Translate", "param_dict": {"src": "Hello", "tgt_lang": "fr"}},
             {"api_name": "Dictionary", "param_dict": {"keyword": "apple"}}]
    with open(d / "lv1-lv2-samples" / "level-1-given-desc" / "T.jsonl", "w") as f:
        for st in steps:
            st["result"] = {"output": sorted(st["param_dict"].items()), "exception": None}
            f.write(json.dumps(st) + "\n")
    for mod in ("evaluator", "tool_manager", "api_call_extraction"):
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield str(d)
    for mod in ("evaluator", "tool_manager", "api_call_extraction"):
        sys.modules.pop(mod, None)


def _l1(sid, text, status="OK"):
    return {"gold_answer": json.dumps({"level": 1, "track": "api", "file": "T.jsonl",
                                       "sample_id": sid}),
            "response_text": text, "status": status}


def test_api_bank_backend_failures_and_failed_requests_are_harness_errors(fake_checkout,
                                                                         monkeypatch):
    monkeypatch.setenv("FAKE_DOWN", "Translate")
    monkeypatch.delenv("ALL_PROXY", raising=False)
    tr_ok = _l1(0, "[Translate(src='Hello', tgt_lang='fr')]")      # right call, backend down
    tr_other = _l1(0, "[Dictionary(keyword='Hello')]")             # wrong API: model's error
    dic_wrong = _l1(1, "[Dictionary(keyword='pear')]")             # backend up: model's error
    dic_right = _l1(1, "[Dictionary(keyword='apple')]")
    failed = {**_l1(1, None, status="TIMEOUT"), "error": "timeout after 600s"}
    A._score_traces(fake_checkout, [tr_ok, tr_other, dic_wrong, dic_right, failed])
    assert tr_ok["harness_error"]["kind"] == "backend_unavailable"
    assert tr_ok["harness_error"]["api"] == "Translate" and tr_ok["scoring_excluded"] is True
    assert tr_ok["api_bank_score"] is None and tr_ok["status"] == "HARNESS_ERROR"
    assert "harness_error" not in tr_other and tr_other["api_bank_score"] == 0.0
    assert "API Name Mismatch" in tr_other["api_bank_detail"]
    assert "harness_error" not in dic_wrong and dic_wrong["is_correct"] is False
    assert dic_right["is_correct"] is True and dic_right["api_bank_score"] == 1.0
    assert failed["harness_error"]["kind"] == "endpoint_timeout"
    assert "ALL_PROXY" not in os.environ                          # restored after scoring


def test_api_bank_l3_tool_searcher_outage_is_a_harness_error(fake_checkout, monkeypatch):
    monkeypatch.setenv("FAKE_DOWN", "ToolSearcher")
    gold = [{"apis": [{"api_name": "ToolSearcher", "input": {"keywords": "AddMeeting"},
                       "output": {"output": "x", "exception": None}},
                      {"api_name": "AddMeeting", "input": {"a": "1"},
                       "output": {"output": [("a", "1")], "exception": None}}],
             "response": "done"}]
    monkeypatch.setattr(A, "_hf", lambda fn: gold)

    def l3(aid, text):
        return {"gold_answer": json.dumps({"level": 3, "track": "api", "sample_id": 0,
                                           "api_id": aid}),
                "response_text": text, "status": "OK"}
    ts, meet = l3(0, "[ToolSearcher(keywords='AddMeeting')]"), l3(1, "[AddMeeting(a='2')]")
    errored = A._score_traces(fake_checkout, [ts, meet])
    assert ts["harness_error"]["api"] == "ToolSearcher" and ts["scoring_excluded"] is True
    assert meet["is_correct"] is False and "harness_error" not in meet
    assert errored == {0}
    assert A._l3_dialogues([ts, meet], errored) == ({0}, set())
    assert A._l3_dialogues([ts], set()) == (set(), {0})


def _api_bank_run(monkeypatch, marks, limit=None):
    """run_api_bank over stubbed steps; `marks` maps each step (its content) to its outcome."""
    samples = []
    for key in marks:
        level, track = int(key[1]), key.split("_")[1][:-1]
        gold = json.dumps({"level": level, "track": track, "sample_id": int(key[-1])})
        samples.append(([{"role": "user", "content": key}], gold,
                        {"category": f"api_bank::L{level}_{track}"}))
    monkeypatch.setattr(A, "_checkout_dir", lambda: "unused")
    monkeypatch.setattr(A, "_load_l12_samples",
                        lambda d: [s for s in samples if json.loads(s[1])["level"] < 3])
    monkeypatch.setattr(A, "_load_l3_samples",
                        lambda: [s for s in samples if json.loads(s[1])["level"] == 3])

    def fake_score(d, traces):
        errored = set()
        for t in traces:
            g = json.loads(t["gold_answer"])
            t["api_bank_level"], t["api_bank_track"] = g["level"], g["track"]
            t["api_bank_sample_id"] = g["sample_id"]
            how = marks[t["messages"][0]["content"]]
            if how == "harness":
                A._mark_harness_error(t, {"kind": "backend_unavailable", "api": "Translate",
                                          "detail": "down"})
            else:
                t["is_correct"] = how == "ok"
                t["api_bank_score"] = 1.0 if how == "ok" else 0.0
                if g["level"] == 3 and how != "ok":
                    errored.add(g["sample_id"])
        return errored
    monkeypatch.setattr(A, "_score_traces", fake_score)

    async def send(**kw):
        return base.Reply("[Api()]", None, "stop", completion_tokens=3)
    with mock.patch.object(base, "_send_single_request", send):
        return A.run_api_bank("m", "http://x", 1, limit=limit)


def test_api_bank_harness_errors_leave_the_level_means_and_the_counts(monkeypatch):
    r = _api_bank_run(monkeypatch, {"L1_api0": "ok", "L1_api1": "harness",
                                    "L1_response0": "ok", "L3_api0": "ok", "L3_api1": "bad"})
    assert r["dimension_scores"]["L1_call"] == 100.0 and r["dimension_scores"]["L3_call"] == 50.0
    assert r["accuracy"] == 75.0
    assert (r["total_questions"], r["effective_n"], r["correct_answers"]) == (4, 3, 2)
    assert r["response_track_samples"] == 1
    assert r["harness_errors"] == {"backend_unavailable": {"Translate": 1}}
    assert r["status"] == "completed_with_errors"
    assert "harness errors, not the model's outcome" in r["leaderboard_comparable_reason"]
    assert r["category_accuracy"]["api_bank::L1_api"] == {"correct": 1, "total": 1,
                                                          "accuracy": 100.0}


def test_api_bank_l3_sample_success_needs_every_l3_step(monkeypatch):
    # every L3 call step of the (stubbed) set was run: lv3_evaluator's formula over the 50
    r = _api_bank_run(monkeypatch, {"L1_api0": "ok", "L3_api0": "ok", "L3_api1": "bad"})
    assert r["dimension_scores"]["L3_sample_success"] == 98.0
    # a subset that did not run them all: not computed (the /50 counted unscored dialogues as
    # successes)
    r = _api_bank_run(monkeypatch, {"L1_api0": "ok", "L1_api1": "ok", "L1_api2": "ok",
                                    "L3_api0": "bad", "L3_api1": "bad"}, limit=3)
    assert r["dimension_scores"]["L3_sample_success"] is None
    assert "needs every one of the 2 Level-3 call steps" in r["L3_sample_success_note"]


def test_api_bank_withholds_the_score_when_no_call_step_was_scored(monkeypatch):
    r = _api_bank_run(monkeypatch, {"L1_api0": "harness", "L1_response0": "ok"})
    assert r["accuracy"] is None and r["status"] == "failed"
    assert "every one of the 1 was a harness error" in r["accuracy_withheld_reason"]


# =========================================================================== #
# gaia / deepsearch_qa: the web_search tool, transcripts, harness errors
# =========================================================================== #
def _search_reply(backend="gemini:gemini-3.8-flash", title="Page"):
    return json.dumps({"query": "q", "results": [{"title": title, "url": "u", "snippet": "s"}],
                       "backend": backend})


def test_the_tool_says_its_results_are_gemini_written():
    fn = G.GROUNDED_SEARCH_TOOL["function"]
    assert fn["name"] == "web_search"
    assert fn["parameters"] == search_tool.WEB_SEARCH_TOOL["function"]["parameters"]
    assert "Gemini model grounded in Google Search" in fn["description"]
    assert "not quoted from the page" in fn["description"]
    assert "snippets" not in fn["description"]
    for doc in ("gaia", "deepsearch_qa"):
        text = (ROOT / "docs" / "evals" / f"{doc}.md").read_text(encoding="utf-8")
        assert "written by Gemini" in text and "`tool_transcript`" in text
        assert text.isascii()


def test_transcript_entry_flags_search_backend_failures():
    ok = G.transcript_entry("web_search", {"query": "q"}, _search_reply())
    assert ok["backend"] == "gemini:gemini-3.8-flash" and ok["search_failed"] is False
    for backend in G.FAILED_SEARCH_BACKENDS:
        entry = G.transcript_entry("web_search", {}, _search_reply(backend, "error"))
        assert entry["search_failed"] is True
    # a model error (an empty query) is not a backend failure
    entry = G.transcript_entry("web_search", {}, json.dumps({"error": "empty"}))
    assert entry["search_failed"] is False


def test_record_web_search_keeps_each_call_on_its_sample(monkeypatch):
    replies = iter([_search_reply(), _search_reply("all-grounding-models-failed", "error")])

    async def fake_execute(name, args):
        return next(replies)
    monkeypatch.setattr(G, "execute_tool", fake_execute)
    G.reset_tool_transcripts()
    call = {"id": "c1", "type": "function",
            "function": {"name": "web_search", "arguments": "{\"query\": \"x\"}"}}
    queue = {0: [base.Reply("", [call], "tool_calls", completion_tokens=3)],
             1: [base.Reply("", [call], "tool_calls", completion_tokens=3)]}

    async def send(**k):
        async with k["semaphore"]:
            pass
        q = queue[int(k["messages"][0]["content"])]
        return q.pop(0) if q else base.Reply("FINAL ANSWER: 3", None, "stop", completion_tokens=5)
    samples = [([{"role": "user", "content": str(i)}], "3", {"category": "Level_1"})
               for i in range(2)]
    with mock.patch.object(base, "_send_single_request", send):
        res = base.run_eval_suite(eval_name="gaia", model_name="m", base_url="http://x",
                                  concurrency=1, samples=samples, eval_fn=lambda r, g: True,
                                  tool_executor=G.record_web_search,
                                  tool_loop=G.GAIA_TOOL_LOOP)
    traces = res["sample_traces"]
    G.attach_tool_transcripts(traces)
    assert [t["web_search_calls"] for t in traces] == [1, 1]
    assert {t["tool_transcript"][0]["search_failed"] for t in traces} == {False, True}
    assert all(t["tool_transcript"][0]["arguments"] == {"query": "x"} for t in traces)


def test_web_research_harness_error_kinds():
    f = G.web_research_harness_error
    assert f({"status": "FAILED", "response_text": None, "error": "HTTP 400"})["kind"] == \
        "endpoint_failure"
    assert f({"status": "TIMEOUT"})["kind"] == "endpoint_timeout"
    assert f({"status": "OVER_CONTEXT"})["kind"] == "over_context"
    assert f({"status": "OK", "response_text": "pre",
              "tool_loop_end": "request_failed"})["kind"] == "request_failed_mid_loop"
    assert f({"status": "OK", "response_text": "a", "tool_transcript": [
        {"search_failed": True, "backend": "deadline-passed"}]})["kind"] == \
        "search_backend_failure"
    assert f({"status": "OK", "response_text": "a",
              "tool_transcript": [{"search_failed": False}]}) is None
    assert f({"status": "EMPTY_RESPONSE", "response_text": ""}) is None   # the model's outcome


def _gaia_run(monkeypatch, traces, transcripts):
    monkeypatch.setattr(G, "gaia_split", lambda: "validation")
    monkeypatch.setattr(G, "check_gaia_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(G, "search_available", lambda: True)
    monkeypatch.setattr(G, "gemini_key_live_valid", lambda key: (True, ""))
    monkeypatch.setattr(G, "_load_gaia_samples", lambda limit=None, split=None: [])
    seen = {}

    def fake(**kw):
        seen.update(kw)
        G._TOOL_TRANSCRIPTS.update(transcripts)
        correct = sum(1 for t in traces if t.get("is_correct"))
        return {"sample_traces": [dict(t) for t in traces], "status": "success",
                "accuracy": round(100.0 * correct / len(traces), 2), "correct_answers": correct}
    monkeypatch.setattr(G, "run_eval_suite", fake)
    return G.run_gaia("m", "http://x", 1), seen


def test_gaia_harness_errors_leave_the_accuracy(monkeypatch):
    ok = {"sample_idx": 0, "status": "OK", "response_text": "FINAL ANSWER: 1",
          "is_correct": True, "category": "Level_1", "extra_payload": {"has_file": False}}
    wrong = dict(ok, sample_idx=1, is_correct=False)
    searched = dict(ok, sample_idx=2)                  # right, but a lookup went unanswered
    failed = dict(ok, sample_idx=3, status="FAILED", response_text=None, is_correct=False,
                  error="HTTP 400: image input not supported",
                  extra_payload={"has_file": True, "attachment_type": "png"})
    bad = G.transcript_entry("web_search", {"query": "q"},
                             _search_reply("all-grounding-models-failed", "error"))
    r, seen = _gaia_run(monkeypatch, [ok, wrong, searched, failed], {2: [bad]})
    assert seen["tool_executor"] is G.record_web_search
    assert seen["extra_payload"]["tools"] == [G.GROUNDED_SEARCH_TOOL]
    assert r["accuracy"] == 50.0 and r["correct_answers"] == 1 and r["effective_n"] == 2
    assert r["harness_errors"] == {"endpoint_failure": 1, "search_backend_failure": 1}
    assert r["status"] == "completed_with_errors"
    assert r["category_accuracy"] == {"Level_1": {"correct": 1, "total": 2, "accuracy": 50.0}}
    assert "accuracy_with_file" not in r and r["file_dependent_n"] == 0
    traces = {t["sample_idx"]: t for t in r["sample_traces"]}
    assert traces[2]["tool_transcript"] == [bad] and traces[2]["scoring_excluded"] is True
    reason = r["leaderboard_comparable_reason"]
    assert "results are Gemini-written" in reason and "(search_backend_failure)" in reason
    assert r["search_tool_description"] == G.GROUNDED_SEARCH_DESCRIPTION


def test_gaia_withholds_the_score_when_every_sample_is_a_harness_error(monkeypatch):
    failed = {"sample_idx": 0, "status": "TIMEOUT", "response_text": None, "is_correct": False,
              "category": "Level_1", "extra_payload": {}}
    r, _ = _gaia_run(monkeypatch, [failed], {})
    assert r["accuracy"] is None and r["status"] == "failed"
    assert "every sample ended in a harness error" in r["accuracy_withheld_reason"]


def _dsqa(resp, status="PENDING_JUDGE", idx=0, **extra):
    return {"sample_idx": idx, "response_text": resp, "status": status, "health": "ok",
            "gold_answer": "A", "messages": [{"role": "user", "content": "Which?"}],
            "extra_payload": {"category": "Science", "answer_type": "Single Answer"}, **extra}


def test_dsqa_harness_errors_are_not_rated_and_not_empty_responses(monkeypatch):
    prompts = []
    rating = ('{"Answer Correctness": {"Explanation": "matches", "Correctness Details": '
              '{"A": true}, "Excessive Answers": []}}')

    async def judge(prompt, **kw):
        prompts.append(prompt)
        return rating, "gemini-3.8-flash"
    monkeypatch.setattr(D, "judge_generate_cascade", judge)
    G.reset_tool_transcripts()
    G._TOOL_TRANSCRIPTS[2] = [G.transcript_entry(
        "web_search", {"query": "q"}, _search_reply("all-grounding-models-failed", "error"))]
    traces = [_dsqa("A"), _dsqa(None, status="FAILED", idx=1, error="HTTP 503"),
              _dsqa("A", idx=2),
              _dsqa("preamble", idx=3, tool_loop_end="request_failed"),
              _dsqa("", status="EMPTY_RESPONSE", idx=4)]
    asyncio.run(D._async_judge_deepsearch_qa(traces))
    assert len(prompts) == 1                           # only the answered sample is rated
    assert traces[0]["status"] == "OK" and traces[0]["judge_model"] == "gemini-3.8-flash"
    assert traces[0]["judge_raw"] == rating and traces[0]["judge_explanation"] == "matches"
    kinds = [(t.get("harness_error") or {}).get("kind") for t in traces]
    assert kinds == [None, "endpoint_failure", "search_backend_failure",
                     "request_failed_mid_loop", None]
    assert all(t["judge_grade"] == D.HARNESS_ERROR_GRADE for t in traces[1:4])
    s = D.summarize(traces)
    assert s["num_rated"] == 2 and s["num_empty_model_response"] == 1
    assert s["num_harness_error"] == 3


def test_dsqa_gate_reads_a_key_list_set_only_as_gemini_api_keys(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEYS", "k1,k2")
    seen = []

    def gate(k):
        seen.append(k)
        return False, base.MODELS_NOT_SERVED_PREFIX + " 'x'"
    monkeypatch.setattr(D, "gemini_key_live_valid", gate)
    with pytest.raises(RuntimeError, match="every model of the Gemini cascade"):
        D.run_deepsearch_qa("m", "http://x", limit=1)
    assert seen == ["k1,k2"]


def test_dsqa_withholds_the_score_when_nothing_was_rated(monkeypatch):
    monkeypatch.setattr(D, "search_available", lambda: True)
    monkeypatch.setattr(D, "gemini_key_live_valid", lambda k: (True, ""))
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(D, "_load_deepsearch_qa_samples", lambda limit=None: [])

    def fake(**kw):
        traces = [_dsqa(None, status="FAILED")]
        asyncio.run(kw["async_eval_fn"](traces))
        return {"sample_traces": traces, "status": "failed", "accuracy": 0.0}
    monkeypatch.setattr(D, "run_eval_suite", fake)
    r = D.run_deepsearch_qa("m", "http://x")
    assert r["accuracy"] is None and r["status"] == "failed"
    assert r["harness_errors"] == {"endpoint_failure": 1}
    assert "num_harness_error 1" in r["accuracy_withheld_reason"]


# =========================================================================== #
# aa_lcr
# =========================================================================== #
def test_aa_lcr_harness_errors_are_excluded_and_judge_replies_kept(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    replies = iter(['{"verdict": "CORRECT"}', "no verdict", "still none", "nope"])

    async def fake(contents, *, config=None):
        return next(replies), "gemini-3.7-flash"
    monkeypatch.setattr(aa_lcr, "judge_generate_cascade", fake)
    ok = {"status": "PENDING_JUDGE", "response_text": "42", "gold_answer": "42",
          "extra_payload": {"question": "q"}}
    no_verdict = dict(ok)
    failed = {"status": "TIMEOUT", "response_text": None, "gold_answer": "1",
              "error": "timeout", "extra_payload": {"question": "q"}}
    empty = {"status": "EMPTY_RESPONSE", "response_text": "", "gold_answer": "1",
             "extra_payload": {"question": "q"}}
    asyncio.run(aa_lcr._async_judge_aa_lcr([ok, no_verdict, failed, empty], judge_concurrency=1))
    assert ok["judge_grade"] == "CORRECT" and ok["judge_replies"] == [
        {"judge_model": "gemini-3.7-flash", "reply": '{"verdict": "CORRECT"}'}]
    assert no_verdict["judge_grade"] == "JUDGE_FAILED" and no_verdict["scoring_excluded"] is True
    assert len(no_verdict["judge_replies"]) == aa_lcr._JUDGE_ATTEMPTS
    assert failed["judge_grade"] == aa_lcr.HARNESS_ERROR_GRADE
    assert failed["harness_error"]["kind"] == "endpoint_timeout" and failed["scoring_excluded"]
    assert empty["judge_grade"] == "EMPTY_RESPONSE" and not empty.get("scoring_excluded")


def test_aa_lcr_run_leaves_harness_errors_out_and_withholds_an_unscored_run(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    samples = [([{"role": "user", "content": "p1"}], "A", {"category": "c", "question": "q1"}),
               ([{"role": "user", "content": "p2"}], "B", {"category": "c", "question": "q2"})]

    async def send(**kw):
        if kw["messages"][0]["content"] == "p2":
            return base.Reply(text=None, tool_calls=None, error="HTTP 503")
        return base.Reply(text="A", tool_calls=None, finish_reason="stop")

    async def fake(contents, *, config=None):
        return '{"verdict": "CORRECT"}', "gemini-e2e"
    with mock.patch.object(base, "_send_single_request", side_effect=send), \
            mock.patch.object(aa_lcr, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(aa_lcr, "_load_aa_lcr_samples", lambda **k: samples), \
            mock.patch.object(aa_lcr, "judge_generate_cascade", fake):
        r = aa_lcr.run_aa_lcr("m", "http://x/v1")
    # 3 attempts each: the 3 failed attempts of p2 leave the denominator (never INCORRECT)
    assert r["accuracy"] == 100.0 and r["harness_errors"] == {"endpoint_failure": 3}
    assert r["status"] == "completed_with_errors"
    assert "harness errors, not the model's outcome" in r["leaderboard_comparable_reason"]

    async def always_fail(**kw):
        return base.Reply(text=None, tool_calls=None, error="HTTP 503")
    with mock.patch.object(base, "_send_single_request", side_effect=always_fail), \
            mock.patch.object(aa_lcr, "gemini_required_skip", lambda *a: None), \
            mock.patch.object(aa_lcr, "_load_aa_lcr_samples", lambda **k: samples[:1]), \
            mock.patch.object(aa_lcr, "judge_generate_cascade", fake):
        r = aa_lcr.run_aa_lcr("m", "http://x/v1")
    assert r["accuracy"] is None and r["status"] == "failed"
    assert "endpoint_failure 3" in r["accuracy_withheld_reason"]


# =========================================================================== #
# beam_128k
# =========================================================================== #
def _turn(q):
    return {"role": "user", "content": beam_128k.FINAL_TURN.format(query=q)}


def test_beam_question_tail_reproduces_upstreams_saved_list_aliasing():
    qs = ["q0", "q1", "q2", "q3"]
    assert beam_128k._question_tail(qs, 0) == [_turn("q0")]
    assert beam_128k._question_tail(qs, 1) == [_turn("q1")]
    assert beam_128k._question_tail(qs, 2) == [_turn("q1"), _turn("q2")]
    assert beam_128k._question_tail(qs, 3) == [_turn("q1"), _turn("q2"), _turn("q3")]


def test_beam_samples_carry_the_earlier_question_turns(tmp_path):
    import pandas as pd
    probing = {a: [{"question": f"{a} q{i}", "rubric": ["item"], "answer": "x"} for i in range(2)]
               for a in beam_128k.ABILITIES}
    chat = [[{"role": "user", "content": f"u{k}"}, {"role": "assistant", "content": f"a{k}"}]
            for k in range(3)]
    path = os.path.join(str(tmp_path), "beam.parquet")
    pd.DataFrame([{"conversation_id": "0", "chat": chat,
                   "probing_questions": repr(probing)}]).to_parquet(path)
    order = [q["question"] for a in beam_128k.ABILITIES for q in probing[a]]
    with mock.patch("huggingface_hub.hf_hub_download", side_effect=lambda *a, **k: path):
        samples = beam_128k._load_beam_128k_samples()
    by_q = {meta["question"]: (messages, meta) for messages, _g, meta in samples}
    for pos, q in enumerate(order):
        messages, meta = by_q[q]
        assert meta["question_position"] == pos
        assert meta["carried_question_turns"] == max(0, pos - 1)
        assert messages[:6] == [m for pair in chat for m in pair]
        assert messages[6:] == beam_128k._question_tail(order, pos)


def test_beam_judge_and_split_see_the_reply_unstripped(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    prompts = []

    async def fake(contents, *, config=None):
        if getattr(config, "system_instruction", None):
            return "NO", "gemini-e"
        prompts.append(contents)
        return '{"score": 1.0, "reason": "ok"}', "gemini-e"
    monkeypatch.setattr(beam_128k, "judge_generate_cascade", fake)
    t = {"category": "event_ordering", "response_text": "<thought>x</thought>A\nB\n",
         "status": "PENDING_JUDGE", "gold_answer": json.dumps({"ideal": "", "rubric": ["E1"]}),
         "extra_payload": {}}
    asyncio.run(beam_128k._make_scorer(Counter(), judge_concurrency=1)([t]))
    assert prompts and "RESPONSE TO EVALUATE: A\nB\n\n" in prompts[0]
    # three lines, the trailing empty one included, as upstream's llm_response.split("\n")
    assert [c["line"] for c in t["beam_alignment_calls"]] == ["A", "B", ""]
    assert t["beam_judge_calls"] == [{"rubric_item": "E1", "replies": [
        {"judge_model": "gemini-e", "reply": '{"score": 1.0, "reason": "ok"}'}],
        "outcome": "OK", "score": 1.0, "judge_model": "gemini-e", "reason": "ok"}]
    assert beam_128k._answer_text("  x \n") == "  x \n"


def test_beam_withholds_the_score_when_no_question_was_scored():
    failed = {"category": "abstention", "status": "FAILED", "response_text": None,
              "judge_grade": beam_128k.HARNESS_ERROR_GRADE, "scoring_excluded": True,
              "harness_error": {"kind": "endpoint_failure", "detail": "x"}}
    r = beam_128k._finalize({"accuracy": 0.0, "sample_traces": [failed], "status": "failed"})
    assert r["accuracy"] is None and r["status"] == "failed"
    assert "endpoint_failure 1" in r["accuracy_withheld_reason"]


def test_beam_docs_no_longer_misstate_upstreams_failed_request_path():
    text = " ".join(beam_128k.__doc__.split())
    doc = " ".join((ROOT / "docs" / "evals" / "beam_128k.md").read_text().split())
    for t in (text, doc):
        assert "its model call raises" not in t
        assert "Error in <ability> question type" in t
    assert "question_position" in doc and "carried_question_turns" in doc


# =========================================================================== #
# acebench: a subset's headline and its ceiling
# =========================================================================== #
def test_acebench_draw_ceiling_matches_the_nine_item_example():
    drawn = ["special_incomplete", "special_error_param", "normal_single_turn_single_function",
             "normal_atom_bool"]
    not_run, ceiling = AB.draw_ceiling({s: {"accuracy": 1.0} for s in drawn})
    assert ceiling == 0.275
    assert "agent_multi_step" in not_run and "normal_atom_bool" not in not_run
    full = AB.NORMAL_CATEGORIES + AB.SPECIAL_CATEGORIES + AB.AGENT_CATEGORIES
    assert AB.draw_ceiling({s: {"accuracy": 1.0} for s in full}) == ([], 1.0)
    assert AB.draw_ceiling({"agent_multi_turn": {"status": "infra_required"}})[1] is None


def test_acebench_subset_reason_names_the_undrawn_categories_and_the_ceiling(monkeypatch):
    row = {"id": "agent_multi_step_0", "question": "q", "initial_config": {},
           "function": [], "involved_classes": ["BaseApi"]}

    def samples(language="en", **k):
        gold = json.dumps({"id": "special_irrelevant_0", "sub": "special_irrelevant",
                           "language": language, "ground_truth": "x", "functions": [],
                           "question": "q"})
        return [([{"role": "user", "content": language}], gold,
                 {"category": AB._category_key("special_irrelevant", language)})]
    monkeypatch.setattr(AB, "_load_acebench_samples", samples)
    monkeypatch.setattr(AB, "_load_agent_items", lambda language="en": [
        {"language": language, "category": "agent_multi_step",
         "key": AB._category_key("agent_multi_step", language), "row": row,
         "answer": {"ground_truth": [], "mile_stone": []}}])
    monkeypatch.setattr(AB, "check_user_simulator", lambda: True)

    async def conversation(shared, item):
        return {"category": item["key"], "acebench_language": item["language"],
                "question_id": item["row"]["id"], "attempt_idx": 0, "status": "OK",
                "gold_answer": json.dumps({"sub": item["category"]}),
                "execution_guard_refusals": [], "model_requests": 1, "truncated_replies": 0,
                "user_sim_requests": 0, "user_sim_models": {}, "is_correct": True,
                "process_accuracy": 1.0, "response_text": "done"}
    monkeypatch.setattr(AB, "_run_conversation", conversation)

    async def reply(**kw):
        return base.Reply("Due to the limitations of the function, I cannot solve this problem.",
                          None, "stop", completion_tokens=5)
    monkeypatch.setattr(base, "_send_single_request", reply)
    r = AB.run_acebench("m", "http://x", 1, limit=4)
    en = r["subcategories_not_run"]["en"]
    assert "special_irrelevant" not in en and "agent_multi_turn" in en
    assert r["summary_ceiling"]["en"] == r["per_language"]["en"]["summary_ceiling"]
    reason = r["leaderboard_comparable_reason"]
    assert "the English Summary (the headline) counts the" in reason
    assert f"would score {r['summary_ceiling']['en']}% (summary_ceiling)" in reason
    assert r["leaderboard_comparable"] is False


def test_acebench_user_simulator_records_answers_and_skips_unserved_models(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gone,live")
    asked = []

    class _NotFound(Exception):
        code = 404

    class _Resp:
        text = "hello"
        candidates = []

    class _Models:
        async def generate_content(self, model, contents, config):
            asked.append(model)
            if model == "gone":
                raise _NotFound("404")
            return _Resp()

    class _Aio:
        models = _Models()

    class _Client:
        aio = _Aio()
    monkeypatch.setattr(base, "_judge_clients", lambda: [_Client()])
    monkeypatch.setattr(base, "_next_judge_client", lambda: _Client())
    before = base.gemini_answer_counts().get("live", 0)
    assert asyncio.run(AB._user_sim_cascade("hi")) == ("hello", "live")
    assert asked == ["gone", "live"]
    assert base.gemini_answer_counts().get("live", 0) == before + 1
