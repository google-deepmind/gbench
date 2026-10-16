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

"""Tool-use suites: canonical non-model tool backends run as upstream, harness errors are never
the model's outcome.

* complexfuncbench runs upstream's response_based call-match tier against the live booking-com15
  API on RapidAPI (RAPID_API_KEY; infra_required without it or when the API refuses it), with
  upstream's request, retries and verdict; a pair the backend could not answer that the Gemini
  judge then finds not equivalent is a harness error (rapidapi_unavailable).
* api_bank's Translate API runs upstream's pinned googletrans==3.1.0a0 in a local image, with
  raise_exception=True so a Google reply other than HTTP 200 is an exception, the proxy
  variables read when a Translator is built and passed to the container, and checks before the
  run.
* bfcl: a multi-turn entry whose checker raised is a harness error (checker_exception), left out
  of the score and named.

Docker, the RapidAPI, Google Translate, the Gemini judge and the model endpoint are stubbed: no
test contacts a network service or a model, and nothing is written outside tmp_path.
"""

import asyncio
import json
import subprocess
import sys
import types
from collections import Counter
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import api_bank as A
from gbench.runners.eval_suites import base, bfcl
from gbench.runners.eval_suites import complexfuncbench as C

ROOT = Path(__file__).resolve().parent.parent
RAPID_SECRET = "rapid-secret-key-0123456789abcdef"


# =========================================================================== #
# complexfuncbench: the response-based tier
# =========================================================================== #
class _Resp:
    def __init__(self, status, body=None, raw=None):
        self.status_code = status
        self._body = body
        self._raw = raw

    def json(self):
        if self._raw is not None:
            return json.loads(self._raw)
        return json.loads(json.dumps(self._body))


@pytest.fixture
def no_sleep(monkeypatch):
    async def _sleep(_s):
        return None
    monkeypatch.setattr(C.asyncio, "sleep", _sleep)
    monkeypatch.setattr(C.time, "sleep", lambda _s: None)


@pytest.fixture
def rapid(monkeypatch, no_sleep):
    """C._rapidapi_get stubbed: `answers` maps a function name to a list of replies (a _Resp or
    an exception instance), used in order and the last one repeated; every request is kept."""
    state = {"answers": {}, "seen": []}

    def fake_get(url, headers, params, timeout):
        name = next(n for n, u in C._load_tool_info()["name_to_url"].items() if u == url)
        state["seen"].append({"name": name, "url": url, "headers": dict(headers),
                              "params": dict(params), "timeout": timeout})
        seq = state["answers"][name]
        idx = sum(1 for s in state["seen"] if s["name"] == name) - 1
        ans = seq[min(idx, len(seq) - 1)]
        if isinstance(ans, BaseException):
            raise ans
        if callable(ans):
            return ans(params)
        return ans
    monkeypatch.setattr(C, "_rapidapi_get", fake_get)
    return state


def _api(stats=None):
    return C._RapidAPICall(C._load_tool_info(), RAPID_SECRET, stats if stats is not None
                           else Counter())


def test_the_vendored_tool_info_is_upstreams_and_the_tier_is_on_the_path():
    assert C._sha256(C._TOOL_INFO_PATH) == C.TOOL_INFO_SHA256
    info = C._load_tool_info()
    assert info["host"] == "booking-com15.p.rapidapi.com" and info["path_params"] == []
    assert C.CALL_MATCH_TIERS == ["rule_based", "value_checker", "response_based", "llm_based"]
    assert C.RESPONSE_BASED_TIER["run"] is True
    assert "rapidapi_unavailable" in C.HARNESS_ERROR_KINDS


def test_a_rapidapi_call_is_sent_and_read_as_upstreams(rapid):
    ok = {"status": True, "message": "Success", "timestamp": 1700000000000,
          "data": [{"hotel_id": 1}]}
    rapid["answers"] = {"Search_Flights_Multi_Stops": [_Resp(200, ok)],
                        "Search_Attractions": [_Resp(200, {"status": False, "message": "bad",
                                                           "timestamp": 1})]}
    api = _api()
    legs = [{"fromId": "ORD.AIRPORT", "toId": "HND.AIRPORT", "date": "2024-09-01"}]
    out = asyncio.run(api.call({"name": "Search_Flights_Multi_Stops",
                                "arguments": {"legs": legs, "adults": 2}}))
    # status true: timestamp dropped, data returned
    assert out == {"response": [{"hotel_id": 1}], "raised": False, "labels": ["ok"]}
    sent = rapid["seen"][0]
    assert sent["url"] == ("https://booking-com15.p.rapidapi.com/api/v1/flights/"
                           "searchFlightsMultiStops")
    assert sent["headers"] == {"X-RapidAPI-Key": RAPID_SECRET,
                               "X-RapidAPI-Host": "booking-com15.p.rapidapi.com"}
    assert sent["params"] == {"legs": json.dumps(legs, ensure_ascii=False), "adults": 2}
    assert sent["timeout"] == C._RAPIDAPI_TIMEOUT_S
    # status false: the whole reply, timestamp kept (upstream pops it only on status true)
    out = asyncio.run(api.call({"name": "Search_Attractions", "arguments": {"id": "x"}}))
    assert out["response"] == {"status": False, "message": "bad", "timestamp": 1}


def test_a_call_without_a_response_is_retried_three_times_and_counted(rapid):
    import requests
    rapid["answers"] = {"Search_Attractions": [_Resp(429), requests.ConnectionError("down"),
                                               _Resp(503)],
                        "Get_Attraction_Details": [_Resp(502), _Resp(200, {"status": True,
                                                                           "data": {"a": 1}})]}
    stats = Counter()
    api = _api(stats)
    out = asyncio.run(api.call({"name": "Search_Attractions", "arguments": {"id": "x"}}))
    assert out == {"response": None, "raised": False,
                   "labels": ["HTTP 429", "ConnectionError", "HTTP 503"]}
    out = asyncio.run(api.call({"name": "Get_Attraction_Details", "arguments": {"slug": "s"}}))
    assert out["response"] == {"a": 1} and out["labels"] == ["HTTP 502", "ok"]
    assert stats["calls"] == 2 and stats["attempts"] == 5
    assert stats["calls_without_response"] == 1 and stats["no_response:HTTP 503"] == 1


def test_an_unreadable_reply_raises_at_once_as_upstreams(rapid):
    rapid["answers"] = {"Search_Attractions": [_Resp(200, raw="<html>proxy error</html>")]}
    out = asyncio.run(_api().call({"name": "Search_Attractions", "arguments": {"id": "x"}}))
    assert out["raised"] is True and out["response"] is None
    # booking-com15 always answers JSON: an HTML 200 is a proxy's page, an outage
    assert out["labels"] == [C.UNREADABLE_BODY] and len(rapid["seen"]) == 1
    assert C._call_state(out) == "outage"


def _pair(api, pred, gold):
    cmp = C._CompareFC(C._load_exact_match_dict(), C._get_embedder(), Counter(), api)
    verdict = asyncio.run(cmp.response_based(pred, gold))
    return verdict, cmp


def _data(d):
    return _Resp(200, {"status": True, "timestamp": 5, "data": d})


PRED = {"name": "Search_Attractions", "arguments": {"id": "eyJ1ZmkiOi0yOTA2OTJ9",
                                                    "sortBy": "trending"}}
GOLD = {"name": "Search_Attractions", "arguments": {"id": "eyJ1ZmkiOi0yOTA2OTJ9"}}


def test_equal_live_responses_match_the_pair(rapid):
    # a default-valued optional parameter: the live API answers the same for both calls
    rapid["answers"] = {"Search_Attractions": [_data({"products": [1, 2]})]}
    verdict, cmp = _pair(_api(), PRED, GOLD)
    assert verdict is True and not cmp.response_unavailable
    assert cmp.response_calls[0]["outcome"] == "equal"
    assert [s["params"] for s in rapid["seen"]] == [PRED["arguments"], GOLD["arguments"]]


@pytest.mark.parametrize("pred_reply, gold_reply, outcome, unavailable, n_calls", [
    (_data({"p": 1}), _data({"p": 2}), "different", False, 2),
    # the backend answered the golden call: the predicted call's own 400 is the model's call
    (_Resp(400), _data({"p": 2}), "no_response", False, 4),
    # a 200 body that is not JSON is an outage of the backend (a proxy's or portal's page)
    (_Resp(200, raw="not json"), _data({"p": 2}), "unavailable", True, 1),
    (_data({"p": 1}), _Resp(200, raw="not json"), "unavailable", True, 2),
    # a 200 JSON body without `status`: upstream's except returns False before the golden call
    (_Resp(200, {"data": 1}), _data({"p": 2}), "predict_unreadable", False, 1),
    (_data({"p": 1}), _Resp(404), "golden_no_response", False, 4),
    # an answer that already makes the tier no match wins over the other call's outage
    (_Resp(429), _Resp(400), "golden_no_response", False, 6),
    (_Resp(400), _Resp(503), "no_response", False, 6),
    # outages: a call's last reply was an outage and no call got an answer of its own
    (_data({"p": 1}), _Resp(503), "unavailable", True, 4),
    (_Resp(429), _data({"p": 2}), "unavailable", True, 4),
    (_Resp(403), _data({"p": 2}), "unavailable", True, 4),
    (ConnectionError("dns"), _data({"p": 2}), "unavailable", True, 4),
    (_Resp(503), TimeoutError("slow"), "unavailable", True, 6),
])
def test_the_tier_tells_an_outage_from_the_models_call(rapid, pred_reply, gold_reply, outcome,
                                                        unavailable, n_calls):
    def by_id(params):
        return pred_reply if "sortBy" in params else gold_reply

    def answer(params):
        r = by_id(params)
        if isinstance(r, BaseException):
            raise r
        return r
    rapid["answers"] = {"Search_Attractions": [answer]}
    verdict, cmp = _pair(_api(), PRED, GOLD)
    assert verdict is False and cmp.response_unavailable is unavailable
    assert cmp.response_calls[0]["outcome"] == outcome
    assert len(rapid["seen"]) == n_calls


def test_a_predicted_call_answering_an_empty_object_is_no_match_without_a_golden_call(rapid):
    rapid["answers"] = {"Search_Attractions": [_data({})]}
    stats = Counter()
    verdict, cmp = _pair(_api(stats), PRED, GOLD)
    assert verdict is False and len(rapid["seen"]) == 1
    rec = cmp.response_calls[0]
    assert rec["outcome"] == "predict_empty_data" and rec["message"].startswith("API call failed")
    assert cmp.error_message == [] and stats["predict_empty_data"] == 1


def _judge(monkeypatch, verdicts):
    it = iter(verdicts)

    async def fake(contents, *, config=None):
        v = next(it)
        return (None, "JUDGE_OUTAGE") if v is None else (json.dumps({"is_equal": v}), "gemini-x")
    monkeypatch.setattr(C, "judge_generate_cascade", fake)


def test_an_unavailable_backend_leaves_the_pair_to_the_judge(rapid, monkeypatch):
    rapid["answers"] = {"Search_Attractions": [_Resp(503)]}
    _judge(monkeypatch, [True, False, None])
    cmp = C._CompareFC(C._load_exact_match_dict(), C._get_embedder(), Counter(), _api())
    # the judge finds it equivalent: a match, as upstream (either tier would match it)
    assert asyncio.run(cmp.compare_single_call([], [], PRED, GOLD)) == (True, None)
    # not equivalent: upstream's verdict is unknown -> a harness error, never the model's failure
    with pytest.raises(C._ResponseTierUnavailable) as ei:
        asyncio.run(cmp.compare_single_call([], [], PRED, GOLD))
    assert ei.value.kind == "rapidapi_unavailable"
    # no verdict at all: the judge's own harness error
    with pytest.raises(C._CallJudgeNoVerdict):
        asyncio.run(cmp.compare_single_call([], [], PRED, GOLD))


def test_a_pair_the_tier_matches_never_reaches_the_judge(rapid, monkeypatch):
    rapid["answers"] = {"Search_Attractions": [_data([1])]}

    async def boom(*a, **k):
        raise AssertionError("the judge must not be asked")
    monkeypatch.setattr(C, "judge_generate_cascade", boom)
    cmp = C._CompareFC(C._load_exact_match_dict(), C._get_embedder(), Counter(), _api())
    assert asyncio.run(cmp.compare_single_call([], [], PRED, GOLD)) == (True, None)
    assert cmp.judge_calls == []


def _item():
    return {"id": "Attraction-107",
            "functions": [{"name": "Search_Attractions",
                           "parameters": {"required": ["id"], "properties": {
                               "id": {"type": "string"}, "sortBy": {"type": "string"}}}}],
            "conversations": [
                {"role": "user", "content": "trending attractions"},
                {"role": "assistant", "function_call": [GOLD]},
                {"role": "observation", "content": [{"status": True, "data": "obs"}]},
                {"role": "assistant", "content": "final"}]}


def _drive(item, api, replies):
    it = iter(replies)

    async def chat(messages, tools):
        return next(it)
    return asyncio.run(C._evaluate_one(item, chat, C._load_exact_match_dict(), C._get_embedder(),
                                       Counter(), api))


def _tool_call(args):
    return {"content": None, "tool_calls": [{"id": "c1", "function": {
        "name": "Search_Attractions", "arguments": json.dumps(args)}}]}


def test_a_sample_the_backend_could_not_answer_is_not_scored(rapid, monkeypatch):
    rapid["answers"] = {"Search_Attractions": [_Resp(429)]}
    _judge(monkeypatch, [False])
    r = _drive(_item(), _api(), [_tool_call(PRED["arguments"])])
    assert r["harness_error"] == "rapidapi_unavailable" and "count_dict" not in r
    assert r["response_based_calls"][0]["outcome"] == "unavailable"
    assert RAPID_SECRET not in json.dumps(r)
    agg = C._aggregate([r, {"id": "Hotels-1", "request_failed": False, "message": "Success.",
                            "count_dict": {"success_turn_num": 1, "total_turn_num": 1,
                                           "correct_call_num": 1, "total_call_num": 1,
                                           "real_turn_num": 1}, "resp_eval": None}], 2, False)
    assert agg["n_scored"] == 1 and agg["harness_error_samples"] == {"rapidapi_unavailable": 1}
    errors, parts = C._harness_error_report(agg, 0)
    assert errors[0]["error"] == C.HARNESS_ERROR_KINDS["rapidapi_unavailable"]
    assert "response-based tier could not answer" in parts[0]


def test_the_sortby_pair_succeeds_through_the_tier(rapid, monkeypatch):
    rapid["answers"] = {"Search_Attractions": [_data({"products": ["same"]})]}

    async def boom(*a, **k):
        raise AssertionError("no judge call")
    monkeypatch.setattr(C, "judge_generate_cascade", boom)
    monkeypatch.setattr(C, "_resp_eval", lambda *a, **k: _none())
    r = _drive(_item(), _api(), [_tool_call(PRED["arguments"]), {"content": "final",
                                                                  "tool_calls": None}])
    assert r["message"] == "Success." and r["count_dict"]["correct_call_num"] == 1


async def _none():
    return None


def _run(monkeypatch, stats, kinds=None):
    monkeypatch.setattr(C, "check_complexfuncbench_prerequisites", lambda: (True, ""))
    kinds = kinds or {}

    async def fake_drive(*a, **k):
        return {"success_rate": 50.0, "call_accuracy": 60.0, "completeness": 1.5,
                "correctness": 1.0, "n_scored": 2, "n_request_failed": 0, "n_success": 1,
                "per_domain_success_rate": None, "_is_full": False, "_n_items": 3,
                "_embed_backend": "transformers", "_judge_usage": Counter({"gemini-x": 3}),
                "_rapidapi_stats": stats, "harness_error_samples": kinds,
                "n_harness_errors": sum(kinds.values())}
    monkeypatch.setattr(C, "_drive_all", fake_drive)
    return C.run_complexfuncbench("m", "http://x/v1", 1, limit=3)


def test_the_result_records_the_tier_and_names_its_outages(monkeypatch):
    monkeypatch.setenv("RAPID_API_KEY", RAPID_SECRET)
    stats = Counter({"calls": 6, "attempts": 10, "calls_without_response": 2,
                     "no_response:HTTP 429": 2, "pairs_equal": 1, "pairs_different": 1,
                     "pairs_unavailable": 1, "predict_empty_data": 1,
                     "pairs_golden_no_response": 2})
    r = _run(monkeypatch, stats, {"rapidapi_unavailable": 1})
    tier = r["response_based_tier"]
    assert tier["run"] is True and tier["tool"] == "booking-com15"
    assert tier["tool_info_sha256"] == C.TOOL_INFO_SHA256
    assert (tier["calls"], tier["attempts"]) == (6, 10)
    assert tier["calls_without_response"] == {"HTTP 429": 2}
    assert (tier["pairs_equal"], tier["pairs_unavailable"], tier["predict_empty_data"]) == (1, 1, 1)
    assert (tier["pairs_golden_no_response"], tier["pairs_no_response"]) == (2, 0)
    reason = r["leaderboard_comparable_reason"]
    assert "not run" not in reason
    assert "could not answer 1 call pair(s) of the response-based tier (HTTP 429 x2)" in reason
    assert "answered an empty object" in reason
    assert "1 sample(s) not scored because the booking-com15 RapidAPI backend" in reason
    assert r["status"] == "completed_with_errors"
    assert r["harness_errors"][0]["error"] == C.HARNESS_ERROR_KINDS["rapidapi_unavailable"]
    assert RAPID_SECRET not in json.dumps(r)


def test_no_rapid_api_key_is_infra_required_before_the_run(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-0123456789")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setattr(C, "gemini_key_live_valid", lambda k: (True, ""))
    monkeypatch.delenv("RAPID_API_KEY", raising=False)
    monkeypatch.setattr(C, "rapidapi_live_check",
                        lambda k: pytest.fail("no live check without a key"))
    ok, why = C.check_complexfuncbench_prerequisites()
    assert not ok and "RAPID_API_KEY is not set" in why and "booking-com15" in why
    assert "never replaced by the Gemini judge" in why
    with pytest.raises(RuntimeError) as e:
        C.run_complexfuncbench("m", "http://x/v1", 1)
    assert "[INFRA REQUIRED] complexfuncbench" in str(e.value) and "RAPID_API_KEY" in str(e.value)


@pytest.mark.parametrize("replies, expect", [
    ([_Resp(200, {"status": True})], None),
    ([_Resp(404)], None),                       # the key works; any answer but an outage
    ([_Resp(403)], "HTTP 403"),
    ([_Resp(401)], "HTTP 401"),
    ([_Resp(503)], "HTTP 503"),
    ([_Resp(429), _Resp(429), _Resp(200, {})], None),
])
def test_the_rapidapi_live_check(monkeypatch, no_sleep, replies, expect):
    seen = []

    def fake_get(url, headers, params, timeout):
        seen.append((url, headers))
        return replies[min(len(seen) - 1, len(replies) - 1)]
    monkeypatch.setattr(C, "_rapidapi_get", fake_get)
    assert C.rapidapi_live_check(RAPID_SECRET) == expect
    assert seen[0][0] == "https://booking-com15.p.rapidapi.com/api/v1/test"
    assert len(seen) <= C._RAPIDAPI_ATTEMPTS


def test_a_refused_key_is_infra_required_and_the_key_never_shown(monkeypatch, no_sleep):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key-0123456789")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setattr(C, "gemini_key_live_valid", lambda k: (True, ""))
    monkeypatch.setenv("RAPID_API_KEY", RAPID_SECRET)
    monkeypatch.setattr(C, "_rapidapi_get", lambda *a, **k: _Resp(403))
    ok, why = C.check_complexfuncbench_prerequisites()
    assert not ok and "HTTP 403" in why and "not subscribed" in why
    assert RAPID_SECRET not in why


def test_the_complexfuncbench_doc_describes_the_tier():
    doc = (ROOT / "docs" / "evals" / "complexfuncbench.md").read_text(encoding="utf-8")
    assert doc.isascii()
    flat = " ".join(doc.split())
    for needle in ("`rapidapi_unavailable`", "`RAPID_API_KEY`", "`response_based_calls`",
                   "`GBENCH_COMPLEXFUNCBENCH_RAPIDAPI_TIMEOUT_S`", "`TOOL_INFO_SHA256`",
                   "`predict_empty_data`", "`pairs_unavailable`", "`calls_without_response`",
                   "`pairs_golden_no_response`", "`golden_no_response`",
                   "`Test_API`", "never replaced by the Gemini judge"):
        assert needle in flat, needle
    assert "is not run" not in flat and "No RapidAPI subscription" not in flat
    assert "response_based tier not run" not in " ".join((C.__doc__ or "").split())


# =========================================================================== #
# api_bank: the Translate API's googletrans in its own image
# =========================================================================== #
@pytest.fixture
def docker_calls(monkeypatch):
    """subprocess.run in api_bank stubbed: `docker run` of the worker answers from `replies`
    (op -> reply dict or a callable(request, argv, env) -> CompletedProcess)."""
    state = {"calls": [], "replies": {}}

    real_run = A.subprocess.run

    def fake_run(argv, input=None, capture_output=True, text=True, timeout=None, env=None, **kw):
        if not isinstance(argv, (list, tuple)) or not argv or argv[0] != "docker":
            return real_run(argv, input=input, capture_output=capture_output, text=text,
                            timeout=timeout, env=env, **kw)
        state["calls"].append({"argv": list(argv), "input": input, "env": dict(env or {})})
        if argv[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, json.dumps(A.TRANSLATE_IMAGE_LABELS), "")
        req = json.loads(input or "{}")
        rep = state["replies"].get(req.get("op"))
        if callable(rep):
            return rep(req, argv, env)
        return subprocess.CompletedProcess(argv, 0, "noise\n" + A._WORKER_REPLY_MARK
                                           + json.dumps(rep) + "\n", "")
    monkeypatch.setattr(A.subprocess, "run", fake_run)
    monkeypatch.setattr(A.shutil, "which", lambda name: "/usr/bin/docker")
    A._PROBE.clear()
    A._TRANSLATE_STATS.clear()
    yield state
    A._PROBE.clear()
    A._TRANSLATE_STATS.clear()


def _worker_sha():
    import hashlib
    return hashlib.sha256(Path(A._TRANSLATE_WORKER).read_bytes()).hexdigest()


def _probe(**over):
    return {"googletrans": "3.1.0a0", "httpx": "0.13.3", "worker_sha256": _worker_sha(),
            "languages": {"en": "english", "fr": "french", "es": "spanish"},
            "langcodes": {"english": "en"}, **over}


@pytest.fixture
def clean_proxies(monkeypatch):
    for n in A._PROXY_URL_VARS + A._NO_PROXY_VARS:
        monkeypatch.delenv(n, raising=False)


def test_the_image_files_pin_upstreams_googletrans():
    dockerfile = (ROOT / A.TRANSLATE_DOCKERFILE).read_text(encoding="utf-8")
    assert dockerfile.isascii() and "FROM python:3.10-slim" in dockerfile
    assert A.translate_image_build_command() in " ".join(dockerfile.split())
    for k, v in A.TRANSLATE_IMAGE_LABELS.items():
        assert f'{k}="{v}"' in dockerfile, k
    assert "USER 65534:65534" in dockerfile
    pins = (ROOT / "gbench" / "docker" / "api_bank.constraints.txt").read_text(encoding="utf-8")
    assert "googletrans==3.1.0a0" in pins.splitlines() and "httpx==0.13.3" in pins.splitlines()
    worker = Path(A._TRANSLATE_WORKER).read_text(encoding="utf-8")
    assert "Translator(raise_exception=True)" in worker and A._WORKER_REPLY_MARK in worker


def test_the_translate_backend_checks_pass_and_run_the_metadata_guard(docker_calls, monkeypatch,
                                                                      clean_proxies):
    docker_calls["replies"]["probe"] = _probe()
    guard = []
    import gbench.runners.eval_suites.container_guard as CG
    monkeypatch.setattr(CG, "require_metadata_blocked",
                        lambda eval_name, image, network, docs, resolve=(): guard.append(
                            (eval_name, image, network, list(resolve))))
    probe = A.check_translate_backend()
    assert probe["googletrans"] == "3.1.0a0"
    assert guard == [("api_bank", A.TRANSLATE_IMAGE, "bridge", ["translate.googleapis.com"])]
    run = [c for c in docker_calls["calls"] if c["argv"][:2] == ["docker", "run"]][0]["argv"]
    assert "--network" in run and run[run.index("--network") + 1] == "bridge"
    assert "--read-only" in run and "--pull" in run and run[-1] == A.TRANSLATE_IMAGE
    assert "host" not in run[run.index("--network") + 1]


@pytest.mark.parametrize("probe, needle", [
    ({"googletrans": "4.0.2"}, "googletrans 4.0.2"),
    ({"httpx": "0.28.1"}, "httpx 0.28.1"),
    ({"worker_sha256": "0" * 64}, "rebuild it"),
])
def test_a_foreign_image_is_infra_required(docker_calls, clean_proxies, probe, needle):
    docker_calls["replies"]["probe"] = _probe(**probe)
    with pytest.raises(RuntimeError) as e:
        A.check_translate_backend()
    assert "[INFRA REQUIRED] api_bank" in str(e.value) and needle in str(e.value)
    assert A.translate_image_build_command() in str(e.value)


def test_a_missing_image_or_docker_is_infra_required(monkeypatch, clean_proxies):
    monkeypatch.setattr(A.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="docker CLI is not on PATH"):
        A.check_translate_backend()
    monkeypatch.setattr(A.shutil, "which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr(A.subprocess, "run", lambda argv, **k: subprocess.CompletedProcess(
        argv, 1, "", "No such image"))
    with pytest.raises(RuntimeError) as e:
        A.check_translate_backend()
    assert "not built" in str(e.value) and A.translate_image_build_command() in str(e.value)


def test_a_proxy_with_a_password_never_enters_the_container(docker_calls, monkeypatch,
                                                            clean_proxies):
    monkeypatch.setenv("HTTPS_PROXY", "http://user:proxy-pass-0123456789@proxy.corp:3128")
    docker_calls["replies"]["probe"] = _probe()
    with pytest.raises(RuntimeError) as e:
        A.check_translate_backend()
    assert "HTTPS_PROXY" in str(e.value) and "proxy-pass-0123456789" not in str(e.value)
    assert not [c for c in docker_calls["calls"] if c["argv"][:2] == ["docker", "run"]]


def _translate_module(tmp_path):
    """apis/translate.py of the pinned checkout's shape (its import lines and the calls it
    makes), in a scratch package under tmp_path."""
    pkg = tmp_path / "co" / "apis"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "translate.py").write_text(
        "import googletrans\nfrom googletrans import Translator\nimport os\n\n"
        "class Translate:\n"
        "    def __init__(self):\n"
        "        self.translator = Translator()\n"
        "        self.LANGUAGES = googletrans.LANGUAGES\n"
        "    def call(self, src, tgt_lang):\n"
        "        proxy = os.environ.get('ALL_PROXY')\n"
        "        os.environ['ALL_PROXY'] = ''\n"
        "        try:\n"
        "            out = self.translator.translate(src, dest=tgt_lang).text\n"
        "        except Exception as e:\n"
        "            os.environ['ALL_PROXY'] = proxy\n"
        "            return {'output': None, 'exception': str(e)}\n"
        "        os.environ['ALL_PROXY'] = proxy\n"
        "        return {'output': out, 'exception': None}\n")
    return str(tmp_path / "co")


def test_translate_py_runs_unchanged_against_the_pinned_release(docker_calls, tmp_path,
                                                                 monkeypatch, clean_proxies):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    monkeypatch.setenv("NO_PROXY", "localhost")
    docker_calls["replies"]["probe"] = _probe()
    docker_calls["replies"]["translate"] = {"result": {"text": "Hola", "src": "en",
                                                       "dest": "es", "origin": "Hello",
                                                       "pronunciation": "Hola"}}
    d = _translate_module(tmp_path)
    monkeypatch.delitem(sys.modules, "apis", raising=False)
    monkeypatch.delitem(sys.modules, "apis.translate", raising=False)
    before = sys.modules.get("googletrans")
    try:
        with A._in_checkout(d), A._proxy_env():
            from apis.translate import Translate
            tool = Translate()
            assert tool.LANGUAGES == {"en": "english", "fr": "french", "es": "spanish"}
            assert tool.call("Hello", "es") == {"output": "Hola", "exception": None}
    finally:
        sys.modules.pop("apis.translate", None)
        sys.modules.pop("apis", None)
    assert sys.modules.get("googletrans") is before
    call = [c for c in docker_calls["calls"]
            if c["input"] and json.loads(c["input"]).get("op") == "translate"][0]
    assert json.loads(call["input"]) == {"op": "translate", "text": "Hello", "dest": "es",
                                         "src": "auto"}
    argv = call["argv"]
    # the proxy read when Translator() was built, as the container reaches a loopback proxy
    at = argv.index("HTTPS_PROXY")
    assert argv[at - 1:at + 1] == ["-e", "HTTPS_PROXY"]
    assert call["env"]["HTTPS_PROXY"] == "http://host.docker.internal:3128"
    assert call["env"]["NO_PROXY"] == "localhost"
    assert "ALL_PROXY" not in argv                       # '' while scoring: no proxy
    assert "host.docker.internal:host-gateway" in argv
    assert A._TRANSLATE_STATS["calls"] == 1


def test_a_google_refusal_is_an_exception_and_a_backend_outage(docker_calls, tmp_path,
                                                               monkeypatch, clean_proxies):
    docker_calls["replies"]["probe"] = _probe()
    msg = "Unexpected status code \"429\" from ['translate.googleapis.com']"
    docker_calls["replies"]["translate"] = {"error": {"type": "Exception", "message": msg}}
    d = _translate_module(tmp_path)
    monkeypatch.delitem(sys.modules, "apis", raising=False)
    monkeypatch.delitem(sys.modules, "apis.translate", raising=False)
    try:
        with A._in_checkout(d), A._proxy_env():
            from apis.translate import Translate
            res = Translate().call("Hello", "es")
            gold = Translate().call("Hello", "es")
    finally:
        sys.modules.pop("apis.translate", None)
        sys.modules.pop("apis", None)
    assert res == {"output": None, "exception": msg}
    err = A._backend_harness_error("Translate", "Translate", res, lambda: gold["exception"])
    assert err["kind"] == "backend_unavailable" and err["api"] == "Translate"
    rec = A._translate_record(dict(A._PROBE))
    assert rec["calls"] == 2 and rec["calls_raised"] == {"Exception": 2}
    assert rec["raise_exception"] is True and rec["googletrans"] == "3.1.0a0"


def test_a_worker_that_gives_no_reply_is_an_exception(docker_calls, tmp_path, monkeypatch,
                                                      clean_proxies):
    docker_calls["replies"]["probe"] = _probe()
    docker_calls["replies"]["translate"] = lambda req, argv, env: subprocess.CompletedProcess(
        argv, 125, "", "docker: Error response from daemon")
    tr = A._WorkerTranslator()
    with pytest.raises(A._TranslateWorkerError, match="gave no reply"):
        tr.translate("Hello", dest="fr")
    assert A._TRANSLATE_STATS["worker_failed"] == 1


def test_the_api_bank_doc_describes_the_translate_backend():
    doc = (ROOT / "docs" / "evals" / "api_bank.md").read_text(encoding="utf-8")
    assert doc.isascii()
    flat = " ".join(doc.split())
    for needle in (A.TRANSLATE_IMAGE, A.translate_image_build_command(), "raise_exception=True",
                   "`translate_backend`", "`Translate-level-1-1.jsonl` to "
                   "`Translate-level-1-4.jsonl`", "HTTP 429", "`HTTPS_PROXY`", "`ALL_PROXY`",
                   "host.docker.internal", "api_bank.constraints.txt"):
        assert needle in flat, needle
    assert "pip install rouge sentence-transformers nltk googletrans" not in flat
    assert "gbench does not install it" not in flat


# =========================================================================== #
# bfcl: a multi-turn checker exception is a harness error
# =========================================================================== #
def _mt(i, status="OK", correct=True, cat="multi_turn_base", **extra):
    return {"sample_idx": i, "source_sample_idx": i, "attempt_idx": 0, "category": cat,
            "question_id": f"{cat}_{i}", "is_correct": correct, "status": status,
            "error": None if status == "OK" else f"{status} error",
            "health": "ok" if status == "OK" else "request_failed", "steps": 1, **extra}


def _checker_raised(i, cat="multi_turn_base"):
    return _mt(i, correct=False, cat=cat, bfcl_error_type=bfcl.CHECKER_EXCEPTION,
               bfcl_error=["multi_turn_checker raised KeyError: 'x'"])


def test_the_checker_exception_record_is_what_the_scorer_writes(monkeypatch):
    runner = types.ModuleType("bfcl_eval.eval_checker.eval_runner")

    def raising(*a, **k):
        raise KeyError("x")
    runner._evaluate_single_multi_turn_entry = raising
    for name in ("bfcl_eval", "bfcl_eval.eval_checker"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "bfcl_eval.eval_checker.eval_runner", runner)
    monkeypatch.setattr(bfcl, "_drop_instances", lambda *a: None)

    class Handler:
        _local = types.SimpleNamespace()

        def inference(self, entry, include_input_log=False, exclude_state_log=True):
            return [[["f()"]]], {}
    item = {"id": "multi_turn_base_0", "category": "multi_turn_base", "entry": {},
            "ground_truth": [], "eval_entry": {}}
    trace = bfcl._score_multi_turn_entry(Handler(), item, "k")
    assert trace["status"] == "OK" and trace["is_correct"] is False
    assert trace["bfcl_error_type"] == "multi_turn:checker_exception"
    assert bfcl.multi_turn_harness_error(trace) == {
        "kind": "checker_exception", "detail": "multi_turn_checker raised KeyError: 'x'"}


def test_a_checker_exception_is_left_out_of_the_score():
    traces = [_mt(0), _mt(1, correct=False), _checker_raised(2),
              _checker_raised(3, cat="multi_turn_miss_func"),
              _mt(4, "FAILED", False, bfcl_error_type=bfcl.CHECKER_EXCEPTION)]
    counts = bfcl._multi_turn_counts(traces)
    assert counts["category_accuracy"] == {"multi_turn_base": {"correct": 1, "total": 2,
                                                               "accuracy": 50.0}}
    assert counts["categories_unscored"] == ["multi_turn_miss_func"]
    assert counts["scoring_excluded"] == 3 and counts["failed_requests"] == 1
    # the failed request is an endpoint failure first; a checker exception is not a request
    assert counts["request_harness_errors"] == {"checker_exception": 2, "endpoint_failure": 1}
    assert counts["harness_errors_left_out"] == {
        "multi_turn_base": {"checker_exception": 1, "endpoint_failure": 1},
        "multi_turn_miss_func": {"checker_exception": 1}}
    assert traces[2]["scoring_excluded"] and traces[2]["harness_error"]["kind"] == \
        "checker_exception"
    assert not traces[1].get("scoring_excluded")          # the model's own wrong answer


def test_bfcl_names_a_checker_exception_and_completes_with_errors(monkeypatch):
    mt_traces = [_mt(0), _checker_raised(1)]
    samples = [([{"role": "user", "content": "q-ok-0"}],
                {"id": "simple_python_0", "category": "simple_python", "ground_truth": []},
                {"category": "simple_python"})]
    monkeypatch.setattr(bfcl, "_harness_probe", lambda name, multi, single_: (
        bfcl.BFCL_EVAL_RELEASE, [{"id": t["question_id"], "category": t["category"]}
                                 for t in mt_traces]))
    monkeypatch.setattr(bfcl, "_load_bfcl_samples", lambda cats: (
        [s for s in samples if s[2]["category"] in cats], {s[1]["id"]: [] for s in samples}))
    monkeypatch.setattr(bfcl, "check_single_turn", lambda items, eval_name="bfcl": [
        {"valid": True, "error_type": None, "error": []} for _ in items])
    monkeypatch.setattr(bfcl, "endpoint_network", lambda url: None)
    monkeypatch.setattr(bfcl, "run_multi_turn_in_image", lambda *a, **k: {
        "traces": [dict(t) for t in mt_traces], "guard_refusals": [], "harness_error": None,
        "endpoint_relay": None})
    for name in ("OPENAI_API_KEY", "HTTP_PROXY", "HTTPS_PROXY", "GBENCH_BFCL_TIMEOUT_S"):
        monkeypatch.delenv(name, raising=False)

    async def send(**kw):
        return base.Reply("", [{"id": "c", "type": "function",
                                "function": {"name": "f", "arguments": "{}"}}], "tool_calls",
                          completion_tokens=3)
    with mock.patch.object(base, "_send_single_request", send):
        res = bfcl.run_bfcl("m", "http://127.0.0.1:9/v1", 1,
                            eval_categories="simple_python,multi_turn_base")
    assert res["category_accuracy"]["multi_turn_base"] == {"correct": 1, "total": 1,
                                                           "accuracy": 100.0}
    assert res["request_harness_errors"] == {"checker_exception": 1}
    assert res["scoring_excluded"] == 1 and res["failed_requests"] == 0
    assert res["status"] == "completed_with_errors"
    assert "checker_exception 1" in res["status_reason"]
    assert ("1 single-turn and multi-turn entry left out of the score as harness errors "
            "(checker_exception: 1; BFCL scores a failed request wrong; its evaluation aborts "
            "where the multi-turn checker raises)") in res["leaderboard_comparable_reason"]
    assert res["multi_turn_report"]["harness_errors_left_out"] == {
        "multi_turn_base": {"checker_exception": 1}}


def test_the_bfcl_doc_names_the_checker_exception():
    flat = " ".join((ROOT / "docs" / "evals" / "bfcl.md").read_text(encoding="utf-8").split())
    assert "A checker exception scores that entry wrong" not in flat
    assert "`checker_exception`" in flat and "`multi_turn:checker_exception`" in flat
    assert "Upstream would abort the whole evaluation; one entry is marked invalid" not in \
        (ROOT / "gbench" / "runners" / "eval_suites" / "bfcl.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("value, flagged", [
    ("http://user:pw1@proxy.corp:3128", True),          # a short password is a password too
    ("user:secret@proxy.corp:3128", True),
    ("http://user@proxy.corp:3128", False),             # a bare user name is not a credential
    ("http://proxy.corp:3128", False),
])
def test_any_password_in_a_proxy_url_is_a_credential(monkeypatch, value, flagged):
    monkeypatch.setattr(A, "secret_values", lambda: [])
    assert A._proxy_credentials({"HTTPS_PROXY": value}) == (["HTTPS_PROXY"] if flagged else [])
