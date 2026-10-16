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

"""Function-calling suites: a backend outage is never the model's call, and a headline says what
it covers.

* api_bank: a Translate call that failed on the harness's or Google's side is a harness error
  without a gold re-run; the name-lookup check reads ALL_PROXY as the worker's httpx does.
* gorilla_apibench: the 0-shot headline names the hubs and n it rests on, and is withheld when
  the draw holds no 0-shot question.

No network, no docker.
"""

import json

import pytest

from gbench.runners.eval_suites import api_bank as AB
from gbench.runners.eval_suites import gorilla_apibench as GA


@pytest.fixture(autouse=True)
def _fresh_failures(monkeypatch):
    monkeypatch.setattr(AB, "_BACKEND_FAILURES", {})


def _translator(monkeypatch, reply=None, raise_worker=None):
    def run(request, proxy_env):
        if raise_worker:
            raise AB._TranslateWorkerError(raise_worker)
        return reply
    monkeypatch.setattr(AB, "_run_translate_worker", run)
    return AB._WorkerTranslator()


@pytest.mark.parametrize("reply, worker, kind", [
    (None, "gbench's googletrans worker gave no reply within 120 s", "translate_worker"),
    ({"error": {"type": "ConnectError", "message": "[Errno 111] Connection refused"}}, None,
     "translate_transport"),
    ({"error": {"type": "Exception", "message":
                'Unexpected status code "429" from [\'translate.googleapis.com\']'}}, None,
     "translate_backend_status"),
])
def test_a_translate_outage_on_the_models_call_is_a_harness_error(monkeypatch, reply, worker,
                                                                  kind):
    tr = _translator(monkeypatch, reply=reply, raise_worker=worker)
    with pytest.raises(AB._TranslateWorkerError) as e:
        tr.translate("hola", dest="en")
    res = {"api_name": "Translate", "exception": str(e.value)}

    def no_rerun():
        raise AssertionError("the gold call is not re-run for a plain outage")
    err = AB._backend_harness_error("Translate", "Translate", res, no_rerun)
    assert err["kind"] == "backend_unavailable" and err["detail"].startswith(kind)


def test_another_translate_error_still_needs_the_gold_call_to_fail(monkeypatch):
    tr = _translator(monkeypatch, reply={"error": {"type": "ValueError",
                                                   "message": "Language not supported."}})
    with pytest.raises(AB._TranslateWorkerError):
        tr.translate("hola", dest="xx")
    res = {"exception": "Language not supported."}
    assert AB._backend_harness_error("Translate", "Translate", res, lambda: None) is None
    assert AB._backend_harness_error("Translate", "Translate", res,
                                     lambda: "HTTPError: 503")["kind"] == "backend_unavailable"


def test_the_lookup_check_reads_all_proxy_as_the_worker_does():
    env = AB.lookup_proxy_env({"ALL_PROXY": "http://proxy:3128"})
    assert env["HTTPS_PROXY"] == env["HTTP_PROXY"] == "http://proxy:3128"
    kept = AB.lookup_proxy_env({"ALL_PROXY": "http://a:1", "https_proxy": "http://b:2"})
    assert kept["https_proxy"] == "http://b:2" and "HTTPS_PROXY" not in kept
    assert kept["HTTP_PROXY"] == "http://a:1"


def _gorilla_run(monkeypatch, rows):
    """run_gorilla_apibench over traces of (hub, setting, correct)."""
    traces = []
    for i, (hub, setting, ok) in enumerate(rows):
        traces.append({"sample_idx": i, "status": "OK", "is_correct": ok,
                       "response_text": "x",
                       "gold_answer": json.dumps({"hub": hub, "setting": setting})})
    monkeypatch.setattr(GA, "check_prereqs", lambda: None)
    monkeypatch.setattr(GA, "_load_gorilla_apibench_samples", lambda limit=None: [])
    monkeypatch.setattr(GA, "_score", lambda text, gold: (json.loads(gold).get("ok", True),
                                                          False))
    pooled = round(100.0 * sum(1 for *_, ok in rows if ok) / len(rows), 2)
    monkeypatch.setattr(GA, "run_eval_suite", lambda **kw: {"sample_traces": traces,
                                                            "accuracy": pooled})
    return GA.run_gorilla_apibench("m", "http://x/v1", 1, limit=len(rows))


def test_a_headline_over_fewer_hubs_names_them(monkeypatch):
    res = _gorilla_run(monkeypatch, [("torchhub", GA.HEADLINE_SETTING, True),
                                     ("huggingface", "bm25", False)])
    assert res["accuracy"] == 100.0 and res["headline_n"] == 1
    assert "mean over 1 of 3 hubs (torchhub, n=1)" in res["metric"]
    assert "the headline covers 1 of the 3 hubs (torchhub, n=1" in \
        res["leaderboard_comparable_reason"]


def test_a_draw_without_a_zero_shot_question_withholds_the_headline(monkeypatch):
    res = _gorilla_run(monkeypatch, [("huggingface", "bm25", True),
                                     ("tensorflowhub", "oracle", False)])
    assert res["accuracy"] is None and res["headline_n"] == 0
    assert "no 0-shot question was scored" in res["accuracy_withheld_reason"]
