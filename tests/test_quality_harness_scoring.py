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
"""The quality runner never turns a harness failure into a score of the model under test.

* When no scenario is left to score (every one had a harness error fall in it or needs another
  provider, or the suite reported none), pass_rate is withheld (null, with
  pass_rate_withheld_reason naming why), never 0.0.
* A model-endpoint 404 (and every other 4xx but 400 and 422) on a request the suite makes is a
  harness error: an endpoint that serves the model under another name, or has no such path, is not
  the model's failure. The same 4xx on a request that names another model is the model's side.
* The QA container runs on `--network none`: the cloud metadata server guard is called with the
  network of the run's own docker command line and needs no probe; if the container ever got a
  network, a reachable metadata server would stop the run before the container starts.

Model-free: stub endpoints, docker stubbed, the guard's probe stubbed; no GemmaClaw build, no Gemini
call, no metadata-server contact. Files are written under tmp only.
"""

import json
from pathlib import Path

import pytest

from gbench.core.config import BenchmarkConfig
from gbench.core.models import ModelConfig, ModelFormat
from gbench.runners import quality as Q
from gbench.runners.eval_suites import container_guard as cg

# The stub upstreams, the relay fixture and the stubbed whole run of the isolation tests.
import test_quality_isolation as ISO
from test_quality_isolation import gemini, model_up, relay_to  # noqa: F401 - pytest fixtures

ROOT = Path(__file__).resolve().parents[1]
SERVED = "google/gemma-4-E2B-it"         # the model id a remote run configures (hf_model_id)
#: vLLM's answer to a request for a model name it does not serve
NOT_SERVED = (b'{"object": "error", "message": "The model `google/gemma-4-E2B-it` does not '
              b'exist.", "type": "NotFoundError", "code": 404}')


def _chat(model=SERVED):
    return json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}],
                       "stream": True}).encode("utf-8")


# --------------------------------------------------------------------------- #
# pass_rate is withheld, never 0.0, when nothing is left to score
# --------------------------------------------------------------------------- #
def _runner(tmp_path):
    config = BenchmarkConfig()
    config.results_dir = tmp_path / "results"
    config.results_dir.mkdir(exist_ok=True)
    config.initialize()
    return Q.QualityBenchmarkRunner(config)


def _score(tmp_path, summary, scenario_summaries, events=(), meta=None):
    meta = meta or tmp_path / "meta"
    (meta / "qa" / "scenarios").mkdir(parents=True, exist_ok=True)
    errors = {}
    for _t, kind in events:
        errors[kind] = errors.get(kind, 0) + 1
    report = {"harness_error_events": list(events), "harness_errors": errors,
              "gemini_calls": [], "model_endpoint_calls": [], "refused": [], "rejected": 0}
    model = ModelConfig(short_name="m", hf_model_id="o/m", total_params_b=1.0, is_moe=False)
    out = tmp_path / "quality.json"
    res = _runner(tmp_path)._score(
        model, ModelFormat.HF, "c" * 40, summary,
        {"scenario_summaries": scenario_summaries, "forwarder_errors": [], "entrypoint": {}},
        report, meta, False, False, [], {}, out)
    return res, json.loads(out.read_text())


def test_every_scenario_hit_by_a_harness_error_withholds_the_pass_rate(tmp_path):
    """Two scenarios, a model_unreachable in each one's window."""
    summary = {"counts": {"total": 2, "passed": 0, "failed": 2},
               "scenarios": [{"name": "A", "status": "fail", "steps": [{"name": "flow"}]},
                             {"name": "B", "status": "fail", "steps": [{"name": "flow"}]}]}
    scen = {"a": ISO._scen_summary("A", 1000, 1060, "fail"),
            "b": ISO._scen_summary("B", 1100, 1160, "fail")}
    res, saved = _score(tmp_path, summary, scen,
                        events=[(1030.0, "model_unreachable"), (1130.0, "model_unreachable")])
    assert res["status"] == "completed_with_errors"
    assert (res["total_scenarios"], res["passed_scenarios"], res["failed_scenarios"]) == (0, 0, 0)
    assert res["pass_rate"] is None and saved["pass_rate"] is None
    assert res["pass_rate_withheld_reason"] == (
        "no scenario could be scored: of the 2 scenario(s) the suite reported, 2 had a harness "
        "error fall in them")
    assert [r["id"] for r in res["harness_error_scenarios"]] == ["a", "b"]
    assert res["unattributed_harness_errors"] == {}
    assert ("pass_rate withheld: no scenario could be scored"
            in res["leaderboard_comparable_reason"])


def test_scenarios_that_all_need_another_provider_withhold_the_pass_rate(tmp_path):
    meta = tmp_path / "meta"
    ISO._write_meta(meta)                    # gpt54-thinking (title "E") needs openai/gpt-5.4
    summary = {"counts": {"total": 1, "passed": 0, "failed": 1},
               "scenarios": [{"name": "E", "status": "fail"}]}
    res, _saved = _score(tmp_path, summary, {"gpt54-thinking": ISO._scen_summary("E", 1, 2)},
                         meta=meta)
    assert res["status"] == "completed" and res["harness_errors"] == {}
    assert res["pass_rate"] is None and res["total_scenarios"] == 0
    assert res["pass_rate_withheld_reason"] == (
        "no scenario could be scored: of the 1 scenario(s) the suite reported, 1 need a model "
        "provider gbench does not run")
    assert [r["id"] for r in res["infra_required_scenarios"]] == ["gpt54-thinking"]


def test_harness_errors_and_other_providers_together_are_both_named(tmp_path):
    meta = tmp_path / "meta"
    ISO._write_meta(meta)
    summary = {"counts": {"total": 2, "passed": 1, "failed": 1},
               "scenarios": [{"name": "A", "status": "pass"}, {"name": "E", "status": "fail"}]}
    scen = {"a": ISO._scen_summary("A", 1000, 1010),
            "gpt54-thinking": ISO._scen_summary("E", 2000, 2010, "fail")}
    res, _saved = _score(tmp_path, summary, scen, events=[(1005.0, "gemini_http_503")], meta=meta)
    assert res["pass_rate"] is None
    assert res["pass_rate_withheld_reason"] == (
        "no scenario could be scored: of the 2 scenario(s) the suite reported, 1 had a harness "
        "error fall in them and 1 need a model provider gbench does not run")


def test_a_suite_that_reports_no_scenario_has_no_pass_rate(tmp_path):
    res, _saved = _score(tmp_path, {"counts": {"total": 0, "passed": 0, "failed": 0},
                                    "scenarios": []}, {})
    assert res["pass_rate"] is None and res["total_scenarios"] == 0
    assert res["pass_rate_withheld_reason"] == \
        "no scenario could be scored: the suite reported none"


def test_an_unattributed_error_and_nothing_to_score_give_both_reasons(tmp_path):
    summary = {"counts": {"total": 1, "passed": 0, "failed": 1},
               "scenarios": [{"name": "A", "status": "fail"}]}
    res, _saved = _score(tmp_path, summary, {"a": ISO._scen_summary("A", 1000, 1010, "fail")},
                         events=[(1005.0, "model_http_503"), (5000.0, "gemini_http_429")])
    reason = res["pass_rate_withheld_reason"]
    assert res["pass_rate"] is None
    assert reason.startswith("harness errors that no scenario's window holds")
    assert "gemini_http_429 x1" in reason
    assert reason.endswith("no scenario could be scored: of the 1 scenario(s) the suite "
                           "reported, 1 had a harness error fall in them")


def test_a_scored_scenario_still_gets_a_pass_rate(tmp_path):
    """Withholding does not overreach: one clean scenario is enough for a number (0.0 is a real
    score when the model failed every scored scenario)."""
    summary = {"counts": {"total": 2, "passed": 0, "failed": 2},
               "scenarios": [{"name": "A", "status": "fail"}, {"name": "B", "status": "fail"}]}
    scen = {"a": ISO._scen_summary("A", 1000, 1010, "fail"),
            "b": ISO._scen_summary("B", 2000, 2010, "fail")}
    res, _saved = _score(tmp_path, summary, scen, events=[(1005.0, "model_unreachable")])
    assert res["pass_rate"] == 0.0 and res["pass_rate_withheld_reason"] is None
    assert (res["total_scenarios"], res["failed_scenarios"]) == (1, 1)


def test_an_error_in_the_window_of_a_summary_naming_no_scenario_withholds_the_pass_rate(
        tmp_path):
    """A scenario's own summary that names no scenario (or several) is keyed by its id, so its
    window belongs to no scenario the run lists: an error in it may have been any scenario's and
    withholds the pass rate (it used to be neither excluded nor withheld: pass_rate 50.0)."""
    summary = {"counts": {"total": 2, "passed": 1, "failed": 1},
               "scenarios": [{"name": "A title", "status": "fail"},
                             {"name": "B title", "status": "pass"}]}
    scen = {"a-id": {"run": {"startedAt": ISO._iso(1000), "finishedAt": ISO._iso(1060)},
                     "scenarios": []},
            "b-id": ISO._scen_summary("B title", 1100, 1160)}
    res, _saved = _score(tmp_path, summary, scen, events=[(1030.0, "model_http_503")])
    assert res["pass_rate"] is None and res["status"] == "completed_with_errors"
    assert res["unattributed_harness_errors"] == {"model_http_503": 1}
    assert res["harness_error_scenarios"] == []
    assert "model_http_503 x1" in res["pass_rate_withheld_reason"]
    # an error in a named scenario's window only is still tied to it and the rest is scored
    res, _saved = _score(tmp_path, summary, scen, events=[(1130.0, "model_http_503")])
    assert [r["title"] for r in res["harness_error_scenarios"]] == ["B title"]
    assert res["unattributed_harness_errors"] == {} and res["pass_rate"] == 0.0


def test_an_event_in_a_named_and_an_unnamed_window_is_listed_and_counted_once():
    by_title, unattributed = Q.attribute_events(
        [(5.0, "model_http_503"), (50.0, "gemini_http_500")], {"A": [(0.0, 10.0)]},
        unknown_spans=[(4.0, 6.0), (3.0, 7.0)])
    assert by_title == {"A": {"model_http_503": 1}}
    assert unattributed == {"model_http_503": 1, "gemini_http_500": 1}


def test_an_endpoint_down_for_the_whole_run_is_not_a_score(tmp_path, monkeypatch, gemini,
                                                          model_up):
    """A remote endpoint that serves the model under another name: every chat call of the suite
    gets vLLM's 404, every scenario is a harness-error scenario, and the run withholds its pass
    rate instead of reporting 0 (it used to be status completed with pass_rate 0)."""
    model_up.default = (404, NOT_SERVED)
    real_call = ISO._relay_call

    def suite_call(relay, path, **kw):   # the suite's own chat calls name the configured model
        if path.startswith("/model/") and kw.get("method", "POST") == "POST":
            kw["body"] = _chat()
        return real_call(relay, path, **kw)
    monkeypatch.setattr(ISO, "_relay_call", suite_call)
    summary = {"counts": {"total": 2, "passed": 0, "failed": 2},
               "scenarios": [{"name": "A", "status": "fail"}, {"name": "B", "status": "fail"}]}
    scen = {"a": ISO._scen_summary("A", 1000, 1010, "fail"),
            "b": ISO._scen_summary("B", 2000, 2010, "fail")}
    _r, res, seen, saved, _log = ISO._run_on_stubs(
        tmp_path, monkeypatch, gemini, model_up,
        [(1001, ISO.CHAT, "POST"), (2001, ISO.CHAT, "POST")], summary, scen, remote=True)
    assert seen["statuses"] == [404, 404]                 # relayed unchanged to the suite
    assert json.loads(model_up.requests[0]["body"])["model"] == SERVED
    assert res["harness_errors"] == {"model_http_404": 2}
    assert res["status"] == "completed_with_errors"
    assert [r["title"] for r in res["harness_error_scenarios"]] == ["A", "B"]
    assert res["pass_rate"] is None and json.loads(saved)["pass_rate"] is None
    assert "no scenario could be scored" in res["pass_rate_withheld_reason"]
    assert "model_http_404 x2" in res["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# model-endpoint 4xx: endpoint/config failures are harness errors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status, names_served, harness", [
    (500, False, True), (503, True, True), (401, False, True), (403, True, True),
    (408, False, True), (429, False, True),
    (404, True, True), (405, True, True), (409, True, True), (413, True, True),
    (415, True, True),
    (404, False, False), (415, False, False),
    (400, True, False), (400, False, False), (422, True, False),
    (200, True, False), (302, True, False),
])
def test_which_model_endpoint_answers_are_harness_errors(status, names_served, harness):
    assert Q.model_status_is_harness_error(status, names_served) is harness


@pytest.mark.parametrize("status", [404, 405, 413, 415])
def test_an_endpoint_or_config_4xx_on_the_suites_request_is_a_harness_error(relay_to, model_up,
                                                                           status):
    relay = relay_to(model_upstream=model_up.url + "/v1", model_id=SERVED)
    model_up.script = [(status, NOT_SERVED)]
    got, body = ISO._relay_call(relay, "/model/chat/completions", body=_chat())
    assert (got, body) == (status, NOT_SERVED)
    rep = relay.report()
    assert rep["harness_errors"] == {f"model_http_{status}": 1}
    assert [kind for _t, kind in rep["harness_error_events"]] == [f"model_http_{status}"]
    assert rep["model_endpoint_calls"][0]["statuses"] == {str(status): 1}


def test_a_404_on_get_models_is_a_harness_error(relay_to, model_up):
    relay = relay_to(model_upstream=model_up.url + "/v1", model_id=SERVED)
    model_up.script = [(404, b'{"detail": "Not Found"}')]
    assert ISO._relay_call(relay, "/model/models", method="GET")[0] == 404
    assert relay.report()["harness_errors"] == {"model_http_404": 1}


def test_the_endpoints_answer_to_the_request_content_is_the_models_side(relay_to, model_up):
    relay = relay_to(model_upstream=model_up.url + "/v1", model_id=SERVED)
    model_up.script = [(400, b'{"message": "maximum context length"}'),
                       (422, b'{"error": "Input validation error"}')]
    assert ISO._relay_call(relay, "/model/chat/completions", body=_chat())[0] == 400
    assert ISO._relay_call(relay, "/model/chat/completions", body=_chat())[0] == 422
    assert relay.report()["harness_errors"] == {}


def test_a_4xx_on_a_request_naming_another_model_is_the_models_side(relay_to, model_up):
    """Only the model's own tools, through the forwarder, send a request that names another
    model (or is not a chat request at all): what the endpoint answers it is the model's
    outcome, not a harness error."""
    relay = relay_to(model_upstream=model_up.url + "/v1", model_id=SERVED)
    model_up.script = [(404, NOT_SERVED), (415, b"{}"), (404, NOT_SERVED)]
    assert ISO._relay_call(relay, "/model/chat/completions", body=_chat("other/model"))[0] == 404
    assert ISO._relay_call(relay, "/model/completions", body=b"not json")[0] == 415
    assert ISO._relay_call(relay, "/model/chat/completions", body=b'{"messages": []}')[0] == 404
    rep = relay.report()
    assert rep["harness_errors"] == {}
    assert sum(sum(r["statuses"].values()) for r in rep["model_endpoint_calls"]) == 3
    # a 5xx stays a harness error whoever sent the request
    model_up.script = [(503, b"{}")]
    assert ISO._relay_call(relay, "/model/chat/completions", body=_chat("other/model"))[0] == 503
    assert relay.report()["harness_errors"] == {"model_http_503": 1}


def test_a_relay_not_told_the_model_counts_every_endpoint_4xx_but_400_and_422(relay_to,
                                                                            model_up):
    """A relay given no model_id and an upstream answering 404."""
    relay = relay_to(model_upstream=model_up.url + "/v1")
    model_up.script = [(404, NOT_SERVED), (400, b"{}"), (422, b"{}")]
    for _ in range(3):
        ISO._relay_call(relay, "/model/chat/completions")
    assert relay.report()["harness_errors"] == {"model_http_404": 1}


def test_the_run_gives_the_relay_the_model_id_it_configures(tmp_path, monkeypatch, gemini,
                                                           model_up):
    made = {}
    real_init = Q.KeyRelay.__init__

    def init(self, *a, **kw):
        made.update(kw)
        real_init(self, *a, **kw)
    monkeypatch.setattr(Q.KeyRelay, "__init__", init)
    _r, _res, seen, _saved, _log = ISO._run_on_stubs(
        tmp_path, monkeypatch, gemini, model_up, [],
        {"counts": {"total": 1, "passed": 1, "failed": 0}}, {}, remote=True)
    assert made["model_id"] == SERVED
    assert json.loads(seen["config"])["models"]["providers"]["vllm"]["models"][0]["id"] == SERVED
    assert f"vllm/{SERVED}" in seen["argv"]


# --------------------------------------------------------------------------- #
# the cloud metadata server guard for the QA container
# --------------------------------------------------------------------------- #
@pytest.fixture
def no_cached_verdicts(monkeypatch):
    monkeypatch.setattr(cg, "_VERDICTS", {})


def test_the_container_has_no_network_and_the_guard_needs_no_probe(tmp_path, monkeypatch,
                                                                  gemini, model_up,
                                                                  no_cached_verdicts):
    calls = []
    real = cg.require_metadata_blocked

    def spy(eval_name, image, network, docs_url=None):
        calls.append((eval_name, image, network, docs_url))
        return real(eval_name, image, network, docs_url)

    def no_probe(*a, **kw):
        raise AssertionError("--network none must not be probed")
    monkeypatch.setattr(cg, "require_metadata_blocked", spy)
    monkeypatch.setattr(cg, "_run_probe", no_probe)
    _r, res, seen, _saved, _log = ISO._run_on_stubs(
        tmp_path, monkeypatch, gemini, model_up, [],
        {"counts": {"total": 1, "passed": 1, "failed": 0}}, {})
    argv = seen["argv"]
    assert Q.argv_network(argv) == "none" == Q.QUALITY_CONTAINER_NETWORK
    assert argv.count("--network") == 1 and not any(a.startswith("--network=") for a in argv)
    assert calls == [("quality", "gbench-quality:test", "none", "docs/quality.md")]
    assert res["sandbox"]["metadata_guard"] == cg.NO_NETWORK
    assert cg.cached_verdicts() == {}


def test_a_container_with_a_network_is_probed_and_stopped_when_metadata_is_reachable(
        tmp_path, monkeypatch, gemini, model_up, no_cached_verdicts):
    """If the container ever got a network, the guard probes that network with the suite's own
    image before the container starts, and a reachable metadata server stops the run with the
    host firewall rule; the suite never runs."""
    probes = []

    def probe(eval_name, image, network, docs_url, names=()):
        probes.append((eval_name, image, network, docs_url))
        return cg.REACHABLE, "ipv4", {}
    monkeypatch.setattr(cg, "_run_probe", probe)
    monkeypatch.setattr(Q, "QUALITY_CONTAINER_NETWORK", "bridge")
    with pytest.raises(RuntimeError) as err:
        ISO._run_on_stubs(tmp_path, monkeypatch, gemini, model_up,
                          [(1001, ISO.CHAT, "POST")],
                          {"counts": {"total": 1, "passed": 1, "failed": 0}}, {})
    msg = str(err.value)
    assert "[INFRA REQUIRED] quality" in msg and cg.FIREWALL_COMMAND in msg
    assert cg.PERSIST_COMMAND in msg and "docs/quality.md" in msg
    assert probes == [("quality", "gbench-quality:test", "bridge", "docs/quality.md")]
    assert model_up.requests == [] and gemini.requests == []     # the container never started


def test_a_container_with_a_network_runs_when_the_metadata_server_is_blocked(
        tmp_path, monkeypatch, gemini, model_up, no_cached_verdicts):
    monkeypatch.setattr(cg, "_run_probe", lambda *a: (cg.UNREACHABLE, "", {}))
    monkeypatch.setattr(Q, "QUALITY_CONTAINER_NETWORK", "bridge")
    _r, res, seen, _saved, _log = ISO._run_on_stubs(
        tmp_path, monkeypatch, gemini, model_up, [],
        {"counts": {"total": 1, "passed": 1, "failed": 0}}, {})
    assert Q.argv_network(seen["argv"]) == "bridge"
    assert res["sandbox"]["metadata_guard"] == cg.UNREACHABLE


@pytest.mark.parametrize("argv, network", [
    (["docker", "run", "--network", "none", "img"], "none"),
    (["docker", "run", "--network=host", "img"], "host"),
    (["docker", "run", "img", "cmd"], None),
])
def test_argv_network_reads_the_docker_run_network(argv, network):
    assert Q.argv_network(argv) == network


# --------------------------------------------------------------------------- #
# the doc says what the code does
# --------------------------------------------------------------------------- #
def test_the_quality_doc_describes_the_withheld_rate_the_4xx_rule_and_the_guard():
    flat = " ".join((ROOT / "docs" / "quality.md").read_text(encoding="utf-8").split())
    assert "When no scenario is left to score, `pass_rate` is withheld (`null`), never `0`." in flat
    assert "`pass_rate_withheld_reason` says `no scenario could be scored: ...`" in flat
    assert ("and for any other 4xx but 400 and 422 on a request the suite makes (`GET /models`, "
            "or a body whose `model` is the model gbench configured)") in flat
    assert "any 4xx on a request that names another model" in flat
    assert ("or that falls in the window of a scenario summary that does not name exactly one "
            "scenario of the run") in flat
    assert "`container_guard.require_metadata_blocked`" in flat
    assert "`--network none` needs no probe" in flat and "`sandbox.metadata_guard`" in flat
    assert Q._MODEL_HARNESS_ERROR_STATUSES == (401, 403, 408, 429)
    assert Q._MODEL_CONTENT_REJECTION_STATUSES == (400, 422)
    assert "model_http_<status>` for a model-endpoint answer of 401, 403, 408, 429 or 5xx" in flat
