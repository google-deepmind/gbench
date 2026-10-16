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

"""gaia2 and toolbench: skipped capabilities, thinking-run stop strings, harness-error runs that
ARE validated as successes, and per-query model endpoint failures.

gaia2
  * a capability config that raised in ARE (ARE skips it and reports the others with exit code
    0), or a capability without a validated run outside harness errors, is named, sets
    completed_with_errors and is left out of the macro (ARE's formula would count it as 0.0);
  * a --thinking run sends no `stop` to the server (the server matches stop strings inside the
    reasoning); its request is streamed and closed at the first stop sequence in the answer, and
    ARE's client-side split still applies;
  * a run with a harness error stays out of the rates whatever ARE validated it as; the ones ARE
    validated as successes are counted and named.
toolbench
  * the image records every DFSDT step with a model request that failed at the endpoint, with
    its query; a query with a step that ended on such a failure is a `model_endpoint` harness
    error, left out of SoPR and SoWR; a step a later try answered is counted, its query scored.

Docker, Gemini, the network and the model endpoint are stubbed throughout.
"""

import importlib
import json
import os
import subprocess
import sys
import types

import pytest

from gbench.runners.eval_suites import gaia2 as G
from gbench.runners.eval_suites import toolbench as T

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "gbench", "docker"))
sys.path.insert(0, os.path.join(_REPO, "docker"))
sys.path.insert(0, _HERE)
import gaia2_run as GRUN  # noqa: E402
from test_gaia2_toolbench_judge_protocol_and_credentials import (_gaia2_run, _stub_are, _engine, tb,  # noqa: E402,F401
                                      _read, _flat)
from test_delegated_harness_request_settings import _UPSTREAM_CHATGPT  # noqa: E402


# =========================================================================== #
# gaia2: capability configs
# =========================================================================== #
def _row(sid, status="failed", config="search", phase="standard", run=1, **kw):
    return dict({"config": config, "phase_name": phase, "base_scenario_id": sid,
                 "run_number": run, "status": status, "has_exception": False}, **kw)


def _are_like_calculate(rows):
    """A stand-in of report_stats.calculate_statistics with ARE's macro formula: a capability
    without a validated run gets a success rate of 0.0."""
    per = {}
    for r in rows:
        key = GRUN.capability_key(r["config"], r["phase_name"])
        slot = per.setdefault(key, {"validated_runs": 0, "success": 0})
        if r["status"] != "no_validation":
            slot["validated_runs"] += 1
            slot["success"] += int(r["status"] == "success")
    for slot in per.values():
        slot["success_rate"] = (100.0 * slot["success"] / slot["validated_runs"]
                                if slot["validated_runs"] else 0.0)
    rates = [s["success_rate"] for s in per.values()]
    return {"per_capability": per,
            "global": {"macro_success_rate": sum(rates) / len(rates) if rates else 0.0,
                       "validated_runs": sum(s["validated_runs"] for s in per.values())}}


def test_capability_keys_follow_ares_statistics():
    assert GRUN.capability_key("mini", "agent2agent") == "mini_agent2agent"
    assert GRUN.capability_key("search", "standard") == "search"
    assert G.capability_key("mini") == "mini_standard" == GRUN.capability_key("mini", None)
    assert G.expected_capabilities("run", "") == list(G._CAPABILITY_CONFIGS)
    assert G.expected_capabilities("run", "mini") == ["mini_standard"]
    assert G.expected_capabilities("gaia2-run", "") == list(G._CAPABILITY_CONFIGS) + [
        "mini_agent2agent", "mini_noise"]
    # a group the launcher saw ARE run is expected too
    extra = {"configs_attempted": [{"capability": "demo", "ok": True}]}
    assert G.expected_capabilities("run", "", extra)[-1] == "demo"


def test_the_macro_leaves_out_a_capability_without_a_validated_run():
    rows = [_row("s1", "success", config="search"), _row("s2", "failed", config="search"),
            _row("t1", "no_validation", config="time")]
    out = GRUN.gbench_statistics(rows, {}, [], _are_like_calculate)
    g = out["statistics"]["global"]
    # ARE's formula: (50 + 0) / 2; over the capabilities with a validated run: 50
    assert g["macro_success_rate_are_formula"] == 25.0 and g["macro_success_rate"] == 50.0
    assert out["capabilities_without_validated_run"] == ["time"]
    # nothing to change when every capability has a validated run; non-ARE results untouched
    out = GRUN.gbench_statistics(rows[:2], {}, [], _are_like_calculate)
    assert out["capabilities_without_validated_run"] == []
    assert "macro_success_rate_are_formula" not in out["statistics"]["global"]
    assert GRUN.macro_over_validated(["not", "stats"]) == []


def test_the_hook_records_each_config_and_the_one_that_raised(monkeypatch, tmp_path):
    monkeypatch.setattr(GRUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("GAIA2_ENABLE_THINKING", "false")
    are = _stub_are(monkeypatch, lambda kw, M: M("x", "served"))

    def run_dataset(**kw):
        if kw["config"] == "time":
            raise ValueError("scenario build failed")
        return kw
    sys.modules["are.simulation.benchmark.cli"].run_dataset = run_dataset
    GRUN.install_are_hooks()
    cli = sys.modules["are.simulation.benchmark.cli"]
    assert cli.run_dataset(config="search")["config"] == "search"
    with pytest.raises(ValueError):
        cli.run_dataset(config="time")
    sub = sys.modules["are.simulation.benchmark.gaia2_submission"]
    sub.run_dataset(config="mini", phase_name="noise")
    recs = GRUN._read_jsonl(GRUN.CONFIGS, str(tmp_path))
    assert [(r["capability"], r["ok"]) for r in recs] == [
        ("search", True), ("time", False), ("mini_noise", True)]
    assert recs[1]["error"] == "ValueError" and "scenario build failed" in recs[1]["message"]
    out = GRUN.gbench_statistics([], {}, [], lambda k: {}, configs=recs)
    assert [c["capability"] for c in out["failed_configs"]] == ["time"]
    assert are is not None


def test_missing_capabilities_say_why():
    summary = {
        "per_capability": {"search": {"validated_runs": 3}, "ambiguity": {"validated_runs": 0}},
        "configs_attempted": [
            {"capability": c, "ok": c != "time"} for c in
            ("adaptability", "ambiguity", "search", "time")],
        "failed_configs": [{"capability": "time", "ok": False, "error": "ValueError",
                            "message": "boom"}]}
    got = {m["capability"]: m["why"] for m in G.missing_capabilities(
        summary, list(G._CAPABILITY_CONFIGS))}
    assert set(got) == {"adaptability", "ambiguity", "execution", "time"}
    assert got["time"].startswith("its config raised in ARE (ValueError: boom)")
    assert got["execution"] == "ARE did not run it"
    assert got["adaptability"] == "every scenario run of it hit a harness error"
    assert got["ambiguity"] == "none of its scenario runs could be validated"


def _full_capabilities(skip=()):
    caps = [c for c in G._CAPABILITY_CONFIGS if c not in skip]
    return {"per_capability": {c: {"success_rate": 40.0, "validated_runs": 3} for c in caps},
            "configs_attempted": [{"config": c, "phase_name": "standard", "capability": c,
                                   "ok": c not in skip, "error": None if c not in skip
                                   else "RuntimeError", "message": None if c not in skip
                                   else "loader failed"} for c in G._CAPABILITY_CONFIGS]}


def test_gaia2_a_full_run_with_every_capability_is_a_success(monkeypatch, tmp_path):
    res, _, _ = _gaia2_run(monkeypatch, tmp_path, summary=_full_capabilities())
    assert res["status"] == "success" and res["missing_capabilities"] == []
    assert res["expected_capabilities"] == list(G._CAPABILITY_CONFIGS)
    assert res["failed_configs"] == []


def test_gaia2_a_config_ares_skipped_is_named_and_sets_the_status(monkeypatch, tmp_path):
    summary = _full_capabilities(skip=("time",))
    summary["failed_configs"] = [c for c in summary["configs_attempted"] if not c["ok"]]
    res, _, _ = _gaia2_run(monkeypatch, tmp_path, summary=summary)
    assert res["status"] == "completed_with_errors"
    [miss] = res["missing_capabilities"]
    assert miss["capability"] == "time" and "RuntimeError: loader failed" in miss["why"]
    assert res["failed_configs"][0]["capability"] == "time"
    reason = res["leaderboard_comparable_reason"]
    assert "the macro averages 4 of the run's 5 capabilities, not the GAIA2 Overall: time: its " \
        "config raised in ARE (RuntimeError: loader failed)" in reason
    assert res["accuracy"] == 40.0     # ARE's rates over what was scored, named above


def test_gaia2_a_capability_with_only_harness_error_runs_is_named(monkeypatch, tmp_path):
    summary = _full_capabilities()
    del summary["per_capability"]["search"]
    summary.update({"n_harness_error_runs": 3, "harness_error_kinds": {"model_endpoint": 3}})
    res, _, _ = _gaia2_run(monkeypatch, tmp_path, summary=summary)
    assert res["status"] == "completed_with_errors"
    assert res["missing_capabilities"] == [
        {"capability": "search", "why": "every scenario run of it hit a harness error"}]


# =========================================================================== #
# gaia2: no server-side stop on a --thinking run
# =========================================================================== #
def test_a_thinking_run_sends_no_stop_to_the_server():
    stops = ["<end_action>", "Observation:"]
    off = GRUN.agent_settings({"GAIA2_ENABLE_THINKING": "false"})
    on = GRUN.agent_settings({"GAIA2_ENABLE_THINKING": "true", "GAIA2_AGENT_MAX_TOKENS": "65536"})
    unset = GRUN.agent_settings({})
    assert GRUN.agent_completion_kwargs(off, stops)["stop"] == stops
    assert GRUN.agent_completion_kwargs(unset, stops)["stop"] == stops
    assert "stop" not in GRUN.agent_completion_kwargs(on, stops)
    assert GRUN.server_stop(off) and GRUN.server_stop(unset) and not GRUN.server_stop(on)


@pytest.fixture
def thinking_hooked(monkeypatch, tmp_path):
    monkeypatch.setattr(GRUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("GAIA2_JUDGE_PORT", "18790")
    monkeypatch.setenv("GAIA2_ENABLE_THINKING", "true")
    monkeypatch.setenv("GAIA2_AGENT_MAX_TOKENS", "65536")
    mode = {"agent": "past_stop"}

    def chunk(finish=None, **delta):
        return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}}

    def responder(kw, ModelResponse):
        assert kw.get("stream") is True          # a --thinking agent request is streamed
        if mode["agent"] == "in_reasoning":
            chunks = [chunk(reasoning="I wait for the Observation: ..."), chunk(finish="stop")]
        else:
            chunks = [chunk(reasoning="plan the Observation: step"),
                      chunk(content='Thought: ok\nAction:\n{"a": 1}<end_'),
                      chunk(content="action>Observation: made up"),
                      chunk(content=" and more")]
        return types.SimpleNamespace(completion_stream=iter(chunks))

    are = _stub_are(monkeypatch, responder)
    GRUN.install_are_hooks()
    return types.SimpleNamespace(are=are, mode=mode, workdir=tmp_path)


def test_the_thinking_agent_request_has_no_stop_and_the_answer_is_still_split(thinking_hooked):
    are = thinking_hooked.are
    agent = _engine(are, "http://172.17.0.1:5678/v1")
    res, meta = agent([{"role": "user", "content": "q"}],
                      stop_sequences=["<end_action>", "Observation:"])
    sent = are.sent[-1]
    assert "stop" not in sent and sent["max_tokens"] == 65536
    assert sent["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True},
                                  "return_token_ids": True}
    assert sent["stream"] is True and sent["stream_options"] == GRUN.STREAM_OPTIONS
    assert res == 'Thought: ok\nAction:\n{"a": 1}'           # ARE's client-side split
    [call] = GRUN._read_jsonl(GRUN.AGENT_CALLS, str(thinking_hooked.workdir))
    # closed at the stop sequence in the answer, not at the one in the reasoning
    assert call["stop_sent"] is False and call["stopped_client_side"] is True
    assert call["ran_past_stop"] is False and call["stopped_in_reasoning"] is False
    assert meta["completion_tokens"] == 7


def test_a_thinking_reply_that_stopped_in_reasoning_is_counted(thinking_hooked):
    thinking_hooked.mode["agent"] = "in_reasoning"
    agent = _engine(thinking_hooked.are, "http://172.17.0.1:5678/v1")
    with pytest.raises(AssertionError):        # ARE's own assert on a reply with no content
        agent([{"role": "user", "content": "q"}], stop_sequences=["<end_action>"])
    [call] = GRUN._read_jsonl(GRUN.AGENT_CALLS, str(thinking_hooked.workdir))
    assert call["stopped_in_reasoning"] and call["server_parser"] is None
    # with thinking on it is the model's outcome: no harness-error record
    assert GRUN._read_jsonl(GRUN.CALL_FAILURES, str(thinking_hooked.workdir)) == []
    summary = GRUN.agent_call_summary(str(thinking_hooked.workdir))
    assert summary["stopped_in_reasoning"] == 1 and summary["stop_sent"] == 0


def test_gaia2_records_the_stop_choice_per_mode(monkeypatch, tmp_path):
    res, _, _ = _gaia2_run(monkeypatch, tmp_path, thinking=True, max_output_tokens=65536)
    assert res["agent_protocol"]["stop_sent_to_server"] is False
    assert "reasoning included" in res["agent_protocol"]["stop_note"]
    assert res["official_sampling"]["stop_sent_to_server"] is False
    assert "applied to the streamed answer by the client" in res["sampling"]
    assert res["agent_protocol"]["stop_applied_by"] == "client_stream"
    res, _, _ = _gaia2_run(monkeypatch, tmp_path)
    assert res["agent_protocol"]["stop_sent_to_server"] is True
    assert res["official_sampling"]["stop_sent_to_server"] is True


# =========================================================================== #
# gaia2: harness-error runs ARE validated as successes
# =========================================================================== #
def test_a_harness_error_run_stays_out_whatever_its_outcome_and_is_counted():
    rows = [_row("s1", "success"), _row("s2", "failed")]
    failures = {GRUN._run_key("search", "standard", "s1", "1"): [
        {"role": "judge", "no_verdict": True}]}
    out = GRUN.gbench_statistics(rows, failures, [],
                                 lambda keep: [r["base_scenario_id"] for r in keep])
    assert out["statistics"] == ["s2"]
    assert out["n_harness_error_success_runs"] == 1
    assert out["harness_error_success_kinds"] == {"judge_no_verdict": 1}


def test_gaia2_names_the_harness_error_runs_ares_validated_as_successes(monkeypatch, tmp_path):
    res, _, _ = _gaia2_run(monkeypatch, tmp_path, summary={
        "n_harness_error_runs": 2, "harness_error_kinds": {"judge_no_verdict": 2},
        "n_harness_error_success_runs": 1, "harness_error_success_kinds": {"judge_no_verdict": 1}})
    assert res["n_harness_error_success_runs"] == 1
    assert res["harness_error_success_kinds"] == {"judge_no_verdict": 1}
    assert ("1 of them ARE had validated as successes, left out all the same"
            in res["leaderboard_comparable_reason"])


def test_gaia2_label_and_docs_follow_the_launcher():
    assert G.HARNESS_ERRORS_VERSION == "7"
    df = _read("gbench", "docker", "gaia2.Dockerfile")
    assert f'LABEL {G.HARNESS_ERRORS_LABEL}="{G.HARNESS_ERRORS_VERSION}"' in df
    doc = _flat(_read("docs", "evals", "gaia2.md"))
    for needle in ("`missing_capabilities`", "`failed_configs`", "`configs_attempted`",
                   "`capabilities_without_validated_run`", "`macro_success_rate_are_formula`",
                   "`agent_protocol.stop_sent_to_server`", "`stopped_in_reasoning`",
                   "`ran_past_stop`",
                   "`n_harness_error_success_runs`", "`gbench_configs.jsonl`",
                   "_match_agent_oracle_event", "No server-side `stop` with `--thinking`"):
        assert needle in doc, needle
    assert "gbench_configs.jsonl" in G.ARTIFACT_FILES


# =========================================================================== #
# toolbench: the model client's record of failed model requests
# =========================================================================== #
_CHATGPT_FUNCTION = '''
class ChatGPTFunction:
    def __init__(self, model="m", openai_key="", base_url=None):
        self.model, self.openai_key, self.base_url = model, openai_key, base_url
        self.conversation_history = []
        self.TRY_TIME = 6

    def change_messages(self, messages):
        self.conversation_history = messages

    def parse(self, tools, process_id, key_pos=None, **args):
        for _ in range(self.TRY_TIME):
            response = chat_completion_request(self.openai_key, self.base_url,
                                               self.conversation_history, model=self.model)
            try:
                return response["choices"][0]["message"], 0, response["usage"]["total_tokens"]
            except BaseException:
                pass
        return {"role": "assistant", "content": str(response)}, -1, 0


if __name__ == "__main__":
    pass
'''


@pytest.fixture
def patched_client(monkeypatch, tmp_path):
    target = tmp_path / "chatgpt_function_model.py"
    target.write_text(_UPSTREAM_CHATGPT + _CHATGPT_FUNCTION)
    patch_src = _read("docker", "toolbench_patch_llm.py").replace(
        'PATH = "/stb/toolbench/inference/LLM/chatgpt_function_model.py"', f"PATH = {str(target)!r}")
    exec(compile(patch_src, "toolbench_patch_llm.py", "exec"), {"__name__": "__main__"})
    patched = target.read_text()
    assert patched.index("ChatGPTFunction.parse = _gbench_parse") \
        < patched.index('if __name__ == "__main__":')
    monkeypatch.setitem(sys.modules, "termcolor", types.SimpleNamespace(colored=lambda *a, **k: ""))
    monkeypatch.syspath_prepend(os.path.join(_REPO, "docker"))
    monkeypatch.syspath_prepend(str(tmp_path))
    log = tmp_path / "model_request_failures.jsonl"
    monkeypatch.setenv("TB_MODEL_FAILURE_LOG", str(log))
    monkeypatch.setenv("TB_GROUP", "G1_tool")
    monkeypatch.setenv("TB_ENABLE_THINKING", "false")
    monkeypatch.setenv("TB_REQUEST_LOG", str(tmp_path / "requests.jsonl"))
    for k in ("TB_ENDPOINT", "TB_TEMPERATURE", "TB_MAX_TOKENS", "TB_REQUEST_TIMEOUT_S"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delitem(sys.modules, "chatgpt_function_model", raising=False)
    mod = importlib.import_module("chatgpt_function_model")
    plan = []

    class Server503(Exception):
        status_code = 503

    class Rejected(Exception):
        status_code = 400

        def __str__(self):
            return ("This model's maximum context length is 100 tokens. However, you requested "
                    "2000 tokens. prompt contains 1000 input tokens")

    class _Resp:
        choices = [types.SimpleNamespace(finish_reason="stop", message=types.SimpleNamespace(
            content="hi", tool_calls=None))]

        def dict(self):
            return {"choices": [{"message": {"role": "assistant", "content": "hi"}}],
                    "usage": {"total_tokens": 3}}

    def create(**kw):
        step = plan.pop(0)
        if step == "503":
            raise Server503("unavailable")
        if step == "ctx":
            raise Rejected()
        return _Resp()

    class _OpenAI:
        def __init__(self, base_url=None, api_key=None):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    monkeypatch.setattr(mod, "OpenAI", _OpenAI)
    llm = mod.ChatGPTFunction(base_url="http://127.0.0.1:1/v1")
    llm._gbench_query_id = 42
    llm.change_messages([{"role": "user", "content": "q"}])
    return types.SimpleNamespace(mod=mod, llm=llm, plan=plan, log=log)


def _records(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def test_the_model_client_records_each_step_with_a_failed_request_and_its_query(patched_client):
    pc = patched_client
    pc.plan[:] = ["ok"]
    assert pc.llm.parse([], 0)[1] == 0
    assert _records(pc.log) == []                          # nothing failed: no record
    pc.plan[:] = ["503", "ok"]
    msg, code, _ = pc.llm.parse([], 0)
    assert code == 0 and msg["content"] == "hi"            # upstream's retry answered
    pc.plan[:] = ["503"] * 6
    msg, code, _ = pc.llm.parse([], 0)
    assert code == -1 and "unavailable" in msg["content"]  # the error text becomes the reply
    recovered, lost = _records(pc.log)
    assert recovered == {"group": "G1_tool", "query_id": "42", "tries": 2,
                         "endpoint_failures": 1, "content_rejections": 0, "unrecovered": False,
                         "rejected": False, "errors": ["Server503 503"]}
    assert lost["unrecovered"] is True and lost["tries"] == 6 and lost["endpoint_failures"] == 6


def test_a_context_rejection_left_to_upstream_is_not_an_endpoint_failure(patched_client):
    pc = patched_client
    pc.plan[:] = ["ctx"] * 6               # no --thinking budget: rejected_over_context
    assert pc.llm.parse([], 0)[1] == -1
    [rec] = _records(pc.log)               # recorded as a content rejection (HTTP 400)
    assert rec["endpoint_failures"] == 0 and rec["content_rejections"] == 6
    assert rec["unrecovered"] is False and rec["rejected"] is True
    pc.log.unlink()
    # a request outside a step (no parse) is left alone
    pc.plan[:] = ["503"]
    out = pc.mod.chat_completion_request("EMPTY", "http://127.0.0.1:1/v1",
                                         [{"role": "user", "content": "q"}], model="m")
    assert out == {"error": "unavailable", "total_tokens": 0} and _records(pc.log) == []


def test_the_client_patch_hands_the_query_id_to_the_model_client(tmp_path):
    src = ("import os\nclass rapidapi_wrapper:\n    def __init__(self, query_json):\n"
           '        self.input_description = query_json["query"]\n'
           "    def _step(self):\n"
           '                        headers = {"toolbench_key": self.toolbench_key}\n'
           "class pipeline_runner:\n    def method_converter(self, env):\n        if 1:\n"
           "            llm_forward = ChatGPTFunction(model=self.args.chatgpt_model, "
           "openai_key=openai_key, base_url=self.args.base_url)\n"
           "        return llm_forward\n")
    f = tmp_path / "rapidapi_multithread.py"
    f.write_text(src)
    patch = _read("docker", "toolbench_patch_client.py").replace(
        "/stb/toolbench/inference/Downstream_tasks/rapidapi_multithread.py", str(f))
    exec(compile(patch, "toolbench_patch_client.py", "exec"), {"__name__": "patch"})
    ns = {}
    exec(compile(f.read_text(), str(f), "exec"), ns)

    class _LLM:
        def __init__(self, **kw):
            pass
    ns["ChatGPTFunction"] = _LLM
    ns["openai_key"] = "EMPTY"
    runner = ns["pipeline_runner"]()
    runner.args = types.SimpleNamespace(chatgpt_model="m", base_url="u")
    env = types.SimpleNamespace(_gbench_query_id="q7")
    assert runner.method_converter(env)._gbench_query_id == "q7"


def test_model_failures_summary_and_the_model_endpoint_kind():
    recs = [{"group": "G1_tool", "query_id": "1", "unrecovered": True, "endpoint_failures": 6,
             "errors": ["InternalServerError 503"]},
            {"group": "G1_tool", "query_id": "2", "unrecovered": False, "endpoint_failures": 1,
             "errors": ["APITimeoutError"]},
            {"group": "", "query_id": "", "unrecovered": False, "endpoint_failures": 1}]
    s = T.summarize_model_failures(recs)
    assert s["steps_unrecovered"] == 1 and s["steps_recovered"] == 2
    assert s["failed_requests"] == 8 and s["unattributed_recovered"] == 1
    assert s["unattributed_unrecovered"] == 0
    assert s["per_query"][("G1_tool", "1")]["unrecovered_steps"] == 1
    cand = {"G1_tool": {"1": {}, "2": {}}}
    got = T.harness_error_queries({"G1_tool": ["1", "2"]}, cand, {}, {}, s["per_query"])
    assert got == {"G1_tool": {"1": ["model_endpoint"]}}
    assert "model_endpoint" in dict(T.QUERY_HARNESS_ERROR_LABELS)
    assert T.MODEL_FAILURES in T.ARTIFACT_FILES


def test_toolbench_leaves_out_a_query_whose_step_ended_on_an_endpoint_failure(tb):  # noqa: F811
    tb.state["model_failures"] = [
        {"group": "G1_tool", "query_id": "1", "unrecovered": True, "tries": 6,
         "endpoint_failures": 6, "errors": ["InternalServerError 503"]},
        {"group": "G2_category", "query_id": "2", "unrecovered": False, "tries": 2,
         "endpoint_failures": 1, "errors": ["APITimeoutError"]}]
    res = T.run_toolbench("m", "http://127.0.0.1:8000/v1", 2)
    assert res["status"] == "completed_with_errors"
    assert res["harness_error_query_kinds"] == {"model_endpoint": 1}
    assert res["harness_error_queries_by_group"] == {"G1_tool": ["1"]}
    assert res["queries_by_group"]["G1_tool"] == 1 and res["queries_by_group"]["G2_category"] == 2
    assert res["model_request_failures"]["steps_unrecovered"] == 1
    assert res["model_request_failures"]["steps_recovered"] == 1
    reason = res["leaderboard_comparable_reason"]
    assert ("1 a DFSDT step's model request whose last try (of upstream's 6) failed at the "
            "endpoint") in reason
    assert "1 DFSDT step(s) got a reply on a later try" in reason
    traces = {(t["group"], t["query_id"]): t for t in res["sample_traces"]}
    assert traces[("G1_tool", "1")]["harness_errors"] == ["model_endpoint"]
    assert traces[("G2_category", "2")]["model_request_failures"]["recovered_steps"] == 1
    assert "sopr_passes" in traces[("G2_category", "2")]          # scored
    assert os.path.isfile(os.path.join(res["artifacts_dir"], T.MODEL_FAILURES))


def test_toolbench_refuses_a_failed_step_it_cannot_tie_to_a_query(tb):  # noqa: F811
    tb.state["model_failures"] = [{"group": "", "query_id": "", "unrecovered": True,
                                   "endpoint_failures": 6}]
    with pytest.raises(RuntimeError, match="could not tie them to a query"):
        T.run_toolbench("m", "http://127.0.0.1:8000/v1", 2)


def test_toolbench_without_the_model_failure_record_publishes_no_score(tb, monkeypatch):  # noqa: F811
    real = T.read_model_failures
    monkeypatch.setattr(T, "read_model_failures",
                        lambda path: None if path.endswith(T.MODEL_FAILURES) else real(path))
    with pytest.raises(RuntimeError, match="wrote no model_request_failures.jsonl"):
        T.run_toolbench("m", "http://127.0.0.1:8000/v1", 2)


def test_the_image_writes_and_checks_the_model_failure_record():
    ep = _read("docker", "toolbench_entrypoint.sh")
    assert 'export TB_MODEL_FAILURE_LOG="/out/model_request_failures.jsonl"' in ep
    assert ': > "$TB_MODEL_FAILURE_LOG"' in ep
    df = _read("docker", "toolbench.Dockerfile")
    assert "assert m.ChatGPTFunction.parse.__name__ == '_gbench_parse'" in df
    assert f'LABEL {T.HARNESS_LABEL}="{T.HARNESS_VERSION}"' in df and T.HARNESS_VERSION == "5"
    doc = _flat(_read("docs", "evals", "toolbench.md"))
    for needle in ("`model_endpoint`", "`model_request_failures.jsonl`", "`model_request_failures`",
                   "`steps_recovered`", f"`{T.HARNESS_LABEL}={T.HARNESS_VERSION}`"):
        assert needle in doc, needle
    assert "named only by the count" not in doc


def test_the_gate_names_the_new_harness_label(monkeypatch):
    monkeypatch.setattr(T.shutil, "which", lambda _x: "/usr/bin/docker", raising=False)
    monkeypatch.setattr(T.subprocess, "run", lambda cmd, *a, **k: subprocess.CompletedProcess(
        cmd, 0, stdout="1", stderr=""))
    monkeypatch.setattr(T, "image_applies_request_policy", lambda image: True)
    ok, why = T.check_toolbench_prerequisites()
    assert not ok and "org.gbench.toolbench-harness=5" in why
