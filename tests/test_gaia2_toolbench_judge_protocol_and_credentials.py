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

"""gaia2 and toolbench: the upstream judge path, the agent protocol, harness errors, credentials.

gaia2
  * the judge reply reaches ARE's LLMChecker verbatim and its verdict marker is matched
    case-insensitively (gaia2-cli's judge; ARE 1.2.0's engine lowercased "[[True]]");
  * the agent requests carry the ARE paper's temperature 0.5, 16,000 tokens per turn, the stop
    sequences and the thinking switch, and return the measured generation time as metadata;
  * a reply the server moved into its reasoning field, and a soft check without a verdict, are
    harness errors;
  * the container runs on the bridge network with relays and placeholders only.
toolbench
  * tool calls the cached server failed or the simulator could not answer are harness errors,
    their queries left out of SoPR and SoWR;
  * SoWR drops comparisons without a verdict and withholds when no query has one;
  * the answering Gemini models, the cache-vs-simulator usage and per-query records are kept;
  * the container runs on the bridge network with relays and placeholders only.

Docker, Gemini, the network and the model endpoint are stubbed throughout.
"""

import asyncio
import json
import os
import subprocess
import sys
import types
import urllib.error

import pytest

from gbench.runners.eval_suites import base as B
from gbench.runners.eval_suites import gaia2 as G
from gbench.runners.eval_suites import toolbench as T

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOCKER_DIR = os.path.join(_REPO, "gbench", "docker")
sys.path.insert(0, _DOCKER_DIR)
sys.path.insert(0, os.path.join(_REPO, "docker"))
import gaia2_cascade_judge as GJ  # noqa: E402
import gaia2_run as GRUN  # noqa: E402
import toolbench_observations as OBS  # noqa: E402

# Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
REAL_KEY = "AIza" + "REALKEY00000000000000000000000000000"
PLACEHOLDER = "gbench-gemini-relay-" + "ab" * 24


def _read(*rel):
    with open(os.path.join(_REPO, *rel), encoding="utf-8") as f:
        return f.read()


def _flat(text):
    return " ".join(text.split())


class FakeGeminiRelay:
    instances = []

    def __init__(self, eval_name, bind_host, **kw):
        self.eval_name, self.bind_host, self.kw = eval_name, bind_host, kw
        self.placeholder = PLACEHOLDER
        self.stopped = False
        FakeGeminiRelay.instances.append(self)

    def start(self):
        return self

    def stop(self):
        self.stopped = True

    def client_env(self):
        origin = f"http://{self.bind_host}:1234"
        return {"GEMINI_API_KEY": self.placeholder, "GEMINI_API_KEYS": self.placeholder,
                "GEMINI_OPENAI_BASE_URL": origin + "/v1beta/openai",
                "GBENCH_GEMINI_RELAY_URL": origin}

    def record(self):
        return {"listens": f"http://{self.bind_host}:1234", "harness_errors": {}}

    def harness_error_count(self):
        return 0


class FakeEndpointRelay:
    TIMEOUT_S = 3600
    instances = []

    def __init__(self, upstream, bind_host, key=None, name="endpoint-relay", **kw):
        self.upstream, self.bind_host, self.key, self.name = upstream, bind_host, key, name
        self.stopped = False
        FakeEndpointRelay.instances.append(self)

    @property
    def url(self):
        return f"http://{self.bind_host}:5678/v1"

    def start(self):
        return self

    def stop(self):
        self.stopped = True

    def record(self):
        return {"upstream": self.upstream, "requests_relayed": 0}


class FakeDetokenizeRelay:
    """Stands in for gaia2.DetokenizeRelay (no socket)."""

    def __init__(self, bind_host, decode=None, name="gaia2-detokenize"):
        self.bind_host = bind_host

    def start(self):
        return self

    @property
    def url(self):
        return f"http://{self.bind_host}:7777/detokenize"

    def stop(self):
        pass

    def record(self):
        return {"listens": self.url, "requests": 0}


# =========================================================================== #
# gaia2: the launcher's pure pieces
# =========================================================================== #
def test_the_verdict_marker_is_matched_case_insensitively_as_gaia2_core_does():
    assert GRUN.verdict_of("- Evaluation: [[True]]", "[[True]]", "[[False]]") is True
    # ARE 1.2.0's engine turned the judge's "[[True]]" into "[[true]]": still a True vote
    assert GRUN.verdict_of("- Evaluation: [[true]]", "[[True]]", "[[False]]") is True
    assert GRUN.verdict_of("[[FALSE]]", "[[True]]", "[[False]]") is False
    assert GRUN.verdict_of("no marker", "[[True]]", "[[False]]") is None
    assert GRUN.verdict_of(None, "[[True]]", "[[False]]") is None
    # success is looked for first, as gaia2_core's LLMChecker does
    assert GRUN.verdict_of("[[success]] then [[failure]]", "[[Success]]", "[[Failure]]") is True


def test_the_agent_protocol_is_the_papers_by_default():
    s = GRUN.agent_settings({})
    assert s == {"temperature": 0.5, "max_tokens": 16000, "enable_thinking": None,
                 "timeout_s": 600, "max_iterations": 200}
    kw = GRUN.agent_completion_kwargs(s, ["<end_action>", "Observation:"])
    assert kw == {"max_tokens": 16000, "timeout": 600, "temperature": 0.5,
                  "stop": ["<end_action>", "Observation:"]}
    s = GRUN.agent_settings({"GAIA2_AGENT_TEMPERATURE": "", "GAIA2_AGENT_MAX_TOKENS": "65536",
                             "GAIA2_ENABLE_THINKING": "true", "GAIA2_AGENT_TIMEOUT_S": "2458"})
    kw = GRUN.agent_completion_kwargs(s, [])
    assert "temperature" not in kw and "stop" not in kw and kw["max_tokens"] == 65536
    assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert GRUN.AGENT_STOP_SEQUENCES == tuple(G.AGENT_STOP_SEQUENCES)


def test_the_request_timeout_scales_with_the_turn_budget():
    assert GRUN.agent_timeout_s(16000) == G.agent_timeout_s(16000) == 600
    assert GRUN.agent_timeout_s(None) == 600 and GRUN.agent_timeout_s(8000) == 600
    assert GRUN.agent_timeout_s(65536) == G.agent_timeout_s(65536) == 2458


def test_a_reply_moved_into_the_reasoning_field_is_a_harness_error_only_without_thinking():
    # The server-parser policy replaced this rule: text moved into the reasoning field on a run
    # without thinking is recovered as the model's output (no harness error); only deleted text
    # that cannot be recovered is one (server_parser_text_lost).
    text, event, src, _ = GRUN.apply_server_parser_policy(
        None, "Thought: x", {"completion_tokens": 9}, False, None, None)
    assert (text, event, src) == ("Thought: x", GRUN.SERVER_PARSER_MOVED_TEXT,
                                  "reasoning_then_content")
    text, event, _, _ = GRUN.apply_server_parser_policy(
        None, "Thought: x", {"completion_tokens": 9}, True, None, None)
    assert event is None and text is None      # with thinking on: the model's reasoning
    row = {"status": "failed", "has_exception": False}
    assert GRUN.classify_run(row, [{"role": "agent", "server_parser_text_lost": True,
                                    "endpoint_failure": False}]) == ["server_parser_text_lost"]
    assert GRUN.classify_run(row, [{"role": "judge", "no_verdict": True,
                                    "endpoint_failure": True}]) == ["judge_no_verdict"]
    timeout = {"status": "failed", "has_exception": True,
               "exception_type": GRUN.SCENARIO_TIMEOUT, "rationale": "x"}
    assert GRUN.classify_run(timeout, []) == []
    assert [k for k, _ in G.HARNESS_ERROR_LABELS] == list(GRUN.HARNESS_ERROR_KINDS)


def test_reply_info_reads_the_reasoning_field_and_the_usage():
    msg = types.SimpleNamespace(content=None, reasoning_content=None,
                                provider_specific_fields={"reasoning": "Thought: x"})
    usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15,
                                  completion_tokens_details=types.SimpleNamespace(
                                      reasoning_tokens=3))
    resp = types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg,
                                                                finish_reason="length")],
                                 usage=usage)
    info = GRUN.reply_info(resp)
    assert info["content_null"] and info["reasoning_chars"] == len("Thought: x")
    assert info["finish_reason"] == "length" and info["reasoning_tokens"] == 3


def test_agent_call_and_verdict_summaries(tmp_path):
    rows = [{"scenario_id": "s1", "run_number": 1, "config": "time", "phase_name": "standard",
             "finish_reason": "stop", "content_chars": 20, "completion_tokens": 7,
             "duration_s": 1.5},
            {"scenario_id": "s1", "run_number": 1, "config": "time", "phase_name": "standard",
             "finish_reason": "length", "content_chars": 0, "content_null": True,
             "completion_tokens": 16000, "duration_s": 60.0, "reply_in_reasoning": False}]
    (tmp_path / GRUN.AGENT_CALLS).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    checks = [{"scenario_id": "s1", "run_number": 1, "config": "time", "phase_name": "standard",
               "checker": "signature_checker", "verdict": True,
               "replies": [{"served_model": "gemini-3.7-flash", "vote": True, "reply": "[[True]]"}]}]
    (tmp_path / GRUN.JUDGE_VERDICTS).write_text(json.dumps(checks[0]) + "\n")
    a = GRUN.agent_call_summary(str(tmp_path))
    assert a["calls"] == 2 and a["null_content"] == 1 and a["cut_at_max_tokens"] == 1
    assert a["cut_without_content"] == 1 and a["max_completion_tokens"] == 16000
    j = GRUN.judge_verdict_summary(str(tmp_path))
    assert j["by_checker"] == {"signature_checker": {"true": 1, "false": 0, "no_verdict": 0}}
    assert j["answered_by"] == {"gemini-3.7-flash": 1}
    per = GRUN.per_run_calls(str(tmp_path))
    slot = per[GRUN._run_key("time", "standard", "s1", 1)]
    assert slot["agent_calls"] == 2 and slot["judge_models"] == {"gemini-3.7-flash": 1}


# --------------------------------------------------------------------------- #
# gaia2: the hooks against a stand-in of ARE 1.2.0's modules
# --------------------------------------------------------------------------- #
def _stub_are(monkeypatch, responder):
    """Stand-ins for the ARE and LiteLLM modules install_are_hooks patches."""
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__path__ = []
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
        parent, _, child = name.rpartition(".")
        if parent in sys.modules:
            monkeypatch.setattr(sys.modules[parent], child, m, raising=False)
        return m

    class APIError(Exception):
        pass

    class AuthenticationError(Exception):
        pass

    class Choices:
        def __init__(self, message, finish_reason="stop"):
            self.message, self.finish_reason = message, finish_reason

    class ModelResponse:
        def __init__(self, content, model, reasoning=None, finish_reason="stop"):
            msg = types.SimpleNamespace(content=content, reasoning_content=reasoning,
                                        provider_specific_fields={})
            self.choices = [Choices(msg, finish_reason)]
            self.model = model
            self.usage = types.SimpleNamespace(prompt_tokens=11, completion_tokens=7,
                                               total_tokens=18, completion_tokens_details=None)

    class LLMEngineException(Exception):
        pass

    class LiteLLMEngine:
        def __init__(self, model_config):
            self.model_config, self.mock_response = model_config, None

        def _convert_message_to_litellm_format(self, m):
            return m

        def __call__(self, messages, stop_sequences=[], **kw):  # noqa: B006
            return self.chat_completion(messages, stop_sequences, **kw)

    sent = []

    def completion(**kw):
        sent.append(kw)
        return responder(kw, ModelResponse)

    class ARESimulationAgent:
        def __init__(self, *a, max_iterations=80, **kw):
            self.max_iterations = max_iterations

    class LLMChecker:
        def __init__(self, engine, prompt_templates, num_votes=1, success_str="[[Success]]",
                     failure_str="[[Failure]]"):
            self.prompt_templates, self.num_votes = prompt_templates, num_votes
            self.success_str, self.failure_str = success_str, failure_str
            self.judge = lambda args: engine([{"role": "user", "content": "check"}])[0]

    for name in ("are", "are.simulation", "are.simulation.agents", "are.simulation.agents.llm",
                 "are.simulation.agents.llm.litellm", "are.simulation.agents.default_agent",
                 "are.simulation.benchmark", "are.simulation.validation",
                 "are.simulation.validation.utils", "litellm", "litellm.types"):
        mod(name)
    mod("are.simulation.logging_config", get_logger_scenario_id=lambda: "s1",
        get_logger_run_number=lambda: 2)
    mod("are.simulation.agents.llm.litellm.litellm_engine", LiteLLMEngine=LiteLLMEngine,
        completion=completion)
    mod("are.simulation.agents.llm.llm_engine", LLMEngineException=LLMEngineException)
    mod("are.simulation.agents.default_agent.are_simulation_main",
        ARESimulationAgent=ARESimulationAgent)
    mod("are.simulation.benchmark.cli", run_dataset=lambda **kw: kw,
        generate_and_save_reports=lambda *a, **k: None)
    mod("are.simulation.benchmark.gaia2_submission", run_dataset=lambda **kw: kw)
    templates = object()
    mod("are.simulation.validation.utils.llm_utils", LLMChecker=LLMChecker,
        SIGNATURE_CHECKER_TEMPLATES=templates)
    mod("litellm.exceptions", APIError=APIError, AuthenticationError=AuthenticationError)
    mod("litellm.types.utils", Choices=Choices, ModelResponse=ModelResponse)
    return types.SimpleNamespace(sent=sent, LiteLLMEngine=LiteLLMEngine, LLMChecker=LLMChecker,
                                 ARESimulationAgent=ARESimulationAgent, templates=templates)


@pytest.fixture
def hooked(monkeypatch, tmp_path):
    monkeypatch.setattr(GRUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("GAIA2_JUDGE_PORT", "18790")
    for k in ("GAIA2_AGENT_TEMPERATURE", "GAIA2_AGENT_MAX_TOKENS", "GAIA2_ENABLE_THINKING",
              "GAIA2_AGENT_TIMEOUT_S", "GAIA2_MAX_ITERATIONS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GAIA2_ENABLE_THINKING", "false")
    mode = {"agent": "ok"}

    def responder(kw, ModelResponse):
        if kw["api_base"].startswith("http://127.0.0.1:18790"):
            return ModelResponse("- Evaluation: [[True]]", "gemini-3.7-flash")
        if mode["agent"] == "reasoning":
            return ModelResponse(None, "served", reasoning="Thought: x Action: {}")
        return ModelResponse('Thought: ok\nAction:\n{"action_input": {"x": True}}<end_action>'
                             "Observation: made up", "served")

    are = _stub_are(monkeypatch, responder)
    GRUN.install_are_hooks()
    return types.SimpleNamespace(are=are, mode=mode, workdir=tmp_path)


def _engine(are, endpoint):
    return are.LiteLLMEngine(types.SimpleNamespace(model_name="openai/m", provider="local",
                                                   endpoint=endpoint, api_key=None))


def test_the_judge_reply_is_read_verbatim_and_its_true_verdict_passes(hooked):
    are = hooked.are
    judge = _engine(are, "http://127.0.0.1:18790/v1")
    res, meta = judge([{"role": "user", "content": "q"}])
    assert res == "- Evaluation: [[True]]" and meta is None       # no True -> true rewrite
    sent = are.sent[-1]
    assert not ({"temperature", "max_tokens", "stop", "extra_body"} & set(sent))
    checker = are.LLMChecker(judge, are.templates, success_str="[[True]]",
                             failure_str="[[False]]")
    assert checker({"x": "y"}) is True
    [rec] = GRUN._read_jsonl(GRUN.JUDGE_VERDICTS, str(hooked.workdir))
    assert rec["checker"] == "signature_checker" and rec["verdict"] is True
    assert rec["replies"][0]["served_model"] == "gemini-3.7-flash"
    assert rec["scenario_id"] == "s1" and rec["run_number"] == 2


def test_a_soft_check_without_a_verdict_is_a_judge_harness_error(hooked, monkeypatch):
    are = hooked.are
    judge = _engine(are, "http://127.0.0.1:18790/v1")
    checker = are.LLMChecker(judge, are.templates, success_str="[[Yes]]", failure_str="[[No]]")
    assert checker({}) is None
    [fail] = GRUN._read_jsonl(GRUN.CALL_FAILURES, str(hooked.workdir))
    assert fail["role"] == "judge" and fail["no_verdict"] and fail["error"] == "NoVerdict"
    by_run, loose = GRUN.read_call_failures(str(hooked.workdir))
    assert not loose and GRUN.classify_run({"status": "failed"}, next(iter(by_run.values()))) \
        == ["judge_no_verdict"]


def test_the_agent_request_carries_the_papers_protocol_and_returns_the_generation_time(hooked):
    are = hooked.are
    agent = _engine(are, "http://172.17.0.1:5678/v1")
    res, meta = agent([{"role": "user", "content": "q"}],
                      stop_sequences=["<end_action>", "Observation:"])
    sent = are.sent[-1]
    assert sent["temperature"] == 0.5 and sent["max_tokens"] == 16000 and sent["timeout"] == 600
    assert sent["stop"] == ["<end_action>", "Observation:"]
    # the thinking switch, and the token ids for the server-parser policy (metadata only)
    assert sent["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False},
                                  "return_token_ids": True}
    # ARE's own post-processing of the agent's reply is kept: True -> true, the stop split
    assert res == 'Thought: ok\nAction:\n{"action_input": {"x": true}}'
    assert meta["completion_duration"] >= 0 and meta["completion_tokens"] == 7
    assert meta["prompt_tokens"] == 11 and meta["total_tokens"] == 18
    [call] = GRUN._read_jsonl(GRUN.AGENT_CALLS, str(hooked.workdir))
    assert call["max_tokens"] == 16000 and call["finish_reason"] == "stop"
    a = are.ARESimulationAgent()
    assert a.max_iterations == 200
    assert are.ARESimulationAgent(max_iterations=12).max_iterations == 12


def test_a_reply_the_server_moved_into_reasoning_is_recorded_and_attributed(hooked):
    # The server-parser policy: the moved text is the model's output (recovered from the
    # reasoning field here, the stand-in returns no token ids), handed to ARE, and recorded.
    hooked.mode["agent"] = "reasoning"
    agent = _engine(hooked.are, "http://172.17.0.1:5678/v1")
    res, _meta = agent([{"role": "user", "content": "q"}], stop_sequences=["<end_action>"])
    assert res == "Thought: x Action: {}"
    assert GRUN._read_jsonl(GRUN.CALL_FAILURES, str(hooked.workdir)) == []
    [call] = GRUN._read_jsonl(GRUN.AGENT_CALLS, str(hooked.workdir))
    assert call["content_null"] and call["reasoning_chars"] > 0
    assert call["server_parser"] == GRUN.SERVER_PARSER_MOVED_TEXT
    assert call["recovered_from"] == "reasoning_then_content" and call["scenario_id"] == "s1"


# --------------------------------------------------------------------------- #
# gaia2: the judge proxy reaches Gemini only through the relay
# --------------------------------------------------------------------------- #
@pytest.fixture
def proxy_env(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", PLACEHOLDER)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("GBENCH_GEMINI_RELAY_URL", "http://172.17.0.1:4321")
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-3.8-flash,gemini-3.7-flash")
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", "1")
    monkeypatch.setenv("WILDCLAW_WORKDIR", str(tmp_path))
    return tmp_path


def test_the_judge_proxy_asks_the_relay_with_the_placeholder_at_temperature_0(proxy_env,
                                                                             monkeypatch):
    posts = []

    def post(url, payload, headers, timeout):
        posts.append((url, payload, headers))
        if payload["model"] == "gemini-3.8-flash":
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        return {"choices": [{"message": {"role": "assistant", "content": "[[True]]"}}]}

    monkeypatch.setattr(GJ.cascade, "_post_json", post)
    reply = GJ.judge_chat_completion({"model": "openai/gbench-cascade", "temperature": 0.7,
                                      "messages": [{"role": "user", "content": "judge"}]})
    assert reply["model"] == "gemini-3.7-flash"
    assert all(u == "http://172.17.0.1:4321/v1beta/openai/chat/completions" for u, _, _ in posts)
    assert all(h["Authorization"] == f"Bearer {PLACEHOLDER}" for _, _, h in posts)
    assert posts[-1][1]["temperature"] == 0.0 and "max_tokens" not in posts[-1][1]
    [line] = (proxy_env / GJ.JUDGE_CALLS).read_text().splitlines()
    rec = json.loads(line)
    assert rec["served_model"] == "gemini-3.7-flash" and rec["ok"] and rec["reply"] == "[[True]]"


def test_the_judge_proxy_refuses_a_real_key_before_any_call(proxy_env, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", REAL_KEY)
    monkeypatch.setattr(GJ.cascade, "_post_json", lambda *a, **k: pytest.fail("called"))
    with pytest.raises(Exception, match="placeholder"):
        GJ.judge_chat_completion({"messages": [{"role": "user", "content": "judge"}]})
    assert GJ.main() == 7


def test_the_judge_proxy_needs_the_cascade_from_gbench(proxy_env, monkeypatch):
    monkeypatch.delenv("GBENCH_JUDGE_MODELS")
    assert GJ.main() == 9


# --------------------------------------------------------------------------- #
# gaia2: the runner (container, network, credentials, result)
# --------------------------------------------------------------------------- #
def _gaia2_run(monkeypatch, tmp_path, summary=None, thinking=False, env=None, **kwargs):
    FakeGeminiRelay.instances.clear()
    FakeEndpointRelay.instances.clear()
    monkeypatch.setattr(G, "check_gaia2_prerequisites", lambda: (True, ""))
    guard = []
    monkeypatch.setattr(G.container_guard, "require_metadata_blocked",
                        lambda *a, **k: guard.append((a, k)) or "unreachable")
    monkeypatch.setattr(G.gemini_relay, "network_gateway", lambda *a, **k: "172.17.0.1")
    monkeypatch.setattr(G.gemini_relay, "GeminiRelay", FakeGeminiRelay)
    monkeypatch.setattr(G, "EndpointRelay", FakeEndpointRelay)
    monkeypatch.setattr(G, "DetokenizeRelay", FakeDetokenizeRelay)
    monkeypatch.setenv("GEMINI_API_KEY", REAL_KEY)
    monkeypatch.setenv("GBENCH_GAIA2_MODEL_API_KEY", "sk-endpoint-secret-0000")
    monkeypatch.setenv("HF_TOKEN", "hf_secret_token_000000")
    monkeypatch.setenv("GBENCH_RESULTS_DIR", str(tmp_path / "results"))
    for k in ("GBENCH_SHARD", "GBENCH_GAIA2_CONFIG", "GBENCH_GAIA2_LIMIT", "GBENCH_GAIA2_MODE",
              "GEMINI_OPENAI_BASE_URL", "GBENCH_GAIA2_TEMPERATURE", "GBENCH_GAIA2_NUM_RUNS",
              "GEMINI_API_KEYS", "GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL"):
        monkeypatch.delenv(k, raising=False)
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    wd = tmp_path / "gaia2_wd"
    wd.mkdir(exist_ok=True)
    monkeypatch.setattr(G.tempfile, "mkdtemp",
                        lambda prefix="", dir=None: str(wd) if prefix == "gbench_gaia2_"
                        else os.path.join(dir, "run_x") if os.makedirs(
                            os.path.join(dir, "run_x"), exist_ok=True) is None else "")
    full = {"harness_accounting": True, "macro_success_rate": 40.0, "micro_success_rate": 38.0,
            "total_scenarios": 5, "n_runs": 15, "total_runs": 15, "validated_runs": 12,
            "success_runs": 5, "no_validation_runs": 0, "agent_invoked": True,
            "per_capability": {}, "harness_events": {}, "n_harness_error_runs": 0,
            "harness_error_kinds": {}, "n_timeout_runs": 1, "n_scenarios_run": 5,
            "runs": [{"scenario_id": "s1", "run_number": 1, "status": "failed",
                      "exception_type": "ScenarioTimeoutError", "harness_errors": [],
                      "judge_models": {"gemini-3.7-flash": 2}}],
            "gemini_usage": {"judge": {"gemini-3.7-flash": 9, "failed": 1}},
            "agent_protocol": {"temperature": 0.5}}
    full.update(summary or {})
    runs = []

    def fake_run(cmd, *a, **k):
        if cmd[:3] == ["docker", "run", "--rm"]:
            runs.append((cmd, k))
            (wd / "gaia2_summary.json").write_text(json.dumps(full))
            (wd / "gbench_judge_verdicts.jsonl").write_text('{"checker": "tone_checker"}\n')
            return subprocess.CompletedProcess(cmd, 0, stdout="[gaia2] Retrying request",
                                               stderr=f"key {REAL_KEY}")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    monkeypatch.setattr(G.subprocess, "run", fake_run)
    res = G.run_gaia2("m", "http://127.0.0.1:9123/v1", enable_thinking=thinking, **kwargs)
    return res, runs, guard


def test_gaia2_runs_on_the_guarded_bridge_with_relays_and_no_real_credential(monkeypatch,
                                                                             tmp_path):
    res, [(cmd, kw)], guard = _gaia2_run(monkeypatch, tmp_path)
    assert cmd[cmd.index("--network") + 1] == "bridge" and "host" not in cmd
    [(args, kwargs)] = guard
    # no name to resolve: the relays are reached by the gateway address, the datasets offline
    assert args[2] == "bridge" and not kwargs["resolve"]
    env = kw["env"]
    text = " ".join(cmd) + json.dumps(env)
    for secret in (REAL_KEY, "sk-endpoint-secret-0000", "hf_secret_token_000000"):
        assert secret not in text
    assert env["GEMINI_API_KEY"] == PLACEHOLDER and env["GEMINI_API_KEYS"] == PLACEHOLDER
    names = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-e"]
    assert "HF_TOKEN" not in names and "GBENCH_MODEL_API_KEY" not in names
    assert "GBENCH_MODEL_BASE_URL=http://172.17.0.1:5678/v1" in cmd
    [ep] = FakeEndpointRelay.instances
    assert ep.upstream == "http://127.0.0.1:9123/v1" and ep.key == "sk-endpoint-secret-0000"
    assert ep.stopped and FakeGeminiRelay.instances[0].stopped
    assert f"GBENCH_JUDGE_MODELS={','.join(B.judge_cascade())}" in cmd
    for want in ("GAIA2_AGENT_TEMPERATURE=0.0", "GAIA2_AGENT_MAX_TOKENS=16000",
                 "GAIA2_ENABLE_THINKING=false", "GAIA2_AGENT_TIMEOUT_S=600",
                 "GAIA2_MAX_ITERATIONS=200", f"GAIA2_HOST_UID={os.getuid()}"):
        assert want in cmd, want
    assert res["docker_network"] == "bridge" and res["gemini_relay"]["listens"]


def test_gaia2_result_records_protocol_counts_models_and_traces(monkeypatch, tmp_path):
    res, _, _ = _gaia2_run(monkeypatch, tmp_path)
    assert res["status"] == "success" and res["accuracy"] == 40.0
    assert res["official_sampling"]["temperature"] == 0.5
    assert res["official_sampling"]["max_tokens"] == 16000
    assert res["official_sampling"]["max_tokens_applied"] is True
    assert res["agent_protocol"]["max_iterations"] == 200
    assert res["agent_protocol"]["stop"] == ["<end_action>", "Observation:"]
    assert res["total_questions"] == 15 and res["effective_n"] == 12
    assert res["correct_answers"] == 5 and res["count_unit"] == "scenario runs"
    assert res["headline_metrics"]["macro"] == 40.0 and res["n_timeout_runs"] == 1
    assert res["gemini_models_answered"] == {"gemini-3.7-flash": 9}
    assert res["sample_traces"][0]["judge_models"] == {"gemini-3.7-flash": 2}
    reason = res["leaderboard_comparable_reason"]
    assert "standard phase only (3 run(s) per scenario)" in reason
    assert "single-run" not in reason and "case-insensitively" in reason
    # the orchestrator's output and small records are kept, redacted
    kept = res["artifacts_dir"]
    log = open(os.path.join(kept, "gaia2_orchestrator.log")).read()
    assert "Retrying request" in log and REAL_KEY not in log
    assert os.path.isfile(os.path.join(kept, "gbench_judge_verdicts.jsonl"))


def test_gaia2_thinking_run_is_never_capped_below_the_flag(monkeypatch, tmp_path):
    res, [(cmd, _)], _ = _gaia2_run(monkeypatch, tmp_path, thinking=True,
                                    max_output_tokens=65536)
    assert "GAIA2_AGENT_MAX_TOKENS=65536" in cmd and "GAIA2_ENABLE_THINKING=true" in cmd
    assert "GAIA2_AGENT_TIMEOUT_S=2458" in cmd
    assert res["official_sampling"]["max_tokens_applied"] is False
    assert B.REASONING_CAP_NOTE in res["leaderboard_comparable_reason"]


def test_gaia2_samples_the_agent_at_one_with_thinking_and_zero_without(monkeypatch, tmp_path):
    # The maintainer's rule: a --thinking run samples at 1.0 and a no-think run at 0.0, never the
    # paper's 0.5; the orchestrator's agent requests carry it (gaia2_run.agent_completion_kwargs)
    # and the result names the difference.
    res, [(cmd, _)], _ = _gaia2_run(monkeypatch, tmp_path, thinking=True)
    assert "GAIA2_AGENT_TEMPERATURE=1.0" in cmd and "GAIA2_AGENT_TEMPERATURE=0.5" not in cmd
    assert (res["temperature"], res["temperature_source"]) == (1.0, "thinking")
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert "agent temperature 1.0 set by thinking (the ARE paper runs 0.5)" in (
        res["leaderboard_comparable_reason"])
    assert "temperature 1.0 (thinking; the paper's 0.5 is recorded, not sent)" in res["sampling"]
    on = GRUN.agent_settings({"GAIA2_AGENT_TEMPERATURE": "1.0", "GAIA2_ENABLE_THINKING": "true"})
    assert GRUN.agent_completion_kwargs(on, ["<end_action>"])["temperature"] == 1.0
    res, [(cmd, _)], _ = _gaia2_run(monkeypatch, tmp_path, thinking=False)
    assert "GAIA2_AGENT_TEMPERATURE=0.0" in cmd and "GAIA2_AGENT_TEMPERATURE=0.5" not in cmd
    assert (res["temperature"], res["temperature_source"]) == (0.0, "default")
    assert res["official_sampling"]["temperature_matches_official"] is False
    assert "agent temperature 0.0 set by default (the ARE paper runs 0.5)" in (
        res["leaderboard_comparable_reason"])
    # the in-image launcher reads the 0.0 and sends it on the agent's request
    off = GRUN.agent_settings({"GAIA2_AGENT_TEMPERATURE": "0.0", "GAIA2_ENABLE_THINKING": "false"})
    assert off["temperature"] == 0.0
    assert GRUN.agent_completion_kwargs(off, ["<end_action>"])["temperature"] == 0.0


def test_gaia2_new_harness_kinds_set_the_status_and_are_named(monkeypatch, tmp_path):
    res, _, _ = _gaia2_run(monkeypatch, tmp_path, summary={
        "n_harness_error_runs": 3, "harness_error_kinds": {"server_parser_text_lost": 2,
                                                          "judge_no_verdict": 1}})
    assert res["status"] == "completed_with_errors"
    reason = res["leaderboard_comparable_reason"]
    assert ("2 the server's tool-call parser deleted an agent reply and its text could not be "
            "recovered from token ids, 1 a judge reply held no verdict") in reason


def test_gaia2_num_runs_reaches_the_standard_run(monkeypatch, tmp_path):
    res, [(cmd, _)], _ = _gaia2_run(monkeypatch, tmp_path, env={"GBENCH_GAIA2_NUM_RUNS": "1"})
    assert "GAIA2_NUM_RUNS=1" in cmd and res["num_runs_per_scenario"] == 1
    assert "standard phase only (1 run(s) per scenario)" in res["leaderboard_comparable_reason"]
    src = _read("gbench", "docker", "gaia2_run.py")
    assert 'cmd += ["--num_runs", os.environ["GAIA2_NUM_RUNS"]]' in src


def test_gaia2_image_gate_names_the_new_label(monkeypatch):
    monkeypatch.setattr(G.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(G.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 0, stdout="1", stderr=""))
    monkeypatch.setattr(G, "_docker_image_exists", lambda _i: True)
    ok, why = G.check_gaia2_prerequisites()
    assert not ok and f"{G.HARNESS_ERRORS_LABEL}={G.HARNESS_ERRORS_VERSION}" in why
    assert G.HARNESS_ERRORS_VERSION == "7"
    df = _read("gbench", "docker", "gaia2.Dockerfile")
    assert f'LABEL {G.HARNESS_ERRORS_LABEL}="{G.HARNESS_ERRORS_VERSION}"' in df


def test_gaia2_entrypoint_hands_the_workdir_back_and_keeps_the_relay_from_are():
    ep = _read("gbench", "docker", "gaia2_entrypoint.sh")
    assert 'chown -R "${GAIA2_HOST_UID}:${GAIA2_HOST_GID}" "$GAIA2_WORKDIR"' in ep
    assert ("env -u GEMINI_API_KEYS -u GEMINI_API_KEY -u GBENCH_GEMINI_RELAY_URL "
            "-u GEMINI_OPENAI_BASE_URL") in ep
    src = _read("gbench", "docker", "gaia2_cascade_judge.py")
    assert "base_check=relay_openai_base" in src and "cascade._gemini_base =" not in src


def test_gaia2_keep_artifacts_and_remove_workdir(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", REAL_KEY)
    wd = tmp_path / "wd"
    wd.mkdir()
    (wd / "output.jsonl").write_text(f'{{"k": "{REAL_KEY}"}}\n')
    dest = tmp_path / "dest"
    dest.mkdir()
    assert G.keep_artifacts(str(wd), str(dest), f"log {REAL_KEY}") == str(dest)
    assert REAL_KEY not in (dest / "output.jsonl").read_text()
    assert REAL_KEY not in (dest / "gaia2_orchestrator.log").read_text()
    assert G.remove_workdir(str(wd)) == [] and not wd.exists()


# =========================================================================== #
# toolbench: the image's records
# =========================================================================== #
def _calls(*recs):
    return [dict({"group": "G1_tool", "type": "cached_real_response"}, **r) for r in recs]


def test_server_calls_summary_counts_cache_proxy_simulator_and_failures():
    recs = _calls({"query_id": "1"},
                  {"query_id": "1", "type": "fake_response", "simulator_models": ["gemini-3.7-flash"]},
                  {"query_id": "2", "type": "real_response"},
                  {"query_id": "2", "type": "server_error", "server_error": True,
                   "stage": "simulator"},
                  {"query_id": "3", "type": "server_error", "server_error": True,
                   "stage": "real_api_proxy"},
                  {"query_id": "4", "type": "fake_response", "simulator_failed_reply": True,
                   "simulator_models": ["gemini-3.8-flash"]},
                  {"query_id": None, "type": "server_error", "server_error": True,
                   "stage": "cache"})
    s = T.summarize_server_calls(recs)
    assert s["usage"] == {"tool_calls": 7, "cached": 1, "real_api": 1, "simulated": 2,
                          "simulator_failed_replies": 1, "server_errors": 3,
                          "tool_input_parse_errors": 0,
                          "simulator_replies_cut_at_max_tokens": 0,
                          "simulator_empty_replies_at_max_tokens": 0,
                          "simulator_outages_after_max_tokens": 0}
    assert s["server_errors_by_stage"] == {"real_api_proxy": 1, "server": 1, "simulator": 1}
    assert s["simulator_models_used"] == {"gemini-3.7-flash": 1, "gemini-3.8-flash": 1}
    assert s["unattributed_failures"] == 1
    assert s["per_query"][("G1_tool", "2")]["server_errors"] == 1
    assert s["per_query"][("G1_tool", "1")]["by_type"] == {"cached_real_response": 1,
                                                           "fake_response": 1}


def test_harness_error_queries_join_the_server_record_the_trees_and_missing_answers():
    s = T.summarize_server_calls(_calls({"query_id": "2", "server_error": True,
                                         "type": "server_error", "stage": "simulator"}))
    obs = {"G1_tool": {"3": {"server_error_observations": 1},
                       "4": {"simulator_failed_observations": 2},
                       "5": {"timeout_observations": 1}, "1": {"tool_calls": 4}}}
    cand = {"G1_tool": {q: {} for q in ("1", "2", "3", "4", "5")}}
    out = T.harness_error_queries({"G1_tool": ["1", "2", "3", "4", "5", "6"]}, cand,
                                  s["per_query"], obs)
    assert out == {"G1_tool": {"2": ["server_error"], "3": ["server_error"],
                               "4": ["simulator_failed"], "5": ["tool_call_timeout"],
                               "6": ["no_answer"]}}
    assert T.describe_query_harness_errors({"no_answer": 1}).startswith("1 the harness wrote")


def test_the_observation_scanner_reads_the_whole_search_tree(tmp_path):
    tree = {"tree": {"tree": {"observation": "", "children": [
        {"observation": '{"error": "request invalid, data error. status_code=500"}',
         "children": [{"observation": '{"error": "", "response": "ok"}', "children": []}]},
        {"observation": '{"error": "Timeout error...", "response": ""}', "children": []}]}},
        "answer_generation": {"train_messages": [[{"content": "request invalid, data error"}]]}}
    assert OBS.count_answer(tree) == {"tool_calls": 3, "server_error_observations": 1,
                                      "simulator_failed_observations": 0,
                                      "timeout_observations": 1}
    (tmp_path / "G1_tool").mkdir()
    (tmp_path / "G1_tool" / "42_DFS_woFilter_w2.json").write_text(json.dumps(tree))
    (tmp_path / "G1_tool" / "43_DFS_woFilter_w2.json").write_text("{not json")
    out = tmp_path / "obs.json"
    assert OBS.main(["x", str(tmp_path), "DFS_woFilter_w2", str(out), "G1_tool", "G2_x"]) == 0
    data = json.loads(out.read_text())["groups"]
    assert data["G1_tool"]["42"]["server_error_observations"] == 1
    assert data["G1_tool"]["43"]["unreadable"] and data["G2_x"] == {}


# --------------------------------------------------------------------------- #
# toolbench: the image patches, applied to a stand-in of upstream's files
# --------------------------------------------------------------------------- #
_UPSTREAM_MAIN = (
    "import json, os, time\n"
    "import requests\n"
    "def write_log(request, response, type):\n"
    "    LOGS.append(type)\n"
    "\n"
    "@app.post('/virtual')\n"
    "# @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(1))\n"
    "def get_virtual_response(request: Request, info: Info):\n"
    "    if info.tool_input == 'cached':\n"
    "        write_log(request=info, response={}, type='cached_real_response')\n"
    "        return {'error': '', 'response': 'cached'}\n"
    "    headers = {}\n"
    "    data = {}\n"
    "    real_response = requests.post(CONFIG['toolbench_url'], headers=headers, "
    "data=json.dumps(data))\n"
    "    result = fake_response_function_chat([], info.tool_input, {})\n"
    "    write_log(request=info, response=result, type='fake_response')\n"
    "    return json.loads(result)\n"
    "\n"
    "def fake_response_function_chat(api_example, tool_input, api_doc):\n"
    "    system_prompt = {'role': 'system', 'content': 'sim'}\n"
    "    user_prompt = {'role': 'user', 'content': str(tool_input)}\n"
    "    client = OpenAI(\n"
    "        api_key = OPENAI_API_KEY,\n"
    "        base_url = OPENAI_API_BASE,\n"
    "    )\n"
    "    max_retries = 3 \n"
    "    flag = False\n"
    "    for attempt in range(max_retries):\n"
    "        response = client.chat.completions.create(\n"
    "            model = CONFIG['model'],\n"
    "            messages=[system_prompt, user_prompt],\n"
    "            max_tokens = 1024,\n"
    "            temperature=CONFIG['temperature'],\n"
    "            response_format={\"type\": \"json_object\"},\n"
    "        )\n"
    "        result = response.choices[0].message.content\n"
    "        if result.startswith('{'):\n"
    "            flag = True\n"
    "            break\n"
    "    return result if flag else json.dumps({'error': 'Failed to generate fake response', "
    "'response': ''})\n"
    "\n"
    "if __name__ == \"__main__\":\n"
    "    pass\n"
)


def _patched_server(tmp_path, monkeypatch, sim):
    main = tmp_path / "main.py"
    main.write_text(_UPSTREAM_MAIN)
    patch = _read("docker", "toolbench_patch_simulator.py").replace("/stb/server/main.py",
                                                                    str(main))
    exec(compile(patch, "toolbench_patch_simulator.py", "exec"), {"__name__": "patch"})
    patched = main.read_text()
    routes = {}

    class App:
        def post(self, path):
            def deco(fn):
                routes[path] = fn
                return fn
            return deco

    class NotFound(Exception):
        status_code = 404

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None, max_retries=None):
            self.key, self.max_retries = api_key, max_retries
            outer = self

            class Completions:
                def create(self, **kw):
                    sim.append((outer.key, kw))
                    if kw["model"] == "m404":
                        raise NotFound("model not found")
                    if MODE["sim"] == "down":
                        raise RuntimeError("503")
                    msg = types.SimpleNamespace(content='{"error": "", "response": "sim"}')
                    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])
            self.chat = types.SimpleNamespace(completions=Completions())

    class FakeRequests:
        @staticmethod
        def post(url, timeout=None, **kw):
            if MODE["proxy"] == "down":
                raise ConnectionError("refused")
            PROXY.append(timeout)
            return types.SimpleNamespace(status_code=401)

    MODE = {"sim": "ok", "proxy": "ok"}
    PROXY = []
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("TB_SERVER_LOG", str(log))
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "m404,m2")
    monkeypatch.setenv("TB_SIMULATOR_ROUNDS", "2")
    monkeypatch.setenv("TB_SIMULATOR_BACKOFF", "0")
    monkeypatch.delenv("TB_REAL_API_TIMEOUT_S", raising=False)
    import time as _time
    ns = {"app": App(), "Request": object, "Info": object, "OpenAI": FakeOpenAI,
          "OPENAI_API_KEY": PLACEHOLDER, "OPENAI_API_BASE": "http://172.17.0.1:1/v1beta/openai/",
          "CONFIG": {"model": "m0", "temperature": 0, "toolbench_url": "http://proxy"},
          "LOGS": [], "__name__": "main"}
    code = patched.replace("import requests\n", "")
    exec(compile(code, "main.py", "exec"), ns)
    ns["requests"] = FakeRequests
    ns["time"] = _time
    return types.SimpleNamespace(route=routes["/virtual"], mode=MODE, log=log, proxy=PROXY,
                                 patched=patched)


def _req(group="G1_tool", qid="7"):
    return types.SimpleNamespace(headers={"x-gbench-group": group, "x-gbench-query-id": qid})


def _info(tool_input):
    return types.SimpleNamespace(category="c", tool_name="t", api_name="a",
                                 tool_input=tool_input)


def test_the_server_patch_records_every_call_and_the_stage_a_failure_happened_in(tmp_path,
                                                                                monkeypatch):
    sim = []
    srv = _patched_server(tmp_path, monkeypatch, sim)
    assert "@app.post('/virtual')\ndef _gbench_virtual" in srv.patched
    assert "requests.post(CONFIG['toolbench_url']" not in srv.patched
    assert srv.route(_req(qid="1"), _info("cached")) == {"error": "", "response": "cached"}
    assert srv.route(_req(qid="2"), _info("x")) == {"error": "", "response": "sim"}
    # the model the API does not serve is asked once, then skipped; the request is upstream's
    assert [kw["model"] for _, kw in sim] == ["m404", "m2"]
    assert set(sim[1][1]) == {"model", "messages", "max_tokens", "temperature",
                              "response_format"}
    assert sim[1][0] == PLACEHOLDER and srv.proxy == [120.0]
    srv.mode["sim"] = "down"
    with pytest.raises(RuntimeError):
        srv.route(_req(qid="3"), _info("y"))
    srv.mode["proxy"] = "down"
    with pytest.raises(ConnectionError):
        srv.route(_req(qid="4"), _info("z"))
    recs = [json.loads(line) for line in srv.log.read_text().splitlines()]
    assert [r["query_id"] for r in recs] == ["1", "2", "3", "4"]
    assert recs[0]["type"] == "cached_real_response" and not recs[0].get("server_error")
    assert recs[1]["type"] == "fake_response" and recs[1]["simulator_models"] == ["m2"]
    assert recs[1]["real_api_status"] == 401 and recs[1]["simulator_failed_reply"] is False
    assert recs[2]["server_error"] and recs[2]["stage"] == "simulator"
    assert recs[3]["server_error"] and recs[3]["stage"] == "real_api_proxy"
    assert recs[3]["error"] == "ConnectionError"
    assert PLACEHOLDER not in srv.log.read_text()


def test_the_client_patch_tags_each_call_with_group_and_query(tmp_path):
    src = ("import os\nclass rapidapi_wrapper:\n    def __init__(self, query_json):\n"
           '        self.input_description = query_json["query"]\n'
           "    def _step(self):\n"
           '                        headers = {"toolbench_key": self.toolbench_key}\n'
           "class pipeline_runner:\n"
           "    def method_converter(self, env):\n"
           "        if True:\n"
           "            llm_forward = ChatGPTFunction(model=self.args.chatgpt_model, "
           "openai_key=openai_key, base_url=self.args.base_url)\n")
    f = tmp_path / "rapidapi_multithread.py"
    f.write_text(src)
    patch = _read("docker", "toolbench_patch_client.py").replace(
        "/stb/toolbench/inference/Downstream_tasks/rapidapi_multithread.py", str(f))
    exec(compile(patch, "toolbench_patch_client.py", "exec"), {"__name__": "patch"})
    out = f.read_text()
    assert 'self._gbench_query_id = query_json.get("query_id")' in out
    assert 'headers["X-Gbench-Group"] = os.environ.get("TB_GROUP", "")' in out
    assert 'headers["X-Gbench-Query-Id"] = str(' in out
    assert 'llm_forward._gbench_query_id = getattr(env, "_gbench_query_id", None)' in out


def _entrypoint(env):
    base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "TB_MODEL": "m",
            "TB_ENDPOINT": "http://172.17.0.1:5678/v1"}
    base.update(env)
    return subprocess.run(["bash", os.path.join(_REPO, "docker", "toolbench_entrypoint.sh")],
                          env=base, capture_output=True, text=True, timeout=60)


def test_the_entrypoint_needs_the_cascade_and_refuses_a_real_key():
    r = _entrypoint({"GEMINI_API_KEY": PLACEHOLDER,
                     "GBENCH_GEMINI_RELAY_URL": "http://172.17.0.1:1234"})
    assert r.returncode == 2 and "GBENCH_JUDGE_MODELS" in r.stderr
    r = _entrypoint({"GBENCH_JUDGE_MODELS": "gemini-3.8-flash", "GEMINI_API_KEY": REAL_KEY,
                     "GBENCH_GEMINI_RELAY_URL": "http://172.17.0.1:1234"})
    assert r.returncode == 7 and "real key is refused" in r.stderr
    r = _entrypoint({"GBENCH_JUDGE_MODELS": "gemini-3.8-flash", "GEMINI_API_KEY": PLACEHOLDER,
                     "GBENCH_GEMINI_RELAY_URL": "https://generativelanguage.googleapis.com/x"})
    assert r.returncode == 7 and "origin" in r.stderr
    r = _entrypoint({"GBENCH_JUDGE_MODELS": "gemini-3.8-flash", "GEMINI_API_KEY": PLACEHOLDER,
                     "GBENCH_GEMINI_RELAY_URL": "http://172.17.0.1:1234",
                     "GEMINI_OPENAI_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai/"})
    assert r.returncode == 7 and "not the Gemini relay's URL" in r.stderr
    ep = _read("docker", "toolbench_entrypoint.sh")
    assert "gemini-3" not in ep and "TB_SIMULATOR_MODELS" not in ep
    assert 'chown -R "${TB_HOST_UID}:${TB_HOST_GID}" /out' in ep
    assert 'TB_GROUP="$g" python /stb/toolbench/inference/qa_pipeline_multithread.py' in ep
    assert 'export TB_MODEL_FAILURE_LOG="/out/model_request_failures.jsonl"' in ep
    df = _read("docker", "toolbench.Dockerfile")
    assert f'LABEL {T.HARNESS_LABEL}="{T.HARNESS_VERSION}"' in df and T.HARNESS_VERSION == "5"
    assert "RUN python /tmp/toolbench_patch_simulator.py && python /tmp/toolbench_patch_client.py" \
        in df


# --------------------------------------------------------------------------- #
# toolbench: ToolEval parsing, answering models, SoWR without a verdict
# --------------------------------------------------------------------------- #
def test_upstream_reads_no_reason_from_select_better_answer():
    assert T._parse_function_args("select_better_answer", '{"index": 1}', True) == {"index": 1}
    assert T._parse_function_args("check_answer_status", '{"answer_status": "Unsure"}',
                                  True) == {"answer_status": "Unsure"}
    assert T._parse_function_args("check_answer_status", '{"answer_status": "Solved"}',
                                  True) is None
    assert T._parse_function_args("parse_answer_status", '{"answer_status": "Unsure"}',
                                  True) is None
    assert T._parse_function_args("check_answer_status", '{"answer_status": "Solved"}',
                                  False) == {"answer_status": "Solved"}


def test_the_tooleval_cascade_records_the_answering_model_and_skips_a_404(monkeypatch):
    asked = []

    class NotServed(Exception):
        code = 404

    class Models:
        async def generate_content(self, model, contents, config):
            asked.append(model)
            if model == "gemini-3.8-flash":
                raise NotServed("not found")
            if len(asked) < 4:
                raise RuntimeError("503")
            call = types.SimpleNamespace(name="check_answer_status",
                                         args={"answer_status": "Solved", "reason": "ok"})
            return types.SimpleNamespace(function_calls=[call], candidates=[])

    client = types.SimpleNamespace(aio=types.SimpleNamespace(models=Models()))
    monkeypatch.setattr(B, "gemini_api_keys", lambda: ["k"])
    monkeypatch.setattr(B, "_judge_clients", lambda: [client])
    monkeypatch.setattr(B, "_next_judge_client", lambda: client)
    monkeypatch.setattr(B, "judge_cascade", lambda: ["gemini-3.8-flash", "gemini-3.7-flash"])
    monkeypatch.setattr(B, "_judge_cascade_rounds", lambda: 3)
    monkeypatch.setattr(B, "_JUDGE_BACKOFF", 0.0)
    before = B.gemini_answer_counts().get("gemini-3.7-flash", 0)
    verdicts = T._Verdicts()
    T._VERDICTS.set(verdicts)
    config = T._judge_generation_config("check_answer_status", True)
    args, model = asyncio.run(T._tooleval_cascade("p", config=config))
    assert model == "gemini-3.7-flash" and json.loads(args)["answer_status"] == "Solved"
    assert asked == ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.7-flash",
                     "gemini-3.7-flash"]
    assert B.gemini_answer_counts()["gemini-3.7-flash"] == before + 1


def _answer(final="done", finish=True):
    tool = {"name": "Finish" if finish else "x", "arguments": "{}", "response": ""}
    return {"query": "q", "answer": {"final_answer": final, "answer_details": [
        {"role": "tool", "message": str(tool).replace('"', "'"), "next": []}]}}


def _stub_tooleval(monkeypatch, pref_votes):
    async def tooleval_cascade(prompt, config=None):
        func = T._forced_function(config)
        if func == "check_answer_status":
            return json.dumps({"answer_status": "Solved", "reason": "r"}), "gemini-3.7-flash"
        return None, "JUDGE_OUTAGE"
    monkeypatch.setattr(T, "_tooleval_cascade", tooleval_cascade)
    it = iter(pref_votes)

    async def preference(query, cand, ref, sem):
        T._log_call({"function": "preference", "choice": "x"})
        return next(it)
    monkeypatch.setattr(T, "_preference", preference)


def test_sowr_drops_comparisons_without_a_verdict_and_withholds_when_none_has_one(monkeypatch):
    cand = {"1": _answer(), "2": _answer()}
    ref = {"1": _answer("r"), "2": _answer("r")}
    # query 1: cand, None, None -> a win on its one vote; query 2: None x3 -> no vote
    _stub_tooleval(monkeypatch, ["cand", None, None, None, None, None])
    r = asyncio.run(T._score_all(cand, ref, 3, 2, test_ids=["1", "2"]))
    assert r["sopr"] == 1.0 and r["sowr"] in (1.0, 0.0)
    assert r["sowr_votes_missing"] == 5 and len(r["sowr_queries_without_vote"]) == 1
    _stub_tooleval(monkeypatch, [None] * 6)
    r = asyncio.run(T._score_all(cand, ref, 3, 2, test_ids=["1", "2"]))
    assert r["sowr"] is None and sorted(r["sowr_queries_without_vote"]) == ["1", "2"]


def test_excluded_queries_are_left_out_of_both_scores_and_records_keep_each_verdict(
        monkeypatch):
    cand = {"1": _answer(), "2": _answer(), "3": _answer()}
    ref = {q: _answer("r") for q in cand}
    _stub_tooleval(monkeypatch, ["cand"] * 6)
    r = asyncio.run(T._score_all(cand, ref, 3, 2, test_ids=["1", "2", "3"], excluded={"3"}))
    assert r["n_candidate"] == 2 and r["sowr"] == 1.0 and "3" not in r["records"]
    rec = r["records"]["1"]
    assert [p["status"] for p in rec["sopr_passes"]] == ["Solved"] * 3
    call = rec["sopr_passes"][0]["calls"][0]
    assert call["model"] == "gemini-3.7-flash" and call["reason"] == "r"
    assert call["function"] == "check_answer_status" and call["outcome"] == "verdict"
    assert [rd["vote"] for rd in rec["sowr_rounds"]] == ["cand"] * 3
    assert r["judge_models_used"] == {"gemini-3.7-flash": 6}
    assert r["solved_equivalent"] == 2.0


# --------------------------------------------------------------------------- #
# toolbench: the runner
# --------------------------------------------------------------------------- #
def _tb_answer(solved=True):
    return {"query": "q", "answer": {"final_answer": "S" if solved else "N",
                                     "answer_details": []}}


@pytest.fixture
def tb(monkeypatch, tmp_path):
    FakeGeminiRelay.instances.clear()
    FakeEndpointRelay.instances.clear()
    monkeypatch.setenv("GEMINI_API_KEY", REAL_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-endpoint-secret-0000")
    monkeypatch.setenv("GBENCH_RESULTS_DIR", str(tmp_path / "results"))
    for k in ("GBENCH_TOOLBENCH_GROUPS", "GBENCH_TOOLBENCH_REFERENCE_DIR", "GEMINI_API_KEYS",
              "GBENCH_TOOLBENCH_REFERENCE", "GBENCH_TOOLBENCH_EVAL_TIMES", "GBENCH_SHARD",
              "GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL", "GEMINI_OPENAI_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(T, "check_toolbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(T, "_served_model_id", lambda endpoint, fallback: fallback)
    guard = []
    monkeypatch.setattr(T, "_bridge_gateway", lambda image: guard.append(image) or "172.17.0.1")
    monkeypatch.setattr(T.gemini_relay, "GeminiRelay", FakeGeminiRelay)
    monkeypatch.setattr(T, "EndpointRelay", FakeEndpointRelay)

    async def status(query, answer, sem):
        T._log_call({"function": "check_answer_status", "model": "gemini-3.7-flash"})
        return "Solved" if answer.get("final_answer") == "S" else "Unsolved"

    async def pref(query, cand, ref, sem):
        return "cand"

    monkeypatch.setattr(T, "_answer_status", status)
    monkeypatch.setattr(T, "_preference", pref)
    ref_root = tmp_path / "ref"
    (ref_root / T._REFERENCE_DEFAULT).mkdir(parents=True)
    for g in T._GROUPS:
        (ref_root / T._REFERENCE_DEFAULT / f"{g}.json").write_text(
            json.dumps({"1": _tb_answer(False), "2": _tb_answer(False)}))
    monkeypatch.setattr(T, "_baseline_reference_root", lambda ref, groups, image: str(ref_root))
    monkeypatch.setattr(T, "_group_test_ids",
                        lambda image, groups, limit: {g: ["1", "2"] for g in groups})
    state = {"server": [], "observations": {}, "answers": {g: ["1", "2"] for g in T._GROUPS},
             "write_records": True, "model_failures": []}
    runs = []

    def fake_run(cmd, *a, **kw):
        if cmd[:2] == ["docker", "run"]:
            runs.append((cmd, kw))
            out = cmd[cmd.index("-v") + 1].split(":")[0]
            state["workdir"] = out
            os.makedirs(os.path.join(out, "candidate"), exist_ok=True)
            for g in T._GROUPS:
                with open(os.path.join(out, "candidate", f"{g}.json"), "w") as fh:
                    json.dump({q: _tb_answer(True) for q in state["answers"][g]}, fh)
            if state["write_records"]:
                with open(os.path.join(out, T.SERVER_CALLS), "w") as fh:
                    fh.write("".join(json.dumps(r) + "\n" for r in state["server"]))
                with open(os.path.join(out, T.OBSERVATIONS), "w") as fh:
                    json.dump({"groups": state["observations"]}, fh)
                with open(os.path.join(out, T.MODEL_FAILURES), "w") as fh:
                    fh.write("".join(json.dumps(r) + "\n" for r in state["model_failures"]))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(T.subprocess, "run", fake_run)
    return types.SimpleNamespace(state=state, runs=runs, guard=guard)


def test_toolbench_runs_on_the_guarded_bridge_with_relays_and_no_real_credential(tb):
    res = T.run_toolbench("m", "http://127.0.0.1:8000/v1", 2)
    [(cmd, kw)] = tb.runs
    assert tb.guard == [T._image()]
    assert cmd[cmd.index("--network") + 1] == "bridge" and "host" not in cmd
    text = " ".join(cmd) + json.dumps(kw["env"])
    assert REAL_KEY not in text and "sk-endpoint-secret-0000" not in text
    assert kw["env"]["GEMINI_API_KEY"] == PLACEHOLDER
    assert "TB_ENDPOINT=http://172.17.0.1:5678/v1" in cmd
    assert f"GBENCH_JUDGE_MODELS={','.join(B.judge_cascade())}" in cmd
    assert not any(a.startswith(("TB_SIMULATOR_MODELS", "TB_SERVER_PORT")) for a in cmd)
    [ep] = FakeEndpointRelay.instances
    assert ep.upstream == "http://127.0.0.1:8000/v1" and ep.key == "sk-endpoint-secret-0000"
    assert ep.stopped and FakeGeminiRelay.instances[0].stopped
    assert res["status"] == "success" and res["docker_network"] == "bridge"
    assert res["headline_metrics"] == {"SoPR": 100.0, "SoWR": 100.0}
    assert res["total_questions"] == 12 and res["effective_n"] == 12
    assert res["correct_answers"] == 12.0 and res["count_unit"] == "queries"


def test_toolbench_harness_error_queries_are_left_out_and_named(tb):
    tb.state["server"] = [
        {"group": "G1_tool", "query_id": "1", "type": "server_error", "server_error": True,
         "stage": "real_api_proxy"},
        {"group": "G1_tool", "query_id": "2", "type": "fake_response",
         "simulator_models": ["gemini-3.7-flash"]},
        {"group": "G2_category", "query_id": "1", "type": "cached_real_response"}]
    tb.state["observations"] = {"G3_instruction": {"2": {"simulator_failed_observations": 1}}}
    tb.state["answers"]["G2_instruction"] = ["1"]
    res = T.run_toolbench("m", "http://127.0.0.1:8000/v1", 2)
    assert res["status"] == "completed_with_errors"
    assert res["n_harness_error_queries"] == 3
    assert res["harness_error_query_kinds"] == {"no_answer": 1, "server_error": 1,
                                                "simulator_failed": 1}
    assert res["harness_error_queries_by_group"] == {"G1_tool": ["1"], "G2_instruction": ["2"],
                                                     "G3_instruction": ["2"]}
    assert res["queries_by_group"]["G1_tool"] == 1 and res["effective_n"] == 9
    assert res["simulator_usage"]["simulated"] == 1 and res["simulator_usage"]["cached"] == 1
    assert res["simulator_usage"]["server_errors"] == 1
    assert res["simulator_models_used"] == {"gemini-3.7-flash": 1}
    assert res["gemini_models_answered"]["gemini-3.7-flash"] >= 1
    reason = res["leaderboard_comparable_reason"]
    assert "3 quer(ies) met a harness error and are left out of SoPR and SoWR" in reason
    assert "of 3 tool call(s), 1 were answered from the cache, 0 by the real-API proxy and 1" \
        in reason
    traces = {(t["group"], t["query_id"]): t for t in res["sample_traces"]}
    assert len(traces) == 12
    assert traces[("G1_tool", "1")]["harness_errors"] == ["server_error"]
    assert traces[("G1_tool", "1")]["server_error_stages"] == {"real_api_proxy": 1}
    assert traces[("G1_tool", "2")]["simulator_models"] == {"gemini-3.7-flash": 1}
    assert traces[("G1_tool", "2")]["sopr_passes"][0]["status"] == "Solved"
    assert any(h.get("queries") == 3 for h in res["harness_errors"])


def test_toolbench_withholds_a_group_whose_every_query_is_a_harness_error(tb):
    tb.state["observations"] = {"G1_tool": {"1": {"server_error_observations": 1},
                                            "2": {"server_error_observations": 2}}}
    res = T.run_toolbench("m", "http://127.0.0.1:8000/v1", 2)
    assert res["accuracy"] is None and res["status"] == "failed"
    why = res["accuracy_withheld_reason"]
    assert "every query of ['G1_tool'] is a harness error" in why
    assert "Provision the prerequisites" not in why and "re-run" in why
    # the diagnostics of the run are kept with the withheld result
    assert res["n_harness_error_queries"] == 2 and "judge_models_used" in res


def test_toolbench_without_the_images_records_publishes_no_score(tb):
    tb.state["write_records"] = False
    with pytest.raises(RuntimeError, match="wrote no toolbench_server_calls.jsonl"):
        T.run_toolbench("m", "http://127.0.0.1:8000/v1", 2)


def test_toolbench_refuses_an_endpoint_url_with_user_info(tb):
    with pytest.raises(RuntimeError, match="user name or password"):
        T.run_toolbench("m", "http://u:pw@127.0.0.1:8000/v1", 2)
    assert not tb.runs


def test_toolbench_keeps_its_records_and_cleans_its_workdir(tb, tmp_path):
    res = T.run_toolbench("m", "http://127.0.0.1:8000/v1", 2)
    kept = res["artifacts_dir"]
    assert kept.startswith(str(tmp_path / "results"))
    assert os.path.isfile(os.path.join(kept, T.SERVER_CALLS))
    assert os.path.isfile(os.path.join(kept, "candidate", "G1_tool.json"))
    assert not os.path.exists(tb.state["workdir"])


def test_toolbench_gate_refuses_an_image_without_the_harness_label(monkeypatch):
    monkeypatch.setattr(T.shutil, "which", lambda _x: "/usr/bin/docker", raising=False)
    seen = []

    def fake_run(cmd, *a, **k):
        seen.append(cmd)
        out = "2" if "org.gbench.model-request-policy" in " ".join(cmd) else ""
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
    monkeypatch.setattr(T.subprocess, "run", fake_run)
    monkeypatch.setattr(T, "image_applies_request_policy", lambda image: True)
    ok, why = T.check_toolbench_prerequisites()
    assert not ok and f"{T.HARNESS_LABEL}={T.HARNESS_VERSION}" in why


# --------------------------------------------------------------------------- #
# Docs follow the code
# --------------------------------------------------------------------------- #
def test_the_guides_describe_the_new_behavior():
    g = _flat(_read("docs", "evals", "gaia2.md"))
    for needle in (f"`{G.HARNESS_ERRORS_LABEL}={G.HARNESS_ERRORS_VERSION}`",
                   "`server_parser_text_lost`",
                   "`judge_no_verdict`", "temperature 0.5", "16,000 tokens per turn",
                   "200 steps", "`completion_duration`", "case-insensitively",
                   "issues/29", "`--network bridge`", "`sample_traces`",
                   "`gaia2_orchestrator.log`", "`n_timeout_runs`"):
        assert needle in g, needle
    assert "single-run" not in g and "--network host" not in g
    t = _flat(_read("docs", "evals", "toolbench.md"))
    for needle in (f"`{T.HARNESS_LABEL}={T.HARNESS_VERSION}`", "`toolbench_server_calls.jsonl`",
                   "`tool_observations.json`", "`simulator_usage`", "`sample_traces`",
                   "`n_harness_error_queries`", "`sowr_queries_without_vote_by_group`",
                   "`gemini_models_answered`", "`--network bridge`", "`GBENCH_JUDGE_MODELS`",
                   "`headline_metrics`"):
        assert needle in t, needle
    assert "--network host" not in t
