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

"""tau2 / tau3: the Gemini user simulator and judge, and the keys of the tau2 container.

* cascade    - both roles use gbench's one Gemini cascade (base.judge_cascade(), handed to the
               image as GBENCH_JUDGE_MODELS) in its order, or the one model their knob names;
               the search cascade never applies.
* retries    - each model gets the call's num_retries on a transient error, LiteLLM is called
               with num_retries=0 so its own retry never re-enters the cascade, an HTTP 404 skips
               the model for the rest of the run, and the agent passes through once.
* params     - LiteLLM's default thinking level for Gemini 3 models is removed, so every model
               gets upstream's arguments only.
* recording  - the result names the models that answered, per role and domain, and each
               NL-assertion verdict names its model.
* keys       - the container holds only gbench's Gemini relay placeholder (and, with a real
               endpoint key, reaches gbench's endpoint relay): the Gemini gate runs on the host,
               a real key inside the image is refused, the embedder and LiteLLM go to the relay.
* probe      - the endpoint probe's `docker rm -f` after a timeout is bounded and never replaces
               the infra_required.
No docker, no network (loopback test servers only), no Gemini call, no model server.
"""

import contextlib
import json
import os
import subprocess
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import container_guard as cg
from gbench.runners.eval_suites import gemini_relay as GR
from gbench.runners.eval_suites import tau_common as TC

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLACEHOLDER = GR.PLACEHOLDER_PREFIX + "ab" * 24
# Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
REAL_KEY = "AIza" + "TauTestFakeKey000000000000000000001"
ENDPOINT_KEY = "sk-endpoint-tau-test-0123456789"
_KNOBS = ("GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL", "GBENCH_SEARCH_MODELS",
          "GBENCH_SEARCH_MODEL", "GBENCH_TAU2_USER_LLM", "GBENCH_TAU2_EVAL_LLM", "TAU2_USER_LLM",
          "TAU2_EVAL_LLM", "GEMINI_API_KEY", "GEMINI_API_KEYS", "GEMINI_OPENAI_BASE_URL",
          GR.RELAY_URL_ENV, "OPENAI_API_KEY", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy",
          "https_proxy", "NO_PROXY", "no_proxy", TC._AGENT_KEY_ENV, "GBENCH_TAU2_USER_TEMPERATURE",
          "TAU2_USER_TEMPERATURE")


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in _KNOBS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(TC.time, "sleep", lambda *_a, **_k: None)
    TC.reset_gemini_answers()
    TC._ROLE_STATE.__dict__.clear()
    monkeypatch.setitem(TC._CURRENT_DOMAIN, "domain", None)
    yield
    TC.reset_gemini_answers()
    TC._ROLE_STATE.__dict__.clear()


def _read(*parts):
    with open(os.path.join(_REPO, *parts), encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------- #
# the roles' models: gbench's one cascade, in its order
# --------------------------------------------------------------------------- #
def test_both_roles_use_the_unified_cascade_by_default():
    expected = [f"gemini/{m}" for m in base.GEMINI_CASCADE]
    assert TC._role_cascade(TC._USER_ROLE) == expected == TC._role_cascade(TC._JUDGE_ROLE)
    assert TC._user_llm() == TC._eval_llm() == "gemini/gemini-3.8-flash"
    assert TC._role_source(TC._USER_ROLE) == "default"
    assert TC._role_models() == [("user simulator", "gemini/gemini-3.8-flash"),
                                 ("NL-assertion judge", "gemini/gemini-3.8-flash")]


def test_gbench_judge_models_orders_both_roles_and_search_models_never_do(monkeypatch):
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "gemini-search-a,gemini-search-b")
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-3.6-flash, gemini/gemini-3.5-flash")
    for role in (TC._USER_ROLE, TC._JUDGE_ROLE):
        assert TC._role_cascade(role) == ["gemini/gemini-3.6-flash", "gemini/gemini-3.5-flash"]
        assert TC._role_source(role) == "GBENCH_JUDGE_MODELS"
    monkeypatch.delenv("GBENCH_JUDGE_MODELS")
    monkeypatch.setenv("GBENCH_JUDGE_MODEL", "gemini-3.7-flash")
    assert TC._role_cascade(TC._USER_ROLE) == ["gemini/gemini-3.7-flash"], "a one-model cascade"
    # the same rule over an explicit mapping (the host's check before the container)
    assert TC._role_cascade(TC._JUDGE_ROLE, {"GBENCH_JUDGE_MODELS": "a,b"}) == \
        ["gemini/a", "gemini/b"]


def test_a_role_knob_names_one_model_without_fallback(monkeypatch):
    monkeypatch.setenv("GBENCH_TAU2_USER_LLM", "gemini/gemini-3.5-flash")
    assert TC._role_cascade(TC._USER_ROLE) == ["gemini/gemini-3.5-flash"]
    assert TC._role_source(TC._USER_ROLE) == "GBENCH_TAU2_USER_LLM"
    assert TC._role_cascade(TC._JUDGE_ROLE)[0] == "gemini/gemini-3.8-flash"
    assert TC._role_text(TC._USER_ROLE) == "gemini-3.5-flash alone (GBENCH_TAU2_USER_LLM)"
    assert TC._role_text(TC._JUDGE_ROLE).startswith(
        "gbench's Gemini cascade gemini-3.8-flash > gemini-3.7-flash > ")


def test_no_second_default_or_search_cascade_remains_in_tau():
    src = _read("gbench", "runners", "eval_suites", "tau_common.py")
    for gone in ("DEFAULT_JUDGE_MODEL", "_search_cascade", "GBENCH_SEARCH_BACKOFF",
                 '"GBENCH_SEARCH_"', "_credential_map"):
        assert gone not in src, gone
    assert TC.endpoint_container_hosts("tau2") == ([], True)
    assert TC.endpoint_container_hosts("tau3") == ([], True)


# --------------------------------------------------------------------------- #
# the cascade call: per-model retries, no nested retries, 404, the agent
# --------------------------------------------------------------------------- #
def _transient(status=503):
    import httpx
    import openai
    req = httpx.Request("POST", "http://relay/v1beta/models/m:generateContent")
    return openai.APIStatusError("overloaded", response=httpx.Response(status, request=req),
                                 body=None)


class _NotFound(Exception):
    status_code = 404


def test_each_model_gets_the_calls_retries_and_litellm_none():
    seen = []

    def inner(**kw):
        seen.append((kw["model"], kw["num_retries"]))
        if kw["model"] != "gemini/c":
            raise _transient()
        return {"ok": kw["model"]}

    casc = TC._build_gemini_cascade(inner, 0.0, cascade_fn=lambda: ["a", "b", "c"])
    assert casc(model="gemini/a", messages=[], num_retries=3) == {"ok": "gemini/c"}
    assert seen == [("gemini/a", 0)] * 4 + [("gemini/b", 0)] * 4 + [("gemini/c", 0)]

    tried = []

    def always(**kw):
        tried.append(kw["model"])
        raise _transient()

    with pytest.raises(Exception):
        TC._build_gemini_cascade(always, 0.0, cascade_fn=lambda: ["a", "b"])(
            model="gemini/a", messages=[], num_retries=1)
    assert tried == ["gemini/a", "gemini/a", "gemini/b", "gemini/b"]


def test_a_non_transient_error_moves_on_at_once_and_404_skips_the_model_for_the_run():
    seen = []

    def inner(**kw):
        seen.append(kw["model"])
        if kw["model"] == "gemini/a":
            raise _NotFound("models/a is not found")
        if kw["model"] == "gemini/b":
            raise ValueError("bad reply shape")
        return "ok"

    casc = TC._build_gemini_cascade(inner, 0.0, cascade_fn=lambda: ["a", "b", "c"])
    assert casc(model="gemini/a", messages=[], num_retries=3) == "ok"
    assert seen == ["gemini/a", "gemini/b", "gemini/c"]
    seen.clear()
    assert casc(model="gemini/a", messages=[], num_retries=3) == "ok"
    assert seen == ["gemini/b", "gemini/c"], "a model the API does not serve is not asked again"
    assert TC._NOT_SERVED == {"a": "_NotFound"}


def test_the_agent_is_sent_once_unchanged():
    seen = []

    def inner(*args, **kw):
        seen.append(kw)
        raise _transient()

    casc = TC._build_gemini_cascade(inner, 0.0, cascade_fn=lambda: ["a", "b"])
    with pytest.raises(Exception):
        casc(model="openai/m", messages=[], num_retries=3, api_base="http://x/v1")
    assert seen == [{"model": "openai/m", "messages": [], "num_retries": 3,
                     "api_base": "http://x/v1"}]


def _fake_litellm_stack(monkeypatch, fail_models):
    """A litellm whose completion retries itself through litellm.main.completion, as LiteLLM's
    client wrapper does (completion_with_retries calls the module's completion), and a tau2
    llm_utils that calls it."""
    calls = []
    litellm = types.ModuleType("litellm")
    main = types.ModuleType("litellm.main")

    class _Resp:
        choices = [types.SimpleNamespace(message=types.SimpleNamespace(content="hello",
                                                                        tool_calls=None))]

    def completion(*args, **kw):
        calls.append((kw["model"], kw.get("num_retries")))
        if kw["model"] in fail_models:
            n = kw.get("num_retries") or 0
            for _ in range(n):          # LiteLLM: retry through the module-level completion
                try:
                    return main.completion(*args, **dict(kw, num_retries=0))
                except Exception:       # noqa: BLE001
                    pass
            raise _transient()
        return _Resp()

    litellm.completion = completion
    main.completion = completion
    litellm.main = main
    llm_utils = types.ModuleType("tau2.utils.llm_utils")
    llm_utils.completion = completion
    mods = {"litellm": litellm, "litellm.main": main, "tau2": types.ModuleType("tau2"),
            "tau2.utils": types.ModuleType("tau2.utils"), "tau2.utils.llm_utils": llm_utils}
    monkeypatch.setattr(TC, "_PATCHED", False)
    monkeypatch.setattr(TC, "_RETRY_PATCHES", {"completion": False, "generate": False})
    monkeypatch.setattr(TC, "_install_gemini_param_guard", lambda: None)
    monkeypatch.setattr(TC, "_install_judge_model_record", lambda: False)
    TC.reset_llm_failures()
    return calls, mods, litellm, llm_utils


def test_litellms_own_retry_never_reenters_the_cascade(monkeypatch):
    first, second = (f"gemini/{m}" for m in base.GEMINI_CASCADE[:2])
    calls, mods, litellm, llm_utils = _fake_litellm_stack(monkeypatch, {first})
    with mock.patch.dict(sys.modules, mods):
        TC._install_robustness_patches()
        resp = llm_utils.completion(model=first, messages=[], num_retries=3)
        assert resp.choices[0].message.content == "hello"
        # 4 attempts on the first model (1 + num_retries), all with LiteLLM's retries off, then
        # the second model answers: 5 requests, where re-entering the cascade would make dozens
        assert calls == [(first, 0)] * 4 + [(second, 0)]
        assert TC.gemini_answers()["total"] == {second.split("/", 1)[1]: 1}
        assert TC.llm_failures().get("other_llm_error") == 4
        # litellm.completion itself carries no cascade: LiteLLM's internal retries stay on the
        # model they were asked for
        calls.clear()
        with pytest.raises(Exception):
            litellm.completion(model=first, messages=[], num_retries=2)
        assert {m for m, _n in calls} == {first}


def test_an_empty_reply_is_requested_again_at_most_three_times_in_all(monkeypatch):
    calls, mods, litellm, llm_utils = _fake_litellm_stack(monkeypatch, set())
    empty = types.SimpleNamespace(choices=[types.SimpleNamespace(
        message=types.SimpleNamespace(content="  ", tool_calls=None))])
    seen = []

    def completion(*a, **kw):
        seen.append(kw["model"])
        return empty
    litellm.completion = completion
    llm_utils.completion = completion
    gen = []

    def generate(model, messages, **kw):
        gen.append(model)
        llm_utils.completion(model=model, messages=messages)
        return types.SimpleNamespace(content="```\nhi\n```", tool_calls=None)
    llm_utils.generate = generate
    with mock.patch.dict(sys.modules, mods):
        TC._install_robustness_patches()
        assert llm_utils.generate("gemini/x", []).content == "hi"
    assert seen == ["gemini/x"] * 3 and gen == ["gemini/x"], "no second retry layer"


# --------------------------------------------------------------------------- #
# LiteLLM's thinking level
# --------------------------------------------------------------------------- #
class _StudioConfig:
    """LiteLLM 1.81.11's VertexGeminiConfig.map_openai_params behaviour for Gemini 3 models."""

    def map_openai_params(self, non_default_params, optional_params, model, drop_params):
        for k, v in non_default_params.items():
            if k == "temperature":
                optional_params["temperature"] = v
            elif k == "reasoning_effort":
                optional_params["thinkingConfig"] = {"thinkingLevel": "high",
                                                     "includeThoughts": True}
        if "gemini-3" in model:
            cfg = optional_params.get("thinkingConfig", {})
            if "thinkingLevel" not in cfg and "thinkingBudget" not in cfg:
                cfg["thinkingLevel"] = "minimal" if "gemini-3-flash" in model else "low"
                optional_params["thinkingConfig"] = cfg
        return optional_params


def test_no_model_gets_a_thinking_level_litellm_adds():
    fake = types.ModuleType("litellm")
    config = type("GoogleAIStudioGeminiConfig", (_StudioConfig,), {})
    fake.GoogleAIStudioGeminiConfig = config
    with mock.patch.dict(sys.modules, {"litellm": fake}):
        TC._install_gemini_param_guard()
        TC._install_gemini_param_guard()          # idempotent
    for model in list(base.GEMINI_CASCADE) + ["gemini-3-flash-preview"]:
        out = config().map_openai_params(non_default_params={"temperature": 0.0},
                                         optional_params={}, model=model, drop_params=False)
        assert out == {"temperature": 0.0}, model
    asked = config().map_openai_params({"reasoning_effort": "high"}, {}, "gemini-3.8-flash",
                                       False)
    assert asked["thinkingConfig"]["thinkingLevel"] == "high", "an explicit ask is kept"
    preset = config().map_openai_params({}, {"thinkingConfig": {"thinkingBudget": 0}},
                                        "gemini-3.8-flash", False)
    assert preset["thinkingConfig"] == {"thinkingBudget": 0}


def test_the_guard_is_required():
    with mock.patch.dict(sys.modules, {"litellm": types.ModuleType("litellm")}):
        with pytest.raises(RuntimeError, match="thinking level"):
            TC._install_gemini_param_guard()


# --------------------------------------------------------------------------- #
# which model answered
# --------------------------------------------------------------------------- #
def test_the_verdict_records_its_judge_model(monkeypatch):
    class RewardInfo:
        def __init__(self):
            self.info = None
            self.reward = 1.0

    class NLAssertionsEvaluator:
        @classmethod
        def calculate_reward(cls, task, full_trajectory):
            TC._note_gemini_answer("gemini-3.7-flash")
            return RewardInfo()

    mod = types.ModuleType("tau2.evaluator.evaluator_nl_assertions")
    mod.NLAssertionsEvaluator = NLAssertionsEvaluator
    mods = {"tau2": types.ModuleType("tau2"), "tau2.evaluator": types.ModuleType("tau2.evaluator"),
            "tau2.evaluator.evaluator_nl_assertions": mod}
    monkeypatch.setitem(TC._CURRENT_DOMAIN, "domain", "retail")
    TC._ROLE_STATE.role = TC._JUDGE_ROLE
    with mock.patch.dict(sys.modules, mods):
        assert TC._install_judge_model_record() is True
        info = NLAssertionsEvaluator.calculate_reward(task=None, full_trajectory=[])
    assert info.info == {"judge_model": "gemini-3.7-flash"}
    sim = types.SimpleNamespace(reward_info=types.SimpleNamespace(
        info={"env": None, "nl": info.info}))
    assert TC._verdict_judge_model(sim) == "gemini-3.7-flash"
    assert TC.gemini_answers()["by_domain"] == {
        "retail": {TC._JUDGE_ROLE: {"gemini-3.7-flash": 1}}}


def test_the_result_names_the_models_that_answered(monkeypatch):
    for name, value in (("check_tau_env_prerequisites", lambda: (True, "")),
                        ("_install_robustness_patches", lambda *a, **k: None),
                        ("_configure_evaluator_llm", lambda *a, **k: None),
                        ("_quiet_tau2_noise", lambda: None),
                        ("_prune_stale_search_sandboxes", lambda: None),
                        ("_refuse_resume_prompts", lambda: None),
                        ("_tau2_version", lambda: "1.0.1")):
        monkeypatch.setattr(TC, name, value)
    monkeypatch.setenv("GBENCH_TAU2_SAVE_TRACES", "")

    def one(domain, *a, **k):
        TC._build_agent_args("http://x/v1", False, "tau2")
        monkeypatch.setitem(TC._CURRENT_DOMAIN, "domain", domain)
        for role, model in ((TC._USER_ROLE, "gemini-3.8-flash"),
                            (TC._USER_ROLE, "gemini-3.7-flash"),
                            (TC._JUDGE_ROLE, "gemini-3.8-flash")):
            TC._ROLE_STATE.role = role
            TC._note_gemini_answer(model)
        TC._DOMAIN_PASS_HAT_K[domain] = {"pass^1": 0.5}
        TC._DOMAIN_JUDGE_MODELS[domain] = {"gemini-3.8-flash": 2}
        return 2, 1.0, 1, 0

    monkeypatch.setattr(TC, "_run_one_domain", one)
    r = TC.run_tau_env("tau2", ["retail"], "m", "http://x/v1", 1, None, "docs/evals/tau2.md")
    assert r["gemini_models_answered"] == {"gemini-3.7-flash": 1, "gemini-3.8-flash": 2}
    assert r["gemini_cascade"] == list(base.GEMINI_CASCADE)
    rep = r["tau2_report"]
    assert rep["gemini_models_answered_by_role"] == {
        TC._JUDGE_ROLE: {"gemini-3.8-flash": 1},
        TC._USER_ROLE: {"gemini-3.7-flash": 1, "gemini-3.8-flash": 1}}
    assert rep["gemini_models_answered_by_domain"]["retail"][TC._USER_ROLE] == {
        "gemini-3.7-flash": 1, "gemini-3.8-flash": 1}
    assert rep["nl_judge_models_by_domain"] == {"retail": {"gemini-3.8-flash": 2}}
    assert rep["user_llm_models"] == [f"gemini/{m}" for m in base.GEMINI_CASCADE]
    assert rep["user_llm_source"] == rep["eval_llm_source"] == "default"
    params = rep["gemini_call_parameters"]
    assert params["user_simulator"] == {"temperature": 0.0}
    assert params["nl_assertion_judge"] == {"temperature": 0.0}
    assert params["thinking"].startswith("none sent")
    reason = r["leaderboard_comparable_reason"]
    assert "gbench's Gemini cascade gemini-3.8-flash > gemini-3.7-flash" in reason


# --------------------------------------------------------------------------- #
# keys: the container holds the relay's placeholder only
# --------------------------------------------------------------------------- #
def test_the_placeholder_needs_the_relay_address_and_a_real_key_is_refused_in_the_image(
        monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", PLACEHOLDER)
    ok, why = TC.check_gemini_route()
    assert not ok and GR.RELAY_URL_ENV in why
    with pytest.raises(TC.GeminiRouteError):
        TC._gemini_native_base()
    for bad in ("http://user:pw@172.17.0.1:5", "https://172.17.0.1:5", "http://172.17.0.1",
                "http://172.17.0.1:5/v1beta", "not a url"):
        monkeypatch.setenv(GR.RELAY_URL_ENV, bad)
        assert TC.check_gemini_route()[0] is False, bad
    monkeypatch.setenv(GR.RELAY_URL_ENV, "http://172.17.0.1:4567")
    assert TC.check_gemini_route() == (True, "")
    assert TC._gemini_native_base() == "http://172.17.0.1:4567/v1beta"
    assert TC._gemini_route() == {"api_base": "http://172.17.0.1:4567/v1beta",
                                  "api_key": PLACEHOLDER}
    monkeypatch.setenv("GEMINI_API_KEY", REAL_KEY)
    assert TC._gemini_route() == {} and TC._gemini_native_base() == TC._GEMINI_API_BASE
    assert TC.check_gemini_route() == (True, ""), "outside the image a real key goes direct"
    monkeypatch.setattr(TC, "_in_image", lambda: True)
    ok, why = TC.check_gemini_route()
    assert not ok and "real Gemini key reached the tau2 container" in why


def test_the_image_does_not_ping_gemini_with_the_placeholder(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", PLACEHOLDER)
    monkeypatch.setenv(GR.RELAY_URL_ENV, "http://172.17.0.1:4567")
    for name in ("check_tau2_revision", "check_tau2_dependencies", "check_role_credentials"):
        monkeypatch.setattr(TC, name, lambda: (True, ""))
    monkeypatch.setattr(TC, "_import_tau2", lambda: (True, ""))
    monkeypatch.setattr(base, "gemini_key_live_valid",
                        lambda key: pytest.fail("the host checks the real key"))
    assert TC.check_tau_env_prerequisites() == (True, "")


def test_the_cascade_call_goes_to_the_relay_with_the_placeholder(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", PLACEHOLDER)
    monkeypatch.setenv(GR.RELAY_URL_ENV, "http://172.17.0.1:4567")
    seen = []
    casc = TC._build_gemini_cascade(lambda **kw: seen.append(kw) or "ok", 0.0,
                                    cascade_fn=lambda: ["gemini-3.8-flash"])
    assert casc(model="gemini/gemini-3.8-flash", messages=[], temperature=0.0) == "ok"
    assert seen == [{"model": "gemini/gemini-3.8-flash", "messages": [], "temperature": 0.0,
                     "api_base": "http://172.17.0.1:4567/v1beta", "api_key": PLACEHOLDER,
                     "num_retries": 0}]


class _FakeGemini:
    """A stand-in for the Gemini API on 127.0.0.1: records each request, answers
    batchEmbedContents with one vector per text."""

    def __init__(self):
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                return

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                payload = json.loads(body or b"{}")
                outer.requests.append({"path": self.path,
                                       "key": self.headers.get("x-goog-api-key"),
                                       "body": payload})
                data = json.dumps({"embeddings": [{"values": [0.1, 0.2]}]
                                   * len(payload.get("requests") or [])}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def test_the_embedder_reaches_gemini_only_through_the_relay(monkeypatch):
    """End to end on loopback: the relay (a real GeminiRelay) holds the real key; the embedder
    holds the placeholder, sends to the relay, and the stand-in Gemini API sees the real key."""
    up = _FakeGemini()
    try:
        monkeypatch.setenv("GEMINI_API_KEY", REAL_KEY)
        monkeypatch.setenv("GEMINI_OPENAI_BASE_URL",
                           f"http://127.0.0.1:{up.port}/v1beta/openai/")
        relay = GR.GeminiRelay("tau3", "127.0.0.1", openai_calls=(),
                               native_methods=("generateContent", "batchEmbedContents"),
                               native_models=[TC.GEMINI_EMBED_MODEL]).start()
        try:
            monkeypatch.setenv("GEMINI_API_KEY", relay.placeholder)
            monkeypatch.setenv(GR.RELAY_URL_ENV, relay.origin)
            vectors = TC.GeminiEmbedder()._embed_batch(["a", "b"])
        finally:
            relay.stop()
        assert vectors == [[0.1, 0.2], [0.1, 0.2]]
        (req,) = up.requests
        assert req["path"] == f"/v1beta/models/{TC.GEMINI_EMBED_MODEL}:batchEmbedContents"
        assert req["key"] == REAL_KEY, "the relay added the real key"
        assert relay.record()["requests_relayed"] == 1
    finally:
        up.close()


# --------------------------------------------------------------------------- #
# the host: the Gemini gate and the relays
# --------------------------------------------------------------------------- #
def test_the_gemini_gate_runs_on_the_host(monkeypatch):
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*GEMINI_API_KEY"):
        TC._require_gemini_access("tau2", "docs/evals/tau2.md")
    monkeypatch.setenv("GEMINI_API_KEY", REAL_KEY)
    monkeypatch.setattr(base, "gemini_key_live_valid", lambda key: (False, "HTTP 401"))
    with pytest.raises(RuntimeError, match="refused the key.*HTTP 401"):
        TC._require_gemini_access("tau2", "docs/evals/tau2.md")
    why = f"{base.MODELS_NOT_SERVED_PREFIX} 'gemini-x' from GBENCH_JUDGE_MODELS (HTTP 404)"
    monkeypatch.setattr(base, "gemini_key_live_valid", lambda key: (False, why))
    with pytest.raises(RuntimeError, match="user simulator and NL-assertion judge may"):
        TC._require_gemini_access("tau2", "docs/evals/tau2.md")
    asked = []
    monkeypatch.setattr(base, "gemini_key_live_valid", lambda key: (True, ""))
    monkeypatch.setattr(base, "gemini_models_available",
                        lambda key=None, models=None: asked.append(models) or (True, ""))
    TC._require_gemini_access("tau2", "docs/evals/tau2.md")
    assert asked == [], "the cascade's models are checked by gemini_key_live_valid"
    monkeypatch.setenv("GBENCH_TAU2_USER_LLM", "gemini/gemini-own")
    TC._require_gemini_access("tau2", "docs/evals/tau2.md")
    assert asked == [[("gemini-own", "GBENCH_TAU2_USER_LLM")]]


class _Relay:
    started = []

    def __init__(self, eval_name, bind, **kw):
        self.kw, self.bind = kw, bind
        self.placeholder = PLACEHOLDER
        self.origin = f"http://{bind}:4567"
        self.stopped = False
        _Relay.started.append(self)

    def start(self):
        return self

    def stop(self):
        self.stopped = True

    def record(self):
        return {"requests_relayed": 0}


class _EndpointRelay:
    made = []

    def __init__(self, upstream, bind, key=None, name="", **kw):
        self.upstream, self.key, self.kw = upstream, key, kw
        self.url = f"http://{bind}:4568/v1"
        self.stopped = False
        _EndpointRelay.made.append(self)

    def start(self):
        return self

    def stop(self):
        self.stopped = True

    def record(self):
        return {"requests_relayed": 0}


@pytest.fixture
def relays(monkeypatch):
    from gbench.runners.eval_suites import swebench_common
    _Relay.started, _EndpointRelay.made = [], []
    probes = []
    monkeypatch.setattr(TC, "_require_gemini_access", lambda *a: probes.append(("gate",) + a))
    monkeypatch.setattr(GR, "network_gateway", lambda *a: "172.17.0.1")
    monkeypatch.setattr(GR, "GeminiRelay", _Relay)
    monkeypatch.setattr(swebench_common, "EndpointRelay", _EndpointRelay)
    monkeypatch.setattr(TC, "_require_relay_reachable",
                        lambda eval_name, image, host, port, docs, relay:
                        probes.append((relay, host, port)))
    return probes


def test_the_container_gets_the_placeholder_the_address_and_the_cascade(relays, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", REAL_KEY)
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-3.7-flash,gemini-3.6-flash")
    with TC._bridge_services("tau3", "img", "http://127.0.0.1:9/v1",
                             "http://host.docker.internal:9/v1", ["banking_knowledge"],
                             "docs/evals/tau3.md") as bridge:
        env = bridge.env
        (relay,) = _Relay.started
        assert relay.bind == "172.17.0.1" and relay.kw["openai_calls"] == ()
        assert relay.kw["native_methods"] == ["generateContent", "batchEmbedContents"]
        assert relay.kw["native_models"] == ["gemini-3.7-flash", "gemini-3.6-flash",
                                             TC.GEMINI_EMBED_MODEL]
        assert bridge.agent_base_url == "http://host.docker.internal:9/v1"
        assert _EndpointRelay.made == [], "no real endpoint key: no endpoint relay"
    assert env == {"GEMINI_API_KEY": PLACEHOLDER, GR.RELAY_URL_ENV: "http://172.17.0.1:4567",
                   "GBENCH_JUDGE_MODELS": "gemini-3.7-flash,gemini-3.6-flash",
                   # where the host's cascade came from (recorded as the roles' source)
                   "TAU_JUDGE_CASCADE_SOURCE": "GBENCH_JUDGE_MODELS"}
    assert relays == [("gate", "tau3", "docs/evals/tau3.md"),
                      ("Gemini relay", "172.17.0.1", 4567)]
    assert relay.stopped
    assert REAL_KEY not in json.dumps(env)
    launch = TC._launch_env(dict(os.environ), bridge_env=env)
    assert launch["GEMINI_API_KEY"] == PLACEHOLDER and REAL_KEY not in json.dumps(launch)


def test_a_real_endpoint_key_stays_with_the_endpoint_relay(relays, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", REAL_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", ENDPOINT_KEY)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp:3128")
    with TC._bridge_services("tau2", "img", "https://models.example.com/v1", None, ["retail"],
                             "docs/evals/tau2.md") as bridge:
        (ep,) = _EndpointRelay.made
        assert ep.upstream == "https://models.example.com/v1" and ep.key == ENDPOINT_KEY
        assert ep.kw == {"smoke_from_env": False}   # a tau relay is no smoke relay
        assert bridge.agent_base_url == "http://172.17.0.1:4568/v1"
        assert _Relay.started[0].kw["native_methods"] == ["generateContent"]
        env = bridge.env
    assert env[TC._AGENT_KEY_ENV] == "EMPTY"
    assert env["NO_PROXY"].split(",")[-1] == "172.17.0.1" and "172.17.0.1" in env["no_proxy"]
    assert ("endpoint relay", "172.17.0.1", 4568) in relays and ep.stopped
    launch = TC._launch_env(dict(os.environ), bridge_env=env)
    assert ENDPOINT_KEY not in json.dumps(launch) and launch[TC._AGENT_KEY_ENV] == "EMPTY"
    assert bridge.record()["endpoint_relay"] == {"requests_relayed": 0}


def test_the_relays_stop_when_a_probe_fails(relays, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", REAL_KEY)

    def closed(*a):
        raise RuntimeError("[INFRA REQUIRED] tau2 ... firewall")
    monkeypatch.setattr(TC, "_require_relay_reachable", closed)
    with pytest.raises(RuntimeError, match="firewall"):
        with TC._bridge_services("tau2", "img", "http://127.0.0.1:9/v1", None, ["retail"],
                                 "docs/evals/tau2.md"):
            pytest.fail("the block must not run")
    assert _Relay.started[0].stopped


def test_run_tau_suite_with_a_real_endpoint_key_checks_only_the_metadata_server(monkeypatch,
                                                                                  tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", ENDPOINT_KEY)
    monkeypatch.setenv("GBENCH_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(TC, "check_tau_image", lambda: (True, ""))
    monkeypatch.setattr(TC, "_image_id", lambda image: "sha256:x")
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda *a, **k: pytest.fail("the container never reaches the endpoint"))
    guarded = []
    monkeypatch.setattr(cg, "require_metadata_blocked",
                        lambda *a, **k: guarded.append((a, k)))
    removed = []
    monkeypatch.setattr(TC.subprocess, "run",
                        lambda cmd, **kw: removed.append((cmd, kw)) or
                        subprocess.CompletedProcess(cmd, 0, "", ""))

    @contextlib.contextmanager
    def bridge(eval_name, image, base_url, agent_base_url, domains, docs_url):
        assert agent_base_url is None
        b = TC._TauBridge("http://172.17.0.1:4568/v1")
        b.env = {"GEMINI_API_KEY": PLACEHOLDER, TC._AGENT_KEY_ENV: "EMPTY"}
        yield b
    monkeypatch.setattr(TC, "_bridge_services", bridge)
    seen = {}

    def launch(cmd, env=None):
        io = next(a.split(":")[0] for a in cmd if a.endswith(":/gbench_io:rw"))
        with open(os.path.join(io, "job.json"), encoding="utf-8") as fh:
            seen["job"] = json.load(fh)
        seen["env"] = env
        with open(os.path.join(io, "result.json"), "w", encoding="utf-8") as fh:
            json.dump({"ok": True, "result": {"tau2_report": {}}}, fh)
        return 0, ""
    monkeypatch.setattr(TC, "_launch_image", launch)
    r = TC.run_tau_suite("tau2", ["airline"], "m", "https://models.example.com/v1", 1, None,
                         "docs/evals/tau2.md")
    assert guarded == [(("tau2", TC.tau2_image(), TC.ENDPOINT_NETWORK, "docs/evals/tau2.md"),
                        {"resolve": []})]
    assert seen["job"]["base_url"] == "http://172.17.0.1:4568/v1"
    assert ENDPOINT_KEY not in json.dumps(seen["env"])
    assert "gemini_relay" in r["tau2_report"] and "endpoint_relay" in r["tau2_report"]
    rm = [(c, k) for c, k in removed if c[:3] == ["docker", "rm", "-f"]]
    assert len(rm) == 1 and rm[0][1].get("timeout") == cg.PROBE_RM_TIMEOUT_S


# --------------------------------------------------------------------------- #
# the endpoint probe's cleanup after a timeout
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rm_error", [subprocess.TimeoutExpired(["docker"], 30),
                                      OSError("docker gone"), None])
def test_a_timed_out_probe_is_removed_with_a_bound_and_the_infra_error_survives(monkeypatch,
                                                                              rm_error):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        if cmd[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout"))
        if rm_error is not None:
            raise rm_error
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(TC.subprocess, "run", run)
    with pytest.raises(RuntimeError) as ei:
        TC._require_endpoint_reachable("tau2", "img", "http://127.0.0.1:9/v1",
                                       "http://host.docker.internal:9/v1", "docs/evals/tau2.md")
    assert "[INFRA REQUIRED] tau2" in str(ei.value) and "TimeoutExpired" in str(ei.value)
    rm = [(c, k) for c, k in calls if c[:3] == ["docker", "rm", "-f"]]
    assert len(rm) == 1 and rm[0][1]["timeout"] == cg.PROBE_RM_TIMEOUT_S
    assert rm[0][0][3].startswith("gbench-endpoint-probe-")


def test_a_closed_relay_port_names_the_firewall(monkeypatch):
    monkeypatch.setattr(TC.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, "GBENCH_ENDPOINT_PROBE=closed ConnectionRefusedError 111\n", ""))
    with pytest.raises(RuntimeError) as ei:
        TC._require_relay_reachable("tau3", "img", "172.17.0.1", 4567, "docs/evals/tau3.md",
                                    "Gemini relay")
    msg = str(ei.value)
    assert "172.17.0.1:4567" in msg and "Gemini relay" in msg and "never holds one" in msg


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #
def test_the_docs_describe_the_cascade_the_relays_and_the_records():
    tau2 = " ".join(_read("docs", "evals", "tau2.md").split())
    tau3 = " ".join(_read("docs", "evals", "tau3.md").split())
    for text in ("GBENCH_JUDGE_MODELS", "gemini-3.8-flash", "Gemini relay",
                 "num_retries=0", "thinkingLevel", "gemini_models_answered",
                 "nl_judge_models_by_domain", "judge_model", "endpoint relay",
                 "gemini_call_parameters"):
        assert text in tau2, text
    assert "GBENCH_SEARCH_MODELS" not in tau2 and "gemini/gemini-3.6-flash" not in tau2
    assert "Gemini relay" in tau3 and "placeholder" in tau3
    assert "Only the Gemini key enters the container" not in tau3
