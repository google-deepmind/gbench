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

"""tau3: one fixed retrieval path, and a failed Gemini embedding call is a harness error.

  * single path - banking_knowledge always runs upstream's alltools with the Gemini embedder
                  (alltools-gemini); GBENCH_TAU2_RETRIEVAL_CONFIG / TAU2_RETRIEVAL_CONFIG change
                  nothing (warned, never forwarded into the image), and the helpers that only
                  served that knob are gone; the registration and key checks remain.
  * the guard   - a tool call during which a Gemini embedding call failed after its retries raises
                  TauHarnessError out of Environment.get_response instead of handing the agent an
                  `Error: ...` tool result; other tool errors still reach the agent; the marker is
                  per thread and per tool call; the guard is installed once, for banking_knowledge.
  * end to end  - run_tau_env over a stand-in tau2 whose run_domain retries a failed simulation
                  attempt the way tau2's run_with_retry does: the failures are counted per domain,
                  the status is completed_with_errors, leaderboard_comparable_reason names them,
                  and no failure is scored as the agent's outcome.
  * real tau2   - the guard against tau2's own Environment.get_response and run_with_retry.
  * docs        - tau3.md / tau2.md state both as coded.

No Docker, no network (urllib.request.urlopen is stubbed), no model server.
"""

import inspect
import io
import json
import os
import sys
import threading
import types
import urllib.error
import urllib.request
from unittest import mock

import pytest

from gbench.runners.eval_suites import tau_common as TC

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEMINI_KEY = "gemini-single-path-key-0123456789"
ENDPOINT_KEY = "endpoint-single-path-key-99"
EMBED_REASON = "retrieval embeddings by gemini-embedding-001 (upstream: text-embedding-3-large)"
KNOBS = ("GBENCH_TAU2_RETRIEVAL_CONFIG", "TAU2_RETRIEVAL_CONFIG")


def _read(*parts):
    with open(os.path.join(_REPO, *parts), encoding="utf-8") as fh:
        return " ".join(fh.read().split())


@pytest.fixture
def clean(monkeypatch):
    for name in KNOBS + ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GBENCH_TAU2_USER_LLM",
                         "TAU2_USER_LLM", "GBENCH_TAU2_EVAL_LLM", "TAU2_EVAL_LLM",
                         "GBENCH_TAU2_NUM_TRIALS", "TAU2_NUM_TRIALS", "OPENAI_API_KEY",
                         TC._AGENT_KEY_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(TC, "_SECRETS", [])
    monkeypatch.setattr(TC, "_AGENT_ROUTE", {})
    monkeypatch.setattr(TC, "_LAST_RETRIEVAL", {})
    monkeypatch.setattr(TC, "_LLM_FAILURES", {})
    monkeypatch.setattr(TC, "_EMBEDDING_FAILURES", {})
    monkeypatch.setattr(TC, "_EMBEDDING_ABORTS", {})
    monkeypatch.setattr(TC, "_CURRENT_DOMAIN", {"domain": None})
    monkeypatch.setattr(TC, "_EMBED_FAILURE_SEEN", threading.local())
    monkeypatch.setattr(TC.time, "sleep", lambda s: None)


# --------------------------------------------------------------------------- #
# a stand-in for the Gemini API: texts listed in `outage` fail on every request, texts in `flaky`
# fail for their first `flaky[text]` HTTP requests (one embedding call makes up to 4)
# --------------------------------------------------------------------------- #
class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _GeminiAPI:
    def __init__(self):
        self.outage, self.flaky, self.requests = set(), {}, 0

    def __call__(self, request, timeout=None):
        self.requests += 1
        texts = [r["content"]["parts"][0]["text"]
                 for r in json.loads(request.data.decode("utf-8"))["requests"]]
        if any(t in self.outage for t in texts):
            raise urllib.error.HTTPError(request.full_url, 503, "unavailable", {},
                                         io.BytesIO(b"UNAVAILABLE"))
        for t in texts:
            if self.flaky.get(t, 0) > 0:
                self.flaky[t] -= 1
                raise urllib.error.HTTPError(request.full_url, 500, "internal", {},
                                             io.BytesIO(b"INTERNAL"))
        body = {"embeddings": [{"values": [float(len(t)), 0.5, 0.25]} for t in texts]}
        return _Resp(json.dumps(body).encode("utf-8"))


@pytest.fixture
def api(monkeypatch):
    fake = _GeminiAPI()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


# --------------------------------------------------------------------------- #
# a stand-in for the tau2 surface a banking_knowledge run touches (upstream's attribute names)
# --------------------------------------------------------------------------- #
class _Spec(types.SimpleNamespace):
    pass


def _alltools():
    return types.SimpleNamespace(
        name="alltools", prompt_template="prompts/all_tools.md",
        kb_search=None, kb_search_bm25=_Spec(type="bm25"),
        kb_search_dense=_Spec(type="embedding", embedder_type="openai",
                              embedder_model="text-embedding-3-large", top_k=10),
        shell=_Spec(allow_writes=False))


class _StandIn:
    """Environment.get_response as upstream writes it (every tool exception becomes an
    `Error: ...` tool result with error=True), a KB_search_dense tool that embeds its query with
    the registered embedder, and a run_domain that re-runs a failed simulation attempt up to
    max_retries times and records an infrastructure error once they are spent, as tau2's
    run_with_retry does. A simulation scores 1.0 unless the agent saw a tool error."""

    def __init__(self, queries, trials=1, max_retries=3):
        self.queries, self.trials, self.max_retries = list(queries), trials, max_retries
        self.configs, self.tool_contents, self.attempts = [], [], []

    def modules(self):
        me = self
        names = ["tau2", "tau2.utils", "tau2.utils.llm_utils", "tau2.data_model",
                 "tau2.data_model.simulation", "tau2.runner", "tau2.runner.checkpoint",
                 "tau2.environment", "tau2.environment.environment", "tau2.knowledge",
                 "tau2.knowledge.sandbox_manager", "tau2.run", "tau2.metrics",
                 "tau2.metrics.agent_metrics", "litellm", "tau2.domains",
                 "tau2.domains.banking_knowledge", "tau2.domains.banking_knowledge.retrieval",
                 "tau2.domains.banking_knowledge.environment",
                 "tau2.knowledge.document_preprocessors",
                 "tau2.knowledge.document_preprocessors.embedding_indexer",
                 "tau2.knowledge.input_preprocessors",
                 "tau2.knowledge.input_preprocessors.embedding_encoder"]
        m = {n: types.ModuleType(n) for n in names}
        for n in names:
            parent, _, child = n.rpartition(".")
            if parent in m:
                setattr(m[parent], child, m[n])
        m["litellm"].completion = lambda **kw: {"choices": [{"message": {"content": "ok"}}]}
        m["tau2.utils.llm_utils"].completion = m["litellm"].completion
        m["tau2.runner.checkpoint"].json = json
        import subprocess
        m["tau2.knowledge.sandbox_manager"].subprocess = subprocess

        class TextRunConfig:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        class Results:
            def __init__(self, simulations):
                self.simulations = simulations

            def model_dump_json(self, indent=None):
                return json.dumps({"n": len(self.simulations)})

            def save(self, path, format="json"):
                with open(path, "w") as fh:
                    fh.write(self.model_dump_json())

        m["tau2.data_model.simulation"].TextRunConfig = TextRunConfig
        m["tau2.data_model.simulation"].Results = Results

        r = m["tau2.domains.banking_knowledge.retrieval"]
        r.RETRIEVAL_VARIANTS = {"alltools": _alltools(),
                                "bm25": types.SimpleNamespace(name="bm25", kb_search=_Spec(
                                    type="bm25"), kb_search_dense=None, shell=None)}
        r.format_all_tools_dense_instructions = lambda v: (
            f"The `KB_search_dense` tool uses **{v.kb_search_dense.embedder_type}** with "
            f"embedding model `{v.kb_search_dense.embedder_model}` for dense retrieval.")
        r.get_or_create_docs = lambda kb: list(kb)
        m["tau2.domains.banking_knowledge.environment"].get_knowledge_base = (
            lambda: [{"id": "d1", "text": "doc one"}, {"id": "d2", "text": "doc two"}])
        idx = m["tau2.knowledge.document_preprocessors.embedding_indexer"]
        enc = m["tau2.knowledge.input_preprocessors.embedding_encoder"]
        idx.EMBEDDER_REGISTRY = {"openai": object}
        enc.EMBEDDER_REGISTRY = {"openai": object}

        class EmbeddingIndexer:
            def __init__(self, embedder_type="openai", embedder_params=None, **kw):
                self.embedder_type, self.params = embedder_type, embedder_params or {}

            def process(self, documents, state):
                idx.EMBEDDER_REGISTRY[self.embedder_type](**self.params).embed(
                    [d["text"] for d in documents])
                return documents

        idx.EmbeddingIndexer = EmbeddingIndexer

        class ToolMessage:
            def __init__(self, content, error=False):
                self.content, self.error = content, error

        class Environment:
            def __init__(self, variant):
                self.variant = variant

            def make_tool_call(self, name, requestor="assistant", **kwargs):
                if name != "KB_search_dense":
                    raise ValueError(f"unknown tool {name}")
                query = kwargs.get("query") or ""
                if not query.strip():
                    raise ValueError("Empty or missing input for key 'query'")
                spec = self.variant.kb_search_dense
                enc.EMBEDDER_REGISTRY[spec.embedder_type](model=spec.embedder_model).embed([query])
                return f"1. a document about {query}"

            def get_response(self, message):
                try:
                    resp = self.make_tool_call(message.name, requestor=message.requestor,
                                               **message.arguments)
                    error = False
                except Exception as e:                           # noqa: BLE001 - as upstream
                    resp, error = f"Error: {e}", True
                return ToolMessage(resp, error=error)

        m["tau2.environment.environment"].Environment = Environment
        me.Environment = Environment

        def run_domain(config):
            me.configs.append(config)
            variant = r.RETRIEVAL_VARIANTS[config.retrieval_config]
            sims = []
            for trial in range(me.trials):
                for task_id, query in enumerate(me.queries):
                    sim = None
                    for attempt in range(me.max_retries + 1):
                        me.attempts.append((task_id, trial, attempt))
                        call = types.SimpleNamespace(name="KB_search_dense",
                                                     arguments={"query": query},
                                                     requestor="assistant")
                        try:
                            tool = Environment(variant).get_response(call)
                        except Exception:                        # noqa: BLE001 - retried
                            continue
                        me.tool_contents.append(tool.content)
                        reward = 0.0 if tool.error else 1.0
                        sim = types.SimpleNamespace(
                            task_id=task_id, trial=trial, termination_reason="agent_stop",
                            reward_info=types.SimpleNamespace(reward=reward))
                        break
                    if sim is None:
                        sim = types.SimpleNamespace(task_id=task_id, trial=trial,
                                                    termination_reason="infrastructure_error",
                                                    reward_info=None)
                    sims.append(sim)
            os.makedirs(config.save_to, exist_ok=True)
            Results(sims).save(os.path.join(config.save_to, "results.json"))
            return types.SimpleNamespace(simulations=sims, num_trials=me.trials)

        def compute_metrics(results):
            scored = [s.reward_info.reward for s in results.simulations
                      if s.reward_info is not None]
            return types.SimpleNamespace(
                pass_hat_ks={1: sum(scored) / len(scored)} if scored else {})

        m["tau2.run"].run_domain = run_domain
        m["tau2.metrics.agent_metrics"].compute_metrics = compute_metrics
        return m


def _stub_prereqs(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    monkeypatch.setenv(TC._AGENT_KEY_ENV, ENDPOINT_KEY)
    monkeypatch.setenv("GBENCH_TAU2_SAVE_TRACES", str(tmp_path / "traces"))
    monkeypatch.setenv("GBENCH_TAU2_NUM_TRIALS", "1")
    for name, value in (("check_tau_env_prerequisites", lambda: (True, "")),
                        ("check_banking_prerequisites", lambda: (True, "")),
                        ("_install_robustness_patches", lambda *a, **k: None),
                        ("_configure_evaluator_llm", lambda *a, **k: None),
                        ("_quiet_tau2_noise", lambda: None),
                        ("_prune_stale_search_sandboxes", lambda: None),
                        ("_refuse_resume_prompts", lambda: None),
                        ("_tau2_version", lambda: "1.0.1")):
        monkeypatch.setattr(TC, name, value)


def _run(stand_in):
    with mock.patch.dict(sys.modules, stand_in.modules()):
        return TC.run_tau_env("tau3", ["banking_knowledge"], "m", "http://x/v1", 1, None,
                              "docs/evals/tau3.md")


# --------------------------------------------------------------------------- #
# single path
# --------------------------------------------------------------------------- #
def test_the_knob_helpers_are_gone_and_only_the_removed_knob_tuple_names_it():
    for gone in ("_retrieval_config_name", "_is_upstream_default", "_retrieval_variant",
                 "_variant_providers", "_service_free_configs", "_dense_embedders",
                 "_retrieval_has_shell", "_PROVIDER_NAMES", "_RETRIEVAL_ALIASES"):
        assert not hasattr(TC, gone), gone
    src = inspect.getsource(TC)
    assert src.count('"GBENCH_TAU2_RETRIEVAL_CONFIG"') == 1
    assert src.count('"TAU2_RETRIEVAL_CONFIG"') == 1
    assert set(KNOBS) <= set(TC._REMOVED_KNOBS) and set(KNOBS) <= set(TC._NOT_FORWARDED)


@pytest.mark.parametrize("knob, value", [(None, None), ("GBENCH_TAU2_RETRIEVAL_CONFIG", "bm25"),
                                         ("TAU2_RETRIEVAL_CONFIG", "alltools-qwen"),
                                         ("GBENCH_TAU2_RETRIEVAL_CONFIG", "alltools-gemini"),
                                         ("GBENCH_TAU2_RETRIEVAL_CONFIG", "no-such-config")])
def test_setup_always_runs_alltools_with_the_gemini_embedder(clean, api, monkeypatch, knob, value):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    if knob:
        monkeypatch.setenv(knob, value)
    stand_in = _StandIn(["q"])
    mods = stand_in.modules()
    with mock.patch.dict(sys.modules, mods):
        assert TC._setup_banking_retrieval() == ("alltools-gemini", None)
    variants = mods["tau2.domains.banking_knowledge.retrieval"].RETRIEVAL_VARIANTS
    dense = variants["alltools-gemini"].kb_search_dense
    assert (dense.embedder_type, dense.embedder_model) == ("gemini", "gemini-embedding-001")
    assert TC._LAST_RETRIEVAL["retrieval_config"] == "alltools"
    assert TC._LAST_RETRIEVAL["retrieval_variant"] == "alltools-gemini"
    assert TC._LAST_RETRIEVAL["embedders"] == ["gemini:gemini-embedding-001"]
    assert api.requests == 1, "the documents are embedded once, through the Gemini API"
    cfg = TC._run_config("tau3", ["banking_knowledge"], "m", False, None, 4)
    assert cfg["retrieval_config"] == "alltools"
    assert cfg["retrieval_embedders"] == ["gemini:gemini-embedding-001"]
    assert TC._run_config("tau2", ["retail"], "m", False, None, 4)["retrieval_config"] is None


def test_the_removed_knob_warns_and_is_not_forwarded(clean, monkeypatch, caplog):
    monkeypatch.setenv("GBENCH_TAU2_RETRIEVAL_CONFIG", "bm25")
    monkeypatch.setenv("TAU2_RETRIEVAL_CONFIG", "terminal_use")
    monkeypatch.setattr(TC, "check_tau_image", lambda: (False, "no image"))
    with caplog.at_level("WARNING"), pytest.raises(RuntimeError):
        TC.run_tau_suite("tau3", ["banking_knowledge"], "m", "http://x/v1", 1, None,
                         "docs/evals/tau3.md")
    assert "GBENCH_TAU2_RETRIEVAL_CONFIG, TAU2_RETRIEVAL_CONFIG no longer change anything" \
        in caplog.text
    assert "always runs upstream's alltools retrieval" in caplog.text
    names = TC._forwarded_env(dict(os.environ, GEMINI_API_KEY=GEMINI_KEY),
                              domains=["banking_knowledge"],
                              bridge_env={"GEMINI_API_KEY": "gbench-gemini-relay-" + "0" * 48})
    assert not set(KNOBS) & set(names)
    assert "GEMINI_API_KEY" in names, "the name carries the Gemini relay's placeholder"


def test_the_registration_and_key_checks_remain(clean, monkeypatch):
    stand_in = _StandIn(["q"])
    mods = stand_in.modules()
    registry = mods["tau2.domains.banking_knowledge.retrieval"].RETRIEVAL_VARIANTS
    with mock.patch.dict(sys.modules, mods):
        ok, why = TC.check_retrieval_credentials()
        assert not ok and "needs GEMINI_API_KEY (or GEMINI_API_KEYS)" in why
        monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
        monkeypatch.setenv("GBENCH_TAU2_RETRIEVAL_CONFIG", "bm25")      # removed: no effect
        assert TC.check_retrieval_credentials() == (True, "")
        registry["alltools"].kb_search_dense = None
        ok, why = TC.check_retrieval_credentials()
        assert not ok and "no dense embedding pipeline" in why and "docs/evals/tau3.md" in why
        with pytest.raises(RuntimeError, match="no dense embedding pipeline"):
            TC._register_gemini_retrieval()
        del registry["alltools"]
        ok, why = TC.check_retrieval_credentials()
        assert not ok and "has no alltools config" in why


def test_banking_prerequisites_always_probe_the_shell(clean, monkeypatch):
    probes = []
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    monkeypatch.setenv("GBENCH_TAU2_RETRIEVAL_CONFIG", "bm25")          # removed: no effect
    monkeypatch.setattr(TC, "check_shell_sandbox", lambda: probes.append(1) or (True, ""))
    monkeypatch.setitem(sys.modules, "rank_bm25", types.ModuleType("rank_bm25"))
    with mock.patch.dict(sys.modules, _StandIn(["q"]).modules()):
        assert TC.check_banking_prerequisites() == (True, "")
    assert probes == [1]


# --------------------------------------------------------------------------- #
# the guard
# --------------------------------------------------------------------------- #
def _guarded_env(stand_in):
    """A stand-in Environment for alltools-gemini whose get_response carries the guard."""
    mods = stand_in.modules()
    with mock.patch.dict(sys.modules, mods):
        variant = TC._register_gemini_retrieval()
    cls = stand_in.Environment
    cls.get_response = TC._embedding_failure_guard(cls.get_response)
    return cls(variant)


def _call(query):
    return types.SimpleNamespace(name="KB_search_dense", arguments={"query": query},
                                 requestor="assistant")


def test_a_failed_query_embedding_stops_the_attempt_instead_of_reaching_the_agent(
        clean, api, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    TC._CURRENT_DOMAIN["domain"] = "banking_knowledge"
    stand_in = _StandIn(["q"])
    env = _guarded_env(stand_in)
    unguarded = stand_in.Environment.get_response.__wrapped__
    api.outage.add("card fees")
    # what upstream does without the guard: the agent reads the outage as its own tool error
    tool = unguarded(env, _call("card fees"))
    assert tool.error and tool.content.startswith("Error: Gemini embeddings (gemini-embedding-001)"
                                                  " failed: HTTP 503")
    assert api.requests == 4, "the embedder's own retries ran first"
    # with the guard: a harness error, raised out of the tool call
    with pytest.raises(TC.TauHarnessError) as ei:
        env.get_response(_call("card fees"))
    msg = str(ei.value)
    assert msg.startswith("harness error, not the agent's outcome: Gemini embeddings")
    assert "during KB_search_dense" in msg and GEMINI_KEY not in msg
    assert TC.embedding_failures() == {"banking_knowledge": 2}
    assert TC._EMBEDDING_ABORTS == {"banking_knowledge": 1}
    assert TC.llm_failures() == {"embedding_error": 2}
    # the next call on the same thread starts clean
    assert env.get_response(_call("wire limits")).content == "1. a document about wire limits"


def test_other_tool_errors_still_reach_the_agent(clean, api, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    env = _guarded_env(_StandIn(["q"]))
    tool = env.get_response(_call("   "))                       # the agent's blank query
    assert tool.error and "Empty or missing input" in tool.content
    tool = env.get_response(types.SimpleNamespace(name="no_such_tool", arguments={},
                                                  requestor="assistant"))
    assert tool.error and "unknown tool" in tool.content
    assert TC.embedding_failures() == {} and TC.llm_failures() == {}


def test_the_marker_is_per_thread(clean, api, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    env = _guarded_env(_StandIn(["q"]))
    started, release, seen = threading.Event(), threading.Event(), {}
    orig = TC._EMBED_FAILURE_SEEN

    def other_thread():
        # a failure recorded on this thread while the main thread is inside a tool call
        TC._note_embedding_failure("Gemini embeddings (gemini-embedding-001) failed: boom")
        seen["other"] = getattr(orig, "message", None)
        started.set()
        release.wait(5)

    t = threading.Thread(target=other_thread)
    t.start()
    assert started.wait(5)
    try:
        assert env.get_response(_call("savings rates")).error is False
    finally:
        release.set()
        t.join(5)
    assert seen["other"].endswith("boom")


def test_the_guard_is_installed_once_and_only_for_banking_knowledge(clean):
    stand_in = _StandIn(["q"])
    mods = stand_in.modules()
    with mock.patch.dict(sys.modules, mods):
        Environment = mods["tau2.environment.environment"].Environment
        TC._install_credential_guards("airline")
        assert getattr(Environment.get_response, "_gbench_redacting", False)
        assert not getattr(Environment.get_response, "_gbench_embedding_guard", False)
        TC._install_credential_guards("banking_knowledge")
        guarded = Environment.get_response
        assert guarded._gbench_embedding_guard and guarded._gbench_redacting
        TC._install_credential_guards("banking_knowledge")
        TC._install_credential_guards("retail")
        assert Environment.get_response is guarded, "not wrapped twice"


def test_the_guard_cannot_be_skipped(clean):
    """No Environment to guard is a hard error (the run never starts without it)."""
    mods = _StandIn(["q"]).modules()
    mods["tau2.environment.environment"] = None
    with mock.patch.dict(sys.modules, mods), pytest.raises(ImportError):
        TC._install_credential_guards("banking_knowledge")


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_embedding_failures_are_harness_errors_in_the_result(clean, api, monkeypatch, tmp_path):
    _stub_prereqs(monkeypatch, tmp_path)
    api.outage.add("card fees")          # every attempt of this simulation fails
    api.flaky["wire limits"] = 4         # one embedding call fails (4 HTTP attempts), then works
    stand_in = _StandIn(["savings rates", "card fees", "wire limits"])
    r = _run(stand_in)
    # no failure reached the agent as a tool error
    assert not any(c.startswith("Error") for c in stand_in.tool_contents)
    # the outage simulation: 4 attempts, all stopped -> an infrastructure error, not scored
    assert [a for a in stand_in.attempts if a[0] == 1] == [(1, 0, i) for i in range(4)]
    # the flaky one was re-run once and scored on its clean attempt
    assert [a for a in stand_in.attempts if a[0] == 2] == [(2, 0, 0), (2, 0, 1)]
    assert r["accuracy"] == 100.0 and r["tau2_report"]["infra_errors"] == 1
    assert r["status"] == "completed_with_errors"
    assert r["harness_errors"] == [{
        "error": "Gemini embedding calls (gemini-embedding-001) failed after their retries",
        "domain": "banking_knowledge", "failed_embedding_calls": 5,
        "stopped_simulation_attempts": 5}, {
        "error": ("simulations ended in an infrastructure error from a harness failure (tau2 "
                  "records one when every attempt of a simulation failed; compute_metrics leaves "
                  "it out of pass^k)"),
        "domain": "banking_knowledge", "infrastructure_error_simulations": 1,
        "sources": {"unrecorded": 1},                   # the stand-in records no traceback
        "error_types": {"unknown": 1}, "example_error": None}]
    assert r["tau2_report"]["embedding_failures_by_domain"] == {"banking_knowledge": 5}
    assert r["llm_failures"]["embedding_error"] == 5
    reason = r["leaderboard_comparable_reason"].split("; ")
    assert r["leaderboard_comparable"] is False
    assert EMBED_REASON in reason
    assert ("5 Gemini embedding call(s) (gemini-embedding-001) failed in banking_knowledge after "
            "their retries: harness errors, not the agent's outcome (5 simulation attempt(s) "
            "stopped there and re-run by tau2, which counts a simulation whose every attempt "
            "failed as an infrastructure error)") in reason
    assert any(g.startswith("1 simulation(s) ended in infrastructure errors") for g in reason)
    assert TC._CURRENT_DOMAIN["domain"] is None
    assert GEMINI_KEY not in json.dumps(r) and ENDPOINT_KEY not in json.dumps(r)


def test_without_the_guard_the_outage_would_be_scored_as_the_agent(clean, api, monkeypatch,
                                                                   tmp_path):
    """What the guard prevents: the same outage, with the guard left out, reaches the agent as a
    tool error and scores 0 as its outcome."""
    _stub_prereqs(monkeypatch, tmp_path)
    monkeypatch.setattr(TC, "_embedding_failure_guard", lambda orig: orig)
    api.outage.add("card fees")
    stand_in = _StandIn(["savings rates", "card fees"])
    r = _run(stand_in)
    assert r["accuracy"] == 50.0 and r["tau2_report"]["infra_errors"] == 0
    assert any(c.startswith("Error: Gemini embeddings") for c in stand_in.tool_contents)


def test_a_clean_run_is_success_with_zero_failures(clean, api, monkeypatch, tmp_path):
    _stub_prereqs(monkeypatch, tmp_path)
    r = _run(_StandIn(["savings rates", "card fees"]))
    assert r["status"] == "success" and r["harness_errors"] == []
    assert r["tau2_report"]["embedding_failures_by_domain"] == {"banking_knowledge": 0}
    assert r["leaderboard_comparable_reason"].split("; ")[-1] == EMBED_REASON
    assert not any("Gemini embedding call" in g
                   for g in r["leaderboard_comparable_reason"].split("; "))


def test_a_domain_lost_to_an_embedding_outage_is_a_hard_error_naming_it(clean, api, monkeypatch,
                                                                        tmp_path):
    """Every simulation an infrastructure error: no score at all (never a partial one), and the
    hard error names the failed embedding calls."""
    _stub_prereqs(monkeypatch, tmp_path)
    api.outage.update({"card fees", "wire limits"})
    with pytest.raises(RuntimeError) as ei:
        _run(_StandIn(["card fees", "wire limits"]))
    msg = str(ei.value)
    assert msg.startswith("[INFRA REQUIRED]")
    assert "(all 2 ended in an infrastructure error; 8 Gemini embedding call(s) failed after " \
           "their retries)" in msg


def test_a_new_run_starts_from_zero(clean, api, monkeypatch, tmp_path):
    _stub_prereqs(monkeypatch, tmp_path)
    api.outage.add("card fees")
    assert _run(_StandIn(["card fees", "ok"]))["status"] == "completed_with_errors"
    api.outage.clear()
    r = _run(_StandIn(["card fees", "ok"]))
    assert r["status"] == "success" and r["llm_failures"] == {}


def test_gaps_name_only_the_run_domains(clean, monkeypatch):
    monkeypatch.setattr(TC, "_LAST_AGENT_TEMPERATURE", {"value": 0.0})
    TC._EMBEDDING_FAILURES.update({"banking_knowledge": 2})
    TC._EMBEDDING_ABORTS.update({"banking_knowledge": 1})
    assert not any("Gemini embedding call" in g for g in TC._comparability_gaps(["retail"], 4,
                                                                                None, 0))
    assert TC._harness_error_records(["retail"]) == []
    gaps = TC._comparability_gaps(["banking_knowledge"], 4, None, 0)
    assert any(g.startswith("2 Gemini embedding call(s) (gemini-embedding-001) failed in "
                            "banking_knowledge after their retries") for g in gaps)
    assert "(1 simulation attempt(s) stopped there" in " ".join(gaps)


def test_an_embedding_failure_outside_a_run_is_not_counted_per_domain(clean, api, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    api.outage.add("q")
    with pytest.raises(TC.GeminiEmbeddingError):
        TC.GeminiEmbedder().embed(["q"])
    assert TC.embedding_failures() == {} and TC.llm_failures() == {"embedding_error": 1}


# --------------------------------------------------------------------------- #
# real tau2 (the installed tau2-bench): Environment.get_response and run_with_retry
# --------------------------------------------------------------------------- #
def test_the_guard_against_tau2s_own_get_response_and_run_with_retry(clean, api, monkeypatch):
    env_mod = pytest.importorskip("tau2.environment.environment")
    from tau2.data_model.message import ToolCall
    from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
    from tau2.runner.progress import run_with_retry

    class KB(ToolKitBase):
        @is_tool(ToolType.READ)
        def KB_search_dense(self, query: str, k: int = 10) -> str:
            """Search the knowledge base using dense embedding retrieval.

            Args:
                query: The search query.
                k: Maximum number of documents to return.

            Returns:
                Documents.
            """
            TC.GeminiEmbedder().embed([query])
            return f"found {query}"

    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    TC._CURRENT_DOMAIN["domain"] = "banking_knowledge"
    # tau2's own get_response (without a guard another test may have installed on the class)
    base = env_mod.Environment.get_response
    while getattr(base, "_gbench_embedding_guard", False):
        base = base.__wrapped__
    Plain = type("Plain", (env_mod.Environment,), {"get_response": base})
    Guarded = type("Guarded", (env_mod.Environment,),
                   {"get_response": TC._embedding_failure_guard(base)})
    env = Guarded(domain_name="banking_knowledge", policy="p", tools=KB())
    call = ToolCall(id="c1", name="KB_search_dense", arguments={"query": "card fees"})
    assert env.get_response(call).content == "found card fees"
    api.outage.add("card fees")
    # tau2's own get_response turns the failure into a tool result for the agent ...
    plain = Plain(domain_name="banking_knowledge", policy="p", tools=KB())
    tool = plain.get_response(call)
    assert tool.error is True and "Gemini embeddings (gemini-embedding-001) failed" in tool.content
    # ... the guarded one raises, and tau2's run_with_retry re-runs and then records an
    # infrastructure error (never a scored simulation)
    runs = []

    def attempt():
        runs.append(1)
        env.get_response(call)
        raise AssertionError("not reached")

    sim = run_with_retry(attempt, task=types.SimpleNamespace(id="t1"), trial=0, seed=300,
                         max_retries=3, retry_delay=0.0, console_display=False)
    assert len(runs) == 4
    assert getattr(sim.termination_reason, "value", sim.termination_reason) == \
        "infrastructure_error"
    assert sim.reward_info is None and sim.info["error_type"] == "TauHarnessError"
    assert TC._EMBEDDING_ABORTS == {"banking_knowledge": 4}
    assert TC.embedding_failures() == {"banking_knowledge": 5}   # 4 guarded + 1 unguarded


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #
def test_the_docs_state_the_single_path_and_the_harness_errors():
    tau3, tau2 = _read("docs", "evals", "tau3.md"), _read("docs", "evals", "tau2.md")
    assert "This is the only retrieval configuration gbench runs: no knob selects another one" \
        in tau3
    assert "`GBENCH_TAU2_RETRIEVAL_CONFIG` (and its alias `TAU2_RETRIEVAL_CONFIG`) no longer " \
           "changes anything" in tau3
    assert "### Retrieval knobs (env)" not in tau3 and "### Removed knobs" in tau3
    for text in ("**A failed embedding call is a harness error.**", "`TauHarnessError`",
                 "`run_with_retry`", "`tau2_report.embedding_failures_by_domain`",
                 "`failed_embedding_calls`", "`stopped_simulation_attempts`",
                 "status `completed_with_errors`", "failed in banking_knowledge after their "
                 "retries: harness errors, not the agent's outcome"):
        assert text in tau3, text
    assert "reaches the agent as a tool error, as an OpenAI failure does upstream" not in tau3
    assert "`harness_errors` lists harness failures that are never the agent's outcome" in tau2
    assert "a retrieval config other than `alltools`" not in tau2
    from gbench.runners.eval_suites import tau3 as T3
    src = inspect.getsource(T3)
    assert "GBENCH_TAU2_RETRIEVAL_CONFIG" not in src and "no knob selects another config" in src
