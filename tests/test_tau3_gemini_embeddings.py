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

"""tau3: banking_knowledge retrieval runs upstream's alltools with its dense search on Gemini
embeddings (gemini-embedding-001 through the Gemini API), and the Gemini API is the only external
service a tau run calls.

  * embedder   - GeminiEmbedder's batchEmbedContents requests (URL, header, body, batching,
                 order), key selection, retries, failures (redacted, counted, never a fallback).
  * variant    - alltools-gemini is a copy of upstream's alltools with only the dense embedder
                 changed; upstream's entry is untouched; no config is operator-selectable.
  * setup      - _setup_banking_retrieval registers the variant, embeds the documents once and
                 records the embedder; a failure stops the run; the removed retrieval-config
                 knob changes nothing.
  * checks     - a missing Gemini key is a failed prerequisite (infra_required at run time),
                 whatever the removed retrieval-config knob says.
  * result     - the substitution is recorded and named in leaderboard_comparable_reason, so the
                 run is not leaderboard_comparable; the key reaches no result field.
  * credentials- only the Gemini key enters the image; the removed OpenAI knobs are inert.
  * docs       - tau3.md / tau2.md state the substitution as coded.

No Docker, no network (urllib.request.urlopen is stubbed), no model server.
"""

import dataclasses
import io
import json
import os
import sys
import types
import urllib.error
import urllib.request
from typing import Optional
from unittest import mock

import pytest

from gbench.runners.eval_suites import tau_common as TC

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEMINI_KEY = "gemini-test-key-0123456789abcdef"
SECOND_KEY = "gemini-second-key-9876543210"
ENDPOINT_KEY = "endpoint-secret-1111aaaa"
REASON = "retrieval embeddings by gemini-embedding-001 (upstream: text-embedding-3-large)"
EMBED_URL = ("https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001"
             ":batchEmbedContents")


def _read(*parts):
    with open(os.path.join(_REPO, *parts), encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture
def clean(monkeypatch):
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "GBENCH_TAU2_RETRIEVAL_CONFIG",
                 "TAU2_RETRIEVAL_CONFIG", "GBENCH_TAU2_USER_LLM", "TAU2_USER_LLM",
                 "GBENCH_TAU2_EVAL_LLM", "TAU2_EVAL_LLM", "GBENCH_TAU2_OPENAI_API_KEY",
                 "GBENCH_TAU2_OPENAI_BASE_URL", "OPENAI_API_KEY", TC._AGENT_KEY_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(TC, "_SECRETS", [])
    monkeypatch.setattr(TC, "_AGENT_ROUTE", {})
    monkeypatch.setattr(TC, "_LAST_RETRIEVAL", {})
    monkeypatch.setattr(TC, "_LLM_FAILURES", {})
    monkeypatch.setattr(TC.time, "sleep", lambda s: None)


# --------------------------------------------------------------------------- #
# a stand-in for the Gemini API (urllib.request.urlopen)
# --------------------------------------------------------------------------- #
class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _GeminiAPI:
    """Answers batchEmbedContents like the Gemini API: one embedding per request, in order. A
    queue of failures (HTTP codes or exceptions) is served first."""

    def __init__(self, dim=4, failures=(), body=None):
        self.dim, self.failures, self.body = dim, list(failures), body
        self.calls = []

    def __call__(self, request, timeout=None):
        assert isinstance(request, urllib.request.Request)
        assert request.full_url.startswith("https://generativelanguage.googleapis.com/"), \
            request.full_url
        payload = json.loads(request.data.decode("utf-8"))
        self.calls.append({"url": request.full_url, "method": request.get_method(),
                           "headers": {k.lower(): v for k, v in request.header_items()},
                           "payload": payload, "timeout": timeout})
        if self.failures:
            failure = self.failures.pop(0)
            if isinstance(failure, BaseException):
                raise failure
            code, text = failure
            raise urllib.error.HTTPError(request.full_url, code, "error", {},
                                         io.BytesIO(text.encode("utf-8")))
        if self.body is not None:
            return _Resp(json.dumps(self.body).encode("utf-8"))
        out = []
        for item in payload["requests"]:
            text = item["content"]["parts"][0]["text"]
            out.append({"values": [float(len(text))] + [0.5] * (self.dim - 1)})
        return _Resp(json.dumps({"embeddings": out}).encode("utf-8"))


@pytest.fixture
def api(monkeypatch):
    fake = _GeminiAPI()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


# --------------------------------------------------------------------------- #
# the embedder
# --------------------------------------------------------------------------- #
def test_embedder_sends_batch_embed_contents_with_the_key_in_a_header(clean, api, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    texts = ["t" * (i + 1) for i in range(250)]
    vectors = TC.GeminiEmbedder().embed(texts)
    assert vectors.shape == (250, 4)
    assert [v[0] for v in vectors] == [float(i + 1) for i in range(250)], "input order is kept"
    assert [len(c["payload"]["requests"]) for c in api.calls] == [100, 100, 50]
    for call in api.calls:
        assert call["url"] == EMBED_URL and call["method"] == "POST"
        assert call["headers"]["x-goog-api-key"] == GEMINI_KEY
        assert GEMINI_KEY not in call["url"]
        for item in call["payload"]["requests"]:
            # texts and model only, like tau2's OpenAIEmbedder: no task type, no dimensionality
            assert set(item) == {"model", "content"}
            assert item["model"] == "models/gemini-embedding-001"
            assert list(item["content"]) == ["parts"] and len(item["content"]["parts"]) == 1
    assert TC.GeminiEmbedder().get_name() == "gemini_gemini-embedding-001"


def test_embedder_rejects_no_text(clean, api):
    with pytest.raises(ValueError):
        TC.GeminiEmbedder().embed([])
    assert api.calls == []


def test_the_first_key_of_gemini_api_keys_wins(clean, api, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "single-key-000000000")
    monkeypatch.setenv("GEMINI_API_KEYS", f"{GEMINI_KEY}; {SECOND_KEY}")
    TC.GeminiEmbedder().embed(["q"])
    assert api.calls[0]["headers"]["x-goog-api-key"] == GEMINI_KEY
    assert TC._gemini_key({"GEMINI_API_KEY": f"{SECOND_KEY},{GEMINI_KEY}"}) == SECOND_KEY
    assert TC._gemini_key({}) == ""


def test_transient_errors_are_retried(clean, api, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    api.failures = [(429, "RESOURCE_EXHAUSTED"), (503, "UNAVAILABLE"),
                    urllib.error.URLError("connection reset")]
    assert TC.GeminiEmbedder().embed(["q"]).shape == (1, 4)
    assert len(api.calls) == 4
    assert TC.llm_failures() == {}, "a request that finally succeeds is no failure"


def test_a_persistent_failure_raises_redacted_and_counted(clean, api, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    api.failures = [(500, f"internal error for key {GEMINI_KEY}")] * 4
    with pytest.raises(RuntimeError) as ei:
        TC.GeminiEmbedder().embed(["q"])
    assert len(api.calls) == TC._GEMINI_EMBED_ATTEMPTS == 4
    msg = str(ei.value)
    assert GEMINI_KEY not in msg and "[REDACTED]" in msg and "HTTP 500" in msg
    assert "attempt 4 of 4" in msg
    assert TC.llm_failures() == {"embedding_error": 1}


def test_a_client_error_is_not_retried(clean, api, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    api.failures = [(400, '{"error": {"message": "API key not valid", "status": '
                          '"INVALID_ARGUMENT"}}')]
    with pytest.raises(RuntimeError, match="HTTP 400") as ei:
        TC.GeminiEmbedder().embed(["q"])
    assert len(api.calls) == 1 and "API key not valid" in str(ei.value)


@pytest.mark.parametrize("body, fragment", [
    ({"embeddings": []}, "expected 1 embeddings, got 0"),
    ({"nothing": 1}, "got none"),
    ({"embeddings": [{"values": []}]}, "without values"),
])
def test_a_malformed_answer_raises(clean, api, monkeypatch, body, fragment):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    api.body = body
    with pytest.raises(RuntimeError, match=fragment):
        TC.GeminiEmbedder().embed(["q"])


def test_without_a_key_nothing_is_sent(clean, api):
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        TC.GeminiEmbedder().embed(["q"])
    assert api.calls == []


# --------------------------------------------------------------------------- #
# a stand-in for the pinned tau2 retrieval surface (upstream's attribute names)
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class PipelineSpec:
    type: str
    embedder_type: Optional[str] = None
    embedder_model: Optional[str] = None
    top_k: int = 10
    reranker: bool = False
    reranker_min_score: int = 5


@dataclasses.dataclass
class ShellSpec:
    allow_writes: bool = False
    file_format: str = "md"


@dataclasses.dataclass
class GrepSpec:
    top_k: int = 10
    case_sensitive: bool = False


@dataclasses.dataclass
class RetrievalVariant:
    name: str
    prompt_template: str
    build_prompt: object = None
    kb_search: Optional[PipelineSpec] = None
    kb_search_bm25: Optional[PipelineSpec] = None
    kb_search_dense: Optional[PipelineSpec] = None
    grep: Optional[GrepSpec] = None
    shell: Optional[ShellSpec] = None
    supports_top_k: bool = False


def _alltools(name, embedder_type, model):
    return RetrievalVariant(name=name, prompt_template="prompts/all_tools.md",
                            build_prompt="standard_prompt",
                            kb_search_bm25=PipelineSpec(type="bm25"),
                            kb_search_dense=PipelineSpec(type="embedding",
                                                         embedder_type=embedder_type,
                                                         embedder_model=model),
                            shell=ShellSpec(allow_writes=False))


def _dense_line(variant):
    """retrieval.format_all_tools_dense_instructions at the pinned revision."""
    kind, model = variant.kb_search_dense.embedder_type, variant.kb_search_dense.embedder_model
    provider = {"openai": "OpenAI API", "openrouter": "OpenRouter"}.get(kind, kind)
    return (f"The `KB_search_dense` tool uses **{provider}** with embedding model `{model}` for "
            "dense retrieval.")


class _FakeTau2:
    """The tau2 modules _setup_banking_retrieval touches, with an EmbeddingIndexer that builds its
    embedder from EMBEDDER_REGISTRY the way tau2's does, and the pinned registry's config names."""

    def __init__(self, n_docs=698):
        self.docs = [{"id": f"doc_{i}", "text": f"document {i}", "title": f"T{i}"}
                     for i in range(n_docs)]
        self.indexed = []

    def modules(self):
        me = self
        names = ("tau2", "tau2.domains", "tau2.domains.banking_knowledge",
                 "tau2.domains.banking_knowledge.retrieval",
                 "tau2.domains.banking_knowledge.environment", "tau2.knowledge",
                 "tau2.knowledge.document_preprocessors",
                 "tau2.knowledge.document_preprocessors.embedding_indexer",
                 "tau2.knowledge.input_preprocessors",
                 "tau2.knowledge.input_preprocessors.embedding_encoder")
        m = {n: types.ModuleType(n) for n in names}
        for n in names:
            parent, _, child = n.rpartition(".")
            if parent in m:
                setattr(m[parent], child, m[n])
        r = m["tau2.domains.banking_knowledge.retrieval"]
        bm25 = RetrievalVariant(name="bm25", prompt_template="p",
                                kb_search=PipelineSpec(type="bm25"), supports_top_k=True)
        r.RETRIEVAL_VARIANTS = {
            "no_knowledge": RetrievalVariant(name="no_knowledge", prompt_template="p"),
            "full_kb": RetrievalVariant(name="full_kb", prompt_template="p"),
            "golden_retrieval": RetrievalVariant(name="golden_retrieval", prompt_template="p"),
            "bm25": bm25,
            "bm25_grep": dataclasses.replace(bm25, name="bm25_grep", grep=GrepSpec()),
            "bm25_reranker": RetrievalVariant(name="bm25_reranker", prompt_template="p",
                                              kb_search=PipelineSpec(type="bm25", reranker=True)),
            "openai_embeddings": RetrievalVariant(
                name="openai_embeddings", prompt_template="p",
                kb_search=PipelineSpec(type="embedding", embedder_type="openai",
                                       embedder_model="text-embedding-3-large")),
            "grep_only": RetrievalVariant(name="grep_only", prompt_template="p", grep=GrepSpec()),
            "terminal_use": RetrievalVariant(name="terminal_use", prompt_template="p",
                                             shell=ShellSpec()),
            "terminal_use_write": RetrievalVariant(name="terminal_use_write", prompt_template="p",
                                                   shell=ShellSpec(allow_writes=True)),
            "alltools": _alltools("alltools", "openai", "text-embedding-3-large"),
            "alltools-qwen": _alltools("alltools-qwen", "openrouter", "qwen3-embedding-8b"),
        }
        r.RETRIEVAL_VARIANT_ALIASES = {"AllTools": "alltools"}
        r.format_all_tools_dense_instructions = _dense_line
        r.get_or_create_docs = lambda kb: list(kb)
        m["tau2.domains.banking_knowledge.environment"].get_knowledge_base = lambda: me.docs
        idx = m["tau2.knowledge.document_preprocessors.embedding_indexer"]
        enc = m["tau2.knowledge.input_preprocessors.embedding_encoder"]
        idx.EMBEDDER_REGISTRY = {"openai": object, "openrouter": object}
        enc.EMBEDDER_REGISTRY = {"openai": object, "openrouter": object}

        class EmbeddingIndexer:
            def __init__(self, embedder_type="openai", embedder_params=None, **kw):
                self.embedder_type, self.params = embedder_type, embedder_params or {}

            def process(self, documents, state):
                embedder = idx.EMBEDDER_REGISTRY[self.embedder_type](**self.params)
                state["doc_embeddings"] = embedder.embed([d["text"] for d in documents])
                me.indexed.append((self.embedder_type, dict(self.params), len(documents)))
                return documents

        idx.EmbeddingIndexer = EmbeddingIndexer
        return m


def test_the_variant_is_upstreams_alltools_with_only_the_embedder_changed(clean):
    fake = _FakeTau2()
    mods = fake.modules()
    with mock.patch.dict(sys.modules, mods):
        variant = TC._register_gemini_retrieval()
    assert not hasattr(TC, "_retrieval_variant"), "no config is operator-selectable"
    registry = mods["tau2.domains.banking_knowledge.retrieval"].RETRIEVAL_VARIANTS
    upstream = registry["alltools"]
    assert registry["alltools-gemini"] is variant
    assert variant.kb_search_dense == dataclasses.replace(
        upstream.kb_search_dense, embedder_type="gemini", embedder_model="gemini-embedding-001")
    assert dataclasses.replace(variant, name="alltools", kb_search_dense=upstream.kb_search_dense
                               ) == upstream, "every other field is upstream's"
    assert upstream == _alltools("alltools", "openai", "text-embedding-3-large"), "untouched"
    for n in ("tau2.knowledge.document_preprocessors.embedding_indexer",
              "tau2.knowledge.input_preprocessors.embedding_encoder"):
        assert mods[n].EMBEDDER_REGISTRY["gemini"] is TC.GeminiEmbedder
        assert mods[n].EMBEDDER_REGISTRY["openai"] is object, "upstream's embedders untouched"


def test_setup_embeds_the_documents_once_and_records_the_embedder(clean, api, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    fake = _FakeTau2()
    before = dict(os.environ)
    with mock.patch.dict(sys.modules, fake.modules()):
        name, kwargs = TC._setup_banking_retrieval()
    assert (name, kwargs) == ("alltools-gemini", None)
    assert fake.indexed == [("gemini", {"model": "gemini-embedding-001"}, 698)]
    assert [len(c["payload"]["requests"]) for c in api.calls] == [100] * 6 + [98]
    assert all(c["headers"]["x-goog-api-key"] == GEMINI_KEY for c in api.calls)
    assert dict(os.environ) == before, "no environment is rewritten"
    rec = TC._LAST_RETRIEVAL
    assert rec["retrieval_config"] == "alltools" and rec["retrieval_variant"] == "alltools-gemini"
    assert rec["embedders"] == ["gemini:gemini-embedding-001"]
    assert rec["embed_model"] == "gemini:gemini-embedding-001"
    assert rec["embedder"] == {
        "embedder": "gemini:gemini-embedding-001",
        "api": f"Gemini API {EMBED_URL}",
        "through": "the Gemini API directly",
        "upstream_embedder": "openai:text-embedding-3-large",
        "batch_size": 100, "documents": 698,
        "dense_tool_description": "The `KB_search_dense` tool uses **gemini** with embedding "
                                  "model `gemini-embedding-001` for dense retrieval."}


def test_a_failed_document_embedding_stops_the_run(clean, api, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    monkeypatch.setattr(TC, "_SECRETS", [GEMINI_KEY])
    api.failures = [(403, f"PERMISSION_DENIED for {GEMINI_KEY}")]
    with mock.patch.dict(sys.modules, _FakeTau2().modules()):
        with pytest.raises(RuntimeError) as ei:
            TC._setup_banking_retrieval()
    msg = str(ei.value)
    assert msg.startswith("embedding the banking_knowledge documents with gemini-embedding-001 "
                          "through the Gemini API failed")
    assert GEMINI_KEY not in msg and "HTTP 403" in msg
    assert TC._LAST_RETRIEVAL == {}, "nothing recorded, no other embedder or config tried"


def test_the_removed_retrieval_knob_changes_nothing_in_setup(clean, api, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    monkeypatch.setenv("GBENCH_TAU2_RETRIEVAL_CONFIG", "bm25")
    monkeypatch.setenv("TAU2_RETRIEVAL_CONFIG", "terminal_use")
    with mock.patch.dict(sys.modules, _FakeTau2().modules()):
        assert TC._setup_banking_retrieval() == ("alltools-gemini", None)
    assert len(api.calls) == 7, "the documents are embedded with the Gemini embedder"
    assert TC._LAST_RETRIEVAL["retrieval_config"] == "alltools"
    assert TC._LAST_RETRIEVAL["embedders"] == ["gemini:gemini-embedding-001"]


# --------------------------------------------------------------------------- #
# prerequisites
# --------------------------------------------------------------------------- #
# The removed GBENCH_TAU2_RETRIEVAL_CONFIG is set in some cases: it changes nothing, the one
# configuration (alltools with Gemini embeddings) always needs the Gemini key.
@pytest.mark.parametrize("config, env, ok, fragment", [
    (None, {}, False, "needs GEMINI_API_KEY (or GEMINI_API_KEYS)"),
    ("AllTools", {"OPENAI_API_KEY": "sk-openai-0000000000"}, False, "GEMINI_API_KEY"),
    (None, {"GEMINI_API_KEY": GEMINI_KEY}, True, None),
    (None, {"GEMINI_API_KEYS": GEMINI_KEY}, True, None),
    ("bm25", {}, False, "needs GEMINI_API_KEY (or GEMINI_API_KEYS)"),
    ("terminal_use_write", {}, False, "needs GEMINI_API_KEY (or GEMINI_API_KEYS)"),
    ("alltools-qwen", {"GEMINI_API_KEY": GEMINI_KEY}, True, None),
    ("openai_embeddings", {"GEMINI_API_KEY": GEMINI_KEY}, True, None),
    ("bm25_reranker", {"GEMINI_API_KEY": GEMINI_KEY}, True, None),
    ("alltools-gemini", {"GEMINI_API_KEY": GEMINI_KEY}, True, None),
])
def test_retrieval_prerequisites(clean, monkeypatch, config, env, ok, fragment):
    if config:
        monkeypatch.setenv("GBENCH_TAU2_RETRIEVAL_CONFIG", config)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with mock.patch.dict(sys.modules, _FakeTau2().modules()):
        got_ok, why = TC.check_retrieval_credentials()
    assert got_ok is ok
    if fragment:
        assert fragment in why and "docs/evals/tau3.md" in why
        assert "gemini-embedding-001" in why and "text-embedding-3-large" in why
    else:
        assert why == ""


def test_the_missing_key_message_names_every_gemini_use(clean, monkeypatch):
    monkeypatch.setattr(TC, "_import_tau2", lambda: (True, ""))
    ok, why = TC.check_tau_env_prerequisites()
    assert not ok and "GEMINI_API_KEY (or GEMINI_API_KEYS)" in why
    assert "retrieval embeddings" in why and "Gemini API" in why
    assert "another" not in why, "no other provider is offered"


def test_a_missing_key_is_infra_required_before_tau2_runs(clean, monkeypatch):
    monkeypatch.setattr(TC, "check_tau_env_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(TC, "_prune_stale_search_sandboxes", lambda: None)
    monkeypatch.setattr(TC, "check_shell_sandbox", lambda: (True, ""))
    monkeypatch.setitem(sys.modules, "rank_bm25", types.ModuleType("rank_bm25"))
    with mock.patch.dict(sys.modules, _FakeTau2().modules()):
        with pytest.raises(RuntimeError) as ei:
            TC.run_tau_env("tau3", ["banking_knowledge"], "m", "http://x/v1", 1, None,
                           "docs/evals/tau3.md")
    assert str(ei.value).startswith("[INFRA REQUIRED]") and "GEMINI_API_KEY" in str(ei.value)


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #
def _run_modules(fake, configs):
    mods = fake.modules()
    names = ("tau2.data_model", "tau2.data_model.simulation", "tau2.run", "tau2.metrics",
             "tau2.metrics.agent_metrics")
    mods.update({n: types.ModuleType(n) for n in names})

    class TextRunConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def run_domain(config):
        configs.append(config)
        sims = [types.SimpleNamespace(reward_info=types.SimpleNamespace(reward=1.0),
                                      termination_reason="agent_stop")]
        return types.SimpleNamespace(simulations=sims, num_trials=1)

    mods["tau2.data_model.simulation"].TextRunConfig = TextRunConfig
    mods["tau2.run"].run_domain = run_domain
    mods["tau2.metrics.agent_metrics"].compute_metrics = (
        lambda results: types.SimpleNamespace(pass_hat_ks={1: 1.0}))
    return mods


def test_the_result_records_the_substitution(clean, api, monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    monkeypatch.setenv(TC._AGENT_KEY_ENV, ENDPOINT_KEY)
    monkeypatch.setenv("GBENCH_TAU2_SAVE_TRACES", str(tmp_path / "traces"))
    for name, value in (("check_tau_env_prerequisites", lambda: (True, "")),
                        ("check_banking_prerequisites", lambda: (True, "")),
                        ("_install_robustness_patches", lambda *a, **k: None),
                        ("_install_credential_guards", lambda *a, **k: None),
                        ("_configure_evaluator_llm", lambda *a, **k: None),
                        ("_quiet_tau2_noise", lambda: None),
                        ("_prune_stale_search_sandboxes", lambda: None),
                        ("_refuse_resume_prompts", lambda: None),
                        ("_tau2_version", lambda: "1.0.1")):
        monkeypatch.setattr(TC, name, value)
    configs = []
    with mock.patch.dict(sys.modules, _run_modules(_FakeTau2(), configs)):
        r = TC.run_tau_env("tau3", ["banking_knowledge"], "m", "http://x/v1", 1, None,
                           "docs/evals/tau3.md")
    assert configs[0].retrieval_config == "alltools-gemini"
    assert r["leaderboard_comparable"] is False
    assert REASON in r["leaderboard_comparable_reason"].split("; ")
    assert "retrieval config" not in r["leaderboard_comparable_reason"]
    rep = r["tau2_report"]
    assert rep["retrieval_config"] == "alltools" and rep["retrieval_variant"] == "alltools-gemini"
    assert rep["embedders"] == ["gemini:gemini-embedding-001"]
    assert rep["retrieval_embedder"]["upstream_embedder"] == "openai:text-embedding-3-large"
    assert rep["run_config"]["retrieval_embedders"] == ["gemini:gemini-embedding-001"]
    assert "openai_base_url" not in rep and "openai_base_url" not in rep["run_config"]
    text = json.dumps(r)
    assert GEMINI_KEY not in text and ENDPOINT_KEY not in text


def test_the_retrieval_gap_is_only_the_embedding_substitution(clean, monkeypatch):
    monkeypatch.setenv("GBENCH_TAU2_USER_LLM", "gemini/gemini-3.6-flash")
    monkeypatch.setenv("GBENCH_TAU2_RETRIEVAL_CONFIG", "bm25")          # removed: no effect
    monkeypatch.setattr(TC, "_LAST_AGENT_TEMPERATURE", {"value": 0.0})
    monkeypatch.setattr(TC, "_LAST_RETRIEVAL", {"retrieval_config": "bm25", "embedders": []})
    gaps = TC._comparability_gaps(["banking_knowledge"], 4, None, 0)
    assert REASON in gaps
    assert not any("retrieval config" in g for g in gaps)
    # the embeddings substitution is a tau3 difference only
    monkeypatch.setattr(TC, "_LAST_RETRIEVAL", {"retrieval_config": "alltools",
                                                "embedders": ["gemini:gemini-embedding-001"]})
    assert REASON not in TC._comparability_gaps(["retail"], 4, None, 0)


# --------------------------------------------------------------------------- #
# credentials: only the Gemini relay's placeholder enters the image
# --------------------------------------------------------------------------- #
def test_only_the_gemini_key_enters_the_image(clean):
    host = {"OPENAI_API_KEY": ENDPOINT_KEY, "GBENCH_TAU2_OPENAI_API_KEY": "sk-openai-0000000000",
            "GBENCH_TAU2_OPENAI_BASE_URL": "http://gateway/v1", "OPENROUTER_API_KEY": "r" * 12,
            "ANTHROPIC_API_KEY": "a" * 12, "GEMINI_API_KEYS": f"{GEMINI_KEY},{SECOND_KEY}",
            "GEMINI_OPENAI_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "GBENCH_TAU2_NUM_TRIALS": "4", "PATH": "/usr/bin"}
    bridge = {"GEMINI_API_KEY": "gbench-gemini-relay-" + "a" * 48,
              "GBENCH_GEMINI_RELAY_URL": "http://172.17.0.1:4000",
              "GBENCH_JUDGE_MODELS": "gemini-a"}
    names = TC._forwarded_env(host, domains=["banking_knowledge"], bridge_env=bridge)
    assert names == sorted(["GBENCH_TAU_AGENT_KEY", "GEMINI_API_KEY", "GBENCH_GEMINI_RELAY_URL",
                            "GBENCH_JUDGE_MODELS", "GBENCH_TAU2_NUM_TRIALS"])
    env = TC._launch_env(host, bridge_env=bridge)
    assert env["GEMINI_API_KEY"] == bridge["GEMINI_API_KEY"] and env[TC._AGENT_KEY_ENV] == "EMPTY"
    assert "OPENAI_API_KEY" not in env and "GEMINI_OPENAI_BASE_URL" not in env
    assert not any(k in v for v in env.values() for k in (GEMINI_KEY, SECOND_KEY, ENDPOINT_KEY))


def test_the_removed_openai_knobs_warn_and_change_nothing(clean, monkeypatch, caplog):
    monkeypatch.setenv("GBENCH_TAU2_OPENAI_BASE_URL", "http://gateway/v1")
    monkeypatch.setattr(TC, "check_tau_image", lambda: (False, "no image"))
    with caplog.at_level("WARNING"), pytest.raises(RuntimeError):
        TC.run_tau_suite("tau3", ["banking_knowledge"], "m", "http://x/v1", 1, None,
                         "docs/evals/tau3.md")
    assert "GBENCH_TAU2_OPENAI_BASE_URL no longer change anything" in caplog.text
    assert "GBENCH_TAU2_OPENAI_BASE_URL" not in TC._forwarded_env()
    assert TC._run_config("tau3", ["banking_knowledge"], "m", False, None, 4)[
        "retrieval_embedders"] == ["gemini:gemini-embedding-001"]


@pytest.mark.parametrize("model", ["openai/gpt-5.2", "gpt-4.1-2025-04-14", "anthropic/claude-x",
                                   "openrouter/qwen/qwen3"])
def test_a_non_gemini_role_is_refused_on_the_host(clean, monkeypatch, model):
    monkeypatch.setenv("GBENCH_TAU2_EVAL_LLM", model)
    ok, why = TC.check_role_providers()
    assert not ok and "NL-assertion judge" in why and "only through the Gemini API" in why
    assert "gemini/<model>" in why


# --------------------------------------------------------------------------- #
# docs and module text
# --------------------------------------------------------------------------- #
def test_the_docs_state_the_substitution():
    tau3 = " ".join(_read("docs", "evals", "tau3.md").split())
    tau2 = " ".join(_read("docs", "evals", "tau2.md").split())
    assert REASON in tau3
    assert "models/gemini-embedding-001:batchEmbedContents" in tau3
    assert "`x-goog-api-key`" in tau3 and "at most 100 texts per request" in tau3
    assert "The `KB_search_dense` tool uses **gemini** with embedding model " \
           "`gemini-embedding-001` for dense retrieval." in tau3
    assert "`llm_failures.embedding_error`" in tau3
    for knob in ("GBENCH_TAU2_OPENAI_API_KEY", "GBENCH_TAU2_OPENAI_BASE_URL"):
        assert f'export {knob}' not in tau3
    assert "`GBENCH_TAU2_RETRIEVAL_CONFIG` (and its alias `TAU2_RETRIEVAL_CONFIG`) no longer " \
           "changes anything" in tau3
    assert "`bm25_grep`" not in tau3, "no other retrieval config is offered"
    assert "No key enters the container." in tau2 and "Gemini relay" in tau2
    assert "tau3's retrieval embeds with `gemini-embedding-001`" in tau2
    import inspect
    from gbench.runners.eval_suites import tau3 as T3
    src = inspect.getsource(T3)
    assert "GBENCH_TAU2_OPENAI" not in src and "gemini-embedding-001" in src
