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

"""tau2 / tau3: credential and cache handling of a run.

* run_tau_env's returned result is redacted.
* Every run works in one cache directory; the former OpenAI gateway knob changes nothing and
  is not passed into the image.
* A user simulator or judge that is not a gemini/ model is a hard error naming its provider;
  inside the image LiteLLM's own routing is checked too.

No Docker, no network, no model server.
"""

import contextlib
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from gbench.runners.eval_suites import tau_common as TC

REPO = Path(__file__).resolve().parents[1]

GEMINI_KEY = "gemini-secret-9999zzzz"
ENDPOINT_KEY = "endpoint-secret-8888yyyy"


@pytest.fixture
def tau_env(monkeypatch, tmp_path):
    monkeypatch.setattr(TC, "_SECRETS", [])
    monkeypatch.setattr(TC, "_AGENT_ROUTE", {})
    for name in ("GBENCH_TAU2_USER_LLM", "TAU2_USER_LLM", "GBENCH_TAU2_EVAL_LLM", "TAU2_EVAL_LLM",
                 "GBENCH_TAU2_RETRIEVAL_CONFIG", "TAU2_RETRIEVAL_CONFIG", "OPENAI_API_KEY",
                 "GBENCH_TAU2_OPENAI_API_KEY", "GBENCH_TAU2_OPENAI_BASE_URL", TC._AGENT_KEY_ENV,
                 "GBENCH_JUDGE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path


def test_the_result_run_tau_env_returns_is_redacted(monkeypatch, tau_env):
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    monkeypatch.setenv(TC._AGENT_KEY_ENV, ENDPOINT_KEY)
    seen = {}

    def inner(*args):
        seen["secrets"] = list(TC._SECRETS)
        return {"accuracy": 1.0, "tau2_report": {
            "tau2_version": f"1.0.1 {GEMINI_KEY}",
            "domains": {"airline": {"log": [f"401 for {ENDPOINT_KEY}", {"k": GEMINI_KEY}]}}}}
    monkeypatch.setattr(TC, "_run_tau_env", inner)
    out = TC.run_tau_env("tau2", ["airline"], "m", "http://x/v1", 1, None, "docs/evals/tau2.md")
    text = json.dumps(out)
    assert GEMINI_KEY not in text and ENDPOINT_KEY not in text
    assert out["tau2_report"]["tau2_version"] == "1.0.1 [REDACTED]"
    assert out["tau2_report"]["domains"]["airline"]["log"] == ["401 for [REDACTED]",
                                                               {"k": "[REDACTED]"}]
    assert GEMINI_KEY in seen["secrets"] and ENDPOINT_KEY in seen["secrets"]
    assert TC._SECRETS == [] and TC._AGENT_ROUTE == {}


def test_every_run_works_in_one_cache(tau_env, monkeypatch):
    root = str(tau_env / "cache")
    assert TC._tau_cache_dir() == os.path.join(root, "tau2")
    monkeypatch.setenv("GBENCH_TAU2_OPENAI_BASE_URL", "http://gateway.example/v1")
    assert TC._tau_cache_dir() == os.path.join(root, "tau2"), "the former gateway knob is inert"


def _launch_capture(monkeypatch):
    captured = {}
    monkeypatch.setattr(TC, "check_tau_image", lambda: (True, ""))
    # the network checks before the container are docker calls of their own
    # (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC.container_url(url))
    monkeypatch.setattr(TC.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, b"", b""))

    @contextlib.contextmanager
    def bridge(eval_name, image, base_url, agent_base_url, domains, docs_url):
        # the Gemini gate and the relays on the bridge gateway
        # (tests/test_tau_gemini_cascade_and_relays.py)
        yield TC._TauBridge(agent_base_url or base_url)
    monkeypatch.setattr(TC, "_bridge_services", bridge)

    def launch(cmd, env=None):
        captured["cmd"] = cmd
        io_dir = next(cmd[i + 1].split(":")[0] for i, a in enumerate(cmd)
                      if a == "-v" and cmd[i + 1].split(":")[1] == "/gbench_io")
        with open(os.path.join(io_dir, "result.json"), "w") as fh:
            json.dump({"ok": True, "result": {"tau2_report": {}}}, fh)
        return 0, ""
    monkeypatch.setattr(TC, "_launch_image", launch)
    return captured


def test_the_container_works_in_the_cache(tau_env, monkeypatch):
    monkeypatch.setenv("GBENCH_RESULTS_DIR", str(tau_env / "results"))
    monkeypatch.setenv("GBENCH_TAU2_OPENAI_BASE_URL", "http://gateway.example/v1")
    captured = _launch_capture(monkeypatch)
    TC.run_tau_suite("tau3", ["banking_knowledge"], "m", "http://x/v1", 1, None,
                     "docs/evals/tau3.md")
    cmd = captured["cmd"]
    cache = TC._tau_cache_dir()
    assert f"{cache}:/gbench_cache:rw" in cmd and os.path.isdir(cache)
    assert cache == str(tau_env / "cache" / "tau2")
    assert "GBENCH_TAU2_OPENAI_BASE_URL" not in cmd, "the former gateway knob is not passed in"


@pytest.mark.parametrize("model, provider", [
    ("vertex_ai/gemini-3-pro", "vertex_ai"), ("bedrock/anthropic.claude-x", "bedrock"),
    ("azure/gpt-4", "azure"), ("deepseek/chat", "deepseek"), ("gemini-2.5-flash", "vertex_ai"),
    ("openai/gpt-4.1-2025-04-14", "openai"), ("gpt-5.2", "openai"),
    ("anthropic/claude-x", "anthropic"), ("claude-sonnet-4-5", "anthropic"),
    ("openrouter/qwen/qwen3", "openrouter"),
])
def test_a_role_provider_gbench_cannot_pass_keys_for_is_a_hard_error(tau_env, monkeypatch,
                                                                    model, provider):
    monkeypatch.setenv("GBENCH_TAU2_USER_LLM", model)
    ok, why = TC.check_role_providers()
    assert ok is False and f"the user simulator {model}" in why and f"'{provider}'" in why
    assert "GBENCH_TAU2_USER_LLM" in why
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret-1234")
    names = TC._forwarded_env(domains=["airline"])
    assert "DEEPSEEK_API_KEY" not in names and not [n for n in names if n.startswith("VERTEX")]
    # the run stops on the host, before any container
    monkeypatch.setattr(TC, "check_tau_image", lambda: pytest.fail("image checked"))
    with pytest.raises(RuntimeError, match=re.escape(f"'{provider}'")):
        TC.run_tau_suite("tau2", ["airline"], "m", "http://x/v1", 1, None, "docs/evals/tau2.md")
    # and inside the image the credential check refuses it too
    monkeypatch.setattr(TC, "_litellm_provider", lambda m: None)
    ok, why = TC.check_role_credentials()
    assert ok is False and f"'{provider}'" in why


@pytest.mark.parametrize("model", ["gemini/gemini-3.6-flash", "gemini/gemini-3.6-pro"])
def test_supported_role_providers_pass(tau_env, monkeypatch, model):
    monkeypatch.setenv("GBENCH_TAU2_USER_LLM", model)
    monkeypatch.setenv("GBENCH_TAU2_EVAL_LLM", model)
    assert TC.check_role_providers() == (True, "")


def test_inside_the_image_litellm_routing_is_checked(tau_env, monkeypatch):
    fake = types.ModuleType("litellm")

    def get_llm_provider(model):
        routes = {"gemini/gemini-3.6-flash": "gemini", "gemini/gemini-3.6-pro": "gemini",
                  "gemini/odd-alias": "vertex_ai"}
        if model not in routes:
            raise ValueError("LLM Provider NOT provided")
        return model, routes[model], None, None
    fake.get_llm_provider = get_llm_provider
    monkeypatch.setitem(sys.modules, "litellm", fake)
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    monkeypatch.setenv("GBENCH_TAU2_EVAL_LLM", "gemini/gemini-3.6-flash")
    # outside the image the loaded LiteLLM is not consulted
    monkeypatch.setattr(TC, "_in_image", lambda: False)
    monkeypatch.setenv("GBENCH_TAU2_USER_LLM", "gemini/unknown-x")
    assert TC.check_role_credentials() == (True, "")
    monkeypatch.setattr(TC, "_in_image", lambda: True)
    ok, why = TC.check_role_credentials()
    assert ok is False and "LiteLLM cannot route the user simulator gemini/unknown-x" in why
    monkeypatch.setenv("GBENCH_TAU2_USER_LLM", "gemini/odd-alias")
    ok, why = TC.check_role_credentials()
    assert ok is False and "routes the user simulator gemini/odd-alias to the provider " \
                           "'vertex_ai', not to the Gemini API" in why
    for good in ("gemini/gemini-3.6-pro", "gemini/gemini-3.6-flash"):
        monkeypatch.setenv("GBENCH_TAU2_USER_LLM", good)
        assert TC.check_role_credentials() == (True, ""), good
    # a non-Gemini role is refused before LiteLLM is asked
    monkeypatch.setenv("GBENCH_TAU2_USER_LLM", "gpt-5.2")
    ok, why = TC.check_role_credentials()
    assert ok is False and "'openai'" in why and "only through the Gemini API" in why


def test_in_image_is_decided_by_the_package_name():
    assert TC._in_image() is False
    assert TC._IMAGE_PACKAGE == "gbench_tau_suites"


def test_a_registry_without_upstreams_alltools_is_a_hard_error(tau_env, monkeypatch):
    # The removed GBENCH_TAU2_RETRIEVAL_CONFIG cannot select another config: without upstream's
    # alltools (and its dense embedding pipeline) in tau2's registry, the prerequisite fails.
    spec = types.SimpleNamespace(type="embedding", embedder_type="voyage",
                                 embedder_model="voyage-3", reranker=False)
    variant = types.SimpleNamespace(kb_search=None, kb_search_bm25=None, kb_search_dense=spec,
                                    shell=None)
    retrieval = types.ModuleType("tau2.domains.banking_knowledge.retrieval")
    retrieval.RETRIEVAL_VARIANTS = {"voyage_cfg": variant}
    retrieval.RETRIEVAL_VARIANT_ALIASES = {}
    pkg = types.ModuleType("tau2.domains.banking_knowledge")
    pkg.retrieval = retrieval
    monkeypatch.setitem(sys.modules, "tau2.domains.banking_knowledge", pkg)
    monkeypatch.setitem(sys.modules, "tau2.domains.banking_knowledge.retrieval", retrieval)
    monkeypatch.setenv("GBENCH_TAU2_RETRIEVAL_CONFIG", "voyage_cfg")
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    ok, why = TC.check_retrieval_credentials()
    assert ok is False and "has no alltools config" in why and "docs/evals/tau3.md" in why
    no_dense = types.SimpleNamespace(kb_search=None, kb_search_bm25=spec, kb_search_dense=None)
    retrieval.RETRIEVAL_VARIANTS = {"alltools": no_dense}
    ok, why = TC.check_retrieval_credentials()
    assert ok is False and "no dense embedding pipeline" in why


def test_the_tau_docs_match_the_credential_and_cache_code():
    tau2 = (REPO / "docs" / "evals" / "tau2.md").read_text(encoding="utf-8")
    tau3 = (REPO / "docs" / "evals" / "tau3.md").read_text(encoding="utf-8")
    assert "<PROVIDER>_*" not in tau2 and "does *not* set a global" not in tau2
    assert "a global `OPENAI_API_BASE` never redirects the agent" in tau2
    for doc in (tau2, tau3):
        assert "tau2-openai-gateway" not in doc and "<gbench cache>/tau2" in doc
    flat2, flat3 = " ".join(tau2.split()), " ".join(tau3.split())
    assert "hard-errors (`infra_required`) naming the provider" in flat2
    assert "A user simulator or judge of any other provider hard-errors naming it" in flat3
