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

"""Tier-4 agentic / execution-environment suites: swe_lancer, gaia2, tau2/tau3 env.

- swe_lancer: gated OpenAI SWELancer Docker harness (substring heuristic removed).
- gaia2: canonical Meta-ARE harness delegated to a LOCAL container (are-benchmark); missing
  Docker / orchestrator image / GEMINI judge hard-error via infra_required (no skip, no proxy).
  Deterministic parse/gate helpers are covered in tests/test_gaia2.py.
- tau2/tau3: a plain single-turn /v1 endpoint CANNOT score tau-bench (assertion-only
  tasks + environment-dependent gold args), so the default reports a clean skip instead
  of a misleading 0%; the only scored path is the gated full environment (TAU2_ENV_RUN=1
  + the tau2-bench simulator), which itself skips cleanly when opted-in-but-unavailable.
- gdpval: grades produced file deliverables, so a text endpoint can't be scored -> skips.

All harnesses are absent on CI, so these lock in the gates, skip dicts, loaders (no
fabrication), and the env result-parsing/aggregation that runs fully offline.
"""

import json
import os
from unittest.mock import patch, MagicMock

import pytest

from gbench.runners.eval_suites import SUITES


# --------------------------------------------------------------------------- #
# registration + sandbox wiring
# --------------------------------------------------------------------------- #
def test_tier4_suites_registered():
    # swe_lancer is DEREGISTERED (roadmap-only: upstream Expensify image drift) - see test_missing_evals.
    for name in ("gaia2", "tau2", "tau3"):
        assert name in SUITES, f"{name} not registered in SUITES"


@pytest.mark.parametrize("name", ["tau2", "tau3"])
def test_tier4_execution_suites_are_sandbox_evals(name):
    from gbench.runners.evals import EvalsBenchmarkRunner
    from gbench.core.config import BenchmarkConfig

    config = BenchmarkConfig()
    config.sandboxes = 5
    runner = EvalsBenchmarkRunner(config)
    with patch.dict("gbench.runners.eval_suites.SUITES",
                    {name: MagicMock(return_value={"status": "success"})}) as mock_suites:
        runner._run_single_eval(name, "gemma-4-E4B-it", "http://localhost:8000/v1", num_threads=128)
        assert mock_suites[name].call_args[1]["concurrency"] == 5


# --------------------------------------------------------------------------- #
# swe_lancer: gated Docker harness (no heuristic scoring)
# --------------------------------------------------------------------------- #
def test_swe_lancer_hard_errors_on_missing_prereqs(monkeypatch):
    # No-skip policy: a missing prerequisite (here the harness image) must hard-error, not skip.
    from gbench.runners.eval_suites.swe_lancer import run_swe_lancer
    monkeypatch.setenv("GBENCH_SWE_LANCER_IMAGE", "gbench-no-such-image-for-tests")
    with pytest.raises(RuntimeError, match="docs/evals/swe_lancer.md"):
        run_swe_lancer("gemma-4-E4B-it", "http://localhost:8000/v1", 2)


def test_swe_lancer_has_no_opt_in_gate():
    # Cost is not a prerequisite: with Docker, the socket, the harness image at the pinned
    # commit and iptables access the check passes without any opt-in variable.
    from gbench.runners.eval_suites import swe_lancer as SL
    labels = json.dumps({"org.gbench.frontier_evals": SL.HARNESS_COMMIT,
                         "org.gbench.swelancer_tasks_sha256": SL.TASKS_SHA256})
    with patch("shutil.which", return_value="/usr/bin/docker"), \
         patch.object(SL.subprocess, "run",
                      return_value=MagicMock(returncode=0, stdout=labels, stderr="")), \
         patch.object(SL.os.path, "exists", return_value=True):
        os.environ.pop("SWELANCER_RUN", None)
        os.environ.pop("GBENCH_SWE_LANCER_RUN", None)
        ok, reason = SL.check_swe_lancer_prerequisites()
        assert ok is True and reason == ""


def test_swe_lancer_ships_the_harness_entry_and_image():
    # gbench runs SWE-Lancer's own run_swelancer.py main through swe_lancer_eval.py, which the
    # harness image copies in; both files ship with gbench.
    import py_compile
    here = os.path.join(os.path.dirname(__file__), "..", "gbench", "docker")
    py_compile.compile(os.path.join(here, "swe_lancer_eval.py"), doraise=True)
    dockerfile = open(os.path.join(here, "swe_lancer_harness.Dockerfile"), encoding="utf-8").read()
    assert "COPY swe_lancer_eval.py /opt/gbench/swe_lancer_eval.py" in dockerfile
    assert "COPY swe_lancer_entrypoint.sh" in dockerfile


def test_swe_lancer_loader_reads_the_pinned_task_table(monkeypatch, tmp_path):
    from gbench.runners.eval_suites import swe_lancer as SL
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path))
    table = {"sha256": SL.TASKS_SHA256, "tasks": [
        {"question_id": "1_1", "variant": "ic_swe", "set": "diamond", "price": 500.0},
        {"question_id": "2-manager-0", "variant": "swe_manager", "set": "diamond", "price": 1000.0}]}
    calls = []

    def fake_run(cmd, **k):
        calls.append(cmd)
        return MagicMock(returncode=0, stdout=json.dumps(table) + "\n", stderr="")
    monkeypatch.setattr(SL.subprocess, "run", fake_run)
    assert SL.load_tasks() == table["tasks"]
    assert SL.load_tasks() == table["tasks"] and len(calls) == 1      # cached
    assert "--network" in calls[0] and "none" in calls[0]
    assert SL.select_tasks(table["tasks"], "ic_swe", None) == (["1_1"], None)
    table["sha256"] = "0" * 64
    (tmp_path / "swe_lancer" / "issues_manifest.json").unlink()
    with pytest.raises(RuntimeError, match="sha256"):
        SL.load_tasks()


def test_swe_lancer_reads_the_harness_report(monkeypatch, tmp_path):
    from gbench.runners.eval_suites import swe_lancer as SL
    rows = [{"question_id": "a", "variant": "ic_swe", "price": 500.0, "correct": True,
             "earned": 500.0, "available": 500.0, "system_error": None},
            {"question_id": "b", "variant": "ic_swe", "price": 250.0, "correct": False,
             "earned": 0.0, "available": 250.0, "system_error": None}]

    def fake_run(cmd, **k):
        if cmd[:2] == ["docker", "run"]:
            report = [a.split("=", 1)[1] for a in cmd if a.startswith("swelancer.report_path=")][0]
            open(report, "w").write(json.dumps({"summary": {"total_earned": 500.0}, "tasks": rows}))
        return MagicMock(returncode=0)
    monkeypatch.setattr(SL.subprocess, "run", fake_run)
    res = SL._run_variant("ic_swe", ["a", "b"], True, "m", "http://127.0.0.1:1/v1", 1, str(tmp_path))
    assert (res["correct_answers"], res["total_questions"], res["accuracy"]) == (1, 2, 50.0)
    assert (res["earned_usd"], res["available_usd"], res["status"]) == (500.0, 750.0, "success")


# --------------------------------------------------------------------------- #
# gaia2: canonical Meta-ARE harness (no-skip; hard-errors when un-provisioned)
# --------------------------------------------------------------------------- #
def test_gaia2_hard_errors_on_missing_prereqs():
    # No-skip policy: missing Docker / orchestrator image / GEMINI judge must hard-error, not skip.
    from gbench.runners.eval_suites.gaia2 import run_gaia2
    with pytest.raises(RuntimeError, match="docs/evals/gaia2.md"):
        run_gaia2("gemma-4-E4B-it", "http://localhost:8000/v1", 2)


# --------------------------------------------------------------------------- #
# tau2 / tau3: proxy default + gated full environment
# --------------------------------------------------------------------------- #
def test_tau_needs_no_env_opt_in():
    """tau2 runs the simulator (in the local image) without any opt-in variable."""
    from gbench.runners.eval_suites import tau2 as T2
    with patch.dict(os.environ, {}, clear=False), \
         patch.object(T2, "run_tau_suite", return_value={"status": "success"}) as m:
        os.environ.pop("TAU2_ENV_RUN", None)
        os.environ.pop("GBENCH_TAU2_ENV_RUN", None)
        assert T2.run_tau2("m", "http://x", 2) == {"status": "success"}
    assert m.call_args[0][:2] == ("tau2", ["retail", "airline", "telecom"])


def test_tau2_and_tau3_default_hard_error_not_a_misleading_zero():
    """Without their harness (Docker + the local tau2 image), tau2/tau3 must HARD-ERROR with the
    build command (not score 0, not silently skip), since single-turn can't measure them."""
    from gbench.runners.eval_suites import tau2 as T2, tau3 as T3
    from gbench.runners.eval_suites import tau_common as TC
    with patch.object(TC.shutil, "which", return_value=None):
        for run, name in ((T2.run_tau2, "tau2"), (T3.run_tau3, "tau3")):
            with pytest.raises(RuntimeError, match="docker build -t gbench-tau2:"):
                run("m", "http://x", 2)
            with pytest.raises(RuntimeError, match=f"docs/evals/{name}.md"):
                run("m", "http://x", 2)


def test_gdpval_hard_errors_without_a_judge(monkeypatch):
    """gdpval needs the Gemini judge; per the no-skip policy a missing GEMINI_API_KEY hard-errors
    (the judge gate runs before the file/text-endpoint check). (gdpval's own text-endpoint
    handling is a Group-D item; this only pins the no-judge hard-error.)"""
    import pytest
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)   # the gate reads the key list first
    from gbench.runners.eval_suites.gdpval import run_gdpval
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        run_gdpval("m", "http://x", 2)


def test_tau2_and_tau3_env_opt_in_but_missing_hard_errors():
    # No-skip policy: the tau2 image is missing -> hard-error telling the operator to build it,
    # not a silent skip (the opt-in variable no longer changes anything).
    from gbench.runners.eval_suites import tau2 as T2, tau3 as T3
    from gbench.runners.eval_suites import tau_common as TC
    missing = (False, "the tau2-bench image 'gbench-tau2:x' is not built. Build it ...")
    with patch.dict(os.environ, {"TAU2_ENV_RUN": "1"}, clear=False), \
         patch.object(TC, "check_tau_image", return_value=missing):
        with pytest.raises(RuntimeError, match="docs/evals/tau2.md"):
            T2.run_tau2("m", "http://x", 2)
        with pytest.raises(RuntimeError, match="docs/evals/tau3.md"):
            T3.run_tau3("m", "http://x", 2)


def test_tau_env_prereq_requires_gemini_key():
    """With tau2 importable but no GEMINI_API_KEY, the gemini user/judge defaults -> skip reason."""
    from gbench.runners.eval_suites import tau_common as TC
    old = os.environ.pop("GEMINI_API_KEY", None)
    try:
        with patch.object(TC, "_import_tau2", return_value=(True, "")):
            ok, why = TC.check_tau_env_prerequisites()
        assert ok is False and "GEMINI_API_KEY" in why
    finally:
        if old is not None:
            os.environ["GEMINI_API_KEY"] = old


def test_tau_env_prereq_surfaces_import_error_with_audioop_hint():
    """A findable-but-broken tau2 (e.g. py3.13 audioop) yields the real reason + backport hint."""
    from gbench.runners.eval_suites import tau_common as TC
    with patch.object(TC, "_import_tau2",
                      return_value=(False, "ModuleNotFoundError: No module named 'audioop'")), \
         patch.dict(os.environ, {"GEMINI_API_KEY": "x"}, clear=False):
        ok, why = TC.check_tau_env_prerequisites()
    assert ok is False and "audioop-lts" in why and "audioop" in why


def test_tau_common_run_env_aggregates_reward_across_domains():
    """run_tau_env aggregates per-domain (total, reward_sum, perfect) into the result dict.

    Mocks the tau2 Python-API layer (_run_one_domain) so the gbench-side aggregation +
    result schema are what's under test, not the external simulator.
    """
    from gbench.runners.eval_suites import tau_common as TC
    canned = {"airline": (2, 1.0, 1, 0), "retail": (2, 2.0, 2, 0)}  # (total, reward_sum, perfect, infra)
    pass1 = {"airline": 0.5, "retail": 1.0}     # tau2's compute_metrics pass^1 for those runs

    def one(d, *a, **k):
        TC._DOMAIN_PASS_HAT_K[d] = {"pass^1": pass1[d]}
        return canned[d]

    with patch.object(TC, "check_tau_env_prerequisites", return_value=(True, "")), \
         patch.object(TC, "_install_robustness_patches", lambda *a, **k: None), \
         patch.object(TC, "_configure_evaluator_llm", lambda *a, **k: None), \
         patch.object(TC, "_run_one_domain", side_effect=one):
        r = TC.run_tau_env("tau2", ["airline", "retail"], "m", "http://x", 4, None, "docs/evals/tau2.md")

    assert r["status"] == "success" and r["mode"] == "full_environment"
    # headline = mean of the per-domain pass^1 (50, 100) = 75.0; mean reward 3.0/4 = 0.75; perfect=3
    assert r["total_questions"] == 4 and r["correct_answers"] == 3 and r["accuracy"] == 75.0
    assert r["category_accuracy"]["airline"]["accuracy"] == 50.0    # 1.0/2
    assert r["category_accuracy"]["retail"]["accuracy"] == 100.0    # 2.0/2
    assert r["tau2_report"]["mean_reward"] == 0.75 and r["tau2_report"]["perfect_tasks"] == 3


# --------------------------------------------------------------------------- #
# tau3 == τ³-bench banking_knowledge (RAG) repurpose
# --------------------------------------------------------------------------- #
def test_tau3_default_hard_error_names_banking_knowledge():
    """tau3's default hard-error should describe the τ³ banking_knowledge RAG domain, not telecom."""
    from gbench.runners.eval_suites import tau3 as T3
    from gbench.runners.eval_suites import tau_common as TC
    with patch.object(TC.shutil, "which", return_value=None):
        with pytest.raises(RuntimeError) as ei:
            T3.run_tau3("m", "http://x", 2)
    msg = str(ei.value)
    assert "banking_knowledge" in msg
    assert "telecom" not in msg.lower()


def test_tau3_env_runs_banking_knowledge_domain():
    """With the env opt-in, tau3 must drive the banking_knowledge domain (not telecom)."""
    from gbench.runners.eval_suites import tau3 as T3
    with patch.dict(os.environ, {"TAU2_ENV_RUN": "1"}, clear=False), \
         patch.object(T3, "run_tau_suite", return_value={"status": "success"}) as m:
        T3.run_tau3("m", "http://x", 8, enable_thinking=False, limit=5)
    args, kwargs = m.call_args
    assert args[0] == "tau3" and args[1] == ["banking_knowledge"]


def test_check_banking_prerequisites_requires_rank_bm25():
    """Missing rank-bm25 -> a failed prerequisite (a hard error at run time) naming it, with the
    docs pointer."""
    import sys as _sys
    from gbench.runners.eval_suites import tau_common as TC
    with patch.dict(_sys.modules, {"rank_bm25": None}):   # makes `import rank_bm25` raise
        ok, why = TC.check_banking_prerequisites()
    assert ok is False and "rank-bm25" in why and "docs/evals/tau3.md" in why


def test_check_banking_prerequisites_default_needs_the_gemini_key():
    """rank-bm25 present but no Gemini key and no config override -> hard error naming the key for
    the alltools dense search, which gbench runs with gemini-embedding-001 through the Gemini API;
    an OpenAI key does not stand in. (The shell-sandbox probe that follows is stubbed; it has its
    own tests.)"""
    pytest.importorskip("tau2.domains.banking_knowledge.retrieval")
    from gbench.runners.eval_suites import tau_common as TC
    with patch.dict(os.environ, {}, clear=False), \
         patch.object(TC, "check_shell_sandbox", return_value=(True, "")):
        for k in ("TAU2_RETRIEVAL_CONFIG", "GBENCH_TAU2_RETRIEVAL_CONFIG", "GEMINI_API_KEY",
                  "GEMINI_API_KEYS", "TAU2_EMBED_API_KEY", "OPENAI_API_KEY"):
            os.environ.pop(k, None)
        os.environ["OPENAI_API_KEY"] = "k"
        ok, why = TC.check_banking_prerequisites()
        assert ok is False and "GEMINI_API_KEY" in why
        assert "gemini-embedding-001" in why and "text-embedding-3-large" in why
        os.environ["GEMINI_API_KEY"] = "k"
        ok2, _ = TC.check_banking_prerequisites()
        assert ok2 is True


def test_check_banking_prerequisites_removed_knob_does_not_bypass_key_check():
    """TAU2_RETRIEVAL_CONFIG (removed) no longer changes anything: bm25 there does not lift the
    Gemini key the one retrieval config (alltools with Gemini embeddings) needs."""
    pytest.importorskip("tau2.domains.banking_knowledge.retrieval")
    from gbench.runners.eval_suites import tau_common as TC
    with patch.dict(os.environ, {"TAU2_RETRIEVAL_CONFIG": "bm25"}, clear=False), \
         patch.object(TC, "check_shell_sandbox", return_value=(True, "")):
        for k in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "TAU2_EMBED_API_KEY", "OPENAI_API_KEY"):
            os.environ.pop(k, None)
        ok, why = TC.check_banking_prerequisites()
    assert ok is False and "GEMINI_API_KEY" in why


def test_run_tau_env_gates_on_banking_prereq():
    """run_tau_env must apply the banking prereq gate for the banking_knowledge domain
    as a hard-error (no-skip policy)."""
    from gbench.runners.eval_suites import tau_common as TC
    with patch.object(TC, "check_tau_env_prerequisites", return_value=(True, "")), \
         patch.object(TC, "check_banking_prerequisites",
                      return_value=(False, "needs rank-bm25 ... docs/evals/tau3.md")):
        with pytest.raises(RuntimeError, match="rank-bm25"):
            TC.run_tau_env("tau3", ["banking_knowledge"], "m", "http://x", 4, None, "docs/evals/tau3.md")


def test_setup_banking_retrieval_is_alltools_with_gemini_embeddings_and_ignores_removed_knob():
    """Banking retrieval is upstream's alltools, run as gbench's alltools-gemini: a copy of
    tau2's own alltools variant whose dense pipeline embeds with gemini-embedding-001 (no
    environment is rewritten, upstream's alltools is left as it is); the removed
    TAU2_RETRIEVAL_CONFIG changes nothing. The document embedding call is stubbed: no
    network."""
    pytest.importorskip("tau2.domains.banking_knowledge.retrieval")
    from gbench.runners.eval_suites import tau_common as TC
    from tau2.domains.banking_knowledge import retrieval as R
    upstream_dense = (R.RETRIEVAL_VARIANTS["alltools"].kb_search_dense.embedder_type,
                      R.RETRIEVAL_VARIANTS["alltools"].kb_search_dense.embedder_model)
    with patch.dict(os.environ, {"GEMINI_API_KEY": "k"}, clear=False), \
         patch.object(TC, "_warm_gemini_document_embeddings", return_value=698):
        for k in ("TAU2_RETRIEVAL_CONFIG", "GBENCH_TAU2_RETRIEVAL_CONFIG", "OPENAI_API_KEY",
                  "OPENAI_BASE_URL"):
            os.environ.pop(k, None)
        before = dict(os.environ)
        name, kw = TC._setup_banking_retrieval()
        assert name == "alltools-gemini" and kw is None
        assert TC._LAST_RETRIEVAL["retrieval_config"] == "alltools"
        assert TC._LAST_RETRIEVAL["embedders"] == ["gemini:gemini-embedding-001"]
        assert dict(os.environ) == before
    ours = R.RETRIEVAL_VARIANTS["alltools-gemini"]
    assert (ours.kb_search_dense.embedder_type, ours.kb_search_dense.embedder_model) == (
        "gemini", "gemini-embedding-001")
    assert (R.RETRIEVAL_VARIANTS["alltools"].kb_search_dense.embedder_type,
            R.RETRIEVAL_VARIANTS["alltools"].kb_search_dense.embedder_model) == upstream_dense
    with patch.dict(os.environ, {"TAU2_RETRIEVAL_CONFIG": "bm25"}, clear=False), \
         patch.object(TC, "_warm_gemini_document_embeddings", return_value=698):
        os.environ.pop("GBENCH_TAU2_RETRIEVAL_CONFIG", None)
        name2, _ = TC._setup_banking_retrieval()
        assert name2 == "alltools-gemini"
        assert TC._LAST_RETRIEVAL["embedders"] == ["gemini:gemini-embedding-001"]


def _fake_tau2_modules():
    """Fake tau2 submodules that _quiet_tau2_noise patches (functions + rich console)."""
    import types
    llm = types.ModuleType("tau2.utils.llm_utils")
    llm.get_response_cost = lambda r: 999.0
    utils = types.ModuleType("tau2.utils.utils")
    utils.get_commit_hash = lambda: "realhash"
    helpers = types.ModuleType("tau2.runner.helpers")
    helpers.get_commit_hash = lambda: "realhash"
    display = types.ModuleType("tau2.utils.display")

    class ConsoleDisplay:
        console = "loud-console"  # a placeholder tau2 would set to a rich Console()
    display.ConsoleDisplay = ConsoleDisplay
    return {
        "tau2": types.ModuleType("tau2"),
        "tau2.utils": types.ModuleType("tau2.utils"),
        "tau2.utils.llm_utils": llm,
        "tau2.utils.utils": utils,
        "tau2.utils.display": display,
        "tau2.runner": types.ModuleType("tau2.runner"),
        "tau2.runner.helpers": helpers,
    }


def test_tau_common_quiet_noops_functions_and_mutes_console():
    """_quiet_tau2_noise no-ops the two noisy functions (incl. the by-value import in
    runner.helpers) AND mutes tau2's rich console by default."""
    import sys
    from gbench.runners.eval_suites import tau_common as TC
    fake = _fake_tau2_modules()
    TC._QUIETED = False
    old = os.environ.pop("TAU2_VERBOSE", None)
    try:
        with patch.dict(sys.modules, fake):
            TC._quiet_tau2_noise()
        assert fake["tau2.utils.llm_utils"].get_response_cost("x") == 0.0
        assert fake["tau2.utils.utils"].get_commit_hash() == "unknown"
        assert fake["tau2.runner.helpers"].get_commit_hash() == "unknown"
        # rich console replaced with a quiet one
        console = fake["tau2.utils.display"].ConsoleDisplay.console
        assert getattr(console, "quiet", False) is True
        # litellm's noisy loggers raised to ERROR (kills the per-call Gemini deprecation spam)
        import logging as _logging
        assert _logging.getLogger("LiteLLM").level == _logging.ERROR
    finally:
        TC._QUIETED = False
        if old is not None:
            os.environ["TAU2_VERBOSE"] = old


def test_tau_common_verbose_env_keeps_console():
    """TAU2_VERBOSE=1 leaves tau2's rich console untouched (full panels)."""
    import sys
    from gbench.runners.eval_suites import tau_common as TC
    fake = _fake_tau2_modules()
    TC._QUIETED = False
    try:
        with patch.dict(sys.modules, fake), patch.dict(os.environ, {"TAU2_VERBOSE": "1"}):
            TC._quiet_tau2_noise()
        assert fake["tau2.utils.display"].ConsoleDisplay.console == "loud-console"
    finally:
        TC._QUIETED = False


def test_tau_common_hard_errors_when_no_domain_produces_results():
    # No-skip policy: if the simulator produces no results, hard-error (not a silent skip).
    from gbench.runners.eval_suites import tau_common as TC
    with patch.object(TC, "check_tau_env_prerequisites", return_value=(True, "")), \
         patch.object(TC, "_install_robustness_patches", lambda *a, **k: None), \
         patch.object(TC, "_configure_evaluator_llm", lambda *a, **k: None), \
         patch.object(TC, "_run_one_domain", side_effect=RuntimeError("simulator boom")):
        with pytest.raises(RuntimeError, match="docs/evals/tau2.md"):
            TC.run_tau_env("tau2", ["airline"], "m", "http://x", 2, None, "docs/evals/tau2.md")
