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

"""Harness errors in tau2 / tau3, plus complexfuncbench and livebench data and grading checks.

  * tau2 / tau3   - a simulation that ends in an infrastructure error is never scored (left out
                    of pass^k and the mean reward, status completed_with_errors, named in
                    leaderboard_comparable_reason); one from a harness failure (an outage of the
                    Gemini user simulator or judge, of the model endpoint, failed embedding
                    calls) is a harness_errors entry per domain with its sources and error
                    classes (the ones raised on the agent's own turn are covered in
                    test_tau_mediums_2m).
  * complexfuncbench - the dataset is read at a pinned Hub commit and checked by sha256; the
                    embedder's dtype and upstream's fp16 FlagModel are recorded; the cascade
                    comment says 3 tiers; the doc names the `evals` extra for scipy.
  * livebench     - instruction_following scored with no answer rows to count is not comparable;
                    the image build bakes in the NLTK / spaCy data and fails when a download fails.

No network, no Docker, no model server, no Gemini call: everything external is stubbed.
"""

import hashlib
import inspect
import json
import os
import re
import sys
import threading
import tomllib
import types
from collections import Counter
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import complexfuncbench as C
from gbench.runners.eval_suites import livebench as LB
from gbench.runners.eval_suites import tau_common as TC
from gbench.runners.eval_suites import tau_common as TC_GUARD

ROOT = Path(__file__).resolve().parent.parent
GEMINI_KEY = "gemini-infra-status-key-0123456789"
ENDPOINT_KEY = "endpoint-infra-status-key-99"


def _read(*parts):
    return " ".join((ROOT.joinpath(*parts)).read_text(encoding="utf-8").split())


# =========================================================================== #
# tau2 / tau3
# =========================================================================== #
_TAU_ENV = ("GBENCH_TAU2_TEMPERATURE", "TAU2_TEMPERATURE", "GBENCH_TAU3_TEMPERATURE",
            "GBENCH_TAU2_TOP_P", "TAU2_TOP_P", "GBENCH_TAU2_TOP_K", "TAU2_TOP_K",
            "GBENCH_TAU2_USER_LLM", "TAU2_USER_LLM", "GBENCH_TAU2_EVAL_LLM", "TAU2_EVAL_LLM",
            "GBENCH_TAU2_NUM_TRIALS", "TAU2_NUM_TRIALS", "GBENCH_TAU2_USER_TEMPERATURE",
            "TAU2_USER_TEMPERATURE", "GBENCH_TAU2_SAVE_TRACES", "TAU2_SAVE_TRACES",
            "GEMINI_API_KEY", "GEMINI_API_KEYS", "OPENAI_API_KEY", TC._AGENT_KEY_ENV)


def _ok(task, reward):
    return types.SimpleNamespace(task_id=task, termination_reason="agent_stop",
                                 reward_info=types.SimpleNamespace(reward=reward), info=None)


def _infra(task, error_type, error):
    """A simulation as tau2's run_with_retry records one whose every attempt raised."""
    return types.SimpleNamespace(
        task_id=task, termination_reason="infrastructure_error", reward_info=None,
        info={"error": error, "error_type": error_type, "error_traceback": "...",
              "failed_after_attempts": 4})


class _Tau2:
    """The tau2 surface _run_one_domain touches: TextRunConfig, run_domain (writes results.json
    under save_to and returns the canned simulations of the domain) and compute_metrics (pass^1
    over the simulations that did not end in an infrastructure error, as tau2 computes it)."""

    def __init__(self, sims_by_domain):
        self.sims_by_domain = sims_by_domain

    def modules(self):
        names = ["tau2", "tau2.data_model", "tau2.data_model.simulation", "tau2.run",
                 "tau2.metrics", "tau2.metrics.agent_metrics"]
        m = {n: types.ModuleType(n) for n in names}

        class TextRunConfig:
            def __init__(self, **kw):
                self.__dict__.update(kw)

        def run_domain(config):
            os.makedirs(config.save_to, exist_ok=True)
            with open(os.path.join(config.save_to, "results.json"), "w") as fh:
                fh.write("{}")
            return types.SimpleNamespace(simulations=list(self.sims_by_domain[config.domain]))

        def compute_metrics(results):
            scored = [s.reward_info.reward for s in results.simulations
                      if s.reward_info is not None]
            return types.SimpleNamespace(
                pass_hat_ks={1: sum(scored) / len(scored)} if scored else {})

        m["tau2.data_model.simulation"].TextRunConfig = TextRunConfig
        m["tau2.run"].run_domain = run_domain
        m["tau2.metrics.agent_metrics"].compute_metrics = compute_metrics
        return m


@pytest.fixture
def tau(monkeypatch, tmp_path):
    for name in _TAU_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", GEMINI_KEY)
    monkeypatch.setenv(TC._AGENT_KEY_ENV, ENDPOINT_KEY)
    monkeypatch.setenv("GBENCH_TAU2_SAVE_TRACES", str(tmp_path / "traces"))
    monkeypatch.setattr(TC, "_SECRETS", [])
    monkeypatch.setattr(TC, "_AGENT_ROUTE", {})
    monkeypatch.setattr(TC, "_LAST_RETRIEVAL", {})
    monkeypatch.setattr(TC, "_LLM_FAILURES", {})
    monkeypatch.setattr(TC, "_EMBEDDING_FAILURES", {})
    monkeypatch.setattr(TC, "_EMBEDDING_ABORTS", {})
    monkeypatch.setattr(TC, "_DOMAIN_INFRA", {})
    monkeypatch.setattr(TC, "_CURRENT_DOMAIN", {"domain": None})
    monkeypatch.setattr(TC, "_EMBED_FAILURE_SEEN", threading.local())
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

    def run(sims_by_domain, eval_name="tau2"):
        with mock.patch.dict(sys.modules, _Tau2(sims_by_domain).modules()):
            return TC.run_tau_env(eval_name, list(sims_by_domain), "m", "http://x/v1", 1, None,
                                  f"docs/evals/{eval_name}.md")
    return run


def _gaps(r):
    return (r["leaderboard_comparable_reason"] or "").split("; ")


def test_a_user_simulator_outage_is_a_harness_error_never_the_agents_outcome(tau):
    """The review's reproduction: the Gemini user simulator fails every call of one task, tau2's
    run_with_retry records INFRASTRUCTURE_ERROR, and the run must be completed_with_errors."""
    outage = (f"litellm.APIConnectionError: GeminiException - model overloaded "
              f"(key={GEMINI_KEY})")
    r = tau({"retail": [_ok(0, 1.0), _ok(1, 0.0), _ok(2, 1.0),
                        _infra(3, "APIConnectionError", outage)]})
    assert r["status"] == "completed_with_errors"
    assert r["tau2_report"]["infra_errors"] == 1
    # never scored as the agent's outcome: out of pass^1 and out of the mean reward
    assert r["accuracy"] == round(100.0 * 2 / 3, 2)
    assert r["tau2_report"]["mean_reward"] == round(2 / 3, 4)
    assert r["tau2_report"]["mean_reward_by_domain"] == {"retail": round(2 / 3, 4)}
    assert r["harness_errors"] == [{
        "error": ("simulations ended in an infrastructure error from a harness failure (tau2 "
                  "records one when every attempt of a simulation failed; compute_metrics leaves "
                  "it out of pass^k)"),
        "domain": "retail", "infrastructure_error_simulations": 1,
        "sources": {"unrecorded": 1},                   # the stand-in records no traceback
        "error_types": {"APIConnectionError": 1},
        "example_error": ("litellm.APIConnectionError: GeminiException - model overloaded "
                          "(key=[REDACTED])")}]
    assert r["agent_turn_errors"] == []
    assert r["leaderboard_comparable"] is False
    assert ("1 simulation(s) ended in infrastructure errors from harness failures, not the "
            "agent's outcome (retail 1 [unrecorded/APIConnectionError x1]): tau2's metrics "
            "exclude them, so pass^k covers fewer than every task x trial") in _gaps(r)
    dumped = json.dumps(r)
    assert GEMINI_KEY not in dumped and ENDPOINT_KEY not in dumped


def test_each_domain_and_error_class_is_named(tau):
    r = tau({"airline": [_ok(0, 1.0), _infra(1, "InternalServerError", "judge 500"),
                         _infra(2, "APIConnectionError", "endpoint refused")],
             "retail": [_ok(0, 1.0), _ok(1, 1.0)],
             "telecom": [_ok(0, 0.0), _infra(1, "APIConnectionError", "endpoint refused"),
                         _infra(2, "APIConnectionError", "endpoint refused"),
                         _infra(3, "Timeout", "judge timed out")]})
    assert r["status"] == "completed_with_errors"
    assert r["tau2_report"]["infra_errors"] == 5
    records = {h["domain"]: h for h in r["harness_errors"]}
    assert set(records) == {"airline", "telecom"}           # retail had none
    assert records["telecom"]["error_types"] == {"APIConnectionError": 2, "Timeout": 1}
    assert records["telecom"]["example_error"] == "endpoint refused"
    gap = next(g for g in _gaps(r) if g.startswith("5 simulation(s) ended in infrastructure"))
    # domains in run order; within one, the most frequent source/error class first
    assert ("(airline 2 [unrecorded/APIConnectionError x1, unrecorded/InternalServerError x1], "
            "telecom 3 [unrecorded/APIConnectionError x2, unrecorded/Timeout x1])") in gap
    assert r["tau2_report"]["mean_reward_by_domain"] == {"airline": 1.0, "retail": 1.0,
                                                         "telecom": 0.0}


def test_a_clean_run_is_success_with_no_harness_errors(tau):
    r = tau({"retail": [_ok(0, 1.0), _ok(1, 0.0)]})
    assert r["status"] == "success" and r["harness_errors"] == []
    assert r["tau2_report"]["infra_errors"] == 0
    assert not any("infrastructure errors" in g for g in _gaps(r))


def test_a_domain_lost_to_an_outage_hard_errors_naming_the_error_classes(tau):
    with pytest.raises(RuntimeError) as ei:
        tau({"retail": [_infra(0, "APIConnectionError", "down"),
                        _infra(1, "APIConnectionError", "down")]})
    msg = str(ei.value)
    assert msg.startswith("[INFRA REQUIRED]")
    assert "(all 2 ended in an infrastructure error: APIConnectionError x2)" in msg


def test_an_infra_count_without_detail_still_sets_the_status(monkeypatch, tau):
    """A domain whose count arrives without per-simulation detail (no `info`) is still a harness
    error: the status and the entry do not depend on the error class being known."""
    def one(domain, *a, **k):
        TC._DOMAIN_PASS_HAT_K[domain] = {"pass^1": 0.5}
        return 4, 1.0, 1, 2
    monkeypatch.setattr(TC, "_run_one_domain", one)
    r = tau({"retail": []})
    assert r["status"] == "completed_with_errors"
    assert r["harness_errors"] == [{
        "error": ("simulations ended in an infrastructure error from a harness failure (tau2 "
                  "records one when every attempt of a simulation failed; compute_metrics leaves "
                  "it out of pass^k)"),
        "domain": "retail", "infrastructure_error_simulations": 2, "sources": {},
        "error_types": {}, "example_error": None}]
    assert r["tau2_report"]["mean_reward"] == 0.5           # 1.0 over the 2 evaluable
    assert "(retail 2)" in r["leaderboard_comparable_reason"]


def test_the_embedding_and_infrastructure_records_sit_side_by_side(tau, monkeypatch):
    """tau3: failed embedding calls and the simulation they cost are two different records."""
    TC._EMBEDDING_FAILURES.update({"banking_knowledge": 4})
    TC._EMBEDDING_ABORTS.update({"banking_knowledge": 4})
    TC._DOMAIN_INFRA.update({"banking_knowledge": {
        "count": 1, "error_types": {"TauHarnessError": 1}, "example_error": "harness error"}})
    records = TC._harness_error_records(["banking_knowledge"], dict(TC._DOMAIN_INFRA))
    assert [sorted(r) for r in records] == [
        ["domain", "error", "failed_embedding_calls", "stopped_simulation_attempts"],
        ["domain", "error", "error_types", "example_error", "infrastructure_error_simulations",
         "sources"]]
    assert TC._harness_error_records(["retail"], dict(TC._DOMAIN_INFRA)) == []


def test_tau2s_own_run_with_retry_record_is_summarized(monkeypatch):
    progress = pytest.importorskip("tau2.runner.progress")

    class ServiceUnavailableError(Exception):
        pass

    def attempt():
        raise ServiceUnavailableError(f"GeminiException - 503 UNAVAILABLE key {GEMINI_KEY}")

    sim = progress.run_with_retry(attempt, task=types.SimpleNamespace(id="t1"), trial=0,
                                  seed=300, max_retries=1, retry_delay=0.0,
                                  console_display=False)
    assert TC._is_infra_error(sim)
    monkeypatch.setattr(TC, "_SECRETS", [GEMINI_KEY])
    summary = TC._infra_error_summary([sim])
    # the attempt raised outside any agent, user simulator, tool or grading frame
    assert summary == {"count": 1, "error_types": {"ServiceUnavailableError": 1},
                       "sources": {"other": 1}, "causes": {"other/ServiceUnavailableError": 1},
                       "example_error": "GeminiException - 503 UNAVAILABLE key [REDACTED]"}


def test_long_error_text_is_cut():
    sim = _infra(0, "X", "y" * 1000)
    assert len(TC._infra_error_summary([sim])["example_error"]) == TC._INFRA_ERROR_TEXT_CHARS
    assert TC._INFRA_ERROR_TEXT_CHARS == 300


def test_the_status_rule_is_in_the_code():
    src = inspect.getsource(TC._run_domains)
    assert '"completed_with_errors" if (harness_errors or infra_all) else "success"' in src


def test_the_docs_state_the_infrastructure_error_rule():
    tau2, tau3 = _read("docs", "evals", "tau2.md"), _read("docs", "evals", "tau3.md")
    for text in ("**Infrastructure errors are never scored.**", "`run_with_retry`",
                 "`infrastructure_error_simulations`", "`error_types`", "`example_error`",
                 "cut to 300 characters", "status `completed_with_errors` instead of `success`",
                 "an outage of the Gemini user simulator or judge",
                 "gbench leaves it out of `tau2_report.mean_reward`",
                 "infrastructure errors from harness failures, not the agent's outcome (<domain> "
                 "<n> [<source>/<error class> x<n>], ...)"):
        assert text in tau2, text
    assert "**Infrastructure errors are harness errors.**" not in tau2
    assert "error counts as 0" not in tau2 and "error counts as 0" not in tau3
    assert "counts as 0), not a leaderboard number" not in tau3
    assert "infrastructure error for any cause" in tau3
    assert "`infrastructure_error_simulations`, `error_types`, `example_error`" in tau3
    from gbench.runners.eval_suites import tau2 as T2, tau3 as T3
    assert "status completed_with_errors" in inspect.getsource(T2)
    assert "infrastructure error for any cause" in " ".join(inspect.getsource(T3).split())


# =========================================================================== #
# complexfuncbench
# =========================================================================== #
def test_complexfuncbench_dataset_is_pinned_and_checked(monkeypatch, tmp_path):
    import huggingface_hub
    assert re.fullmatch(r"[0-9a-f]{40}", C.HF_REVISION)
    assert re.fullmatch(r"[0-9a-f]{64}", C.HF_SHA256)
    data = tmp_path / "ComplexFuncBench.jsonl"
    data.write_text("".join(json.dumps({"id": f"Hotels-{i}"}) + "\n" for i in range(3)),
                    encoding="utf-8")
    seen = {}

    def fake_download(**kw):
        seen.update(kw)
        return str(data)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    with pytest.raises(RuntimeError) as ei:
        C._load_dataset(None)
    assert seen == {"repo_id": "THUDM/ComplexFuncBench", "filename": "ComplexFuncBench.jsonl",
                    "repo_type": "dataset", "revision": C.HF_REVISION}
    msg = str(ei.value)
    assert msg.startswith("[INFRA REQUIRED] complexfuncbench")
    assert f"not the pinned {C.HF_SHA256}" in msg
    monkeypatch.setattr(C, "HF_SHA256", hashlib.sha256(data.read_bytes()).hexdigest())
    assert len(C._load_dataset(None)) == 3


def test_complexfuncbench_result_records_the_dataset_and_the_embedder(monkeypatch):
    monkeypatch.setattr(C, "check_complexfuncbench_prerequisites", lambda: (True, ""))

    async def fake_drive(*a, **k):
        return {"success_rate": 50.0, "call_accuracy": 60.0, "completeness": 1.5,
                "correctness": 1.0, "n_scored": 2, "n_request_failed": 0, "n_success": 1,
                "per_domain_success_rate": None, "_is_full": False, "_n_items": 2,
                "_embed_backend": "transformers", "_embed_dtype": "float32",
                "_judge_usage": Counter({"gemini-x": 3})}
    monkeypatch.setattr(C, "_drive_all", fake_drive)
    r = C.run_complexfuncbench("m", "http://x/v1", 1, limit=2)
    assert r["dataset"] == f"THUDM/ComplexFuncBench@{C.HF_REVISION}"
    assert r["dataset_sha256"] == C.HF_SHA256
    assert r["embedding_backend"] == "transformers" and r["embedding_dtype"] == "float32"
    assert "use_fp16=True" in r["embedding_upstream"] and "float32" in r["embedding_upstream"]


def test_complexfuncbench_embedder_dtype_is_read_from_the_loaded_model():
    torch = pytest.importorskip("torch")
    emb = C._Embedder()
    assert emb.dtype is None                                 # not loaded: nothing to report
    model = types.SimpleNamespace(parameters=lambda: iter([torch.zeros(1)]))
    emb._impl = (None, model)
    assert emb.dtype == "float32"


def test_complexfuncbench_comment_and_doc_follow_the_code():
    src = (ROOT / "gbench" / "runners" / "eval_suites" / "complexfuncbench.py").read_text(
        encoding="utf-8")
    assert "4-tier" not in src and "3-tier match cascade: rule_based -> value_checker" in src
    with open(ROOT / "pyproject.toml", "rb") as fh:
        extras = tomllib.load(fh)["project"]["optional-dependencies"]
    assert any(d.split(">")[0].split("=")[0].strip() == "scipy" for d in extras["evals"])
    assert not any(d.split(">")[0].split("=")[0].strip() == "scipy" for d in extras["dev"])
    doc = _read("docs", "evals", "complexfuncbench.md")
    assert "it is in gbench's `evals` extra" in doc and "`dev` extra" not in doc
    for text in (C.HF_REVISION, C.HF_SHA256, "`dataset_sha256`", "`embedding_dtype`",
                 "`embedding_upstream`", "`use_fp16=True`"):
        assert text in doc, text
    assert doc.isascii()


# =========================================================================== #
# livebench
# =========================================================================== #
def _write_answers(tmp_path, category, task, n):
    d = tmp_path / LB._ANSWER_DIR / "data" / "live_bench" / category / task / "model_answer"
    d.mkdir(parents=True, exist_ok=True)
    row = {"question_id": "q", "choices": [{"index": 0, "turns": ["ok"]}],
           "total_output_tokens": 10}
    (d / "m.jsonl").write_text("".join(json.dumps(dict(row, question_id=f"q{i}")) + "\n"
                                       for i in range(n)), encoding="utf-8")


def _run_lb(monkeypatch, tmp_path, if_rows):
    base._RUN_KNOBS.clear()
    for name in ("GBENCH_LIVEBENCH_RELEASE", "GBENCH_LIVEBENCH_IMAGE",
                 "GBENCH_LIVEBENCH_CATEGORIES", "GBENCH_LIVEBENCH_TEMPERATURE",
                 "GBENCH_LIVEBENCH_MODEL", "GBENCH_LIVEBENCH_EXTRA_ARGS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(LB, "_check_prereqs", lambda image: None)
    # the network checks before the run container are docker calls of their own
    # (tests/test_guard_code_images.py)
    monkeypatch.setattr(TC_GUARD, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC_GUARD.container_url(url))
    monkeypatch.setattr(LB, "upstream_model_config", lambda image, names: {
        "name": None, "config": None, "commit": LB.LIVEBENCH_COMMIT})
    monkeypatch.setattr(LB.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))

    def fake_run(cmd, **_kw):
        cols = LB._CATEGORIES
        (tmp_path / "all_groups.csv").write_text(
            "model,average," + ",".join(cols) + "\nm,50.0," + ",".join("50.0" for _ in cols)
            + "\n", encoding="utf-8")
        for c in cols:
            if c != "instruction_following":
                _write_answers(tmp_path, c, "t", 2)
        for task, n in if_rows.items():
            _write_answers(tmp_path, "instruction_following", task, n)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(LB.subprocess, "run", fake_run)
    return LB.run_livebench("m", "http://h/v1", 4, max_output_tokens=4096)


_IF_NOT_FOUND = ("instruction_following answer rows were not found in the run's output, so the "
                 "equal task counts that make LiveBench's pooled old-format "
                 "instruction_following score equal the mean of the task means (which the "
                 "recorded patch relies on) could not be verified")


def test_livebench_if_scored_without_answer_rows_is_not_comparable(monkeypatch, tmp_path):
    """The default no-think run sends the official cap, so the cut-off count does not look at
    the answer rows; the instruction_following count gate must not pass silently without them."""
    r = _run_lb(monkeypatch, tmp_path, {})
    assert "instruction_following" in r["dimension_scores"]
    assert r["max_output_tokens"] == 4096                   # the official cap: no cut-off gate
    assert r["leaderboard_comparable"] is False
    assert _IF_NOT_FOUND in r["leaderboard_comparable_reason"]
    assert "unequal numbers" not in r["leaderboard_comparable_reason"]


def test_livebench_equal_if_rows_stay_comparable(monkeypatch, tmp_path):
    r = _run_lb(monkeypatch, tmp_path, {"paraphrase": 5, "simplify": 5})
    assert r["leaderboard_comparable"] is True, r.get("leaderboard_comparable_reason")


def test_livebench_image_bakes_the_grading_data_and_fails_on_a_failed_download():
    text = (ROOT / "docker" / "livebench.Dockerfile").read_text(encoding="utf-8")
    joined = re.sub(r"\\\n", "", text)
    run = next(l for l in joined.splitlines() if l.startswith("RUN ") and "nltk_data" in l)
    assert "|| true" not in run
    # each NLTK package is downloaded from the pinned nltk_data commit and checked against its
    # sha256; a download that fails or differs exits non-zero, which fails the build
    assert "hashlib.sha256(b).hexdigest() != pins[k]" in run and "sys.exit(" in run
    for name in ("'tokenizers/punkt'", "'tokenizers/punkt_tab'",
                 "'taggers/averaged_perceptron_tagger'", "'corpora/stopwords'"):
        assert name in run, name
    assert "python -m spacy download en_core_web_sm" in run
    assert "spacy.load('en_core_web_sm')" in run
    # the Python program is valid once Docker joins the continuation lines
    program = re.search(r'python -c "(import hashlib, os, sys, urllib.request, zipfile;.*?)"',
                        run).group(1)
    compile(program, "<RUN>", "exec")
    doc = _read("docs", "evals", "livebench.md")
    assert "`averaged_perceptron_tagger` and `stopwords`, and spaCy's `en_core_web_sm`" in doc
    assert "A download that fails fails the build" in doc
    assert "instruction_following answer rows were not found in the run's output" in doc
