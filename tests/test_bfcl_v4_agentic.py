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

"""Tests for the BFCL v4 agentic wrapper (canonical `bfcl-eval` harness).

Everything here runs offline: the harness itself, the model endpoint and the search
backend are all exercised through their seams, never over the network.
"""

import json
import os

import pytest

from gbench.runners.eval_suites import SUITES
from gbench.runners.eval_suites.bfcl_v4_agentic import (
    WEB_SEARCH_CATEGORIES,
    _parse_scores,
    check_bfcl_prerequisites,
    run_bfcl_v4_agentic,
    search_backend,
)

def _needs_bfcl_eval():
    """The canonical harness package, for the tests that import it (the rest run without it)."""
    return pytest.importorskip("bfcl_eval", reason="canonical BFCL harness not installed")


def test_both_bfcl_suites_are_registered():
    """The v3 LIVE slice keeps running under an honest name; v4 is the agentic track."""
    assert "bfcl_live" in SUITES
    assert "bfcl_v4_agentic" in SUITES


def test_agentic_categories_match_the_harness():
    _needs_bfcl_eval()
    from bfcl_eval.constants.category_mapping import TEST_COLLECTION_MAPPING
    cats = set(TEST_COLLECTION_MAPPING["agentic"])
    assert cats == {"memory_kv", "memory_vector", "memory_rec_sum",
                    "web_search_base", "web_search_no_snippet"}
    assert set(WEB_SEARCH_CATEGORIES) <= cats


def test_search_backend_selection(monkeypatch):
    """Gemini Google Search grounding is the only backend: no Gemini key, no backend, whatever
    else is configured (no other search engine, no opt-in)."""
    import gbench.runners.eval_suites.bfcl_v4_agentic as B
    checked = []
    monkeypatch.setattr(B, "gemini_key_live_valid", lambda key: checked.append(key) or (True, ""))
    for k in ("GEMINI_API_KEY", "GEMINI_API_KEYS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SERPAPI_API_KEY", "serp-key-not-used-0123")
    monkeypatch.setenv("BFCL_SEARCH_BACKEND", "serpapi")
    backend, why = search_backend()
    assert backend is None and "GEMINI_API_KEY" in why and checked == []
    # a set key is also checked with the Gemini API (stubbed here: live key)
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key-0123456789")
    assert search_backend() == (B.SEARCH_BACKEND, "") and checked == ["gem-key-0123456789"]
    monkeypatch.setattr(B, "gemini_key_live_valid", lambda key: (False, "HTTP 401"))
    backend, why = search_backend()
    assert backend is None and "rejected" in why and "HTTP 401" in why
    assert "gem-key-0123456789" not in why


def test_prereq_unreachable_endpoint_hard_errors(tmp_path, monkeypatch):
    # No-skip policy: an un-provisioned canonical harness must hard-error, not skip.
    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    ok, reason, cats = check_bfcl_prerequisites("http://127.0.0.1:9/v1")
    assert ok is False and "not reachable" in reason and cats == []

    with pytest.raises(RuntimeError, match="docs/evals/bfcl_v4_agentic.md"):
        run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1)


def test_prereq_unwritable_project_root_fails_prerequisites(monkeypatch):
    monkeypatch.setenv("BFCL_PROJECT_ROOT", "/proc/definitely/not/writable")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    ok, reason, _ = check_bfcl_prerequisites("http://127.0.0.1:9/v1")
    assert ok is False and "not writable" in reason


def test_missing_search_backend_is_infra_required_unless_memory_is_selected(tmp_path,
                                                                            monkeypatch):
    """No search key: the suite's own run of the whole track hard-errors (infra_required; its
    score weighs the Web Search setting), naming the Memory categories to select instead, and
    nothing is generated. Selecting the Memory categories runs them. bfcl's agentic part (the
    same harness, run for bfcl) still reports the Web Search setting as its own infra_required
    entry and runs Memory. A run of web_search alone hard-errors. The prerequisite check itself
    always offers all five categories."""
    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path))
    for k in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "BFCL_SEARCH_BACKEND"):
        monkeypatch.delenv(k, raising=False)

    import gbench.runners.eval_suites.bfcl_v4_agentic as B
    monkeypatch.setattr(B, "_endpoint_ok", lambda *_a, **_k: (True, ""), raising=False)
    # endpoint check is inline, so aim at the live-check seam via a stub urlopen
    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())

    # the harness image and its file pin have their own tests; this one is about the search key
    monkeypatch.setattr(B, "image_problem", lambda: None)
    monkeypatch.setattr(B, "run_in_image", lambda *a, **k: {
        "release": B.BFCL_EVAL_RELEASE, "agentic_categories": list(B.AGENTIC_CATEGORIES),
        "agentic_mismatch": None})
    # the network checks for `bfcl generate` are docker calls of their own
    # (tests/test_guard_code_images.py)
    import gbench.runners.eval_suites.tau_common as TC
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC.container_url(url))
    ok, _reason, cats = check_bfcl_prerequisites("http://x/v1")
    assert ok is True
    assert set(WEB_SEARCH_CATEGORIES) <= set(cats) and len(cats) == 5
    calls = []
    monkeypatch.setattr(B, "_run_cli", lambda argv, env, t, model=None, backend=None:
                        (calls.append((list(argv), dict(env))), (0, ""))[1])
    monkeypatch.setattr(B, "_parse_scores", lambda root, model, cats: {
        c: {"correct": 1, "total": 2, "accuracy": 50.0} for c in cats})
    with pytest.raises(RuntimeError, match="INFRA REQUIRED") as ei:
        run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    assert "--eval-categories memory_kv,memory_vector,memory_rec_sum" in str(ei.value)
    assert not calls
    mem = run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1,
                              eval_categories="memory_kv,memory_vector,memory_rec_sum")
    gen = next(argv for argv, _env in calls if argv[0] == "generate")
    assert gen[gen.index("--test-category") + 1] == "memory_kv,memory_vector,memory_rec_sum"
    assert "web_search" not in mem["settings"] and mem["accuracy"] is not None
    calls.clear()
    with B.suite_context("bfcl", "docs/evals/bfcl.md"):
        res = run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    gen = next(argv for argv, _env in calls if argv[0] == "generate")
    assert gen[gen.index("--test-category") + 1] == "memory_kv,memory_vector,memory_rec_sum"
    assert res["settings"]["web_search"]["status"] == "infra_required"
    assert "GEMINI_API_KEY" in res["settings"]["web_search"]["infra_required"]
    assert res["settings"]["web_search"]["prerequisite"] == "GEMINI_API_KEY"
    assert res["settings_infra_required"] == ["web_search"]
    assert res["status"] == "completed_with_errors" and res["overall_accuracy_agentic"] is None
    calls.clear()
    with pytest.raises(RuntimeError, match="INFRA REQUIRED"):
        run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1,
                            eval_categories=",".join(WEB_SEARCH_CATEGORIES))
    assert not calls


def test_the_only_search_engine_is_gemini_grounding():
    """`search_engine_query` is served by Gemini Google Search grounding (search_tool's cascade);
    no SerpAPI path, key probe or selectable backend is left."""
    import gbench.runners.eval_suites.bfcl_v4_agentic as B
    for name in ("serpapi_account", "_observe_serpapi_calls", "SERPAPI_ACCOUNT_URL",
                 "serpapi_log_summary", "_serpapi_watch", "Unconfirmed", "_gemini_search",
                 "_patch_search_backend"):
        assert not hasattr(B, name), name
    assert callable(B._serve_web_search_with_gemini) and B.SEARCH_KEY_NAME == "GEMINI_API_KEY"
    assert not [n for n in B._CHILD_ENV_NAMES if "SERP" in n]


def test_score_parsing_into_gbench_shape(tmp_path):
    """_parse_scores must turn the harness' score files into category_accuracy.

    The harness nests scores by track (`<model>/agentic/memory/kv/..._score.json`) and the
    summary is the FIRST line of a JSONL file whose remaining lines are per-sample records -
    a flat glob or a whole-file json.load both yield nothing.
    """
    model = "google/gemma-4-26B-A4B-it"
    d = tmp_path / model.replace("/", "_") / "agentic" / "memory" / "kv"
    d.mkdir(parents=True)
    (d / "BFCL_v4_memory_kv_score.json").write_text(
        json.dumps({"accuracy": 0.14838709677419354, "correct_count": 23, "total_count": 155})
        + "\n" + json.dumps({"id": "memory_kv_0", "valid": False}) + "\n",
        encoding="utf-8")
    per_cat = _parse_scores(tmp_path, model, ["memory_kv", "memory_vector"])
    assert per_cat["memory_kv"] == {"correct": 23, "total": 155, "accuracy": 14.84}
    assert "memory_vector" not in per_cat          # categories that did not run are absent


def test_child_bootstrap_registers_model_and_patches_search():
    """The model registration is re-applied inside the harness subprocess, and the web_search
    tool is served there by Gemini grounding when the run searches."""
    import gbench.runners.eval_suites.bfcl_v4_agentic as B
    code = B._CHILD_BOOTSTRAP.format(model="m", args=["generate"])
    assert "_register_model('m')" in code or '_register_model("m")' in code
    assert '_serve_web_search_with_gemini(os.environ.get("GBENCH_BFCL_SEARCH_LOG"))' in code
    assert "_patch_search_backend" not in code and "serpapi" not in code.lower()
    assert "from bfcl_eval.__main__ import cli" in code


def test_registered_model_uses_native_function_calling():
    """A prompt-mode handler's decode fails on a model served with native tool calling; the
    registered entry is FC mode."""
    _needs_bfcl_eval()
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
    import gbench.runners.eval_suites.bfcl_v4_agentic as B
    name = "test-only/served-model-fake-it"
    try:
        B._register_model(name)
        cfg = MODEL_CONFIG_MAPPING[name]
        assert cfg.is_fc_model is True
        # the registered handler is the OpenAI completions handler plus the --thinking toggle
        assert issubclass(cfg.model_handler, OpenAICompletionsHandler)
        assert cfg.model_handler.__name__ == "_GbenchOpenAICompletionsHandler"
    finally:
        MODEL_CONFIG_MAPPING.pop(name, None)


def test_openai_handler_shim_survives_a_prose_turn():
    """The gbench handler decodes exactly as the harness' own `OpenAICompletionsHandler`.

    A prose (no tool call) turn makes the FC decoder raise; the harness catches it, logs
    "Failed to decode the model response. Proceed to next turn." and ends the turn, for every
    FC model. gbench no longer re-routes prose to the prompting decoder, which could execute a
    call written as text that the harness would not.
    """
    _needs_bfcl_eval()
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
    from gbench.runners.eval_suites.bfcl_v4_agentic import _patched_openai_handler

    patched_cls = _patched_openai_handler()
    patched = patched_cls.__new__(patched_cls)
    patched.is_fc_model = True
    upstream = OpenAICompletionsHandler.__new__(OpenAICompletionsHandler)
    upstream.is_fc_model = True

    for prose in ("I do not have your first name stored in my memory records.",
                  "[get_user_info(user_id=7)]"):
        with pytest.raises(AttributeError):
            upstream.decode_execute(prose, False)
        with pytest.raises(AttributeError):
            patched.decode_execute(prose, False)

    # and genuine tool calls decode exactly as before
    tool_calls = [{"get_user_info": '{"user_id": 7, "verbose": true}'}]
    assert patched.decode_execute(tool_calls, False) == \
        upstream.decode_execute(tool_calls, False)


def test_openai_handler_shim_is_registered():
    """The registration must use the patched handler, not the upstream one."""
    _needs_bfcl_eval()
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
    from gbench.runners.eval_suites.bfcl_v4_agentic import _register_model

    name = "test/gbench-shim-probe"
    MODEL_CONFIG_MAPPING.pop(name, None)
    try:
        _register_model(name)
        handler = MODEL_CONFIG_MAPPING[name].model_handler
        assert handler.__name__ == "_GbenchOpenAICompletionsHandler"
        assert "decode_execute" not in vars(handler) and "decode_ast" not in vars(handler)
        assert "generate_with_backoff" in vars(handler)
    finally:
        MODEL_CONFIG_MAPPING.pop(name, None)


def test_generate_regenerates_by_default(monkeypatch, tmp_path):
    """`bfcl generate` must not silently re-score stale generations.

    Upstream defaults `--allow-overwrite=False`, which loads whatever is already in
    <BFCL_PROJECT_ROOT>/result/, subtracts those ids from the work list and generates
    nothing when the file is complete. Observed live: a 665-question track "completed" in
    19 s against generations from two days earlier. gbench's --skip-existing is a
    different mechanism entirely and does not affect it.
    """
    import gbench.runners.eval_suites.bfcl_v4_agentic as B

    calls = []

    def fake_cli(argv, env, timeout, model=None, backend=None):
        calls.append(argv)
        return 0, ""

    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(B, "_run_cli", fake_cli)
    monkeypatch.setattr(B, "check_bfcl_prerequisites", lambda url: (True, "", ["memory_kv"]))
    monkeypatch.setattr(B, "_register_model", lambda m: m)
    monkeypatch.setattr(B, "_parse_scores", lambda *a, **k: {
        "memory_kv": {"correct": 23, "total": 155, "accuracy": 14.84}})

    monkeypatch.delenv("BFCL_REUSE_GENERATIONS", raising=False)
    res = B.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    gen = next(c for c in calls if c[0] == "generate")
    assert "--allow-overwrite" in gen
    assert res["bfcl_report"]["generations"] == "regenerated"

    # explicit opt-in re-scores instead, and says so on the result
    calls.clear()
    monkeypatch.setenv("BFCL_REUSE_GENERATIONS", "1")
    res = B.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    gen = next(c for c in calls if c[0] == "generate")
    assert "--allow-overwrite" not in gen
    assert "reused" in res["bfcl_report"]["generations"]


# --------------------------------------------------------------------------- #
# --eval-limit: `bfcl generate` has no --limit, so gbench writes an id file
# --------------------------------------------------------------------------- #
AGENTIC_CATEGORIES = ["memory_kv", "memory_vector", "memory_rec_sum",
                      "web_search_base", "web_search_no_snippet"]


def _scenario(cat, sid):
    tail = sid[len(cat) + 1:] if sid.startswith(f"{cat}_") else sid
    parts = tail.split("-")
    return parts[1] if len(parts) >= 2 else cat


def test_eval_limit_selects_exactly_that_many_scored_questions():
    """The 2026-08-15 sweep ran all 665 questions under `--eval-limit 20`."""
    _needs_bfcl_eval()   # the id selection reads the harness' data
    from gbench.runners.eval_suites.bfcl_v4_agentic import _select_limited_ids
    sel = _select_limited_ids(AGENTIC_CATEGORIES, 20)
    scored = [i for ids in sel.values() for i in ids if "_prereq" not in i]
    assert len(scored) == 20


def test_eval_limit_spreads_across_categories_and_scenarios():
    _needs_bfcl_eval()   # the id selection reads the harness' data
    from gbench.runners.eval_suites.bfcl_v4_agentic import _select_limited_ids
    sel = _select_limited_ids(AGENTIC_CATEGORIES, 20)
    per_cat = {c: [i for i in ids if "_prereq" not in i] for c, ids in sel.items()}
    assert sorted(per_cat) == sorted(AGENTIC_CATEGORIES), "every category must be sampled"
    assert all(len(v) == 4 for v in per_cat.values()), per_cat
    # memory_kv's first 12 ids are all the `customer` scenario; a head would collapse.
    kv = {_scenario("memory_kv", i) for i in per_cat["memory_kv"]}
    assert len(kv) >= 3, f"memory_kv collapsed to {kv}"


def test_eval_limit_pulls_in_memory_prerequisites():
    """A memory question scored without its `depends_on` setup measures nothing."""
    _needs_bfcl_eval()   # the id selection reads the harness' data
    from gbench.runners.eval_suites.bfcl_v4_agentic import _select_limited_ids
    sel = _select_limited_ids(AGENTIC_CATEGORIES, 20)
    assert any("_prereq" in i for i in sel["memory_kv"])
    assert not any("_prereq" in i for i in sel["web_search_base"]), \
        "web_search has no prerequisites and must not gain any"


def test_eval_limit_is_deterministic():
    _needs_bfcl_eval()   # the id selection reads the harness' data
    from gbench.runners.eval_suites.bfcl_v4_agentic import _select_limited_ids
    assert _select_limited_ids(AGENTIC_CATEGORIES, 12) == \
           _select_limited_ids(AGENTIC_CATEGORIES, 12)


def test_limit_above_the_dataset_returns_everything():
    _needs_bfcl_eval()   # the id selection reads the harness' data
    from gbench.runners.eval_suites.bfcl_v4_agentic import _select_limited_ids
    sel = _select_limited_ids(AGENTIC_CATEGORIES, 100000)
    scored = [i for ids in sel.values() for i in ids if "_prereq" not in i]
    assert len(scored) == 665


def test_limited_run_uses_run_ids_and_a_separate_result_dir(tmp_path, monkeypatch):
    """`--run-ids` updates result files in place, so a limited run must not share the
    full run's directory - the harness would score the leftovers too."""
    _needs_bfcl_eval()   # the id selection reads the harness' data
    from gbench.runners.eval_suites import bfcl_v4_agentic as B

    calls = []

    def _fake_cli(argv, env, timeout, model=None, backend=None):
        calls.append(list(argv))
        return 0, "ok"

    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(B, "_run_cli", _fake_cli)
    # the id file is built in the harness image; here in this process
    monkeypatch.setattr(B, "run_in_image",
                        lambda name, module, func, kwargs, **k: getattr(B, func)(**kwargs))
    monkeypatch.setattr(B, "check_bfcl_prerequisites",
                        lambda url: (True, "", AGENTIC_CATEGORIES))
    monkeypatch.setattr(B, "search_backend", lambda: (B.SEARCH_BACKEND, ""))
    monkeypatch.setattr(B, "_register_model", lambda m: "gbench-model")
    # every generated category must be scored (a partial score set now hard-errors)
    monkeypatch.setattr(B, "_parse_scores",
                        lambda root, model, cats: {c: {"correct": 1, "total": 2} for c in cats})

    res = B.run_bfcl_v4_agentic("m", "http://x", concurrency=2, limit=20)

    gen, ev = calls[0], calls[1]
    assert "--run-ids" in gen and "--result-dir" in gen
    assert "--partial-eval" in ev, "a partial category set raises without --partial-eval"
    assert gen[gen.index("--result-dir") + 1] == ev[ev.index("--result-dir") + 1]
    assert gen[gen.index("--result-dir") + 1] != "result"

    id_file = tmp_path / "test_case_ids_to_generate.json"
    assert id_file.exists(), "the harness reads the id list from this file"
    written = json.loads(id_file.read_text())
    assert sum(1 for ids in written.values() for i in ids if "_prereq" not in i) == 20
    assert res["bfcl_report"]["eval_limit"] == 20


def test_unlimited_run_keeps_the_default_directories(tmp_path, monkeypatch):
    from gbench.runners.eval_suites import bfcl_v4_agentic as B

    calls = []
    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(B, "_run_cli",
                        lambda argv, env, t, model=None, backend=None: (calls.append(list(argv)), (0, "ok"))[1])
    monkeypatch.setattr(B, "check_bfcl_prerequisites", lambda url: (True, "", ["memory_kv"]))
    monkeypatch.setattr(B, "search_backend", lambda: (None, "no backend"))
    monkeypatch.setattr(B, "_register_model", lambda m: "gbench-model")
    monkeypatch.setattr(B, "_parse_scores",
                        lambda root, model, cats: {"memory_kv": {"correct": 1, "total": 2}})

    res = B.run_bfcl_v4_agentic("m", "http://x", concurrency=2)
    assert not any("--run-ids" in c for c in calls)
    assert not (tmp_path / "test_case_ids_to_generate.json").exists()
    assert res["bfcl_report"]["eval_limit"] is None
