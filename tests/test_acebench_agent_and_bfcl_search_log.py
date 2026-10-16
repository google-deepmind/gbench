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

"""Function-calling suites: the web search log of the BFCL agentic track, T-Eval's per-setting
embedder, Gorilla APIBench's retriever settings, and ACEBench's agent category and Chinese
setting.

Everything runs offline against local stubs (no model server, no Gemini API)."""

import asyncio
import hashlib
import json
import re
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import acebench as AB
from gbench.runners.eval_suites import bfcl_v4_agentic as AG
from gbench.runners.eval_suites import gorilla_apibench as G
from gbench.runners.eval_suites import t_eval as TE

DOCS = Path(__file__).resolve().parents[1] / "docs" / "evals"
#: The real key check, kept before any fixture replaces it.
_REAL_SEARCH_BACKEND = AG.search_backend
VENDOR = Path(AB.__file__).resolve().parents[2] / "third_party" / "_acebench_vendor"


def _blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# --------------------------------------------------------------------------- #
# bfcl_v4_agentic: every web search is recorded; a search no model served stops the suite
# --------------------------------------------------------------------------- #

KEY = "gemKEY-w2e-0123456789abcdef"


@pytest.fixture
def agentic(monkeypatch, tmp_path):
    """run_bfcl_v4_agentic with the harness stubbed; `seen["log"]` rows are written to the
    web search log during the stubbed generate, as the harness child would. With
    `seen["timeout"]` the first generate runs past its wall clock (a resume after it does not)."""
    seen = {"cli": [], "log": [], "env": None, "timeout": False, "envs": []}
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(tmp_path / "bfcl"))
    monkeypatch.delenv("GBENCH_BFCL_V4_AGENTIC_TEMPERATURE", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setattr(AG, "check_bfcl_prerequisites",
                        lambda url: (True, "", list(AG.AGENTIC_CATEGORIES)))
    monkeypatch.setattr(AG, "search_backend", lambda: (AG.SEARCH_BACKEND, ""))
    monkeypatch.setattr(AG, "_register_model", lambda m: m)

    def cli(args, env, timeout, model="", backend=None):
        seen["cli"].append(list(args))
        if args[0] == "generate":
            first = seen["env"] is None
            seen["env"] = dict(env)
            seen["envs"].append(dict(env))
            log = env.get(AG.SEARCH_LOG_ENV)
            if log:
                with open(log, "a", encoding="utf-8") as f:
                    for row in seen["log"]:
                        f.write(json.dumps(row) + "\n")
            if seen["timeout"] and first:
                raise subprocess.TimeoutExpired(cmd="bfcl generate", timeout=timeout)
        return 0, ""
    monkeypatch.setattr(AG, "_run_cli", cli)
    monkeypatch.setattr(AG, "_clear_stale_scores", lambda *a: 0)
    monkeypatch.setattr(AG, "_parse_scores", lambda root, m, cats: {
        c: {"correct": 1, "total": 2, "accuracy": 50.0} for c in cats})
    monkeypatch.setattr(AG, "_parse_score_traces", lambda *a: [])
    monkeypatch.setattr(AG, "_score_error_types", lambda *a: {})
    return seen


def _row(backend="gemini:gemini-3.7-flash", results=3, error=None):
    return {"backend": backend, "results": results, "error": error}


FAILED = _row("all-grounding-models-failed", 0, "search backend failed: HTTP Error 429")


def test_search_log_summary_counts_each_grounding_model():
    rows = [_row(), _row(), _row("gemini:gemini-3.6-flash"), _row(results=0), FAILED,
            _row("none", 0, "web search requires GEMINI_API_KEY")]
    s = AG.search_log_summary(rows)
    assert s["requests"] == 6 and s["failed"] == 2 and s["empty_results"] == 1
    assert s["by_backend"] == {"all-grounding-models-failed": 1, "gemini:gemini-3.6-flash": 1,
                               "gemini:gemini-3.7-flash": 3, "none": 1}
    assert s["failures"] == {
        "all-grounding-models-failed: search backend failed: HTTP Error 429": 1,
        "none: web search requires GEMINI_API_KEY": 1}


def test_a_search_no_model_served_leaves_web_search_infra_required(agentic):
    agentic["log"] = [_row(), FAILED]
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    web = res["settings"]["web_search"]
    assert web["status"] == "infra_required" and web["when"] == "during generation"
    assert re.search(r"for 1 of the 2 web searches.*HTTP Error 429", web["infra_required"])
    assert web["prerequisite"] == "GEMINI_API_KEY"
    # generate had finished: the memory entries are whole, only memory is evaluated
    assert [c[0] for c in agentic["cli"]] == ["generate", "evaluate"]
    ev = agentic["cli"][1]
    assert ev[ev.index("--test-category") + 1] == ",".join(AG.MEMORY_CATEGORIES)
    assert set(res["category_accuracy"]) == set(AG.MEMORY_CATEGORIES)


def test_the_search_log_is_reported_and_the_substitution_breaks_comparability(agentic):
    agentic["log"] = [_row(), _row("gemini:gemini-3.6-flash"), _row(results=0)]
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    assert res["status"] == "success" and "settings_infra_required" not in res
    assert res["leaderboard_comparable"] is False
    # the no-think run's 0.0 (gbench's rule) is named next to the search substitution
    assert res["leaderboard_comparable_reason"] == AG.SEARCH_SUBSTITUTION_REASON + "; temperature 0.0 set by default (BFCL generates at 0.001)"
    report = res["bfcl_report"]["web_search_calls"]
    assert report["requests"] == 3 and report["failed"] == 0 and report["empty_results"] == 1
    assert res["bfcl_report"]["search_substitution"]["leaderboard_comparable"] is False


def test_the_log_is_set_only_for_web_search_runs(agentic):
    AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    log = agentic["env"][AG.SEARCH_LOG_ENV]
    assert log.endswith(AG.SEARCH_LOG_NAME) and Path(log).parent == Path(
        agentic["env"]["BFCL_PROJECT_ROOT"])
    agentic["cli"].clear()
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1,
                                 eval_categories=",".join(AG.MEMORY_CATEGORIES))
    assert AG.SEARCH_LOG_ENV not in agentic["env"]
    assert res["bfcl_report"]["web_search_calls"] is None
    assert res["bfcl_report"]["search_substitution"] is None
    assert res["leaderboard_comparable_reason"] and \
        AG.SEARCH_SUBSTITUTION_REASON not in res["leaderboard_comparable_reason"]


def test_a_generate_timeout_with_a_failed_search_names_it(agentic):
    agentic["log"] = [_row(), FAILED]
    agentic["timeout"] = True
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    web = res["settings"]["web_search"]
    assert re.search(r"GBENCH_BFCL_TIMEOUT_S=.*for 1 of the 2 web searches", web["infra_required"])
    # the memory entries the stop left unfinished are generated by the harness' resume
    assert [c[0] for c in agentic["cli"]] == ["generate", "generate", "evaluate"]
    resume = agentic["cli"][1]
    assert resume[resume.index("--test-category") + 1] == ",".join(AG.MEMORY_CATEGORIES)
    assert agentic["envs"][0]["GEMINI_API_KEY"] == KEY
    assert "--allow-overwrite" not in resume and "GEMINI_API_KEY" not in agentic["envs"][1]
    assert res["harness_errors"] == [] and res["bfcl_report"]["memory_resumed"] is True


def test_a_generate_timeout_with_several_failed_searches_counts_them(agentic):
    agentic["timeout"] = True
    agentic["log"] = [FAILED] * 3
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    assert "for 3 of the 3 web searches" in res["settings"]["web_search"]["infra_required"]
    assert res["settings_infra_required"] == ["web_search"]


def test_a_generate_timeout_with_every_search_served_is_a_plain_timeout(agentic, monkeypatch):
    # A stopped generate no longer loses the run: the finished entries are scored and the rest
    # count as timed out (harness_errors); searches that were all served make it no
    # infra_required. Here no entry finished, so nothing measured the model: the score is
    # withheld (status failed, accuracy null), never reported as 0.
    agentic["timeout"] = True
    agentic["log"] = [_row()]
    monkeypatch.setattr(AG, "run_in_image", lambda *a, **k: {c: 4 for c in AG.AGENTIC_CATEGORIES})
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    assert res["status"] == "failed" and res["leaderboard_comparable"] is False
    assert res["accuracy"] is None and "every one of the 20 scored entries" in \
        res["accuracy_withheld_reason"]
    [err] = res["harness_errors"]
    assert "did not finish within GBENCH_BFCL_TIMEOUT_S" in err["error"]
    # the stubbed harness wrote no result file: nothing to evaluate, all 5 x 4 unfinished
    assert "INFRA REQUIRED" not in err["error"] and err["entries_unfinished"] == 20
    assert res["timed_out_requests"] == res["failed_requests"] == 20
    assert [c[0] for c in agentic["cli"]] == ["generate"]


def test_the_real_key_check_reads_the_key_list_first(agentic, monkeypatch):
    """GEMINI_API_KEYS wins over GEMINI_API_KEY (base.gemini_api_keys); the web_search generate
    gets the whole list and its first key, the live check sees the list."""
    checked = []
    monkeypatch.setattr(AG, "search_backend", _REAL_SEARCH_BACKEND)
    monkeypatch.setattr(AG, "gemini_key_live_valid", lambda key: checked.append(key) or (True, ""))
    monkeypatch.setenv("GEMINI_API_KEYS", "gemLIST-first-0123456789;gemLIST-second-0123456789")
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    assert checked == ["gemLIST-first-0123456789,gemLIST-second-0123456789"]
    env = agentic["envs"][0]
    assert env["GEMINI_API_KEYS"] == "gemLIST-first-0123456789,gemLIST-second-0123456789"
    assert env["GEMINI_API_KEY"] == "gemLIST-first-0123456789"
    assert res["settings"]["web_search"]["status"] == "success"
    assert "gemLIST" not in json.dumps(res)


def _fake_bfcl_web_search(monkeypatch):
    """Stand-ins for the two bfcl_eval modules the Gemini search is installed into, with
    upstream's shapes: WebSearchAPI (show_snippet, search_engine_query) and base_handler's
    execute_multi_turn_func_call."""
    class WebSearchAPI:
        def __init__(self):
            self.show_snippet = True

        def search_engine_query(self, keywords, max_results=10, region="wt-wt"):
            """upstream's docstring"""
            raise AssertionError("the harness' own search must not run")
    ws = types.ModuleType("bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search")
    ws.WebSearchAPI = WebSearchAPI
    bh = types.ModuleType("bfcl_eval.model_handler.base_handler")
    bh.execute_multi_turn_func_call = lambda calls, *a, **k: ([str(c) for c in calls], {"i": 1})
    fsc = types.ModuleType("bfcl_eval.eval_checker.multi_turn_eval.func_source_code")
    fsc.web_search = ws
    mh = types.ModuleType("bfcl_eval.model_handler")
    mh.base_handler = bh
    for name, mod in (("bfcl_eval.eval_checker.multi_turn_eval.func_source_code", fsc),
                      (ws.__name__, ws), ("bfcl_eval.model_handler", mh), (bh.__name__, bh)):
        monkeypatch.setitem(sys.modules, name, mod)
    return WebSearchAPI, bh


def test_the_search_tool_returns_bfcls_shape_and_logs_each_search(monkeypatch, tmp_path):
    from gbench.runners.eval_suites import search_tool
    WebSearchAPI, bh = _fake_bfcl_web_search(monkeypatch)
    replies = iter([
        ([{"title": "a", "url": "https://a", "snippet": "sa secret-key-1234"},
          {"title": "b", "url": "https://b", "snippet": "sb"}], "gemini:gemini-3.7-flash"),
        ([{"title": "error", "url": "", "snippet": "search backend failed: secret-key-1234"}],
         "all-grounding-models-failed")])
    monkeypatch.setattr(search_tool, "search_with_fallback",
                        lambda query, max_results=8: next(replies))
    monkeypatch.setenv("GEMINI_API_KEY", "secret-key-1234")
    log = tmp_path / "log.jsonl"
    AG._serve_web_search_with_gemini(str(log))
    first = WebSearchAPI().search_engine_query("x", max_results=1)
    assert first == [{"title": "a", "href": "https://a", "body": f"sa {base.REDACTED}"}]
    assert WebSearchAPI().search_engine_query("y") == {"error": AG.SEARCH_FAILED_TEXT}
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["backend"] for r in rows] == ["gemini:gemini-3.7-flash",
                                            "all-grounding-models-failed"]
    assert rows[0]["results"] == 1 and rows[1]["error"].startswith("search backend failed")
    assert "secret-key-1234" not in log.read_text()
    # every tool result the harness hands the model is masked too
    assert bh.execute_multi_turn_func_call(["k=secret-key-1234"]) == (
        [f"k={base.REDACTED}"], {"i": 1})
    assert WebSearchAPI.search_engine_query.__doc__ == "upstream's docstring"


def test_the_child_bootstrap_registers_the_run_project_root_and_installs_the_search(
        monkeypatch, tmp_path):
    """The bootstrap globs $BFCL_PROJECT_ROOT (default $GBENCH_CACHE_DIR/bfcl), never a fixed
    ~/.cache path, and installs the Gemini search with the log GBENCH_BFCL_SEARCH_LOG names."""
    root = Path(AG.default_project_root())
    assert root == Path(base.gbench_cache_dir()) / "bfcl"
    (root / "score" / "old-org_old-model").mkdir(parents=True)
    registered, installed, argv_seen = [], [], []
    monkeypatch.setattr(AG, "_register_model", lambda m: registered.append(m) or m)
    monkeypatch.setattr(AG, "_serve_web_search_with_gemini", lambda p: installed.append(p))
    fake_main = types.ModuleType("bfcl_eval.__main__")
    fake_main.cli = lambda: argv_seen.append(list(sys.argv))
    monkeypatch.setitem(sys.modules, "bfcl_eval.__main__", fake_main)
    monkeypatch.setattr(sys, "argv", ["pytest"])
    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(root))
    monkeypatch.setenv(AG.SEARCH_LOG_ENV, str(tmp_path / "s.jsonl"))
    exec(AG._CHILD_BOOTSTRAP.format(model="m", args=["evaluate"]), {})
    assert registered == ["m", "old-org/old-model"]
    assert installed == [str(tmp_path / "s.jsonl")]
    assert argv_seen == [["bfcl", "evaluate"]]
    assert "~/.cache" not in AG._CHILD_BOOTSTRAP


# --------------------------------------------------------------------------- #
# T-Eval: each setting's embedder, as test.py configures it
# --------------------------------------------------------------------------- #

@pytest.fixture
def recorded_embedders(monkeypatch):
    import numpy as np
    import sentence_transformers
    loaded = []

    class _Recorder:
        def __init__(self, name, device=None, revision=None, **k):
            loaded.append((name, revision))

        def encode(self, sents, *a, **k):
            return np.ones((len(sents), 4), dtype="float32")
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _Recorder)
    monkeypatch.setattr(TE, "_SHARED_ST", {})
    monkeypatch.setenv("GBENCH_T_EVAL_DEVICE", "cpu")
    return loaded


def _reason_trace():
    sample = {"template": "", "meta_data": {"response_format": "str"},
              "ground_truth": {"thought": "read the file", "name": "FileOperation.read",
                               "args": {}}}
    return {"gold_answer": json.dumps({"dimension": "reason_str", "sample": sample}),
            "response_text": "read the file"}


def test_t_eval_scores_each_setting_with_its_own_embedder(recorded_embedders):
    TE._score_file("reason_str", [_reason_trace()], language="zh")
    TE._score_file("reason_str", [_reason_trace()], language="en")
    TE._score_file("reason_str", [_reason_trace()], language="zh")        # cached, not reloaded
    assert recorded_embedders == [
        ("thenlper/gte-large-zh", "64c364e579de308104a9b2c170ca009502f4f545"),
        ("sentence-transformers/all-mpnet-base-v2", "e8c3b32edf5434bc2275fc9bab85f82640a19130")]


def test_t_eval_run_hands_each_file_its_setting(monkeypatch):
    seen = {}

    def load(dim, stem, limit=None):
        return [([{"role": "user", "content": stem}],
                 json.dumps({"dimension": dim, "sample": {}}), {"category": f"t_eval::{dim}"})]
    monkeypatch.setattr(TE, "_load_file_samples", load)

    def score(dim, traces, language="en"):
        seen.setdefault(language, []).append(dim)
        return {}
    monkeypatch.setattr(TE, "_score_file", score)

    async def reply(**kw):
        return base.Reply("ok", None, "stop", completion_tokens=3)
    with mock.patch.object(base, "_send_single_request", reply):
        r = TE.run_t_eval("m", "http://x", 1)
    assert seen == {"en": [d for d, _ in TE._FILES], "zh": [d for d, _ in TE._FILES]}
    assert r["embedding_models"]["zh"].startswith("thenlper/gte-large-zh@64c364e5")


def test_t_eval_an_embedder_that_cannot_load_is_infra_required(monkeypatch):
    import sentence_transformers

    def boom(name, **k):
        raise OSError("hub unreachable")
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", boom)
    monkeypatch.setattr(TE, "_SHARED_ST", {})
    monkeypatch.setenv("GBENCH_T_EVAL_DEVICE", "cpu")
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*t_eval.*thenlper/gte-large-zh"):
        TE._shared_st("zh")


# --------------------------------------------------------------------------- #
# Gorilla APIBench: 0-shot and the BM25 / GPT-Index / Oracle retriever settings
# --------------------------------------------------------------------------- #

def test_gorilla_pins_a_question_file_for_every_hub_and_setting():
    pairs = {(h, s): G.question_file(h, s) for h in G.HUBS for s in G.SETTINGS}
    assert len(pairs) == 12 and len({b for _, b in pairs.values()}) == 12
    for (hub, setting), (rel, blob) in pairs.items():
        assert rel == f"{hub}/questions_{hub}_{setting}.jsonl" and len(blob) == 40
    assert G.SETTINGS[0] == G.HEADLINE_SETTING == "0_shot"


def test_gorilla_loads_all_four_settings(monkeypatch, tmp_path):
    import huggingface_hub

    def fake_hf(repo, filename, **k):
        p = tmp_path / filename
        if filename.endswith("_eval.json"):
            p.write_text("\n".join(json.dumps({"api_data": {"domain": f"d{i}"}})
                                   for i in range(1000)))
        else:
            p.write_text(json.dumps({"api_call": "torch.hub.load(repo_or_dir='a', model='b')",
                                     "domain": "d0"}))
        return str(p)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf)

    def fake_questions(rel, blob):
        hub = rel.split("/")[0]
        p = tmp_path / rel.replace("/", "__")
        suffix = "" if rel.endswith("0_shot.jsonl") else f"\n Use this API documentation: {rel}"
        p.write_text("\n".join(json.dumps({"question_id": i + 1, "text": f"q{i}{suffix}"})
                               for i in range(G.HUB_SIZES[hub])))
        return str(p)
    monkeypatch.setattr(G, "_questions_file", fake_questions)
    monkeypatch.setattr(G, "HubDatabase", lambda hub, rows, parser=None: object())
    samples = G._load_gorilla_apibench_samples()
    assert len(samples) == 4 * sum(G.HUB_SIZES.values())
    cats = {m["category"] for _, _, m in samples}
    assert cats == {f"{h}/{s}" for h in G.HUBS for s in G.SETTINGS}
    oracle = next(s for s in samples if s[2]["category"] == "torchhub/oracle")
    assert json.loads(oracle[1])["setting"] == "oracle"
    assert "Use this API documentation" in oracle[0][1]["content"]
    assert oracle[0][0]["content"] == ("You are a helpful API writer who can write APIs based on "
                                       "requirements.")


def test_gorilla_reports_every_setting_and_heads_with_0_shot(monkeypatch):
    monkeypatch.setattr(G, "check_prereqs", lambda: None)
    monkeypatch.setattr(G, "_load_gorilla_apibench_samples", lambda limit=None: [])
    traces = []
    for setting, text in (("0_shot", "bad"), ("oracle", "ok"), ("bm25", "hal")):
        for hub in G.HUBS:
            traces.append({"gold_answer": json.dumps({"hub": hub, "setting": setting,
                                                      "question_id": 1, "domain": "d"}),
                           "response_text": text})
    monkeypatch.setattr(G, "run_eval_suite", lambda **k: {
        "sample_traces": traces, "accuracy": 0.0, "temperature": 0.0, "attempts": None,
        "decoding_penalties": None})
    verdict = {"ok": (True, False), "bad": (False, False), "hal": (False, True)}
    monkeypatch.setattr(G, "_score", lambda text, gold: verdict[text])
    r = G.run_gorilla_apibench("m", "http://x", 1)
    assert set(r["per_setting"]) == {"0_shot", "oracle", "bm25"}
    assert r["per_setting"]["oracle"]["mean_accuracy"] == 100.0
    assert r["per_setting"]["bm25"]["mean_hallucination"] == 100.0
    assert r["accuracy"] == 0.0 and r["per_hub"] == r["per_setting"]["0_shot"]["per_hub"]


# --------------------------------------------------------------------------- #
# ACEBench: vendored upstream files
# --------------------------------------------------------------------------- #

def test_acebench_vendored_files_rebuild_the_upstream_blobs():
    from gbench.third_party import _acebench_vendor as V
    for rel, (up_path, blob, shim) in V.UPSTREAM_BLOBS.items():
        text = (VENDOR / rel).read_bytes().decode("utf-8")
        header, body = text.split("\n", 3)[:3], text.split("\n", 3)[3]
        assert up_path in header[1], rel
        if shim:
            assert body.count(shim[1]) == 1, rel
            body = body.replace(shim[1], shim[0])
        assert _blob(body.encode("utf-8")) == blob, rel
    for rel, digest in V.PROMPT_SEGMENT_SHA256.items():
        text = (VENDOR / rel).read_text(encoding="utf-8")
        assert hashlib.sha256(text.split("\n", 5)[5].encode("utf-8")).hexdigest() == digest, rel


def test_acebench_english_prompts_are_the_vendored_ones_and_chinese_ones_exist():
    from gbench.third_party._acebench_vendor import prompt_zh
    row = {"function": [{"name": "Api"}], "time": "T", "profile": "P", "question": "q"}
    assert AB._system_prompt("normal_atom_bool", row, "zh") == \
        prompt_zh.SYSTEM_PROMPT_FOR_NORMAL_DATA_ZH.format(time="T", function=[{"name": "Api"}])
    assert AB._system_prompt("special_incomplete", row, "zh").startswith(
        prompt_zh.SYSTEM_PROMPT_FOR_SPECIAL_DATA_ZH[:20])
    assert AB._PROMPTS["zh"][3] == prompt_zh.USER_PROMPT_ZH
    assert set(AB.PINNED_BLOBS["zh"]) == set(AB.PINNED_BLOBS["en"]) == set(
        AB.NORMAL_CATEGORIES + AB.SPECIAL_CATEGORIES + AB.AGENT_CATEGORIES)


def test_acebench_chinese_files_are_fetched_from_data_zh(monkeypatch):
    urls = []

    class _Resp:
        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.data
    data = b'{"id": "normal_atom_bool_0"}\n'
    monkeypatch.setattr(AB.urllib.request, "urlopen",
                        lambda url, timeout=60: urls.append(url) or _Resp(data))
    rows = AB._pinned_file("data_normal_atom_bool.json", _blob(data), "zh")
    assert rows == [{"id": "normal_atom_bool_0"}]
    assert urls[0].endswith(f"{AB.ACEBENCH_COMMIT}/data_all/data_zh/data_normal_atom_bool.json")
    cached = Path(base.gbench_cache_dir()) / "acebench" / AB.ACEBENCH_COMMIT / "data_zh"
    assert (cached / "data_normal_atom_bool.json").read_bytes() == data


# --------------------------------------------------------------------------- #
# ACEBench agent: decoding, execution, checker (upstream quirks kept)
# --------------------------------------------------------------------------- #

def test_the_three_roles_decode_non_call_elements_as_upstream():
    assert AB.decode_function_list("[f(a=1), g(b='x')]", "fail") == ["f(a=1)", "g(b='x')"]
    with pytest.raises(TypeError):                         # CommonAgent: ast_parse -> False
        AB.decode_function_list("[f(a=1), 5]", "fail")
    assert AB.decode_function_list("[f(a=1), 5]", "skip") == ["f(a=1)"]
    with pytest.raises(ValueError):
        AB.decode_function_list("[f(a=1), 5]", "raise")
    with pytest.raises(AttributeError):                    # a dict argument reads as a nested call
        AB.decode_function_list("[f(a={'k': 'v'})]", "skip")
    assert AB.decode_function_list("[f(a=g(b=2), c=h())]", "skip") == ["f(a=g(b=2), c='h()')"]
    assert AB.decode_function_list("[f(a=2*3, b=-1)]", "skip") == ["f(a=6, b=-1)"]


def _env(classes=("BaseApi", "MessageApi"), config=None, language="en"):
    return AB._Environment(config or {"BaseApi": {"wifi": False, "logged_in": True}},
                           list(classes), language, "m", "0")


def test_a_shared_method_runs_on_every_instance_and_the_state_is_upstreams():
    env = _env()
    results, instances = AB.execute_agent_func_call(["turn_on_wifi()"], env, step=False)
    assert json.loads(results[0])["status"] is True
    assert instances["BaseApi"].wifi is True and instances["MessageApi"].wifi is True
    state = AB._json_roundtrip(AB._saved_state([instances]))
    assert state[0] == {"BaseApi": {"wifi": True, "logged_in": True}}
    assert set(state[1]["MessageApi"]["inbox"]) == {"1", "2", "3", "4", "5", "6"}


def test_an_unknown_method_reuses_the_last_result_or_fails_as_upstream():
    env = _env()
    res, _ = AB.execute_agent_func_call(["turn_on_wifi()", "no_such_api()"], env, step=True)
    assert res[0] == res[1]                                # the previous call's result, again
    res, _ = AB.execute_agent_func_call(["no_such_api()"], _env(), step=True)
    assert res[0].startswith("Error during execution: ") and "func_call_result" in res[0]
    res, classes = AB.execute_agent_func_call(["no_such_api()"], _env(), step=False)
    assert res == "Error during execution: {str(e)}" and classes == ["BaseApi", "MessageApi"]
    assert AB._parse_execution_results(res)[:5] == ["E", "r", "r", "o", "r"]


def test_the_execution_guard_refuses_what_upstream_would_eval():
    env = _env()
    env.instances()
    res, _ = AB.execute_agent_func_call(
        ["send_message(sender_name=json.dumps(obj=1), receiver_name='Frank', message='x')"],
        env, step=True)
    assert "gbench does not evaluate" in res[0] and env.refusals == ["json.dumps(obj=1)"]
    env = _env()
    res, _ = AB.execute_agent_func_call(
        ["send_message(sender_name=undefined_fn(obj=1), receiver_name='Frank', message='x')",
         "send_message(sender_name=inf, receiver_name='Frank', message='x')"], env, step=True)
    assert res == ["Error during execution: name 'undefined_fn' is not defined",
                   "Error during execution: name 'inf' is not defined"]
    assert env.refusals == []
    env = _env()
    assert env.instances() and env.call(f"{env.instance_name('MessageApi')}.turn_on_wifi()")[
        "status"] is True
    assert env._value(__import__("ast").parse("dict(a=1, b=[2])", mode="eval").body) == \
        {"a": 1, "b": [2]}


def test_the_agent_checker_keeps_upstreams_verdict_rules():
    gt = [{"BaseApi": {"wifi": True, "logged_in": True}},
          {"MessageApi": {"inbox": {"1": {"a": 1}}}}]
    ok = [{"BaseApi": {"wifi": True, "logged_in": True}},
          {"MessageApi": {"inbox": {"1": {"a": 1}, "9": {"extra": 1}}}}]   # extra keys ignored
    assert AB.agent_entry_valid(ok, gt) == (True, "")
    assert AB.agent_entry_valid(ok[:1], gt) == (False, "wrong number of class")
    bad = [{"BaseApi": {"wifi": False, "logged_in": True}}, ok[1]]
    assert AB.agent_entry_valid(bad, gt)[0] is False
    # a possible-answer class missing from the result reuses the previous verdict (here: valid)
    other = [{"BaseApi": {"wifi": True, "logged_in": True}}, {"Travel": {"users": {}}}]
    assert AB.agent_entry_valid(other, gt) == (True, "")


def test_agent_process_score_follows_agent_eval_process():
    calls = ["[a()]", " [x()] ", "[b()]", "[c()]"]
    assert AB.agent_process_score(calls, ["[a()]", "[b()]", "[d()]"]) == pytest.approx(2 / 3)
    assert AB.agent_process_score(calls, [["[d()]"], ["[a()]", "[c()]", "[e()]"]]) == 0.667
    assert AB.agent_process_score(["[b()]", "[a()]"], ["[a()]", "[b()]"]) == 0.5


def test_acebench_summary_is_the_leaderboards_overall():
    rows = [("normal_atom_bool", "normal_atom_bool_0", True, 0),
            ("special_irrelevant", "special_irrelevant_0", True, 0)]
    agent = {"agent_multi_turn": [(True, 1.0), (False, 0.5)], "agent_multi_step": [(False, 0.25)]}
    agg = AB.aggregate(rows, agent)
    assert agg["per_sub_category"]["agent_multi_turn"] == {
        "accuracy": 0.5, "process_accuracy": 0.75, "conversations": 2, "correct": 1, "total": 2}
    assert agg["agent_summary"] == 0.25
    assert agg["summary"] == round(0.2676 * agg["special_summary"] + 0.578 * agg["normal_summary"]
                                   + 0.1545 * 0.25, 3)
    cols = agg["leaderboard_columns"]
    assert set(cols) == {"atom", "single", "multi", "api", "pref", "summary", "special", "agent",
                         "overall"}
    assert cols["atom"] == round(1.0 / 6, 3) and cols["overall"] == agg["summary"]


# --------------------------------------------------------------------------- #
# ACEBench agent conversations, end to end with a scripted model and user
# --------------------------------------------------------------------------- #

_MT_ROW = {"id": "agent_multi_turn_7", "question": "You are Grace; send Frank 'hi'.",
           "initial_config": {"BaseApi": {"wifi": False, "logged_in": True}},
           "function": [{"name": "turn_on_wifi"}, {"name": "send_message"}],
           "involved_classes": ["BaseApi", "MessageApi"]}


def _script(replies):
    it = iter(replies)
    return lambda *a: next(it)


def test_a_multi_turn_conversation_runs_as_upstreams_scene(monkeypatch):
    prompts, user_turns = [], []

    class _Model(AB._AgentModel):
        agent = _script(["[turn_on_wifi()]",
                         "[send_message(sender_name='Grace', receiver_name='Frank', "
                         "message='hi')]", "Done. finish conversation"])

        async def inference(self, system_prompt, user_prompt):
            prompts.append((system_prompt, user_prompt))
            self.requests += 1
            return self.agent()

    async def user(messages):
        user_turns.append([dict(m) for m in messages])
        return "I want to message Frank.", "gemini-x"
    monkeypatch.setattr(AB, "_AgentModel", _Model)
    monkeypatch.setattr(AB, "_user_sim_generate", user)
    item = {"language": "en", "category": "agent_multi_turn", "key": "agent_multi_turn",
            "row": _MT_ROW, "answer": {"ground_truth": [{"BaseApi": {"wifi": True, "logged_in": True}}],
                                       "mile_stone": ["[turn_on_wifi()]"]}}
    trace = asyncio.run(AB._run_conversation({"model_name": "m"}, item))
    assert trace["status"] == "OK" and trace["model_requests"] == 3
    assert [m["sender"] for m in trace["agent_dialogue"]] == [
        "user", "agent", "execution", "agent", "execution", "agent"]
    assert trace["agent_process"] == ["[turn_on_wifi()]", "[send_message(sender_name='Grace', "
                                      "receiver_name='Frank', message='hi')]"]
    # the scenario's own answer: the inbox starts full (6 of 6), so the message is refused
    sent = trace["agent_dialogue"][4]["message"][0]
    assert sent["status"] is False and "capacity is full" in sent["message"]
    assert len(trace["agent_result"][1]["MessageApi"]["inbox"]) == 6
    # the scene's history as the agent prompt, and the agent + base prompts as the system message
    assert prompts[0][1].endswith("Conversation history 1..t:\nuser:I want to message Frank.\n")
    assert "execution:[{'status': True" in prompts[1][1]
    from gbench.third_party._acebench_vendor import multi_turn_prompts, prompt_en
    assert prompts[0][0] == (multi_turn_prompts.MULTI_TURN_AGENT_PROMPT_SYSTEM_EN
                             + prompt_en.BASE_PROMPT_EN)
    # APIUSER's opening; the user simulator was asked once (the reply routed to it ends the run)
    assert user_turns[0][1] == {"role": "user", "content": "Is there anything you need help with today?"}
    assert trace["user_sim_requests"] == 1 and trace["user_sim_models"] == {"gemini-x": 1}
    assert trace["is_correct"] is False                      # ground truth lacks MessageApi
    assert trace["agent_error_type"] == "wrong number of class"


def test_the_user_simulator_turns_are_mapped_for_gemini(monkeypatch):
    seen = {}

    async def cascade(contents, config=None):
        seen["contents"], seen["config"] = contents, config
        return "ok", "gemini-y"
    monkeypatch.setattr(AB, "_user_sim_cascade", cascade)
    msgs = [{"role": "system", "content": "INSTR"}, {"role": "user", "content": "open"},
            {"role": "system", "content": "sim said"}, {"role": "user", "content": "agent said"}]
    assert asyncio.run(AB._user_sim_generate(msgs)) == ("ok", "gemini-y")
    assert [c["role"] for c in seen["contents"]] == ["user", "model", "user"]
    assert seen["contents"][1]["parts"][0]["text"] == "sim said"
    cfg = seen["config"]
    if cfg is not None:     # APIUSER's own sampling: temperature 0.001, top_p 1, max_tokens 1000
        assert cfg.system_instruction == "INSTR" and cfg.temperature == 0.001 and cfg.top_p == 1.0
        assert cfg.max_output_tokens == 1000 + 8192   # plus the Gemini thinking allowance


def test_a_multi_step_conversation_answers_non_calls_as_upstream(monkeypatch):
    class _Model(AB._AgentModel):
        agent = _script(["Which user?", "[turn_on_wifi()]", "finish conversation"])

        async def inference(self, system_prompt, user_prompt):
            self.requests += 1
            return self.agent()
    monkeypatch.setattr(AB, "_AgentModel", _Model)
    row = dict(_MT_ROW, id="agent_multi_step_3", involved_classes=["BaseApi"])
    item = {"language": "zh", "category": "agent_multi_step", "key": "agent_multi_step_zh",
            "row": row, "answer": {"ground_truth": [{"BaseApi": {"wifi": True, "logged_in": True}}],
                                   "mile_stone": [["[turn_on_wifi()]"], ["[login_device()]"]]}}
    trace = asyncio.run(AB._run_conversation({"model_name": "m"}, item))
    assert trace["status"] == "OK"
    assert trace["agent_dialogue"][2]["message"] == AB._STEP_NOT_A_CALL
    # "finish conversation" ends the loop before the execution role sees it
    assert trace["agent_process"] == ["Which user?", "[turn_on_wifi()]"]
    assert trace["is_correct"] is True and trace["process_accuracy"] == 1.0


# --------------------------------------------------------------------------- #
# ACEBench run: both languages, the agent part merged, the user simulator required
# --------------------------------------------------------------------------- #

def _agent_item(language, category, idx=0):
    row = dict(_MT_ROW, id=f"{category}_{idx}")
    return {"language": language, "category": category,
            "key": AB._category_key(category, language), "row": row,
            "answer": {"ground_truth": [], "mile_stone": ["[turn_on_wifi()]"]}}


@pytest.fixture
def ace_run(monkeypatch):
    seen = {"agent_runs": []}

    def samples(language="en", **k):
        sub = "special_irrelevant"
        gold = json.dumps({"id": f"{sub}_0", "sub": sub, "language": language,
                           "ground_truth": "x", "functions": [], "question": "q"})
        return [([{"role": "user", "content": language}], gold,
                 {"category": AB._category_key(sub, language)})]
    monkeypatch.setattr(AB, "_load_acebench_samples", samples)
    monkeypatch.setattr(AB, "_load_agent_items", lambda language="en": [
        _agent_item(language, "agent_multi_step"), _agent_item(language, "agent_multi_turn")])
    monkeypatch.setattr(AB, "check_user_simulator", lambda: seen.setdefault("checked", True))
    seen["statuses"] = {}

    async def conversation(shared, item):
        seen["agent_runs"].append((item["key"], shared["temperature"], shared["payload"]))
        status = seen["statuses"].get(item["key"], "OK")
        t = {"category": item["key"], "acebench_language": item["language"],
             "question_id": item["row"]["id"], "attempt_idx": 0, "status": status,
             "gold_answer": json.dumps({"sub": item["category"]}), "execution_guard_refusals": [],
             "model_requests": 2, "truncated_replies": 0, "user_sim_requests": 1,
             "user_sim_models": {"g": 1}, "is_correct": status == "OK",
             "process_accuracy": 1.0 if status == "OK" else None,
             "response_text": "done" if status == "OK" else None}
        if status == "USER_SIM_OUTAGE":
            t["judge_grade"], t["response_text"] = "JUDGE_OUTAGE", ""
        if status == "GUARD":
            t.update(status="OK", execution_guard_refusals=["json.dumps(obj=1)"])
        return t
    monkeypatch.setattr(AB, "_run_conversation", conversation)

    async def reply(**kw):
        return base.Reply("Due to the limitations of the function, I cannot solve this problem.",
                          None, "stop", completion_tokens=5)
    monkeypatch.setattr(base, "_send_single_request", reply)
    return seen


def test_acebench_runs_both_languages_with_the_agent_category(ace_run):
    r = AB.run_acebench("m", "http://x", 2)
    assert ace_run["checked"] is True
    assert sorted(k for k, _t, _p in ace_run["agent_runs"]) == [
        "agent_multi_step", "agent_multi_step_zh", "agent_multi_turn", "agent_multi_turn_zh"]
    assert all(t == 0.0 and p.get("top_p") == 0.9 for _k, t, p in ace_run["agent_runs"])
    ca = r["category_accuracy"]
    assert ca["agent_multi_turn"] == {"correct": 1, "total": 1, "accuracy": 100.0}
    assert ca["special_irrelevant_zh"]["accuracy"] == 100.0
    en = r["per_language"]["en"]
    expected = round(round(0.2676 * round(1 / 3, 3) + 0.1545 * 1.0, 3) * 100, 2)
    assert r["accuracy"] == en["overall"] == expected and r["overall_zh"] == expected
    assert en["leaderboard_columns"]["agent"] == 100.0
    assert r["total_questions"] == 6 and r["correct_answers"] == 6
    assert r["user_simulator"]["replaced_by"] == "gbench Gemini judge cascade"
    assert r["leaderboard_comparable"] is True, r["leaderboard_comparable_reason"]


def test_acebench_guard_refusals_and_user_sim_outages_are_flagged(ace_run):
    ace_run["statuses"] = {"agent_multi_turn": "USER_SIM_OUTAGE", "agent_multi_step_zh": "GUARD"}
    r = AB.run_acebench("m", "http://x", 1)
    assert r["judge_outages"] == 1 and r["status"] == "completed_with_errors"
    assert "agent_multi_turn" not in r["category_accuracy"]          # left out, not scored 0
    assert r["acebench_report"]["execution_guard_refusals"] == 1
    reason = r["leaderboard_comparable_reason"]
    assert r["leaderboard_comparable"] is False
    assert "lost their user simulator" in reason and "execution guard" in reason


def test_acebench_needs_the_user_simulator_key_before_generating(monkeypatch):
    monkeypatch.setattr(AB, "_load_acebench_samples", lambda **k: [])
    monkeypatch.setattr(AB, "_load_agent_items",
                        lambda language="en": [_agent_item(language, "agent_multi_turn")])
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    called = []
    monkeypatch.setattr(AB, "run_eval_suite", lambda **k: called.append(k))
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*acebench.*user simulator.*"
                                           r"GEMINI_API_KEY"):
        AB.run_acebench("m", "http://x", 1)
    assert called == []


def test_acebench_eval_limit_splits_over_rows_and_conversations(ace_run, monkeypatch):
    rows = []

    def many(language="en", **k):
        out = []
        for i in range(90):
            gold = json.dumps({"id": f"normal_atom_bool_{i}", "sub": "normal_atom_bool",
                               "language": language, "ground_truth": {}, "functions": [],
                               "question": "q"})
            out.append(([{"role": "user", "content": "x"}], gold,
                        {"category": AB._category_key("normal_atom_bool", language)}))
        rows.append(language)
        return out
    monkeypatch.setattr(AB, "_load_acebench_samples", many)
    monkeypatch.setattr(AB, "_load_agent_items", lambda language="en": [
        _agent_item(language, c, i) for c in AB.AGENT_CATEGORIES for i in range(5)])
    r = AB.run_acebench("m", "http://x", 1, limit=20)
    # --eval-limit 20 is 20 samples for the suite, dealt over the sub-categories of both parts
    # (one per sub-category before any gets two): the two single-turn and the four agent ones.
    single_q, agent_q = AB._part_quotas(
        20, {AB._category_key("normal_atom_bool", lang): 90 for lang in ("en", "zh")},
        {AB._category_key(c, lang): 5 for c in AB.AGENT_CATEGORIES for lang in ("en", "zh")})
    assert single_q + agent_q == 20 and single_q >= 2 and agent_q >= 4
    assert len(ace_run["agent_runs"]) == agent_q and r["total_questions"] == 20
    assert r["eval_limit_plan"] == {"limit": 20, "samples": 20, "attempts_per_sample": 1,
                                    "per_setting": {"single_turn": single_q, "agent": agent_q},
                                    "generations": 20}
    assert "--eval-limit 20" in r["leaderboard_comparable_reason"]


@pytest.mark.parametrize("limit", [1, 2, 3, 5, 7])
def test_acebench_eval_limit_is_n_samples_for_the_whole_suite(limit):
    """A part none of whose samples is drawn gets 0 (not run), never 'at least one each'."""
    singles = {AB._category_key(f"normal_{i}", lang): 40 for i in range(12) for lang in ("en", "zh")}
    agents = {AB._category_key(c, lang): 5 for c in AB.AGENT_CATEGORIES for lang in ("en", "zh")}
    s, a = AB._part_quotas(limit, singles, agents)
    assert s + a == limit
    assert AB._part_quotas(None, singles, agents) == (None, None)
    assert AB._part_quotas(5, singles, {}) == (5, 0)


# --------------------------------------------------------------------------- #
# docs match the code
# --------------------------------------------------------------------------- #

def test_docs_describe_the_new_behaviour():
    ace = (DOCS / "acebench.md").read_text(encoding="utf-8")
    for needle in ("agent_multi_turn", "agent_multi_step", "data_zh", "overall_zh",
                   "GEMINI_API_KEY", "execution_guard_refusals", "user_simulator", "0.1545",
                   "leaderboard_columns", "--max-output-tokens", "_acebench_vendor"):
        assert needle in ace, needle
    teval = (DOCS / "t_eval.md").read_text(encoding="utf-8")
    for needle in ("thenlper/gte-large-zh", "64c364e579de308104a9b2c170ca009502f4f545",
                   "e8c3b32edf5434bc2275fc9bab85f82640a19130", "16 data files",
                   "embedding_models"):
        assert needle in teval, needle
    assert "same `all-mpnet-base-v2` embedder" not in teval
    gor = (DOCS / "gorilla_apibench.md").read_text(encoding="utf-8")
    for needle in ("bm25", "gpt_index", "oracle", "per_setting", "RETRIEVER_QUESTION_BLOBS",
                   "7100"):
        assert needle in gor, needle
    ag = (DOCS / "bfcl_v4_agentic.md").read_text(encoding="utf-8")
    for needle in ("web_search_requests.jsonl", AG.SEARCH_FAILED_TEXT, "GBENCH_BFCL_SEARCH_LOG",
                   AG.SEARCH_SUBSTITUTION_REASON, "GBENCH_BFCL_TIMEOUT_S"):
        assert needle in " ".join(ag.split()), needle
    assert "SERPAPI_API_KEY" not in ag
    assert "late searches may have failed" not in ag
    for text in (ace, teval, gor, ag, (DOCS / "bfcl.md").read_text(encoding="utf-8")):
        assert text.isascii()
