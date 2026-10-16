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

"""Deterministic unit tests for the wildclawbench canonical harness (no Docker / model / real judge)."""

import json
import os
import sys

import pytest

from gbench.runners.eval_suites import wildclawbench as W

_DOCKER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gbench", "docker")
sys.path.insert(0, _DOCKER_DIR)
import wildclawbench_run as RUN            # noqa: E402
import wildclawbench_cascade_judge as J    # noqa: E402


# --------------------------------------------------------------------------- #
# Harness-side headline scoring.
# --------------------------------------------------------------------------- #
def test_compute_score_valid_and_none():
    s = W.compute_wildclawbench_score({"global_mean": 0.42, "multimodal_mean": 0.3,
                                       "pure_text_mean": 0.5, "weighted_overall": 0.4,
                                       "avg_time_min": 6.4, "avg_cost_usd": 0.0})
    assert s["global_mean"] == pytest.approx(0.42)
    assert s["multimodal_mean"] == pytest.approx(0.3)
    assert W.compute_wildclawbench_score({"n_tasks": 3}) is None      # no global_mean
    assert W.compute_wildclawbench_score({"global_mean": None}) is None


# --------------------------------------------------------------------------- #
# The model URL the orchestrator's proxy relays to. The proxy runs on the host network whatever
# the task network is, so a model on this host keeps its 127.0.0.1/localhost address.
# --------------------------------------------------------------------------- #
def test_task_endpoint_rewrites_localhost(monkeypatch):
    # (name kept; the behavior changed) a local model URL is no longer rewritten
    monkeypatch.delenv("GBENCH_WILDCLAWBENCH_TASK_ENDPOINT", raising=False)
    monkeypatch.delenv("GBENCH_WILDCLAWBENCH_TASK_HOST", raising=False)
    assert W._model_relay_target("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"
    assert W._model_relay_target("http://localhost:8000") == "http://localhost:8000/v1"


def test_task_endpoint_custom_host_and_override(monkeypatch):
    monkeypatch.delenv("GBENCH_WILDCLAWBENCH_TASK_ENDPOINT", raising=False)
    monkeypatch.setenv("GBENCH_WILDCLAWBENCH_TASK_HOST", "host.docker.internal")
    # the task host is where the task containers reach the proxy; it never moves the model URL
    assert W._model_relay_target("http://127.0.0.1:9000/v1") == "http://127.0.0.1:9000/v1"
    monkeypatch.setenv("GBENCH_WILDCLAWBENCH_TASK_ENDPOINT", "http://gw:1/v1")
    assert W._model_relay_target("http://127.0.0.1:8000/v1") == "http://gw:1/v1"


def test_task_endpoint_remote_host_unchanged(monkeypatch):
    monkeypatch.delenv("GBENCH_WILDCLAWBENCH_TASK_ENDPOINT", raising=False)
    assert W._model_relay_target("http://10.0.0.5:8000/v1") == "http://10.0.0.5:8000/v1"
    assert W._model_relay_target("http://10.0.0.5:8000") == "http://10.0.0.5:8000/v1"


# --------------------------------------------------------------------------- #
# Prerequisite gate (raises infra_required, never skips).
# --------------------------------------------------------------------------- #
def test_gate_missing_host_dir(monkeypatch):
    monkeypatch.setattr(W.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(W.subprocess, "run",
                        lambda *a, **k: type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    monkeypatch.setattr(W, "_docker_image_exists", lambda _i: True)
    monkeypatch.delenv("GBENCH_WILDCLAWBENCH_HOST_DIR", raising=False)
    ok, reason = W.check_wildclawbench_prerequisites()
    assert not ok and "GBENCH_WILDCLAWBENCH_HOST_DIR" in reason


def test_gate_missing_docker():
    # No docker CLI on PATH under a scrubbed env -> gate fails with a build hint (never skips).
    import unittest.mock as mock
    with mock.patch.object(W.shutil, "which", return_value=None):
        ok, reason = W.check_wildclawbench_prerequisites()
    assert not ok and "docker CLI not found" in reason


def test_gate_requires_gemini_key_not_brave(tmp_path, monkeypatch):
    # Everything present except the key: the Gemini key is required (judge cascade + the OpenClaw
    # web_search, both on the Gemini API); no Brave key is required or read.
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "run_batch.py").write_text("x")
    (tmp_path / "workspace").mkdir()
    monkeypatch.setattr(W.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(W.subprocess, "run",
                        lambda *a, **k: type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    monkeypatch.setattr(W, "_docker_image_exists", lambda _i: True)
    monkeypatch.setenv("GBENCH_WILDCLAWBENCH_HOST_DIR", str(tmp_path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    ok, reason = W.check_wildclawbench_prerequisites()
    assert not ok and "GEMINI_API_KEY" in reason
    # the key present (no BRAVE_API_KEY at all) -> reaches live validation; stub the ping offline
    monkeypatch.setattr(W, "_gemini_key_valid", lambda _k: (True, ""))
    monkeypatch.setattr(W, "_image_accounts_harness_errors", lambda _i: True)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    ok, reason = W.check_wildclawbench_prerequisites()
    assert ok and reason == ""


def _gate_env_ready(tmp_path, monkeypatch):
    (tmp_path / "eval").mkdir()
    (tmp_path / "eval" / "run_batch.py").write_text("x")
    (tmp_path / "workspace").mkdir()
    monkeypatch.setattr(W.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(W.subprocess, "run",
                        lambda *a, **k: type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})())
    monkeypatch.setattr(W, "_docker_image_exists", lambda _i: True)
    monkeypatch.setattr(W, "_image_accounts_harness_errors", lambda _i: True)
    monkeypatch.setenv("GBENCH_WILDCLAWBENCH_HOST_DIR", str(tmp_path))
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION", raising=False)


def test_gate_rejects_invalid_gemini_key(tmp_path, monkeypatch):
    _gate_env_ready(tmp_path, monkeypatch)
    monkeypatch.setattr(W, "_gemini_key_valid", lambda _k: (False, "HTTP 401"))
    ok, reason = W.check_wildclawbench_prerequisites()
    assert not ok and "GEMINI_API_KEY) was rejected" in reason


def test_gate_has_no_brave_check(tmp_path, monkeypatch):
    # The Brave key, its live ping and every Brave path are gone: a set BRAVE_API_KEY is ignored.
    _gate_env_ready(tmp_path, monkeypatch)
    monkeypatch.setenv("BRAVE_API_KEY", "definitely-not-read")
    monkeypatch.setattr(W, "_gemini_key_valid", lambda _k: (True, ""))
    assert not hasattr(W, "_brave_key_valid")
    ok, reason = W.check_wildclawbench_prerequisites()
    assert ok and reason == ""


def test_gate_skip_key_validation_env(tmp_path, monkeypatch):
    _gate_env_ready(tmp_path, monkeypatch)
    monkeypatch.setenv("GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION", "1")

    def _boom(_k):
        raise AssertionError("validation must be skipped")
    monkeypatch.setattr(W, "_gemini_key_valid", _boom)
    ok, reason = W.check_wildclawbench_prerequisites()
    assert ok and reason == ""


def test_key_validators_classify_responses(monkeypatch):
    import urllib.error
    import urllib.request

    def _mk_urlopen(status=None, http_code=None, body=b"", exc=None):
        def _open(req, timeout=0):
            if exc is not None:
                raise exc
            if http_code is not None:
                raise urllib.error.HTTPError(req.full_url, http_code, "err", {}, __import__("io").BytesIO(body))
            return type("R", (), {"status": status, "__enter__": lambda s: s,
                                  "__exit__": lambda s, *a: False})()
        return _open

    # 200 -> valid
    monkeypatch.setattr(urllib.request, "urlopen", _mk_urlopen(status=200))
    assert W._gemini_key_valid("k")[0] is True
    # 401 -> rejected
    monkeypatch.setattr(urllib.request, "urlopen", _mk_urlopen(http_code=401))
    assert W._gemini_key_valid("k")[0] is False
    # Gemini 400 API_KEY_INVALID -> rejected
    monkeypatch.setattr(urllib.request, "urlopen",
                        _mk_urlopen(http_code=400, body=b'{"error":{"status":"API_KEY_INVALID"}}'))
    assert W._gemini_key_valid("k")[0] is False
    # network error -> inconclusive -> valid (never block a good key on infra flakiness)
    monkeypatch.setattr(urllib.request, "urlopen", _mk_urlopen(exc=OSError("timeout")))
    assert W._gemini_key_valid("k")[0] is True
    # Gemini 500 -> inconclusive -> valid
    monkeypatch.setattr(urllib.request, "urlopen", _mk_urlopen(http_code=500))
    assert W._gemini_key_valid("k")[0] is True


# --------------------------------------------------------------------------- #
# Launcher: per-task scoring + judge-fallback detection.
# --------------------------------------------------------------------------- #
def test_score_of_prefers_overall_then_mean():
    assert RUN._score_of({"overall_score": 0.8, "a": 0.2}) == pytest.approx(0.8)
    assert RUN._score_of({"a": 0.4, "b": 0.6}) == pytest.approx(0.5)     # mean when no overall_score
    assert RUN._score_of({"error": "x", "reason": "y"}) is None          # no numeric metrics


def test_judge_fell_back_detection():
    # prefixed + regex
    assert RUN._judge_fell_back({"recognized_fact_conflict_judge_method": "regex_fallback"}) is True
    # prefixed judge_error
    assert RUN._judge_fell_back({"llm_judge_error": "boom"}) is True
    # BARE judge_error (04_Search_Retrieval task_1 etc.)
    assert RUN._judge_fell_back({"judge_error": "outage"}) is True
    # BARE judge_method = rule_fallback (06_Safety task_3/task_6)
    assert RUN._judge_fell_back({"judge_method": "rule_fallback"}) is True
    # prefixed keyword_fallback (02_Code task_8/9/12)
    assert RUN._judge_fell_back({"desc_judge_method": "keyword_fallback"}) is True
    # judge_method == failed (02_Code task_10/11)
    assert RUN._judge_fell_back({"judge_method": "failed"}) is True
    # success markers are NOT fallbacks
    assert RUN._judge_fell_back({"recognized_fact_conflict_judge_method": "llm",
                                 "overall_score": 1.0}) is False
    assert RUN._judge_fell_back({"image_judge_method": "vlm", "overall_score": 0.8}) is False


def test_write_models_config(tmp_path, monkeypatch):
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_MODEL_BASE_URL", "http://172.17.0.1:8000")
    monkeypatch.setenv("WILDCLAW_MODEL_RELAY_URL", "http://127.0.0.1:18795/model/v1")
    monkeypatch.setenv("GBENCH_MODEL_NAME", "google/gemma-4-E4B-it")
    monkeypatch.setenv("GBENCH_MODEL_API_KEY", "sk-dummy")
    path, model_arg = RUN._write_models_config()
    cfg = json.loads(open(path, encoding="utf-8").read())
    prov = cfg["providers"]["gbench-model"]
    # OpenClaw calls the proxy's model relay with a placeholder; the proxy adds the real key.
    assert prov["baseUrl"] == "http://127.0.0.1:18795/model/v1"
    assert prov["apiKey"] == "gbench-model-relay"
    assert prov["api"] == "openai-completions"
    assert prov["models"][0]["id"] == "google/gemma-4-E4B-it"
    assert model_arg == "gbench-model/google/gemma-4-E4B-it"


def _make_harness_with_tasks(tmp_path, modalities):
    """Create a minimal tasks/ tree so RUN._modality_map() can resolve id -> modality."""
    for tid, mod in modalities.items():
        cat = tid.split("_task_")[0]
        d = tmp_path / "tasks" / cat
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{tid}.md").write_text(
            f"---\nid: {tid}\nname: X\ncategory: {cat}\ntimeout_seconds: 60\nmodality: {mod}\n---\n## Prompt\nx\n",
            encoding="utf-8")
    return str(tmp_path)


def _write_run(out_root, category, task_id, score, usage):
    d = os.path.join(out_root, "openclaw", category, task_id, "m_20260101_0000_abc123")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "score.json"), "w", encoding="utf-8") as f:
        json.dump(score, f)
    with open(os.path.join(d, "usage.json"), "w", encoding="utf-8") as f:
        json.dump(usage, f)


def test_collect_and_summarize(tmp_path, monkeypatch):
    # 4 SELECTED tasks: scored PT, scored+fallback MM, a PT whose score.json holds no numeric
    # metric, and a MISSING PT that never ran (no score.json, no run record). The judge fallback,
    # the score without a metric and the task that never ran are harness errors: counted, listed
    # per task, and left out of the headline (never scored as 0).
    harness = _make_harness_with_tasks(tmp_path / "h", {
        "01_Productivity_Flow_task_1": "pure-text",
        "05_Creative_Synthesis_task_2": "multimodal",
        "06_Safety_Alignment_task_3": "pure-text",
        "06_Safety_Alignment_task_4": "pure-text",   # selected but never scored (missing)
    })
    monkeypatch.setattr(RUN, "HARNESS_DIR", harness)
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path / "wd"))  # no launcher/proxy records
    monkeypatch.delenv("WILDCLAW_TASK", raising=False)
    monkeypatch.delenv("WILDCLAW_CATEGORIES", raising=False)   # -> all categories -> all 4 selected
    out_root = str(tmp_path / "out")
    _write_run(out_root, "01_Productivity_Flow", "01_Productivity_Flow_task_1",
               {"overall_score": 1.0}, {"elapsed_time": 120.0, "cost_usd": 0.0})
    _write_run(out_root, "05_Creative_Synthesis", "05_Creative_Synthesis_task_2",
               {"overall_score": 0.5, "desc_judge_method": "keyword_fallback"},
               {"elapsed_time": 240.0, "cost_usd": 0.0})
    _write_run(out_root, "06_Safety_Alignment", "06_Safety_Alignment_task_3",
               {"error": "grade script failed"}, {"elapsed_time": 60.0})
    # task_4: no run dir at all -> missing

    index = RUN._task_index()
    summary = RUN._summarize(out_root, index)
    assert summary["n_tasks"] == 4                 # SELECTED tasks, not files found
    assert summary["n_scored"] == 1                # only task_1 is in the headline
    assert summary["n_headline_tasks"] == 1
    assert summary["n_missing"] == 1               # task_4
    assert summary["n_grading_errors"] == 1        # task_3
    assert summary["n_harness_errors"] == 3
    assert summary["harness_error_kinds"] == {"no_score": 1, "not_run": 1, "judge_fallback": 1}
    kinds = {r["task_id"]: r["harness_errors"] for r in summary["per_task"]}
    assert kinds == {"01_Productivity_Flow_task_1": [],
                     "05_Creative_Synthesis_task_2": ["judge_fallback"],
                     "06_Safety_Alignment_task_3": ["no_score"],
                     "06_Safety_Alignment_task_4": ["not_run"]}
    # headline over the tasks without a harness error: task_1 only
    assert summary["global_mean"] == pytest.approx(1.0)
    assert summary["multimodal_mean"] is None                             # the MM task is out
    assert summary["pure_text_mean"] == pytest.approx(1.0)
    assert summary["weighted_overall"] is None
    assert summary["n_multimodal"] == 1 and summary["n_pure_text"] == 3
    assert summary["judge_fallback_tasks"] == 1                           # keyword_fallback detected


def test_summarize_single_task_selection(tmp_path, monkeypatch):
    harness = _make_harness_with_tasks(tmp_path / "h", {
        "06_Safety_Alignment_task_7_skill_injection": "pure-text",
        "01_Productivity_Flow_task_1": "pure-text",
    })
    monkeypatch.setattr(RUN, "HARNESS_DIR", harness)
    monkeypatch.delenv("WILDCLAW_CATEGORIES", raising=False)
    monkeypatch.setenv("WILDCLAW_TASK",
                       os.path.join(harness, "tasks", "06_Safety_Alignment",
                                    "06_Safety_Alignment_task_7_skill_injection.md"))
    out_root = str(tmp_path / "out")
    _write_run(out_root, "06_Safety_Alignment", "06_Safety_Alignment_task_7_skill_injection",
               {"overall_score": 0.0}, {"elapsed_time": 3.5, "cost_usd": 0.0})
    summary = RUN._summarize(out_root, RUN._task_index())
    assert summary["n_tasks"] == 1                 # only the single selected task, not both
    assert summary["global_mean"] == pytest.approx(0.0)


def test_collect_newest_run_wins(tmp_path, monkeypatch):
    harness = _make_harness_with_tasks(tmp_path / "h", {"01_Productivity_Flow_task_1": "pure-text"})
    monkeypatch.setattr(RUN, "HARNESS_DIR", harness)
    out_root = str(tmp_path / "out")
    base = os.path.join(out_root, "openclaw", "01_Productivity_Flow", "01_Productivity_Flow_task_1")
    old = os.path.join(base, "m_20260101_0000_old000")
    new = os.path.join(base, "m_20260101_0100_new000")
    for d, score in ((old, 0.1), (new, 0.9)):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "score.json"), "w") as f:
            json.dump({"overall_score": score}, f)
    os.utime(os.path.join(new, "score.json"), (2e9, 2e9))   # make 'new' clearly newer
    os.utime(os.path.join(old, "score.json"), (1e9, 1e9))
    rows = RUN._collect(out_root, RUN._task_index())
    assert len(rows) == 1 and rows["01_Productivity_Flow_task_1"]["overall_score"] == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# Gemini proxy (judge): config, model override, backoff, outage.
# --------------------------------------------------------------------------- #
def test_cascade_config(monkeypatch):
    monkeypatch.delenv("GBENCH_JUDGE_MODELS", raising=False)
    monkeypatch.delenv("GBENCH_JUDGE_MODEL", raising=False)
    assert J._cascade() == J._DEFAULT_JUDGE_CASCADE
    monkeypatch.setenv("GBENCH_JUDGE_MODEL", "gemini-x")
    assert J._cascade() == ["gemini-x"]
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "a, b ,c")
    assert J._cascade() == ["a", "b", "c"]                    # csv wins over single
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", "5")
    assert J._rounds() == 5


class _FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode("utf-8")
    def read(self):
        return self._b
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def test_cascade_overrides_model_and_temperature(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-primary,gemini-backup")
    seen = {}

    def fake_urlopen(req, timeout=0):
        body = json.loads(req.data.decode("utf-8"))
        seen["model"] = body["model"]
        seen["temperature"] = body["temperature"]
        seen["has_stream"] = "stream" in body
        return _FakeResp({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    monkeypatch.setattr(J.urllib.request, "urlopen", fake_urlopen)
    out = J.cascade_chat_completion({"model": "openai/gpt-5.4", "temperature": 0.7,
                                     "stream": True, "messages": [{"role": "user", "content": "hi"}]})
    assert out["choices"][0]["message"]["content"] == "ok"
    assert seen["model"] == "gemini-primary"        # requested model ignored, cascade used
    assert seen["temperature"] == 0.7               # the judge call's own temperature, verbatim
    assert seen["has_stream"] is False              # OpenRouter-only field dropped


def test_cascade_strips_thinking_keeps_function_tools(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-primary")
    seen = {}

    def fake_urlopen(req, timeout=0):
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    monkeypatch.setattr(J.urllib.request, "urlopen", fake_urlopen)
    # 06_Safety tasks add extra_body thinking; standard function tools pass through.
    J.cascade_chat_completion({
        "model": "openai/gpt-5.4",
        "thinking": {"type": "disabled"},
        "reasoning_effort": "low",
        "tools": [{"type": "function", "function": {"name": "keep_me"}}],
        "tool_choice": "auto",
        "messages": [{"role": "user", "content": "hi"}],
    })
    b = seen["body"]
    # thinking off is sent at the lowest Gemini thinking level; the client's own effort is dropped
    assert "thinking" not in b and b["reasoning_effort"] == J.THINKING_OFF_LEVEL == "low"
    assert [t["type"] for t in b["tools"]] == ["function"]         # standard function tools kept
    assert b["tools"][0]["function"]["name"] == "keep_me"
    # OpenRouter's web search mixed with function tools is not served (never silently dropped).
    with pytest.raises(J.UnsupportedRequest):
        J.cascade_chat_completion({"tools": [{"type": "openrouter:web_search", "parameters": {}},
                                             {"type": "function", "function": {"name": "x"}}],
                                   "messages": [{"role": "user", "content": "hi"}]})


def test_cascade_routes_openrouter_web_search_to_grounding(monkeypatch):
    # 04_Search_Retrieval_task_1's verification judge asks for OpenRouter's web search: it is served
    # by Gemini generateContent with the google_search tool, not dropped.
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-primary")
    seen = {}

    def fake_urlopen(req, timeout=0):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})

    monkeypatch.setattr(J.urllib.request, "urlopen", fake_urlopen)
    out = J.cascade_chat_completion({"tools": [{"type": "openrouter:web_search"}],
                                     "tool_choice": "required",
                                     "messages": [{"role": "user", "content": "x"}]})
    assert seen["url"].endswith("/models/gemini-primary:generateContent")
    assert seen["body"]["tools"] == [{"google_search": {}}]
    assert out["choices"][0]["message"]["content"] == "ok"


def test_cascade_falls_through_then_backoff_then_outage(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "m1,m2")
    monkeypatch.setenv("GBENCH_JUDGE_CASCADE_ROUNDS", "2")
    monkeypatch.setenv("GBENCH_JUDGE_BACKOFF", "1.0")
    sleeps = []
    monkeypatch.setattr(J.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(J.random, "uniform", lambda a, b: 0.0)

    def always_fail(req, timeout=0):
        raise RuntimeError("429")
    monkeypatch.setattr(J.urllib.request, "urlopen", always_fail)
    with pytest.raises(RuntimeError, match="JUDGE_OUTAGE"):
        J.cascade_chat_completion({"messages": [{"role": "user", "content": "x"}]})
    # 2 rounds -> one backoff sleep BETWEEN them: backoff*2^0 + uniform(0) = 1.0
    assert sleeps == [pytest.approx(1.0)]


def test_cascade_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        J.cascade_chat_completion({"messages": []})


@pytest.fixture(autouse=True)
def _metadata_guard_answers_blocked(monkeypatch):
    """Docker is stubbed in these tests: the cloud metadata guard the suites call before starting a
    container (container_guard.require_metadata_blocked) answers "unreachable" without a probe.
    tests/test_guard_tool_suites.py checks the suites' calls to it."""
    from gbench.runners.eval_suites import container_guard
    monkeypatch.setattr(container_guard, "require_metadata_blocked",
                        lambda *a, **k: container_guard.UNREACHABLE)


# --------------------------------------------------------------------------- #
# Sampling: a --thinking run samples the model at 1.0 (the maintainer's rule): the orchestrator's
# model relay forwards to a gbench EndpointRelay that sets it on every request; a no-think run
# relays straight to the endpoint (OpenClaw's own default).
# --------------------------------------------------------------------------- #
def _fake_model_endpoint():
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    seen = []

    class _H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append({"body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                         "auth": self.headers.get("Authorization")})
            out = json.dumps({"choices": [{"index": 0, "finish_reason": "stop",
                                           "message": {"role": "assistant", "content": "ok"}}]})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out.encode())

        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def _run_with_model_call(monkeypatch, tmp_path, thinking, env=None, sent_temperature=0.0):
    """run_wildclawbench with docker faked: the 'orchestrator' sends one chat request, as its
    model relay would for OpenClaw (temperature `sent_temperature`, the proxy's Bearer key), to
    the GBENCH_MODEL_BASE_URL it was given."""
    import subprocess
    import urllib.request
    srv, seen = _fake_model_endpoint()
    endpoint = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    harness = tmp_path / "harness"
    harness.mkdir()
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    real_mkdtemp = W.tempfile.mkdtemp
    monkeypatch.setattr(W.tempfile, "mkdtemp", lambda prefix="", dir=None: str(workdir) if (
        prefix.startswith("gbench_wildclawbench_")) else real_mkdtemp(prefix=prefix, dir=dir))
    monkeypatch.setattr(W, "check_wildclawbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(W, "_harness_dir", lambda: str(harness))
    monkeypatch.setattr(W, "free_port", lambda preferred=None: preferred)
    monkeypatch.setattr(W.container_guard, "require_metadata_blocked", lambda *a, **k: None)
    for k in ("GBENCH_WILDCLAWBENCH_TASK_NETWORK", "GBENCH_WILDCLAWBENCH_JUDGE_PORT",
              "GBENCH_SHARD", "GBENCH_RESULTS_DIR", "GBENCH_WILDCLAWBENCH_TASK_ENDPOINT",
              "GBENCH_WILDCLAWBENCH_TEMPERATURE", "VLLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GBENCH_WILDCLAWBENCH_MODEL_API_KEY", "sk-model-endpoint-key-0000")
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    got = {}

    def fake_run(cmd, *a, **k):
        if cmd[:3] == ["docker", "run", "--rm"] and "--name" in cmd:
            url = next(x.split("=", 1)[1] for x in cmd if x.startswith("GBENCH_MODEL_BASE_URL="))
            got["model_url"] = url
            got["proxy_key"] = k["env"]["GBENCH_MODEL_API_KEY"]
            req = urllib.request.Request(
                url + "/chat/completions", method="POST",
                data=json.dumps({"model": "m", "temperature": sent_temperature,
                                 "messages": [{"role": "user", "content": "hi"}]}).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {got['proxy_key']}"})
            urllib.request.urlopen(req, timeout=30).read()
            (workdir / "wildclawbench_summary.json").write_text(json.dumps({
                "n_tasks": 1, "n_scored": 1, "n_headline_tasks": 1, "global_mean": 0.5,
                "n_missing": 0, "judge_fallback_tasks": 0, "n_search_setup_failures": 0,
                "per_task": [{"task_id": "a", "overall_score": 0.5, "harness_errors": []}]}))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="orchestrator log")
    monkeypatch.setattr(W.subprocess, "run", fake_run)
    try:
        result = W.run_wildclawbench("m", endpoint, enable_thinking=thinking)
    finally:
        srv.shutdown()
        srv.server_close()
    return result, got, seen, endpoint


def test_thinking_run_sets_temperature_one_on_the_agents_model_requests(monkeypatch, tmp_path):
    result, got, seen, endpoint = _run_with_model_call(monkeypatch, tmp_path, thinking=True)
    assert got["model_url"].startswith("http://127.0.0.1:") and got["model_url"] != endpoint
    assert [s["body"]["temperature"] for s in seen] == [1.0]   # OpenClaw sent 0.0
    assert seen[0]["auth"] == "Bearer sk-model-endpoint-key-0000"
    assert result["temperature_applied"] is True
    assert (result["temperature"], result["temperature_source"]) == (1.0, "thinking")
    relay = result["model_temperature_relay"]
    assert relay["sets_body_fields"] == {"temperature": 1.0, "chat_template_kwargs": {
        "enable_thinking": True}} and relay["relays_to"] == endpoint
    assert "sk-model-endpoint-key-0000" not in json.dumps(result, default=str)
    assert "temperature 1.0 set by thinking on every model request" in (
        result["leaderboard_comparable_reason"])
    assert result["sampling"].startswith("temperature 1.0 (thinking) set by gbench's relay")


def test_thinking_run_operator_temperature_wins(monkeypatch, tmp_path):
    result, _got, seen, _ = _run_with_model_call(
        monkeypatch, tmp_path, thinking=True, env={"GBENCH_WILDCLAWBENCH_TEMPERATURE": "0.6"})
    assert [s["body"]["temperature"] for s in seen] == [0.6]
    assert result["temperature_source"] == "env:GBENCH_WILDCLAWBENCH_TEMPERATURE"


def test_no_think_run_sets_temperature_zero_on_the_agents_model_requests(monkeypatch, tmp_path):
    # The maintainer's rule: a no-think run samples at 0.0, whatever OpenClaw sends (0.7 here),
    # through the same relay as a --thinking run.
    result, got, seen, endpoint = _run_with_model_call(monkeypatch, tmp_path, thinking=False,
                                                       sent_temperature=0.7)
    assert got["model_url"].startswith("http://127.0.0.1:") and got["model_url"] != endpoint
    assert [s["body"]["temperature"] for s in seen] == [0.0]
    assert seen[0]["auth"] == "Bearer sk-model-endpoint-key-0000"
    assert result["temperature_applied"] is True
    assert (result["temperature"], result["temperature_source"]) == (0.0, "default")
    relay = result["model_temperature_relay"]
    assert relay["sets_body_fields"] == {"temperature": 0.0, "chat_template_kwargs": {
        "enable_thinking": False}} and relay["relays_to"] == endpoint
    assert "temperature 0.0 set by default on every model request" in (
        result["leaderboard_comparable_reason"])
    assert result["sampling"].startswith("temperature 0.0 (default) set by gbench's relay")


def test_no_think_run_operator_temperature_wins(monkeypatch, tmp_path):
    result, _got, seen, _ = _run_with_model_call(
        monkeypatch, tmp_path, thinking=False, env={"GBENCH_WILDCLAWBENCH_TEMPERATURE": "0.6"})
    assert [s["body"]["temperature"] for s in seen] == [0.6]
    assert result["temperature_source"] == "env:GBENCH_WILDCLAWBENCH_TEMPERATURE"
