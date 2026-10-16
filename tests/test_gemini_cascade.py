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

"""One Gemini model cascade for every Gemini text role.

* base.GEMINI_CASCADE is the maintainers' list, and judge_cascade() reads GBENCH_JUDGE_MODELS,
  then GBENCH_JUDGE_MODEL, then the constant; resolve_judge_model() returns its head.
* search_tool uses the same list (GBENCH_SEARCH_MODELS overrides search only) and a model the
  API answers 404 for is not retried in place nor asked again in the lookup; the judge cascade
  skips it the same way.
* gemini_cascade_env() hands the list to containers.
* The key gate checks that every cascade model is served (cached per process) and names a
  missing model and its knob.
* Every hardcoded model list under gbench/docker, gbench/runners and docker equals the constant.
* README and env-knobs state the exact default list; the gdpval judge-validation script uses the
  cascade and the suite's current pairwise prompt.
No network: the Gemini API is a local stub or a fake client.
"""

import ast
import asyncio
import json
import os
import re
import subprocess
import sys
import threading
import urllib.error
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from gbench.runners.eval_suites import base
from gbench.runners.eval_suites import search_tool as ST

REPO = Path(__file__).resolve().parents[1]
CASCADE = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash",
           "gemini-3.1-pro-preview"]
KEY = "AIzaFAKEKEYcascade000000000000000001"
KNOBS = ("GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL", "GBENCH_SEARCH_MODELS",
         "GBENCH_SEARCH_MODEL")


@pytest.fixture(autouse=True)
def _clean_knobs(monkeypatch):
    for name in KNOBS:
        monkeypatch.delenv(name, raising=False)
    base.reset_gemini_model_checks()
    yield
    base.reset_gemini_model_checks()


def _read(rel):
    return (REPO / rel).read_text(encoding="utf-8")


def _flat(text):
    return " ".join(re.sub(r"(?m)^\s*>\s?", "", text).split())


# --------------------------------------------------------------------------- #
# the source of truth
# --------------------------------------------------------------------------- #
def test_the_constant_is_the_maintainers_cascade():
    assert list(base.GEMINI_CASCADE) == CASCADE
    assert isinstance(base.GEMINI_CASCADE, tuple)
    assert "gemini-2.5-flash" not in base.GEMINI_CASCADE
    assert "gemini-3-flash-preview" not in base.GEMINI_CASCADE


def test_judge_cascade_precedence(monkeypatch):
    assert base.judge_cascade() == CASCADE and base.judge_cascade_source() == "default"
    monkeypatch.setenv("GBENCH_JUDGE_MODEL", "gemini-3.5-flash")
    assert base.judge_cascade() == ["gemini-3.5-flash"]
    assert base.judge_cascade_source() == "GBENCH_JUDGE_MODEL"
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", " gemini-3.7-flash , gemini-3.6-flash ")
    assert base.judge_cascade() == ["gemini-3.7-flash", "gemini-3.6-flash"]
    assert base.judge_cascade_source() == "GBENCH_JUDGE_MODELS"
    # a value that names no model is not a cascade of zero models
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", " , ,")
    assert base.judge_cascade() == ["gemini-3.5-flash"]
    monkeypatch.delenv("GBENCH_JUDGE_MODEL")
    assert base.judge_cascade() == CASCADE


def test_resolve_judge_model_returns_the_head_and_logs_the_cascade(monkeypatch, caplog):
    base._JUDGE_RESOLVE_LOGGED.clear()
    with caplog.at_level("WARNING"):
        assert base.resolve_judge_model("google/gemma-4-26B-A4B-it") == (
            "gemini-3.8-flash", "default", False)
    assert str(CASCADE) in caplog.text
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-3.6-flash,gemini-3.5-flash")
    base._JUDGE_RESOLVE_LOGGED.clear()
    assert base.resolve_judge_model() == ("gemini-3.6-flash", "env:GBENCH_JUDGE_MODELS", False)
    monkeypatch.delenv("GBENCH_JUDGE_MODELS")
    monkeypatch.setenv("GBENCH_JUDGE_MODEL", "gemini-3.5-flash")
    base._JUDGE_RESOLVE_LOGGED.clear()
    assert base.resolve_judge_model() == ("gemini-3.5-flash", "env:GBENCH_JUDGE_MODEL", False)


def test_the_import_default_alias_and_the_warning_name_the_cascade():
    code = ("import logging, sys; logging.basicConfig(stream=sys.stdout, level=logging.WARNING);"
            "from gbench.runners.eval_suites import base; print('HEAD=' + base.DEFAULT_JUDGE_MODEL)")
    env = {k: v for k, v in os.environ.items() if k not in KNOBS}
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True,
                         text=True, timeout=120).stdout
    assert "HEAD=gemini-3.8-flash" in out and "override is ACTIVE" not in out
    env["GBENCH_JUDGE_MODEL"] = "gemini-3.5-flash"
    env["GBENCH_JUDGE_MODELS"] = "gemini-3.7-flash"
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True,
                         text=True, timeout=120).stdout
    assert "GBENCH_JUDGE_MODELS override is ACTIVE" in out
    assert "['gemini-3.7-flash']" in out and str(CASCADE) in out
    assert "GBENCH_JUDGE_MODEL is also set and is ignored" in out
    assert "gemini-3.6-flash" not in out.split("instead of the default")[0]


def test_the_container_env_carries_the_cascade(monkeypatch):
    assert base.gemini_cascade_env() == {"GBENCH_JUDGE_MODELS": ",".join(CASCADE),
                                         "GBENCH_SEARCH_MODELS": ",".join(CASCADE)}
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "a,b")
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "c")
    assert base.gemini_cascade_env() == {"GBENCH_JUDGE_MODELS": "a,b",
                                         "GBENCH_SEARCH_MODELS": "c"}


# --------------------------------------------------------------------------- #
# search_tool: the same list, and a 404 is permanent
# --------------------------------------------------------------------------- #
def test_search_uses_the_same_cascade_and_its_own_override(monkeypatch):
    assert ST._DEFAULT_SEARCH_CASCADE == CASCADE
    assert ST._search_cascade() == CASCADE == base.search_cascade()
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-3.6-flash,gemini-3.5-flash")
    assert ST._search_cascade() == base.search_cascade() == ["gemini-3.6-flash",
                                                             "gemini-3.5-flash"]
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "gemini-3.7-flash")
    assert ST._search_cascade() == base.search_cascade() == ["gemini-3.7-flash"]
    assert base.judge_cascade() == ["gemini-3.6-flash", "gemini-3.5-flash"]
    assert base.search_cascade_source() == "GBENCH_SEARCH_MODELS"


def _http_error(code):
    return urllib.error.HTTPError("https://x/y", code, "err", {}, None)


class _Grounded:
    def __init__(self):
        self._body = json.dumps({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
                                ).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stub_search(monkeypatch, codes):
    """urlopen answering by model: an HTTP status to raise, or 200 (a grounded answer)."""
    calls = []

    def urlopen(req, timeout=None):
        model = re.search(r"models/([^:]+):", req.full_url).group(1)
        calls.append(model)
        code = codes.get(model, 200)
        if code != 200:
            raise _http_error(code)
        return _Grounded()
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(ST.time, "sleep", lambda s: None)
    monkeypatch.setattr(ST, "_RATE_LIMITER", ST._RateLimiter(0, 1))
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GBENCH_GEMINI_RELAY_URL", raising=False)
    return calls


def test_a_not_served_search_model_is_not_retried_and_the_next_one_answers(monkeypatch):
    calls = _stub_search(monkeypatch, {"gemini-3.8-flash": 404})
    results, backend = ST.search_with_fallback("q")
    assert backend == "gemini:gemini-3.7-flash"
    assert calls == ["gemini-3.8-flash", "gemini-3.7-flash"]    # no in-place 404 retries


def test_a_not_served_search_model_is_not_asked_again_in_later_rounds(monkeypatch):
    codes = {m: 503 for m in CASCADE}
    codes["gemini-3.8-flash"] = 404
    calls = _stub_search(monkeypatch, codes)
    monkeypatch.setenv("GBENCH_SEARCH_ATTEMPTS", "3")
    monkeypatch.setattr(ST, "_SEARCH_ATTEMPTS", 2)
    results, backend = ST.search_with_fallback("q")
    assert backend == "all-grounding-models-failed"
    assert calls.count("gemini-3.8-flash") == 1                 # once, in the first round only
    assert calls.count("gemini-3.7-flash") == 2 * ST._cascade_rounds()   # 503: retried in place


def test_a_lookup_whose_every_model_is_not_served_stops_after_one_round(monkeypatch):
    calls = _stub_search(monkeypatch, {m: 404 for m in CASCADE})
    results, backend = ST.search_with_fallback("q")
    assert backend == "all-grounding-models-failed" and calls == CASCADE
    assert results[0]["title"] == "error" and "404" in results[0]["snippet"]


def test_a_stubbed_gemini_search_still_drives_the_cascade(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    seen = []

    def fake(q, n=8, model=None):
        seen.append(model)
        if model == CASCADE[0]:
            return [{"title": "error", "url": "", "snippet": "429"}]
        return [{"title": "t", "url": "", "snippet": "s"}]
    monkeypatch.setattr(ST, "gemini_search", fake)
    assert ST.search_with_fallback("q")[1] == f"gemini:{CASCADE[1]}"
    assert seen == CASCADE[:2]


# --------------------------------------------------------------------------- #
# the judge cascade: order, 404 skip, answering model recorded
# --------------------------------------------------------------------------- #
class _Err(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


class _FakeClient:
    def __init__(self, behavior):
        self.calls = []
        outer = self

        class _Models:
            async def generate_content(self, model, contents, config=None):
                outer.calls.append(model)
                what = behavior.get(model, "ok")
                if isinstance(what, int):
                    raise _Err(what)

                class _R:
                    text = f"answer from {model}"
                    candidates = []
                return _R()

        class _Aio:
            models = _Models()
        self.aio = _Aio()


def _install_judge(monkeypatch, behavior):
    client = _FakeClient(behavior)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.setattr(base, "_judge_clients", lambda: [client])
    monkeypatch.setattr(base, "_next_judge_client", lambda: client)
    monkeypatch.setattr(base, "_JUDGE_BACKOFF", 0.0)
    return client


def test_the_judge_asks_the_cascade_in_order_and_records_who_answered(monkeypatch):
    client = _install_judge(monkeypatch, {"gemini-3.8-flash": 429})
    before = base.gemini_answer_counts().get("gemini-3.7-flash", 0)
    text, model = asyncio.run(base.judge_generate_cascade("grade this"))
    assert (text, model) == ("answer from gemini-3.7-flash", "gemini-3.7-flash")
    assert client.calls == ["gemini-3.8-flash", "gemini-3.7-flash"]
    assert base.gemini_answer_counts()["gemini-3.7-flash"] == before + 1


def test_the_judge_does_not_ask_a_not_served_model_again(monkeypatch):
    behavior = {m: 503 for m in CASCADE}
    behavior["gemini-3.8-flash"] = 404
    client = _install_judge(monkeypatch, behavior)
    text, model = asyncio.run(base.judge_generate_cascade("grade this"))
    assert (text, model) == (None, "JUDGE_OUTAGE")
    rounds = base._judge_cascade_rounds()
    assert client.calls.count("gemini-3.8-flash") == 1
    assert client.calls.count("gemini-3.7-flash") == rounds


def test_run_eval_suite_records_the_gemini_models_that_answered(monkeypatch):
    _install_judge(monkeypatch, {"gemini-3.8-flash": 429})

    async def fake_async(**kw):
        await base.judge_generate_cascade("x")
        await base.judge_generate_cascade("y")
        return {"status": "success", "accuracy": 100.0, "sample_traces": []}
    monkeypatch.setattr(base, "_run_suite_async", fake_async)
    res = base.run_eval_suite("unit_judged", "m", "http://127.0.0.1:1/v1", 1,
                              [([{"role": "user", "content": "q"}], "a", {})],
                              eval_fn=lambda r, g: True)
    assert res["gemini_models_answered"] == {"gemini-3.7-flash": 2}
    assert res["gemini_cascade"] == CASCADE


# --------------------------------------------------------------------------- #
# the key gate's model check
# --------------------------------------------------------------------------- #
@pytest.fixture
def gemini_api(monkeypatch):
    """A local stand-in for the Gemini OpenAI-compatible endpoint: GET /v1/models answers 200,
    GET /v1/models/<name> answers state["models"].get(name, 200)."""
    state = {"models": {}, "hits": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["hits"].append(self.path)
            status = 200
            if self.path.startswith("/v1/models/"):
                status = state["models"].get(self.path[len("/v1/models/"):], 200)
            body = json.dumps({"error": {"message": "models/x is not found"}}
                              if status != 200 else {"data": []}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}/v1")
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    yield state
    srv.shutdown()


def test_the_gate_checks_every_cascade_model_once_per_process(gemini_api):
    base.gemini_required_skip("simpleqa", "m")
    assert gemini_api["hits"] == ["/v1/models"] + [f"/v1/models/{m}" for m in CASCADE]
    base.gemini_required_skip("frames", "m")
    # the key is pinged again, the served models are not asked again
    assert gemini_api["hits"][len(CASCADE) + 1:] == ["/v1/models"]


def test_a_missing_default_model_is_infra_required_naming_it_and_the_knob(gemini_api):
    gemini_api["models"]["gemini-3.8-flash"] = 404
    with pytest.raises(RuntimeError) as ei:
        base.gemini_required_skip("simpleqa", "m")
    msg = str(ei.value)
    assert "[INFRA REQUIRED] simpleqa" in msg and "'gemini-3.8-flash'" in msg
    assert "default Gemini cascade" in msg and "GBENCH_JUDGE_MODELS" in msg
    assert "was rejected by the judge endpoint" not in msg and KEY not in msg
    ok, why = base.gemini_key_live_valid(KEY)
    assert ok is False and "gemini-3.8-flash" in why


def test_a_mistyped_knob_model_is_named_with_its_knob(gemini_api, monkeypatch):
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-3.7-flash,gemni-3.6-flash")
    gemini_api["models"]["gemni-3.6-flash"] = 404
    ok, why = base.gemini_key_live_valid(KEY)
    assert ok is False and "'gemni-3.6-flash' from GBENCH_JUDGE_MODELS" in why
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "gemini-3.7-flash,gemini-9-flash")
    gemini_api["models"]["gemini-9-flash"] = 403
    ok, why = base.gemini_key_live_valid(KEY)
    assert "'gemini-9-flash' from GBENCH_SEARCH_MODELS" in why and "GBENCH_SEARCH_MODELS" in why


def test_an_inconclusive_model_answer_never_blocks_and_is_asked_again(gemini_api):
    gemini_api["models"]["gemini-3.8-flash"] = 503
    assert base.gemini_key_live_valid(KEY) == (True, "")
    n = gemini_api["hits"].count("/v1/models/gemini-3.8-flash")
    assert base.gemini_key_live_valid(KEY) == (True, "")
    assert gemini_api["hits"].count("/v1/models/gemini-3.8-flash") == n + 1


def test_an_inconclusive_key_ping_skips_the_model_check(monkeypatch):
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", f"http://127.0.0.1:{base.free_port()}/v1")
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    ok, why = base.gemini_key_live_valid(KEY)
    assert ok is True and why.startswith("inconclusive")


# --------------------------------------------------------------------------- #
# every hardcoded copy equals the constant
# --------------------------------------------------------------------------- #
_MODEL = r"(?:models/|gemini/)?gemini-[0-9][\w.\-]*"
_RUN = re.compile(_MODEL + r"(?:[\"']?\s*,\s*[\"']?" + _MODEL + r")+")
_ONE = re.compile(r"^" + _MODEL + r"$")
#: plugin files are never read
_SKIP = {"swe_bench_pro_agentic.py", "jetski_agent.py", "slashbin_agent.py"}


def _norm(m):
    for prefix in ("models/", "gemini/"):
        if m.startswith(prefix):
            m = m[len(prefix):]
    return m


def _hardcoded_lists():
    found = []
    for top in ("gbench/docker", "gbench/runners", "docker"):
        for dirpath, dirnames, filenames in os.walk(REPO / top):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for name in filenames:
                if ".bak" in name or name in _SKIP or name.endswith(".pyc"):
                    continue
                path = Path(dirpath) / name
                try:
                    src = path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                rel = str(path.relative_to(REPO))
                if name.endswith(".py"):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", SyntaxWarning)
                            tree = ast.parse(src)
                    except SyntaxError:
                        continue
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
                            vals = [e.value for e in node.elts
                                    if isinstance(e, ast.Constant) and isinstance(e.value, str)]
                            if (len(vals) == len(node.elts) >= 2
                                    and all(_ONE.match(v) for v in vals)):
                                found.append((rel, node.lineno, [_norm(v) for v in vals]))
                        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                            v = node.value.strip()
                            if re.fullmatch(_MODEL + r"(?:\s*,\s*" + _MODEL + r")+", v):
                                found.append((rel, node.lineno,
                                              [_norm(x.strip()) for x in v.split(",")]))
                else:
                    for i, line in enumerate(src.splitlines(), 1):
                        if line.lstrip().startswith("#"):
                            continue
                        for m in _RUN.finditer(line):
                            found.append((rel, i, [_norm(x) for x in
                                                   re.findall(_MODEL, m.group(0))]))
    # non-text roles keep their own model (embeddings, image models)
    return [(f, n, v) for f, n, v in found
            if not any("embedding" in x or "image" in x for x in v)]


def test_the_scan_finds_the_constant_and_the_search_copy():
    where = {f for f, _, _ in _hardcoded_lists()}
    assert "gbench/runners/eval_suites/base.py" in where
    assert "gbench/runners/eval_suites/search_tool.py" in where


def test_every_hardcoded_model_list_equals_the_constant():
    wrong = [f"{f}:{n} {v}" for f, n, v in _hardcoded_lists() if v != CASCADE]
    assert not wrong, ("these model lists differ from base.GEMINI_CASCADE (read "
                       "GBENCH_JUDGE_MODELS from the environment, or copy the constant): "
                       + "; ".join(wrong))


# --------------------------------------------------------------------------- #
# docs and the gdpval validation script
# --------------------------------------------------------------------------- #
def test_readme_and_env_knobs_state_the_exact_default_list():
    listed = ", ".join(f"`{m}`" for m in CASCADE)
    # The README's "Environment variables" section moved verbatim to docs/environment-variables.md.
    for doc in ("docs/environment-variables.md", "docs/evals/env-knobs.md"):
        text = _flat(_read(doc))
        assert listed in text, doc
        assert "gemini_cascade_env" in text or doc == "docs/environment-variables.md"
    knobs = _flat(_read("docs/evals/env-knobs.md"))
    assert "`GBENCH_JUDGE_MODELS` wins" in knobs
    assert "GBENCH_SEARCH_MODELS" in knobs and "search only" in knobs


def test_the_gdpval_validation_script_uses_the_cascade_and_the_current_prompt(monkeypatch):
    import importlib.util
    from gbench.runners.eval_suites import gdpval as G
    src = _read("scripts/gdpval_validate_judge.py")
    assert "_JUDGE_PROMPT" not in src.replace("_PAIRWISE_PROMPT", "")
    assert "DEFAULT_JUDGE_MODEL" not in src and "genai.Client" not in src
    spec = importlib.util.spec_from_file_location("gdpval_validate_judge",
                                                  REPO / "scripts/gdpval_validate_judge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prompts = []

    async def fake_cascade(prompt, *, config=None):
        prompts.append(prompt)
        return json.dumps({"winner": "tie", "reason": "same"}), "gemini-3.8-flash"
    monkeypatch.setattr(mod, "judge_generate_cascade", fake_cascade)
    mean, verdicts, models = asyncio.run(mod.swapped_mean("task", "[1 pts] c", "REF", "REF"))
    assert mean == 0.5 and verdicts == ("tie", "tie") and models == ("gemini-3.8-flash",) * 2
    assert all(p.startswith(G._PAIRWISE_PROMPT.split("{", 1)[0]) for p in prompts)
    assert "# DELIVERABLE A\nREF" in prompts[0]
