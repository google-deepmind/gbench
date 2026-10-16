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

"""bfcl and bfcl_v4_agentic: the project root's .env, endpoint failures, labels, the cascade.

* Upstream's `bfcl generate` / `bfcl evaluate` load <project root>/.env with override=True, and
  the project root is mounted into their containers: a root that holds one is refused
  (infra_required) before anything is generated and before each container starts, and the
  harness child makes that load_dotenv load nothing.
* Endpoint failures gbench's endpoint relay counted are harness errors: completed_with_errors,
  `endpoint_failures` and a named non-comparability reason. Scored entries whose request raised
  are failed requests; a run (or bfcl part) where every entry failed withholds its score.
* Every error of the agentic part names the suite that runs (bfcl when bfcl runs the track).
* The web search asks gbench's one Gemini cascade (base.search_cascade()), hands it to the
  container, and records which models answered.
* The endpoint check is told when the container gets a proxy.

Docker, the network checks and the relays are stubbed; nothing contacts a model or Gemini.
"""

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from gbench.runners.eval_suites import base, bfcl
from gbench.runners.eval_suites import bfcl_v4_agentic as AG
from gbench.runners.eval_suites import tau_common as TC

REPO = Path(__file__).resolve().parents[1]
# Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
DOTENV_KEY = "sk-" + "DOTENVuserKEYfromBFCLsetup0123"


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GEMINI_API_KEYS", "GEMINI_OPENAI_BASE_URL",
                 "GBENCH_GEMINI_RELAY_URL", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy",
                 "https_proxy", "NO_PROXY", "no_proxy", "GBENCH_SEARCH_MODELS",
                 "GBENCH_SEARCH_MODEL", "GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL",
                 "GBENCH_BFCL_TIMEOUT_S", "BFCL_TIMEOUT_S", "GBENCH_SHARD", "BFCL_PROJECT_ROOT",
                 "GBENCH_BFCL_PROJECT_ROOT", AG.SEARCH_LOG_ENV, AG.GUARD_LOG_ENV):
        monkeypatch.delenv(name, raising=False)
    AG._RELAY_HOSTS_OK.clear()
    yield
    AG._RELAY_HOSTS_OK.clear()


def _forwarded(cmd, env):
    """{name: value} of the variables the container receives (`-e NAME`)."""
    names = [cmd[i + 1] for i, a in enumerate(cmd[:-1]) if a == "-e" and "=" not in cmd[i + 1]]
    return {n: env.get(n) for n in names}


# --------------------------------------------------------------------------- #
# the project root's .env never reaches a harness container
# --------------------------------------------------------------------------- #
def _write_dotenv(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / ".env"
    path.write_text(f"OPENAI_API_KEY={DOTENV_KEY}\nOPENAI_BASE_URL=http://10.9.9.9:1/v1\n",
                    encoding="utf-8")
    return path


def test_no_dotenv_no_problem(tmp_path):
    assert AG.dotenv_problem(tmp_path) is None


@pytest.mark.parametrize("kind", ["file", "dir", "dangling_link"])
def test_any_dotenv_entry_is_named_and_never_read(tmp_path, kind):
    path = tmp_path / ".env"
    if kind == "file":
        _write_dotenv(tmp_path)
    elif kind == "dir":
        path.mkdir()
    else:
        path.symlink_to(tmp_path / "missing")
    why = AG.dotenv_problem(tmp_path)
    assert why and str(path) in why and "GBENCH_BFCL_PROJECT_ROOT" in why
    assert "load_dotenv" in why and DOTENV_KEY not in why


def test_the_default_root_is_the_suite_knob_then_upstreams_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path / "upstream"))
    assert AG.harness_project_root() == tmp_path / "upstream"
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(tmp_path / "gbench"))
    assert AG.harness_project_root() == tmp_path / "gbench"
    _write_dotenv(tmp_path / "gbench")
    assert AG.dotenv_problem() is not None


def _no_probe(monkeypatch):
    """The steps after the .env check fail loudly if they are reached."""
    def boom(*a, **k):
        raise AssertionError("reached a step after the .env check")
    monkeypatch.setattr(AG.urllib.request, "urlopen", boom)
    monkeypatch.setattr(AG, "image_problem", boom)
    monkeypatch.setattr(AG, "run_in_image", boom)


def test_the_prerequisite_check_refuses_a_dotenv_first(monkeypatch, tmp_path):
    root = tmp_path / "bfcl"
    _write_dotenv(root)
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(root))
    _no_probe(monkeypatch)
    ok, why, cats = AG.check_bfcl_prerequisites("http://127.0.0.1:9/v1")
    assert ok is False and cats == [] and str(root / ".env") in why


def test_bfcl_v4_agentic_stops_with_its_own_label(monkeypatch, tmp_path):
    root = tmp_path / "bfcl"
    _write_dotenv(root)
    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(root))      # upstream's own variable
    _no_probe(monkeypatch)
    with pytest.raises(RuntimeError) as err:
        AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1)
    message = str(err.value)
    assert message.startswith("[INFRA REQUIRED] bfcl_v4_agentic")
    assert "docs/evals/bfcl_v4_agentic.md" in message and DOTENV_KEY not in message


def test_bfcl_stops_before_anything_is_generated(monkeypatch, tmp_path):
    root = tmp_path / "bfcl"
    _write_dotenv(root)
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(root))
    _no_probe(monkeypatch)
    monkeypatch.setattr(AG, "search_backend", lambda: (AG.SEARCH_BACKEND, ""))
    monkeypatch.setattr(bfcl, "run_eval_suite",
                        lambda **k: (_ for _ in ()).throw(AssertionError("generated")))
    with pytest.raises(RuntimeError) as err:
        bfcl.run_bfcl("m", "http://127.0.0.1:9/v1", 1, eval_categories="agentic")
    message = str(err.value)
    assert message.startswith("[INFRA REQUIRED] bfcl cannot run")
    assert "'docs/evals/bfcl.md'" in message and ".env" in message
    assert "bfcl_v4_agentic" not in message


@pytest.mark.parametrize("command", ["generate", "evaluate"])
def test_no_container_starts_while_the_root_holds_a_dotenv(monkeypatch, tmp_path, command):
    _write_dotenv(tmp_path)
    launched = []
    monkeypatch.setattr(AG, "_launch", lambda *a, **k: launched.append(a) or (0, ""))
    monkeypatch.setattr(AG, "bridge_services",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("relays started")))
    env = {"BFCL_PROJECT_ROOT": str(tmp_path), "OPENAI_BASE_URL": "http://127.0.0.1:9/v1"}
    with AG.suite_context("bfcl", "docs/evals/bfcl.md"):
        with pytest.raises(RuntimeError) as err:
            AG._run_cli([command, "--model", "m"], env, None, model="m")
    assert launched == []
    assert str(err.value).startswith("[INFRA REQUIRED] bfcl") and "docs/evals/bfcl.md" in \
        str(err.value)


def test_the_child_makes_upstreams_dotenv_load_a_no_op(monkeypatch, tmp_path):
    """The bootstrap replaces bfcl_eval.__main__.load_dotenv after importing the CLI and before
    it dispatches: `generate` / `evaluate` call it with the project root's .env and
    override=True, and it loads nothing."""
    path = _write_dotenv(tmp_path)
    calls = []

    def upstream_load_dotenv(dotenv_path=None, verbose=False, override=False):
        calls.append(("upstream", str(dotenv_path)))
        os.environ["OPENAI_API_KEY"] = DOTENV_KEY
        return True
    fake_main = types.ModuleType("bfcl_eval.__main__")
    fake_main.DOTENV_PATH = path
    fake_main.load_dotenv = upstream_load_dotenv

    def cli():
        # as upstream's `generate`: the module global, looked up when the command runs
        calls.append(("cli", fake_main.load_dotenv(dotenv_path=fake_main.DOTENV_PATH,
                                                   verbose=True, override=True)))
    fake_main.cli = cli
    monkeypatch.setitem(sys.modules, "bfcl_eval.__main__", fake_main)
    for name in ("_register_model", "_serve_web_search_with_gemini", "_install_execution_guard",
                 "_allow_one_latency"):
        monkeypatch.setattr(AG, name, lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["pytest"])
    monkeypatch.setenv("BFCL_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")
    code = AG._CHILD_BOOTSTRAP.format(model="m", args=["generate"])
    assert code.index("from bfcl_eval.__main__ import cli") < code.index("_disable_dotenv()") \
        < code.index("\ncli()")
    exec(code, {})
    assert calls == [("cli", False)]
    assert os.environ["OPENAI_API_KEY"] == "EMPTY"
    assert fake_main.load_dotenv is AG._skip_dotenv


# --------------------------------------------------------------------------- #
# endpoint failures, failed requests and a withheld score (harness errors are never the
# model's outcome)
# --------------------------------------------------------------------------- #
def test_endpoint_failures_sum_the_relays_and_leave_out_context_rejections():
    assert AG.endpoint_failures([None, {}]) is None
    assert AG.endpoint_failures([{"upstream_errors": {}, "requests_refused": 0,
                                  "context_length_rejections": 7}]) is None
    got = AG.endpoint_failures([
        {"upstream_errors": {"connection": 2, "http_502": 1}, "requests_refused": 0},
        None,
        {"upstream_errors": {"connection": 1}, "requests_refused": 3,
         "context_length_rejections": 4}])
    assert got == {"upstream_errors": {"connection": 3, "http_502": 1}, "requests_refused": 3}
    reason = AG.endpoint_failure_reason(got, "`bfcl generate`")
    assert reason.startswith("harness errors during `bfcl generate`: the served endpoint failed 4 "
                             "relayed model request(s) (connection: 3, http_502: 1)")
    assert "the relay refused 3 request(s)" in reason


def _result_file(root: Path, model: str, category: str, rows):
    d = root / model.replace("/", "_") / "agentic"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"BFCL_v4_{category}_result.json").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_inference_errors_are_read_from_the_result_files(tmp_path):
    _result_file(tmp_path, "org/m", "memory_kv", [
        {"id": "memory_kv_prereq_0", "result": "Error during inference: Connection error."},
        {"id": "memory_kv_0", "result": [["ok"]]},
        {"id": "memory_kv_1", "result": "Error during inference: Connection error."},
        {"id": "memory_kv_2", "result": "Error during inference: Request timed out."},
        {"id": "memory_kv_3", "result": "Error during inference: Error code: 400 - {'message': "
                                        "\"This model's maximum context length is 8192 tokens\"}"},
        {"id": "memory_kv_4", "result": "Error during inference: Error code: 502 - bad gateway"}])
    _result_file(tmp_path, "org/m", "memory_vector", [{"id": "memory_vector_0", "result": []}])
    got = AG.inference_errors(tmp_path, "org/m", ["memory_kv", "memory_vector"])
    assert got == {"memory_kv": {"failed": 2, "timed_out": 1, "over_context": 1}}


@pytest.fixture
def agentic(monkeypatch, tmp_path):
    """run_bfcl_v4_agentic with the harness CLI stubbed. `state["relay"]` is the endpoint
    relay record each `bfcl generate` appends; `state["rows"]` the result rows it writes per
    category; `state["scores"]` the per-category scores evaluate reports; `state["log"]` the
    web search log rows of a generate that searches."""
    root = tmp_path / "bfcl"
    state = {"cli": [], "relay": None, "rows": {}, "scores": None, "log": []}
    monkeypatch.setenv("GBENCH_BFCL_PROJECT_ROOT", str(root))
    monkeypatch.setattr(AG, "check_bfcl_prerequisites",
                        lambda url: (True, "", list(AG.MEMORY_CATEGORIES)))

    def cli(args, env, timeout, model="", backend=None):
        state["cli"].append({"args": list(args), "env": dict(env)})
        if args[0] == "generate":
            for cat, rows in state["rows"].items():
                _result_file(root / "result", model, cat, rows)
            if env.get(AG.SEARCH_LOG_ENV):
                with open(env[AG.SEARCH_LOG_ENV], "a", encoding="utf-8") as fh:
                    for row in state["log"]:
                        fh.write(json.dumps(row) + "\n")
            ctx = AG._suite()
            if ctx.relays is not None and state["relay"] is not None:
                ctx.relays.append({"command": "generate", "endpoint_relay": state["relay"],
                                   "gemini_relay": None})
        return 0, ""
    monkeypatch.setattr(AG, "_run_cli", cli)
    monkeypatch.setattr(AG, "_clear_stale_scores", lambda *a: 0)
    monkeypatch.setattr(AG, "_parse_scores", lambda r, m, cats: state["scores"] or {
        c: {"correct": 1, "total": 2, "accuracy": 50.0} for c in cats})
    monkeypatch.setattr(AG, "_parse_score_traces", lambda *a: [])
    monkeypatch.setattr(AG, "_score_error_types", lambda *a: {})
    return state


def test_the_result_says_the_dotenv_is_never_loaded(agentic):
    res = AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1)
    assert ".env is never loaded" in res["bfcl_report"]["credentials"]["dotenv"]


def test_a_clean_run_has_no_endpoint_failures(agentic):
    agentic["relay"] = {"upstream_errors": {}, "requests_refused": 0,
                        "context_length_rejections": 2}
    res = AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1)
    assert res["status"] == "success" and "endpoint_failures" not in res
    assert res["failed_requests"] == 0 and res["bfcl_report"]["inference_errors"] == {}


def test_relay_counted_endpoint_failures_are_harness_errors(agentic):
    agentic["relay"] = {"upstream_errors": {"connection": 2, "http_503": 1},
                        "requests_refused": 1}
    res = AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1)
    assert res["status"] == "completed_with_errors"
    assert res["endpoint_failures"] == {"upstream_errors": {"connection": 2, "http_503": 1},
                                        "requests_refused": 1}
    assert res["leaderboard_comparable"] is False
    assert ("harness errors during `bfcl generate`: the served endpoint failed 3 relayed model "
            "request(s) (connection: 2, http_503: 1)") in res["leaderboard_comparable_reason"]
    assert res["settings"]["memory"]["status"] == "completed_with_errors"
    assert res["accuracy"] == 50.0          # retried requests: the entries were still scored


def test_failed_requests_are_counted_and_make_the_run_completed_with_errors(agentic):
    agentic["rows"] = {"memory_kv": [
        {"id": "memory_kv_0", "result": "Error during inference: Connection error."},
        {"id": "memory_kv_1", "result": [["ok"]]}]}
    res = AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1)
    assert res["status"] == "completed_with_errors" and res["failed_requests"] == 1
    assert res["bfcl_report"]["inference_errors"] == {
        "memory_kv": {"failed": 1, "timed_out": 0, "over_context": 0}}
    # the failed entry is a harness error, left out of the score: memory_kv 1/1, the others 1/2
    assert res["category_accuracy"]["memory_kv"] == {"correct": 1, "total": 1,
                                                     "accuracy": 100.0}
    assert res["summary_memory"] == round((100.0 + 50.0 + 50.0) / 3, 2)
    # the Memory setting alone has no Agentic score: the pooled accuracy is the headline
    assert res["overall_accuracy_agentic"] is None
    assert res["accuracy"] == res["accuracy_pooled"] == round(100.0 * 3 / 5, 2)
    assert res["settings"]["memory"]["failed_requests"] == 1
    assert res["scoring_excluded"] == 1
    assert res["request_harness_errors"] == {"endpoint_failure": 1}


def test_a_run_whose_every_entry_failed_withholds_its_score(agentic):
    agentic["relay"] = {"upstream_errors": {"connection": 6}, "requests_refused": 0}
    agentic["rows"] = {c: [{"id": f"{c}_{i}", "result": "Error during inference: Connection "
                                                         "error."} for i in range(2)]
                       for c in AG.MEMORY_CATEGORIES}
    agentic["scores"] = {c: {"correct": 0, "total": 2, "accuracy": 0.0}
                         for c in AG.MEMORY_CATEGORIES}
    res = AG.run_bfcl_v4_agentic("m", "http://127.0.0.1:9/v1", concurrency=1)
    assert res["status"] == "failed" and res["accuracy"] is None
    assert res["summary_memory"] is None and res["accuracy_pooled"] is None
    assert "every one of the 6 scored entries failed its request" in \
        res["accuracy_withheld_reason"]
    assert "the served endpoint failed 6 relayed model request(s)" in \
        res["accuracy_withheld_reason"]
    assert "score withheld" in res["leaderboard_comparable_reason"]
    assert res["failed_requests"] == 6


# --------------------------------------------------------------------------- #
# bfcl: relay failures of both parts, and a part whose every entry failed
# --------------------------------------------------------------------------- #
def _mt_trace(i, status="OK", correct=True, cat="multi_turn_base"):
    return {"sample_idx": i, "source_sample_idx": i, "attempt_idx": 0, "category": cat,
            "question_id": f"{cat}_{i}", "is_correct": correct, "status": status,
            "health": "ok" if status == "OK" else "request_failed", "steps": 1}


def _bfcl_stubs(monkeypatch, mt_traces, mt_relay=None, ag=None):
    monkeypatch.setattr(bfcl, "_harness_probe", lambda name, multi, single: (
        bfcl.BFCL_EVAL_RELEASE, [{"id": t["question_id"], "category": t["category"]}
                                 for t in mt_traces]))
    monkeypatch.setattr(bfcl, "endpoint_network", lambda url: None)
    monkeypatch.setattr(bfcl, "run_multi_turn_in_image", lambda *a, **k: {
        "traces": [dict(t) for t in mt_traces], "guard_refusals": [], "harness_error": None,
        "endpoint_relay": mt_relay})
    monkeypatch.setattr(bfcl, "_check_agentic", lambda url, cats: None)
    if ag is not None:
        monkeypatch.setattr(AG, "run_bfcl_v4_agentic", lambda *a, **k: ag)


def _ag_result(status="success", failed=0, relays=(), withheld=None):
    cats = list(AG.MEMORY_CATEGORIES)
    # a withheld part left every entry out of the score (harness errors): no category in it
    return {"status": status, "category_accuracy": {} if withheld else {
        c: {"correct": 1, "total": 2, "accuracy": 50.0} for c in cats},
        "categories_unscored": cats if withheld else [],
        "total_questions": 2 * len(cats), "scoring_excluded": 2 * len(cats) if withheld else 0,
        "failed_requests": failed, "timed_out_requests": 0,
        "over_context_prompts": 0, "harness_errors": [], "sample_traces": [],
        "settings": {}, "leaderboard_comparable": False,
        "leaderboard_comparable_reason": "--eval-categories subset",
        "temperature": 0.001, "temperature_source": "official",
        "accuracy_withheld_reason": withheld,
        "bfcl_report": {"relays": list(relays)}}


def test_multi_turn_relay_failures_make_bfcl_completed_with_errors(monkeypatch):
    _bfcl_stubs(monkeypatch, [_mt_trace(0), _mt_trace(1, correct=False)],
                mt_relay={"upstream_errors": {"http_500": 2}, "requests_refused": 0})
    res = bfcl.run_bfcl("m", "http://127.0.0.1:8000/v1", 1, eval_categories="multi_turn_base")
    assert res["status"] == "completed_with_errors"
    assert res["endpoint_failures"] == {"upstream_errors": {"http_500": 2}, "requests_refused": 0}
    assert res["multi_turn_report"]["endpoint_failures"] == res["endpoint_failures"]
    assert ("harness errors during the multi-turn part: the served endpoint failed 2 relayed "
            "model request(s) (http_500: 2)") in res["leaderboard_comparable_reason"]
    assert res["category_accuracy"]["multi_turn_base"] == {"correct": 1, "total": 2,
                                                           "accuracy": 50.0}


def test_bfcl_sums_the_endpoint_failures_of_both_parts(monkeypatch):
    ag = _ag_result(relays=[{"command": "generate", "gemini_relay": None, "endpoint_relay": {
        "upstream_errors": {"connection": 1}, "requests_refused": 2}}])
    _bfcl_stubs(monkeypatch, [_mt_trace(0)],
                mt_relay={"upstream_errors": {"connection": 3}, "requests_refused": 0}, ag=ag)
    res = bfcl.run_bfcl("m", "http://127.0.0.1:8000/v1", 1,
                        eval_categories="multi_turn_base,memory")
    assert res["endpoint_failures"] == {"upstream_errors": {"connection": 4},
                                        "requests_refused": 2}
    assert res["status"] == "completed_with_errors"


def test_a_multi_turn_part_whose_every_entry_failed_is_left_unscored(monkeypatch):
    _bfcl_stubs(monkeypatch, [_mt_trace(0, "FAILED", False), _mt_trace(1, "TIMEOUT", False)],
                ag=_ag_result())
    res = bfcl.run_bfcl("m", "http://127.0.0.1:8000/v1", 1,
                        eval_categories="multi_turn_base,memory")
    assert "multi_turn_base" not in res["category_accuracy"]
    [err] = res["harness_errors"]
    assert err["part"] == "the multi-turn part" and err["categories_unscored"] == [
        "multi_turn_base"]
    assert "FAILED 1, TIMEOUT 1" in err["error"]
    assert res["failed_requests"] == 2 and res["status"] == "completed_with_errors"
    assert res["accuracy"] is not None     # the memory categories were scored


def test_a_withheld_agentic_part_is_left_unscored(monkeypatch):
    ag = _ag_result(status="failed", failed=6,
                    withheld="every one of the 6 scored entries failed its request")
    _bfcl_stubs(monkeypatch, [_mt_trace(0)], ag=ag)
    res = bfcl.run_bfcl("m", "http://127.0.0.1:8000/v1", 1,
                        eval_categories="multi_turn_base,memory")
    assert not any(c in res["category_accuracy"] for c in AG.MEMORY_CATEGORIES)
    [err] = res["harness_errors"]
    assert err["part"] == "the agentic part"
    assert err["categories_unscored"] == sorted(AG.MEMORY_CATEGORIES)
    assert res["agentic_report"]["accuracy_withheld_reason"].startswith("every one of the 6")
    assert res["status"] == "completed_with_errors"
    assert res["category_accuracy"]["multi_turn_base"]["correct"] == 1


def test_a_bfcl_run_where_nothing_was_scored_withholds_its_score(monkeypatch):
    ag = _ag_result(status="failed", failed=6,
                    withheld="every one of the 6 scored entries failed its request")
    _bfcl_stubs(monkeypatch, [_mt_trace(0, "FAILED", False)], ag=ag)
    res = bfcl.run_bfcl("m", "http://127.0.0.1:8000/v1", 1,
                        eval_categories="multi_turn_base,memory")
    assert res["status"] == "failed" and res["accuracy"] is None
    assert res["accuracy_withheld_reason"].startswith("nothing measured the model")
    assert "score withheld" in res["leaderboard_comparable_reason"]


# --------------------------------------------------------------------------- #
# every error of the agentic part names the suite that runs
# --------------------------------------------------------------------------- #
def test_the_limit_id_selection_runs_under_the_bfcl_label(agentic, monkeypatch):
    seen = []

    def run_in_image(eval_name, module, func, kwargs, **kw):
        seen.append((eval_name, module, func))
        raise RuntimeError("no image")
    monkeypatch.setattr(AG, "run_in_image", run_in_image)
    with AG.suite_context("bfcl", "docs/evals/bfcl.md"):
        with pytest.raises(RuntimeError) as err:
            AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1, limit=3)
    assert seen == [("bfcl", "bfcl_v4_agentic", "_select_limited_ids")]
    assert str(err.value).startswith("bfcl: could not build the id file")
    assert "docs/evals/bfcl.md" in str(err.value)


def test_a_failed_generate_names_bfcl_and_its_doc(agentic, monkeypatch):
    monkeypatch.setattr(AG, "_run_cli", lambda *a, **k: (2, "boom"))
    with AG.suite_context("bfcl", "docs/evals/bfcl.md"):
        with pytest.raises(RuntimeError) as err:
            AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    assert str(err.value).startswith("bfcl: `bfcl generate` failed (rc=2)")
    assert "(see docs/evals/bfcl.md)" in str(err.value)
    assert "bfcl_v4_agentic" not in str(err.value)


def test_a_stopped_generate_names_bfcl(agentic, monkeypatch):
    monkeypatch.setenv("GBENCH_BFCL_TIMEOUT_S", "5")
    monkeypatch.setattr(AG, "run_in_image", lambda *a, **k: {c: 2 for c in AG.MEMORY_CATEGORIES})

    def cli(args, env, timeout, model="", backend=None):
        raise subprocess.TimeoutExpired(cmd="bfcl", timeout=timeout)
    monkeypatch.setattr(AG, "_run_cli", cli)
    with AG.suite_context("bfcl", "docs/evals/bfcl.md"):
        res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    [err] = res["harness_errors"]
    assert err["error"].startswith("bfcl: `bfcl generate` did not finish")
    # no entry finished: nothing measured the model, the score is withheld
    assert res["status"] == "failed" and res["accuracy"] is None


def test_the_web_search_errors_name_bfcl():
    with AG.suite_context("bfcl", "docs/evals/bfcl.md"):
        e = AG.SearchUnavailable("GEMINI_API_KEY is unset", "before generation")
        entry = AG.web_search_infra_required(list(AG.WEB_SEARCH_CATEGORIES), "no key",
                                             "before generation")
    assert str(e).startswith("[INFRA REQUIRED] bfcl:") and "'docs/evals/bfcl.md'" in str(e)
    assert entry["error"].startswith("[INFRA REQUIRED] bfcl:")
    assert "'docs/evals/bfcl.md'" in entry["error"]
    # bfcl names itself outside the block too; the track alone keeps its own label
    outside = AG.web_search_infra_required(["web_search_base"], "no key", "before generation",
                                           "bfcl")
    assert "'docs/evals/bfcl.md'" in outside["error"]
    alone = AG.web_search_infra_required(["web_search_base"], "no key", "before generation")
    assert alone["error"].startswith("[INFRA REQUIRED] bfcl_v4_agentic:")
    assert f"'{AG.DOCS_URL}'" in alone["error"]


# --------------------------------------------------------------------------- #
# the web search asks gbench's one Gemini cascade
# --------------------------------------------------------------------------- #
def test_the_search_models_are_the_unified_cascade(monkeypatch):
    assert AG.search_models() == list(base.GEMINI_CASCADE)
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-3.6-flash,gemini-3.5-flash")
    assert AG.search_models() == ["gemini-3.6-flash", "gemini-3.5-flash"]
    assert AG.search_substitution()["models_source"] == "GBENCH_JUDGE_MODELS"
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "gemini-3.7-flash")
    assert AG.search_models() == ["gemini-3.7-flash"]
    assert AG.search_substitution()["models"] == ["gemini-3.7-flash"]
    assert AG.search_substitution()["models_source"] == "GBENCH_SEARCH_MODELS"


def test_the_web_search_container_gets_the_cascade_the_relay_forwards(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaREALgeminiKEY0123456789abcdef")
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-3.6-flash,gemini-3.5-flash")
    monkeypatch.setattr(AG, "_metadata_gate", lambda *a, **k: None)
    monkeypatch.setattr(AG, "_bridge_gateway", lambda *a, **k: "127.0.0.1")
    monkeypatch.setattr(AG, "_require_relay_reachable", lambda *a, **k: None)
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC.container_url(url))
    seen = {}

    def launch(cmd, env, timeout, name, stream=False):
        seen.update(cmd=list(cmd), env=dict(env))
        return 0, ""
    monkeypatch.setattr(AG, "_launch", launch)
    relays = []
    env = {"BFCL_PROJECT_ROOT": str(tmp_path), "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
           "GEMINI_API_KEY": "AIzaREALgeminiKEY0123456789abcdef",
           AG.SEARCH_LOG_ENV: str(tmp_path / "s.jsonl")}
    with AG.suite_context("bfcl_v4_agentic", AG.DOCS_URL, relays=relays):
        AG._run_cli(["generate", "--model", "m"], env, None, model="m")
    held = _forwarded(seen["cmd"], seen["env"])
    assert held["GBENCH_SEARCH_MODELS"] == "gemini-3.6-flash,gemini-3.5-flash"
    assert held["GBENCH_JUDGE_MODELS"] == "gemini-3.6-flash,gemini-3.5-flash"
    assert relays[0]["gemini_relay"]["native_models"] == ["gemini-3.5-flash", "gemini-3.6-flash"]
    assert "GBENCH_JUDGE_MODELS" in AG._CHILD_ENV_NAMES
    assert "GBENCH_SEARCH_MODELS" in AG._CHILD_ENV_NAMES


def test_the_container_default_equals_the_base_cascade():
    from gbench.runners.eval_suites import search_tool
    assert search_tool._DEFAULT_SEARCH_CASCADE == list(base.GEMINI_CASCADE)
    assert base.gemini_cascade_env()["GBENCH_SEARCH_MODELS"] == ",".join(base.GEMINI_CASCADE)


def test_the_models_that_answered_are_recorded(agentic, monkeypatch):
    monkeypatch.setattr(AG, "search_backend", lambda: (AG.SEARCH_BACKEND, ""))
    monkeypatch.setattr(AG, "gemini_api_keys", lambda: ["AIzaREALgeminiKEY0123456789abcdef"])
    monkeypatch.setattr(AG, "check_bfcl_prerequisites",
                        lambda url: (True, "", list(AG.AGENTIC_CATEGORIES)))
    agentic["log"] = [{"backend": "gemini:gemini-3.8-flash", "results": 3, "error": None},
                      {"backend": "gemini:gemini-3.8-flash", "results": 1, "error": None},
                      {"backend": "gemini:gemini-3.7-flash", "results": 2, "error": None}]
    before = base.gemini_answer_counts()
    res = AG.run_bfcl_v4_agentic("m", "http://x/v1", concurrency=1)
    assert res["gemini_models_answered"] == {"gemini-3.7-flash": 1, "gemini-3.8-flash": 2}
    assert res["gemini_cascade"] == list(base.GEMINI_CASCADE)
    after = base.gemini_answer_counts()
    assert after.get("gemini-3.8-flash", 0) - before.get("gemini-3.8-flash", 0) == 2
    assert after.get("gemini-3.7-flash", 0) - before.get("gemini-3.7-flash", 0) == 1


def test_bfcl_carries_the_models_that_answered(monkeypatch):
    ag = _ag_result()
    ag.update(gemini_models_answered={"gemini-3.8-flash": 4}, gemini_cascade=["gemini-3.8-flash"])
    _bfcl_stubs(monkeypatch, [_mt_trace(0)], ag=ag)
    res = bfcl.run_bfcl("m", "http://127.0.0.1:8000/v1", 1,
                        eval_categories="multi_turn_base,memory")
    assert res["gemini_models_answered"] == {"gemini-3.8-flash": 4}
    assert res["gemini_cascade"] == ["gemini-3.8-flash"]


def test_a_cascade_model_the_key_is_not_served_blocks_the_setting(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaREALgeminiKEY0123456789abcdef")
    reason = (f"{base.MODELS_NOT_SERVED_PREFIX} 'gemini-3.8-flash' from gbench's default Gemini "
              "cascade (base.GEMINI_CASCADE) (HTTP 404, not found) for this key. Set "
              "GBENCH_JUDGE_MODELS (comma separated, tried in order) to models the key serves, "
              "or correct the name")
    monkeypatch.setattr(AG, "gemini_key_live_valid", lambda key: (False, reason))
    backend, why = AG.search_backend()
    assert backend is None and why.startswith(base.MODELS_NOT_SERVED_PREFIX)
    assert "rejected the key" not in why and "GBENCH_JUDGE_MODELS" in why
    assert "AIzaREALgeminiKEY" not in why


def test_a_refused_base_url_is_not_called_a_rejected_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaREALgeminiKEY0123456789abcdef")
    monkeypatch.setattr(AG, "gemini_key_live_valid", lambda key: (
        False, "GEMINI_OPENAI_BASE_URL is not the Gemini API (https://generativelanguage."
               "googleapis.com/...) or a loopback address, so the Gemini key is not sent to it; "
               "unset it"))
    backend, why = AG.search_backend()
    assert backend is None and why.startswith("the Gemini key cannot be used")
    assert "rejected" not in why


# --------------------------------------------------------------------------- #
# the endpoint check is told when the container gets a proxy
# --------------------------------------------------------------------------- #
def _gate_kwargs(monkeypatch):
    seen = []

    def gate(eval_name, image, url, docs_url, **kw):
        seen.append((eval_name, url, docs_url, kw))
        return TC.container_url(url)
    monkeypatch.setattr(TC, "require_endpoint_network", gate)
    return seen


def test_the_endpoint_gate_is_proxy_aware_only_with_a_proxy(monkeypatch):
    seen = _gate_kwargs(monkeypatch)
    AG.endpoint_gate("bfcl_v4_agentic", "https://models.example.com/v1", AG.DOCS_URL)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp.example:3128")
    AG.endpoint_gate("bfcl_v4_agentic", "https://models.example.com/v1", AG.DOCS_URL)
    assert [s[3] for s in seen] == [{}, {"proxied": True}]


def test_the_proxied_check_resolves_the_proxy_not_the_endpoint(monkeypatch):
    """With proxied=True, tau_common resolves the proxy's host from the bridge network, not the
    endpoint's, as it does for tau2 / tau3."""
    resolved = []
    from gbench.runners.eval_suites import container_guard as CG
    monkeypatch.setattr(CG, "require_metadata_blocked",
                        lambda eval_name, image, network, docs_url=None, *, resolve=():
                        resolved.append(list(resolve)))
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp.example:3128")
    assert AG.endpoint_gate("bfcl", "https://models.example.com/v1", "docs/evals/bfcl.md") \
        == "https://models.example.com/v1"
    assert resolved == [["proxy.corp.example"]]


def test_the_multi_turn_check_uses_the_same_gate(monkeypatch):
    seen = _gate_kwargs(monkeypatch)
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.corp.example:3128")
    bfcl.endpoint_network("http://127.0.0.1:8000/v1")
    assert seen == [("bfcl", "http://127.0.0.1:8000/v1", "docs/evals/bfcl.md",
                     {"proxied": True})]


# --------------------------------------------------------------------------- #
# docs match the code
# --------------------------------------------------------------------------- #
def _flat(rel):
    return " ".join((REPO / rel).read_text(encoding="utf-8").split())


@pytest.mark.parametrize("rel", ["docs/evals/bfcl_v4_agentic.md", "docs/evals/bfcl.md"])
def test_the_guides_state_the_relay_limit_the_dotenv_rule_and_endpoint_failures(rel):
    text = _flat(rel)
    assert text.isascii()
    assert "plus 60 s, and at least 3600 s" in text
    assert "timeout plus 60 s." not in text and "(see Bounds) plus 60 s." not in text
    for needle in (".env", "endpoint_failures", "upstream_errors", "requests_refused",
                   "gemini_models_answered", "accuracy_withheld_reason"):
        assert needle in text, (rel, needle)
    for model in base.GEMINI_CASCADE:
        assert f"`{model}`" in text, (rel, model)
    for gone in ("gemini-2.5-flash", "gemini-3-flash-preview"):
        assert gone not in text, (rel, gone)


def test_the_relay_limit_in_the_docs_is_the_code():
    from gbench.runners.eval_suites.swebench_common import EndpointRelay
    assert EndpointRelay.TIMEOUT_S == 3600


def test_the_agentic_guide_names_the_new_records():
    text = _flat("docs/evals/bfcl_v4_agentic.md")
    for needle in ("bfcl_report.inference_errors", "models_source", "gemini_cascade",
                   "_disable_dotenv", "GBENCH_JUDGE_MODELS", "base.search_cascade()"):
        assert needle in text, needle
