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

"""The evals and quality runners: which endpoint and served id they use, the stamps every eval
result carries (served id, run shape, shard, provenance, unapplied flags), and --skip-existing's
reuse rules. No live server: a stub HTTP server on port 0 plays /v1/models, suites are stubs."""

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import pytest

from gbench.core.config import BenchmarkConfig
from gbench.core.models import ModelConfig, ModelFormat
from gbench.runners import evals as E
from gbench.runners import quality as Q
from gbench.runners.eval_suites import SUITES
from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class _Models:
    """A stub /v1/models: answers `listing` (a dict, or (status, body)) and records requests."""

    def __init__(self, listing):
        self.listing, self.requests = listing, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                return

            def do_GET(self):
                outer.requests.append({"path": self.path,
                                       "headers": {k.lower(): v for k, v in self.headers.items()}})
                status, body = (outer.listing if isinstance(outer.listing, tuple)
                                else (200, outer.listing))
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()


@pytest.fixture
def served():
    stubs = []

    def make(listing):
        stubs.append(_Models(listing))
        return stubs[-1]
    yield make
    for s in stubs:
        s.close()


def _listing(*ids, windows=None):
    windows = windows or {}
    return {"object": "list", "data": [{"id": i, "object": "model",
                                        **({"max_model_len": windows[i]} if i in windows else {})}
                                       for i in ids]}


def _model():
    return ModelConfig(short_name="m", hf_model_id="org/m", total_params_b=1.0, is_moe=False)


def _cfg(tmp_path, base_dir="results", **kw):
    config = BenchmarkConfig()
    config.evals = ["gsm8k"]
    config.batch_sizes = [1]
    config.skip_existing = False
    config.eval_max_output_tokens = 256
    config.results_dir = tmp_path / base_dir
    for k, v in kw.items():
        setattr(config, k, v)
    config.initialize()
    return config


def _ok(**kw):
    return {"status": "success", "total_questions": 2, "correct_answers": 1, "accuracy": 50.0,
            "category_accuracy": {}, "thinking": kw.get("enable_thinking"),
            "temperature": 0.0, "max_tokens_sent": [256]}


def _run(config, suites, model=None, listing=None):
    """EvalsBenchmarkRunner.run with `suites` stubbed; /v1/models answered by `listing` (the
    served ids) unless the endpoint is a real stub server."""
    model = model or _model()
    patches = [mock.patch.dict(SUITES, suites),
               mock.patch("gbench.runners.eval_suites.discover_and_register_plugins",
                          return_value={})]
    if listing is not None:
        patches.append(mock.patch("gbench.utils.endpoint.list_served_models",
                                  lambda base_url, timeout=5.0, headers=None: listing))
    for p in patches:
        p.start()
    try:
        return E.EvalsBenchmarkRunner(config).run(model, ModelFormat.HF)
    finally:
        for p in reversed(patches):
            p.stop()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("GBENCH_SHARD", "GBENCH_RESULTS_DIR", "VLLM_API_KEY", "OPENAI_API_KEY",
                 "GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    E._SERVED_MODEL.clear()


LOCAL = [{"id": "org/m", "max_model_len": 32768}]


# --------------------------------------------------------------------------- #
# the suites a selection runs
# --------------------------------------------------------------------------- #
def _names(**kw):
    cfg = type("C", (), {"evals": None, "eval_plugins_dir": None, "eval_custom_jsonl": None})()
    for k, v in kw.items():
        setattr(cfg, k, v)
    with mock.patch("gbench.runners.eval_suites.discover_and_register_plugins",
                    return_value={"my_plugin": object()}):
        return E.resolve_eval_names(cfg)


def test_the_comma_form_and_the_list_form_name_the_same_suites():
    assert _names(evals=["gpqa,mmlu"]) == _names(evals=["gpqa", "mmlu"]) == ["gpqa", "mmlu"]


def test_all_is_the_builtin_list_plus_the_discovered_plugins_and_custom_jsonl_is_added():
    assert _names(evals=["all"]) == E.builtin_all_suites() + ["my_plugin"]
    assert "swe_lancer" not in _names(evals=["all"])
    assert _names(evals=["plugins"]) == ["my_plugin"]
    assert _names(evals=["gpqa"], eval_custom_jsonl="/x.jsonl") == ["gpqa", "custom_jsonl"]


def test_an_old_suite_name_resolves_to_the_new_one():
    assert _names(evals=["culer"]) == ["longbench_v2_code"]


def test_eval_concurrency_defaults_to_one_and_never_goes_below_it():
    cfg = type("C", (), {})()
    assert E.eval_concurrency(cfg) == 1
    cfg.batch_sizes = [4]
    assert E.eval_concurrency(cfg) == 4
    cfg.eval_concurrency = 16
    assert E.eval_concurrency(cfg) == 16
    cfg.eval_concurrency = 0
    assert E.eval_concurrency(cfg) == 1


def test_a_sandboxed_suite_runs_sandboxes_else_batch_sizes_and_never_zero(tmp_path):
    seen = {}
    config = BenchmarkConfig()
    config.batch_sizes = [3]
    config.eval_concurrency = 16
    runner = E.EvalsBenchmarkRunner(config)
    with mock.patch.dict(SUITES, {"terminal_bench": lambda **k: seen.update(tb=k["concurrency"]),
                                  "gsm8k": lambda **k: seen.update(gsm=k["concurrency"])}):
        runner._run_single_eval("terminal_bench", "m", "http://127.0.0.1:1/v1", num_threads=16)
        runner._run_single_eval("gsm8k", "m", "http://127.0.0.1:1/v1", num_threads=16)
        assert seen == {"tb": 3, "gsm": 16}       # the CLI guard checks --batch-sizes[0]
        config.sandboxes, config.batch_sizes = 0, [0]
        runner._run_single_eval("terminal_bench", "m", "http://127.0.0.1:1/v1", num_threads=0)
        runner._run_single_eval("gsm8k", "m", "http://127.0.0.1:1/v1", num_threads=0)
    assert seen == {"tb": 1, "gsm": 1}


# --------------------------------------------------------------------------- #
# the served model
# --------------------------------------------------------------------------- #
Q_REMOTE = E.ServedModelQuery(("org/m", "m"))


def test_one_served_id_that_matches_is_used_with_its_window(served):
    ep = served(_listing("org/m", windows={"org/m": 8192}))
    out = E.served_model_for(ep.url + "/v1", Q_REMOTE)
    assert (out["id"], out["max_model_len"]) == ("org/m", 8192)
    assert E._server_max_model_len(ep.url + "/v1") == 8192


def test_several_served_ids_take_the_match_and_its_window_not_the_first(served):
    ep = served(_listing("other/big", "org/m", windows={"other/big": 4096, "org/m": 65536}))
    assert E._resolve_remote_model_id(ep.url + "/v1", Q_REMOTE) == "org/m"
    assert E._server_max_model_len(ep.url + "/v1") == 65536


def test_several_served_ids_without_a_match_is_an_error_naming_served_model_id(served):
    ep = served(_listing("a/x", "b/y"))
    with pytest.raises(E.ServedModelError, match="--served-model-id") as err:
        E.served_model_for(ep.url + "/v1", Q_REMOTE)
    assert "a/x" in str(err.value) and "b/y" in str(err.value)


def test_served_model_id_must_be_listed_and_wins(served):
    ep = served(_listing("org/m", "org/m-lora"))
    q = E.ServedModelQuery(("org/m",), override="org/m-lora")
    assert E.served_model_for(ep.url + "/v1", q)["id"] == "org/m-lora"
    with pytest.raises(E.ServedModelError, match="not served"):
        E.served_model_for(ep.url + "/v1", E.ServedModelQuery(("org/m",), override="nope"))


def test_a_local_server_serving_another_model_is_a_foreign_server(served):
    ep = served(_listing("someone/else"))
    with pytest.raises(E.ServedModelError, match="none of which"):
        E.served_model_for(ep.url + "/v1", E.ServedModelQuery(("org/m",), local=True))


def test_a_remote_listing_failure_asks_for_the_configured_id_and_says_so(served):
    ep = served((404, {"error": "no"}))
    out = E.served_model_for(ep.url + "/v1", Q_REMOTE)
    assert out["id"] == "org/m" and out["how"].startswith("configured (listing failed")
    with pytest.raises(E.ServedModelError, match="could not list"):
        E.served_model_for(ep.url + "/v1", E.ServedModelQuery(("org/m",), local=True))


def test_the_listing_sends_the_endpoint_key(served, monkeypatch):
    ep = served(_listing("org/m"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-endpoint-alias-key")
    E.served_model_for(ep.url + "/v1", Q_REMOTE)
    monkeypatch.setenv("VLLM_API_KEY", "vllm-endpoint-key")
    E._SERVER_MAX_LEN.clear()
    E._server_max_model_len(ep.url + "/v1")
    assert [r["headers"].get("authorization") for r in ep.requests] == [
        "Bearer sk-endpoint-alias-key", "Bearer vllm-endpoint-key"]


def test_a_multi_model_endpoint_without_a_match_runs_no_suite(tmp_path, served):
    ep = served(_listing("a/x", "b/y"))
    ran = []
    config = _cfg(tmp_path, remote_endpoint=ep.url + "/v1", evals=["gsm8k", "mmlu"])
    out = _run(config, {"gsm8k": lambda **k: ran.append(1), "mmlu": lambda **k: ran.append(1)})
    assert ran == [] and [r["eval_name"] for r in out] == ["gsm8k", "mmlu"]
    assert all(r["failed"] and r["accuracy"] is None and "--served-model-id" in r["error"]
               for r in out)


def test_every_result_records_the_served_id_and_how_it_was_chosen(tmp_path, served):
    ep = served(_listing("a/x", "org/m"))
    config = _cfg(tmp_path, remote_endpoint=ep.url, evals=["gsm8k", "mmlu"])   # no /v1: normalized

    def boom(**kw):
        raise RuntimeError("harness crash")
    seen = {}
    out = _run(config, {"gsm8k": lambda **k: seen.update(k) or _ok(**k), "mmlu": boom})
    assert seen["model_name"] == "org/m" and seen["base_url"] == ep.url + "/v1"
    for r in out:
        assert r["served_model_id"] == "org/m" and "matches 'org/m'" in r["served_model_id_source"]
        assert r["model_name"] == "m"


def test_the_local_endpoint_is_used_and_no_server_is_started(tmp_path, served):
    ep = served(_listing("org/m", windows={"org/m": 16384}))
    config = _cfg(tmp_path, local_endpoint=ep.url + "/v1")
    seen = {}
    with mock.patch("gbench.runners.serving.ServingBenchmarkRunner._start_server",
                    side_effect=AssertionError("no server may start")):
        out = _run(config, {"gsm8k": lambda **k: seen.update(k) or _ok(**k)})
    assert seen["base_url"] == ep.url + "/v1" and out[0]["run_shape"]["served_max_model_len"] == \
        16384


def test_a_direct_caller_without_an_endpoint_gets_a_native_server(tmp_path):
    config = _cfg(tmp_path)
    calls = []

    def start(self, model, format, context="perf"):
        calls.append(context)
        self.server_port = 1
        return True
    with mock.patch("gbench.runners.serving.ServingBenchmarkRunner._start_server", start), \
         mock.patch("gbench.runners.serving.ServingBenchmarkRunner._cleanup_server",
                    lambda self: calls.append("cleanup")):
        out = _run(config, {"gsm8k": _ok}, listing=LOCAL)
    assert calls == ["native", "cleanup"] and out[0]["status"] == "success"


# --------------------------------------------------------------------------- #
# shard stamps
# --------------------------------------------------------------------------- #
def test_a_malformed_shard_fails_every_suite_before_anything_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "0/4")
    ran = []
    config = _cfg(tmp_path, remote_endpoint="http://127.0.0.1:1/v1")
    with mock.patch("gbench.utils.endpoint.list_served_models",
                    side_effect=AssertionError("no listing")):
        out = _run(config, {"gsm8k": lambda **k: ran.append(1)})
    assert ran == [] and out[0]["failed"] and "GBENCH_SHARD" in out[0]["error"]


def test_a_partial_shard_is_never_comparable_and_names_its_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "2/3")

    def sharded(**kw):
        return {**_ok(**kw), "shard": "2/3", "leaderboard_comparable": True}
    config = _cfg(tmp_path, remote_endpoint="http://127.0.0.1:1/v1", evals=["gsm8k", "mmlu"])
    out = _run(config, {"gsm8k": sharded, "mmlu": _ok}, listing=[{"id": "org/m"}])
    by = {r["eval_name"]: r for r in out}
    assert by["gsm8k"]["leaderboard_comparable"] is False
    assert "sharded run (--shard 2/3)" in by["gsm8k"]["leaderboard_comparable_reason"]
    assert by["gsm8k"]["shard_applied"] is True and by["gsm8k"]["run_shape"]["shard"] == "2/3"
    # a suite that ran its full set says so
    assert by["mmlu"]["shard_applied"] is False and by["mmlu"]["shard_requested"] == "2/3"
    saved = json.loads((config.log_manager.results_dir / "eval_gsm8k_m_hf.shard-2of3.json")
                       .read_text())
    assert saved["leaderboard_comparable"] is False and saved["eval_limit"] is None


def test_a_failed_suite_records_the_requested_shard_only(tmp_path, monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "1/2")

    def boom(**kw):
        raise RuntimeError("crash")
    config = _cfg(tmp_path, remote_endpoint="http://127.0.0.1:1/v1")
    [r] = _run(config, {"gsm8k": boom}, listing=[{"id": "org/m"}])
    assert r["failed"] and r["shard_requested"] == "1/2" and "shard_applied" not in r


def test_no_shard_leaves_comparability_alone_and_records_the_limit(tmp_path):
    config = _cfg(tmp_path, remote_endpoint="http://127.0.0.1:1/v1", eval_limit=7)
    [r] = _run(config, {"gsm8k": lambda **k: {**_ok(**k), "leaderboard_comparable": True}},
               listing=[{"id": "org/m"}])
    assert r["leaderboard_comparable"] is True and r["eval_limit"] == 7
    assert "shard_applied" not in r
    assert (config.log_manager.results_dir / "eval_gsm8k_m_hf.json").exists()


# --------------------------------------------------------------------------- #
# flags a suite did not record applying, provenance
# --------------------------------------------------------------------------- #
def test_unrecorded_flags_get_a_note_and_recorded_ones_do_not(tmp_path):
    config = _cfg(tmp_path, remote_endpoint="http://127.0.0.1:1/v1", evals=["gsm8k", "mmlu"],
                  attempt_count=4, temperature=0.6, eval_thinking=True)

    def bare(**kw):
        return {"status": "success", "total_questions": 1, "correct_answers": 1,
                "accuracy": 100.0, "category_accuracy": {}, "attempts": 2}

    def full(**kw):
        return {**_ok(**kw), "max_tokens_sent": None,
                "attempts": {"attempts_requested": 4, "attempts_per_sample": 4}}
    out = {r["eval_name"]: r for r in _run(config, {"gsm8k": bare, "mmlu": full},
                                           listing=[{"id": "org/m"}])}
    b, f = out["gsm8k"], out["mmlu"]
    assert b["max_output_tokens_note"] == ("--max-output-tokens 256: this suite did not record "
                                           "applying it")
    assert b["attempts"] == {"attempts_requested": 4, "attempts_per_sample": 2,
                             "not_applied": "gsm8k: its harness runs a fixed number of trials per "
                                            "task; --attempt-count 4 was not applied"}
    assert b["temperature_applied"] is False and b["thinking_applied"] is False
    for key in ("max_output_tokens_note", "temperature_applied", "thinking_applied"):
        assert key not in f, key                       # a key present (even None) is recorded
    assert f["attempts"] == {"attempts_requested": 4, "attempts_per_sample": 4}


def test_every_result_carries_the_evals_scaffold_and_the_gbench_commit(tmp_path):
    config = _cfg(tmp_path, remote_endpoint="http://127.0.0.1:1/v1")
    [r] = _run(config, {"gsm8k": _ok}, listing=[{"id": "org/m"}])
    from gbench.core.scaffold import Pillar, build_condition
    cond = build_condition(Pillar.EVALS, _model(), ModelFormat.HF, config=config)
    assert r["scaffold_id"] == cond.scaffold.scaffold_id and r["scaffold_id"]
    assert r["subject_id"] == cond.subject.subject_id
    assert set(r["gbench"]) == {"commit", "dirty"}


def test_a_plugin_dataset_gap_is_infra_required_not_a_zero(tmp_path):
    class DatasetUnavailableError(Exception):
        pass

    def gap(**kw):
        raise DatasetUnavailableError("the private dataset is not provided")
    config = _cfg(tmp_path, remote_endpoint="http://127.0.0.1:1/v1")
    [r] = _run(config, {"gsm8k": gap}, listing=[{"id": "org/m"}])
    assert r["failed"] is True and r["failure_kind"] == "infra_required"
    assert r["accuracy"] is None and r["status"] == "error"
    assert E.failure_label(r) == "INFRA REQUIRED"


# --------------------------------------------------------------------------- #
# --skip-existing
# --------------------------------------------------------------------------- #
def _first_run(tmp_path, **kw):
    config = _cfg(tmp_path, remote_endpoint="http://127.0.0.1:1/v1", **kw)
    [r] = _run(config, {"gsm8k": _ok}, listing=[{"id": "org/m"}])
    return config, r


def _second_run(tmp_path, **kw):
    ran = []
    config = _cfg(tmp_path, remote_endpoint="http://127.0.0.1:1/v1", skip_existing=True, **kw)
    [r] = _run(config, {"gsm8k": lambda **k: ran.append(1) or {**_ok(**k), "accuracy": 99.0}},
               listing=[{"id": "org/m"}])
    return ran, r


def test_a_result_of_the_same_configuration_is_reused(tmp_path):
    _first_run(tmp_path)
    ran, r = _second_run(tmp_path)
    assert ran == [] and r["accuracy"] == 50.0 and r["reused_from"]


@pytest.mark.parametrize("second", [{"eval_limit": 10}, {"eval_thinking": True},
                                    {"temperature": 0.7}, {"eval_max_output_tokens": 512}])
def test_a_result_of_another_configuration_is_not_reused(tmp_path, second):
    _first_run(tmp_path)
    ran, r = _second_run(tmp_path, **second)
    assert ran == [1] and r["accuracy"] == 99.0


def test_another_shard_is_not_reused(tmp_path, monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "1/2")
    _first_run(tmp_path)
    monkeypatch.setenv("GBENCH_SHARD", "2/2")
    ran, _r = _second_run(tmp_path)
    assert ran == [1]


def test_a_reconciled_suite_is_not_reused(tmp_path, monkeypatch):
    _first_run(tmp_path)
    monkeypatch.setitem(CANONICAL_SYNC, "gsm8k",
                        {**CANONICAL_SYNC["gsm8k"], "synced": "2099-01-01"})
    ran, _r = _second_run(tmp_path)
    assert ran == [1]


def test_a_failed_or_legacy_or_unreadable_result_is_rerun(tmp_path):
    config, r = _first_run(tmp_path)
    path = config.log_manager.results_dir / "eval_gsm8k_m_hf.json"
    for broken in ({**r, "status": "error", "failed": True},
                   {k: v for k, v in r.items() if k != "run_shape"}):
        path.write_text(json.dumps(broken))
        ran, out = _second_run(tmp_path)
        assert ran == [1] and out["accuracy"] == 99.0
        shutil.rmtree(tmp_path / "results")
        config, r = _first_run(tmp_path)
        path = config.log_manager.results_dir / "eval_gsm8k_m_hf.json"
    path.write_text('{"status": "succ')                      # a run killed mid-write
    ran, out = _second_run(tmp_path)
    assert ran == [1] and out["accuracy"] == 99.0             # run, never dropped


def test_the_newest_result_wins_by_its_own_timestamp_not_the_directory_name(tmp_path):
    config, r = _first_run(tmp_path, base_dir="first")
    for stamp, finished, acc in (("12-31-235959", "2025-12-31T23:59:59+00:00", 11.0),
                                 ("01-01-000001", "2026-01-01T00:00:01+00:00", 22.0)):
        d = tmp_path / "results" / stamp
        d.mkdir(parents=True)
        (d / "eval_gsm8k_m_hf.json").write_text(json.dumps({**r, "finished_at": finished,
                                                            "accuracy": acc}))
    ran, out = _second_run(tmp_path)
    assert ran == [] and out["accuracy"] == 22.0


# --------------------------------------------------------------------------- #
# quality: endpoint, served id, window and output cap
# --------------------------------------------------------------------------- #
GEM = "AIzaSyTESTw3bEvalsRouting000000001"


def _quality_run(tmp_path, monkeypatch, listing, model=None, **cfg_kw):
    """A quality run with docker, the image and the suite's outputs stubbed; returns the result,
    what the fake `docker run` saw and whether gbench started a server."""
    monkeypatch.setenv("GEMINI_API_KEY", GEM)
    config = BenchmarkConfig()
    config.results_dir = tmp_path / "qresults"
    config.gemmaclaw_commit = "0123456789abcdef0123456789abcdef01234567"
    for k, v in cfg_kw.items():
        setattr(config, k, v)
    config.initialize()
    runner = Q.QualityBenchmarkRunner(config)
    runner._preflight_check = mock.MagicMock(return_value=True)
    runner._setup_gemmaclaw_repo = mock.MagicMock(return_value="c0ffee")
    runner._start_server = mock.MagicMock(return_value=True)
    runner._cleanup_server = mock.MagicMock()
    runner._remove_container = mock.MagicMock()
    runner._image = "gbench-quality:test"
    runner._image_metadata = lambda image, dest: {"patches": [], "commit": "c0ffee"}
    runner._collect_container_outputs = lambda name: {
        "summary": {"counts": {"total": 2, "passed": 1, "failed": 1}},
        "scenario_summaries": {}, "entrypoint": {"command_exit": 0,
                                                 "forwarder_alive_at_end": True},
        "forwarder_errors": []}
    seen = {}

    def popen(argv, stdin=None, stdout=None, stderr=None, env=None, **kw):
        seen["argv"] = list(argv)
        for i, a in enumerate(argv):
            if a == "--volume" and argv[i + 1].endswith(":/etc/gbench/openclaw.json:ro"):
                seen["config"] = json.loads(Path(argv[i + 1].split(":")[0]).read_text())
        proc = mock.MagicMock()
        proc.poll.return_value = 0
        proc.wait.return_value = 0
        proc.returncode = 0
        return proc
    monkeypatch.setattr(Q.subprocess, "Popen", popen)
    made = {}
    real_init = Q.KeyRelay.__init__

    def init(self, *a, **kw):
        made.update(kw)
        real_init(self, *a, **kw)
    monkeypatch.setattr(Q.KeyRelay, "__init__", init)
    patches = []
    if listing is not None:
        patches.append(mock.patch("gbench.utils.endpoint.list_served_models",
                                  lambda base_url, timeout=5.0, headers=None: listing))
    for p in patches:
        p.start()
    try:
        res = runner.run(model or ModelConfig(short_name="m", hf_model_id="org/m",
                                              total_params_b=1.0, is_moe=False), ModelFormat.HF)
    finally:
        for p in patches:
            p.stop()
    vllm = seen["config"]["models"]["providers"]["vllm"]["models"][0]
    return res, seen, vllm, made, runner


def test_quality_uses_the_local_endpoint_and_starts_no_server(tmp_path, monkeypatch, served):
    ep = served(_listing("other/x", "org/m", windows={"other/x": 4096, "org/m": 40960}))
    res, seen, vllm, made, runner = _quality_run(tmp_path, monkeypatch, None,
                                                 local_endpoint=ep.url + "/v1")
    runner._start_server.assert_not_called()
    assert made["model_upstream"] == ep.url + "/v1"
    assert vllm["id"] == "org/m" and vllm["contextWindow"] == 40960      # the match's window
    assert res["served_model_id"] == "org/m" and res["context_window"] == 40960


def test_quality_asks_for_the_served_id_not_the_models_value(tmp_path, monkeypatch, served):
    ep = served(_listing("org/m", "my-tag"))
    res, seen, vllm, made, _r = _quality_run(tmp_path, monkeypatch, None,
                                             remote_endpoint=ep.url, served_model_id="my-tag")
    assert made["model_id"] == vllm["id"] == res["served_model_id"] == "my-tag"
    assert "vllm/my-tag" in seen["argv"]
    assert made["model_upstream"] == ep.url + "/v1"                     # normalized


def test_quality_refuses_an_endpoint_that_serves_other_models(tmp_path, monkeypatch, served):
    ep = served(_listing("a/x", "b/y"))
    with pytest.raises(RuntimeError, match="does not serve the model under test"):
        _quality_run(tmp_path, monkeypatch, None, remote_endpoint=ep.url + "/v1")


def test_quality_declares_gemmaclaws_self_hosted_output_cap(tmp_path, monkeypatch):
    res, _seen, vllm, _m, _r = _quality_run(
        tmp_path, monkeypatch, [{"id": "org/m", "max_model_len": 32768}],
        remote_endpoint="http://127.0.0.1:1/v1")
    assert vllm["maxTokens"] == Q.GEMMACLAW_SELF_HOSTED_MAX_TOKENS == 8192
    assert res["model_max_tokens"] == 8192 and res["max_output_tokens_note"] is None


def test_quality_refuses_a_window_that_cannot_hold_the_cap_and_notes_the_eval_flag(tmp_path,
                                                                                  monkeypatch):
    # A window no larger than GemmaClaw's declared output cap cannot hold it plus a prompt:
    # refused up front (infra_required), since a clamp to the window would still fail every
    # request.
    with pytest.raises(RuntimeError, match="infra_required: the served context window"):
        _quality_run(tmp_path, monkeypatch, [{"id": "org/m", "max_model_len": 4096}],
                     remote_endpoint="http://127.0.0.1:1/v1", eval_max_output_tokens=65536)
    res, _seen, vllm, _m, _r = _quality_run(
        tmp_path, monkeypatch, [{"id": "org/m", "max_model_len": 16384}],
        remote_endpoint="http://127.0.0.1:1/v1", eval_max_output_tokens=65536)
    assert vllm["maxTokens"] == 8192 and vllm["contextWindow"] == 16384
    assert res["max_output_tokens_note"].startswith("--max-output-tokens 65536 not applied")


def test_the_self_hosted_cap_is_read_from_the_commit_that_runs(tmp_path):
    repo = tmp_path / "gc"
    repo.mkdir()
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_AUTHOR_NAME": "t",
           "GIT_AUTHOR_EMAIL": "t@example.com", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.com"}

    def git(*a):
        return subprocess.run(["git", *a], cwd=repo, env=env, check=True, capture_output=True,
                              text=True).stdout.strip()
    git("init", "-q")
    (repo / "README.md").write_text("x\n")
    git("add", ".")
    git("commit", "-qm", "c1")
    bare = git("rev-parse", "HEAD")
    target = repo / Q._SELF_HOSTED_DEFAULTS_FILE
    target.parent.mkdir(parents=True)
    target.write_text("export const SELF_HOSTED_DEFAULT_MAX_TOKENS = 12288;\n")
    git("add", ".")
    git("commit", "-qm", "c2")
    sha = git("rev-parse", "HEAD")
    assert Q.self_hosted_max_tokens(repo, sha, env)[0] == 12288
    value, source = Q.self_hosted_max_tokens(repo, bare, env)
    assert value == Q.GEMMACLAW_SELF_HOSTED_MAX_TOKENS and "not found" in source


def test_the_pinned_cap_is_the_cached_checkouts_value_when_present():
    """GEMMACLAW_SELF_HOSTED_MAX_TOKENS is the pinned commit's SELF_HOSTED_DEFAULT_MAX_TOKENS
    (checked against a GemmaClaw checkout at that commit when this machine has one)."""
    from gbench.core.config import DEFAULT_GEMMACLAW_COMMIT
    repo = Path(os.environ.get("GBENCH_REAL_GEMMACLAW", "") or
                Path.home() / ".cache" / "gbench" / "gemmaclaw")
    if (repo / ".git").exists():
        value, source = Q.self_hosted_max_tokens(repo, DEFAULT_GEMMACLAW_COMMIT,
                                                 {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null"})
        if "not found" not in source:
            assert value == Q.GEMMACLAW_SELF_HOSTED_MAX_TOKENS
    assert Q.GEMMACLAW_SELF_HOSTED_MAX_TOKENS == 8192


# --------------------------------------------------------------------------- #
# the scaffold's version signal per suite
# --------------------------------------------------------------------------- #
def test_a_suites_version_follows_its_canonical_sync_entry(monkeypatch):
    before = E.suite_scoring_version("gpqa")
    assert before["synced"] == CANONICAL_SYNC["gpqa"].get("synced")
    monkeypatch.setitem(CANONICAL_SYNC, "gpqa", {**CANONICAL_SYNC["gpqa"], "synced": "2099-01-01"})
    assert E.suite_scoring_version("gpqa") != before


def test_a_plugins_version_is_the_hash_of_its_module_and_custom_jsonl_adds_its_file(tmp_path):
    import importlib.util
    mod_path = tmp_path / "my_plugin_mod.py"
    mod_path.write_text("def run_my_plugin(**kw):\n    return {}\n")
    spec = importlib.util.spec_from_file_location("my_plugin_mod", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with mock.patch.dict(SUITES, {"my_plugin": mod.run_my_plugin}):
        first = E.suite_scoring_version("my_plugin")
        mod_path.write_text("def run_my_plugin(**kw):\n    return {'changed': 1}\n")
        second = E.suite_scoring_version("my_plugin")
    assert first["version"].startswith("sha256:") and first != second
    data = tmp_path / "set.jsonl"
    data.write_text('{"q": 1}\n')
    cfg = type("C", (), {"eval_custom_jsonl": str(data)})()
    one = E.suite_scoring_version("custom_jsonl", cfg)
    data.write_text('{"q": 2}\n')
    assert E.suite_scoring_version("custom_jsonl", cfg)["data_sha256"] != one["data_sha256"]
