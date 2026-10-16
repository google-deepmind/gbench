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

"""Run guards, the records a run writes, and the merge rules.

* A result returned with status "skipped" is a failed configuration (infra_required), never a
  0.00% row; swebench_common.skipped_result says so too.
* The summary: a Correct count prints only where the headline is correct/total; a category row
  of a multi-attempt suite counts distinct questions; a suite row without categories is named
  after its traces' category only when they share one; a null headline shows its reason;
  summary.json keeps eval traces in each eval_*.json.
* Guards: --served-model-id on a local run, GBENCH_SERVER_HOST, the served-model match across
  organisations, the tokenizer of --vllm-serve-args, a reused run directory of another node.
* Records: remote serve settings and parallelism are unknown, not CLI defaults; the throughput
  scaffold pins only what the runner reads; MathArena suites record the attempts they run.
* Merge: a suite hook that raises is a refused merge; the over-10% budget rule and the
  server-parser record are recounted over the union.
* canonical_sync: a data .txt is hashed byte for byte; the medxpertqa and bigcodebench methods
  describe the code.

Nothing here touches the network, docker, GPUs or Gemini.
"""

import json
import os
import types
from pathlib import Path
from unittest import mock

import pytest

from gbench import cli
from gbench.core import config as C
from gbench.runners import evals as E

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# a skipped result is a failed configuration
# --------------------------------------------------------------------------- #
def test_a_skipped_result_is_a_failed_infra_required_configuration():
    from gbench.runners.eval_suites.swebench_common import skipped_result
    res = E.mark_unscored({"status": "skipped", "accuracy": 0.0, "total_questions": 0,
                           "skip_reason": "no data"})
    assert res["failed"] is True and res["failure_kind"] == "infra_required"
    assert res["accuracy"] is None and res["accuracy_withheld_reason"]
    assert E.failure_label(res) == "INFRA REQUIRED"
    helper = E.mark_unscored(skipped_result("p", "m", "no data.", "docs/x.md"))
    assert helper["failed"] is True and helper["accuracy"] is None
    assert helper["skip_reason"].startswith("[INFRA REQUIRED]")
    assert cli.eval_failures_exit_code([dict(helper, benchmark_type="eval",
                                             eval_name="p")]) == cli.EXIT_HARNESS_ERROR


# --------------------------------------------------------------------------- #
# the summary
# --------------------------------------------------------------------------- #
def _csv_rows(tmp_path, results):
    import csv
    path = cli._save_eval_summary_csv(tmp_path, results, [], [], {})
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_the_correct_column_prints_only_a_count_the_headline_is_made_of(tmp_path):
    mean = {"eval_name": "docvqa", "model_short": "m", "format": "hf", "status": "success",
            "total_questions": 8, "correct_answers": 7, "accuracy": 87.22,
            "category_accuracy": {"free_text": {"correct": 1, "total": 1, "accuracy": 97.73},
                                  "table": {"correct": 3, "total": 4, "accuracy": 75.0}}}
    rate = {"eval_name": "gsm8k", "model_short": "m", "format": "hf", "status": "success",
            "total_questions": 4, "correct_answers": 3, "accuracy": 75.0}
    rows = {(r["Eval Suite"], r["Subcategory"]): r for r in _csv_rows(tmp_path, [mean, rate])}
    assert rows[("DOCVQA", "OVERALL")]["Correct"] == "-"
    assert rows[("DOCVQA", "free_text")]["Correct"] == "-"          # 1/1 is not 97.73
    assert rows[("DOCVQA", "table")]["Correct"] == "3"               # 3/4 is 75.0
    assert rows[("GSM8K", "-")]["Correct"] == "3"
    assert cli._correct_cell(mean, 8) == "-" and cli._correct_cell(rate, 4) == 3


def test_category_rows_of_a_multi_attempt_suite_count_distinct_questions(tmp_path):
    traces = [{"category": "a", "source_sample_idx": 0, "sample_idx": i} for i in range(3)]
    res = {"eval_name": "gdpval", "model_short": "m", "format": "hf", "status": "success",
           "total_questions": 4, "correct_answers": 2, "accuracy": 50.0, "effective_n": 2,
           "attempts": {"attempts_per_sample": 3},
           "category_accuracy": {"a": {"correct": 2, "total": 3, "accuracy": 66.67},
                                 "b": {"correct": 0, "total": 1, "accuracy": 0.0}},
           "sample_traces": traces + [{"category": "b", "source_sample_idx": 1,
                                       "sample_idx": 3, "scoring_excluded": True}]}
    rows = {r["Subcategory"]: r for r in _csv_rows(tmp_path, [res])}
    assert rows["a"]["Effective N"] == "1" and rows["a"]["Questions"] == "3"
    assert rows["b"]["Effective N"] == "0"


def test_a_row_without_categories_is_named_after_a_category_only_when_all_share_it(tmp_path):
    mixed = {"eval_name": "livebench", "model_short": "m", "format": "hf", "status": "success",
             "total_questions": 2, "correct_answers": 1, "accuracy": 50.0,
             "sample_traces": [{"category": "math"}, {"category": "coding"}]}
    one = dict(mixed, eval_name="t_eval", sample_traces=[{"category": "x"}, {"category": "x"}])
    rows = {r["Eval Suite"]: r for r in _csv_rows(tmp_path, [mixed, one])}
    assert rows["LIVEBENCH"]["Subcategory"] == "OVERALL"
    assert rows["T_EVAL"]["Subcategory"] == "x"


def test_a_success_row_with_no_headline_shows_why(tmp_path):
    res = {"eval_name": "omnidocbench", "model_short": "m", "format": "hf", "status": "success",
           "total_questions": 2, "correct_answers": 0, "accuracy": None,
           "accuracy_withheld_reason": "composite undefined: no display formula"}
    (row,) = _csv_rows(tmp_path, [res])
    assert row["Headline Metrics"] == "accuracy withheld: composite undefined: no display formula"


def test_summary_json_keeps_eval_traces_in_the_eval_files():
    res = {"benchmark_type": "eval", "eval_name": "lcb", "model_short": "m", "format": "hf",
           "sample_traces": [{"gold_answer": "x" * 10}] * 3, "shard": "2/4", "shard_index": 2,
           "shard_count": 4}
    out = cli._summary_record(res)
    assert "sample_traces" not in out and "sample_traces" in res
    assert out["sample_traces_omitted"] == {"count": 3, "file": "eval_lcb_m_hf.shard-2of4.json"}
    assert cli._summary_record({"benchmark_type": "golden", "x": 1}) == {"benchmark_type": "golden",
                                                                          "x": 1}


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_another_organisations_checkpoint_is_not_matched_by_its_name():
    cands = ["google/gemma-4-E4B-it", "gemma-4-E4B-it"]
    served, why = C.match_served_model(["otherorg/gemma-4-E4B-it"], cands)
    assert served is None and "--served-model-id" in why
    assert C.match_served_model(["/models/gemma-4-E4B-it"], cands)[0] == "/models/gemma-4-E4B-it"
    assert C.match_served_model(["google/gemma-4-E4B-it"],
                                ["/ckpt/gemma-4-E4B-it", "gemma-4-E4B-it"])[0] == \
        "google/gemma-4-E4B-it"


@pytest.mark.parametrize("value,ok", [("", True), ("127.0.0.1", True), ("127.0.1.5", True),
                                      ("::1", True), ("localhost", True), ("0.0.0.0", True),
                                      ("::", True), ("10.1.2.3", False), ("example.com", False)])
def test_the_server_bind_address_must_be_reachable_at_loopback(value, ok):
    from gbench.runners.serving import server_host_env_error
    assert (server_host_env_error(value) is None) is ok


def test_a_local_run_refuses_a_served_model_id_its_server_does_not_serve(tmp_path, monkeypatch,
                                                                         caplog):
    from gbench.core.models import ModelConfig
    monkeypatch.delenv("GBENCH_SERVER_HOST", raising=False)
    model = ModelConfig(short_name="m", hf_model_id="org/m", total_params_b=2.0, is_moe=False)
    model.metadata_source = "config"
    monkeypatch.setattr(cli, "get_models_from_args", lambda args: [model])
    argv = ["--models", "org/m", "--num-gpus", "1", "--evals", "gsm8k",
            "--max-output-tokens", "64", "--dry-run", "--results-dir", str(tmp_path)]
    with caplog.at_level("ERROR"):
        rc = cli.main(argv + ["--served-model-id", "other/id"])
    assert rc == cli.EXIT_HARNESS_ERROR
    assert "names the id a --remote-endpoint serves the model under" in caplog.text
    caplog.clear()
    with caplog.at_level("ERROR"):
        cli.main(argv + ["--served-model-id", "org/m"])             # the id it serves: allowed
    assert "names the id a --remote-endpoint serves" not in caplog.text
    caplog.clear()
    monkeypatch.setenv("GBENCH_SERVER_HOST", "10.1.2.3")
    with caplog.at_level("ERROR"):
        assert cli.main(argv) == cli.EXIT_HARNESS_ERROR
    assert "GBENCH_SERVER_HOST='10.1.2.3'" in caplog.text


def test_the_servers_tokenizer_is_gbenchs_when_none_is_named(tmp_path):
    parser = cli.create_parser()
    args = parser.parse_args(["--models", "org/m", "--vllm-serve-args",
                              "--tokenizer_mode auto --tokenizer org/tok",
                              "--results-dir", str(tmp_path), "--dry-run"])
    config = cli.get_config_from_args(args)
    assert config.tokenizer == "org/tok"
    assert config.tokenizer_source == "--vllm-serve-args --tokenizer"
    assert C.serve_arg_value(["--tokenizer=org/a"], "--tokenizer") == "org/a"
    assert C.serve_arg_flag(["--trust_remote_code"], "--trust-remote-code") is True
    cfg = types.SimpleNamespace(tokenizer_trust_remote_code=False,
                                vllm_serve_args=["--trust-remote-code"])
    assert C.tokenizer_trust_remote_code(cfg) == (True, "--vllm-serve-args --trust-remote-code")


def test_a_reused_run_directory_of_another_shard_is_not_continued(tmp_path):
    from gbench.utils.log_manager import LogManager, foreign_run_reason
    run = tmp_path / "run"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({"host": "h1", "shard": "1/2"}))
    lm = LogManager(run, identity={"host": "h1", "shard": "2/2", "gbench": None})
    try:
        assert lm.results_dir != run and lm.results_dir.parent == run
        assert "another node, shard or commit" in lm.shared_dir_note
    finally:
        lm.release_run_lock()
    same = LogManager(run, identity={"host": "h1", "shard": "1/2", "gbench": None})
    try:
        assert same.results_dir == run
    finally:
        same.release_run_lock()
    assert foreign_run_reason(run, None) == ""


def test_the_weights_advisory_reads_the_selected_gpus_only(monkeypatch):
    monkeypatch.setattr(C, "_gpu_min_free_vram_gb", lambda: 1.0)     # a busy unselected GPU
    monkeypatch.setattr(C, "_gpu_total_vram_gb", lambda: 80.0)
    chosen = [C.GpuInfo(0, "GPU-0", 80.0, 79.0)]
    fits, _msg = C.weights_fit_advisory(8.0, 1, 0.9, gpus=chosen)
    assert fits is True
    assert C.weights_fit_advisory(8.0, 1, 0.9)[0] is False


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
def test_a_remote_endpoints_serve_settings_are_recorded_as_unknown():
    cfg = types.SimpleNamespace(remote_endpoint="http://h:1/v1", tool_call_parser=None,
                                reasoning_parser=None, vllm_serve_args=[])
    rec = C.serve_settings_record(cfg)
    assert rec["server_started_by_gbench"] is False and "not started by gbench" in rec["note"]
    args = types.SimpleNamespace(models=None)
    model = types.SimpleNamespace(hf_model_id="org/m")
    assert cli._served_model_id_record(types.SimpleNamespace(served_model_id=None,
                                                             remote_endpoint="u"),
                                       args, [model])["served_model_id"] == "org/m"


def test_the_throughput_scaffold_pins_only_what_the_runner_reads():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_scaffold import _Config
    from gbench.core.scaffold import Pillar, build_performance_scaffold
    load = build_performance_scaffold(Pillar.THROUGHPUT, _Config()).pinned["load_shape"]
    assert load.get("batch_sizes") is None and load.get("request_rate") is None
    assert load.get("warmup_iterations") is None and load["num_iterations"] == 3
    a = build_performance_scaffold(Pillar.THROUGHPUT, _Config(batch_sizes=[1]))
    b = build_performance_scaffold(Pillar.THROUGHPUT, _Config(batch_sizes=[1, 64]))
    assert a.scaffold_id == b.scaffold_id
    ds = build_performance_scaffold(Pillar.THROUGHPUT, _Config(dataset="sharegpt"))
    assert ds.pinned["dataset_snapshot"]["dataset"] == "random"


def test_matharena_suites_record_the_attempts_they_run():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_scaffold import _Config
    from gbench.core.scaffold import build_evals_scaffold
    pin = build_evals_scaffold(_Config(evals=["aime", "gsm8k"])).pinned["attempts"]
    assert pin == {"attempts_per_sample": 1, "suite_attempts_per_sample": {"aime": 4}}
    cfg = types.SimpleNamespace(attempt_count=1)
    assert E._suite_attempt_count(cfg, "hmmt") == 4 and E._suite_attempt_count(cfg, "gsm8k") == 1


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def test_a_suite_merge_hook_that_raises_is_a_refused_merge(monkeypatch):
    from gbench import merge as M
    module = types.SimpleNamespace(merge_shards=lambda results: (_ for _ in ()).throw(
        RuntimeError("the evaluator wrote no summary")))
    monkeypatch.setattr(M.importlib, "import_module", lambda name: module)
    with pytest.raises(M.MergeError, match="merging failed while re-scoring the union"):
        M.merge_suite("omnidocbench", [{}, {}])


def _shard(label, traces, **kw):
    i, n = label.split("/")
    return {"eval_name": "unit", "shard": label, "shard_index": int(i), "shard_count": int(n),
            "sample_traces": traces, "total_questions": len(traces),
            "correct_answers": sum(1 for t in traces if t.get("is_correct")),
            "accuracy": round(100.0 * sum(1 for t in traces if t.get("is_correct"))
                              / len(traces), 2),
            "category_accuracy": {}, **kw}


def _trace(key, gidx, health="ok", event=None, sample_idx=0):
    t = {"sample_key": key, "global_idx": gidx, "attempt_idx": 0, "is_correct": True,
         "health": health, "max_tokens_sent": 100, "sample_idx": sample_idx, "category": "c"}
    if event:
        t["server_parser"] = {"event": event}
    return t


def test_the_budget_rule_and_the_parser_record_are_recounted_over_the_union():
    from gbench import merge as M
    cut = [_trace(f"k{i}", i, health=("truncated" if i < 3 else "ok"),
                  event=("server_parser_moved_text" if i == 0 else None), sample_idx=i)
           for i in range(20)]
    a = _shard("1/2", cut, status="completed_with_errors",
               status_reason="replies_cut_off_by_non_official_budget 3 of 20 (over 10%: ...)",
               server_parser_events={"server_parser_moved_text": 1},
               server_parser_samples={"server_parser_moved_text": [0]})
    b_traces = [_trace(f"j{i}", 20 + i, event=("server_parser_moved_text" if i == 0 else None),
                       sample_idx=i) for i in range(20)]
    b = _shard("2/2", b_traces, status="success", status_reason=None,
               server_parser_events={"server_parser_moved_text": 1},
               server_parser_samples={"server_parser_moved_text": [0]})
    merged = M.generic_merge("unit", [a, b])
    assert merged["status"] == "success" and merged["status_reason"] is None   # 3 of 40 < 10%
    assert merged["server_parser_events"] == {"server_parser_moved_text": 2}
    assert merged["server_parser_samples"] == {"server_parser_moved_text": [0, 20]}


# --------------------------------------------------------------------------- #
# canonical_sync
# --------------------------------------------------------------------------- #
def test_a_data_txt_is_hashed_byte_for_byte_and_a_constraint_list_without_comments(tmp_path):
    from gbench.runners.eval_suites import canonical_sync as cs
    data = tmp_path / "13.6.txt"
    data.write_text("# prompt line\ncode\n")
    before = cs._file_digest(data)
    data.write_text("# edited prompt line\ncode\n")
    assert cs._file_digest(data) != before
    pins = tmp_path / "x.constraints.txt"
    pins.write_text("# a comment\nfoo==1\n")
    before = cs._file_digest(pins)
    pins.write_text("# another comment\nfoo==1\n")
    assert cs._file_digest(pins) == before


def test_the_medxpertqa_and_bigcodebench_methods_describe_the_code():
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as CS
    med = CS["medxpertqa"]["method"]
    assert "system turn dropped" not in med and "chat branch on both turns, in both modes" in med
    big = CS["bigcodebench"]["method"]
    assert "reasoning_effort 'medium' sent" not in big and "top_p 0.95" in big


# --------------------------------------------------------------------------- #
# the Golden Set leak check, throughput, calibration, writers, parameter counts
# --------------------------------------------------------------------------- #
def test_the_leak_check_reads_the_served_template_and_names_the_trust_flag(monkeypatch,
                                                                           tmp_path):
    from gbench.runners import golden as G
    tok = types.SimpleNamespace(all_special_tokens=["<eos>"], get_added_vocab=lambda: {},
                                chat_template="{{ messages }}")
    loads = []

    def _load(name, trust_remote_code=False):
        loads.append((name, trust_remote_code))
        return tok
    monkeypatch.setattr(G, "_load_tokenizer", _load)
    cfg = C.BenchmarkConfig(results_dir=tmp_path)
    cfg.vllm_serve_args = ["--chat_template", "<|user|>{{ m }}<|end|><start_of_turn>",
                           "--trust_remote_code"]
    rec = G.GoldenBenchmarkRunner(cfg)._template_leak_tokens(
        types.SimpleNamespace(hf_model_id="org/m"), None)
    assert loads == [("org/m", True)] and rec["chat_template_source"] == "--chat-template"
    assert rec["template_literal_delimiters"] == ["<start_of_turn>", "<|end|>", "<|user|>"]
    assert {"<eos>", "<|user|>"} <= set(rec["tokens"])
    assert rec["trust_remote_code_source"] == "--vllm-serve-args --trust-remote-code"

    def _fail(name, trust_remote_code=False):
        raise ValueError("requires you to execute code: pass trust_remote_code=True")
    monkeypatch.setattr(G, "_load_tokenizer", _fail)
    remote = C.BenchmarkConfig(results_dir=tmp_path)
    remote.remote_endpoint = "http://h:1/v1"
    rec = G.GoldenBenchmarkRunner(remote)._template_leak_tokens(
        types.SimpleNamespace(hf_model_id="org/m"), None)
    assert "--tokenizer-trust-remote-code" in rec["error"]


def test_the_offline_engine_keeps_a_negative_value_and_counts_images_with_its_kwargs(monkeypatch):
    from gbench.runners import throughput as T
    kept, dropped = T.split_engine_args(["--seed", "-1", "--max-num-seqs", "4",
                                         "--chat-template", "t"],
                                        frozenset({"--seed", "--max-num-seqs"}))
    assert kept == ["--seed", "-1", "--max-num-seqs", "4"] and dropped == ["--chat-template", "t"]
    runner = T.ThroughputBenchmarkRunner.__new__(T.ThroughputBenchmarkRunner)
    runner.config = types.SimpleNamespace(tokenizer_trust_remote_code=False, vllm_serve_args=[])
    runner._engine_args_split = (["--mm-processor-kwargs", '{"do_pan_and_scan": true}'], [])
    runner._image_tokens = {}
    seen = {}

    def _count(path, mm_kwargs=None):
        seen["kwargs"] = mm_kwargs
        return 300
    runner._count_image_tokens = _count
    model = types.SimpleNamespace(get_model_path=lambda fmt: "org/m", name="m")
    n, source = runner._mm_image_tokens(model, None)
    assert (n, seen["kwargs"]) == (300, {"do_pan_and_scan": True})
    assert source == 'processor (mm_processor_kwargs {"do_pan_and_scan": true})'


def test_calibration_names_the_network_capped_sandbox_count():
    from gbench.utils import calibrate as CAL
    hw = {"cpu_cores": 64, "ram_available_gb": 500, "ram_total_gb": 600, "docker": True,
          "docker_networks": {"free": 3, "capacity": 31, "pools_source": "default"}, "gpus": 0,
          "gpu_total_vram_gb": 0, "gpu_free_vram_gb": 0}
    rec = CAL.recommend(hw)
    assert rec["sandboxes"] > 3 and rec["sandboxes_network_suites"] == 3
    assert "--sandboxes 3   (for swe_bench_pro, swe_lancer, terminal_bench" in \
        CAL.format_report(hw, rec)


def test_the_json_writer_streams_and_still_redacts_a_split_value(monkeypatch, tmp_path):
    from gbench.runners.eval_suites import base
    from gbench.utils import log_manager as L
    secret = "sk-unit-stream-secret-0123456789"
    monkeypatch.setattr(base, "secret_values", lambda environ=None: [secret])
    monkeypatch.setattr(L, "_JSON_STREAM_CHUNK", 7)
    data = {"a": ["x" * 5 + secret + "y" for _ in range(20)], "b": {"n": None, "f": 1.5}}
    text = L.write_redacted_json(tmp_path / "r.json", data).read_text(encoding="utf-8")
    assert secret not in text and text.count(base.REDACTED) == 20
    assert text == base.redact_text(json.dumps(data, indent=2), [secret])


def test_a_consolidated_file_beside_shards_is_not_counted_twice(tmp_path):
    import struct
    from gbench.core import models as MD

    def _write(name, shape):
        header = json.dumps({"w": {"dtype": "BF16", "shape": shape,
                                   "data_offsets": [0, 0]}}).encode()
        (tmp_path / name).write_bytes(struct.pack("<Q", len(header)) + header)
    _write("model-00001-of-00001.safetensors", [1000, 1000])
    _write("consolidated.safetensors", [1000, 1000])
    assert MD._local_safetensors_params(str(tmp_path)) == pytest.approx(0.001)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"w": "consolidated.safetensors"}}))
    assert MD._local_safetensors_params(str(tmp_path)) == pytest.approx(0.001)
