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

"""The scaffold pins what each runner resolves, and the code-fingerprint gate catches drift.

Two halves, both about the same failure: two runs that asked different questions sharing a
scaffold_id. The first half checks that every input a runner reads (the evals knobs, the
run-shaping environment, the judge cascade, the shared modules' anchors, the stress load and SLOs,
the serving counts and images, the server's arguments, engine and GPUs) moves the id, and that
inputs a runner never reads do not. The second half runs the fingerprint rules on a scratch
package: a docstring or comment edit keeps a fingerprint, a prompt or scorer edit moves it, and a
moved fingerprint needs a recorded decision.

No model, no network, no GPU: nvidia-smi and the engine record are stubbed.
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

from gbench.core import scaffold
from gbench.core.scaffold import (
    EVALS_COVERS,
    SCAFFOLD_FIELDS,
    SUBJECT_FIELDS,
    ContractField,
    Pillar,
    build_condition,
    build_evals_scaffold,
    build_golden_scaffold,
    build_performance_scaffold,
    build_quality_scaffold,
    build_serving_environment,
    build_stress_scaffold,
    build_subject,
    render_conditions,
)
from gbench.runners.eval_suites import SUITES
from gbench.runners.eval_suites import base as base_mod
from gbench.runners.eval_suites import canonical_sync as cs

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_scaffold import DATASET, REMOTE_ENDPOINT, _Config, _Model  # noqa: E402

LOAD = ContractField.LOAD_SHAPE.value
ENV = ContractField.SERVING_ENVIRONMENT.value
DS = ContractField.DATASET_SNAPSHOT.value
DECODE = ContractField.DECODE.value
PROMPT = ContractField.PROMPT_VERSION.value
SCORING = ContractField.SCORING_VERSION.value

#: Every run-shaping variable the tests below may set, cleared first so the host's shell cannot
#: leak into an id.
_ENV_KNOBS = (sorted(base_mod.RUN_SHAPING_GLOBAL_ENV)
              + sorted({k for v in base_mod.RUN_SHAPING_SUITE_ENV.values() for k in v})
              + ["GBENCH_SHARD", "GBENCH_GSM8K_TEMPERATURE", "GBENCH_STRESS_REPS",
                 "GBENCH_STRESS_MAX_QPS", "GBENCH_STRESS_MAX_PROMPTS",
                 "GBENCH_STRESS_MIN_SAMPLES", "GBENCH_STRESS_CLIENT_PROCS", "GBENCH_VLLM_BIN",
                 "GBENCH_SERVER_PORT", "GBENCH_EVAL_PLUGINS_PATH", "CUDA_VISIBLE_DEVICES"])


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV_KNOBS:
        monkeypatch.delenv(name, raising=False)
    scaffold._gpu_models.cache_clear()
    yield
    scaffold._gpu_models.cache_clear()


def _evals(**overrides):
    overrides.setdefault("evals", ["gsm8k"])
    return build_evals_scaffold(_Config(**overrides))


def _id(sc):
    return sc.scaffold_id


# --------------------------------------------------------------------------- #
# Evals: every knob the runner hands the suites
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("knob,value", [
    ("temperature", 1.0),
    ("eval_max_output_tokens", 65536),
    ("eval_limit", 10),
    ("shard", "3/8"),
    ("sandboxes", 64),
    ("eval_concurrency", 16),
    ("attempt_count", 4),
    ("eval_max_soft_tokens", 280),
    ("eval_categories", "math"),
    ("eval_n_shot", 5),
    ("eval_thinking", True),
])
def test_each_evals_knob_moves_the_id(knob, value):
    assert _id(_evals(**{knob: value})) != _id(_evals())


def test_identical_configs_keep_the_id():
    assert _id(_evals(temperature=0.0, eval_limit=5)) == _id(_evals(temperature=0.0, eval_limit=5))


def test_attempts_are_covered_and_pinned_as_attempts_per_sample():
    sc = _evals(attempt_count=10)
    assert ContractField.ATTEMPTS in sc.covers and ContractField.ATTEMPTS in EVALS_COVERS
    assert sc.pinned[ContractField.ATTEMPTS.value] == {"attempts_per_sample": 10}


def test_the_shard_the_suites_read_from_the_environment_is_pinned(monkeypatch):
    full = _id(_evals())
    monkeypatch.setenv("GBENCH_SHARD", "2/4")
    sc = _evals()
    assert sc.pinned[DS]["shard"] == "2/4"
    assert _id(sc) != full
    assert _id(_evals(shard="2/4")) == _id(sc)


def test_a_one_of_one_shard_is_the_whole_set():
    assert _id(_evals(shard="1/1")) == _id(_evals())


def test_a_malformed_shard_fails_like_it_does_for_the_suites():
    with pytest.raises(ValueError):
        _evals(shard="9/3")


def test_container_concurrency_is_pinned_when_a_containerized_suite_runs():
    plain = _evals()
    assert plain.pinned[LOAD]["container_concurrency"] is None
    boxed = _evals(evals=["lcb"], batch_sizes=[4])
    assert boxed.pinned[LOAD]["container_concurrency"] == 4
    assert _id(_evals(evals=["lcb"], batch_sizes=[4], eval_concurrency=8)) != _id(
        _evals(evals=["lcb"], batch_sizes=[2], eval_concurrency=8))
    assert _evals(evals=["lcb"], sandboxes=3).pinned[LOAD]["container_concurrency"] == 3


def test_the_custom_jsonl_is_pinned_by_content_not_path(tmp_path):
    a, b, c = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    a.write_text('{"q": 1}\n')
    b.write_text('{"q": 1}\n')
    c.write_text('{"q": 2}\n')
    ids = {p: build_evals_scaffold(_Config(evals=["custom_jsonl"], eval_custom_jsonl=str(p)))
           for p in (a, b, c)}
    assert ids[a].pinned[DS]["custom_jsonl_sha256"] == hashlib.sha256(b'{"q": 1}\n').hexdigest()
    assert _id(ids[a]) == _id(ids[b])
    assert _id(ids[a]) != _id(ids[c])


def test_a_custom_jsonl_without_the_suite_named_is_added_as_the_runner_adds_it(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text("{}\n")
    sc = build_evals_scaffold(_Config(evals=["gsm8k"], eval_custom_jsonl=str(f)))
    assert sc.pinned[DS]["evals"] == ["custom_jsonl", "gsm8k"]
    assert sc.pinned[DS]["custom_jsonl_sha256"]


# --------------------------------------------------------------------------- #
# Evals: the suites that actually run
# --------------------------------------------------------------------------- #
def test_a_comma_list_is_the_same_run_as_a_list():
    assert _id(_evals(evals=["gpqa,mmlu"])) == _id(_evals(evals=["mmlu", "gpqa"]))


def test_a_renamed_suite_resolves_to_its_new_name():
    sc = _evals(evals=["semantic_keypoint"])
    assert sc.pinned[DS]["evals"] == ["screenspot_pointing"]
    assert "screenspot_pointing" in sc.pinned[SCORING]["suites"]


def test_all_is_the_set_the_runner_runs():
    from gbench.runners.evals import builtin_all_suites

    sc = _evals(evals=["all"])
    assert sc.pinned[DS]["evals"] == sorted(builtin_all_suites())
    assert "swe_lancer" not in sc.pinned[SCORING]["suites"]


def test_bumping_a_member_of_a_comma_list_moves_the_id(monkeypatch):
    before = _id(_evals(evals=["gpqa,mmlu"]))
    monkeypatch.setitem(cs.CANONICAL_SYNC, "gpqa", {**cs.CANONICAL_SYNC["gpqa"],
                                                   "synced": "2099-01-01"})
    assert _id(_evals(evals=["gpqa,mmlu"])) != before


def test_a_plugin_is_pinned_by_its_module_bytes(tmp_path, monkeypatch):
    path = tmp_path / "my_plugin_suite.py"
    path.write_text("def run_zz_w3b_plugin(**kw):\n    return {}\n")
    spec = importlib.util.spec_from_file_location("gbench_custom_zz_w3b", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "gbench_custom_zz_w3b", module)
    monkeypatch.setitem(SUITES, "zz_w3b_plugin", module.run_zz_w3b_plugin)
    sc = _evals(evals=["zz_w3b_plugin"])
    assert sc.pinned[SCORING]["plugins"] == {
        "zz_w3b_plugin": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert "zz_w3b_plugin" not in sc.pinned[SCORING]["suites"]
    path.write_text("def run_zz_w3b_plugin(**kw):\n    return {'changed': 1}\n")
    assert _id(_evals(evals=["zz_w3b_plugin"])) != _id(sc)


# --------------------------------------------------------------------------- #
# Evals: run-shaping environment variables
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("suite,var,value", [
    ("swe_bench_live", "GBENCH_SWE_BENCH_LIVE_MODE", "oneshot"),
    ("lcb", "GBENCH_LCB_VERSION", "release_v5"),
    ("gaia", "GBENCH_GAIA_SPLIT", "test"),
    ("tau2", "GBENCH_TAU2_NUM_TRIALS", "4"),
    ("gsm8k", "GBENCH_GSM8K_TEMPERATURE", "0.7"),
    ("gsm8k", "GBENCH_REPETITION_PENALTY", "1.1"),
])
def test_a_declared_run_shaping_variable_moves_the_id(monkeypatch, suite, var, value):
    before = _evals(evals=[suite])
    monkeypatch.setenv(var, value)
    after = _evals(evals=[suite])
    assert _id(after) != _id(before)
    blob = json.dumps(after.pinned)
    assert var in blob and value in blob


def test_a_variable_of_a_suite_that_did_not_run_leaves_the_id(monkeypatch):
    before = _id(_evals(evals=["gsm8k"]))
    monkeypatch.setenv("GBENCH_LCB_VERSION", "release_v5")
    assert _id(_evals(evals=["gsm8k"])) == before


def test_each_variable_lands_in_its_class():
    env = {"GBENCH_SWE_BENCH_LIVE_MODE": "oneshot", "GBENCH_LCB_VERSION": "v5",
           "GBENCH_REPETITION_PENALTY": "1.1"}
    snap = base_mod.run_shaping_env(["swe_bench_live", "lcb"], env)
    assert snap["prompt"] == {"GBENCH_SWE_BENCH_LIVE_MODE": "oneshot"}
    assert snap["dataset"] == {"GBENCH_LCB_VERSION": "v5"}
    assert snap["decode"] == {"GBENCH_REPETITION_PENALTY": "1.1"}


# --------------------------------------------------------------------------- #
# Evals: the judge cascade and the scoring anchors
# --------------------------------------------------------------------------- #
def test_the_judge_cascade_is_pinned_and_moves_the_id(monkeypatch):
    before = _evals()
    judge = before.pinned[SCORING]["judge"]
    assert judge["judge_cascade"] == list(base_mod.GEMINI_CASCADE)
    assert judge["judge_temperature"] == base_mod.JUDGE_TEMPERATURE
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", "gemini-3.7-flash")
    assert _id(_evals()) != _id(before)


def test_naming_the_default_cascade_is_the_same_grading(monkeypatch):
    before = _id(_evals())
    monkeypatch.setenv("GBENCH_JUDGE_MODELS", ",".join(base_mod.GEMINI_CASCADE))
    assert _id(_evals()) == before


def test_a_new_default_cascade_moves_the_id(monkeypatch):
    before = _id(_evals())
    monkeypatch.setattr(base_mod, "GEMINI_CASCADE", ("gemini-9-flash",))
    assert _id(_evals()) != before


def test_a_search_cascade_override_moves_the_id(monkeypatch):
    before = _id(_evals())
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "gemini-3.6-flash")
    assert _id(_evals()) != before


@pytest.mark.parametrize("field,value", [
    ("dataset", "openai/gsm8k@0000000000000000000000000000000000000000"),
    ("revision", "1111111111111111111111111111111111111111"),
    ("checksum", "sha256:" + "0" * 64),
    ("synced", "2099-01-01"),
])
def test_every_anchor_of_an_entry_moves_the_id(monkeypatch, field, value):
    before = _id(_evals())
    monkeypatch.setitem(cs.CANONICAL_SYNC, "gsm8k", {**cs.CANONICAL_SYNC["gsm8k"], field: value})
    assert _id(_evals()) != before


def test_editing_the_method_prose_leaves_the_id(monkeypatch):
    before = _id(_evals())
    monkeypatch.setitem(cs.CANONICAL_SYNC, "gsm8k", {**cs.CANONICAL_SYNC["gsm8k"],
                                                    "method": "reworded"})
    assert _id(_evals()) == before


def test_bumping_a_shared_module_moves_every_suite_that_imports_it(monkeypatch):
    before = _id(_evals())
    monkeypatch.setitem(cs.SHARED_SYNC, "base", {**cs.SHARED_SYNC["base"], "synced": "2099-01-01"})
    assert _id(_evals()) != before


def test_a_suite_that_imports_another_moves_with_it(monkeypatch):
    assert "gpqa" in cs.DEPENDS_ON["browsecomp"]
    before = _id(_evals(evals=["browsecomp"]))
    monkeypatch.setitem(cs.CANONICAL_SYNC, "gpqa", {**cs.CANONICAL_SYNC["gpqa"],
                                                   "synced": "2099-01-01"})
    assert _id(_evals(evals=["browsecomp"])) != before


def test_the_code_fingerprint_is_not_an_id_input(monkeypatch):
    """Declared, not derived: a refactor must not break a series by itself."""
    before = _id(_evals())
    monkeypatch.setitem(cs.CODE_FP, "gsm8k", "0" * 16)
    assert _id(_evals()) == before


# --------------------------------------------------------------------------- #
# Serving environment
# --------------------------------------------------------------------------- #
def test_the_parsers_reach_only_the_native_server(monkeypatch):
    plain = _Config()
    parsed = _Config(tool_call_parser="some_parser", reasoning_parser="some_reasoner")
    assert _id(build_quality_scaffold(parsed)) != _id(build_quality_scaffold(plain))
    assert _id(build_golden_scaffold(DATASET, config=parsed)) != _id(
        build_golden_scaffold(DATASET, config=plain))
    assert _id(_evals(tool_call_parser="some_parser")) != _id(_evals())
    assert _id(build_performance_scaffold(Pillar.SERVING, parsed)) == _id(
        build_performance_scaffold(Pillar.SERVING, plain))
    native = build_serving_environment(parsed, context="native")
    assert native.serve_args == ("--enable-auto-tool-choice", "--tool-call-parser", "some_parser",
                                 "--reasoning-parser", "some_reasoner")


def test_extra_serve_args_reach_every_server():
    extra = _Config(vllm_serve_args=["--kv-cache-dtype", "auto"])
    for pillar in (Pillar.SERVING, Pillar.THROUGHPUT, Pillar.STRESS_TEST, Pillar.EVALS,
                   Pillar.QUALITY):
        a = build_condition(pillar, _Model(), "hf", config=_Config(evals=["gsm8k"])).scaffold
        b = build_condition(pillar, _Model(), "hf",
                            config=_Config(evals=["gsm8k"],
                                           vllm_serve_args=["--kv-cache-dtype", "auto"])).scaffold
        assert _id(a) != _id(b), pillar
    assert build_serving_environment(extra).serve_args == ("--kv-cache-dtype", "auto")


def test_data_parallel_replicas_move_the_id():
    one = build_performance_scaffold(Pillar.SERVING, _Config(num_gpus=2, tensor_parallel_size=2))
    two = build_performance_scaffold(Pillar.SERVING, _Config(num_gpus=2, tensor_parallel_size=1,
                                                             data_parallel_size=2))
    assert two.pinned[ENV]["data_parallel_size"] == 2
    assert _id(one) != _id(two)


def test_the_engine_is_pinned(monkeypatch):
    from gbench.utils import dependency_checks as dc

    def record(version, override=None):
        return lambda: {"cli": override or [sys.executable, "-m", "vllm.entrypoints.cli.main"],
                        "cli_source": dc.VLLM_BIN_ENV if override else "gbench's interpreter",
                        "package_version": version, "package_path": "/somewhere/vllm"}

    monkeypatch.setattr(dc, "vllm_engine_record", record("0.1.0"))
    a = build_performance_scaffold(Pillar.SERVING, _Config())
    assert a.pinned[ENV]["engine"] == {"vllm_version": "0.1.0", "cli_override": None}
    monkeypatch.setattr(dc, "vllm_engine_record", record("0.2.0"))
    b = build_performance_scaffold(Pillar.SERVING, _Config())
    monkeypatch.setattr(dc, "vllm_engine_record", record("0.2.0", ["/opt/other/vllm"]))
    c = build_performance_scaffold(Pillar.SERVING, _Config())
    assert len({_id(a), _id(b), _id(c)}) == 3
    assert "/somewhere" not in json.dumps(c.pinned)


def _smi(monkeypatch, stdout, returncode=0):
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(scaffold.subprocess, "run", run)
    return calls


def test_the_gpu_models_of_the_selection_are_pinned_not_their_uuids(monkeypatch):
    _smi(monkeypatch, "GPU-a, NVIDIA H100 80GB HBM3\nGPU-b, NVIDIA H100 80GB HBM3\n"
                      "GPU-c, NVIDIA A100-SXM4-80GB\n")
    h100 = build_performance_scaffold(Pillar.SERVING, _Config(selected_gpus=["GPU-a"]))
    other_h100 = build_performance_scaffold(Pillar.SERVING, _Config(selected_gpus=["GPU-b"]))
    a100 = build_performance_scaffold(Pillar.SERVING, _Config(selected_gpus=["GPU-c"]))
    assert h100.pinned[ENV]["gpu_models"] == ["NVIDIA H100 80GB HBM3"]
    assert _id(h100) == _id(other_h100)
    assert _id(h100) != _id(a100)
    assert "GPU-a" not in json.dumps(h100.pinned)


@pytest.mark.parametrize("stdout,code", [("", 9), ("GPU-a, NVIDIA H100\n", 0)])
def test_unreadable_or_partial_gpu_names_pin_nothing(monkeypatch, stdout, code):
    _smi(monkeypatch, stdout, code)
    sc = build_performance_scaffold(Pillar.SERVING, _Config(selected_gpus=["GPU-a", "GPU-z"]))
    assert sc.pinned[ENV]["gpu_models"] is None


def test_a_remote_run_pins_none_of_the_server_settings(monkeypatch):
    calls = _smi(monkeypatch, "")
    env = build_serving_environment(_Config(remote_endpoint=REMOTE_ENDPOINT,
                                            selected_gpus=["GPU-a"], tool_call_parser="p",
                                            vllm_serve_args=["--x"], data_parallel_size=2))
    assert env.serve_args is None and env.engine is None and env.gpu_models is None
    assert env.data_parallel_size is None
    assert not calls


def test_the_port_and_served_id_are_not_in_the_id(monkeypatch):
    before = _id(_evals())
    monkeypatch.setenv("GBENCH_SERVER_PORT", "8123")
    assert _id(_evals()) == before
    assert "8123" not in json.dumps(_evals().pinned)


# --------------------------------------------------------------------------- #
# Performance pillars
# --------------------------------------------------------------------------- #
def test_flags_serving_never_reads_do_not_move_it():
    base = build_performance_scaffold(Pillar.SERVING, _Config())
    cfg = _Config()
    cfg.num_prompts = 5              # --num-prompts: serving auto-sizes its count instead
    cfg.images_per_request = [9]     # dead: the runners send mm_images_per_request or 4
    assert _id(build_performance_scaffold(Pillar.SERVING, cfg)) == _id(base)
    assert _id(build_performance_scaffold(Pillar.SERVING_MULTIMODAL, cfg)) == _id(
        build_performance_scaffold(Pillar.SERVING_MULTIMODAL, _Config()))


def test_a_file_backed_serving_dataset_is_named_by_its_path():
    a = build_performance_scaffold(Pillar.SERVING, _Config(dataset="custom", dataset_path="/d/a"))
    b = build_performance_scaffold(Pillar.SERVING, _Config(dataset="custom", dataset_path="/d/b"))
    assert _id(a) != _id(b)


@pytest.mark.parametrize("pillar,local_moves", [
    (Pillar.SERVING, False),
    (Pillar.SERVING_MULTIMODAL, True),
    (Pillar.THROUGHPUT, False),
    (Pillar.THROUGHPUT_MULTIMODAL, False),
    (Pillar.STRESS_TEST, True),
    (Pillar.STRESS_TEST_MULTIMODAL, True),
])
def test_thinking_moves_exactly_the_pillars_whose_requests_carry_it(pillar, local_moves):
    def build(**kw):
        return build_condition(pillar, _Model(), "hf", config=_Config(**kw)).scaffold

    assert (_id(build(thinking=True)) != _id(build())) is local_moves
    remote_moves = local_moves or pillar is Pillar.SERVING
    if pillar not in (Pillar.THROUGHPUT, Pillar.THROUGHPUT_MULTIMODAL):
        assert (_id(build(thinking=True, remote_endpoint=REMOTE_ENDPOINT))
                != _id(build(remote_endpoint=REMOTE_ENDPOINT))) is remote_moves


def test_the_no_think_prompt_text_is_pinned(monkeypatch):
    from gbench.runners.serving import ServingBenchmarkRunner

    before = _id(build_performance_scaffold(Pillar.SERVING_MULTIMODAL, _Config()))
    monkeypatch.setattr(ServingBenchmarkRunner, "NOTHINK_SYSTEM", "a different system prompt")
    assert _id(build_performance_scaffold(Pillar.SERVING_MULTIMODAL, _Config())) != before


# --------------------------------------------------------------------------- #
# Stress
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("knob,value", [
    ("stress_reps", 5), ("stress_max_qps", 64.0), ("stress_max_prompts", 300),
    ("stress_min_samples", 5), ("stress_ttft_slo_ms", 900), ("stress_itl_slo_ms", 40),
])
def test_each_stress_knob_moves_the_stress_id(knob, value):
    assert _id(build_stress_scaffold(_Config(**{knob: value}))) != _id(
        build_stress_scaffold(_Config()))


@pytest.mark.parametrize("var,value", [("GBENCH_STRESS_REPS", "7"),
                                       ("GBENCH_STRESS_MAX_QPS", "32"),
                                       ("GBENCH_STRESS_MAX_PROMPTS", "99"),
                                       ("GBENCH_STRESS_MIN_SAMPLES", "4")])
def test_an_environment_set_stress_knob_is_pinned_as_the_runner_resolves_it(monkeypatch, var,
                                                                            value):
    before = _id(build_stress_scaffold(_Config()))
    monkeypatch.setenv(var, value)
    assert _id(build_stress_scaffold(_Config())) != before


def test_the_stress_load_is_what_the_runner_resolves():
    from gbench.runners.stress import StressTestRunner

    sc = build_stress_scaffold(_Config(stress_reps=2, stress_ttft_slo_ms=700))
    runner = StressTestRunner(_Config(stress_reps=2), ttft_threshold_ms=700)
    assert sc.pinned[LOAD]["reps"] == runner._reps == 2
    assert sc.pinned[LOAD]["max_qps"] == runner.max_qps
    assert sc.pinned[LOAD]["min_samples"] == runner._min_steady
    from gbench.analysis.perf_method import methodology_record
    assert sc.pinned[SCORING] == {"ttft_slo_ms": 700, "itl_slo_ms": runner.tpot_threshold_ms,
                                  "keep_up_frac": runner.KEEP_UP_FRAC,
                                  "itl_gate": "gate4", "certify": "auto",
                                  "certify_budget_min": 120.0}
    assert sc.pinned[DS]["workload_version"] == methodology_record("stress")["methodology_version"]


def test_a_methodology_bump_moves_the_stress_id(monkeypatch):
    """A change to the measurement method changes the verdict, so it breaks the series."""
    from gbench.analysis import perf_method
    before = _id(build_stress_scaffold(_Config()))
    monkeypatch.setitem(perf_method.METHODOLOGY_COMPONENTS, "gate", 99)
    assert _id(build_stress_scaffold(_Config())) != before


@pytest.mark.parametrize("pillar", ["SERVING", "SERVING_MULTIMODAL"])
def test_a_methodology_bump_moves_the_serving_id(monkeypatch, pillar):
    from gbench.analysis import perf_method
    before = _id(build_performance_scaffold(getattr(Pillar, pillar), _Config()))
    monkeypatch.setitem(perf_method.METHODOLOGY_COMPONENTS, "payload", 99)
    assert _id(build_performance_scaffold(getattr(Pillar, pillar), _Config())) != before


@pytest.mark.parametrize("component", ["gate", "itl_metric", "schedule"])
def test_a_stress_only_bump_moves_stress_but_not_serving(monkeypatch, component):
    """Serving never reads the server ITL histogram or runs the sweep."""
    from gbench.analysis import perf_method
    serving = _id(build_performance_scaffold(Pillar.SERVING, _Config()))
    stress = _id(build_stress_scaffold(_Config()))
    monkeypatch.setitem(perf_method.METHODOLOGY_COMPONENTS, component, 99)
    assert _id(build_performance_scaffold(Pillar.SERVING, _Config())) == serving
    assert _id(build_stress_scaffold(_Config())) != stress


def test_stress_lengths_count_only_when_the_geometry_was_pinned():
    neutral = build_stress_scaffold(_Config(input_lengths=[4096]))
    assert neutral.pinned[LOAD]["input_lengths"] == [128]
    assert _id(neutral) == _id(build_stress_scaffold(_Config()))
    pinned = build_stress_scaffold(_Config(input_lengths=[4096], workload_shape_explicit=True))
    assert pinned.pinned[LOAD]["input_lengths"] == [4096]
    assert _id(pinned) != _id(neutral)


def test_multimodal_stress_pins_its_images_per_request():
    four = build_stress_scaffold(_Config(), Pillar.STRESS_TEST_MULTIMODAL)
    one = build_stress_scaffold(_Config(mm_images_per_request=1), Pillar.STRESS_TEST_MULTIMODAL)
    assert four.pinned[LOAD]["images_per_request"] == [4]
    assert _id(four) != _id(one)


# --------------------------------------------------------------------------- #
# Subject and rendering
# --------------------------------------------------------------------------- #
def test_the_subject_id_ignores_a_gguf_file_attribute():
    model = _Model()
    model.gguf_file = "anything.gguf"
    assert build_subject(model, "hf").subject_id == build_subject(_Model(), "hf").subject_id
    assert "gguf_file" not in build_subject(model, "hf").to_dict()


def test_coverage_is_counted_over_each_axis():
    conds = [build_condition(Pillar.EVALS, _Model(), "hf", config=_Config(evals=["gsm8k"]))]
    out = render_conditions(conds)
    assert f"pins 7/{len(SCAFFOLD_FIELDS)}" in out
    assert f"Subject ids pin 0/{len(SUBJECT_FIELDS)}" in " ".join(out.split())
    assert "model_checkpoint" not in out.split("pins 7/")[1].split("Subject ids")[0]


# --------------------------------------------------------------------------- #
# The fingerprint rules, on a scratch package
# --------------------------------------------------------------------------- #
SUITE_SRC = '''"""Suite docstring."""
from .base import helper
from .scorer_bits import grade

PROMPT = "Answer the question."


def run_toy():
    """Runner docstring."""
    return grade(PROMPT)  # a comment
'''

BASE_SRC = '''def helper():
    return 1
'''

BITS_SRC = '''def grade(text):
    return text.lower()
'''


@pytest.fixture
def scratch(tmp_path):
    pkg = tmp_path / "eval_suites"
    pkg.mkdir()
    (pkg / "toy.py").write_text(SUITE_SRC)
    (pkg / "other.py").write_text("from .base import helper\n\ndef run_other():\n    return 2\n")
    (pkg / "base.py").write_text(BASE_SRC)
    (pkg / "scorer_bits.py").write_text(BITS_SRC)
    (pkg / "toy.py.bak_old").write_text("garbage")
    docker = tmp_path / "docker"
    docker.mkdir()
    (docker / "toy.Dockerfile").write_text("# comment\nFROM python:3.12\nRUN pip install x==1\n")
    (docker / "toy_entrypoint.sh").write_text("#!/bin/sh\necho hi\n")
    (docker / "other_tool.py").write_text("x = 1\n")

    def make():
        return cs._Package(pkg_dir=pkg, docker_dirs=[docker], root=tmp_path,
                           suites=["toy", "other"], shared=["base"])
    return types.SimpleNamespace(pkg=pkg, docker=docker, make=make)


def _fp(scratch, name="toy"):
    return scratch.make().fingerprint(name)


def test_a_docstring_or_comment_edit_keeps_the_fingerprint(scratch):
    before = _fp(scratch)
    src = SUITE_SRC.replace('"""Suite docstring."""', '"""A reworded docstring."""')
    src = src.replace('"""Runner docstring."""', '"""Also reworded."""')
    (scratch.pkg / "toy.py").write_text(src.replace("# a comment", "# another comment") + "\n\n")
    assert _fp(scratch) == before


def test_a_prompt_or_scorer_edit_moves_the_fingerprint(scratch):
    before = _fp(scratch)
    (scratch.pkg / "toy.py").write_text(SUITE_SRC.replace("Answer the question.", "Answer."))
    prompt = _fp(scratch)
    assert prompt != before
    (scratch.pkg / "scorer_bits.py").write_text(BITS_SRC.replace("lower", "upper"))
    assert _fp(scratch) not in (before, prompt)


def test_a_shared_module_edit_flags_the_shared_entry_not_the_suites(scratch):
    pkg = scratch.make()
    toy, other, base = pkg.fingerprint("toy"), pkg.fingerprint("other"), pkg.fingerprint("base")
    assert pkg.depends_on("toy") == ["base"] and pkg.depends_on("other") == ["base"]
    (scratch.pkg / "base.py").write_text(BASE_SRC.replace("1", "2"))
    pkg = scratch.make()
    assert pkg.fingerprint("base") != base
    assert (pkg.fingerprint("toy"), pkg.fingerprint("other")) == (toy, other)


def test_a_helper_two_suites_own_must_be_declared_shared(scratch):
    (scratch.pkg / "other.py").write_text("from .scorer_bits import grade\n\ndef run_other():\n"
                                          "    return grade('x')\n")
    assert scratch.make().shared_candidates() == {"scorer_bits": ["other", "toy"]}


def test_the_suites_docker_files_count_but_not_their_comments(scratch):
    files = [p.name for p in scratch.make().files("toy")]
    assert "toy.Dockerfile" in files and "toy_entrypoint.sh" in files
    assert "other_tool.py" not in files and "toy.py.bak_old" not in files
    before = _fp(scratch)
    (scratch.docker / "toy.Dockerfile").write_text(
        "# reworded comment\n\nFROM python:3.12\nRUN pip install x==1\n")
    assert _fp(scratch) == before
    (scratch.docker / "toy.Dockerfile").write_text("FROM python:3.12\nRUN pip install x==2\n")
    assert _fp(scratch) != before


def test_a_docker_file_the_suite_names_counts_with_its_siblings(scratch):
    (scratch.docker / "shared_agent.Dockerfile").write_text("FROM scratch\n")
    (scratch.docker / "shared_agent_run.py").write_text("y = 2\n")
    (scratch.pkg / "toy.py").write_text(SUITE_SRC + 'IMAGE = "docker/shared_agent.Dockerfile"\n')
    files = [p.name for p in scratch.make().files("toy")]
    assert "shared_agent.Dockerfile" in files and "shared_agent_run.py" in files


def _ns(code_fp, synced="2026-01-01", declared=None):
    return {"CODE_FP": {"toy": code_fp},
            "CANONICAL_SYNC": {"toy": {"synced": synced, "method": "m"}},
            "SHARED_SYNC": {}, "VERDICT_PRESERVING": declared or {}}


def test_a_moved_fingerprint_needs_a_recorded_decision():
    assert cs.bump_violations(_ns("a" * 16), _ns("b" * 16)) == ["toy"]


def test_a_bumped_anchor_is_a_decision():
    assert cs.bump_violations(_ns("a" * 16), _ns("b" * 16, synced="2026-02-01")) == []


def test_a_verdict_preserving_declaration_must_name_the_new_fingerprint():
    ok = {"toy": {"code_fp": "b" * 16, "reason": "renamed a local variable"}}
    stale = {"toy": {"code_fp": "c" * 16, "reason": "an older refactor"}}
    empty = {"toy": {"code_fp": "b" * 16, "reason": " "}}
    assert cs.bump_violations(_ns("a" * 16), _ns("b" * 16, declared=ok)) == []
    assert cs.bump_violations(_ns("a" * 16), _ns("b" * 16, declared=stale)) == ["toy"]
    assert cs.bump_violations(_ns("a" * 16), _ns("b" * 16, declared=empty)) == ["toy"]


def test_adopting_the_gate_or_adding_an_entry_is_not_a_violation():
    assert cs.bump_violations({"CODE_FP": {}}, _ns("b" * 16)) == []
    assert cs.bump_violations({}, _ns("b" * 16)) == []


def test_refresh_rewrites_only_the_generated_block(tmp_path, monkeypatch):
    src = (Path(cs.__file__).read_text(encoding="utf-8"))
    copy = tmp_path / "canonical_sync.py"
    copy.write_text(src, encoding="utf-8")
    monkeypatch.setattr(cs, "code_fingerprints", lambda pkg=None: {"toy": "f" * 16})
    monkeypatch.setattr(cs, "dependency_table", lambda pkg=None: {"toy": ["base"]})
    cs.refresh_code_fingerprints(copy)
    out = copy.read_text(encoding="utf-8")
    head, tail = src.split(cs._BEGIN + "\n", 1)
    assert out.startswith(head)
    assert out.count(cs._BEGIN) == src.count(cs._BEGIN)
    ns = cs._load_version(out)
    assert ns["CODE_FP"] == {"toy": "f" * 16} and ns["DEPENDS_ON"] == {"toy": ["base"]}
    assert ns["CANONICAL_SYNC"] == cs.CANONICAL_SYNC


def test_the_soft_token_setting_is_part_of_the_native_server_every_native_pillar_asks():
    """The eval server gets --mm-processor-kwargs {"max_soft_tokens": N}; golden and quality
    run on the same server, so their scaffolds move with it too, and the perf servers do not."""
    plain, soft = _Config(), _Config(eval_max_soft_tokens=280)
    env = build_serving_environment(soft, context="native")
    assert env.serve_args == ("--mm-processor-kwargs", '{"max_soft_tokens": 280}')
    assert _id(build_quality_scaffold(soft)) != _id(build_quality_scaffold(plain))
    assert _id(build_golden_scaffold(DATASET, config=soft)) != _id(
        build_golden_scaffold(DATASET, config=plain))
    assert _id(build_performance_scaffold(Pillar.SERVING, soft)) == _id(
        build_performance_scaffold(Pillar.SERVING, plain))


def test_the_native_server_args_match_the_command_the_runner_builds():
    """The scaffold's serve_args are the tail of ServingBenchmarkRunner.build_server_cmd."""
    from gbench.runners.serving import ServingBenchmarkRunner

    if not hasattr(ServingBenchmarkRunner, "build_server_cmd"):
        pytest.skip("the runner has no pure command builder")
    cfg = _Config(eval_max_soft_tokens=280, tool_call_parser="p", reasoning_parser="r",
                  vllm_serve_args=["--kv-cache-dtype", "auto"])
    args = build_serving_environment(cfg, context="native").serve_args
    import inspect
    params = inspect.signature(ServingBenchmarkRunner.build_server_cmd).parameters
    runner = ServingBenchmarkRunner.__new__(ServingBenchmarkRunner)
    runner.config = cfg
    model = _Model()
    model.get_model_path = lambda fmt: model.hf_model_id
    kwargs = {"model": model, "format": scaffold_format(), "context": "native"}
    if "port" in params:
        kwargs["port"] = 8000
    try:
        cmd = runner.build_server_cmd(**kwargs)
    except Exception as err:                                              # noqa: BLE001
        pytest.skip(f"the builder needs more than a stand-in config: {err}")
    assert tuple(cmd[-len(args):]) == args
    perf_args = build_serving_environment(cfg, context="perf").serve_args
    assert perf_args == ("--kv-cache-dtype", "auto")
    kwargs["context"] = "perf"
    assert tuple(runner.build_server_cmd(**kwargs)[-len(perf_args):]) == perf_args


def scaffold_format():
    from gbench.core.models import ModelFormat
    return ModelFormat.HF
