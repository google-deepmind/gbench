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

"""Tests for the scaffold versioning contract.

The contract's whole value is that an id moves when the harness moves and
holds still when it does not. Both halves are failure modes, so both are
tested here: a false break costs a spurious dashboard discontinuity, and a
missed break costs a wrong conclusion, which is the one the contract exists
to prevent.

A second theme runs through the serialisation tests below. Every defect
found while building this module was the same shape: a value that was not
JSON-serialisable reached ``metadata.json``. The scaffold is built at run
start, so that is not a degraded report, it is a dead run. The ``to_dict``
seam is tested directly for that reason.
"""

import json
import shutil
import subprocess
from pathlib import Path
from unittest import mock

import pytest

from gbench.cli import create_parser
from gbench.core import scaffold
from gbench.core.config import BenchmarkConfig, DEFAULT_GEMMACLAW_COMMIT
from gbench.core.models import ModelFormat
from gbench.core.scaffold import (
    CONTRACT_FIELDS,
    CONTRACT_VERSION,
    SCAFFOLD_FIELDS,
    SUBJECT_FIELDS,
    EVALS_COVERS,
    GOLDEN_COVERS,
    PERFORMANCE_COVERS,
    QUALITY_COVERS,
    STRESS_COVERS,
    Condition,
    ContractField,
    DatasetSnapshot,
    LoadShape,
    Pillar,
    Scaffold,
    ServingEnvironment,
    Subject,
    build_condition,
    build_evals_scaffold,
    build_golden_scaffold,
    build_performance_scaffold,
    build_quality_scaffold,
    build_serving_environment,
    build_stress_scaffold,
    build_subject,
    build_unmodelled_scaffold,
    canonical_json,
    dataset_snapshot,
    gbench_commit,
    render_conditions,
    resolve_gemmaclaw_sha,
    strip_changelog,
)

DATASET = Path(__file__).parent.parent / "gbench" / "golden_dataset"

# Shaped like a real Cloud Run URL on purpose. The host name is where the
# project and deployment identifiers live, which is exactly what the
# fingerprint exists to keep out of metadata.json.
REMOTE_ENDPOINT = "https://gbench-vllm-abc123-uc.a.run.app/v1"

LOAD_SHAPE = ContractField.LOAD_SHAPE.value
SERVING_ENVIRONMENT = ContractField.SERVING_ENVIRONMENT.value


class _Model:
    """Minimal stand-in for ModelConfig, so these tests do not need the registry.

    It satisfies the ModelLike Protocol, which is the point of that Protocol
    existing: the contract depends on three attributes, not on a full registry
    entry, and the type should say so.
    """

    def __init__(self, name="gemma-4-E2B-it", hf_model_id="google/gemma-4-E2B-it",
                 supports_multimodal=False):
        self.name = name
        self.short_name = name
        self.hf_model_id = hf_model_id
        self.supports_multimodal = supports_multimodal


#: A second checkpoint of the same model (a quantized export, say): another subject.
_QUANT = dict(name="gemma-4-E2B-it-int4", hf_model_id="google/gemma-4-E2B-it-int4")


class _Config:
    """Minimal stand-in for BenchmarkConfig, satisfying the ConfigLike Protocol.

    Constructing the real one runs ``__post_init__``, which builds a
    ``LogManager`` and creates a timestamped directory on disk. A test of
    a pure hashing contract should not leave directories behind, which is
    the same reason the contract depends on a Protocol rather than on the
    concrete class.

    Defaults mirror ``DEFAULT_CONFIG`` rather than the bare dataclass
    defaults, because ``DEFAULT_CONFIG`` is what a real run uses and the
    two disagree.
    """

    def __init__(self, **overrides):
        self.remote_endpoint = None
        self.num_gpus = 1
        self.tensor_parallel_size = None
        self.data_parallel_size = 1
        self.gpu_memory_utilization = 0.90
        self.max_num_seqs = 256
        self.enable_chunked_prefill = True
        self.max_num_batched_tokens = 16384
        self.max_model_len = None
        self.max_model_len_explicit = None
        self.selected_gpus = None
        self.tool_call_parser = None
        self.reasoning_parser = None
        self.vllm_serve_args = []
        self.batch_sizes = [1, 16, 50, 100]
        self.input_lengths = [128]
        self.output_lengths = [512]
        self.mm_images_per_request = None
        self.serving_num_prompts_override = None
        self.num_prompts_throughput = 1000
        self.request_rate = "inf"
        self.num_iterations = 3
        self.warmup_iterations = 1
        self.workload_shape_explicit = False
        self.stress_reps = None
        self.stress_max_qps = None
        self.stress_max_prompts = None
        self.stress_min_samples = None
        self.stress_ttft_slo_ms = None
        self.stress_itl_slo_ms = None
        self.eval_concurrency = None
        self.sandboxes = None
        self.thinking = False
        self.eval_thinking = False
        self.eval_n_shot = 0
        self.eval_max_soft_tokens = None
        self.eval_max_output_tokens = None
        self.temperature = None
        self.attempt_count = 1
        self.dataset = "random"
        self.dataset_path = None
        self.gemmaclaw_commit = "main"
        self.selected_scenarios = None
        self.evals = None
        self.eval_categories = None
        self.eval_limit = None
        self.shard = None
        self.eval_custom_jsonl = None
        self.eval_plugins_dir = None
        for key, value in overrides.items():
            # Guards against a test silently configuring nothing when a
            # field is renamed, which would make it pass for the wrong
            # reason rather than fail.
            if not hasattr(self, key):
                raise AttributeError(f"_Config has no field {key!r}")
            setattr(self, key, value)


def _scaffolds_for_every_pillar(config=None):
    """One scaffold per pillar, built the way ``cli.py`` builds them.

    Keyed by pillar and driven off the enum rather than off a hand-written
    list, so a pillar added later is covered by every test using this
    without anyone remembering to extend a fixture.
    """
    return {
        pillar: build_condition(
            pillar, _Model(), ModelFormat.HF,
            dataset_dir=DATASET if pillar is Pillar.GOLDEN else None,
            config=config,
        ).scaffold
        for pillar in Pillar
    }


# Fields a pillar covers by subsumption rather than through a key of its
# own. Golden's dataset hash is taken over the case JSON, which holds the
# prompt, the schema and the tool declarations, so hashing the dataset
# detects a change in any of the three. No other pillar does this, and
# declaring it here keeps "covered as a side effect" from being read as
# "covered by a mechanism".
_SUBSUMED = {
    Pillar.GOLDEN: {
        ContractField.PROMPT_VERSION,
        ContractField.SCHEMA_VERSION,
        ContractField.TOOL_PERMISSIONS,
    },
}

_FIELD_VALUES = {f.value for f in ContractField}


@pytest.fixture
def dataset_copy(tmp_path):
    """A writable copy of the real dataset, so mutation tests hit real cases."""
    dst = tmp_path / "golden_dataset"
    shutil.copytree(DATASET, dst)
    return dst


# ----------------------------------------------------------------------
# Canonicaliser
# ----------------------------------------------------------------------

def test_key_order_does_not_affect_canonical_form():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_int_and_float_of_equal_value_canonicalise_the_same():
    """`temperature: 0` and `temperature: 0.0` are the same decode setting.

    JSON has one number type. Letting these differ would break a series on a
    reformat, which is exactly the false alarm that gets a gate switched off.
    """
    assert canonical_json({"temperature": 0}) == canonical_json({"temperature": 0.0})


def test_booleans_are_not_coerced_to_numbers():
    """bool is a subclass of int in Python, so a naive numeric branch eats it."""
    assert canonical_json({"x": True}) != canonical_json({"x": 1})
    assert canonical_json({"x": False}) != canonical_json({"x": 0})


def test_nested_structures_are_canonicalised_recursively():
    a = {"outer": {"b": 1, "a": [{"y": 2, "x": 1}]}}
    b = {"outer": {"a": [{"x": 1, "y": 2}], "b": 1}}
    assert canonical_json(a) == canonical_json(b)


def test_non_ascii_survives_canonicalisation():
    """Cases include non-ASCII prompts, so escaping them would be lossy."""
    assert "Zürich" in canonical_json({"city": "Zürich"})


# ----------------------------------------------------------------------
# Changelog exclusion
# ----------------------------------------------------------------------

def test_strip_changelog_removes_changelog_but_keeps_version():
    case = {"id": "x", "meta": {"version": "1.1", "changelog": [{"version": "1.1"}]}}
    stripped = strip_changelog(case)
    assert stripped["meta"] == {"version": "1.1"}
    assert case["meta"]["changelog"], "strip_changelog must not mutate its input"


def test_appending_a_changelog_entry_does_not_move_the_snapshot(dataset_copy):
    """The bootstrap paradox: writing the reason for a bump must not be a bump.

    Without this exclusion the contract is circular, because the changelog
    entry explaining a version change would itself be a content change
    demanding another version change.
    """
    before = dataset_snapshot(dataset_copy).snapshot_id

    path = dataset_copy / "math_canonical.json"
    case = json.loads(path.read_text())
    case["meta"].setdefault("changelog", []).append(
        {"version": case["meta"]["version"], "reason": "seeded"}
    )
    path.write_text(json.dumps(case, indent=2))

    assert dataset_snapshot(dataset_copy).snapshot_id == before


# ----------------------------------------------------------------------
# Dataset snapshot
# ----------------------------------------------------------------------

def test_snapshot_is_stable_across_repeated_calls():
    assert dataset_snapshot(DATASET).snapshot_id == dataset_snapshot(DATASET).snapshot_id


def test_snapshot_accepts_a_string_path():
    """cli.py may hand over a str. A crash here kills the run before it starts."""
    assert dataset_snapshot(str(DATASET)).snapshot_id is not None


def test_reformatting_a_case_does_not_move_the_snapshot(dataset_copy):
    before = dataset_snapshot(dataset_copy).snapshot_id
    path = dataset_copy / "math_canonical.json"
    case = json.loads(path.read_text())
    path.write_text(json.dumps(case, indent=8))  # same content, different bytes
    assert dataset_snapshot(dataset_copy).snapshot_id == before


def test_changing_a_prompt_moves_the_snapshot(dataset_copy):
    before = dataset_snapshot(dataset_copy).snapshot_id
    path = dataset_copy / "math_canonical.json"
    case = json.loads(path.read_text())
    case["conversation"]["messages"][0]["content"] += " Show your working."
    path.write_text(json.dumps(case, indent=2))
    assert dataset_snapshot(dataset_copy).snapshot_id != before


def test_changing_the_expected_answer_moves_the_snapshot(dataset_copy):
    """Schema and scoring expectations are pinned by subsumption, not directly."""
    before = dataset_snapshot(dataset_copy).snapshot_id
    path = dataset_copy / "math_canonical.json"
    case = json.loads(path.read_text())
    case["golden_truth"]["expected_outputs"] = ["999999"]
    path.write_text(json.dumps(case, indent=2))
    assert dataset_snapshot(dataset_copy).snapshot_id != before


def test_changing_a_tool_declaration_moves_the_snapshot(dataset_copy):
    """Tool permissions live in the case file, so the dataset hash covers them."""
    before = dataset_snapshot(dataset_copy).snapshot_id
    path = dataset_copy / "tool_call_minimal.json"
    case = json.loads(path.read_text())
    assert "tools" in case, "fixture assumption: this case declares tools"
    case["tools"][0]["function"]["name"] = "renamed_tool"
    path.write_text(json.dumps(case, indent=2))
    assert dataset_snapshot(dataset_copy).snapshot_id != before


def test_adding_a_case_moves_the_snapshot_and_the_count(dataset_copy):
    before = dataset_snapshot(dataset_copy)
    (dataset_copy / "zzz_new_case.json").write_text(json.dumps({
        "id": "zzz_new_case",
        "title": "new",
        "category": "misc",
        "conversation": {"messages": [{"role": "user", "content": "hi"}]},
        "golden_truth": {"match_type": "contains_all", "expected_outputs": ["hi"]},
        "meta": {"version": "1.0"},
    }))
    after = dataset_snapshot(dataset_copy)
    assert after.snapshot_id != before.snapshot_id
    assert after.case_count == before.case_count + 1


def test_deleting_a_case_moves_the_snapshot(dataset_copy):
    before = dataset_snapshot(dataset_copy).snapshot_id
    (dataset_copy / "math_canonical.json").unlink()
    assert dataset_snapshot(dataset_copy).snapshot_id != before


def test_a_subset_run_is_a_different_condition(dataset_copy):
    """A filtered run asked fewer questions, so it is not the same scaffold."""
    full = dataset_snapshot(dataset_copy).snapshot_id
    subset = dataset_snapshot(dataset_copy, selected_tasks=["math_canonical"]).snapshot_id
    assert subset != full


def test_missing_dataset_dir_degrades_instead_of_raising(tmp_path):
    snap = dataset_snapshot(tmp_path / "does_not_exist")
    assert snap.snapshot_id is None
    assert snap.case_count == 0


def test_case_versions_are_reported_for_every_case():
    snap = dataset_snapshot(DATASET)
    assert len(snap.case_versions) == snap.case_count
    assert all(v for v in snap.case_versions.values())


def test_case_count_is_derived_and_cannot_disagree_with_the_versions():
    """A stored count is a second source of truth waiting to drift."""
    snap = DatasetSnapshot(snapshot_id="gd_test", case_versions={"a": "1.0", "b": "1.1"})
    assert snap.case_count == 2
    assert snap.to_dict()["case_count"] == 2


# ----------------------------------------------------------------------
# Subject
# ----------------------------------------------------------------------

def test_subject_accepts_an_enum_serving_format():
    """cli.py holds a ModelFormat enum, which is not JSON serializable.

    The scaffold is built at run start, so a TypeError here would kill the
    whole run before a single model loaded rather than degrading gracefully.
    """
    subject = build_subject(_Model(), ModelFormat.HF)
    assert subject.serving_format == "hf"
    assert subject.subject_id.startswith("sb_")


def test_subject_accepts_a_plain_string_serving_format():
    assert build_subject(_Model(), "hf").subject_id == \
        build_subject(_Model(), ModelFormat.HF).subject_id


def test_an_unsupported_serving_format_type_fails_loudly():
    """Fail at the boundary, not later inside json.dump with a useless message."""
    with pytest.raises(TypeError, match="serving_format"):
        build_subject(_Model(), object())


def test_subject_is_honest_that_nothing_is_pinned():
    """The block is identity, not a pin. No HF revision is recorded anywhere.

    Its ``unpinned`` names the subject fields only: the scaffold fields are
    the scaffold's to report.
    """
    subject = build_subject(_Model(), ModelFormat.HF)
    assert subject.covers == ()
    assert set(subject.unpinned) == set(SUBJECT_FIELDS)


def test_quantization_changes_the_subject_but_not_the_scaffold():
    """The central design claim, and the reason for two axes instead of one.

    A single combined hash would give these two different ids and mark the
    comparison illegal, which is precisely the comparison a compression sweep
    exists to make.
    """
    bf16 = build_condition(Pillar.GOLDEN, _Model(), ModelFormat.HF, dataset_dir=DATASET)
    q4 = build_condition(Pillar.GOLDEN, _Model(**_QUANT), ModelFormat.HF, dataset_dir=DATASET)
    assert bf16.subject.subject_id != q4.subject.subject_id
    assert bf16.scaffold.scaffold_id == q4.scaffold.scaffold_id


# ----------------------------------------------------------------------
# Scaffold blocks
# ----------------------------------------------------------------------

def test_golden_scaffold_covers_what_it_claims_with_and_without_a_config():
    """The environment is the only field a config unlocks for golden.

    Asserted as sets rather than as a count, so a coverage change reads
    as the specific field that moved instead of as an arithmetic diff.
    """
    with_config = build_golden_scaffold(DATASET, config=_Config())
    assert set(with_config.covers) == set(GOLDEN_COVERS)
    assert set(with_config.unpinned) == set(SCAFFOLD_FIELDS) - set(GOLDEN_COVERS)
    assert with_config.contract_version == CONTRACT_VERSION

    without = build_golden_scaffold(DATASET)
    assert set(without.covers) == set(GOLDEN_COVERS) - {
        ContractField.SERVING_ENVIRONMENT
    }


def test_golden_never_claims_a_load_shape():
    """Golden runs sequentially and nothing records that as a decision.

    It has a concurrency, it is one, and it is not pinned. Exempting the
    pillar from the field would read as "does not apply" when the honest
    reading is "applies and is not captured".
    """
    scaffold = build_golden_scaffold(DATASET, config=_Config())
    assert ContractField.LOAD_SHAPE in scaffold.unpinned


def test_covers_and_unpinned_always_partition_the_scaffold_fields():
    """`unpinned` is derived, never hand maintained, so it cannot rot.

    A scaffold partitions the scaffold fields, and a subject the subject
    fields; the two axes together are the contract.
    """
    for scaffold in (build_golden_scaffold(DATASET),
                     build_unmodelled_scaffold(Pillar.STRESS_TEST)):
        assert not set(scaffold.covers) & set(scaffold.unpinned)
        assert set(scaffold.covers) | set(scaffold.unpinned) == set(SCAFFOLD_FIELDS)
    subject = build_subject(_Model(), ModelFormat.HF)
    assert set(subject.covers) | set(subject.unpinned) == set(SUBJECT_FIELDS)
    assert set(SCAFFOLD_FIELDS) | set(SUBJECT_FIELDS) == set(CONTRACT_FIELDS)


def test_pinned_holds_exactly_what_covers_claims():
    """An id that hashed fewer fields than it advertises would over-promise."""
    scaffold = build_golden_scaffold(DATASET)
    hashed_directly = {ContractField(k) for k in scaffold.pinned if k in
                       {f.value for f in ContractField}}
    subsumed = {ContractField.PROMPT_VERSION, ContractField.SCHEMA_VERSION,
                ContractField.TOOL_PERMISSIONS}
    assert hashed_directly | subsumed == set(scaffold.covers)


def test_an_unmodelled_pillar_has_a_null_id_rather_than_being_omitted():
    """A visible gap is information. An absent pillar looks like one that did not run."""
    scaffold = build_unmodelled_scaffold(Pillar.STRESS_TEST)
    assert scaffold.scaffold_id is None
    assert scaffold.covers == ()


def test_unmodelled_pillars_do_not_all_collide_on_one_id():
    assert build_unmodelled_scaffold("stress_test").pillar is Pillar.STRESS_TEST
    assert build_unmodelled_scaffold("quality").pillar is Pillar.QUALITY


def test_an_unknown_pillar_name_is_rejected():
    """A reader that matches benchmark types by name would miss a typo silently."""
    with pytest.raises(ValueError):
        build_unmodelled_scaffold("stress")


def test_golden_scaffold_pins_the_decode_settings_actually_used():
    from gbench.runners.golden import DEFAULT_SAMPLING
    assert build_golden_scaffold(DATASET).pinned["decode"] == DEFAULT_SAMPLING


def test_golden_scaffold_pins_the_attempt_policy_actually_used():
    from gbench.runners.golden import MAX_ATTEMPTS
    assert build_golden_scaffold(DATASET).pinned["attempts"] == {"max_attempts": MAX_ATTEMPTS}


def test_golden_pins_the_scoring_version_the_runner_declares():
    """Declared rather than derived, so an unrelated edit cannot break a series."""
    from gbench.runners.golden import SCORING_VERSION
    assert build_golden_scaffold(DATASET).pinned["scoring_version"] == SCORING_VERSION


# ----------------------------------------------------------------------
# The config seam
# ----------------------------------------------------------------------
#
# `ConfigLike` is a Protocol, so it is structural and nothing verifies it
# at runtime. A field renamed in `core/config.py` would not fail an
# import, it would fail at run start when the scaffold is built, killing
# the run before a model loaded. These two tests are the check that
# Protocol does not give us.

def test_the_config_protocol_only_reads_fields_the_real_config_has():
    from gbench.core.config import BenchmarkConfig
    from gbench.core.scaffold import ConfigLike

    # Annotations rather than an instance, because constructing a
    # BenchmarkConfig runs __post_init__ and creates directories.
    missing = set(ConfigLike.__annotations__) - set(BenchmarkConfig.__annotations__)
    assert not missing, f"ConfigLike reads fields BenchmarkConfig does not have: {missing}"


def test_the_stand_in_covers_every_field_the_contract_reads():
    """Otherwise a test configures nothing and passes for the wrong reason."""
    from gbench.core.scaffold import ConfigLike

    stub = _Config()
    absent = [name for name in ConfigLike.__annotations__ if not hasattr(stub, name)]
    assert not absent, f"_Config is missing {absent}"


# ----------------------------------------------------------------------
# Every pillar, not just golden
# ----------------------------------------------------------------------
#
# Before the run configuration was modelled, seven of the eight pillars
# emitted an empty block and reported 0/N. These tests are what stop that
# regressing quietly, because an empty block is still a valid block.

def test_every_pillar_pins_something_once_a_config_is_supplied():
    for pillar, scaffold in _scaffolds_for_every_pillar(_Config()).items():
        assert scaffold.covers, f"{pillar.value} still pins nothing"
        assert scaffold.scaffold_id, f"{pillar.value} has a null id despite covering fields"


def test_a_pillar_with_no_config_still_reports_the_gap_rather_than_raising():
    """`cli.py` always passes one, but the contract must not depend on it.

    A missing config yields the empty block this module emitted before,
    which is visible in the table. Raising would kill the run at start
    over a reporting concern.
    """
    for pillar, scaffold in _scaffolds_for_every_pillar().items():
        if pillar is Pillar.GOLDEN:
            continue  # golden pins the dataset without a config
        assert scaffold.covers == ()
        assert scaffold.scaffold_id is None


def test_every_pillar_pins_exactly_what_it_claims_to_cover():
    """Generalises the golden-only invariant to all eight builders.

    Over-promising is the failure this whole module exists to prevent, so
    the check has to run against every block rather than against the one
    block that happened to be written first.
    """
    for pillar, scaffold in _scaffolds_for_every_pillar(_Config()).items():
        hashed_directly = {
            ContractField(k) for k in scaffold.pinned if k in _FIELD_VALUES
        }
        assert hashed_directly | _SUBSUMED.get(pillar, set()) == set(scaffold.covers), (
            f"{pillar.value} claims coverage it does not hash, or hashes a field "
            "it does not claim"
        )


def test_no_pillar_pins_a_field_from_the_subject_axis():
    """The two axes have to stay separate or the quant sweep stops being legal."""
    for pillar, scaffold in _scaffolds_for_every_pillar(_Config()).items():
        leaked = set(scaffold.covers) & {
            ContractField.MODEL_CHECKPOINT, ContractField.QUANTIZATION
        }
        assert not leaked, f"{pillar.value} pins subject fields {leaked} on the scaffold"


# ----------------------------------------------------------------------
# Serving environment
# ----------------------------------------------------------------------

def test_a_remote_endpoint_is_hashed_rather_than_written_in_the_clear():
    """Detection without disclosure. This repo publishes to a public org."""
    env = build_serving_environment(_Config(remote_endpoint=REMOTE_ENDPOINT))
    serialised = json.dumps(env.to_dict())

    assert env.endpoint.startswith("ep_")
    assert "run.app" not in serialised
    assert "abc123" not in serialised


def test_a_loopback_endpoint_is_kept_readable():
    """Nothing to protect there, and a readable value is easier to check.

    ``gbench_started_server`` is still False, because the endpoint was
    given rather than started here. Loopback says where, not who.
    """
    env = build_serving_environment(_Config(remote_endpoint="http://127.0.0.1:8000/v1"))
    assert env.endpoint == "http://127.0.0.1:8000/v1"
    assert env.gbench_started_server is False


def test_a_trailing_slash_is_not_a_new_serving_environment():
    a = build_serving_environment(_Config(remote_endpoint="http://127.0.0.1:8000/v1"))
    b = build_serving_environment(_Config(remote_endpoint="http://127.0.0.1:8000/v1/"))
    assert a.endpoint == b.endpoint


def test_localhost_and_the_loopback_address_are_deliberately_not_aliased():
    """They usually reach the same server and the contract does not get to assume it.

    Treating two addresses as equivalent is the speculative mapping that
    turns a real difference into a silent one. A spurious break here
    costs a glance at a diff.
    """
    a = build_serving_environment(_Config(remote_endpoint="http://localhost:8000/v1"))
    b = build_serving_environment(_Config(remote_endpoint="http://127.0.0.1:8000/v1"))
    assert a.endpoint != b.endpoint


def test_moving_a_run_to_a_remote_endpoint_moves_the_scaffold_id():
    """The gap this closes: golden already runs against ``remote_endpoint``.

    Before the environment was pinned, the one pillar shipping real
    coverage produced the same id whether it hit a local vLLM or a Cloud
    Run deployment on different hardware.
    """
    local = build_golden_scaffold(DATASET, config=_Config())
    remote = build_golden_scaffold(
        DATASET, config=_Config(remote_endpoint=REMOTE_ENDPOINT))
    assert local.scaffold_id != remote.scaffold_id


def test_a_remote_run_leaves_the_vllm_flags_unset():
    """`cli.py` skips the GPU check on that path, so the flags never applied."""
    env = build_serving_environment(_Config(remote_endpoint=REMOTE_ENDPOINT))
    assert env.gbench_started_server is False
    assert env.num_gpus is None
    assert env.max_model_len is None


def test_a_local_run_records_the_flags_it_actually_passed_to_vllm():
    from gbench.core.config import get_max_model_len
    env = build_serving_environment(_Config())
    assert env.gbench_started_server is True
    assert env.num_gpus == 1
    assert env.gpu_memory_utilization == 0.90
    assert env.max_model_len == get_max_model_len()


def test_an_inert_flag_cannot_break_a_remote_series():
    """A false match is worse than a false break, but this is neither.

    ``tensor_parallel_size`` reaches no process gbench owns on a remote
    run, so two runs differing only in it are the same condition. Hashing
    the configured value would break the series for a setting that had no
    effect on the numbers.
    """
    a = build_stress_scaffold(_Config(remote_endpoint=REMOTE_ENDPOINT,
                                      tensor_parallel_size=1))
    b = build_stress_scaffold(_Config(remote_endpoint=REMOTE_ENDPOINT,
                                      tensor_parallel_size=8))
    assert a.scaffold_id == b.scaffold_id


def test_the_same_flag_does_break_a_local_series():
    """The other half of the pair. Locally it is the server we started."""
    a = build_stress_scaffold(_Config(tensor_parallel_size=1))
    b = build_stress_scaffold(_Config(tensor_parallel_size=8))
    assert a.scaffold_id != b.scaffold_id


# ----------------------------------------------------------------------
# Load shape
# ----------------------------------------------------------------------

def test_the_sweep_is_pinned_rather_than_a_point_inside_it():
    """One curve, one ``scaffold_id``. That is what makes the plot legal.

    A serving run writes one result file per batch size. If the id were
    per point, those four files would carry four scaffolds and plotting
    them as one line would mean comparing four harnesses. Pinning the
    sweep definition gives them one id by construction, and which point a
    file holds is already in its name.
    """
    scaffold = build_performance_scaffold(Pillar.SERVING, _Config())
    assert scaffold.pinned[LOAD_SHAPE]["batch_sizes"] == [1, 16, 50, 100]


def test_a_different_sweep_is_a_different_curve_and_breaks_the_series():
    base = build_performance_scaffold(Pillar.SERVING, _Config())
    widened = build_performance_scaffold(
        Pillar.SERVING, _Config(batch_sizes=[1, 16, 50, 100, 256]))
    assert widened.scaffold_id != base.scaffold_id


def test_serving_pins_the_lengths_it_uses_and_only_those():
    """Serving reads the first input and output length (its request shape).

    A different first length is a different workload and breaks the series;
    a length only throughput sweeps does not.
    """
    base = build_performance_scaffold(Pillar.SERVING, _Config())
    assert base.pinned[LOAD_SHAPE]["input_lengths"] == [128]
    assert base.pinned[LOAD_SHAPE]["output_lengths"] == [512]
    for pillar in (Pillar.SERVING, Pillar.SERVING_MULTIMODAL):
        a = build_performance_scaffold(pillar, _Config())
        assert build_performance_scaffold(
            pillar, _Config(input_lengths=[2048])).scaffold_id != a.scaffold_id
        assert build_performance_scaffold(
            pillar, _Config(output_lengths=[64])).scaffold_id != a.scaffold_id
        assert build_performance_scaffold(
            pillar, _Config(input_lengths=[128, 4096])).scaffold_id == a.scaffold_id


def test_throughput_does_pin_the_lengths_it_sweeps():
    """The other half. Prefill-heavy and decode-heavy are different conditions."""
    base = build_performance_scaffold(Pillar.THROUGHPUT, _Config())
    moved = build_performance_scaffold(Pillar.THROUGHPUT, _Config(input_lengths=[2048]))
    assert base.pinned[LOAD_SHAPE]["input_lengths"] == [128]
    assert moved.scaffold_id != base.scaffold_id


def test_only_the_multimodal_pillars_pin_the_image_count():
    """The images each request carries: the runner's default, or the override."""
    from gbench.runners.serving import ServingBenchmarkRunner

    text = build_performance_scaffold(Pillar.SERVING, _Config())
    multimodal = build_performance_scaffold(Pillar.SERVING_MULTIMODAL, _Config())
    assert text.pinned[LOAD_SHAPE]["images_per_request"] is None
    assert multimodal.pinned[LOAD_SHAPE]["images_per_request"] == [
        ServingBenchmarkRunner.IMAGES_PER_REQUEST]
    one = build_performance_scaffold(Pillar.SERVING_MULTIMODAL,
                                     _Config(mm_images_per_request=1))
    assert one.pinned[LOAD_SHAPE]["images_per_request"] == [1]
    assert one.scaffold_id != multimodal.scaffold_id


def test_throughput_counts_its_prompts_off_the_throughput_setting():
    """Two prompt counts exist and the pillars read different ones.

    Throughput sends num_prompts_throughput; serving sizes its count per
    batch size (BenchmarkConfig._serving_num_prompts) and never reads
    --num-prompts, so it pins the counts it sends.
    """
    from gbench.core.config import BenchmarkConfig

    serving = build_performance_scaffold(Pillar.SERVING, _Config())
    throughput = build_performance_scaffold(
        Pillar.THROUGHPUT, _Config(num_prompts_throughput=750))
    assert throughput.pinned[LOAD_SHAPE]["num_prompts"] == 750
    assert serving.pinned[LOAD_SHAPE]["num_prompts"] is None
    assert serving.pinned[LOAD_SHAPE]["num_prompts_per_batch"] == [
        BenchmarkConfig._serving_num_prompts(_Config(), b)[0] for b in [1, 16, 50, 100]]
    personal = build_performance_scaffold(Pillar.SERVING,
                                          _Config(serving_num_prompts_override=12))
    assert personal.pinned[LOAD_SHAPE]["num_prompts_per_batch"] == [12, 12, 12, 12]
    assert personal.scaffold_id != serving.scaffold_id


def test_the_multimodal_pillars_name_their_generated_workload():
    """The multimodal runners build their own images and filler text; no dataset is read."""
    text = build_performance_scaffold(Pillar.SERVING, _Config())
    multimodal = build_performance_scaffold(Pillar.SERVING_MULTIMODAL, _Config())
    assert text.pinned["dataset_snapshot"]["dataset"] == "random"
    assert multimodal.pinned["dataset_snapshot"]["dataset"] == "generated"


def test_a_performance_dataset_says_it_was_not_content_hashed():
    """Weaker than the golden snapshot, and it must not read as the stronger one.

    Naming the source catches a switch from one set to another. It does
    not catch a mutated file at the same path, which the golden hash
    would.
    """
    scaffold = build_performance_scaffold(Pillar.SERVING, _Config())
    assert scaffold.pinned["dataset_snapshot"]["content_hashed"] is False


# ----------------------------------------------------------------------
# Quality and evals
# ----------------------------------------------------------------------

def test_quality_pins_the_resolved_scorer_commit():
    """``cli.py`` resolves the ref to a sha before this builder is called.

    The builder deliberately does not resolve. One resolution per run is
    what makes the sha hashed here the same one the runner checks out.
    Resolving per condition would let ``main`` advance mid-loop.
    """
    sha = "08c0584d660591fa713928df9249e4ec37322ea5"
    pinned = build_quality_scaffold(_Config(gemmaclaw_commit=sha)).pinned[
        ContractField.SCORING_VERSION.value
    ]
    assert pinned == {"gemmaclaw_commit": sha}


def test_an_unresolvable_scorer_ref_is_pinned_as_itself():
    """Honest about the gap rather than guessing at a sha.

    This is the one remaining case where two runs a week apart can share
    an id while the scorer moved underneath them. It also predicts a
    failed run, because a ref the resolver could not resolve is a ref the
    runner cannot check out either.
    """
    pinned = build_quality_scaffold(_Config()).pinned[
        ContractField.SCORING_VERSION.value
    ]
    assert pinned == {"gemmaclaw_commit": "main"}


def test_a_different_gemmaclaw_commit_is_a_different_quality_scaffold():
    base = build_quality_scaffold(_Config())
    pinned_sha = build_quality_scaffold(_Config(gemmaclaw_commit="a1b2c3d"))
    assert pinned_sha.scaffold_id != base.scaffold_id


def test_a_scenario_subset_is_a_different_quality_scaffold():
    """A filtered run asked fewer questions, exactly as with golden.

    The value is a path relative to ``qa/scenarios/`` in the GemmaClaw
    repo, which gbench neither vendors nor validates. The contract
    records what the run was told to execute.
    """
    full = build_quality_scaffold(_Config())
    subset = build_quality_scaffold(_Config(selected_scenarios=["smoke.yaml"]))
    assert subset.scaffold_id != full.scaffold_id


def test_evals_concurrency_is_the_value_the_runner_uses():
    """The runner's own resolution (runners.evals.eval_concurrency) is pinned.

    --eval-concurrency when set, else ``batch_sizes[0]``, else 1.
    """
    assert build_evals_scaffold(_Config()).pinned[LOAD_SHAPE]["concurrency"] == 1
    assert build_evals_scaffold(_Config(batch_sizes=[])).pinned[LOAD_SHAPE]["concurrency"] == 1
    explicit = build_evals_scaffold(_Config(eval_concurrency=16))
    assert explicit.pinned[LOAD_SHAPE]["concurrency"] == 16


def test_the_few_shot_count_and_thinking_flag_are_part_of_the_prompt():
    """Neither needs a field of its own. Both change what the model was asked."""
    base = build_evals_scaffold(_Config())
    assert base.pinned["prompt_version"] == {"n_shot": 0, "thinking": False, "env": {}}

    for changed in (_Config(eval_n_shot=5), _Config(eval_thinking=True)):
        assert build_evals_scaffold(changed).scaffold_id != base.scaffold_id


def test_which_eval_suites_ran_is_part_of_the_dataset():
    base = build_evals_scaffold(_Config())
    subset = build_evals_scaffold(_Config(evals=["gsm8k", "mmlu"]))
    assert base.pinned["dataset_snapshot"]["evals"] is None
    assert subset.pinned["dataset_snapshot"]["evals"] == ["gsm8k", "mmlu"]
    assert subset.scaffold_id != base.scaffold_id


def test_the_suite_list_is_order_independent():
    """A reordered CLI flag is the same run, so it must not break the series."""
    a = build_evals_scaffold(_Config(evals=["mmlu", "gsm8k"]))
    b = build_evals_scaffold(_Config(evals=["gsm8k", "mmlu"]))
    assert a.scaffold_id == b.scaffold_id


# ----------------------------------------------------------------------
# Resolving the scorer ref
# ----------------------------------------------------------------------

def _git(repo, *argv):
    subprocess.run(["git", "-C", str(repo), *argv], check=True, capture_output=True)


@pytest.fixture
def scorer_repo(tmp_path):
    """A real git repo standing in for GemmaClaw.

    Real rather than mocked because the thing under test is how git's
    own output is parsed, and `git ls-remote` is happy to treat a local
    directory as a remote. A mock would only assert that the parser
    agrees with my memory of the output format.
    """
    repo = tmp_path / "gemmaclaw"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "scorer.txt").write_text("v1\n")
    _git(repo, "add", "scorer.txt")
    _git(repo, "commit", "-q", "-m", "first")
    return repo


def _head(repo):
    done = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return done.stdout.strip()


def test_a_sha_resolves_to_itself_without_touching_git():
    """Already a pin. The fast path is what keeps a pinned run offline.

    The remote is set to a path that cannot exist, so a subprocess call
    would fail the resolution rather than silently succeed.
    """
    sha = "08c0584d660591fa713928df9249e4ec37322ea5"
    assert resolve_gemmaclaw_sha(sha, repo_path="/nonexistent") == sha


def test_a_branch_resolves_against_a_local_checkout(scorer_repo):
    """``--gemmaclaw-path`` makes the local repo the authority.

    The runner checks out in place and never fetches in that mode, so
    the remote's tip is not what the run would use.
    """
    assert resolve_gemmaclaw_sha("main", repo_path=str(scorer_repo)) == _head(scorer_repo)


def test_an_annotated_tag_resolves_to_the_commit_it_points_at(scorer_repo):
    """A tag object's own sha is not where a checkout of that tag lands."""
    _git(scorer_repo, "tag", "-a", "v1", "-m", "release")
    resolved = resolve_gemmaclaw_sha("v1", repo_path=str(scorer_repo))
    assert resolved == _head(scorer_repo)


def test_a_branch_resolves_against_the_remote(scorer_repo, monkeypatch):
    """The default path, with a local directory standing in as the remote."""
    monkeypatch.setattr(scaffold, "GEMMACLAW_REMOTE", str(scorer_repo))
    assert resolve_gemmaclaw_sha("main") == _head(scorer_repo)


def test_an_annotated_tag_from_the_remote_is_peeled(scorer_repo, monkeypatch):
    """``ls-remote`` lists a tag twice and only the ``^{}`` line is the commit.

    Taking the first line would pin the tag object's sha, which is not a
    commit at all and which no checkout would ever produce.
    """
    _git(scorer_repo, "tag", "-a", "v1", "-m", "release")
    monkeypatch.setattr(scaffold, "GEMMACLAW_REMOTE", str(scorer_repo))
    assert resolve_gemmaclaw_sha("v1") == _head(scorer_repo)


def test_a_missing_ref_resolves_to_none_rather_than_a_guess(scorer_repo, monkeypatch):
    """``ls-remote`` exits 0 with no output for a ref that does not exist.

    So the empty stdout has to be read as a miss. Treating the exit code
    alone as success would pin an empty string.
    """
    monkeypatch.setattr(scaffold, "GEMMACLAW_REMOTE", str(scorer_repo))
    assert resolve_gemmaclaw_sha("no-such-branch") is None
    assert resolve_gemmaclaw_sha("no-such-branch", repo_path=str(scorer_repo)) is None


def test_resolution_survives_git_being_absent(monkeypatch):
    """A benchmark run is not worth failing over a missing git binary."""
    def _boom(*_args, **_kwargs):
        raise OSError("no git here")

    monkeypatch.setattr(scaffold.subprocess, "run", _boom)
    assert resolve_gemmaclaw_sha("main") is None


def test_resolving_moves_the_quality_scaffold_id(scorer_repo, monkeypatch):
    """The whole point, end to end.

    A run that pins the literal string ``main`` and a run that pins what
    ``main`` resolved to are different scaffolds. Before resolution both
    advances of the branch produced one id, which is a false match.
    """
    monkeypatch.setattr(scaffold, "GEMMACLAW_REMOTE", str(scorer_repo))
    unresolved = build_quality_scaffold(_Config(gemmaclaw_commit="main"))

    first = build_quality_scaffold(_Config(gemmaclaw_commit=resolve_gemmaclaw_sha("main")))
    (scorer_repo / "scorer.txt").write_text("v2\n")
    _git(scorer_repo, "commit", "-qam", "second")
    second = build_quality_scaffold(_Config(gemmaclaw_commit=resolve_gemmaclaw_sha("main")))

    assert first.scaffold_id != unresolved.scaffold_id
    assert first.scaffold_id != second.scaffold_id


# ----------------------------------------------------------------------
# The pinned default scorer
# ----------------------------------------------------------------------

def test_the_default_scorer_is_a_sha_and_not_a_branch():
    """The default is the configuration almost every run uses.

    Pointing it at a branch would make two unflagged runs a week apart
    two different experiments, and the id would correctly move to say so,
    which breaks the series for a reason nobody chose. Promoting the pin
    is a reviewable commit instead.
    """
    assert len(DEFAULT_GEMMACLAW_COMMIT) == 40
    assert all(c in "0123456789abcdef" for c in DEFAULT_GEMMACLAW_COMMIT)


def test_the_default_scorer_needs_no_network_to_resolve():
    """A bare ``gbench --quality-only`` must not depend on GitHub being up.

    ``git`` is made to explode, so any subprocess call at all fails the
    resolution rather than quietly succeeding on a cached answer.
    """
    with mock.patch.object(
        scaffold.subprocess, "run", autospec=True, side_effect=AssertionError("called git")
    ):
        assert resolve_gemmaclaw_sha(DEFAULT_GEMMACLAW_COMMIT) == DEFAULT_GEMMACLAW_COMMIT


def test_the_cli_flag_and_the_config_field_share_one_default():
    """Two spellings of the default would drift the moment one is promoted.

    ``BenchmarkConfig`` is what a library caller gets and argparse is what
    a CLI caller gets, so a mismatch means the same command scores against
    two different commits depending on the entry point.
    """
    assert create_parser().get_default("gemmaclaw_commit") == DEFAULT_GEMMACLAW_COMMIT
    assert BenchmarkConfig().gemmaclaw_commit == DEFAULT_GEMMACLAW_COMMIT


def test_the_default_scorer_pins_a_stable_quality_scaffold():
    """The whole reason for the pin, stated as an id.

    A default run and an explicit ``--gemmaclaw-commit <the same sha>``
    are the same experiment and must share an id. A different commit must
    not.
    """
    default = build_quality_scaffold(_Config(gemmaclaw_commit=DEFAULT_GEMMACLAW_COMMIT))
    explicit = build_quality_scaffold(_Config(gemmaclaw_commit=DEFAULT_GEMMACLAW_COMMIT))
    other = build_quality_scaffold(
        _Config(gemmaclaw_commit="08c0584d660591fa713928df9249e4ec37322ea5")
    )
    assert default.scaffold_id == explicit.scaffold_id
    assert default.scaffold_id != other.scaffold_id


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------

def test_gbench_commit_reports_a_sha_and_whether_the_tree_was_dirty():
    """A sha alone is the more confident of the two possible lies.

    Most gbench runs happen on a workstation with uncommitted edits, so a
    sha without the dirty flag claims a published tree that never
    existed.
    """
    provenance = gbench_commit()
    assert set(provenance) == {"commit", "dirty"}
    json.dumps(provenance)

    if provenance["commit"] is None:
        # An installed wheel, or a container built by COPY. A real
        # deployment shape, and one where unknown beats optimistic.
        assert provenance["dirty"] is None
        return

    assert len(provenance["commit"]) == 40
    assert provenance["dirty"] in (True, False)


def test_the_gbench_commit_is_not_an_input_to_any_id():
    """As a hash input it would invalidate every scaffold on every commit.

    Asserted against the sha itself rather than against the word, because
    ``gemmaclaw_commit`` is a legitimate pin. It names the scorer's
    revision, which is a property of how we asked, not this harness's.
    """
    sha = gbench_commit()["commit"]
    if sha is None:
        pytest.skip("not a git checkout, so there is no sha that could leak")

    for pillar, scaffold in _scaffolds_for_every_pillar(_Config()).items():
        assert sha not in json.dumps(scaffold.pinned), pillar.value


# ----------------------------------------------------------------------
# Conditions
# ----------------------------------------------------------------------

def test_condition_carries_both_axes_and_the_pillar():
    cond = build_condition(Pillar.GOLDEN, _Model(), ModelFormat.HF, dataset_dir=DATASET)
    assert cond.pillar is Pillar.GOLDEN
    assert cond.model == "gemma-4-E2B-it"
    assert cond.serving_format == "hf"
    assert cond.scaffold.scaffold_id.startswith("sc_")
    assert cond.subject.subject_id.startswith("sb_")


def test_a_pillar_may_be_passed_as_a_plain_string():
    """cli.py builds its plan from string literals, so both forms must work."""
    assert build_condition("quality", _Model(), "hf").pillar is Pillar.QUALITY


def test_two_models_over_two_pillars_produce_four_distinct_conditions():
    """The case a run level scalar could not have represented."""
    models = [_Model(name="a", hf_model_id="org/a"), _Model(name="b", hf_model_id="org/b")]
    conditions = [
        build_condition(
            pillar, m, ModelFormat.HF,
            dataset_dir=DATASET if pillar is Pillar.GOLDEN else None,
        )
        for m in models
        for pillar in (Pillar.GOLDEN, Pillar.QUALITY)
    ]
    assert len(conditions) == 4
    assert len({(c.pillar, c.model) for c in conditions}) == 4


def test_same_model_same_scaffold_is_reproducible():
    a = build_condition(Pillar.GOLDEN, _Model(), ModelFormat.HF, dataset_dir=DATASET)
    b = build_condition(Pillar.GOLDEN, _Model(), ModelFormat.HF, dataset_dir=DATASET)
    assert a.scaffold.scaffold_id == b.scaffold.scaffold_id
    assert a.subject.subject_id == b.subject.subject_id


# ----------------------------------------------------------------------
# Serialisation, the single seam that reaches metadata.json
# ----------------------------------------------------------------------

def test_a_condition_survives_json_dump():
    """Every defect found building this module died here. Enums included."""
    cond = build_condition(Pillar.GOLDEN, _Model(), ModelFormat.HF, dataset_dir=DATASET)
    round_tripped = json.loads(json.dumps(cond.to_dict()))
    assert round_tripped["pillar"] == "golden"
    assert round_tripped["format"] == "hf"


def test_serialised_contract_fields_are_plain_strings_not_enum_reprs():
    """Result readers read these labels directly, so `ContractField.DECODE` would show."""
    scaffold = build_golden_scaffold(DATASET).to_dict()
    assert "decode" in scaffold["covers"]
    assert "load_shape" in scaffold["unpinned"]
    assert "model_checkpoint" in build_subject(_Model(), ModelFormat.HF).to_dict()["unpinned"]
    assert all(isinstance(f, str) and "." not in f
               for f in scaffold["covers"] + scaffold["unpinned"])


def test_every_object_in_the_contract_serialises():
    """One `to_dict` seam per object, so there is one place to fix a leak."""
    for obj in (
        dataset_snapshot(DATASET),
        build_subject(_Model(), ModelFormat.HF),
        build_golden_scaffold(DATASET),
        build_unmodelled_scaffold(Pillar.QUALITY),
        build_condition(Pillar.GOLDEN, _Model(), ModelFormat.HF, dataset_dir=DATASET),
    ):
        json.dumps(obj.to_dict())


def test_every_pillar_scaffold_reaches_metadata_json_intact():
    """Five new builders is five new chances at this module's recurring defect.

    The scaffold is built at run start, so a value that will not
    serialise is not a degraded report, it is a dead run.
    """
    for pillar, scaffold in _scaffolds_for_every_pillar(_Config()).items():
        round_tripped = json.loads(json.dumps(scaffold.to_dict()))
        assert round_tripped["pillar"] == pillar.value
        assert round_tripped["scaffold_id"] == scaffold.scaffold_id


def test_the_load_shape_serialises_its_tuples_as_lists():
    """Stored as tuples so the block can be frozen, written as JSON arrays."""
    load = LoadShape(batch_sizes=(1, 32), input_lengths=(128,))
    written = json.loads(json.dumps(load.to_dict()))
    assert written["batch_sizes"] == [1, 32]
    assert written["output_lengths"] is None


def test_the_serving_environment_serialises_when_nothing_is_known():
    env = ServingEnvironment(gbench_started_server=True, endpoint=None)
    assert json.loads(json.dumps(env.to_dict()))["num_gpus"] is None


def test_an_unmodelled_scaffold_omits_the_dataset_annotations():
    """Reporting `case_count: 0` for a pillar with no dataset would read as a bug."""
    scaffold = build_unmodelled_scaffold(Pillar.QUALITY).to_dict()
    assert "case_versions" not in scaffold
    assert "case_count" not in scaffold


def test_objects_are_frozen_so_a_serialised_id_cannot_be_edited_after_the_fact():
    subject = build_subject(_Model(), ModelFormat.HF)
    with pytest.raises(Exception):
        subject.subject_id = "sb_forged"


def test_the_declared_types_are_the_ones_actually_returned():
    cond = build_condition(Pillar.GOLDEN, _Model(), ModelFormat.HF, dataset_dir=DATASET)
    assert isinstance(cond, Condition)
    assert isinstance(cond.scaffold, Scaffold)
    assert isinstance(cond.subject, Subject)
    assert isinstance(cond.scaffold.dataset, DatasetSnapshot)


# ----------------------------------------------------------------------
# Console rendering
# ----------------------------------------------------------------------
#
# A dashboard is not a dependency. Plenty of gbench runs are headless,
# on a workstation or in CI, and some consumers only ever use the CLI. So
# the console has to carry the same facts `metadata.json` does, and these
# tests pin that equivalence rather than the exact prose.

def test_the_console_shows_every_id_that_metadata_json_shows():
    conditions = [
        build_condition(Pillar.GOLDEN, model, ModelFormat.HF, dataset_dir=DATASET)
        for model in (_Model(), _Model(**_QUANT))
    ]
    out = render_conditions(conditions)

    for condition in conditions:
        assert condition.subject.subject_id in out
        assert condition.scaffold.scaffold_id in out
        assert condition.scaffold.dataset.snapshot_id in out


def test_a_quant_sweep_reads_as_one_scaffold_over_two_subjects():
    """The headline comparison has to be legible without a dashboard."""
    conditions = [
        build_condition(Pillar.GOLDEN, model, ModelFormat.HF, dataset_dir=DATASET)
        for model in (_Model(), _Model(**_QUANT))
    ]
    out = render_conditions(conditions)

    scaffold_id = conditions[0].scaffold.scaffold_id
    assert conditions[1].scaffold.scaffold_id == scaffold_id
    assert conditions[0].subject.subject_id != conditions[1].subject.subject_id

    # Once per table row, then once more for the footer that explains it.
    assert out.count(scaffold_id) == 3


def test_an_unmodelled_pillar_says_so_rather_than_printing_a_bare_dash():
    out = render_conditions([build_condition(Pillar.SERVING, _Model(), ModelFormat.HF)])
    assert f"0/{len(SCAFFOLD_FIELDS)}" in out
    assert "cannot be attributed" in out


def test_every_unmodelled_pillar_is_named_not_just_the_first_one():
    """The null-scaffold footer must not speak for pillars it does not name.

    All three of these share the label "none", so a footer keyed on the
    label alone would report "pins 0/N for serving" and leave a reader
    believing throughput and stress_test were pinned by something else.
    A null scaffold exists to make a gap visible, so a renderer that
    hides two thirds of the gap defeats the whole block.
    """
    conditions = [
        build_condition(pillar, _Model(), ModelFormat.HF)
        for pillar in (Pillar.SERVING, Pillar.THROUGHPUT, Pillar.STRESS_TEST)
    ]
    # Whitespace-normalised, because the footer is wrapped to 80 columns
    # and a phrase may legitimately straddle a line break.
    footer = " ".join(render_conditions(conditions).split()).split("none pins")[-1]

    for pillar in ("serving", "throughput", "stress_test"):
        assert pillar in footer, (
            f"{pillar} shares the 'none' scaffold but is absent from its footer"
        )
    # Still one footer, because they genuinely share one empty scaffold.
    assert footer.count(f"0/{len(SCAFFOLD_FIELDS)}") == 1
    assert "these pillars' harnesses is captured" in footer


def test_a_lone_unmodelled_pillar_still_reads_as_singular():
    footer = " ".join(
        render_conditions(
            [build_condition(Pillar.SERVING, _Model(), ModelFormat.HF)]
        ).split()
    )
    assert "this pillar's harness is captured" in footer
    assert "these pillars'" not in footer


def test_the_scaffold_footer_appears_once_however_many_rows_share_it():
    conditions = [
        build_condition(Pillar.GOLDEN, model, fmt, dataset_dir=DATASET)
        for model, fmt in ((_Model(), ModelFormat.HF), (_Model(**_QUANT), ModelFormat.HF),
                           (_Model(), ModelFormat.REMOTE))
    ]
    out = render_conditions(conditions)
    assert out.count(f"pins 7/{len(SCAFFOLD_FIELDS)}") == 1


def test_unpinned_fields_are_named_not_merely_counted():
    """An honest id is one you can check. A bare count is not checkable.

    Asserted against the text after "unpinned:" rather than against the
    whole block, because the covers line names fields too. Searching the
    whole block would let a pinned field satisfy an assertion about a
    gap, which is how this test quietly stopped testing anything when
    ``scoring_version`` moved into ``covers``.
    """
    condition = build_condition(
        Pillar.GOLDEN, _Model(), ModelFormat.HF, dataset_dir=DATASET)
    disclosure = render_conditions([condition]).split("unpinned:")[1]

    assert condition.scaffold.unpinned, "golden still has gaps, so it must disclose them"
    for field in condition.scaffold.unpinned:
        assert field.value in disclosure, f"{field.value} is unpinned but undisclosed"


def test_rendering_fits_the_eighty_column_banner_the_cli_uses():
    conditions = [
        build_condition(pillar, model, ModelFormat.HF,
                        dataset_dir=DATASET if pillar is Pillar.GOLDEN else None)
        for pillar in (Pillar.GOLDEN, Pillar.SERVING, Pillar.THROUGHPUT_MULTIMODAL)
        for model in (_Model(), _Model(**_QUANT))
    ]
    over = [line for line in render_conditions(conditions).split("\n") if len(line) > 80]
    assert not over, f"lines wider than 80 columns: {over}"


def test_nothing_planned_renders_nothing():
    """An empty matrix must not print an empty banner with no rows under it."""
    assert render_conditions([]) == ""
