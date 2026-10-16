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

"""Scaffold and subject versioning for a benchmark run.

A benchmark number is only interpretable if two things held still: the
model, and the way we asked. If the prompt, the schema, the decode
settings, the scoring or the dataset moved between two runs, a score
change no longer tells you whether the model got better. It may only
tell you the question changed. So the harness gets versioned with the
same seriousness as the weights, and every result says which version
produced it.

Two axes, not one
-----------------
The fields that describe a run split into two kinds, and collapsing
them into a single id would break the most important comparison we
have.

* **Subject** is the thing under test: which checkpoint, at which
  quantization.
* **Scaffold** is how we asked: prompt, schema, scoring, tools,
  attempts, decode settings, dataset, the serving environment and the
  load shape.

The sweep is the scaffold, not a point in it
--------------------------------------------
A performance pillar does not run one measurement, it runs a sweep, and
``batch_sizes=[1, 32, 128, 256]`` is a property of the harness in
exactly the way the prompt is. So ``LOAD_SHAPE`` pins the sweep
*definition* rather than the individual point.

That distinction is what keeps the latency-versus-concurrency curve
legal. Every point on one curve shares a ``scaffold_id``, because they
came from one sweep, so plotting them as one line is correct by
construction. Today's b64 compares against last week's b64 whenever the
sweep and the environment held still. And editing ``--batch-sizes``
moves the id, which is also right, because a different sweep is a
different curve. Which point a given result file holds is already in its
name, so pinning per point would buy nothing and would forbid the plot.

Comparing quantization levels at a fixed harness is the whole point of
a compression sweep. A single combined hash would mark that comparison
illegal, because the quant is one of the inputs. Hashing the two axes
separately keeps it legal and still refuses the comparisons that are
genuinely meaningless:

===========================  ===========================================
what moved                   verdict
===========================  ===========================================
subject only                 Valid. This is the quant or model delta
scaffold only                New condition. Break the series, diff it
both                         Uninterpretable. Do not plot as one line
===========================  ===========================================

Honest coverage
---------------
Not every field is pinnable today. Rather than emit an id that
implies more rigour than exists, every block carries the ``covers`` set
it actually hashed and the ``unpinned`` remainder, and the id is
computed over exactly the covered fields. An id therefore cannot
over-promise by construction, and a results viewer can badge coverage
next to the number. ``unpinned`` is derived from :class:`ContractField`
rather than hand-maintained, because a hand-maintained list of what is
missing is a list that rots the first time someone adds a field.

A pillar we have not modelled yet emits a condition with an empty
``covers`` and a null ``scaffold_id``. That is deliberate. It says "this
ran and we pinned nothing about it", which is true, visible, and better
than silence.

Everything here is a real object with a single ``to_dict`` seam
---------------------------------------------------------------
The blocks below are dataclasses rather than free-form dicts, and each
one serialises through exactly one ``to_dict``. That is not decoration.
Three separate defects during bring-up were all the same shape: a value
that was not JSON-serialisable leaked into ``metadata.json`` and blew up
``json.dump`` at run start, before a single model had loaded. A typed
object with one serialisation seam turns that class of bug into a place
you can fix once.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple, Union

# Re-exported rather than redefined. The changelog gate reads the same
# three names from `gbench.canonical`, which is what makes it impossible
# for the gate and this contract to disagree about what a change is.
from ..analysis.perf_method import METHODOLOGY_COMPONENTS, methodology_record
from ..canonical import CHANGELOG_KEY, JSONValue, canonical_json, strip_changelog

if TYPE_CHECKING:
    from .models import ModelFormat

logger = logging.getLogger(__name__)

# Bumped when the meaning of the blocks below changes in a way that
# makes ids from an older gbench incomparable with ids from this one.
# Adding a field to `covers` does NOT need a bump, because `covers`
# already tells the reader what moved.
#
# 2: every pillar pins the inputs its runner resolves (the evals knobs,
# run-shaping environment variables, judge cascade and shared-module
# anchors; the stress load and SLOs; the effective serving counts,
# lengths and images; the server's serve arguments, engine version,
# data-parallel size and GPU models), the subject id no longer hashes
# the unused GGUF file field, and `unpinned` lists only the fields of
# its own axis. Every id moved, so v1 and v2 ids never compare.
CONTRACT_VERSION = "2"

_HASH_LEN = 8

# Matches the banner rules the rest of the CLI prints at.
_LINE_WIDTH = 80


# ----------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------

class ContractField(str, Enum):
    """The fields a fully pinned run would fix.

    Coverage is always reported against this enum so that "covers 6"
    means the same thing in every pillar and every release. Subclassing
    ``str`` keeps these JSON-serialisable and comparable to plain
    strings, so the wire format is unchanged.

    There is deliberately no not-applicable category. Exempting a pillar
    from a field it "obviously" cannot use hides a gap, and a hidden gap
    is the one failure mode this module exists to prevent. Golden looks
    like a candidate for exemption from ``LOAD_SHAPE``, but it really
    does run sequentially at an implicit concurrency of one and nothing
    records that, so reporting it unpinned is accurate rather than
    pessimistic. Under-claiming coverage costs a glance. Over-claiming
    it costs a wrong conclusion.
    """

    # Subject axis: what we are testing.
    MODEL_CHECKPOINT = "model_checkpoint"
    QUANTIZATION = "quantization"

    # Scaffold axis: how we asked.
    PROMPT_VERSION = "prompt_version"
    SCHEMA_VERSION = "schema_version"
    SCORING_VERSION = "scoring_version"
    TOOL_PERMISSIONS = "tool_permissions"
    ATTEMPTS = "attempts"
    DECODE = "decode"
    DATASET_SNAPSHOT = "dataset_snapshot"
    SERVING_ENVIRONMENT = "serving_environment"
    LOAD_SHAPE = "load_shape"


SUBJECT_FIELDS: Tuple[ContractField, ...] = (
    ContractField.MODEL_CHECKPOINT,
    ContractField.QUANTIZATION,
)

SCAFFOLD_FIELDS: Tuple[ContractField, ...] = (
    ContractField.PROMPT_VERSION,
    ContractField.SCHEMA_VERSION,
    ContractField.SCORING_VERSION,
    ContractField.TOOL_PERMISSIONS,
    ContractField.ATTEMPTS,
    ContractField.DECODE,
    ContractField.DATASET_SNAPSHOT,
    ContractField.SERVING_ENVIRONMENT,
    ContractField.LOAD_SHAPE,
)

CONTRACT_FIELDS: Tuple[ContractField, ...] = SUBJECT_FIELDS + SCAFFOLD_FIELDS

# What a Golden Set run genuinely pins today. Three of these six are
# covered by subsumption rather than by a field of their own: the case
# JSON holds the prompt (`conversation.messages`), the schema
# (`golden_truth`) and the tool declarations (top-level `tools` and
# `tool_choice`, present on 3 of the 16 cases), so a content hash over
# the dataset detects a change in any of them. The contract doc says so
# in as many words, because an id that pins something as a side effect
# should not read as though it has a dedicated mechanism.
GOLDEN_COVERS: Tuple[ContractField, ...] = (
    ContractField.DECODE,
    ContractField.ATTEMPTS,
    ContractField.DATASET_SNAPSHOT,
    ContractField.PROMPT_VERSION,
    ContractField.SCHEMA_VERSION,
    ContractField.TOOL_PERMISSIONS,
    ContractField.SCORING_VERSION,
    ContractField.SERVING_ENVIRONMENT,
)

# What each remaining pillar pins, derived from what its runner actually
# reads rather than from what it plausibly might: the config fields, the
# environment variables and the constants the runner resolves them with.
# Deriving coverage from the plausible set rather than the real one is how
# a block starts claiming a field nothing populates, and leaving out an
# input the runner reads is how two different runs come to share an id.
#
# The throughput pillars. Offline `vllm bench throughput` sends random
# token ids with no chat template, so `--thinking` never reaches it and
# there is no prompt to pin.
PERFORMANCE_COVERS: Tuple[ContractField, ...] = (
    ContractField.SERVING_ENVIRONMENT,
    ContractField.LOAD_SHAPE,
    ContractField.DATASET_SNAPSHOT,
)

# The serving pillars whose requests carry the no-think protocol (the
# system prompt and request fields `--thinking` switches off): multimodal
# serving always, text serving against a remote endpoint (a local text
# run drives `vllm bench serve` over /v1/completions, which has none).
SERVING_PROMPT_COVERS: Tuple[ContractField, ...] = PERFORMANCE_COVERS + (
    ContractField.PROMPT_VERSION,
)

# Stress pins its resolved load (workload shape, reps, rate cap, prompts
# per point, sample floor, images per request), the no-think protocol it
# sends, and what decides its max-sustainable-QPS verdict, which is its
# score: the TTFT/ITL SLOs, the keep-up fraction, the ITL gate version and
# the certification mode and budget.
STRESS_COVERS: Tuple[ContractField, ...] = (
    ContractField.SERVING_ENVIRONMENT,
    ContractField.LOAD_SHAPE,
    ContractField.DATASET_SNAPSHOT,
    ContractField.PROMPT_VERSION,
    ContractField.SCORING_VERSION,
)

QUALITY_COVERS: Tuple[ContractField, ...] = (
    ContractField.SERVING_ENVIRONMENT,
    ContractField.SCORING_VERSION,
    ContractField.DATASET_SNAPSHOT,
)

EVALS_COVERS: Tuple[ContractField, ...] = (
    ContractField.SERVING_ENVIRONMENT,
    ContractField.LOAD_SHAPE,
    ContractField.DATASET_SNAPSHOT,
    ContractField.DECODE,
    ContractField.PROMPT_VERSION,
    ContractField.SCORING_VERSION,
    ContractField.ATTEMPTS,
)


class Pillar(str, Enum):
    """The benchmark families a run can produce results for.

    These mirror the ``benchmark_type`` values stamped in ``cli.py``, so
    a condition can be lined up against the result it describes. An enum
    rather than bare strings so that any reader that matches benchmark
    types by name cannot silently mismatch "stress" and "stress_test".
    """

    SERVING = "serving"
    THROUGHPUT = "throughput"
    SERVING_MULTIMODAL = "serving_multimodal"
    THROUGHPUT_MULTIMODAL = "throughput_multimodal"
    STRESS_TEST = "stress_test"
    STRESS_TEST_MULTIMODAL = "stress_test_multimodal"
    QUALITY = "quality"
    GOLDEN = "golden"
    EVALS = "evals"


# The pillars that sweep a load rather than a scenario list. Defined
# after the enum rather than beside the ``*_COVERS`` tuples above,
# because they name members of it.
_PERFORMANCE_PILLARS = frozenset({
    Pillar.SERVING,
    Pillar.THROUGHPUT,
    Pillar.SERVING_MULTIMODAL,
    Pillar.THROUGHPUT_MULTIMODAL,
})

_MULTIMODAL_PILLARS = frozenset({
    Pillar.SERVING_MULTIMODAL,
    Pillar.THROUGHPUT_MULTIMODAL,
})

_THROUGHPUT_PILLARS = frozenset({
    Pillar.THROUGHPUT,
    Pillar.THROUGHPUT_MULTIMODAL,
})


class ModelLike(Protocol):
    """The slice of ``ModelConfig`` the contract actually reads.

    A Protocol rather than importing ``ModelConfig`` directly keeps this
    module honest about its real dependency, which is three attributes,
    and lets a test pass a stand-in without constructing a full registry
    entry.
    """

    name: str
    short_name: str
    hf_model_id: str


class ConfigLike(Protocol):
    """The slice of ``BenchmarkConfig`` the contract actually reads.

    A Protocol for the same reason :class:`ModelLike` is one, plus a
    second: ``BenchmarkConfig.__post_init__`` builds a ``LogManager`` and
    creates directories on disk, so importing and constructing it here
    would give this module a filesystem side effect it has no business
    having. Declaring the attributes we read keeps the dependency legible
    and lets a test pass a plain stand-in.

    A Protocol is structural, so nothing checks this at runtime and a
    rename in ``core/config.py`` would surface as an ``AttributeError``
    at run start rather than at import. ``test_scaffold.py`` compares the
    two annotation sets for that reason.
    """

    # Serving environment.
    remote_endpoint: Optional[str]
    num_gpus: int
    tensor_parallel_size: Optional[int]
    data_parallel_size: int
    gpu_memory_utilization: float
    max_num_seqs: Optional[int]
    enable_chunked_prefill: bool
    max_num_batched_tokens: int
    max_model_len: Optional[int]
    max_model_len_explicit: Optional[int]
    selected_gpus: Optional[List[str]]
    tool_call_parser: Optional[str]
    reasoning_parser: Optional[str]
    vllm_serve_args: List[str]

    # Load shape.
    batch_sizes: List[int]
    input_lengths: List[int]
    output_lengths: List[int]
    mm_images_per_request: Optional[int]
    serving_num_prompts_override: Optional[int]
    num_prompts_throughput: int
    request_rate: str
    num_iterations: int
    warmup_iterations: int
    workload_shape_explicit: bool
    stress_reps: Optional[int]
    stress_max_qps: Optional[float]
    stress_max_prompts: Optional[int]
    stress_min_samples: Optional[int]
    stress_ttft_slo_ms: Optional[int]
    stress_itl_slo_ms: Optional[int]
    eval_concurrency: Optional[int]
    sandboxes: Optional[int]

    # Prompt and decode.
    thinking: bool
    eval_thinking: bool
    eval_n_shot: Optional[int]
    eval_max_soft_tokens: Optional[int]
    eval_max_output_tokens: Optional[int]
    temperature: Optional[float]
    attempt_count: int

    # Datasets and scorers.
    dataset: str
    dataset_path: Optional[str]
    gemmaclaw_commit: str
    selected_scenarios: Optional[List[str]]
    evals: Optional[List[str]]
    eval_categories: Optional[str]
    eval_limit: Optional[int]
    shard: Optional[str]
    eval_custom_jsonl: Optional[str]
    eval_plugins_dir: Optional[List[str]]


def _format_value(serving_format: Union["ModelFormat", str]) -> str:
    """Normalise a serving format to its wire string.

    ``cli.py`` holds a :class:`ModelFormat` enum, which is not
    JSON-serialisable. Normalising at this one seam is what stops the
    enum reaching ``json.dump``, which would raise at run start and kill
    the run before a single model loaded.

    Anything else raises here rather than being passed through. A
    pass-through would defer the failure to ``json.dump``, several
    frames away, with a message that names the dict key and not the
    caller that supplied the bad value.

    The ``ModelFormat`` import is deferred, for the same reason the
    runners import below is. Importing it at module load builds the
    whole model registry, which contacts Hugging Face, and the changelog
    gate reads this module's canonicaliser from CI where there must be
    no network dependency and nothing to do with models.
    """
    from .models import ModelFormat

    if isinstance(serving_format, ModelFormat):
        return serving_format.value
    if isinstance(serving_format, str):
        return serving_format
    raise TypeError(
        f"serving_format must be a ModelFormat or str, got "
        f"{type(serving_format).__name__}: {serving_format!r}"
    )


# ----------------------------------------------------------------------
# Identifiers
# ----------------------------------------------------------------------

def _short_hash(prefix: str, payload: JSONValue) -> str:
    """A prefixed, truncated sha256 over canonical form.

    Eight hex characters is 32 bits, which is short enough to read out
    loud in a review and to fit in a dashboard column. Collisions are
    not a security concern here: these ids label conditions in a
    benchmark run, and nothing trusts them.
    """
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:_HASH_LEN]}"


# ----------------------------------------------------------------------
# Objects
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class AttemptPolicy:
    """How many times a case may be retried before it is called a failure."""

    max_attempts: int

    def to_dict(self) -> Dict[str, JSONValue]:
        return {"max_attempts": self.max_attempts}


# Decode settings stay a mapping rather than a dataclass on purpose. A
# case may override sampling with keys this module has never heard of,
# and a fixed shape would silently drop them, which would mean an id
# that claims to pin decode while ignoring part of it.
DecodeSettings = Mapping[str, JSONValue]


@dataclass(frozen=True)
class DatasetSnapshot:
    """A content hash of the Golden Set, plus per-case versions.

    ``case_versions`` is reported beside the id and never hashed into it.
    Case versions already move with the content hash, so folding them in
    would only make the same change count twice. Their job is to let a
    diff view say *which* case moved.
    """

    snapshot_id: Optional[str]
    case_versions: Dict[str, str] = field(default_factory=dict)

    @property
    def case_count(self) -> int:
        return len(self.case_versions)

    def to_dict(self) -> Dict[str, JSONValue]:
        return {
            "snapshot_id": self.snapshot_id,
            "case_versions": dict(self.case_versions),
            "case_count": self.case_count,
        }


def _is_local_endpoint(endpoint: Optional[str]) -> bool:
    """Whether an endpoint address is loopback.

    Deliberately the same test ``runners/golden.py:_auth_headers`` uses to
    decide whether a bearer token is needed. Two definitions of "local"
    that could drift apart would drift silently, and this one governs
    whether a URL is safe to write to disk in the clear.
    """
    ep = (endpoint or "").lower()
    return (
        not ep
        or "localhost" in ep
        or "127.0.0.1" in ep
        or "0.0.0.0" in ep
    )


def _endpoint_fingerprint(endpoint: Optional[str]) -> Optional[str]:
    """A loopback URL in the clear, anything else as a hash.

    A remote URL carries the project and deployment in its host name, and
    this repo publishes to a public org, so the raw address does not go
    into ``metadata.json``. Hashing still detects a change, which is the
    whole job of a pin. Disclosure and detection are separable and only
    the second one is needed here.

    Normalisation is deliberately minimal: whitespace and one trailing
    slash. No host aliasing, so ``localhost`` and ``127.0.0.1`` fingerprint
    differently even though they usually reach the same server. Guessing
    that two addresses are equivalent is the speculative mapping that
    turns a real difference into a silent one.

    The address is a proxy for the serving hardware and not the hardware
    itself. A Cloud Run URL survives a redeploy onto a different machine,
    so a stable fingerprint here does not prove a stable GPU. Nothing
    reachable from the client side does, which is why this is the best
    available pin rather than a sufficient one.
    """
    if endpoint is None:
        return None
    normalised = endpoint.strip().rstrip("/")
    if _is_local_endpoint(normalised):
        return normalised
    return _short_hash("ep", normalised)


@dataclass(frozen=True)
class ServingEnvironment:
    """Where the model was served from, and on what.

    The vLLM flags are populated only when gbench started the server. On
    a run against an endpoint somebody else is hosting, ``cli.py`` skips
    the GPU count check entirely and none of these flags reach any
    process we own, so recording the configured values would pin numbers
    that had no effect. Two such runs differing only in an unused
    ``tensor_parallel_size`` would then get different ids for no reason,
    which is a false break: the same defect as hashing a source file that
    only had a docstring edited. ``None`` here means "did not apply",
    which is the true statement.

    The port the server listens on and the id it serves the model under
    are deliberately not here. The port changes nothing about what is
    measured, so pinning it would break a series every time the automatic
    port search landed elsewhere, and the served id names the subject,
    which ``metadata.json`` records as ``served_model_id``.
    """

    gbench_started_server: bool
    endpoint: Optional[str]
    num_gpus: Optional[int] = None
    tensor_parallel_size: Optional[int] = None
    #: --num-gpus / --tensor-parallel replicas; passed as --data-parallel-size when above 1.
    data_parallel_size: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    max_num_seqs: Optional[int] = None
    enable_chunked_prefill: Optional[bool] = None
    max_num_batched_tokens: Optional[int] = None
    #: Served context window: the --max-model-len passed to vLLM, or "native" when the
    #: server was left at the model's own window (eval and quality servers).
    max_model_len: Union[int, str, None] = None
    #: The arguments appended to this server's command after gbench's own, as the runner
    #: appends them: --mm-processor-kwargs with --eval-max-soft-tokens on the eval, quality and
    #: Golden Set server, then config.serve_args_for (the tool-call and reasoning parsers on that
    #: server, --vllm-serve-args on every server).
    serve_args: Optional[Tuple[str, ...]] = None
    #: The engine a server gbench launches runs (utils.dependency_checks.vllm_engine_record):
    #: {"vllm_version": the vllm package version of gbench's interpreter, "cli_override": the
    #: GBENCH_VLLM_BIN command when set, else None}. The interpreter's own path is left out: it
    #: names a host, not an engine.
    engine: Optional[Dict[str, JSONValue]] = None
    #: The product names of the GPUs the server was given (config.selected_gpus), sorted;
    #: None when no selection was made or nvidia-smi could not name them. The UUIDs are in
    #: metadata.json; they name physical cards, so pinning them would break a series whenever
    #: the same model ran on another card of the same kind.
    gpu_models: Optional[Tuple[str, ...]] = None

    def to_dict(self) -> Dict[str, JSONValue]:
        return {
            "gbench_started_server": self.gbench_started_server,
            "endpoint": self.endpoint,
            "num_gpus": self.num_gpus,
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_num_seqs": self.max_num_seqs,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_model_len": self.max_model_len,
            "serve_args": list(self.serve_args) if self.serve_args is not None else None,
            "engine": dict(self.engine) if self.engine is not None else None,
            "gpu_models": list(self.gpu_models) if self.gpu_models is not None else None,
        }


@dataclass(frozen=True)
class LoadShape:
    """The sweep a pillar ran, not one point inside it.

    ``batch_sizes`` is a list because that is what the harness was told
    to sweep, and the sweep is the property worth pinning. Pinning a
    single point would give every point on one curve a different id and
    so declare the curve unplottable, while pinning the sweep makes the
    curve legal by construction and still breaks the series when someone
    changes what is swept.

    Fields a pillar does not use stay ``None`` rather than being filled
    with a default, and each holds the value the runner actually uses
    rather than the flag it was derived from: throughput sweeps every
    input and output length, serving and stress use the first of each,
    so their scaffolds pin that one and a throughput-only length moves
    neither of them.
    """

    batch_sizes: Optional[Tuple[int, ...]] = None
    input_lengths: Optional[Tuple[int, ...]] = None
    output_lengths: Optional[Tuple[int, ...]] = None
    images_per_request: Optional[Tuple[int, ...]] = None
    #: Throughput's prompt count (config.num_prompts_throughput).
    num_prompts: Optional[int] = None
    #: Serving's effective request count per swept batch size (auto-sized from the batch size
    #: and the output length, or the --personal override), in batch_sizes order.
    num_prompts_per_batch: Optional[Tuple[int, ...]] = None
    request_rate: Optional[str] = None
    num_iterations: Optional[int] = None
    warmup_iterations: Optional[int] = None
    concurrency: Optional[int] = None
    #: The --sandboxes value handed to every eval suite (a plugin may honor it too).
    sandboxes: Optional[int] = None
    #: How many containers a selected containerized suite (eval_suites.SANDBOXED_SUITES) runs at
    #: once: --sandboxes, else --batch-sizes[0]; None when no such suite is selected.
    container_concurrency: Optional[int] = None
    #: Stress: full sweeps per point, the open-loop rate cap, requests per rate point and the
    #: steady-state sample floor, each resolved flag > GBENCH_STRESS_* > runner default.
    reps: Optional[int] = None
    max_qps: Optional[float] = None
    max_prompts: Optional[int] = None
    min_samples: Optional[int] = None

    def to_dict(self) -> Dict[str, JSONValue]:
        return {
            "batch_sizes": list(self.batch_sizes) if self.batch_sizes else None,
            "input_lengths": list(self.input_lengths) if self.input_lengths else None,
            "output_lengths": list(self.output_lengths) if self.output_lengths else None,
            "images_per_request": (
                list(self.images_per_request) if self.images_per_request else None
            ),
            "num_prompts": self.num_prompts,
            "num_prompts_per_batch": (
                list(self.num_prompts_per_batch) if self.num_prompts_per_batch else None
            ),
            "request_rate": self.request_rate,
            "num_iterations": self.num_iterations,
            "warmup_iterations": self.warmup_iterations,
            "concurrency": self.concurrency,
            "sandboxes": self.sandboxes,
            "container_concurrency": self.container_concurrency,
            "reps": self.reps,
            "max_qps": self.max_qps,
            "max_prompts": self.max_prompts,
            "min_samples": self.min_samples,
        }


@dataclass(frozen=True)
class Subject:
    """What was tested. Identity, deliberately not a pin.

    ``subject_id`` hashes the identity we can actually observe: the model
    id and the serving format. Two checkpoints, or the same checkpoint
    served in another format, get different ids, which is the distinction
    a sweep turns on.

    Nothing here records an HF revision, so if the upstream repo is
    updated in place the id will not move. Both ``model_checkpoint`` and
    ``quantization`` are therefore reported unpinned until the server
    gbench launches is given a resolved checkpoint revision and a
    structured quantization level. The empty ``covers`` is the
    machine-readable form of that admission, and ``unpinned`` names only
    the subject fields: the scaffold fields are the scaffold's to report.
    """

    subject_id: str
    name: str
    short_name: str
    model: str
    serving_format: str

    @property
    def covers(self) -> Tuple[ContractField, ...]:
        return ()

    @property
    def unpinned(self) -> Tuple[ContractField, ...]:
        return tuple(f for f in SUBJECT_FIELDS if f not in self.covers)

    def to_dict(self) -> Dict[str, JSONValue]:
        return {
            "subject_id": self.subject_id,
            "name": self.name,
            "short_name": self.short_name,
            "model": self.model,
            "serving_format": self.serving_format,
            "covers": [f.value for f in self.covers],
            "unpinned": [f.value for f in self.unpinned],
        }


@dataclass(frozen=True)
class Scaffold:
    """How we asked, and an honest statement of how much of that is pinned.

    ``pinned`` holds exactly the fields named in ``covers`` and is the
    sole input to ``scaffold_id``. Keeping those two in lockstep is what
    makes it structurally impossible for an id to over-promise.

    The gbench commit is recorded beside the id and never inside it. If
    it were an input, every commit to this repo would invalidate every
    scaffold, the id would churn on changes that touch nothing relevant,
    and it would stop carrying information.
    """

    pillar: Pillar
    covers: Tuple[ContractField, ...]
    pinned: Dict[str, JSONValue]
    dataset: Optional[DatasetSnapshot] = None
    contract_version: str = CONTRACT_VERSION

    @property
    def scaffold_id(self) -> Optional[str]:
        """None when nothing is pinned, so an unmodelled pillar cannot pose as one."""
        return _short_hash("sc", self.pinned) if self.covers else None

    @property
    def unpinned(self) -> Tuple[ContractField, ...]:
        """The scaffold fields not covered. Derived, never hand-maintained, so it cannot rot.

        The subject fields are the subject's to report (:class:`Subject`); listing them here as
        well would count against every scaffold two fields no scaffold may pin."""
        return tuple(f for f in SCAFFOLD_FIELDS if f not in self.covers)

    def to_dict(self) -> Dict[str, JSONValue]:
        out: Dict[str, JSONValue] = {
            "scaffold_id": self.scaffold_id,
            "contract_version": self.contract_version,
            "pillar": self.pillar.value,
            "covers": [f.value for f in self.covers],
            "unpinned": [f.value for f in self.unpinned],
            "pinned": dict(self.pinned),
        }
        if self.dataset is not None:
            out["case_versions"] = dict(self.dataset.case_versions)
            out["case_count"] = self.dataset.case_count
        return out


@dataclass(frozen=True)
class Condition:
    """One ``(pillar, model, format)`` cell of the run's experimental matrix.

    A run is multi-model and multi-pillar, so a single run-level id
    cannot describe it. Conditions are a list for that reason: two models
    across two pillars is four conditions and one ``metadata.json``.
    """

    pillar: Pillar
    model: str
    serving_format: str
    scaffold: Scaffold
    subject: Subject

    def to_dict(self) -> Dict[str, JSONValue]:
        return {
            "pillar": self.pillar.value,
            "model": self.model,
            "format": self.serving_format,
            "scaffold": self.scaffold.to_dict(),
            "subject": self.subject.to_dict(),
        }


# ----------------------------------------------------------------------
# Dataset snapshot
# ----------------------------------------------------------------------

def dataset_snapshot(
    dataset_dir: Union[Path, str],
    selected_tasks: Optional[Sequence[str]] = None,
) -> DatasetSnapshot:
    """Hash the Golden Set on disk and report each case's version.

    Args:
        dataset_dir: Directory holding the case JSON files.
        selected_tasks: Optional subset filter, matching ``load_tasks``
            semantics (task id or file name).

    Returns:
        A :class:`DatasetSnapshot`.

    Binary assets are hashed by content, and all of them are hashed even
    for a subset run. Attributing an image to the cases that reference it
    would mean parsing every message, and over-detecting here is the safe
    direction: a spurious break costs a glance at a diff, a missed one
    costs a wrong conclusion.
    """
    entries: List[Tuple[str, str]] = []
    case_versions: Dict[str, str] = {}

    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        logger.error(
            f"Golden dataset directory {dataset_dir} does not exist, so no "
            "dataset snapshot can be taken."
        )
        return DatasetSnapshot(snapshot_id=None)

    for path in sorted(dataset_dir.glob("*.json")):
        raw = path.read_bytes()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as err:
            # A case that will not parse still changes the dataset, so
            # hash its bytes rather than dropping it silently.
            logger.warning(f"Golden case {path.name} did not parse: {err}")
            entries.append((path.name, hashlib.sha256(raw).hexdigest()))
            case_versions[path.stem] = "unparseable"
            continue

        task_id = data.get("id", path.stem)
        if selected_tasks and task_id not in selected_tasks and path.name not in selected_tasks:
            continue

        payload = canonical_json(strip_changelog(data)).encode("utf-8")
        entries.append((path.name, hashlib.sha256(payload).hexdigest()))
        case_versions[task_id] = data.get("meta", {}).get("version", "unversioned")

    assets_dir = dataset_dir / "assets"
    if assets_dir.exists():
        for path in sorted(assets_dir.rglob("*")):
            if path.is_file():
                rel = path.relative_to(dataset_dir).as_posix()
                entries.append((rel, hashlib.sha256(path.read_bytes()).hexdigest()))

    return DatasetSnapshot(
        snapshot_id=_short_hash("gd", [list(e) for e in entries]),
        case_versions=case_versions,
    )


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------

def gbench_commit() -> Dict[str, JSONValue]:
    """Which revision of this harness produced a run.

    Recorded beside the ids and never inside them, for the reason given
    on :class:`Scaffold`: as a hash input it would invalidate every
    scaffold on every commit and stop carrying information.

    ``dirty`` is not decoration. A sha alone claims the run came from a
    published tree, and most gbench runs happen on a workstation with
    uncommitted edits, so a sha without that flag is the more confident
    of the two possible lies. It is also what makes the sha safe to trust
    when it is absent.

    Returns ``{"commit": None, "dirty": None}`` outside a git checkout,
    such as from an installed wheel or inside a container built by COPY.
    That is a real deployment shape rather than an error, so it is
    reported rather than raised.
    """
    unknown: Dict[str, JSONValue] = {"commit": None, "dirty": None}
    repo = Path(__file__).resolve().parent

    def _git(*argv: str) -> Optional[str]:
        try:
            done = subprocess.run(
                ["git", *argv],
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as err:
            # No git binary, or a hung index lock. Neither is worth
            # failing a benchmark run over.
            logger.debug(f"Could not read gbench provenance via git {argv}: {err}")
            return None
        if done.returncode != 0:
            return None
        return done.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    if not commit:
        return unknown

    # `--porcelain` is empty exactly when the tree is clean. An error is
    # reported as unknown rather than clean, so the optimistic reading is
    # never the default.
    status = _git("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


# Duplicated nowhere. The quality runner clones from this same constant,
# because a sha resolved against one remote and a checkout performed
# against another would pin a scorer the run never used.
GEMMACLAW_REMOTE = "https://github.com/gemmaclaw/gemmaclaw.git"


def _is_full_sha(ref: str) -> bool:
    """Whether a ref is already a pin rather than a pointer."""
    return len(ref) == 40 and all(c in "0123456789abcdef" for c in ref)


def resolve_gemmaclaw_sha(ref: str, repo_path: Optional[str] = None) -> Optional[str]:
    """Resolve a GemmaClaw ref to the commit it names right now.

    A branch name is not a pin. ``main`` names a different commit next
    week, so a ``scaffold_id`` hashed over the literal string ``"main"``
    holds still while the scorer moves underneath it. That is a false
    match, which this module treats as the worse of the two failures: a
    false break costs a glance at a diff, a false match puts two
    incomparable runs on one line and gives you no reason to doubt it.

    Called once per run from ``cli.py``, which writes the result back
    onto the config before any scaffold is built or any checkout
    happens. Resolving inside the builder instead would reintroduce the
    race it removes, because the condition loop builds one scaffold per
    model and format and ``main`` can advance between two of them.

    Args:
        ref: A branch, tag or commit-ish, as passed to
            ``--gemmaclaw-commit``.
        repo_path: A local checkout, as passed to ``--gemmaclaw-path``.
            When set it is the authority and the remote is not consulted,
            because the runner checks out in place and never fetches in
            that mode, so the remote's tip is not what the run will use.

    Returns:
        The full 40-character sha, or ``None`` when the ref cannot be
        resolved. ``None`` leaves the caller holding the ref, which is
        honest about the gap rather than guessing at a sha. It also
        predicts a failed run: a ref this cannot resolve is a ref the
        runner cannot check out either.
    """
    if _is_full_sha(ref):
        return ref

    # Both patterns, because `ls-remote` filters its output by the
    # patterns given and an exact `v1` therefore hides the `v1^{}` line
    # that carries the commit an annotated tag points at. Asking for the
    # peeled form too costs nothing for a branch, which has none.
    argv = (
        ["git", "-C", repo_path, "rev-parse", f"{ref}^{{commit}}"]
        if repo_path
        else ["git", "ls-remote", GEMMACLAW_REMOTE, ref, f"{ref}^{{}}"]
    )
    try:
        # The operator's git configuration is not read: a personal `insteadOf` rewrite of
        # https://github.com to SSH made this public ls-remote fail (2026-10-09).
        from ..runners.eval_suites.swebench_common import isolated_git_env
        done = subprocess.run(
            argv, capture_output=True, text=True, timeout=30, check=False,
            env=isolated_git_env(),
        )
    except (OSError, subprocess.SubprocessError) as err:
        # No git binary, or a network that hung past the timeout.
        logger.warning(f"Could not resolve GemmaClaw ref '{ref}': {err}")
        return None
    if done.returncode != 0 or not done.stdout.strip():
        # `ls-remote` exits 0 with no output for a ref that does not
        # exist, so an empty stdout is a miss rather than a success.
        logger.warning(f"Could not resolve GemmaClaw ref '{ref}' to a commit.")
        return None

    if repo_path:
        sha = done.stdout.strip()
        return sha if _is_full_sha(sha) else None

    # `ls-remote` prints one `<sha>\t<ref>` line per match. An annotated
    # tag matches twice, and the `^{}` line carries the commit the tag
    # points at, which is where checking that tag out actually lands.
    resolved: Optional[str] = None
    for line in done.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2 or not _is_full_sha(parts[0]):
            continue
        sha, name = parts
        if name.endswith("^{}"):
            return sha
        if resolved is None:
            resolved = sha
    return resolved


# ----------------------------------------------------------------------
# Builders
# ----------------------------------------------------------------------

def build_subject(model: ModelLike, serving_format: Union[ModelFormat, str]) -> Subject:
    """Describe what was tested, and be explicit about what is not pinned."""
    fmt = _format_value(serving_format)
    identity: Dict[str, JSONValue] = {
        "model": model.hf_model_id,
        "serving_format": fmt,
    }
    return Subject(
        subject_id=_short_hash("sb", identity),
        name=model.name,
        short_name=model.short_name,
        model=model.hf_model_id,
        serving_format=fmt,
    )


def build_serving_environment(config: ConfigLike, context: str = "perf") -> ServingEnvironment:
    """Describe where the model was served from, and on what.

    The vLLM flags are read only when ``remote_endpoint`` is unset, which
    is precisely when gbench starts the server itself and those flags
    reach a process we own. Against somebody else's endpoint they are
    left ``None``, because ``cli.py`` skips the GPU check on that path and
    the configured values never take effect. See
    :class:`ServingEnvironment` for why pinning an unused value is worse
    than pinning nothing.

    ``context`` is the server's: ``"perf"`` for the serving, throughput and
    stress servers, ``"native"`` for the one server the quality, Golden Set
    and eval pillars share. The two serve different windows and receive
    different operator arguments, and each block records its own.
    """
    endpoint = getattr(config, "remote_endpoint", None)

    if endpoint is not None:
        return ServingEnvironment(
            gbench_started_server=False,
            endpoint=_endpoint_fingerprint(endpoint),
        )

    # Deferred for the same reason the other core imports here are:
    # `.config` builds a LogManager on construction, and this module is
    # read by the changelog gate in CI where nothing may touch disk.
    from .config import get_max_model_len, serve_args_for

    return ServingEnvironment(
        gbench_started_server=True,
        endpoint=None,
        num_gpus=config.num_gpus,
        tensor_parallel_size=config.tensor_parallel_size,
        data_parallel_size=int(getattr(config, "data_parallel_size", None) or 1),
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_num_seqs=config.max_num_seqs,
        enable_chunked_prefill=config.enable_chunked_prefill,
        max_num_batched_tokens=config.max_num_batched_tokens,
        max_model_len=_served_context(config, context, get_max_model_len),
        serve_args=_applied_serve_args(config, context, serve_args_for),
        engine=_engine_record(),
        gpu_models=_gpu_models(tuple(getattr(config, "selected_gpus", None) or ())),
    )


def _applied_serve_args(config: ConfigLike, context: str, serve_args_for) -> Tuple[str, ...]:
    """The arguments ServingBenchmarkRunner.build_server_cmd appends after its own for this
    server, in its order: on the native server, ``--mm-processor-kwargs`` carrying
    ``--eval-max-soft-tokens`` when it is set, then the operator's serve settings
    (``config.serve_args_for``: the parsers on the native server, ``--vllm-serve-args`` on
    every server)."""
    out: List[str] = []
    soft = getattr(config, "eval_max_soft_tokens", None)
    if context == "native" and soft is not None:
        out += ["--mm-processor-kwargs", json.dumps({"max_soft_tokens": int(soft)})]
    out += list(serve_args_for(config, context))
    return tuple(out)


def _served_context(config: ConfigLike, context: str, get_max_model_len) -> Union[int, str]:
    """The window gbench's server actually served, mirroring ServingBenchmarkRunner._start_server:
    perf servers get the uniform (or campaign/explicit) window; eval and quality servers get
    the user's explicit --max-model-len, else the model's native window."""
    if context == "native":
        return getattr(config, "max_model_len_explicit", None) or "native"
    return get_max_model_len(getattr(config, "max_model_len", None))


def _engine_record() -> Dict[str, JSONValue]:
    """The engine a server gbench launches runs, from the record the runners write
    (``vllm_engine_record``: package metadata, vLLM is not imported): the vllm version of
    gbench's interpreter and the ``GBENCH_VLLM_BIN`` command when it replaces that engine."""
    from ..utils.dependency_checks import VLLM_BIN_ENV, vllm_engine_record

    record = vllm_engine_record()
    return {
        "vllm_version": record.get("package_version"),
        "cli_override": (list(record.get("cli") or [])
                         if record.get("cli_source") == VLLM_BIN_ENV else None),
    }


@lru_cache(maxsize=None)
def _gpu_models(uuids: Tuple[str, ...]) -> Optional[Tuple[str, ...]]:
    """The product names of the GPUs with these UUIDs, sorted, as nvidia-smi reports them.

    None for no UUIDs, or when nvidia-smi cannot be run or does not list every one of them:
    a partial answer would pin a smaller machine than the one that ran. Cached, because the
    condition loop builds one scaffold per pillar, model and format.
    """
    if not uuids:
        return None
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError) as err:
        logger.debug(f"Could not read the GPU names via nvidia-smi: {err}")
        return None
    if done.returncode != 0:
        return None
    names: Dict[str, str] = {}
    for line in done.stdout.splitlines():
        uuid, _, name = line.partition(",")
        if uuid.strip() and name.strip():
            names[uuid.strip()] = name.strip()
    if not all(u in names for u in uuids):
        return None
    return tuple(sorted(names[u] for u in uuids))


def _without_environment(
    covers: Tuple[ContractField, ...],
) -> Tuple[ContractField, ...]:
    """Drop the environment field from a covers tuple.

    Used when no config was supplied, which happens in tests and in any
    caller that only wants the dataset half of the contract. ``covers``
    and ``pinned`` have to stay in lockstep, so a block that cannot
    populate the environment must not claim it.
    """
    return tuple(f for f in covers if f is not ContractField.SERVING_ENVIRONMENT)


def build_golden_scaffold(
    dataset_dir: Union[Path, str],
    selected_tasks: Optional[Sequence[str]] = None,
    config: Optional[ConfigLike] = None,
) -> Scaffold:
    """Build the scaffold block for a Golden Set condition.

    ``LOAD_SHAPE`` stays unpinned here and that is not an oversight.
    Golden issues its cases one at a time, so it does have a concurrency,
    it is one, and nothing in the harness records it as a decision. When
    that becomes explicit it should move into ``covers``.

    The environment is the native server's: the CLI starts one server per
    model for the quality, Golden Set and eval pillars, at the model's own
    window and with the operator's parsers, and golden asks that server.
    """
    # Deferred so that core does not import runners at module load.
    # cli.py imports the runners the same way, for the same reason.
    from ..runners.golden import DEFAULT_SAMPLING, MAX_ATTEMPTS, SCORING_VERSION

    snapshot = dataset_snapshot(dataset_dir, selected_tasks)

    pinned: Dict[str, JSONValue] = {
        ContractField.DECODE.value: dict(DEFAULT_SAMPLING),
        ContractField.ATTEMPTS.value: AttemptPolicy(MAX_ATTEMPTS).to_dict(),
        ContractField.DATASET_SNAPSHOT.value: snapshot.snapshot_id,
        ContractField.SCORING_VERSION.value: SCORING_VERSION,
        "selected_tasks": sorted(selected_tasks) if selected_tasks else None,
    }

    covers = GOLDEN_COVERS
    if config is None:
        covers = _without_environment(covers)
    else:
        pinned[ContractField.SERVING_ENVIRONMENT.value] = (
            build_serving_environment(config, context="native").to_dict()
        )

    return Scaffold(
        pillar=Pillar.GOLDEN,
        covers=covers,
        pinned=pinned,
        dataset=snapshot,
    )


def _first(values: Optional[Sequence[int]]) -> Optional[Tuple[int, ...]]:
    """The first value of a length list as a one-element tuple (what serving and stress use)."""
    return (int(values[0]),) if values else None


def _perf_prompt_version(config: ConfigLike) -> Dict[str, JSONValue]:
    """The no-think protocol the serving and stress requests carry, and whether it was on.

    ``thinking`` is the switch (``--thinking``); the system prompt is pinned by its sha256 and
    the request fields by value, as ``ServingBenchmarkRunner`` defines them, so an edit to
    either moves the id even when the switch did not.
    """
    from types import SimpleNamespace

    from ..runners.serving import ServingBenchmarkRunner

    fields = ServingBenchmarkRunner.no_think_fields(
        SimpleNamespace(config=SimpleNamespace(thinking=False)))
    return {
        "thinking": bool(getattr(config, "thinking", False)),
        "no_think_system_sha256": hashlib.sha256(
            ServingBenchmarkRunner.NOTHINK_SYSTEM.encode("utf-8")).hexdigest(),
        "no_think_request_fields": json.loads(json.dumps(fields)),
    }


def _mm_images_per_request(config: ConfigLike, runner_default: int) -> int:
    """Images per multimodal request as the runners resolve it: the config override
    (``--personal`` sets 1), else the runner's default."""
    return int(getattr(config, "mm_images_per_request", None) or runner_default)


def build_performance_scaffold(
    pillar: Pillar,
    config: ConfigLike,
) -> Scaffold:
    """Build the scaffold for one of the four load-sweeping pillars.

    ``DECODE`` is left unpinned, and the reason is worth stating because
    it looks like an omission. The performance runners set ``max_tokens``
    and ``ignore_eos`` and nothing else, so temperature and top-p are
    whatever the serving stack defaults to. Those values are genuinely
    unknown to gbench rather than merely unrecorded, which means a vLLM
    default change would move these numbers with no signal anywhere. That
    is a real gap and it should read as one.

    Each load field holds what the runner uses. Throughput sweeps every
    input and output length at ``num_prompts_throughput`` prompts, for
    ``num_iterations`` runs, on random tokens (random-mm for the multimodal
    pillar): ``vllm bench throughput`` reads no batch size or request rate
    and each run is a fresh engine with no warmup, so batch sizes, request
    rate, warmup iterations and ``--dataset`` are not pinned for it.
    Serving uses the first length of each and sizes its request count per
    batch size (``BenchmarkConfig._serving_num_prompts``: from the batch
    size and the output length, or the ``--personal`` override), so it
    pins those counts rather than ``--num-prompts``, which it never reads.
    The multimodal pillars pin the images per request they send and name
    their source as generated: they build their own images and filler text
    rather than read a dataset.
    """
    from .config import BenchmarkConfig

    is_multimodal = pillar in _MULTIMODAL_PILLARS
    is_throughput = pillar in _THROUGHPUT_PILLARS

    images = None
    if is_multimodal:
        from ..runners.serving import ServingBenchmarkRunner
        images = (_mm_images_per_request(config, ServingBenchmarkRunner.IMAGES_PER_REQUEST),)

    if is_throughput:
        load = LoadShape(
            input_lengths=tuple(config.input_lengths),
            output_lengths=tuple(config.output_lengths),
            images_per_request=images,
            num_prompts=config.num_prompts_throughput,
            num_iterations=config.num_iterations,
        )
    else:
        load = LoadShape(
            batch_sizes=tuple(config.batch_sizes),
            input_lengths=_first(config.input_lengths),
            output_lengths=_first(config.output_lengths),
            images_per_request=images,
            num_prompts_per_batch=tuple(
                BenchmarkConfig._serving_num_prompts(config, b)[0] for b in config.batch_sizes),
            request_rate=config.request_rate,
            num_iterations=config.num_iterations,
            warmup_iterations=config.warmup_iterations,
        )

    # Names the source rather than hashing its content, unlike the golden
    # snapshot. "random" has no content to hash, and for a file-backed set
    # a mutated file at the same path will not move this id. Recorded as a
    # pin because which source was used is the change that actually
    # happens, but it is the weaker of the two mechanisms and should not be
    # read as the stronger one.
    if is_multimodal:
        dataset: Dict[str, JSONValue] = {"dataset": "generated", "content_hashed": False}
    elif is_throughput:
        # the offline benchmark always sends random tokens, whatever --dataset says
        dataset = {"dataset": "random", "content_hashed": False}
    else:
        dataset = {"dataset": config.dataset,
                   "dataset_path": getattr(config, "dataset_path", None),
                   "content_hashed": False}

    if not is_throughput:
        # The measurement method (perf_method): a change there changes the numbers,
        # so it changes the id. Offline throughput uses none of it.
        dataset = {**dataset, "workload_version": methodology_record("serving")["methodology_version"]}
    pinned: Dict[str, JSONValue] = {
        ContractField.SERVING_ENVIRONMENT.value: build_serving_environment(config).to_dict(),
        ContractField.LOAD_SHAPE.value: load.to_dict(),
        ContractField.DATASET_SNAPSHOT.value: dataset,
    }
    covers = PERFORMANCE_COVERS
    if pillar is Pillar.SERVING_MULTIMODAL or (
            pillar is Pillar.SERVING and getattr(config, "remote_endpoint", None)):
        covers = SERVING_PROMPT_COVERS
        pinned[ContractField.PROMPT_VERSION.value] = _perf_prompt_version(config)

    return Scaffold(pillar=pillar, covers=covers, pinned=pinned)


def _evals_attempts(config: ConfigLike, suites: Sequence[str]) -> Dict[str, JSONValue]:
    """The evals attempts pin: --attempt-count, and for a selected suite that always runs more
    attempts per sample (matharena_common.SUITE_MIN_ATTEMPTS: MathArena's runs per problem), the
    count it runs."""
    k = int(getattr(config, "attempt_count", 1) or 1)
    out: Dict[str, JSONValue] = {"attempts_per_sample": k}
    try:
        from ..runners.eval_suites.matharena_common import SUITE_MIN_ATTEMPTS
    except Exception:                                                     # noqa: BLE001
        return out
    per_suite = {s: max(k, n) for s, n in sorted(SUITE_MIN_ATTEMPTS.items())
                 if s in (suites or ()) and n > k}
    if per_suite:
        out["suite_attempts_per_sample"] = per_suite
    return out


def build_stress_scaffold(
    config: ConfigLike, pillar: Pillar = Pillar.STRESS_TEST
) -> Scaffold:
    """Build the scaffold for a stress pillar (text or multimodal).

    The load is read off a ``StressTestRunner`` built from the same config
    and SLOs the CLI gives the real one, so every knob is pinned as the
    runner resolves it (flag, else ``GBENCH_STRESS_*``, else the runner's
    default) rather than as a raw flag that may not have applied. The
    workload is the pinned geometry (``--campaign`` or explicit lengths,
    on the configured dataset) or the runner's neutral random 128/128.
    The TTFT and ITL SLOs are the scoring version: they decide which rate
    counts as sustainable, which is the stress verdict. The number of
    client processes is left out: it follows the host's core count, so
    pinning it would break a series between hosts for a load generator
    setting rather than a workload one.
    """
    from ..runners.stress import StressTestRunner, certify_settings

    multimodal = pillar is Pillar.STRESS_TEST_MULTIMODAL
    _certify = certify_settings(config)
    runner = StressTestRunner(
        config,
        ttft_threshold_ms=getattr(config, "stress_ttft_slo_ms", None),
        tpot_threshold_ms=getattr(config, "stress_itl_slo_ms", None),
    )
    dataset_name, in_len, out_len = runner._workload_shape()
    load = LoadShape(
        input_lengths=(int(in_len),),
        output_lengths=(int(out_len),),
        images_per_request=(int(runner._images_per_request()),) if multimodal else None,
        reps=int(runner._reps),
        max_qps=float(runner.max_qps),
        # None = no cap (the default: points get the requests the method needs); an
        # explicit --stress-max-prompts / GBENCH_STRESS_MAX_PROMPTS is a recorded cap.
        max_prompts=(int(runner.max_prompts) if getattr(runner, "_max_prompts_explicit", False)
                     else None),
        min_samples=int(runner._min_steady),
    )
    if multimodal:
        dataset: Dict[str, JSONValue] = {"dataset": "generated", "content_hashed": False}
    else:
        dataset = {
            "dataset": dataset_name,
            "dataset_path": (getattr(config, "dataset_path", None)
                             if getattr(config, "workload_shape_explicit", False) else None),
            "content_hashed": False,
        }
    dataset = {**dataset, "workload_version": methodology_record("stress")["methodology_version"]}
    return Scaffold(
        pillar=pillar,
        covers=STRESS_COVERS,
        pinned={
            ContractField.SERVING_ENVIRONMENT.value:
                build_serving_environment(config).to_dict(),
            ContractField.LOAD_SHAPE.value: load.to_dict(),
            ContractField.DATASET_SNAPSHOT.value: dataset,
            ContractField.PROMPT_VERSION.value: _perf_prompt_version(config),
            ContractField.SCORING_VERSION.value: {
                "ttft_slo_ms": int(runner.ttft_threshold_ms),
                "itl_slo_ms": int(runner.tpot_threshold_ms),
                # The gate decides the verdict as much as the SLOs do.
                "keep_up_frac": float(runner.KEEP_UP_FRAC),
                "itl_gate": f"gate{METHODOLOGY_COMPONENTS['gate']}",
                # Certification decides whether (and which) limit is reported.
                "certify": _certify[0],
                "certify_budget_min": _certify[1],
            },
        },
    )


def build_quality_scaffold(config: ConfigLike) -> Scaffold:
    """Build the scaffold for the GemmaClaw quality pillar.

    ``gemmaclaw_commit`` is the scoring version here, because GemmaClaw
    is the scorer. By the time this runs it holds a resolved 40-character
    sha rather than the ref that was typed. The default is already a sha,
    ``DEFAULT_GEMMACLAW_COMMIT`` in ``core/config.py``, so an unflagged
    run pins a released scorer and this id is stable across months. For
    anything else ``cli.py`` calls :func:`resolve_gemmaclaw_sha` once,
    before any scaffold is built, and writes the sha back onto the config.

    Two things follow, and both are the point. Advancing the scorer, by
    promoting the pinned default or by passing ``--gemmaclaw-commit
    main``, breaks the series instead of silently extending it. And the
    sha hashed here is the same one the runner checks out, because both
    read the one field that was resolved once, rather than each asking
    the remote at a different moment.

    A ref that fails to resolve is left as-is, so the block pins the
    literal string. That is honest but it is not a pin, and it is the one
    case where two runs a week apart can still share an id while the
    scorer moved. It also predicts a failed run, because a ref this
    cannot resolve is a ref the runner cannot check out either.
    """
    scenarios = config.selected_scenarios
    return Scaffold(
        pillar=Pillar.QUALITY,
        covers=QUALITY_COVERS,
        pinned={
            ContractField.SERVING_ENVIRONMENT.value:
                build_serving_environment(config, context="native").to_dict(),
            ContractField.SCORING_VERSION.value: {
                "gemmaclaw_commit": config.gemmaclaw_commit,
            },
            ContractField.DATASET_SNAPSHOT.value: {
                "selected_scenarios": sorted(scenarios) if scenarios else None,
            },
        },
    )


#: The environment variables that choose the Gemini judge and search cascades. The evals
#: scaffold pins the cascades they resolve to (judge_cascade(), search_cascade()), so their raw
#: values are left out of its run-shaping snapshot: setting GBENCH_JUDGE_MODELS to the default
#: list is the same grading as leaving it unset.
_CASCADE_ENV = frozenset({
    "GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL", "GBENCH_SEARCH_MODELS", "GBENCH_SEARCH_MODEL",
})


def resolved_eval_suites(config: ConfigLike) -> List[str]:
    """The suites an evals run executes: ``runners.evals.resolve_eval_names``, the list the
    evals runner itself runs (comma lists split, renamed suites mapped to their names, ``all``
    and ``plugins`` expanded with the discovered plugins, ``custom_jsonl`` added when
    ``--eval-custom-jsonl`` is set)."""
    from ..runners.evals import resolve_eval_names

    return list(resolve_eval_names(config))


def _container_concurrency(config: ConfigLike, suites: Sequence[str]) -> Optional[int]:
    """The containers a selected containerized suite runs at once, as the evals runner resolves
    it: --sandboxes, else --batch-sizes[0], never below 1; None when none is selected."""
    from ..runners.eval_suites import SANDBOXED_SUITES

    if not any(name in SANDBOXED_SUITES for name in suites):
        return None
    sandboxes = getattr(config, "sandboxes", None)
    if sandboxes is None:
        sizes = getattr(config, "batch_sizes", None) or []
        sandboxes = sizes[0] if sizes else 1
    return max(1, int(sandboxes))


def _file_sha256(path: Optional[str]) -> Optional[str]:
    """sha256 of a file's bytes, or None when there is no path or it cannot be read."""
    if not path:
        return None
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError as err:
        logger.warning(f"Could not hash {path} for the evals scaffold: {err}")
        return None


def _plugin_version(name: str) -> Optional[str]:
    """sha256 of the module file a discovered plugin suite was loaded from (read-only), or None
    when it is not a registered plugin with a readable file. The one implementation of a
    plugin's version: evals.suite_scoring_version reports it too."""
    import inspect
    import sys as _sys

    from ..runners.eval_suites import SUITES

    fn = SUITES.get(name)
    if fn is None:
        return None
    module = _sys.modules.get(getattr(fn, "__module__", "") or "")
    path = getattr(module, "__file__", None)
    if not path:
        try:
            path = inspect.getsourcefile(fn)
        except TypeError:
            path = None
    return _file_sha256(path)


def _normalized_shard(config: ConfigLike) -> Optional[str]:
    """The run's shard as the suites read it (``config.shard``, else ``GBENCH_SHARD``), in its
    normalized I/N form; None for the whole set, a 1/1 shard included. A malformed value raises
    ValueError, as it does for the suites: the run cannot start with it."""
    import os

    from ..runners.eval_suites.sampling import parse_shard

    raw = getattr(config, "shard", None) or (os.environ.get("GBENCH_SHARD") or "").strip()
    parsed = parse_shard(raw) if raw else None
    if not parsed or parsed[1] <= 1:
        return None
    return f"{parsed[0]}/{parsed[1]}"


def build_evals_scaffold(config: ConfigLike) -> Scaffold:
    """Build the scaffold for the academic evals pillar.

    Every input ``runners/evals.py`` hands the suites is pinned where it
    belongs, resolved the way the run resolves it:

    * ``load_shape``: the request concurrency (``runners.evals.
      eval_concurrency``: ``--eval-concurrency``, else ``--batch-sizes[0]``),
      the ``--sandboxes`` value every suite receives, and the containers a
      selected containerized suite runs at once (``--sandboxes``, else
      ``--batch-sizes[0]``).
    * ``dataset_snapshot``: the suites the run executes (``--evals``
      expanded as the runner expands it), ``--eval-categories``,
      ``--eval-limit``, the shard, the sha256 of the ``--eval-custom-jsonl``
      file, and the set dataset-class run-shaping variables.
    * ``decode``: ``--temperature`` (null = each suite's think-aware
      default), ``--max-output-tokens``, ``--eval-max-soft-tokens`` and the
      set decode-class variables (penalties, soft-token overrides, the
      tool-loop budget, ``GBENCH_<SUITE>_TEMPERATURE``).
    * ``prompt_version``: few-shot count, ``--thinking`` and the
      prompt-class variables (agentic vs one-shot modes, prompt variants).
    * ``attempts``: ``--attempt-count``, the independent attempts per
      sample behind avg@k / pass@k.
    * ``scoring_version``: each suite's canonical-sync anchors and those of
      every suite and shared module it imports (``canonical_sync.
      scoring_signal``), a discovered plugin's module sha256, the Gemini
      judge and search cascades with the judge temperature, and the
      scoring-class variables.

    The run-shaping variables are the ones ``base.run_shaping_env``
    declares; only the ones that are set are recorded, so an unset
    variable leaves the id where it was.
    """
    from ..runners.eval_suites.base import (
        JUDGE_TEMPERATURE,
        judge_cascade,
        run_shaping_env,
        search_cascade,
    )
    from ..runners.eval_suites.canonical_sync import CANONICAL_SYNC, scoring_signal

    suites = sorted(set(resolved_eval_suites(config)))
    env = run_shaping_env(suites)
    scoring_env = {k: v for k, v in env.get("scoring", {}).items() if k not in _CASCADE_ENV}

    from ..runners.evals import eval_concurrency

    signal = scoring_signal(suites)
    plugins = {name: _plugin_version(name) for name in suites if name not in CANONICAL_SYNC}
    for name in plugins:
        signal["suites"].pop(name, None)

    return Scaffold(
        pillar=Pillar.EVALS,
        covers=EVALS_COVERS,
        pinned={
            ContractField.SERVING_ENVIRONMENT.value:
                build_serving_environment(config, context="native").to_dict(),
            ContractField.LOAD_SHAPE.value: LoadShape(
                concurrency=eval_concurrency(config),
                sandboxes=getattr(config, "sandboxes", None),
                container_concurrency=_container_concurrency(config, suites),
            ).to_dict(),
            ContractField.DATASET_SNAPSHOT.value: {
                "evals": suites or None,
                "eval_categories": config.eval_categories,
                "eval_limit": getattr(config, "eval_limit", None),
                "shard": _normalized_shard(config),
                "custom_jsonl_sha256": (
                    _file_sha256(getattr(config, "eval_custom_jsonl", None))
                    if "custom_jsonl" in suites else None),
                "env": env.get("dataset", {}),
            },
            ContractField.DECODE.value: {
                "max_soft_tokens": config.eval_max_soft_tokens,
                "temperature": getattr(config, "temperature", None),
                "max_output_tokens": getattr(config, "eval_max_output_tokens", None),
                "env": env.get("decode", {}),
            },
            ContractField.PROMPT_VERSION.value: {
                "n_shot": config.eval_n_shot,
                "thinking": config.eval_thinking,
                "env": env.get("prompt", {}),
            },
            ContractField.ATTEMPTS.value: _evals_attempts(config, suites),
            ContractField.SCORING_VERSION.value: {
                **signal,
                "plugins": plugins,
                "judge": {
                    "judge_cascade": judge_cascade(),
                    "search_cascade": search_cascade(),
                    "judge_temperature": JUDGE_TEMPERATURE,
                },
                "env": scoring_env,
            },
        },
    )


def build_unmodelled_scaffold(pillar: Union[Pillar, str]) -> Scaffold:
    """A pillar that ran before its scaffold was modelled.

    Emitted rather than omitted so the coverage gap is visible in the
    results instead of looking like the pillar did not run.
    """
    return Scaffold(pillar=Pillar(pillar), covers=(), pinned={})


def _scaffold_for(
    pillar: Pillar,
    dataset_dir: Optional[Union[Path, str]],
    selected_tasks: Optional[Sequence[str]],
    config: Optional[ConfigLike],
) -> Scaffold:
    """Pick the builder for a pillar, falling back to an empty block.

    Every path that cannot pin anything returns an unmodelled scaffold
    rather than raising or omitting the condition. A pillar added to the
    enum with no builder yet therefore reports ``0/9`` and shows up in
    the table, which is the visible-gap behaviour the null scaffold
    exists for. Failing open to silence would be the one wrong answer.
    """
    if pillar is Pillar.GOLDEN:
        if dataset_dir is None:
            return build_unmodelled_scaffold(pillar)
        return build_golden_scaffold(dataset_dir, selected_tasks, config)

    if config is None:
        return build_unmodelled_scaffold(pillar)

    if pillar in _PERFORMANCE_PILLARS:
        return build_performance_scaffold(pillar, config)
    if pillar in (Pillar.STRESS_TEST, Pillar.STRESS_TEST_MULTIMODAL):
        return build_stress_scaffold(config, pillar)
    if pillar is Pillar.QUALITY:
        return build_quality_scaffold(config)
    if pillar is Pillar.EVALS:
        return build_evals_scaffold(config)

    return build_unmodelled_scaffold(pillar)


def build_condition(
    pillar: Union[Pillar, str],
    model: ModelLike,
    serving_format: Union[ModelFormat, str],
    dataset_dir: Optional[Union[Path, str]] = None,
    selected_tasks: Optional[Sequence[str]] = None,
    config: Optional[ConfigLike] = None,
) -> Condition:
    """Build one ``(pillar, model, format)`` condition for ``metadata.json``.

    ``config`` is optional so that a caller holding only a dataset can
    still get the golden half of the contract, but every non-golden
    pillar needs it to pin anything at all. Omitting it yields the same
    empty blocks this module emitted before the run configuration was
    modelled.
    """
    pillar = Pillar(pillar)
    scaffold = _scaffold_for(pillar, dataset_dir, selected_tasks, config)

    return Condition(
        pillar=pillar,
        model=model.name,
        serving_format=_format_value(serving_format),
        scaffold=scaffold,
        subject=build_subject(model, serving_format),
    )


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------

def render_conditions(conditions: Sequence[Condition]) -> str:
    """Render the contract for a terminal.

    The ids have to be readable without any dashboard. Plenty of gbench
    runs are headless, on a workstation or in CI, and a run whose scaffold
    is only legible after uploading it somewhere is a run whose scaffold
    people will not look at. Console and ``metadata.json`` are two views
    of the same objects, and neither is required for the other to work.

    The per-scaffold footer is printed once per distinct ``scaffold_id``
    rather than once per row, because the interesting property is that a
    scaffold is *shared* across subjects. Repeating it on every line
    would bury exactly the thing worth noticing.
    """
    if not conditions:
        return ""

    lines = [
        "=" * 80,
        f"SCAFFOLD CONTRACT (v{CONTRACT_VERSION})",
        "=" * 80,
        "",
        "A result is the pair (subject, scaffold). Holding scaffold_id while",
        "subject_id moves is a like-for-like model comparison. A scaffold_id that",
        "moved means the harness moved, so scores either side are not comparable.",
        "",
    ]

    rows = [
        (c.pillar.value, c.model, c.serving_format,
         c.subject.subject_id, c.scaffold.scaffold_id or "none")
        for c in conditions
    ]
    headers = ("PILLAR", "MODEL", "FORMAT", "SUBJECT", "SCAFFOLD")
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def row(cells: Sequence[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    lines.append(row(headers))
    lines.extend(row(r) for r in rows)

    # dict rather than set, so the footers come out in the order the
    # scaffolds were first planned instead of in hash order.
    #
    # Pillars are accumulated per label rather than read off the first
    # condition that claimed it. Every unmodelled pillar shares the label
    # "none", so taking the first would print "pins 0/9 for serving"
    # while silently speaking for throughput and stress_test as well.
    # Under-disclosure is the one failure mode a null scaffold exists to
    # prevent, so it must not be reintroduced by the renderer.
    seen: Dict[str, Scaffold] = {}
    pillars_for: Dict[str, List[str]] = {}
    for c in conditions:
        label = c.scaffold.scaffold_id or "none"
        seen.setdefault(label, c.scaffold)
        named = pillars_for.setdefault(label, [])
        if c.pillar.value not in named:
            named.append(c.pillar.value)

    # Width of the label gutter, from the labels themselves rather than
    # from the table above, so the hanging indent lines up whichever of
    # the two happens to be wider.
    gutter = max(len(label) for label in seen) + 2
    pad = " " * gutter

    def wrapped(prefix: str, body: str) -> List[str]:
        return textwrap.wrap(
            body, width=_LINE_WIDTH,
            initial_indent=prefix, subsequent_indent=pad,
        )

    for label, scaffold in seen.items():
        lines.append("")
        # Out of the scaffold fields: the two subject fields are the subject
        # id's to report, and counting them here would cap every scaffold at
        # 9/11 for fields no scaffold may pin.
        head = f"{label.ljust(gutter - 2)}  pins {len(scaffold.covers)}/{len(SCAFFOLD_FIELDS)}"

        if not scaffold.covers:
            named = pillars_for[label]
            # "Nothing" is the subject of the sentence, so the verb stays
            # singular however many pillars are listed.
            harness = (
                "this pillar's harness" if len(named) == 1
                else "these pillars' harnesses"
            )
            lines.extend(wrapped(
                f"{head} for {', '.join(named)}. ",
                f"Nothing about {harness} is captured yet, so a score change "
                "here cannot be attributed to anything.",
            ))
            continue

        lines.extend(wrapped(
            f"{head}: ", ", ".join(sorted(f.value for f in scaffold.covers))))
        if scaffold.dataset is not None and scaffold.dataset.snapshot_id:
            lines.append(
                f"{pad}dataset {scaffold.dataset.snapshot_id} over "
                f"{scaffold.dataset.case_count} case(s)"
            )
        lines.extend(wrapped(
            f"{pad}unpinned: ",
            ", ".join(f.value for f in scaffold.unpinned) or "none"))

    # The subject axis, once for the run: every subject id is identity
    # (model id and serving format), and saying which of its fields are
    # unpinned keeps a reader from taking a held subject id for a held
    # checkpoint.
    subject_unpinned = [
        f.value for f in SUBJECT_FIELDS
        if any(f in c.subject.unpinned for c in conditions)
    ]
    lines.append("")
    lines.extend(textwrap.wrap(
        f"Subject ids pin {len(SUBJECT_FIELDS) - len(subject_unpinned)}/"
        f"{len(SUBJECT_FIELDS)}; unpinned: {', '.join(subject_unpinned) or 'none'}. A subject "
        "id hashes the model id and serving format; no checkpoint revision or quantization "
        "level is resolved.",
        width=_LINE_WIDTH,
    ))
    lines.append("")
    return "\n".join(lines)
