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

# Evaluation Suite: swe_bench_multilingual
# Description: SWE-bench Multilingual (300 issues across 9 non-Python languages, resolved rate)

"""gbench native built-in runner for swe_bench_multilingual (Coding & Software Engineering).

Canonical SWE-bench Multilingual (SWE-bench/SWE-bench_Multilingual at a pinned commit, 300
instances) scored by execution-based resolved rate with the SWE-bench v5 harness (swebench
5.0.2, which grades each instance with the dataset's `image`, `eval_script` and `log_parser`
columns; per-language tests run inside the prebuilt instance images, so host toolchains are
irrelevant). Thin wrapper over swebench_common; SANDBOX_EVAL. Hard-errors (infra_required,
never skips) if Docker, the local images or the dataset are absent.

Protocol (default, agentic): mini-swe-agent 2.4.6 with its bundled `benchmarks/swebench.yaml`,
unchanged, works inside each instance image and its submitted diff is scored. The Multilingual
leaderboard's mini-SWE-agent rows are 2.x; 2.4.6 is the release of the SWE-bench maintainers'
current Multilingual submission (SWE-bench/experiments, 20260902_mini-v2.4.6_gemini-3-5-flash:
"mini's bundled swebench.yaml config unchanged", verdicts from the v5 harness). gemma4-env pins
mini-swe-agent 1.x, so 2.4.6 runs in the local image gbench-mini-swe-agent-v2:2.4.2-2.4.6 and the harness in
gbench-swebench-v5:5.0.2. A full-split run with every instance scored follows the
leaderboard protocol except for the temperature (see Sampling).
One-shot is opt-in with GBENCH_SWE_BENCH_MULTILINGUAL_MODE=oneshot and is never comparable.

Sampling: the temperature follows gbench's rule for every suite: 1.0 for a `--thinking` run
(temperature_source "thinking"), 0.0 without it (temperature_source "default"), in both modes. An
explicit `--temperature` or `GBENCH_SWE_BENCH_MULTILINGUAL_TEMPERATURE` (which takes precedence) beats both.
The bundled 2.4.6 config sends no temperature, so the agentic run's temperature is recorded and
makes the run not leaderboard-comparable.
"""

from typing import Any, Dict, Optional
from . import swe_agentic
from .swebench_common import (
    SWEBENCH_V5_COLUMNS, check_swebench_v5_prereq, load_swe_samples, execute_swebench,
    execute_swebench_agentic, infra_required, pinned_swe_snapshot, swebench_v5_image,
)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/swe_bench_multilingual.md"
_DATASET = "SWE-bench/SWE-bench_Multilingual"
#: Hub commit the rollout and the harness both read (pinned_swe_snapshot): the head since
#: 2026-08-17, with the v5 harness columns.
_REVISION = "846e647b9f33c0b51b739d005d13d85493c9af09"
_SPLIT = "test"
_NAMESPACE = "swebench"
#: What a snapshot of _REVISION must hold before it is used (pinned_swe_snapshot checks all
#: three): the columns the v5 harness builds a TestSpec from, the split's row count at that
#: commit, and the content fingerprint of its rows (swebench_common.snapshot_sha256).
SNAPSHOT_COLUMNS = SWEBENCH_V5_COLUMNS
SPLIT_ROWS = {"test": 300}
SPLIT_SHA256 = {"test": "83441ebb91ffa12cb167d02ff816ba4d2d30afa5f036ade5f6ab56bee09e37ce"}
#: mini-swe-agent release of the SWE-bench maintainers' submission this protocol reproduces.
AGENT_VERSION = "2.4.6"


def check_swe_bench_multilingual_prerequisites():
    try:
        snapshot = pinned_swe_snapshot(_DATASET, _SPLIT, _REVISION, "swe_bench_multilingual",
                                       columns=SNAPSHOT_COLUMNS, rows=SPLIT_ROWS.get(_SPLIT),
                                       sha256=SPLIT_SHA256.get(_SPLIT))
    except RuntimeError as e:
        return False, str(e)
    return check_swebench_v5_prereq(snapshot, _SPLIT)


def _load_swe_bench_multilingual_samples(limit: Optional[int] = None, split: str = _SPLIT):
    return load_swe_samples(_DATASET, split, limit, "swe_bench_multilingual", revision=_REVISION,
                            columns=SNAPSHOT_COLUMNS, rows=SPLIT_ROWS.get(split),
                            sha256=SPLIT_SHA256.get(split))


def run_swe_bench_multilingual(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run SWE-bench Multilingual: agentic by default, one-shot when opted in."""
    # The split goes to the executor positionally, so it must not stay in kwargs too.
    split = kwargs.pop("split", None) or _SPLIT
    if split not in SPLIT_ROWS:
        raise ValueError(f"swe_bench_multilingual: split {split!r} is not a pinned split of "
                         f"{_DATASET}@{_REVISION} (pinned: {', '.join(sorted(SPLIT_ROWS))})")
    ok, reason = check_swe_bench_multilingual_prerequisites()
    if not ok:
        # No-skip policy: hard-error when Docker, the images or the dataset are unavailable
        # (matches the sibling copilot_bench_swe / swe_bench_live suites), never a skip row.
        raise infra_required("swe_bench_multilingual", reason, DOCS_URL)
    kwargs["enable_thinking"] = enable_thinking
    kwargs["dataset_revision"] = _REVISION
    kwargs["snapshot_columns"] = SNAPSHOT_COLUMNS
    kwargs["snapshot_rows"] = SPLIT_ROWS[split]
    kwargs["snapshot_sha256"] = SPLIT_SHA256[split]
    kwargs["harness_image"] = swebench_v5_image()
    kwargs["harness_v5"] = True
    mode, _ = swe_agentic.resolve_mode("swe_bench_multilingual")
    if mode == swe_agentic.AGENTIC:
        swe_agentic.require_agent_v2("swe_bench_multilingual", DOCS_URL, AGENT_VERSION)
        return execute_swebench_agentic(
            "swe_bench_multilingual", model_name, base_url, concurrency,
            _DATASET, split, _NAMESPACE, agent_v2=AGENT_VERSION,
            **kwargs,
        )
    return swe_agentic.mark_oneshot(execute_swebench(
        "swe_bench_multilingual", model_name, base_url, concurrency,
        _DATASET, split, _NAMESPACE, **kwargs,
    ), "swe_bench_multilingual")
