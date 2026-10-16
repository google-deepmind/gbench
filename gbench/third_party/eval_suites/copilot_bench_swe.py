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

# Evaluation Suite: copilot_bench_swe
# Description: SWE-bench Verified (resolved rate via the SWE-bench v5 Docker harness)

"""gbench native built-in runner for copilot_bench_swe (Coding & Software Engineering).

SWE-bench Verified (500 instances) from the SWE-bench maintainers' current dataset,
SWE-bench/SWE-bench_Verified at a pinned commit, scored by execution-based resolved rate with
the SWE-bench v5 harness (swebench 5.0.2, which grades each instance with the `image`,
`eval_script` and `log_parser` columns of that dataset). SANDBOX_EVAL; hard-errors
(infra_required) when Docker or the local images are missing.

Protocol (default, agentic): mini-swe-agent 2.4.2 with its bundled `benchmarks/swebench.yaml`,
unchanged, works inside each instance image (repo at /testbed) and submits its patch; that patch
is scored. This is the setup of the SWE-bench maintainers' current Verified submission
(SWE-bench/experiments, 20260901_mini-v2.4.2_gemini-3-5-flash: "mini's bundled swebench.yaml
config unchanged", verdicts from the v5 harness). gemma4-env pins mini-swe-agent 1.x, so 2.4.2
runs in the local image gbench-mini-swe-agent-v2:2.4.2-2.4.6 and the harness in gbench-swebench-v5:5.0.2. A
full-split run with every instance scored follows the leaderboard protocol except for the
temperature (see Sampling). One-shot (one diff from
the issue text, no repository access) is opt-in with GBENCH_COPILOT_BENCH_SWE_MODE=oneshot and
is never comparable. The name is historical: nothing here selects or measures long context.

Sampling: the temperature follows gbench's rule for every suite: 1.0 for a `--thinking` run
(temperature_source "thinking"), 0.0 without it (temperature_source "default"), in both modes. An
explicit `--temperature` or `GBENCH_COPILOT_BENCH_SWE_TEMPERATURE` (which takes precedence) beats both.
The bundled 2.4.2 config sends no temperature, so the agentic run's temperature is recorded and
makes the run not leaderboard-comparable.
"""

from typing import Any, Dict, Optional
from . import swe_agentic
from .swebench_common import (
    SWEBENCH_V5_COLUMNS, check_swebench_v5_prereq, load_swe_samples, execute_swebench,
    execute_swebench_agentic, infra_required, pinned_swe_snapshot, swebench_v5_image,
)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/copilot_bench_swe.md"
_DATASET = "SWE-bench/SWE-bench_Verified"
#: Hub commit the rollout and the harness both read (pinned_swe_snapshot): the head of the SWE-bench
#: maintainers' dataset since 2026-08-16, carrying the v5 harness columns (added 2026-08-09).
_REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"
_SPLIT = "test"
_NAMESPACE = "swebench"
#: What a snapshot of _REVISION must hold before it is used (pinned_swe_snapshot checks all
#: three): the columns the v5 harness builds a TestSpec from, the split's row count at that
#: commit, and the content fingerprint of its rows (swebench_common.snapshot_sha256).
SNAPSHOT_COLUMNS = SWEBENCH_V5_COLUMNS
SPLIT_ROWS = {"test": 500}
SPLIT_SHA256 = {"test": "1939f0b7e4fca575369dad9bca3a843c48d745750ccc866999728c54d3dafd04"}
#: mini-swe-agent release of the SWE-bench maintainers' submission this protocol reproduces.
AGENT_VERSION = "2.4.2"


def check_copilot_bench_swe_prerequisites():
    try:
        snapshot = pinned_swe_snapshot(_DATASET, _SPLIT, _REVISION, "copilot_bench_swe",
                                       columns=SNAPSHOT_COLUMNS, rows=SPLIT_ROWS.get(_SPLIT),
                                       sha256=SPLIT_SHA256.get(_SPLIT))
    except RuntimeError as e:
        return False, str(e)
    return check_swebench_v5_prereq(snapshot, _SPLIT)


def _load_copilot_bench_swe_dataset(limit: Optional[int] = None, split: str = _SPLIT):
    return load_swe_samples(_DATASET, split, limit, "copilot_bench_swe", revision=_REVISION,
                            columns=SNAPSHOT_COLUMNS, rows=SPLIT_ROWS.get(split),
                            sha256=SPLIT_SHA256.get(split))


def run_copilot_bench_swe(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run SWE-bench Verified: agentic by default, one-shot when opted in."""
    # The split goes to the executor positionally, so it must not stay in kwargs too.
    split = kwargs.pop("split", None) or _SPLIT
    if split not in SPLIT_ROWS:
        raise ValueError(f"copilot_bench_swe: split {split!r} is not a pinned split of "
                         f"{_DATASET}@{_REVISION} (pinned: {', '.join(sorted(SPLIT_ROWS))})")
    ok, reason = check_copilot_bench_swe_prerequisites()
    if not ok:
        raise infra_required("copilot_bench_swe", reason, DOCS_URL)
    kwargs["enable_thinking"] = enable_thinking
    kwargs["dataset_revision"] = _REVISION
    kwargs["snapshot_columns"] = SNAPSHOT_COLUMNS
    kwargs["snapshot_rows"] = SPLIT_ROWS[split]
    kwargs["snapshot_sha256"] = SPLIT_SHA256[split]
    kwargs["harness_image"] = swebench_v5_image()
    kwargs["harness_v5"] = True
    mode, _ = swe_agentic.resolve_mode("copilot_bench_swe")
    if mode == swe_agentic.AGENTIC:
        swe_agentic.require_agent_v2("copilot_bench_swe", DOCS_URL, AGENT_VERSION)
        return execute_swebench_agentic(
            "copilot_bench_swe", model_name, base_url, concurrency,
            _DATASET, split, _NAMESPACE, agent_v2=AGENT_VERSION, **kwargs,
        )
    return swe_agentic.mark_oneshot(execute_swebench(
        "copilot_bench_swe", model_name, base_url, concurrency,
        _DATASET, split, _NAMESPACE, **kwargs,
    ), "copilot_bench_swe")
