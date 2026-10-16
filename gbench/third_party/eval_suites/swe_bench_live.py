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

# Evaluation Suite: swe_bench_live
# Description: SWE-bench-Live - continuously-updated real GitHub issue resolution (resolved rate)

"""gbench native built-in runner for swe_bench_live (Coding & Software Engineering).

SWE-bench-Live, every tab of its leaderboard, each scored by the harness upstream uses for it,
each with its own resolved-rate headline.

Leaderboard settings: the SWE-bench-Live leaderboard (SWE-bench-Live/swe-bench-live.github.io
@90e8720259d7) has ten tabs (index.html tab buttons; leaderboard.js validSets), each ranking
resolved / total of its own rows (leaderboard.js: resolved_percentage = resolved /
report.total). A run evaluates all ten by default, each with its own rollouts, its own harness
pass and its own headline; the tabs run at once within --sandboxes (tab_schedule), except the
Python tabs, which run one after the other:

  * the Python dataset `SWE-bench-Live/SWE-bench-Live`, scored by the SWE-bench-Live fork of the
    swebench harness (FORK_COMMIT) in the local image:
      - lite: split `lite` at _REVISION (300 rows, frozen upstream);
      - full: split `full` at _FULL_REVISION (1,319 rows). Every Full row on the leaderboard has
        total 1319: the `full` split of the May 2025 release (issues created 2024-01 to
        2025-04). The split has grown since (12b98891, 2025-07-02, made it 1,565 rows; 1,888 at
        _REVISION) and no leaderboard row scores the larger set; every one of the 1,319 rows is
        unchanged in it;
      - verified: split `verified` at _REVISION (500 rows, frozen upstream; it lists one
        instance id twice, and the headline divides by rows as the leaderboard does);
  * `SWE-bench-Live/MultiLang` at MULTILANG_REVISION, scored by upstream's multi-language
    harness (microsoft/SWE-bench-Live `evaluation/evaluation.py` at MAIN_COMMIT with its
    RepoLaunch submodule at REPOLAUNCH_COMMIT, `--platform linux`) in the same local image:
      - ccpp (tab C/C++): splits `c` + `cpp`; csharp (C#): `cs`; java: `java`; tsjs (TS/JS):
        `js` + `ts`; go: `go`; rust: `rust`;
  * windows: `SWE-bench-Live/Windows` split `test` at WINDOWS_REVISION, whose instance images
    are Windows containers. The same harness scores it with `--platform windows`, and the
    rollout runs Win-Agent (WIN_AGENT_COMMIT), the only scaffold on that tab. Both have to run
    on a Windows host whose Docker engine runs Windows containers (RepoLaunch bind-mounts the
    harness's working directory into each task container).

`--eval-categories` (routed here by evals.route_eval_categories) or the programmatic `split=`
picks a subset, e.g. `--eval-categories lite,java`.

Per-setting prerequisites (missing_prerequisites): every selected setting whose prerequisites
are present runs, and each selected setting whose prerequisite is missing is reported in its own
entry with status `infra_required` and the missing prerequisite named (never scored, never
dropped; the suite's status is then `completed_with_errors`, with the setting in
`settings_infra_required`). The rule works in both directions: the Windows tab needs a Windows
host (check_windows_prerequisites), and the Python and MultiLang tabs need the Linux images (the
harness image, and in the agentic mode SWE-agent-for-eval's), which a Windows-containers engine
cannot hold; a pinned snapshot that cannot be fetched stops its own setting. A run whose
selected settings all lack their prerequisites hard-errors (infra_required), e.g.
`--eval-categories windows` on Linux.

Cloud metadata server: every tab runs model-controlled code in task containers with network
access (the rollouts, then the tests over the patches). Before the first Linux tab starts, and
again before each SWE-agent batch and each MultiLang harness pass (cached),
swebench_common.require_task_network_blocked checks that 169.254.169.254 is unreachable from
docker's default bridge and from a user-defined bridge network (infra_required otherwise, for the
whole suite); in the agentic mode it also checks that the hosts SWE-ReX's per-instance runtime
build fetches from resolve there (SWEREX_BUILD_HOSTS). The Windows tab runs only after
check_windows_metadata_guard established the same verdict on the Windows-containers engine:
swebench_common.check_windows_task_network runs a PowerShell probe (a TCP connect to
169.254.169.254:80 and to [fd20:ce::254]:80, the metadata server's IPv6 address, nothing sent) in
one of the tab's own instance images already on the host, on docker's default network, where
Win-Agent and the harness start their task containers. On a Linux
engine, with no such image present, or without a clear verdict, the tab is reported
infra_required with the reason.

Protocol (default, agentic): SWE-bench-Live's rollout rule is the problem statement and the
instance image only (microsoft/SWE-bench-Live evaluation/README.md). The Python and MultiLang
tabs run SWE-agent-for-eval (njukenanli/SWE-agent-for-eval at SWEAGENT_COMMIT), the SWE-agent
fork that README names for the leaderboard's runs, with its own config/default.yaml unmodified
and its own rollout command (`sweagent run-batch --instances.type swe_bench`), from the LOCAL
image gbench/docker/sweagent_for_eval.Dockerfile, one batch per tab; its submission (`git add -A
&& git diff --cached` in the repository, SWE-agent's submit tool) is the patch, as in the
leaderboard's SWE-agent rows. The Windows tab runs Win-Agent with its own config/default.yaml.
One-shot is opt-in with GBENCH_SWE_BENCH_LIVE_MODE=oneshot. Instance images are pulled on the host
(swebench_common.host_pull_images, the operator's Docker login) before the SWE-agent container
or a Linux harness pass starts task containers from them; neither container gets any Docker
client configuration. An endpoint that kept failing ends a SWE-agent rollout with exit_error
("litellm.<Class>: ..."): an infrastructure failure when the tab's relay saw an endpoint failure
(the text alone can come from the runtime in the model's container); exit_api (a tenacity
RetryError, which the fork's model calls never raise) is one too on the same condition; any
other exit_error or exit_environment_error (e.g. the EOF of a shell the model killed) is the
model's outcome (sweagent_exit_verdict). A Win-Agent "error" before the model's first turn is an
infrastructure failure; after it, only an error naming an endpoint failure while the relay saw
one is (the log also holds the model's command output); any other is the model's outcome
(win_agent_error_verdict). Both kinds of kept ending are counted in
`exits_kept_as_model_outcome`. A SWE-agent instance with no progress for longer than the fork's
limits allow is ended by gbench's watchdog and is a harness error (`agent_stalled`). The task
containers run gbench's patched SWE-ReX runtime (SWEREX_RUNTIME_SHA256), so the model's
commands run as bash runs them, heredocs included. The relay's counts of endpoint failures and of
model turns cut off at a limit apply per tab (swebench_common.relay_result_reasons). An infrastructure failure of a
rollout (`agent_infra`), an instance a harness gave no verdict (`harness_no_verdict`: the fork's
errored instance whose log shows neither a patch-apply failure nor a test timeout, the
multi-language harness's error_ids / incomplete_ids) and every submitted patch of a harness pass
that wrote no report (`harness_no_report`) are harness errors: left out of the tab's resolved rate
(numerator and denominator, swebench_common.scored_headline), counted in `harness_errors`, and a
tab with nothing scored withholds its rate (None, status error).

The fork is the `swebench` PACKAGE at a version incompatible with the upstream `swebench` the
main env carries, so gbench ISOLATES the harnesses in a LOCAL image
(gbench/docker/swe_bench_live.Dockerfile): only the scoring step runs inside it,
docker-out-of-docker. A missing prerequisite (Docker, the SWE-agent image for the agentic
Python/MultiLang tabs, the harness image, a pinned dataset snapshot that cannot be fetched or
does not match) follows the per-setting rule above; with no selected setting runnable the suite
HARD-ERRORS (infra_required, never skips).

Sampling: SWE-agent-for-eval sends no temperature and no max_tokens (SWEAGENT_SAMPLING: its
model call has the temperature and top_p arguments commented out), and Win-Agent sends neither
either (WIN_AGENT_SAMPLING); `--max-output-tokens` is not applied (recorded), with or without
`--thinking`. The temperature follows gbench's rule for every suite: 1.0 for a `--thinking` run
(temperature_source "thinking"), 0.0 without it (temperature_source "default"). Set a temperature
for a whole run with `--temperature`, or for this suite alone with
`GBENCH_SWE_BENCH_LIVE_TEMPERATURE`, which takes precedence over both. gbench's relay sets the
temperature on every model request and the result records it; since the scaffolds send none, the
tab is not leaderboard-comparable. The one-shot mode follows the same rule.
"""

import json
import logging
import os
import platform as _platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import swe_agentic, swe_thread_cap, swebench_common
from .base import (
    OfficialSampling, decoding_penalties, get_run_knob, resolve_official_sampling,
    resolve_token_budget, run_eval_suite, tally_batch_scores,
)
from .limit_plan import eval_limit_plan
from .sampling import allocate_limit
from .sandbox import credential_free_env
from .swebench_common import (
    AGENT_INFRA, CONTEXT_OVERFLOW_RE, NO_REPORT, NO_VERDICT, EndpointRelay,
    _pinned_snapshot_rows, execute_swebench, harness_errors_text, host_pull_images,
    image_pull_summary, infra_required, load_swe_samples, mark_harness_error,
    pinned_swe_snapshot, record_harness_errors, relay_failure_count, relay_result_reasons,
    repeated_ids_record, row_total_headline, scaffold_error_harness_reason, scored_headline,
    SWE_LIMIT_SEED, swe_samples_from_rows, swebench_report_reasons, trace_patch,
    unique_instance_rows,
    withhold_on_error,
)

logger = logging.getLogger(__name__)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/swe_bench_live.md"
_DATASET = "SWE-bench-Live/SWE-bench-Live"
#: Hub commit of the Lite and Verified settings (Hub main since 2026-09-04). Both splits are
#: frozen upstream.
_REVISION = "b51a86422e10cfd403beb4773e5a2947953e36ec"
#: Hub commit of the Full setting: the last commit whose `full` split is the 1,319 rows the
#: leaderboard's Full rows are scored on (the May 2025 release; the next data upload,
#: 12b98891 on 2025-07-02, grew it to 1,565).
_FULL_REVISION = "3da151be953883d3cd4e1ea0dc7a042ff2abf8f4"
_NAMESPACE = "starryzhang"

MULTILANG_DATASET = "SWE-bench-Live/MultiLang"
#: Hub main as of 2026-09-20 (its data files are those of the 2026-08-20 upload, 62dc0745: 1,077
#: instances in 8 language splits).
MULTILANG_REVISION = "3638632e8153a10ca422c1022bed79023084b5c9"
WINDOWS_DATASET = "SWE-bench-Live/Windows"
#: Hub main as of 2026-09-20 (data of 868cf7b0, 2026-09-10: 66 instances).
WINDOWS_REVISION = "b9fa25f2732dbc2c93b9738b610d053d8b7bd7a9"

#: The leaderboard's tabs, in its order (leaderboard.js validSets = index.html tab buttons).
SETTINGS = ("lite", "full", "verified", "ccpp", "csharp", "java", "tsjs", "go", "rust",
            "windows")
PYTHON_SETTINGS = ("lite", "full", "verified")
MULTILANG_SETTINGS = ("ccpp", "csharp", "java", "tsjs", "go", "rust")
WINDOWS_SETTINGS = ("windows",)
#: Each tab's label on the leaderboard (index.html).
TAB_LABELS = {"lite": "Lite", "full": "Full", "verified": "Verified", "ccpp": "C/C++",
              "csharp": "C#", "java": "Java", "tsjs": "TS/JS", "go": "Go", "rust": "Rust",
              "windows": "Windows"}
#: The setting whose resolved rate is the result's `accuracy` (the leaderboard's default tab,
#: leaderboard.js selectedSet = 'lite'); every setting's rate is in `headline_metrics`.
HEADLINE_SETTING = "lite"
_DEFAULT_SPLIT = HEADLINE_SETTING
#: Hub commit each Python setting's split is read at.
SETTING_REVISION = {"lite": _REVISION, "full": _FULL_REVISION, "verified": _REVISION}
#: What a Python setting's snapshot must hold before it is used (pinned_swe_snapshot checks
#: all three): the columns the fork's make_test_spec reads, the split's row count at the
#: setting's commit, and the content fingerprint of its rows (swebench_common.snapshot_sha256).
#: `verified` repeats one instance id as published (conan-io__conan-18153, two rows that
#: differ in test_cmds and all_hints_text); it is used as published: the fork harness reads
#: every row, the rollout runs once per instance id (swebench_common.unique_instance_rows) and
#: the headline counts rows (swebench_common.row_total_headline).
SNAPSHOT_COLUMNS = ("instance_id", "repo", "base_commit", "problem_statement", "test_patch",
                    "test_cmds", "log_parser", "FAIL_TO_PASS", "PASS_TO_PASS")
SPLIT_ROWS = {"lite": 300, "full": 1319, "verified": 500}
SPLIT_SHA256 = {
    "lite": "b9252aeb085eb949f01b8464bee76240d474146c9a4f6aed1bbb693c2b94755e",
    "full": "b231d1932ce614df07b6b4237db2a6f4eb6020213d0d9e919d95a62b976cf95e",
    "verified": "d3ae023598f62e0c24a025a5091ec6ef97224b62dd8aeb6224d7b14154e55974",
}
#: The columns upstream's multi-language harness reads from a MultiLang or Windows row
#: (evaluation.py run_instance / evaluate_instance / get_instance_month), plus what the
#: rollout reads: the problem statement, and the base commit SWE-agent-for-eval's instance
#: loader requires (SimpleBatchInstance.from_swe_bench).
LAUNCH_COLUMNS = ("instance_id", "repo", "problem_statement", "base_commit", "test_patch",
                  "test_cmds", "print_cmds", "rebuild_cmds", "log_parser", "FAIL_TO_PASS",
                  "PASS_TO_PASS", "docker_image", "created_at")
#: MultiLang splits at MULTILANG_REVISION: (rows, content sha256).
MULTILANG_SPLITS = {
    "c": (111, "713f362bef00486539bf457dd08b74713066d764a07fbbea9c12d1936461a0a8"),
    "cpp": (142, "6f0c50cca9be037126623f7f8af9d10ffc55f405dc51fdfe14d35b635e57eae5"),
    "cs": (136, "4086e1640f6d69ce7a2ef9c7b14d39b510c77cfddc4cae86efef5934177a3236"),
    "go": (138, "261851b6adc60e90747bd07e0b48ff5ad6f3ba7a5f8808c4e9b780a9c9fe8528"),
    "java": (160, "aa6dce064c88cf51109da87a6e993b8d05b786ef2e1a9da3791e1e8054289480"),
    "js": (108, "9cf48e9b78e56670ea210778bad59341fe2fda00e3d2d4e7937f7e35e4c4ef8e"),
    "rust": (171, "b60c2965b1def9aa90d1e8b3a22d09d8544336108202a972798e2aa804150917"),
    "ts": (111, "a3944c6d25f2d7112910e6d9a76cd5fc7f3dcadb2c5387a33d4db4c20826124f"),
}
#: The MultiLang splits behind each tab. Verified against the board: its 2026-01-08 rows have
#: C/C++ total 48 and TS/JS total 162, the c + cpp (31 + 17) and js + ts (75 + 87) row counts
#: of the MultiLang commit then current (3430730b).
MULTILANG_TAB_SPLITS = {"ccpp": ("c", "cpp"), "csharp": ("cs",), "java": ("java",),
                        "tsjs": ("js", "ts"), "go": ("go",), "rust": ("rust",)}
#: The Windows split at WINDOWS_REVISION: (rows, content sha256).
WINDOWS_SPLITS = {"test": (66, "1f2a2e4057b1408b8932b1474cf8c9f0ce94b1862c25c4fdb53525b23fc432c5")}
#: The totals the leaderboard's rows of each tab were scored on (reports-0605.jsonl
#: @90e8720259d7). The multi-language and Windows datasets grow, and each row keeps the total
#: of the snapshot it ran on; a gbench number compares directly only with rows of the same
#: total.
LEADERBOARD_TOTALS = {
    "lite": (300,), "full": (1319,), "verified": (500,), "ccpp": (48, 66),
    "csharp": (20, 28, 78), "java": (39, 62, 102, 109, 160), "tsjs": (162, 204),
    "go": (68, 98, 138), "rust": (45, 64, 94), "windows": (50, 60),
}
_IMAGE_DEFAULT = "gbench-swe-bench-live:a145aa87"
#: Commit of the SWE-bench-Live fork the image installs (gbench/docker/swe_bench_live.Dockerfile
#: SWE_BENCH_LIVE_REF), recorded in its `org.gbench.swe_bench_live_fork` label.
FORK_COMMIT = "a145aa87c62361532dacaa243398978164b234b7"
#: microsoft/SWE-bench-Live main (2026-09-25): `evaluation/evaluation.py`, the harness upstream's
#: README gives for SWE-bench-Live/MultiLang and SWE-bench-Live/Windows. Image label
#: `org.gbench.swe_bench_live_main`.
MAIN_COMMIT = "62d02936a0726fbf4597b05bd07b09fb99eddae6"
#: microsoft/RepoLaunch, the `launch` submodule MAIN_COMMIT points at (the harness imports its
#: container runtime and log-parser runner). Image label `org.gbench.repolaunch`.
REPOLAUNCH_COMMIT = "c4b623d930f3728e5338664bb634021b98492cbf"
#: Where the image keeps that tree, and the import path its harness runs with.
MAIN_TREE = "/opt/swe-bench-live-main"
MAIN_PYTHONPATH = f"{MAIN_TREE}/launch:{MAIN_TREE}"
#: njukenanli/Win-Agent main (2026-06-02), the scaffold of every Windows row on the leaderboard
#: (upstream evaluation/README.md names it for Windows tasks).
WIN_AGENT_COMMIT = "47d85cffd5342a513fa3fb72ec4b5347ced9bb36"
#: Win-Agent sends neither a temperature nor a max_tokens (src/llm.py: litellm.completion with
#: model, api_key, base_url, messages, tools, tool_choice).
WIN_AGENT_SAMPLING = OfficialSampling(
    reference=f"njukenanli/Win-Agent@{WIN_AGENT_COMMIT} src/llm.py LLM.query",
    send_temperature=False, send_max_tokens=False)

#: njukenanli/SWE-agent-for-eval main (2026-06-02): the SWE-agent microsoft/SWE-bench-Live's
#: evaluation README (@62d02936 evaluation/README.md, "The forked source codes of the agents we
#: used to run SWE-bench-Live") names, and the scaffold of the Python and MultiLang tabs here. It
#: runs from the LOCAL image SWEAGENT_IMAGE_DEFAULT (gbench/docker/sweagent_for_eval.Dockerfile,
#: label `org.gbench.sweagent_for_eval`).
SWEAGENT_COMMIT = "824224225e1e952b48da91f2d1bde38adb2115f2"
#: The config the fork's README runs, unmodified, and its sha256 at SWEAGENT_COMMIT (image label
#: `org.gbench.sweagent_for_eval_config_sha256`).
SWEAGENT_CONFIG = "config/default.yaml"
SWEAGENT_CONFIG_SHA256 = "05e707df58e9732ac4053f8bda0c0b175aa10f5c833f5419ba1443a01d9eb981"
#: That config's agent.model.per_instance_call_limit: model calls per instance.
SWEAGENT_CALL_LIMIT = 100
#: The fork's time limits (sweagent/agent/agents.py at SWEAGENT_COMMIT): handle_action runs every
#: command with `timeout=90*60` ("1.5 h limit to align with evaluation code"), whatever
#: ToolConfig's default execution_timeout (30 s, sweagent/tools/tools.py; config/default.yaml
#: does not override it) says; ToolConfig's default max_consecutive_execution_timeouts (3)
#: consecutive command timeouts end the rollout; forward ends it once its commands took more
#: than 5 h in all. The fork sets no wall-clock limit per instance or per batch, so one instance
#: may take hours. Every result records these limits (`agent`).
SWEAGENT_COMMAND_TIMEOUT_S = 90 * 60
SWEAGENT_MAX_CONSECUTIVE_TIMEOUTS = 3
SWEAGENT_TOTAL_EXECUTION_S = 5 * 60 * 60
#: The runtime that runs the model's commands in each task container, fixed by gbench. SWE-ReX's
#: DockerDeployment builds a runtime layer on every instance image (swerex/deployment/docker.py
#: glibc_dockerfile) and upstream installs the latest swe-rex from PyPI into it (swe-rex 1.4.0 with
#: bashlex 0.18 as of 2026-10-08). Its BashSession split every command with bashlex
#: (swerex/runtime/local.py _split_bash_command) and joined the pieces with " ; ": a heredoc whose
#: operator is not the last word of its line (`cat <<EOF > file`, `python3 - <<EOF 2>&1`,
#: `cat <<EOF | sh`) lost its body, and any heredoc followed by another command got that command
#: on its terminator's line, so the shell waited for a terminator that never came until the
#: fork's 90-minute command limit, although the model's command was valid bash (2026-10-08 smoke
#: run: 2-4 such timeouts per instance, 6 h for 5 instances). The SWE-agent image
#: (SWEAGENT_IMAGE_DEFAULT) applies gbench/docker/sweagent_for_eval_swerex_heredoc.patch to the
#: SWE-ReX it vendors: the runtime layer then installs swe-rex 1.4.0 and its dependencies at
#: pinned versions (SWEREX_RUNTIME_PINS, no resolution) and gbench's session code, which runs a
#: command that holds a heredoc whole (its text in a file, `eval "$(cat FILE; rm -f FILE)"`, which
#: bash parses as it parses a script) and splits and joins every other command as before (a piece
#: ending with the background operator `&` is no longer followed by " ; ", a syntax error). The
#: sha256 of that session code (swerex/runtime/local.py) is the image's
#: `org.gbench.sweagent_for_eval_swerex_runtime` label; an image without it runs the defective
#: runtime and is refused (check_sweagent_prerequisites).
SWEREX_RUNTIME_SHA256 = "d67d68966754cc73a3c2ae084fdfe90e7fb274e3ae36da6e619e1c4155372ed4"
SWEREX_RUNTIME_UPSTREAM_SHA256 = "d56f22b48b02d4df6644cfa04cc16c979b0498ef1c26058b82fb2d4a60bf9ac7"
SWEREX_RUNTIME_PATCH = "gbench/docker/sweagent_for_eval_swerex_heredoc.patch"
#: When the model submits, SWE-agent's PatchFormatter reads each file the model's patch modified to
#: record the edited regions (the trajectory's edited_files30/50/70 in the default config),
#: decoding strictly as UTF-8: a modified file with one byte that is not
#: UTF-8 raised UnicodeDecodeError and the instance was lost with no patch (2026-10-08:
#: privacyidea__privacyidea-4251, byte 0x81). A modified path need not be a regular file either:
#: a patch that moves a git submodule's commit lists the submodule's directory, whose read raised
#: IsADirectoryError and lost the instance the same way (2026-10-09: duckdb__ducklake-1224,
#: /testbed/duckdb). gbench/docker/sweagent_for_eval_review_decode.patch makes that read use
#: errors="replace" and first ask the task container whether the path exists and is not a
#: regular file (a directory, a FIFO, a device); such a path is not read and its entry in the
#: record is empty. The patch itself is untouched. The sha256 of the patched
#: sweagent/agent/agents.py is the image's `org.gbench.sweagent_for_eval_agents` label.
SWEAGENT_AGENTS_SHA256 = "5a56a297803679c850ee356554a175a45215226b9ca09e3717ee958be1eb4217"
SWEAGENT_AGENTS_PATCH = "gbench/docker/sweagent_for_eval_review_decode.patch"
SWEREX_RUNTIME_PINS = (
    "swe-rex==1.4.0", "annotated-doc==0.0.5", "annotated-types==0.8.0", "anyio==4.15.1",
    "bashlex==0.18", "certifi==2026.7.22", "charset-normalizer==3.5.2", "click==8.5.0",
    "fastapi==0.142.4", "h11==0.16.0", "idna==3.20", "markdown-it-py==4.2.0", "mdurl==0.1.2",
    "opentelemetry-api==1.45.1", "pexpect==4.9.0", "ptyprocess==0.7.0", "pydantic==2.13.5",
    "pydantic_core==2.46.5", "Pygments==2.21.0", "python-multipart==0.0.32", "requests==2.34.2",
    "rich==15.0.0", "starlette==1.7.0", "typing-inspection==0.4.4", "typing_extensions==4.16.0",
    "urllib3==2.8.0", "uvicorn==0.54.0",
)
#: gbench's watchdog over a SWE-agent batch (SweagentWatch). SWE-agent writes an instance's logs
#: and trajectory as it goes. Outside a model request the longest a live rollout stays silent is
#: one command (at most SWEAGENT_COMMAND_TIMEOUT_S): an instance with no file change for longer
#: than that (or EndpointRelay's class default limit), plus SWEAGENT_STALL_MARGIN_S, is stalled
#: (SWEAGENT_STALL_S, 6300 s). Only while its trace log shows a model request in flight
#: (sweagent_model_request_inflight) is the limit sweagent_stall_s of SWE-agent's client timeout
#: (the relay's safety cap on a model request plus 60 s, computed when the batch starts): the
#: relay ends a model request that stops making progress after its idle timeout (300 s) and
#: answers SWE-agent, which logs the retry, so that silence lasts only while a reply is still
#: being generated. 2026-10-11 review: one static limit sized for the longest model request
#: (34928 s at a 262144-token window) would have left a wedged command or runtime undetected
#: for 9.7 h. A batch with no file change for SWEAGENT_STALL_GRACE_S more than the limit of its
#: live instances is stopped.
SWEAGENT_STALL_MARGIN_S = 15 * 60
SWEAGENT_STALL_S = (max(SWEAGENT_COMMAND_TIMEOUT_S, EndpointRelay.TIMEOUT_S)
                    + SWEAGENT_STALL_MARGIN_S)
SWEAGENT_STALL_GRACE_S = 10 * 60
SWEAGENT_WATCH_POLL_S = 30


def sweagent_stall_s(request_timeout_s: float) -> int:
    """The watchdog's stall limit of an instance waiting for a model request that may take up
    to `request_timeout_s` (SWE-agent's client timeout): the longer of that and one command
    (SWEAGENT_COMMAND_TIMEOUT_S), plus SWEAGENT_STALL_MARGIN_S. The limit of any other silence
    is SWEAGENT_STALL_S."""
    return int(max(SWEAGENT_COMMAND_TIMEOUT_S, float(request_timeout_s))
               + SWEAGENT_STALL_MARGIN_S)


#: SWE-agent-for-eval's model client (SWEAGENT_IMAGE_DEFAULT, gbench/docker/
#: sweagent_for_eval.constraints.txt): LiteLLMModel._single_query passes the config's
#: agent.model.completion_kwargs to every litellm.completion call.
SWEAGENT_LITELLM_VERSION = "1.87.0"
#: The label gbench gives every task container of a SWE-agent batch (value: the batch's
#: container name), through run-batch's --instances.deployment.docker_args, so the task
#: containers SWE-ReX starts on the host daemon can be found and removed.
SWEAGENT_BATCH_LABEL = "org.gbench.sweagent_batch"
#: The harness error kind of an instance the watchdog found stalled.
AGENT_STALLED = "agent_stalled"
SWEAGENT_IMAGE_DEFAULT = "gbench-sweagent-for-eval:82422422-rex3"
_SWEAGENT_ROOT = "/opt/SWE-agent-for-eval"
#: The fork's model call (sweagent/agent/models.py LiteLLMModel._single_query) has its
#: temperature, top_p and api_version arguments commented out for every non-Azure model and sets
#: max_tokens only for an anthropic provider: no temperature, no max_tokens.
SWEAGENT_SAMPLING = OfficialSampling(
    reference=f"njukenanli/SWE-agent-for-eval@{SWEAGENT_COMMIT} sweagent/agent/models.py "
              "LiteLLMModel._single_query",
    send_temperature=False, send_max_tokens=False)
#: SWE-agent exit statuses that name a tenacity RetryError (exit_api), with or without an
#: autosubmitted patch. The fork's model calls re-raise their last error (reraise=True), so an
#: endpoint failure never ends as exit_api, and SWE-ReX rebuilds the exceptions the runtime in the
#: model's container reports: a harness error only when the tab's relay saw an endpoint failure
#: (as for an endpoint-named exit_error), else the model's outcome. See sweagent_exit_verdict.
SWEAGENT_INFRA_STATUSES = ("exit_api",)
#: The fork's forward_with_handling (sweagent/agent/agents.py) maps every exception it does not
#: name to `exit_error` ("Exit due to runtime error: ..." / "Exit due to unknown error: ...") and
#: a SwerexException to `exit_environment_error` ("Exit due to environment error: ..."),
#: autosubmitting whatever the repository holds. Both carry harness failures and the model's own:
#:   * the endpoint failing for good: LiteLLMModel.query (sweagent/agent/models.py) retries with
#:     tenacity `reraise=True`, so after its retries the last litellm exception reaches the
#:     catch-all as exit_error with "litellm.<Class>: ..." (never a RetryError);
#:   * the model ending its own shell or container: `kill -9 $$` makes SWE-ReX's BashSession
#:     hit pexpect's EOF in the task container; the client cannot import pexpect (the image's
#:     venv has none), so it re-raises a SwerexException -> exit_environment_error ("End Of File
#:     (EOF) ..."); a killed runtime server answers no more requests (an aiohttp error ->
#:     exit_error); a command that hijacks the shell's output raises NoExitCodeError, and one
#:     that breaks SWE-agent's state or tool commands NonZeroExitCodeError (both SwerexException);
#:   * a prompt the served context window refused with a message the fork does not recognise as
#:     a context overflow (its own check is "is longer than the model's context length"; vLLM
#:     also says "... is longer than the maximum model length ..."): exit_context's outcome.
#: The runtime that answers SWE-agent runs inside the task container, where the model's commands
#: run as root, and SWE-ReX rebuilds the exception it reports from the class path and message
#: the runtime sends (swerex/runtime/remote.py _handle_transfer_exception), so the text of either
#: status can be the model's own. A rollout ending with one of them is a harness error only for
#: an exit_error whose exception text starts with a model-endpoint failure while the tab's relay
#: saw an endpoint failure (sweagent_exit_verdict); anything else is the model's outcome, as
#: upstream counts it, with a note on the trace.
SWEAGENT_ERROR_STATUSES = ("exit_error", "exit_environment_error")
SWEAGENT_ERROR_STATUS = "exit_error"
#: The prefix forward_with_handling puts before the exception text in the error step's thought.
_SWEAGENT_EXIT_PREFIX = re.compile(r"^Exit due to (?:unknown|runtime|environment) error: ")
#: The image SWE-ReX's DockerDeployment builds each instance's runtime layer from
#: (swerex/deployment/docker.py glibc_dockerfile, `FROM python:3.11.9-slim-bookworm AS builder`,
#: the swerex 1.4.0 the fork vendors); gbench pulls it on the host with the instance images.
SWEREX_BUILDER_BASE = "python:3.11.9-slim-bookworm"
_DOCKER_SOCKET = "/var/run/docker.sock"
#: The Windows tab's instance images (the `docker_image` of every SWE-bench-Live/Windows row,
#: Windows containers with PowerShell, which Win-Agent and RepoLaunch drive): the images the
#: Windows metadata server probe may run in (check_windows_metadata_guard).
WINDOWS_IMAGE_REFERENCE = "starryzhang/sweb.eval.win.*"
#: The hosts SWE-ReX's runtime layer build fetches from for each instance image (the vendored
#: swerex deployment/docker.py glibc_dockerfile: apt in python:3.11.9-slim-bookworm and in the
#: instance image, Python-3.11.8.tgz from www.python.org, the pinned swe-rex from PyPI). The
#: gate checks that they resolve before the agentic Linux tabs start.
SWEREX_BUILD_HOSTS = ("deb.debian.org", "archive.ubuntu.com", "www.python.org", "pypi.org",
                      "files.pythonhosted.org")

#: Fields of a setting's result that describe that setting alone: they stay under
#: `settings.<name>` and are not copied to the top level of the combined result.
_PER_SETTING_FIELDS = (
    "dataset", "total_questions", "total_instance_ids", "correct_answers", "failed_requests",
    "accuracy", "category_accuracy", "status", "error", "leaderboard_comparable",
    "leaderboard_comparable_reason", "partial_evaluation", "swebench_report",
    "swebench_workdir", "harness_report_retries", "dataset_repeated_instance_ids",
    "dataset_repeated_instance_ids_note", "agent_output_dir", "agent_exit_statuses",
    "agent_steps_total", "truncated_turns", "context_retry_turns", "duration_s", "shard",
    "sample_traces", "empty_responses", "truncated_responses", "non_convergent_responses",
    "genuinely_truncated", "over_context_prompts", "timed_out_requests", "looping_responses",
    "discarded_outputs", "response_health", "attempts", "judge_failures", "judge_outages",
    "judge_outage_rate", "scoring_excluded", "max_tokens_sent", "decode_tok_s",
    "harness_report", "harness_workdir", "split_accuracy", "leaderboard_totals",
    "leaderboard_totals_note", "harness", "relay", "official_sampling", "tab",
    "empty_after_reasoning_turns", "model_replies", "endpoint_failures", "image_pulls",
    "harness_image_pulls", "infra_required", "scored_instances", "harness_errors",
    "harness_error_ids", "accuracy_withheld_reason", "status_reason", "effective_n",
    "exits_kept_as_model_outcome", "exits_kept_as_model_outcome_ids", "agent_watchdog",
    "answer_in_thinking", "answer_in_thinking_of", "answer_in_thinking_rate",
    "tool_call_dropped_turns", "unclassified_empty_turns",
)


def _image() -> str:
    return os.environ.get("GBENCH_SWE_BENCH_LIVE_IMAGE", _IMAGE_DEFAULT)


def setting_family(setting: str) -> str:
    """'python', 'multilang' or 'windows'."""
    if setting in PYTHON_SETTINGS:
        return "python"
    if setting in MULTILANG_SETTINGS:
        return "multilang"
    return "windows"


def setting_splits(setting: str) -> List[Dict[str, Any]]:
    """The pinned splits behind a setting: dataset, revision, split, rows, sha256, columns."""
    family = setting_family(setting)
    if family == "python":
        return [{"dataset": _DATASET, "revision": SETTING_REVISION[setting], "split": setting,
                 "rows": SPLIT_ROWS[setting], "sha256": SPLIT_SHA256[setting],
                 "columns": SNAPSHOT_COLUMNS}]
    if family == "multilang":
        return [{"dataset": MULTILANG_DATASET, "revision": MULTILANG_REVISION, "split": s,
                 "rows": MULTILANG_SPLITS[s][0], "sha256": MULTILANG_SPLITS[s][1],
                 "columns": LAUNCH_COLUMNS} for s in MULTILANG_TAB_SPLITS[setting]]
    return [{"dataset": WINDOWS_DATASET, "revision": WINDOWS_REVISION, "split": s,
             "rows": n, "sha256": sha, "columns": LAUNCH_COLUMNS + ("language",)}
            for s, (n, sha) in WINDOWS_SPLITS.items()]


def setting_rows(setting: str) -> int:
    return sum(s["rows"] for s in setting_splits(setting))


def setting_dataset_label(setting: str) -> str:
    parts = setting_splits(setting)
    return (f"{parts[0]['dataset']}@{parts[0]['revision']} "
            f"({' + '.join(p['split'] for p in parts)})")


def resolve_settings(split: Optional[str] = None,
                     eval_categories: Optional[str] = None) -> List[str]:
    """The settings a run evaluates, in leaderboard order: `split` alone when given, else the
    `--eval-categories` names among SETTINGS, else all of them. An unknown name is an error."""
    if split:
        names = [split]
    elif eval_categories:
        names = [c.strip() for c in str(eval_categories).split(",") if c.strip()]
    else:
        return list(SETTINGS)
    unknown = [n for n in names if n not in SETTINGS]
    if unknown:
        raise ValueError(f"swe_bench_live: {unknown} are not leaderboard settings; use "
                         f"{', '.join(SETTINGS)}")
    return [s for s in SETTINGS if s in names]


# --------------------------------------------------------------------------------------------
# prerequisites
# --------------------------------------------------------------------------------------------
def _build_hint() -> str:
    return (f"Build the image LOCALLY (gbench never pulls it):\n"
            f"  docker build -t {_image()} -f gbench/docker/swe_bench_live.Dockerfile "
            f"gbench/docker\nIt isolates the SWE-bench-Live fork of `swebench` from the main "
            f"env's upstream `swebench` and carries upstream's multi-language harness. See "
            + DOCS_URL)


def _linux_tabs_hint() -> str:
    """On a Windows host: why a Linux tab's prerequisite is missing there."""
    if _platform.system() != "Windows":
        return ""
    return (" This is a Windows host: the Python and MultiLang tabs run Linux images (the "
            "harness, SWE-agent-for-eval and the instance images) and need a Docker engine "
            "running Linux containers; run them on a Linux Docker host.")


def _check_image(families: Sequence[str]) -> Tuple[bool, str]:
    """The local image carries the harness each selected family needs (its labels)."""
    build = _build_hint()
    inspect = subprocess.run(["docker", "image", "inspect", "--format",
                              "{{json .Config.Labels}}", _image()],
                             capture_output=True, text=True)
    if inspect.returncode != 0:
        return False, (f"the swe_bench_live harness image {_image()!r} is not built. " + build
                       + _linux_tabs_hint())
    try:
        labels = json.loads(getattr(inspect, "stdout", "") or "null") or {}
    except ValueError:
        labels = {}
    if "python" in families and labels.get("org.gbench.swe_bench_live_fork") != FORK_COMMIT:
        return False, (f"the image {_image()!r} was built from SWE-bench-Live fork "
                       f"{labels.get('org.gbench.swe_bench_live_fork')!r}, not {FORK_COMMIT}; "
                       "rebuild it. " + build)
    if "multilang" in families:
        for label, want in (("org.gbench.swe_bench_live_main", MAIN_COMMIT),
                            ("org.gbench.repolaunch", REPOLAUNCH_COMMIT)):
            if labels.get(label) != want:
                return False, (f"the image {_image()!r} carries {label}={labels.get(label)!r}, "
                               f"not {want} (upstream's multi-language harness); rebuild it. "
                               + build)
    return True, ""


def _check_launch_harness(rows_path: str) -> Tuple[bool, str]:
    """Inside the image, network off: upstream's multi-language harness imports, and every
    row's log parser compiles (the harness exec()s it on the test log)."""
    snap_dir = os.path.dirname(os.path.abspath(rows_path))
    probe = ("import json, sys\n"
             "import evaluation.evaluation\n"
             "from launch.core.runtime import SetupRuntime\n"
             "from launch.scripts.parser import run_parser\n"
             "rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]\n"
             "[compile(r['log_parser'], r['instance_id'], 'exec') for r in rows]\n"
             "print(len(rows))\n")
    try:
        r = subprocess.run(["docker", "run", "--rm", "--network", "none",
                            "-e", f"PYTHONPATH={MAIN_PYTHONPATH}",
                            "-v", f"{snap_dir}:{snap_dir}:ro", _image(), "python", "-c", probe,
                            os.path.abspath(rows_path)],
                           capture_output=True, text=True, timeout=600)
    except Exception as e:
        return False, f"the multi-language harness probe could not run in {_image()!r}: {e}"
    if r.returncode != 0:
        return False, (f"upstream's multi-language harness in {_image()!r} cannot read "
                       f"{rows_path}: {(r.stderr or r.stdout)[-300:]}")
    return True, ""


def _git_head(path: str) -> Optional[str]:
    try:
        r = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                           text=True, timeout=60)
    except Exception:
        return None
    return (r.stdout or "").strip() if r.returncode == 0 else None


def win_agent_dir() -> str:
    return (os.environ.get("GBENCH_SWE_BENCH_LIVE_WIN_AGENT_DIR") or "").strip()


def windows_harness_dir() -> str:
    return (os.environ.get("GBENCH_SWE_BENCH_LIVE_HARNESS_DIR") or "").strip()


def check_windows_prerequisites() -> Tuple[bool, str]:
    """The Windows tab: a Windows host whose Docker engine runs Windows containers, and
    checkouts of Win-Agent and of SWE-bench-Live (with its RepoLaunch submodule) at the pinned
    commits. Win-Agent and upstream's harness both bind-mount a directory of the host they run
    on into each task container (Win-Agent ./mnt -> C:\\mnt, RepoLaunch ./tmp -> C:\\mnt_tmp),
    so they must run on the Windows host itself."""
    where = ("The SWE-bench-Live Windows tab needs a Windows host whose Docker engine runs "
             "Windows containers (Docker Desktop switched to Windows containers, or Docker "
             "Engine on Windows Server), with gbench running on that host: its instance images "
             "(starryzhang/sweb.eval.win.*) are Windows containers, and Win-Agent and "
             "upstream's harness bind-mount a directory of the host they run on into each task "
             "container. ")
    if _platform.system() != "Windows":
        return False, where + (f"This host runs {_platform.system() or 'an unknown OS'}. Run the "
                               "other settings here with --eval-categories (e.g. "
                               f"{','.join(PYTHON_SETTINGS + MULTILANG_SETTINGS)}).")
    try:
        r = subprocess.run(["docker", "version", "--format", "{{.Server.Os}}"],
                           capture_output=True, text=True, timeout=60)
    except Exception as e:
        return False, where + f"`docker version` failed: {e}"
    if r.returncode != 0 or (r.stdout or "").strip().lower() != "windows":
        return False, where + (f"The Docker engine reports server OS "
                               f"{(r.stdout or '').strip() or 'unknown'!r}; switch it to "
                               "Windows containers.")
    agent = win_agent_dir()
    if not agent or not os.path.isfile(os.path.join(agent, "main.py")):
        return False, (f"set GBENCH_SWE_BENCH_LIVE_WIN_AGENT_DIR to a checkout of "
                       f"github.com/njukenanli/Win-Agent at {WIN_AGENT_COMMIT}.")
    if _git_head(agent) != WIN_AGENT_COMMIT:
        return False, (f"the Win-Agent checkout {agent} is at {_git_head(agent)!r}, not "
                       f"{WIN_AGENT_COMMIT}; check out that commit.")
    harness = windows_harness_dir()
    if not harness or not os.path.isfile(os.path.join(harness, "evaluation", "evaluation.py")):
        return False, (f"set GBENCH_SWE_BENCH_LIVE_HARNESS_DIR to a checkout of "
                       f"github.com/microsoft/SWE-bench-Live at {MAIN_COMMIT} with its launch "
                       "submodule (git submodule update --init).")
    if _git_head(harness) != MAIN_COMMIT:
        return False, (f"the SWE-bench-Live checkout {harness} is at {_git_head(harness)!r}, "
                       f"not {MAIN_COMMIT}; check out that commit.")
    if _git_head(os.path.join(harness, "launch")) != REPOLAUNCH_COMMIT:
        return False, (f"the launch submodule of {harness} is not at {REPOLAUNCH_COMMIT}; run "
                       "`git submodule update --init` in that checkout.")
    for module in ("litellm", "docker", "yaml", "datasets"):
        try:
            __import__(module)
        except ImportError:
            return False, (f"Python package {module!r} (Win-Agent requirements.txt / "
                           "upstream's harness) is not installed (pip install -e \".[evals]\" "
                           "from the gbench checkout).")
    return True, ""


def _windows_instance_images() -> List[str]:
    """The Windows tab's instance images in the pinned snapshot's order, each once ([] when the
    snapshot cannot be read)."""
    try:
        rows = load_setting_rows("windows")
    except Exception as e:  # the reason reaches the operator through the check's own message
        logger.warning("swe_bench_live: the Windows snapshot could not be read for the metadata "
                       "server probe (%s: %s)", type(e).__name__, e)
        return []
    return list(dict.fromkeys(str(r["docker_image"]) for r in rows if r.get("docker_image")))


def check_windows_metadata_guard() -> Tuple[bool, str]:
    """The cloud metadata server check for the Windows tab's task containers: Win-Agent's
    rollouts and the harness's tests run model-controlled code there with network access.
    (True, "") only when swebench_common.check_windows_task_network established on this
    Windows-containers engine (`docker info` OSType windows) that 169.254.169.254:80 and
    [fd20:ce::254]:80 are unreachable from docker's default network, probing with PowerShell in
    one of the tab's own instance images already on the host (the snapshot's `docker_image`, else
    any local WINDOWS_IMAGE_REFERENCE image; nothing is pulled). Otherwise (False, the reason): a
    Linux engine, no such image (the reason names the `docker pull` of the first instance's
    image), a probe without a clear verdict, or a reachable metadata server."""
    return swebench_common.check_windows_task_network(
        "swe_bench_live", _windows_instance_images(), references=[WINDOWS_IMAGE_REFERENCE],
        hint=f"Win-Agent and the harness pull the others themselves (see {DOCS_URL}).")


def check_swe_bench_live_prerequisites(
        settings: Optional[Sequence[str]] = None) -> Tuple[bool, str]:
    """datasets + Docker; the LOCAL image with the harness each selected family needs (the
    fork at FORK_COMMIT for the Python tabs, upstream's multi-language harness at MAIN_COMMIT +
    REPOLAUNCH_COMMIT for the MultiLang tabs); for the Windows tab, a Windows host
    (check_windows_prerequisites); and the pinned snapshot of every selected setting, fetched
    once and checked (pinned_swe_snapshot). `settings` defaults to every setting."""
    settings = list(SETTINGS) if settings is None else list(settings)
    families = [f for f in ("python", "multilang", "windows")
                if any(setting_family(s) == f for s in settings)]
    if "windows" in families:
        # First: off a Windows host nothing else matters for this tab.
        ok, why = check_windows_prerequisites()
        if not ok:
            return False, why
    try:
        import datasets  # noqa: F401
    except ImportError:
        return False, ("Python package 'datasets' is not installed (pip install -e \".[evals]\" "
                       "from the gbench checkout).")
    if not shutil.which("docker"):
        return False, "docker CLI not found. " + _build_hint()
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return False, "docker daemon not reachable. " + _build_hint()
    linux = [f for f in families if f != "windows"]
    if linux:
        ok, why = _check_image(linux)
        if not ok:
            return False, why
    for setting in settings:
        paths = []
        for part in setting_splits(setting):
            try:
                paths.append(pinned_swe_snapshot(
                    part["dataset"], part["split"], part["revision"], "swe_bench_live",
                    columns=part["columns"], rows=part["rows"], sha256=part["sha256"]))
            except RuntimeError as e:
                return False, str(e)
        if setting_family(setting) == "multilang":
            for path in paths:
                ok, why = _check_launch_harness(path)
                if not ok:
                    return False, why
    return True, ""


def _load_swe_bench_live_samples(limit: Optional[int] = None, split: str = _DEFAULT_SPLIT):
    return load_swe_samples(_DATASET, split, limit, "swe_bench_live",
                            revision=SETTING_REVISION[split], columns=SNAPSHOT_COLUMNS,
                            rows=SPLIT_ROWS[split], sha256=SPLIT_SHA256[split])


def load_setting_rows(setting: str) -> List[Dict[str, Any]]:
    """Every row of a setting, split by split in its pinned order, each marked with its
    `split` under the key `_split` (not written back to any snapshot)."""
    out: List[Dict[str, Any]] = []
    for part in setting_splits(setting):
        rows = _pinned_snapshot_rows(part["dataset"], part["split"], part["revision"],
                                     "swe_bench_live", part["columns"], part["rows"],
                                     part["sha256"])[1]
        out.extend({**r, "_split": part["split"]} for r in rows)
    return out


# --------------------------------------------------------------------------------------------
# upstream's multi-language harness (evaluation/evaluation.py)
# --------------------------------------------------------------------------------------------
def _harness_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The rows as the harness reads them (the published columns only)."""
    return [{k: v for k, v in r.items() if k != "_split"} for r in rows]


def launch_harness_command(workdir: str, rows_path: str, preds_path: str, out_dir: str,
                           workers: int, threads: int, platform: str) -> Tuple[List[str],
                                                                                Dict[str, str]]:
    """(argv, extra environment) of one scoring pass of upstream's multi-language harness:
    `python -m evaluation.evaluation --dataset <rows> --patch_dir <preds> --platform <p>
    --workers N --output_dir <dir> --overwrite 1`, run with the working directory at
    `workdir` (RepoLaunch bind-mounts `<cwd>/tmp` into each task container).

    Linux: inside the local image, docker-out-of-docker, `workdir` identity-mounted so the
    bind-mount source exists on the host daemon, as the caller's uid, with the thread-cap shim
    (swe_thread_cap) as for the fork harness. Windows: on this host, from the pinned checkout
    (GBENCH_SWE_BENCH_LIVE_HARNESS_DIR), with PYTHONUTF8 as upstream's README asks."""
    harness = ["python", "-m", "evaluation.evaluation", "--dataset", rows_path,
               "--patch_dir", preds_path, "--platform", platform,
               "--workers", str(max(1, workers)), "--output_dir", out_dir, "--overwrite", "1"]
    if platform == "windows":
        tree = windows_harness_dir()
        env = {"PYTHONPATH": os.pathsep.join([os.path.join(tree, "launch"), tree]),
               "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        return [sys.executable, *harness[1:]], env
    cmd = ["docker", "run", "--rm", "--network", "host",
           "-v", "/var/run/docker.sock:/var/run/docker.sock",
           "-v", f"{workdir}:{workdir}:rw", "-w", workdir,
           "-e", f"PYTHONPATH={MAIN_PYTHONPATH}", "-e", "HOME=/tmp", "-e", "PYTHONUNBUFFERED=1"]
    if hasattr(os, "getuid"):
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        try:
            cmd += ["--group-add", str(os.stat("/var/run/docker.sock").st_gid)]
        except OSError:
            pass
    if threads and threads > 0:
        shim = os.path.abspath(swe_thread_cap.__file__)
        cmd += ["-v", f"{shim}:{shim}:ro"]
        for var in swe_thread_cap.THREAD_VARS:
            cmd += ["-e", f"{var}={threads}"]
        harness = swe_thread_cap.wrap_command(harness, threads)
    return cmd + [_image(), *harness], {}


def make_launch_scorer(setting: str, rows: Sequence[Dict[str, Any]], platform: str,
                       workers: int, metrics: Dict[str, Any]):
    """Async scorer: upstream's multi-language harness over the patches of `rows`' instances;
    an instance resolves iff it is in the harness's `success_ids` (evaluation.py run_instance:
    no PASS_TO_PASS failure, no FAIL_TO_PASS failure, every FAIL_TO_PASS test passed). On Linux
    each submitted instance's `docker_image` is pulled on the host first (host_pull_images,
    recorded in `harness_image_pulls`): the harness image has no Docker client configuration.
    Before that, the cloud metadata server check of the task containers runs
    (swebench_common.require_task_network_blocked, probing with the harness image)."""
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        workdir = tempfile.mkdtemp(prefix="gbench_swe_live_launch_")
        metrics["harness_workdir"] = workdir
        rows_path = os.path.join(workdir, "instances.jsonl")
        with open(rows_path, "w", encoding="utf-8") as f:
            for r in _harness_rows(rows):
                f.write(json.dumps(r, default=str) + "\n")
        preds = {}
        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("instance_id")
            if iid:
                preds[iid] = {"model_patch": trace_patch(tr)}
        preds_path = os.path.join(workdir, "preds.json")
        with open(preds_path, "w", encoding="utf-8") as f:
            json.dump(preds, f)
        out_dir = os.path.join(workdir, "eval")
        if platform == "linux" and preds:
            # RepoLaunch runs the tests over each patch in the instance's task container, with
            # network access: the cloud metadata server must be blocked there first.
            await asyncio.to_thread(
                swebench_common.require_task_network_blocked, "swe_bench_live", [_image()],
                docs_url=DOCS_URL, hint=f"Build the harness image {_image()} (see {DOCS_URL}).")
            images = {r["instance_id"]: r.get("docker_image") for r in rows}
            pulls = await asyncio.to_thread(
                host_pull_images, [images.get(i) for i in preds if images.get(i)], workers)
            metrics["harness_image_pulls"] = image_pull_summary(pulls)
        threads = swe_thread_cap.resolve_threads(workers) if platform == "linux" else 0
        cmd, extra_env = launch_harness_command(workdir, rows_path, preds_path, out_dir,
                                                workers, threads, platform)
        metrics["harness"] = {
            "tree": f"microsoft/SWE-bench-Live@{MAIN_COMMIT} evaluation/evaluation.py",
            "repolaunch": f"microsoft/RepoLaunch@{REPOLAUNCH_COMMIT}",
            "platform": platform,
            "runs_in": (f"local image {_image()} (docker-out-of-docker)" if platform == "linux"
                        else f"this host ({windows_harness_dir()})"),
        }
        if platform == "linux":
            metrics["harness_image"] = _image()
            metrics["docker_thread_cap"] = threads or None
        # No credential variable reaches the harness (it needs none).
        env = credential_free_env()
        env.update(extra_env)
        cwd = workdir

        def _run():
            return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
        proc = await asyncio.to_thread(_run)
        log_path = os.path.join(workdir, "harness_output.txt")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write((proc.stdout or "") + "\n----- STDERR -----\n" + (proc.stderr or ""))
        except OSError:
            pass
        report_path = os.path.join(out_dir, "results.json")
        resolved: set = set()
        verdictless: set = set()
        report_ok = os.path.isfile(report_path)
        if report_ok:
            with open(report_path, encoding="utf-8") as f:
                rep = json.load(f)
            resolved = set(rep.get("success_ids") or [])
            verdictless = set(rep.get("error_ids") or []) | set(rep.get("incomplete_ids") or [])
            metrics["harness_report"] = {k: rep.get(k) for k in (
                "submitted", "success", "failure", "empty_patch", "error", "incomplete",
                "success_ids", "failure_ids", "empty_patch_ids", "error_ids",
                "incomplete_ids")}
        else:
            logger.error("swe_bench_live %s: the harness wrote no results.json (rc=%s); its "
                         "output is in %s", setting, proc.returncode, log_path)
            # `report_error`, not `error`: results.json's own `error` is a count.
            metrics["harness_report"] = {"report_error": "harness report not produced",
                                         "harness_output_file": log_path}
        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("instance_id")
            tr["is_correct"] = iid in resolved
            tr["status"] = "OK"
            if not report_ok and (preds.get(iid) or {}).get("model_patch", "").strip():
                # No report, so no verdict for any submitted patch (an empty patch is the
                # model's outcome, unresolved, with or without the harness).
                mark_harness_error(tr, NO_REPORT, "the multi-language harness wrote no "
                                   "results.json", status="HARNESS_ERROR")
            elif iid in verdictless:
                # error_ids / incomplete_ids: the harness raised or never finished the instance.
                mark_harness_error(tr, NO_VERDICT, "the multi-language harness gave this "
                                   "instance no verdict (error_ids / incomplete_ids)",
                                   status="HARNESS_ERROR")
    return _score


def launch_report_reasons(result: Dict[str, Any], metrics: Dict[str, Any]) -> List[str]:
    """The harness-report checks for upstream's multi-language harness. No results.json is an
    error. An instance in `error_ids` raised inside the harness (container start, image pull,
    a patch or parser step that crashed): a harness error, not a model verdict (the scorer left
    it out of the score), which makes the setting partial, or an error when nothing was scored.
    `failure_ids` (tests ran and failed, including a patch that did not apply) and
    `empty_patch_ids` are the model's outcome."""
    report = metrics.get("harness_report")
    if not isinstance(report, dict):
        return []
    if report.get("report_error"):
        result["status"] = "error"
        result["error"] = report["report_error"]
        return ["harness error (no valid resolved rate)"]
    errors = list(report.get("error_ids") or []) + list(report.get("incomplete_ids") or [])
    scored = (len(report.get("success_ids") or []) + len(report.get("failure_ids") or [])
              + len(report.get("empty_patch_ids") or []))
    submitted = report.get("submitted") or 0
    if not errors:
        return []
    if scored == 0:
        result["status"] = "error"
        result["error"] = (f"the harness could not score any of {submitted} submitted "
                           f"instance(s) ({len(errors)} harness errors); this is an "
                           "infrastructure failure, not a 0% resolved rate")
        return ["harness infrastructure failure (no scoreable instances)"]
    partial = result.setdefault("partial_evaluation", {})
    partial.update({"harness_scoreable_instances": scored,
                    "harness_submitted_instances": submitted,
                    "harness_error_instances": len(errors)})
    if result.get("status") == "success":
        result["status"] = "completed_with_errors"
    return [f"partial: {len(errors)} of {submitted} submitted instance(s) hit harness errors "
            "(not model-scoreable)"]


def _oneshot_harness_errors(result: Dict[str, Any]) -> None:
    """A one-shot tab's harness errors by kind (the scorer's, and base's endpoint failures), all
    left out of the score; an error tab reports no score."""
    record_harness_errors(result, scored_headline(result.get("sample_traces") or []))
    withhold_on_error(result)


def _split_accuracy(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Per-split resolved counts of a tab built from more than one split (C/C++, TS/JS)."""
    counts: Dict[str, Dict[str, Any]] = {}
    for tr in result.get("sample_traces") or []:
        split = (tr.get("extra_payload") or {}).get("split")
        if not split or tr.get("scoring_excluded"):      # a harness error is not scored
            continue
        st = counts.setdefault(split, {"correct": 0, "total": 0})
        st["total"] += 1
        st["correct"] += bool(tr.get("is_correct"))
    if len(counts) < 2:
        return None
    for st in counts.values():
        st["accuracy"] = round(st["correct"] / st["total"] * 100.0, 2) if st["total"] else 0.0
    return counts


def _leaderboard_totals(setting: str, result: Dict[str, Any]) -> List[str]:
    """Record the board's totals for the tab; a reason when no board row has this run's."""
    totals = LEADERBOARD_TOTALS.get(setting, ())
    result["leaderboard_totals"] = list(totals)
    n = setting_rows(setting)
    if n in totals:
        return []
    result["leaderboard_totals_note"] = (
        f"the {TAB_LABELS[setting]} tab's rows were scored on {', '.join(map(str, totals))} "
        f"instances (earlier snapshots of a growing dataset); this run's pinned snapshot has "
        f"{n}")
    return [f"no {TAB_LABELS[setting]} leaderboard row scores the pinned {n} instances "
            f"(board totals {', '.join(map(str, totals))})"]


# --------------------------------------------------------------------------------------------
# the Python and MultiLang tabs' scaffold: SWE-agent-for-eval
# --------------------------------------------------------------------------------------------
def sweagent_image() -> str:
    return (os.environ.get("GBENCH_SWE_BENCH_LIVE_AGENT_IMAGE") or "").strip() or \
        SWEAGENT_IMAGE_DEFAULT


def check_sweagent_prerequisites() -> Tuple[bool, str]:
    """The docker socket SWE-agent starts each instance image through, and the LOCAL image with
    SWE-agent-for-eval at SWEAGENT_COMMIT and its config (the image's labels)."""
    image = sweagent_image()
    build = (f"Build it LOCALLY (gbench never pulls it):\n  docker build -t {image} -f "
             "gbench/docker/sweagent_for_eval.Dockerfile gbench/docker\nSee " + DOCS_URL)
    if not os.path.exists(_DOCKER_SOCKET):
        return False, (f"{_DOCKER_SOCKET} does not exist; SWE-agent starts each instance image "
                       "on the host daemon through it. " + build + _linux_tabs_hint())
    try:
        r = subprocess.run(["docker", "image", "inspect", "--format",
                            "{{json .Config.Labels}}", image],
                           capture_output=True, text=True, timeout=60)
    except Exception as e:
        return False, f"cannot inspect the SWE-agent image {image!r}: {e}"
    if r.returncode != 0:
        return False, (f"the SWE-agent-for-eval image {image!r} is not built. " + build
                       + _linux_tabs_hint())
    try:
        labels = json.loads(getattr(r, "stdout", "") or "null") or {}
    except ValueError:
        labels = {}
    for label, want in (("org.gbench.sweagent_for_eval", SWEAGENT_COMMIT),
                        ("org.gbench.sweagent_for_eval_config_sha256", SWEAGENT_CONFIG_SHA256),
                        ("org.gbench.sweagent_for_eval_swerex_runtime", SWEREX_RUNTIME_SHA256),
                        ("org.gbench.sweagent_for_eval_agents", SWEAGENT_AGENTS_SHA256)):
        if labels.get(label) != want:
            why = ""
            if label == "org.gbench.sweagent_for_eval_swerex_runtime":
                why = (" It was built without gbench's fix of the task containers' runtime "
                       f"({SWEREX_RUNTIME_PATCH}): its runtime loses the body of a heredoc in the "
                       "model's command, and the shell waits until the 90-minute command limit.")
            elif label == "org.gbench.sweagent_for_eval_agents":
                why = (" It was built without gbench's fix of SWE-agent's read of the edited files "
                       f"({SWEAGENT_AGENTS_PATCH}): a modified file with a byte that is not UTF-8, "
                       "or a modified path that is a directory (a git submodule), ends the "
                       "instance with no patch.")
            return False, (f"the image {image!r} carries {label}={labels.get(label)!r}, not "
                           f"{want}; rebuild it.{why} " + build)
    return True, ""


def python_instance_image(instance_id: str) -> str:
    """The image upstream's harness gives a Python-dataset instance (evaluation.py
    get_default_image_name; the `starryzhang` images the fork harness runs)."""
    return "starryzhang/sweb.eval.x86_64." + instance_id.replace("__", "_1776_").lower()


def sweagent_instance(row: Dict[str, Any]) -> Dict[str, Any]:
    """One line of the instance file SWE-agent-for-eval reads (`--instances.type swe_bench`,
    SimpleBatchInstance.from_swe_bench): the problem statement and the instance image, which is
    all SWE-bench-Live's rollout rule lets the agent see, plus the instance id and the base
    commit that loader requires (it records the commit and resets nothing). The image is the
    row's `docker_image` (MultiLang), else the name upstream's harness gives the instance
    (evaluation.py get_default_image_name: the `starryzhang` images the fork harness uses)."""
    iid = row["instance_id"]
    image = row.get("docker_image") or python_instance_image(iid)
    return {"instance_id": iid, "problem_statement": row["problem_statement"],
            "base_commit": row["base_commit"], "image_name": image}


def sweagent_command(instances_path: str, out_dir: str, model_name: str, relay_url: str,
                     workers: int, name: str,
                     client_timeout_s: Optional[float] = None) -> List[str]:
    """`docker run` of one SWE-agent-for-eval batch: the fork's README rollout command
    (`sweagent run-batch --config config/default.yaml --num_workers N --instances.type swe_bench
    --instances.subset <file>`) with the served model (`openai/<model>` at the relay's address)
    and the output directory. Host network (the relay on 127.0.0.1 and the port each task
    container publishes), the docker socket (SWE-agent's DockerDeployment builds and starts the
    instance images on the host daemon), the output directory at the same path, the caller's
    uid. The container gets OPENAI_API_KEY=EMPTY: the relay adds the endpoint's key.

    Two additions change nothing SWE-agent does: `--init` (an init process as PID 1 passes on
    the SIGTERM the docker CLI forwards, which SWE-agent as PID 1 would ignore, so the batch
    ends with the process that started it), and the label SWEAGENT_BATCH_LABEL=<name> on every
    task container (--instances.deployment.docker_args), so gbench can remove the task
    containers of a stopped or finished batch (remove_batch_task_containers).

    With `client_timeout_s` (run_sweagent always passes it: swe_agentic.
    client_timeout_for_relay_limit, the relay's safety cap on a model request plus
    swe_agentic.AGENT_CLIENT_MARGIN_S), one transport setting:
    `--agent.model.completion_kwargs {"timeout": T, "max_retries": 0}`, which
    LiteLLMModel._single_query passes to every litellm.completion call. Without it litellm 1.87
    waited 600 s for a reply and the OpenAI SDK sent it twice more (2026-10-10 smoke run: 12
    resamples of one trimesh instance's turns); with max_retries 0 a request is one HTTP attempt
    and SWE-agent's own retry loop (RetryConfig, 20 attempts) handles an error the relay
    reports. It changes nothing the model is sent or allowed."""
    transport: List[str] = []
    if client_timeout_s:
        transport = ["--agent.model.completion_kwargs", json.dumps(
            {"timeout": int(client_timeout_s), "max_retries": swe_agentic.AGENT_CLIENT_RETRIES})]
    cmd = ["docker", "run", "--rm", "--name", name, "--init", "--network", "host",
           "-v", f"{_DOCKER_SOCKET}:{_DOCKER_SOCKET}", "-v", f"{out_dir}:{out_dir}:rw",
           "-w", out_dir, "-e", "HOME=/tmp", "-e", "OPENAI_API_KEY=EMPTY"]
    if hasattr(os, "getuid"):
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        try:
            cmd += ["--group-add", str(os.stat(_DOCKER_SOCKET).st_gid)]
        except OSError:
            pass
    return cmd + [sweagent_image(), f"{_SWEAGENT_ROOT}/venv/bin/sweagent", "run-batch",
                  "--config", f"{_SWEAGENT_ROOT}/{SWEAGENT_CONFIG}",
                  "--agent.model.name", f"openai/{model_name}",
                  "--agent.model.api_base", relay_url,
                  *transport,
                  "--instances.type", "swe_bench", "--instances.subset", instances_path,
                  f"--instances.deployment.docker_args=--label={SWEAGENT_BATCH_LABEL}={name}",
                  "--num_workers", str(max(1, int(workers or 1))),
                  "--output_dir", out_dir, "--progress_bar", "False"]


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _sweagent_exit_statuses(out_dir: str) -> Dict[str, str]:
    """instance id -> exit status from run-batch's run_batch_exit_statuses.yaml, which also
    names the instances whose run raised ("Uncaught <Exception>")."""
    try:
        import yaml
        with open(os.path.join(out_dir, "run_batch_exit_statuses.yaml"), encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return {}
    out: Dict[str, str] = {}
    for status, ids in (data.get("instances_by_exit_status") or {}).items():
        for iid in ids or []:
            out[str(iid)] = str(status)
    return out


def _sweagent_exit_message(traj: Dict[str, Any]) -> str:
    """The message SWE-agent wrote for the rollout's last step (for an error exit,
    handle_error_with_autosubmission puts "Exit due to ...: <exception>" in its thought and
    response)."""
    steps = traj.get("trajectory") or []
    last = steps[-1] if steps and isinstance(steps[-1], dict) else {}
    parts = [str(last.get(k)) for k in ("thought", "response", "observation") if last.get(k)]
    return " | ".join(dict.fromkeys(parts))


def sweagent_exit_verdict(status: str, message: str, relay_failures: Optional[int] = None
                          ) -> Tuple[Optional[str], Optional[str]]:
    """(harness error reason, model-outcome note) of a finished SWE-agent rollout's exit status
    and its last step's message (_sweagent_exit_message); (None, None) for an ordinary ending.
    `relay_failures`: the endpoint failures the tab's relay saw (relay_failure_count), None when
    no relay carried the run.

    * exit_api (SWEAGENT_INFRA_STATUSES, with or without autosubmission): a harness error when
      the relay saw an endpoint failure in the tab (relay_failures > 0, or None), else the
      model's outcome with the note saying so (the fork's model calls never raise a RetryError,
      and the runtime in the model's container can report one);
    * exit_error / exit_environment_error (SWEAGENT_ERROR_STATUSES): a context-window overflow in
      the message is the model's outcome (as exit_context). An exit_error whose exception text
      (after forward_with_handling's "Exit due to ... error: ") starts with a model-endpoint
      failure (swebench_common.scaffold_error_harness_reason) is a harness error when the relay
      saw an endpoint failure in the tab (relay_failures > 0, or None): gbench gives the fork's
      litellm calls a client timeout longer than the relay's safety cap on a model request,
      with no transport retries (sweagent_command), so a real endpoint failure is one the relay
      answered as failed, while the text alone could come from the runtime in the model's
      container (see SWEAGENT_ERROR_STATUSES). (Until 2026-10-11 the fork's calls ran under
      litellm 1.87's 600 s fallback timeout, shorter than the relay's limit.) The corroboration is per tab (one SWE-agent batch), not per instance: one endpoint failure the
      relay saw anywhere in the tab makes every such exit of that tab a harness error. Anything
      else stays the model's outcome, with the note saying so: an exit_environment_error (a
      SwerexException comes from the runtime or SWE-ReX's connection to it, never from the
      model endpoint), a docker daemon message (SWE-agent's steps reach the task container
      through its runtime, not the docker CLI), e.g. the EOF of a shell the model killed."""
    if any(s in status for s in SWEAGENT_INFRA_STATUSES):
        if relay_failures is None or relay_failures > 0:
            return (f"SWE-agent ended the rollout with {status} (not the model's outcome: a "
                    "tenacity RetryError while the relay saw the endpoint fail): "
                    f"{(message or '')[:300]}"), None
        return None, (f"SWE-agent ended the rollout with {status}, a tenacity RetryError while "
                      "the relay answered every model request of the tab (the fork's model "
                      "calls re-raise their last error, and the runtime in the model's container "
                      "can report a RetryError): the model's outcome, as upstream counts it: "
                      f"{(message or '')[:300]}")
    if not any(s in status for s in SWEAGENT_ERROR_STATUSES):
        return None, None
    if CONTEXT_OVERFLOW_RE.search(message or ""):
        return None, None
    first = str(message or "").split(" | ")[0]
    reason = scaffold_error_harness_reason(_SWEAGENT_EXIT_PREFIX.sub("", first, count=1))
    endpoint = (reason == "a model endpoint failure" and "exit_environment_error" not in status)
    if endpoint and (relay_failures is None or relay_failures > 0):
        return (f"SWE-agent ended the rollout with {status} (not the model's outcome: {reason}): "
                f"{first[:300]}"), None
    why = ("names a model endpoint failure the relay did not see (the relay answered every "
           "model request of the tab)" if endpoint else "names no model endpoint failure")
    return None, (f"SWE-agent ended the rollout with {status}, which {why} (the model's "
                  "commands can end its shell, its container or SWE-agent's state command): the "
                  f"model's outcome, as upstream counts it: {first[:300]}")


def sweagent_exit_is_infra(status: str, message: str,
                           relay_failures: Optional[int] = None) -> bool:
    """Whether a finished SWE-agent rollout ended for an infrastructure reason
    (sweagent_exit_verdict gives a harness error reason)."""
    return sweagent_exit_verdict(status, message, relay_failures)[0] is not None


_UUID4_RE = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"


def task_container_name_re(image: str) -> "re.Pattern[str]":
    """The names SWE-ReX gives the task containers of `image` (DockerDeployment
    _get_container_name: the image name's letters, digits, '-', '_' and '.', then '-' and a
    uuid4)."""
    sanitized = "".join(c for c in str(image) if c.isalnum() or c in "-_.")
    return re.compile(re.escape(sanitized) + "-" + _UUID4_RE)


def batch_task_containers(batch: str) -> List[Tuple[str, str]]:
    """(id, name) of every container, running or not, that carries SWEAGENT_BATCH_LABEL=<batch>
    (the task containers of that SWE-agent batch); [] when docker cannot list them."""
    try:
        r = subprocess.run(["docker", "ps", "-a", "--filter",
                            f"label={SWEAGENT_BATCH_LABEL}={batch}",
                            "--format", "{{.ID}}\t{{.Names}}"],
                           capture_output=True, text=True, timeout=60)
    except Exception as e:
        logger.warning("swe_bench_live: could not list the task containers of %s (%s)", batch,
                       type(e).__name__)
        return []
    out = getattr(r, "stdout", "")
    found = []
    for line in (out if isinstance(out, str) else "").splitlines():
        cid, _, name = line.strip().partition("\t")
        if cid:
            found.append((cid, name.strip()))
    return found


def remove_batch_task_containers(batch: str, image: Optional[str] = None) -> List[str]:
    """Remove (docker rm -f) the task containers of SWE-agent batch `batch`, only those SWE-ReX
    started from `image` when it is given; returns their names."""
    pattern = task_container_name_re(image) if image else None
    hits = [(cid, name) for cid, name in batch_task_containers(batch)
            if pattern is None or pattern.fullmatch(name)]
    if hits:
        try:
            subprocess.run(["docker", "rm", "-f", *[cid for cid, _ in hits]],
                           capture_output=True, timeout=180)
        except Exception as e:
            logger.warning("swe_bench_live: could not remove the task containers %s (%s)",
                           [n for _, n in hits], type(e).__name__)
    return [name for _, name in hits]


def _newest_mtime(path: str) -> Optional[float]:
    """The newest modification time of `path` and, for a directory, of everything under it."""
    try:
        newest = os.stat(path).st_mtime
    except OSError:
        return None
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for name in dirs + files:
                try:
                    newest = max(newest, os.stat(os.path.join(root, name)).st_mtime)
                except OSError:
                    pass
    return newest


#: The start of a record in SWE-agent's per-instance logs ("2026-09-29 00:12:55,750 - TRACE -").
_SWEAGENT_LOG_RECORD_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - [A-Z]+ - ")
#: SWEEnv.communicate logs each command at TRACE ("Input:") before it runs it in the task
#: container, and its output ("Output:") when the command returns (<id>.trace.log). In a batch
#: the logger is named per instance ("swea-env-<instance id>"; "swea-env" alone in a single run):
#: on 2026-10-08 a pattern for the bare name alone matched no batch record, so the watchdog never
#: saw the command an instance was running.
_SWEAGENT_TRACE_INPUT_RE = re.compile(r" - swea-env(?:-\S+)? - Input:")
_SWEAGENT_TRACE_OUTPUT_RE = re.compile(r" - swea-env(?:-\S+)? - Output:")
#: How much of the end of a trace log sweagent_inflight_command reads.
_TRACE_TAIL_BYTES = 4 << 20
#: The start of a model request in SWE-agent-for-eval's trace log (sweagent/agent/models.py):
#: LiteLLMModel.query logs "n_cache_control: N" (_history_to_messages) before its first attempt,
#: and its tenacity loop logs "Retrying LM query" before each further attempt; _single_query logs
#: "Response: " when the reply arrived. Logger "swea-lm" ("swea-lm-<instance id>" in a batch).
_SWEAGENT_LM_RECORD_RE = re.compile(r" - swea-lm(?:-\S+)? - ")
_SWEAGENT_LM_START_RE = re.compile(r" - swea-lm(?:-\S+)? - (?:n_cache_control: |Retrying LM query)")
_SWEAGENT_LM_RESPONSE_RE = re.compile(r" - swea-lm(?:-\S+)? - Response: ")


def sweagent_model_request_inflight(trace_log: str, tail_bytes: int = _TRACE_TAIL_BYTES
                                    ) -> bool:
    """True when <id>.trace.log (its last `tail_bytes`) shows a model request SWE-agent started
    and has not had answered: a request start (_SWEAGENT_LM_START_RE) followed only by other
    swea-lm records that are not its "Response: " (e.g. the per-query warning about the model's
    input limit). Any other record after it (the response, a command, an error the agent logs)
    ends it; no start in the tail is False."""
    try:
        with open(trace_log, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return False
    inflight = False
    for line in text.split("\n"):
        if not _SWEAGENT_LOG_RECORD_RE.match(line):
            continue
        if _SWEAGENT_LM_START_RE.search(line):
            inflight = True
        elif not _SWEAGENT_LM_RECORD_RE.search(line) or _SWEAGENT_LM_RESPONSE_RE.search(line):
            inflight = False
    return inflight


def sweagent_inflight_command(trace_log: str, tail_bytes: int = _TRACE_TAIL_BYTES
                              ) -> Optional[str]:
    """The command SWE-agent sent to the task container's runtime that has not returned yet:
    the text of the last `Input:` record of <id>.trace.log when no `Output:` record follows it;
    None when the last command returned, or the log (its last `tail_bytes`) holds no command."""
    try:
        with open(trace_log, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - tail_bytes))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if not _SWEAGENT_LOG_RECORD_RE.match(line):
            continue
        if _SWEAGENT_TRACE_INPUT_RE.search(line):
            start = i
        elif _SWEAGENT_TRACE_OUTPUT_RE.search(line):
            start = None
    if start is None:
        return None
    body: List[str] = []
    for line in lines[start + 1:]:
        if _SWEAGENT_LOG_RECORD_RE.match(line):
            break
        body.append(line)
    command = "\n".join(body).strip()
    return command or None


class SweagentWatch:
    """gbench's watchdog over one SWE-agent batch: it bounds a hang, and names it.

    SWE-agent writes each instance's logs and trajectory under <out_dir>/<instance id>/ as the
    rollout goes, and the batch's output to <out_dir>/sweagent_run.log. Every `poll_s` seconds
    (`check`), over the instances that started (their directory exists) and have not ended (no
    .pred yet, and not listed in run_batch_exit_statuses.yaml, which also names an instance
    whose run raised, "Uncaught <Exception>"):
      * an instance whose files have not changed for more than its limit is stalled: `stall_s`
        (SWEAGENT_STALL_S: longer than any command the fork's limits allow), or, while its trace
        log shows a model request in flight (sweagent_model_request_inflight), `model_stall_s`
        (sweagent_stall_s of `request_timeout_s`, SWE-agent's client timeout); its task container is
        removed (remove_batch_task_containers), and whatever SWE-agent then records for it, the
        instance is a harness error (AGENT_STALLED, read_sweagent_run); the reason names the
        command it was running (sweagent_inflight_command), if one had not returned;
      * a batch in which no file has changed for more than `grace_s` beyond the longest limit
        of its live instances (`stall_s`, or `model_stall_s` when one waits for a model request)
        is stopped: the
        SWE-agent container and every task container of the batch are removed, and every
        instance without a .pred is a harness error.
    The fork's own limits (SWEAGENT_CALL_LIMIT model calls, SWEAGENT_COMMAND_TIMEOUT_S per
    command, SWEAGENT_MAX_CONSECUTIVE_TIMEOUTS consecutive command timeouts,
    SWEAGENT_TOTAL_EXECUTION_S of commands) end every live rollout; the watchdog only acts when
    nothing moves for longer than they allow. (Until 2026-10-08 it also ended, after 5 minutes,
    an instance whose command held a heredoc the task containers' runtime dropped; the runtime
    is fixed (SWEREX_RUNTIME_SHA256) and a heredoc now runs as bash runs it, so a long command
    with a heredoc is an ordinary long command.)"""

    def __init__(self, out_dir: str, images: Dict[str, str], container: str, *,
                 stall_s: Optional[float] = None, grace_s: Optional[float] = None,
                 poll_s: Optional[float] = None, clock=time.time,
                 request_timeout_s: Optional[float] = None,
                 model_stall_s: Optional[float] = None):
        self.out_dir = out_dir
        #: the longest one model request of the batch may take (SWE-agent's client timeout),
        #: named in a stall's reason; EndpointRelay.TIMEOUT_S when not given
        self.request_timeout_s = float(request_timeout_s or EndpointRelay.TIMEOUT_S)
        self.images = dict(images)
        self.container = container
        self.stall_s = float(SWEAGENT_STALL_S if stall_s is None else stall_s)
        #: the limit of an instance waiting for a model request: sweagent_stall_s of
        #: request_timeout_s, never below stall_s
        self.model_stall_s = max(self.stall_s, float(
            sweagent_stall_s(self.request_timeout_s) if model_stall_s is None
            else model_stall_s))
        self.grace_s = float(SWEAGENT_STALL_GRACE_S if grace_s is None else grace_s)
        self.poll_s = float(SWEAGENT_WATCH_POLL_S if poll_s is None else poll_s)
        self.clock = clock
        self.started = clock()
        #: instance id -> the reason it is a harness error (its kind is in `kinds`)
        self.stalled: Dict[str, str] = {}
        #: instance id -> its harness error kind (AGENT_STALLED)
        self.kinds: Dict[str, str] = {}
        #: why the batch was stopped, None while it was not
        self.stopped: Optional[str] = None
        self.removed: Dict[str, List[str]] = {}
        import threading
        self._halt = threading.Event()
        self._thread: Optional[Any] = None
        #: the thread that runs the batch: a signal handed to it (a suite under --suite-timeout
        #: runs in a worker thread) stops the batch too
        self._owner = threading.get_ident()

    def finished(self, instance_id: str) -> bool:
        return os.path.isfile(os.path.join(self.out_dir, instance_id, f"{instance_id}.pred"))

    def _flag(self, iid: str, image: str, kind: str, why: str) -> None:
        removed = remove_batch_task_containers(self.container, image)
        self.removed[iid] = removed
        self.kinds[iid] = kind
        self.stalled[iid] = (
            why + "; " + (f"gbench removed its task container {', '.join(removed)}" if removed
                          else "gbench found no task container of it to remove")
            + ". Not the model's outcome")
        logger.error("swe_bench_live: %s: %s; removed its task container(s) %s and counted it "
                     "as a harness error (%s)", iid, why, removed or "(none found)", kind)

    def _inflight(self, iid: str) -> str:
        """' (it was running: <first line of its command>)' for an instance whose last command
        has not returned, else ''."""
        command = sweagent_inflight_command(
            os.path.join(self.out_dir, iid, f"{iid}.trace.log"))
        if not command:
            return ""
        return f" (the command it was running, first line: {command.split(chr(10), 1)[0][:200]!r})"

    def check(self, now: Optional[float] = None) -> None:
        """One look at the batch's files (see the class docstring)."""
        now = self.clock() if now is None else float(now)
        if self.stopped:
            return
        ended = set(_sweagent_exit_statuses(self.out_dir))
        batch_limit = self.stall_s
        for iid, image in self.images.items():
            if iid in self.stalled or iid in ended or self.finished(iid):
                continue
            last = _newest_mtime(os.path.join(self.out_dir, iid))
            if last is None:
                continue
            silent = now - last
            waiting = sweagent_model_request_inflight(
                os.path.join(self.out_dir, iid, f"{iid}.trace.log"))
            limit = self.model_stall_s if waiting else self.stall_s
            if waiting:
                batch_limit = self.model_stall_s
            if silent > limit:
                self._flag(iid, image, AGENT_STALLED, (
                    f"stalled: no file of this rollout changed for {int(silent)} s, longer than "
                    + (f"its model request may take (SWE-agent's client timeout "
                       f"{int(self.request_timeout_s)} s, plus "
                       f"{int(self.model_stall_s - self.request_timeout_s)} s)" if waiting else
                       f"any single step outside a model request SWE-agent-for-eval's limits "
                       f"allow (a command ends at {SWEAGENT_COMMAND_TIMEOUT_S} s; limit "
                       f"{int(self.stall_s)} s)")
                    + self._inflight(iid)))
        last = max(_newest_mtime(self.out_dir) or self.started, self.started)
        if now - last > batch_limit + self.grace_s:
            self.stop(f"no file of the SWE-agent batch changed for {int(now - last)} s (longer "
                      f"than any single step SWE-agent-for-eval's limits allow, "
                      f"{int(batch_limit)} s, plus {int(self.grace_s)} s)")

    def stop(self, reason: str) -> None:
        """Stop the batch: remove its SWE-agent container and every task container."""
        if self.stopped:
            return
        self.stopped = reason
        logger.error("swe_bench_live: stopping the SWE-agent batch %s: %s", self.container,
                     reason)
        try:
            subprocess.run(["docker", "rm", "-f", self.container], capture_output=True,
                           timeout=180)
        except Exception as e:
            logger.warning("swe_bench_live: could not remove %s (%s)", self.container,
                           type(e).__name__)
        remove_batch_task_containers(self.container)

    def _loop(self) -> None:
        from ...utils.process_groups import stop_requested
        while not self._halt.wait(self.poll_s):
            try:
                stop = stop_requested(self._owner)
                if stop:
                    # The CLI's SIGTERM/SIGHUP handler raised SystemExit in the main thread; a
                    # batch run from a worker thread (--suite-timeout) would go on, and hold
                    # the process open until it ended.
                    self.stop(f"the run was stopped by signal {stop[0]}")
                    return
                self.check()
            except Exception:
                logger.exception("swe_bench_live: the SWE-agent watchdog check failed")

    def start(self) -> "SweagentWatch":
        import threading
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"sweagent-watch-{self.container}")
        self._thread.start()
        return self

    def halt(self) -> None:
        """Stop the watchdog and wait for a check in progress to end, so `stalled`, `kinds` and
        `stopped` are final when read after it (a check is bounded by the timeouts of its
        docker calls)."""
        self._halt.set()
        if self._thread is not None:
            self._thread.join()

    def record(self) -> Dict[str, Any]:
        """What the watchdog saw, for the tab's result (`agent_watchdog`)."""
        return {"stall_limit_s": int(self.stall_s),
                "batch_stop_limit_s": int(self.stall_s + self.grace_s),
                "model_request_limit_s": int(self.request_timeout_s),
                "model_request_stall_limit_s": int(self.model_stall_s),
                "model_request_batch_stop_limit_s": int(self.model_stall_s + self.grace_s),
                "stalled_instances": sorted(i for i in self.stalled
                                            if self.kinds.get(i, AGENT_STALLED) == AGENT_STALLED),
                "removed_task_containers": {k: v for k, v in sorted(self.removed.items())},
                "batch_stopped": self.stopped}


def read_sweagent_run(tasks: List[swe_agentic.AgentTask], out_dir: str, started: float,
                      pull_errors: Optional[Dict[str, str]] = None,
                      relay_failures: Optional[int] = None,
                      stalled: Optional[Dict[str, str]] = None,
                      stopped: Optional[str] = None,
                      stalled_kinds: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """One trace per task from SWE-agent's own files: <id>/<id>.pred (written when the run
    finished: `model_patch` is the submission) and <id>/<id>.traj (`info`: exit_status,
    model_stats; its last step's message). Infrastructure failures (kept out of the scored set):
    an instance whose image could not be pulled on the host (`pull_errors`, instance id ->
    error; it was not handed to SWE-agent), no .pred, and a finished rollout that
    sweagent_exit_verdict classifies so (exit_api; an exit_error whose exception names a model
    endpoint failure the relay saw, `relay_failures`). Anything else is the model's outcome as
    SWE-agent records it (an empty or missing submission is an empty patch); an error exit kept
    as the model's outcome carries `agent_exit_note`. Model turns cut off at a limit are counted
    per tab by the relay (relay_result_reasons), so each trace records `truncated_turns: None`.
    An instance gbench's watchdog ended (`stalled`, instance id -> reason, SweagentWatch) is a
    harness error of the kind `stalled_kinds` gives it (AGENT_STALLED by default) whatever
    SWE-agent then recorded for it; an instance without a .pred after
    the watchdog stopped the batch (`stopped`: why) names that stop."""
    statuses = _sweagent_exit_statuses(out_dir)
    pull_errors = pull_errors or {}
    stalled = stalled or {}
    stalled_kinds = stalled_kinds or {}
    traces = []
    for i, t in enumerate(tasks):
        inst = os.path.join(out_dir, t.instance_id)
        pred_path = os.path.join(inst, f"{t.instance_id}.pred")
        traj_path = os.path.join(inst, f"{t.instance_id}.traj")
        swe_agentic.scrub_secrets_in_files(pred_path, traj_path)
        traj = _read_json(traj_path)
        info = traj.get("info") or {}
        message = _sweagent_exit_message(traj)
        finished = os.path.isfile(pred_path) and t.instance_id not in pull_errors
        patch = (_read_json(pred_path).get("model_patch") or "") if finished else ""
        status = str(info.get("exit_status") or statuses.get(t.instance_id) or "no_status")
        error = note = kind = None
        if t.instance_id in pull_errors:
            infra, status = True, "image_pull_failed"
            error = f"instance image: {pull_errors[t.instance_id]}"
        elif t.instance_id in stalled:
            infra, kind = True, stalled_kinds.get(t.instance_id, AGENT_STALLED)
            error = (f"{stalled[t.instance_id]}; SWE-agent's status: {status}; see "
                     f"{os.path.join(inst, t.instance_id + '.info.log')}")
        elif not finished:
            infra = True
            error = (f"SWE-agent did not finish this instance ({status})"
                     + (f": gbench stopped the batch, {stopped}" if stopped else "")
                     + f"; see {os.path.join(inst, t.instance_id + '.info.log')}")
        else:
            error, note = sweagent_exit_verdict(status, message, relay_failures)
            infra = error is not None
        traces.append({
            "sample_idx": i, "source_sample_idx": i, "attempt_idx": 0, "category": t.category,
            "messages": [{"role": "user", "content": t.problem_statement}],
            "extra_payload": {"instance_id": t.instance_id, **t.meta}, "gold_answer": "",
            "response_text": patch, "agent_patch": patch, "agent_exit_status": status,
            "agent_exit_message": (message[:1000] or None) if "exit_" in status else None,
            "agent_error": error, "agent_infra_error": infra, "agent_image": t.image,
            **({"agent_exit_note": note} if note else {}),
            **({"agent_harness_error": kind} if kind else {}),
            "agent_steps": int((info.get("model_stats") or {}).get("api_calls") or 0),
            "trajectory": [], "trajectory_file": traj_path if os.path.isfile(traj_path) else None,
            "truncated_turns": None,
            "duration_s": round(time.time() - started, 2), "is_correct": False,
            "status": "AGENT_INFRA_ERROR" if infra else "OK"})
    return traces


def _raise_if_stop_requested(where: str) -> None:
    """Raise SystemExit (128 + the signal, as the CLI's handler does) when the run was asked to
    stop by SIGTERM or SIGHUP (utils.process_groups.stop_requested)."""
    from ...utils.process_groups import stop_requested
    stop = stop_requested()
    if stop:
        logger.error("swe_bench_live: the run was stopped by signal %d (%s); nothing more is "
                     "started or scored", stop[0], where)
        raise SystemExit(128 + stop[0])


def run_sweagent(tasks: List[swe_agentic.AgentTask], rows: Sequence[Dict[str, Any]],
                 model_name: str, relay_url: str, concurrency: int,
                 out_dir: str, relay: Optional[EndpointRelay] = None) -> List[Dict[str, Any]]:
    """SWE-agent-for-eval's rollout on the selected instances (one run-batch in its image);
    one trace per instance (read_sweagent_run, with the endpoint failures `relay`, the
    EndpointRelay at `relay_url`, saw during the batch).

    First the instance images and SWE-ReX's builder base (SWEREX_BUILDER_BASE) are pulled on the
    host (host_pull_images: the operator's Docker login; the SWE-agent container gets no Docker
    client configuration, so SWE-ReX finds them present and pulls nothing). An instance whose
    image cannot be pulled is not handed to SWE-agent: its trace is an infrastructure failure.
    The pulls are recorded in <out_dir>/image_pulls.json. A builder base that cannot be pulled is
    recorded too; SWE-ReX's build then fetches it itself or, failing that, uses its own fallback
    (building on the instance image).

    Before anything is pulled, the cloud metadata server check runs for the task containers
    SWE-ReX starts (on docker's default bridge; the model's commands run there):
    swebench_common.require_task_network_blocked, probing with the SWE-agent image, which also
    checks that the hosts SWE-ReX's runtime build fetches from resolve (SWEREX_BUILD_HOSTS).

    While the batch runs, SweagentWatch bounds a hang: a stalled instance (no file change for
    SWEAGENT_STALL_S, or, while its trace log shows a model request in flight, for
    sweagent_stall_s of SWE-agent's client timeout, swe_agentic.client_timeout_for_relay_limit
    of the relay's limit) loses its task container and is a harness error (AGENT_STALLED), and a
    batch with no file change for SWEAGENT_STALL_GRACE_S more is stopped. What it saw is in
    <out_dir>/watchdog.json. When the batch ends, however it ends, the SWE-agent container and
    every task container of the batch (label SWEAGENT_BATCH_LABEL) are removed: a SIGTERM or
    SIGHUP to a gbench run reaches this `finally` too (the CLI turns them into SystemExit)."""
    swebench_common.require_task_network_blocked(
        "swe_bench_live", [sweagent_image()], docs_url=DOCS_URL,
        hint=f"Build the SWE-agent-for-eval image {sweagent_image()} (see {DOCS_URL}).",
        resolve=SWEREX_BUILD_HOSTS)
    images = {t.instance_id: t.image for t in tasks}
    pulls = host_pull_images([*images.values(), SWEREX_BUILDER_BASE], concurrency)
    pull_errors = {iid: pulls[img]["error"] for iid, img in images.items()
                   if pulls.get(img, {}).get("state") == "failed"}
    with open(os.path.join(out_dir, "image_pulls.json"), "w", encoding="utf-8") as f:
        json.dump({"instance_images": image_pull_summary({images[i]: pulls[images[i]]
                                                          for i in images}),
                   "builder_base": {"image": SWEREX_BUILDER_BASE,
                                    **pulls.get(SWEREX_BUILDER_BASE, {})}}, f, indent=1)
    if pull_errors:
        logger.error("swe_bench_live: %d instance image(s) could not be pulled on the host and "
                     "count as infrastructure failures (see image_pulls.json in %s)",
                     len(pull_errors), out_dir)
    wanted = {t.instance_id for t in tasks} - set(pull_errors)
    started = time.time()
    if not wanted:
        return read_sweagent_run(tasks, out_dir, started, pull_errors)
    instances_path = os.path.join(out_dir, "instances.jsonl")
    with open(instances_path, "w", encoding="utf-8") as f:
        for r in unique_instance_rows(rows):
            if r["instance_id"] in wanted:
                f.write(json.dumps(sweagent_instance(r)) + "\n")
    name = f"gbench-sweagent-{uuid.uuid4().hex[:12]}"
    # SWE-agent's client timeout: the relay's safety cap on a model request plus the margin for
    # its answer (sweagent_command), and from it, at run time, the watchdog's limit for an
    # instance waiting for a model request; any other silence keeps SWEAGENT_STALL_S. No relay
    # (a caller that runs the batch against the endpoint itself): the run's own sizing.
    relay_limit = getattr(relay, "TIMEOUT_S", None) if relay is not None else None
    client_timeout_s = (swe_agentic.client_timeout_for_relay_limit(relay_limit) if relay_limit
                        else swe_agentic.agent_request_timeouts().client_s)
    watch = SweagentWatch(out_dir, {i: images[i] for i in images if i in wanted}, name,
                          request_timeout_s=client_timeout_s,
                          model_stall_s=sweagent_stall_s(client_timeout_s))

    def _cleanup() -> None:
        # The SWE-agent container, then every task container of the batch: SWE-ReX removes a
        # task container itself only when its rollout ends normally.
        try:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=180)
        except Exception as e:
            logger.warning("swe_bench_live: could not remove %s (%s)", name, type(e).__name__)
        left = remove_batch_task_containers(name)
        if left:
            logger.warning("swe_bench_live: removed %d task container(s) SWE-agent left behind: "
                           "%s", len(left), left)
    try:
        with open(os.path.join(out_dir, "sweagent_run.log"), "w", encoding="utf-8") as log:
            watch.start()
            subprocess.run(sweagent_command(instances_path, out_dir, model_name, relay_url,
                                            concurrency, name, client_timeout_s),
                           stdout=log, stderr=subprocess.STDOUT, env=credential_free_env())
        # A SIGTERM/SIGHUP to the run ends the batch (its docker client gets the signal) but
        # reaches a suite running in a worker thread (--suite-timeout) only through this flag:
        # stop here, after the cleanup below, rather than score and start the next tab.
        _raise_if_stop_requested("after the SWE-agent batch")
    finally:
        watch.halt()
        _cleanup()
        swe_agentic.scrub_secrets_in_files(os.path.join(out_dir, "sweagent_run.log"))
        try:
            with open(os.path.join(out_dir, "watchdog.json"), "w", encoding="utf-8") as f:
                json.dump(watch.record(), f, indent=1)
        except OSError:
            pass
    return read_sweagent_run(tasks, out_dir, started, pull_errors,
                             relay_failure_count(relay.record()) if relay is not None else None,
                             stalled=watch.stalled, stopped=watch.stopped,
                             stalled_kinds=watch.kinds)


def _agent_tab_result(setting: str, *, model_name: str, thinking: bool,
                      traces: List[Dict[str, Any]], scorer, metrics: Dict[str, Any],
                      sampling: OfficialSampling, temperature: float,
                      temperature_source: str, budget: Any, agent: Dict[str, Any],
                      relay: EndpointRelay, out_dir: str, limit: Optional[int],
                      limit_note: Optional[str] = None,
                      shard: Optional[Tuple[int, int]], penalties: Optional[Dict[str, float]],
                      attempts: int, start: float, finalize,
                      model_client: Optional[Dict[str, Any]] = None,
                      run_max_tokens: Optional[int] = None) -> Dict[str, Any]:
    """Score the rollouts that ran and build one tab's agentic result (SWE-agent or
    Win-Agent). `finalize(result, metrics)` applies the tab's harness-report checks and returns
    extra non-comparability reasons. What the relay counted (endpoint failures, turns cut off at
    a limit, turns empty after reasoning) is applied by swebench_common.relay_result_reasons.
    `model_client` is the agent's model_client_timeout record; `run_max_tokens` the max_tokens
    the relay set on every request (GBENCH_AGENT_MAX_TOKENS_FROM_RUN), None when none."""
    import asyncio
    import collections
    # A rollout that could not run is a harness error: out of the numerator and the denominator,
    # as is an instance the harness gave no verdict (the scorer marks it).
    for t in traces:
        if t["agent_infra_error"]:
            # AGENT_STALLED for a rollout gbench's watchdog ended, AGENT_INFRA otherwise
            mark_harness_error(t, t.get("agent_harness_error") or AGENT_INFRA,
                               t.get("agent_error") or t.get("agent_exit_status"))
    scoreable = [t for t in traces if not t["agent_infra_error"]]
    if scoreable:
        asyncio.run(scorer(scoreable))
    total = len(traces)
    infra = total - len(scoreable)
    head = scored_headline(traces)
    scaffold = agent.get("scaffold", "the agent")
    result: Dict[str, Any] = {
        "benchmark_type": "eval", "eval_name": "swe_bench_live", "model_name": model_name,
        "thinking": thinking, "mode": swe_agentic.AGENTIC,
        "mode_source": swe_agentic.resolve_mode("swe_bench_live")[1],
        "scoring_mode": "execution", "temperature": temperature,
        "temperature_source": temperature_source,
        "official_sampling": {**sampling.record(temperature, temperature_source),
                              **budget.record()},
        "max_output_tokens_per_turn": run_max_tokens, "max_output_tokens_note": (
            f"--max-output-tokens={run_max_tokens} is sent with every request (set by gbench's "
            f"relay, {swe_agentic.AGENT_MAX_TOKENS_FROM_RUN_ENV}=1); {scaffold} "
            "sends none" if run_max_tokens else budget.note),
        "model_client_timeout": model_client,
        "decoding_penalties": penalties or None,
        "agent": agent, "relay": relay.record(), "agent_output_dir": out_dir,
        "total_questions": total, "correct_answers": head["correct_answers"],
        "scored_instances": head["scored_instances"],
        "scoring_excluded": head["scoring_excluded"],
        "failed_requests": head["scoring_excluded"],
        "accuracy": head["accuracy"],
        "category_accuracy": head["category_accuracy"],
        "agent_exit_statuses": dict(collections.Counter(str(t["agent_exit_status"])
                                                        for t in traces)),
        "agent_steps_total": (sum(int(t["agent_steps"]) for t in traces)
                              if all(t.get("agent_steps") is not None for t in traces)
                              else None),
        "sample_traces": traces, "status": "success",
    }
    kept = sorted(str((t.get("extra_payload") or {}).get("instance_id")) for t in traces
                  if t.get("agent_exit_note"))
    if kept:
        # Error exits that name no endpoint or docker daemon failure: the model's outcome
        # (sweagent_exit_verdict, win_agent_error_verdict), scored, and counted here.
        result["exits_kept_as_model_outcome"] = len(kept)
        result["exits_kept_as_model_outcome_ids"] = kept
    pulls_path = os.path.join(out_dir, "image_pulls.json")
    if os.path.isfile(pulls_path):
        result["image_pulls"] = _read_json(pulls_path)
    watch_path = os.path.join(out_dir, "watchdog.json")
    if os.path.isfile(watch_path):
        result["agent_watchdog"] = _read_json(watch_path)
    if attempts > 1:
        # The harness keys predictions by instance id, so k attempts would collapse into one.
        result["attempts"] = {
            "attempts_requested": attempts, "attempts_per_sample": 1,
            "not_applied": "one rollout per instance: the harness scores one patch per "
                           "instance id, so an @k would be fabricated"}
    result.update(metrics)
    noncanon: List[str] = []
    if shard:
        result["shard"] = f"{shard[0]}/{shard[1]}"
        noncanon.append(f"shard {shard[0]}/{shard[1]} of the instance set")
    if limit:
        noncanon.append(limit_note or f"subset run (--eval-limit {limit})")
    noncanon.append(f"temperature={temperature} "
                    f"{swe_agentic._temperature_origin(temperature_source)}; "
                    f"{scaffold} sends none")
    if penalties:
        noncanon.append(f"decoding penalties {penalties}")
    if run_max_tokens:
        rec, why = swe_agentic.agent_max_tokens_record(run_max_tokens, scaffold,
                                                       "the relay's body fields")
        result["official_sampling"]["agent_max_tokens_from_run"] = rec
        noncanon.append(why)
    if not total:
        if shard:
            # an empty shard is an honest empty subset, with no score (not 0%)
            swe_agentic.mark_empty_shard(result, shard)
        else:
            result.update(status="error", error="no instances were selected, so nothing was "
                                                "evaluated")
        noncanon.append("no instances evaluated")
    elif infra == total:
        first = next((t for t in traces if t.get("agent_error")), {})
        result.update(status="error", error=(
            f"{scaffold} could not run on any of {total} instance(s) (e.g. "
            f"{(first.get('extra_payload') or {}).get('instance_id')}: "
            f"{str(first.get('agent_error'))[:200]}); this is an infrastructure failure, not a "
            "0% resolved rate"))
        noncanon.append("agent infrastructure failure (no instance could run)")
    elif infra:
        result["status"] = "completed_with_errors"
        result["partial_evaluation"] = {"total_instances": total,
                                        "scoreable_instances": total - infra,
                                        "agent_infra_error_instances": infra}
        noncanon.append(f"partial: {infra} of {total} instance(s) could not run the agent "
                        "(infrastructure)")
    if total:
        noncanon += finalize(result, metrics) or []
        record_harness_errors(result, head)
        if result.get("status") == "success" and head["scoring_excluded"]:
            result["status"] = "completed_with_errors"
        if result.get("accuracy") is None and result.get("status") != "error":
            result.update(status="error", error=(
                f"no instance was scored: all {total} are harness errors "
                f"({harness_errors_text(head['harness_errors'])}); this is an infrastructure "
                "failure, not a 0% resolved rate"))
            noncanon.append("harness infrastructure failure (no scoreable instances)")
    withhold_on_error(result)
    noncanon += relay_result_reasons(result, result["relay"], scaffold)
    noncanon += _leaderboard_totals(setting, result)
    result["leaderboard_comparable"] = not noncanon
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    result["duration_s"] = round(time.time() - start, 2)
    return result


def _sweagent_tab(setting: str, rows: Sequence[Dict[str, Any]], scorer, metrics: Dict[str, Any],
                  finalize, model_name: str, base_url: str, concurrency: int,
                  kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """A Python or MultiLang tab, agentic: SWE-agent-for-eval rollouts through gbench's relay,
    then the tab's harness over the patches of the rollouts that ran."""
    start = time.time()
    thinking = bool(kwargs.get("enable_thinking"))
    limit = kwargs.get("limit")
    tasks = [swe_agentic.AgentTask(instance_id=r["instance_id"],
                                   problem_statement=r["problem_statement"],
                                   image=sweagent_instance(r)["image_name"],
                                   category=str(r.get("repo")),
                                   meta={"split": r.get("_split", setting)})
             for r in unique_instance_rows(rows)]
    tasks, shard = swe_agentic.select_tasks(tasks, limit, "swe_bench_live")
    temperature, temperature_source = resolve_official_sampling(
        "swe_bench_live", SWEAGENT_SAMPLING, thinking=thinking)
    budget = resolve_token_budget("swe_bench_live", SWEAGENT_SAMPLING, thinking=thinking,
                                  max_output_tokens=kwargs.get("max_output_tokens"))
    penalties = decoding_penalties()
    body: Dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": thinking}}
    body["temperature"] = temperature
    body.update(penalties or {})
    # GBENCH_AGENT_MAX_TOKENS_FROM_RUN (default on; =0 turns it off): the run's --max-output-tokens on every
    # request, set by the relay (the fork sends no max_tokens), recorded as a deviation.
    run_max_tokens = swe_agentic.agent_max_tokens_from_run(kwargs.get("max_output_tokens"))
    body = swe_agentic.with_run_max_tokens(body, run_max_tokens)
    # The relay's per-request limit; SWE-agent's client waits that plus a margin (run_sweagent).
    timeouts = swe_agentic.agent_request_timeouts(run_max_tokens)
    out_dir = swe_agentic.agent_output_dir("swe_bench_live")
    relay = EndpointRelay(base_url, "127.0.0.1", key=_endpoint_key(), body_fields=body,
                          name="swe-live-sweagent-relay", timeout_s=timeouts.relay_s,
                          client_timeout_suspect_s=timeouts.suspect_s()).start()
    try:
        traces = run_sweagent(tasks, rows, model_name, relay.url, concurrency,
                              out_dir, relay=relay) if tasks else []
    finally:
        relay.stop()
    agent = {"scaffold": "SWE-agent-for-eval", "commit": SWEAGENT_COMMIT,
             "config": SWEAGENT_CONFIG, "config_sha256": SWEAGENT_CONFIG_SHA256,
             "per_instance_call_limit": SWEAGENT_CALL_LIMIT,
             "command_timeout_s": SWEAGENT_COMMAND_TIMEOUT_S,
             "max_consecutive_command_timeouts": SWEAGENT_MAX_CONSECUTIVE_TIMEOUTS,
             "total_command_time_limit_s": SWEAGENT_TOTAL_EXECUTION_S,
             "model": f"openai/{model_name}", "image": sweagent_image(),
             "runner": f"container {sweagent_image()} (sweagent run-batch)",
             "task_runtime": {"swe_rex": "1.4.0", "patch": SWEREX_RUNTIME_PATCH,
                              "session_code_sha256": SWEREX_RUNTIME_SHA256,
                              "pins": list(SWEREX_RUNTIME_PINS)},
             "relay_timeout_s": timeouts.relay_s}
    client = swe_agentic.model_client_timeout(
        "litellm (SWE-agent-for-eval LiteLLMModel, --agent.model.completion_kwargs)",
        SWEAGENT_LITELLM_VERSION, timeouts.client_s, swe_agentic.AGENT_CLIENT_RETRIES,
        timeouts.source, replaces=(
            "litellm 1.87's fallback client timeout of 600 s (request_timeout 6000 -> "
            "COMPLETION_HTTP_FALLBACK_SECONDS) with 2 OpenAI SDK retries: 1800 s per SWE-agent "
            "attempt, any reply longer than 600 s abandoned and resampled"))
    attempts = int(kwargs.get("attempt_count") or get_run_knob("attempt_count", 1) or 1)
    return _agent_tab_result(
        setting, model_name=model_name, thinking=thinking, traces=traces, scorer=scorer,
        metrics=metrics, sampling=SWEAGENT_SAMPLING, temperature=temperature,
        temperature_source=temperature_source, budget=budget, agent=agent, relay=relay,
        out_dir=out_dir, limit=limit, limit_note=kwargs.get("limit_note"), shard=shard,
        penalties=penalties, attempts=attempts,
        start=start, finalize=finalize, model_client=client, run_max_tokens=run_max_tokens)


def execute_sweagent_python(eval_name: str, model_name: str, base_url: str, concurrency: int,
                            dataset: str, split: str, namespace: str, *,
                            dataset_revision: str, snapshot_columns: Sequence[str],
                            snapshot_rows: int, snapshot_sha256: str,
                            harness_image: Optional[str] = None,
                            **kwargs: Any) -> Dict[str, Any]:
    """A Python tab, agentic (the same call shape as swebench_common.execute_swebench for the
    one-shot mode): the pinned snapshot, SWE-agent-for-eval rollouts, then the SWE-bench-Live
    fork of the swebench harness in the local image reading the same snapshot file; its
    report checks, and the headline over rows when an evaluated id repeats."""
    path, rows = _pinned_snapshot_rows(dataset, split, dataset_revision, eval_name,
                                       snapshot_columns, snapshot_rows, snapshot_sha256)
    metrics: Dict[str, Any] = {"dataset": f"{dataset}@{dataset_revision} ({split})",
                               "tab": TAB_LABELS.get(split, split)}
    metrics.update(repeated_ids_record(rows))
    scorer = swebench_common.make_swebench_scorer(
        eval_name, model_name, path, split, namespace, concurrency, metrics,
        harness_image=harness_image, instance_image=python_instance_image)

    def _finalize(res: Dict[str, Any], m: Dict[str, Any]) -> List[str]:
        reasons = swebench_report_reasons(eval_name, res, m)
        row_total_headline(res, rows)
        return reasons
    return _sweagent_tab(split, rows, scorer, metrics, _finalize, model_name, base_url,
                         concurrency, kwargs)


def _run_python(setting: str, mode: str, model_name: str, base_url: str, concurrency: int,
                kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """A Python tab (Lite, Full, Verified): the rollouts (SWE-agent-for-eval, or one-shot), then
    the SWE-bench-Live fork of the swebench harness in the local image, on the same pinned
    snapshot file."""
    kw = dict(kwargs)
    kw["harness_image"] = _image()
    kw["instance_image"] = python_instance_image
    kw["dataset_revision"] = SETTING_REVISION[setting]
    kw["snapshot_columns"] = SNAPSHOT_COLUMNS
    kw["snapshot_rows"] = SPLIT_ROWS[setting]
    kw["snapshot_sha256"] = SPLIT_SHA256[setting]
    if mode == swe_agentic.AGENTIC:
        return execute_sweagent_python("swe_bench_live", model_name, base_url, concurrency,
                                       _DATASET, setting, _NAMESPACE, **kw)
    return swe_agentic.mark_oneshot(execute_swebench(
        "swe_bench_live", model_name, base_url, concurrency,
        _DATASET, setting, _NAMESPACE, **kw,
    ), "swe_bench_live")


def _run_multilang(setting: str, mode: str, model_name: str, base_url: str,
                   concurrency: int, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """A MultiLang tab: rollouts (SWE-agent-for-eval in each instance's `docker_image`, or
    one-shot), then one pass of upstream's multi-language harness (`--platform linux`)."""
    rows = load_setting_rows(setting)
    metrics: Dict[str, Any] = {"dataset": setting_dataset_label(setting),
                               "tab": TAB_LABELS[setting]}
    metrics.update(repeated_ids_record(rows))
    scorer = make_launch_scorer(setting, rows, "linux", concurrency, metrics)
    if mode == swe_agentic.AGENTIC:
        result = _sweagent_tab(setting, rows, scorer, metrics,
                               lambda res, m: launch_report_reasons(res, m), model_name,
                               base_url, concurrency, kwargs)
    else:
        # Every instance: run_eval_suite applies --shard, then the --eval-limit draw
        # (SWE_LIMIT_SEED, stratified by repo as the loader drew it).
        split_of = {}
        for r in rows:
            split_of.setdefault(r["instance_id"], r["_split"])
        samples = [(s[0], s[1], {**s[2], "split": split_of[s[2]["instance_id"]]})
                   for s in swe_samples_from_rows(rows, setting, None, "swe_bench_live",
                                                  setting_dataset_label(setting))]
        result = run_eval_suite(
            eval_name="swe_bench_live", model_name=model_name, base_url=base_url,
            concurrency=concurrency, samples=samples, async_eval_fn=scorer,
            thinking=kwargs.get("enable_thinking", False),
            extra_payload=kwargs.get("extra_payload"), limit=kwargs.get("limit"),
            limit_seed=SWE_LIMIT_SEED,
            max_output_tokens=kwargs.get("max_output_tokens", 8192),
            temperature=kwargs.get("temperature"))
        result.update(metrics)
        for reason in launch_report_reasons(result, metrics):
            swe_agentic.add_noncanon(result, reason)
        _oneshot_harness_errors(result)
        swe_agentic.mark_oneshot(result, "swe_bench_live")
        for reason in _leaderboard_totals(setting, result):
            swe_agentic.add_noncanon(result, reason)
    split_acc = _split_accuracy(result)
    if split_acc:
        result["split_accuracy"] = split_acc
    return result


# --------------------------------------------------------------------------------------------
# the Windows tab: Win-Agent + upstream's harness, on a Windows host
# --------------------------------------------------------------------------------------------
def win_agent_config(agent_dir: str, *, dataset_path: str, model_name: str, endpoint: str,
                     workers: int) -> Dict[str, Any]:
    """Win-Agent's own config/default.yaml with only what names this run changed: the
    dataset file, the served model (litellm `openai/<model>`) and its endpoint, and the number
    of parallel rollouts. Its tools, prompt, max_steps (100), instance_timeout (480 min) and
    platform (windows) are kept."""
    import yaml
    with open(os.path.join(agent_dir, "config", "default.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.update({"dataset": dataset_path, "model": f"openai/{model_name}", "base_url": endpoint,
                "workers": max(1, int(workers or 1))})
    return cfg


def _win_agent_run_id(cfg: Dict[str, Any]) -> str:
    """Win-Agent main.py's run id: the output directory under ./output."""
    return f"{cfg['dataset']}_{cfg['model']}".replace("/", "_").replace("\\", "_")


#: Win-Agent's log entry header (src/agent.py Agent.logger: "\n[%Y-%m-%d %H:%M:%S]: \n<log>\n").
#: The model's replies and tool results are logged json.dumps'ed, so none of their lines can
#: look like a header or a step line.
_WIN_AGENT_ENTRY = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]: ?$", re.M)
#: The entry the step loop logs before each model call ("Step 0...").
_WIN_AGENT_STEP = re.compile(r"^Step \d+\.\.\.$")
#: How Win-Agent ends the entry of an exception it caught: inside rollout's step loop
#: ("Patch might not be saved.") and around the whole rollout ("Patch not saved.").
_WIN_AGENT_ERROR_TAILS = ("Patch might not be saved.", "Patch not saved.")


def win_agent_error_verdict(log_text: Optional[str], relay_failures: Optional[int] = None
                            ) -> Tuple[Optional[str], Optional[str]]:
    """(harness error reason, model-outcome note) of a Win-Agent rollout it recorded as
    "error", from the instance's log (output/<run>/logs/<id>.txt) and, when gbench's relay
    carried the run's model requests, the number of endpoint failures it saw (`relay_failures`:
    its upstream_errors plus refused requests).

    Win-Agent (src/agent.py, WIN_AGENT_COMMIT) records "error" for any exception: the container
    launch and setup before the model's first turn (Runtime.from_launch_image, install_py,
    init_git), the model call after its 5 tries (src/llm.py re-raises the litellm exception),
    a tool that raised while running the model's call (e.g. the shell of a container the
    model's command stopped), and saving the patch after the loop. It logs each as an entry
    "<exception>\\n<traceback>\\nPatch [might] not [be] saved.". The log also holds the raw
    output of every shell command (src/runtime.py logs "Status inside container: <output>"),
    which the model's commands write, so after the model's first turn an entry of the log can
    be the model's own text.

    * no log, or no "Step N..." entry before the last error entry: the model never had a turn
      (nothing the model wrote is in the log then), a harness error;
    * after the model's first turn, an error entry whose exception text starts with a
      model-endpoint failure (swebench_common.scaffold_error_harness_reason) is a harness error
      only when the relay saw an endpoint failure in the run (relay_failures > 0, or None when
      no relay carried the run): Win-Agent's litellm calls send no timeout, and gbench sets
      litellm's own (REQUEST_TIMEOUT, win_agent_env) longer than the relay's safety cap on a
      model request, so each of its failed tries is one the relay answered as failed (until 2026-10-11 litellm
      fell back to 600 s, shorter than that limit);
    * anything else, including an error entry naming the docker daemon after the model's first
      turn (Win-Agent reaches the daemon for the rollout only in Runtime.from_launch_image,
      before it): the model's outcome, as upstream counts it, with the note saying so."""
    text = log_text or ""
    heads = list(_WIN_AGENT_ENTRY.finditer(text))
    bodies = [text[h.end():(heads[i + 1].start() if i + 1 < len(heads) else len(text))].strip()
              for i, h in enumerate(heads)]
    errors = [i for i, b in enumerate(bodies) if b.endswith(_WIN_AGENT_ERROR_TAILS)]
    last = errors[-1] if errors else len(bodies)
    if not any(_WIN_AGENT_STEP.match(b) for b in bodies[:last]):
        return ("Win-Agent raised before the model's first turn (container launch or setup; "
                "see its log)" if text else "Win-Agent raised and logged nothing for this "
                "instance (see win_agent.log)"), None
    exception = bodies[last].splitlines()[0].strip() if errors else ""
    reason = scaffold_error_harness_reason(exception)
    if reason == "a model endpoint failure" and (relay_failures is None or relay_failures > 0):
        return f"Win-Agent recorded error (not the model's outcome: {reason}): {exception[:300]}", \
            None
    why = ("names a model endpoint failure the relay did not see (the relay answered every "
           "model request of the run)" if reason == "a model endpoint failure" else
           "names no endpoint failure")
    return None, (f"Win-Agent recorded error after the model's turns, which {why} (the model's "
                  "commands can end its shell or container): the model's outcome, as upstream "
                  "counts it: " + (exception[:300] or "no error entry in its log"))


#: Transport retries of Win-Agent's litellm calls: the OpenAI SDK default, which src/llm.py does
#: not override and no environment variable sets (litellm 1.100 checked: 3 attempts per call).
WIN_AGENT_CLIENT_RETRIES = 2


def win_agent_request_timeout(client_timeout_s: float) -> str:
    """REQUEST_TIMEOUT for Win-Agent's litellm: `client_timeout_s`, never exactly litellm's
    package default 6000 (completion_timeout.py replaces that value by its 600 s fallback)."""
    value = int(client_timeout_s)
    return str(value + 1 if value == 6000 else value)


def win_agent_env(client_timeout_s: float) -> Dict[str, str]:
    """Win-Agent's environment: this process's without any credential variable
    (sandbox.credential_free_env), UTF-8 I/O, and REQUEST_TIMEOUT (win_agent_request_timeout).
    Win-Agent's src/llm.py passes no timeout to litellm.completion, so litellm uses its
    request_timeout, read from REQUEST_TIMEOUT when it is imported; unset, it is the package
    default 6000, which litellm's chat completion replaces by a 600 s client timeout, and a
    reply longer than that was abandoned and sent again. The variable is set in Win-Agent's
    process only, never in gbench's."""
    return dict(credential_free_env(), PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
                REQUEST_TIMEOUT=win_agent_request_timeout(client_timeout_s))


def run_win_agent(tasks: List[swe_agentic.AgentTask], rows: Sequence[Dict[str, Any]],
                  model_name: str, relay_url: str, concurrency: int,
                  out_dir: str, relay: Optional[EndpointRelay] = None) -> List[Dict[str, Any]]:
    """Run Win-Agent's main.py on the selected Windows instances; one trace per instance.
    `relay` (the EndpointRelay at `relay_url`) supplies the endpoint failures it saw during the
    run to win_agent_error_verdict.

    Win-Agent writes under its working directory (./output, and ./mnt, which it mounts into
    each task container), so it runs from a fresh copy of the pinned checkout inside
    `out_dir`. It gets the relay's address as its base_url and the placeholder key EMPTY on its
    command line; the relay adds the endpoint's key. Its environment is this process's without
    any credential variable (sandbox.credential_free_env)."""
    agent_dir = os.path.join(out_dir, "win-agent")
    shutil.copytree(win_agent_dir(), agent_dir, ignore=shutil.ignore_patterns(".git", "output"))
    wanted = {t.instance_id for t in tasks}
    # Inside Win-Agent's working directory and named relatively: main.py makes its output
    # directory name from the dataset string, and an absolute Windows path would put a drive
    # colon into it.
    dataset_name = "instances.jsonl"
    with open(os.path.join(agent_dir, dataset_name), "w", encoding="utf-8") as f:
        for r in _harness_rows(rows):
            if r["instance_id"] in wanted:
                f.write(json.dumps(r, default=str) + "\n")
    import yaml
    cfg = win_agent_config(agent_dir, dataset_path=dataset_name, model_name=model_name,
                           endpoint=relay_url, workers=concurrency)
    cfg_path = os.path.join(out_dir, "win_agent.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    # No credential variable reaches Win-Agent (it gets the relay and the placeholder key).
    relay_limit = getattr(relay, "TIMEOUT_S", None) if relay is not None else None
    env = win_agent_env(swe_agentic.client_timeout_for_relay_limit(relay_limit) if relay_limit
                        else swe_agentic.agent_request_timeouts().client_s)
    started = time.time()
    with open(os.path.join(out_dir, "win_agent.log"), "w", encoding="utf-8") as log:
        subprocess.run([sys.executable, "main.py", "--config_path", cfg_path,
                        "--api_key", "EMPTY"], cwd=agent_dir, stdout=log,
                       stderr=subprocess.STDOUT, env=env)
    run_dir = os.path.join(agent_dir, "output", _win_agent_run_id(cfg))

    def _load(name: str) -> Dict[str, Any]:
        try:
            with open(os.path.join(run_dir, name), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}
    preds, statuses = _load("preds.json"), _load("exit_status.json")
    relay_failures = relay_failure_count(relay.record()) if relay is not None else None
    traces = []
    for i, t in enumerate(tasks):
        status = statuses.get(t.instance_id)
        patch = (preds.get(t.instance_id) or {}).get("model_patch") or ""
        log_path = os.path.join(run_dir, "logs", f"{t.instance_id}.txt")
        error = note = None
        if status is None:
            error = "Win-Agent recorded no exit status"
        elif status == "error":
            log_text = None
            if os.path.isfile(log_path):
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    log_text = f.read()
            error, note = win_agent_error_verdict(log_text, relay_failures)
        infra = error is not None
        traces.append({
            "sample_idx": i, "source_sample_idx": i, "attempt_idx": 0, "category": t.category,
            "messages": [{"role": "user", "content": t.problem_statement}],
            "extra_payload": {"instance_id": t.instance_id, **t.meta}, "gold_answer": "",
            "response_text": patch, "agent_patch": patch,
            "agent_exit_status": status or "no_status",
            "agent_error": error, **({"agent_exit_note": note} if note else {}),
            "agent_infra_error": infra, "agent_image": t.image, "agent_steps": None,
            "trajectory": [], "trajectory_file": log_path,
            "truncated_turns": None,
            "duration_s": round(time.time() - started, 2), "is_correct": False,
            "status": "AGENT_INFRA_ERROR" if infra else "OK"})
    return traces


def _run_windows(setting: str, mode: str, model_name: str, base_url: str, concurrency: int,
                 kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """The Windows tab on a Windows host: Win-Agent rollouts (or one-shot), then upstream's
    harness with `--platform windows`. Both run model-controlled code in Windows task containers
    with network access, so nothing starts unless check_windows_metadata_guard passes
    (infra_required otherwise)."""
    ok, why = check_windows_metadata_guard()
    if not ok:
        raise infra_required("swe_bench_live", why, DOCS_URL)
    rows = load_setting_rows(setting)
    metrics: Dict[str, Any] = {"dataset": setting_dataset_label(setting),
                               "tab": TAB_LABELS[setting]}
    scorer = make_launch_scorer(setting, rows, "windows", concurrency, metrics)
    thinking = bool(kwargs.get("enable_thinking"))
    if mode != swe_agentic.AGENTIC:
        # Every instance: run_eval_suite applies --shard, then the --eval-limit draw.
        samples = swe_samples_from_rows(rows, setting, None, "swe_bench_live",
                                        setting_dataset_label(setting))
        result = run_eval_suite(
            eval_name="swe_bench_live", model_name=model_name, base_url=base_url,
            concurrency=concurrency, samples=samples, async_eval_fn=scorer, thinking=thinking,
            extra_payload=kwargs.get("extra_payload"), limit=kwargs.get("limit"),
            limit_seed=SWE_LIMIT_SEED,
            max_output_tokens=kwargs.get("max_output_tokens", 8192),
            temperature=kwargs.get("temperature"))
        result.update(metrics)
        for reason in launch_report_reasons(result, metrics) + _leaderboard_totals(setting,
                                                                                    result):
            swe_agentic.add_noncanon(result, reason)
        _oneshot_harness_errors(result)
        return swe_agentic.mark_oneshot(result, "swe_bench_live")

    start = time.time()
    limit = kwargs.get("limit")
    tasks = [swe_agentic.AgentTask(instance_id=r["instance_id"],
                                   problem_statement=r["problem_statement"],
                                   image=r["docker_image"], category=str(r.get("repo")),
                                   meta={"split": r["_split"],
                                         "language": r.get("language")})
             for r in unique_instance_rows(rows)]
    tasks, shard = swe_agentic.select_tasks(tasks, limit, "swe_bench_live")
    temperature, temperature_source = resolve_official_sampling(
        "swe_bench_live", WIN_AGENT_SAMPLING, thinking=thinking)
    budget = resolve_token_budget("swe_bench_live", WIN_AGENT_SAMPLING, thinking=thinking,
                                  max_output_tokens=kwargs.get("max_output_tokens"))
    penalties = decoding_penalties()
    body: Dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": thinking}}
    body["temperature"] = temperature
    body.update(penalties or {})
    run_max_tokens = swe_agentic.agent_max_tokens_from_run(kwargs.get("max_output_tokens"))
    body = swe_agentic.with_run_max_tokens(body, run_max_tokens)
    timeouts = swe_agentic.agent_request_timeouts(run_max_tokens)
    out_dir = swe_agentic.agent_output_dir("swe_bench_live")
    relay = EndpointRelay(base_url, "127.0.0.1", key=_endpoint_key(), body_fields=body,
                          name="swe-live-win-agent-relay", timeout_s=timeouts.relay_s,
                          client_timeout_suspect_s=timeouts.suspect_s()).start()
    try:
        traces = run_win_agent(tasks, rows, model_name, relay.url, concurrency,
                               out_dir, relay=relay) if tasks else []
    finally:
        relay.stop()
    agent = {"scaffold": "Win-Agent", "commit": WIN_AGENT_COMMIT,
             "config": "config/default.yaml (dataset, model, base_url and workers set by "
                       "gbench)",
             "model": f"openai/{model_name}", "relay_timeout_s": timeouts.relay_s}
    client = swe_agentic.model_client_timeout(
        "litellm (Win-Agent src/llm.py, REQUEST_TIMEOUT in its environment)",
        swe_agentic._package_version("litellm"),
        float(win_agent_request_timeout(timeouts.client_s)), WIN_AGENT_CLIENT_RETRIES,
        timeouts.source + "; transport retries: the OpenAI SDK default, which Win-Agent's "
        "litellm call does not set and no environment variable reaches (the client timeout is "
        "above the relay's safety cap on a model request, so they retry only an error the relay "
        "answers)",
        replaces="litellm's fallback client timeout of 600 s (COMPLETION_HTTP_FALLBACK_SECONDS)")
    attempts = int(kwargs.get("attempt_count") or get_run_knob("attempt_count", 1) or 1)
    return _agent_tab_result(
        setting, model_name=model_name, thinking=thinking, traces=traces, scorer=scorer,
        metrics=metrics, sampling=WIN_AGENT_SAMPLING, temperature=temperature,
        temperature_source=temperature_source, budget=budget, agent=agent, relay=relay,
        out_dir=out_dir, limit=limit, limit_note=kwargs.get("limit_note"), shard=shard,
        penalties=penalties, attempts=attempts,
        start=start, finalize=lambda res, m: launch_report_reasons(res, m),
        model_client=client, run_max_tokens=run_max_tokens)


def _endpoint_key() -> Optional[str]:
    """The served endpoint's key (base.endpoint_api_key: VLLM_API_KEY, else OPENAI_API_KEY), or
    None when it has none; gbench's relay adds it outside the agent containers."""
    from .base import endpoint_api_key
    return endpoint_api_key()


# --------------------------------------------------------------------------------------------
# one setting, and the combined result
# --------------------------------------------------------------------------------------------
def _run_setting(setting: str, mode: str, model_name: str, base_url: str, concurrency: int,
                 kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """One leaderboard setting, end to end: its rollouts, its harness pass, its result."""
    family = setting_family(setting)
    if family == "multilang":
        return _run_multilang(setting, mode, model_name, base_url, concurrency, kwargs)
    if family == "windows":
        return _run_windows(setting, mode, model_name, base_url, concurrency, kwargs)
    return _run_python(setting, mode, model_name, base_url, concurrency, kwargs)


def infra_required_setting(setting: str, reason: str) -> Dict[str, Any]:
    """The entry of a selected setting whose prerequisite is missing while other settings run:
    not run, not scored, the missing prerequisite named."""
    return {"status": "infra_required", "tab": TAB_LABELS[setting],
            "dataset": setting_dataset_label(setting),
            "error": (f"[INFRA REQUIRED] swe_bench_live {TAB_LABELS[setting]} tab cannot run "
                      f"here and was not scored: {reason} See '{DOCS_URL}'."),
            "infra_required": reason,
            "leaderboard_comparable": False,
            "leaderboard_comparable_reason": (f"the {TAB_LABELS[setting]} tab was not run "
                                              "(missing prerequisite, infra_required)"),
            "total_questions": 0, "correct_answers": 0, "accuracy": None}


def _grouped_reasons(reasons: Dict[str, str], ran: Sequence[str]) -> str:
    """Per-setting non-comparability reasons as one string: a reason every setting shares is
    written once, one that only some share names them."""
    order: List[str] = []
    where: Dict[str, List[str]] = {}
    for setting in ran:
        for part in (p.strip() for p in (reasons.get(setting) or "").split("; ")):
            if part:
                if part not in where:
                    order.append(part)
                    where[part] = []
                where[part].append(setting)
    return "; ".join(part if len(where[part]) == len(ran) else
                     f"{part} ({', '.join(where[part])})" for part in order)


def _scored_total(res: Dict[str, Any]) -> int:
    """A setting's resolved-rate denominator: its scored instances (rows when the headline is
    over rows), i.e. total_questions less the harness errors left out of the score."""
    if res.get("scored_instances") is not None:
        return int(res["scored_instances"])
    return int(res.get("total_questions") or 0) - int(res.get("scoring_excluded") or 0)


def combine_settings(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """One suite result from the per-setting results (keys in leaderboard order).

    `settings.<name>` holds each setting's result without its traces; `headline_metrics` has
    every setting's resolved rate (None for a setting that errored or was not run for a missing
    prerequisite); `accuracy`, `total_questions` and `correct_answers` are HEADLINE_SETTING's
    when it ran (else the first setting that ran, i.e. was not `infra_required`);
    `category_accuracy` is per setting; `sample_traces` holds every setting's traces, each
    marked with its `setting`. A setting with status `infra_required` is listed in
    `settings_infra_required` and makes the status `completed_with_errors`."""
    ran = list(results)
    run = [s for s in ran if results[s].get("status") != "infra_required"]
    head = (HEADLINE_SETTING if HEADLINE_SETTING in run else
            run[0] if run else HEADLINE_SETTING if HEADLINE_SETTING in results else ran[0])
    first = results[head]
    combined: Dict[str, Any] = {k: v for k, v in first.items() if k not in _PER_SETTING_FIELDS}
    settings: Dict[str, Dict[str, Any]] = {}
    traces: List[Dict[str, Any]] = []
    for setting in ran:
        res = results[setting]
        settings[setting] = {k: v for k, v in res.items() if k != "sample_traces"}
        offset = len(traces)
        for tr in res.get("sample_traces") or []:
            tr = dict(tr)
            tr["setting"] = setting
            for key in ("sample_idx", "source_sample_idx"):
                if isinstance(tr.get(key), int):
                    tr[key] += offset
            traces.append(tr)

    def ok(res: Dict[str, Any]) -> bool:
        return res.get("status") not in ("error", "infra_required")

    combined.update({
        "leaderboard_settings": ran,
        "headline_setting": head,
        "dataset": "; ".join(f"{s}: {results[s].get('dataset') or setting_dataset_label(s)}"
                             for s in ran),
        "total_questions": first.get("total_questions", 0) if ok(first) else 0,
        "scored_instances": _scored_total(first) if ok(first) else 0,
        "effective_n": _scored_total(first) if ok(first) else 0,
        "correct_answers": first.get("correct_answers", 0) if ok(first) else 0,
        "accuracy": first.get("accuracy") if ok(first) else None,
        "headline_metrics": {f"{s}_resolved_rate": (results[s].get("accuracy")
                                                    if ok(results[s]) else None) for s in ran},
        "headline_metrics_note": (
            "resolved rate (%) of each leaderboard tab: resolved instance ids / rows of the "
            "tab, the leaderboard's resolved / total"),
        "category_accuracy": {
            s: {"correct": results[s].get("correct_answers", 0),
                "total": _scored_total(results[s]),
                "accuracy": results[s].get("accuracy")}
            for s in ran if ok(results[s])},
        "settings": settings,
        "sample_traces": traces,
    })
    if "total_instance_ids" in first and ok(first):
        combined["total_instance_ids"] = first["total_instance_ids"]
    if any("answer_in_thinking" in results[s] for s in ran):
        # every tab's model turns with the answer inside the thinking channel, added (base's
        # ANSWER_IN_THINKING_POLICY); each tab keeps its own under settings.<name>
        from .base import answer_in_thinking_fields
        combined.update(answer_in_thinking_fields(
            sum(int(results[s].get("answer_in_thinking") or 0) for s in ran),
            sum(int(results[s].get("answer_in_thinking_of") or 0) for s in ran)))
    for field in ("tool_call_dropped_turns", "unclassified_empty_turns"):
        # every tab's empty turns that are not the model's answer (relay_result_reasons), added
        if any(field in results[s] for s in ran):
            combined[field] = sum(int(results[s].get(field) or 0) for s in ran)
    errored = [s for s in ran if results[s].get("status") == "error"]
    missing = [s for s in ran if results[s].get("status") == "infra_required"]
    if len(errored) + len(missing) == len(ran):
        combined["status"] = "error"
        combined["error"] = "; ".join(f"{s}: {results[s].get('error')}" for s in ran)
    elif errored or missing or any(results[s].get("status") == "completed_with_errors"
                                   for s in ran):
        combined["status"] = "completed_with_errors"
        if errored:
            combined["settings_with_errors"] = errored
    else:
        combined["status"] = "success"
    if missing:
        # Each such setting's entry names its missing prerequisite (infra_required_setting).
        combined["settings_infra_required"] = missing
    if combined.get("accuracy") is None:
        combined["accuracy_withheld_reason"] = (
            f"the headline setting {head} has no score: "
            + str(first.get("accuracy_withheld_reason") or first.get("error")
                  or first.get("status")))
    reasons = {s: results[s].get("leaderboard_comparable_reason") or
               ("" if results[s].get("leaderboard_comparable") else
                "not comparable (no reason recorded)") for s in ran}
    comparable = all(results[s].get("leaderboard_comparable") for s in ran)
    combined["leaderboard_comparable"] = comparable
    if not comparable:
        combined["leaderboard_comparable_reason"] = _grouped_reasons(reasons, ran)
    return combined


def _is_infra_required(exc: BaseException) -> bool:
    return str(exc).startswith("[INFRA REQUIRED]")


def missing_prerequisites(settings: Sequence[str], mode: str) -> Dict[str, str]:
    """setting -> its missing prerequisite, for each selected setting that cannot run here.

    The rule applies per setting and in both directions: the Windows tab needs a Windows host
    (check_windows_prerequisites), the Python and MultiLang tabs need the Linux images (the
    harness image with their family's labels, and in the agentic mode the SWE-agent-for-eval
    image and the docker socket), and every setting its pinned snapshots. The Windows tab also
    needs check_windows_metadata_guard to pass (the cloud metadata server is unreachable from its
    Windows task containers), checked last. When the Windows tab is selected with others, its
    prerequisite is checked first on its own. Then one check of the
    remaining selection (check_swe_bench_live_prerequisites, which covers the Windows tab when
    it is selected alone); only when that fails, and more than one setting remains, is each of
    them checked alone, to name the settings that lack what the selection lacks."""
    missing: Dict[str, str] = {}
    windows = [s for s in settings if setting_family(s) == "windows"]
    if windows and len(windows) < len(settings):
        ok, why = check_windows_prerequisites()
        if not ok:
            missing.update({s: why for s in windows})
    runnable = [s for s in settings if s not in missing]
    if runnable:
        ok, reason = check_swe_bench_live_prerequisites(runnable)
        if not ok and len(runnable) == 1:
            missing[runnable[0]] = reason
        elif not ok:
            for setting in runnable:
                ok_one, why = check_swe_bench_live_prerequisites([setting])
                if not ok_one:
                    missing[setting] = why
    linux = [s for s in settings if s not in missing and setting_family(s) != "windows"]
    if mode == swe_agentic.AGENTIC and linux:
        ok, reason = check_sweagent_prerequisites()
        if not ok:
            why = (f"{reason} The agentic protocol (the default) runs SWE-agent-for-eval on the "
                   "Python and MultiLang tabs; see the doc for the one-shot opt-in "
                   f"{swe_agentic.mode_env_var('swe_bench_live')}=oneshot and what it does not "
                   "measure.")
            missing.update({s: why for s in linux})
    if any(setting_family(s) == "windows" and s not in missing for s in settings):
        ok, why = check_windows_metadata_guard()
        if not ok:
            missing.update({s: why for s in settings
                            if setting_family(s) == "windows" and s not in missing})
    return {s: missing[s] for s in settings if s in missing}


def _no_setting_can_run(missing: Dict[str, str]) -> str:
    """The hard error's reason when no selected setting can run: each distinct missing
    prerequisite once, with the tabs it stops."""
    by_reason: Dict[str, List[str]] = {}
    for setting, why in missing.items():
        by_reason.setdefault(why, []).append(TAB_LABELS[setting])
    if len(by_reason) == 1:
        return next(iter(by_reason))
    return "no selected setting can run here. " + " ".join(
        f"{', '.join(tabs)}: {why}" for why, tabs in by_reason.items())


def run_swe_bench_live(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run SWE-bench-Live's leaderboard tabs (all ten by default): rollout with each tab's
    reference scaffold (agentic by default), score with each tab's upstream harness, one
    headline per tab. A selected setting whose prerequisite is missing here
    (missing_prerequisites: the Windows tab off a Windows host, the Linux tabs on one, a
    snapshot that cannot be fetched) is reported infra_required in its own entry while the
    other selected settings run; when no selected setting can run, or when the Linux tabs' cloud
    metadata server gate fails (before the first Linux tab), the whole suite hard-errors."""
    settings = resolve_settings(kwargs.pop("split", None), kwargs.pop("eval_categories", None))
    mode, _ = swe_agentic.resolve_mode("swe_bench_live")
    missing = missing_prerequisites(settings, mode)
    if len(missing) == len(settings):
        raise infra_required("swe_bench_live", _no_setting_can_run(missing), DOCS_URL)
    for setting, why in missing.items():
        logger.error("swe_bench_live: the %s tab cannot run here and is reported as "
                     "infra_required; the other selected settings run. %s",
                     TAB_LABELS[setting], why)
    # --eval-limit N is N instances for the whole suite, dealt over the tabs that can run here
    # (limit_shares); a tab whose share is 0 is not run in that limited run.
    limit = kwargs.get("limit")
    shares, plan = limit_shares(limit, [s for s in settings if s not in missing])
    not_run = [s for s, n in shares.items() if n == 0]
    if any(setting_family(s) != "windows" for s in settings
           if s not in missing and s not in not_run):
        # Every Linux tab runs model-controlled code in task containers with network access
        # (the rollouts, then the tests over the patches): the cloud metadata server must be
        # blocked there before the first tab starts. A host that fails it fails every Linux tab.
        agentic = mode == swe_agentic.AGENTIC
        images = [_image()] + ([sweagent_image()] if agentic else [])
        swebench_common.require_task_network_blocked(
            "swe_bench_live", images, docs_url=DOCS_URL,
            hint=f"Build the harness image {_image()} (see {DOCS_URL}).",
            resolve=SWEREX_BUILD_HOSTS if agentic else ())
    kwargs["enable_thinking"] = enable_thinking
    runnable = [s for s in settings if s not in missing and s not in not_run]
    instances = {s: (setting_rows(s) if shares.get(s) is None else int(shares[s]))
                 for s in runnable}
    schedule = tab_schedule(runnable, instances, concurrency)
    tab_results = run_tab_schedule(schedule, mode, model_name, base_url, kwargs, limit, shares)
    results: Dict[str, Dict[str, Any]] = {}
    for setting in settings:
        if setting in missing:
            results[setting] = infra_required_setting(setting, missing[setting])
        elif setting in tab_results:
            results[setting] = tab_results[setting]
    combined = combine_settings(results)
    combined["eval_limit_plan"] = plan
    if runnable:
        combined["tab_schedule"] = schedule
    if not_run:
        combined["settings_not_run_for_limit"] = not_run
        swe_agentic.add_noncanon(combined, f"--eval-limit {limit} for the suite gives "
                                 f"{', '.join(TAB_LABELS[t] for t in not_run)} no instance "
                                 "(not run)")
    if mode != swe_agentic.AGENTIC:
        swe_agentic.mark_oneshot(combined, "swe_bench_live")
    return combined


#: How a run schedules the tabs it runs (tab_schedule): tabs in different lanes run at once, the
#: tabs of one lane one after the other. The Python tabs (Lite, Full, Verified) share one lane:
#: their instance sets overlap (Lite and 457 of Verified's 500 ids are in Full, 174 ids are in
#: both Lite and Verified), and the fork harness names each task container
#: sweb.eval.<instance id>.<run id> with one run id for the suite
#: (swebench_common.make_swebench_scorer), so two Python tabs scoring one instance at once would
#: collide. Each MultiLang tab has a lane of its own (its splits share no instance id with any
#: other tab; upstream's multi-language harness names every task container with a uuid), as has
#: the Windows tab.
def tab_lanes(settings: Sequence[str]) -> List[List[str]]:
    """The lanes of `settings` (in their order): one lane for every Python tab, one lane per
    other tab."""
    lanes: List[List[str]] = []
    python: List[str] = []
    for s in settings:
        if setting_family(s) == "python":
            if not python:
                lanes.append(python)
            python.append(s)
        else:
            lanes.append([s])
    return lanes


def tab_schedule(settings: Sequence[str], instances: Dict[str, int],
                 concurrency: int) -> Dict[str, Any]:
    """How the tabs `settings` run within `concurrency` (--sandboxes): their lanes (tab_lanes)
    run at once, at most `concurrency` of them, and the budget is dealt over the lanes so that
    the rollouts (SWE-agent's --num_workers, Win-Agent's workers) and harness workers of the
    tabs running at one time never add up to more than `concurrency`. Each lane gets at least
    one, at most as many as its largest tab has instances (`instances`: tab -> instances it
    runs), and the rest goes, one at a time, to the lane with the most instances per worker.
    With fewer workers than lanes, each lane gets one and `concurrency` lanes run at a time.
    Returns {"concurrency", "concurrent_lanes", "lanes", "workers": {tab: n}}; every tab of a
    lane runs with that lane's workers as its concurrency."""
    budget = max(1, int(concurrency or 1))
    lanes = tab_lanes(settings)
    alloc = [1] * len(lanes)
    if len(lanes) <= budget:
        work = [sum(max(0, int(instances.get(t) or 0)) for t in lane) for lane in lanes]
        cap = [max(1, max(int(instances.get(t) or 0) for t in lane)) for lane in lanes]
        for _ in range(budget - len(lanes)):
            open_ = [i for i in range(len(lanes)) if alloc[i] < cap[i]]
            if not open_:
                break
            best = max(open_, key=lambda i: (work[i] / alloc[i], -i))
            alloc[best] += 1
    return {"concurrency": budget, "concurrent_lanes": min(len(lanes), budget),
            "lanes": [list(lane) for lane in lanes],
            "workers": {t: alloc[i] for i, lane in enumerate(lanes) for t in lane}}


def _setting_could_not_run(setting: str, e: BaseException) -> Dict[str, Any]:
    logger.error("swe_bench_live: the %s setting could not run: %s: %s", setting,
                 type(e).__name__, e, exc_info=True)
    return {"status": "error", "error": f"{type(e).__name__}: {e}",
            "leaderboard_comparable": False,
            "leaderboard_comparable_reason": "setting could not run",
            "total_questions": 0, "correct_answers": 0, "accuracy": None}


#: How often (seconds) run_tab_schedule looks for a stop handed to the thread running the suite,
#: to hand it to the tab threads.
TAB_STOP_POLL_S = 1.0


class TabScheduleStopped(SystemExit):
    """The SystemExit (code 128 + the signal, as the CLI's handler and _raise_if_stop_requested
    raise) run_tab_schedule raises when a SIGTERM/SIGHUP stopped the suite while its tabs ran.

    `record` is the suite's failed record (a harness error, no score, as evals gives a suite a
    signal cut short) naming the signal and every tab the stop halted: `tabs_cut_short` (running
    when the stop arrived; whatever they did afterwards is not a result), `tabs_not_started`, and
    their union in schedule order, `tabs_stopped`; `tabs_completed` ended before the stop.
    `results` holds the completed tabs' results, which are not reported as the suite's."""

    def __init__(self, signum: int, record: Dict[str, Any],
                 results: Dict[str, Dict[str, Any]]) -> None:
        super().__init__(128 + int(signum))
        self.signum = int(signum)
        self.record = record
        self.results = results


def tab_stop_record(signum: int, tabs: Sequence[str], completed: Sequence[str],
                    cut_short: Sequence[str]) -> Dict[str, Any]:
    """The failed suite record of a run_tab_schedule a signal stopped (TabScheduleStopped):
    `tabs` the scheduled tabs in order, `completed` those that ended before the stop,
    `cut_short` those running when it arrived; every other tab never started."""
    import signal as _signal
    try:
        name = _signal.Signals(signum).name
    except ValueError:
        name = f"signal {signum}"
    done = [t for t in tabs if t in set(completed) and t not in set(cut_short)]
    cut = [t for t in tabs if t in set(cut_short)]
    not_started = [t for t in tabs if t not in set(done) and t not in set(cut)]
    stopped = [t for t in tabs if t in set(cut) or t in set(not_started)]
    labels = lambda ts: ", ".join(TAB_LABELS[t] for t in ts) or "none"    # noqa: E731
    return {"status": "error", "failed": True, "failure_kind": "error",
            "stopped_by_signal": int(signum),
            "error": (f"terminated by signal {signum} ({name}) while the tabs ran (the run was "
                      f"stopped from outside, e.g. an outer timeout or a scheduler); no score. "
                      f"Tabs stopped: {labels(stopped)} (cut short: {labels(cut)}; not started: "
                      f"{labels(not_started)}); completed before the stop: {labels(done)}"),
            "tabs_stopped": stopped, "tabs_cut_short": cut, "tabs_not_started": not_started,
            "tabs_completed": done, "total_questions": 0, "correct_answers": 0,
            "accuracy": None, "leaderboard_comparable": False,
            "leaderboard_comparable_reason": "the run was stopped by a signal while the tabs ran",
            "accuracy_withheld_reason": "the run was stopped by a signal while the tabs ran"}


def run_tab_schedule(schedule: Dict[str, Any], mode: str, model_name: str, base_url: str,
                     kwargs: Dict[str, Any], limit: Optional[int],
                     shares: Dict[str, Optional[int]]) -> Dict[str, Dict[str, Any]]:
    """Run the tabs of `schedule` (tab_schedule): its lanes at once, in threads, at most
    `concurrent_lanes` of them, the tabs of a lane one after the other, each tab exactly as a
    run of that tab alone (_run_setting: its own rollouts, relay, watchdog, harness pass and
    headline) with its lane's workers as concurrency. Returns {tab: result}.

    A tab that raises keeps the other tabs' results (its own result is an error), except a
    missing prerequisite (infra_required: a hard error for the suite, as a missing one is) and
    a stop by SIGTERM/SIGHUP (SystemExit, _raise_if_stop_requested): then no further tab is
    started, the tabs already running are waited for (each removes its containers as it ends),
    and TabScheduleStopped is raised, naming every tab the stop halted.

    A stop recorded for the calling thread (process_groups.stop_requested: the run's, or the one
    evals._run_with_timeout hands to a suite run in a worker thread under --suite-timeout) is
    handed to every running tab thread within TAB_STOP_POLL_S (process_groups.request_stop),
    so each running tab's watchdog (SweagentWatch) stops its SWE-agent batch, the rollouts it
    cut short are not scored, and no further tab starts. Each tab runs with
    swe_thread_cap.SUITE_CONTAINERS set to the whole budget, so the per-container thread cap
    shares the box over every lane's containers, as a run of the tabs one after the other, each
    with the whole budget, did.

    Whenever a stop reached the schedule (handed over as above, raised in this thread, or raised
    by a tab), the schedule never returns normally, even when every running tab was in harness
    scoring (which does not look for a stop) and ended without raising: once the pool has shut
    down it logs and raises TabScheduleStopped (SystemExit 128 + the signal) whose record names
    the signal and every tab the stop halted (cut short or never started), so no halted tab goes
    missing from a result that looks complete and the run stops, as the serial run did."""
    import concurrent.futures
    import contextvars
    import threading

    from ...utils import process_groups
    caller = threading.get_ident()
    #: ident -> lane of every tab thread running a lane now
    tab_threads: Dict[int, List[str]] = {}
    used_threads: set = set()
    handed_to: set = set()
    lanes: List[List[str]] = schedule["lanes"]
    workers: Dict[str, int] = schedule["workers"]
    results: Dict[str, Dict[str, Any]] = {}
    fatal: List[BaseException] = []
    lock = threading.Lock()
    halt = threading.Event()
    #: ident -> the tab that thread runs now; the tabs running when the stop was handed over
    running: Dict[int, str] = {}
    cut_short: List[str] = []
    stop_sig: List[int] = []
    if len(lanes) > 1:
        logger.info("swe_bench_live: %d tab lanes, at most %d at a time within a concurrency "
                    "of %d (--sandboxes): %s", len(lanes), schedule["concurrent_lanes"],
                    schedule["concurrency"],
                    "; ".join(" then ".join(f"{TAB_LABELS[t]} ({workers[t]} worker(s))"
                                            for t in lane) for lane in lanes))
    # Every tab's pinned snapshots were fetched and checked, one at a time, by the prerequisite
    # check (missing_prerequisites) before this; the tabs only read them.

    def _lane(lane: List[str]) -> None:
        for setting in lane:
            if halt.is_set():
                return
            _raise_if_stop_requested(f"before the {TAB_LABELS[setting]} tab")
            with lock:
                if halt.is_set():
                    return
                running[threading.get_ident()] = setting
            tab_kwargs = dict(kwargs)
            if limit:
                tab_kwargs["limit"] = shares[setting]
                tab_kwargs["limit_note"] = limit_note(limit, setting, shares)
            try:
                res = _run_setting(setting, mode, model_name, base_url, workers[setting],
                                   tab_kwargs)
            except Exception as e:
                if not _is_infra_required(e):
                    # Anything else keeps the other settings' results.
                    res = _setting_could_not_run(setting, e)
                else:
                    # A prerequisite that disappeared mid-run is a hard error, like a missing
                    # one.
                    with lock:
                        logger.error("swe_bench_live: the %s setting hit a missing "
                                     "prerequisite; the settings already run were %s; no "
                                     "further tab starts", setting,
                                     {s: r.get("accuracy") for s, r in results.items()})
                    raise
            with lock:
                results[setting] = res
                running.pop(threading.get_ident(), None)

    def _guarded(lane: List[str]) -> None:
        me = threading.get_ident()
        with lock:
            tab_threads[me] = lane
            used_threads.add(me)
        try:
            _lane(lane)
        except BaseException as e:
            with lock:
                fatal.append(e)
                if me in running and isinstance(e, SystemExit):
                    # stopped by a signal it saw itself (_raise_if_stop_requested)
                    cut_short.append(running[me])
                halt.set()
            raise
        finally:
            with lock:
                tab_threads.pop(me, None)
                running.pop(me, None)

    def _hand_stop(signum: int) -> None:
        # No further tab starts; every running tab thread gets the stop as its own (its
        # watchdog, SweagentWatch._owner, and _raise_if_stop_requested look up that thread).
        with lock:
            halt.set()
            if not stop_sig:
                stop_sig.append(int(signum))
            for tab in running.values():
                if tab not in cut_short:
                    cut_short.append(tab)
            new = {tid: lane for tid, lane in tab_threads.items() if tid not in handed_to}
            handed_to.update(new)
        for tid in new:
            process_groups.request_stop(signum, thread=tid)
        if new:
            logger.error("swe_bench_live: the run was stopped by signal %d; handed to the "
                         "running tab lane(s) %s; no further tab starts", signum,
                         "; ".join(" then ".join(TAB_LABELS[t] for t in lane)
                                   for lane in new.values()))

    def _tab_context() -> contextvars.Context:
        ctx = contextvars.copy_context()
        # the per-container thread cap shares the box over the whole suite's containers
        ctx.run(swe_thread_cap.SUITE_CONTAINERS.set, int(schedule["concurrency"]))
        return ctx

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, schedule["concurrent_lanes"]),
            thread_name_prefix="swe-live-tab") as ex:
        # submitted inside the try: a signal that arrives as soon as a tab runs is handled too
        futures: List[concurrent.futures.Future] = []
        interrupted: Optional[BaseException] = None
        try:
            try:
                for lane in lanes:
                    futures.append(ex.submit(_tab_context().run, _guarded, lane))
                pending = set(futures)
                handed = False
                while pending:
                    _, pending = concurrent.futures.wait(pending, timeout=TAB_STOP_POLL_S)
                    stop = process_groups.stop_requested(caller)
                    if stop:
                        # Re-hand on every poll: a tab thread that took up a lane since then
                        # (it sees `halt` and starts nothing) gets it too.
                        _hand_stop(stop[0])
                        if not handed:
                            handed = True
                            for f in futures:
                                f.cancel()
            except BaseException as e:
                # The CLI's SIGTERM/SIGHUP handler (SystemExit) or Ctrl-C in this thread: start
                # no further tab, hand the stop to the running ones; leaving the pool waits for
                # them, and they end with their batches and remove their containers.
                interrupted = e
                stop = process_groups.stop_requested(caller)
                if stop is None and isinstance(e, SystemExit) and isinstance(e.code, int) \
                        and 128 < e.code < 128 + 65:
                    stop = (e.code - 128, 0.0)
                if stop:
                    _hand_stop(stop[0])
                halt.set()
                for f in futures:
                    f.cancel()
        finally:
            ex.shutdown(wait=True)
            for tid in used_threads:
                # pool thread idents are reused by later threads: forget what was handed over
                process_groups.clear_stop(tid)
    if not stop_sig:
        # a tab that saw the stop itself (the run's, _raise_if_stop_requested) and raised
        for e in fatal:
            if isinstance(e, SystemExit) and isinstance(e.code, int) and 128 < e.code < 128 + 65:
                stop_sig.append(e.code - 128)
                break
    if stop_sig:
        tabs = [t for lane in lanes for t in lane]
        record = tab_stop_record(stop_sig[0], tabs, list(results), cut_short)
        logger.error("swe_bench_live: %s", record["error"])
        cause = interrupted or (fatal[0] if fatal else None)
        raise TabScheduleStopped(stop_sig[0], record,
                                 {t: results[t] for t in record["tabs_completed"]}) from cause
    if interrupted is not None:
        raise interrupted
    if fatal:
        raise fatal[0]
    return results


def limit_shares(limit: Optional[int], runnable: Sequence[str]
                 ) -> Tuple[Dict[str, Optional[int]], Dict[str, Any]]:
    """({tab: instances it runs}, the run's eval_limit_plan) for the tabs that can run here.

    `--eval-limit N` is N instances for the whole suite, never N per tab: sampling.allocate_limit
    deals them over the tabs (seed "swe_bench_live"; each tab capped at its pinned rows), so a
    tab gets 0 when N is below the number of tabs and is then not run. Without a limit every tab
    runs every instance (None). The plan is logged once, here."""
    # the headline tab (HEADLINE_SETTING, Lite) gets the first instance of a limited run
    shares = allocate_limit(limit, list(runnable),
                            available={s: setting_rows(s) for s in runnable},
                            seed="swe_bench_live", first=(HEADLINE_SETTING,))
    per_tab = {s: (setting_rows(s) if n is None else n) for s, n in shares.items()}
    plan = eval_limit_plan("swe_bench_live", limit, sum(per_tab.values()), per_tab,
                           setting_noun="tab", attempts=1, unit="rollouts")
    return shares, plan


def limit_note(limit: int, setting: str, shares: Dict[str, Optional[int]]) -> str:
    """A limited tab's non-comparability reason: the suite's --eval-limit and this tab's share."""
    ran = sum(1 for n in shares.values() if n)
    return (f"subset run (--eval-limit {limit} for the suite, dealt over {ran} tab(s): "
            f"{shares[setting]} {TAB_LABELS[setting]} instance(s))")
