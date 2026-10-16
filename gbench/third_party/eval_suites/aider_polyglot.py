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

# Evaluation Suite: aider_polyglot
# Description: Aider Polyglot (native-edit-format code editing across 6 languages, pass@2)

"""gbench native built-in runner for aider_polyglot (Coding & Software Engineering).

Canonical Aider Polyglot (Aider-AI/aider `benchmark/benchmark.py`) drives aider's own Coder over
225 Exercism exercises in 6 languages (C++, Go, Java, JavaScript, Python, Rust). The model edits
the stub in its NATIVE edit format (diff / SEARCH-REPLACE for capable models, whole-file
otherwise); aider applies the edit, runs the exercise's hidden unit tests, and on failure feeds
the errors back for a second try. The headline is **pass@2**.

gbench delegates to aider's OWN benchmark harness inside aider's OWN benchmark container
(`aider-benchmark:5dc9490b`, built from docker/aider_polyglot.Dockerfile: aider's `benchmark/Dockerfile`
at AIDER_COMMIT plus a setuptools-scm version and jest pinned to the exercises' 29.7.0),
for two reasons:

* aider + litellm would churn the torch/vLLM serving env (installing aider downgrades `openai`
  and moves `numpy`/`huggingface_hub`/`pydantic`/`pillow` - it breaks the pinned graph), so it is
  NOT installed into the serving venv; it runs in the isolated image.
* the image also carries the 6 language toolchains (go/rust/openjdk-21/node/gcc/python) needed to
  execute the exercises' unit tests.

gbench generates nothing itself here: it `docker run`s the image on docker's bridge network with
this host reachable at host.docker.internal (as upstream's benchmark/docker.sh does; never the
host network, since the exercises' tests run the model's code in that container and the host's
DOCKER-USER rule keeps the cloud metadata server out of reach of a bridge network only), points
aider at the gbench-served `/v1` endpoint via litellm (`OPENAI_API_BASE` + `--model
openai/<id>`; an endpoint on this host becomes host.docker.internal), lets aider run its
native-edit-format pass@2 loop, and reads back the per-exercise `.aider.results.json`. Before
the container starts, tau_common.require_endpoint_network checks that the metadata server is
blocked on that network (container_guard), that the probe container resolves the host names the
run needs there (the endpoint's host when it is a name, and the toolchain hosts of the run's
languages, TOOLCHAIN_HOSTS: the Java exercises download Gradle and Maven Central packages and
three Rust exercises crates at test time, as upstream's image does) and that an endpoint on this
host answers there; any of them failing is a hard error (infra_required). The container keeps
the image's HOME (/root), as upstream's benchmark/docker.sh does: the image installs Rust with
rustup under /root/.rustup and /root/.cargo and sets neither RUSTUP_HOME nor CARGO_HOME, so any
other HOME leaves cargo with no default toolchain. It also has upstream's memory limit
(CONTAINER_MEMORY_ARGS: `--memory=12g --memory-swap=12g`, as benchmark/docker.sh passes), so
model-written code that allocates without bound is killed inside the container, a failed test
run as upstream scores it, instead of taking memory from this host. The limit is shared by the
exercises benchmark.py runs at once (--threads: gbench's eval concurrency; upstream's
benchmark/README.md runs 10), so the container prints, last, how many processes the kernel's OOM
killer killed in its cgroup (memory.events oom_kill); the result records it (container_oom_kills)
and any kill, or a count that could not be read, makes the run not comparable, since a kill there
cannot be attributed to the exercise whose code used the memory. An exercise is a harness error
(exercise_harness_error) when one of its test runs' outputs (exercise_test_outputs) shows that a
language toolchain in the image could not run (toolchain) or that its tests could not reach one of
the toolchain hosts (toolchain_network), or when aider gave up on a model request after the endpoint
kept failing with a connection, DNS, 5xx, 429 or timeout error (model_endpoint, request_give_ups).
Such an exercise is left out of the score, counted and named, and the status is
completed_with_errors; when every exercise has one, no score is reported (infra_required). A
request aider gave up on for any other cause is the model's outcome, scored as upstream scores it
and counted (scored_request_give_ups).

Request timeout: aider gives each request 600 s (aider/models.py request_timeout; the openai
client makes up to 3 attempts of it) and sends no max_tokens, so a reply longer than that timed
out, and at a temperature of 0 every retry produced the
same reply and timed out again: benchmark.py retries for 24 h, so the run hung instead of scoring
it. gbench sizes the timeout as base does for every request without max_tokens
(base.request_timeout_s of the served context window, or of base.UNCAPPED_TIMEOUT_TOKENS when the
server reports none) and passes it in the settings file's extra_params (`timeout`, which
Model.send_completion uses instead of its default, with `max_retries` 0, so each of aider's
attempts is one HTTP attempt and aider's own loop does the retrying). A reply then ends at the window, which aider
counts itself (exhausted_context_windows), and a request that still times out means the endpoint
decoded below base.MIN_DECODE_TOK_S, an endpoint failure. The result records the value
(`model_request_timeout_s`). The model's
code keeps the container's general internet access, as upstream runs it; only the metadata server
is blocked. This suite hard-errors (never skips) if Docker, the image, or the exercises are
missing. Because it shells out, it does NOT use run_eval_suite.

Thinking: `--thinking` reaches aider's requests through a model-settings file passed with
`--read-model-settings`. It holds aider's special `aider/extra_params` entry, whose extra_params
aider deep-merges into every litellm call without touching the model's own settings (edit format
included), so each request carries `extra_body.chat_template_kwargs.enable_thinking` (true or
false, like every other gbench suite).

Sampling: gbench's rule for every suite (base.resolve_temperature): temperature 1.0 with
--thinking, 0.0 without; `--temperature`, or `GBENCH_AIDER_POLYGLOT_TEMPERATURE` for this suite
alone (it takes precedence), overrides it in either mode. aider's benchmark sets temperature 0
itself (aider/models.py Model.send_completion, use_temperature for an `openai/<id>` model), so
gbench always writes the resolved value in the settings file's extra_params (`temperature`), which
Model.send_completion applies with kwargs.update after its own temperature: it reaches every
litellm call in both modes, and the pre-flight confirms aider loaded it. The result records
`temperature`, `temperature_source`, `temperature_applied: true` and, under `official_sampling`,
aider's 0 and whether the run matched it; a temperature other than 0 makes the run not
leaderboard_comparable (the reason names it).
There is no LLM judge (grading is the exercises' unit tests, executed
in-container), so the usual pinned-0.0 judge note does not apply. aider's benchmark runs each
exercise once with `--tries 2`; `--attempt-count` above 1 is recorded as not applied
(`attempts.not_applied`).

Pinned sources: the image is built from aider at AIDER_COMMIT (docker/aider_polyglot.Dockerfile
fetches that commit and labels the image `org.gbench.aider.commit`; an image with another label is
infra_required, naming the build command), and the exercises checkout must be polyglot-benchmark at
POLYGLOT_BENCHMARK_COMMIT with no modified tracked file (`git -C <dir> rev-parse HEAD`; a missing
.git, another commit or a modified file is infra_required, naming the checkout command). Both are
recorded (`aider_commit`, `polyglot_benchmark_commit`).

Model id: the id gbench's eval runner passes (`model_name`, the served id it resolved against
--models) is sent as `openai/<id>` (GBENCH_AIDER_MODEL overrides the whole litellm model name); the
suite does not re-read the endpoint's model list.

Exercise selection (`--shard I/N`, `--eval-limit`): gbench selects the exercises on the host, never
with benchmark.py's `--num-tests` (which takes the first N after an unseeded random.shuffle, a
different subset every run). The run's languages' exercise directories
(`polyglot-benchmark/<lang>/exercises/practice/<name>`) are listed in sorted order, the shard is
taken first (base.apply_shard_indexed: round-robin over that order), then `--eval-limit` exercises
are drawn from it, stratified by language with a fixed seed (sampling.stratified_sample, seed
"aider_polyglot"). A subset is copied into a temporary exercises tree with the checkout's layout,
mounted read-only at /gbench-exercises and passed as `--exercises-dir`, so benchmark.py runs every
exercise it sees (its shuffle only orders them). A full run uses the checkout itself and keeps the
225-exercise gate; a subset run must finish every selected exercise. The result records
`selected_exercises`, the shard identity (base.shard_identity) and one trace per exercise
(`sample_traces`: `exercise`, `language`, `pass_1`, `pass_2`, `tries`, `request_give_ups`, a harness
error with `scoring_excluded`, `sample_key`, `global_idx`); an empty shard reports `accuracy` null
with `accuracy_withheld_reason`. merge_shards (gbench --merge) re-counts pass@1 / pass@2 over the
union of the shards' traces.

max_tokens: aider sends none, with and without --thinking (aider/models.py Model.send_completion
adds max_tokens only from a model's settings, and an `openai/<id>` model has none), so a thinking
run is never capped; `--max-output-tokens` is not applied, as base's token-limit policy records
for a harness that sends no max_tokens (`max_output_tokens_note`, read from the extra_params the
pre-flight reports; a model whose aider settings carry a max_tokens gets that value from aider).

A reply that runs to the end of the served context window is counted by aider itself:
aider/coders/base_coder.py send_message sets `exhausted` on finish_reason "length" (for a model
without assistant prefill, as an `openai/<id>` model is) or on litellm's
ContextWindowExceededError (a request the window refused), and adds one to
num_exhausted_context_windows, which benchmark.py writes into every `.aider.results.json` and
summarize_results sums into the `exhausted_context_windows` leaderboard column. gbench sums it
(`exhausted_context_windows`, also `replies_cut_off_by_budget`); any makes the run not comparable,
since those replies measure the served window, not the model. The result also carries aider's
other leaderboard columns (`aider_leaderboard_columns`: pass rates and counts, well-formed
percentage, error outputs, malformed responses, user asks, lazy comments, syntax and indentation
errors, test timeouts, token counts, seconds per case, cost, and the model, edit format and
commit seen), computed as summarize_results computes them over the scored exercises.
"""

import glob
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from .base import (SHARD_REASON, UNCAPPED_TIMEOUT_TOKENS, OfficialSampling,
                   add_comparability_reason, apply_shard_indexed, current_shard, get_run_knob,
                   request_timeout_s, resolve_temperature, resolve_token_budget, shard_identity)
from .limit_plan import eval_limit_plan
from .sampling import stratified_sample
from .swebench_common import infra_required, prereqs_path

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/aider_polyglot.md"
PILLAR = "Coding & Software Engineering"

_IMAGE_DEFAULT = "aider-benchmark:5dc9490b"
#: The polyglot benchmark's 6 languages.
_LANGUAGES = ["cpp", "go", "java", "javascript", "python", "rust"]
_EXPECTED_FULL = 225  # exercises in the canonical polyglot set across all 6 languages
#: The aider commit the image is built from (docker/aider_polyglot.Dockerfile ARG AIDER_COMMIT; the
#: image carries it as the label AIDER_COMMIT_LABEL, checked before every run).
AIDER_COMMIT = "5dc9490bb35f9729ef2c95d00a19ccd30c26339c"
AIDER_COMMIT_LABEL = "org.gbench.aider.commit"
#: The polyglot-benchmark commit the exercises checkout must be at (github.com/Aider-AI/
#: polyglot-benchmark, 225 exercises: cpp 26, go 39, java 47, javascript 49, python 34, rust 30).
POLYGLOT_BENCHMARK_COMMIT = "7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f"
POLYGLOT_BENCHMARK_REPO = "https://github.com/Aider-AI/polyglot-benchmark"
#: Where a selected subset of the exercises is mounted in the container (read-only), passed to
#: benchmark.py as an absolute --exercises-dir.
_EXERCISES_MOUNT = "/gbench-exercises"
#: The result's sampling note: gbench's temperature (1.0 with --thinking, 0.0 without, or the
#: operator's) replaces aider's own 0 on every request.
AIDER_SAMPLING_NOTE = ("gbench sends its temperature (1.0 with --thinking, 0.0 without, unless "
                       "--temperature / GBENCH_AIDER_POLYGLOT_TEMPERATURE sets one) in the "
                       "aider/extra_params settings (Model.send_completion applies extra_params "
                       "after its own temperature 0), so every request carries it")
#: aider's official temperature (Model.send_completion: 0 for an openai/<id> model), recorded
#: under `official_sampling` on every result.
AIDER_SAMPLING = OfficialSampling(
    reference=("github.com/Aider-AI/aider aider/models.py Model.send_completion: temperature 0 "
               "(use_temperature True for an openai/<id> model)"),
    temperature=0.0, send_max_tokens=False)
#: The benchmark container's memory limit, as upstream's benchmark/docker.sh runs the image:
#: `--memory=12g --memory-swap=12g` (12 GiB, no swap on top). The exercises' tests run the model's
#: code in that container, so code that allocates without bound is killed by the kernel inside
#: the container's cgroup (its test run fails, the model's outcome as upstream scores it) instead
#: of taking memory from this host and the model server on it. The limit covers the whole
#: container, shared by the exercises benchmark.py runs at once (--threads; UPSTREAM_THREADS in
#: upstream's documented run), so the container reports the OOM kills in it (_OOM_REPORT).
CONTAINER_MEMORY = "12g"
CONTAINER_MEMORY_ARGS = ("--memory", CONTAINER_MEMORY, "--memory-swap", CONTAINER_MEMORY)
#: The exit status of a container whose main process was killed with SIGKILL (128 + 9): the
#: kernel's OOM killer picked it when the container reached its memory limit, or `docker kill`.
_SIGKILL_EXIT = 137
#: The --threads of upstream's documented run (benchmark/README.md: `./benchmark/benchmark.py
#: ... --threads 10`, inside the container benchmark/docker.sh starts with the limit above).
#: gbench passes its eval concurrency instead (--sandboxes, else the first --batch-sizes value).
UPSTREAM_THREADS = 10
#: The line the container prints last (_OOM_REPORT), after benchmark.py has exited, and the
#: number on it: how many processes the kernel's OOM killer killed in the container's cgroup
#: (cgroup v2 memory.events `oom_kill`, which counts the cgroup and its descendants; cgroup v1
#: memory.oom_control `oom_kill`). Docker leaves memory.oom.group at 0, so at the limit the
#: kernel kills one process, the one using the most memory, typically an exercise's test or build
#: process: that test run fails and benchmark.py records it as it records any failed run, scored
#: as the model's outcome (a killed benchmark.py ends the container with exit 137). The limit
#: is shared by every exercise benchmark.py runs at once, so a kill cannot be attributed to the
#: exercise whose code used the memory; the result records the count (container_oom_kills) and
#: any kill, or a count that could not be read, makes the run not comparable.
_OOM_MARKER = "GBENCH_OOM_KILLS "
#: The report starts with an empty line, so it is a line of its own even when benchmark.py's
#: output does not end with a newline.
#: The line before it: the memory limit the kernel enforces on the container's cgroup (cgroup v2
#: memory.max, "max" when there is none; cgroup v1 memory.limit_in_bytes), empty when neither file
#: can be read. Docker drops --memory (with a warning) on a host without a memory cgroup
#: controller, so the requested flags alone do not say the limit held.
_MEMORY_MARKER = "CONTAINER_MEMORY_LIMIT "
_OOM_REPORT = (
    "; rc=$?; echo; cg=$(sed -n 's/^0:://p' /proc/self/cgroup 2>/dev/null); "
    "m=/sys/fs/cgroup${cg%/}/memory.max; [ -r \"$m\" ] || m=/sys/fs/cgroup/memory.max; "
    "[ -r \"$m\" ] || m=/sys/fs/cgroup/memory/memory.limit_in_bytes; "
    "echo \"" + _MEMORY_MARKER + "$(cat \"$m\" 2>/dev/null)\"; "
    "f=/sys/fs/cgroup${cg%/}/memory.events; [ -r \"$f\" ] || f=/sys/fs/cgroup/memory.events; "
    "[ -r \"$f\" ] || f=/sys/fs/cgroup/memory/memory.oom_control; "
    "echo \"" + _OOM_MARKER + "$(awk '/^oom_kill /{print $2}' \"$f\" 2>/dev/null)\"; "
    "exit $rc")
#: aider's request settings as far as max_tokens goes: none is sent (Model.send_completion builds
#: each litellm call from the model's settings and extra_params; gbench's extra_params carry only
#: chat_template_kwargs).
AIDER_TOKENS = OfficialSampling(
    reference=("github.com/Aider-AI/aider aider/models.py Model.send_completion: no max_tokens "
               "unless the model's settings set one (none for an openai/<id> model)"),
    temperature=0.0, send_max_tokens=False)
#: Where the gbench model-settings file is mounted inside the container.
_SETTINGS_MOUNT = "/gbench-aider"
_SETTINGS_FILE = "model-settings.yml"
#: Pre-flight run in the same container just before benchmark.py: it loads the settings file
#: with aider's own `models.register_models` (what benchmark.py's --read-model-settings calls)
#: and prints the extra_params aider's `Model` ends up with. register_models skips a file it
#: cannot read without an error, so without this nothing would confirm the file was applied.
_CHECK_FILE = "check_settings.py"
_CHECK_MARKER = "GBENCH_AIDER_SETTINGS "
_CHECK_SRC = """import json, sys
from aider import models
loaded = models.register_models([sys.argv[1]])
extra = models.Model(sys.argv[2]).extra_params if loaded else None
print("GBENCH_AIDER_SETTINGS " + json.dumps({"files_loaded": loaded, "extra_params": extra},
                                            default=str), flush=True)
sys.exit(0 if loaded else 3)
"""

#: The hosts the exercises' tests fetch from at run time, by language. Upstream's
#: benchmark/Dockerfile bakes no Gradle distribution, Maven package or crate into the image, and
#: the container (HOME /root, its Gradle and cargo caches included) is removed after the run, so
#: each run downloads them: every Java exercise's
#: gradle/wrapper/gradle-wrapper.properties names the Gradle 8.7 distribution on
#: services.gradle.org, and its build.gradle takes JUnit and AssertJ from mavenCentral()
#: (repo.maven.apache.org); three Rust exercises (gigasecond, grep, simple-cipher) depend on
#: crates, which cargo fetches through crates.io's sparse index (index.crates.io) and its download
#: host (static.crates.io). C++, Go, JavaScript (the image's /npm-install) and Python fetch
#: nothing. require_endpoint_network has the probe container resolve the names of the run's
#: languages before the benchmark starts (container_hosts).
TOOLCHAIN_HOSTS = {"java": ("services.gradle.org", "repo.maven.apache.org"),
                   "rust": ("index.crates.io", "static.crates.io")}
#: The hosts (and their subdomains) whose failed lookup or connection in an exercise's test output
#: is a harness error: TOOLCHAIN_HOSTS' domains and GitHub, where services.gradle.org redirects
#: the distribution download.
TOOLCHAIN_FAILURE_HOSTS = ("gradle.org", "maven.apache.org", "maven.org", "crates.io",
                           "github.com", "githubusercontent.com")
#: The file aider's benchmark.py keeps each exercise's chat in (the model's replies, aider's own
#: messages and every test run's output), next to its .aider.results.json.
_HISTORY_FILE = ".aider.chat.history.md"
#: A retry aider logs for a model request ("Retrying in <delay> seconds..."): send_message
#: (aider/coders/base_coder.py) retries a failed request (a connection error, a timeout, a 5xx, a
#: 429) with a delay that starts at 0.125 s and doubles, and stops once the next delay would pass
#: RETRY_TIMEOUT. benchmark.py raises RETRY_TIMEOUT to 24 h ("Don't give up when benchmarking":
#: sendchat, base_coder and models.RETRY_TIMEOUT = 24*60*60), so aider keeps retrying, the delays
#: doubling up to 65536 s (about 36 h in all), and "Retrying in 32.0 seconds..." is not its last
#: retry. The last retry line of a request (no later one before the reply, the usage line or the
#: next test run) is followed either by the reply (aider's `Tokens:` line) or by no reply: aider
#: logged the final failure (check_and_open_urls) and ended the turn, or the run ended while it
#: was still retrying (request_give_ups).
_AIDER_RETRY = "Retrying in "
#: The usage line aider logs after every request that got a reply (show_usage_report).
_AIDER_USAGE = "Tokens:"
#: litellm's exception name as aider logs a failed request (`litellm.<Name>: ...`).
_LITELLM_ERROR = re.compile(r"\blitellm\.([A-Za-z]+)\b")
#: The litellm exceptions of a request that failed at the endpoint, not because of the reply
#: (litellm_core_utils/exception_mapping_utils.py for an OpenAI-compatible endpoint): no
#: connection (no HTTP status), HTTP 500, 502, 503 and 429, and litellm.Timeout (a request timeout,
#: or an HTTP 408 or 504).
ENDPOINT_ERRORS = ("APIConnectionError", "InternalServerError", "BadGatewayError",
                   "ServiceUnavailableError", "RateLimitError",
                   # gbench's timeout (model_request_timeout_s) covers a reply to the end of the
                   # served window, so a request that still times out means a slow endpoint
                   "Timeout")
#: What a test run prints when a language toolchain in the image cannot run at all, whatever the
#: model wrote: rustup with no default toolchain (current and older wording), as when the
#: container's HOME is not the /root the image installed Rust under.
_TOOLCHAIN_BROKEN = re.compile(
    r"rustup could not choose a version of \S+ to run|error: no default toolchain configured",
    re.I)


#: Printed by the toolchain check (_TOOLCHAIN_CHECK) when the image cannot run the exercises as
#: upstream's does; benchmark.py does not start then.
_TOOLCHAIN_MARKER = "GBENCH_AIDER_TOOLCHAIN_CHECK_FAILED "
#: Run in the benchmark container before anything else: the JavaScript exercises' jest must be
#: 29.x (every exercise's package.json asks for ^29.7.0; Jest 30 removed the matcher aliases two
#: of their specs call, docker/aider_polyglot.Dockerfile pins it), and cargo must run (rustup's
#: default toolchain under the container's HOME, /root). An image built before either held (for
#: example one whose unpinned `jest` resolved to 30.x) would fail those exercises whatever the
#: model wrote.
_TOOLCHAIN_CHECK = (
    "jest=$(node -p 'require(\"/npm-install/node_modules/jest/package.json\").version' "
    "2>/dev/null); case \"$jest\" in 29.*) ;; *) echo \"" + _TOOLCHAIN_MARKER
    + "jest=${jest:-missing}\"; exit 4;; esac; "
    "cargo --version >/dev/null 2>&1 || { echo \"" + _TOOLCHAIN_MARKER
    + "cargo: $(cargo --version 2>&1 | head -1)\"; exit 4; }; ")


def container_oom_kills(output: str) -> Optional[int]:
    """The OOM-kill count the container printed (_OOM_REPORT), or None.

    Only the last non-empty line of the container's standard output counts, since the report is
    the last thing the container prints: None when that line is not the report, or the report
    carries no number (the container's main process was killed before it ran, or neither cgroup
    file could be read)."""
    lines = [ln for ln in (output or "").splitlines() if ln.strip()]
    if not lines or not lines[-1].startswith(_OOM_MARKER):
        return None
    value = lines[-1][len(_OOM_MARKER):].strip()
    return int(value) if value.isdigit() else None


#: CONTAINER_MEMORY in bytes, as docker writes it to the cgroup.
CONTAINER_MEMORY_BYTES = 12 * 1024 ** 3


def container_memory_limit(output: str) -> Optional[Any]:
    """The memory limit the container reported (_MEMORY_MARKER): its bytes, "max" for no limit,
    or None when the line is missing or empty (the cgroup file could not be read)."""
    value = None
    for line in (output or "").splitlines():
        if line.startswith(_MEMORY_MARKER):
            value = line[len(_MEMORY_MARKER):].strip()
    if not value:
        return None
    return int(value) if value.isdigit() else value


def _memory_reason(enforced: Optional[Any]) -> Optional[str]:
    """Why the enforced memory limit makes the run not comparable, or None when it is the
    CONTAINER_MEMORY upstream's benchmark/docker.sh asks for."""
    if enforced == CONTAINER_MEMORY_BYTES:
        return None
    if enforced is None:
        return ("the benchmark container's enforced memory limit could not be read (no cgroup "
                f"memory.max / memory.limit_in_bytes), so upstream's {CONTAINER_MEMORY} limit "
                "cannot be confirmed (container_memory_limit.enforced_bytes: null)")
    return (f"the benchmark container ran under a memory limit of {enforced} "
            f"(container_memory_limit.enforced_bytes), not the {CONTAINER_MEMORY} "
            f"({CONTAINER_MEMORY_BYTES} bytes) upstream's benchmark/docker.sh sets (docker drops "
            "--memory on a host without a memory cgroup controller)")


def _oom_text(oom_kills: Optional[int], threads: int) -> str:
    """The OOM kills the container reported, for a hard error (empty when none or unknown)."""
    if not oom_kills:
        return ""
    return (f"; the kernel's OOM killer killed {oom_kills} process(es) in the container at its "
            f"{CONTAINER_MEMORY} limit, shared by {threads} thread(s) (container_oom_kills)")


def _oom_reason(oom_kills: Optional[int], threads: int) -> Optional[str]:
    """Why the OOM-kill count makes the run not comparable, or None when it is 0."""
    if oom_kills == 0:
        return None
    more = (f", more than the {UPSTREAM_THREADS} of upstream's documented run "
            f"(benchmark/README.md --threads {UPSTREAM_THREADS})"
            if threads > UPSTREAM_THREADS else "")
    if oom_kills is None:
        return ("the benchmark container's OOM-kill count could not be read (no "
                f"{_OOM_MARKER.strip()} line with a number at the end of its output: its main "
                "process was killed, or its cgroup's memory.events is not readable), so a test "
                f"process killed at the {CONTAINER_MEMORY} memory limit shared by the {threads} "
                f"exercise(s) benchmark.py runs at once (--threads){more} cannot be ruled out "
                "(container_oom_kills: null)")
    return (f"the kernel's OOM killer killed {oom_kills} process(es) in the benchmark container "
            f"(container_oom_kills, its cgroup's memory.events oom_kill) at the "
            f"{CONTAINER_MEMORY} memory limit shared by the {threads} exercise(s) benchmark.py "
            f"runs at once (--threads){more}: a killed test process fails that exercise's test "
            "run, which is scored as upstream scores it, but a kill in a shared container cannot "
            "be attributed to the exercise whose code used the memory")


def _toolchain_problem(output: str) -> Optional[str]:
    """What the toolchain check (_TOOLCHAIN_CHECK) reported in the container output, or None."""
    for line in (output or "").splitlines():
        if line.startswith(_TOOLCHAIN_MARKER):
            return line[len(_TOOLCHAIN_MARKER):].strip() or "unknown"
    return None


def container_hosts() -> List[str]:
    """The TOOLCHAIN_HOSTS of the run's languages (GBENCH_AIDER_LANGUAGES, default all six)."""
    names: List[str] = []
    for lang in _languages():
        for host in TOOLCHAIN_HOSTS.get(lang, ()):
            if host not in names:
                names.append(host)
    return names


def _aider_text(line: str) -> str:
    """The message of one of aider's own lines (io.append_chat_history writes them as blockquotes,
    '> ' + message + two spaces)."""
    return line[1:].strip() if line.startswith(">") else ""


def _is_aider_line(line: str) -> bool:
    """One of aider's own lines: io.append_chat_history writes them as "> " + the message with its
    whitespace stripped + two spaces (a markdown line break). The model's reply (io.ai_output) is
    written as it came, so its own blockquote lines do not end that way."""
    return line.startswith("> ") and line.endswith("  ") and bool(line[2:].strip())


def _test_block_opens(lines: List[str], i: int) -> bool:
    """Whether lines[i] opens a test run's output block: benchmark.py's run_unit_tests appends each
    run's output as a bare fence ("```", then the output, then "```") right after aider's own
    lines for the turn (or at the top of the file). The model's replies (io.ai_output: a blank
    line, then the reply, then a blank line) and the instructions ("#### " lines) never put a fence
    right after one of aider's lines (_is_aider_line), so a fence in them does not count, a
    fence after a blockquote line the model wrote included."""
    return lines[i].rstrip() == "```" and (i == 0 or _is_aider_line(lines[i - 1]))


def exercise_test_outputs(history: str) -> List[str]:
    """The output of each test run in an exercise's chat history (.aider.chat.history.md), in
    order: the text of every block _test_block_opens finds, up to the next line that starts with
    a fence."""
    lines = (history or "").splitlines()
    blocks: List[str] = []
    i = 0
    while i < len(lines):
        if not _test_block_opens(lines, i):
            i += 1
            continue
        j = i + 1
        while j < len(lines) and not lines[j].startswith("```"):
            j += 1
        blocks.append("\n".join(lines[i + 1:j]))
        i = j + 1
    return blocks


def _is_retry(text: str) -> bool:
    return text.startswith(_AIDER_RETRY) and text.endswith(" seconds...")


def request_give_ups(history: str) -> List[Dict[str, Any]]:
    """Each model request aider gave up on in an exercise's chat history, in order.

    A request whose LAST retry line (_AIDER_RETRY; no later retry line follows before a reply,
    aider's usage line or the next test run) is followed by no reply: only aider's own lines,
    without its usage line (_AIDER_USAGE), come before the next test run's output (or the end of
    the file); any other text there is the model's reply (io.ai_output writes it after a blank
    line), so that request succeeded. Each is {"cause": litellm's exception name,
    or "unknown", "endpoint": whether it failed at the endpoint, "detail": the logged failure}.
    The cause is the last failure aider logged after that last retry (the final attempt), else
    the one it logged just before it (the run ended while aider was still retrying). `endpoint`
    is true for ENDPOINT_ERRORS and for a logged failed host-name lookup or connection
    (litellm.Timeout included) and for a logged failed host-name lookup or connection
    (tau_common.network_failures); any other cause is not one."""
    from .tau_common import network_failures
    lines = (history or "").splitlines()
    found: List[Dict[str, Any]] = []
    for i, line in enumerate(lines):
        if not (line.startswith(">") and _is_retry(_aider_text(line))):
            continue
        replied, later_retry, after, contiguous = False, False, [], True
        for j in range(i + 1, len(lines)):
            if _test_block_opens(lines, j):
                break
            if not lines[j].startswith(">"):
                if lines[j].strip():
                    replied = True          # the model's reply
                    break
                contiguous = False
                continue
            text = _aider_text(lines[j])
            if text.startswith(_AIDER_USAGE):
                replied = True              # a reply, possibly with no text
                break
            if _is_retry(text):
                later_retry = True          # not this request's last retry
                break
            if contiguous:
                after.append(text)          # what aider logged after the last retry
        if replied or later_retry:
            continue
        cause = [t for t in after if t]
        if any(_LITELLM_ERROR.search(t) for t in cause):
            detail = [t for t in cause if _LITELLM_ERROR.search(t)][-1]
        else:
            before: List[str] = []
            k = i - 1
            while (k >= 0 and lines[k].startswith(">")
                   and not _is_retry(_aider_text(lines[k]))):
                before.insert(0, _aider_text(lines[k]))
                k -= 1
            cause = [t for t in before if t] or cause
            detail = next((t for t in cause if _LITELLM_ERROR.search(t)),
                          cause[0] if cause else "")
        text = "\n".join(cause)
        named = _LITELLM_ERROR.search(detail) or _LITELLM_ERROR.search(text)
        found.append({"cause": named.group(1) if named else "unknown",
                      "endpoint": bool((named and named.group(1) in ENDPOINT_ERRORS)
                                       or network_failures(text)),
                      "detail": " ".join(f"{detail} ({_aider_text(line)})".split())[:300]})
    return found


def exercise_harness_error(history: str) -> Optional[Dict[str, str]]:
    """The harness error an exercise's chat history shows, or None.

    * `toolchain`: a test run whose output (exercise_test_outputs) shows that a language
      toolchain in the image could not run at all (_TOOLCHAIN_BROKEN: rustup with no default
      toolchain), so the tests never ran over the model's code;
    * `toolchain_network`: a test run whose output shows a failed name lookup or connection for a
      TOOLCHAIN_FAILURE_HOSTS host (the Gradle distribution, Maven Central, crates.io), so the
      tests never ran over the model's code;
    * `model_endpoint`: aider gave up on a model request after the endpoint kept failing
      (request_give_ups with `endpoint`: eight failed attempts, then a ninth, with a connection,
      DNS, 5xx or 429 error), so the turn ended with no reply and the tests ran over code the
      model never got to write.
    None of them is the model's outcome. Only the test runs' outputs are read for the first two:
    the model's replies and the instructions are not."""
    from .tau_common import network_failures
    outputs = exercise_test_outputs(history)
    for out in outputs:
        broken = _TOOLCHAIN_BROKEN.search(out)
        if broken:
            line = next((ln for ln in out.splitlines() if _TOOLCHAIN_BROKEN.search(ln)),
                        broken.group(0))
            return {"kind": "toolchain", "detail": " ".join(line.split())[:300]}
    for out in outputs:
        lines = network_failures(out, TOOLCHAIN_FAILURE_HOSTS)
        if lines:
            return {"kind": "toolchain_network", "detail": lines[0]}
    for give_up in request_give_ups(history):
        if give_up["endpoint"]:
            return {"kind": "model_endpoint", "detail": give_up["detail"]}
    return None


def _read_history(results_file: str) -> str:
    try:
        with open(os.path.join(os.path.dirname(results_file), _HISTORY_FILE),
                  encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _image() -> str:
    return os.environ.get("GBENCH_AIDER_IMAGE", _IMAGE_DEFAULT)


def _bench_dir() -> str:
    """Host directory that holds `polyglot-benchmark/` and receives the run's results
    (mounted to /benchmarks; = aider's AIDER_BENCHMARK_DIR)."""
    d = prereqs_path("aider-bench", os.environ.get("GBENCH_AIDER_BENCHMARK_DIR"))
    return os.path.abspath(os.path.expanduser(d)) if d else ""


def _languages() -> List[str]:
    """GBENCH_AIDER_LANGUAGES (comma or space list), lowercased as benchmark.py lowercases
    --languages, without repeats; default all six."""
    raw = (os.environ.get("GBENCH_AIDER_LANGUAGES") or "").replace(",", " ").lower().split()
    return list(dict.fromkeys(raw)) or list(_LANGUAGES)


def _served_model_id(base_url: str, model_name: str) -> str:
    """The model id litellm sends to the endpoint: `model_name`, the served id gbench's eval runner
    resolved and checked against --models, unchanged. The endpoint's model list is not re-read (its
    first entry may be another model)."""
    return model_name


def _litellm_model(base_url: str, model_name: str) -> Tuple[str, str]:
    """(the litellm model name aider sends, where it came from): `openai/<served id>` (litellm
    routes openai/* to OPENAI_API_BASE, the served endpoint), or GBENCH_AIDER_MODEL. An override
    must be `openai/<id>` for an id the endpoint's /v1/models lists (checked, unless it is the
    served id gbench passed): any other litellm provider would send the requests elsewhere."""
    default = f"openai/{_served_model_id(base_url, model_name)}"
    override = (os.environ.get("GBENCH_AIDER_MODEL") or "").strip()
    if not override or override == default:
        return default, "the served model id gbench passed"
    if not override.startswith("openai/") or not override[len("openai/"):]:
        raise infra_required("aider_polyglot", f"GBENCH_AIDER_MODEL={override!r} must be "
                             "openai/<served id> (litellm's route to OPENAI_API_BASE, the served "
                             "endpoint); another provider would send the requests elsewhere",
                             DOCS_URL)
    from ...utils.endpoint import EndpointError, resolve_served_model
    wanted = override[len("openai/"):]
    try:
        served = resolve_served_model(base_url, [wanted], wanted)
    except EndpointError as e:
        raise infra_required("aider_polyglot", f"GBENCH_AIDER_MODEL={override!r} names a model "
                             f"the endpoint does not serve ({e}); unset it or name a served id",
                             DOCS_URL) from e
    return f"openai/{served.id}", f"GBENCH_AIDER_MODEL ({served.source})"


def _build_hint(image: str) -> str:
    return (
        "Build the aider-benchmark:5dc9490b image (gbench's Dockerfile = aider's own benchmark/Dockerfile "
        f"at the pinned aider commit {AIDER_COMMIT[:12]} + jest 29.7.0) and provide the "
        "exercises at the pinned commit:\n"
        f"  docker build -t {image} -f docker/aider_polyglot.Dockerfile docker\n"
        f"  git clone {POLYGLOT_BENCHMARK_REPO} \\\n"
        "      $GBENCH_PREREQS_DIR/aider-bench/polyglot-benchmark\n"
        "  git -C $GBENCH_PREREQS_DIR/aider-bench/polyglot-benchmark checkout "
        f"{POLYGLOT_BENCHMARK_COMMIT}\n"
        "  export GBENCH_AIDER_BENCHMARK_DIR=$GBENCH_PREREQS_DIR/aider-bench"
    )


def image_aider_commit(image: str) -> Optional[str]:
    """The AIDER_COMMIT_LABEL of `image` ("" when absent), or None when docker cannot inspect it."""
    r = subprocess.run(["docker", "image", "inspect", "--format",
                        '{{index .Config.Labels "' + AIDER_COMMIT_LABEL + '"}}', image],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    value = (r.stdout or "").strip()
    return "" if value == "<no value>" else value


def polyglot_checkout_problem(exercises_dir: str) -> Optional[str]:
    """Why the exercises checkout is not polyglot-benchmark at POLYGLOT_BENCHMARK_COMMIT with its
    tracked files unmodified, or None when it is."""
    fix = (f"git -C {exercises_dir} fetch origin && git -C {exercises_dir} checkout "
           f"{POLYGLOT_BENCHMARK_COMMIT} (or clone {POLYGLOT_BENCHMARK_REPO} there and check out "
           "that commit)")
    try:
        head = subprocess.run(["git", "-C", exercises_dir, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return f"its commit could not be read ({type(e).__name__}: {e}); {fix}"
    commit = (head.stdout or "").strip() if head.returncode == 0 else ""
    if not commit:
        return (f"it is not a git checkout whose commit can be read (git rev-parse HEAD: "
                f"{(head.stderr or '').strip()[:200] or 'no output'}); {fix}")
    if commit != POLYGLOT_BENCHMARK_COMMIT:
        return (f"it is at commit {commit[:12]}, not the pinned {POLYGLOT_BENCHMARK_COMMIT[:12]}; "
                f"{fix}")
    dirty = subprocess.run(["git", "-C", exercises_dir, "status", "--porcelain",
                            "--untracked-files=no"], capture_output=True, text=True, timeout=60)
    if dirty.returncode == 0 and (dirty.stdout or "").strip():
        first = (dirty.stdout or "").strip().splitlines()[0]
        return (f"tracked files are modified (e.g. {first.strip()}); restore them with "
                f"git -C {exercises_dir} checkout -- .")
    return None


def _check_prereqs(image: str, bench_dir: str) -> None:
    build = _build_hint(image)
    if not shutil.which("docker"):
        raise infra_required("aider_polyglot", "docker CLI not found. " + build, DOCS_URL)
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        raise infra_required("aider_polyglot", "docker daemon not reachable. " + build, DOCS_URL)
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        raise infra_required("aider_polyglot", f"image {image!r} not found. " + build, DOCS_URL)
    if not bench_dir or not os.path.isdir(os.path.join(bench_dir, "polyglot-benchmark")):
        raise infra_required(
            "aider_polyglot",
            "the polyglot exercises were not found. Set GBENCH_AIDER_BENCHMARK_DIR to a directory "
            "containing `polyglot-benchmark/`. " + build, DOCS_URL)
    check_pinned_sources(image, bench_dir)


def check_pinned_sources(image: str, bench_dir: str) -> None:
    """infra_required unless the image is built from AIDER_COMMIT (its AIDER_COMMIT_LABEL) and the
    exercises checkout is POLYGLOT_BENCHMARK_COMMIT, unmodified (polyglot_checkout_problem)."""
    build = _build_hint(image)
    found = image_aider_commit(image)
    if found != AIDER_COMMIT:
        raise infra_required(
            "aider_polyglot", f"image {image!r} is not built from the pinned aider commit "
            f"{AIDER_COMMIT[:12]} (label {AIDER_COMMIT_LABEL}: "
            f"{repr(found) if found else 'none'}); rebuild it with the current "
            "docker/aider_polyglot.Dockerfile. " + build, DOCS_URL)
    exercises = os.path.join(bench_dir, "polyglot-benchmark")
    problem = polyglot_checkout_problem(exercises)
    if problem:
        raise infra_required(
            "aider_polyglot", f"the exercises checkout {exercises} is not the pinned "
            f"polyglot-benchmark: {problem}. " + build, DOCS_URL)


def list_exercises(bench_dir: str, langs: List[str]) -> List[str]:
    """The run's exercises as `<lang>/<name>`, sorted: each directory under
    polyglot-benchmark/<lang>/exercises/practice/ for the languages `langs`."""
    out: List[str] = []
    root = os.path.join(bench_dir, "polyglot-benchmark")
    for lang in sorted(set(langs)):
        practice = os.path.join(root, lang, "exercises", "practice")
        try:
            names = sorted(n for n in os.listdir(practice)
                           if os.path.isdir(os.path.join(practice, n)))
        except OSError:
            continue
        out.extend(f"{lang}/{n}" for n in names)
    return sorted(out)


def select_exercises(exercises: List[str], limit: Optional[int]
                     ) -> Tuple[List[str], Optional[str], List[int]]:
    """(the exercises this run executes, the shard label or None, their positions in `exercises`):
    the shard first (base.apply_shard_indexed over the sorted list), then `limit` of them, drawn
    stratified by language with the seed "aider_polyglot" (sampling.stratified_sample)."""
    selected, label, positions = apply_shard_indexed(list(exercises), "aider_polyglot")
    if limit and int(limit) > 0 and len(selected) > int(limit):
        tagged = stratified_sample(list(zip(selected, positions)), int(limit),
                                   key_fn=lambda t: t[0].split("/", 1)[0], seed="aider_polyglot")
        selected = [t[0] for t in tagged]
        positions = [t[1] for t in tagged]
    return selected, label, positions


def build_exercise_tree(bench_dir: str, selected: List[str], dest: str) -> str:
    """Copy the `selected` exercises (`<lang>/<name>`) into `dest` with the checkout's layout
    (<lang>/exercises/practice/<name>), readable by any uid; returns `dest`."""
    root = os.path.join(bench_dir, "polyglot-benchmark")
    for ex in selected:
        lang, name = ex.split("/", 1)
        src = os.path.join(root, lang, "exercises", "practice", name)
        dst = os.path.join(dest, lang, "exercises", "practice", name)
        shutil.copytree(src, dst, symlinks=True)
    for dirpath, dirnames, filenames in os.walk(dest):
        os.chmod(dirpath, 0o755)
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            if not os.path.islink(path):
                os.chmod(path, os.stat(path).st_mode | 0o444)
    return dest


def model_request_timeout_s() -> int:
    """The litellm timeout of each of aider's requests: base.request_timeout_s of the served
    context window (base.UNCAPPED_TIMEOUT_TOKENS when the server reports none), base's rule for a
    request that carries no max_tokens. Aider's own 600 s cut off long replies."""
    return request_timeout_s(get_run_knob("max_model_len") or UNCAPPED_TIMEOUT_TOKENS)


def _write_model_settings(settings_dir: str, enable_thinking: bool,
                          timeout_s: Optional[int] = None,
                          temperature: Optional[float] = None) -> Dict[str, Any]:
    """Write the aider model-settings file that carries enable_thinking and the request timeout
    (`timeout_s`, default model_request_timeout_s); returns its extra_params. `temperature` (the
    run's, _run_temperature, in both modes) goes in as extra_params `temperature`, which
    Model.send_completion's kwargs.update(extra_params) puts over aider's own temperature 0 on
    every litellm call; None writes none.

    `aider/extra_params` is aider's documented global override: only its extra_params are used,
    merged into every model's litellm call, so the native edit format is unaffected. Its
    `timeout` replaces aider's 600 s default (Model.send_completion sets one only when absent);
    `max_retries` 0 leaves retrying to aider's own loop. litellm consumes both; neither reaches
    the request body."""
    extra_params = {"extra_body": {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}},
                    "timeout": int(timeout_s or model_request_timeout_s()),
                    # one HTTP attempt per aider attempt: the openai client otherwise makes 3, each
                    # up to `timeout`, before aider sees the failure and retries it itself
                    "max_retries": 0}
    if temperature is not None:
        extra_params["temperature"] = float(temperature)
    path = os.path.join(settings_dir, _SETTINGS_FILE)
    with open(path, "w", encoding="utf-8") as f:
        # JSON is valid YAML; aider reads the file with yaml.safe_load.
        json.dump([{"name": "aider/extra_params", "extra_params": extra_params}], f)
    check = os.path.join(settings_dir, _CHECK_FILE)
    with open(check, "w", encoding="utf-8") as f:
        f.write(_CHECK_SRC)
    # aider's register_models silently skips a file it cannot see, so keep it readable to any uid
    # (the pre-flight in _build_command then confirms it was actually applied).
    os.chmod(settings_dir, 0o755)
    os.chmod(path, 0o644)
    os.chmod(check, 0o644)
    return extra_params


def _max_tokens_note(extra_params: Any, thinking: bool, knob: Optional[int]) -> Optional[str]:
    """The result's max_output_tokens_note: what aider sent as max_tokens, from the extra_params
    its Model reported (the pre-flight). None there, the usual case (an `openai/<id>` model),
    is base's token-limit record for a harness that sends none; a model whose aider settings
    carry a max_tokens gets that value from aider on every request, which gbench does not change."""
    cap = extra_params.get("max_tokens") if isinstance(extra_params, dict) else None
    if cap is None:
        return resolve_token_budget("aider_polyglot", AIDER_TOKENS, thinking=thinking,
                                    max_output_tokens=knob).note
    return (f"aider sends max_tokens={cap} from the model's aider settings with every request"
            + (f"; --max-output-tokens {knob} not applied" if knob else ""))


def _contains(actual: Any, expected: Any) -> bool:
    """Every key/value of `expected` is present in `actual` (recursively)."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            k in actual and _contains(actual[k], v) for k, v in expected.items())
    return actual == expected


def _settings_confirmation(output: str) -> Optional[Dict[str, Any]]:
    """The pre-flight's report (files_loaded, extra_params) from the container output, or None."""
    for line in reversed((output or "").splitlines()):
        if line.startswith(_CHECK_MARKER):
            try:
                rep = json.loads(line[len(_CHECK_MARKER):])
            except ValueError:
                return None
            return rep if isinstance(rep, dict) else None
    return None


def _build_command(image: str, bench_dir: str, run_name: str, model: str, base_url: str,
                   threads: int, langs: List[str], edit_format: str,
                   exercises_dir: Optional[str], extra: str,
                   settings_dir: Optional[str] = None,
                   owner: Optional[str] = None) -> List[str]:
    """The `docker run` that runs aider's own benchmark.py inside the image.

    `exercises_dir` is a host tree holding a selected subset of the exercises
    (build_exercise_tree), mounted read-only at /gbench-exercises and passed as
    `--exercises-dir`; None runs the checkout's polyglot-benchmark (the whole set). `--num-tests`
    is never passed: benchmark.py takes the first N after an unseeded shuffle.

    The container runs on docker's bridge network with this host reachable at
    host.docker.internal (tau_common.ENDPOINT_NETWORK_ARGS), as upstream's benchmark/docker.sh
    runs it, never the host network: aider runs the exercises' tests over the model's code there,
    and the host's DOCKER-USER rule keeps the cloud metadata server out of reach of a bridge
    network only. It has the memory limit docker.sh gives it (CONTAINER_MEMORY_ARGS), and prints
    the OOM kills in its cgroup last (_OOM_REPORT). `base_url`
    is the endpoint as that container reaches it (run_aider_polyglot passes
    require_endpoint_network's URL). `owner` ("uid:gid", the host user running gbench) gets the
    run directory benchmark.py wrote (its newest `*--<run_name>` under /benchmarks) once it has
    exited: the container runs as the image's root (its HOME, /root, holds the Rust toolchain),
    and the per-exercise evidence stays readable and removable by the host user."""
    # benchmark.py reads the aider repo's git HEAD (`repo.head.object.hexsha`) for its run label.
    # The checkout's `.git` may be unreadable by the container's older git (it was written with the
    # `refstorage=reftable` extension), so if HEAD can't be read we replace `/aider/.git` with a
    # fresh single-commit repo. `safe.directory '*'` also clears the "dubious ownership" git raises
    # on the bind-mounted exercises. (This all happens on the container's ephemeral copy, never the
    # host checkout.)
    git_ensure = (
        "git config --global --add safe.directory '*' "
        "&& git config --global user.email gbench@localhost "
        "&& git config --global user.name gbench "
        "&& cd /aider "
        "&& { git rev-parse --verify HEAD >/dev/null 2>&1 "
        "|| { rm -rf /aider/.git && git init -q -b main && git add -A "
        "&& git commit -qm gbench-snapshot; }; } && "
    )
    preflight = ""
    if settings_dir:
        # Stop before benchmark.py (hours of work) unless aider itself loads the settings file.
        preflight = (f"python3 {_SETTINGS_MOUNT}/{_CHECK_FILE} "
                     f"{_SETTINGS_MOUNT}/{_SETTINGS_FILE} {shlex.quote(model)} && ")
    args = (
        _TOOLCHAIN_CHECK + git_ensure + preflight +
        f"./benchmark/benchmark.py {run_name} --model {model} "
        f"--exercises-dir {_EXERCISES_MOUNT if exercises_dir else 'polyglot-benchmark'} "
        f"--new --tries 2 "
        f"--threads {threads} --languages {','.join(langs)} "
    )
    if edit_format:
        args += f"--edit-format {edit_format} "
    if settings_dir:
        args += f"--read-model-settings {_SETTINGS_MOUNT}/{_SETTINGS_FILE} "
    if extra:
        args += extra + " "
    # After benchmark.py, whatever its exit status (kept as the container's): the run directory
    # handed to the host user, then the enforced memory limit and the OOM kills in the
    # container's cgroup (_OOM_REPORT), the last line the container prints.
    handover = ""
    if owner:
        handover = ("; rc=$?; d=$(ls -td /benchmarks/*--" + run_name + " 2>/dev/null | head -1); "
                    "[ -n \"$d\" ] && chown -R " + shlex.quote(owner) + " \"$d\" 2>/dev/null; "
                    "(exit $rc)")
    script = args.strip() + handover + _OOM_REPORT
    mounts = ["-v", f"{bench_dir}:/benchmarks"]
    if exercises_dir:
        mounts += ["-v", f"{exercises_dir}:{_EXERCISES_MOUNT}:ro"]
    if settings_dir:
        mounts += ["-v", f"{settings_dir}:{_SETTINGS_MOUNT}:ro"]
    from .tau_common import ENDPOINT_NETWORK_ARGS
    return [
        "docker", "run", "--rm", *ENDPOINT_NETWORK_ARGS,
        # upstream's benchmark/docker.sh: --memory=12g --memory-swap=12g
        *CONTAINER_MEMORY_ARGS,
        *mounts,
        "-e", "AIDER_DOCKER=1",
        "-e", "AIDER_BENCHMARK_DIR=/benchmarks",
        "-e", f"OPENAI_API_BASE={base_url}",
        "-e", "OPENAI_API_KEY=EMPTY",
        # No HOME override: it stays the image's /root, as upstream's benchmark/docker.sh leaves
        # it. rustup installed Rust under /root/.rustup and /root/.cargo with no RUSTUP_HOME or
        # CARGO_HOME, so any other HOME leaves cargo with no default toolchain and every Rust
        # exercise's `cargo test` fails.
        image, "bash", "-lc", script,
    ]


#: The per-exercise counters of `.aider.results.json` that aider's summarize_results sums into
#: its leaderboard columns (benchmark/benchmark.py): result-file key -> column name.
_AIDER_COUNTERS = {
    "num_error_outputs": "error_outputs",
    "num_malformed_responses": "num_malformed_responses",
    "num_user_asks": "user_asks",
    "lazy_comments": "lazy_comments",
    "syntax_errors": "syntax_errors",
    "indentation_errors": "indentation_errors",
    "num_exhausted_context_windows": "exhausted_context_windows",
    "test_timeouts": "test_timeouts",
    "prompt_tokens": "prompt_tokens",
    "completion_tokens": "completion_tokens",
}


def _number(value: Any) -> float:
    """A counter from a result file (aider writes ints; cost and duration are floats)."""
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _parse_results(run_dir: str) -> Dict[str, Any]:
    """Aggregate the per-exercise `.aider.results.json` under a benchmark run directory.

    Each file has `tests_outcomes` (a bool per try) and `language`. pass@1 = passed on the first
    try; pass@2 = passed within two tries (aider stops early once a try passes). `columns` holds
    aider's leaderboard columns as its summarize_results computes them (benchmark/benchmark.py),
    over the exercises scored here: the counters summed, `num_with_malformed_responses` and
    `percent_cases_well_formed`, `seconds_per_case`, `total_cost`, `total_tests` (the exercise
    directories of the run) and the `model` / `edit_format` / `commit_hash` values seen.

    An exercise whose chat history shows a harness error (exercise_harness_error: a toolchain in
    the image could not run, its tests could not reach a toolchain host, or aider gave up on a
    request the endpoint kept failing) is not scored: it is listed in `harness_errors`
    ({exercise, language, kind, detail}) and counted nowhere else. A scored exercise where aider
    gave up on a request for another cause (request_give_ups without `endpoint`, e.g.
    litellm.Timeout) is listed in `request_give_ups` ({exercise, language, cause, detail}).

    `traces` holds one record per exercise with a result file (exercise_trace), sorted by
    exercise."""
    files = glob.glob(os.path.join(run_dir, "**", ".aider.results.json"), recursive=True)
    traces: List[Dict[str, Any]] = []
    per_lang: Dict[str, Dict[str, int]] = {}
    harness_errors: List[Dict[str, str]] = []
    give_ups: List[Dict[str, str]] = []
    n = p1 = p2 = 0
    sums = {col: 0.0 for col in _AIDER_COUNTERS.values()}
    with_malformed = 0
    duration = cost = 0.0
    variants: Dict[str, List[str]] = {"model": [], "edit_format": [], "commit_hash": []}
    for f in files:
        try:
            data = json.loads(open(f, encoding="utf-8").read())
        except Exception:
            continue
        if not isinstance(data, dict) or "tests_outcomes" not in data:
            continue                        # skip exception/placeholder result files
        outcomes = data.get("tests_outcomes") or []
        lang = str(data.get("language") or _lang_from_path(f) or "unknown")
        history = _read_history(f)
        exercise = os.path.relpath(os.path.dirname(f), run_dir).replace(os.sep, "/")
        harness = exercise_harness_error(history)
        give_up_list = request_give_ups(history)
        traces.append(exercise_trace(exercise, lang, outcomes, harness,
                                     sum(1 for g in give_up_list if not g["endpoint"])))
        if harness:
            harness_errors.append({"exercise": exercise, "language": lang, **harness})
            continue
        for give_up in give_up_list:
            give_ups.append({"exercise": exercise, "language": lang, "cause": give_up["cause"],
                             "detail": give_up["detail"]})
        passed1 = bool(outcomes and outcomes[0])
        passed2 = bool(True in outcomes[:2])
        n += 1
        p1 += 1 if passed1 else 0
        p2 += 1 if passed2 else 0
        d = per_lang.setdefault(lang, {"total": 0, "pass1": 0, "pass2": 0})
        d["total"] += 1
        d["pass1"] += 1 if passed1 else 0
        d["pass2"] += 1 if passed2 else 0
        for key, col in _AIDER_COUNTERS.items():
            sums[col] += _number(data.get(key))
        if data.get("num_malformed_responses"):
            with_malformed += 1
        duration += _number(data.get("duration"))
        cost += _number(data.get("cost"))
        for key, seen in variants.items():
            val = data.get(key)
            if val and str(val) not in seen:
                seen.append(str(val))
    columns: Dict[str, Any] = {
        "test_cases": n,
        "total_tests": len(glob.glob(os.path.join(run_dir, "*", "exercises", "practice", "*"))),
        "pass_rate_1": round(100.0 * p1 / n, 1) if n else None,
        "pass_rate_2": round(100.0 * p2 / n, 1) if n else None,
        "pass_num_1": p1,
        "pass_num_2": p2,
        "percent_cases_well_formed": round(100.0 * (1.0 - with_malformed / n), 1) if n else None,
        "num_with_malformed_responses": with_malformed,
        **{col: int(v) for col, v in sums.items()},
        "seconds_per_case": round(duration / n, 1) if n else None,
        "total_cost": round(cost, 4),
        **{key: ", ".join(seen) or None for key, seen in variants.items()},
    }
    harness_errors.sort(key=lambda h: h["exercise"])
    give_ups.sort(key=lambda h: h["exercise"])
    traces.sort(key=lambda t: t["exercise"])
    return {"n": n, "pass1": p1, "pass2": p2, "per_lang": per_lang, "n_files": len(files),
            "columns": columns, "harness_errors": harness_errors, "request_give_ups": give_ups,
            "traces": traces}


def exercise_key(exercise: str) -> str:
    """`<lang>/<name>` of an exercise path under a run directory
    (`<lang>/exercises/practice/<name>`), or the path itself when it has another shape."""
    parts = exercise.replace(os.sep, "/").split("/")
    if len(parts) >= 4 and parts[1:3] == ["exercises", "practice"]:
        return f"{parts[0]}/{parts[3]}"
    return exercise


def exercise_trace(exercise: str, lang: str, outcomes: List[Any],
                   harness: Optional[Dict[str, str]], scored_give_ups: int) -> Dict[str, Any]:
    """One exercise's trace: `exercise` (<lang>/<name>), `language` (also `category`), `pass_1`,
    `pass_2` (= `is_correct`, the headline), `tries` (the test runs recorded), the requests aider
    gave up on for a cause that is not an endpoint failure (`request_give_ups`), and for a harness
    error `harness_error` ({kind, detail}) with `scoring_excluded`; `sample_key` is the exercise."""
    key = exercise_key(exercise)
    outcomes = list(outcomes or [])
    trace: Dict[str, Any] = {
        "exercise": key, "language": lang, "category": lang,
        "pass_1": bool(outcomes and outcomes[0]), "pass_2": bool(True in outcomes[:2]),
        "tries": len(outcomes), "request_give_ups": int(scored_give_ups),
        "sample_key": f"exercise:{key}",
    }
    trace["is_correct"] = trace["pass_2"] and not harness
    if harness:
        trace["harness_error"] = dict(harness)
        trace["scoring_excluded"] = True
    return trace


def _lang_from_path(path: str) -> Optional[str]:
    m = re.search(r"/(cpp|go|java|javascript|python|rust)/exercises/", path)
    return m.group(1) if m else None


def _kinds(harness: List[Dict[str, str]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for h in harness:
        out[h["kind"]] = out.get(h["kind"], 0) + 1
    return dict(sorted(out.items()))


_KIND_TEXT = {
    "toolchain": "tests that a language toolchain in the image could not run (rustup with no "
                 "default toolchain)",
    "toolchain_network": "tests that could not reach a toolchain host (Gradle, Maven Central, "
                         "crates.io)",
    "model_endpoint": "aider gave up on a model request the endpoint kept failing",
}


def _kinds_text(harness: List[Dict[str, str]]) -> str:
    return ", ".join(f"{n} {kind}: {_KIND_TEXT.get(kind, kind)}"
                     for kind, n in _kinds(harness).items())


def _all_harness_errors_message(harness: List[Dict[str, str]]) -> str:
    """The infra_required reason when every exercise aider finished had a harness error."""
    from .container_guard import FIREWALL_COMMAND
    first = harness[0]
    text = (f"every exercise aider finished ({len(harness)}) had a harness error, so no score is "
            f"reported ({_kinds_text(harness)}; first: {first['exercise']}: {first['detail']}).")
    kinds = _kinds(harness)
    if "toolchain" in kinds:
        text += (" A language toolchain in the image could not run the tests (for Rust, rustup "
                 "found no default toolchain: the image installs it under /root/.rustup, so the "
                 "container must keep HOME=/root, as gbench runs it). Rebuild the image with "
                 "docker/aider_polyglot.Dockerfile and check that `cargo --version` runs in it.")
    if "toolchain_network" in kinds:
        text += (" The exercises' tests download the Gradle distribution, Maven Central packages "
                 "and crates from the benchmark container on the docker bridge network: it needs "
                 "working DNS and internet access. On a cloud VM, block the metadata server with "
                 f"`{FIREWALL_COMMAND}` (a plain DROP rule for 169.254.169.254 also drops DNS, "
                 "since the VPC DNS resolver is that address).")
    if "model_endpoint" in kinds:
        text += (" aider's requests to the model endpoint kept failing (connection errors, DNS "
                 "failures, 5xx or 429 responses) until it gave up; check that the endpoint is up "
                 "and reachable from the docker bridge network.")
    return text


def _exit_hint(returncode: Optional[int]) -> str:
    """What a container exit status says beyond its number (empty when nothing more)."""
    if returncode == _SIGKILL_EXIT:
        return (f": its main process was killed with SIGKILL, as the kernel does when the "
                f"container reaches its {CONTAINER_MEMORY} memory limit (upstream's "
                "benchmark/docker.sh limit) and picks that process, or `docker kill` stopped it")
    return ""


def run_aider_polyglot(model_name: str, base_url: str, concurrency: int = 1,
                       enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run canonical Aider Polyglot (native edit format, pass@2) inside aider's benchmark image."""
    image = _image()
    bench_dir = _bench_dir()
    _check_prereqs(image, bench_dir)
    langs = _languages()
    limit = kwargs.get("limit")
    limit = int(limit) if limit and int(limit) > 0 else None
    shard = current_shard()
    # Host-side selection (see the module docstring): the shard first, then --eval-limit drawn
    # stratified by language; a subset runs from a copied exercises tree, never --num-tests.
    subset = bool(limit or (shard and shard.partial))
    all_exercises = list_exercises(bench_dir, langs)
    if subset and not all_exercises:
        raise infra_required(
            "aider_polyglot", f"no exercise directories were found for {langs} under "
            f"{os.path.join(bench_dir, 'polyglot-benchmark')}/<lang>/exercises/practice, so no "
            "subset can be selected. " + _build_hint(image), DOCS_URL)
    selected, shard_label, positions = select_exercises(all_exercises, limit)
    # --eval-limit N is N exercises for the whole suite, spread over the languages (the
    # stratified draw above), never N per language. Each exercise is one rollout of up to
    # `--tries 2` (aider's pass@2 protocol), not more exercises.
    per_language: Dict[str, int] = {lang: 0 for lang in langs}
    for ex in selected:
        lang = ex.split("/", 1)[0]
        per_language[lang] = per_language.get(lang, 0) + 1
    limit_plan = eval_limit_plan("aider_polyglot", limit, len(selected), per_language,
                                 setting_noun="language", attempts=1, unit="rollouts")
    limit_plan["tries_per_rollout_at_most"] = 2
    if subset and not selected:
        # An empty shard (more shards than exercises): nothing runs here, nothing is scored.
        reason = (f"empty shard {shard_label}: no items on this node ({len(all_exercises)} in "
                  "the full set)")
        logger.info("aider_polyglot: %s", reason)
        return {
            "benchmark_type": "eval", "eval_name": "aider_polyglot", "model_name": model_name,
            "thinking": bool(enable_thinking), "status": "success", "status_reason": reason,
            "total_questions": 0, "correct_answers": 0, "accuracy": None,
            "accuracy_withheld_reason": reason, "empty_shard": True,
            "full_set_items": len(all_exercises), "category_accuracy": {}, "sample_traces": [],
            "selected_exercises": [], **shard_identity(shard), "eval_limit_plan": limit_plan,
            "leaderboard_comparable": False, "leaderboard_comparable_reason": reason,
        }
    position_of = {ex: pos for ex, pos in zip(selected, positions)}
    # The benchmark runs the model's code on the bridge network: the metadata server must be
    # blocked there, and a loopback endpoint is reached at host.docker.internal. An endpoint key
    # never enters the container: its requests go through gbench's relay, which adds it.
    from .tau_common import require_endpoint_network
    relay = _endpoint_relay(base_url)
    try:
        container_base_url = require_endpoint_network(
            "aider_polyglot", image, relay.url if relay else base_url, DOCS_URL)
        result = _run_selected(
            model_name, base_url, container_base_url, image, bench_dir, langs, limit, shard,
            subset, all_exercises, selected, position_of, concurrency, enable_thinking, relay,
            kwargs)
        result["eval_limit_plan"] = limit_plan
        return result
    finally:
        if relay is not None:
            relay.stop()


def _endpoint_relay(base_url: str):
    """An EndpointRelay on the docker bridge gateway, started, for every run: it adds the
    endpoint's key (base.endpoint_api_key) when there is one, and puts the run's
    --max-output-tokens on requests that carry none (swe_agentic.AGENT_MAX_TOKENS_SUITES). Until
    2026-10-11 it started only with a key, so on a keyless endpoint one aider reply ran to the end
    of the 262,144-token window (1,738 s in that day's smoke run)."""
    from .base import endpoint_api_key
    key = endpoint_api_key() or None
    from . import gemini_relay
    from .swebench_common import EndpointRelay
    gateway = gemini_relay.network_gateway("aider_polyglot", "bridge", DOCS_URL)
    try:
        relay = EndpointRelay(base_url, gateway, key=key, name="aider-polyglot-model-relay")
    except ValueError as e:
        raise infra_required("aider_polyglot", f"the model endpoint is not an http(s) URL with "
                             f"a host ({e}).", DOCS_URL) from None
    try:
        return relay.start()
    except OSError as e:
        raise infra_required("aider_polyglot", f"gbench's relay to the model endpoint could not "
                             f"listen on {gateway} ({type(e).__name__}: {e}).", DOCS_URL) from None


def _run_selected(model_name: str, base_url: str, container_base_url: str, image: str,
                  bench_dir: str, langs: List[str], limit: Optional[int], shard: Any,
                  subset: bool, all_exercises: List[str], selected: List[str],
                  position_of: Dict[str, int], concurrency: int, enable_thinking: bool,
                  relay: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """run_aider_polyglot after the selection and the network checks."""
    # litellm routes `openai/<id>` to OPENAI_API_BASE; <id> is the served id gbench passes.
    model, model_source = _litellm_model(base_url, model_name)
    edit_format = os.environ.get("GBENCH_AIDER_EDIT_FORMAT", "")   # "" => aider's native default
    extra = os.environ.get("GBENCH_AIDER_EXTRA_ARGS", "").strip()
    if "--read-model-settings" in extra:
        # benchmark.py keeps only the last --read-model-settings, so a user file would silently
        # drop gbench's enable_thinking settings.
        raise ValueError("aider_polyglot: GBENCH_AIDER_EXTRA_ARGS must not pass "
                         "--read-model-settings; gbench passes its own to carry --thinking. See "
                         + DOCS_URL)
    if re.search(r"(^|\s)(--num-tests|-n)(\s|=|$)", extra):
        raise ValueError("aider_polyglot: GBENCH_AIDER_EXTRA_ARGS must not pass --num-tests: "
                         "gbench selects the exercises itself (--eval-limit, --shard). See "
                         + DOCS_URL)
    threads = max(1, int(concurrency or 1))
    run_name = "gbench"
    # the run's temperature (1.0 with --thinking, 0.0 without, or the operator's), sent on every
    # request through the settings file's extra_params
    temperature, temp_source = _run_temperature(enable_thinking)

    # Snapshot existing run dirs so we can identify the one this run creates (--new is timestamped).
    before = set(glob.glob(os.path.join(bench_dir, f"*--{run_name}")))
    settings_dir = tempfile.mkdtemp(prefix="gbench_aider_settings_")
    tree = tempfile.mkdtemp(prefix="gbench_aider_exercises_") if subset else None
    try:
        if tree:
            build_exercise_tree(bench_dir, selected, tree)
        model_extra_params = _write_model_settings(settings_dir, enable_thinking,
                                                   temperature=temperature)
        cmd = _build_command(image, bench_dir, run_name, model, container_base_url, threads,
                             langs, edit_format, tree, extra, settings_dir=settings_dir,
                             owner=f"{os.getuid()}:{os.getgid()}")
        logger.info("aider_polyglot: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        shutil.rmtree(settings_dir, ignore_errors=True)
        if tree:
            shutil.rmtree(tree, ignore_errors=True)

    # The image must run the exercises as upstream's does (jest 29, a working cargo): checked in
    # the container before benchmark.py starts.
    problem = _toolchain_problem(proc.stdout or "")
    if problem:
        raise infra_required(
            "aider_polyglot", f"image {image!r} cannot run the exercises' tests as upstream's "
            f"image does ({problem}): the JavaScript exercises need jest 29.x (their package.json "
            "asks for ^29.7.0; Jest 30 fails two of their specs whatever the model writes) and "
            "cargo must run with rustup's default toolchain under HOME=/root. Rebuild it with "
            f"the current Dockerfile: docker build -t {image} -f docker/aider_polyglot.Dockerfile "
            "docker", DOCS_URL)
    # Runtime confirmation that aider applied gbench's settings (and so --thinking reached every
    # request): the pre-flight's own report, checked before any result is read.
    confirmed = _settings_confirmation(proc.stdout or "")
    if (not confirmed or not confirmed.get("files_loaded")
            or not _contains(confirmed.get("extra_params"), model_extra_params)):
        raise RuntimeError(
            "aider_polyglot: aider did not confirm that it loaded gbench's model settings "
            f"(expected extra_params {model_extra_params}; aider reported {confirmed}), so "
            "--thinking/no-think would not reach its requests. Check that the image runs "
            f"`python3 -c 'from aider import models'`. See {DOCS_URL}. Last output:\n"
            + (proc.stderr or proc.stdout or "")[-1500:])

    after = sorted(set(glob.glob(os.path.join(bench_dir, f"*--{run_name}"))) - before,
                   key=lambda d: os.path.getmtime(d) if os.path.isdir(d) else 0)
    if not after:
        raise RuntimeError(
            "aider_polyglot: aider's benchmark produced no results directory (inference/build "
            "failed) - a harness/infra failure, not a 0%. Last output:\n"
            + (proc.stderr or proc.stdout or "")[-1500:])
    run_dir = after[-1]
    # processes the kernel's OOM killer killed in the container (None: the count was not printed)
    oom_kills = container_oom_kills(proc.stdout or "")
    # the memory limit the kernel enforced on it (None: not readable)
    enforced_memory = container_memory_limit(proc.stdout or "")
    agg = _parse_results(run_dir)
    harness = agg["harness_errors"]
    if not agg["n"] and harness:
        # Every exercise aider finished had a harness error: no score is reported (not a 0%).
        raise infra_required("aider_polyglot", _all_harness_errors_message(harness), DOCS_URL)
    if not agg["n"]:
        raise RuntimeError(
            "aider_polyglot: no exercise results were scored under "
            f"{run_dir!r} (found {agg['n_files']} result files). Last output:\n"
            + (proc.stderr or proc.stdout or "")[-1500:])

    n = agg["n"]
    columns = agg["columns"]
    exhausted = int(columns["exhausted_context_windows"])
    default_langs = set(langs) == set(_LANGUAGES)

    # No-partial gate. A canonical full run (all 6 languages, no --eval-limit, no shard) must
    # complete the entire 225-exercise set, and a subset run every exercise it selected; anything
    # short is a crashed/interrupted container, not a low score, so refuse to report it as a
    # result. A non-zero container exit still means the run aborted mid-flight -> hard-error.
    is_full = default_langs and not subset
    finished = n + len(harness)     # an exercise with a harness error ran to its end, unscored
    expected = _EXPECTED_FULL if is_full else (len(selected) if subset else None)
    if expected is not None and finished != expected:
        raise RuntimeError(
            f"aider_polyglot: partial run - finished {finished}/{expected} exercises "
            f"({n} scored, {len(harness)} with a harness error) "
            f"(container exit={proc.returncode}{_exit_hint(proc.returncode)}"
            f"{_oom_text(oom_kills, threads)}). A run must complete every exercise it "
            f"selected ({expected}); refusing to report a partial score. Last output:\n"
            + (proc.stderr or proc.stdout or "")[-1500:])
    if not is_full and proc.returncode != 0:
        raise RuntimeError(
            f"aider_polyglot: benchmark container exited {proc.returncode}"
            f"{_exit_hint(proc.returncode)}{_oom_text(oom_kills, threads)} after scoring {n} "
            "exercise(s) - the run aborted "
            "mid-flight (harness failure, not a 0%). Last output:\n"
            + (proc.stderr or proc.stdout or "")[-1500:])

    result = {
        "benchmark_type": "eval",
        "eval_name": "aider_polyglot",
        "model_name": model_name,
        "thinking": bool(enable_thinking),
        # The extra_params aider's Model reported after loading gbench's settings file, in the
        # same container, just before benchmark.py (merged into every request it sends).
        "model_extra_params": confirmed.get("extra_params"),
        "model_settings_confirmed": True,
        # the timeout of each of aider's requests (model_request_timeout_s), in seconds
        "model_request_timeout_s": (confirmed.get("extra_params") or {}).get("timeout"),
        # the model id sent (litellm's openai/<id>) and where it came from
        "aider_model": model,
        "request_model_id": model.split("/", 1)[1] if model.startswith("openai/") else model,
        "request_model_id_source": model_source,
        # benchmark.py's run directory: each exercise's .aider.results.json and
        # .aider.chat.history.md (the model's replies, aider's lines and every test run's output)
        "aider_run_dir": run_dir,
        "status": "completed_with_errors" if harness else "success",
        "total_questions": n,
        "correct_answers": agg["pass2"],
        "accuracy": round(100.0 * agg["pass2"] / n, 2),          # canonical headline = pass@2
        "pass_rate_1": round(100.0 * agg["pass1"] / n, 2),
        "pass_rate_2": round(100.0 * agg["pass2"] / n, 2),
        "tries": 2,
        "languages_evaluated": sorted(agg["per_lang"].keys()),
        "category_accuracy": {
            lang: {"total": d["total"], "correct": d["pass2"],
                   "accuracy": round(100.0 * d["pass2"] / d["total"], 2)}
            for lang, d in sorted(agg["per_lang"].items()) if d["total"]},
        "edit_format": edit_format or "native (aider default per model)",
        # the benchmark container's limit, as upstream's benchmark/docker.sh passes it (shared by
        # the `threads` exercises benchmark.py runs at once)
        "container_memory_limit": {"memory": CONTAINER_MEMORY, "memory_swap": CONTAINER_MEMORY,
                                   "threads": threads,
                                   # what the kernel enforced (cgroup memory.max /
                                   # memory.limit_in_bytes; "max": none; null: unreadable)
                                   "enforced_bytes": enforced_memory},
        # processes the kernel's OOM killer killed in that container (its cgroup's memory.events
        # oom_kill, printed after benchmark.py); null when the count could not be read. Any kill,
        # or null, makes the run not comparable: a kill in the shared container cannot be
        # attributed to the exercise whose code used the memory
        "container_oom_kills": oom_kills,
        "max_output_tokens_note": _max_tokens_note(
            confirmed.get("extra_params"), bool(enable_thinking),
            kwargs.get("max_output_tokens") or get_run_knob("max_output_tokens")),
        # aider's leaderboard columns for this run (polyglot_leaderboard.yml carries them per row)
        "aider_leaderboard_columns": columns,
        # aider's num_exhausted_context_windows, summed: replies that stopped at the served window
        # (no max_tokens is sent) and requests the window refused
        "exhausted_context_windows": exhausted,
        "replies_cut_off_by_budget": exhausted,
        # exercises left out of every figure above: a toolchain in the image could not run
        # (toolchain), their tests could not reach a toolchain host (toolchain_network), or aider
        # gave up on a request the endpoint kept failing (model_endpoint)
        "harness_errors": len(harness),
        "harness_error_kinds": _kinds(harness),
        "harness_error_exercises": harness,
        # the requests aider gave up on in scored exercises for a cause that is not an endpoint
        # failure (litellm.Timeout: a reply past aider's 600 s request timeout), one entry per
        # request: the model's outcome, scored as upstream scores it
        "scored_request_give_ups": len(agg["request_give_ups"]),
        "scored_request_give_up_exercises": agg["request_give_ups"],
        "metric": ("execution pass@2 via aider's own benchmark.py in the aider-benchmark:5dc9490b "
                   "container (native edit format); pass_rate_1 also reported"),
        # Comparable to the published Aider Polyglot leaderboard when it is the FULL 225-exercise
        # set (no --eval-limit) across all 6 languages with aider's native edit format and no
        # exhausted context window, no harness error and no OOM kill in the container.
        "leaderboard_comparable": bool(default_langs and not limit and not edit_format
                                       and not exhausted and not harness and oom_kills == 0
                                       and _memory_reason(enforced_memory) is None
                                       and not _off_official(temperature)),
        # the pinned sources (check_pinned_sources verified both before the run)
        "aider_commit": AIDER_COMMIT,
        "polyglot_benchmark_commit": POLYGLOT_BENCHMARK_COMMIT,
        # the exercises this run executed (host-side selection; <lang>/<name>), and one trace
        # per exercise with a result file
        "selected_exercises": list(selected) if subset else (list(all_exercises) or None),
        "sample_traces": [dict(t, **({"global_idx": position_of[t["exercise"]]}
                                     if t["exercise"] in position_of else {}))
                          for t in agg["traces"]],
        **shard_identity(shard),
        # the run's temperature went in extra_params (confirmed above)
        **_sampling_record(temperature, temp_source),
        **_attempts_record(kwargs),
    }
    if relay is not None:
        result["model_endpoint_relay"] = relay.record()
    window = get_run_knob("max_model_len")
    noncanon = [why for bad, why in (
        (bool(limit), f"subset run (--eval-limit {limit}: {len(selected)} exercise(s) drawn "
                      "stratified by language with a fixed seed)"),
        (not default_langs, "language subset (GBENCH_AIDER_LANGUAGES)"),
        (bool(edit_format), f"forced edit format {edit_format!r} (the leaderboard uses aider's "
                            "native format per model)"),
        (bool(exhausted), f"{exhausted} exhausted context window(s) (aider's "
                          "num_exhausted_context_windows: a reply that stopped at finish_reason "
                          "length, which with no max_tokens sent is the end of the served "
                          f"{f'{window}-token ' if window else ''}context window, or a request "
                          "the window refused): they measure the served window, not the model "
                          "(serve a larger --max-model-len)"),
        (bool(harness), f"{len(harness)} exercise(s) with a harness error left out of the score "
                        f"({_kinds_text(harness)}; harness_error_exercises)"),
        (oom_kills != 0, _oom_reason(oom_kills, threads)),
        (_memory_reason(enforced_memory) is not None, _memory_reason(enforced_memory)),
        (_off_official(temperature), f"temperature {temperature} ({temp_source}): aider's "
                                     "benchmark samples at temperature 0")) if bad]
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    if shard is not None and shard.partial:
        add_comparability_reason(result, SHARD_REASON.format(
            label=shard.label, index=shard.index, count=shard.count))
    return result


def _run_temperature(enable_thinking: bool) -> Tuple[float, str]:
    """(temperature, source) the run sends in aider's extra_params:
    GBENCH_AIDER_POLYGLOT_TEMPERATURE, then --temperature, else gbench's rule
    (base.resolve_temperature): 1.0 with --thinking, 0.0 without."""
    return resolve_temperature("aider_polyglot", None, thinking=enable_thinking)


def _off_official(temperature: Optional[float]) -> bool:
    """True when gbench sent a temperature other than aider's official 0."""
    return temperature is not None and abs(float(temperature) - AIDER_SAMPLING.temperature) > 1e-9


def _sampling_record(temperature: float, source: str) -> Dict[str, Any]:
    """The result's sampling fields: the temperature gbench sent in extra_params and where it
    came from, with aider's official 0 under `official_sampling`."""
    return {"sampling": AIDER_SAMPLING_NOTE, "temperature_applied": True,
            "temperature": temperature, "temperature_source": source,
            "official_sampling": AIDER_SAMPLING.record(temperature, source)}


def _attempts_record(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """{"attempts": {...not_applied}} when --attempt-count asked for more than 1 attempt: aider's
    benchmark.py runs each exercise once with --tries 2 (its pass@2 protocol)."""
    k = int(kwargs.get("attempt_count") or get_run_knob("attempt_count") or 1)
    if k <= 1:
        return {}
    return {"attempts": {"attempts_requested": k, "attempts_per_sample": 1,
                         "not_applied": ("aider_polyglot: aider's benchmark.py runs each exercise "
                                         "once with --tries 2 (the pass@2 protocol); "
                                         f"--attempt-count {k} is not applied")}}


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of a sharded run's results (gbench --merge): the generic re-count of the
    union of their traces (pass@2, the headline, per language), with pass@1, the harness errors,
    the scored give-ups, the exhausted context windows and the OOM kills recomputed or summed, and
    the 225-exercise check of a full run applied to the union."""
    from ...merge import generic_merge
    merged = generic_merge("aider_polyglot", results)
    traces = merged.get("sample_traces") or []
    scored = [t for t in traces if not t.get("scoring_excluded")]
    n = len(scored)
    merged["total_questions"] = n
    merged["correct_answers"] = sum(1 for t in scored if t.get("pass_2"))
    merged["pass_rate_1"] = round(100.0 * sum(1 for t in scored if t.get("pass_1")) / n, 2) if n else None
    merged["pass_rate_2"] = merged["accuracy"]
    merged["languages_evaluated"] = sorted({t.get("language") for t in scored if t.get("language")})
    merged["harness_error_exercises"] = sorted(
        (h for r in results for h in (r.get("harness_error_exercises") or [])),
        key=lambda h: h.get("exercise", ""))
    merged["harness_errors"] = len(merged["harness_error_exercises"])
    kinds: Dict[str, int] = {}
    for h in merged["harness_error_exercises"]:
        kinds[h.get("kind", "unknown")] = kinds.get(h.get("kind", "unknown"), 0) + 1
    merged["harness_error_kinds"] = dict(sorted(kinds.items()))
    for field in ("scored_request_give_ups", "exhausted_context_windows",
                  "replies_cut_off_by_budget"):
        merged[field] = sum(int(r.get(field) or 0) for r in results)
    merged["scored_request_give_up_exercises"] = [
        g for r in results for g in (r.get("scored_request_give_up_exercises") or [])]
    ooms = [r.get("container_oom_kills") for r in results]
    merged["container_oom_kills"] = None if any(o is None for o in ooms) else sum(ooms)
    merged["selected_exercises"] = sorted(
        {e for r in results for e in (r.get("selected_exercises") or [])})
    merged.pop("aider_leaderboard_columns", None)
    merged["aider_leaderboard_columns_by_shard"] = [r.get("aider_leaderboard_columns")
                                                    for r in results]
    if not results[0].get("eval_limit") and len(traces) != _EXPECTED_FULL and set(
            merged["languages_evaluated"]) == set(_LANGUAGES):
        add_comparability_reason(merged, f"the merged shards hold {len(traces)} exercises, not "
                                         f"the full {_EXPECTED_FULL}")
    return merged
