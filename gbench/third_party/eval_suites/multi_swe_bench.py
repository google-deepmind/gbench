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

# Evaluation Suite: multi_swe_bench
# Description: Multi-SWE-bench (ByteDance, 7 languages) - resolved rate via its own harness

"""gbench native built-in runner for multi_swe_bench (Coding & Software Engineering).

Multi-SWE-bench (ByteDance-Seed/Multi-SWE-bench at a pinned dataset commit): the 1,632 instances
over the benchmark's seven languages (C, C++, Go, Java, JavaScript, Rust, TypeScript), scored by
execution-based resolved-rate via the project's OWN harness
(`multi_swe_bench.harness.run_evaluation`; vanilla swebench cannot score it - different
prediction schema and test fields). SANDBOX_EVAL. Hard-errors (infra_required) if the harness,
Docker or the MSWE-agent image (agentic mode) is absent.

Protocol (default, agentic): MSWE-agent (multi-swe-bench/MSWE-agent at MSWE_AGENT_COMMIT), the
Multi-SWE-bench team's SWE-agent and the scaffold of the leaderboard's MSWE-agent rows
(multi-swe-bench/experiments evaluation/<language>/verified/*_MSWE-agent_*), runs from the LOCAL
image gbench/docker/mswe_agent.Dockerfile through its own parallel entry point (multirun.py) with
its README's run arguments and its own config/default.yaml, one language at a time, on the
instance images the team publishes (`mswebench/<org>_m_<repo>:pr-<number>`, tagged with the name
MSWE-agent's harness gives them). Its task text, repository setup and submission are its own:
`TITLE:/CONTENT:` of the first resolved issue, the repository reset to the base commit under
/home/<repo>, `rm /home/fix.patch`, and the patch its `submit` command prints. The published
images are pulled on the host first (swebench_common.host_pull_images, the operator's Docker
login; the MSWE-agent container gets no Docker client configuration, and its driver finds each
image present and only tags it). What gbench's relay counts (endpoint failures, model turns cut
off at a limit or empty after reasoning) applies to the result
(swebench_common.relay_result_reasons). Three additions,
all recorded: the 50-step limit of the leaderboard's MSWE-agent runs (its README passes
`--per_instance_api_calls_limit 50`, which the public code does not define; every published
MSWE-agent trajectory stops at 50 model steps with "Exceed maximum number of steps: 50"), applied
by gbench/docker/mswe_agent_run.py; the harness definitions of the six Java repositories its
vendored harness does not register (the 78 SWE-bench-java instances; see the Dockerfile); and the
driver's archive.debian.org fallback for the `apt-get update` of MSWE-agent's reset on instance
images whose Debian release the Debian mirrors no longer serve (buster: the gcc:7-based C++
images, 48 instances), as SWE-bench_Pro-os's LockedMiniSwe.install does for old Debian images:
applied only after that `apt-get update` fails because a suite is gone from its mirror, recorded
per trace (`agent_apt_archive`) and per run (`apt_archive_fallback`); when apt still fails the
instance stays an infrastructure failure. The harness itself runs through mswe_harness_build:
upstream's run_evaluation, with the same archive.debian.org repair for its image builds, applied
only when building upstream's own Dockerfile fails with apt reporting a debian.org suite gone from
its mirror (the gcc:7 C++ base images, e.g. mswebench/simdjson_m_simdjson:base-cpp-7). The image
is then built again from that Dockerfile, with the repair step added after each FROM line, and the
retry is recorded in `harness_apt_archive_fallback`. One-shot is opt-in with
GBENCH_MULTI_SWE_BENCH_MODE=oneshot.

Cloud metadata server: MSWE-agent's instance containers (the model's commands) and the harness's
containers (the tests over each patch) have network access. Before either starts,
swebench_common.require_task_network_blocked checks that 169.254.169.254 is unreachable from
docker's default bridge and from a user-defined bridge network, and that the package hosts the
instance containers' builds fetch from at test time resolve there (runtime_hosts: Maven Central,
the Sonatype OSS repository, the Gradle plugin portal and distributions, the npm registry, the Go
module proxy, the crates.io index and downloads, GitHub, for the languages of the run);
infra_required otherwise.

DNS failures in the harness pass: fix-run.sh runs the repository's own build (mvn, gradlew, pnpm,
go, cargo, ...) in the instance container, which fetches packages at test time. An instance the
harness reports unresolved whose fix-patch run captured no test result (report.json's
fix_patch_result counts all 0) and whose fix-patch-run.log shows one of those package hosts failing
to resolve (Maven's "Unknown host", a java.net.UnknownHostException, npm's "getaddrinfo EAI_AGAIN",
Go's "dial tcp: lookup", curl's "Could not resolve host", ...; harness_dns_failure) is a harness
error when something outside that container corroborates it (another such instance in the run, or
a fresh probe container after the run that cannot resolve one of the hosts either;
swebench_common.corroborated_dns_failures): left out of the score (`scoring_excluded`), counted in
`harness_errors` and multi_swe_bench_report.harness_dns_failures, run completed_with_errors, and a
run whose every submitted instance is a harness error withholds its score. Uncorroborated, it stays
unresolved (the model's outcome) with `dns_failure_note`. Once tests produced results, the verdict
stands.

simdjson's checkperf reference: the build of simdjson-949, -954 and -958 (benchmark/checkperf.cmake
at their base commits) fetches simdjson's `master` from GitHub at test time and builds it in the
gcc:7 image; today's master no longer configures there, so no test would run whatever the patch.
mswe_harness_build pins that fetch to the master of upstream's own instance image clone
(726c3eb611a1, 2025-03-17; CHECKPERF_REFERENCE), with which upstream's gold patch of simdjson-949
reproduces the dataset's 46 passing tests exactly; recorded per run in
`harness_checkperf_reference`. The pin is set up before the patch is applied; when it could not be
set up the instance is a harness error (`checkperf_reference`), whatever the patch. The other
harness errors are left out of the score the same way (numerator and
denominator, swebench_common.scored_headline; counted in `harness_errors` by kind): a rollout that
is an infrastructure failure (`agent_infra`), a submission with a patch the harness gave no
verdict (`harness_no_verdict`), and every submitted patch of a harness pass that wrote no report
(`harness_no_report`). A rollout MSWE-agent ended with an exit_error other than the step limit is
an infrastructure failure only when its recorded exception names an endpoint or docker daemon
failure; otherwise it is the model's outcome, and so is an early_exit (the container stopped
answering while it ran the model's command); both are counted in `exits_kept_as_model_outcome`
(mswe_agent_trace).

Sampling: MSWE-agent sends temperature 0.0 and top_p 0.95 (run.py / multirun.py get_args
defaults, which its README's run command keeps) and no max_tokens (MSWE_AGENT_SAMPLING);
`--max-output-tokens` is not applied (recorded). The temperature follows gbench's rule for every
suite, not the official value: 1.0 for a `--thinking` run (temperature_source "thinking"), 0.0
without it (temperature_source "default"); top_p 0.95 unchanged. Override the temperature for a
whole run with `--temperature`, or for this suite alone with `GBENCH_MULTI_SWE_BENCH_TEMPERATURE`,
which takes precedence over both. The value is passed as MSWE-agent's own `--temperature`,
recorded, and anything other than 0.0 makes the run not leaderboard-comparable. The one-shot mode
follows the same rule.
"""

import importlib.metadata
import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from . import mswe_harness_build, swe_agentic, swe_thread_cap, swebench_common
from .base import (
    OfficialSampling, decoding_penalties, get_run_knob, resolve_official_sampling,
    resolve_token_budget, run_eval_suite, strip_thinking_tags, tally_batch_scores,
)
from .limit_plan import eval_limit_plan
from .sampling import stratified_sample
from .sandbox import credential_free_env
from .swebench_common import (
    AGENT_INFRA, NO_REPORT, NO_VERDICT, EndpointRelay, harness_errors_text, host_pull_images,
    image_pull_summary, infra_required, mark_harness_error, record_harness_errors,
    relay_result_reasons, scaffold_error_harness_reason, scored_headline, trace_patch,
    withhold_on_error,
)

logger = logging.getLogger(__name__)

PILLAR = "Coding & Software Engineering"
DOCS_URL = "docs/evals/multi_swe_bench.md"
_REPO = "ByteDance-Seed/Multi-SWE-bench"
#: Dataset commit the loader reads (HF main as of 2026-09-24).
_REVISION = "56ff018c04a38e27ada1e9d0a6d5839a51f88f0d"
#: The harness release scored with: PyPI multi-swe-bench 1.1.2 = multi-swe-bench/multi-swe-bench
#: at 24f493f8 (every .py file identical). Any other installed version is a prerequisite failure.
HARNESS_RELEASE = "1.1.2"
#: The benchmark's language directories. Upstream README: "7 languages (i.e., Java, TypeScript,
#: JavaScript, Go, Rust, C, and C++) with 1,632 high-quality instances", and the harness's
#: scripts/images_verified.txt lists exactly those 1,632 instance images. The dataset repo also
#: carries kotlin/ (105 instances outside that set) and a python/ file; neither is loaded.
_LANGUAGES = ("c", "cpp", "go", "java", "js", "rust", "ts")
_SRC_FILES: Dict[str, str] = {}  # instance_id -> local dataset jsonl path (for the scorer)

#: multi-swe-bench/MSWE-agent main (2025-04-02, its last commit): the scaffold of the
#: leaderboard's MSWE-agent rows. It runs from the LOCAL image MSWE_AGENT_IMAGE_DEFAULT
#: (gbench/docker/mswe_agent.Dockerfile, label `org.gbench.mswe_agent`).
MSWE_AGENT_COMMIT = "88217624b637646b886cb0462995c07559e96f58"
#: The config its README runs, and its sha256 at MSWE_AGENT_COMMIT (image label
#: `org.gbench.mswe_agent_config_sha256`).
MSWE_AGENT_CONFIG = "config/default.yaml"
MSWE_AGENT_CONFIG_SHA256 = "913fc5fd89b3bdffe261b9cd2e38b2e45ad9612ea57bf314bcda9f7e314b4537"
#: multi-swe-bench commit whose Java repository definitions the image adds (label
#: `org.gbench.mswe_agent_java_repos`): the commit that announced the 78 SWE-bench-java instances.
MSWE_AGENT_JAVA_REF = "e0dcb3c4ce0ae27997fb213f2a5294e1eaa487b8"
MSWE_AGENT_JAVA_ORGS = "apache fasterxml google googlecontainertools"
MSWE_AGENT_IMAGE_DEFAULT = "gbench-mswe-agent:88217624"
#: The version of gbench/docker/mswe_agent_run.py the image must carry (label
#: `org.gbench.mswe_agent_driver`): 2 = with the archive.debian.org fallback of the reset's
#: `apt-get update` (install_apt_archive_fallback); 3 = also recording which setup step of an
#: instance's reset failed (`setup_error`: apt, repo_reset, container_start, other), named in
#: that instance's harness error; 4 = also recording the exception an instance's rollout raised
#: (`rollout_error`: type, message, model turns started, kind of the last model reply) and
#: giving MSWE-agent's OpenAI client the spec's client_timeout_s (`openai_client`).
MSWE_AGENT_DRIVER_LABEL = "org.gbench.mswe_agent_driver"
MSWE_AGENT_DRIVER_VERSION = "4"
#: EndpointRelay's class default per-turn limit plus 60 s: the driver's own fallback
#: (mswe_agent_run.CLIENT_TIMEOUT_S) for a spec that names no client_timeout_s. gbench's specs
#: always name one: this run's relay safety cap plus swe_agentic.AGENT_CLIENT_MARGIN_S
#: (mswe_agent_timeouts, swe_agentic.client_timeout_for_relay_limit, sized for the served context window, since MSWE-agent sends no
#: max_tokens; until 2026-10-11 it was this constant, below the window's decode time), never the
#: SDK's default 600 s, at which the SDK abandons a long model turn (a long thinking turn, which
#: the endpoint answers only when it is complete) and sends it again. Harness-only: it changes
#: how long the client waits for a turn, nothing the model is sent or allowed.
MSWE_AGENT_CLIENT_TIMEOUT_S = EndpointRelay.TIMEOUT_S + 60
#: The OpenAI SDK release of the MSWE-agent image (gbench/docker/mswe_agent.constraints.txt), and
#: the transport retries its client keeps: driver 4 sets the timeout only, so the SDK's default
#: max_retries (2) stays. With the client timeout above the relay's safety cap on a model
#: request the SDK never gives up on a slow reply first; it retries only an error the relay
#: answers.
MSWE_AGENT_OPENAI_VERSION = "1.70.0"
MSWE_AGENT_CLIENT_RETRIES = 2


def mswe_agent_timeouts() -> "swe_agentic.AgentTimeouts":
    """The relay limit and MSWE-agent client timeout of this run (swe_agentic.
    agent_request_timeouts): sized for the max_tokens the relay sets under
    GBENCH_AGENT_MAX_TOKENS_FROM_RUN=1 (the run's --max-output-tokens knob), else for the served
    context window. _run_agentic sizes its relay with it and run_mswe_agent the client, from the
    same run knobs, so the client always waits longer than the relay."""
    return swe_agentic.agent_request_timeouts(swe_agentic.agent_max_tokens_from_run())
#: Model steps per instance of the leaderboard's MSWE-agent runs (every trajectory in
#: ByteDance-Seed/Multi-SWE-bench_trajs stops at 50 with "Exceed maximum number of steps: 50").
MSWE_AGENT_STEP_LIMIT = 50
#: MSWE-agent's sampling: run.py / multirun.py get_args defaults ModelArguments(temperature=0.0,
#: top_p=0.95), which its README's run command keeps; OpenAIModel.query sends temperature and
#: top_p and no max_tokens.
MSWE_AGENT_SAMPLING = OfficialSampling(
    reference=f"multi-swe-bench/MSWE-agent@{MSWE_AGENT_COMMIT} multirun.py get_args "
              "(ModelArguments temperature=0.0, top_p=0.95) + sweagent/agent/models.py "
              "OpenAIModel.query",
    temperature=0.0, params={"top_p": 0.95}, send_max_tokens=False)
_MSWE_ROOT = "/opt/MSWE-agent"
_DOCKER_SOCKET = "/var/run/docker.sock"
#: The package hosts each language's builds fetch from at test time: the instance images'
#: fix-run.sh runs the repository's own build (`mvn clean test`, `./gradlew clean test`,
#: `pnpm test`, `go test`, `cargo test`, ...), which resolves what the image does not cache.
#: Java: Maven Central and the Sonatype OSS repository (jackson-core's snapshots parent POM:
#: offline, `mvn clean test` in mswebench/fasterxml_m_jackson-core:pr-1208 fails with "Unknown
#: host oss.sonatype.org"), the Gradle plugin portal and the Gradle distributions its wrapper
#: downloads (ktlint: `gradlePluginPortal()`, distributionUrl services.gradle.org); JS/TS: the
#: npm registry (svelte's pnpm registry); Go: the module proxy (GOPROXY's default); Rust: the
#: crates.io sparse index and downloads. COMMON_RUNTIME_HOSTS apply to every language (git
#: dependencies, Go and Cargo git sources, CMake FetchContent).
RUNTIME_HOSTS: Dict[str, Tuple[str, ...]] = {
    "java": ("repo.maven.apache.org", "oss.sonatype.org", "plugins.gradle.org",
             "services.gradle.org"),
    "js": ("registry.npmjs.org",),
    "ts": ("registry.npmjs.org",),
    "go": ("proxy.golang.org",),
    "rust": ("index.crates.io", "static.crates.io"),
    "c": (),
    "cpp": (),
}
COMMON_RUNTIME_HOSTS = ("github.com",)
#: The domains a harness DNS failure is recognised under (harness_dns_failure): the installers'
#: (swebench_common.INSTALL_DOMAINS) plus the JVM, Go and Rust package hosts' and those they
#: redirect to.
RUNTIME_DOMAINS = tuple(dict.fromkeys((
    *swebench_common.INSTALL_DOMAINS, "maven.apache.org", "maven.org", "sonatype.org",
    "gradle.org", "maven.google.com", "dl.google.com", "yarnpkg.com", "golang.org",
    "crates.io")))


def runtime_hosts(languages) -> Tuple[str, ...]:
    """The package hosts the instance containers of `languages` fetch from at test time
    (RUNTIME_HOSTS of each, then COMMON_RUNTIME_HOSTS), each once, in order."""
    hosts: List[str] = []
    for lang in _LANGUAGES:
        if lang in set(languages or ()):
            hosts.extend(RUNTIME_HOSTS.get(lang, ()))
    hosts.extend(COMMON_RUNTIME_HOSTS)
    return tuple(dict.fromkeys(hosts))


def _uncaptured_fix_log(workdir: str, e: Dict[str, Any]) -> Optional[str]:
    """The fix-patch-run.log of instance `e` when the harness's fix-patch run captured no test
    result, else None.

    The harness writes each instance's run under <workdir>/<org>/<repo>/evals/pr-<number>/:
    fix-patch-run.log (the build and test output) and report.json (its Report, whose
    fix_patch_result counts the tests the run captured). A run that captured a test result is a
    verdict on the patch whatever else its log says, and an instance without a readable
    report.json keeps its verdict too."""
    inst = os.path.join(workdir, str(e.get("org")), str(e.get("repo")), "evals",
                        f"pr-{e.get('number')}")
    try:
        with open(os.path.join(inst, "report.json"), encoding="utf-8") as f:
            report = json.load(f)
    except (OSError, ValueError):
        return None
    fix = (report or {}).get("fix_patch_result") if isinstance(report, dict) else None
    if not isinstance(fix, dict):
        return None
    try:
        if any(int(fix.get(k) or 0) for k in ("passed_count", "failed_count", "skipped_count")):
            return None
    except (TypeError, ValueError):
        return None
    return swebench_common.read_texts([os.path.join(inst, "fix-patch-run.log")])


def harness_dns_failure(workdir: str, e: Dict[str, Any]) -> List[str]:
    """The package hosts (RUNTIME_DOMAINS) the harness's fix-patch run of instance `e` could not
    resolve, when that run captured no test result (_uncaptured_fix_log), else []."""
    text = _uncaptured_fix_log(workdir, e)
    if text is None:
        return []
    return swebench_common.dns_failure_hosts(text, RUNTIME_DOMAINS)


#: The harness error kind of a submission whose harness test run fetches simdjson's checkperf
#: reference (mswe_harness_build.CHECKPERF_REFERENCE) but ran without the pin.
CHECKPERF_REFERENCE = "checkperf_reference"


def checkperf_reference_failures(entries: List[Dict[str, Any]],
                                 pins: List[Dict[str, Any]]) -> Dict[str, str]:
    """{instance_id: why} for the simdjson/simdjson `entries` whose PR is one of
    CHECKPERF_REFERENCE's instances and whose harness test run has no record (`pins`, from
    mswe_harness_build) of the pinned reference: its setup failed, or no run was recorded. The
    setup runs before the patch is applied, so this never depends on the patch. Pass only entries
    the harness gave a verdict: one with no test run at all (its image never built) is already a
    harness error (harness_unscored_ids)."""
    want = mswe_harness_build.CHECKPERF_REFERENCE["instances"]
    by_number = {}
    for r in pins or ():
        try:
            by_number[int(r.get("number"))] = r
        except (TypeError, ValueError):
            continue
    out: Dict[str, str] = {}
    for e in entries:
        try:
            number = int(e.get("number"))
        except (TypeError, ValueError):
            continue
        if (e.get("org"), e.get("repo")) != ("simdjson", "simdjson") or number not in want:
            continue
        r = by_number.get(number)
        if r is None:
            out[e["instance_id"]] = "no record of the pin for this test run"
        elif not r.get("pinned"):
            out[e["instance_id"]] = str(r.get("detail") or "the pin was not applied")
    return out


def check_multi_swe_bench_prerequisites() -> Tuple[bool, str]:
    if not shutil.which("docker"):
        return False, "Docker CLI is not found on PATH."
    try:
        if subprocess.run(["docker", "info"], capture_output=True, timeout=20).returncode != 0:
            return False, "Docker daemon is not reachable."
    except Exception as e:
        return False, f"Cannot connect to Docker daemon: {e}"
    try:
        # find_spec raises (not returns None) when the PARENT package is absent.
        if importlib.util.find_spec("multi_swe_bench.harness.run_evaluation") is None:
            raise ModuleNotFoundError
    except ModuleNotFoundError:
        return False, ("Python package 'multi_swe_bench' is not installed "
                       f"(pip install 'multi-swe-bench=={HARNESS_RELEASE}').")
    version = _installed_harness_version()
    if version != HARNESS_RELEASE:
        return False, (f"multi-swe-bench {version or '(unknown version)'} is installed, but this "
                       f"suite scores with the pinned harness release {HARNESS_RELEASE} "
                       f"(pip install 'multi-swe-bench=={HARNESS_RELEASE}').")
    return True, ""


def _installed_harness_version() -> Optional[str]:
    try:
        return importlib.metadata.version("multi-swe-bench")
    except importlib.metadata.PackageNotFoundError:
        return None


def _extract_patch(text: str) -> str:
    """Delegate to the shared extractor.

    This file used to carry its own copy, which only inspected the FIRST fenced block and
    otherwise took everything from `diff --git` to end-of-response - sweeping up the
    closing fence and the prose after it, so the harness rejected the patch as malformed
    ("patch: **** malformed patch at line 20: ```"). Three suites had the same copy.
    """
    from .swebench_common import extract_patch
    return extract_patch(strip_thinking_tags(text or ""))


def _read_instances() -> List[Tuple[str, Dict[str, Any]]]:
    """(language, row) for every instance of the seven canonical languages at `_REVISION`, via
    list_repo_files + hf_hub_download (load_dataset is broken for this repo)."""
    global _SRC_FILES
    _SRC_FILES = {}
    try:
        from huggingface_hub import HfApi, hf_hub_download
        files = HfApi().list_repo_files(_REPO, repo_type="dataset", revision=_REVISION)
        jsonl_files = sorted(f for f in files if f.endswith("_dataset.jsonl")
                             and f.split("/")[0] in _LANGUAGES)
    except Exception as e:
        logger.error(f"Failed to list dataset for multi_swe_bench: {e}")
        raise RuntimeError(f"Could not load dataset for multi_swe_bench: {e}") from e

    out: List[Tuple[str, Dict[str, Any]]] = []
    for jf in jsonl_files:
        lang = jf.split("/")[0]
        local_p = hf_hub_download(repo_id=_REPO, filename=jf, repo_type="dataset",
                                  revision=_REVISION)
        with open(local_p, encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                item = json.loads(line)
                org, repo, number = item.get("org"), item.get("repo"), item.get("number")
                if not org or not repo or number is None:
                    raise RuntimeError(
                        "multi_swe_bench: unexpected schema (org/repo/number); "
                        "refusing to fabricate sample data"
                    )
                item["instance_id"] = item.get("instance_id") or f"{org}__{repo}-{number}"
                _SRC_FILES[item["instance_id"]] = local_p
                out.append((lang, item))
    if not out:
        raise RuntimeError("multi_swe_bench returned empty rows")
    return out


def _load_multi_swe_bench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """One-shot samples (GBENCH_MULTI_SWE_BENCH_MODE=oneshot)."""
    samples: List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]] = []
    for lang, item in _read_instances():
        org, repo, number = item["org"], item["repo"], item["number"]
        title = str(item.get("title") or "").strip()
        body = str(item.get("body") or "").strip()
        prompt = (
            f"Repository: {org}/{repo} ({lang})\n"
            f"Base commit: {(item.get('base') or {}).get('sha', '')}\n\n"
            f"Issue: {title}\n\n{body}\n\n"
            "Output ONLY a unified git diff (`diff --git a/... b/...`), rooted at the "
            "repository top, that resolves the issue."
        )
        samples.append(([{"role": "user", "content": prompt}],
                        item.get("fix_patch") or "",
                        {"category": lang, "instance_id": item["instance_id"],
                         "org": org, "repo": repo, "number": number}))

    # Stratified across LANGUAGES, not a contiguous head. The loader used to
    # `break` as soon as it had `limit` rows while walking `sorted(jsonl_files)`; `c/...`
    # sorts first, so `--eval-limit 20` returned 20 C instances (all facebook/zstd) and
    # never opened java/go/rust/cpp/js/ts. A *Multi*-SWE-bench score measured on one
    # language is not the benchmark. Measured 2026-08-17: 1 distinct category in 20 samples.
    before = len(samples)
    samples = stratified_sample(
        samples, limit,
        key_fn=lambda s: (s[2] or {}).get("category"),
        seed="multi_swe_bench")
    langs = sorted({(s[2] or {}).get("category") for s in samples})
    logger.info("Loaded %d multi_swe_bench samples from %d instances across %d language(s): %s",
                len(samples), before, len(langs), ", ".join(str(l) for l in langs))
    return samples


def _issue_text(item: Dict[str, Any]) -> str:
    """MSWE-agent's task text (sweagent/environment/swe_env.py, reset): the first resolved
    issue, not the PR description. MSWE-agent builds it itself; gbench records it on the trace
    and refuses a row without one."""
    issues = item.get("resolved_issues") or []
    if not issues or not isinstance(issues[0], dict):
        raise RuntimeError(f"multi_swe_bench: {item.get('instance_id')} has no resolved_issues; "
                           "refusing to fabricate the task text")
    return ('TITLE:\n' + str(issues[0].get("title") or "") + '\n CONTENT:\n'
            + str(issues[0].get("body") or ""))


def published_image(org: str, repo: str, number: Any) -> str:
    """The instance image the Multi-SWE-bench team publishes on Docker Hub (the harness's
    `mswebench/<org>_m_<repo>:pr-<number>`)."""
    return f"mswebench/{org}_m_{repo}:pr-{number}".lower()


def _load_multi_swe_bench_tasks() -> Tuple[List["swe_agentic.AgentTask"],
                                           Dict[str, Dict[str, Any]]]:
    """(agentic tasks, instance_id -> dataset row) for every instance of the seven languages.
    A task records what MSWE-agent is given: the task text it builds and the published image;
    MSWE-agent itself reads the row (the driver hands it the rows of the selected tasks)."""
    tasks, rows = [], {}
    for lang, item in _read_instances():
        org, repo, number = item["org"], item["repo"], item["number"]
        rows[item["instance_id"]] = item
        tasks.append(swe_agentic.AgentTask(
            instance_id=item["instance_id"], problem_statement=_issue_text(item),
            image=published_image(org, repo, number), category=lang, cwd=f"/home/{repo}",
            meta={"org": org, "repo": repo, "number": number}))
    return tasks, rows


# --------------------------------------------------------------------------------------------
# the scaffold: MSWE-agent in its local image
# --------------------------------------------------------------------------------------------
def mswe_agent_image() -> str:
    return (os.environ.get("GBENCH_MULTI_SWE_BENCH_AGENT_IMAGE") or "").strip() or \
        MSWE_AGENT_IMAGE_DEFAULT


def check_mswe_agent_prerequisites() -> Tuple[bool, str]:
    """The docker socket MSWE-agent starts the instance images through, and the LOCAL image with
    MSWE-agent at MSWE_AGENT_COMMIT, its config, the added Java definitions and gbench's driver at
    MSWE_AGENT_DRIVER_VERSION (its labels)."""
    image = mswe_agent_image()
    build = (f"Build it LOCALLY (gbench never pulls it):\n  docker build -t {image} -f "
             "gbench/docker/mswe_agent.Dockerfile gbench/docker\nSee " + DOCS_URL)
    if not os.path.exists(_DOCKER_SOCKET):
        return False, (f"{_DOCKER_SOCKET} does not exist; MSWE-agent starts each instance image "
                       "on the host daemon through it. " + build)
    try:
        r = subprocess.run(["docker", "image", "inspect", "--format",
                            "{{json .Config.Labels}}", image],
                           capture_output=True, text=True, timeout=60)
    except Exception as e:
        return False, f"cannot inspect the MSWE-agent image {image!r}: {e}"
    if r.returncode != 0:
        return False, f"the MSWE-agent image {image!r} is not built. " + build
    try:
        labels = json.loads(getattr(r, "stdout", "") or "null") or {}
    except ValueError:
        labels = {}
    for label, want in (
            ("org.gbench.mswe_agent", MSWE_AGENT_COMMIT),
            ("org.gbench.mswe_agent_config_sha256", MSWE_AGENT_CONFIG_SHA256),
            ("org.gbench.mswe_agent_java_repos",
             f"multi-swe-bench@{MSWE_AGENT_JAVA_REF}: {MSWE_AGENT_JAVA_ORGS}"),
            (MSWE_AGENT_DRIVER_LABEL, MSWE_AGENT_DRIVER_VERSION)):
        if labels.get(label) != want:
            return False, (f"the image {image!r} carries {label}={labels.get(label)!r}, not "
                           f"{want!r}; rebuild it. " + build)
    return True, ""


def mswe_agent_command(workdir: str, name: str, relay_url: str) -> List[str]:
    """`docker run` of the driver for one language: host network (the relay on 127.0.0.1), the
    docker socket (MSWE-agent starts each instance image on the host daemon), the language's
    work directory at /mswe (MSWE-agent reads the language from the data file's path, so the
    path must not depend on a temporary name), the caller's uid. OPENAI_API_KEY is the
    placeholder EMPTY and OPENAI_API_BASE_URL the relay (MSWE-agent's keys_config reads both
    from the environment); the relay adds the endpoint's key."""
    cmd = ["docker", "run", "--rm", "--name", name, "--network", "host",
           "-v", f"{_DOCKER_SOCKET}:{_DOCKER_SOCKET}", "-v", f"{workdir}:/mswe:rw", "-w", "/mswe",
           "-e", "HOME=/tmp", "-e", "USER=gbench", "-e", "OPENAI_API_KEY=EMPTY",
           "-e", f"OPENAI_API_BASE_URL={relay_url}"]
    if hasattr(os, "getuid"):
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]
        try:
            cmd += ["--group-add", str(os.stat(_DOCKER_SOCKET).st_gid)]
        except OSError:
            pass
    return cmd + [mswe_agent_image(), f"{_MSWE_ROOT}/venv/bin/python",
                  "/opt/gbench/mswe_agent_run.py", "--spec", "/mswe/spec.json"]


def _host_path(workdir: str, path: Optional[str]) -> Optional[str]:
    """A path the driver reported inside the container (/mswe/...) as it is on this host."""
    if not path:
        return None
    return os.path.join(workdir, os.path.relpath(path, "/mswe")) if path.startswith("/mswe") \
        else path


#: The prefix MSWE-agent's forward_with_error_check puts before a RuntimeError's text in the
#: response of the step that ends a rollout with exit_error ("exit due to runtime error: <e>").
_MSWE_EXIT_PREFIX = re.compile(r"^exit due to runtime error: ", re.IGNORECASE)


def _mswe_exit_message(traj_path: Optional[str]) -> str:
    """The response of the last step of an MSWE-agent trajectory file (for an exit_error step,
    MSWE-agent's own "exit due to runtime error: <exception>"); "" when it cannot be read."""
    if not traj_path or not os.path.isfile(traj_path):
        return ""
    try:
        with open(traj_path, encoding="utf-8") as f:
            steps = (json.load(f) or {}).get("trajectory") or []
    except (OSError, ValueError, AttributeError):
        return ""
    last = steps[-1] if steps and isinstance(steps[-1], dict) else {}
    return " ".join(str(last.get("response") or "").split())


def _mswe_exit_observation(traj_path: Optional[str]) -> str:
    """The observation of the last step of an MSWE-agent trajectory file (for an early_exit,
    SWEEnv.step's "COMMAND FAILED TO EXECUTE.", "BROKEN PIPE ERROR." or "EXECUTION TIMED OUT
    AND INTERRUPT FAILED." after the command's output); "" when it cannot be read."""
    if not traj_path or not os.path.isfile(traj_path):
        return ""
    try:
        with open(traj_path, encoding="utf-8") as f:
            steps = (json.load(f) or {}).get("trajectory") or []
    except (OSError, ValueError, AttributeError):
        return ""
    last = steps[-1] if steps and isinstance(steps[-1], dict) else {}
    text = " ".join(str(last.get("observation") or "").split())
    return text[-300:]


def _apt_archive_failure(apt: Dict[str, Any]) -> str:
    """What the driver's archive.debian.org fallback (mswe_agent_run.install_apt_archive_fallback)
    reports for an instance whose reset did not get past `apt-get update`."""
    if apt.get("applied"):
        return (f"; its apt sources ({apt.get('release')}) were pointed at archive.debian.org and "
                f"apt-get update then {apt.get('apt_update_after') or 'did not finish'}")
    return (f"; apt-get update failed ({str(apt.get('apt_error') or '')[-200:]}) and the "
            f"archive.debian.org fallback did not apply ({apt.get('release') or 'unknown system'})")


def rollout_empty_reply(crash: Optional[Dict[str, Any]]) -> bool:
    """True when the exception an MSWE-agent rollout raised (the driver's `rollout_error`) came
    from a model reply without content: the last OpenAIModel.query reply was None (`last_reply`
    "none") and the exception is the TypeError MSWE-agent's parser raises on it."""
    crash = crash or {}
    return crash.get("last_reply") == "none" and crash.get("type") == "TypeError"


#: MSWE-agent's model-side limits (sweagent/agent/models.py): raised from the model request but
#: the model's outcome (its context window, or the run's cost limit).
MSWE_MODEL_SIDE_ERRORS = ("ContextWindowExceededError", "CostLimitExceededError")


def rollout_infra_reason(crash: Optional[Dict[str, Any]]) -> Optional[str]:
    """Why a rollout exception (the driver's `rollout_error`) is not the model's outcome: it was
    raised before the model's first turn ("before the model's first turn"), inside the model
    request itself (last_reply "pending", other than MSWE_MODEL_SIDE_ERRORS), or its text starts
    with a model-endpoint failure or the docker daemon's unreachable message
    (scaffold_error_harness_reason); None otherwise."""
    crash = crash or {}
    if rollout_empty_reply(crash):
        return None
    if int(crash.get("step") or 0) < 1:
        return "before the model's first turn"
    # Raised inside the model request itself (the driver's last_reply "pending"), e.g. an
    # endpoint answering 200 with a body that is not a completion: not the model's outcome,
    # whatever the exception type, except MSWE-agent's own model-side limits.
    if (crash.get("last_reply") == "pending"
            and str(crash.get("type") or "") not in MSWE_MODEL_SIDE_ERRORS):
        return "raised inside the model request (the endpoint's reply could not be read)"
    return (scaffold_error_harness_reason(crash.get("error"))
            or scaffold_error_harness_reason(crash.get("message")))


def rollout_model_outcome(crash: Optional[Dict[str, Any]]) -> bool:
    """True when an MSWE-agent rollout exception (the driver's `rollout_error`) is the model's
    outcome: a reply without content (rollout_empty_reply), or an exception raised after the
    model's first turn that names no endpoint or docker daemon failure (rollout_infra_reason),
    the rule `exit_error` follows."""
    return bool(crash) and rollout_infra_reason(crash) is None


def mswe_agent_trace(idx: int, task: "swe_agentic.AgentTask", summary: Optional[Dict[str, Any]],
                     workdir: str, started: float) -> Dict[str, Any]:
    """One instance's trace from the driver's summary.json. Infrastructure (kept out of the
    scored set): no summary, an image that could not be pulled, a run MSWE-agent did not finish
    because its setup failed (`setup_error`), or whose rollout raised before the model's first
    turn or with a model-endpoint or docker daemon failure (`rollout_error`,
    rollout_infra_reason), or that recorded neither, `exit_api` (a tenacity RetryError), and an
    `exit_error` other than the step limit whose recorded exception starts with a model-endpoint
    failure or the docker daemon's unreachable message
    (swebench_common.scaffold_error_harness_reason). Everything else is the model's outcome as
    MSWE-agent records it (`submitted`, `submitted (exit_*)`, `exit_cost` = context window,
    `exit_format`, the step limit's `exit_error`, any other `exit_error`, and `early_exit`); the
    last two carry `agent_exit_note`. A rollout that raised on a model reply with no content
    (reasoning only: `crashed_empty_reply`), or raised after the model's first turn naming no
    endpoint or docker daemon failure (`crashed`), is the model's outcome too: no patch,
    unresolved, as upstream's run leaves it (multirun.main logs "Failed on <id>" and moves on),
    with `agent_exit_note`. A trace whose reset met the driver's archive.debian.org
    fallback carries `agent_apt_archive` (and, when the instance did not finish, its outcome in
    `agent_error`). `early_exit` is what SWEEnv.step records when the model's
    own command left the container unable to answer (the command raised RuntimeError or
    BrokenPipeError, or timed out and could not be interrupted, e.g. a shell the model killed or
    a command that hung): the model's outcome of that run, unresolved, as a single pass of
    upstream's run scores it (upstream's resume, multirun.py should_skip with
    --skip_existing, would roll such an instance out again; gbench rolls each instance out
    once). An endpoint that keeps failing raises out of MSWE-agent's rollout
    (OpenAIModel.query retries with `reraise=True`, and forward_with_error_check handles only
    RuntimeError, ContextWindowExceededError, CostLimitExceededError and RetryError), so that
    instance never completes. Model turns cut off at a limit are counted per run by the relay,
    so `truncated_turns` is None here."""
    iid = task.instance_id
    patch, status, error, steps, traj = "", "no_status", None, 0, None
    note: Optional[str] = None
    step_limit = False
    apt = (((summary or {}).get("instances") or {}).get(iid) or {}).get("apt_archive") \
        or ((summary or {}).get("apt_archive") or {}).get(iid)
    if summary is None:
        infra = True
        error = f"the MSWE-agent container wrote no summary; see {workdir}/mswe_agent.log"
    else:
        image = (summary.get("images") or {}).get(iid) or {}
        inst = (summary.get("instances") or {}).get(iid)
        if image.get("error"):
            infra, error = True, f"instance image: {image['error']}"
        elif inst and not inst.get("completed") and not inst.get("setup_error") \
                and rollout_model_outcome(inst.get("rollout_error")):
            # The rollout raised after the model's turns began, on the model's own reply (or with
            # no endpoint or docker daemon failure named): MSWE-agent ended the instance without
            # a prediction, which upstream's run leaves unresolved (no patch).
            crash = inst["rollout_error"]
            infra = False
            empty = rollout_empty_reply(crash)
            status = "crashed_empty_reply" if empty else "crashed"
            steps = int(inst.get("api_calls") or inst.get("steps") or 0)
            traj = _host_path(workdir, inst.get("trajectory_file"))
            note = (("MSWE-agent's rollout raised on a model reply with no content (reasoning "
                     "only; OpenAIModel.query returns message.content unchecked and its parser "
                     "raises on None)" if empty else
                     "MSWE-agent's rollout raised an exception that names no endpoint or docker "
                     "daemon failure")
                    + f" after {crash.get('step')} model turn(s): {crash.get('error')}; the "
                    "instance ends with no patch: the model's outcome, unresolved, as upstream's "
                    "run counts it")
        elif not inst or not inst.get("completed"):
            infra = True
            setup = (inst or {}).get("setup_error") or {}
            crash = (inst or {}).get("rollout_error") or {}
            if setup:
                where = f"its setup failed at {setup.get('step')}: {setup.get('error')}"
            elif crash:
                where = (f"its rollout raised {crash.get('error')} after {crash.get('step')} "
                         "model turn(s)" + (f" ({rollout_infra_reason(crash)})"
                                            if rollout_infra_reason(crash) else ""))
            else:
                where = "its run did not finish and recorded no setup or rollout error"
            error = (f"MSWE-agent did not finish this instance ({where}); see "
                     f"{workdir}/mswe_agent.log"
                     + (f"; {summary['run_error']}" if summary.get("run_error") else "")
                     + (f"; {summary['error']}" if summary.get("error") else "")
                     + (_apt_archive_failure(apt) if apt else ""))
        else:
            status = str(inst.get("exit_status") or "no_status")
            step_limit = bool(inst.get("step_limit_reached"))
            patch = inst.get("patch") or ""
            steps = int(inst.get("api_calls") or inst.get("steps") or 0)
            traj = _host_path(workdir, inst.get("trajectory_file"))
            if "exit_api" in status:
                infra = True
                error = (f"MSWE-agent ended the rollout with {status} (not the model's outcome: "
                         "a tenacity RetryError)")
            elif status == "early_exit":
                infra = False
                note = ("MSWE-agent ended the rollout with early_exit: the container stopped "
                        "answering while it ran the model's command (the command raised, broke "
                        "the pipe, or timed out and could not be interrupted): the model's "
                        "outcome, unresolved: "
                        + (_mswe_exit_observation(traj)[:300] or "no observation in its "
                           "trajectory"))
            elif "exit_error" in status and not step_limit:
                message = _mswe_exit_message(traj)
                reason = scaffold_error_harness_reason(_MSWE_EXIT_PREFIX.sub("", message, 1))
                infra = reason is not None
                if infra:
                    error = (f"MSWE-agent ended the rollout with {status} (not the model's "
                             f"outcome: {reason}): {message[:300]}")
                else:
                    note = (f"MSWE-agent ended the rollout with {status}, which names no endpoint "
                            "or docker daemon failure: the model's outcome, as upstream counts "
                            f"it: {message[:300] or 'no message in its trajectory'}")
            else:
                infra = False
    return {
        "sample_idx": idx, "source_sample_idx": idx, "attempt_idx": 0, "category": task.category,
        "messages": [{"role": "user", "content": task.problem_statement}],
        "extra_payload": {"instance_id": iid, **task.meta}, "gold_answer": "",
        "response_text": patch, "agent_patch": patch, "agent_exit_status": status,
        "agent_error": error, "agent_infra_error": infra, "agent_image": task.image,
        **({"agent_exit_note": note} if note else {}),
        **({"agent_apt_archive": apt} if apt else {}),
        "agent_steps": steps, "agent_step_limit_reached": step_limit,
        "trajectory": [], "trajectory_file": traj, "truncated_turns": None,
        "duration_s": round(time.time() - started, 2),
        "is_correct": False, "status": "AGENT_INFRA_ERROR" if infra else "OK"}


def run_mswe_agent(tasks: List["swe_agentic.AgentTask"], rows: Dict[str, Dict[str, Any]],
                   model_name: str, relay_url: str, concurrency: int, out_dir: str,
                   temperature: float, top_p: float,
                   client_timeout_s: Optional[float] = None) -> List[Dict[str, Any]]:
    """MSWE-agent on the selected instances, one language at a time (its data registry holds one
    data file per process, and its README runs one language per invocation); one trace per
    task, in task order.

    Before a language's container starts, its published instance images are pulled on the host
    (host_pull_images: the operator's Docker login; recorded in <lang>/image_pulls.json). An
    instance whose image cannot be pulled is not handed to MSWE-agent: its trace is an
    infrastructure failure (`instance image: ...`).

    Before any of that, the cloud metadata server check runs for the instance containers
    MSWE-agent starts (on docker's default bridge; the model's commands run there):
    swebench_common.require_task_network_blocked, probing with the MSWE-agent image, also
    checking that the package hosts of the run's languages resolve (runtime_hosts)."""
    swebench_common.require_task_network_blocked(
        "multi_swe_bench", [mswe_agent_image()], docs_url=DOCS_URL,
        hint=f"Build the MSWE-agent image {mswe_agent_image()} (see {DOCS_URL}).",
        resolve=runtime_hosts({t.category for t in tasks}))
    by_lang: Dict[str, List[Tuple[int, "swe_agentic.AgentTask"]]] = {}
    for i, t in enumerate(tasks):
        by_lang.setdefault(t.category, []).append((i, t))
    traces: List[Optional[Dict[str, Any]]] = [None] * len(tasks)
    for lang in [l for l in _LANGUAGES if l in by_lang]:
        workdir = os.path.join(out_dir, lang)
        os.makedirs(workdir, exist_ok=True)
        pulls = host_pull_images([t.image for _, t in by_lang[lang]], concurrency)
        with open(os.path.join(workdir, "image_pulls.json"), "w", encoding="utf-8") as f:
            json.dump(image_pull_summary(pulls), f, indent=1)
        failed = {t.instance_id: pulls[t.image]["error"] for _, t in by_lang[lang]
                  if pulls.get(t.image, {}).get("state") == "failed"}
        for i, t in by_lang[lang]:
            if t.instance_id in failed:
                traces[i] = mswe_agent_trace(
                    i, t, {"images": {t.instance_id: {"error": failed[t.instance_id]}},
                           "instances": {}}, workdir, time.time())
        todo = [(i, t) for i, t in by_lang[lang] if t.instance_id not in failed]
        if failed:
            logger.error("[multi_swe_bench] %d %s instance image(s) could not be pulled on the "
                         "host and count as infrastructure failures (see %s)", len(failed),
                         lang, os.path.join(workdir, "image_pulls.json"))
        if not todo:
            continue
        with open(os.path.join(workdir, "rows.jsonl"), "w", encoding="utf-8") as f:
            for _, t in todo:
                f.write(json.dumps(rows[t.instance_id]) + "\n")
        spec = {"workdir": "/mswe", "language": lang, "rows_file": "/mswe/rows.jsonl",
                "model_name": model_name, "max_context": get_run_knob("max_model_len") or 0,
                "temperature": temperature, "top_p": top_p,
                "step_limit": MSWE_AGENT_STEP_LIMIT, "workers": max(1, int(concurrency or 1)),
                "client_timeout_s": client_timeout_s or mswe_agent_timeouts().client_s}
        with open(os.path.join(workdir, "spec.json"), "w", encoding="utf-8") as f:
            json.dump(spec, f)
        name = f"gbench-mswe-agent-{uuid.uuid4().hex[:12]}"
        started = time.time()
        log_path = os.path.join(workdir, "mswe_agent.log")
        logger.info("[multi_swe_bench] MSWE-agent on %d %s instance(s), %d worker(s), in %s",
                    len(todo), lang, spec["workers"], workdir)
        try:
            with open(log_path, "w", encoding="utf-8") as log:
                subprocess.run(mswe_agent_command(workdir, name, relay_url), stdout=log,
                               stderr=subprocess.STDOUT, env=credential_free_env())
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        summary_path = os.path.join(workdir, "summary.json")
        try:
            with open(summary_path, encoding="utf-8") as f:
                summary: Optional[Dict[str, Any]] = json.load(f)
        except (OSError, ValueError):
            summary = None
        for i, t in todo:
            traces[i] = mswe_agent_trace(i, t, summary, workdir, started)
        swe_agentic.scrub_secrets_in_files(
            log_path, summary_path, *[traces[i].get("trajectory_file") for i, _ in todo])
    return [t for t in traces if t is not None]


def _run_agentic(model_name: str, base_url: str, concurrency: int, enable_thinking: bool,
                 kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """MSWE-agent rollouts through gbench's relay, then the pinned harness over the patches of
    the rollouts that ran."""
    import asyncio
    import collections
    start = time.time()
    limit = kwargs.get("limit")
    all_tasks, rows = _load_multi_swe_bench_tasks()
    tasks, shard = swe_agentic.select_tasks(all_tasks, limit, "multi_swe_bench")
    # One stratified draw over the flattened (language, instance) list: --eval-limit N is N
    # instances for the suite, spread over the languages (never N per language).
    plan = eval_limit_plan("multi_swe_bench", limit, len(tasks),
                           {lang: sum(1 for t in tasks if t.category == lang)
                            for lang in dict.fromkeys(t.category for t in all_tasks)},
                           setting_noun="language", attempts=1, unit="rollouts")
    temperature, temperature_source = resolve_official_sampling(
        "multi_swe_bench", MSWE_AGENT_SAMPLING, thinking=enable_thinking)
    top_p = float((MSWE_AGENT_SAMPLING.params or {})["top_p"])
    budget = resolve_token_budget("multi_swe_bench", MSWE_AGENT_SAMPLING,
                                  thinking=enable_thinking,
                                  max_output_tokens=kwargs.get("max_output_tokens"))
    penalties = decoding_penalties()
    body: Dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}
    body.update(penalties or {})
    # GBENCH_AGENT_MAX_TOKENS_FROM_RUN (default on; =0 turns it off): the run's --max-output-tokens on every
    # request, set by the relay (MSWE-agent sends no max_tokens), recorded as a deviation.
    run_max_tokens = swe_agentic.agent_max_tokens_from_run()
    body = swe_agentic.with_run_max_tokens(body, run_max_tokens)
    # The relay's per-request limit; run_mswe_agent gives MSWE-agent's client that plus a margin.
    timeouts = mswe_agent_timeouts()
    out_dir = swe_agentic.agent_output_dir("multi_swe_bench")
    from .base import endpoint_api_key
    # the endpoint's key (VLLM_API_KEY, else OPENAI_API_KEY; a placeholder is no key), added by
    # the relay outside the agent containers
    relay = EndpointRelay(base_url, "127.0.0.1", key=endpoint_api_key(),
                          body_fields=body, name="mswe-agent-relay",
                          timeout_s=timeouts.relay_s,
                          client_timeout_suspect_s=timeouts.suspect_s()).start()
    try:
        traces = run_mswe_agent(tasks, rows, model_name, relay.url, concurrency, out_dir,
                                float(temperature), top_p) if tasks else []
    finally:
        relay.stop()
    metrics: Dict[str, Any] = {}
    pull_records = {lang: _read_pulls(os.path.join(out_dir, lang, "image_pulls.json"))
                    for lang in _LANGUAGES}
    if any(pull_records.values()):
        metrics["image_pulls"] = {k: v for k, v in pull_records.items() if v}
    # A rollout that could not run is a harness error: out of the numerator and the denominator.
    for t in traces:
        if t["agent_infra_error"]:
            mark_harness_error(t, AGENT_INFRA, t.get("agent_error") or t.get("agent_exit_status"))
    scoreable = [t for t in traces if not t["agent_infra_error"]]
    if scoreable:
        asyncio.run(_make_scorer(model_name, concurrency, metrics)(scoreable))
    total = len(traces)
    infra = total - len(scoreable)
    # Harness errors (the agent could not run; the harness gave no verdict, wrote no report, or
    # the instance's test run lost its package hosts to a corroborated DNS failure): out of the
    # numerator and the denominator.
    head = scored_headline(traces)
    official = MSWE_AGENT_SAMPLING.record(temperature, temperature_source)
    result: Dict[str, Any] = {
        "eval_limit_plan": plan,
        "benchmark_type": "eval", "eval_name": "multi_swe_bench", "model_name": model_name,
        "thinking": enable_thinking, "mode": swe_agentic.AGENTIC,
        "mode_source": swe_agentic.resolve_mode("multi_swe_bench")[1],
        "scoring_mode": "execution", "temperature": temperature,
        "temperature_source": temperature_source,
        "official_sampling": {**official, **budget.record()},
        "max_output_tokens_per_turn": run_max_tokens, "max_output_tokens_note": (
            f"--max-output-tokens={run_max_tokens} is sent with every request (set by gbench's "
            f"relay, {swe_agentic.AGENT_MAX_TOKENS_FROM_RUN_ENV}=1); MSWE-agent sends none"
            if run_max_tokens else budget.note),
        "model_client_timeout": swe_agentic.model_client_timeout(
            "openai (MSWE-agent OpenAIModel, timeout from gbench's driver)",
            MSWE_AGENT_OPENAI_VERSION, timeouts.client_s, MSWE_AGENT_CLIENT_RETRIES,
            timeouts.source + "; the relay's limit is the same value. Transport retries: the "
            "SDK default, which driver 4 keeps (it retries an error the relay answers; it never "
            "gives up on a slow reply first)",
            replaces="the OpenAI SDK's default client timeout of 600 s with 2 retries"),
        "decoding_penalties": penalties or None,
        "dataset": f"{_REPO}@{_REVISION} ({', '.join(_LANGUAGES)})",
        "agent": {"scaffold": "MSWE-agent", "commit": MSWE_AGENT_COMMIT,
                  "config": MSWE_AGENT_CONFIG, "config_sha256": MSWE_AGENT_CONFIG_SHA256,
                  "entry": "multirun.py (RUNNING_THREADS = concurrency), one language at a time",
                  "step_limit": MSWE_AGENT_STEP_LIMIT, "top_p": top_p,
                  "java_repos_added": f"multi-swe-bench@{MSWE_AGENT_JAVA_REF}: "
                                      f"{MSWE_AGENT_JAVA_ORGS}",
                  "driver": f"gbench/docker/mswe_agent_run.py ({MSWE_AGENT_DRIVER_LABEL}="
                            f"{MSWE_AGENT_DRIVER_VERSION})",
                  "model": model_name, "image": mswe_agent_image(),
                  "openai_client_timeout_s": timeouts.client_s,
                  "relay_timeout_s": timeouts.relay_s,
                  "openai_client": _client_records(out_dir) or None},
        "relay": relay.record(), "agent_output_dir": out_dir,
        "total_questions": total, "correct_answers": head["correct_answers"],
        "scored_instances": head["scored_instances"],
        "failed_requests": head["scoring_excluded"],
        "scoring_excluded": head["scoring_excluded"],
        "accuracy": head["accuracy"],
        "category_accuracy": head["category_accuracy"],
        "agent_exit_statuses": dict(collections.Counter(str(t["agent_exit_status"])
                                                        for t in traces)),
        "agent_steps_total": sum(int(t.get("agent_steps") or 0) for t in traces),
        "agent_step_limit_reached": sum(1 for t in traces if t.get("agent_step_limit_reached")),
        "sample_traces": traces, "status": "success",
    }
    apt = {t["extra_payload"]["instance_id"]: t["agent_apt_archive"] for t in traces
           if t.get("agent_apt_archive")}
    if apt:
        # The driver's archive.debian.org fallback for the reset's apt-get update (instance images
        # of a Debian release the Debian mirrors no longer serve): recorded, not a deviation.
        result["apt_archive_fallback"] = {
            "instances": len(apt),
            "applied": sum(1 for a in apt.values() if a.get("applied")),
            "releases": dict(collections.Counter(str(a.get("release")) for a in apt.values()
                                                 if a.get("applied"))),
            "apt_update_failed_after": sorted(
                i for i, a in apt.items()
                if str(a.get("apt_update_after") or "").startswith("failed")),
            "ids": sorted(apt)}
    empty = sorted(t["extra_payload"]["instance_id"] for t in traces
                   if t.get("agent_exit_status") == "crashed_empty_reply")
    if empty:
        # rollouts MSWE-agent's parser ended on a model reply with no content: the model's
        # outcome (unresolved, no patch), counted here (also in exits_kept_as_model_outcome).
        result["empty_reply_crashes"] = len(empty)
        result["empty_reply_crash_ids"] = empty
    kept = sorted(t["extra_payload"]["instance_id"] for t in traces if t.get("agent_exit_note"))
    if kept:
        # early_exit rollouts, and exit_error rollouts that name no endpoint or docker daemon
        # failure: the model's outcome (mswe_agent_trace), scored, and counted here.
        result["exits_kept_as_model_outcome"] = len(kept)
        result["exits_kept_as_model_outcome_ids"] = kept
    attempts = int(kwargs.get("attempt_count") or get_run_knob("attempt_count", 1) or 1)
    if attempts > 1:
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
        noncanon.append(f"--eval-limit subset ({int(limit)} tasks)")
    if official.get("temperature_matches_official") is False:
        noncanon.append(f"temperature={temperature} ({temperature_source}); MSWE-agent runs at "
                        f"{MSWE_AGENT_SAMPLING.temperature}")
    if penalties:
        noncanon.append(f"decoding penalties {penalties}")
    if run_max_tokens:
        rec, why = swe_agentic.agent_max_tokens_record(run_max_tokens, "MSWE-agent",
                                                       "the relay's body fields")
        result["official_sampling"]["agent_max_tokens_from_run"] = rec
        noncanon.append(why)
    if not total:
        if shard:
            # an empty shard is an honest empty subset, with no score (not 0%)
            swe_agentic.mark_empty_shard(result, shard, len(all_tasks))
        else:
            result.update(status="error", error="no instances were selected, so nothing was "
                                                "evaluated")
        noncanon.append("no instances evaluated")
    elif infra == total:
        first = next((t for t in traces if t.get("agent_error")), {})
        result.update(status="error", error=(
            f"MSWE-agent could not run on any of {total} instance(s) (e.g. "
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
        noncanon.extend(_report_reasons(result, metrics))
        record_harness_errors(result, head)
        if result.get("status") == "success" and head["scoring_excluded"]:
            result["status"] = "completed_with_errors"
        if head["accuracy"] is None and result.get("status") != "error":
            result.update(status="error", error=(
                f"no instance was scored: all {total} are harness errors "
                f"({harness_errors_text(head['harness_errors'])}); this is an infrastructure "
                "failure, not a 0% resolved rate"))
            noncanon.append("harness infrastructure failure (no scoreable instances)")
    withhold_on_error(result)
    noncanon.extend(relay_result_reasons(result, result["relay"], "MSWE-agent"))
    result["leaderboard_comparable"] = not noncanon
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    result["duration_s"] = round(time.time() - start, 2)
    return result


def _client_records(out_dir: str) -> Dict[str, Any]:
    """language -> the OpenAI client settings the driver reported (summary.json `openai_client`)
    for the languages that ran."""
    out: Dict[str, Any] = {}
    for lang in _LANGUAGES:
        try:
            with open(os.path.join(out_dir, lang, "summary.json"), encoding="utf-8") as f:
                rec = (json.load(f) or {}).get("openai_client")
        except (OSError, ValueError, AttributeError):
            continue
        if rec:
            out[lang] = rec
    return out


def _read_pulls(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _make_scorer(model_name: str, max_workers: int, metrics: Dict[str, Any]):
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        workdir = tempfile.mkdtemp(prefix="gbench_mswe_")
        preds_path = os.path.join(workdir, "preds.jsonl")
        subset_path = os.path.join(workdir, "dataset.jsonl")
        out_dir = os.path.join(workdir, "out")
        os.makedirs(out_dir, exist_ok=True)

        # predictions: {org, repo, number, fix_patch}
        wanted = set()
        submitted: List[Tuple[Dict[str, Any], bool]] = []   # (payload, patch is empty)
        with open(preds_path, "w", encoding="utf-8") as pf:
            for tr in sample_traces:
                e = tr.get("extra_payload") or {}
                iid = e.get("instance_id")
                if not iid:
                    continue
                wanted.add(iid)
                fix_patch = trace_patch(tr, _extract_patch)
                submitted.append((e, not (fix_patch or "").strip()))
                pf.write(json.dumps({"org": e["org"], "repo": e["repo"],
                                     "number": e["number"],
                                     "fix_patch": fix_patch}) + "\n")

        # dataset subset: pull the exact rows we ran from the cached source files
        seen_files = set(_SRC_FILES.get(i) for i in wanted if _SRC_FILES.get(i))
        with open(subset_path, "w", encoding="utf-8") as sf:
            for src in seen_files:
                for line in open(src, encoding="utf-8"):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    iid = row.get("instance_id") or f"{row.get('org')}__{row.get('repo')}-{row.get('number')}"
                    if iid in wanted:
                        sf.write(line)

        # `repo_dir` is the ONE field the harness requires that has no argparse default
        # (every other one -- log_level, max_workers_build_image, stop_on_error, ... --
        # is filled in for us). Omitting it made the harness die in about a second with
        #     ValueError: Invalid repo_dir: None
        # on every run since the suite was written, so it has never produced a score.
        # `_check_repo_dir` also requires the directory to EXIST, hence the makedirs.
        #
        # It lives outside `workdir` on purpose: workdir is a fresh mkdtemp per run, and
        # with `need_clone: True` the harness clones every repo in the sample into
        # repo_dir. Putting it in the temp dir would re-clone Java/Rust/Go/C++/TS repos
        # from scratch on every single run.
        repo_dir = os.getenv("GBENCH_MULTI_SWE_REPO_DIR") or os.path.join(
            os.getenv("GBENCH_CACHE_DIR") or os.path.join(
                os.getenv("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "gbench"),
            "multi_swe_bench", "repos")
        try:
            os.makedirs(repo_dir, exist_ok=True)
        except OSError as e:
            logger.error("multi_swe_bench: cannot create repo_dir %s (%s); set "
                         "GBENCH_MULTI_SWE_REPO_DIR to a writable path.", repo_dir, e)
        if os.path.isdir(repo_dir) and not os.listdir(repo_dir):
            logger.warning(
                "multi_swe_bench: %s is empty, so this run will CLONE every repo in the "
                "sample and build one Docker image per instance (Java/Rust/Go/C++/TS). "
                "The first run is slow and disk-hungry; later runs reuse the clones.",
                repo_dir)

        config = {
            "mode": "evaluation", "workdir": workdir, "output_dir": out_dir,
            "log_dir": os.path.join(workdir, "logs"),
            "patch_files": [preds_path], "dataset_files": [subset_path],
            "repo_dir": repo_dir,
            "force_build": False, "need_clone": True, "clear_env": True,
            "max_workers": max(1, max_workers),
        }
        cfg_path = os.path.join(workdir, "config.json")
        with open(cfg_path, "w", encoding="utf-8") as cf:
            json.dump(config, cf)

        published = [published_image(e["org"], e["repo"], e["number"])
                     for e, _empty in submitted]
        if submitted:
            # The harness runs the tests over each patch in the instance's container, on docker's
            # default bridge: the cloud metadata server must be blocked there first, and the
            # package hosts the builds fetch from at test time must resolve (runtime_hosts). The
            # probe runs in the MSWE-agent image, a submitted instance's published image or any
            # local mswebench image already on the host; nothing is pulled for it. Without one it
            # is infra_required, naming the MSWE-agent build and the `docker pull` of the first
            # submitted instance's image.
            await asyncio.to_thread(
                swebench_common.require_task_network_blocked, "multi_swe_bench",
                [mswe_agent_image(), *published[1:]], docs_url=DOCS_URL,
                references=["mswebench/*"], pull=published[0],
                hint=f"Or build the MSWE-agent image {mswe_agent_image()} (see {DOCS_URL}).",
                resolve=runtime_hosts({tr.get("category") for tr in sample_traces}))
        # the harness runs through mswe_harness_build: upstream's run_evaluation with the
        # archive.debian.org fallback for image builds of archived Debian releases (the gcc:7 C++
        # base images), each use recorded in build_record
        build_record = os.path.join(workdir, "gbench_apt_archive_builds.jsonl")
        cmd, threads = swe_thread_cap.apply(
            [sys.executable, mswe_harness_build.__file__, mswe_harness_build.RECORD_FLAG,
             build_record, "--config", cfg_path],
            max_workers, "multi_swe_bench")
        metrics["docker_thread_cap"] = threads or None

        def _run():
            # the harness clones each repository over public HTTPS: the operator's own git
            # configuration (an insteadOf rewrite to SSH) must not apply (isolated_git_env)
            return subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                                  env=swebench_common.isolated_git_env())
        proc = await asyncio.to_thread(_run)
        records = mswe_harness_build.read_records(build_record)
        builds = [r for r in records if not r.get("kind")]
        # the test runs of the simdjson instances whose build fetches the checkperf reference
        # (mswe_harness_build.CHECKPERF_REFERENCE): pinned, or a harness error
        pins = [r for r in records if r.get("kind") == "checkperf_reference"]
        if pins:
            metrics["harness_checkperf_reference"] = {
                "commit": mswe_harness_build.CHECKPERF_REFERENCE["commit"],
                "pinned": sum(1 for r in pins if r.get("pinned")),
                "failed": [r.get("image") for r in pins if not r.get("pinned")],
                "records": pins}
        if builds:
            metrics["harness_apt_archive_fallback"] = {
                "images": len(builds),
                "built": sum(1 for b in builds if b.get("retry") == "ok"),
                "failed": [b.get("image") for b in builds if b.get("retry") != "ok"],
                "records": builds}

        resolved_ids: List[str] = []
        report = None
        for root, _, fs in os.walk(workdir):
            if "final_report.json" in fs:
                with open(os.path.join(root, "final_report.json"), encoding="utf-8") as f:
                    report = json.load(f)
                resolved_ids = report.get("resolved_ids", [])
                break

        def _resolved(e: Dict[str, Any]) -> bool:
            if e.get("instance_id") in resolved_ids:
                return True
            # spelling of ids varies; fall back to matching repo + number. Match the
            # number as a BOUNDED token (not a substring) so issue #5 never matches
            # "...-15" / "...-50", and tolerate org/repo vs org__repo spelling.
            rep, num = str(e.get("repo")), str(e.get("number"))
            rep_variants = {rep, rep.replace("/", "__")}
            num_re = re.compile(rf"(?<!\d){re.escape(num)}(?!\d)")
            return any(num_re.search(str(r)) and any(rv in str(r) for rv in rep_variants)
                       for r in resolved_ids)

        if report is None:
            logger.error("multi_swe_bench: final_report.json not found. stderr tail: %s",
                         (proc.stderr or "")[-800:])
            metrics["multi_swe_bench_report"] = {
                "error": "harness report not produced",
                "stderr_tail": (proc.stderr or "")[-400:],
            }
        else:
            rep = {k: report.get(k) for k in (
                "total_instances", "submitted_instances", "completed_instances",
                "incomplete_instances", "resolved_instances", "unresolved_instances",
                "error_instances", "empty_patch_instances")}
            rep["error_ids"] = list(report.get("error_ids") or [])
            rep["incomplete_ids"] = list(report.get("incomplete_ids") or [])
            # Which submissions got a verdict. The harness only reports instances whose run
            # directory exists, so one whose image never built is absent altogether, not just
            # listed in error_ids. An instance with no verdict is a harness failure unless the
            # submission was empty (the model produced nothing; unresolved, as swebench counts
            # an empty patch).
            completed = report.get("completed_ids")
            if completed is None:
                completed = list(report.get("resolved_ids") or []) + \
                    list(report.get("unresolved_ids") or [])
            done = {str(i) for i in completed}
            listed = {str(i) for i in rep["error_ids"] + rep["incomplete_ids"]}
            unscored = [e for e, empty in submitted
                        if not empty and not (_spellings(e) & done) and not _resolved(e)]
            rep["gbench_submitted_instances"] = len(submitted)
            rep["empty_submission_instances"] = sum(1 for _, empty in submitted if empty)
            rep["harness_unscored_ids"] = [e["instance_id"] for e in unscored]
            rep["harness_unscored_instances"] = len(unscored)
            # the subset the report does not mention at all (no run directory)
            rep["harness_missing_ids"] = [e["instance_id"] for e in unscored
                                          if not (_spellings(e) & listed)]
            # Unresolved instances whose fix-patch run captured no test result because the build
            # could not resolve a package host: harness errors once corroborated (another such
            # instance, or a fresh probe after the run that cannot resolve one of the hosts).
            unresolved = {str(i) for i in report.get("unresolved_ids") or []}
            # Submissions whose test run fetches simdjson's checkperf reference but whose pin
            # could not be set up (it runs before the patch is applied, so this does not depend on
            # the patch): harness errors.
            no_run = {e["instance_id"] for e in unscored}
            checkperf = checkperf_reference_failures(
                [e for e, empty in submitted if not empty and e["instance_id"] not in no_run],
                pins)
            rep["harness_checkperf_reference_failures"] = checkperf
            candidates = {}
            for e, empty in submitted:
                if empty or not (_spellings(e) & unresolved) or _resolved(e) \
                        or e["instance_id"] in checkperf:
                    continue
                hosts = harness_dns_failure(workdir, e)
                if hosts:
                    candidates[e["instance_id"]] = hosts
            images = [mswe_agent_image(), *published]
            dns_errors, dns_kept = swebench_common.corroborated_dns_failures(
                candidates, what="the harness's test run",
                recheck=lambda hosts: swebench_common.unresolved_after_run(
                    "multi_swe_bench", images, hosts, references=["mswebench/*"],
                    docs_url=DOCS_URL))
            rep["harness_dns_failures"] = {
                i: {"hosts": candidates[i], "evidence": why} for i, why in dns_errors.items()}
            rep["dns_failures_kept"] = {
                i: {"hosts": candidates[i], "note": why} for i, why in dns_kept.items()}
            metrics["multi_swe_bench_report"] = rep

        dns_errors = (metrics.get("multi_swe_bench_report") or {}).get("harness_dns_failures") \
            or {}
        dns_kept = (metrics.get("multi_swe_bench_report") or {}).get("dns_failures_kept") or {}
        checkperf = (metrics.get("multi_swe_bench_report") or {}).get(
            "harness_checkperf_reference_failures") or {}
        no_verdict = set((metrics.get("multi_swe_bench_report") or {}).get(
            "harness_unscored_ids") or ())
        with_patch = {e.get("instance_id") for e, empty in submitted if not empty}
        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("instance_id")
            tr["is_correct"] = _resolved(tr.get("extra_payload") or {})
            tr["status"] = "OK"
            if report is None and iid in with_patch:
                # No report, so no verdict for any submitted patch (an empty submission is the
                # model's outcome, unresolved, with or without the harness).
                mark_harness_error(tr, NO_REPORT, "the Multi-SWE-bench harness wrote no "
                                   "final_report.json", status="HARNESS_ERROR")
            elif iid in no_verdict:
                # A submission with a patch the harness gave no verdict (error_ids, or absent from
                # its report: image build, clone or run failure): a harness error.
                mark_harness_error(tr, NO_VERDICT, "the Multi-SWE-bench harness gave this "
                                   "instance no verdict (image build, clone or run failure)",
                                   status="HARNESS_ERROR")
            elif iid in checkperf:
                # the pinned checkperf reference was not set up before the patch was applied
                mark_harness_error(tr, CHECKPERF_REFERENCE, "the harness's test run could not "
                                   "pin simdjson's checkperf reference to "
                                   + mswe_harness_build.CHECKPERF_REFERENCE["commit"][:12]
                                   + ": " + checkperf[iid], status="HARNESS_ERROR")
            elif iid in dns_errors:
                # A harness error: left out of the numerator and the denominator.
                tr.update(is_correct=False, scoring_excluded=True, harness_error="dns",
                          dns_failure_hosts=dns_errors[iid]["hosts"],
                          dns_failure_evidence=dns_errors[iid]["evidence"])
            elif iid in dns_kept:
                tr.update(dns_failure_hosts=dns_kept[iid]["hosts"],
                          dns_failure_note=dns_kept[iid]["note"])
    return _score


def _spellings(e: Dict[str, Any]) -> set:
    """Every spelling of an instance id: the harness's `org/repo:pr-N` (PullRequestBase.id), the
    dataset's `org__repo-N`, and the row's own instance_id."""
    org, repo, number = e.get("org"), e.get("repo"), e.get("number")
    return {str(s) for s in (e.get("instance_id"), f"{org}/{repo}:pr-{number}",
                             f"{org}__{repo}-{number}") if s}


def _report_reasons(result: Dict[str, Any], metrics: Dict[str, Any]) -> List[str]:
    """The harness-report checks (both modes; the agentic runner calls it after scoring).

    No report is an error. A submission with no harness verdict (error_ids, or missing from
    the report because its image or clone failed; `harness_no_verdict`) is not model-scoreable,
    and neither is one whose test run lost its package hosts to a corroborated DNS failure
    (harness_dns_failures, `dns`): the scorer left both out of the score. When no submission got
    a verdict the run is an error (its score withheld), when some did not it is partial
    (completed_with_errors). Mutates `result` (status, error, partial_evaluation and the
    report's `harness_errors` by kind) and returns the non-comparability reasons."""
    report = metrics.get("multi_swe_bench_report")
    if not isinstance(report, dict):
        return []
    if report.get("error"):
        # A harness that did not actually evaluate is an error, not a 0% score.
        # Measured 2026-08-17: the harness died in 1 second with
        #   ValueError: Invalid repo_dir: None
        # and this suite still published "20 questions, 0 correct, 0.00%".
        result["status"] = "error"
        result["error"] = (
            "the Multi-SWE-bench harness produced no report, so nothing was "
            "evaluated; this is a harness/infrastructure failure, not a 0% "
            "resolved rate ({})".format(report.get("stderr_tail", "").strip()[-200:]
                                        or report["error"]))
        return ["harness error (no valid resolved rate)"]
    submitted = report.get("gbench_submitted_instances") or 0
    unscored = report.get("harness_unscored_instances") or 0
    dns = report.get("harness_dns_failures") or {}
    checkperf = report.get("harness_checkperf_reference_failures") or {}
    kinds = {k: v for k, v in (("dns", len(dns)), (NO_VERDICT, unscored),
                               (CHECKPERF_REFERENCE, len(checkperf))) if v}
    if kinds:
        result["harness_errors"] = {**(result.get("harness_errors") or {}), **kinds}
        result["harness_error_ids"] = sorted(set(result.get("harness_error_ids") or ())
                                             | set(dns) | set(checkperf)
                                             | set(report.get("harness_unscored_ids") or ()))
    if not submitted or not (unscored or dns or checkperf):
        return []
    missing = len(report.get("harness_missing_ids") or [])
    hosts = sorted({h for d in dns.values() for h in d.get("hosts") or ()})
    if unscored + len(dns) + len(checkperf) >= submitted:
        # Every submission had a patch and none got a verdict: the score is withheld.
        result["status"] = "error"
        result["error"] = (
            "the Multi-SWE-bench harness could not score any of {} submitted instances ({} "
            "errored, {} missing from its report: image build, clone or run failure{}); this is "
            "a harness/infrastructure failure, not a 0% resolved rate".format(
                submitted, unscored - missing, missing,
                (f"; {len(dns)} whose test run could not resolve {', '.join(hosts)}" if dns
                 else "") + (f"; {len(checkperf)} whose test run could not pin simdjson's "
                             "checkperf reference" if checkperf else "")))
        return ["harness infrastructure failure (no scoreable instances)"]
    partial = result.setdefault("partial_evaluation", {})
    partial.update({"harness_submitted_instances": submitted,
                    "harness_unscored_instances": unscored,
                    "harness_missing_instances": missing})
    if dns:
        partial["harness_dns_failure_instances"] = len(dns)
    if checkperf:
        partial["harness_checkperf_reference_failure_instances"] = len(checkperf)
    if result.get("status") == "success":
        result["status"] = "completed_with_errors"
    reasons: List[str] = []
    if unscored:
        logger.warning("[multi_swe_bench] %d of %d submitted instance(s) got no harness verdict "
                       "(image build, clone or run failure): harness errors, left out of the "
                       "score; see multi_swe_bench_report.harness_unscored_ids.", unscored,
                       submitted)
        reasons.append(f"partial: {unscored} of {submitted} submitted instance(s) hit harness "
                       "errors (not model-scoreable)")
    if dns:
        logger.warning("[multi_swe_bench] %d of %d submitted instance(s) lost their test run to a "
                       "DNS failure (%s): harness errors, left out of the score; see "
                       "multi_swe_bench_report.harness_dns_failures.", len(dns), submitted,
                       ", ".join(hosts))
        reasons.append(f"{len(dns)} of {submitted} submitted instance(s) lost to harness errors "
                       "(DNS failures in the harness's test runs), left out of the score")
    if checkperf:
        logger.warning("[multi_swe_bench] %d of %d submitted instance(s) ran without the pinned "
                       "simdjson checkperf reference (its setup failed before the patch was "
                       "applied): harness errors, left out of the score; see "
                       "multi_swe_bench_report.harness_checkperf_reference_failures.",
                       len(checkperf), submitted)
        reasons.append(f"{len(checkperf)} of {submitted} submitted instance(s) lost to harness "
                       "errors (the pinned simdjson checkperf reference could not be set up), "
                       "left out of the score")
    return reasons


def run_multi_swe_bench(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run Multi-SWE-bench resolved-rate: MSWE-agent by default, one-shot when opted in."""
    ok, reason = check_multi_swe_bench_prerequisites()
    if not ok:
        raise infra_required("multi_swe_bench", reason, DOCS_URL)
    mode, _ = swe_agentic.resolve_mode("multi_swe_bench")
    metrics: Dict[str, Any] = {}
    if mode == swe_agentic.AGENTIC:
        ok, reason = check_mswe_agent_prerequisites()
        if not ok:
            raise infra_required(
                "multi_swe_bench", f"{reason} The agentic protocol (the default) runs "
                "MSWE-agent; see the doc for the one-shot opt-in "
                f"{swe_agentic.mode_env_var('multi_swe_bench')}=oneshot and what it does not "
                "measure.", DOCS_URL)
        return _run_agentic(model_name, base_url, concurrency, enable_thinking, kwargs)
    # Every instance: run_eval_suite applies --shard to the full ordered set, then the
    # --eval-limit draw, stratified by language (the samples' category) with the loader's seed.
    samples = _load_multi_swe_bench_samples(limit=None)
    result = run_eval_suite(
        eval_name="multi_swe_bench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        async_eval_fn=_make_scorer(model_name, concurrency, metrics),
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 8192),
        temperature=kwargs.get("temperature"),
    )
    result.update(metrics)
    harness = _report_reasons(result, metrics)
    # Harness errors by kind (the scorer's, and base's endpoint failures), all left out of the
    # score; an error row reports no score.
    record_harness_errors(result, scored_headline(result.get("sample_traces") or []))
    withhold_on_error(result)
    # A single-turn issue-text -> git-diff generation with NO repository access (honoring
    # --eval-limit), whereas the benchmark is resolved in the repository. Never comparable.
    noncanon = ["single-turn generation with no repository access (the published "
                "Multi-SWE-bench numbers come from agentic scaffolds working in the repo)"]
    if kwargs.get("limit"):
        noncanon.append(f"--eval-limit subset ({int(kwargs['limit'])} tasks)")
    noncanon.extend(harness)
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    swe_agentic.mark_oneshot(result, "multi_swe_bench")
    return result
