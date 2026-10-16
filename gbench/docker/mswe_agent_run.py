#!/usr/bin/env python3
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

"""MSWE-agent rollouts of one Multi-SWE-bench language, inside the local image gbench-mswe-agent.

gbench (multi_swe_bench) writes a spec and runs this file with the image's Python 3.9 venv, the
run directory mounted at /mswe (the working directory) and the docker socket mounted. It runs
MSWE-agent's own parallel entry point, `multirun.main(multirun.get_args([...]))`, with the
arguments of MSWE-agent's README / run.sh, on the instances of one language, and writes
`summary.json` with each instance's outcome. What it adds, and why:

  * Instance images. MSWE-agent starts each instance from the image its vendored harness names
    (`<org>/<repo>:pr-<number>`, lower case) and, when that image is missing, builds it. Its README
    offers the images the Multi-SWE-bench team publishes on Docker Hub instead
    (`mswebench/<org>_m_<repo>:pr-<number>`): this driver pulls each one (unless present) and tags
    it with the name MSWE-agent looks for, so MSWE-agent uses the published image and builds
    nothing. An instance whose image cannot be pulled is left out of the rollout and reported
    (`image_error`). The tags this driver created are removed at the end.
  * The served model. MSWE-agent's OpenAIModel only drives the model names it registers, so the
    served model is registered in OpenAIModel.MODELS (zero cost, the served context window) and
    routed to OpenAIModel, which talks to OPENAI_API_BASE_URL with OPENAI_API_KEY from the
    environment (gbench's relay and the placeholder EMPTY). Sampling: run.py's `--temperature` is
    the one gbench resolved (1.0 with --thinking, 0.0 without, unless the operator set one) and
    `--top_p` is run.py's default 0.95; no max_tokens.
  * The step limit. MSWE-agent's README runs it with `--per_instance_api_calls_limit 50`, an option
    the public code does not define (it would make the argument parser exit), and every trajectory
    of the leaderboard's MSWE-agent rows (ByteDance-Seed/Multi-SWE-bench_trajs) stops at 50 model
    steps with "Exceed maximum number of steps: 50", raised as a RuntimeError from the model call
    and handled by MSWE-agent as `exit_error` with autosubmission. `install_step_limit` does exactly
    that: the 51st model step of an instance raises that RuntimeError.
  * The data file. MSWE-agent reads the language from the data file's path (specify_languages:
    `<lang>_` in the path), so each language's instances are written to
    /mswe/data/<lang>_verified.jsonl, a path in which only that language's marker occurs.
  * Archived Debian releases. MSWE-agent's reset (SWEEnv.reset) runs `apt-get update` and
    `apt-get install -y jq` in the instance container and gives up on the instance when either
    fails. The published images of the Debian releases the Debian mirrors no longer serve
    (buster: the gcc:7-based C++ images, 48 of the 1,632 instances) name deb.debian.org and
    security.debian.org, which now answer 404 for them, so that reset fails before the model's
    first turn; when upstream ran them (2025-04) the mirrors still served buster.
    `install_apt_archive_fallback` does what SWE-bench_Pro-os's LockedMiniSwe.install does for old
    Debian images: when that `apt-get update` fails because a suite is gone from its mirror, the
    container's apt sources are pointed at archive.debian.org (which serves those releases
    unchanged) with Acquire::Check-Valid-Until false, and MSWE-agent's own `apt-get update` then
    runs as upstream wrote it (it still gives up on the instance when that fails). Nothing is
    changed on an image whose `apt-get update` works. Each instance it touched is reported
    (`apt_archive` in summary.json).
  * Rollout exceptions. An exception MSWE-agent's rollout (Agent.run) raises ends the instance:
    multirun.main logs "Failed on <id>" and goes on, and the instance has no prediction (no
    patch, unresolved, as upstream's run leaves it). `install_rollout_error_record` records the
    exception (type, message, the model turns started, and what the last model reply was) in the
    instance's `rollout_error`, so gbench can tell a rollout the model ended from a setup
    failure: MSWE-agent's OpenAIModel.query returns `message.content` unchecked, and its parser
    raises TypeError on a reply that carried reasoning only (content None), which ends the
    rollout (`last_reply` "none").
  * The client's read timeout. MSWE-agent's OpenAIModel builds `OpenAI(api_key=..., base_url=...)`
    with the SDK's defaults (openai 1.70.0: 600 s per request, 2 retries), so a model turn that
    takes longer than 600 s (a long thinking turn, answered by the endpoint only when it is
    complete) is abandoned and sent again. `install_client_timeout` gives that client a timeout
    of `client_timeout_s` (gbench sets it above its relay's own per-turn limit), so such a turn
    is waited for once, the way it is sent; nothing else of the client changes (its retries
    stay). The values the client carries are reported (`openai_client` in summary.json).
"""

import argparse
import concurrent.futures
import glob
import json
import os
import re
import subprocess
import sys
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple

MSWE_ROOT = "/opt/MSWE-agent"
#: The benchmark's language directories -> the language names MSWE-agent's config uses.
MSWE_LANGUAGE = {"c": "c", "cpp": "cpp", "go": "go", "java": "java", "js": "javascript",
                 "rust": "rust", "ts": "typescript"}
#: Model steps per instance of the leaderboard's MSWE-agent runs.
STEP_LIMIT = 50
STEP_LIMIT_MESSAGE = "Exceed maximum number of steps"
#: The OpenAI client's timeout (seconds) when the spec names none: gbench's relay waits up to
#: 3600 s for one model turn (EndpointRelay.TIMEOUT_S), plus 60 s for its answer to arrive.
CLIENT_TIMEOUT_S = 3660


def hub_image(org: str, repo: str, number: Any) -> str:
    """The instance image the Multi-SWE-bench team publishes on Docker Hub."""
    return f"mswebench/{org}_m_{repo}:pr-{number}".lower()


def instances_path(workdir: str, language: str) -> str:
    """Where one language's instances go: `<lang>_verified.jsonl`, as in MSWE-agent's README."""
    return os.path.join(workdir, "data", f"{language}_verified.jsonl")


def run_args(spec: Dict[str, Any], pr_file: str) -> List[str]:
    """MSWE-agent's README / run.sh arguments, with the served model and gbench's sampling.
    `--per_instance_api_calls_limit` is not passed (the code does not define it; see
    install_step_limit). `--pre_build_all_images False`: the published images are used, so
    nothing is built (multirun passes the same value to each instance's run)."""
    return ["--model_name", spec["model_name"],
            "--cache_task_images", "True",
            "--pre_build_all_images", "False",
            "--remove_image", "False",
            "--pr_file", pr_file,
            "--config_file", os.path.join(MSWE_ROOT, "config", "default.yaml"),
            "--skip_existing=True",
            "--per_instance_cost_limit", "5.00",
            "--print_config=False",
            "--max_workers_build_image", "16",
            "--temperature", str(spec["temperature"]),
            "--top_p", str(spec["top_p"])]


def register_served_model(name: str, max_context: int) -> None:
    """Make MSWE-agent's OpenAIModel drive the served model (see the module docstring)."""
    from sweagent.agent import agents, models
    models.OpenAIModel.MODELS[name] = {"max_context": int(max_context),
                                       "cost_per_input_token": 0.0,
                                       "cost_per_output_token": 0.0}
    original = agents.get_model

    def get_model(args, commands=None):
        if args.model_name == name:
            return models.OpenAIModel(args, commands or [])
        return original(args, commands)
    agents.get_model = get_model
    models.get_model = get_model


def install_step_limit(limit: int = STEP_LIMIT) -> None:
    """The leaderboard runs' step limit: the (limit+1)-th model step of an instance raises
    RuntimeError("[<instance>] Exceed maximum number of steps: <limit>")."""
    from sweagent.agent import agents
    run, forward_model = agents.Agent.run, agents.Agent.forward_model

    def limited_run(self, setup_args, env, *args, **kwargs):
        self._gbench_steps = 0
        record = getattr(env, "record", None)
        self._gbench_instance = (record.data.get("instance_id") if record is not None else "")
        return run(self, setup_args, env, *args, **kwargs)

    def limited_forward_model(self, observation, state, lang):
        self._gbench_steps = getattr(self, "_gbench_steps", 0) + 1
        if self._gbench_steps > limit:
            raise RuntimeError(f"[{getattr(self, '_gbench_instance', '')}] "
                               f"{STEP_LIMIT_MESSAGE}: {limit}")
        return forward_model(self, observation, state, lang)
    agents.Agent.run = limited_run
    agents.Agent.forward_model = limited_forward_model


#: instance_id -> {type, error, message, step, last_reply}: the exception the instance's rollout
#: (Agent.run) raised (install_rollout_error_record).
ROLLOUT_ERRORS: Dict[str, Dict[str, Any]] = {}
_ROLLOUT_LOCK = threading.Lock()


def reply_kind(reply: Any) -> str:
    """What one OpenAIModel.query reply was: "none" (no content: a reply with reasoning only),
    "empty" (blank text) or "text"."""
    if reply is None:
        return "none"
    return "empty" if isinstance(reply, str) and not reply.strip() else "text"


def install_rollout_error_record() -> None:
    """Record the exception an instance's rollout raises (see the module docstring): its type, its
    text (`error`: "<type>: <message>", the type qualified by its package for the OpenAI SDK,
    httpx and the docker SDK) and message, the model turns started (`step`, counted by
    install_step_limit) and the kind of the last reply OpenAIModel.query returned (`last_reply`:
    reply_kind, or "pending" when the query raised, "unknown" before any). The exception is raised on unchanged, so MSWE-agent handles it
    as upstream does. Call after install_step_limit."""
    from sweagent.agent import agents, models
    run, query = agents.Agent.run, models.OpenAIModel.query

    def tracked_query(self, history):
        self._gbench_last_reply = "pending"
        reply = query(self, history)
        self._gbench_last_reply = reply_kind(reply)
        return reply

    def recorded_run(self, setup_args, env, *args, **kwargs):
        try:
            return run(self, setup_args, env, *args, **kwargs)
        except Exception as e:
            record = getattr(env, "record", None)
            iid = str(getattr(self, "_gbench_instance", "")
                      or (record.data.get("instance_id") if record is not None else "") or "")
            module = type(e).__module__ or ""
            name = (f"{module.split('.')[0]}.{type(e).__name__}"
                    if module.split(".")[0] in ("openai", "httpx", "docker") else type(e).__name__)
            with _ROLLOUT_LOCK:
                ROLLOUT_ERRORS[iid] = {
                    "type": type(e).__name__, "error": _tail(f"{name}: {e}", 300),
                    "message": _tail(e, 300),
                    "step": int(getattr(self, "_gbench_steps", 0) or 0),
                    "last_reply": getattr(getattr(self, "model", None), "_gbench_last_reply",
                                          "unknown")}
            raise
    agents.Agent.run = recorded_run
    models.OpenAIModel.query = tracked_query


def install_client_timeout(seconds: float) -> Dict[str, Any]:
    """Give the OpenAI client MSWE-agent's OpenAIModel builds a timeout of `seconds` (see the
    module docstring); returns what a client built that way carries (timeout, read timeout,
    retries)."""
    import openai
    from sweagent.agent import models
    original = models.OpenAI

    def client(*args, **kwargs):
        kwargs.setdefault("timeout", openai.Timeout(float(seconds)))
        return original(*args, **kwargs)
    models.OpenAI = client
    probe = client(api_key="EMPTY", base_url="http://127.0.0.1:1/v1")
    timeout = probe.timeout
    return {"timeout": str(timeout),
            "read_timeout_s": getattr(timeout, "read", timeout),
            "max_retries": probe.max_retries, "openai": openai.__version__}


#: What apt prints when a suite it is configured with is gone from its mirror: apt 1.1 and later
#: ("The repository '<uri> <suite> Release' no longer has a Release file." or "... does not have
#: a Release file.") and older apt ("404  Not Found" for the suite's files).
ARCHIVED_SUITE_RE = re.compile(r"no longer has a Release file|does not have a Release file|"
                               r"404\s+Not Found")
#: The first word of each line the repair prints: "<mark> applied (<release>)" or "<mark> not
#: applied (<os id> <release>)".
APT_ARCHIVE_MARK = "apt archive fallback:"
#: The repair, one shell line run in the instance container (MSWE-agent's session is bash; the
#: script runs in `sh -c`, so its `exit` ends only that shell). For a Debian release that
#: archive.debian.org serves (wheezy, jessie, stretch, buster, bullseye; the release from
#: VERSION_CODENAME, or from VERSION on jessie, whose os-release has no VERSION_CODENAME), every
#: apt source line of /etc/apt/sources.list and /etc/apt/sources.list.d/*.list that names a
#: debian.org mirror is pointed at archive.debian.org/debian, and the security suites at
#: archive.debian.org/debian-security; the -updates suites of wheezy, jessie and stretch, which
#: the archive does not carry (its dists/ holds -updates for buster and bullseye alone), are
#: dropped; Acquire::Check-Valid-Until is set false (the archive's Release files have expired
#: Valid-Until dates). Any other system is left as it is ("not applied").
APT_ARCHIVE_REPAIR = (
    r"""sh -c '. /etc/os-release 2>/dev/null; c="${VERSION_CODENAME:-}"; [ -n "$c" ] || """
    r"""c=$(echo "${VERSION:-}" | sed -n "s/.*(\([a-z]*\)).*/\1/p"); case "${ID:-}:$c" in """
    r"""debian:wheezy|debian:jessie|debian:stretch|debian:buster|debian:bullseye) ;; *) """
    r"""echo "apt archive fallback: not applied (${ID:-unknown} ${c:-unknown})"; exit 0;; esac; """
    r"""for f in /etc/apt/sources.list /etc/apt/sources.list.d/*.list; do [ -f "$f" ] || """
    r"""continue; sed -i -e "s#//security\.debian\.org/debian-security#//archive.debian.org/"""
    r"""debian-security#g" -e "s#//deb\.debian\.org/debian-security#//archive.debian.org/"""
    r"""debian-security#g" -e "s#//security\.debian\.org/*\([[:space:]]\)#//archive.debian.org/"""
    r"""debian-security\1#g" -e "s#//[a-z0-9.-]*\.debian\.org/debian\([/[:space:]]\)#//"""
    r"""archive.debian.org/debian\1#g" -e "/[[:space:]]\(wheezy\|jessie\|stretch\)-updates"""
    r"""[[:space:]]/d" "$f"; done; echo "Acquire::Check-Valid-Until \"false\";" > """
    r"""/etc/apt/apt.conf.d/99archive-debian-valid-until; echo "apt archive fallback: applied """
    r"""($c)"'""")
_APT_RESULT_RE = re.compile(re.escape(APT_ARCHIVE_MARK) + r" (applied|not applied) \(([^)]*)\)")
#: instance_id -> what the fallback did in that instance's container (summary.json `apt_archive`).
APT_ARCHIVE: Dict[str, Dict[str, Any]] = {}
_APT_LOCK = threading.Lock()


def _tail(text: Any, limit: int = 600) -> str:
    return " ".join(str(text or "").split())[-limit:]


def install_apt_archive_fallback() -> None:
    """MSWE-agent's reset `apt-get update` on an image whose Debian release its mirror no longer
    serves (see the module docstring). SWEEnv.communicate_with_handling closes the container
    when a command fails, so the first `apt-get update` of each reset is sent with
    SWEEnv.communicate (the same call communicate_with_handling makes): when it succeeds, its
    output is returned, as communicate_with_handling would; when apt says a suite is gone
    (ARCHIVED_SUITE_RE), APT_ARCHIVE_REPAIR runs and is recorded; then (after any other failure
    too) MSWE-agent's own communicate_with_handling runs the `apt-get update`, with its own
    failure path."""
    from sweagent.environment import swe_env
    env_cls = swe_env.SWEEnv
    reset, handled = env_cls.reset, env_cls.communicate_with_handling

    def tracked_reset(self, instance_id, *args, **kwargs):
        self._gbench_instance_id = instance_id
        self._gbench_apt_checked = False
        try:
            return reset(self, instance_id, *args, **kwargs)
        except Exception as e:
            # MSWE-agent's Main logs "Failed on <id>" and moves on: record which setup step
            # failed, so the instance's harness error names it
            iid = str(instance_id if not isinstance(instance_id, int)
                      else getattr(self, "_gbench_instance_id", instance_id))
            with _APT_LOCK:
                SETUP_ERRORS[iid] = {"step": setup_step(e, APT_ARCHIVE.get(iid)),
                                     "error": _tail(f"{type(e).__name__}: {e}", 300)}
            raise

    def communicate_with_handling(self, input, error_msg, timeout_duration=25,
                                  no_output_timeout_duration=25, except_error_msgs=[]):
        def upstream():
            return handled(self, input, error_msg, timeout_duration, no_output_timeout_duration,
                           except_error_msgs)
        if input.strip() != "apt-get update" or getattr(self, "_gbench_apt_checked", True):
            return upstream()
        self._gbench_apt_checked = True
        logs = self.communicate(input, timeout_duration=timeout_duration,
                                no_output_timeout_duration=no_output_timeout_duration)
        if self.returncode == 0:
            return logs
        if not ARCHIVED_SUITE_RE.search(logs or ""):
            return upstream()
        out = self.communicate(APT_ARCHIVE_REPAIR, timeout_duration=120,
                               no_output_timeout_duration=120)
        found = _APT_RESULT_RE.search(out or "")
        note: Dict[str, Any] = {
            "applied": bool(found and found.group(1) == "applied"),
            "release": found.group(2) if found else None,
            "apt_error": _tail(logs)}
        if not found:
            note["repair_output"] = _tail(out)
        iid = str(getattr(self, "_gbench_instance_id", "") or "")
        with _APT_LOCK:
            APT_ARCHIVE[iid] = note
        try:
            result = upstream()
        except RuntimeError as e:
            note["apt_update_after"] = "failed: " + _tail(e, 300)
            raise
        note["apt_update_after"] = "ok"
        return result
    env_cls.reset = tracked_reset
    env_cls.communicate_with_handling = communicate_with_handling


#: instance_id -> {step, error}: the SWEEnv.reset of the instance that raised (tracked_reset).
SETUP_ERRORS: Dict[str, Dict[str, Any]] = {}


def setup_step(error: BaseException, apt_note: Optional[Dict[str, Any]] = None) -> str:
    """Which setup step of MSWE-agent's SWEEnv.reset a raised error came from, by its text:
    apt (the reset's `apt-get update`, or an archive repair recorded for the instance),
    repo_reset (git clone / checkout / reset of the repository), container_start (the docker
    container or its image), else other."""
    text = f"{type(error).__name__}: {error}".lower()
    if apt_note or "apt-get" in text or "apt " in text or " apt" in text:
        return "apt"
    if any(w in text for w in ("git ", "clone", "checkout", "repository", "repo ")):
        return "repo_reset"
    if any(w in text for w in ("docker", "container", "image")):
        return "container_start"
    return "other"


def default_image_name() -> None:
    """The six Java repositories added to the vendored harness come from a later multi-swe-bench
    tree whose Image classes inherit image_name from the base class; the vendored base class has
    none. Give them the vendored harness's naming (`<org>/<repo>`, lower case), which every
    vendored repository defines for itself."""
    from multi_swe_bench.harness import image as image_mod

    def image_name(self) -> str:
        return f"{self.pr.org}/{self.pr.repo}".lower()
    image_mod.Image.image_name = image_name


def local_image_names(rows: List[Dict[str, Any]]) -> Dict[str, str]:
    """instance_id -> the image name MSWE-agent starts the instance from (its vendored harness's
    Instance.name())."""
    import multi_swe_bench.harness.repos  # noqa: F401  (registers the repositories)
    from multi_swe_bench.harness.image import Config
    from multi_swe_bench.harness.instance import Instance
    from multi_swe_bench.harness.pull_request import PullRequest
    config = Config(need_clone=True, global_env={}, clear_env=False)
    return {r["instance_id"]: Instance.create(PullRequest.from_dict(r), config).name()
            for r in rows}


def _docker(*args: str, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def ensure_image(hub: str, local: str) -> Tuple[Optional[str], bool]:
    """(error or None, whether a tag was created): `local` exists, or `hub` is pulled (unless
    present) and tagged `local`."""
    if _docker("image", "inspect", local).returncode == 0:
        return None, False
    if _docker("image", "inspect", hub).returncode != 0:
        pulled = _docker("pull", hub)
        if pulled.returncode != 0:
            return f"docker pull {hub} failed: {(pulled.stderr or pulled.stdout)[-300:]}", False
    tagged = _docker("tag", hub, local)
    if tagged.returncode != 0:
        return f"docker tag {hub} {local} failed: {(tagged.stderr or tagged.stdout)[-300:]}", False
    return None, True


def collect(workdir: str, instance_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Each instance's outcome from MSWE-agent's own files: trajectories/<user>/<run>/<id>.traj
    and all_preds.jsonl (written when the instance's run completed), with `apt_archive` when
    install_apt_archive_fallback acted in the instance's container."""
    root = os.path.join(workdir, "trajectories")
    trajs = {os.path.basename(p)[:-len(".traj")]: p
             for p in glob.glob(os.path.join(root, "**", "*.traj"), recursive=True)}
    preds: Dict[str, Any] = {}
    for path in glob.glob(os.path.join(root, "**", "all_preds.jsonl"), recursive=True):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    preds[row["instance_id"]] = row.get("model_patch")
    out: Dict[str, Dict[str, Any]] = {}
    for iid in instance_ids:
        info: Dict[str, Any] = {}
        steps: List[Dict[str, Any]] = []
        path = trajs.get(iid)
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                info, steps = data.get("info") or {}, data.get("trajectory") or []
            except (OSError, ValueError):
                pass
        last = steps[-1] if steps else {}
        out[iid] = {
            "completed": iid in preds,
            "patch": preds.get(iid) or "",
            "exit_status": info.get("exit_status"),
            "api_calls": (info.get("model_stats") or {}).get("api_calls"),
            "steps": len(steps),
            "step_limit_reached": (last.get("action") == "exit_error"
                                   and STEP_LIMIT_MESSAGE in str(last.get("response") or "")),
            "trajectory_file": path,
        }
        if iid in APT_ARCHIVE:
            out[iid]["apt_archive"] = APT_ARCHIVE[iid]
        if iid in SETUP_ERRORS and iid not in preds:
            out[iid]["setup_error"] = SETUP_ERRORS[iid]
        if iid in ROLLOUT_ERRORS and iid not in preds:
            out[iid]["rollout_error"] = ROLLOUT_ERRORS[iid]
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    spec_path = parser.parse_args(argv).spec
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    workdir = spec["workdir"]
    os.chdir(workdir)
    sys.path.insert(0, MSWE_ROOT)
    language = spec["language"]
    with open(spec["rows_file"], encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    summary: Dict[str, Any] = {"language": language, "instances": {}, "images": {},
                               "created_tags": []}
    created: List[str] = []
    try:
        default_image_name()
        from sweagent.environment.utils import specify_languages
        pr_file = instances_path(workdir, language)
        detected = specify_languages(pr_file)
        if detected != MSWE_LANGUAGE[language]:
            raise RuntimeError(f"MSWE-agent reads language {detected!r} from {pr_file}, not "
                               f"{MSWE_LANGUAGE[language]!r}")
        names = local_image_names(rows)
        ready: List[Dict[str, Any]] = []

        def _prepare(row):
            hub = hub_image(row["org"], row["repo"], row["number"])
            return row, hub, ensure_image(hub, names[row["instance_id"]])
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(spec["workers"]))) as ex:
            for row, hub, (error, made) in ex.map(_prepare, rows):
                iid = row["instance_id"]
                summary["images"][iid] = {"published": hub, "mswe_agent": names[iid],
                                          "error": error}
                if made:
                    created.append(names[iid])
                if error is None:
                    ready.append(row)
        summary["created_tags"] = list(created)
        os.makedirs(os.path.dirname(pr_file), exist_ok=True)
        with open(pr_file, "w", encoding="utf-8") as f:
            for row in ready:
                f.write(json.dumps(row) + "\n")
        if ready:
            register_served_model(spec["model_name"], spec.get("max_context") or 0)
            install_step_limit(int(spec.get("step_limit") or STEP_LIMIT))
            install_rollout_error_record()
            install_apt_archive_fallback()
            summary["openai_client"] = install_client_timeout(
                float(spec.get("client_timeout_s") or CLIENT_TIMEOUT_S))
            os.environ["RUNNING_THREADS"] = str(max(1, int(spec["workers"])))
            import multirun
            try:
                multirun.main(multirun.get_args(run_args(spec, pr_file)))
            except Exception as e:     # an instance's Main could not be built; the rest ran
                summary["run_error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
        summary["instances"] = collect(workdir, [r["instance_id"] for r in ready])
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        for tag in created:
            _docker("image", "rm", tag)
        with _APT_LOCK:
            summary["apt_archive"] = dict(APT_ARCHIVE)
        with open(os.path.join(workdir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
