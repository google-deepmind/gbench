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

"""A skillsbench task whose environment build failed in the Docker daemon is retried, and a task
with no reward says why.

2026-10-13 validation run (pool-10-13-003542, --eval-limit 5, thinking on, eight suites at once):
lab-unit-harmonization wrote no reward. Its rollout result.json shows the task environment's
`docker compose build` failing in step 4/7 after the step's `pip install` had finished: the
daemon's containerd image store could not commit the layer ("failed to export layer: CreateDiff:
... failed to commit: rename .../ingest/<ref>/data .../blobs/sha256/<digest>: no such file or
directory"), 78 s into the trial, before the agent started (0 tool calls, 0 tokens, empty
trajectory). The same Dockerfile built and was scored on 2026-10-10 and 2026-10-11. BenchFlow
classified the error "other" and did not retry it: its docker sandbox retries a failed build only
for the failures in _DOCKER_BUILD_RETRYABLE_ERRORS. The run reported {"no_reward": 1} with no
cause.

Now the image's skillsbench_benchflow_build_retry.patch adds that failure to BenchFlow's list, the
launcher records each retry (sandbox_build_retries), and the runner's no-reward entry carries the
rollout's error category, failed stage, whether the agent ran and the error's end. The task stays
without a reward: it is never scored 0.

No network, no docker: the orchestrator is faked; the patch's pattern is compiled from the file.
"""

import json
import os
import re
import subprocess
import sys

import pytest

from gbench.runners.eval_suites import base as B
from gbench.runners.eval_suites import skillsbench as SB

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "gbench", "docker"))
import skillsbench_run as SRUN  # noqa: E402

PATCH = os.path.join(REPO, "gbench", "docker", "skillsbench_benchflow_build_retry.patch")
DOCKERFILE = os.path.join(REPO, "gbench", "docker", "skillsbench.Dockerfile")

#: The 2026-10-13 error, shortened in the middle (the build log), its start and end verbatim.
CONTAINERD_ERROR = (
    "Docker compose command failed for environment lab-unit-harmonization. Command: docker "
    "compose --project-name lab-unit-harmonization__f7d62a94 --project-directory "
    "/tmp/gbench_skillsbench_0wc01oum/benchflow-task-oplzap9a/lab-unit-harmonization/environment "
    "-f /app/skillsbench/.venv/lib/python3.12/site-packages/benchflow/sandbox/_compose_files/"
    "docker-compose-base.yaml -f /app/skillsbench/.venv/lib/python3.12/site-packages/benchflow/"
    "sandbox/_compose_files/docker-compose-build.yaml build. Return code: 1. Stdout: "
    "time=\"2026-10-13T01:29:35Z\" level=warning msg=\"buildx Docker CLI plugin not found: "
    "falling back to the classic builder.\"\nStep 4/7 : RUN pip install --no-cache-dir "
    "numpy==1.26.4 pandas==2.2.2\nSuccessfully installed numpy-1.26.4 pandas-2.2.2\n"
    "failed to export layer: CreateDiff: mount callback failed on "
    "/var/lib/containerd/tmpmounts/containerd-mount2570210320: mount callback failed on "
    "/var/lib/containerd/tmpmounts/containerd-mount779052287: failed to commit: rename "
    "/var/lib/containerd/io.containerd.content.v1.content/ingest/171f29bdcf0a45b9907a3ad715ee220"
    "52b18061fe48a0b1986e54a25331cf5ec/data /var/lib/containerd/io.containerd.content.v1.content/"
    "blobs/sha256/deab47ce585d853a4f1b31662b36b8687732981a373bca173477c80d9222863e: no such file "
    "or directory\n. Stderr: None. ")

#: The 2026-10-13 rollout result.json, the fields the runner reads.
NO_REWARD_RESULT = {
    "task_name": "lab-unit-harmonization", "rollout_name": "lab-unit-harmonization__f7d62a94",
    "rewards": None, "n_tool_calls": 0, "n_prompts": 1,
    "agent_result": {"n_tool_calls": 0, "n_input_tokens": 0, "n_output_tokens": 0,
                     "total_tokens": 0},
    "trajectory_summary": {"steps": 0}, "error": CONTAINERD_ERROR, "error_category": "other",
    "verifier_error": None, "timing": {"total": 78.0}}


def _patch_patterns():
    """BenchFlow's _DOCKER_BUILD_RETRYABLE_ERRORS entries the patch adds, compiled."""
    added = "\n".join(line[1:] for line in open(PATCH, encoding="utf-8").read().splitlines()
                      if line.startswith("+") and not line.startswith("+++"))
    added = "\n".join(line for line in added.splitlines() if not line.strip().startswith("#"))
    return eval("(" + added + ")", {"re": re})   # noqa: S307 - the repo's own patch file


def test_the_patch_retries_the_containerd_commit_failure_and_nothing_else():
    patterns = _patch_patterns()
    assert len(patterns) == 1
    assert patterns[0].search(CONTAINERD_ERROR)
    # docker's embedded containerd keeps its store under /var/lib/docker/containerd/daemon/
    assert patterns[0].search(CONTAINERD_ERROR.replace(
        "/var/lib/containerd/", "/var/lib/docker/containerd/daemon/"))
    for other in (
            "Step 4/7 : RUN pip install x\nERROR: No matching distribution found for x\nThe "
            "command '/bin/sh -c pip install x' returned a non-zero code: 1",
            "failed to export layer: write /var/lib/x: input/output error",
            "failed to commit: rename /a/ingest/1/data /a/blobs/sha256/2: permission denied",
            "failed to solve: process \"/bin/sh -c make\" did not complete successfully"):
        assert not patterns[0].search(other), other


def test_the_patch_targets_the_checked_upstream_file():
    text = open(PATCH, encoding="utf-8").read()
    assert "--- a/benchflow/sandbox/docker.py\n+++ b/benchflow/sandbox/docker.py\n" in text
    assert text.count("\n+    re.compile(") == 1          # one new entry, nothing removed
    assert not re.search(r"^-(?!--)", text, re.M)
    df = open(DOCKERFILE, encoding="utf-8").read()
    assert "COPY skillsbench_benchflow_build_retry.patch" in df
    assert ("ARG BENCHFLOW_DOCKER_SANDBOX_UPSTREAM_SHA256=426645c0be56d0f441869ef752b41b6ba0e15ec"
            "e1e54dac359110acc8d4fe5f2") in df
    assert '"${BENCHFLOW_DOCKER_SANDBOX_SHA256}  benchflow/sandbox/docker.py" | sha256sum -c -' \
        in df
    assert 'LABEL org.gbench.benchflow_docker_sandbox_sha256="${BENCHFLOW_DOCKER_SANDBOX_SHA256}"' \
        in df
    assert SB.HARNESS_VERSION == "7" and SB._IMAGE_DEFAULT == "gbench-skillsbench:9a1f4dd5-h7"


def test_the_launcher_counts_benchflows_build_retries(monkeypatch):
    warning = ("Retrying Docker build for lab-unit-harmonization after transient failure "
               "(attempt {}/3, retrying in 2.0s): " + CONTAINERD_ERROR)
    stderr = "\n".join(["Rollout starting", warning.format(1), warning.format(2), "Score: 0/1"])
    assert SRUN.build_retries(stderr) == 2
    assert SRUN.build_retries("") == SRUN.build_retries(None) == 0
    assert SRUN.build_retries(CONTAINERD_ERROR) == 0       # the error alone is not a retry
    monkeypatch.setattr(SRUN.subprocess, "run",
                        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, "", stderr))
    monkeypatch.setattr(SRUN, "_reward_from_jobs", lambda jobs_dir: 0.5)
    retried = {}
    assert SRUN._run_one("lab-unit-harmonization", "without-skills", "oracle", "/tmp/j",
                         retried) == 0.5
    assert retried == {"lab-unit-harmonization": 2}
    monkeypatch.setattr(SRUN.subprocess, "run",
                        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, "", "Score: 1/1"))
    retried = {}
    SRUN._run_one("t", "without-skills", "oracle", "/tmp/j", retried)
    assert retried == {}


def test_the_launcher_summary_records_the_retries(monkeypatch, tmp_path):
    monkeypatch.setattr(SRUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("SKILLSBENCH_AGENT", "oracle")
    monkeypatch.setenv("SKILLSBENCH_TASKS", "a,b")
    monkeypatch.delenv("GBENCH_SHARD", raising=False)

    def one(task, cond, agent, jobs_dir, retries=None):
        if task == "b":
            retries[task] = 1
        return 1.0
    monkeypatch.setattr(SRUN, "_run_one", one)
    assert SRUN.main() == 0
    summary = json.loads((tmp_path / "skillsbench_summary.json").read_text())
    assert summary["sandbox_build_retries"] == {"without-skills": {"b": 1}}
    monkeypatch.setattr(SRUN, "_run_one", lambda *a, **k: 1.0)
    SRUN.main()
    summary = json.loads((tmp_path / "skillsbench_summary.json").read_text())
    assert summary["sandbox_build_retries"] == {}


def test_the_2026_10_13_rollout_is_an_environment_build_failure_before_the_agent_ran():
    cause = SB.no_reward_cause(NO_REWARD_RESULT)
    assert cause["benchflow_error_category"] == "other"
    assert cause["stage"] == "environment_build" and cause["agent_ran"] is False
    assert cause["error_excerpt"].endswith(": no such file or directory . Stderr: None.")
    assert len(cause["error_excerpt"]) == SB.ERROR_EXCERPT_CHARS
    up = dict(NO_REWARD_RESULT, error=CONTAINERD_ERROR.replace(
        "docker-compose-build.yaml build. Return", "docker-compose-build.yaml up --detach "
        "--wait. Return"))
    assert SB.no_reward_cause(up)["stage"] == "environment_start"
    # "build" in the compose file's name or the project path is not the subcommand
    exec_cmd = dict(NO_REWARD_RESULT, error=CONTAINERD_ERROR.replace(
        "docker-compose-build.yaml build. Return", "docker-compose-build.yaml exec -T main "
        "make. Return"))
    assert SB.no_reward_cause(exec_cmd)["stage"] == "sandbox_command"
    assert SB.no_reward_cause(exec_cmd)["compose_command"] == "exec"
    ran = dict(NO_REWARD_RESULT, error="Agent idle for 600s", error_category="idle_timeout",
               agent_result={"n_tool_calls": 4, "total_tokens": 9000})
    assert SB.no_reward_cause(ran) == {"benchflow_error_category": "idle_timeout",
                                       "stage": "agent", "agent_ran": True,
                                       "error_excerpt": "Agent idle for 600s"}
    verifier = dict(NO_REWARD_RESULT, error=None, error_category=None,
                    verifier_error="verifier crashed: exit 2", n_tool_calls=3)
    assert SB.no_reward_cause(verifier)["stage"] == "verifier"
    assert SB.no_reward_cause(verifier)["error_excerpt"] == "verifier crashed: exit 2"


def _workdir(tmp_path, result=NO_REWARD_RESULT):
    rollout = (tmp_path / "jobs" / "without-skills" / "lab-unit-harmonization"
               / "2026-10-13__01-29-35" / "lab-unit-harmonization__f7d62a94")
    rollout.mkdir(parents=True)
    if result is not None:
        (rollout / "result.json").write_text(
            result if isinstance(result, str) else json.dumps(result))
    return str(tmp_path)


SUMMARY = {"conditions": {"without-skills": {
    "per_task": {"debug-trl-grpo": 0.25, "lab-unit-harmonization": None},
    "mean_reward": 0.25, "n_scored": 1, "n_tasks": 2}},
    "sandbox_build_retries": {"without-skills": {"lab-unit-harmonization": 2}}}


def test_no_reward_causes_reads_the_rollout_result(tmp_path):
    causes = SB.no_reward_causes(_workdir(tmp_path), SUMMARY)
    cause = causes["without-skills"]["lab-unit-harmonization"]
    assert list(causes["without-skills"]) == ["lab-unit-harmonization"]   # scored tasks: none
    assert cause["stage"] == "environment_build" and cause["agent_ran"] is False
    assert cause["rollout"] == "lab-unit-harmonization__f7d62a94"


@pytest.mark.parametrize("result,note", [(None, "wrote no result.json"),
                                         ("{not json", "could not be read"),
                                         ("[1]", "could not be read")])
def test_no_reward_causes_without_a_readable_result(tmp_path, result, note):
    cause = SB.no_reward_causes(_workdir(tmp_path, result), SUMMARY)[
        "without-skills"]["lab-unit-harmonization"]
    assert cause["stage"] is None and note in cause["note"]


def test_the_cause_rides_on_the_no_reward_entry_and_the_task_stays_unscored(tmp_path):
    causes = SB.no_reward_causes(_workdir(tmp_path), SUMMARY)
    scoring = SB.exclude_harness_errors(SUMMARY, {}, causes)
    info = scoring["conditions"]["without-skills"]
    entry = info["harness_errors"]["lab-unit-harmonization"]
    assert entry["kind"] == "no_reward" and entry["stage"] == "environment_build"
    assert entry["build_retries"] == 2 and entry["agent_ran"] is False
    assert info["per_task"]["lab-unit-harmonization"] is None          # never a 0
    assert info["mean_reward"] == 0.25 and info["n_scored"] == 1
    trace = [t for t in SB.task_traces(scoring) if t["task"] == "lab-unit-harmonization"][0]
    assert trace["reward"] is None and trace["scoring_excluded"] is True
    assert trace["harness_error"]["stage"] == "environment_build"
    # without causes (callers that pass none) the entry is the bare kind, as before
    bare = SB.exclude_harness_errors(dict(SUMMARY, sandbox_build_retries={}), {})
    assert bare["conditions"]["without-skills"]["harness_errors"] == {
        "lab-unit-harmonization": {"kind": "no_reward"}}


def test_run_skillsbench_reports_the_cause_and_the_retries(monkeypatch, tmp_path):
    monkeypatch.setitem(B._RUN_KNOBS, "max_model_len", 262144)
    for k in ("GBENCH_SKILLSBENCH_API_KEY", "VLLM_API_KEY", "OPENAI_API_KEY",
              "GBENCH_SKILLSBENCH_TASK_ENDPOINT", "GBENCH_SKILLSBENCH_TEMPERATURE", "GBENCH_SHARD"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GBENCH_SKILLSBENCH_TASK_HOST", "127.0.0.1")
    monkeypatch.setattr(SB, "check_skillsbench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(SB.swebench_common, "require_task_network_blocked", lambda *a, **k: None)
    monkeypatch.setattr(SB, "dns_harness_errors", lambda workdir, summary: ({}, {}))
    monkeypatch.setattr(SB, "keep_artifacts", lambda workdir, dest: None)
    monkeypatch.setattr(SB, "remove_workdir", lambda workdir, image=None: [])

    def run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            workdir = next(a.split("=", 1)[1] for a in cmd
                           if a.startswith("SKILLSBENCH_WORKDIR="))
            _workdir(__import__("pathlib").Path(workdir))
            with open(os.path.join(workdir, "skillsbench_summary.json"), "w") as f:
                json.dump(dict(SUMMARY, n_tasks_selected=2,
                               tasks_selected=["debug-trl-grpo", "lab-unit-harmonization"]), f)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(SB.subprocess, "run", run)
    res = SB.run_skillsbench("m", "http://127.0.0.1:9/v1", 1, enable_thinking=True)
    assert res["status"] == "completed_with_errors"
    assert res["harness_errors"] == {"no_reward": 1}
    entry = res["harness_error_tasks"]["without-skills"]["lab-unit-harmonization"]
    assert entry["kind"] == "no_reward" and entry["stage"] == "environment_build"
    assert entry["benchflow_error_category"] == "other" and entry["build_retries"] == 2
    assert res["sandbox_build_retries"] == {"without-skills": {"lab-unit-harmonization": 2}}
    assert res["accuracy"] == 25.0 and res["effective_n"] == 1 and res["total_questions"] == 2


@pytest.mark.parametrize("change, stage", [
    # another compose command BenchFlow checks (copying the skills in): not the agent
    ({"error": "Docker compose command failed for environment x. Command: docker compose "
               "--project-name x -f /a/docker-compose.yaml -f /b/build.yaml cp /skills "
               "main:/skills. Return code: 1. Stdout: no such container.",
      "error_category": "other"}, "sandbox_command"),
    ({"error": "sandbox startup failed: timed out", "error_category": "sandbox_setup"},
     "environment_setup"),
    ({"error": None, "error_category": None, "sandbox_startup_info": {"attempts": 3}},
     "environment_setup"),
    ({"error": "Agent deepagents install failed (rc=1)", "error_category": "install_failure"},
     "agent_setup"),
    # the agent's first request failed: the agent's phase, no token
    ({"error": "provider api error [rate_limit/transient] 429", "error_category": "api_error"},
     "agent"),
    # nothing says what failed, and the agent never ran
    ({"error": None, "error_category": None}, "unknown"),
    ({"error": "something else", "error_category": "other"}, "unknown"),
])
def test_a_failure_the_agent_never_reached_is_not_labelled_agent(change, stage):
    """2026-10-13 review: no_reward_cause labelled every failure other than a compose build / up
    or a verifier error `agent`, also when the agent never ran (a compose cp failure, a sandbox
    startup failure, a result with no error)."""
    result = {"error": None, "error_category": None, "n_tool_calls": 0,
              "agent_result": {"n_tool_calls": 0, "total_tokens": 0},
              "trajectory_summary": {"steps": 0}}
    result.update(change)
    cause = SB.no_reward_cause(result)
    assert cause["stage"] == stage and cause["agent_ran"] is False
    assert ("compose_command" in cause) == (stage == "sandbox_command")


def test_a_merge_keeps_every_shards_harness_errors_and_build_retries():
    """2026-10-13 review: skillsbench.merge_shards started from shard 1's result, so the merged
    sandbox_build_retries (and harness errors) were shard 1's only."""
    def shard(label, task, reward, retries, errors):
        return {"shard": label, "status": "completed_with_errors" if errors else "success",
                "leaderboard_comparable": False, "leaderboard_comparable_reason": None,
                "harness_errors": {"no_reward": len(errors)} if errors else {},
                "harness_error_tasks": {"without-skills": errors} if errors else {},
                "sandbox_build_retries": retries,
                "sample_traces": [{"sample_key": f"without-skills/{task}",
                                   "condition": "without-skills", "task": task,
                                   "reward": reward, "scoring_excluded": reward is None}]}
    a = shard("1/3", "a", 0.5, {}, {})
    b = shard("2/3", "b", None, {"without-skills": {"b": 2}},
              {"b": {"kind": "no_reward", "stage": "environment_build"}})
    c = shard("3/3", "c", 1.0, None, {})                    # an image that records no retries
    merged = SB.merge_shards([a, b, c])
    assert merged["sandbox_build_retries"] == {"without-skills": {"b": 2}}
    assert merged["harness_errors"] == {"no_reward": 1}
    assert merged["harness_error_tasks"] == {
        "without-skills": {"b": {"kind": "no_reward", "stage": "environment_build"}}}
    assert merged["status"] == "completed_with_errors" and merged["mean_reward"] == 0.75
    assert SB.merge_shards([c])["sandbox_build_retries"] is None
