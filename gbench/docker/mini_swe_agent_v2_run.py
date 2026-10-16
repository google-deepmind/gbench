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

"""Run ONE SWE-bench instance with mini-swe-agent 2.x (inside gbench-mini-swe-agent-v2).

gbench's swe_agentic starts one container of this image per instance and hands it a JSON spec.
This script follows ``minisweagent.run.benchmarks.swebench.process_instance`` step for step,
with the bundled ``benchmarks/swebench.yaml`` config unmodified apart from the overrides in the
spec (the served model endpoint, gbench's sampling knobs, container keep-alive and pull timeout,
per-container thread cap). It is standalone: it imports nothing from gbench, only the
mini-swe-agent release of the interpreter that runs it.

Operational differences from upstream, none of which changes the protocol:
  * only when the spec's overrides carry a per-turn max_tokens (gbench's SWE suites send none,
    as the bundled config sets none): a turn the server refuses only because of that cap is
    resent without it (the request upstream sends) and counted in ``context_retry_turns``;
  * the task container is removed synchronously (upstream's cleanup is a background process,
    which would die with this container);
  * a rollout that could not run for an infrastructure reason is marked ``infra_error`` so
    gbench keeps it out of the model-scoreable set.

Usage:
  python mini_swe_agent_v2_run.py --describe            # version + bundled config + sha256
  python mini_swe_agent_v2_run.py --spec /path/spec.json

Outputs (under ``spec["output_dir"]/<instance_id>/``): ``<instance_id>.traj.json`` (upstream's
trajectory format, via ``agent.save``) and ``result.json`` (what gbench reads).
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

import yaml  # noqa: E402

import minisweagent  # noqa: E402
from minisweagent.agents.default import DefaultAgent  # noqa: E402
from minisweagent.config import builtin_config_dir  # noqa: E402
from minisweagent.models.litellm_model import LitellmModel  # noqa: E402
from minisweagent.run.benchmarks.swebench import get_sb_environment  # noqa: E402
from minisweagent.utils.serialize import recursive_merge  # noqa: E402

#: The config every SWE-bench submission of this release runs (``--config`` default of
#: ``mini-extra swebench``).
CONFIG_NAME = "benchmarks/swebench.yaml"
#: The one mid-rollout exception that is the model's own outcome: the conversation outgrew the
#: served context window (upstream records it as the exit status and moves on).
MODEL_ATTRIBUTABLE = ("ContextWindowExceededError",)


def load_config():
    """(bundled swebench config, sha256 of the file)."""
    path = builtin_config_dir / CONFIG_NAME
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load(text), hashlib.sha256(text.encode("utf-8")).hexdigest()


class EndpointModel(LitellmModel):
    """LitellmModel that resends a turn refused only because of gbench's per-turn cap.

    vLLM refuses a request whose prompt plus max_tokens exceeds its context window, even when
    a shorter completion would fit. Upstream sets no max_tokens, so it never sees that refusal.
    When the operator's cap causes it, the turn is resent the way upstream sends it (no
    max_tokens: the server grants what the window has left). If the prompt alone overflows,
    the resend fails the same way and ends the rollout as the model's outcome."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.context_retry_turns = 0

    def query(self, messages, **kwargs):
        budget = (self.config.model_kwargs or {}).get("max_tokens")
        try:
            return super().query(messages, **kwargs)
        except Exception as e:
            capped = budget and kwargs.get("max_tokens", budget) is not None
            if not capped or not any(c.__name__ in MODEL_ATTRIBUTABLE for c in type(e).__mro__):
                raise
        reply = super().query(messages, **dict(kwargs, max_tokens=None))
        self.context_retry_turns += 1
        return reply


def condensed_trajectory(messages):
    """(role/content messages with reasoning, actions and finish reason, truncated turns)."""
    out, truncated = [], 0
    for m in messages or []:
        entry = {"role": m.get("role"), "content": m.get("content")}
        extra = m.get("extra") or {}
        choices = ((extra.get("response") or {}).get("choices") or [])
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            reasoning = msg.get("reasoning_content") or msg.get("reasoning")
            if reasoning:
                entry["reasoning"] = reasoning
            finish = choices[0].get("finish_reason")
            if finish:
                entry["finish_reason"] = finish
                truncated += finish == "length"
        if extra.get("actions"):
            entry["actions"] = [a.get("command") for a in extra["actions"] if isinstance(a, dict)]
        out.append(entry)
    return out, truncated


def _remove_container(env):
    cid = getattr(env, "container_id", None)
    if not cid:
        return
    try:
        subprocess.run([env.config.executable, "rm", "-f", cid], capture_output=True, timeout=180)
    except Exception as e:  # the gbench side removes it by id if this fails
        print(f"container cleanup failed for {cid}: {e}", file=sys.stderr)


def run_instance(spec):
    instance = dict(spec["instance"])
    iid = instance["instance_id"]
    inst_dir = Path(spec["output_dir"]) / iid
    inst_dir.mkdir(parents=True, exist_ok=True)
    base, sha = load_config()
    config = recursive_merge(base, spec.get("overrides") or {})
    model_cfg = copy.deepcopy(config.get("model", {}))
    model_class = model_cfg.pop("model_class", "") or "litellm"
    if model_class != "litellm":
        raise SystemExit(f"{CONFIG_NAME} names model_class={model_class!r}; this runner drives "
                         "the litellm tool-calling model only")
    started = time.time()
    model = EndpointModel(**model_cfg)
    task = instance["problem_statement"]
    agent = env = None
    phase = "environment"
    exit_status, submission, extra, infra = None, "", {}, False
    try:
        env = get_sb_environment(config, instance)
        (inst_dir / "container_id").write_text(env.container_id or "", encoding="utf-8")
        phase = "agent"
        agent = DefaultAgent(model, env, **config.get("agent", {}))
        info = agent.run(task)
        exit_status = info.get("exit_status")
        submission = info.get("submission") or ""
    except Exception as e:
        exit_status, submission = type(e).__name__, ""
        extra = {"traceback": traceback.format_exc(), "exception_str": str(e)}
        infra = phase == "environment" or exit_status not in MODEL_ATTRIBUTABLE
    finally:
        traj_path = inst_dir / f"{iid}.traj.json"
        if agent is not None:
            agent.save(traj_path, {"info": {"exit_status": exit_status, "submission": submission,
                                            **extra},
                                   "instance_id": iid})
        if env is not None:
            _remove_container(env)
            (inst_dir / "container_id").unlink(missing_ok=True)
    messages, truncated = condensed_trajectory(getattr(agent, "messages", None))
    result = {
        "instance_id": iid,
        "exit_status": exit_status,
        "submission": submission,
        "error": extra.get("exception_str"),
        "traceback": extra.get("traceback"),
        "failed_phase": phase if extra else None,
        "infra_error": infra,
        "n_calls": int(getattr(agent, "n_calls", 0) or 0),
        "context_retry_turns": model.context_retry_turns,
        "truncated_turns": truncated,
        "trajectory": messages,
        "trajectory_file": str(traj_path) if agent is not None else None,
        "mini_version": minisweagent.__version__,
        "config_name": CONFIG_NAME,
        "config_sha256": sha,
        "duration_s": round(time.time() - started, 2),
    }
    tmp = inst_dir / "result.json.tmp"
    tmp.write_text(json.dumps(result, default=str), encoding="utf-8")
    os.replace(tmp, inst_dir / "result.json")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--describe", action="store_true",
                       help="print the release, its bundled config and the config sha256")
    group.add_argument("--spec", help="JSON spec of the instance to run")
    args = parser.parse_args(argv)
    if args.describe:
        config, sha = load_config()
        print(json.dumps({"version": minisweagent.__version__, "config_name": CONFIG_NAME,
                          "config_sha256": sha, "config": config}))
        return 0
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    want = spec.get("expected_version")
    if want and want != minisweagent.__version__:
        raise SystemExit(f"this interpreter runs mini-swe-agent {minisweagent.__version__}, "
                         f"the spec asks for {want}")
    run_instance(spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
