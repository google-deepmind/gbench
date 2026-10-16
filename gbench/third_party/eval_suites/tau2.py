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

"""Native Tau-bench v2 Multi-Domain Agentic Tool Calling evaluation suite.

Scored path: the tau2-bench simulator at the pinned revision, run in the local gbench-tau2 image
(docker/tau2.Dockerfile), runs every task of the retail, airline and telecom `base` splits 4
times; the headline is pass^1 averaged over the three domains with equal weight, the tau2-bench
leaderboard track's Overall, with pass^2..pass^4 and the per-domain numbers alongside. A
simulation that ends in an infrastructure error, whatever the cause (tau2's run_with_retry spent
its retries), is never scored: left out of pass^k, the mean reward and the evaluable counts,
status completed_with_errors, named in leaderboard_comparable_reason. One from a harness failure
(an outage of the Gemini user simulator or judge, of the model endpoint, ...) is a harness error,
never the agent's outcome, counted per domain in harness_errors; one raised on the agent's own
turn (a reply tau2 rejects, a request over the endpoint's context window) is not, and is counted
per domain in agent_turn_errors. See tau_common and docs/evals/tau2.md.

Sampling: the agent (the model under test) follows gbench's rule for every suite: temperature
0.0 without `--thinking` (temperature_source "default"; also tau2-bench's upstream
DEFAULT_LLM_TEMPERATURE_AGENT), 1.0 with it (temperature_source "thinking", named in
leaderboard_comparable_reason). Like upstream, only
the temperature is sent, so top_p / top_k stay at the server's defaults unless
GBENCH_TAU2_TOP_P / GBENCH_TAU2_TOP_K are set. Override the temperature for a whole run with
`--temperature`, or for this suite alone with `GBENCH_TAU2_TEMPERATURE`,
which takes precedence over both. The user simulator and the LLM-judge grading are pinned at
0.0 and are not affected by either; both are gbench's Gemini cascade (GBENCH_JUDGE_MODELS),
reached through gbench's Gemini relay, and the result records which model answered.
"""

import json
import logging
from typing import Any, Dict, List, Tuple

from .tau_common import CORE_DOMAINS, merge_tau_shards, run_tau_suite

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/tau2.md"

#: gbench --merge's hook for a sharded tau2 run: pass^k recomputed per domain over the union of
#: the shards' per-task outcomes (tau_common.merge_tau_shards).
merge_shards = merge_tau_shards


# A single-turn strict-action check kept as an offline diagnostic (tests only): it is never a
# scorer. The scored path is the tau2-bench simulator (run_tau2 -> tau_common.run_tau_suite).
def _extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """Extract all top-level JSON objects from free text (for tool-call detection)."""
    objs: List[Dict[str, Any]] = []
    dec = json.JSONDecoder()
    i, n = 0, len(text or "")
    while i < n:
        if text[i] == "{":
            try:
                obj, end = dec.raw_decode(text, i)
                if isinstance(obj, dict):
                    objs.append(obj)
                i = end
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return objs


def _required_calls(gold: Any) -> List[Tuple[str, Dict[str, Any]]]:
    """Normalize gold into a list of required (tool_name, args) calls."""
    calls: List[Tuple[str, Dict[str, Any]]] = []
    if isinstance(gold, dict) and gold.get("actions"):
        for a in gold["actions"]:
            if isinstance(a, dict):
                name = a.get("name") or a.get("action") or a.get("tool")
                args = a.get("arguments") or a.get("kwargs") or a.get("args") or {}
                if name:
                    calls.append((str(name), args if isinstance(args, dict) else {}))
    elif isinstance(gold, dict) and gold.get("tool"):
        calls.append((str(gold["tool"]), gold.get("args") or {}))
    return calls


def _call_present(req_name: str, req_args: Dict[str, Any], objs: List[Dict[str, Any]]) -> bool:
    for o in objs:
        name = o.get("name") or o.get("tool")
        args = o.get("arguments") or o.get("args") or {}
        if str(name) == req_name and isinstance(args, dict):
            if all(str(args.get(k)) == str(v) for k, v in req_args.items()):
                return True
    return False


def _eval_tau2(response_text: str, gold_scenario: Dict[str, Any]) -> bool:
    """Strict required-action match: every gold tool call must appear (name + args).

    NOTE: canonical tau2-bench is a multi-turn *environment* benchmark (final
    DB-state check + nl_assertion LLM judge). This single-turn harness scores only
    the required tool actions; assertion-only tasks (no required actions) cannot be
    verified deterministically here and score False. Full fidelity needs the
    sierra-research/tau2-bench environment simulator.
    """
    if not response_text:
        return False
    required = _required_calls(gold_scenario)
    if not required:
        return False
    objs = _extract_json_objects(response_text)
    return all(_call_present(n, a, objs) for n, a in required)


def run_tau2(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run Tau-bench v2 Multi-Domain Agentic Tool Calling evaluation suite.

    tau2-bench is a multi-turn *environment* benchmark, so it is scored only by the real
    simulator: this runs tau2-bench at the pinned revision in the local gbench-tau2 image
    (tau_common.run_tau_suite) over retail, airline and telecom, the tau2-bench track's three
    domains. A missing Docker or image hard-errors (infra_required) with the build command.
    `--shard I/N` runs shard I of each domain's tasks; `--eval-limit N` is N tasks for the
    whole run, dealt over the domains (tau_common.domain_limit_shares; a domain whose share is 0
    is not run), each domain's share a seeded draw passed to tau2 as task_ids. The single-turn strict-action check `_eval_tau2` is an offline
    diagnostic for tests only; it is never a scorer.
    """
    return run_tau_suite("tau2", list(CORE_DOMAINS), model_name, base_url, concurrency,
                         kwargs.get("limit"), DOCS_URL, enable_thinking=enable_thinking)
