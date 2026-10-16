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

"""The output budget of a Gemini model that fills a role written for a non-thinking model.

Upstream judges, graders and user simulators (gpt-4o, gpt-4.1, gpt-4-turbo, ...) do not think:
their max_tokens is all room for the visible reply. The gbench Gemini cascade thinks before it
answers and counts that thinking against max_output_tokens, so the upstream cap sent unchanged
leaves the reply less room than upstream had, and the longest, most complex inputs are the
likeliest to be cut. A cut verdict is not random noise: it biases the score.

The rule (applied first in wildbench, 2026-10-09): send the upstream visible budget plus
GEMINI_THINKING_ALLOWANCE, so the reply keeps upstream's room, and count every reply that still
stops at that limit (parsed as it arrived, as upstream parses a reply cut at its max_tokens) and
every call whose answers all stopped there with no text (a judge outage, a harness error).

Code that cannot import gbench (the wildclawbench in-container judge proxy) carries a copy of
GEMINI_THINKING_ALLOWANCE; tests/test_gemini_thinking_allowance.py checks it equals this one.

A sharded run (`gbench --merge`) sums the counts (gbench.merge.SUMMED_COUNTS holds the simple-evals
grader's judge_replies_at_max_output_tokens and judge_outages_after_max_output_tokens) and states
the cut-off reason once, over the union (merge_with_cutoff_reason), never one shard's count.
"""

from typing import Any, Callable, Dict, List

#: Room on top of the upstream visible budget for the Gemini model's thinking (the 2026-10-08
#: smoke run's wildbench judge stopped mid-verdict 55 times at its upstream 1024 alone).
GEMINI_THINKING_ALLOWANCE = 8192


def gemini_output_limit(upstream_max_tokens: int) -> int:
    """What a Gemini role sends as max_output_tokens: the upstream visible budget plus the
    thinking allowance."""
    return int(upstream_max_tokens) + GEMINI_THINKING_ALLOWANCE


def output_limit_record(upstream_max_tokens: int, upstream_source: str,
                        upstream_models: str) -> Dict[str, Any]:
    """The max_output_tokens fields a result records for a Gemini role.

    `upstream_source` names where the upstream cap is set (e.g. "grading_sampler's
    max_tokens"), `upstream_models` the non-thinking models upstream ran (e.g. "gpt-4.1")."""
    limit = gemini_output_limit(upstream_max_tokens)
    return {
        "max_output_tokens": limit,
        "upstream_max_tokens": int(upstream_max_tokens),
        "thinking_allowance": GEMINI_THINKING_ALLOWANCE,
        "max_output_tokens_note": (
            f"{upstream_source}={int(upstream_max_tokens)} for the reply plus "
            f"{GEMINI_THINKING_ALLOWANCE} for the Gemini model's thinking, which counts against "
            f"max_output_tokens ({upstream_models} do(es) not think); a reply that still stops "
            f"at {limit} is used as it arrived and counted, and a call whose every answer "
            "stopped there with no text is a judge outage (a harness error), counted too"),
    }


def cutoff_reason(role: str, at_cap: int, outages_at_cap: int, upstream_max_tokens: int) -> str:
    """The leaderboard_comparable reason for replies of a Gemini role still cut at the limit, or
    "" when there are none."""
    if not at_cap and not outages_at_cap:
        return ""
    limit = gemini_output_limit(upstream_max_tokens)
    return (f"{at_cap} {role} repl(ies) stopped at max_output_tokens {limit} (upstream's "
            f"{int(upstream_max_tokens)} plus {GEMINI_THINKING_ALLOWANCE} for the Gemini "
            f"model's thinking) and {outages_at_cap} {role} outage(s) where every answer "
            "stopped there with no text")


def merge_with_cutoff_reason(suite: str, results: List[Dict[str, Any]],
                             reason_of: Callable[[Dict[str, Any]], str],
                             recount: Callable[[Dict[str, Any]], None] = lambda merged: None
                             ) -> Dict[str, Any]:
    """gbench.merge.generic_merge of a sharded run's results, with the cut-off reason stated once
    over the union: every shard's own `reason_of(result)` entry is dropped from the merged
    leaderboard_comparable_reason, `recount(merged)` (optional) rebuilds counts the generic merge
    does not sum, and `reason_of(merged)` (over the summed counts) is added back. A cut-off
    reason is one "; "-free entry, so it is matched whole."""
    from ...merge import generic_merge
    shard_reasons = {reason_of(r) for r in results} - {""}
    merged = generic_merge(suite, results)
    recount(merged)
    kept = [x for x in str(merged.get("leaderboard_comparable_reason") or "").split("; ")
            if x and x not in shard_reasons]
    again = reason_of(merged)
    if again:
        kept.append(again)
    merged["leaderboard_comparable"] = not kept
    merged["leaderboard_comparable_reason"] = "; ".join(kept) or None
    return merged
