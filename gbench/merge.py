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

"""Merge the eval results of a sharded run (`gbench --merge DIR [DIR ...] --results-dir OUT`).

Each node of a sharded run (`--shard I/N`) writes its own results directory. This module reads
them back, checks they are the shards of ONE run, and writes one merged result per
(suite, model, format):

* every eval result file (eval_<suite>_<model>_<format>[.shard-IofN].json) under each DIR is
  read, with the metadata.json of its run directory;
* the shards must agree: the same shard count N, --eval-limit, gbench commit and evals scaffold
  (the pinned fields and subject of the evals condition, with the shard itself set aside), and
  shards 1..N must each appear exactly once with status success or completed_with_errors;
  anything else is a MergeError naming the missing, repeated or failed shards;
* the merged result is the suite's own merge when its module defines
  `merge_shards(results: list[dict]) -> dict` (for corpus or macro metrics: a corpus score over the
  union of predictions, a macro average over per-item rows); otherwise the generic merge, which
  re-counts the traces: every shard's sample_traces kept (union_traces: an item is its
  `sample_key` at its position in the full ordered set, `global_idx`, with its attempt; rows with
  identical content stay apart, and an item found in two shards is a MergeError, since the
  shards then do not partition one ordered set; each trace is kept as its shard recorded it, so
  its sample_idx is its index in that shard), accuracy = correct / scored traces
  (base.is_scored), per-category accuracy and @k recomputed. The generic merge refuses a suite
  whose shards' own accuracy is not that count (a headline that is not a correct-over-scored
  fraction needs the suite's merge_shards);
* no field of the merged result is one shard's value standing for the union: counts are summed,
  lists of values joined, the Gemini models every shard used combined (which models answered,
  per role or group, with their call counts: generic or suite merge alike), and a field the
  shards disagree on that no rule combines is moved to `merge_unreconciled_fields`, which
  records each shard's value;
* the merged result records merged_shards, merged_from and the union's comparability: the
  shard reason is dropped when the union is the full set, every other shard reason is kept;
* every suite is checked and merged before anything is written (a refusal leaves nothing
  behind); `gbench --merge ... --dry-run` checks and merges, and writes nothing.

Suites whose results carry no per-item traces cannot be merged generically: they need
merge_shards, else the merge refuses them (MergeError) rather than average shard scores.

Unsharded runs that each ran different suites (a --parallel-suites pool writes one run directory
per suite; separate runs of disjoint suites) are joined instead of merged (union_runs): when no
result under the directories records a shard, every (suite, model, format) must appear exactly
once (a suite found twice is a MergeError naming both files), each result file is taken as it is
(copied, or linked in place for the pool's own directory), and the union gets one summary.json
and eval_summary.csv written exactly as a single run writes them (cli._summary_record and
cli._save_eval_summary_csv), plus a metadata.json naming every source. Nothing is re-scored. A
mix of sharded and unsharded results is refused.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_RESULT_RE = re.compile(r"^eval_(?P<rest>.+?)(?:\.shard-(?P<i>\d+)of(?P<n>\d+))?\.json$")


class MergeError(ValueError):
    """The directories are not the complete, consistent shards of one run."""


def _without_shard(obj: Any) -> Any:
    """`obj` with every key named shard / shard_index / shard_count removed, recursively."""
    if isinstance(obj, dict):
        return {k: _without_shard(v) for k, v in obj.items()
                if k not in ("shard", "shard_index", "shard_count")}
    if isinstance(obj, list):
        return [_without_shard(v) for v in obj]
    return obj


def _evals_condition(metadata: Dict[str, Any], models: Tuple[str, ...], fmt: str
                     ) -> Optional[Dict[str, Any]]:
    """The run's evals condition for this model (its name as the condition records it) and format;
    the only evals condition when there is just one."""
    evals = [c for c in (metadata.get("conditions") or []) if c.get("pillar") == "evals"]
    for cond in evals:
        if cond.get("model") in models and cond.get("format") in (fmt, None):
            return cond
    return evals[0] if len(evals) == 1 else None


def load_shard_results(dirs: List[Path]) -> List[Tuple[Dict[str, Any], Dict[str, Any], Path]]:
    """(result, metadata, path) for every eval result file under `dirs`. A file reached by more
    than one path (a --parallel-suites pool links each suite's result into the pool directory;
    two given directories that overlap) is taken once, by its real path (2026-10-11 review:
    `gbench --merge <pool-dir>` was refused, each suite appearing twice)."""
    found = []
    seen = set()
    for root in dirs:
        root = Path(root)
        if not root.is_dir():
            raise MergeError(f"{root} is not a directory")
        for path in sorted(root.rglob("eval_*.json")):
            if not _RESULT_RE.match(path.name) or ".bak" in path.name:
                continue
            real = path.resolve()
            if real in seen:
                continue
            seen.add(real)
            if path.is_symlink():
                path = real
            try:
                result = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:                                           # noqa: BLE001
                raise MergeError(f"{path} is not readable JSON ({e})") from e
            if not isinstance(result, dict) or result.get("benchmark_type", "eval") != "eval":
                continue
            meta_path = path.parent / "metadata.json"
            metadata = {}
            if meta_path.is_file():
                try:
                    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:                                            # noqa: BLE001
                    metadata = {}
            found.append((result, metadata, path))
    return found


def _group(found) -> Dict[Tuple[str, str, str], List[Tuple[Dict[str, Any], Dict[str, Any], Path]]]:
    groups: Dict[Tuple[str, str, str], list] = {}
    for result, metadata, path in found:
        key = (str(result.get("eval_name") or ""), str(result.get("model_short") or ""),
               str(result.get("format") or ""))
        groups.setdefault(key, []).append((result, metadata, path))
    return groups


def check_shards(key: Tuple[str, str, str], entries) -> int:
    """The shard count N after checking `entries` are shards 1..N of one run (MergeError)."""
    suite, model, fmt = key
    specs = []
    for result, metadata, path in entries:
        label = result.get("shard") or result.get("shard_requested")
        m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", str(label or ""))
        if not m:
            raise MergeError(f"{path}: the result records no shard (it is not from a sharded run, "
                             "or its suite could not shard: shard_applied=false)")
        if result.get("shard_applied") is False:
            raise MergeError(f"{path}: {suite} ran its full set on this node "
                             f"(shard_applied=false); its shards cannot be merged")
        specs.append((int(m.group(1)), int(m.group(2)), result, metadata, path))
    counts = {n for _i, n, *_ in specs}
    if len(counts) != 1:
        raise MergeError(f"{suite} ({model}, {fmt}): shards of different counts {sorted(counts)}")
    n = counts.pop()
    seen: Dict[int, Path] = {}
    for i, _n, result, _meta, path in specs:
        if i in seen:
            raise MergeError(f"{suite} ({model}, {fmt}): shard {i}/{n} appears twice "
                             f"({seen[i]} and {path})")
        seen[i] = path
        status = result.get("status")
        if status not in ("success", "completed_with_errors"):
            raise MergeError(f"{suite} ({model}, {fmt}): shard {i}/{n} has status {status!r} "
                             f"({path}); re-run it before merging")
    missing = [i for i in range(1, n + 1) if i not in seen]
    if missing:
        raise MergeError(f"{suite} ({model}, {fmt}): missing shard(s) "
                         f"{', '.join(f'{i}/{n}' for i in missing)}")
    limits = {json.dumps(r.get("eval_limit", (m or {}).get("eval_limit"))) for _i, _n, r, m, _p
              in specs}
    if len(limits) != 1:
        raise MergeError(f"{suite} ({model}, {fmt}): shards ran with different --eval-limit "
                         f"values {sorted(limits)}")
    commits = {json.dumps((m or {}).get("gbench"), sort_keys=True) for *_x, m, _p in specs}
    if len(commits) != 1:
        raise MergeError(f"{suite} ({model}, {fmt}): shards ran different gbench code "
                         f"({len(commits)} distinct commits in metadata.json)")
    scaffolds = set()
    for _i, _n, result, meta, _p in specs:
        cond = _evals_condition(meta or {}, (model, str(result.get("model_name") or "")), fmt)
        scaffolds.add(json.dumps(_without_shard(
            {"pinned": ((cond or {}).get("scaffold") or {}).get("pinned"),
             "subject": ((cond or {}).get("subject") or {}).get("subject_id")}), sort_keys=True))
    if len(scaffolds) != 1:
        raise MergeError(f"{suite} ({model}, {fmt}): shards ran under different evals scaffolds "
                         "(decoding, dataset or serving settings differ)")
    return n


def _scored_count(traces: List[Dict[str, Any]]) -> Tuple[int, int]:
    from .runners.eval_suites.base import is_scored
    scored = [t for t in traces if isinstance(t, dict) and is_scored(t)]
    return sum(1 for t in scored if t.get("is_correct")), len(scored)


#: Counters every shard counts over its own items: the merged value is their sum.
SUMMED_COUNTS = (
    "failed_requests", "empty_responses", "truncated_responses", "judge_failures",
    "judge_outages", "scoring_excluded", "over_context_prompts", "timed_out_requests",
    "looping_responses", "discarded_outputs", "non_convergent_responses",
    "genuinely_truncated", "wall_clock_limited", "replies_at_official_max_tokens",
    "replies_at_context_window", "server_aborted_requests", "letters_missing",
    "questions_without_a_letter", "short_top_logprobs", "shots_reduced",
    "replies_cut_off_by_budget", "replies_cut_off_in_reasoning", "answer_in_thinking",
    "answer_in_thinking_of",
    # a Gemini grader's replies still cut at upstream's cap plus the thinking allowance, and its
    # outages after such replies (gemini_budget; simple-evals' grader, wildbench's judge)
    "judge_replies_at_max_output_tokens", "judge_outages_after_max_output_tokens")
#: Maps combined name by name (_combine: counts summed, lists of names or ids joined).
COMBINED_MAPS = ("request_harness_errors", "response_health")
#: The server-parser policy's per-run record, rebuilt from the union's traces (each shard's
#: server_parser_samples holds its own local sample_idx values, which collide across shards).
SERVER_PARSER_FIELDS = ("server_parser_events", "server_parser_samples", "server_parser_note")
#: base's status cause for replies a non-official budget cut off, over 10% of the samples: the
#: rule is judged again over the union (a shard at 15% and one at 0% can be under 10% together).
_BUDGET_CUT_CAUSE = "replies_cut_off_by_non_official_budget "
#: Maps of counts that also hold a setting every shard ran with (the value must agree):
#: {field: its setting keys}.
COUNTS_WITH_SETTINGS = {"tool_rounds": ("per_turn_max_tokens",),
                        "official_cap_not_applied_to_reasoning": ("sent",)}
#: Lists of values: the merged value is their union (sorted, each value once).
UNION_LISTS = ("max_tokens_sent", "official_max_tokens_sent")
#: Numbers whose merged value is the largest shard's.
MAX_FIELDS = ("top_logprobs_returned_max",)
#: The fields that say which shard a result covers (merge_runs removes them).
SHARD_FIELDS = ("shard", "shard_index", "shard_count", "shard_applied", "shard_requested",
                "empty_shard", "full_set_items")
#: A field that names the Gemini models a result used (gemini_models_answered,
#: judge_models_used, self_judge_models, ...): the merged value is every shard's, combined.
_MODELS_FIELD = re.compile(r"(^|_)models(_|$)")


class _Unreconcilable(ValueError):
    """Values of one field that no rule combines (see _combine)."""


def _same(values: List[Any]) -> bool:
    first = json.dumps(values[0], sort_keys=True, default=str)
    return all(json.dumps(v, sort_keys=True, default=str) == first for v in values[1:])


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _combine(values: List[Any]) -> Any:
    """The shards' values of a count or provenance field, combined: numbers summed; maps
    combined key by key (a name only one shard has keeps that shard's value, numbers under one
    name are summed, other values must agree); lists joined (each value once, first-seen
    order). Raises _Unreconcilable for anything else."""
    if all(_is_number(v) for v in values):
        total = sum(values)
        return round(total, 6) if isinstance(total, float) else total
    if all(isinstance(v, dict) for v in values):
        out: Dict[str, Any] = {}
        for key in dict.fromkeys(k for v in values for k in v):
            present = [v[key] for v in values if key in v]
            if len(present) == 1:
                out[key] = present[0]
            elif all(_is_number(x) or isinstance(x, (dict, list)) for x in present):
                out[key] = _combine(present)
            elif _same(present):
                out[key] = present[0]
            else:
                raise _Unreconcilable(key)
        return out
    if all(isinstance(v, list) for v in values):
        seen: Dict[str, Any] = {}
        for v in values:
            for item in v:
                seen.setdefault(json.dumps(item, sort_keys=True, default=str), item)
        return list(seen.values())
    raise _Unreconcilable(type(values[0]).__name__)


def _combine_with_settings(values: List[Dict[str, Any]], settings: Tuple[str, ...]
                           ) -> Dict[str, Any]:
    """_combine over maps whose `settings` keys are not counts: those must agree."""
    fixed = {}
    for key in settings:
        present = [v[key] for v in values if key in v]
        if present and not _same(present):
            raise _Unreconcilable(key)
        if present:
            fixed[key] = present[0]
    out = _combine([{k: x for k, x in v.items() if k not in settings} for v in values])
    out.update(fixed)
    return out


def _sort_key(value: Any) -> Tuple[bool, Any]:
    return (False, value) if _is_number(value) else (True, json.dumps(value, sort_keys=True,
                                                                       default=str))


def _present(results: List[Dict[str, Any]], field: str) -> List[Any]:
    return [r[field] for r in results if field in r and r[field] is not None]


def _combine_models_fields(merged: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
    """Every shard's Gemini-model record in the merged result: each field naming models
    (_MODELS_FIELD: which models answered, how many calls each, per role or group) combined
    over the shards (_combine: counts summed, names joined), never one shard's alone; self_judge
    is true when any shard's was. A field whose values cannot be combined keeps no shard's value
    (merge_unreconciled_fields holds each shard's)."""
    fields = dict.fromkeys(k for r in results for k in r if _MODELS_FIELD.search(k))
    for field in fields:
        values = _present(results, field)
        if not values:
            continue
        try:
            merged[field] = values[0] if len(values) == 1 else _combine(values)
        except _Unreconcilable:
            if _same(values):           # one name every shard records alike (not a count)
                merged[field] = values[0]
            else:
                _unreconciled(merged, results, field)
    selfish = _present(results, "self_judge")
    if selfish:
        merged["self_judge"] = any(bool(v) for v in selfish)


def _shard_label(result: Dict[str, Any]) -> str:
    return str(result.get("shard") or result.get("shard_requested") or "?")


def _unreconciled(merged: Dict[str, Any], results: List[Dict[str, Any]], field: str) -> None:
    """`field` differs between the shards and no rule combines it: the merged result keeps no
    shard's value for it; merge_unreconciled_fields records each shard's."""
    merged.pop(field, None)
    merged.setdefault("merge_unreconciled_fields", {})[field] = {
        _shard_label(r): r.get(field) for r in results if field in r}


def other_reasons(result: Dict[str, Any], known: List[str]) -> List[str]:
    """The comparability reasons of one shard's result other than `known` (whole reason texts,
    which may themselves contain "; ") and its shard or empty-shard reason: for a suite's
    merge_shards that states its own reasons again over the union and keeps the others (one the
    eval runner stamped, such as the self-judge reason)."""
    from .runners.eval_suites.base import SHARD_REASON
    text = str(result.get("leaderboard_comparable_reason") or "")
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", str(result.get("shard") or ""))
    if m:
        text = text.replace(SHARD_REASON.format(label=f"{m.group(1)}/{m.group(2)}",
                                                index=int(m.group(1)), count=int(m.group(2))), "")
    for reason in sorted(set(known), key=len, reverse=True):
        if reason:
            text = text.replace(reason, "")
    shard_prefix = SHARD_REASON.split("{", 1)[0]
    return [x.strip() for x in text.split("; ") if x.strip()
            and not x.strip().startswith(("empty shard ", shard_prefix))]


def _global_idx(trace: Dict[str, Any]) -> Optional[int]:
    gi = trace.get("global_idx")
    return int(gi) if _is_number(gi) and int(gi) == gi else None


def _trace_identity(trace: Dict[str, Any]) -> Tuple[str, Optional[int], Any]:
    """The item a trace is: its sample_key at its position in the full ordered set (global_idx,
    None when the trace records none), with its attempt. Rows with identical content (the same
    sample_key) at different positions are different items."""
    return (str(trace.get("sample_key")), _global_idx(trace), trace.get("attempt_idx", 0))


def union_traces(suite: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Every trace of every shard (a shard's traces are the items it counted, all kept; items
    with identical content stay apart), in the order of the full set when the traces record
    their position (global_idx). An item (_trace_identity: the same sample_key, position and
    attempt) in two shards is a MergeError: the shards overlap, so they are not one partition
    of one ordered set (different data or code on the nodes)."""
    first_shard: Dict[Tuple[str, Optional[int], Any], str] = {}
    union: List[Dict[str, Any]] = []
    for r in results:
        label = _shard_label(r)
        for t in r.get("sample_traces") or []:
            ident = _trace_identity(t)
            other = first_shard.setdefault(ident, label)
            if other != label:
                where = f", global_idx {ident[1]}" if ident[1] is not None else ""
                raise MergeError(
                    f"{suite}: sample {ident[0]!r}{where} (attempt {ident[2]}) is in shard {other} "
                    f"and in shard {label}: the shards overlap, so they are not the partition of "
                    "one ordered set (the nodes ran different data or code); re-run the shards "
                    "from one gbench version and dataset revision")
            union.append(t)
    if union and all(_global_idx(t) is not None for t in union):
        union.sort(key=lambda t: (_global_idx(t), t.get("attempt_idx", 0) or 0))
    return union


def generic_merge(suite: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Re-count the union of the shards' traces (see the module docstring).

    Nothing in the merged result is one shard's value standing for the union: the headline,
    counts per category and @k are re-counted from the union's traces; SUMMED_COUNTS,
    SUMMED_COUNT_MAPS and duration_s are summed (duration_s_by_shard keeps each); UNION_LISTS
    joined; the Gemini-model fields combined (_combine_models_fields); finished_at is the latest.
    Any other field keeps its value when every shard (that is not an empty shard) has the same
    one, and otherwise moves to merge_unreconciled_fields, which records each shard's value."""
    from .runners.eval_suites.base import SHARD_REASON, is_scored, tally_batch_scores
    for r in results:
        traces = r.get("sample_traces")
        if not isinstance(traces, list):
            raise MergeError(f"{suite}: a shard carries no sample_traces; the suite needs its own "
                             "merge_shards to be merged")
        if traces and not all(isinstance(t, dict) and t.get("sample_key") for t in traces):
            raise MergeError(f"{suite}: its traces carry no sample_key; they cannot be "
                             "matched across shards")
        correct, scored = _scored_count(traces)
        reported = r.get("accuracy")
        mine = round(correct / scored * 100.0, 2) if scored else None
        if reported is not None and (mine is None or abs(float(reported) - mine) > 0.011):
            raise MergeError(f"{suite}: a shard's accuracy {reported} is not correct / scored of "
                             f"its traces ({mine}); its headline needs the suite's own "
                             "merge_shards")
    traces = union_traces(suite, results)
    correct, category_stats = tally_batch_scores(traces)
    _c, scored = _scored_count(traces)
    # An empty shard (no item on its node) sets no field's value for the union.
    live = [r for r in results if not r.get("empty_shard")] or list(results)
    merged = dict(live[0])
    handled = {"sample_traces", "total_questions", "correct_answers", "accuracy",
               "accuracy_withheld_reason", "category_accuracy", "attempts", "effective_n",
               "judge_outage_rate", "answer_in_thinking_rate", "status", "status_reason",
               "leaderboard_comparable", "leaderboard_comparable_reason", "duration_s",
               "duration_s_by_shard",
               "finished_at", "run_shape", "self_judge", "merge_unreconciled_fields",
               "server", "server_by_shard", "eval_limit_plan",
               *SUMMED_COUNTS, *COMBINED_MAPS, *COUNTS_WITH_SETTINGS, *UNION_LISTS,
               *MAX_FIELDS, *SHARD_FIELDS, *SERVER_PARSER_FIELDS}
    merged.pop("merge_unreconciled_fields", None)
    merged["sample_traces"] = traces
    merged["total_questions"] = len(traces)
    merged["correct_answers"] = correct
    merged["accuracy"] = round(correct / scored * 100.0, 2) if scored else None
    merged["category_accuracy"] = {
        cat: {"correct": st["correct"], "total": st["total"],
              "accuracy": round(st["correct"] / st["total"] * 100.0, 2) if st["total"] else 0.0}
        for cat, st in sorted(category_stats.items())}
    if any("effective_n" in r for r in results):
        merged["effective_n"] = scored
    for field in SUMMED_COUNTS:
        values = [r[field] for r in results if field in r and r[field] is not None]
        if values and all(_is_number(v) for v in values):
            merged[field] = _combine(values)
        elif values:
            _unreconciled(merged, results, field)
    for field in (*COMBINED_MAPS, *COUNTS_WITH_SETTINGS):
        values = [r[field] for r in results if isinstance(r.get(field), dict)]
        if not values:
            continue
        try:
            combined = _combine_with_settings(values, COUNTS_WITH_SETTINGS.get(field, ()))
            merged[field] = dict(sorted(combined.items()))
        except _Unreconcilable:
            _unreconciled(merged, results, field)
    for field in UNION_LISTS:
        values = [r[field] for r in results if isinstance(r.get(field), list)]
        if values:
            merged[field] = sorted({json.dumps(v, sort_keys=True, default=str): v
                                    for lst in values for v in lst}.values(), key=_sort_key)
    for field in MAX_FIELDS:
        values = [r[field] for r in results if _is_number(r.get(field))]
        if values:
            merged[field] = max(values)
    if "judge_outage_rate" in merged:
        outages = int(merged.get("judge_outages") or 0)
        merged["judge_outage_rate"] = round(outages / len(traces), 4) if traces else 0.0
    if "answer_in_thinking_rate" in merged:
        # base's ANSWER_IN_THINKING_POLICY record, over the union (the two counts were summed)
        from .runners.eval_suites.base import answer_in_thinking_fields
        merged.update(answer_in_thinking_fields(merged.get("answer_in_thinking") or 0,
                                                merged.get("answer_in_thinking_of") or 0))
    # the shards' --eval-limit plans: samples, per-setting counts and generations summed
    from .runners.eval_suites.limit_plan import merge_plans
    plan = merge_plans(results)
    if plan is not None:
        merged["eval_limit_plan"] = plan
    durations = [r.get("duration_s") for r in results]
    if any(_is_number(d) for d in durations):
        merged["duration_s"] = round(sum(d for d in durations if _is_number(d)), 2)
        merged["duration_s_by_shard"] = {_shard_label(r): r.get("duration_s") for r in results}
    stamps = [str(r["finished_at"]) for r in results if r.get("finished_at")]
    if stamps:
        merged["finished_at"] = max(stamps)
    shapes = [_without_shard(r["run_shape"]) for r in live if isinstance(r.get("run_shape"), dict)]
    if shapes and _same(shapes):
        merged["run_shape"] = shapes[0]
    elif shapes:
        _unreconciled(merged, live, "run_shape")
    attempts = {int((r.get("attempts") or {}).get("attempts_per_sample") or 1) for r in results}
    if max(attempts) > 1:
        k = max(attempts)
        by_sample: Dict[str, List[bool]] = {}
        for t in traces:
            if is_scored(t):
                by_sample.setdefault(str(t.get("sample_key")), []).append(bool(t.get("is_correct")))
        n_src = len(by_sample)
        # pass^k over the samples with every attempt scored, as base counts it
        complete = [v for v in by_sample.values() if len(v) >= k]
        merged["attempts"] = dict(live[0].get("attempts") or {}, samples=n_src,
                                  generations=len(traces),
                                  avg_at_k=merged["accuracy"],
                                  pass_at_k=round(sum(1 for v in by_sample.values() if any(v))
                                                  / n_src * 100.0, 2) if n_src else None,
                                  pass_hat_k=round(sum(1 for v in complete if all(v))
                                                   / len(complete) * 100.0, 2) if complete else None,
                                  pass_hat_k_samples=len(complete),
                                  samples_with_fewer_scored_attempts=n_src - len(complete))
        merged["attempts"].pop("fewer_scored_attempts_note", None)
        if n_src > len(complete):
            merged["attempts"]["fewer_scored_attempts_note"] = (
                f"{n_src - len(complete)} sample(s) have fewer than {k} scored attempts: counted "
                "in pass@k and avg@k over the attempts that were scored, left out of pass^k")
    else:
        values = [r["attempts"] for r in live if "attempts" in r]
        if values and not _same(values):
            _unreconciled(merged, live, "attempts")
    # The status over the union: each shard's causes, each named with its shard (their counts
    # are that shard's), except base's over-10% budget-cut rule, which is judged again over the
    # union's traces.
    causes: List[str] = []
    for r in results:
        for cause in str(r.get("status_reason") or "").split("; "):
            if cause and not cause.startswith(_BUDGET_CUT_CAUSE) and not cause.startswith(
                    "empty shard "):
                causes.append(f"shard {_shard_label(r)}: {cause}")
    official = set(merged.get("official_max_tokens_sent") or [])
    budget_cut = sum(1 for t in traces if t.get("health") == "truncated"
                     and t.get("max_tokens_sent") is not None
                     and t.get("max_tokens_sent") not in official)
    if budget_cut * 10 > len(traces):
        causes.append(f"{_BUDGET_CUT_CAUSE}{budget_cut} of {len(traces)} (over 10%: cut off by a "
                      "max_tokens that is not an official cap while still writing)")
    merged["status"] = "completed_with_errors" if causes else "success"
    merged["status_reason"] = "; ".join(dict.fromkeys(causes)) or None
    if merged["accuracy"] is None:
        merged["accuracy_withheld_reason"] = "no trace of any shard was scored"
    else:
        merged.pop("accuracy_withheld_reason", None)
    shard_prefix = SHARD_REASON.split("{", 1)[0]
    kept: List[str] = []
    for r in results:
        for reason in str(r.get("leaderboard_comparable_reason") or "").split("; "):
            if reason and not reason.startswith(shard_prefix) and not reason.startswith(
                    "empty shard ") and reason not in kept:
                kept.append(reason)
    # metrics.record_budget_cutoffs' reason names a count: stated once, over the union
    budget_reason = re.compile(r"^\d+ repl\(ies\) cut off by this run's token budget")
    if any(budget_reason.match(x) for x in kept):
        kept = [x for x in kept if not budget_reason.match(x)]
        from .runners.eval_suites.metrics import record_budget_cutoffs
        again = record_budget_cutoffs({"sample_traces": traces,
                                       "official_max_tokens_sent": sorted(official)})
        if again:
            kept.append(again)
    merged["leaderboard_comparable"] = not kept
    merged["leaderboard_comparable_reason"] = "; ".join(kept) or None
    _rebuild_server_parser(merged, results, traces)
    servers = {_shard_label(r): r["server"] for r in results if isinstance(r.get("server"), dict)}
    merged.pop("server", None)
    if servers:
        # each node's own server (its port, GPUs, bind address)
        merged["server_by_shard"] = servers
    _combine_models_fields(merged, results)
    models_fields = {k for r in results for k in r if _MODELS_FIELD.search(k)}
    # Every other field: one value for the union only when the (non-empty) shards agree on it.
    for field in dict.fromkeys(k for r in live for k in r):
        if field in handled or field in models_fields:
            continue
        values = [r.get(field, _ABSENT) for r in live]
        if not _same(values):
            _unreconciled(merged, live, field)
    return merged


#: A field a shard's result does not have (compared as a value of its own).
_ABSENT = "<absent>"


def _rebuild_server_parser(merged: Dict[str, Any], results: List[Dict[str, Any]],
                           traces: List[Dict[str, Any]]) -> None:
    """The server-parser policy's record for the union, from its traces (each trace's
    server_parser event), keyed by the trace's global_idx (else "<shard>:<sample_idx>"), so
    server_parser_events[k] == len(server_parser_samples[k]); nothing when no trace has one."""
    for field in SERVER_PARSER_FIELDS:
        merged.pop(field, None)
    shard_of = {id(t): _shard_label(r) for r in results for t in (r.get("sample_traces") or [])}
    samples: Dict[str, List[Any]] = {}
    sources: Dict[str, int] = {}
    for t in traces:
        sp = t.get("server_parser") if isinstance(t.get("server_parser"), dict) else {}
        event = sp.get("event")
        if not event:
            continue
        gidx = _global_idx(t)
        samples.setdefault(event, []).append(
            gidx if gidx is not None else f"{shard_of.get(id(t), '?')}:{t.get('sample_idx')}")
        if sp.get("recovered_from"):
            sources[sp["recovered_from"]] = sources.get(sp["recovered_from"], 0) + 1
    if not samples:
        return
    from .runners.eval_suites.base import server_parser_note
    merged["server_parser_events"] = {k: len(v) for k, v in sorted(samples.items())}
    merged["server_parser_samples"] = {k: sorted(v, key=_sort_key)
                                       for k, v in sorted(samples.items())}
    merged["server_parser_note"] = server_parser_note(merged["server_parser_events"], sources)


def merge_suite(suite: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The suite's own merge_shards when its module defines one, else generic_merge. Either way
    the merged result records every shard's Gemini models (_combine_models_fields) and the
    shards' summed eval_limit_plan (limit_plan.merge_plans), not the first shard's alone."""
    hook = None
    try:
        module = importlib.import_module(f"gbench.runners.eval_suites.{suite}")
        hook = getattr(module, "merge_shards", None)
    except Exception:                                                        # noqa: BLE001
        hook = None
    if callable(hook):
        try:
            merged = hook(results)
        except MergeError:
            raise
        except Exception as e:                                             # noqa: BLE001
            # a suite's own merge (re-scoring the union, fetching its annotations) that fails is
            # a refused merge with nothing written, never a traceback
            raise MergeError(f"{suite}: merging failed while re-scoring the union "
                             f"({type(e).__name__}: {e})") from e
    else:
        merged = generic_merge(suite, results)
    # the merged run's --eval-limit plan is the shards' plans summed, whichever path merged it
    # (a suite's own merge_shards often starts from shard 1's result and would keep its plan)
    from .runners.eval_suites.limit_plan import merge_plans
    plan = merge_plans(results)
    if plan is not None and isinstance(merged, dict):
        merged["eval_limit_plan"] = plan
    _combine_models_fields(merged, results)
    return merged


def merge_runs(dirs: List[Path], out_dir: Path, *, dry_run: bool = False,
               unique: bool = False) -> List[Path]:
    """Merge the sharded eval results under `dirs` into `out_dir` (see the module docstring).

    Every suite is checked and merged before anything is written, so a refusal (MergeError)
    leaves nothing behind. `dry_run`: nothing is created or written; the paths a real merge
    would write are returned. `unique`: `out_dir` is claimed atomically as a new directory
    (<name>, else <name>-2, ...: log_manager.create_unique_dir), so two merges started in the
    same second never share one; otherwise it is created when missing. Returns the files
    written (or, with dry_run, the files that would be)."""
    from .utils.log_manager import create_unique_dir, write_redacted_json
    found = load_shard_results([Path(d) for d in dirs])
    if not found:
        raise MergeError(f"no eval result files under {', '.join(str(d) for d in dirs)}")
    sharded = [p for r, _m, p in found if _shard_label(r) is not None]
    if not sharded:
        return union_runs(dirs, out_dir, dry_run=dry_run, unique=unique, found=found)
    if len(sharded) != len(found):
        unsharded = [str(p) for r, _m, p in found if _shard_label(r) is None]
        raise MergeError("the directories mix sharded and unsharded results (unsharded: "
                         + ", ".join(unsharded[:5]) + ("..." if len(unsharded) > 5 else "")
                         + "); merge the shards of one run, or join unsharded runs, not both")
    groups = _group(found)
    out_dir = Path(out_dir)
    planned: List[Tuple[str, Dict[str, Any]]] = []
    merged_meta: Dict[str, Any] = {"merged_from": [str(Path(d)) for d in dirs], "suites": {}}
    commits = {json.dumps(m.get("gbench"), sort_keys=True) for _r, m, _p in found
               if isinstance(m, dict) and m.get("gbench") is not None}
    if len(commits) == 1:
        merged_meta["gbench"] = json.loads(commits.pop())
    for key, entries in sorted(groups.items()):
        n = check_shards(key, entries)
        ordered = sorted(entries, key=lambda e: int(str(e[0].get("shard")).split("/")[0]))
        merged = merge_suite(key[0], [e[0] for e in ordered])
        for k in SHARD_FIELDS:
            merged.pop(k, None)
        merged["merged_shards"] = n
        merged["merged_from"] = [str(e[2]) for e in ordered]
        suite, model, fmt = key
        planned.append((f"eval_{suite}_{model}_{fmt}.json", merged))
        merged_meta["suites"][suite] = {"shards": n, "files": merged["merged_from"]}
    if dry_run:
        for name, _m in planned:
            logger.info("dry run: would write %s", out_dir / name)
        return [out_dir / name for name, _m in planned]
    if unique:
        out_dir = create_unique_dir(out_dir.parent, out_dir.name)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for name, merged in planned:
        path = out_dir / name
        # The one results writer: the text is redacted after serialization.
        write_redacted_json(path, merged, default=str)
        written.append(path)
        logger.info("merged %d shards of %s into %s", merged["merged_shards"],
                    merged.get("eval_name") or name, path)
    write_redacted_json(out_dir / "metadata.json", merged_meta)
    return written


def _shard_label(result: Dict[str, Any]) -> Optional[str]:
    """The shard a result records (shard, else shard_requested), or None for an unsharded one."""
    label = result.get("shard") or result.get("shard_requested")
    return str(label) if label else None


def union_runs(dirs: List[Path], out_dir: Path, *, dry_run: bool = False, unique: bool = False,
               link: bool = False, found=None,
               extra_failures: Optional[List[Dict[str, Any]]] = None,
               extra_metadata: Optional[Dict[str, Any]] = None) -> List[Path]:
    """Join the unsharded eval results under `dirs` into `out_dir` (module docstring): each
    (suite, model, format) exactly once, its file copied (`link`: a relative symlink, for a
    directory that contains the sources) unchanged; then summary.json and eval_summary.csv over
    all of them, as one run writes them, and metadata.json. `extra_failures` are failed records
    of suites that wrote no result file (a pool child that crashed): they appear in the summary
    and the CSV, never as a file. Returns the result files written (with dry_run, those that
    would be)."""
    from .utils.log_manager import create_unique_dir, write_redacted_json
    if found is None:
        found = load_shard_results([Path(d) for d in dirs])
    if not found and not extra_failures:
        raise MergeError(f"no eval result files under {', '.join(str(d) for d in dirs)}")
    sharded = [str(p) for r, _m, p in found if _shard_label(r) is not None]
    if sharded:
        raise MergeError("union_runs joins unsharded runs; these results are shards: "
                         + ", ".join(sharded[:5]))
    out_dir = Path(out_dir)
    planned: List[Tuple[str, Dict[str, Any], Path]] = []
    seen: Dict[Tuple[str, str, str], Path] = {}
    for key, entries in sorted(_group(found).items()):
        if len(entries) > 1:
            raise MergeError(f"{key[0]} ({key[1]}, {key[2]}) appears in more than one run: "
                             + ", ".join(str(e[2]) for e in entries)
                             + "; keep one result per suite before joining")
        result, _meta, path = entries[0]
        seen[key] = path
        planned.append((path.name, result, path))
    if dry_run:
        return [out_dir / name for name, _r, _p in planned]
    if unique:
        out_dir = create_unique_dir(out_dir.parent, out_dir.name)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    import shutil
    for name, _result, src in planned:
        dest = out_dir / name
        if dest.resolve() == src.resolve():
            written.append(dest)
            continue
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        if link:
            dest.symlink_to(os.path.relpath(src, out_dir))
        else:
            shutil.copyfile(src, dest)
        written.append(dest)
    results = [r for _n, r, _p in planned] + list(extra_failures or [])
    write_union_summary(out_dir, results)
    commits = {json.dumps(m.get("gbench"), sort_keys=True) for _r, m, _p in found
               if isinstance(m, dict) and m.get("gbench") is not None}
    meta: Dict[str, Any] = {"joined_from": [str(Path(d)) for d in dirs],
                            "suites": {k[0]: str(p) for k, p in seen.items()}}
    if len(commits) == 1:
        meta["gbench"] = json.loads(commits.pop())
    elif commits:
        meta["gbench_commits"] = sorted(json.loads(c).get("commit") if c != "null" else None
                                        for c in commits)
    if extra_failures:
        meta["suites_without_result"] = sorted(str(r.get("eval_name")) for r in extra_failures)
    meta.update(extra_metadata or {})
    write_redacted_json(out_dir / "metadata.json", meta)
    return written


def write_union_summary(out_dir: Path, results: List[Dict[str, Any]]) -> None:
    """summary.json and eval_summary.csv for `results` (eval results of several runs), in the
    format print_results_summary writes for one run."""
    from .cli import _save_eval_summary_csv, _summary_record
    from .runners.eval_suites import CUSTOM_PILLARS
    from .runners.evals import BUILTIN_PILLARS
    from .utils.log_manager import write_redacted_json
    ok = [r for r in results if not r.get("failed", False)]
    failed = [r for r in results if r.get("failed", False)]
    payload = {
        "timestamp": Path(out_dir).name,
        "results_dir": str(Path(out_dir).absolute()),
        "total_configurations": len(results),
        "successful_configurations": len(ok),
        "failed_configurations": len(failed),
        "models": [_summary_record(r) for r in ok],
        "failures": [_summary_record(r) for r in failed],
    }
    write_redacted_json(Path(out_dir) / "summary.json", payload)
    _save_eval_summary_csv(Path(out_dir), [r for r in ok if r.get("benchmark_type") == "eval"],
                           [r for r in failed if r.get("benchmark_type") == "eval"],
                           BUILTIN_PILLARS, CUSTOM_PILLARS)
