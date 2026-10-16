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

"""The planned workload of a suite run, logged once at its start and recorded as `eval_limit_plan`.

`--eval-limit N` is N samples (distinct dataset items) for the whole suite; the attempts a protocol
makes per item multiply the generations, not the samples. A suite that spreads N over settings
(tracks, splits, prompt modes, context bands) names them, so a limited run that would run more
than asked shows it on its first line:

    [charxiv] --eval-limit 5: 5 sample(s) over 2 track(s) (descriptive 3, reasoning 2) x 1 attempt = 5 generations
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

__all__ = ["eval_limit_plan", "base_attempts", "per_category", "planned_for", "merge_plans",
           "record", "clear_plan"]

#: The plan each suite logged in this process (eval_name -> plan), so base.run_eval_suite can tell
#: a suite that logged its own plan from one it should plan for.
_PLANNED: Dict[str, Dict[str, Any]] = {}


def base_attempts(batch_scored: bool = False, supports_attempts: bool = False) -> int:
    """The attempts per sample base.run_eval_suite makes from the run's --attempt-count (a
    batch-scored suite that does not declare supports_attempts runs 1)."""
    from .base import get_run_knob
    attempts = max(1, int(get_run_knob("attempt_count", 1) or 1))
    return 1 if batch_scored and not supports_attempts else attempts


def per_category(samples: Any, key: str = "category") -> Dict[str, int]:
    """How many of `samples` ((messages, gold, meta) tuples) each meta `key` value holds, in
    first-seen order."""
    counts: Dict[str, int] = {}
    for sample in samples:
        meta = sample[2] if len(sample) > 2 and isinstance(sample[2], dict) else {}
        value = str(meta.get(key))
        counts[value] = counts.get(value, 0) + 1
    return counts


def eval_limit_plan(eval_name: str, limit: Optional[int], samples: int,
                    per_setting: Optional[Mapping[str, int]] = None, *,
                    setting_noun: str = "setting", attempts: int = 1,
                    unit: str = "generations", register: bool = True) -> Dict[str, Any]:
    """Log the suite's planned workload (one line) and return the `eval_limit_plan` record.

    `samples`: the distinct items this run (this shard) holds; `per_setting`: how many of them each
    setting holds, for a suite that has settings (a setting with 0 is not run); `attempts`: the
    attempts per sample (protocol repeats x --attempt-count); `unit`: what one attempt is
    ("generations", "conversations", "requests"). `register`: remember it as `eval_name`'s plan
    for this process (planned_for); base.run_eval_suite's own per-call plan passes False, so a
    suite that calls it several times gets each call's plan, not the first call's."""
    attempts = max(1, int(attempts or 1))
    samples = int(samples)
    lim = int(limit) if limit and int(limit) > 0 else None
    settings = {str(k): int(v) for k, v in (per_setting or {}).items()} or None
    head = (f"--eval-limit {lim}" if lim else "full run (no --eval-limit)")
    line = f"[{eval_name}] {head}: {samples} sample(s)"
    if settings:
        ran = {k: v for k, v in settings.items() if v}
        line += (f" over {len(ran)} {setting_noun}(s) ("
                 + ", ".join(f"{k} {v}" for k, v in ran.items()) + ")")
        idle = [k for k, v in settings.items() if not v]
        if len(idle) > 6:
            line += f" ({len(idle)} other {setting_noun}(s) not run)"
        elif idle:
            line += f" ({', '.join(idle)}: 0, not run)"
    line += (f" x {attempts} attempt{'s' if attempts != 1 else ''} = {samples * attempts} {unit}")
    logger.info(line)
    plan = {"limit": lim, "samples": samples, "per_setting": settings,
            "attempts_per_sample": attempts, unit: samples * attempts}
    if register:
        record(eval_name, plan)
    return plan


def record(eval_name: str, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Remember `plan` as the plan `eval_name` logged in this process (a suite that logs its own
    line, e.g. a full run whose counts are not known yet, registers it here so base does not log
    and record a second one). Returns `plan`."""
    _PLANNED[eval_name] = plan
    return plan


def clear_plan(eval_name: str) -> None:
    """Forget the plan an earlier run of `eval_name` logged in this process (evals.py calls it
    before each suite starts, so planned_for only sees this run's plan)."""
    _PLANNED.pop(eval_name, None)


def planned_for(eval_name: str) -> Optional[Dict[str, Any]]:
    """The plan `eval_name` logged in this process, or None."""
    return _PLANNED.get(eval_name)


def _is_count(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _per_sample(key: str) -> bool:
    """A run constant, the same on every shard, not a count to sum: the limit and the per-sample
    (per-rollout, per-part) fields such as attempts_per_sample, tries_per_rollout_at_most and
    phase_runs_per_mini_scenario."""
    return key == "limit" or "_per_" in key or (key.startswith("per_") and key != "per_setting")


def merge_plans(results: Any) -> Optional[Dict[str, Any]]:
    """The `eval_limit_plan` of a merged sharded run, or None when no shard recorded one.

    Counts (samples, per-setting counts, the unit totals) are summed over the shards; a count a
    shard does not know (None) keeps the merged count None, never 0. The limit, the per-sample
    constants (attempts_per_sample, *_per_*) and non-numeric fields (a note) are the run's: kept
    when every shard has the same value, else dropped and named in `unmerged_fields`. A shard
    that ran 0 samples (an empty shard) contributes 0 to a count it did not record, and a field
    only empty shards recorded is left out."""
    plans = [r.get("eval_limit_plan") for r in results
             if isinstance(r, dict) and isinstance(r.get("eval_limit_plan"), dict)]
    if not plans:
        return None
    _MISSING = object()

    def empty(p: Mapping[str, Any]) -> bool:
        return p.get("samples") == 0

    def total(values: list) -> Any:
        if any(v is None for v in values):
            return None
        out = sum(values)
        return int(out) if float(out).is_integer() else out

    merged: Dict[str, Any] = {}
    unmerged = []
    # an empty shard's plan may name a unit the others do not (base's "generations" where the
    # suite counts "episodes"): its fields count only where a shard that ran samples has them
    ran = [p for p in plans if not empty(p)] or plans
    keys = []
    for p in ran:
        keys.extend(k for k in p if k not in keys)
    for key in keys:
        if key == "per_setting":
            continue
        raw = [p.get(key, _MISSING) for p in plans]
        present = [v for v in raw if v is not _MISSING]
        if _per_sample(key) or not all(v is None or _is_count(v) for v in present):
            # a run constant: the same on every shard that recorded it (an empty shard may not)
            seen = [v for p, v in zip(plans, raw) if v is not _MISSING and not empty(p)] or present
            if all(v == seen[0] for v in seen):
                merged[key] = seen[0]
            else:
                unmerged.append(key)
            continue
        values = [(0 if empty(p) else None) if v is _MISSING else v for p, v in zip(plans, raw)]
        merged[key] = total(values)
    if any(isinstance(p.get("per_setting"), dict) for p in plans):
        names = []
        for p in plans:
            names.extend(k for k in (p.get("per_setting") or {}) if k not in names)
        settings: Dict[str, Any] = {}
        for name in names:
            values = []
            for p in plans:
                ps = p.get("per_setting")
                if isinstance(ps, dict):
                    values.append(ps.get(name, 0))
                else:
                    values.append(0 if empty(p) else None)
            settings[name] = total(values)
        merged["per_setting"] = settings or None
    else:
        merged["per_setting"] = None
    if unmerged:
        merged["unmerged_fields"] = unmerged
    missing = sum(1 for r in results if isinstance(r, dict)
                  and not isinstance(r.get("eval_limit_plan"), dict))
    if missing:
        # a shard that recorded no plan: the summed counts leave its work out
        merged["shards_without_plan"] = missing
    return merged
