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

"""Shared structural parsing/matching for function-calling evaluation suites.

Several suites were scoring tool use with free-text substring checks: "does the gold tool
name appear anywhere in the response". That credits a model for echoing a tool name out of
the prompt's own API list and ignores the arguments entirely. This module parses BOTH sides
into (name, args) and compares them structurally.

Formats handled (models and datasets each use a different one):
  * OpenAI tool-call rendering from base.py:  ``name({"a": 1}) name(a=1)``
  * bare JSON:                                ``{"name": "f", "arguments": {...}}``
  * python-ish call:                          ``f(a=1, b="x")``
  * ReAct / ToolLLaMA:                        ``Action: f`` + ``Action Input: {"a": 1}``
  * API-Bank:                                 ``API-Request: [f(a=1)]``
"""

from __future__ import annotations

import ast
import copy
import json
import logging
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from . import limit_plan
from .base import OfficialSampling, get_run_knob
from .sampling import allocate_limit, parse_shard, shard_select

logger = logging.getLogger(__name__)

ToolCall = Tuple[str, Dict[str, Any]]

_IDENT = r"[A-Za-z_][A-Za-z0-9_.\-]*"

# OpenAI function names must match ^[a-zA-Z0-9_-]{1,64}$. Datasets that namespace their
# tools (`default_api:flight_book`) have to be rewritten before the schema is sent, so the
# same rewrite is applied to the gold before comparing.
_ILLEGAL_NAME_CHARS = re.compile(r"[^A-Za-z0-9_-]")


# --------------------------------------------------------------------------- #
# --eval-limit: N samples for the whole suite, and the plan every suite logs
# --------------------------------------------------------------------------- #

def run_attempts(kwargs: Dict[str, Any]) -> int:
    """The --attempt-count a suite runs per sample (the run knob when the call sets none)."""
    return max(1, int(kwargs.get("attempt_count") or get_run_knob("attempt_count", 1) or 1))


def part_quotas(limit: Optional[int], parts: Dict[str, Dict[Any, Optional[int]]],
                seed: str) -> Dict[str, Optional[int]]:
    """`--eval-limit N` over a suite's parts (passes it runs separately), N in total.

    `parts` maps each part to its categories and their sizes (None: not known). The N samples
    are dealt over every category of every part (sampling.allocate_limit, seeded on the suite),
    so a part gets the samples of its categories: one per category before any category gets
    two, a category capped at its size. A part can get 0 (not run in that limited run). No limit:
    every part gets None (all of its samples)."""
    names = [p for p, cats in parts.items() if cats]
    if not limit or int(limit) <= 0:
        return {p: None for p in names}
    keys = [(p, c) for p in names for c in parts[p]]
    available = {(p, c): n for p in names for c, n in parts[p].items() if n is not None}
    counts = allocate_limit(int(limit), keys, available=available, seed=seed)
    return {p: sum(counts[(p, c)] or 0 for c in parts[p]) for p in names}


def eval_limit_plan(eval_name: str, limit: Optional[int], samples: Optional[int],
                    per_setting: Optional[Dict[str, Optional[int]]] = None, attempts: int = 1,
                    unit: str = "generations", setting_word: str = "setting",
                    note: str = "") -> Dict[str, Any]:
    """limit_plan.eval_limit_plan (the one-line plan every suite logs at its start, and the
    result's `eval_limit_plan`), for these suites. `per_setting` holds the samples of each
    setting (part, file, category) the suite runs as a separate unit (0: not run), None for a
    suite with no such dimension; `note` (recorded as the plan's `note`) names generations that
    are not samples (prerequisite entries, repair calls). A full run whose counts are not known
    before it starts (`samples` or a setting None) logs and records them as "all" (None)."""
    if samples is not None and all(v is not None for v in (per_setting or {}).values()):
        plan = limit_plan.eval_limit_plan(eval_name, limit, samples, per_setting,
                                          setting_noun=setting_word, attempts=attempts, unit=unit)
    else:
        attempts = max(1, int(attempts))
        settings = ({str(k): None if v is None else int(v) for k, v in per_setting.items()}
                    if per_setting else None)
        logger.info("[%s] full run (no --eval-limit): all samples%s x %d attempt%s = all %s",
                    eval_name, f" of {len(settings)} {setting_word}(s)" if settings else "",
                    attempts, "s" if attempts != 1 else "", unit)
        plan = limit_plan.record(eval_name, {"limit": None, "samples": None,
                                             "per_setting": settings,
                                             "attempts_per_sample": attempts, unit: None})
    if note:
        plan["note"] = note
    return plan


def base_suite_plan(eval_name: str, samples: List[Any], limit: Optional[int], attempts: int,
                    **kw: Any) -> Dict[str, Any]:
    """eval_limit_plan for a suite that runs its samples in one run_eval_suite call: the samples
    base will run of `samples` (the list it is given: GBENCH_SHARD's shard of it, empty
    presharded slots left out, then at most `limit`)."""
    items = [x for x in shard_select(list(samples), parse_shard(os.environ.get("GBENCH_SHARD")))
             if x is not None]
    n = min(int(limit), len(items)) if limit and int(limit) > 0 else len(items)
    return eval_limit_plan(eval_name, limit, n, attempts=attempts, **kw)


def _coerce(value: Any) -> Any:
    """Normalize a scalar for comparison (numbers vs strings, casing, whitespace)."""
    if isinstance(value, str):
        v = value.strip().strip("'\"").strip()
        low = v.lower()
        if low in ("true", "false"):
            return low == "true"
        if low in ("none", "null", ""):
            return None
        try:
            f = float(v)
            return int(f) if f.is_integer() else f
        except ValueError:
            return low
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        f = float(value)
        return int(f) if f.is_integer() else f
    if isinstance(value, dict):
        return {str(k).strip().lower(): _coerce(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    return value


def _norm_args(args: Any) -> Dict[str, Any]:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return {}
    if not isinstance(args, dict):
        return {}
    return {str(k).strip().lower(): _coerce(v) for k, v in args.items()}


def _json_objects(text: str) -> List[Dict[str, Any]]:
    """Every top-level JSON object in free text."""
    out, dec, i, n = [], json.JSONDecoder(), 0, len(text or "")
    while i < n:
        if text[i] == "{":
            try:
                obj, end = dec.raw_decode(text, i)
                if isinstance(obj, dict):
                    out.append(obj)
                i = end
                continue
            except json.JSONDecodeError:
                pass
        i += 1
    return out


def _split_top_level(s: str) -> List[str]:
    """Split on commas that are not inside quotes/brackets."""
    parts, buf, depth, in_str, quote = [], [], 0, False, ""
    for ch in s:
        if in_str:
            if ch == quote:
                in_str = False
        elif ch in "\"'":
            in_str, quote = True, ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _parse_loose_call(snippet: str) -> Optional[ToolCall]:
    """Parse a call whose arguments are not valid Python literals.

    API-Bank golds look like ``ModifyRegistration(appointment_id=abc123,
    new_time=2023-10-10 11:00:00)`` - unquoted identifiers and bare timestamps that `ast`
    rejects outright. Treat each ``k=v`` value as a raw string.
    """
    m = re.match(rf"\s*({_IDENT})\s*\((.*)\)\s*$", snippet.strip(), re.DOTALL)
    if not m:
        return None
    name, inner = m.group(1), m.group(2)
    args: Dict[str, Any] = {}
    for idx, part in enumerate(_split_top_level(inner)):
        if "=" in part:
            k, _, v = part.partition("=")
            args[k.strip().lower()] = _coerce(v)
        else:
            args[f"__pos{idx}"] = _coerce(part)
    return name, args


def _parse_python_call(snippet: str) -> Optional[ToolCall]:
    """Parse ``f(a=1, b="x")`` (and positional args) via the AST, not regex."""
    try:
        node = ast.parse(snippet.strip(), mode="eval").body
    except Exception:
        return _parse_loose_call(snippet)
    if not isinstance(node, ast.Call):
        return None
    fn = node.func
    name = getattr(fn, "id", None) or getattr(fn, "attr", None)
    if not name:
        return None
    args: Dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            continue
        try:
            args[str(kw.arg).lower()] = _coerce(ast.literal_eval(kw.value))
        except Exception:
            # An unquoted value (`unit=units`) parses as a Name/Attribute node, so
            # literal_eval fails while ast.parse succeeded - the loose-parser fallback
            # never runs. Dropping it to None silently discarded the argument and made
            # correct calls unscoreable; keep the source text instead.
            try:
                args[str(kw.arg).lower()] = _coerce(ast.unparse(kw.value))
            except Exception:
                args[str(kw.arg).lower()] = None
    for idx, a in enumerate(node.args):
        try:
            args[f"__pos{idx}"] = _coerce(ast.literal_eval(a))
        except Exception:
            pass
    return name, args


def parse_tool_calls(text: str) -> List[ToolCall]:
    """Extract every tool call we can find, in order of appearance."""
    calls: List[ToolCall] = []
    if not text:
        return calls
    s = str(text)

    # 1) JSON objects carrying a name/arguments pair
    for obj in _json_objects(s):
        name = obj.get("name") or obj.get("action") or obj.get("tool") or obj.get("api")
        if not name or not isinstance(name, str):
            continue
        raw = (obj.get("arguments") if "arguments" in obj else
               obj.get("args") if "args" in obj else
               obj.get("parameters") if "parameters" in obj else
               obj.get("action_input") if "action_input" in obj else {})
        calls.append((name.strip(), _norm_args(raw)))

    # 2) ReAct / ToolLLaMA: "Action: f" with an optional "Action Input: {...}"
    # NB the separator must not be a greedy \s* - that would swallow the newline the
    # optional "Action Input" group needs, silently dropping every argument.
    for m in re.finditer(rf"Action\s*:[ \t]*({_IDENT})[ \t]*(?:\s*Action\s*Input\s*:[ \t]*(\{{.*?\}}|\S[^\n]*))?",
                         s, re.IGNORECASE | re.DOTALL):
        name = m.group(1).strip()
        if name.lower() in ("none", "finish"):
            continue
        calls.append((name, _norm_args(m.group(2) or {})))

    # 3) python-ish calls, incl. API-Bank's "API-Request: [f(a=1)]"
    for m in re.finditer(rf"\b({_IDENT})\s*\(", s):
        start = m.start()
        depth, i, in_str, quote = 0, m.end() - 1, False, ""
        while i < len(s):
            ch = s[i]
            if in_str:
                if ch == quote and s[i - 1] != "\\":
                    in_str = False
            elif ch in "\"'":
                in_str, quote = True, ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        parsed = _parse_python_call(s[start:i + 1])
        if parsed:
            calls.append(parsed)

    # de-duplicate, preserving order
    seen, uniq = set(), []
    for name, args in calls:
        key = (name.lower(), json.dumps(args, sort_keys=True, default=str))
        if key not in seen:
            seen.add(key)
            uniq.append((name, args))
    return uniq


def parse_gold_call(gold: Any) -> Optional[ToolCall]:
    """Parse the dataset's gold into (name, args). Handles dict/list/JSON/ReAct/API-Request."""
    if gold is None:
        return None
    if isinstance(gold, dict):
        name = gold.get("name") or gold.get("action") or gold.get("tool") or gold.get("api")
        if name:
            raw = gold.get("arguments", gold.get("args", gold.get("parameters", {})))
            return str(name).strip(), _norm_args(raw)
        return None
    if isinstance(gold, list):
        for item in gold:                      # ComplexFuncBench stores a LIST of calls
            parsed = parse_gold_call(item)
            if parsed:
                return parsed
        return None
    text = str(gold).strip()
    if not text:
        return None
    # Some datasets store the gold as a PYTHON literal, not JSON: seal_tools ships
    # "[{'api': 'analyzeEvidence', 'parameters': {...}}]". Single quotes defeat
    # json.loads, and there is no ReAct or `f(...)` form to fall back on, so the gold
    # parsed to None and every row failed regardless of the answer (seal_tools: 0/20).
    if text[:1] in "[{" and '"' not in text[:200]:
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            value = None
        if isinstance(value, (list, dict)):
            parsed = parse_gold_call(value)
            if parsed:
                return parsed
    # API-Bank golds are prefixed "API-Request: [f(a=1)]" while the prompt asks for "[f(...)]"
    text = re.sub(r"^\s*API-Request\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    calls = parse_tool_calls(text)
    return calls[0] if calls else None


def parse_gold_calls(gold: Any) -> List[ToolCall]:
    """ALL gold calls as (name, args) - parse_gold_call() returns only the first.

    Needed by multi-call suites (seal_tools, nestful) whose gold is a LIST of calls; scoring
    only the first collapsed multi-call answers to a single call.
    """
    if gold is None:
        return []
    if isinstance(gold, dict):
        c = parse_gold_call(gold)
        return [c] if c else []
    if isinstance(gold, list):
        out: List[ToolCall] = []
        for item in gold:
            out.extend(parse_gold_calls(item))
        return out
    text = str(gold).strip()
    if not text:
        return []
    # python-literal list/dict with single quotes (seal_tools ships these): literal_eval then
    # recurse, so every call is kept rather than only calls[0].
    if text[:1] in "[{" and '"' not in text[:200]:
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            value = None
        if isinstance(value, (list, dict)):
            return parse_gold_calls(value)
    text = re.sub(r"^\s*API-Request\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    return parse_tool_calls(text)


def gold_call_dicts(gold: Any) -> List[Dict[str, Any]]:
    """Gold as a list of ``{name, args}`` dicts, ready for score_exact_call_set/score_sequence."""
    return [{"name": n, "args": a} for n, a in parse_gold_calls(gold)]


def call_matches(gold: ToolCall, candidates: List[ToolCall], require_args: bool = True) -> bool:
    """True iff some candidate call has the gold's name and (optionally) its arguments.

    Argument matching is subset-based: every gold argument must be present with an
    equivalent value. Extra model-supplied arguments are tolerated (datasets often omit
    optional defaults), but a missing or contradicting gold argument fails.
    """
    if not gold:
        return False
    gold_name, gold_args = gold[0].strip().lower(), gold[1] or {}
    for name, args in candidates:
        if name.strip().lower() != gold_name:
            continue
        if not require_args or not gold_args:
            return True
        args = args or {}
        # Value comparison goes through _values_equal (float tolerance, case-insensitive
        # strings, and a bool-before-numeric guard so a boolean gold answered with 1 does
        # NOT match) instead of a bare `==`, which accepted True==1 and rejected 3==3.0.
        if all(k in args and _values_equal(v, args[k]) for k, v in gold_args.items()):
            return True
    return False


def score_tool_call(response_text: str, gold: Any, require_args: bool = True) -> bool:
    """Convenience: parse both sides and compare structurally."""
    g = parse_gold_call(gold)
    if not g:
        return False
    return call_matches(g, parse_tool_calls(response_text), require_args=require_args)


# --------------------------------------------------------------------------- #
# Exact FunctionCall-AST matching (`exact_tool_call_accuracy`)
#
# Distinct from call_matches() above, which is deliberately lenient: it looks for ONE gold
# call among the candidates and tolerates extra arguments. The exact match metric requires an
# exact match over the whole response - function name, argument key set, parameter types
# and values - and penalises both omitted and hallucinated calls/arguments.
# --------------------------------------------------------------------------- #

def normalize_tool_name(name: str) -> str:
    """Canonical form used to compare a gold tool name with an emitted one."""
    return _ILLEGAL_NAME_CHARS.sub("_", str(name or "").strip()).casefold()


def parse_raw_tool_calls(tool_calls: Any) -> List[ToolCall]:
    """Convert an OpenAI ``tool_calls`` array into (name, args) keeping JSON types.

    Unlike parse_tool_calls(), this does not go through the text rendering, so an INTEGER
    parameter answered with the string ``"2010"`` stays a string and can be caught by the
    type-conformance check.
    """
    out: List[ToolCall] = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else call
        name = fn.get("name")
        if not name:
            continue
        raw = fn.get("arguments", {})
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = {}
        out.append((str(name), dict(raw) if isinstance(raw, dict) else {}))
    return out


def _values_equal(gold: Any, pred: Any) -> bool:
    """Value equivalence: float tolerance, case-insensitive strings, structural containers.

    `bool` is checked before the numeric branch on purpose - in Python ``True == 1``, so a
    boolean parameter answered with ``1`` would otherwise be accepted.
    """
    if isinstance(gold, bool) or isinstance(pred, bool):
        return isinstance(gold, bool) and isinstance(pred, bool) and gold == pred
    if gold is None or pred is None:
        return gold is None and pred is None
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        return math.isclose(float(gold), float(pred), rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(gold, str) and isinstance(pred, (int, float)):
        return False
    if isinstance(gold, (int, float)) and isinstance(pred, str):
        try:
            return math.isclose(float(gold), float(pred.strip()), rel_tol=1e-6, abs_tol=1e-9)
        except ValueError:
            return False
    if isinstance(gold, str) and isinstance(pred, str):
        return gold.strip().casefold() == pred.strip().casefold()
    if isinstance(gold, (list, tuple)) and isinstance(pred, (list, tuple)):
        return len(gold) == len(pred) and all(_values_equal(g, p) for g, p in zip(gold, pred))
    if isinstance(gold, dict) and isinstance(pred, dict):
        gkeys = {str(k).strip().casefold() for k in gold}
        pkeys = {str(k).strip().casefold() for k in pred}
        if gkeys != pkeys:
            return False
        lowered = {str(k).strip().casefold(): v for k, v in pred.items()}
        return all(_values_equal(v, lowered[str(k).strip().casefold()]) for k, v in gold.items())
    return gold == pred


_JSON_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, (list, tuple)),
    "object": lambda v: isinstance(v, dict),
}


def _schema_properties(tool_schemas: Any, name: str) -> Dict[str, Any]:
    for tool in tool_schemas or []:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        if normalize_tool_name(fn.get("name", "")) == normalize_tool_name(name):
            params = fn.get("parameters") or {}
            props = params.get("properties")
            return props if isinstance(props, dict) else {}
    return {}


def _types_conform(name: str, args: Dict[str, Any], props: Dict[str, Any]) -> Tuple[bool, str]:
    """Check emitted argument types against the declared schema (rubric rule 3)."""
    if not props:
        return True, ""
    for key, value in args.items():
        spec = props.get(key)
        if not isinstance(spec, dict):
            continue
        declared = spec.get("type")
        if value is None and spec.get("nullable"):
            continue
        check = _JSON_TYPE_CHECKS.get(declared)
        if check and not check(value):
            return False, f"{name}.{key}: expected {declared}, got {type(value).__name__}"
    return True, ""


def _call_equal(gold: Dict[str, Any], pred: ToolCall) -> bool:
    if normalize_tool_name(gold.get("name", "")) != normalize_tool_name(pred[0]):
        return False
    gargs = gold.get("args") or {}
    pargs = pred[1] or {}
    gkeys = {str(k).strip().casefold() for k in gargs}
    pkeys = {str(k).strip().casefold() for k in pargs}
    if gkeys != pkeys:                        # omitted or hallucinated argument -> fail
        return False
    lowered = {str(k).strip().casefold(): v for k, v in pargs.items()}
    return all(_values_equal(v, lowered[str(k).strip().casefold()]) for k, v in gargs.items())


def score_exact_call_set(
    gold_calls: List[Dict[str, Any]],
    pred_calls: List[ToolCall],
    tool_schemas: Any = None,
) -> Dict[str, Any]:
    """Exact match over the whole set of emitted function calls.

    Rules (docs/eval-rubrics/agent_and_tool_use_rubrics.md §1):
      1. every gold call is emitted with the same name;
      2. no missing and no hallucinated call (cardinality must match exactly, so an
         unwarranted extra call fails the sample);
      3. per call the argument key sets are equal and each value is equivalent;
      4. each emitted value conforms to the declared parameter type.

    Order is not significant: these are parallel, independent calls in one turn.
    Returns the verdict plus the sub-checks, so a run can report *why* it failed rather
    than only a pass rate.
    """
    detail: Dict[str, Any] = {
        "n_gold": len(gold_calls), "n_pred": len(pred_calls),
        "names_ok": False, "args_ok": False, "types_ok": True, "reason": "",
    }
    if not gold_calls:
        detail["reason"] = "no gold calls"
        return {"exact": False, **detail}

    gold_names = sorted(normalize_tool_name(g.get("name", "")) for g in gold_calls)
    pred_names = sorted(normalize_tool_name(p[0]) for p in pred_calls)
    detail["names_ok"] = gold_names == pred_names
    if len(pred_calls) != len(gold_calls):
        detail["reason"] = (f"expected {len(gold_calls)} call(s), got {len(pred_calls)}"
                            if pred_calls else "no tool call emitted")
        return {"exact": False, **detail}

    remaining = list(pred_calls)
    for gold in gold_calls:
        match = next((p for p in remaining if _call_equal(gold, p)), None)
        if match is None:
            same_name = [p for p in remaining
                         if normalize_tool_name(p[0]) == normalize_tool_name(gold.get("name", ""))]
            detail["reason"] = (f"arguments differ for {gold.get('name')}" if same_name
                                else f"missing call {gold.get('name')}")
            return {"exact": False, **detail}
        remaining.remove(match)
    detail["args_ok"] = True

    for name, args in pred_calls:
        # Prefer the full declared schema; fall back to the compact `param_types` the gold
        # carries, so type conformance still works when the 100+ tool schemas of the
        # request are not retained alongside the result.
        props = _schema_properties(tool_schemas, name)
        if not props:
            gold = next((g for g in gold_calls
                         if normalize_tool_name(g.get("name", "")) == normalize_tool_name(name)), None)
            props = {k: {"type": v} for k, v in ((gold or {}).get("param_types") or {}).items()}
        ok, why = _types_conform(name, args, props)
        if not ok:
            detail["types_ok"] = False
            detail["reason"] = why
            return {"exact": False, **detail}

    return {"exact": True, **detail}


def score_possible_answer(response_text: str, gold: Any,
                          calls: Optional[List[ToolCall]] = None,
                          exact_keys: Optional[bool] = None) -> bool:
    """Name + accepted-value match against a BFCL-style `possible_answer`.

    Not BFCL's checker: it has no call-count, type or string-standardisation rules. The bfcl
    suites score with bfcl_eval's own eval_runner in the harness image (bfcl.py); bfcl_score()
    below is a port of it.

    Gold is a list of ``{func_name: {param: [accepted values...]}}``. A parameter whose
    accepted list contains "" is optional. Matching is structural: the model must emit a
    call with that function name, and every required parameter must be present with one of
    its accepted values. The previous check tested whether the function name and some value
    appeared anywhere in the response text, which a model could satisfy by echoing the
    prompt's API list without ever emitting a call.

    `calls`: pre-parsed (name, args) pairs. Pass the STRUCTURED tool calls
    (parse_raw_tool_calls(reply.tool_calls)) so scoring does not go through base.py's lossy
    text render `name({json}) name(a=1)` (which drops spaces in unquoted values, splits on
    commas inside unquoted strings, truncates namespaced function names, and swallows the
    JSON object as a positional arg). Falls back to parsing `response_text` when omitted.

    `exact_keys`: match parameter names exactly, as BFCL's checker does. Defaults to True when
    `calls` are given (structured calls keep the model's key case) and False for the text
    fallback, whose parser lower-cases keys.
    """
    if exact_keys is None:
        exact_keys = calls is not None
    if not gold:
        return False
    specs = gold if isinstance(gold, list) else [gold]
    if calls is None:
        if not response_text:
            return False
        calls = parse_tool_calls(response_text)
    if not calls:
        return False

    for spec in specs:
        if isinstance(spec, str):
            parsed = parse_gold_call(spec)
            if not parsed or not call_matches(parsed, calls):
                return False
            continue
        if not isinstance(spec, dict):
            return False
        for fname, params in spec.items():
            # EVERY candidate with this name must be tried, not just the first.
            # base.py renders one tool call twice - `f({"a": 1}) f(a=1)` - so
            # parse_tool_calls returns two entries for it: the first has the whole JSON
            # object swallowed as a positional `__pos0`, the second has the named args.
            # Binding to the first match scored correct calls as wrong (bfcl: 0/20).
            candidates = [args or {} for name, args in calls
                          if name.strip().lower() == str(fname).strip().lower()]
            if not candidates:
                return False                      # required function never called
            if not isinstance(params, dict):
                continue
            if not any(_args_satisfy(cand, params, exact_keys) for cand in candidates):
                return False
    return True


def _value_matches(accepted: Any, cand_val: Any) -> bool:
    """A candidate value satisfies a possible_answer accepted value if it is case-insensitively /
    structurally equal (this handles STRUCTURED args, which keep their original case and JSON
    types -- e.g. 'J.K. Rowling', ['bookA','bookB'], the string '12345'), OR equals the
    type-coerced accepted value (keeps the legacy path where a string gold like "5" is answered
    with the int 5). The text-render path relied on both sides being lowercased by _coerce; the
    structured path does not lowercase the candidate, so the case-insensitive comparator is needed."""
    return _values_equal(accepted, cand_val) or cand_val == _coerce(accepted)


def _args_satisfy(cand_args: Dict[str, Any], params: Dict[str, Any],
                  exact_keys: bool = False) -> bool:
    """Does this candidate call's argument dict satisfy a BFCL `possible_answer` spec?"""
    # No UNEXPECTED parameters: possible_answer lists every acceptable param (optional ones
    # carry "" among their accepted values), so an argument key absent from the spec is a
    # hallucinated parameter and fails, per BFCL's AST checker. (A double-rendered candidate
    # carrying a positional `__pos0` fails here; the correctly-named candidate still passes
    # via score_possible_answer's any-candidate match.)
    # Parameter names match exactly for structured calls (canonical BFCL); the text parser
    # lower-cases keys, so that path compares both sides lower-cased.
    norm = (lambda k: str(k).strip()) if exact_keys else (lambda k: str(k).strip().lower())
    spec_keys = {norm(p) for p in params}
    cand_args = {norm(k): v for k, v in (cand_args or {}).items()}
    for key in cand_args:
        if key not in spec_keys:
            return False
    for pname, accepted in params.items():
        allowed = accepted if isinstance(accepted, list) else [accepted]
        key = norm(pname)
        optional = any(isinstance(a, str) and a == "" for a in allowed)
        if key not in cand_args:
            # OMITTED: fine iff the param is optional ("" among its accepted values means
            # "may be omitted"); a missing required param fails.
            if optional:
                continue
            return False
        # SUPPLIED: the value must match an accepted value EVEN IF the param is optional. "" means
        # "may be omitted", NOT "any value is acceptable" - the canonical BFCL AST checker validates
        # a provided value regardless of optionality. (The old code short-circuited on any optional
        # param, over-crediting a supplied MISMATCH e.g. include_revoked=true vs accepted ["", false].)
        cand_val = cand_args[key]
        # Nested possible_answer: an accepted value can itself be a {subparam: [accepted...]}
        # object (BFCL's nested-dict params, e.g. profile_data). Recurse rather than flat-compare
        # the concrete dict against a spec whose values are accepted-lists (which never == it).
        dict_specs = [a for a in allowed if isinstance(a, dict)]
        if dict_specs and isinstance(cand_val, dict) and any(
                _args_satisfy(cand_val, ds, exact_keys) for ds in dict_specs):
            continue
        if any(_value_matches(a, cand_val) for a in allowed if not isinstance(a, dict)):
            continue
        return False
    return True


# --------------------------------------------------------------------------- #
# Corpus-pooled precision/recall/F1 and position-aligned sequence matching.
#
# seal_tools reports micro Tool/Parameter P/R/F1 and NESTFUL reports Full/Partial
# Sequence Match plus F1-Func / F1-Param. Both are CORPUS-level metrics: precision and
# recall pool true/false positives across the whole run, so they CANNOT be averaged per
# sample (mean-of-ratios is wrong). The scorers below return per-sample (tp, fp, fn)
# counts; a suite pools them over result['sample_traces'] in a trace-walk and calls
# prf_from_counts() once at the end.
# --------------------------------------------------------------------------- #


def _call_name_args(call: Any) -> Tuple[str, Dict[str, Any]]:
    """(name, args) from either a ToolCall tuple or a {name, args} gold dict."""
    if isinstance(call, tuple):
        return call[0], (call[1] or {})
    if isinstance(call, dict):
        name = call.get("name") or call.get("action") or call.get("tool") or call.get("api") or ""
        args = call.get("args", call.get("arguments", call.get("parameters", {}))) or {}
        return name, args
    return "", {}


def _canon_value(value: Any) -> str:
    """A hashable, comparison-stable rendering of a (coerced) argument value."""
    try:
        return json.dumps(_coerce(value), sort_keys=True, default=str)
    except Exception:
        return str(value)


def tool_items(calls: List[Any]) -> List[str]:
    """Tool-name multiset for a list of calls (F1-Func / Tool-P/R/F1)."""
    return [normalize_tool_name(_call_name_args(c)[0]) for c in (calls or [])]


def param_items(calls: List[Any]) -> List[Tuple[str, str, str]]:
    """(tool, key, value) triple multiset for parameter-level P/R/F1."""
    out: List[Tuple[str, str, str]] = []
    for c in calls or []:
        name, args = _call_name_args(c)
        nn = normalize_tool_name(name)
        for k, v in (args or {}).items():
            out.append((nn, str(k).strip().casefold(), _canon_value(v)))
    return out


def prf_counts(gold_items: List[Any], pred_items: List[Any]) -> Tuple[int, int, int]:
    """Multiset (tp, fp, fn) between two collections of hashable items."""
    g, p = Counter(gold_items), Counter(pred_items)
    tp = sum((g & p).values())
    return tp, sum((p - g).values()), sum((g - p).values())


def prf_from_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Micro precision/recall/F1 from pooled counts."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def score_sequence(gold_calls: List[Dict[str, Any]], pred_calls: List[ToolCall]) -> Dict[str, Any]:
    """Position-aligned sequence scoring for NESTFUL-style multi-call answers.

    Returns per sample:
      * full    - the whole sequence matches in order (name+args equal, same length);
      * partial - fraction of gold positions matched at the SAME index;
      * func    - (tp, fp, fn) over the tool-name multiset (pool -> F1-Func);
      * param   - (tp, fp, fn) over (tool, key, value) triples (pool -> F1-Param).

    Order matters (a nested call consumes an earlier call's output), so matching is
    index-aligned, unlike the order-insensitive set membership of call_matches().
    """
    n, m = len(gold_calls), len(pred_calls)
    pos_correct = sum(1 for i in range(min(n, m)) if _call_equal(gold_calls[i], pred_calls[i]))
    full = (n == m and pos_correct == n)
    partial = (pos_correct / n) if n else (1.0 if m == 0 else 0.0)
    return {
        "full": full,
        "partial": partial,
        "func": prf_counts(tool_items(gold_calls), tool_items(pred_calls)),
        "param": prf_counts(param_items(gold_calls), param_items(pred_calls)),
    }


# --------------------------------------------------------------------------- #
# BFCL canonical single-turn pipeline
#
# Ported from gorilla's bfcl_eval at BFCL_EVAL_COMMIT (bfcl-eval 2026.3.23 on PyPI is byte-identical
# to that tree). The bfcl and bfcl_live suites build their single-turn requests with the request
# half of this port (bfcl_request_functions, bfcl_convert_to_tools) and score the replies with
# bfcl_eval's own eval_runner in the local harness image (bfcl.single_turn_verdicts); bfcl_score is
# the ported checker, kept as a reference implementation that the tests hold against eval_runner.
# BFCL's multi-turn categories run inside the package itself (bfcl.py). Sources, all under
# berkeley-function-call-leaderboard/bfcl_eval/:
#   constants/type_mappings.py                        GORILLA_TO_OPENAPI, *_TYPE_CONVERSION
#   utils.py                                          _func_doc_language_specific_pre_processing,
#                                                     is_function_calling_format_output,
#                                                     is_empty_output
#   model_handler/utils.py                            convert_to_tool, _cast_to_openai_type
#   model_handler/api_inference/openai_completion.py  decode_ast (FC mode)
#   eval_checker/eval_runner.py                       _evaluate_single_relevance_entry,
#                                                     _evaluate_single_ast_entry
#   eval_checker/ast_eval/ast_checker.py              ast_checker and helpers
#   eval_checker/ast_eval/type_convertor/*.py         java/js type converters
# Logic, error strings and error_type codes follow upstream, so a trace reads like a BFCL
# score file. Tools are advertised the way BFCL's OpenAI chat-completions FC handler does
# (ModelStyle.OPENAI_COMPLETIONS, whose models run with underscore_to_dot=True).
# --------------------------------------------------------------------------- #

BFCL_EVAL_COMMIT = "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"

#: BFCL generates every model at temperature 0.001: the default of `bfcl generate --temperature`
#: (bfcl_eval/__main__.py, and _llm_response_generation.py's --temperature), which the handler
#: sends on every request (OpenAICompletionsHandler._query_FC, with messages, model, store=False
#: and the tools, and no max_tokens). bfcl, bfcl_live and bfcl_v4_agentic record it; they sample
#: at gbench's rule, 1.0 with --thinking and 0.0 without (base.resolve_official_sampling).
BFCL_OFFICIAL_SAMPLING = OfficialSampling(
    reference=(f"ShishirPatil/gorilla@{BFCL_EVAL_COMMIT[:12]} berkeley-function-call-leaderboard/"
               "bfcl_eval/__main__.py (generate --temperature, default 0.001), sent by "
               "OpenAICompletionsHandler._query_FC without max_tokens"),
    temperature=0.001,
    send_max_tokens=False,
)

BFCL_GORILLA_TO_OPENAPI = {
    "integer": "integer",
    "number": "number",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "dict": "object",
    "object": "object",
    "tuple": "array",
    "any": "string",
    "byte": "integer",
    "short": "integer",
    "long": "integer",
    "double": "number",
    "char": "string",
    "ArrayList": "array",
    "Array": "array",
    "HashMap": "object",
    "Hashtable": "object",
    "Queue": "array",
    "Stack": "array",
    "Any": "string",
    "String": "string",
    "Bigint": "integer",
}

_BFCL_JAVA_TYPE_CONVERSION = {
    "byte": int,
    "short": int,
    "integer": int,
    "float": float,
    "double": float,
    "long": int,
    "boolean": bool,
    "char": str,
    "Array": list,
    "ArrayList": list,
    "Set": set,
    "HashMap": dict,
    "Hashtable": dict,
    "Queue": list,
    "Stack": list,
    "String": str,
    "any": str,
}

_BFCL_JS_TYPE_CONVERSION = {
    "String": str,
    "integer": int,
    "float": float,
    "Bigint": int,
    "Boolean": bool,
    "dict": dict,
    "array": list,
    "any": str,
}

_BFCL_PYTHON_TYPE_MAPPING = {
    "string": str,
    "integer": int,
    "float": float,
    "boolean": bool,
    "array": list,
    "tuple": list,
    "dict": dict,
    "any": str,
}

# Types whose values are checked one level deep.
_BFCL_PYTHON_NESTED_TYPE_CHECK_LIST = ["array", "tuple"]
_BFCL_NESTED_CONVERSION_TYPE_LIST = ["Array", "ArrayList", "array"]


def bfcl_language(category: str) -> str:
    """'java', 'javascript' or 'python' for a BFCL test category (utils.is_java / is_js)."""
    if "javascript" in category:
        return "javascript"
    if "java" in category:
        return "java"
    return "python"


def _bfcl_language_hint(category: str) -> str:
    language = bfcl_language(category)
    if language == "java":
        return " Note that the provided function is in Java 8 SDK syntax."
    if language == "javascript":
        return " Note that the provided function is in JavaScript syntax."
    return " Note that the provided function is in Python 3 syntax."


def bfcl_request_functions(functions: List[Dict[str, Any]], category: str) -> List[Dict[str, Any]]:
    """The function docs as BFCL shows them to the model.

    Port of utils._func_doc_language_specific_pre_processing: appends the language hint to
    every description and, for Java/JavaScript, re-types every parameter as a string that
    must hold the value in source syntax. Works on a copy, because scoring uses the ORIGINAL
    docs (BFCL loads them with include_language_specific_hint=False for evaluation).
    """
    function = copy.deepcopy(functions)
    if len(function) == 0:
        return function
    language = bfcl_language(category)
    for item in function:
        item["description"] = item["description"] + _bfcl_language_hint(category)
        properties = item["parameters"]["properties"]
        if language == "java":
            for key, value in properties.items():
                if value["type"] == "any":
                    properties[key]["description"] += (
                        " This parameter can be of any type of Java object in string representation.")
                else:
                    value["description"] += (
                        f" This is Java {value['type']} type parameter in string representation.")
                if value["type"] == "ArrayList" or value["type"] == "Array":
                    value["description"] += (
                        f" The list elements are of type {value['items']['type']}; they are not in "
                        "string representation.")
                    del value["items"]
                value["type"] = "string"
        elif language == "javascript":
            for key, value in properties.items():
                if value["type"] == "any":
                    properties[key]["description"] += (
                        " This parameter can be of any type of JavaScript object in string representation.")
                else:
                    value["description"] += (
                        f" This is JavaScript {value['type']} type parameter in string representation.")
                if value["type"] == "array":
                    value["description"] += (
                        f" The list elements are of type {value['items']['type']}; they are not in "
                        "string representation.")
                    del value["items"]
                if value["type"] == "dict":
                    if "properties" in value:  # not every dict has properties
                        value["description"] += (
                            " The dictionary entries have the following schema; they are not in "
                            f"string representation. {json.dumps(value['properties'])}")
                        del value["properties"]
                value["type"] = "string"
    return function


def _bfcl_cast_to_openai_type(properties: Dict[str, Any]) -> Dict[str, Any]:
    """model_handler/utils._cast_to_openai_type with the GORILLA_TO_OPENAPI mapping."""
    mapping = BFCL_GORILLA_TO_OPENAPI
    for key, value in properties.items():
        if "type" not in value:
            properties[key]["type"] = "string"
        else:
            var_type = value["type"]
            if var_type == "float":
                properties[key]["format"] = "float"
                properties[key]["description"] += " This is a float type value."
            if var_type in mapping:
                properties[key]["type"] = mapping[var_type]
            else:
                properties[key]["type"] = "string"

        # Supported nesting: list of any, list of list of any, list of dict, list of list
        # of dict, dict of any.
        if properties[key]["type"] == "array" or properties[key]["type"] == "object":
            if "properties" in properties[key]:
                properties[key]["properties"] = _bfcl_cast_to_openai_type(
                    properties[key]["properties"])
            elif "items" in properties[key]:
                properties[key]["items"]["type"] = mapping[properties[key]["items"]["type"]]
                if (properties[key]["items"]["type"] == "array"
                        and "items" in properties[key]["items"]):
                    properties[key]["items"]["items"]["type"] = mapping[
                        properties[key]["items"]["items"]["type"]]
                elif (properties[key]["items"]["type"] == "object"
                        and "properties" in properties[key]["items"]):
                    properties[key]["items"]["properties"] = _bfcl_cast_to_openai_type(
                        properties[key]["items"]["properties"])
    return properties


def bfcl_convert_to_tools(functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """OpenAI `tools` for BFCL function docs: convert_to_tool(functions, GORILLA_TO_OPENAPI,
    ModelStyle.OPENAI_COMPLETIONS). Dots in function names become underscores, since OpenAI
    names must match ^[a-zA-Z0-9_-]{1,64}$; the checker expects the rewritten name back."""
    functions = copy.deepcopy(functions)
    oai_tool = []
    for item in functions:
        if "." in item["name"]:
            item["name"] = re.sub(r"\.", "_", item["name"])
        item["parameters"]["type"] = "object"
        item["parameters"]["properties"] = _bfcl_cast_to_openai_type(item["parameters"]["properties"])
        oai_tool.append({"type": "function", "function": item})
    return oai_tool


def bfcl_decode_fc(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Structured tool calls as BFCL's FC decode_ast returns them: ``[{name: arguments}]``.

    Raises when an `arguments` string is not valid JSON, as upstream does (the reply then
    fails an AST category and counts as "no call" for irrelevance).
    """
    decoded_output = []
    for call in tool_calls:
        fn = call.get("function") if isinstance(call.get("function"), dict) else call
        args = fn.get("arguments")
        decoded_output.append({fn["name"]: json.loads(args) if isinstance(args, str) else args})
    return decoded_output


def bfcl_is_function_calling_format_output(decoded_output: Any) -> bool:
    """utils.is_function_calling_format_output: ``[{func: {param: value}}, ...]`` ([] included)."""
    if type(decoded_output) != list:
        return False
    for item in decoded_output:
        if type(item) != dict:
            return False
        if len(item) != 1:
            return False
        if type(list(item.values())[0]) != dict:
            return False
    return True


def bfcl_is_empty_output(decoded_output: Any) -> bool:
    """utils.is_empty_output: [], [{}] and anything not in call format hold no call."""
    if not bfcl_is_function_calling_format_output(decoded_output):
        return True
    if len(decoded_output) == 0:
        return True
    if len(decoded_output) == 1 and len(decoded_output[0]) == 0:
        return True
    return False


# ---- ast_eval/type_convertor/java_type_converter.py ---------------------------------------

def _bfcl_java_type_converter(value, expected_type, nested_type=None):
    if expected_type not in _BFCL_JAVA_TYPE_CONVERSION:
        raise ValueError(f"Unsupported type: {expected_type}")
    if expected_type == "byte" or expected_type == "short" or expected_type == "integer":
        if not re.match(r"^-?\d+$", value):
            return str(value)  # default to string
        return int(value)
    elif expected_type == "float":
        if not re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?[fF]$", value):
            return str(value)  # default to string
        return float(re.sub(r"[fF]$", "", value))
    elif expected_type == "double":
        if not re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$", value):
            return str(value)  # default to string
        return float(value)
    elif expected_type == "long":
        if not re.match(r"^-?\d+[lL]$", value):
            return str(value)  # default to string
        return int(re.sub(r"[lL]$", "", value))
    elif expected_type == "boolean":
        if value not in ["true", "false"]:
            return str(value)  # default to string
        return value == "true"
    elif expected_type == "char":
        # Upstream's pattern (r"^\'.$\'") can never match, so a char is always kept as-is.
        if not re.match(r"^\'.$\'", value):
            return str(value)  # default to string
        return value
    elif expected_type == "Array" or expected_type == "ArrayList":
        return _bfcl_parse_java_collection(value, expected_type, nested_type)
    elif expected_type == "Set":
        raise NotImplementedError("Set conversion is not implemented")
    elif expected_type == "HashMap":
        return _bfcl_parse_java_collection(value, expected_type, nested_type)
    elif expected_type == "Hashtable":
        raise NotImplementedError("Set conversion is not implemented")
    elif expected_type == "Queue" or expected_type == "Stack":
        raise NotImplementedError(f"{expected_type} conversion is not implemented")
    elif expected_type == "String" or expected_type == "any":
        return str(value)  # we output as string for `any` type
    else:
        raise ValueError(f"Unsupported type: {expected_type}")


def _bfcl_parse_java_collection(input_str, type_str, nested_type=None):
    if type_str == "ArrayList":
        return _bfcl_parse_arraylist(input_str, nested_type)
    elif type_str == "Array":
        return _bfcl_parse_array(input_str, nested_type)
    elif type_str == "HashMap":
        return _bfcl_parse_hashmap(input_str)
    else:
        raise ValueError(f"Unsupported type: {type_str}")


def _bfcl_parse_arraylist(input_str, nested_type=None):
    match_as_list = re.search(r"new\s+ArrayList<\w*>\(Arrays\.asList\((.+?)\)\)", input_str)
    if match_as_list:
        elements_str = match_as_list.group(1)
        elements = []
        for element_str in elements_str.split(","):
            element_str = element_str.strip()
            if nested_type == "char":
                element = element_str[1:-1]  # Remove the single quotes
            elif nested_type == "String":
                element = element_str[1:-1]  # Remove the double quotes
            else:
                element = (_bfcl_java_type_converter(element_str, nested_type)
                           if nested_type else _bfcl_parse_java_value(element_str))
            elements.append(element)
        return elements

    match_add = re.search(r"new\s+ArrayList<\w*>\(\)\s*\{\{\s*(.+?)\s*\}\}", input_str, re.DOTALL)
    if match_add:
        adds_str = match_add.group(1)
        elements = []
        matches = re.findall(r"add\((.+?)\)", adds_str)
        for match in matches:
            value_str = match.strip()
            if nested_type == "char":
                value = value_str[1:-1]  # Remove the single quotes
            elif nested_type == "String":
                value = value_str[1:-1]  # Remove the double quotes
            else:
                value = (_bfcl_java_type_converter(value_str, nested_type)
                         if nested_type else _bfcl_parse_java_value(value_str))
            elements.append(value)
        return elements

    match_empty = re.search(r"new\s+ArrayList<\w*>\(\)", input_str)
    if match_empty:
        return []  # Return an empty list for an empty ArrayList

    return input_str  # default to string


def _bfcl_parse_array(input_str, nested_type=None):
    match = re.search(r"new\s+\w+\[\]\s*\{(.*?)\}", input_str)
    if match:
        elements_str = match.group(1)
        if nested_type:
            return [_bfcl_java_type_converter(x.strip(), nested_type)
                    for x in elements_str.split(",") if x.strip()]
        return [_bfcl_parse_java_value(x.strip()) for x in elements_str.split(",") if x.strip()]
    return input_str  # default to string


def _bfcl_parse_hashmap(input_str):
    elements = {}
    match = re.search(r"new\s+HashMap<.*?>\s*\(\)\s*\{\s*\{?\s*(.*?)\s*\}?\s*\}", input_str, re.DOTALL)
    if match:
        puts_str = match.group(1)
        if puts_str.strip():
            matches = re.findall(r"put\(\"(.*?)\",\s*(.*?)\)", puts_str)
            for match in matches:
                elements[match[0]] = _bfcl_parse_java_value(match[1].strip())
        return elements

    match_empty = re.search(r"new\s+HashMap<.*?>\s*\(\)", input_str)
    if match_empty:
        return {}  # Return an empty dictionary for an empty HashMap

    return input_str  # default to string


def _bfcl_parse_java_value(value_str):
    """Parse a Java literal without knowing its declared type."""
    if value_str == "true":
        return True
    elif value_str == "false":
        return False
    elif value_str.startswith('"') and value_str.endswith('"'):
        return value_str[1:-1]
    elif re.match(r"^-?\d+[lL]$", value_str):
        return int(value_str[:-1])
    elif re.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?[fF]$", value_str):
        return float(re.sub(r"[fF]$", "", value_str))
    else:
        try:
            return int(value_str)
        except ValueError:
            try:
                return float(value_str)
            except ValueError:
                return value_str  # all other types are kept as strings


# ---- ast_eval/type_convertor/js_type_converter.py -----------------------------------------

def _bfcl_js_type_converter(value, expected_type, nested_type=None):
    if expected_type not in _BFCL_JS_TYPE_CONVERSION:
        raise ValueError(f"Unsupported type: {expected_type}")

    if expected_type == "String":
        if not (value.startswith('"') and value.endswith('"')) and not (
                value.startswith("'") and value.endswith("'")):
            return str(value)
        return value[1:-1]
    elif expected_type == "integer":
        if not re.match(r"^-?\d+$", value):
            return str(value)  # default to string
        return int(value)
    elif expected_type == "float":
        if not re.match(r"^-?\d+(\.\d+)?$", value):
            return str(value)  # default to string
        return float(value)
    elif expected_type == "Bigint":
        if not re.match(r"^-?\d+n$", value):
            return str(value)  # default to string
        return int(value[:-1])
    elif expected_type == "Boolean":
        if value not in ["true", "false"]:
            return str(value)  # default to string
        return value == "true"
    elif expected_type == "dict":
        return _bfcl_parse_js_collection(value, "dict", nested_type)
    elif expected_type == "array":
        return _bfcl_parse_js_collection(value, "array", nested_type)
    elif expected_type == "any":
        return str(value)
    else:
        raise ValueError(f"Unsupported type: {expected_type}")


def _bfcl_parse_js_collection(code, type_str, nested_type=None):
    code = code.strip()
    if type_str == "array":
        array_2d_pattern = (r"\[\s*\[.*?\]\s*(,\s*\[.*?\]\s*)*\]|"
                            r"\bnew\s+Array\(\s*\[.*?\]\s*(,\s*\[.*?\]\s*)*\)")
        array_pattern = r"\[(.*?)\]|\bnew\s+Array\((.*?)\)"

        array_2d_match = re.match(array_2d_pattern, code)
        try:
            if array_2d_match:
                elements_str = array_2d_match.group(0)
                inner_arrays = re.findall(r"\[(.*?)\]", elements_str)
                elements = []
                for idx, inner_array_str in enumerate(inner_arrays):
                    inner_array_str = inner_array_str.strip()
                    if idx == 0 and inner_array_str.startswith("["):
                        inner_array_str = inner_array_str[1:]
                    inner_array_elements = [e.strip() for e in inner_array_str.split(",")]
                    elements.append([_bfcl_parse_js_value(e) for e in inner_array_elements])
                return elements

            array_match = re.match(array_pattern, code)
            if array_match:
                if array_match.group(1) is not None:
                    elements_str = array_match.group(1).strip()
                    elements = elements_str.split(",") if elements_str else []
                elif array_match.group(2) is not None:
                    elements_str = array_match.group(2).strip()
                    elements = elements_str.split(",") if elements_str else []
                else:
                    elements = []
                if nested_type:
                    elements = [
                        (_bfcl_js_type_converter(e.strip(), nested_type, "String")
                         if (e.strip().startswith("'") or e.strip().startswith('"'))
                         else _bfcl_js_type_converter(e.strip(), nested_type))
                        for e in elements
                    ]
                else:
                    elements = [_bfcl_parse_js_value(e.strip()) for e in elements]
                return elements
            return code
        except Exception:
            return code

    elif type_str == "dict":
        if code == "{}":
            return {}  # Return an empty dictionary for an empty object
        dict_match = re.match(r"\{(.*?)\}", code)
        if dict_match:
            try:
                content = dict_match.group(1)
                pairs = re.findall(r"([^:]+):\s*(.*?)(?:,\s*(?=[^,]+:)|$)", content)
                dictionary = {}
                for key, value in pairs:
                    key = key.strip().strip("'\"")
                    value = value.strip()
                    if value.startswith("[") and value.endswith("]"):
                        dictionary[key] = _bfcl_parse_js_collection(value, "array")
                    elif value.startswith("{") and value.endswith("}"):
                        dictionary[key] = _bfcl_parse_js_collection(value, "dict")
                    else:
                        dictionary[key] = _bfcl_parse_js_value(value.strip("'\""))
                return dictionary
            except Exception:
                return code
        return code  # default to string
    else:
        raise ValueError(f"Unsupported type: {type_str}")


def _bfcl_parse_js_value(value_str):
    value_str = value_str.strip()
    if value_str == "true":
        return True
    elif value_str == "false":
        return False
    elif (value_str.startswith('"') and value_str.endswith('"')) or (
            value_str.startswith("'") and value_str.endswith("'")):
        return value_str[1:-1]
    else:
        try:
            return int(value_str)
        except ValueError:
            try:
                return float(value_str)
            except ValueError:
                return value_str


# ---- ast_eval/ast_checker.py --------------------------------------------------------------

def bfcl_ast_checker(func_description: List[Dict[str, Any]], model_output: List[Dict[str, Any]],
                     possible_answer: List[Dict[str, Any]], category: str) -> Dict[str, Any]:
    """ast_checker: dispatch on the category, with BFCL's call-count rules.

    `func_description` are the ORIGINAL function docs, `model_output` the decoded calls
    ``[{name: {param: value}}]``, `possible_answer` the category's ground truth.
    """
    language = bfcl_language(category)
    if "parallel" in category:
        return _bfcl_parallel_function_checker_no_order(
            func_description, model_output, possible_answer, language)
    elif "multiple" in category:
        return _bfcl_multiple_function_checker(
            func_description, model_output, possible_answer, language)
    else:
        if len(model_output) != 1:
            return {"valid": False, "error": ["Wrong number of functions."],
                    "error_type": "simple_function_checker:wrong_count"}
        return _bfcl_simple_function_checker(
            func_description[0], model_output[0], possible_answer[0], language)


def _bfcl_find_description(func_descriptions, name):
    if type(func_descriptions) == list:
        for func_description in func_descriptions:
            if func_description["name"] == name:
                return func_description
        return None
    return func_descriptions  # a dict: there is only one function


def _bfcl_get_possible_answer_type(possible_answer: list):
    for answer in possible_answer:
        if answer != "":  # Optional parameter
            return type(answer)
    return None


def _bfcl_convert_func_name(function_name: str) -> str:
    # The name the tool was advertised under (bfcl_convert_to_tools rewrites "." to "_").
    if "." in function_name:
        return re.sub(r"\.", "_", function_name)
    return function_name


def _bfcl_type_checker(param, value, possible_answer, expected_type_description,
                       expected_type_converted, nested_type_converted):
    # Nested types are checked one level deep only, as upstream does.
    result = {"valid": True, "error": [], "is_variable": False, "error_type": "type_error:simple"}

    is_variable = False
    # A variable name may be used instead of an actual value; the type of the possible
    # answer then differs from the declared type.
    possible_answer_type = _bfcl_get_possible_answer_type(possible_answer)
    if possible_answer_type is not None:
        if possible_answer_type != expected_type_converted:
            is_variable = True

    # value is the same type as in the function description
    if type(value) == expected_type_converted:
        if nested_type_converted is None:
            result["is_variable"] = is_variable
            return result
        else:
            for possible_answer_item in possible_answer:
                flag = True  # Each parameter should match to at least one possible answer type.
                if type(possible_answer_item) == list:
                    for value_item in value:
                        checker_result = _bfcl_type_checker(
                            param, value_item, possible_answer_item, str(nested_type_converted),
                            nested_type_converted, None)
                        if not checker_result["valid"]:
                            flag = False
                            break
                if flag:
                    return {"valid": True, "error": [], "is_variable": is_variable}

            result["valid"] = False
            result["error"] = [
                f"Nested type checking failed for parameter {repr(param)}. Expected outer type "
                f"{expected_type_description} with inner type {str(nested_type_converted)}. "
                f"Parameter value: {repr(value)}."
            ]
            result["error_type"] = "type_error:nested"

    # value is not as expected: accept it as a variable when it has the possible answer's type
    possible_answer_type = _bfcl_get_possible_answer_type(possible_answer)
    if possible_answer_type is not None:
        if type(value) == possible_answer_type:
            result["is_variable"] = True
            return result

    result["valid"] = False
    result["error"].append(
        f"Incorrect type for parameter {repr(param)}. Expected type {expected_type_description}, "
        f"got {type(value).__name__}. Parameter value: {repr(value)}.")
    result["error_type"] = "type_error:simple"
    return result


def bfcl_standardize_string(input_string: str) -> str:
    """Drop spaces and ,./-_*^ punctuation, lower-case, and turn ' into " (so that
    'April 1, 2024' and 'April 1 2024' compare equal)."""
    regex_string = r"[ \,\.\/\-\_\*\^]"
    return re.sub(regex_string, "", input_string).lower().replace("'", '"')


def _bfcl_string_checker(param, model_output, possible_answer):
    standardize_possible_answer = []
    standardize_model_output = bfcl_standardize_string(model_output)
    for i in range(len(possible_answer)):
        if type(possible_answer[i]) == str:
            standardize_possible_answer.append(bfcl_standardize_string(possible_answer[i]))

    if standardize_model_output not in standardize_possible_answer:
        return {"valid": False,
                "error": [f"Invalid value for parameter {repr(param)}: {repr(model_output)}. "
                          f"Expected one of {possible_answer}. Case insensitive."],
                "error_type": "value_error:string"}
    return {"valid": True, "error": []}


def _bfcl_list_checker(param, model_output, possible_answer):
    standardize_model_output = list(model_output)
    for i in range(len(standardize_model_output)):
        if type(standardize_model_output[i]) == str:
            standardize_model_output[i] = bfcl_standardize_string(model_output[i])

    standardize_possible_answer = []
    for i in range(len(possible_answer)):
        standardize_possible_answer.append([])
        for j in range(len(possible_answer[i])):
            if type(possible_answer[i][j]) == str:
                standardize_possible_answer[i].append(bfcl_standardize_string(possible_answer[i][j]))
            else:
                standardize_possible_answer[i].append(possible_answer[i][j])

    if standardize_model_output not in standardize_possible_answer:
        return {"valid": False,
                "error": [f"Invalid value for parameter {repr(param)}: {repr(model_output)}. "
                          f"Expected one of {possible_answer}."],
                "error_type": "value_error:list/tuple"}
    return {"valid": True, "error": []}


def _bfcl_dict_checker(param, model_output, possible_answers):
    # Simple dictionaries only; the dataset has no nested ones.
    result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
    for i in range(len(possible_answers)):
        if possible_answers[i] == "":
            continue

        result = {"valid": False, "error": [], "error_type": "dict_checker:unclear"}
        flag = True
        possible_answer = possible_answers[i]

        for key, value in model_output.items():
            if key not in possible_answer:
                result["valid"] = False
                result["error"].append(f"Unexpected dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break

            standardize_value = value
            if type(value) == str:
                standardize_value = bfcl_standardize_string(value)

            standardize_possible_answer = []
            for j in range(len(possible_answer[key])):
                if type(possible_answer[key][j]) == str:
                    standardize_possible_answer.append(bfcl_standardize_string(possible_answer[key][j]))
                else:
                    standardize_possible_answer.append(possible_answer[key][j])

            if standardize_value not in standardize_possible_answer:
                result["valid"] = False
                result["error"].append(
                    f"Invalid value for parameter {repr(key)}: {repr(value)}. "
                    f"Expected one of {standardize_possible_answer}.")
                result["error_type"] = "value_error:dict_value"
                flag = False
                break

        for key, value in possible_answer.items():
            if key not in model_output and "" not in value:
                result["valid"] = False
                result["error"].append(f"Missing dict key parameter: '{key}'.")
                result["error_type"] = "value_error:dict_key"
                flag = False
                break

        if flag:
            return {"valid": True, "error": []}

    return result


def _bfcl_list_dict_checker(param, model_output, possible_answers):
    # The order of the dictionaries must match the order in the possible answer.
    result = {"valid": False, "error": [], "error_type": "list_dict_checker:unclear"}

    for answer_index in range(len(possible_answers)):
        flag = True
        if len(model_output) != len(possible_answers[answer_index]):
            result["valid"] = False
            result["error"] = ["Wrong number of dictionaries in the list."]
            result["error_type"] = "value_error:list_dict_count"
            flag = False
            continue

        for dict_index in range(len(model_output)):
            result = _bfcl_dict_checker(param, model_output[dict_index],
                                        [possible_answers[answer_index][dict_index]])
            if not result["valid"]:
                flag = False
                break
        if flag:
            return {"valid": True, "error": []}

    return result


def _bfcl_simple_function_checker(func_description, model_output, possible_answer, language):
    possible_answer = list(possible_answer.values())[0]
    func_name = func_description["name"]
    param_details = func_description["parameters"]["properties"]
    required_params = func_description["parameters"]["required"]

    result = {"valid": True, "error": [], "error_type": "simple_function_checker:unclear"}

    func_name = _bfcl_convert_func_name(func_name)

    if func_name not in model_output:
        result["valid"] = False
        result["error"].append(f"Function name {repr(func_name)} not found in model output.")
        result["error_type"] = "simple_function_checker:wrong_func_name"
        return result

    model_params = model_output[func_name]

    for param in required_params:
        if param not in model_params:
            result["valid"] = False
            result["error"].append(f"Missing required parameter: {repr(param)}.")
            result["error_type"] = "simple_function_checker:missing_required"
            return result

    for param, value in model_params.items():
        if param not in param_details or param not in possible_answer:
            result["valid"] = False
            result["error"].append(f"Unexpected parameter: {repr(param)}.")
            result["error_type"] = "simple_function_checker:unexpected_param"
            return result

        full_param_details = param_details[param]
        expected_type_description = full_param_details["type"]  # This is a string
        is_variable = False
        nested_type_converted = None

        if language == "java":
            expected_type_converted = _BFCL_JAVA_TYPE_CONVERSION[expected_type_description]
            if expected_type_description in _BFCL_JAVA_TYPE_CONVERSION:
                if type(value) != str:
                    result["valid"] = False
                    result["error"].append(
                        f"Incorrect type for parameter {repr(param)}. Expected type String, got "
                        f"{type(value).__name__}. Parameter value: {repr(value)}.")
                    result["error_type"] = "type_error:java"
                    return result
                if expected_type_description in _BFCL_NESTED_CONVERSION_TYPE_LIST:
                    nested_type = param_details[param]["items"]["type"]
                    nested_type_converted = _BFCL_JAVA_TYPE_CONVERSION[nested_type]
                    value = _bfcl_java_type_converter(value, expected_type_description, nested_type)
                else:
                    value = _bfcl_java_type_converter(value, expected_type_description)

        elif language == "javascript":
            expected_type_converted = _BFCL_JS_TYPE_CONVERSION[expected_type_description]
            if expected_type_description in _BFCL_JS_TYPE_CONVERSION:
                if type(value) != str:
                    result["valid"] = False
                    result["error"].append(
                        f"Incorrect type for parameter {repr(param)}. Expected type String, got "
                        f"{type(value).__name__}. Parameter value: {repr(value)}.")
                    result["error_type"] = "type_error:js"
                    return result
                if expected_type_description in _BFCL_NESTED_CONVERSION_TYPE_LIST:
                    nested_type = param_details[param]["items"]["type"]
                    nested_type_converted = _BFCL_JS_TYPE_CONVERSION[nested_type]
                    value = _bfcl_js_type_converter(value, expected_type_description, nested_type)
                else:
                    value = _bfcl_js_type_converter(value, expected_type_description)

        else:
            expected_type_converted = _BFCL_PYTHON_TYPE_MAPPING[expected_type_description]
            if expected_type_description in _BFCL_PYTHON_NESTED_TYPE_CHECK_LIST:
                nested_type = param_details[param]["items"]["type"]
                nested_type_converted = _BFCL_PYTHON_TYPE_MAPPING[nested_type]

        # A tuple in the possible answer is a list after the JSON round trip.
        if expected_type_description == "tuple" and type(value) == tuple:
            value = list(value)

        # Python auto-converts int to float.
        if language == "python" and expected_type_description == "float" and type(value) == int:
            value = float(value)

        # Only Python needs this: the Java/JS converters already produced the declared type.
        type_check_result = _bfcl_type_checker(
            param, value, possible_answer[param], expected_type_description,
            expected_type_converted, nested_type_converted)
        is_variable = type_check_result["is_variable"]
        if not type_check_result["valid"]:
            return type_check_result

        # A variable is compared as a plain string by the generic check below.
        if not is_variable:
            if expected_type_converted == dict:
                result = _bfcl_dict_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue
            elif expected_type_converted == list and nested_type_converted == dict:
                result = _bfcl_list_dict_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue
            elif expected_type_converted == str:
                # Strings are compared standardised, unless they are a variable.
                result = _bfcl_string_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue
            elif expected_type_converted == list:
                result = _bfcl_list_checker(param, value, possible_answer[param])
                if not result["valid"]:
                    return result
                continue

        if value not in possible_answer[param]:
            result["valid"] = False
            result["error"].append(
                f"Invalid value for parameter {repr(param)}: {repr(value)}. "
                f"Expected one of {possible_answer[param]}.")
            result["error_type"] = "value_error:others"
            return result

    for param in possible_answer:
        if param not in model_params and "" not in possible_answer[param]:
            result["valid"] = False
            result["error"].append(
                f"Optional parameter {repr(param)} not provided and not marked as optional.")
            result["error_type"] = "simple_function_checker:missing_optional"
            return result

    return result


def _bfcl_parallel_function_checker_no_order(func_descriptions, model_output, possible_answers,
                                             language):
    if len(model_output) != len(possible_answers):
        return {"valid": False, "error": ["Wrong number of functions."],
                "error_type": "parallel_function_checker_no_order:wrong_count"}

    matched_indices = []
    # Walk the possible answers one by one and eliminate the model output that matches: the
    # ground truth is what selects the function description.
    for i in range(len(possible_answers)):
        func_name_expected = list(possible_answers[i].keys())[0]
        func_description = _bfcl_find_description(func_descriptions, func_name_expected)

        all_errors = []
        for index in range(len(model_output)):
            if index in matched_indices:
                continue
            result = _bfcl_simple_function_checker(
                func_description, model_output[index], possible_answers[i], language)
            if result["valid"]:
                matched_indices.append(index)
                break
            all_errors.append({
                f"Model Result Index {index}": {
                    "sub_error": result["error"],
                    "sub_error_type": result["error_type"],
                    "model_output_item": model_output[index],
                    "possible_answer_item": possible_answers[i],
                }
            })

        if not result["valid"]:
            considered_indices = [j for j in range(len(model_output)) if j not in matched_indices]
            all_errors.insert(
                0, f"Could not find a matching function among index {considered_indices} of "
                   f"model output for index {i} of possible answers.")
            return {"valid": False, "error": all_errors,
                    "error_type": "parallel_function_checker_no_order:cannot_find_match"}

    return {"valid": True, "error": []}


def _bfcl_multiple_function_checker(func_descriptions, model_output, possible_answers, language):
    if len(model_output) != len(possible_answers):
        return {"valid": False, "error": ["Wrong number of functions."],
                "error_type": "multiple_function_checker:wrong_count"}

    # possible_answers is a list of one dictionary with one key
    func_name_expected = list(possible_answers[0].keys())[0]
    func_description = _bfcl_find_description(func_descriptions, func_name_expected)
    return _bfcl_simple_function_checker(
        func_description, model_output[0], possible_answers[0], language)


# ---- eval_runner.py: one reply -> one verdict ---------------------------------------------

def bfcl_score(category: str, functions: List[Dict[str, Any]], ground_truth: Any,
               tool_calls: Any = None, response_text: Optional[str] = "") -> Dict[str, Any]:
    """BFCL verdict for one single-turn FC reply: ``{"valid", "error", "error_type"}``.

    Mirrors bfcl_eval's evaluate_task for an FC model. Relevance/irrelevance categories ask
    only whether the reply decodes to a non-empty call (an empty reply, a refusal or an
    undecodable call is "no call"); every other category decodes the calls and runs
    ast_checker against the ORIGINAL function docs, with BFCL's call-count, type and
    string-standardisation rules.

    `tool_calls` is the OpenAI tool-call array of the reply. Reply text is never decoded, as
    in BFCL's FC handler (OpenAICompletionsHandler.decode_ast): with no tool calls an empty
    reply is an empty call list and any other text fails to decode (no call). That is what
    upstream does with a reply that carries no `tool_calls` field, which is how OpenAI and
    current vLLM answer when no call is made (_parse_query_response_FC then reads the text).
    """
    decode_error = None
    decoded: Any = []
    if tool_calls:
        try:
            decoded = bfcl_decode_fc(tool_calls)
        except Exception as e:
            decode_error = str(e)
    elif (response_text or "").strip():
        decode_error = "the reply has no tool calls (BFCL's FC handler reads its text as no call)"

    if "relevance" in category:          # utils.is_relevance_or_irrelevance
        contain_func_call = decode_error is None and not bfcl_is_empty_output(decoded)
        if "irrelevance" in category:
            if not contain_func_call:
                return {"valid": True, "error": [], "error_type": ""}
            return {"valid": False,
                    "error": ["Valid syntax. Successfully decode AST when it should not."],
                    "error_type": "irrelevance_error:decoder_success"}
        if contain_func_call:
            return {"valid": True, "error": [], "error_type": ""}
        return {"valid": False,
                "error": [f"Invalid syntax. Failed to decode AST when it should have. {decode_error}"],
                "error_type": "relevance_error:decoder_failed"}

    if decode_error is not None:
        return {"valid": False, "error": [f"Invalid syntax. Failed to decode AST. {decode_error}"],
                "error_type": "ast_decoder:decoder_failed"}
    if not bfcl_is_function_calling_format_output(decoded):
        return {"valid": False, "error": ["Did not output in the specified format."],
                "error_type": "ast_decoder:decoder_wrong_output_format"}
    try:
        return bfcl_ast_checker(functions, decoded, ground_truth, category)
    except Exception as e:
        # Upstream would abort the whole evaluation on these (e.g. a list-of-dict parameter
        # answered with a list of strings under an optional possible answer); one malformed
        # reply is an invalid call, not a reason to lose the run.
        return {"valid": False, "error": [f"ast_checker raised {type(e).__name__}: {e}"],
                "error_type": "ast_checker:exception"}
