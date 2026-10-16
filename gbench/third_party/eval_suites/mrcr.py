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

"""Native MRCR (Needle in a Haystack / Long-Context Retrieval) evaluation suite.

CONTEXT WINDOW: openai/mrcr spans 8 length bins up to 1M tokens. As in the upstream reference
loop (`if n_tokens(messages) > MAX_CONTEXT_WINDOW: continue`), rows whose prompt does not fit
the served window are SKIPPED rather than scored 0, and the evaluated window is recorded. A
smaller window therefore evaluates fewer bins; it does not lower the score. Results are also
reported per (n_needles, length bin), which is how published MRCR numbers are stated.

Sampling: the upstream reference loop (the "How to run" section of the openai/mrcr dataset card
at _REVISION) calls `client.chat.completions.create(model=MODEL, messages=messages)` with no
temperature and no max_tokens (OFFICIAL_SAMPLING, recorded). gbench sends no max_tokens either,
in both modes (the room left in the context window), but always sends the temperature of its
rule (base.resolve_temperature): 1.0 with `--thinking`, 0.0 without, which
`leaderboard_comparable_reason` names. The per-row fit rule above
still decides which rows are sent. `--temperature`, or `GBENCH_MRCR_TEMPERATURE` for this suite
alone (which wins), sends another temperature instead; the result records it under
`official_sampling` and names it in `leaderboard_comparable_reason`.
"""

import csv
import difflib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from . import base
from .limit_plan import base_attempts, eval_limit_plan
from .base import OfficialSampling, apply_shard, finalize_shard_fields, run_eval_suite
from .metrics import finalize_mean_metric
from .sampling import presharded_for_base
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/mrcr.md"

# openai/mrcr bins rows by prompt + answer length in o200k_base tokens:
# [4096, 8192], (8192, 16384], ..., (524288, 1048576].
_BIN_EDGES = [4096 * 2 ** i for i in range(9)]
#: The dataset revision the cell totals below were measured on; the loader pins it.
_REVISION = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"
#: Rows per (n_needles, bin) cell at _REVISION, from binning all 2,400 rows with o200k_base
#: (prompt + answer). Nominally 100 each; the small bins hold 96-106. A cell with fewer rows
#: inside the window than this is only partly evaluated.
_CELL_ROWS = {
    "2needle/4k-8k": 101, "2needle/8k-16k": 100, "2needle/16k-32k": 102,
    "2needle/32k-64k": 97, "2needle/64k-128k": 100, "2needle/128k-256k": 100,
    "2needle/256k-512k": 100, "2needle/512k-1024k": 100,
    "4needle/4k-8k": 103, "4needle/8k-16k": 100, "4needle/16k-32k": 99,
    "4needle/32k-64k": 98, "4needle/64k-128k": 100, "4needle/128k-256k": 100,
    "4needle/256k-512k": 100, "4needle/512k-1024k": 100,
    "8needle/4k-8k": 106, "8needle/8k-16k": 96, "8needle/16k-32k": 98,
    "8needle/32k-64k": 100, "8needle/64k-128k": 100, "8needle/128k-256k": 100,
    "8needle/256k-512k": 100, "8needle/512k-1024k": 100,
}
#: Upstream's sampling: the dataset card's reference loop sends neither a temperature nor a
#: max_tokens.
OFFICIAL_SAMPLING = OfficialSampling(
    reference=(f"openai/mrcr@{_REVISION[:8]} dataset card README 'How to run': "
               "client.chat.completions.create(model=MODEL, messages=messages), no temperature, "
               "no max_tokens"),
    send_temperature=False,
    send_max_tokens=False,
)
# Room the prompt must leave at minimum: the answer margin + template slack
# base.clamp_to_context uses. Each row also needs room for its own answer (_answer_room).
_ANSWER_ROOM = 256 + 64


def _bin_label(n_tokens: int) -> str:
    for lo, hi in zip(_BIN_EDGES, _BIN_EDGES[1:]):
        if n_tokens <= hi:
            return f"{lo // 1024}k-{hi // 1024}k"
    return f">{_BIN_EDGES[-1] // 1024}k"


def _o200k_encoder():
    """Upstream's bin tokenizer (tiktoken o200k_base), or None when unavailable."""
    try:
        import tiktoken
        return tiktoken.get_encoding("o200k_base")
    except Exception as e:
        logger.warning("mrcr: tiktoken o200k_base unavailable (%s); length bins are "
                       "estimated at 4 chars/token and recorded as such. See %s", e, DOCS_URL)
        return None


def _bin_tokens(enc, messages: List[Dict[str, Any]], answer: str, n_chars: int) -> int:
    """Upstream bin size: prompt + answer tokens (o200k_base), or a 4 chars/token estimate."""
    if enc is None:
        return int((n_chars + len(answer)) / 4)
    enc_len = lambda t: len(enc.encode(str(t or ""), disallowed_special=()))
    return sum(enc_len((m or {}).get("content")) for m in messages) + enc_len(answer)


def _parse_prompt(prompt_raw: Any) -> List[Dict[str, Any]]:
    import json
    if isinstance(prompt_raw, str) and prompt_raw.strip().startswith("["):
        try:
            return json.loads(prompt_raw)
        except Exception:
            pass
    return [{"role": "user", "content": str(prompt_raw)}]


def _answer_room(answer_tokens: int) -> int:
    """Room a row needs to answer: its gold answer twice over (the model reproduces a whole
    message, after the required prefix) plus 512 tokens, never less than _ANSWER_ROOM. A flat
    320 kept edge rows whose answers could not fit, which then scored near 0."""
    return max(_ANSWER_ROOM, 2 * int(answer_tokens) + 512)


def _fits_window(get_prompt, n_chars: int, n_msgs: int, window: int,
                 room: int = _ANSWER_ROOM) -> Tuple[bool, str]:
    """Does the prompt leave room to answer in ``window``? Returns (fits, how it was decided).

    Cheap bounds settle most rows without tokenizing: over even at 5 chars/token (template
    aside), or under even at 3 chars/token plus the chat-template allowance (MRCR's text runs
    3.6-5.3 chars per Gemma token). Borderline rows are counted exactly (the server's
    /tokenize, else the run's tokenizer, via base._prompt_token_bounds). With only a
    character estimate, the decision is conservative: a row that reaches the server over
    the window scores 0, which is exactly what skipping it is meant to prevent.
    """
    budget = window - room
    template = (base._TEMPLATE_TOKENS_PER_REQUEST
                + n_msgs * base._TEMPLATE_TOKENS_PER_MESSAGE)
    if n_chars / 5.0 > budget:
        return False, "bounds"
    if n_chars / 3.0 + template <= budget:
        return True, "bounds"
    bounds = base._prompt_token_bounds(_parse_prompt(get_prompt()))
    if bounds is None:
        return False, "estimate"
    low, high, mode = bounds
    return (high <= budget if mode == "estimate" else low <= budget), mode


def _load_mrcr_samples(limit: Optional[int] = None, window: Optional[int] = None,
                       stats: Optional[Dict[str, Any]] = None,
                       ) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load MRCR samples from the canonical HF Hub dataset ('openai/mrcr').

    With a known ``window``, rows that do not fit are dropped BEFORE the --eval-limit subset
    is drawn (so a subset is never mostly unanswerable rows), and the subset is stratified
    over (n_needles, length bin). ``stats`` receives the counts and per-cell coverage.
    """
    from collections import Counter
    from datasets import load_dataset
    from .sampling import stratified_indices

    ds = load_dataset("openai/mrcr", split="train", revision=_REVISION)
    total = len(ds)
    n_chars, n_msgs = ds["n_chars"], ds["total_messages"]
    enc = _o200k_encoder()
    answers = ds["answer"]
    rooms = [_answer_room(len(enc.encode(a, disallowed_special=())) if enc else len(a) / 4)
             for a in answers]
    decided = Counter()
    keep = []
    if window:
        # Borderline rows are counted exactly (server /tokenize or tokenizer), 8 at a time.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=8) as pool:
            verdicts = list(pool.map(
                lambda i: _fits_window(lambda: ds[i]["prompt"], n_chars[i], n_msgs[i], window,
                                       rooms[i]), range(total)))
        for i, (fits, how) in enumerate(verdicts):
            decided[how] += 1
            if fits:
                keep.append(i)
    else:
        keep = list(range(total))

    rows = []                                   # (dataset index, category, parsed messages)
    for i in keep:
        item = ds[i]
        messages = _parse_prompt(item["prompt"])
        n_tok = _bin_tokens(enc, messages, str(item.get("answer", "")), n_chars[i])
        rows.append((i, f"{item.get('n_needles', 8)}needle/{_bin_label(n_tok)}", messages))

    cells = Counter(cat for _, cat, _ in rows)
    if stats is not None:
        stats.update(
            total_rows=total, rows_in_window=len(rows),
            rows_skipped_over_window=total - len(rows),
            window_decided_by=dict(decided),
            bins_tokenizer="o200k_base" if enc is not None else "chars/4 estimate",
            # Cells not fully inside the window: compare them to published figures with care.
            # Only meaningful with upstream's tokenizer: estimated bins misplace rows, so no
            # partial-bin claim is made from them.
            partial_bins=({cat: f"{n}/{_CELL_ROWS[cat]}" for cat, n in sorted(cells.items())
                           if cat in _CELL_ROWS and n < _CELL_ROWS[cat]}
                          if enc is not None else None))

    # --shard first, over the full ordered set of rows that fit the window, then the --eval-limit
    # draw within the shard.
    rows, _ = apply_shard(rows, "mrcr")
    # Stratified over (n_needles, bin), not a contiguous head.
    if limit and limit > 0 and len(rows) > limit:
        rows = [rows[j] for j in stratified_indices([cat for _, cat, _ in rows], limit,
                                                    seed="mrcr")]
    samples = []
    for i, category, messages in rows:
        item = ds[i]
        ans = str(item.get("answer", ""))
        canary = str(item.get("random_string_to_prepend", "") or "")
        # The canary is half the metric (the response must START with it), so it has to
        # reach the scorer. It used to be recorded in the sample metadata and never read.
        samples.append((messages, {"answer": ans, "canary": canary},
                        {"category": category, "canary": canary}))

    logger.info(f"Loaded {len(samples)} MRCR samples from HF Hub ('openai/mrcr').")
    return samples


def grade_mrcr(response_text: str, gold_target: Any) -> float:
    """Canonical MRCR grade in [0, 1] (openai/mrcr `grade`).

        if not response.startswith(random_string_to_prepend): return 0
        response = response.removeprefix(random_string_to_prepend)
        answer   = answer.removeprefix(random_string_to_prepend)
        return SequenceMatcher(None, response, answer).ratio()

    Two things the previous implementation did not do: it never applied the canary gate
    (the dataset's `random_string_to_prepend` was loaded into the sample metadata and left
    unread), and it graded with case-insensitive containment instead of the sequence
    ratio - so a response that merely contained the needle anywhere scored the same as a
    verbatim reproduction, and a response that ignored the required prefix was not
    penalised at all.
    """
    if isinstance(gold_target, dict):
        answer = str(gold_target.get("answer") or "")
        canary = str(gold_target.get("canary") or "")
    else:
        answer, canary = str(gold_target or ""), ""
    response = str(response_text or "")
    if not response or not answer:
        return 0.0
    if canary:
        if not response.startswith(canary):
            return 0.0
        response = response[len(canary):]
        if answer.startswith(canary):
            answer = answer[len(canary):]
    # Canonical openai/mrcr grade() uses SequenceMatcher's default (autojunk=True); forcing
    # autojunk=False changed the ratio on the long, block-duplicated MRCR strings and made
    # the number non-comparable to published MRCR.
    return float(difflib.SequenceMatcher(None, response, answer).ratio())


def _eval_mrcr(response_text: str, gold_target: Any) -> bool:
    """Harness pass/fail (exact_match_rate, not an upstream metric): a verbatim reproduction,
    i.e. the reply starts with the canary and, with the canary stripped from both sides, equals
    the answer character for character. A near-copy (a grade of 0.99 that differs by a missing
    sign-off or added markup) is not one. The reported metric is the mean grade."""
    if isinstance(gold_target, dict):
        answer = str(gold_target.get("answer") or "")
        canary = str(gold_target.get("canary") or "")
    else:
        answer, canary = str(gold_target or ""), ""
    response = str(response_text or "")
    if not response or not answer or not response.startswith(canary):
        return False
    if canary:
        response = response[len(canary):]
        if answer.startswith(canary):
            answer = answer[len(canary):]
    return response == answer


def _get_server_max_model_len(base_url: str, model_name: Optional[str] = None) -> Optional[int]:
    """The served context window (max_model_len) of the model under test, from the endpoint's
    GET /v1/models: the entry whose id is `model_name` (config.match_served_model: exact,
    case-insensitive or by the last path component), never the first entry of an endpoint that
    serves several models. Without `model_name`, the only entry of a one-model endpoint. None when
    the listing fails, lists no such model, or reports no max_model_len for it."""
    import json
    import urllib.request
    from ...core.config import match_served_model

    url = f"{base_url.rstrip('/')}/models"
    try:
        req = urllib.request.Request(url, headers=base.endpoint_auth_headers())
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.debug(f"Could not query max_model_len from {base_url}: {e}")
        return None
    models = [m for m in (data.get("data") or []) if isinstance(m, dict) and m.get("id")]
    served_id, how = match_served_model([str(m["id"]) for m in models],
                                        [model_name] if model_name else [])
    if served_id is None:
        logger.warning("mrcr: no context window taken from %s: %s", url, how)
        return None
    entry = next(m for m in models if str(m["id"]) == served_id)
    if entry.get("max_model_len") is None:
        return None
    return int(entry["max_model_len"])


def run_mrcr(model_name: str, base_url: str, concurrency: int, enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run native MRCR long-context evaluation suite."""
    raw = os.environ.get("GBENCH_MRCR_WINDOW", "").strip()
    try:
        override = int(raw) if raw else None
    except ValueError:
        raise infra_required("mrcr", f"GBENCH_MRCR_WINDOW={raw!r} is not a number of tokens.",
                             DOCS_URL)
    # An explicit override wins; otherwise the window of the server being evaluated.
    window = (override or base.get_run_knob("max_model_len")
              or _get_server_max_model_len(base_url, model_name) or None)
    if not window:
        # Sending every row blind would score each over-window row 0 (the rows run up to
        # 1M tokens); upstream skips them, which needs the window.
        raise infra_required(
            "mrcr", f"the server does not report the context window of {model_name!r} (GET "
            "/v1/models lists no max_model_len for it), so rows that do not fit cannot be "
            "skipped. Set GBENCH_MRCR_WINDOW to the served window in tokens (e.g. 131072).",
            DOCS_URL)
    logger.info(f"mrcr: evaluating rows that fit the served {window:,}-token window")

    stats: Dict[str, Any] = {}
    samples = _load_mrcr_samples(limit=kwargs.get("limit"), window=window, stats=stats)
    # No row fits the window: an error. Rows that fit but none on this node (an empty shard of a
    # --shard I/N run): base's honest empty row.
    if not samples and not (stats.get("rows_in_window") and base.current_shard()):
        raise infra_required(
            "mrcr", f"no MRCR row fits the {window:,}-token window (the shortest rows need "
            f"about 5K tokens). Serve a larger context window.", DOCS_URL)
    eval_plan = eval_limit_plan(
        "mrcr", kwargs.get("limit"), len(samples), attempts=base_attempts())
    result = run_eval_suite(
        eval_name="mrcr",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's rows, kept as is
        eval_fn=_eval_mrcr,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=OFFICIAL_SAMPLING,   # no max_tokens, as upstream (temperature: the rule)
    )
    result["eval_limit_plan"] = eval_plan
    # MRCR's canonical score is the AVERAGE sequence-match ratio, not a pass rate. An empty
    # response is a 0, NOT a sample to drop from the mean - dropping the hard ones silently
    # inflated the score. A request that got no reply from the served endpoint is a harness
    # error (base marks it scoring_excluded): left out of the mean and of mrcr_bins, counted and
    # named in status_reason. (Rows over the window never reach this point: they were skipped up
    # front, as upstream does.)
    result = finalize_mean_metric(
        result,
        metric_name=("mean SequenceMatcher ratio after the canary-prefix gate (canonical MRCR "
                     "grade), pooled over n_needles and the length bins that fit the window"),
        scorer=lambda t: grade_mrcr(t.get("response_text"), t.get("gold_answer")),
        stash_key="mrcr_grade",
        secondary_key="exact_match_rate",
        failure_score=0.0,
    )
    breakdown = _mrcr_breakdown(result.get("sample_traces") or [])
    result.update(breakdown)
    # The per-category table must show the canonical grade, not the exact-match pass rate
    # base.py tallies, or it contradicts mrcr_bins next to it.
    for needles, per in breakdown["mrcr_bins"].items():
        for bin_, mean in per.items():
            cell = (result.get("category_accuracy") or {}).get(f"{needles}/{bin_}")
            if isinstance(cell, dict):
                cell["exact_match_accuracy"] = cell.get("accuracy")
                cell["accuracy"] = mean
    result["mrcr_window_tokens"] = window
    result.update({f"mrcr_{k}": v for k, v in stats.items()})
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(
        _comparability_reasons(result, window=window, limit=kwargs.get("limit")))
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)


def _comparability_reasons(result: Dict[str, Any], *, window: int,
                           limit: Optional[int]) -> List[str]:
    """Why this MRCR result is not a published MRCR number: the pooled headline always, plus an
    --eval-limit subset, a temperature upstream does not send, and requests with no reply."""
    reasons = [
        "the headline pools n_needles and the length bins that fit the served "
        f"{window:,}-token window; published MRCR numbers are per (n_needles, length bin), "
        "see mrcr_bins (and mrcr_partial_bins for cells only partly inside the window)"]
    traces = result.get("sample_traces") or []
    if limit:
        reasons.append(f"--eval-limit subset ({len(traces)} of the "
                       f"{result.get('mrcr_rows_in_window', 'in-window')} rows that fit the "
                       "window)")
    if (result.get("official_sampling") or {}).get("temperature_matches_official") is False:
        reasons.append(f"temperature {result.get('temperature')} "
                       f"({result.get('temperature_source')}; upstream's reference loop sends "
                       "no temperature)")
    no_reply = sum(1 for t in traces if t.get("harness_error"))
    if no_reply:
        reasons.append(f"{no_reply} request(s) got no reply from the served endpoint (harness "
                       "errors, left out of the mean and mrcr_bins)")
    return reasons


def _mrcr_breakdown(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean grade (x100) per n_needles and length bin, from each trace's category, over the
    traces the headline mean counts: a request that got no reply from the served endpoint is a
    harness error that base marks scoring_excluded, left out as in finalize_mean_metric."""
    from .base import is_scored
    acc: Dict[str, Dict[str, List[float]]] = {}
    for t in traces:
        if not is_scored(t):
            continue
        cat, grade = str(t.get("category") or ""), t.get("mrcr_grade")
        if grade is None and t.get("response_text") is None:
            grade = 0.0      # a reply with no text counts as 0, exactly as in the headline mean
        if "/" not in cat or grade is None:
            continue
        needles, bin_ = cat.split("/", 1)
        acc.setdefault(needles, {}).setdefault(bin_, []).append(float(grade))
    order = {_bin_label(hi): i for i, hi in enumerate(_BIN_EDGES[1:])}
    bins = {n: {b: round(100.0 * sum(v) / len(v), 2)
                for b, v in sorted(per.items(), key=lambda kv: order.get(kv[0], 99))}
            for n, per in sorted(acc.items())}
    counts = {n: {b: len(v) for b, v in per.items()} for n, per in sorted(acc.items())}
    return {"mrcr_bins": bins, "mrcr_bin_counts": counts}
