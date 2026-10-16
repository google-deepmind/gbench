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

# Evaluation Suite: beam_128k
# Description: BEAM (Beyond a Million Tokens: Long-Term Memory in LLMs) - the 128K tier (HF split "100K"): 20 conversations, 400 probing questions over 10 memory abilities

"""gbench native built-in runner for beam_128k (Long Context & Retrieval).

BEAM (Tavakoli et al., ICLR 2026; github.com/mohammadtavakoli78/BEAM), long-context
protocol, 128K tier (the HF split is named "100K"): 20 conversations with 20 probing
questions each, two per memory ability. Ported from the pinned upstream commit:
  * the conversation goes in as real chat turns (every session's messages, in order),
    followed by upstream's final user turn ("NOTE: Only provide the answer ..."). As in
    upstream's harness, from a conversation's third question on the prompt also carries the
    question turns of its questions 2..k-1 (in answer_generation.py's order), before its own:
    probing_question_evaluation appends each question turn to the saved history list it reuses
    by reference (_question_tail);
  * a conversation longer than the served window is pruned from the head, keeping an even
    number of the most recent messages (upstream prune_from_tail). The pruned prompt leaves
    upstream's 2,000 tokens for the answer; with --thinking it leaves the run's budget
    (--max-output-tokens, else the thinking floor), which the token-limit policy never cuts;
  * every rubric item is judged on its own with upstream's unified_llm_judge_base_prompt
    (the judge sees only the item and the response) and the 0 / 0.5 / 1 item scores are
    averaged;
  * event_ordering is scored by tau_norm: response lines are aligned to the rubric events with
    upstream's LLM equivalence check, then Kendall tau-b, normalised to [0, 1];
  * the headline is the unweighted mean of the 10 per-ability means (the paper's Average row);
  * a failed request (no response: timeout, over-context, server error) is a harness error: it
    is left out of the ability means and the pass rate, counted in `harness_errors` and named in
    leaderboard_comparable_reason, never scored as the model's answer. Upstream's
    answer_generation.py catches the exception and records the text "Error in <ability> question
    type and <question> and for <file>" as the reply, which its judge then scores. A judge that
    gives no usable score in _JUDGE_ATTEMPTS replies (upstream raises) is a harness error too. A
    run with no question scored withholds its score (accuracy null, status failed);
  * the rubric-item judge and the event_ordering split receive the reply as upstream's do, with
    only inline reasoning blocks removed (no whitespace stripped); every trace keeps each judge
    reply with the Gemini model that gave it (`beam_judge_calls`) and, for event_ordering, each
    alignment check (`beam_alignment_calls`).
The judge is the gbench Gemini cascade instead of upstream's gpt-4.1-mini, so the result carries
leaderboard_comparable=False.

Sampling: upstream's answer generation runs at temperature 0 (answer_generation.py
`--temperature` default and model_config_default), recorded (base.OfficialSampling); the request
follows gbench's rule (base.resolve_temperature): 1.0 with `--thinking`, 0.0 without. Its reader
model
(src/llm.py BuildLLm, a
langchain ChatOpenAI, or init_chat_model) is given no max_tokens, so no request carries one: each
answer may use what the window leaves after the (pruned) prompt, and `--max-output-tokens` is not
sent (`max_output_tokens_note`; with --thinking it sizes the answer room pruning keeps). Override
the temperature for a whole run with `--temperature`, or for this suite alone with
`GBENCH_BEAM_128K_TEMPERATURE`, which takes precedence over both; the result records
`official_sampling`. The judge calls follow src/llm.py's gpt-4.1-mini BuildLLm: temperature 0 and
no max_tokens (`judge_sampling`); none of the settings above changes them.
"""

import ast
import asyncio
import json
import logging
import math
import os
import re
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from . import base
from .limit_plan import base_attempts, eval_limit_plan
from .base import (apply_shard, finalize_shard_fields, run_eval_suite, gemini_api_keys,
                   gemini_required_skip, get_run_knob, judge_cascade, judge_config,
                   judge_generate_cascade, strip_thinking_tags, OfficialSampling)
from .metrics import record_budget_cutoffs
from .sampling import stratified_sample, presharded_for_base

logger = logging.getLogger(__name__)

PILLAR = "Long Context & Retrieval"
DOCS_URL = "docs/evals/beam_128k.md"

_DATASET = "Mohammadta/BEAM"
_REVISION = "3205395e897e7318c7b094ef4e6047b9b82dbb03"
_PARQUET = "data/100K-00000-of-00001.parquet"
UPSTREAM = "github.com/mohammadtavakoli78/BEAM"
UPSTREAM_COMMIT = "b2da22eac88bb0874c64665f13457eb99835774a"
OFFICIAL_JUDGE = "gpt-4.1-mini at temperature 0"
#: src/answer_probing_questions/answer_generation.py @ UPSTREAM_COMMIT: `--temperature` defaults to
#: 0 and model_config_default sets temperature 0 for the long-context model; the reader model is
#: built without a max_tokens (src/llm.py BuildLLm.build_llm, long_term_memory_methods
#: init_chat_model), so its requests carry none.
OFFICIAL_SAMPLING = OfficialSampling(
    reference=(f"{UPSTREAM} src/answer_probing_questions/answer_generation.py "
               f"@{UPSTREAM_COMMIT[:8]} (temperature 0; the reader is sent no max_tokens)"),
    temperature=0.0, send_max_tokens=False)
#: prune_from_tail keeps `max_tokens - 2000` tokens of history: 2,000 tokens for the answer.
UPSTREAM_ANSWER_RESERVE = 2000
#: src/llm.py @ UPSTREAM_COMMIT: the judge is BuildLLm(model_name="gpt-4.1-mini", temperature=0), a
#: ChatOpenAI with no max_tokens.
JUDGE_REFERENCE = (f"{UPSTREAM} src/llm.py @{UPSTREAM_COMMIT[:8]}: BuildLLm(model_name="
                   "'gpt-4.1-mini', temperature=0), no max_tokens")
WINDOW_ENV = "GBENCH_BEAM_128K_WINDOW"

#: The ten memory abilities (report_results.py column order).
ABILITIES = ("abstention", "contradiction_resolution", "event_ordering", "information_extraction",
             "instruction_following", "knowledge_update", "multi_session_reasoning",
             "preference_following", "summarization", "temporal_reasoning")
#: Where each ability keeps its reference answer (kept on the trace for reading; the canonical
#: judge scores against the rubric only).
_IDEAL_KEYS = ("ideal_response", "ideal_answer", "answer", "ideal_summary", "expected_compliance")
#: A judge reply that still has no usable score after upstream's parse and json_repair fallback
#: is asked again this many times in total before the question is marked JUDGE_FAILED (upstream
#: raises there).
_JUDGE_ATTEMPTS = 3

# long_term_memory_methods.probing_question_evaluation: the final user turn, verbatim (leading
# newline, 12-space indents and the space after "explanations." included).
FINAL_TURN = ("\n"
              "            NOTE: Only provide the answer without any explanations. \n"
              "            Question: {query}")

# src/prompts.py unified_llm_judge_base_prompt, verbatim (the curly quotes are upstream's).
JUDGE_PROMPT = """
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- RUBRIC CRITERION (what to check): <rubric_item>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate. 

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT
A compliant response must be **on-topic** and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., \u201c$68,000\u201d, \u201c68k\u201d, \u201c68,000 USD\u201d, or \u201csixty-eight thousand dollars\u201d). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., \u201citemized list\u201d, \u201cno citations\u201d, \u201cone sentence\u201d).
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.
  
- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.
  
- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)
   
3. **Check Compliance**: 
   - For positive requirements: Look for the presence and quality of the required element
   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.

5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"
}

NOTE: ONLY output the json object, without any explanation before or after that
"""

# compute_metrics.llm_equivalence: system and user messages, verbatim (typo and indentation
# included).
EQUIV_SYSTEM = ("\n"
                "            You are a binary classifier.\n"
                "            If the TWO snippets describe the SAME event/fact, reply **YES**\n"
                "            Otherwise reply **NO**. No extra words.\n"
                "            DO NOT provide any exaplanation.\n"
                "        ")
EQUIV_USER = ("First snippet: {first} \n"
              "\n"
              "                       Second snippet: {second}\n"
              "                    ")


def _flatten_chat(chat: Any) -> List[Dict[str, str]]:
    """Every session's messages, in order, as {'role', 'content'} turns (upstream's flattening)."""
    messages: List[Dict[str, str]] = []
    for session in list(chat if chat is not None else []):
        for turn in ([session] if isinstance(session, dict) else list(session)):
            messages.append({"role": str(turn["role"]), "content": str(turn["content"])})
    return messages


def _parse_probing(raw: Any) -> Dict[str, Any]:
    """`probing_questions` ships as a stringified dict of ability -> list of question dicts."""
    if isinstance(raw, str):
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return json.loads(raw)
    return raw if isinstance(raw, dict) else {}


def _parse_rubric(raw: Any) -> List[str]:
    """BEAM's `rubric` is a list of criteria (nuggets), sometimes stringified."""
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    text = str(raw or "").strip()
    if not text:
        return []
    for loader in (ast.literal_eval, json.loads):
        try:
            val = loader(text)
            if isinstance(val, list):
                return [str(x) for x in val if str(x).strip()]
        except Exception:
            continue
    return [text]


def _beam_gold(gold_target: Any) -> Tuple[str, List[str]]:
    """(reference answer, rubric items) from the JSON gold payload."""
    try:
        obj = json.loads(gold_target)
        if isinstance(obj, dict):
            return str(obj.get("ideal") or ""), _parse_rubric(obj.get("rubric"))
    except Exception:
        pass
    return str(gold_target or ""), []


def _prune_from_tail(history: List[Dict[str, str]], final_turn: Any, window: int,
                     reserve: int) -> List[Dict[str, str]]:
    """Upstream prune_from_tail, sized to the served window.

    Keeps the longest suffix with an even number of messages (so it opens on a user turn, as
    upstream's odd-length trim does) for which the prompt plus the output budget fits the
    window. `final_turn` is what follows the history: one question turn (a string) or a list of
    messages (the question turns upstream's prompt carries, _question_tail). Sizes come from
    base._prompt_token_bounds (server /tokenize, else the run's tokenizer, else a conservative
    estimate) with clamp_to_context's slack, so base never has to shrink max_tokens afterwards.
    """
    tail = ([{"role": "user", "content": final_turn}] if isinstance(final_turn, str)
            else list(final_turn))

    def fits(pairs: int) -> bool:
        kept = history[len(history) - 2 * pairs:] if pairs else []
        bounds = base._prompt_token_bounds(kept + tail)
        if bounds is None:
            return False
        _low, high, mode = bounds
        return high + (16 if mode == "server" else 64) + reserve <= window

    pairs = len(history) // 2
    if fits(pairs):
        return list(history)
    if not fits(0):
        from .swebench_common import infra_required
        raise infra_required(
            "beam_128k", f"the {window:,}-token window cannot hold even the question plus the "
            f"{reserve:,}-token answer room; serve a larger window (or, with --thinking, set a "
            "smaller --max-output-tokens)",
            DOCS_URL)
    lo, hi = 0, pairs          # fits(lo) holds, fits(hi) does not
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return history[len(history) - 2 * lo:] if lo else []


def _longest_final_turn(turns: List[str]) -> str:
    """The question turn with the most prompt tokens (ties: the most characters).

    A conversation's questions share one pruned history, so it is sized for the question that
    needs the most room. Characters are not tokens: on the pinned data, in 7 of the 20
    conversations the longest question by characters is not the longest in Gemma tokens. The
    turns differ only in the question at their end, so the one that is longest alone is also
    the longest after the shared history.
    """
    def size(turn: str) -> Tuple[int, int]:
        bounds = base._prompt_token_bounds([{"role": "user", "content": turn}])
        return (bounds[1] if bounds else len(turn), len(turn))
    return max(turns, key=size)


def _question_tail(questions: List[str], position: int) -> List[Dict[str, str]]:
    """The turns upstream's long-context prompt puts after the history for the conversation's
    question at `position` (0-based, in answer_generation.py's order: probing_questions' keys,
    then each list's order).

    long_term_memory_methods.probing_question_evaluation reuses the saved history list by
    reference (`if saved_messages: messages = saved_messages`) and appends the question turn to
    it, and answer_generation.py saves the list only once per conversation (after its first
    question, as the copy messages[:-1]). So from the third question on, the saved list, and every
    later prompt, also carries the question turns of questions 2..k-1: position p gets the turns
    of positions 1..p-1, then its own."""
    return [{"role": "user", "content": FINAL_TURN.format(query=q)}
            for q in questions[1:position] + [questions[position]]]


def _largest_tail(tails: List[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """The tail with the most prompt tokens (ties: the most characters): a conversation's
    questions share one pruned history, sized for the tail that needs the most room."""
    def size(tail: List[Dict[str, str]]) -> Tuple[int, int]:
        chars = sum(len(m["content"]) for m in tail)
        bounds = base._prompt_token_bounds(tail)
        return (bounds[1] if bounds else chars, chars)
    return max(tails, key=size)


def _load_beam_128k_samples(
    limit: Optional[int] = None,
    window: Optional[int] = None,
    output_reserve: int = 0,
    stats: Optional[Dict[str, Any]] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """BEAM 128K tier from the pinned HF revision, as chat turns plus upstream's question turn.

    With a `window`, each conversation is tail-pruned to fit it (see _prune_from_tail).
    """
    conversations: List[Tuple[str, List[Dict[str, str]]]] = []
    #: each conversation's questions in answer_generation.py's order (the aliasing, _question_tail)
    questions_of: List[List[str]] = []
    entries: List[Tuple[int, str, Dict[str, Any], int]] = []
    try:
        from huggingface_hub import hf_hub_download
        import pandas as pd
        parquet_file = hf_hub_download(repo_id=_DATASET, filename=_PARQUET, repo_type="dataset",
                                       revision=_REVISION)
        df = pd.read_parquet(parquet_file)
        for _, row in df.iterrows():
            conversations.append((str(row.get("conversation_id", "")), _flatten_chat(row.get("chat"))))
            order: List[str] = []
            questions_of.append(order)
            for ability, items in _parse_probing(row.get("probing_questions")).items():
                for q_item in (items if isinstance(items, list) else []):
                    if isinstance(q_item, dict) and str(q_item.get("question") or "").strip():
                        if not _parse_rubric(q_item.get("rubric")):
                            raise ValueError(f"{ability} question without a rubric in conversation "
                                             f"{conversations[-1][0]}")
                        entries.append((len(conversations) - 1, str(ability), q_item, len(order)))
                        order.append(str(q_item["question"]))
    except Exception as e:
        logger.error(f"Failed to load dataset for beam_128k: {e}")
        raise RuntimeError(f"Could not load dataset for beam_128k: {e}") from e

    if not entries:
        raise RuntimeError("Dataset for beam_128k returned empty rows")

    # --shard first, over the full ordered set, then the --eval-limit draw within the shard.
    entries, _ = apply_shard(entries, "beam_128k")
    # Stratified across the ten abilities, not a contiguous head.
    entries = stratified_sample(entries, limit, lambda e: e[1], seed="beam_128k")

    histories: Dict[int, List[Dict[str, str]]] = {}
    for conv_idx in sorted({e[0] for e in entries}):
        history = conversations[conv_idx][1]
        if window:
            # One pruned history per conversation, sized for the drawn question whose tail (its
            # turn plus the earlier question turns upstream's prompt carries) has the most tokens,
            # so all its questions share the same (cacheable) prefix and every prompt fits.
            largest = _largest_tail([_question_tail(questions_of[conv_idx], pos)
                                     for c, _, _q, pos in entries if c == conv_idx])
            history = _prune_from_tail(history, largest, window, output_reserve)
        histories[conv_idx] = history
    if stats is not None:
        stats["conversations"] = len(histories)
        stats["pruned_conversations"] = sum(
            1 for i, h in histories.items() if len(h) < len(conversations[i][1]))

    samples = []
    for conv_idx, ability, q_item, position in entries:
        conv_id, full = conversations[conv_idx]
        question = str(q_item["question"])
        # upstream saves an empty history as [] (falsy), which is rebuilt, never reused: no
        # earlier question turns are carried then
        tail = (_question_tail(questions_of[conv_idx], position) if histories[conv_idx]
                else [{"role": "user", "content": FINAL_TURN.format(query=question)}])
        messages = histories[conv_idx] + tail
        ideal = next((q_item[k] for k in _IDEAL_KEYS if q_item.get(k)), "")
        gold_payload = json.dumps({"ideal": str(ideal), "rubric": _parse_rubric(q_item.get("rubric"))})
        samples.append((messages, gold_payload, {
            "category": ability,
            "conversation_id": conv_id,
            "difficulty": q_item.get("difficulty", "medium"),
            "question": question,
            "history_messages": len(histories[conv_idx]),
            "history_messages_total": len(full),
            # the question's place in answer_generation.py's order, and the earlier question
            # turns its prompt carries (_question_tail)
            "question_position": position,
            "carried_question_turns": len(tail) - 1,
        }))

    logger.info(f"Loaded {len(samples)} beam_128k samples.")
    return samples


def parse_json_response(response: Optional[str]) -> Any:
    """compute_metrics.parse_json_response: a fenced block, the whole reply, then the first {...}.

    Returns None where upstream raises; parse_judge_reply then applies upstream's json_repair
    fallback.
    """
    response = str(response or "").strip()
    if response.startswith("```"):
        match = re.search(r"```(?:json)?\s*(\[.*\]|\{.*\})\s*```", response, re.DOTALL)
        if match:
            response = match.group(1).strip()
    try:
        return json.loads(response)
    except ValueError:
        pass
    match = re.search(r'(\{.*?\}|\[.*?\])', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except ValueError:
            return None
    return None


def parse_judge_reply(response: Optional[str]) -> Any:
    """The rubric-item parse of compute_metrics.evaluate_*: parse_json_response, and when that
    raises, json.loads(repair_json(response)). Returns None where the repaired text still does not
    load (upstream raises there)."""
    parsed = parse_json_response(response)
    if parsed is not None:
        return parsed
    from gbench.third_party._json_repair_vendor import repair_json
    try:
        return json.loads(repair_json(str(response or "").strip()))
    except ValueError:
        return None


def event_ordering_score(reference_canon: List[str], system_canon: List[str]) -> Dict[str, float]:
    """compute_metrics.event_ordering_score after alignment: P/R/F1 and Kendall tau-b over ranks."""
    from scipy.stats import kendalltau
    tp = len(set(reference_canon) & set(system_canon))
    fp = len([x for x in system_canon if x not in reference_canon])
    fn = len([x for x in reference_canon if x not in system_canon])
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    union = list(dict.fromkeys(reference_canon + system_canon))
    tie_rank = len(union) + 1

    def to_rank(seq):
        r = {item: i + 1 for i, item in enumerate(seq)}
        return [r.get(u, tie_rank) for u in union]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # constant-input warnings; tau is NaN there
        tau_b, _p = kendalltau(to_rank(reference_canon), to_rank(system_canon),
                               variant="b", method="auto")
    # Upstream maps a None tau to 0; a NaN (constant ranks) would poison the mean, so it
    # gets the same treatment.
    tau_b_norm = (tau_b + 1) / 2 if tau_b is not None and not math.isnan(tau_b) else 0
    return dict(precision=precision, recall=recall, f1=f1, tau_norm=tau_b_norm,
                final_score=tau_b_norm * f1)


def _require_scipy() -> None:
    try:
        import scipy.stats  # noqa: F401
    except ImportError:
        from .swebench_common import infra_required
        raise infra_required("beam_128k", "event_ordering is scored with scipy.stats.kendalltau; "
                             "install scipy (pip install -e '.[evals]')", DOCS_URL)


#: json_repair release BEAM's requirements.txt pins (b2da22ea, line 59). repair_json is heuristic
#: and changes across releases, so this exact release is vendored as _json_repair_vendor: the
#: Python files of the PyPI wheel json_repair-0.44.1-py3-none-any.whl (sha256
#: 51d82532c3b8263782a301eb7904c75dce5fee8c0d1aba490287fc0ab779ac50), byte-identical (MIT, see
#: _json_repair_vendor/LICENSE), so the installed json_repair release plays no part.
UPSTREAM_JSON_REPAIR = "0.44.1"
#: sha256 of each vendored file, as the wheel's RECORD lists it (__main__.py, the CLI entry
#: point, and the py.typed marker are not vendored).
JSON_REPAIR_VENDOR_SHA256 = {
    "__init__.py": "7382f6919ac7bd610a7e3fce0d4da76a588dbaf53a165155c6d17485b2deeacf",
    "json_context.py": "9a6e9d3b2acf275b03b244ce459497096ed6f7ee6f78c9542aa5d0dc7c37106e",
    "json_parser.py": "c260e05c0ae843881865d8b84db767dcb2979f1db1daffee6a7cd2cea3f46924",
    "json_repair.py": "afe32daf5e94fe7db09875ffccd4592366652dcd005747cb5a52f31845a325ad",
    "object_comparer.py": "49e89c07afcde011c010fa27eecd8110b11a25ce28c91f5985f5f64603e4e81c",
    "string_file_wrapper.py": "92866676adbe6792bb5c5d5b0ea5fa7446cd69530961c3138eaf9a19ee8d42f0",
}


def _json_repair_version() -> str:
    """The json_repair release that repairs judge replies: the vendored one."""
    return UPSTREAM_JSON_REPAIR


def _require_json_repair() -> None:
    """Refuse to score with a vendored json_repair that is not the pinned release's files."""
    import hashlib
    vendor = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party",
        "_json_repair_vendor"))
    for name, expected in JSON_REPAIR_VENDOR_SHA256.items():
        path = os.path.join(vendor, name)
        try:
            with open(path, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
        except OSError as e:
            raise RuntimeError(f"beam_128k: the vendored json_repair {UPSTREAM_JSON_REPAIR} file "
                               f"{path} cannot be read ({e})") from e
        if digest != expected:
            raise RuntimeError(f"beam_128k: the vendored json_repair {UPSTREAM_JSON_REPAIR} file "
                               f"{path} has sha256 {digest}, expected {expected}")


#: The judge_grade of a trace that ended in a harness error (its request produced no reply).
HARNESS_ERROR_GRADE = "HARNESS_ERROR"
#: base's statuses of a request that produced no reply, and the harness error each is.
_NO_REPLY_KINDS = {"FAILED": "endpoint_failure", "TIMEOUT": "endpoint_timeout",
                   "OVER_CONTEXT": "over_context"}
#: Characters of a rubric-item judge reply kept on a trace.
_JUDGE_REPLY_CHARS = 2000
_THOUGHT_RE = re.compile(r"<thought>.*?</thought>", re.DOTALL | re.IGNORECASE)
_REASONING_RE = re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE)


def _answer_text(raw: Optional[str]) -> str:
    """The reply as upstream's judge and event_ordering split receive it (the model's .content):
    inline <thought>/<reasoning> blocks removed, as base.strip_thinking_tags removes them, but no
    whitespace stripped, so response.split("\n") keeps a leading or trailing empty line as
    upstream's does. A reply that is only such blocks is kept as it is."""
    if not raw:
        return ""
    cleaned = _REASONING_RE.sub("", _THOUGHT_RE.sub("", raw))
    return cleaned if cleaned.strip() else raw


def _make_scorer(judge_usage: Counter, judge_concurrency: int = 64):
    """BEAM's per-rubric-item judge and event_ordering tau_norm over the Gemini judge cascade."""
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        from tqdm import tqdm

        if not gemini_api_keys():
            # run_beam_128k's gate already hard-errors on a missing key; never degrade to
            # string matching.
            from .swebench_common import infra_required
            raise infra_required("beam_128k", "requires GEMINI_API_KEY (or GEMINI_API_KEYS) for "
                                 "the BEAM judge", DOCS_URL)

        semaphore = asyncio.Semaphore(judge_concurrency)
        item_cfg = judge_config()     # upstream's judge.invoke(prompt): no response format
        equiv_cfg = judge_config(system_instruction=EQUIV_SYSTEM)

        async def _item_score(item: str, response: str
                              ) -> Tuple[str, Optional[float], Dict[str, Any]]:
            """(outcome, score, record): the record keeps every judge reply of the item with the
            Gemini model that gave it, as upstream's llm_judge_responses keeps each parsed
            judge JSON (its `reason` included)."""
            prompt = JUDGE_PROMPT.replace("<rubric_item>", item).replace("<llm_response>", response)
            rec: Dict[str, Any] = {"rubric_item": item, "replies": []}
            for _ in range(_JUDGE_ATTEMPTS):
                async with semaphore:
                    text, model = await judge_generate_cascade(prompt, config=item_cfg)
                if text is None:
                    rec["outcome"] = "JUDGE_OUTAGE"
                    return "JUDGE_OUTAGE", None, rec
                judge_usage[model] += 1
                rec["replies"].append({"judge_model": model, "reply": text[:_JUDGE_REPLY_CHARS]})
                parsed = parse_judge_reply(text)
                try:
                    score = float(parsed["score"])
                except (TypeError, KeyError, ValueError, IndexError):
                    continue
                rec.update(outcome="OK", score=score, judge_model=model,
                           reason=parsed.get("reason") if isinstance(parsed, dict) else None)
                return "OK", score, rec
            rec["outcome"] = "JUDGE_FAILED"
            return "JUDGE_FAILED", None, rec

        async def _align(reference: List[str], system: List[str],
                         calls: List[Dict[str, Any]]) -> Optional[List[str]]:
            """compute_metrics.align_with_llm: each response line takes the first unused
            rubric event the equivalence check calls the same. Repeated (event, line) pairs are
            asked once; at judge temperature 0 re-asking returns the same answer. Every check is
            appended to `calls` (the event, the line, the answering model, its reply, the
            decision)."""
            asked: Dict[Tuple[str, str], bool] = {}
            used, system_out = set(), []
            for s in system:
                matched = None
                for index, r in enumerate(reference):
                    if index in used:
                        continue
                    if (r, s) not in asked:
                        async with semaphore:
                            text, model = await judge_generate_cascade(
                                EQUIV_USER.format(first=r, second=s), config=equiv_cfg)
                        if text is None:
                            calls.append({"event": r, "line": s, "judge_model": model,
                                          "reply": None, "same": None})
                            return None
                        judge_usage[model] += 1
                        asked[(r, s)] = "yes" in text.lower()
                        calls.append({"event": r, "line": s, "judge_model": model,
                                      "reply": text[:200], "same": asked[(r, s)]})
                    if asked[(r, s)]:
                        matched = index
                        break
                if matched is not None:
                    system_out.append(reference[matched])
                    used.add(matched)
                else:
                    system_out.append(s)
            return system_out

        async def _judge(trace: Dict[str, Any], pbar) -> None:
            status = str(trace.get("status") or "")
            if status in _NO_REPLY_KINDS:
                # The request produced no reply (an endpoint error, a timeout, a prompt longer
                # than the served window): a harness error, left out of the ability means and
                # the pass rate, never scored as the model's answer.
                trace["is_correct"] = False
                trace["judge_grade"] = HARNESS_ERROR_GRADE
                trace["harness_error"] = {"kind": _NO_REPLY_KINDS[status],
                                          "detail": str(trace.get("error") or "no reply")[:300]}
                trace["scoring_excluded"] = True
                pbar.update(1)
                return
            # upstream judges and splits the reply's .content as it is: only inline reasoning
            # blocks are removed, never the surrounding whitespace
            response = _answer_text(trace.get("response_text"))
            _ideal, rubric = _beam_gold(trace.get("gold_answer"))
            graded = await asyncio.gather(*[_item_score(item, response) for item in rubric])
            if trace.get("status") == "PENDING_JUDGE":   # keep EMPTY_RESPONSE etc.
                trace["status"] = "OK"
            trace["beam_judge_calls"] = [rec for _o, _s, rec in graded]
            outcomes = {o for o, _s, _r in graded}
            if "JUDGE_OUTAGE" in outcomes:
                trace["judge_grade"] = "JUDGE_OUTAGE"   # excluded from the mean, not scored 0
                pbar.update(1)
                return
            if "JUDGE_FAILED" in outcomes or not rubric:
                # the judge answered without a usable score (upstream raises there): a harness
                # error, left out of the means and the pass rate
                trace["judge_grade"] = "JUDGE_FAILED"
                trace["is_correct"] = False
                trace["scoring_excluded"] = True
                pbar.update(1)
                return
            item_scores = [sc for _o, sc, _r in graded]
            judge_score = sum(item_scores) / len(item_scores)
            score = judge_score
            if trace.get("category") == "event_ordering":
                calls: List[Dict[str, Any]] = []
                aligned = await _align(rubric, response.split("\n"), calls)
                trace["beam_alignment_calls"] = calls
                if aligned is None:
                    trace["judge_grade"] = "JUDGE_OUTAGE"
                    pbar.update(1)
                    return
                eo = event_ordering_score(rubric, aligned)
                trace["beam_event_ordering"] = {k: round(float(v), 6) for k, v in eo.items()}
                score = eo["tau_norm"]
            trace["beam_rubric_scores"] = item_scores
            trace["beam_llm_judge_score"] = round(judge_score, 6)
            trace["beam_score"] = round(float(score), 6)
            trace["is_correct"] = score >= 0.5           # secondary pass rate only
            trace["judge_grade"] = f"SCORE_{score:.2f}"
            pbar.update(1)

        with tqdm(total=len(sample_traces), desc="Judging [BEAM_128K]") as pbar:
            await asyncio.gather(*[_judge(t, pbar) for t in sample_traces])
    return _score


def _finalize(result: Dict[str, Any]) -> Dict[str, Any]:
    """Headline = unweighted mean of the per-ability means (tau_norm for event_ordering)."""
    per: Dict[str, List[float]] = {}
    for t in result.get("sample_traces") or []:
        if "beam_score" in t and t.get("judge_grade") not in ("JUDGE_OUTAGE", "JUDGE_FAILED"):
            per.setdefault(str(t.get("category")), []).append(float(t["beam_score"]))
    means = {a: sum(v) / len(v) for a, v in per.items() if v}
    traces = result.get("sample_traces") or []
    harness: Dict[str, int] = {}
    for t in traces:
        if t.get("harness_error"):
            kind = t["harness_error"]["kind"]
            harness[kind] = harness.get(kind, 0) + 1
    failed_judges = sum(1 for t in traces if t.get("judge_grade") == "JUDGE_FAILED")
    if failed_judges:
        harness["judge_no_score"] = failed_judges
    result["harness_errors"] = dict(sorted(harness.items()))
    result["pass_rate_at_0.5"] = result.get("accuracy")
    result["metric"] = ("BEAM score: unweighted mean over memory abilities of the per-ability mean "
                        "(rubric-item judge score; tau_norm for event_ordering)")
    if means:
        result["accuracy"] = round(sum(means.values()) / len(means) * 100.0, 2)
        result["scored_samples"] = sum(len(v) for v in per.values())
    elif traces:
        # every question ended in a harness error or a judge outage: no number to report
        outages = sum(1 for t in traces if t.get("judge_grade") == "JUDGE_OUTAGE")
        result["accuracy"] = None
        result["status"] = "failed"
        result["accuracy_withheld_reason"] = (
            "no question was scored: " + ", ".join(
                f"{k} {v}" for k, v in sorted({**harness, "judge_outage": outages}.items())
                if v))
    result["beam_ability_scores"] = {a: round(means[a] * 100.0, 2) for a in ABILITIES if a in means}
    # The per-category table must show the BEAM score, not base's 0.5 pass rate.
    for ability, mean in means.items():
        cell = (result.get("category_accuracy") or {}).get(ability)
        if isinstance(cell, dict):
            cell["pass_rate_at_0.5"] = cell.get("accuracy")
            cell["accuracy"] = round(mean * 100.0, 2)
    return result


def _output_reserve(enable_thinking: bool, max_output_tokens: Optional[int]) -> int:
    """The answer room the pruned prompt leaves in the window. No request carries a max_tokens
    (upstream's reader sends none), so an answer may use all of it: upstream's 2,000 tokens, or
    with --thinking the run's budget (--max-output-tokens, else the thinking floor), which a
    thinking run is never given less than (base.thinking_run_budget)."""
    if enable_thinking:
        return base.thinking_run_budget("beam_128k", max_output_tokens)
    return UPSTREAM_ANSWER_RESERVE


def judge_sampling_record() -> Dict[str, Any]:
    """The `judge_sampling` block of a BEAM result (rubric-item and alignment calls)."""
    return {"reference": JUDGE_REFERENCE, "temperature": 0.0,
            "max_output_tokens": "not sent (upstream sends none)",
            "response_mime_type": ("not sent (upstream's invoke(prompt) asks for no format; "
                                   "the reply is read with its parse and json_repair fallback)")}


def run_beam_128k(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run native beam_128k evaluation suite."""
    from .swebench_common import infra_required
    skip = gemini_required_skip("beam_128k", model_name)
    if skip is not None:
        return skip
    _require_scipy()
    _require_json_repair()
    raw = os.environ.get(WINDOW_ENV, "").strip()
    try:
        window = int(raw) if raw else get_run_knob("max_model_len")
    except ValueError:
        raise infra_required("beam_128k", f"{WINDOW_ENV}={raw!r} is not a number of tokens.", DOCS_URL)
    if not window:
        # Without the window the history cannot be pruned the way upstream prunes it.
        raise infra_required(
            "beam_128k", "the server does not report its context window (GET /v1/models), so the "
            f"conversation cannot be pruned to fit it. Set {WINDOW_ENV} to the served window in "
            "tokens (e.g. 131072).", DOCS_URL)
    reserve = _output_reserve(enable_thinking, kwargs.get("max_output_tokens"))
    stats: Dict[str, Any] = {}
    samples = _load_beam_128k_samples(limit=limit, window=int(window), output_reserve=reserve,
                                      stats=stats)
    judge_usage: Counter = Counter()
    eval_plan = eval_limit_plan(
        "beam_128k", limit, len(samples),
        attempts=base_attempts(batch_scored=True, supports_attempts=True))
    result = run_eval_suite(
        eval_name="beam_128k",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's questions, kept as is
        async_eval_fn=_make_scorer(judge_usage),
        limit=limit,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        max_output_tokens=kwargs.get("max_output_tokens"),   # not sent: no max_tokens, as upstream
        supports_attempts=True,   # every trace is judged on its own
        official_sampling=OFFICIAL_SAMPLING,
    )
    result["eval_limit_plan"] = eval_plan
    result = _finalize(result)
    result["dataset"] = f"{_DATASET}@{_REVISION} (100K split = the paper's 128K tier)"
    result["beam_window_tokens"] = int(window)
    result["beam_output_reserve_tokens"] = reserve
    result["beam_conversations"] = stats.get("conversations")
    result["beam_pruned_conversations"] = stats.get("pruned_conversations")
    result["judge"] = "gbench-gemini-cascade"
    result["judge_models"] = dict(judge_usage)   # answered judge calls per model
    result["judge_cascade"] = judge_cascade()
    result["judge_sampling"] = judge_sampling_record()
    result["json_repair_version"] = _json_repair_version()
    reasons = [f"judged by the gbench Gemini judge cascade "
               f"({', '.join(sorted(judge_usage)) or 'no judge call answered'}), not upstream's "
               f"{OFFICIAL_JUDGE}"]
    if stats.get("pruned_conversations") and reserve != UPSTREAM_ANSWER_RESERVE:
        reasons.append(
            f"{stats['pruned_conversations']} of {stats['conversations']} conversations were "
            f"tail-pruned to the {int(window):,}-token window reserving the {reserve:,}-token "
            f"thinking budget (upstream reserves {UPSTREAM_ANSWER_RESERVE:,} tokens)")
    harness = result.get("harness_errors") or {}
    if harness:
        reasons.append("harness errors, not the model's outcome and left out of the ability "
                       "means: " + ", ".join(f"{k} {v}" for k, v in harness.items())
                       + " (endpoint_failure / endpoint_timeout / over_context: the request "
                       "produced no reply, which upstream's answer_generation.py would record as "
                       "an 'Error in ...' text and judge; judge_no_score: the judge gave no "
                       f"usable score in {_JUDGE_ATTEMPTS} replies, where upstream raises)")
    if limit:
        reasons.append(f"--eval-limit {limit} runs a subset of the 400 questions")
    if (result.get("official_sampling") or {}).get("temperature_matches_official") is False:
        reasons.append(f"temperature {result.get('temperature')} (upstream generates at 0)")
    cut = record_budget_cutoffs(result)
    if cut:
        reasons.append(cut)
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)
