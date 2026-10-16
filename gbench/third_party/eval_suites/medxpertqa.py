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

"""Native MedXpertQA (expert-level medical exam QA, Text + MM subsets) evaluation suite.

Runs the official zero-shot chain-of-thought protocol of TsinghuaC3I/MedXpertQA
(`eval/main.py` zero_shot_cot + `eval/config/prompt_templates.py` + `eval/utils.py`
answer_cleansing @ 5f1db162d0d2b03d62da2d4fe086076a36105e18) over BOTH subsets of the
pinned dataset: Text (2,450 questions, options A-J) and MM (2,000 questions with images,
options A-E). Each question is a two-turn conversation, exactly as upstream:

  1. system "You are a helpful medical assistant.", user
     "Q: {question}\\nA: Let's think step by step." (the question already ends with its
     "Answer Choices: (A) ... " list, so the options are shown once), followed by the MM
     images from the dataset's images.zip;
  2. the model's rationale goes back as the assistant turn, then the user sends
     "Therefore, among A through {last letter}, the answer is" and the reply to that is
     scored with upstream's answer_cleansing (text after the trigger or " answer is ",
     then the FIRST option letter).

The headline is accuracy over all 4,450 questions (upstream eval.ipynb "Full"); the Text
and MM accuracies and the Reasoning / Understanding split within each subset are reported
alongside. A question whose rationale or answer request got no reply from the served endpoint
(failed, timed out, or over the context window) is a harness error: left out of every accuracy,
counted and named in status_reason, never scored as the model's answer.
Missing images.zip (or an image it should contain) is a hard error: scoring MM questions
without their images would measure a different task.

Sampling: base.OfficialSampling, as upstream's APIAgent (eval/model/api_agent.py) sends both turns
for a served model, in both modes. APIAgent keys its settings on the model name: get_response sends
names containing o1 or o3, and deepseek-reasoner, down a branch that drops the system turn and
sends no temperature and max_completion_tokens, and every other name down the chat branch; __init__
gives names containing o1, o3, qwq, qvq or reasoner 8192 tokens and every other name 2048
(config/prompt_templates.py also changes the prompts for deepseek-reasoner and qvq by name). Those
are model names of hosted APIs (OpenAI's o-series, DeepSeek's deepseek-reasoner) and of two Qwen
checkpoints, not settings upstream declares for reasoning models in general, so they are not
reasoning-model settings for a served model: a served model's name (for example
google/gemma-4-31b-it) takes the chat branch and the 2048, as would the other open models listed in
upstream's config/model_info.json (qwen2.5-72b-instruct, qwen2.5-vl-72b-instruct, the Llama 3
models) once added to a client list, as the README says to do for more models. The chat branch
keeps main.py's system turn and sends temperature 0 (eval/scripts/run.sh `temperature=${7:-0}`,
main.py `--temperature` default 0), seed 0 and max_tokens 2048. Without `--thinking` the 2048 is
sent exactly, so `--max-output-tokens` is not applied and a turn that stops at 2048 tokens goes on
as it arrived, as upstream's does (`replies_at_official_max_tokens` counts them). With `--thinking`
the same request is sent except for the temperature and the 2048: each turn is sampled at
temperature 1.0 (gbench's rule, base.resolve_temperature: 1.0 with `--thinking`, 0.0 without,
whatever the upstream sends) and, the 2048 being written for non-reasoning generation, each turn
is sent the run's `--max-output-tokens`, else the thinking floor
(base.resolve_token_budget; the result records the cap as not applied to reasoning and the budget
sent). The chat branch's logprobs=True only returns token log-probabilities for upstream's
confidence field; it does not change the reply or the score and is not requested. Override the
temperature for a whole run with `--temperature`, or for this suite alone with
`GBENCH_MEDXPERTQA_TEMPERATURE`, which takes precedence over both (and over the rule's 1.0 / 0.0);
the result records
`official_sampling`. The result is leaderboard-comparable only for a full, unsharded run at the
official temperature with every request answered and no reply cut off by the run's token budget
(`replies_cut_off_by_budget`, over both turns).
"""

import base64
import functools
import logging
import os
import re
import zipfile
from typing import Any, Dict, List, Optional, Tuple
from .limit_plan import base_attempts, eval_limit_plan, per_category
from .base import (OfficialSampling, answer_in_thinking_fields, is_scored, run_eval_suite,
                   strip_thinking_tags)
from .metrics import record_budget_cutoffs
from .sampling import parse_shard, shard_select, stratified_indices
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

DOCS_URL = "docs/evals/medxpertqa.md"
DATASET = "TsinghuaC3I/MedXpertQA"
DATASET_REVISION = "7e7c465a68eb2b866926bfa59c8c9d17a8daba65"
SUBSETS = ("Text", "MM")

# Upstream eval/config/prompt_templates.py, dataset "medxpertqa", zero_shot + cot.
SYSTEM_ROLE = "You are a helpful medical assistant."
COT_PROMPT = "Q: {question}\nA: Let's think step by step."
COT_TRIGGER = "Therefore, among {start} through {end}, the answer is"
#: eval/model/api_agent.py @ 5f1db162 APIAgent.__init__: max_tokens 2048 for every model name
#: that does not contain o1, o3, qwq, qvq or reasoner (those get 8192 by name).
UPSTREAM_MAX_TOKENS = 2048
#: get_response's chat branch, the one a served model takes (the other branch is keyed on names
#: containing o1 or o3, and deepseek-reasoner), at the temperature of eval/scripts/run.sh +
#: main.py (0). No `reasoning` settings: upstream declares none that are not keyed on a model
#: name, so a --thinking run sends the same request, with the run's budget in place of the 2048
#: and temperature 1.0 (base.resolve_temperature: gbench's rule, 1.0 with --thinking, 0.0
#: without).
OFFICIAL_SAMPLING = OfficialSampling(
    reference=("TsinghuaC3I/MedXpertQA eval/model/api_agent.py @5f1db162 get_response chat "
               "branch and APIAgent.__init__ max_tokens, the settings of a model name without "
               "o1, o3, qwq, qvq or reasoner in it (those name-keyed settings are not used for a "
               "served model, in either mode): system turn, temperature 0 (eval/scripts/run.sh), "
               "max_tokens 2048, seed 0"),
    temperature=0.0, params={"seed": 0}, max_tokens=UPSTREAM_MAX_TOKENS)


@functools.lru_cache(maxsize=1)
def _medxpertqa_image_dir() -> str:
    """Extract MedXpertQA's `images.zip` once and return the dir its image files live in.

    The MM subset's `images` column is a LIST OF FILENAMES ('MM-0-a.jpeg'); the pixels ship
    in a separate images.zip in the dataset repo. Without it the MM questions cannot be
    posed, so failing to get it is a hard error rather than a text-only run.
    """
    try:
        from huggingface_hub import hf_hub_download
        zpath = hf_hub_download(DATASET, "images.zip", repo_type="dataset",
                                revision=DATASET_REVISION)
        extract_root = zpath + "_extracted"
        if not os.path.isdir(extract_root):
            # Extract beside the target and rename, so an interrupted extraction never
            # leaves a half-populated directory that later runs would trust.
            tmp_root = extract_root + ".partial"
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(tmp_root)
            os.replace(tmp_root, extract_root)
    except Exception as e:
        raise infra_required(
            "medxpertqa",
            f"could not download/extract {DATASET}@{DATASET_REVISION[:8]} images.zip ({e}); "
            "the MM subset cannot be posed without its images.", DOCS_URL) from e
    for root, _dirs, files in os.walk(extract_root):
        if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
            return root
    raise infra_required("medxpertqa", f"images.zip extracted to {extract_root} holds no images.",
                         DOCS_URL)


def _image_part(image_dir: str, filename: str) -> Dict[str, Any]:
    """Upstream api_agent.image_content: the raw file bytes as a JPEG data URI."""
    path = os.path.join(image_dir, str(filename).strip())
    if not os.path.isfile(path):
        raise infra_required("medxpertqa", f"image {filename!r} is missing from images.zip "
                             f"({image_dir}).", DOCS_URL)
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def _load_medxpertqa_samples(
    limit: Optional[int] = None,
    subsets: Tuple[str, ...] = SUBSETS,
) -> List[Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]]:
    """Build the turn-1 (rationale) request for every question of the requested subsets.

    Gold is `{"label": "A", "n_options": 5}`: the option count fixes the trigger's letter
    range and the letters answer_cleansing accepts. Every request carries the system turn, as
    upstream's chat branch sends it, in both modes.
    """
    from datasets import load_dataset

    rows: List[Tuple[str, Dict[str, Any]]] = []
    for subset in subsets:
        ds = load_dataset(DATASET, subset, split="test", revision=DATASET_REVISION)
        rows.extend((subset, item) for item in ds)
    if not rows:
        raise RuntimeError("medxpertqa: the dataset loaded no rows")

    # Stratified over subset x question type, not a contiguous head.
    keys = [(subset, item.get("question_type")) for subset, item in rows]
    rows = [rows[i] for i in stratified_indices(keys, limit, seed="medxpertqa")]

    image_dir = _medxpertqa_image_dir() if any(item.get("images") for _, item in rows) else None
    samples = []
    for subset, item in rows:
        options = item.get("options") or {}
        label = str(item.get("label", item.get("answer", ""))).strip().upper()
        if not options or not label:
            raise RuntimeError(f"medxpertqa: row {item.get('id')!r} has no options/label")
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": COT_PROMPT.format(question=str(item["question"]).strip())}]
        for filename in item.get("images") or []:
            content.append(_image_part(image_dir, filename))
        messages = [{"role": "system", "content": SYSTEM_ROLE},
                    {"role": "user", "content": content}]
        gold = {"label": label, "n_options": len(options)}
        meta = {"category": f"{subset}/{item.get('question_type') or 'unknown'}",
                "medxpertqa_id": item.get("id"), "subset": subset}
        samples.append((messages, gold, meta))

    logger.info("Loaded %d MedXpertQA questions (%s) from %s@%s.", len(samples),
                ", ".join(subsets), DATASET, DATASET_REVISION[:8])
    return samples


def _answer_cleansing(prediction: str, n_options: int) -> List[str]:
    """Upstream utils.answer_cleansing for method "zero_shot" on MedXpertQA (same logic)."""
    answer_trigger = COT_TRIGGER.format(start="A", end=chr(65 + n_options - 1))
    try:
        prediction = prediction.split(answer_trigger)[1].strip()
    except IndexError:
        try:
            prediction = prediction.split(" answer is ")[1].strip()
        except IndexError:
            pass
    for unwanted_phrase in ["I understand", "A through J", "A through E", "A through D"]:
        prediction = prediction.replace(unwanted_phrase, "")
    options = [chr(65 + i) for i in range(n_options)]
    options_str = r"\b(" + "|".join(options) + r")\b"
    found = re.findall(options_str, prediction)
    if len(found) == 0:
        return []
    # Single-letter label: upstream keeps the FIRST match for zero-shot.
    found = [found[0]]
    if found[0] and found[0].endswith("."):
        found[0] = found[0][:-1]
    return found


def _eval_medxpertqa(response_text: str, gold: Any) -> bool:
    """Score the reply to the answer trigger (upstream answer_cleansing + set_correctness).

    `gold` is the loader's `{"label", "n_options"}`; a bare letter is accepted too and
    then allows A-J, the widest option range in the dataset.
    """
    if isinstance(gold, dict):
        label, n_options = str(gold.get("label") or ""), int(gold.get("n_options") or 10)
    else:
        label, n_options = str(gold or ""), 10
    label = label.strip()
    if not response_text or not label:
        return False
    prediction = _answer_cleansing(response_text.strip(), n_options)
    if len(prediction) == 0:
        return False
    return label.lower() == prediction[0].lower()


async def _hold_for_answer_turn(sample_traces: List[Dict[str, Any]]) -> None:
    """Turn 1 produces the rationale only; scoring happens on the turn-2 reply."""
    for tr in sample_traces:
        if tr.get("status") == "PENDING_JUDGE":
            tr["status"] = "OK"


def _presharded(samples: List[Any]) -> List[Any]:
    """Turn-2 samples laid out so base's GBENCH_SHARD selection keeps every one of them.

    Turn 1 was already cut to this node's shard, and base.run_eval_suite cuts every call again
    with shard_select(items, (I, N)) = items[I-1::N]. Each turn-2 sample is therefore placed
    at an index that selection keeps; the other slots are None and are dropped by that same
    selection before anything reads them. The process environment is never changed, so a
    suite abandoned by --suite-timeout cannot leave the next suite unsharded.
    """
    shard = parse_shard(os.environ.get("GBENCH_SHARD"))
    if shard is None:
        return samples
    i, n = shard
    padded: List[Any] = [None] * (len(samples) * n)
    padded[i - 1::n] = samples
    if shard_select(padded, shard) != list(samples):
        raise RuntimeError("medxpertqa: turn-2 samples do not survive the GBENCH_SHARD selection")
    return padded


def _answer_turn_samples(
    turn1_traces: List[Dict[str, Any]],
    turn1_messages: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]], List[Dict[str, Any]]]:
    """Turn-2 requests for every turn-1 reply, plus the turn-1 traces whose request failed.

    An empty reply (status EMPTY_RESPONSE, e.g. reasoning that used the whole budget) still
    gets the trigger, after an empty assistant turn: upstream's zero_shot_cot appends whatever
    turn 1 returned and always sends the trigger. Only a request that returned no reply at all
    (failed, timed out, over the context window) has no turn to continue from.
    """
    samples, no_rationale = [], []
    for tr in turn1_traces:
        meta = tr.get("extra_payload") or {}
        if tr.get("response_text") is None:
            no_rationale.append(tr)
            continue
        rationale = strip_thinking_tags(tr.get("response_text") or "").strip()
        gold = tr.get("gold_answer") or {}
        n_options = int(gold.get("n_options") or 10)
        trigger = COT_TRIGGER.format(start="A", end=chr(65 + n_options - 1))
        messages = list(turn1_messages[meta["medxpertqa_id"]]) + [
            {"role": "assistant", "content": rationale},
            {"role": "user", "content": trigger},
        ]
        samples.append((messages, gold, {**meta, "rationale_status": tr.get("status"),
                                         "rationale_attempt": tr.get("attempt_idx", 0)}))
    return samples, no_rationale


#: Counters that add up across the two turns.
_SUMMED_COUNTERS = ("failed_requests", "empty_responses", "truncated_responses", "over_context_prompts",
                    "timed_out_requests", "looping_responses", "discarded_outputs",
                    "genuinely_truncated", "non_convergent_responses",
                    "replies_at_official_max_tokens", "answer_in_thinking")


def _accuracy(traces: List[Dict[str, Any]]) -> Optional[float]:
    """Accuracy over the scored traces (None when none is scored)."""
    traces = [t for t in traces if is_scored(t)]
    if not traces:
        return None
    return round(100.0 * sum(1 for t in traces if t.get("is_correct")) / len(traces), 2)


def _turn_reasons(turn1: Dict[str, Any], turn2: Dict[str, Any], two_turns: bool) -> List[str]:
    """Each turn's status_reason, labelled by turn."""
    parts = [("rationale turn", turn1.get("status_reason"))]
    if two_turns:
        parts.append(("answer turn", turn2.get("status_reason")))
    return [f"{label}: {reason}" for label, reason in parts if reason]


def _combine(turn1: Dict[str, Any], turn2: Dict[str, Any],
             no_rationale: List[Dict[str, Any]], attempts: int) -> Dict[str, Any]:
    """One result over every turn-1 generation.

    A question whose rationale request (turn 1) or answer request (turn 2) got no reply from the
    served endpoint is a harness error: base marks its trace (harness_error, scoring_excluded)
    in that turn, and it is left out of every accuracy here, counted and named in status_reason,
    never scored as the model's answer. With no question scored the score is withheld (None and
    accuracy_withheld_reason). An empty turn-1 reply still gets turn 2 and is scored."""
    two_turns = bool(turn2.get("sample_traces")) or not no_rationale
    for tr in no_rationale:
        tr["is_correct"] = False
        tr["error"] = tr.get("error") or "no rationale from turn 1"
    traces = list(turn2.get("sample_traces") or []) + no_rationale
    result = dict(turn2)
    result["sample_traces"] = traces
    total = len(traces)
    scored = [t for t in traces if is_scored(t)]
    correct = sum(1 for t in scored if t.get("is_correct"))
    result["total_questions"] = total
    result["correct_answers"] = correct
    result["accuracy"] = round(100.0 * correct / len(scored), 2) if scored else None
    result["scoring_excluded"] = sum(1 for t in traces if t.get("scoring_excluded"))
    kinds: Dict[str, int] = {}
    for t in traces:
        kind = (t.get("harness_error") or {}).get("kind")
        if kind:
            kinds[kind] = kinds.get(kind, 0) + 1
    result.pop("request_harness_errors", None)
    if kinds:
        result["request_harness_errors"] = dict(sorted(kinds.items()))
    for key in _SUMMED_COUNTERS:
        result[key] = int(turn1.get(key) or 0) + int(turn2.get(key) or 0)
    # over the requests of both turns (base's ANSWER_IN_THINKING_POLICY)
    result.update(answer_in_thinking_fields(
        result["answer_in_thinking"],
        len(turn1.get("sample_traces") or []) + len(turn2.get("sample_traces") or [])))
    result["duration_s"] = round(float(turn1.get("duration_s") or 0) + float(turn2.get("duration_s") or 0), 2)
    result["medxpertqa_rationale_turn"] = {
        key: turn1.get(key) for key in ("failed_requests", "empty_responses", "truncated_responses",
                                        "over_context_prompts", "timed_out_requests",
                                        "max_tokens_sent", "replies_at_official_max_tokens",
                                        "response_health")}

    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    by_subset: Dict[str, List[Dict[str, Any]]] = {}
    for t in scored:
        by_cat.setdefault(t.get("category") or "unknown", []).append(t)
        by_subset.setdefault((t.get("extra_payload") or {}).get("subset") or "unknown", []).append(t)
    result["category_accuracy"] = {
        cat: {"correct": sum(1 for t in ts if t.get("is_correct")), "total": len(ts),
              "accuracy": _accuracy(ts)} for cat, ts in sorted(by_cat.items())}
    result["medxpertqa_text_accuracy"] = _accuracy(by_subset.get("Text", []))
    result["medxpertqa_mm_accuracy"] = _accuracy(by_subset.get("MM", []))

    if attempts > 1:
        # Over the scored attempts only; a question with none scored is not counted.
        by_q: Dict[Any, List[bool]] = {}
        for t in scored:
            by_q.setdefault((t.get("extra_payload") or {}).get("medxpertqa_id"), []).append(
                bool(t.get("is_correct")))
        n_q = len(by_q)
        # With no question scored (every attempt a harness error) the rates have no value
        # (None, as base reports them), never 0.
        result["attempts"] = {
            "attempts_requested": attempts, "attempts_per_sample": attempts, "samples": n_q,
            "generations": total, "avg_at_k": result["accuracy"],
            "pass_at_k": (round(100.0 * sum(1 for v in by_q.values() if any(v)) / n_q, 2)
                          if n_q else None),
            "pass_hat_k": (round(100.0 * sum(1 for v in by_q.values() if v and all(v)) / n_q, 2)
                           if n_q else None),
            "unstable_samples": sum(1 for v in by_q.values() if any(v) and not all(v)),
        }

    # The summed counters above report both turns. Whether a question was answered at all is a
    # per-question outcome: it has no answer when turn 1 returned no reply (no_rationale) or its
    # turn-2 request got none (harness errors, left out of the score). An empty turn-1 reply
    # still gets turn 2, so it alone does not make the run "failed".
    unanswered = sum(1 for t in traces if t.get("harness_error"))
    result["medxpertqa_unanswered_questions"] = unanswered
    failed = result["failed_requests"]
    reasons = _turn_reasons(turn1, turn2, two_turns)
    result.pop("accuracy_withheld_reason", None)
    if total == 0:
        result["status"] = "error"
        result["status_reason"] = "no samples were run"
    elif not scored:
        result["status"] = "failed"
        result["status_reason"] = "nothing was scored: " + (
            "; ".join(reasons) or f"all {total} question(s) left out of the score")
        result["accuracy_withheld_reason"] = result["status_reason"]
    # A turn that stopped at upstream's max_tokens is the protocol's outcome, not a truncation.
    elif failed == 0 and not unanswered and (result["truncated_responses"]
                                             - result["replies_at_official_max_tokens"]) * 10 \
            <= total:
        result["status"] = "success"
        result["status_reason"] = None
    else:
        result["status"] = "completed_with_errors"
        result["status_reason"] = "; ".join(reasons) or (
            f"failed_requests {failed}" if failed else
            "replies cut off by a max_tokens that is not an official cap (over 10%)")
    return result


def run_medxpertqa(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run MedXpertQA Text + MM with the official two-turn zero-shot CoT protocol."""
    limit = kwargs.get("limit")
    sharded = bool(os.environ.get("GBENCH_SHARD"))
    samples = _load_medxpertqa_samples(limit=limit)
    # One sample = one question, asked as a two-turn conversation (rationale, then the trigger).
    # The plan counts the questions base runs on this node: turn 1 is GBENCH_SHARD's shard of
    # the drawn list (base shards it), so the shards' plans add up to the questions drawn.
    on_shard = [x for x in shard_select(samples, parse_shard(os.environ.get("GBENCH_SHARD")))
                if x is not None]
    eval_plan = eval_limit_plan(
        "medxpertqa", limit, len(on_shard), per_category(on_shard, "subset"),
        setting_noun="subset",
        attempts=int(kwargs.get("attempt_count") or 0)
        or base_attempts(batch_scored=True, supports_attempts=True),
        unit="conversations")
    turn1_messages = {meta["medxpertqa_id"]: messages for messages, _gold, meta in samples}
    # Both turns carry upstream's per-call max_tokens under the token-limit policy (2048 for a
    # no-think run; with --thinking the run's --max-output-tokens, the 2048 not applied).
    common = dict(model_name=model_name, base_url=base_url, concurrency=concurrency,
                  thinking=enable_thinking, extra_payload=kwargs.get("extra_payload"),
                  max_output_tokens=kwargs.get("max_output_tokens"),
                  official_sampling=OFFICIAL_SAMPLING)

    turn1 = run_eval_suite(
        eval_name="medxpertqa", samples=samples, async_eval_fn=_hold_for_answer_turn,
        declared_scoring_mode="deterministic", limit=None, supports_attempts=True,
        attempt_count=kwargs.get("attempt_count"), **common)
    attempts = int(((turn1.get("attempts") or {}).get("attempts_per_sample")) or 1)
    answer_samples, no_rationale = _answer_turn_samples(turn1.get("sample_traces") or [],
                                                        turn1_messages)
    if answer_samples:
        turn2 = run_eval_suite(
            eval_name="medxpertqa", samples=_presharded(answer_samples), eval_fn=_eval_medxpertqa,
            limit=None, attempt_count=1, **common)
    else:
        # Every rationale request failed, so there is nothing to send the trigger after.
        turn2 = {**{k: v for k, v in turn1.items() if k not in _SUMMED_COUNTERS},
                 "sample_traces": [], "duration_s": 0.0}

    result = _combine(turn1, turn2, no_rationale, attempts)
    result["eval_limit_plan"] = eval_plan
    result["metric"] = ("MedXpertQA accuracy over Text + MM (official two-turn zero-shot CoT, "
                        "answer_cleansing first letter); medxpertqa_text_accuracy / "
                        "medxpertqa_mm_accuracy and Subset/QuestionType in category_accuracy")
    reasons = []
    if limit:
        reasons.append(f"--eval-limit {limit} subset of the 4,450 questions")
    if sharded:
        reasons.append("GBENCH_SHARD run covers one shard")
    if (result.get("official_sampling") or {}).get("temperature_matches_official") is False:
        reasons.append(f"temperature {result.get('temperature')} (upstream runs at 0)")
    # A turn-1 request with no reply stays in the traces with its status, next to turn 2's.
    from .vqa_common import lost_requests_reason
    lost = lost_requests_reason(result.get("sample_traces") or [])
    if lost:
        reasons.append(lost)
    # Replies a token budget cut off (not an official cap), over both turns.
    cut = record_budget_cutoffs(result, parts=[turn1, turn2])
    if cut:
        reasons.append(cut)
    result["leaderboard_comparable"] = not reasons
    result["leaderboard_comparable_reason"] = "; ".join(reasons) or None
    return result
