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
#
# Evaluation Suite: mmlu_redux
# Description: MMLU-Redux-2.0 (re-annotated MMLU, 57 subjects), scored as its authors score models

"""gbench native built-in runner for mmlu_redux (General Knowledge & Reasoning).

Dataset: MMLU-Redux-2.0 (`edinburgh-dawg/mmlu-redux-2.0`, pinned revision): 100 manually
re-annotated MMLU test questions for each of the 57 subjects (5,700 rows).

Protocol: the MMLU-Redux authors ship no answering harness (aryopg/mmlu-redux covers error
detection only). Their model re-evaluation (arXiv:2406.04127, Sec. 3.2, Table 2, Fig. 5 and
App. B; the dataset card's "Dataset Sources") scores the predictions of the HELM MMLU
leaderboard v1.3.0 by exact match on the "correct instances", the rows annotated
`error_type == "ok"`. So gbench runs HELM's MMLU protocol (stanford-crfm/helm `get_mmlu_spec`,
identical at release v0.5.1 ccf1df596b and at 63754d05) on those rows:
  * rows: `error_type == "ok"` only (5,330 at the pinned revision), keyed by the MMLU answer;
  * prompt: `multiple_choice_joint`, 5-shot. The instructions "The following are multiple
    choice questions (with answers) about <subject with spaces>.", then five in-context
    examples, then the question, each as "Question: <q>\\nA. ..\\nB. ..\\nC. ..\\nD. ..\\n"
    followed by "Answer: X\\n" (examples) or "Answer:" (the question), blocks joined by "\\n".
    The examples are the subject's MMLU dev questions (HELM's train split for MMLU), ordered by
    HELM's `InContextLearningAdapter.sample_examples` with seed 0, ported below;
  * request: the prompt as one user message, `max_tokens=1`, `temperature=0`, no stop
    sequences (MultipleChoiceJointAdapter.generate_requests);
  * scoring: exact match after HELM's output mapping: the stripped reply must be a letter
    whose option text equals the correct option's text (evaluate_reference_metrics);
  * headline: exact match over all scored rows combined (the paper's "all subjects combined"),
    with the per-subject mean (HELM's MMLU aggregate) alongside.

Token limit: HELM's `max_tokens=1` is the suite's official cap (base.OfficialSampling.max_tokens),
applied by base's token-limit policy (resolve_token_budget). A no-think request is sent
`max_tokens=1` exactly, and `--max-output-tokens` does not change it (`max_output_tokens_note`);
a reply that stops at that one token is counted in `replies_at_official_max_tokens`: the
protocol, not a truncation. A `--thinking` run applies thinking: the one-token cap was written for
non-reasoning generation, so each request is sent the run's budget instead (`--max-output-tokens`,
else the thinking floor), fitted to the context window, and the result records that the official
cap was not applied to reasoning. The rest of the request stays HELM's: HELM reaches a
vLLM-served model through VLLMChatClient (src/helm/clients/vllm_client.py, an OpenAIClient),
which sends the same request whether the model reasons or not. OpenAIClient's reasoning-model
branch (openai_client.py at 63754d05, lines 286-314: temperature and top_p dropped, max_tokens
renamed max_completion_tokens) is taken only for OpenAI model names (`is_reasoning_model`:
`^o\\d+` or `^gpt-5`) and exists to avoid parameters OpenAI's hosted reasoning models reject, so
it is not a reasoning-model setting for a served model. The temperature of a thinking request
is 1.0, not HELM's 0 (base.resolve_temperature: gbench's rule). The reply
is the message content (the server returns the reasoning separately, and any inline thought
block is removed), read with the same exact match. HELM's MMLU runs read a one-token answer
without reasoning, so a thinking run is flagged non-comparable.

A thinking reply is classified as base.run_eval_suite classifies it (base.classify_reply). One
cut off at the budget is scored on what arrived and counted (`truncated_responses`;
`replies_cut_off_in_reasoning` when no answer text arrived because the budget ran out during
the reasoning); one that stopped with its answer inside the thinking channel (reasoning, or a
content holding only a thought block, and no answer text) is the model's outcome, scores 0 and
is counted in `answer_in_thinking` (base's ANSWER_IN_THINKING_POLICY: not a failed request);
one that came back with nothing at all is counted in `empty_responses` and `failed_requests`
and scores 0. A cut-off or empty reply names itself in `leaderboard_comparable_reason`, and the
run reports `completed_with_errors` when more than 10% of the replies were cut off or any
request failed, as base.run_eval_suite does. A request that got no reply from the served
endpoint (failed, timed out, or a prompt over the context window) is a harness error, never a
wrong answer: it is left out of `accuracy`, `subject_mean_accuracy` and `category_accuracy`
(`harness_error` and `scoring_excluded` on its trace, `scoring_excluded` and
`request_harness_errors` in the result), counted in `failed_requests` and named in
`status_reason`; with no row scored, `accuracy` is null with `accuracy_withheld_reason`.

When more than 10% of the replies read are not a bare option letter (`non_letter_replies`;
every answered reply without thinking, every complete reply with it), the run logs a WARNING and
is flagged `leaderboard_comparable=False`: HELM's exact match still scores those replies 0, but
the number then measures how the served chat template (or, with thinking, the reply format after
the reasoning) lets the model give its answer rather than what the model knows.

Sharding (`--shard I/N`) selects shard I of the scored rows before the `--eval-limit` draw, as
base.run_eval_suite does, and the result records `shard`. Each trace records the row's
`sample_key` and its position among the scored rows (`global_idx`), so rows with identical
content stay apart; `gbench --merge` combines the shards through merge_shards, which counts the
union again by the run's own rules (_score_summary: counts, `subject_mean_accuracy`, the
non-comparable reasons with their 10% thresholds, the status), so the merged result of a full
run's shards equals the unsharded result.

Sampling: gbench's rule (base.resolve_temperature): temperature 1.0 with `--thinking`, 0.0
without. HELM decodes at temperature 0 (get_multiple_choice_adapter_spec), recorded as the
official value (base.OfficialSampling; the result records `official_sampling`). `--temperature` or
`GBENCH_MMLU_REDUX_TEMPERATURE` (which takes precedence over both) override it in both modes;
a value other than 0 makes the run non-comparable. LLM-judge grading is pinned
at 0.0 and is not affected by either (this suite has no judge).
"""

import asyncio
import logging
import os
import random
import re
import time
from collections import Counter, defaultdict
from itertools import cycle
from typing import Any, Dict, List, Optional, Tuple
from .limit_plan import eval_limit_plan, merge_plans
from .base import (OVER_CONTEXT_ERROR, OfficialSampling, Reply, _send_single_request,
                   answer_in_thinking_fields, apply_shard, clamp_to_context_async, classify_reply,
                   effective_chat_template_kwargs, finalize_shard_fields, get_run_knob,
                   record_shard, request_timeout_s, sample_key,
                   resolve_official_sampling, resolve_token_budget, strip_thinking_tags)
from .sampling import parse_shard, stratified_sample
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "General Knowledge & Reasoning"
DOCS_URL = "docs/evals/mmlu_redux.md"

DATASET = "edinburgh-dawg/mmlu-redux-2.0"
DATASET_REVISION = "372ea425445d51e1ba1188c56e5e893f8138621f"
#: HELM's MMLU in-context examples come from the MMLU dev split (5 per subject).
MMLU_DATASET = "cais/mmlu"
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"

OPTION_LETTERS = ["A", "B", "C", "D"]
#: get_multiple_choice_adapter_spec defaults used by get_mmlu_spec.
MAX_TRAIN_INSTANCES = 5
MAX_TOKENS = 1
TEMPERATURE = 0.0
#: run_expander.OutputFormatInstructions("mmlu"): what HELM's MMLU runs of instruction-tuned
#: models prepend (their run_spec.json from leaderboard v1.3.0 on; base models run without it).
OUTPUT_FORMAT_INSTRUCTION = "Answer with only a single letter."
#: HELM's first (and only) train trial.
TRAIN_TRIAL_SEED = 0
OFFICIAL_SAMPLING = OfficialSampling(
    reference=("stanford-crfm/helm get_mmlu_spec -> get_multiple_choice_adapter_spec "
               "(temperature=0.0, max_tokens=1), identical at v0.5.1 ccf1df596b and 63754d05"),
    temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
#: Share of answered replies that are not a bare option letter above which the run warns and
#: is flagged non-comparable: HELM reads only the first token, so a model that opens its reply
#: with anything else scores 0.
_NON_LETTER_WARN_SHARE = 0.1
#: Rows with `error_type == "ok"` at the pinned revision.
SCORED_ROWS = 5330
#: base.classify_reply health of a reply cut off at the output-token budget ...
_CUT_OFF = ("truncated", "looping")
#: ... and of one that came back with no answer text (empty_reasoned: the answer written inside
#: the thinking channel, the model's outcome counted in answer_in_thinking, base's
#: ANSWER_IN_THINKING_POLICY; the others counted in empty_responses).
_EMPTY = ("empty", "empty_reasoned", "output_discarded")
#: An inline thought block in a reply's content, closed or cut off before its closing tag.
_THOUGHT_BLOCK = re.compile(r"<(thought|reasoning)>.*?(?:</\1>|\Z)", re.DOTALL | re.IGNORECASE)

# The 57 subject configs of MMLU-Redux-2.0 (100 rows each, 5,700 total).
MMLU_REDUX_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics", "high_school_physics",
    "high_school_psychology", "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality", "international_law",
    "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing",
    "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations",
    "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions",
]


# --- HELM (stanford-crfm/helm), ported ---------------------------------------------------------
# An MMLU instance is a question with four references; MMLUScenario.process_csv tags every
# reference whose text equals the keyed answer's text as correct.

def _csv_text(value: Any) -> str:
    """A field as HELM reads it: MMLUScenario opens the Hendrycks CSVs in text mode, so
    universal newlines turn "\\r\\n" (kept verbatim in cais/mmlu) into "\\n"."""
    return str(value).replace("\r\n", "\n").replace("\r", "\n")


def _correct_indices(choices: List[str], answer: int) -> List[int]:
    return [i for i, c in enumerate(choices) if c == choices[answer]]


def sample_examples(all_train_instances: List[Dict[str, Any]], seed: int,
                    max_train_instances: int = MAX_TRAIN_INSTANCES) -> List[Dict[str, Any]]:
    """InContextLearningAdapter.sample_examples (sample_train=True), on {"label": ...} dicts.

    `label` is the text of the instance's first correct reference. HELM reseeds the module
    RNG with `random.seed(seed)`; a private `random.Random(seed)` draws the same sequence."""
    rng = random.Random(seed)
    num_instances_to_sample: int = min(len(all_train_instances), max_train_instances)

    examples: List[Dict[str, Any]] = []
    unlabeled_instances: List[Dict[str, Any]] = []
    label_to_instances: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for instance in all_train_instances:
        if instance.get("label") is not None:
            label_to_instances[instance["label"]].append(instance)
        else:
            unlabeled_instances.append(instance)

    # Build Instance counts to labels
    counts_to_labels: Dict[int, List[str]] = defaultdict(list)
    for label, instances in sorted(label_to_instances.items()):
        counts_to_labels[len(instances)].append(label)

    sorted_labels: List[str] = []
    # Sort the labels by the number of Instances that belong to them
    for count in sorted(counts_to_labels, reverse=True):
        labels: List[str] = counts_to_labels[count]
        # Break ties by randomly shuffling labels that have the same number of Instances
        rng.shuffle(labels)
        sorted_labels.extend(labels)

    labels_iterable = cycle(sorted_labels)
    while num_instances_to_sample > 0:
        next_label: Optional[str] = next(labels_iterable, None)
        if not next_label:
            break

        instances = label_to_instances[next_label]
        # If there are no Instances to sample for this particular label, skip it.
        if len(instances) == 0:
            continue

        # Randomly sample without replacement
        examples.append(instances.pop(rng.randrange(len(instances))))
        num_instances_to_sample -= 1

    # If we ran out of Instances with correct References, sample the rest from
    # the pool of Instances without any References
    examples += rng.sample(unlabeled_instances, num_instances_to_sample)
    return examples


def _example_block(question: str, choices: List[str], answer: Optional[int]) -> str:
    """MultipleChoiceJointAdapter.construct_example_prompt with get_mmlu_spec's adapter spec
    (input "Question: " + "\\n", references "A. " + "\\n", output "Answer: " + "\\n")."""
    result = "Question: " + question + "\n"
    for i, text in enumerate(choices):
        result += f"{OPTION_LETTERS[i]}. " + text + "\n"
    if answer is not None:
        result += "Answer: " + OPTION_LETTERS[_correct_indices(choices, answer)[0]] + "\n"
    else:
        result += "Answer:"
    return result


def _instructions(subject: str) -> str:
    """get_mmlu_spec instructions after format_instructions (a trailing newline), with the
    `output_format_instructions=mmlu` expander HELM applies to instruction-tuned models
    (OutputFormatInstructions.expand: "<instruction>\\n\\n<instructions>")."""
    return (OUTPUT_FORMAT_INSTRUCTION + "\n\n"
            + "The following are multiple choice questions (with answers) about "
            f"{subject.replace('_', ' ')}.\n")


def helm_prompt(subject: str, train_blocks: List[str], question: str,
                choices: List[str]) -> str:
    """Prompt.text: instructions, train blocks and the eval block joined by instance_prefix."""
    return "\n".join([_instructions(subject), *train_blocks,
                      _example_block(question, choices, None)])


def _exact_match(reply: Optional[str], choices: List[str],
                 answer: int) -> Tuple[bool, Optional[str]]:
    """compute_reference_metrics exact_match after the output mapping: (correct, letter read).

    The completion is stripped, mapped through {letter: option text}; a reply that is not a
    letter maps to None and scores 0; otherwise the mapped text must equal the correct text."""
    pred = (reply or "").strip()
    mapped = dict(zip(OPTION_LETTERS, choices)).get(pred)
    if not mapped:
        return False, (pred if pred in OPTION_LETTERS else None)
    return mapped.strip() == choices[answer].strip(), pred


# --- data ------------------------------------------------------------------------------------

def _dev_blocks() -> Dict[str, List[str]]:
    """Per subject, HELM's five in-context blocks: the MMLU dev questions in sample_examples'
    order. A load failure is a hard error (never zero-shot)."""
    try:
        from datasets import load_dataset
        dev = list(load_dataset(MMLU_DATASET, "all", split="dev", revision=MMLU_REVISION))
    except Exception as e:
        raise infra_required(
            "mmlu_redux", f"HELM's 5-shot examples come from the MMLU dev split, which could not "
            f"be loaded from {MMLU_DATASET} (revision {MMLU_REVISION[:8]}): {e}. Check Hugging "
            "Face Hub access.", DOCS_URL) from e
    by_subject: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in dev:
        choices = [_csv_text(c) for c in item.get("choices") or []]
        answer = int(item.get("answer"))
        by_subject[str(item.get("subject"))].append(
            {"question": _csv_text(item.get("question")), "choices": choices, "answer": answer,
             "label": choices[answer]})
    blocks = {}
    for subject in MMLU_REDUX_SUBJECTS:
        train = by_subject.get(subject) or []
        if len(train) < MAX_TRAIN_INSTANCES:
            raise infra_required(
                "mmlu_redux", f"the MMLU dev split has {len(train)} questions for '{subject}' "
                f"(HELM uses {MAX_TRAIN_INSTANCES}).", DOCS_URL)
        picked = sample_examples(train, seed=TRAIN_TRIAL_SEED)
        blocks[subject] = [_example_block(t["question"], t["choices"], t["answer"])
                           for t in picked]
    return blocks


def _load_mmlu_redux_items(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """The `error_type == "ok"` rows of MMLU-Redux-2.0 (all 57 subjects, pinned revision) with
    their HELM prompt. Raises on load or schema failure (no fabricated fallback)."""
    try:
        from datasets import load_dataset
        subject_rows = []
        for subject in MMLU_REDUX_SUBJECTS:
            ds = load_dataset(DATASET, subject, split="test", revision=DATASET_REVISION)
            # Each row keeps its position in its subject's pinned split: rows with identical
            # content stay distinct samples across shards (sample_key).
            for row_index, item in enumerate(ds):
                subject_rows.append((subject, item, row_index))
    except Exception as e:
        logger.error(f"Failed to load dataset for mmlu_redux: {e}")
        raise RuntimeError(f"Could not load dataset for mmlu_redux: {e}") from e

    if not subject_rows:
        raise RuntimeError("Dataset for mmlu_redux returned empty rows")

    # Filter BEFORE the stratified --eval-limit draw, so a subset holds only scored rows.
    kept, excluded = [], Counter()
    for subject, item, row_index in subject_rows:
        question, choices = item.get("question"), item.get("choices")
        if (not question or not isinstance(choices, list) or len(choices) != 4
                or not isinstance(item.get("answer"), int)):
            raise RuntimeError(
                "mmlu_redux: unexpected dataset schema "
                "(need 'question' str, 'choices' list of 4, 'answer' int index); "
                "refusing to fabricate sample data"
            )
        err = str(item.get("error_type") or "").strip()
        if err != "ok":
            excluded[err] += 1
            continue
        kept.append((subject, item, row_index))

    # --shard first, over the full ordered set of scored rows, then the --eval-limit draw. Each
    # row keeps its position in that set (global_idx).
    kept = [(subject, item, row_index, pos)
            for pos, (subject, item, row_index) in enumerate(kept)]
    kept, _ = apply_shard(kept, "mmlu_redux")
    # Stratified, not a contiguous head.
    kept = stratified_sample(kept, limit, lambda r: r[0], seed="mmlu_redux")
    blocks = _dev_blocks()
    items = []
    for subject, item, row_index, pos in kept:
        choices = [_csv_text(c) for c in item["choices"]]
        items.append({"subject": subject, "row_index": row_index, "global_idx": pos,
                      "choices": choices,
                      "answer": int(item["answer"]),
                      "prompt": helm_prompt(subject, blocks[subject],
                                            _csv_text(item["question"]), choices)})
    logger.info(f"Loaded {len(items)} mmlu_redux rows (error_type 'ok'); excluded "
                f"{sum(excluded.values())} annotated rows: {dict(excluded)}.")
    return items


def _load_mmlu_redux_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """The scored rows as (messages, gold letter, meta)."""
    return [([{"role": "user", "content": it["prompt"]}], OPTION_LETTERS[it["answer"]],
             {"category": it["subject"]})
            for it in _load_mmlu_redux_items(limit=limit)]


# --- run -------------------------------------------------------------------------------------

async def _generate_all(base_url: str, model_name: str, prompts: List[str], concurrency: int,
                        temperature: float, thinking: bool = False,
                        max_tokens: int = MAX_TOKENS,
                        sent: Optional[Dict[int, Optional[int]]] = None) -> List[Any]:
    """One reply per prompt. A no-think request is sent `max_tokens` (HELM's 1) as it is; a
    thinking request's budget is fitted to the context window first, as base.run_eval_suite fits
    it, and a prompt that cannot fit is not sent (an OVER_CONTEXT reply). `sent`, when given,
    receives each prompt's max_tokens (None for one not sent)."""
    import aiohttp
    api_url = f"{base_url.rstrip('/')}/chat/completions"
    semaphore = asyncio.Semaphore(max(1, concurrency))
    timeout = aiohttp.ClientTimeout(total=max(300, request_timeout_s(max_tokens)))
    ctk = effective_chat_template_kwargs(None, thinking)
    sent = sent if sent is not None else {}

    async def _one(session: Any, idx: int, prompt: str) -> Any:
        messages = [{"role": "user", "content": prompt}]
        budget: Optional[int] = max_tokens
        if thinking:
            budget, oversize = await clamp_to_context_async(messages, max_tokens, None, ctk)
            if oversize:
                sent[idx] = None
                return Reply(None, None, None, None, f"{OVER_CONTEXT_ERROR}: {oversize}")
        sent[idx] = budget
        return await _send_single_request(
            session=session, api_url=api_url, model_name=model_name, messages=messages,
            extra_payload={}, semaphore=semaphore, pbar=None, max_output_tokens=budget,
            temperature=temperature, thinking=thinking)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        return await asyncio.gather(*[_one(session, i, p) for i, p in enumerate(prompts)])


def _answer_text(content: Optional[str]) -> str:
    """A reply's content without its inline thought blocks (closed, or cut off before the
    closing tag): the answer text, empty when none arrived."""
    return _THOUGHT_BLOCK.sub("", content or "").strip()


def _reply_health(reply: Reply, thinking: bool) -> str:
    """base.classify_reply on the reply's answer text: with thinking, the content without inline
    thought blocks, so a content that held only a thought counts as an empty reply after the
    reasoning, like one whose reasoning the server returned separately."""
    if reply.text is None or not thinking:
        return classify_reply(reply)
    health = classify_reply(reply._replace(text=_answer_text(reply.text)))
    if health in ("empty", "output_discarded") and (reply.text or "").strip():
        return "empty_reasoned"
    return health


def _score_summary(traces: List[Dict[str, Any]], *, thinking: bool, max_tokens: int,
                   limit: Optional[int], temperature: float, temperature_source: str,
                   warn: bool = True) -> Tuple[Dict[str, Any], List[str], str, Optional[str]]:
    """The counts, headline, comparability reasons and status of a set of traces: a run's, or the
    union of a sharded run's (merge_shards), counted by the same rules. Returns (the result's
    count and score fields, the non-comparable reasons, status, status_reason). `warn` logs the
    run's warnings (not repeated by a merge)."""
    from .vqa_common import own_loop_status
    total = len(traces)
    failed = sum(1 for t in traces if t.get("scoring_excluded"))
    correct = sum(1 for t in traces if t.get("is_correct"))
    health_counts: Counter = Counter(t.get("health") for t in traces)
    stats: Dict[str, Dict[str, int]] = {}
    for t in traces:
        st = stats.setdefault(t["category"], {"correct": 0, "total": 0})
        if not t.get("scoring_excluded"):
            st["total"] += 1
            st["correct"] += 1 if t.get("is_correct") else 0
    answered = [t for t in traces if not t.get("scoring_excluded")]
    cut_off = [t for t in answered if t["health"] in _CUT_OFF]
    empty = [t for t in answered if t["health"] in _EMPTY and t["health"] != "empty_reasoned"]
    in_thinking = sum(1 for t in answered if t["health"] == "empty_reasoned")
    # The replies whose letter is read: every answered one without thinking (HELM scores the
    # one-token completion, empty or not); with thinking the complete ones (not cut off by the
    # budget, not empty), since the others are counted as budget or empty replies instead.
    read = [t for t in answered if t["status"] == "OK"]
    non_letter = sum(1 for t in read if t["predicted"] is None)
    cut_in_reasoning = sum(1 for t in cut_off if not _answer_text(t["response_text"]))
    # base.run_eval_suite's rule: an empty reply is a failed request, and replies cut off at a
    # budget that is not an official cap make the run completed_with_errors above 10%. Without
    # thinking the one-token stops are at HELM's official cap: the protocol's outcome.
    failures = failed + (len(empty) if thinking else 0)
    budget_cut = len(cut_off) if thinking else 0
    template_suspect = bool(read and non_letter > _NON_LETTER_WARN_SHARE * len(read))
    if warn and template_suspect and thinking:
        logger.warning(
            "[mmlu_redux] %d of %d complete replies are not a bare option letter (see "
            "non_letter_replies and the traces' response_text). HELM's exact match reads the "
            "whole reply after the reasoning, so these score 0; check that the server returns "
            "the reasoning separately (its reasoning parser) and that the model ends with the "
            "letter alone, as the prompt's first line asks.", non_letter, len(read))
    elif warn and template_suspect:
        logger.warning(
            "[mmlu_redux] %d of %d replies are not a bare option letter (see non_letter_replies "
            "and the traces' response_text). HELM's protocol reads one token, so these score 0; "
            "check that the served chat template, with thinking off, lets the model answer "
            "with the letter as its first token.", non_letter, len(read))
    looping = sum(1 for t in cut_off if t["health"] == "looping")
    if warn and budget_cut:
        logger.warning(
            "[mmlu_redux] %d/%d replies hit the output-token budget (max_tokens=%s) and were "
            "scored on what arrived: %d with no answer text (the budget ran out during the "
            "reasoning), %d repeating themselves. Raise --max-output-tokens before quoting "
            "this (more budget does not help the repeating ones).", budget_cut, total,
            ", ".join(str(v) for v in sorted({t["max_tokens_sent"] for t in cut_off})),
            cut_in_reasoning, looping)
    category_accuracy = {c: {**s, "accuracy": round(s["correct"] / s["total"] * 100.0, 2)}
                         for c, s in sorted(stats.items()) if s["total"]}
    macro = (round(sum(v["accuracy"] for v in category_accuracy.values())
                   / len(category_accuracy), 2) if category_accuracy else None)
    noncanon = []
    if thinking:
        noncanon.append(f"--thinking run: each request was sent max_tokens={max_tokens} for "
                        f"the reasoning and the answer (HELM's MMLU protocol reads a one-token "
                        f"answer, max_tokens={MAX_TOKENS}, which is not applied to reasoning)")
    if limit:
        noncanon.append(f"--eval-limit subset ({total} of the {SCORED_ROWS:,} scored rows)")
    if abs(float(temperature) - TEMPERATURE) > 1e-9:
        noncanon.append(f"temperature {temperature} ({temperature_source}; HELM decodes at 0)")
    if failed:
        noncanon.append(f"{failed} request(s) got no reply from the served endpoint (failed, "
                        "timed out or over the context window: harness errors, left out of the "
                        "score)")
    if thinking and empty:
        noncanon.append(f"{len(empty)} repl(ies) came back with no answer text and no "
                        "reasoning (empty_responses) and count as wrong")
    if budget_cut:
        noncanon.append(
            f"{budget_cut} of {total} replies were cut off at the max_tokens budget "
            f"(truncated_responses) and scored on what arrived, {cut_in_reasoning} of them with "
            "no answer text because the budget ran out during the reasoning "
            "(replies_cut_off_in_reasoning)")
    if template_suspect:
        shown = "complete replies" if thinking else "replies"
        cause = ("the format of the reply after the reasoning" if thinking else
                 "how the served chat template lets the model open its reply")
        noncanon.append(
            f"{non_letter} of {len(read)} {shown} (non_letter_replies) are not a bare option "
            f"letter, above the {_NON_LETTER_WARN_SHARE:.0%} warning threshold; HELM's exact "
            f"match scores them 0, so the number reflects {cause} rather than the model's "
            "knowledge")
    other_causes = []
    if thinking and empty:
        other_causes.append(f"empty_responses {len(empty)}")
    if budget_cut * 10 > total:
        other_causes.append(f"replies_cut_off_by_non_official_budget {budget_cut} of {total} "
                            "(over 10%)")
    status, status_reason, harness_kinds = own_loop_status(traces, other_causes)
    if total and failures == total and status != "failed":
        # base.run_eval_suite's rule: every request failed (no reply, or empty after thinking)
        status, status_reason = "failed", f"every request failed: {status_reason}"
    fields = {
        "total_questions": total,
        "correct_answers": correct,
        "failed_requests": failures,
        # the requests left out of the score (harness errors), by kind
        "scoring_excluded": failed,
        **({"request_harness_errors": harness_kinds} if harness_kinds else {}),
        "effective_n": len(answered),
        "empty_responses": len(empty),
        # scored as wrong (an empty answer), not failed requests (ANSWER_IN_THINKING_POLICY)
        **answer_in_thinking_fields(in_thinking, total),
        "truncated_responses": len(cut_off),
        "looping_responses": looping,
        **({"replies_cut_off_in_reasoning": cut_in_reasoning} if thinking else
           {"official_max_tokens_sent": [MAX_TOKENS],
            "replies_at_official_max_tokens": len(cut_off)}),
        "response_health": dict(sorted(health_counts.items())),
        "non_letter_replies": non_letter,
        # over the scored rows; None (withheld) when none is scored
        "accuracy": round(correct / len(answered) * 100.0, 2) if answered else None,
        **({"accuracy_withheld_reason": status_reason} if traces and not answered else {}),
        "subject_mean_accuracy": macro,
        "category_accuracy": category_accuracy,
    }
    return fields, noncanon, status, status_reason


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of a sharded run (`gbench --merge`): the union of the shards' rows (the
    generic merge), with every count, the headline, subject_mean_accuracy, the non-comparable
    reasons and the status counted again over that union by the run's own rules
    (_score_summary), so a threshold (the 10% non-letter warning, the 10% cut-off status) is
    judged on the full set, never on one shard. A reason no shard's own rows produce (one the
    eval runner stamped) is kept."""
    from ...merge import generic_merge, other_reasons
    merged = generic_merge("mmlu_redux", results)
    merged_plan = merge_plans(results)
    if merged_plan is not None:
        merged["eval_limit_plan"] = merged_plan
    live = [r for r in results if not r.get("empty_shard")] or list(results)
    first = live[0]
    sampling = first.get("official_sampling") or {}
    kw = dict(thinking=bool(first.get("thinking")),
              max_tokens=int(sampling.get("max_tokens_sent") or sampling.get("max_tokens")
                             or MAX_TOKENS),
              limit=first.get("eval_limit"),
              temperature=float(first.get("temperature", TEMPERATURE)),
              temperature_source=str(first.get("temperature_source") or ""), warn=False)
    fields, reasons, status, status_reason = _score_summary(merged.get("sample_traces") or [],
                                                            **kw)
    for r in live:
        own = _score_summary(r.get("sample_traces") or [], **kw)[1]
        reasons += [x for x in other_reasons(r, own) if x not in reasons]
    merged.pop("accuracy_withheld_reason", None)
    merged.update(fields)
    unreconciled = merged.get("merge_unreconciled_fields") or {}
    for key in fields:
        unreconciled.pop(key, None)
    if not unreconciled:
        merged.pop("merge_unreconciled_fields", None)
    merged["leaderboard_comparable"] = not reasons
    merged["leaderboard_comparable_reason"] = "; ".join(reasons) or None
    merged["status"], merged["status_reason"] = status, status_reason
    return merged


def run_mmlu_redux(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Execute mmlu_redux: HELM's 5-shot MMLU protocol on the error_type 'ok' rows."""
    limit = kwargs.get("limit")
    items = _load_mmlu_redux_items(limit=limit)
    if not items:
        raise RuntimeError("mmlu_redux: no rows to score")
    eval_plan = eval_limit_plan("mmlu_redux", limit, len(items))
    thinking = bool(enable_thinking)
    temperature, temperature_source = resolve_official_sampling("mmlu_redux", OFFICIAL_SAMPLING,
                                                                thinking=thinking)
    # The token-limit policy (base): HELM's max_tokens=1 exactly without thinking; with it the
    # run's budget, since the one-token cap was written for non-reasoning generation.
    plan = resolve_token_budget("mmlu_redux", OFFICIAL_SAMPLING, thinking=thinking,
                                max_output_tokens=(get_run_knob("max_output_tokens")
                                                   or kwargs.get("max_output_tokens")))
    max_tokens = int(plan.max_tokens)
    start = time.time()
    sent_budgets: Dict[int, Optional[int]] = {}
    # A no-think run sends HELM's request as it is (the defaults: no thinking, max_tokens=1); a
    # thinking run sends the run's budget, fitted to the window per prompt.
    thinking_args = (dict(thinking=True, max_tokens=max_tokens, sent=sent_budgets)
                     if thinking else {})
    replies = asyncio.run(_generate_all(base_url, model_name, [it["prompt"] for it in items],
                                        concurrency, temperature, **thinking_args))

    from .vqa_common import mark_no_reply
    traces = []
    for idx, (it, reply) in enumerate(zip(items, replies)):
        text = reply.text
        if text is not None and thinking:
            text = strip_thinking_tags(text)
        health = _reply_health(reply, thinking)
        if text is None:
            # No reply from the served endpoint: a harness error, left out of the score (never a
            # wrong answer), counted and named, as base.run_eval_suite does.
            ok, letter = False, None
            status = ("OVER_CONTEXT" if str(reply.error or "").startswith(OVER_CONTEXT_ERROR)
                      else "TIMEOUT" if classify_reply(reply) == "request_timeout"
                      else "FAILED")
        else:
            ok, letter = _exact_match(text, it["choices"], it["answer"])
            # Without thinking every reply is HELM's one-token completion (it stops at the
            # official max_tokens=1: the protocol). With thinking a reply cut off at the run's
            # budget or empty after the reasoning is counted apart, as base.run_eval_suite does.
            status = ("TRUNCATED" if thinking and health in _CUT_OFF else
                      "EMPTY_RESPONSE" if thinking and health in _EMPTY else "OK")
        traces.append({
            "sample_idx": idx, "category": it["subject"], "prompt": it["prompt"],
            # the row's id across shards (base.sample_key), for `gbench --merge`
            "sample_key": sample_key({"subject": it["subject"], "row_index": it.get("row_index"),
                                      "prompt": it["prompt"]}),
            # its position in the full ordered set of scored rows the shards partition
            "global_idx": it.get("global_idx"),
            "gold_answer": OPTION_LETTERS[it["answer"]], "response_text": reply.text,
            "reasoning": getattr(reply, "reasoning", None),
            "predicted": letter, "finish_reason": reply.finish_reason, "health": health,
            "is_correct": ok, "status": status, "error": reply.error,
            "max_tokens_sent": sent_budgets.get(idx, max_tokens),
        })
        if text is None:
            mark_no_reply(traces[-1], status, reply.error)
    summary, noncanon, status, status_reason = _score_summary(
        traces, thinking=thinking, max_tokens=max_tokens, limit=limit, temperature=temperature,
        temperature_source=temperature_source)
    attempts_requested = int(kwargs.get("attempt_count") or get_run_knob("attempt_count", 1) or 1)
    result = {
        "benchmark_type": "eval",
        "eval_name": "mmlu_redux",
        "eval_limit_plan": eval_plan,
        "model_name": model_name,
        "thinking": thinking,
        "thinking_requested": thinking,
        "thinking_applied": thinking,
        **summary,
        "scoring_mode": "deterministic",
        "metric": ("exact match over the error_type 'ok' rows of all subjects combined (HELM "
                   "MMLU 5-shot multiple_choice_joint, max_tokens=1); subject_mean_accuracy is "
                   "HELM's per-subject mean"),
        "n_shot": MAX_TRAIN_INSTANCES,
        "max_tokens_sent": sorted({t["max_tokens_sent"] for t in traces
                                   if t["max_tokens_sent"] is not None}) or None,
        "temperature": temperature,
        "temperature_source": temperature_source,
        "official_sampling": {**OFFICIAL_SAMPLING.record(temperature, temperature_source),
                              **plan.record()},
        "attempts": ({"attempts_requested": attempts_requested, "attempts_per_sample": 1,
                      "not_applied": "HELM scores one greedy completion per row"}
                     if attempts_requested > 1 else None),
        "leaderboard_comparable": not noncanon,
        "sample_traces": traces,
        # base.run_eval_suite's rule: a request with no reply or an empty reply, or more than
        # 10% cut off, is a measurement problem; nothing scored is `failed`
        "status": status,
        "status_reason": status_reason,
        "duration_s": round(time.time() - start, 2),
    }
    if plan.note:
        result["max_output_tokens_note"] = plan.note
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    shard = parse_shard(os.environ.get("GBENCH_SHARD"))
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(record_shard(result, f"{shard[0]}/{shard[1]}" if shard else None))
