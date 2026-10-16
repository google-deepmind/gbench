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

"""Native GPQA (Main 448-question science benchmark) evaluation suite, simple-evals protocol.

Prompt, repeats and scoring ported from openai/simple-evals (commit 652c89d0): `GPQAEval` with
`n_repeats=10` (simple_evals.py's default), i.e. the question set repeated 10 times, each copy
with a fresh option permutation drawn from one `random.Random(0)` in simple-evals' order; the
zero-shot CoT `QUERY_TEMPLATE_MULTICHOICE` (common.py) as the user message; and `gpqa_eval.py`'s
scoring, the first match of `ANSWER_PATTERN_MULTICHOICE` compared with the gold letter. The
headline is the mean score over all repeats. The user message is the same with and without
`--thinking`.

Sampling: simple-evals' chat sampler, which also shapes the request (simple_evals_messages and
SIMPLE_EVALS_SAMPLING, which base's token-limit policy applies per mode). Every request, with
and without `--thinking`, is shaped as simple_evals.py's `gpt-4o` entry sends it, a
`ChatCompletionSampler` (sampler/chat_completion_sampler.py) with
`system_message=OPENAI_SYSTEM_MESSAGE_API` and `max_tokens=2048`: the system message "You
are a helpful assistant." before the user message. The temperature follows gbench's rule
(base.resolve_temperature), not the sampler's default 0.5: 1.0 with `--thinking`, 0.0 without;
the sampler's 0.5 is recorded under `official_sampling` (a run at another value is
non-comparable). The 2048 is the sampler's official cap (OfficialSampling.max_tokens):
a no-think request is sent exactly 2048, and a reply cut off there is scored on what arrived, as
the sampler returns it (`replies_at_sampler_max_tokens` counts them; base reports them as
`replies_at_official_max_tokens`, the protocol's outcome rather than a truncation problem), and
`--max-output-tokens` is not applied (`max_output_tokens_note`). A `--thinking` request is sent
the run's budget (`--max-output-tokens`) instead, because the 2048 was written for non-reasoning
generation (`official_sampling` records it as not applied to reasoning, with
`max_tokens_sent`). simple_evals.py's reasoning-model entries (o3, o4-mini and o1-pro through
ResponsesSampler with `reasoning_model=True`; o1, o1-preview, o1-mini and o3-mini through
OChatCompletionSampler), which send no system message, no temperature and no max_tokens, are
keyed on OpenAI model names and fit OpenAI's hosted reasoning models, so they are not reasoning
settings for a served model: a served model takes the chat sampler in both modes, as HELM's and
lm-eval's o-series branches are not taken for a served model either (mmlu_redux, gsm8k). See
base.OfficialSampling; the result records `official_sampling`. A reply cut off before it
finished at anything but the 2048 of a no-think request (a thinking reply at the run's budget,
or a reply at a max_tokens the context clamp lowered) is scored on what arrived and makes the
run non-comparable; a no-think reply that stops at the 2048 sent is the sampler's outcome.
Override the temperature for a whole run with `--temperature`, or for this suite alone with
`GBENCH_GPQA_TEMPERATURE`, which takes precedence over both (and over the rule's 1.0 / 0.0); a
value other than 0.5 makes the run non-comparable. LLM-judge grading is pinned at 0.0 and is not
affected by either.

Null replies: simple-evals' sampler asks a reply whose `message.content` is None again (the same
request, no limit); base's requests go through a loopback relay that does the same
(swebench_common.upstream_reask_relay, up to NULL_REPLY_REASKS), so a reply with its answer
inside the thinking channel (null content) is asked again rather than scored wrong. The result
records `upstream_reask`; a reply still null after the re-asks makes the run non-comparable.

Sharding: `--shard I/N` selects shard I of the QUESTIONS (round-robin in dataset order) before
the `--eval-limit` draw, so every copy of a question runs in the same shard.
"""

import logging
import random
import re
from typing import Any, Dict, List, Optional, Tuple
from .arc_agi import _for_base   # lays out an already-sharded sample list for run_eval_suite
from .base import (OfficialSampling, apply_shard, finalize_shard_fields, is_scored,
                   run_eval_suite)
from .limit_plan import eval_limit_plan
from .sampling import limit_dataset
from .swebench_common import (merge_reask_summaries, reask_divergence_reasons, reask_summary,
                              reply_content_is_null, upstream_reask_relay)

logger = logging.getLogger(__name__)

#: simple-evals' ChatCompletionSampler asks a reply whose `message.content` is None again (it
#: raises "OpenAI API returned empty response; retrying" inside its retry loop, with no limit).
#: gbench re-asks such a reply, the same request, up to this many times
#: (swebench_common.upstream_reask_relay), so a run ends; a reply still null after them is scored
#: as base scores it and makes the run non-comparable.
NULL_REPLY_REASKS = 5
SIMPLE_EVALS_REASK_RULE = ("simple-evals ChatCompletionSampler: a reply whose message.content "
                           "is None is asked again (the same request)")

DATASET = "Idavidrein/gpqa"
DATASET_REVISION = "83022cefff930aea54f654c0b282e74b9eeda5c6"

# openai/simple-evals common.py, verbatim.
QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

ANSWER_PATTERN_MULTICHOICE = r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?"

OPTION_LETTERS = "ABCD"

#: simple_evals.py: `GPQAEval(n_repeats=... args.n_repeats or 10)`.
N_REPEATS = 10

#: sampler/chat_completion_sampler.py OPENAI_SYSTEM_MESSAGE_API, verbatim: the system message of
#: simple_evals.py's `gpt-4o` entry (and of its other current OpenAI chat entries, gpt-4.1 and
#: gpt-4o-mini among them). ChatCompletionSampler.__call__ prepends it as a "system" message.
SIMPLE_EVALS_SYSTEM_MESSAGE = "You are a helpful assistant."
#: simple_evals.py `gpt-4o` entry: ChatCompletionSampler(..., max_tokens=2048).
SIMPLE_EVALS_CHAT_MAX_TOKENS = 2048

#: simple-evals' chat sampler (shared with gpqa_diamond, multilingual_mmlu, simpleqa,
#: healthbench and browsecomp), the path a served model takes in both modes.
SIMPLE_EVALS_CHAT_SAMPLING = OfficialSampling(
    reference=("openai/simple-evals@652c89d0 simple_evals.py 'gpt-4o' entry: "
               "ChatCompletionSampler (sampler/chat_completion_sampler.py) with "
               "system_message=OPENAI_SYSTEM_MESSAGE_API, max_tokens=2048 and the sampler's "
               "default temperature=0.5"),
    temperature=0.5)
#: The chat sampler with its official cap, as one declaration for base's token-limit policy
#: (resolve_token_budget): max_tokens=2048 sent exactly on a no-think request, the run's budget
#: on a --thinking request (the cap is not applied to reasoning). No `reasoning` block:
#: simple-evals' o-series samplers are keyed on OpenAI model names and are not reasoning
#: settings for a served model. gpqa, gpqa_diamond, multilingual_mmlu, simpleqa, healthbench and
#: browsecomp pass it whole to run_eval_suite.
SIMPLE_EVALS_SAMPLING = SIMPLE_EVALS_CHAT_SAMPLING._replace(
    max_tokens=SIMPLE_EVALS_CHAT_MAX_TOKENS)


def simple_evals_messages(prompt_messages: List[Dict[str, Any]],
                          thinking: bool = False) -> List[Dict[str, Any]]:
    """The messages simple-evals' chat sampler sends for an eval's `prompt_messages`: its
    system message first (ChatCompletionSampler.__call__), the same in both modes."""
    return [{"role": "system", "content": SIMPLE_EVALS_SYSTEM_MESSAGE}, *prompt_messages]


def _permutations(n_examples: int, n_repeats: int) -> List[List[int]]:
    """GPQAEval.__init__ (num_examples=None): `rng = random.Random(0)`, the examples repeated
    `n_repeats` times, then one `rng.sample(range(4), 4)` per copy, in that order."""
    rng = random.Random(0)
    return [rng.sample(range(4), 4) for _ in range(n_examples * n_repeats)]


def _gpqa_sample(item: Dict[str, Any], permutation: List[int], thinking: bool = False
                 ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """GPQAEval.__call__'s prompt and gold letter for one row and its permutation, as the
    sampler sends it (simple_evals_messages)."""
    choices = [
        item["Correct Answer"],
        item["Incorrect Answer 1"],
        item["Incorrect Answer 2"],
        item["Incorrect Answer 3"],
    ]
    choices = [choices[i] for i in permutation]
    correct_index = choices.index(item["Correct Answer"])
    correct_answer = OPTION_LETTERS[correct_index]
    choices_dict = dict(
        A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=item["Question"]
    )
    prompt = QUERY_TEMPLATE_MULTICHOICE.format(**choices_dict)
    category = str(item.get("Subdomain") or item.get("High-level domain") or "")
    return (simple_evals_messages([{"role": "user", "content": prompt}], thinking),
            correct_answer, {"category": category})


def _gpqa_items(rows: List[Dict[str, Any]], n_repeats: int, limit: Optional[int],
                seed: str, thinking: bool = False) -> List[Dict[str, Any]]:
    """Every (question, repeat) copy of this shard's questions, with simple-evals' permutation.

    Copy i is question i % N in repeat i // N, exactly as `examples * n_repeats` orders them,
    so each copy's permutation is the one a full simple-evals run gives it. `--shard I/N`
    (GBENCH_SHARD) first keeps shard I of the QUESTIONS (round-robin in dataset order); `limit`
    then keeps a stratified subset of those questions (by Subdomain). Either way a kept
    question keeps all of its repeats."""
    n = len(rows)
    questions, _ = apply_shard(list(range(n)), seed)
    if limit:
        tagged = [dict(rows[q], _gbench_qidx=q) for q in questions]
        questions = [r["_gbench_qidx"] for r in limit_dataset(tagged, limit, "Subdomain",
                                                               seed=seed)]
    keep = set(questions)
    items = []
    for i, perm in enumerate(_permutations(n, n_repeats)):
        q = i % n
        if q not in keep:
            continue
        items.append({"question_idx": q, "repeat_idx": i // n, "permutation": perm,
                      "sample": _gpqa_sample(rows[q], perm, thinking)})
    return items


def _load_gpqa_rows(config: str) -> List[Dict[str, Any]]:
    from datasets import load_dataset
    ds = load_dataset(DATASET, config, split="train", revision=DATASET_REVISION)
    rows = [dict(r) for r in ds]
    logger.info(f"Loaded {len(rows)} GPQA ({config}) questions from HF Hub ('{DATASET}').")
    return rows


def _load_gpqa_items(config: str, limit: Optional[int], n_repeats: int,
                     seed: str, thinking: bool = False) -> List[Dict[str, Any]]:
    return _gpqa_items(_load_gpqa_rows(config), n_repeats, limit, seed, thinking)


def _load_gpqa_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
    n_repeats: int = N_REPEATS,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """GPQA Main ('Idavidrein/gpqa', 'gpqa_main', pinned revision) as simple-evals requests,
    one per (question, repeat) copy, each carrying the chat sampler's system message (the same
    in both modes)."""
    return [it["sample"] for it in _load_gpqa_items("gpqa_main", limit, n_repeats, "gpqa",
                                                     enable_thinking)]


def _eval_gpqa(response_text: str, gold_letter: str) -> bool:
    """simple-evals scoring: the FIRST `Answer: X` match, compared case-sensitively.

    The pattern's `(?i)` lets a lower-case letter match ("Answer: because ..." reads "b"),
    but "b" never equals the gold "B", so such a reply scores 0 rather than being credited.
    """
    match = re.search(ANSWER_PATTERN_MULTICHOICE, str(response_text or ""))
    extracted_answer = match.group(1) if match else None
    return extracted_answer == gold_letter


def _flag_comparability(result: Dict[str, Any], limit: Optional[int], scored: int,
                        total: int, n_repeats: Optional[int] = None,
                        thinking: bool = False) -> Dict[str, Any]:
    """simple-evals' protocol over the whole set, with every request answered, is comparable.

    simple-evals' chat sampler asks a failed request (and one whose content is null) again until
    it succeeds, so a request gbench could not complete makes the run a different measurement:
    one that got no reply from the served endpoint (failed, timed out, over the context window)
    is a harness error that base leaves out of the score. GPQA and GPQA-Diamond re-ask a
    null-content reply as the sampler does (`upstream_reask`, NULL_REPLY_REASKS): only a reply
    still null after those re-asks is a difference. A suite without that relay
    (multilingual_mmlu) scores a reply that came back empty wrong (gbench cannot tell a null
    reply from an empty one) and names it. A temperature other
    than the sampler's 0.5 (`official_sampling`; gbench samples at 1.0 with --thinking and 0.0
    without) is one too, and so
    is a no-think request sent a
    max_tokens other than the chat sampler's 2048 (base sends the context-clamped value when the
    prompt plus 2048 does not fit the window). A thinking request is sent the run's budget, the
    2048 not being applied to reasoning; `max_output_tokens_note` comes from base's token-limit
    policy. A reply cut off before it finished at anything but the 2048 of a no-think request (a
    thinking reply at the run's budget or at a max_tokens the context clamp lowered, or a
    no-think reply at a clamped max_tokens) is one too: base counts it in `truncated_responses`
    apart from `replies_at_official_max_tokens`. `sampler_max_tokens` is the 2048 on a no-think
    run and None on a thinking run, where it is not applied. Shared with gpqa_diamond and
    multilingual_mmlu (which has no repeats: `n_repeats` None)."""
    reasons = []
    official = result.get("official_sampling") or {}
    if official.get("temperature_matches_official") is False:
        reasons.append(f"temperature {result.get('temperature')} "
                       f"({result.get('temperature_source')}; simple-evals' sampler sends "
                       f"temperature {official.get('temperature')})")
    if limit:
        reasons.append(f"--eval-limit subset ({scored} of {total:,} questions)")
    if n_repeats is not None and n_repeats != N_REPEATS:
        reasons.append(f"{n_repeats} repeats per question (simple-evals averages {N_REPEATS})")
    traces = result.get("sample_traces") or []
    # base's failed_requests counts the requests with no reply (harness errors) and the empty
    # replies; the traces carry `harness_error` on the former.
    empty = int(result.get("empty_responses") or 0)
    no_reply = max(sum(1 for t in traces if t.get("harness_error")),
                   int(result.get("failed_requests") or 0) - empty,
                   int(result.get("timed_out_requests") or 0)
                   + int(result.get("over_context_prompts") or 0))
    if no_reply:
        reasons.append(f"{no_reply} request(s) got no reply from the served endpoint (failed, "
                       "timed out or over the context window: harness errors, left out of the "
                       "score; simple-evals' samplers ask a failed request again)")
    reask = result.get("upstream_reask")
    if isinstance(reask, dict):
        # The sampler's re-ask was applied (upstream_reask_relay): a null-content reply (an
        # answer inside the thinking channel among them) was asked again, so what base scored is
        # what the sampler returns; an empty-string reply is a reply to the sampler too. Only a
        # reply still null after gbench's NULL_REPLY_REASKS differs from upstream.
        reasons.extend(reask_divergence_reasons(
            reask, "null content", "as base scores them",
            "simple-evals' sampler asks a null reply again without limit"))
    else:
        if empty:
            reasons.append(f"{empty} request(s) came back empty and were scored wrong "
                           "(simple-evals' samplers ask a null reply again; gbench cannot tell "
                           "a null reply from an empty one)")
        # A reply with its answer inside the thinking channel is the model's outcome (base's
        # ANSWER_IN_THINKING_POLICY), but its content is null, which simple-evals' samplers
        # ask again.
        in_thinking = int(result.get("answer_in_thinking") or 0)
        if in_thinking:
            reasons.append(f"{in_thinking} repl(ies) stopped with the answer inside the thinking "
                           "channel (null content) and were scored wrong (simple-evals' samplers "
                           "ask a null reply again)")
    # Replies cut off before they finished, other than the no-think ones that stopped at the
    # chat sampler's 2048 (the sampler's outcome, scored on what arrived as it returns them).
    cut_off = (int(result.get("truncated_responses") or 0)
               - int(result.get("replies_at_official_max_tokens") or 0))
    if cut_off > 0:
        where = ("stopped at the run's token budget (--max-output-tokens, or a max_tokens the "
                 "context clamp lowered; the chat sampler's 2048 is not applied to reasoning)"
                 if thinking else
                 "stopped at a max_tokens the context clamp set below the chat sampler's 2048")
        reasons.append(f"{cut_off} repl(ies) {where} before finishing and were scored on what "
                       "arrived")
    cap = None if thinking else SIMPLE_EVALS_CHAT_MAX_TOKENS
    result["sampler_max_tokens"] = cap
    result["replies_at_sampler_max_tokens"] = (
        sum(1 for t in traces if t.get("finish_reason") == "length") if cap else 0)
    if cap:
        other = sorted({t.get("max_tokens_sent") for t in traces
                        if t.get("max_tokens_sent") not in (None, cap)})
        if other:
            reasons.append(f"some no-think requests were sent max_tokens {other} (the prompt "
                           f"plus the chat sampler's {cap} did not fit the context window)")
    result["leaderboard_comparable"] = not reasons
    if reasons:
        result["leaderboard_comparable_reason"] = "; ".join(reasons)
    else:
        result.pop("leaderboard_comparable_reason", None)
    # A result that covers one shard of a --shard I/N run (`shard`, recorded by base) is a partial
    # set: the shard reason is added here too, so the rewrite above never drops it.
    return finalize_shard_fields(result)


def _run_simple_evals_gpqa(eval_name: str, config: str, total: int, model_name: str,
                           base_url: str, concurrency: int, enable_thinking: bool,
                           **kwargs) -> Dict[str, Any]:
    """GPQAEval for one variant: generate every copy, score it, average over all copies.

    The repeat count is simple-evals' 10; a larger `--attempt-count` raises it (each extra
    repeat gets its own fresh permutation) and is flagged. Every copy is a distinct prompt, so
    base's own attempt expansion is not used."""
    limit = kwargs.get("limit")
    n_repeats = max(N_REPEATS, int(kwargs.get("attempt_count") or 1))
    # This shard's questions (all of their copies), then the --eval-limit draw among them.
    items = _load_gpqa_items(config, limit, n_repeats, eval_name, enable_thinking)
    samples = [it["sample"] for it in items]
    # The samples are the questions; the n_repeats copies of each are its attempts.
    plan = eval_limit_plan(eval_name, limit, len({it["question_idx"] for it in items}),
                           attempts=n_repeats)
    # simple-evals' sampler asks a null-content reply again: base's requests go through a relay
    # that does the same (NULL_REPLY_REASKS), so base scores the reply the sampler would return.
    with upstream_reask_relay(eval_name, base_url, reply_content_is_null,
                              NULL_REPLY_REASKS) as relay:
        result = run_eval_suite(
            eval_name=eval_name,
            model_name=model_name,
            base_url=relay.url,
            concurrency=concurrency,
            samples=_for_base(samples),   # already this shard's copies; base keeps exactly them
            eval_fn=_eval_gpqa,
            thinking=enable_thinking,
            extra_payload=kwargs.get("extra_payload"),
            limit=None,            # the question subset is drawn above, with all its repeats
            max_output_tokens=kwargs.get("max_output_tokens"),
            attempt_count=1,
            official_sampling=SIMPLE_EVALS_SAMPLING,   # per mode, cap included (token-limit policy)
        )
    result["upstream_reask"] = reask_summary(relay, SIMPLE_EVALS_REASK_RULE)
    traces = result.get("sample_traces") or []
    per_repeat: Dict[int, List[bool]] = {}
    for t in traces:
        idx = t.get("sample_idx")
        if isinstance(idx, int) and 0 <= idx < len(items):
            it = items[idx]
            t["source_sample_idx"] = it["question_idx"]
            t["attempt_idx"] = it["repeat_idx"]
            t["permutation"] = it["permutation"]
            if is_scored(t):
                per_repeat.setdefault(it["repeat_idx"], []).append(bool(t.get("is_correct")))
    # Over the scored copies, as base's accuracy: a copy with no reply from the endpoint is a
    # harness error, left out.
    scores = [1.0 if t.get("is_correct") else 0.0 for t in traces if is_scored(t)]
    mean = sum(scores) / len(scores) if scores else 0.0
    questions = len({it["question_idx"] for it in items})
    result["n_repeats"] = n_repeats
    result["repeats"] = {
        "n_repeats": n_repeats,
        "questions": questions,
        "generations": len(items),
        "per_repeat_accuracy": [round(sum(v) / len(v) * 100.0, 2)
                                for _, v in sorted(per_repeat.items()) if v],
        # simple-evals aggregate_results "score:std": np.std over the per-copy 0/1 scores.
        "score_std": round((sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5, 4)
        if scores else None,
    }
    result["metric"] = (f"accuracy: mean score over {n_repeats} repeats of the question set, "
                        "each with a fresh option permutation (simple-evals GPQAEval)")
    result["eval_limit_plan"] = plan
    return _flag_comparability(result, limit, questions, total, n_repeats,
                               thinking=enable_thinking)


def _repeats_report(traces: List[Dict[str, Any]], n_repeats: int) -> Dict[str, Any]:
    """The `repeats` block over `traces` (each carries its question's index in
    `source_sample_idx` and its repeat in `attempt_idx`): per-repeat accuracy and simple-evals'
    score:std over the scored copies."""
    per_repeat: Dict[int, List[bool]] = {}
    scores = []
    for t in traces:
        if is_scored(t):
            repeat = int(t.get("attempt_idx") or 0)
            per_repeat.setdefault(repeat, []).append(bool(t.get("is_correct")))
            scores.append(1.0 if t.get("is_correct") else 0.0)
    mean = sum(scores) / len(scores) if scores else 0.0
    return {
        "n_repeats": n_repeats,
        "questions": len({t.get("source_sample_idx") for t in traces}),
        "generations": len(traces),
        "per_repeat_accuracy": [round(sum(v) / len(v) * 100.0, 2)
                                for _, v in sorted(per_repeat.items()) if v],
        "score_std": round((sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5, 4)
        if scores else None,
    }


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of a sharded GPQA / GPQA-Diamond run (`gbench --merge`): the generic
    merge (the union of the shards' copies re-counted; the accuracy is correct over scored
    copies, as each shard's is) with the `repeats` block recomputed over the union."""
    from ...merge import generic_merge
    merged = generic_merge(str(results[0].get("eval_name") or "gpqa"), results)
    n_repeats = int((merged.get("repeats") or {}).get("n_repeats") or merged.get("n_repeats")
                    or N_REPEATS)
    merged["repeats"] = _repeats_report(merged.get("sample_traces") or [], n_repeats)
    reasks = [r["upstream_reask"] for r in results if isinstance(r.get("upstream_reask"), dict)]
    if reasks:
        # every shard's re-asks of null-content replies, added
        merged["upstream_reask"] = merge_reask_summaries(reasks)
    return merged


def run_gpqa(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native GPQA Main evaluation suite (simple-evals GPQAEval, 10 repeats)."""
    return _run_simple_evals_gpqa("gpqa", "gpqa_main", 448, model_name, base_url, concurrency,
                                  enable_thinking, **kwargs)
