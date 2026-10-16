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

# Evaluation Suite: docvqa
# Description: DocVQA (Document Visual Question Answering Benchmark)

"""Native DocVQA (Document Visual Question Answering) evaluation suite.

Runs lmms-eval's `docvqa_val` task (EvolvingLMMs-Lab/lmms-eval
`lmms_eval/tasks/docvqa/_default_template_docvqa_yaml`, `docvqa_val.yaml` and
`utils.docvqa_doc_to_text` @ 42296f8fd2f0929d88a6ac4f7a1f2d784537f8d8): the DocVQA validation
split of `lmms-lab-encoder/DocVQA` (config `DocVQA`, pinned below; 5,349 questions, the split
with public answers), the document image followed by `{question}` and the `default` model
kwargs' post_prompt "\\nAnswer the question using a single word or phrase." The headline is
mean ANLS (threshold 0.5), the DocVQA metric lmms-eval aggregates.

The reply is scored as it comes back, with no answer extraction: the task's metric is lmms-eval's
`anls` (lmms_eval/api/metrics.py; vqa_common.lmms_anls), given results[0] stripped of surrounding
whitespace by lmms-eval's task layer.

Sampling: base.OfficialSampling from vqa_common.lmms_eval_sampling: the task's generation_kwargs
as lmms-eval's chat-endpoint client sends them, temperature 0 (recorded; the client runs a
thinking model with enable_thinking_kwarg, which keeps the task's sampling). The temperature sent
follows gbench's rule (base.resolve_temperature): 1.0 with `--thinking`, 0.0 without.
Without `--thinking` every request carries max_tokens 32 (max_new_tokens), so
`--max-output-tokens` is not applied and a reply that stops at 32 tokens is scored as it arrived,
as lmms-eval scores it (`replies_at_official_max_tokens` counts them). lmms-eval declares no
reasoning-model cap, so with `--thinking` the 32, written for non-reasoning generation, is not
applied: each request is sent the run's `--max-output-tokens`, else the thinking floor
(base.resolve_token_budget; `official_sampling.max_tokens_applied` false and the cap recorded as
not applied to reasoning). Override the temperature for a whole run with `--temperature`, or for
this suite alone with `GBENCH_DOCVQA_TEMPERATURE`, which takes precedence over both; the
result records `official_sampling`. The result is leaderboard-comparable for a full, unsharded
run at the official temperature with every request answered and no reply cut off by the run's
token budget (`replies_cut_off_by_budget`).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from .limit_plan import base_attempts, eval_limit_plan
from .base import apply_shard, finalize_shard_fields, run_eval_suite
from .sampling import limit_dataset, presharded_for_base
from .dataset_utils import extract_lossless_image_b64
from .metrics import finalize_mean_metric, record_budget_cutoffs
from .vqa_common import lmms_eval_sampling
from .charxiv import image_mime   # the MIME type the bytes carry (PNG, GIF, WebP, else JPEG)

logger = logging.getLogger(__name__)

DATASET = "lmms-lab-encoder/DocVQA"
CONFIG = "DocVQA"
DATASET_REVISION = "539088ef8a8ada01ac8e2e6d4e372586748a265e"
SPLIT = "validation"
#: lmms-eval docvqa `default` post_prompt (the pre_prompt is empty), verbatim.
POST_PROMPT = "\nAnswer the question using a single word or phrase."
N_QUESTIONS = 5349
#: docvqa/_default_template_docvqa_yaml generation_kwargs: max_new_tokens 32, temperature 0,
#: do_sample False.
UPSTREAM_MAX_NEW_TOKENS = 32
OFFICIAL_SAMPLING = lmms_eval_sampling(
    "lmms-eval tasks/docvqa/_default_template_docvqa_yaml @42296f8f", UPSTREAM_MAX_NEW_TOKENS)


def _load_docvqa_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load the pinned DocVQA validation split with the original image bytes (no re-encoding)."""
    from datasets import load_dataset, Image

    try:
        ds = load_dataset(DATASET, CONFIG, split=SPLIT, revision=DATASET_REVISION)
    except Exception as e:
        # NO silent fallback to hf-internal-testing/fixtures_docvqa: that is a 5-sample unit
        # test fixture, and reporting a score over it under the name 'docvqa' is a fabricated
        # headline. Fail loudly instead.
        raise RuntimeError(
            f"docvqa: could not load {DATASET} ({CONFIG}, {SPLIT}, revision {DATASET_REVISION}) "
            f"({e}); refusing to fall back to any other dataset and report it as 'docvqa'.") from e
    if hasattr(ds, "cast_column"):
        ds = ds.cast_column("image", Image(decode=False))

    # --shard first (by row index over the full ordered split, before any image is read),
    # then the --eval-limit draw within the shard.
    keep, shard_label = apply_shard(list(range(len(ds))), "docvqa")
    if shard_label:
        ds = ds.select(keep) if hasattr(ds, "select") else [ds[i] for i in keep]
    # Stratified, not a contiguous head.
    ds = limit_dataset(ds, limit, None, seed="docvqa")
    samples = []
    for item in ds:
        b64_str = extract_lossless_image_b64(item.get("image"))
        if not b64_str:
            # A question without its document cannot be answered; dropping it would shrink the
            # denominator without saying so.
            raise RuntimeError(f"docvqa: question {item.get('questionId')!r} has no decodable image")
        # docvqa_doc_to_text: pre_prompt + question + post_prompt.
        prompt = f"{item.get('question', '')}{POST_PROMPT}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{image_mime(b64_str)};base64,{b64_str}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        question_types = list(item.get("question_types") or [])
        samples.append((messages, list(item.get("answers") or []),
                        {"category": question_types[0] if question_types else "document"}))

    logger.info(f"Loaded {len(samples)} DocVQA samples with direct lossless image bytes.")
    return samples


def _eval_docvqa(response_text: str, gold_answers: Any):
    """Binary form of lmms-eval's anls (a non-zero ANLS) on the reply as returned."""
    from .vqa_common import lmms_anls
    if gold_answers is None:
        return False
    return lmms_anls(response_text, gold_answers) > 0.0


def run_docvqa(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native DocVQA document understanding evaluation suite."""
    from .vqa_common import lmms_anls, lmms_temperature_reason, lost_requests_reason
    samples = _load_docvqa_samples(limit=kwargs.get("limit"))
    eval_plan = eval_limit_plan(
        "docvqa", kwargs.get("limit"), len(samples), attempts=base_attempts())
    result = run_eval_suite(
        eval_name="docvqa",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's questions, kept as is
        eval_fn=_eval_docvqa,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=OFFICIAL_SAMPLING,   # its max_tokens follows the token-limit policy
    )
    result["eval_limit_plan"] = eval_plan
    # Canonical DocVQA headline is MEAN ANLS (the continuous per-question score), not the
    # share clearing the 0.5 threshold. An empty reply scores 0 ANLS and stays in the mean; a
    # request that got no reply from the endpoint is a harness error that base marks
    # scoring_excluded, left out of the mean like base's accuracy.
    result = finalize_mean_metric(
        result,
        metric_name="mean ANLS of the reply as returned (lmms-eval docvqa_val anls)",
        scorer=lambda t: lmms_anls(t.get("response_text"), t.get("gold_answer")),
        stash_key="anls",
        secondary_key="pass_rate_at_0.5",
        failure_score=0.0,
        # the category rows and the Correct count say mean ANLS too (the binary tally
        # moves to pass_rate_at_0.5 / pass_count)
        category_scores=True,
    )
    reasons = []
    if kwargs.get("limit"):
        reasons.append(f"--eval-limit {kwargs['limit']} subset of the {N_QUESTIONS:,} questions")
    temperature_reason = lmms_temperature_reason(result)
    if temperature_reason:
        reasons.append(temperature_reason)
    result["dataset"] = f"{DATASET}@{DATASET_REVISION} ({CONFIG}, {SPLIT})"
    lost = lost_requests_reason(result.get("sample_traces") or [])
    if lost:
        reasons.append(lost)
    cut = record_budget_cutoffs(result)
    if cut:
        reasons.append(cut)
    result["leaderboard_comparable"] = not reasons
    result["leaderboard_comparable_reason"] = "; ".join(reasons) or None
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields,
    # which `gbench --merge` drops once the shards' union is the full set).
    return finalize_shard_fields(result)
