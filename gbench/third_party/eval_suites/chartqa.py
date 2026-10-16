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

# Evaluation Suite: chartqa
# Description: ChartQA (Plot and Chart Visual Reasoning Benchmark)

"""Native ChartQA (Plot and Visual Reasoning) evaluation suite.

Runs the full ChartQA test split (`ahmed-masry/ChartQA`, pinned below): 2,500 questions,
1,250 human-written and 1,250 machine-generated ("augmented"). Scoring is ChartQA's relaxed
accuracy exactly as pix2struct implements it (`vqa_common.relaxed_correctness`). The
headline is relaxed accuracy over all 2,500 questions, which equals the mean of the two
halves; the per-half numbers (ChartQA-H / ChartQA-A in the paper) are in
`category_accuracy` under `human` and `augmented`.

The prompt, sampling and scoring are lmms-eval's `chartqa` task (EvolvingLMMs-Lab/lmms-eval
`lmms_eval/tasks/chartqa/chartqa.yaml` + `utils.py` @ 42296f8fd2f0929d88a6ac4f7a1f2d784537f8d8, the
`default` model kwargs): the chart, then `{question}` followed by "\nAnswer the question with a
single word."; the reply is scored as it comes back (chartqa_process_results passes results[0],
stripped of surrounding whitespace by lmms-eval's task layer, to relaxed_correctness), with no
answer extraction.

Sampling: base.OfficialSampling from vqa_common.lmms_eval_sampling: the task's generation_kwargs
as lmms-eval's chat-endpoint client sends them, temperature 0 (recorded; the client runs a
thinking model with enable_thinking_kwarg, which keeps the task's sampling). The temperature sent
follows gbench's rule (base.resolve_temperature): 1.0 with `--thinking`, 0.0 without.
Without `--thinking` every request carries max_tokens 16 (max_new_tokens), so
`--max-output-tokens` is not applied and a reply that stops at 16 tokens is scored as it arrived,
as lmms-eval scores it (`replies_at_official_max_tokens` counts them). lmms-eval declares no
reasoning-model cap, so with `--thinking` the 16, written for non-reasoning generation, is not
applied: each request is sent the run's `--max-output-tokens`, else the thinking floor
(base.resolve_token_budget; `official_sampling.max_tokens_applied` false and the cap recorded as
not applied to reasoning). Override the temperature for a whole run with `--temperature`, or for
this suite alone with `GBENCH_CHARTQA_TEMPERATURE`, which takes precedence over both; the result
records `official_sampling`. The result is leaderboard-comparable for a full, unsharded run at the
official temperature with every request answered and no reply cut off by the run's token budget
(`replies_cut_off_by_budget`).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from .limit_plan import base_attempts, eval_limit_plan
from .base import apply_shard, finalize_shard_fields, run_eval_suite
from .sampling import limit_dataset, presharded_for_base
from .dataset_utils import extract_lossless_image_b64
from .metrics import record_budget_cutoffs
from .vqa_common import lmms_eval_sampling
from .charxiv import image_mime   # the MIME type the bytes carry (PNG, GIF, WebP, else JPEG)

logger = logging.getLogger(__name__)

DATASET = "ahmed-masry/ChartQA"
DATASET_REVISION = "af8b6f5c08c95085271561c2a3f9d15f2b5a9031"
#: lmms-eval chartqa `default` post_prompt (the pre_prompt is empty), verbatim.
POST_PROMPT = "\nAnswer the question with a single word."
#: chartqa.yaml generation_kwargs: max_new_tokens 16, temperature 0, do_sample False.
UPSTREAM_MAX_NEW_TOKENS = 16
OFFICIAL_SAMPLING = lmms_eval_sampling("lmms-eval tasks/chartqa/chartqa.yaml @42296f8f",
                                       UPSTREAM_MAX_NEW_TOKENS)


def _load_chartqa_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load the ChartQA test split with the original PNG bytes (no re-encoding)."""
    from datasets import load_dataset, Image

    ds = load_dataset(DATASET, split="test", revision=DATASET_REVISION)
    if hasattr(ds, "cast_column"):
        ds = ds.cast_column("image", Image(decode=False))

    # --shard first (by row index over the full ordered split, before any image is read),
    # then the --eval-limit draw within the shard.
    keep, shard_label = apply_shard(list(range(len(ds))), "chartqa")
    if shard_label:
        ds = ds.select(keep) if hasattr(ds, "select") else [ds[i] for i in keep]
    # Stratified over the human/augmented halves, not a contiguous head.
    ds = limit_dataset(ds, limit, "type", seed="chartqa")
    samples = []
    for item in ds:
        q_text = item.get("query", "") or item.get("question", "")
        label = item.get("label", [""])
        b64_str = extract_lossless_image_b64(item.get("image"))
        if not b64_str:
            # A question without its chart cannot be answered; dropping it would shrink the
            # denominator without saying so.
            raise RuntimeError(f"chartqa: row {item.get('imgname')!r} has no decodable image")

        prompt = f"{q_text}{POST_PROMPT}"   # chartqa_doc_to_text: pre_prompt + question + post_prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{image_mime(b64_str)};base64,{b64_str}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        samples.append((messages, label, {"category": item.get("type") or "unknown"}))

    logger.info(f"Loaded {len(samples)} ChartQA samples with direct lossless PNG bytes.")
    return samples


def _eval_chartqa(response_text: str, gold_answers: Any):
    """chartqa_process_results: relaxed_correctness of the reply as returned."""
    from .vqa_common import lmms_chartqa_relaxed
    if gold_answers is None:
        return False
    return lmms_chartqa_relaxed(response_text, gold_answers)


def run_chartqa(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native ChartQA evaluation suite."""
    samples = _load_chartqa_samples(limit=kwargs.get("limit"))
    from .vqa_common import lmms_temperature_reason, lost_requests_reason
    eval_plan = eval_limit_plan(
        "chartqa", kwargs.get("limit"), len(samples), attempts=base_attempts())
    result = run_eval_suite(
        eval_name="chartqa",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's questions, kept as is
        eval_fn=_eval_chartqa,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=OFFICIAL_SAMPLING,   # its max_tokens follows the token-limit policy
    )
    result["eval_limit_plan"] = eval_plan
    result["metric"] = ("ChartQA relaxed accuracy of the reply as returned (lmms-eval chartqa: "
                        "relaxed_correctness, whole-answer float within 5% of a non-zero gold, "
                        "trailing % = /100, else case-insensitive exact match); per-half accuracy "
                        "in category_accuracy (human / augmented)")
    reasons = []
    if kwargs.get("limit"):
        reasons.append(f"--eval-limit {kwargs['limit']} subset of the 2,500 questions")
    temperature_reason = lmms_temperature_reason(result)
    if temperature_reason:
        reasons.append(temperature_reason)
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
