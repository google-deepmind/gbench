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

# Evaluation Suite: textvqa
# Description: TextVQA (Visual Question Answering on Text in Images Benchmark)

"""Native TextVQA (Visual Question Answering on Text in Images) evaluation suite.

Runs the TextVQA validation split (`lmms-lab/textvqa`, pinned below; 5,000 questions with 10
human answers each) as lmms-eval's `textvqa_val` task (EvolvingLMMs-Lab/lmms-eval
`lmms_eval/tasks/textvqa/_default_template_textvqa_yaml` + `utils.py` @
42296f8fd2f0929d88a6ac4f7a1f2d784537f8d8, the `default` model kwargs, no OCR tokens): the image,
then `question.capitalize()` followed by "\nAnswer the question using a single word or phrase."
The reply, as it comes back (lmms-eval's task layer strips only surrounding whitespace), and
every human answer go through the EvalAI answer processor that the TextVQA evaluation server
uses (vendored verbatim in vqa_common), and the headline is the mean VQA accuracy: per question, the leave-one-out average over the 10 annotators of
min(#matching other answers / 3, 1) (textvqa_process_results; mmf's TextVQAAccuracyEvaluator).
There is no answer extraction.

Sampling: base.OfficialSampling from vqa_common.lmms_eval_sampling. The task sets neither a
temperature nor max_new_tokens, so lmms-eval's chat-endpoint client (models/chat/openai.py) sends
its defaults: temperature 0 (recorded; the client runs a thinking model with
enable_thinking_kwarg, which keeps the sampling). The temperature sent follows gbench's rule
(base.resolve_temperature): 1.0 with `--thinking`, 0.0 without. Without
`--thinking` every request carries
max_tokens 1024, so `--max-output-tokens` is not applied and a reply that stops at 1024 tokens is
scored as it arrived (`replies_at_official_max_tokens` counts them). lmms-eval declares no
reasoning-model cap, so with `--thinking` the 1024, written for non-reasoning generation, is not
applied: each request is sent the run's `--max-output-tokens`, else the thinking floor
(base.resolve_token_budget; `official_sampling.max_tokens_applied` false and the cap recorded as
not applied to reasoning). Override the temperature for a whole run with `--temperature`, or for
this suite alone with `GBENCH_TEXTVQA_TEMPERATURE`, which takes precedence over both; the result
records `official_sampling`. The result is leaderboard-comparable for a full, unsharded run at the
official temperature with every request answered and no reply cut off by the run's token budget
(`replies_cut_off_by_budget`).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from .limit_plan import base_attempts, eval_limit_plan
from .base import apply_shard, finalize_shard_fields, run_eval_suite
from .sampling import limit_dataset, presharded_for_base
from .charxiv import image_mime   # the MIME type the bytes carry (PNG, GIF, WebP, else JPEG)
from .dataset_utils import extract_lossless_image_b64
from .metrics import record_budget_cutoffs
from .vqa_common import lmms_eval_sampling

logger = logging.getLogger(__name__)

DATASET = "lmms-lab/textvqa"
DATASET_REVISION = "9c0699cd19768ac5ab97568f6b3cbac4c0062884"
#: lmms-eval textvqa `default` post_prompt (the pre_prompt is empty, ocr false), verbatim.
POST_PROMPT = "\nAnswer the question using a single word or phrase."
#: The task sets neither max_new_tokens nor temperature; lmms-eval's chat-endpoint client
#: (models/chat/openai.py) then sends max_tokens 1024 and temperature 0.
UPSTREAM_MAX_NEW_TOKENS = 1024
OFFICIAL_SAMPLING = lmms_eval_sampling(
    "lmms-eval tasks/textvqa (no generation_kwargs: the client's defaults) @42296f8f",
    UPSTREAM_MAX_NEW_TOKENS)


def _load_textvqa_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load the TextVQA validation split with the original image bytes (no re-encoding)."""
    from datasets import load_dataset, Image

    ds = load_dataset(DATASET, split="validation", revision=DATASET_REVISION)
    if hasattr(ds, "cast_column"):
        ds = ds.cast_column("image", Image(decode=False))

    # --shard first (by row index over the full ordered split, before any image is read),
    # then the --eval-limit draw within the shard.
    keep, shard_label = apply_shard(list(range(len(ds))), "textvqa")
    if shard_label:
        ds = ds.select(keep) if hasattr(ds, "select") else [ds[i] for i in keep]
    # Stratified, not a contiguous head.
    ds = limit_dataset(ds, limit, None, seed="textvqa")
    samples = []
    for item in ds:
        q_text = item.get("question", "")
        answers = item.get("answers", [""])
        img_obj = item.get("image")

        b64_str = extract_lossless_image_b64(img_obj)
        if not b64_str:
            # A question without its image cannot be answered; dropping it would shrink the
            # denominator without saying so.
            raise RuntimeError(f"textvqa: question {item.get('question_id')!r} has no decodable image")

        # textvqa_doc_to_text: pre_prompt + question.capitalize() + post_prompt.
        prompt = f"{q_text.capitalize()}{POST_PROMPT}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:{image_mime(b64_str)};base64,{b64_str}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        samples.append((messages, answers, {"category": "scene_ocr"}))

    logger.info(f"Loaded {len(samples)} TextVQA samples with direct lossless image bytes.")
    return samples


def _eval_textvqa(response_text: str, gold_answers: Any):
    """Binary form of VQA accuracy (soft score >= 0.5) of the reply as returned.

    The headline is the soft mean (`_promote_soft_headline`); this per-sample pass flag
    feeds `binary_pass_rate` and the per-sample `is_correct`.
    """
    from .vqa_common import lmms_textvqa_accuracy
    if gold_answers is None:
        return False
    return lmms_textvqa_accuracy(response_text, gold_answers) >= 0.5

def run_textvqa(
    model_name: str,
    base_url: str,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run native TextVQA evaluation suite."""
    samples = _load_textvqa_samples(limit=kwargs.get("limit"))
    from .vqa_common import lmms_temperature_reason, lost_requests_reason
    eval_plan = eval_limit_plan(
        "textvqa", kwargs.get("limit"), len(samples), attempts=base_attempts())
    result = run_eval_suite(
        eval_name="textvqa",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's questions, kept as is
        eval_fn=_eval_textvqa,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=OFFICIAL_SAMPLING,   # its max_tokens follows the token-limit policy
    )
    result["eval_limit_plan"] = eval_plan
    result = _promote_soft_headline(result)
    reasons = []
    if kwargs.get("limit"):
        reasons.append(f"--eval-limit {kwargs['limit']} subset of the 5,000 questions")
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


def _promote_soft_headline(result: Dict[str, Any]) -> Dict[str, Any]:
    """Promote the canonical TextVQA MEAN soft VQA-accuracy (mean of min(#annotators-match/3, 1))
    to the headline. The binary pass-rate (fraction clearing >=0.5) under-reports and drops
    defensible partial credit, so recompute the soft mean per trace and keep the binary form as
    `binary_pass_rate`. A request that got no reply from the endpoint (a harness error base
    marks scoring_excluded) is left out of the mean, as it is out of base's accuracy. Mutates and
    returns `result`."""
    from .base import is_scored
    from .vqa_common import lmms_textvqa_accuracy
    from .metrics import replace_category_headline
    soft = []
    for tr in result.get("sample_traces", []):
        gold = tr.get("gold_answer")
        if gold is None or not is_scored(tr):
            continue
        s = lmms_textvqa_accuracy(tr.get("response_text"), gold)
        tr["vqa_soft_score"] = round(s, 4)
        soft.append((tr, s))
    if soft:
        result["binary_pass_rate"] = result.get("accuracy")
        result["accuracy"] = round(sum(s for _tr, s in soft) / len(soft) * 100.0, 2)
        result["metric"] = ("TextVQA VQA accuracy x100 = mean over questions of the leave-one-out "
                            "min(#annotator-match/3, 1) after the EvalAI answer processor, on the "
                            "reply as returned (lmms-eval textvqa_val); binary_pass_rate = "
                            "fraction of samples with soft>=0.5")
        # the category rows and the Correct count say the soft mean too (the binary tally
        # moves to binary_pass_rate / pass_count)
        replace_category_headline(result, soft, metric_name=result["metric"],
                                  binary_key="binary_pass_rate")
    return result
