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

# Evaluation Suite: coco_caption
# Description: MS-COCO Captions, Karpathy 5k test split (lmms-eval coco_karpathy_test) - image captioning, CIDEr headline

"""gbench native built-in runner for coco_caption (Multimodal Vision).

MS-COCO image captioning on the Karpathy test split, the split published COCO captioning
results use, with lmms-eval's `coco_karpathy_test` protocol (EvolvingLMMs-Lab/lmms-eval
`lmms_eval/tasks/coco_cap/coco_karpathy_test.yaml` + `utils.py` @
42296f8fd2f0929d88a6ac4f7a1f2d784537f8d8):

* Data: the 5,000 images of `yerevann/coco-karpathy` `test` (the dataset lmms-eval loads,
  pinned below) with their Karpathy reference sentences (5 per image, 6 for ten images), as
  `coco_process_result_karpathy` passes them to the scorer.
* Images: lmms-eval downloads each row's `url` (images.cocodataset.org/val2014/...). gbench
  takes the same files from the pinned `lmms-lab/COCO-Caption` `val` split, matched by file
  name; its bytes are the val2014 JPEGs served at those URLs, sent as they are.
* Prompt: the image, then "Describe the image briefly." (the task's doc_to_text).
* Scoring: corpus level with pycocoevalcap, as `coco_aggregation_result`: the PTB tokenizer,
  then BLEU-1..4, METEOR, ROUGE-L and CIDEr (pycocoevalcap's CIDEr-D implementation), the seven
  metrics of the task's metric_list (SPICE is commented out there). CIDEr is the headline.

Requires 'pycocoevalcap' and a Java runtime (the PTB tokenizer and METEOR run Java). The suite
runs every scorer on a two-image fixture before generating anything and hard-errors if any of
them fails, and a scorer failure after generation raises instead of dropping a metric.

Sampling: base.OfficialSampling from vqa_common.lmms_eval_sampling: coco_karpathy_test's
generation_kwargs as lmms-eval's chat-endpoint client sends them, temperature 0 and top_p 1.0
(num_beams and do_sample are not sent by that client; it runs a thinking model with
enable_thinking_kwarg, which keeps the task's sampling). top_p 1.0 is sent in both modes; the
temperature follows gbench's rule (base.resolve_temperature): 1.0 with `--thinking`, 0.0
without. Without
`--thinking` every request carries max_tokens 64 (max_new_tokens), so `--max-output-tokens` is
not applied and a caption that stops at 64 tokens is scored as it arrived
(`replies_at_official_max_tokens` counts them). lmms-eval declares no reasoning-model cap, so
with `--thinking` the 64, written for non-reasoning generation, is not applied: each request is
sent the run's `--max-output-tokens`, else the thinking floor (base.resolve_token_budget;
`official_sampling.max_tokens_applied` false and the cap recorded as not applied to reasoning).
Override the temperature for a whole run with `--temperature`, or for this suite alone with
`GBENCH_COCO_CAPTION_TEMPERATURE`, which takes precedence over both; the result records
`official_sampling`. A caption cut off by the run's token budget (`replies_cut_off_by_budget`)
makes the run not comparable.
"""

import logging
import shutil
from typing import Any, Dict, List, Optional, Tuple
from .limit_plan import base_attempts, eval_limit_plan
from .base import (apply_shard, finalize_shard_fields, is_scored, run_eval_suite,
                   strip_thinking_tags, tally_batch_scores)
from .sampling import limit_dataset, presharded_for_base
from .swebench_common import infra_required
from .dataset_utils import extract_lossless_image_b64
from .metrics import record_budget_cutoffs
from .vqa_common import lmms_eval_sampling

logger = logging.getLogger(__name__)

PILLAR = "Multimodal & Vision"
DOCS_URL = "docs/evals/coco_caption.md"
#: The Karpathy split as lmms-eval coco_karpathy_test loads it: split membership, file names
#: and reference sentences.
_SPLIT_DATASET = "yerevann/coco-karpathy"
_SPLIT_REVISION = "448fdb1bc7b2d09e46881c4541a14d796a3d41e8"
_SPLIT = "test"
_N_IMAGES = 5000
#: The COCO-2014 val images (with file names) the Karpathy test images are taken from.
_IMAGE_DATASET = "lmms-lab/COCO-Caption"
_IMAGE_REVISION = "c3213b54e173ae86517d481de17f31439d6f8c72"
_IMAGE_SPLIT = "val"
# lmms-eval coco_karpathy_test.yaml doc_to_text, verbatim.
_PROMPT = "Describe the image briefly."
#: lmms-eval coco_karpathy_test generation_kwargs: max_new_tokens 64, temperature 0, top_p 1.0
#: (num_beams 1 and do_sample false are not sent by the chat-endpoint client).
UPSTREAM_MAX_NEW_TOKENS = 64
OFFICIAL_SAMPLING = lmms_eval_sampling(
    "lmms-eval tasks/coco_cap/coco_karpathy_test.yaml @42296f8f (top_p 1.0)",
    UPSTREAM_MAX_NEW_TOKENS, params={"top_p": 1.0})
#: The task's metric_list, in pycocoevalcap's names.
METRICS = ("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr")
_RESULT_KEYS = {"Bleu_1": "bleu1", "Bleu_2": "bleu2", "Bleu_3": "bleu3", "Bleu_4": "bleu4",
                "METEOR": "meteor", "ROUGE_L": "rouge_l", "CIDEr": "cider"}
# Per-image CIDEr threshold used only for the trace-level pass flag; the reported
# canonical number is the corpus 'cider' score in the result dict.
_PASS_CIDER = 0.5


def check_coco_caption_prerequisites() -> Tuple[bool, str]:
    """Check datasets + Pillow + pycocoevalcap are importable, Java is on PATH, and every scorer
    runs on a two-image fixture."""
    try:
        import datasets  # noqa: F401
        import PIL  # noqa: F401
    except ImportError:
        return False, "Python packages 'datasets'/'Pillow' are not installed."
    try:
        from pycocoevalcap.cider.cider import Cider  # noqa: F401
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer  # noqa: F401
    except ImportError:
        return False, (
            "Python package 'pycocoevalcap' is not installed "
            "(pip install -e \".[evals]\" from the gbench checkout; needs Java for the PTB "
            "tokenizer and METEOR)."
        )
    if shutil.which("java") is None:
        # The PTB tokenizer runs Java inside the scorer; finding out after 5,000 captions
        # were generated would throw the whole run away.
        return False, "No 'java' on PATH (pycocoevalcap's PTB tokenizer and METEOR need a Java runtime)."
    broken = _scorer_smoke_test()
    if broken:
        return False, (f"a pycocoevalcap scorer failed on a two-image fixture ({broken}); fix the "
                       "Java runtime or the pycocoevalcap install before generating")
    return True, ""


def compute_caption_metrics(gts: Dict[Any, List[str]], res: Dict[Any, str]
                            ) -> Tuple[Dict[str, float], Dict[Any, float]]:
    """coco_aggregation_result's scoring: PTB-tokenize, then the seven metrics.

    `gts` maps an image id to its reference sentences, `res` to the one caption. Returns the
    corpus scores keyed as METRICS and the per-image CIDEr. Raises if any scorer fails: a
    missing metric is never dropped silently.
    """
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    tok = PTBTokenizer()
    gts_t = tok.tokenize({iid: [{"caption": r} for r in refs] for iid, refs in gts.items()})
    res_t = tok.tokenize({iid: [{"caption": cap}] for iid, cap in res.items()})
    scores: Dict[str, float] = {}
    bleu, _ = Bleu(4).compute_score(gts_t, res_t)
    for n, value in enumerate(bleu, start=1):
        scores[f"Bleu_{n}"] = float(value)
    scores["METEOR"] = float(Meteor().compute_score(gts_t, res_t)[0])
    scores["ROUGE_L"] = float(Rouge().compute_score(gts_t, res_t)[0])
    cider, cider_per = Cider().compute_score(gts_t, res_t)
    scores["CIDEr"] = float(cider)
    per_image = dict(zip(list(gts_t.keys()), [float(c) for c in cider_per]))
    return scores, per_image


def _scorer_smoke_test() -> Optional[str]:
    """Run every scorer on a two-image fixture; the error text if one fails, else None."""
    gts = {1: ["a man riding a motorcycle on a dirt road", "a person on a motor bike outside"],
           2: ["a cat sitting on a laptop keyboard", "a kitten lying on a computer"]}
    res = {1: "a man rides a motorbike down a dirt road", 2: "a cat on a laptop"}
    try:
        scores, _ = compute_caption_metrics(gts, res)
    except Exception as e:   # Java, the METEOR jar, the tokenizer jar
        return f"{type(e).__name__}: {e}"
    missing = [m for m in METRICS if m not in scores]
    return f"metrics {missing} were not computed" if missing else None


def _load_coco_caption_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """The Karpathy test images with their reference sentences; raises on failure."""
    try:
        from datasets import load_dataset, Image
        split = load_dataset(_SPLIT_DATASET, split=_SPLIT, revision=_SPLIT_REVISION)
        images = load_dataset(_IMAGE_DATASET, split=_IMAGE_SPLIT, revision=_IMAGE_REVISION)
        if hasattr(images, "cast_column"):
            images = images.cast_column("image", Image(decode=False))   # the original JPEG bytes
    except Exception as e:
        logger.error(f"Failed to load dataset for coco_caption: {e}")
        raise RuntimeError(f"Could not load dataset for coco_caption: {e}") from e

    if len(split) != _N_IMAGES:
        raise RuntimeError(f"coco_caption: {_SPLIT_DATASET}@{_SPLIT_REVISION} '{_SPLIT}' has "
                           f"{len(split)} rows, expected the {_N_IMAGES:,} Karpathy test images")
    # --shard first (by row index over the full ordered split), then the --eval-limit draw
    # within the shard.
    keep, shard_label = apply_shard(list(range(len(split))), "coco_caption")
    if shard_label:
        split = split.select(keep) if hasattr(split, "select") else [split[i] for i in keep]
    if limit is not None and limit > 0:
        # Stratified, not a contiguous head.
        split = limit_dataset(split, limit, None, seed="coco_caption")

    # Read the file-name column alone (no image bytes) to index the 40,504 val rows.
    names = (images["file_name"] if hasattr(images, "column_names")
             else [row["file_name"] for row in images])
    row_of = {}
    for i, name in enumerate(names):
        row_of.setdefault(name, i)

    samples = []
    for item in split:
        name, cocoid = item.get("filename"), item.get("cocoid")
        refs = [str(c) for c in (item.get("sentences") or []) if str(c).strip()]
        if not name or cocoid is None or not refs:
            raise RuntimeError("coco_caption: unexpected Karpathy split schema "
                               "(filename/cocoid/sentences); refusing to fabricate sample data")
        if name not in row_of:
            raise RuntimeError(f"coco_caption: {name} (Karpathy test) is not in "
                               f"{_IMAGE_DATASET}@{_IMAGE_REVISION} '{_IMAGE_SPLIT}'")
        b64 = extract_lossless_image_b64(images[row_of[name]]["image"])
        if not b64:
            raise RuntimeError(f"coco_caption: {name} has no decodable image")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                {"type": "text", "text": _PROMPT},
            ],
        }]
        samples.append((messages, refs, {"category": "captioning", "image_id": int(cocoid)}))

    if not samples:
        raise RuntimeError("coco_caption returned empty rows")
    logger.info(f"Loaded {len(samples)} coco_caption Karpathy test samples.")
    return samples


def _make_scorer(metrics: Dict[str, Any]):
    """Build the corpus-level pycocoevalcap async scorer; writes scores into `metrics`.

    A request that got no reply from the endpoint is a harness error (base marks it
    scoring_excluded before the scorer runs): its image is left out of the corpus, never scored
    as an empty caption. An empty reply is the model's caption and is scored as one (lmms-eval:
    result[0] or ""). With no image left the corpus metrics are not computed and the score is
    withheld."""
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        gts: Dict[Any, List[str]] = {}
        res: Dict[Any, str] = {}
        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("image_id")
            if iid is None or tr.get("scoring_excluded"):
                continue
            gts[iid] = list(tr.get("gold_answer") or [])
            res[iid] = strip_thinking_tags(tr.get("response_text") or "")
        if not gts:
            return
        scores, per_image = compute_caption_metrics(gts, res)
        for name in METRICS:
            metrics[_RESULT_KEYS[name]] = round(scores[name], 4)
        for tr in sample_traces:
            iid = (tr.get("extra_payload") or {}).get("image_id")
            if tr.get("scoring_excluded"):
                tr["is_correct"] = False
                if tr.get("status") == "PENDING_JUDGE":
                    tr["status"] = "OK"
                continue
            tr["cider"] = round(float(per_image.get(iid, 0.0)), 4)
            tr["is_correct"] = bool(per_image.get(iid, 0.0) >= _PASS_CIDER)
            if tr.get("status") == "PENDING_JUDGE":
                tr["status"] = "OK"
    return _score


def run_coco_caption(
    model_name: str,
    base_url: str,
    concurrency: int = 1,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run COCO Karpathy-test captioning (hard-errors if pycocoevalcap, Java or a scorer is missing)."""
    ok, reason = check_coco_caption_prerequisites()
    if not ok:
        raise infra_required("coco_caption", reason, DOCS_URL)

    limit = kwargs.get("limit")
    samples = _load_coco_caption_samples(limit=limit)
    metrics: Dict[str, Any] = {}
    eval_plan = eval_limit_plan(
        "coco_caption", limit, len(samples), attempts=base_attempts(batch_scored=True))
    result = run_eval_suite(
        eval_name="coco_caption",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's images, kept as is
        async_eval_fn=_make_scorer(metrics),
        declared_scoring_mode="deterministic",  # corpus caption metrics, not an LLM judge
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=limit,
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=OFFICIAL_SAMPLING,   # its max_tokens follows the token-limit policy
    )
    result["eval_limit_plan"] = eval_plan
    result.update(metrics)  # corpus BLEU-1..4 / METEOR / ROUGE-L / CIDEr
    _headline_and_comparability(result, metrics, limit)
    # One shard of a --shard I/N run is a partial set (CIDEr's IDF weights come from the shard's
    # references): the shard reason (finalize_shard_fields, which `gbench --merge` drops once it
    # has scored the union of the shards' captions, merge_shards).
    return finalize_shard_fields(result)


def _headline_and_comparability(result: Dict[str, Any], metrics: Dict[str, Any],
                                limit: Optional[int]) -> Dict[str, Any]:
    """The headline (corpus CIDEr x100) and the comparability of a scored result (a run's, or
    the merged union of a sharded run's)."""
    # The headline is corpus CIDEr x100 (the reporting convention for COCO captioning), NOT the
    # per-image CIDEr>=0.5 pass rate base.py computed as `accuracy` from is_correct. The pass
    # rate is kept as a secondary field.
    # With no image scored (every request a harness error) CIDEr is not computed and base's
    # accuracy is already null with accuracy_withheld_reason.
    result["metric"] = ("CIDEr x100 on the COCO Karpathy test split (lmms-eval "
                        "coco_karpathy_test; pycocoevalcap CIDEr-D; a score, not a percentage)")
    if "cider" in metrics:
        result["per_image_pass_rate"] = result.get("accuracy")
        result["accuracy"] = round(float(metrics["cider"]) * 100.0, 2)
        # The per-image CIDEr >= 0.5 flag is a secondary diagnostic: its count and per-category
        # tally move to their own fields, and the category rows carry the headline metric.
        if result.get("correct_answers") is not None:
            result["per_image_passes"] = result.get("correct_answers")
        result["correct_answers"] = None
        result["correct_answers_note"] = (
            f"not applicable: the headline is corpus CIDEr; per_image_passes counts the images "
            f"whose own CIDEr is at least {_PASS_CIDER}")
        cats = result.get("category_accuracy") or {}
        if cats and "category_cider_pass_rate" not in result:
            result["category_cider_pass_rate"] = cats
        pass_cats = result.get("category_cider_pass_rate") or {}
        result["category_accuracy"] = (
            {cat: {"total": st.get("total"), "correct": None, "accuracy": result["accuracy"]}
             for cat, st in pass_cats.items()} if len(pass_cats) == 1 else {})
    result["dataset"] = (f"{_SPLIT_DATASET}@{_SPLIT_REVISION} ({_SPLIT}) with images from "
                         f"{_IMAGE_DATASET}@{_IMAGE_REVISION} ({_IMAGE_SPLIT})")
    traces = result.get("sample_traces") or []
    reasons = []
    # Repeated attempts of one image count once.
    n_images = len({(t.get("extra_payload") or {}).get("image_id", t.get("source_sample_idx", i))
                    for i, t in enumerate(traces)})
    if limit:
        reasons.append(f"--eval-limit {limit} subset ({n_images:,} of the {_N_IMAGES:,} Karpathy "
                       "test images; CIDEr's IDF weights come from the scored images' references)")
    from .vqa_common import lmms_temperature_reason, lost_requests_reason
    temperature_reason = lmms_temperature_reason(result)
    if temperature_reason:
        reasons.append(temperature_reason)
    lost = lost_requests_reason(traces)
    if lost:
        reasons.append(lost)
    cut = record_budget_cutoffs(result)
    if cut:
        reasons.append(cut)
    result["leaderboard_comparable"] = not reasons
    result["leaderboard_comparable_reason"] = "; ".join(reasons) or None
    return result


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of a sharded run (`gbench --merge`): the corpus metrics (CIDEr's IDF
    weights come from the references of every scored image) are computed again over the union of
    the shards' captions (traces de-duplicated by sample_key, counts re-counted as the generic
    merge does), never averaged over shards; the merge therefore needs pycocoevalcap and Java,
    as a run does (a MergeError names what is missing)."""
    import asyncio
    from ...merge import MergeError, generic_merge
    ok, reason = check_coco_caption_prerequisites()
    if not ok:
        raise MergeError(f"coco_caption: merging scores the union of the shards' captions with "
                         f"pycocoevalcap, which is not available here: {reason}")
    merged = generic_merge("coco_caption", [dict(r, accuracy=None) for r in results])
    for key in ("per_image_pass_rate", "per_image_passes", "category_cider_pass_rate",
                "correct_answers_note") + tuple(_RESULT_KEYS.values()):
        merged.pop(key, None)
        (merged.get("merge_unreconciled_fields") or {}).pop(key, None)
    traces = merged.get("sample_traces") or []
    metrics: Dict[str, Any] = {}
    try:
        asyncio.run(_make_scorer(metrics)(traces))
    except Exception as e:                                              # noqa: BLE001
        raise MergeError(f"coco_caption: scoring the union of the shards' captions failed "
                         f"({type(e).__name__}: {e}); nothing was merged") from e
    merged.update(metrics)
    # the per-image pass flags come from the union's CIDEr: re-count them
    correct, categories = tally_batch_scores(traces)
    scored = sum(1 for t in traces if is_scored(t))
    merged["correct_answers"] = correct
    merged["accuracy"] = round(correct / scored * 100.0, 2) if scored else None
    merged["category_accuracy"] = {
        cat: {**st, "accuracy": round(st["correct"] / st["total"] * 100.0, 2) if st["total"]
              else 0.0} for cat, st in sorted(categories.items())}
    limit = next((r.get("eval_limit") for r in results if r.get("eval_limit")), None)
    return _headline_and_comparability(merged, metrics, limit)
