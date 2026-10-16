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

"""Native MMMU-Pro multimodal vision evaluation suite.

Follows the official MMMU-Pro inference and scoring code (MMMU-Benchmark/MMMU `mmmu-pro/`
@ 268471d0d488258990025331c7528359c324aa25: `infer/infer_gpt.py`, `prompts.yaml`,
`evaluate.py`) on the pinned `MMMU/MMMU_Pro` test split, 1,730 questions per setting:

* **standard (10 options)** - the question, then the options as ``A. ...`` lines, then the
  prompt for the mode, in one text part. Each ``<image n>`` token becomes ``<image>`` and
  the referenced images follow the text in the order they are referenced (upstream
  replace_images_tokens / origin_mmmu_doc_to_visual).
* **vision** - the screenshot that holds the whole question, sent after the mode's vision
  prompt, with no other text.

Images go out as their original bytes (upstream re-encodes the decoded image as PNG, which
carries the same pixels).

Both prompt modes of `prompts.yaml` run by default (CoT and Direct). Replies are parsed with
upstream's parse_multi_choice_response; its last resort, a random letter, is drawn from an
RNG seeded with the question id so a run is reproducible. The paper reports, per setting,
the higher of the CoT and Direct accuracies, and the MMMU-Pro score is the mean of the
standard-10 and vision numbers; that mean is the headline `accuracy`.
`GBENCH_MMMU_PRO_PROMPT=cot` (or `direct`) runs one mode only, and the result is then
flagged as not leaderboard-comparable.

Request, as infer_gpt.py's request_with_images sends it (the script whose option format and
image order the suite follows): the system message "You are ChatGPT, a large language model
trained by OpenAI, based on the GPT-4 architecture." (verbatim), then one user turn with the text
followed by the images; no temperature (the API default applies) and max_tokens 4096.

Sampling: base.OfficialSampling; infer_gpt.py sends no temperature (recorded), but gbench always
sends the temperature of its rule (base.resolve_temperature): 1.0 with `--thinking`, 0.0
without, which `leaderboard_comparable_reason` names. Without `--thinking`
every request carries infer_gpt.py's
max_tokens=4096, so `--max-output-tokens` is not applied and a reply that stops there is parsed
as it arrived, as the script parses it (`replies_at_official_max_tokens` counts them). The script
defines no reasoning-model settings, so with `--thinking` the 4096 cap, written for non-reasoning
generation, is not applied: each request is sent the run's `--max-output-tokens` (the token-limit
policy, base.resolve_token_budget; the result records it). Override the temperature for a whole
run with `--temperature`, or for this suite alone with `GBENCH_MMMU_PRO_TEMPERATURE`, which takes
precedence over both; the result records `official_sampling`.
"""

import ast
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple
from .base import OfficialSampling, run_eval_suite, suite_env
from .metrics import record_budget_cutoffs
from .dataset_utils import extract_lossless_image_b64
from .limit_plan import base_attempts, eval_limit_plan
from .sampling import allocate_limit, parse_shard, shard_select, stratified_indices

logger = logging.getLogger(__name__)

DATASET = "MMMU/MMMU_Pro"
DATASET_REVISION = "563f3e84bb3b90893083a1f039cfa13077f2302b"
STANDARD = "standard (10 options)"
VISION = "vision"
SETTING_KEYS = {STANDARD: "standard_10", VISION: "vision"}

# mmmu-pro/prompts.yaml, verbatim.
PROMPTS = {
    "cot": {
        "vision": "Write out the multiple-choice question in the image and then solve it. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.",
        "standard": "Answer the preceding multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.",
    },
    "direct": {
        "vision": "Answer with the option letter from the given choices directly. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options.",
        "standard": "Answer with the option letter from the given choices directly.",
    },
}
MODE_CHOICES = {"both": ("cot", "direct"), "cot": ("cot",), "direct": ("direct",)}

#: mmmu-pro/infer/infer_gpt.py @268471d0 request_with_images: the system turn, verbatim.
SYSTEM_MESSAGE = ("You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 "
                  "architecture.")
#: The same request sends no temperature and max_tokens=4096 (infer(..., max_tokens=4096)).
UPSTREAM_MAX_TOKENS = 4096
OFFICIAL_SAMPLING = OfficialSampling(
    reference="MMMU-Benchmark/MMMU mmmu-pro/infer/infer_gpt.py @268471d0 request_with_images "
              "(no temperature; max_tokens 4096)",
    send_temperature=False, max_tokens=UPSTREAM_MAX_TOKENS)


def _with_system(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """infer_gpt.py's messages: the system turn, then the user turn."""
    return [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": content}]


def _prompt_modes() -> Tuple[str, ...]:
    raw = (suite_env("GBENCH_MMMU_PRO_PROMPT", default="both") or "both").strip().lower()
    if raw not in MODE_CHOICES:
        raise ValueError(f"GBENCH_MMMU_PRO_PROMPT={raw!r}: expected one of {sorted(MODE_CHOICES)}")
    return MODE_CHOICES[raw]


def _image_url(image: Any, where: str) -> str:
    """The image's original bytes as a data URI.

    Upstream re-encodes the decoded PIL image as PNG; the original bytes carry the same
    pixels, are smaller, and are encoded once per question and shared by both prompt modes
    (a full run holds 1,730 screenshots in memory).
    """
    b64 = extract_lossless_image_b64(image)
    if not b64:
        raise RuntimeError(f"mmmu_pro: {where} has no decodable image")
    mime = "image/jpeg" if b64.startswith("/9j/") else "image/png"
    return f"data:{mime};base64,{b64}"


# --- upstream infer_gpt.py prompt construction ------------------------------------------
def replace_images_tokens(input_string):
    image_order = [int(num) for num in re.findall(r'<image\s+(\d+)>', input_string)]
    input_string = re.sub(r'<image\s+\d+>', '<image>', input_string)
    return input_string, image_order


def parse_options(options):
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
    return choices_str


def construct_prompt(doc, mode):
    question = doc["question"]
    parsed_options = parse_options(ast.literal_eval(str(doc["options"])))
    question = f"{question}\n{parsed_options}\n{PROMPTS[mode]['standard']}"
    return question


# --- upstream evaluate.py answer parsing -------------------------------------------------
def get_multi_choice_info(options):
    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))
    return index2ans, all_choices


def parse_multi_choice_response(response, all_choices, index2ans, rng=random):
    """Upstream parse_multi_choice_response, same logic.

    Two mechanical changes: the final random pick comes from `rng` (seeded per question by
    the caller, so a rerun scores the same), and `np.argmax` is the equivalent first-max
    index in plain Python.
    """
    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        # Extract the string after "Answer:"
        answer_str = response[last_answer_pos + len("Answer:"):].strip()

        # Find a unique match in the options
        matching_options = [option for option in all_choices if option in answer_str]

        # If a unique match is found, return that option
        if len(matching_options) == 1:
            return matching_options[0]

    if isinstance(response, str):
        for char in [",", ".", "!", "?", ";", ":", "'"]:
            response = response.strip(char)
        response = " " + response + " "  # add space to avoid partial match
    else:
        response = ""

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        pred_index = rng.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[start_indexes.index(max(start_indexes))]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index


def mmmu_pro_limit_shares(limit: Optional[int], modes: Tuple[str, ...] = ("cot", "direct"),
                          ) -> Dict[str, Optional[int]]:
    """How many questions each `<setting>/<mode>` unit runs: `--eval-limit N` is N samples for the
    suite (a request in one setting and prompt mode is one), dealt over the units
    (sampling.allocate_limit); None (no limit) is every question in every unit."""
    units = [f"{SETTING_KEYS[setting]}/{mode}" for setting in (STANDARD, VISION) for mode in modes]
    return allocate_limit(limit, units, seed="mmmu_pro")


def _load_mmmu_pro_samples(max_soft_tokens: Optional[int] = None, limit: Optional[int] = None,
                           modes: Tuple[str, ...] = ("cot", "direct"),
                           ) -> List[Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]]:
    """Both settings x the requested prompt modes for the same questions.

    `--eval-limit N` runs N requests in total, dealt over the setting x mode units
    (mmmu_pro_limit_shares); each unit draws its share of question ids stratified by subject
    with the same seed, so a smaller share's questions are among a larger share's. A unit dealt
    0 is not run. Without a limit every question runs in every setting and mode.
    """
    from datasets import load_dataset, Image
    standard = load_dataset(DATASET, STANDARD, split="test", revision=DATASET_REVISION)
    vision = load_dataset(DATASET, VISION, split="test", revision=DATASET_REVISION)
    if hasattr(standard, "cast_column"):
        for i in range(1, 8):
            standard = standard.cast_column(f"image_{i}", Image(decode=False))
        vision = vision.cast_column("image", Image(decode=False))

    std_ids = list(standard["id"])
    subjects = list(standard["subject"])
    shares = mmmu_pro_limit_shares(limit, modes)
    unit_keep = {unit: set(stratified_indices(subjects, share, seed="mmmu_pro"))
                 for unit, share in shares.items() if share != 0}
    keep = sorted(set().union(*unit_keep.values())) if unit_keep else []
    vision_units = [f"{SETTING_KEYS[VISION]}/{mode}" for mode in modes]
    wanted = {std_ids[i] for unit in vision_units for i in unit_keep.get(unit, ())}
    vis_index = {vid: i for i, vid in enumerate(vision["id"])}
    missing = sorted(wanted - set(vis_index))
    if missing:
        raise RuntimeError(f"mmmu_pro: {len(missing)} standard-10 ids have no vision row "
                           f"(e.g. {missing[0]!r}) in {DATASET}@{DATASET_REVISION[:8]}")

    def _runs(setting: str, mode: str, i: int) -> bool:
        return i in unit_keep.get(f"{SETTING_KEYS[setting]}/{mode}", ())

    extra_mm = {"mm_processor_kwargs": {"max_soft_tokens": max_soft_tokens}} if max_soft_tokens else {}
    samples = []
    for i in keep:
        doc = standard[i]
        std_modes = [mode for mode in modes if _runs(STANDARD, mode, i)]
        vis_modes = [mode for mode in modes if _runs(VISION, mode, i)]
        if std_modes:
            options = ast.literal_eval(str(doc["options"]))
            gold = {"answer": str(doc["answer"]).strip(), "options": options, "id": doc["id"]}
        urls: Dict[int, str] = {}
        for mode in std_modes:
            prompt, image_order = replace_images_tokens(construct_prompt(doc, mode))
            content = [{"type": "text", "text": prompt}]
            for idx in image_order:
                if idx not in urls:
                    urls[idx] = _image_url(doc.get(f"image_{idx}"), f"{doc['id']} <image {idx}>")
                content.append({"type": "image_url", "image_url": {"url": urls[idx]}})
            meta = {"category": f"{SETTING_KEYS[STANDARD]}/{mode}", "subject": doc.get("subject"),
                    "mmmu_pro_id": doc["id"], **extra_mm}
            samples.append((_with_system(content), gold, meta))

        if not vis_modes:
            continue
        vdoc = vision[vis_index[doc["id"]]]
        voptions = ast.literal_eval(str(vdoc["options"]))
        vgold = {"answer": str(vdoc["answer"]).strip(), "options": voptions, "id": vdoc["id"]}
        vurl = _image_url(vdoc.get("image"), f"vision {vdoc['id']}")
        for mode in vis_modes:
            content = [{"type": "text", "text": PROMPTS[mode]["vision"]},
                       {"type": "image_url", "image_url": {"url": vurl}}]
            meta = {"category": f"{SETTING_KEYS[VISION]}/{mode}", "subject": vdoc.get("subject"),
                    "mmmu_pro_id": vdoc["id"], **extra_mm}
            samples.append((_with_system(content), vgold, meta))

    logger.info("Loaded %d MMMU-Pro requests: %d questions over 2 settings x %d prompt mode(s) "
                "from %s@%s.", len(samples), len(keep), len(modes), DATASET, DATASET_REVISION[:8])
    return samples


def _eval_mmmu_pro(response_text: str, gold_answer: Any) -> bool:
    """Upstream mmmu_process_results: parse_multi_choice_response(...) == answer.

    `gold_answer` is the loader's `{"answer", "options", "id"}`; a bare letter is accepted
    too (choices A-J, no option texts to match against).
    """
    if isinstance(gold_answer, dict):
        answer = str(gold_answer.get("answer") or "")
        options = list(gold_answer.get("options") or [])
        seed = str(gold_answer.get("id") or "")
    else:
        answer, options, seed = str(gold_answer or ""), [], ""
    if not options:
        index2ans, all_choices = {}, [chr(ord("A") + i) for i in range(10)]
    else:
        index2ans, all_choices = get_multi_choice_info(options)
    response = response_text or ""
    rng = random.Random(f"mmmu_pro:{seed or response}")
    return parse_multi_choice_response(response, all_choices, index2ans, rng=rng) == answer


def _setting_scores(traces: List[Dict[str, Any]], modes: Tuple[str, ...]) -> Dict[str, Any]:
    """Accuracy per setting x mode, the paper's higher-of-modes per setting, and their mean.

    Over the scored requests, as base's accuracy: a request that got no reply from the endpoint
    is a harness error (scoring_excluded), left out."""
    from .base import is_scored
    by_cat: Dict[str, List[bool]] = {}
    for t in traces:
        if not is_scored(t):
            continue
        by_cat.setdefault(t.get("category") or "", []).append(bool(t.get("is_correct")))
    per_mode: Dict[str, Dict[str, Optional[float]]] = {}
    best: Dict[str, Optional[float]] = {}
    for key in SETTING_KEYS.values():
        per_mode[key] = {}
        for mode in modes:
            hits = by_cat.get(f"{key}/{mode}") or []
            per_mode[key][mode] = round(100.0 * sum(hits) / len(hits), 2) if hits else None
        vals = [v for v in per_mode[key].values() if v is not None]
        best[key] = max(vals) if vals else None
    overall = (round((best["standard_10"] + best["vision"]) / 2.0, 2)
               if best["standard_10"] is not None and best["vision"] is not None else None)
    return {"per_mode": per_mode, "best": best, "overall": overall}


def _question_id(trace: Dict[str, Any]) -> Any:
    """The MMMU-Pro question id of a trace (its sample's mmmu_pro_id, else its gold's id)."""
    gold = trace.get("gold_answer")
    return (trace.get("mmmu_pro_id") or (trace.get("extra_payload") or {}).get("mmmu_pro_id")
            or (gold.get("id") if isinstance(gold, dict) else None))


def _withheld_reason(scores: Dict[str, Any], traces: List[Dict[str, Any]],
                     status_reason: Optional[str], shard: Optional[str]) -> str:
    """Why MMMU-Pro has no value: per setting without a score, either every request of it was
    left out of the score (the harness errors in `status_reason`), or the run held none of its
    requests (a shard: MMMU-Pro needs the merged shards)."""
    missing = [k for k, v in scores["best"].items() if v is None]
    present = {str(t.get("category") or "").split("/", 1)[0] for t in traces}
    unscored = [k for k in missing if k in present]
    absent = [k for k in missing if k not in present]
    parts = []
    if unscored:
        parts.append(f"no scored request in the {' and '.join(unscored)} setting(s) ("
                     + str(status_reason or "every request left out of the score") + ")")
    if absent:
        held = sorted(k for k in present if k)
        parts.append(f"this run holds no request of the {' and '.join(absent)} setting(s)"
                     + (f" (GBENCH_SHARD {shard} holds only the "
                        f"{' and '.join(held) or 'no'} setting(s); MMMU-Pro needs the merged "
                        "shards, and this shard's per-setting scores are in "
                        "mmmu_pro_standard_10 / mmmu_pro_vision)" if shard else ""))
    return ("MMMU-Pro (the mean of the two settings) has no value: " + "; ".join(parts))


def run_mmmu_pro(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    eval_max_soft_tokens: Optional[int] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run MMMU-Pro (standard-10 + vision, CoT + Direct) with the official prompts and parser."""
    modes = _prompt_modes()
    limit = kwargs.get("limit")
    sharded = bool(os.environ.get("GBENCH_SHARD"))
    samples = _load_mmmu_pro_samples(eval_max_soft_tokens, limit=limit, modes=modes)
    units = list(mmmu_pro_limit_shares(None, modes))
    # The plan counts the requests base runs on this node (GBENCH_SHARD's shard of the list it is
    # given), so the shards' plans add up to the requests drawn.
    on_shard = shard_select(samples, parse_shard(os.environ.get("GBENCH_SHARD")))
    per_unit = {u: sum(1 for _m, _g, meta in on_shard if meta.get("category") == u)
                for u in units}
    plan = eval_limit_plan("mmmu_pro", limit, len(on_shard), per_unit,
                           setting_noun="setting/mode", attempts=base_attempts())
    result = run_eval_suite(
        eval_name="mmmu_pro",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=samples,
        eval_fn=_eval_mmmu_pro,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=None,   # already dealt over the settings x modes in the loader; do not re-cut it
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=OFFICIAL_SAMPLING,   # its max_tokens follows the token-limit policy
    )
    result["eval_limit_plan"] = plan
    scores = _setting_scores(result.get("sample_traces") or [], modes)
    result["pooled_accuracy"] = result.get("accuracy")
    # The headline is not a count over the replies, so the summary row does not pair it with
    # base's pooled counts: those are kept under pooled_*, and the row names the questions.
    result["pooled_total_questions"] = result.get("total_questions")
    result["pooled_correct_answers"] = result.get("correct_answers")
    # the row's questions: the distinct questions asked, and those with a scored reply
    result["total_questions"] = len({_question_id(t)
                                     for t in result.get("sample_traces") or []} - {None})
    result["effective_n"] = len({_question_id(t) for t in result.get("sample_traces") or []
                                 if not t.get("scoring_excluded")} - {None})
    result["correct_answers"] = None
    result["correct_answers_note"] = ("headline is mean(best standard-10, best vision) over the "
                                      "prompt modes, not a count; base's pooled counts over "
                                      "every reply are pooled_total_questions / "
                                      "pooled_correct_answers")
    result["mmmu_pro_standard_10"] = scores["best"]["standard_10"]
    result["mmmu_pro_vision"] = scores["best"]["vision"]
    result["mmmu_pro_by_mode"] = scores["per_mode"]
    result["mmmu_pro_prompt_modes"] = list(modes)
    if scores["overall"] is not None:
        result["accuracy"] = scores["overall"]
    elif result.get("total_questions"):
        # MMMU-Pro is the mean of the two settings, so a setting without a score leaves it
        # without a value: withheld, never the pooled accuracy under MMMU-Pro's name. The reason
        # tells apart a setting whose requests were all left out of the score (harness errors)
        # from one this run did not hold at all (a GBENCH_SHARD shard of the fan-out).
        result["accuracy"] = None
        result["accuracy_withheld_reason"] = _withheld_reason(
            scores, result.get("sample_traces") or [], result.get("status_reason"),
            os.environ.get("GBENCH_SHARD") if sharded else None)
    result["metric"] = ("MMMU-Pro = mean(standard-10, vision); each setting takes the higher of the "
                        f"prompt modes run ({', '.join(modes)}); upstream parse_multi_choice_response")
    traces = result.get("sample_traces") or []
    reasons = []
    if limit:
        reasons.append(f"--eval-limit {limit}: {len(samples)} request(s) over the settings x "
                       "prompt modes, a subset of the 1,730 questions")
    if sharded:
        reasons.append("GBENCH_SHARD run covers one shard")
    if set(modes) != {"cot", "direct"}:
        reasons.append(f"only the {modes[0]} prompt was run; MMMU-Pro reports the higher of CoT and "
                       "Direct per setting (GBENCH_MMMU_PRO_PROMPT=both)")
    if (result.get("official_sampling") or {}).get("temperature_matches_official") is False:
        reasons.append(f"temperature {result.get('temperature')} (infer_gpt.py sends none)")
    from .vqa_common import lost_requests_reason
    lost = lost_requests_reason(traces)
    if lost:
        reasons.append(lost)
    cut = record_budget_cutoffs(result)
    if cut:
        reasons.append(cut)
    result["leaderboard_comparable"] = not reasons
    result["leaderboard_comparable_reason"] = "; ".join(reasons) or None
    return result
