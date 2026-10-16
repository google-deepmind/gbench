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

# Evaluation Suite: charxiv
# Description: CharXiv (Princeton Complex Academic Chart Reasoning & Numerical Understanding)

"""gbench native built-in runner for charxiv (Multimodal & Vision).

Canonical CharXiv (Wang et al., NeurIPS 2024 D&B; arXiv:2406.18521;
github.com/princeton-nlp/CharXiv; HF princeton-nlp/CharXiv) has TWO tracks over the
1000-chart `validation` split:

  * Descriptive - 4 questions per chart (one of 19 templates each), 4000 total.
  * Reasoning   - 1 open-ended question per chart, 1000 total.

Both are single-shot multimodal VQA graded by an LLM judge against a per-question rubric.
This runner fans each chart out into its 5 canonical questions, sends the exact upstream
prompt templates (vendored in `charxiv_constants.py`), and grades with the exact upstream
descriptive/reasoning rubrics via the gbench judge cascade. It reports the descriptive and
reasoning accuracies SEPARATELY (the paper reports no combined figure; the leaderboard is
sorted by Reasoning, so that is the headline `accuracy`, and `total_questions` /
`correct_answers` count the headline track).

The model request is built as the declared sampling reference, src/generate_lib/gpt.py, builds
it: the question text first, then the chart as a base64 data URL whose MIME type is the one of
the image bytes (CharXiv's charts are JPEG, which gpt.py labels image/jpeg). The chart bytes are
the files the official harness sends: its README has the charts unzipped from the dataset's
images.zip into images/, and generate.py reads images/{figure_id}.jpg; gbench reads the same
member of images.zip at DATASET_REVISION (sha256 IMAGES_ZIP_SHA256). The `image` column of the
HF parquet holds re-encoded JPEGs (same size, different pixels) and is not used.

Grading follows src/evaluate.py: descriptive answers are grouped by template id (in dataset
order across charts) and graded up to 5 per prompt (build_descriptive_grading_queries,
nq_per_query=5), with NUM_TRIPLETS and the JSON keys of the batch; each reasoning answer is
graded alone. A descriptive reply the server returned empty is graded in its batch position, as
upstream batches every response; a request with no reply (a harness error) is not graded. A
reply that is not JSON, lacks a key or (descriptive) holds a score outside {0, 1} is asked
again, up to upstream's 10 attempts in total, as get_descriptive_result_gpt /
get_reasoning_result_gpt do; after that the answers of that prompt are JUDGE_FAILED: the grader
is gbench's Gemini substitute, so its failure is a judge failure, left out of each track's
denominator (upstream scores -1 as 0; that figure is kept as accuracy_upstream_invalid_as_zero),
counted in judge_failures and named. A judge cascade outage is JUDGE_OUTAGE (excluded). Every graded trace records the model that answered (`judge_model`), the
raw reply (`judge_raw`) and its batch (`judge_batch`); the result records `judge_models` and
`judge_cascade`.

Judge backend: canonical CharXiv pins gpt-4o-2024-05-13; gbench grades with its standard
Gemini judge cascade - the reference grader gbench uses across all judged suites, by
convention (a grader choice, not a fidelity defect). The prompts, rubrics and JSON score
contract are ported verbatim. Numbers are therefore CharXiv-protocol-faithful, graded by
gbench's cascade rather than the paper's GPT-4o, i.e. a gbench-internal number rather than a
like-for-like leaderboard entry (leaderboard_comparable=False); label accordingly when
comparing to the paper's GPT-4o leaderboard.

Sampling: base.OfficialSampling, per mode, as CharXiv's API clients send it
(src/generate_lib/ @ 7ebe88f). Without `--thinking`, the chat-model client gpt.py: top_p 1.0,
seed 42 and max_tokens 1000 (its temperature 0.0 is recorded), so `--max-output-tokens` is not applied and a reply
that stops at 1000 tokens is scored as it arrived (`replies_at_official_max_tokens` counts them).
The temperature follows gbench's rule in both modes (base.resolve_temperature): 1.0 with
`--thinking`, 0.0 without. With `--thinking` the same client settings apply, and under the
token-limit policy its 1000-token cap (written for
non-reasoning generation) is replaced by `--max-output-tokens`. CharXiv picks its
reasoning client o1.py by model name (OpenAI's o1 models); a setting keyed on a hosted model's
name is not a reasoning-model setting for a served model, so it is not used. Override the temperature for a
whole run with `--temperature`, or for this suite alone with `GBENCH_CHARXIV_TEMPERATURE`, which
takes precedence over both; the result records `official_sampling`.

The judge call follows CharXiv's get_descriptive_result_gpt / get_reasoning_result_gpt
(descriptive_utils.py, reasoning_utils.py): JSON mode, temperature 0, top_p 1 and seed 42; the
result records it under `judge_sampling`. Their max_tokens (256, doubled up to 1024 when the JSON
is cut) is not sent: it was written for gpt-4o, which does not think, and the Gemini judge counts
its thinking against the same limit (named in `leaderboard_comparable_reason`).
"""

import base64
import binascii
import functools
import hashlib
import json
import logging
import os
import zipfile
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from .base import (apply_shard, run_eval_suite, gemini_api_keys, gemini_required_skip,
                   judge_cascade, judge_generate_cascade, judge_config, OfficialSampling)
from .metrics import record_budget_cutoffs
from .limit_plan import base_attempts, eval_limit_plan
from .sampling import allocate_limit, stratified_sample, presharded_for_base
from .swebench_common import infra_required
from .charxiv_constants import (DESCRIPTIVE_RESP_INST, DESCRIPTIVE_GRADING_PREFIX,
                                DESCRIPTIVE_GRADING_QMAP, DESCRIPTIVE_GRADING_ICL,
                                REASONING_RESP_INST, REASONING_GRADING_PREFIX,
                                REASONING_GRADING_INST)

logger = logging.getLogger(__name__)

#: CharXiv's two tracks, reported separately; an --eval-limit's questions are dealt over them.
TRACKS = ("descriptive", "reasoning")

DATASET = "princeton-nlp/CharXiv"
#: HEAD of princeton-nlp/CharXiv (the validation split's 1,000 charts).
DATASET_REVISION = "f441eb632fc62f6f777830a0f47619e6e86459b0"
DOCS_URL = "docs/evals/charxiv.md"
#: The chart files the official harness sends (README: unzip into images/; generate.py reads
#: images/{figure_id}.jpg): 2,323 JPEGs, of which the validation split uses 1,000.
IMAGES_ZIP = "images.zip"
IMAGES_ZIP_SHA256 = "c4b151cef2aba35523764dd1091d719d64acf81a16904d72f31da2723efbd855"
#: src/generate_lib/gpt.py @ 7ebe88f (the chat-model client): max_tokens 1000, temperature 0.0,
#: top_p 1.0, seed 42.
UPSTREAM_MAX_TOKENS = 1000
OFFICIAL_SAMPLING = OfficialSampling(
    reference=("princeton-nlp/CharXiv src/generate_lib/gpt.py @7ebe88f (temperature 0.0, top_p "
               "1.0, seed 42, max_tokens 1000)"),
    temperature=0.0, params={"top_p": 1.0, "seed": 42}, max_tokens=UPSTREAM_MAX_TOKENS)
#: descriptive_utils.get_descriptive_result_gpt / reasoning_utils.get_reasoning_result_gpt
#: @ 7ebe88f: gpt-4o-2024-05-13, response_format json_object, n=1, temperature 0, top_p 1,
#: seed 42, max_tokens 256 (doubled up to 1024 when the JSON is cut off).
JUDGE_REFERENCE = ("princeton-nlp/CharXiv src/descriptive_utils.py and src/reasoning_utils.py "
                   "@7ebe88f: gpt-4o-2024-05-13, response_format json_object, n=1, temperature 0, "
                   "top_p 1, seed 42, max_tokens 256 (doubled up to 1024 on a cut reply)")
JUDGE_TOP_P = 1.0
JUDGE_SEED = 42
#: descriptive_utils.build_descriptive_grading_queries(groups, nq_per_query=5), as evaluate.py
#: calls it: up to 5 answers of one template id per grading prompt.
DESCRIPTIVE_BATCH_SIZE = 5
#: get_descriptive_result_gpt / get_reasoning_result_gpt max_retries=10: calls per grading
#: prompt before its answers get upstream's dummy score -1 (JUDGE_FAILED here).
JUDGE_MAX_ATTEMPTS = 10


def charxiv_judge_config() -> Any:
    """The judge config of CharXiv's grading calls (see JUDGE_REFERENCE): JSON, temperature 0,
    top_p 1, seed 42. The max_tokens is not sent (see the module docstring)."""
    return judge_config(response_mime_type="application/json", top_p=JUDGE_TOP_P,
                        seed=JUDGE_SEED)


def judge_sampling_record() -> Dict[str, Any]:
    """The `judge_sampling` block of a CharXiv result."""
    return {"reference": JUDGE_REFERENCE, "temperature": 0.0, "top_p": JUDGE_TOP_P,
            "seed": JUDGE_SEED, "response_mime_type": "application/json",
            "max_output_tokens": ("not sent (upstream max_tokens=256, up to 1024, written for "
                                  "gpt-4o, which does not think; the Gemini judge counts its "
                                  "thinking against max_output_tokens)"),
            "descriptive_answers_per_prompt": DESCRIPTIVE_BATCH_SIZE,
            "max_attempts_per_prompt": JUDGE_MAX_ATTEMPTS}

PILLAR = "Multimodal & Vision"


# --------------------------------------------------------------------------- #
# Canonical helpers, ported verbatim from CharXiv src/descriptive_utils.py and
# src/reasoning_utils.py (behavior preserved exactly).
# --------------------------------------------------------------------------- #
def get_rubric(qid: int) -> str:
    """The in-context grading rubric for a descriptive template id (upstream get_rubric)."""
    if qid in (1,):
        return DESCRIPTIVE_GRADING_ICL['title']
    if qid in (2, 3, 4, 5, 6, 7):
        return DESCRIPTIVE_GRADING_ICL['ocr']
    if qid in (8, 9, 10, 12, 14, 15, 17, 19):
        return DESCRIPTIVE_GRADING_ICL['quant']
    if qid in (11,):
        return DESCRIPTIVE_GRADING_ICL['bool']
    if qid in (13,):
        return DESCRIPTIVE_GRADING_ICL['enum']
    if qid in (16,):
        return DESCRIPTIVE_GRADING_ICL['trend']
    if qid in (18,):
        return DESCRIPTIVE_GRADING_ICL['layout']
    raise ValueError(f"CharXiv: no rubric for descriptive qid {qid}")


def descriptive_query_helper(qid: int, subplot_loc: Any) -> str:
    """Build the descriptive question from template id + subplot location (upstream)."""
    if qid in (18, 19):
        # layout / #subplots questions take no subplot prefix
        return DESCRIPTIVE_RESP_INST[qid]
    if isinstance(subplot_loc, list):
        if subplot_loc[0] == 0:
            prefix = "For the current plot, "
        else:
            prefix = f"For the subplot at row {subplot_loc[0]} and column {subplot_loc[1]}, "
    elif isinstance(subplot_loc, str):
        prefix = f"For {subplot_loc}, "
    else:
        raise ValueError(f"CharXiv: invalid subplot_loc: {subplot_loc!r}")
    return DESCRIPTIVE_RESP_INST[qid].format(prefix)


def get_number_instruction(answer: str) -> str:
    """Decimal/integer format instruction for reasoning inst_category 4 (upstream)."""
    base = str(answer).split('.')
    whole, decimal = base[0], None if len(base) == 1 else base[1]
    if whole is not None and decimal is None:
        return "* Your final answer must be an exact integer."
    if whole is not None and decimal is not None:
        return f"* Your final answer must be a number with {len(decimal)} decimal places."
    raise ValueError(f"CharXiv: invalid answer for number instruction: {answer!r}")


def image_mime(b64: str) -> str:
    """The MIME type of base64 image bytes: PNG, GIF and WebP by their signatures, else
    image/jpeg (CharXiv's charts are .jpg files, and gpt.py labels every chart image/jpeg)."""
    try:
        head = base64.b64decode(b64[:24] + "=" * (-len(b64[:24]) % 4))
    except (binascii.Error, ValueError):
        return "image/jpeg"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _mk_sample(b64: Optional[str], question: str, gold_blob: Dict[str, Any],
               track: str) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """One (messages, gold, meta) sample: the question text, then the chart image, as gpt.py
    sends them. Scoring metadata rides in the gold JSON; the meta dict carries only
    `category`."""
    content: List[Dict[str, Any]] = [{"type": "text", "text": question}]
    if b64:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:{image_mime(b64)};base64,{b64}"}})
    return ([{"role": "user", "content": content}], json.dumps(gold_blob), {"category": track})


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chart_images_zip() -> str:
    """Local path of images.zip at DATASET_REVISION (hf_hub_download, cached), sha256-checked.
    A failed download is infra_required: the charts would otherwise not be the harness's."""
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=DATASET, filename=IMAGES_ZIP, repo_type="dataset",
                               revision=DATASET_REVISION)
    except Exception as e:                                          # noqa: BLE001
        raise infra_required(
            "charxiv", f"could not download {DATASET}@{DATASET_REVISION[:12]} {IMAGES_ZIP} "
            f"({type(e).__name__}: {e}), the chart files the official harness sends; check HF "
            "Hub access or pre-fetch it", DOCS_URL) from e
    digest = _sha256_file(path)
    if digest != IMAGES_ZIP_SHA256:
        raise RuntimeError(f"charxiv: {path} has sha256 {digest}, expected {IMAGES_ZIP_SHA256} "
                           f"({DATASET}@{DATASET_REVISION[:12]} {IMAGES_ZIP})")
    return path


def _load_charxiv_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load CharXiv validation and fan each chart into its 5 canonical questions. Each chart is
    sent as the file the official harness sends: images/{figure_id}.jpg from images.zip."""
    rows = []
    try:
        from datasets import load_dataset, Image
        ds = load_dataset(DATASET, split="validation", revision=DATASET_REVISION)
        try:
            ds = ds.cast_column("image", Image(decode=False))
        except Exception:
            pass
        rows = list(ds)
    except Exception as e:
        logger.error(f"Failed to load dataset for charxiv: {e}")
        raise RuntimeError(f"Could not load dataset for charxiv: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for charxiv returned empty rows")

    # --shard first, at the CHART level over the full ordered set (a chart's questions stay on one
    # node). Then --eval-limit N: N QUESTIONS for the suite, dealt over the two tracks
    # (sampling.allocate_limit), each track drawing its share stratified by chart subject.
    rows, shard_label = apply_shard(rows, "charxiv")

    def _loc(item: Dict[str, Any]) -> Any:
        loc = item.get("subplot_loc")
        return [item.get("subplot_row"), item.get("subplot_col")] if loc in (None, "", []) else loc

    # Every question of the shard, as (chart index, track, descriptive slot or None), in the
    # order a full run sends them: a chart's 4 descriptive questions, then its reasoning one.
    questions: List[Tuple[int, str, Optional[int]]] = []
    for r, item in enumerate(rows):
        questions += [(r, "descriptive", i) for i in range(1, 5)
                      if item.get(f"descriptive_q{i}") is not None]
        if str(item.get("reasoning_q") or "").strip() and item.get("reasoning_a_type") is not None:
            questions.append((r, "reasoning", None))
    if limit and limit > 0:
        by_track = {t: [q for q in questions if q[1] == t] for t in TRACKS}
        # the headline track (Reasoning, the result's accuracy) gets the first question
        shares = allocate_limit(limit, TRACKS, available={t: len(v) for t, v in by_track.items()},
                                seed="charxiv", first=("reasoning",))
        chosen = set()
        for track, share in shares.items():
            if not share:               # dealt none (0 would read as "no limit" below)
                continue
            chosen.update(stratified_sample(by_track[track], share,
                                            lambda q: (rows[q[0]] or {}).get("category"),
                                            seed=f"charxiv/{track}"))
        questions = [q for q in questions if q in chosen]

    samples: List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]] = []
    # figure_path is "images/<figure_id>.jpg"; images.zip holds "<figure_id>.jpg".
    def _member(item: Dict[str, Any]) -> str:
        return os.path.basename(str(item.get("figure_path") or ""))

    used_rows = sorted({r for r, _, _ in questions})
    with zipfile.ZipFile(chart_images_zip()) as zf:
        members = set(zf.namelist())
        charts = {m: base64.b64encode(zf.read(m)).decode("ascii")
                  for m in {_member(rows[r]) for r in used_rows} if m in members}
    for r in used_rows:
        if not charts.get(_member(rows[r])):
            # A chart question without its chart cannot be answered; sending it text-only would
            # score a different task.
            raise RuntimeError(f"charxiv: figure {rows[r].get('figure_path')!r} has no decodable "
                               f"image in {IMAGES_ZIP}")

    for r, track, i in questions:
        item = rows[r]
        b64 = charts[_member(item)]
        if track == "descriptive":
            # Descriptive: 4 questions/chart (template ids live in descriptive_q1..q4).
            qid = int(item.get(f"descriptive_q{i}"))
            gold = "" if item.get(f"descriptive_a{i}") is None else str(item.get(f"descriptive_a{i}"))
            question = descriptive_query_helper(qid, _loc(item))
            samples.append(_mk_sample(b64, question,
                                      {"track": "descriptive", "qid": qid, "answer": gold},
                                      "descriptive"))
            continue
        # Reasoning: 1 open-ended question/chart. The question and answer are used as the dataset
        # holds them (build_reasoning_queries / build_reasoning_grading_queries take d['query']
        # and data['answer'] unstripped); only the emptiness check strips.
        rq = str(item.get("reasoning_q") or "")
        ra = str(item.get("reasoning_a") or "")
        ic = int(item.get("reasoning_a_type"))
        if ic == 4:
            question = REASONING_RESP_INST[ic].format(rq, get_number_instruction(ra))
        else:
            question = REASONING_RESP_INST[ic].format(rq)
        samples.append(_mk_sample(b64, question,
                                  {"track": "reasoning", "inst_category": ic,
                                   "raw_question": rq, "answer": ra},
                                  "reasoning"))

    if not samples and not shard_label:     # an empty shard is base's honest empty row
        raise RuntimeError("CharXiv produced no questions")
    logger.info("Loaded %d charxiv questions (from %d charts, descriptive+reasoning).",
                len(samples), len(used_rows))
    return samples


# --------------------------------------------------------------------------- #
# Grading - build the exact upstream descriptive/reasoning grading prompts.
# --------------------------------------------------------------------------- #
def build_json_keys(length: int) -> str:
    """descriptive_utils.build_json_keys: the JSON keys a grading prompt of `length` answers asks
    for, as the string upstream interpolates."""
    keys = []
    for i in range(1, length + 1):
        keys.append(f"extract_answer_T{i}")
        keys.append(f"score_T{i}")
    return str(keys)


def populate_grading_inputs(batch: List[Tuple[Any, str, str]]) -> str:
    """descriptive_utils.populate_grading_inputs: one `T{i}` response / ground-truth triplet per
    (key, response, answer) of the batch."""
    query = ""
    for i, (_, response, answer) in enumerate(batch):
        query += "T{}:\nResponse {}: {}\nGround Truth {}: {}\n\n".format(
            i + 1, i + 1, response, i + 1, answer)
    return query


def _build_descriptive_grading_prompt(qid: int, batch: List[Tuple[Any, str, str]]) -> str:
    """The grading query build_descriptive_grading_queries sends for one batch of answers to
    template `qid`: the prefix with NUM_TRIPLETS, the overarching question and the batch's JSON
    keys, the template's rubric, then the triplets."""
    prefix = (DESCRIPTIVE_GRADING_PREFIX
              .replace("<|NUM_TRIPLETS|>", str(len(batch)))
              .replace("<|OVERARCHING_QUESTION|>", DESCRIPTIVE_GRADING_QMAP[qid])
              .replace("<|JSON_KEYS|>", build_json_keys(len(batch))))
    return prefix + get_rubric(qid) + populate_grading_inputs(batch)


def descriptive_batches(items: List[Tuple[int, Tuple[Any, str, str]]], num_templates: int = 19,
                        nq_per_query: int = DESCRIPTIVE_BATCH_SIZE
                        ) -> List[Tuple[int, List[Tuple[Any, str, str]]]]:
    """preprocess_descriptive_grading_queries + build_descriptive_grading_queries: the
    (qid, (key, response, answer)) items grouped by template id (1..num_templates, each group in
    the items' order), each group cut into batches of nq_per_query. Returns (qid, batch) pairs in
    upstream's query order."""
    groups: Dict[int, List[Tuple[Any, str, str]]] = {i: [] for i in range(1, num_templates + 1)}
    for qid, item in items:
        groups[qid].append(item)
    out = []
    for qid, data in groups.items():
        for i in range(0, len(data), nq_per_query):
            out.append((qid, data[i:i + nq_per_query]))
    return out


def _build_reasoning_grading_prompt(inst_category: int, raw_question: str,
                                    answer: str, resp: str) -> str:
    return (REASONING_GRADING_PREFIX
            + REASONING_GRADING_INST[inst_category]
            .replace("<|question|>", raw_question)
            .replace("<|ground_truth|>", answer)
            .replace("<|response|>", resp))


def verify_grading_output(data: Any, length: int) -> bool:
    """descriptive_utils.verify_grading_output: every extract_answer_T{i} and score_T{i} key is
    present and every score is in [0, 1]."""
    if not isinstance(data, dict):
        return False
    for i in range(1, length + 1):
        if f"extract_answer_T{i}" not in data or f"score_T{i}" not in data:
            return False
        if data[f"score_T{i}"] not in [0, 1]:
            return False
    return True


def _parse_descriptive_reply(text: str, length: int) -> Optional[Dict[str, Any]]:
    """get_descriptive_result_gpt's acceptance test: json.loads of the whole reply, then
    verify_grading_output. None when upstream would ask again."""
    try:
        content = json.loads(text)
    except Exception:                                           # noqa: BLE001
        return None
    return content if verify_grading_output(content, length) else None


def _parse_reasoning_reply(text: str) -> Optional[Tuple[Any, Any]]:
    """get_reasoning_result_gpt's acceptance test: json.loads of the whole reply and its
    `extracted_answer` and `score` keys (the score's value is checked later, as get_stats does).
    None when upstream would ask again."""
    try:
        content = json.loads(text)
        return content["extracted_answer"], content["score"]
    except Exception:                                           # noqa: BLE001
        return None


async def _grade_prompt(prompt: str, parse, cfg: Any, semaphore, usage: Counter
                        ) -> Tuple[str, Any, Optional[str], Optional[str], int]:
    """Ask the judge cascade for one grading prompt, up to JUDGE_MAX_ATTEMPTS calls while the
    reply fails `parse`, as upstream's retry loop does.

    Returns (outcome, parsed, raw reply, model, calls): outcome "ok", "failed" (no reply passed
    `parse`) or "outage" (the cascade was exhausted)."""
    raw: Optional[str] = None
    model: Optional[str] = None
    for attempt in range(1, JUDGE_MAX_ATTEMPTS + 1):
        async with semaphore:
            text, used = await judge_generate_cascade(prompt, config=cfg)
        if text is None:
            return "outage", None, raw, model, attempt
        usage[used] += 1
        raw, model = text, used
        parsed = parse(text)
        if parsed is not None:
            return "ok", parsed, raw, model, attempt
    return "failed", None, raw, model, JUDGE_MAX_ATTEMPTS


def _eval_charxiv(response_text: str, gold_target: str) -> bool:
    """No-key/phase-1 fallback: relaxed match on the answer inside the gold blob.

    Canonical CharXiv is judge-graded; `run_charxiv` hard-requires GEMINI_API_KEY, so this
    only stands in for the (skipped) no-judge path and the pre-judge phase-1 pass.
    """
    from .vqa_common import eval_relaxed, extract_short_answer
    if not response_text or not gold_target:
        return False
    try:
        answer = str(json.loads(gold_target).get("answer") or "")
    except Exception:
        answer = str(gold_target)
    return eval_relaxed(extract_short_answer(response_text), answer)


def _set_verdict(trace: Dict[str, Any], outcome: str, score: Any, extracted: Any,
                 raw: Optional[str], model: Optional[str], batch: Dict[str, Any]) -> None:
    """Record one answer's grade. `score` is the judge's score for it (None on an outage or when
    no reply was accepted)."""
    trace["status"] = "OK"
    trace["judge_batch"] = batch
    if model is not None:
        trace["judge_model"] = model
    if raw is not None:
        trace["judge_raw"] = raw
    if outcome == "outage":
        # Judge cascade exhausted (infra outage): excluded from the denominator.
        trace["judge_grade"] = "JUDGE_OUTAGE"
        return
    if outcome == "failed" or score not in [0, 1]:
        # Upstream scores this answer -1 (get_stats counts it as invalid and as 0). The grader is
        # gbench's Gemini substitute, so a verdict it did not give readably is a judge failure:
        # left out of the denominator, counted in judge_failures and named.
        trace["is_correct"] = False
        trace["scoring_excluded"] = True
        trace["judge_grade"] = "JUDGE_FAILED"
        trace["judge_score"] = -1 if outcome == "failed" else score
        trace["extracted_answer"] = "Failed to parse response" if outcome == "failed" else extracted
        return
    trace["is_correct"] = score == 1
    trace["judge_grade"] = "CORRECT" if score == 1 else "INCORRECT"
    trace["judge_score"] = score
    trace["extracted_answer"] = extracted


async def _async_judge_charxiv(
    sample_traces: List[Dict[str, Any]],
    judge_concurrency: int = 32,
    judge_usage: Optional[Counter] = None,
) -> None:
    """Canonical descriptive + reasoning grading via the gbench judge cascade (JSON scores):
    descriptive answers batched by template id, up to 5 per prompt; reasoning answers one per
    prompt. `judge_usage`, when given, counts the answered judge calls per model."""
    import asyncio
    from tqdm import tqdm

    # Read like the gate reads it (GEMINI_API_KEYS or GEMINI_API_KEY), so a key the gate
    # accepted can never leave this on the relaxed-match path.
    have_key = bool(gemini_api_keys())
    semaphore = asyncio.Semaphore(judge_concurrency)
    cfg = charxiv_judge_config()
    usage = judge_usage if judge_usage is not None else Counter()

    descriptive: Dict[Any, List[Tuple[int, Tuple[Any, str, str]]]] = {}
    reasoning: List[Tuple[Dict[str, Any], Dict[str, Any], str]] = []
    # Dataset order (sample_idx); within one attempt, as upstream has one reply per question.
    for trace in sorted(sample_traces, key=lambda t: t.get("sample_idx", 0)):
        try:
            blob = json.loads(trace.get("gold_answer") or "{}")
        except Exception:
            blob = {}
        track = blob.get("track") or trace.get("category")
        answer = str(blob.get("answer") or "")
        resp = trace.get("response_text")

        if resp is None or trace.get("harness_error") or (not resp and track != "descriptive"):
            # No reply from the endpoint (a harness error base leaves out), or an empty reasoning
            # answer (nothing to grade): not sent to the judge.
            trace["is_correct"] = False
            trace["judge_grade"] = "FAILED"
            trace["status"] = "OK"
            continue

        if not have_key:
            # Dead path (gemini_required_skip gates first), kept defensively.
            trace["is_correct"] = _eval_charxiv(resp, trace.get("gold_answer"))
            trace["scoring_mode"] = "judge_fallback"
            trace["status"] = "OK"
            continue

        if track == "descriptive":
            descriptive.setdefault(trace.get("attempt_idx", 0), []).append(
                (int(blob["qid"]), (trace, resp, answer)))
        else:
            reasoning.append((trace, blob, answer))

    jobs = []
    for attempt in sorted(descriptive):
        for qid, batch in descriptive_batches(descriptive[attempt]):
            jobs.append(("descriptive", qid, batch))
    jobs += [("reasoning", None, item) for item in reasoning]

    async def _descriptive(qid: int, batch: List[Tuple[Any, str, str]], pbar: tqdm) -> None:
        prompt = _build_descriptive_grading_prompt(qid, batch)
        outcome, content, raw, model, calls = await _grade_prompt(
            prompt, lambda text: _parse_descriptive_reply(text, len(batch)), cfg, semaphore,
            usage)
        for i, (trace, _, _) in enumerate(batch, start=1):
            score = content.get(f"score_T{i}") if content else None
            extracted = content.get(f"extract_answer_T{i}") if content else None
            _set_verdict(trace, outcome, score, extracted, raw, model,
                         {"template": qid, "position": i, "size": len(batch), "calls": calls})
        pbar.update(len(batch))

    async def _reasoning(item: Tuple[Dict[str, Any], Dict[str, Any], str], pbar: tqdm) -> None:
        trace, blob, answer = item
        prompt = _build_reasoning_grading_prompt(
            int(blob["inst_category"]), str(blob.get("raw_question") or ""), answer,
            trace.get("response_text"))
        outcome, fields, raw, model, calls = await _grade_prompt(
            prompt, _parse_reasoning_reply, cfg, semaphore, usage)
        extracted, score = fields if fields else (None, None)
        _set_verdict(trace, outcome, score, extracted, raw, model,
                     {"position": 1, "size": 1, "calls": calls})
        pbar.update(1)

    total = sum(len(j[2]) if j[0] == "descriptive" else 1 for j in jobs)
    with tqdm(total=total, desc="Judging [CHARXIV]") as pbar:
        await asyncio.gather(*[_descriptive(qid, item, pbar) if kind == "descriptive"
                               else _reasoning(item, pbar) for kind, qid, item in jobs])


def run_charxiv(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    results_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run native CharXiv (descriptive + reasoning tracks, canonical rubrics)."""
    skip = gemini_required_skip("charxiv", model_name)
    if skip is not None:
        return skip
    samples = _load_charxiv_samples(limit=limit)
    plan = eval_limit_plan(
        "charxiv", limit, len(samples),
        {t: sum(1 for _m, _g, meta in samples if meta.get("category") == t) for t in TRACKS},
        setting_noun="track", attempts=base_attempts(batch_scored=True))
    used: Counter = Counter()
    res = run_eval_suite(
        eval_name="charxiv",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's charts, kept as is
        eval_fn=_eval_charxiv,
        async_eval_fn=functools.partial(_async_judge_charxiv, judge_usage=used),
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=None,  # the loader already dealt the --eval-limit questions over the tracks
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=OFFICIAL_SAMPLING,   # its max_tokens follows the token-limit policy
    )

    # Canonical reporting: descriptive & reasoning accuracy SEPARATELY (no combined figure).
    # Category == track; judge outages are excluded from each denominator.
    res["eval_limit_plan"] = plan
    traces = res.get("sample_traces", []) or []

    def _track(name: str) -> Dict[str, Any]:
        rows = [t for t in traces if t.get("category") == name]
        scored = [t for t in rows if t.get("judge_grade") != "JUDGE_OUTAGE"
                  and not t.get("scoring_excluded")]
        correct = sum(1 for t in scored if t.get("is_correct"))
        failed = [t for t in rows if t.get("judge_grade") == "JUDGE_FAILED"]
        # upstream's get_stats: a -1 (JUDGE_FAILED) answer counted as 0 in the denominator
        upstream_n = len(scored) + len(failed)
        return {"questions": len(rows), "scored": len(scored), "correct": correct,
                "accuracy": round(100.0 * correct / len(scored), 2) if scored else None,
                "accuracy_upstream_invalid_as_zero": (round(100.0 * correct / upstream_n, 2)
                                                      if upstream_n else None),
                "judge_outages": sum(1 for t in rows if t.get("judge_grade") == "JUDGE_OUTAGE"),
                "judge_failed": len(failed)}

    tracks = {"reasoning": _track("reasoning"), "descriptive": _track("descriptive")}
    res["charxiv_tracks"] = tracks
    res["charxiv_descriptive_accuracy"] = tracks["descriptive"]["accuracy"]
    res["charxiv_reasoning_accuracy"] = tracks["reasoning"]["accuracy"]
    # Headline = Reasoning (paper's flagship + leaderboard sort key); descriptive when no reasoning
    # answer was scored; withheld (None) when neither track has a scored answer. The question and
    # correct counts beside it are the headline track's; the pooled counts stay in
    # questions_all_tracks / correct_all_tracks.
    headline = next((n for n in ("reasoning", "descriptive")
                     if tracks[n]["accuracy"] is not None), None)
    res["questions_all_tracks"] = res.get("total_questions", len(traces))
    res["correct_all_tracks"] = res.get("correct_answers",
                                        sum(1 for t in traces if t.get("is_correct")))
    res["headline_track"] = headline
    if headline is None:
        res["accuracy"] = None
        res["accuracy_withheld_reason"] = "no CharXiv answer was graded in either track"
        if res.get("status") not in ("error", "failed", "timeout"):
            res["status"] = "failed"
            res["status_reason"] = ("nothing was scored: " + (res.get("status_reason")
                                                              or "no answer was graded"))
    else:
        res["accuracy"] = tracks[headline]["accuracy"]
        res["total_questions"] = tracks[headline]["questions"]
        res["correct_answers"] = tracks[headline]["correct"]
        res["effective_n"] = tracks[headline]["scored"]
    res["metric"] = ("CharXiv reasoning accuracy (headline) + descriptive accuracy; canonical "
                     "per-qid / per-inst_category rubrics graded by gbench's standard Gemini judge "
                     "cascade (the paper's grader is GPT-4o; gbench-internal, not a like-for-like "
                     "leaderboard entry)")
    reasons = ["graded by gbench's standard Gemini judge cascade instead of CharXiv's "
               "gpt-4o-2024-05-13 (a gbench convention); prompts and rubrics are upstream's",
               "the judge call's max_tokens (256, doubled up to 1024 on a cut reply) is not sent: "
               "it was written for gpt-4o, which does not think, and the Gemini judge counts its "
               "thinking against it"]
    if limit:
        reasons.append(f"--eval-limit {limit} subset ({len(samples)} of the 5,000 questions, "
                       "dealt over the descriptive and reasoning tracks); descriptive answers are "
                       "batched for grading within this subset, not as in the full set")
    if os.environ.get("GBENCH_SHARD"):
        reasons.append(f"GBENCH_SHARD {os.environ['GBENCH_SHARD']} shard (descriptive grading "
                       "batches formed within the shard)")
    official = res.get("official_sampling") or {}
    if official.get("temperature_matches_official") is False:
        reasons.append(f"temperature {res.get('temperature')} (CharXiv's "
                       f"{'reasoning-model' if enable_thinking else 'chat-model'} client sends "
                       f"{official.get('temperature')})")
    failed = sum(t["judge_failed"] for t in tracks.values())
    if failed:
        reasons.append(f"{failed} answers got no valid judge score in {JUDGE_MAX_ATTEMPTS} "
                       "attempts (judge failures, left out of the track accuracies; upstream "
                       "scores them -1, counted as 0: accuracy_upstream_invalid_as_zero in "
                       "charxiv_tracks)")
    outages = sum(t["judge_outages"] for t in tracks.values())
    if outages:
        reasons.append(f"{outages} answers were not graded (judge cascade outage; excluded)")
    res["judge"] = "gbench-gemini-cascade"
    res["judge_models"] = dict(used)      # answered judge calls per model, retries included
    res["judge_cascade"] = judge_cascade()
    res["judge_sampling"] = judge_sampling_record()
    res["dataset"] = f"{DATASET}@{DATASET_REVISION} (validation)"
    res["chart_images"] = {"file": f"{DATASET}@{DATASET_REVISION[:12]} {IMAGES_ZIP}",
                           "sha256": IMAGES_ZIP_SHA256,
                           "member": "<figure_id>.jpg (the harness's images/{figure_id}.jpg)"}
    cut = record_budget_cutoffs(res)
    if cut:
        reasons.append(cut)
    res["leaderboard_comparable"] = False
    res["leaderboard_comparable_reason"] = "; ".join(reasons)
    return res
