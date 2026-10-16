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

# Evaluation Suite: aa_lcr
# Description: AA-LCR (Artificial Analysis Long Context Reasoning Benchmark - 100 multi-document reasoning questions)

"""gbench native built-in runner for aa_lcr (Long Context & Retrieval).

AA-LCR v1.1 (`ArtificialAnalysis/AA-LCR`, pinned to the v1.1 revision): 100 questions, each
asked over its own document set of ~100k tokens. The prompt template, the document order
(`data_source_filenames`) and the v1.1 equality-checker prompts (system + user) are the
dataset card's, verbatim. Artificial Analysis runs 3 repeats and reports pass@1 aggregated
across them, so the suite runs at least 3 attempts per question and the headline is avg@k.
The equality checker is the gbench Gemini judge cascade instead of AA's GPT-5.6 Luna
(medium), so the result carries leaderboard_comparable=False.

Sampling: Artificial Analysis's General Testing Parameters (Intelligence Benchmarking
methodology): temperature 0 for non-reasoning models, and for reasoning models 0.6 unless the
model lab recommends another temperature. The request follows gbench's rule
(base.resolve_temperature): 1.0 with `--thinking`, 0.0 without; the official value is recorded
(base.OfficialSampling): 0 for no-think runs, and with `--thinking` the lab's recommendation: the
temperature the served model's shipped `generation_config.json` sets (read from --tokenizer, else
the served model id, ojbench.served_model_recommended_sampling), else AA's 0.6; a config that
cannot be read is infra_required. `official_sampling` records the official value, its source
and whether 1.0 matched it (it does for Gemma 4, whose generation_config.json sets 1.0). Override it
for
a whole run with `--temperature`, or for this suite alone with `GBENCH_AA_LCR_TEMPERATURE`, which
takes precedence over both; the result records `official_sampling`. The same parameters set the
maximum output tokens: 16,384 for non-reasoning models, adjusted downward where the context window
is smaller, so a no-think request carries max_tokens 16,384 (shrunk to what the window leaves) and
`--max-output-tokens` is not applied; for reasoning models the maximum the model allows, so a
`--thinking` request carries no max_tokens and may use the rest of the context window.

The equality checker call sends the card's system and user prompts and nothing else the card
does not publish: no JSON mode, and no temperature, since the official checker, GPT-5.6 Luna
(medium), is a reasoning model, which takes none. The result records it under `judge_sampling`.
Every judged attempt keeps the checker's replies and the Gemini model that gave each
(`judge_replies`).

Harness errors are never the model's outcome: an attempt whose request produced no reply (an
endpoint error, a timeout, a prompt longer than the served window) or whose checker gave no
CORRECT/INCORRECT verdict in _JUDGE_ATTEMPTS replies is left out of the accuracy (numerator and
denominator), counted in `harness_errors` and named in `leaderboard_comparable_reason`; a run with
no attempt scored withholds its score (accuracy null, status failed).
"""

import asyncio
import csv
import functools
import io
import json
import logging
import re
import unicodedata
import zipfile
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from .limit_plan import eval_limit_plan
from .base import (apply_shard, finalize_shard_fields, run_eval_suite, gemini_api_keys,
                   gemini_required_skip, judge_cascade, judge_config, judge_generate_cascade,
                   strip_thinking_tags, OfficialSampling)
from .metrics import record_budget_cutoffs
from .sampling import stratified_sample, presharded_for_base

logger = logging.getLogger(__name__)

PILLAR = "Long Context & Retrieval"
DOCS_URL = "docs/evals/aa_lcr.md"

_DATASET = "ArtificialAnalysis/AA-LCR"
#: v1.1 (2026-09-04): 16 corrected answer keys and the judge system prompt. v1.0 scores are not
#: comparable with v1.1 ones; v1.0 stays at bdae010bbce259820c0e34c1d7cce210d966fb75.
_REVISION = "9a77ef56b717057ade24ceab4d273712a0b4f19e"
_CSV_FILE = "AA-LCR_Dataset.csv"
_ZIP_FILE = "extracted_text/AA-LCR_extracted-text.zip"

#: Artificial Analysis Intelligence Index methodology: 3 repeats, pass@1 aggregated across them.
OFFICIAL_REPEATS = 3
OFFICIAL_JUDGE = "GPT-5.6 Luna (medium)"


#: General Testing Parameters, "Maximum output tokens": non-reasoning models 16,384 (adjusted
#: downward where models have a smaller context window).
NON_REASONING_MAX_TOKENS = 16384
_METHODOLOGY = ("Artificial Analysis Intelligence Benchmarking methodology, General Testing "
                "Parameters")


#: General Testing Parameters, reasoning models: "temperature 0.6 unless the lab recommends
#: another".
AA_REASONING_TEMPERATURE = 0.6


def official_sampling(thinking: bool, recommended: Optional[Dict[str, Any]] = None,
                      source: Optional[str] = None) -> OfficialSampling:
    """Artificial Analysis's testing parameters for the run's mode (see the module docstring).
    With `thinking`, `recommended` / `source` are the served model's generation_config.json
    fields and where they were read (reasoning_sampling reads them): its temperature when it sets
    one, else AA's 0.6."""
    if thinking:
        lab = (recommended or {}).get("temperature")
        if lab is not None:
            where = f"the lab's recommendation, temperature {lab}, read from {source}"
        elif source:
            where = f"{source} sets no temperature, so AA's 0.6"
        else:
            where = "no generation_config.json read, so AA's 0.6"
        return OfficialSampling(
            reference=(f"{_METHODOLOGY}, reasoning models: temperature 0.6 unless the lab "
                       f"recommends another (the shipped generation_config.json): {where}; "
                       "maximum output tokens allowed (no max_tokens is sent)"),
            temperature=float(lab) if lab is not None else AA_REASONING_TEMPERATURE,
            send_max_tokens=False)
    return OfficialSampling(
        reference=(f"{_METHODOLOGY}, non-reasoning models: temperature 0; maximum output tokens "
                   "16,384, adjusted downward where the context window is smaller"),
        temperature=0.0, max_tokens=NON_REASONING_MAX_TOKENS)


def reasoning_sampling(model_name: str) -> OfficialSampling:
    """The --thinking OfficialSampling for the served model: its generation_config.json
    temperature (ojbench.served_model_recommended_sampling: --tokenizer, else the served model
    id), else AA's 0.6. A config that cannot be read is infra_required: the lab's recommendation
    is unknown, and 0.6 would then be a guess."""
    from .ojbench import served_model_recommended_sampling
    from .swebench_common import infra_required
    try:
        recommended, source = served_model_recommended_sampling(model_name)
    except RuntimeError as e:
        raise infra_required(
            "aa_lcr", "Artificial Analysis runs a reasoning model at the temperature its lab "
            f"recommends (the shipped generation_config.json), else 0.6, and {e}",
            DOCS_URL) from e
    return official_sampling(True, recommended, source)


#: The dataset card's "Scoring Approach" (v1.1): the checker model and its two prompts.
JUDGE_REFERENCE = (f"{_DATASET}@{_REVISION[:8]} dataset card, Scoring Approach (v1.1): the system "
                   f"and user prompts, checker {OFFICIAL_JUDGE}")


def judge_sampling_record() -> Dict[str, Any]:
    """The `judge_sampling` block of an AA-LCR result."""
    return {"reference": JUDGE_REFERENCE, "system_instruction": "the card's system prompt",
            "temperature": ("not sent (the official checker is a reasoning model, which takes "
                            "no temperature)"),
            "response_mime_type": "not sent (the card specifies none; the prompt asks for JSON)",
            "reasoning_effort": ("medium belongs to the official checker model; the Gemini "
                                 "cascade runs at its own default")}


def _checker_config() -> Any:
    """The equality checker's config: the card's system prompt, no temperature, no JSON mode."""
    cfg = judge_config(system_instruction=JUDGE_SYSTEM_PROMPT)
    return cfg.model_copy(update={"temperature": None}) if cfg is not None else None


#: A judge reply with no CORRECT/INCORRECT verdict is asked again this many times in total
#: before the sample is marked JUDGE_FAILED.
_JUDGE_ATTEMPTS = 3

# Dataset card, "Prompt Template" (verbatim).
PROMPT_TEMPLATE = """BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""

# Dataset card, "Scoring Approach": the v1.1 equality-checker system prompt (verbatim).
JUDGE_SYSTEM_PROMPT = """Decide whether the CANDIDATE ANSWER is correct or incorrect against the OFFICIAL ANSWER.
Note the following points when assessing correctness:

- Numbers should still match when they are the same value written differently, e.g., a
  percentage, a count of percentage points, and the equivalent decimal fraction are the same
  value: 0.675, "67.5%" and "67.5 percentage points" all match. So do different scales
  (thousand, million, bn) and different notations (thousands separators, currency symbols,
  LaTeX markup, and numbers written as words).
- Where the question asks for a particular format (e.g., a percentage, a number of decimal
  places, a unit, a rounding, or an ordering) the CANDIDATE ANSWER must meet it. If the
  question asks for no particular format, accept any equivalent form.
- In cases where the question asks for an ordered list, a title, honorific or article added
  to an entry in the CANDIDATE ANSWER can change where that entry sorts. Accept the ordering
  if it is correct either with those additions or without them.
- Grade the value the CANDIDATE ANSWER finally commits to, and it must commit to one. Values
  reached while working, and alternatives it considers and sets aside, do not count. If it
  offers several values without selecting one, it is incorrect even if one of them is right.
  Hedging is fine as long as one clearly definitive answer is given."""

# Dataset card, "Scoring Approach": the v1.1 equality-checker user prompt (verbatim).
JUDGE_USER_TEMPLATE = """Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: START QUESTION {question}

END QUESTION

The OFFICIAL ANSWER: {official_answer}

END OFFICIAL ANSWER

BEGIN CANDIDATE ANSWER TO ASSESS

{candidate_answer}

END CANDIDATE ANSWER TO ASSESS

Reply as JSON, with a verdict of CORRECT or INCORRECT."""


def _zip_entry_name(info: zipfile.ZipInfo) -> str:
    """An archive entry's name as its author wrote it.

    Entries without the UTF-8 flag (0x800) are decoded by zipfile as cp437, which turns this
    archive's non-ASCII filenames (an em dash, a curly apostrophe) into mojibake that never
    matches the CSV. Re-decode those as UTF-8, then NFC-normalize: some names are stored
    decomposed while the CSV lists them composed.
    """
    name = info.filename
    if not info.flag_bits & 0x800:
        try:
            name = name.encode("cp437").decode("utf-8")
        except UnicodeError:
            pass  # a genuine cp437 name
    return _nfc(name)


def _nfc(text: Any) -> str:
    return unicodedata.normalize("NFC", str(text or ""))


def _load_documents(zip_path: str) -> Dict[Tuple[str, str, str], str]:
    """{(document_category, document_set_id, filename): text} for every document in the archive.

    Keyed by the full path, as upstream reads `{category}/{set_id}/{filename}`: a basename
    alone repeats across document sets. Read as text with universal newlines, like open().
    """
    docs: Dict[Tuple[str, str, str], str] = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = _zip_entry_name(info).split("/")
            if len(parts) < 3:
                continue
            with zf.open(info) as fh:
                docs[tuple(parts[-3:])] = io.TextIOWrapper(fh, encoding="utf-8").read()
    return docs


def build_prompt(docs: List[str], question: str) -> str:
    """The dataset card's prompt: numbered documents in data_source_filenames order, then the question."""
    documents_text = "\n\n".join(
        f"BEGIN DOCUMENT {i + 1}:\n{doc}\nEND DOCUMENT {i + 1}" for i, doc in enumerate(docs))
    return PROMPT_TEMPLATE.format(documents_text=documents_text, question=question)


def _load_aa_lcr_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load AA-LCR v1.1 from the pinned HF revision; raises if any listed document is missing."""
    try:
        from huggingface_hub import hf_hub_download
        csv_file = hf_hub_download(repo_id=_DATASET, filename=_CSV_FILE, repo_type="dataset",
                                   revision=_REVISION)
        zip_path = hf_hub_download(repo_id=_DATASET, filename=_ZIP_FILE, repo_type="dataset",
                                   revision=_REVISION)
        docs = _load_documents(zip_path)
        with open(csv_file, "r", encoding="utf-8") as fp:
            rows = list(csv.DictReader(fp))
    except Exception as e:
        logger.error(f"Failed to load dataset for aa_lcr: {e}")
        raise RuntimeError(f"Could not load dataset for aa_lcr: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for aa_lcr returned empty rows")

    # --shard first, over the full ordered set, then the --eval-limit draw within the shard.
    rows, shard_label = apply_shard(rows, "aa_lcr")
    # Stratified, not a contiguous head.
    rows = stratified_sample(rows, limit, lambda r: (r or {}).get("document_category"), seed="aa_lcr")

    samples, missing = [], []
    for item in rows:
        category = item.get("document_category") or ""
        set_id = item.get("document_set_id") or ""
        texts = []
        for fn in str(item.get("data_source_filenames") or "").split(";"):
            key = (_nfc(category), _nfc(set_id), _nfc(fn))
            if key in docs:
                texts.append(docs[key])
            else:
                missing.append(f"question {item.get('question_id')}: {category}/{set_id}/{fn}")
        question = item.get("question") or ""
        messages = [{"role": "user", "content": build_prompt(texts, question)}]
        samples.append((messages, item.get("answer") or "", {
            "category": category, "question_id": item.get("question_id") or "",
            # The judge sees the raw question, not the 100k-token prompt.
            "question": question}))

    # Every listed document is part of the question: dropping one leaves a question whose
    # evidence is gone, which is a data error, not a sample to score.
    if missing:
        raise RuntimeError(
            f"aa_lcr: {len(missing)} listed source document(s) are not in {_ZIP_FILE} at "
            f"revision {_REVISION[:7]}: " + "; ".join(missing[:5]))
    if not samples and not shard_label:     # an empty shard is base's honest empty row
        raise RuntimeError("Dataset for aa_lcr returned empty rows")

    logger.info(f"Loaded {len(samples)} aa_lcr samples.")
    return samples


def parse_verdict(text: Optional[str]) -> Optional[str]:
    """'CORRECT' / 'INCORRECT' from the checker's reply, or None when it gave neither
    (parse_verdict_detailed)."""
    return parse_verdict_detailed(text)[0]


#: A verdict word on its own (upper case, as the card's prompt spells it).
_VERDICT_WORD = re.compile(r"(?<![A-Za-z])(INCORRECT|CORRECT)(?![A-Za-z])")


def parse_verdict_detailed(text: Optional[str]) -> Tuple[Optional[str], bool]:
    """(verdict, read by the relaxed parse): the verdict of the checker's JSON object; when the
    reply holds no such object, a relaxed read that accepts exactly one whole-word CORRECT or
    INCORRECT (a bare word, a JSON string, "Verdict: INCORRECT"; markup stripped), never a reply
    naming both (a format echo such as "CORRECT or INCORRECT"). (None, False) otherwise."""
    verdict = _json_verdict(text)
    if verdict is not None:
        return verdict, False
    body = re.sub(r"[`*_\"']", " ", str(text or ""))
    if re.search(r"[{}]", body):
        return None, False          # a JSON object whose verdict could not be read
    found = set(_VERDICT_WORD.findall(body))
    return (found.pop(), True) if len(found) == 1 else (None, False)


def _json_verdict(text: Optional[str]) -> Optional[str]:
    """'CORRECT' / 'INCORRECT' from the checker's JSON object, or None."""
    body = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", body, re.DOTALL)
    if fenced:
        body = fenced.group(1)
    m = re.search(r"\{.*\}", body, re.DOTALL)
    try:
        obj = json.loads(m.group(0)) if m else None
    except ValueError:
        obj = None
    if not isinstance(obj, dict):
        return None
    values = [v for k, v in obj.items() if str(k).lower() == "verdict"] or list(obj.values())
    found = {str(v).strip().upper() for v in values if isinstance(v, str)} & {"CORRECT", "INCORRECT"}
    return found.pop() if len(found) == 1 else None


#: The judge_grade of a trace that ended in a harness error (its request produced no reply).
HARNESS_ERROR_GRADE = "HARNESS_ERROR"
#: base's statuses of a request that produced no reply, and the harness error each is.
_NO_REPLY_KINDS = {"FAILED": "endpoint_failure", "TIMEOUT": "endpoint_timeout",
                   "OVER_CONTEXT": "over_context"}
#: Characters of a checker reply kept on a trace.
_JUDGE_REPLY_CHARS = 4000


async def _async_judge_aa_lcr(
    sample_traces: List[Dict[str, Any]],
    judge_concurrency: int = 64,
    judge_usage: Optional[Counter] = None,
) -> None:
    """AA-LCR v1.1 equality checker (card system + user prompts) over the Gemini judge cascade.

    `judge_usage`, when given, counts the answered judge calls per model (retries included).
    Outcomes per trace:
      * a request that produced no reply (base's FAILED, TIMEOUT or OVER_CONTEXT): a harness
        error, judge_grade HARNESS_ERROR and `scoring_excluded`, never the model's answer;
      * an empty reply the server did return: EMPTY_RESPONSE, incorrect (the model's outcome);
      * every cascade model failed: JUDGE_OUTAGE (excluded from the accuracy);
      * no CORRECT/INCORRECT verdict after _JUDGE_ATTEMPTS replies: JUDGE_FAILED, a harness error
        (`scoring_excluded`), never scored INCORRECT;
      * else the checker's verdict.
    Every judged trace keeps the checker's replies with the model that gave each
    (`judge_replies`) and the model of the last one (`judge_model`).
    """
    from tqdm import tqdm

    if not gemini_api_keys():
        # run_aa_lcr's gate already hard-errors on a missing key; never degrade to string matching.
        from .swebench_common import infra_required
        raise infra_required("aa_lcr", "requires GEMINI_API_KEY (or GEMINI_API_KEYS) for the "
                             "AA-LCR equality checker", DOCS_URL)
    usage = judge_usage if judge_usage is not None else Counter()

    semaphore = asyncio.Semaphore(judge_concurrency)
    config = _checker_config()

    async def _judge_single(trace: Dict[str, Any], pbar: tqdm) -> None:
        status = str(trace.get("status") or "")
        if status in _NO_REPLY_KINDS:
            trace["is_correct"] = False
            trace["judge_grade"] = HARNESS_ERROR_GRADE
            trace["harness_error"] = {"kind": _NO_REPLY_KINDS[status],
                                      "detail": str(trace.get("error") or "no reply")[:300]}
            trace["scoring_excluded"] = True
            pbar.update(1)
            return
        candidate = strip_thinking_tags(trace.get("response_text"))
        if trace.get("status") == "PENDING_JUDGE":   # keep EMPTY_RESPONSE etc.
            trace["status"] = "OK"
        if not candidate:
            trace["is_correct"] = False
            trace["judge_grade"] = "EMPTY_RESPONSE"
            pbar.update(1)
            return
        meta = trace.get("extra_payload") or {}
        prompt = JUDGE_USER_TEMPLATE.format(
            question=meta.get("question", ""),
            official_answer=str(trace.get("gold_answer") or ""),
            candidate_answer=candidate)
        verdict, model = None, None
        replies: List[Dict[str, Any]] = []
        for _ in range(_JUDGE_ATTEMPTS):
            async with semaphore:
                text, model = await judge_generate_cascade(prompt, config=config)
            if text is None:
                # Judge cascade exhausted (infra outage): excluded from base's accuracy
                # denominator, NOT scored INCORRECT.
                trace["judge_grade"] = "JUDGE_OUTAGE"
                if replies:
                    trace["judge_replies"] = replies
                pbar.update(1)
                return
            usage[model] += 1
            replies.append({"judge_model": model, "reply": text[:_JUDGE_REPLY_CHARS]})
            verdict, relaxed = parse_verdict_detailed(text)
            if verdict is not None:
                if relaxed:
                    # a clear verdict that is not a JSON object: read, not asked again
                    trace["judge_verdict_relaxed_parse"] = True
                break
        trace["judge_model"] = model
        trace["judge_replies"] = replies
        if verdict is None:
            # The checker answered but gave no verdict: a harness error, not a wrong answer.
            trace["is_correct"] = False
            trace["judge_grade"] = "JUDGE_FAILED"
            trace["scoring_excluded"] = True
        else:
            trace["is_correct"] = verdict == "CORRECT"
            trace["judge_grade"] = verdict
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Judging [AA_LCR]") as pbar:
        await asyncio.gather(*[_judge_single(trace, pbar) for trace in sample_traces])


def run_aa_lcr(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run native aa_lcr evaluation suite."""
    skip = gemini_required_skip("aa_lcr", model_name)
    if skip is not None:
        return skip
    samples = _load_aa_lcr_samples(limit=limit)
    # AA reports pass@1 over 3 repeats; a larger explicit --attempt-count is honoured.
    attempts = max(OFFICIAL_REPEATS, int(kwargs.get("attempt_count") or 1))
    used: Counter = Counter()
    eval_plan = eval_limit_plan("aa_lcr", limit, len(samples), attempts=attempts)
    result = run_eval_suite(
        eval_name="aa_lcr",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's questions, kept as is
        async_eval_fn=functools.partial(_async_judge_aa_lcr, judge_usage=used),
        limit=limit,
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        max_output_tokens=kwargs.get("max_output_tokens"),
        attempt_count=attempts,
        supports_attempts=True,   # every attempt is judged on its own
        official_sampling=(reasoning_sampling(model_name) if enable_thinking
                           else official_sampling(False)),
    )
    result["eval_limit_plan"] = eval_plan
    result["dataset"] = f"{_DATASET}@{_REVISION} (v1.1)"
    result["metric"] = (f"accuracy: pass@1 averaged over {attempts} attempts per question "
                        "(AA-LCR v1.1 equality checker)")
    result["judge"] = "gbench-gemini-cascade"
    result["judge_models"] = dict(used)   # answered judge calls per model
    result["judge_cascade"] = judge_cascade()
    result["judge_sampling"] = judge_sampling_record()
    reasons = [f"the equality checker is the gbench Gemini judge cascade "
               f"({', '.join(sorted(used)) or 'no judge call answered'}), not Artificial "
               f"Analysis's {OFFICIAL_JUDGE}"]
    if limit:
        reasons.append(f"--eval-limit {limit} runs a subset of the 100 questions")
    sampling = result.get("official_sampling") or {}
    if sampling.get("temperature_matches_official") is False:
        reasons.append(f"temperature {result.get('temperature')} (Artificial Analysis uses "
                       f"{sampling.get('temperature')} for this mode)")
    cut = record_budget_cutoffs(result)
    if cut:
        reasons.append(cut)
    traces = result.get("sample_traces") or []
    relaxed = sum(1 for t in traces if t.get("judge_verdict_relaxed_parse"))
    result["judge_verdicts_read_by_relaxed_parse"] = relaxed
    if relaxed:
        reasons.append(f"{relaxed} checker verdict(s) given as a bare CORRECT / INCORRECT rather "
                       "than the JSON object the prompt asks for, read by gbench's relaxed parse")
    harness: Dict[str, int] = {}
    for t in traces:
        if t.get("harness_error"):
            kind = t["harness_error"]["kind"]
            harness[kind] = harness.get(kind, 0) + 1
    failed_checks = sum(1 for t in traces if t.get("judge_grade") == "JUDGE_FAILED")
    if failed_checks:
        harness["judge_no_verdict"] = failed_checks
    result["harness_errors"] = dict(sorted(harness.items()))
    if harness:
        reasons.append("harness errors, not the model's outcome and left out of the accuracy: "
                       + ", ".join(f"{k} {v}" for k, v in sorted(harness.items()))
                       + " (endpoint_failure / endpoint_timeout / over_context: the request "
                       "produced no reply; judge_no_verdict: the checker gave no "
                       f"CORRECT/INCORRECT verdict in {_JUDGE_ATTEMPTS} replies)")
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    scored = [t for t in traces if not t.get("scoring_excluded")
              and t.get("judge_grade") != "JUDGE_OUTAGE"]
    if traces and not scored:
        # every attempt ended in a harness error or a judge outage: no number to report
        result["accuracy"] = None
        result["status"] = "failed"
        result["accuracy_withheld_reason"] = (
            "no attempt was scored: " + ", ".join(
                f"{k} {v}" for k, v in sorted({**harness, "judge_outage": sum(
                    1 for t in traces if t.get("judge_grade") == "JUDGE_OUTAGE")}.items()) if v))
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)
