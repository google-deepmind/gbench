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

# Evaluation Suite: deepsearch_qa
# Description: DeepSearchQA (Google DeepMind google/deepsearchqa - multi-step web research, set answers)

"""gbench native built-in runner for deepsearch_qa (Agentic & Web Research).

Google DeepMind's DeepSearchQA (`google/deepsearchqa`, 900 prompts across 17 fields, 65%
with a set of answers), run the way the dataset card and the official Kaggle starter
notebook (andrewmingwang/deepsearchqa-starter-code, version 1) run it:

* **Prompt.** The raw `problem` is the only message: no system prompt, header or footer.
  `answer_type` is withheld from the model, as the card requires.
* **Web access.** The notebook hands the question to a model with Google Search. gbench
  offers its `web_search` tool and drives a tool loop whose settings follow that reference: no
  step limit and no injected "answer now" turn; no turn carries a max_tokens, as the notebook's
  model call sets none. See `DSQA_TOOL_LOOP`. The tool's results are written by Gemini: each call
  is one Gemini generateContent request grounded in Google Search with the model's query as the
  prompt, returning the cited sources' titles and URLs with segments of that Gemini answer, never
  page text; the tool description the model sees (gaia.GROUNDED_SEARCH_DESCRIPTION), the result
  (`search_backend_kind`) and `leaderboard_comparable_reason` say so. Every trace keeps its
  web_search calls (`tool_transcript`: query, answering Gemini backend, the reply the model read).
* **Grading.** The notebook's rater prompt (DEEPSEARCH_QA_PROMPT + GRADER_RATING_OUTPUT_EXAMPLE,
  copied verbatim below) returns per-expected-answer booleans plus a list of excessive
  answers; precision / recall / F1 and the Fully Correct / Fully Incorrect / Correct with
  Extraneous Answers categories are computed from them exactly as the notebook's
  `aggregate_ratings` does (technical report section 3.1).
* **Headline.** `accuracy` is the Fully Correct rate (the number the Kaggle leaderboard
  quotes); `f1_score`, the report's ranking metric, is alongside.

Deviations, all stated on the result in `leaderboard_comparable_reason`:

* The rater model is gbench's Gemini judge cascade rather than the card's `gemini-2.5-flash`;
  it is called as the notebook calls its rater (the prompt alone, no generation config, so the
  API's default sampling); `judge_models` records which models graded.
* The search tool is gbench's Gemini-grounded tool above, not the model's own Google Search,
  so the score partly reflects the Gemini model's research.
* An empty model response is scored Fully Incorrect (the report's set definition: nothing
  submitted, nothing found). The notebook instead drops such rows from every metric;
  `fully_correct_excluding_empty` reports that variant.

Harness errors are never the model's outcome: a sample whose model request produced no reply (an
endpoint error, a timeout, a prompt longer than the served window) or failed mid tool loop, or
that made a web_search call the Gemini search backend could not answer, is not sent to the rater:
it is marked HARNESS_ERROR, left out of every metric, counted in `harness_errors` and named in
`leaderboard_comparable_reason`, and the status is completed_with_errors. A run in which no sample
was rated withholds its score (accuracy null, status failed).

Sampling: the notebook calls the model with only the google_search tool in its config (no
temperature); gbench always sends one, its rule for every suite (base.resolve_temperature): 1.0
with `--thinking`, 0.0 without. `--temperature`, or `GBENCH_DEEPSEARCH_QA_TEMPERATURE` for this
suite alone (it takes precedence), sets another; the result records the notebook's setting under
`official_sampling`. The rater does not follow either. No max_tokens is sent either (the
token-limit policy's "not sent" case): each turn may use the room the context window leaves, in
both modes, `--max-output-tokens` is recorded as not applied in `max_output_tokens_note`, and a
turn cut off there is counted in `truncated_responses`.
"""

import csv
import json
import logging
import math
import os
import textwrap
from typing import Any, Dict, List, Optional, Tuple

from .base import (MODELS_NOT_SERVED_PREFIX, OfficialSampling, ToolLoopProtocol, apply_shard,
                   finalize_shard_fields, gemini_api_keys, gemini_key_live_valid,
                   judge_generate_cascade, run_eval_suite, strip_thinking_tags)
from .gaia import (GROUNDED_SEARCH_DESCRIPTION, GROUNDED_SEARCH_TOOL, SEARCH_BACKEND_KIND,
                   attach_tool_transcripts, harness_error_counts, harness_error_reason,
                   record_web_search, reset_tool_transcripts, web_research_harness_error)
from .sampling import stratified_sample, presharded_for_base
from .fc_common import base_suite_plan, run_attempts
from .search_tool import (backend_tally, reset_backend_tally, search_available,
                          search_backend_name, unavailable_reason)
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Agentic & Web Research"
DOCS_URL = "docs/evals/deepsearch_qa.md"

DATASET_REPO = "google/deepsearchqa"
#: Pinned dataset commit (DSQA-full.csv, 900 rows).
DATASET_REVISION = "b2623f8653065c2672de6d941fc5434cd652376c"
DATASET_FILE = "DSQA-full.csv"

#: The rater model the dataset card prescribes. gbench grades with its own cascade instead;
#: a run is only judge-matched when every grade came from this model.
CANONICAL_RATER_MODEL = "gemini-2.5-flash"

#: The notebook's model call passes GenerateContentConfig(tools=[google_search]) and nothing
#: else: no sampling field and no max_output_tokens, so the model's defaults apply and no turn
#: carries a max_tokens.
DSQA_SAMPLING = OfficialSampling(
    reference=("google/deepsearchqa Kaggle starter notebook (andrewmingwang/deepsearchqa-"
               "starter-code v1) generate_gemini_with_search: GenerateContentConfig with the "
               "google_search tool only (no temperature, no max_output_tokens)"),
    send_temperature=False,
    send_max_tokens=False,
)


def _rater_config() -> Any:
    """The notebook's rater call passes no config (the API's default sampling); None when
    google-genai is missing."""
    try:
        from google.genai import types
    except ImportError:
        return None
    return types.GenerateContentConfig()


#: The reference loop is one Gemini call with Google Search: the search loop runs inside the
#: model API with no harness step limit and no forced answer, and the reply is whatever the
#: model returns. So: no round cap, no recovery turns, and no max_tokens on any turn
#: (DSQA_SAMPLING), never the house turn ceiling.
DSQA_TOOL_LOOP = ToolLoopProtocol(
    reference=("google/deepsearchqa Kaggle starter notebook (andrewmingwang/"
               "deepsearchqa-starter-code v1): one generate_content call with the "
               "google_search tool; no harness step limit, no forced answer"),
    max_rounds=None,
    max_tool_calls=None,
    house_turn_ceiling=False,
    forced_final=False,
    rendered_calls_in_history=False,
)

# ---- Rater prompt: verbatim from the starter notebook, cell "Define DeepSearchQA Rater Prompt" ----
DEEPSEARCH_QA_PROMPT = textwrap.dedent("""\
Your task is to evaluate whether a given "AI Response" for a specific "User Prompt" arrived at the correct answer.

**Answer Correctness Task**

*   **Purpose:** Assess whether the AI response provides the correct answer(s) based on the provided "Correct Answer" and "Prompt Type".
*   **Process:**
    *   Identify the "Prompt Type": "<prompt_type>".
    *   Refer to the "Correct Answer": "<answer>".
    *   Based on the "Prompt Type", determine if the "AI Response" contains the expected answer(s).
        *   **'Single Answer'**: Check if the response provides the answer that addresses the user's question. It does not have to match the exact wording of the provided answer.
        *   **'Set Answer'**: Check if the response includes *each* item from the provided ground truth answers. The order might not matter unless specified otherwise. The response might include more answers than the list. Determine the correctness *only* based on the list first and then check if the response includes answers not in the list.
    *   **Explanation:** Provide a brief explanation justifying your assessment of answer correctness, referencing specific parts of the AI response and the correct answer.
    *   **Correctness Details:** Provide a dictionary, one key for each expected answer part, and value is a boolean indicating whether each expected answer part was found.
        *   For 'Set Answer', this will be a list of attributes, one for each item/part in the "Correct Answer". Each key will be a string indicating the expected answer part, and the value will be a boolean indicating whether that part was found in the response.
    *   **Excessive Answers:** Provide a list of strings, each indicating an excessive answer part. If the response provides answers that are **not** in the "Correct Answer" list, add these answers as excessive answers. Return an empty list when there's no excessive answers in the response.


**Output Format:**

Your evaluation *must* be structured as a nested JSON dictionary with the following top-level keys: `"Answer Correctness"`. Please return NULL if any of "Prompt", "AI Response" or "Correct Answer" is empty.
The value for `"Answer Correctness"` should be a dictionary containing `"Explanation"` (a string), `"Correctness Details"` (a dictionary where each key is the expected correct answer, and the value is a boolean indicating whether the response contains the correct answer), and `"Excessive Answers"` (a list of strings indicating the excessive answers).

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.


""")

GRADER_RATING_OUTPUT_EXAMPLE = r"""**Example (Partial):**

"```json
{{
  "Answer Correctness": {{
    "Explanation": "The response correctly identified Belgium and France but also includes an excessive answer, Italy.",
    "Correctness Details": {{
      "Belgium": true,
      "France": true,
    }},
    "Excessive Answers": [ "Italy" ]
  }}
}}
```"

**Now, proceed with the evaluation using the provided User Prompt, AI Response, and Correct Answer.**

User Prompt (Wrapped in <prompt> and </prompt>):
<prompt>
{prompt}
</prompt>
--------------------
**  Correct Answer (Wrapped in <answer> and </answer>):
Prompt Type: {prompt_type}
<answer>
{answer}
</answer>
--------------------
AI assistant response (Wrapped in <response> and </response>):
<response>
{response}
</response>

--------------------
Rating:"""


def grader_prompt(problem: str, answer_type: str, answer: str, response: str) -> str:
    """`_get_grader_model_input_for_row` from the starter notebook."""
    return DEEPSEARCH_QA_PROMPT + GRADER_RATING_OUTPUT_EXAMPLE.format(
        prompt=str(problem).strip(),
        prompt_type=str(answer_type).strip(),
        answer=str(answer).strip(),
        response=str(response).strip(),
    )


# ---- Rater-reply parsing: the notebook's helpers, same rules ----
def _parse_json_response(ori_json_response: str) -> Any:
    """JSON from the rater reply, unwrapping a ```json fence; None when unparseable."""
    try:
        json_str = ori_json_response.strip()
        start_marker = "```json"
        start_idx = json_str.find(start_marker)
        if start_idx != -1:
            json_str = json_str[start_idx + len(start_marker):].strip()
            end_idx = json_str.rfind("```")
            if end_idx != -1:
                json_str = json_str[:end_idx].strip()
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def _get_answer_correctness_details(json_response: Any) -> Optional[Dict[str, bool]]:
    try:
        details = json_response["Answer Correctness"]["Correctness Details"]
        if isinstance(details, dict):
            if (all(isinstance(k, str) for k in details.keys())
                    and all(isinstance(v, bool) for v in details.values())):
                return details
        return None
    except (KeyError, TypeError):
        return None


def _get_excessive_answers(json_response: Any) -> Optional[List[str]]:
    """[] when the key is absent (valid), None when present but malformed."""
    try:
        excessive = json_response["Answer Correctness"]["Excessive Answers"]
        if isinstance(excessive, list) and all(isinstance(i, str) for i in excessive):
            return excessive
        return None
    except KeyError:
        return []
    except TypeError:
        return None


def parse_rating(text: str) -> Tuple[Optional[Dict[str, bool]], Optional[List[str]], Optional[str]]:
    """(correctness_details, excessive_answers, error) per `_reduce_llm_response_to_item_rating`."""
    if not text:
        return None, None, "Auto-rater response was empty."
    parsed = _parse_json_response(text)
    if not parsed:
        return None, None, "Invalid JSON response from auto-rater."
    node = parsed.get("Answer Correctness") if isinstance(parsed, dict) else None
    if not isinstance(node, dict):
        return None, None, "Missing or malformed 'Answer Correctness' node."
    if not isinstance(node.get("Explanation"), str):
        return None, None, "Missing or malformed 'Explanation' in Answer Correctness."
    details = _get_answer_correctness_details(parsed)
    if details is None:
        return None, None, "Invalid 'Correctness Details' in Answer Correctness."
    excessive = _get_excessive_answers(parsed)
    if excessive is None:
        return None, None, "Invalid 'Excessive Answers' in Answer Correctness."
    return details, excessive, None


def _calculate_metric(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1_score": f1}


def score_item(details: Dict[str, bool], excessive: List[str]) -> Dict[str, Any]:
    """One rated item, exactly as the per-item body of the notebook's `aggregate_ratings`."""
    ratings = list(details.values())
    num_correct = sum(1 for r in ratings if r)
    has_expected = bool(ratings)
    all_expected_correct = has_expected and num_correct == len(ratings)
    has_excessive = bool(excessive)
    is_all_correct = (all_expected_correct or not has_expected) and not has_excessive
    fully_incorrect = has_expected and num_correct == 0
    correct_with_excessive = has_excessive and (all_expected_correct or not has_expected)
    if is_all_correct:
        category = "fully_correct"
    elif fully_incorrect:
        category = "fully_incorrect"
    elif correct_with_excessive:
        category = "correct_with_extraneous_answers"
    else:
        category = "partially_correct"
    out = {"is_all_correct": is_all_correct, "fully_incorrect": fully_incorrect,
           "correct_with_excessive": correct_with_excessive, "category": category,
           "expected_found": num_correct, "expected_total": len(ratings),
           "excessive_answers": list(excessive)}
    out.update(_calculate_metric(num_correct, len(excessive), len(ratings) - num_correct))
    return out


def _ci95(count: int, total: int) -> Optional[float]:
    """Half-width of the notebook's normal-approximation 95% interval, in percent."""
    if total <= 0:
        return None
    p = count / total
    return round(1.96 * math.sqrt(p * (1.0 - p) / total) * 100.0, 2)


def _load_deepsearch_qa_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """DSQA-full.csv at the pinned revision; raises on download or schema failure.

    Read with the csv module, not pandas: four set-answer rows have the literal answer
    "None" (no item qualifies), which pandas' default NA parsing turns into NaN.
    """
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=DATASET_REPO, filename=DATASET_FILE,
                               repo_type="dataset", revision=DATASET_REVISION)
        with open(path, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except Exception as e:
        logger.error(f"Failed to load dataset for deepsearch_qa: {e}")
        raise RuntimeError(f"Could not load dataset for deepsearch_qa: {e}") from e
    if not rows:
        raise RuntimeError("Dataset for deepsearch_qa returned empty rows")
    required = ("problem", "problem_category", "answer", "answer_type")
    if any(k not in rows[0] for k in required):
        raise RuntimeError(f"deepsearch_qa: unexpected dataset schema {list(rows[0])}; "
                           f"expected columns {list(required)}")

    # --shard first, over the full ordered set, then the --eval-limit draw within the shard.
    rows, _ = apply_shard(rows, "deepsearch_qa")
    # Stratified by field, not a contiguous head.
    rows = stratified_sample(rows, limit, lambda r: r.get("problem_category"),
                             seed="deepsearch_qa")
    samples = []
    for item in rows:
        problem = str(item.get("problem") or "")
        if not problem.strip():
            raise RuntimeError("deepsearch_qa: a row has an empty 'problem'; refusing to "
                               "fabricate sample data")
        messages = [{"role": "user", "content": problem}]
        # answer_type rides in the sample meta for the rater only; base forwards nothing but
        # API keys to the server, so the model never sees it.
        samples.append((messages, str(item.get("answer") or ""), {
            "category": str(item.get("problem_category") or "Unknown"),
            "answer_type": str(item.get("answer_type") or ""),
        }))
    logger.info(f"Loaded {len(samples)} deepsearch_qa samples ({DATASET_REPO}@{DATASET_REVISION[:12]}).")
    return samples


async def _async_judge_deepsearch_qa(
    sample_traces: List[Dict[str, Any]],
    judge_concurrency: int = 32,
) -> None:
    """Grade every trace with the starter notebook's rater prompt via the Gemini cascade.

    Outcomes per trace (each trace first gets its web_search calls, `tool_transcript`):
      * a harness error (gaia.web_research_harness_error: a model request that produced no reply
        or failed mid tool loop, or a web_search call the Gemini search backend could not answer):
        not sent to the rater; judge_grade HARNESS_ERROR and `scoring_excluded`, so it is left out
        of every metric, never scored as the model's outcome;
      * empty model response: not sent to the rater; Fully Incorrect, F1 0
        (`dsqa_empty_response`), and kept in the denominator;
      * rater cascade exhausted: JUDGE_OUTAGE (base excludes it from the denominator);
      * unparseable rater reply: judge_grade "unparsed" (a judge failure) and
        `scoring_excluded`, as the notebook drops invalid ratings from every metric.
    A graded trace keeps the rater's model (`judge_model`), its raw reply (`judge_raw`) and its
    explanation (`judge_explanation`), as the notebook's ItemRating keeps rating_response and
    answer_correctness_explanation.
    """
    import asyncio
    from tqdm import tqdm

    semaphore = asyncio.Semaphore(judge_concurrency)
    attach_tool_transcripts(sample_traces)

    async def _judge_single(trace: Dict[str, Any], pbar: tqdm) -> None:
        if trace.get("status") == "PENDING_JUDGE":
            trace["status"] = ("TRUNCATED" if trace.get("health") in ("truncated", "looping")
                               else "OK")
        err = web_research_harness_error(trace)
        if err:
            trace["harness_error"] = err
            trace["is_correct"] = False
            trace["judge_grade"] = HARNESS_ERROR_GRADE
            trace["scoring_excluded"] = True
            pbar.update(1)
            return
        response = strip_thinking_tags(trace.get("response_text") or "").strip()
        meta = trace.get("extra_payload") or {}
        if not response:
            trace["is_correct"] = False
            trace["dsqa_empty_response"] = True
            trace["dsqa"] = {"category": "fully_incorrect", "precision": 0.0, "recall": 0.0,
                             "f1_score": 0.0}
            trace["judge_grade"] = "no response (fully incorrect)"
            pbar.update(1)
            return
        msgs = trace.get("messages") or []
        problem = msgs[-1].get("content", "") if msgs else ""
        prompt = grader_prompt(problem, meta.get("answer_type", ""),
                               trace.get("gold_answer") or "", response)
        async with semaphore:
            text, judge_used = await judge_generate_cascade(prompt, config=_rater_config())
        if text is None:
            trace["judge_grade"] = "JUDGE_OUTAGE"
            pbar.update(1)
            return
        trace["judge_model"] = judge_used
        trace["judge_raw"] = text[:_JUDGE_RAW_CHARS]
        details, excessive, error = parse_rating(text)
        if error:
            trace["is_correct"] = False
            trace["judge_grade"] = "unparsed"
            trace["judge_error"] = error
            trace["scoring_excluded"] = True
            pbar.update(1)
            return
        try:
            trace["judge_explanation"] = _parse_json_response(text)["Answer Correctness"][
                "Explanation"]
        except (KeyError, TypeError):
            pass
        item = score_item(details, excessive)
        item["correctness_details"] = details
        trace["dsqa"] = item
        trace["is_correct"] = item["is_all_correct"]
        trace["judge_grade"] = (f"{item['category']} ({item['expected_found']}/"
                                f"{item['expected_total']} expected, "
                                f"{len(excessive)} excessive)")
        pbar.update(1)

    with tqdm(total=len(sample_traces), desc="Judging [DEEPSEARCH_QA]") as pbar:
        await asyncio.gather(*[_judge_single(t, pbar) for t in sample_traces])


#: The judge_grade of a trace that ended in a harness error (never sent to the rater).
HARNESS_ERROR_GRADE = "HARNESS_ERROR"
#: Characters of the rater's reply kept on a trace.
_JUDGE_RAW_CHARS = 4000


def summarize(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The notebook's project-level metrics over the rated traces (outages/invalid excluded)."""
    rated = [t for t in traces
             if t.get("judge_grade") != "JUDGE_OUTAGE" and not t.get("scoring_excluded")
             and isinstance(t.get("dsqa"), dict)]
    n = len(rated)
    out: Dict[str, Any] = {
        "num_rated": n,
        "num_empty_model_response": sum(1 for t in rated if t.get("dsqa_empty_response")),
        "num_invalid_auto_rater_response": sum(
            1 for t in traces if t.get("judge_grade") == "unparsed"),
        "num_judge_outage": sum(1 for t in traces if t.get("judge_grade") == "JUDGE_OUTAGE"),
        "num_harness_error": sum(1 for t in traces if t.get("harness_error")),
    }
    if not n:
        return out

    def pct(k: int, total: int) -> float:
        return round(100.0 * k / total, 2) if total else 0.0

    cats = [t["dsqa"].get("category") for t in rated]
    fc = cats.count("fully_correct")
    out.update({
        "fully_correct": pct(fc, n),
        "fully_correct_ci95": _ci95(fc, n),
        "fully_incorrect": pct(cats.count("fully_incorrect"), n),
        "correct_with_extraneous_answers": pct(cats.count("correct_with_extraneous_answers"), n),
        "partially_correct": pct(cats.count("partially_correct"), n),
        "precision": round(100.0 * sum(t["dsqa"]["precision"] for t in rated) / n, 2),
        "recall": round(100.0 * sum(t["dsqa"]["recall"] for t in rated) / n, 2),
        "f1_score": round(100.0 * sum(t["dsqa"]["f1_score"] for t in rated) / n, 2),
    })
    # The starter notebook's own aggregation drops empty model responses from every metric.
    kept = [t for t in rated if not t.get("dsqa_empty_response")]
    if kept:
        out["fully_correct_excluding_empty"] = pct(
            sum(1 for t in kept if t["dsqa"].get("category") == "fully_correct"), len(kept))
        out["f1_score_excluding_empty"] = round(
            100.0 * sum(t["dsqa"]["f1_score"] for t in kept) / len(kept), 2)
    by_type: Dict[str, List[bool]] = {}
    for t in rated:
        at = (t.get("extra_payload") or {}).get("answer_type") or "Unknown"
        by_type.setdefault(at, []).append(t["dsqa"].get("category") == "fully_correct")
    out["fully_correct_by_answer_type"] = {k: {"n": len(v), "fully_correct": pct(sum(v), len(v))}
                                           for k, v in sorted(by_type.items())}
    return out


def run_deepsearch_qa(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    results_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run DeepSearchQA: raw prompt + web_search tool loop, graded by the notebook's rater prompt."""
    # A web-research benchmark: the search backend and the rater both need a Gemini key, so
    # without it the suite cannot run canonically - hard-error before any download.
    if not search_available():
        raise infra_required("deepsearch_qa", unavailable_reason(), DOCS_URL)
    # Live check so a present-but-invalid key, or a cascade model the API does not serve, fails
    # here, not after every search and every rating has failed.
    _ok, _why = gemini_key_live_valid(",".join(gemini_api_keys()))
    if not _ok:
        if str(_why).startswith(MODELS_NOT_SERVED_PREFIX):
            raise infra_required(
                "deepsearch_qa", f"{_why}; every model of the Gemini cascade must be one the API "
                "serves for Google-Search grounding and the rater.", DOCS_URL)
        raise infra_required(
            "deepsearch_qa",
            f"the Gemini key (GEMINI_API_KEYS / GEMINI_API_KEY) was rejected by the search/judge "
            f"endpoint ({_why}); a valid key is required for Google-Search grounding and the "
            "rater.", DOCS_URL)
    reset_backend_tally()   # per-run, not per-process (a sweep runs many search suites)
    reset_tool_transcripts()
    samples = _load_deepsearch_qa_samples(limit=limit)
    extra_payload = dict(kwargs.get("extra_payload") or {})
    extra_payload.setdefault("tools", [GROUNDED_SEARCH_TOOL])

    limit_plan = base_suite_plan("deepsearch_qa", presharded_for_base(samples), limit,
                                 run_attempts(kwargs))
    result = run_eval_suite(
        eval_name="deepsearch_qa",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's problems, kept as is
        async_eval_fn=_async_judge_deepsearch_qa,
        thinking=enable_thinking,
        extra_payload=extra_payload,
        limit=limit,
        max_output_tokens=kwargs.get("max_output_tokens"),
        tool_executor=record_web_search,
        tool_loop=DSQA_TOOL_LOOP,
        supports_attempts=True,     # the rater grades each trace on its own
        official_sampling=DSQA_SAMPLING,
    )
    result["eval_limit_plan"] = limit_plan
    traces = result.get("sample_traces", [])
    summary = summarize(traces)
    result.update(summary)
    result["metric"] = ("fully_correct (DeepSearchQA Fully Correct rate; f1_score alongside, "
                        "technical report section 3.1)")
    result["dataset"] = f"{DATASET_REPO}@{DATASET_REVISION}"
    judges: Dict[str, int] = {}
    for t in traces:
        if t.get("judge_model"):
            judges[t["judge_model"]] = judges.get(t["judge_model"], 0) + 1
    result["judge_models"] = judges
    result["search_backend"] = search_backend_name()
    result["search_backend_kind"] = SEARCH_BACKEND_KIND
    result["search_backend_calls"] = backend_tally()
    result["search_tool_description"] = GROUNDED_SEARCH_DESCRIPTION
    harness = harness_error_counts(traces)
    result["harness_errors"] = harness
    reasons = []
    if set(judges) != {CANONICAL_RATER_MODEL}:
        reasons.append(f"rater is the gbench Gemini judge cascade "
                       f"({', '.join(sorted(judges)) or 'no grades'}), not "
                       f"{CANONICAL_RATER_MODEL}, which the dataset card prescribes")
    official = result.get("official_sampling") or {}
    if official.get("temperature_matches_official") is False:
        reasons.append(f"temperature {result.get('temperature')} set by "
                       f"{str(result.get('temperature_source') or '').split(':', 1)[-1]} (the "
                       "notebook sends none)")
    reasons.append("web access is gbench's web_search tool, whose results are Gemini-written: "
                   "each call is a Gemini model grounded in Google Search answering the model's "
                   "query, returning the cited sources' titles and URLs with segments of that "
                   "Gemini answer (no page text, no page fetch), where the notebook's model runs "
                   "Google Search itself; so the score partly reflects the Gemini model's "
                   "research")
    if summary.get("num_empty_model_response"):
        reasons.append(f"{summary['num_empty_model_response']} empty response(s) scored fully "
                       f"incorrect; the starter notebook drops them "
                       f"(fully_correct_excluding_empty)")
    if harness:
        reasons.append(harness_error_reason(harness))
    if limit:
        reasons.append(f"--eval-limit subset ({len(samples)} of the 900 prompts)")
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    if harness:
        why = harness_error_reason(harness)
        if result.get("status") == "success":
            result["status"] = "completed_with_errors"
        result["status_reason"] = "; ".join(x for x in (result.get("status_reason"), why) if x)
    if traces and not summary.get("num_rated"):
        # nothing was rated: every sample ended in a harness error, a judge outage or an
        # unreadable rating, so there is no number to report
        result["accuracy"] = None
        result["status"] = "failed"
        result["accuracy_withheld_reason"] = (
            "no sample was rated: " + ", ".join(
                f"{k} {summary.get(k, 0)}" for k in ("num_harness_error", "num_judge_outage",
                                                     "num_invalid_auto_rater_response")))
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)
