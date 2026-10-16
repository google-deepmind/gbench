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

# Evaluation Suite: lab_bench
# Description: LAB-Bench (Language Agent Biology Benchmark), all 8 task categories

"""gbench native built-in runner for lab_bench (Reasoning & Knowledge).

LAB-Bench (Future-House, arXiv:2407.10362) has 8 task categories - LitQA2, DbQA, SuppQA, FigQA,
TableQA, ProtocolQA, SeqQA and CloningScenarios - and this suite runs all of them over the
public split (1,967 questions), as the repository's zero-shot baseline harness does
(`scripts/score_baseline.py` with `labbench.Evaluator` and the zero-shot agent, commit
998a8e0a):
  * questions: each category's `task.py` - ProtocolQA prepends the protocol text, SuppQA
    prepends "Paper title: <title>\\nDOI: <source>\\n", FigQA attaches the figure and TableQA
    the table images (after the text, as `openai_utils.get_completion` sends them);
  * choices: `utils.randomize_choices` - the ideal answer, the refusal choice REFUSE_CHOICE
    ("Insufficient information to answer the question") and the distractors, shuffled and
    shown as "(A) ...". Upstream shuffles with the unseeded global RNG; here the shuffle is
    seeded per item so a rerun asks the same questions;
  * prompt: `zero_shot.MCQ_INSTRUCT_TEMPLATE` verbatim with the CoT line score_baseline.py uses
    ("Think step by step.", chembench's COT_PROMPT), after chembench's post_process_prompts;
  * scoring: exactly upstream's - chembench `prepare_mcq_answer(reply, MCQ_REGEX_TEMPLATE_1)`
    (an [ANSWER]...[/ANSWER] tag, else a reply made only of letters) and then
    `run_regex(create_multiple_choice_regex(offered letters), ..., return_first=True)` on the
    reply as the server returned it. Any other reply has no answer and is wrong (where
    upstream's run_regex would raise on it, gbench records no answer instead);
  * metrics: accuracy, precision (correct / answered, where choosing the refusal option is not
    answering) and coverage per category and subtask, as `Evaluator.compute_metrics` reports
    them. The headline is the mean of the 8 category accuracies.

The Hugging Face mirror read here was built by upstream's own `update_datasets_in_hub.py` from
the evaluator's instances; its rows equal the repository's `*-public.jsonl` files at 998a8e0a
in content and order.

Sampling: score_baseline.py builds each agent with `model_kwargs={"model": <name>}` and the
agents pass only those to the API (labbench/openai_utils.py `chat.completions.create(messages=
[msg], **self.model_kwargs)`), so no temperature and no max_tokens are sent and the server's
defaults apply (for a vLLM server, the model's `generation_config.json` and the rest of the
context window) (base.OfficialSampling with `send_temperature=False` and
`send_max_tokens=False`; the result records `official_sampling`). No max_tokens is sent in either
mode. gbench always sends a temperature, by its rule (base.resolve_temperature): 1.0 with
`--thinking`, 0.0 without, recorded as not matching the harness, which sends none.
Under base's token-limit policy no request carries a max_tokens, `--max-output-tokens` is not
applied (`max_output_tokens_note`), and each prompt is still counted so one that cannot fit the
window is refused as over-context.
`--temperature`, or for this suite alone `GBENCH_LAB_BENCH_TEMPERATURE` (which takes precedence
over both), replaces it in both modes. LLM-judge grading is pinned at 0.0 and is not
affected by
either (this suite has no judge).

Sharding: `--shard I/N` selects shard I of the questions (round-robin over the question list,
which runs category by category in upstream's order) before the `--eval-limit` draw.
"""

import base64
import hashlib
import logging
import random
import re
import string
from typing import Any, Dict, List, Optional, Tuple
from .arc_agi import _for_base   # lays out an already-sharded sample list for run_eval_suite
from .base import OfficialSampling, apply_shard, run_eval_suite
from .sampling import stratified_sample
from .fc_common import base_suite_plan, run_attempts
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

PILLAR = "Reasoning & Knowledge"
DOCS_URL = "docs/evals/lab_bench.md"

_DATASET_ID = "futurehouse/lab-bench"
#: HF dataset revision the loader reads (the public split of all 8 categories, 1,967 questions).
_DATASET_REVISION = "5c77cec648430f30611808808861eb86f81d5eaa"
#: labbench.Eval, in upstream's order, with each category's public size.
CATEGORIES = (("TableQA", 244), ("ProtocolQA", 108), ("FigQA", 181), ("LitQA2", 199),
              ("SeqQA", 600), ("DbQA", 520), ("SuppQA", 82), ("CloningScenarios", 33))
PUBLIC_SIZE = sum(n for _, n in CATEGORIES)
#: scripts/score_baseline.py get_agent: model_kwargs={"model": args.model}; openai_utils.py
#: sends `messages` plus those kwargs only.
OFFICIAL_SAMPLING = OfficialSampling(
    reference=("Future-House/LAB-Bench@998a8e0a scripts/score_baseline.py "
               "(model_kwargs={'model': ...}) + labbench/openai_utils.py: no sampling fields "
               "and no max_tokens"),
    send_temperature=False, send_max_tokens=False)

# --- Future-House/LAB-Bench (commit 998a8e0a), verbatim -------------------------------------
ALPHABET = string.ascii_uppercase
REFUSE_CHOICE = "Insufficient information to answer the question"
MCQ_INSTRUCT_TEMPLATE = """The following is a multiple choice question about biology.
Please answer by responding with the letter of the correct answer.{cot}

Question: {question}

Options:
{answers}

You MUST include the letter of the correct answer within the following tags: [ANSWER] and [/ANSWER].
For example, '[ANSWER]<answer>[/ANSWER]', where <answer> is the correct letter.
Always answer in exactly this format of a single letter between the two tags, even if you are unsure.
We require this because we use automatic parsing."""

# --- chembench (the constants and helpers LAB-Bench's zero_shot.py imports), verbatim --------
COT_PROMPT = "Think step by step."
MCQ_REGEX_TEMPLATE_1 = r"(?:\[ANSWER\]|\[ANS\]|<ANS>|<ANSWER>|<ans>)\s*([A-Z](?:,\s*[A-Z])*)\s*(?:\..*?|,.*?)?(?:\[/?ANSWER\]|\[/ANS\]|</ANS>|</ANSWER>|</ans>)"
LATEX_ENV_REGEX = {
    "ce_pattern": r"\\ce\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
    "math_pattern": r"\$([^$]+)\$",
    "smiles_pattern": r"\[START_SMILES\](.*?)\[END_SMILES\]",
    "pu_pattern": r"\\pu\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
    "rxnsmiles_pattern": r"\[START_RXNSMILES\](.*?)\[END_RXNSMILES\]",
}


def _post_process_prompt(prompt: str) -> str:
    """chembench post_process_prompts with its defaults: strip \\ce{}, $..$, SMILES tags,
    reaction-SMILES tags and \\pu{} wrappers, keeping their contents."""
    for key in ("ce_pattern", "math_pattern", "smiles_pattern", "rxnsmiles_pattern", "pu_pattern"):
        prompt = re.sub(LATEX_ENV_REGEX[key], lambda x: x.group(1), prompt)
    return prompt


def prepare_mcq_answer(text: str, pattern: str = MCQ_REGEX_TEMPLATE_1,
                       alphabet: Optional[List[str]] = None) -> Optional[str]:
    """chembench prompter.prepare_mcq_answer as LAB-Bench's zero_shot.py calls it (no LLM
    extractor): the sorted set of tag matches, else a reply made only of letters, else ""
    (or None when the reply has neither "," nor " ")."""
    if alphabet is None:
        alphabet = [chr(i) for i in range(ord("A"), ord("Z") + 1)]
    try:
        matches = sorted(list(set(re.findall(pattern, text, re.DOTALL))))
    except IndexError:
        matches = None
    if matches:
        return str(matches)
    else:
        if text in alphabet:
            return text
        elif "," in text:
            split_text = text.split(",")
            all_letters = [letter for letter in split_text if letter.strip() in alphabet]
            if len(all_letters) == len(split_text):
                return ",".join(all_letters)
            else:
                return ""
        elif " " in text:
            split_text = text.split(" ")
            all_letters = [letter for letter in split_text if letter.strip() in alphabet]
            if len(all_letters) == len(split_text):
                return ",".join(all_letters)
            else:
                return ""


def run_regex(pattern: str, string: str, return_first: bool = True):
    """chembench utils.run_regex."""
    matches = re.findall(pattern, string, re.IGNORECASE)
    if not matches:
        return None

    if return_first:
        return matches[0]

    return matches


def create_multiple_choice_regex(allowed_letters: List[str]) -> str:
    """chembench utils.create_multiple_choice_regex."""
    pattern = "|".join(map(re.escape, allowed_letters))
    pattern = f"(?:{pattern})"

    return pattern


def _randomize_choices(seed_text: str, ideal: str,
                       distractors: List[str]) -> Tuple[List[str], str, str]:
    """LAB-Bench utils.randomize_choices: (shuffled "(X) text" choices, answer letter, refusal
    letter). Upstream shuffles with the global RNG; here the shuffle is seeded on the item so
    runs are reproducible. It is a real shuffle, not a sort: a sort would put the correct
    answer in a position set by its own text."""
    choices = [ideal, REFUSE_CHOICE, *distractors]
    if len(choices) > len(ALPHABET):
        raise ValueError("Too many choices")
    perm = list(range(len(choices)))
    random.Random(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()).shuffle(perm)
    shuffled = [f"({letter}) {choices[i]}" for letter, i in zip(ALPHABET, perm)]
    return shuffled, ALPHABET[perm.index(0)], ALPHABET[perm.index(1)]


def _format_prompt(question: str, choices: List[str]) -> str:
    """BaseZeroShotAgent.run_task (use_cot=True): the MCQ template, then post_process_prompts."""
    text = MCQ_INSTRUCT_TEMPLATE.format(question=question, cot="\n" + COT_PROMPT,
                                        answers="\n".join(choices))
    return _post_process_prompt(text)


def _build_mcq_prompt(protocol: str, question: str, ideal: str,
                      distractors: List[str]) -> Tuple[str, str]:
    """The ProtocolQA prompt for one item and its gold letter (ProtocolQA/task.py:
    `input.question = self.protocol + input.question`, no separator).

    The footer names the ANSWER FORMAT only; it never contains the gold letter."""
    choices, gold_letter, _ = _randomize_choices(protocol + "\n" + question, ideal, distractors)
    return _format_prompt(protocol + question, choices), gold_letter


def _task_question(category: str, item: Dict[str, Any]) -> str:
    """Each category's EvalInstance.get_input_output question."""
    question = str(item.get("question") or "")
    if category == "ProtocolQA":
        return str(item.get("protocol") or "") + question
    if category == "SuppQA":
        return f"Paper title: {item.get('paper-title')}\nDOI: {item.get('source')}\n{question}"
    return question


def _image_url(img: Any) -> str:
    """A figure or table as a data URL, encoded as upstream's openai_utils.get_completion sends
    it: labbench/utils.py encode_image re-saves the PIL image the task opened in its own format
    (`image.save(buf, format=image.format or "JPEG")`, PIL's defaults: a JPEG figure is
    re-encoded at quality 75, so its pixels differ from the file's) and labels it
    image/<format>."""
    raw = img.get("bytes") if isinstance(img, dict) else img
    if not raw:
        raise RuntimeError("lab_bench: a FigQA/TableQA row has no image bytes; refusing to ask "
                           "the question without its figure")
    import io
    from PIL import Image
    image = Image.open(io.BytesIO(bytes(raw)))
    fmt = image.format or "JPEG"
    with io.BytesIO() as buf:
        image.save(buf, format=fmt)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def _figures(category: str, item: Dict[str, Any]) -> List[Any]:
    if category == "FigQA":
        return [item.get("figure")]
    if category == "TableQA":
        return list(item.get("tables") or [])
    return []


def _load_category(category: str) -> List[Dict[str, Any]]:
    from datasets import Image, List as HFList, load_dataset
    ds = load_dataset(_DATASET_ID, category, split="train", revision=_DATASET_REVISION)
    # Keep the image bytes as stored instead of decoding to PIL.
    if category == "FigQA":
        ds = ds.cast_column("figure", Image(decode=False))
    elif category == "TableQA":
        ds = ds.cast_column("tables", HFList(Image(decode=False)))
    return [dict(r) for r in ds]


def _load_lab_bench_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, Any]]]:
    """All 8 LAB-Bench categories from HF Hub (futurehouse/lab-bench, pinned revision)."""
    rows = []
    try:
        for category, _ in CATEGORIES:
            for item in _load_category(category):
                rows.append((category, item))
    except Exception as e:
        logger.error(f"Failed to load dataset for lab_bench: {e}")
        raise RuntimeError(f"Could not load dataset for lab_bench: {e}") from e

    if not rows:
        raise RuntimeError("Dataset for lab_bench returned empty rows")

    # --shard first, over the full ordered question list, then the --eval-limit draw.
    rows, _ = apply_shard(rows, "lab_bench")
    # Stratified over the categories, not a contiguous head.
    rows = stratified_sample(rows, limit, lambda r: r[0], seed="lab_bench")

    samples = []
    for category, item in rows:
        question = _task_question(category, item)
        ideal = item.get("ideal")
        distractors = list(item.get("distractors") or [])
        if not item.get("question") or ideal is None or not distractors:
            raise RuntimeError(
                "lab_bench: unexpected dataset schema (missing question/ideal/distractors); "
                "refusing to fabricate sample data")
        choices, gold_letter, refuse_letter = _randomize_choices(
            (str(item.get("protocol")) + "\n" + str(item.get("question")))
            if category == "ProtocolQA" else f"{category}\n{item.get('id')}\n{question}",
            str(ideal), [str(d) for d in distractors])
        prompt = _format_prompt(question, choices)
        figures = _figures(category, item)
        content: Any = prompt
        if figures:
            content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": _image_url(f)}} for f in figures]
        gold = {"letter": gold_letter, "letters": ALPHABET[:len(choices)],
                "refuse": refuse_letter, "category": category,
                "subtask": str(item.get("subtask") or category)}
        samples.append(([{"role": "user", "content": content}], gold, {"category": category}))

    logger.info(f"Loaded {len(samples)} lab_bench samples over "
                f"{len({s[2]['category'] for s in samples})} categories.")
    return samples


def _reply_text(text: Optional[str]) -> Optional[str]:
    """The reply as the server returned it: <thought>/<reasoning> blocks, if the server left
    any in the content, are removed, and nothing else (upstream reads message.content raw)."""
    if text is None:
        return None
    cleaned = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"<reasoning>.*?</reasoning>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)


def _answer_letter(response_text: Optional[str], letters: str) -> Tuple[Optional[str], str]:
    """(letter, how it was read): zero_shot.run_task's prepare_mcq_answer + run_regex.

    "tag" is an [ANSWER] tag, "letters_only" a reply made only of letters, "none" no letter.
    Where upstream's run_regex would raise (prepare_mcq_answer returned None), there is no
    letter."""
    text = _reply_text(response_text)
    if text is None:
        return None, "none"
    prepared = prepare_mcq_answer(text, MCQ_REGEX_TEMPLATE_1)
    if prepared is None:
        return None, "none"
    answer = run_regex(create_multiple_choice_regex(list(letters)), prepared, return_first=True)
    if answer is None:
        return None, "none"
    tagged = bool(re.findall(MCQ_REGEX_TEMPLATE_1, text, re.DOTALL))
    return answer.upper(), ("tag" if tagged else "letters_only")


def _gold_parts(gold_target: Any) -> Tuple[str, str, Optional[str]]:
    """(gold letter, offered letters, refusal letter). A bare letter means A-G, no refusal."""
    if isinstance(gold_target, dict):
        return (str(gold_target["letter"]).upper(), str(gold_target["letters"]),
                gold_target.get("refuse"))
    return str(gold_target).strip().upper(), "ABCDEFG", None


def _eval_lab_bench(response_text: str, gold_target: Any) -> bool:
    """Evaluator.score_agent: `correct = agent_output == target_output`."""
    gold, letters, _ = _gold_parts(gold_target)
    letter, _ = _answer_letter(response_text, letters)
    return letter is not None and letter == gold


async def _async_score_lab_bench(sample_traces: List[Dict[str, Any]]) -> None:
    """Score each reply as returned by the server (base hands eval_fn a whitespace-trimmed
    copy, and upstream parses the raw text)."""
    for t in sample_traces:
        if t.get("response_text") is None:
            t["is_correct"] = False
            continue
        t["is_correct"] = _eval_lab_bench(t.get("response_text"), t.get("gold_answer"))
        if t.get("status") == "PENDING_JUDGE":
            t["status"] = "TRUNCATED" if t.get("health") in ("truncated", "looping") else "OK"


def _metrics(rows: List[Tuple[bool, bool]]) -> Dict[str, Any]:
    """Evaluator.compute_metrics over (correct, sure) pairs, in percent."""
    n = len(rows)
    correct = sum(1 for c, _ in rows if c)
    sure = sum(1 for _, s in rows if s)
    # None, never 0, when there is nothing to divide by (no question, no answered question)
    return {"accuracy": round(correct / n * 100.0, 2) if n else None,
            "precision": round(correct / sure * 100.0, 2) if sure else None,
            "coverage": round(sure / n * 100.0, 2) if n else None,
            "n_total": n, "n_correct": correct, "n_sure": sure}


def _lab_bench_report(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Evaluator.compute_metrics per category (metrics_all) and per subtask (metrics_<subset>).

    `sure` is upstream's `agent_output != unsure_choice`: a reply with no parseable letter
    still counts as answered (and wrong). Over the scored questions, as base's accuracy: a
    request that got no reply from the endpoint (failed, timed out, or a prompt the server
    refused as longer than its context window) is a harness error that base marks
    scoring_excluded, left out of every metric here. Also counted: replies read from an answer
    tag, replies made only of letters, replies with no letter, and refusal choices."""
    from .base import is_scored
    by_cat: Dict[str, List[Tuple[bool, bool]]] = {}
    by_sub: Dict[str, Dict[str, List[Tuple[bool, bool]]]] = {}
    counts = {"tagged_answers": 0, "letters_only_answers": 0, "no_letter": 0,
              "chose_refusal": 0}
    for t in traces:
        if not is_scored(t):
            continue
        gold = t.get("gold_answer") or {}
        _, letters, refuse = _gold_parts(gold)
        letter, source = _answer_letter(t.get("response_text"), letters)
        is_sure = letter != refuse
        cat = ((gold.get("category") if isinstance(gold, dict) else None) or t.get("category")
               or "unknown")
        sub = (gold.get("subtask") if isinstance(gold, dict) else None) or cat
        pair = (bool(t.get("is_correct")), is_sure)
        by_cat.setdefault(cat, []).append(pair)
        by_sub.setdefault(cat, {}).setdefault(sub, []).append(pair)
        counts["tagged_answers"] += source == "tag"
        counts["letters_only_answers"] += source == "letters_only"
        counts["no_letter"] += letter is None
        counts["chose_refusal"] += letter is not None and letter == refuse
    categories = {cat: {**_metrics(rows),
                        "subtasks": {s: _metrics(v) for s, v in sorted(by_sub[cat].items())}}
                  for cat, rows in by_cat.items()}
    order = [c for c, _ in CATEGORIES if c in categories] + sorted(
        c for c in categories if c not in dict(CATEGORIES))
    categories = {c: categories[c] for c in order}
    mean = (round(sum(v["accuracy"] for v in categories.values()) / len(categories), 2)
            if categories else None)
    return {"categories": categories, "mean_category_accuracy": mean,
            "all_questions": _metrics([p for rows in by_cat.values() for p in rows]), **counts}


#: An image must add at least this many prompt tokens to count as processed (the same
#: self-calibrating rule as the serving benchmark's multimodal preflight).
_MIN_IMAGE_PROMPT_TOKENS = 16


def _require_image_input(base_url: str, model_name: str, samples: List[Any]) -> None:
    """Stop with infra_required when FigQA/TableQA are selected and the endpoint does not
    process images: those questions cannot be asked without their figures."""
    image = next((part for s in samples if isinstance(s[0][0].get("content"), list)
                  for part in s[0][0]["content"] if part.get("type") == "image_url"), None)
    if image is None:
        return
    import json
    import urllib.request
    url = f"{base_url.rstrip('/')}/chat/completions"

    def prompt_tokens(content: List[Dict[str, Any]]) -> int:
        # A one-token probe that only reads the prompt-token count; no answer is scored.
        payload = dict(model=model_name, max_tokens=1,
                       messages=[{"role": "user", "content": content}],
                       chat_template_kwargs={"enable_thinking": False})
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            return int((json.loads(resp.read()).get("usage") or {}).get("prompt_tokens") or 0)

    text = [{"type": "text", "text": "Describe."}]
    try:
        added = prompt_tokens(text + [image]) - prompt_tokens(text)
    except Exception as e:
        raise infra_required(
            "lab_bench", f"FigQA and TableQA send figures and tables as images, and {url} did "
            f"not accept an image request ({e}). Serve the model with image input enabled.",
            DOCS_URL) from e
    if added < _MIN_IMAGE_PROMPT_TOKENS:
        raise infra_required(
            "lab_bench", f"FigQA and TableQA send figures and tables as images, but an image "
            f"added only {added} prompt tokens at {url}, so the endpoint is not processing "
            "images. Serve the model with image input enabled.", DOCS_URL)


def run_lab_bench(
    model_name: str,
    base_url: str,
    limit: Optional[int] = None,
    concurrency: int = 4,
    enable_thinking: bool = False,
    results_dir: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run LAB-Bench (all 8 categories) with the upstream zero-shot CoT baseline protocol."""
    samples = _load_lab_bench_samples(limit=limit)
    _require_image_input(base_url, model_name, samples)
    limit_plan = base_suite_plan("lab_bench", _for_base(samples), limit, run_attempts(kwargs))
    result = run_eval_suite(
        eval_name="lab_bench",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=_for_base(samples),  # already this shard's questions; base keeps exactly them
        async_eval_fn=_async_score_lab_bench,
        declared_scoring_mode="deterministic",
        supports_attempts=True,      # each trace is scored on its own, never keyed by id
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=None,          # already drawn by the loader, after the shard
        max_output_tokens=kwargs.get("max_output_tokens"),
        official_sampling=OFFICIAL_SAMPLING,
    )
    result["eval_limit_plan"] = limit_plan
    report = _lab_bench_report(result.get("sample_traces") or [])
    result["lab_bench_report"] = report
    result["accuracy"] = report["mean_category_accuracy"]
    traces = result.get("sample_traces") or []
    ran = {t.get("category") for t in traces}
    scored_cats = report.get("categories") or {}
    averaged = list(scored_cats)
    # A category the run asked but none of whose questions is in the score (every request a
    # harness error): the 8-category mean would silently drop it, so it is withheld.
    unscored = [c for c, _ in CATEGORIES if c in ran and c not in scored_cats]
    not_drawn = [c for c, _ in CATEGORIES if c not in ran]
    report["categories_unscored"] = unscored
    report["categories_not_run"] = not_drawn
    if unscored and result.get("accuracy") is not None:
        result["accuracy"] = None
        result["accuracy_withheld_reason"] = (
            f"{', '.join(unscored)} ran but has no question in the score (every request a "
            "harness error), so the mean over LAB-Bench's categories is withheld; the scored "
            "categories are in lab_bench_report")
    result["metric"] = (f"mean of the per-category accuracies over {len(averaged)} of LAB-Bench's "
                        f"{len(CATEGORIES)} categories"
                        + (f" ({', '.join(not_drawn)} not run)" if not_drawn else "")
                        + "; accuracy, precision (correct / answered, where the 'Insufficient "
                        "information' choice is not an answer) and coverage per category and "
                        "subtask are in lab_bench_report")
    reasons = ["public split only (1,967 questions; LAB-Bench holds back about 20% as a private "
               "test set, so published figures cover more questions)",
               "the headline is the mean of the 8 category accuracies; LAB-Bench publishes the "
               "categories separately (see lab_bench_report)"]
    if (result.get("official_sampling") or {}).get("temperature_matches_official") is False:
        reasons.append(f"temperature {result.get('temperature')} "
                       f"({result.get('temperature_source')}; the baseline harness sends none)")
    no_reply = sum(1 for t in (result.get("sample_traces") or []) if t.get("harness_error"))
    if no_reply:
        reasons.append(f"{no_reply} request(s) got no reply from the served endpoint (failed, "
                       "timed out, or a prompt over the context window, which upstream answers "
                       "as unsure: harness errors, left out of the score)")
    empty = int(result.get("empty_responses") or 0)
    if empty:
        reasons.append(f"{empty} request(s) came back empty and count as wrong")
    if limit:
        reasons.append(f"--eval-limit subset ({len(samples)} of {PUBLIC_SIZE:,} questions)")
    if not_drawn:
        reasons.append(f"the headline averages {len(averaged)} of the {len(CATEGORIES)} "
                       f"categories ({', '.join(not_drawn)} not run)")
    if unscored:
        reasons.append(f"categories with no question in the score (every request a harness "
                       f"error): {', '.join(unscored)}")
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    return result
