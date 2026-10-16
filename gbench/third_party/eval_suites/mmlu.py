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

"""Native MMLU evaluation suite (57 subjects, 4 choices A-D): the original Hendrycks protocol.

MMLU is scored the way hendrycks/test `evaluate.py` (commit 4450500f) scores it: 5-shot, with
the subject header, then the first five dev-split exemplars and the test question rendered as
`{question}\\nA. ..\\nB. ..\\nC. ..\\nD. ..\\nAnswer:` (exemplars followed by ` X\\n\\n`); the
log-probability of each answer letter " A" .. " D" as the next token is read, the highest one
is the prediction (`np.argmax`, the first on a tie), and the headline is the accuracy pooled
over all 14,042 test questions (the reference's "Average accuracy").

The request is evaluate.py's: one `/v1/completions` call per question with the prompt as text,
`max_tokens=1`, `logprobs=100` and `temperature=0`. Each letter's log-probability is read from
the top-100 list of the one generated token (`top_logprobs[-1][" A"]` ..), and a letter outside
that list counts -100, as in the reference. evaluate.py also sets `echo=True`, which only adds
the prompt tokens' entries ahead of the generated token's; `top_logprobs[-1]` is the same list
either way. gbench leaves `echo` out: on vLLM, `echo` with `logprobs` also computes the top-100
log-probabilities at every prompt position (over the full vocabulary), which none of the scoring
reads and which ran a default vLLM server out of GPU memory. So nothing is scored from the
prompt, and nothing is tokenized for the request: the server tokenizes the text, as the
reference's did.

vLLM rejects a request for more than 20 log-probabilities per token unless the server is
started with `--max-logprobs 100` (or more). The suite sends one question first; a server that
rejects the request, or returns no top-log-probability list, stops the suite with an
infra_required error naming that flag. A reply whose list holds fewer than the 100 requested
entries while a letter is missing from it is counted (`short_top_logprobs`) and makes the run
non-comparable, since that letter's -100 may come from the server's shorter list rather than
the model.

evaluate.py reads the letters by the token text " A" .. " D", so a server that keys the list
differently would score every question A (four -100s, np.argmax's first index). That is the
server's format, not the model's answer, and is never reported as a score. A reply that holds
none of " A" .. " D" and whose keys show another format fails as a harness error: every key a
`token_id:<id>` placeholder (a vLLM server started with `--return-tokens-as-token-ids`), or
all four letters present without the leading space (vLLM before v0.26.0, PR #48674, keys tokens
that way for tokenizers whose one-token decode drops the space; Gemma 4's are keyed " A" on
either). On the first question that stops the suite with infra_required; on a later one the
question is a harness error, like a request that still fails after its retries: left out of
`accuracy` and `category_accuracy` (`harness_error` and `scoring_excluded` on its trace,
`scoring_excluded` and `request_harness_errors` in the result), counted in `failed_requests`,
named in `status_reason` (status `completed_with_errors`) and not comparable; with no question
scored, `accuracy` is null with `accuracy_withheld_reason`. And when no scored reply of the run
lists any of " A" .. " D", whatever its keys, the suite stops with infra_required instead of
reporting the all-A accuracy.

The reference drops trailing exemplars while a prompt does not fit its model's context
(`crop`, sized for GPT-3's 2,048-token window). gbench applies that rule to the served model's
window when the server reports it (the prompt plus the one generated token must fit), counting
tokens with the served model's tokenizer and its own defaults (see wmdp.py, which shares the
encoder), and counts the questions it shortened (`shots_reduced`). BOS is present exactly when
the tokenizer adds it by default (Gemma 4's does not); the result records `bos_prepended`.

Sharding (`--shard I/N`) selects shard I of the full question list before the `--eval-limit`
draw, as base.run_eval_suite does, and the result records `shard`. Each trace records the
question's `sample_key` and its position in the pinned split (`global_idx`), so rows with
identical content stay apart; `gbench --merge` combines the shards through merge_shards, which
states the non-comparable reasons and the status again for the union, so the merged result of a
full run's shards equals the unsharded result.

Only the one token evaluate.py asks for is generated, and it is not scored, so
`--max-output-tokens` does not apply, and `--thinking` cannot be applied: a thinking request is
logged as a WARNING and recorded as `thinking: False, thinking_requested: True,
thinking_applied: False`. A server that cannot return the top log-probabilities is an
infrastructure error, not a reason to fall back to a generative proxy.

Sampling: the request carries gbench's no-think temperature 0.0 (base.resolve_temperature; the
request is never a thinking one, see above), the same value as evaluate.py's `temperature=0`;
`--temperature` or `GBENCH_MMLU_TEMPERATURE` replaces it. The letters are read from the
log-probabilities, not from the sampled token, so the temperature does not change the score. LLM-judge grading is pinned at 0.0 and
is not affected by either (this suite has no judge).
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from .limit_plan import eval_limit_plan, merge_plans
from .base import (apply_shard, finalize_shard_fields, get_run_knob, record_shard,
                   resolve_temperature, sample_key)
from .sampling import limit_dataset
from .swebench_common import infra_required
from .wmdp import _Encoder, _post, _shard_label

logger = logging.getLogger(__name__)

DATASET = "cais/mmlu"
DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
DOCS_URL = "docs/evals/mmlu.md"

OPTION_LETTERS = ["A", "B", "C", "D"]

#: The dev split holds exactly 5 exemplars per subject; evaluate.py's default `--ntrain 5`.
MAX_N_SHOT = 5
#: The full test split.
TEST_SIZE = 14042

#: evaluate.py's request: `openai.Completion.create(engine=engine, prompt=prompt, max_tokens=1,
#: logprobs=100, temperature=0, echo=True)`; only `top_logprobs[-1]` of it is read.
TOP_LOGPROBS = 100
#: evaluate.py's log-probability for a letter that is not in that top list.
MISSING_LETTER_LOGPROB = -100.0
#: The key vLLM gives a token when it returns tokens as ids (format_token_id_placeholder).
TOKEN_ID_KEY_PREFIX = "token_id:"
#: evaluate.py's temperature, recorded for reference. The request carries gbench's resolved
#: temperature (0.0 unless the operator overrides it); the letters are read from the
#: log-probabilities, not from the sampled token, so it does not change the score.
REQUEST_TEMPERATURE = 0

_REQUEST_TIMEOUT_S = 300
_RETRIES = 3
#: Keys of a reply without a letter quoted in the error when no reply of the run lists one.
_KEYS_SHOWN = 8


def _gold_letter(ans: Any, choices: Any) -> str:
    if isinstance(ans, str) and ans.strip().upper() in OPTION_LETTERS:
        return ans.strip().upper()
    gold_idx = int(ans)
    if not 0 <= gold_idx < len(OPTION_LETTERS):
        # A bad key is a data error; keying it as "A" would fabricate a gold answer.
        raise ValueError(f"mmlu: answer index {ans!r} is outside A-D")
    return OPTION_LETTERS[gold_idx]


# hendrycks/test evaluate.py, verbatim; format_example below reads a dataset row instead of
# a DataFrame row. format_subject's leading space gives the reference's "about  <subject>."
# double space.
def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(question, choices, answer=None):
    prompt = question
    for j in range(len(choices)):
        prompt += "\n{}. {}".format(OPTION_LETTERS[j], choices[j])
    prompt += "\nAnswer:"
    if answer is not None:
        prompt += " {}\n\n".format(answer)
    return prompt


def gen_prompt_header(subject):
    return "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject))


def _dev_exemplars(n_shot: int) -> Dict[str, List[str]]:
    """The first `n_shot` dev-split exemplars of each subject, each in the Hendrycks format.

    A failed load is a hard error: running 0-shot while the result records `n_shot` would
    label the result with a protocol it did not run.
    """
    if n_shot <= 0:
        return {}
    if n_shot > MAX_N_SHOT:
        raise ValueError(f"mmlu: --eval-n-shot {n_shot} exceeds the {MAX_N_SHOT} dev-split "
                         "exemplars per subject")
    from datasets import load_dataset
    try:
        dev = list(load_dataset(DATASET, "all", split="dev", revision=DATASET_REVISION))
    except Exception as e:
        raise infra_required(
            "mmlu", f"the {n_shot}-shot exemplars could not be loaded from {DATASET} (dev split, "
            f"revision {DATASET_REVISION[:8]}): {e}. Check Hugging Face Hub access.",
            DOCS_URL) from e
    by_subject: Dict[str, List[str]] = {}
    for item in dev:
        shots = by_subject.setdefault(str(item.get("subject", "")), [])
        if len(shots) < n_shot:
            choices = item.get("choices", [])
            shots.append(format_example(item.get("question", ""), choices,
                                        _gold_letter(item.get("answer", 0), choices)))
    short = sorted(s for s, shots in by_subject.items() if len(shots) < n_shot)
    if short:
        raise infra_required(
            "mmlu", f"the dev split has fewer than {n_shot} exemplars for {short}.", DOCS_URL)
    return by_subject


def _few_shot_exemplars(n_shot: int) -> Dict[str, str]:
    """The joined exemplar block of each subject (evaluate.py `gen_prompt` without header)."""
    return {s: "".join(shots) for s, shots in _dev_exemplars(n_shot).items()}


def _load_mmlu_items(limit: Optional[int] = None, n_shot: int = MAX_N_SHOT) -> List[Dict[str, Any]]:
    """MMLU test questions (`cais/mmlu`, 'all', pinned revision) with their Hendrycks parts.

    Each item holds the subject, the header, the exemplar list, the test question block
    (`format_example` without the answer) and the gold index.
    """
    from datasets import load_dataset
    ds = load_dataset(DATASET, "all", split="test", revision=DATASET_REVISION)
    # --shard first, over the full ordered set, then the --eval-limit draw (base's order). Each
    # row keeps its position in the pinned split: rows with identical content (the test split
    # has some) stay distinct samples across shards (sample_key).
    rows, _ = apply_shard([dict(row, _row_index=i) for i, row in enumerate(ds)], "mmlu")
    # Stratified, not a contiguous head: the rows are stored grouped by
    # `subject`, so `[:limit]` returned a single category.
    raw_samples = list(limit_dataset(rows, limit, "subject", seed="mmlu"))
    logger.info(f"Loaded {len(raw_samples)} MMLU samples from HF Hub ('{DATASET}').")

    shots_by_subject = _dev_exemplars(n_shot)
    if shots_by_subject:
        logger.info("[mmlu] prepending %d-shot exemplars for %d subjects",
                    n_shot, len(shots_by_subject))
    items = []
    for item in raw_samples:
        choices = item.get("choices", [])
        subject = str(item.get("subject", "general")).strip()
        if len(choices) != len(OPTION_LETTERS):
            raise RuntimeError("mmlu: unexpected dataset schema (need 4 choices); refusing to "
                               "fabricate sample data")
        gold = OPTION_LETTERS.index(_gold_letter(item.get("answer", 0), choices))
        if n_shot > 0 and subject not in shots_by_subject:
            raise infra_required(
                "mmlu", f"no dev-split exemplars for test subject '{subject}'.", DOCS_URL)
        items.append({"subject": subject, "header": gen_prompt_header(subject),
                      "shots": list(shots_by_subject.get(subject, [])),
                      "question": format_example(str(item.get("question", "")), choices),
                      "gold": gold, "row_index": item.get("_row_index")})
    return items


def _prompt(item: Dict[str, Any], k: Optional[int] = None) -> str:
    """evaluate.py `gen_prompt(dev_df, subject, k) + format_example(test_df, i, False)`."""
    shots = item["shots"] if k is None else item["shots"][:k]
    return item["header"] + "".join(shots) + item["question"]


def _load_mmlu_samples(
    enable_thinking: bool = False,
    limit: Optional[int] = None,
    n_shot: int = 0,
) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """The Hendrycks prompt of each test question as (messages, gold letter, meta).

    `enable_thinking` does not change the prompt; it is accepted for loader-signature parity.
    """
    return [([{"role": "user", "content": _prompt(it)}], OPTION_LETTERS[it["gold"]],
             {"category": it["subject"]})
            for it in _load_mmlu_items(limit=limit, n_shot=n_shot)]


def _fit_to_window(item: Dict[str, Any], enc: _Encoder, window: Optional[int]) -> Tuple[str, int]:
    """The prompt and its shot count after evaluate.py's crop rule, applied to `window`.

    evaluate.py: `while crop(prompt) != prompt: k -= 1`, i.e. drop the LAST exemplar until the
    prompt fits the model's context (crop.py keeps the first 2,048 GPT-2 tokens, in a
    2,049-token API context). Here the context is the served model's window, and the request
    must hold the prompt and the one token it generates. Without a known window every exemplar
    is kept."""
    k = len(item["shots"])
    prompt = _prompt(item, k)
    if not window:
        return prompt, k
    while k > 0 and len(enc.encode(prompt)) + 1 > window:
        k -= 1
        prompt = _prompt(item, k)
    return prompt, k


def _pick(lls: List[float]) -> int:
    """np.argmax over the letter log-probabilities: the first maximum wins."""
    return max(range(len(lls)), key=lambda i: lls[i])


def _payload(model_name: str, prompt: str, temperature: float = 0.0) -> Dict[str, Any]:
    """evaluate.py's request, without `echo` (see the module docstring): the text prompt, one
    generated token and the top-100 log-probabilities of that token."""
    return dict(model=model_name, prompt=prompt, max_tokens=1, logprobs=TOP_LOGPROBS,
                temperature=temperature)


def _generated_top(choice: Dict[str, Any]) -> Dict[str, Any]:
    """The top-log-probability list of the generated token (`top_logprobs[-1]`)."""
    tops = (choice.get("logprobs") or {}).get("top_logprobs")
    top = tops[-1] if isinstance(tops, list) and tops else None
    if not isinstance(top, dict) or not top:
        raise ValueError("the response carries no top_logprobs for the generated token")
    return top


def _foreign_key_format(top: Dict[str, Any]) -> Optional[str]:
    """How a top list that holds none of " A" .. " D" shows it is keyed otherwise than by the
    token text evaluate.py reads, or None when its keys do not show that."""
    keys = list(top)
    if all(isinstance(k, str) and k.startswith(TOKEN_ID_KEY_PREFIX)
           and k[len(TOKEN_ID_KEY_PREFIX):].isdigit() for k in keys):
        return ("every key is a 'token_id:<id>' placeholder, not token text (a vLLM server "
                "started with --return-tokens-as-token-ids)")
    if all(letter in top for letter in OPTION_LETTERS):
        return ("it lists 'A' .. 'D' without the leading space, as vLLM before v0.26.0 (PR "
                "#48674) keys tokens for tokenizers whose one-token decode drops that space")
    return None


def _letter_logprobs(choice: Dict[str, Any]) -> Tuple[List[float], List[str], int]:
    """evaluate.py's read: `top_logprobs[-1][" A"]` .. `[" D"]`, -100 for a letter not listed.

    Returns the four log-probabilities, the letters that were not listed and the size of the
    list. A reply without the list is not the model's answer but a server that did not return
    what was asked for, so it raises (ValueError) instead of scoring four -100s; so does a list
    that holds none of the four letters and whose keys show another format
    (_foreign_key_format)."""
    top = _generated_top(choice)
    values, missing = [], []
    for letter in OPTION_LETTERS:
        value = top.get(" " + letter)
        if value is None:
            missing.append(letter)
            values.append(MISSING_LETTER_LOGPROB)
        else:
            values.append(float(value))
    if len(missing) == len(OPTION_LETTERS):
        foreign = _foreign_key_format(top)
        if foreign:
            raise ValueError("the generated token's top_logprobs list is not keyed by token "
                             f"text as evaluate.py reads it (' A' .. ' D'): {foreign}")
    return values, missing, len(top)


async def _score_sample(session: Any, url: str, model_name: str,
                        sample: Dict[str, Any], semaphore: Any,
                        temperature: float = 0.0) -> Dict[str, Any]:
    failed = {"loglikelihoods": None, "letters_missing": None, "top_logprobs_returned": None,
              "pred": None, "is_correct": False, "status": "FAILED"}
    last_err = None
    for attempt in range(_RETRIES):
        try:
            async with semaphore:
                data = await _post(session, url, _payload(model_name, sample["context"],
                                                         temperature))
            choices = data.get("choices") or []
            if len(choices) != 1:
                raise ValueError(f"expected 1 choice, got {len(choices)}")
            lps, missing, returned = _letter_logprobs(choices[0])
            pred = _pick(lps)
            res = {"loglikelihoods": [round(x, 6) for x in lps], "letters_missing": missing,
                   "top_logprobs_returned": returned, "pred": pred,
                   "is_correct": pred == sample["gold"], "status": "OK", "error": None}
            if len(missing) == len(OPTION_LETTERS):
                # Kept for the message should no reply of the run list a letter.
                res["top_keys_sample"] = list(_generated_top(choices[0]))[:_KEYS_SHOWN]
            return res
        except ValueError as e:        # a well-formed reply without the data: do not retry
            return {**failed, "error": str(e)}
        except Exception as e:         # transport / server error: retry with backoff
            last_err = e
            await asyncio.sleep(2 ** attempt)
    return {**failed, "error": f"{type(last_err).__name__}: {last_err}"}


async def _score_all(base_url: str, model_name: str, enc: _Encoder,
                     samples: List[Dict[str, Any]], concurrency: int,
                     temperature: float = 0.0) -> List[Dict[str, Any]]:
    """Score every sample. `enc` sizes the prompts (_fit_to_window) before this is called; the
    requests themselves carry text, which the server tokenizes."""
    import aiohttp
    url = f"{base_url.rstrip('/')}/completions"
    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_S)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Probe with one question first: a server that rejects logprobs=100 fails every
        # request the same way, and should stop the suite with one clear error.
        first = await _score_sample(session, url, model_name, samples[0], semaphore,
                                    temperature)
        if first["status"] != "OK":
            raise infra_required(
                "mmlu", f"MMLU reads the top-{TOP_LOGPROBS} log-probabilities of the token "
                f"that follows the prompt, as hendrycks/test evaluate.py requests them "
                f"(logprobs={TOP_LOGPROBS}, max_tokens=1), and needs the endpoint's "
                f"/v1/completions to return them keyed by token text (' A' .. ' D'). A vLLM "
                f"server must be started with --max-logprobs {TOP_LOGPROBS} (its default "
                f"allows 20) and without --return-tokens-as-token-ids. The first request to "
                f"{url} failed: {first['error']}", DOCS_URL)
        rest = await asyncio.gather(*[
            _score_sample(session, url, model_name, s, semaphore, temperature)
            for s in samples[1:]])
    return [first, *rest]


def _load_tokenizer(name: str) -> Any:
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(name)
    except Exception as e:
        raise infra_required(
            "mmlu", f"log-likelihood scoring tokenizes each prompt with the served model's "
            f"tokenizer, and {name!r} could not be loaded ({e}). Pass --tokenizer <HF id of the "
            f"served model> (and make sure `transformers` is installed).", DOCS_URL)


def _noncanon_reasons(n_shot: int, limit: Optional[int], total: int, failed: int,
                      short_lists: int) -> List[str]:
    """Why a run (or the union of a sharded run's shards, merge_shards) is not comparable with
    the reference: the shot count, an --eval-limit subset, questions without a usable reply,
    letters missing from a short top-log-probability list."""
    noncanon = []
    if n_shot != MAX_N_SHOT:
        noncanon.append(f"{n_shot}-shot (the reference is 5-shot)")
    if limit:
        noncanon.append(f"--eval-limit subset ({total} of {TEST_SIZE:,} questions)")
    if failed:
        noncanon.append(f"{failed} question(s) got no usable reply from the served endpoint "
                        "(harness errors, left out of the score)")
    if short_lists:
        noncanon.append(f"{short_lists} question(s) lacked a letter in a top-log-probability list "
                        f"shorter than the {TOP_LOGPROBS} requested, so their -100 may come from "
                        "the server's list rather than the model")
    return noncanon


def merge_shards(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The merged result of a sharded run (`gbench --merge`): the generic merge (the union of
    the shards' questions, the accuracy, per-subject counts and every count re-counted or
    summed), with the non-comparable reasons and the status stated again for that union by the
    run's own rules (_noncanon_reasons, own_loop_status), so the counts they quote are the
    full set's. A reason no shard's own questions produce (one the eval runner stamped) is
    kept."""
    from .vqa_common import own_loop_status
    from ...merge import generic_merge, other_reasons
    merged = generic_merge("mmlu", results)
    merged_plan = merge_plans(results)
    if merged_plan is not None:
        merged["eval_limit_plan"] = merged_plan
    live = [r for r in results if not r.get("empty_shard")] or list(results)
    n_shot = int(live[0].get("n_shot") if live[0].get("n_shot") is not None else MAX_N_SHOT)
    limit = live[0].get("eval_limit")
    reasons = _noncanon_reasons(n_shot, limit, int(merged.get("total_questions") or 0),
                                int(merged.get("failed_requests") or 0),
                                int(merged.get("short_top_logprobs") or 0))
    for r in live:
        own = _noncanon_reasons(n_shot, limit, int(r.get("total_questions") or 0),
                                int(r.get("failed_requests") or 0),
                                int(r.get("short_top_logprobs") or 0))
        reasons += [x for x in other_reasons(r, own) if x not in reasons]
    merged["leaderboard_comparable"] = not reasons
    merged["leaderboard_comparable_reason"] = "; ".join(reasons) or None
    status, status_reason, harness_kinds = own_loop_status(merged.get("sample_traces") or [])
    merged["status"], merged["status_reason"] = status, status_reason
    if harness_kinds:
        merged["request_harness_errors"] = harness_kinds
    return merged


def run_mmlu(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run MMLU by the 5-shot next-token log-probabilities of the answer letters (hendrycks/test
    evaluate.py protocol).

    `--eval-n-shot N` overrides the shot count (0 to 5; 0 = zero-shot) and makes the run
    non-comparable."""
    _ns = kwargs.get("eval_n_shot")
    if _ns is None:
        _ns = kwargs.get("n_shot")
    n_shot = MAX_N_SHOT if _ns is None else int(_ns)   # canonical 5-shot default; explicit 0 respected
    if n_shot > MAX_N_SHOT:
        raise ValueError(f"mmlu: --eval-n-shot {n_shot} exceeds the {MAX_N_SHOT} dev-split "
                         "exemplars per subject")
    limit = kwargs.get("limit")
    tok_name = kwargs.get("tokenizer") or get_run_knob("tokenizer") or model_name
    enc = _Encoder(_load_tokenizer(tok_name))
    items = _load_mmlu_items(limit=limit, n_shot=n_shot)
    if not items:
        raise RuntimeError("mmlu: no questions to score")
    eval_plan = eval_limit_plan("mmlu", limit, len(items))
    window = get_run_knob("max_model_len")
    samples, shots_used = [], []
    for it in items:
        context, k = _fit_to_window(it, enc, window)
        samples.append({"subject": it["subject"], "context": context, "gold": it["gold"]})
        shots_used.append(k)
    shots_reduced = sum(1 for k, it in zip(shots_used, items) if k < len(it["shots"]))
    if shots_reduced:
        logger.info("[mmlu] %d prompt(s) dropped trailing exemplars to fit the %s-token window "
                    "(evaluate.py's crop rule).", shots_reduced, window)
    thinking_note = None
    if enable_thinking:
        thinking_note = ("--thinking was requested but not applied: MMLU is scored from the "
                         "log-probabilities of the answer letters as the next token of a "
                         "plain-text prompt (evaluate.py's one generated token is not scored), so "
                         "this is the same score a no-think run gives")
        logger.warning("[mmlu] %s.", thinking_note)
    # the request is a no-think one whatever --thinking says (see thinking_note)
    temperature, temperature_source = resolve_temperature("mmlu", thinking=False)
    start = time.time()
    scored = asyncio.run(_score_all(base_url, model_name, enc, samples, concurrency,
                                    temperature))

    from .vqa_common import mark_no_reply, own_loop_status
    traces, category_stats = [], {}
    correct = failed = letters_missing = no_letter = short_lists = 0
    returned_sizes = []
    for idx, (sample, res, k) in enumerate(zip(samples, scored, shots_used)):
        if res["status"] == "OK":
            st = category_stats.setdefault(sample["subject"], {"correct": 0, "total": 0})
            st["total"] += 1
            if res["is_correct"]:
                st["correct"] += 1
                correct += 1
        else:
            failed += 1
        missing = res.get("letters_missing") or []
        returned = res.get("top_logprobs_returned")
        letters_missing += len(missing)
        no_letter += len(missing) == len(OPTION_LETTERS)
        if returned is not None:
            returned_sizes.append(returned)
            short_lists += bool(missing) and returned < TOP_LOGPROBS
        traces.append({
            "sample_idx": idx, "category": sample["subject"], "prompt": sample["context"],
            # the question's id across shards (base.sample_key), for `gbench --merge`
            "sample_key": sample_key({"subject": sample["subject"],
                                      "row_index": items[idx].get("row_index"),
                                      "question": items[idx]["question"], "gold": sample["gold"]}),
            # its position in the full ordered set the shards partition (the pinned split)
            "global_idx": items[idx].get("row_index"),
            "n_shot": k, "gold_answer": OPTION_LETTERS[sample["gold"]],
            "predicted": OPTION_LETTERS[res["pred"]] if res["pred"] is not None else None,
            "choice_loglikelihoods": res["loglikelihoods"], "letters_missing": missing,
            "top_logprobs_returned": returned, "is_correct": res["is_correct"],
            "status": res["status"], "error": res["error"],
        })
        if res["status"] != "OK":
            # No usable reply from the endpoint after the retries (a failed request, or a reply
            # keyed in another format): a harness error, left out of the score, never a wrong
            # answer; counted and named.
            mark_no_reply(traces[-1], "FAILED", res["error"])
    if letters_missing:
        logger.warning(
            "[mmlu] %d answer letter(s) were not among the top %d log-probabilities and count "
            "-100, as evaluate.py counts them (%d question(s) had no letter there at all and "
            "take A, np.argmax's first index).", letters_missing, TOP_LOGPROBS, no_letter)
    total = len(samples)
    scored_ok = total - failed
    if scored_ok and no_letter == scored_ok:
        keys = next((r.get("top_keys_sample") for r in scored if r.get("top_keys_sample")), [])
        raise infra_required(
            "mmlu", f"none of the {scored_ok} scored replies listed ' A', ' B', ' C' or ' D' "
            f"among the top-{TOP_LOGPROBS} log-probabilities of the generated token, so "
            f"evaluate.py's read would give every question A (np.argmax of four -100s). That is "
            f"a server whose top_logprobs are keyed otherwise than by token text with the "
            f"leading space, not a model result (keys of the first such reply: {keys!r}). A vLLM "
            f"server keys them so unless started with --return-tokens-as-token-ids; vLLM before "
            f"v0.26.0 drops the space for tokenizers whose one-token decode drops it.", DOCS_URL)
    status, status_reason, harness_kinds = own_loop_status(traces)
    attempts_requested = int(kwargs.get("attempt_count") or get_run_knob("attempt_count", 1) or 1)
    noncanon = _noncanon_reasons(n_shot, limit, total, failed, short_lists)
    result = {
        "benchmark_type": "eval",
        "eval_name": "mmlu",
        "eval_limit_plan": eval_plan,
        "model_name": model_name,
        "thinking": False,
        "thinking_requested": bool(enable_thinking),
        "thinking_applied": False,
        "total_questions": total,
        "correct_answers": correct,
        "failed_requests": failed,
        # the questions left out of the score (harness errors), by kind
        "scoring_excluded": failed,
        **({"request_harness_errors": harness_kinds} if harness_kinds else {}),
        # over the scored questions; None (withheld) when none is scored
        "accuracy": round(correct / scored_ok * 100.0, 2) if scored_ok else None,
        **({"accuracy_withheld_reason": status_reason} if not scored_ok else {}),
        "effective_n": scored_ok,
        "category_accuracy": {c: {**s, "accuracy": round(s["correct"] / s["total"] * 100.0, 2)}
                              for c, s in sorted(category_stats.items())},
        "scoring_mode": "loglikelihood",
        "metric": (f"accuracy: {n_shot}-shot, the answer letter with the highest next-token "
                   f"log-probability among the top {TOP_LOGPROBS} (-100 for a letter outside "
                   "them; hendrycks/test evaluate.py), pooled over all questions"),
        "top_logprobs_requested": TOP_LOGPROBS,
        "top_logprobs_returned_max": max(returned_sizes) if returned_sizes else None,
        "letters_missing": letters_missing,
        "questions_without_a_letter": no_letter,
        "short_top_logprobs": short_lists,
        "n_shot": n_shot,
        "shots_reduced": shots_reduced,
        "context_window": window,
        "temperature": temperature,
        "temperature_source": temperature_source,
        "temperature_note": (f"evaluate.py sends temperature={REQUEST_TEMPERATURE}; the letters "
                             "are read from the log-probabilities, so the temperature does not "
                             "change the score"),
        "tokenizer": tok_name,
        "bos_prepended": enc.bos_prepended,
        "attempts": ({"attempts_requested": attempts_requested, "attempts_per_sample": 1,
                      "not_applied": ("evaluate.py makes one request per question; the "
                                      "letters are read from one greedy step (the server's "
                                      "numerics may still vary slightly between runs under "
                                      "concurrent load)")}
                     if attempts_requested > 1 else None),
        "leaderboard_comparable": not noncanon,
        "sample_traces": traces,
        "status": status,
        # what made the status other than success (None on success)
        "status_reason": status_reason,
        "duration_s": round(time.time() - start, 2),
    }
    if thinking_note:
        result["thinking_not_applied_reason"] = thinking_note
    mot = kwargs.get("max_output_tokens")
    if mot is not None:
        result["max_output_tokens_note"] = (
            f"--max-output-tokens {mot} not applied: hendrycks/test evaluate.py requests "
            f"max_tokens=1 and reads the letters from its top-{TOP_LOGPROBS} log-probabilities")
    if noncanon:
        result["leaderboard_comparable_reason"] = "; ".join(noncanon)
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(record_shard(result, _shard_label()))
