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

"""Native MMLU-Pro evaluation suite (10 options A-J): TIGER-Lab's runner for OpenAI-compatible
endpoints.

Ported from TIGER-AI-Lab/MMLU-Pro `evaluate_from_apiX.py` (commit f418b116), the runner the
repository's README documents "to use universal (OpenAI chat.completions) API", i.e. a served
model behind an OpenAI-compatible `/v1` URL, run with its command-line defaults
(`--max_tokens 32768`, `--retry 3`, `--retry_wrong 0`, no `--save-thinking`):
  * data: `TIGER-Lab/MMLU-Pro` (pinned), the `test` split scored and the `validation` split's
    worked exemplars (5 per category) in the prompt; options equal to "N/A" are dropped
    (`preprocess`);
  * prompt (`process_one`): one user message and no system message, the same with and without
    `--thinking`: the category header, every validation exemplar of the category
    (`format_example` with its chain of thought), then the question ending
    "Answer: Let's think step by step.";
  * request (`streaming_request`): `temperature=0.1`, `top_p=0.95`, `max_tokens=32768`, streamed
    with `stream_options={"include_usage": True}`, through the openai client's `AsyncOpenAI`
    (timeout 3600 s, the client's default retries). The runner sends the same request to
    reasoning models and counts their streamed reasoning against max_tokens, so its 32768 is an
    explicit reasoning-model setting too (base.OfficialSampling.reasoning). Under base's
    token-limit policy a no-think run sends 32768 exactly; a `--thinking` run sends 32768, or the
    run's `--max-output-tokens` when that is larger (a thinking run is never capped below it),
    and that value is also the limit of the client's token count (upstream's `--max_tokens`);
  * try bound: upstream wraps each streamed try in `async with asyncio.timeout(3600)`, which
    needs Python 3.11. gbench applies the same 3600 s bound with `asyncio.wait_for`, available on
    every Python gbench supports (3.10 and later): the request, the whole stream and the usage
    check must finish within it, else the try fails with TimeoutError and is asked again after
    5 s like any other failed try, and the stream is closed either way;
  * token limit: the client counts the prompt and every streamed reasoning and content delta
    with tiktoken's `cl100k_base`. Once the count passes max_tokens it drops the stream
    (MaxTokensExceeded) and asks again after min(2^n, 30) s, 3 tries in all; after the third the
    question is recorded as MaxTokensExceeded and scores 0. A stream that ends without usage
    statistics, or fails, is asked again after 5 s;
  * reply: the streamed content is scored. The reasoning channel is counted, and kept in the
    trace's `reasoning` field for inspection, but never scored (upstream without
    `--save-thinking` keeps only the content);
  * extraction (`extract_answer`): for `answer is \\(?([A-J])\\)?`, then `[aA]nswer:\\s*([A-J])`,
    then `\\b[A-J]\\b`, the LAST match of the first pattern that matches; a reply with no match
    scores 0 (there is no random guess, and wrong answers are not asked again);
  * metric: correct / questions, per category and over all questions (`save_summary`);
  * pacing: the categories run one after another. Within one, request starts are at least 2 s
    apart (`enforce_start_delay`) and no new question starts while a running one has streamed
    nothing for 25 s (the stall check). gbench's concurrency plays the part of `--num_workers`
    (recorded as `harness.num_workers`; it changes the pacing, not the scores). A full run
    therefore takes at least 12,032 x 2 s, about 6.7 hours; each `--shard` has its own spacing.
    Each category's start and end, and every failed try, are logged.

Where gbench differs: upstream asks a failing request again without limit; gbench stops after
10 tries and records the question as FAILED: a harness error, never a wrong answer, left out of
`accuracy` and `category_accuracy` (`harness_error` and `scoring_excluded` on its trace,
`scoring_excluded` and `request_harness_errors` in the result), counted in `failed_requests`,
named in `status_reason` (status `completed_with_errors`), and the run is non-comparable; with
no question scored, `accuracy` is null with `accuracy_withheld_reason`. An endpoint that
rejects the request outright (HTTP 4xx other than 408, 409 and 429, for example a context
window too small for the prompt plus max_tokens) is a hard error, since no try can succeed, and
so is a TypeError, AttributeError, ImportError or NameError raised by the client's request call,
`chat.completions.create` (an incompatible `openai` package or Python, which no further try can
fix). The same errors raised while the stream is read (a malformed chunk, for example one with no
`delta`) fail that try like any other error, and it is asked again after 5 s, as upstream's
process_one asks it again. The run's thinking mode is sent as
`chat_template_kwargs.enable_thinking`, as every gbench suite sends it. `--max-output-tokens` is
not applied to a no-think run (the result says so in `max_output_tokens_note`).

Requirements: the `openai` (2.31 or later) and `tiktoken` packages, both in gbench's `evals`
extra; tiktoken downloads `cl100k_base` on first use (set TIKTOKEN_CACHE_DIR to reuse it
offline). Either missing is a hard error.

Sampling: top_p 0.95 in both modes; the temperature follows gbench's rule
(base.resolve_temperature), not the upstream 0.1: 1.0 with `--thinking`, 0.0 without. The
upstream 0.1 is recorded (base.OfficialSampling; the result records `official_sampling`), and a
run at another value is non-comparable. Override the temperature for a whole run with
`--temperature`, or for this suite alone with `GBENCH_MMLU_PRO_TEMPERATURE`, which takes
precedence over both (and over the rule's 1.0 / 0.0). LLM-judge grading is pinned at 0.0
and is not affected by either (this suite has no judge).

Sharding: `--shard I/N` selects shard I of the test questions (round-robin in dataset order)
before the `--eval-limit` draw.
"""

import asyncio
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from .limit_plan import eval_limit_plan
from .base import (OfficialSampling, _prompt_token_bounds, answer_in_thinking_fields,
                   apply_shard, decoding_penalties,
                   effective_chat_template_kwargs, finalize_shard_fields, get_run_knob,
                   record_shard, redact_secrets, sample_key,
                   redact_text, resolve_official_sampling, resolve_token_budget)
from .sampling import limit_dataset, parse_shard
from .swebench_common import infra_required

logger = logging.getLogger(__name__)

DATASET = "TIGER-Lab/MMLU-Pro"
DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
DOCS_URL = "docs/evals/mmlu_pro.md"
UPSTREAM = "TIGER-AI-Lab/MMLU-Pro@f418b116 evaluate_from_apiX.py"

OPTION_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

#: The validation split holds exactly 5 exemplars per category; upstream puts all of them in.
MAX_N_SHOT = 5
#: TIGER-Lab/MMLU-Pro test split size (the denominator of a full run).
_TEST_SIZE = 12032

#: `--max_tokens` default: sent as max_tokens and the limit of the client's own token count.
UPSTREAM_MAX_TOKENS = 32768
#: streaming_request: `temperature=0.1`, `top_p=0.95` (no other sampling field) and
#: `max_tokens=args.max_tokens`. The same request goes to a reasoning model, whose streamed
#: reasoning the client counts against max_tokens: that is the upstream's explicit
#: reasoning-model setting (`reasoning`), which base's token-limit policy applies to a --thinking
#: run without capping it below --max-output-tokens.
OFFICIAL_SAMPLING = OfficialSampling(
    reference=(f"{UPSTREAM} streaming_request: temperature=0.1, top_p=0.95, max_tokens="
               f"--max_tokens (default {UPSTREAM_MAX_TOKENS}), the same request for every model"),
    temperature=0.1, params={"top_p": 0.95}, max_tokens=UPSTREAM_MAX_TOKENS,
    reasoning=OfficialSampling(
        reference=(f"{UPSTREAM} streaming_request for a reasoning model: the same request, with "
                   "the streamed reasoning counted against max_tokens (default "
                   f"{UPSTREAM_MAX_TOKENS})"),
        temperature=0.1, params={"top_p": 0.95}, max_tokens=UPSTREAM_MAX_TOKENS))
#: `--retry` default: tries per question when the token count passes max_tokens.
UPSTREAM_TOKEN_TRIES = 3
#: The client's token counter (get_tokenizer).
TOKEN_ENCODING = "cl100k_base"
#: get_async_client's client timeout and streaming_request's per-try bound (`asyncio.timeout`
#: upstream, asyncio.wait_for here so the bound also holds on Python 3.10).
REQUEST_TIMEOUT_S = 3600.0
#: process_one: the wait before asking again after a failed try.
ERROR_RETRY_WAIT_S = 5.0
#: gbench's bound on tries after errors (upstream asks again without limit).
ERROR_TRIES = 10
#: enforce_start_delay: minimum spacing of request starts within a category.
START_SPACING_S = 2.0
#: _STALL_THRESHOLD_SECS: seconds without a streamed token before a question counts as stalled.
STALL_S = 25.0
#: render_interval: how often upstream's stall check runs.
STALL_CHECK_INTERVAL_S = 1.0
#: HTTP statuses the endpoint may return for a request that could succeed if asked again.
_RETRYABLE_4XX = (408, 409, 429)
#: Exceptions of the installed client or Python, not of the endpoint, when the request call
#: itself (chat.completions.create) raises them: every try would raise the same, so the suite
#: stops at the first one instead of spending ERROR_TRIES tries on each question. Raised while
#: the stream is read (a malformed chunk), they fail that try like any other error.
_CLIENT_ERRORS = (TypeError, AttributeError, ImportError, NameError)

# evaluate_from_apiX.py `_PATTERNS`, verbatim.
_PATTERNS = (r"answer is \(?([A-J])\)?", r"[aA]nswer:\s*([A-J])", r"\b[A-J]\b")


def _preprocess_options(options: Any) -> List[str]:
    """Upstream `preprocess`: drop the "N/A" padding options."""
    return [opt for opt in (options or []) if opt != "N/A"]


def format_example(question, options, cot_content=""):
    if not cot_content:
        cot_content = "Let's think step by step."
    if cot_content.startswith("A: "):
        cot_content = cot_content[3:]
    example = f"Question: {question}\nOptions: "
    choice_map = "ABCDEFGHIJ"
    for i, opt in enumerate(options):
        example += f"{choice_map[i]}. {opt}\n"
    if cot_content:
        example += f"Answer: {cot_content}\n\n"
    else:
        example += "Answer: "
    return example


def extract_answer(text):
    for pattern in _PATTERNS:
        if matches := re.findall(pattern, text):
            return matches[-1]
    return None


def _prompt_header(category: str) -> str:
    """process_one's prompt header for `category`, verbatim."""
    return (f"The following are multiple choice questions (with answers) about {category}. "
            f"Think step by step and then output the answer in the format of "
            f'"The answer is (X)" at the end.\n\n')


def _few_shot_exemplars(n_shot: int) -> Dict[str, str]:
    """The first `n_shot` validation exemplars of each category, rendered as upstream does.

    A failed load is a hard error: running 0-shot while the scaffold records `n_shot` would
    label the result with a protocol it did not run.
    """
    if n_shot <= 0:
        return {}
    if n_shot > MAX_N_SHOT:
        raise ValueError(f"mmlu_pro: --eval-n-shot {n_shot} exceeds the {MAX_N_SHOT} official "
                         "exemplars per category in the validation split")
    from datasets import load_dataset
    try:
        dev = list(load_dataset(DATASET, split="validation", revision=DATASET_REVISION))
    except Exception as e:
        raise infra_required(
            "mmlu_pro", f"the {n_shot}-shot exemplars could not be loaded from {DATASET} "
            f"(validation split, revision {DATASET_REVISION[:8]}): {e}. Check Hugging Face "
            "Hub access.", DOCS_URL) from e

    by_category: Dict[str, List[str]] = {}
    for item in dev:
        shots = by_category.setdefault(str(item.get("category", "")), [])
        if len(shots) < n_shot:
            shots.append(format_example(item["question"], _preprocess_options(item["options"]),
                                        item["cot_content"]))
    short = sorted(c for c, shots in by_category.items() if len(shots) < n_shot)
    if short:
        raise infra_required(
            "mmlu_pro", f"the validation split has fewer than {n_shot} exemplars for {short}.",
            DOCS_URL)
    return {c: "".join(shots) for c, shots in by_category.items()}


def _load_mmlu_pro_samples(enable_thinking: bool = False, limit: Optional[int] = None,
                           n_shot: int = 0) -> List[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]]:
    """Load MMLU-Pro (test split, pinned revision) as TIGER-Lab's CoT prompts.

    Returns (messages, gold letter, meta) with meta `category`, `question_id` and `row` (the
    question's position in the test split). `enable_thinking` does not change the prompt; it is
    accepted for loader-signature parity.
    """
    from datasets import load_dataset
    ds = load_dataset(DATASET, split="test", revision=DATASET_REVISION)
    rows = [dict(item, _gbench_row=i) for i, item in enumerate(ds)]
    # --shard first (round-robin in dataset order), then the --eval-limit draw.
    rows, _ = apply_shard(rows, "mmlu_pro")
    # Stratified, not a contiguous head.
    raw_samples = list(limit_dataset(rows, limit, "category", seed="mmlu_pro"))
    logger.info(f"Loaded {len(raw_samples)} MMLU-Pro samples from HF Hub ('{DATASET}').")

    few_shot = _few_shot_exemplars(n_shot)
    if few_shot:
        logger.info("[mmlu_pro] prepending %d-shot CoT exemplars for %d categories",
                    n_shot, len(few_shot))

    samples = []
    for item in raw_samples:
        ans = item.get("answer")
        if ans is None:
            ans = item.get("answer_index", 0)
        if isinstance(ans, str) and ans.strip().upper() in OPTION_LETTERS:
            gold_letter = ans.strip().upper()
        else:
            gold_letter = OPTION_LETTERS[int(ans)]

        category = str(item.get("category", ""))
        if n_shot > 0 and category not in few_shot:
            raise infra_required(
                "mmlu_pro", f"no validation exemplars for test category '{category}'.", DOCS_URL)
        prompt = (_prompt_header(category) + few_shot.get(category, "")
                  + format_example(item.get("question", ""),
                                   _preprocess_options(item.get("options"))))
        messages = [{"role": "user", "content": prompt}]
        samples.append((messages, gold_letter, {"category": category,
                                                "question_id": item.get("question_id"),
                                                "row": item["_gbench_row"]}))
    return samples


def _eval_mmlu_pro(response_text: str, gold_letter: str) -> bool:
    """Upstream scoring: the extracted letter equals the gold one; no letter scores 0."""
    pred = extract_answer(str(response_text or ""))
    return pred is not None and pred == gold_letter


# --- the runner --------------------------------------------------------------------------------

def _token_retry_wait(tries: int) -> float:
    """process_one's wait before asking again after MaxTokensExceeded: min(2^n, 30) s."""
    return min(2 ** tries, 30)


class _MaxTokensExceeded(Exception):
    """streaming_request's MaxTokensExceeded: the client's token count passed max_tokens."""


class _MissingUsage(Exception):
    """streaming_request's MissingUsageError: the stream ended without usage statistics."""


class _ClientCallError(Exception):
    """One of _CLIENT_ERRORS raised by the request call itself (chat.completions.create), before
    any chunk was read; the original exception is the __cause__."""


class _Pacer:
    """evaluate_subject_async's scheduling state for one category: request starts at least
    START_SPACING_S apart, and a stall check (a running question with no streamed token for
    STALL_S) that holds back new questions."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._last_start = 0.0
        self._last_token: Dict[int, Optional[float]] = {}
        self._stalled: Dict[int, bool] = {}
        self._last_check = 0.0

    async def start_delay(self) -> None:
        """enforce_start_delay."""
        async with self._lock:
            now = time.time()
            if self._last_start > 0:
                delay = max(0.0, START_SPACING_S - (now - self._last_start))
                if delay > 0:
                    await asyncio.sleep(delay)
            self._last_start = time.time()

    def reset(self, q: int) -> None:
        """update_token(q, 0): no token history, not stalled."""
        self._last_token[q] = None
        self._stalled[q] = False

    def token(self, q: int) -> None:
        """update_token(q, n > 0): a token arrived now."""
        self._last_token[q] = time.time()
        self._stalled[q] = False

    def done(self, q: int) -> None:
        self._last_token.pop(q, None)
        self._stalled.pop(q, None)

    def any_stalled(self) -> bool:
        """_check_stalls (at most once per render interval), then any_stalled."""
        now = time.time()
        if now - self._last_check >= STALL_CHECK_INTERVAL_S:
            self._last_check = now
            for q, ts in self._last_token.items():
                if ts is not None and now - ts > STALL_S:
                    self._stalled[q] = True
        return any(self._stalled.values())


def _hard_error(exc: BaseException, max_tokens: int = UPSTREAM_MAX_TOKENS
                ) -> Optional[RuntimeError]:
    """A failure no retry can fix, as an infra_required error; else None: an endpoint
    rejection (HTTP 4xx other than 408/409/429), or an error of the installed client or Python
    raised by the request call itself (_ClientCallError), which every try would raise again. The
    same exception types raised while the stream is read are ordinary failed tries."""
    if isinstance(exc, _ClientCallError):
        inner = exc.__cause__ if exc.__cause__ is not None else exc
        message = redact_text(f"{type(inner).__name__}: {inner}")[:300]
        return infra_required(
            "mmlu_pro", f"the openai client's chat.completions.create raised {message}, an error "
            "of the installed client or Python rather than of the endpoint, which no further try "
            "can fix. Install gbench's `evals` extra (openai 2.31 or later) on Python 3.10 or "
            "later.", DOCS_URL)
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int) or not 400 <= status < 500 or status in _RETRYABLE_4XX:
        return None
    message = redact_text(str(getattr(exc, "message", None) or exc))[:300]
    return infra_required(
        "mmlu_pro", f"the endpoint rejected the request evaluate_from_apiX.py sends "
        f"(HTTP {status}: {message}). It must accept a chat request with max_tokens="
        f"{max_tokens} on top of the prompt, so serve a context window of at least "
        f"the longest prompt plus {max_tokens:,} tokens.", DOCS_URL)


async def _stream_once(client: Any, request: Dict[str, Any], prompt: str, encoding: Any,
                       pacer: _Pacer, q: int) -> Dict[str, Any]:
    """streaming_request: one streamed try, counting tokens as upstream does.

    The try is bounded by REQUEST_TIMEOUT_S around the request, the whole stream and the usage
    check, as upstream's `async with asyncio.timeout(3600)` bounds it. asyncio.timeout needs
    Python 3.11, so the bound is applied with asyncio.wait_for, which every supported Python
    has: past it the try raises TimeoutError (asyncio.TimeoutError on 3.10), which _process_one
    asks again like any other failed try. The stream is closed either way."""
    limit = int(request["max_tokens"])
    total = len(encoding.encode(prompt))
    pacer.token(q)
    opened: Dict[str, Any] = {}

    async def _consume() -> Dict[str, Any]:
        count = total
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        usage, finish = None, None
        try:
            stream = await client.chat.completions.create(**request)
        except _CLIENT_ERRORS as e:
            # The request call itself: the installed client or Python (a hard error). The same
            # exception types raised below, while chunks are read, fail only this try.
            raise _ClientCallError(e) from e
        opened["stream"] = stream
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
                continue
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta
                finish = getattr(choice, "finish_reason", None) or finish
                content = delta.content or ""
                thinking = (getattr(delta, "reasoning_content", None)
                            or getattr(delta, "reasoning", None) or "")
                if content or thinking:
                    count += len(encoding.encode(thinking + content))
                    pacer.token(q)
                    if count > limit:
                        tail = ("".join(reasoning_parts) + "".join(content_parts))[-2000:]
                        raise _MaxTokensExceeded(count, tail)
                    if content:
                        content_parts.append(content)
                    if thinking:
                        reasoning_parts.append(thinking)
        if usage is None:
            raise _MissingUsage("Server did not provide usage statistics")
        return {"content": "".join(content_parts), "reasoning": "".join(reasoning_parts) or None,
                "client_tokens": count, "finish_reason": finish,
                "usage": {k: getattr(usage, k, None)
                          for k in ("prompt_tokens", "completion_tokens", "total_tokens")}}

    try:
        return await asyncio.wait_for(_consume(), REQUEST_TIMEOUT_S)
    finally:
        stream = opened.get("stream")
        if stream is not None:
            try:
                await stream.close()
            except Exception:
                pass


def _qname(meta: Dict[str, Any], q: int) -> str:
    """How a question is named in the log: its MMLU-Pro question_id, else its index."""
    qid = meta.get("question_id")
    return f"question {qid}" if qid is not None else f"sample {q}"


async def _process_one(client: Any, request_base: Dict[str, Any], encoding: Any, pacer: _Pacer,
                       q: int, sample: Tuple[List[Dict[str, Any]], str, Dict[str, Any]]
                       ) -> Dict[str, Any]:
    """process_one with `--retry 3 --retry_wrong 0`: one question, scored."""
    messages, gold, meta = sample
    prompt = messages[0]["content"]
    request = {**request_base, "messages": messages}
    pacer.reset(q)
    token_exceeded, errors, partial = 0, [], None
    reply: Optional[Dict[str, Any]] = None
    status = "OK"
    while True:
        try:
            await pacer.start_delay()
            reply = await _stream_once(client, request, prompt, encoding, pacer, q)
            break
        except _MaxTokensExceeded as e:
            token_exceeded += 1
            partial = e.args[1] if len(e.args) > 1 else None
            if token_exceeded >= UPSTREAM_TOKEN_TRIES:
                status = "MAX_TOKENS_EXCEEDED"
                logger.info("[mmlu_pro] %s: the token count passed max_tokens=%d on %d tries; "
                            "recorded as MaxTokensExceeded", _qname(meta, q),
                            request["max_tokens"], token_exceeded)
                break
            logger.info("[mmlu_pro] %s: the token count passed max_tokens=%d (try %d of %d); "
                        "asking again in %d s", _qname(meta, q), request["max_tokens"],
                        token_exceeded, UPSTREAM_TOKEN_TRIES, _token_retry_wait(token_exceeded))
            pacer.reset(q)
            await asyncio.sleep(_token_retry_wait(token_exceeded))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            hard = _hard_error(e, int(request["max_tokens"]))
            if hard is not None:
                raise hard from None
            errors.append(redact_text(f"{type(e).__name__}: {e}")[:300])
            pacer.reset(q)
            if len(errors) >= ERROR_TRIES:
                status = "FAILED"
                logger.warning("[mmlu_pro] %s: no reply after %d tries (last: %s); recorded "
                               "as FAILED", _qname(meta, q), ERROR_TRIES, errors[-1])
                break
            logger.warning("[mmlu_pro] %s: try %d failed (%s); asking again in %g s",
                           _qname(meta, q), len(errors), errors[-1], ERROR_RETRY_WAIT_S)
            await asyncio.sleep(ERROR_RETRY_WAIT_S)
    pacer.done(q)
    content = reply["content"] if reply else None
    pred = extract_answer(content) if content is not None else None
    usage = (reply or {}).get("usage") or {}
    return {
        "category": meta.get("category"),
        "question_id": meta.get("question_id"),
        "row": meta.get("row"),
        "messages": messages,
        "gold_answer": gold,
        "response_text": content,
        "reasoning": (reply or {}).get("reasoning"),
        "predicted": pred,
        "is_correct": pred is not None and pred == gold,
        "status": status,
        "finish_reason": (reply or {}).get("finish_reason"),
        "client_token_count": (reply or {}).get("client_tokens"),
        "usage": usage or None,
        "completion_tokens": usage.get("completion_tokens"),
        "max_tokens_sent": int(request["max_tokens"]),
        "max_tokens_exceeded_tries": token_exceeded,
        "error_tries": len(errors),
        "error": errors[-1] if status == "FAILED" else None,
        **({"partial_response_tail": partial} if status == "MAX_TOKENS_EXCEEDED" else {}),
    }


async def _run_category(client: Any, request_base: Dict[str, Any], encoding: Any,
                        workers: int, questions: List[Tuple[int, Any]]) -> Dict[int, Any]:
    """evaluate_subject_async's scheduling loop for one category."""
    pacer = _Pacer()
    pending = list(questions)
    running: Dict[asyncio.Task, int] = {}
    outcomes: Dict[int, Any] = {}
    category = str(questions[0][1][2].get("category", "")) if questions else ""
    started = time.time()
    logger.info("[mmlu_pro] category %r: %d question(s)", category, len(questions))
    try:
        while pending or running:
            while len(running) < workers and pending and not pacer.any_stalled():
                idx, sample = pending.pop(0)
                task = asyncio.ensure_future(
                    _process_one(client, request_base, encoding, pacer, idx, sample))
                running[task] = idx
            if running:
                done, _ = await asyncio.wait(set(running), timeout=0.2,
                                             return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    idx = running.pop(task)
                    outcomes[idx] = task.result()        # a hard error propagates here
            else:
                await asyncio.sleep(0.1)
    finally:
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)
    done_list = list(outcomes.values())
    logger.info("[mmlu_pro] category %r done in %.0f s: %d/%d correct, %d MaxTokensExceeded, "
                "%d failed", category, time.time() - started,
                sum(1 for o in done_list if o.get("is_correct")), len(done_list),
                sum(1 for o in done_list if o.get("status") == "MAX_TOKENS_EXCEEDED"),
                sum(1 for o in done_list if o.get("status") == "FAILED"))
    return outcomes


def _categories_in_order(samples: List[Tuple[Any, str, Dict[str, Any]]]
                         ) -> List[List[Tuple[int, Any]]]:
    """Upstream's run order: categories in order of first appearance in the test split, each
    category's questions in dataset order."""
    groups: Dict[str, List[Tuple[int, Any]]] = {}
    ordered = sorted(enumerate(samples), key=lambda p: (p[1][2].get("row") is None,
                                                         p[1][2].get("row") or 0, p[0]))
    for idx, sample in ordered:
        groups.setdefault(str(sample[2].get("category", "")), []).append((idx, sample))
    return list(groups.values())


#: The openai client's key: a placeholder that is never sent (_drop_authorization).
CLIENT_KEY_PLACEHOLDER = "EMPTY"


async def _drop_authorization(request: Any) -> None:
    """httpx request hook: no key goes to the model endpoint, so the Authorization header the
    openai client builds from its placeholder key is removed."""
    request.headers.pop("Authorization", None)


async def _run_harness(base_url: str, model_name: str, samples: List[Any], concurrency: int,
                       temperature: float, thinking: bool,
                       extra_body: Dict[str, Any],
                       max_tokens: int = UPSTREAM_MAX_TOKENS) -> List[Dict[str, Any]]:
    """evaluate_from_apiX.py's main loop over the categories; one outcome per sample."""
    try:
        from openai import AsyncOpenAI, DefaultAsyncHttpxClient
    except ImportError as e:
        raise infra_required(
            "mmlu_pro", f"the `openai` package, the client evaluate_from_apiX.py streams with, "
            f"is not installed ({e}). Install it with `pip install openai`.", DOCS_URL) from e
    try:
        import tiktoken
        encoding = tiktoken.get_encoding(TOKEN_ENCODING)
    except Exception as e:
        raise infra_required(
            "mmlu_pro", f"tiktoken's {TOKEN_ENCODING} encoding, which evaluate_from_apiX.py "
            f"counts tokens with, could not be loaded ({e}). It downloads from "
            "openaipublic.blob.core.windows.net on first use; set TIKTOKEN_CACHE_DIR to a "
            "directory that holds it for an offline run.", DOCS_URL) from e
    request_base: Dict[str, Any] = {
        "model": model_name, "max_tokens": int(max_tokens), "temperature": temperature,
        "top_p": 0.95, "stream": True, "stream_options": {"include_usage": True},
        "extra_body": {"chat_template_kwargs": effective_chat_template_kwargs(None, thinking),
                       **extra_body},
    }
    # No key: gbench sends none to the model endpoint. openai 2.54 and later refuse a client
    # without credentials, so the client holds a placeholder and its HTTP client drops the
    # Authorization header from every request before it is sent.
    client = AsyncOpenAI(api_key=CLIENT_KEY_PLACEHOLDER, base_url=base_url,
                         timeout=REQUEST_TIMEOUT_S,
                         http_client=DefaultAsyncHttpxClient(
                             event_hooks={"request": [_drop_authorization]}))
    outcomes: Dict[int, Any] = {}
    try:
        for questions in _categories_in_order(samples):
            outcomes.update(await _run_category(client, request_base, encoding,
                                                max(1, int(concurrency or 1)), questions))
    finally:
        await client.close()
    return [outcomes[i] for i in range(len(samples))]


def _check_window(samples: List[Any], thinking: bool,
                  max_tokens: int = UPSTREAM_MAX_TOKENS) -> None:
    """Hard error when the served window cannot hold the longest prompt plus max_tokens."""
    window = get_run_knob("max_model_len")
    if not window or not samples:
        return
    longest = max(samples, key=lambda s: len(s[0][0]["content"]))[0]
    try:
        bounds = _prompt_token_bounds(longest, None,
                                      effective_chat_template_kwargs(None, thinking))
    except Exception:
        bounds = None
    if bounds and int(window) < bounds[0] + max_tokens:
        raise infra_required(
            "mmlu_pro", f"the served context window ({int(window):,} tokens) cannot hold the "
            f"longest prompt (~{bounds[0]:,} tokens) plus the max_tokens={max_tokens} "
            "sent with every request (evaluate_from_apiX.py's --max_tokens). Serve a window of "
            f"at least {bounds[0] + max_tokens:,} tokens.", DOCS_URL)


def run_mmlu_pro(model_name: str, base_url: str, concurrency: int, enable_thinking: bool = False, **kwargs) -> Dict[str, Any]:
    """Run MMLU-Pro with evaluate_from_apiX.py (canonical 5-shot CoT by default; `--eval-n-shot N`
    overrides, 0 = 0-shot, and makes the run non-comparable)."""
    _ns = kwargs.get("eval_n_shot")
    if _ns is None:
        _ns = kwargs.get("n_shot")
    n_shot = MAX_N_SHOT if _ns is None else int(_ns)   # canonical 5-shot CoT default; explicit 0 respected
    limit = kwargs.get("limit")
    samples = _load_mmlu_pro_samples(enable_thinking=enable_thinking, limit=limit, n_shot=n_shot)
    eval_plan = eval_limit_plan("mmlu_pro", limit, len(samples))
    temperature, temperature_source = resolve_official_sampling("mmlu_pro", OFFICIAL_SAMPLING,
                                                                thinking=enable_thinking)
    knob = get_run_knob("max_output_tokens") or kwargs.get("max_output_tokens")
    # The token-limit policy (base): 32768 exactly without thinking; with it, the upstream's
    # reasoning-model setting (the same 32768), never below --max-output-tokens.
    plan = resolve_token_budget("mmlu_pro", OFFICIAL_SAMPLING, thinking=enable_thinking,
                                max_output_tokens=knob)
    max_tokens = int(plan.max_tokens)
    penalties = decoding_penalties()
    _check_window(samples, enable_thinking, max_tokens)
    start = time.time()
    traces = (asyncio.run(_run_harness(base_url, model_name, samples, concurrency, temperature,
                                       enable_thinking, dict(penalties), max_tokens))
              if samples else [])
    for idx, t in enumerate(traces):
        t["sample_idx"] = idx
        # the question's id across shards (base.sample_key: its MMLU-Pro question_id), for
        # `gbench --merge`
        t["sample_key"] = sample_key({"question_id": t.get("question_id"),
                                      "messages": t.get("messages")})

    # A question with no reply after ERROR_TRIES tries is a harness error: left out of every
    # accuracy (never a wrong answer), counted and named. A MaxTokensExceeded question is
    # upstream's own outcome and scores 0.
    from .vqa_common import mark_no_reply, own_loop_status
    for t in traces:
        if t["status"] == "FAILED":
            mark_no_reply(t, "FAILED", t.get("error"))
    scored = [t for t in traces if not t.get("scoring_excluded")]
    stats: Dict[str, Dict[str, int]] = {}
    for t in scored:
        st = stats.setdefault(str(t.get("category")), {"correct": 0, "total": 0})
        st["total"] += 1
        st["correct"] += 1 if t["is_correct"] else 0
    total = len(traces)
    correct = sum(1 for t in scored if t["is_correct"])
    failed = sum(1 for t in traces if t["status"] == "FAILED")
    exceeded = sum(1 for t in traces if t["status"] == "MAX_TOKENS_EXCEEDED")
    answered = [t for t in traces if t["status"] == "OK"]
    no_answer = sum(1 for t in answered if t["predicted"] is None)
    attempts_requested = int(kwargs.get("attempt_count") or get_run_knob("attempt_count", 1) or 1)

    reasons = []
    if n_shot != MAX_N_SHOT:
        reasons.append(f"{n_shot}-shot (evaluate_from_apiX.py puts all {MAX_N_SHOT} validation "
                       "exemplars of the category in the prompt)")
    mode_sampling = OFFICIAL_SAMPLING.for_mode(enable_thinking)
    official = mode_sampling.record(temperature, temperature_source)
    official.update(plan.record())
    # The upstream's 32768 was applied only when it is what was sent: a thinking run with a larger
    # --max-output-tokens sends that instead (max_tokens_sent).
    official["max_tokens_applied"] = max_tokens == plan.official_cap
    if mode_sampling is not OFFICIAL_SAMPLING:
        official["reasoning_settings"] = True
    if max_tokens != UPSTREAM_MAX_TOKENS:
        reasons.append(f"max_tokens={max_tokens} sent and counted against (--max-output-tokens "
                       f"{knob}; a --thinking run is never capped below it), where "
                       f"evaluate_from_apiX.py's --max_tokens is {UPSTREAM_MAX_TOKENS}")
    if official.get("temperature_matches_official") is False:
        reasons.append(f"sampled at temperature {temperature} ({temperature_source}; "
                       f"evaluate_from_apiX.py sends temperature {OFFICIAL_SAMPLING.temperature})")
    if penalties:
        reasons.append(f"decoding penalties {penalties} (evaluate_from_apiX.py sends none)")
    if failed:
        reasons.append(f"{failed} question(s) got no reply from the served endpoint after "
                       f"{ERROR_TRIES} tries (harness errors, left out of the score; "
                       "evaluate_from_apiX.py asks a failing request again without limit)")
    status, status_reason, harness_kinds = own_loop_status(traces)
    if limit:
        reasons.append(f"--eval-limit subset ({total} of {_TEST_SIZE:,} questions)")

    result: Dict[str, Any] = {
        "benchmark_type": "eval",
        "eval_name": "mmlu_pro",
        "eval_limit_plan": eval_plan,
        "model_name": model_name,
        "thinking": bool(enable_thinking),
        "total_questions": total,
        "correct_answers": correct,
        "failed_requests": failed,
        # the questions left out of the score (harness errors), by kind
        "scoring_excluded": failed,
        **({"request_harness_errors": harness_kinds} if harness_kinds else {}),
        "effective_n": len(scored),
        "max_tokens_exceeded": exceeded,
        "no_answer_replies": no_answer,
        "empty_responses": sum(1 for t in answered if not (t["response_text"] or "").strip()),
        # of those, the replies that stopped with the answer inside the thinking channel (the
        # model's outcome, scored 0 as upstream scores an empty answer; base's
        # ANSWER_IN_THINKING_POLICY)
        **answer_in_thinking_fields(
            sum(1 for t in answered if not (t["response_text"] or "").strip()
                and str(t.get("reasoning") or "").strip() and t.get("finish_reason") != "length"),
            total),
        "truncated_responses": sum(1 for t in answered if t.get("finish_reason") == "length"),
        # over the scored questions; None (withheld) when none is scored
        "accuracy": round(correct / len(scored) * 100.0, 2) if scored else None,
        **({"accuracy_withheld_reason": status_reason} if traces and not scored else {}),
        "category_accuracy": {c: {**s, "accuracy": round(s["correct"] / s["total"] * 100.0, 2)}
                              for c, s in sorted(stats.items())},
        "scoring_mode": "deterministic",
        "metric": ("accuracy: correct / questions over all test questions (evaluate_from_apiX.py "
                   "save_summary; a MaxTokensExceeded question or a reply with no extractable "
                   "letter scores 0; a question with no reply from the endpoint is a harness "
                   "error, left out); per category in category_accuracy"),
        "n_shot": n_shot,
        "harness": {
            "reference": UPSTREAM,
            "max_tokens": max_tokens,
            "upstream_max_tokens": UPSTREAM_MAX_TOKENS,
            "token_counter": f"tiktoken {TOKEN_ENCODING} (prompt + streamed reasoning and "
                             "content deltas)",
            "token_limit_tries": UPSTREAM_TOKEN_TRIES,
            "retry_wrong": 0,
            "request_timeout_s": REQUEST_TIMEOUT_S,
            "start_spacing_s": START_SPACING_S,
            "stall_s": STALL_S,
            "num_workers": max(1, int(concurrency or 1)),
            "error_tries": ERROR_TRIES,
        },
        "max_tokens_sent": [max_tokens],
        "temperature": temperature,
        "temperature_source": temperature_source,
        "official_sampling": official,
        "decoding_penalties": penalties or None,
        "attempts": ({"attempts_requested": attempts_requested, "attempts_per_sample": 1,
                      "not_applied": "evaluate_from_apiX.py asks each question once "
                                     "(--retry_wrong 0)"}
                     if attempts_requested > 1 else None),
        "leaderboard_comparable": not reasons,
        "sample_traces": traces,
        "status": status,
        # what made the status other than success (None on success)
        "status_reason": status_reason,
        "duration_s": round(time.time() - start, 2),
    }
    if plan.note:
        result["max_output_tokens_note"] = plan.note
    if reasons:
        result["leaderboard_comparable_reason"] = "; ".join(reasons)
    shard = parse_shard(os.environ.get("GBENCH_SHARD"))
    # No credential of the environment leaves the suite in a trace or a result field.
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return redact_secrets(finalize_shard_fields(
        record_shard(result, f"{shard[0]}/{shard[1]}" if shard else None)))
