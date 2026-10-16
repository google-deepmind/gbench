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

"""Base async execution engine for self-contained evaluation benchmark suites.

Handles concurrent HTTP dispatching to OpenAI-compatible server, real-time tqdm
progress logging, error retries, and standardized metric calculations.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import itertools
import json
import logging
import os
import random
import re
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import weakref
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from .sampling import stratified_sample, parse_shard, shard_select


@functools.lru_cache(maxsize=None)
def _eval_fn_accepts_tool_calls(fn: Callable) -> bool:
    """True if `fn` declares a `tool_calls` parameter (or **kwargs). Function-calling suites opt in
    to being scored off the STRUCTURED tool calls (the text render is lossy); every other 2-arg
    eval_fn is called unchanged."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "tool_calls" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _apply_eval_fn(fn: Optional[Callable], pred: Any, gold: Any, tool_calls: Any) -> bool:
    """Call `fn(pred, gold)`, additionally passing the structured `tool_calls` when `fn` accepts it."""
    if fn is None:
        return False
    if _eval_fn_accepts_tool_calls(fn):
        return fn(pred, gold, tool_calls=tool_calls)
    return fn(pred, gold)


class Reply(NamedTuple):
    """One model response, with the metadata needed to tell WHY it looks the way it does."""
    text: Optional[str]
    tool_calls: Optional[List[Dict[str, Any]]]
    finish_reason: Optional[str] = None
    reasoning: Optional[str] = None
    error: Optional[str] = None
    completion_tokens: int = 0
    stop_reason: Optional[Any] = None
    #: Observed per-sequence decode rate for this request. Every timeout in the harness is
    #: sized from an ASSUMED rate; this is the measured one, so the assumption can be
    #: checked on the box that is actually running rather than the one it was tuned on.
    decode_tok_s: Optional[float] = None
    #: The server-parser policy's event for this reply (see recover_server_parser_text): None,
    #: SERVER_PARSER_MOVED_TEXT or SERVER_PARSER_DROPPED_TEXT (the text was recovered and is the
    #: model's output), or SERVER_PARSER_TEXT_LOST (not recoverable: a harness error).
    parser_event: Optional[str] = None
    #: Where a recovered text came from: "token_ids" (the generation decoded from the reply's
    #: token ids) or "reasoning_then_content" (the reasoning field followed by the content).
    parser_recovered_from: Optional[str] = None
    #: The max_tokens the answered request carried (a context-length retry can lower it below
    #: the budget first sent), None when it carried none.
    max_tokens_sent: Optional[int] = None


#: A 20-gram repeated this many times in a response that ran to the token cap is taken as
#: degenerate repetition rather than a long answer. Calibrated on the 2026-08-15 run:
#: looping responses showed 12-21 repeats, ordinary long outputs (patches, programs) stay
#: in the low single digits because real code rarely repeats a 20-word window verbatim.
REPETITION_NGRAM = 20
REPETITION_THRESHOLD = 8


def repetition_run(text: Optional[str]) -> int:
    """Most times any 20-word window repeats. 0 when the text is too short to judge."""
    words = str(text or "").split()
    if len(words) < REPETITION_NGRAM * 3:
        return 0
    from collections import Counter
    grams = Counter(" ".join(words[i:i + REPETITION_NGRAM])
                    for i in range(len(words) - REPETITION_NGRAM))
    return grams.most_common(1)[0][1] if grams else 0


def looping_channel(text: Optional[str], reasoning: Optional[str]) -> Tuple[Optional[str], str]:
    """(the text to report repetition from, its channel): the reply's content, unless only its
    reasoning repeats (a reply cut off while looping in its reasoning), then the reasoning."""
    if (repetition_run(text) < REPETITION_THRESHOLD and degenerate_char_run(text) is None
            and (repetition_run(reasoning) >= REPETITION_THRESHOLD
                 or degenerate_char_run(reasoning) is not None)):
        return reasoning, "reasoning"
    return text, "content"


def repetition_onset(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Where degenerate repetition begins, and what the model is stuck on.

    Reports the first occurrence of the most-repeated 20-word window, so a run can be
    correlated against prompt shape: does the model degenerate immediately (a prompt it
    cannot start), or only deep into a long derivation (a search it cannot terminate)?

    * `onset_word` / `onset_frac` - where the cycling text first appears, absolute and as a
      fraction of the whole response
    * `period_words` - distance between consecutive repeats, i.e. the size of the cycle
    * `cycle_preview` - what it is repeating, so the cause is legible without opening traces
    """
    words = str(text or "").split()
    if len(words) < REPETITION_NGRAM * 3:
        return None
    positions: Dict[str, List[int]] = {}
    for i in range(len(words) - REPETITION_NGRAM):
        positions.setdefault(" ".join(words[i:i + REPETITION_NGRAM]), []).append(i)
    gram, hits = max(positions.items(), key=lambda kv: len(kv[1]))
    if len(hits) < REPETITION_THRESHOLD:
        return None
    return {
        "onset_word": hits[0],
        "onset_frac": round(hits[0] / len(words), 3),
        "period_words": (hits[1] - hits[0]) if len(hits) > 1 else None,
        "repeat_count": len(hits),
        "total_words": len(words),
        "cycle_preview": gram[:160],
    }


#: A reply that ran to the cap made almost entirely of one repeated character (spaces, newlines,
#: '=' ...) is degenerate as well, though it may hold too few words for repetition_run to judge:
#: measured on an omnidocbench page, 991,323 of 991,363 characters were spaces at a 32,000-token
#: cap (6 words).
DEGENERATE_CHAR_SHARE = 0.9
DEGENERATE_MIN_CHARS = 1000


def degenerate_char_run(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """The run of one character (or of whitespace) a degenerate reply is made of, or None: a
    reply of at least DEGENERATE_MIN_CHARS characters whose whitespace, or whose most frequent
    character, is at least DEGENERATE_CHAR_SHARE of it. The record has repetition_onset's
    shape (onset_frac, repeat_count, cycle_preview) plus the run's own position and length."""
    s = str(text or "")
    if len(s) < DEGENERATE_MIN_CHARS:
        return None
    from collections import Counter
    ch, count = Counter(s).most_common(1)[0]
    blank = sum(1 for c in s if c.isspace())
    if blank / len(s) >= DEGENERATE_CHAR_SHARE:
        pattern, label, share = r"\s+", "whitespace", blank / len(s)
    elif count / len(s) >= DEGENERATE_CHAR_SHARE:
        pattern, label, share = re.escape(ch) + "+", repr(ch), count / len(s)
    else:
        return None
    longest = max(re.finditer(pattern, s), key=lambda m: m.end() - m.start())
    return {"kind": "character_run", "repeated": label, "share": round(share, 4),
            "onset_char": longest.start(), "onset_frac": round(longest.start() / len(s), 3),
            "period_words": None, "repeat_count": longest.end() - longest.start(),
            "total_chars": len(s), "cycle_preview": label}


#: A capped response this many times longer than the suite's own healthy median is
#: reclassified `truncated` -> `non_convergent`. 0 disables.
NON_CONVERGENT_LENGTH_RATIO = float(os.environ.get("GBENCH_NON_CONVERGENT_RATIO", "4.0"))


def reclassify_non_convergent(sample_traces: List[Dict[str, Any]],
                              official_caps: Optional[Dict[int, int]] = None) -> int:
    """Relabel capped responses that are far longer than this suite's healthy ones.

    `repetition_run` detects a 20-gram repeating verbatim. It cannot see a model that keeps
    re-deriving the same quantity with different algebra, which is what long-form reasoning
    failure actually looks like. Measured on AIME 2026-08-20: 7 samples capped at exactly
    16,384 tokens against a healthy median of 1,987 (8.2x), `repetition_run` 1-2 against a
    threshold of 8, tails reading "Let me re-calculate everything. Is it possible m=37 and
    n=...?" and 70 enumerated case markers on one of them.

    The distinction is operational, not cosmetic: `truncated` tells the operator to raise
    --max-output-tokens. For these it buys more of the same at more wall-clock. Calibrated
    against the run's OWN healthy median rather than a constant, so it transfers to suites
    whose answers are legitimately long. A reply stopped at the suite's official max_tokens
    (`official_caps`: sample_idx -> cap) keeps `truncated`: the protocol's own outcome, which a
    larger --max-output-tokens would not change (counted in replies_at_official_max_tokens).
    """
    official_caps = official_caps or {}
    if NON_CONVERGENT_LENGTH_RATIO <= 0:
        return 0
    healthy = [t.get("completion_tokens") or 0 for t in sample_traces
               if t.get("health") == "ok"]
    healthy = [n for n in healthy if n > 0]
    if len(healthy) < 8:
        return 0
    import statistics
    cutoff = statistics.median(healthy) * NON_CONVERGENT_LENGTH_RATIO
    n = 0
    for t in sample_traces:
        if (t.get("health") == "truncated" and t.get("sample_idx") not in official_caps
                and (t.get("completion_tokens") or 0) >= cutoff):
            t["health"] = "non_convergent"
            n += 1
    return n


#: Error prefix of a sample stopped by its ToolLoopProtocol.max_sample_wall_s limit.
WALL_CLOCK_LIMIT_ERROR = "wall_clock_limit"


class _ClockedSemaphore:
    """One sample's view of the run's request semaphore, with the sample's own clock.

    The clock starts when the sample's first request gets a request slot and does not run while
    the sample waits for a slot again later: gbench shares one request semaphore across all
    samples, so that waiting is time spent behind other samples, which a harness that runs each
    task in its own worker never sees. `active_s()` is the sample's elapsed time without it.
    """

    def __init__(self, semaphore: asyncio.Semaphore):
        self._semaphore = semaphore
        self.started = asyncio.Event()
        self._t0: Optional[float] = None
        self._queued = 0.0
        self._wait_since: Optional[float] = None

    async def __aenter__(self):
        loop = asyncio.get_running_loop()
        if self._t0 is not None:
            self._wait_since = loop.time()
        try:
            await self._semaphore.acquire()
        finally:
            if self._wait_since is not None:
                self._queued += loop.time() - self._wait_since
                self._wait_since = None
        if self._t0 is None:
            self._t0 = loop.time()
            self.started.set()
        return self

    async def __aexit__(self, *exc):
        self._semaphore.release()
        return False

    def active_s(self) -> float:
        if self._t0 is None:
            return 0.0
        now = asyncio.get_running_loop().time()
        queued = self._queued + (now - self._wait_since if self._wait_since is not None else 0.0)
        return now - self._t0 - queued


def classify_reply(reply: "Reply") -> str:
    """`ok` | `request_failed` | `truncated` | `empty_reasoned` | `empty`.

    Without this every one of these looked like an ordinary wrong answer:
      * `truncated`       - hit the output-token budget mid-answer. In the 2026-08-15
                            sweep this alone was codeforces' IOI 0/6 and 10/20 of
                            swe_bench_multilingual.
      * `empty_reasoned`  - the reply stopped (not at the budget) with reasoning but no
                            `content` and no tool call: the model wrote its answer inside its
                            thinking channel and ended the turn there (gemma-4 with
                            `--reasoning-parser gemma4`). The model's outcome, scored as an
                            upstream harness reading `content` scores an empty answer, and
                            counted as `answer_in_thinking` (ANSWER_IN_THINKING_POLICY).
      * `empty`           - the server returned 200 with nothing at all.
    """
    if reply.text is None:
        if (reply.error or "").startswith(WALL_CLOCK_LIMIT_ERROR):
            return "wall_clock_limit"
        return "request_timeout" if (reply.error or "").startswith("timeout") else "request_failed"
    has_output = bool(str(reply.text).strip()) or bool(reply.tool_calls)
    if reply.finish_reason == "length":
        # A response that hit the cap is only "truncated" if it was going somewhere. When
        # the model is stuck re-deriving the same step, more budget buys more of the same:
        # measured on 2026-08-15, gpqa_diamond produced 163,976 chars with a 20-gram
        # repeated 20x and ended on "Step 375: Let's try k=-283", and arc_agi re-enumerated
        # grid rows with "Wait, the rows are 1, 2,". Telling the operator to raise
        # --max-output-tokens there is wrong advice, so name it differently.
        # A reply made almost entirely of one character (a run of spaces) is the same failure
        # with too few words for the 20-gram check. The reasoning counts too: a reply cut off
        # while looping in its reasoning has little or no content (2026-10-06: 40 lcb replies
        # repeating one reasoning line up to 2,388 times were reported as truncated).
        if any(repetition_run(t) >= REPETITION_THRESHOLD or degenerate_char_run(t) is not None
               for t in (reply.text, reply.reasoning)):
            return "looping"
        return "truncated"
    if not has_output:
        if (reply.reasoning or "").strip():
            return "empty_reasoned"
        # The server generated tokens and returned none of them. On gemma-4 this is the
        # model emitting a tool call followed by `<|tool_response>` (token 50, a configured
        # stop token): vLLM's tool-call parser lifts the call out of `content`, and when the
        # request declared no `tools` there is nowhere to put it, so it is DISCARDED.
        # Measured live: completion_tokens=34, content=null, tool_calls=[], stop_reason=50.
        # That is a suite/serving mismatch, not a model that said nothing - scoring it as an
        # ordinary wrong answer hid 28 samples across bfcl_live / mcp_atlas / skillsbench.
        if reply.completion_tokens:
            return "output_discarded"
        return "empty"
    return "ok"


# --- The answer-in-thinking policy -------------------------------------------------------------
# A --thinking reply that stopped on its own (finish_reason "stop", not the budget) with reasoning
# and no content and no tool call (health `empty_reasoned`): the model wrote its answer inside its
# thinking channel and ended the turn there (measured on gemma-4, 2026-10-09: the reasoning ends
# with the answer, then `<turn|>` with no `<channel|>`, or with `<channel|><channel|><|channel>
# thought\n<channel|>` and a stop token). An upstream harness reads `content`, finds no answer and
# scores the sample as it scores an empty answer, so gbench does the same in every suite:
#   * the model's outcome, scored (an empty answer: wrong unless the benchmark's own rule makes an
#     empty answer right, `empty_is_answer`), never left out of the score;
#   * not a failed request and not a measurement problem: not in failed_requests or
#     empty_responses, does not make the run completed_with_errors or not comparable;
#   * counted and named per run: `answer_in_thinking` (replies) and `answer_in_thinking_rate`
#     (of the samples / turns), and `answer_in_thinking: true` on the trace.
# Checked on 2026-10-10 that gbench's rendering does not cause it: the multi-turn histories the
# agent harnesses send carry each earlier assistant turn's content only (Gemma 4's documented
# convention: earlier turns' thoughts are stripped), and replaying the requests rendered by the
# model's own chat template gave the same rate as the requests as sent.
ANSWER_IN_THINKING_POLICY = (
    "a reply that stopped with its answer inside the thinking channel (reasoning, no content, no "
    "tool call; health empty_reasoned) is the model's outcome, scored as an upstream harness "
    "reading content scores an empty answer; counted in answer_in_thinking, not a failed request. "
    "Relayed agentic suites tell it apart by token ids: a reply holding the tool-call token is a "
    "native tool call the server's parser dropped (tool_call_dropped, a comparability reason), "
    "one without token ids is unclassified (also a reason); gpqa, causalbench and "
    "longbench_v2_code first re-ask a null reply the model stopped itself, as their upstreams do")


def answer_in_thinking_fields(count: int, of: int) -> Dict[str, Any]:
    """The per-run answer-in-thinking record: `answer_in_thinking` (replies),
    `answer_in_thinking_of` (the replies, samples or model turns it is counted over) and
    `answer_in_thinking_rate` (count / of, 4 decimals; None when `of` is 0). `gbench --merge`
    sums the first two over the shards and recomputes the rate."""
    count, of = int(count or 0), int(of or 0)
    return {"answer_in_thinking": count, "answer_in_thinking_of": of,
            "answer_in_thinking_rate": round(count / of, 4) if of > 0 else None}


# --- The server-parser policy ------------------------------------------------------------------
# The served model runs behind the server's reasoning parser and tool-call parser (vLLM
# `--reasoning-parser` / `--tool-call-parser`, as the model's serving recipe sets them). They split
# a generation into `content`, `reasoning` and `tool_calls`, and two of their paths take text away
# from what a harness reads:
#   * moved text: on a request sent with thinking off, the reasoning parser can still put text in
#     `reasoning` (gemma-4: a thought channel the model opens on its own, or the parser's
#     first-feed rule that reads output starting with 'thought\n' or holding '<channel|>' as
#     reasoning). With thinking off that text is the model's output, not a reasoning trace.
#   * dropped text: the tool-call parser lifts a native tool call out of `content`; when the
#     request declared no tools there is nowhere to put it, and the reply comes back with no
#     content, no reasoning and no tool call although completion_tokens > 0.
# Canonical harnesses score the model's whole generation, so neither is the model saying nothing,
# and neither is scored as a model failure or dropped silently:
#   * moved text is recovered in generation order and scored as the model's output: the
#     generation decoded from the reply's token ids when the server returned them (base asks for
#     them with `return_token_ids`), else the reasoning field followed by the content;
#   * dropped text is recovered from the token ids and scored as the model's output; when the
#     server returned no token ids, or they could not be decoded, the reply is a harness error
#     (SERVER_PARSER_TEXT_LOST), left out of the score, counted and named;
#   * every event is counted and named per run (`server_parser_events`), and each trace records
#     its event and source (`server_parser`).
# A reply the server bound tool calls for is left as it is, and so is a --thinking reply with
# reasoning (the reasoning channel is the model's reasoning there). The token ids are decoded
# with special tokens kept (the server's /detokenize, else the run's tokenizer), which is the text
# vLLM builds before any parser runs; the stop token vLLM leaves out of the text is left out,
# and a matched stop string cuts the text as the server cuts it.
SERVER_PARSER_MOVED_TEXT = "server_parser_moved_text"
SERVER_PARSER_DROPPED_TEXT = "server_parser_dropped_text"
SERVER_PARSER_TEXT_LOST = "server_parser_text_lost"
SERVER_PARSER_EVENTS = (SERVER_PARSER_MOVED_TEXT, SERVER_PARSER_DROPPED_TEXT,
                        SERVER_PARSER_TEXT_LOST)


class ParserRecovery(NamedTuple):
    """What the server-parser policy made of one reply (recover_server_parser_text)."""
    #: None (nothing to recover) or one of SERVER_PARSER_EVENTS.
    event: Optional[str]
    #: The model's output to score in place of the content; None when `event` is None or
    #: SERVER_PARSER_TEXT_LOST.
    text: Optional[str]
    #: "token_ids" or "reasoning_then_content"; None when nothing was recovered.
    recovered_from: Optional[str]


def server_parser_signal(*, content: Optional[str], reasoning: Optional[str],
                         tool_calls: Any, thinking: bool,
                         completion_tokens: Optional[int]) -> Optional[str]:
    """Which server-parser path a reply shows, from the reply alone (no decoding).

    SERVER_PARSER_MOVED_TEXT: the request was sent with thinking off, the reply has a non-empty
    reasoning field and no bound tool call. SERVER_PARSER_DROPPED_TEXT: the reply has no content,
    no reasoning and no tool call, and the server generated tokens (completion_tokens > 0), in
    either thinking mode. None otherwise."""
    if tool_calls:
        return None
    if not thinking and str(reasoning or "").strip():
        return SERVER_PARSER_MOVED_TEXT
    if (not str(content or "").strip() and not str(reasoning or "").strip()
            and int(completion_tokens or 0) > 0):
        return SERVER_PARSER_DROPPED_TEXT
    return None


def generation_text_from_token_ids(token_ids: Any, decode: Optional[Callable[[List[int]], Any]],
                                   finish_reason: Optional[str] = None,
                                   stop_reason: Any = None) -> Optional[str]:
    """The server's output text rebuilt from a reply's generated token ids, as vLLM builds it
    before any parser runs.

    `decode(ids)` must keep special tokens (tokenizer.decode(ids, skip_special_tokens=False), what
    vLLM's /detokenize returns) and return None when it cannot decode. A reply that finished on
    a stop token (finish_reason "stop" with no stop string: the end-of-sequence token or a stop
    token id, which vLLM appends to token_ids but not to the text) has that last id left out; a
    matched stop string (`stop_reason` a str) cuts the text where it starts, as the server's text
    is cut. None when there are no ids or they cannot be decoded."""
    if decode is None or not isinstance(token_ids, (list, tuple)) or not token_ids:
        return None
    try:
        ids = [int(t) for t in token_ids]
    except (TypeError, ValueError):
        return None
    if finish_reason == "stop" and not isinstance(stop_reason, str):
        ids = ids[:-1]
    if not ids:
        return ""
    try:
        text = decode(ids)
    except Exception:                                                       # noqa: BLE001
        return None
    if not isinstance(text, str):
        return None
    if isinstance(stop_reason, str) and stop_reason:
        cut = text.find(stop_reason)
        if cut >= 0:
            text = text[:cut]
    return text


def recover_server_parser_text(*, content: Optional[str], reasoning: Optional[str],
                               tool_calls: Any, thinking: bool,
                               completion_tokens: Optional[int],
                               generation_text: Optional[str] = None) -> ParserRecovery:
    """The server-parser policy for one reply (see SERVER_PARSER_MOVED_TEXT above).

    `thinking` is the request's effective enable_thinking; `generation_text` is the reply's
    generation rebuilt from its token ids (generation_text_from_token_ids), None when the server
    returned no token ids or they could not be decoded.

    * moved text (server_parser_signal): the model's output is `generation_text` when it holds
      text, else the reasoning field followed by the content;
    * dropped text: the model's output is `generation_text` when it holds text; a generation that
      decodes to nothing but whitespace is an empty reply (the model's, no event); with no
      `generation_text` the reply is SERVER_PARSER_TEXT_LOST, a harness error;
    * anything else: no event, the reply is left as it is.
    Pure (no I/O): suites with their own model client apply the same function."""
    signal = server_parser_signal(content=content, reasoning=reasoning, tool_calls=tool_calls,
                                  thinking=thinking, completion_tokens=completion_tokens)
    if signal is None:
        return ParserRecovery(None, None, None)
    have_generation = isinstance(generation_text, str) and bool(generation_text.strip())
    if signal == SERVER_PARSER_MOVED_TEXT:
        if have_generation:
            return ParserRecovery(SERVER_PARSER_MOVED_TEXT, generation_text, "token_ids")
        return ParserRecovery(SERVER_PARSER_MOVED_TEXT, str(reasoning or "") + str(content or ""),
                              "reasoning_then_content")
    if have_generation:
        return ParserRecovery(SERVER_PARSER_DROPPED_TEXT, generation_text, "token_ids")
    if isinstance(generation_text, str):
        return ParserRecovery(None, None, None)
    return ParserRecovery(SERVER_PARSER_TEXT_LOST, None, None)


#: Markup a parser removes around the segments it returns: a tag with no whitespace inside
#: (`<|...|>`, `<...>`), the form special and added tokens of chat and reasoning templates take.
_PARSER_MARKUP_RE = re.compile(r"<[^<>\s]{1,64}>")


def server_parser_residue(*, content: Optional[str], reasoning: Optional[str], tool_calls: Any,
                          thinking: bool, generation_text: Optional[str]) -> ParserRecovery:
    """The server-parser policy's check for text a tool-call parser deleted from a reply that
    still carries content or reasoning (server_parser_signal sees neither case).

    Two shapes, both from a request that declared no tools: a reply sent with thinking off whose
    content is the text BEFORE a native tool call the parser lifted out and dropped (content
    'Let me look that up.', the call gone), and a --thinking reply whose reasoning holds the
    thought and whose answer was a native tool call the parser dropped (content empty). Rule:
    locate the returned reasoning, then the returned content, verbatim in `generation_text` (the
    reply's generation rebuilt from its token ids, special tokens kept); the text after the last
    located segment is the tail. When the tail holds a letter or digit outside tag markup
    (_PARSER_MARKUP_RE), the parser removed generated text: SERVER_PARSER_DROPPED_TEXT, and the
    model's output is the generation after the reasoning on a --thinking reply (the reasoning
    stays the reasoning), else the whole generation. A segment that cannot be located, a tail of
    markup and whitespace only, a reply with bound tool calls, or no generation text: no event.
    Pure (no I/O)."""
    if tool_calls or not isinstance(generation_text, str) or not generation_text.strip():
        return ParserRecovery(None, None, None)
    gen = generation_text
    r = str(reasoning or "").strip()
    c = str(content or "").strip()
    if not r and not c:
        return ParserRecovery(None, None, None)
    pos = 0
    reasoning_end = 0
    if r:
        at = gen.find(r)
        if at < 0:
            return ParserRecovery(None, None, None)
        pos = reasoning_end = at + len(r)
    if c:
        at = gen.find(c, pos)
        if at < 0:
            return ParserRecovery(None, None, None)
        pos = at + len(c)
    tail = gen[pos:]
    visible = _PARSER_MARKUP_RE.sub("", tail)
    if not any(ch.isalnum() for ch in visible):
        return ParserRecovery(None, None, None)
    if thinking and r:
        return ParserRecovery(SERVER_PARSER_DROPPED_TEXT, gen[reasoning_end:], "token_ids")
    return ParserRecovery(SERVER_PARSER_DROPPED_TEXT, gen, "token_ids")


#: The detail a SERVER_PARSER_TEXT_LOST harness error carries.
SERVER_PARSER_TEXT_LOST_DETAIL = (
    "the server generated {n} token(s) and returned no content, no reasoning and no tool call "
    "(its tool-call parser deleted them: a native tool call on a request that declared no tools), "
    "and the text could not be recovered: the reply carried no token_ids (the request asks for "
    "them with return_token_ids) or they could not be decoded (no /detokenize endpoint and no "
    "run tokenizer)")

#: Servers whose /detokenize endpoint is absent: not asked again.
_NO_DETOKENIZE_ENDPOINT: set = set()


def _detokenize_with_server(ids: List[int]) -> Optional[str]:
    """`ids` decoded by the served model's own tokenizer (vLLM's /detokenize keeps special
    tokens), or None when the server has no such endpoint or the call fails."""
    url, model = get_run_knob("tokenize_url"), get_run_knob("served_model")
    if not url or not model or url in _NO_DETOKENIZE_ENDPOINT:
        return None
    try:
        data = json.dumps({"model": model, "tokens": list(ids)}).encode("utf-8")
        req = urllib.request.Request(url.rstrip("/") + "/detokenize", data=data,
                                     headers={"Content-Type": "application/json",
                                              **endpoint_auth_headers()})
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")
        except Exception:                                                   # noqa: BLE001
            detail = ""
        if e.code in (405, 501) or (e.code == 404 and "does not exist" not in detail):
            _NO_DETOKENIZE_ENDPOINT.add(url)
        return None
    except Exception:                                                       # noqa: BLE001
        return None
    text = payload.get("prompt") if isinstance(payload, dict) else None
    if not isinstance(text, str):
        _NO_DETOKENIZE_ENDPOINT.add(url)
        return None
    return text


def decode_generated_token_ids(ids: List[int]) -> Optional[str]:
    """Generated token ids as text with special tokens kept: the server's /detokenize (the
    served tokenizer), else the run's tokenizer (--tokenizer, else the model's HF id); None when
    neither can decode them."""
    text = _detokenize_with_server(ids)
    if text is not None:
        return text
    tok = _run_tokenizer()
    if tok is None:
        return None
    try:
        with _TOKENIZER_LOCK:
            out = tok.decode(list(ids), skip_special_tokens=False)
    except Exception:                                                       # noqa: BLE001
        return None
    return out if isinstance(out, str) else None


def server_parser_note(counts: Dict[str, int], sources: Optional[Dict[str, int]] = None) -> str:
    """The sentence naming a run's server-parser events (`counts` by event, `sources` by
    recovered_from), as base puts it in `server_parser_note`."""
    parts = []
    moved = counts.get(SERVER_PARSER_MOVED_TEXT, 0)
    dropped = counts.get(SERVER_PARSER_DROPPED_TEXT, 0)
    lost = counts.get(SERVER_PARSER_TEXT_LOST, 0)
    if moved:
        parts.append(f"{SERVER_PARSER_MOVED_TEXT} {moved} (the server's reasoning parser put text "
                     "of a reply sent with thinking off in its reasoning field; recovered and "
                     "scored as the model's output)")
    if dropped:
        parts.append(f"{SERVER_PARSER_DROPPED_TEXT} {dropped} (the server's tool-call parser "
                     "deleted a reply with no tools declared; recovered from its token ids and "
                     "scored as the model's output)")
    if lost:
        parts.append(f"{SERVER_PARSER_TEXT_LOST} {lost} (deleted by the server's tool-call parser "
                     "and not recoverable: harness errors, left out of the score)")
    note = "server-parser events: " + "; ".join(parts)
    if sources:
        note += " [recovered from: " + ", ".join(
            f"{k} {v}" for k, v in sorted(sources.items())) + "]"
    return note

#: Set to 0 to disable the one-shot recovery of a forced-final turn whose answer the
#: tool-call parser discarded.
RETRY_DISCARDED_FINAL = os.environ.get("GBENCH_RETRY_DISCARDED_FINAL", "1") != "0"


async def _retry_discarded_final(*, reply, convo, session, api_url, model_name, payload,
                                 semaphore, budget, temperature):
    """Re-ask once when a forced-final turn came back `output_discarded`.

    The final turn withdraws `tools` so the model has to commit to an answer. gemma-4
    answers that by emitting a tool call anyway, terminated by `<|tool_response>` (token 50,
    a configured stop token); vLLM's parser lifts the call out of `content`, finds no
    `tools` on the request to bind it to, and drops it - so a turn that generated real
    tokens returns `content=null`. Nothing downstream can recover it, and the sample is
    graded as though the model said nothing.

    Returns the recovered reply, or the original if recovery is off, unnecessary or failed
    (never raises, and never returns something emptier than what it was handed).
    """
    # A forced-final reply whose tool call the parser deleted is asked again whether or not the
    # server-parser policy recovered the deleted text (SERVER_PARSER_DROPPED_TEXT: the recovered
    # text is the tool call itself, not an answer).
    if not RETRY_DISCARDED_FINAL or (classify_reply(reply) != "output_discarded"
                                     and reply.parser_event not in (SERVER_PARSER_DROPPED_TEXT,
                                                                    SERVER_PARSER_TEXT_LOST)):
        return reply
    convo.append({"role": "user", "content": (
        "That response could not be read because it was a tool call. Tools are disabled "
        "and no tool call can succeed. Reply with plain text only - no function call, no "
        "tool syntax - stating your final answer.")})
    final_max, oversize = await clamp_to_context_async(
        convo, budget, (payload or {}).get("tools"),
        effective_chat_template_kwargs((payload or {}).get("chat_template_kwargs"), False))
    if oversize:
        return reply
    try:
        retry = await _send_single_request(
            session=session, api_url=api_url, model_name=model_name, messages=convo,
            extra_payload=payload, semaphore=semaphore, pbar=None,
            max_output_tokens=final_max if budget is not None else None,
            temperature=temperature, thinking=False)
    except Exception as e:                                                  # noqa: BLE001
        logger.debug("discarded-final retry failed: %s", e)
        return reply
    return retry if (retry.text or "").strip() else reply


#: Run-level generation knobs, set once by the runner and consulted by `run_eval_suite`
#: for anything a suite did not pass explicitly.
#:
#: `evals.py` hands every suite the same kwargs dict, but a suite only honours a knob if
#: it forwards it - and 107 of 122 never forwarded `temperature`, so `--temperature`
#: (documented as applying "across all benchmarks") silently did nothing for them.
#: Resolving here means a knob cannot be lost by omission; a suite that
#: passes a value explicitly still wins.
_RUN_KNOBS: Dict[str, Any] = {}

#: Minimum output-token budget for EVERY suite, and the per-suite exceptions to it.
#:
#: Set from measurement, not from judgement about which benchmark is "long-form". Across
#: 32 suites measured on 2026-08-17, the largest *healthy* completion - one that finished
#: on its own with no degenerate repetition - was 8,020 tokens (hmmt). Second 5,730
#: (livebench), third 5,239 (putnam); everything else under 3,600. Every *looping*
#: response, by contrast, ran 10,035-56,201 tokens. The two populations do not overlap.
#:
#: 16384 is therefore 2x the largest genuine answer ever observed and below every observed
#: loop: it cannot truncate a real answer, and it caps a spiral at a quarter of what a
#: 65536 budget costs. A per-suite table encoding "this one writes patches, that one writes
#: letters" was tried and got hmmt, codeforces and putnam wrong.
#:
#: EXCEPTIONS go in the dict below, and only with a measurement: if a suite genuinely needs
#: more, the run reports `genuinely_truncated` (distinct from `looping`), which is the
#: signal to raise it. Nothing is here yet because nothing has earned it - the 65536 tier
#: previously rested on a 114,292-char copilot_bench_swe response recorded before
#: repetition detection existed, so it was never established whether that was a patch or a
#: loop.
DEFAULT_MIN_OUTPUT_TOKENS = 16384

#: Floor when `--thinking` is on. A reasoning trace and the answer share one `max_tokens`,
#: so the non-thinking floor truncates mid-reasoning: on the 2026-08-18 smoke run `aime`
#: lost 3/3 responses to the cap and scored 0%, which is a measurement artefact rather than
#: a model result. Canonical AIME budgets are 32,768 (DeepSeek-R1, open-r1, Evalchemy) to
#: 64K (R1-0528); 32,768 is the common floor.
#:
#: This exists because `_run_suite_async` already had a `32768 if thinking` default that
#: could never fire: `run_eval_suite` applies the floor FIRST, so `max_output_tokens` was
#: never None by the time that branch was reached. Keep the two in agreement.
#:
#: Deliberately a plain constant, like DEFAULT_MIN_OUTPUT_TOKENS above: the value follows
#: from `--thinking`, and `--max-output-tokens` already overrides both. A third knob would
#: only add a way for the budget to disagree with the mode it was chosen for.
THINKING_MIN_OUTPUT_TOKENS = 32768
SUITE_MIN_OUTPUT_TOKENS: Dict[str, int] = {}

#: Ceiling on the per-turn budget for suites that run a TOOL LOOP - the mirror image of the
#: floors above, and the only place a floor is not what a suite needs.
#:
#: `max_output_tokens` is sent on EVERY turn, so in a loop it does not bound the run, it
#: licenses that much output per turn. Measured on browsecomp 2026-08-18 at the operator's
#: `--max-output-tokens 65536`: 3/3 samples spent the entire 65,536 in one turn enumerating
#: candidates ("Maybe the movie is "The Blind Side"? No." on repeat) and never emitted an
#: answer. Forcing `tool_choice=required` was tried first and REFUTED the obvious
#: explanation: the model then made 5/2/4 real searches, got real grounded results, and
#: still burned 70k+ tokens per sample without answering. The turn length is the problem,
#: not tool availability.
#:
#: Re-running the same 3 questions at 8192/turn: 12 clean search rounds with ZERO truncated
#: turns on the sample that had managed one, 13.1k tokens/sample (5.4x less) and 257s
#: (4.0x faster). 4096 was also tried and is too tight - the model cannot finish reasoning
#: and emit the call, so every turn truncates and tool calls drop from 12/0/4 to 2/1/0.
#:
#: This is a no-op for every other tool-loop suite: the largest turn any of them produced
#: on the same run was mcp_atlas at 3,014 tokens (gaia 1,063; other tool-loop suites were
#: lower), so the ceiling sits 2.7x above the observed maximum. It deliberately does
#: NOT apply to single-shot suites, where a long turn IS the answer (aime spends 31,252).
TOOL_LOOP_MAX_OUTPUT_TOKENS = 8192


def tool_loop_turn_ceiling(has_tool_loop: bool, max_output_tokens: Optional[int]) -> Optional[int]:
    """The capped per-turn budget, or None to leave `max_output_tokens` alone."""
    if not has_tool_loop or not max_output_tokens:
        return None
    ceiling = TOOL_LOOP_MAX_OUTPUT_TOKENS
    raw = os.environ.get("GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS")
    if raw is not None:
        try:
            ceiling = max(0, int(raw))
        except ValueError:
            logger.warning(
                "GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS=%r is not an integer; ignoring", raw)
    if not ceiling or max_output_tokens <= ceiling:
        return None
    return ceiling


class ToolLoopProtocol(NamedTuple):
    """How one suite's tool loop runs, taken from that suite's upstream reference setup.

    Opt-in: a suite that passes none keeps the house loop (`max_tool_rounds`, the
    TOOL_LOOP_MAX_OUTPUT_TOKENS per-turn ceiling, gbench's recovery turns, rendered calls
    echoed into the transcript), so existing suites and plugins are unaffected. The fields
    are recorded on the result as `tool_loop_protocol`.
    """
    #: Where the settings come from (upstream file + commit), recorded verbatim.
    reference: str
    #: Tool-enabled model turns before the loop stops (each turn's calls are executed); then
    #: the forced final turn runs, or the loop stops when `forced_final` is False. None = no
    #: cap (the context window still ends it).
    max_rounds: Optional[int] = 8
    #: Tool executions per sample before the loop stops; None = no cap.
    max_tool_calls: Optional[int] = None
    #: Apply TOOL_LOOP_MAX_OUTPUT_TOKENS to every turn. False = each turn gets the run's
    #: own budget (--max-output-tokens, else the thinking/non-thinking floor).
    house_turn_ceiling: bool = True
    #: gbench's recovery turns: a forced final answer (tools withdrawn) when the round cap is
    #: hit or a turn ends inside the reasoning channel, plus the discarded-final retry.
    #: False = the loop just stops and the answer is the last non-empty assistant content
    #: of the transcript, the way scaleapi/mcp-atlas run_eval.py extracts it.
    forced_final: bool = True
    #: Echo `Reply.text` (content prefixed with gbench's text rendering of the calls) into
    #: the assistant turn sent back to the model. False = send the model's own content only,
    #: as an OpenAI-style agent loop does.
    rendered_calls_in_history: bool = True
    #: A tool call whose `arguments` string is not valid JSON. False = parse it to {} and call
    #: the tool (the house loop). True = do not call the tool; the tool message is
    #: "Error: <first line of the JSON parse error>", as an agent loop that feeds the parse
    #: failure back to the model does. The call still counts toward `max_tool_calls`.
    invalid_arguments_as_error: bool = False
    #: Wall-clock limit per sample in seconds, measured from the moment the sample's first
    #: request gets a request slot; time the sample spends waiting for a slot behind other
    #: samples does not count. When it expires the sample stops: its trace has status
    #: WALL_CLOCK_LIMIT and scores 0, and the result counts it under `wall_clock_limited`.
    #: The stopped sample always stays in the denominator: a batch scorer's exclusion markers
    #: (scoring_excluded, a judge outage, empty_is_answer) on it are set aside, so a suite
    #: whose official harness excludes some tasks whatever happens must not mark those tasks
    #: in a scorer that runs after a wall-clock stop. None = no limit.
    max_sample_wall_s: Optional[float] = None

    def record(self, per_turn_max_tokens: Optional[int]) -> Dict[str, Any]:
        rec = self._asdict()
        rec["per_turn_max_tokens"] = per_turn_max_tokens
        return rec


def _rendered_tool_call_prefix(tool_calls: List[Dict[str, Any]]) -> str:
    """The text `_send_single_request` puts in front of `content` when a reply has tool calls.

    Must stay byte-identical to the rendering there; tests/test_agentic_tool_suites_canonical.py runs a
    reply through `_send_single_request` to hold the two together.
    """
    call_strs = []
    for tc in tool_calls:
        fn = tc.get("function", {})
        fname = fn.get("name", "")
        fargs_str = fn.get("arguments", "{}")
        py_kwargs_clean = ""
        try:
            args_dict = json.loads(fargs_str) if isinstance(fargs_str, str) else fargs_str
            if isinstance(args_dict, dict):
                py_kwargs_clean = fname + "(" + ", ".join(f"{k}={v}" for k, v in args_dict.items()) + ")"
        except Exception:
            py_kwargs_clean = ""
        call_strs.append(f"{fname}({fargs_str}) {py_kwargs_clean}")
    return "\n".join(call_strs) + "\n"


def reply_content(reply: "Reply") -> str:
    """The model's own message content, without gbench's text rendering of its tool calls."""
    text = reply.text or ""
    if not reply.tool_calls:
        return text
    prefix = _rendered_tool_call_prefix(reply.tool_calls)
    if text.startswith(prefix):
        return text[len(prefix):]
    logger.warning("tool-call rendering changed in _send_single_request; reply_content cannot "
                   "strip it, so the rendered calls stay in the content")
    return text


def set_run_knobs(**knobs: Any) -> None:
    """Record run-level generation knobs (called by the runner, once per suite)."""
    _RUN_KNOBS.update({k: v for k, v in knobs.items() if v is not None})


#: Executors are `(name, args)`. One that also accepts a third parameter is handed the
#: sample index, so a stateful environment can keep per-sample state instead of sharing it
#: across concurrently-running samples. Cached because it is consulted per tool call, keyed on
#: the function object itself (weakly), never on id(): a freed per-run closure's id can be
#: reused by another executor with a different arity.
_EXECUTOR_ARITY: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _executor_arity_wants_session(fn: Callable) -> bool:
    try:
        import inspect
        params = list(inspect.signature(fn).parameters.values())
        positional = [p for p in params
                      if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        return len(positional) >= 3
    except (TypeError, ValueError):
        return False


def _executor_wants_session(fn: Callable) -> bool:
    try:
        cached = _EXECUTOR_ARITY.get(fn)
    except TypeError:                      # not weakly referenceable: computed every time
        return _executor_arity_wants_session(fn)
    if cached is None:
        cached = _executor_arity_wants_session(fn)
        _EXECUTOR_ARITY[fn] = cached
    return cached


def get_run_knob(name: str, default: Any = None) -> Any:
    return _RUN_KNOBS.get(name, default)


#: Default sampling temperature for EVERY suite that declares no official sampling.
#:
#: A fixed project baseline, the same for every served model: it is not read from the served
#: model's generation_config (suites whose upstream prescribes the model's own sampling, e.g.
#: hmmt, aime and ojbench, read that file themselves). gbench used to send 0.0 on every request.
#:
#: The reason it changed is measured, not doctrinal. Replaying the 2026-08-17
#: copilot_bench_swe prompts, 4 attempts each, only the temperature differing:
#:
#:     prompt set          T=0.0        T=1.0     T=1.0 +top_p .95/top_k 64
#:     10 that looped      31/40 (78%)  16/40 (40%)  15/40 (38%)
#:     10 that did not     10/40 (25%)  -            3/40  (8%)
#:
#: Paired per-prompt, 13 of 14 discordant prompts improved (one-sided sign test
#: p = 0.0009), 0 of 10 loopers got worse, and the healthy control improved too - so the
#: reduction is not bought by degrading answers that already worked. `top_k`/`top_p` add
#: nothing detectable on top of the temperature (40% vs 38% at n=40).
#:
#: Caveats worth keeping in view: 38% residual looping means this is a mitigation, not a
#: cure; and the experiment measured DEGENERATION, not accuracy - a suite whose canonical
#: protocol is greedy (BigCodeBench, IFEval, RULER, the ANLS/relaxed-accuracy vision
#: suites) is no longer leaderboard-comparable at 1.0. Set `--temperature 0.0`, or the
#: per-suite env var below, to restore it.
#:
#: The resolved DEFAULT is THINK-AWARE (project baseline): a no-think run defaults to
#: `NOTHINK_TEMPERATURE` (0.0 - greedy, canonical-comparable) and a `--thinking` run to
#: `DEFAULT_TEMPERATURE` (1.0; the reasoning pass makes looping far less of a concern than the
#: 0.0 no-think figure above). Both replace every suite's official temperature
#: (resolve_temperature: the maintainer's rule). Both are overridden by
#: `--temperature` or the per-suite env var. NOTE the tradeoff: a no-think run at 0.0
#: carries the higher degenerate-repetition rate measured above - surfaced by the
#: looping/non-convergent metrics, not silently absorbed.
DEFAULT_TEMPERATURE = 1.0     #: --thinking default (the project baseline above)
NOTHINK_TEMPERATURE = 0.0     #: no-think default (greedy; canonical-comparable)

#: Grading temperature for the LLM judge. Deliberately NOT tied to DEFAULT_TEMPERATURE or
#: to `--temperature`: those describe the model under test, and a grader that drifts with
#: the run's sampling knob would make two runs of the same answers disagree.
#:
#: Until 2026-08-17 all 15 judge call sites passed no config at all, so grading ran at the
#: Gemini API default of 1.0 - an undocumented source of run-to-run variance sitting on top
#: of every judge-scored suite.
JUDGE_TEMPERATURE = 0.0


def judge_config(**overrides: Any) -> Any:
    """`GenerateContentConfig` for judge calls: deterministic grading.

    Returns None if google-genai is unavailable, which callers pass through harmlessly
    (`config=None` is the SDK's own default).
    """
    try:
        from google.genai import types
    except ImportError:
        return None
    return types.GenerateContentConfig(temperature=JUDGE_TEMPERATURE, **overrides)


# --------------------------------------------------------------------------- #
# The Gemini model cascade. Quota is per model (a 429 on one Gemini version can be a 200 on
# another), so every Gemini text role tries each model in order, moves to the next on a
# failure, backs off and retries the whole cascade on a synchronized all-model failure, and
# only when EVERY model fails EVERY round declares a JUDGE_OUTAGE (excluded from accuracy, a
# harness error) rather than scoring the sample wrong. search_tool.search_with_fallback runs
# the same cascade for Google Search grounding.
# --------------------------------------------------------------------------- #
#: THE default Gemini model cascade, the one source of truth for every Gemini text role in
#: gbench: LLM judges, graders and raters, labelers, user simulators (tau2 / tau3 user
#: simulator and NL judge, acebench, toolbench's API simulator), web search / grounding, and
#: the judges that run inside suite containers (gaia2, wildclawbench, mcp_bench, toolbench).
#: Tried in order. Code that cannot import gbench (a container, search_tool) receives the list
#: from gbench at run time as GBENCH_JUDGE_MODELS (gemini_cascade_env()); a default baked into
#: such a copy must equal this tuple, and tests/test_gemini_cascade.py checks every copy.
#: Non-text roles keep their own model: tau3's retrieval embeddings (gemini-embedding-001) and
#: the quality pillar's pinned GemmaClaw helper models.
GEMINI_CASCADE: Tuple[str, ...] = ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash",
                                   "gemini-3.5-flash", "gemini-3.1-pro-preview")
#: The knob that replaces the cascade (comma separated, tried in order).
JUDGE_MODELS_ENV = "GBENCH_JUDGE_MODELS"
#: A one-model cascade, read only when JUDGE_MODELS_ENV is unset or empty.
JUDGE_MODEL_ENV = "GBENCH_JUDGE_MODEL"
#: Search / grounding only: replaces the cascade for web search (default: judge_cascade()).
SEARCH_MODELS_ENV = "GBENCH_SEARCH_MODELS"
#: A one-model search cascade, read only when SEARCH_MODELS_ENV is unset or empty.
SEARCH_MODEL_ENV = "GBENCH_SEARCH_MODEL"
_JUDGE_BACKOFF = float(os.environ.get("GBENCH_JUDGE_BACKOFF", "1.0"))


def _model_list(raw: Optional[str]) -> List[str]:
    """The model names of a comma separated knob value, in order ([] for none)."""
    return [m.strip() for m in (raw or "").split(",") if m.strip()]


def judge_cascade() -> List[str]:
    """The ordered Gemini models every Gemini text role tries: GBENCH_JUDGE_MODELS (comma
    separated) when it names a model, else GBENCH_JUDGE_MODEL (one model), else GEMINI_CASCADE.
    Read at call time."""
    models = _model_list(os.environ.get(JUDGE_MODELS_ENV))
    if models:
        return models
    single = (os.environ.get(JUDGE_MODEL_ENV) or "").strip()
    if single:
        return [single]
    return list(GEMINI_CASCADE)


def judge_cascade_source() -> str:
    """Where judge_cascade() came from: "GBENCH_JUDGE_MODELS", "GBENCH_JUDGE_MODEL" or
    "default" (GEMINI_CASCADE)."""
    if _model_list(os.environ.get(JUDGE_MODELS_ENV)):
        return JUDGE_MODELS_ENV
    if (os.environ.get(JUDGE_MODEL_ENV) or "").strip():
        return JUDGE_MODEL_ENV
    return "default"


def search_cascade() -> List[str]:
    """The ordered Gemini models web search / grounding tries: GBENCH_SEARCH_MODELS (comma
    separated) when it names a model, else GBENCH_SEARCH_MODEL (one model), else judge_cascade().
    search_tool._search_cascade applies the same rule in code that cannot import gbench."""
    models = _model_list(os.environ.get(SEARCH_MODELS_ENV))
    if models:
        return models
    single = (os.environ.get(SEARCH_MODEL_ENV) or "").strip()
    if single:
        return [single]
    return judge_cascade()


def search_cascade_source() -> str:
    """Where search_cascade() came from: "GBENCH_SEARCH_MODELS", "GBENCH_SEARCH_MODEL", or
    judge_cascade_source()."""
    if _model_list(os.environ.get(SEARCH_MODELS_ENV)):
        return SEARCH_MODELS_ENV
    if (os.environ.get(SEARCH_MODEL_ENV) or "").strip():
        return SEARCH_MODEL_ENV
    return judge_cascade_source()


def gemini_cascade_env() -> Dict[str, str]:
    """The variables that hand the run's cascade to a container or a subprocess:
    {"GBENCH_JUDGE_MODELS": <judge_cascade() comma separated>, "GBENCH_SEARCH_MODELS":
    <search_cascade() comma separated>}. Pass them to every process that makes a Gemini call
    and cannot import gbench (in-container judges and simulators, search_tool, the tau image),
    so it calls exactly the models gbench records; a default baked into it is then never used."""
    return {JUDGE_MODELS_ENV: ",".join(judge_cascade()),
            SEARCH_MODELS_ENV: ",".join(search_cascade())}


#: Deprecated alias of the cascade's head at import time, kept for code outside this module
#: that still imports it. Call judge_cascade() (every model, read at call time) instead.
DEFAULT_JUDGE_MODEL = judge_cascade()[0]


def _judge_cascade_rounds() -> int:
    """How many times to retry the WHOLE cascade when every model fails one pass (burst)."""
    return max(1, int(os.environ.get("GBENCH_JUDGE_CASCADE_ROUNDS", "3")))


def gemini_api_keys() -> List[str]:
    """All configured Gemini API keys, in order, de-duplicated.

    GEMINI_API_KEYS wins when both it and GEMINI_API_KEY are set (gbench/__init__.py applies the
    same rule at import); otherwise GEMINI_API_KEY is read. Either may hold one key or a comma OR
    semicolon separated list, so a key LIST works uniformly for the judge, the liveness gate, and
    search grounding - not just search. A single key returns a 1-element list.
    """
    def _split(raw: Optional[str]) -> List[str]:
        return [k.strip() for k in re.split(r"[;,]", raw or "") if k.strip()]

    keys = _split(os.environ.get("GEMINI_API_KEYS")) or _split(os.environ.get("GEMINI_API_KEY"))
    return list(dict.fromkeys(keys))


_JUDGE_CLIENTS = None
_JUDGE_CLIENT_CYCLE = None

#: Seconds one Gemini judge request may take (google-genai HttpOptions.timeout). The client had
#: none, so a judge request the API never answered held its sample, and the cascade behind it,
#: forever (2026-10-10 investigation). A request past it is asked once more of the SAME model
#: (_judge_request), and only then does the cascade move to its next model, recording the reason
#: (record_judge_fallback, judge_fallback_counts). It bounds the judge only, never the model under
#: test. GBENCH_JUDGE_REQUEST_TIMEOUT_S replaces it.
JUDGE_REQUEST_TIMEOUT_S = float(os.environ.get("GBENCH_JUDGE_REQUEST_TIMEOUT_S", "600"))
#: Attempts of one judge request on the same model when it times out, including the first: one
#: retry, then the cascade's next model.
JUDGE_SAME_MODEL_ATTEMPTS = 2


def _judge_http_options() -> Dict[str, Any]:
    """The google-genai HttpOptions of every judge client: the Gemini API's base URL and the
    request timeout (JUDGE_REQUEST_TIMEOUT_S, in the milliseconds google-genai takes; it also
    sends it as X-Server-Timeout). No retry_options: google-genai's async path (aiohttp, when
    installed) raises asyncio.TimeoutError, which its retry rule does not match (checked against
    a slow local server, 2026-10-11), so the same-model retry is _judge_request's."""
    return {"base_url": GEMINI_API_BASE_URL, "timeout": int(JUDGE_REQUEST_TIMEOUT_S * 1000)}


def _is_judge_timeout(exc: BaseException) -> bool:
    """A judge request that ran out of time: asyncio's / the builtin TimeoutError (google-genai's
    aiohttp path), an httpx or aiohttp timeout class, or HTTP 408."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or getattr(exc, "code", None) == 408:
        return True
    return "Timeout" in type(exc).__name__


#: Why gbench's judge cascade left a model in this process, {"<model>: <reason>": n}: the
#: exception class of the failure (after the same-model retry of a timeout), "same-model retry
#: after timeout" for each timeout asked again. judge_fallback_counts() reads it.
_JUDGE_FALLBACKS: Dict[str, int] = {}


def record_judge_fallback(model: str, reason: str) -> None:
    """Count one judge fallback (`_JUDGE_FALLBACKS`)."""
    key = f"{model}: {reason}"
    _JUDGE_FALLBACKS[key] = _JUDGE_FALLBACKS.get(key, 0) + 1


def judge_fallback_counts() -> Dict[str, int]:
    """A copy of the judge fallbacks so far in this process, {"<model>: <reason>": n}."""
    return dict(_JUDGE_FALLBACKS)


async def _judge_request(client: Any, model: str, contents: Any, cfg: Any) -> Any:
    """One judge request to `model`, asked again of the same model once when it times out
    (JUDGE_SAME_MODEL_ATTEMPTS; recorded as "same-model retry after timeout"); any other error,
    or a second timeout, is raised for the cascade to move on."""
    for attempt in range(JUDGE_SAME_MODEL_ATTEMPTS):
        try:
            return await client.aio.models.generate_content(model=model, contents=contents,
                                                            config=cfg)
        except Exception as e:  # noqa: BLE001 - re-raised unless a timeout with a retry left
            if attempt + 1 >= JUDGE_SAME_MODEL_ATTEMPTS or not _is_judge_timeout(e):
                raise
            record_judge_fallback(model, "same-model retry after timeout")
            logger.debug("judge model %s timed out after %ss; asking it once more", model,
                         JUDGE_REQUEST_TIMEOUT_S)
    raise RuntimeError("unreachable")


def judge_client_record() -> Dict[str, Any]:
    """The judge client's transport settings, as a result may record them."""
    return {"client": "google-genai Client (Gemini API)",
            "timeout_s": JUDGE_REQUEST_TIMEOUT_S,
            "same_model_attempts": JUDGE_SAME_MODEL_ATTEMPTS,
            "retry_on": "timeout (asyncio/httpx/aiohttp timeout, HTTP 408)",
            "source": "base.JUDGE_REQUEST_TIMEOUT_S (GBENCH_JUDGE_REQUEST_TIMEOUT_S)"}


def _judge_clients():
    """Lazily-built google-genai clients, ONE per configured key, cached. Rotated across judge calls
    (see `_next_judge_client`) so a multi-key `GEMINI_API_KEY` spreads per-key rate limits instead of
    hammering (and failing on) a single key."""
    global _JUDGE_CLIENTS, _JUDGE_CLIENT_CYCLE
    if _JUDGE_CLIENTS is None:
        from google import genai
        # Always the Gemini API: an explicit base URL and vertexai=False, so google-genai's own
        # GOOGLE_GEMINI_BASE_URL / GOOGLE_GENAI_USE_VERTEXAI variables cannot send the key or the
        # judge's calls to another host or service.
        # A bounded request with one retry on the same model (_judge_http_options).
        _JUDGE_CLIENTS = [genai.Client(api_key=k, vertexai=False,
                                       http_options=_judge_http_options())
                          for k in gemini_api_keys()]
        _JUDGE_CLIENT_CYCLE = itertools.cycle(_JUDGE_CLIENTS) if _JUDGE_CLIENTS else None
    return _JUDGE_CLIENTS


def _next_judge_client():
    """The next judge client in round-robin order (spreads load across keys), or None if no key."""
    _judge_clients()
    return next(_JUDGE_CLIENT_CYCLE) if _JUDGE_CLIENT_CYCLE else None


def _finish_reason_name(resp: Any) -> Optional[str]:
    """The finish reason of a Gemini response's first candidate (e.g. "MAX_TOKENS"), or None."""
    try:
        candidates = getattr(resp, "candidates", None) or []
        reason = getattr(candidates[0], "finish_reason", None) if candidates else None
    except Exception:  # noqa: BLE001 - an odd response object has no readable reason
        return None
    if reason is None:
        return None
    name = getattr(reason, "name", None)
    return str(name if name is not None else reason)


#: Gemini finish reasons that mean the reply was blocked (a refusal of the prompt), not cut off.
_BLOCK_FINISH_REASONS = frozenset({"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII",
                                   "RECITATION", "IMAGE_SAFETY", "OTHER"})


def _block_reason(resp: Any) -> Optional[str]:
    """Why Gemini blocked a reply, or None: "prompt_feedback:<reason>" when the prompt was blocked
    (prompt_feedback.block_reason), else "finish_reason:<reason>" when the candidate stopped for
    one of _BLOCK_FINISH_REASONS."""
    block = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
    if block:
        name = getattr(block, "name", None)
        return f"prompt_feedback:{name if name is not None else block}"
    reason = _finish_reason_name(resp)
    if reason in _BLOCK_FINISH_REASONS:
        return f"finish_reason:{reason}"
    return None


async def judge_generate_cascade_detailed(contents: Any, *, config: Any = None
                                          ) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """judge_generate_cascade, plus what the cascade saw: `(text, model_used, info)`.

    `info["finish_reason"]` is the finish reason of the reply returned (e.g. "MAX_TOKENS" for a
    reply cut off at the config's max_output_tokens; None when nothing was returned), and
    `info["max_tokens_without_text"]` counts the replies of the cascade that stopped at
    max_output_tokens with no text (the cascade then asks its next model, or ends in
    JUDGE_OUTAGE), so a caller can count judge replies its token limit cut off.

    Gemini refusals: `info["block_reasons"]` lists (sorted) every block reason a model gave
    instead of text (_block_reason), and `info["refused"]` is True when, in one round, every
    model asked answered without text because of a block. A refusal is a property of the
    prompt, so the cascade then stops without retrying further rounds. The call still returns
    (None, "JUDGE_OUTAGE", info); a caller that tells a refusal from an outage reads
    `info["refused"]` / `info["block_reasons"]`.

    `config` may be a callable (model name -> config) to send each cascade model its own
    generation config; None means judge_config()."""
    info: Dict[str, Any] = {"finish_reason": None, "max_tokens_without_text": 0,
                            "refused": False, "block_reasons": []}
    if not gemini_api_keys():
        return None, "JUDGE_OUTAGE", info
    try:
        if not _judge_clients():
            return None, "JUDGE_OUTAGE", info
    except ImportError:
        return None, "JUDGE_OUTAGE", info
    cfg = config if config is not None else judge_config()
    cascade = judge_cascade()
    rounds = _judge_cascade_rounds()
    not_served: set = set()
    blocks: set = set()
    for rnd in range(rounds):
        client = _next_judge_client()   # rotate key per round + across calls -> spreads rate limits
        asked = blocked = 0
        for model in cascade:
            if model in not_served:
                continue
            asked += 1
            try:
                resp = await _judge_request(client, model, contents,
                                            cfg=cfg(model) if callable(cfg) else cfg)
                text = getattr(resp, "text", None)
                reason = _finish_reason_name(resp)
                if text is not None:
                    info["finish_reason"] = reason
                    info["block_reasons"] = sorted(blocks)
                    record_gemini_answer(model)
                    return text, model, info
                if reason == "MAX_TOKENS":
                    info["max_tokens_without_text"] += 1
                block = _block_reason(resp)
                if block:
                    blocks.add(block)
                    blocked += 1
            except Exception as e:  # noqa: BLE001 - 429/network/timeout: move to next model
                # why the cascade left this model (a timeout was already retried once on it)
                code = getattr(e, "code", None)
                record_judge_fallback(model, type(e).__name__
                                      + (f" (HTTP {code})" if isinstance(code, int) else ""))
                if getattr(e, "code", None) == 404:
                    # A model the API does not serve never answers: not asked again in this call.
                    not_served.add(model)
                    _warn_not_served(model)
                logger.debug("judge model %s failed: %s", model, e)
        info["block_reasons"] = sorted(blocks)
        if asked and blocked == asked:
            # Every model refused the prompt: retrying the same prompt is refused again.
            info["refused"] = True
            break
        if len(not_served) >= len(cascade):
            break
        if rnd < rounds - 1:  # every model failed this pass -> burst; back off and retry all
            await asyncio.sleep(_JUDGE_BACKOFF * (2 ** rnd) + random.uniform(0, _JUDGE_BACKOFF))
    return None, "JUDGE_OUTAGE", info


#: Replies each Gemini model gave through judge_generate_cascade in this process, {model: n}.
#: run_eval_suite records the replies of its own run as `gemini_models_answered`.
_GEMINI_ANSWERED: Dict[str, int] = {}
_NOT_SERVED_WARNED: set = set()


def record_gemini_answer(model: str, n: int = 1) -> None:
    """Count a reply `model` gave (judge_generate_cascade does; a suite with its own Gemini call
    may too), so the run records which Gemini models actually answered."""
    _GEMINI_ANSWERED[model] = _GEMINI_ANSWERED.get(model, 0) + int(n)


def gemini_answer_counts() -> Dict[str, int]:
    """A copy of the reply counts per Gemini model so far in this process."""
    return dict(_GEMINI_ANSWERED)


def _warn_not_served(model: str) -> None:
    if model not in _NOT_SERVED_WARNED:
        _NOT_SERVED_WARNED.add(model)
        logger.warning("[judge] Gemini model '%s' answered HTTP 404 (not served for this key); "
                       "the cascade skips it. Check %s / %s.", model, JUDGE_MODELS_ENV,
                       JUDGE_MODEL_ENV)


async def judge_generate_cascade(contents: Any, *, config: Any = None) -> Tuple[Optional[str], str]:
    """Grade `contents` via the Gemini judge cascade.

    Returns `(text, model_used)` from the first model that answers, or `(None, "JUDGE_OUTAGE")`
    when every model fails every round. On the latter the caller MUST set
    `trace["judge_grade"] = "JUDGE_OUTAGE"` (so base.run_eval_suite excludes it from the
    accuracy denominator) rather than scoring the sample wrong. Grading stays deterministic
    (JUDGE_TEMPERATURE) via `judge_config()`. judge_generate_cascade_detailed also reports the
    finish reasons the cascade saw.
    """
    text, model, _info = await judge_generate_cascade_detailed(contents, config=config)
    return text, model


def temperature_env_var(eval_name: str) -> str:
    """Per-suite temperature override, e.g. `GBENCH_COPILOT_BENCH_SWE_TEMPERATURE`."""
    slug = re.sub(r"[^A-Z0-9]+", "_", str(eval_name).upper()).strip("_")
    return f"GBENCH_{slug}_TEMPERATURE"


#: Renamed built-in suites, old name -> new name. The registry resolves an old name to the new
#: suite with a deprecation warning (it is not listed in --help, --list or `--evals all`), and
#: the old suite's `GBENCH_<OLD>_TEMPERATURE` is still read as a deprecated alias. Kept here,
#: not in the package registry, because resolve_temperature needs it and base imports no suite.
RENAMED_SUITES: Dict[str, str] = {
    "semantic_keypoint": "screenspot_pointing",
    "lmsys_noncoding_hard": "wildbench",
    "bfcl_v3_live": "bfcl_live",
    "loft_x_arxiv": "loft_scifact",
    "culer": "longbench_v2_code",
    "nexus_function_calling": "nexusraven_api_eval",
}

#: Deregistered suites that are planned in docs/evals_roadmap.md, name -> roadmap entry: suites
#: with no implementation yet. They are not in the registry (no `--evals all`, no --list); naming
#: one on the command line is an error that points at its roadmap entry instead of an
#: unknown-suite error. An implemented suite whose prerequisite a host lacks is registered and
#: hard-errors infra_required instead (see evals.NAMED_ONLY_SUITES).
ROADMAP_SUITES: Dict[str, str] = {
    "lmarena_web_agent": "#1",
    "cybergym": "#4",
}


def roadmap_suite_error(name: str) -> str:
    """The error for `--evals <name>` when `name` is a deregistered roadmap suite."""
    return (f"eval suite '{name}' is deregistered: it is planned in docs/evals_roadmap.md "
            f"(entry {ROADMAP_SUITES[name]}) and returns as a runnable suite only when its "
            f"canonical harness lands.")


def legacy_temperature_env_vars(eval_name: str) -> Tuple[str, ...]:
    """The pre-rename temperature variables a renamed suite still reads (deprecated)."""
    return tuple(temperature_env_var(old) for old, new in RENAMED_SUITES.items()
                 if new == eval_name)


def gbench_cache_dir() -> str:
    """gbench's cache root, as an absolute path: `$GBENCH_CACHE_DIR` (`~` expanded, a relative
    value taken from the working directory), else ~/.cache/gbench. Absolute because suites
    bind-mount paths under it into containers, and docker reads a relative `-v` source as a
    volume name. Read at call time, so every suite cache under it follows the variable (tests
    point it at a temporary directory)."""
    value = (os.environ.get("GBENCH_CACHE_DIR")
             or os.path.join(os.path.expanduser("~"), ".cache", "gbench"))
    return os.path.abspath(os.path.expanduser(value))


#: The served endpoint's key, in the order endpoint_api_key reads them: VLLM_API_KEY (the documented
#: name), then OPENAI_API_KEY (the name OpenAI-compatible harnesses read; a documented alias).
ENDPOINT_KEY_ENVS = ("VLLM_API_KEY", "OPENAI_API_KEY")


def endpoint_api_key(environ: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The served endpoint's API key, or None: VLLM_API_KEY, else OPENAI_API_KEY, skipping an unset
    variable and a placeholder value (EMPTY, dummy, ...: PLACEHOLDER_SECRET_VALUES). The key is sent
    only from gbench's own process (base's requests, the endpoint probes, gbench's relays), never
    into a process or container that runs model-written code or the model's tool calls."""
    env = os.environ if environ is None else environ
    for name in ENDPOINT_KEY_ENVS:
        value = (env.get(name) or "").strip()
        if value and value.lower() not in PLACEHOLDER_SECRET_VALUES:
            return value
    return None


def endpoint_auth_headers(environ: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """{"Authorization": "Bearer <key>"} for the served endpoint (endpoint_api_key), or {}."""
    key = endpoint_api_key(environ)
    return {"Authorization": f"Bearer {key}"} if key else {}


# --------------------------------------------------------------------------- #
# Shard identity. `--shard I/N` (the CLI validates it and exports GBENCH_SHARD; an operator may
# also export GBENCH_SHARD per node) selects shard I of N of every suite's full, deterministically
# ordered item list, round-robin (sampling.shard_select), BEFORE the --eval-limit draw: a limited
# sharded run draws up to --eval-limit items from its own shard, so N shards run up to
# N x --eval-limit items. A malformed GBENCH_SHARD is an error (ValueError), never a silent full run.
# --------------------------------------------------------------------------- #
class ShardSpec(NamedTuple):
    """The run's shard: 1-based `index` of `count`, and its label "I/N"."""
    index: int
    count: int

    @property
    def label(self) -> str:
        return f"{self.index}/{self.count}"

    @property
    def partial(self) -> bool:
        """True when the shard is not the whole set (count > 1)."""
        return self.count > 1


def current_shard(environ: Optional[Dict[str, str]] = None) -> Optional[ShardSpec]:
    """The run's shard from GBENCH_SHARD, or None when it is unset or empty. A malformed value
    (not I/N with 1 <= I <= N) raises ValueError."""
    env = os.environ if environ is None else environ
    parsed = parse_shard(env.get("GBENCH_SHARD"))
    return ShardSpec(*parsed) if parsed else None


def shard_identity(shard: Optional[ShardSpec]) -> Dict[str, Any]:
    """The fields a result covering `shard` carries: {"shard": "I/N", "shard_index": I,
    "shard_count": N, "shard_applied": True}; {} for no shard."""
    if shard is None:
        return {}
    return {"shard": shard.label, "shard_index": shard.index, "shard_count": shard.count,
            "shard_applied": True}


#: The sample-meta keys that identify a sample (sample_key reads them) or group it (category):
#: bookkeeping, never request fields, so run_eval_suite does not forward them to the server.
#: Every other key of a sample's meta dict goes into its request body (a per-sample max_tokens,
#: tools, logprobs, ...), after the suite-level extra_payload.
SAMPLE_META_BOOKKEEPING_KEYS = frozenset({"category", "instance_id", "task_id", "question_id", "id",
                                          "uid", "sample_id", "problem_id", "name"})


def sample_key(sample: Any) -> str:
    """A shard-independent id for one sample, the same on every node and in every shard.

    The sample's own id when its metadata (a dict, or the third element of a
    (messages, gold, meta) tuple) carries one of instance_id, task_id, question_id, id, uid,
    sample_id, problem_id or name; else "sha256:<16 hex>" of the canonical JSON of the sample
    (messages and gold included, bulky media data URIs replaced by their length)."""
    meta = sample
    if isinstance(sample, (list, tuple)) and len(sample) > 2:
        meta = sample[2]
    if isinstance(meta, dict):
        for field_name in ("instance_id", "task_id", "question_id", "id", "uid", "sample_id",
                           "problem_id", "name"):
            value = meta.get(field_name)
            if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value):
                return f"{field_name}:{value}"

    def _plain(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): _plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_plain(v) for v in obj]
        if isinstance(obj, str) and len(obj) > 256 and obj.startswith("data:"):
            return f"<data uri, {len(obj)} chars>"
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return repr(obj)

    blob = json.dumps(_plain(sample), sort_keys=True, ensure_ascii=True, default=str)
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def apply_shard_indexed(items: List[Any], eval_name: str = ""
                        ) -> Tuple[List[Any], Optional[str], List[int]]:
    """apply_shard, plus each selected item's position in the full ordered list:
    (this shard's items, "I/N" or None, their global indices). For a trace's `global_idx`."""
    shard = current_shard()
    if not shard:
        return list(items), None, list(range(len(items)))
    positions = shard_select(list(range(len(items))), (shard.index, shard.count))
    selected = [items[i] for i in positions]
    logger.info("[%s] shard %d/%d: %d of %d items", eval_name or "eval", shard.index,
                shard.count, len(selected), len(items))
    return selected, shard.label, positions


#: The comparability reason a result of a partial shard carries.
SHARD_REASON = ("sharded run (--shard {label}): shard {index} of {count}, a partial set (the "
                "full-set score needs every shard merged)")


def add_comparability_reason(result: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Mark `result` not leaderboard-comparable for `reason`: leaderboard_comparable=False, and the
    reason appended to leaderboard_comparable_reason ("; "-joined, each reason once)."""
    reasons = [r for r in str(result.get("leaderboard_comparable_reason") or "").split("; ") if r]
    if reason not in reasons:
        reasons.append(reason)
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = "; ".join(reasons)
    return result


def finalize_shard_fields(result: Dict[str, Any]) -> Dict[str, Any]:
    """The central shard stamp for any suite's result (native or not), applied after the suite
    returned (evals.py). With no shard active (or 1/1): nothing changes. With a partial shard:
      * a result that says which shard it covers (`shard` set: run_eval_suite, apply_shard +
        record_shard, swe_agentic.select_tasks, ...) gets shard_index / shard_count /
        shard_applied=True and the SHARD_REASON comparability reason;
      * a result without `shard` ran the FULL set on this node (the suite cannot shard):
        shard_applied=False and shard_requested="I/N", with a warning; its own comparability is
        left as the suite reported it (a full-set result is not a partial one).
    A result whose `shard` differs from GBENCH_SHARD keeps the suite's value."""
    shard = current_shard()
    if shard is None or not shard.partial:
        return result
    if result.get("shard"):
        idx = parse_shard(str(result["shard"]))
        spec = ShardSpec(*idx) if idx else shard
        result.update({"shard_index": spec.index, "shard_count": spec.count,
                       "shard_applied": True})
        if spec.partial:
            add_comparability_reason(result, SHARD_REASON.format(
                label=spec.label, index=spec.index, count=spec.count))
    else:
        result["shard_applied"] = False
        result["shard_requested"] = shard.label
        logger.warning("[%s] --shard %s was not applied: this suite ran its full set on this "
                       "node (shard_applied=false).", result.get("eval_name") or "eval",
                       shard.label)
    return result


def finalize_central_stamps(result: Dict[str, Any], model_under_test: Optional[str]
                            ) -> Dict[str, Any]:
    """The stamps the eval runner applies to every suite's result after the suite returned, so a
    suite that rewrites leaderboard_comparable after run_eval_suite cannot drop them:
      * finalize_shard_fields (shard identity and the partial-shard comparability reason);
      * the Gemini judge provenance (judge_provenance) for a suite whose Gemini gate ran or whose
        result says a Gemini model answered, and the self-judge comparability reason when the
        cascade holds the model under test."""
    finalize_shard_fields(result)
    name = result.get("eval_name")
    used_gemini = bool(result.get("gemini_models_answered")) or (name in _GEMINI_GATED)
    if used_gemini:
        if "self_judge" not in result:
            result.update(judge_provenance(model_under_test))
        if result.get("self_judge"):
            add_comparability_reason(result, SELF_JUDGE_REASON)
    return result


# --------------------------------------------------------------------------- #
# Endpoint rejections (HTTP 400/422) and context-window rejections: one rule for every suite.
#   * Where the benchmark's own harness defines the outcome of such a rejection, its semantics
#     win (for example an agent loop whose history the model grew past the window fails the task,
#     as upstream's loop does): the model's outcome, recorded with the upstream reference.
#   * Otherwise a 400/422 on a request gbench (or a harness gbench drives) built is a harness
#     error: counted, named, left out of the score, never the model's outcome.
# attribute_endpoint_rejection() applies it; ENDPOINT_REJECTION_ATTRIBUTION lists what each suite
# does (docs/evals/env-knobs.md repeats it).
# --------------------------------------------------------------------------- #
#: The HTTP statuses an endpoint answers a request's content with.
CONTENT_REJECTION_STATUSES = (400, 422)
REJECTION_MODEL_OUTCOME = "model_outcome"
REJECTION_HARNESS_ERROR = "harness_error"


def is_context_window_rejection(status: Optional[int], body: Any) -> bool:
    """True when an endpoint's HTTP 400/422 says the prompt (or prompt + max_tokens) exceeds its
    context window (vLLM, OpenAI and LiteLLM wordings)."""
    if status not in CONTENT_REJECTION_STATUSES:
        return False
    text = str(body or "").lower()
    return any(k in text for k in ("maximum context length", "context length", "context_length",
                                   "context window", "input tokens", "prompt is too long",
                                   "too many tokens"))


def attribute_endpoint_rejection(status: Optional[int], body: Any = "", *,
                                 upstream_outcome: Optional[str] = None,
                                 upstream_reference: Optional[str] = None) -> Dict[str, Any]:
    """The attribution of one endpoint rejection under the uniform rule.

    `upstream_outcome` is what the benchmark's own harness makes of this rejection when it
    defines it (e.g. "task failed: context exceeded"), with `upstream_reference` naming where;
    None when the harness defines nothing (or gbench built the request). Returns {"kind":
    "over_context" or "endpoint_rejected", "status", "attribution": "model_outcome" (upstream's
    semantics) or "harness_error", "rule": the sentence that decided it, "upstream": the
    reference or None}."""
    kind = "over_context" if is_context_window_rejection(status, body) else "endpoint_rejected"
    if upstream_outcome:
        return {"kind": kind, "status": status, "attribution": REJECTION_MODEL_OUTCOME,
                "rule": f"the benchmark's harness defines it: {upstream_outcome}",
                "upstream": upstream_reference}
    return {"kind": kind, "status": status, "attribution": REJECTION_HARNESS_ERROR,
            "rule": "a rejection of a request the harness built is a harness error",
            "upstream": None}


#: What each suite does with an endpoint's HTTP 400/422 and context-window rejections under the
#: rule above: "harness_error" (the rule's default), or "upstream: ..." naming the harness
#: semantics that decide instead. A suite not listed runs through run_eval_suite (base): every
#: 400/422 is a harness error (endpoint_failure, or over_context for a context-window rejection).
ENDPOINT_REJECTION_ATTRIBUTION: Dict[str, str] = {
    "bfcl": "harness_error (ENDPOINT_FAILURE_KINDS in the multi-turn and agentic parts)",
    "bfcl_live": "harness_error",
    "bfcl_v4_agentic": "harness_error (ENDPOINT_FAILURE_KINDS)",
    "copilot_bench_swe": "harness_error (swebench_common.ENDPOINT_EXCEPTIONS)",
    "swe_bench_multilingual": "harness_error (swebench_common.ENDPOINT_EXCEPTIONS)",
    "swe_bench_live": ("harness_error for a 400/422 (swebench_common.ENDPOINT_EXCEPTIONS); "
                       "upstream: SWE-agent's exit_context ends the task when the history the "
                       "model grew exceeds the window"),
    "multi_swe_bench": "harness_error (swebench_common.ENDPOINT_EXCEPTIONS)",
    "swe_bench_pro": "harness_error (the endpoint failures gbench's relay counts)",
    "terminal_bench": ("harness_error for a 400/422 (swebench_common.ENDPOINT_EXCEPTIONS); "
                       "upstream: Terminus-2's ContextLengthExceededError handling for a history "
                       "the model grew past the window"),
    "mcp_bench": ("harness_error (model_replies.failed_requests: every endpoint or connection "
                  "error of a model request); upstream: MCP-Bench's own handling of a rejected "
                  "over-context request (rejected_over_context)"),
    "gaia2": ("upstream: ARE's agent loop takes the rejection of a request carrying the model's "
              "own conversation into the agent's history and goes on, so a 400/422 or a "
              "context-window rejection is the model's outcome; a run in which no agent request "
              "got a reply is infra_required"),
    "toolbench": ("upstream: StableToolBench's chat_completion_request turns the rejection into an "
                  "error reply the DFSDT loop goes on with (model_content_rejections)"),
    "quality": ("upstream: GemmaClaw's agent loop gets the rejection as the reply to its request "
                "(runners/quality.py _MODEL_CONTENT_REJECTION_STATUSES)"),
    "scicode": ("harness_error for a prompt built from the dataset alone (over_context when the "
                "window cannot hold it); upstream: a sub-step prompt that carries the model's own "
                "earlier code and exceeds the window fails that step and the steps after it "
                "(gencode.py has no handling; test_generated_code.py's fixed denominators), the "
                "model's outcome"),
    "mobilegym": ("upstream: HTTP 400/422 and context-window rejections on multi-turn GUI "
                  "episodes are scored as 0.0 model outcomes; HTTP 5xx/429/401/403/404 and "
                  "timeouts are harness_error"),
    "mobilegym_pp": ("upstream: HTTP 400/422 and context-window rejections on multi-turn hybrid "
                     "episodes are scored as 0.0 model outcomes; HTTP 5xx/429/401/403/404 and "
                     "timeouts are harness_error"),
}


# --------------------------------------------------------------------------- #
# Run-shaping environment knobs: every env variable that changes WHAT a run measures (the dataset,
# the protocol or prompt, the decoding, or the scoring). The evals scaffold snapshots the ones that
# are set (run_shaping_env), so two runs that differ in one get different scaffold ids.
# Classes: "dataset" (which items / split / version), "prompt" (protocol, prompt or mode),
# "decode" (sampling and budgets), "scoring" (the judge and what it sees).
# --------------------------------------------------------------------------- #
RUN_SHAPING_GLOBAL_ENV: Dict[str, str] = {
    "GBENCH_REPETITION_PENALTY": "decode",
    "GBENCH_FREQUENCY_PENALTY": "decode",
    "GBENCH_PRESENCE_PENALTY": "decode",
    "GBENCH_IMAGE_SOFT_TOKENS": "decode",
    "GBENCH_AUDIO_SOFT_TOKENS": "decode",
    "GBENCH_VIDEO_SOFT_TOKENS": "decode",
    "GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS": "decode",
    "GBENCH_RETRY_DISCARDED_FINAL": "prompt",
    "GBENCH_JUDGE_MODELS": "scoring",
    "GBENCH_JUDGE_MODEL": "scoring",
    "GBENCH_SEARCH_MODELS": "scoring",
    "GBENCH_SEARCH_MODEL": "scoring",
    # search_tool: how many results a web_search call returns to the model
    "GBENCH_SEARCH_MAX_RESULTS": "prompt",
    # agent suites: the run's --max-output-tokens on agent requests that carry none (on unless 0)
    "GBENCH_AGENT_MAX_TOKENS_FROM_RUN": "decode",
    # EndpointRelay: stream from the endpoint and rebuild the reply (replies can differ)
    "GBENCH_RELAY_STREAM_UPSTREAM": "decode",
}

#: Per-suite run-shaping knobs, {suite: {ENV: class}}. GBENCH_<SUITE>_TEMPERATURE is covered by
#: pattern for every suite (temperature_env_var) and needs no entry.
RUN_SHAPING_SUITE_ENV: Dict[str, Dict[str, str]] = {
    "copilot_bench_swe": {"GBENCH_COPILOT_BENCH_SWE_MODE": "prompt"},
    "swe_bench_live": {"GBENCH_SWE_BENCH_LIVE_MODE": "prompt"},
    "swe_bench_multilingual": {"GBENCH_SWE_BENCH_MULTILINGUAL_MODE": "prompt"},
    "multi_swe_bench": {"GBENCH_MULTI_SWE_BENCH_MODE": "prompt"},
    "swe_bench_pro": {"GBENCH_SWE_BENCH_PRO_MODE": "prompt", "GBENCH_SWEBENCH_PRO_AGENTIC": "prompt"},
    "gaia": {"GBENCH_GAIA_SPLIT": "dataset"},
    # GBENCH_GAIA2_MAX_CONCURRENT: how many scenarios load the endpoint at once, and the Time
    # capability's clock advances by the measured generation time
    "gaia2": {"GBENCH_GAIA2_CONFIG": "dataset", "GBENCH_GAIA2_MODE": "prompt",
              "GBENCH_GAIA2_NUM_RUNS": "dataset", "GBENCH_GAIA2_LIMIT": "dataset",
              "GBENCH_GAIA2_MAX_CONCURRENT": "decode", "GBENCH_GAIA2_SCENARIO_TIMEOUT": "prompt"},
    "lcb": {"GBENCH_LCB_VERSION": "dataset", "GBENCH_LCB_MIN_DATE": "dataset",
            "GBENCH_LCB_INCLUDE_TEST_GEN": "dataset", "GBENCH_LCB_INCLUDE_SELF_REPAIR": "dataset"},
    "bigcodebench": {"GBENCH_BIGCODEBENCH_SUBSET": "dataset"},
    "livebench": {"GBENCH_LIVEBENCH_RELEASE": "dataset", "GBENCH_LIVEBENCH_CATEGORIES": "dataset",
                  "GBENCH_LIVEBENCH_MODEL_CONFIG": "decode", "GBENCH_LIVEBENCH_MODEL": "decode",
                  "GBENCH_LIVEBENCH_EXTRA_ARGS": "decode"},
    "aider_polyglot": {"GBENCH_AIDER_LANGUAGES": "dataset", "GBENCH_AIDER_EDIT_FORMAT": "prompt",
                       "GBENCH_AIDER_SETTINGS": "decode", "GBENCH_AIDER_MODEL": "prompt",
                       "GBENCH_AIDER_EXTRA_ARGS": "prompt"},
    "multipl_e": {"GBENCH_MULTIPL_E_LANGS": "dataset", "GBENCH_MULTIPL_E_SAMPLES": "dataset"},
    "codeforces": {"GBENCH_CODEFORCES_CONFIG": "dataset", "GBENCH_CODEFORCES_SPLIT": "dataset",
                   "GBENCH_CODEFORCES_MAX_GENERATED_TESTS": "scoring",
                   "GBENCH_CODEFORCES_TIME_MULTIPLIER": "scoring"},
    "mmmu_pro": {"GBENCH_MMMU_PRO_PROMPT": "prompt"},
    "ojbench": {"GBENCH_OJBENCH_SAMPLES": "dataset"},
    "ruler": {"GBENCH_RULER_NUM_SAMPLES": "dataset", "GBENCH_RULER_LENGTHS": "dataset",
              "GBENCH_RULER_TOKENIZER": "dataset"},
    "mrcr": {"GBENCH_MRCR_WINDOW": "prompt"},
    "beam_128k": {"GBENCH_BEAM_128K_WINDOW": "prompt"},
    "agent_dojo": {"GBENCH_AGENTDOJO_VERSION": "dataset", "GBENCH_AGENTDOJO_ATTACK": "prompt",
                   "GBENCH_AGENTDOJO_MODEL": "prompt", "GBENCH_AGENTDOJO_MODEL_LABEL": "prompt"},
    "wildclawbench": {"GBENCH_WILDCLAWBENCH_CATEGORIES": "dataset",
                      "GBENCH_WILDCLAWBENCH_TASK": "dataset",
                      "GBENCH_WILDCLAWBENCH_THINKING": "decode"},
    "skillsbench": {"GBENCH_SKILLSBENCH_TASKS": "dataset", "GBENCH_SKILLSBENCH_CONDITIONS": "dataset",
                    "GBENCH_SKILLSBENCH_AGENT": "prompt"},
    "ui_control_osworld": {"GBENCH_OSWORLD_DOMAIN": "dataset", "GBENCH_OSWORLD_MAX_STEPS": "prompt",
                           "GBENCH_OSWORLD_LIMIT": "dataset", "GBENCH_OSWORLD_TEST_META": "dataset",
                           "GBENCH_OSWORLD_OBS_TYPE": "prompt"},
    # tau2 and tau3 share tau_common's knobs (USER_LLM / EVAL_LLM select a Gemini model for the
    # user simulator and the NL judge)
    "tau2": {"GBENCH_TAU2_NUM_TRIALS": "dataset", "GBENCH_TAU2_MAX_STEPS": "prompt",
             "GBENCH_TAU2_MAX_ERRORS": "prompt", "GBENCH_TAU2_SEED": "prompt",
             "GBENCH_TAU2_TOP_P": "decode", "GBENCH_TAU2_TOP_K": "decode",
             "GBENCH_TAU2_USER_TEMPERATURE": "prompt", "GBENCH_TAU2_USER_LLM": "scoring",
             "GBENCH_TAU2_EVAL_LLM": "scoring"},
    "tau3": {"GBENCH_TAU2_NUM_TRIALS": "dataset", "GBENCH_TAU2_MAX_STEPS": "prompt",
             "GBENCH_TAU2_MAX_ERRORS": "prompt", "GBENCH_TAU2_SEED": "prompt",
             "GBENCH_TAU2_TOP_P": "decode", "GBENCH_TAU2_TOP_K": "decode",
             "GBENCH_TAU2_USER_TEMPERATURE": "prompt", "GBENCH_TAU2_USER_LLM": "scoring",
             "GBENCH_TAU2_EVAL_LLM": "scoring"},
    "gdpval": {"GBENCH_GDPVAL_MAX_FILE_CHARS": "scoring", "GBENCH_GDPVAL_MAX_JUDGE_CHARS": "scoring",
               "GBENCH_GDPVAL_MAX_RUBRIC_CHARS": "scoring", "GBENCH_GDPVAL_MAX_PDF_PAGES": "scoring",
               "GBENCH_GDPVAL_PDF_PAGE_DPI": "prompt"},
    "longbench_v2_code": {"GBENCH_LONGBENCH_V2_CODE_WINDOW": "prompt",
                          "GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS": "dataset"},
    "terminal_bench": {"GBENCH_TB_PARSER": "prompt", "GBENCH_TB_TURN_MAX_TOKENS": "decode",
                       "GBENCH_TB_TIMEOUT_MULTIPLIER": "prompt",
                       "GBENCH_TB_CONTAINER_THREADS": "prompt", "GBENCH_TB_SALVAGE_PATCH": "prompt"},
    "toolbench": {"GBENCH_TOOLBENCH_GROUPS": "dataset", "GBENCH_TOOLBENCH_METHOD": "prompt",
                  "GBENCH_TOOLBENCH_EVAL_TIMES": "scoring", "GBENCH_TOOLBENCH_REFERENCE": "scoring",
                  "GBENCH_TOOLBENCH_REFERENCE_DIR": "scoring",
                  "GBENCH_TOOLBENCH_MAX_QUERY_COUNT": "prompt"},
    "scicode": {"GBENCH_SCICODE_WITH_BACKGROUND": "prompt", "GBENCH_SCICODE_EVAL_TIMEOUT_S": "scoring"},
    "complexfuncbench": {"GBENCH_COMPLEXFUNCBENCH_EMBED_MODEL": "scoring",
                         "GBENCH_COMPLEXFUNCBENCH_MAX_ROUNDS": "prompt"},
    "mcp_bench": {"GBENCH_MCP_BENCH_DISTRACTION": "prompt",
                  "GBENCH_MCP_BENCH_DISABLE_STABILITY": "scoring"},
    "putnam_formal": {"GBENCH_PUTNAM_FORMAL_COMPILE_TIMEOUT_S": "scoring",
                      "GBENCH_PUTNAM_FORMAL_CHECK_TIMEOUT_S": "scoring"},
    # bfcl runs bfcl_v4_agentic's generation for its agentic part
    "bfcl": {"GBENCH_BFCL_ENABLE_THINKING": "decode", "GBENCH_BFCL_REUSE_GENERATIONS": "decode"},
    "bfcl_v4_agentic": {"GBENCH_BFCL_ENABLE_THINKING": "decode",
                        "GBENCH_BFCL_REUSE_GENERATIONS": "decode"},
    # a local ScreenSpot-Pro data dir scored instead of the official ScreenSpot-v2 release
    "screenspot": {"GBENCH_SCREENSPOT_DATA_DIR": "dataset"},
}

#: Deprecated bare-prefixed aliases of run-shaping knobs (base.suite_env reads the canonical name,
#: then these): run_shaping_env snapshots an alias under its canonical name when only the alias
#: is set, so the legacy spelling moves the scaffold id too.
RUN_SHAPING_ALIASES: Dict[str, Tuple[str, ...]] = {
    "GBENCH_TAU2_NUM_TRIALS": ("TAU2_NUM_TRIALS",),
    "GBENCH_TAU2_MAX_STEPS": ("TAU2_MAX_STEPS",),
    "GBENCH_TAU2_MAX_ERRORS": ("TAU2_MAX_ERRORS",),
    "GBENCH_TAU2_SEED": ("TAU2_SEED",),
    "GBENCH_TAU2_TOP_P": ("TAU2_TOP_P",),
    "GBENCH_TAU2_TOP_K": ("TAU2_TOP_K",),
    "GBENCH_TAU2_USER_TEMPERATURE": ("TAU2_USER_TEMPERATURE",),
    "GBENCH_TAU2_USER_LLM": ("TAU2_USER_LLM",),
    "GBENCH_TAU2_EVAL_LLM": ("TAU2_EVAL_LLM",),
    "GBENCH_LCB_INCLUDE_TEST_GEN": ("LCB_INCLUDE_TEST_GEN",),
    "GBENCH_BFCL_REUSE_GENERATIONS": ("BFCL_REUSE_GENERATIONS",),
    "GBENCH_SCICODE_EVAL_TIMEOUT_S": ("SCICODE_EVAL_TIMEOUT_S",),
    "GBENCH_PUTNAM_FORMAL_COMPILE_TIMEOUT_S": ("PUTNAM_COMPILE_TIMEOUT_S",),
    "GBENCH_SCREENSPOT_DATA_DIR": ("SCREENSPOT_PRO_DATA_DIR",),
    "GBENCH_LONGBENCH_V2_CODE_WINDOW": ("GBENCH_CULER_WINDOW",),
    "GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS": ("GBENCH_CULER_ALL_DOMAINS", "CULER_ALL_DOMAINS"),
}

#: Deprecated aliases of a suite's GBENCH_<SUITE>_TEMPERATURE that the suite reads at run time
#: (tau_common copies a bare TAU2_TEMPERATURE onto the canonical name of tau2 and of tau3):
#: run_shaping_env snapshots one under the canonical name when only the alias is set.
RUN_SHAPING_TEMPERATURE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "tau2": ("TAU2_TEMPERATURE",),
    "tau3": ("TAU2_TEMPERATURE",),
}

#: Every other GBENCH_* variable the eval_suites modules read, with the reason it does not change
#: what a run measures: "infra" (where a pinned input lives, an image the suite checks by label,
#: a port, host, network or endpoint address, a device, thread or CPU cap, a concurrency, a
#: request or wall-clock timeout whose expiry is a harness error, retries and rate limits of the
#: Gemini roles, the shard, which the run shape records itself), "credential", "diagnostic" (logs,
#: kept work dirs, traces, progress output, health labels), "internal" (names gbench sets for its
#: own child processes and containers, or markers in their output) and "removed" (a removed knob
#: a suite names only to warn about or refuse it). A GBENCH_* name a suite module reads is either
#: run-shaping (above, or its GBENCH_<SUITE>_TEMPERATURE) or listed here; a test enforces it, so a
#: new knob gets a decision when it is added.
NON_RUN_SHAPING_ENV: Dict[str, str] = {
    **{name: "infra" for name in (
        "GBENCH_AIDER_BENCHMARK_DIR", "GBENCH_AIDER_IMAGE", "GBENCH_APIBANK_DIR",
        "GBENCH_BFCL_IMAGE", "GBENCH_BFCL_PROJECT_ROOT", "GBENCH_BFCL_REQUEST_TIMEOUT_S",
        "GBENCH_BFCL_TIMEOUT_S", "GBENCH_BIGCODEBENCH_CPUS", "GBENCH_BIGCODEBENCH_IMAGE",
        "GBENCH_CACHE_DIR", "GBENCH_CODEFORCES_GENERATED_TESTS",
        "GBENCH_COMPLEXFUNCBENCH_EMBED_DEVICE", "GBENCH_COMPLEXFUNCBENCH_RAPIDAPI_TIMEOUT_S",
        "GBENCH_COMPLEXFUNCBENCH_TIMEOUT_S", "GBENCH_EVAL_PLUGINS_PATH",
        "GBENCH_GAIA2_IMAGE", "GBENCH_GAIA2_JUDGE_PORT", "GBENCH_GAIA2_MODEL_ENDPOINT",
        "GBENCH_GAIA2_SKIP_KEY_VALIDATION", "GBENCH_GAIA2_TIMEOUT_S", "GBENCH_JUDGE_BACKOFF",
        "GBENCH_JUDGE_CASCADE_ROUNDS", "GBENCH_LIVEBENCH_IMAGE", "GBENCH_MATHARENA_IMAGE",
        "GBENCH_MCP_ATLAS_IMAGE", "GBENCH_MCP_ATLAS_MONGO_IMAGE", "GBENCH_MCP_ATLAS_SANDBOX_URL",
        "GBENCH_MCP_BENCH_ASSUME_NETWORK", "GBENCH_MCP_BENCH_IMAGE", "GBENCH_MCP_BENCH_NETWORK",
        "GBENCH_MCP_BENCH_TIMEOUT_S", "GBENCH_MIN_DECODE_TOK_S", "GBENCH_MINI_SWE_AGENT_V2_IMAGE",
        "GBENCH_MOBILEGYM_IMAGE", "GBENCH_MOBILEGYM_SIMULATOR_PORT", "GBENCH_MOBILEGYM_TIMEOUT",
        "GBENCH_MOBILEGYM_WORKERS",
        "GBENCH_MULTIPL_E_IMAGE", "GBENCH_MULTI_SWE_BENCH_AGENT_IMAGE", "GBENCH_MULTI_SWE_REPO_DIR",
        "GBENCH_NESTFUL_FUNC_DIR", "GBENCH_OJBENCH_IMAGE", "GBENCH_OJBENCH_JUDGE_TIMEOUT_S",
        "GBENCH_OJBENCH_TESTDATA", "GBENCH_OMNIDOCBENCH_IMAGE", "GBENCH_OSWORLD_MODEL_ENDPOINT",
        "GBENCH_OSWORLD_VM_DIR", "GBENCH_PREREQS_DIR", "GBENCH_PUTNAM_FORMAL_LEAN_IMAGE",
        "GBENCH_PUTNAM_FORMAL_LEAN_PROJECT", "GBENCH_REQUEST_TIMEOUT_S", "GBENCH_RULER_DIR",
        "GBENCH_SANDBOX", "GBENCH_SCICODE_HTTP_TIMEOUT_S", "GBENCH_SCICODE_TEST_DATA",
        "GBENCH_SCICODE_TEST_DATA_REPO", "GBENCH_SCICODE_THREADS", "GBENCH_SEARCH_ATTEMPTS",
        "GBENCH_SEARCH_BACKOFF", "GBENCH_SEARCH_CASCADE_ROUNDS", "GBENCH_SEARCH_HEDGE_DELAY_S",
        "GBENCH_SEARCH_INITIAL_WIDTH", "GBENCH_SEARCH_REDIRECT_TIMEOUT_S",
        "GBENCH_SEARCH_MAX_BURST", "GBENCH_SEARCH_MAX_RPS", "GBENCH_SHARD",
        "GBENCH_SKILLSBENCH_IMAGE",
        "GBENCH_SKILLSBENCH_TASK_ENDPOINT", "GBENCH_SKILLSBENCH_TASK_HOST",
        "GBENCH_SKILLSBENCH_TIMEOUT_S", "GBENCH_SPIDER2_GOLD_DIR", "GBENCH_SPIDER2_LOCALDB_DIR",
        "GBENCH_SPIDER2_REPO_DIR", "GBENCH_SWEBENCH_V5_IMAGE", "GBENCH_SWE_BENCH_LIVE_AGENT_IMAGE",
        "GBENCH_SWE_BENCH_LIVE_HARNESS_DIR", "GBENCH_SWE_BENCH_LIVE_IMAGE",
        "GBENCH_SWE_BENCH_LIVE_WIN_AGENT_DIR", "GBENCH_SWE_BENCH_PRO_HARNESS_DIR",
        "GBENCH_SWE_BENCH_PRO_IMAGE", "GBENCH_SWE_BENCH_PRO_RAW_SAMPLE",
        "GBENCH_SWE_BENCH_PRO_TASK_ENDPOINT", "GBENCH_SWE_BENCH_PRO_TASK_HOST",
        "GBENCH_SWE_LANCER_IMAGE", "GBENCH_SWE_OMP_THREADS", "GBENCH_T_EVAL_DEVICE",
        "GBENCH_TAU2_BENCH_SRC", "GBENCH_TAU2_IMAGE", "GBENCH_TAU2_LLM_BACKOFF",
        "GBENCH_TAU_SANDBOX_MAX_AGE_H", "GBENCH_TOOLBENCH_IMAGE", "GBENCH_UI_CONTROL_OSWORLD_IMAGE",
        "GBENCH_UI_CONTROL_OSWORLD_TASK_IMAGE", "GBENCH_UI_CONTROL_OSWORLD_TIMEOUT_S",
        "GBENCH_WILDCLAWBENCH_HOST_DIR", "GBENCH_WILDCLAWBENCH_IMAGE",
        "GBENCH_WILDCLAWBENCH_JUDGE_PORT", "GBENCH_WILDCLAWBENCH_JUDGE_REQUEST_TIMEOUT_S",
        "GBENCH_WILDCLAWBENCH_PARALLEL", "GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION",
        "GBENCH_WILDCLAWBENCH_TASK_ENDPOINT", "GBENCH_WILDCLAWBENCH_TASK_HOST",
        "GBENCH_WILDCLAWBENCH_TASK_IMAGE", "GBENCH_WILDCLAWBENCH_TASK_NETWORK",
        "GBENCH_WILDCLAWBENCH_TIMEOUT_S")},
    **{name: "credential" for name in (
        "GBENCH_GAIA2_MODEL_API_KEY", "GBENCH_OSWORLD_MODEL_API_KEY", "GBENCH_SKILLSBENCH_API_KEY",
        "GBENCH_WILDCLAWBENCH_MODEL_API_KEY", "GBENCH_MODEL_API_KEY", "GBENCH_TAU_AGENT_KEY")},
    **{name: "diagnostic" for name in (
        "GBENCH_BFCL_GUARD_LOG", "GBENCH_BFCL_SEARCH_LOG", "GBENCH_GAIA2_KEEP_WORKDIR",
        "GBENCH_MOBILEGYM_SAVE_SCREENSHOTS",
        "GBENCH_NON_CONVERGENT_RATIO", "GBENCH_RESULTS_DIR", "GBENCH_SWE_BENCH_LIVE_TRAJ_DIR",
        "GBENCH_TAU2_PROGRESS_SECS", "GBENCH_TAU2_SAVE_TRACES", "GBENCH_TAU2_VERBOSE",
        "GBENCH_TB_KEEP_JOBS_DIR")},
    **{name: "internal" for name in (
        "GBENCH_AIDER_TOOLCHAIN_CHECK_FAILED", "GBENCH_CHECKER", "GBENCH_CREDENTIAL_PROBE",
        "GBENCH_DETOKENIZE_URL", "GBENCH_DNS_PROBE", "GBENCH_ENDPOINT_PROBE", "GBENCH_EOF",
        "GBENCH_GEMINI_RELAY_URL", "GBENCH_GID", "GBENCH_LIVEBENCH_PATCH_SHA256",
        "GBENCH_LIVEBENCH_QUESTIONS", "GBENCH_MATHARENA_RESULT", "GBENCH_METADATA_PROBE",
        "GBENCH_MOBILEGYM_PP_ROOT", "GBENCH_MOBILEGYM_ROOT",
        "GBENCH_MODEL_BASE_URL", "GBENCH_MODEL_NAME", "GBENCH_OOM_KILLS", "GBENCH_PRINTED",
        "GBENCH_RELAY_PROBE", "GBENCH_TAU2_ENV_RUN", "GBENCH_TRANSLATE_WORKER", "GBENCH_UID",
        "GBENCH_VERDICT", "GBENCH_WORKDIR", "GBENCH_SMOKE", "GBENCH_SMOKE_RELAY_REPLIES",
        "GBENCH_SMOKE_STOP", "GBENCH_POOL_RECORD", "GBENCH_POOL_ACTIVITY_FILE",
        "GBENCH_REUSE_ROOT")},
    # infrastructure limits of the relay, the smoke run, the suite pool and the request clients
    **{name: "infra" for name in (
        "GBENCH_RELAY_IDLE_TIMEOUT_S", "GBENCH_SMOKE_DEADLINE_S", "GBENCH_SMOKE_HANG_S",
        "GBENCH_POOL_DURATIONS_FILE", "GBENCH_POOL_MAX_INFLIGHT", "GBENCH_POOL_STALL_S",
        "GBENCH_PREFLIGHT_MIN_DISK_GB", "GBENCH_PREFLIGHT_TIMEOUT_S",
        "GBENCH_JUDGE_REQUEST_TIMEOUT_S", "GBENCH_MODEL_REQUEST_TIMEOUT_S")},
    **{name: "diagnostic" for name in ("GBENCH_RELAY_EXPECTED_SHAPE",)},
    **{name: "removed" for name in (
        "GBENCH_CYBERSECEVAL_ICD", "GBENCH_GAIA2_JUDGE_ENDPOINT", "GBENCH_GAIA2_JUDGE_MODEL",
        "GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT", "GBENCH_TAU2_EMBED_API_KEY",
        "GBENCH_TAU2_EMBED_BASE_URL", "GBENCH_TAU2_EMBED_BATCH", "GBENCH_TAU2_EMBED_MODEL",
        "GBENCH_TAU2_OPENAI_API_KEY", "GBENCH_TAU2_OPENAI_BASE_URL", "GBENCH_TAU2_RETRIEVAL_CONFIG",
        "GBENCH_TOOLBENCH_SIMULATOR_KEY")},
}


def run_shaping_env(suites: Optional[List[str]] = None,
                    environ: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, str]]:
    """The run-shaping knobs that are SET for `suites` (every registered suite when None), as
    {class: {ENV: value}}: RUN_SHAPING_GLOBAL_ENV, each suite's RUN_SHAPING_SUITE_ENV entries (a
    deprecated alias of RUN_SHAPING_ALIASES under its canonical name when only the alias is set)
    and its GBENCH_<SUITE>_TEMPERATURE (class "decode"). The scaffold records this snapshot."""
    env = os.environ if environ is None else environ
    out: Dict[str, Dict[str, str]] = {}

    def _take(name: str, cls: str, aliases: Tuple[str, ...] = ()) -> None:
        for source in (name, *RUN_SHAPING_ALIASES.get(name, ()), *aliases):
            value = env.get(source)
            if value is not None and str(value).strip() != "":
                out.setdefault(cls, {})[name] = str(value)
                return

    for name, cls in RUN_SHAPING_GLOBAL_ENV.items():
        _take(name, cls)
    names = list(suites) if suites is not None else sorted(RUN_SHAPING_SUITE_ENV)
    for suite in names:
        for name, cls in RUN_SHAPING_SUITE_ENV.get(suite, {}).items():
            _take(name, cls)
        _take(temperature_env_var(suite), "decode",
              RUN_SHAPING_TEMPERATURE_ALIASES.get(suite, ()))
        for old in legacy_temperature_env_vars(suite):
            _take(old, "decode")
    return {cls: dict(sorted(v.items())) for cls, v in sorted(out.items())}


def free_port(preferred: Optional[int] = None) -> int:
    """Return an available localhost TCP port. If `preferred` is given and currently free, return it
    (so a single run stays deterministic); otherwise return an OS-assigned free port. Used so a local
    judge proxy / helper server does not collide with a fixed default when two runs overlap. There is
    a small TOCTOU window (the port could be taken between check and bind), acceptable for avoiding a
    fixed-default clash; an OS-assigned port is effectively never re-handed immediately."""
    if preferred:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", int(preferred)))
                return int(preferred)
            except OSError:
                pass
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_LEGACY_ENV_WARNED: set = set()


def suite_env(canonical: str, *legacy: str, default: Optional[str] = None) -> Optional[str]:
    """Read a per-suite env knob by its canonical ``GBENCH_<SUITE>_<KNOB>`` name, falling back to any
    legacy bare-prefixed aliases (kept for back-compat). Warns once per legacy name when it is used.

    Convention: every per-suite knob is ``GBENCH_<SUITE>_<KNOB>``. Older suites shipped bare-prefixed
    names (e.g. ``TAU2_NUM_TRIALS``, ``SCICODE_EVAL_TIMEOUT_S``); those still work as deprecated
    aliases so nothing breaks, but the canonical name takes precedence and is what the docs and
    ``--help`` advertise. Third-party/standard vars (HF_TOKEN, OPENAI_API_KEY, GEMINI_API_KEY, ...)
    are NOT suite knobs and are read directly, not through this helper.
    """
    v = os.environ.get(canonical)
    if v is not None:
        return v
    for name in legacy:
        v = os.environ.get(name)
        if v is not None:
            if name not in _LEGACY_ENV_WARNED:
                _LEGACY_ENV_WARNED.add(name)
                logging.getLogger(__name__).warning(
                    "env var %s is a deprecated alias; use %s instead.", name, canonical)
            return v
    return default


def resolve_temperature(eval_name: str, suite_value: Optional[float] = None,
                        *, thinking: bool = False) -> Tuple[float, str]:
    """Resolve the temperature for `eval_name`, returning (value, where_it_came_from).

    Precedence, highest first:
      1. `GBENCH_<EVAL>_TEMPERATURE`  - per-suite env override, for tuning one eval
         without disturbing the rest of a sweep. Beats everything, including the CLI.
      2. `--temperature`             - the operator was explicit for this whole run.
      3. the maintainer's rule, for every suite: `DEFAULT_TEMPERATURE` (1.0, source
         "thinking") for a `thinking` run, else `NOTHINK_TEMPERATURE` (0.0, source "default").
         A suite's own or official temperature (`suite_value`) never applies: gbench samples
         every suite the same way by design, knowingly not as the benchmarks' harnesses do (a
         reasoning model sampled at 0.0-0.2 loops in its reasoning until the token budget ends:
         lcb at its official 0.2 looped in 40 of 50 replies on 2026-10-06). The suite's value
         is still recorded (OfficialSampling.record). Pass the run's `thinking` flag.

    Recording the source matters as much as the value: a number produced at 1.0 is not
    comparable with a greedy leaderboard entry, and the result must be able to say so.
    """
    var = temperature_env_var(eval_name)
    raw = (os.environ.get(var) or "").strip()
    if not raw:
        # A renamed suite still honours its old variable, with a one-time warning.
        for old in legacy_temperature_env_vars(eval_name):
            old_raw = (os.environ.get(old) or "").strip()
            if old_raw:
                if old not in _LEGACY_ENV_WARNED:
                    _LEGACY_ENV_WARNED.add(old)
                    logger.warning("env var %s is a deprecated alias; use %s instead.", old, var)
                raw, var = old_raw, old
                break
    if raw:
        try:
            return float(raw), f"env:{var}"
        except ValueError:
            logger.warning("%s=%r is not a number; ignoring it.", var, raw)
    knob = get_run_knob("temperature")
    if knob is not None:
        return float(knob), "cli:--temperature"
    if thinking:
        return DEFAULT_TEMPERATURE, "thinking"
    return NOTHINK_TEMPERATURE, "default"


class OfficialSampling(NamedTuple):
    """A suite's official sampling settings, taken from its upstream harness (opt-in).

    Passed as `run_eval_suite(official_sampling=...)`: the request carries the official `params`
    and max_tokens policy. Its temperature is never sent: every suite samples at the
    maintainer's 0.0 (no-think) / 1.0 (thinking), or the operator's `GBENCH_<SUITE>_TEMPERATURE`
    / `--temperature` (resolve_temperature). The result records `temperature_source` and, under
    `official_sampling`, the official values and whether the run matched them. A suite that
    does not go through run_eval_suite calls resolve_official_sampling and records the same
    dict itself.
    """
    #: Where the settings come from (upstream file + commit), recorded verbatim.
    reference: str
    #: The official temperature; None when the reference does not fix one.
    temperature: Optional[float] = None
    #: False = the official harness sends no temperature, so none is sent and the server's own
    #: default applies (unless the operator sets one).
    send_temperature: bool = True
    #: Other official sampling fields (top_p, top_k, min_p, seed, stop, penalties), sent unless the
    #: suite's extra_payload or a sample sets the same field. Every key must be in
    #: OFFICIAL_SAMPLING_PARAM_KEYS: request fields that are not sampling (tools, max_tokens, n,
    #: chat_template_kwargs, ...) are refused, so they are never recorded as official sampling.
    params: Optional[Dict[str, Any]] = None
    #: False = the official harness sends no max_tokens. No request then carries a max_tokens
    #: (the server's own default applies: the room left in the context window); the prompt is
    #: still counted so one that cannot fit is refused as over-context. The run's
    #: --max-output-tokens and the house floors and per-turn ceiling are not applied, and the
    #: result says so in `max_output_tokens_note`.
    send_max_tokens: bool = True
    #: The official per-request max_tokens (the cap the upstream sampler sends), or None when the
    #: suite declares none. Applied by the token-limit policy (resolve_token_budget): a no-think
    #: request sends exactly this value (no --max-output-tokens, floor or per-turn ceiling); a
    #: --thinking request does not apply it, because an official cap was written for
    #: non-reasoning generation, and is sent the run's own budget instead (recorded as "official
    #: cap not applied to reasoning"). Explicit reasoning-model settings go in `reasoning`.
    max_tokens: Optional[int] = None
    #: The upstream's explicit reasoning-model settings (for example OJBench's 64k max_tokens
    #: for reasoning-oriented models), used instead of these fields for a --thinking
    #: run. Its own `max_tokens` is applied to reasoning, never below the run's
    #: --max-output-tokens. None = the upstream defines no reasoning-model settings.
    reasoning: Optional["OfficialSampling"] = None

    def for_mode(self, thinking: bool) -> "OfficialSampling":
        """The settings a request of this mode uses: `reasoning` for a --thinking run when the
        upstream defines reasoning-model settings, else these."""
        return self.reasoning if (thinking and self.reasoning is not None) else self

    def record(self, temperature: Optional[float], source: str) -> Dict[str, Any]:
        """The `official_sampling` block of a result that ran at `temperature` from `source`."""
        if not self.send_temperature:
            matches: Optional[bool] = temperature is None
        elif self.temperature is None:
            matches = None
        else:
            matches = temperature is not None and abs(float(temperature)
                                                      - float(self.temperature)) < 1e-9
        rec = {"reference": self.reference, "temperature": self.temperature,
               "send_temperature": self.send_temperature, "params": dict(self.params or {}),
               "temperature_source": source, "temperature_matches_official": matches}
        if not self.send_max_tokens:
            rec["send_max_tokens"] = False
            rec["max_tokens"] = "not sent (server default: rest of the context window)"
        elif self.max_tokens is not None:
            rec["max_tokens"] = int(self.max_tokens)
        return rec


#: The request fields OfficialSampling.params may carry: the sampling and decoding controls, a
#: subset of ALLOWED_API_KEYS (the temperature has its own fields).
OFFICIAL_SAMPLING_PARAM_KEYS = frozenset({
    "top_p", "top_k", "min_p", "seed", "stop", "presence_penalty", "frequency_penalty",
    "repetition_penalty", "logit_bias",
})


def check_official_sampling_params(eval_name: str, official: OfficialSampling) -> Dict[str, Any]:
    """`official.params` as a dict, or ValueError when a key is not a sampling field
    (OFFICIAL_SAMPLING_PARAM_KEYS): tools, max_tokens, n, chat_template_kwargs and the like are
    never recorded as official sampling. The official cap has its own field
    (OfficialSampling.max_tokens), checked here too, in both modes."""
    for mode in (official, official.reasoning):
        if mode is None:
            continue
        cap = mode.max_tokens
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0):
            raise ValueError(f"[{eval_name}] OfficialSampling.max_tokens must be a positive "
                             f"integer, got {cap!r}")
        if cap is not None and not mode.send_max_tokens:
            raise ValueError(f"[{eval_name}] OfficialSampling sets max_tokens={cap} and "
                             "send_max_tokens=False; an official harness either sends a cap or "
                             "sends none")
        if mode.reasoning is not None and mode is official.reasoning:
            raise ValueError(f"[{eval_name}] OfficialSampling.reasoning cannot nest another "
                             "reasoning block")
    params = dict(official.params or {})
    unknown = sorted(k for k in params if k not in OFFICIAL_SAMPLING_PARAM_KEYS)
    if official.reasoning is not None:
        unknown += sorted(k for k in (official.reasoning.params or {})
                          if k not in OFFICIAL_SAMPLING_PARAM_KEYS and k not in unknown)
    if unknown:
        raise ValueError(f"[{eval_name}] official sampling params {unknown} are not sampling "
                         "fields (set the temperature with OfficialSampling.temperature and the "
                         "cap with OfficialSampling.max_tokens; other keys must be in "
                         "base.OFFICIAL_SAMPLING_PARAM_KEYS)")
    return params


class TokenBudget(NamedTuple):
    """The max_tokens a suite's requests carry, under the token-limit policy.

    `policy` is one of
      * "house": the suite declares no official cap for this mode; --max-output-tokens, the
        floors and the tool-loop per-turn ceiling apply as for any suite (`max_tokens` None:
        the caller keeps its own resolution);
      * "official": a no-think run of a suite with an official cap; exactly that cap is sent;
      * "not_sent": the official harness (or its reasoning-model sampler) sends no max_tokens,
        so none is sent;
      * "reasoning_run_budget": a --thinking run of a suite whose official cap was written for
        non-reasoning generation; the cap is not applied and the run's budget is sent (its
        --max-output-tokens, else the thinking floor);
      * "official_reasoning": a --thinking run with the upstream's explicit reasoning-model
        cap; that cap is sent, or --max-output-tokens when it is larger (a thinking run is never
        capped below it). record() reports the cap as applied only when the cap itself is sent.
    """
    policy: str
    #: The per-request max_tokens to send; None = send none ("not_sent"), or the caller's own
    #: resolution ("house").
    max_tokens: Optional[int]
    #: The official cap of this mode's settings, or None.
    official_cap: Optional[int]
    #: The result's `max_output_tokens_note`, or None.
    note: Optional[str]

    @property
    def uncapped(self) -> bool:
        return self.policy == "not_sent"

    def record(self) -> Dict[str, Any]:
        """Fields added to the result's `official_sampling` block."""
        if self.policy == "official":
            return {"max_tokens": self.official_cap, "max_tokens_applied": True}
        if self.policy == "official_reasoning":
            # applied only when the reasoning cap itself is what is sent: a larger
            # --max-output-tokens replaces it (a thinking run is never capped below the flag)
            applied = self.max_tokens == self.official_cap
            return {"max_tokens": self.official_cap, "max_tokens_applied": applied,
                    "max_tokens_sent": self.max_tokens,
                    **({} if applied else {"max_tokens_note": RAISED_REASONING_CAP_NOTE})}
        if self.policy == "reasoning_run_budget":
            return {"max_tokens": self.official_cap, "max_tokens_applied": False,
                    "max_tokens_note": REASONING_CAP_NOTE, "max_tokens_sent": self.max_tokens}
        return {}


#: Recorded when a suite's official max_tokens is not applied to a --thinking run.
REASONING_CAP_NOTE = ("official cap not applied to reasoning: it was defined for non-reasoning "
                      "generation")

#: Recorded when --max-output-tokens is larger than an upstream's explicit reasoning-model cap.
RAISED_REASONING_CAP_NOTE = ("--max-output-tokens is larger than the official reasoning-model cap "
                             "and was sent instead (a thinking run is never capped below "
                             "--max-output-tokens)")


def thinking_run_budget(eval_name: str, max_output_tokens: Optional[int] = None) -> int:
    """The per-request budget of a --thinking run that applies no official cap: the run's
    --max-output-tokens, else the thinking floor (THINKING_MIN_OUTPUT_TOKENS, or the suite's
    SUITE_MIN_OUTPUT_TOKENS entry when larger)."""
    knob = max_output_tokens if max_output_tokens else get_run_knob("max_output_tokens")
    if knob:
        return int(knob)
    return max(SUITE_MIN_OUTPUT_TOKENS.get(eval_name, 0), THINKING_MIN_OUTPUT_TOKENS)


def resolve_token_budget(eval_name: str, official: Optional[OfficialSampling], *,
                         thinking: bool, max_output_tokens: Optional[int] = None) -> TokenBudget:
    """The token-limit policy, in one place for every suite.

    Official max-token caps come from harnesses written for non-reasoning generation, so:
      * no-think: the official cap is sent exactly (`OfficialSampling.max_tokens`), or no
        max_tokens at all when the official sampler sends none (`send_max_tokens=False`);
        --max-output-tokens, the floors and the per-turn ceiling are not applied;
      * --thinking: the generation is never capped below the run's --max-output-tokens. An
        official cap is not applied to reasoning (the result records that it was defined for
        non-reasoning generation); the upstream's explicit reasoning-model settings
        (`OfficialSampling.reasoning`) are used when it defines them;
      * a suite that declares no official cap for the mode keeps the house budget (policy
        "house"), unchanged.

    `max_output_tokens` is the run's --max-output-tokens (default: the run knob). Suites outside
    run_eval_suite call this and send `max_tokens` (None = no field), recording `note` as
    `max_output_tokens_note` and `record()` under `official_sampling`.
    """
    knob = max_output_tokens if max_output_tokens else get_run_knob("max_output_tokens")
    if official is None:
        return TokenBudget("house", None, None, None)
    explicit_reasoning = bool(thinking and official.reasoning is not None)
    mode = official.for_mode(thinking)
    who = ("the official reasoning-model sampler" if explicit_reasoning
           else "the official harness")
    not_applied = f"--max-output-tokens {knob} not applied" if knob else ""
    if not mode.send_max_tokens:
        return TokenBudget("not_sent", None, None, (
            f"not sent: {who} sends no max_tokens, so the server's default applies (the room "
            "left in the context window)" + (f"; {not_applied}" if not_applied else "")))
    cap = mode.max_tokens
    if cap is None:
        return TokenBudget("house", None, None, None)
    cap = int(cap)
    if knob and int(knob) == cap:
        not_applied = ""                 # the flag asks for the official cap itself
    if not thinking:
        return TokenBudget("official", cap, cap, (
            (f"{not_applied}: " if not_applied else "")
            + f"{who} sends max_tokens={cap} with every request ({mode.reference})"))
    if explicit_reasoning:
        sent = max(cap, int(knob)) if knob else cap
        if sent > cap:
            note = (f"{who} sends max_tokens={cap}; --max-output-tokens {knob} is larger and was "
                    "sent instead (a thinking run is never capped below --max-output-tokens)")
        else:
            note = (f"{who} sends max_tokens={cap} with every request ({mode.reference})"
                    + (f"; {not_applied}" if not_applied else ""))
        return TokenBudget("official_reasoning", sent, cap, note)
    budget = thinking_run_budget(eval_name, knob)
    source = f"--max-output-tokens {knob}" if knob else f"the thinking floor {budget}"
    return TokenBudget("reasoning_run_budget", budget, cap, (
        f"{REASONING_CAP_NOTE} ({who} sends max_tokens={cap}); each request was sent this run's "
        f"budget, {source}"))


def resolve_official_sampling(eval_name: str, official: OfficialSampling,
                              *, thinking: bool = False) -> Tuple[Optional[float], str]:
    """(temperature, source) for a suite with official sampling settings.

    Precedence: `GBENCH_<SUITE>_TEMPERATURE`, then `--temperature` (as in resolve_temperature),
    then the maintainer's rule (resolve_temperature): 1.0 for a --thinking run (source
    "thinking"), 0.0 otherwise (source "default"), whatever the official setting says,
    `send_temperature=False` included (a temperature is always sent). The official values are
    only recorded (OfficialSampling.record). `official.params` is checked
    here too (check_official_sampling_params), so a suite outside run_eval_suite cannot record
    a non-sampling field as official. A --thinking run uses `official.reasoning` when set.
    """
    check_official_sampling_params(eval_name, official)
    official = official.for_mode(thinking)
    return resolve_temperature(eval_name, official.temperature, thinking=thinking)


#: Pessimistic per-sequence decode rate used to size the HTTP timeout. Under heavy
#: concurrency each sequence gets a small share of aggregate throughput, so a long
#: generation legitimately takes a long time.
MIN_DECODE_TOK_S = float(os.environ.get("GBENCH_MIN_DECODE_TOK_S", "8"))
REQUEST_TIMEOUT_FLOOR_S = int(os.environ.get("GBENCH_REQUEST_TIMEOUT_S", "1200"))

#: Error prefix for a request the server rejected because the prompt alone exceeds its
#: context window; such rows are reported as OVER_CONTEXT, like a client-side refusal.
OVER_CONTEXT_ERROR = "over_context"


#: Decoding penalties, OFF by default. The 2026-08-15 run showed the model hitting the
#: token cap while repeating itself (a 20-gram recurring up to 379 times), which a penalty
#: can suppress - but canonical benchmark protocol is greedy with no penalty, so switching
#: one on silently would make every number non-comparable with published results and with
#: gbench's own earlier runs. Opt in per run and the value is recorded on the result.
#:
#: Note it does NOT address unbounded enumeration ("Step 375: Let's try k=-283"), which is
#: search rather than repetition and shows a 20-gram count of 2.
def decoding_penalties() -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, env in (("repetition_penalty", "GBENCH_REPETITION_PENALTY"),
                     ("frequency_penalty", "GBENCH_FREQUENCY_PENALTY"),
                     ("presence_penalty", "GBENCH_PRESENCE_PENALTY")):
        raw = os.environ.get(env)
        if raw:
            try:
                out[key] = float(raw)
            except ValueError:
                logger.warning("%s=%r is not a number; ignoring", env, raw)
    return out



def normalize_function_calls(tool_calls: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """The model's emitted calls as `[{"name": str, "arguments": dict}]`.

    Recorded so a tool-use suite can be scored OFFLINE later. Some tool-use suites are graded
    by comparing the emitted FunctionCall AST against a golden one; when the golden is not
    in the export yet, the model's side of that comparison is still worth banking - it means
    scoring, once targets arrive, is a diff over stored JSON rather than a re-run (no GPU,
    no re-sampling, no temperature variance between the two halves).

    The wire form nests the call under `function` and carries `arguments` as a JSON STRING,
    which is not diffable. Arguments that will not parse are kept verbatim under
    `arguments_raw` rather than dropped - a malformed argument list is itself a result.
    """
    out: List[Dict[str, Any]] = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else call
        name = fn.get("name")
        if not name:
            continue
        raw = fn.get("arguments")
        entry: Dict[str, Any] = {"name": name}
        if isinstance(raw, dict):
            entry["arguments"] = raw
        elif isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                entry["arguments"] = parsed if isinstance(parsed, dict) else {"_value": parsed}
            except ValueError:
                entry["arguments"] = None
                entry["arguments_raw"] = raw[:4000]
        else:
            entry["arguments"] = {}
        out.append(entry)
    return out


def request_timeout_s(max_output_tokens: Optional[int]) -> int:
    """How long one request may take, given how much it is allowed to generate.

    A fixed 1200 s was fine at 8192 tokens and wrong at 65536: at 256-way concurrency each
    sequence gets a small slice of aggregate decode throughput, so a long answer needs well
    over an hour. The request would time out, retry three times and land as a bare
    `request_failed` - i.e. the campaign would silently drop exactly its longest answers,
    which are the ones the big budgets exist to capture.

    Scales with the token budget rather than the concurrency because the budget is what
    bounds the work; `GBENCH_MIN_DECODE_TOK_S` tunes the assumed floor rate.
    """
    budget = max_output_tokens or 8192
    return max(REQUEST_TIMEOUT_FLOOR_S,
               int(REQUEST_TIMEOUT_FLOOR_S + budget / max(1.0, MIN_DECODE_TOK_S)))


#: Generation length the request timeout assumes for a request that carries no max_tokens when
#: the server's context window is not known either: 262,144 tokens (a 256K window), so an
#: uncapped reply is not cut short by a timeout sized for a capped one.
UNCAPPED_TIMEOUT_TOKENS = 262144


#: chars-per-token heuristics, used only when neither the server's /tokenize endpoint nor
#: the run's tokenizer can count a prompt. 3.0 over-estimates English text and 5.0
#: under-estimates it, but neither is a guaranteed bound; exact counts are always preferred.
#: Refusing on the 3.0 estimate threw away long-context prompts that fit: on a 128K window
#: it refused 58/100 AA-LCR and 100/100 LOFT SciFact prompts.
_CHARS_PER_TOKEN = 3.0
_CHARS_PER_TOKEN_LOWER = 5.0
#: Chat-template tokens when counting locally: per message (Gemma's template adds about 4)
#: and per request (BOS plus the generation prompt, about 13).
_TEMPLATE_TOKENS_PER_MESSAGE = 8
_TEMPLATE_TOKENS_PER_REQUEST = 16

#: The run's tokenizer (``--tokenizer``, else the model's HF id), loaded once. False = tried
#: and unavailable.
_TOKENIZER_CACHE: Dict[str, Any] = {}
_TOKENIZER_LOCK = threading.Lock()
#: Servers whose /tokenize endpoint is absent (e.g. Ollama): not asked again.
_NO_TOKENIZE_ENDPOINT: set = set()
#: Exact counts already fetched, keyed by (server, request digest). Bounded.
_COUNT_CACHE: Dict[Tuple[str, str], int] = {}
_COUNT_CACHE_MAX = 4096


def _run_tokenizer() -> Any:
    name = get_run_knob("tokenizer")
    if not name:
        return None
    with _TOKENIZER_LOCK:
        if name not in _TOKENIZER_CACHE:
            try:
                from transformers import AutoTokenizer
                # trust_remote_code as the runner resolved it (--tokenizer-trust-remote-code,
                # else --trust-remote-code in --vllm-serve-args)
                _TOKENIZER_CACHE[name] = AutoTokenizer.from_pretrained(
                    name, trust_remote_code=bool(get_run_knob("tokenizer_trust_remote_code",
                                                              False)))
            except Exception as e:  # unknown id (e.g. an Ollama tag), no network, ...
                logger.warning("Prompt sizes fall back to a character estimate: could not load "
                               "tokenizer %r (%s). Pass --tokenizer for exact counts.", name, e)
                _TOKENIZER_CACHE[name] = False
        return _TOKENIZER_CACHE[name] or None


def _count_tokens(text: str) -> Optional[int]:
    """Token count of ``text`` under the run's tokenizer (no template), or None."""
    tok = _run_tokenizer()
    if tok is None:
        return None
    try:
        with _TOKENIZER_LOCK:  # fast tokenizers are not safe to call concurrently
            return len(tok(text, add_special_tokens=False)["input_ids"])
    except Exception:
        return None


#: A whole image is a fixed, small number of soft tokens (gemma-4 tops out at 1120), not a
#: function of its base64 length. Counting the payload as text made a 2 MB document page
#: look like ~2.9 MILLION tokens and blocked the request as over-context: on the
#: 2026-08-17 run that silently dropped 56 requests across 10 vision/long-context suites
#: (omnidocbench 9/20, screenspot 9/20, mrcr 7/20, ...). Budget the maximum.
IMAGE_SOFT_TOKENS = int(os.environ.get("GBENCH_IMAGE_SOFT_TOKENS", "1120"))

_IMAGE_PART_TYPES = ("image_url", "image", "input_image")

#: An audio part is budgeted the same way: a fixed number of soft tokens, not its base64
#: length (gaia sends its mp3 attachments as `input_audio`; counted as text, about 370 KB of
#: base64 would look like ~100k tokens and refuse the task as over-context).
AUDIO_SOFT_TOKENS = int(os.environ.get("GBENCH_AUDIO_SOFT_TOKENS", "750"))

_AUDIO_PART_TYPES = ("input_audio", "audio_url")

#: A video part is budgeted the same way. The default is gemma-4's maximum per video in vLLM
#: (gemma4_mm.py: _VIDEO_MAX_FRAMES 32 frames x (_VIDEO_MAX_SOFT_TOKENS 70 + 2 + 6)). gaia's test
#: split sends a 30.7 MB .MOV as a `video_url` data URI: counted as text it looked like millions
#: of tokens and was refused as over-context, and a /tokenize count would upload the whole video.
VIDEO_SOFT_TOKENS = int(os.environ.get("GBENCH_VIDEO_SOFT_TOKENS", "2496"))

_VIDEO_PART_TYPES = ("video_url",)


def _media_soft_tokens(images: int, audios: int, videos: int = 0) -> int:
    """Soft-token budget for the image, audio and video parts of a prompt."""
    return images * IMAGE_SOFT_TOKENS + audios * AUDIO_SOFT_TOKENS + videos * VIDEO_SOFT_TOKENS


def _video_parts(messages: Any) -> int:
    """Number of video parts in a chat message list (0 for anything else)."""
    if not isinstance(messages, list):
        return 0
    return sum(1 for m in messages if isinstance(m, dict) and isinstance(m.get("content"), list)
               for part in m["content"]
               if isinstance(part, dict) and part.get("type") in _VIDEO_PART_TYPES)


def effective_chat_template_kwargs(chat_template_kwargs: Any, thinking: bool) -> Dict[str, Any]:
    """The `chat_template_kwargs` `_send_single_request` sends for a payload and thinking flag.

    A suite that set `enable_thinking` itself keeps it; otherwise the run's thinking flag is
    added. Prompt counting sends the same dict to /tokenize, so the count matches the request.
    """
    ctk = chat_template_kwargs if isinstance(chat_template_kwargs, dict) else {}
    if "enable_thinking" in ctk:
        return dict(ctk)
    return {**ctk, "enable_thinking": bool(thinking)}


def _text_only_messages(messages: Any) -> Optional[Tuple[List[Dict[str, Any]], int, int]]:
    """``messages`` with image, audio and video parts removed (they are counted as soft tokens),
    and the image and audio counts, or None when the messages are not a readable chat list.
    Video parts are counted by `_video_parts`."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}], 0, 0
    if not isinstance(messages, list):
        return None
    out, images, audios = [], 0, 0
    for m in messages:
        if not isinstance(m, dict):
            return None
        content = m.get("content")
        if isinstance(content, list):
            kept = []
            for part in content:
                if isinstance(part, dict) and part.get("type") in _IMAGE_PART_TYPES:
                    images += 1
                elif isinstance(part, dict) and part.get("type") in _AUDIO_PART_TYPES:
                    audios += 1
                elif isinstance(part, dict) and part.get("type") in _VIDEO_PART_TYPES:
                    continue
                else:
                    kept.append(part)
            m = {**m, "content": kept or ""}
        out.append(m)
    return out, images, audios


def _server_prompt_tokens(messages: Any, tools: Any = None,
                          chat_template_kwargs: Optional[Dict[str, Any]] = None,
                          prefill: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Exact prompt tokens, chat template and tools included, from the server's /tokenize.

    Set up per suite by the eval runner (run knobs ``tokenize_url`` and ``served_model``).
    ``chat_template_kwargs`` is forwarded in the /tokenize body (vLLM renders the template
    with it, e.g. ``enable_thinking``), so pass the dict the chat request sends; so is
    ``prefill`` (the request's ``continue_final_message`` / ``add_generation_prompt``, see
    prefill_fields), so a prefilled assistant turn is counted as the server renders it. Image,
    audio and video parts are not sent; they are added as soft-token budgets. Returns None when the server has no such
    endpoint or the call fails.
    """
    url, model = get_run_knob("tokenize_url"), get_run_knob("served_model")
    if not url or not model or url in _NO_TOKENIZE_ENDPOINT:
        return None
    got = _text_only_messages(messages)
    if got is None:
        return None
    clean, images, audios = got
    media = _media_soft_tokens(images, audios, _video_parts(messages))
    body: Dict[str, Any] = {"model": model, "messages": clean, "add_generation_prompt": True}
    if prefill:
        body.update(prefill)
    if tools:
        body["tools"] = tools
    if chat_template_kwargs is not None:
        body["chat_template_kwargs"] = chat_template_kwargs
    try:
        data = json.dumps(body).encode("utf-8")
    except (TypeError, ValueError):
        return None
    # The key digests the whole body, so the tools and chat_template_kwargs are part of it.
    key = (url, hashlib.sha256(data).hexdigest())
    if key in _COUNT_CACHE:
        return _COUNT_CACHE[key] + media
    try:
        req = urllib.request.Request(url.rstrip("/") + "/tokenize", data=data,
                                     headers={"Content-Type": "application/json",
                                              **endpoint_auth_headers()})
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")
        except Exception:
            detail = ""
        # 404 "The model `x` does not exist" is vLLM rejecting a model name, not a missing
        # endpoint; only an absent route disables /tokenize for the rest of the run.
        if e.code in (405, 501) or (e.code == 404 and "does not exist" not in detail):
            _NO_TOKENIZE_ENDPOINT.add(url)
            logger.info("%s has no /tokenize endpoint; prompt sizes are counted locally.", url)
        return None
    except Exception:
        return None
    count = payload.get("count") if isinstance(payload, dict) else None
    if not isinstance(count, int):
        # Some other server's /tokenize: its answer is not a vLLM count, so stop asking.
        _NO_TOKENIZE_ENDPOINT.add(url)
        return None
    if len(_COUNT_CACHE) >= _COUNT_CACHE_MAX:
        _COUNT_CACHE.clear()
    _COUNT_CACHE[key] = count
    return count + media


def _prompt_text_and_images(messages: Any) -> Optional[Tuple[str, int, int, int]]:
    """The prompt's text (images, audio and video excluded), its image count, its message count
    and its audio count. Video parts are counted by `_video_parts`."""
    import json as _json
    if isinstance(messages, str):
        return messages, 0, 1, 0
    if not isinstance(messages, list):
        return None
    parts: List[str] = []
    images = audios = 0
    try:
        for m in messages:
            content = (m or {}).get("content") if isinstance(m, dict) else None
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        parts.append(str(part)); continue
                    if part.get("type") in _IMAGE_PART_TYPES:
                        images += 1
                    elif part.get("type") in _AUDIO_PART_TYPES:
                        audios += 1
                    elif part.get("type") in _VIDEO_PART_TYPES:
                        continue
                    elif isinstance(part.get("text"), str):
                        parts.append(part["text"])
                    else:
                        parts.append(_json.dumps(part))
            elif content is not None:
                parts.append(_json.dumps(content))
            # tool calls / names / role overhead
            for k in ("tool_calls", "name", "tool_call_id"):
                if isinstance(m, dict) and m.get(k):
                    parts.append(_json.dumps(m[k]))
    except Exception:
        return None
    return "\n".join(parts), images, len(messages), audios


def _prompt_token_bounds(messages: Any, tools: Any = None,
                         chat_template_kwargs: Optional[Dict[str, Any]] = None,
                         prefill: Optional[Dict[str, Any]] = None,
                         ) -> Optional[Tuple[int, int, str]]:
    """(low, high, mode) prompt size in tokens.

    mode "server": count from /tokenize (low == high; ``chat_template_kwargs`` and ``prefill``
    are forwarded to it). "tokenizer": the run's tokenizer plus a per-message template allowance and the
    tools' JSON (low == high, conservative). "estimate": character heuristics. Images, audio
    and video count as soft tokens, not base64 length.
    """
    exact = (_server_prompt_tokens(messages, tools, chat_template_kwargs, prefill) if prefill
             else _server_prompt_tokens(messages, tools, chat_template_kwargs))
    if exact is not None:
        return exact, exact, "server"
    got = _prompt_text_and_images(messages)
    if got is None:
        return None
    text, images, n_msgs, audios = got
    if tools:
        try:
            text += "\n" + json.dumps(tools)
        except (TypeError, ValueError):
            text += "\n" + str(tools)
    img = _media_soft_tokens(images, audios, _video_parts(messages))
    counted = _count_tokens(text) if text else 0
    if counted is not None:
        n = (counted + _TEMPLATE_TOKENS_PER_REQUEST
             + n_msgs * _TEMPLATE_TOKENS_PER_MESSAGE + img)
        return n, n, "tokenizer"
    low = int(len(text) / _CHARS_PER_TOKEN_LOWER) + n_msgs * (_TEMPLATE_TOKENS_PER_MESSAGE // 2) + img
    high = (int(len(text) / _CHARS_PER_TOKEN) + _TEMPLATE_TOKENS_PER_REQUEST
            + n_msgs * _TEMPLATE_TOKENS_PER_MESSAGE + img)
    return low, high, "estimate"


def _estimate_prompt_tokens(messages: Any) -> Optional[int]:
    """Prompt size in tokens (exact when countable, else a conservative high estimate)."""
    bounds = _prompt_token_bounds(messages)
    return bounds[1] if bounds else None


def _is_text_only(messages: Any) -> bool:
    """True when the prompt is a readable chat list with no image, audio or video part."""
    got = _text_only_messages(messages)
    return got is not None and got[1] == 0 and got[2] == 0 and _video_parts(messages) == 0


#: Minimum answer room: a prompt that leaves less than this (or less than the request's own
#: `max_tokens`, when that is smaller) is refused as over-context.
_MIN_ANSWER_ROOM = 256


def clamp_to_context(messages: Any, max_tokens: Optional[int],
                     max_model_len: Optional[int] = None,
                     tools: Any = None,
                     chat_template_kwargs: Optional[Dict[str, Any]] = None,
                     prefill: Optional[Dict[str, Any]] = None,
                     ) -> Tuple[Optional[int], Optional[str]]:
    """Shrink `max_tokens` so `prompt + max_tokens` fits the server's context window.

    vLLM rejects a request whose prompt+max_tokens exceeds max_model_len. The old code
    only recovered *reactively*, by parsing the 400 and retrying - which costs a round
    trip per request and, when the prompt alone is over the limit, simply retried three
    times and recorded `request_failed` with no reason at all. On the 2026-08-15 sweep
    that was mrcr: 6 rows with 1.3-4.2 MB prompts (~330k-1M tokens) against a 262144
    window, reported as unexplained failures.

    Returns `(clamped_max_tokens, reason_if_unanswerable)`. A non-None reason means the
    prompt cannot fit at all and the request should not be sent.

    The prompt is counted exactly (server /tokenize, else the run's tokenizer) whenever
    possible. With only the character heuristics, a request is refused when even the 5
    chars/token heuristic says it cannot fit (a heuristic, not a bound: very repetitive text
    packs more characters per token and could still fit); a borderline one is sent, and if
    the server rejects it, _send_single_request shrinks max_tokens and, when even 256 tokens
    do not fit, records OVER_CONTEXT, exactly as if it had been refused here.

    The answer room required is 256 tokens, or the request's own `max_tokens` when that is
    smaller (a suite that asks for 64 tokens needs only 64). Pass `tools` and
    `chat_template_kwargs` exactly as the chat request sends them: with both, a text-only
    prompt counted by the server's /tokenize is exact for the full request and gets no slack,
    so a prompt built to fill the window up to its answer cap (RULER's top band) still fits.
    `prefill` (prefill_fields of the request) is forwarded to /tokenize as well.
    """
    if not max_model_len:
        max_model_len = get_run_knob("max_model_len")
    if not max_model_len:
        return max_tokens, None
    try:
        bounds = (_prompt_token_bounds(messages, tools, chat_template_kwargs, prefill) if prefill
                  else _prompt_token_bounds(messages, tools, chat_template_kwargs))
    except Exception:          # an unreadable prompt must not abort the suite; let the server decide
        bounds = None
    if bounds is None:
        return max_tokens, None
    low, high, mode = bounds
    need = (min(_MIN_ANSWER_ROOM, max_tokens)
            if isinstance(max_tokens, int) and max_tokens > 0 else _MIN_ANSWER_ROOM)
    if mode == "server" and chat_template_kwargs is not None and _is_text_only(messages):
        slack = 0            # the server counted the same messages, tools and template kwargs
    else:
        slack = 16 if mode == "server" else 64           # template overhead not yet counted
    if max_model_len - low - slack < need:
        size = f"{low:,}" if low == high else f"{low:,}-{high:,}"
        return max_tokens, (
            f"prompt is ~{size} tokens, which does not fit the model's "
            f"{max_model_len:,}-token context window (no room left to answer)")
    headroom = max_model_len - high - slack
    if headroom < need:                                   # borderline estimate: optimistic
        headroom = max_model_len - low - slack
    if max_tokens is None or max_tokens > headroom:
        return headroom, None
    return max_tokens, None


async def clamp_to_context_async(messages: Any, max_tokens: Optional[int],
                                 tools: Any = None,
                                 chat_template_kwargs: Optional[Dict[str, Any]] = None,
                                 prefill: Optional[Dict[str, Any]] = None,
                                 ) -> Tuple[Optional[int], Optional[str]]:
    """clamp_to_context off the event loop: counting may tokenize or call /tokenize, and
    doing that inline stalled every in-flight request (25 s on AA-LCR, 151 s on MRCR)."""
    if prefill:
        return await asyncio.to_thread(clamp_to_context, messages, max_tokens, None, tools,
                                       chat_template_kwargs, prefill)
    return await asyncio.to_thread(clamp_to_context, messages, max_tokens, None, tools,
                                   chat_template_kwargs)

_GRADE_VERDICT_RE = re.compile(r"GRADE\s*:\s*(NOT_ATTEMPTED|INCORRECT|CORRECT)", re.IGNORECASE)


def parse_grade_verdict(grade_text: str) -> bool:
    """Is the judge's verdict CORRECT? Reads the verdict, not the whole reply.

    The suites all asked for `Grade: CORRECT / INCORRECT` and then tested
    ``"CORRECT" in grade_str and "INCORRECT" not in grade_str`` over the entire response.
    Judges routinely ignore the one-word instruction and return several paragraphs of
    analysis, so a reply reasoning "the patch is not incorrect ... GRADE: CORRECT" scored
    as wrong: the word `INCORRECT` appeared somewhere in the prose.

    The last explicit `GRADE:` line wins (a judge that revises itself means the later
    one); only when the reply contains no explicit verdict at all do we fall back to the
    old substring behaviour.
    """
    text = str(grade_text or "")
    verdicts = _GRADE_VERDICT_RE.findall(text)
    if verdicts:
        return verdicts[-1].upper() == "CORRECT"
    upper = text.upper()
    return "CORRECT" in upper and "INCORRECT" not in upper
try:
    import aiohttp
except ImportError:
    aiohttp = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else None

logger = logging.getLogger(__name__)
logging.getLogger("huggingface_hub.repocard").setLevel(logging.ERROR)
for noisy_logger in ["google_genai", "google", "grpc", "httpx", "httpcore", "urllib3"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# --- Judge-model resolution: loud, guarded, recordable ------------------------------
# GBENCH_JUDGE_MODELS / GBENCH_JUDGE_MODEL redirect EVERY Gemini text role (judges, graders,
# labelers, user simulators, and web search unless GBENCH_SEARCH_MODELS is set). Left unlogged
# and unrecorded, a whole sweep could be graded by the wrong model - even the model UNDER TEST
# judging its own outputs - and the only symptom was an exhausted API quota for that model. The
# pieces below make an override announce itself, flag self-judging, and let callers record the
# resolved models on the result.
def _log_cascade_override() -> None:
    source = judge_cascade_source()
    if source == "default":
        return
    shadowed = (f" ({JUDGE_MODEL_ENV} is also set and is ignored: {JUDGE_MODELS_ENV} wins)"
                if source == JUDGE_MODELS_ENV and (os.environ.get(JUDGE_MODEL_ENV) or "").strip()
                else "")
    logger.warning(
        "[judge] %s override is ACTIVE: every Gemini text role (judges, graders, labelers, user "
        "simulators, and web search unless %s is set) uses the cascade %s instead of the "
        "default %s%s. Unset it to restore the default.", source, SEARCH_MODELS_ENV,
        judge_cascade(), list(GEMINI_CASCADE), shadowed)


_log_cascade_override()


def _norm_model_id(m: Optional[str]) -> str:
    """Lowercased model id without provider prefix: `google/Gemma-4-26B-A4B-it` -> `gemma-4-26b-a4b-it`."""
    m = (m or "").strip().lower()
    return m.rsplit("/", 1)[-1] if "/" in m else m


def _same_model_family(judge: Optional[str], under_test: Optional[str]) -> bool:
    """True when the judge model is (a variant of) the model under test - i.e. self-judging."""
    j, u = _norm_model_id(judge), _norm_model_id(under_test)
    if not j or not u:
        return False
    return j == u or j.startswith(u) or u.startswith(j)


_JUDGE_RESOLVE_LOGGED: set = set()


def resolve_judge_model(under_test_model: Optional[str] = None) -> Tuple[str, str, bool]:
    """Resolve the LLM-judge model LOUDLY, with a self-judge guard: the HEAD of judge_cascade().

    Returns `(model, source, is_self_judge)`: `model` is judge_cascade()[0], `source` is
    'env:GBENCH_JUDGE_MODELS', 'env:GBENCH_JUDGE_MODEL' or 'default' (GEMINI_CASCADE), and
    `is_self_judge` says whether that model is the model under test. Logs the full cascade once
    per (cascade, source, has-under-test). If a cascade model is the model under test, logs an
    ERROR - a model grading its own outputs is not a valid judgement and it routes grading to
    that model's own API quota. Never raises; grading proceeds, flagged. For a caller that makes
    a single-model Gemini call; code that can use every model should call judge_cascade().
    """
    cascade = judge_cascade()
    model = cascade[0]
    src = judge_cascade_source()
    source = "default" if src == "default" else f"env:{src}"
    is_self = bool(under_test_model and _same_model_family(model, under_test_model))
    key = (tuple(cascade), source, bool(under_test_model))
    if key not in _JUDGE_RESOLVE_LOGGED:
        _JUDGE_RESOLVE_LOGGED.add(key)
        logger.warning("[judge] grading with '%s', the head of the Gemini cascade %s (source: %s)",
                       model, cascade, source)
        selfish = [m for m in cascade
                   if under_test_model and _same_model_family(m, under_test_model)]
        if selfish:
            logger.error(
                "[judge] SELF-JUDGE DETECTED: Gemini cascade model(s) %s are the model under test "
                "('%s'). A model grading its own outputs is not a valid judgement and burns that "
                "model's API quota. Unset %s / %s to use the default cascade.",
                selfish, under_test_model, JUDGE_MODELS_ENV, JUDGE_MODEL_ENV)
    return model, source, is_self


#: The self-judge comparability reason (the model under test is in the Gemini cascade).
SELF_JUDGE_REASON = "judge cascade contains the model under test"

#: Suites whose Gemini gate ran in this process (gemini_required_skip), with the model under test
#: each was gated for: {eval_name: model_name}.
_GEMINI_GATED: Dict[str, str] = {}


def judge_provenance(model_under_test: Optional[str]) -> Dict[str, Any]:
    """The judge-provenance fields of a result: {"gemini_cascade": judge_cascade(),
    "gemini_cascade_source": "default" | "env:GBENCH_JUDGE_MODELS" | "env:GBENCH_JUDGE_MODEL",
    "self_judge": bool, "self_judge_models": [...]}. `self_judge` is True when a cascade model is
    (a variant of) the model under test (_same_model_family, a model-agnostic name comparison).
    run_eval_suite stamps it on every result; a suite outside run_eval_suite merges it into its
    own, and apply_self_judge() marks such a result non-comparable."""
    cascade = judge_cascade()
    src = judge_cascade_source()
    selfish = [m for m in cascade if model_under_test and _same_model_family(m, model_under_test)]
    return {"gemini_cascade": cascade,
            "gemini_cascade_source": "default" if src == "default" else f"env:{src}",
            "self_judge": bool(selfish), "self_judge_models": selfish}


def apply_self_judge(result: Dict[str, Any], model_under_test: Optional[str]) -> Dict[str, Any]:
    """Stamp judge_provenance on `result`; when the cascade holds the model under test, mark the
    result not leaderboard-comparable (SELF_JUDGE_REASON). Grading still ran (flagged, not
    refused); resolve_judge_model logs it as an error at the gate."""
    prov = judge_provenance(model_under_test)
    result.update(prov)
    if prov["self_judge"]:
        add_comparability_reason(result, SELF_JUDGE_REASON)
    return result


def strip_thinking_tags(text: Optional[str]) -> str:
    """Strip <thought>...</thought> or <reasoning>...</reasoning> blocks if present."""
    if not text:
        return ""
    cleaned = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<reasoning>.*?</reasoning>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    # If output was entirely wrapped in thought tags, return stripped raw text
    return cleaned.strip() if cleaned.strip() else text.strip()


def _is_bulky_data_uri(value: Any) -> bool:
    """An inline image or audio data URI too long to keep verbatim in a trace."""
    return (isinstance(value, str) and len(value) > 64
            and (value.startswith("data:image/") or value.startswith("data:audio/")))


def _sanitize_for_trace(obj: Any) -> Any:
    """Recursively truncate bulky payloads for human-readable JSON traces."""
    if isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            if _is_bulky_data_uri(v):
                res[k] = v[:48] + "..."
            elif (k == "input_audio" and isinstance(v, dict) and isinstance(v.get("data"), str)
                  and len(v["data"]) > 64):
                # An `input_audio` part carries raw base64 (no data: prefix) under `data`.
                res[k] = {**_sanitize_for_trace({kk: vv for kk, vv in v.items() if kk != "data"}),
                          "data": v["data"][:48] + "..."}
            elif k == "tools" and isinstance(v, list) and len(v) > 4:
                # A tool-use suite may offer 100+ schemas, identical on every sample; kept
                # verbatim they dominate the result file.
                # Names are what a failure analysis needs, so keep only those.
                res[k] = [t.get("function", {}).get("name", t) if isinstance(t, dict) else t
                          for t in v]
            else:
                res[k] = _sanitize_for_trace(v)
        return res
    elif isinstance(obj, list):
        return [_sanitize_for_trace(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_sanitize_for_trace(item) for item in obj)
    elif _is_bulky_data_uri(obj):
        return obj[:48] + "..."
    return obj


# --------------------------------------------------------------------------- #
# credential redaction: the backstop for everything gbench writes
# --------------------------------------------------------------------------- #
#: Written in place of a credential value in traces, results, summaries and log lines.
REDACTED = "[REDACTED]"
#: Credential variables whose values must never reach a trace, a result file or a log line. Any
#: other variable whose name ends in _API_KEY(S), _TOKEN(S), _SECRET(S), _SECRET_KEY, _ACCESS_KEY,
#: _PASSWORD or _CREDENTIAL(S) is treated the same way (is_secret_env_name). Names no suite reads
#: any more (GBENCH_TOOLBENCH_SIMULATOR_KEY, TB_SIMULATOR_KEY) stay listed so a value left set in
#: an operator's environment is still masked.
SECRET_ENV_NAMES = (
    "GEMINI_API_KEY", "GEMINI_API_KEYS", "OPENAI_API_KEY", "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "VLLM_API_KEY",
    "GBENCH_MODEL_API_KEY", "GBENCH_TAU_AGENT_KEY", "GBENCH_TOOLBENCH_SIMULATOR_KEY",
    "TB_SIMULATOR_KEY", "MONGODB_CONNECTION_STRING",
)
_SECRET_NAME_RE = re.compile(
    r"(_API_KEYS?|_TOKENS?|_SECRETS?|_SECRET_KEY|_ACCESS_KEY|_PASSWORD|_PASSWD|_CREDENTIALS?)$",
    re.IGNORECASE)
#: The userinfo of a URL (`scheme://user:password@host`).
_SECRET_URL_USERINFO_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://([^/\s@]+)@")
#: Values shorter than this, or made only of digits, are not treated as secrets: redacting a
#: placeholder such as EMPTY or a numeric knob (a *_TOKENS budget) would corrupt ordinary text.
_MIN_SECRET_LEN = 8
#: Placeholder values that stand in for "no key" (compared case-insensitively). A server that
#: needs no key accepts any of them, so they are not credentials, and redacting them would turn
#: ordinary words in traces into [REDACTED].
PLACEHOLDER_SECRET_VALUES = frozenset({
    "empty", "dummy", "none", "null", "placeholder", "not-needed", "not_needed", "notneeded",
    "anything", "no-key", "nokey", "sk-no-key-required", "token-abc123", "changeme",
    "your-api-key", "your_api_key", "sk-placeholder", "sk-dummy", "sk-empty", "unused",
})
#: Credentials that never enter the environment (a token fetched at run time, such as a GCP
#: identity token), added with register_secret.
_REGISTERED_SECRETS: set = set()
_REGISTERED_LOCK = threading.Lock()
#: The huggingface_hub token file, read once per (path, mtime): huggingface_hub uses it when
#: HF_TOKEN is unset, so its value is a credential although it is not in the environment.
_HF_TOKEN_FILE_CACHE: Dict[str, Any] = {}


def is_secret_env_name(name: str) -> bool:
    """Is `name` a credential variable (SECRET_ENV_NAMES, or a credential-style suffix)?"""
    return name in SECRET_ENV_NAMES or bool(_SECRET_NAME_RE.search(name or ""))


def _is_secret_value(value: str) -> bool:
    return (len(value) >= _MIN_SECRET_LEN and not value.isdigit()
            and value.lower() not in PLACEHOLDER_SECRET_VALUES)


def register_secret(value: Optional[str]) -> None:
    """Treat `value` as a credential for the rest of the process: every redaction (results,
    logs, files under the results directory) replaces it. For a credential that is not in the
    environment, such as an identity token fetched at run time."""
    value = (value or "").strip()
    if _is_secret_value(value):
        with _REGISTERED_LOCK:
            _REGISTERED_SECRETS.add(value)


def _environ_snapshot() -> Dict[str, str]:
    """A copy of the process environment that another thread's writes cannot break.

    Iterating os.environ walks its mapping in Python and raises KeyError or RuntimeError when
    another thread sets or removes a variable meanwhile; copying the underlying dict is one C
    call. Retried through the public mapping if that is unavailable."""
    data = getattr(os.environ, "_data", None)
    decode = getattr(os.environ, "decodevalue", None)
    decode_key = getattr(os.environ, "decodekey", decode)
    if isinstance(data, dict) and callable(decode) and callable(decode_key):
        try:
            raw = dict(data)
            return {decode_key(k): decode(v) for k, v in raw.items()}
        except Exception:                                                  # noqa: BLE001
            pass
    for _ in range(5):
        try:
            return dict(os.environ)
        except (KeyError, RuntimeError):
            continue
    return {}


def _hf_token_file_value(env: Dict[str, str]) -> Optional[str]:
    """The token huggingface_hub reads from its token file when HF_TOKEN is unset, or None."""
    path = env.get("HF_TOKEN_PATH") or os.path.join(
        env.get("HF_HOME") or os.path.join(
            env.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"),
            "huggingface"),
        "token")
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        return None
    cached = _HF_TOKEN_FILE_CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            value = fh.read().strip()
    except OSError:
        value = None
    _HF_TOKEN_FILE_CACHE[path] = (mtime, value)
    return value


def secret_values(environ: Optional[Dict[str, str]] = None) -> List[str]:
    """The credential values to redact, longest first so a key is replaced before any shorter
    overlap: the values of credential variables in `environ` (default: the process
    environment); for a list-valued one (GEMINI_API_KEYS="k1,k2") the whole value and each
    member; the password of any variable's URL that has `user:password@` (a bare user name is
    not a credential and is left alone); and, for the process environment, the huggingface_hub
    token file and every register_secret value. Placeholders (PLACEHOLDER_SECRET_VALUES),
    values under 8 characters and all-digit values are left out."""
    env = _environ_snapshot() if environ is None else dict(environ)
    found = set()

    def add(value: Optional[str]) -> None:
        value = (value or "").strip()
        if _is_secret_value(value):
            found.add(value)

    for name, value in env.items():
        if not value:
            continue
        if is_secret_env_name(name):
            add(value)
            for part in re.split(r"[,;\s]+", value):
                add(part)
        for match in _SECRET_URL_USERINFO_RE.finditer(value):
            userinfo = match.group(1)
            if ":" in userinfo:
                add(userinfo.split(":", 1)[1])
    if environ is None:
        add(_hf_token_file_value(env))
        with _REGISTERED_LOCK:
            found.update(_REGISTERED_SECRETS)
    return sorted(found, key=lambda v: (-len(v), v))


@functools.lru_cache(maxsize=1024)
def _secret_forms(secret: str) -> Tuple[str, ...]:
    from urllib.parse import quote, quote_plus
    forms = [secret]
    for form in (json.dumps(secret)[1:-1], quote(secret, safe=""), quote_plus(secret, safe=""),
                 quote(secret)):
        if form not in forms:
            forms.append(form)
    return tuple(forms)


def secret_forms(secret: str) -> List[str]:
    """The ways `secret` can appear in written text: as is, JSON-escaped, and URL-encoded
    (percent-encoding and the form encoding of a query string)."""
    return list(_secret_forms(secret))


@functools.lru_cache(maxsize=64)
def _redaction_needles(secrets: Tuple[str, ...]) -> Tuple[str, ...]:
    """Every form of every secret, longest first. A shorter form can be a prefix of a longer
    one (a key ending in a backslash is a prefix of its JSON-escaped form), so replacing the
    shorter first would leave the rest of the longer form behind, invalid JSON included."""
    forms = {form for secret in secrets for form in _secret_forms(secret)}
    return tuple(sorted(forms, key=lambda f: (-len(f), f)))


def redact_text(text: Any, secrets: Optional[List[str]] = None) -> Any:
    """`text` with every credential value (as is, JSON-escaped or URL-encoded) replaced by
    REDACTED, longest form first. Anything that is not a string is returned unchanged."""
    if not isinstance(text, str) or not text:
        return text
    for form in _redaction_needles(tuple(secret_values() if secrets is None else secrets)):
        if form in text:
            text = text.replace(form, REDACTED)
    return text


def redact_bytes(data: bytes, secrets: Optional[List[str]] = None) -> bytes:
    """`data` with every credential value (in the forms of secret_forms, UTF-8 encoded)
    replaced by REDACTED, longest form first."""
    if not data:
        return data
    redacted = REDACTED.encode("utf-8")
    for form in _redaction_needles(tuple(secret_values() if secrets is None else secrets)):
        needle = form.encode("utf-8")
        if needle in data:
            data = data.replace(needle, redacted)
    return data


def redact_secrets(obj: Any, secrets: Optional[List[str]] = None) -> Any:
    """A JSON-like value (dicts, lists, tuples, strings) with every credential value of the
    environment replaced by REDACTED in its strings, dict keys included (two keys that redact
    to the same text are kept apart as "[REDACTED]", "[REDACTED]#2", ...). Copy on write: a
    container is copied only when something inside it changed, so a value without a credential
    comes back as the very same object. Applied to every result run_eval_suite returns. It
    walks only JSON-native containers; the final text every writer produces is redacted again
    after serialization (gbench.utils.log_manager), so a set, an exception, a path or any other
    object that is turned into text later cannot carry a key to disk."""
    secrets = secret_values() if secrets is None else secrets
    if not secrets:
        return obj

    def walk(value: Any) -> Any:
        if isinstance(value, str):
            return redact_text(value, secrets)
        if isinstance(value, dict):
            items = [(walk(k) if isinstance(k, str) else k, walk(v)) for k, v in value.items()]
            if all(nk is k and nv is v for (nk, nv), (k, v) in zip(items, value.items())):
                return value
            out: Dict[Any, Any] = {}
            for nk, nv in items:
                key, n = nk, 1
                while key in out:
                    n += 1
                    key = f"{nk}#{n}"
                out[key] = nv
            return out
        if isinstance(value, (list, tuple)):
            new = [walk(v) for v in value]
            if all(n is v for n, v in zip(new, value)):
                return value
            return new if isinstance(value, list) else tuple(new)
        return value
    return walk(obj)


#: Files redact_tree has already checked, path -> (mtime_ns, size, secrets fingerprint).
_TREE_SEEN: Dict[str, Tuple[int, int, str]] = {}
_TREE_LOCK = threading.Lock()
#: A file modified this recently may still be open for writing (a live server or harness log).
#: redact_file rewrites such a file without shrinking it, so a writer whose descriptor does
#: not append keeps writing right after the rewritten text instead of leaving a run of NUL
#: bytes where the text used to end.
LIVE_FILE_WINDOW_S = 60.0


def _rewrite_in_place(fh: Any, data: bytes, new: bytes, keep_length: bool) -> None:
    """Write `new` over `data` (the file's content when it was read) in the open file `fh`.

    Bytes another process appended after the read are kept: with `keep_length` (a file that
    may still be open for writing) `new` is padded with newlines to the length of `data`, so
    the appended bytes stay where they are and the file never shrinks; otherwise every byte
    past the old end is copied to follow `new` before the file is cut to its new length. When
    `new` is longer than `data` (a credential shorter than REDACTED), the bytes appended after
    the read are written after it."""
    if len(new) > len(data):
        # A credential shorter than REDACTED: the text grows. What was appended after the read
        # is read now and written after `new`.
        fh.seek(len(data))
        tail = fh.read()
        fh.seek(0)
        fh.write(new + tail)
        fh.truncate(len(new) + len(tail))
        fh.flush()
        return
    if keep_length:
        fh.seek(0)
        fh.write(new + b"\n" * (len(data) - len(new)))
        fh.flush()
        return
    fh.seek(0)
    fh.write(new)
    # Every byte past the old end (what was appended meanwhile) moves up to follow `new`; the
    # destination is always behind the source, so nothing is overwritten before it is read.
    src, dst = len(data), len(new)
    while True:
        fh.seek(src)
        tail = fh.read()
        if not tail:
            break
        fh.seek(dst)
        fh.write(tail)
        src += len(tail)
        dst += len(tail)
    fh.truncate(dst)
    fh.flush()


def redact_file(path: str, secrets: Optional[List[str]] = None) -> Optional[bool]:
    """Rewrite the regular file `path` with every credential value replaced (redact_bytes).
    True = rewritten, False = no credential in it, None = it carries one and could not be
    rewritten (not writable). A symlink or a non-regular file is left alone (False).

    The file is rewritten in place and keeps its inode, and what another process appends
    meanwhile is kept (_rewrite_in_place). A file modified in the last LIVE_FILE_WINDOW_S
    seconds is not shortened: the rewritten text is padded with newlines to its old length."""
    import mmap
    import stat as _stat
    secrets = secret_values() if secrets is None else secrets
    if not secrets:
        return False
    try:
        st = os.lstat(path)
    except OSError:
        return False
    if not _stat.S_ISREG(st.st_mode) or st.st_size == 0:
        return False
    needles = [f.encode("utf-8") for f in _redaction_needles(tuple(secrets))]
    try:
        with open(path, "rb") as fh:
            with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                if not any(mm.find(n) != -1 for n in needles):
                    return False
    except (OSError, ValueError):
        return False
    live = (time.time() - st.st_mtime) < LIVE_FILE_WINDOW_S
    try:
        # In place, so the file keeps its inode: a process that still appends to it (the local
        # server's log) keeps writing into the file everyone reads.
        with open(path, "r+b") as fh:
            data = fh.read()
            new = redact_bytes(data, secrets)
            if new == data:
                return False
            _rewrite_in_place(fh, data, new, keep_length=live)
        return True
    except OSError:
        pass
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    new = redact_bytes(data, secrets)
    if new == data:
        return False
    tmp = f"{path}.redact-{os.getpid()}.tmp"      # a read-only file in a writable directory
    try:
        with open(tmp, "wb") as fh:
            fh.write(new)
        os.chmod(tmp, _stat.S_IMODE(st.st_mode))
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return None
    return True


def redact_tree(root: Any, secrets: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """Rewrite every regular file under `root` that carries a credential value (in any form of
    secret_forms) with it replaced by REDACTED: the backstop for files that a suite, a harness
    or a subprocess writes under the results directory without going through gbench's writers
    (agent trajectories, submissions, harness logs, benchmark tool output). Symlinks are not
    followed. A file already checked with the same credentials and unchanged since is skipped;
    a rewritten file is checked again at once (a writer may have appended to it meanwhile) and
    recorded as checked only when a pass finds nothing.
    Returns {"redacted": [...], "unwritable": [...]}: an unwritable file that still holds a key
    is logged as an error naming the file (never the key)."""
    report: Dict[str, List[str]] = {"redacted": [], "unwritable": []}
    if not root:
        return report
    root = str(root)
    if not os.path.isdir(root):
        return report
    secrets = secret_values() if secrets is None else secrets
    if not secrets:
        return report
    fingerprint = hashlib.sha256("\0".join(secrets).encode("utf-8")).hexdigest()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for fname in filenames:
            path = os.path.join(dirpath, fname)
            try:
                st = os.lstat(path)
            except OSError:
                continue
            key = (st.st_mtime_ns, st.st_size, fingerprint)
            with _TREE_LOCK:
                if _TREE_SEEN.get(path) == key:
                    continue
            # The file is recorded as checked under the stat taken BEFORE the scan that found it
            # clean: bytes appended during or after that scan change its (mtime, size), so the
            # next sweep scans it again.
            scanned: Optional[os.stat_result] = st
            outcome = redact_file(path, secrets)
            if outcome:
                report["redacted"].append(path)
                # What another process appended while the file was rewritten is kept as it
                # was written: check the file again (a few passes) before recording it clean.
                for _ in range(3):
                    try:
                        scanned = os.lstat(path)
                    except OSError:
                        scanned = None
                    outcome = redact_file(path, secrets)
                    if not outcome:
                        break
            if outcome is None:
                report["unwritable"].append(path)
                logger.error("A credential value reached %s and the file cannot be rewritten; "
                             "remove it by hand.", path)
                continue
            if outcome or scanned is None:
                continue      # still being written with a credential: the next sweep checks it
            with _TREE_LOCK:
                _TREE_SEEN[path] = (scanned.st_mtime_ns, scanned.st_size, fingerprint)
    if report["redacted"]:
        logger.warning("Replaced credential values in %d file(s) under %s.",
                       len(report["redacted"]), root)
    return report


def _redact_log_record(record: logging.LogRecord) -> None:
    """Replace credential values in a record's message, arguments, exception and stack text.
    Never raises: a failure here must not turn an ordinary log call into an exception."""
    try:
        secrets = secret_values()
        if not secrets:
            return
        try:
            message = record.getMessage()
        except Exception:                     # a malformed record: leave it to the formatter
            return
        redacted = redact_text(message, secrets)
        if redacted != message:
            record.msg, record.args = redacted, None
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = redact_text(record.exc_text, secrets)
        if record.stack_info:
            record.stack_info = redact_text(record.stack_info, secrets)
    except Exception:                                                      # noqa: BLE001
        return


class SecretRedactingFilter(logging.Filter):
    """A logging filter that replaces credential values in a record's message, arguments and
    exception text, so a key never reaches a log line (a subprocess command line in an
    exception, a streamed harness log). Attach it to a handler: handler filters see every
    record the handler emits, including those propagated from child loggers."""

    def filter(self, record: logging.LogRecord) -> bool:
        _redact_log_record(record)
        return True


def install_log_redaction(target: Optional[logging.Logger] = None) -> None:
    """Redact credential values from every log record the process creates: a log-record factory
    that redacts each record as it is made (so a handler a suite attaches to its own logger, a
    per-run agent log file for example, gets redacted records too), plus SecretRedactingFilter
    on every handler of `target` (default: the root logger) that does not have one yet."""
    current = logging.getLogRecordFactory()
    if not getattr(current, "_gbench_redacting", False):
        def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = current(*args, **kwargs)
            _redact_log_record(record)
            return record
        factory._gbench_redacting = True      # type: ignore[attr-defined]
        logging.setLogRecordFactory(factory)
    for handler in (target or logging.getLogger()).handlers:
        if not any(isinstance(f, SecretRedactingFilter) for f in handler.filters):
            handler.addFilter(SecretRedactingFilter())


#: Request fields a suite may set through its extra_payload / sample meta; anything else is not
#: sent. `continue_final_message` / `add_generation_prompt` let a suite prefill the assistant
#: turn as its official harness does (vLLM reads both as top-level fields, and they are also
#: forwarded to /tokenize so the prompt count matches the request).
ALLOWED_API_KEYS = {
    "tools", "tool_choice", "response_format", "top_p", "top_k", "min_p", "presence_penalty",
    "frequency_penalty", "repetition_penalty", "stop", "seed", "stream", "n", "logit_bias",
    # Native reasoning toggle, vision soft-token control, and per-sample generation overrides.
    "chat_template_kwargs", "mm_processor_kwargs", "extra_body", "max_tokens", "temperature",
    "continue_final_message", "add_generation_prompt",
    # An upstream reasoning-model request field (bigcodebench's make_request sends
    # reasoning_effort "medium"); vLLM's chat request accepts it.
    "reasoning_effort",
}

#: The prompt-rendering fields that /tokenize must see too, so a prefilled prompt is counted as sent.
_PREFILL_FIELDS = ("continue_final_message", "add_generation_prompt")


def prefill_fields(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The `continue_final_message` / `add_generation_prompt` fields a request payload sets."""
    if not isinstance(payload, dict):
        return {}
    return {k: payload[k] for k in _PREFILL_FIELDS if k in payload}


#: Set when the endpoint refused `return_token_ids` (HTTP 400/422 naming the field): the rest of
#: the run sends requests without it, and every result records the refusal
#: (server_parser_token_ids).
_TOKEN_IDS_REFUSED: Dict[str, Any] = {}


def _refuse_token_ids(error_text: str) -> None:
    if not _TOKEN_IDS_REFUSED:
        logger.warning("The endpoint refused the return_token_ids field (%s); requests are sent "
                       "without it, and a reply a server parser moved or deleted is recovered "
                       "from the reasoning field (or named a harness error).",
                       redact_text(str(error_text))[:200])
        _TOKEN_IDS_REFUSED.update(refused=True, error=redact_text(str(error_text))[:300])


def server_parser_token_ids_record() -> Dict[str, Any]:
    """Whether the endpoint returned generated token ids for the server-parser policy: always
    requested (return_token_ids), unless the endpoint refused the field."""
    if _TOKEN_IDS_REFUSED:
        return {"requested": True, "available": False,
                "reason": "the endpoint refused return_token_ids: "
                          + str(_TOKEN_IDS_REFUSED.get("error") or "")}
    return {"requested": True, "available": None}


async def _send_single_request(
    session: aiohttp.ClientSession,
    api_url: str,
    model_name: str,
    messages: List[Dict[str, Any]],
    extra_payload: Dict[str, Any],
    semaphore: asyncio.Semaphore,
    pbar: tqdm,
    max_output_tokens: int = 8192,
    temperature: Optional[float] = 0.0,
    thinking: bool = False,
) -> "Reply":
    """Send a single chat completion request.

    Returns a `Reply`: rendered text, raw `tool_calls`, `finish_reason` and `reasoning`.

    The raw `tool_calls` array is kept because the text rendering is lossy (it stringifies
    every argument, hiding an INTEGER answered as "2010"). `finish_reason` and `reasoning`
    are kept because without them a TRUNCATED answer and an answer that lived entirely in
    the model's reasoning channel are both indistinguishable from an ordinary wrong answer
    (22 truncated, 32 empty in the 2026-08-15 sweep, none of it visible).
    """
    payload = {"model": model_name, "messages": messages}
    # None = send no max_tokens: the server grants the rest of the context window (a suite whose
    # official harness sends none and whose window is unknown; see OfficialSampling).
    if max_output_tokens is not None:
        payload["max_tokens"] = max_output_tokens
    # None = send no temperature, so the server's default applies (a suite whose official
    # harness sends none; see OfficialSampling.send_temperature).
    if temperature is not None:
        payload["temperature"] = temperature
    if extra_payload:
        for k, v in extra_payload.items():
            if k in ALLOWED_API_KEYS:
                payload[k] = v
    # Actually toggle the model's native reasoning channel to match --eval-thinking.
    # Only inject when a suite hasn't already set enable_thinking explicitly (e.g. via a
    # sample's extra_payload), so per-suite intent still wins. The same helper sizes the
    # prompt (clamp_to_context), so the count and the request agree.
    payload["chat_template_kwargs"] = effective_chat_template_kwargs(
        payload.get("chat_template_kwargs"), thinking)
    # The generated token ids, so text a server parser moved or deleted can be recovered (the
    # server-parser policy, recover_server_parser_text). Metadata only: vLLM returns them per
    # choice as `token_ids`; it changes nothing in the generation. An endpoint that refuses the
    # field (a strict OpenAI-compatible server) gets requests without it for the rest of the run
    # (_TOKEN_IDS_REFUSED); recovery then falls back to the reasoning field.
    if not _TOKEN_IDS_REFUSED:
        payload["return_token_ids"] = True
    request_thinking = bool(payload["chat_template_kwargs"].get("enable_thinking"))

    last_error: Optional[str] = None
    async with semaphore:
        # Transport retries and context-length retries are counted apart: shrinking max_tokens
        # after a "prompt too long" 400 must not use up the attempts a flaky network needs.
        attempt, context_retries = 0, 0
        while attempt < 3:
            try:
                sent_at = time.monotonic()
                async with session.post(api_url, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        elapsed = max(1e-3, time.monotonic() - sent_at)
                        choice = data["choices"][0]
                        msg = choice["message"]
                        content = msg.get("content") or ""
                        tool_calls = msg.get("tool_calls")
                        finish = choice.get("finish_reason")
                        stop_reason = choice.get("stop_reason")
                        usage_tokens = int((data.get("usage") or {}).get("completion_tokens") or 0)
                        # vLLM exposes the gemma-4 reasoning channel under either key.
                        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or None
                        # The server-parser policy: text the server's reasoning parser moved out
                        # of a no-think reply, or its tool-call parser deleted, is recovered as
                        # the model's output (or named a harness error when it cannot be).
                        parser_event = recovered_from = None
                        if server_parser_signal(content=msg.get("content"), reasoning=reasoning,
                                                tool_calls=tool_calls, thinking=request_thinking,
                                                completion_tokens=usage_tokens):
                            token_ids = choice.get("token_ids")
                            generation = None
                            if isinstance(token_ids, list) and token_ids:
                                generation = await asyncio.to_thread(
                                    generation_text_from_token_ids, token_ids,
                                    decode_generated_token_ids, finish, stop_reason)
                            recovery = recover_server_parser_text(
                                content=msg.get("content"), reasoning=reasoning,
                                tool_calls=tool_calls, thinking=request_thinking,
                                completion_tokens=usage_tokens, generation_text=generation)
                            parser_event = recovery.event
                            recovered_from = recovery.recovered_from
                            if recovery.text is not None:
                                content = recovery.text
                        elif (not tool_calls and not payload.get("tools") and usage_tokens > 0
                              and isinstance(choice.get("token_ids"), list)
                              and choice.get("token_ids")):
                            # A tool-call parser can also delete a native call from a reply that
                            # keeps content or reasoning (server_parser_residue).
                            generation = await asyncio.to_thread(
                                generation_text_from_token_ids, choice.get("token_ids"),
                                decode_generated_token_ids, finish, stop_reason)
                            recovery = server_parser_residue(
                                content=msg.get("content"), reasoning=reasoning,
                                tool_calls=tool_calls, thinking=request_thinking,
                                generation_text=generation)
                            if recovery.event is not None:
                                parser_event = recovery.event
                                recovered_from = recovery.recovered_from
                                content = recovery.text
                        if tool_calls:
                            call_strs = []
                            for tc in tool_calls:
                                fn = tc.get("function", {})
                                fname = fn.get("name", "")
                                fargs_str = fn.get("arguments", "{}")
                                py_kwargs_clean = ""
                                try:
                                    import json
                                    args_dict = json.loads(fargs_str) if isinstance(fargs_str, str) else fargs_str
                                    if isinstance(args_dict, dict):
                                        py_kwargs_clean = fname + "(" + ", ".join(f"{k}={v}" for k, v in args_dict.items()) + ")"
                                except Exception:
                                    py_kwargs_clean = ""
                                call_strs.append(f"{fname}({fargs_str}) {py_kwargs_clean}")
                        rate = round(usage_tokens / elapsed, 1) if usage_tokens else None
                        sent_max = payload.get("max_tokens")
                        if tool_calls:
                            return Reply("\n".join(call_strs) + "\n" + str(content),
                                         tool_calls, finish, reasoning, None,
                                         usage_tokens, stop_reason, rate,
                                         max_tokens_sent=sent_max)
                        return Reply(str(content), None, finish, reasoning, None,
                                     usage_tokens, stop_reason, rate,
                                     parser_event, recovered_from, sent_max)
                    else:
                        err = await resp.text()
                        if (resp.status in (400, 422) and "return_token_ids" in payload
                                and "return_token_ids" in err):
                            # The endpoint refuses the metadata field: resend without it, and
                            # leave it out of the run's later requests.
                            _refuse_token_ids(err)
                            payload.pop("return_token_ids", None)
                            continue
                        if resp.status == 400 and ("maximum context length" in err or "input tokens" in err):
                            # vLLM: "... contains [at least ]N input tokens" (N exact without
                            # "at least"), or, from its character pre-check, "... contains N
                            # characters (more than M characters, which is the upper bound for
                            # K input tokens)", a lower bound of N * K / M prompt tokens.
                            match_input = re.search(r"contains (?:at least )?(\d+) input tokens",
                                                    err)
                            match_chars = re.search(
                                r"contains (\d+) characters \(more than (\d+) characters, which "
                                r"is the upper bound for (\d+) input tokens\)", err)
                            match_max = re.search(r"maximum context length is (\d+) tokens", err)
                            cur = int(payload.get("max_tokens") or max_output_tokens or 0)
                            if match_chars and match_max and not match_input:
                                max_len = int(match_max.group(1))
                                chars, bound, k = (int(g) for g in match_chars.groups())
                                lower = -(-chars * k // max(1, bound))
                                over = (f"{OVER_CONTEXT_ERROR}: the server rejected the prompt as "
                                        f"longer than its {max_len:,}-token context window")
                                if lower >= max_len - 256:
                                    return Reply(None, None, None, None, over)
                                if cur > 256 and context_retries < 8:
                                    payload["max_tokens"] = max(256, min(cur // 4,
                                                                         max_len - lower - 16))
                                    context_retries += 1
                                    continue
                                return Reply(None, None, None, None, over)
                            if match_input and match_max:
                                inp_tokens = int(match_input.group(1))
                                max_len = int(match_max.group(1))
                                over = (f"{OVER_CONTEXT_ERROR}: the server rejected the prompt as "
                                        f"longer than its {max_len:,}-token context window")
                                if inp_tokens >= max_len - 256:
                                    return Reply(None, None, None, None, over)
                                # vLLM stops tokenizing at max_input + 1, so it reports
                                # N = window - max_tokens + 1: only a LOWER bound on the prompt.
                                # Shrink geometrically (floor 256); a rejection at 256 means the
                                # prompt itself does not fit. A server that reports the true
                                # prompt length gets an exact clamp instead.
                                if inp_tokens == max_len - cur + 1:
                                    if cur > 256 and context_retries < 8:
                                        payload["max_tokens"] = max(256, cur // 4)
                                        context_retries += 1
                                        continue
                                    return Reply(None, None, None, None, over)
                                clamped_tokens = max(64, max_len - inp_tokens - 16)
                                if clamped_tokens < cur and context_retries < 8:
                                    payload["max_tokens"] = clamped_tokens
                                    context_retries += 1
                                    continue
                        # Kept as the request's error, so a failed request names its cause.
                        last_error = f"HTTP {resp.status}: {err[:200]}"
                        logger.warning(f"HTTP {resp.status} on eval request: {err[:200]}")
            except asyncio.TimeoutError:
                # Distinct from any other failure: a timeout means the generation was still
                # running, so it is the LONG answers that get dropped - exactly the ones the
                # large budgets exist to capture. Recorded so it is never a mystery failure.
                # The session's own timeout is the one that fired (sized by the caller).
                limit_s = getattr(getattr(session, "timeout", None), "total", None)
                if not isinstance(limit_s, (int, float)) or limit_s <= 0:
                    limit_s = request_timeout_s(max_output_tokens)
                last_error = f"timeout after {int(limit_s)}s"
                if attempt == 2:
                    logger.warning("Request timed out after %ss (max_tokens=%s). Raise "
                                   "GBENCH_REQUEST_TIMEOUT_S / lower --batch-sizes, or the "
                                   "longest answers will be lost.",
                                   int(limit_s), max_output_tokens)
                await asyncio.sleep(0.5 * (attempt + 1))
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt == 2:
                    logger.warning(f"Request exception on attempt {attempt+1}: {e}")
                await asyncio.sleep(0.5 * (attempt + 1))
            attempt += 1
        return Reply(None, None, None, None, last_error)


#: base's statuses of a request that got no reply from the served endpoint after its retries, and
#: the harness error each is (rule: never the model's outcome). An endpoint error, a timeout (the
#: request timeout is sized from the request's budget at an assumed decode rate, so a timeout means
#: the server ran slower than that), and a prompt longer than the served context window (a
#: dataset/serving mismatch). `request_failed_mid_loop`: a tool-loop request that failed after
#: earlier turns, where the loop keeps no forced answer.
ENDPOINT_FAILURE_KINDS: Dict[str, str] = {"FAILED": "endpoint_failure",
                                          "TIMEOUT": "endpoint_timeout",
                                          "OVER_CONTEXT": "over_context"}
MID_LOOP_FAILURE_KIND = "request_failed_mid_loop"
#: The harness-error kinds of a request that got no reply from the served endpoint. A trace's
#: `harness_error` can also be SERVER_PARSER_TEXT_LOST (the reply came, the server's tool-call parser
#: deleted it), so a suite that names "no reply" counts these kinds only.
NO_REPLY_HARNESS_KINDS = frozenset((*ENDPOINT_FAILURE_KINDS.values(), MID_LOOP_FAILURE_KIND))


def endpoint_harness_error(status: Optional[str], error: Any) -> Optional[Dict[str, str]]:
    """The harness error ({"kind", "detail"}) of a request that got no reply, from base's trace
    status (ENDPOINT_FAILURE_KINDS), or None for any other status."""
    kind = ENDPOINT_FAILURE_KINDS.get(str(status or ""))
    if kind is None:
        return None
    return {"kind": kind, "detail": redact_text(str(error or "no reply"))[:300]}


def is_scored(trace: Dict[str, Any]) -> bool:
    """Whether a trace counts in the score: not a judge outage and not left out of it
    (`scoring_excluded`: an endpoint failure, a grader failure, a sample the suite cannot
    measure)."""
    return (str(trace.get("judge_grade") or "") != "JUDGE_OUTAGE"
            and not trace.get("scoring_excluded"))


def tally_batch_scores(sample_traces: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Dict[str, int]]]:
    """Numerator + per-category tally for batch-scored (async_eval_fn) suites.

    Samples the judge could not score (`judge_grade == "JUDGE_OUTAGE"`) or left out of the score
    (`scoring_excluded`: a request that got no reply from the endpoint, which base marks, or a
    sample the suite flagged unmeasurable) are excluded from BOTH the numerator and every
    per-category total, so an unscored sample never surfaces as a category "0/1 failure" that
    contradicts a headline which already excluded it.

    Returns (correct_count, category_stats) where category_stats[cat] = {"correct", "total"}.
    """
    correct_count = 0
    category_stats: Dict[str, Dict[str, int]] = {}
    for trace in sample_traces:
        if not is_scored(trace):
            continue
        is_corr = bool(trace.get("is_correct", False))
        if is_corr:
            correct_count += 1
        cat = trace.get("category")
        if cat:
            st = category_stats.setdefault(cat, {"correct": 0, "total": 0})
            st["total"] += 1
            if is_corr:
                st["correct"] += 1
    return correct_count, category_stats


def classify_scoring_mode(has_fallback: bool, declared: Optional[str],
                          has_async_scorer: bool) -> str:
    """Decide the honest top-level scoring_mode label for a run.

    Precedence:
      1. `judge_fallback` if any sample was scored by the no-judge substring fallback
         (a lower bound, not the canonical metric) - this overrides everything.
      2. the suite's `declared` mode when it told us how it actually scored - e.g.
         code-execution suites (bigcodebench/multipl_e/lcb/ojbench/spider2/cyberseceval)
         batch through async_eval_fn but run tests, not an LLM judge; coco_caption batches
         a deterministic corpus CIDEr metric.
      3. `judge` if a batch scorer is present and the suite did NOT declare otherwise (the
         legacy heuristic - most async scorers are judges).
      4. `deterministic` for a plain per-sample eval_fn.
    """
    if has_fallback:
        return "judge_fallback"
    if declared:
        return declared
    if has_async_scorer:
        return "judge"
    return "deterministic"


def apply_shard(items: List[Any], eval_name: str = "") -> Tuple[List[Any], Optional[str]]:
    """`--shard I/N` (transported as GBENCH_SHARD) applied to an ordered item list.

    For a suite that does not go through run_eval_suite (run_eval_suite applies it itself).
    Pass the FULL, deterministically ordered item list and call this BEFORE the --eval-limit
    draw, so a limited sharded run caps within its shard. Returns (this shard's items,
    "I/N"), or (items, None) when no shard is set; put the label on the result with
    record_shard. A malformed GBENCH_SHARD raises ValueError.
    """
    shard = parse_shard(os.environ.get("GBENCH_SHARD"))
    if not shard:
        return list(items), None
    selected = shard_select(items, shard)
    logger.info("[%s] shard %d/%d: %d of %d items", eval_name or "eval", shard[0], shard[1],
                len(selected), len(items))
    return selected, f"{shard[0]}/{shard[1]}"


def record_shard(result: Dict[str, Any], shard_label: Optional[str]) -> Dict[str, Any]:
    """Record the shard a result covers (`result["shard"] = "I/N"`); a no-op without one."""
    if shard_label:
        result["shard"] = shard_label
    return result


async def _run_suite_async(
    eval_name: str,
    model_name: str,
    base_url: str,
    concurrency: int,
    samples: List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]],
    eval_fn: Optional[Callable[[str, Any], bool]] = None,
    async_eval_fn: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
    declared_scoring_mode: Optional[str] = None,
    thinking: bool = False,
    extra_payload: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = 0.0,
    tool_executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    max_tool_rounds: int = 8,
    attempt_count: int = 1,
    supports_attempts: bool = False,
    tool_loop: Optional[ToolLoopProtocol] = None,
    limit_seed: Optional[str] = None,
    uncapped: bool = False,
    official_budget: bool = False,
    full_set_items: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute evaluation suite samples concurrently with tqdm progress bar.

    `uncapped`: no request carries a max_tokens (the official harness sends none); prompts are
    still counted, so one that cannot fit is refused as over-context. `official_budget`:
    `max_output_tokens` is the suite's official cap, so a reply that stops there is the
    protocol's outcome, not a truncation to warn about (as is a reply that stops at a
    sample-declared max_tokens on a no-think run).
    """
    api_url = f"{base_url.rstrip('/')}/chat/completions"
    # Tool-loop settings: the suite's own protocol when it passes one, else the house loop.
    round_cap = tool_loop.max_rounds if tool_loop else max_tool_rounds
    call_cap = tool_loop.max_tool_calls if tool_loop else None
    forced_final = tool_loop.forced_final if tool_loop else True
    rendered_history = tool_loop.rendered_calls_in_history if tool_loop else True
    invalid_args_as_error = bool(tool_loop and tool_loop.invalid_arguments_as_error)
    wall_s = (tool_loop.max_sample_wall_s if tool_loop else None) or None

    # Sharding for horizontal scale. GBENCH_SHARD="I/N" selects shard I of N over the
    # deterministically-ordered full sample set (round-robin), so a swarm node runs a reproducible,
    # non-overlapping subset that unions back to the full set. Applied BEFORE --eval-limit, so a
    # sharded smoke run caps within the shard. n_scored/n_total on the result reflect the shard.
    # Each sample keeps its position in the full list (a trace's `global_idx`) and a
    # shard-independent `sample_key`, so the shards' traces can be merged and de-duplicated.
    # A suite that selects its own shard first (sampling.presharded_for_base) passes the size of
    # its full ordered set: the list it gives base holds this node's items only. An empty list
    # without it is such a suite's empty shard, whose full set's size base does not know.
    full_count: Optional[int] = (int(full_set_items) if full_set_items is not None
                                 else (len(samples) or None))
    samples, shard_label, global_positions = apply_shard_indexed(samples, eval_name)
    shard_spec = current_shard() if shard_label else None
    if shard_label and not samples:
        # An empty shard (more shards than items in the set this node shards) is an honest empty
        # row, not an error: nothing was generated or scored here.
        reason = (f"empty shard {shard_label}: no items on this node "
                  + (f"({full_count} in the full set)" if full_count is not None
                     else "(the suite selected its own shard; the full set's size was not "
                          "passed to base)"))
        logger.info("[%s] %s", eval_name, reason)
        from .limit_plan import eval_limit_plan as _plan
        from .limit_plan import planned_for as _planned_for
        empty_plan = _planned_for(eval_name) or _plan(
            eval_name, limit, 0, register=False,
            attempts=(1 if async_eval_fn is not None and not supports_attempts
                      else max(1, int(attempt_count or 1))))
        return {
            "eval_limit_plan": empty_plan,
            "benchmark_type": "eval", "eval_name": eval_name, "model_name": model_name,
            "thinking": thinking, "total_questions": 0, "correct_answers": 0,
            "failed_requests": 0, "scoring_excluded": 0, "accuracy": None,
            "accuracy_withheld_reason": reason, "category_accuracy": {}, "sample_traces": [],
            "status": "success", "status_reason": reason, "empty_shard": True,
            "full_set_items": full_count, **shard_identity(shard_spec),
            "leaderboard_comparable": False, "leaderboard_comparable_reason": reason,
        }

    # Enforce limit if provided. Stratified across the samples' own `category` metadata,
    # not a contiguous head: benchmark rows are stored grouped by category, so `[:limit]`
    # returned one category (56 of 93 scored suites collapsed to a single
    # category at --eval-limit 20). Deterministic: seeded on the suite name, or on `limit_seed`
    # (a renamed suite passes its old name, so a subset stays the same rows across the rename).
    # Each kept sample carries its position in the full list with it.
    if limit and limit > 0 and len(samples) > limit:
        tagged = stratified_sample(
            list(zip(samples, global_positions)), limit,
            key_fn=lambda t: ((t[0][2] or {}).get("category")
                              if len(t[0]) > 2 and isinstance(t[0][2], dict) else None),
            seed=limit_seed or eval_name)
        samples = [t[0] for t in tagged]
        global_positions = [t[1] for t in tagged]
    # The identity of each sample that runs (keys are computed only for those).
    sample_meta = [{"global_idx": pos, "sample_key": sample_key(s)}
                   for pos, s in zip(global_positions, samples)]

    # --- repeated attempts (@k) -------------------------------------------------------
    # Most published numbers for small benchmarks are NOT one sample: AIME is avg@4-64,
    # GPQA-Diamond avg@10, ARC-AGI pass@2 by rule, tau-bench pass^k by definition. A
    # single greedy sample is the estimate every lab deliberately avoids - and on
    # 2026-08-17 we watched `astropy__astropy-14309` flip RESOLVED -> unresolved between
    # two identical temperature=0.0 runs, which is exactly the variance those protocols
    # average away.
    #
    # Implementation: replicate each sample k times and let the existing pipeline treat
    # every attempt as an ordinary sample. `accuracy` then IS avg@k for free, because it
    # is (correct attempts / total attempts). pass@k and pass^k are derived afterwards by
    # grouping on `source_sample_idx`.
    attempts_requested = max(1, int(attempt_count or 1))
    attempts = attempts_requested
    attempt_skip_reason = None
    if attempts > 1 and async_eval_fn is not None and not supports_attempts:
        # Batch scorers receive the whole trace list at once and several key it by an id
        # (`preds[instance_id]`, `results[task_id]`). Replicating samples would collapse
        # k attempts into one prediction and hand the same verdict back to all k traces -
        # a fabricated @k. Refuse rather than report a number we did not measure.
        attempt_skip_reason = (
            "this suite scores in a batch (async_eval_fn) and has not declared "
            "supports_attempts=True; running 1 attempt instead of "
            f"{attempts_requested} rather than reporting an @k it did not measure")
        logger.warning("[%s] --attempt-count=%d ignored: %s.",
                       eval_name, attempts_requested, attempt_skip_reason)
        attempts = 1
    # The run's --eval-limit plan (limit_plan): the suite's own when it logged one (a suite with
    # settings or several passes), else this run's samples x attempts, logged here.
    from .limit_plan import eval_limit_plan, planned_for
    limit_plan_record = (planned_for(eval_name)
                         or eval_limit_plan(eval_name, limit, len(samples), attempts=attempts,
                                            register=False))
    #: expanded index -> (index of the sample it is an attempt of, attempt number)
    attempt_of: Dict[int, Tuple[int, int]] = {}
    if attempts > 1:
        expanded = []
        for src_idx, sample in enumerate(samples):
            for a in range(attempts):
                attempt_of[len(expanded)] = (src_idx, a)
                expanded.append(sample)
        logger.info("[%s] %d sample(s) x %d attempts = %d generations; accuracy will be "
                    "reported as avg@%d, with pass@%d and pass^%d alongside.",
                    eval_name, len(samples), attempts, len(expanded), attempts,
                    attempts, attempts)
        samples = expanded
    else:
        attempt_of = {i: (i, 0) for i in range(len(samples))}

    # Global fallback for suites with no floor entry. Raised from 8192/16384: the sweep
    # truncated 10 suites that ran at the old non-thinking default, and a thinking run
    # spends most of its budget in the reasoning channel before the answer starts (the
    # whole completion shares one max_tokens).
    effective_max_tokens = (
        None if uncapped
        else max_output_tokens if max_output_tokens is not None
        else (32768 if thinking else 16384)
    )
    if uncapped and not thinking:
        # The official harness sends no max_tokens: a per-request cap declared as well would
        # be sent while the result says none was.
        declared_caps = sorted({str(p["max_tokens"]) for p in
                                [extra_payload or {}] + [s[2] for s in samples if len(s) > 2]
                                if isinstance(p, dict) and p.get("max_tokens") is not None})
        if declared_caps:
            raise ValueError(
                f"[{eval_name}] the official sampling sends no max_tokens "
                f"(send_max_tokens=False), but the suite declares max_tokens "
                f"{', '.join(declared_caps)} per request; declare the cap in "
                "OfficialSampling.max_tokens or send none")
    # The request timeout covers the longest answer a request may produce: its budget, or the
    # whole context window when no budget is sent (UNCAPPED_TIMEOUT_TOKENS when the window is
    # not known either).
    timeout_budget = (effective_max_tokens if effective_max_tokens is not None
                      else get_run_knob("max_model_len") or UNCAPPED_TIMEOUT_TOKENS)

    tool_rounds_used: Dict[int, int] = {}
    #: samples where the tool budget ran out and a final answer was forced
    forced_finals: Dict[int, bool] = {}
    #: why each tool-loop sample stopped (answered / max_rounds / max_tool_calls / ...)
    loop_ends: Dict[int, str] = {}
    tool_calls_used: Dict[int, int] = {}
    sent_budgets: Dict[int, Any] = {}
    #: samples whose first request carried an official cap (a declared per-request max_tokens on
    #: a no-think run, or the suite's official cap): the cap each was sent
    official_caps: Dict[int, int] = {}
    #: --thinking samples whose declared max_tokens was replaced by the run's budget: the cap
    lifted_caps: Dict[int, int] = {}
    concurrency = max(1, int(concurrency or 1))
    semaphore = asyncio.Semaphore(concurrency)
    if aiohttp is None:
        raise RuntimeError(
            "The 'aiohttp' package is required for native asynchronous evaluation runners.\n"
            "Install dependencies via: pip install aiohttp (or pip install -e .)"
        )
    connector = aiohttp.TCPConnector(limit=concurrency + 20)
    timeout_s = request_timeout_s(timeout_budget)
    if timeout_s > REQUEST_TIMEOUT_FLOOR_S:
        logger.info("[%s] per-request timeout %ds (max_tokens=%s at >=%.0f tok/s)",
                    eval_name, timeout_s, effective_max_tokens, MIN_DECODE_TOK_S)
    timeout = aiohttp.ClientTimeout(total=timeout_s)

    correct_count = 0
    total_count = len(samples)
    failed_requests = 0
    empty_responses = 0
    truncated_responses = 0
    judge_failures = 0
    #: the samples counted in judge_failures (named there once, not again as scoring_excluded)
    judge_failed_idxs: set = set()
    judge_outages = 0
    scoring_excluded = 0
    health_counts: Dict[str, int] = {}
    category_stats = {}
    #: samples counted under empty_responses (a batch scorer may still find the empty reply
    #: to be the correct answer; see `empty_is_answer` below)
    empty_idxs: set = set()

    #: samples stopped by the protocol's per-sample wall clock (max_sample_wall_s)
    wall_limited: set = set()
    #: tool calls whose arguments were not valid JSON, fed back as an error (per sample)
    argument_errors: Dict[int, int] = {}
    #: tool-loop samples: (their conversation, the number of prompt messages), so each trace keeps
    #: the loop's turns after the prompt (`tool_loop_transcript`), a stopped sample's included
    loop_convos: Dict[int, Tuple[List[Dict[str, Any]], int]] = {}
    #: tool loops without a forced answer whose last request failed after earlier turns: the error
    #: (the loop's last content is kept on the trace, but it is not the model's answer)
    mid_loop_errors: Dict[int, Optional[str]] = {}
    #: samples whose request got no reply from the served endpoint (ENDPOINT_FAILURE_KINDS, or a
    #: tool-loop request that failed mid loop): {sample: (base status, harness error)}. Harness
    #: errors: left out of the score, counted and named, never the model's outcome.
    harness_errors: Dict[int, Tuple[str, Dict[str, str]]] = {}

    with tqdm(total=total_count, desc=f"Eval [{eval_name.upper()}]") as pbar:
        async def _fetch_sample(idx, messages, gold_answer, sample_extra_payload):
            """One sample, under the protocol's per-sample wall clock when it sets one."""
            if not wall_s:
                return await _fetch_sample_body(idx, messages, gold_answer, sample_extra_payload,
                                                semaphore)
            clock = _ClockedSemaphore(semaphore)
            body = asyncio.ensure_future(_fetch_sample_body(
                idx, messages, gold_answer, sample_extra_payload, clock))
            waiter = asyncio.ensure_future(clock.started.wait())
            try:
                await asyncio.wait({body, waiter}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                waiter.cancel()
            while not body.done():
                remaining = wall_s - clock.active_s()
                if remaining > 0:
                    await asyncio.wait({body}, timeout=remaining)
                    continue
                body.cancel()
                await asyncio.wait({body})          # let it unwind; its own error is not raised
                wall_limited.add(idx)
                loop_ends[idx] = "wall_clock_limit"
                sample_cat = (sample_extra_payload.get("category")
                              if isinstance(sample_extra_payload, dict)
                              else sample_extra_payload if isinstance(sample_extra_payload, str)
                              else None)
                error = (f"{WALL_CLOCK_LIMIT_ERROR}: the sample ran past its {wall_s:g} s "
                         f"wall-clock limit and was stopped")
                return (idx, Reply(None, None, None, None, error), gold_answer, sample_cat,
                        messages, sample_extra_payload, None)
            return body.result()

        async def _fetch_sample_body(idx, messages, gold_answer, sample_extra_payload, semaphore):
            # Merge suite-level extra_payload with the sample's request fields: its meta without
            # the identity and category keys (SAMPLE_META_BOOKKEEPING_KEYS), which upstream never
            # sends and a strict OpenAI-compatible server may reject as unknown fields
            request_payload = {}
            if extra_payload:
                request_payload.update(extra_payload)
            sample_cat = None
            if isinstance(sample_extra_payload, dict):
                request_payload.update({k: v for k, v in sample_extra_payload.items()
                                        if k not in SAMPLE_META_BOOKKEEPING_KEYS})
                sample_cat = sample_extra_payload.get("category")
            elif isinstance(sample_extra_payload, str):
                sample_cat = sample_extra_payload

            # Fit the request to the context window before sending, rather than paying a
            # 400 + retry (or three silent failures when the prompt alone is over the limit).
            # A sample's own meta may carry `max_tokens` (culer 512, screenspot 128,
            # mmmu_pro 8192). That is legitimate - a per-sample override - but it lands in
            # the payload AFTER the resolved budget, so the suite-level number in the logs
            # was not what the server saw: culer logged `max_tokens=16384` while every
            # request sent 512, and its "8/20 truncated" was against 512, not 16384.
            # Resolve it here so one value is used, reported and clamped: it leaves the payload,
            # and the clamped value is what the request sends (the payload copy would otherwise
            # overwrite the clamp on the wire).
            declared = request_payload.pop("max_tokens", None)
            if not (isinstance(declared, int) and not isinstance(declared, bool)
                    and declared > 0):
                declared = None
            if declared is None:
                budget = effective_max_tokens
            elif thinking:
                # The token-limit policy: a declared cap is an official cap, written for
                # non-reasoning generation; a --thinking request gets the run's budget instead,
                # whether it is larger or smaller (as for OfficialSampling.max_tokens).
                lifted_caps[idx] = declared
                budget = effective_max_tokens
            else:
                budget = declared
            # Counted with the tools and chat_template_kwargs the request will carry, so a
            # server count is exact for the whole request (see clamp_to_context).
            request_ctk = request_payload.get("chat_template_kwargs")
            turn_ctk = effective_chat_template_kwargs(request_ctk, thinking)
            # A prefilled assistant turn (continue_final_message) is counted as the server
            # renders it; it applies to the sample's first request only.
            prefill = prefill_fields(request_payload)
            sample_max_tokens, oversize = await clamp_to_context_async(
                messages, budget, request_payload.get("tools"), turn_ctk,
                **({"prefill": prefill} if prefill else {}))
            if budget is None:
                # No max_tokens is sent (the official harness sends none); the count above only
                # refuses a prompt that cannot fit.
                sample_max_tokens = None
            sent_budgets[idx] = sample_max_tokens
            if sample_max_tokens is not None and (
                    (declared is not None and idx not in lifted_caps and budget == declared)
                    or (declared is None and official_budget)) and sample_max_tokens == budget:
                official_caps[idx] = budget
            if oversize:
                if pbar is not None:
                    pbar.update(1)
                return (idx, Reply(None, None, None, None), gold_answer, sample_cat,
                        messages, sample_extra_payload, oversize)

            # Agentic loop. A web-research suite cannot be answered in one shot: the model
            # asks for a search, reads the results, and asks again. Without it `gaia` and
            # `deepsearch_qa` reply "I do not have access to the internet" and score a
            # structural 0. Only runs when the suite supplies an executor; single-turn
            # suites take the same path they always did (one request, no extra state).
            convo = list(messages)
            if tool_executor is not None:
                loop_convos[idx] = (convo, len(messages))
            rounds = 0
            no_tools_payload = {k: v for k, v in request_payload.items()
                                if k not in ("tools", "tool_choice") and k not in _PREFILL_FIELDS}
            followup_payload = {k: v for k, v in request_payload.items()
                                if k not in _PREFILL_FIELDS}

            async def _forced_final_answer():
                """One last turn with the tools withdrawn so the model commits to an answer.
                Returns the reply, or None when the transcript no longer fits the window."""
                convo.append({"role": "user", "content": (
                    "No further tool calls are possible. Answer now, using only what "
                    "you already have, and state your final answer explicitly.")})
                final_max, oversize = await clamp_to_context_async(
                    convo, budget, None, effective_chat_template_kwargs(request_ctk, thinking))
                if oversize:
                    return None
                final = await _send_single_request(
                    session=session, api_url=api_url, model_name=model_name,
                    messages=convo, extra_payload=no_tools_payload,
                    semaphore=semaphore, pbar=None,
                    max_output_tokens=final_max if budget is not None else None,
                    temperature=temperature, thinking=thinking,
                )
                forced_finals[idx] = True
                # Withdrawing the tools does not stop gemma-4 emitting a tool call - it just
                # leaves the call nowhere to go, so vLLM's parser strips it out of `content`
                # and the turn returns NOTHING. Measured on the 2026-08-20 mcp_atlas run: 35
                # of 152 forced finals (23%) came back `output_discarded`, i.e. 7% of the
                # suite scored zero on an answer the model had already paid to generate. The
                # prose instruction above is evidently not enough on its own; say it in the
                # imperative and give it one more turn.
                return await _retry_discarded_final(
                    reply=final, convo=convo, session=session, api_url=api_url,
                    model_name=model_name, payload=no_tools_payload,
                    semaphore=semaphore, budget=budget, temperature=temperature)

            while True:
                reply = await _send_single_request(
                    session=session,
                    api_url=api_url,
                    model_name=model_name,
                    messages=convo,
                    extra_payload=request_payload if rounds == 0 else followup_payload,
                    semaphore=semaphore,
                    pbar=pbar if rounds == 0 else None,
                    max_output_tokens=sample_max_tokens,
                    temperature=temperature,
                    thinking=thinking,
                )
                if tool_executor is None or not reply.tool_calls:
                    # A turn that ran out of budget INSIDE the reasoning channel leaves no
                    # content and no tool call, so this break used to end the sample with
                    # nothing at all - the judge saw an empty response and recorded
                    # JUDGE_FAILED, which is indistinguishable from an outage. Measured
                    # 2026-08-18 on browsecomp: 2/3 samples reasoned 28,538 and 25,618
                    # characters into the 8192-token turn and emitted not one character of
                    # answer. Same remedy as running out of rounds, and for the same
                    # reason: ask once more, with thinking OFF so the model has to open the
                    # content channel, and grade whatever it commits to. A wrong answer is
                    # a real result; silence is not.
                    if (forced_final and tool_executor is not None
                            and reply.finish_reason == "length"
                            and not (reply.text or "").strip()):
                        convo.append({"role": "user", "content": (
                            "You ran out of room to think. Do not reason further. State "
                            "your best answer now from what you already have, in the "
                            "requested format.")})
                        final_max, oversize = await clamp_to_context_async(
                            convo, budget, None, effective_chat_template_kwargs(request_ctk, False))
                        if not oversize:
                            reply = await _send_single_request(
                                session=session, api_url=api_url, model_name=model_name,
                                messages=convo, extra_payload={
                                    k: v for k, v in request_payload.items()
                                    if k not in ("tools", "tool_choice")
                                    and k not in _PREFILL_FIELDS},
                                semaphore=semaphore, pbar=None,
                                max_output_tokens=(final_max if budget is not None
                                                   else None),
                                temperature=temperature, thinking=False,
                            )
                            forced_finals[idx] = True
                            reply = await _retry_discarded_final(
                                reply=reply, convo=convo, session=session, api_url=api_url,
                                model_name=model_name, payload=no_tools_payload,
                                semaphore=semaphore, budget=budget,
                                temperature=temperature)
                    if tool_executor is not None:
                        loop_ends[idx] = ("forced_final" if forced_finals.get(idx)
                                          else "request_failed" if reply.text is None
                                          else "answered")
                    break
                if round_cap is not None and rounds >= round_cap:
                    # The house loop's cap counts executed tool rounds, so this turn's calls
                    # are dropped. (A protocol suite stops right after its last allowed turn,
                    # below, and only gets here with max_rounds=0.)
                    if not forced_final:
                        # The suite's reference loop just stops here; the answer is taken
                        # from the transcript after the loop.
                        loop_ends[idx] = "max_rounds"
                        break
                    # Out of rounds while the model is still calling tools. Breaking here
                    # returns a reply whose only content is a TOOL CALL, which then gets
                    # flattened to text and graded as the answer - measured 2026-08-18 on
                    # browsecomp, where a 183-character `web_search({...})` was scored
                    # against the gold. Give it one final turn with the tools withdrawn so
                    # it commits to an answer; a wrong answer is a real result, a dangling
                    # tool call is not.
                    convo.append({"role": "assistant",
                                  "content": ((reply.text or "") if rendered_history
                                              else reply_content(reply)),
                                  "tool_calls": reply.tool_calls})
                    forced = await _forced_final_answer()
                    if forced is not None:
                        reply = forced
                    loop_ends[idx] = "forced_final" if forced_finals.get(idx) else "context_exhausted"
                    break
                rounds += 1
                convo.append({"role": "assistant",
                              "content": ((reply.text or "") if rendered_history
                                          else reply_content(reply)),
                              "tool_calls": reply.tool_calls})
                for call in reply.tool_calls:
                    if call_cap is not None and tool_calls_used.get(idx, 0) >= call_cap:
                        # As in mcp-atlas agent-eval.ts: calls past the cap are not executed.
                        break
                    tool_calls_used[idx] = tool_calls_used.get(idx, 0) + 1
                    fn = (call.get("function") or {})
                    raw_args = fn.get("arguments")
                    try:
                        parsed = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except Exception as e:
                        if invalid_args_as_error:
                            # The protocol's loop does not call the tool: it answers with the
                            # parse error.
                            argument_errors[idx] = argument_errors.get(idx, 0) + 1
                            first = (str(e).splitlines() or [type(e).__name__])[0]
                            convo.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                          "name": fn.get("name", ""), "content": f"Error: {first}"})
                            continue
                        parsed = {}
                    try:
                        # Stateful tool environments (a notebook kernel, a scratch
                        # filesystem) need to keep state PER SAMPLE - concurrent samples
                        # sharing one dict would interleave their cells. Executors that
                        # want it declare a third parameter; the two-argument ones are
                        # unaffected.
                        if _executor_wants_session(tool_executor):
                            out = await tool_executor(fn.get("name", ""), parsed, idx)
                        else:
                            out = await tool_executor(fn.get("name", ""), parsed)
                    except Exception as e:                     # a broken tool is reported,
                        out = json.dumps({"error": f"tool raised {type(e).__name__}: {e}"})
                    convo.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                  "name": fn.get("name", ""), "content": str(out)})
                if call_cap is not None and tool_calls_used.get(idx, 0) >= call_cap:
                    # ...and no further model turn is requested once the cap is reached.
                    loop_ends[idx] = "max_tool_calls"
                    break
                if tool_loop is not None and round_cap is not None and rounds >= round_cap:
                    # A protocol's cap counts MODEL TURNS, as its reference loop does
                    # (smolagents max_steps, mcp-atlas `for i < maxTurns`): that was the last
                    # tool-enabled turn, so no further one is requested.
                    if not forced_final:
                        loop_ends[idx] = "max_rounds"
                        break
                    forced = await _forced_final_answer()
                    if forced is not None:
                        reply = forced
                    loop_ends[idx] = "forced_final" if forced_finals.get(idx) else "context_exhausted"
                    break
                # the follow-up must fit too, now that the transcript has grown
                sample_max_tokens, oversize = await clamp_to_context_async(
                    convo, budget, followup_payload.get("tools"), turn_ctk)
                if budget is None:
                    sample_max_tokens = None
                if oversize:
                    loop_ends[idx] = "context_exhausted"
                    break
            if rounds:
                tool_rounds_used[idx] = rounds
            if tool_executor is not None and not forced_final:
                # No forced answer: grade the last assistant message that has content, as
                # scaleapi/mcp-atlas run_eval.py does. The final turn may be a dangling
                # call (cap hit), an empty turn, or a failed request.
                final = reply_content(reply) if reply.text is not None else ""
                if not final.strip():
                    final = next((m["content"] for m in reversed(convo[len(messages):])
                                  if m.get("role") == "assistant"
                                  and str(m.get("content") or "").strip()), final)
                if reply.text is None and final.strip():
                    # The loop's last request failed at the endpoint: the content kept is an
                    # earlier turn's, not the model's answer to the task, so the sample is a
                    # harness error (kept on the trace with the request's error).
                    mid_loop_errors[idx] = reply.error
                if reply.text is not None or final.strip():
                    reply = reply._replace(text=final)
            if (reply.max_tokens_sent is not None and not tool_rounds_used.get(idx)
                    and sent_budgets.get(idx) is not None
                    and reply.max_tokens_sent != sent_budgets[idx]):
                # A context-length retry lowered max_tokens: the reply answers that budget, which
                # is not the official cap (record_budget_cutoffs counts a length stop at it).
                sent_budgets[idx] = reply.max_tokens_sent
                official_caps.pop(idx, None)
            return (idx, reply, gold_answer, sample_cat, messages, sample_extra_payload, None)

        sample_traces = []
        completed = 0
        # The endpoint key (VLLM_API_KEY, else OPENAI_API_KEY) on every request base sends.
        async with aiohttp.ClientSession(connector=connector, timeout=timeout,
                                         headers=endpoint_auth_headers()) as session:
            coroutines = [
                _fetch_sample(idx, sample[0], sample[1], sample[2] if len(sample) > 2 else {})
                for idx, sample in enumerate(samples)
            ]
            for future in asyncio.as_completed(coroutines):
                (idx, reply, gold_answer, category, messages,
                 sample_extra_payload, skip_reason) = await future
                resp_text, tool_calls = reply.text, reply.tool_calls
                health = classify_reply(reply)
                health_counts[health] = health_counts.get(health, 0) + 1
                completed += 1
                pbar.update(1)

                if resp_text is None and health == "wall_clock_limit":
                    # The protocol's per-sample wall clock stopped it: a 0, as the reference
                    # harness scores it, counted apart from failed requests.
                    is_correct = False
                    status = "WALL_CLOCK_LIMIT"
                    skip_reason = reply.error
                elif resp_text is None:
                    failed_requests += 1
                    is_correct = False
                    # Record WHY. A bare `request_failed` with `error: None` is what mrcr's
                    # six over-context rows looked like on the 2026-08-15 sweep.
                    if skip_reason:
                        status = "OVER_CONTEXT"
                    elif str(reply.error or "").startswith(OVER_CONTEXT_ERROR):
                        status = "OVER_CONTEXT"
                        skip_reason = reply.error
                    elif health == "request_timeout":
                        status = "TIMEOUT"
                        skip_reason = reply.error
                    else:
                        status = "FAILED"
                        skip_reason = reply.error
                    # No reply from the served endpoint after the retries: a harness error, left
                    # out of the score (never the model's outcome), counted and named.
                    harness_errors[idx] = (status, endpoint_harness_error(status, skip_reason))
                elif idx in mid_loop_errors:
                    # A tool loop without a forced answer whose last request failed after earlier
                    # turns: the content kept is an earlier turn's, not the model's answer. A
                    # harness error like a failed first request; the error stays on the trace.
                    failed_requests += 1
                    is_correct = False
                    status = "PENDING_JUDGE" if async_eval_fn is not None else "OK"
                    skip_reason = mid_loop_errors[idx] or "a request of the tool loop failed"
                    harness_errors[idx] = (MID_LOOP_FAILURE_KIND, {
                        "kind": MID_LOOP_FAILURE_KIND,
                        "detail": redact_text(str(skip_reason))[:300]})
                elif reply.parser_event == SERVER_PARSER_TEXT_LOST:
                    # The server's tool-call parser deleted the reply and the text could not be
                    # recovered from token ids (the server-parser policy): a harness error, left
                    # out of the score, never the model's outcome, counted and named.
                    is_correct = False
                    status = "SERVER_PARSER_TEXT_LOST"
                    skip_reason = SERVER_PARSER_TEXT_LOST_DETAIL.format(
                        n=int(reply.completion_tokens or 0))
                    harness_errors[idx] = (status, {"kind": SERVER_PARSER_TEXT_LOST,
                                                    "detail": skip_reason[:300]})
                elif health in ("truncated", "looping"):
                    # The generation hit the output-token budget mid-answer. Scored on
                    # what arrived (it may still be right), but counted so the run says so
                    # instead of reporting a quietly depressed accuracy.
                    truncated_responses += 1
                    cleaned_pred = strip_thinking_tags(resp_text)
                    if async_eval_fn is not None:
                        is_correct, status = False, "PENDING_JUDGE"
                    else:
                        is_correct = _apply_eval_fn(eval_fn, cleaned_pred, gold_answer, tool_calls)
                        if is_correct:
                            correct_count += 1
                        status = "TRUNCATED"
                elif health == "empty_reasoned":
                    # The model wrote its answer inside its thinking channel and stopped there
                    # (ANSWER_IN_THINKING_POLICY): the model's outcome, scored as an upstream
                    # harness reading `content` scores an empty answer (a batch scorer may still
                    # find an empty answer right: empty_is_answer), not a failed request.
                    is_correct = False
                    status = "EMPTY_RESPONSE"
                elif not str(resp_text).strip() and not tool_calls:
                    # The server answered 200 with empty `content`. That is not a wrong
                    # answer, it is a request that produced nothing - most often the
                    # generation hitting the output-token budget mid-reasoning. Scored as
                    # incorrect either way, but counted so the run reports
                    # `completed_with_errors` instead of a quietly depressed accuracy.
                    empty_responses += 1
                    failed_requests += 1
                    empty_idxs.add(idx)
                    is_correct = False
                    status = "EMPTY_RESPONSE"
                else:
                    cleaned_pred = strip_thinking_tags(resp_text)
                    if async_eval_fn is not None:
                        is_correct = False
                        status = "PENDING_JUDGE"
                    else:
                        is_correct = _apply_eval_fn(eval_fn, cleaned_pred, gold_answer, tool_calls)
                        if is_correct:
                            correct_count += 1
                        status = "OK"

                if category and async_eval_fn is None and idx not in harness_errors:
                    if category not in category_stats:
                        category_stats[category] = {"correct": 0, "total": 0}
                    category_stats[category]["total"] += 1
                    if is_correct:
                        category_stats[category]["correct"] += 1

                _src = attempt_of.get(idx, (idx, 0))[0]
                _loop_text, _loop_channel = looping_channel(resp_text, reply.reasoning)
                sample_traces.append({
                    "sample_idx": idx,
                    "source_sample_idx": _src,
                    "attempt_idx": attempt_of.get(idx, (idx, 0))[1],
                    # the sample's position in the suite's full ordered list (before --shard and
                    # --eval-limit) and its shard-independent id (sample_key)
                    **(sample_meta[_src] if 0 <= _src < len(sample_meta) else {}),
                    "category": category,
                    "messages": _sanitize_for_trace(messages),
                    "extra_payload": _sanitize_for_trace(sample_extra_payload) if sample_extra_payload else None,
                    "gold_answer": gold_answer,
                    "response_text": resp_text,
                    "tool_calls": tool_calls,
                    # Diffable form of the same thing; see `normalize_function_calls`.
                    "emitted_function_calls": normalize_function_calls(tool_calls),
                    "finish_reason": reply.finish_reason,
                    "health": health,
                    # the answer written inside the thinking channel (ANSWER_IN_THINKING_POLICY)
                    **({"answer_in_thinking": True} if health == "empty_reasoned" else {}),
                    "reasoning_chars": len(reply.reasoning or ""),
                    "reasoning": reply.reasoning,   # thinking content (None on no-think)
                    # from the content, or from the reasoning when only it repeats
                    # (looping_channel); `repetition_channel` says which
                    "repetition_run": repetition_run(_loop_text),
                    "repetition_onset": (repetition_onset(_loop_text)
                                         or (degenerate_char_run(_loop_text)
                                             if health == "looping" else None)),
                    "repetition_channel": _loop_channel,
                    "max_tokens_sent": sent_budgets.get(idx),
                    **({"official_cap_not_applied": lifted_caps[idx]}
                       if idx in lifted_caps else {}),
                    "completion_tokens": reply.completion_tokens,
                    "decode_tok_s": reply.decode_tok_s,
                    "stop_reason": reply.stop_reason,
                    "error": skip_reason,
                    "is_correct": is_correct,
                    "status": status,
                    **({"tool_loop_end": loop_ends.get(idx),
                        "tool_calls_executed": tool_calls_used.get(idx, 0)}
                       if tool_executor is not None else {}),
                    **({"tool_argument_errors": argument_errors.get(idx, 0)}
                       if invalid_args_as_error else {}),
                    # the server-parser policy's event for the scored reply (recovered text is
                    # the response_text; `reasoning` keeps the server's field as it came)
                    **({"server_parser": {"event": reply.parser_event,
                                          "recovered_from": reply.parser_recovered_from}}
                       if reply.parser_event else {}),
                })
                if idx in harness_errors:
                    sample_traces[-1]["harness_error"] = dict(harness_errors[idx][1])
                    sample_traces[-1]["scoring_excluded"] = True
                if idx in loop_convos:
                    # The tool loop's turns after the prompt: the assistant tool calls (names and
                    # arguments), the tool replies and any forced-answer prompt; secrets are
                    # redacted from the whole result by run_eval_suite.
                    convo_ref, n_prompt = loop_convos[idx]
                    sample_traces[-1]["tool_loop_transcript"] = _sanitize_for_trace(
                        list(convo_ref[n_prompt:]))

                if async_eval_fn is None:
                    pbar.set_postfix(
                        correct=f"{correct_count}/{completed} ({correct_count / completed * 100.0:.1f}%)"
                    )

    # Phase 2: Post-generation batch/async judging if provided
    if async_eval_fn is not None:
        await async_eval_fn(sample_traces)
        # A sample stopped by the wall clock scores 0 whatever the batch scorer made of it, as
        # the reference harness scores a task past its timeout. Keyed on the recorded set, not
        # on the trace's status, which scorers rewrite. It also stays in the denominator: a
        # scorer's exclusion markers (judge outage, scoring_excluded, empty_is_answer) are
        # moved aside, and a judge grade the scorer set is kept under `scorer_judge_grade`.
        for t in sample_traces:
            if t.get("sample_idx") in wall_limited:
                t["is_correct"] = False
                t["status"] = "WALL_CLOCK_LIMIT"
                t.pop("scoring_excluded", None)
                t.pop("empty_is_answer", None)
                if t.get("judge_grade") is not None:
                    t["scorer_judge_grade"] = t.pop("judge_grade")
        # A request that got no reply from the endpoint stays a harness error whatever the batch
        # scorer made of it (scorers rewrite statuses, and some grade a missing reply): out of the
        # score, never the model's outcome. Keyed on the recorded set, as above.
        for t in sample_traces:
            recorded = harness_errors.get(t.get("sample_idx"))
            if recorded is not None:
                t["is_correct"] = False
                t["scoring_excluded"] = True
                t.pop("empty_is_answer", None)
                if not t.get("harness_error"):
                    t["harness_error"] = dict(recorded[1])
        # An empty reply can BE the answer (BFCL irrelevance: the right response is no call).
        # A trace the scorer marks `empty_is_answer` and scored correct is an answered sample,
        # not a failed request, so it leaves empty_responses / failed_requests.
        empty_answers = sum(1 for t in sample_traces
                            if t.get("sample_idx") in empty_idxs and t.get("empty_is_answer")
                            and t.get("is_correct"))
        empty_responses -= empty_answers
        failed_requests -= empty_answers
        correct_count = 0
        category_stats = {}
        # A grader that never returned a verdict is a harness failure, not a "no". Without
        # this the run still reported `success` and the unjudged samples simply counted as
        # incorrect, so a judge outage looked like a low score.
        # Only an explicit grader marker counts. `status == "FAILED"` must NOT be treated as
        # a judge failure: execution-based suites use it for an ordinary wrong answer (lcb
        # sets it whenever a solution does not pass its tests), so inferring from it counted
        # every incorrect sample as a harness error and downgraded clean runs to
        # `completed_with_errors`.
        # Only markers that mean THE JUDGE failed. An empty response is not one: the
        # judge is never called, and the sample is already counted under
        # `empty_responses` / `failed_requests` / `discarded_outputs`. Adding it here
        # would blame grading for a generation problem and count the same sample a
        # fourth time.
        JUDGE_FAILURE_GRADES = ("judge_error", "unparsed", "JUDGE_FAILED")
        # (A request that got no reply from the endpoint is counted there, not as a judge event.)
        judge_failed = [t for t in sample_traces
                        if str(t.get("judge_grade") or "") in JUDGE_FAILURE_GRADES
                        and t.get("sample_idx") not in harness_errors]
        judge_failures = len(judge_failed)
        judge_failed_idxs = {t.get("sample_idx") for t in judge_failed}
        failed_requests += judge_failures
        # JUDGE_OUTAGE is a distinct outcome: the judge (Gemini) cascade was exhausted for
        # this sample - an infrastructure outage, NOT a model failure. It is excluded from
        # BOTH the accuracy numerator and denominator (below) and reported on its own, so an
        # outage can never masquerade as a wrong answer. Kept separate from
        # JUDGE_FAILURE_GRADES (parse/format errors that count as failed_requests).
        judge_outages = sum(
            1 for t in sample_traces
            if str(t.get("judge_grade") or "") == "JUDGE_OUTAGE"
            and t.get("sample_idx") not in harness_errors)
        # A suite may declare a sample UNMEASURABLE for this endpoint (e.g. gdpval: a task
        # whose whole rubric grades a produced file a text model cannot emit). Like a judge
        # outage, it is excluded from the numerator, the denominator AND the per-category
        # accuracy - counting it as a category "0/1 failure" contradicts a headline that says
        # it was not scored at all.
        scoring_excluded = sum(1 for t in sample_traces if t.get("scoring_excluded"))
        correct_count, category_stats = tally_batch_scores(sample_traces)

    sample_traces.sort(key=lambda x: x["sample_idx"])

    # Judge outages, requests that got no reply from the endpoint and samples the suite left out
    # of the score are excluded from the denominator: accuracy is over the samples that were
    # actually scored. Counted per trace, so a trace with two markers is left out once. Nothing
    # scored -> no accuracy (None), never 0.
    scoring_excluded = sum(1 for t in sample_traces if t.get("scoring_excluded"))
    scored_count = sum(1 for t in sample_traces if is_scored(t))
    accuracy = (correct_count / scored_count * 100.0) if scored_count > 0 else None

    # --- @k metrics -------------------------------------------------------------------
    # `accuracy` above is already avg@k (correct attempts / scored attempts). The other two
    # shapes are what the remaining protocols ask for and cannot be derived from it:
    #   pass@k  - at least one attempt correct   (ARC-AGI's rule, OJBench, aider)
    #   pass^k  - EVERY attempt correct          (tau-bench's reliability metric)
    # Over the scored attempts only: an attempt that is a judge outage or was left out of the
    # score (a harness error) is not a failed attempt; a sample with no scored attempt is not
    # counted, and with none at all the rates are None. pass^k needs all k attempts: it is over
    # the samples with k scored attempts (a sample whose other attempts were harness errors
    # cannot show k correct), and those with fewer are counted apart.
    attempts_report = None
    if attempts > 1:
        by_source: Dict[int, List[bool]] = {}
        for t in sample_traces:
            if not is_scored(t):
                continue
            by_source.setdefault(t.get("source_sample_idx", t["sample_idx"]), []).append(
                bool(t.get("is_correct")))
        n_src = len(by_source)
        any_ok = sum(1 for v in by_source.values() if any(v))
        complete = [v for v in by_source.values() if len(v) >= attempts]
        all_ok = sum(1 for v in complete if all(v))
        short = n_src - len(complete)
        # How often the k attempts disagreed with each other. This is the run's own
        # measurement of its reproducibility, and the reason to use @k at all.
        unstable = sum(1 for v in by_source.values() if any(v) and not all(v))
        attempts_report = {
            "attempts_requested": attempts_requested,
            "attempts_per_sample": attempts,
            "samples": n_src,
            "generations": total_count,
            "avg_at_k": round(accuracy, 2) if accuracy is not None else None,
            "pass_at_k": round(any_ok / n_src * 100.0, 2) if n_src else None,
            "pass_hat_k": round(all_ok / len(complete) * 100.0, 2) if complete else None,
            "pass_hat_k_samples": len(complete),
            "samples_with_fewer_scored_attempts": short,
            "unstable_samples": unstable,
        }
        if short:
            attempts_report["fewer_scored_attempts_note"] = (
                f"{short} sample(s) have fewer than {attempts} scored attempts (the others were "
                "harness errors or judge outages): counted in pass@k and avg@k over the "
                "attempts that were scored, left out of pass^k")
        if n_src:
            logger.info(
                "[%s] avg@%d %.2f%% | pass@%d %.2f%% | pass^%d %s over %d sample(s); "
                "%d sample(s) gave different verdicts across attempts.",
                eval_name, attempts, accuracy, attempts, any_ok / n_src * 100.0,
                attempts, (f"{all_ok / len(complete) * 100.0:.2f}% over {len(complete)} "
                           "sample(s) with every attempt scored" if complete else "-"),
                n_src, unstable)
    elif attempts_requested > 1:
        attempts_report = {
            "attempts_requested": attempts_requested,
            "attempts_per_sample": 1,
            "not_applied": attempt_skip_reason,
        }

    category_accuracy = {}
    for cat, stats in category_stats.items():
        cat_acc = (stats["correct"] / stats["total"] * 100.0) if stats["total"] > 0 else 0.0
        category_accuracy[cat] = {
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": round(cat_acc, 2),
        }

    # Replies that hit the cap far longer than this suite's healthy ones without repeating
    # verbatim are relabelled `non_convergent` first, so every count below (response_health
    # included) reads the final classification of each trace.
    non_convergent = reclassify_non_convergent(sample_traces, official_caps)
    health_counts = {}
    for t in sample_traces:
        h = t.get("health")
        health_counts[h] = health_counts.get(h, 0) + 1
    # Replies cut off while still writing by a max_tokens that is not an official cap: the only
    # replies a larger budget would have let finish, so the only cut-off replies that make the
    # run a measurement problem. Not among them: a reply at an official cap (the protocol's
    # outcome, scored as upstream scores it), a reply that reached the end of the context window
    # when no max_tokens is sent (the official harness sends none: the protocol's outcome too),
    # and a looping or non-convergent reply (the model's own outcome, counted in
    # looping_responses / non_convergent_responses).
    budget_cut = sum(1 for t in sample_traces
                     if t.get("health") == "truncated" and t.get("sample_idx") not in official_caps
                     and t.get("max_tokens_sent") is not None)
    at_context_window = sum(1 for t in sample_traces
                            if t.get("health") == "truncated" and t.get("max_tokens_sent") is None)
    # Never emit a clean "success" 0% when nothing was actually scored. Zero samples
    # (loader produced nothing / dataset failed to materialize) is an error, not a 0% pass.
    # Requests that got no reply from the endpoint (harness errors), by kind.
    request_harness_errors: Dict[str, int] = {}
    for _status, err in harness_errors.values():
        request_harness_errors[err["kind"]] = request_harness_errors.get(err["kind"], 0) + 1
    # A suite's own follow-up request (a MathArena last-chance reprompt) that got no reply: the
    # suite marks the trace with base's endpoint harness error (endpoint_harness_error) and leaves
    # it out of the score. Counted as base counts its own, not as a suite exclusion.
    followup_no_reply: Dict[str, int] = {}
    followup_idxs = set()
    for t in sample_traces:
        he = t.get("harness_error")
        if (isinstance(he, dict) and he.get("kind") in NO_REPLY_HARNESS_KINDS
                and t.get("scoring_excluded") and t.get("sample_idx") not in harness_errors):
            followup_no_reply[he["kind"]] = followup_no_reply.get(he["kind"], 0) + 1
            followup_idxs.add(t.get("sample_idx"))
    for kind, n in followup_no_reply.items():
        request_harness_errors[kind] = request_harness_errors.get(kind, 0) + n
    # Samples the suite left out of the score (grader failures, samples it cannot measure), other
    # than the endpoint failures above and the exclusions the benchmark's own protocol makes
    # (a trace the suite marks `excluded_by_protocol`, e.g. a context upstream's metric discards).
    # A judge failure the suite leaves out is named once, under the judge failures.
    suite_excluded = sum(1 for t in sample_traces
                         if t.get("scoring_excluded") and t.get("sample_idx") not in harness_errors
                         and t.get("sample_idx") not in judge_failed_idxs
                         and t.get("sample_idx") not in followup_idxs
                         and not t.get("excluded_by_protocol"))
    judge_failures_excluded = sum(1 for t in sample_traces
                                  if t.get("sample_idx") in judge_failed_idxs
                                  and t.get("scoring_excluded"))
    status_causes: List[str] = []
    # The server-parser policy's harness errors (a reply the server's tool-call parser deleted
    # that could not be recovered) are not requests that got no reply: named on their own.
    parser_lost = sum(1 for _st, err in harness_errors.values()
                      if err["kind"] == SERVER_PARSER_TEXT_LOST)
    endpoint_errors: Dict[str, int] = {}
    for _st, err in harness_errors.values():
        if err["kind"] != SERVER_PARSER_TEXT_LOST:
            endpoint_errors[err["kind"]] = endpoint_errors.get(err["kind"], 0) + 1
    if failed_requests:
        by_status: Dict[str, int] = {}
        for base_status, _err in harness_errors.values():
            if _err["kind"] == SERVER_PARSER_TEXT_LOST:
                continue
            by_status[base_status] = by_status.get(base_status, 0) + 1
        if empty_responses:
            by_status["EMPTY_RESPONSE"] = empty_responses
        detail = ", ".join(f"{k} {v}" for k, v in sorted(by_status.items()))
        if judge_failures:
            detail = (detail + ", " if detail else "") + f"judge failures {judge_failures}"
            if judge_failures_excluded:
                detail += f", of which {judge_failures_excluded} left out of the score"
        status_causes.append(f"failed_requests {failed_requests}"
                             + (f" ({detail})" if detail else ""))
    if endpoint_errors:
        status_causes.append(
            f"harness errors left out of the score: {sum(endpoint_errors.values())} request(s) "
            "that got no reply from the served endpoint ("
            + ", ".join(f"{k} {v}" for k, v in sorted(endpoint_errors.items())) + ")")
    if followup_no_reply:
        status_causes.append(
            f"harness errors left out of the score: {sum(followup_no_reply.values())} follow-up "
            "request(s) of the suite's protocol that got no reply from the served endpoint ("
            + ", ".join(f"{k} {v}" for k, v in sorted(followup_no_reply.items())) + ")")
    if parser_lost:
        status_causes.append(
            f"harness errors left out of the score: {SERVER_PARSER_TEXT_LOST} {parser_lost} (the "
            "server's tool-call parser deleted the reply and the text could not be recovered "
            "from its token ids)")
    if judge_outages:
        status_causes.append(f"judge_outages {judge_outages} (Gemini cascade exhausted)")
    if suite_excluded:
        status_causes.append(f"scoring_excluded {suite_excluded} (left out of the score by the "
                             "suite)")
    if budget_cut * 10 > total_count:
        status_causes.append(
            f"replies_cut_off_by_non_official_budget {budget_cut} of {total_count} (over 10%: "
            "cut off by a max_tokens that is not an official cap while still writing)")
    if total_count == 0:
        status_str = "error"
        status_reason = "no samples were run"
    elif (scored_count <= 0 or failed_requests == total_count
          or judge_outages >= total_count):
        # nothing was scored - every sample failed generation, got no reply from the endpoint,
        # hit a judge outage or was left out of the score, in any mix
        status_str = "failed"
        status_reason = "nothing was scored: " + (
            "; ".join(status_causes)
            or f"all {total_count} sample(s) left out of the score "
               f"(scoring_excluded {scoring_excluded})")
    elif not status_causes:
        status_str = "success"
        status_reason = None
    else:
        # A failed request, a judge outage, or more than 10% of answers cut off by a budget that
        # is not an official cap: a measurement problem, not a model result.
        status_str = "completed_with_errors"
        status_reason = "; ".join(status_causes)
    # A suite that fell back to substring matching because the judge was unavailable must
    # not report its number as the canonical metric.
    fallback = sum(1 for t in sample_traces if t.get("scoring_mode") == "judge_fallback")
    scoring_mode = classify_scoring_mode(
        has_fallback=bool(fallback),
        declared=declared_scoring_mode,
        has_async_scorer=async_eval_fn is not None,
    )
    if fallback:
        logger.warning(
            "[%s] %d/%d samples were graded by the no-judge fallback (substring match), "
            "not the canonical judge. This is a lower bound, not the benchmark's metric.",
            eval_name, fallback, total_count)

    # From base's own record, not the traces' statuses, which batch scorers rewrite.
    over_context = sum(1 for st, _e in harness_errors.values() if st == "OVER_CONTEXT")
    discarded = sum(1 for t in sample_traces if t.get("health") == "output_discarded")
    if discarded:
        logger.warning(
            "[%s] %d/%d responses were GENERATED BUT DISCARDED by the server: the model "
            "emitted a tool call and the tool-call parser removed it, but the request "
            "declared no `tools` so there was nowhere to put it. These are not empty "
            "answers. Declare the suite's tool schemas in the sample meta "
            "(`{\"tools\": [...]}`) to capture them.", eval_name, discarded, total_count)
    # The server-parser policy's events, per run: replies whose text the server's reasoning parser
    # moved out of a no-think reply or its tool-call parser deleted (recovered and scored as the
    # model's output), and deleted replies that could not be recovered (harness errors).
    parser_samples: Dict[str, List[int]] = {}
    parser_sources: Dict[str, int] = {}
    for t in sample_traces:
        ev = (t.get("server_parser") or {}).get("event")
        if ev:
            parser_samples.setdefault(ev, []).append(t.get("sample_idx"))
            src = (t.get("server_parser") or {}).get("recovered_from")
            if src:
                parser_sources[src] = parser_sources.get(src, 0) + 1
    parser_note = None
    if parser_samples:
        parser_note = server_parser_note({k: len(v) for k, v in parser_samples.items()},
                                         parser_sources)
        logger.warning("[%s] %s", eval_name, parser_note)
    timed_out = sum(1 for st, _e in harness_errors.values() if st == "TIMEOUT")
    if timed_out:
        logger.warning(
            "[%s] %d/%d requests timed out at %ds. They are harness errors, left out of the "
            "score, but timeouts drop the LONGEST answers, so the accuracy is biased. Raise "
            "GBENCH_REQUEST_TIMEOUT_S or lower --batch-sizes before quoting this.",
            eval_name, timed_out, total_count, request_timeout_s(timeout_budget))
    if over_context:
        logger.warning(
            "[%s] %d/%d prompts exceed the model's context window and could not be sent. "
            "These are a dataset/serving mismatch, not wrong answers: harness errors, left "
            "out of the score and named in the status.",
            eval_name, over_context, total_count)

    in_thinking = sum(1 for t in sample_traces if t.get("health") == "empty_reasoned")
    if in_thinking:
        logger.info(
            "[%s] %d/%d replies stopped with the answer inside the thinking channel (reasoning, "
            "no content): the model's outcome, scored as an empty answer, as an upstream harness "
            "reading content scores it (answer_in_thinking).", eval_name, in_thinking, total_count)
    looping = sum(1 for t in sample_traces if t.get("health") == "looping")
    if looping:
        onsets = [t["repetition_onset"] for t in sample_traces
                  if t.get("health") == "looping" and t.get("repetition_onset")]
        detail = ""
        if onsets:
            fracs = sorted(o["onset_frac"] for o in onsets)
            periods = sorted(o["period_words"] or 0 for o in onsets)
            detail = (f" Onset at {fracs[len(fracs)//2]:.0%} through the output (range "
                      f"{fracs[0]:.0%}-{fracs[-1]:.0%}), cycle period "
                      f"{periods[len(periods)//2]} words. Stuck on: "
                      f"{onsets[0]['cycle_preview'][:70]!r}.")
        if uncapped:
            # No max_tokens was sent: these replies ran to the end of the context window, and
            # --max-output-tokens does not apply to this suite.
            logger.warning(
                "[%s] %d/%d responses ran to the end of the context window (no max_tokens is "
                "sent, as the official harness sends none) while REPEATING THEMSELVES (a "
                "20-word window recurring >=%d times, or a run of one character); this is a "
                "decoding/model behaviour.%s",
                eval_name, looping, total_count, REPETITION_THRESHOLD, detail)
        else:
            logger.warning(
                "[%s] %d/%d responses hit the cap while REPEATING THEMSELVES (a 20-word window "
                "recurring >=%d times, or a run of one character), not while writing a longer "
                "answer. Raising --max-output-tokens buys more of the same and costs "
                "wall-clock; this is a decoding/model behaviour, not a budget problem.%s",
                eval_name, looping, total_count, REPETITION_THRESHOLD, detail)

    # Every timeout in this harness is sized from an ASSUMED decode rate. Nothing checked
    # that assumption against the box actually running, so on slower hardware a suite could
    # only fail as a bare timeout with no hint that the sizing, not the model, was wrong.
    # The rate is measured for free from `completion_tokens / wall time`, so state it, and
    # say so when reality is under the assumption.
    rates = sorted(t["decode_tok_s"] for t in sample_traces if t.get("decode_tok_s"))
    observed_rate = rates[len(rates) // 2] if rates else None
    if observed_rate is not None and observed_rate < MIN_DECODE_TOK_S:
        logger.warning(
            "[%s] measured decode rate is %.1f tok/s, BELOW the %.1f tok/s the request "
            "timeout assumes. Long generations on this endpoint will time out: those are "
            "harness errors, left out of the score, but timeouts drop the LONGEST answers, so "
            "the accuracy is biased. Lower GBENCH_MIN_DECODE_TOK_S (it sizes the timeout) or "
            "reduce --batch-sizes.",
            eval_name, observed_rate, MIN_DECODE_TOK_S)

    # Looping responses also hit the cap, but "raise the budget" is exactly the wrong
    # advice for them - they are counted and explained above instead. Same for responses
    # that ran many times longer than this suite's healthy median without converging
    # (relabelled before the status was decided).
    if non_convergent:
        logger.warning(
            "[%s] %d/%d responses hit the cap while running >=%.0fx this suite's healthy "
            "median length WITHOUT repeating verbatim - the model kept re-deriving instead "
            "of converging. Raising --max-output-tokens buys more of the same; these are "
            "reported as `non_convergent`, not `truncated`.",
            eval_name, non_convergent, total_count, NON_CONVERGENT_LENGTH_RATIO)
    official_cap_hits = [t for t in sample_traces
                         if t.get("health") in ("truncated", "looping", "non_convergent")
                         and t.get("sample_idx") in official_caps]
    if official_cap_hits:
        caps = sorted({official_caps[t["sample_idx"]] for t in official_cap_hits})
        logger.info(
            "[%s] %d/%d replies stopped at the suite's official max_tokens (%s) and were scored "
            "on what arrived, as the official harness scores them.",
            eval_name, len(official_cap_hits), total_count, ", ".join(str(c) for c in caps))
    # Replies cut off by a budget that is not an official cap, still writing (not looping and
    # not non-convergent): the ones a larger budget would have let finish.
    genuinely_truncated = sum(1 for t in sample_traces
                              if t.get("health") == "truncated"
                              and t.get("sample_idx") not in official_caps)
    if genuinely_truncated > 0 and uncapped:
        logger.warning(
            "[%s] %d/%d responses reached the end of the context window (no max_tokens is "
            "sent, as the official harness sends none) and were scored on a partial answer.",
            eval_name, genuinely_truncated, total_count)
    elif genuinely_truncated > 0:
        # Report the budget the SERVER saw, not the suite-level resolution: a sample's meta
        # can override it, and quoting the wrong number sent culer's investigation astray.
        hit = sorted({t.get("max_tokens_sent") for t in sample_traces
                      if t.get("status") == "TRUNCATED" and t.get("max_tokens_sent")
                      and t.get("sample_idx") not in official_caps})
        shown = ", ".join(str(h) for h in hit) or str(effective_max_tokens)
        if tool_rounds_used:
            # "Raise the budget" is the wrong advice once a suite runs a tool loop: the
            # budget bounds ONE TURN, and browsecomp at 65536/turn produced no answer at
            # all where 8192/turn produced a judged one. A truncated turn here means the
            # model chose to spend the turn thinking rather than acting.
            logger.warning(
                "[%s] %d/%d TURNS hit the per-turn budget (max_tokens=%s). In a tool loop "
                "this bounds one turn, not the run - raising it lets the model think for "
                "longer without acting, which is how this suite stopped answering at all. "
                "Adjust GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS if the turn is genuinely too "
                "small (tool calls per sample will be near zero).",
                eval_name, genuinely_truncated, total_count, shown)
        else:
            logger.warning(
                "[%s] %d/%d responses hit the output-token budget (max_tokens=%s) and were "
                "scored on a partial answer. Raise --max-output-tokens before quoting this.",
                eval_name, genuinely_truncated, total_count, shown)

    return {
        "eval_limit_plan": limit_plan_record,
        "benchmark_type": "eval",
        "eval_name": eval_name,
        "model_name": model_name,
        "thinking": thinking,
        "total_questions": total_count,
        "correct_answers": correct_count,
        "failed_requests": failed_requests,
        "empty_responses": empty_responses,
        "truncated_responses": truncated_responses,
        "non_convergent_responses": non_convergent,
        "genuinely_truncated": genuinely_truncated,
        "max_tokens_sent": sorted({t.get("max_tokens_sent") for t in sample_traces
                                   if t.get("max_tokens_sent")}) or None,
        # the official caps the requests carried, and the replies that stopped at one (the
        # protocol's outcome, not a truncation)
        **({"official_max_tokens_sent": sorted(set(official_caps.values())),
            "replies_at_official_max_tokens": len(official_cap_hits)}
           if official_caps else {}),
        # --thinking samples whose declared cap (written for non-reasoning generation) was
        # replaced by the run's per-request budget
        **({"official_cap_not_applied_to_reasoning": {
            "declared_max_tokens": sorted(set(lifted_caps.values())),
            "samples": len(lifted_caps), "sent": effective_max_tokens}}
           if lifted_caps else {}),
        "over_context_prompts": over_context,
        "timed_out_requests": timed_out,
        **({"wall_clock_limited": len(wall_limited)} if wall_s else {}),
        **shard_identity(shard_spec),
        "looping_responses": looping,
        "discarded_outputs": discarded,
        # replies whose answer the model wrote inside its thinking channel (the model's outcome,
        # scored; ANSWER_IN_THINKING_POLICY), and their share of the samples
        **answer_in_thinking_fields(
            sum(1 for t in sample_traces if t.get("health") == "empty_reasoned"), total_count),
        "request_timeout_s": request_timeout_s(timeout_budget),
        # What the harness assumed vs what this endpoint actually did. Every derived
        # timeout follows from `assumed`; publishing both is what makes a timeout on
        # unfamiliar hardware diagnosable instead of a bare failure.
        "decode_tok_s": {"observed_median": observed_rate,
                         "assumed_floor": MIN_DECODE_TOK_S} if observed_rate else None,
        "tool_rounds": {"samples_using_tools": len(tool_rounds_used),
                        "total_rounds": sum(tool_rounds_used.values()),
                        # samples that exhausted the round budget and were made to commit to
                        # an answer instead of being graded on a dangling tool call
                        "forced_final_answers": len(forced_finals),
                        # the budget each TURN was given, which in a loop is not the run's
                        "per_turn_max_tokens": effective_max_tokens,
                        "tool_calls_executed": sum(tool_calls_used.values()),
                        # how the loops ended: answered / forced_final / max_rounds /
                        # max_tool_calls / context_exhausted / request_failed
                        "loop_ends": {k: sum(1 for v in loop_ends.values() if v == k)
                                      for k in sorted(set(loop_ends.values()))},
                        } if tool_rounds_used else None,
        # The loop settings this run used, so a tool-loop number can be read against them.
        **({"tool_loop_protocol": (tool_loop or ToolLoopProtocol(
            reference="gbench house loop", max_rounds=max_tool_rounds)).record(
                effective_max_tokens)} if tool_executor is not None else {}),
        "scoring_mode": scoring_mode,
        "judge_failures": judge_failures,
        "judge_outages": judge_outages,
        "judge_outage_rate": round(judge_outages / total_count, 4) if total_count else 0.0,
        # every sample left out of the score (the endpoint failures below included)
        "scoring_excluded": scoring_excluded,
        # harness errors by kind (requests that got no reply from the served endpoint, and replies
        # the server's tool-call parser deleted that could not be recovered): left out of the
        # score and named in status_reason
        **({"request_harness_errors": dict(sorted(request_harness_errors.items()))}
           if request_harness_errors else {}),
        **({"server_parser_token_ids": server_parser_token_ids_record()}
           if _TOKEN_IDS_REFUSED else {}),
        # the server-parser policy's events (see recover_server_parser_text): counts, the samples,
        # and a note naming them
        **({"server_parser_events": {k: len(v) for k, v in sorted(parser_samples.items())},
            "server_parser_samples": {k: sorted(v) for k, v in sorted(parser_samples.items())},
            "server_parser_note": parser_note}
           if parser_samples else {}),
        "response_health": health_counts,
        # the end of the context window, reached with no max_tokens sent (the protocol's outcome)
        **({"replies_at_context_window": at_context_window} if at_context_window else {}),
        "attempts": attempts_report,
        # None when nothing was scored (the score is withheld, never 0)
        "accuracy": round(accuracy, 2) if accuracy is not None else None,
        **({"accuracy_withheld_reason": status_reason} if accuracy is None else {}),
        "category_accuracy": category_accuracy,
        "sample_traces": sample_traces,
        "status": status_str,
        # what made the status other than success (None on success)
        "status_reason": status_reason,
    }


def _samples_carry_media(samples: List[Any]) -> bool:
    """True when a sample's messages carry an image, audio or video part."""
    kinds = set(_IMAGE_PART_TYPES) | set(_AUDIO_PART_TYPES) | set(_VIDEO_PART_TYPES)
    for sample in samples or []:
        messages = sample[0] if isinstance(sample, (list, tuple)) and sample else None
        for msg in messages or []:
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, list) and any(isinstance(p, dict) and p.get("type") in kinds
                                                 for p in content):
                return True
    return False


def run_eval_suite(
    eval_name: str,
    model_name: str,
    base_url: str,
    concurrency: int,
    samples: List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]],
    eval_fn: Optional[Callable[[str, Any], bool]] = None,
    async_eval_fn: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
    declared_scoring_mode: Optional[str] = None,
    thinking: bool = False,
    extra_payload: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    tool_executor: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
    max_tool_rounds: int = 8,
    attempt_count: Optional[int] = None,
    supports_attempts: bool = False,
    tool_loop: Optional[ToolLoopProtocol] = None,
    limit_seed: Optional[str] = None,
    official_sampling: Optional[OfficialSampling] = None,
    full_set_items: Optional[int] = None,
) -> Dict[str, Any]:
    """Synchronous entry point to run an evaluation suite.

    `temperature`, `max_output_tokens` and `attempt_count` fall back to the run-level
    knobs when the suite does not pass them, so `--temperature` / `--attempt-count` reach
    every suite rather than only the 15 that happened to forward them.

    `supports_attempts` is opt-in and only consulted for batch-scored suites: a scorer
    that keys its predictions by instance id cannot be handed k copies of the same
    sample. See `_run_suite_async` for why that would fabricate an @k.

    `limit_seed` seeds the stratified `--eval-limit` subset (default: `eval_name`). A renamed
    suite passes its old name so the same rows are drawn as before the rename.

    `full_set_items`: the size of the suite's full ordered set, for a suite that hands base its
    own shard already (sampling.presharded_for_base); an empty shard's row names it
    (`full_set_items`). Default: the number of samples given.

    `official_sampling` (opt-in, see OfficialSampling) makes the upstream harness's sampling
    the suite's default: its temperature (or no temperature at all) and its `params`. The
    operator's per-suite env variable or --temperature still wins, and the result records
    `official_sampling` with `temperature_matches_official`. A suite that passes an explicit
    `temperature` as well keeps that value (source "suite"). A --thinking run uses the
    upstream's reasoning-model settings (`official_sampling.reasoning`) when it defines them.

    max_tokens follows the token-limit policy (resolve_token_budget): an official cap is sent
    exactly on a no-think run and not applied to a --thinking run, which is never capped below
    --max-output-tokens. A per-request max_tokens a sample's meta (or `extra_payload`) declares
    is an official cap too: sent as declared without --thinking; with it, the declared cap is
    replaced by the run's per-request budget, larger or smaller, as an
    OfficialSampling.max_tokens is, and the result counts it under
    `official_cap_not_applied_to_reasoning`. A suite whose official sampling sends no max_tokens
    cannot declare one per request (ValueError).
    """
    concurrency = max(1, int(concurrency or 1))
    mode_sampling = None
    if official_sampling is not None:
        check_official_sampling_params(eval_name, official_sampling)
        mode_sampling = official_sampling.for_mode(thinking)
        params = dict(mode_sampling.params or {})
        if params:
            extra_payload = {**params, **(extra_payload or {})}
    if official_sampling is not None and temperature is None:
        temperature, temperature_source = resolve_official_sampling(
            eval_name, official_sampling, thinking=thinking)
    else:
        temperature, temperature_source = resolve_temperature(eval_name, temperature,
                                                              thinking=thinking)
    if temperature_source.startswith("env:"):
        logger.info("[%s] temperature %.2f from %s (overrides --temperature)",
                    eval_name, temperature, temperature_source.split(":", 1)[1])
    if attempt_count is None:
        attempt_count = get_run_knob("attempt_count", 1)
    penalties = decoding_penalties()
    if penalties:
        extra_payload = {**(extra_payload or {}), **penalties}
        logger.warning("Decoding penalties active: %s. Canonical protocol is greedy with no "
                       "penalty, so these numbers are NOT comparable with published results "
                       "or with runs that did not set them.", penalties)
    knob = get_run_knob("max_output_tokens")
    # The floor must know about thinking: reasoning and the answer share one budget, so the
    # non-thinking floor cuts the trace off before the answer starts.
    default_floor = THINKING_MIN_OUTPUT_TOKENS if thinking else DEFAULT_MIN_OUTPUT_TOKENS
    floor = max(SUITE_MIN_OUTPUT_TOKENS.get(eval_name, 0), default_floor)
    # The token-limit policy: an official cap (or its absence) decides the budget for this mode;
    # a suite that declares none keeps the house budget below.
    plan = resolve_token_budget(eval_name, official_sampling, thinking=thinking,
                                max_output_tokens=knob)
    uncapped = plan.uncapped
    if plan.policy != "house":
        max_output_tokens = plan.max_tokens   # no knob override, floor or turn ceiling
    elif knob:
        max_output_tokens = knob          # operator was explicit; never override it
    elif floor and (max_output_tokens is None or max_output_tokens < floor):
        if max_output_tokens is not None:
            logger.info("[%s] raising max_output_tokens %d -> %d (%s floor)",
                        eval_name, max_output_tokens, floor,
                        "thinking" if thinking else "long-answer suite")
        max_output_tokens = floor
    # A suite whose protocol opts out of the house ceiling runs every turn at the run's own
    # budget: its reference setup has no per-turn cap, and the operator's --max-output-tokens
    # (or the thinking floor) is never cut below. Neither does a suite with an official cap.
    turn_ceiling = (tool_loop_turn_ceiling(tool_executor is not None, max_output_tokens)
                    if plan.policy == "house"
                    and (tool_loop is None or tool_loop.house_turn_ceiling)
                    else None)
    turn_ceiling_requested = max_output_tokens
    if turn_ceiling:
        logger.warning(
            "[%s] capping the PER-TURN budget %d -> %d: this suite runs a tool loop, so "
            "`max_output_tokens` bounds one turn, not the run. Set "
            "GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS=0 to disable.",
            eval_name, max_output_tokens, turn_ceiling)
        max_output_tokens = turn_ceiling
    start_time = time.time()
    answered_before = gemini_answer_counts()
    result = asyncio.run(
        _run_suite_async(
            eval_name=eval_name,
            model_name=model_name,
            base_url=base_url,
            concurrency=concurrency,
            samples=samples,
            eval_fn=eval_fn,
            async_eval_fn=async_eval_fn,
            declared_scoring_mode=declared_scoring_mode,
            thinking=thinking,
            extra_payload=extra_payload,
            limit=limit,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            tool_executor=tool_executor,
            max_tool_rounds=max_tool_rounds,
            attempt_count=attempt_count,
            supports_attempts=supports_attempts,
            tool_loop=tool_loop,
            limit_seed=limit_seed,
            uncapped=uncapped,
            full_set_items=full_set_items,
            # the budget is an official cap only when that cap itself is what is sent
            official_budget=(plan.policy == "official"
                             or (plan.policy == "official_reasoning"
                                 and plan.max_tokens == plan.official_cap)),
        )
    )
    result["duration_s"] = round(time.time() - start_time, 2)
    # Which Gemini models answered this run's judge_generate_cascade calls (the suite may also
    # record them per verdict), and the cascade they were asked from.
    answered = {m: n - answered_before.get(m, 0) for m, n in gemini_answer_counts().items()
                if n - answered_before.get(m, 0) > 0}
    if answered:
        result["gemini_models_answered"] = dict(sorted(answered.items()))
    # The configured Gemini cascade is stamped on every result; whether the model under test is in
    # it (self_judge) only for a run that used Gemini (an answer here, or the suite's Gemini gate).
    prov = judge_provenance(model_name)
    result["gemini_cascade"] = prov["gemini_cascade"]
    result["gemini_cascade_source"] = prov["gemini_cascade_source"]
    if answered or eval_name in _GEMINI_GATED:
        result["self_judge"] = prov["self_judge"]
        result["self_judge_models"] = prov["self_judge_models"]
        if prov["self_judge"]:
            add_comparability_reason(result, SELF_JUDGE_REASON)
    # The per-turn ceiling a tool loop ran under (TOOL_LOOP_MAX_OUTPUT_TOKENS), which cut the run's
    # budget: recorded, and named, so a plugin or custom tool-loop suite says so on its result.
    if turn_ceiling:
        result["tool_loop_turn_ceiling"] = {"applied": turn_ceiling,
                                            "requested": turn_ceiling_requested}
        add_comparability_reason(result, (
            f"tool-loop per-turn ceiling {turn_ceiling} applied below the run's budget "
            f"{turn_ceiling_requested} (GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS=0 disables it)"))
    # The media soft-token budgets base sized image/audio/video prompts with (a sample carried one).
    if _samples_carry_media(samples):
        result["media_soft_token_budget"] = {"image": IMAGE_SOFT_TOKENS,
                                             "audio": AUDIO_SOFT_TOKENS,
                                             "video": VIDEO_SOFT_TOKENS}
    result["decoding_penalties"] = penalties or None
    result["temperature"] = temperature
    result["temperature_source"] = temperature_source
    if official_sampling is not None:
        rec = mode_sampling.record(temperature, temperature_source)
        rec.update(plan.record())
        if mode_sampling is not official_sampling:
            rec["reasoning_settings"] = True
        result["official_sampling"] = rec
    notes = [plan.note] if plan.note else []
    declared_sent = result.get("official_max_tokens_sent") if plan.policy == "house" else None
    if declared_sent and knob and any(c != knob for c in declared_sent):
        notes.append(f"--max-output-tokens {knob} not applied to the requests that carry the "
                     f"suite's official max_tokens ({', '.join(str(c) for c in declared_sent)})")
    lifted = result.get("official_cap_not_applied_to_reasoning")
    if lifted:
        notes.append(
            f"{REASONING_CAP_NOTE}: the per-request max_tokens "
            f"{', '.join(str(c) for c in lifted['declared_max_tokens'])} the suite declares for "
            f"{lifted['samples']} sample(s) was replaced by this run's budget "
            f"({lifted['sent'] if lifted['sent'] is not None else 'no max_tokens'})")
    if notes:
        result["max_output_tokens_note"] = "; ".join(notes)
    # No credential of the environment leaves the suite in a trace or a result field.
    return redact_secrets(result)


def gemini_key_live_valid(key: str) -> Tuple[bool, str]:
    """Live ping of the Gemini OpenAI-compat endpoint, then the model check. Returns (False,
    reason) ONLY on a definitive rejection: the key's auth rejection, or a cascade model the API
    does not serve (gemini_models_available); a network/other error is inconclusive -> (True, ...)
    so a valid key is never blocked by gate-time infra flakiness. Shared by every judge suite
    (gaia2/wildclawbench import it) so the presence+live-ping gate is uniform.

    `key` may be a comma/semicolon-separated LIST (a multi-key GEMINI_API_KEY): it is live when ANY
    member is live, so one dead/expired key in the list never blocks the run. A
    GEMINI_OPENAI_BASE_URL that gemini_openai_base_url() refuses is a rejection naming it (once,
    whatever the number of keys), and no key is sent anywhere. When the key that passed was
    accepted outright (HTTP 2xx, not an inconclusive answer), every model of judge_cascade() and
    search_cascade() is checked with it (GET <base>/models/<name>, cached per process): a model
    the API answers 404 (or 403, a model this key may not use) for is a rejection naming the
    model and the knob it came from."""
    ok, why, _kind = _gemini_gate(key)
    return ok, why


def _gemini_gate(key: str) -> Tuple[bool, str, str]:
    """gemini_key_live_valid's verdict with what failed: (ok, reason, kind), kind "key" (no key,
    a refused GEMINI_OPENAI_BASE_URL, or the key's rejection), "models" (a cascade model the API
    does not serve) or "" when it passed."""
    cands = [k.strip() for k in re.split(r"[;,]", str(key or "")) if k.strip()]
    if not cands:
        return False, "no GEMINI_API_KEY", "key"
    base, why = gemini_openai_base_url()
    if base is None:
        return False, why, "key"
    reasons = []
    for k in cands:
        ok, why = _single_key_live(k)
        if ok:
            if why:          # inconclusive: the models cannot be checked either
                return True, why, ""
            ok, why = gemini_models_available(k)
            return ok, why, ("" if ok else "models")
        reasons.append(why)
    return False, "; ".join(reasons), "key"


#: How the model check's rejection reason starts (gemini_models_available).
MODELS_NOT_SERVED_PREFIX = "the Gemini API does not serve"
#: The model check's per-process cache: {(base URL, key digest, model): (served, why)}. Only a
#: definitive answer is cached (served, or not served); an inconclusive one is asked again.
_MODEL_CHECKS: Dict[Tuple[str, str, str], Tuple[bool, str]] = {}
_MODEL_CHECKS_LOCK = threading.Lock()


def gemini_cascade_models() -> List[Tuple[str, str]]:
    """[(model, knob)] for every Gemini text model this run may call, in order and de-duplicated:
    judge_cascade(), then the search_cascade() models not in it. `knob` names the setting the
    model came from (GBENCH_JUDGE_MODELS, GBENCH_JUDGE_MODEL, GBENCH_SEARCH_MODELS,
    GBENCH_SEARCH_MODEL, or "default" for GEMINI_CASCADE)."""
    out: List[Tuple[str, str]] = []
    seen = set()
    for models, knob in ((judge_cascade(), judge_cascade_source()),
                         (search_cascade(), search_cascade_source())):
        for m in models:
            if m not in seen:
                seen.add(m)
                out.append((m, knob))
    return out


def _model_check(base: str, key: str, model: str) -> Tuple[Optional[bool], str]:
    """(True, "") when GET <base>/models/<model> answers 2xx, (False, why) on a 404, a 403 or a
    400 saying the model is not found or not supported, (None, why) when inconclusive."""
    name = model[len("models/"):] if model.startswith("models/") else model
    url = f"{base}/models/{urllib.parse.quote(name, safe='')}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            code = getattr(r, "status", 200)
            if 200 <= code < 300:
                return True, ""
            return None, f"inconclusive HTTP {code}"
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "ignore")[:300]
        except Exception:                                                 # noqa: BLE001
            pass
        bl = body.lower()
        if e.code == 404:
            return False, "HTTP 404, not found"
        if e.code == 403:
            return False, "HTTP 403, this key may not use it"
        if e.code == 400 and ("not found" in bl or "not supported" in bl
                              or "invalid model" in bl or "unknown model" in bl):
            return False, "HTTP 400, not found or not supported"
        return None, f"inconclusive HTTP {e.code}"
    except Exception as e:                                               # noqa: BLE001
        return None, f"inconclusive {type(e).__name__}"


def gemini_models_available(key: Optional[str] = None,
                            models: Optional[List[Tuple[str, str]]] = None) -> Tuple[bool, str]:
    """(True, "") unless the Gemini API definitively does not serve a model of the run's cascade
    to `key` (default: the first configured key), then (False, reason) naming every such model and
    the knob to fix. `models` is [(model, knob)] (default gemini_cascade_models()). Each model is
    asked once per process (GET <GEMINI_OPENAI_BASE_URL>/models/<name>); an inconclusive answer
    (network error, 429, 5xx) never blocks and is asked again at the next check."""
    keys = [key] if key else gemini_api_keys()
    if not keys:
        return False, "no GEMINI_API_KEY"
    base, why = gemini_openai_base_url()
    if base is None:
        return False, why
    digest = hashlib.sha256(keys[0].encode("utf-8")).hexdigest()[:16]
    missing: List[str] = []
    knobs: List[str] = []
    for model, knob in (models if models is not None else gemini_cascade_models()):
        ck = (base, digest, model)
        with _MODEL_CHECKS_LOCK:
            cached = _MODEL_CHECKS.get(ck)
        if cached is None:
            served, why = _model_check(base, keys[0], model)
            if served is not None:
                cached = (served, why)
                with _MODEL_CHECKS_LOCK:
                    _MODEL_CHECKS[ck] = cached
        if cached is not None and not cached[0]:
            where = ("gbench's default Gemini cascade (base.GEMINI_CASCADE)" if knob == "default"
                     else knob)
            missing.append(f"'{model}' from {where} ({cached[1]})")
            fix = JUDGE_MODELS_ENV if knob in ("default", JUDGE_MODEL_ENV) else knob
            if fix not in knobs:
                knobs.append(fix)
    if missing:
        return False, (f"{MODELS_NOT_SERVED_PREFIX} {'; '.join(missing)} for this key. Set "
                       f"{' / '.join(knobs)} (comma separated, tried in order) to models the key "
                       f"serves, or correct the name")
    return True, ""


def reset_gemini_model_checks() -> None:
    """Forget the cached model checks (tests, or after the operator changed keys or models)."""
    with _MODEL_CHECKS_LOCK:
        _MODEL_CHECKS.clear()


#: The Gemini API's OpenAI-compatible endpoint: the default of GEMINI_OPENAI_BASE_URL.
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
#: The Gemini API's own base URL. The judge client (google-genai, _judge_clients) is always built
#: on it, so google-genai's GOOGLE_GEMINI_BASE_URL / GOOGLE_GENAI_USE_VERTEXAI never move it.
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/"
_GEMINI_API_HOST = "generativelanguage.googleapis.com"


def check_gemini_openai_base_url(raw: Optional[str]) -> Tuple[Optional[str], str]:
    """(base URL without its trailing slash, "") when `raw` may receive a Gemini key, else
    (None, reason). An empty `raw` is the Gemini API itself (GEMINI_OPENAI_BASE_URL).

    Only two kinds of value are accepted: the Gemini API over https (host
    generativelanguage.googleapis.com, any port and path), or a loopback address over http or
    https (localhost, 127.0.0.0/8 or ::1: a local stub or relay on this host, for tests). A URL
    with a user name or password, whitespace or control characters, a bad port or any other host
    or scheme is refused. The reason never repeats the value, which may hold a credential.
    Standard library only, so container code that cannot import gbench can mirror it verbatim."""
    import ipaddress
    from urllib.parse import urlsplit
    raw = (raw or "").strip()
    if not raw:
        return GEMINI_OPENAI_BASE_URL.rstrip("/"), ""
    refused = (None, "GEMINI_OPENAI_BASE_URL is not the Gemini API (https://"
                     f"{_GEMINI_API_HOST}/...) or a loopback address, so the Gemini key is not "
                     "sent to it; unset it")
    if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in raw):
        return refused
    try:
        u = urlsplit(raw)
        host = (u.hostname or "").lower()
        u.port  # noqa: B018 - raises ValueError on a bad port
    except ValueError:
        return refused
    if "@" in u.netloc or not host:
        return refused
    if u.scheme == "https" and host == _GEMINI_API_HOST:
        return raw.rstrip("/"), ""
    loopback = host == "localhost"
    if not loopback:
        try:
            loopback = ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
    if loopback and u.scheme in ("http", "https"):
        return raw.rstrip("/"), ""
    return refused


def gemini_openai_base_url() -> Tuple[Optional[str], str]:
    """(base URL, "") of the Gemini API's OpenAI-compatible endpoint, or (None, reason).

    The one reader of GEMINI_OPENAI_BASE_URL in os.environ. Every host-side path that sends a
    Gemini key to it, or hands it to a container or proxy that will (the key's live check, the
    gaia2 and wildclawbench judge proxies, gbench's Gemini relay, which the tau2/tau3 container
    reaches instead of the variable), goes through this helper, require_gemini_openai_base_url or
    that check, independently of any skip-key-validation knob. Accepted values:
    check_gemini_openai_base_url (the Gemini API over https, or a loopback address); anything else
    is refused, never contacted."""
    return check_gemini_openai_base_url(os.environ.get("GEMINI_OPENAI_BASE_URL"))


def require_gemini_openai_base_url(eval_name: str, docs_url: Optional[str] = None) -> str:
    """gemini_openai_base_url()'s URL, or an infra_required hard error naming the refused
    variable (never its value). For a suite that forwards GEMINI_OPENAI_BASE_URL into a container
    or proxy: forward the returned URL, never the raw variable."""
    base, why = gemini_openai_base_url()
    if base is None:
        from .swebench_common import infra_required
        raise infra_required(eval_name, why + ".", docs_url or f"docs/evals/{eval_name}.md")
    return base


def _single_key_live(key: str) -> Tuple[bool, str]:
    """Live ping of ONE key (see gemini_key_live_valid for the (False, reason)-only-on-auth contract).
    A GEMINI_OPENAI_BASE_URL that is not the Gemini API or a loopback address is a rejection
    naming it (gemini_openai_base_url): the key is never sent to another host."""
    base, why = gemini_openai_base_url()
    if base is None:
        return False, why
    req = urllib.request.Request(base + "/models", headers={"Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return (200 <= getattr(r, "status", 200) < 300), ""
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return False, f"HTTP {e.code}"
        body = ""
        try:
            body = e.read().decode("utf-8", "ignore")[:300]
        except Exception:
            pass
        # Gemini returns 400 for a bad key, e.g. {"error":{"message":"Please pass a valid API key",
        # "status":"INVALID_ARGUMENT"}} or "API key not valid" / "API_KEY_INVALID". Any 400 that
        # mentions the API key is an auth rejection; other 400s are inconclusive (malformed request).
        bl = body.lower()
        if e.code == 400 and ("api key" in bl or "api_key" in bl):
            return False, f"HTTP 400 {body[:120]}"
        return True, f"inconclusive HTTP {e.code}"
    except Exception as e:
        return True, f"inconclusive {e}"


def gemini_required_skip(eval_name: str, model_name: str) -> None:
    """HARD-ERROR (infra_required) if no Gemini key is set OR it is live-rejected, else return None.

    The key is read like the judge client reads it (gemini_api_keys: GEMINI_API_KEYS, else
    GEMINI_API_KEY, comma/semicolon lists allowed), so a key list set only as GEMINI_API_KEYS
    passes the gate.

    An LLM-judge suite needs the key for canonical grading; per the no-skip policy a missing judge
    key is missing INFRA and must hard-error (never a silent skip, never a heuristic downgrade) so
    a judge-less run is loudly flagged as status:"error", not omitted as status:"skipped". The
    presence check is followed by a live auth ping (gemini_key_live_valid): an invalid or expired
    key (HTTP 401/403, or a 400 naming the API key) fails FAST at gate time instead of fail-slow
    after a full generation run turns every sample into a JUDGE_OUTAGE. The ping lists the models
    (GET <base>/models), which spends no generation quota, so an exhausted quota is not seen here:
    it surfaces at run time as JUDGE_OUTAGE samples (harness errors, left out of the score; a run
    whose every sample is one is failed with its score withheld). A network error or any other
    status (429 included) at gate time is inconclusive and never blocks a valid key. Once the key
    is accepted, every model of the run's Gemini cascade is checked (gemini_models_available): a
    model the API does not serve (a missing or mistyped name) is infra_required naming the model
    and the knob to fix.

    The gate also resolves the judge for `model_name` (resolve_judge_model: logs the cascade, and
    an ERROR when a cascade model is the model under test) and records the suite as Gemini-gated,
    so run_eval_suite and finalize_central_stamps stamp judge_provenance on its result and mark a
    self-judged result not leaderboard-comparable.

    Historical name kept to avoid churn across ~15 call sites, which use
    ``skip = gemini_required_skip(...); if skip: return skip`` - the call now RAISES on a missing or
    rejected key, so that guard is a no-op (this returns None when the key is present and valid).
    """
    _GEMINI_GATED[eval_name] = model_name
    resolve_judge_model(model_name)
    key = ",".join(gemini_api_keys())
    if not key:
        from .swebench_common import infra_required
        raise infra_required(
            eval_name,
            "requires GEMINI_API_KEY (or GEMINI_API_KEYS) for canonical LLM-judge grading",
            f"docs/evals/{eval_name}.md")
    ok, why = gemini_key_live_valid(key)
    if not ok:
        from .swebench_common import infra_required
        if str(why).startswith(MODELS_NOT_SERVED_PREFIX):
            raise infra_required(
                eval_name, f"{why}; every model of the Gemini cascade must be one the API serves "
                "for canonical LLM-judge grading", f"docs/evals/{eval_name}.md")
        raise infra_required(
            eval_name,
            f"the Gemini key (GEMINI_API_KEYS / GEMINI_API_KEY) was rejected by the judge "
            f"endpoint ({why}); a valid key is required for canonical LLM-judge grading",
            f"docs/evals/{eval_name}.md")
    return None
