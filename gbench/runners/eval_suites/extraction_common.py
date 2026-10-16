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

"""Shared answer-extraction helpers.

Several suites decided correctness by scanning the whole response for a token: the first
option letter, a bare "yes" anywhere, the last number. Those read the model's *reasoning*
rather than its *answer*, so a response that argues its way to the right conclusion can be
scored wrong and vice-versa.

The rule implemented here: prefer an explicitly anchored answer ("Answer:", "Final
Answer:", \\boxed{}), fall back to the LAST candidate (the conclusion, not the first
thought), and refuse to guess when the response asserts both or neither.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

_ANCHORS = (
    r"(?i)final\s*answer\s*[:=]\s*",
    r"(?i)\banswer\s*[:=]\s*",
    r"(?i)\b(?:the\s+)?(?:answer|choice|option)\s+is\s+",
)


def anchored_span(text: str) -> Optional[str]:
    """Text following the LAST explicit answer anchor, by textual position.

    Precedence is where the anchor sits in the text, not which pattern matched it. The old
    code looped patterns outer-most and overwrote on every match, so the *weakest* phrasing
    ("... the answer is X", last in ``_ANCHORS``) always won even when a stronger "Final
    Answer:" appeared later in the response -- e.g. "the answer is 42. Final Answer: 7"
    returned "42". We now keep the tail of the match with the greatest end position; ties
    (the "Answer:" embedded inside "Final Answer:") break toward the stronger/earlier
    pattern via ``-idx``.
    """
    text = text or ""
    best = None
    best_key = None  # (end_position, -pattern_index); larger wins
    for idx, pat in enumerate(_ANCHORS):
        for m in re.finditer(pat, text):
            tail = text[m.end():].strip()
            if not tail:
                continue
            key = (m.end(), -idx)
            if best_key is None or key > best_key:
                best_key = key
                best = tail.split("\n")[0].strip()
    return best


def boxed_values(text: str) -> Sequence[str]:
    return [m.strip() for m in re.findall(r"\\boxed\{([^{}]+)\}", text or "")]


def last_mc_letter(text: str, letters: str = "ABCD") -> Optional[str]:
    """The model's chosen multiple-choice letter.

    Order: \\boxed{X} -> an explicit anchor -> the LAST standalone letter. Taking the FIRST
    standalone letter matched the leading "A" of an ordinary sentence.
    """
    if not text:
        return None
    up = text.upper()
    cls = f"[{letters}]"

    for b in reversed(boxed_values(up)):
        m = re.fullmatch(rf"\(?({cls})\)?\.?", b.strip())
        if m:
            return m.group(1)

    span = anchored_span(up)
    if span:
        m = re.match(rf"\(?({cls})\)?\b", span.strip())
        if m:
            return m.group(1)

    whole = up.strip().strip("().")
    if len(whole) == 1 and whole in letters:
        return whole

    found = re.findall(rf"\b({cls})\b", up)
    return found[-1] if found else None


def final_number(text: str) -> Optional[str]:
    """The model's stated numeric answer: anchored/boxed first, else the last number."""
    if not text:
        return None

    def _num(s: str) -> Optional[str]:
        m = re.findall(r"-?\d+(?:\.\d+)?", (s or "").replace(",", ""))
        return m[0] if m else None

    for b in reversed(boxed_values(text)):
        n = _num(b)
        if n is not None:
            return n
    span = anchored_span(text)
    if span:
        n = _num(span)
        if n is not None:
            return n
    allnum = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return allnum[-1] if allnum else None


def binary_verdict(text: str,
                   positive: Sequence[str] = ("yes", "true", "valid", "correct"),
                   negative: Sequence[str] = ("no", "false", "invalid", "incorrect")) -> Optional[bool]:
    """Resolve a yes/no verdict, or None when the response is ambiguous.

    Scanning for `\\b(yes|true|valid)\\b` anywhere marked "No, this is not valid" as YES,
    because it never checked whether the opposite verdict was also present. Here an
    anchored verdict wins; otherwise the LAST verdict word decides, and a response with
    neither returns None (the caller must not guess).
    """
    if not text:
        return None
    t = text.lower()
    pos = "|".join(re.escape(p) for p in positive)
    neg = "|".join(re.escape(n) for n in negative)

    span = anchored_span(t)
    if span:
        m = re.match(rf"\W*\b({pos}|{neg})\b", span)
        if m:
            return m.group(1) in positive

    # A verdict word can be negated ("this is not valid"), so a bare token scan is not
    # enough: flip the polarity when a negation immediately precedes it.
    neg_prefix = re.compile(r"(?:\bnot\b|n't\b|\bno\b|\bnever\b|\bisn't\b|\bcannot\b)\s+(?:\w+\s+){0,2}$")

    def _polarity(match: "re.Match") -> bool:
        word = match.group(1)
        base = word in positive
        preceding = t[max(0, match.start() - 40):match.start()]
        if neg_prefix.search(preceding):
            return not base
        return base

    hits = list(re.finditer(rf"\b({pos}|{neg})\b", t))
    if not hits:
        return None
    return _polarity(hits[-1])
