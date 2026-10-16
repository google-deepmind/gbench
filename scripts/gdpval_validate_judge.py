#!/usr/bin/env python3
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

"""Validate the gdpval pairwise judge against GDPval's own gold deliverables.

The question underneath any gdpval score is "is the judge any good?". This answers it
without a model in the loop. gdpval scores a model by a blind, position-swapped pairwise
comparison of its deliverable against the expert reference (gdpval._PAIRWISE_PROMPT, graded by
gbench's Gemini cascade, base.judge_generate_cascade). GDPval ships the reference artifact for
each task (`deliverable_files`), rendered to text the same way the suite renders it
(gdpval.render_reference), so the judge can be put in two situations whose right answer is
known:

  identical   the reference against itself, in both positions: a judge that is not biased by
              position calls it a tie (position-swapped mean 0.5)
  degraded    the full reference against its own first quarter, in both positions: a judge
              that reads the substance prefers the full reference both times (mean 1.0)

Interpreting the output:

  identical ~0.5 and degraded ~1.0   the judge is usable; model scores are meaningful
  identical far from 0.5             the judge prefers a position (A or B), which the suite's
                                     position swap cancels only on average
  degraded well below 1.0            the judge cannot tell a complete deliverable from a
                                     truncated one; the win rate measures the judge

Every judge call goes through the same cascade, prompt and parser the suite uses, and each row
names the Gemini model(s) that answered. The key gate (base.gemini_required_skip) runs first,
so an unset or rejected key, or a cascade model the API does not serve, stops the script before
any call.

Usage:
    GEMINI_API_KEY=... python scripts/gdpval_validate_judge.py [--tasks 5]
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gbench.runners.eval_suites import gdpval as G  # noqa: E402
from gbench.runners.eval_suites.base import (  # noqa: E402
    gemini_required_skip, judge_cascade, judge_generate_cascade)

#: The share of the reference text the "degraded" candidate keeps.
DEGRADED_FRACTION = 0.25


def rubric_text(criteria):
    """The rubric as the suite renders it for the pairwise prompt."""
    return "\n".join(f'[{c.get("score")} pts] {c.get("criterion")}'
                     for c in criteria) or "(no rubric)"


async def compare(prompt_text, rubric, a_text, b_text):
    """One pairwise judge call, exactly as the suite makes it: ('a' | 'b' | 'tie' | None, the
    Gemini model that answered or JUDGE_OUTAGE)."""
    prompt = G._PAIRWISE_PROMPT.format(
        prompt=prompt_text[:G._MAX_JUDGE_CHARS],
        rubric=rubric[:G._MAX_RUBRIC_CHARS],
        a=a_text[:G._MAX_JUDGE_CHARS],
        b=b_text[:G._MAX_JUDGE_CHARS])
    text, model = await judge_generate_cascade(prompt)
    return (G._parse_winner(text) if text is not None else None), model


async def swapped_mean(prompt_text, rubric, first, second):
    """`first` against `second` in both positions: (the position-swapped mean score of
    `first`, or None when a call gave no verdict; the two verdicts; the models that answered)."""
    w1, m1 = await compare(prompt_text, rubric, first, second)     # first = A
    w2, m2 = await compare(prompt_text, rubric, second, first)     # first = B
    if w1 is None or w2 is None:
        return None, (w1, w2), (m1, m2)
    return (G._model_score(w1, "a") + G._model_score(w2, "b")) / 2.0, (w1, w2), (m1, m2)


async def main(n):
    from datasets import load_dataset
    ds = load_dataset(G._REPO, split="train", revision=G._REVISION)
    rows = [r for r in ds if r["deliverable_files"]][:n]
    print(f"Gemini cascade: {', '.join(judge_cascade())}\n")
    print(f"{'task':<10}{'occupation':<30}{'identical':>10}{'degraded':>10}   answered by")
    print("-" * 96)
    identical, degraded = [], []
    for r in rows:
        ref_text, note = G.render_reference(list(r["deliverable_files"]))
        if not ref_text:
            print(f"{r['task_id'][:8]:<10}{r['occupation'][:28]:<30}{'n/a':>10}{'n/a':>10}   "
                  f"{(note or 'no text rendering')[:40]}")
            continue
        rubric = rubric_text(json.loads(r["rubric_json"] or "[]"))
        prompt_text = str(r["prompt"] or "")
        same, _, m_same = await swapped_mean(prompt_text, rubric, ref_text, ref_text)
        cut = ref_text[:max(1, int(len(ref_text) * DEGRADED_FRACTION))]
        full, _, m_full = await swapped_mean(prompt_text, rubric, ref_text, cut)
        if same is not None:
            identical.append(same)
        if full is not None:
            degraded.append(full)
        models = ", ".join(sorted(set(m_same + m_full)))
        cell = (lambda v: "no verdict" if v is None else f"{v:.2f}")
        print(f"{r['task_id'][:8]:<10}{r['occupation'][:28]:<30}{cell(same):>10}"
              f"{cell(full):>10}   {models}")

    print("-" * 96)
    if identical:
        ties = sum(1 for v in identical if v == 0.5)
        print(f"identical: mean {sum(identical) / len(identical):.3f}, a tie on "
              f"{ties}/{len(identical)} (expected 0.5, a tie)")
    if degraded:
        wins = sum(1 for v in degraded if v == 1.0)
        print(f"degraded:  mean {sum(degraded) / len(degraded):.3f}, the full reference won "
              f"both calls on {wins}/{len(degraded)} (expected 1.0)")
    if not identical and not degraded:
        print("No comparison got a verdict from the judge (see the answered-by column).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=5)
    a = ap.parse_args()
    try:
        gemini_required_skip("gdpval", "gdpval_validate_judge")
    except RuntimeError as e:
        sys.exit(str(e))
    asyncio.run(main(a.tasks))
