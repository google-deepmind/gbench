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

"""lab_bench regression tests: no answer leak + LAB-Bench's own letter parse."""

from gbench.runners.eval_suites.lab_bench import _eval_lab_bench, _build_mcq_prompt


def test_extractor_tolerates_wrappers_around_letter():
    # LAB-Bench reads an [ANSWER] tag (chembench MCQ_REGEX_TEMPLATE_1 also accepts <ANS> and
    # <ANSWER>); an untagged "Final Answer: <C>" has no answer, as upstream scores it.
    assert _eval_lab_bench("reasoning...\n\n[ANSWER]C[/ANSWER]", "C")
    assert _eval_lab_bench("[ANSWER] C [/ANSWER]", "c")
    assert _eval_lab_bench("blah\n<ANSWER>B</ANSWER>", "B")
    assert _eval_lab_bench("<ANS>A</ANS>", "A")
    assert not _eval_lab_bench("reasoning...\n\nFinal Answer: <C>", "C")


def test_extractor_rejects_wrong_or_missing_answer():
    assert not _eval_lab_bench("[ANSWER]C[/ANSWER]", "A")     # wrong letter
    assert not _eval_lab_bench("I cannot decide.", "A")       # no answer line
    assert not _eval_lab_bench("", "A")                       # empty response


def test_prompt_does_not_leak_gold_letter():
    # The concluding instruction is format-only and must never name the correct
    # letter via the "Final Answer:" cue, regardless of where the shuffle places it.
    p1, g1 = _build_mcq_prompt("proto", "q1?", "aaa ideal", ["mmm", "zzz"])
    p2, g2 = _build_mcq_prompt("proto", "q2?", "zzz ideal", ["aaa", "mmm"])
    for p, g in ((p1, g1), (p2, g2)):
        assert f"Final Answer: <{g}>" not in p
        assert f"Final Answer: {g}" not in p
    assert p1.splitlines()[-1] == p2.splitlines()[-1]         # answer-independent footer


def test_gold_letter_uses_deterministic_shuffle_not_alphabetical():
    # Options are shuffled DETERMINISTICALLY by question (reproducible), not sorted,
    # so the correct answer is not pinned to the alphabetical slot (positional bias).
    _, g_a = _build_mcq_prompt("p", "same-q", "aaa", ["mmm", "zzz"])
    _, g_a2 = _build_mcq_prompt("p", "same-q", "aaa", ["mmm", "zzz"])
    assert g_a == g_a2 and g_a in ("A", "B", "C", "D")        # reproducible + valid letter
    # Across different questions the gold lands in more than one slot (bias removed);
    # under a plain alphabetical sort "aaa" would always be "A".
    letters = {_build_mcq_prompt("p", f"q{i}", "aaa", ["mmm", "zzz"])[1] for i in range(8)}
    assert len(letters) > 1


def test_one_response_is_credited_for_at_most_one_gold():
    """Regression: the old scorer OR'ed several tiers and ended
    in a substring check, so one reply was scored correct for golds A, B and D (the article
    "A", the rejected "Option (B)", and the real answer D). The tag alone decides now."""
    resp = ("Step 2 is the error. A protocol this long rarely fails at step 1. "
            "Option (B) would lower yield, so no.\n[ANSWER]D[/ANSWER]")
    credited = [g for g in "ABCDEFGHIJ" if _eval_lab_bench(resp, g)]
    assert credited == ["D"]


def test_reasoning_letters_never_count_without_an_answer():
    assert not _eval_lab_bench("A protocol step is wrong. (B) and **C** are both plausible.", "A")
    assert not _eval_lab_bench("A protocol step is wrong. (B) and **C** are both plausible.", "B")
    assert not _eval_lab_bench("A protocol step is wrong. (B) and **C** are both plausible.", "C")


def test_last_final_answer_wins_and_other_anchors():
    # prepare_mcq_answer keeps the sorted set of tag matches and run_regex takes the first
    # offered letter, so two tags keep the alphabetically first one; untagged anchors
    # ("Final Answer:", "the answer is", \boxed{}) are not read.
    assert _eval_lab_bench("[ANSWER]C[/ANSWER]\nWait.\n[ANSWER]A[/ANSWER]", "A")
    assert not _eval_lab_bench("[ANSWER]C[/ANSWER]\nWait.\n[ANSWER]A[/ANSWER]", "C")
    assert not _eval_lab_bench("so the result is \\boxed{B}", "B")
    assert not _eval_lab_bench("After weighing it, the answer is (D).", "D")
    assert _eval_lab_bench("C", "C")                          # a reply made only of a letter
    assert not _eval_lab_bench("(C).", "C")
    assert not _eval_lab_bench("Final Answer: Definitely the protocol is fine", "D")


def test_hedged_answers_score_nothing():
    """Review finding: 'Final Answer: B or D' was credited whenever B was gold."""
    for resp in ("Final Answer: B, D", "Final Answer: B or D", "Final Answer: B/D",
                 "Final Answer: B & D"):
        assert not any(_eval_lab_bench(resp, g) for g in "ABCDEFG"), resp


def test_honest_single_answers_in_common_phrasings_are_accepted():
    # Only LAB-Bench's own formats are answers: an answer tag, or a reply made only of
    # letters. Every other phrasing is scored wrong, exactly as upstream scores it.
    for resp, gold in (("[ANSWER]C[/ANSWER]", "C"), ("[ANS]C[/ANS]", "C"),
                       ("<ans>C</ans>", "C"), ("[ANSWER]C. Add buffer[/ANSWER]", "C"),
                       ("C", "C"), ("C, D", "C"), ("C D", "C")):
        assert _eval_lab_bench(resp, gold), resp
    for resp in ("Final Answer: Option C", "Answer: C", "The answer is option C.",
                 "**Final Answer:**\n**C**", "final answer: c"):
        assert not _eval_lab_bench(resp, "C"), resp


def test_lowercase_words_and_never_offered_letters_do_not_count():
    """Review finding: the article 'a', the pronoun 'I' and letters beyond the offered A-G
    were read as answers."""
    assert not _eval_lab_bench("my final answer: a quick recap\nThe answer is C", "A")
    assert not _eval_lab_bench("[ANSWER]I[/ANSWER]", "I")      # not an offered letter (A-G)
    assert not _eval_lab_bench("[ANSWER]H[/ANSWER]", "H")
    assert not _eval_lab_bench("[ANSWER]c[/ANSWER]", "C")      # the tag regex takes capitals


def test_option_text_after_the_letter_is_not_a_hedge():
    """Round-2 review: option text after the letter must not turn an answer into a hedge."""
    assert _eval_lab_bench("[ANSWER]B. Incubate at 37 C for 1 hour[/ANSWER]", "B")
    assert _eval_lab_bench("[ANSWER]C, add buffer A before step 3[/ANSWER]", "C")
    assert _eval_lab_bench("[ANSWER]A. Use fresh reagents[/ANSWER]", "A")


def test_the_last_explicit_answer_wins_and_articles_are_not_letters():
    # A leading article, or letters padded with spaces or newlines, are not a letters-only
    # reply for LAB-Bench, so they score nothing; only the tag counts.
    assert _eval_lab_bench("A protocol step 4 is wrong, so [ANSWER]C[/ANSWER]", "C")
    assert not _eval_lab_bench("A protocol step 4 is wrong, so [ANSWER]C[/ANSWER]", "A")
    for padded in (" B", "B ", "B\n", "\nB"):
        assert not _eval_lab_bench(padded, "B"), repr(padded)


def test_long_whitespace_runs_do_not_hang_the_extractor():
    import time
    t = time.time()
    _eval_lab_bench("[ANSWER]" + " " * 200_000 + "x", "A")
    assert time.time() - t < 1.0
