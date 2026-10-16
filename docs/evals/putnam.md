# putnam setup

PutnamBench's informal problems: William Lowell Putnam Competition problems (1962-2025) that the
model solves in natural language. PutnamBench's leaderboard is formal only (Lean, Isabelle,
Rocq), so this track is a gbench protocol and is never leaderboard-comparable. For the
machine-checked Lean 4 track, see [putnam_formal](putnam_formal.md).

## Dataset (pinned)
`informal/putnam.json` of `trishullab/PutnamBench` at commit
`97b5e9458b04e3d1044e8f8afe1724ff133f5a53` (main on 2026-09-24), the commit putnam_formal also
uses: 673 problems (one more than the Lean track, `putnam_1997_a1`, has no Lean formalization).
345 have an `informal_solution` ("Show that the solution is ..."); the other 328 are proofs,
marked `None.` in the release and given to the judge without a reference. The file is fetched
with the rest of the release (a GitHub tarball of about 1 MB) on first use, cached under
`$GBENCH_CACHE_DIR/putnambench/<commit>/` (default `~/.cache/gbench`), and checked against its
git blob id `77675840ebadc2bc9b5a93f390bb0a678f33dbbc` on every load. The category is the
problem's first tag.

## Protocol
**Prompt** (gbench's; PutnamBench publishes no informal protocol): the problem statement,
followed by "Solve the mathematical problem and state your final answer on the last line in the
format: 'Final Answer: <answer>'." With `--eval-thinking` the instruction asks for a step-by-step
proof instead of a plain solution; the rest is unchanged.

**Scoring** splits on what the reference contains:
- a reference stating a *single* closed-form value (one `$...$` span, or a bare value; 170 of the
  345) is graded **on the answer**. The answer expression is extracted from both sides and
  normalized (LaTeX fractions and roots with nested braces, `\sqrt2`, `\dfrac`, `\left`/`\right`;
  a `\boxed{}` answer is read with its braces balanced). An explicitly anchored answer
  (`\boxed{}` / `Final Answer:`) that equals the reference is correct; an unanchored one is
  searched for the value with word boundaries, so `2` is not found inside `12`. A reply that does
  **not** match is not scored wrong on that basis: it goes to the **answer grader**, the
  AnswerAutoGrader prompt of IMO-AnswerBench (arXiv 2511.01846 App. A.5, the one
  [imo_answer_bench](imo_answer_bench.md) uses), with the problem statement, the reply and the
  reference, which decides whether the final answer is mathematically equivalent
  (`(-5+4\sqrt{2})/3` against `(4\sqrt{2}-5)/3`, "The sum is 3." against `$3$`) and answers
  `\boxed{Correct}` or `\boxed{Incorrect}`.
- everything else (a prose reference, a case split, or a proof problem with no reference) goes to
  the **LLM proof judge**, which receives the dataset's problem statement (not gbench's
  instruction around it) and the reference when the release has one, and answers
  `Grade: CORRECT / INCORRECT`.

Both judges run on the gbench Gemini judge cascade at temperature 0.0 (gbench's deterministic
judge default; the informal track has no upstream grader). The proof judge's verdict is its
last `Grade: CORRECT` / `Grade: INCORRECT` line, read after markdown and LaTeX around it are
removed (`**Grade:** CORRECT`, `Grade: **CORRECT**` and `**Grade**: INCORRECT` all count), or a
reply that is only the word CORRECT or INCORRECT; a line that restates the format
(`Grade: CORRECT / INCORRECT`) is not a verdict. A judge reply with no verdict (no
`\boxed{Correct|Incorrect}` for the answer grader; none of the above for the proof judge) is
asked again, 3 calls in total, and is then `JUDGE_FAILED`: a grading failure, not the model's
outcome, so the sample is `scoring_excluded` (left out of the accuracy's numerator and
denominator), counted in `judge_failures` (the run reports `completed_with_errors`) and named in
`leaderboard_comparable_reason`. A judge outage (every model of the cascade failed) is
`JUDGE_OUTAGE`: excluded from the accuracy too, never scored as wrong. A request that got no
reply from the served endpoint (error, timeout, over-context prompt) is a harness error, left
out of the score and counted; an empty reply the model ended itself is scored incorrect. A run
in which nothing was scored reports `status: failed` with `accuracy` null.

Each judged trace records `grading` (`answer_match`, `answer_grader` or `proof_judge`),
`judge_model` (the Gemini model that answered), `judge_raw` (its reply) and `judge_calls`. The
result records `judge`, `judge_cascade`, `judge_models` (answered judge calls per model),
`judge_sampling` and `grading_counts`.

The headline `accuracy` is the share of the scored problems graded correct (pass@1).

The problems with a single closed-form reference are graded on their answer, while every other
problem (including the 175 whose reference states an answer in prose or as a case split) is
graded by the proof judge on the rigor of the proof. This mixed standard is part of the gbench
protocol, which has no upstream counterpart.

## Requirements
- **`GEMINI_API_KEY`** (or `GEMINI_API_KEYS`, a comma- or semicolon-separated list), for the
  judges, which are **required**: a closed-form answer that does not match its reference and all
  503 other problems are judged. The suite **hard-errors** (`infra_required`, `status: "error"`,
  never skips, never scores proofs as wrong) if no key is set, or the live check rejects the key
  or finds a model of the judge cascade (`GBENCH_JUDGE_MODELS`, default `base.GEMINI_CASCADE`)
  that the API does not serve.
- **`google-genai`**, the judge client (in gbench's `evals` extra): `pip install -e ".[evals]"` in
  the gbench checkout.
- **Network** access to `codeload.github.com` on the first run (the dataset tarball) and to the
  Gemini API for judging.

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals putnam --eval-thinking \
       --max-output-tokens 32768
```
Add `--eval-limit 20` for a quick subset of 20 problems (stratified by tag). The run logs its plan on its first line and records it as `eval_limit_plan`.

## Notes / caveats
- **Sampling:** `GBENCH_PUTNAM_TEMPERATURE` overrides the temperature for this suite (else the run
  default: 0.0 greedy / 1.0 with `--thinking`); it takes precedence over `--temperature`. The
  judges are pinned at temperature 0.0.
- **Token limit:** the informal track has no official cap, so every request carries the run's
  `--max-output-tokens` (recorded in `max_tokens_sent`), with and without `--thinking`. A reply
  it cut off is scored on what arrived and counted (`replies_cut_off_by_budget`;
  `replies_cut_off_in_reasoning` counts those that ran out inside the reasoning channel with no
  answer), and `leaderboard_comparable_reason` names them.
- **`leaderboard_comparable` is always `false`**: PutnamBench has no informal leaderboard, and the
  judge is gbench's. `leaderboard_comparable_reason` also names an `--eval-limit` subset, a
  `GBENCH_SHARD` shard, judge failures (`JUDGE_FAILED`, excluded) and judge outages. The
  result's `dataset` field records the pinned release.
