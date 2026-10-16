# imo_answer_bench

**IMO-AnswerBench** (Google DeepMind IMO-Bench; Luong et al., "Towards Robust Mathematical
Reasoning", EMNLP 2025, arXiv 2511.01846): 400 olympiad problems with a short final answer, 100
each of **Algebra / Combinatorics / Geometry / Number theory** (`category_accuracy` breaks down by
those four). Answers are LaTeX expressions, sets and tuples; correctness is decided by an LLM
grader, with no partial credit.

## Dataset
- **Version 2**: `imobench/answerbench_v2.csv` from
  [google-deepmind/superhuman](https://github.com/google-deepmind/superhuman) at commit
  `955558ad67ef24dccd27545c326988bdad1b4471` (the last commit touching that file, 2026-02-13),
  CC-BY-4.0. Upstream deprecated v1 (`answerbench.csv`, which the `OpenEvals/IMO-AnswerBench` HF
  mirror still serves) on 2026-02-12; v2 corrects about 10 gold answers and about 23 problem
  statements (for example combinatorics-005 changes from `1431655765` to `1431655764` and geometry-002
  from `4` to `1`).
- The file is downloaded once from `raw.githubusercontent.com` at that commit, verified against
  sha256 `275877a9d988d85278fad3a5f8a41d7f83393a60bf259531ec0a5161e6b21cf9`, and cached at
  `~/.cache/gbench/imo_answer_bench/` (`GBENCH_CACHE_DIR` overrides the root). A mismatch is a hard
  error.
- The pinned file has one malformed row: `imo-bench-algebra-036` lost the quote that closes its
  problem field, so a CSV reader folds the answer into the problem. The loader restores that
  quote (as the v1 row is quoted) and checks that the result is 400 well-formed rows with 100
  per category; anything else is a hard error.
- `--eval-limit N` draws a subset stratified over the four categories.

## Protocol
- **Solver prompt**: the paper does not publish the prompt it used to query models. gbench sends
  `Solve the following olympiad problem:` + the problem + a request for the final answer as
  `Final Answer: \boxed{answer}` (with `--eval-thinking`, preceded by "Reason step by step").
  The grader extracts the answer from the whole solution, so the format request only has to make
  the final answer clear.
- **Grader**: the AnswerAutoGrader prompt of the paper's Appendix A.5 ("Query prompt for
  AnswerAutoGrader", `# System Role: Deterministic Mathematical Autograder`), taken from the
  paper's LaTeX source with the typesetting removed and the wording, numbering (`# 1.`, `# 3.`,
  `# 4.`) and the `Output exmaple` typo kept. It receives the dataset problem, the **full** model
  solution (reasoning-channel text excluded) and the golden answer; it extracts the final answer
  itself, checks equivalence (algebraic, numerical, order-free sets) and ends with
  `\boxed{Correct}` or `\boxed{Incorrect}`. The last boxed verdict decides. Every non-empty
  response is graded; there is no textual-match shortcut. Each graded trace keeps the grader's
  last reply in `judge_raw` (its `<thinking>` analysis and boxed verdict, so the verdict can be
  audited and parsed again), the model that gave it in `judge_model` and the number of grader
  calls in `judge_calls`.
- A reply with no boxed verdict is asked again, up to 3 calls in total, then the sample is marked
  `judge_grade: JUDGE_FAILED`: a grading failure, not the model's outcome, so it is
  `scoring_excluded` (left out of the numerator and the denominator), counted in
  `judge_failures`, named in `leaderboard_comparable_reason`, and the run is
  `completed_with_errors`. A judge outage (every model of the cascade failed) is `JUDGE_OUTAGE`
  and is excluded from the denominator too. An empty response the model ended itself is
  incorrect without a grader call (no text to grade) and stays in the
  denominator; a request that got no reply from the endpoint (error, timeout, over-context
  prompt) is a harness error, left out of the score and counted. A run in which nothing was
  scored reports `status: failed` with `accuracy` null.
- **Runs**: the paper's results are averaged over 8 runs, so the suite runs at least 8 attempts
  per problem (a larger `--attempt-count` is honoured).
- **Sampling**: temperature 0.0 without `--eval-thinking`, 1.0 (the model's shipped
  `generation_config.json`) with it; `--temperature` or `GBENCH_IMO_ANSWER_BENCH_TEMPERATURE`
  override. `max_tokens` is `--max-output-tokens` (required by the CLI).

## Headline metric and how to read it
`accuracy` = correct attempts / graded attempts: accuracy averaged over the 8 (or more) runs.
`attempts` adds pass@k, pass^k and how many problems changed verdict across runs. The result
records `dataset` (repo, commit, file), `judge` (`gbench-gemini-cascade`), `judge_models`
(answered grader calls per model, retries included), `judge_cascade` (the configured cascade
order) and `judge_sampling`: the paper publishes no grader sampling settings, so the grader runs
at gbench's deterministic judge temperature 0.

`leaderboard_comparable` is always **false**, and `leaderboard_comparable_reason` says why: the
paper's AnswerAutoGrader runs on Gemini 2.5 Pro, while gbench runs the same prompt on its Gemini
judge cascade (the models that answered are named). On a no-think run the reason also notes that
the 8 runs are greedy (temperature 0.0), so they differ only through server nondeterminism
rather than sampling. An `--eval-limit` subset is also named, and so are replies cut off by the
run's token budget rather than at an official cap (`replies_cut_off_by_budget`;
`replies_cut_off_in_reasoning` counts those that ran out inside the reasoning channel with no
answer), and so are responses that got no grader verdict (`JUDGE_FAILED`).

> **Dataset history:** this suite once loaded `AI-MO/NuminaMath-CoT` rows mentioning "IMO" (a
> training corpus, not the benchmark), then the deprecated v1 answer set. Numbers from either are
> not comparable with the current ones.

## Requirements
- A running OpenAI-compatible `/v1` endpoint.
- **`GEMINI_API_KEY`** or **`GEMINI_API_KEYS`** (one key, or a comma/semicolon list;
  `GEMINI_API_KEYS` wins when both are set) for the AnswerAutoGrader. It is required: a missing
  or rejected key **hard-errors** (`infra_required`) before generation, and there is no
  textual-match fallback.
- The `google-genai` client (in the `evals` extra).
- Network access to `raw.githubusercontent.com` for the first run (or a pre-seeded cache file
  with the pinned sha256).

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals imo_answer_bench --eval-thinking \
       --max-output-tokens 32768
```
The full run is 3,200 generations (400 problems x 8 runs). Add `--eval-limit 20` for a quick
(non-comparable) subset of 20 problems (160 generations: the 8 runs are attempts, not more problems).
The run logs its plan on its first line and records it as `eval_limit_plan`.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered problems
(round-robin: positions I, I+N, I+2N, ...), selected before the `--eval-limit` draw, which is then
taken within the shard (N shards run up to N x `--eval-limit` problems). The result records
`shard`, `shard_index` and `shard_count`, and a shard of more than one is not
leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`); `gbench
--merge` combines the shards' results as docs/distributed-gbench.md describes: it re-counts the
union of their traces, and refuses a suite whose headline is not a correct / scored count over
them unless the suite defines its own merge.
