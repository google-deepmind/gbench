# frames

Google FRAMES (Factuality, Retrieval, And reasoning MEasurement Set, arXiv 2409.12941): 824
multi-hop questions whose answers need facts from 2-15 Wikipedia articles, with numerical,
tabular, temporal, multi-constraint and post-processing reasoning.

## Dataset
- `google/frames-benchmark`, split `test` (824 rows), pinned at revision
  `58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef`.
- The per-category table uses each row's `reasoning_types`.

## Protocol
- **The paper's "Naive Prompt" single-step setting** (Section 4.1): the question alone is the
  user turn. gbench gives the model no search tool and no Wikipedia articles, so it answers from
  parametric memory. This is the paper's closed-book baseline; it is not comparable with the
  paper's BM25-retrieval or Oracle rows, or with numbers from browsing / retrieval agents.
- The autorater gets the paper's auto-rating prompt **verbatim** (appendix figure: Figure 6 in
  arXiv v3, Figure 7 in v1), transcribed from the paper's LaTeX source (`auto_rate_prompt.tex`)
  line by line, with its system turn `You are a helpful assistant.` The source's typographic
  apostrophe (U+2019, in `LLM's response`) and en dash (U+2013, after `substance of the answers`)
  are sent as those characters (written as escapes in the ASCII source). It asks for an
  `Explanation:` and then a `Decision:` of `TRUE` or `FALSE`.
- The verdict is the last `Decision` label's `TRUE` / `FALSE` (quotes and markdown around the
  label are tolerated, since the format line shows it in quotes). A reply with no Decision is an
  autorater failure, not the model's answer (`JUDGE_FAILED`, `scoring_excluded: true`): left out
  of the accuracy, counted in `judge_failures` / `failed_requests` (the run is
  `completed_with_errors`) and named in `leaderboard_comparable_reason`, instead of a guessed
  verdict or a wrong answer. The paper publishes no parser that would decide it otherwise.
- A request that got no reply from the served endpoint is a harness error: not sent to the
  autorater, marked `not_judged_failed_request`, counted in `failed_requests`, left out of the
  score and named in `status_reason` (the run is `completed_with_errors`). An empty reply the
  server returned is the model's outcome: not sent to the autorater, marked
  `not_judged_empty_response`, counted in `empty_responses`, and it scores wrong.
- Each question is sent, and shown to the autorater, exactly as the dataset holds it (67 of the
  824 prompts end in whitespace, which is kept), and so is its gold answer.

## Headline metric and how to read the result
- `accuracy` (headline) = Decision TRUE / graded questions, in percent (the paper reports the
  same accuracy, e.g. 0.408 for Gemini-Pro-1.5 with the Naive Prompt).
- A judge-cascade outage and an autorater failure are excluded from the denominator and counted
  in `judge_outages` / `judge_failures`; when nothing could be graded the run is `failed` and its
  score withheld.
- `judge` is `gbench-gemini-cascade`, `judge_models` counts the answered autorater calls per
  Gemini model (one call per graded question) and `judge_cascade` lists the configured cascade
  order; each trace keeps `judge_reply` (the explanation and decision) and `judge_model`.
- `leaderboard_comparable` is **False**. `leaderboard_comparable_reason` says why: the autorater
  is gbench's Gemini cascade rather than the paper's `Gemini-Pro-1.5-0514` (a gbench
  convention), and, on an `--eval-limit` run, that only a subset was graded; it also names
  autorater replies without a Decision line and replies cut off by the run's token budget rather
  than at an official cap
  (`replies_cut_off_by_budget`; `replies_cut_off_in_reasoning` counts those that ran out inside
  the reasoning channel with no answer). The paper publishes no autorater sampling settings, so
  the autorater runs with the paper's system turn at gbench's deterministic judge temperature 0;
  `judge_sampling` records this.

## Requirements
- An OpenAI-compatible `/v1` chat endpoint serving the model under test.
- **`GEMINI_API_KEY`** or **`GEMINI_API_KEYS`** (one key, or a comma/semicolon list;
  `GEMINI_API_KEYS` wins when both are set) for the autorater. The suite checks it with a live
  call before generating and **hard-errors** (`infra_required`) when it is unset or rejected;
  there is no substring fallback. Network access to the Gemini API
  (`generativelanguage.googleapis.com`) is needed; `GEMINI_OPENAI_BASE_URL` moves only the key
  check.
- Packages: the base install (`datasets`) plus `google-genai` (the `evals` extra:
  `pip install -e ".[evals]"`).
- Hugging Face Hub access to download `google/frames-benchmark` (public; `HF_TOKEN` optional).
- Optional judge knobs (shared by all judged suites): `GBENCH_JUDGE_MODELS`,
  `GBENCH_JUDGE_MODEL`, `GBENCH_JUDGE_CASCADE_ROUNDS`, `GBENCH_JUDGE_BACKOFF`.
- Sampling: `GBENCH_FRAMES_TEMPERATURE` overrides the temperature for this suite; otherwise the
  rule applies (1.0 with `--eval-thinking`, 0.0 without; `--temperature` replaces it).

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals frames --eval-thinking \
       --max-output-tokens 32768
```
`--eval-thinking` is recommended (multi-hop). Add `--eval-limit 20` for a quick stratified subset
(flagged in the reason). `--max-output-tokens` is required by the CLI and is sent as every
request's `max_tokens`.

The run logs its planned workload in one line at its start (for example `[frames] --eval-limit 5:
5 sample(s) x 1 attempt = 5 generations`) and records it in the result as `eval_limit_plan`
(`limit`, `samples`, `per_setting`, `attempts_per_sample`, `generations`). The
suite is scored in a batch and runs 1 attempt per sample whatever `--attempt-count` asks, so the
plan says 1 attempt.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered questions
(round-robin: positions I, I+N, I+2N, ...), selected before the `--eval-limit` draw, which is then
taken within the shard (N shards run up to N x `--eval-limit` questions). The result records
`shard`, `shard_index` and `shard_count`, and a shard of more than one is not
leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`); `gbench
--merge` combines the shards' results as docs/distributed-gbench.md describes: it re-counts the
union of their traces, and refuses a suite whose headline is not a correct / scored count over
them unless the suite defines its own merge.
