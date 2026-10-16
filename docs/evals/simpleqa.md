# simpleqa

OpenAI SimpleQA: short-form factuality with abstention. Each of the 4,326 questions has one
short, verifiable answer; the benchmark rewards answering correctly and not answering over
answering wrongly.

## Dataset
- `basicv8vc/SimpleQA`, split `test`, pinned at revision
  `e319282ab125c3dbd0c7fd00be2e4dd54e7e8f94`. At that revision `simple_qa_test_set.csv` is
  byte-identical to simple-evals' canonical
  `https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv`
  (sha256 `feee3f7e7db3617e94e8fcf1977b756ec420ef8568f4e0fcbbe0e92e9d5fc032`, 4,326 rows).
- The per-category table uses the `topic` inside each row's `metadata`.

## Protocol (openai/simple-evals `simpleqa_eval.py`, commit `652c89d0ca9d`)
- The model gets the question as the user turn, sent as simple-evals' chat sampler sends it
  (`simple_evals.py` at `652c89d0`, the `gpt-4o` entry, lines 236-240), with and without
  `--eval-thinking`: `ChatCompletionSampler(system_message=OPENAI_SYSTEM_MESSAGE_API,
  max_tokens=2048)`, so every request carries the system message `You are a helpful
  assistant.`. The temperature is gbench's rule, not the sampler's default 0.5: **1.0 with
  `--eval-thinking`, 0.0 without** (`official_sampling` records the 0.5 with
  `temperature_matches_official: false`).
  Without `--eval-thinking`
  it also carries **`max_tokens` 2048**. A reply that reaches 2048 tokens is cut there and graded
  as it stands, as simple-evals grades it (`replies_at_sampler_max_tokens` counts them, as
  `replies_at_official_max_tokens` does). The 2048 is the chat sampler's official cap in gpqa's
  `SIMPLE_EVALS_SAMPLING` (shared with gpqa, applied per mode by base's token-limit policy), so
  `--max-output-tokens` is not applied (`max_output_tokens_note` says so; `official_sampling`
  records `max_tokens` 2048 and `max_tokens_applied: true`). With `--eval-thinking` the 2048 is
  not applied, because it was written for non-reasoning generation: each request carries the
  run's budget, `--max-output-tokens`, instead (`official_sampling` records `max_tokens` 2048
  with `max_tokens_applied: false` and `max_tokens_sent`; `max_output_tokens_note` says so).
  `official_sampling` also records the temperature sent and where it came from
  (`temperature_matches_official`); `sampler_max_tokens` is 2048 without `--eval-thinking` and
  null with it, and `max_tokens_sent` lists the values sent.
- simple-evals' reasoning-model entries (`simple_evals.py` lines 145-212: o3, o4-mini and o1-pro
  through `ResponsesSampler` with `reasoning_model=True`, o1, o1-preview, o1-mini and o3-mini
  through `OChatCompletionSampler`), which send no system message, no temperature and no
  max_tokens, are not used: they are keyed on OpenAI model names and fit what OpenAI's hosted
  reasoning models accept, so they are not reasoning settings for a served model, which takes
  the chat sampler in both modes (see gpqa).
- The grader gets simple-evals' `GRADER_TEMPLATE` **verbatim** (its few-shot CORRECT /
  INCORRECT / NOT_ATTEMPTED rubric with the significant-figure and inferable-information
  rules). The call follows simple-evals' `grading_sampler`
  (`ChatCompletionSampler(model="gpt-4.1-2025-04-14", system_message=OPENAI_SYSTEM_MESSAGE_API,
  max_tokens=2048)`): the system message `You are a helpful assistant.`, the sampler's default
  temperature **0.5**, its `max_tokens` 2048 for the reply and no response format, with the
  gbench Gemini cascade as the model. gpt-4.1 does not think; the Gemini judge counts its thinking
  against `max_output_tokens`, so the grader is sent **`max_output_tokens` 10240**: the 2048 plus
  the Gemini thinking allowance of 8192 (`gemini_budget.GEMINI_THINKING_ALLOWANCE`, the rule
  wildbench's judge follows too), so its reply keeps upstream's 2048 tokens. A grader reply that
  still stops there is parsed as it arrived, as simple-evals parses a reply cut at its
  `max_tokens`, and a call whose every answer stopped there with no text ends as a judge outage
  (excluded and counted in `judge_outages`); both are counted (below) and named in
  `leaderboard_comparable_reason`. `judge_sampling` records these settings (`max_output_tokens`
  10240, `upstream_max_tokens` 2048, `thinking_allowance` 8192, `max_output_tokens_note`).
  HealthBench and BrowseComp share this grader call.
- The gold target enters the grader prompt exactly as the CSV stores it, surrounding whitespace
  included (simple-evals passes `row["answer"]` unchanged; 211 of the 4,326 answers have leading
  or trailing whitespace).
- The grade is simple-evals' own parse: the first `A`, `B` or `C` in the grader's reply
  (A = CORRECT, B = INCORRECT, C = NOT_ATTEMPTED), and `C` when the reply has none.
- An empty model reply (the endpoint answered with no text) is the model's answer and is not sent
  to the grader. It gives no answer and contradicts nothing, so it counts as NOT_ATTEMPTED (and
  under `failed_requests` / `empty_responses`).
- A request that got no reply from the served endpoint (a server error, a timeout, an
  over-context rejection) is a **harness error**, never the model's outcome. It is not graded
  (`judge_grade: not_judged_failed_request`), carries `harness_error` and `scoring_excluded: true`,
  and is left out of every figure below; it is counted in `failed_requests` and `scoring_excluded`
  and named in `status_reason`, and the run is `completed_with_errors`.

## Headline metric and how to read the result
- `accuracy` (headline) = simple-evals' `score`: **is_correct over all graded questions**,
  i.e. `correct / (correct + incorrect + not_attempted)`, in percent.
- Reported alongside, as simple-evals prints them:
  - `correct`, `incorrect`, `not_attempted`: the three-way split, in percent.
  - `accuracy_given_attempted` = correct / (correct + incorrect).
  - `f_score` = harmonic mean of `correct` and `accuracy_given_attempted`.
- A judge-cascade outage (every Gemini model failed every retry round) is excluded from
  every figure and counted in `judge_outages`; it is never scored as wrong.
- When no question was graded (every one a harness error or a judge outage), the run is `failed`
  and its score withheld: `accuracy`, `correct`, `incorrect`, `not_attempted`,
  `accuracy_given_attempted` and `f_score` are null and `accuracy_withheld_reason` says why,
  never 0.
- `judge` is `gbench-gemini-cascade`, `judge_models` counts the answered judge calls per Gemini
  model (one call per graded question) and `judge_cascade` lists the configured cascade order;
  each trace keeps `judge_reply`, `judge_model`, `judge_finish_reason` (how the grader's reply
  ended; `MAX_TOKENS` when it stopped at the 10240 limit) and `simpleqa_grade`.
  `judge_replies_at_max_output_tokens` counts the grader replies that stopped at the 10240 (parsed
  as they arrived), and `judge_outages_after_max_output_tokens` the judge outages whose cascade
  answers stopped there with no text (each such trace has `judge_max_tokens_without_text`).
- `grader_replies_not_bare_letter` counts grader replies that were not just `A`, `B` or `C`.
  simple-evals' parse reads the first capital A/B/C anywhere in the reply (so "Based on ..."
  reads as B); if the count is not 0, check `judge_reply` on those traces.
- `leaderboard_comparable` is **False**. `leaderboard_comparable_reason` says why: the grader
  is gbench's Gemini cascade, not simple-evals' `gpt-4.1-2025-04-14` (a gbench convention), and,
  on an `--eval-limit` run, that only a subset was graded; a temperature other than
  simple-evals' 0.5 (gbench's 1.0 with thinking or 0.0 without, or a value set with
  `--temperature` or the variable below) is named too, and so are
  no-think requests whose prompt left less than 2048 tokens of the context window (they got a
  smaller `max_tokens`) and replies cut off by a budget other than the no-think 2048 (the run's
  budget with `--eval-thinking`, or a `max_tokens` the context window lowered:
  `replies_cut_off_by_budget`, with `replies_cut_off_in_reasoning` for those that ran out inside
  the reasoning channel). The
  prompt, sampler settings, rubric, parse and aggregation are simple-evals'.

## Requirements
- An OpenAI-compatible `/v1` chat endpoint serving the model under test.
- **`GEMINI_API_KEY`** or **`GEMINI_API_KEYS`** (one key, or a comma/semicolon list;
  `GEMINI_API_KEYS` wins when both are set) for the grader. The suite checks it with a live call
  before generating and **hard-errors** (`infra_required`) when it is unset or rejected; there is
  no heuristic fallback. Network access to the Gemini API (`generativelanguage.googleapis.com`)
  is needed; `GEMINI_OPENAI_BASE_URL` moves only the key check.
- Packages: the base install (`datasets`) plus `google-genai` (the `evals` extra:
  `pip install -e ".[evals]"`).
- Hugging Face Hub access to download `basicv8vc/SimpleQA` (public; `HF_TOKEN` optional).
- Optional judge knobs (shared by all judged suites): `GBENCH_JUDGE_MODELS` (cascade order),
  `GBENCH_JUDGE_MODEL`, `GBENCH_JUDGE_CASCADE_ROUNDS`, `GBENCH_JUDGE_BACKOFF`.
- Sampling: `GBENCH_SIMPLEQA_TEMPERATURE` (or `--temperature`) overrides the temperature for
  this suite, in both modes; otherwise gbench's rule applies: 1.0 with `--eval-thinking`, 0.0
  without. `max_tokens` has no override of its own: 2048 without
  `--eval-thinking`,
  `--max-output-tokens` with it.

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals simpleqa --max-output-tokens 16384
```
Add `--eval-limit 20` for a quick stratified subset of 20 questions (flagged in the reason;
The run logs its planned workload at its start and records it as `eval_limit_plan`.) and
`--eval-thinking` to measure the thinking mode. `--max-output-tokens` is required by the CLI; a
no-think request does not apply it (it sends simple-evals' `max_tokens` 2048), and a thinking
request sends it in place of the 2048.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered questions
(round-robin: positions I, I+N, I+2N, ...), selected before the `--eval-limit` draw, which is then
taken within the shard (N shards run up to N x `--eval-limit` questions). The result records
`shard`, `shard_index` and `shard_count`, and a shard of more than one is not
leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`); `gbench
--merge` combines the shards' results as docs/distributed-gbench.md describes: it re-counts the
union of their traces, and refuses a suite whose headline is not a correct / scored count over
them unless the suite defines its own merge. SimpleQA's merge (`merge_shards`) is that re-count,
with the grader cut-off counts (`judge_replies_at_max_output_tokens`,
`judge_outages_after_max_output_tokens`) summed over the shards and their reason named once, over
the sums, in `leaderboard_comparable_reason`.
