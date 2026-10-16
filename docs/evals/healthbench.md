# healthbench

HealthBench (OpenAI): 5,000 realistic health conversations, each ending in a user turn the model
must answer, scored against physician-written **rubrics**. Each rubric item is a criterion with a
point value: positive for desirable behaviour, negative for undesirable behaviour (for example
"Fails to refer to emergency care", -8). An **LLM grader** decides, item by item, whether the
response meets the criterion.

## Dataset
- `openai/healthbench`, file `2025-05-07-06-14-12_oss_eval.jsonl` (the main HealthBench set),
  pinned to revision `40ee1968852fc57f625934251ac22be47077a8fb`. That file is byte-identical
  (sha256 `e99dd3c6372c10d6fcc5e385c5fae69d0dd40392dae56836ef9493ae324ecd2f`) to the one
  simple-evals downloads (`INPUT_PATH` in `healthbench_eval.py`).
- 57,237 rubric items; every conversation has at least one positive item. Each carries
  `example_tags` (a `theme:` tag, and for some a `physician_agreed_category:` tag) and each item
  carries rubric tags (`axis:`, `level:`, some `cluster:`).
- The model is sent the conversation's `prompt` messages as they are, after simple-evals' chat
  sampler system message, with and without `--eval-thinking` (see Sampling below).
- `--eval-limit N` draws a subset stratified over `example_tags`.
  The run logs its planned workload at its start and records it as `eval_limit_plan`.

## Protocol
Ported from [openai/simple-evals](https://github.com/openai/simple-evals) `healthbench_eval.py` at
commit `652c89d0ca9df547706735883097e9537d40dc47`:
- **Per-item grading** (`grade_sample`): one grader call per rubric item with `GRADER_TEMPLATE`
  (verbatim). The conversation plus the model's response are rendered as `role: content` blocks
  joined by blank lines, and the item as `[points] criterion` (`RubricItem.__str__`), so the
  grader sees the point value and applies the template's rule for negative criteria (a good
  response does not meet an undesirable criterion). The grader call follows simple-evals' default
  HealthBench grader (`grading_sampler`,
  `ChatCompletionSampler(model="gpt-4.1-2025-04-14", system_message=OPENAI_SYSTEM_MESSAGE_API,
  max_tokens=2048)`): the system message `You are a helpful assistant.`, the sampler's default
  temperature **0.5** and no response format (grading_sampler sends none; the template itself
  asks for a JSON object, read by `parse_json_to_dict` below) and its `max_tokens` 2048 for the
  reply, with the gbench Gemini cascade as the model. gpt-4.1 does not think; the Gemini judge
  counts its thinking against `max_output_tokens`, so the grader is sent **`max_output_tokens`
  10240**, the 2048 plus the Gemini thinking allowance of 8192 (`simpleqa.grader_config`,
  `gemini_budget.GEMINI_THINKING_ALLOWANCE`). A reply that still stops there without a boolean
  `criteria_met` is asked again like any unreadable reply, and a call whose every answer stopped
  there with no text is a judge outage; both are counted (below) and named in
  `leaderboard_comparable_reason`. `judge_sampling` records these settings.
- **Parsing** (`parse_json_to_dict`): strip a ```` ```json ```` fence and parse; the item counts only
  when `criteria_met` is a boolean. Upstream retries until it gets one; gbench asks up to 3 times
  per item and then marks the conversation `judge_grade: JUDGE_FAILED`. Such a conversation has no
  score: it is excluded from the mean, the 0.5 pass rate and the per-theme counts
  (`scoring_excluded: true`) and counted in `judge_failures` (the run status becomes
  `completed_with_errors`), never scored 0; its `judge_failed_items` keep, per failed item, the
  criterion, the Gemini model and the last reply. A judge outage (every model of the cascade
  failed) is `JUDGE_OUTAGE`, excluded the same way and counted in `judge_outages`. When no
  conversation receives a score (every one a request with no reply, a judge outage or a grading
  failure), the run is `failed` and its score withheld (`accuracy` null,
  `accuracy_withheld_reason`, which counts each cause), never reported as 0.
- **Requests that got no reply from the served endpoint** (a server error, a timeout, an
  over-context rejection) are **harness errors**, never the model's outcome. Such a conversation
  is not graded (no grader call; `judge_grade: not_judged_failed_request`), carries
  `harness_error` and `scoring_excluded: true`, and is left out of the mean, the 0.5 pass rate
  and every per-tag mean. It is counted once in `failed_requests` (and `scoring_excluded`), named
  in `status_reason` and in `leaderboard_comparable_reason`, and makes the run
  `completed_with_errors`. This differs from simple-evals, whose sampler returns
  `No response (bad request).` for a request its API rejects and grades that text like any
  response (and retries other errors until a response arrives). An empty reply (the endpoint
  answered with no text) is the model's answer and is graded as it stands.
- **Per-conversation score** (`calculate_score`): points of the items met / total positive points.
  It is **not clipped**: undesirable criteria that are met can take it below 0.
- **Aggregation** (`_aggregate_get_clipped_mean`): the mean over conversations, clipped to [0, 1]
  once. The same clipped mean is computed for every example tag (the conversation's score) and
  every rubric tag (`calculate_score` over that conversation's items with the tag, when any has
  positive points).
- **Sampling and system message**: as simple-evals samples the model under test
  (`simple_evals.py` at `652c89d0`). In both modes it runs as the chat sampler of the `gpt-4o`
  entry (lines 236-240), `ChatCompletionSampler(system_message=OPENAI_SYSTEM_MESSAGE_API,
  max_tokens=2048)`: the system message `You are a helpful assistant.` precedes the conversation
  and the temperature is gbench's rule, not the sampler's 0.5: **1.0 with `--eval-thinking`, 0.0
  without** (recorded as `temperature_matches_official: false`). `grade_sample` grades the sampler's
  `actual_queried_message_list`, so the grader's conversation starts with that `system: ...`
  block too. Without `--eval-thinking` every request carries **`max_tokens` 2048** (a reply that
  reaches it is cut there and graded as it stands, as simple-evals grades it;
  `replies_at_sampler_max_tokens` counts them, as `replies_at_official_max_tokens` does; the 2048
  is the chat sampler's official cap in gpqa's `SIMPLE_EVALS_SAMPLING`, which gpqa, simpleqa and
  this suite pass to base's token-limit policy), and `--max-output-tokens` (required by the CLI)
  is not applied. With `--eval-thinking` the 2048 is not applied, because it was written for
  non-reasoning generation: every request carries the run's budget, `--max-output-tokens`,
  instead (`official_sampling` records `max_tokens` 2048 with `max_tokens_applied: false` and
  `max_tokens_sent`). `max_output_tokens_note` records which applies. simple-evals'
  reasoning-model entries (lines 145-212: `ResponsesSampler` with `reasoning_model=True`,
  `OChatCompletionSampler`), which send no system message, no temperature and no max_tokens, are
  not used: they are keyed on OpenAI model names and fit what OpenAI's hosted reasoning models
  accept, so they are not reasoning settings for a served model, which takes the chat sampler in
  both modes (see gpqa). `--temperature` or `GBENCH_HEALTHBENCH_TEMPERATURE` override the
  temperature in both modes; `official_sampling` records the setting, the temperature sent and
  `temperature_matches_official`, `sampler_max_tokens` the cap (2048, or null with
  `--eval-thinking`, where it is not applied) and `max_tokens_sent` the values sent.
  Reasoning-channel text is not graded.
  `--attempt-count` repeats every conversation (simple-evals' `n_repeats`, default 1).

## Headline metric and how to read it
- `accuracy` = the HealthBench overall score x100: clip(mean of per-conversation scores, 0, 1).
  `healthbench_mean_rubric_score` is the same value in [0, 1] and
  `healthbench_scored_conversations` its sample count.
- `healthbench_tag_scores`: the clipped mean per tag, e.g. `theme:emergency_referrals`,
  `axis:accuracy`, `axis:completeness`. The per-category table is keyed by theme and shows the
  theme score (base's 0.5 pass rate is kept as `pass_rate_at_0.5`).
- `pass_rate_at_0.5` (top level): share of conversations scoring at least 0.5, a gbench extra.
- Each trace keeps `healthbench_score` (unclipped) and `rubric_grades` (criterion, points, tags,
  `criteria_met`, the grader's explanation, the Gemini model that gave the verdict,
  `judge_model`, the attempt it took, `grader_attempts`, and how that grader reply ended,
  `judge_finish_reason`: `MAX_TOKENS` when it stopped at the 10240 limit and was parsed as it
  arrived); `judge_finish_reasons` lists every rubric item's, and a trace whose grading cascade
  saw replies stop at the cap with no text carries `judge_max_tokens_without_text`.
- `judge` is `gbench-gemini-cascade`, `judge_models` counts the answered grader calls per Gemini
  model (retries included) and `judge_cascade` lists the configured cascade order.
  `judge_replies_at_max_output_tokens` counts the grader replies that stopped at the 10240, on
  every attempt of a rubric item (a cut reply without the `criteria_met` boolean that a retry
  replaced counts too; each trace carries its own `judge_replies_at_max_output_tokens`), and `judge_outages_after_max_output_tokens` the judge-outage conversations whose
  cascade answers stopped there with no text.

`leaderboard_comparable` is always **false**: simple-evals grades with gpt-4.1-2025-04-14, gbench
with its Gemini judge cascade (at the same temperature 0.5; the models that answered are named in
`leaderboard_comparable_reason`). The reason also states how many requests got no reply from the
served endpoint and were left out of the mean (simple-evals grades a request its API rejects as
`No response (bad request).` and retries other errors until a response arrives), and names an
`--eval-limit` subset, a temperature other than simple-evals' 0.5 (gbench's 1.0 with thinking
or 0.0 without, or an operator-set value) and no-think requests whose prompt
left less than 2048 tokens of the context window (they got a smaller `max_tokens`), and the
replies cut off by a budget other than the no-think 2048 (the run's budget with
`--eval-thinking`, or a `max_tokens` the context window lowered: `replies_cut_off_by_budget`,
with `replies_cut_off_in_reasoning` for those that ran out inside the reasoning channel).

## Requirements
- **`GEMINI_API_KEY`** or **`GEMINI_API_KEYS`** (one key, or a comma/semicolon list;
  `GEMINI_API_KEYS` wins when both are set) for the grader. A missing or rejected key
  **hard-errors** (`infra_required`) before generation; there is no heuristic fallback.
- The `google-genai` client (in the `evals` extra).
- Network access to the Hugging Face Hub (or a warm HF cache) for the pinned revision.
- Grading cost: one grader call per rubric item, about 11.4 per conversation (57,237 for the
  full set).

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals healthbench --max-output-tokens 16384
```
Add `--eval-limit 20` for a quick (non-comparable) subset. `--max-output-tokens` is required by
the CLI; a no-think request does not apply it (it sends simple-evals' `max_tokens` 2048), and a
thinking request sends it in place of the 2048.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered conversations
(round-robin: positions I, I+N, I+2N, ...), selected before the `--eval-limit` draw, which is then
taken within the shard (N shards run up to N x `--eval-limit` conversations). The result records
`shard`, `shard_index` and `shard_count`, and a shard of more than one is not
leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`); `gbench
--merge` combines the shards' results as docs/distributed-gbench.md describes: it re-counts the
union of their traces, and refuses a suite whose headline is not a correct / scored count over
them unless the suite defines its own merge. HealthBench's merge (`merge_shards`) is that re-count,
with the grader cut-off counts (`judge_replies_at_max_output_tokens`,
`judge_outages_after_max_output_tokens`) summed over the shards and their reason named once, over
the sums, in `leaderboard_comparable_reason` (HealthBench's headline is a rubric score, not a correct / scored count, so the re-count refuses its shards).
