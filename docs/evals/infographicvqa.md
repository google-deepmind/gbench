# infographicvqa

InfographicVQA (`lmms-lab-encoder/DocVQA`, config `InfographicVQA`, validation split, pinned
revision `539088ef`): question answering over infographics (dense text, charts and layout).
2,801 questions, the split whose answers are public (the test split's answers are held by the
evaluation server). Images are sent losslessly (the original bytes).

## Protocol
- One request per question, as lmms-eval's `infovqa_val` task (EvolvingLMMs-Lab/lmms-eval
  `lmms_eval/tasks/infovqa/_default_template_infovqa_yaml`, `infovqa_val.yaml` and `utils.py` @
  `42296f8f`, the `default` model kwargs): the infographic, then the question followed by
  `\nAnswer the question using a single word or phrase.` (empty pre-prompt).
- Sampling (the token-limit policy): the task's `generation_kwargs` (temperature 0,
  `max_new_tokens: 32`, greedy) as lmms-eval's chat-endpoint client
  (`lmms_eval/models/chat/openai.py` @ `42296f8f`) sends them. Without `--thinking` every request
  carries temperature 0 and **`max_tokens` 32**, so `--max-output-tokens` is not applied
  (`max_output_tokens_note` says so), and a reply that stops at 32 tokens is scored as it arrived,
  as lmms-eval scores it (`replies_at_official_max_tokens` counts them). With `--thinking`
  lmms-eval's sampling is the same, temperature 0, but gbench
  samples every thinking request at **temperature 1.0**, by its rule
  (1.0 with `--thinking`, 0.0 without)
  (`official_sampling` records `temperature_matches_official: false`). lmms-eval's client runs a
  thinking model with its
  `enable_thinking_kwarg`, which only adds `chat_template_kwargs.enable_thinking` and keeps the
  task's `generation_kwargs` (`build_payload_for_index`, lines 190-223). The client's other
  branch, for model names containing o1, o3, o4 or gpt-5 (temperature dropped,
  `max_completion_tokens`), matches only those OpenAI models and is not used. lmms-eval declares
  no reasoning-model cap, so the 32, written for non-reasoning generation, is not applied to a
  `--thinking` run: each request is sent the run's `--max-output-tokens`, else the 32,768 thinking
  floor, and the result records the cap as not applied to reasoning
  (`official_sampling.max_tokens_applied: false`, `max_output_tokens_note`). `--temperature` or
  `GBENCH_INFOGRAPHICVQA_TEMPERATURE` override the temperature in both modes.
- The reply is scored as it comes back (`results[0]`, stripped of surrounding whitespace by
  lmms-eval's task layer), with no answer extraction.
- **Scoring: ANLS**, lmms-eval's `anls` metric (`lmms_eval/api/metrics.py`, the task's
  `metric_list`): per question, `1 - edit_distance / length` against each gold answer, where the
  edit distance is taken between the lower-cased, whitespace-collapsed strings and `length` is
  the longer of the gold answer as stored and the stripped reply (as upstream), the best gold
  counts, and a value below 0.5 is set to 0.
- A question whose image cannot be decoded stops the run (it is never dropped from the
  denominator), and so does a dataset that cannot be loaded at the pinned revision.

## Reading the result
- `accuracy` is mean ANLS x100 over all questions (the InfographicVQA metric); an empty reply
  scores 0. A request that got no reply from the served endpoint (failed, timed out, or over the
  context window) is a harness error: left out of the mean and of `pass_rate_at_0.5`, counted and
  named in `status_reason`; when every request got none, `accuracy` is null with
  `accuracy_withheld_reason`. `pass_rate_at_0.5` is the share of questions with a non-zero ANLS,
  and each scored trace records its `anls`. `category_accuracy` groups by the question's first
  answer type (`single span`, `multi-span`, `non-extractive`, `question span`).
- `dataset` names the pinned revision.
- `leaderboard_comparable` is `true` for a full, unsharded run at the official temperature (0;
  a `--thinking` run samples at 1.0, so it is not comparable) in which every request got a reply and
  no reply was cut off by the
  run's token budget. Otherwise `leaderboard_comparable_reason` names each difference:
  `--eval-limit N subset of the 2,801 questions`, the shard of a `--shard I/N` run, a temperature
  other than 0 (set with `--temperature` or the variable above, or the 1.0 of a thinking run), the
  requests that got no reply (failed, timed
  out or over the context window; harness errors, left out of the score), and the replies cut off
  by the run's token budget (`replies_cut_off_by_budget`; `replies_cut_off_in_reasoning` counts
  those that ran out inside the reasoning channel with no answer).
- `official_sampling` records the settings (temperature 0 and `max_tokens` 32, the same in both
  modes), the temperature the run used, where it came from, `temperature_matches_official` and
  `max_tokens_applied` (true without `--thinking`; with it false, plus `max_tokens_note` and
  `max_tokens_sent`).
- `--eval-limit N` draws a seeded random subset of N questions.
  The run logs its planned workload at its start and records it as `eval_limit_plan`.

## Requirements
- A **vision-language model** endpoint (OpenAI-compatible `/v1/chat/completions` that accepts
  `image_url` data URIs).
- `datasets` (downloads the pinned revision of the `InfographicVQA` config from the Hugging Face
  Hub on first use: its validation and test parquet files, about 570 MB).
- No API keys.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals infographicvqa \
       --max-output-tokens 65536 --eval-limit 20
```
Drop `--eval-limit` for the full 2,801-question run.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered questions
(the split's row order) (round-robin: positions I, I+N, I+2N, ...), selected before the
`--eval-limit` draw, which is then taken within the shard (N shards run up to N x `--eval-limit`
questions). The rows are selected by index before any infographic is read. The result records
`shard`, `shard_index` and `shard_count`, and a shard of more than one is not
leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`); `gbench
--merge` combines the shards' results as docs/distributed-gbench.md describes: it re-counts the
union of their traces, and refuses a suite whose headline is not a correct / scored count over
them unless the suite defines its own merge.
