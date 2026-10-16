# chartqa

ChartQA (`ahmed-masry/ChartQA`, test split, pinned revision `af8b6f5c`): question answering
over real charts, 2,500 questions. Half (1,250) are written by people ("human"), half are
generated from chart summaries ("augmented"). Images are sent losslessly (the original PNG
bytes). `ahmed-masry/ChartQA` is the ChartQA authors' copy; lmms-eval's `chartqa.yaml` loads
`lmms-lab-encoder/ChartQA` instead. The two hold the same 2,500 questions and answers in the same
order, and all 2,500 images are pixel-identical (checked row by row), so the scores are the same.

## Protocol
- One request per question, as lmms-eval's `chartqa` task (EvolvingLMMs-Lab/lmms-eval
  `lmms_eval/tasks/chartqa/chartqa.yaml` + `utils.py` @ `42296f8f`, the `default` model
  kwargs): the chart image, then the question followed by
  `\nAnswer the question with a single word.` (empty pre-prompt).
- Sampling (the token-limit policy): the task's `generation_kwargs` (temperature 0,
  `max_new_tokens: 16`, greedy) as lmms-eval's chat-endpoint client
  (`lmms_eval/models/chat/openai.py` @ `42296f8f`) sends them. Without `--thinking` every request
  carries temperature 0 and **`max_tokens` 16**, so `--max-output-tokens` is not applied
  (`max_output_tokens_note` says so), and a reply that stops at 16 tokens is scored as it arrived,
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
  no reasoning-model cap, so the 16, written for non-reasoning generation, is not applied to a
  `--thinking` run: each request is sent the run's `--max-output-tokens`, else the 32,768 thinking
  floor, and the result records the cap as not applied to reasoning
  (`official_sampling.max_tokens_applied: false`, `max_output_tokens_note`). `--temperature` or
  `GBENCH_CHARTQA_TEMPERATURE` override the temperature in both modes.
- The reply is scored as it comes back, exactly as `chartqa_process_results` passes
  `results[0]` to the scorer (lmms-eval's task layer only strips surrounding whitespace): there
  is no answer extraction and no markdown removal, so a reply of `**14**` or `Yes.` does not
  match `14` or `Yes`.
- **Scoring: ChartQA relaxed accuracy**, exactly as `relaxed_correctness` in
  google-research/pix2struct `pix2struct/metrics.py` (@ `6fe25c1d`, the implementation
  lmms-eval also uses):
  - the WHOLE answer is parsed as a number, with a trailing `%` dividing it by 100;
  - if both the answer and the gold parse and the gold is non-zero, the answer is correct when
    it is within 5% of the gold;
  - otherwise the two strings must match case-insensitively (so `Q3 2019` is not the number 3,
    `1234` does not match `1,234`, and a gold of `0` needs the string `0`).

## Reading the result
- `accuracy` is relaxed accuracy over all 2,500 questions, which equals the mean of the two
  halves (they are the same size). `category_accuracy` holds the halves as `human` and
  `augmented` (ChartQA-H and ChartQA-A in the paper). A request that got no reply from the
  served endpoint (failed, timed out, or over the context window) is a harness error: left out
  of both, counted and named in `status_reason`; when every request got none, `accuracy` is null
  with `accuracy_withheld_reason`.
- `leaderboard_comparable` is `true` for a full, unsharded run at the official temperature (0;
  a `--thinking` run samples at 1.0, so it is not comparable) in which every request got a reply and
  no reply was cut off by the
  run's token budget. Otherwise `leaderboard_comparable_reason` names each difference:
  `--eval-limit N subset of the 2,500 questions`, the shard of a `--shard I/N` run, a temperature
  other than 0 (set with `--temperature` or the variable above, or the 1.0 of a thinking run), the
  requests that got no reply (failed, timed
  out or over the context window; harness errors, left out of the score), and the replies cut off
  by the run's token budget (`replies_cut_off_by_budget`; `replies_cut_off_in_reasoning` counts
  those that ran out inside the reasoning channel with no answer).
- `official_sampling` records the settings (temperature 0 and `max_tokens` 16, the same in both
  modes), the temperature the run used, where it came from, `temperature_matches_official` and
  `max_tokens_applied` (true without `--thinking`; with it false, plus `max_tokens_note` and
  `max_tokens_sent`).
- `--eval-limit N` draws N questions stratified over the two halves.
  The run logs its planned workload at its start and records it as `eval_limit_plan`.

## Requirements
- A **vision-language model** endpoint (OpenAI-compatible `/v1/chat/completions` that accepts
  `image_url` data URIs).
- `datasets` (downloads the pinned dataset revision from the Hugging Face Hub on first use).

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals chartqa \
       --max-output-tokens 65536 --eval-limit 20
```
Drop `--eval-limit` for the full 2,500-question run.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered questions
(the test split's row order) (round-robin: positions I, I+N, I+2N, ...), selected before the
`--eval-limit` draw, which is then taken within the shard (N shards run up to N x `--eval-limit`
questions). The rows are selected by index before any chart image is read. The result records
`shard`, `shard_index` and `shard_count`, and a shard of more than one is not
leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`); `gbench
--merge` combines the shards' results as docs/distributed-gbench.md describes: it re-counts the
union of their traces, and refuses a suite whose headline is not a correct / scored count over
them unless the suite defines its own merge.
