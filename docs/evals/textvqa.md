# textvqa

TextVQA (`lmms-lab/textvqa`, validation split, pinned revision `9c0699cd`): visual question
answering that requires reading text in natural images (scene OCR). 5,000 questions, each
with 10 human answers. Images are sent losslessly (the original bytes).

## Protocol
- One request per question, as lmms-eval's `textvqa_val` task (EvolvingLMMs-Lab/lmms-eval
  `lmms_eval/tasks/textvqa/_default_template_textvqa_yaml` + `utils.py` @ `42296f8f`, the
  `default` model kwargs, no OCR tokens): the image, then the question passed through Python's
  `str.capitalize()` (first letter upper-case, the rest lower-case, exactly as lmms-eval does)
  followed by `\nAnswer the question using a single word or phrase.` (empty pre-prompt).
- Sampling (the token-limit policy): the task sets neither a temperature nor `max_new_tokens`,
  so lmms-eval's chat-endpoint client (`lmms_eval/models/chat/openai.py` @ `42296f8f`) sends
  its defaults, temperature 0 and `max_new_tokens` 1024. Without `--thinking` every request
  carries temperature 0 and **`max_tokens` 1024**, so `--max-output-tokens` is not applied
  (`max_output_tokens_note` says so), and a reply that stops at 1024 tokens is scored as it
  arrived, as lmms-eval scores it (`replies_at_official_max_tokens` counts them). With `--thinking`
  lmms-eval's sampling is the same, temperature 0, but gbench
  samples every thinking request at **temperature 1.0**, by its rule
  (1.0 with `--thinking`, 0.0 without)
  (`official_sampling` records `temperature_matches_official: false`). lmms-eval's client runs a
  thinking model
  with its `enable_thinking_kwarg`, which only adds `chat_template_kwargs.enable_thinking` and
  keeps the task's `generation_kwargs` (`build_payload_for_index`, lines 190-223). The client's
  other branch, for model names containing o1, o3, o4 or gpt-5 (temperature dropped,
  `max_completion_tokens`), matches only those OpenAI models and is not used. lmms-eval declares
  no reasoning-model cap, so the 1024, written for non-reasoning generation, is not applied to a
  `--thinking` run: each request is sent the run's `--max-output-tokens`, else the 32,768
  thinking floor, and the result records the cap as not applied to reasoning
  (`official_sampling.max_tokens_applied: false`, `max_output_tokens_note`). `--temperature` or
  `GBENCH_TEXTVQA_TEMPERATURE` override the temperature in both modes.
- The reply is scored as it comes back, exactly as `textvqa_process_results` receives
  `results[0]` (stripped of surrounding whitespace by lmms-eval's task layer), with
  no answer extraction.
- The reply and all 10 human answers go through the **EvalAI answer processor** used
  by the TextVQA evaluation server, vendored verbatim from facebookresearch/mmf
  `mmf/utils/m4c_evaluators.py` (@ `b0ffc7fb`; lmms-eval carries the same class): lower-case,
  the VQA punctuation list (removed or replaced by a space), a period dropped unless a digit
  follows it, number words to digits, articles dropped, and the contraction map. So `coca-cola`
  and `coca cola` match, and `10:30` keeps its colon.
- **Scoring: VQA accuracy.** Per question, the leave-one-out average over the 10 annotators of
  `min(#other annotators who gave the same processed answer / 3, 1)` (lmms-eval
  `textvqa_process_results`, mmf `TextVQAAccuracyEvaluator`), over every answer in the row.

## Reading the result
- `accuracy` is the mean VQA accuracy x100 over the questions (the published TextVQA metric).
  A request that got no reply from the served endpoint (failed, timed out, or over the context
  window) is a harness error: left out of the mean and of `binary_pass_rate`, counted and named
  in `status_reason`; when every request got none, `accuracy` is null with
  `accuracy_withheld_reason`.
- `binary_pass_rate` is the share of questions whose VQA accuracy is at least 0.5, and each
  scored trace records its `vqa_soft_score`.
- `leaderboard_comparable` is `true` for a full, unsharded run at the official temperature (0;
  a `--thinking` run samples at 1.0, so it is not comparable) in which every request got a reply and
  no reply was cut off by the
  run's token budget. Otherwise `leaderboard_comparable_reason` names each difference:
  `--eval-limit N subset of the 5,000 questions`, the shard of a `--shard I/N` run, a temperature
  other than 0 (set with `--temperature` or the variable above, or the 1.0 of a thinking run), the
  requests that got no reply (failed, timed
  out or over the context window; harness errors, left out of the score), and the replies cut off
  by the run's token budget (`replies_cut_off_by_budget`; `replies_cut_off_in_reasoning` counts
  those that ran out inside the reasoning channel with no answer).
- `official_sampling` records the settings (temperature 0 and `max_tokens` 1024, the same in
  both modes), the temperature the run used, where it came from, `temperature_matches_official`
  and `max_tokens_applied` (true without `--thinking`; with it false, plus `max_tokens_note` and
  `max_tokens_sent`).

## Requirements
- A **vision-language model** endpoint (OpenAI-compatible `/v1/chat/completions` that accepts
  `image_url` data URIs).
- `datasets` (downloads the pinned dataset revision from the Hugging Face Hub on first use).

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals textvqa \
       --max-output-tokens 65536 --eval-limit 20
```
Drop `--eval-limit` for the full 5,000-question run.
The run logs its planned workload at its start and records it as `eval_limit_plan`.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered questions
(the split's row order) (round-robin: positions I, I+N, I+2N, ...), selected before the
`--eval-limit` draw, which is then taken within the shard (N shards run up to N x `--eval-limit`
questions). The rows are selected by index before any image is read. The result records `shard`,
`shard_index` and `shard_count`, and a shard of more than one is not leaderboard-comparable (the
shard reason is named in `leaderboard_comparable_reason`); `gbench --merge` combines the shards'
results as docs/distributed-gbench.md describes: it re-counts the union of their traces, and
refuses a suite whose headline is not a correct / scored count over them unless the suite defines
its own merge.
