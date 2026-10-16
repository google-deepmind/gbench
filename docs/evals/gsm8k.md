# gsm8k

GSM8K (`openai/gsm8k`): 1,319 grade-school multi-step arithmetic word problems (the `main`
config, `test` split). **Scoring: lm-evaluation-harness `gsm8k_cot`, 8-shot chain of thought
with the fixed Wei et al. (2022) exemplars, greedy decoding, and exact match of the number after
"The answer is" (the `strict-match` filter).**

## Dataset
- `openai/gsm8k`, config `main`, split `test`, pinned to revision
  `740312add88f781978c0658806c59bc2815b9866` (the task's `dataset_path`). The reference is the
  text after `####` in the reference solution (`doc_to_target`).
- `--shard I/N` and `--eval-limit N` are applied by gbench's common runner, shard first: shard I
  of N of the full set, then a seeded random subset of N problems within it; a shard of more than
  one is not leaderboard-comparable (the shard reason is named in
  `leaderboard_comparable_reason`).

## Protocol
EleutherAI lm-evaluation-harness at commit `d6de81643928d653435c431bae19945d41d32520`, task
`gsm8k_cot` (`lm_eval/tasks/gsm8k/gsm8k-cot.yaml`, version 3.0), run the way lm-eval runs it
against a chat endpoint (`local-chat-completions` with `--apply_chat_template`, which turns
`fewshot_as_multiturn` on by default):
- **Messages**: the yaml's 8 fixed exemplars (`fewshot_config.samples`, sampler `first_n`) as
  alternating turns, user `Q: <question>\nA:` and assistant `<target>` (each target is the
  worked solution ending `The answer is N.`), then the test problem as the last user turn,
  `Q: <question>\nA:` (`doc_to_text`). No system message: the task has no description.
- `--eval-n-shot N` uses the first N exemplars (0 = the question alone). Only 8 exist, so N above
  8 stops the suite with an error before any request is sent.
- **Generation**: greedy. The task sets `do_sample: false`; lm-eval's chat backend then sends
  `temperature: 0` and `seed: 1234`, which gbench sends too. The stop sequences are the task's
  `until`: `Q:`, `</s>` and `<|im_end|>`. lm-eval's chat backend sends them to the server as
  the request's `stop` (its `parse_generations` calls `postprocess_generated_text` with
  `stop=None`), and a no-think request carries them the same way. A `--eval-thinking` request
  carries no stop strings, because a stop string also matches inside the reasoning channel;
  its reply is cut at the first of them after the reasoning is removed, the way lm-eval's HF
  backend cuts a generation (`models/huggingface.py` calls `postprocess_generated_text` with
  `until` and `think_end_token`).
- **Filters**, applied to the reply as lm-eval's `RegexFilter` and `take_first` do (no match
  gives `[invalid]`):
  - `strict-match`: `The answer is (\-?[0-9\.\,]+).`, the first match;
  - `flexible-extract`: `(-?[$0-9.,]{2,})|(-?[0-9]+)`, the last match (its first non-empty
    group).
- **Metric**: `exact_match` (`lm_eval/api/metrics.py`) with `ignore_case`, no punctuation
  stripping, and `regexes_to_ignore` `,`, `\$`, `(?s).*#### ` and `\.$` removed from both the
  extracted answer and the reference. So `The answer is 1,234.` matches `1234`.
- lm-eval's chat backend caps a reply at `max_gen_toks`, 256 tokens by default
  (`lm_eval/models/api_models.py`; the task sets none), and sends it as the request's
  `max_tokens`. That is the suite's official cap (`OfficialSampling.max_tokens`), applied by
  gbench's token-limit policy: a no-think request carries `max_tokens=256` exactly, whatever
  `--max-output-tokens` says (`max_output_tokens_note` records a flag that was not applied); a
  reply cut off at 256 tokens is scored on what arrived, as lm-eval scores it (counted as
  `replies_at_official_max_tokens`, not as a truncation problem). A `--eval-thinking` request
  carries the run's budget instead (`--max-output-tokens`, else the thinking floor), because the
  256 was written for non-reasoning generation and the reasoning shares the budget;
  `official_sampling` records `max_tokens: 256` and `max_tokens_applied: false`, and
  `max_output_tokens_note` says the official cap was not applied to reasoning. Its replies
  longer than 256 tokens, reasoning included, are counted and flagged (see below).

## Headline metric
`accuracy` is `exact_match` under the **strict-match** filter, in percent.
`flexible_extract_accuracy` is the same under `flexible-extract`; lm-eval's results table lists
both. `gsm8k_cot_report` holds both (`strict-match`, `flexible-extract`),
`max_gen_toks_sent` (256 without thinking, null with it) and `replies_over_max_gen_toks`. Each
trace records `strict_match`, `flexible_extract` and whether
each was correct. `n_shot` is the exemplar count applied.

Both filters are computed over the scored replies, as base's `accuracy` is: a request that got
no reply from the served endpoint (failed, timed out, or over the context window) is a harness
error, never the model's outcome. It is left out of both numbers, counted, and named in
`status_reason` (status `completed_with_errors`); when every request got none, both are null
and `accuracy_withheld_reason` says why. A reply that came back empty is the model's and scores
0.

`leaderboard_comparable` is true for a full 8-shot run at temperature 0 in which every request
was answered and no reply was longer than 256 tokens. Otherwise
`leaderboard_comparable_reason` names each cause: another `--eval-n-shot`, more than one sample
per problem (`--attempt-count`; the accuracy is then averaged over them), a temperature other
than 0 (every `--eval-thinking` run, source `thinking`, or an operator-set one), replies longer than lm-eval's 256-token `max_gen_toks` (which it would have cut
off; only a `--eval-thinking` run can have them, and it almost always does, since the reasoning
counts toward the reply), a no-think request sent a `max_tokens` other than 256 (only when the
prompt plus 256 does not fit the context window), requests that got no reply (harness errors,
left out of the score), empty replies (scored 0), or an `--eval-limit` subset.

## Sampling
gbench's rule for every suite (`base.resolve_temperature`): **temperature 0.0** without
`--eval-thinking` (lm-eval's 0), **1.0** with it, seed 1234 in both; the result records lm-eval's
values under `official_sampling`. A `--eval-thinking` run is therefore not comparable. `--temperature`, or for this suite alone
`GBENCH_GSM8K_TEMPERATURE` (which takes precedence), overrides the temperature in either mode, and
any value other than 0 makes the run non-comparable.

lm-eval's `openai-chat-completions` model, the class for OpenAI's hosted API, sends
`temperature: 1` and no `stop` when the model name contains `o1`, `o3`, `o4` or `5`
(`OpenAIChatCompletion._create_payload` at `d6de8164`; its constructor warns that "o1 models do
not support `stop` and only support temperature=1"). That branch adapts the request to what
OpenAI's hosted reasoning models accept. `local-chat-completions`, the backend lm-eval runs a
served model with, has no such branch and sends the same request whether the model reasons or
not, so it is not a reasoning-model setting for a served model. Apart from the thinking switch
itself, the temperature (above) and the token budget (see Protocol), a gbench thinking request differs from the request `local-chat-completions` sends in
one more way: the stop sequences are applied to the reply after the reasoning instead of being
sent (see Protocol), because a stop string would also match inside the reasoning channel.

## Requirements
- A running OpenAI-compatible `/v1` chat endpoint that honours the `stop`, `seed` and
  `max_tokens` parameters (vLLM does). No judge, no sandbox, no API key.
- The `datasets` package (base install) and network access to the Hugging Face Hub on the
  first run.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals gsm8k --max-output-tokens 65536
```
`--max-output-tokens` applies to a `--eval-thinking` run only. Add `--eval-thinking` to let the
model reason in its thinking channel (flagged, see above), or `--eval-limit 20` for a quick
check.
