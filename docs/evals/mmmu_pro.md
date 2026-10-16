# mmmu_pro

MMMU-Pro (`MMMU/MMMU_Pro`, test split, pinned revision `563f3e84`): multimodal,
multi-discipline multiple choice, 1,730 questions, in the two settings the MMMU-Pro score is
built from:

- **standard (10 options)**: the question text with its images, up to 10 options (one
  question has 12).
- **vision**: a screenshot or photo that contains the whole question and its options; no text
  besides the prompt.

## Protocol
Ported from the official MMMU-Pro code (MMMU-Benchmark/MMMU `mmmu-pro/` @ `268471d0`:
`infer/infer_gpt.py`, `prompts.yaml`, `evaluate.py`):

- **Prompts**: `prompts.yaml` verbatim, both modes. CoT standard: "Answer the preceding
  multiple choice question. The last line of your response should be of the following format:
  'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before
  answering." Direct standard: "Answer with the option letter from the given choices
  directly." The vision prompts are the `vision` entries of the same file.
- **System message** (`infer_gpt.py` `request_with_images`, verbatim): every request starts with
  the system turn `You are ChatGPT, a large language model trained by OpenAI, based on the
  GPT-4 architecture.`, then one user turn holding the text and then the images.
- **standard (10 options)**: one user text part, `{question}\nA. {option}\nB. {option}...\n{prompt}`.
  Every `<image n>` token becomes `<image>`, and the referenced images follow the text in the
  order they are referenced (upstream `replace_images_tokens` + `origin_mmmu_doc_to_visual`,
  so an image referenced twice is sent twice).
- **vision**: the mode's vision prompt, then the screenshot.
- Images are sent as their original bytes (upstream re-encodes the decoded image as PNG; the
  pixels are the same).
- **Parsing**: upstream `parse_multi_choice_response`: a unique option letter after the last
  `Answer:`, else `(X)` / `X ` / `X.` candidates (the last one wins), else an option's text
  quoted in a reply of more than 5 words, else a random letter. The random letter is drawn
  from an RNG seeded with the question id, so a rerun scores the same.
- **Sampling**: `infer_gpt.py` sends no temperature and no top-p. gbench sends no top-p either,
  in both modes, but always sends the temperature of its rule: **1.0 with `--eval-thinking`, 0.0
  without** (`temperature_matches_official: false`, named in `leaderboard_comparable_reason`). `--temperature` or `GBENCH_MMMU_PRO_TEMPERATURE` override
  it in both modes; `official_sampling` records the setting, the
  temperature sent and `temperature_matches_official`.
- **Token limit** (the token-limit policy): without `--eval-thinking` every request carries
  `infer_gpt.py`'s **`max_tokens` 4096**, so `--max-output-tokens` is not applied
  (`max_output_tokens_note` says so), and a reply that stops at 4096 tokens is parsed as it
  arrived, as the script parses it (`replies_at_official_max_tokens` counts them). The script
  defines no reasoning-model settings, so with `--eval-thinking` that cap, written for
  non-reasoning generation, is not applied: each request is sent the run's
  `--max-output-tokens`, and `official_sampling` records `max_tokens: 4096`,
  `max_tokens_applied: false` and the budget sent.

## Reading the result
- For each setting the suite takes the higher of the CoT and Direct accuracies, as the paper
  does for its overall results. `accuracy` is the MMMU-Pro score: the mean of the
  standard-10 and vision numbers.
- `mmmu_pro_standard_10` and `mmmu_pro_vision` are the two settings; `mmmu_pro_by_mode` has
  every setting x mode accuracy; `pooled_accuracy` is the plain share of correct requests (its
  counts are `pooled_total_questions` / `pooled_correct_answers`; the summary row's
  `correct_answers` is null, with `correct_answers_note`, since the headline is a mean of two
  best-of-mode rates, not a count; `total_questions` is the number of distinct questions asked
  and `effective_n` the number with a scored reply);
  `category_accuracy` is keyed `standard_10/cot`, `vision/direct`, and so on.
- `leaderboard_comparable` would be true for a full, unsharded run with both prompt modes, no
  temperature sent, a reply for every request and no reply cut off by the run's token budget (a
  reply that stops at the 4,096 cap without `--eval-thinking` is the protocol's outcome); gbench
  always sends a temperature (its rule), so the reason always names it. It is
  false, with the reason, for an `--eval-limit` subset, a `GBENCH_SHARD` shard, a single prompt
  mode, the temperature sent (gbench's 1.0 with thinking or 0.0 without, or a value set with
  `--temperature` or the variable above),
  requests that got no
  reply, or replies cut off by the run's token budget rather than at an official cap
  (`replies_cut_off_by_budget`; `replies_cut_off_in_reasoning` counts those that ran out inside
  the reasoning channel with no answer).
- A request that gets no reply from the served endpoint (failed, timed out, over the context
  window) is a harness error, never the model's answer: it is left out of every number above
  (the settings, the modes, `pooled_accuracy`, `category_accuracy`), counted and named in
  `status_reason` and in the reason. When a whole setting has no scored request, `accuracy` is
  null with `accuracy_withheld_reason` (MMMU-Pro needs both settings), which names the harness
  errors behind it.
- A `GBENCH_SHARD` shard splits the settings x modes fan-out, so a shard may hold no request of a
  setting at all. Its `accuracy` is then null too, with an `accuracy_withheld_reason` that says so
  (the shard holds only the other setting; MMMU-Pro needs the merged shards) rather than naming
  harness errors; the status stays `success` when every request was answered, and
  `mmmu_pro_standard_10` / `mmmu_pro_vision` keep the shard's per-setting scores.

## Options
- `GBENCH_MMMU_PRO_PROMPT`: `both` (default), `cot` or `direct`. A single mode halves the number
  of requests, and the result is then marked not leaderboard-comparable.
- `--eval-limit N` runs N requests in total (a question in one setting and prompt mode is one
  sample), dealt over the setting x mode units (standard_10/cot, standard_10/direct, vision/cot,
  vision/direct by default); each unit draws its share of questions stratified by subject with a
  fixed seed, so a smaller share's questions are among a larger share's. `--eval-limit 5` sends 5
  requests, not 20. A unit dealt no sample is not run, and MMMU-Pro (the mean of the two
  settings) is withheld when a setting got none. The run logs the plan at its start and records
  it as `eval_limit_plan`.
- `--eval-max-soft-tokens` (`70`/`140`/`280`/`560`/`1120`) sets the image soft-token budget.
  gbench sets it on the server it launches, so it cannot be combined with `--remote-endpoint`;
  on a remote server configure the budget server-side.

## Requirements
- A **vision-language model** endpoint (OpenAI-compatible `/v1/chat/completions` that accepts
  several `image_url` data URIs per message).
- `datasets` (downloads both pinned configs, about 2.3 GB, from the Hugging Face Hub on first
  use).

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals mmmu_pro \
       --max-output-tokens 65536 --eval-limit 20
```
With a server that gbench launches, the soft-token budget can be set too:
```bash
gbench --evals-only --models google/gemma-4-E4B-it --num-gpus 1 --evals mmmu_pro \
       --eval-max-soft-tokens 280 --max-output-tokens 65536 --eval-limit 20
```
Drop `--eval-limit` for the full run; add `--eval-thinking` for reasoning mode.
