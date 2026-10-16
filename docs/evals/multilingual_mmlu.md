# multilingual_mmlu

Multilingual MMLU (MMMLU, OpenAI): the MMLU test set translated into 14 languages by
professional human translators. **Scoring: the simple-evals protocol; the reply must state its
answer as a localized "Answer:" anchor followed by the letter, which must equal the gold letter,
and the headline is accuracy over all languages.** `category_accuracy` breaks it down by
language.

## Dataset
- `openai/MMMLU`, pinned revision `325a01dc3e173cac1578df94120499aaca2e2504`, `test` split of
  each of the 14 language configs: `AR_XY`, `BN_BD`, `DE_DE`, `ES_LA`, `FR_FR`, `HI_IN`,
  `ID_ID`, `IT_IT`, `JA_JP`, `KO_KR`, `PT_BR`, `SW_KE`, `YO_NG`, `ZH_CN`. Each holds the full
  14,042-question test set, so a full run is 196,588 questions. The `default` config is not a
  language split and is not loaded.
- Columns: `Question`, `A` to `D`, `Answer` (a letter), `Subject`. The options keep their Latin
  labels A-D in every language.

## Protocol
MMMLU was released with openai/simple-evals as its harness; prompt, extraction and scoring are
ported verbatim from simple-evals (commit `652c89d0`, `common.py` and `mmlu_eval.py`):
- Zero-shot, one user message, the English `QUERY_TEMPLATE_MULTICHOICE` around the translated
  question and options:

  ```
  Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

  {Question}

  A) {A}
  B) {B}
  C) {C}
  D) {D}
  ```
- The user message is the same with and without `--eval-thinking`, and so is simple-evals'
  chat-sampler system message before it (see Request and sampling below).
- Extraction: `normalize_response` strips markdown bold and LaTeX wrappers; then the
  `MULTILINGUAL_ANSWER_REGEXES` anchors ("Answer:", "Respuesta:", "答案：", "الإجابة:", and so on
  for every language) are tried in order, each followed by optional spaces and the letter. The
  first anchor that matches decides, at its first occurrence. `normalize_extracted_answer` maps
  the Arabic, Bengali and full-width forms of A-D to Latin letters. The result is compared with
  the gold letter exactly, so a lower-case letter never scores.
- A reply that does not state its answer in that form scores 0, for example
  `Answer: (B)` (the parenthesis is not read) or "La respuesta correcta es B" (no anchor and
  colon). The words "correcta", "correta", "corretta" and "correcte" are never read as option C.

## Reading the result
- `accuracy`: percent of questions answered correctly, pooled over all languages. Every
  language has the same number of questions, so this equals the mean of the per-language
  accuracies.
- `category_accuracy`: per language, keyed by the lower-case language code (`ar`, `bn`, `de`,
  `es`, `fr`, `hi`, `id`, `it`, `ja`, `ko`, `pt`, `sw`, `yo`, `zh`).
- `leaderboard_comparable` is true for a full run with every request answered, since the dataset,
  prompt, extraction and metric are simple-evals'. It is false, with
  `leaderboard_comparable_reason` naming the cause, for an `--eval-limit` subset and when any
  request got no reply or came back empty, since simple-evals' samplers ask a failed request,
  and one whose content is null, again until it succeeds. A request that got no reply from the
  served endpoint (failed, timed out, or over the context window) is a harness error: left out
  of `accuracy` and `category_accuracy`, counted and named in `status_reason` (status
  `completed_with_errors`; when every request got none, `accuracy` is null with
  `accuracy_withheld_reason`). A reply that came back empty is scored wrong (gbench cannot tell
  a null reply from an empty one). A
  temperature that differs from simple-evals' sampler's 0.5 (gbench's 1.0 with thinking or 0.0 without, or an
  operator-set value; see below) is flagged too, and
  so is a no-think request sent a max_tokens other than 2048 (only when the prompt plus 2048 does
  not fit the context window) and a reply cut off before it finished at anything but the 2048
  (a thinking reply at the run's budget or at a max_tokens the context clamp lowered, or a
  no-think reply at such a clamped max_tokens), scored on what arrived.
- `--eval-limit N` scores a subset stratified by language. That is not a full-benchmark number.
  The run logs its planned workload at its start and records it as `eval_limit_plan`.
- `--shard I/N` selects shard I of the rows (round-robin over the language-interleaved order)
  before the `--eval-limit` draw; the result records `shard`, and a shard of more than one is not
  leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`).

## Requirements
- An OpenAI-compatible `/v1` chat endpoint serving the model.
- Hugging Face Hub access to the public `openai/MMMLU` dataset. It downloads on first use and is
  cached by `datasets`. No API keys, Docker images or checkouts.

## Request and sampling
simple-evals' chat sampler decides what is sent around the prompt, with and without
`--eval-thinking` (`simple_evals.py` at `652c89d0`):
- The request is what `simple_evals.py`'s `gpt-4o` entry (lines 236-240) sends: a
  `ChatCompletionSampler` (`sampler/chat_completion_sampler.py`) with
  `system_message=OPENAI_SYSTEM_MESSAGE_API` and `max_tokens=2048`. So the request is the system
  message `You are a helpful assistant.` followed by the user message, in both modes.
  **The temperature is gbench's rule, not the sampler's default 0.5: 1.0 with `--eval-thinking`,
  0.0 without** (`base.resolve_temperature`). `official_sampling` records the sampler's 0.5 with
  `temperature_matches_official: false`, and the temperature is named in
  `leaderboard_comparable_reason`.
- **Without thinking** it is sent `max_tokens=2048`. The 2048 is the suite's official cap
  (`OfficialSampling.max_tokens`), which gbench's token-limit policy sends exactly on a no-think
  request, whatever `--max-output-tokens` says (`max_output_tokens_note` records a flag that was
  not applied; the cap is recorded as `sampler_max_tokens` and under `official_sampling`). A
  reply that stops at the 2048 sent, still writing or repeating itself, is scored on what
  arrived, as the sampler returns it; `replies_at_sampler_max_tokens` counts replies that stopped
  at their max_tokens (base's `replies_at_official_max_tokens` counts those at the 2048 itself).
  Those at the 2048 are the sampler's outcome: they do not make the run non-comparable and do not
  count toward the share of truncated replies that makes a run `completed_with_errors`.
- **With `--eval-thinking`** the 2048 is not applied, because it was written for non-reasoning
  generation: each request is sent the run's budget, `--max-output-tokens`, instead
  (`official_sampling` records `max_tokens` 2048 with `max_tokens_applied: false` and
  `max_tokens_sent`; `max_output_tokens_note` says so). `sampler_max_tokens` is null. A reply cut
  off at that budget, or at a max_tokens the context clamp lowered, is scored on what arrived and
  makes the run non-comparable.
- simple-evals' reasoning-model entries are not used. `simple_evals.py` (lines 145-212) sends
  OpenAI's o-series models through other samplers: o3, o4-mini and o1-pro through
  `ResponsesSampler` with `reasoning_model=True` (`sampler/responses_sampler.py`), and o1,
  o1-preview, o1-mini and o3-mini through `OChatCompletionSampler`
  (`sampler/o_chat_completion_sampler.py`); both send no system message, no temperature and no
  max_tokens. Those entries are keyed on OpenAI model names and fit what OpenAI's hosted reasoning
  models accept, so they are not reasoning settings for a served model: a served model takes the
  chat sampler in both modes. HELM's and lm-eval's o-series branches are treated the same way:
  they adapt the request to what OpenAI's hosted models accept, are never taken for a served
  model, and are not applied (see mmlu_redux and gsm8k).
- The result records the sampling under `official_sampling`. `--temperature`, or for this suite
  alone `GBENCH_MULTILINGUAL_MMLU_TEMPERATURE` (which takes precedence), overrides the temperature
  in
  both modes; a value other than 0.5 makes the run non-comparable.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals multilingual_mmlu --max-output-tokens 65536
```
Add `--eval-thinking` to enable the reasoning channel.
