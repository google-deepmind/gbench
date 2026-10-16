# gpqa_diamond

GPQA Diamond (Rein et al., 2023): the 198 expert-curated, highest-quality questions of GPQA,
graduate-level multiple choice in biology, physics and chemistry with 4 options. It is
simple-evals' default GPQA variant. **Scoring: the simple-evals protocol; the question set is
asked 10 times, each time with a fresh option order, the first `Answer: X` line in the reply must
name the gold letter, and the headline is the mean score over all 1,980 copies.** For the
448-question main set, see [gpqa](gpqa.md).

## Dataset
- `Idavidrein/gpqa`, config `gpqa_diamond`, `train` split (198 questions), pinned revision
  `83022cefff930aea54f654c0b282e74b9eeda5c6`. The dataset is gated on Hugging Face.
- Each row holds the question, the correct answer and three incorrect answers.
- The pinned rows are identical, row for row and field for field, to the `gpqa_diamond.csv`
  simple-evals downloads (checked on all 198 rows), so every copy below gets exactly the option
  order it gets in a simple-evals run.

## Protocol
Identical to [gpqa](gpqa.md) (the suites share the code), ported from openai/simple-evals
(commit `652c89d0`, `simple_evals.py`, `gpqa_eval.py` and `common.py`):
- **Repeats and option order** (`GPQAEval`, `n_repeats=10` as `simple_evals.py` sets it): the
  question list is repeated 10 times (`examples * n_repeats`), and one `random.Random(0)` draws
  `rng.sample(range(4), 4)` for each copy in that order. The permutation reorders
  `[Correct Answer, Incorrect Answer 1, Incorrect Answer 2, Incorrect Answer 3]`, and the gold
  letter is where the correct answer lands. Every copy is its own request.
- Zero-shot chain of thought, one user message, `QUERY_TEMPLATE_MULTICHOICE`:

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
- Extraction: the first match of `ANSWER_PATTERN_MULTICHOICE`,
  `(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?`, compared with the gold letter exactly. There is no
  `\boxed{}` tier and no bare-letter or last-line fallback, so a reply without an
  `Answer: X` line scores 0, and so does `Answer: (B)`. The pattern's `(?i)` lets
  "Answer: because ..." match as a lower-case "b", which never equals a gold letter, so such a
  reply scores 0 and is never credited.
- `--attempt-count K` with K above 10 raises the repeat count to K (each extra repeat gets its
  own fresh permutation from the same generator); that is flagged as below. Values of 10 or less
  keep simple-evals' 10.

## Reading the result
- `accuracy`: the mean score over all copies (simple-evals' `score`), in percent, leaving out
  any copy whose request got no reply from the endpoint (a harness error, see below).
  `category_accuracy` breaks it down by `Subdomain` over the same copies.
- `repeats`: `n_repeats`, `questions`, `generations`, `per_repeat_accuracy` (one value per pass
  over the question set) and `score_std` (simple-evals' `score:std`, the standard deviation of
  the per-copy 0/1 scores). Each trace records its question (`source_sample_idx`), repeat
  (`attempt_idx`) and `permutation`.
- `leaderboard_comparable` is true for a full run with every request answered. It is false, with
  `leaderboard_comparable_reason` naming the cause, for an `--eval-limit` subset, for a repeat
  count other than 10, and when any request got no reply, since simple-evals' samplers ask a
  failed request again until it succeeds. A
  request that got no reply from the served endpoint (failed, timed out, or over the context
  window) is a harness error: left out of `accuracy`, `per_repeat_accuracy` and `score_std`,
  counted and named in `status_reason` (status `completed_with_errors`; when every request got
  none, `accuracy` is null with `accuracy_withheld_reason`).
- **Null replies are asked again, as simple-evals' sampler does.** `ChatCompletionSampler`
  raises "OpenAI API returned empty response; retrying" on a reply whose `message.content` is
  None and sends the same request again (no limit). gbench's requests go through a loopback
  relay that does the same (`swebench_common.upstream_reask_relay`, rule
  `reply_content_is_null`), up to `NULL_REPLY_REASKS` = 5 re-asks, so base scores the reply the
  sampler would return. A reply with its answer inside the thinking channel has null content, so
  it is asked again rather than scored wrong. An empty-string reply is a reply to the sampler and
  is scored as it is. Only a reply the model ended itself (`finish_reason` "stop") is asked again: a
  reply cut off at the token budget (`finish_reason` "length", e.g. a thinking reply that ran out of
  budget inside its reasoning) is the model's own outcome, never re-asked, scored as base scores a
  truncated reply and named (`upstream_reask.cut_off_not_reasked`). The re-asks stay inside base's
  per-request timeout (`EndpointRelay.reask_deadline`); a request whose re-asks run out, reach that
  deadline or get no reply from the endpoint is scored on the last reply it got (`reasks_exhausted`,
  of them `reasks_cut_short`), never a harness error. A reply still null after the 5 re-asks is
  scored as base scores it (an
  answer-in-thinking one is counted in `answer_in_thinking`) and named in
  `leaderboard_comparable_reason`. The result's `upstream_reask` records the rule, `reask_limit`,
  `requests_reasked`, `reasks_sent`, `reasks_exhausted`, `reasks_cut_short`, `cut_off_not_reasked`,
  the replies the relay saw and their empty-reply classes (`answer_in_thinking_seen`,
  `answer_in_thinking_rate_seen`, ...); see
  [answer_in_thinking.md](answer_in_thinking.md). A temperature that differs from
  simple-evals' sampler's 0.5 (gbench's 1.0 with thinking or 0.0 without, or an
  operator-set value; see below) is flagged
  too, and so is a no-think request sent a max_tokens other than 2048 (the
  context clamp sends less only when the prompt plus 2048 does not fit the context window) and a
  reply cut off before it finished at anything but the 2048 (a thinking reply at the run's
  budget or at a max_tokens the context clamp lowered, or a no-think reply at such a clamped
  max_tokens), scored on what arrived.
- `--eval-limit N` keeps N questions stratified by `Subdomain`, with all 10 of their repeats and
  the same option orders a full run gives them. That is not a full-benchmark number.
  The run logs its planned workload at its start and records it as `eval_limit_plan`.
- `--shard I/N` selects shard I of the questions (round-robin in dataset order) before the
  `--eval-limit` draw, so every copy of a question runs in the same shard; the result records
  `shard`, and a shard of more than one is not leaderboard-comparable (the shard reason is named
  in `leaderboard_comparable_reason`). `gbench --merge` re-counts the union of the shards' copies
  and recomputes `repeats` over it.

## Requirements
- An OpenAI-compatible `/v1` chat endpoint serving the model.
- `Idavidrein/gpqa` is **gated**: accept its terms on the dataset page with your Hugging Face
  account and set `HF_TOKEN` so `datasets` can download it (about 5 MB). No API keys beyond
  that, no Docker images or checkouts.
- A full run is 1,980 generations.

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
  alone `GBENCH_GPQA_DIAMOND_TEMPERATURE` (which takes precedence), overrides the temperature in
  both modes; a value other than 0.5 makes the run non-comparable.

## Run
```bash
HF_TOKEN=<your token> gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals gpqa_diamond --eval-thinking \
       --max-output-tokens 65536
```
