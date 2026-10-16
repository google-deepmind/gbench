# mmlu_redux

MMLU-Redux-2.0 (Gema et al., NAACL 2025, [arXiv:2406.04127](https://arxiv.org/abs/2406.04127)):
100 manually re-annotated MMLU test questions for each of the 57 subjects, every row labelled
by error type (a correct row, a wrong ground truth, an unclear question, and so on).
**Scoring: the way the MMLU-Redux authors score models; HELM's 5-shot MMLU protocol with a
one-letter answer, exact match on the rows annotated as correct (`error_type == "ok"`), and
the headline is exact match over all those rows combined.**

## Dataset
- `edinburgh-dawg/mmlu-redux-2.0`, pinned revision
  `372ea425445d51e1ba1188c56e5e893f8138621f`: 57 subject configs, `test` split, 5,700 rows. The
  v1 dataset (`edinburgh-dawg/mmlu-redux`, 30 subjects) is superseded; its card points to 2.0.
- Scored rows: `error_type == "ok"`, keyed by the original MMLU answer. At the pinned revision
  that is **5,330 rows**; the other 370 (`wrong_groundtruth`, `bad_question_clarity`,
  `multiple_correct_answers`, `no_correct_answer`, `expert`, `bad_options_clarity`) are the
  paper's erroneous instances and are not scored. The filter runs before any `--eval-limit`
  subset is drawn, and the loader logs the counts.
- In-context examples: the MMLU dev split, `cais/mmlu` config `all`, pinned revision
  `c30699e8356da336a370243923dbaf21066bb9fe` (5 questions per subject; HELM's MMLU train split).

## Protocol
The MMLU-Redux authors ship no answering harness (github.com/aryopg/mmlu-redux covers error
detection only). Their re-evaluation of models (paper Sec. 3.2, Table 2, Fig. 5 and App. B; the
dataset card's "Dataset Sources") scores the predictions of the **HELM MMLU leaderboard** by
exact match on the "correct instances". gbench therefore runs HELM's MMLU protocol
(stanford-crfm/helm `get_mmlu_spec`, the same at release v0.5.1 `ccf1df596b` and at `63754d05`)
on those rows, as HELM runs an instruction-tuned model:
- **Prompt** (`multiple_choice_joint`, one user message, no system prompt):

  ```
  Answer with only a single letter.

  The following are multiple choice questions (with answers) about <subject, "_" as spaces>.

  Question: <dev question 1>
  A. ...
  B. ...
  C. ...
  D. ...
  Answer: <letter>

  ... (5 examples)

  Question: <question>
  A. ...
  B. ...
  C. ...
  D. ...
  Answer:
  ```

  The first line is HELM's `output_format_instructions=mmlu` expander, which the HELM MMLU
  runs of instruction-tuned models carry (their `run_spec.json` from leaderboard v1.3.0 on:
  GPT-4o, Claude 3.5 Sonnet, Llama 3.1 Instruct, Qwen2.5 Instruct, DeepSeek-V3, Gemini); base
  models run without it. The five examples are the subject's MMLU dev questions in the order
  HELM's `InContextLearningAdapter.sample_examples` gives them with seed 0 (ported). Fields are
  read as HELM reads the MMLU CSV files, so `\r\n` inside a field becomes `\n`.
- **Request**: `max_tokens=1`, `temperature=0`, no stop sequences, thinking off
  (`MultipleChoiceJointAdapter.generate_requests`). The `max_tokens=1` is the suite's official
  cap (`OfficialSampling.max_tokens`), applied by gbench's token-limit policy; see the thinking
  item below.
- **Scoring** (`evaluate_reference_metrics`, exact match after the output mapping): the reply,
  stripped, must be a single letter A-D whose option text equals the correct option's text.
  Anything else (`**B`, `The`, an empty reply) scores 0 and is counted in `non_letter_replies`
  (with thinking, a reply cut off at the budget or empty after the reasoning is counted apart;
  see below).
- **Checked against HELM itself**: over all 14,042 MMLU test questions, gbench's prompts are
  byte-identical to the requests in HELM's public v1.3.0 GPT-4o run (`display_requests.json`),
  and gbench's scorer reproduces HELM's recorded `exact_match` on all 14,042 of that run's
  predictions. For the 5,330 scored Redux rows, 5,168 prompts are byte-identical to a HELM
  request; the other 162 differ only because the Redux 2.0 text differs from `cais/mmlu`
  (trailing spaces removed, non-breaking spaces turned into spaces, a few characters lost in
  encoding, and one `high_school_microeconomics` row whose question field holds a URL). gbench
  uses the Redux text as published.
- Without `--eval-thinking`, `--max-output-tokens` does not change the request (it is recorded
  as not applied in `max_output_tokens_note`), and a reply that stops at HELM's one token is
  counted in `replies_at_official_max_tokens`: that is the protocol, not a truncation. With
  `--eval-thinking` the reasoning channel is on (`thinking`, `thinking_requested` and
  `thinking_applied` are true): HELM's one-token cap was written for non-reasoning generation,
  so each request is sent the run's budget instead (`--max-output-tokens`, else the thinking
  floor), fitted to the context window (a prompt that cannot fit is recorded as
  `OVER_CONTEXT`); `official_sampling` records `max_tokens: 1` and `max_tokens_applied: false`,
  and `max_output_tokens_note` says the official cap was not applied to reasoning. The reply
  read is the message content (the server returns the reasoning separately, kept in each
  trace's `reasoning`; an inline thought block is removed), scored with the same exact match.
  HELM's MMLU runs read a one-token answer without reasoning, so a thinking run is flagged
  non-comparable. `--attempt-count` is recorded as not applied (HELM scores one greedy
  completion).
- The rest of a thinking request stays HELM's, except the temperature: a thinking request is
  sampled at 1.0, not HELM's 0, by gbench's rule (1.0 with thinking, 0.0 without). HELM reaches a
  vLLM-served model through `VLLMChatClient` (`src/helm/clients/vllm_client.py`, an
  `OpenAIClient`), which sends the same request whether the model reasons or not.
  `OpenAIClient`'s reasoning-model branch (`openai_client.py` at `63754d05`, lines 286-314:
  temperature and `top_p` dropped, `max_tokens` renamed `max_completion_tokens`) is taken only
  for OpenAI model names (`is_reasoning_model`: `^o\d+` or `^gpt-5`) and exists to avoid
  parameters OpenAI's hosted reasoning models reject; it is not a reasoning-model setting for a
  served model, so gbench does not apply it.
- A thinking reply is classified as every gbench suite classifies it. One cut off at the budget
  is scored on what arrived and counted in `truncated_responses` (and in
  `replies_cut_off_in_reasoning` when no answer text arrived because the budget ran out during
  the reasoning; `looping_responses` counts the ones that were repeating themselves). One that
  stopped with its answer inside the thinking channel (reasoning, or a content holding only an
  inline thought block, and no answer text) is the model's outcome: it scores 0 and is counted in
  `answer_in_thinking` (with `answer_in_thinking_rate`), not as a failed request and not a
  comparability reason (see [answer_in_thinking.md](answer_in_thinking.md)). One that came back
  with nothing at all (no answer text, no reasoning) is counted in `empty_responses` and
  `failed_requests` and scores 0. A cut-off or empty reply is named in `leaderboard_comparable_reason`; the log
  shows a WARNING for replies cut off at the budget; and the run reports
  `completed_with_errors` when more than 10% of the replies were cut off or any request failed.
- A request that fails with a server or transport error is retried twice. A row that still has
  no reply from the served endpoint (failed, timed out, or over the context window) is a
  harness error, never a wrong answer: it is left out of `accuracy`, `subject_mean_accuracy` and
  `category_accuracy` (`harness_error` and `scoring_excluded` on its trace; `scoring_excluded`,
  `request_harness_errors` and `effective_n` in the result), counted in `failed_requests` and
  named in `status_reason`; when no row is scored, `accuracy` is null with
  `accuracy_withheld_reason`.
- HELM drops trailing examples when a prompt does not fit the model's context window
  (`_make_prompt_fit`). The longest prompt here is about 3,100 tokens with the Gemma 4
  tokenizer, so on any window of 4K tokens or more that never applies; a server with a smaller
  window rejects the request, a harness error (above) that is flagged.

## Reading the result
- `accuracy`: exact match over all scored rows combined, in percent (the paper's "all subjects
  combined"). `subject_mean_accuracy` is the mean of the per-subject values (HELM's own MMLU
  aggregate), and `category_accuracy` gives each subject.
- `non_letter_replies` counts replies that were not a single letter: every answered reply
  without thinking (an empty one-token completion included; `empty_responses` counts those, and
  they are HELM's outcome, not failures), and every complete reply (not cut off, not empty) with
  thinking. Each trace records the `response_text`, the `predicted` letter, the gold, the
  `finish_reason` and its `health`. When more than 10% of those replies are not a letter the run
  logs a WARNING and is flagged non-comparable: without thinking HELM reads one token, so a model
  whose chat template (with thinking off) makes it open with anything but the letter scores 0 on
  those rows, and the number then measures the serving template rather than the model; with
  thinking the reply after the reasoning must be the letter alone, so the WARNING asks you to
  check that the server returns the reasoning separately and that the model ends with the
  letter. The scoring itself stays HELM's exact match. Look at a few `response_text` values when
  you see it.
- `truncated_responses`, `looping_responses`, `empty_responses`, `answer_in_thinking` (with
  `answer_in_thinking_of` and `answer_in_thinking_rate`) and `response_health` (the count of each
  reply health) describe how the replies ended. Without thinking the result also has
  `official_max_tokens_sent` (`[1]`) and `replies_at_official_max_tokens`; with thinking,
  `replies_cut_off_in_reasoning`.
- `status` is `success` when every request was answered and, with thinking, no reply came back
  empty and at most 10% were cut off at the budget; `failed` when no row got an answer (every
  request got no reply or, with thinking, came back empty); `completed_with_errors` otherwise.
  `status_reason` names what made it other than `success`.
- `leaderboard_comparable` is true for a full no-think run at temperature 0 with every request
  answered and at most 10% non-letter replies. Otherwise `leaderboard_comparable_reason` names
  the cause: a `--eval-thinking` run, replies cut off at the budget or with nothing at all,
  an `--eval-limit` subset, a temperature other than 0, requests with no reply (harness errors,
  left out of the score), or
  `non_letter_replies` above 10% of the replies read.
- `max_tokens_sent` lists the per-request max_tokens values sent (`[1]` without thinking); each
  trace records its own.
- `--eval-limit N` scores a subset stratified by subject. That is not a full-benchmark number.
  The run logs its planned workload at its start and records it as `eval_limit_plan`.
- `--shard I/N` scores shard I of N of the 5,330 scored rows (round-robin over the dataset order),
  selected before `--eval-limit`; the result records `shard`, and a shard of more than one is not
  leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`); each
  trace carries a shard-independent `sample_key`, so `gbench --merge` can combine the shards.
  Each trace also records its position among the scored rows (`global_idx`), so rows with
  identical content stay apart. The suite's merge_shards counts the union again by the run's own
  rules (its 10% thresholds and `subject_mean_accuracy` included) and states its non-comparable
  reasons and status again, so the merged result of a full run's shards equals the unsharded
  result.
- `official_sampling` records HELM's setting (temperature 0), where the run's temperature came
  from (`temperature_source`) and whether it matched.

## Requirements
- An OpenAI-compatible `/v1` chat endpoint serving the model.
- Hugging Face Hub access to the public `edinburgh-dawg/mmlu-redux-2.0` (about 4 MB) and
  `cais/mmlu` (about 52 MB for the `all` config) datasets. They download on first use and are
  cached by `datasets`. No API keys, Docker images or checkouts.
- Temperature: gbench's rule, 1.0 for a `--eval-thinking` run and 0.0 for a no-think run (the
  same value as HELM's 0; `official_sampling` records HELM's 0, with
  `temperature_matches_official: false` for a thinking run). `--temperature`, or for this suite alone
  `GBENCH_MMLU_REDUX_TEMPERATURE`, override it in both modes; a value other than 0 makes the run
  non-comparable.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals mmlu_redux --max-output-tokens 65536
```
Without `--eval-thinking` this suite sends HELM's `max_tokens=1` whatever `--max-output-tokens`
says; with `--eval-thinking` it sends `--max-output-tokens` (65536 above).
