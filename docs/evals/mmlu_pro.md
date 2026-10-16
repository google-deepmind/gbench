# mmlu_pro

MMLU-Pro (Wang et al., 2024): 12,032 reasoning-focused multiple-choice questions in 14
categories, with up to 10 options (A-J). **Scoring: TIGER-Lab's runner for OpenAI-compatible
endpoints (`evaluate_from_apiX.py`) with its defaults; the letter extracted from the reply must
equal the gold letter, and the headline is accuracy over all test questions.**

## Dataset
- `TIGER-Lab/MMLU-Pro`, pinned revision `b189ec765aa7ed75c8acfea42df31fdae71f97be` (the
  dataset's current revision; upstream loads the latest one).
- `test` split (12,032 questions) is scored; the `validation` split (5 worked chain-of-thought
  exemplars per category, 70 in total) supplies the exemplars.

## Protocol
Ported from `evaluate_from_apiX.py` in TIGER-AI-Lab/MMLU-Pro (commit `f418b116`), the runner the
repository's README documents "to use universal (OpenAI chat.completions) API", i.e. a model
served behind an OpenAI-compatible `/v1` URL such as a local vLLM server. The repository's other
API runner, `evaluate_from_api.py`, only has request code for a fixed list of hosted models
(gpt-4, gpt-4o, deepseek-chat, deepseek-coder, o1-preview). gbench runs `evaluate_from_apiX.py`
with its command-line defaults: `--max_tokens 32768`, `--retry 3`, `--retry_wrong 0`, no
`--save-thinking`.
- **5-shot CoT by default.** Upstream puts every validation exemplar of the question's category
  (5) in the prompt. `--eval-n-shot N` uses the first N (0 to 5) and makes the run
  non-comparable; `--eval-n-shot 0` is zero-shot (header and question only). A value above 5 is
  rejected.
- One user message and no system message: the header `The following are multiple choice questions
  (with answers) about <category>. Think step by step and then output the answer in the format of
  "The answer is (X)" at the end.`, a blank line, then each exemplar as `Question: ...\nOptions:
  A. ..\nB. ..\n...Answer: <worked reasoning ending "The answer is (X).">` followed by a blank
  line, then the test question in the same format ending in `Answer: Let's think step by step.`
  Options equal to "N/A" are dropped, as upstream's `preprocess` does.
- The prompt and the request are the same with and without `--eval-thinking`; the run's thinking
  mode is sent as `chat_template_kwargs.enable_thinking`, as every gbench suite sends it.
- Request (`streaming_request`): `temperature=0.1`, `top_p=0.95`, `max_tokens=32768`, streamed
  with `stream_options={"include_usage": true}`, through the `openai` client's `AsyncOpenAI`
  (timeout 3600 s, the client's default retries). The request body, parsed as JSON, equals the
  one openai 1.30.1 (the version the repository's `requirements.txt` pins) sends; only the
  whitespace of its serialization differs.
- Token limit (`max_tokens`): `evaluate_from_apiX.py` sends the same request to every model and
  counts a reasoning model's streamed reasoning against `max_tokens`, so its 32768 is also an
  explicit reasoning-model setting. gbench's token-limit policy (`OfficialSampling.max_tokens`
  and `.reasoning`) applies it: a no-think run sends `max_tokens=32768` exactly; a
  `--eval-thinking` run sends 32768, or the run's `--max-output-tokens` when that is larger (a
  thinking run is never capped below it), and that value is also the limit of the client's
  token count (upstream's `--max_tokens`). `harness.max_tokens` and `max_tokens_sent` record
  the value used.
- Token count: the client counts the prompt and every streamed reasoning and content delta with
  tiktoken's `cl100k_base`. Once the count passes `max_tokens` it drops the stream
  (MaxTokensExceeded) and asks again after min(2^n, 30) seconds, 3 tries in all; after the third
  the question is recorded as MaxTokensExceeded and scores 0. A stream that ends without usage
  statistics, or fails, is asked again after 5 seconds.
- Try bound: upstream wraps each streamed try in `async with asyncio.timeout(3600)`, which needs
  Python 3.11. gbench applies the same 3600-second bound with `asyncio.wait_for`, which every
  Python gbench supports (3.10 and later) has: the request, the whole stream and the usage check
  must finish within it, else the try fails with a TimeoutError and is asked again after 5
  seconds, as upstream asks it again; the stream is closed either way.
- The reply is the streamed content, and only the content is scored. The reasoning channel is
  counted and kept in each trace's `reasoning` field for inspection, but never scored (upstream
  without `--save-thinking` keeps only the content).
- Extraction (`extract_answer`): for `answer is \(?([A-J])\)?`, then `[aA]nswer:\s*([A-J])`,
  then `\b[A-J]\b`, the LAST match of the first pattern that matches anywhere in the reply. A
  reply with no match scores 0; there is no random guess, and a wrong answer is not asked again.
- Metric (`save_summary`): correct / questions, per category and over all questions.
- Pacing: the categories run one after another, in order of first appearance in the test split,
  each category's questions in dataset order. Within a category, request starts are at least 2
  seconds apart (`enforce_start_delay`, retries included) and no new question starts while a
  running one has streamed nothing for 25 seconds (the stall check). gbench's concurrency
  (`--batch-sizes`, recorded as `harness.num_workers`) plays the part of `--num_workers`
  (upstream's default is 4); it changes the pacing, not the scores. A full run therefore takes
  at least 12,032 x 2 s, about 6.7 hours; a `--suite-timeout` must allow for that, and each
  `--shard` has its own spacing. The log shows each category's start and end (with its correct,
  MaxTokensExceeded and failed counts) and every failed or over-limit try.
- If the exemplars cannot be loaded, the suite stops with an `[INFRA REQUIRED]` error. It never
  falls back to zero-shot.

Where gbench differs from upstream:
- Upstream asks a failing request again without limit. gbench stops after 10 tries and records
  the question as `FAILED`: a harness error, never a wrong answer. It is left out of `accuracy`
  and `category_accuracy` (`harness_error` and `scoring_excluded` on its trace;
  `scoring_excluded`, `request_harness_errors` and `effective_n` in the result), named in
  `status_reason` (status `completed_with_errors`), and makes the run non-comparable; when no
  question is scored, `accuracy` is null with `accuracy_withheld_reason`.
- An endpoint that rejects the request outright (HTTP 4xx other than 408, 409 and 429) stops the
  suite with an `[INFRA REQUIRED]` error, since no try can succeed; upstream would ask forever.
  The usual cause is a context window too small for the prompt plus `max_tokens`. When the
  server reports its window, the suite checks this before the first request.
- A `TypeError`, `AttributeError`, `ImportError` or `NameError` raised by the client's request
  call, `chat.completions.create` (an incompatible `openai` package or Python), also stops the
  suite with an `[INFRA REQUIRED]` error at the first question, instead of spending 10 tries on
  every question. The same errors raised while the stream is read (a malformed chunk, for
  example one with no `delta`) fail only that try: it is asked again after 5 seconds, as
  upstream's `process_one` asks it again, and counts toward the 10 tries.

## Reading the result
- `accuracy`: percent of test questions answered correctly, pooled over all scored questions.
  `category_accuracy` breaks it down by category (the 14 leaderboard columns). `n_shot` is the
  exemplar count applied.
- `max_tokens_exceeded`: questions whose three tries all passed the `max_tokens` count (32768
  unless a thinking run's `--max-output-tokens` is larger; scored 0, as upstream scores them).
  Each trace records `max_tokens_exceeded_tries` and, for an answered question, the client's
  `client_token_count` and the server's `usage`; an exceeded question keeps the last 2,000
  characters of its last try in `partial_response_tail`.
- `no_answer_replies`: answered questions with no extractable letter (scored 0, as upstream
  scores them). `empty_responses` and `truncated_responses` (the server stopped at
  `max_tokens`) are counted among the answered questions for diagnosis, and so is
  `answer_in_thinking` (with `answer_in_thinking_of` and `answer_in_thinking_rate`): the empty
  replies that carry reasoning and did not stop at `max_tokens`, where the model wrote its answer
  inside the thinking channel (the model's outcome, scored 0; see
  [answer_in_thinking.md](answer_in_thinking.md)).
- `failed_requests`: questions with no reply after 10 tries (harness errors, left out of the
  score).
- `harness` records the runner's settings (`max_tokens` used and `upstream_max_tokens`
  32768); `max_tokens_sent` is `[max_tokens]`.
- Sampling is recorded under `official_sampling`, with `max_tokens` (32768), `max_tokens_applied`
  (true when 32768 was sent, false when a thinking run sent a larger `--max-output-tokens`
  instead) and, for a `--eval-thinking` run, `reasoning_settings: true` and `max_tokens_sent`
  (the value sent). `max_output_tokens_note` says how `--max-output-tokens` was treated: not
  applied to a no-think run, and sent instead of 32768 by a thinking run when it is larger.
- `leaderboard_comparable` is true when the run is upstream's protocol end to end: the full test
  split, 5-shot, temperature 0.1, `max_tokens=32768` and no decoding penalties, and every
  question answered or recorded as MaxTokensExceeded. Otherwise it is false and
  `leaderboard_comparable_reason` names each cause: a different `--eval-n-shot`, a thinking run
  whose `--max-output-tokens` is above 32768 (so a larger `max_tokens` was sent and counted), an
  temperature other than 0.1 (gbench's 1.0 with thinking or 0.0 without, or an operator-set
  value), decoding
  penalties (`GBENCH_REPETITION_PENALTY` and the like),
  questions that got no reply after 10 tries, or an `--eval-limit` subset.
- `--eval-limit N` scores a subset stratified by category. That is not a full-benchmark number.
  The run logs its planned workload at its start and records it as `eval_limit_plan`.
- `--shard I/N` selects shard I of the test questions (round-robin in dataset order) before the
  `--eval-limit` draw; the result records `shard`, and a shard of more than one is not
  leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`); each
  trace carries a shard-independent `sample_key`, so `gbench --merge` can combine the shards.

## Requirements
- An OpenAI-compatible `/v1` chat endpoint serving the model that streams (`stream: true`) and
  returns usage statistics with `stream_options.include_usage`, as vLLM does. Its context window
  must hold the longest prompt (several thousand tokens with 5 exemplars) plus `max_tokens`
  (32,768, or a thinking run's larger `--max-output-tokens`).
- Python 3.11 or later (gbench's supported versions; the per-try bound does not need
  `asyncio.timeout`).
- The `openai` Python package, 2.31 or later (the client `evaluate_from_apiX.py` uses; gbench
  gives it a placeholder key, which 2.54 and later require, and its HTTP client drops the
  `Authorization` header from every request, so no key is sent), and `tiktoken` with its
  `cl100k_base` encoding. Both are in gbench's `evals` extra (`pip install -e ".[evals]"`).
  The encoding downloads from `openaipublic.blob.core.windows.net` on first use; set
  `TIKTOKEN_CACHE_DIR` to a directory that holds it for an offline run. A missing package or
  encoding is an `[INFRA REQUIRED]` error.
- Hugging Face Hub access to the public `TIGER-Lab/MMLU-Pro` dataset. It downloads on first use
  and is cached by `datasets`. No API keys, Docker images or checkouts; no key is sent to the
  model endpoint.
- Temperature: gbench's rule, not upstream's 0.1: every request is sampled at **1.0 with
  `--eval-thinking` and 0.0 without** (top_p stays 0.95 in both modes). `official_sampling`
  records 0.1 with `temperature_matches_official: false`, so the run is non-comparable. `--temperature`, or for
  this suite alone `GBENCH_MMLU_PRO_TEMPERATURE` (which takes precedence), overrides the
  temperature in both modes; a value other than 0.1 makes the run non-comparable.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals mmlu_pro --max-output-tokens 65536
```
Without `--eval-thinking`, `--max-output-tokens` is not applied (the suite sends
`max_tokens=32768`). With `--eval-thinking`, the suite sends 32768, or `--max-output-tokens` when
that is larger (65536 in the command above; `max_output_tokens_note` and
`leaderboard_comparable_reason` record it). Add `--eval-n-shot 0` for zero-shot.
