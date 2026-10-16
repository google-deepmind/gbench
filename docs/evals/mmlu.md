# mmlu

MMLU (Hendrycks et al., 2021): a 57-subject multiple-choice knowledge test with 4 options
(A-D). **Scoring: the original protocol of `evaluate.py` in hendrycks/test; 5-shot, the
prediction is the answer letter with the highest log-probability as the next token, and the
headline is accuracy over all 14,042 test questions.** See also [mmlu_pro](mmlu_pro.md) (10
options), [mmlu_redux](mmlu_redux.md) (re-annotated subset) and
[multilingual_mmlu](multilingual_mmlu.md) (human translations).

## Dataset
- `cais/mmlu`, config `all`, pinned revision `c30699e8356da336a370243923dbaf21066bb9fe`.
- `test` split (14,042 questions) is scored; the `dev` split (5 questions per subject) supplies
  the exemplars.

## Protocol
Ported from `evaluate.py` in hendrycks/test (commit `4450500f`):
- **Prompt** (no chat template, one plain-text prompt per question): the header `The following are
  multiple choice questions (with answers) about  <subject words>.` (the reference's
  `format_subject` gives the double space, kept as is), a blank line, then the first five dev
  exemplars of the question's subject as `question\nA. ..\nB. ..\nC. ..\nD. ..\nAnswer: X` each
  followed by a blank line, then the test question in the same format ending in `Answer:`.
  Question text is used as stored, unstripped, as the reference does. One difference from running
  `evaluate.py` literally: it reads the CSVs with pandas, whose default `na_values` turn an option
  whose text is `None` into `nan`, so for 10 test questions (college_computer_science 2,
  college_mathematics 3, college_physics 1, electrical_engineering 4) the reference prompt shows
  `nan` where the dataset says `None`. gbench keeps the dataset text. The other 14,032 prompts,
  and all 14,042 gold answers, are byte-identical to the reference's.
- **5-shot by default** (`--ntrain 5`). `--eval-n-shot N` uses the first N dev exemplars
  (0 to 5); a value above 5 is rejected. If the exemplars cannot be loaded, the suite stops with
  an `[INFRA REQUIRED]` error; it never falls back to zero-shot.
- **Request** (evaluate.py's `openai.Completion.create` call): one `POST /v1/completions` per
  question with the prompt as text, `max_tokens=1`, `logprobs=100` and `temperature=0`. The
  server tokenizes the text, as the reference's did.
- **Scoring**: each letter's log-probability is read from the top-100 list of the one generated
  token, under the keys `" A"`, `" B"`, `" C"`, `" D"` (`top_logprobs[-1][" A"]` ..); a letter
  that is not in the list counts -100, as in the reference, which logs a warning for it (gbench
  logs one summary warning). The prediction is the highest (the first on a tie, like
  `np.argmax`, so a question with no letter in the list predicts A), and the question counts
  when that is the gold letter. The generated token itself is not scored.
- **Key format**: that read works only when the server keys the list by token text with the
  leading space. A server that keys it otherwise would give every question four -100s and so an
  all-A score; gbench never reports that as a result. A reply that lists none of `" A"` .. `" D"`
  and whose keys show another format is a harness error: every key a `token_id:<id>` placeholder
  (a vLLM server started with `--return-tokens-as-token-ids`), or all four letters listed without
  the space (`"A"` .. `"D"`, as vLLM before v0.26.0, PR #48674, keys tokens for tokenizers whose
  one-token decode drops the space, such as Llama- and Mistral-style SentencePiece tokenizers;
  Gemma 4's tokenizer is keyed `" A"` on either). On the first question this stops the suite with
  an `[INFRA REQUIRED]` error naming the cause; later it makes the question a harness error (see
  the next point but one). And when no scored reply of the run lists any of the four letters,
  whatever its keys, the suite stops with an `[INFRA REQUIRED]` error that quotes the keys of such
  a reply, instead of reporting the all-A accuracy. A reply that lists other text tokens but no
  letter is the model's outcome and is scored as the reference scores it (A), as long as some
  reply of the run lists a letter.
- **No `echo`**: evaluate.py also sets `echo=True`, which only puts the prompt tokens' entries
  ahead of the generated token's in the reply; `top_logprobs[-1]` is the same list either way.
  gbench leaves `echo` out, because on vLLM `echo` with `logprobs` also computes the top-100
  log-probabilities at every prompt position, over the full vocabulary. None of the scoring
  reads them, and that computation at every position of several 5-shot prompts at once has run
  a vLLM server started with default settings out of GPU memory.
- **Context window**: the reference drops trailing exemplars while a prompt does not fit its
  model's context (`crop` keeps GPT-2's first 2,048 tokens, for GPT-3's 2,049-token context).
  gbench applies the same rule to the served model's context window, read from the server: the
  prompt and the one generated token must fit. Tokens are counted with the served model's
  tokenizer and its own defaults (`tokenizer.encode`, the encoder shared with [wmdp](wmdp.md)),
  which is how the server tokenizes the text, so a BOS token is counted exactly when the
  tokenizer adds one by default; Gemma 4's tokenizer does not (`encode("Hello")` gives
  `[9259]`), so Gemma 4 prompts carry no BOS under the reference protocol. Expect this to cost
  Gemma-family models points: they are trained on sequences that start with BOS, so a prompt
  without it likely gets lower log-probabilities for the right letter than the same prompt with
  it; gbench keeps the reference protocol and does not add one. `shots_reduced` counts the
  questions it shortened and each trace records its
  `n_shot`; the result records `tokenizer` and `bos_prepended`. The longest 5-shot prompt is
  about 3,100 tokens with the Gemma 4 tokenizer, so on any window of 4K tokens or more nothing
  is dropped.
- A request that fails with a server or transport error is retried twice with backoff. A
  question that still has no usable reply is a harness error, never a wrong answer: it is left
  out of `accuracy` and `category_accuracy` (`harness_error` and `scoring_excluded` on its trace;
  `scoring_excluded`, `request_harness_errors` and `effective_n` in the result), counted in
  `failed_requests` and named in `status_reason` (status `completed_with_errors`, not
  comparable). When no question is scored, `accuracy` is null with `accuracy_withheld_reason`. A
  server that stops answering mid-run is reported that way, never as a model result.
- The request carries gbench's no-think temperature 0.0 (the request is never a thinking one),
  the same value as evaluate.py's `temperature=0`; `--temperature` or `GBENCH_MMLU_TEMPERATURE`
  replaces it (the result records `temperature` and `temperature_source`). The letters are read
  from the log-probabilities rather than the sampled token, so the temperature does not change
  the score;
  `--max-output-tokens` has none either (the request asks for the one token evaluate.py asks
  for), and `--attempt-count` is recorded as not applied (evaluate.py makes one request per
  question). The letters are read from one greedy step, but the server's numerics can vary
  slightly between runs under concurrent load (batch-dependent kernels), so a question whose top
  two letters are nearly tied can flip between runs: a small `--eval-limit` score can move by a
  whole question from run to run.
- `--eval-thinking` cannot be applied either. A thinking run logs a WARNING, and the result
  records `thinking: false`, `thinking_requested: true`, `thinking_applied: false` and the reason
  in `thinking_not_applied_reason`. The score is the same one a no-think run gives.

## Reading the result
- `accuracy`: percent of test questions answered correctly, pooled over all questions like the
  reference's "Average accuracy". `category_accuracy` breaks it down by subject. `n_shot` is the
  exemplar count requested. Each trace records the four letters' log-probabilities
  (`choice_loglikelihoods`, -100 for a letter outside the top 100), `letters_missing`, the size
  of the returned list (`top_logprobs_returned`), the `predicted` letter and the gold.
- `letters_missing` (letters that counted -100), `questions_without_a_letter` (questions with no
  letter in the list), `top_logprobs_requested` (100) and `top_logprobs_returned_max` (the
  longest list any reply carried) describe the replies. Missing letters are part of the
  reference protocol and do not change the flag (a run in which no scored reply lists any
  letter stops instead; see Key format).
- `short_top_logprobs` counts questions whose reply listed fewer than the 100 entries requested
  while a letter was missing from it: that letter's -100 may come from the server's shorter
  list rather than the model, so any such question makes the run non-comparable. (A vLLM server
  lists 100 entries, keyed by each token's text; two tokens with the same text share one key.)
- `leaderboard_comparable` is true for a full 5-shot run in which every question was scored.
  Otherwise `leaderboard_comparable_reason` names the cause: an `--eval-n-shot` other than 5, an
  `--eval-limit` subset, questions with no usable reply (harness errors, left out of the score;
  including replies whose keys show another format, see Key format above), or
  `short_top_logprobs`.
- `--eval-limit N` scores a subset stratified by subject. That is not a full-benchmark number.
  The run logs its planned workload at its start and records it as `eval_limit_plan`.
- `--shard I/N` scores shard I of N of the full question list (round-robin over the dataset
  order), selected before `--eval-limit`; the result records `shard`, and a shard of more than one
  is not leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`).
  The shards are disjoint and together cover all 14,042 questions; each trace carries a
  shard-independent `sample_key`, so `gbench --merge` can combine the shards. Each trace also
  records its position in the full set (`global_idx`), so rows with identical content stay
  apart. The suite's merge_shards states the union's non-comparable reasons and status again, so
  the merged result of a full run's shards equals the unsharded result.
- Scores from a vLLM server in bf16 can differ in the last decimal places from other backends
  because of numerics.

## Requirements
- An OpenAI-compatible endpoint whose **`/v1/completions`** returns the top 100
  log-probabilities of a generated token (`logprobs=100`), keyed by token text (`" A"`).
  **A vLLM server must be started with `--max-logprobs 100`** (or a larger value, or `-1` for
  no cap): its default allows 20 and rejects the request with HTTP 400 (`Requested sample
  logprobs of 100, which is greater than max allowed: 20`). For example:
  `vllm serve google/gemma-4-E4B-it --max-logprobs 100`. The flag changes no score; it only
  lets the server return the list the protocol asks for. The eval server gbench starts itself
  (a run without `--remote-endpoint`) is given the flag; a server you start yourself needs it.
  The server must not be started with
  `--return-tokens-as-token-ids`, and a model whose tokenizer drops the leading space in a
  one-token decode needs vLLM v0.26.0 or later (see Key format above). The suite sends one
  question first and stops with an `[INFRA REQUIRED]` error naming these requirements if the
  endpoint rejects the request, returns no top-log-probability list or keys it otherwise,
  rather than falling back to a generative prompt. Chat-only endpoints
  (`/v1/chat/completions` only) cannot run this suite.
- The served model's Hugging Face tokenizer, loadable with `transformers`
  (`AutoTokenizer.from_pretrained`), to fit the prompts to the window. gbench uses
  `--tokenizer`, else the model id; if it cannot be loaded the suite stops with an error.
- Hugging Face Hub access to the public `cais/mmlu` dataset. It downloads on first use (about
  52 MB for the `all` config) and is cached by `datasets`. No API keys, Docker images or
  checkouts.

## Run
Against a server started with `--max-logprobs 100` (see Requirements):
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals mmlu --max-output-tokens 65536
```
`--max-output-tokens` is required by the CLI for every eval run, but it does not affect the
MMLU score: each letter is read from the top-100 log-probabilities of the one token the request
generates, and that token is not scored. The result says so in `max_output_tokens_note`. Add `--eval-n-shot 0` for a zero-shot run (not
leaderboard-comparable).
