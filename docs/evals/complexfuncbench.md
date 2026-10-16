# complexfuncbench

Canonical **ComplexFuncBench** (`THUDM/ComplexFuncBench`, arXiv:2501.10132; prompts, cascade and
judge models read from commit `c37b284e2f2e03ee456115b7c4b7e537f534be37`): 1000 multi-STEP,
multi-constraint tool-calling episodes over the Booking.com API surface. Each sample is an
agentic loop: the model is driven through a conversation while the harness feeds back the
**recorded** observations for the calls it gets right (the "Golden Function Call List Updating"
strategy), advancing the golden call chain as the model consumes it. This is **not** a
single-turn answer-match; gbench runs the real loop in-process and scores it with
**ComplexEval**.

## What is measured

| metric | meaning |
| --- | --- |
| **Success Rate** (headline `accuracy`) | fraction of samples where the model completes the whole golden call chain and then stops at the right moment |
| **Call Accuracy** | correct calls / total golden calls, pooled |
| **Completeness** (0-2) | a judge scores whether the final natural-language answer covers every part of the request |
| **Correctness** (0-2) | a judge scores whether the final answer is consistent with the observations |

A predicted call is compared with a golden call as upstream's `utils/compare_method.py`
`compare_single_call` does (ported verbatim), tier by tier (`call_match_tiers`:
`["rule_based", "value_checker", "response_based", "llm_based"]`):

1. rule-based exact match (`rule_based`);
2. a `value_checker` over each function's critical parameters (`utils/exact_match_values.json`,
   vendored under `gbench/third_party/eval_suites/_complexfuncbench_data/`); a critical parameter that
   is missing or differs fails the call here;
3. the response-based tier (`response_based`): both calls sent to the live Booking.com API on
   RapidAPI, a match when the two responses are equal (below);
4. the LLM equivalence judge (`llm_based`).

When a step emits several PARALLEL calls, they are aligned to the golden calls with
**bge-large-en-v1.5** embeddings and a max-weight assignment
(`scipy.optimize.linear_sum_assignment`, as upstream) before comparison.

### The response-based tier: the live Booking.com API

The Booking.com API is a tool backend, not a model role, so it runs as upstream runs it (never
replaced by the Gemini judge). Upstream's `response_based` sends both calls of a pair to the live
`booking-com15` API on RapidAPI through its `utils/rapidapi.py` `RapidAPICall`, and gbench ports
both verbatim:

- **Request.** The function's URL and the RapidAPI host come from upstream's
  `utils/tool_info.json`, vendored byte for byte (`TOOL_INFO_SHA256`, checked before the run);
  the headers are `X-RapidAPI-Key` (the value of `RAPID_API_KEY`) and `X-RapidAPI-Host`; the
  call's arguments are the query parameters (`legs` as JSON), sent with `requests.get`, which
  honors `HTTPS_PROXY` / `ALL_PROXY` / `NO_PROXY` as upstream's does. The key is read from the
  environment in gbench's own process and never logged or written to the result.
- **Retries.** A call that gets no response (an HTTP status other than 200, a network error) is
  sent again, 3 attempts in all, 2 s apart (`utils.utils.retry(max_attempts=3)`). A 200 reply is
  read as upstream reads it: when its `status` is true, `timestamp` is dropped and `data` is the
  response. Each request has a timeout (`GBENCH_COMPLEXFUNCBENCH_RAPIDAPI_TIMEOUT_S`, 60 s by
  default; upstream sets none and could wait forever); a request past it is a network error.
- **Verdict.** The pair matches when the two responses are equal. As upstream, a predicted call
  that answers an empty object, or a call that got no response or whose reply cannot be read, is
  no match, and the judge decides the pair. As upstream, the golden call is sent unless the
  predicted call answered an empty object or its reply could not be read.
- **Outages.** A call's last reply is an outage when it is a network error, a timeout, HTTP 5xx,
  401, 403, 407 or 429, or an HTTP 200 whose body is not JSON (`HTTP 200 unreadable body`:
  booking-com15 always answers JSON, so such a body is a proxy's or captive portal's page, and it
  is not asked again, as upstream's retry does not catch it); any other HTTP status, or a 200 JSON
  body without `status` (the `predict_unreadable` outcome for the predicted call), is the
  backend's own answer to that call, and it makes the tier "no match" whatever the other call got
  (as upstream, where either call without a response is no match). When a call's last reply is an
  outage and neither call got such an answer, the backend could not answer the pair and upstream's
  tier might have matched it: the judge still decides it, as upstream, an equivalent verdict
  matches, and a not-equivalent one makes the sample a harness error (`rapidapi_unavailable`,
  below), never the model's failure.
- **One difference.** When the predicted call answers an empty object, upstream also replaces the
  step's error list with the string `API call failed for ...`, so a later error of the same step
  raises `AttributeError`. gbench keeps the list and records the message on the pair (outcome
  `predict_empty_data`); `leaderboard_comparable_reason` names such pairs when they occur.

The result records `response_based_tier`: what upstream does and what gbench does (`upstream`,
`gbench`, `run: true`, `tool`, `tool_info_sha256`) and the run's counts: `calls` (each of up to 3
attempts), `attempts`, `calls_without_response` (by the last reply: `HTTP <status>` or the
exception's class), and the pairs by outcome (`pairs_equal`: matched by this tier;
`pairs_different`, `pairs_no_response` (the predicted call's own answer), `pairs_predict_unreadable`,
`pairs_golden_no_response` (the golden call's own answer), `predict_empty_data`: the judge
decided; `pairs_unavailable`: the backend could not answer). Each sample's record keeps
every comparison in `response_based_calls` (the two calls, each call's replies, the outcome).

### Judges: the Gemini API

Upstream uses GPT-4o in two judge roles: the call-equivalence judge (`llm_based`,
`gpt-4o-2024-05-13`) and the response completeness/correctness judge
(`runner/response_runner.py`, `gpt-4o-2024-08-06`). gbench runs both roles on the **Gemini API**
through its judge cascade (`base.judge_generate_cascade`). Everything but the judge model follows
upstream:

- **Prompts, verbatim.** The system prompts of `prompts/compare.py` and `prompts/response.py`
  are copied character for character (the leading `"` character upstream's strings open with,
  the `## Example` outputs, the reason hints and the "Note: The output must be a valid JSON
  format" line included), and the user templates are filled as upstream's `SimpleTemplatePrompt`
  fills them (`[args1]`, `[args2]`, ... replaced in order, the trailing `output:` lines kept).
  `PROMPT_SHA256` holds each string's sha256 at the pinned commit and a test checks them.
- **Messages.** Upstream's `models/gpt.py` `GPTModel._predict` sends the system prompt as the
  system message and the filled template as the user message at temperature 0.0. gbench sends the
  same two texts as the Gemini `system_instruction` and the user content, at temperature 0.0,
  with no other generation setting.
- **Reading the reply.** Upstream's `utils/utils.py` `decode_json`, verbatim (the fence
  characters stripped, newlines dropped, `True`/`False` turned into JSON booleans, then
  `json.loads`). The call-equivalence verdict is the decoded object's `is_equal`, taken for its
  truth value as `compare_single_call` takes it; a completeness / correctness score must be 0, 1
  or 2. A reply upstream cannot read never becomes a verdict: see "Harness errors" below.
- **Retries.** The response judges ask again, up to 10 requests in all and 2 s apart, whenever
  the reply does not decode to an object with a 0/1/2 score (`runner/response_runner.py` wraps
  `completeness_eval` and `correctness_eval` in `retry(max_attempts=10)`). The call-equivalence
  judge is asked once, as `llm_based` is. A failed API call is retried by the Gemini cascade
  itself (every model, several rounds), where upstream's `GPTModel._predict` retries it 10 times.

The result records:

- `judge_substitution`: each role, its upstream model and the Gemini replacement;
- `judge_prompts`: where the prompts come from and how they are sent;
- `judge_cascade`: the Gemini models tried, in order;
- `judge_models` and `gemini_models_answered`: the answered judge calls per Gemini model;
- `judge_outages`: judge calls (both roles) that every model failed. Neither role's failures are
  scored as the model's outcome; see "Harness errors" below;
- in each sample's record (`sample_traces`, below), every judge call with the Gemini model that
  answered it and its reply.

**`leaderboard_comparable` is always False**, and `leaderboard_comparable_reason` names the
difference (Gemini judges), plus any harness errors of the run and any pair the RapidAPI backend
could not answer.

### Model request timeout

Each model request may take `base.request_timeout_s` of the turn's `max_tokens`: 1456 s for the
2048 a no-think turn sends, 9392 s for a 65536-token `--thinking` turn (the served context window
when no `max_tokens` is sent). `GBENCH_COMPLEXFUNCBENCH_TIMEOUT_S` replaces it. Upstream's 5
attempts, 20 s apart, are unchanged. Until 2026-10-11 the timeout was a fixed 600 s: a
`--thinking` turn of 65536 tokens needs about 546 s at the ~120 tok/s one request of
gemma-4-26B-A4B-it decodes on an idle endpoint, and more under load, so such a turn timed out, was
asked again and could end the sample as an `endpoint_failure` (a hidden cap on the turn's length).
Results from before that date are not comparable on `--thinking` runs. The result records
`model_client_timeout`: `{client, version, timeout_s, connect_timeout_s, retries, source,
replaces, hidden_cap_removed}`. Opening the TCP connection keeps its own 10 s bound (aiohttp
`sock_connect`), so an unreachable endpoint fails an attempt in seconds rather than after the
request timeout. The Gemini judges' requests are bounded too (`GBENCH_JUDGE_REQUEST_TIMEOUT_S`,
600 s; a timeout is asked once more of the same model, then the cascade's next model is): the result
records `judge_client` and the run's `judge_fallbacks` (`{"<model>: <reason>": n}`).

### Harness errors

A failure of the harness, the endpoint or the Gemini judge is never the model's outcome: it is
counted, the run's `status` is `completed_with_errors`, and `leaderboard_comparable_reason` names
it. A sample is **not scored** (left out of the numerator and the denominator of Success Rate
and Call Accuracy, and out of the judge means) when:

| `harness_error_samples` key | cause | upstream |
| --- | --- | --- |
| `endpoint_failure` | the model endpoint request failed on all 5 attempts, 20 s apart (upstream's `models/qwen.py` `retry(max_attempts=5, delay=20)`): an HTTP error (`HTTP 400 (context window)` when the served context window rejected the request), a connection error or a timeout (`GBENCH_COMPLEXFUNCBENCH_TIMEOUT_S`), or a 200 reply that holds no message | `unknown_error`; evaluation.py writes no result for the sample |
| `call_judge_outage` | the Gemini call-equivalence judge failed on every cascade model | `llm_based` returns `None` and the call counts as not equivalent |
| `call_judge_unparsed_reply` | the Gemini call-equivalence judge's reply does not decode (`decode_json`) to an object with an `is_equal` field (a reply cut off before its verdict included) | `llm_based` returns `None` (the call counts as not equivalent), or raises when the decoded reply has no `is_equal` |
| `rapidapi_unavailable` | the `booking-com15` RapidAPI backend could not answer a pair (a call's last reply, after 3 attempts, was a network error, HTTP 5xx, 401, 403, 407 or 429, or at once an HTTP 200 whose body is not JSON, and neither call got another reply) and the Gemini judge then found the pair not equivalent | `response_based` returns `False` and the judge decides; the pair counts as not equivalent |
| `exception` | the harness raised while driving the sample (the result records only the exception class; the log line is redacted) | none |

A completeness / correctness judge that failed on every cascade model, or gave no reply with a
0/1/2 score in its 10 requests (a reply cut off before its score included), leaves that score out
of its mean (as upstream's -1 score), and is a harness error of the run as well. A final answer
that is the empty string gets upstream's -2 ("Do not generate response successfully.") without a
judge call, and is left out of the means as upstream leaves it out.

Success Rate is therefore over the evaluable samples (`n_scored`) on full and subset runs alike.
On a full run with no harness error that is upstream's /1000, and `per_domain_success_rate` is
over each domain's evaluable samples (`per_domain_evaluated`; 150 per domain, Cross 400, when
none failed). The result records:

- `n_harness_errors` and `harness_error_samples`: the samples not scored, by the keys above;
  `n_request_failed` is the `endpoint_failure` count;
- `harness_errors`: one entry per cause, with its counts (`by_error`: the last attempt's error of
  each failed request; `by_type`: the exception classes; `outages` / `unparsed_replies` for the
  judges; `scores_left_out` for the completeness / correctness judge; `judge_calls` for any
  other judge call that every cascade model failed);
- `response_judge_unscored`: the completeness / correctness scores the judge did not give;
- `request_failure_note` when an endpoint failure occurred.

### Per-sample records

`sample_traces` holds one record per sample, with what upstream's `evaluation.py`
`process_example` writes to its output file (`id`, `gen_convs`: the conversation as driven, with
the model's function calls and the observations fed back; `message`; `count_dict`; `resp_eval`)
plus:

- `category` (the domain), `is_correct` (the sample succeeded), `status` (`OK`, or
  `HARNESS_ERROR` for a sample that is not scored) and `harness_error` (its
  `harness_error_samples` key, else null), with `endpoint_error` / `exception` when set;
- `call_judge_calls`: every call-equivalence judge call (the predicted and golden call, the
  Gemini model that answered, or `JUDGE_OUTAGE`, its reply and the `is_equal` read from it);
- `response_based_calls`: every response-based comparison (the predicted and golden call, each
  call's replies, `ok`, `HTTP <status>` or the exception's class, and the outcome: `equal`,
  `different`, `no_response`, `predict_unreadable`, `golden_no_response`, `predict_empty_data`
  or `unavailable`);
- `resp_judge_calls`: every completeness / correctness judge request (role, attempt, model,
  reply, score);
- `turns`: one entry per model reply (its `finish_reason`, `completion_tokens`, the
  `max_tokens_sent`, the number of tool calls and how many failed attempts preceded it).

`total_questions` and `effective_n` are the evaluable samples (`n_scored`) and `correct_answers`
the successful ones (`n_success`), the numerator and denominator of the headline, so the summary
table and `eval_summary.csv` show them next to the Success Rate.

A model reply with **neither content nor tool calls** is the model's outcome, not a harness
error: it ends the sample as a failure (`error_type` `empty_model_message`) and is counted in
`n_empty_model_message`. Upstream's `gpt_runner` files such a reply under `unknown_error`, as it
does an API failure, and writes no result for it, so when it occurs
`leaderboard_comparable_reason` names it.

## Requirements

- **A tool-calling `/v1` endpoint.** The served model must support native tool calls
  (return `message.tool_calls`). For vLLM that means the server was started with
  `--enable-auto-tool-choice` and a matching `--tool-call-parser`: a server gbench launches
  itself (no `--remote-endpoint`) gets both from `--tool-call-parser NAME` (the model's parser,
  e.g. `gemma4` for Gemma 4), and the CLI refuses a local run of this suite without it; a remote
  endpoint must be started that way. A model that instead narrates the call in text will finish
  the first turn as a "final answer" and score as a legitimate failure (not a fabricated number).
- **A Gemini key**: `GEMINI_API_KEYS` (a comma or semicolon separated list) or
  `GEMINI_API_KEY`; `GEMINI_API_KEYS` wins when both are set. It powers both judge roles. The suite
  **hard-errors** (`infra_required`, never skips) when no key is set, or when a live check before
  the run rejects the key or finds a model of the cascade the API does not serve to it. Get a key
  in Google AI Studio (<https://aistudio.google.com/apikey>); the Gemini API has a free tier with
  lower rate limits and a paid tier billed to a Google Cloud billing account.
- **`RAPID_API_KEY`**: the key of a RapidAPI account subscribed to the `booking-com15` API, for
  the response-based tier, as upstream's `utils/rapidapi.py` reads it (upstream's README: "To
  enable response-based evaluation, you need to subscribe the Booking API from RapidAPI"). The
  tier is never replaced by the Gemini judge. To get it:
  1. create a RapidAPI account at <https://rapidapi.com>;
  2. open the API, <https://rapidapi.com/DataCrawler/api/booking-com15>, choose a plan under
     Pricing and subscribe;
  3. copy the key: the `X-RapidAPI-Key` value the API's page shows for your app (the same key as
     in the app's Authorization settings of your RapidAPI developer dashboard), and
     `export RAPID_API_KEY=...`.

  The plans on RapidAPI's pricing page on 2026-09-28 (check it for current terms): BASIC, free,
  50 requests a month (a hard limit); PRO, 8.99 USD a month, 35,000 requests; ULTRA, 50 USD a
  month, 500,000 requests; MEGA, 100 USD a month, 3,000,000 requests. The paid plans bill
  requests past the quota, and some endpoints also count against a smaller "Premium" quota of
  each plan. RapidAPI counts every request against the quota: the check before the run sends one
  to three, and each pair that reaches the tier sends up to two calls, each up to 3 attempts. A
  quota that runs out during the run gives HTTP 429, and the pairs affected follow the outage rule
  above (a harness error when the judge then finds the pair not equivalent), so pick a plan whose
  quota covers the run; the free plan's 50 requests suit only a short `--eval-limit` check.

  Network access from gbench's host to `booking-com15.p.rapidapi.com` (directly or through
  `HTTPS_PROXY` / `ALL_PROXY`). The suite **hard-errors** (`infra_required`) before the run when
  the variable is unset or empty, or when one live call of its `Test_API` function with the key
  gets an outage reply on each of 3 attempts, 2 s apart (a network error, HTTP 5xx, 401, 403, 407
  or 429; 401 / 403: the key is missing, rejected or not subscribed to `booking-com15`; RapidAPI
  answers an unknown key with 403 as well; 429: rate limit or quota). A key that passes the check
  is used for the whole run; it is read in gbench's own process and never logged or written to
  the result.
- **`requests`** for those calls (the suite hard-errors before the run when it cannot be
  imported).
- **The dataset** `THUDM/ComplexFuncBench/ComplexFuncBench.jsonl` (~40 MB, 1000 rows; the Hub
  now redirects the repository to `zai-org/ComplexFuncBench`), pinned to Hub commit
  `5dc7739343f8c87aee465931ba66db13a0fb5dd4` (`HF_REVISION`) and fetched via `huggingface_hub` at
  run time (needs network/HF access, or an HF cache holding that commit; the dataset is public and
  not gated, so no `HF_TOKEN` is needed). The file was uploaded
  once (commit `558831035a`, 2025-01-17) and is byte-identical at every later commit, including
  the one current when the pinned GitHub code revision was made. After the download gbench checks
  its sha256 against
  `be1e0f5951b666f6543b946e915a431a39bbda1807481878eb331245276ac088` (`HF_SHA256`, the file's Git
  LFS oid). Hard-errors (`infra_required`) if it is unreachable or its sha256 differs. The result
  records `dataset` (`THUDM/ComplexFuncBench@<commit>`) and `dataset_sha256`.
- **The embedder**: `BAAI/bge-large-en-v1.5` through `transformers.AutoModel`, CLS pooling + L2
  normalisation (bge's `FlagModel.encode`, which upstream calls; upstream loads it with
  `FlagEmbedding` in fp16 (`use_fp16=True`), gbench at the checkpoint's float32 on the device
  below). The result records `embedding_backend` (`transformers`), `embedding_dtype` (`float32`
  when a step needed the embedder, `null` when no step did, so it was never loaded) and
  `embedding_upstream` (upstream's `FlagModel` call). `transformers` and `torch` come
  from the serving stack; `bge-large-en-v1.5` ships `tokenizer.json`, so no `sentencepiece` is
  needed. On first use it downloads the model (~1.3 GB) into the HF cache. The
  single-call-per-step case (the majority) needs no embedding (a 1x1 assignment is trivial), so
  the model is only loaded when a step has several non-exact parallel calls.
- **`scipy`** for the max-weight assignment (`pip install scipy`, or `pip install -e ".[evals]"`
  in the gbench checkout: it is in gbench's `evals` extra).
- The suite hard-errors (`infra_required`) before the run when `huggingface_hub`, `transformers`,
  `torch` or `scipy` cannot be imported.

## Environment knobs

| var | default | effect |
| --- | --- | --- |
| `GBENCH_COMPLEXFUNCBENCH_TEMPERATURE` | 0.0; 1.0 with `--thinking` (gbench's rule) | per-suite temperature override |
| `GBENCH_COMPLEXFUNCBENCH_MAX_ROUNDS` | `40` | safety cap on loop rounds (longest golden chain is about 19) |
| `GBENCH_COMPLEXFUNCBENCH_TIMEOUT_S` | unset: `base.request_timeout_s` of the turn's max_tokens (2048 without `--thinking`: 1456 s; 65536 with it: 9392 s; the served context window when none is sent) | per-request HTTP timeout of the model driver; setting it replaces the sized value and the result's `model_client_timeout.hidden_cap_removed` becomes false |
| `GBENCH_COMPLEXFUNCBENCH_RAPIDAPI_TIMEOUT_S` | `60` | timeout of one RapidAPI request of the response-based tier (upstream sets none; a request past it is a network error) |
| `GBENCH_COMPLEXFUNCBENCH_EMBED_MODEL` | `BAAI/bge-large-en-v1.5` | embedding model id |
| `GBENCH_COMPLEXFUNCBENCH_EMBED_DEVICE` | `cpu` | embedder device (CPU by default so it never contends with the served model's GPU) |

Sampling: upstream's model request, from its vLLM-served path (`models/qwen.py`
`QwenModel.__call__`; `models/gpt.py` `FunctionCallGPT`, `models/glm.py` and `models/mistral.py`
send the same): `temperature=0.0`, `tool_choice="auto"`, `max_tokens=2048` (`CFB_SAMPLING`).

- **Temperature**: 0.0 without `--thinking`, 1.0 with it (gbench's rule for every suite);
  `--temperature` or
  the per-suite variable
  above sets another, and the result records it (`official_sampling.temperature_matches_official`
  and `leaderboard_comparable_reason`).
- **max_tokens** follows the token-limit policy (`base.resolve_token_budget`). Without
  `--thinking` every turn sends `max_tokens` 2048, whatever `--max-output-tokens` says. With
  `--thinking` the 2048 is not applied, because it was written for non-reasoning generation, and
  every turn is sent the run's `--max-output-tokens` (else the thinking floor); the result records
  that the official cap was not applied to reasoning, and names the per-turn budget in
  `leaderboard_comparable_reason`.
- The result records `official_sampling` (the reference, the official values, the temperature's
  source and whether the 2048 was applied), `max_tokens_sent`, `max_output_tokens_note`,
  `model_turns` and `turns_stopped_at_max_tokens` (turns that stopped at `max_tokens`: at the
  official 2048 on a no-think run, the protocol's outcome, scored as upstream scores it).
- `chat_template_kwargs.enable_thinking` follows `--thinking`.
- The Gemini judges run at 0.0, as upstream's `GPTModel`.

## Run

The full run (all 1000 samples) against a served tool-calling endpoint (the model id is an
example; any served model):

```bash
export GEMINI_API_KEY=...            # or GEMINI_API_KEYS=k1,k2; the two judge roles
export RAPID_API_KEY=...             # booking-com15 on RapidAPI; the response-based tier
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals complexfuncbench \
       --eval-concurrency 8 --max-output-tokens 2048
```

Add `--eval-limit 20` for a quick check on a stratified subset. Without `--remote-endpoint`
gbench launches the server itself and needs the model's tool-call parser
(`--tool-call-parser NAME`) and the number of GPUs to serve it on (`--num-gpus N`, which the CLI
requires for a local run; tensor parallelism follows it), for example:

The run logs its planned workload in one line at its start (for example `[complexfuncbench]
--eval-limit 5: 5 sample(s) x 1 attempt = 5 conversations`) and records it in the result as
`eval_limit_plan` (`limit`, `samples`, `per_setting`, `attempts_per_sample`, `conversations`).

```bash
gbench --evals-only --models google/gemma-4-E4B-it --tool-call-parser gemma4 --num-gpus 1 \
       --evals complexfuncbench --eval-concurrency 8 --max-output-tokens 2048
```

`--max-output-tokens` is required by the CLI; a no-think run sends upstream's per-turn 2048
whatever its value, and a `--thinking` run sends the run's budget (never below
`--max-output-tokens`), as described under Sampling.

`--shard I/N` takes shard I of the 1000 rows in the dataset's order (every N-th row, before
`--eval-limit`, which then draws within the shard); the result records `shard`, and
`gbench --merge` recomputes ComplexEval over the union of the shards' per-sample records
(`merge_shards`, the per-domain rates when the union is the full set). A shard with no row
reports no score (`accuracy` null, `empty_shard`). `--attempt-count` is not applied (the harness
drives each sample once) and a value above 1 is recorded under `attempts`.

`--eval-limit` takes a stratified sample across the five domains (Car-Rental / Hotels /
Attraction / Flights / Cross), and `leaderboard_comparable_reason` then names the subset
(`--eval-limit N: n of ComplexFuncBench's 1000 samples`); a **full** run over all 1000 samples
reports the per-domain Success Rate over each domain's evaluable samples (upstream's
normalisation, 150 per domain and Cross=400, when no harness error occurred). The result payload
carries `status` (`success`, or `completed_with_errors` when a harness error occurred),
`success_rate`, `total_questions`, `correct_answers`, `effective_n`, `call_accuracy`,
`completeness`, `correctness`, `per_domain_success_rate`, `per_domain_evaluated`, `aggregation`,
`n_samples`, `n_scored`, `n_success`, `n_request_failed`, `n_harness_errors`,
`harness_error_samples`, `harness_errors`, `response_judge_unscored`, `n_empty_model_message`,
`embedding_backend` (`transformers`), `embedding_dtype`, `embedding_upstream`,
`call_match_tiers`, `response_based_tier`, `judge_substitution`, `judge_prompts`,
`judge_cascade`, `judge_models`, `gemini_models_answered`, `judge_outages`, `upstream_revision`,
`dataset`, `dataset_sha256`, the sampling provenance (`temperature`, `temperature_source`,
`official_sampling`, `max_tokens_sent`, `max_output_tokens_note`, `model_turns`,
`turns_stopped_at_max_tokens`) and `sample_traces`. `leaderboard_comparable` is always False
(Gemini judges instead of GPT-4o), and `leaderboard_comparable_reason` also names a subset, the
embedder's float32 (when a step loaded it), a temperature other than 0.0, a per-turn budget other
than 2048, pairs the RapidAPI backend could not answer and pairs whose predicted call answered an
empty object. When every sample ends
in a harness error there is no number to report and the suite hard-errors (`infra_required`)
with the counts.
