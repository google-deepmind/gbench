# browsecomp

Canonical BrowseComp (OpenAI, `openai/simple-evals` `browsecomp_eval.py`): 1,266 hard
web-browsing/research questions with short, verifiable answers. **Scoring: the canonical
BrowseComp LLM grader** decides correct (yes/no) against the decrypted gold answer.

> **Closed book, and that is the canonical protocol.** Upstream BrowseComp declares **no
> tools** and runs **one turn**: browsing-capable models bring their own browser, everything
> else answers from parametric memory. gbench matches that exactly: no injected search, no
> tool loop. BrowseComp is *built* to defeat a non-browsing model (upstream reports **0.6%**
> for GPT-4o without browsing vs 51.5% for Deep Research), so a low score here is the model's
> genuine closed-book fact-retrieval result, **not** a harness bug. An earlier gbench build
> injected a Gemini-grounded `web_search`; that deviated from the protocol and has been removed.

## Dataset
- `smolagents/browse_comp`, split `test`, pinned at revision
  `f975f60afc3c811de82951165f5a9a921d024e14`. At that revision the rows are identical, row for
  row, to simple-evals' canonical
  `https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv` (sha256
  `7b24471cd5b3eb2a46830a14802b5c029ea62f488ff75a0f88af7923d1454abf`, 1,266 rows).
- `problem` and `answer` are XOR-encrypted with a per-row `canary` password; gbench decrypts them
  with simple-evals' own `derive_key` / `decrypt` before use.
- The per-category table uses each row's `problem_topic`. `--eval-limit N` draws a subset
  stratified over it.

The run logs its planned workload in one line at its start (for example `[browsecomp] --eval-limit
5: 5 sample(s) x 1 attempt = 5 generations`) and records it in the result as `eval_limit_plan`
(`limit`, `samples`, `per_setting`, `attempts_per_sample`, `generations`). The
suite is scored in a batch and runs 1 attempt per sample whatever `--attempt-count` asks, so the
plan says 1 attempt.

## Protocol (openai/simple-evals, commit `652c89d0ca9df547706735883097e9537d40dc47`)
- **Prompt**: `QUERY_TEMPLATE` byte for byte (the question, then the Explanation / Exact Answer /
  Confidence format block) as the user turn. `n_repeats` is 1, as upstream.
- **Sampling**: simple-evals samples every eval, BrowseComp included, through the `gpt-4o` entry
  of `simple_evals.py`, `ChatCompletionSampler(system_message=OPENAI_SYSTEM_MESSAGE_API,
  max_tokens=2048)` at the sampler's default temperature 0.5, which prepends the system message.
  gbench sends the same system message and max_tokens, as simpleqa and healthbench do (gpqa's
  `SIMPLE_EVALS_SAMPLING`, applied per mode by base's token-limit policy): the system message
  `You are a helpful assistant.` before the question. The temperature is not the sampler's 0.5 but
  gbench's rule for every suite: 0.0 without `--eval-thinking`, 1.0 with it, named in
  `leaderboard_comparable_reason`. Without `--eval-thinking`
  every request carries **`max_tokens` 2048**: a reply that reaches it is cut there and graded as
  it stands, as simple-evals grades it (`replies_at_sampler_max_tokens` counts them, as
  `replies_at_official_max_tokens` does), and `--max-output-tokens` is not applied
  (`max_output_tokens_note`). With `--eval-thinking` the 2048 is not applied, because it was
  written for non-reasoning generation: each request carries the run's `--max-output-tokens`
  instead (`official_sampling` records `max_tokens` 2048 with `max_tokens_applied: false` and
  `max_tokens_sent`). simple-evals' reasoning-model entries (o-series samplers) are keyed on
  OpenAI model names and are not reasoning settings for a served model (see gpqa).
  `official_sampling` records the temperature sent and `temperature_matches_official`;
  `sampler_max_tokens` is 2048 without `--eval-thinking` and null with it.
- **Grader prompt**: upstream's `GRADER_TEMPLATE` exactly, including its duplicated "if there if
  there", the two blank lines before `confidence:` and the literal `0|\%|` (upstream writes `\%`
  in a plain Python string, which keeps the backslash). `[question]` receives the **raw decrypted
  problem**, as upstream's `grade_sample(problem, answer, response_text)` does, not the
  format-wrapped user turn; `[response]` the model's reply and `[correct_answer]` the decrypted
  gold answer.
- **Grader call**: as simple-evals' `grading_sampler`
  (`ChatCompletionSampler(model="gpt-4.1-2025-04-14", system_message=OPENAI_SYSTEM_MESSAGE_API,
  max_tokens=2048)`): the system message `You are a helpful assistant.`, temperature **0.5**,
  its `max_tokens` 2048 for the reply and no response format, with the prompt as the only user
  message; the model is gbench's Gemini cascade (a gbench convention). gpt-4.1 does not think; the
  Gemini judge counts its thinking against `max_output_tokens`, so the grader is sent
  **`max_output_tokens` 10240**, the 2048 plus the Gemini thinking allowance of 8192
  (`simpleqa.grader_config`, `gemini_budget.GEMINI_THINKING_ALLOWANCE`). A grader reply that still
  stops there is parsed as it arrived and counted in `judge_replies_at_max_output_tokens`, and a
  call whose every answer stopped there with no text is a judge outage, counted in
  `judge_outages_after_max_output_tokens`; either is named in `leaderboard_comparable_reason`.
  `judge_sampling` records these settings.
- **Verdict**: upstream's `grade_sample` returns `re.search(r"correct: (yes|no)", reply).group(0)`
  (`"correct: yes"`) and `__call__` compares that with `"yes"`, so upstream's code as written never
  scores a sample correct. gbench reads the captured `yes` / `no` (the evident intent) with the
  pattern `correct:\s*(yes|no)`, case-insensitive; `judge_verdicts_read_only_by_relaxed_parse`
  counts the verdicts upstream's case-sensitive single-space pattern would miss, and a non-zero
  count is named in `leaderboard_comparable_reason`.
- **A grader reply with no verdict** (no `correct: yes|no` line) is a grader failure, not the
  model's answer: it is marked `JUDGE_FAILED` (`scoring_excluded: true`), left out of the score,
  counted in `judge_failures` (the run is `completed_with_errors`) and named in
  `leaderboard_comparable_reason`. Upstream would score such a reply `no`.
- A request that got no reply from the served endpoint is a harness error: not sent to the
  grader, marked `not_judged_failed_request`, counted in `failed_requests`, left out of the score
  and named in `status_reason` (the run is `completed_with_errors`). An empty reply the server
  returned is the model's outcome: not sent to the grader, marked `not_judged_empty_response`,
  counted in `empty_responses`, and it scores wrong.

## Headline metric and how to read the result
- `accuracy` (headline) = grader `correct: yes` / graded questions, in percent (simple-evals'
  `is_correct`; `metric` says so). A judge-cascade outage and a grader failure are excluded from
  the denominator and counted in `judge_outages` / `judge_failures`; when nothing could be graded
  the run is `failed` and its score withheld.
- `judge` is `gbench-gemini-cascade`, `judge_models` counts the answered grader calls per Gemini
  model (one call per graded question) and `judge_cascade` lists the configured cascade order;
  each trace keeps `judge_model` (the model that answered), `judge_reply` (the full grader reply),
  `judge_finish_reason` and `judge_grade`.
- `leaderboard_comparable` is **False**. `leaderboard_comparable_reason` names every difference
  that applies to the run: the grader is gbench's Gemini cascade, not simple-evals'
  `gpt-4.1-2025-04-14`; an `--eval-limit` subset (n of 1,266); a temperature other than the
  sampler's 0.5; no-think requests whose prompt left less than 2048 tokens of the context window;
  grader failures and verdicts read only by the tolerant parse; grader replies stopped at the
  2048; and replies cut off by a budget other than the no-think 2048 (`replies_cut_off_by_budget`,
  with `replies_cut_off_in_reasoning`).

## Requirements
- An OpenAI-compatible `/v1` chat endpoint serving the model under test.
- **`GEMINI_API_KEY`** or **`GEMINI_API_KEYS`** (one key, or a comma/semicolon list;
  `GEMINI_API_KEYS` wins when both are set) for the grader. The suite checks it with a live call
  before generating and **hard-errors** (`infra_required`) when it is unset or rejected; there is
  no heuristic fallback. Network access to the Gemini API (`generativelanguage.googleapis.com`)
  is needed; `GEMINI_OPENAI_BASE_URL` moves only the key check.
- Packages: the base install (`datasets`) plus `google-genai` (the `evals` extra:
  `pip install -e ".[evals]"`).
- Hugging Face Hub access to download `smolagents/browse_comp` (public; `HF_TOKEN` optional).
- Optional judge knobs (shared by all judged suites): `GBENCH_JUDGE_MODELS` (cascade order),
  `GBENCH_JUDGE_MODEL`, `GBENCH_JUDGE_CASCADE_ROUNDS`, `GBENCH_JUDGE_BACKOFF`,
  `GBENCH_JUDGE_REQUEST_TIMEOUT_S` (default 600 s per grader request; a timed-out request is asked
  once more of the same model, then the cascade moves to its next model).
- Sampling: `GBENCH_BROWSECOMP_TEMPERATURE` (or `--temperature`) overrides the temperature for
  this suite; otherwise gbench's rule applies: 0.0 without `--eval-thinking`, 1.0 with it. `max_tokens` has no override of its own: 2048 without `--eval-thinking`,
  `--max-output-tokens` with it. The grader call is not affected by either.

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals browsecomp --max-output-tokens 16384
```
Add `--eval-limit 20` for a quick stratified subset (flagged in the reason) and
`--eval-thinking` to measure the thinking mode. `--max-output-tokens` is required by the CLI; a
no-think request does not apply it (it sends simple-evals' `max_tokens` 2048), and a thinking
request sends it in place of the 2048.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered questions
(round-robin: positions I, I+N, I+2N, ...), selected before the `--eval-limit` draw, which is then
taken within the shard (N shards run up to N x `--eval-limit` questions). The result records
`shard`, `shard_index` and `shard_count`, and a shard of more than one is not
leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`); `gbench
--merge` combines the shards' results as docs/distributed-gbench.md describes: it re-counts the
union of their traces, and refuses a suite whose headline is not a correct / scored count over
them unless the suite defines its own merge. BrowseComp's merge (`merge_shards`) is that re-count,
with the grader cut-off counts (`judge_replies_at_max_output_tokens`,
`judge_outages_after_max_output_tokens`) summed over the shards and their reason named once, over
the sums, in `leaderboard_comparable_reason`.
