# mrcr

Canonical MRCR (`openai/mrcr`): Multi-Round Co-reference Resolution, a needle-in-a-
haystack long-context retrieval where the target is one of several near-identical
"needles" planted in a long multi-turn context. **Scoring: the canonical MRCR metric.** The response must START with the row's
`random_string_to_prepend` canary (0 otherwise), the canary is stripped from both
sides, and the score is the `difflib.SequenceMatcher` ratio. The headline `accuracy`
is the **mean ratio**; `exact_match_rate` (gbench's, not an upstream metric) reports verbatim
reproductions: replies that start with the canary and, with it stripped from both sides, equal
the answer character for character. A near-copy with a grade of 0.99 (a missing sign-off, added
`**` markup) counts in the mean grade but is not a verbatim reproduction.

## Context window and reporting
- `openai/mrcr` (pinned revision `f4c69fae`) has 2,400 rows: 2, 4 and 8 needles across 8 length
  bins from 4K to 1M tokens, binned by prompt + answer length in `o200k_base` tokens as upstream
  defines them. Each (needles, bin) cell holds about 100 rows (96-106 in the small bins).
- As in the upstream reference loop, rows whose prompt does not fit the served context window
  are **skipped**, not scored 0. Each row must leave room for its own answer: twice its gold
  answer's length plus 512 tokens. A smaller window evaluates fewer bins; it does not lower the
  score.
- Fit is decided exactly where it matters: rows clearly under or over are settled by cheap bounds
  (including the chat template's per-message cost), and borderline rows are counted exactly by
  the server's `/tokenize` endpoint (vLLM), or else with the run's tokenizer plus a conservative
  template allowance. With only a character estimate
  (no `/tokenize`, no loadable tokenizer), borderline rows are skipped unless even the high
  estimate fits, because an over-window row sent to the server would score 0.
- The window comes from the server (`GET /v1/models`): the `max_model_len` of the entry whose id
  is the model under test (matched as `--served-model-id` / `--models` are), never the first entry
  of an endpoint that serves several models. `GBENCH_MRCR_WINDOW` (tokens) overrides it, and is
  required when the server reports none for that model; otherwise the suite stops with an error
  instead of sending rows blind.
- The skip happens before `--eval-limit` draws its subset, and the subset is stratified over
  (needles, bin), so a subset only contains rows that fit.
  The run logs its planned workload at its start and records it as `eval_limit_plan`.
- `--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the rows that fit the window
  (round-robin in dataset order), selected before the `--eval-limit` draw, which is then taken
  within the shard. The result records `shard`, `shard_index` and `shard_count`, and a shard of
  more than one is not leaderboard-comparable (the shard reason is named in
  `leaderboard_comparable_reason`). A shard with no row while other rows fit the window reports
  `accuracy: null` (an empty shard), not an error.
- The result records `mrcr_window_tokens`, `mrcr_rows_in_window`, `mrcr_rows_skipped_over_window`,
  `mrcr_window_decided_by`, `mrcr_bins_tokenizer`, and `mrcr_bins`: the mean grade per needle
  count and bin, with `mrcr_bin_counts`. `mrcr_partial_bins` lists cells only partly inside the
  window (e.g. `98/100`); compare those with published figures with care.
- The per-category table shows the same mean grade; the verbatim-reproduction rate is kept as
  `exact_match_accuracy`, and `correct_answers` counts the verbatim reproductions.
- An empty reply scores 0 and stays in every mean. A request that got no reply from the served
  endpoint (failed, timed out, or over the context window) is a harness error, never the model's
  answer: it is left out of the headline, of `mrcr_bins` and of the per-category table, counted
  (`failed_requests`, `scoring_excluded`, `request_harness_errors`) and named in `status_reason`
  (status `completed_with_errors`); when every request got none, `accuracy` is null with
  `accuracy_withheld_reason`.
- The headline `accuracy` pools needle counts and the bins that fit, so it is marked
  `leaderboard_comparable: false`; compare published MRCR figures against `mrcr_bins`.
  `leaderboard_comparable_reason` also names an `--eval-limit` subset, the temperature, which
  upstream does not send (gbench's 1.0 with `--thinking` or 0.0 without, or an operator-set value), and requests that got no reply.
- Loading counts the rows near the window edge (about 220-240 of them, 8 at a time) and bins
  every row that fits, even for `--eval-limit` runs: expect a few minutes at a 128K window and
  longer at 256K, depending on the server.

## Sampling
The upstream reference loop (the "How to run" section of the `openai/mrcr` dataset card at the
pinned revision) calls `client.chat.completions.create(model=MODEL, messages=messages)` with no
temperature and no `max_tokens`. gbench sends no `max_tokens` either, in both modes (generation
up to the end of the context window), but always sends the temperature of its rule: **1.0 with
`--thinking`, 0.0 without**; the run is recorded with
`temperature_matches_official: false` and the temperature is named in
`leaderboard_comparable_reason`. The run's
`--max-output-tokens` is not applied (`max_output_tokens_note` says so), and the per-row fit rule
above still decides which rows are sent. The result records this under `official_sampling`.
`--temperature`, or `GBENCH_MRCR_TEMPERATURE` for this suite alone (it wins), sends another
temperature instead; the run is then recorded with `temperature_matches_official: false` and the temperature is
named in `leaderboard_comparable_reason`.

## Requirements
- A serving endpoint; the larger its context window, the more length bins are evaluated.
- `tiktoken` (in the `evals` extra) for the `o200k_base` bins; its BPE file downloads on first
  use, so set `TIKTOKEN_CACHE_DIR` to reuse it offline. Without it, bins are estimated at 4
  characters per token, `mrcr_bins_tokenizer` says so, and no `mrcr_partial_bins` claim is made.
- For exact fit decisions, a vLLM endpoint (it has `/tokenize`) or `--tokenizer`.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals mrcr --eval-limit 20 \
       --max-output-tokens 16384
```
