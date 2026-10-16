# bfcl_live

The BFCL **Live** categories: user-contributed function-calling prompts (first released with
BFCL v2/v3), run with BFCL's own protocol. It measures whether the model emits the right
function call(s) for real-world requests, and whether it abstains when no provided function fits.

> **Renamed.** This suite was registered as `bfcl_v3_live`, and before that as
> `bfcl_v4_agentic`. `live` is BFCL's own name for this category group (`bfcl_eval`'s
> `TEST_COLLECTION_MAPPING["live"]` holds exactly the six categories below), and the data is the
> Live set as shipped in the current BFCL v4 files, so the `v3` was dropped. `bfcl_v3_live` still
> works as a deprecated alias (one warning per run), and `GBENCH_BFCL_V3_LIVE_TEMPERATURE` is
> still read as a deprecated alias of `GBENCH_BFCL_LIVE_TEMPERATURE`. The older name
> `bfcl_v4_agentic` implied the v4 *agentic* track (web search / memory / format sensitivity);
> it never loaded agentic data. The real agentic track is
> [`bfcl_v4_agentic`](bfcl_v4_agentic.md), which drives Berkeley's `bfcl-eval` harness. Results
> labelled `bfcl_v4_agentic` from before that rename are Live numbers.

## Dataset
The Live files among gorilla's `bfcl_eval` `BFCL_v4_*.json` data, pinned to commit
[`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/data)
(`fc_common.BFCL_EVAL_COMMIT`, identical to `bfcl-eval==2026.3.23`). The stale Hugging Face
v3 mirror is not used: against it the current files fix ground truth and questions, and move two
entries that were mislabelled `live_relevance` into `live_irrelevance`.

| Category | Entries | Correct when |
|---|---|---|
| `live_simple` | 258 | the one expected call |
| `live_multiple` | 1,053 | the one expected call, chosen among several functions |
| `live_parallel` | 16 | all expected calls, any order |
| `live_parallel_multiple` | 24 | all expected calls, any order, among several functions |
| `live_irrelevance` | 884 | no call |
| `live_relevance` | 16 | at least one call |

## Protocol
Identical to [`bfcl`](bfcl.md#protocol) (the two suites share the code in `bfcl.py`): the entry's
turn-0 messages are sent verbatim, system messages included and no gbench system prompt; the
function docs go on the request as `tools` after BFCL's "Note that the provided function is in
Python 3 syntax." hint and `convert_to_tool` (Gorilla types to JSON-schema types, dots in names to
underscores); the request carries no `max_tokens`, as BFCL's handler sends none. Each reply
becomes the model result BFCL's FC handler records (its tool calls, else its text) and is scored
by `bfcl_eval`'s own `eval_runner` (`decode_ast`, `ast_checker` and the relevance rule) in the
local harness image `gbench-bfcl:2026.3.23`: exact number of calls, advertised function name,
required parameters present, no unexpected parameters, exact parameter names, declared types,
strings compared after `standardize_string`. Reply text is never a call, as in BFCL's FC handler:
a reply without structured tool calls makes no call, whatever its text says (a JSON object shaped
like a call included). Irrelevance is correct iff the reply makes no call; an empty reply or a
plain-language refusal is correct.

Sampling: gbench's rule for every suite: temperature 1.0 with `--thinking`, 0.0 without (BFCL's
own 0.001, the default of `bfcl generate --temperature`, is recorded). No `max_tokens` (`--max-output-tokens` is not applied;
`max_output_tokens_note` says so); override the temperature with `--temperature`, or
`GBENCH_BFCL_LIVE_TEMPERATURE` for this suite alone (it takes precedence). `official_sampling` on
the result records the setting, the temperature used and whether it matched. `--eval-limit N` draws a subset stratified over the six
categories.

The run logs its planned workload in one line at its start (for example `[bfcl_live] --eval-limit
5: 5 sample(s) x 1 attempt = 5 generations`) and records it in the result as `eval_limit_plan`
(`limit`, `samples`, `per_setting`, `attempts_per_sample`, `generations`).

## Reading the result
- `accuracy` (headline): BFCL's **Live Overall Acc** column: accuracy over `live_simple`,
  `live_multiple`, `live_parallel` and `live_parallel_multiple`, weighted by entry count (so
  `live_multiple` dominates). Irrelevance and relevance are not part of it, as on the leaderboard.
- `bfcl_summary.live`: the row of BFCL's `data_live.csv`, in percent: Live Overall Acc, AST
  Summary, Python Simple / Multiple / Parallel / Parallel Multiple AST, Irrelevance Detection
  (`live_irrelevance`) and Relevance Detection (`live_relevance`). Compare these with the Live
  columns of the BFCL leaderboard (FC entries).
- `pooled_accuracy`: plain correct/total over all six categories. `category_accuracy`: per
  category.
- `leaderboard_comparable` is always `false`: the headline is one column of BFCL, not BFCL Overall
  (run [`bfcl`](bfcl.md) for that). The reason also lists an `--eval-limit` subset, a
  `--shard` or a temperature other than 0.001.
- Each wrong trace carries `bfcl_error_type` and `bfcl_error` with BFCL's own error codes;
  `single_turn_checker` names the release and image that scored them.
- An empty reply that is the correct answer (on `live_irrelevance`) is marked
  `empty_is_answer` on its trace and is not counted under `empty_responses` or
  `failed_requests`; any other empty reply is counted there (and can make the run status
  `completed_with_errors`).
- A request that got no reply from the endpoint (an endpoint error, a timeout, a prompt over the
  served context window) is a harness error, never the model's outcome, as in every base-run
  suite: it is left out of the score (out of its category's correct and total; the trace's
  status `FAILED`, `TIMEOUT` or `OVER_CONTEXT` and `scoring_excluded`; by kind in
  `request_harness_errors`), counted under `failed_requests` and named in `status_reason`; the
  run is `completed_with_errors`, and `failed` with the score withheld (`accuracy` `null`) when
  nothing was scored. The [`bfcl`](bfcl.md) suite treats its multi-turn and agentic entries the
  same way.
- `harness_errors`: empty unless `GBENCH_BFCL_TIMEOUT_S` is set (seconds; unset by default, as
  BFCL's checker has no wall clock) and the checker ran past it. The replies are then left
  unscored (`HARNESS_ERROR` traces, `scoring_excluded`, out of every accuracy), the entry records
  the part, the error, `timeout_s` and `replies_unscored`, and the status is `failed` when nothing
  was scored (`completed_with_errors` otherwise).
- `dataset_revision` records the pinned commit.

## Requirements
- An OpenAI-compatible `/v1` endpoint that returns structured `tool_calls`: vLLM needs
  `--enable-auto-tool-choice --tool-call-parser <the model's parser>` (e.g. `gemma4`). vLLM
  rejects requests that declare tools with HTTP 400 when it has no parser, and a server gbench
  launches itself does not enable one yet, so serve the model yourself with those flags and point
  `--remote-endpoint` at it.
- **Docker and the local harness image `gbench-bfcl:2026.3.23`** (shared with `bfcl`), built once
  from the gbench repository root:
  ```bash
  docker build -t gbench-bfcl:2026.3.23 -f gbench/docker/bfcl.Dockerfile gbench/docker
  ```
  See [`bfcl`](bfcl.md#requirements) for what it holds. No Docker, no image, an image for another
  release, or a checker file that differs from the pinned commit stops the suite with an
  `[INFRA REQUIRED]` error before anything is generated. Both of this suite's containers, the
  image probe and the single-turn checker, run with `--network none`, so it needs no host
  firewall rule and gbench's metadata-server guard has nothing to probe (see
  [`bfcl`](bfcl.md#containers-and-network)).
- Network access to `raw.githubusercontent.com` on the first run: the Live data files are
  downloaded from the pinned commit into `$GBENCH_CACHE_DIR/bfcl_data/<commit>/` (default
  `~/.cache/gbench/bfcl_data/<commit>/`, shared with `bfcl`). Offline, copy the `BFCL_v4_live_*.json`
  files and `possible_answer/BFCL_v4_live_*.json` from a checkout of that commit (or from the
  harness image, under `/usr/local/lib/python3.10/site-packages/bfcl_eval/data/`) into that
  directory.
- Every data file, cached or downloaded, is checked against its git blob id at the pinned commit
  before it is read; a cached file that does not match is downloaded again. A file that cannot be
  downloaded, or whose download does not match, stops the suite with an `[INFRA REQUIRED]` error.
- No API keys.

## Run
```bash
# serve the model with a tool-call parser, then:
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals bfcl_live --max-output-tokens 4096
```
`--max-output-tokens` is required for every eval run, but this suite does not apply it: its
requests carry no `max_tokens`, as BFCL's do.
