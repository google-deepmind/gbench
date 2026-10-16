# bfcl_v4_agentic

The **BFCL v4 Agentic** track (Berkeley Function-Calling Leaderboard), run through
Berkeley's own `bfcl-eval` harness, in the local image `gbench-bfcl:2026.3.23`. This is a
*stateful tool-execution* benchmark, not an answer-matching dataset: the agent calls memory /
web-search tools across many turns and is graded on the resulting state and answer, so gbench
runs the canonical harness rather than re-implementing it (the same approach as
[tau2](tau2.md) / [tau3](tau3.md)).

> **Not to be confused with [`bfcl_live`](bfcl_live.md).** That suite (formerly
> mis-named `bfcl_v4_agentic`, then `bfcl_v3_live`) loads the BFCL **Live single-turn**
> categories plus irrelevance abstention, roughly the "Live" 10% slice of the v4 leaderboard. It measures
> something real, just not the agentic track.

## Categories
From the pinned harness, `bfcl-eval` 2026.3.23 (`TEST_COLLECTION_MAPPING["agentic"]`), which is
byte-identical to `bfcl_eval/` at gorilla commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`. The
v4 data ships inside the package (`BFCL_v4_memory.json`, `BFCL_v4_web_search.json`, their
possible answers, the memory prerequisite conversations and the function docs); nothing is
downloaded from HF. Before a run every package file the track reads or runs (the CLI, the
generation driver, the FC handler and its loop, the memory and web-search backends, the agentic
checker and aggregation, and those data files: 35 files, `_AGENTIC_PINNED_BLOBS`) is checked
inside the harness image against its git blob id at that commit; a copy that differs, an older
release included, stops the suite with an `[INFRA REQUIRED]` error naming the files.

| Category | Scored questions | Needs a search backend |
|---|---|---|
| `memory_kv` | 155 | no |
| `memory_vector` | 155 | no (local faiss + sentence-transformers) |
| `memory_rec_sum` | 155 | no |
| `web_search_base` | 100 | **yes** |
| `web_search_no_snippet` | 100 | **yes** |

Leaderboard weighting for context: Agentic 40%, Multi-Turn 30%, Live 10%, Non-Live 10%,
Hallucination 10%.

A run covers all five categories. `--eval-categories` (comma-separated category names)
narrows it to a subset. The flag is shared with the `bfcl` suite and routed per suite: this
suite receives only the agentic names above (the same `MEMORY_CATEGORIES` and
`WEB_SEARCH_CATEGORIES` the CLI validates against), and runs its full track when the list holds
none of them (for example `--evals all --eval-categories simple_python`). A name that no
selected suite knows is rejected by the CLI, and an unknown name passed to the suite directly
is an error. A subset is recorded in `bfcl_report.categories_subset` and is never
leaderboard-comparable. The suite refuses (`infra_required`) a `bfcl-eval` whose agentic
category mapping differs from these five.

## Protocol
- `bfcl generate` then `bfcl evaluate`, each in its own container of the harness image
  (`gbench/docker/bfcl.Dockerfile`, see Requirements) against the gbench endpoint
  (`OPENAI_BASE_URL` / `REMOTE_OPENAI_BASE_URL` and `--skip-server-setup`, so no second GPU
  server is started). `bfcl generate` runs on docker's bridge network and `bfcl evaluate` with
  `--network none`, never the host network (see
  [Containers and network](#containers-and-network)). Each container gets the host user's
  uid/gid, gbench's `eval_suites` directory read-only and the project root at the same path,
  where the harness writes its results and scores and gbench reads them back; gbench's own
  process never imports `bfcl_eval`. Only the variables the harness reads are passed in (the
  endpoint URL and a placeholder for its key, the project root, the thinking switch, the
  execution-guard log and, for the `bfcl generate` that runs a web_search category, the web
  search log, the Gemini relay's placeholder and address, gbench's Gemini cascade and the search
  knobs), through the environment, never the command line. No key enters a container (see
  [Credentials and the execution guard](#credentials-and-the-execution-guard)); the containers'
  output is logged with every credential masked. `--sandboxes`/concurrency maps to
  `--num-threads`.
- Generations are regenerated on every run (`--allow-overwrite`); BFCL's own cache would
  otherwise re-score old generations as this run's result.
- `--eval-limit N` writes `test_case_ids_to_generate.json` (chosen in the image from the
  release's data: N scored questions spread
  round-robin across the categories and stratified by scenario inside each, plus the memory
  `*_prereq` entries they depend on) and runs with `--run-ids --partial-eval` in separate
  `result_limitN` / `score_limitN` directories. If the id file cannot be built the suite
  hard-errors rather than run the whole track.
- `--shard I/N` uses the same id file: each category's scored questions, in the release's
  dataset order, are cut to shard I (every N-th question starting at the I-th) before
  `--eval-limit` draws within the shard, plus the prerequisite entries they depend on (two
  shards may both run one; prerequisites are never scored). The directories are
  `result_shardIofN` / `score_shardIofN` (`result_limitN_shardIofN` with a limit), the result
  records `shard`, and a shard with no question of any selected category reports no score
  (`accuracy` null, `empty_shard`) without generating anything. `gbench --merge` sums each
  category's correct / total over the shards (their entries are disjoint) and applies BFCL's
  Agentic aggregation to the sums (`merge_shards`), which is what one run computes. When bfcl
  runs this track as its agentic part, bfcl runs it whole in its shard 1 (bfcl's own rule) and
  this track is not sharded again.
- `--attempt-count` is not applied (`bfcl generate` generates each entry once); a value above 1
  is recorded under `attempts`.
- Before `bfcl evaluate`, the model's score files for the selected categories are deleted:
  `bfcl-eval` never removes old score files, so a failed or partial evaluate would otherwise
  leave an earlier run's numbers in place. A non-zero exit from `bfcl evaluate` is a hard error
  even when some score files were written.
- Only the selected categories' score files, under this model's own score directory, are read.
  Every generated category must produce a score file; a missing one is a hard error, never a
  partial result.

The run logs its planned workload in one line at its start (for example `[bfcl_v4_agentic]
--eval-limit 5: 5 sample(s) x 1 attempt = 5 generations`, with the scored questions of each
category) and records it in the result as `eval_limit_plan` (`limit`, `samples`, `per_setting`,
`attempts_per_sample`, `generations`, and `note` for the memory prerequisite entries).

### Model registration
BFCL grades through per-model handlers and ships **no entry for the served model**, so gbench
registers it at runtime under its served id with **`OpenAICompletionsHandler` in
function-calling mode**. That entry always replaces what the registry holds under the id: a
served id that equals one of BFCL's own entries (a provider's API model, a prompt-mode handler)
is still run as the served endpoint with gbench's handler. Its metadata names no vendor
(`display_name` `<id> (gbench)`, `org` the prefix of an `org/name` id or `unknown`, `license`
`unknown`, no URL). A local prompt-mode handler is deliberately *not* used: it hand-builds a raw
prompt string and parses prompt-style `[func(a=1)]` text back out, which fails on every row
against a model served with native tool calling ("Failed to decode the model response"). The
endpoint must therefore parse tool calls: a server gbench launches itself gets
`--enable-auto-tool-choice --tool-call-parser NAME` from gbench's `--tool-call-parser NAME` (the
CLI refuses a local run of this suite without it); a remote endpoint must be started that way.
The handler subclass
adds only the `--thinking` toggle below; requests, decoding (`decode_ast` / `decode_execute` are
not overridden) and the turn loop are the harness' own, so BFCL's FC rules apply exactly. A reply
without tool calls carries no `tool_calls` field (as OpenAI and vLLM answer; vLLM stopped sending
an empty list in June 2026, PR #44105), so `_parse_query_response_FC` keeps its text, the FC
decoder raises on it, and the harness logs "Failed to decode the model response. Proceed to next
turn." and ends the turn, as it does for every FC model; that text is what the agentic checker
grades (the last message that does not decode to a call). Nothing in the reply text is ever run
as a call. The registration happens in the container, before the CLI dispatches, with
`underscore_to_dot` as upstream's entry for that handler has it; the model directories already
under the project root are registered too, because `bfcl evaluate` builds its CSVs over all of
them.

### Search backend
BFCL's web_search tool, `WebSearchAPI.search_engine_query` (`web_search.py`), queries DuckDuckGo
**via SerpAPI** upstream. gbench serves it with **Gemini Google Search grounding** instead: every
external service a gbench harness calls goes through the Gemini API, the one the judged suites
use, and there is no other search backend and no opt-in. The benchmark stays BFCL's: the data,
the prompts, the counts, the harness, the scoring and the tool contract the model sees.

- **How it is served.** In the `bfcl generate` container that runs a web_search category, before
  the CLI dispatches, gbench replaces `WebSearchAPI.search_engine_query` on the class, in memory
  (a plain function, so the harness maps it as a method like upstream's; no `bfcl_eval` file is
  changed and the pinned-file check still holds). The replacement sends the query to the
  grounding model cascade of gbench's `search_tool` (`search_with_fallback`: Gemini
  `generateContent` with the `google_search` tool and the query as the prompt, a global
  request-rate limit and retries of the whole cascade; see Env knobs). The models, in order, are
  gbench's one Gemini cascade, the same for every Gemini role (`base.GEMINI_CASCADE`):
  `gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-3.5-flash`,
  `gemini-3.1-pro-preview`, unless `GBENCH_SEARCH_MODELS` (else `GBENCH_SEARCH_MODEL`) names
  others for the search alone, or `GBENCH_JUDGE_MODELS` (else `GBENCH_JUDGE_MODEL`) for every
  Gemini role (`base.search_cascade()`). gbench hands that list to the container as
  `GBENCH_SEARCH_MODELS` and `GBENCH_JUDGE_MODELS` (`base.gemini_cascade_env`), so the searches
  ask exactly the models the Gemini relay forwards and the result records; a model the API
  answers 404 for is not asked again during that search.
- **The tool contract stays BFCL's.** The function docs the model sees
  (`multi_turn_func_doc/web_search.json`), the arguments (`keywords`, `max_results=10`,
  `region="wt-wt"`) and the result shape: a list of `{"title", "href", "body"}`
  (`{"title", "href"}` in `web_search_no_snippet`, as upstream's `show_snippet`), cut to
  `max_results` as upstream cuts it (a value the slice cannot take raises the same error, before
  a search is spent). `fetch_url_content` is the harness' own.
- **What differs** (recorded in `bfcl_report.search_substitution`): `region` is accepted and not
  applied (grounding takes none); `title` is the source's title or domain as grounding reports
  it; `href` is Google's grounding redirect URL to the source, which `fetch_url_content` follows
  (it fetches with redirects); `body` is the grounded answer text Gemini attributes to that
  source, or the start of the grounded answer when it attributes none, not a page snippet; a
  query with a grounded answer and no source returns one result titled `grounded-answer`, one
  with neither an empty list.
- **Comparability.** A run that scored the Web Search setting is not leaderboard-comparable;
  `leaderboard_comparable_reason` holds "web search served by Gemini Google Search grounding
  (BFCL queries DuckDuckGo via SerpAPI)". `bfcl_report.search_backend` starts with
  `gemini-google-search-grounding` and names the model cascade when a web_search category ran,
  with `none:` and the reason when the Gemini key was missing or rejected before generation, and
  reads `not needed` when no web_search category was selected. The memory categories perform no
  search.
- **Which model answered.** Every search is logged with the model that served it; the result
  has `gemini_models_answered` (`{model: searches}`) and `gemini_cascade` (the cascade asked)
  whenever a web_search category was generated, and `bfcl_report.search_substitution` names the
  cascade (`models`) and where it came from (`models_source`: `GBENCH_SEARCH_MODELS`,
  `GBENCH_SEARCH_MODEL`, `GBENCH_JUDGE_MODELS`, `GBENCH_JUDGE_MODEL` or `default`).

BFCL V4's Agentic track has two settings, each its own leaderboard column: Memory (the three
`memory_*` categories, which never search) and Web Search (the two web_search categories). The
Web Search setting runs whenever a Gemini key is set and one of its entries is to run (an
`--eval-limit` or `--shard` id file is drawn first: a draw with no web_search entry checks no key,
starts no Gemini relay and reports nothing about the setting), the same gate the judged suites
use: the
key is read from `GEMINI_API_KEYS`, else `GEMINI_API_KEY` (one key or a comma or semicolon
separated list), and checked with the Gemini API before anything is generated
(`base.gemini_key_live_valid`; a rejection, HTTP 401 or 403 or a 400 that names the API key,
blocks it, and so does a model of gbench's Gemini cascade, the judge and search models, that the
API does not serve to that key, named with the knob to fix; a check that gets no answer does
not). Without a key, or with one the Gemini API rejects, a run that selected a web_search
category **hard-errors** (`infra_required`) before anything is generated: the suite's score,
BFCL's Agentic score, weighs the Web Search setting, and a Memory-only number is never reported
as it. The error names the Memory categories to select for a Memory-only run
(`--eval-categories memory_kv,memory_vector,memory_rec_sum`, a subset, not comparable). When
bfcl runs this track as its agentic part, the setting is reported `infra_required` in its own
entry instead, `settings.web_search` (`status: infra_required`, `categories`,
`prerequisite: GEMINI_API_KEY`, `when: before generation`, `infra_required` with the reason,
`error`), listed in `settings_infra_required`; it is not generated, never scored and never
dropped, and the Memory setting runs (bfcl's own result decides its headline).

During generation the harness child records every search in
`<project root>/web_search_requests.jsonl`: the backend that served it (`gemini:<model>`), the
number of results, and the error when no model served it. A search no model of the cascade
could serve, after the cascade's retries, gets BFCL's own error text, "Failed to retrieve the
search results from server. Please try again later.", and makes the Web Search setting
`infra_required` (`when: during generation`), naming the failures: the web_search scores would
measure the search service, not the model. For this suite's own run that is a harness failure
of the run: `settings` keeps what was measured (the Memory setting's numbers), `accuracy` is
`null` with `accuracy_withheld_reason` naming the Web Search setting, the status is `error` and
`failure_kind` `infra_required` (the run counts as failed, never as the model's outcome). gbench reads the log every 30 s while `bfcl generate`
runs and stops the container as soon as it shows one; one recorded in the last seconds is caught
after generation. If `bfcl generate` runs past `GBENCH_BFCL_TIMEOUT_S` (when set; see Bounds),
the log is read: a failed search makes the Web Search setting `infra_required`.
`bfcl_report.web_search_calls` holds the search count, the count per backend (`by_backend`), the
failures and the answered searches that had no result (`empty_results`).

The Gemini key never enters the container: the `bfcl generate` that runs a web_search category
holds a per-run placeholder and sends its searches to gbench's Gemini relay, which adds the key
(see [Credentials and the execution guard](#credentials-and-the-execution-guard)); no other
harness job uses the relay. gbench masks every credential value of a container's environment in
the search results and errors, in every tool result the harness hands the model, and in the
search log, and every credential of the run in the result and score files and in the
containers' output.

Whenever `bfcl generate` is stopped for the web search (by the log watch, or by the wall clock
with a failed search in the log), the memory entries it had not finished are generated by the
harness' own resume: a second `bfcl generate` of the memory categories without
`--allow-overwrite`, which loads the entries already in the result files and generates only the
others (`_llm_response_generation.collect_test_cases`; `utils.clean_up_memory_prereq_entries`
drops finished prerequisites from `depends_on`), after a line the stop cut off is dropped. It
runs without the Gemini relay, bounded by `GBENCH_BFCL_TIMEOUT_S` on its own when that is set;
`bfcl_report.memory_resumed` records it. Only the memory categories are then evaluated. When
failed searches are found only after `bfcl generate` finished, the memory entries are already
whole and nothing is resumed. To run only the memory categories on purpose, pass
`--eval-categories memory_kv,memory_vector,memory_rec_sum` (a subset, as above).

### Credentials and the execution guard
`bfcl generate` runs the model's calls with `eval()` in the harness process (BFCL's agentic loop,
`BaseHandler.inference_multi_turn_FC`, hands them to
`multi_turn_utils.execute_multi_turn_func_call`), so nothing in that container may hold a key,
and only the calls upstream means to run may run.

- **Execution guard.** Before the CLI dispatches, the harness child of every `bfcl` command puts
  the [`bfcl`](bfcl.md) suite's execution guard (`bfcl._ExecutionGuard`, the one its multi-turn
  part runs under) in place of `eval` in `multi_turn_utils` (`_install_execution_guard`; no
  `bfcl_eval` file is changed). Upstream maps every public method of the entry's involved
  backends (the memory API of a memory category, `WebSearchAPI` for web_search) to
  `<instance>.<method>`, evaluates the call string BFCL's FC decoder renders,
  `name(k=<repr of a JSON value>)`, and tries to keep anything else out with a name blocklist
  (`kill`, `exit`, `popen`, ...). The guard lets exactly those calls through: an instance of the
  entry's backends, a public method, literal arguments (an unknown bare name still raises
  `NameError`, as upstream), with the same result and error text. Any other call string (a
  builtin such as `open` or `exec`, a module global, code in the name or in the arguments) gets
  upstream's own error, "Function call X is not allowed.", with X named as upstream names it. Each
  refusal is recorded in `<project root>/execution_guard_refusals.jsonl`
  (`GBENCH_BFCL_GUARD_LOG`, emptied at the start of each run), counted in
  `bfcl_report.execution_guard_refusals` (the first 20 in
  `bfcl_report.execution_guard_refused_calls`) and makes the result not leaderboard-comparable:
  "N agentic calls refused by the execution guard (upstream would have run them)".
- **The endpoint key.** When the endpoint has a key (`VLLM_API_KEY`, else `OPENAI_API_KEY`;
  unset, `EMPTY`, `dummy`, `not-needed` and gbench's other placeholders are no key), it
  stays in gbench's process. `bfcl generate` reaches the endpoint through gbench's endpoint relay
  on every run, with or without a key (since 2026-10-13; it adds the key only when there is one)
  (`swebench_common.EndpointRelay`), a thread of the gbench process listening on the docker bridge
  gateway (from `docker network inspect bridge`, usually `172.17.0.1`), which adds
  `Authorization: Bearer <key>` and forwards only `/v1/chat/completions`, `/v1/completions` and
  `/v1/models` to the endpoint as given (it may listen on `127.0.0.1` only). A relayed model
  request fails only when it stops making progress: no event of its reply for 300 s
  (`GBENCH_RELAY_IDLE_TIMEOUT_S`) while the server's token counters and KV-cache usage also stand
  still. That is counted as `stalled_no_tokens`, an endpoint failure, and the client gets a 504.
  Only a safety cap far above any reply bounds its total duration. Its floor is the client's
  timeout (see Bounds) plus 60 s, and at least 3600 s (`EndpointRelay.TIMEOUT_S`); it is
  `base.request_timeout_s` of the request's `max_tokens` (or of the served window when it sends
  none) plus 120 s when that is larger. See [relay.md](../relay.md). The container gets the relay's URL as
  `OPENAI_BASE_URL` / `REMOTE_OPENAI_BASE_URL` and the placeholder `EMPTY` as `OPENAI_API_KEY` /
  `REMOTE_OPENAI_API_KEY`; `bfcl evaluate` gets `EMPTY` too, and the other harness jobs get no
  endpoint key variable. Without a key the placeholder `EMPTY` is passed and the relay adds no key.
  The prerequisite check's request to the endpoint's `/models` is made
  from gbench's process, with the key when there is one.
- **The Gemini key.** The web_search `bfcl generate` gets a per-run placeholder
  (`gbench-gemini-relay-` and 48 hex digits) as `GEMINI_API_KEY` / `GEMINI_API_KEYS` and the
  address of gbench's Gemini relay (`gemini_relay.GeminiRelay`, on the same gateway) as
  `GBENCH_GEMINI_RELAY_URL`. `search_tool` sends each grounding request there with the
  placeholder; the relay adds the key (the keys of `GEMINI_API_KEYS`, else `GEMINI_API_KEY`, in
  turn), forwards native `generateContent` calls of the cascade's models (see Env knobs) to the
  Gemini API (the origin of `GEMINI_OPENAI_BASE_URL`), and refuses everything else (HTTP 403).
- `OPENAI_DEFAULT_HEADERS`, which upstream's handler would read, is never passed: headers can
  carry a credential, and gbench's endpoint takes none. A launch whose forwarded variables would
  still carry a credential of gbench's environment (for example a proxy URL with a password)
  stops with an `[INFRA REQUIRED]` error naming the variable.
- **No `.env` in the project root.** Upstream's `bfcl generate` and `bfcl evaluate` load
  `<project root>/.env` into their process with `load_dotenv(..., override=True)`
  (`bfcl_eval/__main__.py`), and upstream's setup tells users to put their API keys there; the
  project root is mounted into both containers. A key from that file would sit in the process
  that runs the model's calls, and an `OPENAI_BASE_URL` from it would replace the relay's URL.
  So a project root that holds a `.env` (a file, a directory or a link) stops the suite with an
  `[INFRA REQUIRED]` error naming it, before anything is generated and again before each
  `bfcl` container starts: move that file out, or point `GBENCH_BFCL_PROJECT_ROOT` at a directory
  used only by gbench (gbench passes everything the harness needs itself). As a second guard, the
  harness child replaces that `load_dotenv` with one that loads nothing, in memory
  (`_disable_dotenv`; no `bfcl_eval` file changes). The file is never read, only named.
- `bfcl_report.credentials` says where each key stays (and that the project root's `.env` is
  never loaded); `bfcl_report.relays` holds, for each `bfcl generate`, the record of each relay
  it used (`endpoint_relay`: address, endpoint, requests relayed and refused, endpoint failures
  in `upstream_errors`; `gemini_relay`: address, calls by status, harness errors, rejected and
  refused requests; never a key). A search the cascade could not serve through the relay is a
  failed search (see Search backend). An endpoint request the relay could not complete reaches
  the harness as the same error it would get without the relay (an inference error, see Bounds)
  and is counted in `upstream_errors`.
- **Endpoint failures are harness errors.** After generation, the endpoint relay's
  `upstream_errors` (the endpoint did not answer, `connection`; a reply that is not 2xx,
  `http_<status>`; a completions reply that is not JSON; an error event in a stream) and its
  `requests_refused` are summed over every `bfcl generate` of the run, as
  `swebench_common.relay_result_reasons` counts them. Any of them makes the result
  `completed_with_errors`, is listed in `endpoint_failures` (`upstream_errors`,
  `requests_refused`) and adds a named reason to `leaderboard_comparable_reason` ("harness errors
  during `bfcl generate`: the served endpoint failed N relayed model request(s) (...)"). A 400 that
  names the context window is not one of them: the relay counts it apart
  (`context_length_rejections`); the entry it ends is an `over_context` inference error. The
  client retries some failures itself, and the relay cannot tell which entry a request belonged
  to, so the count is per run; an entry whose request still failed is an inference error, left
  out of the score as a harness error (see Bounds).

### Bounds
As BFCL's own. Each entry is bounded by the 20-step limit per turn (`MAXIMUM_STEP_LIMIT`,
`constants/default_prompts.py`); an entry whose request fails is recorded as "Error during
inference: ..." and the run goes on (`_llm_response_generation.multi_threaded_inference`), and the
agentic checker scores it wrong. A harness error is never the model's outcome, so gbench reads
those entries back from the result files and leaves each scored one out of the score, as base
does for a request that got no reply: it comes off its category's total (the checker scored it
wrong, so the correct count stays), counts under `failed_requests` (a timeout also under
`timed_out_requests`, a request the server refused for the context window under
`over_context_prompts`; per category in `bfcl_report.inference_errors` as `failed`, `timed_out`
and `over_context`), in `scoring_excluded` and, by kind, in `request_harness_errors`
(`endpoint_failure`, `endpoint_timeout`, `over_context`), and adds "N scored entries whose request
failed (...) left out of the score as harness errors (upstream scores them wrong)" to
`leaderboard_comparable_reason`; any of them makes the result `completed_with_errors`. A category
that ran with no entry left in the score is unscored: absent from `category_accuracy`, listed in
`categories_unscored`, and the summaries that weigh it `null` (never 0; see Headline metric).
When no scored entry is left in the score (each one failed its request, was not finished or ran
against an incomplete memory; see below), nothing measured the model: the score is withheld
(`accuracy`, `accuracy_pooled` and the three summaries `null`, `accuracy_withheld_reason` saying
why, status `failed`, and "score withheld: ..." in `leaderboard_comparable_reason`), never
reported as 0.
The memory prerequisite entries (the `*_prereq` conversations that write a scenario's memory
before its questions run; upstream writes them to `BFCL_v4_<category>_prereq_result.json`) are
not scored, but upstream records a failed request there the same way and goes on, and such an
entry never saves its memory snapshot (`BaseHandler.inference_multi_turn_FC` saves it only after
the whole conversation), so the scenario's questions run against a memory that lacks what it was
to write (`MemoryAPI._prepare_snapshot` loads the last snapshot saved, or starts empty). gbench
counts them apart, as harness errors: `memory_prereq_failed_requests` (per category in
`bfcl_report.prereq_inference_errors`, same `failed` / `timed_out` / `over_context` split, and
`prereq_failed_requests` in the Memory setting), kept out of `failed_requests`, since they are not
scored entries. Upstream makes every scored entry of a scenario depend on all of that scenario's
prerequisites (`utils.process_memory_test_case`), and which question needed the missing
conversation cannot be told, so every scored entry of that scenario in that category (the
scenario is in the entry id, `memory_kv_7-healthcare-3`) is left out of the score as a harness
error (`memory_prereq_failed` in `request_harness_errors` and
`bfcl_report.harness_errors_left_out`): off its category's total, and off its correct count when the score file, which lists every entry
the checker scored wrong, does not list it (a result file and score file that disagree on those
counts stop the suite with an error, never a guessed score). Any of them makes the result and the
Memory setting `completed_with_errors` and adds a named reason to `leaderboard_comparable_reason`
("N memory prerequisite entries failed their request (...); the memory entries of those scenarios
ran against an incomplete memory and are left out of the score (harness errors)").
`bfcl generate` and `bfcl evaluate` have no wall clock. Each
request keeps the OpenAI client's 2 retries (`openai_completion.py _build_client_kwargs` relies on
the SDK's defaults; openai 3.19.2 in the image) with a timeout sized, as the [`bfcl`](bfcl.md)
suite's multi-turn client's, for the longest reply the served context window allows, since no
`max_tokens` is sent by the handler (gbench's relay fills the run's `--max-output-tokens`, see
Sampling; the window-sized timeout is kept, at least as long as one for that value):
`base.request_timeout_s` of the window, or of 262,144 tokens when the server
reports none (`bfcl_v4_agentic.agentic_request_timeout`, passed to the handler as
`GBENCH_BFCL_REQUEST_TIMEOUT_S` and recorded as `bfcl_report.request_timeout_s`). The SDK's own
600 s is written for OpenAI's hosted API; BFCL's handler for a locally served model sends
`timeout=72000` "to avoid timeout errors" (`base_oss_handler.py`), and a 600 s timeout would turn
a long reply from a slow server into an inference error.

`GBENCH_BFCL_TIMEOUT_S` (legacy `BFCL_TIMEOUT_S`), when set, bounds each of the two on its own.
A `bfcl generate` it stops keeps what the harness finished: the harness writes every entry to its
result file as it completes, a line the stop cut off is dropped, and `bfcl evaluate
--partial-eval` scores the entries present. In an `--eval-limit` run the harness writes with
`--run-ids`, which rewrites each result file whole, sorted, after every finished entry
(`BaseHandler.write` with `update_mode`), so a stop during such a rewrite can also drop entries it
had finished; those count as unfinished (never as correct). With a wall clock set, a full run
reads each category's scored entry count from the image before generating, so every entry the
run was to generate is accounted for: each category's `total` is the entries the harness
finished (less those left out above), its `unfinished` the planned ones it did not; those were never measured, so they are
harness errors left out of the score (`unfinished` in `request_harness_errors` and
`bfcl_report.harness_errors_left_out`), counted under `failed_requests` and `timed_out_requests`,
and `harness_errors` records the stop (`part`, `error`, `timeout_s`, `entries_planned`,
`entries_unfinished`). A category none of whose entries finished is unscored. The result is then
`completed_with_errors` and not leaderboard-comparable (`failed`, with the score withheld, when no
entry finished).
A `bfcl evaluate` it stops raises the error (the [`bfcl`](bfcl.md) suite keeps its other parts
and leaves the agentic categories unscored); when `bfcl generate` was stopped before it, the
error names both and carries the generate stop's record (`part`, `error`, `timeout_s`,
`entries_planned`, `entries_unscored`), which the `bfcl` suite keeps in `harness_errors` ahead of
the evaluate stop. A stop whose cause is the web search (see Search backend) is not a harness
error: the Web Search setting is `infra_required` and the memory categories are resumed; a
resume the wall clock stops is recorded like a stopped generate (`part` "`bfcl generate` (the
memory categories, resumed)"). Before an `--eval-limit` run generates, this model's
result files for the chosen categories in that limit's result directory are removed (upstream
removes them itself for a full run with `--allow-overwrite`), so only this run's entries are
scored.

### Thinking (`--thinking`)
The harness builds its own OpenAI requests, so gbench's usual `chat_template_kwargs`
injection never runs here. `--thinking` sets `GBENCH_BFCL_ENABLE_THINKING=1` for the harness
container, and the handler subclass adds
`extra_body={"chat_template_kwargs": {"enable_thinking": true}}` to every call. With
`--thinking` the temperature is 1.0 (see Sampling); the result's `thinking` field records the mode.

### Sampling
gbench's rule for every suite: temperature 1.0 with `--thinking`, 0.0 without. gbench passes the
value to `bfcl generate --temperature`, which the handler sends on every request. BFCL's own
temperature, 0.001 (that flag's default), is recorded under `official_sampling`, and the run's
temperature is named in the comparability reason.

**Token budget.** The handler sends no `max_tokens`. gbench's endpoint relay puts the run's
`--max-output-tokens` on every `bfcl generate` request that carries none when this suite runs on
its own (`bfcl_v4_agentic` is an agent suite, `swe_agentic.AGENT_MAX_TOKENS_SUITES`, the user's
decision of 2026-10-11; the relay reads the run's `suite` knob). The result records it in
`max_output_tokens_note`, `official_sampling.agent_max_tokens_from_run` (`{max_tokens_sent, source,
sent_via, upstream, requests_filled}`) and each relay record's `max_tokens_from_run`
(`bfcl_report.relays`), and names it in `leaderboard_comparable_reason` (`max_tokens=<N> (the run's
--max-output-tokens) sent with <n> agent request(s) whose upstream harness sends none`, the eval
runner's own words, `evals._stamp_agent_max_tokens`, so it is stated once). The cap is recorded
only when a relay filled at least one request (with `GBENCH_BFCL_REUSE_GENERATIONS=1` and complete
result files `bfcl generate` sends nothing, and the note says `not sent`). `gbench --merge` unites
the shards' `bfcl_report.relays` (each entry with its `shard`), sums the fill over them and takes
the other fields from the first shard that holds an entry, not an empty one.
`GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0` turns it off (no `max_tokens`, as upstream).
When the [`bfcl`](bfcl.md) suite runs this track as its agentic part (run knob `bfcl`, not an
agent suite; its single-turn and multi-turn parts send no `max_tokens` either), nothing is
filled, as upstream. Until 2026-10-13 the relay started only with a real endpoint key, so on a
keyless endpoint no request was capped: smoke run `smoke-10-12-235114` tapped 83
`bfcl_v4_agentic` requests, none with `max_tokens` (a live check that day through the suite's
keyless relay: run value 48 gave `finish_reason` length at 48 completion tokens; under knob `bfcl`
the same relay sent none). `--temperature`, or `GBENCH_BFCL_V4_AGENTIC_TEMPERATURE` for this suite alone (it
takes precedence), overrides it. When the [`bfcl`](bfcl.md) suite runs this track as its agentic
part it hands over its own temperature (the same rule, or `GBENCH_BFCL_TEMPERATURE` /
`--temperature`),
which is used as is; a differing `GBENCH_BFCL_V4_AGENTIC_TEMPERATURE` is then ignored with a
warning. `official_sampling` on the result records BFCL's setting, the temperature used, its
source and whether it matched.

## Headline metric and how to read it
`accuracy` is BFCL's Agentic score, computed as in `bfcl_eval/eval_checker/eval_runner_helper.py`:

    summary_web_search       = mean(web_search_base, web_search_no_snippet)
    summary_memory           = mean(memory_kv, memory_vector, memory_rec_sum)
    overall_accuracy_agentic = mean(summary_web_search, summary_memory)

All three are on the result. The means are unweighted: memory has 465 scored questions to web
search's 200, so pooling every question (`accuracy_pooled`, kept as a secondary number) would
weigh memory 70% instead of 50%. `category_accuracy` has each category's correct / total /
accuracy. The means use each category's exact correct / total, as BFCL averages the unrounded
accuracies, and are rounded once, at the end; the 2-decimal per-category `accuracy` is for
display. A group is `null` unless every one of its categories ran and is in the score, and
`overall_accuracy_agentic` is `null` unless both groups are, as the bfcl suite reports its
columns: a `--eval-categories` subset or an `--eval-limit` draw that left a category out has no
Agentic score, and its `accuracy` is `accuracy_pooled`, the pooled rate over the entries in the
score (the `metric` field names the categories not in the score). BFCL itself counts a missing
category as 0 (and displays N/A); that value is kept as
`overall_accuracy_agentic_bfcl_missing_as_zero`. When the Web Search setting is
`infra_required`, `summary_web_search` and `overall_accuracy_agentic` are `null`; for this
suite's own run (a search outage during generation) `accuracy` is then `null`, withheld with the
reason (the `metric` says it is withheld), the status `error`; when bfcl runs the track as its agentic part, `accuracy` is
`summary_memory` (the `metric` field says so) and bfcl decides its own headline. Each category's correct / total is over the
entries in the score (harness errors left out, see Bounds; `effective_n` is their sum). A
category every entry of which was a harness error is unscored (`categories_unscored`): the
summary of its group and `overall_accuracy_agentic` are `null`, never a mean over the rest, and
`accuracy` is then `accuracy_pooled`, the pooled rate over the entries in the score (the `metric`
field names the unscored categories). `total_questions` counts the scored entries the run was to
score, in the score or left out of it (`scoring_excluded`, by kind in `request_harness_errors`).
`settings` has each selected setting: its categories, `categories_unscored`, summary, counts
(`total_questions` with its entries left out, `correct_answers`, `scoring_excluded`),
`failed_requests` (the Memory setting also `prereq_failed_requests`) and status
(`completed_with_errors` when one of its entries failed its request, was not finished or was left
out of the score, for Memory also when one of its prerequisite entries failed its request, or
when a harness error or an endpoint failure touched the run), or the Web Search setting's
`infra_required` entry. When no scored entry is left in the score, `accuracy` and the summaries
are `null` (see Bounds).

`leaderboard_comparable` is false for every run that scores the Web Search setting: its web
search is served by Gemini Google Search grounding, not BFCL's DuckDuckGo via SerpAPI (see Search
backend). A run of the memory categories alone is a category subset, so no run of this suite is
leaderboard-comparable. `leaderboard_comparable_reason` lists each deviation: the search substitution ("web
search served by Gemini Google Search grounding (BFCL queries DuckDuckGo via SerpAPI)"), an
`infra_required` Web Search setting (its Gemini key missing or rejected, or a search no grounding
model served), a category subset, an `--eval-limit` subset, re-scored cached generations, another
temperature, a `bfcl generate` stopped by `GBENCH_BFCL_TIMEOUT_S` (see Bounds), endpoint
failures gbench's endpoint relay counted (`endpoint_failures`), memory prerequisite entries whose
request failed (`memory_prereq_failed_requests`), scored entries whose request failed, left out of
the score, categories left unscored, a withheld score, calls the execution guard
refused, or every scored question failing with `agentic:no_last_message` (the harness found no final reply to grade
anywhere, which is what a server that sends an empty `tool_calls` list with a plain reply
produces). `bfcl_report` records the categories run, the search backend (`search_backend`), the
substitution (`search_substitution`), the search counts (`web_search_calls`), the entries whose
request failed (`inference_errors`) and the memory prerequisite entries whose request failed
(`prereq_inference_errors`), the entries left out of the score as harness errors per category and
kind (`harness_errors_left_out`) and the categories left unscored (`categories_unscored`), the
harness release
and image, the per-request timeout (`request_timeout_s`), the execution-guard refusals
(`execution_guard_refusals`, `execution_guard_refused_calls`), where the keys stay
(`credentials`) and the relays' records (`relays`),
`error_types` (how often each BFCL error type occurs among the failed questions in the score;
an entry left out as a harness error is not counted), whether generations were fresh, the
id-file limit, the selected categories the id file gave no entry (`categories_not_drawn`: they
are not in `categories_run`, `settings.*.categories` or `web_search_categories_run`) and the score
directory. `sample_traces` holds one trace per scored entry the harness generated, correct ones
included: `is_correct` is whether the category's score file (which lists only the entries its
checker scored wrong) leaves it out, with that row's error type, message and expected answers for
a wrong one, and the model's answer (clipped to 4,000 characters); harness errors are marked
`scoring_excluded` with their `harness_error`.

## Containers and network
Every container is started from the local harness image (`--pull never`, every capability
dropped, `no-new-privileges`):

| Container | What runs in it | Network |
| --- | --- | --- |
| image probe (`image_probe`) | the release, category and pinned-file checks | `--network none` |
| `--eval-limit` id selection, entry counts (`_select_limited_ids`, `scored_entry_counts`) | reads the release's data | `--network none` |
| `bfcl generate` (and its resume of the memory categories) | BFCL's inference loop, which executes the model's calls with `eval()` in the harness process under gbench's execution guard, talking to the model endpoint (through gbench's endpoint relay, on every run) and, for the web_search categories, gbench's Gemini relay | docker's bridge network, relies on the host rule |
| `bfcl evaluate` | BFCL's checker over the result files | `--network none` |

The host network is never used. On a cloud VM the `bfcl generate` container relies on the host
rule that keeps a bridge network away from the metadata server at `169.254.169.254` (which hands
out the VM's service-account token):

```bash
sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save   # keep it
```

Before anything is generated (`check_bfcl_prerequisites`), gbench's guard
(`require_metadata_blocked`, see [toolchains](toolchains.md#cloud-vms-the-metadata-server)) starts
one short-lived container of the image on the bridge network that opens and closes a TCP connection
to `169.254.169.254:80` (no request is sent); if it opens, the suite stops with an
`[INFRA REQUIRED]` error naming the rule, and a probe that gives no clear answer (neither `python3`
nor `bash` in the image, a docker error or timeout) is a hard error too. Before a web_search
category is generated, the guard's probe container also resolves
`vertexaisearch.cloud.google.com`, the host of the grounding links `fetch_url_content` follows (or
the proxy's host when the container fetches through a proxy), and a name that does not resolve is
a hard error: fetches would fail on the harness's side. gbench never runs sudo or changes the host's firewall itself.

Since a relay always runs for `bfcl generate` (the endpoint relay, keyless too since 2026-10-13,
and the Gemini relay for the web_search categories), gbench also checks, before anything is
generated, that a container of the image on the bridge network can open a TCP
connection to a listener of its own on the bridge gateway, where the relays listen, and stops with
an `[INFRA REQUIRED]` error otherwise (a host firewall must let the docker bridge reach that
address). The relays listen there only while their `bfcl generate` runs.

`bfcl generate` gets the endpoint relay's URL as `OPENAI_BASE_URL` / `REMOTE_OPENAI_BASE_URL`;
the relay uses your endpoint URL as given, from this host (a server bound to `127.0.0.1` only is
fine), so the container never connects to the endpoint itself and its reachability from the bridge
network is not checked (until 2026-10-13 a keyless endpoint was reached directly, at
`host.docker.internal` for one on this host). A proxy variable
(`HTTP_PROXY`, `HTTPS_PROXY` and their lowercase forms) on a loopback address is rewritten the same
way, and `host.docker.internal` (and, while a relay runs, the gateway address) is added to
`NO_PROXY` / `no_proxy` when a proxy is set.

When the [`bfcl`](bfcl.md) suite runs this track as its agentic part, these checks and the
prerequisite check run as that suite's: an `[INFRA REQUIRED]` error names `bfcl` and points to
`docs/evals/bfcl.md`, and so do the run's other errors (a failed `bfcl generate` or
`bfcl evaluate`, the `--eval-limit` id selection, a wall clock stop, the Web Search setting's
entry).

## Requirements
- **Docker and the local harness image `gbench-bfcl:2026.3.23`** (the `bfcl` suite's; override
  the tag with `GBENCH_BFCL_IMAGE`), built once from the gbench repository root:
  ```bash
  docker build -t gbench-bfcl:2026.3.23 -f gbench/docker/bfcl.Dockerfile gbench/docker
  ```
  It holds the PyPI release `bfcl-eval==2026.3.23` (*not* the unrelated `bfcl` package; the
  wheel's sha256 is checked) with its own pins on Python 3.10, the rest of its dependencies at
  the versions in `gbench/docker/bfcl.constraints.txt`, `soundfile` (which the release's
  `qwen_agent` import needs), and `memory_vector`'s encoder
  `sentence-transformers/all-MiniLM-L6-v2` at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
  (loaded offline on the CPU, as upstream loads it). See [`bfcl`](bfcl.md#requirements). No
  Docker, no image, an image built for another release, or a harness file that differs from the
  pinned commit is a hard error before anything is generated.
- **On a cloud VM, the host rule** `sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP`,
  made persistent, and an endpoint the docker bridge reaches, or, while a relay runs, a bridge
  gateway the docker bridge reaches (see [Containers and network](#containers-and-network)).
- **A reachable model endpoint** that accepts tool calls (vLLM:
  `--enable-auto-tool-choice --tool-call-parser <parser>`) and leaves `tool_calls` out of a reply
  that makes no call (vLLM does since PR #44105, June 2026). With an older vLLM, which sends
  `tool_calls: []`, the harness reads such a reply as an empty call list, never sees its text,
  and every question fails with `agentic:no_last_message`; the result flags that case.
- **A writable project root** (`GBENCH_BFCL_PROJECT_ROOT`, default `$GBENCH_CACHE_DIR/bfcl`, i.e.
  `~/.cache/gbench/bfcl`), mounted into the image at the same path; results and scores are
  written there, never next to a Python package. It must hold **no `.env`** (upstream's setup
  puts API keys there, and its commands load it into the process that runs the model's calls; see
  [Credentials and the execution guard](#credentials-and-the-execution-guard)): use a directory
  only gbench uses, not an upstream BFCL checkout.
- **`GEMINI_API_KEY`** (or `GEMINI_API_KEYS`, which wins when both are set; a comma or semicolon
  separated list is used round-robin) for the two web_search categories, with Google Search
  grounding quota for the whole run on the cascade's models, network access to
  `generativelanguage.googleapis.com` from the host (the key check, and the searches, which
  gbench's Gemini relay forwards from the gbench process), and name resolution of
  `vertexaisearch.cloud.google.com` from docker's bridge network (the grounding links
  `fetch_url_content` follows from the `bfcl generate` container). A web_search question may
  search several times. Without a
  key, with one the Gemini API rejects, or when a search no grounding model served reaches the
  model during the run, the Web Search setting is reported `infra_required` and the Memory
  setting is still scored. The `google-search-results` package upstream's `web_search.py` imports
  comes with `bfcl-eval` in the image; gbench never calls it.
- Any other missing prerequisite hard-errors (`infra_required`) with the reason, as does a missing
  key when only web_search categories are selected; the suite never skips and never reports a
  fabricated score. The other partial result is a `bfcl generate` stopped by
  `GBENCH_BFCL_TIMEOUT_S`, which the operator sets: its unfinished entries are harness errors left
  out of the score and the result says so (`harness_errors`, not leaderboard-comparable).

## Env knobs
- `GBENCH_BFCL_PROJECT_ROOT` (legacy `BFCL_PROJECT_ROOT`): harness project root, default
  `$GBENCH_CACHE_DIR/bfcl`; a `.env` in it stops the suite (see Requirements).
- `GBENCH_BFCL_TIMEOUT_S` (legacy `BFCL_TIMEOUT_S`): wall clock in seconds for `bfcl generate`
  and for `bfcl evaluate`, each on its own; unset (the default), empty or 0 means none, as
  upstream (see Bounds).
- `GBENCH_BFCL_IMAGE`: the harness image tag, default `gbench-bfcl:2026.3.23` (its labels must
  match the release).
- `GBENCH_BFCL_REUSE_GENERATIONS=1` (legacy `BFCL_REUSE_GENERATIONS`): re-score existing
  generations instead of regenerating (flagged on the result).
- `GBENCH_BFCL_V4_AGENTIC_TEMPERATURE`: sampling temperature for this suite (overrides
  `--temperature`; default 1.0 with `--thinking`, 0.0 without). Not used when `bfcl` runs the
  track.
- `GBENCH_BFCL_ENABLE_THINKING` is set by gbench from `--thinking` for the harness container; do
  not set it by hand.
- `GBENCH_SEARCH_MODELS` (comma list; legacy single `GBENCH_SEARCH_MODEL`),
  `GBENCH_SEARCH_MAX_RPS` (default 8), `GBENCH_SEARCH_MAX_BURST`, `GBENCH_SEARCH_CASCADE_ROUNDS`
  (default 3), `GBENCH_SEARCH_ATTEMPTS` (default 3) and `GBENCH_SEARCH_BACKOFF` (default 1.0 s):
  the grounding cascade of gbench's `search_tool` (the model order, the global request-rate
  limit, whole-cascade retries, per-model attempts for transient errors and the backoff), passed
  to the `bfcl generate` container that runs a web_search category. Unset, the models are
  gbench's one Gemini cascade (`GBENCH_JUDGE_MODELS`, else `GBENCH_JUDGE_MODEL`, else
  `base.GEMINI_CASCADE`, listed under Search backend); gbench sets `GBENCH_SEARCH_MODELS` and
  `GBENCH_JUDGE_MODELS` in that container to the resolved lists itself. `GBENCH_SEARCH_MAX_RESULTS`
  does not apply here: the model's `max_results` cuts the results, as upstream.
- `GBENCH_BFCL_SEARCH_LOG` is set by gbench for the `bfcl generate` container that runs a
  web_search category (the web search log); do not set it by hand.
- `GBENCH_BFCL_REQUEST_TIMEOUT_S` is set by gbench for the harness container (each request's
  timeout, see Bounds); do not set it by hand.
- `GBENCH_BFCL_GUARD_LOG` (the execution-guard log) and `GBENCH_GEMINI_RELAY_URL` (the Gemini
  relay's address, web_search `bfcl generate` only) are set by gbench for the harness container;
  do not set them by hand.
- `VLLM_API_KEY` (else `OPENAI_API_KEY`): the endpoint's key. It is added by gbench's endpoint
  relay and never enters a container; without one the containers get the placeholder `EMPTY`
  (see [Credentials and the execution guard](#credentials-and-the-execution-guard)).
  `OPENAI_DEFAULT_HEADERS` is not passed.
- `GBENCH_AGENT_MAX_TOKENS_FROM_RUN` (default 1): `0` stops gbench's endpoint relay from putting
  the run's `--max-output-tokens` on the `bfcl generate` requests of this suite (see *Token
  budget* under Sampling).

## Run
```bash
# full agentic track (web search served by Gemini Google Search grounding: not
# leaderboard-comparable, the reason recorded on the result)
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-26B-A4B-it --evals bfcl_v4_agentic --sandboxes 16 \
       --max-output-tokens 65536

# memory categories only (no Gemini key needed; recorded as a subset). A full run without a
# working key also scores the memory categories, with the Web Search setting infra_required.
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-26B-A4B-it --evals bfcl_v4_agentic \
       --eval-categories memory_kv,memory_vector,memory_rec_sum --max-output-tokens 65536
```
`--max-output-tokens` is required for every eval run. The harness builds its own requests and
sends no `max_tokens`; gbench's endpoint relay puts the run's `--max-output-tokens` on each of
them (see *Token budget* under Sampling).

> **Runtime:** each item is a multi-turn agentic episode (~30 s or more per item on a 26B
> model), so a full category takes tens of minutes.
