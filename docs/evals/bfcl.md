# bfcl

The Berkeley Function Calling Leaderboard (BFCL) v4 in full: every category BFCL scores, run with
BFCL's own protocol, and the leaderboard's headline, **BFCL Overall Acc**. The suite measures
whether the model emits the right function calls with the right arguments (single-turn), whether
it abstains when no function fits, whether it can carry out multi-step, multi-turn tasks against
stateful tool backends (multi-turn), and whether it can use memory and web-search tools as an
agent (agentic).

## Dataset
All 22 of `bfcl_eval`'s `ALL_SCORING_CATEGORIES` at gorilla commit
[`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval)
(`fc_common.BFCL_EVAL_COMMIT`, the current upstream; PyPI `bfcl-eval==2026.3.23` is byte-identical
to its `bfcl_eval/` tree, all 183 files), 5,106 entries:

| Part | Category | Entries |
|---|---|---|
| Non-Live | `simple_python`, `simple_java`, `simple_javascript` | 400, 100, 50 |
| Non-Live | `multiple`, `parallel`, `parallel_multiple` | 200, 200, 200 |
| Non-Live | `irrelevance` | 240 |
| Live | `live_simple`, `live_multiple`, `live_parallel`, `live_parallel_multiple` | 258, 1,053, 16, 24 |
| Live | `live_irrelevance`, `live_relevance` | 884, 16 |
| Multi-Turn | `multi_turn_base`, `multi_turn_miss_func`, `multi_turn_miss_param`, `multi_turn_long_context` | 200 each |
| Agentic | `memory_kv`, `memory_vector`, `memory_rec_sum` | 155 each |
| Agentic | `web_search_base`, `web_search_no_snippet` | 100 each |

- **Single-turn** (13 categories, 3,641 entries): the `BFCL_v4_*.json` files and their
  `possible_answer/` files, downloaded from the pinned commit into
  `$GBENCH_CACHE_DIR/bfcl_data/<commit>/` and checked against their git blob ids
  (`bfcl._PINNED_BLOBS`) before they are read. The Hugging Face mirror
  `gorilla-llm/Berkeley-Function-Calling-Leaderboard` is not used: it stopped at a February 2025
  v3 snapshot that predates several ground-truth and question fixes.
- **Multi-turn** (4 categories, 800 entries): read by `bfcl_eval` itself from the package in the
  harness image (`load_dataset_entry` / `load_ground_truth_entry`), together with the function
  docs of the 8 tool backends (`multi_turn_func_doc/`). Before anything runs, every package file
  the multi-turn path reads or executes (38 files: the 8 data files, the 8 function-doc files, the
  backend sources, the inference loop, the checker, the tool conversion and the loaders:
  `bfcl._MT_PINNED_BLOBS`) is checked against its git blob id at the pinned commit.
- **Agentic** (5 categories, 665 scored questions): run by the
  [`bfcl_v4_agentic`](bfcl_v4_agentic.md) suite, which drives the same `bfcl-eval` harness; see
  its doc for the data and the memory prerequisite entries.

The categories BFCL no longer scores (`sql`, `exec_*`, `rest`, `chatable`,
`multi_turn_composite`) are not run. `format_sensitivity` is not a scoring category and BFCL
skips it for function-calling models, so its leaderboard columns are N/A here as they are for
every FC entry. [`bfcl_live`](bfcl_live.md) runs the Live categories alone.

## Protocol
BFCL's FC mode for an OpenAI chat-completions model (`OpenAICompletionsHandler`), in three parts.
Everything that runs `bfcl_eval` (the single-turn checker, the multi-turn loop, backends and
checker, and the agentic harness) runs in the local harness image `gbench-bfcl:2026.3.23`, built
from `gbench/docker/bfcl.Dockerfile` (see Requirements); gbench's own process never imports
`bfcl_eval`. Each container gets the host user's uid/gid and gbench's `eval_suites` directory
read-only, and the job and its result travel through a per-run exchange directory, never the
command line. Only the containers that talk to the model endpoint have a network, and never the
host network (see [Containers and network](#containers-and-network)).

**Single-turn.**
- **Messages:** the entry's turn-0 messages, verbatim. Live entries that carry a system message
  keep it; gbench adds no system prompt of its own.
- **Tools:** the entry's function docs, prepared exactly as upstream does it:
  `_func_doc_language_specific_pre_processing` appends "Note that the provided function is in
  Python 3 syntax." (Java 8 SDK / JavaScript for those categories) to every description and, for
  Java and JavaScript, re-types every parameter as a string that must hold the value in source
  syntax (`"5L"`, `"new int[]{1, 2}"`, `"[1, 2]"`). `convert_to_tool` then maps the Gorilla types
  to JSON-schema types (`dict` to `object`, `float` to `number`, ...) and rewrites dots in function
  names to underscores. No `tool_choice` is set.
- **Request:** the messages, the model and the tools, at the run's temperature (below), and no
  `max_tokens`, as `OpenAICompletionsHandler._query_FC` sends them.
- **Scoring** (`bfcl.single_turn_verdicts`, run in the harness image): each reply becomes the
  model result `_parse_query_response_FC` records (its tool calls as `[{name: arguments}]`, else
  its text), and `bfcl_eval`'s own `eval_runner` checks it: `_evaluate_single_ast_entry` (with the
  category's language and return format, as `ast_file_runner` sets them) or
  `_evaluate_single_relevance_entry`, against the entry's original function docs and possible
  answer, for an OpenAI chat-completions FC model (`underscore_to_dot`, as upstream's entry for
  that handler). The structured `tool_calls` are decoded as BFCL's FC `decode_ast` does
  (`arguments` must be valid JSON). Reply text is never a call: a reply without tool calls is an
  empty call list when its text is empty and a decode failure otherwise, as in BFCL's FC handler.
  Before anything is generated, the 13 package files the checker imports or runs
  (`bfcl._ST_PINNED_BLOBS`) are checked against their git blob ids. For `irrelevance` /
  `live_irrelevance` the reply is correct iff it makes no call (an empty reply, a refusal or an
  undecodable call is no call); for `live_relevance` it is correct iff it makes one. Every other
  category runs the AST checker against the original function docs:
  - the number of calls must equal the ground truth's (exactly 1 for simple and multiple);
    parallel calls may come in any order;
  - the function name must match the advertised (underscored) name;
  - every required parameter present, no parameter outside the docs or the possible answer,
    parameter names matched exactly;
  - Python values must have the declared type (an int is accepted for a float; a string `"5"` for
    an integer is not); Java/JavaScript values must be strings and are converted by BFCL's type
    converters before comparison;
  - strings, and strings inside lists and dicts, are compared after `standardize_string`
    (case, spaces and `,./-_*^` ignored);
  - an optional parameter (`""` among its accepted values) may be omitted, but a supplied value
    must still be one of the accepted ones.

  The request half of gbench's port of the pipeline (`fc_common`: the language hint and
  `convert_to_tool`) builds the tools; `fc_common.bfcl_score`, the ported checker, is kept as a
  reference and gives the same validity and error type as the image's `eval_runner` on 34,104
  reply variants over all 3,641 entries (gold, text, empty, a JSON call as text, wrong name, no
  arguments, malformed arguments, duplicated, perturbed and reordered calls).

**Multi-turn.** Run inside `bfcl-eval` itself, in the harness image, which reaches the model
endpoint from docker's bridge network (no key enters the container: a real `OPENAI_API_KEY` is
added by gbench's endpoint relay, see [Containers and network](#containers-and-network)): each
entry goes through `OpenAICompletionsHandler.inference_multi_turn_FC`, BFCL's own loop, and the
traces come back to gbench.
- The tools are the involved backends' function docs with the Python hint, compiled by
  `convert_to_tool`. Each turn's user message is sent; while the reply carries tool calls, they are
  rendered by `convert_to_function_call`, executed in `bfcl_eval`'s executable backends
  (`execute_multi_turn_func_call`: the file system, messaging, posting, ticket, trading, travel,
  vehicle and math APIs, loaded with the entry's initial state, or its long-context state for
  `multi_turn_long_context`) and returned as `tool` messages. A reply without tool calls (empty,
  prose or undecodable) ends the turn.
- `multi_turn_miss_func`: a held-out function is added to the tools on its turn, with BFCL's user
  message "I have updated some more functions you can choose from. What about now?".
- More than 20 steps in one turn (`MAXIMUM_STEP_LIMIT`) force-terminates the entry.
- **Scoring:** `eval_runner._evaluate_single_multi_turn_entry` with `multi_turn_checker`, called
  directly. The model's calls and the ground-truth calls are replayed turn by turn in fresh
  backend instances; after every turn that has ground truth, all public backend state must equal
  the ground truth's (`state_checker`) and every ground-truth execution result must appear among
  the model's results so far, in any order (`response_checker`). A turn with ground truth but no
  model call fails the entry; turns whose ground truth is empty (`miss_func`, `miss_param`) are
  executed but not checked. A force-terminated entry fails. Where the checker raises, upstream's
  evaluation aborts; gbench marks that entry `multi_turn:checker_exception` and leaves it out of
  the score as a harness error (`checker_exception`, see Reading the result), never the model's
  wrong answer, and the run goes on.
- The whole path was checked in the image with a local server that answers every turn with its
  ground-truth calls: every entry scores correct through the real loop, backends and checker.

**Agentic.** Delegated to [`bfcl_v4_agentic`](bfcl_v4_agentic.md): `bfcl generate` + `bfcl
evaluate` in the harness image, with the per-category scores read back into this result. The
agentic part generates at this suite's temperature (below), whatever
`GBENCH_BFCL_V4_AGENTIC_TEMPERATURE` says (a differing value is ignored with a warning). The
web-search categories are served by Gemini Google Search grounding in place of BFCL's DuckDuckGo
via SerpAPI search (the tool contract the model sees stays BFCL's; see
[`bfcl_v4_agentic`](bfcl_v4_agentic.md#search-backend)), so a run that scores them is not
leaderboard-comparable: the agentic part's reason, "web search served by Gemini Google Search
grounding (BFCL queries DuckDuckGo via SerpAPI)", is carried into this result's
`leaderboard_comparable_reason`. BFCL V4 scores them as their own setting (the leaderboard's Web
Search column); single-turn, multi-turn and memory do not search. Before generating anything
this suite checks the Gemini key (`GEMINI_API_KEYS`, else `GEMINI_API_KEY`) with the Gemini API,
the judged suites' gate. When no key is set, or the Gemini API rejects it, the Web Search setting
is reported `infra_required` in its own entry (`web_search_setting`, listed in
`settings_infra_required`, naming the reason) and is never scored or dropped, while every other
selected category runs: Overall Acc and the Agentic column, which weigh web search, stay `null`;
on a run of every category `accuracy` is withheld (`null`, with `accuracy_withheld_reason` naming
the setting and its prerequisite), since the missing setting is a harness error and a pooled
figure over the rest would read as the headline; `pooled_accuracy` and `bfcl_summary` keep the
scored columns; the status is `completed_with_errors` and the run is not leaderboard-comparable.
With an explicit `--eval-categories` subset `accuracy` is `pooled_accuracy`. Only a run whose selected categories are all web_search has nothing
else to run and hard-errors (`infra_required`) before generating. The suite also hard-errors
before generating when the image's harness differs from the pinned commit in a file the agentic
track uses. During generation every search is recorded; a search no grounding model served,
which the model got as BFCL's own error text, makes the Web Search setting `infra_required` the
same way: the log is read every 30 s while `bfcl generate` runs, and the container is stopped as
soon as it shows one (one in the last seconds is caught after generation). When generate is
stopped for the web search, the memory entries it had not finished are generated by the
harness' own resume and scored, and the single-turn and multi-turn results stand; only the
web_search categories go unscored. That generate holds only a placeholder for the Gemini key,
which gbench's Gemini relay adds outside the container (see
[`bfcl_v4_agentic`](bfcl_v4_agentic.md#credentials-and-the-execution-guard)); the key reaches
neither the model, nor a result file, nor a log line. `bfcl generate` runs the model's calls
under the same execution guard as the multi-turn part (below).

**Request settings (single-turn and multi-turn).**
- **Sampling:** gbench's rule for every suite, for all three parts: temperature 1.0 with
  `--thinking`, 0.0 without. BFCL's own temperature, 0.001 (the default of `bfcl generate
  --temperature` in `bfcl_eval/__main__.py`, which `OpenAICompletionsHandler._query_FC` sends on
  every request), is recorded under `official_sampling`, and the run's temperature is named in the
  comparability reason. Override it with
  `--temperature`, or
  `GBENCH_BFCL_TEMPERATURE` for this suite alone (it takes precedence); the agentic part then
  generates at that value too. `--thinking` sets `chat_template_kwargs.enable_thinking`; only the
  final reply and its tool calls are scored.
- **Budget:** no `max_tokens`, in any part, as BFCL's handler sends none: the server grants the
  rest of its context window. `--max-output-tokens` is not applied (`max_output_tokens_note`
  says so); a reply cut off at the end of the context window is counted in
  `truncated_responses`, as it would be cut off in BFCL's own run. The agentic part's
  `bfcl generate` goes through gbench's endpoint relay on every run since 2026-10-13 (keyless
  too), but under this suite (run knob `bfcl`, not one of `swe_agentic.AGENT_MAX_TOKENS_SUITES`)
  the relay fills in no `max_tokens`; only the standalone
  [`bfcl_v4_agentic`](bfcl_v4_agentic.md#sampling) suite gets the run's `--max-output-tokens`
  there.
- **Execution guard:** `execute_multi_turn_func_call` runs each call string with Python's `eval`.
  gbench keeps that and only refuses a call that would reach something other than a backend
  method: a Python builtin (`open`, `exec`, ...), a module global, or code inside the arguments.
  Such a call returns BFCL's own error text, "Function call X is not allowed.", with X named as
  upstream names it (the bare function or method name). Every call a model can make through the
  tools it was given evaluates exactly as upstream (same result, same error text);
  `multi_turn_report.execution_guard_refusals` counts refusals. The agentic part's
  `bfcl generate` (the memory and web_search backends) runs under the same guard, installed in
  its harness child before the CLI dispatches; its refusals are in
  `agentic_report.execution_guard_refusals` and in the agentic part's reason.
- **Bounds**, as BFCL's own: each request by its client's timeout, each multi-turn entry by the
  20-step limit per turn, and an entry whose request fails is recorded as an inference error while
  the run goes on (`_llm_response_generation.multi_threaded_inference`). BFCL's checker scores such
  an entry wrong; gbench leaves it out of the score as a harness error, in every part alike (see
  Reading the result). No part is bounded as a whole: `bfcl generate` has no wall clock. The
  multi-turn client's timeout is sized for the longest answer the served context window allows
  (`base.request_timeout_s`; BFCL's own handler for a locally served model, `base_oss_handler.py`,
  sends `timeout=72000` "to avoid timeout errors"); the agentic part's requests get the same
  timeout (see [`bfcl_v4_agentic`](bfcl_v4_agentic.md#bounds)). When the server reports no context
  window, both are sized for 262,144 tokens (`base.UNCAPPED_TIMEOUT_TOKENS`, base's rule for a
  request that carries no `max_tokens`). `GBENCH_BFCL_TIMEOUT_S` (legacy `BFCL_TIMEOUT_S`), when
  set, bounds each harness part on its own: the single-turn checker, the multi-turn part, `bfcl
  generate` and `bfcl evaluate`. A part it stops is recorded under `harness_errors` and the rest of
  the run is kept (see Reading the result).
- **Backend state:** each attempt of each entry has its own backend instances (named after the
  run, the attempt and the entry), and an entry's cleanup removes only those, so attempts that
  run at the same time never share or wipe each other's state, whatever `--attempt-count` is.

## Reading the result
- `accuracy` (headline): **BFCL Overall Acc**, computed as `eval_runner_helper.py` does:
  10% Non-Live AST (the unweighted mean of Simple AST, itself the mean of Python, Java and
  JavaScript simple, Multiple, Parallel and Parallel Multiple), 10% Live AST (the four live AST
  categories weighted by entry count), 10% Irrelevance (the mean of `irrelevance` and
  `live_irrelevance`), 30% Multi-Turn (the mean of the four multi-turn categories) and 40% Agentic
  (the mean of the web-search mean and the memory mean). `live_relevance` is reported but not
  weighed. Compare it with the leaderboard's Overall Acc (FC entries).
- `bfcl_summary`: BFCL's score tables in percent, rounded as its CSVs are:
  - `overall`: the accuracy columns of `data_overall.csv` (Overall Acc, Non-Live AST Acc and its
    parts, Live Acc and its parts, Multi Turn Acc and its parts, Web Search Acc, Memory Acc and
    their parts, Relevance Detection, Irrelevance Detection);
  - `non_live`, `live`, `multi_turn`, `agentic`: the rows of `data_non_live.csv`,
    `data_live.csv`, `data_multi_turn.csv` and `data_agentic.csv`.

  Every column was checked against `generate_leaderboard_csv` on random per-category scores.
  A column whose categories did not all run is `null` (BFCL shows N/A there).
- When BFCL Overall Acc is `null` because the run did not cover every category (a category
  subset, an `--eval-limit` too small to reach every category, a shard other than 1),
  `accuracy` is `pooled_accuracy` instead (`null`, never 0, when nothing was scored); the
  `metric` field says which of the two it is. On a run of every category, a category left
  unscored because every one of its entries was a harness error, or a Web Search setting reported
  `infra_required` (above), withholds `accuracy` instead (`null` with `accuracy_withheld_reason`
  naming the category): Overall Acc weighs it, and a pooled figure over the rest would read as
  the headline.
- `pooled_accuracy`: plain correct/total over every entry in the score (harness errors left out).
  `category_accuracy`: per category, over the entries in the score; `effective_n`: how many
  entries that is in all (the summary's Effective N).
- `multi_turn_report`: the harness release and image, entries, attempts, the budget (no
  `max_tokens`), the per-request timeout (`request_timeout_s`), model requests, force-terminated
  entries, execution-guard refusals, `endpoint_relay` (the record of gbench's endpoint relay:
  address, endpoint, requests relayed and refused, endpoint failures in `upstream_errors`; `null`
  when the endpoint has no real key (`VLLM_API_KEY`, else `OPENAI_API_KEY`)), `endpoint_failures` (what of that record
  counts as harness errors, or `null`) and `harness_errors_left_out` (per category, the entries
  left out of the score as harness errors, by kind).
- `single_turn_checker`: the release and image whose `eval_runner` scored the single-turn replies.
- `max_output_tokens_note`: that no request carried `max_tokens` (and that `--max-output-tokens`
  was not applied).
- `agentic_report`: the delegated suite's comparability flag and reason, its `settings` (Memory
  and Web Search, each with its categories, summary and counts, or its `infra_required` entry),
  search backend (`search_backend`, `search_substitution`, the search counts in
  `web_search_calls`), temperature, harness release, BFCL error-type counts, project paths, the
  execution-guard refusals of `bfcl generate` (`execution_guard_refusals`,
  `execution_guard_refused_calls`), where the keys stay (`credentials`), the relays' records
  (`relays`), the entries whose request failed (`inference_errors`) and the memory prerequisite
  entries whose request failed (`prereq_inference_errors`), per category, and the entries it left
  out of the score as harness errors (`scoring_excluded`; per category and kind in
  `harness_errors_left_out`, and `categories_unscored`).
- `web_search_setting` and `settings_infra_required: ["web_search"]`, only when the Web Search
  setting's prerequisite was missing: its entry (`status: infra_required`, `categories`,
  `prerequisite: GEMINI_API_KEY`, `when`, `infra_required` with the reason, `error`); the
  web_search categories are then absent from `category_accuracy` and `categories_run`.
- `official_sampling`: BFCL's setting (temperature 0.001 and where it is defined), the temperature
  this run used, its source (`default` for 0.0, `thinking` for 1.0, `cli:--temperature` or
  `env:GBENCH_BFCL_TEMPERATURE`)
  and whether it matched.
- Single-turn traces carry `bfcl_error_type` and `bfcl_error` with BFCL's own error codes when
  wrong (e.g. `simple_function_checker:wrong_count`, `type_error:simple`). Multi-turn traces carry
  `multi_turn_result` (the model's calls per turn and step, as BFCL's result file holds them),
  `steps`, `finish_reasons`, `completion_tokens` and, when wrong, `bfcl_error_type` (e.g.
  `multi_turn:instance_state_mismatch`, `multi_turn:execution_response_mismatch`,
  `multi_turn:empty_turn_model_response`, `multi_turn:force_terminated`), `bfcl_error` and
  `bfcl_error_details`. Agentic traces are one per scored entry the harness generated (correct
  ones included), as `bfcl_v4_agentic` records them.
- An empty reply that is the correct answer (on irrelevance entries) is marked `empty_is_answer`
  and is not counted under `empty_responses` or `failed_requests`. A reply that stopped with its
  answer inside the thinking channel is the model's outcome, scored as an empty answer and counted
  in `answer_in_thinking` (see [answer_in_thinking.md](answer_in_thinking.md)); a checker that
  stopped on such a reply is counted among the unscored replies, since base does not count it as
  a failed request.
- **Harness errors are never the model's outcome**, in every part alike. An entry whose request
  got no reply (server unreachable or an HTTP error, a timeout, a request the server refused for
  the context window; BFCL records it as an inference error and its checker scores it wrong) is
  left out of the score: out of its category's correct and total, counted under
  `failed_requests` (and `timed_out_requests` / `over_context_prompts`), in `scoring_excluded`
  and, by kind, in `request_harness_errors` (`endpoint_failure`, `endpoint_timeout`,
  `over_context`), and named in `status_reason` and `leaderboard_comparable_reason`; the run is
  `completed_with_errors`. The single-turn part gets this from base (the trace's status `FAILED`,
  `TIMEOUT` or `OVER_CONTEXT`, `scoring_excluded`); a multi-turn trace is marked
  `scoring_excluded` with its `harness_error` (`kind`, `detail`); the agentic part reads such
  entries back from its result files and takes them out of `bfcl evaluate`'s counts (per category
  in `agentic_report.inference_errors`). A multi-turn entry whose checker raised (its trace's
  `bfcl_error_type` `multi_turn:checker_exception`; upstream's evaluation aborts there) is a
  harness error too: marked `scoring_excluded` with its `harness_error` (`kind`
  `checker_exception`, the exception in `detail`), left out of its category's correct and total,
  counted in `scoring_excluded`, in `request_harness_errors` and in
  `multi_turn_report.harness_errors_left_out` as `checker_exception`, and named in
  `status_reason` and `leaderboard_comparable_reason`. The agentic part also leaves out the memory entries of a
  scenario whose prerequisite entry failed its request (`memory_prereq_failed`, see
  `memory_prereq_failed_requests` below), and an entry a stopped part did not finish is left out
  as `unfinished` (see `harness_errors`). A category that ran with no entry left in the score is
  unscored: absent from `category_accuracy`, its columns `null` (never 0), listed in
  `categories_unscored`. A multi-turn or agentic part whose every entry failed its request
  measured nothing of the model: its categories are unscored and the part is also recorded under
  `harness_errors` (`part`, `error`, `categories_unscored`; a part a wall clock stopped before any
  entry finished gets `categories_unscored` on its stop record instead). When nothing of the run
  was scored, the score is withheld: `accuracy` is `null`, `accuracy_withheld_reason` says why and
  the status is `failed`. `status_reason` names what made the status other than `success`.
- `memory_prereq_failed_requests` (when the agentic part ran): its memory prerequisite entries
  (the conversations that write a scenario's memory before its questions run) whose request
  failed, per category in `agentic_report.prereq_inference_errors` (`failed`, `timed_out`,
  `over_context`). They are not scored entries, so they are not in `failed_requests`, but the
  scenario's questions ran against a memory that lacks what they were to write (upstream makes
  every question of a scenario depend on all of its prerequisites): those questions are left out
  of the score as harness errors (`memory_prereq_failed` in `request_harness_errors` and
  `agentic_report.harness_errors_left_out`). Any of them makes the run `completed_with_errors` and
  adds the agentic part's named reason (see [`bfcl_v4_agentic`](bfcl_v4_agentic.md#bounds)).
- `endpoint_failures`: the endpoint failures gbench's endpoint relay counted (see
  [Containers and network](#containers-and-network)), only when there were any.
- `gemini_models_answered` (`{model: searches}`) and `gemini_cascade`: the Gemini models that
  answered the agentic part's web searches and the cascade they were asked from, when a
  web_search category was generated.
- `harness_errors` (empty when none happened): the harness parts `GBENCH_BFCL_TIMEOUT_S` stopped,
  each with `part`, `error`, `timeout_s` and its counts, and the multi-turn or agentic parts whose
  every entry failed its request, with no wall clock involved (`part`, `error`,
  `categories_unscored`, and for the multi-turn part `entries_failed`; see above). The rest of the
  run is kept: a stopped multi-turn part keeps the entries it finished, and every other entry
  becomes a `TIMEOUT` trace (`harness_timeout: true`) counted under `failed_requests` and
  `timed_out_requests` and left out of the score as a harness error (`unfinished`); a stopped `bfcl
  generate` scores the agentic entries the harness finished (`bfcl evaluate --partial-eval`) and
  leaves the others out the same way (in an `--eval-limit` run a stop during the harness' rewrite
  of a result file can also drop entries it had finished, which then count as unfinished; see
  [`bfcl_v4_agentic`](bfcl_v4_agentic.md#bounds)); a category none of whose entries finished is
  unscored, and a multi-turn part or `bfcl generate` stopped before any of its entries finished
  leaves its categories unscored (still counted under `failed_requests`, `categories_unscored` on
  its stop record, columns `null`, never scored wrong); a stopped single-turn checker leaves the
  replies unscored (`HARNESS_ERROR` traces, `scoring_excluded`, out of every accuracy, so those
  columns are `null`); a stopped `bfcl evaluate` leaves the agentic categories unscored (after a
  stopped `bfcl generate`, both stops are recorded, generate first). Any harness error, like a
  setting reported `infra_required`, makes the status `completed_with_errors` (`failed` when
  nothing was scored).
- `leaderboard_comparable` is `false` for every run that includes the agentic part's web_search
  categories, since their web search is served by Gemini Google Search grounding (the agentic
  part's reason), and for every run that leaves any segment out, since BFCL Overall Acc weighs
  all of them; so a `bfcl` result is never leaderboard-comparable.
  `leaderboard_comparable_reason` lists what differs: a segment that did not run in full, a
  setting reported `infra_required` (the Gemini key missing or rejected, or a search no
  grounding model served), an `--eval-categories` / `--eval-limit` subset, a shard, the
  temperature (of any part), repeated attempts, an execution-guard refusal, a harness error,
  endpoint failures the relay counted, single-turn and multi-turn entries left out of the score as
  harness errors ("N single-turn and multi-turn entries left out of the score as harness errors
  (kind: n, ...; BFCL scores a failed request wrong)"), categories left unscored, a withheld score
  or the agentic part's own reason (the search substitution, its entries left out of the score and
  failed memory prerequisite entries among them).
- `dataset_revision` records the pinned commit.

## Subsets
- `--eval-categories` takes single category names or BFCL's group names: `all` / `all_scoring`,
  `single_turn`, `non_live`, `live`, `multi_turn`, `agentic`, `memory`, `web_search`, `python`,
  `non_python`. The pre-v4 names `simple`, `java` and `javascript` are accepted for
  `simple_python`, `simple_java` and `simple_javascript`; any other unknown name is an error.
- `--eval-limit N` is N entries for the whole suite, never N (or one) per part: the N are dealt
  over the categories of every part that runs (one per category before any gets two, in a seeded
  order), so each part gets the entries of its drawn categories (13 : 4 : 5 at `--eval-limit 22`),
  then drawn stratified by category inside each part (the agentic part picks its questions as
  `bfcl_v4_agentic` does, plus the memory prerequisite entries they depend on, which are not
  scored questions). A part that gets 0 is not run (nor are its prerequisites checked), e.g. at
  `--eval-limit 1`.
- A part with a single scored entry (e.g. the agentic part at `--eval-limit 5`) gives upstream's
  `bfcl evaluate` one request latency, and its leaderboard summary takes `statistics.stdev` of
  them, which raises. The harness child makes that standard deviation NaN instead
  (`_allow_one_latency`); gbench reads no latency column and no score changes.
- `GBENCH_SHARD=I/N` shards the single-turn and multi-turn entries; the agentic harness cannot be
  sharded, so it runs whole in shard 1 only and the shards still add up to the benchmark once.
- `--attempt-count k` runs k attempts of every single-turn and multi-turn entry (reported as
  avg@k with pass@k and pass^k in `attempts`, over the attempts in the score: `null`, never 0,
  when every one was a harness error); the agentic harness generates once.

The run logs its planned workload in one line at its start (for example `[bfcl] --eval-limit 5: 5
sample(s) x 1 attempt = 5 generations`, with the entries of each part) and records it in the
result as `eval_limit_plan` (`limit`, `samples`, `per_setting`, `attempts_per_sample`,
`generations`, `attempts_per_part`, and `note` for the memory prerequisite entries). With
`--attempt-count k` above 1 and agentic entries drawn, the line and `generations` count k attempts
for each single- and multi-turn entry and 1 for each agentic one (e.g. `x single 3 attempts, multi
3 attempts, agentic 1 attempt = 13 generations`).

## Containers and network
Every container is started from the local `gbench-bfcl:2026.3.23` image (`--pull never`, every
capability dropped, `no-new-privileges`):

| Container | What runs in it | Network |
| --- | --- | --- |
| image probe (`bfcl.image_probe`) | the release and pinned-file checks | `--network none` |
| single-turn checker (`bfcl.single_turn_verdicts`) | BFCL's `eval_runner` over the replies gbench collected | `--network none` |
| multi-turn part (`bfcl.multi_turn_job`) | BFCL's multi-turn loop, which executes the model's calls with `eval()` in the harness process under the execution guard, talking to the model endpoint (through gbench's endpoint relay when the endpoint has a real key: `VLLM_API_KEY`, else `OPENAI_API_KEY`) | docker's bridge network, relies on the host rule |
| agentic part | `bfcl generate` on the bridge network, `bfcl evaluate` and the other jobs with `--network none`; see [`bfcl_v4_agentic`](bfcl_v4_agentic.md#containers-and-network) | as listed there |

The host network is never used. On a cloud VM the containers on the bridge network rely on the
host rule that keeps them away from the metadata server at `169.254.169.254` (which hands out the
VM's service-account token):

```bash
sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save   # keep it
```

Before the first generation of a run with multi-turn categories, gbench's guard
(`require_metadata_blocked`, see [toolchains](toolchains.md#cloud-vms-the-metadata-server)) starts
one short-lived container of the image on the bridge network that opens and closes a TCP connection
to `169.254.169.254:80` (no request is sent); if it opens, the suite stops with an
`[INFRA REQUIRED]` error naming the rule, and a probe that gives no clear answer (neither `python3`
nor `bash` in the image, a docker error or timeout) is a hard error too. gbench never runs sudo or
changes the host's firewall itself. An endpoint on this host (`localhost`, `127.0.0.0/8`, `0.0.0.0`
or `[::1]` in the URL) is reached from the bridge network at `host.docker.internal`
(`--add-host host.docker.internal:host-gateway`), so it must listen on an address the docker bridge
reaches: vLLM as gbench starts it listens on every interface, a server bound to `127.0.0.1` only
does not. gbench checks first, from a container of the image on that network, that a TCP connection
to it opens (and that a named endpoint host resolves there, or, when a proxy is set and the
request goes through it, the proxy's host: the container gets the proxy variables), and stops with
an `[INFRA REQUIRED]` error otherwise, before anything is generated. Any other endpoint URL is used
as given. A proxy variable (`HTTP_PROXY`, `HTTPS_PROXY` and their
lowercase forms) on a loopback address is rewritten the same way, and `host.docker.internal` (and,
while a relay runs, the bridge gateway's address) is added to `NO_PROXY` / `no_proxy` when a proxy
is set.

No key enters a container that runs the model's calls. When the endpoint has a real key
(`VLLM_API_KEY`, else `OPENAI_API_KEY`; unset, `EMPTY`, `dummy`, `not-needed` and gbench's other
placeholders are no key), it stays in gbench's process, and the multi-turn part reaches the
endpoint through gbench's endpoint relay (`swebench_common.EndpointRelay`). The relay is a thread
of the gbench process listening on the docker bridge gateway (from `docker network inspect
bridge`, usually `172.17.0.1`). It adds `Authorization: Bearer <key>` and forwards only
`/v1/chat/completions`, `/v1/completions` and `/v1/models` to the endpoint as given, which may
then listen on `127.0.0.1` only. The container gets the relay's URL, and the placeholder `EMPTY`
as `OPENAI_API_KEY`. Without a key the multi-turn part gets no relay and reaches the endpoint as
described above. The agentic part's `bfcl generate` goes through the endpoint relay on every run,
keyless too (since 2026-10-13); the relay adds the key only when there is one. A relayed model
request fails only when it stops making progress: no event of its reply for 300 s
(`GBENCH_RELAY_IDLE_TIMEOUT_S`) while the server's token counters and KV-cache usage also stand
still. That is counted as `stalled_no_tokens`, an endpoint failure, and the client gets a 504.
Only a safety cap far above any reply bounds its total duration. Its floor is the client's
timeout plus 60 s, and at least 3600 s (`EndpointRelay.TIMEOUT_S`); it is `base.request_timeout_s`
of the request's `max_tokens` (or of the served window when it sends none) plus 120 s when that
is larger. See
[relay.md](../relay.md). Whenever a relay runs, gbench checks before the first generation, from a
container of the image on the bridge network, that a TCP connection to a listener of its own on
the gateway opens, instead of the check of an endpoint on this host. Otherwise it stops with an
`[INFRA REQUIRED]` error (a host firewall must let the docker bridge reach that address). The
agentic part's web_search `bfcl generate` holds a placeholder for the Gemini key and searches
through gbench's Gemini relay on the same gateway; its container must resolve
`vertexaisearch.cloud.google.com`, which this suite checks before the first generation when that setting will run (see
[`bfcl_v4_agentic`](bfcl_v4_agentic.md#credentials-and-the-execution-guard)). A launch whose
forwarded variables would still carry a credential of gbench's environment (for example a proxy
URL with a password) stops with an `[INFRA REQUIRED]` error naming the variable.

Endpoint failures the endpoint relay counts (the endpoint did not answer, a reply that is not
2xx, a completions reply that is not JSON, an error event in a stream: `upstream_errors`; and
`requests_refused`), in the multi-turn part and in each `bfcl generate` of the agentic part, are
harness errors, never the model's outcome: any of them makes the run `completed_with_errors`, is
listed in `endpoint_failures` (both parts summed; per part in `multi_turn_report.endpoint_failures`
and `agentic_report.endpoint_failures`) and adds a named reason to `leaderboard_comparable_reason`
("harness errors during the multi-turn part: the served endpoint failed N relayed model
request(s) (...)", or the agentic part's own). A 400 that names the context window is counted
apart by the relay (`context_length_rejections`), not as an endpoint failure; the entry it ends is
an `over_context` inference error. The client retries some failures itself and the relay cannot
tell which entry a request belonged to, so the count is per part; an entry whose request still
failed counts under `failed_requests` and is left out of the score as a harness error (see
Reading the result).

The agentic part's harness project root must not hold a `.env`: upstream's `bfcl generate` and
`bfcl evaluate` load `<project root>/.env` into the process that runs the model's calls, so one
there stops the suite with an `[INFRA REQUIRED]` error naming it, before the first generation (see
[`bfcl_v4_agentic`](bfcl_v4_agentic.md#credentials-and-the-execution-guard)).

The agentic part's prerequisite and network checks run as this suite's: an `[INFRA REQUIRED]`
error from them names `bfcl` and points to `docs/evals/bfcl.md`, whether it comes before the first
generation or from the agentic part itself; so do the agentic part's other errors (a failed
`bfcl generate` or `bfcl evaluate`, the `--eval-limit` id selection, a wall clock stop).

## Requirements
- An OpenAI-compatible `/v1` endpoint that returns structured `tool_calls`: vLLM needs
  `--enable-auto-tool-choice --tool-call-parser <the model's parser>` (e.g. `gemma4`). Without a
  parser vLLM rejects every request that declares tools (HTTP 400), and a server gbench launches
  itself does not enable one yet, so serve the model yourself with those flags and point
  `--remote-endpoint` at it. Serve the model's full context window: `multi_turn_long_context`
  conversations grow long, and a conversation that no longer fits is an `over_context` harness
  error, left out of the score, so the score then covers fewer entries. Such a conversation may
  be model-caused (it grows with the model's own tool calls and their results), which upstream
  scores as the model's outcome; gbench leaves these multi-turn and agentic entries out and names
  their count in `leaderboard_comparable_reason` ("... left out as harness errors because the
  conversation outgrew the served context window ...").
- **Docker and the local harness image `gbench-bfcl:2026.3.23`**, built once from the gbench
  repository root (gbench never pulls it; override the tag with `GBENCH_BFCL_IMAGE`):
  ```bash
  docker build -t gbench-bfcl:2026.3.23 -f gbench/docker/bfcl.Dockerfile gbench/docker
  ```
  The image is `python:3.10-slim` (the Python the BFCL README asks for) with the PyPI release
  `bfcl-eval==2026.3.23`, the current one and byte-identical to `bfcl_eval/` at the pinned commit
  (all 183 files); its wheel is checked against PyPI's sha256 before it is installed with the
  release's own pins (`numpy==1.26.4`, `filelock==3.20.0`, `tree_sitter==0.21.3`,
  `faiss-cpu==1.11.0`, ...). Those pins conflict with the main gbench environment, which is why
  the harness lives in the image. `gbench/docker/bfcl.constraints.txt` holds the resolved set of
  the dependencies the release leaves open (the validated build's `pip freeze`), so a rebuild gets
  the same versions. `soundfile` is added because the release cannot start without it: its model
  registry imports the Qwen handler, whose `qwen_agent` imports `soundfile` without declaring it
  (gorilla issue #1338); nothing gbench runs uses it. The image also holds `memory_vector`'s
  encoder, `sentence-transformers/all-MiniLM-L6-v2` at revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, loaded offline on the CPU as upstream loads it. The
  build downloads about 5 GB (PyTorch comes with the release's `sentence-transformers`) and the
  image is about 14 GB. Its labels (`gbench.bfcl.release`, `gbench.bfcl.commit`, `gbench.bfcl.minilm`) must match; no
  Docker, no image, an image for another release, or a file that differs from the pinned commit
  in the image stops the suite with an `[INFRA REQUIRED]` error before anything is generated.
- On a cloud VM, for the multi-turn and agentic categories: the host rule
  `sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP`, made persistent, and an endpoint
  the docker bridge reaches, or, while a relay runs, a bridge gateway the docker bridge reaches
  (see [Containers and network](#containers-and-network)).
- **`GEMINI_API_KEY`** (or `GEMINI_API_KEYS`) with Google Search grounding quota for the two
  web_search categories, which Gemini Google Search grounding serves with gbench's one Gemini
  cascade (`gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-3.5-flash`,
  `gemini-3.1-pro-preview` unless `GBENCH_SEARCH_MODELS` / `GBENCH_JUDGE_MODELS` name others; the
  result records the models that answered in `gemini_models_answered`), plus the agentic suite's
  other prerequisites (a reachable endpoint, a writable project root); see
  [`bfcl_v4_agentic`](bfcl_v4_agentic.md#requirements). Without a key, with one the Gemini API
  rejects, or with a cascade model it does not serve to that key, the Web Search setting is
  reported `infra_required` and the rest of the run is scored (not leaderboard-comparable); a run
  of the web_search categories alone hard-errors.
- A writable `bfcl-eval` project root for the agentic part's lock, result and score
  directories: `GBENCH_BFCL_PROJECT_ROOT` (default `$GBENCH_CACHE_DIR/bfcl`, i.e.
  `~/.cache/gbench/bfcl`, shared with `bfcl_v4_agentic`; upstream's own `BFCL_PROJECT_ROOT` is
  read as a legacy alias when `GBENCH_BFCL_PROJECT_ROOT` is unset),
  mounted into the image at the same path. It must hold no `.env` (see
  [Containers and network](#containers-and-network)); use a directory only gbench uses, not an
  upstream BFCL checkout.
- Network access to `raw.githubusercontent.com` on the first run for the 23 single-turn data files
  (about 8 MB). Offline, copy `bfcl_eval/data/BFCL_v4_*.json` and
  `bfcl_eval/data/possible_answer/BFCL_v4_*.json` from a checkout of the pinned commit (or from the
  harness image's `/usr/local/lib/python3.10/site-packages/bfcl_eval/data/`, whose single-turn
  files are identical) into
  `$GBENCH_CACHE_DIR/bfcl_data/<commit>/` (default `~/.cache/gbench/bfcl_data/<commit>/`),
  keeping the `possible_answer/` subdirectory. A cached file that does not match its blob id is
  downloaded again; a file that cannot be downloaded, or whose download does not match, stops
  the suite with an `[INFRA REQUIRED]` error. No category is ever skipped.
- No LLM judge.

## Run
```bash
# serve the model with a tool-call parser, then (the Gemini key serves the web search):
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals bfcl --max-output-tokens 4096
# a subset (single names or group names), recorded as not leaderboard-comparable:
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals bfcl --max-output-tokens 4096 \
       --eval-categories single_turn,multi_turn
```
`--max-output-tokens` is required for every eval run, but this suite does not apply it: its
requests carry no `max_tokens`, as BFCL's do. A full run makes one request per single-turn entry, one
request per step of every multi-turn turn, and runs the whole agentic track, so it takes hours on
one server; `--batch-sizes` sets the concurrency of all three parts. No part has a wall clock
unless `GBENCH_BFCL_TIMEOUT_S` sets one (seconds, each part timed on its own).
