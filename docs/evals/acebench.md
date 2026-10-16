# acebench

ACEBench (Chen et al., 2025; arXiv:2501.12851;
[chenchen0103/ACEBench](https://github.com/chenchen0103/ACEBench)): function-calling evaluation
across atomic parameter types, single- and multi-turn dialogues, similar APIs, user preferences
(Normal), ill-posed requests the model must flag (Special), and multi-turn agent tasks in a
simulated phone and airline environment (Agent). gbench runs all three categories in both settings
the leaderboard publishes (English and Chinese) as the repo's `generate.py` + `eval_main.py` do. It
is a tool-use benchmark, not agentic coding.

## Dataset (pinned)
`data_all/data_<lang>/data_<category>.json` and
`data_all/data_<lang>/possible_answer/data_<category>.json` for the 17 categories of
`category.py` (12 Normal, 3 Special, 2 Agent) in `data_en` and `data_zh`, from `chenchen0103/ACEBench` at
commit `56dd66cf6439b0d9655ee1b353e4cd745c6f664e` (HEAD of `main`): 973 English and 967 Chinese
Normal + Special rows, and 50 agent conversations per language (20 `agent_multi_step`, 30
`agent_multi_turn`). 68 files, about 9 MB, downloaded once to
`~/.cache/gbench/acebench/<commit>/data_<lang>/` (`GBENCH_CACHE_DIR` overrides the root) and checked
against each file's git blob id (`PINNED_BLOBS`; a cached file that does not match is downloaded
again; a download that fails or does not match stops the suite with an `[INFRA REQUIRED]` error).
This revision carries 5 ground-truth, 1 time and 1 question fix missing from the older HF
reformatting `oliveirabruno01/acebench`, which is not used.

The agent environment and the prompts come from the same commit, vendored in
`gbench/third_party/_acebench_vendor/` (MIT): `model_inference/prompt_en.py` and
`prompt_zh.py`; the scenario classes `model_inference/multi_turn/scenarios{en,zh}/`
(`phone_platform/base_api.py`, `message.py`, `reminder.py`, `food_services.py`, `travel.py`;
verbatim except the `BaseApi` import, made package-relative; the `multi_step/` copies differ only
in comments and blank lines); and the prompt constants of `multi_turn/common_agent.py`,
`multi_turn/APIModel_user.py` and `multi_step/common_agent_step.py`. `_acebench_vendor/__init__.py`
records each file's upstream blob id.

## Protocol
- **Normal and Special prompts**: `SYSTEM_PROMPT_FOR_SPECIAL_DATA_<LANG>`,
  `SYSTEM_PROMPT_FOR_PREFERENCE_DATA_<LANG>` or `SYSTEM_PROMPT_FOR_NORMAL_DATA_<LANG>` as the system
  message (the functions inserted by `str.format`, i.e. rendered as a Python list; `time` or
  `profile` from the row) and `USER_PROMPT_<LANG>` with the row's conversation as the user message,
  as `common_inference.single_turn_inference` builds them.
- **Normal scoring**: the official `decode_ast` + `normal_checker` (`model_eval/checker.py`,
  `model_eval/utils.py`, `model_inference/utils.py`), vendored line for line with their quirks
  (for example, an `enum` parameter is only type-checked against the answer). Single-turn replies
  go through `extract_outermost_bracket_content` (the first outermost `[...]`); multi-turn replies
  have all whitespace removed and must then be exactly a call list. A row passes if any candidate
  answer passes. A checker exception scores the row wrong instead of aborting the run.
- **`BinOp` arguments** (`x='code ' + str(42)`, `x=[1] + [2]`, `x=1 << 3`): upstream resolves
  them with `eval(ast.unparse(node))` inside the decoding function (`model_inference/utils.py`
  for Normal rows; `multi_turn/common_agent.py`, `multi_turn/execution_role.py` and
  `multi_step/execution_role_step.py` for the agent roles), and gbench evaluates them the same
  way. Arithmetic over number and string literals is evaluated in-process (the same value). Any
  other `BinOp` goes to Python's own `eval` (`python -I`, gbench's interpreter) in an
  isolated process, a bubblewrap container of its own, whatever `GBENCH_SANDBOX` says
  (`acebench._evaluator_argv`):
  - new user, PID, IPC, UTS, network and cgroup namespaces (`--unshare-all --unshare-user`, and
    `--disable-userns` where the installed bwrap has it, so no nested namespace), all
    capabilities dropped, an empty environment, a new session, killed with gbench;
  - a filesystem that holds only the interpreter: `/usr` and the interpreter's own directories
    (its virtual environment and base prefix) read-only, the `/bin`, `/sbin` and `/lib*` links,
    the dynamic loader's cache, a private minimal `/dev` and a private `/tmp` (64 MiB where the
    installed bwrap has `--size`). No `/proc` is mounted, so the host's processes and their
    environments are out of reach, and `/run`, `/var`, the gbench cache and every other host path
    are absent, `/home` included (only an interpreter directory under a home directory is there,
    read-only, when the interpreter lives there), so neither the Docker daemon's socket nor a
    credential file is there;
  - limits of 10 s CPU, 2 GiB of memory, 16 MiB per written file and 30 s wall time.

  Its globals are the ones upstream's `eval` sees in that module: the builtins, the
  module's name and its stdlib imports (`ast` and `json` in `utils.py`; `re` and `ast` in
  `common_agent.py`; `importlib`, `inspect`, `json`, `re`, `copy`, `typing.List` and `ast` in
  the execution roles). The value comes back as a literal. A builtin exception comes back as the
  same exception with the same text (`NameError` for an undefined name, `ZeroDivisionError`,
  ...); any other (for example `json.JSONDecodeError`, whose constructor needs more than its
  text) as an `Exception` subclass with the same name, module and text. Upstream's callers catch
  `Exception`, so a row or a message decodes, or fails to, exactly as upstream. A value that
  depends on the evaluating process itself (its working directory, files, environment or clock,
  for example `__import__('os').getcwd() + ''`) comes from the container, not from upstream's
  process. In an agent conversation the decoding runs in a worker thread, and the single-turn
  rows are scored after generation in worker threads (`acebench._row_scorer`, with the same row
  check), so an evaluation never stalls the event loop or the other requests. What the isolated
  `eval` cannot
  give as upstream would is refused and counted (below, and `single_turn_guard_refusals` for
  Normal rows): a name only upstream's process has (the module's own functions, classes and
  prompt constants, the method's `self` and `value`, `__file__`), a value that is not a literal
  (a function, an object), an evaluation past the limits, and literal arithmetic whose result
  would be huge (an integer power larger than 100,000 bits, an integer product whose two
  operands together exceed 100,000 bits, a string repeated past 1,000,000 characters), which
  upstream would compute without bound. A run with any refusal is not leaderboard-comparable.
  bubblewrap is a prerequisite (see Requirements).
- **Special scoring** (`eval_main.special_eval`): plain, case-sensitive substrings of the reply:
  `Missing necessary parameters` plus the API name and every missing parameter (incomplete);
  `There is incorrect value` plus every incorrect value (error_param);
  `the limitations of the function` (irrelevant). The Chinese prompts ask for the same English
  phrases, and the rules are the same.
- **Agent** (`common_inference.CommonInference.multi_turn_inference` / `multi_step_inference`,
  replayed turn by turn, at most 40 steps per conversation, `generate.py --max-dialog-turns`):
  - The model under test is the agent. Each step is one request with the agent system prompt
    (`multi_turn/common_agent.py` `MULTI_TURN_AGENT_PROMPT_SYSTEM_<LANG>` plus `TRAVEL_PROMPT_<LANG>`
    and/or `BASE_PROMPT_<LANG>` for the classes involved; `multi_step/common_agent_step.py`'s own
    prompt for multi-step) and the user prompt with the function list and the conversation so far,
    exactly as the upstream scene accumulates it (multi-step history holds the task and the
    execution results, as upstream's `Mulit_Step_Scene` does).
  - A reply that decodes as a call list goes to the execution role, which runs the calls against
    the scenario instances of that conversation (created from the row's `initial_config`) with a
    replica of `execute_agent_func_call`, quirks included: a method several classes share (for
    example `turn_on_wifi`) runs on each; a call no class knows repeats the previous call's result
    or fails; a failing call makes the multi-turn execution answer with upstream's literal
    `Error during execution: {str(e)}` text (its missing f-prefix), split into characters as
    upstream's result parsing does, and the multi-step execution record the error message and go
    on. A call with a dict-valued argument is routed as upstream routes it (its call-string builder
    reads the dict as a nested call). In multi-step, every agent message goes to the execution
    role, which answers a non-call with "Please do not ask me any questions, use the known
    conditions to solve the problem".
  - In `agent_multi_turn`, a reply that is not a call goes to the **user simulator**
    (`multi_turn/APIModel_user.py` `APIUSER`, upstream gpt-4o, `generate.py --user-model`), which
    starts the conversation from the row's instruction and ends it with "finish conversation".
    The user simulator is the one replaced component: its prompts and conversation are upstream's,
    and gbench's Gemini judge cascade answers them (the instructions as the system instruction; the
    simulator's own earlier replies, which upstream sends back with the system role, as its model
    turns, since Gemini takes no system turns inside a conversation). Every request carries
    `APIUSER`'s own sampling (its `__init__` defaults, which `common_inference` uses): temperature
    0.001, top_p 1 and `max_tokens` 1000 for the reply. gpt-4o does not think; the Gemini
    simulator counts its thinking against `max_output_tokens`, so it is sent
    `max_output_tokens` 9192, the 1000 plus the Gemini thinking allowance of 8192
    (`gemini_budget.GEMINI_THINKING_ALLOWANCE`, the rule gbench's other Gemini graders follow),
    so its reply keeps upstream's 1000 tokens. The result records it under `user_simulator` (the
    cascade, the sampling, `max_output_tokens`, `upstream_max_tokens`, `thinking_allowance`, the
    models that answered, the request count). `user_simulator` also counts the replies that
    still stopped there (finish reason `MAX_TOKENS`): `replies_cut_at_max_tokens` (with text, used
    as they are, as upstream uses a gpt-4o reply cut at 1000), `empty_replies_at_max_tokens` (no
    text: the cascade asks its next model, and a conversation whose every model and round fails is
    a user-simulator outage) and `outages_after_max_tokens` (user-simulator outages whose cascade
    saw such a reply). Any count makes the run not leaderboard-comparable; each trace carries its
    own (`user_sim_replies_cut_at_max_tokens`, `user_sim_empty_at_max_tokens`,
    `user_sim_outages_after_max_tokens`). `gbench --merge` (`merge_shards`: the generic re-count
    of the union's traces, which refuses shards whose headline, the English Summary, is not a
    correct / scored count) counts `user_simulator` again over the union's agent conversations
    and names the cut-off reason once, over those counts.
  - **Without the user simulator**: only `agent_multi_turn` needs it. When no Gemini key is set,
    or a live check rejects every key, the selected `agent_multi_turn` conversations are not run
    and that category is reported `infra_required` in its own entry (below), while the other 16
    categories (Normal, Special and `agent_multi_step`, both languages) run and are scored as
    usual. (Upstream generates the categories `generate.py --category` names, from
    `category.py` `ACE_DATA_CATEGORY`, and its `generate_result_csv` reads a category with no
    result as 0; gbench does not score the missing category at all, not even as 0.) If
    `agent_multi_turn` is all the run selected (a shard or `--eval-limit` that leaves nothing
    else), the suite hard-errors (`infra_required`).
  - **Execution guard**: upstream builds each call as a string and `eval()`s it. gbench
    dispatches each call to the scenario method directly: the argument values it evaluates are
    the literals upstream's call strings contain (including negative and complex numbers),
    literal arithmetic with the size bounds above, `dict(...)`, and the `NameError` an undefined
    name raises there. Anything else upstream's `eval()` of a call string would run (a nested call
    of a builtin or module function such as `json.dumps(...)`) is refused with an error in its
    place, counted in `acebench_report.execution_guard_refusals` and on the trace (together with
    the `BinOp` arguments refused while decoding that conversation's messages), and makes the run
    not leaderboard-comparable. `BinOp` arguments in the agent's messages are evaluated while
    decoding, as above.
  - **Agent scoring** (`eval_main.agent_eval` with `checker.agent_checker`): the `SAVED_CLASS`
    attributes of the scenario instances after the conversation (round-tripped through JSON, as
    upstream's result file is) against the possible answer; a conversation is right end to end
    only if every class matches (quirk kept: a possible-answer class with no class of the same
    attributes in the result reuses the previous class's verdict). `process_accuracy` is
    `agent_eval_process`: the share of milestone calls found in order among the agent's calls
    (the best of several milestone lists), 1.0 for a right conversation.
- **Aggregation** (`evaluation_helper.generate_result_csv`, per language): accuracy per
  sub-category, rounded to 3 decimals except the unrounded Special ones, as upstream. Multi-turn
  Normal rows are grouped by conversation (`..._<conv>_<turn>`) and a conversation counts only if
  every turn is right (`multiplt_turn_accuracy`). `normal_summary`, `special_summary` and
  `agent_summary` are the unweighted means of their sub-categories (a sub-category that did not
  run counts 0, as upstream reads it), and **Summary** = 0.2676 x special + 0.578 x normal + 0.1545
  x agent, rounded to 3 decimals: the leaderboard's **overall**. A sub-category reported
  `infra_required` does not count 0: every aggregate that weighs it is null instead (for
  `agent_multi_turn`: `agent_summary`, the leaderboard's `agent` and `overall` columns and the
  Summary, in each language where it was selected); the other columns are computed as usual.
- The Normal/Special port was checked against `eval_main.py` at the pinned commit with synthetic
  replies for all 973 English rows (identical per-sub-category accuracies, summaries and prompts).
  The agent replica was checked against the upstream classes run in-process on all 100 agent
  conversations of both languages with a scripted agent and user: once replaying the milestone
  calls, and twice with perturbed replies (prose around calls, unknown and misspelled methods,
  wrong keywords, non-call list elements, dict arguments, undefined names, `inf`, negative and
  complex literals, questions to the user): identical prompts, dialogue histories, final states,
  milestones, verdicts and process scores in all 300 conversations, apart from the replies with a
  nested `json.dumps(...)` call, which the guard refuses. `BinOp` arguments were checked against
  upstream's `CommonAgent.decode_function_list` and `utils.ast_parse` run on the upstream files
  (string and list concatenation, tuple repetition, shifts and bitwise operators, builtin calls,
  `%` formatting, set union, the modules' own imports and `__name__`, `ZeroDivisionError`,
  undefined names): the same values and exceptions, apart from expressions that name `self` or
  `value`, which are refused and counted.

## Headline and how to read the result
- `accuracy` (headline): the English Summary (the English leaderboard's overall), on the 0-100
  scale. `overall_zh` is the Chinese Summary (the Chinese leaderboard's overall). Both are null
  when `agent_multi_turn` is reported `infra_required` (the Summary weighs it).
- `agent_multi_turn` reported `infra_required` (no usable Gemini key):
  `categories_infra_required` is `["agent_multi_turn"]` and `agent_multi_turn_category` is its
  entry (`status: "infra_required"`, the `languages` and `conversations_not_run` per language, the
  `prerequisite`, the reason in `infra_required`, an `[INFRA REQUIRED]` `error` text,
  `leaderboard_comparable: false`, `total_questions` 0, `accuracy` null). Its
  `per_sub_category` / `category_accuracy` rows stay, with `status: "infra_required"`, `total` 0
  and a null `accuracy` (and `process_accuracy`); `user_simulator.infra_required` repeats the
  reason and `acebench_report.agent_conversations_infra_required` counts the conversations not
  run. The status is `completed_with_errors` (or worse, by the usual rules), never `success`,
  and the run is not leaderboard-comparable.
- `normal_summary`, `special_summary`, `agent_summary` and `leaderboard_columns` are the English
  leaderboard's columns (`leaderboard_columns`: `atom`, `single`, `multi`, `api`, `pref`,
  `summary` (the normal summary), `special`, `agent`, `overall`, as the leaderboard page names
  them). `per_language.<en|zh>` holds all of them for each language, with its `per_sub_category`.
- `per_sub_category` (English) and `per_sub_category_zh`: every sub-category's `accuracy`,
  `correct`, `total` and `rows` or `conversations`; multi-turn Normal and agent entries also carry
  `process_accuracy` (for agent sub-categories, `agent_multi_turn_process` /
  `agent_multi_step_process` of `generate_result_csv`).
- `user_simulator.models_used`: the Gemini model that answered each user-simulator request (the
  per-conversation list is `user_sim_models`); a cascade model the API answers HTTP 404 for is not
  asked again within a request, as in gbench's judge cascade.
- `category_accuracy`: `correct`, `total` and `accuracy` per sub-category, English under its name
  and Chinese as `<name>_zh`. Multi-turn Normal and agent sub-categories count conversations.
- Agent traces (`category` `agent_multi_turn`, `agent_multi_step`, or with `_zh`) carry the whole
  dialogue (`agent_dialogue`), the final `SAVED_CLASS` state (`agent_result`), the milestone calls
  (`agent_process`), `process_accuracy`, `model_requests`, `user_sim_requests`, `user_sim_models`,
  `execution_guard_refusals` (calls the guard refused and `BinOp` arguments refused while
  decoding) and `status`.
- A request of the model that fails (after base's retries) ends that conversation: it counts in
  `failed_requests` and is left out of the accuracies, like a failed single-turn row. A user
  simulator that gets no reply from any cascade model ends the conversation too: it counts in
  `judge_outages` and is left out. A case where upstream's own run would abort (for example a final
  state its result writer cannot serialize) scores the conversation wrong and is counted in
  `acebench_report.agent_errors`.
- `row_accuracy` is the plain share of correct Normal and Special rows over both languages.
- **Subsets.** Because `generate_result_csv` reads a sub-category with no result as accuracy 0,
  an `--eval-limit` or shard run that does not draw every sub-category caps the Summary below
  100 even for a model right on every drawn item. `subcategories_not_run` lists, per language,
  the sub-categories the run did not run, and `summary_ceiling` (also under
  `per_language.<en|zh>`) is the Summary such a model would get on this draw (null when a
  sub-category was reported `infra_required`). When the English list of such a subset run is not
  empty, `leaderboard_comparable_reason` names those sub-categories and the ceiling, so a headline such as
  18.60 on a 9-item draw is read against its ceiling (27.50 for that draw), not against 100.
- `leaderboard_comparable` is `true` for a full run of both languages at temperature 0.0 with one
  attempt, no category reported `infra_required`, no decoding penalties, no failed requests or
  conversations, no user-simulator outage, no
  aborted conversation, no execution-guard refusal, no refused `BinOp` argument in a Normal
  reply (`acebench_report.single_turn_guard_refusals`, with the first 20 rows and expressions in
  `single_turn_guard_refused`), no reply cut off by a budget below the official cap
  (`acebench_report.single_turn_replies_cut_off`, `agent_replies_cut_off`; see Budget under Run),
  no prompt near LLMInfer's window that could not be counted exactly, and no
  user-simulator reply stopped at its 9192-token limit (APIUSER's 1000 plus the 8192 thinking
  allowance; the user simulator's replacement is the one
  allowed deviation and is recorded). Otherwise `leaderboard_comparable_reason` lists what differs,
  including `--eval-limit` and a shard (which can split conversations).
- `official_sampling`: ACEBench's settings for a model served from local weights (temperature
  0.0, top_p 0.9, max_tokens 1024), the temperature this run used, its source and whether it
  matched, and whether the 1024 cap was applied (see Budget under Run). `acebench_report` records
  the languages, the agent conversation and request counts, the 40-step limit, the per-request
  token budget of the agent steps (`max_tokens_per_request`), upstream's 1024
  (`upstream_max_tokens`), the policy that chose the budget (`token_policy`: `official` without
  `--thinking`, `reasoning_run_budget` with it), the agent replies that stopped at the official
  cap, the replies cut off below it (`single_turn_replies_cut_off`, `agent_replies_cut_off`) and
  `upstream_context_window` (`max_model_len` 8192, whether it was `applied`, the single-turn
  rows and agent requests it limited, the rows, agent steps and conversations whose prompt was
  longer than the window, and the prompts that could not be counted exactly).

## Requirements
- A running OpenAI-compatible `/v1` endpoint.
- **`GEMINI_API_KEY`** (or `GEMINI_API_KEYS`) for the user simulator of `agent_multi_turn`: the
  suite checks the key with a live request before generating. When it is unset or rejected,
  `agent_multi_turn` alone is reported `infra_required` (see Headline) and the other categories
  run; the suite hard-errors only when `agent_multi_turn` is all the run selected. The cascade's
  models are `base.judge_cascade()` (`GBENCH_JUDGE_MODELS` overrides them).
- Network on the first run (raw.githubusercontent.com for the data files; the Gemini API during the
  run); the data is cached afterwards.
- **bubblewrap** (`apt install bubblewrap`, with unprivileged user namespaces allowed) for the
  isolated evaluator of `BinOp` arguments. Before generating, the suite starts the container and
  checks it (`acebench.container_problem`): the evaluator must be one of the first processes of
  its own PID namespace, see no `/proc`, `/run` or Docker socket and nothing on its root but the
  interpreter's directories, and evaluate `[1] + [2 << 1]` to `[1, 4]`. When bubblewrap is
  missing or blocked, or the container is not isolated so, the suite hard-errors
  (`infra_required`). `GBENCH_SANDBOX` does not apply (not even `none`): the evaluator always runs
  in its container.
- No other API key. Normal, Special and the agent verdicts are deterministic (no LLM judge).

## Run
```bash
export GEMINI_API_KEY=...
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals acebench --max-output-tokens 2048
```
Without `GEMINI_API_KEY` the same command runs every category but `agent_multi_turn`, which is
reported `infra_required`, and ends `completed_with_errors` with null Summaries.
A full run sends 1940 single-turn requests plus the agent conversations (100, each up to 40 steps).
Add `--eval-limit N` for a quick check: N samples for the whole suite, dealt over the
sub-categories (per language) of the single-turn rows and the agent conversations, one per
sub-category before any gets two, in a seeded order; each part is then drawn stratified by
language and sub-category (flagged in the result). A part that gets 0 is not run. `--sandboxes` sets the concurrency of both parts
(conversations run concurrently, their steps in order).

The run logs its planned workload in one line at its start (for example `[acebench] --eval-limit
5: 5 sample(s) x 1 attempt = 5 generations`, with the samples of the single-turn and agent parts)
and records it in the result as `eval_limit_plan` (`limit`, `samples`, `per_setting`,
`attempts_per_sample`, `generations`).

Sampling: temperature 0.0 without `--thinking`, 1.0 with it (gbench's rule for every suite) and
`top_p` 0.9 in both modes, on every
request of the model under test (single-turn rows and agent steps): the `SamplingParams` ACEBench's
`model_inference/model_infer.py` (`LLMInfer`) uses for a model served from local weights, which is
how an open-weight model such as gemma-4 runs there (`generate.py --model-path`); the hosted-API
classes use `generate.py`'s `--temperature 0.7 --top-p 1` instead. `--temperature`, or
`GBENCH_ACEBENCH_TEMPERATURE` for this suite alone (it takes precedence), overrides the temperature
and is recorded under `official_sampling`.

Budget: `LLMInfer` sends `max_tokens=1024` with every request (`model_infer.py` line 32 at the
pinned commit), and ACEBench defines no settings of its own for reasoning models. Without
`--thinking` every request of the model under test (single-turn rows and agent steps) carries
exactly `max_tokens=1024`, whatever `--max-output-tokens` says (`max_output_tokens_note` records
that the flag was not applied); a reply that stops there is scored on what arrived, as upstream
scores it, and is counted apart from truncations (`replies_at_official_max_tokens` for the rows,
`acebench_report.agent_replies_at_official_max_tokens` for the agent steps), so it does not make
the run `completed_with_errors`. With `--thinking` the 1024 cap, written for non-reasoning
generation, is not applied: each request gets the run's `--max-output-tokens` (else the thinking
floor), and `official_sampling` records `max_tokens: 1024`, `max_tokens_applied: false`, the
note "official cap not applied to reasoning" and `max_tokens_sent`
(`base.resolve_token_budget`).

Context window: `LLMInfer` serves the model with vLLM's `max_model_len=8192` (`model_infer.py`
lines 28 and 36) on the vLLM that ACEBench pins (`requirements.txt`: `vllm==0.6.1.post1`, the V0
engine). That engine, at `v0.6.1.post1` (commit `acda0b35d00e`), does this with the window:
chunked prefill stays off for a window of 32K or less (`engine/arg_utils.py`), so the
scheduler's prompt limit is 8192 (`core/scheduler.py` `_get_prompt_limit`, with
`max_num_batched_tokens = max(8192, 2048)` in `config.py`). A text prompt of more than 8192
tokens is not rejected (`engine/llm_engine.py` `_validate_model_inputs` checks the length only
for multimodal models): the scheduler marks it `FINISHED_IGNORED` with a warning
(`_schedule_prefills`), and the engine still returns its output, empty text with finish reason
`length` (`llm_engine.py` `_process_model_outputs`, `sequence.py` `get_finished_reason`). So
`LLMInfer.inference` returns `""` and ACEBench's run goes on: a single-turn row is scored on `""`,
and an agent step's `""` goes to the user simulator (`agent_multi_turn`) or to the execution
role, which answers a non-call with its fixed text (`agent_multi_step`), and the conversation
continues. A prompt of P tokens, P at most 8192, is generated until its length exceeds 8192
(`engine/output_processor/stop_checker.py`), so it gets at most 8193 - P tokens.

Without `--thinking` gbench applies the same window to every request of the model under test
(`acebench.upstream_window_budget`). A prompt whose text in bytes, plus 512 tokens for the chat
template and the 1024, fits in 8193 cannot reach it and is not counted. Any other is counted as
upstream's vLLM counts it: the served model's `/tokenize` renders its chat template and encodes
the text with special tokens (`add_special_tokens: true`), as vLLM 0.6.1 encodes the text
`LLMInfer` renders with `apply_chat_template(tokenize=False)` (`TokenizerGroup.encode` calls
`tokenizer.encode(prompt)`, which adds special tokens). A request the window leaves less than
1024 gets that room (a reply stopping there stops at the official cap). A prompt longer than 8192
tokens is not sent: its reply is `""`, scored by the row's checker on a single-turn row
(`over_upstream_window: true`, `request_sent: false` on the trace), or handed on in the agent
conversation, which goes on and is scored on its final state and milestones. These are counted
in `upstream_context_window` (`single_turn_rows_over_window`, `agent_steps_over_window`,
`agent_conversations_over_window`) and are ACEBench's own outcome, so they do not make the run
not leaderboard-comparable. Without a `/tokenize` count, the run's tokenizer plus a per-message
template allowance gives a conservative count: a prompt it puts clearly inside the window (its
count plus the 1024 fits in 8193) gets the 1024, and any other, like a prompt that could only be
estimated from its characters, is sent the 1024 and counted in `single_turn_rows_uncounted` or
`agent_requests_uncounted`, which make the run not leaderboard-comparable. Rows longer than the
window take part in `--eval-limit` and `GBENCH_SHARD` like the others: the single-turn part's share
of the limit is split over the rows sent and the rows not sent in proportion to their counts, and each group is sharded and
drawn stratified by sub-category as base draws the rows it sends. With the gemma-4 tokenizer the
longest single-turn prompt is 3,041 tokens, so only long agent conversations can reach the
window. With `--thinking` the window, like the 1024, is not applied
(`upstream_context_window.applied: false`).

A reply cut off by a budget before it finished (finish reason `length`, an empty reply after
reasoning included) is counted in `acebench_report.single_turn_replies_cut_off` and
`agent_replies_cut_off` and makes the run not leaderboard-comparable: with `--thinking`, every
such reply (the run's budget); without it, one that stopped below its official cap, because the
served context window left less room than LLMInfer's.
