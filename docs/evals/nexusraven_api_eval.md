# nexusraven_api_eval

NexusRaven API evaluation (Nexusflow;
[nexusflowai/NexusRaven](https://github.com/nexusflowai/NexusRaven)): single-turn, zero-shot
function calling on real APIs. For each query the model must call one function from the given
options with the right arguments. gbench runs the repo's `raven/eval/evaluator.py` the way its
chat-model scripts do (`scripts/evaluate_gpt4.sh`: llm `openai_gpt4`, agent `SIMPLE`).

The dataset is licensed CC-BY-NC-4.0 (non-commercial).

Renamed from `nexus_function_calling`; the suite runs the NexusRaven_API_evaluation tasks.
`--evals nexus_function_calling` still runs this suite, with a deprecation warning;
`GBENCH_NEXUS_FUNCTION_CALLING_TEMPERATURE` is still read as a deprecated alias, and
`--eval-limit` draws the same queries as before the rename.

## Dataset (pinned)
`Nexusflow/NexusRaven_API_evaluation` at revision `85c7c326fcfb9c6587daabd9582e7cfaddcc65bd`:
all five tasks the repo evaluates, 339 queries: `cve_cpe` (78), `emailrep` (38), `virustotal`
(151), `toolalpaca` (51) from `standardized_queries` + `standardized_api_list`, and `toolllm`
(21) from `raw_queries`. Protocol ported from `nexusflowai/NexusRaven` at commit
`f32882763bb3e220722883821919c14abdd3d156` (which pins `langchain==0.0.294`).

## Protocol
- **Functions**: each task's API list is exec'd into Python functions with
  `raven.utils.build_functions` (required arguments first, `JSON:` defaults decoded, body
  `return (name, locals())`); `toolllm` uses `ToolLLMEvaluationDataHelper.build_functions` on the
  raw queries.
- **Prompt**: LangChain 0.0.294's structured-chat agent (`STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION`,
  `max_iterations=1`) with the query's context functions as tools, turned into the single prompt
  string `OpenAIChat` sends: one user message `System: <PREFIX, one
  "name: name(signature) - docstring, args: {...}" line per tool, FORMAT_INSTRUCTIONS,
  SUFFIX>\nHuman: <query>\n\n`, with stop `["Observation:"]`. As in the original prompts, the
  args schema appears with doubled braces.
- **Reply handling**: LangChain's `StructuredChatOutputParser`. A fenced JSON action naming a
  context tool runs that tool (pydantic-v1 validation of `action_input`, which coerces types such
  as `"10"` to `10` and drops unknown keys, then the function); a `Final Answer` string, or a reply
  with no fenced block, is parsed as a Python call and applied to the function; an action naming
  an unknown tool fails. A reply the parser rejects gets LangChain's one `OutputFixingParser`
  repair call (the `NAIVE_FIX` prompt, no stop sequence), and the repaired reply is parsed once
  more.
- **Scoring**: correct iff the function name and the **full bound argument dict** (defaults
  included) equal those of the reference call, as `evaluator.py` compares them. Inline reasoning
  tags are removed from the reply before it is parsed (and before it goes into the repair
  prompt). An argument left
  at its default matches a reference that also leaves it at the default; a falsy value such as
  `verbose=False` against a default of `None` does not match.
- The port was checked against real LangChain 0.0.294 plus the upstream evaluator on
  339 x 12 synthetic replies (exact action, stringified values, dropped/extra arguments, final-answer
  call, bare call, malformed-then-repaired, twice malformed, unknown tool, list blob, string
  input): identical verdicts, identical prompts and identical repair prompts.

## Headline and how to read the result
- `domain_accuracy`: the two numbers the NexusRaven repo publishes for its zero-shot evaluation
  (README figure `docs/Zero-shot_Evaluation.png` at the pinned commit): **Generic Domain**
  (`toolalpaca`, `toolllm`) and **Cybersecurity Domain** (`cve_cpe`, `emailrep`, `virustotal`),
  each the unweighted mean of its tasks' accuracies. That is how the figure's bars are formed:
  its GPT-4 Cybersecurity bar, 78.12, is the mean of GPT-4's `cve_cpe` 64.10 and `virustotal`
  80.80 (the README's single-attempt figure) and `emailrep` 34/38; a count-weighted mean fits no
  whole number of correct answers. These are the comparable numbers.
- `accuracy` (headline): the unweighted mean of the two domain accuracies, gbench's single number
  for the suite. When a domain that ran has no scored query (every one a harness error, below),
  it has no accuracy and neither has the mean: `accuracy` is null with
  `accuracy_withheld_reason`.
- A query whose request, or whose `OutputFixingParser` repair call, got no reply from the served
  endpoint (failed, timed out, or over the context window) is a harness error, never the model's
  answer: it is left out of every number here (`per_task`, `domain_accuracy`,
  `category_accuracy`, `accuracy`; `effective_n` counts the scored queries), counted
  (`scoring_excluded`, `output_fixing.harness_errors` for the repair calls) and named in
  `status_reason` (status `completed_with_errors`).
- `per_task.<task>`: `accuracy` and `n`, the per-task accuracy `evaluator.py` prints (to three
  decimals; two here).
- `category_accuracy.<task>`: `correct`, `total` and `accuracy` per task (after the repair call),
  the same numbers as `per_task`.
- `output_fixing`: how many replies needed the repair call, how many of those ended correct and
  how many repair calls got no reply (`harness_errors`).
- `leaderboard_comparable` is always `false`: NexusRaven publishes the two domain numbers, so their
  mean is a gbench aggregate. The reason also lists `--eval-limit`, a shard, a temperature other
  than 0 (the SIMPLE-agent scripts sample at 0), decoding penalties, `--attempt-count > 1` or
  queries whose request or repair call got no reply (harness errors) when present. `official_sampling` records `evaluator.py`'s setting, the temperature this run used and
  whether it matched.

## Requirements
- A running OpenAI-compatible `/v1` endpoint (plain text generation; no tool-call parser needed).
- Network and the `datasets` client on the first run (HF Hub); cached afterwards. A dataset that
  cannot be loaded stops the suite with an `[INFRA REQUIRED]` error.
- `pydantic` 2.x (it ships the `pydantic.v1` API used for the tool schemas); already a gbench
  dependency. LangChain itself is not needed.
- No API key and no LLM judge.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals nexusraven_api_eval --max-output-tokens 1024
```
Add `--eval-limit N` for a quick check (stratified by task; flagged in the result).

The run logs its planned workload in one line at its start (for example `[nexusraven_api_eval]
--eval-limit 5: 5 sample(s) x 1 attempt = 5 generations`) and records it in the result as
`eval_limit_plan` (`limit`, `samples`, `per_setting`, `attempts_per_sample`, `generations`,
`note` for the OutputFixingParser repair calls, `repair_calls_at_most` (one per generation) and
`repair_calls`, the repair calls that ran). Under `--shard` the repair calls of a shard all run on
that shard, attached to the queries they repair.

Sampling: temperature 0.0 without `--thinking`, 1.0 with it, for the query and the repair call
alike (gbench's rule for every suite). `evaluator.py` `build_llm` runs the OpenAI chat models as
`OpenAIChat(temperature=0)`, which sends no other sampling field; that setting is recorded under
`official_sampling`. `--temperature`, or `GBENCH_NEXUSRAVEN_API_EVAL_TEMPERATURE` for this
suite alone (legacy alias `GBENCH_NEXUS_FUNCTION_CALLING_TEMPERATURE`; it takes precedence),
overrides it and is recorded under `official_sampling`. No `max_tokens` is sent, with or without
`--thinking`, as `OpenAIChat` sends none (LangChain 0.0.294 `llms/openai.py`: its request is the
model plus `model_kwargs`): the server grants the room left in the context window,
`--max-output-tokens` is not applied (`max_output_tokens_note` says so), and a reply cut off at
the end of the context window is counted in `truncated_responses`, as it would be cut off
upstream.
