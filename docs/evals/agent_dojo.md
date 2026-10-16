# agent_dojo

Canonical AgentDojo (ethz-spylab/agentdojo): prompt-injection security for **tool-calling
agents**. The agent runs real multi-step tool episodes across 4 task suites (workspace, travel,
banking, slack); attacker-controlled tool outputs carry injected instructions. gbench **delegates
to the agentdojo package's own suites, attacks and scorers**; it does not reimplement them.

## Pinned sources

The benchmark is the `agentdojo` package: **`agentdojo==0.1.35`** (the latest release on PyPI),
benchmark version **`v1.2.2`** (the latest the package ships), attack **`important_instructions`**.
The suite records the installed package version in `agentdojo_version`. The HF dump
`ffuuugor/agentdojo-dump` is not used.

## What it measures

For each suite, gbench runs what AgentDojo's suite-level functions run, one episode at a time
through AgentDojo's own per-task runners (`benchmark.py` `run_task_without_injection_tasks` /
`run_task_with_injection_tasks`, the functions `benchmark_suite_without_injections` and
`benchmark_suite_with_injections` loop over):
- **benign**: **utility** = fraction of user tasks the agent actually completes (headline
  `accuracy`);
- **under attack** (`important_instructions` attack): **utility under attack** and **attack
  success rate (ASR)** = fraction of (user x injection) pairs where the injected task was achieved
  (`security_results == True`; **lower is better**);
- as `benchmark_suite_with_injections` does for a non-DoS attack, each injection task run once as
  a user task (`injection_tasks_as_user_tasks`, a check reported apart from every number).

The pipeline is AgentDojo's no-defense pipeline (`SystemMessage(load_system_message(None))`,
`InitQuery`, `OpenAILLM`, `ToolsExecutionLoop`). ASR is reported **separately** as
`attack_success_rate` and is *not* folded into `accuracy`. `dimension_scores` carries per-suite
benign_utility / utility_under_attack / attack_success_rate.

## Reading the result

- `accuracy`: benign utility over all scored user tasks; `utility_under_attack` and
  `attack_success_rate` alongside, all three in `headline_metrics` (so the CSV and the console
  show them); `category_accuracy` holds each suite's benign utility, and `suites` names the
  suites that ran (`suites_available` every suite of the benchmark version).
- **Harness errors are never scored.** An episode that an endpoint failure (connection refused, a
  timeout, an HTTP error after `OpenAILLM`'s own 3 attempts) or any other exception ends is a
  harness error: left out of benign utility, utility under attack and attack success, counted by
  kind in `harness_errors` (`item_harness_errors` names the episodes), named in `status_reason`
  and `leaderboard_comparable_reason`; the run is `completed_with_errors`, and a run with no
  scored benign episode reports `accuracy` null (`status: failed`, `accuracy_withheld_reason`),
  never 0. Upstream stops the whole run on such an exception. An episode that outgrew the served
  context window (a 400 `context_length_exceeded` / "reduce the length of the messages") is
  scored as AgentDojo's benchmark functions score it (utility False, attack not successful),
  counted in `context_length_exceeded` and named in `leaderboard_comparable_reason`.
- The endpoint's replies are counted over every episode (informational; the score does not
  change): `replies_at_context_window` (finish_reason `length`: the reply reached the run's
  `--max-output-tokens`, or the end of the context window with
  `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0`; the field keeps its name), `empty_responses` (no content and no tool call)
  and `empty_after_reasoning` (of those, the ones with a reasoning field). Those that stopped on
  their own (not at `length`) are split by their generated token ids (once a reply shows the
  endpoint is vLLM, each request also asks for `return_token_ids`; sampling is unchanged):
  `answer_in_thinking` (no tool-call token: the answer inside the thinking channel, the model's
  outcome) with `answer_in_thinking_of` (every request of the episodes) and
  `answer_in_thinking_rate`; `tool_call_dropped` (the served tokenizer's tool-call token, Gemma 4
  `<|tool_call>` id 48: a native tool call the server's tool-call parser returned neither as a
  tool call nor as content, a server-parser artifact); and `empty_unclassified` (no token ids,
  e.g. the run's first reply). The last two are named in `leaderboard_comparable_reason`; see
  [answer_in_thinking.md](answer_in_thinking.md).
- `sample_traces`: every episode (benign, under attack, injection task as a user task) with its
  verdict (`utility`, `attack_succeeded`), its request count, the counters above, any harness
  error, and the messages AgentDojo's TraceLogger logged for it (strings clipped to 2,000
  characters). AgentDojo's log directory is a temporary directory removed when the run ends.
- `leaderboard_comparable` is `false`: gbench always sends a temperature, which upstream's
  `vllm_parsed` provider does not. `leaderboard_comparable_reason` lists what differs: a limited
  run, the temperature, a different
  installed `agentdojo` version, or a `GBENCH_AGENTDOJO_VERSION` / `GBENCH_AGENTDOJO_ATTACK` /
  `GBENCH_AGENTDOJO_MODEL_LABEL` override. `attack_model_label` records the label used.
- Sampling: gbench's rule for every suite: temperature 1.0 with `--thinking`, 0.0 without.
  `--temperature`, or `GBENCH_AGENT_DOJO_TEMPERATURE` (it takes precedence), replaces it in either
  mode. AgentDojo's `vllm_parsed` provider builds `OpenAILLM(client, model)` with its default
  temperature 0.0, and `OpenAILLM` sends `temperature or NOT_GIVEN`, which drops a 0.0; gbench puts
  the run's temperature on every request of the OpenAI client itself, so a 0.0 is sent as 0.0.
  `temperature` on the result is the value every request carried; `official_sampling` records
  upstream's setting (no temperature sent) and that the run did not match it.
- Thinking: the same wrapper puts the chat template's thinking switch on every request
  (`extra_body.chat_template_kwargs.enable_thinking`: true with `--thinking`, false without),
  which AgentDojo's OpenAI tools path never sends. The result records `thinking`,
  `thinking_applied` (true when every request carried the run's switch) and `request_shape`
  (requests counted by the value they carried). Before 2026-10-11 the switch was not sent, so a
  `--thinking` run changed only the temperature; those results are not comparable.
- Request timeout: gbench builds the OpenAI client with `timeout` = `base.request_timeout_s` of the
  served context window (AgentDojo sends no `max_tokens`; 33,968 s for a 262,144-token window) and
  `max_retries=0`, in place of the SDK's 600 s and 2 retries, which dropped and re-asked any turn
  needing more than 600 s of decode (a hidden cap, found by the 2026-10-10 investigation).
  AgentDojo's own retry of a failed request (3 attempts, not on HTTP 400/422) is unchanged; each of
  its attempts is now one HTTP attempt. Opening the TCP connection keeps its own 10 s bound
  (`httpx.Timeout(T, connect=10)`), so an unreachable endpoint fails an attempt in seconds rather
  than after T. The result records `model_client_timeout` (`{client, version, timeout_s,
  connect_timeout_s, retries, outer_attempts, source, replaces, hidden_cap_removed}`).
  The client talks to the endpoint directly, so a request the server accepts and then never
  answers is bounded only by T per attempt; catching such a hang within a run's time budget is
  left to the run's own stall alarm.

## Requirements

1. An OpenAI-compatible `/v1` endpoint **with vLLM tool-calling enabled**. A server gbench
   launches itself (no `--remote-endpoint`) gets it from `--tool-call-parser NAME` (gbench adds
   `--enable-auto-tool-choice --tool-call-parser NAME`; the name is the model's parser, e.g.
   `gemma4` for Gemma 4), and the CLI refuses a local run of this suite without that flag. A
   remote endpoint must be started with `--enable-auto-tool-choice --tool-call-parser <parser>`.
   gbench uses AgentDojo's `OpenAILLM` (native tools API); with tool-calling off, the server's
   parser swallows the model's function calls and every task returns empty.
2. **The `agentdojo` package** (REQUIRED; the suite hard-errors if it cannot be imported):
   ```bash
   pip install agentdojo==0.1.35
   ```
   It is also part of `pip install -e .[evals]`. The base install is enough. The optional `transformers` extra (a DeBERTa PI detector) is a
   *defense* and is not used by the canonical attack run. Installing the base package into a
   torch/vLLM env is safe: it only adds the langchain/langgraph stack, touching no torch /
   transformers / openai / pydantic pins (verified via `pip install --dry-run`).

No LLM judge and no external service: AgentDojo scores against its environments' ground
truth. When the endpoint needs a key, set `VLLM_API_KEY` (or `OPENAI_API_KEY`); gbench sends it
from its own process (AgentDojo's tools are in-memory functions with no shell or environment
access).

## Configuration

| Env var | Purpose |
| --- | --- |
| `GBENCH_AGENTDOJO_VERSION` | benchmark version (default `v1.2.2`; the package ships `v1` to `v1.2.2`); a different value makes the run non-comparable |
| `GBENCH_AGENTDOJO_ATTACK` | attack name (default `important_instructions`, the canonical targeted attack); a different value makes the run non-comparable |
| `GBENCH_AGENTDOJO_MODEL` | the model id sent to the endpoint (default: the served model id gbench passes, which the eval runner matched against `--models`). A value set here is checked against the endpoint's `/v1/models` listing first: an id the endpoint does not serve is `infra_required`, never sent. Recorded as `request_model_id` / `request_model_id_source` |
| `GBENCH_AGENTDOJO_MODEL_LABEL` | the prose model label the attack addresses (default `vllm_parsed`, i.e. "Local model"; set a `gemini-*` id for the Google-targeted variant); a different value changes the injected text and makes the run non-comparable |
| `GBENCH_AGENT_DOJO_TEMPERATURE` | per-suite temperature (default: 1.0 with `--thinking`, 0.0 without) |

## Run

```bash
pip install agentdojo==0.1.35
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-26B-A4B-it --evals agent_dojo \
       --max-output-tokens 65536 --suite-timeout 36000
```

AgentDojo's `OpenAILLM` sets no `max_tokens`. gbench's client wrapper sends the run's
`--max-output-tokens` as `max_tokens` with every request (`max_tokens_sent`,
`max_output_tokens_note`; `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0`: none, as upstream), a recorded
deviation. The value caps every turn.

## Notes / caveats

- **Runtime:** each (user x injection) pair is a full multi-step tool episode. The full run is
  large (about user_tasks x injection_tasks x 4 suites, with injections). AgentDojo runs the tasks
  one at a time (`execution` says so): `--batch-sizes` / `--sandboxes` do not change it.
  `--eval-limit N` is **N user tasks for the whole run**, never N per suite: the N are dealt
  over the suites (`sampling.allocate_limit`, seed `agent_dojo`: each suite gets one before any
  gets two, capped at its user tasks) and each suite's share is a seeded draw of its user tasks
  (`sampling.stratified_sample`, seed `agent_dojo:<suite>`); a suite whose share is 0 is not run.
  Each selected user task runs once without injections and once under attack with **one**
  injection task of its suite (a seeded order of the suite's injection tasks dealt to its user
  tasks in turn), plus AgentDojo's injection-task-as-user check for the injection tasks used, so
  `--eval-limit 5` is 5 user tasks = 10 scored episodes (and up to 5 checks), not 5 x the
  injection tasks. The run logs its planned workload on one line at its start (e.g.
  `[agent_dojo] --eval-limit 5: 5 sample(s) over 4 suite(s) (workspace 2, travel 1, banking 1,
  slack 1) x 2 attempts = 10 episodes`) and records it as `eval_limit_plan` (with the exact
  `episodes` and `injection_task_checks`). A limited run is a subset and is not
  `leaderboard_comparable`.
- **`--shard I/N`:** the unit is a user task. Every (suite, user task) in the package's order is
  listed, shard I takes every N-th starting at the I-th (before `--eval-limit`, which then caps
  within the shard), and each selected user task runs without injections and with every
  injection task of its suite. The result records `shard` and `item_results` (every benign,
  under-attack and attack-success verdict keyed `<suite>/<user task>/<injection task>`) with
  `item_harness_errors`; `gbench --merge` recomputes every metric, the harness errors and the
  counters over the union of the shards and joins their traces (`merge_shards`). A
  shard with no user task reports no score (`accuracy` null, `empty_shard`).
- **`--attempt-count`:** AgentDojo runs each task once; a larger value is recorded under
  `attempts` as not applied.
- **Attack label:** the `important_instructions` attack addresses the agent by a prose model name
  from `agentdojo.models.MODEL_NAMES`. gemma isn't in that map, so the default label is
  "Local model" (canonical for a locally-served model).
