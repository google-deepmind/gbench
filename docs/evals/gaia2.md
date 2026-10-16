# gaia2 setup

Canonical **GAIA2**, Meta's **Agents Research Environments (ARE)**
([github.com/facebookresearch/meta-agents-research-environments](https://github.com/facebookresearch/meta-agents-research-environments),
PyPI `meta-agents-research-environments`, HF dataset
[`meta-agents-research-environments/gaia2`](https://huggingface.co/datasets/meta-agents-research-environments/gaia2)).
A **stateful, multi-turn, time-sensitive** agentic benchmark: the agent acts inside an in-process
simulator (contacts / calendar / email / files with async, time-sensitive events), issuing tool
calls over many turns. Scoring is **hybrid**: deterministic "hard validation" (scripted/exact
checks) **plus a load-bearing LLM judge** for soft/semantic validation.

Pinned harness: `meta-agents-research-environments==1.2.0`, the current PyPI release (checked
2026-09-27; the repository's `main` has not changed the files gbench hooks below).

## What gbench changes (everything else, incl. judge PROMPTS, is upstream verbatim)
- **No docker-out-of-docker.** ARE is a pure in-process Python simulator, so the orchestrator runs
  `are-benchmark` directly (no `/var/run/docker.sock`, no sibling task containers).
- **Model wiring**, LiteLLM `local` provider: `--model openai/<served> --provider local --endpoint
  <gbench's endpoint relay>/v1` (verified against `litellm_engine.py`: provider `local` maps to
  `custom_llm_provider=None` + `api_base=endpoint`; the `openai/` prefix routes LiteLLM to the
  OpenAI-compatible endpoint).
- **The agent protocol of the ARE paper** (arXiv:2509.17158, section 4.1: "temperature set to 0.5,
  and generation limits of 16,000 tokens per turn", at most 200 steps; the stop sequences
  `<end_action>` and `Observation:`, which ARE's agent passes to its engine on every step; the
  paper's appendix B.5 adds them as custom stop sequences "for models that tend to continue past a
  single tool call (Claude, Kimi, Qwen)" and ran GPT-5 without them, so the paper applied them per
  model; gbench sends them for every served model, as ARE's `HuggingFaceLLMEngine` does). ARE 1.2.0's
  `LiteLLMEngine.chat_completion` sends none of it: its `completion()` call carries only the model,
  messages, `api_base` and `api_key` (the `ModelConfig` defaults `max_tokens=16384` /
  `temperature=0.5` are never forwarded), and ARE's main agent stops at 80 iterations
  (`ARESimulationAgent(max_iterations=80)`). gbench's hooked engine sends, with every agent request:
  the run's temperature (below), `max_tokens` 16000, the agent's stop sequences `["<end_action>",
  "Observation:"]` as `stop` (as ARE's own `HuggingFaceLLMEngine` sends them; LiteLLMEngine only
  cut the text after the reply came back), the chat template's thinking switch of the run, and a
  request timeout sized to the per-turn budget (LiteLLM's 600 s for 16,000 tokens, scaled with a
  larger budget); the main agent gets 200 steps. With `--thinking` the paper's cap, written for
  non-reasoning generation, is not applied: every turn is sent the run's budget
  (`--max-output-tokens`, else the thinking floor), per the token-limit policy
  (`base.resolve_token_budget`). The temperature is gbench's rule for every suite, not the
  paper's temperature 0.5: 1.0 with `--thinking`, 0.0 without. `GBENCH_GAIA2_TEMPERATURE`, else
  `--temperature`, replaces either; a temperature other than 0.5 is named in `leaderboard_comparable_reason`. Recorded in `agent_protocol`, `official_sampling`
  and `max_output_tokens_note`.
- **No server-side `stop` with `--thinking`; the answer still ends at the stop sequences.** The
  served stack matches stop strings over the whole generated text, the reasoning included (vLLM's
  `v1/engine/detokenizer.py` checks them on the detokenized output before the reasoning parser
  splits it), so on a thinking run a thought containing `Observation:` (natural in ReAct
  reasoning) or `<end_action>` would end the reply before any answer. A `--thinking` run
  therefore sends no `stop`; its agent request is streamed instead (`stream: true`,
  `stream_options: {"include_usage": true, "continuous_usage_stats": true}`: vLLM sends the
  reasoning and the answer in separate fields, with the usage so far in every chunk), and the
  client closes it as soon as the answer holds a stop sequence; vLLM aborts a request whose client
  went away, so the generation ends there on the server too. The answer, the measured generation
  time (the Time clock below) and the request then end where the paper's stop sequences end them,
  as on a run without thinking, and no text generated past the action reaches the clock. A stop
  string inside the reasoning is never matched. ARE's client-side split of the answer at the first
  stop sequence is kept on both runs. A streamed request still running when the request timeout
  runs out is closed and raises LiteLLM's `Timeout` the first time it runs over (a
  `model_endpoint` harness error, below); a non-streamed request that runs that long raises the
  same `Timeout`, but only after LiteLLM's client has re-sent it (`max_retries=2`).
  `agent_protocol.stop_sent_to_server`, `agent_protocol.stop_applied_by` (`server` or
  `client_stream`) and `official_sampling.stop_sent_to_server` record the choice; `agent_calls`
  counts `stop_sent`, `streamed`, `stopped_client_side` (streamed replies closed at a stop
  sequence), `ran_past_stop` (answers that still held a stop sequence: the server did not end the
  reply at it; named in `leaderboard_comparable_reason`, since the Time clock counted that
  generation) and `stopped_in_reasoning` (thinking-run replies that stopped with no answer and a
  non-empty reasoning field). On a `--thinking` run, replies cut at `max_tokens` are named in
  `leaderboard_comparable_reason` too.
- **The server-parser policy** (`base.recover_server_parser_text` and
  `base.server_parser_residue`, the gbench-wide rule; the launcher carries a copy of their pure
  functions, held equal to base's by tests). The served
  model runs behind the server's reasoning parser and tool-call parser. Every agent request asks
  for the generated token ids (`return_token_ids`, metadata only; LiteLLM drops a choice's
  `token_ids` and `stop_reason` when it converts a non-streamed reply, so a hook on its openai
  conversion keeps them, and a streamed reply carries them in its chunks). On a run without
  thinking, text the reasoning parser put in the reasoning field (a thought channel the model
  opened on its own, or the parser's first-feed rule) is the model's output
  (`server_parser_moved_text`); in either mode, a reply with no content, no reasoning and
  `completion_tokens` > 0 lost its text to the tool-call parser (`server_parser_dropped_text`: a
  native `<|tool_call>` on a request that declared no tools; ARE declares none and reads only the
  content, so a tool call the server returned is not seen either). Both are recovered in
  generation order, from the token ids decoded with special tokens kept by gbench's decoder
  (`DetokenizeRelay`, a thread of the gbench process on the bridge gateway that only turns token
  ids into text: the served tokenizer's `/detokenize`, else the run's tokenizer), or, for moved
  text without token ids, from the reasoning followed by the content, and handed to ARE as the
  reply. ARE parses it as it parses any reply, so a native tool call gets ARE's format-error
  feedback: the model's outcome. Deleted text that cannot be recovered (no token ids, or not
  decodable) is a harness error (`server_parser_text_lost`, below), and ARE gets no reply
  (LiteLLMEngine's `assert`, as upstream). A reply that kept its content or its reasoning is
  checked as well (`server_parser_residue`): the returned reasoning, then the returned content,
  are located in the generation rebuilt from the token ids, and generated text after them outside
  tag markup is text the tool-call parser deleted (`server_parser_dropped_text`). On a
  `--thinking` reply whose answer was a native tool call the parser dropped (reasoning kept,
  content empty), ARE gets the generation after the thought, the model's output, parsed as any
  reply; on a no-think reply whose native tool call followed some content, it gets the whole
  generation. Each event is counted (`server_parser_events`, `server_parser_recovered_from`,
  `agent_calls.server_parser_events`), named in `server_parser_note` and
  `leaderboard_comparable_reason`. The token ids of every reply are decoded, and each recovered
  call, and every call without content, keeps the text it was rebuilt from (`generation_text`,
  first 4000 characters; null only when the reply carried no token ids or they could not be
  decoded) with vLLM's `stop_reason` in `gbench_agent_calls.jsonl`. A `--thinking` reply that
  ended inside the reasoning channel with nothing generated after it stays the model's outcome
  (`stopped_in_reasoning`).
- **The datasets, read offline.** The image holds local copies of
  `meta-agents-research-environments/gaia2` (revision `78ea3bdbdeec2bdcd6afa5420915d8a22f23ed99`)
  and `gaia2_filesystem` (revision `132e26376f5e963bb59f64bcccdd02188cb08dee`) and runs with
  `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`: ARE's loader reads the local gaia2 copy (the
  launcher passes it as `--hf-dataset`; `load_dataset(<dir>, name=<config>, streaming=True)` reads
  the configs from its README, as it reads them from the Hub), and the filesystem app's fallback
  reads the local `gaia2_filesystem` copy (ARE's `DEMO_FS_PATH`). A run makes no Hugging Face Hub
  request: online, ARE's anonymous `paths-info` requests were rate-limited (HTTP 429), which made
  4 of 5 capability configs load no scenario. `dataset` on the result records the Hub ids, the
  revisions and the local paths.
- **The Time capability's clock.** The paper runs Time in the generation-time mode (section 4.3:
  the environment's time is paused while the model generates, then "incremented by the generation
  duration measured on the client side"; "instant" is its ablation). ARE's `measured` mode (the CLI
  default) advances the clock by the engine's `completion_duration`, which `LiteLLMEngine` never
  returned (`return res, None`), so the clock did not move while the model generated. The hooked
  engine returns the client-measured wall time of the request that answered, up to the end of the
  answer (for a streamed `--thinking` reply, up to the chunk that completed the stop sequence),
  with the reply's token counts.
- **The judge path.** ARE 1.2.0's `LiteLLMEngine` rewrites every reply with
  `.replace("False", "false").replace("True", "true")` (for the agent's JSON actions) and
  `create_judge_engine` always builds that engine, so a judge verdict `[[True]]` arrived as
  `[[true]]` and `LLMChecker`'s case-sensitive `success_str in response` never matched for the
  signature, sanity, cab and tone checkers (every email that is not character-identical to the
  oracle, every free-text message to the user, every messaging reply was rejected). Upstream:
  issue [#29](https://github.com/facebookresearch/meta-agents-research-environments/issues/29)
  and PR [#26](https://github.com/facebookresearch/meta-agents-research-environments/pull/26), both
  open, no release carries a fix. The current upstream judge, gaia2-cli
  (`gaia2-cli/cli/gaia2_cli/judge/engine.py` `create_litellm_engine`: the reply content verbatim,
  temperature 0; `gaia2-cli/core/gaia2_core/judge/checkers.py` `LLMChecker`: success / failure
  marker matched case-insensitively), has neither defect. gbench's hooked judge engine returns the
  reply verbatim and `LLMChecker` matches the marker case-insensitively, as gaia2_core does; ARE's
  prompts and every other check are unchanged, and the agent keeps ARE's rewrite (its JSON action
  parser needs it). Verified 2026-09-27 in the image with ARE 1.2.0 and a stub judge endpoint
  answering `- Evaluation: [[True]]`: without the hooks the signature, sanity, cab and tone checkers
  return `None`, with them `True`.
- **Judge model**: ARE's default judge is `meta-llama/Meta-Llama-3.3-70B-Instruct` (huggingface
  provider); gbench grades with its **Gemini cascade** (`GBENCH_JUDGE_MODELS`, default
  `base.GEMINI_CASCADE`; rounds and backoff of `base.judge_generate_cascade`; the requested model
  ignored) at temperature 0 (the paper's verifier setting; gaia2-cli sends 0) and no `max_tokens`
  (ARE sends none), through Gemini's OpenAI-compatible endpoint, via an in-container proxy
  (`gaia2_cascade_judge.py`, listening on `127.0.0.1` only): `--judge_provider local
  --judge_endpoint http://127.0.0.1:<port>/v1 --judge_model openai/gbench-cascade`. The judge
  endpoint and model are fixed; no knob points the judge anywhere else. The proxy gets the cascade
  from gbench (`GBENCH_JUDGE_MODELS`, `base.gemini_cascade_env()`) and does not start without it.
  The substitution is recorded on the result (`gemini_substitutions`), the Gemini model that
  answered every soft check in `judge_verdicts` and per run in `sample_traces` (`judge_models`),
  the proxy's answers by model in `gemini_models_answered`, and the substitution is named in
  `leaderboard_comparable_reason`.
- **`leaderboard_comparable` is always False**: the judge is the Gemini cascade, and gbench
  defaults to the standard phase (`run`, 3 runs per scenario) where the leaderboard runs the 3-phase
  `gaia2-run` with Pass@k.

## Metric
Headline **`accuracy`** = the per-capability equal-weight **macro success rate**, the GAIA2
Overall, computed by ARE's own statistics code (`report_stats.calculate_statistics`, the code that
writes `benchmark_stats.json`) over the scenario runs **without a harness error** (see below). ARE
writes its success rates in percent (0-100), so `accuracy` is that value as written;
`macro_success_rate` and `micro_success_rate` (per-scenario) are the same rates as fractions (0-1).
`headline_metrics` holds macro, micro, pass@k and pass^k (percent). The counts are scenario runs
(`count_unit`): `total_questions` every run, `effective_n` the validated runs outside harness
errors, `correct_answers` the successful ones among them. Also reported:
`pass_at_k_percent`/`pass_k_percent`, `total_scenarios`, `n_scenarios_run`,
`num_runs_per_scenario`, `no_validation_runs`, `n_timeout_runs` (runs that ended in ARE's
`ScenarioTimeoutError`: the model's outcome, as ARE counts it; ARE's own `exception_runs` is always
0 on this path, because ARE turns every exception into a failed run), the `per_capability`
breakdown, all over the same runs (a capability without a validated run has `success_rate`
`null`, never ARE's 0.0, which is kept as `success_rate_are_formula`, as the merged result
reports it); `metric` names this run's mode, capability configs and runs per scenario (with
where the count came from, `num_runs_source`); `are_reported` (ARE's own global rates and run counts as
`benchmark_stats.json` writes them, harness errors counted as failures, for provenance);
`gemini_usage` (the judge proxy's Gemini calls by serving model, `failed` for calls no model served)
and `judge_failures`. gbench never fabricates a scalar: a run that produces no
`benchmark_stats.json` hard-errors.

### Capabilities in the macro
ARE's CLI runs each capability config in a `try`/`except` (`benchmark/cli.py`, and
`gaia2_submission.py` for `gaia2-run`): a config that raised (for example a loader or scenario build
error) is logged, skipped, and the report is written for the others with exit code 0. ARE's macro
formula also gives a capability without a validated run a success rate of 0.0
(`report_stats._calculate_success_rate_stats`), although its run-level std/sem leave it out. The
launcher records every `run_dataset` call with its config, phase and outcome
(`gbench_configs.jsonl`; the configs that raised in `failed_configs`, each with its exception), and
gbench's recomputed macro averages the capabilities with a validated run
(`capabilities_without_validated_run` names the others; ARE's formula value is kept as
`macro_success_rate_are_formula` in `raw_summary`). The runner compares the scored capabilities
with the ones the run covers (`expected_capabilities`: the 5 configs of `run`, the 5 standard
configs plus `mini_agent2agent` and `mini_noise` of `gaia2-run`, or `GBENCH_GAIA2_CONFIG`); each
one missing is named with why in `missing_capabilities` and `leaderboard_comparable_reason` (its
config raised in ARE, ARE did not run it, every run of it was a harness error, or none of its runs
could be validated) and sets `status` to `completed_with_errors`: the macro is then over fewer
capabilities than the GAIA2 Overall. `configs_attempted` lists every config ARE ran.

## Records
- `sample_traces`: one record per scenario run: ARE's status, exception type and rationale, its
  harness errors and failed calls, its agent calls (`agent_calls`, replies without content,
  replies cut at `max_tokens`) and its soft checks with the Gemini models that answered them.
- `agent_calls`: over every agent request, the finish reasons, `null_content` and
  `empty_content` replies (as the server sent them), the server-parser events
  (`server_parser_events`, `server_parser_recovered_from`), replies that came with token ids
  (`token_ids_returned`), thinking-run replies
  that stopped inside the reasoning channel (`stopped_in_reasoning`), answers that still held a
  stop sequence (`ran_past_stop`), requests sent `stop` (`stop_sent`), streamed requests
  (`streamed`) and those closed at a stop sequence in the answer (`stopped_client_side`), replies
  cut at `max_tokens` (and those with no content), token and generation-time totals.
- `judge_verdicts`: the soft checks by checker and verdict (`true`, `false`, `no_verdict`) and the
  Gemini models that answered them.
- With a results directory (`GBENCH_RESULTS_DIR`, set by gbench's run manager),
  `gaia2_artifacts/run_<id>/` keeps, with every credential value redacted: `gaia2_summary.json`,
  `gbench_stats.json`, `benchmark_stats.json`, ARE's per-run rows `output.jsonl`, the launcher's
  `gbench_configs.jsonl`, `gbench_llm_call_failures.jsonl`, `gbench_agent_calls.jsonl` and
  `gbench_judge_verdicts.jsonl`
  (each soft check with the judge's reply, its first 600 characters), the proxy's
  `judge_calls.jsonl` (time, the request's sha256, the model that answered or the outage, the
  reply), `gemini_usage.json`, `harness_events.jsonl`, `cascade_judge.log`, and
  `gaia2_orchestrator.log` (the orchestrator's stdout and stderr: ARE's per-scenario results,
  timeouts and retry lines). ARE's per-scenario traces (`hf/`, `lite/`) stay in the workdir, which
  is removed after the run unless `GBENCH_GAIA2_KEEP_WORKDIR=1`; the entrypoint hands every file
  of the workdir to the runner's user first, so the removal succeeds (a path it could not remove
  is logged).

## Harness errors
A failure of the harness or of a service is never scored as the model's outcome. ARE itself turns a
judge call that raised into a failed validation (`are/simulation/validation/base.py`
`validate_current_turn`: `success=False`, rationale `Exception`) and an agent run that raised into
a failure (`scenario_runner.py`), and its agent loop logs a failed model call and goes on, so ARE's
own rates count all of these as the model's failures. gbench therefore runs ARE in a child process
of the launcher (`gaia2_run.py --are-hooked`) with its hooks:
- every LLM call ARE makes through LiteLLM (the agent's calls to the model endpoint and the judge's
  calls to the proxy) that raises is recorded with its scenario run (ARE's scenario id and run
  number for the thread, and the config and phase being run);
- every agent call (finish reason, vLLM's `stop_reason`, token counts, generation time, whether
  the content was empty, the server-parser event and the text it was rebuilt from) and every soft
  check (checker, verdict, the Gemini model that answered, the judge's reply) is recorded;
- when ARE writes its report, gbench classifies each scenario run from ARE's own per-run rows and
  recomputes the statistics without the harness-error runs (`gbench_stats.json`).

A scenario run is a harness error when (kinds in `harness_error_kinds`, per run in
`harness_error_runs`):
- `model_endpoint`: a model endpoint call of its agent failed on the endpoint's side (any HTTP
  status other than 400 and 422: 5xx, 401, 403, 404, 408, 429 and the other 4xx; or no HTTP
  status: a refused connection, a timeout, an error inside a streamed reply). Every agent request
  carries the per-turn cap and a timeout sized to it, so a timeout means the endpoint decoded the
  capped reply too slowly, not that the model generated without end. A rejection of the request's
  own content (HTTP 400/422, a context-window rejection) is the model's outcome here, as gbench's
  quality runner and toolbench count it; gbench does not apply one rule in every suite yet
  (base-run suites, bfcl, mcp_bench and the SWE family count a 400/422 as a harness error; see
  [env-knobs.md](env-knobs.md));
- `server_parser_text_lost`: the server's tool-call parser deleted an agent reply (no content, no
  reasoning, generated tokens) and its text could not be recovered from token ids (none came
  back, or gbench's decoder could not decode them);
- `judge`: a judge call of its validation raised (for example when no Gemini cascade model served
  it and the proxy answered HTTP 503);
- `judge_no_verdict`: a judge reply held neither verdict marker (a soft check without a verdict,
  which ARE's validation treats as a rejection);
- `validation_exception`: its validation raised otherwise;
- `run_exception`: its run raised (ARE turned the exception into a failure).

A run with a harness error is left out whatever ARE validated it as. A judge pairing that failed
(no verdict, or a judge call that raised) changes which agent event ARE's greedy event matching
(`validation/judge.py` `_match_agent_oracle_event`: each oracle event takes the first unmatched agent
event that passes, and the causality check reads the matched indices) gives each later oracle event,
and a failed agent call puts an error into the agent's history, so the error's effect on the outcome
cannot be told either way; keeping only the runs that still succeeded would also leave out failures
and not successes. The runs ARE validated as successes among them are counted
(`n_harness_error_success_runs`, by kind in `harness_error_success_kinds`) and named in
`leaderboard_comparable_reason`.

An agent reply with no content whose token ids decode to nothing (the model emitted nothing) is
the model's outcome, counted in `agent_calls`; so is every reply the server-parser policy
recovered. Runs ARE could not validate (`no_validation_runs`) are
left out of the rates by ARE itself. A scenario that ran past ARE's scenario timeout is the model's
outcome, as ARE and the leaderboard count it (`n_timeout_runs`). The harness-error runs are counted
(`n_harness_error_runs`), named in `leaderboard_comparable_reason`, left out of every rate and set
`status` to `completed_with_errors`; so do `no_validation_runs` and judge calls the proxy recorded as
served by no Gemini cascade model (`judge_failures`, from its `harness_events`). A failed LLM call
that cannot be tied to a scenario run, or no validated run left, is a hard error
(`infra_required`): no rate is published rather than one that scores harness failures as the
model's. The message names the harness-error kinds with what to check for each (the model
endpoint for `model_endpoint`, `GEMINI_API_KEY` and the judge cascade for `judge` /
`judge_no_verdict`, `return_token_ids` and `/detokenize` for `server_parser_text_lost`, the
orchestrator log for a raised validation or run), the capability configs that raised
(`capability: error`) and the directory the run's artifacts were kept in.

## Network and credentials
The ARE agent's tool calls run in the orchestrator container, so it follows the rule for
containers that run model-controlled actions:
- It runs with **`--network bridge`**, and only after `container_guard.require_metadata_blocked`
  found the cloud metadata server blocked on that network. It needs no name resolution: it reaches
  gbench's relays at the network's gateway address, and reads the datasets offline. See
  [toolchains.md](toolchains.md#cloud-vms-the-metadata-server) for the host firewall rule.
- It holds **no real key or token**. The model endpoint is reached through gbench's endpoint relay
  (`swebench_common.EndpointRelay`, a thread of the gbench process on the bridge gateway, which adds
  `GBENCH_GAIA2_MODEL_API_KEY`); the Gemini API through gbench's Gemini relay
  (`gemini_relay.GeminiRelay` on the same gateway, `POST /v1beta/openai/chat/completions` only,
  which adds the Gemini key). The relay's per-run placeholder (as `GEMINI_API_KEY` /
  `GEMINI_API_KEYS`) and the relays' addresses go into the container **by name** (`docker run -e
  NAME`, the value copied from the runner's environment), so nothing is on a command line. No HF
  token goes in, and none is needed (the datasets are read offline from the image). gbench's
  token-id decoder (`DetokenizeRelay`, `GBENCH_DETOKENIZE_URL`) listens on the same gateway; it
  holds no credential and only turns token ids into text.
- Inside the container only the entrypoint and the judge proxy hold the placeholder and the relay's
  address: the entrypoint starts the launcher (and with it ARE and the agent) with
  `GEMINI_API_KEY`, `GEMINI_API_KEYS`, `GBENCH_GEMINI_RELAY_URL` and `GEMINI_OPENAI_BASE_URL`
  removed from its environment. The proxy refuses to start when a key is not a relay placeholder or
  the relay's URL is missing. The runner redacts credential values from the orchestrator output it
  keeps or quotes.
- `GEMINI_OPENAI_BASE_URL`, where the relay sends the Gemini key, must be the Gemini API
  (`https://generativelanguage.googleapis.com/...`) or a loopback address: any other value is a
  hard error at gate time (also with `GBENCH_GAIA2_SKIP_KEY_VALIDATION=1`).
- The judge proxy listens on `127.0.0.1` inside the container's own network namespace, so no host
  port is taken: `GBENCH_GAIA2_JUDGE_PORT` (default 18790) only picks the in-container port.

## Modes
- **`run`** (default, `GBENCH_GAIA2_MODE=run`): `are-benchmark run` with **no `--hf-config`** runs
  the standard phase of all 5 capability configs (adaptability / ambiguity / execution / search /
  time), `--num_runs` runs per scenario (ARE's default 3; `GBENCH_GAIA2_NUM_RUNS` sets it, else an
  `--attempt-count` above 3 raises it to that value, and a smaller one never cuts the 3: the
  result's `attempts` records the runs and their source), and writes one aggregated report. `GBENCH_GAIA2_CONFIG=mini` (or `demo`) narrows it for a smoke.
  `--eval-limit N` is **N scenarios for the whole run**, never N per config (ARE's `--limit` is
  per config, so passing N on would run 5 x N): gbench deals the N over the run's configs
  (`sampling.allocate_limit`, seed `gaia2`: each config gets one before any gets two, so
  `--eval-limit 5` is 1 scenario of each capability and `--eval-limit 3` runs 3 of the 5 configs)
  and passes the shares as `GAIA2_CONFIG_LIMITS`; the launcher's hook keeps a seeded draw of each
  config's share of its (shard's) sorted scenario ids (its copy of `sampling.stratified_indices`,
  seed `gaia2`), not the first scenarios of the stream, and a config whose share is 0 is not run
  (ARE skips it; the result lists it under `limit_skipped_configs` and the macro averages the
  configs that ran). The result records the draw (`selection`: `config_limits`, `limit_draw` and
  each config's scenario ids), gbench refuses an image whose launcher does not report it (rebuild
  the image), and `n_scenarios_run` records the count. Each scenario still runs ARE's
  `--num_runs` runs (repeats, not extra samples). The run logs its planned workload on one line
  at its start, e.g. `[gaia2] --eval-limit 5: 5 sample(s) over 5 config(s) (adaptability 1,
  ambiguity 1, execution 1, search 1, time 1) x 3 attempts = 15 scenario_runs`, and records it as
  `eval_limit_plan`. `--eval-limit` applies the same way in `gaia2-run` mode, over its 5
  capability configs and the mini config (each mini scenario runs in both the Agent2Agent and the
  Noise phase: `phase_runs_per_mini_scenario`; the logged line counts that second phase too, e.g.
  `... x 3 attempts = 18 scenario_runs + 3 (the mini scenario(s)' second phase) = 21
  scenario_runs`). `--eval-limit` always wins over `GBENCH_GAIA2_LIMIT` (below): with
  `--eval-limit` set the knob is ignored with a warning. Only without `--eval-limit`, and only in
  `run` mode, `GBENCH_GAIA2_LIMIT` N runs at most N scenarios of each config; the plan line then
  reads `[gaia2] GBENCH_GAIA2_LIMIT N per config (no --eval-limit): at most ...`. gbench passes **`--agent`** (`GAIA2_AGENT`,
  default `default` - ARE's only registered agent) explicitly: `are-benchmark run` defaults
  `--agent` to `None`, which runs scenarios **agent-less** (the model-under-test is never invoked)
  yet still returns a real macro=0.0 - a fabricated score. gbench hard-errors (`infra_required`) if
  a run validated scenarios but no per-run trace shows any agent LLM usage, so an agent-less no-op
  can never be published as a 0%.
- **`gaia2-run`** (`GBENCH_GAIA2_MODE=gaia2-run`), the full canonical leaderboard submission: 3
  phases (Standard / Agent2Agent-mini / Noise-mini) x 3 runs per scenario (ARE's
  `gaia2_submission` hard-codes `NUM_RUNS = 3`; `GBENCH_GAIA2_NUM_RUNS` and an `--attempt-count`
  above 3 are ignored here with a warning), with Pass@k / Pass^k. Very heavy; opt-in.
- **Concurrency.** ARE runs `GBENCH_GAIA2_MAX_CONCURRENT` scenarios at once, else the suite's
  concurrency (`--eval-concurrency`, else `--batch-sizes`), never ARE's own default (its CPU
  count): the Time capability's clock advances by the measured generation time, which depends on
  the load on the endpoint. The result records `max_concurrent_scenarios` and its source.

## Sharding (`--shard I/N`)
gbench validates the shard and passes it to the container (`GAIA2_SHARD`). ARE's CLI has no shard
flag, so the launcher (`gaia2_run.py install_shard_hook`) wraps the dataset load ARE streams its
scenarios from: each capability config keeps shard I of N of its scenario ids (the ids sorted,
every N-th starting at the I-th), in the stream's own order, before the `--eval-limit` draw
(above), which ARE's `--limit` then runs. Every phase and run of a kept scenario runs as upstream.
The result records `shard` and the per-config selection (`shard_selection`: scenarios and selected
counts); a shard with no scenario of any configured capability reports no score (`accuracy` null,
`empty_shard`) without starting ARE. `gbench --merge` recomputes ARE's statistics
(`report_stats.calculate_statistics`: each capability's success rate over its validated runs, the
macro over the capabilities with a validated run, the micro, Pass@k and Pass^k) over the union of
the shards' scenario runs (`sample_traces`) without a harness error (`merge_shards`); the shards'
harness-error runs are counted, and each shard's `--eval-limit` draw is kept under
`selection_by_shard` (no shard's `selection` stands for the union). An image that does not apply the shard is refused
(`infra_required`).

## Prerequisites (the suite HARD-ERRORS via `infra_required`, never skips)

### 1. Build the LOCAL orchestrator image (gbench never pulls)
```bash
docker build -t gbench-gaia2:are-1.2.0 -f gbench/docker/gaia2.Dockerfile gbench/docker
```
ARE 1.2.0 installs against `gbench/docker/gaia2.constraints.txt` (the resolved set of the checked
image: ARE hard-pins about 22 packages, the file pins the rest of the closure), and the build fails
unless the interpreter holds exactly that set; the Python set and the Debian packages (which float
with the Debian archive) are recorded in `/opt/gbench`.

The image carries the label `org.gbench.gaia2-harness-errors=7` (its launcher applies the agent
protocol with no server-side `stop` on a `--thinking` run, whose streamed agent request it closes at
the first stop sequence in the answer, the upstream judge path and the measured generation time,
counts every model endpoint status but 400/422 as an endpoint failure, applies the server-parser
policy (the text a tool-call parser deleted after a reply's reasoning or content included),
applies `--shard` and `--eval-limit` dealt over the configs (`GAIA2_CONFIG_LIMITS`, since label
7), leaves harness-error runs out of the rates, records every capability config's
outcome, averages the macro over the capabilities with a validated run, and reads the datasets
offline);
gbench refuses an image without it. The build checks that the pinned LiteLLM still hands the hook
the raw stream (`CustomStreamWrapper(completion_stream=...)`) and still converts openai replies
through the `convert_to_model_response_object` global the token-id hook wraps.
Bakes the pinned ARE simulator + the **public** cc-by-4.0 GAIA2 datasets (`gaia2` and
`gaia2_filesystem` at the pinned revisions above, as local copies, no HF token; the build fails
unless each of the 7 gaia2 configs loads offline with rows) + the gbench adapter (launcher, Gemini judge proxy with the
cascade module it shares with wildclawbench, entrypoint). ARE hard-pins ~22 deps with `==`
(numpy/litellm/mcp/pydantic) that would break the vLLM stack, which is exactly why it lives in this
isolated container, never in the gbench serving environment.

### 2. The host network rule
The metadata server must be blocked for docker's bridge network (gbench checks it with a probe
container before the run and never changes the firewall itself):
```bash
sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
```
The bridge network must reach the host's gateway address (where the relays and the token-id
decoder listen); the run itself needs no internet access from the container.

### 3. Keys + model
- **`GEMINI_API_KEY`** (or `GEMINI_API_KEYS`, required): GAIA2's soft/semantic validation is done
  by an LLM judge, which gbench runs on its Gemini cascade. The key is **live-validated** at gate
  time, together with every model of the cascade (a definitive auth rejection, or a model the API
  does not serve, hard-errors; set `GBENCH_GAIA2_SKIP_KEY_VALIDATION=1` for setups where the
  Gemini API cannot be reached at gate time). The key stays with the gbench process (the relay).
- A **served model at `/v1`** reachable from the gbench host (the endpoint relay forwards to it);
  its key, if it needs one, in `GBENCH_GAIA2_MODEL_API_KEY` (else the endpoint key every pillar
  sends, `VLLM_API_KEY`, else `OPENAI_API_KEY`). Serve it with its chat template and its reasoning
  parser: start a remote endpoint with vLLM's `--reasoning-parser`; a server gbench launches
  itself runs one only when `--reasoning-parser NAME` is given (gbench sets none on its own). Every
  agent request sends the run's thinking switch (`chat_template_kwargs.enable_thinking`) and asks
  for the token ids (`return_token_ids`, a vLLM request field); text the server's parsers take out
  of a reply is recovered as the model's output (the server-parser policy above), and the
  decoder uses the served tokenizer's `/detokenize` (vLLM serves it), else the run's `--tokenizer`.

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals gaia2 --max-output-tokens 16000
```

## Environment knobs
| Env var | Default | Meaning |
|---|---|---|
| `GBENCH_GAIA2_IMAGE` | `gbench-gaia2:are-1.2.0` | orchestrator image tag |
| `GBENCH_GAIA2_MODE` | `run` | `run` (standard phase of all 5 capabilities) or `gaia2-run` (3-phase xN) |
| `GBENCH_GAIA2_CONFIG` | (all 5) | single capability config, or `mini`/`demo` (smoke) |
| `GBENCH_GAIA2_LIMIT` | (none) | scenarios per config (a seeded draw of each config's scenario ids, `run` mode only); applies only without `--eval-limit`: with `--eval-limit` set (N scenarios for the whole run) it is ignored with a warning; a positive integer, anything else (`0`, `-1`, `abc`) is refused before any container starts |
| `GBENCH_GAIA2_NUM_RUNS` | 3 (or `--attempt-count` when larger) | runs per scenario in `run` mode (Pass@k/Pass^k); `gaia2-run` always runs 3 (ARE's `gaia2_submission` hard-codes `NUM_RUNS = 3` for every phase), so there it and an `--attempt-count` above 3 are ignored with a warning |
| `GBENCH_GAIA2_MODEL_ENDPOINT` | (derived) | override the model `/v1` URL the endpoint relay forwards to |
| `GBENCH_GAIA2_MODEL_API_KEY` | `VLLM_API_KEY`, else `OPENAI_API_KEY` | the model endpoint's key (the endpoint relay adds it; it never enters the container) |
| `GBENCH_GAIA2_JUDGE_PORT` | 18790 | port of the in-container judge proxy (on `127.0.0.1` in the container's own network namespace) |
| `GBENCH_GAIA2_KEEP_WORKDIR` | (unset) | `1` keeps the run's workdir (ARE per-scenario logs and traces) for audit |
| `GBENCH_GAIA2_SCENARIO_TIMEOUT` | ARE default | passthrough tuning |
| `GBENCH_GAIA2_MAX_CONCURRENT` | the suite's concurrency | scenarios ARE runs at once (recorded as `max_concurrent_scenarios`) |
| `GBENCH_GAIA2_TIMEOUT_S` | 86400 | orchestrator wall-clock cap (a timeout is infra_required and the container is reaped) |
| `GBENCH_GAIA2_SKIP_KEY_VALIDATION` | (unset) | `1` to skip the live GEMINI key ping (air-gapped) |
| `GBENCH_GAIA2_TEMPERATURE` / `--temperature` | 0.0; 1.0 with `--thinking` (gbench's rule; the ARE paper's 0.5 is recorded) | the agent's temperature; a value other than 0.5 is named in `leaderboard_comparable_reason` |
| `GBENCH_JUDGE_MODELS` / `GBENCH_JUDGE_MODEL` / `GBENCH_JUDGE_CASCADE_ROUNDS` / `GBENCH_JUDGE_BACKOFF` | gbench defaults | judge cascade (shared with all judged suites; passed into the container) |
| `GEMINI_OPENAI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` | where the Gemini relay sends the judge's calls; only the Gemini API or a loopback address is accepted |

Removed: `GBENCH_GAIA2_JUDGE_ENDPOINT` and `GBENCH_GAIA2_JUDGE_MODEL` no longer change anything
(setting one logs a warning): ARE's judge is always the gbench Gemini cascade behind the
in-container proxy.

## What hard-errors (`infra_required`, never a skip / never a fake number)
- Docker CLI / daemon not reachable, or the orchestrator image `gbench-gaia2:are-1.2.0` not built, or built
  before the current launcher (no label `org.gbench.gaia2-harness-errors=7`).
- `GEMINI_API_KEY` unset, or rejected by a live validation ping, or a model of the Gemini cascade
  the API does not serve (skip the ping with `GBENCH_GAIA2_SKIP_KEY_VALIDATION=1` where the Gemini
  API cannot be reached at gate time).
- `GEMINI_OPENAI_BASE_URL` set to something other than the Gemini API or a loopback address.
- The metadata server reachable from docker's bridge network; the bridge network has no gateway;
  a relay or the token-id decoder cannot listen on it; the model endpoint is not an http(s) URL.
- The orchestrator produced no summary (a harness failure, `status:error`, not 0%): no
  `benchmark_stats.json`, or the orchestrator's own checks failed (no Gemini placeholder in the
  container, the judge proxy not starting), with the orchestrator's last output in the message.
- The orchestrator exceeded `GBENCH_GAIA2_TIMEOUT_S`.
- The orchestrator wrote no harness-error accounting (`gbench_stats.json`).
- A failed LLM call could not be tied to a scenario run.
- The agent was never invoked, not one agent request got a reply from the model endpoint (no
  request then carried text the model had generated, so not even a 400/422 rejection is the
  model's outcome), or no scenario run produced a validated score outside harness errors.

## Leaderboard comparability
Always **`leaderboard_comparable=False`**: the judge is the gbench Gemini cascade (not ARE's
default Llama-3.3-70B), and gbench defaults to the standard phase only (`run`, x3 runs per
scenario) where the canonical leaderboard runs the 3-phase `gaia2-run` with Pass@k (and entries
are maintainer-audited). Both are named in `leaderboard_comparable_reason`, with the judge path
(the verdict read verbatim and matched case-insensitively), a subset, an agent temperature other
than 0.5, a `--thinking` budget in place of the paper's cap, answers that still held a stop
sequence, `--thinking` replies cut at `max_tokens`, the server-parser events, the harness errors
above and the capabilities the macro leaves out when present.
