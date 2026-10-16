# Preflight, smoke test and parallel suites

Three flags for running a large sweep (all 84 built-in suites, or any `--evals` selection)
against one served model:

| Flag | What it does | Calls the model | Typical wall time |
|---|---|---|---|
| `--preflight` | Checks what every selected suite needs, prints a PASS/FAIL table | no | about 30 s for `--evals all` |
| `--smoke` | The preflight, then one sample of every suite end to end, PASS/FAIL per suite | yes, one sample per suite | at most `GBENCH_SMOKE_DEADLINE_S` (default 65 min), usually the slowest suite's one sample (42.4 min for the 84 suites on 2026-10-11, see [Wall time](#wall-time-and-the-defaults)) |
| `--parallel-suites N` | Runs a sweep (a validation run at `--eval-limit 5`, or a full run) as a pool of up to N suite processes | yes | bounded by the slowest suite |

Why they exist, from the 2026-10-09/10 runs of the 84 suites at `--eval-limit 5`:

- The serial sweep took 783 min. Five of its seven hard failures were missing keys, credentials or
  images that a check before generation names in seconds; `mcp_bench` and `mcp_atlas` failed in
  0.0 min, at the end of their lane.
- Three hand-made lanes still left one lane at 0 of 28 suites after 3 h 29 min behind
  `swe_bench_live`, and `gaia`, `gaia2` and `wildclawbench` collided on ports 18789-18795 when two of
  them ran at once.
- `skillsbench` recorded `thinking_applied: false` under `--thinking` (the switch never reached the
  server), which only reading its result showed.

None of the three flags changes a suite's protocol: `--thinking`, `--max-output-tokens`, the
temperature rule (1.0 with `--thinking`, 0.0 without; `GBENCH_<SUITE>_TEMPERATURE` and
`--temperature` override), the datasets, scaffolds and protocols are the run's own. What a pool
does change is the load: other suites share the server, and the search suites share the Gemini
search rate (each gets `GBENCH_SEARCH_MAX_RPS` / k). A suite whose score depends on wall-clock time
(agent loops under wall-clock budgets, fixed per-request or per-lookup deadlines: wildclawbench's
57 s and mcp_atlas's 60 s search waits) can score differently under a pool than alone; each result
records the concurrency it ran under (`pool`, below). `--smoke` adds `--eval-limit 1`, ends agent
episodes early and has a hang rule and a deadline, so its results are never scores (below).

## Prerequisites

- A served OpenAI-compatible endpoint: `--remote-endpoint <URL>/v1` and `--tokenizer <id>`.
  `--smoke` and `--parallel-suites` refuse to run without one (each suite's process would
  otherwise start its own vLLM server on the same GPUs). `--preflight` runs without it and then
  skips only the endpoint checks.
- `--max-output-tokens` (required for every eval run).
- Docker, the local images, checkouts, datasets and keys each suite's own page in
  [docs/evals/](evals/) lists: `--preflight` tells you which are missing. The Gemini key
  (`GEMINI_API_KEY` or `GEMINI_API_KEYS`) is needed by the suites that grade, search or simulate
  users with gbench's Gemini cascade.
- The server must listen on an address Docker's bridge gateway reaches (for example `0.0.0.0`)
  for the suites whose containers reach the endpoint directly; the preflight checks it.
- Nothing else to install: the three flags are part of gbench and use only the standard library,
  Docker's CLI and the suites' own checks.

## `--preflight`

```bash
gbench --preflight --evals all --remote-endpoint http://127.0.0.1:8000/v1 \
  --tokenizer google/gemma-4-26B-A4B-it --thinking --max-output-tokens 65536
```

Every check runs at once (one thread each), except that the suites of one pool family (the
table under `--parallel-suites`: a shared harness root, image or ports) are checked one after
another: on 2026-10-11 bfcl's and bfcl_v4_agentic's checks wrote and removed the same probe file
in the BFCL project root at once, and bfcl_v4_agentic failed "not writable" in 2 of 3 runs. No
check sends a model request or builds an image.

Per suite:

- The suite's own prerequisite check, called exactly as its runner calls it first:
  `check_<suite>_prerequisites` (bigcodebench, multipl_e, ojbench, spider2, coco_caption,
  complexfuncbench, gaia, gaia2, mcp_atlas, mcp_bench, skillsbench, terminal_bench, toolbench,
  wildclawbench, putnam, putnam_formal, swe_lancer, ui_control_osworld), the MathArena image check
  (aime, hmmt), the checkout and image checks of aider_polyglot, livebench, omnidocbench,
  gorilla_apibench and api_bank, the bubblewrap sandbox (codeforces, cruxeval, scicode; lcb also
  its grader self-check), the SWE harness and agent image checks in the protocol the run will use
  (copilot_bench_swe, swe_bench_multilingual, multi_swe_bench, swe_bench_live per tab,
  swe_bench_pro V2 or the v1 engine), the BFCL harness and bridge checks against the endpoint
  (bfcl, bfcl_v4_agentic), and the tau2 image and role providers (tau2, tau3).
- A suite without a check of its own: the local image its module names, if any
  (`docker image inspect`, never pulled).
- The suites that use the Gemini cascade: the key is set (presence only; the suites whose own check
  validates the key with the Gemini API's model list do so, which spends no generation quota).
- `swe_bench_live`'s Windows tab needs a Windows host: on Linux it is listed as a note, because the
  suite reports that tab `infra_required` in its own entry and runs the other tabs. Every other
  missing tab is a FAIL.

Global:

| Check | FAIL when | Stops `--smoke` |
|---|---|---|
| disk free at `--results-dir` | below `GBENCH_PREFLIGHT_MIN_DISK_GB` (default 100) GB | yes |
| docker daemon | `docker info` fails | yes |
| endpoint from this host | `GET <endpoint>/v1/models` fails or lists no model | yes |
| endpoint from a bridge container | the same request from an existing local image (`--network bridge`, `host.docker.internal` mapped to the bridge gateway) fails | no (the suites that need it fail their own checks) |
| Gemini key | unset while a selected suite needs it | no (those suites fail) |

A check still running after `GBENCH_PREFLIGHT_TIMEOUT_S` (default 300) is a FAIL ("did not
finish"). The table lists the global checks, then the failed suites with their reasons, then the
passed ones. Exit status: 0 when everything passed, 2 on any FAIL. Nothing is skipped: a failed
suite is reported, never left out.

Measured 2026-10-11 on the 84 built-in suites: 31 s wall time.

## `--smoke`

```bash
gbench --smoke --evals all --remote-endpoint http://127.0.0.1:8000/v1 \
  --tokenizer google/gemma-4-26B-A4B-it --thinking --max-output-tokens 65536
```

What happens:

1. The preflight above (its table is printed and saved as `preflight.json`). A FAIL of a check
   that stops the smoke run (disk, Docker, the endpoint from this host) ends it there, every suite
   FAIL "not started". A suite that failed its own checks is FAIL `pre-check` and is not started.
2. Every other suite runs as its own `gbench --evals-only --evals <suite> <your flags>
   --eval-limit 1` process in a `--parallel-suites` pool (default 12; the pool rules below apply),
   with `GBENCH_SMOKE=1` in its environment, and for the agent suites (`smoke.AGENT_SUITES`)
   `GBENCH_SMOKE_RELAY_REPLIES=1`, which gbench's relays read: an agent episode gets
   its first reply, the action that reply asks for runs, and the request after it (the action's
   result going back to the model) is answered with `GBENCH_SMOKE_STOP` instead of being sent to
   the model, which ends the episode (EndpointRelay `smoke_replies`, recorded under `smoke` in the
   relay record). tau2 and tau3 start their relay with `smoke_from_env=False` (a domain whose
   every simulation stopped would have nothing to evaluate), and mcp_bench, mcp_atlas and
   agent_dojo reach the endpoint without a gbench relay. These five run their one episode to its
   own end, and their follow-up is seen on the tap.
3. Each suite's endpoint is a recording tap (`smoke_tap.jsonl` in the suite's directory): a
   transparent proxy on a free local port that forwards every request and reply unchanged (no
   timeout of its own) and records, for each completions request, the temperature,
   `max_tokens`, `chat_template_kwargs.enable_thinking`, the conversation and whether a reply came
   back. When the suite's process ends (or is stopped), the tap closes the server connections of
   its requests still in flight, so the server stops generating replies no one will read.
4. Hang rule and deadline (a smoke run measures nothing, so a limit here changes no result): a
   model request through the tap that receives no reply byte for `GBENCH_SMOKE_HANG_S` (default
   300 s) while the server's token counters (`vllm:generation_tokens_total` +
   `vllm:prompt_tokens_total` from `<endpoint root>/metrics`) stood still too is a hang. Without
   readable counters, only a streamed reply can show it (a non-streamed reply sends nothing until
   its end). The tap closes that request's connection to the server, the suite's processes get
   SIGTERM (SIGKILL 60 s later) and it FAILs at `hang`. A request that waits while the server
   works for other suites is not a hang. The whole run, preflight included, stops at
   `GBENCH_SMOKE_DEADLINE_S` (default 3900 s): the suites still running FAIL at `deadline`, the
   ones not started yet at `not started`.
5. The verdict per suite, and the PASS/FAIL table.

A suite PASSes when every step holds, checked in this order (the first that fails names the
stage):

| Stage | FAIL when |
|---|---|
| `pre-check` | its preflight failed, or it raised `[INFRA REQUIRED]` |
| `hang` / `deadline` | the pool stopped it: a hung model request (the reason names the request and how long it was silent), or the run's deadline |
| `result` | it wrote no result file (the gbench exit code and the last log line are shown) |
| `harness` | its result is a failed configuration (status error / failed / timeout / skipped). Exception: an agent suite whose harness reports the relay's `GBENCH_SMOKE_STOP` (the 400 on the follow-up) as its own error goes on to the checks below when a relay record shows the follow-up arrived (on a failed result the relay records are kept as `relays_at_failure`); an `[INFRA REQUIRED]` error is excused this way only when it names the stop |
| `reply` | no model request reached the endpoint, or none got a reply |
| `request shape` | a model request had no temperature, another temperature than the rule gives, `enable_thinking` not `true` under `--thinking` (or `true` without it) on a chat request, or `max_tokens` above `--max-output-tokens` (an official cap the result records is allowed); an answer request of a `swe_agentic.AGENT_MAX_TOKENS_SUITES` suite reached the endpoint with no `max_tokens`; or the agent relay's own shape check counted a violation |
| `tool execution` | (agent suites) no request continued the conversation after the first reply: the action did not run, or the episode ended before its result went back |
| `scoring` | (other suites) the suite's own scorer scored no sample |

Notes on the rules:

- The expected temperature: `GBENCH_<SUITE>_TEMPERATURE`, else `--temperature`, else 1.0 with
  `--thinking` and 0.0 without it. mmlu and wmdp are scored from log-probabilities: no thinking
  switch, and 0.0 unless `GBENCH_<SUITE>_TEMPERATURE` or `--temperature` is set (they honour both,
  as their results record).
- One-token requests (prompt-token counts, multimodal probes) are not answers: next to real
  answers they are not shape-checked. A suite whose every request is one token (mmlu and wmdp read
  letters from log-probabilities; mmlu_redux without thinking sends HELM's `max_tokens=1`) has
  those requests' temperature checked, and their replies count.
- The agent suites whose scaffold sends no `max_tokens` (`swe_agentic.AGENT_MAX_TOKENS_SUITES`:
  swe_bench_live, swe_bench_pro, swe_bench_multilingual, multi_swe_bench, copilot_bench_swe,
  terminal_bench, skillsbench, wildclawbench, gaia2, tau2, tau3, toolbench, agent_dojo,
  swe_lancer, ui_control_osworld, bfcl_v4_agentic, aider_polyglot) send the run's
  `--max-output-tokens` on every agent request that carries no budget (the user's decision of
  2026-10-11; `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0` turns it off): gbench's relay fills it in,
  agent_dojo sets it in its client. An answer request of these suites that reaches the endpoint
  with no `max_tokens` FAILs `request shape` ("no max_tokens sent on N agent request(s)"). For
  the other suites a request with no `max_tokens` passes the token check. Smaller budgets (the
  8192-token tool-loop turn, a prompt clamped to the window) pass too.
- Agent suites (a follow-up request is required): swe_bench_live, swe_bench_pro,
  swe_bench_multilingual, multi_swe_bench, copilot_bench_swe, terminal_bench, skillsbench,
  wildclawbench, gaia2, tau2, tau3, mcp_bench, mcp_atlas, toolbench, agent_dojo, swe_lancer,
  ui_control_osworld. The follow-up is seen on the tap (a longer conversation after an answered
  request) or in the relay's record (`smoke_second_requests`).
- The scorer is the suite's real one: a smoke result records `smoke_scored` before its accuracy
  is withheld.

Output, in `<--results-dir>/smoke-<timestamp>/`:

- `summary.json`: `smoke: true`, the counts and one verdict per suite (suite, PASS/FAIL, stage,
  reason, requests and replies seen, duration, log and result file).
- `smoke_table.txt`: the table as printed. The smoke directory has no merged `eval_summary.csv`.
- `preflight.json`, `pool.json` (the pool's manifest), and per suite `<suite>/gbench.log`,
  `<suite>/smoke_tap.jsonl` and `<suite>/<timestamp>/` (the suite's own run directory, which holds
  that suite process's own `summary.json` and `eval_summary.csv` as any run writes them: their
  accuracy column is withheld with the smoke reason, while the raw counts of the one sample, such
  as `correct_answers`, stay as the suite's scorer counted them; they are evidence that the
  scorer ran, not a score).
- Each suite's `eval_*.json` carries `smoke: true`, `smoke_scored`, `accuracy: null` with
  `accuracy_withheld_reason`, and `leaderboard_comparable: false`: a smoke run measures nothing,
  and nothing it writes can be read as a score.

Exit status: 0 when every suite passed, 2 on any FAIL (a deadline included), 128+N when stopped
by signal N. `--skip-existing` is refused with `--smoke` (a smoke run reuses nothing), and a real
run never reuses a smoke result.

### Wall time and the defaults

The 2026-10-11 smoke run of the 84 built-in suites (`smoke-10-11-031629`, 8 at a time, deadline
2700 s) took 42.4 min: 37 s of preflight and 41.8 min of pool. The suites' durations added up to
269 min. Two suites set the wall time:

- terminal_bench, 2,285 s: one task drawn x 5 attempts, 17 to 32 agent turns per attempt
  (`turn_budget.observed_turns`). It sent 120 model requests that took 1,829 s in all; most of
  its time was the task's own commands in its containers. Its agent episodes were not ended
  after the first action: terminal_bench then started gbench's relay only when the endpoint had
  an API key, and the smoke stop (`GBENCH_SMOKE_RELAY_REPLIES`) is a relay feature. Its result
  recorded `model_endpoint_relay: null`.
- aider_polyglot, 1,860 s: one exercise. aider's client sends no `max_tokens`; gbench's relay,
  which adds the run's `--max-output-tokens` cap, did not run then (keyless endpoint, see
  Limits), so one reply ran to the end of the 262,144-token window (1,738 s, recorded as one
  exhausted context window).
  It started 11 min into the run, because the duration table then estimated it at 579 s (the
  run records it as 1,860 s for the next one).

Both causes were fixed on 2026-10-11: both suites now always run behind the relay, so
terminal_bench's episodes end after their first action (its smoke run took 3.9 min afterwards)
and aider's reply is capped at `--max-output-tokens`.

The pool's scheduler (the same `build_plan` / `next_suite`, family caps included), replayed with
the run's own per-suite durations, reproduces the measured 41.8 min at 8 and gives, in minutes of
pool time (add about 0.6 min of preflight):

| Suites at once | Order that run used | Order the next run reads | terminal_bench at 300 s and aider_polyglot at 453 s (its time for 5 exercises on 2026-10-10), next run's order |
|---|---|---|---|
| 8 | 41.8 | 38.1 | 26.7 |
| 10 | 38.1 | 38.1 | 22.2 |
| 12 | 38.1 | 38.1 | 22.2 |
| 16 | 38.1 | 38.1 | 22.2 |

No pool size shortens a run below its longest suite. With the relay fix above, the table's
third column (terminal_bench and aider_polyglot short) is the expected case. The defaults follow
from the table and from the load:

- `--parallel-suites` 12 under `--smoke`: 12 is where the replay stops improving (16 gives the
  same). It does not reach every column's lower bound (in the third column 22.2 min against 20.0,
  the suites' total time over 12). Over 8, with the order the next run reads, it gains nothing at
  the measured durations (38.1 both) and 4.5 min in the third column (26.7 -> 22.2); with a wrong
  duration table (the order that run used, where aider_polyglot was estimated at 579 s and ran
  1,860 s) it gains 3.7 and 8.6 min. The cost is load: replaying the run's 588 tapped model
  requests at each suite's new start time gives more than 16 requests in flight for 25.7% of the
  time at 12 against 11.7% at 8, at most 44 in flight at 12 (35 measured at 8, mean 8.3), more
  than 26 for 4.4% of the time, never above the admission gate's 48; at 16 the replay goes above
  48 (1.4% of the time) and above 26 for 10% of it. The long-context family stays at 2 at once
  whatever the pool size.
- `GBENCH_SMOKE_DEADLINE_S` 3900: 1.53 x the slower of the two 2026-10-11 runs' terminal_bench
  paths. `smoke-10-11-031629`: 37 s of preflight plus 2,284.7 s. `smoke-10-11-022420` (same
  command, same task): 37 s plus at least 2,470.2 s; SIGTERM stopped it in Harbor's verifier, and
  its last model request plus 031629's 692 s from last request to end projects 2,505 s, a 2,542 s
  path. The 42.4 min run was 2.6 min short of the old 2700.

Both are single-run figures, and a suite's duration under more concurrency can grow (the replay
keeps each suite's measured duration). `--parallel-suites` and `GBENCH_SMOKE_DEADLINE_S` override
the defaults.

## `--parallel-suites N`

```bash
gbench --evals all --evals-only --eval-limit 5 --parallel-suites 8 \
  --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer google/gemma-4-26B-A4B-it \
  --thinking --max-output-tokens 65536
```

Each selected suite runs as its own `gbench --evals-only --evals <suite> <your flags>` process
(its own interpreter: no suite shares the working directory, the environment or a crash with
another), up to N at once. The default, 1, is the serial sweep in one process. Every flag you pass
reaches every suite's process unchanged, except `--eval-custom-jsonl`, which only the custom_jsonl
process gets (gbench adds that suite to any run with the flag). The pool runs eval suites only: it
refuses `--golden`/`--golden-only`, `--quality`/`--quality-only` and a performance campaign
(`--campaign`, `--input-lengths`, `--output-lengths`); run those on their own. It also refuses
`--shard` / `GBENCH_SHARD` (the pool joins one unsharded result per suite).

`--skip-existing` works as in a serial run: each suite's process searches your `--results-dir`
(earlier `pool-*/` directories included, `GBENCH_REUSE_ROOT`) for a result it may reuse under the
usual rules. To resume a stopped validation run, run the same command with `--skip-existing`; the
new pool directory then holds every suite (reused or run), and is the one to read.

Scheduling:

- Longest first, from earlier wall times: `gbench/pool_durations.json`, at `--eval-limit 5`: per
  suite the longest `duration_s` of the 2026-10-09 serial run and the 2026-10-10 lanes, whatever
  the run's status, since a run that errored after N s still ran N s. A suite with no clean run
  there also takes earlier runs into account, scaled to 5 samples, and its entry says
  `estimate` (terminal_bench's only entry there was 326 s from a run whose trials were
  cancelled, against 9,638 s for 3 samples in a complete run). Your own earlier pool runs at the
  same `--eval-limit` override it (written after each run to `GBENCH_POOL_DURATIONS_FILE`,
  default `<gbench cache>/pool_durations.json`): a clean result (status success, exit 0, not
  partial, no trial errors) replaces the suite's entry; a failed, partial or stopped one only
  raises it. `--smoke` records its durations under a key of its own, which only `--smoke` reads.
  A suite without an entry counts as the median.
- Families never run more members at once than their cap:

  | Family | Members | At once | Why |
  |---|---|---|---|
  | swe | swe_bench_live, swe_bench_pro, swe_bench_multilingual, multi_swe_bench, copilot_bench_swe | 2 | instance images, the docker socket, hours-long rollouts |
  | bfcl | bfcl, bfcl_live, bfcl_v4_agentic, gorilla_apibench | 1 | one BFCL harness image and project root |
  | matharena | aime, hmmt, amc_aime, imo_answer_bench | 1 | the MathArena grader image, 65K-token replies in a row |
  | tau | tau2, tau3 | 1 | one tau2-bench image and cache |
  | mmlu | mmlu, mmlu_pro, mmlu_redux, multilingual_mmlu | 1 | |
  | gpqa | gpqa, gpqa_diamond | 1 | |
  | putnam | putnam, putnam_formal | 1 | |
  | gaia_ports | gaia, gaia2, wildclawbench | 1 | ports 18789-18795 |
  | code_judge | lcb, codeforces, ojbench | 1 | per-test time limits on this host's CPUs |
  | vqa | docvqa, infographicvqa, chartqa, textvqa, charxiv | 1 | |
  | mcp | mcp_bench, mcp_atlas | 1 | MCP server containers |
  | mobilegym | mobilegym, mobilegym_pp | 1 | one gbench-mobilegym image and Playwright Chromium pool |
  | long_context | mrcr, beam_128k, aa_lcr, ruler, longbench_v2_code | 2 | 128K-token prompts fill the KV cache |
  | search | the selected suites among browsecomp, deepsearch_qa, gaia, bfcl, bfcl_v4_agentic, mcp_atlas, wildclawbench | k = min(selected, N, 4) | the Gemini search rate |

- The Gemini search rate is split: each search suite gets `GBENCH_SEARCH_MAX_RPS` = (your value,
  default 8) / k, so the k that may run at once never exceed the run's rate together.
- Admission gate: a suite starts only while the server's `vllm:num_requests_running` +
  `vllm:num_requests_waiting` (from `<endpoint root>/metrics`) is below `GBENCH_POOL_MAX_INFLIGHT`
  (default 48). When none of the pool's suites is running, the next one starts regardless, so
  load from elsewhere cannot stall the pool; an endpoint without `/metrics` leaves the gate open
  (logged once).

While it runs:

- Stall alarm (alert only, never a kill): a WARNING when a running suite has written nothing under
  its directory (its log, its traces, the activity file `GBENCH_POOL_ACTIVITY_FILE`) for
  `GBENCH_POOL_STALL_S` (default 600 s), repeated every such interval while it stays silent. Under
  `--smoke` the tap touches the activity file while replies flow; in a real pool nothing touches it
  yet, so a suite waiting on one long non-streamed reply can raise the alarm (see Limits).
- SIGINT / SIGTERM / SIGHUP: SIGTERM is sent once to every running suite's process group (each
  suite process leads its own session; gbench turns SIGTERM into a clean stop and writes the
  stopped suite's failed record), no further suite starts, and a second signal kills the groups. A `SIGHUP` the pool was started
  ignoring (`nohup`) stays ignored.
- Containers: when a suite's process ends, the containers it left running are removed: the
  `docker run --name` processes seen in its process group, and `gbench-<suite>-*` names that were
  not there when the pool or that suite started (a container of another run that existed before
  is never removed; one another run starts while the suite runs, under the same name prefix,
  still can be, since a name says nothing more). Any
  container created while the pool ran that is still present when it ends (for example a SWE
  instance container an agent started through the docker socket) is listed in a WARNING with the
  `docker rm -f` command, not removed: a container another run started meanwhile looks the same.
  The containers a harness starts through the docker socket under names of its own are among
  them: terminal_bench's Harbor starts each trial with `docker compose up` from inside its harness
  container (project `<task>__<trial id>` lowercased, container `<project>-main-1`, network
  `<project>_default`). Neither the pool's process-group scan (the compose client runs inside the
  harness container) nor the `gbench-<suite>-*` names see them, so the pool does not remove them
  (see Limits).

Output, in `<--results-dir>/pool-<timestamp>/`:

- `<suite>/` per suite: its `gbench.log` and its own `<timestamp>/` run directory (its
  `eval_*.json`, `metadata.json`, traces), exactly as a serial run writes them.
- Each suite's `eval_*.json` linked into the pool directory unchanged, and one `summary.json` and
  `eval_summary.csv` over all of them in the format of a single run; a suite whose process wrote
  no result is a failed configuration there ("exited N without writing a result").
- `metadata.json` (the sources, the gbench commit, the pool's settings) and `pool.json`: the order,
  family caps, and per suite its exit code, duration, log, result files, command, stall alerts,
  removed containers and the concurrency it ran under (`parallel_suites`, `running_at_start`,
  `max_suites_alongside`, `search_max_rps`).
- Each result also records the concurrency it ran under, `pool`: `parallel_suites`,
  `running_at_start`, `search_max_rps`. A suite's score can depend on load (agent loops under
  wall-clock budgets, fixed per-request deadlines); compare like with like.

Exit status: 2 when a suite was not started, its process failed or it wrote no result; else the
highest exit code of the suites (0 when all exited 0); 128+N when stopped by signal N.

`--dry-run` prints the plan (order, estimates, families, the first command) and starts nothing.

## Joining results (`--merge`)

`gbench --merge DIR [DIR ...]` merges the shards of one run (see
[distributed-gbench.md](distributed-gbench.md)). When no result under the directories records a
shard, it joins them instead: every (suite, model, format) must appear exactly once (a suite found
twice is refused, naming both files), each file is copied unchanged, and one `summary.json`,
`eval_summary.csv` and `metadata.json` cover the union. Nothing is re-scored. A mix of sharded and
unsharded results is refused. One file reached by two paths (the pool directory's link to a
suite's result, or two given directories that overlap) counts once, so `gbench --merge
<pool-dir>` and a pool directory together with other runs both work. The pool writes its merged
summary the same way.

## Knobs

| Variable | Default | Meaning |
|---|---|---|
| `GBENCH_PREFLIGHT_MIN_DISK_GB` | 100 | Free disk under `--results-dir` the preflight requires |
| `GBENCH_PREFLIGHT_TIMEOUT_S` | 300 | A preflight check still running after this is a FAIL |
| `GBENCH_POOL_MAX_INFLIGHT` | 48 | The admission gate: start a suite only below this many requests in flight at the server (0 turns the gate off) |
| `GBENCH_POOL_STALL_S` | 600 | The stall alarm's silence interval (0 turns it off) |
| `GBENCH_POOL_DURATIONS_FILE` | `<gbench cache>/pool_durations.json` | Where each pool run records its suites' durations, per `--eval-limit` (smoke runs under their own key) |
| `GBENCH_SMOKE_HANG_S` | 300 | `--smoke` only: a model request silent this long while the server's token counters stood still is a hang (0 turns the rule off) |
| `GBENCH_SMOKE_DEADLINE_S` | 3900 | `--smoke` only: the whole smoke run's limit, preflight included |
| `GBENCH_SEARCH_MAX_RPS` | 8 | The run's Gemini search rate, split between the search suites |

Set by gbench for each suite process (do not set them yourself): `GBENCH_SMOKE=1` (`--smoke`),
`GBENCH_POOL_RECORD` (the `pool` record), `GBENCH_POOL_ACTIVITY_FILE` (the file the stall alarm
reads; the smoke tap touches it), `GBENCH_REUSE_ROOT` (`--skip-existing` in a pool: your
`--results-dir`), `GBENCH_SEARCH_MAX_RPS` for the search suites, `GBENCH_SMOKE_RELAY_REPLIES=1`
(`--smoke`, agent suites) and `GBENCH_RELAY_EXPECTED_SHAPE` (`--smoke`, every suite: the run's
temperature, `enable_thinking` except for mmlu and wmdp, and `max_tokens_at_most` =
`--max-output-tokens` for the `swe_agentic.AGENT_MAX_TOKENS_SUITES`), which every relay of the
suite checks ([relay.md](relay.md)).

## Limits

- The verdict sees what crosses the tap and what each result records. A harness that talks to the
  model through an address other than `--remote-endpoint` (a per-suite endpoint override such as
  `GBENCH_GAIA2_MODEL_ENDPOINT`) bypasses the tap, and its suite fails `reply`.
- An in-process tool loop (gaia, browsecomp, deepsearch_qa, bfcl's agentic part) and the agent
  suites without a smoke relay (tau2 and tau3, whose relay passes `smoke_from_env=False`;
  mcp_bench, mcp_atlas and agent_dojo, which have no gbench relay) run their one sample to its own
  end under `--smoke`, within the hang rule and the deadline. Only the other relayed agents stop
  after their first action.
- The smoke hang rule cannot see a non-streamed request hang while the server is busy with other
  suites (its counters keep moving); the deadline bounds that case.
- The stall alarm cannot tell a long silent reply (a non-streamed 65,536-token reply takes about
  9 min at 120 tok/s) from a hang; it only alerts.
- The admission gate counts every request at the server, the pool's and anyone else's.
- A suite stopped by a signal, the deadline or the hang rule can leave its harness's compose
  containers and networks behind. 2026-10-11: the SIGTERM that stopped smoke run
  `smoke-10-11-022420` reached Harbor (its handler raises KeyboardInterrupt); Harbor cancelled
  the trial still running (`mcmc-sampling-stan__5fLw3gp`, "Trial ... cancelled" in its job.log)
  and exited about 3 s later without a `docker compose down`, and its `main` container and
  network stayed up after the run. List them with
  `docker ps -a --filter label=com.docker.compose.project` and remove them with `docker rm -f`
  and `docker network rm`. The pool's end-of-run WARNING names the containers, not the networks.
- terminal_bench, aider_polyglot, tau2/tau3 and bfcl_v4_agentic always start gbench's relay, with
  or without an endpoint key (the relay adds the key only when there is one). Until 2026-10-11
  (tau2/tau3 and bfcl_v4_agentic until 2026-10-13) the relay was key-gated, and against a keyless
  endpoint nothing the relay does applied to them. All four now get the run's `max_tokens` on
  agent requests (`GBENCH_AGENT_MAX_TOKENS_FROM_RUN`) and the relay's shape check; only
  terminal_bench also gets the smoke stop (aider_polyglot and bfcl_v4_agentic are not in
  `smoke.AGENT_SUITES`, and tau's relay is started with `smoke_from_env=False`, so tau episodes
  run to their end under `--smoke` by design). In the 2026-10-11
  smoke run terminal_bench ran its 5 episodes to their end (about 38 min, the run's longest
  suite), every request of both suites reached the tap with no `max_tokens`, and aider's one reply
  ran to the end of the 262,144-token window; the request-shape stage then let a request without
  `max_tokens` pass, so both suites PASSed. A request of these suites that reaches the endpoint
  with no `max_tokens` now FAILs `request shape`.
