# swe_bench_pro (SWE-bench Pro V2, public set)

> **Docker setup:** this suite runs one container/network per task and can exhaust the
> Docker address pool at high `--sandboxes`. See [docker-sandboxes.md](docker-sandboxes.md)
> for the recommended `default-address-pools` config and pre-run prune, strongly advised
> before running at concurrency.

## What it measures
**SWE-bench Pro V2** (Scale AI): long-horizon issue resolution in 11 Go, Python, JavaScript and
TypeScript repositories (642 tasks: go 256, python 237, js 145, ts 4). Each task gives the agent
a repository at a base commit plus a PR description with explicit requirements and new
interfaces; the agent's patch is graded by the task's hidden `fail_to_pass` / `pass_to_pass`
tests in a pristine container, and the task is resolved iff all required tests pass.

**Dataset.** `ScaleAI/SWE-bench_Pro` pinned to commit `2d52cb3df914a3fcf80c7f66738b3a88ae37fc50`
(Hub head since the V2 release on 2026-09-22): config `default` is V2 (642 tasks, `test`),
config `hard` is the HARD-51 subset (the dataset's `eval.yaml` lists both, as `SWE_Bench_Pro`
and `SWE_Bench_Pro_hard`). gbench reads these rows for the task list, the per-language
stratification and the one-shot prompts; the agent itself sees each task's `instruction.md`.

**Harness.** Scale's V2 reference setup: the Harbor task directories and locked-protocol tooling
in `scaleapi/SWE-bench_Pro-os` at commit `66f92766bba642462d4bbe5479e83f91f9211862` (`v2/`: 642
task directories, `hard51_ids.txt`, `tooling/`), run with **Harbor 0.22.0** (the version of the
tree's release gate, `v2/GATE.md`). The `default` config's instance ids are exactly the task
directory names, and the `hard` config's are exactly `hard51_ids.txt`. Every `task.toml` names
the task image
`ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>` (repository at `/app`, sanitised history),
1 CPU / 4 GB, a 3000 s agent budget and a 3000 s verifier budget, and
`[agent] network_mode = "no-network"`. The verifier (`tests/test.sh`) restores the test files,
applies the hidden `test_patch.patch`, runs the selected tests and parses them with the task's
`parser.py`.

## Protocol (default: the V2 locked protocol)
gbench runs the two Harbor jobs `v2/README.md` documents, with the mini-swe-agent variant of the
locked agents and Harbor's local docker backend instead of Modal:

1. **Agent job.**
   ```
   harbor run -p v2/tasks -e docker -n <sandboxes> -a locked_mini_swe:LockedMiniSwe \
     -m openai/<served model> --ak config_file=<locked config> --ak version=2.4.6 \
     --ae OPENAI_API_BASE=<gbench's relay> --ae OPENAI_API_KEY=EMPTY \
     --allow-agent-host <relay host> --jobs-dir <work dir>/jobs --job-name agent -y
   ```
   In each task sandbox LockedMiniSwe installs mini-swe-agent (pinned to 2.4.6, the newest
   release when V2 shipped; unpinned it would float) with a managed Python 3.12 during the setup
   phase, then runs it on `instruction.md` with `v2/tooling/configs/mini_toolcall.yaml` (sha256
   `7275afa9920a6fb1b9ec66ab74cee44121db9b0276bb9fa90f5ec0597e523a45`: tool-calling actions, no
   step limit, `reasoning_effort: high`) inside an in-sandbox `timeout` of the 50-minute budget.
   The agent phase is offline except the model endpoint, reached through gbench's relay (below):
   Harbor's egress-control sidecar enforces `no-network` plus the one `--allow-agent-host`.
   Afterwards LockedMiniSwe captures the agent's `git diff` as `model.patch`, and the task's
   verifier runs in the same sandbox: that is the **in-sandbox** score.

   **Model request timeout.** gbench writes two transport settings into the locked config's
   `model.model_kwargs`, which mini-swe-agent's LitellmModel passes to every `litellm.completion`
   call: `timeout` and `max_retries: 0`. By default (cap on) the relay's
   per-request limit is `base.request_timeout_s` of the run's `--max-output-tokens` (9,392 s at
   65,536 tokens at the default `GBENCH_MIN_DECODE_TOK_S` of 8), its safety cap on a model request
   that plus 120 s (`EndpointRelay.TOTAL_CAP_MARGIN_S`), and the client's `timeout` the cap plus
   60 s (9,572 s), so the relay always answers first. With `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0`
   both are sized for the served context window (the run knob `max_model_len`, else 262,144
   tokens: 33,968 s relay limit / 34,148 s client). The relay counts a client
   close near that client timeout in `client_timeouts_suspected`. Without them the sandbox's litellm (installed unpinned by
   LockedMiniSwe; 1.87 through 1.102 checked) gives each request 600 s with 2 OpenAI SDK retries,
   so a turn longer than 600 s was abandoned and sent again; results from before 2026-10-11 ran
   under that cap. The 50-minute agent budget of each task is unchanged and still bounds the
   whole rollout. The result records both settings (`agent.gbench_config_additions`) and
   `model_client_timeout` (`client`, `version`, `timeout_s`, `retries`, `source`,
   `hidden_cap_removed`, `replaces`). Nothing the model is sent or allowed changes.

   **Egress sidecar read timeout.** In the agent phase every model request passes through
   Harbor's egress-control sidecar (`harbor-docker-egress-control-sidecar`, a gost transparent
   proxy with HTTP sniffing). gost waits at most `readTimeout` for the upstream's response
   headers and uses 15 s when it is unset (go-gost/x v0.10.9, `handler/redirect/tcp/metadata.go`).
   A non-streamed completion sends its headers only when the whole reply is ready, so every
   turn that took longer than 15 s was cut: the agent's litellm saw "Server disconnected without
   sending a response" and retried. 2026-10-11 validation run (5 tasks, shared server): 128 of
   198 model requests were closed at <= 15 s (relay `client_close_ages.max_s` 15.0), and 4 of 5
   rollouts ended as `agent_infra`. The image now adds `readTimeout: 8760h` to the sidecar's
   `gost.yaml` (handler metadata; gost cannot turn the wait off, 0 means 15 s), far above the
   relay's limit at any `--max-output-tokens` or window (33,968 s at 262,144 tokens), so only the
   relay and client limits above bound a request. The build checks the original file's sha256
   first and labels the image `org.gbench.egress_sidecar_read_timeout`. Harbor builds the sidecar
   image from that directory at run time, tagged by a hash of its contents, so the first run
   builds `harbor-prebuilt:harbor-docker-egress-control-sidecar--f6aa702cd6dfa82e` (linux/amd64)
   from the digest-pinned gost base instead of reusing the 15 s one (`...--b1e6760734a9e6f6`).
   The result records the setting in `harness.egress_sidecar` (`proxy`, `read_timeout`,
   `sidecar_read_timeout_s`, `hidden_cap_removed`, `replaces`). V2 Harbor runs from 2026-09-25
   (when the agent-phase `no-network` + `--allow-agent-host` route was added) up to 2026-10-11 ran
   under the 15 s cut; earlier runs and the `GBENCH_SWEBENCH_PRO_AGENTIC=1` engine route did not
   send model traffic through this sidecar. Check: a request through the sidecar to an upstream that answers after
   30 s failed at 15.0 s with the original `gost.yaml` and returned 200 after 30.0 s (and after
   200 s) with the patched one.

   `GBENCH_AGENT_MAX_TOKENS_FROM_RUN` (default on since 2026-10-11; `=0` turns it off) makes the relay set `max_tokens` to the
   run's `--max-output-tokens` on every agent request and sizes both limits for it. The locked
   config sends no max_tokens, so such a run records `official_sampling.agent_max_tokens_from_run`
   and is not comparable.
2. **Re-grade job** (the authoritative score):
   ```
   harbor run -p v2/tasks -e docker -n <sandboxes> -a patch_replay:PatchReplayAgent -m replay \
     --ak source_job=<work dir>/jobs/agent --jobs-dir <work dir>/jobs --job-name regrade -y
   ```
   PatchReplayAgent applies each captured `model.patch` to a pristine task image and the
   unchanged verifier runs there, so nothing the agent did to its own sandbox can reach the
   verdict. Only tasks whose agent trial is the model's own outcome (see below) are re-graded.

Re-grade verdicts follow Harbor 0.22.0. A trial whose verifier finished is scored with its
reward. A trial whose verifier started and then raised `VerifierTimeoutError` (the patch made a
test hang past the 3000 s verifier budget), `RewardFileNotFoundError`, `RewardFileEmptyError` or
`VerifierOutputParseError` ran on the patched tree and is the patch's outcome: it scores 0, as
Harbor's metrics count a trial without a reward, and it is not re-run (those four are in Harbor's
default `RetryConfig.exclude_exceptions`). Every other re-grade failure (sandbox start, image
pull, a verifier upload or download error, or no trial at all) is infrastructure.

Infrastructure retries, as the V2 release gate did them (it re-ran only sandbox errors such as
`EnvironmentStartTimeoutError` and `ConflictError`): a task whose agent phase never started
(sandbox start, image pull or agent install failure: the model was never queried) runs once
more in a job named `agent_retry1`, and a task whose re-grade failed for an infrastructure reason
is re-graded once more (`regrade_retry1`). A task whose agent phase started is never re-run, so
no model outcome is sampled twice. PatchReplayAgent reads one source job, so the re-grade runs
once per agent job (`regrade` for `agent`, `regrade_agent_retry1` for `agent_retry1`).

The locked config is used as shipped with one addition: the served model's reasoning switch,
`model.model_kwargs.extra_body.chat_template_kwargs.enable_thinking`, set from `--thinking`. The
config's `reasoning_effort: high` reaches a chat template like Gemma's only through the server's
own mapping (vLLM turns a `reasoning_effort` other than `none` into `enable_thinking` unless the
request sets it) and only if litellm forwards the parameter, so gbench sets the switch
explicitly and records it. A temperature is always added, `model.model_kwargs.temperature`, by
gbench's rule for every suite: 1.0 on a `--thinking` run (`temperature_source` `"thinking"`), 0.0
without it (`temperature_source` `"default"`), unless you set one. The derived
file is written to `<work dir>/mini_toolcall.gbench.yaml` and recorded on the result.

Run knobs:

| Knob | Effect |
| --- | --- |
| `--eval-thinking` | `enable_thinking` on every agent request (a run without it is recorded as not comparable: the locked config asks for high reasoning) |
| `--temperature` / `GBENCH_SWE_BENCH_PRO_TEMPERATURE` | replaces gbench's rule (1.0 with `--thinking`, 0.0 without); `GBENCH_SWE_BENCH_PRO_TEMPERATURE` wins over `--temperature`. The locked config sends no temperature, so the temperature sent makes the run not comparable |
| `--max-output-tokens` | sent as every agent turn's `max_tokens` (set by gbench's relay as a body field), on by default since 2026-10-11, a recorded deviation (the locked config sets no per-turn cap). `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0` restores upstream's none (the server grants each turn what the context window has left). The one-shot mode sends it as its single request's `max_tokens` |
| `--sandboxes` (or the eval concurrency) | Harbor `-n` (concurrent trials) for both jobs |
| `--eval-limit` / `--shard` | stratified subset by language / round-robin shard of the 642, passed to Harbor as `-i <instance_id>` |
| `GBENCH_SWE_OMP_THREADS` | OpenMP/BLAS thread cap passed to the agent (`--ae`) and the verifier (`--ve`) environments (`0` disables): with Harbor's 1-CPU quota a test would otherwise spin up threads for every host core |
| `GBENCH_SWE_BENCH_PRO_TASK_HOST` | the address gbench's relay listens on, which the sandboxes reach the endpoint through (default `172.17.0.1`, the docker bridge gateway; see below) |
| `GBENCH_SWE_BENCH_PRO_TASK_ENDPOINT` | not used by the V2 locked protocol or the one-shot mode: the sandboxes reach the endpoint only through gbench's relay, which forwards to the endpoint gbench was given |
| `GBENCH_SWE_BENCH_PRO_IMAGE` | harness image name (default `gbench-swe-bench-pro-v2:harbor-0.22.0-66f92766-sidecar-rt`) |
| `GBENCH_SWE_BENCH_PRO_TRAJ_DIR` | parent of the work directory (default: `<run results dir>/agent_trajectories/`) |
| `GBENCH_SWE_BENCH_PRO_MODE` | `oneshot` for the single-request mode |
| `GBENCH_SWEBENCH_PRO_AGENTIC` | `=1` routes to the separately maintained engine (below) |

**The endpoint, its key and its failures.** In the locked protocol mini-swe-agent runs inside
each task sandbox, and every command the model runs there inherits the agent's environment
(mini-swe-agent 2.x `LocalEnvironment` runs commands with `os.environ | config.env`). Harbor
also passes agent variables as `docker compose exec -e NAME=VALUE`, on a command line. So an
endpoint key never goes to Harbor or a sandbox: the agent always gets `OPENAI_API_KEY=EMPTY` (as
a literal `harbor run` argument; `harness.sandbox_api_key` on the result). For every run gbench
starts a relay: a thread of the gbench process listening on `GBENCH_SWE_BENCH_PRO_TASK_HOST`
(default `172.17.0.1`) at a free port. The sandboxes use it as `OPENAI_API_BASE` (it is also the
one `--allow-agent-host` entry), and it sends each request on to the endpoint gbench was given
and returns the endpoint's reply unchanged. When the endpoint has a key (`VLLM_API_KEY`, else
`OPENAI_API_KEY`; `EMPTY` and the other placeholders are no key) the relay is the key forwarder: it adds `Authorization: Bearer <key>`, and the result
records it under `harness.key_forwarder`; without a key it adds nothing and is recorded under
`harness.endpoint_relay` (both: `listens`, `relays_to`, `forwards`: the methods, the allowed
paths and the one endpoint). The endpoint then only has to be reachable from the gbench host.
If the relay cannot listen on that address, the run hard-errors (`infra_required`) and names the
variable.

The relay counts the endpoint's failures (`relay` on the result: a request the endpoint did not
answer, a reply that is not 2xx except a 400 naming the context window, a 2xx completions reply
that is not JSON, an error event in a stream, a request to a path it does not forward). They
are harness errors, never the model's outcome: listed in `endpoint_failures`, the run becomes
`completed_with_errors` and it is not comparable (a run-level flag that changes no score: the
relay cannot tell the agent's requests from ones the model's own commands send from the same
sandbox). mini-swe-agent retries a failed model request itself (up to 10 attempts with 4 to 60 s
waits; a request that still fails ends the rollout with that exception as its exit status, an
`agent_infra` harness error). A rollout the 50-minute budget cut off (exit status `""`) is the
patch's verdict, as upstream counts it, except when the budget ran out while the endpoint was
failing that rollout's own requests, on evidence the model cannot produce from its sandbox (its
commands can reach the relay and print anything into the agent's log):

- the relay counted an **endpoint-side** failure (no reply, a 5xx, an error event in a stream,
  an unparsable 2xx; never a 4xx, which the sender provokes at will, e.g. a wrong model name) of
  a request carrying **this rollout's conversation** (the relay tags each such failure with a
  fingerprint of the request's first user message, the task prompt, which the trajectory's first
  user message gives: `budget_kill.conversation_fingerprint`; a concurrent rollout's failure has
  another) between the rollout's last event and the end of its agent phase, and
- no request of that conversation got a reply after the last such failure (the rollout never got
  going again: `budget_kill.recovered_after_failure` otherwise).

Its log (`agent/mini-swe-agent.txt`) must also end in tenacity's `Retrying ... as it raised
<exception>` record naming a model-endpoint exception after the last model reply; that text alone
decides nothing. Such a rollout is an `agent_infra` harness error. Every cut-off trace records
`budget_kill` (`last_event_at`, `ended_at`, `last_event_age_s`: the seconds between the rollout's
last event and the cut, `retrying`: the exception it was retrying, if any, and
`conversation_fingerprint`). `relay_failures_in_window` (the endpoint-side failures of its own
requests in that window), `attribution` and `recovered_after_failure` are recorded only for a
cut-off trace whose log ended in a retry record and whose window start and end times are known:
there is nothing to count for the others. A rollout that was waiting on a reply that had not failed (a long
generation, or an endpoint that stopped answering without an error) cannot be told apart from the
model's own use of its budget, so it stays the model's outcome, with that wait recorded. Harbor's
own label of the agent trial's exception is recorded as `agent_harbor_exception`; Harbor reads it
from the agent's output, which also carries the model's command output (a budget cut can be
labelled `ApiOverloadedError`), so the attribution above does not use it.

The relay only ever talks to that one endpoint (`swebench_common.EndpointRelay`). Its
scheme, host and port are fixed when it starts, and each target URL is built from them and the
request path with `urllib.parse.urlunsplit`, never by joining strings. A request is forwarded
only when its method is GET or POST, its path is one of the endpoint paths the agents use,
`/v1/chat/completions`, `/v1/completions` or `/v1/models` (so no other route of the served
endpoint, such as vLLM's adapter-loading routes, can be reached with the key's authority), and
every path segment is made of letters, digits, `-`, `.`, `_` or `~`, with no empty, `.` or `..`
segment, no percent escape and no fragment, and a query of the same characters plus `=` and
`&`. Any other request is answered `400` (`405` for another method) and never forwarded, so a
request line such as `GET :8080/x` or `GET .attacker.example/x` cannot move it to another port
or host. Redirects
from the endpoint are not followed (a 3xx goes back as it is, without its `Location`), so the
`Authorization` header is never sent to another URL. The client's own `Authorization` header is
dropped, and the key appears in no reply, log line or error message.

**The docker socket.** The Harbor container of each job mounts `/var/run/docker.sock` and runs
on the host network, as Harbor's local docker backend needs: it starts each task sandbox (and
its egress sidecar) with `docker compose` on the host daemon and bind-mounts the trial
directories into them. mini-swe-agent and the model's commands run inside the sandbox, which
gets no socket; the Harbor container only reads what comes back (the agent's output, the
trajectory, the patch, the verifier's files). The residual risk: that container holds
root-equivalent access to the host's docker daemon (and so to whatever the host can reach) and
shares the host's network namespace, and it parses text the model influenced; a flaw in Harbor
that let that text run inside it would reach both. gbench keeps upstream's design (Harbor's
docker backend cannot run without a daemon it can reach) and keeps credentials out of that
container (`OPENAI_API_KEY=EMPTY` for the agent, `HOME` in the work directory, no Docker client
configuration; the relay adds the endpoint's key).

**One-shot (opt-in).** `GBENCH_SWE_BENCH_PRO_MODE=oneshot`: one request per task (repository,
PR description, requirements, interface, and an instruction to answer with one diff; no
repository access, the same temperature rule: 1.0 with `--thinking`, 0.0 without). The extracted diffs are written as a job
directory PatchReplayAgent reads and re-graded by the same Harbor re-grade job, with the same
verdict and retry rule (`swe_bench_pro_report.not_regraded_ids` lists the tasks whose re-grade
still failed for an infrastructure reason: harness errors, `harness_error: "no_regrade_verdict"`,
left out of `accuracy`'s numerator and denominator and counted in `harness_errors`). Structurally
near 0 (patches rarely apply) and always `leaderboard_comparable: false`.

**The separately maintained engine.** `GBENCH_SWEBENCH_PRO_AGENTIC=1` routes to
`swe_bench_pro_agentic.py` exactly as before (setting it together with
`GBENCH_SWE_BENCH_PRO_MODE=oneshot` is an error). That engine scores with Scale's **v1** harness
(`swe_bench_pro_eval.py`, `run_scripts/`, `jefzda/sweap-images`) against
`swe_bench_pro.raw_sample_path()`, which is the v1 raw-sample table: `ScaleAI/SWE-bench_Pro` tag
`v1.0` (commit `307bd006a0a3807424d3c5525eaf98ca8e63aed3`, 731 tasks), written as
`swe_bench_pro_full.v1.0-307bd006.csv` in the harness directory (or `$TMPDIR`) and used only
after it matches those rows; `GBENCH_SWE_BENCH_PRO_RAW_SAMPLE` points at your own `.csv` or
`.jsonl` copy instead (checked the same way, never rewritten). The v1 harness can only score v1
rows, so its result is labelled as a v1-harness number and never comparable. Its prerequisites
are the v1 ones: Docker + the Python `docker` SDK, a clone of `scaleapi/SWE-bench_Pro-os` at
`GBENCH_SWE_BENCH_PRO_HARNESS_DIR` (or `$GBENCH_PREREQS_DIR/SWE-bench_Pro-os`) with
`swe_bench_pro_eval.py` and `run_scripts/`, and the v1 raw-sample table above.

## Reading the result
- `accuracy` is the **re-graded** resolved rate over the selected tasks that were scored
  (`scored_instances`; `headline` says so), harness errors left out (below); `correct_answers`
  the re-graded resolved count; `category_accuracy` splits it by language.
- `in_sandbox` is the verifier's rate inside the agent's own sandbox, the second number the V2
  README asks to publish, over the tasks whose rollout was the model's own outcome. `hard51`
  gives the re-graded and in-sandbox rates on the HARD-51 tasks among the selected ones, with
  the same denominators (`complete` is true when all 51 were selected). Each rate records
  `harness_errors` (the tasks it left out) when any. A rollout whose in-sandbox verifier failed for an infrastructure reason (it started and
  raised an exception other than the verifier-outcome ones, with no reward; the re-grade's
  rule) has no in-sandbox verdict: it is left out of `in_sandbox` (numerator and
  denominator) and listed in `in_sandbox_verifier_infra` with the exception.
- `dataset` names the pinned revision and both configs; `harness` the tree commit, Harbor
  version, image, the task endpoint, the agent-phase allowlist, the egress sidecar's read
  timeout (`egress_sidecar`, see "Egress sidecar read timeout" above), the sandboxes' `OPENAI_API_KEY`
  placeholder (`sandbox_api_key`) and the relay (`key_forwarder` when the endpoint has a key,
  `endpoint_relay` otherwise); `relay` what the relay counted (`endpoint_failures` when it saw
  any);
  `agent` the locked agent,
  mini-swe-agent version, config and its sha256, and the gbench additions to it (including the
  client `timeout` and `max_retries`); `model_client_timeout` the agent's client timeout and
  transport retries (see "Model request timeout" above);
  `harbor_jobs` each job's return code and trial count; `infra_retried_ids` the tasks that got
  a second agent job; `temperature` / `temperature_source` the temperature sent (0.0 "default",
  1.0 "thinking" with `--thinking`, or the value you set).
- Every sample trace carries `agent_patch` (the captured `model.patch`), `agent_exit_status`
  (mini-swe-agent's, from the in-sandbox trajectory), `agent_steps`, `in_sandbox_reward`,
  `regrade_reward`, `regrade_exception` (Harbor's exception type for the re-grade trial, if
  any), `agent_harbor_exception` (Harbor's label of the agent trial's exception, if any),
  `budget_kill` (for a rollout the budget cut off, above), `trajectory_file` (the in-sandbox
  `mini-swe-agent.trajectory.json`) and `extra_payload.hard`. `regrade_verifier_exception_ids`
  maps each task scored 0 because its re-grade verifier timed out or left no reward to that
  exception. `agent_output_dir` is the work directory: `jobs/<job>/` (Harbor's own job
  directories, one per trial with its logs, patch and verifier output), one `harbor_<job>.log`
  per job and the derived config.
- A task is the model's own outcome when its agent phase ran to the end (Harbor then started
  the verifier, or recorded only the model's budget running out or the agent exiting non-zero),
  `model.patch` was captured (empty when the model changed nothing, which re-grades to 0), and
  mini-swe-agent ended for a model reason (submitted, limits, format errors, context window, or
  cut off by the budget, unless the budget ran out while the endpoint was failing that
  rollout's own requests, above). Anything else (`harness_error: "agent_infra"`), and a task whose
  re-grade still failed for an infrastructure reason after its retry (`harness_error:
  "no_regrade_verdict"`), is an infrastructure failure: a harness error, left out of `accuracy`'s
  numerator and denominator, counted by kind in `harness_errors` (ids in `harness_error_ids`,
  also `failed_requests`), listed in `partial_evaluation` (`status: "completed_with_errors"`),
  or `status: "error"` with `accuracy` `null` (`accuracy_withheld_reason`) if no task was
  scoreable, never a 0%.
  A re-grade verifier that timed out or left no reward is not an infrastructure failure: that
  task scores 0 (see the verdict rule above).
- `leaderboard_comparable` is `true` only for a full, unsharded run with `--thinking`, no
  temperature sent, the pinned locked config, every task scoreable and no endpoint failure
  counted by the relay. gbench always sends a temperature (above), which the locked config does
  not, so today no run is comparable and every run names that deviation;
  `leaderboard_comparable_reason` lists each deviation. Comparable means the V2 locked protocol
  with its reference mini-swe-agent agent and the re-graded headline; Scale's leaderboard
  ranks model-and-agent pairs, so rows run with another locked agent (Claude Code, Codex) also
  measure that agent.

## Prerequisites
- **Docker Engine** with a reachable daemon at `/var/run/docker.sock`, whose kernel supports
  nftables fib rules (Harbor's egress control checks `CONFIG_NFT_FIB_INET`; without it the
  offline agent phase cannot be enforced and every trial fails). Disk and Docker network egress
  for tens to hundreds of GB: the 642 task images from `ghcr.io/scaleapi/swe-bench_pro-v2`
  (anonymous pulls), plus the small images Harbor pulls itself at run time (its kernel probe and
  the egress sidecar base).
- **The local harness image** (gbench never pulls it; building it needs network access):
  ```bash
  docker build -t gbench-swe-bench-pro-v2:harbor-0.22.0-66f92766-sidecar-rt -f gbench/docker/swe_bench_pro_v2.Dockerfile gbench/docker
  ```
  It installs Harbor 0.22.0 against `gbench/docker/swe_bench_pro_v2.constraints.txt` (the resolved
  set of the checked image; the build fails on any other set) with the Docker CLI, compose and
  buildx plugins at pinned versions (`docker-ce-cli=5:29.8.1-1~ubuntu.24.04~noble`,
  `docker-compose-plugin=5.5.1-1~ubuntu.24.04~noble`,
  `docker-buildx-plugin=0.37.1-1~ubuntu.24.04~noble`; labels `org.gbench.docker_cli`,
  `org.gbench.docker_compose`, `org.gbench.docker_buildx`; Debian packages from the Ubuntu archive
  snapshot `20260925T120000Z`, recorded in `/opt/gbench/dpkg-versions.txt`), fetches
  `scaleapi/SWE-bench_Pro-os` at `66f92766` and checks `v2/` against its `SHA256SUMS`. The
  build also adds `readTimeout: 8760h` to Harbor's egress sidecar `gost.yaml` (see "Egress
  sidecar read timeout"). The prerequisite check refuses an image whose labels
  (`org.gbench.harbor`, `org.gbench.swe_bench_pro_os`, `org.gbench.egress_sidecar_read_timeout`)
  name another version, commit or timeout, so an image built before 2026-10-11 must be rebuilt.
- **Network egress from the task sandboxes in the setup and verifier phases** (the agent phase
  is offline): LockedMiniSwe installs `uv` (astral.sh) and mini-swe-agent 2.4.6 with litellm from
  PyPI in each sandbox; some old Debian images fall back to `archive.debian.org` for apt; a few
  Go tasks fetch modules at test time.
- A served model at an OpenAI-compatible `/v1` endpoint **with tool calling enabled** (vLLM:
  `--enable-auto-tool-choice --tool-call-parser <the model's parser>`, e.g. `gemma4`; a server gbench launches itself (no `--remote-endpoint`) gets both from `--tool-call-parser NAME`, and the CLI refuses a local run of this suite without it),
  reachable
  from the gbench host: the sandboxes reach it only through gbench's relay (above), which listens
  on the docker bridge gateway (`172.17.0.1`, or `GBENCH_SWE_BENCH_PRO_TASK_HOST`), an address of
  the gbench host the sandboxes reach. `VLLM_API_KEY` (else `OPENAI_API_KEY`) if the endpoint
  needs a key; the relay then adds it, and the key never reaches Harbor or a sandbox.
- **Dataset access**: Hugging Face Hub access (or the pinned revision already in the local HF
  cache) for `ScaleAI/SWE-bench_Pro` at `2d52cb3d`, configs `default` and `hard`; the Python
  package `datasets` (a base dependency of gbench).
- **On a cloud VM, the metadata server blocked for docker containers.** The sandboxes run the
  in-sandbox agent's commands and the verifier's tests over each patch, with network egress in
  the setup and verifier phases, and a docker container reaches the VM's metadata server at
  `169.254.169.254`, which hands out the VM's service-account token. Before any Harbor job (in
  both modes), gbench checks that it is unreachable from docker's default bridge and from a
  user-defined bridge network, the kind of network Harbor's compose projects create (gbench
  creates a temporary one, `gbench-metadata-probe-<id>`, and removes it right after): a
  short-lived probe container of `gbench-swe-bench-pro-v2:harbor-0.22.0-66f92766-sidecar-rt` (`--pull never`) opens and closes a
  TCP connection to `169.254.169.254:80` and sends nothing. The same probe container resolves
  the hosts every sandbox installs the agent from before the model's first step
  (`SANDBOX_INSTALL_HOSTS`): `LockedMiniSwe.install` installs curl, git, python3 and build tools
  with the image's apt (`deb.debian.org` and `security.debian.org`, retried against
  `archive.debian.org`; `archive.ubuntu.com` on Ubuntu images), runs astral.sh's uv installer
  (`astral.sh`, which downloads uv from `github.com`) and `uv tool install --python 3.12
  mini-swe-agent==2.4.6` (Python from `github.com`, the package from `pypi.org` and
  `files.pythonhosted.org`); five of the 642 verifiers (gravitational/teleport) also `apt-get
  install`. A name that does not resolve **hard-errors** too (on GCE the VPC DNS resolver is
  `169.254.169.254` itself, so a rule that drops all traffic to it also drops DNS). A connection
  that opens, or a probe without a clear verdict, **hard-errors** (`infra_required`) naming the
  fix. Block it once on the host; gbench never runs sudo or changes the host's firewall itself:
  ```bash
  sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
  sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save   # keep it
  ```
  See [toolchains.md](toolchains.md#cloud-vms-the-metadata-server). The v1 route
  (`GBENCH_SWEBENCH_PRO_AGENTIC=1`) and the v1 scorer run the same metadata server check before
  the engine or the v1 harness starts a container, probing with `gbench-swe-bench-pro-v2:harbor-0.22.0-66f92766-sidecar-rt` or,
  without it, a local `jefzda/sweap-images` image; with neither on the host they hard-error. It
  checks no host names: that route's engine is maintained separately.

No opt-in variable is needed: a machine with these prerequisites runs the suite.

Any missing piece **hard-errors** (`infra_required`, `status: "error"`) with a reason saying what
to provision; the suite never skips.

## Sanity check of the harness image
The V2 README's own check (the reference patch resolves every task, an empty patch none), run
through the image the same way gbench runs it, here on one task (add `-i` flags for more):
```bash
W=$(mktemp -d) && docker run --rm --network host -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$W:$W" -e HOME="$W" --user "$(id -u):$(id -g)" --group-add "$(stat -c %g /var/run/docker.sock)" \
  gbench-swe-bench-pro-v2:harbor-0.22.0-66f92766-sidecar-rt harbor run -p v2/tasks -e docker -n 3 -a oracle -y \
  -i instance_NodeBB__NodeBB-00c70ce7b0541cfc94afe567921d7668cdc8f4ac-vnan \
  --jobs-dir "$W/jobs" --job-name oracle
```
Replace `-a oracle` with `-a nop` for the empty-patch half; the job's `result.json` under
`$W/jobs/<job-name>/` reports the mean reward (1.0 and 0.0).

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals swe_bench_pro --eval-thinking \
       --max-output-tokens 32768 --sandboxes 4
```
Start with `--eval-limit 5` (stratified, not comparable); image pulls dominate wall-clock.
`--max-output-tokens` is required by the CLI. It is sent as every agentic turn's `max_tokens`
(gbench's relay; `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0` turns that off) and as the one-shot request's
`max_tokens`.
