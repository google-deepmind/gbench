# swe_bench_multilingual (SWE-bench Multilingual)

> **Docker setup:** this suite runs one container per instance (the agent's, then the
> harness's) and can exhaust the Docker address pool at high `--sandboxes`. See
> [docker-sandboxes.md](docker-sandboxes.md) for the recommended `default-address-pools`
> config and pre-run prune.

## What it measures
**SWE-bench Multilingual** (`SWE-bench/SWE-bench_Multilingual`, `test` split, 300 issues from 41
repositories in 9 languages: C, C++, Go, Java, JavaScript, TypeScript, PHP, Ruby, Rust). For each
instance the model resolves a real GitHub issue in the repository; the patch is applied in the
instance's prebuilt Docker image and `FAIL_TO_PASS` / `PASS_TO_PASS` run there. An instance is
resolved iff all of them pass. Host toolchains are irrelevant: every language's tests run inside
the images.

**Dataset revision.** Pinned to commit `846e647b9f33c0b51b739d005d13d85493c9af09` (Hub head since
2026-08-17). Its August 2026 commits add the `image`, `eval_script`, `log_parser` and `eval_type`
columns the v5 harness grades with, sync the eval scripts with the image builders and isolate
the test output streams of two instances. On first use gbench writes that commit's `test` split
to
`$GBENCH_CACHE_DIR/swebench_snapshots/SWE-bench__SWE-bench_Multilingual@846e647b9f33c0b51b739d005d13d85493c9af09/test.jsonl`
(default `GBENCH_CACHE_DIR` is `~/.cache/gbench`) and reuses it afterwards, also offline. The
rows are checked before they are written and the file again before every reuse: 300 rows,
every row carrying `instance_id`, `problem_statement` and the v5 harness columns (`image`,
`repo`, `version`, `FAIL_TO_PASS`, `PASS_TO_PASS`, `log_parser`, `eval_type`, `eval_script`),
and the content fingerprint of the rows, sha256
`83441ebb91ffa12cb167d02ff816ba4d2d30afa5f036ade5f6ab56bee09e37ce` (each row as canonical JSON,
sorted keys, one per line in the published order; `swebench_common.snapshot_sha256`, measured
on the rows `datasets` loads at this commit). A cached file that fails is moved aside to
`test.jsonl.invalid-<unix time>` and the commit is fetched again; rows that fail after a fetch
are a hard error naming the problem. The agent's tasks (agentic mode), the prompts (one-shot
mode) and the harness's `--dataset_name` all come from that one file.

## Protocol (default: agentic)
The Multilingual leaderboard's mini-SWE-agent rows run mini-swe-agent 2.x with its bundled
SWE-bench config. gbench runs the release of the SWE-bench maintainers' current Multilingual
submission (`SWE-bench/experiments`, `evaluation/multilingual/20260902_mini-v2.4.6_gemini-3-5-flash`:
"mini-SWE-agent 2.4.6 ... using mini's bundled `swebench.yaml` config unchanged", verdicts
re-derived with the v5 harness; its trajectories, e.g. `apache__druid-13704.traj.json` in
`SWE-bench/20260902_mini-v2.4.6_gemini-3-5-flash`, carry the bundled 2.4.6 templates unchanged):

1. For each instance gbench starts
   `docker.io/swebench/sweb.eval.x86_64.<instance_id with __ -> _1776_>:latest` (repository at
   `/testbed`, base commit checked out): the image the harness scores in, as mini-swe-agent's own
   SWE-bench runner names it.
2. **mini-swe-agent 2.4.6** runs against it with its bundled `benchmarks/swebench.yaml`,
   unmodified (sha256 `9a9c86ac10428b86b932c972b15fefc2f7b6e92230bac5ebdc47e83232a8315e`): only
   the `problem_statement` is shown, actions are `bash` tool calls (several per turn allowed,
   60 s each), at most 250 model calls. The config's cost limit does not trigger for a served
   model litellm has no price for.
3. The submission is what the agent prints after `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
   (its `git diff` of the source files it changed), sent to the harness verbatim.
4. Scoring: the **SWE-bench v5 harness** (swebench 5.0.2, `swebench.harness.run_evaluation`),
   which builds each instance's test run from the row's `image`, `eval_script` and `log_parser`,
   with its default 1800 s test timeout.

gemma4-env keeps mini-swe-agent 1.x and swebench 4.1 for other suites, so both run in LOCAL
images, docker-out-of-docker: `gbench-mini-swe-agent-v2:2.4.2-2.4.6` (one container per instance, driven by
`gbench/docker/mini_swe_agent_v2_run.py`, which follows 2.4.6's
`run.benchmarks.swebench.process_instance` step for step) and `gbench-swebench-v5:5.0.2`. Each agent
container runs on docker's default bridge network with the placeholder key
`OPENAI_API_KEY=EMPTY`; it reaches the served endpoint only through gbench's relay
(`swebench_common.EndpointRelay`, a thread of the gbench process listening on that network's
gateway, e.g. `172.17.0.1`), which forwards only `/v1/chat/completions`, `/v1/completions` and
`/v1/models` to the endpoint URL you pass and adds its key, so no credential enters the
container (`agent.api_base` in the result is the relay's address). Before the first rollout
gbench checks, with a short-lived probe container of `gbench-mini-swe-agent-v2:2.4.2-2.4.6` on that network
(`--pull never`, nothing sent), that the relay is reachable there (`infra_required` otherwise).
The agent container mounts the docker socket because upstream's design needs a docker daemon:
mini-swe-agent 2.x's `DockerEnvironment` starts the instance image with `docker run` and runs
each of the model's commands in it with `docker exec`, so the model's commands run in the
instance container, which gets no socket. The socket gives the agent process, which turns the
model's tool calls into those `docker exec` calls, root-equivalent access to the host's docker
daemon: a flaw in mini-swe-agent, litellm or the driver that let model output run inside the
agent container would reach it. gbench keeps upstream's design (the run does not work without a
daemon the agent can reach) and limits what the agent container itself holds: no credential (the
placeholder key, no Docker client configuration, `HOME=/tmp`), and docker's default bridge
network, where the cloud metadata server gate has checked that `169.254.169.254` is unreachable.
The socket itself stays root-equivalent on the host.

**Model request timeout.** The agent's litellm (1.102.1 in the image) gets two transport settings
in `model.model_kwargs`, which mini-swe-agent's LitellmModel passes to every `litellm.completion`
call: `timeout` and `max_retries: 0`. The relay's limit is `base.request_timeout_s` of the
served context window (the run knob `max_model_len`, else 262,144 tokens; the bundled config
sends no max_tokens), and its safety cap on a model request is that plus 120 s
(`EndpointRelay.TOTAL_CAP_MARGIN_S`); the client's `timeout` is the cap plus 60 s, so the relay
always answers first (`swe_agentic.client_timeout_for_relay_limit`): 33,968 s relay limit /
34,088 s relay cap / 34,148 s client on a 262,144-token window at the default
`GBENCH_MIN_DECODE_TOK_S` of 8. The relay counts a client close near that client timeout in
`client_timeouts_suspected`. Without them litellm gives each
request 600 s (`COMPLETION_HTTP_FALLBACK_SECONDS`) with 2 OpenAI SDK retries, so a turn longer
than 600 s was abandoned and sampled again; results from before 2026-10-11 ran under that cap.
Each request is now one HTTP attempt; mini-swe-agent's own retry loop still handles an error the
relay reports. The result records `model_client_timeout` (`client`, `version`, `timeout_s`,
`retries`, `source`, `hidden_cap_removed`, `replaces`) and `agent.relay_timeout_s`. Nothing the
model is sent or allowed changes. `GBENCH_AGENT_MAX_TOKENS_FROM_RUN` (default on since 2026-10-11; `=0` turns it off) adds
`max_tokens` = the run's `--max-output-tokens` to `model_kwargs` (the driver then resends a turn
the server refused only because of that cap without it, and counts it); both limits stay sized
for the window, since such a resent turn may use all of it; the bundled config sends none, so such a run records
`official_sampling.agent_max_tokens_from_run` and `max_output_tokens_per_turn`, and is not
comparable.

Run knobs reach the agent through litellm (`openai/<served model>` at the endpoint's `/v1`):

| Knob | Effect on the agent |
| --- | --- |
| `--temperature` / `GBENCH_SWE_BENCH_MULTILINGUAL_TEMPERATURE` | replaces gbench's rule for every suite: 1.0 with `--thinking` (`temperature_source` `"thinking"`), 0.0 without (`temperature_source` `"default"`); `GBENCH_SWE_BENCH_MULTILINGUAL_TEMPERATURE` wins over `--temperature`. The temperature is always sent; the bundled config sends none, so an agentic run is not comparable |
| `--eval-thinking` | `chat_template_kwargs.enable_thinking` on every request |
| `--max-output-tokens` | sent as every agent turn's `max_tokens` (in mini-swe-agent's `model_kwargs`), on by default since 2026-10-11, a recorded deviation (the bundled config sends none). `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0` restores upstream's none (the server grants each turn what the context window has left). The one-shot mode sends it as its single request's `max_tokens` |
| `--sandboxes` (or the eval concurrency) | parallel agent rollouts, and the harness `--max_workers` |
| `--eval-limit` / `--shard` | stratified subset by repository / round-robin shard |
| `GBENCH_REPETITION_PENALTY` etc. | forwarded (and recorded as a non-comparability reason) |
| `GBENCH_SWE_BENCH_MULTILINGUAL_TRAJ_DIR` | where the trajectories go (default: `<run results dir>/agent_trajectories/`) |
| `GBENCH_CACHE_DIR` | root of the pinned dataset snapshot (default `~/.cache/gbench`) |
| `GBENCH_MINI_SWE_AGENT_V2_IMAGE` / `GBENCH_SWEBENCH_V5_IMAGE` | image names (defaults `gbench-mini-swe-agent-v2:2.4.2-2.4.6` / `gbench-swebench-v5:5.0.2`) |
| `GBENCH_SWE_OMP_THREADS` | OpenMP/BLAS thread cap in every agent and harness task container (`0` disables) |

Each instance's image is pulled on the host before its agent container starts
(`swebench_common.host_pull_images` from the gbench process: `docker image inspect`, then
`docker pull` only for a missing image), so the pull uses your Docker client configuration
(`docker login`, credential helpers, `DOCKER_CONFIG`); the agent container gets none
(`HOME=/tmp`), so no registry credential enters it or any task container. An image that cannot
be pulled makes that instance an infrastructure failure (`agent_exit_status:
"ImagePullFailed"`) and no container starts. The one-shot mode's harness pass pulls each
submitted instance's image (the pinned row's `image`) on the host the same way
(`harness_image_pulls`). Instance containers get a 7-day keep-alive and a 1-hour pull timeout.
The bundled config sends no per-turn `max_tokens`. gbench adds the run's `--max-output-tokens` to
mini-swe-agent's `model_kwargs` (default on; `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0`: none, as
upstream). A turn the server refuses only because of that cap is resent with no `max_tokens`, as
upstream sends it. A conversation that outgrows the served context window ends the rollout with
`ContextWindowExceededError` (the model's outcome, as upstream records it).

**One-shot (opt-in).** `GBENCH_SWE_BENCH_MULTILINGUAL_MODE=oneshot`: one request per instance with
the issue text only, a diff extracted from the reply, same v5 harness, the same temperature
rule (0.0 no-think, 1.0 with `--thinking`). Always `leaderboard_comparable: false`.

## Reading the result
- `accuracy` = resolved rate over the selected instances; `category_accuracy` is per repository.
- `mode`, `agent` (scaffold, version 2.4.6, config name and sha256, limits, model, endpoint,
  runner image), `dataset` (the pinned commit), `harness_image`, `swebench_report`,
  `harness_report_retries`, `max_output_tokens_note` (the per-turn cap sent with every turn
  and where it was set), `max_output_tokens_per_turn` (the run's `--max-output-tokens`; `null` only
  with `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0`), `truncated_turns`,
  `empty_after_reasoning_turns` and, on every trace, `trajectory`, `agent_exit_status`,
  `agent_steps`, `agent_patch`, `trajectory_file`. `truncated_turns` counts model replies that
  ended with `finish_reason: length`, which means at that cap (at the end of the served window
  with the cap off); above zero it makes the run not comparable (a reply cut off by the budget
  is not a clean measurement), and `empty_after_reasoning_turns` model replies
  that stopped on their own (not at `length`) with reasoning (`reasoning_content` or
  `reasoning`) but neither content nor a tool call. The trajectory keeps only the parsed reply,
  so they are split by the token ids the relay the 2.x containers reach the endpoint through
  sees (swebench_common.classify_empty_reply): the answer inside the thinking channel, the
  model's outcome, counted with its share of the relay's replies (`answer_in_thinking`,
  `answer_in_thinking_rate` over the relay's `completion_replies`, numerator and denominator the
  same replies) and not a comparability reason; a native tool call the server's tool-call parser
  removed (`tool_call_dropped_turns`, a server-parser artifact) and turns the relay could not
  classify (`unclassified_empty_turns`: no token ids, or empty turns in the trajectories beyond
  the ones the relay classified), each a comparability reason (see
  [answer_in_thinking.md](answer_in_thinking.md)). Both count model replies, whatever message holds them:
  mini-swe-agent 2.x files a reply without a tool call as a role `user` format-error message
  whose `extra.response` is that reply (in `trajectory` such a reply appears as that message,
  with the reply's reasoning and finish reason), so each reply is read from its raw response.
  `empty_after_reasoning_turns` is counted from the responses in the instance's `.traj.json`;
  `truncated_turns` (counted in the container from the same responses) and `agent_steps` (the
  agent's model calls) come from the runner's `result.json`, or from the `.traj.json` when the
  container wrote no result.
- An upstream scaffold interaction, kept as it is (canonical): the bundled mini-swe-agent config
  runs every command with `BASH_ENV=/root/.bashrc`, and on an official image without
  `/opt/miniconda3` that file's conda activation fails with `/opt/miniconda3/...: No such file
  or directory` on every command, which can hide the submit command's output, so the zero it
  causes reads like an ordinary format error. Each trace counts the turns whose observation shows
  it (`bash_env_conda_turns`, from the `.traj.json`), and the result counts the instances with
  any (`submit_hidden_by_bash_env`); nothing is scored differently.
- A count that could not be measured for an instance is `null` on its trace, never 0, and
  named in the trace's `unmeasured_counters`: `empty_after_reasoning_turns` when the instance
  made model calls and its `.traj.json` is missing or unreadable; `agent_steps`,
  `truncated_turns` and `empty_after_reasoning_turns` when the container wrote no `result.json`
  and the `.traj.json` is missing or unreadable too (whether any call was made is then
  unknown). The result's `unmeasured_counters` counts the instances per count, the run totals
  cover the other instances, and the run is not comparable.
- `swebench_report` carries v5's `infra_failure_ids`, `ambiguous_failure_ids` and
  `failure_reasons` as the harness reported them: upstream's advisory log signatures (out of
  memory, docker daemon errors, name resolution, ...; matched partly in the test output the
  patched code writes). As upstream specifies, they leave the instance in `unresolved_ids` /
  `error_ids` and change neither the resolved rate, the status nor `leaderboard_comparable`.
- `harness_report_retries` counts re-runs of the harness's final report step alone (swebench's
  own `make_run_report`, with the scoring pass's arguments; at most 5, the first at once, then
  after 5, 15, 30 and 60 s) when the harness scored instances but that step crashed (its
  container listing races with containers other processes remove on a shared daemon). Such a
  re-run starts no container and grades nothing: it reads the per-instance `report.json` files
  the scoring pass wrote, so an instance that pass left without one (a test timeout, a crash
  after the tests ran) stays in `error_ids`, exactly as in a run whose report step did not
  crash. If all 5 fail, the run is `status: "error"`.
- An errored instance (`error_ids`) is classified by its `run_instance.log`, reading only the
  records the harness itself logs: a patch that did not apply, or tests over the applied patch
  that ran past the harness's 1800 s timeout (`Test timed out after 1800 seconds.` after the
  `Test runtime:` record), is the model's outcome, an ordinary unresolved instance in the
  denominator, as upstream counts every `error_ids` instance (the gold patch finishes within
  the timeout, so a hang is what the model's patch did; traces carry `test_timed_out`, the
  report `test_timeout_instances` / `test_timeout_ids` and `patch_apply_failed_instances`).
  `agent_output_dir` is a fresh directory under `<run results dir>/agent_trajectories/` (next to
  the result JSON), or in the system temp dir outside a gbench run. `temperature` /
  `temperature_source` is the temperature sent: 0.0 ("default"), 1.0 with `--thinking`
  ("thinking"), or the value you set.
- Infrastructure failures (an image that could not be pulled, a container that did not start,
  an endpoint that failed: `harness_error: "agent_infra"`; an errored instance whose
  `run_instance.log` shows neither a patch-apply failure nor a test timeout (docker, image build,
  container or a missing log): `harness_no_verdict`; every submitted patch of a pass that wrote
  no report: `harness_no_report`) are harness errors: left out of `accuracy`'s
  numerator and denominator (the denominator is `scored_instances`), counted by kind in
  `harness_errors` (ids in `harness_error_ids`, also `failed_requests`) and listed under
  `partial_evaluation`, the run `completed_with_errors`; if nothing could be scored the run is
  `status: "error"` with `accuracy` `null` (`accuracy_withheld_reason`), never a 0%.
- `leaderboard_comparable` is `true` only for an agentic run over the full, unsharded split,
  with no temperature sent (gbench always sends one, so today no run is comparable), no
  decoding penalties, every instance scored, every count
  measured, no model turn cut off or empty after reasoning, and the bundled config at sha256
  `9a9c86ac...`
  (`agent.config_sha256`); the reason lists each
  deviation otherwise. Comparable means comparable to the SWE-bench maintainers' submission
  `evaluation/multilingual/20260902_mini-v2.4.6_gemini-3-5-flash` in `SWE-bench/experiments`
  (merged 2026-09-02, commit `8fa0575c`): mini-swe-agent 2.4.6 with its bundled config, this
  dataset commit and the v5 harness. The displayed swebench.com Multilingual board (last
  deployed 2026-09-01 02:53 UTC, `SWE-bench/swe-bench.github.io@193160a4`) has no 2.4.x row
  yet: its rows are mini-SWE-agent 2.0.0a0 / 2.0.0 (whose bundled config still set temperature
  0.0), run on the pre-v5 dataset and harness.

## Prerequisites
- **Docker Engine** (reachable daemon at `/var/run/docker.sock`), disk for 300 instance images
  (tens of GB), Docker Hub access. gbench pulls the instance images with the host's Docker CLI
  from its own process, so `docker login` on this host (recommended for the pull-rate limit)
  applies to them.
- **The two local images** (gbench never pulls them; each needs network access at build time):
  ```bash
  docker build -t gbench-mini-swe-agent-v2:2.4.2-2.4.6 -f gbench/docker/mini_swe_agent_v2.Dockerfile gbench/docker
  docker build -t gbench-swebench-v5:5.0.2 -f gbench/docker/swebench_v5.Dockerfile gbench/docker
  ```
  The harness image also carries the Docker CLI at a pinned version
  (`docker-ce-cli=5:29.8.1-1~ubuntu.24.04~noble`, label `org.gbench.docker_cli`; its Debian
  packages come from the Ubuntu archive snapshot `20260925T120000Z` on `ubuntu:24.04` pinned by
  digest and are recorded in `/opt/swebench-dpkg-versions.txt`), which swebench 5.x uses to stop and
  remove each task container. The agent image installs both mini-swe-agent virtualenvs against
  `gbench/docker/mini_swe_agent_v2.constraints.txt` (the build fails on any other set) and carries
  the same pinned Docker CLI from the same Ubuntu snapshot (label `org.gbench.docker_cli`, recorded
  in `/opt/gbench/dpkg-versions.txt`). The prerequisite check refuses an agent image without
  mini-swe-agent 2.4.6 (label `org.gbench.mini_swe_agent_versions`), a harness image other than swebench 5.0.2
  (label `org.gbench.swebench`) or one without the Docker CLI (rebuild it with the command
  above), and runs the v5 harness once on the snapshot (network off) to build every TestSpec.
- **Python packages** on the host: `datasets` (to write the snapshot; a base
  dependency of gbench).
- A served model at an OpenAI-compatible `/v1` endpoint **with tool calling enabled** (vLLM:
  `--enable-auto-tool-choice --tool-call-parser <the model's parser>`, e.g. `gemma4`; a server gbench launches itself (no `--remote-endpoint`) gets both from `--tool-call-parser NAME`, and the CLI refuses a local run of this suite without it);
  `VLLM_API_KEY` (else `OPENAI_API_KEY`) if it needs a key (gbench's relay adds it to each request outside the agent
  containers, which hold only `EMPTY`; it is never written to the spec or trajectory files, and a
  credential value found in the runner's `result.json` or trajectory file is replaced with
  `[REDACTED]`). An endpoint failure the relay saw is a harness error (`endpoint_failures`,
  `completed_with_errors`, not comparable).
- **docker's default bridge network able to reach this host's gateway address** (e.g.
  `172.17.0.1`, where the relay listens on a free port): a host firewall that drops traffic
  from `docker0` to the host makes the run `infra_required` at the relay check.
- Hugging Face Hub access on the first run (to write the pinned snapshot), or the snapshot
  already in `$GBENCH_CACHE_DIR`.
- **On a cloud VM, the metadata server blocked for docker containers.** The model's commands
  (in each instance's task container) and the tests over its patch (in the harness's task
  containers) run with network access, and a docker container reaches the VM's metadata server
  at `169.254.169.254`, which hands out the VM's service-account token. Before the first rollout
  (in the one-shot mode: before the first model call), gbench checks that it is unreachable from
  docker's default bridge and from a user-defined bridge network (gbench creates a temporary
  one, `gbench-metadata-probe-<id>`, and removes it right after): a short-lived probe container
  of `gbench-mini-swe-agent-v2:2.4.2-2.4.6` (in the one-shot mode `gbench-swebench-v5:5.0.2`; `--pull never`)
  opens and closes a TCP connection to `169.254.169.254:80` and sends nothing. Each network's
  verdict is cached for the rest of the run, so the harness pass does not probe again. A
  connection that opens, or a probe without a clear verdict, **hard-errors** (`infra_required`)
  naming the fix. Block it once on the host; gbench never runs sudo or changes the host's
  firewall itself:
  ```bash
  sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
  sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save   # keep it
  ```
  See [toolchains.md](toolchains.md#cloud-vms-the-metadata-server).

Missing Docker, either image, the pinned dataset or a buildable TestSpec, or a failed metadata
server check, **hard-errors** (`infra_required`); the suite never skips.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals swe_bench_multilingual \
       --max-output-tokens 32768 --sandboxes 8
```
Add `--eval-limit 20` for a stratified smoke run (not comparable). The first run pulls sizable
images.
