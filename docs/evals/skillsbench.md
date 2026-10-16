# skillsbench

Canonical **SkillsBench** ([benchflow-ai/skillsbench](https://github.com/benchflow-ai/skillsbench)):
a **deterministic, execution-based** agentic benchmark, **87 tasks**, each a self-contained
per-task Docker environment scored by a **pytest verifier** that writes a reward in `[0,1]` to
`/logs/verifier/reward.txt`. **There is no LLM judge anywhere** (the previous gbench runner both
quarantined the suite *and* scored an LLM-judge-over-text; neither is canonical; both are gone).

## How gbench runs it

The agentic run is **delegated to the upstream BenchFlow runner** (`bench eval run`) inside a LOCAL
orchestrator container (`gbench/docker/skillsbench.Dockerfile`), driven **docker-out-of-docker**:
`bench eval run --sandbox docker` builds and runs each **task** container as a sibling on the host
daemon. Inside each task container the agent produces the solution, then the task's deterministic
verifier produces the reward. gbench authors no agent loop and no judge; it drives `bench` directly
(the repo's bundled open-model runner script targets an older BenchFlow and is unused).

> **docker-out-of-docker note.** BenchFlow bind-mounts its working dirs into the sibling task
> containers, and a `docker run -v` from inside a container refers to the **host** filesystem. The
> harness therefore mounts its workdir as an **identity mount** (same path on host and in the
> orchestrator) and points `TMPDIR` there, so the sibling sandbox sees the real files and the reward
> is collected. This was verified: the model-free `oracle` agent scores `reward=1.0` end-to-end.

- **Headline `accuracy`** = mean reward over the tasks in the primary condition (default
  `without-skills` = the model's raw capability), x100.
- **Conditions**: `without-skills` and `with-skills`; when both run, the **with-skills-vs-without
  lift** (upstream's signature metric) is reported.
- **Harness errors are left out of the means, never scored.** A task that produced no reward, and
  a task with reward 0 whose verifier output (the rollout's `verifier/test-stdout.txt` and
  `test-stderr.txt`, read from the work directory after the run) shows a package host under
  `ubuntu.com`, `debian.org`, `astral.sh`, `github.com`, `githubusercontent.com`, `pypi.org`,
  `pythonhosted.org`, `python.org`, `pytorch.org`, `nodejs.org`, `npmjs.org` or `npmjs.com` failing
  to resolve before its tests started (no pytest `test session starts` header), is a harness
  error, the DNS failure when something outside its task container corroborates it (another such
  task in the run, or a fresh probe container of `gbench-skillsbench:9a1f4dd5-h7` after the run, on docker's
  default bridge and a new user-defined network, that cannot resolve one of the hosts either): it
  is listed per condition in `harness_error_tasks` (`kind` `dns` with the hosts, or
  `no_reward`), counted in `harness_errors`, left out of its condition's mean and `n_scored`, and
  the run is `completed_with_errors` and not comparable. An uncorroborated DNS failure, or one
  whose agent transcript (the rollout's `trajectory/` and `agent/` files) names
  `/etc/resolv.conf`, `/etc/hosts` or `/etc/nsswitch.conf`, may be the model's own doing, so the
  reward stands (the uncorroborated ones are listed with a note in `dns_failures_kept`). A run
  with no task left to score hard-errors (`infra_required`) instead of reporting 0.
- **A task with no reward says why.** Its `harness_error_tasks` entry (`kind` `no_reward`) carries
  what the rollout's `result.json` says: `benchflow_error_category` (BenchFlow's own
  `error_category`), `stage`, `agent_ran` (whether the agent made any tool call or used any
  token), `error_excerpt` (the last 600 characters of the error), `compose_command` when a
  `docker compose` command failed, `rollout` (the rollout directory under `artifacts_dir`) and
  `build_retries` when BenchFlow re-ran the environment build. `stage` is, in this order:
  `environment_build` / `environment_start` when the task environment's `docker compose build` /
  `up` failed; `sandbox_command` for another failed compose command (BenchFlow checks `cp`,
  `stop` and `down`, used before the agent, before the verifier and at teardown, so `agent_ran`
  tells which side of the agent it was); `environment_setup` for a sandbox startup failure
  (BenchFlow category `sandbox_setup`); `agent_setup` when the agent's install failed
  (`install_failure`); `verifier` when the verifier errored; `agent` when the agent ran or the
  category is one of the agent's phase (`idle_timeout`, `pipe_closed`, `acp_error`, `api_error`,
  `suspected_api_error`, `provider_auth`, `provider_rate_limit`, `timeout`); else `unknown` (the
  excerpt says what failed). Until a 2026-10-13 review every failure other than a compose build /
  up or a verifier error was labelled `agent`, also when the agent never ran. The task is still left out of
  the mean: it is never scored 0.
- **Environment builds retried after a Docker daemon failure (2026-10-13).** In the 2026-10-13
  validation run (`--eval-limit 5`, thinking on, eight suites at once on one Docker daemon),
  `lab-unit-harmonization` wrote no reward: its environment build failed in step 4/7 after the
  step's `pip install` had finished, with the daemon's containerd image store unable to commit the
  layer (`failed to export layer: CreateDiff: ... failed to commit: rename
  /var/lib/containerd/io.containerd.content.v1.content/ingest/<ref>/data
  .../blobs/sha256/<digest>: no such file or directory`), 78 s into the trial and before the agent
  started (0 tool calls, 0 tokens, empty trajectory). The same Dockerfile built and was scored on
  2026-10-10 and 2026-10-11 (rewards 0.354, 0.312, 0.312). BenchFlow classified the error `other`
  and did not retry it: its docker sandbox retries a failed `docker compose build` twice (2 s and
  5 s later) only for the transient failures it lists. The image's
  `gbench/docker/skillsbench_benchflow_build_retry.patch` adds this containerd failure to that
  list; every other build failure fails the trial as before, and a third failure in a row still
  does. The build runs before the agent starts, so a retry does not change what the model sees.
  The launcher counts BenchFlow's "Retrying Docker build for" warnings per task and the result
  records them as `sandbox_build_retries` (`{condition: {task: n}}`, `{}` when none; `null` from
  an image that does not record them).

### Agent + model wiring (BenchFlow 0.6.3)

The pinned BenchFlow (**0.6.3**, from the repo's own `uv.lock`) does **not** ship the leaderboard's
`opencode` agent; its open-model agents (`deepagents`, `codex`, ...) read
`BENCHFLOW_PROVIDER_BASE_URL` / `BENCHFLOW_PROVIDER_API_KEY`. gbench defaults to **`deepagents`** and
injects the gbench `/v1` endpoint via `--agent-env BENCHFLOW_PROVIDER_BASE_URL=... --agent-env
BENCHFLOW_PROVIDER_API_KEY=...` (plus `OPENAI_BASE_URL`/`OPENAI_API_KEY`). Because the agent differs
from the leaderboard's `opencode`, gbench runs a single trial vs upstream's 3 and gbench sets the
temperature where the leaderboard runs the agent's own default, `leaderboard_comparable` is always
False, but the **deterministic verifier scoring is canonical either way**.

**Credentials.** The agent runs inside each task container and hands its environment to every
command the model runs there, so no key enters any container or command line. The orchestrator
gets `GBENCH_MODEL_API_KEY` by name from the environment of its `docker run` (the command line
carries only the name), and its value is always the placeholder `dummy`, which the launcher hands
to the agent as its API key. When the served endpoint needs a key, set
`GBENCH_SKILLSBENCH_API_KEY` (else gbench uses the endpoint key every suite reads, `VLLM_API_KEY`,
else `OPENAI_API_KEY`): it stays in the gbench process, and the task containers reach the
model through gbench's relay (`swebench_common.EndpointRelay`, a thread of the gbench process
listening at `GBENCH_SKILLSBENCH_TASK_HOST`, default `172.17.0.1`), which forwards
`/v1/chat/completions`, `/v1/completions`, `/v1/models` and `/v1/responses` to the endpoint gbench
was given, adds `Authorization: Bearer <key>` and refuses every other path. The endpoint then only
has to be reachable from the gbench host, and `GBENCH_SKILLSBENCH_TASK_ENDPOINT` is not used. The
result records the relay under `model_relay` (never the key). With none of those set (or only a
placeholder such as `dummy` / `EMPTY`) the endpoint has no key and the relay adds none; the task
containers still reach the model through it, because it sets the run's temperature (below).

## Requirements

- **Docker** with a **reachable daemon whose socket is mounted** (the harness passes
  `-v /var/run/docker.sock` + the identity-mounted workdir; docker-out-of-docker was verified).
- **The `gbench-skillsbench:9a1f4dd5-h7` image, built LOCALLY** (never pulled): clones benchflow-ai/skillsbench
  at a pinned commit, `uv sync --locked`s BenchFlow 0.6.3 on the image's Python 3.12 (recorded in
  `/opt/gbench/pip-freeze.txt`), and installs the Docker CLI **and the `docker compose` v2 plugin**
  (BenchFlow's docker sandbox drives `docker compose`) at pinned versions
  (`docker-ce-cli=5:29.8.1-1~ubuntu.24.04~noble`,
  `docker-compose-plugin=5.5.1-1~ubuntu.24.04~noble`, labels `org.gbench.docker_cli` and
  `org.gbench.docker_compose`; Debian packages from the Ubuntu archive snapshot `20260925T120000Z`,
  recorded in `/opt/gbench/dpkg-versions.txt`):
  ```bash
  docker build -t gbench-skillsbench:9a1f4dd5-h7 -f gbench/docker/skillsbench.Dockerfile gbench/docker
  ```
  The build applies `gbench/docker/skillsbench_benchflow_transport.patch` to BenchFlow's
  deepagents agent, checking each file's sha256 before and after: the agent's ChatOpenAI gets the
  request timeout gbench passes (`GBENCH_MODEL_REQUEST_TIMEOUT_S`), and the agent install in each
  task container pins `deepagents==0.7.23` and `langchain-openai==1.7.0` (the latest releases when
  the 2026-10-10 gbench run resolved them; their dependencies with `--exclude-newer
  2026-10-10T19:31:32Z`) instead of whatever PyPI serves at run time (labels
  `org.gbench.deepagents`, `org.gbench.langchain_openai`, `org.gbench.benchflow_shim_sha256`,
  `org.gbench.benchflow_registry_sha256`).
  It also applies `gbench/docker/skillsbench_benchflow_build_retry.patch` to BenchFlow's docker
  sandbox (`benchflow/sandbox/docker.py`, sha256 checked before and after, label
  `org.gbench.benchflow_docker_sandbox_sha256`): a task environment build that fails because the
  Docker daemon's containerd store could not commit a layer is retried like BenchFlow's other
  transient build failures (above).
  The image carries the label `org.gbench.skillsbench-harness=7` (the adapter that applies
  `--shard`, stops on a malformed `GBENCH_SHARD` instead of running the whole task set, draws an
  `--eval-limit` subset with a seeded draw balanced across the task categories, records the task
  lists it ran, runs the with-skills condition with `--skill-mode with-skill`, deals
  `--eval-limit` over the conditions of a two-condition run, and passes BenchFlow's idle-watchdog
  timeout and the agent's request timeout, and records the environment builds BenchFlow re-ran,
  `sandbox_build_retries`). gbench refuses
  an image without it, or with another value, and **hard-errors** (`infra_required`) with the build
  command: an older image could ignore a `--shard` (running the whole task set), take the first
  tasks of the sorted list for `--eval-limit` and fail the with-skills condition. Rebuild after pulling gbench changes to `gbench/docker/skillsbench_*`.
  The result records `image` and `image_harness_version`.
- **Network egress**: ~28 of the 87 task images fetch public toolchains/data at build time (uv,
  nodesource, pytorch cpu wheels, apt, a few github/arxiv/HF pulls; all public, no auth), and the
  verifier `pip install`s pytest.
- **A served model at `/v1` reachable FROM the sibling task containers.** The task containers are on
  docker networks; they reach gbench's relay at `GBENCH_SKILLSBENCH_TASK_HOST` (default
  `172.17.0.1`, the bridge gateway) on every run (the relay sets the run's temperature, above), so
  the endpoint only has to be reachable from the gbench host. `GBENCH_SKILLSBENCH_TASK_ENDPOINT` is
  not used.
- **On a cloud VM, the metadata server blocked for docker containers.** The agent runs the model's
  commands in each task container, with network access, and a docker container reaches the VM's
  metadata server at `169.254.169.254`, which hands out the VM's service-account token. Before the
  orchestrator starts, gbench checks that it is unreachable from docker's default bridge and from a
  user-defined bridge network, the kind BenchFlow's `docker compose` sandboxes use (gbench creates a
  temporary one, `gbench-metadata-probe-<id>`, and removes it right after): a short-lived probe
  container of `gbench-skillsbench:9a1f4dd5-h7` (`--pull never`) opens and closes a TCP connection to
  `169.254.169.254:80` and sends nothing. Each task's verifier installs its test tools in the task
  container at run time (`apt-get` from the Ubuntu or Debian archive, uv from `astral.sh`, `uvx`
  fetching Python from GitHub and the test packages from PyPI), and so does BenchFlow's agent
  install, so the same probe container also resolves `archive.ubuntu.com`, `deb.debian.org`,
  `astral.sh`, `github.com`, `pypi.org` and `files.pythonhosted.org`
  (`swebench_common.TASK_INSTALL_HOSTS`); a name that does not resolve hard-errors too, naming the
  DNS-preserving rule below. A connection that opens, or a probe without a clear verdict,
  **hard-errors** (`infra_required`) naming the fix. Block it once on the host; gbench never runs
  sudo or changes the host's firewall itself:
  ```bash
  sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
  sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save   # keep it
  ```
  See [toolchains.md](toolchains.md#cloud-vms-the-metadata-server).
- **No GEMINI key, no paid API**: scoring is deterministic. The 14 credentialed/GPU tasks live in
  upstream's `tasks-extra/` and are excluded by default (only `mhc-layer-impl` is excluded from
  `tasks/`).
- The model-free **`oracle` agent** (`GBENCH_SKILLSBENCH_AGENT=oracle`) runs each task's reference
  solution + verifier, a pipeline smoke needing only Docker + egress (no served model).

## Environment knobs

| var | effect |
| --- | --- |
| `GBENCH_SKILLSBENCH_TEMPERATURE` | the temperature gbench's relay sets on every model request, in both modes (default 1.0 with `--thinking`, 0.0 without, or `--temperature`) |
| `GBENCH_SKILLSBENCH_IMAGE` | override the image name (default `gbench-skillsbench:9a1f4dd5-h7`) |
| `GBENCH_SKILLSBENCH_CONDITIONS` | comma list from `{without-skills,with-skills}` (default `without-skills`) |
| `GBENCH_SKILLSBENCH_AGENT` | agent (default `deepagents`; `oracle` for a model-free smoke) |
| `GBENCH_SKILLSBENCH_TASK_HOST` | bridge host the task containers use to reach `/v1` (default `172.17.0.1`): where gbench's relay listens, which runs on every run |
| `GBENCH_SKILLSBENCH_TASK_ENDPOINT` | no longer used: the task containers always reach the model through gbench's relay (a warning says so when it is set) |
| `GBENCH_SKILLSBENCH_API_KEY` | the served endpoint's key (else `VLLM_API_KEY`, else `OPENAI_API_KEY`); stays in the gbench process, added by its relay (unset, `dummy` or `EMPTY`: no key) |
| `GBENCH_SKILLSBENCH_TASKS` | comma list of task ids to run (only without `--eval-limit`: with `--eval-limit` set it is ignored with a warning) |
| `GBENCH_SKILLSBENCH_TIMEOUT_S` | container timeout (default 24h; full runs are long) |

`--eval-limit N` draws N tasks of the run's shard with a fixed seed, balanced across the tasks'
categories (`task.md` `metadata.category`; the launcher's copy of `sampling.stratified_indices`,
seed `skillsbench`), not the first N of the sorted list; the result records it
(`selection.limit_draw`), and gbench refuses an image whose launcher does not report it (rebuild
the image). `--eval-limit N` is **N rollouts for the whole run**, never N per condition: with
one condition (the default, without-skills) that is N tasks; with both
(`GBENCH_SKILLSBENCH_CONDITIONS=with-skills,without-skills`) gbench deals the N over the two
conditions (`sampling.allocate_limit`, seed `skillsbench`; `--eval-limit 5` is 3 + 2) and passes
the shares as `SKILLSBENCH_CONDITION_LIMITS`; the launcher draws each share with the same seeded
draw, so the smaller condition's tasks are a subset of the larger one's, and
`with_skills_lift` is computed over the tasks both conditions scored
(`raw_summary.with_skills_lift_tasks`). The run logs its
planned workload on one line at its start, e.g. `[skillsbench] --eval-limit 5: 5 sample(s) over
1 condition(s) (without-skills 5) x 1 attempt = 5 rollouts`, and records it as
`eval_limit_plan`. A sample here is one (condition, task) rollout, so with both conditions the
nested draws cover fewer distinct tasks than N (`--eval-limit 5` is 5 rollouts over 3 distinct
tasks); the plan records that count as `distinct_tasks`. `--eval-limit` wins over
`GBENCH_SKILLSBENCH_TASKS`: with a limit set the task list is ignored (a warning names it) and the
N rollouts are drawn as above; without `--eval-limit` the named tasks run in every condition. Run
both conditions for the canonical signature; expect a long wall-clock (per-task image build +
agent + verifier).

**Shard (`--shard I/N`, `GBENCH_SHARD`).** gbench validates the value (a malformed one is an error
before anything runs) and passes the normalized `I/N` to the launcher, which takes task I, I+N, ...
of the sorted task list (or of `GBENCH_SKILLSBENCH_TASKS`) before `--eval-limit`, and itself stops
(exit 1) on a malformed value rather than run the whole set. The result records the shard identity
(`shard`, `shard_index`, `shard_count`, `shard_applied`), `tasks_selected` and, for a partial
shard, `sharded run (--shard I/N) ...` in `leaderboard_comparable_reason`; an empty shard reports
`accuracy: null` with `accuracy_withheld_reason`. Each (condition, task) is a trace in
`sample_traces` (`condition`, `task`, `reward`, a harness error with `scoring_excluded`,
`sample_key`); `gbench --merge` recomputes each condition's mean reward over the union of the
shards' traces (the suite's `merge_shards`), and sums the shards' `harness_errors`, unites their
`harness_error_tasks` and `sandbox_build_retries` (null only when no shard's image records
retries); until 2026-10-13 those three were the first shard's.

**Thinking.** gbench's relay sets the chat template's thinking switch
(`chat_template_kwargs.enable_thinking`: true with `--thinking`, false without) on every
chat-completions and completions request, which deepagents' ChatOpenAI never sends. The result
records `thinking` (the run's `--thinking`), `thinking_applied: true` and the relay's
`model_relay.sets_body_fields` / `sets_thinking_switch_on`. `thinking_applied` is read from the
relay: it is `false`, with `thinking_not_applied_reason`, for the `oracle` agent (no model
requests), for the `codex` agent (the Responses API, `/v1/responses`, is relayed unchanged), and
when the relay counted no chat-completions reply. Before 2026-10-11 the switch was not
sent (`thinking_applied: false`): a `--thinking` run reached the model with thinking off, so those
results are not comparable with later ones.

**Token limit.** deepagents sends no `max_tokens`. gbench's relay puts the run's
`--max-output-tokens` on every request that carries none. The result records
`agent_max_tokens_from_run`, `max_tokens_sent`, `max_output_tokens_note` and
`model_relay.max_tokens_from_run`, and names the cap in `leaderboard_comparable_reason`.
`GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0` turns it off.

**Transport timeouts.** BenchFlow's ACP idle watchdog (`bench eval run --agent-idle-timeout`, 600 s
by default) aborts a trial when the agent sends no ACP event for that long, and BenchFlow then
re-runs it. The deepagents shim sends events per LangGraph step, so a single model turn longer
than 600 s (a long thinking turn, or a slower endpoint under load) was killed and re-run: on the
2026-10-10 smoke run `debug-trl-grpo` was killed with "Agent idle for 600s" and re-run, losing
691 s. The watchdog is a BenchFlow transport default, not SkillsBench's protocol: SkillsBench's
task-creator notes call it "benchflow's idle-600s pitfall" (benchflow issue #211), its leaderboard
export drops idle-failure trials as failed retries, and BenchFlow's help says disabling it falls
back to the task's wall-clock timeout, the task's `[agent].timeout_sec`, which stays the bound of
every trial. gbench sets it to T, `base.request_timeout_s` of the served context window (33,968 s
for a 262,144-token window; deepagents sends no `max_tokens`). The agent's ChatOpenAI had no request
timeout at all (checked in the image with the pinned packages against a slow local server,
2026-10-11), so it hid no cap, but a request the endpoint never answered waited until the task's
timeout; it now gets the same T (retries: the OpenAI SDK's default, 2), recorded with
`hidden_cap_removed: false`. gbench's relay waits T + 120 s for one request. The result records
`model_client_timeout` (only for `deepagents`, the one agent whose shim reads it) and
`agent_idle_timeout` (every agent but `oracle`, which runs without the watchdog)
(`{..., timeout_s, source, replaces, hidden_cap_removed}`) and `launcher_timeouts` (what the
launcher passed). Results from before 2026-10-11 ran under the 600 s watchdog and are not
comparable.

**Temperature.** The model samples at gbench's rule for every suite: temperature 1.0 with
`--thinking`, 0.0 without. `GBENCH_SKILLSBENCH_TEMPERATURE`, else `--temperature`, replaces it in
either mode. The task containers reach the model through gbench's relay (the key relay above,
started on every run even when the endpoint has no key, listening at
`GBENCH_SKILLSBENCH_TASK_HOST`), which sets `temperature` on every chat-completions and
completions request, whatever the agent sent (deepagents calls chat completions; the Responses API
of BenchFlow's codex agent is relayed unchanged). The result records `temperature_applied: true`,
`temperature`, `temperature_source` and the relay under `model_relay` (`sets_body_fields`,
`sets_temperature_on`), and names the temperature in `leaderboard_comparable_reason`: the
leaderboard runs the agent's own default, so a run is never leaderboard-comparable. The launcher
runs each task once per condition, so
`--attempt-count` above 1 is not applied (`attempts.not_applied`).

**Summary row.** `total_questions` is the primary condition's task count, `effective_n` the tasks
scored in it (the mean's denominator, harness errors left out) and `correct_answers` their rewards
summed (a solved-equivalent count, `correct_answers_note`), so the console and CSV row reconcile
with the headline mean reward. The metric names the agent that ran (`agent`).

**Artifacts.** After the run gbench keeps each rollout's `result.json`, `rewards.jsonl`,
`timing.json`, `config.json`, the verifier's `reward.txt`, `ctrf.json`, `test-stdout.txt` and
`test-stderr.txt` (1 MiB each) and the agent's `trajectory/acp_trajectory.jsonl` (4 MiB; a larger
file keeps its first and last halves), with the launcher's `skillsbench_summary.json`, every
credential value redacted, under `<results dir>/skillsbench_artifacts/run_*/<condition>/<task>/`
(`artifacts_dir` in the result). The model-call log `trajectory/llm_trajectory.jsonl` (often
100 MB) is not kept. The task containers write the work directory as root; gbench removes what it
can, then the rest with a throwaway container of the orchestrator image (no network), and logs any
path still left.

## Run

```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals skillsbench --eval-limit 5 \
       --max-output-tokens 65536
```

The result payload carries `mean_reward` (0-1), `condition_means`, `with_skills_lift`, `n_tasks`,
`n_scored`, `harness_errors`, `harness_error_tasks` (a no-reward task with its cause),
`sandbox_build_retries`, `dns_failures_kept` (when any), the raw
summary, a `sampling` note (the temperature the relay set) and the relay's record under
`model_relay`.
