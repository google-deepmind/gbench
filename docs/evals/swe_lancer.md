# swe_lancer setup

> **Status: registered, not in `--evals all`.** `--evals all` leaves `swe_lancer` out
> (`NAMED_ONLY_SUITES` in `gbench/runners/evals.py`): SWE-Lancer's harness edits the Docker
> host's firewall while it runs (iptables rules in the `DOCKER-USER` chain, one set per task
> container, removed afterwards; upstream recommends an ephemeral VM), and a full run pulls
> about 1.3 TB of task images. `--evals swe_lancer` runs it, and it hard-errors
> (`infra_required`) naming whatever is missing.

> **Docker setup:** this suite runs one container/network per task and can exhaust the
> Docker address pool at high `--sandboxes`. See [docker-sandboxes.md](docker-sandboxes.md)
> for the recommended `default-address-pools` config and pre-run prune, strongly advised
> before running at concurrency.

## What it measures
**SWE-Lancer** (OpenAI): real freelance software-engineering tasks from the Expensify repository,
each priced at what it paid. gbench runs it the way its current upstream does:
`openai/frontier-evals` `project/swelancer` at `51052cede8cc608f95bb00346635e03759013e5a`
(`openai/SWELancer-Benchmark` is archived and its README points there; `project/swelancer` last
changed in `5305e936`, 2025-12-02).

- **Tasks**: `all_swelancer_tasks.csv` at that commit (sha256
  `5c3a6d4570b49be0d9fced98f5b32487420b16f25c98d6658830e31fa03f049a`), 463 tasks, all in the
  Diamond set: **198 IC SWE** tasks ($189,300; the release of 2025-07-17 kept 198 of the paper's
  237, "adjusted and verified to run successfully offline") and **265 SWE Manager** tasks
  ($264,500). Both task types run by default, each as its own upstream run
  (`swelancer.task_type=ic_swe` / `swe_manager`; upstream refuses `all`).
- **Agent**: SWE-Lancer's own **SimpleAgentSolver**
  (`swelancer/solvers/swelancer_agent/solver.py`). It sends the task prompt
  (`swelancer/prompts.py`) plus its instructions, then for up to 40 turns runs the one
  Python block of each reply in the task container (or the user tool on `<user-tool>`, or stops
  on `DONE`), and grades.
- **Grading**: SWE-Lancer's own. An IC SWE task is solved when its hidden end-to-end tests pass
  (`ansible-playbook run_tests.yml`, up to `n_test_runs=3` runs until one passes); a SWE Manager
  task when the proposal it writes to `/app/expensify/manager_decisions.json` is the correct one.
  Each solved task earns its price.
- **Environment**: the task images SWE-Lancer publishes, `swelancer/swelancer_x86_<task>:releasev1`
  for IC SWE and `swelancer/swelancer_x86_monolith:releasev1` for SWE Manager (upstream runs
  manager tasks only with that image), pulled by the harness (`LocalConfig(pull_from_registry=
  True)`), with the internet disabled for each task container (`disable_internet=True`, the
  default, and the only setting upstream considers valid).

The run command is upstream README's, for each task type:

```text
swelancer.split=diamond swelancer.task_type=<ic_swe|swe_manager>
swelancer.solver=swelancer.solvers.swelancer_agent.solver:SimpleAgentSolver
swelancer.solver.model=openai/<served model>
swelancer.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime
swelancer.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig
swelancer.solver.computer_runtime.env.pull_from_registry=True
swelancer.docker_image_prefix=swelancer/swelancer_x86 swelancer.docker_image_tag=releasev1
[swelancer.use_single_image=True for swe_manager]
runner.concurrency=<--sandboxes> runner.experimental_use_multiprocessing=False
runner.enable_slackbot=False runner.recorder=nanoeval.recorder:dummy_recorder runner.max_retries=2
```

plus `swelancer.runs_dir` (the run directory), `swelancer.report_path` (below) and, for a subset
run, `swelancer.taskset`.

## How gbench runs it
- **The harness** runs in the local image `gbench-swe-lancer-harness:51052ced`
  (`gbench/docker/swe_lancer_harness.Dockerfile`): `ubuntu:24.04` pinned by digest with its
  packages from the Ubuntu archive snapshot `20260925T120000Z`, the pinned
  `frontier-evals` tree fetched by commit, `uv sync --frozen --no-dev` from the project's own
  `uv.lock` (uv 0.12.12; the resolved set is recorded in `/opt/swe-lancer-pip-freeze.txt`), the
  task table checked against the sha256 above, and tiktoken's encoding cached at build time.
  `gbench/docker/swe_lancer_eval.py` starts `swelancer/run_swelancer.py`'s own `main` with the
  same command line. Its one addition is a report: nanoeval returns only a summary, so the eval
  is SWE-Lancer's `SWELancerEval` with a `report_path` field whose `get_full_summary` calls
  upstream's unchanged and then writes one row per task (question id, task type, price, solved,
  dollars earned, the system error of a rollout that failed) for gbench to read.
- **The task table** gbench selects tasks from is read from the image once (a
  `docker run --network none` of the image's Python over `all_swelancer_tasks.csv`: question id,
  task type, set, price, and the file's sha256) and cached in
  `swe_lancer/issues_manifest.json` under gbench's cache root (`GBENCH_CACHE_DIR`, else
  `~/.cache/gbench`). A cache file that does not hold that
  table with the pinned sha256 (for example the list of issues an older gbench kept under the
  same name) is read again from the image and replaced; a table in the image with another
  sha256 stops the run with an error.
- The harness container runs **docker-out-of-docker** on the host network with `NET_ADMIN` and
  `NET_RAW`: alcatraz starts each task container on the host daemon through the mounted socket
  and, to disable its internet, adds rules tagged `alcatraz_block` to the host's `DOCKER-USER`
  (and `INPUT`) chains, removing them when the container is cleaned up. The image's entrypoint
  picks the iptables backend (nft or legacy) whose tables hold Docker's `DOCKER-USER` chain,
  and gives the run directory back to your uid when the run ends.
- **The docker socket, kept as upstream designed it.** The socket, the host network and
  `NET_ADMIN`/`NET_RAW` are what alcatraz needs to start the task containers and to cut their
  internet on the host; the model's commands run in the task containers, which get no socket.
  The residual risk: the harness process, which turns the model's replies into those commands,
  holds root-equivalent access to the host's docker daemon, the host's network namespace and the
  right to change the host's firewall rules; a flaw in SWE-Lancer, nanoeval or alcatraz that let
  model output run inside the harness container would reach all three. gbench keeps upstream's
  design (the harness cannot run without them) and keeps credentials out of that container: it
  gets `OPENAI_API_KEY=EMPTY` and no Docker client configuration, and the relay (below) adds the
  endpoint's key.
- **The model**: SimpleAgentSolver builds its OpenAI client with `OPENAI_API_KEY` and, for an
  `openai/<model>` name, no `base_url`, so the SDK reads `OPENAI_BASE_URL`. gbench sets
  `swelancer.solver.model=openai/<served model>` and points `OPENAI_BASE_URL` at its relay
  (`swebench_common.EndpointRelay`, a thread of the gbench process on 127.0.0.1). The relay
  forwards only GET and POST requests for `/v1/chat/completions`, `/v1/completions` and
  `/v1/models` to the served endpoint (any other path is refused with 400) and adds the
  endpoint's key; the
  harness itself gets `OPENAI_API_KEY=EMPTY`, so no key is in its environment or the logs it
  writes. The relay also sets `chat_template_kwargs.enable_thinking` from `--eval-thinking` on
  every chat request: that is how the served model's reasoning switch is set, and
  SimpleAgentSolver's request cannot carry it.

> **CRITICAL: running upstream as shipped benchmarks the wrong model.** SimpleAgentSolver's
> default model is `openai/gpt-4o`, and an unconfigured run produces a real, plausible SWE-Lancer
> number for **GPT-4o**, with nothing in the output saying so. gbench always passes
> `swelancer.solver.model=openai/<served model>` and `OPENAI_BASE_URL` (the relay), and the result
> records both (`agent.model`, `relay.relays_to`).

- **Sampling**: SimpleAgentSolver sends no temperature and no `max_tokens` (`get_model_response`:
  `chat.completions.create(messages=..., model=..., reasoning_effort=NotGiven)`). gbench's relay
  puts the run's `--max-output-tokens` on every request as `max_tokens`
  (`relay.max_tokens_from_run`, `agent_max_tokens_from_run`, `max_tokens_sent`, named in
  `leaderboard_comparable_reason`; `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0` turns it off, and then the
  server grants the room left in the context window). The relay sets the temperature on every
  request by gbench's rule for every suite: 1.0 with `--thinking`
  (`temperature_source` `"thinking"`), 0.0 without (`temperature_source` `"default"`). An explicit
  `--temperature` or `GBENCH_SWE_LANCER_TEMPERATURE` (which wins) replaces it. The temperature is
  recorded and, since SimpleAgentSolver sends none, makes the run not comparable. SimpleAgentSolver trims old messages
  once a conversation passes 110,000 tokens (its default for a model it does not know, counted
  with gpt-4's tokenizer); a served context window smaller than that makes the endpoint refuse
  turns, which upstream retries and then records as a system error.
- **Request timeout**: SimpleAgentSolver's `AsyncOpenAI` client keeps the OpenAI SDK's default
  600 s request timeout and 2 retries (gbench does not change them). A turn that needs more than
  600 s of decode is abandoned and sent again. The relay stops the abandoned generation and counts
  the close in `relay.client_timeouts_suspected` / `relay.client_close_ages` (see
  [relay.md](../relay.md)). Removing this cap is a code change, not yet made.
- `.env`: the harness gets `sample.env`'s values (`USE_WEB_PROXY=false`,
  `EXPENSIFY_URL=https://www.expensify.com/`, `NEW_EXPENSIFY_URL=https://new.expensify.com/`).

## Requirements
- **Docker** on a Linux host, with the socket at `/var/run/docker.sock`, a daemon that manages
  iptables (the default), and permission to run a container with `--network host --cap-add
  NET_ADMIN`. The prerequisite check runs the image's probe, which only lists the
  `DOCKER-USER` chain.
- **The harness image, built LOCALLY** (gbench never pulls it):

  ```bash
  docker build -t gbench-swe-lancer-harness:51052ced -f gbench/docker/swe_lancer_harness.Dockerfile gbench/docker
  ```

  The check refuses an image whose `org.gbench.frontier_evals` label is another commit or whose
  `org.gbench.swelancer_tasks_sha256` is another table. `GBENCH_SWE_LANCER_IMAGE` overrides the
  name.
- **Disk and network**: the harness pulls the task images from Docker Hub (`swelancer/*`): each
  IC SWE image is about 6.4 GB (198 of them) and the monolith about 5 GB. alcatraz also starts an
  `alpine/socat` side container per task (for its Jupyter port). alcatraz pulls every image of
  every task itself, inside the harness container, with the docker SDK
  (`alcatraz/clusters/local.py` `_start`: `pull_from_registry=True`, as upstream's README runs
  it, pulls even an image that is present), and that container gets no Docker client
  configuration, so these pulls are anonymous: Docker Hub's unauthenticated pull limit applies,
  and a `docker login` on the host does not reach them.
- A served model at an OpenAI-compatible `/v1` endpoint (`OPENAI_API_KEY` if it needs a key).
- **On a cloud VM, the metadata server blocked for docker containers.** The model's Python
  blocks and the tests run in the task containers, on the network alcatraz creates for them (a
  user-defined bridge network), and alcatraz disables their internet only once each has
  started; a docker container reaches the VM's metadata server at `169.254.169.254`, which hands
  out the VM's service-account token. Before the first task, gbench checks that it is
  unreachable from docker's default bridge and from a user-defined bridge network (gbench
  creates a temporary one, `gbench-metadata-probe-<id>`, and removes it right after): a
  short-lived probe container of `gbench-swe-lancer-harness:51052ced` (`--pull never`) opens and closes a
  TCP connection to `169.254.169.254:80` and sends nothing. A connection that opens, or a probe
  without a clear verdict, **hard-errors** (`infra_required`) naming the fix. Block it once on
  the host; gbench never runs sudo or changes the host's firewall itself (alcatraz's own
  per-task rules are its own, added and removed while it runs):

  ```bash
  sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
  sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save   # keep it
  ```

  See [toolchains.md](toolchains.md#cloud-vms-the-metadata-server).

If Docker, the socket, the image or the iptables access is missing, or the metadata server check
fails, the suite **hard-errors** (`infra_required`) and names it; it never skips.

**Licenses.** `openai/frontier-evals` (including `project/swelancer` and its task table) is MIT;
gbench fetches it into the local image at build time and vendors none of it. The task images are
the ones SWE-Lancer publishes on Docker Hub (`swelancer/*`), built from that project and the
Expensify/App repository (MIT). `docker/swe_lancer.Dockerfile` (the earlier locally built task
image) is no longer used by the suite.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals swe_lancer \
       --max-output-tokens 65536 --sandboxes 4 --eval-limit 5
```
`--sandboxes` is `runner.concurrency`. `--eval-limit N` runs N tasks for the whole suite,
never N per type: the N are dealt over the task types run (`sampling.allocate_limit`, seed
`swe_lancer`; `--eval-limit 5` is 3 + 2, `--eval-limit 1` runs one type only), each type's share a
seeded draw passed as its `swelancer.taskset` (not comparable). The run logs its planned workload
on one line at its start (e.g. `[swe_lancer] --eval-limit 5: 5 sample(s) over 2 task type(s)
(ic_swe 3, swe_manager 2) x 1 attempt = 5 rollouts`) and records it as `eval_limit_plan`; `--shard I/N` a round-robin share of each type's tasks.
Called from Python, `run_swe_lancer(..., variant="ic_swe")` (or `"swe_manager"`) runs one task
type. Start small: the first IC SWE run of a task pulls its 6.4 GB image.

## Reading the result
- `headline_metrics`: `ic_swe_pass_at_1`, `ic_swe_earned_usd`, `swe_manager_pass_at_1`,
  `swe_manager_earned_usd` (per task type), and `diamond_pass_at_1`, `diamond_earned_usd` over
  the Diamond tasks run. pass@1 is tasks solved / tasks with a verdict (SWE-Lancer's grade:
  score 1 = solved; a system error is left out, below); earned is the sum of the solved tasks'
  prices. `accuracy`, `total_questions`, `scored_tasks` (the pass@1 denominator) and
  `correct_answers` are the Diamond ones; `earned_usd` / `available_usd` the dollar totals over
  the tasks with a verdict; `category_accuracy` has one entry per task type (`total` is its
  tasks with a verdict).
- `settings.<ic_swe|swe_manager>`: that run's tasks, `earned_usd`, `available_usd`,
  `system_errors`, upstream's own `summary` (nanoeval's metrics, `total_earned`, `is_valid`),
  `harness_log` and `runs_dir` (SWE-Lancer's run group: `group.log` and, per task, `run.log`
  with the full conversation). Each trace carries its question id, task type, price, verdict,
  dollars earned, token counts, `run_dir`, and `system_error` when the rollout failed.
- A task whose rollout ended in a system error (container start, setup, grading timeout, an
  endpoint that kept failing: SWE-Lancer's `RolloutSystemError`) got no verdict: a harness error
  (`harness_error: "system_error"` on its trace), left out of pass@1's numerator and denominator
  and of `available_usd`, counted in `harness_errors` (`{"system_error": N}`, with `"missing": N`
  for a selected task the report does not list) and `failed_requests`, and makes that task type
  `completed_with_errors` with `partial_evaluation`; a task type where no task produced a
  verdict, or whose harness wrote no report, is `status: "error"` with pass@1 `null`
  (`accuracy_withheld_reason`), never a 0%.
- `agent_output_dir` is the run directory: a fresh `swe_lancer_*` directory under
  `<run results dir>/agent_trajectories/` (or under `GBENCH_SWE_LANCER_TRAJ_DIR`), holding each
  task type's report (`<task type>_report.json`), harness log (`harness_<task type>.log`) and
  SWE-Lancer's `runs/`.
- `agent` (solver, `max_turns` 40, model, `reasoning_effort` none), `harness` (tree, image,
  task images, `n_test_runs`, `disable_internet`, runner settings, environment), `relay` (where
  it listened, where it forwarded, the paths it allows, the body fields it set, requests relayed
  and refused),
  `temperature` / `official_sampling` / `max_output_tokens_note`, and `reference_note`.
- `leaderboard_comparable` is `true` only for a clean run of both task types on the full task
  set with no temperature sent (gbench always sends one, so today no run is comparable). A task
  type not run, `--eval-limit`, `--shard`, the sent temperature, a system error or a failed task type is named in
  `leaderboard_comparable_reason`. SWE-Lancer publishes no leaderboard; its paper's IC SWE
  Diamond numbers were measured on 237 tasks, so an IC SWE number compares with runs on the
  current 198-task release (`reference_note`).
