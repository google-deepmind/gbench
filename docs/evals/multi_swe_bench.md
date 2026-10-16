# multi_swe_bench (Multi-SWE-bench)

> **Docker setup:** this suite runs one container/network per task and can exhaust the
> Docker address pool at high `--sandboxes`. See [docker-sandboxes.md](docker-sandboxes.md)
> for the recommended `default-address-pools` config and pre-run prune, strongly advised
> before running at concurrency.

## What it measures
**Multi-SWE-bench** (ByteDance Seed): issue resolution in **7 languages** (C, C++, Go, Java,
JavaScript, Rust, TypeScript) over **1,632 instances**, scored by execution-based **resolved
rate**: the patch is applied and the task's tests run inside its Docker image via the project's
**own harness** (`multi_swe_bench.harness.run_evaluation`). Vanilla `swebench` cannot score it:
the prediction schema (`{org, repo, number, fix_patch}`) and per-language test fields differ.

**Dataset.** `ByteDance-Seed/Multi-SWE-bench`, pinned to commit
`56ff018c04a38e27ada1e9d0a6d5839a51f88f0d`. gbench reads only the seven benchmark language
directories (`c/ cpp/ go/ java/ js/ rust/ ts/`: 128 + 129 + 428 + 128 + 356 + 239 + 224 = 1,632
instances). The repository also carries `kotlin/` (105 instances outside the published set; the
harness's `scripts/images_verified.txt` has no Kotlin images) and a `python/` file; neither is
loaded. Files come from `list_repo_files` + per-file `hf_hub_download` (the repo's `load_dataset`
path is broken), and the exact rows that ran are re-materialized into the subset `dataset.jsonl`
the harness consumes, so gold/tests always match the predictions.

## Protocol (default: agentic)
The leaderboard's reference scaffolds are MagentLess, MSWE-agent and MopenHands
(`multi-swe-bench/experiments` `evaluation/<language>/verified/*_{MagentLess,MSWE-agent,MopenHands}_*`).
gbench runs **MSWE-agent** (`github.com/multi-swe-bench/MSWE-agent`, the team's SWE-agent for the
benchmark) at `88217624b637646b886cb0462995c07559e96f58` (its last commit, 2025-04-02) from the
local image `gbench-mswe-agent:88217624`, as its README runs it:

1. **Entry and arguments.** MSWE-agent's parallel entry point, `multirun.py`
   (`RUNNING_THREADS` = `--sandboxes`), with its README / `run.sh` arguments:
   `--cache_task_images True --remove_image False --config_file config/default.yaml
   --skip_existing=True --per_instance_cost_limit 5.00 --print_config=False
   --max_workers_build_image 16`, the served model as `--model_name`, and the pinned rows of one
   language as `--pr_file data/<language>_verified.jsonl` (MSWE-agent reads the language, and
   with it the tools and demo lines of its instance template, from that file name, and keeps one
   data file per process, so gbench runs one container per language). `config/default.yaml` is
   unmodified (sha256 `913fc5fd89b3bdffe261b9cd2e38b2e45ad9612ea57bf314bcda9f7e314b4537`):
   its system and instance templates, the editor and search commands, the Python demonstration,
   `Last5Observations`, DISCUSSION + fenced-command actions.
2. **Instance images.** MSWE-agent starts each instance from the image its vendored harness names
   (`<org>/<repo>:pr-<number>`, lower-cased) and builds that image when it is missing. Its README
   offers the images the team publishes on Docker Hub instead
   (`mswebench/<org>_m_<repo>:pr-<number>`). Before a language's container starts, gbench pulls
   each of its published images with the host's Docker CLI from the gbench process
   (`swebench_common.host_pull_images`: `docker image inspect`, then `docker pull` only for a
   missing image), so the pulls use your Docker client configuration (`docker login`, credential
   helpers, `DOCKER_CONFIG`). The MSWE-agent container gets no Docker client configuration
   (`HOME=/tmp`, no `~/.docker`, no `DOCKER_CONFIG`), so no registry credential enters it or any
   task container. An instance whose image cannot be pulled is not handed to MSWE-agent: it is an
   infrastructure failure (`instance image: docker pull ... failed: ...` in `agent_error`). The
   outcome of every pull is in `image_pulls` (per language) and in
   `<agent_output_dir>/<language>/image_pulls.json`. gbench's driver
   (`gbench/docker/mswe_agent_run.py`) then finds each published image present, tags it with the
   name MSWE-agent looks for (`--pre_build_all_images False`, so MSWE-agent builds nothing), and
   removes the tags it created when the language is done.
3. **Everything inside the rollout is MSWE-agent's own**: it resets the repository under
   `/home/<repo>` to the base commit (`git reset --hard`, `git clean -fdxq`), installs `jq` and
   `build-essential` with apt (the task containers need network access), extends `.gitignore`
   with its per-language ignore script, deletes `/home/fix.patch` (it leaves `/home/test.patch`,
   as the leaderboard's runs did), and gives the model `TITLE:\n<title>\n CONTENT:\n<body>` of
   the first resolved issue. The submission is the patch its `submit` command prints (or its
   autosubmission when a rollout ends another way), sent to the harness verbatim as `fix_patch`.

   **Archived Debian releases.** MSWE-agent's reset runs `apt-get update` and
   `apt-get install -y jq` and gives up on the instance when either fails. The C++ instances on
   the harness's `gcc:7` base (48 of the 1,632: nlohmann/json 31, simdjson 9, Catch2 8, from the
   harness's own image definitions) are Debian 10 (buster) images. Their apt sources name
   `deb.debian.org` and `security.debian.org`, which no longer serve buster ("no longer has a
   Release file", 404), so that reset fails before the model's first turn; when upstream ran
   these instances (its 2025-04 MSWE-agent runs list them as completed) the mirrors still served
   buster. The other bases (Ubuntu 16.04 to 22.04 and `latest`, `node:18`/`node:20`,
   `gcc:11`/`gcc:12`/`gcc:latest`, `golang:latest`, `rust:latest`) are releases the mirrors
   still serve (checked 2026-09-28: `archive.ubuntu.com` answers for xenial and bionic,
   `deb.debian.org` for bullseye). gbench's driver does what SWE-bench_Pro-os's
   `LockedMiniSwe.install` does for old Debian images (it also covers wheezy, jessie and
   stretch, and bullseye once the mirrors drop it): when that first `apt-get update` fails
   because a suite is gone from its mirror, it points the container's apt sources at
   `archive.debian.org` (which serves those releases unchanged: `debian` and `debian-security`;
   the `-updates` suites of wheezy, jessie and stretch, which the archive does not carry, are
   dropped) with `Acquire::Check-Valid-Until` false, and MSWE-agent's own `apt-get update` then
   runs as upstream wrote it. It changes nothing on an image whose `apt-get update` works; a
   first `apt-get update` that fails for another reason (e.g. a mirror that did not answer) is
   followed by MSWE-agent's own `apt-get update`, unchanged, whose result decides as upstream's
   does. When apt still fails the instance stays an infrastructure failure (`agent_infra`, its
   `agent_error` naming what apt did). Each instance it acted on carries `agent_apt_archive` on
   its trace (`applied`, `release`, `apt_error`: the tail of the failed `apt-get update`,
   `apt_update_after`), and the run records `apt_archive_fallback`. Checked on
   `mswebench/simdjson_m_simdjson:pr-949`: the rollout then gets past the reset and runs.
4. **The model.** MSWE-agent's `OpenAIModel` drives only the model names it registers, so the
   driver registers the served model (zero cost, the served context window) and routes it to
   `OpenAIModel`, which reads `OPENAI_API_BASE_URL` and `OPENAI_API_KEY` from its environment:
   gbench's relay (`swebench_common.EndpointRelay` on 127.0.0.1; it forwards only
   `/v1/chat/completions`, `/v1/completions` and `/v1/models`, adds the endpoint's key and sets
   `chat_template_kwargs.enable_thinking` from `--eval-thinking`) and the placeholder `EMPTY`. The
   container gets no other credential. `OpenAIModel` builds its client as
   `OpenAI(api_key=..., base_url=...)`, with the SDK's defaults (openai 1.70.0: a 600 s timeout
   per request, 2 retries, plus `OpenAIModel.query`'s own 3 tenacity attempts), so a model turn
   longer than 600 s (a long thinking turn; the endpoint answers a non-streamed request only when
   it has generated all of it) would be abandoned and sent again, up to 9 times. The driver gives
   that client the timeout in the spec's `client_timeout_s`: the relay's safety cap on a model
   request (its per-turn limit plus 120 s, `EndpointRelay.TOTAL_CAP_MARGIN_S`) plus 60 s for its
   answer (`multi_swe_bench.mswe_agent_timeouts`, `swe_agentic.client_timeout_for_relay_limit`).
   By default (cap on) the relay's per-request limit is `base.request_timeout_s` of the run's
   `--max-output-tokens` (9,392 s at 65,536 tokens at the default `GBENCH_MIN_DECODE_TOK_S` of 8),
   its safety cap that plus 120 s, and the client waits the cap plus 60 s (9,572 s), so the relay
   always answers first. With `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0` both are sized for the served
   context window (the run knob `max_model_len`, else 262,144 tokens: 33,968 s relay limit /
   34,148 s client). (Until 2026-10-11 it was a
   fixed 3,600 s relay / 3,660 s client, `MSWE_AGENT_CLIENT_TIMEOUT_S`, now only the driver's
   fallback for a spec without the field, which is shorter than one full-window reply at that
   rate.) Such a turn is waited for once. Its transport retries stay the SDK's 2: driver 4 sets
   the timeout only, and with the client timeout above the relay's safety cap the SDK never gives
   up on a slow reply first, it retries only an error the relay answers (a client close near the
   client timeout is counted in the relay's `client_timeouts_suspected`). This is harness-only: it
   changes how long the client waits, nothing the model is sent or allowed. The value the client
   carries is reported per language (`openai_client` in `summary.json`) and recorded in the result
   (`agent.openai_client_timeout_s`, `agent.relay_timeout_s`, `agent.openai_client`, and
   `model_client_timeout`: `client`, `version`, `timeout_s`, `retries`, `source`,
   `hidden_cap_removed`, `replaces`). The relay also stops a reply whose client gave up on it
   (below).

   `GBENCH_AGENT_MAX_TOKENS_FROM_RUN` (default on since 2026-10-11; `=0` turns it off) makes the relay set `max_tokens` to the
   run's `--max-output-tokens` on every agent request and sizes both limits for it. MSWE-agent
   sends no max_tokens, so such a run records `official_sampling.agent_max_tokens_from_run` and
   `max_output_tokens_per_turn`, and is not comparable.
5. **Sampling.** The temperature is gbench's rule for every suite, passed as MSWE-agent's own
   `--temperature`: 1.0 with `--thinking` (`temperature_source` `"thinking"`), 0.0 without
   (`temperature_source` `"default"`); `--temperature` / `GBENCH_MULTI_SWE_BENCH_TEMPERATURE`
   replace it. top_p 0.95, run.py / multirun.py `get_args` default (whose temperature default
   is 0.0), which its README command keeps; no `max_tokens` (`OpenAIModel.query` sends none),
   with or without `--eval-thinking`.
6. **The 50-step limit.** MSWE-agent's README passes `--per_instance_api_calls_limit 50`, an
   option its public code does not define, and every trajectory of the leaderboard's MSWE-agent
   rows (`ByteDance-Seed/Multi-SWE-bench_trajs`; 2,635 trajectories of 12 archives checked: C, C++,
   Go and Java; Claude-3.7-Sonnet, GPT-4o, DeepSeek-R1, Gemini-2.5-pro, Llama-4-Maverick,
   Qwen2.5-72B) stops at 50 model steps with
   `exit due to runtime error: [<instance>] Exceed maximum number of steps: 50`, a RuntimeError
   from the model call that MSWE-agent handles as `exit_error` with autosubmission. The driver
   raises exactly that on an instance's 51st model step.
7. **Java.** MSWE-agent's vendored harness does not register six repositories of the Java set
   (`apache/dubbo`, `fasterxml/jackson-core`, `fasterxml/jackson-databind`,
   `fasterxml/jackson-dataformat-xml`, `google/gson`, `googlecontainertools/jib`: the 78
   SWE-bench-java instances), so it cannot load that set at all, while the leaderboard's
   MSWE-agent Java rows cover all 128 instances. The image adds their definitions unchanged from
   `multi-swe-bench/multi-swe-bench` at `e0dcb3c4ce0ae27997fb213f2a5294e1eaa487b8` (the commit
   that announced them), with `from __future__ import annotations` added so they import on Python
   3.9; they inherit the vendored harness's image naming. Every one of the 1,632 instances loads.
8. **Scoring.** gbench writes `preds.jsonl` + `config.json` and runs the harness's
   `multi_swe_bench.harness.run_evaluation --config ...` (release 1.1.2) through
   `gbench/runners/eval_suites/mswe_harness_build.py` (below), then reads
   `resolved_ids` from `final_report.json` (matched by `instance_id`, with a repo+number
   fallback), and checks which submissions got a verdict at all (`completed_ids`, in the harness's
   `org/repo:pr-N` spelling).

   **Harness image builds of archived Debian releases.** Before it scores, the harness builds each
   submitted instance's image chain from the Dockerfiles it generates: the repository's base image
   (e.g. `mswebench/simdjson_m_simdjson:base-cpp-7`), then the instance's `pr-<number>` image FROM
   it. It skips any image already present, but it walks the chain from the top, so it builds a
   missing base even when the published `pr-<number>` image MSWE-agent ran on is already on the
   host. When the base build fails, the harness skips that instance and stops (`stop_on_error`), so
   no `final_report.json` is written. The 48 instances on the `gcc:7` base (Debian 10, buster:
   nlohmann/json, simdjson, Catch2) are hit by this. Their base Dockerfile starts with
   `RUN apt-get update && apt-get install -y build-essential pkg-config wget tar && wget
   .../cmake-3.14.0-Linux-x86_64.tar.gz && ...`, and `deb.debian.org`/`security.debian.org` now
   answer 404 for buster. So the step exits 100, as in the 2026-10-09 smoke run (`Error building
   image mswebench/simdjson_m_simdjson:base-cpp-7 ... returned a non-zero code: 100`, then
   `final_report.json not found`). `mswe_harness_build.py` runs the harness unchanged except for
   the image build call (`multi_swe_bench.utils.docker_util.build`):
   it builds upstream's Dockerfile first, untouched. Only when that build fails *and* its log shows
   apt reporting a `debian.org` suite gone from its mirror does it build the image once more, from
   a copy of the same Dockerfile (`Dockerfile.gbench-apt-archive`, next to it in the harness's
   build directory). That copy adds one step after each `FROM` line: the driver's own repair
   (`APT_ARCHIVE_REPAIR` from `gbench/docker/mswe_agent_run.py`, loaded from that file so the two
   cannot drift). On an archived Debian release (a no-op anywhere else) the repair points apt at
   `archive.debian.org` with `Acquire::Check-Valid-Until` false. The copy also adds the label
   `org.gbench.apt-archive-fallback`. Every other step, package and download is upstream's. An
   image whose build works is never touched, and no image is pulled from a registry for it (the
   `FROM` image is used from the host when present, as `docker build` does). Each retry is
   recorded in the result's `harness_apt_archive_fallback` (`images`, `built`, `failed`, and per
   image `records`: `image`, `apt_error`, `upstream_error`, `applied`, `release`, `retry`). An
   image whose retry also fails stays a harness failure (no verdict, `harness_no_verdict` /
   `harness_no_report`), never the model's outcome. Checked 2026-10-10: the upstream base-cpp-7
   Dockerfile fails exactly as in the smoke run. The retried build succeeds (`apt archive
   fallback: applied (buster)`, cmake 3.14.0 downloaded and installed). Its installed Debian
   packages (420, by `dpkg-query` name and version) and `cmake --version` are **identical** to
   those of upstream's published `mswebench/simdjson_m_simdjson:pr-949`, which upstream built from
   that base when the mirrors still served buster. buster's final point release and its final
   security updates are what the archive serves. With the checkperf reference pinned (below), the
   harness then scored the gold patch of simdjson-949 as resolved (checked 2026-10-10 through
   gbench's own scorer: 46 passing tests, `harness_checkperf_reference.pinned` 1). The other bases the dataset's 1,632 instances build FROM (Ubuntu
   16.04/18.04/20.04/22.04/`latest`, `gcc:11`/`gcc:12`/`gcc:latest`, `golang:latest`,
   `rust:latest`, `node:18`/`node:20`) are releases their mirrors still serve. In the same smoke
   run the C, Go, Java and TS instances' base images (`ponylang_m_ponyc:base-16V2`,
   `cli_m_cli:base`, `fasterxml_m_jackson-core:base`, `vuejs_m_core:base`) built or were present.
   The fallback also covers bullseye once the Debian mirrors drop it.

**The docker socket.** The MSWE-agent container mounts `/var/run/docker.sock` and runs on the
host network, as upstream's design needs: MSWE-agent starts each instance container itself with
`docker run` on the host's daemon and sends the model's commands to that container's shell
(docker-out-of-docker), and it reaches gbench's relay on `127.0.0.1`. The model's commands run
in the instance container, which gets no socket. The residual risk: the socket gives the
MSWE-agent process, which turns the model's replies into those commands, root-equivalent access
to the host's docker daemon (and so to whatever the host can reach), and the host network lets
it reach every service listening on the host; a flaw in MSWE-agent or the driver that let model
output run inside the MSWE-agent container would reach both. That container holds no
credential (`OPENAI_API_KEY=EMPTY`, no Docker client configuration). gbench keeps upstream's
design because MSWE-agent cannot run without a daemon it can reach.

Exit statuses: an instance whose image could not be pulled, one MSWE-agent did not finish
because its setup failed (when the reset raised, the driver records the step, `apt`,
`repo_reset`, `container_start` or `other`, with the error text, `setup_error`, and the harness
error names it), or because its rollout raised before the model's first turn or with a
model-endpoint or docker daemon failure (the driver's `rollout_error`, below), or that recorded
neither (the harness error says so; it is never called a setup failure without a recorded setup
error), or whose exit status
names `exit_api` (a tenacity `RetryError`), is an infrastructure failure: a harness error
(`harness_error: "agent_infra"` on its trace), left out of `accuracy`'s numerator and denominator
and listed under `partial_evaluation`. An endpoint that keeps failing is one of those:
`OpenAIModel.query` retries with tenacity `reraise=True` and `forward_with_error_check` handles
only RuntimeError, ContextWindowExceededError, CostLimitExceededError and RetryError, so the
openai exception of a 5xx, 401, 404 or connection failure leaves the rollout and the instance
never finishes. An `exit_error` other than the step limit (MSWE-agent's "exit due to runtime
error: <exception>" for a RuntimeError from the model call) is an infrastructure failure only
when the exception text starts with a model-endpoint failure or the docker daemon's unreachable
message (`swebench_common.scaffold_error_harness_reason`).

Everything else is the model's outcome: `submitted`, `submitted (exit_cost)` when the context
window is exceeded (its `OpenAIModel.query` turns every 400 into that), `exit_format`, the step
limit's `submitted (exit_error)`, any other `exit_error`, and `early_exit`. `early_exit` is what
MSWE-agent's `SWEEnv.step` records when the container stopped answering while it ran the model's
command: the command raised (e.g. the model killed the container's shell), broke the pipe, or
timed out and could not be interrupted. It is the model's outcome of that run (unresolved, an
empty submission), as a single pass of upstream's run scores it; upstream's resume
(`multirun.py` `should_skip` with `--skip_existing`) would roll such an instance out again, and
gbench rolls each instance out once. The kept `exit_error` and `early_exit` endings carry
`agent_exit_note` on their traces and are counted in `exits_kept_as_model_outcome` (ids in
`exits_kept_as_model_outcome_ids`).

Rollouts that raised. An exception MSWE-agent's rollout (`Agent.run`) raises ends the instance:
`multirun.main` logs "Failed on <id>" and moves on, and the instance has no prediction, which
upstream's run leaves unresolved (no patch). The driver records the exception in the instance's
`rollout_error` (`type`, `error`, `message`, `step`: the model turns started, `last_reply`: what
the last `OpenAIModel.query` returned, `none`, `empty`, `text`, `pending` when the query raised,
`unknown` before any). `OpenAIModel.query` returns `message.content` unchecked, so a reply that
carries reasoning only (content `null`) reaches MSWE-agent's parser as None and its
`re.finditer` raises `TypeError: expected string or bytes-like object`, ending the rollout. That
is the model's outcome: status `crashed_empty_reply`, no patch, unresolved and scored, counted in
`empty_reply_crashes` (ids in `empty_reply_crash_ids`). Any other exception raised after the
model's first turn follows the `exit_error` rule: the model's outcome (status `crashed`) unless
its text starts with a model-endpoint failure or the docker daemon's unreachable message (then
infrastructure); one raised before the model's first turn is infrastructure. Both kinds kept as
the model's outcome carry `agent_exit_note` and are counted in `exits_kept_as_model_outcome`.

What the relay counts (`swebench_common.EndpointRelay`, recorded in `relay`, applied by
`swebench_common.relay_result_reasons`): endpoint failures (a request the endpoint did not
answer, a reply that is not 2xx except a 400 whose message names the context window, a 2xx
completions reply that is not JSON, an error event in a stream, a request the relay refused)
are harness errors, listed in `endpoint_failures`; the run's `status` becomes
`completed_with_errors` and it is not comparable. This also catches a 400 that is not about the
context window, which MSWE-agent itself records as `exit_cost`. Model turns that stopped at a
limit (`finish_reason: length`: the run's `--max-output-tokens`, which gbench's relay sends as
max_tokens with every agent request while MSWE-agent sends none; the end of the served context
window with `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0`) are counted in `truncated_turns`, which above zero makes the run not comparable. Turns
that stopped on their own with reasoning but no content are counted in
`empty_after_reasoning_turns` and split by their token ids (the relay asks the endpoint for
`return_token_ids` and looks the tool-call token up in the served tokenizer): no tool-call token,
the model wrote its answer inside its thinking channel (`answer_in_thinking`, with
`answer_in_thinking_rate` over `model_replies`, every reply choice): the model's outcome (the
rollout MSWE-agent's parser ends on such a reply is `crashed_empty_reply`, unresolved), not a
comparability reason; the tool-call token (Gemma 4 `<|tool_call>`, id 48), a native tool call the
server's tool-call parser removed because MSWE-agent declares no tools (`tool_call_dropped_turns`,
a server-parser artifact); no token ids (`unclassified_empty_turns`). The last two each make the
run not comparable; see [answer_in_thinking.md](answer_in_thinking.md). These are per-run counts: on each trace `truncated_turns` is
`null` (a relayed request cannot be tied to an instance). A request whose client closed its
connection before the relay had sent the whole reply (a client that gave up on a turn and
resent it) is counted in `relay.client_closed_streams`, not as an endpoint failure; the relay
notices the client leaving while it still waits for the endpoint (it watches the client's
connection for its end of file) and closes the endpoint's connection of that request at once, so
the served endpoint stops generating the abandoned reply instead of generating it to its end next
to the retry.

Run knobs:

| Knob | Effect |
| --- | --- |
| `--temperature` / `GBENCH_MULTI_SWE_BENCH_TEMPERATURE` | MSWE-agent's `--temperature` (default: gbench's rule, 0.0 without `--thinking`, 1.0 with it); `GBENCH_MULTI_SWE_BENCH_TEMPERATURE` wins over `--temperature`; another value than 0.0 is recorded and makes the run not comparable |
| `--eval-thinking` | `chat_template_kwargs.enable_thinking` on every request (set by the relay) |
| `--max-output-tokens` | sent as every agent turn's `max_tokens` (set by gbench's relay as a body field), on by default since 2026-10-11, a recorded deviation (MSWE-agent sends none). `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0` restores upstream's none (the server grants each turn what the context window has left). The one-shot mode sends it as its single request's `max_tokens` |
| `--sandboxes` (or the eval concurrency) | MSWE-agent's `RUNNING_THREADS`, the host image pulls in parallel, and the harness `max_workers` |
| `--eval-limit` / `--shard` | `--eval-limit N` is N instances for the whole suite, one stratified draw over the flattened (language, instance) list (never N per language; the planned workload is logged on one line at the start and recorded as `eval_limit_plan`) / round-robin shard |
| `GBENCH_MULTI_SWE_REPO_DIR` | where the harness clones repositories (default `~/.cache/gbench/multi_swe_bench/repos`, reused across runs) |
| `GBENCH_SWE_OMP_THREADS` | per-container thread cap for the harness containers |
| `GBENCH_MULTI_SWE_BENCH_TRAJ_DIR` | where the trajectories go (default: `<run results dir>/agent_trajectories/`) |
| `GBENCH_MULTI_SWE_BENCH_AGENT_IMAGE` | the MSWE-agent image name, if you retag it (default `gbench-mswe-agent:88217624`) |

**One-shot (opt-in).** `GBENCH_MULTI_SWE_BENCH_MODE=oneshot`: one request per instance (repository
name, base commit, PR title and body; no repository access), a diff extracted from the reply, same
harness. Always `leaderboard_comparable: false`.

## Reading the result
- `accuracy` = resolved rate; `category_accuracy` is per language; `multi_swe_bench_report`
  carries the harness's total / submitted / completed / incomplete / resolved / unresolved /
  error / empty-patch counts, its `error_ids` and `incomplete_ids`, and gbench's check of the
  submissions: `gbench_submitted_instances`, `empty_submission_instances`,
  `harness_unscored_ids` (a non-empty submission with no harness verdict) and
  `harness_missing_ids` (the part of those the report does not mention at all: the harness only
  reports instances whose run directory exists, so an image that never built leaves no entry),
  `harness_dns_failures` (`{instance_id: {hosts, evidence}}`, the DNS harness errors below) and
  `dns_failures_kept` (`{instance_id: {hosts, note}}`, uncorroborated ones, left the model's
  outcome), and `harness_checkperf_reference_failures` (`{instance_id: why}`, the simdjson
  instances whose pinned checkperf reference could not be set up, below). `accuracy` is over the scored instances (`scored_instances`); a run with harness
  errors also carries `harness_errors` by kind (`agent_infra`: the agent could not run;
  `harness_no_verdict`: a submission with a patch the harness gave no verdict;
  `harness_no_report`: no report at all; `dns`: the DNS failures below; `checkperf_reference`:
  the checkperf pin failures below; in one-shot mode also
  base's endpoint failures), `harness_error_ids`, `scoring_excluded` (their count) and
  `failed_requests`.
- `mode`, `agent` (MSWE-agent: commit, config and sha256, entry, step limit, top_p, the added
  Java definitions, gbench's driver and its label, model, image, `openai_client_timeout_s` and
  `openai_client`: the timeout, read timeout and retries MSWE-agent's OpenAI client carried, per
  language), `empty_reply_crashes` / `empty_reply_crash_ids` (when any), `apt_archive_fallback` (when
  the driver's archive.debian.org fallback acted: `instances`, `applied`, `releases`,
  `apt_update_failed_after`, `ids`), `relay` (where it listened, where it forwarded, the allowed
  paths, the body fields it set, requests relayed and refused, and its reply and failure
  counts), `temperature` / `official_sampling`, `max_output_tokens_note` (the
  per-turn cap sent with every turn and where it was set), `max_output_tokens_per_turn` (the run's
  `--max-output-tokens`; `null` only with `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0`),
  `agent_exit_statuses`, `agent_step_limit_reached` (the instances that hit the 50-step limit),
  `image_pulls`, `truncated_turns`, `empty_after_reasoning_turns`, `model_replies`,
  `endpoint_failures` (when any occurred), `harness_apt_archive_fallback` (when the harness's image
  builds needed the archive.debian.org retry described under **Scoring**: `images`, `built`,
  `failed`, `records`), `harness_checkperf_reference` (when the run included simdjson-949, -954
  or -958: the pinned checkperf reference described under **Scoring**) and, on every trace, `agent_exit_status`,
  `agent_steps` (model calls), `agent_patch`, `agent_infra_error`, `agent_step_limit_reached`,
  `agent_apt_archive` (when the fallback acted) and `trajectory_file` (MSWE-agent's own
  `.traj`). `agent_output_dir` is a fresh directory under
  `<run results dir>/agent_trajectories/` (next to the result JSON), or in the system temp dir
  outside a gbench run, with one subdirectory per language: MSWE-agent's `trajectories/`,
  `data/`, `logs/`, the driver's `spec.json`, `rows.jsonl`, `summary.json` and `mswe_agent.log`,
  and gbench's `image_pulls.json`. No key is in them (MSWE-agent only ever sees `EMPTY`; gbench
  also replaces any credential value of the environment found in the log, the summary or a
  trajectory file with `[REDACTED]`).
- A harness that produced no report is `status: "error"` with its stderr tail and `accuracy`
  `null` (`accuracy_withheld_reason`), never a 0%. A submission with a patch that got no harness
  verdict (it errored or is missing from the report: image build, clone or run failure) is not
  model-scoreable: a harness error (`harness_no_verdict`), left out of `accuracy`'s numerator and
  denominator. An empty submission (the agent hit its step limit or the context window; in
  one-shot mode, no extractable diff) is the model's outcome: it counts as unresolved and as
  model-scoreable. When no submission is model-scoreable (every submission carried a patch and
  none got a verdict) the run is `status: "error"` ("harness infrastructure failure") with its
  score withheld; otherwise, when any submission with a patch got no verdict, the run is
  `completed_with_errors` and `partial_evaluation` counts them (a run where every patch went
  unscored but some submissions were empty is scored over those empty submissions). Both modes
  apply these checks. An instance whose container could not start (image missing, checkout
  missing) or whose endpoint failed is a harness error too (`agent_infra`): left out of the score
  and listed in `partial_evaluation`, or `status: "error"` with the score withheld if no instance
  could run.
- **DNS failures in the harness's test runs.** Each instance image's `fix-run.sh` runs the
  repository's own build (`mvn clean test`, `./gradlew clean test`, `pnpm test`, `go test`,
  `cargo test`, ...), which fetches what the image does not cache at test time: offline,
  `mvn clean test` in `mswebench/fasterxml_m_jackson-core:pr-1208` fails with "Non-resolvable
  parent POM ... Unknown host oss.sonatype.org", and the harness then reports the instance
  unresolved. An instance the harness reports unresolved whose fix-patch run captured no test
  result (`report.json`'s `fix_patch_result` counts all 0, in the harness's
  `<workdir>/<org>/<repo>/evals/pr-<number>/`) and whose `fix-patch-run.log` shows a host under
  `maven.apache.org`, `maven.org`, `sonatype.org`, `gradle.org`, `maven.google.com`,
  `dl.google.com`, `npmjs.org`, `npmjs.com`, `yarnpkg.com`, `golang.org`, `crates.io`,
  `github.com`, `githubusercontent.com` or the other installer domains of
  `swebench_common.INSTALL_DOMAINS` failing to resolve (Maven's `Unknown host`, a
  `java.net.UnknownHostException`, Node's `getaddrinfo EAI_AGAIN` / `ENOTFOUND`, Go's `dial tcp:
  lookup`, curl's and Cargo's `Could not resolve host`, a resolver error with the host's URL
  nearby) scored nothing. It is a harness error when something outside that container
  corroborates the failure: another such instance in the run, or a fresh probe container after
  the run (container_guard's probe, not its cache, in the same images as the gate) that cannot
  resolve one of the hosts either. Then it is left out of `accuracy` (numerator and denominator;
  `scoring_excluded` on its trace, with `harness_error: "dns"`, `dns_failure_hosts` and
  `dns_failure_evidence`), counted in `harness_errors`, the run is `completed_with_errors` and not
  comparable, and a run whose every submitted instance is a harness error (DNS failures plus
  submissions without a verdict) is `status: "error"` with its score withheld. Uncorroborated, it
  stays unresolved (the model's outcome), with `dns_failure_note` on its trace. Once the tests
  produced results, their verdict stands. The other harness errors above (no verdict, no report,
  a container that could not start) are left out of `accuracy` the same way, as across the SWE
  family.
- **simdjson's checkperf reference (pinned to the commit upstream's images recorded).** At the
  base commits of simdjson-949, -954 and -958, `benchmark/checkperf.cmake` puts its
  `checkperf-parse` target in `all`, so the `cmake --build .` of their `fix-run.sh` runs
  `git fetch --depth=1 origin master` of `https://github.com/simdjson/simdjson` into
  `build/benchmark/checkperf-reference/master` and builds that checkout with the image's gcc 7 and
  CMake 3.14. Of the dataset's 20 simdjson instances only these three do in a default build: 11
  others also have a checkperf fetch (in `benchmark/dom/checkperf.cmake`), but it is off by default
  and their `fix-run.sh` does not turn it on, so it never runs; the remaining 6 have none (checked
  by reading the checkperf files at every base commit). `checkperf` appears in no other
  repository's dataset file. Today's master no longer configures there
  (`Target "ondemand_odr_tests" requires the language dialect "CXX20"`), so unpinned the build
  stops before `ctest` and no test result is captured, whatever the patch (checked 2026-10-10 with
  upstream's own gold patch of simdjson-949).
  Upstream's published instance images `mswebench/simdjson_m_simdjson:pr-949/954/958` were built
  on 2025-03-18 and pushed together on 2025-03-27; the clone inside pr-949 has `master` (and
  `origin/master`) at `726c3eb611a15f7ad8690bfe800e5f5dbf95376d` (2025-03-17), the master those
  test runs fetched when upstream recorded the instances' results. gbench pins the fetch to that
  commit (`CHECKPERF_REFERENCE` in `gbench/runners/eval_suites/mswe_harness_build.py`): for those
  three instances only, the harness's test container first runs a short setup that makes a bare
  clone of the image's own `/home/simdjson` (objects shared, so instant, and no network), sets its
  `master` to the pinned commit (fetched from GitHub by its hash only when the image's clone lacks
  it, e.g. an image the harness built locally from an older clone) and sets git's
  `url.<clone>.insteadOf https://github.com/simdjson/simdjson`. The fetch, its depth, the reset and
  every build and test command are upstream's unchanged; only the commit that `master` names is
  fixed. The pin applies to the harness's scoring run only, not to the agent's rollout container
  (a build the model starts there fetches today's master); scoring, which decides the verdict, is
  unaffected. The setup runs before the patch is applied and prints one line,
  `gbench-checkperf-reference: pinned <commit>` or `... failed <why>`, at the top of
  `fix-patch-run.log`. Checked 2026-10-10 on the local published pr-949 image (no pull): with the
  pin, upstream's gold `fix_patch` builds the reference (`HEAD is now at 726c3eb`) and the
  harness's own `Simdjson.parse_log` finds exactly the dataset's `fix_patch_result`: the same 46
  passing tests, 0 failed. (The `checkperf` ctest itself fails in that run; upstream's
  failed-test pattern does not match ctest's `***Failed  0.18 sec` line, so neither gbench's nor
  upstream's parsing counts it, and the recording cannot tell whether it also failed upstream.) Each pinned run is
  recorded in the result's `harness_checkperf_reference` (`commit`, `pinned`, `failed`, and per
  run `records`: `image`, `number`, `commit`, `pinned`, `detail`). When the pin could not be set up
  (or no record exists for that run), the instance is a harness error, whatever the patch: left out
  of `accuracy` (`scoring_excluded`, `harness_error: "checkperf_reference"`), listed in
  `multi_swe_bench_report.harness_checkperf_reference_failures` and counted in `harness_errors`;
  the run is `completed_with_errors` and not comparable, and its score is withheld when every
  submitted instance is a harness error.
- `leaderboard_comparable` is `true` for a full run with MSWE-agent at its own sampling (so a
  no-think run, whose rule temperature 0.0 equals MSWE-agent's), every
  instance rolled out and scored, no decoding penalties, no endpoint failure and no model turn
  cut off at a limit or empty after reasoning. `--eval-limit`, `--shard`, a temperature other
  than 0.0 (so every `--thinking` run, sampled at 1.0), decoding penalties, partial scoring, endpoint failures and cut-off or empty turns are
  added to the reason when they apply.

## Prerequisites
- **Docker** with a reachable daemon and plenty of disk: gbench pulls each instance image from
  Docker Hub (`mswebench/*`, ~1,632 images) with the host's Docker CLI from its own process, so
  `docker login` on this host (recommended for Docker Hub's pull-rate limit) applies to them, and
  the harness (run on the host) may build missing ones. Optional prefetch with the harness's list:
  `bash scripts/download_images.sh scripts/images_verified.txt` from a
  `multi-swe-bench/multi-swe-bench` checkout.
- **Python packages** (`multi-swe-bench` is in gbench's `evals` extra, `huggingface_hub` in the
  base install):
  ```bash
  pip install 'multi-swe-bench==1.1.2' huggingface_hub
  ```
  The harness is pinned: multi-swe-bench 1.1.2 is the release built from harness commit
  `24f493f8a103e72312ded4f6b9c89f081d69cb09` (its `harness/` and `utils/` modules are
  byte-identical to that commit), the revision the result's `canonical_sync` records. The suite
  checks the installed version at startup: any version other than 1.1.2 stops it with an
  `[INFRA REQUIRED]` error naming the install command.
- The **MSWE-agent image** (agentic mode), built once:

  ```bash
  docker build -t gbench-mswe-agent:88217624 -f gbench/docker/mswe_agent.Dockerfile gbench/docker
  ```

  It fetches MSWE-agent at `88217624b637646b886cb0462995c07559e96f58` by commit onto
  `python:3.9.21-slim-bookworm` pinned by digest (Python 3.9, as its `environment.yml` pins; its
  Debian packages from the archive snapshot `20260925T120000Z`, `docker-ce-cli`
  `5:29.8.0-1~debian.12~bookworm`), installs `requirements.txt` plus `dataclasses-json` (its
  vendored harness imports it; `requirements.txt` does not list it) in a venv, every package
  pinned by `gbench/docker/mswe_agent.constraints.txt` (the set PyPI served on 2025-04-03, the day
  after that commit: `openai` 1.70.0, `swebench` 3.0.15, ...; resolved set recorded in
  `/opt/mswe-agent-pip-freeze.txt`), adds the six Java repository definitions described above
  (the build checks they register) and copies gbench's driver (`gbench/docker/mswe_agent_run.py`,
  label `org.gbench.mswe_agent_driver`, version `4`: with the archive.debian.org fallback above,
  recording which setup step of an instance's reset failed and the exception an instance's
  rollout raised, and giving MSWE-agent's OpenAI client the spec's `client_timeout_s`, 3,660 s
  when a spec names none).
  The build checks the commit and the config sha256, and that a client built the way MSWE-agent
  builds it carries that timeout. An image built before driver 4 is refused with the rebuild
  command above. The prerequisite check refuses an image
  whose labels `org.gbench.mswe_agent`, `org.gbench.mswe_agent_config_sha256`,
  `org.gbench.mswe_agent_java_repos` and `org.gbench.mswe_agent_driver` differ (an image built
  with an older driver has to be rebuilt), and needs `/var/run/docker.sock`. MSWE-agent's README
  states the Apache License 2.0; gbench fetches it into the image and vendors none of it.
- A served model at an OpenAI-compatible `/v1` endpoint (`VLLM_API_KEY`, else `OPENAI_API_KEY`,
  if it needs a key; gbench's relay adds it outside the agent containers);
  MSWE-agent parses DISCUSSION + fenced-command replies, so no tool-call support is needed.
- **On a cloud VM, the metadata server blocked for docker containers.** MSWE-agent runs the
  model's commands in each instance container, and the harness runs the tests over each patch in
  its own containers, both with network access; a docker container reaches the VM's metadata
  server at `169.254.169.254`, which hands out the VM's service-account token. Before MSWE-agent
  starts, and before the harness pass, gbench checks that it is unreachable from docker's
  default bridge and from a user-defined bridge network (gbench creates a temporary one,
  `gbench-metadata-probe-<id>`, and removes it right after): a short-lived probe container
  (`--pull never`) opens and closes a TCP connection to `169.254.169.254:80` and sends nothing.
  The same probe container resolves the package hosts the instance containers' builds fetch
  from at test time, for the languages of the run (`runtime_hosts`): Java
  `repo.maven.apache.org`, `oss.sonatype.org`, `plugins.gradle.org`, `services.gradle.org`;
  JavaScript and TypeScript `registry.npmjs.org`; Go `proxy.golang.org`; Rust `index.crates.io`,
  `static.crates.io`; every language `github.com`. A name that does not resolve **hard-errors**
  too (on GCE the VPC DNS resolver is `169.254.169.254` itself, so a rule that drops all traffic
  to it also drops DNS). The probe runs in `gbench-mswe-agent:88217624`; before the harness pass without
  that image (one-shot mode), in a submitted instance's published image or any local
  `mswebench/*` image already on the host. Nothing is pulled for the probe: with none present
  the harness pass hard-errors (`infra_required`), naming the `docker pull` of the first
  submitted instance's published image and the MSWE-agent build. Each network's verdict, and
  each name that resolved there, is cached for the rest of the run. A connection that opens, or
  a probe without a clear verdict, **hard-errors** (`infra_required`) naming the fix. Block it
  once on the host; gbench never runs sudo or changes the host's firewall itself:
  ```bash
  sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
  sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save   # keep it
  ```
  See [toolchains.md](toolchains.md#cloud-vms-the-metadata-server).

Missing Docker, the harness package or the MSWE-agent image, or a failed metadata server check,
**hard-errors** (`infra_required`); the suite never skips and never reports a fabricated number.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals multi_swe_bench \
       --max-output-tokens 32768 --sandboxes 4
```
Add `--eval-limit 14` for a stratified smoke run across all seven languages (not comparable).
