# swe_bench_live (SWE-bench-Live)

> **Docker setup:** this suite runs one container/network per task and can exhaust the
> Docker address pool at high `--sandboxes`. See [docker-sandboxes.md](docker-sandboxes.md)
> for the recommended `default-address-pools` config and pre-run prune, strongly advised
> before running at concurrency.

## What it measures
**SWE-bench-Live**: real GitHub issues, each with its own instance image, scored by
execution-based **resolved rate** (apply the patch, run `FAIL_TO_PASS` / `PASS_TO_PASS`).

**Leaderboard settings.** The SWE-bench-Live leaderboard (swe-bench-live.github.io, source
`SWE-bench-Live/swe-bench-live.github.io@90e8720259d7fd8a311582522e221784aaa30407`) has ten
tabs (`index.html` tab buttons; `leaderboard.js` `validSets`): **Lite**, **Full** and
**Verified** for the Python dataset, **C/C++**, **C#**, **Java**, **TS/JS**, **Go** and
**Rust** for the multi-language dataset, and **Windows**. Each tab ranks resolved / total of its
own rows (`leaderboard.js`: `resolved_percentage = resolved / report.total`). A gbench run
evaluates all ten by default. Each setting gets its own rollouts, its own harness pass and its
own headline. The settings run at the same time within `--sandboxes` (see "Tabs run at the same
time" below), except the three Python tabs, which run one after the other.

**Tabs run at the same time.** The tabs a run selects are grouped in lanes: one lane holds every
selected Python tab (Lite, Full, Verified), and each multi-language tab and the Windows tab has
a lane of its own. Lanes run at the same time, the tabs of a lane one after the other. The Python
tabs share a lane because their instance sets overlap (all of Lite and 457 of Verified's 500 ids
are in Full; 174 ids are in both Lite and Verified), and the fork harness names each task
container `sweb.eval.<instance id>.<run id>` with one run id for the suite, so two Python tabs
scoring one instance at once would collide. The multi-language splits share no instance id with
any other tab, and upstream's multi-language harness names every task container with a uuid.
The concurrency (`--sandboxes`, else the eval concurrency) is a budget for the whole suite, dealt
over the lanes: each lane gets at least 1, at most as many as its largest tab has instances, and
the rest goes one at a time to the lane with the most instances per worker. Every tab of a lane
runs with that lane's share as its concurrency (SWE-agent's `--num_workers`, Win-Agent's
`workers`, the harness workers), so the tabs running at one time never add up to more than the
budget. With a budget smaller than the number of lanes, each lane gets 1 and that many lanes run
at a time. Each tab runs exactly as it would alone: its own SWE-agent batch, relay (and its
endpoint-failure and cut-off accounting), watchdog, harness pass, result and headline. The
suite result records the schedule under `tab_schedule` (`concurrency`, `concurrent_lanes`,
`lanes`, `workers` per tab). A tab that fails keeps the other tabs' results (its own result is an
error); a missing prerequisite during the run or a SIGTERM/SIGHUP starts no further tab, waits
for the running tabs to end (each removes its containers) and then ends the suite, as before.
A SIGTERM/SIGHUP reaches every running tab, also when the suite runs on a worker thread
(`--suite-timeout`): the thread running the suite hands it, within a second, to each tab's
thread, so every running tab's watchdog stops its SWE-agent batch and its cut-short rollouts are
not scored. A tab in its harness pass (the upstream harness does not look for a signal) finishes
that pass, but the suite never returns a result after a stop: once the running tabs have ended,
the schedule logs and raises an exit (`TabScheduleStopped`, exit code 128 + the signal) whose
record names the signal and every tab the stop halted (`tabs_stopped` = `tabs_cut_short`, the
tabs running when it arrived, plus `tabs_not_started`; `tabs_completed` ended before it), for
example `terminated by signal 15 (SIGTERM) while the tabs ran ...; Tabs stopped: Lite, Verified,
C/C++ (cut short: Lite, C/C++; not started: Verified); completed before the stop: none`. No tab
goes missing from a result that looks complete, and the run stops, as when the tabs ran one after
the other. The per-container thread cap (`docker_thread_cap`, see below) is computed from the
whole budget in every tab, not from the tab's share, so the task containers of all lanes share
the CPUs exactly as when the tabs ran one after the other with the whole budget each.
With `--eval-limit 5` over Lite, Verified, C/C++, Go and Rust, for example, four lanes run at
once (Lite then Verified; C/C++; Go; Rust), one worker each, instead of five tabs in a row.

The Python tabs (scored by the SWE-bench-Live fork of the swebench harness):

| Setting | Split @ Hub commit | Rows | Content sha256 |
| --- | --- | --- | --- |
| `lite` | `lite` @ `b51a86422e10cfd403beb4773e5a2947953e36ec` | 300 | `b9252aeb085eb949f01b8464bee76240d474146c9a4f6aed1bbb693c2b94755e` |
| `full` | `full` @ `3da151be953883d3cd4e1ea0dc7a042ff2abf8f4` | 1319 | `b231d1932ce614df07b6b4237db2a6f4eb6020213d0d9e919d95a62b976cf95e` |
| `verified` | `verified` @ `b51a86422e10cfd403beb4773e5a2947953e36ec` | 500 | `d3ae023598f62e0c24a025a5091ec6ef97224b62dd8aeb6224d7b14154e55974` |

- `lite` and `verified` come from Hub main as of 2026-09-24 (`b51a8642`). Upstream keeps both
  splits frozen for leaderboard comparisons.
- `full` comes from `3da151be` (2025-05-21), the last commit whose `full` split holds the 1,319
  issues of the May 2025 release (created 2024-01 to 2025-04). Every Full row on the leaderboard
  has `total: 1319` (`reports-0605.jsonl`). The split has grown since: the next data upload
  (`12b98891`, 2025-07-02) made it 1,565 rows, and at `b51a8642` it holds 1,888 rows. No
  leaderboard row scores the larger set. All 1,319 rows are unchanged in it, and their
  `starryzhang` instance images are the ones built for that release.

The multi-language tabs, from `SWE-bench-Live/MultiLang` at
`3638632e8153a10ca422c1022bed79023084b5c9` (Hub main as of 2026-09-20; its data files are those
of the 2026-08-20 upload, 1,077 instances in eight language splits), and the Windows tab, from
`SWE-bench-Live/Windows` at `b9fa25f2732dbc2c93b9738b610d053d8b7bd7a9` (Hub main as of
2026-09-20, data of 2026-09-10). All are scored by upstream's multi-language harness:

| Setting | Tab | Splits | Rows | Content sha256 (per split) |
| --- | --- | --- | --- | --- |
| `ccpp` | C/C++ | `c` (111) + `cpp` (142) | 253 | `c`: `713f362bef00486539bf457dd08b74713066d764a07fbbea9c12d1936461a0a8`; `cpp`: `6f0c50cca9be037126623f7f8af9d10ffc55f405dc51fdfe14d35b635e57eae5` |
| `csharp` | C# | `cs` | 136 | `4086e1640f6d69ce7a2ef9c7b14d39b510c77cfddc4cae86efef5934177a3236` |
| `java` | Java | `java` | 160 | `aa6dce064c88cf51109da87a6e993b8d05b786ef2e1a9da3791e1e8054289480` |
| `tsjs` | TS/JS | `js` (108) + `ts` (111) | 219 | `js`: `9cf48e9b78e56670ea210778bad59341fe2fda00e3d2d4e7937f7e35e4c4ef8e`; `ts`: `a3944c6d25f2d7112910e6d9a76cd5fc7f3dcadb2c5387a33d4db4c20826124f` |
| `go` | Go | `go` | 138 | `261851b6adc60e90747bd07e0b48ff5ad6f3ba7a5f8808c4e9b780a9c9fe8528` |
| `rust` | Rust | `rust` | 171 | `b60c2965b1def9aa90d1e8b3a22d09d8544336108202a972798e2aa804150917` |
| `windows` | Windows | `test` (Windows dataset) | 66 | `1f2a2e4057b1408b8932b1474cf8c9f0ce94b1862c25c4fdb53525b23fc432c5` |

- The tab-to-split mapping is checked against the board: its 2026-01-08 rows have C/C++ total 48
  and TS/JS total 162, the `c` + `cpp` (31 + 17) and `js` + `ts` (75 + 87) row counts of the
  MultiLang commit current then (`3430730b`).
- The multi-language and Windows datasets grow, and each leaderboard row keeps the total of the
  snapshot it ran on (`reports-0605.jsonl`: C/C++ 48, 66; C# 20, 28, 78; Java 39, 62, 102, 109,
  160; TS/JS 162, 204; Go 68, 98, 138; Rust 45, 64, 94; Windows 50, 60). gbench pins the current
  upstream snapshot. A number compares directly only with board rows of the same total: today
  the Java (160) and Go (138) tabs have such rows. The result records the board's totals of each
  tab (`settings.<name>.leaderboard_totals`) and, when none equals the pinned row count, says so
  in `leaderboard_totals_note` and in the setting's `leaderboard_comparable_reason`.
- Upstream's evaluation README lets a submission divide by the instances that pass with the gold
  patch on its machine. gbench divides by every row of the pinned snapshot, the dataset as
  published.

`--eval-categories` picks a subset, for example `--eval-categories verified` or
`--eval-categories lite,java,go` (gbench routes these names to this suite through
`gbench/runners/evals.py` `eval_category_names`, which reads the suite's `SETTINGS`). Called
from Python, `run_swe_bench_live(..., split="verified")` runs one setting.

**Per-setting prerequisites.** Every selected setting whose prerequisites are present runs, and
every selected setting whose prerequisite is missing on this host is reported in its own entry,
`settings.<name>`, with `status: "infra_required"`, the missing prerequisite in
`infra_required` and an `[INFRA REQUIRED]` `error`: it is never scored and never dropped, its
`headline_metrics.<name>_resolved_rate` is `null`, the suite's `status` is
`completed_with_errors` and `settings_infra_required` lists it. The rule works in both
directions (`swe_bench_live.missing_prerequisites`):

- The Windows tab needs a Windows Docker host (below) and a passing cloud metadata server check
  of its own (`check_windows_metadata_guard`, see Prerequisites): its task containers run
  model-controlled code with network access, so the tab never starts on a verdict gbench could
  not establish. The default ten on Linux run the nine other tabs and report `settings.windows`
  as `infra_required` (the engine runs Linux containers).
- The Python and multi-language tabs need the Linux images (the harness image with the labels
  of their family, and in the agentic mode the SWE-agent-for-eval image and
  `/var/run/docker.sock`), which a Docker engine running Windows containers cannot hold. The
  default ten on such a Windows host report the nine as `infra_required`, each naming the image
  it lacks, and run the Windows tab when its prerequisites and its metadata server check pass.
- A setting whose pinned snapshot cannot be fetched or does not match is `infra_required` on
  its own; the other settings run.

The whole selection is checked first; only when that check fails is each setting checked alone,
to name the settings it stops. When every selected setting lacks its prerequisite
(`--eval-categories windows` on Linux, or no reachable Docker daemon), the suite hard-errors
(`infra_required`), naming each missing prerequisite once with the tabs it stops.

**Dataset snapshots.** On first use gbench writes each split a setting reads to
`$GBENCH_CACHE_DIR/swebench_snapshots/<dataset with / -> __>@<commit>/<split>.jsonl` (default
`GBENCH_CACHE_DIR` is `~/.cache/gbench`) and reuses it afterwards, also offline. The rows are
checked before they are written, and the file is checked again before every reuse. The checks
are: the row count in the tables above; every row carrying the columns its harness reads (the
Python tabs: the fork's `make_test_spec` columns `instance_id`, `repo`, `base_commit`,
`problem_statement`, `test_patch`, `test_cmds`, `log_parser`, `FAIL_TO_PASS`, `PASS_TO_PASS`;
the multi-language and Windows tabs: the columns upstream's `evaluation.py` reads and those the
rollout reads, `instance_id`, `repo`, `problem_statement`, `base_commit`, `test_patch`,
`test_cmds`, `print_cmds`, `rebuild_cmds`, `log_parser`, `FAIL_TO_PASS`, `PASS_TO_PASS`,
`docker_image`, `created_at`, plus `language` for Windows); and the content fingerprint of the rows. The fingerprint is the sha256 of each row as
canonical JSON (sorted keys, one row per line, in the published order;
`swebench_common.snapshot_sha256`), measured on the rows `datasets` loads at that commit.

A cached file that fails is moved aside to `<split>.jsonl.invalid-<unix time>` and the commit is
fetched again. Rows that still fail after a fetch are a hard error naming the problem. The
prerequisite check fetches and checks the snapshot of every selected setting that runs, so a missing or
wrong snapshot is an `infra_required` error before any rollout. For each multi-language split it
also runs upstream's harness imports inside the local image with the network off and compiles
every row's `log_parser` (the harness `exec`s it on the test log). The agent's tasks (agentic
mode) and the prompts (one-shot mode) come from those files. The Python tabs' fork harness gets
the same file as `--dataset_name` (mounted read-only into the image at the same path); the
multi-language harness gets the tab's rows as its `--dataset` file. Both sides read one commit.
Measured in the image with the network off, the fork builds a `TestSpec` for every row of the
three Python snapshots.

**A repeated instance id.** `verified` carries `conan-io__conan-18153` twice: two rows with the
same `problem_statement` that differ in `test_cmds` (`pytest -rA` / `python -m pytest -rA .`)
and `all_hints_text`. The other snapshots have no repeats. The split is used exactly as
published. The fork harness keys predictions by instance id (`run_evaluation.main`), so such an
instance gets one rollout, one prompt and one patch. The harness receives every row and runs it
as it does upstream. Its report counts rows in `total_instances` (500 for `verified`), while its
resolved, unresolved and error ids are unique (`reporting.make_run_report`). The headline
follows the leaderboard, whose Verified rows have `total: 500`: `total_questions` counts rows
(500), `accuracy` is resolved instance ids / rows (the rows of a harness-error instance are left
out, as the instance is; `scored_instances` then counts the scored rows), and
`total_instance_ids` records the id count (499). The result also records the repeat under
`dataset_repeated_instance_ids`.
`settings.verified.category_accuracy` (per repository) counts instance ids.

## Protocol (default: agentic)
SWE-bench-Live's rule for a rollout (`microsoft/SWE-bench-Live` `evaluation/README.md`): the
agent may use only the `problem_statement` and the instance's Docker image, in a single rollout,
with no hints, tests or test patch.

**Python and multi-language tabs: SWE-agent-for-eval.** The evaluation README lists "the forked
source codes of the agents we used to run SWE-bench-Live": SWE-agent
(`github.com/njukenanli/SWE-agent-for-eval`), OpenHands and Claude Code (and Win-Agent for
Windows tasks). gbench runs the first, the SWE-agent fork, the scaffold of the leaderboard's
SWE-agent rows (`SWE-bench-Live/submission` `submissions/multilang/all_languages/sweagent/*`,
`submissions/lite/sweagent/*`).

1. The fork runs at `824224225e1e952b48da91f2d1bde38adb2115f2` (main, 2026-06-02) from the
   local image `gbench-sweagent-for-eval:82422422-rex3`, installed as its README says (Python 3.12
   venv, `pip install --editable .`, its vendored `swerex` 1.4.0 copied into the venv, plus
   gbench's fix of the task containers' runtime, see "The task containers' runtime" below, and a
   fix of SWE-agent's read of the model's edited files at submit:
   `gbench/docker/sweagent_for_eval_review_decode.patch` makes PatchFormatter (which records the
   edited regions in the trajectory's info) read each file the model's patch modified with
   `errors="replace"`, so a modified file holding a byte that is not UTF-8 no longer ends the
   instance with no patch, as it did for privacyidea__privacyidea-4251 on 2026-10-08, and
   first asks the task container (a subprocess outside the model's shell session,
   `sh -c 'test -e "$1" && ! test -f "$1"'`) whether a modified path exists and is not a
   regular file: such a path (a git submodule's directory, a FIFO, a device) is not read and its
   entry in that record is empty, so a patch that moves a submodule's commit no longer ends the
   instance with IsADirectoryError and no patch, as it did for duckdb__ducklake-1224
   (`/testbed/duckdb`) on 2026-10-09 (in that task image the submodule is checked out at another
   commit than the one recorded, so every submission there carries that change); a missing path
   is still read, and fails, as upstream; the submitted patch is produced separately and is
   untouched; the image carries the label `org.gbench.sweagent_for_eval_agents`, the sha256 of
   the patched `sweagent/agent/agents.py`,
   `5a56a297803679c850ee356554a175a45215226b9ca09e3717ee958be1eb4217`), with
   its own
   `config/default.yaml`, unmodified (sha256
   `05e707df58e9732ac4053f8bda0c0b175aa10f5c833f5419ba1443a01d9eb981`): the system and
   instance templates (the `problem_statement` plus general workflow instructions, working
   directory `/testbed`), the `bash`, `str_replace_editor` and `submit` tools
   (`tools/registry`, `tools/edit_anthropic`, `tools/review_on_submit_m`), function-calling
   actions, at most 100 model calls per instance (`per_instance_call_limit`), no cost limit.
   The fork's own changes stay as they are: a 90-minute limit per command (whatever
   ToolConfig's default `execution_timeout` of 30 seconds says, in `sweagent/tools/tools.py`;
   `config/default.yaml` does not override it), 5 hours of command time per instance, `exit`
   wrapped in a subshell, no repository reset. See "Time limits, hangs and cleanup" below.
2. gbench starts its README's rollout command once per tab (tabs in different lanes at the same
   time, see "Tabs run at the same time"), in that image:
   `sweagent run-batch --config config/default.yaml --agent.model.name openai/<served model>
   --agent.model.api_base <relay>/v1 --instances.type swe_bench --instances.subset
   <instances.jsonl> --num_workers <the tab's share of --sandboxes> --output_dir <dir> --progress_bar False`, plus
   `--instances.deployment.docker_args=--label=org.gbench.sweagent_batch=<batch container name>`
   (a label on each task container, nothing else; see "Time limits, hangs and cleanup"). The
   instance file holds, per instance, what the fork's loader (`SimpleBatchInstance.from_swe_bench`)
   reads: `instance_id`, `problem_statement`, `base_commit` (recorded, not used to reset) and
   `image_name`, the image the tab's harness scores in: the row's `docker_image` for the
   multi-language tabs, and for the Python tabs `starryzhang/sweb.eval.x86_64.<instance_id with
   __ -> _1776_>` (upstream's `get_default_image_name`). Nothing else from the row reaches it.
3. Before the batch starts, gbench pulls each selected instance image, and
   `python:3.11.9-slim-bookworm` (the builder stage of SWE-ReX's runtime layer,
   `swerex/deployment/docker.py` `glibc_dockerfile`), with the host's Docker CLI from the gbench
   process
   (`swebench_common.host_pull_images`: `docker image inspect`, then `docker pull` only for a
   missing image). The pulls use your Docker client configuration (`docker login`, credential
   helpers, `DOCKER_CONFIG`). The SWE-agent container gets no Docker client configuration
   (`HOME=/tmp`, no `~/.docker`, no `DOCKER_CONFIG`), so no registry credential enters it or any
   task container, and SWE-ReX (`pull: missing`) finds every image present. An instance whose
   image cannot be pulled is not handed to SWE-agent: it is an infrastructure failure
   (`agent_exit_status: "image_pull_failed"`, the CLI's message in `agent_error`). If the builder
   base cannot be pulled, that is recorded; SWE-ReX's build then fetches it itself or, failing
   that, uses its own fallback (building on the instance image). The outcome of every pull is in `settings.<name>.image_pulls` and in
   `<agent_output_dir>/image_pulls.json`.
4. SWE-agent's `DockerDeployment` builds a derived image of each instance image with a standalone
   Python for its runtime (Python 3.11.8 from python.org as upstream; swe-rex 1.4.0 and its
   dependencies from PyPI at pinned versions with gbench's patched session code, where upstream
   takes the latest swe-rex; on the host daemon, cached per instance image) and starts it with the
   runtime's port published; the
   agent drives a bash session in it. The container runs with docker-out-of-docker and the host
   network, as the caller's uid. The SWE-agent container mounts the docker socket, as upstream's
   design needs (`DockerDeployment` builds and starts the instance containers on the host daemon
   and reaches each one's published runtime port over the host network), and the model's
   commands run in the instance container, which gets no socket. The residual risk: the socket
   gives the SWE-agent process, which turns the model's replies into those commands,
   root-equivalent access to the host's docker daemon (and so to whatever the host can reach),
   and the host network lets it reach every service listening on the host; a flaw in SWE-agent,
   SWE-ReX or litellm that let model output run inside the SWE-agent container would reach both.
   gbench keeps upstream's design (the rollout cannot run without a daemon it can reach) and
   keeps credentials out of that container (`OPENAI_API_KEY=EMPTY`, no Docker client
   configuration; the relay adds the endpoint's key).
5. The submission is the fork's: its `submit` tool shows a review message first and submits on
   the second call, and a rollout that ends another way (call limit, context window, errors) is
   autosubmitted; both run `git add -A && git diff --cached` in `/testbed`. That is the patch the
   leaderboard's SWE-agent rows submitted: 7 of the 224 patches of
   `multilang/all_languages/sweagent/gpt-5.5-medium` contain `Binary files ... differ`, which
   `git diff --cached` writes and the `git --no-pager diff HEAD --text` of upstream's
   `evaluation/README.md` would not. gbench reads each instance's `<id>/<id>.pred`
   (`model_patch`) and `<id>/<id>.traj` (`info.exit_status`, `info.model_stats`, and the
   message of its last step). The patch goes to the harness verbatim.
6. Scoring: the Python tabs run the **SWE-bench-Live fork** of
   `swebench.harness.run_evaluation` (namespace `starryzhang`); the multi-language tabs run
   **upstream's multi-language harness**, `python -m evaluation.evaluation --dataset <tab rows>
   --patch_dir <predictions> --platform linux --workers N --output_dir <dir> --overwrite 1`
   from `microsoft/SWE-bench-Live` at `62d02936a0726fbf4597b05bd07b09fb99eddae6` with its
   RepoLaunch submodule at `c4b623d930f3728e5338664bb634021b98492cbf` (the harness that
   repository's README gives for `SWE-bench-Live/MultiLang` and `SWE-bench-Live/Windows`; the
   README points Python-dataset users to the fork). Both run inside the local image
   `gbench-swe-bench-live:a145aa87`, docker-out-of-docker. An instance resolves iff the multi-language
   harness puts it in `success_ids` (no `PASS_TO_PASS` failure, no `FAIL_TO_PASS` failure, every
   `FAIL_TO_PASS` test passed; `evaluation.py` `run_instance`). That harness starts each task
   container with 4 CPUs and 16 GB (RepoLaunch's limits; its README notes some C++ repositories
   need more) and prunes dangling images on the host daemon when a container is cleaned up.
   Before either harness pass, gbench pulls each submitted instance's image on the host the same
   way (the harness image has no Docker client configuration either): the Python tabs'
   `starryzhang/sweb.eval.x86_64.<instance_id with __ -> _1776_>`, the multi-language tabs'
   `docker_image`. The outcome is in `settings.<name>.harness_image_pulls`. An image that could
   not be pulled is left to the harness, which tries its own pull and, if that fails too, reports
   the instance as an error (a harness error, not the model's verdict). After the agentic
   rollouts these images are already present.

The model: the fork's `LiteLLMModel` (litellm, `openai/<served model>`) goes through gbench's
relay (`swebench_common.EndpointRelay`, a thread of the gbench process on 127.0.0.1). The relay
forwards only `/v1/chat/completions`, `/v1/completions` and `/v1/models` to the served endpoint
(any other path is refused with 400) and adds the endpoint's key; the SWE-agent container gets
the placeholder `OPENAI_API_KEY=EMPTY`. It also sets `chat_template_kwargs.enable_thinking` from
`--eval-thinking`, which the fork's request cannot carry. The fork's model call sends the
function-calling `tools` but no temperature, top_p or max_tokens (they are commented out in
`sweagent/agent/models.py` `LiteLLMModel._single_query` for every non-Azure model). The relay
sets the temperature on every request: gbench's rule for every suite: 1.0 with `--thinking` (`temperature_source` `"thinking"`), 0.0 without (`temperature_source` `"default"`), unless the operator sets one. The served endpoint
must return tool calls (for vLLM, `--enable-auto-tool-choice` with the model's tool-call parser;
a server gbench launches itself (no `--remote-endpoint`) gets both from `--tool-call-parser NAME`, and the CLI refuses a local run of this suite without it).

Exit statuses. These are infrastructure failures (harness errors, `harness_error:
"agent_infra"` on the trace: left out of the setting's resolved rate, numerator and denominator,
and listed under the setting's `partial_evaluation`, never the model's outcome):

- an instance whose image could not be pulled (above), or without a `.pred` file (its run
  raised: image build or container start failed; `run_batch_exit_statuses.yaml` then says
  `Uncaught <Exception>`);
- an exit status that names `exit_api` (a tenacity `RetryError`), with or without an
  autosubmitted patch, when the tab's relay saw an endpoint failure (or no relay carried the
  run). The fork's model calls re-raise their last error (below), so an endpoint failure never
  ends as `exit_api`, and SWE-ReX rebuilds the exceptions the runtime in the task container
  reports; with no endpoint failure the relay saw, it is the model's outcome, kept with its note
  like the other kept endings;
- an exit status that names `exit_error` whose recorded exception is a model-endpoint failure
  the relay saw. The fork's `forward_with_handling`
  (`sweagent/agent/agents.py`) gives a SWE-ReX `SwerexException` the status
  `exit_environment_error` ("Exit due to environment error: ...") and every other exception it
  does not name `exit_error` ("Exit due to runtime error: ..." / "Exit due to unknown error:
  ..."), and puts that message in the last step of the trajectory. An endpoint that keeps
  failing ends there: `LiteLLMModel.query` (`sweagent/agent/models.py`) retries with tenacity
  `reraise=True`, so after its retries the last litellm exception reaches that handler as
  `exit_error` with "litellm.<Class>: ..." (`InternalServerError` for a 5xx, `NotFoundError` for
  a 404, `AuthenticationError` for a 401, `APIConnectionError`, ...), never a `RetryError`. So
  an `exit_error` is a harness error when the exception text after the "Exit due to ... error: "
  prefix starts with a LiteLLM or OpenAI SDK endpoint exception
  (`swebench_common.ENDPOINT_EXCEPTIONS`, `swebench_common.scaffold_error_harness_reason`) and
  the tab's relay saw an endpoint failure (`upstream_errors` or refused requests in
  `settings.<name>.relay`). The corroboration is per tab (one SWE-agent batch), not per
  instance: one endpoint failure the relay saw anywhere in the tab makes every endpoint-named
  exit of that tab a harness error. The text alone is not enough: SWE-agent talks to a SWE-ReX runtime
  inside the task container, where the model's commands run as root, and SWE-ReX rebuilds the
  exception that runtime reports from the class path and message it sends. gbench gives the
  fork's litellm calls a client timeout 60 s longer than the relay's safety cap on a model
  request, with no transport retries (see "Model request timeout"), so a real endpoint failure is always one the
  relay saw.

Every other status is the model's outcome, scored with the submission as it is (an empty one is
an empty patch): `submitted`, `submitted (exit_cost)` after the 100-call limit, `exit_context`,
`exit_format`, `exit_forfeit`, `exit_command_timeout`, `exit_total_execution_time`, an
`exit_error` whose message says the prompt did not fit the context window ("maximum context
length", "maximum model length", "context length", "context window" or
`context_length_exceeded`; vLLM's "... is longer than the maximum model length ..." is not a
message the fork's own check recognises, so it lands in `exit_error`: the model's outcome, as
`exit_context` is), and any other `exit_error` or `exit_environment_error` (a SWE-ReX
`SwerexException` comes from the runtime in the task container or the connection to it, never
from the model endpoint; a docker daemon message cannot come from SWE-agent's steps, which reach
the task container through its runtime, not the docker CLI). The model's commands can end a
rollout that way themselves: `kill -9 $$` makes SWE-ReX's shell session in the task
container hit the end of its output (pexpect's EOF, re-raised to the agent as an environment
error), a command that kills the runtime server leaves the agent's next request unanswered, and
a command that hijacks the shell's output or breaks SWE-agent's state command raises SWE-ReX's
`NoExitCodeError` or `NonZeroExitCodeError`. Upstream counts every such instance in the
leaderboard's denominator, and so does gbench: the trace carries `agent_exit_note` and the
setting counts them in `exits_kept_as_model_outcome` (ids in `exits_kept_as_model_outcome_ids`).
Each trace keeps the message in `agent_exit_message`.

What the relay counts. The relay reads every reply it forwards
(`swebench_common.EndpointRelay`, recorded in `settings.<name>.relay`) and each agentic tab
applies the counts (`swebench_common.relay_result_reasons`):

- Endpoint failures: a request the endpoint did not answer (`connection`), a reply that is not
  2xx (`http_<status>`; a 400 whose message names the context window is counted apart in
  `context_length_rejections`, since the scaffold treats it as the model's outcome), a 2xx
  completions reply that is not JSON, an error event in a stream, and any request the relay
  refused. They are harness errors: listed in `endpoint_failures`, the tab's `status` becomes
  `completed_with_errors`, and the tab is not comparable. The scaffold retries some failures
  itself, and a request cannot be tied to an instance, so an instance is classified by its own
  exit status as above.
- Model turns that stopped at a limit (`finish_reason: length`: the run's `--max-output-tokens`,
  which gbench's relay sends as max_tokens with every agent request while the scaffold sends
  none; the end of the served context window with `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0`) are counted in `truncated_turns`, which above zero
  makes the tab not comparable. Turns that stopped on their own with reasoning but neither
  content nor a tool call are counted in `empty_after_reasoning_turns` and split by their token
  ids (swebench_common.classify_empty_reply): the answer inside the thinking channel
  (`answer_in_thinking`, with `answer_in_thinking_rate` over `model_replies`, every reply
  choice): the model's outcome, not a comparability reason; a native tool call the server's
  tool-call parser removed (`tool_call_dropped_turns`) and turns without token ids
  (`unclassified_empty_turns`), each a comparability reason of the tab (see
  [answer_in_thinking.md](answer_in_thinking.md)); each tab keeps its own under
  `settings.<tab>`, and the top level adds them over the tabs. They are per-tab counts: on each trace `truncated_turns` is `null` (a relayed
  request cannot be tied to an instance).

Time limits, hangs and cleanup. The fork's limits are the only ones on a rollout
(`sweagent/agent/agents.py`): at most 100 model calls; each command ends after 90 minutes
(`handle_action` runs every command with `timeout=90*60`, "1.5 h limit to align with evaluation
code"; the model then sees the "cancelled because it took more than 30 seconds" message that
ToolConfig's default `execution_timeout` of 30 produces); 3 consecutive command timeouts
(ToolConfig's default `max_consecutive_execution_timeouts`) end the rollout
(`exit_command_timeout`); and the
rollout ends once its commands took more than 5 hours in all (`exit_total_execution_time`).
The fork sets no wall-clock limit per instance or per batch, so one instance can take several
hours, and an outer timeout (a `timeout` wrapper, a job scheduler) has to allow for that.

The task containers' runtime (fixed by gbench on 2026-10-08). The model's commands run in a bash
session that SWE-ReX's runtime server drives inside each task container. Upstream's
DockerDeployment installs the latest swe-rex from PyPI into each instance's runtime layer
(swe-rex 1.4.0 with bashlex 0.18 at the time), and its session split every command with bashlex
and joined the pieces with " ; " (`swerex/runtime/local.py` `_split_bash_command`). A heredoc
whose operator is not the last word of its line (`cat <<EOF > file`, `python3 - <<EOF 2>&1`,
`cat <<EOF | sh`) lost its body and terminator in that split, and any heredoc followed by another
command got that command on its terminator's line (`EOF ; python3 file`), so the shell waited for
a terminator that never came until the fork's 90-minute command limit ended the step, although the
command is valid bash. Gemma writes its reproduction scripts that way in almost every instance:
on the 2026-10-08 smoke run each instance hit it 2 to 4 times, and 5 instances took 6 hours. That
defect is in the harness, not in the model, so gbench fixes it:
`gbench/docker/sweagent_for_eval_swerex_heredoc.patch`, applied by the image build to the SWE-ReX
the fork vendors, makes the runtime layer install swe-rex 1.4.0 and its dependencies at pinned
versions (`swe_bench_live.SWEREX_RUNTIME_PINS`, the versions pip resolved on 2026-10-08, installed
with `--no-deps` and checked with `pip check`) and write gbench's session code over swe-rex 1.4.0's
`swerex/runtime/local.py` (both files checked by sha256; the patched one is
`d67d68966754cc73a3c2ae084fdfe90e7fb274e3ae36da6e619e1c4155372ed4`, the image label
`org.gbench.sweagent_for_eval_swerex_runtime`). In that session code a command holding a heredoc
operator (`<<`, `<<-`; not the here-string `<<<`) is never split: its exact text goes to a file
and the session sends one line, `builtin eval "$(command -p cat FILE; command -p rm -f FILE)"`,
so bash reads the whole command before running any of it and parses it as it parses a script
(the file is gone before the first command runs; the working directory, variables and jobs
persist; the exit status is the last command's). A piece ending with the background operator `&`
is no longer followed by " ; " (`a &` then `b` was the syntax error `a & ; b`). Every other
command is split and joined exactly as before, and the model sees the same prompts, the same
tools and the same output format; its commands now run as written. The fix was checked end to end
(SWE-ReX's DockerDeployment in four task images: Debian 12, Ubuntu 22.04 and 24.04, a Go image;
SWE-agent run-batch with its replay model): the model's `cat <<EOF > reproduce_issue.py ...`
returns in 0.1 s where it hung, and plain multi-line commands, `&&`, `;`, `cd` persisting, exit
codes, quoted and `<<-` delimiters, two heredocs, `cat <<EOF | sh`, `python3 - <<EOF 2>&1`,
`<<<`, background jobs, a 5000-line heredoc and the command-timeout and interrupt path behave as
in bash.

gbench adds a watchdog (`swe_bench_live.SweagentWatch`) that only acts when nothing moves for
longer than those limits allow. SWE-agent writes each instance's logs and trajectory as the
rollout goes. Outside a model request the longest a live rollout stays silent is one command
(90 minutes); during a model request it is SWE-agent's client timeout (the relay's safety cap on
a model request plus 60 s; see "Model request timeout"; a retry is logged). The relay ends a
model request that stops making progress after 300 s, so that long silence lasts only while a
reply is still being generated. Every 30 s it checks the batch's files:

- an instance that started, has no `.pred` yet and whose files under `<agent_output_dir>/<id>/`
  have not changed for 6300 s (the command limit plus 15 minutes,
  `agent_watchdog.stall_limit_s`) is stalled. Only while its trace log shows a model request in
  flight (`swe_bench_live.sweagent_model_request_inflight`: the last `swea-lm` record is the start
  of a query, `n_cache_control: N` or `Retrying LM query`, with no `Response:` or any other record
  after it) is the limit SWE-agent's client timeout plus 15 minutes, computed when the batch
  starts (`swe_bench_live.sweagent_stall_s`, recorded as
  `agent_watchdog.model_request_stall_limit_s` with `model_request_limit_s`: 35,048 s on a
  262,144-token window at the default `GBENCH_MIN_DECODE_TOK_S` of 8). A wedged command, runtime
  or docker call is therefore still found after 6300 s (2026-10-11 review: one limit sized for
  the longest model request would have left it undetected for 9.7 h). gbench removes its
  task container, and the instance is a harness error of kind `agent_stalled`
  (`agent_infra_error` true, the reason in `agent_error`, which names the first line of the
  command it was running: the last `Input:` record of `<agent_output_dir>/<id>/<id>.trace.log`
  with no `Output:` after it, logger `swea-env-<id>` in a batch), whatever SWE-agent then records
  for it; it is left out of the tab's resolved rate, numerator and denominator;
- a batch in which no file has changed for 115 minutes (6900 s: the stall limit plus 10;
  `batch_stop_limit_s`), or for the model-request limit plus 10 minutes while one of its live
  instances waits for a model request (`model_request_batch_stop_limit_s`), is stopped: gbench removes the SWE-agent container and every task container of the batch, and
  every instance without a `.pred` is a harness error (`agent_infra`, its `agent_error` naming
  the stop).

An instance SWE-agent already ended (listed in run_batch_exit_statuses.yaml, e.g. `Uncaught
<Exception>`) is never flagged. Until the runtime fix the watchdog also ended, after 5 minutes of
silence, an instance whose command held a heredoc the runtime dropped (`agent_heredoc_dropped`);
that rule is retired, since a heredoc now runs as bash runs it and a long command that holds one
is an ordinary long command.

What the watchdog saw is in `settings.<name>.agent_watchdog` and
`<agent_output_dir>/watchdog.json` (the limits `stall_limit_s`, `batch_stop_limit_s`,
`model_request_limit_s`, `model_request_stall_limit_s`, `model_request_batch_stop_limit_s`;
`stalled_instances`, `removed_task_containers`, `batch_stopped`). Every task container SWE-ReX starts carries the
label `org.gbench.sweagent_batch=<batch container name>`. When the batch ends, however it ends,
gbench removes the SWE-agent container (`gbench-sweagent-<hex>`, started with `--init` so a
SIGTERM the docker CLI forwards stops SWE-agent) and every container with that label (SWE-ReX
removes a task container itself only when its rollout ends normally). A SIGTERM or SIGHUP to a
gbench run while a batch runs (for example from `timeout`) goes the same way: the CLI turns it
into an exit that runs this cleanup. The signal is also recorded for the rest of the run: a
suite on a worker thread (`--suite-timeout`) gets it handed over, and from there every tab
running at that moment gets it (within a second): each tab's watchdog stops its batch within
30 s, and no further tab starts or is scored. The suite's result is a failed record whose
error names the signal and how long the suite ran (`terminated by signal 15 (SIGTERM) after N s
while this suite ran ...`), saved to its eval_*.json and to summary.json. When the signal also
reaches the docker CLI (a signal to the process group, as `timeout` sends), the docker CLI
forwards it and `--init` passes it on to SWE-agent, so the batch and the suite stop even when
the suite runs on a worker thread (`--suite-timeout`).
A SIGKILL cannot be caught: remove what is left with
`docker rm -f $(docker ps -aq --filter label=org.gbench.sweagent_batch)` and
`docker rm -f $(docker ps -aq --filter name=gbench-sweagent-)`.

**Windows tab.** Its instance images (`starryzhang/sweb.eval.win.*`) are Windows containers. The
leaderboard's Windows rows all run **Win-Agent** (`github.com/njukenanli/Win-Agent`, named for
Windows tasks in upstream's `evaluation/README.md`), so gbench runs Win-Agent at
`47d85cffd5342a513fa3fb72ec4b5347ced9bb36` with its own `config/default.yaml` (tools shell,
read, create, string_replace, submit; prompt; `max_steps` 100; `instance_timeout` 480 min;
`platform: windows`). gbench changes only the dataset file (the selected rows of the pinned
snapshot), the model (`openai/<served model>`), its endpoint and `workers` (`--sandboxes`). It
runs `main.py --config_path <that file> --api_key EMPTY` from a fresh copy of the checkout, since
Win-Agent writes `./output` and mounts `./mnt` into each task container. Scoring is the same
multi-language harness with `--platform windows`, run on the host from the pinned
SWE-bench-Live checkout (`PYTHONUTF8=1`, as upstream's README asks). Win-Agent's model goes
through gbench's relay (`swebench_common.EndpointRelay`, a thread of the gbench process on
127.0.0.1; the same allowed paths as above). The relay forwards to the served endpoint and adds
its key (Win-Agent gets the placeholder `EMPTY` and an environment without credential
variables). It also sets `chat_template_kwargs.enable_thinking` from `--thinking`,
which Win-Agent's `litellm.completion` call cannot send. Win-Agent sends no temperature and no
max_tokens. The relay sets the temperature (gbench's rule for every suite: 1.0 with `--thinking`
(`temperature_source` `"thinking"`), 0.0 without (`temperature_source` `"default"`), unless the
operator sets one) and `max_tokens` = the run's `--max-output-tokens`
(`GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0`: none) on every request. Both programs
bind-mount a directory of the host they run on into each task container (`./mnt` ->
`C:\mnt`, `./tmp` -> `C:\mnt_tmp`), so they have to run on the Windows host itself. On any other
host the `windows` setting is `infra_required`, naming this requirement: in its own entry when
other settings run, as the suite's error when it is the only one selected. On a Windows host it
runs only after its own cloud metadata server check passed (see "The Windows tab's metadata
server check" under Prerequisites), and is `infra_required` with the reason otherwise. Win-Agent
records `error` for a rollout that raised (its `exit_status.json`) and logs the exception to the
instance's log (`output/<run id>/logs/<instance_id>.txt`: the exception text, its traceback and
"Patch might not be saved." / "Patch not saved."). gbench reads that log (`swe_bench_live.
win_agent_error_verdict`):

- no log, or no `Step N...` entry before the last error entry: Win-Agent raised before the
  model's first turn (container launch, Python install, git setup): an infrastructure failure
  (`agent_infra`, left out of the tab's rate, listed under `partial_evaluation`);
- after the model's first turn, an error whose exception text starts with a LiteLLM or OpenAI SDK
  endpoint exception (Win-Agent's `src/llm.py` re-raises the last one after its 5 tries) is an
  infrastructure failure only when the relay saw an endpoint failure during the tab
  (`upstream_errors` or refused requests in `settings.windows.relay`). The log also holds the
  raw output of the model's shell commands (Win-Agent's runtime logs "Status inside container:
  <output>"), so an entry written after the first turn could be the model's own text; Win-Agent's
  litellm calls send no timeout, and gbench sets litellm's own (`REQUEST_TIMEOUT` in Win-Agent's
  environment) 60 s longer than the relay's safety cap on a model request, so a real endpoint
  failure is always one the relay saw;
- any other error after the first turn (e.g. the shell of a container the model's command
  stopped) is the model's outcome, unresolved, as upstream counts it, with `agent_exit_note`,
  counted in `exits_kept_as_model_outcome`. An error naming the docker daemon after the first
  turn is one too: Win-Agent reaches the daemon for a rollout only in
  `Runtime.from_launch_image`, before it.

Run knobs reach the agents through gbench's relay (both scaffolds use litellm with
`openai/<served model>`):

| Knob | Effect on the agent |
| --- | --- |
| `--temperature` / `GBENCH_SWE_BENCH_LIVE_TEMPERATURE` | replaces gbench's rule (1.0 with `--thinking`, 0.0 without); `GBENCH_SWE_BENCH_LIVE_TEMPERATURE` wins over `--temperature`. The relay adds the temperature to every model request; SWE-agent-for-eval and Win-Agent send none, so the tab is not leaderboard-comparable |
| `--eval-thinking` | `chat_template_kwargs.enable_thinking` on every request (set by the relay) |
| `--max-output-tokens` | sent as every agent turn's `max_tokens` (set by gbench's relay as a body field), on by default since 2026-10-11, a recorded deviation (neither scaffold sends one). `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0` restores upstream's none (the server grants each turn what the context window has left). The one-shot mode sends it as its single request's `max_tokens` |
| `--sandboxes` (or the eval concurrency) | the suite's budget, dealt over the tab lanes that run at the same time (see "Tabs run at the same time"); each tab's share is its parallel agent rollouts (`--num_workers` / Win-Agent `workers`) and its harness `--max_workers` / `--workers` |
| `--eval-limit` / `--shard` | `--eval-limit N` is N instances for the whole suite, dealt over the tabs that run (`sampling.allocate_limit`, seed `swe_bench_live`; a tab whose share is 0 is not run, `settings_not_run_for_limit`), each tab's share a stratified subset by repository / round-robin shard, applied to each setting |
| `--eval-categories` | the leaderboard settings to run (`lite`, `full`, `verified`, `ccpp`, `csharp`, `java`, `tsjs`, `go`, `rust`, `windows`; default all ten) |
| `GBENCH_SWE_BENCH_LIVE_TRAJ_DIR` | where the trajectories go (default: `<run results dir>/agent_trajectories/`) |
| `GBENCH_SWE_BENCH_LIVE_AGENT_IMAGE` | the SWE-agent-for-eval image name, if you retag it (default `gbench-sweagent-for-eval:82422422-rex3`) |
| `GBENCH_SWE_BENCH_LIVE_WIN_AGENT_DIR` | Windows tab: checkout of Win-Agent at `47d85cff` |
| `GBENCH_SWE_BENCH_LIVE_HARNESS_DIR` | Windows tab: checkout of microsoft/SWE-bench-Live at `62d02936` with its launch submodule |
| `GBENCH_CACHE_DIR` | root of the pinned dataset snapshots (default `~/.cache/gbench`) |

The harness's thread cap (`GBENCH_SWE_OMP_THREADS`) reaches the task containers both Linux
harnesses create: gbench mounts its standalone `swe_thread_cap.py` into the harness image at the
same path and starts the harness through that shim, which adds the variables to every container
the docker SDK creates. Unset, the cap is `min(4, CPUs // N)` threads per container, with N
the suite's whole concurrency budget (`--sandboxes`), the containers that can run at once over
all the tab lanes, even though each tab runs with only its lane's share (recorded as
`docker_thread_cap` in each tab's result). The agent containers run as SWE-agent starts them, with no gbench
setting added. A conversation that outgrows the served context window ends the rollout with
`exit_context` and an autosubmitted patch (the model's outcome, as upstream records it).

**One-shot (opt-in).** `GBENCH_SWE_BENCH_LIVE_MODE=oneshot`: one request per instance with the
issue text only, a diff extracted from the reply, same harness. Always
`leaderboard_comparable: false`.

## Why this suite uses its own image
SWE-bench-Live ships the Python harness as a **fork of the `swebench` package**: the same package
name at a version incompatible with the upstream `swebench` 4.1 the main gbench env carries (and
with the swebench 5.0.2 that `swe_bench_multilingual` and `copilot_bench_swe` now score with, in
their own image). Only one `swebench` can be installed per environment. Measured with swebench
4.1 by building a `TestSpec` for one row of each dataset (at their earlier revisions):

| `swebench` version | `copilot_bench_swe` | `swe_bench_live` | `swe_bench_multilingual` |
| --- | --- | --- | --- |
| upstream (PyPI) | OK | FAIL | OK |
| SWE-bench-Live fork | OK | OK | FAIL |

So gbench **isolates the harnesses in a local Docker image** rather than a separate host
virtualenv. Upstream's multi-language harness lives in the same image as a source tree (its
`pyproject.toml` names the package `swebench` too, so it is not pip-installed; gbench runs it
with `PYTHONPATH=/opt/swe-bench-live-main/launch:/opt/swe-bench-live-main`). The rollout runs in
its own image (`gbench-sweagent-for-eval:82422422-rex3`, or Win-Agent on a Windows host); only the scoring step
runs inside this one, and it spawns the per-instance `starryzhang/*` task containers on the host
daemon. Nothing fork-versioned touches the main env, so all the SWE
suites run from one gbench install.

## Model request timeout

The agents' model clients wait for the relay, not for a hidden client limit. Until 2026-10-11
SWE-agent-for-eval's litellm 1.87 gave each request 600 s (its `request_timeout` is the package
default 6000, which litellm's chat completion replaces by `COMPLETION_HTTP_FALLBACK_SECONDS`) and
the OpenAI SDK made 2 more attempts, so a turn needing more than 600 s of decode was abandoned and
sampled again: 1800 s per SWE-agent attempt, up to 20 attempts (2026-10-10 smoke run: 12 times on
one trimesh instance). Results from before that date were measured under that cap and are not
comparable with later ones on long turns.

Now, by default (cap on), each tab sizes the relay's per-request limit as
`base.request_timeout_s` of the run's `--max-output-tokens` (9,392 s at 65,536 tokens at the
default `GBENCH_MIN_DECODE_TOK_S` of 8). The relay's safety cap on a model request is that plus
120 s (`EndpointRelay.TOTAL_CAP_MARGIN_S`), and the agent's client waits the cap plus 60 s
(T = 9,572 s there, `swe_agentic.client_timeout_for_relay_limit`), so the relay always answers
first. With `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0` both are sized for the served context window (the
run knob `max_model_len`; `base.UNCAPPED_TIMEOUT_TOKENS`, 262,144, when it is unknown: 33,968 s
relay limit / T = 34,148 s client). Each relay counts a client close near T in
`client_timeouts_suspected`:

- SWE-agent-for-eval: `--agent.model.completion_kwargs '{"timeout": T, "max_retries": 0}'` on the
  `run-batch` command, which `LiteLLMModel._single_query` passes to every `litellm.completion`
  call. Each request is one HTTP attempt; SWE-agent's own retry loop (`RetryConfig`, 20 attempts)
  still handles an error the relay reports.
- Win-Agent: `REQUEST_TIMEOUT=T` in Win-Agent's process only (its `src/llm.py` passes no timeout,
  so litellm uses `request_timeout`; never exactly 6000, which litellm reads as unset). Its
  transport retries stay the OpenAI SDK's default (2), which no variable reaches; with the client
  timeout above the relay's safety cap they retry only an error the relay answers.

A request that still runs into the relay's limit means the endpoint decoded below
`GBENCH_MIN_DECODE_TOK_S`: the relay answers it as an endpoint failure. Every agentic tab records
`model_client_timeout` (`client`, `version`, `timeout_s`, `retries`, `source`, and
`hidden_cap_removed` / `replaces` naming the 600 s cap) and `agent.relay_timeout_s`. Nothing the
model is sent or allowed changes.

`GBENCH_AGENT_MAX_TOKENS_FROM_RUN` (default on since 2026-10-11; `=0` turns it off) makes the relay set `max_tokens` to the run's
`--max-output-tokens` on every agent request (body field), and sizes both limits for that value
instead of the window. The scaffolds send no max_tokens, so such a run records
`official_sampling.agent_max_tokens_from_run`, `max_output_tokens_per_turn` and a
`leaderboard_comparable_reason` entry. vLLM refuses a request whose prompt plus max_tokens exceeds
the window, so late in a long rollout a turn can be refused where upstream's would have been
granted the rest of the window (`context_length_rejections` in `relay`).

## Reading the result
- One result holds every setting that ran (`leaderboard_settings`, in the leaderboard's tab
  order). `headline_metrics` has each setting's resolved rate (`lite_resolved_rate`,
  `full_resolved_rate`, `verified_resolved_rate`, `ccpp_resolved_rate`,
  `csharp_resolved_rate`, `java_resolved_rate`, `tsjs_resolved_rate`, `go_resolved_rate`,
  `rust_resolved_rate`, `windows_resolved_rate`, in %; resolved instance ids / rows of the
  setting, harness errors left out). `accuracy`, `total_questions`, `scored_instances` and
  `correct_answers` are Lite's (the leaderboard's default tab; `headline_setting` names it, or
  the first setting run when Lite is not selected or was not run for a missing prerequisite);
  when that setting has no score, `accuracy` is `null` with `accuracy_withheld_reason`.
  `category_accuracy` has one entry per setting (`total` is its scored instances).
- `settings.<name>` is that setting's full result without its traces: `dataset` (the pinned
  commit and splits), `total_questions`, `scored_instances`, `correct_answers`, `accuracy`,
  `category_accuracy` (per repository), `harness_errors` and `harness_error_ids` (when any),
  `status`, `leaderboard_comparable` and its reason, `agent_output_dir`.
  The Python tabs add `swebench_report` and `harness_report_retries` (and for `verified`
  `total_instance_ids` and `dataset_repeated_instance_ids`). The multi-language and Windows
  tabs add `tab` (the board's label), `harness` (tree, RepoLaunch commit, platform, where it
  ran), `harness_report` (the harness's `results.json`: `submitted`, `success`, `failure`,
  `empty_patch`, `error` and their id lists), `harness_workdir` (its logs and per-instance
  reports), `leaderboard_totals` and, for C/C++ and TS/JS, `split_accuracy` (per MultiLang
  split). Every agentic tab adds `agent` (SWE-agent-for-eval: its commit, config and sha256,
  call limit and time limits, model, image; or Win-Agent: its commit and config),
  `agent_watchdog` (SWE-agent tabs; see "Time limits, hangs and cleanup"), `relay` (where it
  listened,
  where it forwarded, the paths it allows, the body fields it set, requests relayed and refused),
  `official_sampling`, `model_client_timeout` (see "Model request timeout"),
  `agent_exit_statuses`, `image_pulls` (SWE-agent tabs),
  `truncated_turns`, `empty_after_reasoning_turns`, `model_replies` and, when any occurred,
  `endpoint_failures` (see "What the relay counts"). The Python and multi-language tabs add
  `harness_image_pulls`. `sample_traces` holds every setting's traces, each with its `setting`.
- `status` is `success` when every setting succeeded, `error` when every setting's result is
  `status: "error"` or `"infra_required"`, and `completed_with_errors` otherwise
  (`settings_with_errors` names the settings whose result is `status: "error"`, their
  `settings.<name>.error` says why; `settings_infra_required` names the settings not run for a
  missing prerequisite, see "Per-setting prerequisites"). The prerequisite check covers every
  selected setting before anything runs, and an `infra_required` error raised while a setting
  runs (for example a snapshot that changed after the check) stops the whole run with that
  error.
- `mode`, `agent`, `harness_image` (the image used), `temperature`, `max_output_tokens_note`
  (the per-turn cap sent with every turn and where it was set) and `max_output_tokens_per_turn`
  (the run's `--max-output-tokens`; `null` only with `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0`) are the
  headline setting's at the top level and each setting's own under `settings.<name>`. On every
  trace: `agent_exit_status`, `agent_steps` (model calls; `null` for Win-Agent, which does not
  record them), `agent_patch`, `agent_infra_error`, `agent_error`, `trajectory_file` (the
  scaffold's own trajectory file) and, for SWE-agent, `agent_exit_message`.
- `settings.<name>.harness_report_retries` (Python tabs) counts re-runs of the fork harness's
  final report step alone (its own `make_run_report`, with the scoring pass's arguments; at most
  5, the first at once, then after 5, 15, 30 and 60 s) when the harness scored instances but
  that step crashed (its container listing races with containers other processes remove on a
  shared daemon). Such a re-run starts no container and grades nothing: it reads the
  per-instance `report.json` files the scoring pass wrote, so an instance that pass left without
  one (a test timeout, a crash after the tests ran) stays in `error_ids`, exactly as in a run
  whose report step did not crash. If all 5 fail, that setting is `status: "error"`.
- The raw trajectories SWE-bench-Live submissions ask for are the scaffolds' own files: for
  SWE-agent-for-eval `<agent_output_dir>/<instance_id>/<instance_id>.traj` (with its `.pred`,
  `.patch`, `.config.yaml` and `.trace/.debug/.info.log` files, `run_batch.log`,
  `run_batch_exit_statuses.yaml`, `preds.json`, the instance file, `sweagent_run.log`,
  `image_pulls.json` and `watchdog.json`), and
  for the Windows tab Win-Agent's `win-agent/output/<run id>/logs/<instance_id>.txt`, one
  `agent_output_dir` per setting (`settings.<name>.agent_output_dir`). Each is a fresh
  directory under `<run results dir>/agent_trajectories/`, next to the result JSON;
  `GBENCH_SWE_BENCH_LIVE_TRAJ_DIR` sets another parent. Outside a gbench run (no results
  directory) it is a new directory in the system temp dir, so copy it before cleanup. The
  endpoint's key (`VLLM_API_KEY`, else `OPENAI_API_KEY`) is not in these files: only the relay holds it, the
  SWE-agent container and Win-Agent get the placeholder `EMPTY`, Win-Agent and the harnesses are
  started without any credential variable in their environment, and gbench replaces any
  credential value of the environment found in the `.pred`, `.traj` and `sweagent_run.log` files
  with `[REDACTED]`.
- For the swebench fork (Python tabs), an errored instance (`error_ids`) is classified by its
  `run_instance.log`, reading only the records the harness itself logs: a patch that did not
  apply, or tests over the applied patch that ran past the harness's timeout (`Test timed out
  after <N> seconds.` after the `Test runtime:` record), is the model's outcome, an ordinary
  unresolved instance in the denominator, as upstream counts every `error_ids` instance (the
  gold patch finishes within the timeout, so a hang is what the model's patch did; traces carry
  `test_timed_out`, the setting's `swebench_report` `test_timeout_instances` /
  `test_timeout_ids` and `patch_apply_failed_instances`).
- Infrastructure failures (an image pull, container start, endpoint, harness errors that are
  not the model's verdict: for the swebench fork an errored instance whose `run_instance.log`
  shows neither a patch-apply failure nor a test timeout (docker, image build, container or a
  missing log), for the multi-language harness an instance in its `error_ids` or
  `incomplete_ids`, which raised inside the harness, and every submitted patch of a pass that
  wrote no report) are harness errors: left out of the setting's resolved rate (numerator and
  denominator; `scored_instances` is the denominator, rows when the headline is over rows),
  counted by kind in the setting's `harness_errors` (`agent_infra`, `agent_stalled`,
  `harness_no_verdict`, `harness_no_report`; in one-shot mode also
  base's endpoint failures) with their ids in `harness_error_ids`, and listed under its `partial_evaluation`; the setting is
  `completed_with_errors`. If nothing in a setting could be scored that setting is `status:
  "error"` with `accuracy` `null` (`accuracy_withheld_reason`), never a 0%. For the
  multi-language harness a patch that does not apply is the model's outcome: the tests run on
  the unpatched code and fail (`failure_ids`).
- `leaderboard_comparable` is `true` for a tab run in full with its reference scaffold
  (SWE-agent-for-eval, Win-Agent), no temperature sent (gbench always sends one, so today no
  tab is comparable), no decoding penalties, every
  instance rolled out and scored, no endpoint failure, no model turn cut off at a limit or
  empty after reasoning, and a pinned row count that equals a board total of that tab (today
  every Python tab, Java and Go). A multi-language or Windows tab whose pinned row count
  matches no board row of that tab records that reason; `--eval-limit`, `--shard`, a sent
  temperature (always: the rule or the operator's), decoding penalties, partial scoring, endpoint failures and cut-off or empty
  turns are added when they apply, and a tab not run for a missing prerequisite records
  that. Each setting records its own flag and
  reason; the top-level reason writes a reason shared by every setting once and names the
  settings of any other.

## Prerequisites
- **Docker** (a reachable daemon). The per-instance task images come from Docker Hub
  (namespace `starryzhang`, and the multi-language rows' `docker_image`): gbench pulls them with
  the host's Docker CLI from its own process, so `docker login` on this host (recommended for
  Docker Hub's pull-rate limit) applies to them. No container gbench starts gets your Docker
  client configuration.
- Python package `datasets` (a base dependency of gbench).
- The **SWE-agent-for-eval image** (the agentic Python and multi-language tabs), built once:

  ```bash
  docker build -t gbench-sweagent-for-eval:82422422-rex3 -f gbench/docker/sweagent_for_eval.Dockerfile gbench/docker
  ```

  (An image built before 2026-10-08, `gbench-sweagent-for-eval:82422422`, lacks the runtime fix
  and its label; `gbench-sweagent-for-eval:82422422-rex1` lacks the fix of the read at submit,
  and `gbench-sweagent-for-eval:82422422-rex2` (built 2026-10-09) has only its UTF-8 half and
  still opens a submodule's directory. gbench refuses each, by its
  `org.gbench.sweagent_for_eval_agents` label, with `infra_required` and this command. The build reuses the
  local `ubuntu:24.04` base by digest and pulls nothing from a container registry other than
  that base if it is missing.)

  It fetches `github.com/njukenanli/SWE-agent-for-eval` at
  `824224225e1e952b48da91f2d1bde38adb2115f2` by commit onto `ubuntu:24.04` pinned by digest (its
  Ubuntu packages from the archive snapshot `20260925T120000Z`, Python 3.12, `docker-ce-cli`
  `5:29.8.0-1~ubuntu.24.04~noble`) and installs it as its README says with every dependency
  pinned by `gbench/docker/sweagent_for_eval.constraints.txt` (the set PyPI served on
  2026-06-03, the day after that commit: `litellm` 1.87.0, `pydantic` 2.13.4, ...; resolved set
  recorded in `/opt/sweagent-for-eval-pip-freeze.txt`), then applies
  `gbench/docker/sweagent_for_eval_swerex_heredoc.patch` to the vendored `swerex` (see "The task
  containers' runtime" above). The build checks the commit, the config sha256, the sha256 of
  `swerex/runtime/local.py` before (swe-rex 1.4.0's
  `d56f22b48b02d4df6644cfa04cc16c979b0498ef1c26058b82fb2d4a60bf9ac7`) and after the patch
  (`d67d68966754cc73a3c2ae084fdfe90e7fb274e3ae36da6e619e1c4155372ed4`), the sha256 of the fork's
  `sweagent/agent/agents.py` before (`836541b17cc3977f298d88e9653ddb1f433a092e1bb63a9a79dfc60843e1dc49`)
  and after `gbench/docker/sweagent_for_eval_review_decode.patch`
  (`5a56a297803679c850ee356554a175a45215226b9ca09e3717ee958be1eb4217`), that the patched
  `sweagent.agent.agents` imports, that the runtime layer it
  generates installs the pinned swe-rex, and that the CLI starts as a non-root user. The
  prerequisite check refuses an image whose labels `org.gbench.sweagent_for_eval`,
  `org.gbench.sweagent_for_eval_config_sha256`, `org.gbench.sweagent_for_eval_swerex_runtime`
  and `org.gbench.sweagent_for_eval_agents` name another commit, config, runtime or
  `sweagent/agent/agents.py` (or lack one), and needs `/var/run/docker.sock`. The
  first rollout on each instance image also builds SWE-agent's runtime layer (from
  `python:3.11.9-slim-bookworm`, which gbench pulls on the host with the instance images, it
  downloads Python 3.11.8 from python.org and the pinned swe-rex 1.4.0 and its dependencies from
  PyPI), so the build needs network access. The fork's license is MIT (its `LICENSE`); gbench fetches it into the image and
  vendors none of it.
- Hugging Face Hub access on the first run (to write the pinned snapshots; the prerequisite
  check does this for every selected setting), or the snapshots already in `$GBENCH_CACHE_DIR`.
- The **local image**, built once:

  ```bash
  docker build -t gbench-swe-bench-live:a145aa87 -f gbench/docker/swe_bench_live.Dockerfile gbench/docker
  ```

  It installs the fork from `github.com/SWE-bench-Live/SWE-bench-Live` at commit
  `a145aa87c62361532dacaa243398978164b234b7` on `ubuntu:24.04` pinned by digest (its Ubuntu
  packages from the archive snapshot `20260925T120000Z`, snapshot.ubuntu.com), with the
  Docker CLI pinned (`docker-ce-cli` `5:29.8.0-1~ubuntu.24.04~noble`) and the fork's Python
  dependencies pinned by `gbench/docker/swe_bench_live.constraints.txt` (`datasets` 5.0.1,
  `docker` 7.2.0, ...). It also holds upstream's multi-language harness,
  `microsoft/SWE-bench-Live` at `62d02936a0726fbf4597b05bd07b09fb99eddae6` with
  `microsoft/RepoLaunch` at `c4b623d930f3728e5338664bb634021b98492cbf` in `launch/`, fetched by
  commit (the build checks that the harness commit's submodule entry names that RepoLaunch
  commit) and imported once at build time. The installed versions are recorded in the image
  (`/opt/swe-bench-live-pip-freeze.txt`, `/opt/swe-bench-live-dpkg-versions.txt`). The
  prerequisite check refuses an image whose labels name other commits: the Python tabs need
  `org.gbench.swe_bench_live_fork`, the multi-language tabs `org.gbench.swe_bench_live_main`
  and `org.gbench.repolaunch` (an image built from an older Dockerfile lacks them: rebuild it).
  Override the image name with `GBENCH_SWE_BENCH_LIVE_IMAGE` if you retag it. gbench never
  pulls this image; you build it.
- A served model at an OpenAI-compatible `/v1` endpoint (`VLLM_API_KEY`, else
  `OPENAI_API_KEY`, if it needs a key) that returns tool calls (SWE-agent-for-eval and Win-Agent
  both act through function calls; for vLLM, `--enable-auto-tool-choice` with the model's
  tool-call parser; a server gbench launches itself (no `--remote-endpoint`) gets both from `--tool-call-parser NAME`, and the CLI refuses a local run of this suite without it).
- For the **Windows tab** only: gbench running on a Windows host whose Docker engine runs
  Windows containers (Docker Desktop switched to Windows containers, or Docker Engine on
  Windows Server); `GBENCH_SWE_BENCH_LIVE_WIN_AGENT_DIR` pointing at a checkout of
  `github.com/njukenanli/Win-Agent` at `47d85cffd5342a513fa3fb72ec4b5347ced9bb36`;
  `GBENCH_SWE_BENCH_LIVE_HARNESS_DIR` pointing at a checkout of
  `github.com/microsoft/SWE-bench-Live` at `62d02936a0726fbf4597b05bd07b09fb99eddae6` after
  `git submodule update --init` (RepoLaunch at `c4b623d9...`); and the Python packages
  `litellm`, `docker`, `pyyaml` and `datasets` (Win-Agent's `requirements.txt` and the
  harness's imports; `datasets` is a base dependency of gbench, the others are in its
  `evals` extra). The prerequisite check verifies the
  platform, the engine's server OS and each checkout's commit. The tab also needs one of its own
  instance images on the host for the Windows metadata server check below (gbench never pulls
  it for the check; Win-Agent and the harness pull the others themselves), for example:

  ```bash
  docker pull starryzhang/sweb.eval.win.libsdl-org_1776_sdl-11761
  ```

  Any image of the tab serves (the `docker_image` of a SWE-bench-Live/Windows row, or any local
  `starryzhang/sweb.eval.win.*` image); without one the tab is reported `infra_required`, naming
  the `docker pull` of the first instance's image.
- **On a cloud VM, the metadata server blocked for docker containers.** Every tab runs
  model-controlled code in task containers with network access (SWE-agent's commands in each
  instance container, then the harness's tests over each patch), and a docker container reaches
  the VM's metadata server at `169.254.169.254`, which hands out the VM's service-account token.
  Before the first Linux tab starts, gbench checks that it is unreachable from docker's default
  bridge and from a user-defined bridge network (gbench creates a temporary one,
  `gbench-metadata-probe-<id>`, and removes it right after): a short-lived probe container of
  `gbench-swe-bench-live:a145aa87` (`--pull never`) opens and closes a TCP connection to
  `169.254.169.254:80` and sends nothing. The SWE-agent batches and the harness passes check
  again before they start containers; each network's verdict is cached for the rest of the run,
  so they do not probe again. In the agentic mode the same probe container also resolves the
  hosts SWE-ReX's runtime layer build fetches from (`deb.debian.org`, `archive.ubuntu.com`,
  `www.python.org`, `pypi.org`, `files.pythonhosted.org`; `SWEREX_BUILD_HOSTS`), and a name that
  does not resolve is a hard error too (a DNS failure there would fail every rollout on the
  harness's side). A connection that opens, or a probe without a clear verdict, **hard-errors**
  the whole suite (`infra_required`) naming the fix. Block it once on the host; gbench never runs
  sudo or changes the host's firewall itself:

  ```bash
  sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
  sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save   # keep it
  ```

  See [toolchains.md](toolchains.md#cloud-vms-the-metadata-server).
- **The Windows tab's metadata server check.** Win-Agent's rollouts and the harness's tests run
  model-controlled code in Windows task containers, which both start with no network option, so
  on docker's default network (`nat` on a Windows engine). Before the tab starts
  (`check_windows_metadata_guard`, with `swebench_common.check_windows_task_network`), gbench
  asks the engine for its `OSType` (`docker info`; it must be `windows`) and runs a short-lived
  probe container (`--pull never`, no `--network`) of one of the tab's instance images already
  on the host: `powershell` (else `pwsh`) with the script passed as `-EncodedCommand`, which
  opens and closes a TCP connection to `169.254.169.254:80` with .NET's `TcpClient`, and one to
  `[fd20:ce::254]:80` (the metadata server's IPv6 address, which a container network with IPv6
  reaches on a dual-stack VM) with an IPv6 `TcpClient`, and sends nothing on either. Only a
  connection that opens within 3 s is "reachable", over either address; a timeout, or a refused,
  unreachable or forbidden connection, is "unreachable", and for IPv6 also no IPv6 support or
  address (`AddressFamilyNotSupported`, `AddressNotAvailable`); anything else (neither program
  starts, a docker error or timeout, another socket error) gives no verdict. The tab runs only
  when both are "unreachable" (cached for the rest of the run); otherwise it is reported
  `infra_required` in `settings.windows` with the reason (naming the IPv6 address when that one
  answered). There is no verified Windows host rule gbench can name: block TCP from those
  containers to `169.254.169.254` (and to `fd20:ce::254` when their network has IPv6) on every
  port but 53 (on GCE the VPC DNS resolver answers at the same address) without cutting the VM's
  own access to it, which its guest agent and credentials use (an outbound rule for the address
  on the whole host would cut both); the next run checks again. gbench never changes the host's
  firewall itself.

A selected setting whose prerequisite is missing (Docker, an image, the Windows host, the
Windows tab's metadata server check or the instance image it probes with, or a pinned snapshot
that cannot be fetched or does not match: rows, columns, content sha256) is reported
`infra_required` in its own entry with the reason (and the build command for a missing image) while the other selected settings run (see
"Per-setting prerequisites"). When no selected setting can run, or the metadata server check of
the Linux tabs fails, the suite **hard-errors** (`infra_required`); it never skips.

**Licenses.** `SWE-bench-Live/SWE-bench-Live`, `SWE-bench-Live/MultiLang` and
`SWE-bench-Live/Windows` are tagged `mit` on the Hub; `microsoft/SWE-bench-Live` and
`microsoft/RepoLaunch` are MIT (Copyright (c) Microsoft Corporation); gbench fetches both into
the local image at build time and vendors neither. `njukenanli/SWE-agent-for-eval` is MIT
(its `LICENSE`), fetched into its image the same way. `njukenanli/Win-Agent` has no license file in
its repository; gbench does not ship it (you provide the checkout for the Windows tab).

## Run

```bash
gbench --evals-only --evals swe_bench_live \
    --remote-endpoint http://127.0.0.1:8000/v1 \
    --tokenizer google/gemma-4-26B-A4B-it \
    --max-output-tokens 32768 --sandboxes 8
```

This runs all ten tabs in the leaderboard's order: Lite (300), Full (1,319), Verified (500),
C/C++ (253), C# (136), Java (160), TS/JS (219), Go (138), Rust (171) and Windows (66). On a
Linux host the nine Linux tabs run and the Windows tab is reported `infra_required` in
`settings.windows` (status `completed_with_errors`); `--eval-categories
lite,full,verified,ccpp,csharp,java,tsjs,go,rust` runs the nine without that entry. Add
`--eval-limit 20` for a stratified smoke run (20 instances in total, dealt over the selected tabs
that can run here: with the 9 Linux tabs, 3 + 3 + 2 x 7 in a seeded order; `--eval-limit 5` runs 5
instances, one in each of 5 tabs, and the other tabs do not run; not comparable). Never N per tab:
the run logs its planned workload on one line at its start, e.g. `[swe_bench_live] --eval-limit 5:
5 sample(s) over 5 tab(s) (lite 1, verified 1, ccpp 1, go 1, rust 1) (full, csharp, java, tsjs: 0,
not run) x 1 attempt = 5 rollouts`, and records it as `eval_limit_plan` (each tab's
`leaderboard_comparable_reason` names the suite's limit and its share). A full run (no limit) runs
every tab completely. No separate virtualenv or `--results-dir` is needed.
`--shard I/N` is honored (each setting's instance set is partitioned before `--eval-limit`).
