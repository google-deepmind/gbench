# aider_polyglot

Canonical Aider Polyglot ([Aider-AI/aider](https://github.com/Aider-AI/aider) `benchmark/`):
225 Exercism exercises across 6 languages (C++, Go, Java, JavaScript, Python, Rust). The model
edits the stub in its **native edit format** (diff / SEARCH-REPLACE for capable models); aider
applies the edit, runs the exercise's hidden unit tests, and on failure feeds the errors back for
a second try. **Headline: pass@2.**

## Why it's containerized

Installing aider into the serving env would **break the pinned graph** (it downgrades `openai`
and moves `numpy`/`huggingface_hub`/`pydantic`/`pillow`), and the exercises need 6 language
toolchains. So gbench does **not** install aider into the gbench serving environment; it delegates to aider's **own
benchmark harness inside aider's own benchmark image** (`aider-benchmark:5dc9490b`, built from gbench's
copy of aider's `benchmark/Dockerfile` at the pinned aider commit, which bundles python3.11 +
openjdk-21 + go + rust + node + gcc; see step 1 for what it adds), and only reads back the per-exercise
`.aider.results.json`. gbench hard-errors (never
silently skips) if Docker, the image, or the exercises are missing.

## Setup (step by step)

### 1. Build the image (aider at the pinned commit)

```bash
# The build context is gbench's docker/ directory (it holds aider_polyglot.constraints.txt and
# pinned_pip_check.py); the Dockerfile fetches aider at the pinned commit
# 5dc9490bb35f9729ef2c95d00a19ccd30c26339c itself. It is aider's own benchmark/Dockerfile (all 6
# toolchains + openjdk-21 + aider), PLUS a setuptools-scm version pin, jest pinned to the
# exercises' 29.7.0, Node.js pinned to NodeSource's 20.20.2-1nodesource1 and the npm packages' own
# dependencies resolved as of a fixed time (npm's before).
docker build -t aider-benchmark:5dc9490b \
    -f docker/aider_polyglot.Dockerfile \
    docker
```
The pip and uv installs resolve against `docker/aider_polyglot.constraints.txt` (the pip-managed
packages of the image built 2026-09-28), and the build fails if anything else is installed
(`pinned_pip_check.py`); the image records the Python packages, Debian packages and toolchain
versions it installed in `/opt/gbench` (`pip-freeze.txt`, `dpkg-versions.txt`,
`toolchain-versions.txt`). Node.js is pinned (`nodejs=20.20.2-1nodesource1`, the build the image
built 2026-09-28 got) and the npm packages' own dependencies resolve as of `NPM_CONFIG_BEFORE`
(2026-09-27T05:30:00Z, when that image ran its npm install); Rust's stable toolchain and the Debian
and Ubuntu packages still resolve at build time, as upstream's do.
This is a large image (all toolchains); the first build is slow. Override the tag with
`GBENCH_AIDER_IMAGE` if you build it under a different name. The image carries the label
`org.gbench.aider.commit=5dc9490bb35f9729ef2c95d00a19ccd30c26339c` (`AIDER_COMMIT` in
`aider_polyglot.py`, the `ARG AIDER_COMMIT` of the Dockerfile); gbench checks it before every run
and **hard-errors** (`infra_required`, naming the build command) on an image with another label or
none. benchmark.py records the commit it ran (`commit_hash`, 7 characters) in every result file.

> Why gbench's Dockerfile and not aider's directly: aider's `benchmark/Dockerfile` copies a local
> checkout (whatever commit it is at), while gbench's fetches the pinned commit and labels the
> image with it. The shallow fetch carries no tags, so `uv pip install -e /aider[dev]` could not
> derive a version with setuptools-scm; gbench's copy adds
> `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AIDER_CHAT` to fix exactly that. It also pins
> `jest@29.7.0`: aider's Dockerfile installs the JavaScript exercises' devDependencies at their
> versions but `jest` unpinned, and every JavaScript exercise's `package.json` asks for
> `"jest": "^29.7.0"`, which is what an unpinned install gave until Jest 30 (June 2025). Jest 30
> removed the matcher aliases, so with it the specs that call `toThrowError` (`affine-cipher`,
> `resistor-color-trio`) fail whatever the model wrote (`expect(...).toThrowError is not a
> function`, even with the reference solution). Everything else (openjdk-21, go, rust, node, gcc)
> is identical to aider's.

### 2. Get the polyglot exercises

`benchmark.py` expects them at `$AIDER_BENCHMARK_DIR/polyglot-benchmark`:

```bash
mkdir -p $GBENCH_PREREQS_DIR/aider-bench
git clone https://github.com/Aider-AI/polyglot-benchmark \
    $GBENCH_PREREQS_DIR/aider-bench/polyglot-benchmark
git -C $GBENCH_PREREQS_DIR/aider-bench/polyglot-benchmark \
    checkout 7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f
export GBENCH_AIDER_BENCHMARK_DIR=$GBENCH_PREREQS_DIR/aider-bench
```
This directory is mounted into the container as `/benchmarks` and also receives the run's
timestamped results directory. The checkout must be at the pinned commit
`7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f` (`POLYGLOT_BENCHMARK_COMMIT`; 225 exercises: cpp 26,
go 39, java 47, javascript 49, python 34, rust 30) with no modified tracked file: gbench checks
`git -C <dir> rev-parse HEAD` and `git status` before the run and **hard-errors**
(`infra_required`, naming the checkout command) on a missing `.git`, another commit or a modified
file. The result records `aider_commit` and `polyglot_benchmark_commit`.

### 3. Run

```bash
export GBENCH_AIDER_BENCHMARK_DIR=$GBENCH_PREREQS_DIR/aider-bench
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-26B-A4B-it --evals aider_polyglot \
       --max-output-tokens 32768 \
       --suite-timeout 36000          # the full 225-exercise pass@2 run is long
```

`--max-output-tokens` is required by the gbench command line for every eval run. aider itself
sends no `max_tokens` (aider/models.py `Model.send_completion` adds one only from a model's
settings, and an `openai/<id>` model has none), with and without `--thinking`. gbench's relay puts
the run's `--max-output-tokens` on every request that carries none
(`GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0`: none, as upstream). The result records this in
`agent_max_tokens_from_run`, `max_tokens_sent`, `max_output_tokens_note` and
`model_endpoint_relay.max_tokens_from_run`, and names it in `leaderboard_comparable_reason`. Whether
aider sends one is read from the `extra_params` aider's `Model` reports in the pre-flight: a model
whose aider settings carry a `max_tokens` keeps aider's value, and the relay fills nothing.

A reply cut at the cap ends with `finish_reason: length`, and aider counts it as exhausted: `aider/coders/base_coder.py` `send_message` marks the reply exhausted on
`finish_reason: length` (for a model without assistant prefill, as an `openai/<id>` model is) or
on litellm's `ContextWindowExceededError` (a request the window refused), and adds one to
`num_exhausted_context_windows`. `benchmark.py` writes that into each `.aider.results.json`, and
its `summarize_results` sums it into the `exhausted_context_windows` column of the leaderboard
(`aider/website/_data/polyglot_leaderboard.yml`). gbench reads the same counter:
`exhausted_context_windows` (also reported as `replies_cut_off_by_budget`) is its sum, and any
exhausted reply makes the run not comparable. With the default cap, such a reply stopped at the
run's `--max-output-tokens` (raise it to give replies more room). With
`GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0` it stopped at the end of the served window (serve a larger
`--max-model-len`), or the window refused the request.

Internally gbench `docker run`s the image on docker's bridge network, points aider via litellm
(`OPENAI_API_BASE` + `--model openai/<id>`) at gbench's relay on that network's gateway, which
forwards each request to the endpoint from the gbench process (so an endpoint such as
`127.0.0.1:8000` works; see "Container and network" below),
and runs `benchmark.py --tries 2 --exercises-dir polyglot-benchmark --new` across the 6
languages. It reads the resulting `.aider.results.json` files and reports **`accuracy` = pass@2**
(with `pass_rate_1` and per-language `category_accuracy`).

Add `--thinking` to measure the thinking model (see "Thinking" below); the full run is long
either way, so keep a generous `--suite-timeout`.

## Container and network

The suite starts one container of the `aider-benchmark:5dc9490b` image, where aider sends the model's
requests and runs each exercise's tests over the code the model wrote. It needs the endpoint, so
it cannot run with `--network none`; it runs on docker's bridge network with
`--add-host host.docker.internal:host-gateway`, as upstream's `benchmark/docker.sh` runs the same
image, and never on the host network. aider's `OPENAI_API_BASE` is always gbench's relay on the
bridge gateway (e.g. `http://172.17.0.1:<relay port>/...`), never the endpoint URL itself: the
relay forwards each request to the endpoint from the gbench process on this host, so an endpoint
on this host may listen on `127.0.0.1` only. The model id is the served
id gbench's eval runner resolved against `--models` (sent as `openai/<id>`, recorded as
`aider_model` and `request_model_id` with `request_model_id_source`); the suite does not re-read
the endpoint's model list unless `GBENCH_AIDER_MODEL` names another id. The container never gets the
endpoint's key. It always sends aider's requests to gbench's relay
(`swebench_common.EndpointRelay`, a thread of the gbench process on the docker bridge gateway),
with or without a key (since 2026-10-11). The relay adds the key when there is one
(`VLLM_API_KEY`, else `OPENAI_API_KEY`) and the run's `--max-output-tokens` on requests without
`max_tokens` (`model_endpoint_relay` on the result). The container gets `OPENAI_API_KEY=EMPTY`. gbench does not change the
container's `HOME`: it stays the image's `/root`, as upstream's `benchmark/docker.sh` leaves it.
The image installs Rust with rustup under `/root/.rustup` and `/root/.cargo` and sets neither
`RUSTUP_HOME` nor `CARGO_HOME`, so with any other `HOME` cargo finds no default toolchain
(`error: rustup could not choose a version of cargo to run`) and every Rust exercise's tests fail.
Gradle and cargo downloads are cached under `/root` and go away with the container, as upstream's
do. The container has upstream's memory limit: `docker run --memory 12g --memory-swap 12g`, the
`--memory=12g --memory-swap=12g` of `benchmark/docker.sh` (12 GiB, no swap on top). The exercises'
tests run the model's code in this container, so code that allocates without bound is killed by
the kernel inside the container (its test process ends with SIGKILL, the test run fails and
`benchmark.py` feeds that back as it does upstream: the model's outcome) instead of taking memory
from this host and the model server on it. The limit covers the whole container, so the
exercises `benchmark.py` runs at once share it. gbench passes its eval concurrency as `--threads`
(`--sandboxes` when given, else the first `--batch-sizes` value); upstream's documented run
(`benchmark/README.md`) uses `--threads 10` in the `docker.sh` container. The result records the
thread count with the limit (`container_memory_limit`), and the limit the kernel actually
enforced on the container's cgroup (`container_memory_limit.enforced_bytes`: cgroup v2
`memory.max`, else v1 `memory.limit_in_bytes`, printed on a `CONTAINER_MEMORY_LIMIT <value>` line
after `benchmark.py` exits; `"max"` means no limit, `null` that neither file could be read).
Docker drops `--memory` with a warning on a host without a memory cgroup controller, so a limit
other than 12 GiB (12884901888 bytes), or one that could not be read, makes the run not
comparable and is named in `leaderboard_comparable_reason`.

At the limit the kernel's OOM killer kills one process in the container (docker leaves
`memory.oom.group` at 0), the one using the most memory: usually an exercise's test or build
process, whose test run then fails and is scored as the model's outcome, as upstream scores it.
Because the limit is shared, a kill cannot be attributed to the exercise whose code used the
memory: one C++ exercise's test build (catch2) alone peaks at about 0.65 GB of the container's
memory (cgroup `memory.peak`, page cache included, measured in this image for `all-your-base`
with its example solution), and every Java exercise
runs `./gradlew test` with the Gradle daemon on, which stays up after its build. So after
`benchmark.py` exits, whatever its exit status, the container prints the number of OOM kills in
its cgroup as its last line, after an empty one (`GBENCH_OOM_KILLS <n>`, the `oom_kill` count of cgroup v2
`memory.events`, or of cgroup v1 `memory.oom_control`), and keeps `benchmark.py`'s exit status.
The result records it as `container_oom_kills`. Any kill makes the run not comparable, and so
does a count gbench could not read (`container_oom_kills: null`: no such line at the end of the
container's output, because its main process was killed or neither cgroup file was readable);
`leaderboard_comparable_reason` gives the count (or says it could not be read) and the thread
count, and says when the threads were more than upstream's 10. The score itself is kept as upstream computes it. A container that
exits 137 (its main process killed with SIGKILL) is named as such in the hard error, with the OOM
kills it reported. Before anything else runs, the same
container checks that the image runs the exercises as upstream's does: the JavaScript exercises' `jest` (`/npm-install`) is 29.x and `cargo --version`
works. If not, `benchmark.py` does not start and the suite hard-errors (`infra_required`) naming
what failed (for example `jest=30.5.1` in an image built before the pin) and the rebuild command. On a cloud VM the container relies
on the host rule that keeps a bridge network away from the metadata server at `169.254.169.254`
(which hands out the VM's service-account token):

```bash
sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save   # keep it
```

Before the container starts, gbench's guard (`require_metadata_blocked`, see
[toolchains](toolchains.md#cloud-vms-the-metadata-server)) starts one short-lived container of the
image on the bridge network that opens and closes a TCP connection to `169.254.169.254:80` (no
request is sent); if it opens, the suite hard-errors (`infra_required`) naming the rule, and a
probe that gives no clear answer (neither `python3` nor `bash` in the image, a docker error or
timeout) is a hard error too. No TCP connection to the endpoint or to the relay is opened from the
container before the run (the guard opens one only for a loopback endpoint URL it rewrote to
`host.docker.internal`, and aider is given the relay's gateway URL instead), so a gateway the
bridge cannot reach shows up as failed model requests. gbench never runs sudo or changes the
host's firewall itself.

The exercises' tests fetch from the internet at run time, as in upstream's image, which bakes in
no Gradle distribution, Maven package or crate: every Java exercise's Gradle wrapper downloads
Gradle 8.7 from `services.gradle.org` and its `build.gradle` takes JUnit and AssertJ from Maven
Central (`repo.maven.apache.org`), and three Rust exercises (`gigasecond`, `grep`,
`simple-cipher`) fetch crates through crates.io's index (`index.crates.io`) and download host
(`static.crates.io`). C++, Go, JavaScript (the image's preinstalled `/npm-install`) and Python fetch
nothing. So the container needs working DNS on the bridge network: the same probe container
resolves the hosts of the run's languages (`GBENCH_AIDER_LANGUAGES`) and the endpoint's host when
the URL names one (not an IP address), and a name that does not resolve stops the suite before
any exercise runs (`infra_required`), naming the rule. On GCE the VPC DNS resolver is
`169.254.169.254` itself, which is why the rule spares port 53 (a plain `-j DROP` for the address
cuts DNS in every container).

A harness failure during the run is a harness error, never the model's outcome. After the run,
gbench reads each exercise's chat history (`.aider.chat.history.md`, next to its
`.aider.results.json`: the instructions, the model's replies, aider's messages and every test
run's output). A test run's output is the bare fenced block that `benchmark.py`'s
`run_unit_tests` appends right after aider's own lines for the turn (aider writes each as `> `,
the message and two trailing spaces); the model's replies (which aider writes as they came,
after a blank line, so a blockquote line the model wrote does not end in two spaces) and the
instructions (`#### ` lines) are not read for the first two kinds. An exercise is a harness error when:

- a test run's output shows that a language toolchain in the image could not run at all (rustup
  with no default toolchain: `rustup could not choose a version of ... to run`, or the older
  `error: no default toolchain configured`), kind `toolchain`;
- a test run's output shows a failed host-name lookup or connection for a toolchain host (Gradle,
  Maven Central, crates.io, or GitHub, where Gradle's download redirects), kind
  `toolchain_network`;
- aider gave up on a model request because the endpoint kept failing, kind `model_endpoint`.
  aider retries a failed request with a delay that starts at 0.125 s and doubles, and stops
  retrying once the next delay would pass `RETRY_TIMEOUT`; `benchmark.py` raises that to 24 h
  ("Don't give up when benchmarking"), so the retries go on past `Retrying in 32.0 seconds...`
  (64, 128, ... up to 65536 s, about 36 h in all). gbench reads each request's last retry line
  (no later retry line before a reply, aider's `Tokens:` line or the next test run): the attempt
  after it either succeeds (the model's reply and aider's `Tokens:` line follow, and the exercise
  is scored as usual) or does not, and the turn ends with no reply (aider logged the final failure,
  or the run ended while it was still retrying). It is `model_endpoint` only when there is no reply
  and the failure aider logged (the last one after that line, else the one just before it) is an
  endpoint failure:
  `litellm.APIConnectionError` (no connection), `litellm.InternalServerError` (HTTP 500),
  `litellm.BadGatewayError` (502), `litellm.ServiceUnavailableError` (503),
  `litellm.RateLimitError` (429), `litellm.Timeout` (see [Request timeout](#request-timeout)), or
  a logged failed host-name lookup or connection.

Such an exercise is left out of every figure (pass rates, per-language counts and
`aider_leaderboard_columns`, except `total_tests`, which counts the run's exercise directories),
counted and named in the result, and the status is `completed_with_errors`; if every exercise
aider finished is one, no score is reported (`infra_required`). The full-run check still requires
all 225 exercises to finish.

A request aider gave up on for any other cause is the model's outcome and is scored, as upstream
scores it. Those give-ups are counted in `scored_request_give_ups`, one per request aider gave up
on (an exercise that gave up in both tries counts twice), and listed with their exercise in
`scored_request_give_up_exercises`; they do not change the status or comparability.

### Request timeout

aider gives each request 600 s (`aider/models.py` `request_timeout`; litellm's openai client
makes up to 3 attempts of it before aider sees the failure) and sends no `max_tokens` (see
[Configuration](#configuration)). A reply longer than 600 s timed out. At temperature 0 (aider's
own, and gbench's without `--thinking`) every retry produced the same reply and timed out again, and `benchmark.py`
keeps retrying for 24 hours: the run hung instead of scoring the reply. gbench sets each request's
timeout as it does for every request without `max_tokens`: `GBENCH_REQUEST_TIMEOUT_S` (1200 s)
plus the served context window at `GBENCH_MIN_DECODE_TOK_S` (8 tokens/s), or a 262,144-token
window when the server reports none (33,968 s for a 256K window). It goes into the settings file's
`extra_params` as `timeout`, which aider uses instead of its default, together with
`max_retries: 0`, so each of aider's attempts is a single HTTP attempt and aider's own retry loop
does the retrying. litellm consumes both; neither is sent to the endpoint. The timeout stays sized for the served
window. gbench's relay adds `max_tokens` = the run's `--max-output-tokens` (with
`GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0`, none), so a long reply ends at that cap (or at the window
with the cap off), and aider counts it in `exhausted_context_windows`. A request that still times out means the endpoint decoded slower
than that rate, so it is an endpoint failure (`model_endpoint`). The result records the value in
`model_request_timeout_s`.

The container has general internet access on the bridge network, as upstream's
`benchmark/docker.sh` runs it; only the metadata server is blocked, so the model's code in the
tests can reach the internet as it can upstream.

## Thinking

`--thinking` reaches aider's requests. gbench writes a model-settings file, mounts it read-only
at `/gbench-aider/model-settings.yml` and passes `--read-model-settings` to `benchmark.py`:

```yaml
[{"name": "aider/extra_params",
  "extra_params": {"extra_body": {"chat_template_kwargs": {"enable_thinking": true}},
                   "timeout": 33968, "max_retries": 0, "temperature": 1.0}}]
```

(`timeout` is the request timeout, see [Request timeout](#request-timeout); `temperature` is the
run's temperature, see [Sampling](#notes--caveats).)

`aider/extra_params` is aider's documented global override: only its `extra_params` are used,
deep-merged into every litellm call, so the model's own settings (native edit format) are
unchanged. Without `--thinking` the same file sends `enable_thinking: false` and `temperature: 0.0`,
so a no-think run does not depend on the chat template's default.

aider's `register_models` skips a settings file it cannot read without any error, so gbench
confirms the file was applied at run time. In the same container, just before `benchmark.py`, it
runs a pre-flight (`python3 /gbench-aider/check_settings.py`, written next to the settings file)
that loads the file with aider's own `models.register_models` (what `--read-model-settings`
calls) and prints the `extra_params` aider's `Model` ends up with for the model under test. If
the file was not loaded, `benchmark.py` does not start, and gbench raises unless the reported
`extra_params` contain its `enable_thinking` value and its `timeout`. The result records `thinking`,
`model_settings_confirmed: true` and `model_extra_params` (the `extra_params` aider reported,
merged into every request it sends). The served endpoint should split the reasoning into
`reasoning_content` (vLLM's reasoning parser); reasoning left inline in the content would reach
aider's edit parser.

## Configuration

| Env var | Purpose |
| --- | --- |
| `GBENCH_AIDER_IMAGE` | image tag (default `aider-benchmark:5dc9490b`) |
| `GBENCH_AIDER_BENCHMARK_DIR` | host dir with `polyglot-benchmark/` (mounted to `/benchmarks`), **required** |
| `GBENCH_AIDER_MODEL` | the model id litellm uses (default `openai/<served id gbench passed>`; litellm routes `openai/*` to `OPENAI_API_BASE`). Another value must be `openai/<id>` for an id the endpoint's `/v1/models` lists (checked before the run; anything else is `infra_required`, since another litellm provider would send the requests elsewhere); recorded as `request_model_id` / `request_model_id_source` |
| `GBENCH_AIDER_EDIT_FORMAT` | force an edit format (default: aider's native choice for the model) |
| `GBENCH_AIDER_LANGUAGES` | restrict to a subset of the 6 (comma/space list, any case: gbench lowercases it, as `benchmark.py` does) |
| `GBENCH_AIDER_EXTRA_ARGS` | extra flags appended to `benchmark.py` (must not include `--read-model-settings` or `--num-tests`, which gbench owns; the suite raises if it does) |

## Notes / caveats

- **`leaderboard_comparable`** is set only for the **full** 225-exercise set (no `--eval-limit`,
  no partial `--shard`) across all 6 languages with aider's **native** edit format, no exhausted
  context window, no harness error and a container that reported 0 OOM kills
  (`container_oom_kills`), under the 12 GiB limit it enforced (`container_memory_limit.enforced_bytes`),
  and aider's temperature 0 (gbench's without `--thinking`; a `--thinking` run samples at 1.0, so
  it is not comparable).
- **Per-exercise evidence:** `aider_run_dir` is the run directory `benchmark.py` wrote under
  `$GBENCH_AIDER_BENCHMARK_DIR` (`<timestamp>--gbench`), with each exercise's
  `.aider.results.json` and `.aider.chat.history.md` (the model's replies, aider's own lines and
  every test run's output). The container runs as the image's root (its `HOME`, `/root`, holds the
  Rust toolchain); after `benchmark.py` exits it hands that directory to the host user running
  gbench (`chown -R <uid>:<gid>`), so the files stay readable and removable without root.
- **Exercise selection (`--eval-limit`, `--shard I/N`).** gbench selects the exercises on the host,
  never with `benchmark.py --num-tests` (the first N after an unseeded `random.shuffle`, a different
  subset every run). It lists the run's languages' exercise directories
  (`polyglot-benchmark/<lang>/exercises/practice/<name>`) in sorted order, takes the shard first
  (exercise I, I+N, ... of that order), then draws `--eval-limit` exercises from it (N for the whole
  suite, never N per language; The run logs its plan on its first line and records it as `eval_limit_plan`), stratified by
  language with a fixed seed (`sampling.stratified_sample`, seed `aider_polyglot`). A subset is
  copied into a temporary exercises tree with the checkout's layout, mounted read-only at
  `/gbench-exercises` and passed as `--exercises-dir`, so `benchmark.py` runs exactly those (its
  shuffle only orders them); every selected exercise must finish, as a full run must finish all
  225. The result records `selected_exercises`, the shard identity (`shard`, `shard_index`,
  `shard_count`, `shard_applied`), `sharded run (--shard I/N) ...` in
  `leaderboard_comparable_reason` for a partial shard, and one trace per exercise in
  `sample_traces` (`exercise` as `<lang>/<name>`, `language`, `pass_1`, `pass_2`, `tries`,
  `request_give_ups`, a harness error with `scoring_excluded`, `sample_key`, `global_idx`). An
  empty shard runs nothing and reports `accuracy: null` with `accuracy_withheld_reason`.
  `gbench --merge` re-counts pass@1 and pass@2 over the union of the shards' traces (the suite's
  `merge_shards`).
- **Runtime:** the full pass@2 run generates + compiles + tests thousands of files. It is slow.
  Raise `--suite-timeout`.
- **Sampling:** gbench's rule for every suite (`base.resolve_temperature`): **temperature 0.0**
  without `--thinking`, **1.0** with it; `GBENCH_AIDER_POLYGLOT_TEMPERATURE`, then `--temperature`,
  replaces it in either mode. aider's benchmark sets its own temperature 0 on every request
  (aider/models.py `Model.send_completion`, `use_temperature` for an `openai/<id>` model), so
  gbench always writes the run's value into the settings file's `extra_params` as `temperature`:
  `Model.send_completion` sets its own temperature and then applies
  `kwargs.update(self.extra_params)`, so the value reaches every litellm call in both modes, and
  the pre-flight confirms aider loaded it (`model_extra_params`). The result records
  `temperature`, `temperature_source` (`default`, `thinking`, or the operator's),
  `temperature_applied: true` and `official_sampling` (aider's temperature 0,
  `temperature_matches_official`); a temperature other than 0 makes the run not
  leaderboard-comparable (`leaderboard_comparable_reason` names it). There is no LLM judge;
  grading is the exercises' unit tests.
- **Attempts:** aider's benchmark runs each exercise once with `--tries 2` (pass@2 is its
  protocol); `--attempt-count` above 1 is recorded as not applied (`attempts.not_applied`).
- **Result keys:** `accuracy` (pass@2), `pass_rate_1`, `pass_rate_2`, `category_accuracy` (per
  language, pass@2), `thinking`, `model_extra_params`, `model_settings_confirmed`,
  `model_request_timeout_s`, `edit_format`,
  `max_output_tokens_note`, `aider_run_dir`, `container_memory_limit` (`memory` and
  `memory_swap`, `12g` as in upstream's `benchmark/docker.sh`, the `threads` that shared it and
  the `enforced_bytes` the kernel applied), `container_oom_kills`
  (processes the kernel's OOM killer killed in that container, or `null` when the count could not
  be read), `exhausted_context_windows`
  and `replies_cut_off_by_budget` (aider's
  `num_exhausted_context_windows`, summed), `aider_leaderboard_columns`, `harness_errors` (the
  number of exercises left out as harness errors), `harness_error_kinds` (that number per kind,
  `toolchain`, `toolchain_network` or `model_endpoint`), `harness_error_exercises` (each one's
  `exercise`, `language`, `kind` and the `detail` line from its chat history),
  `scored_request_give_ups` and `scored_request_give_up_exercises` (the requests aider gave up on
  in scored exercises for a cause other than an endpoint failure, one entry per request: its
  `exercise`, `language`, `cause`, such as `Timeout`, and `detail`), `status` (`success`, or
  `completed_with_errors` with any harness error), `leaderboard_comparable` (with
  `leaderboard_comparable_reason` when it is `false`).
- **`aider_leaderboard_columns`:** the columns aider's `benchmark.py` `summarize_results` prints
  for a run and the leaderboard carries per row, computed the same way over the exercises gbench
  scored: `test_cases`, `total_tests` (exercise directories in the run), `pass_rate_1`,
  `pass_rate_2` (one decimal), `pass_num_1`, `pass_num_2`, `percent_cases_well_formed`
  (100 x (1 - exercises with a malformed response / test cases)), `num_with_malformed_responses`,
  and the sums of the per-exercise counters: `error_outputs`, `num_malformed_responses`,
  `user_asks`, `lazy_comments`, `syntax_errors`, `indentation_errors`,
  `exhausted_context_windows`, `test_timeouts`, `prompt_tokens`, `completion_tokens`; then
  `seconds_per_case`, `total_cost` and the `model`, `edit_format` and `commit_hash` values seen
  (comma-joined). A result file that holds only an exception (aider's `run_test` wrote it) is
  not scored; a full run with one fails the 225-exercise check instead.
