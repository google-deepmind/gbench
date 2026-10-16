# quality (GemmaClaw QA suite)

`--quality` / `--quality-only` runs GemmaClaw's QA suite against the model under test:
`pnpm openclaw qa suite --provider-mode live-frontier --model vllm/<model> --alt-model
google/gemini-3.1-pro-preview --output-dir gbench_qa_out` (plus `--scenario <id>` for each selected
scenario), in gbench's local Docker container built from a GemmaClaw commit (the pinned
`DEFAULT_GEMMACLAW_COMMIT` in `gbench/core/config.py`, `--gemmaclaw-commit` to change it,
`--gemmaclaw-path` for a checkout of your own). The model under test is the suite's primary model,
served by the native vLLM server the CLI starts for a local run (the same server the Golden Set
and eval pillars use) or by `--remote-endpoint`, through a throwaway `openclaw.json`
(`OPENCLAW_CONFIG_PATH`) that declares it as the `vllm` provider with the served context window
and GemmaClaw's own output cap, and with image input when the model takes images (below). The
context window is the endpoint's `/v1/models` `max_model_len`, else `--max-model-len`, else the
model's configured context length; when none is known (the endpoint lists none and the model's
config.json was not read) the pillar stops with `infra_required` rather than declare a
placeholder. `maxTokens` is GemmaClaw's `SELF_HOSTED_DEFAULT_MAX_TOKENS` read from the commit that
runs (8192 at the pinned commit), and a served window no larger than that cap is refused before
the suite runs (infra_required: the window cannot hold the declared output cap plus a prompt;
serve a larger --max-model-len); `--max-output-tokens` (the
eval pillar's flag) is not applied, and the result says so in `max_output_tokens_note`. The result
records `served_model_id`, `served_model_id_source`, `context_window`, `context_window_source`,
`model_max_tokens` and `model_max_tokens_source`. The result's
`pass_rate` is `passed / total` over the scenarios the model under test can be scored on, in
percent: not those that need another provider, and not those a harness error fell in (below).
When no scenario is left to score, `pass_rate` is withheld (`null`), never `0`.

The runner is `gbench/runners/quality.py`; the image is `docker/quality.Dockerfile` with its
helpers `docker/quality_entrypoint.sh`, `docker/quality_relay_forwarder.py` and
`docker/quality_npm_cache.mjs`.

## External models: the Gemini API only

Besides the model under test, the live-frontier lane calls OpenAI models (its `openai` plugin,
keyed). gbench runs each of them on the Gemini API through GemmaClaw's `google` plugin instead.
There is no option for another model or provider.

| Capability | GemmaClaw live-frontier default | gbench runs | How |
| --- | --- | --- | --- |
| alternate model (model-switch scenarios) | `openai/gpt-5.4` | `google/gemini-3.1-pro-preview` | `--alt-model` |
| image-generation model (the media scenarios' `ensureImageGenerationConfigured`) | `openai/gpt-image-1` | `google/gemini-3.1-flash-image-preview` | an edit of the checkout (below) |
| memory embeddings (`memory_search`, memory sync) | `openai/text-embedding-3-small` | `google/gemini-embedding-001` | the memorySearch provider is auto-selected among the keyed ones (openai before gemini); the google plugin is the keyed one |
| image understanding (the `image` tool), only for a model under test without image input | `openai/gpt-5.4-mini` | `google/gemini-3-flash-preview` | the image tool falls back to a keyed media provider (openai before google) |

- The Gemini models are GemmaClaw's own Google defaults at the pinned commit:
  `GOOGLE_GEMINI_DEFAULT_MODEL` (`extensions/google/onboard.ts`), `DEFAULT_GOOGLE_IMAGE_MODEL`
  (`extensions/google/image-generation-provider.ts`), `DEFAULT_GEMINI_EMBEDDING_MODEL`
  (`extensions/google/embedding-provider.ts`) and the plugin's default image model
  (`extensions/google/openclaw.plugin.json`). The OpenAI ones are the openai plugin's
  (`DEFAULT_OPENAI_EMBEDDING_MODEL`, and its default image model in its `openclaw.plugin.json`).
- A model under test that takes images (`supports_multimodal`) is declared with
  `"input": ["text", "image"]`, so image attachments reach it as images and the image tool uses
  it, as they do the lane's own `openai/gpt-5.4`, and no helper model looks at an image for it. A
  text-only model under test is declared with `["text"]`: its image tool gets the Gemini fallback
  of the last row.
- The google plugin also registers a web-search provider, and speech, music and video generation.
  web search has no key: the Gemini web-search provider reads only `GEMINI_API_KEY` or its own
  plugin setting, and the suite gets neither, so the model under test gets no `web_search` tool
  (the lane has none for a model that is not an OpenAI one either). No scenario configures
  speech, music or video generation; the relay forwards none of their calls (below) and records
  any it refuses.
- Every Gemini API call the suite makes, whatever its capability, is counted in `gemini_calls`.

## The container

The suite (the QA lab, its gateways, and every tool the model under test runs, `exec` included)
runs only in a Docker container of gbench's own image. It always does: `GBENCH_SANDBOX` (the
setting of the suites that execute model-written code) does not apply here, and there is no mode
without the container.

### The image

gbench builds the image itself when it is missing, before the run (this takes several minutes the
first time): `docker build -t gbench-quality:<commit[:12]>-<digest[:12]> --build-arg
GEMMACLAW_COMMIT=<commit> --build-arg MIN_NODE=22.19 -f <context>/quality.Dockerfile <context>`,
where `<digest>` is a sha256 over the Dockerfile, its three helpers and gbench's edits, so a
change to any of them builds a new image. The build context is a temporary directory:

- `src/`: `git archive <commit>` of the checkout (the commit's tree only: no working-tree change,
  no `.git`), with gbench's two edits applied (below). Before the edits, every file of it is
  searched for the Gemini keys and `VLLM_API_KEY`: a match stops the setup with `infra_required:
  <path> at GemmaClaw commit <commit> holds a Gemini key or the endpoint's key, and the quality
  image would show it to the model under test; remove it from that commit` (the path only, never
  the value), and the run fails with `Failed to set up gemmaclaw repository: <that message>`.
- `gemmaclaw_patches.json` (the edit records), the Dockerfile and its helpers.

The image is based on the image GemmaClaw's own Dockerfile pins at the default commit
(`node:24-bookworm` by digest), and holds:

- `/opt/gemmaclaw`: the checkout, installed (`pnpm install --frozen-lockfile`) and built the way
  GemmaClaw's Dockerfile builds it (`pnpm build`, `pnpm ui:build`, `pnpm qa:lab:build`), then
  `pnpm openclaw qa suite --help`, so the private-QA build and runtime sync that GemmaClaw's
  `scripts/run-node.mjs` does on first use happen in the image. It is owned by root. Its `.git`
  is an empty directory: GemmaClaw's private QA CLI and its bundled-plugin loader run only from a
  source checkout, which they recognize by a `.git` entry, as in a clone; git finds no repository
  there, so run-node compares the build stamp with source mtimes, and `GIT_COMMIT` (set to the
  commit) is what GemmaClaw's version line and build info show. The last build step runs the
  suite's first step as the suite's user and fails the build if the checkout would rebuild or
  resync at run time.
- `/opt/gbench/npm-cache`: the runtime dependencies of every bundled plugin. GemmaClaw installs a
  plugin's `dependencies` and `optionalDependencies` with `npm install` when it loads the plugin
  (`src/plugins/bundled-runtime-deps.ts`); the container has no network, so
  `quality_npm_cache.mjs` fetches all of them at build time into this cache, and checks that an
  `--offline` install from the cache alone succeeds. The run installs from it offline.
- Google Chrome (the Control UI scenario opens it through Playwright's `channel: "chrome"`),
  installed with the checkout's own playwright-core (`install --with-deps chrome`, the current
  stable Chrome at build time), and the packages GemmaClaw's runtime image installs (`procps`,
  `hostname`, `curl`, `git`, `lsof`, `openssl`), plus `python3` for the forwarder.
- `/opt/gbench`: the entrypoint, the forwarder, the edit records, the commit, and a metadata bundle
  (the QA scenarios and the plugin manifests). Before a run gbench reads these from a container
  created (never started) from the image: the scenarios and manifests for the scenario and plugin
  lists below, the edit records for `gemmaclaw_patches`.
- The users `qa` (uid 10001), which runs the suite, and `gbrelay` (uid 10002), which runs the
  forwarder.

### The run

One container per run, started with
`docker run --name gbench-quality-<random> --interactive --init --network none --dns-search .
--read-only --tmpfs /tmp:rw,nosuid,nodev,mode=1777
--tmpfs /home/qa:rw,nosuid,nodev,uid=10001,gid=10001,mode=0700 --shm-size 1g
--volume /opt/gemmaclaw/gbench_qa_out --volume /opt/gemmaclaw/.artifacts --volume /gbench-status
--volume /opt/gbench/npm-cache
--volume <relay dir>:/run/gbench-relay:ro --volume <openclaw.json>:/etc/gbench/openclaw.json:ro
--cap-drop ALL --cap-add SETUID --cap-add SETGID --security-opt no-new-privileges
--pids-limit 8192 --label gbench.quality.run=1 --env ... <image> pnpm openclaw qa suite ...`:

- `--network none`: the container has only a loopback interface. No host, no cloud metadata
  server (`169.254.169.254`, `metadata.google.internal`), no registry and no web is reachable from
  it. Its one way out is the relay's unix socket (below). gbench calls its cloud metadata server
  guard (`container_guard.require_metadata_blocked`) with the network of this very command line
  before the container starts; `--network none` needs no probe, so the guard starts no container
  and contacts nothing, and the result records its verdict (`no-network`) in
  `sandbox.metadata_guard`. The quality run needs no host firewall rule. Should the container
  ever get a network, the guard probes it and a reachable metadata server stops the run with
  `infra_required` naming the host rule (see "Cloud VMs: the metadata server" in
  `docs/evals/toolchains.md`). The image build (`docker build`) has a network, but runs only
  GemmaClaw's install and build steps, never the model under test.
- `--read-only`: the root file system, the checkout included, cannot be written. The suite writes
  to `/tmp` and `/home/qa` (tmpfs, the fresh home), and to four anonymous volumes that are
  removed with the container: the suite's output directory `/opt/gemmaclaw/gbench_qa_out`, the
  checkout's `.artifacts` (where the suite stages each gateway's bundled-plugin tree,
  `.artifacts/qa-runtime/...`), gbench's status directory `/gbench-status` (the entrypoint's and
  the forwarder's; user `qa` can neither read nor write it) and a per-container copy of the npm
  cache.
- It has its own process namespace: in `/proc` it sees only its own processes, not gbench or any
  other process of the host.
- The only host paths it sees are the relay's socket directory and gbench's `openclaw.json`, both
  read-only. Not the host's home (`~/.bashrc`, gcloud credentials), not its processes, not `/run`
  (agent sockets), not the Docker socket, not any checkout, and no pnpm store or other host
  directory is writable.
- The entrypoint (`quality_entrypoint.sh`) starts as root with only `CAP_SETUID` and
  `CAP_SETGID`. It reads the relay token from the first line of its standard input (gbench writes
  it there; it is never on a command line, in an environment or in a file), starts the forwarder
  as user `gbrelay` with the token on a pipe, waits until it listens, and runs its arguments in
  `/opt/gemmaclaw` as user `qa`. A process whose user changes from root drops every capability,
  and `no-new-privileges` keeps it from gaining any back: the suite and every tool the model
  under test runs hold none, run as a user other than the forwarder's, and so cannot read the
  forwarder's memory, environment or file descriptors.
- The container's environment is a fixed set of variables (below); nothing of gbench's
  environment is passed.
- The suite writes its summaries where the model's tools can write too (its output directory
  belongs to user `qa`), as in GemmaClaw's own runs; gbench only reads them as data.
- After the suite ends, gbench reads the suite summary (`gbench_qa_out/qa-suite-summary.json`),
  each scenario's own summary (`gbench_qa_out/scenarios/<id>/qa-suite-summary.json`) and its
  status files from the stopped container with `docker cp` into memory: regular files of those
  names only (links and other files are skipped, a file over 64 MiB is left out), parsed as JSON.
  Nothing the container wrote is extracted to the host, and gbench never runs anything from it.
  Then it removes the container and its volumes (`docker rm -f -v`).
- The entrypoint exits with 94 to 98 when the container fails before the suite runs (no
  `/opt/gemmaclaw`, no command, not started as root, no relay token, the forwarder did not
  start); these, a `docker run` failure (125) and an entrypoint that cannot run (126, 127) fail
  the run with `the quality container failed before the suite ran: <why>`.
- A suite that has printed its summary line (`QA suite summary: ...`) but has not exited
  `SUITE_EXIT_GRACE_S` (120 s) later (a suite whose isolated worker failed can leave its lab
  servers open) is stopped (`docker stop --time 10`); its summary is read as usual, and the result
  records `suite_stopped_after_summary: true` and the container's exit status in
  `suite_exit_code`.
- A run killed by SIGTERM may leave its stopped container behind; `docker ps -a --filter
  label=gbench.quality.run=1` lists such containers, and `docker rm -f -v <name>` removes one.

tests/test_quality_isolation.py proves this from inside a container started with the runner's own
`docker run` command line, entrypoint and forwarder: as user `qa`, no Gemini key, endpoint key or
relay token is in its environment, in any process environment it can read, or in any file; the
forwarder's `environ`, `mem`, `maps` and file descriptors cannot be read; `169.254.169.254` is
unreachable (`ENETUNREACH`), `metadata.google.internal` does not resolve, and the only network
interface is `lo`; the relay socket refuses a request without the token; writes to the checkout,
the relay directory and `/usr` fail with `EROFS`.

## The key relay

Every credential of a run stays in gbench's process, with a relay that listens on a unix socket
in a temporary directory of the host (directory mode 0711, socket mode 0666), mounted read-only
into the container at `/run/gbench-relay`. Inside the container, the forwarder listens on
`127.0.0.1:18431` and hands every request to the relay over that socket, adding the per-run relay
token in the `x-gbench-relay-token` header. The suite is configured with the forwarder's base URLs
and the placeholder `gbench-relay` where a key would go:

- `models.providers.google` in `openclaw.json` is `{"baseUrl": "http://127.0.0.1:18431/v1beta",
  "apiKey": "gbench-relay", "api": "google-generative-ai", "request": {"allowPrivateNetwork":
  true}, "models": [<the gemini-3.1-pro-preview catalog entry>]}` (GemmaClaw's model-request guard
  refuses the loopback address unless the provider allows a private network), and
  `GBENCH_GEMINI_RELAY_BASE_URL` is set to the same base URL for the second edit below.
- `models.providers.vllm` has `"baseUrl": "http://127.0.0.1:18431/model"` and `"apiKey":
  "gbench-relay"`, for gbench's own vLLM server as for `--remote-endpoint`, so the model
  endpoint's failures are counted too.

The relay:

- answers a request without the token (in `x-gbench-relay-token`; the token in a key header,
  a Bearer token or `key=` opens nothing) with 401 and counts it in `relay_rejected_requests`,
  not as a harness error;
- forwards to `https://generativelanguage.googleapis.com` only (no variable changes it), over
  HTTPS with certificate checks, with the real key in `x-goog-api-key`: `GEMINI_API_KEYS` when
  set, else `GEMINI_API_KEY`, the keys taken in turn and the next one tried on a 401, 403 or 429
  answer. It forwards only the Gemini API calls the suite makes: `POST
  /v1beta/models/<model>:<method>` with the model one of `gemini-3.1-pro-preview`,
  `gemini-3.1-flash-image-preview`, `gemini-embedding-001` and `gemini-3-flash-preview`, and the
  method one of `generateContent`, `streamGenerateContent`, `embedContent` and
  `batchEmbedContents`; the query reduced to `alt`, and only the request headers `content-type`,
  `accept`, `user-agent` and `x-goog-api-client`;
- forwards to the model endpoint only `POST /chat/completions`, `POST /completions` and
  `GET /models` below its base URL, with the endpoint's own key as a Bearer token when it has one
  (`VLLM_API_KEY`, or a GCP identity token with `GBENCH_GCP_AUTH=1`);
- answers anything else with 403, not forwarded, and records it in `relay_refused_requests`
  (route, method, path, count, and the scenarios running then), not as a harness error; listing
  files, cached contents or tuned models, deleting any of them, and any other model are refused;
- drops whatever credential a request carries (the placeholder, the token) before forwarding.
  Responses stream back as they arrive;
- answers a request whose body is over 64 MiB with 413 without reading it, and handles at most
  128 requests at once (more wait), so the container cannot make it hold unbounded memory or
  threads in gbench's process. Such a request is recorded in `relay_refused_requests`, except a
  model-endpoint call the suite makes (an allowed call, with the token): its body cannot be shown
  to name another model, so it is the harness error `relay_body_too_large` (below).

The model under test's tools can reach the forwarder too, so they can make the calls above (the
alternate model included) and reach the model endpoint during the run; they never see a key or
the token, and such calls are counted in `gemini_calls` and `model_endpoint_calls` like the
suite's own. `leaderboard_comparable_reason` says so.

## Harness errors

A harness error is a failure that is not the model under test's doing, counted with the time its
request arrived:

- at the relay: `gemini_http_<status>` for every Gemini answer of 400 or more,
  `gemini_non_answer` for a Gemini reply of 200 that is not an answer (it does not parse, carries
  a block reason or a no-answer finish reason such as `SAFETY`, `RECITATION`, `OTHER`,
  `PROHIBITED_CONTENT` or `NO_IMAGE`, or has no answer part: no text, function call or inline
  data, thoughts not counted, and no image for image generation, or no embedding values),
  `gemini_truncated` for a Gemini reply that ends with `MAX_TOKENS` or a stream that ends without
  a finish reason, `model_http_<status>` for a model-endpoint answer of 401, 403, 408, 429 or 5xx,
  and for any other 4xx but 400 and 422 on a request the suite makes (`GET /models`, or a body
  whose `model` is the model gbench configured): for example the 404 of an endpoint that serves
  the model under another name (vLLM's ``The model `x` does not exist``) or has no such path, a
  405, 413 or 415. A 400 or 422 is the endpoint's answer to the request's content (an over-long or
  invalid request) and is the model's side, and so is any 4xx on a request that names another
  model (only the model's own tools, through the forwarder, send one);
  `gemini_unreachable` / `model_unreachable`, and `gemini_stream_broken` / `model_stream_broken`
  for a response that broke off, and `relay_body_too_large` for a model-endpoint call the relay
  answered 413 itself (over 64 MiB);
- at the forwarder: `forwarder_relay_unreachable` and `forwarder_relay_stream_broken`, from its
  records in the status directory, and `forwarder_exited` when the forwarder was no longer
  running at the end;
- in the suite: `suite_worker_failed` for a scenario the suite reports failed with the single step
  `isolated scenario worker` (its isolated worker threw before or around the scenario's flow:
  the lab, the gateway or the transport failed; the flow's own failures are reported under the
  flow's steps). It is tied to that scenario directly.

The suite runs every scenario in an isolated worker that writes the scenario's own summary with
its run window (`run.startedAt` to `run.finishedAt`). Each harness error is tied to every scenario
whose window, widened by one second on each side, holds its time (for a run of one scenario, the
run's window). Those scenarios are listed in `harness_error_scenarios` (`id`, `title`, the
suite's `suite_status`, and the errors) and left out of `total_scenarios`, `passed_scenarios`,
`failed_scenarios` and `pass_rate`, whether they passed or failed. An error that no window holds
(or has no time), or that falls in the window of a scenario summary that does not name exactly one
scenario of the run (which scenario it belongs to cannot be told, so the error may have been any
scenario's), is in `unattributed_harness_errors`, and then `pass_rate` is withheld (`null`), with
the reason in `pass_rate_withheld_reason`. When no scenario is left to score (every scenario
had a harness error fall in it or needs another provider, as in a run whose endpoint was down
throughout, or the suite reported none), `pass_rate` is withheld too, never `0`, and
`pass_rate_withheld_reason` says `no scenario could be scored: ...` with how many of the reported
scenarios each cause left out. Any harness error makes `status` `completed_with_errors`;
harness errors and a withheld pass rate are named in `leaderboard_comparable_reason`. The CLI
prints `withheld` for a withheld pass rate.

## The two edits of the checkout

gbench makes two recorded edits, each block exactly once in its file, in the image's build
context (never in a checkout).

1. The QA lab hard-codes the image-generation model in
   `extensions/qa-lab/src/providers/live-frontier/index.ts` and has no CLI option or variable for
   it, so gbench replaces these three lines:

   ```ts
     defaultImageGenerationProviderIds: ["openai"],
     defaultImageGenerationModel: ({ modelProviderIds }) =>
       modelProviderIds.includes("openai") ? "openai/gpt-image-1" : null,
   ```

   with:

   ```ts
     // gbench: image generation on the Gemini API (google plugin), see gbench docs/quality.md.
     defaultImageGenerationProviderIds: ["google"],
     defaultImageGenerationModel: ({ modelProviderIds }) =>
       modelProviderIds.includes("google") ? "google/gemini-3.1-flash-image-preview" : null,
   ```

2. The google plugin's HTTP helper (image generation, image understanding, speech) accepts only
   `https://generativelanguage.googleapis.com` as its base URL, with no private network, so
   `extensions/google/api.ts` also accepts exactly the relay's base URL
   (`GBENCH_GEMINI_RELAY_BASE_URL`) and, for it, the loopback address. Every other base URL is
   checked as before. gbench replaces the start of `resolveTrustedGoogleGenerativeAiBaseUrl` with:

   ```ts
   // gbench: the Gemini API through gbench's local key relay, see gbench docs/quality.md.
   function isGbenchGeminiRelayBaseUrl(baseUrl?: string): boolean {
     const relay = process.env.GBENCH_GEMINI_RELAY_BASE_URL?.trim();
     return Boolean(
       relay &&
         baseUrl &&
         normalizeGoogleGenerativeAiBaseUrl(baseUrl) === normalizeGoogleGenerativeAiBaseUrl(relay),
     );
   }

   function resolveTrustedGoogleGenerativeAiBaseUrl(baseUrl?: string): string {
     const normalized =
       normalizeGoogleGenerativeAiBaseUrl(baseUrl ?? DEFAULT_GOOGLE_API_BASE_URL) ??
       DEFAULT_GOOGLE_API_BASE_URL;
     if (isGbenchGeminiRelayBaseUrl(normalized)) {
       return normalized;
     }
   ```

   and, in `resolveGoogleGenerativeAiHttpRequestConfig`, `allowPrivateNetwork: false,` with:

   ```ts
       allowPrivateNetwork: isGbenchGeminiRelayBaseUrl(params.baseUrl),
   ```

   The alternate model's chat transport and the embeddings take the relay's base URL from
   `models.providers.google` without an edit.

- The edits are applied to the `git archive` copy of the commit in the build context, before the
  image installs or builds anything, so the built suite carries them. The checkout the archive is
  taken from is only read.
- If a file does not hold upstream's block (or gbench's) exactly once, nothing in it is changed,
  the setup stops with `infra_required: <file> does not hold the block gbench replaces (...)
  exactly once; the checkout differs from the pinned GemmaClaw commit`, and the run fails with
  `Failed to set up gemmaclaw repository: <that message>`. A commit that differs from the pinned
  one is never run half-patched.

The result records them in `gemmaclaw_patches` (one entry per file: `file`, `change`, `state`:
`applied` or `already applied`, and the `sha256` of the edited file), read from the image.

## Credentials

- **The QA suite** gets none of gbench's environment. Its container's environment is exactly:
  `HOME=/home/qa`, `TMPDIR=/tmp`, `LANG=C.UTF-8`, `npm_config_cache=/opt/gbench/npm-cache`,
  `npm_config_offline=true`, `npm_config_update_notifier=false`,
  `COREPACK_ENABLE_DOWNLOAD_PROMPT=0`, `OPENCLAW_CONFIG_PATH=/etc/gbench/openclaw.json`,
  `GBENCH_GEMINI_RELAY_BASE_URL=http://127.0.0.1:18431/v1beta`, `VLLM_API_KEY=gbench-relay` (a
  placeholder) and `OPENCLAW_ENABLE_PRIVATE_QA_CLI=1`, with the image's own variables (`PATH`,
  `COREPACK_HOME`, `GIT_COMMIT`, and the base image's `NODE_VERSION` and `YARN_VERSION`). So no
  Gemini key (`GEMINI_API_KEY`, `GEMINI_API_KEYS`, or a copy of one under
  any other name), no endpoint key, no proxy URL and no other credential reaches it, whatever its
  name; the names are recorded in `suite_env`. The relay token reaches only the forwarder.
- **Provider variables.** A variable named with one of the suffixes `_API_KEY`, `_API_KEYS`,
  `_API_TOKEN`, `_API_SECRET`, `_OAUTH_TOKEN`, `_TOKEN`, `_AUTHTOKEN`, `_SECRET`, `_PASSWORD` or
  `_CREDENTIALS`, matching `_API_KEY_<n>`, starting with one of `OPENAI_`, `ANTHROPIC_`,
  `OPENROUTER_`, `AZURE_OPENAI_`, `AWS_`, `BEDROCK_`, `CLAUDE_`, `CODEX_`, `OPENCLAW_LIVE_`,
  `GOOGLE_`, `VERTEX`, `MINIMAX_`, `FAL_` or `HUGGINGFACE_`, or named `HF_TOKEN`,
  `HUGGING_FACE_HUB_TOKEN`, `GH_TOKEN`, `GITHUB_TOKEN`, `COPILOT_GITHUB_TOKEN` or `FAL_KEY`, is a
  provider variable (except `GEMINI_API_KEY`, `GEMINI_API_KEYS` and `VLLM_API_KEY`, which gbench
  itself reads). The names (never the values) of those in gbench's environment are logged and
  recorded in `non_gemini_env_removed`.
- **The host-side commands** (`git clone` / `fetch` / `rev-parse` / `archive`, and `docker`) run
  with gbench's environment without the provider variables, without the Gemini keys, without any
  variable named like a key (`*_KEY`, `*_KEYS`, `*_KEY_<n>`), and without any variable whose value
  holds a Gemini key, the endpoint key or another credential value (a credential variable's
  value, the password of a URL). git runs with `-c core.hooksPath=/dev/null -c
  core.fsmonitor=false`: no hook or fsmonitor of the checkout runs. A git proxy that needs a
  password therefore goes in the git config `GBENCH_QUALITY_GIT_CONFIG_GLOBAL` names, not in
  `HTTPS_PROXY`.
- The relay token is redacted like a key. What gbench echoes of the container's output to the
  console is redacted of every credential value of the environment, and so is the suite's log file
  (`quality_<model>_<format>.log`) once the suite ends.
- A scenario that selects a model of another provider finds no key for it; see below.

## Scenarios that need another provider

Some scenarios need a model provider other than the model under test and the Gemini API: their
execution config requires the primary model to be another provider's (`requiredProvider` /
`requiredLiveProvider`, for example `gpt54-thinking-visibility-switch`, `openai/gpt-5.4`), or
names another provider's model (for example `thinking-slash-model-remap`,
`anthropic/claude-sonnet-4-6` and `openai/gpt-5.4`), a provider being any id a plugin of the
checkout declares. GemmaClaw's own lane matching already leaves out a scenario whose
`requiredProvider` / `requiredModel` the model under test does not meet, unless you select it
with `--scenario`.

Those that run anyway cannot be the model under test's outcome here: they are listed in
`infra_required_scenarios` (`id`, `title`, `providers`, `needs`, and the suite's own
`suite_status`), named in `leaderboard_comparable_reason`, and left out of `total_scenarios`,
`passed_scenarios`, `failed_scenarios` and `pass_rate`. The suite's own counts stay in
`suite_counts`. Scenarios are matched to the summary by title; a summary without a scenario list
is scored from its counts (and then any harness error withholds the pass rate).

## Prerequisites

Checked before anything else (preflight); a missing one fails the run with
`Quality runner preflight environment checks failed: <reason>`:

- **A Gemini API key**: `GEMINI_API_KEYS` or `GEMINI_API_KEY`. Without one the error names it
  (`infra_required: neither GEMINI_API_KEYS nor GEMINI_API_KEY is set; ...`).
- **git** (`infra_required: git is not usable ...`).
- **Docker**: the `docker` CLI on `PATH` and a daemon this user can reach (`docker info` must
  succeed; for example membership in the `docker` group). Otherwise `infra_required: Docker (the
  docker CLI) is not on PATH; ...` or `infra_required: the Docker daemon is not reachable:
  <message>; ...`. Node.js, pnpm and bubblewrap are not needed on the host.
- **gbench's `docker/` directory**, which holds the image's Dockerfile: run gbench from its source
  checkout (a wheel install does not carry it; the error says `... quality.Dockerfile is not in
  this gbench installation ...`).
- Network access to GitHub: gbench clones `https://github.com/gemmaclaw/gemmaclaw.git` into
  `$GBENCH_CACHE_DIR/gemmaclaw` (default `~/.cache/gbench/gemmaclaw`) without a working tree on
  first use and fetches it at the start of every run (`--gemmaclaw-path` fetches nothing). On
  the first run for a commit, the image build also needs Docker Hub (the base image), GitHub and
  the npm registry (GemmaClaw's install), and `dl.google.com` (Chrome), and about 8 GB of disk.
  The suite itself runs without a network.
- A host that cannot run Docker containers (for example a Cloud Run container) fails the
  preflight with the Docker message: the quality pillar needs a Docker host.

## Run

```bash
export GEMINI_API_KEY=...        # or GEMINI_API_KEYS=key1,key2 (it wins when both are set)
gbench --quality-only --models google/gemma-4-12B-it \
       --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer google/gemma-4-E4B-it
```

The suite asks for the model by its served id: `--served-model-id`, else the model's HF id, path
or name, matched against the endpoint's `/v1/models`. No match is a harness error; when a remote
endpoint's listing cannot be read, the configured id is used and the run says so.

## Knobs

| Env var | Default | Effect |
| --- | --- | --- |
| `GEMINI_API_KEYS` / `GEMINI_API_KEY` | *(one required)* | the Gemini API key(s) the relay adds to the suite's Gemini calls; `GEMINI_API_KEYS` wins when both are set |
| `VLLM_API_KEY` (a remote endpoint: else `OPENAI_API_KEY`) | `local-vllm` | the model-under-test endpoint's key: a remote endpoint's is `VLLM_API_KEY`, else `OPENAI_API_KEY` (`base.endpoint_api_key`); gbench's own server uses `VLLM_API_KEY` only (the key `vllm serve` reads from its environment); it stays with the relay |
| `GBENCH_GCP_AUTH` | unset | `1` fetches a GCP identity token (metadata server, then `gcloud`) for a `*.run.app` / `*.googleapis.com` endpoint when `VLLM_API_KEY` is unset; it stays with the relay. The CLI's endpoint check, which runs before the quality pillar, sends no token, so an endpoint that refuses unauthenticated calls (an IAM-protected Cloud Run service) stops the run before this applies |
| `GBENCH_QUALITY_GIT_CONFIG_GLOBAL` | `/dev/null` | the git global config of gbench's host-side git commands (blank by default, so a host `insteadOf` rewrite of GitHub HTTPS to SSH does not apply) |
| `GBENCH_CACHE_DIR` | `~/.cache/gbench` | where gbench's GemmaClaw clone lives (`gemmaclaw/`) |

`GBENCH_GEMINI_RELAY_BASE_URL` is set by gbench for the suite; it is not a knob.

## Reading the result

`quality_<model>_<format>.json` carries `pass_rate`, `total_scenarios`, `passed_scenarios`,
`failed_scenarios`, `blocked_scenarios`, `gemmaclaw_commit` and `raw_results` (the suite summary),
and:

- `status`: `completed`, or `completed_with_errors` when there was a harness error;
- `pass_rate_withheld_reason` (`null` unless the pass rate is withheld), `harness_error_scenarios`
  and `unattributed_harness_errors` (above);
- `suite_counts` and `infra_required_scenarios` (above);
- `alternate_model` and `image_generation_model`: the Gemini models the suite is configured with;
  `image_input_declared`: whether the model under test was declared with image input;
- `gemini_substitutions`: one entry per capability of the table above that applies to this run
  (the image-understanding row only for a text-only model under test), with GemmaClaw's default
  (`upstream`), the Gemini model, the API and how gbench set it;
- `gemini_calls`: every Gemini API call the relay forwarded, by `capability` (`alternate model`,
  `image generation`, `memory embeddings`, `media understanding`, `web search (Google Search
  grounding)` for a grounded call to one of the models above, `text generation`, `other`; the
  relay refuses the models and methods of speech, music and video generation), `model`,
  `method`, `requests` and `statuses` (HTTP status, or `unreachable`, with its count), and
  `non_answers` and `truncated` when there were any;
- `model_endpoint_relayed` (always `true`: the endpoint is behind the relay),
  `model_endpoint_key_at_relay` (whether the endpoint has a key of its own, kept by the relay)
  and `model_endpoint_calls`;
- `harness_errors`: every harness error, by kind (above);
- `relay_refused_requests` and `relay_rejected_requests` (above);
- `suite_exit_code` (the container's exit status: the suite's, 1 when a scenario failed) and
  `suite_stopped_after_summary` (above);
- `sandbox`: the container (`tool`: `docker`, the image, network, `metadata_guard` (the cloud
  metadata server guard's verdict, `no-network`), root file system, suite user, what the suite can
  write, gbench's status directory, and the two host paths it sees read-only);
- `gemmaclaw_patches`, `suite_env` and `non_gemini_env_removed` (above);
- `leaderboard_comparable: false`, with `leaderboard_comparable_reason` naming every substitution
  (the live-frontier lane's defaults are OpenAI models, so a pass rate here is not the lane's own
  number), that the model's tools can reach the Gemini calls and the endpoint through the
  forwarder, the infra_required scenarios, the harness errors and the scenarios they fell in, and
  a withheld pass rate.
