# tau2 setup

tau2-bench (Sierra `sierra-research/tau2-bench`, domains: airline / retail / telecom)
is a multi-turn **environment** benchmark: an LLM user and the agent converse over
several turns, tools mutate a domain database, and each task is scored by the
simulator's oracle (final DB-state check, environment assertions, required actions,
communication checks) plus, for tasks whose reward basis includes it, an nl_assertion LLM
judge, yielding a per-task reward.

## Only the real simulator scores it
Against a plain single-turn `/v1` endpoint, tau2 **cannot** be measured: many tasks are
assertion-only (no tool action to match), and the tasks that do require actions carry
gold arguments (e.g. `user_id`, `reservation_id`) that only exist in the multi-turn
environment state the model never sees. gbench therefore always runs the tau2-bench simulator
itself; there is no single-turn score. (The single-turn strict-action matcher is retained as
`_eval_tau2` for diagnostics/tests, but is not wired as a scorer.)

## How gbench runs it: the local `gbench-tau2` image
tau2-bench pins its own dependencies in `uv.lock` (for example `litellm` 1.81.11, where the gbench
environment carries a newer litellm) and is not on PyPI, so gbench runs it in a **local Docker
image** built from [`docker/tau2.Dockerfile`](../../docker/tau2.Dockerfile). The image holds a
tau2-bench checkout at the pinned commit (see *Version*), installed the way upstream installs it
(`uv sync --extra knowledge` from its frozen `uv.lock`), plus Anthropic's sandbox-runtime (`srt`)
0.0.23 with ripgrep, bubblewrap and socat for the banking_knowledge shell tool (see
[tau3.md](tau3.md)). gbench never pulls it; you build it once.

For each run, gbench starts a container from that image and runs its own tau2 driver
(`gbench/runners/eval_suites/tau_common.py`, mounted read-only from your gbench install, so the
image never goes stale when gbench changes) inside it:

- the container runs on **docker's bridge network**, never the host network (see *Container and
  network* below): it reaches gbench's relays at the bridge gateway's address: the Gemini relay
  (and through it the Gemini API) and the **endpoint relay**, through which the agent reaches
  your endpoint on every run, with or without an endpoint key (since 2026-10-13); it runs as
  **your uid/gid** with every capability dropped and `no-new-privileges`;
- gbench forwards the `GBENCH_TAU2_*` / `GBENCH_TAU3_*` / legacy `TAU2_*` knobs,
  `GBENCH_TAU_SANDBOX_MAX_AGE_H`, `LITELLM_LOG` and the proxy variables by name, and hands the
  container the run's Gemini cascade as `GBENCH_JUDGE_MODELS` (and where the host took it from as
  `TAU_JUDGE_CASCADE_SOURCE`, so `user_llm_source` / `eval_llm_source` name the host's source
  rather than the variable the container received) and, in place of any key, the
  per-run placeholder and address of gbench's **Gemini relay** (see *Credentials*), all by name,
  so no value appears on the command line;
- a proxy variable (`HTTP_PROXY`, `HTTPS_PROXY` and their lowercase forms) on a loopback address
  is rewritten to `host.docker.internal` too, and `host.docker.internal` and the bridge gateway
  are added to `NO_PROXY` / `no_proxy` when a proxy is set, so the container's requests to
  gbench's relays go to them directly. A user name or password in a proxy URL never enters the
  container: the variable is forwarded without it (the agent's requests leave from gbench's
  endpoint relay, which uses the host's proxy settings); the result names the variables whose
  credentials stayed on the host (`tau2_report.proxy_credentials_withheld`);
- the per-task traces are written straight into the run's trace directory on the host (see
  *Where the simulations go*), and the result comes back through a temporary exchange
  directory that is removed afterwards;
- the container's working directory is `<gbench cache>/tau2` (`$GBENCH_CACHE_DIR`, default
  `~/.cache/gbench`), where tau2 keeps its `data/.embeddings_cache` (tau3's document embeddings)
  as upstream does in its checkout, keyed on the documents, the embedder type and the model;
- `--temperature` is passed through, and the container's output (the per-domain lines and the
  progress bar) is streamed to gbench's console.

Inside the container the driver refuses to run unless the imported `tau2` is the pinned,
unmodified checkout and every package tau2-bench depends on (its base dependencies and the
`knowledge` extra) is installed at its `uv.lock` version. Docker missing, the image missing, or an
image built at another revision hard-errors (`infra_required`) with the build command; there is no
fallback to running tau2 in gbench's own interpreter.

Three LLM roles are used:

- **agent** = your model, routed to the gbench endpoint via LiteLLM `openai/<model>` (LiteLLM's
  client for OpenAI-compatible servers) with `api_base` passed **per-call**,
  so a global `OPENAI_API_BASE` never redirects the agent (the container has none). The key sent
  to your endpoint is the endpoint key every gbench pillar sends (`VLLM_API_KEY`, else
  `OPENAI_API_KEY`): a key is added by gbench's endpoint relay on the host, outside the
  container (see *Credentials*); the agent itself sends `EMPTY`, added to each agent request
  rather than stored in the agent's `llm_args`, which tau2 writes into every `results.json`. No
  provider key (Gemini, OpenAI, ...) is ever sent to your endpoint. Every agent request goes
  through that relay, keyless too, which puts the run's `--max-output-tokens` on it (see
  *Token budget*); the user simulator and the judge never do.
  The agent
  is sent only a temperature, as upstream's agent (`DEFAULT_LLM_ARGS_AGENT = {"temperature": 0.0}`
  in `src/tau2/config.py`): gbench's rule, 0.0 without `--thinking` and 1.0 with it
  (see *Sampling*). Its requests are sent once and its
  replies are used as they are, as in upstream tau2.
- **user simulator** = gbench's one **Gemini cascade**, the same list and order every Gemini text
  role of gbench uses: the models of `GBENCH_JUDGE_MODELS` (comma separated), else the one model
  of `GBENCH_JUDGE_MODEL`, else the default `gemini-3.8-flash`, `gemini-3.7-flash`,
  `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.1-pro-preview` (`base.GEMINI_CASCADE`), tried
  in that order, at temperature 0.0 (`GBENCH_TAU2_USER_TEMPERATURE`). `GBENCH_TAU2_USER_LLM` names
  one `gemini/<model>` instead, used alone with no fallback. `src/tau2/config.py` defaults to
  `gpt-4.1-2025-04-14`; `gpt-5.2` is recommended by `docs/leaderboard-submission.md` and used by
  Sierra's 2026 standard text entries. That guide counts a non-default user simulator as a
  *custom* submission and the leaderboard shows each entry's user simulator.
- **nl-assertion judge** = the same cascade (or the one `gemini/<model>` of
  `GBENCH_TAU2_EVAL_LLM`), at temperature 0.0, upstream's `DEFAULT_LLM_NL_ASSERTIONS_ARGS`.
  Upstream fixes it to `gpt-4.1-2025-04-14`.

The host hands the image its cascade as `GBENCH_JUDGE_MODELS` (`base.gemini_cascade_env()`), so
the container calls exactly the models gbench records. Every external model a tau run calls goes
through the Gemini API: a user simulator or judge that is not a `gemini/` model hard-errors (see
*Credentials*), and tau3's retrieval embeds with `gemini-embedding-001` (see [tau3.md](tau3.md)).

How a user-simulator or judge call goes through the cascade (`_build_gemini_cascade`):

- It starts at the first model. Each model gets the call's own retry budget, the `num_retries`
  tau2's `generate()` sends (3, as upstream): on a transient error (an `openai.APIError`, LiteLLM's
  own retry rule: an HTTP error status, a timeout or a lost connection) the same model is asked
  again, up to 4 attempts, and only then does the next model get the call, after a short backoff
  (`GBENCH_TAU2_LLM_BACKOFF`, 1 s doubling). LiteLLM itself is called with `num_retries=0`: its
  own retry calls `litellm.completion` again, which carries no cascade, so it can never re-enter
  the cascade and multiply the attempts (with the five default models, a call that fails on
  every one makes 5 x 4 = 20 requests; letting LiteLLM's retries re-enter the cascade would make
  80). A model that answers HTTP 404 is not served for
  the key: it is not retried and is skipped for the rest of the run
  (`tau2_report.gemini_models_not_served`). When every model fails, tau2 ends the simulation
  attempt; a simulation whose every attempt failed is a harness error (see *Reading the result*).
- Every model gets the same request: the conversation (and, for the user simulator, the user's
  tools), the temperature, and nothing else, as upstream sends its user simulator and judge. For
  every Gemini 3 model, LiteLLM 1.81.11 would add a `thinkingConfig` of its own that the caller
  did not ask for (`thinkingLevel` "minimal" for a name containing `gemini-3-flash`, "low" for
  any other); gbench removes it (`_install_gemini_param_guard`, the run hard-errors if it cannot
  be installed), so no model is sent a `thinkingLevel` and each answers at the Gemini API's own
  default. `tau2_report.gemini_call_parameters` records this.
- The requests go to gbench's Gemini relay as `POST /v1beta/models/<model>:generateContent`
  (see *Credentials*), which adds the key and forwards them to the Gemini API.
- The model that answered is counted: `gemini_models_answered` (replies per model over both
  roles), `tau2_report.gemini_models_answered_by_role` and `..._by_domain`, and every
  NL-assertion verdict names the model that gave it: `reward_info.info.nl.judge_model` in each
  simulation of `results.json`, summed per domain in `tau2_report.nl_judge_models_by_domain`.
  Each user-simulator message in `results.json` also carries the model LiteLLM reports for it
  (`raw_data.model`).

gbench's robustness patches belong to that Gemini substitution: an empty or malformed reply from
a `gemini/` user simulator or judge (no text but whitespace and no tool call, or empty/unparseable
tool arguments) is requested again, up to 3 attempts in all, and markdown code fences are
stripped from their text. They never touch the agent.

### Container and network
The suite starts one container per run, and it is the one where the agent's tool calls run (and
tau3's shell tool, see [tau3.md](tau3.md#container-and-network)). It runs on docker's default
bridge network with `--add-host host.docker.internal:host-gateway`; the host network is never used,
and no tau2 container runs with `--network none` (gbench's relays for the agent's endpoint and
for the Gemini user simulator, judge and tau3's embeddings are all reached from it). On a cloud VM it
relies on the host rule that keeps a bridge network away from the metadata server at
`169.254.169.254` (which hands out the VM's service-account token):

```bash
sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save   # keep it
```

LiteLLM in the container runs with `LITELLM_LOCAL_MODEL_COST_MAP=True`, so it uses the model cost
map it ships instead of fetching one from `raw.githubusercontent.com` when it is imported. The
map only feeds LiteLLM's cost bookkeeping; no request changes.

Before the container starts, gbench's guard (`require_metadata_blocked`, see
[toolchains](toolchains.md#cloud-vms-the-metadata-server)) starts one short-lived container of the
tau2 image on the bridge network that opens and closes a TCP connection to `169.254.169.254:80` (no
request is sent); if it opens, the suite hard-errors (`infra_required`) naming the rule, and a probe
that gives no clear answer (neither `python3` nor `bash` in the image, a docker error or timeout) is
a hard error too. The tau2 container looks up no host name itself: the agent's requests go to
gbench's endpoint relay and its Gemini calls to gbench's Gemini relay, both at the bridge gateway's
address (see *Credentials*); the relays, on this host, reach your endpoint and resolve
`generativelanguage.googleapis.com`. So only the metadata server is checked (since 2026-10-13
also on a keyless endpoint, where the container used to reach the endpoint directly and the probe
also resolved its host name). A container on the bridge network still needs working DNS for its
general internet access, and on GCE the VPC DNS resolver is `169.254.169.254` itself, which is
why the rule spares port 53 (a plain `-j DROP` for the address cuts DNS in every container). When
a lookup or connection fails later in the run (the
Gemini API or the endpoint unreachable) on every attempt tau2 makes at a simulation, the
simulation ends in an infrastructure error from the user simulator, the judge, the embeddings or
the model endpoint: a harness error, never the agent's outcome (see *Reading the result*). gbench
never runs sudo or changes the host's firewall itself. Your endpoint URL is used as given by
gbench's endpoint relay, from this host (a server bound to `127.0.0.1` only is fine). gbench's
relays listen on the docker bridge
network's gateway address (`docker network inspect bridge`, e.g. `172.17.0.1`): gbench checks the
same way that a container of the image opens a TCP connection to each relay, and hard-errors
(`infra_required`) naming the address otherwise (a host firewall must let the docker bridge reach
it). When a proxy is set, the gateway is added to the container's `NO_PROXY` / `no_proxy`, so
the requests to the relays go to them directly. `GEMINI_OPENAI_BASE_URL` never enters the
container: gbench's Gemini relay, on the host, forwards to its origin (a loopback value, a local
stub for tests, is this host's loopback).

The container has general internet access on the bridge network, as upstream's harness has on
the machine it runs on; only the metadata server is blocked. The model's tool calls are tau2's
in-process domain tools, and tau3's shell commands run in sandbox-runtime with upstream's settings
(`allowedDomains: []`), which give them no network.

### Credentials
No API key or other credential is visible to the model under test or written to the traces, the
result or the log:

- **No key enters the container.** The container runs the agent's tool calls (and tau3's shell
  commands, as the same user), so it holds no key, token or password. Every external model a tau
  run calls (the user simulator, the judge and tau3's retrieval embeddings) goes through the
  Gemini API by way of gbench's **Gemini relay** (`gemini_relay.GeminiRelay`), a thread of the
  gbench process listening on the docker bridge gateway: the container gets only the relay's
  per-run placeholder as `GEMINI_API_KEY` and its address as `GBENCH_GEMINI_RELAY_URL`, sends its
  Gemini requests there with the placeholder, and the relay adds the real key
  (`GEMINI_API_KEYS`, else `GEMINI_API_KEY`, taken in turn) and forwards them to the Gemini API.
  The relay forwards only native `generateContent` (and, for tau3, `batchEmbedContents`) for the
  run's models; anything else, or a request without the placeholder, is refused. Inside the
  container the driver refuses a real Gemini key and a placeholder without the relay's address.
  Its record is in `tau2_report.gemini_relay` (requests relayed and refused, and the Gemini
  failures it saw, each after every key was tried: every failed request, including the ones a
  retry or the next model recovered from, so not a count of lost simulations). The key is checked
  on the host before the container starts (see *Requirements*); `GEMINI_OPENAI_BASE_URL` is the
  endpoint of that check and the relay's upstream: only the Gemini API over https or a loopback
  address is accepted, any other value hard-errors (`infra_required`) before the container
  starts, and it never enters the container. A user
  simulator or judge that is not a `gemini/` model (an OpenAI model such as upstream's
  `gpt-4.1-2025-04-14`, an `anthropic/`, `openrouter/`, `vertex_ai/`, `bedrock/` or `azure/`
  model, or a bare `gemini-*` name, which LiteLLM 1.81.11 routes to Vertex AI rather than the
  Gemini API) hard-errors (`infra_required`) naming the provider before the container starts;
  inside the container, a `gemini/` name that tau2's LiteLLM routes elsewhere or cannot route
  hard-errors too. Your own `OPENAI_BASE_URL` / `OPENAI_API_BASE` are not passed in, and the
  former `GBENCH_TAU2_OPENAI_API_KEY` / `GBENCH_TAU2_OPENAI_BASE_URL` no longer change anything
  (gbench logs a warning when one is set).
- **Your endpoint key.** The endpoint key (`VLLM_API_KEY`, else `OPENAI_API_KEY`; `EMPTY`,
  `dummy` and the other placeholders count as no key) never enters the container either:
  gbench's **endpoint relay** (`swebench_common.EndpointRelay`) on the bridge gateway adds it,
  the agent's base URL is the relay's, and the agent sends `EMPTY`; its record is in
  `tau2_report.endpoint_relay`. Without a key, the agent's `EMPTY` travels as a private variable
  that the driver removes from the environment before tau2 starts, and is added to each agent
  request (see *agent* above).
- **Redaction.** Tool results and the messages sent to the agent have every credential value of
  the run replaced by `[REDACTED]`; `results.json` is written with them replaced and every file
  of the run directory is checked again after each domain; the result and the streamed container
  output are redacted too. `tau2_report.credentials_redacted` records this. Values shorter than 8
  characters (placeholders such as `EMPTY`) are not treated as credentials.
- tau3's shell tool runs its commands without any credential in their environment; see
  [tau3.md](tau3.md#credentials-and-the-model-under-test). retail, airline and telecom give the
  agent no shell.

### Version (pinned)

gbench runs tau2-bench at commit **`b7ea9074c1cba482b30687fecdb5c8425fd6f619`** (the `main`
branch as of 2026-09-17, package version 1.0.1; still the head of `main` on 2026-09-25). Its task
data, `data/tau2/domains`, is byte-identical to the **v1.0.1** release tag (commit
`fc0055dc4e0a316c3f83133267fbd6faaa770992`); the commits in between change the voice stack, the
batch runner (rebuilt around a work queue, with an optional multi-process mode), the leaderboard
website and the submission docs. At the pinned commit the website splits the text leaderboard
into two tracks (`web/leaderboard/src/components/Leaderboard.jsx`): **tau2-bench**
(retail, airline, telecom; its Overall is the unweighted mean of the three domains) and
**tau3-banking** (`banking_knowledge`, see [tau3.md](tau3.md)).

The image installs exactly the pinned commit's `uv.lock` (`litellm` 1.81.11, `openai` 2.20.0,
`rank-bm25` 0.2.2, `numpy` 2.3.5, ...). One addition is needed: at this commit `import tau2` also
imports `websockets` (`tau2.data_model.simulation` loads the voice stack at module level) although
only the `voice` extra lists it, so upstream's text-mode install alone cannot import tau2; the
image adds the lock's `websockets` 15.0.1 and no other voice package. The image is labelled
`gbench.tau2.revision=<commit>`, and gbench checks that label before it starts a container.

Inside the container the suite checks the revision again: the imported `tau2` package must be the
`src/tau2` of a git checkout whose `HEAD` is the pinned commit, with no local changes to tracked
files under `src/` or `data/`, and tau2 must read its task data from that checkout's `data/` (so
`TAU2_DATA_DIR` is not forwarded into the container). Anything else hard-errors
(`infra_required`) and names what it found. The result records `tau2_report.tau2_revision`,
`tau2_report.tau2_version` and `tau2_report.image` (name and image id).

### Requirements
- **Docker** (the CLI on `PATH` and a reachable daemon).
- **On a cloud VM, the host rule** `sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP`,
  made persistent, and an endpoint the docker bridge reaches (see *Container and network*).
- **The image**, built once from the gbench repository root (about 400 MB; it clones the pinned
  commit and installs its lock, so the build needs network access):
  ```bash
  docker build -t gbench-tau2:b7ea9074 -f docker/tau2.Dockerfile docker/tau2_srt
  ```
  The build context `docker/tau2_srt` holds only sandbox-runtime's `package.json` and
  `package-lock.json` (locked dependency tree with integrity hashes). The build fails unless tau2
  imports from the checkout with the lock's `litellm`, `openai`, `rank-bm25` and `websockets`, and
  `srt`, `rg`, `bwrap` and `socat` all resolve. `GBENCH_TAU2_IMAGE` names another tag. Rebuild
  only when the pinned commit changes. Nothing is installed into gbench's own Python environment.
  Licenses: tau2-bench (code and task data) is MIT (Copyright (c) 2025 Sierra Research);
  sandbox-runtime is Apache-2.0. The image is built on your machine and gbench redistributes
  neither.
- **`GEMINI_API_KEY`** (or `GEMINI_API_KEYS`): required, on the host only. The user simulator and
  the judge are the models of gbench's Gemini cascade (`GBENCH_JUDGE_MODELS`, or the one
  `gemini/` model of `GBENCH_TAU2_USER_LLM` / `GBENCH_TAU2_EVAL_LLM`), and tau3's retrieval embeds
  with the same key. Before the container starts gbench checks the key and every model the roles
  may call (`base.gemini_key_live_valid`, then `base.gemini_models_available` for a role knob's
  model): a missing key, a key the API refuses or a model it does not serve hard-errors
  (`infra_required`) naming it; a network error there is inconclusive and does not block.
- **A served model with tool calling enabled.** tau2's agent sends the domain's tools natively
  (LiteLLM `tools`), so the endpoint must return tool calls (vLLM: `--enable-auto-tool-choice
  --tool-call-parser <the model's parser>`; a server gbench launches itself (no `--remote-endpoint`) gets both from `--tool-call-parser NAME`, and the CLI refuses a local run of this suite without it).
- **Disk** for the simulations gbench keeps (see *Where the simulations go*): tau2 writes every
  message of every trial, roughly 25-30 MB per domain per trial, so about 100 MB per domain at 4
  trials.

If Docker or the image is missing, the image was built at another revision, a key is missing,
or any domain fails or produces no evaluable simulation, the suite **hard-errors**
(`infra_required`); it never emits a misleading or partial score. No opt-in variable is needed
(`GBENCH_TAU2_ENV_RUN` no longer changes anything).

### Protocol
A run follows the leaderboard submission rules (`docs/leaderboard-submission.md` at the pinned
commit): every task of each domain's `base` split (retail 114, airline 50, telecom 114), **4
trials per task** (`GBENCH_TAU2_NUM_TRIALS`, default 4; the leaderboard wants "4+ trials" and
shows pass^1 to pass^4), and upstream's other defaults (200 max steps, 10 max errors, seed 300).
Each simulation is scored by tau2's own evaluator, and pass^k comes from tau2's
`compute_metrics`, which leaves out simulations that ended in an infrastructure error (never
scored; see *Reading the result* for which of them are harness errors). The three domains run one
after the other; a domain that fails, or has no evaluable simulation, stops the run.

### Where the simulations go
Every run writes its simulations to a directory of its own,
`<eval>_<model>_<think|nothink>_<config hash>_<UTC time>-<random id>/`, with one
`<domain>/results.json` per domain (tau2's full `Results`: per-task messages and reward
breakdown). The name is unique to the eval, the model, the thinking mode, the run configuration
and the invocation, gbench creates it fresh, and tau2 is never allowed to resume: `auto_resume`
is off and tau2's "resume the run? (y/n)" prompt is answered "n", so an existing file makes the
domain fail instead of being reused. Two models in one invocation, a re-run into the same
results directory, or a `--thinking` toggle can therefore never pick up another run's
simulations. The directory lives under `<run results dir>/tau_traces/` by default (mounted into
the container) and is recorded, as a host path, in `tau2_report.run_dir` /
`tau2_report.traces`; `GBENCH_TAU2_SAVE_TRACES=<dir>` moves it (a relative path is resolved
against the current directory), and `GBENCH_TAU2_SAVE_TRACES=` (empty) keeps it in a temporary
directory inside the container that is removed when the run ends. The run configuration and its
hash are in `tau2_report.run_config` / `tau2_report.config_hash`.

### Sampling
- Agent temperature follows gbench's rule for every suite: **0.0** without thinking
  (`temperature_source` `"default"`; also upstream's `DEFAULT_LLM_TEMPERATURE_AGENT`) and **1.0**
  with `--thinking` (`temperature_source` `"thinking"`, named in
  `leaderboard_comparable_reason`). Override it for a whole run with `--temperature`,
  or for tau2 alone with `GBENCH_TAU2_TEMPERATURE` (which wins over both). `--eval-thinking`
  turns on the model's thinking channel for the agent.
- Thinking: with `--thinking` every agent request carries the chat template's thinking switch
  (`extra_body.chat_template_kwargs.enable_thinking: true`); a no-think request sends none, as
  upstream's (the served model's default, off for Gemma 4). gbench counts the agent's requests at
  tau2's completion chokepoint by the value they carried (`agent_request_shape`) and records
  `thinking` and `thinking_applied` (with `--thinking`, every agent request carried true; without,
  none did). LiteLLM 1.81.11, the image's, sends `extra_body`'s
  fields in the request's JSON body (checked in the image against a local server, 2026-10-11).
- Request timeout: tau2's `generate()` calls LiteLLM without a timeout, and LiteLLM 1.81.11 then
  gives each request 600 s (its `REQUEST_TIMEOUT` variable is not read there), so an agent turn
  that needed more than 600 s of decode was dropped and asked again: a hidden cap on the turn's
  length (2026-10-10 investigation). gbench adds `timeout` to the agent's LLM arguments only:
  `base.request_timeout_s` of the served context window (the agent sends no `max_tokens`; 33,968 s
  for a 262,144-token window; gbench passes the window into the image). tau2's own retries
  (`num_retries=3`, which LiteLLM 1.81.11 turns into up to 7 HTTP attempts of a call whose every
  attempt times out) are unchanged, and the Gemini user simulator and judge keep their own
  settings. gbench's endpoint relay waits that timeout plus 120 s for one request. Since
  2026-10-13 the relay fills the run's `--max-output-tokens` into each agent request (see *Token
  budget*); the window-sized timeout is kept, at least as long as one for the filled value, so it
  never cuts a reply the relay lets through. The result records `model_client_timeout`
  (`{client, version, timeout_s, retries,
  http_attempts_if_every_attempt_times_out, source, replaces, hidden_cap_removed}`).
- Like upstream, only the temperature is sent among the sampling parameters: `top_p` / `top_k` stay at the server's defaults
  (the model's `generation_config.json` under vLLM) unless you set `GBENCH_TAU2_TOP_P` /
  `GBENCH_TAU2_TOP_K`. The values used are recorded in `tau2_report.agent_sampling` and in
  `temperature` / `temperature_source`.
- The user simulator (`GBENCH_TAU2_USER_TEMPERATURE`, default 0.0) and the judge (0.0) are not
  affected by the agent's knobs.

### Tuning knobs (env)
`GBENCH_TAU2_NUM_TRIALS` (4, the leaderboard's trial count; fewer is named in
`leaderboard_comparable_reason`; unset, `--attempt-count` above 4 raises the trials to that
value, and a smaller `--attempt-count` never cuts the 4: the result's `attempts` records the
trials that ran and why); `GBENCH_TAU2_MAX_STEPS` (200); `GBENCH_TAU2_MAX_ERRORS` (10);
`GBENCH_TAU2_SEED` (300); `GBENCH_TAU2_USER_TEMPERATURE` (0.0); `GBENCH_TAU2_TOP_P` /
`GBENCH_TAU2_TOP_K` (unset); `GBENCH_TAU2_LLM_BACKOFF` (1.0: seconds before the next cascade
model for the user/judge, doubling per model); `GBENCH_TAU2_PROGRESS_SECS` (30, upper bound on
how often the progress bar's
postfix refreshes; the bar itself advances per completed task); `GBENCH_TAU2_SAVE_TRACES`
(default `<run results dir>/tau_traces/`; see *Where the simulations go*); `GBENCH_TAU2_IMAGE`
(default `gbench-tau2:b7ea9074`, the image to run); `GBENCH_TAU2_USER_LLM` /
`GBENCH_TAU2_EVAL_LLM` (one `gemini/` model for the user simulator / judge, used alone instead of
the cascade; see *Credentials*). The cascade itself is gbench's `GBENCH_JUDGE_MODELS` /
`GBENCH_JUDGE_MODEL` (see [env-knobs](env-knobs.md)); `GBENCH_SEARCH_*` does not apply to tau.
The bare `TAU2_*` names still work as deprecated aliases. `GBENCH_TAU2_OPENAI_API_KEY` and
`GBENCH_TAU2_OPENAI_BASE_URL` no longer change anything (no OpenAI service is called).
`GBENCH_TAU2_BENCH_SRC` and `GBENCH_TAU2_ENV_RUN` are no longer used (the image carries the
checkout, and the suite always runs the simulator).
`--sandboxes` maps to the simulator's `max_concurrency`; `--eval-limit N` is N tasks for the
whole run, never N per domain: the N tasks are dealt over the domains (`sampling.allocate_limit`,
seed the suite name; each domain capped at its task count), so `--eval-limit 5` runs exactly 5
tasks (2/1/2 over retail, airline and telecom) and `--eval-limit 2` gives one
domain (retail) no task, which then does not run (`limit_skipped_domains`; the headline averages the
domains that ran and `leaderboard_comparable_reason` names the others). Each domain's share is
drawn from its tasks with a seeded draw (`sampling.stratified_sample`, seed the suite name: the
same tasks on every machine), not tau2's `num_tasks` head of the task file; the selected ids go
to tau2 as `task_ids` and the result records them (`selected_task_ids`, with
`domain_task_counts`). Each task still runs its `num_trials` trials (repeats of one task, not
extra samples). The run logs its planned workload on one line at its start, e.g.
`[tau2] --eval-limit 5: 5 sample(s) over 3 domain(s) (retail 2, airline 1, telecom 2) x 4
attempts = 20 conversations`, and records it as `eval_limit_plan`. `--eval-thinking` enables the
model's thinking channel for the agent.

**`--shard I/N`** (GBENCH_SHARD, validated on the host before the container starts): each
domain's tasks, in tau2's own load order of the `base` split, are split round-robin and shard I
runs every N-th task starting at the I-th, before `--eval-limit` (whose seeded draw then takes
its share from the shard). The result records `shard`, the per-domain task counts
(`domain_task_counts`), the task ids run (`selected_task_ids`) and, per domain and task, the
evaluable and successful trials
(`tau2_report.task_outcomes_by_domain`): `gbench --merge` recomputes each domain's pass^k over
the union of the shards exactly as tau2's `compute_metrics` does, and the headline over the
domains (`merge_shards`), and unites the shards' `selected_task_ids` per domain. A shard that
holds no task of a domain skips it
(`empty_shard_domains`, named in `leaderboard_comparable_reason`); a shard with no task at all
reports no score (`accuracy` null, `empty_shard`).

**Log noise & progress:** by default gbench quiets tau2's own output so the log stays
readable: (1) two cosmetic ERROR logs (litellm cost-mapping miss, `git rev-parse HEAD`
failure), (2) tau2's rich per-task "Simulation Overview" panels + live "Status: X/N
complete" progress reprints (which spam a file-redirected log since rich can't update in
place), (3) litellm's per-call Gemini-3+ sampling `DeprecationWarning` (emitted twice per
user-sim/judge call), and (4) tau2's import-time loguru noise: the `Registry info: {...}`
DEBUG dump (domains/agents/users/task_sets) and the benign `No .env file found` WARNING
(tau2 looks for an optional `.env` to load keys; gbench passes keys via the environment,
so a missing `.env` is irrelevant; genuine tau2 warnings such as task retries still
show). gbench builds its own summary from the results, so these are redundant. In their place gbench keeps the log **evolving like every other eval**:
- **per-domain lines** through its own logger (a "starting domain 'airline'" line then a
  per-domain pass^1 and mean-reward line when it finishes);
- a **tqdm progress bar**, one per domain, driven by tau2's own `StatusMonitor`, the same
  `tqdm(total=..., desc="Eval [X]")` mechanism every other eval uses. It advances per
  completed task and carries a live postfix (`reward`, `running`, `oldest`), e.g.
  `Eval [tau2 telecom]: 60%|######    | 68/114 [12:03<08:15, reward=0.58, running=32, oldest=146s]`.
  Like the other evals' bars it updates **in place via carriage returns**, a single
  updating line on a TTY or under `tail -f`; a file redirect stores each `\r` frame (which
  a terminal collapses on display). The postfix refresh cadence is capped by
  `GBENCH_TAU2_PROGRESS_SECS` so the bar stays live between completions.

Set **`GBENCH_TAU2_VERBOSE=1`** to restore all tau2 panels + litellm logs + the registry dump +
tau2's original console heartbeat. Real errors still surface via loguru either way.

## Run
```bash
# once: build the image (from the gbench repository root)
docker build -t gbench-tau2:b7ea9074 -f docker/tau2.Dockerfile docker/tau2_srt
# then
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals tau2 --sandboxes 8 --max-output-tokens 65536
```
`--max-output-tokens` is required for every eval run.

### Token budget
tau2's agent sends no `max_tokens` (with or without `--thinking`), as upstream. gbench's endpoint
relay puts the run's `--max-output-tokens` on every agent request that carries none (tau2 and tau3
are agent suites, `swe_agentic.AGENT_MAX_TOKENS_SUITES`, the user's decision of 2026-10-11: without
a per-reply bound one runaway turn runs to the end of the served window). Only the agent (the
model under test) is capped: the Gemini user simulator, the NL-assertion judge and tau3's
embeddings go through the Gemini relay and are never capped. The result records it:
`max_tokens_sent` (the value), `max_output_tokens_note`,
`official_sampling.agent_max_tokens_from_run` (`{max_tokens_sent, source, sent_via, upstream, requests_filled}`) and the relay's own record,
`tau2_report.endpoint_relay.max_tokens_from_run` (`{value, requests_filled}`); a run with the cap is
not leaderboard-comparable, and `leaderboard_comparable_reason` says `max_tokens=<N> (the run's
--max-output-tokens) sent with <n> agent request(s) whose upstream harness sends none`, the eval
runner's own words (`evals._stamp_agent_max_tokens`), so it is stated once. The cap is recorded
only when the relay filled at least one request: a run none of whose agent requests went through
the relay says `not sent` in `max_output_tokens_note`. `gbench --merge` sums `requests_filled`
over the shards (the reason, `official_sampling.agent_max_tokens_from_run`,
`agent_max_tokens_from_run`, the note and the relay record's `max_tokens_from_run`, with
`summed_over_shards`) and takes the other fields from the first shard that holds a task, not an
empty one. `GBENCH_AGENT_MAX_TOKENS_FROM_RUN=0` turns the fill off (no `max_tokens`, as upstream;
`max_tokens_sent` null).

Until 2026-10-13 the endpoint relay started only with a real endpoint key (or a proxy with
credentials), so on a keyless endpoint the agent reached the server directly and the cap never
applied: smoke run `smoke-10-12-235114` tapped 86 tau2 and 133 tau3 agent requests, none with
`max_tokens`. Under `gbench --smoke` the tau relay serves every reply (it is no smoke relay:
a tau2 domain whose every simulation ended at the relay's `GBENCH_SMOKE_STOP` would have no
evaluable simulation and hard-error), so a smoke episode runs to its own end, as before.

## Reading the result
- **`accuracy`** is the headline: tau2's pass^1 in %, averaged over retail, airline and telecom
  with equal weight, which is the **tau2-bench leaderboard track's Overall** at the pinned
  commit (on a shard that holds no task of a domain, the mean of the domains it ran: the
  track's per-domain columns); `leaderboard_track` says which. pass^1 excludes simulations that
  ended in an infrastructure error, as `compute_metrics` does.
- `pass_k` carries pass^1 to pass^4 (up to the trial count) at the same level, the other
  columns the leaderboard shows; `pass_k_by_domain` has them per domain, `pass1_by_domain` the
  per-domain pass^1, and `pass1_domain_mean` repeats the headline (the unweighted mean of the
  domains that ran). `tau2_report.pass_hat_k` holds the per-domain pass^k as fractions.
- `category_accuracy` reports each domain's pass^1 (`accuracy`) with its perfect and evaluable
  simulation counts (`correct` / `total`: the simulations that did not end in an infrastructure
  error, the ones pass^1 covers), and apart from them all its simulations (`simulations`) and its
  infrastructure errors (`infra_errors`), so an infrastructure error never reads as a failed
  simulation. `correct_answers` / `total_questions` are the perfect (reward >= 1) and evaluable
  simulations of all domains (4 per task by default, less the infrastructure errors);
  `tau2_report.total_simulations` counts every simulation and
  `tau2_report.evaluable_simulations` the evaluable ones.
- `tau2_report.mean_reward` is the mean reward pooled over every simulation that did not end in
  an infrastructure error (tau2's own `avg_reward` leaves those out too), so each domain weighs
  by its task count; it is not a leaderboard number. `tau2_report.mean_reward_by_domain` has it
  per domain. `tau2_report` also carries `infra_errors` (the simulations that ended in an
  infrastructure error, of both kinds below), the models of each role (`user_llm_models` /
  `eval_llm_models`, in order, and `user_llm_source` / `eval_llm_source`: `GBENCH_JUDGE_MODELS`,
  `GBENCH_JUDGE_MODEL`, `default` or the role's knob), the agent sampling,
  `gemini_call_parameters`, `substitute_retries` (whether the Gemini retry patches were active),
  the relays' records (`gemini_relay`, `endpoint_relay`, `gateway`), the tau2 revision and
  version, the image, and the run directory.
- **Which Gemini model answered.** `gemini_models_answered` counts the replies of each model over
  the user simulator and the judge, and `gemini_cascade` is the cascade they were asked from;
  `tau2_report.gemini_models_answered_by_role` and `tau2_report.gemini_models_answered_by_domain`
  split them per role (`user_simulator`, `nl_assertion_judge`) and per domain.
  `tau2_report.nl_judge_models_by_domain` counts, per domain, the NL-assertion verdicts each
  model gave; each verdict's model is `reward_info.info.nl.judge_model` of its simulation in
  `results.json`. `llm_failures` counts transport failures
  (context overflow, timeouts, rate limits) at every failed attempt, including the ones a retry
  or the next cascade model recovered from, so it is not a count of lost simulations.
- **Infrastructure errors are never scored.** tau2's `run_with_retry` re-runs a simulation
  attempt that raised an exception (up to 3 more times) and records a simulation whose every
  attempt raised as an infrastructure error, whatever the cause. `compute_metrics` leaves such a
  simulation out of pass^k, gbench leaves it out of `tau2_report.mean_reward` and of the
  evaluable counts above, counts it in `tau2_report.infra_errors`, makes the status
  `completed_with_errors` instead of `success` and names it in `leaderboard_comparable_reason`.
  gbench reads where the error came from in the traceback tau2 records with it (`sources`, one
  name per simulation):
  - **harness failures**, never the agent's outcome (an outage of the Gemini user simulator or
    judge, of your model endpoint, ...): `model_endpoint` (the request to your model endpoint
    failed: connection, timeout, HTTP error), `user_simulator` (the Gemini user simulator's call
    failed after gbench's Gemini cascade and retries, or it replied with neither text nor tool
    calls), `grading` (tau2's grading after the conversation: the Gemini NL-assertion judge, or
    replaying the conversation's tool calls), `tool` (a tool call raised out of the
    environment), `gemini_embeddings` (a tau3 embedding call that failed after its retries),
    `other` (any other exception, none of it on the agent's turn, such as building the
    environment) and `unrecorded` (no traceback recorded). They are harness errors: one
    `harness_errors` entry per domain, and in `leaderboard_comparable_reason` as `<n>
    simulation(s) ended in infrastructure errors from harness failures, not the agent's outcome
    (<domain> <n> [<source>/<error class> x<n>], ...): tau2's metrics exclude them, so pass^k
    covers fewer than every task x trial`.
  - **raised on the agent's own turn**, not harness failures: `agent_reply` (tau2 rejected your
    model's reply: a message with neither text nor tool calls, or tool-call arguments that are
    not valid JSON) and `agent_context_overflow` (the agent's request was over your endpoint's
    context window, so it depends on the context length the endpoint serves). tau2 leaves them
    out of pass^k as upstream does, so they are not counted against the model; gbench lists them
    apart, one `agent_turn_errors` entry per domain, and in `leaderboard_comparable_reason` as
    `<n> simulation(s) ended in infrastructure errors raised on the agent's own turn (<domain>
    <n> [<source>/<error class> x<n>], ...), not harness failures: tau2's metrics exclude them
    as upstream does, so pass^k covers fewer than every task x trial and does not count these
    simulations against the model`.

  Each entry has `infrastructure_error_simulations`, `sources`, `error_types` (the class of each
  simulation's last error, as tau2 records it in the simulation's `info`; `unknown` when none is
  recorded) and `example_error` (the first error message, redacted and cut to 300 characters).
  A domain whose every simulation ended in an infrastructure error hard-errors
  (`infra_required`), naming the error classes and how many were raised on the agent's own
  turn.
- A simulation tau2 ends with `USER_ERROR` (orchestrator.py `_check_communication_error`: the user
  simulator sent a message with both text and tool calls, or an empty one), which its evaluator
  scores 0, is a failure of gbench's substituted Gemini user simulator, not of the agent: it is
  left out of pass^k and the rewards (pass^k is computed by `compute_metrics` over the other
  simulations), counted per domain (`category_accuracy.<domain>.user_simulator_protocol_errors`,
  `tau2_report.user_simulator_protocol_errors` and `..._by_domain`, summed by `gbench --merge`),
  recorded in `harness_errors` (kind `user_simulator_protocol`) and named in
  `leaderboard_comparable_reason`.
- `harness_errors` lists harness failures that are never the agent's outcome: the
  infrastructure-error simulations from a harness failure above, the user-simulator protocol
  errors, per domain, and (tau3) Gemini embedding calls that failed after their retries, per
  domain (see [tau3.md](tau3.md)).
  `agent_turn_errors` lists the infrastructure-error simulations raised on the agent's own turn,
  per domain. Any entry of either makes the status `completed_with_errors` instead of `success`
  and is named in `leaderboard_comparable_reason`. retail, airline and telecom call no
  embedder.
- `leaderboard_comparable` is `true` only when gbench detects no difference from a leaderboard
  submission; otherwise `leaderboard_comparable_reason` lists each difference it detects: fewer
  than 4 trials, an `--eval-limit` subset, a user simulator other than `gpt-5.2` (recommended by
  the submission guide and used by Sierra's 2026 standard text entries) or upstream's config
  default `gpt-4.1-2025-04-14` (or not at temperature 0.0), an NL-assertion judge other than
  upstream's `gpt-4.1-2025-04-14`, agent sampling other than temperature 0.0 alone (so every
  `--thinking` run, whose agent samples at 1.0), (tau3) the
  Gemini retrieval embeddings and any Gemini embedding call that failed after its retries (see
  [tau3.md](tau3.md)), and simulations that ended in an infrastructure error, the harness
  failures and the ones raised on the agent's own turn apart, per domain with their sources and
  error classes (tau2's metrics leave them out, so pass^k covers fewer than every task and
  trial). A default run uses the Gemini user
  simulator and judge, so its flag is `false`: compare it only with leaderboard entries that used
  the same user simulator.
