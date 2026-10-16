# mcp_bench

Canonical **MCP-Bench** ([Accenture/mcp-bench](https://github.com/Accenture/mcp-bench),
arXiv:2508.20453): a live agentic benchmark over **28 real Model Context Protocol servers**
(about 250 tools). For each task the model discovers the available tools, plans a multi-step
trajectory (sequential + parallel calls), invokes the tools against the live servers, and
grounds a final answer. gbench runs the **real** agent loop; it does not substitute a
single-turn proxy.

## What is measured

The 0-1 **Overall Score** (headline `accuracy`, reported x100) is the mean of four dimensions:

| dimension | source |
| --- | --- |
| **schema understanding** | rule-based: mean(`valid_tool_name_rate`, `input_schema_compliance`, `tool_call_success_rate`) (0-1) |
| **task completion** | judge: mean(task_fulfillment, grounding) |
| **tool usage** | judge: mean(tool_appropriateness, parameter_accuracy) |
| **planning effectiveness** | judge: mean(dependency_awareness, parallelism_and_efficiency) |

Schema understanding groups the three rule-based metrics exactly as the paper's Table 3
(arXiv:2508.20453) does: Valid Tool Name Rate, Schema Compliance and Execution Success
(`tool_call_success_rate` in upstream's `results_aggregator.py`). The paper prints no formula,
but recomputing its 20 leaderboard rows with these three metrics reproduces the printed Overall
Score for 18 rows (the other two are within 0.001, from rounding of the printed columns), while
leaving Execution Success out matches none.

The three judged dimensions come from six sub-dimensions scored 1-10, each judged **5x with
randomized dimension/criterion order and averaged** (upstream's "judge stability" protocol).
gbench keeps that protocol and the prompts **verbatim**; the 1-10 judge dimensions are divided
by 10 to sit on the 0-1 axis with the rule-based dimension.

### Judge: gbench Gemini cascade (not o4-mini)

Upstream judges with Azure **o4-mini**. gbench injects its **Gemini cascade** as the judge,
leaving upstream's 6-dimension prompts and 5x stability loop untouched; only the model underneath
changes, for consistency with every other gbench judged suite. The models are gbench's one
cascade, `base.GEMINI_CASCADE` (`gemini-3.8-flash`, `gemini-3.7-flash`, `gemini-3.6-flash`,
`gemini-3.5-flash`, `gemini-3.1-pro-preview`, tried in order; `GBENCH_JUDGE_MODELS` replaces it),
with the key rotation, rounds and backoff of `base.judge_generate_cascade`, reached through
Gemini's OpenAI-compatible API. gbench always passes the run's list into the container as
`GBENCH_JUDGE_MODELS` (`base.gemini_cascade_env()`), so the judge calls exactly the models the
result records; the default written into `mcp_bench_cascade_judge.py` equals
`base.GEMINI_CASCADE` and is used only when neither `GBENCH_JUDGE_MODELS` nor `GBENCH_JUDGE_MODEL`
is set. The key check before the run also asks the Gemini API for each model of the cascade
(`GET <GEMINI_OPENAI_BASE_URL>/models/<name>`): a model it does not serve for the key is
`infra_required`, naming the model and the variable to set.

**The judge's Gemini key never enters the container.** The model under test drives servers that
run its text there (the Scientific Computing server's sympy `parse_expr`, the OSINT server's
commands), so a key in that container could be read. The judge runs in the container (it is
upstream's evaluator), but it holds only a per-run placeholder of gbench's Gemini relay
(`gbench/runners/eval_suites/gemini_relay.py`): a thread of the gbench process that listens on
the container network's gateway (`172.17.0.1` on the default bridge, `127.0.0.1` with the host
network) on a free port, checks the placeholder, adds the real key (`GEMINI_API_KEYS`, else
`GEMINI_API_KEY`, rotated) and forwards the call to the Gemini API (`GEMINI_OPENAI_BASE_URL` as
`base.gemini_openai_base_url` checks it). It forwards only `POST /v1beta/openai/chat/completions`,
the judge's one call; anything else gets 403, and a request without the placeholder gets 401.
The container gets `GEMINI_API_KEY` and `GEMINI_API_KEYS` (the placeholder),
`GEMINI_OPENAI_BASE_URL` (`http://<gateway>:<port>/v1beta/openai`) and `GBENCH_GEMINI_RELAY_URL`,
by name. While every key it holds is a placeholder, the in-container judge sends only to that
relay URL; with a real key it accepts only the Gemini API over https or a loopback address
(`check_gemini_openai_base_url`, a copy of gbench's), so a real key is never sent to the relay's
address. The result records the relay (`gemini_relay`: where it listened, what it forwards, the
requests it relayed, refused or rejected, and its per-request failures; never a key). A failed
request the cascade got past (for example a model of the list that is not served) is counted there
and is not a task's harness error; a judge evaluation that could not be made is (see
[Harness errors](#harness-errors)). Each judge request carries what upstream's `LLMProvider.get_completion`
(`llm/provider.py`) sends: the model, the system and user messages, and the evaluator's
`max_tokens` (upstream's switch to `max_completion_tokens` is keyed on Azure OpenAI model names
such as o4-mini and does not apply to the Gemini models). **No temperature** is sent, as upstream
sends none. The judge's OpenAI client makes one attempt per request (`max_retries=0`), so a failed
call moves to the next model of the cascade at once (upstream's provider retries its one model 3
times; the cascade's models and rounds take that role), and a model the API answers HTTP 404 for
(not served) is not asked again in that call. The launcher records the Gemini model that answered
each judge call: per task (`judge_models_per_task`, `{task id: {model: replies}}`) and over the
run (`gemini_models_answered`), next to the cascade the run passed (`gemini_cascade`). The
MCP-Bench README requires o4-mini to reproduce leaderboard numbers, so a gbench run is **never
`leaderboard_comparable=True`**.

### The server keys: the credentialed-servers container

The five servers that need one of the README's keys (Google Maps, BioMCP, Hugging Face, National
Parks, NASA Data: the `commands.json` entries whose `env` lists `GOOGLE_MAPS_API_KEY`,
`NCI_API_KEY`, `HF_TOKEN`, `NPS_API_KEY` or `NASA_API_KEY`) do not run in the model container,
where model text runs (the Scientific Computing server's sympy `parse_expr`, whose default
namespace includes Python's builtins, and the OSINT server's commands), so a key there could be
read. gbench runs them in a second container of the same image, the **credentialed-servers
container**:

- it is started with `docker run -d --init --pull never --entrypoint sleep gbench-mcp-bench:7a8eaeae
  infinity` on a docker network of its own (`gbench_mcp_bench_<id>-cred-net`, created for the
  run), with the five keys by name (`-e NAME`) and nothing else of the run: no port, no volume, no
  service, and none of the servers that run model text. Before it starts,
  `container_guard.require_metadata_blocked` checks that network: the cloud metadata server
  unreachable, and the credentialed servers' API hosts plus `pypi.org` and
  `files.pythonhosted.org` (`uv run` fetches BioMCP's and NASA Data's build requirements when it
  starts them, as upstream) resolving;
- in the model container, the launcher points those five `commands.json` entries at gbench's
  bridge (`gbench/docker/mcp_bench_bridge.py`), with the same transport (the HTTP server, Google
  Maps, keeps its port and endpoint), so upstream's runner and connection check start the bridge
  exactly where they would start the server; the rest of `commands.json` is upstream's, and the
  upstream table is kept as `commands.upstream.json`;
- each session of the bridge is one TCP connection to gbench's relay to the credentialed servers
  (a thread of the gbench process on the model container network's gateway, next to the judge's
  relay). The relay checks a per-run token (not a credential of any service; it keeps other
  containers on the network out), accepts only a session to one of the five servers with its
  transport (stdio, or start and connect for Google Maps' HTTP server), and runs `docker exec -i
  <credentialed container> python3 /app/adapter/mcp_bench_credentialed.py <mode> <server>`
  (fixed arguments, no shell). It answers the bridge only once the script's first line says the
  session is up (the stdio server's process started, the HTTP server accepts connections, the
  connection to it is open): `OK`, or `ERR not started: <why>` when the session did not come up
  or docker exec ended without that line (the credentialed container gone, the exec refused, the
  server's command missing) within 30 s (150 s for the HTTP server's start). After `OK` the
  bridge's bytes go to the session unchanged, and what comes back arrives in frames, ending with
  an end frame that carries docker exec's exit code (the script exits 0 however a session that
  came up ends, so another code is a failure of docker exec or of the container);
- in the credentialed container, `gbench/docker/mcp_bench_credentialed.py` starts the server's
  upstream command in its upstream directory with the environment upstream's MCP client gives a
  stdio server (the MCP SDK's default variables plus the server's own key), or, for Google Maps,
  the way upstream's `MCPConnector` starts an HTTP server (a free port, `MCP_SERVER_PORT`), and
  relays the MCP messages. One deviation for the HTTP server: upstream's
  `MCPConnector.start_http_server` (`mcp_modules/connector.py` at `7a8eaeae`) starts it with its
  whole environment (`os.environ.copy()` plus the server's `env` and `MCP_SERVER_PORT`); gbench
  gives it only the MCP SDK's default variables, its own key and `MCP_SERVER_PORT`, so the other
  four servers' keys never reach it. The servers, their tools, descriptions and schemas, and every request
  and reply are upstream's; the only change on the way back is a key value (as is, JSON-escaped or
  URL-encoded) replaced by as many asterisks.

The model container's environment holds no server key: gbench passes none, its entrypoint refuses
to start when one is set, and `mcp_servers/api_key` stays empty.

**The residual risk of the credentialed-servers container.** Its five servers hold their real
keys in their own processes while they execute the model's tool calls, with the model's
arguments, as upstream designed them (an MCP server that calls a keyed API holds its key). No
model-written code runs in that container: the servers that run model text stay in the model
container, which holds no key, and the relay starts only the five servers' fixed commands there.
The residual risk: a flaw in one of the five servers (Google Maps, BioMCP, Hugging Face,
National Parks, NASA Data) that let the model's arguments run code there would reach the five
keys in that container, which has outbound internet access. gbench keeps upstream's design (a
server cannot call its API without its key) and keeps everything else out of that container: no
Gemini key, no endpoint URL, no port, no volume, the cloud metadata server checked unreachable
before it starts, and every key value masked in what comes back.

A credentialed-server session that fails for a reason outside the model is written by the bridge
(`/tmp/gbench-mcp-bench-bridge-failures.jsonl`) with its time, and the task running then is a
harness error (`credentialed_relay`), never the model's outcome:
- the relay did not answer, or refused the session (its credentialed side did not come up: the
  credentialed container stopped or removed, a failed `docker exec`, the server's command could
  not start, the HTTP server did not accept connections);
- the stdio server exited before it sent a byte while the client still used the session (a
  failed start, for example `uv run` not reaching PyPI for BioMCP's or NASA Data's requirements);
- the HTTP server's start did not report it ready;
- the session ended without the end frame, or with a nonzero docker exec exit, while the client
  still used it (lost mid-task, for example the credentialed container stopped).

A session the client ended (upstream closes it at the end of each task) is no failure, and a
server that exits on its own after it answered is its own outcome, as it would be upstream (the
bridge of the HTTP server then closes its listeners and ends, as the server's port would close; a
connection whose session failed while the server was exiting, within 2 s of that exit, is not
recorded as a failure). On gbench's
side, the relay keeps each failure it saw with its time (not opened, or docker exec failing while
the bridge was attached), and after the run gbench asks docker whether the credentialed-servers
container still runs: a relay failure during the run, or a container that is not running at its
end, is a harness error of the run too (`harness_errors` kind `credentialed_relay`, with the
failures and `container_running_at_end`; status `completed_with_errors`, named in the reason),
also when the bridge could not record it. The result records the container and the relay
(`credentialed_servers`: container, network, servers, the key names, the metadata guard, the
names checked, the relay's sessions, refusals, failures by kind and with their times, and whether
the container still ran when gbench stopped it right after the run; never a key or the token),
and `credentialed_servers_run` sums up the run (`sessions_failed`, `failures`,
`container_running_at_end`, `container_state`). Checked without a model on the built image, with placeholder keys: upstream's
connection check reports 28/28 servers connected with tools, the five credentialed servers
through the bridge (Google Maps 7 tools, BioMCP 35, Hugging Face 10, National Parks 6, NASA Data
21), their tool catalogs (names, descriptions and schemas) are identical to the ones the same
servers list when started directly in one container with the keys, and no key value appears in
the check's output.

## How gbench runs it

The agent loop + all 28 servers are **delegated to the upstream runner inside a LOCAL container**
(`gbench/docker/mcp_bench.Dockerfile`). gbench's launcher (baked into the image):

* points the model-under-test at the gbench `/v1` endpoint (upstream's `openai_compatible`
  provider; the agent uses plain `chat/completions`, so **no native tool-calling is required**)
  and applies the run's model request settings to that provider's requests only (see
  [Sampling](#sampling));
* points the five credentialed servers' `commands.json` entries at the bridge to gbench's
  credentialed-servers container (above);
* injects the Gemini cascade judge (neutralizing the hard-coded o4-mini default), which holds only
  the relay's placeholder (above);
* checks again that every server of the task set can serve, runs all 104 tasks (upstream's task
  files as they are) and writes a manifest of what ran. If a server cannot serve by then, it
  runs no task and names the server in the manifest;
* keeps each task a harness error fell in out of upstream's averages and writes the per-task
  record to `/out/harness_log.json` (see [Harness errors](#harness-errors)).

Before starting the run, gbench starts the credentialed-servers container and runs a check in the
image (one `docker run`, with no server key: the five credentialed servers are reached through the
bridge and the relay, as in the run):
1. the image's own launcher (the functions its `main()` uses: `install_credentialed_bridges`,
   `_all_task_servers`, `_available_servers`, `_built_servers`) reports which of the 28 servers it
   can serve: every server built, the network up and each credentialed server's bridge installed.
   The network is up when every host name of the servers' APIs (`SERVER_API_HOSTS` in
   `mcp_bench.py`, one per online server with a fixed API host, which is every online server but
   OSINT Intelligence, whose targets the model names: `api.nasa.gov`, `developer.nps.gov`,
   `maps.googleapis.com`, `huggingface.co`, `en.wikipedia.org` and 16 more; passed as
   `MCP_DNS_NAMES`) resolves in the container, all looked up together within 15 s, and a TCP
   connection to one of them on port 443 opens. There is no test by IP address alone: an address
   test passes on a network whose name resolution is down, where the servers' API calls fail.
   `GBENCH_MCP_BENCH_ASSUME_NETWORK` skips this part; names that do not resolve are listed in the
   gap (`network_check` in the probe's report and in the manifest);
2. MCP-Bench's own connection check, `utils/collect_mcp_info.py` (the README's first step, in its
   default `individual` mode: one server at a time, up to 5 attempts of 30 s), starts every
   server, connects to it and lists its tools. The README expects "28/28 servers connected" and
   "All successfully connected servers returned tools!".

The five required keys are checked in the gbench process before any container starts (after the
Docker, image and Gemini key checks): each key that is set is checked with a live request to the
service its server calls (Google Maps: one per Maps Platform API its server calls), and a key the
service refuses stops the suite with `infra_required` naming the key, its server and the service's
answer ([Checking the keys](#checking-the-keys)); a key that is unset or
empty stops it too, naming each such key and its server (for example `required API keys unset:
NASA_API_KEY (NASA Data)`). Neither part of the image check sees a key: a server whose key is
wrong, or whose Google Cloud project lacks an API the server calls, still starts and lists its
tools, and its tool calls would then return the API's refusal, which upstream scores as the task's
tool results; that is why gbench checks the keys first. The run then starts only when every server
passes both parts; otherwise it hard-errors (`infra_required`) naming the servers (not built, not
servable without egress or a bridge, not connected, or connected without listing tools, and the
credentialed-server sessions that failed, as listed above). The pre-run check takes about 2
minutes when every server connects.
gbench never runs the tasks of the provisioned servers alone: that would be a different task set.
When the run starts, the launcher repeats the first part (the same functions); a server lost in
between (egress gone, no relay to the credentialed servers) makes it run no task, and the suite
ends `infra_required` naming the servers from its manifest.

### Network and the cloud metadata server
The pre-run check and the run use docker's default **bridge** network (`GBENCH_MCP_BENCH_NETWORK`
names another docker network, `host` included). A model endpoint on this host (`127.0.0.1`,
`localhost`, `0.0.0.0`) is reached at that network's gateway (`172.17.0.1` on the default bridge,
the gateway `docker network inspect` reports for another network), so the served model must listen
on that address too (vLLM's `vllm serve` listens on every interface unless `--host` says
otherwise); a remote endpoint is used as given. The judge's Gemini relay listens on the same
gateway. The result records the network in `docker_network`.

The model under test drives tools that run commands on targets it names (the OSINT Intelligence
server runs `nmap`, `whois`, `dig`, `host`, `dnsrecon` and `dnstwist` on them) and parse
expressions it writes (the Scientific Computing server's sympy `parse_expr`), in a container with
network access. On a cloud VM the metadata server at `169.254.169.254` hands out the VM's
service-account token to anything that can reach it. So before the pre-run check, gbench starts one
short-lived probe container of the image on that network (`container_guard.require_metadata_blocked`:
`--pull never`, no environment, no volumes; it only opens and closes a TCP connection to
`169.254.169.254:80`, sends nothing and reads no token). The same probe container resolves the
host names the container needs: the servers' API hosts (`SERVER_API_HOSTS`) and the model
endpoint's host when it is a name rather than an address (recorded in `dns_names_checked`); the
credentialed-servers container's own network is checked the same way before that container
starts (`credentialed_servers.dns_names_checked`). When
the connection opens, or the probe cannot give a verdict, the suite stops with `infra_required`
naming the host firewall rule to add once; when a name does not resolve, it stops with
`infra_required` saying that containers on that network cannot resolve host names (on GCE the VPC
DNS resolver is `169.254.169.254` too, so a rule that drops all traffic to that address also drops
DNS). A DNS outage is found here, before the run, and is never scored. The rule:
```bash
sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save   # keep it across reboots
```
Docker's `DOCKER-USER` chain covers the bridge and user-defined networks, not the host network:
with `GBENCH_MCP_BENCH_NETWORK=host` on a cloud VM the check fails, so use a bridge or
user-defined network there (blocking the address for the whole VM would also cut the VM's own DNS,
guest agent and credentials). gbench never runs sudo or changes the host firewall itself.

## Requirements

- **Docker** + the **`gbench-mcp-bench:7a8eaeae` image, built LOCALLY** (never pulled: gbench starts it with
  `docker run --pull never`). Heavy: builds ~9
  TypeScript + ~19 Python servers from source (tens of minutes; BioMCP/alphagenome, the
  hand-built metmuseum server, and numpy/scipy servers are the slow/fragile ones):
  ```bash
  docker build -t gbench-mcp-bench:7a8eaeae -f gbench/docker/mcp_bench.Dockerfile gbench/docker
  ```
  - It clones Accenture/mcp-bench at the pinned commit
    `7a8eaeae83a842a2949080acc5473f65e1569daf`; the task files are the ones bundled in the
    upstream repo at that commit.
  - It runs every step of upstream's `mcp_servers/install.sh`. The script stops at its first
    failing command (`set -e`), and metmuseum-mcp's `npx tsc` exits non-zero on type errors
    under today's npm packages (it still writes `dist/`) before the Node servers after it and
    every Python server are installed, so the image runs it with that one line switched to
    `set +e`. A build step of its own fails the build when `install.sh` does not have exactly
    that one `set -e` line (upstream drifted). At the end of the installs, a build check
    (`gbench/docker/mcp_bench_build_check.py`) looks at every server `commands.json` starts, in
    its working directory: the built JavaScript entry point (`node dist/...`, `node
    build/...`), the script (present and compiling), the module (importable, as `python -m`
    finds it) or, for a `uv run` server, its `.venv`. It fails the build naming every server
    that is not ready, so a failed install step (or a failed post-install guard, or a failed
    install of the runner's own packages, which no longer tolerate failure) never produces an
    image. A server that is built but does not connect still shows up in the pre-run check,
    and the suite hard-errors until the image is fixed.
  - **MCP client pinned to `mcp==1.16.0`**
    (`gbench/docker/mcp_bench_constraints.txt`, a pip/uv constraint on every install into the
    image's Python). MCP-Bench declares `mcp[cli]>=1.9.0` (`mcp_servers/requirements.txt`) and
    its README expects the connection check to report 28/28 servers connected with tools. The
    pin is the release that range resolved to on the pinned commit's date (2025-10-07), the
    client a README install at that commit got; with it upstream's check reports 28/28 on the
    README's Python 3.10. Unpinned, the range resolves to 2.2.0 today, with which Game Trends
    and Time MCP list no tools; the newest 1.x release (1.30.0) moves fastmcp to 3.4.7, and then
    Game Trends, OSINT Intelligence, Paper Search and Unit Converter list no tools. The same file
    pins the rest of the system interpreter's packages to the set of the image built 2026-09-28, and
    the build fails unless the interpreter holds exactly that set plus the four servers installed
    editable from the checkout (bibliomantic, mcp-reddit, time-mcp, unit-converter), and if the
    installed client is not 1.16.0. Every server `commands.json` starts with `python` runs on
    that same interpreter, as in upstream's single environment; the `uv run` servers use their
    own `uv.lock` environments.
  - `UV_PYTHON_PREFERENCE=system`: uv builds its environments on the image's own Python and
    downloads a managed CPython only for an environment whose `.python-version` or
    `requires-python` the image's 3.10 cannot satisfy: nasa-mcp's (`.python-version` 3.13; a
    `uv run` server, so it runs on it), and the ones `install.sh`'s `uv sync` builds for
    huggingface-mcp-server (`>=3.13`), mcp-nixos (`>=3.11`) and medcalc (`>=3.12`), which get the
    newest managed CPython the pinned uv release (0.12.19) knows, 3.14.7, but do not serve those
    servers: `commands.json` starts
    all three with `python`, the image's 3.10, as it does in the README's conda env. Without the
    preference, once a managed 3.14 has been fetched, wikipedia-mcp and mcp-reddit switch to it
    too, their locked `cffi` 1.17.1 has no wheel for it and its source build fails (no libffi
    headers in the image), and both servers list no tools.
  - The toolchain is pinned: Node.js `22.23.3-1nodesource1` (NodeSource), pip 26.2.1, setuptools
    84.0.0 and wheel 0.48.0 (get-pip.py), uv 0.12.19, and the managed CPython builds uv downloads
    (3.13.15 for nasa-mcp, 3.14.7; the build fails on any other). Every npm resolution of the
    build (install.sh's global pnpm/yarn/tsx and its `npm install typescript --save-dev` steps,
    the post-install guards) takes the newest release published before `NPM_CONFIG_BEFORE`
    (2026-09-26T00:10:00Z, a build argument). The image records the Ubuntu packages (which still
    float with the Ubuntu archive), Node, npm and the global npm tools, and the changes install.sh
    made to the checkout in `/opt/gbench`.
  - Python 3.10, the version the README's install step creates (`conda create -n mcpbench
    python=3.10`): Ubuntu 22.04's `python3`, the only `python`/`python3` in the image, so
    `install.sh`'s ">= 3.10" check passes and it installs no other Python. The runner, the MCP
    client and every server `commands.json` starts with `python` run on it (huggingface-mcp-server,
    mcp-nixos and medcalc included), and so do the `uv run` servers wikipedia-mcp, biomcp and
    mcp-reddit (their uv environments are 3.10); nasa-mcp runs on its 3.13 (above).
  - `tzdata`: a configured local timezone (`/etc/localtime`, Etc/UTC), as any machine the README
    runs on has. Time MCP's server reads it at start-up and exits without one, so it would list
    no tools.
- **`GEMINI_API_KEY`**: the Gemini cascade judge (`GEMINI_API_KEYS`, a key list, is rotated
  through as gbench's judge does). It stays in the gbench process: the relay adds it, the
  container holds a placeholder (see [Judge](#judge-gbench-gemini-cascade-not-o4-mini)). The relay
  listens on the container network's gateway, so this host must accept connections from the
  container there (it does by default). **Hard-errors** (`infra_required`, never skips) if unset,
  if a live check before the run rejects the key or finds a model of the cascade the API does not
  serve to it, or if the relay cannot listen. Where to get it: [The keys](#the-keys).
- **A served model at `/v1`**: the gbench endpoint (`--remote-endpoint`). On the default bridge
  network the container reaches a model on this host at `172.17.0.1`, so the model must listen
  there (see [Network](#network-and-the-cloud-metadata-server)).
- **On a cloud VM, the metadata server blocked for docker networks** (the host firewall rule above);
  without it the suite stops with `infra_required`.
- **Network egress and the five server keys** the MCP-Bench README lists as required
  ("Required API keys include ..."; upstream's loader reads them from `mcp_servers/api_key`, and
  they are exactly the keys the `env` lists of `mcp_servers/commands.json` name at the pinned
  commit). gbench passes them only to the credentialed-servers container, where each of the five
  servers gets its own key as upstream's loader gives it (see
  [The server keys](#the-server-keys-the-credentialed-servers-container)): `NPS_API_KEY`
  (National Parks), `NASA_API_KEY` (NASA Data), `HF_TOKEN` (Hugging Face), `NCI_API_KEY` (BioMCP)
  and `GOOGLE_MAPS_API_KEY` (Google Maps). What each is for, where to get it and what it costs:
  [The keys](#the-keys).
  The other 23 servers need no key: 6 run offline (Bibliomantic, Math, Medical Calculator,
  Scientific Computing, Time, Unit Converter) and 17 need outbound internet with name
  resolution. A missing key, a key its service refuses, a server that did not build, no egress, a
  server API host name that does not resolve or a server that fails the connection check is a
  hard error before the run (see above).
  Every key reaches the credentialed-servers container by name (`docker run -e NAME`, the value
  taken from the environment gbench passes to docker), never as a value on the command line, and
  never the model container. The endpoint URL reaches the model container by name too; it may
  carry no user name or password (such a URL is `infra_required`: it would sit in the container
  where model text runs; serve the model at a URL without user info). Container output quoted in
  an error message has every credential value replaced by `[REDACTED]`.
- The image carries the label `org.gbench.model-request-policy=2`: it applies the run's model
  request settings, the request timeout of a `--thinking` run included, and counts failed
  requests (`gbench/docker/mcp_bench_request_policy.py`, see [Sampling](#sampling)). It also
  carries `org.gbench.mcp-bench-harness=3`: its judge takes the relay's placeholder and URL, the
  five credentialed servers run in the credentialed-servers container behind the bridge
  (`mcp_bench_bridge.py`, `mcp_bench_credentialed.py`, speaking the session protocol above: the
  credentialed side says when a session is up, the stream ends with an end frame, and the bridge
  records every session that failed outside the model) and its model container refuses a server
  key, its network check resolves the servers' API host names, and its launcher keeps each task's
  harness errors out of the averages and writes `harness_log.json` (layout version 2, with the
  Gemini models that answered each task's judge calls; `gbench/docker/mcp_bench_run.py`,
  `mcp_bench_cascade_judge.py`). An image built from an older Dockerfile lacks either label (or
  carries another value), and the suite hard-errors (`infra_required`) until it is rebuilt: an
  older image would hold the server keys, or a real Gemini key, next to the model-driven servers.

The run covers all **104 tasks** (56 single-server + 30 two-server + 18 three-server);
`leaderboard_comparable` is always False (the judge).

### The keys

Set each in the environment gbench runs in (`export NAME=...`). The Gemini key stays in the gbench
process (the judge's relay adds it), and the five server keys reach only the credentialed-servers
container, by name. Plans and limits are the providers' own on 2026-09-28; check their pages for
current terms.

| variable | used by | what for | where to get it | cost |
| --- | --- | --- | --- | --- |
| `GEMINI_API_KEY` (or the list `GEMINI_API_KEYS`) | gbench's Gemini relay, for the in-container judge | MCP-Bench's LLM judge (o4-mini upstream), run on gbench's Gemini cascade | Google AI Studio: <https://aistudio.google.com/apikey> | the Gemini API has a free tier with lower rate limits and a paid tier billed to a Google Cloud billing account; every model of the cascade must be served to the key (checked before the run) |
| `NPS_API_KEY` | National Parks (`mcp-server-nationalparks`) | the National Park Service API, `developer.nps.gov` (header `X-Api-Key`) | the form at <https://www.nps.gov/subjects/developer/get-started.htm> | free. The API also accepts the shared `DEMO_KEY`, with the low limits given for NASA below: use your own key |
| `NASA_API_KEY` | NASA Data (`nasa-mcp`) | NASA's open APIs, `api.nasa.gov` (query parameter `api_key`) | the sign-up form at <https://api.nasa.gov/> | free; 1,000 requests an hour per key. Do not use the shared `DEMO_KEY`: it allows 30 requests an hour and 50 a day per IP address, and calls past that fail as tool errors |
| `HF_TOKEN` | Hugging Face (`huggingface-mcp-server`) | upstream's setup requires it, so gbench requires a token the Hub accepts; the pinned server calls the Hugging Face Hub API (`huggingface.co`) unauthenticated and never sends it (it builds an `HfApi` client that would read it but never calls it) | a free Hugging Face account, <https://huggingface.co/settings/tokens>; a token with the Read role is enough | free |
| `NCI_API_KEY` | BioMCP (`biomcp`) | the NCI Clinical Trials Search API v2, `clinicaltrialsapi.cancer.gov` (header `x-api-key`) | <https://clinicaltrialsapi.cancer.gov/signin>; the MCP-Bench README notes that this site may need a US IP address to open (its issue #10) | free, per the MCP-Bench README |
| `GOOGLE_MAPS_API_KEY` | Google Maps (`mcp-google-map`) | Google Maps Platform web services, `maps.googleapis.com`: the server calls `geocode`, `reverseGeocode`, `placesNearby`, `placeDetails`, `distancematrix`, `directions` and `elevation` of `@googlemaps/google-maps-services-js` | a Google Cloud project with billing enabled (<https://developers.google.com/maps/get-started>); in it, enable the **Geocoding API**, **Elevation API**, **Places API**, **Directions API** and **Distance Matrix API**, then create the key under APIs & Services > Credentials. Google lists Places API, Directions API and Distance Matrix API as Legacy since 2025-03-01 (<https://developers.google.com/maps/legacy>) and advises keeping them disabled on new projects, but the server calls these, not their replacements (Places API (New), Routes API), so they must be enabled; a key restricted to other APIs gets `REQUEST_DENIED` | needs a Google Cloud billing account; Google Maps Platform bills requests beyond its free monthly usage |

Other variables some vendored servers read are not part of upstream's setup: they are in neither
`mcp_servers/api_key` nor any `commands.json` `env` list, so upstream runs those servers without
them, and gbench passes none of them to either container: `SEMANTIC_SCHOLAR_API_KEY` (Paper Search
queries Semantic Scholar unauthenticated), `OPENFDA_API_KEY`, `ALPHAGENOME_API_KEY` and
`CBIO_TOKEN` (optional in BioMCP). Weather Data's server carries its own weatherapi.com key in its
code, and Movie Recommender's its own TMDB (`api.themoviedb.org`) key, as upstream.

### Checking the keys

gbench checks every set server key itself before the run, in the gbench process and before any
container starts (`server_key_problems` in `mcp_bench.py`): one fixed, read-only request with the
key to the service its server calls, sent the way the server sends it.

| key | request | accepted |
| --- | --- | --- |
| `NPS_API_KEY` | `GET https://developer.nps.gov/api/v1/parks?limit=1`, header `X-Api-Key` | HTTP 200 |
| `NASA_API_KEY` | `GET https://api.nasa.gov/neo/rest/v1/stats?api_key=<key>` | HTTP 200 |
| `HF_TOKEN` | `GET https://huggingface.co/api/whoami-v2`, header `Authorization: Bearer <token>` | HTTP 200 |
| `NCI_API_KEY` | `GET https://clinicaltrialsapi.cancer.gov/api/v2/trials?size=1`, header `x-api-key` | HTTP 200 |
| `GOOGLE_MAPS_API_KEY` | one request to each Maps Platform API its server calls (`geocode`, `elevation`, `place/nearbysearch`, `directions`, `distancematrix` under `https://maps.googleapis.com/maps/api/`, the key as `key=`) | HTTP 200 with `status` `OK` or `ZERO_RESULTS`, for all five |

A refused key stops the suite with `infra_required`, naming the key, its server and what the
service answered, with every key value masked (a key that is unset is named as well). The replies
to a refused key, checked against the live services with a placeholder key: National Parks and
NASA answer `403` (`API_KEY_INVALID`), Hugging Face and NCI `401`, and Google Maps `200` with
`REQUEST_DENIED` and a message that names the cause (an invalid key, billing not enabled, or an
API not enabled for the project). A check that gets no answer of the service's own (a network
error or timeout, HTTP 5xx or 429, Google Maps `OVER_QUERY_LIMIT` or `UNKNOWN_ERROR`) is sent
again, up to 3 attempts 2 s apart, and stops the suite if it still has none. The Hugging Face
check only confirms that the Hub accepts the token, since the pinned server never sends it. The
result records the check in `server_key_check`.

To check the keys before starting gbench, run the same requests on the host (the keys are read
from the environment, so none is on a command line):

```bash
python3 - <<'PY'
import json, os, urllib.error, urllib.request

def get(url, headers=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}),
                                    timeout=30) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except (urllib.error.URLError, OSError) as e:      # no answer: DNS, timeout, refused
        return type(e).__name__, ""

k = {n: os.environ.get(n, "") for n in ("NPS_API_KEY", "NASA_API_KEY", "HF_TOKEN",
                                       "NCI_API_KEY", "GOOGLE_MAPS_API_KEY")}
print("NPS_API_KEY", get("https://developer.nps.gov/api/v1/parks?limit=1",
                         {"X-Api-Key": k["NPS_API_KEY"]})[0])
print("NASA_API_KEY", get("https://api.nasa.gov/neo/rest/v1/stats?api_key=" + k["NASA_API_KEY"])[0])
print("HF_TOKEN", get("https://huggingface.co/api/whoami-v2",
                      {"Authorization": "Bearer " + k["HF_TOKEN"]})[0])
print("NCI_API_KEY", get("https://clinicaltrialsapi.cancer.gov/api/v2/trials?size=1",
                         {"x-api-key": k["NCI_API_KEY"]})[0])
for q in ("geocode/json?address=Paris", "elevation/json?locations=48.8584,2.2945",
          "place/nearbysearch/json?location=48.8584,2.2945&radius=100",
          "directions/json?origin=Paris&destination=Lyon",
          "distancematrix/json?origins=Paris&destinations=Lyon"):
    status, text = get(f"https://maps.googleapis.com/maps/api/{q}&key=" + k["GOOGLE_MAPS_API_KEY"])
    try:
        body = json.loads(text)
    except ValueError:
        body = {"status": f"no JSON reply ({status})"}
    print("GOOGLE_MAPS_API_KEY", q.split("/")[0], body.get("status"), body.get("error_message", ""))
PY
```

Each of the first four lines should print `200`, and each Google Maps line `OK` (`ZERO_RESULTS`
also means the key works). A line that prints an exception class name got no answer (DNS, a
timeout, a refused connection).

## Environment knobs

| var | effect |
| --- | --- |
| `GBENCH_MCP_BENCH_IMAGE` | override the image name (default `gbench-mcp-bench:7a8eaeae`) |
| `GBENCH_MCP_BENCH_DISTRACTION` | `--distraction-count` (upstream default 10; 0 disables distraction servers); another value is named in `protocol_overrides` and `leaderboard_comparable_reason` |
| `GBENCH_MCP_BENCH_DISABLE_STABILITY` | run the judge once instead of the 5x stability average; named in `protocol_overrides` and `leaderboard_comparable_reason` |
| `GBENCH_MCP_BENCH_ASSUME_NETWORK` | skip the launcher's network check (the name lookups and the connection) and assume the network is up, in the pre-run check and in the run; the connection check still runs, and so does gbench's own name check before the container starts |
| `GBENCH_MCP_BENCH_NETWORK` | the container's docker network (default `bridge`); the metadata-server check runs on it, and a model endpoint on this host is reached at its gateway |
| `GBENCH_MCP_BENCH_TIMEOUT_S` | container timeout (default 24h; full runs are long) |
| `GBENCH_JUDGE_MODELS` / `GBENCH_JUDGE_MODEL` | the judge cascade (default `base.GEMINI_CASCADE`); gbench resolves it (`base.judge_cascade()`) and always passes the list into the container as `GBENCH_JUDGE_MODELS` |
| `GBENCH_JUDGE_CASCADE_ROUNDS` / `GBENCH_JUDGE_BACKOFF` | passed through to the in-container cascade judge when set: the cascade's rounds (default 3) and backoff (default 1.0 s), the same knobs gbench uses |

A pre-run check that finds a gap always ends the suite with `infra_required`; no variable runs
the tasks anyway.

## Run

1. Build the image once (from the repository root; see [Requirements](#requirements)):
   `docker build -t gbench-mcp-bench:7a8eaeae -f gbench/docker/mcp_bench.Dockerfile gbench/docker`.
2. On a cloud VM, add the host firewall rule once (see
   [Network](#network-and-the-cloud-metadata-server)).
3. Serve the model so the container reaches it: a server on this host must listen on the docker
   bridge gateway (`172.17.0.1`) too; `vllm serve` listens on every interface unless `--host`
   says otherwise. Without `--remote-endpoint`, the server gbench launches for a run with eval
   suites listens on every interface (unless `GBENCH_SERVER_HOST` says otherwise).
4. Set the keys ([The keys](#the-keys)) and, optionally, check them
   ([Checking the keys](#checking-the-keys)).
5. Run (the model id is an example; any served model):

```bash
export GEMINI_API_KEY=...            # or GEMINI_API_KEYS=k1,k2; the judge
export NPS_API_KEY=... NASA_API_KEY=... HF_TOKEN=... NCI_API_KEY=... GOOGLE_MAPS_API_KEY=...
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals mcp_bench --max-output-tokens 65536
```

`--max-output-tokens` is required for every eval run. Without `--thinking` it is not applied
(recorded in `max_output_tokens_note`): upstream's own caps are sent exactly. With `--thinking`
it is every model-under-test request's budget (see [Sampling](#sampling)). Add
`--eval-limit 10` for a quick check: `--eval-limit N` runs N tasks, a seeded draw of the (shard's)
task ids balanced across the server groups (the launcher's copy of `sampling.stratified_indices`,
seed `mcp_bench`), not the first N in task-file order; the manifest records it
(`subset_manifest.limit_draw`, `selected_task_ids`), and gbench refuses an image whose launcher
does not report it (rebuild the image). The result
payload carries `overall_score` (0-1), the four `dimensions`, `per_file`, `tasks_run`, the full
`raw_metrics`, the `subset_manifest` (available/unavailable servers, kept/dropped counts, the
launcher's `network_check`), the `server_connection_check` counts (servers checked, servers
connected with tools, tools listed), `tasks_scored`, `tasks_unscored`, `judge_calls_failed`,
`dns_watchdog`, the `protocol_overrides` (empty on a run with upstream's settings), the model
request record (`thinking`, `official_sampling`, `max_output_tokens_note`,
`model_request_settings`, `model_replies`, `model_failed_by_error`, `model_max_tokens_sent`,
`model_request_timeout_s`), `harness_errors`, the container's `docker_network` and
`dns_names_checked`, the judge label, the Gemini models that answered the judge
(`gemini_models_answered`, `judge_models_per_task`) and the cascade (`gemini_cascade`), the
`gemini_relay` record and the `credentialed_servers` record. Start small: the multi-server
agent loop + 5x judge stability makes a full run long.

## Sampling

The upstream agent loop builds the model-under-test's requests as upstream does
(`llm/provider.py` `LLMProvider.get_completion`: the model, the system and user messages and a
`max_tokens`; no temperature). The image's request settings
(`gbench/docker/mcp_bench_request_policy.py`, applied by the launcher to the provider upstream's
`LLMFactory` creates for the model under test, never to the judge) change only these fields:

- **Thinking switch**: every request carries `chat_template_kwargs.enable_thinking` (in
  `extra_body`) set to the run's mode, `true` with `--thinking`, `false` without.
- **Temperature**: gbench's rule for every suite: 1.0 with `--thinking`, 0.0 without. Upstream
  sends none; gbench always sends one (`MCP_TEMPERATURE`, which the image's request settings put
  on every model-under-test request). `--temperature`, or `GBENCH_MCP_BENCH_TEMPERATURE` (which
  takes precedence), sets another in either mode. The temperature is recorded under
  `official_sampling` and named in `protocol_overrides` and `leaderboard_comparable_reason`.
- **max_tokens** (the token-limit policy, `resolve_token_budget` in
  `gbench/runners/eval_suites/base.py`): without `--thinking` upstream's own caps are sent exactly
  (`config/benchmark_config.yaml` at the pinned commit, read by `agent/executor.py`:
  `llm.planning_tokens: 12000` for the planning rounds and the final answer,
  `summarization_max_tokens: 10000` for tool-result summaries, `format_conversion_tokens: 8000`
  for JSON format fixes, and a compression call's target length; upstream's own reductions after
  a context error apply). With `--thinking` those caps, written for non-reasoning generation, are
  not applied (`official_sampling.max_tokens_applied: false`, "official cap not applied to
  reasoning" in `max_output_tokens_note`): every request is sent the run's budget
  (`--max-output-tokens`, else the 32,768 thinking floor), cut to the room the served context
  window leaves after the prompt (the server's `/tokenize` count; without that endpoint, a
  rejection that states the prompt's token count, or a `max_tokens` larger than the whole
  window, is read and the request re-sent with less, while a rejection that states only the
  prompt's character count is not refitted and is counted as a failed request). A prompt that
  leaves less than 256 tokens is sent as it is, and upstream's own handling of the server's
  rejection applies.
- **Request timeout**: with `--thinking` every request carries a timeout sized to the budget as
  gbench's own requests are (`request_timeout_s` in `base.py`: `GBENCH_REQUEST_TIMEOUT_S`, default
  1200, plus the budget divided by `GBENCH_MIN_DECODE_TOK_S`, default 8; 9392 s for a 65,536
  budget) to read the reply, and 5 s to connect as in the openai client's default. The upstream
  client builds `AsyncOpenAI` with no timeout, so the openai default (600 s to read the reply,
  2 retries) would abandon a reply that decodes for longer, which at 25 tokens/s per sequence is
  15,000 tokens. The value is recorded in `model_request_timeout_s`. Without `--thinking` the
  client's own timeout applies (`model_request_timeout_s: null`). MCP-Bench's own
  `task_timeout: 5000` (seconds, `config/benchmark_config.yaml`) still bounds a whole task.

The image appends the settings it applied and the counts of the model-under-test's requests to
`/out/model_requests.json`; gbench records them as `model_request_settings`, `model_replies`
(`replies`, `cut_at_max_tokens`: stopped at max_tokens, `cut_in_reasoning`: stopped there with no
answer text, `fitted_to_context`, `refitted_after_rejection`, `failed_requests`, `timed_out`,
`rejected_over_context`, `cancelled`), `model_failed_by_error` and `model_max_tokens_sent`.
Settings that differ from the ones the run asked for are `infra_required`, and so is a missing
record in a `--thinking` run; without `--thinking` a missing record is named in
`leaderboard_comparable_reason`. With `--thinking`, replies cut at the run's budget are named in
`leaderboard_comparable_reason`; without it, a reply cut at one of upstream's caps is upstream's
own outcome and is only counted.

A request that ends in an error which reaches upstream's `get_completion` is counted as follows:

- `failed_requests`: an error of the served endpoint or of the connection (a timeout, a 5xx, a
  401/404, a refused connection, any other rejection), and a context-window rejection of the
  `--thinking` budget that the refits did not resolve. These are harness errors, never the model's
  outcome: `timed_out` counts the timeouts among them, `model_failed_by_error` counts them by
  exception class and HTTP status (for example `APITimeoutError x2`, `InternalServerError 503
  x1`; no message text), `harness_errors` carries one entry with the counts, the reason names
  them, and the status is `completed_with_errors` instead of `success`. When not one request of
  the run got a reply, every outcome is a harness error: the run hard-errors (`infra_required`)
  and publishes no score.
- `rejected_over_context`: the served window rejected prompt + `max_tokens` where upstream's own
  cap was sent (no `--thinking`), or where `/tokenize` reported that the prompt alone leaves less
  than 256 tokens. Upstream's own handling applies (its token reductions and compression), as it
  would upstream on this window; the reason names the count.
- `cancelled`: a request abandoned in flight when upstream cancelled its coroutine (for example
  at the task timeout); the reason names the count.

The openai client's own retries (up to 2) happen before the error reaches the count, and each
of upstream's 3 `get_completion` attempts that fails is counted.

The Gemini cascade judge sends no temperature, as upstream's judge call sends none; its requests
carry only the model, the messages and the evaluator's `max_tokens`, one attempt per model
(`max_retries=0`; see [Judge](#judge-gbench-gemini-cascade-not-o4-mini)).

## Harness errors

A task's outcome is scored only when no harness error fell in it. The launcher
(`gbench/docker/mcp_bench_run.py`, `HarnessAccounting`) follows upstream's runner, which runs the
tasks one after another, and flags a task with:

| kind | when |
| --- | --- |
| `model_endpoint` | the last attempt of a model-under-test `get_completion` call ended in an error of the endpoint's side or the connection's (`failed_requests` above): upstream's 3 attempts did not get past it. The attribution is per call: a call whose last attempt got an empty reply, or a context-window rejection left to upstream's handling, is the model's outcome, whatever an earlier attempt or a concurrent call did |
| `judge_outage` | upstream completed the task but its evaluator returned no evaluation (every judge stability run failed); upstream's aggregator skips such a result, so the task would silently drop out of the averages: gbench counts and names it |
| `dns_failure` | the task's error, or a tool call's error reply (not a successful tool result, whose text may quote anything), shows a resolver that did not answer (`EAI_AGAIN`, "Temporary failure in name resolution", "no servers could be reached") naming one of the servers' API host names or the container's resolver, or a lookup of one of the servers' API host names that failed ("ENOTFOUND", "Name or service not known", ...). A name the model chose that does not exist or whose lookup fails (SERVFAIL also reads as `EAI_AGAIN`), or a `dig @<server the model chose>` that finds no server, is the model's outcome |
| `dns_outage` | the launcher's DNS watchdog, which looks up the servers' API host names every 30 s while the tasks run, found at least half of them unresolvable while the task ran |
| `credentialed_relay` | a session to a credentialed server failed outside the model while the task ran: not opened, its stdio server exited before it answered, the HTTP server did not report ready, or the session was lost or ended by a docker exec failure while in use (see [The server keys](#the-server-keys-the-credentialed-servers-container)) |
| `task_failed` | upstream recorded the task as failed (for example no tools discovered, a server crash, a timeout after its retries); upstream's own averages leave such a task out |
| `task_raised` | the task's execution raised |

Upstream's aggregator (`ResultsAggregator.aggregate_current_metrics`, which produces the metrics
upstream writes) gets the other tasks only, and the launcher writes each task's flags to
`/out/harness_log.json`. gbench reads it: the Overall Score covers `tasks_scored` (per task file,
weighted by the file's scored tasks; a file with none left out), `tasks_unscored` counts the others
per kind with their ids, `harness_errors` carries one entry for them, the reason names them and
the status is `completed_with_errors`. When no task is left to score, the run hard-errors
(`infra_required`) and publishes no score. A missing or unreadable `harness_log.json`, or one
that lists no task although the container wrote results, is `infra_required` too.

A partial judge outage is a harness error of the run too: `judge_calls_failed` counts the judge
calls of scored tasks that failed on every model of the cascade (`judge_calls_failed_tasks` names
the tasks). Upstream's evaluator averages the stability runs of those tasks that succeeded, and
that average stays their score, as upstream computes it; `harness_errors` carries an entry of kind
`judge_outage_partial` with the count and the task ids, the reason names them, and the status is
`completed_with_errors`. `dns_watchdog` records the watchdog's rounds, outages and the names that
failed.

A credentialed-server session that failed on gbench's relay during the run, or a
credentialed-servers container that is not running at its end, is a harness error of the run:
`harness_errors` carries an entry of kind `credentialed_relay` (`sessions_failed`, the first
`failures` with their times, `container_running_at_end`, `container_state`), the reason names it,
and the status is `completed_with_errors`; the tasks the bridge recorded such a session in are
already left out of the score (above).

The bridge's failure file sits in the model container, where the model-driven servers run, so
gbench cross-checks it with its own relay: when tasks are left out as `credentialed_relay` while
the relay recorded no session failure during the run and the credentialed-servers container still
ran at its end, the result counts them (`unconfirmed_bridge_failures`) and the reason names them.
Such a record can be a session that never reached the relay (it did not answer) or one written by
a process in the model container; gbench cannot tell the two apart, and those tasks stay out of
the score (the image's aggregation leaves them out).

## Reading the result

`accuracy` is the Overall Score x 100. `leaderboard_comparable` is always `false`, and
`leaderboard_comparable_reason` names why: the judge is gbench's Gemini cascade instead of
o4-mini, plus, when applicable, the `protocol_overrides` (a distraction count other than 10,
judge stability off, the temperature gbench sends, which upstream does not), replies cut at a `--thinking` budget, model
requests that failed (harness errors; the status is then `completed_with_errors`), tasks left out
of the score because a harness error fell in them, failed judge calls of scored tasks (a partial
judge outage, which also makes the status `completed_with_errors`), requests
the context window rejected or the harness cancelled, a missing model request record, an
`--eval-limit` or a `--shard`. `is_full` is `false` whenever the run
was not the full task set as upstream runs it (`--eval-limit` or `--shard`).

`--shard I/N`: gbench validates the shard (a malformed `GBENCH_SHARD` is an error before anything
starts) and passes it into the container normalized; the launcher keeps shard I of the sorted
task ids (every N-th starting at the I-th) before `MCP_TASK_LIMIT`, and stops on a malformed value
itself. The result records `shard`. A shard that holds no task reports no score (`accuracy` null,
`empty_shard`), never a 0 or an error. The Overall Score is a mean over per-task judge scores in
`raw_metrics`, which gbench does not yet merge across shards: `gbench --merge` refuses mcp_bench
shards rather than average their scores.

MCP-Bench's runner runs the tasks one at a time (`execution` says so); `--batch-sizes` /
`--eval-concurrency` do not change it. `--attempt-count` is not applied (each task runs once; the
judge stability passes re-grade that one run) and a value above 1 is recorded under `attempts`.
