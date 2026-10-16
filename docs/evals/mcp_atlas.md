# mcp_atlas

MCP-Atlas (Scale AI, [arXiv:2602.00933](https://arxiv.org/abs/2602.00933)): single-turn requests
that need 3 to 6 tool calls across real Model Context Protocol servers (GitHub, WHOIS, search,
Notion, Slack, databases, a filesystem, ...). The answer is graded against expert-written
claims (GTFA_CLAIMS) that a complete answer must state. It measures multi-server tool
orchestration and grounded answering.

gbench runs the released harness, [scaleapi/mcp-atlas](https://github.com/scaleapi/mcp-atlas)
at commit `f24ba3fb0bfa484c86acb28431fad6d7282455f9`: its agent environment (built locally),
its agent-loop limits and tool-call handling, and its claim scorer. The one substitution: the
**search tools** of the release's four search-engine servers (brave-search, ddg-search, exa,
oxylabs) are served by Gemini Google Search grounding behind the same tool names and argument
schemas (see [Search engines](#search-engines-gemini-grounding-for-the-search-tools)). Their
**page tools** are not search and run as upstream, on the release's own servers with the
release's own credentials: ddg-search's `fetch_content` (no key), exa's `web_fetch_exa`
(`EXA_API_KEY`) and oxylabs' `universal_scraper` and `amazon_product_scraper`
(`OXYLABS_USERNAME` / `OXYLABS_PASSWORD`). Results are not leaderboard-comparable.

## Dataset
- `ScaleAI/MCP-Atlas`, split `train` (the 500 public tasks, CC-BY-4.0), pinned to revision
  `8c563b55d7c967755f474299848049834d624617`. Columns used: `TASK`, `PROMPT`, `ENABLED_TOOLS`
  (7 to 37 tool names per task, parsed with `run_eval.py`'s own `parse_enabled_tools`),
  `GTFA_CLAIMS`.
- A task with no prompt or no claims is a hard error (none exist at the pinned revision).
- `--eval-limit N` draws a seeded subset of N tasks.

## Agent environment (the MCP servers)
The release's `services/agent-environment` image (version 1.2.7): its MCP servers behind one
HTTP API on port 1984. The release configures 36; gbench's image runs 35 of them, all but
brave-search, whose two tools are both search tools and are served by gbench (next section).
ddg-search, exa and oxylabs run, for their page tools. gbench never pulls a prebuilt image (it
starts its containers with `docker run --pull never`). `docker/mcp_atlas.Dockerfile` is the
release Dockerfile, step by step, plus five marked additions:
- the base is pinned to the exact base of the published 1.2.7 image
  (`ghcr.io/astral-sh/uv:0.9.30-python3.12-bookworm-slim`, by digest), and the source is a clone
  of the pinned commit;
- unpinned dependencies resolve as of the published image's build time,
  `2026-07-17T19:29:31Z` (`UV_EXCLUDE_NEWER` and `NPM_CONFIG_BEFORE` for the server installs and
  the code-executor workspace, `/root/.config/uv/uv.toml` and `/root/.npmrc` for the servers
  that uvx/npx resolve when the container starts). The release pins each server package but
  not its dependencies. Resolved today, the Python servers get `mcp` 2.x (released
  2026-07-28), which removed `mcp.server.fastmcp` and renamed `McpError`: calculator and fetch
  fail to start, and pubmed imports the removed module. With the cutoff all three start and
  answer. `UV_NO_CONFIG=1` keeps the environment's own `uv run` on its committed `uv.lock`;
- the label `gbench.mcp_atlas.revision` records the commit. gbench refuses an image built from
  another commit;
- brave-search is removed from the server template (`mcp_server_template.json`, which
  `entrypoint.sh` renders into the config; the environment starts only servers it lists), and
  the label `gbench.mcp_atlas.search_servers=removed:brave-search` records it. ddg-search
  (`duckduckgo-mcp-server==0.5.0`), exa (`exa-mcp-server@3.2.1`) and oxylabs
  (`oxylabs-mcp==0.4.1`) stay in the template, as the release has them. gbench refuses an image
  without that label (one built by an earlier gbench, which also removed exa and oxylabs);
  rebuild it with the command under Requirements. The release's install step still pre-installs
  brave-search's package; nothing starts it.
- the release's Node.js install takes NodeSource's 20.x build of the checked image
  (`20.20.2-1nodesource1`, an apt pin written before the release's own `apt-get install -y
  nodejs` step, which stays verbatim; the build fails on another build), and the build records the
  Debian packages it installed (which float with the Debian archive, as the release's do), Node
  and npm in `/opt/gbench`.

At the pinned commit the environment differs from the published 1.2.7 image only by a
starlette 1.0.1 bump in its `uv.lock`.

**Two containers, and the credentials in only one of them.** The model under test runs shell
commands, code and file operations in the environment (`cli-mcp-server_run_command`,
`mcp-code-executor_execute_code`, `mcp-server-code-runner_run-code`, desktop-commander,
filesystem), and in the release's single container those tools can read every server's key and
token (the process environment, the rendered `mcp_server_config.json`, the servers' command
lines). So each run starts two containers of the same image, each told which servers to start by
the environment's own `ENABLED_SERVERS` variable (`mcp_client.py`'s explicit mode):
- the **environment container** runs the 20 servers that need no credential
  (`KEYLESS_SERVERS`: arxiv, calculator, cli-mcp-server, clinicaltrialsgov-mcp-server, context7,
  ddg-search, desktop-commander, fetch, filesystem, git, mcp-code-executor,
  mcp-server-code-runner, memory, met-museum, open-library, osm-mcp-server, pubmed, weather,
  whois, wikipedia; gbench sends ddg-search only `fetch_content` calls), gets no
  credential at all, and is published on the host as `docker run -d -p 127.0.0.1:<free
  port>:1984` on a docker network of its own (`gbench-mcp-atlas-<id>-net`);
- the **credentialed-servers container** runs the 15 that need one (`CREDENTIALED_SERVERS`:
  airtable, alchemy, e2b-server, exa, github, google-maps, google-workspace, lara-translate,
  mongodb, national-parks, notion, oxylabs, slack, twelvedata, weather-data) with the variables
  listed under [Requirements](#requirements), passed from gbench's environment with `-e NAME` so
  the values never appear on the command line. gbench sends exa and oxylabs only their page-tool
  calls (and the pre-flight's one check call each, below). It runs on a docker network of its own (`gbench-mcp-atlas-<id>-cred-net`), publishes no
  port, and its API listens on the container's own loopback (the image's command with
  `--host 127.0.0.1`), which gbench reaches through `docker exec` and curl. No server there runs
  a local shell, code or file tool, and no other container or host process reaches its API over
  the network. `e2b-server` (`@e2b/mcp-server` 0.2.0) runs the model's code remotely, in E2B's
  cloud sandboxes (`Sandbox.create()` through the E2B API), which hold no key of this run; exa and
  oxylabs send the model's URLs and queries to the Exa and Oxylabs APIs, which fetch the pages
  from their own infrastructure. Every reply from this container reaches the model with each
  credential value of gbench's environment masked (`[REDACTED]`), so a key a server echoed into
  a reply never enters the model's context. The one server tool that builds a shell command from
  its arguments, `lara-translate_import_tmx` (`curl -L "<tmx_url>"`), gets an error reply from
  gbench for a `tmx_url` that is not a plain http(s) URL (a double quote, `$`, a backquote, a
  backslash, whitespace or a control character, or no `http://`/`https://` in front); no task
  enables that tool, and a plain URL is passed on as upstream passes it. The `github` and
  `weather-data` servers are fetched unpinned from `github.com/geobio` when the environment
  starts and were not inspected.

**The residual risk of the credentialed-servers container.** Its 15 servers hold their real keys
and tokens in their own processes while they execute the model's tool calls, with the model's
arguments, as upstream designed them (an MCP server that calls a keyed API holds its key). No
model-written code runs in that container: no server there runs a local shell, code or file tool,
`e2b-server` runs the model's code in E2B's cloud sandboxes, and gbench does not pass on the one
tool argument a server would put into a shell command (above). The residual risk: a flaw in one
of those servers (the unpinned, uninspected `github` and `weather-data` servers included) that
let the model's arguments run code there would reach every key in that container, which has
outbound internet access. gbench keeps upstream's design (the servers cannot call their APIs
without their keys) and keeps everything else out of that container: no Gemini key, no
`BRAVE_API_KEY`, no model endpoint, no published port, its API on the container's own loopback,
the cloud metadata server checked unreachable before it starts (below), and every credential
value masked in the replies that reach the model.

gbench sends each tool call to the container that runs its server (after `main.py`'s name
mapping, so `MongoDB_find` goes to mongodb), lists the two catalogs as one environment lists them
(by server in template order, each server's tools in its own order), and runs the pre-flight
check over both. The tools, their names, schemas and replies are the release's (the search tools
aside, next section); only where each server runs changes. Both containers are removed at the end, so memory, filesystem and git state
start fresh every run. `BRAVE_API_KEY` and the Gemini key are passed to neither.

The `mongodb` server reads the release's sample database (`video_game_store`, 6 collections,
16,764 documents), which `data_exports/README.md` has the operator restore with `mongorestore`
into their own MongoDB Atlas cluster. When `MONGODB_CONNECTION_STRING` is not set, gbench
provisions that data itself: the `mongo` target of the same Dockerfile is MongoDB 8.0.32 (the
official image, pinned by digest) with the release's
`data_exports/mongo_dump_video_game_store-UNZIP-FIRST.zip` (sha256
`83f2919e33d5237f31a5f12362038f4a23e648701d4cfc190f0dd53cfc2f7453`, checked at build time)
restored. Each run starts it on the credentialed servers' docker network, and the
credentialed-servers container (where the mongodb server runs) gets
`MONGODB_CONNECTION_STRING=mongodb://mongo:27017/`; the containers and the networks are removed
at the end, so the database also starts fresh every run. With `MONGODB_CONNECTION_STRING` set,
the mongodb server uses that server instead (the variable goes to the credentialed-servers
container only, by name). The result's
`agent_environment.mongodb` says which was used. If the MongoDB container stops during the run,
the run hard-errors like a lost environment.

**Cloud metadata server.** The model under test runs shell commands and code in the environment
container (`cli-mcp-server_run_command`, `mcp-code-executor_execute_code`,
`mcp-server-code-runner_run-code`, desktop-commander), which has network access, and its tool
calls run in the credentialed-servers container. On a cloud VM the metadata server at
`169.254.169.254` hands out the VM's service-account token to anything that can reach it. So
before either container starts, gbench starts one short-lived probe container of the environment
image on each of the two docker networks (`container_guard.require_metadata_blocked`:
`--pull never`, no environment, no volumes; it only opens and closes a TCP connection to
`169.254.169.254:80`, sends nothing and reads no token). The same probe container resolves the
host names the container on that network needs (`resolve=`): for the environment container
`ENVIRONMENT_HOSTS` (the package hosts its servers start from, `registry.npmjs.org`,
`pypi.org`, `files.pythonhosted.org` and `github.com`, and `export.arxiv.org`,
`clinicaltrials.gov`, `context7.com`, `collectionapi.metmuseum.org`, `openlibrary.org`,
`nominatim.openstreetmap.org`, `en.wikipedia.org` and `httpbin.org`, the pre-flight's fetch
target); for the credentialed-servers container `CREDENTIALED_HOSTS` (the same
package hosts and `api.airtable.com`, `api.g.alchemy.com`, `api.e2b.app`, `api.exa.ai`,
`api.github.com`, `maps.googleapis.com`, `oauth2.googleapis.com`, `www.googleapis.com`,
`api.laratranslate.com`, `developer.nps.gov`, `api.notion.com`, `realtime.oxylabs.io`,
`slack.com`, `api.twelvedata.com`, `api.weatherapi.com`). A name that does not resolve stops the
run with `infra_required`, so a DNS outage is found before the run, never scored (the result
lists the names in `agent_environment.dns_names_checked`). When the connection opens, or the
probe cannot give a verdict, the run stops with `infra_required` naming the host firewall rule to
add once:
```bash
sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save   # keep it across reboots
```
gbench never runs sudo or changes the host firewall itself. The result records the verdicts in
`agent_environment.metadata_guard`.

**An environment you started yourself** (`GBENCH_MCP_ATLAS_SANDBOX_URL`, e.g.
`http://127.0.0.1:1984`) stands in for the environment container; gbench checks it before the
model runs anything there, and the run stops with `infra_required` when a check fails:
1. every credentialed server's variables must be set in gbench's environment (as for a local run,
   under [Requirements](#requirements));
2. `GET /enabled-servers` must list no server that needs a credential (the 15 credentialed
   servers, exa and oxylabs among them, and brave-search): their keys and tokens would sit in
   that environment, where the model's shell and code tools can read them. Start it without
   those variables and with the 20 keyless servers (ddg-search included: `ddg-search_fetch_content`
   runs there, and the pre-flight stops with `infra_required` when the environment does not run
   it):
   ```bash
   ENABLED_SERVERS=arxiv,calculator,cli-mcp-server,clinicaltrialsgov-mcp-server,context7,ddg-search,desktop-commander,fetch,filesystem,git,mcp-code-executor,mcp-server-code-runner,memory,met-museum,open-library,osm-mcp-server,pubmed,weather,whois,wikipedia
   ```
   `/enabled-servers` does not show a key passed to the container anyway (for example with
   `--env-file .env`), so:
3. gbench checks, through the environment's own code tool (`mcp-code-executor_execute_code`, else
   `mcp-server-code-runner_run-code`, uncached), that none of the release servers' credential
   variables (and neither `GEMINI_API_KEY` nor `GEMINI_API_KEYS`) holds a value in the tool's
   environment or in any process environment it can read (`/proc/<pid>/environ`: the container's
   PID 1, where `docker run --env-file` puts them, and the servers that inherited them): the
   probe prints the names that are set, never a value. A name that is set stops the run with
   `infra_required` naming it; no verdict from either tool stops it too. The result records the
   check in `agent_environment.external_credential_check`;
4. the metadata-server verdict is established through the environment's own code tool: gbench
   runs container_guard's probe script (the same TCP connection test, opened and closed, nothing
   sent, no token read) with `mcp-code-executor_execute_code`, else
   `mcp-server-code-runner_run-code`, uncached (`use_cache: false`). "Reachable" stops the run
   naming the firewall rule (and that docker's `DOCKER-USER` chain does not filter the host
   network, so start the environment on a bridge or user-defined network); no verdict from either
   tool stops it too. The check never depends on a knob.

gbench still runs the credentialed servers (exa and oxylabs included) and the local MongoDB in its
own container, as for a local run, so the image (and the MongoDB image unless
`MONGODB_CONNECTION_STRING` is set) must be built for this mode too.

**Pre-flight.** gbench then waits up to 900 s for `GET /` and runs the release's pre-flight check
as `run_eval.py` does before a run:
1. `GET /enabled-servers`: every configured server must be online, and ddg-search must run (its
   `fetch_content` runs there, and its check call below goes to Gemini, so the list is what shows
   it runs; an environment that does not run it is `infra_required`). brave-search, if an
   external environment still lists it, is ignored: gbench never calls it;
2. one real call per server, all 36 (`services/mcp_eval/test_servers.py` `TEST_CALLS`, its pass
   rule, 30 s timeout, 8 at a time, a fresh random oxylabs query). The calls of the environment
   servers go to its `/call-tool` and those of the credentialed servers to theirs; exa's
   (`exa_web_search_exa`) and oxylabs' (`oxylabs_google_search_scraper`) calls reach their own
   servers, exactly as upstream, so the check proves `EXA_API_KEY` and the Oxylabs credentials
   work. On top of the release's pass rule (which lets oxylabs' text replies through), gbench
   also fails exa's or oxylabs' check when the reply says the server could not reach its
   provider's API (`provider API unreachable`) or that the provider refused the run's account
   (`the provider refused the account (...)`; the replies are listed under
   [Search engines](#search-engines-gemini-grounding-for-the-search-tools)): with either, every
   call of their page tools would be a harness error. brave-search's and ddg-search's calls
   (`brave-search_brave_web_search`, `ddg-search_search`, both search tools) go to gbench's
   Gemini grounding, so the check also proves the grounding works.

Any failure is `infra_required`. The message lists each failing server, the variables it needs
(and which are unset), and the release's sample data it also needs; a failing brave-search or
ddg-search check is named as answered through the Gemini API. A server missing from the
`/enabled-servers` list is reported as not running (usually its variables are unset). There is
no reduced-server or stub mode.

The environment serves tool calls concurrently (`/call-tool` is an async handler). The
pre-flight's outcome depends on the servers' public APIs at the time of the run, as it does
upstream: a server whose API is slow or rate-limits the machine can exceed the 30 s check
timeout, and `test_servers.py`'s pass rule also fails a server whose check returns an empty
result (for example `wikipedia` when its upstream search returns `{"results": []}`). Re-run once
that API answers.

With `GBENCH_MCP_ATLAS_SANDBOX_URL` (above) gbench runs the same pre-flight over that environment
and its own credentialed-servers container. State then carries over between runs on that
environment. The search tools are still served by gbench: calls to those names never reach that
environment; `ddg-search_fetch_content` goes to its ddg-search.

## Search engines: Gemini grounding for the search tools
The release has four search-engine servers. Their **search tools** (a result list for a query)
are served by Gemini Google Search grounding, called from gbench's own process with
`GEMINI_API_KEYS` / `GEMINI_API_KEY` (`GEMINI_API_KEYS` wins), behind the same tool names and
argument schemas (`GEMINI_SERVED_TOOLS` in the code). Their **page tools** (a page's content for a
URL or an Amazon product) are not search: they run as upstream on the release's own servers
(`UPSTREAM_SEARCH_TOOLS` in the code), where gbench passes each call on unchanged and returns the
server's reply unchanged (credential values masked, as for every credentialed server).

| Server (release) | Search tools: Gemini Google Search grounding | Page tools: the release's server |
|---|---|---|
| brave-search (`@modelcontextprotocol/server-brave-search@0.6.2`, Brave Search API) | `brave-search_brave_web_search`, `brave-search_brave_local_search` | (none; the server does not run) |
| ddg-search (`duckduckgo-mcp-server==0.5.0`, DuckDuckGo) | `ddg-search_search` | `ddg-search_fetch_content` (environment container, no key) |
| exa (`exa-mcp-server@3.2.1`, Exa API) | `exa_web_search_exa` | `exa_web_fetch_exa` (credentialed-servers container, `EXA_API_KEY`) |
| oxylabs (`oxylabs-mcp==0.4.1`, Oxylabs Web Scraper API) | `oxylabs_google_search_scraper`, `oxylabs_amazon_search_scraper` | `oxylabs_universal_scraper`, `oxylabs_amazon_product_scraper` (credentialed-servers container, `OXYLABS_USERNAME` / `OXYLABS_PASSWORD`) |

- **Catalog.** The ten entries (name, description, `inputSchema`) are the ones those servers
  list in the pinned image, verbatim (pinned by a sha256 in the code). gbench adds them to the
  environment's `/list-tools` catalog where the release lists them (template order), in place of
  the running servers' own entries, so every task's `ENABLED_TOOLS` resolve and the declared
  tools are the ones the release declares. The running servers (ddg-search, exa, oxylabs) must
  list exactly these entries: before the samples load, gbench compares their live listing with
  the pinned one, and a tool whose description or schema differs, a tool of theirs the pinned
  listing does not have, or a pinned tool of a running server (one that lists at least one
  entry) that it does not list, stops the run with `infra_required` (`catalog_drift`; a running
  server that lists no tool at all is offline, which the pre-flight reports). With the image built from
  `docker/mcp_atlas.Dockerfile` the live listing of exa and oxylabs equals the pinned one
  (checked on 2026-09-28 against the rebuilt image, with placeholder credentials). At the pinned
  revision the tasks enable seven of the ten tools: `oxylabs_universal_scraper` (80 tasks),
  `oxylabs_google_search_scraper` (78), `exa_web_search_exa` (77),
  `brave-search_brave_web_search` (63), `ddg-search_search` (56),
  `oxylabs_amazon_search_scraper` (56) and `ddg-search_fetch_content` (42); `exa_web_fetch_exa`,
  `oxylabs_amazon_product_scraper` and `brave-search_brave_local_search` are in no task's
  `ENABLED_TOOLS`, and their servers are still part of the release's pre-flight.
- **Arguments** (search tools). Each call's arguments are checked against the tool's
  `inputSchema` with `jsonschema.validate`, as the environment's MCP server (mcp 1.12.1,
  `validate_input`) checks every call before it reaches a server. A failure is HTTP 500
  `{"detail": "Failed to call tool '<name>': Input validation error: <message>"}`, for example
  `'query' is a required property`, `'abc' is not of type 'number'`, or `Additional properties
  are not allowed ('x' was unexpected)` for an exa tool. gbench also answers an empty or blank
  `query` with HTTP 500 `Failed to call tool '<name>': Error: Invalid arguments for <tool>: ...`
  instead of sending it to Gemini, which would have nothing to look up (a gbench check). A page
  tool's arguments are checked by its own server, as upstream.
- **Search tools.** They ground their `query` (`count`, `numResults`, `max_results` or `limit`
  sets how many results are kept: default 10 (5 for `brave-search_brave_local_search`); at most
  20 for brave-search and ddg-search, 100 for exa and oxylabs). `oxylabs_amazon_search_scraper`
  grounds `<query> site:amazon.com` (10 results). Arguments grounding has no equivalent for
  (region, locale, domain, geo_location, render, offset, start_page, pages, ...) are accepted and
  unused. Each reply has the replaced server's text layout, filled with the grounding results
  (title, URL, and an empty description/highlights/summary/desc: see "No generated text" below):
  brave-search `Title: / Description: / URL:` blocks; exa `Title: / URL: /
  Published: N/A / Author: N/A / Highlights:` blocks; ddg-search `Found N search results:`
  lists; oxylabs the parsed JSON (`{"results": [{"pos", "url", "title", "desc"}]}`) by default,
  and with `parse` false `[title] url` lines (`output_format` links) or markdown. A lookup with
  no results gives the server's own empty reply (ddg-search's "No results were found ...", which
  the pre-flight's pass rule fails, as upstream). Both go through `search_tool.py`'s model
  cascade (`GBENCH_SEARCH_MODELS`), key rotation, rate limiter (`GBENCH_SEARCH_MAX_RPS`) and
  retry shape. The key never enters the agent environment, a prompt or a tool reply.
- **No generated text reaches the model (search-engine lookups, since 2026-10-12).** Every lookup
  here has a deadline, so `search_tool.py` serves it as a search-engine lookup: the request asks
  `Search the web and list the most relevant sources for: <query>` (the query unchanged inside
  it); the first two cascade models are asked at once (`GBENCH_SEARCH_INITIAL_WIDTH`, default 2);
  only an answer that cites at least one grounding source counts (an answer citing none is a
  failure of that model and the next model is asked, and a later cascade round does not ask that
  model again; a lookup in which no model cited a source is
  a Gemini failure, below, not an empty reply); each result is the source's title as grounding
  reports it, its real URL (Google's grounding redirect link resolved by one HEAD request to the
  redirect service, the `Location` header read, the source page not contacted; a link that does
  not resolve within 3 s, `GBENCH_SEARCH_REDIRECT_TIMEOUT_S`, or before the deadline is returned
  as the redirect link and counted), and an empty description: no part of a Gemini model's
  answer, no grounded-answer segment and no grounded-answer fallback is returned (such
  Gemini-written text, returned before, handed wildclawbench's agent a fabricated pull-request
  number, #30340). The same URL is returned once, in grounding's order, up to the number asked.
  Measured 2026-10-12 on 5 recorded wildclawbench agent queries x 2 models against a 27 s
  deadline: 2/10 answers in time with the bare query, 5/10 with the sources prompt, 4/5 queries
  with the sources prompt sent to both models at once; `site:` and quoted-phrase queries (here
  `oxylabs_amazon_search_scraper`'s `<query> site:amazon.com`) missed 27 s in every variant
  (mcp_atlas' own deadline is 58 s). `search_substitution.search_mode` is `search-engine`,
  `search_mode_rule` states the rule, and `search_engine_lookups` sums `lookups`, `answered`,
  `models_asked`, `answers_without_source`, `links_resolved`, `links_unresolved` and
  `links_direct`.
- **Page tools.** Each call goes to the server that runs the tool, like any other tool call:
  `ddg-search_fetch_content` to the environment container (an httpx page fetch with BeautifulSoup
  text extraction and its own refusal of private and internal addresses), `exa_web_fetch_exa`
  (Exa's `/contents` API) and the two oxylabs page tools (the Oxylabs Web Scraper API at
  `realtime.oxylabs.io`) to the credentialed-servers container. The harness's 60 s tool-call
  timeout, the environment's cache (`main.py` `CACHEABLE_SERVERS`, which lists ddg-search, exa and
  oxylabs), its error bodies and gbench's reply mapping apply as for every tool. The result counts
  these calls per tool in `search_substitution.upstream_tool_calls`. A reply that says the server
  could not reach its provider's API at all (oxylabs-mcp's `Request error during POST request:
  ...`, exa-mcp-server's `web_fetch_exa error: fetch failed`: a network or DNS failure between the
  credentialed-servers container and `realtime.oxylabs.io` or `api.exa.ai`, which no argument of
  the model's causes, since the provider fetches the page) is a harness error: the model gets the
  reply, as upstream, and the task is not scored
  (`search_substitution.upstream_provider_unreachable`, see [Harness errors](#harness-errors)).
  The Exa and Oxylabs accounts are billed per call by their providers (see
  [Requirements](#requirements)), and the pre-flight's exa and oxylabs calls check them before
  the run. A reply that says the provider refused the run's account during the run (its
  credentials rejected, its credit, budget or plan exhausted, or its rate limit reached) is a
  harness error the same way: no argument of the model's causes it. These are:
  - exa-mcp-server 3.2.1 reports an HTTP error of the Exa API as `web_fetch_exa error (<status>):
    <Exa's error>` (in the environment's HTTP 500 reply `{"detail": "Failed to call tool
    'exa_web_fetch_exa': ..."}`); status 401 (`INVALID_API_KEY`; the reply to a rejected key,
    checked against the live API), 402 (`NO_MORE_CREDITS`, `API_KEY_BUDGET_EXCEEDED`,
    `TEAM_BUDGET_EXCEEDED`) or 429. Started over stdio, as the environment starts it, the server
    answers Exa's 429 (`RATE_LIMIT_EXCEEDED`) with its own text, `You've hit Exa's free MCP rate
    limit. ...`, which counts too. Exa's 403 does not: Exa also answers the content-safety
    refusals of a request with it (`PROHIBITED_CONTENT`, `CONTENT_FILTER_ERROR`), and the reply
    does not carry the tag, so it stays the model's outcome, as upstream;
  - oxylabs-mcp 0.4.1 reads the Oxylabs API's reply as JSON before it looks at the status, so a
    reply with no JSON body reaches the model as `Error: Expecting value: line 1 column 1 (char
    0)`. That is how the API refuses the credentials (HTTP 401 with an empty body, checked against
    the live API; a gateway's error page, a provider outage, reads the same), while it answers
    the requests it accepts in JSON; it counts as an account error, named `no JSON body (the
    API's refusal of the credentials, HTTP 401, or a provider-side error page)`. A reply with a job record and
    an HTTP error status reads `HTTP error during POST request: <status> - ...`, and 401, 403
    (the account has no access to the resource), 407 and 429 (the rate limit) count. An error
    reply in JSON without a job record reaches the model as `Error: 'job'` (the server reads the
    job's id before it looks at the status), which is how an account refusal whose body carries
    no job (a 403 or a 429, say) arrives; it counts as an account error too. The reply does not
    say its status, so a request the API rejects as malformed (HTTP 400) with no job record would
    count the same way; the result names these calls apart (`<tool>: no job record in the error
    reply (Error: 'job')`), so their number is visible.

  The model gets the reply, as upstream, and the task is not scored
  (`search_substitution.upstream_provider_account_errors`, per `<tool>: <reason>`). A reply that
  says the provider's API failed on its side, an HTTP 5xx the server reports after its own
  retries (exa-mcp-server's `web_fetch_exa error (503): SERVICE_OVERLOADED`, or 500, 502, 504;
  oxylabs-mcp's `HTTP error during POST request: 5xx - ...`), is a harness error the same way
  (`provider_unavailable`; `search_substitution.upstream_provider_unavailable`, per `<tool>: HTTP
  <status>`), and the pre-flight's exa and oxylabs checks fail on it. Every other reply, a failed
  fetch of the page included, is the model's outcome, as upstream.
- **The environment's handling, kept** (search tools). The reply takes the path a `/call-tool`
  reply takes: one text block on success; the 60 s tool-call timeout; the environment's cache
  for these servers (an identical call in the same run gets the cached successful reply); the old
  name `brave_brave_web_search` served as `brave-search_brave_web_search`, as `main.py` maps it.
- **Gemini failures are harness errors.** A search call the Gemini API fails to serve (every
  model of the cascade failed, or no answer within the 60 s tool-call timeout; the lookups stop
  2 s before it so that they end first and say why) gets HTTP 500 `{"detail": "Failed to call
  tool '<name>': Error: Gemini Google Search grounding failed: <reason>"}` (the harness's `Tool
  call timed out after 60s` on a timeout), which is not cached, and its task is not scored (see
  [Harness errors](#harness-errors)). So is a failure inside gbench's own serving code (HTTP 500
  `Error: gbench could not serve the call (...)`, never a tool exception fed to the model). The
  grounding lookups send nothing after the call's deadline, and the run never waits for their
  threads. Within the deadline a lookup hedges: two models are asked at once, a grounding request
  with no answer after 12 s (`GBENCH_SEARCH_HEDGE_DELAY_S`) has the next cascade model asked in
  parallel, and the first answer citing a source is used, so one slow model cannot use up the
  whole 58 s. A request that lost
  the race sends nothing more, also when it was still waiting for a rate-limiter token.
- **Not read:** `BRAVE_API_KEY`.

The result's `search_substitution` records the four servers and their release packages
(`servers`, `release_servers`), the six tools gbench serves (`tools`, `search_tools`), the four
run on the release's servers (`upstream_tools`, with each one's server in
`upstream_tools_backend`, the run's calls to them in `upstream_tool_calls`, and those whose server
could not reach its provider's API in `upstream_provider_unreachable`, and those whose provider
refused the run's account in `upstream_provider_account_errors`), the server not
run (`servers_not_run`), the grounding backend (`backend`), the argument check
(`argument_validation`), the key not read (`not_read`), and the run's search calls, cache hits,
error replies, timeouts, the grounding model that answered each lookup (`grounding_backends`),
`gemini_outages`, `gemini_calls_cut_by_wall_clock` and `gemini_wait_s` (the time the tasks' tool
calls spent waiting on gbench's Gemini-served search tools: the total, the most of one task, and
the tasks stopped by the wall clock that had waited, with their wait).

## Protocol
- **Prompt.** The task's `PROMPT` is the only message, as `run_eval.py` sends it: no system
  prompt, header or footer.
- **Tools.** The two containers' `POST /list-tools` catalogs with the four search-engine
  servers' ten pinned entries (above), filtered to the task's `ENABLED_TOOLS` (in catalog order),
  converted as `agent-eval.ts` `_transformToolCalls` does:
  `name`, `description`, `parameters` = the server's `inputSchema`, `strict: false`. The dataset
  also names tools of four servers the release does not ship (rijksmuseum-server,
  f1-mcp-server, anili, balldontlie; 85 tasks list some as distractors). They drop out exactly as
  they do upstream, and the result lists them. No task loses all its tools.
- **Tool calls** (`sandbox-client.ts` `callTool` and `agent-eval.ts`): `POST /call-tool` with
  `{"tool_name", "tool_args"}` (`args || {}` with JavaScript truthiness), 60 s timeout; a call
  to one of the six search tools is answered by gbench instead, with the reply described above
  (`tool_calls` counts it too). A call whose `arguments` string is not valid JSON is not sent:
  the model receives `Error: <first line of the parse error>`, as `agent-eval.ts` feeds back a
  `JSON.parse` failure, and the call still counts toward the 100-call cap (the message text is
  Python's JSON error, where the harness has Node's). Otherwise the model receives:
  - a 2xx reply: the text of the returned content blocks, joined by newlines (what vLLM makes
    of a tool message whose content is a list of text parts), or `success` when no block has
    non-blank text;
  - a 2xx reply with an MCP image or resource block next to non-blank text: `Error: [`, the
    first line of the harness's schema error;
  - a non-2xx reply (a failing tool, an unknown tool, invalid arguments): the response body
    verbatim, e.g. `{"detail":"Failed to call tool 'x': ..."}`;
  - no reply in 60 s: `Tool call timed out after 60s` (for a Gemini-served search tool this is a
    harness error, see above);
  - no connection: `Error: Failed to call tool <name>: <reason>` (counted in
    `tool_calls.transport_errors`), a harness error (see [Harness errors](#harness-errors)).
  A reply from the credentialed-servers container reaches the model with every credential value
  of gbench's environment replaced by `[REDACTED]` (a server that echoes no key gets its reply
  through unchanged). `met-museum_get-museum-object` calls get `returnImage: false`, as
  `agent-eval.ts` `prunedTools` sets it. A `lara-translate_import_tmx` call whose `tmx_url` the
  server's shell command would interpret (see [Agent environment](#agent-environment-the-mcp-servers))
  is not passed on: the model receives HTTP 500 `{"detail":"Failed to call tool
  'lara-translate_import_tmx': Error: gbench did not pass this call on: tmx_url must be a plain
  http(s) URL (no whitespace, double quote, $, backquote or backslash), because the server puts it
  into a shell command"}`, the model's outcome, counted in `tool_calls.refused_by_gbench`.
- **Tool loop** (services/agent-harness `agent-eval.ts`):
  - up to 256 turns and 100 tool calls; the call cap is checked before each call, as upstream,
    and always binds first;
  - no forced answer and no injected turns: gbench's "answer now" and recovery prompts are off;
  - the graded response is the last assistant message with content, as `run_eval.py`
    extracts it;
  - no output budget: the harness sends no `max_tokens`, so no turn carries one, with or
    without `--thinking`, and each turn may use the room left in the context window, what the
    server allows when the field is absent (gbench still counts the prompt, so a prompt that
    cannot fit is refused as over-context; the field is never sent, whether or not the window
    is known). `--max-output-tokens`, which the
    CLI requires for every run, is not applied and the result says so in
    `max_output_tokens_note`; the global tool-loop turn ceiling
    (`GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS`) does not apply either. A task whose last turn still
    stops at the end of the window, as it would upstream, is counted in `truncated_responses`;
  - the assistant turns sent back to the model carry the model's own content and tool calls;
  - a task still running after **1800 s** is stopped and scores 0, as `run_eval.py --timeout
    1800` abandons it (`ERROR: timeout after 1800s`, which the scorer scores 0). The clock
    starts when the task's first model request gets a request slot; time spent waiting for a
    slot behind other tasks does not count, as a harness worker that runs each task on its own
    never waits there. Stopped tasks are counted in `wall_clock_limited`.
  The result records these settings in `tool_loop_protocol` and how each sample's loop ended
  (`tool_rounds.loop_ends`, per sample `tool_loop_end`).
- **Scoring** (services/scoring `score_claims.py`):
  - claims are parsed and cleaned by `extract_claims` (bullets and numbering removed, curly
    quotes and dashes normalized, claims of 3 characters or fewer dropped);
  - an empty or failed response, or one that starts with `ERROR:` (how `run_eval.py` records a
    failed or timed-out task), is kept and scored 0.0 without judging;
  - otherwise **one judge call per claim** with the upstream prompt verbatim and the upstream
    JSON schema as structured output; `fulfilled` 1.0, `partially_fulfilled` 0.5,
    `not_fulfilled` 0.0 (a reply with no readable outcome counts as not fulfilled, as upstream
    once its retries are spent);
  - responses longer than 500,000 characters are cut, with the upstream truncation marker;
  - `coverage_score = round(total / number_of_claims, 3)`; a task passes at coverage >= 0.75.
  Each trace records the Gemini model that answered each claim's judge call
  (`claim_judge_models`, in claim order). A judge-cascade outage on any claim marks the task
  `JUDGE_OUTAGE`, which is excluded from every metric rather than scored as a miss.

When the run ends, gbench checks that the containers it started are still running (the
environment container, the credentialed-servers container and the local MongoDB) and that the
environment and the credentialed servers still answer `GET /`. If the agent environment was lost
during the run, the run hard-errors with `infra_required` and no score is published: every call
after the loss got an error reply, so those answers are not MCP-Atlas answers.

### Harness errors
A search tool call the Gemini API fails to serve (see
[Search engines](#search-engines-gemini-grounding-for-the-search-tools)), a call to a
Gemini-served search tool that is still waiting on the Gemini API when the task's 1800 s wall
clock stops the task, a call that cannot reach the agent environment while it answers again
afterwards, or a call to exa's or oxylabs' page tools whose server could not reach its provider's
API, whose provider refused the run's account or whose provider's API failed on its side (an
HTTP 5xx; see [Search engines](#search-engines-gemini-grounding-for-the-search-tools)), is a
harness error, never the model's outcome. Its task is not judged and not
scored: `judge_grade` `HARNESS_ERROR` with `scoring_excluded`, excluded from every metric and
from `category_accuracy`, also when the task later ran past its 1800 s wall clock. The run's
status is `completed_with_errors` (`failed` when no task is left to score, with `accuracy` null
and `accuracy_withheld_reason` saying why, never 0), `harness_errors`
records the calls (`gemini_calls`, `gemini_calls_cut_by_wall_clock`, `environment_calls`,
`provider_unreachable_calls`, `provider_account_error_calls`, `provider_unavailable_calls`), the
number of tasks not scored (`tasks_unscored`) and their ids (`task_ids`), and
`leaderboard_comparable_reason` names them.

**Known gap: the other credentialed servers' account errors.** The start-up pre-flight makes one
call per credentialed server, so a credential that expires or a quota or rate limit that runs out
during the run (an expired `GOOGLE_REFRESH_TOKEN`, Lara's small free allowance, Twelve Data's
800-calls-a-day plan and its per-minute limit, a weather API quota, a revoked Airtable or Notion
token) reaches the model as that server's error reply. gbench has pinned the reply signatures of
exa's and oxylabs' page tools only (above); for the other 13 credentialed servers it does not
know each server's wording at its pinned version, so such a reply is scored as the model's
outcome, as upstream. gbench counts the error replies of those servers that read as their
provider refusing the run's account (an HTTP 401 or 429, rejected or expired credentials,
`invalid_auth` / `invalid_grant`, a used-up quota or rate limit:
`credentialed_account_suspect`), per server in `tool_calls.account_error_suspects`, and names
them in `leaderboard_comparable_reason`; a run with any should be re-run once the account is
fixed. Plan the run within each account's allowance (see [Requirements](#requirements)).

The time a task's tool calls spend waiting on gbench's Gemini-served search tools counts toward
its 1800 s wall clock, as the replaced search services' time does upstream. Each trace records
that time (`gemini_wait_s`), and a task the wall clock stopped (scored 0) after such waiting is
named in `leaderboard_comparable_reason` with the time it waited; it is still scored 0, as it is
not known whether it would have finished with the release's search services.

## Headline metric and how to read the result
`accuracy` is `pass_rate_0.75`: the share of tasks with coverage >= 0.75, which the
[Scale leaderboard](https://scale.com/leaderboard/mcp_atlas) ranks on. A full run corresponds
to its **Pass Rate % (Public 500)** column (`leaderboard_column`). The other column, "All
1000", also scores 500 held-out tasks that are not released, so no public run can reproduce
it. `correct_answers` counts the passing tasks. Alongside: `mean_coverage` and
`pass_rate_0.50` (upstream's other statistics), `tasks_fully_covered`, `claims_scored`,
`claims_unparsed`, `judge_models` (the Gemini models that judged, with the number of tasks each
judged claims of). Each trace records the model that gave each claim's verdict
(`claim_judge_models`, in the order of `claim_scores`) and the models of the task
(`judge_models`).

`agent_environment` records the release, the image, how the environment was provisioned, the
pre-flight result (`servers_served_by_gbench`: brave-search, which does not run;
`search_tools_served_by_gbench`; `tools_run_in_environment`: the four page tools;
`health_check_through_gemini`: brave-search and ddg-search), `search_servers`, where the
credentialed servers ran (`credentialed_servers`), the metadata-server verdicts
(`metadata_guard`), the host names checked (`dns_names_checked`), the catalog size (both
containers' tools with the ten pinned search-engine entries) and the dataset tool names nothing
serves. `search_substitution` records the search tools' replacement (above), and
`harness_errors` the harness errors, if any (above).
`tool_calls` counts the executed calls, error replies, timeouts, transport errors and the calls
gbench answered without passing them on (`refused_by_gbench`);
`tool_argument_errors` counts the calls whose arguments were not valid JSON (not executed);
`wall_clock_limited` counts the tasks stopped at 1800 s. `official_sampling` records the
harness's sampling (no temperature and no `max_tokens` sent) and whether the run matched it;
`truncated_responses` counts the tasks whose last turn stopped at the end of the context
window.

`leaderboard_comparable` is **false**. `leaderboard_comparable_reason` states:
- the judge is gbench's Gemini cascade (models listed), not the harness default
  `gemini/gemini-3.1-pro-preview` (the leaderboard page names Gemini-2.5-Pro at temperature
  0); `GBENCH_JUDGE_MODELS` selects the judge models;
- the search tools of the search-engine servers brave-search, ddg-search, exa and oxylabs are
  replaced by Gemini Google Search grounding behind their tool names and argument schemas (the
  results differ from the Brave, DuckDuckGo, Exa and Oxylabs search services the release calls;
  Gemini grounding search returning sources only, no generated text: each result is a cited
  source's title and URL with an empty description or summary); their page tools run as upstream
  on the release's servers;
- search tool calls the Gemini API failed to serve, if any;
- calls to Gemini-served search tools still waiting on the Gemini API when the task's wall clock
  stopped the task, if any;
- page tool calls whose server could not reach the Exa or Oxylabs API, if any;
- page tool calls whose provider (Exa or Oxylabs) refused the run's account, with each tool and
  reason, if any;
- tool calls that could not reach the environment, if any (the environment answered again
  afterwards; a lost environment is a hard error, see above);
- the tasks not scored because of such a harness error, if any;
- the tasks the wall clock stopped after waiting on gbench's Gemini-served search tools, with the
  time they waited, if any;
- `lara-translate_import_tmx` calls answered by gbench instead of being run, if any;
- the temperature gbench sends (the harness sends none);
- any `--eval-limit` subset.

The leaderboard page also mentions retry handling for transient tool errors. The released
harness at the pinned commit has none (it retries model calls only), and gbench follows the
released code.

## Requirements
- **Docker** and the locally built image. The build needs network access (GitHub, npm, PyPI
  and the pinned base image) and takes several minutes; the image uses about 4.9 GB of disk
  (1.05 GB of content, like the published 1.2.7 image), plus the build cache:
  ```bash
  docker build -t gbench-mcp-atlas:f24ba3fb -f docker/mcp_atlas.Dockerfile docker
  ```
  `GBENCH_MCP_ATLAS_IMAGE` names a different tag. A missing image, one built from another
  commit, or one without the label `gbench.mcp_atlas.search_servers=removed:brave-search` (built
  by an earlier gbench) is `infra_required` with this command. It is needed with
  `GBENCH_MCP_ATLAS_SANDBOX_URL` too (gbench runs the credentialed servers from it).
- Unless `MONGODB_CONNECTION_STRING` is set, the locally built MongoDB image with the release's
  sample data (about 1.5 GB of disk, 318 MB of content; the build pulls the pinned `mongo` base
  image and downloads the dump from GitHub):
  ```bash
  docker build --target mongo -t gbench-mcp-atlas-mongo:f24ba3fb -f docker/mcp_atlas.Dockerfile docker
  ```
  `GBENCH_MCP_ATLAS_MONGO_IMAGE` names a different tag. A missing image, or one built from
  another commit, is `infra_required` with this command.
- **Network access from the containers**: the servers call their public APIs (the host names
  under [Agent environment](#agent-environment-the-mcp-servers)), and some are resolved by uvx/npx
  when the container starts (github, weather, pubmed, weather-data).
- **On a cloud VM, the metadata server blocked for docker networks** (the host firewall rule under
  Agent environment); without it the run stops with `infra_required`.
- **A running OpenAI-compatible `/v1` chat endpoint that accepts tool calls** (vLLM:
  `--enable-auto-tool-choice --tool-call-parser <parser>`; a server gbench launches itself (no
  `--remote-endpoint`) gets both from `--tool-call-parser NAME`, and the CLI refuses a local run
  of this suite without it).
- **`GEMINI_API_KEYS` or `GEMINI_API_KEY`** (`GEMINI_API_KEYS` wins; a key from
  [Google AI Studio](https://aistudio.google.com/apikey)): required for the per-claim judge and
  for the search tools' Google Search grounding. The suite checks it is present and accepted
  before starting anything and hard-errors (`infra_required`) otherwise; there is no
  string-matching fallback. The pre-flight's brave-search and ddg-search checks then use it, so a
  key without grounding access fails there. It stays in gbench's process: no container gets it.
- **The server credentials** (the release's `env.template`), exported in gbench's environment.
  gbench passes them to the credentialed-servers container only (above), never to the container
  the model's shell and code tools run in. Every variable below is required: if any is unset,
  the suite stops with `infra_required` before any container starts (the release's pre-flight
  needs every server; no server is dropped and there is no partial run), naming each server with
  exactly its unset variables and the sample data it also needs. The one exception is
  `MONGODB_CONNECTION_STRING`: without it gbench serves the release's MongoDB data from its own
  container (above). 20 of the environment's 35 servers need no credential, and
  `BRAVE_API_KEY` is not read (brave-search does not run). Where to get each (the links are the
  ones in the release's `env.template` and `data_exports/README.md`; plans and prices as the
  providers' pages state them on 2026-09-28, which may change):

  | Server | Variables | What gbench runs with it | Where to get it (free or paid) | Also needs (the release's `data_exports/`) |
  |---|---|---|---|---|
  | airtable | `AIRTABLE_API_KEY` | `@felores/airtable-mcp-server@0.3.0` (the Airtable API) | an Airtable personal access token ([airtable.com/create/tokens](https://airtable.com/create/tokens); scopes `data.records:read`, `data.records:write`, `schema.bases:read`, `schema.bases:write`, with access to the copied base; see the [server's README](https://github.com/felores/airtable-mcp?tab=readme-ov-file#obtaining-an-airtable-api-key)); free Airtable account | the release's base copied into the account: "Copy base" at [airtable.com/appIF9byLfQwdHqE2/shr1KTZOgPl0qQmA8](https://airtable.com/appIF9byLfQwdHqE2/shr1KTZOgPl0qQmA8) (`airtable_database_online_link.txt`) |
  | alchemy | `ALCHEMY_API_KEY` | `@alchemy/mcp-server@0.1.8` (the Alchemy blockchain APIs) | the API key of an app you create at [alchemy.com](https://www.alchemy.com/docs/); free plan (30M compute units a month) | |
  | e2b-server | `E2B_API_KEY` | `@e2b/mcp-server@0.2.0` (runs the model's code in E2B's cloud sandboxes) | [e2b.dev](https://e2b.dev/) dashboard API key; free Hobby plan with one-time usage credits | |
  | exa | `EXA_API_KEY` | `exa-mcp-server@3.2.1`: its `web_fetch_exa` page tool (the Exa `/contents` API); its search tool is served by Gemini grounding | [dashboard.exa.ai/api-keys](https://dashboard.exa.ai/api-keys) ([exa.ai](https://exa.ai/)); free tier with monthly credits, then paid per request | |
  | github | `GITHUB_TOKEN` | the `github.com/geobio/smitheryai-mcp-servers-github` server (the GitHub API) | a personal access token from [github.com/settings/tokens](https://github.com/settings/tokens); free | |
  | google-maps | `GOOGLE_MAPS_API_KEY` | `@modelcontextprotocol/server-google-maps@0.6.2` (Geocoding, Places Text Search and Place Details, Directions, Distance Matrix and Elevation APIs) | a Google Maps Platform API key from a Google Cloud project with those APIs enabled ([setup](https://www.npmjs.com/package/@modelcontextprotocol/server-google-maps#setup)); the project needs billing enabled, and each API has free monthly calls | |
  | google-workspace | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REFRESH_TOKEN` | `@geobio/google-workspace-server@0.1.0` (Gmail and Google Calendar APIs) | a Google Cloud OAuth client (type Web application) with the Gmail API and Google Calendar API enabled, and a refresh token for the account holding the calendar, per the [prerequisites and setup](https://github.com/epaproditus/google-workspace-mcp-server?tab=readme-ov-file#prerequisites); free Google account and Cloud project | `calendar_mcp_eval_export.zip` (an `.ics` file) imported into that account's Google Calendar (preferably a new account) |
  | lara-translate | `LARA_ACCESS_KEY_ID`, `LARA_ACCESS_KEY_SECRET` | `@translated/lara-mcp@0.0.11` (the Lara Translate API) | the access key ID and secret from a Lara account ([sign-up](https://app.laratranslate.com/sign-up), [docs](https://developers.laratranslate.com/docs/introduction)); free plan with a small monthly API allowance, then paid | |
  | mongodb | `MONGODB_CONNECTION_STRING` (optional) | `mongodb-mcp-server@0.2.0` | without it, nothing: gbench's local MongoDB image serves the release's data; with your own server, its connection string (e.g. a free MongoDB Atlas M0 cluster, per `data_exports/README.md`; add `?tls=true` for Atlas) | with your own server, `mongo_dump_video_game_store-UNZIP-FIRST.zip` restored into it with `mongorestore` (the `video_game_store` database) |
  | national-parks | `NPS_API_KEY` | `mcp-server-nationalparks@1.0.1` (the NPS API) | the sign-up form at [nps.gov/subjects/developer/get-started.htm](https://www.nps.gov/subjects/developer/get-started.htm); free | |
  | notion | `NOTION_TOKEN` | `@notionhq/notion-mcp-server@1.8.1` (the Notion API) | the Internal Integration Secret of an internal integration created at [notion.so/profile/integrations](https://www.notion.so/profile/integrations); free Notion account | `mcp-atlas-notion-data.zip` imported into the workspace (Settings > Import: 6 tables and 1 page; check that every table has data) |
  | oxylabs | `OXYLABS_USERNAME`, `OXYLABS_PASSWORD` | `oxylabs-mcp==0.4.1`: its `universal_scraper` and `amazon_product_scraper` page tools (the Oxylabs Web Scraper API); its two search tools are served by Gemini grounding | the API user's username and password of a Web Scraper API subscription, from the Oxylabs dashboard ([oxylabs.io/products/scraper-api/web](https://oxylabs.io/products/scraper-api/web)); paid, with a free trial | |
  | slack | `SLACK_MCP_XOXC_TOKEN`, `SLACK_MCP_XOXD_TOKEN` | `slack-mcp-server@1.1.23` (the Slack API) | the `xoxc-` token and the `d` cookie (`xoxd-`) of a logged-in browser session, per the [authentication setup](https://github.com/korotovsky/slack-mcp-server/blob/master/docs/01-authentication-setup.md); free workspace | `slack_mcp_eval_export.zip` imported into a new workspace (`https://<workspace>.slack.com/services/import`); its messages are from early December 2025, which a free workspace hides after 90 days (a paid workspace, or the export's timestamps moved forward, keeps them visible) |
  | twelvedata | `TWELVE_DATA_API_KEY` | `mcp-server-twelve-data==0.2.5` (the Twelve Data API) | [twelvedata.com](https://twelvedata.com/docs) API key; free Basic plan (800 calls a day) | |
  | weather-data | `WEATHER_API_KEY` | the `github.com/geobio/weather-mcp-server` server (WeatherAPI.com) | [weatherapi.com](https://www.weatherapi.com/) API key; free plan | |

  The release's step-by-step data setup is in its
  [`data_exports/README.md`](https://github.com/scaleapi/mcp-atlas/blob/f24ba3fb0bfa484c86acb28431fad6d7282455f9/data_exports/README.md).
  gbench cannot see what an account holds, so it cannot check that the sample data is loaded;
  without it, as that README says, the servers still work but return empty results or errors
  where a task refers to the data.
- Python packages `datasets` and `aiohttp` (base install), `google-genai` (the `evals`
  extra: `pip install -e ".[evals]"`) and `jsonschema` (the search tools' argument check:
  `pip install jsonschema`; without it the suite is `infra_required`). The dataset is public.

## Env knobs
- `GBENCH_MCP_ATLAS_IMAGE`: the agent-environment image tag, default `gbench-mcp-atlas:f24ba3fb`.
- `GBENCH_MCP_ATLAS_SANDBOX_URL`: use an environment that is already running in place of the
  environment container. It must run no credentialed server, and gbench establishes the
  metadata-server verdict through its code tool before the model runs anything there (see
  [Agent environment](#agent-environment-the-mcp-servers)); the credentialed servers (exa and
  oxylabs included) and the mongodb data still run in gbench's own container.
- `GBENCH_MCP_ATLAS_MONGO_IMAGE`: the MongoDB sample-data image tag, default
  `gbench-mcp-atlas-mongo:f24ba3fb`.
- `GBENCH_MCP_ATLAS_TEMPERATURE`: send this temperature for this suite (default: 1.0 with
  `--thinking`, 0.0 without; the harness sends none).
- The judge cascade knobs (`GBENCH_JUDGE_MODELS`, ...) and the search knobs for the search tools
  (`GBENCH_SEARCH_MODELS`, `GBENCH_SEARCH_MAX_RPS`, `GBENCH_SEARCH_ATTEMPTS`,
  `GBENCH_SEARCH_BACKOFF`, `GBENCH_SEARCH_CASCADE_ROUNDS`, `GBENCH_SEARCH_HEDGE_DELAY_S`,
  `GBENCH_SEARCH_INITIAL_WIDTH`, `GBENCH_SEARCH_REDIRECT_TIMEOUT_S`) are in
  [env-knobs](env-knobs.md).

## Sampling
The harness's model request is `{model, messages, tools}` (`litellm-strategy.ts`; `run_eval.py`
adds sampling fields only through its opt-in `--extra-llm-params`), so gbench sends no
`max_tokens` and no other sampling field (each turn may use the rest of the context window;
`official_sampling.max_tokens`, `max_output_tokens_note`). The temperature is gbench's rule for
every suite, sent on every turn: 1.0 with `--thinking`, 0.0 without. `--temperature` or
`GBENCH_MCP_ATLAS_TEMPERATURE` (which takes precedence) sets another. The result records it under
`official_sampling` (the harness sends none) and in `leaderboard_comparable_reason`. Besides the temperature, `--thinking`
toggles the chat template's thinking switch. The judge is pinned at temperature 0, as upstream.

## Run
Export the Gemini key and every server credential above, build the two images once, then run:
```bash
export GEMINI_API_KEY=...
export AIRTABLE_API_KEY=... ALCHEMY_API_KEY=... E2B_API_KEY=... EXA_API_KEY=... GITHUB_TOKEN=...
export GOOGLE_MAPS_API_KEY=... GOOGLE_CLIENT_ID=... GOOGLE_CLIENT_SECRET=... GOOGLE_REFRESH_TOKEN=...
export LARA_ACCESS_KEY_ID=... LARA_ACCESS_KEY_SECRET=... NPS_API_KEY=... NOTION_TOKEN=...
export OXYLABS_USERNAME=... OXYLABS_PASSWORD=... SLACK_MCP_XOXC_TOKEN=... SLACK_MCP_XOXD_TOKEN=...
export TWELVE_DATA_API_KEY=... WEATHER_API_KEY=...
# optional: export MONGODB_CONNECTION_STRING=... (else gbench's local MongoDB image is used)
docker build -t gbench-mcp-atlas:f24ba3fb -f docker/mcp_atlas.Dockerfile docker
docker build --target mongo -t gbench-mcp-atlas-mongo:f24ba3fb -f docker/mcp_atlas.Dockerfile docker
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals mcp_atlas --max-output-tokens 16384
```
The CLI requires `--max-output-tokens` for every eval run; this suite records it as not applied
(`max_output_tokens_note`), because the harness sends no `max_tokens`.
Add `--eval-limit 20` for a quick check.
