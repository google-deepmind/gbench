# wildclawbench setup

Canonical **WildClawBench** ([github.com/internlm/WildClawBench](https://github.com/internlm/WildClawBench),
[HF dataset](https://huggingface.co/datasets/internlm/WildClawBench)): 60 hand-built, real-world,
long-horizon agent tasks across 6 categories (EN + ZH). Each task runs in its **own Docker
container** from the prebaked image `wildclawbench-ubuntu:v1.3`, with the **OpenClaw** agent inside
chaining 10-60+ tool calls (browser, bash, filesystem, email, calendar, web search) over 10-20
minutes; after the agent finishes, the task's own `grade()` (a mix of **programmatic checks + an LLM
judge**) is `docker exec`'d for a per-metric 0-1 score. gbench delegates the whole run to the pinned
upstream harness (`eval/run_batch.py`) inside a LOCAL orchestrator image, driven
**docker-out-of-docker**.

Pinned harness commit: `316334ccc4a87b9b5635ad73da99b4dfc0b3887e` (the task image ships OpenClaw
2026.3.11).

## External models and services: the Gemini API only
Every external model or search service the harness calls, or offers the agent as a tool, goes
through the Gemini API (`GEMINI_API_KEYS`, else `GEMINI_API_KEY`; a key list is rotated), served by
one **Gemini proxy** that runs in the orchestrator (`gbench/docker/wildclawbench_cascade_judge.py`).
The task image's other search skills are removed (see "Search tools the agent is offered"). Each
substitution, and the removal, is recorded on the result (`gemini_substitutions`,
`removed_search_skills`) and makes the run not leaderboard-comparable.

The Gemini models are gbench's one cascade, `base.GEMINI_CASCADE` (`gemini-3.8-flash`,
`gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.1-pro-preview`, tried in order;
`GBENCH_JUDGE_MODELS` replaces it, and `GBENCH_SEARCH_MODELS` replaces it for web search only).
gbench always passes the run's lists into the orchestrator (`GBENCH_JUDGE_MODELS` and
`GBENCH_SEARCH_MODELS`, from `base.gemini_cascade_env()`), so the proxy and `search_tool.py` call
exactly the models the result records (`gemini_cascade`, `gemini_search_cascade`); the default
written into the proxy equals `base.GEMINI_CASCADE` and is used only when neither variable is set.
The key check before the run also asks the Gemini API for each model of both lists: a model it does
not serve for the key is `infra_required` naming the model and the variable to set. During the run a
model that answers HTTP 404 is not asked again in that request. The result records which models
answered: `gemini_models_answered` (`{model: answered requests}`),
`gemini_models_answered_by_kind` (judge, grounded judge, agent helper model, agent web search), and
per task `raw_summary.per_task[].gemini_models` (the proxy writes each answered request, with the
model, the time it arrived and the requesting container, to `gemini_answers.jsonl`).

| Component | Official | gbench |
|---|---|---|
| LLM judge of the 43 judged tasks (`grade()` calls the stock OpenAI SDK at `OPENROUTER_BASE_URL` with model `JUDGE_MODEL`) | `openai/gpt-5.4` through OpenRouter | the gbench Gemini judge cascade (the model list, rounds and backoff of `base.judge_generate_cascade`) through Gemini's OpenAI-compatible chat endpoint. The requested model is ignored; the prompts and every other request field are forwarded verbatim, the temperature included (`temperature=0` in 41 of the 43 judged tasks; `03_Social_Interaction_task_2` and `03_Social_Interaction_task_3` send none, and none is sent); OpenRouter routing fields are dropped. The judge calls of `06_Safety_Alignment_task_2_leaked_api`, `06_Safety_Alignment_task_3_leaked_api_pswd` and `06_Safety_Alignment_task_6_prompt_injection` turn thinking off (`extra_body={"thinking": {"type": "disabled"}}`), which no model of the cascade can: they are sent at the nearest Gemini setting, `reasoning_effort: "low"` (Gemini thinking level `low`, the lowest level every model of the default cascade accepts; see "Judge calls that turn thinking off"). Any other `thinking`/`reasoning`/`reasoning_effort` setting is dropped (the pinned tasks send none) |
| Audio judge of `05_Creative_Synthesis_task_11_video_en_to_zh_dub` (its `grade()` calls a hardcoded `AUDIO_MODEL` through the same client, with `input_audio` parts) | `openai/gpt-4o-mini-audio-preview` through OpenRouter | the same Gemini judge cascade (requested model ignored, the audio parts and every other field forwarded verbatim) |
| Web-search verification judge of `04_Search_Retrieval_task_1` (the judge call with `tools=[{"type": "openrouter:web_search", ...}]`) | OpenRouter's `openrouter:web_search` server tool (`engine` auto, `max_results` 5, `max_total_results` 30) | Gemini `generateContent` with the `google_search` tool (Google Search grounding) over the same judge cascade, with the request's temperature (`0`); system messages become `systemInstruction`, `max_tokens`/`max_completion_tokens` plus the Gemini thinking allowance (8192) become `maxOutputTokens` (see "Judge output limit"); Gemini's `finishReason` is returned as the OpenAI `finish_reason` (`MAX_TOKENS` -> `length`, `STOP` -> `stop`, a block reason -> `content_filter`) and kept as `native_finish_reason`; the reply is returned as an OpenAI chat completion with the grounding sources as `url_citation` annotations. The search engine, the queries and the number of results are Gemini's |
| Helper model the prompts of 25 tasks offer the agent through `OPENROUTER_BASE_URL`/`OPENROUTER_API_KEY` (`HELPER_MODEL_TASKS` in `wildclawbench.py`: `01_Productivity_Flow_task_3` and `_8`, `02_Code_Intelligence_task_3` to `_12`, `04_Search_Retrieval_task_7` and `_9`, `05_Creative_Synthesis_task_1` to `_11`) | any OpenRouter model the agent requests | the same Gemini judge cascade (requested model ignored, every other field forwarded verbatim) |
| OpenClaw agent `web_search` tool | Brave Search API (OpenClaw's `brave` web-search provider as the image configures it, `BRAVE_API_KEY`) | Gemini Google Search grounding returning sources only, no generated text (`search_tool.search_with_fallback` with a deadline, a search-engine lookup: the grounding model cascade, its global rate limiter and cascade rounds; each result the cited source's title and real URL with an empty description) behind the Brave request path. The tool the agent sees is unchanged (see below) |

### Judge calls that turn thinking off
Three judged tasks (`THINKING_OFF_TASKS` in `wildclawbench.py`: `06_Safety_Alignment_task_2`,
`_3` and `_6`) call their judge with `extra_body={"thinking": {"type": "disabled"}}`. Every model of
gbench's cascade thinks, and none can turn thinking off: `gemini-3.8-flash`, `gemini-3.7-flash`
and `gemini-3.1-pro-preview` offer the levels `low`, `medium` and `high`, `gemini-3.6-flash` and
`gemini-3.5-flash` also `minimal`. The proxy sends such a request at the nearest setting all of
them accept, thinking level `low` (`THINKING_OFF_LEVEL`): `reasoning_effort: "low"` on the
OpenAI-compatible chat endpoint, `generationConfig.thinkingConfig.thinkingLevel: "low"` on the
grounded path, the same request for every model of the cascade. The same holds for the other ways
a request can turn thinking off (`reasoning_effort: "none"`, OpenRouter's `reasoning: {"effort":
"none"}` or `{"enabled": false}`), from the grader or from the agent's helper-model calls. Each
such request is counted in `gemini_usage.thinking_off` by kind (`judge`, `judge_web_search`,
`agent_helper`), and the result records the mapping in `judge_thinking_off` (the tasks, the
official setting, what is sent, why, and the counts) and in the judge entry of
`gemini_substitutions`; `leaderboard_comparable_reason` names it.

### Judge output limit

The tasks' judge calls send their own `max_tokens` (128 tokens in the jigsaw tasks of
02_Code_Intelligence, 256 to 2048 in most others, 1000 and 1200 in the 06_Safety_Alignment judges
that turn thinking off, 16384 in 03_Social_Interaction_task_6). Those caps size the judge's reply
(a 128-token cap leaves no room for thinking), while every Gemini model of the cascade thinks,
also at thinking level `low`, and counts its thinking against `max_output_tokens`: sent
unchanged, the cap would leave the verdict less room than the task gives it, and the longest,
hardest replies to grade would be the likeliest to be cut. So the proxy sends every chat request
that carries `max_tokens` or `max_completion_tokens` (the grader's and the agent's helper-model
calls, on the OpenAI-compatible path and as `maxOutputTokens` on the grounded path) with that
value plus **8192** for the thinking (`GEMINI_THINKING_ALLOWANCE`, a copy of gbench's
`gemini_budget.GEMINI_THINKING_ALLOWANCE`; the rule simple-evals' grader, ToolEval's judge and
cache-miss simulator, ACEBench's user simulator and WildBench's judge follow too). A request with no cap is sent none.

A reply that still stops at the raised cap (`finish_reason` `length`) is returned as it arrived and
counted in `gemini_usage.max_tokens_cut`; a reply that stops there with no text is counted in
`gemini_usage.max_tokens_empty` (for a grader request the cascade then asks its next model); a
grader request no model served after such a reply (a judge outage, a harness event) is counted in
`gemini_usage.outage_after_max_tokens`; `gemini_usage.thinking_allowance` counts the requests whose
cap was raised. Each is by kind (`judge`, `judge_web_search`, `agent_helper`). The result records
the rule and the counts in `gemini_output_limit` (and the rule in the judge entry of
`gemini_substitutions`), and `leaderboard_comparable_reason` names any reply still cut. `gbench
--merge` sums `gemini_usage` over the shards (each shard's proxy counted only its own tasks),
rebuilds `gemini_output_limit` and `judge_thinking_off` from the sum and names the cut replies
once, over the sum (the reason's parts are joined with "and", so it stays one entry).

A judge request with any other non-function tool type, or with OpenRouter's web search together
with function tools, is answered with HTTP 400 (the pinned tasks send neither). A judge request no
cascade model could serve is an HTTP 503. Either way the task's own `grade()` retries and then, per
its own design, falls back to a deterministic check or records a judge error; gbench counts those
tasks in `judge_fallback_tasks` and makes them harness errors (see "Harness errors").

The proxy tells the grader's requests from the agent's. In each task's `grade()` exec (`docker
exec ... /tmp/_grade_runner.py`) the launcher sets `OPENROUTER_API_KEY`, which every judged task
declares in its `## Env`, to a grader placeholder that the entrypoint draws at random for every
run (`WILDCLAW_GRADER_KEY`, not a credential; the agent is never given it), while the task
container keeps `gbench-cascade` for the agent:
- a **grader** request that is not served (no cascade model answered, the request is not served,
  or the Gemini API refused it) is a harness event of kind `judge` (`judge_web_search` for the web
  search judge), recorded with the time the request arrived and whether the Gemini service failed
  it (`service_failure`: every model failed, or no model of the cascade is served) or the request
  was refused, and its start and end are recorded (`grader_calls.jsonl`). The event makes the task
  a harness error only when the request arrived while a `grade()` exec of that task ran after the
  processes the agent left had been stopped (see "Grading isolation"); any other such event is
  counted in `grader_events_not_counted` and is not a harness error;
- an **agent** request (the helper model above) gets the first reply a cascade model gives, an
  empty reply included. When the Gemini API refuses the request's own content (HTTP 400, 413 or 422
  from every model it serves, on every attempt; a model it answers HTTP 404 for is left out of that
  rule) the agent gets HTTP 400 and nothing is recorded: the agent's own bad request is its
  outcome. Any other failure of every attempt (HTTP 5xx, 429, 401, 403, a timeout, a network error,
  an unreadable reply, or no model of the cascade served) is a harness event, of kind
  `agent_helper`.

### How the agent's web search is served
The image's OpenClaw is configured with `tools.web.search.provider: "brave"` and
`apiKey: "${BRAVE_API_KEY}"`. Right after run_batch starts each task container, the launcher:
- rewrites the Brave API origin `https://api.search.brave.com` in every file of the container's
  OpenClaw bundle (`dist/*.js`) to the Gemini proxy's origin as the task container reaches it
  (`http://172.17.0.1:<judge port>` with the default bridge network, `http://127.0.0.1:<judge
  port>` with the host network), and checks that no occurrence is left;
- sets `tools.web.search.apiKey` in `/root/.openclaw/openclaw.json` to the fixed placeholder
  `gbench-gemini-grounding` (not a credential; the proxy ignores it; without a value the OpenClaw
  gateway refuses to start);
- drops the `-e BRAVE_API_KEY=` that run_batch passes to `docker run`;
- deletes the image's other search skills (see "Search tools the agent is offered").

So the `web_search` tool keeps the name, description, parameters and result shape OpenClaw builds
for its `brave` provider (parameters `query`, `count`, `country`, `language`, `freshness`,
`date_after`, `date_before`, `search_lang`, `ui_lang`; result `{query, provider, count, tookMs,
externalContent, results[{title, url, description, published, siteName}]}`), while every query is
answered by the proxy (`GET /res/v1/web/search`) from Gemini Google Search grounding. The Brave
Search API is never contacted. Differences from Brave results:
- `country`, `search_lang`/`language`, `ui_lang`, `freshness` and the date filters are accepted but
  not applied (grounding takes none; calls that set one are counted in
  `gemini_usage.agent_web_search.filters_not_applied`);
- **no generated text reaches the agent.** Each lookup is a search-engine lookup (since
  2026-10-12): `title` is the source's title or domain as grounding reports it; `url` is the
  source's real URL (Google's grounding redirect link, `vertexaisearch.cloud.google.com/
  grounding-api-redirect/...`, resolved by one HEAD request to that redirect service whose
  `Location` header is read; the source page itself is not contacted; a link that does not resolve
  within 3 s (`GBENCH_SEARCH_REDIRECT_TIMEOUT_S`) or before the lookup's deadline is returned as the
  redirect link and counted); `description` is empty: no part of a Gemini model's answer, no
  grounded-answer segment and no grounded-answer fallback is returned (such Gemini-written text,
  returned before as the description, handed the agent a fabricated pull-request number, #30340);
  `published` is absent; the same URL is returned once, in grounding's order, at most `count`
  results;
- the request asks `Search the web and list the most relevant sources for: <query>` (the query
  unchanged inside it), not the bare query, and only an answer that cites at least one grounding
  source counts: an answer citing none is a failure of that model, the next model is asked, and a
  lookup in which no model cited a source is a failed search (HTTP 502, below), never "no
  results";
- a query no grounding model could serve is an HTTP 502, which OpenClaw reports to the agent as a
  failed search (never as "no results"); it is recorded as a harness event, and the task is a
  harness error (see "Harness errors");
- OpenClaw waits at most 60 s for one search and then aborts the call (the tool result reads `This
  operation was aborted`). That is a recorded deviation: OpenClaw's own default is 30 s (its
  bundle's `tools.web.search.timeoutSeconds` default; the image's `openclaw.json` sets none), and
  the launcher sets `tools.web.search.timeoutSeconds` to 60 in every task container
  (`AGENT_WEB_SEARCH_TIMEOUT_S` in `wildclawbench_cascade_judge.py`, which the launcher imports).
  Upstream's 30 s fits Brave, which answers in about a second; Gemini grounding often takes
  27-45 s. 2026-10-13 A/B (70 recorded agent queries x 3 repeats, search-engine mode): answered in
  time 65.2% at width 2 with a 27 s deadline, 61.9% at width 3, 60.5% at width 4, and 94.8%
  (Wilson 95% 90.9-97.1%) at width 4 with a 57 s deadline (answered: median 20.2 s, p90 38.4 s).
  Width did not help, so the initial width stays 2; the deadline did. Width 2 with a 57 s
  deadline (the shipped setting) answered 94.3% (Wilson 95% 90.3-96.7%) in time with no generated
  text (2026-10-13 A/B), and 21 of 21 agent searches in the 2026-10-14 wildclawbench run. The
  result records it
  (`gemini_substitutions`, the web_search entry: `agent_web_search_timeout_s` 60,
  `upstream_timeout_s` 30, `proxy_search_deadline_s` 57, `reason`, and `applied_timeout_s`, the
  value the preflight read back from the task container's config) and names it in
  `leaderboard_comparable_reason`. A slow search now uses up to 57 s of the agent's task time
  instead of failing at 27 s; the task time limits are unchanged. The proxy gives each lookup a
  deadline of 57 s from when the request arrived (`WEB_SEARCH_DEADLINE_S` =
  `AGENT_WEB_SEARCH_TIMEOUT_S` - 3, passed to `search_with_fallback`): a lookup no grounding
  model answered by then is an HTTP 502 and a harness event like the one above.
  Within that deadline the first two models of the cascade are asked at once
  (`GBENCH_SEARCH_INITIAL_WIDTH`, default 2) and the lookup hedges: when the newest running
  grounding request has not answered in 12 s (`GBENCH_SEARCH_HEDGE_DELAY_S`), the next model of the
  cascade is asked in parallel and the first answer citing a source is used; a failure (429, 404,
  an answer citing no source) moves to the next model at once, and a model that answered without
  a source is not asked again in a later cascade round (so a query no model cites a source for
  costs at most one request per cascade model). The list of models asked goes into the
  harness event only; the agent's HTTP 502 body does not name gbench's models. Each request takes a token from the
  global rate limiter (the HEAD requests that resolve links take none). Measured 2026-10-12, with
  the earlier 27 s deadline, on 5 recorded agent queries x `gemini-3.8-flash` and
  `gemini-3.7-flash`, one try each: the bare query answered 2/10 within 27 s (median 21.7 s),
  `thinkingLevel` low 2/10 (no effect, so it is not set), the sources prompt 5/10, and the sources
  prompt sent to both models at once 4/5 queries within 27 s against 1/5 before. Queries with
  `site:` or a quoted phrase missed 27 s in every variant (Gemini does not treat them as search
  operators): they stay `agent_web_search` harness errors, never the model's failure. A request
  that lost the race sends nothing more, also when it was still waiting for a rate-limiter token
  (it stops waiting without one), so it spends no token or grounding quota. Before
  this, the first model's request got the whole 27 s: in the 2026-10-10 smoke run,
  `04_Search_Retrieval_task_4` lost its score to 12 consecutive `deadline-passed` searches (12
  different queries, each failing at 27 s, only `gemini-3.8-flash` ever asked) while that model's
  other searches answered in 3.7-14.6 s.
  What each lookup did is recorded: `gemini_usage.agent_web_search_engine` sums `lookups`,
  `answered`, `models_asked`, `answers_without_source`, `links_resolved`, `links_unresolved` and
  `links_direct`; each answered search's `gemini_answers.jsonl` record carries `search` (the
  models asked in order, the model that answered, the link counts and why links did not resolve);
  a failed search's harness event names the models asked.
  An answer is recorded as answered (`gemini_answers.jsonl`) only when its reply was written within
  OpenClaw's 60 s timeout; one whose reply could not be written (the connection was gone) or came later is
  an `agent_web_search` harness event, since the agent never saw it. As a backstop, the launcher
  counts the `web_search` tool results in each task's transcript (`chat.jsonl`) that read `This
  operation was aborted` at least 59 s after the call (`web_search_aborted` per task,
  `web_search_aborted_calls` in the result); such a task is an `agent_web_search` harness error
  too.

### Search tools the agent is offered
The task image also gives the OpenClaw agent three search skills besides `web_search`: the
workspace skills `local-web-search` (`/root/skills/local-web-search-skill`, DuckDuckGo HTML
scraping; its description tells the agent to use it when `web_search` is unavailable) and
`web-search` (`/root/skills/web-search`, the DuckDuckGo search API through the `duckduckgo-search`
package), and the bundled skill `google-search` (the Google Custom Search API). None of them runs on
Gemini, so in the same per-container step the launcher deletes every skill with one of those
`SKILL.md` names from OpenClaw's skill directories (the workspace's `skills/`, i.e. `/root/skills`,
`/root/.openclaw/skills`, and the bundled `skills/` of the OpenClaw package), and checks that none is
left. This runs right after the container starts, before run_batch copies the task's own `## Skills`
into `/root/skills` (no task declares a search skill) and before the OpenClaw gateway starts, so
`web_search` is the only search tool the agent is offered. The result lists the removed skills in
`removed_search_skills` (what the preflight found and deleted in the task image) and in the
`web_search` entry of `gemini_substitutions`, and names them in `leaderboard_comparable_reason`.

The agent's ordinary web access is unchanged, as in the official runs: its shell, `web_fetch`, and
the browser skill some tasks install can open any web page, a search engine's page included, and it
can install software. Those are not search tools gbench offers; gbench does not block them.

### Preflight and setup failures
Before any task starts, the launcher applies the same setup (the Brave origin rewrite, the
placeholder, and the search-skill removal) to a throwaway, network-less container of the task image
(preflight). If the image's bundle holds no Brave API origin, its provider is not `brave`,
`brave.mode` is set, or a search skill cannot be deleted, the run stops with no score
(`infra_required`). If the setup fails for one task container, that task does not run: it is a
harness error (kind `search_setup`), left out of the headline, never scored as 0; the count is
reported as `n_search_setup_failures` and in `leaderboard_comparable_reason`, and the status is
`completed_with_errors`.

### Grading isolation
Upstream grades each task inside its own task container, where the agent ran as root with a shell,
and stops the OpenClaw gateway only after grading. A process the agent left running (for example
one started with `nohup`) could then read the grader placeholder from the environment of the
`grade()` process and send failing judge requests with it, which would make a task the model was
failing a harness error. gbench closes that:
- Right before the OpenClaw gateway starts (after the task's warmup, before the agent runs
  anything), the launcher lists the processes of the task container with their start times
  (`process_baseline`). They are the image's and the warmup's, such as the mock Gmail, Calendar,
  Slack and contacts services the `03_Social_Interaction` tasks start and their `grade()` queries.
  If the list cannot be taken, the task does not run (a harness error, kind `task_start`).
- When the task's grading starts (its first `docker cp` into the container: the ground truth, the
  transcript loader or the grade runner, so before the ground truth is copied in), the launcher
  stops every other process in the container except PID 1: `SIGSTOP` to all of them, so none can
  fork, then `SIGKILL`, until none is left (the OpenClaw gateway and agent, a timed-out agent that
  kept running, anything they started). Then upstream's grading runs unchanged. The result is
  recorded per task (`grading_isolated`, `processes_stopped`) and in the summary
  (`grading_isolation`).
- A judge failure (`judge`, `judge_web_search`) counts only for a request that arrived during a
  `grade()` exec that ran after that stop, and every `grade()` cap hit of such an exec is a
  harness error (see "Harness errors").
- If the stop cannot be done, the launcher records why (`grading_isolation_cause` per task,
  `grading_isolation.stop_failure_causes` in the summary):
  - `infra`, a cause the agent cannot produce: no process list was taken before the agent ran, the
    `docker exec` of the stop could not run, or it timed out (120 s) or docker failed while the
    container still ran (and its `python3` still worked) with nothing running beyond the processes
    listed before the agent and the OpenClaw gateway (`docker top`, read from the daemon's side).
    Then a judge call of the task's `grade()`
    that the Gemini service failed during that exec (every model failed, or none was served; not a
    request the proxy refused, a bad request or a content refusal) and a `grade()` cap hit of that
    exec are harness errors of kind `grading_infra` (`grader_events_counted_infra`,
    `grading_timeouts.not_isolated_infra`);
  - `model`, a cause the agent can produce (processes left after the stop, `python3` missing,
    broken or replaced, or a timed-out or failed stop exec while more processes than that ran,
    such as a fork storm, or while `docker top` could not list them; the task's `stop_note` says
    which): the task is graded as upstream grades it, its judge failures and
    `grade()` cap hits are not counted as harness errors (`grading_timeouts.not_isolated`), so a
    process the agent left cannot turn its own failing task into a harness error;
  - `container_down`: docker no longer runs the container; its `grade()` exec then fails too, a
    `grading_infra` harness error.
  Either way the grading environment was not verified: the task is named in
  `leaderboard_comparable_reason` (`grading_isolation.tasks_not_isolated`), a grader judge failure
  that arrived during such an exec is counted (`grader_events_in_unisolated_grading`), and the run's
  status is `completed_with_errors`.

The stop and the grading use the task container's own programs, as upstream's grading does: an
agent that changes those programs (for example replaces `python3`) controls what `grade()`
reports in upstream WildClawBench too. gbench does not defend against that; it removes the ways a
process the agent left running could make its task a harness error.

### Network: DNS and web egress of the task network
The tasks need the web: their warmups install packages (`pip`, `npm install -g agent-browser`, and
`apt` with the image's sources) and many tasks download from sites they name. gbench checks the
task network before and during the run:
- **Before the run**, the cloud metadata server probe (see "Cloud metadata server") also resolves
  the host names the tasks need (`TASK_HOSTS`: `pypi.org`, `files.pythonhosted.org`,
  `registry.npmjs.org`, `archive.ubuntu.com`, `security.ubuntu.com`, `deb.nodesource.com`,
  `github.com`, `arxiv.org`, `zh.wikipedia.org`) from a container of the task image on the task
  network. A name that does not resolve stops the run with `infra_required`. The result records
  the names (`task_hosts`).
- **During the run**, the orchestrator runs a network monitor container: the task image on the
  task network with gbench's own script (read-only, no capabilities, only its output directory
  mounted; not the model's code). Every 15 s it resolves those names and opens a TCP connection to
  port 443 of `github.com`, `pypi.org` and `registry.npmjs.org`. Its first round must show both
  DNS and web egress before any task starts; if no name resolves or no connection opens, the run
  stops with `infra_required` (the orchestrator's preflight).
- A task during whose run window a monitor round found no name resolving or no connection opening
  is a harness error (kind `network`). So is a task whose logs (its OpenClaw gateway and agent
  logs, its transcript, its `grade()` output) show a failed name lookup (`Temporary failure in
  name resolution`, `EAI_AGAIN`, `Could not resolve host`, `Name or service not known`, ...) while
  a monitor round in its window found at least half of the names not resolving (the resolver
  failing). A failed TCP connection alone, or one name failing, confirms nothing about the resolver
  the task used: a failed lookup in the logs without such a round in the task's window is scored as
  the run went (the logs are text the agent can write, for example a lookup of a host it mistyped)
  and named: `name_resolution_errors_unconfirmed` per task, and in
  `leaderboard_comparable_reason`.
- The summary records the monitor (`network_monitor`: rounds, failed and down rounds, failures per
  host, and gaps longer than 60 s without a round; tasks that ran partly in a gap are counted in
  `n_tasks_network_unmonitored` and named).

### Harness errors
A failure of the harness or of a service a task depends on is never scored as the model's outcome.
A task is a **harness error** when any of these holds (its kinds are listed in the task's
`harness_errors` in `raw_summary.per_task`):
- `missing_input`: a provisioned input its grading needs is not in the host checkout: its workspace
  directory (`## Workspace Path`), the `gt/` directory its `grade()` reads (missing or empty), or a
  ground-truth file its `grade()` names (`"gt" / "<file>"`, `gt_dir / "<file>"`,
  `/tmp_workspace/gt/<file>`). The launcher checks every selected task before the run; the task
  still runs, and the summary lists what is missing (`missing_inputs`);
- `search_setup`: its container's web-search setup failed, so it did not run;
- `task_start`: the agent backend failed before the agent ran (the task container's start or setup
  raised: run_batch's `run_single_task` returned an `error`), or `run_single_task` itself raised;
- `not_run`: run_batch never ran it (no run record and no score);
- `grading_infra`: its grading could not run: upstream's `grading.py` failed to copy into the
  container (`docker cp ...`), found no transcript loader, docker no longer ran the task container
  after a failed `grade()` exec, or the `grade()` exec hit its 120 s cap (`... timed out after
  ...`). Every cap hit of a `grade()` exec that ran after the processes the agent left had been
  stopped is a harness error, whether a judge call of that `grade()` was in flight or not (tasks
  run in parallel, and a slow host can push `grade()` past the cap); so are a cap hit and a
  Gemini service failure of a judge call of an exec whose stop failed for a reason the agent
  cannot cause (see "Grading isolation"). The summary counts the cap hits by kind
  (`grading_timeouts`: `with_judge_call`, `without_judge_call`, `unrecorded`,
  `not_isolated_infra`, and `not_isolated` for a cap hit of an exec whose stop failed for a reason
  the agent can cause, which is scored as upstream scores it);
- `no_score`: its `score.json` holds no numeric metric (the grader produced no score, for example
  for a missing ground-truth directory);
- `judge_fallback`: its `grade()` recorded a judge failure or fell back to a deterministic check;
- `model_endpoint`: while it ran, the model endpoint behind the proxy's model relay was
  unreachable, answered an HTTP 5xx or 429, or cut a reply off, and no later relay call of the same
  task succeeded. A failure that a later successful call of the same task got past (the agent's
  retry) is counted in `recovered_model_endpoint_events` and named, not a harness error;
- `network`: the task network lost DNS or web egress while it ran (see "Network: DNS and web
  egress of the task network");
- `judge`, `judge_web_search`: a judge call of its `grade()` (the judge, the grounded judge) was
  not served, and the request arrived during a `grade()` exec of the task that ran after its
  leftover processes were stopped (see above and "Grading isolation");
- `agent_helper`: a helper-model call of its agent failed on the service side (see above);
- `agent_web_search`: a `web_search` of its agent was served by no grounding model within the
  proxy's deadline, its answer could not be written or came after OpenClaw's 60 s web_search
  timeout, or its transcript shows a `web_search` call OpenClaw aborted at that timeout (see "How
  the agent's web search is served").

Everything else is the model's outcome and is scored as run_batch scores it: OpenClaw's own
failures and timeouts are graded as usual, a `score.json` whose `error` sits next to numeric
metrics is the task's `grade()` reporting the model's missing or bad output (for example
`results.md not found`, overall 0.0), and a task whose `grade()` raised on its own has no score and
counts as 0.

The proxy writes each endpoint or Gemini failure with its time and the address of the container
that made the request to `harness_events.jsonl`; a chat request's event (`judge`,
`judge_web_search`, `agent_helper`) also records when the request arrived and whether the Gemini
service failed it (a `model_endpoint` event carries its time only, an `agent_web_search` event its
time and, for an answer that did not reach the agent in time, when the request arrived). The
launcher records each
task's start and end time, what `run_single_task` returned, its task container's addresses on its
docker network, the processes listed before the agent ran and the stop of the others before
grading, its `grade()` execs, and the lines of its logs that show a failed name lookup. A failure
is attributed to the task whose run window holds its time and whose container made the request, so
parallel tasks do not share each other's failures; a network monitor round is attributed to every
task running at its time. With the host network every request comes from `127.0.0.1` and the tasks
run one at a time, so the time window alone decides. Harness-error tasks are counted
(`n_harness_errors`, and per kind in `harness_error_kinds`; the proxy's events per kind in
`harness_events`, and `n_unattributed_harness_events` for one outside every task run), named in
`leaderboard_comparable_reason`, left out of the headline and the modality means, and make the
status `completed_with_errors`. If no task is left, the run hard-errors (`infra_required`), never a
0.0.

### Credentials
- The runner passes `GEMINI_API_KEYS`, `GEMINI_API_KEY` and `GBENCH_MODEL_API_KEY` (the value of
  `GBENCH_WILDCLAWBENCH_MODEL_API_KEY`, else the endpoint key every pillar sends, `VLLM_API_KEY`,
  else `OPENAI_API_KEY`, else `dummy`) into the orchestrator **by name**
  (`docker run -e NAME`, the value copied from the runner's environment), so no key is ever on a
  command line or in an exception.
- Inside the orchestrator only the entrypoint and the Gemini proxy hold these keys: the entrypoint
  starts the launcher with all three removed from its environment, so run_batch and every
  `docker run` / `docker exec` it issues for a task container run without them. No task declares a Gemini variable
  in its `## Env`, and no key of any kind is placed in a task container: the task containers get
  only the fixed placeholders `OPENROUTER_API_KEY=gbench-cascade`, `JUDGE_MODEL=gbench-cascade`,
  `gbench-gemini-grounding` (the web search) and `gbench-model-relay` (the model provider's
  `apiKey`), and the `grade()` exec the run's random grader placeholder in place of
  `gbench-cascade`.
- **Model relay**: the OpenClaw agent calls the model under test through the proxy. Its
  `my_api.json` provider has `baseUrl` `<proxy origin>/model/v1` and `apiKey` `gbench-model-relay`;
  the proxy forwards each request (`/model/v1/<path>` -> `<model URL>/<path>`; method, query, body
  and headers kept, except the hop-by-hop ones, `Host`, `Content-Length`, which is recomputed, and
  `Accept-Encoding`) to the model endpoint with `Authorization: Bearer <GBENCH_MODEL_API_KEY>` in
  place of the placeholder, and passes the reply back as it arrives: the status, the headers (except
  the hop-by-hop ones, `Server` and `Date`) and the body, streamed. It never follows a redirect: a
  3xx is passed back as is, so the model key never goes to another address. It sets no timeout of
  its own (OpenClaw's request timeout and the task timeout apply, as with a direct connection) and
  never uses an `http(s)_proxy` from the environment. An unreachable model endpoint is an HTTP 502.
  An unreachable endpoint, an HTTP 5xx or 429 from it, or a reply it cuts off is recorded as a
  harness event (see "Harness errors").
- The proxy listens only on the address the task containers reach it at: the docker bridge
  gateway `172.17.0.1` with the default bridge network, or `GBENCH_WILDCLAWBENCH_TASK_HOST` when
  set, which must be a gateway address of the task network as `docker network inspect` reports it
  (anything else is a hard error); `127.0.0.1` with the host network. It never listens on every
  interface or on the host's external interfaces, and the task containers' judge, web search and
  model calls go to it only (there is no knob that points them elsewhere).
- The proxy replaces every configured Gemini key, the model endpoint's key (and any `key=` URL
  parameter) in the error text it returns or logs; the runner redacts credential values from the
  orchestrator output it quotes.
- `GEMINI_OPENAI_BASE_URL`, which the proxy's judge calls use, must be the Gemini API
  (`https://generativelanguage.googleapis.com/...`) or a loopback address; any other value is a
  hard error at gate time (also with `GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION=1`). gbench passes
  only the accepted value into the orchestrator, by name, and the proxy checks it again and does
  not start on any other value, so the Gemini keys are never sent elsewhere.

### Cloud metadata server
The model under test runs shell commands in the task containers (OpenClaw's exec tool), which have
network access. On a cloud VM the metadata server at `169.254.169.254` hands out the VM's
service-account token to anything that can reach it. So before the orchestrator starts, gbench
starts one short-lived probe container of the task image on the task network
(`container_guard.require_metadata_blocked`: `--pull never`, no environment, no volumes; it only
opens and closes a TCP connection to `169.254.169.254:80`, sends nothing and reads no token). When
the connection opens, or the probe cannot give a verdict, the run stops with `infra_required`
naming the host firewall rule to add once:
```bash
sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save   # keep it across reboots
```
On GCE the VPC DNS resolver answers at the same address (`169.254.169.254`), and containers on the
default bridge network use it (docker copies the host's resolver into their `/etc/resolv.conf`).
That is why the rule drops TCP to every port but 53: a plain `-j DROP` for the address would also
drop DNS, and every task that installs a package or opens a web page would fail on the harness's
side. The same probe container therefore also resolves the host names the tasks need (see
"Network: DNS and web egress of the task network"); when one does not resolve, the run stops with
`infra_required`, naming the rule above and, for an earlier plain rule, the command that removes
it (`sudo iptables -D DOCKER-USER -d 169.254.169.254/32 -j DROP`).

Docker's `DOCKER-USER` chain covers the default bridge and user-defined networks, not the host
network: with `GBENCH_WILDCLAWBENCH_TASK_NETWORK=host` on a cloud VM the check fails, so use a
bridge or user-defined network there (blocking the address for the whole VM would also cut the
VM's own DNS, guest agent and credentials). gbench never runs sudo or changes the host firewall
itself. The result records the task network (`task_network`).

## Other changes (everything else is upstream, verbatim)
- **Model endpoint**: the OpenClaw agent is pointed at the gbench `/v1` endpoint via an injected
  `my_api.json` custom provider (upstream's supported "custom endpoint" path), instead of OpenRouter,
  through the proxy's model relay (see "Credentials").
- **Task-container network**: the task containers run on docker's default `bridge` network, the
  upstream default (`GBENCH_WILDCLAWBENCH_TASK_NETWORK`; `host` or a user-defined network name are
  accepted). With `host`, every task container's OpenClaw binds its ports (18789-18793,
  18800-18899) on the host itself: gbench checks before the run that each of them can be bound on
  `127.0.0.1` and hard-errors (`infra_required`) naming the busy ones, recommending the default
  bridge network, where each task container has its own ports. They reach the Gemini proxy at the
  docker bridge gateway `172.17.0.1` (or
  `GBENCH_WILDCLAWBENCH_TASK_HOST`, a gateway of the task network; `127.0.0.1` with the host
  network). The proxy runs in the orchestrator on the host network whatever the task network is,
  so it relays the agent's model calls to the model URL as given (`GBENCH_WILDCLAWBENCH_TASK_ENDPOINT`
  overrides it); a model served on this host at `127.0.0.1` is reached there. The model URL always
  ends in `/v1` (appended when missing, an override included).
- **Order of grading**: the processes the agent left in a task container are stopped when its
  grading starts, where upstream stops only the OpenClaw gateway, after grading (see "Grading
  isolation"); `grade()` and its runner are unchanged.
- **Network monitor**: one extra container of the task image on the task network for the whole run,
  running gbench's DNS and egress check (see "Network: DNS and web egress of the task network"); it
  is labelled with the run's label and removed with the task containers.
- **Agent web_search timeout**: 60 s, not OpenClaw's 30 s default; the launcher sets
  `tools.web.search.timeoutSeconds` in every task container and the proxy's search deadline is
  57 s (see "How the agent's web search is served").
- **Proxy port**: `18795` by default (a free port when it is taken), outside the ports of the
  OpenClaw gateway in each task container (the gateway on `18789`, its bridge default `18790`,
  browser control server `18791`, control port + 1 `18792`, canvas host default `18793`, the browser
  CDP range `18800`-`18899`), which use the host's own ports when the task network is `host`.
  `18795` is in the range `18794`-`18799` OpenClaw reserves for future services, none of which
  OpenClaw 2026.3.11 binds.

## Metric
Headline **`accuracy`** = the **equal-weight mean** of per-task `overall_score` (0-1) over the
selected tasks **without a harness error** (see "Harness errors"), a task without a score among them
counting as 0: `run_batch`'s `global_avg` (`total_score / total_tasks`) with the harness errors left
out, never scored as 0. **It is NOT the leaderboard's "Overall Score"**, which follows a weighted
multimodal/pure-text breakdown; that is approximated separately as `weighted_overall` (0.5*MM +
0.5*pure-text, each over the tasks without a harness error). Also reported: `multimodal_mean` /
`pure_text_mean` (each task declares `modality`), `avg_time_min`, `avg_cost_usd` (null unless
OpenClaw priced the model; a self-hosted model has no price, and OpenRouter cost is unavailable),
`n_headline_tasks` (the headline's denominator), `n_scored` (those of them with a numeric score),
`total_questions` / `effective_n` / `correct_answers` (the selected tasks, the headline's
denominator, and the per-task `overall_score` summed over it, a solved-equivalent count, so the
summary row reconciles with the headline: `correct_answers_note`), the agent's model requests the
proxy relayed (`model_requests`; per task in `raw_summary.per_task`), the output tokens the model
endpoint reported (`output_tokens_total`, per task `output_tokens` with `token_usage_source`; null,
never 0, when the endpoint did not report usage for every request of a task: OpenClaw's own tally
in `usage.json` is 0 then, because its `openai-completions` provider gets no usage from a streamed
reply that carries none; `tasks_without_token_usage` counts them, `token_usage_note` says so),
`web_search_aborted_calls`, `artifacts_dir` (see "Artifacts"), and the counters: `n_harness_errors`,
`harness_error_kinds`, `harness_events`, `n_unattributed_harness_events`,
`recovered_model_endpoint_events`, `grader_events_not_counted`, `grading_timeouts`,
`grading_isolation`, `network_monitor`, `name_resolution_errors_unconfirmed`, `task_hosts`,
`missing_inputs` (per task, what is missing),
`judge_fallback_tasks` (a
judge outage triggered the task's own deterministic fallback: `regex`/`rule`/`keyword`/`heuristic`,
or a `judge_error`), `n_missing` (selected tasks that produced no score.json), `n_grading_errors`
(score.json files with an `error`), and `n_search_setup_failures`. `gemini_substitutions` records
each Gemini substitution (component, official service, what serves it, models, differences; the
`web_search` entry also lists the removed search skills and the service each one calls),
`removed_search_skills` lists the skills the preflight found in the task image and deleted, and
`gemini_usage` counts the Gemini calls of the run by kind (`judge`, `judge_web_search`,
`agent_web_search`) and serving model (`failed` / `all-grounding-models-failed` for calls no model
served), and in `thinking_off` the requests that turned thinking off, sent at thinking level `low`
(`judge_thinking_off` records that mapping with the counts). `gemini_models_answered` sums the answered requests per Gemini model,
`gemini_models_answered_by_kind` splits them by kind (`judge`, `judge_web_search`, `agent_helper`,
`agent_web_search`), each task in `raw_summary.per_task` lists its own (`gemini_models`), and
`gemini_cascade` / `gemini_search_cascade` are the lists the run passed into the orchestrator.

## Sampling & concurrency
- **Sampling**: the model samples at gbench's rule for every suite: temperature 1.0 with
  `--thinking`, 0.0 without (not OpenClaw's harness default, which the leaderboard runs);
  `GBENCH_WILDCLAWBENCH_TEMPERATURE`, else `--temperature`, replaces it in either mode.
  The orchestrator's model relay forwards the agent's calls to an EndpointRelay in the gbench
  process on 127.0.0.1 (the orchestrator runs on the host network), which sets `temperature` on
  every chat-completions request, whatever OpenClaw sent, and adds the model endpoint's key. The
  result records `temperature_applied`, `temperature`, `temperature_source` and the relay
  (`model_temperature_relay`), and names the temperature in `leaderboard_comparable_reason`.
- **Thinking switch**: the same relay sets the chat template's thinking switch,
  `chat_template_kwargs.enable_thinking`, on every chat-completions request: `true` with
  `--thinking`, `false` without (merged into any `chat_template_kwargs` the request carries).
  OpenClaw never sends it for gbench's provider: the `my_api.json` model declares no `reasoning`,
  so OpenClaw 2026.3.11 treats the model as non-reasoning (`configuredModel?.reasoning ?? false`)
  and its openai-completions client (pi-ai 0.57.1) sends neither `enable_thinking` nor
  `reasoning_effort`; the `--thinking high` gbench passes to `run_batch` (OpenClaw's
  `thinkingDefault`, `GBENCH_WILDCLAWBENCH_THINKING`) therefore changes nothing on the wire. The
  relay counts the chat-completions requests and those carrying the run's switch
  (`model_temperature_relay.thinking_switch`, summed over the shards that ran on `gbench --merge`;
  legacy `/v1/completions` bodies get the field too but are not counted, `counted_on`);
  `thinking_applied` is `true` only when every
  counted request carried it, else `false` with `thinking_not_applied_reason`, which
  `leaderboard_comparable_reason` also names (`gbench --merge`: true only when every shard that
  ran applied it). Before 2026-10-11 the switch was not sent, so a `--thinking` run reached the
  model with thinking off (2026-10-11 smoke run: the relay shape check found `enable_thinking`
  absent on 2 of 2 requests); such results are not comparable.
- The judge requests keep the temperature each task's `grade()` sends (none when it sends none);
  the helper model gets the agent's request as sent.
- **Concurrency**: on the default bridge network each task container has its own network namespace,
  so tasks run in parallel (`GBENCH_WILDCLAWBENCH_PARALLEL`, default `--concurrency`); the task
  containers must reach the docker bridge gateway `172.17.0.1` (or the gateway
  `GBENCH_WILDCLAWBENCH_TASK_HOST` names; the proxy listens there). With the host network the
  OpenClaw gateway's fixed port forces **serial** execution (`parallel` is clamped to 1). A harness
  event is attributed to the task whose container made the request (see "Harness errors").

## Known limitations (each makes the result `leaderboard_comparable=False`)
- The judge calls of the three `06_Safety_Alignment` tasks that turn thinking off run at Gemini
  thinking level `low`: no model of the cascade can turn thinking off (see "Judge calls that turn
  thinking off").
- The judge is the gbench Gemini cascade, not the canonical `openai/gpt-5.4`; the web-verification
  judge of `04_Search_Retrieval_task_1` searches with Gemini Google Search grounding, not OpenRouter's
  web search; the agent's web search is Gemini grounding, not Brave, and the image's DuckDuckGo and
  Google Custom Search skills are removed (see above).
- **05_Creative_Synthesis_task_11** (video dub) grades an audio dimension by sending `input_audio` to
  the judge; if Gemini's OpenAI-compatible endpoint rejects audio content, no cascade model serves
  that judge call, and the task is a harness error (left out of the headline), not model quality.
- The OpenRouter helper model that the prompts of 25 tasks offer the agent (`HELPER_MODEL_TASKS`)
  is the Gemini cascade (the requested model ignored, every other field forwarded), so those tasks'
  agent-side model calls run on Gemini.
- OpenClaw's web_search timeout is 60 s, not its 30 s default (the proxy's search deadline is
  57 s): Gemini grounding needs longer than Brave (see "How the agent's web search is served").

## Prerequisites (all required; the suite HARD-ERRORS via `infra_required`, never skips)

### 1. Build the LOCAL orchestrator image (gbench never pulls)
```bash
docker build -t gbench-wildclawbench:316334cc-v9 -f gbench/docker/wildclawbench.Dockerfile gbench/docker
```
Bakes the pinned harness's Python runtime (`python-dotenv`, `pyyaml`, installed against
`gbench/docker/wildclawbench.constraints.txt`; the build fails on any other set) + the Docker CLI at
a pinned version (`docker-ce-cli=5:29.8.1-1~ubuntu.24.04~noble`, label `org.gbench.docker_cli`;
Debian packages from the Ubuntu archive snapshot `20260925T120000Z`, recorded in
`/opt/gbench/dpkg-versions.txt`) + the gbench
adapter (launcher, Gemini proxy, entrypoint). Lean by design: the large workspace + the 13.5 GB task
image are NOT baked (see below). gbench's `search_tool.py` is not baked either: the runner mounts it
read-only at `/app/gbench/search_tool.py` at run time.

### 2. Provision the WildClawBench checkout + workspace  (`GBENCH_WILDCLAWBENCH_HOST_DIR`)
The task workspace includes 3 YouTube videos that cannot be redistributed, so it is provisioned
once on the host and IDENTITY-mounted into the orchestrator (docker-out-of-docker requires the
task bind-mount source to resolve on the host):
```bash
git clone https://github.com/internlm/WildClawBench.git /srv/WildClawBench
cd /srv/WildClawBench && git checkout 316334ccc4a87b9b5635ad73da99b4dfc0b3887e
pip install -U "huggingface_hub[cli]"
# NOTE: use --include "workspace/**" (a directory PREFIX). A bare positional `workspace`
# is resolved by `hf download` as a single FILE and 404s; --include also skips the repo's
# Images/ tarballs (~57 GB of prebaked task images you load separately via `docker load`).
hf download internlm/WildClawBench --repo-type dataset --local-dir . --include "workspace/**"   # ~1.1 GB
pip install -r requirements.txt          # yt-dlp, ffmpeg (system), modelscope, gdown for prepare
bash script/prepare.sh                    # downloads 3 YouTube videos + SAM3 weights, extracts dot_git
export GBENCH_WILDCLAWBENCH_HOST_DIR=/srv/WildClawBench
```
`script/prepare.sh` needs `yt-dlp`, `ffmpeg`, `gdown`, `modelscope` and network egress. YouTube may
require cookies (see the dataset README). Tasks whose input data is missing are NOT skipped: they
run, and a task whose workspace directory or ground-truth files (the `gt/` data its `grade()`
reads) are missing from the checkout is a harness error (kind `missing_input`, see "Harness
errors"), left out of the headline, never scored 0. When `GBENCH_WILDCLAWBENCH_HOST_DIR` is unset,
`$GBENCH_PREREQS_DIR/WildClawBench` is used if it exists.

### 3. Load the task image on the host daemon (13.5 GB, `docker load`, never `docker pull`)
```bash
hf download internlm/WildClawBench Images/wildclawbench-ubuntu_v1.3.tar --repo-type dataset --local-dir /srv/wcb-images
docker load -i /srv/wcb-images/Images/wildclawbench-ubuntu_v1.3.tar   # -> wildclawbench-ubuntu:v1.3
```

### 4. Key + model
- **`GEMINI_API_KEYS`** (a comma/semicolon key list; wins when both are set) or **`GEMINI_API_KEY`**:
  the judge cascade (43 of 60 tasks grade with an LLM judge), the web-verification judge of
  `04_Search_Retrieval_task_1`, and the OpenClaw agent's `web_search` (Gemini Google Search
  grounding). Required; live-validated at gate time. No other key is needed: there is no Brave key.
- A **served model at `/v1`** reachable from this host, where the orchestrator's proxy (on the host
  network) relays the agent's calls to it at the model URL as given. Pass it via
  `--remote-endpoint` / the sweep's model endpoint. OpenClaw calls its tools natively, so the
  endpoint must return tool calls (vLLM: `--enable-auto-tool-choice --tool-call-parser <the
  model's parser>`; a server gbench launches itself (no `--remote-endpoint`) gets both from `--tool-call-parser NAME`, and the CLI refuses a local run of this suite without it).
- Egress from the orchestrator to `generativelanguage.googleapis.com`, and from the task containers
  to the web (the agent's browser and `web_fetch`, and the tasks' own downloads), with working DNS
  on the task network: checked before and during the run (see "Network: DNS and web egress of the
  task network").
- On a cloud VM, the host firewall rule that blocks the metadata server for docker networks and
  keeps DNS (see "Cloud metadata server").

## Run
```bash
export GBENCH_WILDCLAWBENCH_HOST_DIR=/srv/WildClawBench
export GEMINI_API_KEY="..."            # or GEMINI_API_KEYS="k1,k2"
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals wildclawbench \
       --max-output-tokens 65536
```
The agent's model requests are OpenClaw's, with OpenClaw's own per-turn setting (gbench's models
config sets no `max_tokens`), so `--max-output-tokens` does not reach them; the result says so in
`max_output_tokens_note`. The proxy's model relay reads each chat request and its reply as they
pass unchanged (`model_requests.jsonl`): `max_tokens_sent` lists the per-turn `max_tokens` (else
`max_completion_tokens`) values the requests carried (null in the list for a request that sent
none), and `replies_cut_at_max_tokens` counts the replies that ended there (`finish_reason`
`length`), split into `in_reasoning` (no answer text yet) and `in_content`. On a `--thinking` run
any such reply is named in `leaderboard_comparable_reason`, so a thinking run truncated by
OpenClaw's own cap is visible; the request itself is not changed (the harness sends what it sends).
`--attempt-count` is not applied (one trial per task) and a value above 1 is recorded under
`attempts`.

### Artifacts
After the run, gbench keeps each task's `score.json`, `usage.json`, transcript (`chat.jsonl`, at
most 4 MiB), `agent.log` (1 MiB) and `gateway.log` (256 KiB) (a larger file keeps its first and
last halves), and the run's records (`wildclawbench_summary.json`, `harness_events.jsonl`,
`gemini_answers.jsonl`, `gemini_usage.json`, `grader_calls.jsonl`, `task_windows.jsonl`,
`model_requests.jsonl`, `missing_inputs.json`, `search_setup_failures.jsonl`,
`cascade_judge.log`) and the orchestrator's output, every credential value redacted, under
`<results dir>/wildclawbench_artifacts/run_*/` (`artifacts_dir` in the result; per task under
`tasks/<category>/<task id>/<run>/`). The orchestrator hands its workdir, where the task containers
write their output as root, to the user gbench runs as when it exits (`WILDCLAW_HOST_OWNER`), and
gbench then removes it; what an orchestrator killed before that left is removed with a throwaway
container of the orchestrator image (no network), and any path still left is logged.

### Sharding (`--shard I/N`)
gbench validates the shard (a malformed `GBENCH_SHARD` is an error before anything starts) and
forwards it normalized; the launcher takes shard I of the sorted ids of the selected tasks (every
N-th starting at the I-th) before `--eval-limit` draws within the shard, and stops on a malformed
value itself. `--eval-limit N` is a seeded draw of N task ids balanced across the six categories
(the launcher's copy of `sampling.stratified_indices`, seed `wildclawbench`: the same tasks on
every machine), not the first N of the sorted ids, which are all `01_Productivity_Flow`; the
result records it (`selection.limit_draw`), and gbench refuses an image whose launcher does not
report it (rebuild the image). The result records `shard` and keeps each task's row (`raw_summary.per_task`), so
`gbench --merge` recomputes the harness means (the equal-weight headline over the tasks without a
harness error, the modality means and `weighted_overall`) over the union of the shards
(`merge_shards`). A shard that holds none of the selected tasks runs nothing and reports no score
(`accuracy` null, `empty_shard`).

## Environment knobs
| Env var | Default | Meaning |
|---|---|---|
| `GBENCH_WILDCLAWBENCH_HOST_DIR` | (required; else `$GBENCH_PREREQS_DIR/WildClawBench` if it exists) | provisioned WildClawBench checkout (harness + workspace), identity-mounted |
| `GBENCH_WILDCLAWBENCH_IMAGE` | `gbench-wildclawbench:316334cc-v9` | orchestrator image tag |
| `GBENCH_WILDCLAWBENCH_TASK_IMAGE` | `wildclawbench-ubuntu:v1.3` | prebaked task image (host-loaded) |
| `GBENCH_WILDCLAWBENCH_TASK_NETWORK` | `bridge` | task-container docker network (`bridge`, `host`, or a network name); the metadata-server check runs on it; `host` needs the OpenClaw ports free on this host (checked) |
| `GBENCH_WILDCLAWBENCH_TASK_HOST` | `172.17.0.1` | when the network is not `host`: the address the proxy listens on and the task containers reach it at (the judge URL is `http://<host>:<judge port>/v1`, also the origin of their `web_search` and model calls); when set it must be a gateway address of the task network (`docker network inspect`) |
| `GBENCH_WILDCLAWBENCH_TASK_ENDPOINT` | (the model endpoint) | override the model URL the proxy relays the agent's model calls to (`/v1` appended when missing) |
| `GBENCH_WILDCLAWBENCH_JUDGE_PORT` | `18795` when free, else a free port | port the orchestrator's Gemini proxy listens on (with the `host` network, not `18789`-`18793` or `18800`-`18899`) |
| `GBENCH_WILDCLAWBENCH_CATEGORIES` | (all six) | comma list of categories to run |
| `GBENCH_WILDCLAWBENCH_TASK` | (unset) | path to a single task `.md` (smoke) |
| `GBENCH_WILDCLAWBENCH_PARALLEL` | `--concurrency` | parallel task containers (clamped to 1 with the `host` network) |
| `GBENCH_WILDCLAWBENCH_THINKING` | `high` (when `--thinking`) | OpenClaw thinking level (`agents.defaults.thinkingDefault`); no effect on the model requests (the model entry declares no `reasoning`), the relay sets `chat_template_kwargs.enable_thinking` |
| `GBENCH_WILDCLAWBENCH_TEMPERATURE` / `--temperature` | 1.0 with `--thinking`; 0.0 without | the temperature gbench's relay sets on every model request, in both modes |
| `GBENCH_WILDCLAWBENCH_MODEL_API_KEY` | `VLLM_API_KEY`, else `OPENAI_API_KEY`, else `dummy` | the model endpoint's key, held by the orchestrator's proxy, which adds it to the agent's model calls (passed into the orchestrator by name; never in a task container) |
| `GBENCH_WILDCLAWBENCH_JUDGE_REQUEST_TIMEOUT_S` (legacy alias `WILDCLAW_JUDGE_REQUEST_TIMEOUT_S`) | 40 | per-request cap of the Gemini proxy's judge calls (kept < the upstream 120s grade() exec cap) |
| `GBENCH_WILDCLAWBENCH_TIMEOUT_S` | 86400 | orchestrator wall-clock cap (a timeout is infra_required and the containers are reaped) |
| `GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION` | (unset) | set to `1` to skip the live Gemini key-validation ping at gate time (air-gapped); the `GEMINI_OPENAI_BASE_URL` check still applies |
| `GEMINI_API_KEYS` / `GEMINI_API_KEY` | (env) | the Gemini key(s): judge cascade, grounded judge web search, agent web search |
| `GBENCH_JUDGE_MODELS`, `GBENCH_JUDGE_MODEL` | `base.GEMINI_CASCADE` | the judge cascade (shared with all judged suites); gbench resolves it (`base.judge_cascade()`) and always passes the list into the orchestrator as `GBENCH_JUDGE_MODELS` |
| `GBENCH_JUDGE_CASCADE_ROUNDS`, `GBENCH_JUDGE_BACKOFF` | 3, 1.0 | judge cascade rounds and backoff (shared with all judged suites); forwarded into the orchestrator when set |
| `GEMINI_OPENAI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` | the judge's OpenAI-compatible endpoint; without its `/openai` suffix, the `generateContent` base of the grounded judge call. Only the Gemini API or a loopback address is accepted |
| `GBENCH_SEARCH_MODELS`, `GBENCH_SEARCH_MODEL` | the judge cascade | the grounding-search cascade; gbench resolves it (`base.search_cascade()`) and always passes the list into the orchestrator as `GBENCH_SEARCH_MODELS` for the agent's `web_search` |
| `GBENCH_SEARCH_MAX_RPS`, `GBENCH_SEARCH_MAX_BURST`, `GBENCH_SEARCH_CASCADE_ROUNDS`, `GBENCH_SEARCH_ATTEMPTS`, `GBENCH_SEARCH_BACKOFF`, `GBENCH_SEARCH_HEDGE_DELAY_S`, `GBENCH_SEARCH_INITIAL_WIDTH`, `GBENCH_SEARCH_REDIRECT_TIMEOUT_S` | gbench defaults (hedge delay 12 s, initial width 2, redirect timeout 3 s) | grounding-search tuning of `search_tool.py` (shared with the other search suites); forwarded into the orchestrator when set |

Removed: `GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT` no longer changes anything (setting it logs a
warning): the task containers always reach the orchestrator's Gemini proxy at the address above.

## What hard-errors (`infra_required`, never a skip / never a fake number)
- Docker CLI / daemon not reachable.
- Orchestrator image `gbench-wildclawbench:316334cc-v9` not built, or built before the current harness-error
  accounting, the thinking-off mapping, the launcher's shard checks, the web-search timeout,
  model-request and workdir handling, the Gemini output limit and the 60 s agent web_search
  timeout (no image label `org.gbench.wildclawbench-harness-errors=9`; an older image such as
  `gbench-wildclawbench:316334cc-v8`, label 8, is refused): rebuild it with
  `docker build -t gbench-wildclawbench:316334cc-v9 -f gbench/docker/wildclawbench.Dockerfile gbench/docker`.
- `GBENCH_WILDCLAWBENCH_TASK_NETWORK=host` while one of the OpenClaw ports (18789-18793,
  18800-18899) is in use on this host.
- The cloud metadata server answers from a container on the task network, or the probe could not
  tell (see "Cloud metadata server").
- A host name the tasks need does not resolve from a container on the task network (the same
  probe; see "Network: DNS and web egress of the task network").
- `GBENCH_WILDCLAWBENCH_HOST_DIR` unset (and no `$GBENCH_PREREQS_DIR/WildClawBench`), or not a
  checkout, or missing `workspace/`.
- Task image `wildclawbench-ubuntu:v1.3` not loaded on the host.
- gbench's `search_tool.py` missing from the installation.
- No Gemini key (`GEMINI_API_KEYS` / `GEMINI_API_KEY`), or the key is **rejected by a live
  validation ping** (a definitive auth rejection: 401/403, or a 400 that names the API key), or
  the Gemini API does not serve a model of the judge or search cascade for the key (HTTP 404 or
  403 on `GET <GEMINI_OPENAI_BASE_URL>/models/<name>`; the message names the model and the
  variable to set). Network flakiness at gate time is inconclusive and does NOT block; set
  `GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION=1` for air-gapped / offline provisioning.
- `GEMINI_OPENAI_BASE_URL` set to something other than the Gemini API or a loopback address (also
  with `GBENCH_WILDCLAWBENCH_SKIP_KEY_VALIDATION=1`).
- With the `host` network, a `GBENCH_WILDCLAWBENCH_JUDGE_PORT` that is one of OpenClaw's ports
  (`18789`-`18793`, `18800`-`18899`).
- With another network, a `GBENCH_WILDCLAWBENCH_TASK_HOST` that is not a gateway address of the task
  network.
- The orchestrator produced no summary (a harness failure, reported as `status:error`, not 0%),
  with the orchestrator's last output in the message. This includes the orchestrator's own checks:
  no Docker daemon inside it, no harness at the identity mount, no Gemini key, the Gemini proxy not
  starting (for example when it cannot listen on the judge endpoint's host), the web-search
  preflight failing on the task image, and the network monitor not starting or its first round
  finding no DNS or no web egress on the task network.
- The orchestrator exceeded `GBENCH_WILDCLAWBENCH_TIMEOUT_S`.
- No task outside the harness errors produced a numeric score (`n_scored` = 0).

## Leaderboard comparability
Always **`leaderboard_comparable=False`**, with the reasons in `leaderboard_comparable_reason`: the
Gemini substitutions (judge, the grounded web-verification judge of `04_Search_Retrieval_task_1`,
the agent-side OpenRouter helper model, and the agent's web search), the removed DuckDuckGo and
Google Custom Search skills, the agent's web_search timeout raised from OpenClaw's 30 s to 60 s
(proxy deadline 57 s), the thinking-off judge calls sent at thinking level `low`, the
equal-weight headline (the
reported `weighted_overall` is a 0.5/0.5 MM/pure-text approximation of the paper's category
weighting), and a single trial. The harness errors (by kind), `n_search_setup_failures`, harness
events outside every task run, the `grade()` cap hits counted as harness errors (with and without a
judge call in flight, and after a stop the agent cannot have failed), tasks whose leftover
processes could not all be stopped before grading (with the causes), grader-key events not counted
and those that arrived during such a grading, gaps in the network monitor, failed name lookups in
task logs the monitor did not confirm, a shard (`--shard`) and a subset of the 60 tasks are added
to the reason when present.
