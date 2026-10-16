# toolbench

Canonical ToolBench scored by **StableToolBench** (THUNLP-MT/StableToolBench, arXiv:2403.07714):
the model runs a **multi-turn DFSDT agentic loop**, calling RapidAPI tools against a **cached** API
server (deterministic; an LLM simulator covers cache misses) and finishing with a `Finish` call. The
answer trees are scored by **ToolEval**:

- **SoPR** (Solvable Pass Rate): each attempt judged solved / unsure / unsolved (1 / 0.5 / 0), mean
  over the group's queries, mean over **3 evaluation passes**, computed **per test group** exactly
  as upstream's `toolbench/tooleval/eval_pass_rate.py` scores each `--test_set`.
- **SoWR** (Solvable Win Rate): pairwise preference vs the README tables' reference, ChatGPT-CoT
  (GPT-3.5-Turbo-0613 CoT, `chatgpt_cot` in the published baseline answers, see below), also per
  group (`eval_preference.py`). In each pass every test id gets one comparison, as `base.py
  annotate_preference` and `rtl.py normalized_openai_completions` make it: the two answers
  (processed as `process_answer` does: `final_answer` cut to 1,000 characters, `answer_details` cut
  to its first 3,000 entries, `method` dropped) in shuffled order, each given a status by
  `check_is_solved`; the one `Solved` answer wins if exactly one is, else the one `Unsure` answer if
  exactly one is, else `select_better_answer` chooses (a random one when the two are identical). A
  query is a win when the candidate takes more passes than the reference; ties and losses are not.
  The rate is over the group's test ids (`solvable_queries/test_query_ids/<group>.json`, the ids
  `eval_preference.py` iterates) that the run covered: all of them on a full run, the selection
  under `--shard` / `--eval-limit` (see [Selection](#selection---shard-and---eval-limit);
  `sowr_queries_by_group` has the counts), less the queries that
  met a harness error (see [Harness errors](#harness-errors)). Every one of those test ids needs a
  reference answer, else the run hard-errors before it starts (see below). A comparison without a
  verdict (a judge outage, or no valid reply after 3 tries) is no vote: upstream's script stops
  rather than go on without one, so the query's other passes decide, and a query no pass could vote
  on is left out of the denominator (`sowr_votes_missing_by_group`,
  `sowr_queries_without_vote_by_group`); a group with no vote at all publishes no SoWR, and the run
  publishes no score (`accuracy` null with `accuracy_withheld_reason`, status `failed`, its
  records kept), because SoWR is a leaderboard column. Two details follow the
  released code rather than its apparent intent: `run_preference.sh` passes `--use_pass_rate true`,
  but the pass-rate file stores each query's verdicts as a dict string, so that shortcut never
  applies and every comparison computes its statuses afresh (without the pass-rate script's
  `Finish` check); and the `check_task_solvable` call upstream makes before
  `select_better_answer` is not made, because its result is not part of that prompt.

**Evaluation count.** StableToolBench runs every model once and evaluates it three times, then
reports the average (README: "we run all models once, evaluate them three times, and take the
average results"; `--evaluate_times 3` in its `run_pass_rate.sh` / `run_preference.sh` and in the
README commands). gbench does the same by default (`GBENCH_TOOLBENCH_EVAL_TIMES=3`); a different
value is named in `leaderboard_comparable_reason`. `sopr_std_by_group` reports the spread of the
per-pass SoPR in each group (the "Std" `eval_pass_rate.py` prints and the "+-" of the README
tables).

**Headline `accuracy` = the unweighted mean of the six per-group SoPRs**, the "Average" column of
the StableToolBench README leaderboard (I1-Inst = `G1_instruction`, I1-Cat = `G1_category`, I1-Tool
= `G1_tool`, I2-Cat = `G2_category`, I2-Inst = `G2_instruction`, I3-Inst = `G3_instruction`).
Queries are never pooled across groups (pooling would weight the groups by size). `sopr_by_group`,
`sowr_by_group` and `queries_by_group` carry the per-group numbers; `sowr` is the unweighted mean
of the per-group SoWRs.

**Pinned versions.** StableToolBench at commit `aa4ed9f4737ad98bd706663f01d63623c3427812`
(`STB_REF` in `docker/toolbench.Dockerfile`), which ships the solvable test queries
(`solvable_queries/test_instruction/<group>.json` and `solvable_queries/test_query_ids/`), plus the
HF `stabletoolbench/Cache` `server_cache.zip` response cache at revision
`c6aca0c1cda74d54915eccabb60984dc083fb14c` (sha256
`a3ed72052c2dc961d8e9b71860374d90ded6d475ca6f6449c78fb248cb82a5b2`, `STB_CACHE_REVISION` /
`STB_CACHE_SHA256` in the Dockerfile).

**SoWR reference answers.** The README tables compare every model with ChatGPT-CoT, whose answers
StableToolBench publishes with the other baselines ("The inference results (run in Feb 2024) can
be found on Huggingface"): `stabletoolbench/baselines` at revision
`ea70398f955e7b58f8807187936818e794ce5140`, `data_baselines.zip` (38 MB, Apache-2.0, sha256
`c3c755f68ccd006788d247eb73d644b4c4ea13620ea4702621d921958a80b79c`, checked). On first use gbench
downloads it, converts the reference model's answer trees with the image's
`convert_to_answer_format.py` (as the README converts any model's answers before
`eval_preference.py`; `chatgpt_cot` holds 163 / 153 / 158 / 124 / 106 / 61 answers, one per test
id of each group) and caches the result under
`$GBENCH_CACHE_DIR/toolbench_baselines/ea70398f955e/converted/<reference>/<group>.json`.
A converted file counts, cached or fresh, only when it holds an answer for every test id of
its group (`solvable_queries/test_query_ids/<group>.json` in the image); a conversion that leaves
any test id without an answer is `infra_required` and nothing is cached, and a cached file that
falls short is converted again. `GBENCH_TOOLBENCH_REFERENCE` picks another published baseline
(e.g. `gpt-4-0613_cot`), and `GBENCH_TOOLBENCH_REFERENCE_DIR` points at converted answers of your
own instead. Either way, before the container runs, every test id the run covers must have a
reference answer: a missing or empty group file, or missing ids, is `infra_required` naming the
groups and the directory, because SoWR is a leaderboard column. The result's `reference_source`
names which was used.

**Self-contained by design.** gbench bundles StableToolBench (pinned) + the 236 MB response cache +
the cached `/virtual` server into a **local Docker image built from `docker/toolbench.Dockerfile`**
(built locally, never pulled). The runner `docker run`s it (with `--pull never`, as every image it starts): the container starts the server, runs the
DFSDT loop against your served `/v1`, and converts the answer trees; gbench then scores them with its
**Gemini cascade judge** (`base.judge_generate_cascade`), the same judge every gbench judged suite
uses, NOT the canonical gpt-4-turbo. Only the judge model changes; the rest of ToolEval is kept:

- **Prompts**: the `<description>` texts of `evaluators/tooleval_gpt-3.5-turbo_default/template.txt`,
  verbatim (with the newline after `<description>` and before `</description>`), as the only user
  message.
- **Reply format**: upstream sends the function from `config.yaml` (`check_answer_status`,
  `parse_answer_status` or `select_better_answer`: its name, its description, e.g. "Select the
  better answer with a comprehensive investigation on given aspects. You should ignore the impact
  of the order of candidate answers.", and its parameters) as the one tool and forces the call
  (`tool_choice`). gbench sends the cascade the same function through Gemini function calling
  (mode `ANY`, only that function allowed) and reads the call's arguments, with the `reason`
  parameter added where the scripts ask for it (`return_reason=True`: the pass-rate checks and
  `select_better_answer`). A reply needs what `rtl.py` reads from it: `answer_status`, plus
  `reason` from the reply whose status the pass-rate check returns, and only `index` from
  `select_better_answer` (upstream's own required-key check in `tooleval.py` never runs: it reads
  `required` with `getattr` on a dict). The cascade is `base.judge_generate_cascade`'s: same
  models (`GBENCH_JUDGE_MODELS`), key rotation, rounds and backoff; a model the API answers HTTP
  404 for is not asked again in that call.
- **Sampling**: `config.yaml`'s `completions_kwargs`, which `tooleval.py` `function_call` sends
  with every request: temperature **0.2** and `max_tokens: 1000`. gpt-4-turbo does not think;
  the Gemini judge counts its thinking against `max_output_tokens`, so it is sent
  `max_output_tokens=9192`, the 1000 plus the Gemini thinking allowance of 8192
  (`gemini_budget.GEMINI_THINKING_ALLOWANCE`, the rule wildbench's and simple-evals' graders
  follow too), so its reply keeps upstream's 1000 tokens (`judge_max_output_tokens` on the
  result, `judge_output_limit` with the parts). The grader samples, which is why the scripts
  evaluate three times. A reply that still stops at the limit without the forced call is counted
  in `judge_replies_cut_at_max_tokens_by_group` and handled like any reply without the call: the
  cascade asks its next model; a ToolEval call whose cascade then ends in a judge outage is also
  counted in `judge_outages_after_max_output_tokens_by_group`. Either count is named in
  `leaderboard_comparable_reason`.
- **Retries**: a reply that is not a valid arguments object (a missing field, a status outside the
  enum, an index other than 0 or 1) is asked for again, 3 tries per call as `tooleval.py`
  `function_call` retries. A call that still has no valid verdict, or whose cascade is exhausted,
  is a harness error: it is left out of its pass or comparison and counted
  (`judge_outages_by_group`, `judge_invalid_replies_by_group`, an entry of `harness_errors`, status
  `completed_with_errors`); it is never scored as Unsolved or as a loss. (Upstream's script stops
  with an error instead.)
- **Answering models**: every ToolEval reply is counted by the Gemini model that answered it
  (`judge_models_used`, `judge_models_used_by_group`), and each verdict of `sample_traces` names
  its model.

**Cache misses.** The cached server runs upstream's `server/config.yml` settings at the pinned
commit: a call it has not cached goes first to the ToolBench real-API proxy
(`toolbench_url: http://8.130.32.149:8080/rapidapi`) with the README's empty toolbench key, which
the proxy answers as unauthorized, and then to the API simulator, whose response `is_save: true`
caches for the rest of the run. The simulator is the one other LLM in the harness: upstream
calls `gpt-4-turbo` (temperature 0, `max_tokens` 1024, `response_format` json_object, 3 tries
for valid JSON); gbench's image keeps that request and puts the gbench Gemini judge cascade in
place of the model. One field changes: the Gemini models think, and on the OpenAI-compatible
endpoint their thinking counts against `max_tokens`, where `gpt-4-turbo` does not think, so the
simulator is sent `max_tokens` 9216, upstream's 1024 for the JSON plus the Gemini thinking
allowance of 8192 (`gbench/runners/eval_suites/gemini_budget.py`, copied into the patch;
`simulator.output_limit` records both). A simulator reply that still stops there
(`finish_reason` `length`) with content is used as it arrived, as upstream uses a reply cut at
its `max_tokens`, and counted in `simulator_usage.simulator_replies_cut_at_max_tokens`; one with
no content makes the cascade ask its next model
(`simulator_usage.simulator_empty_replies_at_max_tokens`); a call no model then served is a server
error, also counted in `simulator_usage.simulator_outages_after_max_tokens`. Any of the three is
named in `leaderboard_comparable_reason`, and a sharded run's merge sums them and names them once
over the union. The cascade is the models gbench passes as `GBENCH_JUDGE_MODELS`
(`base.judge_cascade()`), with the judge's rounds and backoff; a model the API answers HTTP 404 for is not asked again in that
call (`docker/toolbench_patch_simulator.py`), on the Gemini API's OpenAI-compatible endpoint
(`https://generativelanguage.googleapis.com/v1beta/openai/`), reached through gbench's Gemini relay
(the entrypoint writes the relay's URL into `config.yml`). That is the only path:
no variable points the simulator or the
grader at another provider, base URL, key or model (the judge cascade's own
`GBENCH_JUDGE_MODELS` / `GBENCH_JUDGE_MODEL` pick the Gemini models of both). So **the only key
you need is a Gemini API key** (`GEMINI_API_KEYS`, else `GEMINI_API_KEY`), no OpenAI anywhere, and
it stays with the gbench process (see [Network and credentials](#network-and-credentials)).
The patched server gives the real-API proxy call a read timeout of 120 s (upstream sends none, so a
proxy that never answers would hang the query and the run); a proxy that answers goes on as
upstream. The result's `simulator` field records the models, the provider
(`Gemini API`), the API base (`simulator.base_url`), that the model is not upstream's
(`matches_upstream_model` is `false`) and the request settings; `gemini_substitutions` lists both
substitutions (the ToolEval grader and the simulator, each with upstream's model, the Gemini
models and the API), and `leaderboard_comparable_reason` names them.
The proxy must
be reachable (checked before the run, `infra_required` otherwise): without it every uncached
call would end in a server error. It is not a model service and gbench keeps it as upstream's
`server/config.yml` sets it (with the README's empty key it answers every call as unauthorized, so
the simulator answers); it receives the tool name and the model's tool arguments of every call the
cache does not hold.
Because the judge differs from the published StableToolBench leaderboard (gpt-4-turbo), a run is
**never `leaderboard_comparable=true`**.

It **hard-errors** (`infra_required`, never skips, never a fabricated number) if Docker, the image
(or one without the labels `org.gbench.toolbench-harness=5` and
`org.gbench.model-request-policy=2`), or a Gemini API key (`GEMINI_API_KEYS`, else
`GEMINI_API_KEY`) is missing or rejected, or a model of the Gemini cascade is not served for the
key, if the metadata server is reachable from docker's bridge network (or the network has no
gateway, or a relay cannot listen on it), if the model endpoint URL carries user info, if the
SoWR reference answers cannot be downloaded, checked or converted or do not cover the run's test
ids, if the container wrote no record of its tool calls (`toolbench_server_calls.jsonl`,
`tool_observations.json`) or of its failed model requests (`model_request_failures.jsonl`), if a
DFSDT step ended on a model request that failed at the endpoint and the image could not tie it to
a query, or if not one model request of the run got a reply, and
records
`status:"error"` (not a 0) if the container/judge produces nothing parseable.

## Selection (`--shard` and `--eval-limit`)

gbench selects the queries on the host, then the container runs exactly those:

1. It reads each run group's test ids from the image (`solvable_queries/test_query_ids/<group>.json`,
   the ids `eval_pass_rate.py` and `eval_preference.py` score, among the queries of
   `test_instruction/<group>.json`).
2. It orders every (group, query id) pair by group, then id, and takes the shard first
   (`--shard I/N`: pair I, I+N, I+2N, ... of that order; the N shards are disjoint and cover the
   whole set).
3. Under `--eval-limit` it draws that many pairs from the shard, stratified by group with the seed
   `toolbench` (`sampling.stratified_sample`): the same pairs on every run, spread over the groups,
   never the first N of each group's file.
4. The selection goes into the container as a JSON file of query ids per group
   (`TB_QUERY_IDS_FILE`); the entrypoint runs exactly those queries of each group, in that order, a
   group with none selected is not run, and SoPR/SoWR score exactly the selected ids. An id the
   image's query file does not hold stops the entrypoint (exit 4).

The result records `selected_query_ids` (`{group: ids}`; `null` on a run with neither flag), the
shard identity (`shard`, `shard_index`, `shard_count`, `shard_applied`) and, for a partial shard,
`sharded run (--shard I/N) ...` in `leaderboard_comparable_reason`. An empty shard (more shards
than queries) runs nothing and reports `accuracy: null` with `accuracy_withheld_reason`.
`gbench --merge` recombines the shards with the suite's own `merge_shards`: each group's SoPR
(per evaluation pass the mean of Solved 1 / Unsure 0.5 / Unsolved 0 over the queries with a
verdict, then the mean over the passes) and SoWR (a query is a win when the candidate takes more of
its votes than the reference) over the union of the shards' per-query records (`sample_traces`),
the headline their unweighted Average. StableToolBench runs every model once and evaluates it
three times, so `--attempt-count` above 1 is not applied (`attempts.not_applied`).

## Network and credentials

The container runs the model's tool calls (the DFSDT loop posts each one to the in-container
cached server, which may pass it on to the real-API proxy and the simulator), so it follows the
rule for containers that run model-controlled actions:

- It runs with **`--network bridge`**, and only after `container_guard.require_metadata_blocked`
  found the cloud metadata server blocked on that network (see
  [toolchains.md](toolchains.md#cloud-vms-the-metadata-server) for the host rule, `sudo iptables
  -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP`). The container reaches the
  real-API proxy by IP address and gbench's relays at the bridge gateway, so no host name has to
  resolve there.
- It holds **no real key**. The simulator's Gemini calls go to gbench's Gemini relay
  (`gemini_relay.GeminiRelay`, a thread of the gbench process on the bridge gateway,
  `POST /v1beta/openai/chat/completions` only, which adds the Gemini key); the container gets the
  relay's per-run placeholder as `GEMINI_API_KEYS` / `GEMINI_API_KEY` and the relay's address, and
  the entrypoint refuses to start with a key that is not a relay placeholder. The model requests go
  to gbench's endpoint relay (`swebench_common.EndpointRelay`, which adds the endpoint's key,
  `VLLM_API_KEY`, else `OPENAI_API_KEY`, when one is set); an endpoint URL with user info is refused
  (`infra_required`), because nothing may carry a credential into the container. Besides the `/v1`
  model routes (chat completions, completions, models), toolbench's relay forwards one route
  outside `/v1`: the server's root `POST /tokenize` (`TOKENIZE_ROOT_PATHS`, passed as the relay's
  `root_paths`, POST only: a GET is refused 405), which the image's request policy asks before
  every model request to fit `max_tokens` to the room the context window leaves (vLLM serves
  `/tokenize` at the server root, not under `/v1`). The relay adds the key to it as to the model requests, never patches its body,
  and never counts it as a model reply or a model-request failure: the relay record lists it under
  `root_paths`, `root_requests_relayed` and `root_path_errors`. (In the 2026-10-10 smoke test, before
  this, the relay refused all 27 `/tokenize` calls, one per model request, `requests_refused: 27`,
  and the policy fell back to sending its budget without knowing the room.) The relay listens with
  a backlog of 128 connections, so at high `--concurrency` (two connections per model request,
  `/tokenize` then the completion) the kernel does not reset connections before the relay sees
  them, which no relay counter would show. The model id the
  container sends (`TB_MODEL`) is the served id gbench's eval runner resolved against `--models`,
  unchanged; the endpoint's model list is not re-read.
- The placeholder and the relays' addresses reach the container by name (`docker run -e NAME`, the
  value taken from the environment gbench passes to docker), never as values on the command line.
  The entrypoint writes the placeholder into the server's `config.yml` and then unsets it, so the
  server, the DFSDT loop that talks to the model under test and the converter run without it in
  their environment. Container output quoted in an error message has every credential value
  replaced by `[REDACTED]`.
- At exit the entrypoint hands everything it wrote to `/out` to the runner's user (`TB_HOST_UID` /
  `TB_HOST_GID`), so the runner can remove the workdir; a path it could not remove is logged.

## Build (one-time, local, never pulled)
```bash
docker build -t gbench-toolbench:aa4ed9f4-v5 -f docker/toolbench.Dockerfile docker
```
This clones pinned StableToolBench, installs its inference + ToolEval env (every pip install
resolved against `docker/toolbench_constraints.txt`, the package versions of the image checked
before that file existed, so a rebuild does not pick up newer releases of the packages upstream
leaves unpinned), checks the installed set against that file (`docker/pinned_pip_check.py`; the
build fails on any other set) and records the Python and Debian sets in `/opt/gbench`, downloads the
236 MB cache (HF `stabletoolbench/Cache@c6aca0c1cda74d54915eccabb60984dc083fb14c`
`server_cache.zip`, checked against sha256
`a3ed72052c2dc961d8e9b71860374d90ded6d475ca6f6449c78fb248cb82a5b2`) into the server, and
bundles the cached `/virtual` server. Override the tag with `GBENCH_TOOLBENCH_IMAGE`. The image
carries the label `org.gbench.toolbench-harness=5`: its cached server records every `/virtual`
call (`docker/toolbench_patch_simulator.py`) and sends the simulator upstream's 1024 plus the
Gemini thinking allowance, counting the replies still cut there (version 5; a version-4 image,
such as the earlier `gbench-toolbench:aa4ed9f4`, sends the bare 1024 and is refused), its DFSDT client tags each call, and the query's
model client, with its test group and query id (`docker/toolbench_patch_client.py`), its model
client records every DFSDT step with a model request that failed at the endpoint or that the
endpoint rejected for its content (HTTP 400/422, not an endpoint failure;
`docker/toolbench_patch_llm.py`), it counts each query's failure observations
(`docker/toolbench_observations.py`), and its entrypoint takes the Gemini cascade from
`GBENCH_JUDGE_MODELS` and only relay placeholders, and it runs exactly the queries gbench selects
(`TB_QUERY_IDS_FILE`, below); gbench refuses an image without it. It also
carries the label `org.gbench.model-request-policy=2`:
its `chat_completion_request` sends every DFSDT request through the run's model request settings,
the request timeout of a `--thinking` run included, and counts failed requests
(`docker/toolbench_request_policy.py`, see [Sampling](#sampling)). An image built from an older
Dockerfile lacks it (or carries `=1`), and the suite hard-errors (`infra_required`) until it is
rebuilt. Rebuild after pulling changes to `docker/toolbench_entrypoint.sh` as well (with the earlier
layers still in Docker's build cache, only the last layers are rebuilt): the entrypoint is what fixes the
simulator to gbench's Gemini relay and removes the placeholder from the environment of the
processes it starts.

## Run
```bash
export GEMINI_API_KEY=...        # or GEMINI_API_KEYS=key1,key2 (it wins when both are set)
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals toolbench --max-output-tokens 65536
```
`--max-output-tokens` is required for every eval run. Without `--thinking` it is not applied
(recorded in `max_output_tokens_note`): upstream's `max_tokens=1024` is sent on every turn. With
`--thinking` it is every turn's budget (see [Sampling](#sampling)).
The runner starts the image on docker's bridge network with gbench's relays (see
[Network and credentials](#network-and-credentials)), reads back the converted answers and the
image's records, and computes SoPR/SoWR with the Gemini judge. The model under test writes the
arguments of RapidAPI tool calls, which the DFSDT loop posts to the in-container cached server
(`SERVICE_URL`, `http://localhost:8080/virtual`); a call the cache does not hold goes to the fixed
ToolBench real-API proxy URL from `server/config.yml` and then to the Gemini simulator. The suite's
other containers (the reference answers' conversion and the listing of the test ids) run with
`--network none`.

## Harness errors

A failure of the harness or of a service is never scored as the model's outcome. Upstream's server
answers HTTP 500 when a call it could not answer raises (the real-API proxy call failing, the
simulator failing; `get_virtual_response` has no handler), and upstream's client
(`rapidapi_multithread.py` `_step`) hands the model the observation
`{"error": "request invalid, data error. status_code=500", "response": ""}` as if it were the
tool's answer; after three simulator replies without valid JSON the server answers
`{"error": "Failed to generate fake response"}` (and caches it). gbench leaves every query that met
such a failure out of SoPR and SoWR. A query is a harness error (kinds in
`harness_error_query_kinds`, per group in `harness_error_queries_by_group`, the count in
`n_harness_error_queries`) when:

- `server_error`: a tool call to the cached server failed on the server's side: the server's
  record of the call (`toolbench_server_calls.jsonl`, with the stage it failed in: `real_api_proxy`,
  `simulator` or `server`), or the failure observation anywhere in the query's whole search tree
  (`tool_observations.json`, which also catches a failure the model met on a branch it later left);
- `simulator_failed`: the simulator gave no valid JSON in three tries (the reply, fresh or replayed
  from the cache);
- `tool_call_timeout`: a tool call timed out;
- `no_answer`: the harness wrote no answer for the query (its DFSDT run raised). Upstream's
  preference script would count such a query as a loss;
- `model_endpoint`: a DFSDT step of the query ended on a model request that failed at the endpoint
  (see [Sampling](#sampling)): upstream's `ChatGPTFunction.parse` asks up to 6 times, and when the
  last try failed it passes the error text into the search tree as the model's reply. The image's
  model client records, per query, every step with a request that failed at the endpoint or was
  rejected for its content (`model_request_failures.jsonl`: the test group and query id, the tries,
  `endpoint_failures` and `content_rejections` (of those, the context-window rejections left to
  upstream are also counted as `over_context_rejections`, present only when there were any), the
  errors by class and HTTP status, and
  whether the step ended on an endpoint failure, `unrecovered`, or on a rejection, `rejected`). A
  step a later try answered left the search unchanged, so its query is scored; those steps are
  counted (`model_request_failures`, `steps_recovered`) and named in
  `leaderboard_comparable_reason`. A request the endpoint rejected for its content (HTTP 400 or
  422) is not an endpoint failure but the model's outcome, as gbench's quality runner and gaia2
  count it (the request carried the model's own conversation; base-run suites, bfcl, mcp_bench and
  the SWE family count a 400/422 as a harness error, see
  [env-knobs.md](env-knobs.md)): a step that ended on one
  (`steps_rejected`) leaves its query scored.

The queries are counted, named in `leaderboard_comparable_reason` and in `harness_errors`, and set
the status to `completed_with_errors`; so do failed calls the server could not tie to a query and
ToolEval calls without a verdict. A group whose every query is a harness error publishes no SoPR,
and the run publishes no score rather than averaging over the other groups: `accuracy` is null
with `accuracy_withheld_reason` (naming the groups whose every query is a harness error apart from
those whose ToolEval calls gave no usable verdict, and saying to re-run), status `failed`, and the
run's records (per-group scores, the Gemini models that answered, the harness errors) are kept.
When the request log counts more endpoint failures (content rejections left out) than the image's
per-step records hold, a failed request may have left its query scored: the run names it
(`model_failure_records_inconsistent` in `harness_errors`, with both counts) and is
`completed_with_errors`. A step
that ended on a model request that failed at the endpoint which the image could not tie to a query
is `infra_required` (the query it belongs to could not be left out). Every failed request is also
counted per process in the request log (`model_replies.failed_requests`, see
[Sampling](#sampling)); the content rejections among them are `model_content_rejections`.

## Records

- `simulator_usage`: how the cached server answered the run's tool calls: `tool_calls`, `cached`
  (from the response cache), `real_api` (the real-API proxy), `simulated` (the Gemini simulator),
  `simulator_failed_replies`, `server_errors`, `tool_input_parse_errors`, and the simulator replies
  at its output limit (`simulator_replies_cut_at_max_tokens`,
  `simulator_empty_replies_at_max_tokens`, `simulator_outages_after_max_tokens`, see
  **Cache misses** above); `tool_calls_by_type`
  holds upstream's log types, `server_errors_by_stage` the failures by stage. The simulated share
  is named in `leaderboard_comparable_reason`.
- `gemini_models_answered`: every Gemini answer by model, the ToolEval replies
  (`judge_models_used`) and the simulator's answers (`simulator_models_used`) together.
- `model_request_failures`: the DFSDT steps with a model request that failed at the endpoint or
  was rejected for its content: `steps_unrecovered` (the step ended on an endpoint failure: the
  query is a `model_endpoint` harness error), `steps_recovered` (a later try got a reply),
  `steps_rejected` (the step ended on a content rejection: the model's outcome, the query scored),
  `failed_requests` (endpoint failures), `content_rejections` (HTTP 400/422), `errors` (by
  exception class and HTTP status) and the steps with an endpoint failure the image could not tie
  to a query. `content_rejections` and `steps_rejected` also include the context-window rejections
  left to upstream (`rejected_over_context` below), which `model_content_rejections` does not
  count.
- `model_content_rejections`: of the request log's `failed_requests`, the requests the endpoint
  rejected for their content (`requests`, `by_error`, for example `BadRequestError 400 x3`): the
  model's outcome, not harness errors, named in `leaderboard_comparable_reason`.
- `sample_traces`: one record per test id the run covers: its group, harness errors, tool calls by
  answer type, the simulator's models and the server-error stages, its failure observations, its
  failed model requests (`model_request_failures`: unrecovered, recovered and rejected steps,
  content rejections, errors), and
  for a scored query its final answer (first 500 characters), whether its last step is a `Finish`
  call, every SoPR pass (status and the ToolEval calls behind it: function, verdict, `reason` up to
  300 characters, the answering model, attempts) and every SoWR comparison (the vote, the order of
  the two answers, their statuses and how the choice was made).
- With a results directory (`GBENCH_RESULTS_DIR`, set by gbench's run manager),
  `toolbench_artifacts/run_<id>/` keeps the image's `model_requests.jsonl`,
  `toolbench_server_calls.jsonl`, `tool_observations.json`, `model_request_failures.jsonl` and the
  converted answers
  (`candidate/<group>.json`), every credential value redacted.

## Knobs

| Env var | Default | Effect |
| --- | --- | --- |
| `GEMINI_API_KEYS` / `GEMINI_API_KEY` | *(one required)* | the gbench Gemini cascade judge **and** the cache-miss simulator, both on the Gemini API; `GEMINI_API_KEYS` (a key list, rotated through by both) wins when both are set. The keys stay with the gbench process: the container gets the Gemini relay's placeholder |
| `VLLM_API_KEY` / `OPENAI_API_KEY` | *(none)* | the model endpoint's key (`VLLM_API_KEY` first; a placeholder such as `EMPTY` counts as unset), added by gbench's endpoint relay (never in the container) |
| `GBENCH_TOOLBENCH_IMAGE` | `gbench-toolbench:aa4ed9f4-v5` | the locally-built image tag |
| `GBENCH_TOOLBENCH_GROUPS` | all 6 | comma/space test groups; a subset makes the headline an Average over those groups only (named in `leaderboard_comparable_reason`) |
| `GBENCH_TOOLBENCH_METHOD` | `DFS_woFilter_w2` | inference method (canonical DFSDT) |
| `GBENCH_TOOLBENCH_REFERENCE` | `chatgpt_cot` | SoWR reference model: a model of the published baselines (`chatgpt_cot`, `chatgpt_dfs`, `gpt-4-0613_cot`, ...), or the directory name under `GBENCH_TOOLBENCH_REFERENCE_DIR` |
| `GBENCH_TOOLBENCH_REFERENCE_DIR` | *(the published baselines, converted and cached)* | host `model_predictions_converted` dir for SoWR, laid out as upstream's `<dir>/<reference model>/<group>.json`. Every test id the run covers needs an answer there: a missing or empty group file, or missing ids, hard-errors (`infra_required`) before the container runs |
| `GBENCH_TOOLBENCH_EVAL_TIMES` | `3` | evaluation passes over the answers (StableToolBench's `--evaluate_times 3`); SoPR is their mean, SoWR counts each query's votes over the passes. A value other than 3 is named in `leaderboard_comparable_reason`; a non-positive or non-integer value hard-errors |
| `GBENCH_TOOLBENCH_MAX_QUERY_COUNT` | `200` | per-task DFS **search budget** (canonical StableToolBench default). This is NOT a task count - `--shard` / `--eval-limit` select the queries (see [Selection](#selection---shard-and---eval-limit)); `--max_query_count` bounds the search within each task. |
| `GBENCH_JUDGE_MODELS` / `GBENCH_JUDGE_MODEL` / `GBENCH_JUDGE_CASCADE_ROUNDS` / `GBENCH_JUDGE_BACKOFF` | gbench's defaults (`base.GEMINI_CASCADE`) | the Gemini judge cascade, which the simulator also uses (the same models, rounds and backoff, on the Gemini API; the container gets the list as `GBENCH_JUDGE_MODELS`) |

## Sampling

StableToolBench's DFSDT loop builds the model-under-test's requests as upstream does
(`toolbench/inference/LLM/chatgpt_function_model.py` `chat_completion_request`: `max_tokens`
1024, `frequency_penalty` 0, `presence_penalty` 0, no temperature). The image sends each request
through gbench's request settings (`docker/toolbench_request_policy.py`, hooked into
`chat_completion_request` by `docker/toolbench_patch_llm.py`), which change only these fields:

- **Thinking switch**: every request carries `chat_template_kwargs.enable_thinking` (in
  `extra_body`) set to the run's mode, `true` with `--thinking`, `false` without.
- **Temperature**: gbench's rule for every suite: 1.0 with `--thinking`, 0.0 without. Upstream
  sends none; gbench always sends one (`TB_TEMPERATURE`, which the image's request settings put on
  every request). `--temperature`, or `GBENCH_TOOLBENCH_TEMPERATURE` (which takes precedence),
  sets another in either mode. The temperature is recorded under `official_sampling` and named in
  `leaderboard_comparable_reason`.
- **max_tokens** (the token-limit policy, `resolve_token_budget` in
  `gbench/runners/eval_suites/base.py`): without `--thinking` upstream's 1024 is sent exactly on
  every turn. With `--thinking` that cap, written for non-reasoning generation, is not applied
  (`official_sampling.max_tokens_applied: false`, "official cap not applied to reasoning" in
  `max_output_tokens_note`): every turn is sent the run's budget (`--max-output-tokens`, else the
  32,768 thinking floor), cut to the room the served context window leaves after the prompt (the
  server's `/tokenize` count; without that endpoint, a rejection that states the prompt's token
  count, or a `max_tokens` larger than the whole window, is read and the request re-sent with
  less, while a rejection that states only the prompt's character count is not refitted and is
  counted as a failed request). A prompt that leaves less than 256 tokens is sent as it is, and
  the server's rejection reaches the DFSDT loop as upstream's error reply.
- **Request timeout**: with `--thinking` every turn carries a timeout sized to the budget as
  gbench's own requests are (`request_timeout_s` in `base.py`: `GBENCH_REQUEST_TIMEOUT_S`, default
  1200, plus the budget divided by `GBENCH_MIN_DECODE_TOK_S`, default 8; 9392 s for a 65,536
  budget) to read the reply, and 5 s to connect as in the openai client's default. Upstream builds
  `OpenAI(base_url=..., api_key=...)` with no timeout, so the openai default (600 s to read the
  reply, 2 retries) would abandon a reply that decodes for longer, which at 25 tokens/s per
  sequence is 15,000 tokens. The value is recorded in `model_request_timeout_s`. Without
  `--thinking` the client's own timeout applies (`model_request_timeout_s: null`).

The image appends the settings it applied and the counts of the model-under-test's requests to
`/out/model_requests.jsonl` (one line per inference process); gbench records them as
`model_request_settings`, `model_replies` (`replies`, `cut_at_max_tokens`: stopped at max_tokens,
`cut_in_reasoning`: stopped there with no answer text and no tool call, `fitted_to_context`,
`refitted_after_rejection`, `failed_requests`, `timed_out`, `rejected_over_context`,
`cancelled`), `model_failed_by_error` and `model_max_tokens_sent`. Settings that differ from the
ones the run asked for are `infra_required`, and so is a missing record in a `--thinking` run;
without `--thinking` a missing record is named in `leaderboard_comparable_reason`. With
`--thinking`, replies cut at the run's budget are named in `leaderboard_comparable_reason`;
without it, a reply cut at upstream's 1024 is upstream's own outcome and is only counted.

A request that ends in an error (after the openai client's own retries, up to 2) reaches upstream's
`chat_completion_request`, which returns it as an error reply; `ChatGPTFunction.parse` asks up to
6 times, 15 s apart, and each failed try is counted:

- `failed_requests`: every error the model client received other than the context rejections
  below, counted by `model_failed_by_error` by exception class and HTTP status (for example
  `APITimeoutError x2`, `InternalServerError 503 x1`; no message text). An error of the served
  endpoint or of the connection (a timeout, a 5xx, a 401/404/429, any HTTP status other than 400
  and 422, a refused connection) is a harness error, never the model's outcome: `timed_out` counts
  the timeouts among them, `harness_errors` carries one entry with the counts, the reason names
  them, and the status is `completed_with_errors` instead of `success`. A query with a step whose
  last try failed that way is left out of SoPR and SoWR (the `model_endpoint` kind in
  [Harness errors](#harness-errors)). A rejection of the request's content (HTTP 400 or 422, among
  them a context-window rejection of the `--thinking` budget that the refits did not resolve) is
  the model's outcome, as gbench's quality runner and gaia2 count it (gbench does not apply one
  rule in every suite yet: base-run suites, bfcl, mcp_bench and the SWE family count a 400/422 as
  a harness error, see [env-knobs.md](env-knobs.md)): it is taken out of the
  harness errors and counted in `model_content_rejections` (named in the reason), and its query
  is scored. When not one request of the run got a reply, every outcome is a harness error,
  content rejections included (no request then carried text the model had generated): the run
  hard-errors (`infra_required`) and publishes no score.
- `rejected_over_context`: the served window rejected prompt + `max_tokens` where upstream's 1024
  was sent (no `--thinking`), or where `/tokenize` reported that the prompt alone leaves less
  than 256 tokens. Upstream's own handling of the error reply applies, as it would upstream on
  this window; the reason names the count.
- `cancelled` stays 0 here (the DFSDT loop's client is synchronous).

The Gemini judge runs at ToolEval's temperature 0.2 with its `max_tokens` 1000 (plus the 8192
thinking allowance) and evaluates 3 times, as upstream's README does with its gpt-4-turbo evaluator.

## Reading the result

`leaderboard_comparable` is **always `false`**: judged by gbench's Gemini cascade, not the canonical
gpt-4-turbo ToolEval judge the published StableToolBench leaderboard uses, and cache misses are
simulated by the Gemini cascade instead of gpt-4-turbo. The aggregation matches
upstream (per-group SoPR averaged over 3 evaluation passes, headline = their unweighted
Average); `leaderboard_comparable_reason` names the Gemini models of the grader and of the
simulator, and also the temperature gbench sends (upstream sends none), replies cut at a `--thinking`
budget, model requests that failed (harness errors; the status is then `completed_with_errors`),
queries left out as harness errors, queries without a preference verdict, the share of tool calls
the simulator answered, requests the context window rejected, a missing model request record, a
group subset
(`GBENCH_TOOLBENCH_GROUPS`), an
`--eval-limit` draw (how many of the queries, drawn stratified by group with a fixed seed), a
partial shard, an evaluation count other than 3, and
ToolEval calls left without a verdict, when present. `evaluate_times`, `sopr_passes_by_group`
(passes that produced a verdict), `sopr_std_by_group`, `judge_temperature`,
`judge_max_output_tokens`, `judge_models`, `judge_outages_by_group`,
`judge_invalid_replies_by_group`, `judge_replies_cut_at_max_tokens_by_group`,
`judge_outages_after_max_output_tokens_by_group`, `judge_output_limit`, `simulator`,
`gemini_substitutions`, `reference_source`, `headline_metrics` (`SoPR`, `SoWR`), the counts
(`count_unit` queries: `total_questions` the test ids covered, `effective_n` those scored,
`correct_answers` their Solved-equivalent, Solved 1 and Unsure 0.5 averaged over the passes),
`sowr_queries_by_group`, the records above, the relays' records (`gemini_relay`,
`model_endpoint_relay`), `docker_network`, the model request record (`thinking`, `official_sampling`,
`max_output_tokens_note`, `model_request_settings`, `model_replies`, `model_failed_by_error`,
`model_max_tokens_sent`, `model_request_timeout_s`, `model_request_failures`,
`model_content_rejections`) and `harness_errors`
are on the result. The number is a
faithful gbench-internal SoPR/SoWR (ToolEval prompts, reply format and protocol kept), consistent
with every other Gemini-judged gbench suite.

A group whose converted answers are missing or empty makes the run hard-error (`infra_required`);
one for which the judge returned no verdict at all, or for which no SoWR could be computed, leaves
the headline without a value (`accuracy` null, `accuracy_withheld_reason`, status `failed`)
instead of averaging over the remaining groups.
