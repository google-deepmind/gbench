# tau3 setup (tau3-bench `banking_knowledge`)

**What tau3 is.** tau3-bench (upstream's brand is tau^3-bench, "tau three") is the current
generation of the Sierra tau-bench line (`sierra-research/tau2-bench`, the repo now *is*
tau3-bench). Its defining new **text**
domain is **`banking_knowledge`**: a retrieval-augmented (RAG) customer-service domain where
the agent answers banking tasks over a ~700-document knowledge base using retrieval tools
(BM25 + dense embeddings + shell/grep). gbench's `tau3` runs this domain.

> **Why not telecom?** telecom is a **tau2** domain and is already covered by the
> [tau2](tau2.md) suite (airline + retail + telecom). Running it again under "tau3" was a
> mislabel and a duplicate, so tau3 now runs the domain tau3-bench added instead.

The tau-bench line: tau-bench (airline, retail), then tau2-bench (adds telecom and
dual-control), then tau3-bench (adds `banking_knowledge`, voice full-duplex and 75+ task fixes).

## Only the real simulator scores it
banking_knowledge is a multi-turn **environment** benchmark with a retrieval backend; a plain
single-turn `/v1` endpoint cannot measure it (the agent must retrieve over the KB and act on
environment state the model never sees), so gbench always runs the tau2-bench simulator. (The
`_eval_tau3` helper is a legacy telecom-derived single-turn tool-call diagnostic kept only for
tests; it is unrelated to the banking_knowledge scoring path.)

## How gbench runs it
tau3 runs in the same local **`gbench-tau2` image** as tau2, and everything in
[tau2.md](tau2.md) applies: the image built from `docker/tau2.Dockerfile` (tau2-bench commit
`b7ea9074c1cba482b30687fecdb5c8425fd6f619`, package version 1.0.1, whose task data is identical to
the v1.0.1 release tag, installed from its own `uv.lock`), the checks inside the container that
the checkout is exactly that revision and the packages match the lock, the forwarded knobs, the
container that holds no key (gbench's Gemini relay adds it), the protocol (all 97 tasks of
banking_knowledge's `base` split, **4 trials per
task**, tau2's own pass^k), the run directory that is unique per run and never resumed, the agent
sampled by gbench's rule for every suite with no other sampling parameter sent (temperature 0.0,
also upstream's, without `--thinking`; 1.0 with it; override with `--temperature` or
`GBENCH_TAU3_TEMPERATURE`), the chat template's thinking switch on every `--thinking` agent request with
`thinking_applied` / `agent_request_shape` recording it, the agent's request timeout sized from the
served context window in place of LiteLLM's 600 s default (`model_client_timeout`; see
[tau2.md](tau2.md#sampling)), and the Gemini user simulator and nl-assertion judge (gbench's Gemini
cascade, `GBENCH_JUDGE_MODELS`, with the call handling described in
[tau2.md](tau2.md#how-gbench-runs-it-the-local-gbench-tau2-image)) in place of upstream's models.
At the pinned commit the leaderboard ranks banking_knowledge in its own
**tau3-banking** track, so the headline is banking_knowledge's pass^1.

### Retrieval: upstream's `alltools`, dense search on Gemini embeddings
The retrieval tools are part of the benchmark's environment, so gbench runs upstream's
**`alltools`** configuration, the default (`src/tau2/domains/banking_knowledge/retrieval.py`
`DEFAULT_RETRIEVAL_VARIANT`, also what `src/tau2/data_model/simulation.py` sets when no config is
given, and the config 21 of the leaderboard's standard banking_knowledge entries used at the
pinned commit): `KB_search_bm25` (BM25), `KB_search_dense` (dense search) and `shell`. Upstream's
dense search embeds with OpenAI `text-embedding-3-large`. gbench calls every external model
through the Gemini API, so its dense search embeds with **`gemini-embedding-001`** instead; nothing
else changes. This is the only retrieval configuration gbench runs: no knob selects another one,
and nothing falls back to another embedder or configuration.

- **The variant.** gbench registers `alltools-gemini` with tau2: a deep copy of tau2's own
  `alltools` variant (same prompt template, BM25 pipeline, shell tool, `top_k` and every other
  field) whose dense pipeline names `embedder_type` `gemini` and `embedder_model`
  `gemini-embedding-001`, and runs the domain with it, so `results.json` records
  `retrieval_config: alltools-gemini`. Upstream's `alltools` entry is left as it is. Before the
  run gbench checks that tau2's registry holds `alltools` with its dense embedding pipeline (the
  pipeline it substitutes); if not, the suite hard-errors (`infra_required`).
- **The embedder.** gbench's `GeminiEmbedder` (in `tau_common.py`) is added under `gemini` to the
  embedder registries of tau2's document indexer and query encoder. It sends texts to the Gemini
  API's `models/gemini-embedding-001:batchEmbedContents` through gbench's Gemini relay (see
  *Credentials* below), with the relay's placeholder in the `x-goog-api-key` header (the relay
  puts the real key there on the host), each text as one content part with no task type and the
  model's default output
  dimensionality (3072 values), the way tau2's `OpenAIEmbedder` sends only the texts and the
  model. It sends at most 100 texts per request and keeps the vectors in input order. A 429, a 5xx
  or a transport error is retried, up to 4 attempts with exponential backoff; the last failure
  raises, and no other embedder is tried. tau2's cosine retriever, `top_k` and embeddings cache run
  unchanged.
- **The agent's prompt.** tau2 fills the `KB_search_dense` paragraph of `all_tools.md` from the
  variant's embedder (`format_all_tools_dense_instructions`), so the agent reads "The
  `KB_search_dense` tool uses **gemini** with embedding model `gemini-embedding-001` for dense
  retrieval." where upstream's reads "... uses **OpenAI API** with embedding model
  `text-embedding-3-large` ...". The rest of the prompt is upstream's.
  `tau2_report.retrieval_embedder.dense_tool_description` records the line.
- **Document embeddings.** Before the first simulation gbench embeds the 698 documents once,
  through tau2's own `EmbeddingIndexer` (7 requests of up to 100 texts), and tau2 caches them under
  `<gbench cache>/tau2/data/.embeddings_cache`, keyed on the documents, the embedder type (`gemini`)
  and the model; later runs reuse them while those stay the same. (tau2's own pre-run warm-up
  knows only upstream's config names, so it does not cover `alltools-gemini`; without this step
  every concurrent simulation would embed the documents itself.) If this fails, the run stops
  (`infra_required`). Each `KB_search_dense` call then embeds its query in one request.
- **A failed embedding call is a harness error.** A Gemini embedding call that still fails after
  its retries (above) during a simulation is never scored as the agent's outcome. For a
  `KB_search_dense` query, upstream's `Environment.get_response` would turn the failure into an
  `Error: ...` tool result that the agent reads and that counts toward the 10 max errors; gbench
  instead stops that simulation attempt by raising `TauHarnessError` out of the tool call. (A
  document embedding that is not cached fails while tau2 builds the environment, which stops the
  attempt by itself.) tau2's own `run_with_retry` re-runs a stopped attempt as it re-runs any
  failed one (up to 3 more times); a simulation whose every attempt fails is recorded by tau2 as
  an infrastructure error, which `compute_metrics` leaves out of pass^k and which gbench reports
  as a harness error of its own (see [tau2.md](tau2.md#reading-the-result)).
  Each failed call is counted per domain (`tau2_report.embedding_failures_by_domain`, one
  `harness_errors` entry per domain with `failed_embedding_calls` and
  `stopped_simulation_attempts`, and `llm_failures.embedding_error` for the run), makes the
  status `completed_with_errors`, and is named in `leaderboard_comparable_reason` (`<n> Gemini
  embedding call(s) (gemini-embedding-001) failed in banking_knowledge after their retries:
  harness errors, not the agent's outcome ...`).
- **Comparability.** `leaderboard_comparable_reason` names the substitution as `retrieval
  embeddings by gemini-embedding-001 (upstream: text-embedding-3-large)`, so a tau3 result is
  never `leaderboard_comparable`.
- **BM25** (`rank-bm25` 0.2.2) comes from tau2-bench's `knowledge` extra, installed at its
  `uv.lock` version (the extra also installs the OpenAI SDK, `openai` 2.20.0, which gbench does
  not use for retrieval).
- **The shell tool.** `alltools` gives the agent a `shell` tool that runs each command as
  `srt --settings <file> <command>` in a fresh per-simulation sandbox holding the knowledge base.
  `srt` is **Anthropic's sandbox-runtime**, pinned by tau2-bench to **0.0.23**
  (`npm install -g @anthropic-ai/sandbox-runtime@0.0.23`; 0.0.24+ has a Linux regression, see
  `src/tau2/knowledge/sandbox_manager.py`), and needs `rg` (ripgrep), `bwrap` (bubblewrap) and
  `socat` on Linux. The image installs all four (sandbox-runtime with its dependency tree locked
  in `docker/tau2_srt/package-lock.json`). Note that Debian's `python3-pysrt` package also
  installs a program named `srt` (a subtitle tool); it satisfies tau2's PATH check but every
  shell call then fails, which is one reason tau3 no longer runs outside the image.
- **Docker options for the shell sandbox.** bubblewrap creates user, mount, network and PID
  namespaces and mounts its own `/proc`, which Docker's default seccomp and AppArmor profiles and
  its masked `/proc` paths block. A banking_knowledge run therefore starts the container with
  `--security-opt seccomp=unconfined --security-opt apparmor=unconfined --security-opt
  systempaths=unconfined` (still as your uid, with every capability dropped and
  `no-new-privileges`); tau2's retail/airline/telecom runs use Docker's defaults.
  `tau2_report.image.shell_sandbox_docker_opts` records which applied.
- **Shell probe.** tau2 itself only checks that the four binaries are on `PATH`, and raises only
  when `srt` reports missing dependencies; any other sandbox failure reaches the agent as an
  ordinary tool result on every call. Before every tau3 run (`alltools` has the shell tool),
  gbench therefore exports one file into a sandbox through tau2's own `SandboxManager` and runs
  `cat gbench_probe.txt && printenv && cd / && tr '\0' '\n' < proc/1/environ`: `cat` must return
  the file, and neither the command's environment nor that of the sandbox's PID 1 may show any
  of the run's credential values (see *Credentials* below); otherwise the suite hard-errors
  (`infra_required`).
- The sandboxes live in the container's `/tmp` and disappear with the container.

### Credentials and the model under test
No API key or other credential is visible to the model under test, and none is written to the
traces, the result or the log:

- **No key enters the container.** The agent's shell commands run in it as the same user as the
  tau2 driver, so the driver holds no key either: it gets only the per-run placeholder of gbench's
  Gemini relay (a thread of the gbench process on the docker bridge gateway) as
  `GEMINI_API_KEY`, and the relay's address. The user simulator, the judge and the dense-search
  embeddings send their Gemini requests to the relay, which adds the real key (`GEMINI_API_KEYS`,
  else `GEMINI_API_KEY`) on the host and forwards only `generateContent` and
  `batchEmbedContents` for the run's models (see [tau2.md](tau2.md#credentials)); a real
  endpoint key stays with gbench's endpoint relay the same way, through which the agent reaches
  your endpoint on every run, keyless too (since 2026-10-13). No other provider's key is
  passed in. A user simulator or judge of any other provider hard-errors naming it. The
  embedder sends the placeholder in a request header, never in a URL, and its error messages
  never carry it.
- **The shell tool's environment holds no credential.** Upstream's `SandboxManager` runs `srt`
  with the whole process environment, so `printenv` in the agent's shell would print the
  provider keys. gbench runs those commands with every credential-named variable (`*_API_KEY`,
  `*_KEY`, `*_TOKEN`, `*_SECRET`, `*_PASSWORD`, ...) and every variable holding a credential value
  removed, and bubblewrap's own PID namespace and `/proc` keep the driver's environment out of
  view; the shell probe checks both before the run. The rest of the environment is unchanged.
- **Your endpoint key** (`VLLM_API_KEY`, else `OPENAI_API_KEY`) stays with gbench's endpoint
  relay. Without one (unset, or a placeholder such as `EMPTY`) the agent sends `EMPTY`, which
  reaches the container under a private name that the driver removes from the environment before
  tau2 starts; it is added to each agent request rather than stored in the agent's `llm_args`
  (which tau2 writes into every `results.json`). The container has no `VLLM_API_KEY` or
  `OPENAI_API_KEY`.
- **Redaction.** Tool results and the messages sent to the agent have every credential value of
  the run replaced by `[REDACTED]`, `results.json` is written with them replaced, every file of
  the run directory is checked again after each domain, and the result and the streamed
  container output are redacted as well (`tau2_report.credentials_redacted`). Values shorter than
  8 characters (placeholders such as `EMPTY`) are not treated as credentials.
- As upstream warns in `sandbox_manager.py`, the shell sandbox restricts writes and network but
  not reads: the agent can read files elsewhere in the container (the tau2-bench checkout, this
  run's trace directory), just as upstream's sandbox can read the host. No credential is stored
  in any file there.

### Container and network
tau3 starts the same single container as tau2 (see
[tau2.md](tau2.md#container-and-network)): docker's default bridge network with
`--add-host host.docker.internal:host-gateway`, never the host network and never
`--network none`, since gbench's relays for the agent's endpoint and for the Gemini user
simulator, judge and `gemini-embedding-001` embeddings are all reached from it. The agent's shell
tool runs in it, in
sandbox-runtime's bubblewrap sandbox, for which the container lifts Docker's default seccomp and
AppArmor profiles and its masked `/proc` paths (`seccomp=unconfined`, `apparmor=unconfined`,
`systempaths=unconfined`; every capability is still dropped). On a cloud VM the container relies
on the host rule that keeps a bridge network away from the metadata server at `169.254.169.254`:

```bash
sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save   # keep it
```

Before the container starts, gbench's guard (`require_metadata_blocked`) probes the bridge network
with a short-lived container of the tau2 image (a TCP connection to `169.254.169.254:80` opened and
closed, no request sent) and hard-errors (`infra_required`) naming the rule if it opens; a probe
that gives no clear answer is a hard error too, and each of gbench's relays on the bridge gateway
must accept a TCP connection from that network, checked the same way. The agent reaches your
endpoint through gbench's endpoint relay on every run (since 2026-10-13, keyless too; the relay
uses your endpoint URL from this host, so a server bound to `127.0.0.1` only is fine), so the
container neither connects to the endpoint nor resolves its host. A container on the bridge
network still needs working DNS for its general internet access, and on GCE the VPC DNS resolver
is `169.254.169.254`, which is why the rule spares port 53. The Gemini requests (the user simulator,
the judge and the
`gemini-embedding-001` embeddings) go to gbench's Gemini relay at the gateway's address, so the
container resolves no Gemini host (the relay, on this host, resolves
`generativelanguage.googleapis.com`). A lookup or connection that fails later on every attempt at a
simulation is a harness error, as in [tau2.md](tau2.md#container-and-network). gbench never runs
sudo or changes the host's firewall itself.

The container has general internet access on the bridge network, as upstream's harness has on
the machine it runs on; only the metadata server is blocked. The agent's shell commands run in
sandbox-runtime with upstream's settings (`allowedDomains: []`), which give them no network.

### Requirements
- **Docker** and the **`gbench-tau2` image**, as in [tau2.md](tau2.md#requirements):
  ```bash
  docker build -t gbench-tau2:b7ea9074 -f docker/tau2.Dockerfile docker/tau2_srt
  ```
- **On a cloud VM, the host rule** `sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP`,
  made persistent, and an endpoint the docker bridge reaches (see *Container and network*).
- **`GEMINI_API_KEY`** (or `GEMINI_API_KEYS`), on the host, for the Gemini user simulator and
  judge, as in tau2 (checked before the container starts, with every model of the cascade), and
  for the dense search's `gemini-embedding-001` embeddings; gbench's Gemini relay adds it to the
  container's requests. Without it tau3 **hard-errors** (`infra_required`) naming it; gbench
  never falls back to another embedder or retrieval config.
- A served model with tool calling enabled, as in tau2 (the agent sends the domain's tools
  natively; vLLM: `--enable-auto-tool-choice --tool-call-parser <the model's parser>`; a server gbench launches itself (no `--remote-endpoint`) gets both from `--tool-call-parser NAME`, and the CLI refuses a local run of this suite without it).
- If the image, the key or the shell sandbox is missing or broken, or tau2's registry has no
  `alltools` config with a dense embedding pipeline, tau3 **hard-errors** (`infra_required`); it
  never emits a misleading score.

### Removed knobs
- `GBENCH_TAU2_RETRIEVAL_CONFIG` (and its alias `TAU2_RETRIEVAL_CONFIG`) no longer changes
  anything: banking_knowledge always runs upstream's `alltools` with its dense search on
  `gemini-embedding-001`.
- `GBENCH_TAU2_OPENAI_API_KEY`, `GBENCH_TAU2_OPENAI_BASE_URL`, `GBENCH_TAU2_EMBED_MODEL`,
  `GBENCH_TAU2_EMBED_BASE_URL`, `GBENCH_TAU2_EMBED_API_KEY` and `GBENCH_TAU2_EMBED_BATCH` (and the
  `TAU2_EMBED_*` aliases) no longer change anything: the embedder is `gemini-embedding-001`, fixed,
  and no OpenAI service is called.

gbench logs a warning when one of these is set and passes none into the container.

All the `GBENCH_TAU2_*` trials / progress / trace / sampling / image knobs from
[tau2.md](tau2.md) apply here too, and so do `--shard I/N` (shard I of the domain's tasks, before
`--eval-limit N`, which is a seeded draw of N tasks from them, as in tau2.md: tau3 has one
domain, so N tasks in total, each run its `num_trials` trials, the plan logged on one line at
the start and recorded as `eval_limit_plan`; `gbench --merge` recomputes
pass^k over the shards' per-task outcomes) and
`--attempt-count` (the trials, never below 4) (`GBENCH_TAU2_NUM_TRIALS` (default 4), `GBENCH_TAU2_TOP_P`,
`GBENCH_TAU2_TOP_K`, `GBENCH_TAU2_SAVE_TRACES`, `GBENCH_TAU2_PROGRESS_SECS`, `GBENCH_TAU2_IMAGE`,
...); the agent temperature for tau3 is `GBENCH_TAU3_TEMPERATURE`. The bare `TAU2_*` names still
work as deprecated aliases. The run directory is named
`tau3_<model>_<think|nothink>_<config hash>_<UTC time>-<random id>/` and holds
`banking_knowledge/results.json`.

## Comparability caveats
- **Retrieval embeddings.** Upstream's `alltools` embeds with OpenAI `text-embedding-3-large`;
  gbench's embeds with `gemini-embedding-001`, which can retrieve different documents for the same
  query. Every tau3 result names this in `leaderboard_comparable_reason`, so none is
  leaderboard-comparable.
- **Retrieval config.** The leaderboard shows each banking submission's retrieval config as a
  badge. Every tau3 run uses upstream's `alltools` configuration, like most standard entries, so
  compare only with entries that used `alltools`.
- **Harness errors.** A Gemini embedding call that failed after its retries is named in
  `leaderboard_comparable_reason` and makes the status `completed_with_errors` (see *A failed
  embedding call is a harness error* above), and so does a simulation that ended in an
  infrastructure error for any cause, which is never scored: one from a harness failure (an
  outage of the Gemini user simulator or judge, of the model endpoint, failed embedding calls,
  ...) is a harness error, never the agent's outcome, and one raised on the agent's own turn (a
  reply tau2 rejects, a request over the endpoint's context window) is listed apart as not a
  harness failure (see [tau2.md](tau2.md#reading-the-result)). A simulation tau2 ends with
  `USER_ERROR` (the Gemini user simulator broke the message protocol) is a harness error too, left
  out of pass^k (kind `user_simulator_protocol`, see tau2.md).
- **v1.0.1 grading.** tau2-bench v1.0.1 changed `banking_knowledge` grading; results from
  `< 1.0.1` and `>= 1.0.1` are **not comparable**. The pinned revision is past v1.0.1, and
  `tau2_report.tau2_revision` / `tau2_report.tau2_version` record what ran.
- `leaderboard_comparable_reason` lists each difference from a leaderboard submission that
  gbench detects in the run (trials, task subset, user simulator, nl-assertion judge, agent
  sampling, retrieval embeddings, failed Gemini embedding calls, infrastructure errors); see
  [tau2.md](tau2.md#reading-the-result). A default run uses the Gemini user simulator, judge and
  retrieval embeddings, so the flag is `false`.
- banking is the hardest tau3-bench domain; even frontier models score low, so a modest number is
  expected, not a bug.

## Run
```bash
# once: build the image (from the gbench repository root)
docker build -t gbench-tau2:b7ea9074 -f docker/tau2.Dockerfile docker/tau2_srt
# then (Gemini user simulator, judge and alltools retrieval embeddings)
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals tau3 --sandboxes 8 --max-output-tokens 65536
```
`--max-output-tokens` is required for every eval run. As for tau2 (see
[tau2.md](tau2.md#token-budget)), the agent sends no `max_tokens` (as upstream) and gbench's
endpoint relay puts the run's `--max-output-tokens` on every agent request (tau3 is in
`swe_agentic.AGENT_MAX_TOKENS_SUITES`); the Gemini user simulator, judge and embeddings are never
capped. The result records it in `max_tokens_sent`, `max_output_tokens_note`,
`official_sampling.agent_max_tokens_from_run` and `tau2_report.endpoint_relay.max_tokens_from_run`,
and names it in `leaderboard_comparable_reason`. Until 2026-10-13 the relay did not start on a
keyless endpoint: smoke run `smoke-10-12-235114` tapped 133 tau3 agent requests, none with
`max_tokens`.

`accuracy` is the headline: tau2's own **pass^1** for `banking_knowledge` in % (infrastructure
errors excluded, as `compute_metrics` does), the number the tau3-banking leaderboard track ranks
by default; `pass_k` carries pass^1 to pass^4 (up to the trial count), also in
`pass_k_by_domain` / `pass1_by_domain`, and `tau2_report.pass_hat_k` holds them as fractions.
`tau2_report.mean_reward` is the mean reward over every simulation that did not end in an
infrastructure error (as tau2's own `avg_reward`), not a leaderboard number. `correct_answers` /
`total_questions` are the perfect (reward >= 1) and evaluable simulations (the ones that did not
end in an infrastructure error, which pass^1 covers); `category_accuracy` reports the
`banking_knowledge` pass^1 with the same counts, plus `simulations` and `infra_errors`;
`tau2_report` also carries `total_simulations`, `evaluable_simulations`, `infra_errors`, the models
for each role and which of them answered (`gemini_models_answered` at the top level, per role
and domain in `tau2_report`, and the model of each NL-assertion verdict, see
[tau2.md](tau2.md#reading-the-result)), the agent sampling, the
retrieval config (`retrieval_config`, always `alltools`), the variant tau2 ran
(`retrieval_variant`, `alltools-gemini`), its dense embedders (`embedders`, `embed_model`:
`gemini:gemini-embedding-001`), the Gemini embedder's details (`retrieval_embedder`: the
embedder, the API endpoint and the way it is reached (`through`: gbench's Gemini relay), upstream's
embedder, the batch size, the number of documents and the
agent's `KB_search_dense` line), the relays' records (`gemini_relay`, `endpoint_relay`), the
failed Gemini embedding calls per domain
(`embedding_failures_by_domain`), the tau2 revision and version, the image, and the run
directory. `llm_failures.embedding_error` counts Gemini embedding calls that failed after their
retries; they are summed per domain in a `harness_errors` entry (`failed_embedding_calls`,
`stopped_simulation_attempts`), the simulations that ended in an infrastructure error from a
harness failure (this one or any other) have a `harness_errors` entry of their own, those raised
on the agent's own turn an `agent_turn_errors` entry (`infrastructure_error_simulations`,
`error_types`, `example_error` and `sources`; see [tau2.md](tau2.md#reading-the-result)), and any
entry makes the status `completed_with_errors` instead of `success`.
