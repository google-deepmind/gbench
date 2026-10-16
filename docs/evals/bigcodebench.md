# bigcodebench setup

## What it measures

BigCodeBench: practical Python tasks that call real libraries (pandas, numpy, matplotlib,
requests, sklearn, ...), each checked by a `unittest` TestCase. The leaderboard publishes two
columns, and gbench runs both: **Instruct** (the task as a natural-language instruction) and
**Complete** (a function signature and docstring to complete). The score is **calibrated
pass@1**: the share of tasks whose generated solution passes every test. The headline `accuracy`
is Instruct; Complete sits beside it in `splits`.

## Dataset

- `bigcode/bigcodebench`, split `v0.1.4` (1140 tasks), pinned to HF commit
  `b74c0d0bf70d2c0bc459be537895cca163007f1a`. Every task is generated once per split (2280
  generations).
- BigCodeBench-Hard: `bigcode/bigcodebench-hard`, split `v0.1.4` (148 tasks), pinned to HF commit
  `298d2cc7b96612e15e47313c3603ee124cee0c1f`. Its rows are identical rows of the full set, so a
  full run also reports Hard Instruct and Hard Complete (the full run reads the Hard task ids).
  `GBENCH_BIGCODEBENCH_SUBSET=hard` runs only those 148 tasks.

`v0.1.4` is the split the pinned harness evaluates against (`BIGCODEBENCH_VERSION` in
`bigcodebench/data/bigcodebench.py`).

## Protocol

1. **Prompt** (`bigcodebench/generate.py` defaults and `provider/utility.py::make_raw_chat_prompt`
   at bigcodebench@`09dd993f`). The user turn is the default `instruction_prefix`, a newline, the
   task prompt with surrounding whitespace stripped, and a trailing newline; for Complete the
   task's `complete_prompt` is inside a bare fence:
   ```
   Please provide a self-contained Python script that solves the following problem in a markdown code block:
   <instruct_prompt.strip()>
   ```
   ````
   Please provide a self-contained Python script that solves the following problem in a markdown code block:
   ```
   <complete_prompt.strip()>
   ```
   ````
2. **How the reply starts** follows upstream's two generation paths, recorded as `prefill` and
   `generation_path`:
   - **no-think runs: prefill on**, upstream's default `--backend vllm` path
     (`prefill=not skip_prefill`), which every Gemma row on the leaderboard used
     (`bigcode/bigcodebench-results`, column `prefill`). The assistant turn is prefilled with the
     default `response_prefix` and an open python fence,
     ````
     Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:
     ```python
     ````
     sent as the final assistant message with `continue_final_message: true` and
     `add_generation_prompt: false`, so the model continues it. The one difference from upstream's
     rendered prompt: upstream's ends with that fence plus a newline, and a chat template trims
     the trailing newline from an unfinished assistant turn, so the model generates the newline
     as its first token. When it does, the token sequence is exactly upstream's;
     `prefill_first_char_newline` counts those completions, and any that did not start with the
     newline make the run non-comparable. (For the same reason a stop string that starts with a
     newline, such as `\nprint(`, could also end a completion whose very first line matches it,
     which upstream's cannot.)
   - **`--eval-thinking` runs: no prefill**, upstream's `--backend openai` path, its path for a
     model behind an OpenAI-compatible endpoint (`--base_url`): one user turn, since a prefilled
     assistant turn cannot think first.
3. **Sampling and max_tokens, per path** (`SAMPLING` in `bigcodebench.py`, recorded under
   `official_sampling`; max_tokens resolved by gbench's token-limit policy,
   `base.resolve_token_budget`). The temperature is gbench's rule for every suite
   (`base.resolve_temperature`): **0.0** without `--eval-thinking`, **1.0** with it.
   - no-think, prefill (vllm) path: upstream's `run_codegen` defaults, `temperature=0.0` with
     `n_samples=1` (greedy, the 0.0 gbench sends), plus the backend's own request fields: `top_p=1.0` (the greedy
     `do_sample=False` value of `provider/vllm.py`) and `stop` = `provider/utility.py`'s `EOS` list
     (`<|endoftext|>`, `<|endofmask|>`, `</s>`, `\nif __name__`, `\ndef main(`, `\nprint(`).
     Every request carries `run_codegen`'s `max_new_tokens=1280` (the vllm backend's
     `SamplingParams(max_tokens=...)`); the run's `--max-output-tokens` is not applied. Only a
     prompt that leaves less room in the served context window gets less, which the result names.
   - `--eval-thinking`, openai path: `gen/util/openai_request.py` `make_request` as it runs for a
     served model: `run_codegen`'s `temperature=0.0` (greedy with `n_samples=1`), `top_p=0.95` and
     `max_completion_tokens=1280`, except the temperature: a thinking run samples at **1.0**
     (`base.resolve_temperature`), so it is not comparable; `top_p` 0.95 stays. `make_request`'s reasoning-model branch, for a model name that
     starts or ends with `o1-`, `o3-`, `reasoner` or `grok-3-mini-beta` (it pops `top_p`,
     `max_completion_tokens` and `temperature` and adds `reasoning_effort`), keys on hosted vendor
     model names, so it is not a reasoning-model setting for a served model and is not used; no
     `reasoning_effort` is sent. BigCodeBench declares no model-agnostic reasoning setting, so
     gbench's token-limit policy applies: the 1280 cap was written for non-reasoning generation
     and is not applied; each request is sent the run's budget (`--max-output-tokens`, else
     gbench's thinking floor 32768; less only where the prompt leaves less room in the served
     window), and `official_sampling` records `max_tokens_applied: false`, `max_tokens_note`
     "official cap not applied to reasoning: ..." and `max_tokens_sent`. A reply that stopped at
     that budget (`finish_reason` `length`) is counted in `replies_cut_off_by_budget` (one with no
     answer text, cut off inside the reasoning channel, also in `replies_cut_off_in_reasoning`)
     and makes the run not comparable (raise `--max-output-tokens`).

   `max_output_tokens_note` states the mode's max_tokens choice. `--temperature`, or
   `GBENCH_BIGCODEBENCH_TEMPERATURE` for this suite alone (it takes precedence), overrides the
   temperature in either mode. A decoding penalty (`GBENCH_REPETITION_PENALTY`, `GBENCH_FREQUENCY_PENALTY`,
   `GBENCH_PRESENCE_PENALTY`) reaches every request and makes the run not comparable, as does any
   other request field the path does not send.
4. **Post-processing**, as `generate.py` does it: on the prefill path every tab in the output is
   replaced with four spaces (`provider/vllm.py`), then each response (on the prefill path, the
   continuation after the open fence) goes through the harness's `bigcodebench.sanitize` without
   `--calibrate`, which is generate.py's `sanitize(completion, entry_point)` (it extracts the code
   for the task's entry point). The `--calibrate` rewrite of that script is not part of
   generate.py and is not used.
5. **Scoring**: one harness pass per split, `bigcodebench.evaluate --execution local --split
   <instruct|complete> --no-gt --pass-k 1,`, runs every task's tests inside the harness container.
   `evaluate` calibrates every solution (its default `calibrated=True` prepends the task's
   `code_prompt` and `pass`), as for the leaderboard's `sanitized_calibrated` files. A
   `--eval-limit` run passes the sampled task ids through `--selective_evaluate`. The evaluate
   container sees at most 64 CPUs (`--cpuset-cpus`, the first 64 this process may use; set
   `GBENCH_BIGCODEBENCH_CPUS` for another count) and runs no more test workers than that. The
   harness caps each test process's address space (30 GB, its `reliability_guard`) and TensorFlow
   sizes its thread pools by the CPUs it can see, so on a host where the container sees many more
   CPUs the TensorFlow tasks hang until the 240 s task limit: `BigCodeBench/418`'s reference
   solution passes in about 15 s with 104 visible CPUs and times out with 160 or 208.
   Each split runs in its own temporary work directory (the samples and the harness's result
   files; the harness's Hugging Face cache stays in the container's own `/tmp`), which is removed
   after its results are read.

If the harness produces no parsed results for a split, that split's samples are harness errors,
never scored (see "Containers and network"): the status is `completed_with_errors`, or `error`
when no split was scored (never a clean 0%). A task the harness returned no result for while it
scored others is a harness error too, never the model's outcome: it is left out of `accuracy`,
`splits` and `category_accuracy` (`scoring_excluded`, with `harness_error` kind
`no_harness_result`), marked `scoring_note: no harness result for this task`, counted in
`bigcodebench_unscored`, named in `status_reason`, and the status is `completed_with_errors`.

On a no-think run the server's reasoning parser can move part of a reply into its reasoning field
(vLLM's gemma-4 parser does so when the model opens a thought channel in the middle of the
continuation). That text is the model's output: base's server-parser policy recovers it in
generation order, from the reply's token ids (special tokens kept) or, when the server returned
none, as the reasoning field followed by the content, and that is what is sanitized and scored.
Upstream sanitizes the whole generation of its in-process vLLM (`provider/vllm.py`,
`skip_special_tokens=False`), so the token-id text is upstream's text; the reasoning-then-content
text lacks the special tokens between the two parts, and the run is then not leaderboard-comparable.
gbench does not send `skip_special_tokens=False` itself (the served endpoint's reasoning and
tool-call parsers read the decoded text), so an ordinary reply is the server's default decode with
special tokens stripped: a reply whose text held a special token differs from upstream's there.
Every prefill run's result says so in `special_tokens_note`.
Such replies are counted in `replies_split_by_reasoning_parser` and, as for every suite, in base's
`server_parser_events` / `server_parser_note`.

A task whose request got no reply from the served endpoint (it failed, timed out or did not fit
the served context window after gbench's retries) is a harness error, never the model's outcome:
it is left out of `accuracy`, `splits` and `category_accuracy`, counted in
`request_harness_errors` by kind, named in `status_reason`, and the status is
`completed_with_errors` (`failed`, with `accuracy` null, when no task was scored). When every
Instruct task is one, `accuracy` is `null` with `accuracy_withheld_reason` (never the Complete
figure, never 0).

## Reading the result

- `accuracy`: calibrated pass@1 (%) on Instruct (over the Hard tasks with
  `GBENCH_BIGCODEBENCH_SUBSET=hard`). `bigcodebench_pass_at_1` is the harness's own Instruct
  figure; `bigcodebench_pass_at_1_by_split` has both splits. The harness's figure is over the
  tasks in its results: it counts a task whose request got no reply from the endpoint as failed
  and leaves out a task it returned no result for, so for a split with either it is `null` and
  `bigcodebench_pass_at_1_withheld` says why for that split.
- `splits`: per split (`instruct`, `complete`), `correct` / `total` / `pass_at_1` for the run's
  subset (`full` or `hard`) and, for a full run, for the Hard tasks (`hard`).
  `blended_splits_accuracy` pools both splits and is not a leaderboard number.
- `category_accuracy`: the same per split.
- `subset`, `dataset_revision`, `metric`: what was run.
- `prefill`, `generation_path`, `sanitize`, and on the prefill path `prefill_first_char_newline`
  and `prefill_note`.
- `temperature`, `temperature_source` and `official_sampling`: the path's official temperature and
  request fields, whether the run's temperature matched, and the official `max_tokens` 1280 with
  whether it was applied (`max_tokens_applied: false` and `max_tokens_sent` on the
  `--eval-thinking` path). `max_tokens_sent` is `[1280]` without `--eval-thinking` and the run's
  budget with it (plus any smaller value a prompt's room in the window gave);
  `max_output_tokens_note` states the mode's max_tokens choice, and with `--eval-thinking` that
  BigCodeBench declares no reasoning-model setting for a served model. `replies_cut_off_by_budget`:
  replies that stopped at a limit other than the official 1280 (with `--eval-thinking`, the run's
  budget; without it, a `max_tokens` the window lowered); `replies_cut_off_in_reasoning`: those of
  them with no answer text.
- `bigcodebench_unscored`: tasks the harness returned no result for, harness errors left out of
  the score (present only when non-zero).
- `replies_split_by_reasoning_parser`: no-think replies whose text the server's reasoning parser
  split between its reasoning field and the content, `count` and `recovered_from` (`token_ids` or
  `reasoning_then_content`); present only when non-zero.
- `bigcodebench_harness_errors`: per split whose harness produced no results, `kind` and
  `samples` (present only then). `kind` is `dns` when its output shows a failed host-name lookup,
  `hub_unreachable` when it shows that the Hugging Face Hub could not be reached (the `datasets`
  library's `Couldn't reach '<repo>' on the Hub (...)` or `huggingface_hub`'s
  `LocalEntryNotFoundError`), each with that line as `detail`, else `no_results`. When the split is Instruct, `accuracy` is `null`,
  `correct_answers` and `total_questions` are 0, and `accuracy_withheld_reason` says why.
- `harness_cpus` (how many CPUs the evaluate container ran on) and `harness_parallel` (its test
  workers).
- `leaderboard_comparable` is `True` only for a full run with the path's settings and nothing
  else: **temperature 0.0** (upstream's `run_codegen` is greedy on both paths) and the backend's
  request fields (`top_p` 1.0 and the `EOS` stops without `--eval-thinking`, `top_p` 0.95 with
  it); without `--eval-thinking`, `max_tokens` 1280 on every request; with it, the run's budget
  (the 1280 is not applied to reasoning); in both, no reply cut off by a limit other than the
  official cap. The harness must have scored every task and, on the prefill path, every
  completion must begin with the newline.
  `leaderboard_comparable_reason` names what differs:
  - `harness error: <error>` (a split's harness produced no results; the run has `status:
    error`, or `completed_with_errors` when the other split was scored);
  - `<N> task(s) got no harness result (harness errors, left out of the score;
    bigcodebench_unscored)`;
  - `<N> request(s) got no reply from the served endpoint (harness errors, left out of the score;
    request_harness_errors)`;
  - `<N> repl(ies) are other harness errors (...)`: for example a reply the server's tool-call
    parser deleted that could not be recovered;
  - `<N> no-think repl(ies) the server's reasoning parser split were rebuilt as reasoning then
    content, without the special tokens between them ...`: the server returned no token ids;
  - `subset run (--eval-limit N)`;
  - `temperature=<t> (<source>; the leaderboard is greedy, temperature 0.0)`: every
    `--eval-thinking` run (1.0, source `thinking`), and in either mode when `--temperature` or
    `GBENCH_BIGCODEBENCH_TEMPERATURE` set a value other than 0.0 (the no-think default);
  - `request fields ... differ from upstream's backend defaults (...)`: an extra payload changed
    `top_p` or `stop`, or added another sampling field, or a decoding penalty was set;
  - `reasoning_effort=<v> (make_request sends none for a served model; ...)`: an extra payload
    added `reasoning_effort`;
  - `max_tokens=[...] (run_codegen's max_new_tokens is 1280; ...)`: a no-think request whose
    prompt left less room in the context window was sent less;
  - `<N> repl(ies) stopped at this run's max_tokens (<budget>), <K> of them inside the reasoning
    channel with no answer ...`: with `--eval-thinking`, replies cut off by the run's budget, or
    by less where the prompt left less room in the window (they measure the budget, not the
    model);
  - `<N> repl(ies) stopped at a max_tokens other than run_codegen's 1280 ...`: without
    `--eval-thinking`, replies cut off at a `max_tokens` the context window lowered;
  - `<N> prefilled completion(s) did not start with the newline ...`.

  A comparable run carries `leaderboard_comparable_note`: compare it with leaderboard rows whose
  `prefill` flag matches the run's.

## Containers and network

Both containers are started from the local `gbench-bigcodebench:09dd993f` image with the working directory
mounted at `/app`:

| Container | What runs in it | Network |
| --- | --- | --- |
| sanitize (`bigcodebench.sanitize`) | generate.py's `sanitize` over the replies (it parses them, runs none) | docker's default bridge network |
| evaluate (`bigcodebench.evaluate`) | every task's unittest over the model's code | docker's default bridge network, relies on the host rule |

Both load the v0.1.4 split from the Hugging Face Hub at run time (the cache stays in the
container's own `/tmp`), and bridge is the network upstream's documented `docker run` of its
evaluate image gives it, so neither runs with `--network none`; neither talks to the model
endpoint, and the host network is never used. On a cloud VM the evaluate container relies on the
host rule that keeps a bridge network away from the metadata server at `169.254.169.254` (which
hands out the VM's service-account token):

```bash
sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save   # keep it
```

Before anything is generated (`check_bigcodebench_prerequisites`), gbench's guard
(`require_metadata_blocked`, see [toolchains](toolchains.md#cloud-vms-the-metadata-server)) starts
one short-lived container of the image on the bridge network that opens and closes a TCP connection
to `169.254.169.254:80` (no request is sent); if it opens, the suite hard-errors (`infra_required`)
naming the rule, and a probe that gives no clear answer (neither `python3` nor `bash` in the image,
a docker error or timeout) is a hard error too. gbench never runs sudo or changes the host's
firewall itself.

Both containers load the dataset from the Hugging Face Hub, so they need working DNS on the bridge
network: the same probe container resolves `huggingface.co` (`HARNESS_HOSTS`) before anything is
generated, and a name that does not resolve stops the suite then (`infra_required`), naming the
rule, instead of after every reply was generated. On GCE the VPC DNS resolver is
`169.254.169.254` itself, which is why the rule spares port 53 (a plain `-j DROP` for the address
cuts DNS in every container). If a split's sanitize or evaluate still produces no results, that
split is a harness error: its samples are marked `HARNESS_ERROR` and not scored
(`bigcodebench_harness_errors` gives the split, the kind and the sample count), the split is left
out of `splits`, and the status is `completed_with_errors`, or `error` when no split was scored. When that split is
Instruct, the headline, `accuracy` is `null` (withheld, with `accuracy_withheld_reason`; see
"Reading the result"), never 0. `error` names what the harness printed: a failed host-name lookup
(kind `dns`), or, when the Hub could not be reached (kind `hub_unreachable`), that line. With no
DNS the image prints only the second: `datasets` (2.17.0 in the image) raises
`ConnectionError: Couldn't reach 'bigcode/bigcodebench' on the Hub (ConnectionError)` with
`raise ... from None`, so no resolver text reaches the output. Any other empty pass is kind
`no_results`.

The evaluate container has general internet access on the bridge network, as upstream's
documented `docker run` gives it; only the metadata server is blocked, so the model's code can
reach the internet as it can upstream.

## Prerequisites (else the suite hard-errors `infra_required`)

- **Docker** with a reachable daemon. The sanitize and evaluate containers download the v0.1.4
  split from the Hugging Face Hub at run time, so they need network access.
- **On a cloud VM, the host rule** `sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP`,
  made persistent (see "Containers and network").
- **The local harness image**, built once from gbench's Dockerfile before the first run (gbench
  never pulls a registry image, and an old `bigcodebench/bigcodebench-evaluate` image is not used
  by default; without `gbench-bigcodebench:09dd993f` the suite hard-errors, so build it before a smoke test):
  ```bash
  docker build -t gbench-bigcodebench:09dd993f -f docker/bigcodebench.Dockerfile docker
  ```
  The Dockerfile is upstream's `Docker/Evaluate.Dockerfile` with the source pinned to
  `bigcode-project/bigcodebench@09dd993f46c3fbf3a799465bb96d524edcb0b199`, `requirements-eval.txt`
  read from that commit, and the base image and top-level pip packages pinned. After upstream's pip
  steps (which still resolve their unpinned transitive dependencies), the build enforces the Python
  package set of the image built 2026-09-25 (`docker/bigcodebench.pins.txt`, applied and checked by
  `docker/pinned_pip_check.py --enforce`) and records the Python and Debian packages it installed in
  `/opt/gbench`. The image is large
  (about 25 GB on disk: R, GDAL, TensorFlow, torch). Use `GBENCH_BIGCODEBENCH_IMAGE` if you tag it
  differently.
- **An endpoint that continues a final assistant message** for no-think runs: vLLM's chat
  endpoint does, through `continue_final_message` / `add_generation_prompt`. Before a no-think run
  the suite checks that gbench's request builder forwards both fields and hard-errors if it does
  not.
- `datasets` in the gbench env (base install) to read the prompts. Do not pip-install
  `bigcodebench` into the gbench env (it pins transformers/accelerate and would clobber the vLLM
  env); the image carries the harness.

## Configuration

| Env var | Purpose |
| --- | --- |
| `GBENCH_BIGCODEBENCH_SUBSET` | `full` (default, also reports Hard) or `hard` (the 148 Hard tasks only) |
| `GBENCH_BIGCODEBENCH_IMAGE` | harness image tag (default `gbench-bigcodebench:09dd993f`) |
| `GBENCH_BIGCODEBENCH_TEMPERATURE` | per-suite temperature (overrides `--temperature`; default 0.0, upstream's, or 1.0 with `--eval-thinking`) |
| `GBENCH_BIGCODEBENCH_CPUS` | CPUs the evaluate container may use (positive integer; default 64, see Scoring) |

## Run

```bash
docker build -t gbench-bigcodebench:09dd993f -f docker/bigcodebench.Dockerfile docker   # once
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals bigcodebench \
       --max-output-tokens 4096 --sandboxes 8
```

`--sandboxes` maps to the harness `--parallel` (at most the container's CPUs, 64 by default). Add
`--eval-limit N` for a smoke run (reported as not leaderboard-comparable): N samples for the whole
suite, drawn once by the loader and spread over the two splits (`sampling.allocate_limit`, seeded
`bigcodebench`: `--eval-limit 5` runs 3 in one split and 2 in the other), never N tasks in each
split. Both splits draw from the same seeded task order, so the split with fewer tasks runs a
subset of the other's. The run logs its plan on its first line and records it as
`eval_limit_plan`. `--max-output-tokens`
is required on the command line: it is not applied to a no-think run, whose every request carries
upstream's 1280, and it is the budget of every `--eval-thinking` request (the 1280 is not applied
to reasoning).

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered tasks
(round-robin: positions I, I+N, I+2N, ...), selected before the `--eval-limit` draw, which is then
taken within the shard (N shards run up to N x `--eval-limit` tasks). The result records `shard`,
`shard_index` and `shard_count`, and a shard of more than one is not leaderboard-comparable (the
shard reason is named in `leaderboard_comparable_reason`); `gbench --merge` combines the shards'
results as docs/distributed-gbench.md describes: it re-counts the union of their traces, and
refuses a suite whose headline is not a correct / scored count over them unless the suite defines
its own merge.
