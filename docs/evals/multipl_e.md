# multipl_e setup

## What it measures

MultiPL-E: HumanEval translated into other programming languages. The model completes each
function; the program is compiled/run against the problem's tests. Headline: execution-based
**pass@1** (per problem, the share of its completions whose tests all pass; then the mean over
problems, as upstream's `pass_k.py`), over **23** target languages:
`cpp cs d go java js jl lua php pl r rb rkt rs scala sh swift ts clj dart elixir hs ml`.
The dataset has 24 HumanEval configs; Ada (`adb`) needs a separate upstream Dockerfile and is not in
the default image or language set.

## Dataset

`nuprl/MultiPL-E`, configs `humaneval-<lang>`, split `test`, pinned to HF commit
`28441b6024e71d4a1c1c0f6bf171c935cd5a43f2`.

## Protocol

gbench follows the chat protocol of the pinned MultiPL-E revision
(`nuprl/MultiPL-E@3025a531af7450e7df8b96fe0440e9804480bbad`, `dataset_builder/chat_completions.py`,
which the upstream README names as the way to evaluate chat models):

0. **The upstream script, fetched at run time.** `chat_completions.py` is licensed BSD 3-Clause
   with a Machine Learning Restriction, so gbench does not ship any of it. At run time the suite
   downloads it from
   `https://raw.githubusercontent.com/nuprl/MultiPL-E/3025a531af7450e7df8b96fe0440e9804480bbad/dataset_builder/chat_completions.py`,
   checks its sha256 (`604b7afc237ae416e02c97166759b3a56667ad76bd214f6799fcd1d3cfbef29f`), caches it
   as `$GBENCH_CACHE_DIR/multipl_e/chat_completions-3025a531af74.py` (default
   `~/.cache/gbench/...`; a cached copy is used only while its sha256 matches), and reads three
   things out of it: `stop_at_stop_token`, `postprocess_completion`, and the `Completion`
   signature (its docstring and the `lang` / `prefix` / `completion` field descriptions).
1. **Prompt.** Upstream runs `dspy.ChainOfThought(Completion)` with DSPy 3.0.0b1's ChatAdapter.
   gbench sends the exact messages that adapter produces (DSPy's rendering, whitespace included,
   filled with the fetched signature): a system message describing the input fields `lang` and
   `prefix`, the output fields `reasoning` and `completion`, and the `Completion` instructions
   ("I will give you the prefix of an incomplete program ..."), then a user message:
   ```
   [[ ## lang ## ]]
   <language code, e.g. jl>

   [[ ## prefix ## ]]
   <the dataset prompt>

   Respond with the corresponding output fields, starting with the field `[[ ## reasoning ## ]]`, then `[[ ## completion ## ]]`, and then ending with the marker for `[[ ## completed ## ]]`.
   ```
2. **Parse.** The `completion` field is read exactly like `ChatAdapter.parse` (sections split on
   `[[ ## <field> ## ]]` header lines; both `reasoning` and `completion` must be present). Inline
   `<think>` blocks are stripped first when a server returns them in the content. A reply that
   parse rejects is scored as upstream scores it, with the completion `""`: the script calls
   `completion_cot.aforward`, and in DSPy 3.0.0b1 that async path (`Predict.aforward` ->
   `Adapter.acall`) has no JSONAdapter fallback, because `ChatAdapter` overrides only the
   synchronous `__call__` with one. (The async fallback, `ChatAdapter.acall`, came in
   stanfordnlp/dspy commit `1df67689`, 2025-06-21, after the 3.0.0b1 tag `c3e79ce8`,
   2025-06-11.) The `AdapterParseError` therefore reaches the script's `do_task`, which records
   `completion: ""`. An empty reply (message content `""` or null) is different: DSPy's
   `Adapter._call_postprocess` tests the raw content (`if text:`), parses nothing and returns every
   field as `None`, and the script's `postprocess_completion` then fails on `None` outside
   `do_task`'s error handling, which stops upstream's run. gbench scores an empty reply as an empty
   program and reports the run as not comparable. The test is on the raw content, as DSPy's: a
   reply of whitespace only is not empty there, so `ChatAdapter.parse` rejects it and it counts as
   a parse failure (completion `""`), not as an empty reply.
3. **Post-process** (the fetched `postprocess_completion`, run as is): drop a surrounding Markdown
   fence, keep the first `len(prompt)` characters, and cut the rest at the first dataset stop
   token.
4. **Execute.** Each completion is written with `prompt: ""` (the completion is the whole program,
   as upstream writes it) and run by the official evaluator: `completion + "\n" + tests`. A
   completion passes iff the evaluator reports `OK` with exit code 0.

No other repair is applied (no signature de-duplication, no brace balancing).

**Sampling.** The upstream chat recipe (`OFFICIAL_SAMPLING` in `multipl_e.py`): 20 completions
per problem and no `top_p` (the script sends one only when `--top-p` is given; its default is
None). The temperature is gbench's rule for every suite (`base.resolve_temperature`), not the
recipe's 0.2: **0.0** without `--eval-thinking`, **1.0** with it; such a run is not
leaderboard-comparable. `--temperature`, or `GBENCH_MULTIPL_E_TEMPERATURE` for this suite alone
(it takes precedence), overrides the temperature in either mode; the result records the official
values and whether the run matched them under `official_sampling`. The script sends only `temperature` and `max_tokens` (`dspy.LM(...)`). A
decoding penalty (`GBENCH_REPETITION_PENALTY`, `GBENCH_FREQUENCY_PENALTY`,
`GBENCH_PRESENCE_PENALTY`) or any other sampling field is not upstream's and makes the run not
comparable.

**max_tokens, per mode.** The script has one mode: `dspy.LM(max_tokens=...)` from its
`--max-tokens`, default 2048, which the recipe does not change, and no setting for reasoning models
(DSPy 3.0.0b1's `dspy.LM` only refuses OpenAI o-series model names unless `temperature=1.0` and
`max_tokens >= 20000`, an OpenAI API constraint, not a MultiPL-E setting). 2048 is
`OFFICIAL_SAMPLING.max_tokens`, and gbench's token-limit policy (`base.resolve_token_budget`)
applies it:

| mode | max_tokens sent |
| --- | --- |
| no-think | 2048 (less only for a prompt that leaves less room in the served context window, which the result names); `--max-output-tokens` is not applied |
| `--eval-thinking` | the run's budget (`--max-output-tokens`, else gbench's thinking floor 32768): the 2048 cap was written for non-reasoning generation and is not applied to reasoning |

`max_output_tokens_note` states the mode's choice, and `official_sampling` records `max_tokens`
2048 with `max_tokens_applied` (`false` with `--eval-thinking`, plus `max_tokens_note` "official
cap not applied to reasoning"). A reply that stopped (finish reason `length`) at a limit other
than 2048 (with `--eval-thinking`, the run's budget; without it, a `max_tokens` the context window
lowered) is counted in `replies_cut_off_by_budget`, those with no answer text, cut off inside the
reasoning channel, also in `replies_cut_off_in_reasoning`, and makes the run not comparable. A
no-think reply that stopped at 2048 is the recipe's own outcome.

## Reading the result

- `accuracy`: pass@1 (%) as upstream's `pass_k.py` computes it: for each problem the unbiased
  estimator with k=1, c/n over its scored completions, then the mean over problems. With the
  same number of scored completions per problem this is the mean over completions; when a
  harness error leaves a problem with fewer, it is not `correct_answers` over the scored
  completions, and a problem with no scored completion is left out. The suite draws the upstream chat recipe's 20
  completions per problem (a larger `--attempt-count` is honoured; `GBENCH_MULTIPL_E_SAMPLES`
  sets another count explicitly); `attempts` also carries pass@k and pass^k (its `avg_at_k` is
  the same per-problem mean).
- `category_accuracy`: per language, the same per-problem mean over that language's problems
  (`correct` / `total` are completion counts). `languages_requested`: the configured languages;
  `languages_evaluated`: the languages the run's problems came from (a shard or an
  `--eval-limit` subset can cover fewer). `samples_per_problem`, `dataset_revision`, `metric`:
  what was run. `chat_script`: the URL and sha256 of the fetched upstream script.
- `temperature`, `temperature_source`, `official_sampling`: the temperature used, where it came
  from, and the recipe's. `max_tokens_sent` is 2048 without `--eval-thinking` and the run's
  budget with it; `max_output_tokens_note` states the mode's max_tokens choice.
  `replies_cut_off_by_budget`: replies that stopped at a limit other than 2048;
  `replies_cut_off_in_reasoning` (`--eval-thinking`): those of them with no answer text.
- `chat_adapter_parse_failures`: non-empty responses (whitespace-only ones included) without the
  `[[ ## completion ## ]]` field, scored as empty programs as upstream scores them (see *Parse*);
  they do not affect comparability. `chat_adapter_empty_responses`: empty responses (for example a thinking run that
  spent its budget reasoning), scored as empty programs; upstream's script stops on one, so they
  make the run not comparable. `multipl_e_no_response`: requests that got no response from the
  served endpoint (they failed, timed out or did not fit the served context window after gbench's
  retries). They are harness errors, never the model's outcome: left out of `accuracy`,
  `category_accuracy` and `attempts`, counted in `request_harness_errors` by kind, named in
  `status_reason` (status `completed_with_errors`; `failed` with `accuracy` null when nothing was
  scored) and in `leaderboard_comparable_reason`.
- `multipl_e_unscored`: completions the evaluator returned no result for while it scored others.
  They are harness errors, never the model's outcome: left out of `accuracy`,
  `category_accuracy` and `attempts` (`scoring_excluded`, with `harness_error` kind
  `no_harness_result`), named in `status_reason` (status `completed_with_errors`) and in
  `leaderboard_comparable_reason`. If the evaluator scores nothing at all, every completion is a
  harness error (kind `no_results`), `accuracy` is null and `status` is `error`.
- `leaderboard_comparable` is `True` only for the upstream chat recipe: the full 23-language set,
  no `--eval-limit`, at least 20 completions per problem, temperature 0.2, `max_tokens` 2048
  without `--eval-thinking`, no reply cut off at a limit other than 2048, no
  other sampling field, no empty response or unanswered request, and a clean evaluator run.
  `leaderboard_comparable_reason` lists each difference: a harness error (`status: error`),
  completions without an evaluator result, subset run, language subset, completions per problem,
  temperature, other sampling fields (decoding penalties included), `max_tokens` (a no-think
  request whose prompt left less room in the context window was sent less than 2048), replies cut
  off at a limit other than 2048 (named separately when cut off inside the reasoning channel with
  `--eval-thinking`), empty responses, unanswered requests.

## Containers and network

The suite's only container at run time is the evaluator of the local `gbench-multipl-e:3025a531` image,
which compiles and runs the model's programs with their tests. It runs with
`--network none` (the programs and tests are in the mounted work directory; the evaluator needs
nothing else), so it needs no host firewall rule and gbench's metadata-server guard has nothing to
probe; there is no host name to resolve, and the model's programs have no internet access. The
model's replies are generated by gbench itself, outside any container. The
`docker run --rm gbench-multipl-e-src:3025a531` of the build steps below only writes the checked build
context to its standard output; it runs no model code.

## Prerequisites (else the suite hard-errors `infra_required`)

- `datasets` (a base dependency of gbench) and Hugging Face Hub access to `nuprl/MultiPL-E`
  on first run (cached afterwards).
- **Docker** and the **`gbench-multipl-e:3025a531`** evaluator image, built **locally** (never pulled)
  with MultiPL-E's own `evaluation/Dockerfile` at the pinned commit `3025a531`:
  ```bash
  docker build -t gbench-multipl-e-src:3025a531 -f docker/multipl_e.Dockerfile docker
  docker run --rm gbench-multipl-e-src:3025a531 | docker build -t gbench-multipl-e:3025a531 -
  ```
  The first build downloads MultiPL-E's `evaluation/` directory at that commit
  (`codeload.github.com`), checks every file against `docker/multipl_e.sha256` (and that no other
  file is present), and stops on any difference. The second command builds the evaluator from that
  directory, as upstream's `evaluation/Makefile` does (`docker build -t multipl-e-evaluation .`),
  with three changes to its Dockerfile: its `FROM ubuntu:22.04` is pinned by digest
  (`ubuntu:22.04@sha256:2edbbc5dc405e9612ba3584ce95480277e3eb374407b5505fe26f17df77c7dbc`, the base
  the evaluator image was built on); the `javatuples-1.2.jar` download from `repo.mavenlibs.com` now
  gets an HTML page, which leaves the Java evaluator unable to compile any program, so the jar comes
  from Maven Central (`repo1.maven.org`) and its sha256
  (`2eda5b19d9820e1cc2f69fcd01639a715a673c11f8507e3d1ed593cf765d5e0a`) is checked; and a last step
  records what the build installed in the evaluator image's `/opt/gbench` (`dpkg-versions.txt`,
  `pip-freeze.txt`, `npm-global.txt`, `clojure-tools.txt`). The changed Dockerfile is checked
  against its own sha256. Upstream's toolchain installs (the Ubuntu packages, NodeSource's current
  Node.js, typescript, the D, Mono and Dart repositories, the Clojure CLI, numpy) still resolve when
  the evaluator is built; the record says what a given build got. The build needs network access to
  `codeload.github.com` and to the package and toolchain hosts upstream's recipe downloads from.
  Override the tag with `GBENCH_MULTIPL_E_IMAGE`.
- **Network access to `raw.githubusercontent.com`** on the first run, to fetch MultiPL-E's
  `chat_completions.py` at the pinned commit (13 KB; cached under `$GBENCH_CACHE_DIR/multipl_e/`
  afterwards). Offline, place a copy with the sha256 above at that path. The suite hard-errors
  (`infra_required`) if the file cannot be fetched or does not match.
- No dspy install is needed: gbench carries DSPy's ChatAdapter rendering and fills it with the
  fetched signature.

`multipl_e.py` carries none of MultiPL-E's code or text (BSD 3-Clause License with Machine
Learning Restriction); the DSPy ChatAdapter text and parse port in it carry DSPy's MIT notice.
gbench does not ship MultiPL-E's evaluator recipe either: `docker/multipl_e.Dockerfile` only
downloads and checks it, and `docker/multipl_e.sha256` lists the file names and checksums.

## Configuration

| Env var | Purpose |
| --- | --- |
| `GBENCH_MULTIPL_E_SAMPLES` | completions per problem (positive integer; default 20, the upstream chat recipe); overrides `--attempt-count`, and fewer than 20 is not leaderboard-comparable |
| `GBENCH_MULTIPL_E_TEMPERATURE` | per-suite temperature (overrides `--temperature`; default 0.0, or 1.0 with `--eval-thinking`; the upstream chat recipe uses 0.2) |
| `GBENCH_MULTIPL_E_LANGS` | comma/space list restricting the languages (not leaderboard-comparable) |
| `GBENCH_MULTIPL_E_IMAGE` | evaluator image tag (default `gbench-multipl-e:3025a531`) |

## Run

Run (the upstream chat recipe's 20 completions per problem and `max_tokens` 2048 are the
default; the temperature is gbench's 0.0, the recipe's 0.2 only with `--temperature 0.2`):
```bash
docker build -t gbench-multipl-e-src:3025a531 -f docker/multipl_e.Dockerfile docker   # once
docker run --rm gbench-multipl-e-src:3025a531 | docker build -t gbench-multipl-e:3025a531 -     # once
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals multipl_e --sandboxes 16 \
       --max-output-tokens 4096
```

`--max-output-tokens` is required on the command line but not applied without
`--eval-thinking`: every request carries the script's 2048. With `--eval-thinking` each request
gets `--max-output-tokens` instead (the 2048 cap is not applied to reasoning). The reply carries a `reasoning` field before the program; a reply cut off before
its `completion` field is scored as an empty program, as upstream scores it.

gbench generates the completions against the served model, writes one evaluator input file per
completion into a work dir, runs the evaluator once (`docker run --network none <image> --dir /out
--output-dir /out --recursive`), and reads back the `*.results.json` files. `--sandboxes` bounds
concurrent generation and the evaluator's `--max-workers`. Add `--eval-limit N` for a smoke run
(N problems for the whole suite, stratified across languages, never N per language; the
completions per problem are attempts; reported as not leaderboard-comparable). The run logs its plan on its first line and records it as `eval_limit_plan` (problems
per language, completions per problem, generations).

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered problems
(every requested language, one language after another) (round-robin: positions I, I+N, I+2N, ...),
selected before the `--eval-limit` draw, which is then taken within the shard (N shards run up to
N x `--eval-limit` problems). A problem's completions stay in one shard, and each problem carries
the id `<language>/<name>` across shards. The result records `shard`, `shard_index` and
`shard_count`, and a shard of more than one is not leaderboard-comparable (the shard reason is
named in `leaderboard_comparable_reason`); `gbench --merge` combines the shards by re-counting the
union of their completions and recomputing the per-problem pass@1 over it.
