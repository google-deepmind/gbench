# scicode

Canonical SciCode (`SciCode1/SciCode`, test split: 65 main problems, 291 sub-steps of which
288 are scored): research-level scientific computing problems (physics, chemistry, math,
biology, materials) requiring multi-step code. **Scoring: execution-based**. The generated
Python is run against the problem's unit tests in a subprocess; it passes iff the tests pass.

## Dataset

`SciCode1/SciCode`, split `test`, pinned to HF commit
`4510f6a6aa27c43fad7b43da2c59602a86e88480`, plus the reference outputs in `test_data.h5`
from the mirror `Srimadh/Scicode-test-data-h5`, pinned to commit
`72c247d3a8410921b2e848e046d71ed63d9a0ddb` and checked against sha256
`48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890` (see "Test data" below).

## Protocol (sequential, per sub-step)

gbench follows SciCode's canonical protocol rather than a single whole-problem shot. A
problem has N sub-steps; for step *k* the prompt contains every prior step's description
and **the model's own generated code for that step** (separated by `------`), then step
*k*'s description, function header and return line, plus the shared dependency block. The
model writes only step *k*'s function, which is fed forward as context for step *k+1*,
so a problem costs N model calls. Generation runs in parallel **across** problems and
strictly sequentially **within** a problem.

Two prompt modes match upstream: the default *without background* (the model must emit its
own `# Background:` comment) and, with `GBENCH_SCICODE_WITH_BACKGROUND=1`, *with background*
(the human-written step background is injected, canonical `--with-background`).

**Sampling.** gbench's rule for every suite (`base.resolve_temperature`): without `--thinking`,
**temperature 0.0**, which is also SciCode's own default (`eval/scripts/gencode.py`
`--temperature`, default 0, at scicode-bench/SciCode@`e3158ea0`; `OFFICIAL_SAMPLING` in
`scicode.py`); with `--thinking`, **temperature 1.0**, so such a run is not
leaderboard-comparable. `--temperature`, or `GBENCH_SCICODE_TEMPERATURE` for this suite alone
(it takes precedence), overrides either; the result records the official value and
whether the run matched it under `official_sampling`. No `max_tokens` is sent: SciCode sends none
for a `litellm/*` model, the route its `eval/scripts` README gives for vLLM
(`src/scicode/gen/models.py` `generate_litellm_response`; only its Anthropic route sets 4096), so
the server grants each request the rest of the context window, with and without `--thinking`
(gbench's token-limit policy, `base.resolve_token_budget`, records this for a harness that sends no
`max_tokens`; a thinking run is never capped). `--max-output-tokens` is not applied
(`max_output_tokens_note` records it), and the per-request HTTP timeout defaults to one that covers
a reply the length of the served window (base's `UNCAPPED_TIMEOUT_TOKENS`, 262144 tokens, when the
server does not report its window).

Three sub-steps (13.6, 62.1, 76.3) are **supplied** by the benchmark (they depend on prior
human scaffolding): gbench ships them as package data, injects them as context for the later
steps of those problems, and excludes them from scoring, exactly as upstream does.

## Metrics and how to read the result

SciCode reports two numbers and so does gbench:

- **`accuracy` (headline) = main-problem resolve rate**: correct problems / problems scored. A
  problem is correct iff *every* scored sub-step passes. This is the column the SciCode
  leaderboard ranks by. `correct_answers` counts correct problems, `effective_n` the problems
  scored (the headline's denominator) and `total_questions` the problems run.
- **`substep_accuracy`**: correct sub-steps / scored sub-steps (the leaderboard's "Subproblem"
  column). Raw counts: `correct_steps`/`total_steps`, `correct_problems`/`total_problems`
  (`problem_accuracy` repeats the headline).

A full run covers upstream's fixed test split (`test_generated_code.py`: 65 problems, 288
sub-steps). Harness errors are never scored as the model's outcome: a sub-step whose generation
got no reply from the served endpoint after the retries (`failed_generations`), the sub-steps of
its problem after it (built on its code, so they are not generated: `steps_not_generated`), a
sub-step whose test script was never started (the sandbox did not start, the script, the
interpreter or `test_data.h5` is not there inside it, an OS error: `scoring_failed_steps`; a
script that started and runs past its time limit is a failed step, as upstream counts it), and
every sub-step of a problem whose processing crashed (`errored_problems`; its
traces carry `error: "problem_crashed: ..."`). Their traces carry `harness_error` and
`scoring_excluded`; they
are left out of `substep_accuracy`, and their problem (`harness_error_problems`) out of the
headline. Any of them makes `status` `completed_with_errors` and is named in `status_reason`;
with no problem scored, `accuracy` is null with `accuracy_withheld_reason`.

A context-window rejection (HTTP 400/422 naming the window) is not retried: the same prompt is
refused the same way every time. When the sub-step's prompt carries the model's own code of
earlier sub-steps (a runaway reply fed forward, as upstream's `gencode.py` feeds it), it is the
model's outcome: upstream has no handling for it and `test_generated_code.py` scores against
fixed denominators, so that sub-step and the ones after it are failed steps and the problem stays
in the headline (`over_context_steps`, `steps_not_generated_after_over_context`; the step's
trace carries the `over_context` attribution, `base.ENDPOINT_REJECTION_ATTRIBUTION`). A prompt
built from the dataset alone that the window cannot hold is a harness error (kind
`over_context`: serve a larger window).

`leaderboard_comparable` is `True` only for the leaderboard setting: the full test split
(65/288), **without background**, **temperature 0.0** (SciCode's default, and the suite's
without `--thinking`; a `--thinking` run samples at 1.0 and is not comparable), no harness errors
(generations with no reply or crashed problems) and no sub-step reply that reached the end of the served context
window (SciCode sends no `max_tokens`, so the window is the only limit: such replies are counted
in `replies_cut_off_by_budget`, those with no answer text also in `replies_cut_off_in_reasoning`,
those repeating themselves, in their content or, for a reply cut off while looping in its
reasoning, in their reasoning, in `replies_cut_off_looping`; the reason advises a larger
`--max-model-len` only when some of them are not repetition loops), graded against a `test_data.h5` with the pinned digest.
`leaderboard_comparable_reason` names each difference, including `unverified reference file:
test_data.h5 from <source> has sha256 ...` when you supplied a file whose digest differs. The
reference outputs' provenance is recorded under `test_data` (see below).

## Test data (fetched automatically)

SciCode scores generated code against **per-problem reference outputs**, bound as `target`
from the benchmark's `test_data.h5` (`process_hdf5_to_tuple`). Without that file every test
raises `NameError`, so even a perfect solution scores 0 - a structural zero, not a model
result. gbench therefore **hard-errors** (`infra_required`, never skips) if the file can be neither
fetched nor found at `GBENCH_SCICODE_TEST_DATA`.

The file is ~1 GB and is *not* in the HF dataset (`SciCode1/SciCode` ships only
`problems_dev.jsonl` / `problems_test.jsonl`).

**You do not normally need to do anything**: gbench fetches it at eval time like any other
dataset, from the mirror below **at its pinned commit** (`72c247d3a8410921b2e848e046d71ed63d9a0ddb`),
and huggingface_hub caches it. Before generating anything the suite hashes the file:

- The pinned mirror's file must have sha256
  `48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890` (its LFS digest at that
  commit). A different digest there is a **hard error** (`infra_required`): the cached copy is
  corrupt or was replaced; delete it so it is fetched again, or supply a verified copy.
- A file you supply (`GBENCH_SCICODE_TEST_DATA`, or a different mirror through
  `GBENCH_SCICODE_TEST_DATA_REPO`, which is read at its HEAD) is used as given. If its digest
  differs from the pinned one the run is **not leaderboard-comparable**, and
  `leaderboard_comparable_reason` names the file and its digest.

The result records the file's `path`, `source` (`hf:<repo>@<commit>/test_data.h5`, or
`GBENCH_SCICODE_TEST_DATA`), `sha256` and `verified` (digest equals the pinned one) under
`test_data`. The bare `SCICODE_TEST_DATA` / `SCICODE_TEST_DATA_REPO` still work as deprecated
aliases. If neither the local path nor the fetch works, the suite **hard-errors**
(`infra_required`, never skips) rather than reporting a structural 0%.

**Canonical source** - the SciCode repo README links a Google Drive folder, to be saved as
`./eval/data/test_data.h5`:
<https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR>
Google Drive folders are not scriptable without `gdown`/auth, so this is a manual download.

**Scriptable mirror** - a third-party copy on the Hub (1000.7 MiB), pinned as above. The pin
guarantees every run grades against the same bytes; it has not been compared with the Google
Drive original (see below):

```bash
hf download Srimadh/Scicode-test-data-h5 test_data.h5 \
    --repo-type dataset --revision 72c247d3a8410921b2e848e046d71ed63d9a0ddb \
    --local-dir ./scicode-data
```

Then point the suite at it:

```bash
export GBENCH_SCICODE_TEST_DATA=./scicode-data/test_data.h5
```

Doing this manually is only worth it if you want to pin a verified copy - the automatic
fetch pulls the same file.

### Verifying the mirror before publishing a number

The reference outputs in this file decide every pass/fail, so a headline SciCode score is
only as trustworthy as the h5 it was graded against. For a **published** number, verify the
mirror is byte-identical to the canonical Google Drive copy (for a smoke/validation run you
can skip this):

1. Every run records the h5's digest and source on the result JSON, under
   `test_data.sha256` and `test_data.source`. That is the exact file your score used (for the
   default mirror it is always the pinned digest above).
2. Download the canonical file from the Google Drive link above, then hash both and compare:
   ```bash
   sha256sum ./scicode-data/test_data.h5        # the mirror (== result["test_data"]["sha256"])
   sha256sum /path/to/canonical/test_data.h5     # canonical, from Google Drive
   ```
3. If the two digests match, the mirror is identical to canonical - pin your verified copy so
   future runs use it and the provenance is unambiguous:
   ```bash
   export GBENCH_SCICODE_TEST_DATA=/path/to/canonical/test_data.h5
   ```
   If they differ, do **not** quote the number: regrade against the canonical file.

## Requirements
- **`h5py`, `scipy`, `numpy`, `sympy` and `matplotlib`** (`pip install -e .[evals]` installs
  them; numpy is a base dependency). One test problem imports `mpl_toolkits.mplot3d`
  (matplotlib), and the scored code runs with gbench's interpreter. Before generation gbench
  checks that every module the selected problems' `required_dependencies` import can be imported
  there; a missing one is `infra_required` naming it and the package to install, never a failed
  sub-step. Each scored sub-step runs as upstream's `test_generated_code.py` builds it: the
  assembled code, then `from scicode.parse.parse import process_hdf5_to_tuple`, the targets from
  `test_data.h5` and the test cases, written to a script file run with `python <file>` (so a long
  program is no problem; `python -c` would hit the kernel's 128 KiB argument limit). The script
  sits alone in a fresh temporary directory that is bound into the sandbox and removed after the
  step. A small `sh` launcher checks that the interpreter, the script and `test_data.h5` are there
  inside the sandbox, prints a start line, then replaces itself with `python <file>` (`exec`: the
  process, its arguments, `sys.path[0]` and exit status are upstream's). A run without that start
  line never reached the step's code (bubblewrap did not start, or a file is missing inside it):
  it is a harness error (`scoring_failed_steps`, the trace's `harness_error` names the cause),
  never a failed step. Before any generation the suite runs a trivial sub-step that way in the
  sandbox (it binds no test and only opens `test_data.h5`; with `GBENCH_SANDBOX=none` there is no
  jail to check); if it does not pass, the suite **hard-errors**
  (`infra_required`, naming the cause) instead of leaving every sub-step without a verdict. gbench
  **vendors** upstream's h5 loader functions verbatim as `scicode/parse/parse.py` (Apache-2.0,
  header in the file; the module's other functions and its `datasets` import are left out). Each
  sample trace keeps the sub-step's `execution` (return code, stdout and stderr tails) and its
  `prompt` (up to 8000 characters). 90 test cases of the test split (31 scored sub-steps of 20 problems) also import
  SciCode's comparison helpers (`from scicode.compare.cmp import cmp_tuple_or_list` or
  `are_dicts_close`), which upstream's `test_generated_code.py` finds in the installed `scicode`
  package. gbench vendors `src/scicode/__init__.py`, `src/scicode/compare/__init__.py`,
  `src/scicode/compare/cmp.py` and the loader part of `src/scicode/parse/parse.py` from
  SciCode@e3158ea0 in
  `gbench/third_party/_scicode_vendor/` and puts that directory on the `PYTHONPATH` of
  every scored sub-step (`cmp.py` needs `numpy`, `scipy` and `sympy`). The upstream `scicode`
  *package* as a whole pulls an invasively broad, conflicting dependency tree (numpy 2.x,
  litellm, inspect-ai, ...), so only the modules the scoring uses are copied. Without these
  libraries, or when the sandbox cannot import the vendored `scicode.compare.cmp` (checked before
  generation), the suite **hard-errors** (`infra_required`, never skips; it will not report a
  structural 0%). The result records the module's upstream path and sha256 as
  `scicode_compare_module`. Each sub-step's tests are keyed by `step_number` (e.g. `77.1`); a
  problem is correct iff **every** sub-step's tests pass.
- **`bubblewrap` is required.** This suite executes model-written code, so by default
  (`GBENCH_SANDBOX=required`) it runs each step inside a bubblewrap jail (read-only root,
  private `/tmp` and `/run`, an empty home directory with only the interpreter, the Hugging Face
  cache and `GBENCH_SCICODE_TEST_DATA` bound back, no network); the step's script directory is
  bound into it. The jail cannot show a `test_data.h5` under `/tmp` or `/run`, so keep
  `GBENCH_SCICODE_TEST_DATA` elsewhere (the automatic fetch uses the Hugging Face cache, which is
  shown); a file the jail cannot show stops the run before generation (`infra_required`). Install it once (`sudo apt-get install bubblewrap`) and
  ensure unprivileged user namespaces are permitted (Ubuntu 23.10+/24.04 needs one extra
  `sysctl`, see [toolchains.md](toolchains.md#enabling-bubblewrap)). If bubblewrap is
  unavailable the suite **hard-errors** (`infra_required`) rather than executing code
  unsandboxed. To run without isolation on a trusted host, set `GBENCH_SANDBOX=none`.
  `--sandboxes` bounds how many problems run at once.
- **Validation note:** SciCode withholds gold solutions (`ground_truth_code`/
  `general_solution` are empty in the HF dataset), so there is no gold self-check.
  Validate the wiring with a capable model (scores should be non-zero and track
  difficulty; an all-zero run means the h5/package/keying is wrong, not the model).

## Configuration

| Env var | Purpose |
| --- | --- |
| `GBENCH_SCICODE_TEST_DATA` | local `test_data.h5` (else fetched from the pinned mirror); a digest other than the pinned one makes the run non-comparable; with the default sandbox it must not be under `/tmp` or `/run`, which the jail empties |
| `GBENCH_SCICODE_TEST_DATA_REPO` | HF dataset repo of the mirror (default `Srimadh/Scicode-test-data-h5`, read at the pinned commit; another repo is read at its HEAD, and a digest other than the pinned one makes the run non-comparable) |
| `GBENCH_SCICODE_WITH_BACKGROUND` | `1` for the with-background prompt (not leaderboard-comparable) |
| `GBENCH_SCICODE_TEMPERATURE` | per-suite temperature (overrides `--temperature`; default 0.0, also SciCode's, or 1.0 with `--thinking`) |
| `GBENCH_SCICODE_EVAL_TIMEOUT_S` | per-step execution timeout (default 1800, upstream's) |
| `GBENCH_SCICODE_HTTP_TIMEOUT_S` | per-request HTTP timeout in seconds (default: gbench's request timeout for a reply the length of the served context window, as base computes it; `UNCAPPED_TIMEOUT_TOKENS` when the window is unknown) |
| `GBENCH_SCICODE_THREADS` | BLAS/OpenMP threads per scoring subprocess (default 1) |
| `GBENCH_SANDBOX` | `required` (default, bubblewrap) or `none` |

## Run

Leaderboard setting (full test split, without background, greedy):
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals scicode --sandboxes 8 \
       --max-output-tokens 16384
```
`--max-output-tokens` is required on the command line but not applied here: like SciCode, the
suite sends no `max_tokens`.
Add `--thinking` to measure the thinking model (at temperature 1.0, so not leaderboard-comparable), or
`--eval-limit N` for a smoke run (N problems; reported as not leaderboard-comparable). A problem is
one sample: its sub-steps build on each other and are generated in order, once each, so they are
not capped separately. The run logs its plan on its first line and records it as `eval_limit_plan` (with the sub-step count, `sub_step_generations_at_most`).

**Selection (`--eval-limit`, `--shard I/N`).** The problems are ordered by `problem_id`; the shard
is taken first (problem I, I+N, ... of that order), then `--eval-limit` problems are drawn from it
with a fixed seed (`sampling.stratified_sample`, seed `scicode`); every sub-step of a selected
problem runs, since each builds on the previous one. The result records the shard identity
(`shard`, `shard_index`, `shard_count`, `shard_applied`) and, for a partial shard,
`sharded run (--shard I/N) ...` in `leaderboard_comparable_reason`; each sub-step trace carries
`sample_key` (problem and step) and `global_idx` (its problem's position in the ordered list). An
empty shard runs nothing and reports `accuracy: null` with `accuracy_withheld_reason`.
`gbench --merge` re-counts the problem resolve rate and the sub-step accuracy over the union of the
shards' traces (the suite's `merge_shards`). SciCode generates each sub-step once, so
`--attempt-count` above 1 is not applied (`attempts.not_applied`).

**Endpoint key.** gbench's own OpenAI client sends the endpoint key (`VLLM_API_KEY`, else
`OPENAI_API_KEY`; none: `EMPTY`); the scored code runs in the sandbox without any credential.
