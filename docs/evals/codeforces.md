# codeforces

Competitive programming from [`open-r1/codeforces`](https://huggingface.co/datasets/open-r1/codeforces),
following the protocol that dataset's card prescribes, the only reproducible open
Codeforces protocol there is.

> **This is not a Codeforces Elo.** Frontier labs report a Codeforces **Elo rating or
> percentile**, obtained by simulating real contests on the live platform. That is not
> reproducible offline. This is a pass-rate over a held-out split, and
> `leaderboard_comparable` is always `false`; `leaderboard_comparable_reason` says so and also
> names the two judging deviations below (CPython time-limit multiplier, memory limit not
> enforced).

## Protocol

| | |
| --- | --- |
| dataset | `open-r1/codeforces`, revision `fbe3f6e903ee854eec2e69e9d96d0306cde59baf` (pinned in `codeforces.py`) |
| config | `verifiable`: `executable` **and** (`official_tests_complete` **or** has `generated_tests`) |
| split | `test`: "problems from late 2024 and early 2025" (the card asks you not to train on these) |
| **N** | **422** |
| prompt | the dataset's own `verifiable-prompts`, Python variant |
| tests | `official_tests` + the separately-downloaded `generated_tests/` shards |
| extraction | open-r1's `extract_code` verbatim ([huggingface/open-r1@1416fa0c](https://github.com/huggingface/open-r1/blob/1416fa0cf21595d2083b399a2a0bbddd7f6e9563/src/open_r1/rewards.py) `src/open_r1/rewards.py`): the **last** `` ```python `` block, closed by the next `` ``` `` wherever it stands |
| judge | the piston `codeforces` package's `run` script the card points to ([guipenedo/piston@7a38da20](https://github.com/guipenedo/piston/blob/7a38da202481bb7a85647a2c6cdc00990ad051bc/packages/codeforces/1.0.0/run) `packages/codeforces/1.0.0/run`), ported for python3 (see [Judging](#judging)) |
| checker | `generated_checker`, where present |
| limits | per-problem `time_limit` x `GBENCH_CODEFORCES_TIME_MULTIPLIER` (default 3, since CPython is far slower than the C++ the limits assume) |

Accepted iff **every** test case passes (open-r1's `score_submission` in `pass_fail` mode). A
reply with no `` ```python `` block is **no submission** and fails, as open-r1 scores it
(`no_submission_reward`); the reply itself is never run as code. A problem with no usable tests is
reported incorrect, never auto-passed.

## Generated test cases: required for a rigorous number

`official_tests` alone covers a **median of 9.7%** of each problem's real test suite
(mean 21.5%; `official_tests_complete` is true for only 21 of 422). The platform truncates
visible tests to ~400 characters, so those cases "can be solved with an easy brute force
solution". The full generated set lives in the same repo but is **not** part of the dataset
load (~110 GB):

```bash
hf download open-r1/codeforces --repo-type dataset \
   --revision fbe3f6e903ee854eec2e69e9d96d0306cde59baf \
   --include 'generated_tests/*.parquet' --max-workers 8 \
   --local-dir $GBENCH_PREREQS_DIR/codeforces
```

Download from the same pinned revision the suite loads the problems from.

`huggingface-cli` is deprecated; the current CLI is `hf`. Then point the suite at it, in
fish:

```fish
set -gx GBENCH_CODEFORCES_GENERATED_TESTS $GBENCH_PREREQS_DIR/codeforces
```

or bash:

```bash
export GBENCH_CODEFORCES_GENERATED_TESTS=$GBENCH_PREREQS_DIR/codeforces
```

The suite reads `<DIR>/generated_tests/test_cases_<contest>.parquet`, and also accepts the
shards sitting directly in `<DIR>`. Only the shards for contests in the split are read, so
a **partial mirror is fine**: you do not need all 1,705 files to run a subset, but it must
cover the contests in your split.

The mirror is **required**: `official_tests` alone (~9.7% coverage) is a near-meaningless
partial, so the suite **hard-errors** (`infra_required`) if `GBENCH_CODEFORCES_GENERATED_TESTS`
is unset or the directory is missing. It also hard-errors when the mirror holds fewer generated
cases for a problem than the row's own `generated_tests` count declares (a missing or partial
shard), naming the affected problems. It never silently runs a reduced-coverage number. (On
2026-09-24 all 422 test problems matched a full mirror exactly.)

The canonical prompts (`verifiable-prompts`) are likewise **required** and load from the same
dataset repo (needs network on first load). A load failure hard-errors rather than silently
substituting a built-in prompt (prompt wording changes the measured quantity).

## Judging

Each test case is judged as the piston `run` script judges a python3 submission, step by step:

- The program is `main.py` in a fresh working directory, run as `python main.py` with the
  script's `ulimit -s unlimited` (the hard limit where that is lower). It fails on a non-zero
  exit or on exceeding the time limit.
- The test input loses a carriage return at the end of each line (`sed 's/\r$//'`) before the
  program reads it. `input_mode` `stdio` feeds it on stdin; `file` (older problems) writes it to
  `input.txt` and reads the answer from `output.txt`, the program's stdout discarded. Every
  `verifiable`/`test` row is `stdio`.
- The program's output loses its empty lines (`sed '/^$/d'`). The expected output is read with
  `$(cat ...)` and written back with `echo`, then loses carriage returns at line ends and empty
  lines.
- **No checker:** both outputs go through the script's `normalize_output` (whole-word `yes`/`no`
  folded to `YES`/`NO` case-insensitively, every run of whitespace one space, trimmed) and must be
  equal (`diff -bq`). So `Yes` matches an expected `YES`, as on the platform for problems that
  accept any case.
- **Checker:** see below.

The script runs on bytes in the C locale, and so does the port (ASCII word boundaries and case
folding, whitespace `[ \t\n\v\f\r]`).

## Checkers are not optional

Some problems accept **multiple valid outputs** and need a checker program to grade (45 of the
422 `verifiable`/`test` problems ship a `generated_checker`). Exact stdout match marks correct
solutions wrong. Where `generated_checker` is present, gbench runs it as the `run` script does:

- argv is `checker.py <input> <reference/jury answer> <submission/model output>`, the three files
  prepared as above;
- the checker **prints its verdict to STDOUT** (`0` = wrong, `1` or `100` = correct). The
  submission passes iff the checker exits 0 within the script's 10 s and its stdout, all whitespace
  removed, is exactly `1` or `100` (open-r1 then requires the run's exit code 0 and stdout `1`).
  Reading only the return code was a bug that once auto-passed every checker-graded problem.

The program and each checker run in the bubblewrap jail; the checker reads the program's output as
a file.

> **Memory limit:** each problem's `memory_limit` is recorded but **not enforced** (reliable
> RLIMIT_AS enforcement on CPython causes false MLEs from interpreter overhead). Like the
> CPython time multiplier, this is a documented deviation from platform judging.

## What changed on 2026-08-20, and why

The suite previously loaded `open-r1/codeforces-cots` **train** and graded against the
statement `examples`. Four separate problems, all measured:

- **`-cots` is a chain-of-thought distillation set with 5 generations per problem.** The
  "500 samples" were **100 distinct problems counted five times each**.
- **A hardcoded 40/30/30 CF/ICPC/IOI quota**, invented here and called "balanced". IOI is
  85 of 10,024 problems (**0.85%**), so the quota over-represented it about 35x. 46% of IOI
  problems ship no `input_format` at all (they are grader-based and cannot run as
  stdin/stdout), which is what produced IOI 17.33% against CF 68% / ICPC 76%.
- **Grading on statement examples** (mean 1.95 cases; 205 of 500 rows had exactly one).
- **Exact stdout match**, which fails the ~30% of problems with multiple valid answers.

The split is also temporal (2024-2025) where `-cots` starts at 2010, so the new number has
far better contamination properties.

## Environment

| var | default | effect |
| --- | --- | --- |
| `GBENCH_CODEFORCES_GENERATED_TESTS` | *(unset)* | directory holding the `generated_tests/` mirror |
| `GBENCH_CODEFORCES_MAX_GENERATED_TESTS` | `0` | cap on generated cases per problem (`0` = no cap, run all, the canonical default). The shards carry a plain `test_i` index, not a hardness order, so a cap keeps arbitrary cases and makes the run non-canonical; the cap is recorded as `max_generated_tests` and named in `leaderboard_comparable_reason` |
| `GBENCH_CODEFORCES_TIME_MULTIPLIER` | `3.0` | multiple of the declared `time_limit` |
| `GBENCH_CODEFORCES_CONFIG` / `_SPLIT` | `verifiable` / `test` | override the subset |
| `GBENCH_CODEFORCES_TEMPERATURE` | *(run default)* | per-suite sampling override; else 0.0 greedy (no-think) / 1.0 (`--thinking`) |
| `GBENCH_SANDBOX` | `required` | must isolate: with `required` (default) the suite **hard-errors** if bubblewrap is unavailable/blocked (it runs model-written code); set `none` to run UNSANDBOXED on purpose |

## Reading the result

- `accuracy`: share of problems where every official and generated case passed.
- `category_accuracy`: per `contest_type` (CF / ICPC / IOI, in their natural proportions).
- `test_coverage_pct`, `tests_per_problem_median`, `tests_per_problem_min`,
  `problems_under_5_tests`: how many cases each verdict rests on.
- `problems_with_generated_tests`, `problems_with_checker`: coverage of the two graders.
- `dataset`: `open-r1/codeforces@<revision>:<config>/<split>`.
- `generated_tests_mirrored`: whether a `generated_tests/` mirror was configured.
- `replies_without_python_block`: scored replies with no `` ```python `` block (no submission,
  scored as failed).
- `extraction`, `judge`: the upstream code the extraction and the judging follow, with commits.
- `max_generated_tests`: the `GBENCH_CODEFORCES_MAX_GENERATED_TESTS` cap in force (`0` = no cap,
  every generated case ran).
- `tests_in_traces` and each trace's `gold_answer`: the tests are not stored in the result (the
  generated tests of the full split are about 20 GB, and the gold is stored in every trace). The
  run keeps them and the checkers in memory and judges from there; each gold names its problem
  (`problem_id`, in the `dataset` revision) with `official_tests_count`,
  `generated_tests_count`, `tests_sha256` (per field: sha256 over the cases in order, each case's
  input, NUL, output, NUL in UTF-8), `has_checker` and `checker_sha256`, plus the time and
  memory limits, `testset_size` and `input_mode`. The coverage fields above are computed from
  those counts.
- `leaderboard_comparable`: always `false` (see the note at the top).
  `leaderboard_comparable_reason` always names the pass-rate-not-Elo difference and the two
  judging deviations, and adds, when they apply: a generated-test cap (`generated tests capped at
  N per problem`), a `GBENCH_CODEFORCES_CONFIG`/`_SPLIT` override, and an `--eval-limit` subset.

## Requirements

- A running OpenAI-compatible `/v1` endpoint.
- The **`generated_tests` mirror** from the pinned revision (see above) and
  `GBENCH_CODEFORCES_GENERATED_TESTS` (or `$GBENCH_PREREQS_DIR/codeforces`) pointing at it. The
  download needs the `hf` CLI (`pip install -U huggingface_hub`).
- **`pyarrow`** to read the shards (installed with `datasets`).
- **bubblewrap** (`sudo apt-get install bubblewrap`, with unprivileged user namespaces allowed),
  or `GBENCH_SANDBOX=none` to run model-written code unsandboxed on purpose.
- Network access to the Hugging Face Hub on first run for the `verifiable` and
  `verifiable-prompts` configs (public; no token needed).

## Run
```bash
export GBENCH_CODEFORCES_GENERATED_TESTS=$GBENCH_PREREQS_DIR/codeforces
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals codeforces --sandboxes 8 \
       --max-output-tokens 32768
```
`--max-output-tokens` is required on the command line; with `--eval-thinking` give the reasoning
room (for example 65536).
Generated code runs in a bubblewrap jail (read-only root, no network, strict timeout);
`--sandboxes` bounds concurrency.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered problems (the
split's order) (round-robin: positions I, I+N, I+2N, ...), selected before the `--eval-limit`
draw, which is then taken within the shard (N shards run up to N x `--eval-limit` problems).
`--eval-limit N` is N problems for the whole suite; The run logs its plan on its first line and records it as `eval_limit_plan`. The
result records `shard`, `shard_index` and `shard_count`, and a shard of more than one is not
leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`). `gbench
--merge` combines the shards' results with the suite's own merge (`merge_shards` in
`codeforces.py`): the generic re-count of the union of their traces (docs/distributed-gbench.md),
with the coverage fields (`test_coverage_pct`, `tests_per_problem_*`, `problems_under_5_tests`,
`problems_with_generated_tests`, `problems_with_checker`, `replies_without_python_block`)
recomputed over the union from the stored test counts.
