# lcb

[LiveCodeBench](https://livecodebench.github.io): contamination-controlled algorithmic coding
problems (LeetCode, AtCoder, Codeforces) tagged with their contest date, so a model can be scored
only on problems published after its training cutoff. gbench follows LiveCodeBench's own runner,
`lcb_runner` ([github.com/LiveCodeBench/LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench)
at commit `28fef95e`), for an OpenAI-compatible chat model (`LMStyle.OpenAIChat`).

## Datasets

| scenario | dataset | pinned revision | default |
| --- | --- | --- | --- |
| **code generation** (headline) | `livecodebench/code_generation_lite`, JSONL shards for the release window | `0fe84c3912ea0c4d4a78037083943e8f0c4dd505` | on |
| code execution | `livecodebench/execution-v2` (test, 479) | `ff6ea0e2a638001006ddcc41259eff23a4283fb2` | on |
| test output prediction | `livecodebench/test_generation` (test, 442) | `6f3ac40bbecf81eba15899139d279b077f2816fd` | on |
| self-repair | the code generation problems and this run's own code generation samples | (as code generation) | on |

All three revisions are pinned in `lcb.py`, and all four scenarios run by default, as
LiveCodeBench publishes each on its leaderboard (`GBENCH_LCB_INCLUDE_TEST_GEN=0` leaves test
output prediction out, `GBENCH_LCB_INCLUDE_SELF_REPAIR=0` leaves self-repair out; the result lists
the scenarios that ran in `lcb_scenarios`). The code-generation window is chosen with:

- **`GBENCH_LCB_VERSION`** (default `release_v6`): a cumulative release (`release_v1` ...
  `release_v6`, `release_latest` = `release_v6`), a single release's new problems (`v5`), or a
  window of new problems (`v4_v6`). The shard mapping is copied from the dataset's own loading
  script (`datasets` 5.x no longer runs dataset scripts, so gbench reads the shards directly).
  `release_v1` (2023-05 to 2024-03) predates every current model and is flagged non-comparable.
- **`GBENCH_LCB_MIN_DATE`** (e.g. `2025-01`): keep only problems whose `contest_date` is on or
  after this value, the precise contamination control (upstream's `--start_date`). Release tags
  are cumulative, so `release_v6` alone still includes 2023 problems.

The result's `contest_date_range` says which problem dates the score actually covers.

## Protocol

**Code generation** (the headline, `accuracy`):

- *Prompt:* upstream's system message ("You are an expert Python programmer. You will be given a
  question (problem specification) and will generate a correct Python program that matches the
  specification and passes all tests.") and user prompt, verbatim: `### Question:` + the problem,
  then `### Format:` with the starter code (call-based problems) or the stdin/stdout instruction
  and a `# YOUR CODE HERE` block, then `### Answer: (use the provided format with backticks)`.
- *Extraction:* the **last** fenced code block (`extract_code`); no fenced block means an empty
  program.
- *Grading:* upstream's own grader, `lcb_runner/evaluation/testing_util.py`, vendored verbatim
  into `lcb.py` and run in the bubblewrap jail: all public and private tests, 6 s per test, a
  global limit of `(6 + 1) x tests + 5` s. Stdin problems compare stdout line by line after
  stripping, with a `Decimal` fallback for numeric tokens. Call-based problems call the function
  named by the problem's `metadata.func_name` (on `Solution()` when the code defines it) with
  one JSON argument per input line, and accept a tuple where a list is expected. A problem passes
  only if every test passes.
- *Verdict isolation:* the process that runs the model's program never holds the expected
  outputs, so it cannot produce a pass it did not earn. gbench sends the grader two
  length-prefixed parts and then a per-call nonce: the part the program may see (its code, the
  test inputs, the entry point) and the expected outputs. The grader process reads only the first
  part, keeps a private copy of stdout, marks itself non-dumpable (so no other process of this
  user can read its memory, open its file descriptors through `/proc/<pid>`, or ptrace it) and
  forks. The child re-points stdin at `/dev/null`, closes that stdout copy, waits until the grader
  has read the expected outputs, and runs the program through upstream's own helpers
  (`compile_code`, `get_function`, `call_method`, `Capturing`, `reliability_guard`, the same
  per-test alarms and error codes); instead of comparing, it sends what the program produced (the
  captured stdout, or the return value in a typed encoding that keeps Python and numpy data types)
  to the grader. The grader, which never runs model code, applies upstream's comparison
  (`grade_stdio`'s line-wise check with the `Decimal` fallback, `grade_call_based`'s
  `prediction == gt_out`), builds upstream's per-test result list, reads the nonce once the child
  is gone and writes the verdict tagged with it; gbench ignores any line without its nonce. So a
  program that writes a fake pass to every file descriptor, steals the nonce through `/proc`,
  imitates the child's messages, or reads the expected output from its own memory scores a fail
  unless the output it reports is the right one (regression tests
  `test_lcb_verdict_cannot_be_forged_by_fd_write_and_exit`,
  `test_lcb_verdict_cannot_be_forged_through_proc`,
  `test_lcb_runner_accepts_only_a_verdict_carrying_its_nonce`,
  `test_lcb_program_cannot_read_the_expected_output_from_memory` and
  `test_lcb_imitating_the_childs_messages_reports_only_an_output`; the differential test
  `test_lcb_verdict_matches_upstream_run_test` holds the verdict equal to upstream's `run_test` on
  honest programs). The one difference from
  upstream: a returned object that is not plain Python or numpy data (an instance of a class with
  its own `__eq__`, say) never equals the expected output, where upstream would call its
  `__eq__`.

**Code execution** (reported under `scenarios.code_execution`): upstream's system message and
direct-output prompt (two examples, then `assert <call> == ??` and an open `[ANSWER]`),
`extract_execution_code`, and scoring by executing `assert <gold output> == <prediction>` with
upstream's `BASE_IMPORTS` (3 s, `utils_execute.unsafe_execute`). A prediction that repeats the
call is not executed and scores 0, as upstream. The gold output stays with the grader: the
program runs `assert <stand-in> == <prediction>`, and the stand-in's `__eq__` asks the grader,
which compares with the gold value; the sample passes only if exactly one such comparison
happened, it was true, and the program then finished. The chain-of-thought variant
(`--cot_code_execution`) is not used; upstream's default is the direct prompt.

**Test output prediction** (reported under `scenarios.test_output_prediction`; on by default,
`GBENCH_LCB_INCLUDE_TEST_GEN=0` or the legacy `LCB_INCLUDE_TEST_GEN=0` leaves it out): upstream's
system message and prompt for the first test case, `extract_test_output_code`, and
`check_testcase_output` (the asserted value is evaluated in the jail, sent to the grader, and
compared there with the gold output).

**Self-repair** (reported under `scenarios.self_repair` and `self_repair`; on by default,
`GBENCH_LCB_INCLUDE_SELF_REPAIR=0` leaves it out) follows `runner/base_runner.py`
`run_main_repair` (`--scenario selfrepair`, which requires `--n 1`) on this run's code generation
samples, each with its grade and the grader's failure metadata:

- An accepted program is kept as it is (`format_prompt_self_repair` returns no prompt for it).
- Every other program gets `format_prompt_self_repair`'s `LMStyle.OpenAIChat` prompt, verbatim
  from `prompts/self_repair.py`: its system message, then `### Question:` + the problem,
  `### Answer:` + the extracted program, the failure message of `get_check_prompt`, the stdin
  `### Format:` block, `### Answer: (use the provided format with backticks)` and the closing
  "First reason about the code ..." instruction. It gets one new generation with the code
  generation requests' sampling and `max_tokens` (2000 without `--eval-thinking`, the run's
  budget with it; see *Sampling*).
- All programs, kept and repaired, are graded again with the code generation grader, and the
  scenario's pass@1 is the mean over problems of c_i/n_i (c/10 when every repair is scored).

The failure message comes from the metadata the grader reports for the program, which is
upstream's `run_test` metadata field for field (the grader returns it with the verdict;
`get_check_prompt` shows the failing test's input, the program's output and the expected output,
truncated to 300 characters by upstream's `truncatefn`, or the error). Two metadata shapes of the
pinned `run_test` have no message in `get_check_prompt`, which was written for the grader
LiveCodeBench had when self-repair was added (commit `8eac093f`): a program that fails before its
first test (it does not compile, or has no entry point) comes back as error code -4 without the
test fields, and a grading run that returned nothing comes back as -5. The earlier grader reported
the first as a compilation error (error code -1 with the error text) and the second with no
metadata, so gbench gives them the compilation-error message and no failure message respectively.
The result counts the messages sent by kind in `self_repair.failure_messages`. Before generation,
the grader self-check also requires the failure metadata of a known wrong answer.

**Sampling.** `lcb_runner`'s (`OFFICIAL_SAMPLING` in `lcb.py`): top_p 0.95 and `frequency_penalty` / `presence_penalty` 0 on
every request (its temperature 0.2 is recorded, not sent; see below), as `runner/oai_runner.py` (and `runner/vllm_runner.py`) send them, self-repair
included, and n=10 samples per problem in every scenario except self-repair (a larger
`--attempt-count` is honoured), so the headline `accuracy` is the mean over problems of c/10,
upstream's pass@1 estimator (`evaluation/pass_k_utils.py` `compute_metrics_from_results`, k=1).
Every scenario's accuracy is computed that way, per problem over its scored generations: the mean
of c_i/n_i, where n_i is the number of problem i's generations that were scored. When a harness
error leaves a problem with fewer scored generations, this is not the pooled correct / total
(`correct_answers` / `effective_n`), and a problem with no scored generation is left out.

**Temperature: 1.0 with `--eval-thinking`, 0.0 without.** gbench samples every suite this way
(`base.resolve_temperature`), not at lcb_runner's 0.2 (at 0.2 Gemma's reasoning looped until the
token budget ended in 40 of 50 replies of a 2026-10-06 run), self-repair included; top_p 0.95 and
the penalties stay lcb_runner's. Such a run is not leaderboard-comparable (its reason names the
temperature and its source, `thinking` or `default`). `--temperature`, or `GBENCH_LCB_TEMPERATURE`
for this suite alone (it takes precedence), overrides the temperature in either mode; the result
records the official values and whether the run matched them under `official_sampling`. A decoding penalty set through
`GBENCH_REPETITION_PENALTY`, `GBENCH_FREQUENCY_PENALTY` or `GBENCH_PRESENCE_PENALTY` reaches every
request, self-repair included, and is named there too unless it is the 0 lcb_runner sends.

**max_tokens, per mode.** lcb_runner's `max_tokens` 2000 (`parser.py --max_tokens` default, sent by
`runner/oai_runner.py`) is `OFFICIAL_SAMPLING.max_tokens`, and gbench's token-limit policy
(`base.resolve_token_budget`) applies it to every request, self-repair included:

| mode | max_tokens sent | why |
| --- | --- | --- |
| no-think | 2000 (less only for a prompt that leaves less room in the served context window, which the result names); `--max-output-tokens` (required on the command line) is not applied | the official cap of the `LMStyle.OpenAIChat` runner gbench follows |
| `--eval-thinking` | the run's budget: `--max-output-tokens`, else gbench's thinking floor (32768) | the 2000 cap was written for non-reasoning generation, so it is not applied to reasoning. lcb_runner has no reasoning-model setting for the OpenAIChat style: its reasoning styles are vendor-API variants with their own prompts (`OpenAIReason`: no max_tokens, temperature or top_p; `OpenAIReasonPreview`: `max_completion_tokens` 25000; `Claude3Thinking`: `max_tokens` 32000 with a 24000 thinking budget; `GeminiThinking`: no `max_output_tokens`), and its open reasoning models (`LMStyle.DeepSeekR1`, `LMStyle.QwQ`) run through `runner/vllm_runner.py` with `--max_tokens` given on the command line |

`max_output_tokens_note` states the mode's choice, and `official_sampling` records `max_tokens`
2000 with `max_tokens_applied` (`false` with `--eval-thinking`, plus `max_tokens_note` "official
cap not applied to reasoning" and `max_tokens_sent`). A reply, code generation or self-repair,
that stopped (finish reason `length`) at a limit other than lcb_runner's 2000 (with
`--eval-thinking`, the run's budget; without it, a `max_tokens` the context window lowered) is
counted in `replies_cut_off_by_budget` (`self_repair.cut_off_by_budget` for self-repair), those
with no answer text, cut off inside the reasoning channel, also in `replies_cut_off_in_reasoning`
(`self_repair.cut_off_in_reasoning`), and makes the run not comparable: its score measures the
limit, not the model. A no-think reply that stopped at 2000 is lcb_runner's own outcome.

## Reading the result

| field | meaning |
| --- | --- |
| `accuracy`, `correct_answers`, `total_questions`, `effective_n` | code generation pass@1, lcb_runner's estimate: the mean over problems of c_i/n_i over each problem's scored samples (10 or more per problem); `correct_answers` counts the correct scored samples, `total_questions` every code generation sample, `effective_n` the scored ones (with a harness error on some problems, `accuracy` is not `correct_answers` / `effective_n`). A request that got no reply from the served endpoint is a harness error: left out of every number here and in `scenarios` and `attempts`, counted and named in `status_reason`; when no code generation sample is scored, `accuracy` is null with `accuracy_withheld_reason` |
| `attempts` | avg@k (the per-problem mean, as `accuracy`) / pass@k / pass^k over code generation only (`attempts_all_scenarios` keeps base's block pooled over every scenario) |
| `scenarios` | per-scenario correct / total / accuracy (`self_repair` included), over the scored samples; accuracy is the per-problem mean of c_i/n_i |
| `self_repair` | the self-repair scenario: `correct` / `total` / `accuracy` (the per-problem mean), `at_k`, `kept_accepted_programs`, `repair_requests`, `failed_requests`, `harness_errors` (repair requests that got no reply: left out of `correct` / `total` / `accuracy` and named in `status_reason`, never scored as a failed program), `truncated_responses`, `cut_off_by_budget`, `cut_off_in_reasoning`, `failure_messages` (by kind), `max_output_tokens` and the prompt source; each scored code generation trace carries `lcb_self_repair` (the repair reply with its `max_tokens_sent`, or `kept`) and `lcb_grading` (the failure metadata) |
| `temperature`, `temperature_source`, `official_sampling` | the temperature used, where it came from, and lcb_runner's values with `temperature_matches_official` |
| `max_tokens_sent`, `max_output_tokens_note` | the budgets the requests carried (2000 without `--eval-thinking`, the run's budget with it) and the mode's max_tokens choice |
| `replies_cut_off_by_budget`, `replies_cut_off_in_reasoning` | replies (generation and self-repair) that stopped at a limit other than lcb_runner's 2000, and (with `--eval-thinking`) those of them cut off inside the reasoning channel with no answer |
| `blended_all_scenarios_accuracy` | all scenarios pooled; not a LiveCodeBench number |
| `category_accuracy` | per `code_gen_<difficulty>`, `code_execution`, `test_gen_<difficulty>`: `correct` / `total` over the scored samples, `accuracy` the per-problem mean of c_i/n_i as in `scenarios` |
| `lcb_version`, `lcb_min_contest_date`, `contest_date_range` | the problem window scored |
| `lcb_scenarios`, `lcb_scenarios_not_run` | the scenarios that ran (those with a trace, and self-repair when it ran), and for each of the other four why it did not: `left out with GBENCH_LCB_INCLUDE_TEST_GEN=0` / `..._SELF_REPAIR=0`, `dropped by --eval-limit N` (the N samples are spread over the scenarios, so with N below their number one gets none), `no code generation sample was scored` (self-repair), or no sample loaded / on the shard |
| `dataset`, `lcb_runner_commit` | pinned dataset revision and the upstream commit the protocol was ported from |
| `tests_in_traces`, each code generation trace's `gold_answer` | the tests are not stored in the result: a full release holds about 9.4 GB of public and private tests, and the gold is stored once per sample (10 per problem). The run keeps them in memory and grades from there; each code generation gold names its problem (`question_id`, in the `dataset` revision), `fn_name`, `question_content`, `contest_date`, `test_counts` (public / private) and `tests_sha256` (the sha256 of the row's `public_test_cases` and `private_test_cases` strings as the dataset stores them), so the tests a verdict was graded on can be fetched from the pinned revision and checked. The gold is never sent to the server |
| `leaderboard_comparable` | `true` only with lcb_runner's sampling (temperature 0.2, top_p 0.95, frequency/presence penalties 0, no other sampling field; `max_tokens` 2000 without `--eval-thinking`), 10 or more samples per problem, all four scenarios, the full window (no `--eval-limit`), a post-v1 release, every generation and self-repair request answered and no reply cut off at a limit other than lcb_runner's 2000 |

When it is `false`, `leaderboard_comparable_reason` lists each difference: fewer than 10 samples
per problem, a temperature other than 0.2, any other sampling field that differs from lcb_runner's
(a top_p other than 0.95, a decoding penalty, and so on), a no-think request sent less than
`max_tokens` 2000 (its prompt left less room in the context window), replies cut off at a limit
other than 2000 (named separately when they have no answer text, cut off inside the reasoning
channel), a
scenario that did not run, with why (`GBENCH_LCB_INCLUDE_TEST_GEN=0`,
`GBENCH_LCB_INCLUDE_SELF_REPAIR=0`, or dropped by `--eval-limit`), an
`--eval-limit` subset, a `release_v1` window, generation requests that got no reply from the
served endpoint (harness errors, left out of the score; lcb_runner retries a failed request up
to 10 times; an empty reply is the model's answer and does not count, and neither does a reply
that stopped with its answer inside the thinking channel: graded as lcb_runner grades an empty
extraction, counted in `answer_in_thinking` with `answer_in_thinking_rate`, never a failed request,
see [answer_in_thinking.md](answer_in_thinking.md)), or self-repair requests
that got no reply (harness errors, left out of the self-repair pass@1).
A comparable number is comparable to LiveCodeBench results for the same release window and date
range.

## Requirements

- A running OpenAI-compatible `/v1` endpoint.
- **bubblewrap** (`sudo apt-get install bubblewrap`, with unprivileged user namespaces allowed):
  the suite executes model-written code and **hard-errors** (`infra_required`) when isolation is
  unavailable. `GBENCH_SANDBOX=none` runs it unsandboxed on purpose.
- **`numpy` and `pandas`** in the Python that runs gbench (the graders pre-load them as
  upstream's evaluator process does, and code-execution checks import both); both come with
  `datasets`.
- Linux `prctl` (through `ctypes`) and `os.fork` in the grading interpreter: the grader marks
  itself non-dumpable and forks the process that runs the model's code (see *Verdict isolation*).
- Before generating anything, the suite grades four known-correct answers (stdin, call-based,
  code execution, test output prediction), and, with self-repair on, checks the failure metadata
  of a known wrong answer; it **hard-errors** (`infra_required`) if any check fails, so a broken
  grading environment cannot report a model as scoring 0%.
- A problem whose `metadata` is present but is not a JSON object stops the run (upstream's loader
  would raise too) instead of being graded as a stdin problem.
- Network access to the Hugging Face Hub on first run (public datasets; no token needed). The
  code-generation shards are about 1.3 GB for `release_v6`.
- Code execution and test output prediction are part of the default run: if either dataset
  cannot be loaded the suite raises (test output prediction can be left out on purpose with
  `GBENCH_LCB_INCLUDE_TEST_GEN=0`).

## Run

```bash
# Leaderboard-shaped run: lcb_runner's sampling is the default (10 samples per problem, then
# one self-repair generation for every rejected code generation sample, max_tokens 2000 on
# every request; --max-output-tokens is required but not applied without --eval-thinking)
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals lcb --sandboxes 8 \
       --max-output-tokens 16384

# Only problems published since 2025-01, with thinking (every request gets 65536 tokens: the
# 2000 cap is not applied to reasoning)
GBENCH_LCB_MIN_DATE=2025-01 gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals lcb --eval-thinking --sandboxes 8 \
       --max-output-tokens 65536
```

`--sandboxes` bounds generation concurrency (self-repair requests included); grading uses up to
32 worker processes. `GBENCH_LCB_TEMPERATURE` overrides the temperature for this suite alone
(else 0.0 without `--eval-thinking`, 1.0 with it); any value other than lcb_runner's 0.2 is
flagged non-comparable. A full `release_v6` run makes about (1055 + 479 + 442) x 10 generations plus one
self-repair generation per rejected code generation sample.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered samples (code
generation, then code execution, then test output prediction, concatenated) (round-robin:
positions I, I+N, I+2N, ...), selected before the `--eval-limit` draw, which is then taken within
the shard (N shards run up to N x `--eval-limit` samples). `--eval-limit N` is N samples for the
whole suite, spread over the scenarios that run (code generation, code execution, test output
prediction: `sampling.allocate_limit`, seeded `lcb`, each scenario one before any gets two), never
N per scenario; self-repair is a second pass on the same code generation samples, and the 10
generations per problem are attempts, not samples. The run logs its plan on its first line
(`[lcb] --eval-limit 5: 5 sample(s) over 3 scenario(s) (code_generation 2, code_execution 2,
test_output_prediction 1) x 10 attempts = 50 generations`) and records it as `eval_limit_plan`. The
result records `shard`, `shard_index` and `shard_count`, and a shard of more than one is not
leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`). `gbench
--merge` combines the shards' results with the suite's own merge (`merge_shards` in `lcb.py`): the
union of the shards' traces (docs/distributed-gbench.md), with every LiveCodeBench field
recomputed over it from the stored verdicts as one run computes them: each scenario's pass@1 (the
per-problem mean of c_i/n_i, problems matched across shards by `sample_key` and `global_idx`), the
code generation headline and its @k, self-repair from each trace's `lcb_self_repair`,
`lcb_scenarios` / `lcb_scenarios_not_run`, `contest_date_range`, the cut-off counts and the
comparability reasons that come from them (the reasons that come from the run's settings are the
shards' own). Nothing is graded again at merge time.
