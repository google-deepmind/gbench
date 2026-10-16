# cruxeval

CRUXEval (Gu et al., 2024, [facebookresearch/cruxeval](https://github.com/facebookresearch/cruxeval)):
code reasoning over 800 short Python functions, as two tasks.
- **CRUXEval-O** (output prediction): given `f` and an input, complete `assert f(input) == ??`.
- **CRUXEval-I** (input prediction): given `f` and an output, complete `assert f(??) == output`
  with any input that produces it.

gbench runs both tasks on every function (1,600 items).

## Dataset
- `cruxeval-org/cruxeval`, split `test`, pinned to revision
  `b96af0450242eb4da433032b90998f25588a5d0f` (800 rows: `code`, `input`, `output`).
- `--eval-limit N` draws N items in total, stratified over the two tasks with a fixed seed (never N
  per task); the pass@1 and pass@5 runs send the same items (their generations are attempts, not
  more items). The run logs its plan on its first line and records it as `eval_limit_plan`.
- `--shard I/N` selects shard I of the 1,600 items (round-robin in dataset order, with each
  function's CRUXEval-O and CRUXEval-I items side by side) before the `--eval-limit` draw; the
  result records `shard`.

## Protocol
Ported from facebookresearch/cruxeval at commit `190faf16`:
- **Prompts**: `prompts.py` verbatim. Without `--eval-thinking`, the 2-shot direct prompts
  (`make_direct_output_prompt`, `make_direct_input_prompt`), which end with an open `[ANSWER]`
  tag. With `--eval-thinking`, the chain-of-thought prompts (`make_cot_output_prompt`,
  `make_cot_input_prompt`, the published "+CoT" protocol). Each is sent as the user message
  after the system prompt of the repository's chat-model path (`openai/openai_prompt.py`):
  `You are an expert at Python programming, code execution, test case generation, and
  fuzzing.`
- **Stop and max_tokens**: `openai_run.py` sends `max_tokens=100` with the direct prompts and
  `max_tokens=1000` with the CoT prompts; each is the official cap of its mode
  (`OfficialSampling.max_tokens`), applied by gbench's token-limit policy. A direct (no-think)
  run sends the stop sequence `[/ANSWER]` and `max_tokens=100` exactly, whatever
  `--max-output-tokens` says (`max_output_tokens_note` records a flag that was not applied). A
  thinking (CoT) run is sent the run's budget (`--max-output-tokens`, else the thinking floor)
  instead of the 1000, which was written for non-reasoning CoT (the reasoning shares the
  budget); `official_sampling` records `max_tokens: 1000` and `max_tokens_applied: false`. It
  sends no stop sequence, because a stop string also matches inside the reasoning channel; the
  reply is cut at its first `[/ANSWER]` instead, which leaves the same text a stop sequence
  would.
- **Extraction**: `openai_prompt.py`'s `extract_answer_direct_output` / `_direct_input` /
  `_cot_output` / `_cot_input`, verbatim. For CRUXEval-O that is the text after the first `==`
  (after the `[ANSWER]` tag in the CoT variant); for CRUXEval-I, the `f(...)` call before `==`.
  There is no other answer format: a reply such as `Final Answer: 5` is executed as written
  and fails.
- **Scoring** (`evaluation/utils_general.evaluate_score`): run the function source followed by
  `assert <gold output> == <answer>` with upstream's executor, `evaluation/utils_execute.py`
  (vendored verbatim: `unsafe_execute` with `reliability_guard`, `swallow_io` and a 3-second
  limit); the item passes when the program finishes. So the check is Python equality, not
  string matching: `1` and `1.0` agree, as do `True` and `1`, while `'0'` and `0` do not. As
  upstream, a CRUXEval-O answer containing `f(<input>)` and a CRUXEval-I answer without `f(`
  are excluded rather than executed.
- **Who decides pass or fail**: the program runs in gbench's bubblewrap sandbox (read-only
  filesystem, private `/tmp`, no network), in the grader/child process split that `lcb` uses
  for its code-execution scenario. The child process runs the program with a stand-in in place
  of the gold output (`assert _gbench_expected_ == <answer>`); the stand-in sends the value it
  is compared with to a separate grader process, which alone holds the gold output and does
  the comparison. A sample passes only when exactly one comparison happened, the grader found
  it true, and the program then finished; the grader writes that verdict with a per-call
  nonce. The program's exit status plays no part in it, so an answer that exits the
  interpreter, raises `SystemExit` or kills its own process fails, as it does upstream.
- Values cross from the child to the grader as plain data (`None`, bool, int, float, complex,
  str, bytes, bytearray, list, tuple, dict, set, frozenset, Decimal, Fraction); every CRUXEval
  gold output is one of these. An answer whose value is of any other type (an instance of a
  class with its own `__eq__`, for example) never equals the gold output, where upstream would
  call that `__eq__`. An answer that is not a single value compared once fails even
  where upstream's `assert` would accept it: `f(5) or 2` makes upstream's assert true through
  `or 2`, and here it is one comparison that the grader finds false.
- Before generating anything the suite checks the grader on known answers (three correct ones
  must pass, a wrong one must fail) and stops with INFRA REQUIRED if it does not, since a broken
  grading environment would otherwise score every sample 0.

## Sampling
The published protocol (README; `openai/openai_run.py`: `n=10`, temperatures 0.2 and 0.8, run
for the direct and the CoT prompts alike):
- **10 samples per problem**. `--attempt-count K` above 10 raises it (flagged); smaller values
  keep 10.
- **Two passes**, with and without `--eval-thinking`: pass@1 from 10 samples (published at
  temperature 0.2), and pass@5 from 10 more samples (published at temperature 0.8). A full run
  is 2 x 1,600 x 10 = 32,000 generations. Each pass records its published settings under
  `official_sampling` (the pass@5 pass under `pass_at_5_run`).
- Both passes sample at gbench's temperature for every suite (`base.resolve_temperature`):
  **0.0** without `--eval-thinking`, **1.0** with it. Each pass's temperature is flagged with
  its source (`default` or `thinking`).
- `--temperature`, or for this suite alone `GBENCH_CRUXEVAL_TEMPERATURE` (which takes
  precedence), overrides the temperature of both passes in either mode and is flagged.

## Headline metric
`accuracy` is the **mean of CRUXEval-I pass@1 and CRUXEval-O pass@1**. pass@k is computed as in
`evaluate_generations.py`: the unbiased estimate from `n` samples and `c` passes per problem,
averaged over problems, where excluded generations are dropped from `n` for a problem only when
one of its samples passed. A generation whose request got no reply from the served endpoint
(failed, timed out, or over the context window) is a harness error, never the model's answer:
it is left out of its problem's samples (a problem with none left is not counted), counted and
named in `status_reason` and in the column reason. A task that ran with no scored generation at
all has no pass@1, so the mean has none either: `accuracy` is then null with
`accuracy_withheld_reason`. The pass@5 pass's own failures and harness errors are named in
`status_reason` too (`pass@5 run: ...`) and make the status `completed_with_errors`. An empty
reply fails its check.

`cruxeval_report` has one entry per task (`CRUXEval-O`, `CRUXEval-I`), which are CRUXEval's
leaderboard columns: `problems`, `samples_per_problem`, `pass_at_1` (with
`pass_at_1_temperature`), `pass_at_5` (with `pass_at_5_temperature`; from the second pass), `excluded_generations`, and each column's own `leaderboard_comparable` flag and
reason. `category_accuracy` shows the same two tasks as `output_prediction` and
`input_prediction` (per-sample accuracy of the pass@1 pass). `pass_at_5_run` summarizes the
second pass (temperature, sampling, status, failed requests, `max_tokens_sent`,
`response_health`) and keeps its generations in `pass_at_5_run.sample_traces`, so every pass@5
can be re-derived from the result. Each trace records its execution verdict in `cruxeval_check`.

A column (its pass@1 and pass@5) is comparable for a full no-think run with 10 samples, pass@1
at temperature 0.2 and pass@5 at 0.8 (operator-set: gbench's default is 0.0 for both), and in either pass no request without a reply (harness
errors) and no empty reply. Every direct request
is sent `openai_run.py`'s `max_tokens=100`; a direct request sent another value (only when the
prompt plus 100 does not fit the context window) or a direct reply longer than 100 tokens is
flagged. A `--eval-thinking` run is flagged: it adds the model's native reasoning to the CoT
prompts.

The result's `leaderboard_comparable` is always **false**, because upstream publishes
CRUXEval-I and CRUXEval-O separately and the headline averages them;
`leaderboard_comparable_reason` says so and adds the column reasons that apply (the temperature
of either pass, answer length, failed requests, a thinking run, an `--eval-limit` subset).

## Requirements
- A running OpenAI-compatible `/v1` chat endpoint that honours the `stop` and `max_tokens`
  parameters (vLLM does).
- **bubblewrap** for sandboxed execution (`sudo apt-get install bubblewrap`, with unprivileged
  user namespaces allowed; on Ubuntu 23.10+ run
  `sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0`). If it is missing or blocked the
  suite stops with an error; `GBENCH_SANDBOX=none` runs the checks unsandboxed on purpose.
- `numpy` and `pandas` importable by the Python that runs gbench (both are in the base
  install): the shared grader imports them before it forks.
- The `datasets` package (base install) and network access to the Hugging Face Hub on the
  first run. No judge and no API key.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals cruxeval --max-output-tokens 65536
```
This runs the published setting (10 samples, pass@1 at 0.2 and pass@5 at 0.8).
`--max-output-tokens` applies to a `--eval-thinking` run only. Add `--eval-thinking` for the
CoT prompts with native reasoning (same two passes).
