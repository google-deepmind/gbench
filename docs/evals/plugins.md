# Evaluation plugins

`gbench` discovers custom evaluation suites from plugin files, so a new suite can be added without
changing gbench's code:

```bash
gbench --evals-only --evals all \
       --eval-plugins-dir /path/to/plugins \
       --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer <model> \
       --max-output-tokens 16384
```

`--evals all` runs the built-in suites and every discovered plugin; `--evals plugins` runs only the
plugins; a plugin can also be named on its own (`--evals <plugin_name>`).

## Start here: a worked example

`gbench` ships a minimal, dependency-free example under [`examples/custom_evals/`](../../examples/custom_evals/):

- [`custom_qa_eval.py`](../../examples/custom_evals/custom_qa_eval.py) is a complete plugin. It
  defines `run_custom_qa(model_name, base_url, concurrency, enable_thinking=False, **kwargs)` (the
  loader registers any `run_<name>` function), builds its samples, and delegates to the shared
  `run_eval_suite` helper with a scoring function. Copy it and adapt the loader and scorer:
  ```bash
  gbench --evals-only --eval-plugins-dir examples/custom_evals --evals custom_qa \
         --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer <model> \
         --max-output-tokens 16384
  ```
- [`sample_benchmark.jsonl`](../../examples/custom_evals/sample_benchmark.jsonl) is a dataset for
  the built-in `custom_jsonl` runner, the no-code form of the same idea: point it at a JSONL file
  of your own prompts and it scores each row deterministically. Use it when your questions have a
  checkable answer and you do not need custom loading, an LLM judge or a metric of your own:
  ```bash
  gbench --evals-only --eval-custom-jsonl examples/custom_evals/sample_benchmark.jsonl \
         --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer <model> \
         --max-output-tokens 16384
  ```
  Row format: a prompt (`prompt` / `question` / `input` / `messages`), a gold (`gold` / `answer` /
  `gold_answer` / `target`), an optional `eval_type` (`contains` [default], `exact`, `numeric`,
  `multiple_choice`), and an optional `category`. See [custom_jsonl.md](custom_jsonl.md).

Which to use: the built-in `custom_jsonl` (a data file, no code) when the four `eval_type` checks
are enough; a plugin like `custom_qa_eval.py` (code) when you need custom sample loading, a judge,
or a metric the built-in scorer does not cover. `custom_jsonl` is itself structured like a plugin,
so it doubles as a second reference implementation
([`gbench/runners/eval_suites/custom_jsonl.py`](../../gbench/runners/eval_suites/custom_jsonl.py)).

## Discovery

- Where: the paths given to `--eval-plugins-dir` (alias `--eval-plugins-path`; directories or
  single `.py` files) and the `GBENCH_EVAL_PLUGINS_PATH` environment variable (paths separated by
  `:`, the OS path separator). A directory is searched recursively for `*.py`. Files whose name
  starts with `_`, `.` or `test_`, files named `dataset_utils.py`, and anything under a directory
  whose name starts with `.` or `__` are not loaded. A path that does not exist is logged as a
  warning.
- What registers a suite: every function named `run_<name>` defined in the module (the suite name
  is `<name>` in lower case), and every callable in a module-level `SUITES` dict (`{"name": fn}`).
- `PILLAR` (or `PILLAR_TITLE`), a module-level string, is the heading the module's suites get in
  the results table (`[CUSTOM] <PILLAR>`, a leading `7. ` style number dropped); suites of a module
  without one are listed under `[CUSTOM] OTHER EVALUATIONS`.
- A module that raises while it is imported is logged as an error and registers nothing, so naming
  its suite with `--evals` is then an unknown-suite error.
- `gbench --list plugins --eval-plugins-dir <dir>` (or `gbench --list-plugins <dir>`) lists what
  a directory registers.

## The runner contract

```python
def run_<name>(model_name: str, base_url: str, concurrency: int,
               enable_thinking: bool = False, **kwargs) -> dict: ...
```

- `model_name` is the id the endpoint serves the model under, `base_url` the endpoint (ending in
  `/v1`), `concurrency` the number of requests to send at once (`--eval-concurrency`, default
  `--batch-sizes[0]`), `enable_thinking` whether `--eval-thinking` is set.
- `kwargs` carries the run's settings: `max_output_tokens` (`--max-output-tokens`, required for
  every eval run), `limit` (`--eval-limit`: N samples for the whole plugin suite in total, never N per setting; spread it over your suite's settings with `gbench.runners.eval_suites.sampling.allocate_limit`, keep the attempts per sample unchanged, log the planned workload in one line at the start and return it as `eval_limit_plan`), `temperature` (`--temperature`, or None for the
  default, the same for every suite: 1.0 with `--eval-thinking`, else 0.0), `attempt_count` (`--attempt-count`), `tokenizer` (`--tokenizer`),
  `sandboxes` (`--sandboxes`, for a plugin that runs containers), `eval_max_soft_tokens` and
  `eval_n_shot` (set for a plugin only in an `--evals all` run, since the CLI accepts
  `--eval-max-soft-tokens` and `--eval-n-shot` only with the built-in suites that take them or
  `all`), and `eval_categories` (routed per suite, so None for a plugin). Accept `**kwargs` so new
  settings do not break the plugin.
- Return the standard result dict. The simplest way is to build `(messages, gold, extra)` samples
  and return `run_eval_suite(...)` from `gbench.runners.eval_suites.base`, as both examples do: it
  sends the requests, applies the run's temperature, token budget and attempt count, scores each
  reply with your `eval_fn(response_text, gold) -> bool`, counts harness errors (an endpoint that
  gave no reply is left out of the score and named, never scored wrong) and records the result's
  traces and run shape. `extra` is a dict: its `category` groups the sample, and an id under
  `instance_id`, `task_id`, `question_id`, `id`, `uid`, `sample_id`, `problem_id` or `name` becomes
  the sample's `sample_key` (the key `--merge` matches shards by); those keys stay in gbench
  (`base.SAMPLE_META_BOOKKEEPING_KEYS`). Every other key of `extra` is a request field, sent in the
  sample's request body (for example a per-sample `max_tokens`, which follows the token-limit
  policy), so put nothing else there.
- A plugin that sends native tools to the model sets `requires_tool_calling = True` on its runner
  function (`run_<name>.requires_tool_calling = True`). A local run of it then requires
  `--tool-call-parser`, as the built-in `[tools]` suites do.
- Optional result fields the summary shows: `effective_n` (the headline's denominator, printed as
  Effective N; otherwise the scored traces or the question count) and `low_diversity` (true prints
  a low-diversity warning with `effective_n` under the suite's row). `category_accuracy` (a
  `{category: accuracy}` dict, filled by `run_eval_suite` from each sample's `extra["category"]`)
  is printed per category when it has 2 to 25 entries.

## Scoring helpers

- Tool calls: `gbench.runners.eval_suites.fc_common` parses tool calls from text or from the
  endpoint's `tool_calls` (`parse_tool_calls`, `parse_raw_tool_calls`, `parse_gold_calls`) and
  scores them: `score_tool_call` (one expected call) and `score_exact_call_set` (the exact set of
  expected calls: tool names, argument keys, equivalent values and, given the tool schemas,
  parameter types).
- LLM judge: a judged plugin calls `base.gemini_required_skip(eval_name, model_name)` before its
  first request (it hard-errors `infra_required` when no Gemini key is set, when the key is
  rejected, or when the API does not serve a model of the cascade) and grades with
  `base.judge_generate_cascade(contents, config=base.judge_config())`, which returns
  `(text, model_used)` from the first model of the Gemini cascade that answers, or
  `(None, "JUDGE_OUTAGE")`. Record an outage as `trace["judge_grade"] = "JUDGE_OUTAGE"`:
  `run_eval_suite` then leaves the sample out of the accuracy (a harness error, never scored 0).
  Record `model_used` on the sample so the result says which Gemini model answered.

## Missing prerequisites

A plugin that cannot produce a trustworthy score raises; it never returns a guessed or zeroed
result and never a "skipped" status. A result returned with status "skipped" (for example from
swebench_common.skipped_result, kept for existing callers) is treated the same way: a failed
configuration of kind infra_required, never a 0.00% row. A plugin that raises is a failed
configuration: the summary
prints `INFRA REQUIRED` or `ERROR` in place of its numbers, its accuracy is withheld (`null`, with
`accuracy_withheld_reason`), summary.json counts it in `failed_configurations`, and the run exits
2. The sweep continues with the next suite.

| Cause | Raise |
|---|---|
| Its dataset file is not provided | an exception class named `DatasetUnavailableError` (define it in the plugin); the summary shows `[INFRA REQUIRED] <name>: <message> (see docs/evals/plugins.md)` |
| A judged suite without a Gemini key (or with a rejected key) | nothing to write: `base.gemini_required_skip` raises `infra_required` |
| A missing toolchain, container image or other prerequisite | `raise infra_required(name, reason, docs_url)` from `gbench.runners.eval_suites.swebench_common` (see [toolchains.md](toolchains.md)) |

Any other exception is reported as `ERROR` with its message.

## Environment variables

- `GBENCH_EVAL_PLUGINS_PATH`: plugin paths, in addition to `--eval-plugins-dir`.
- `GEMINI_API_KEY` (or a key list in `GEMINI_API_KEYS`): the key for a judged plugin.
- `GBENCH_JUDGE_MODELS` / `GBENCH_JUDGE_MODEL`: the Gemini cascade (default `base.GEMINI_CASCADE`;
  see the judge cascade section of [env-knobs](env-knobs.md)). `base.resolve_judge_model()`
  returns its head for a plugin that makes a single-model call: the first model of
  `GBENCH_JUDGE_MODELS`, else `GBENCH_JUDGE_MODEL`, else `gemini-3.8-flash`, the head of the
  default cascade.
