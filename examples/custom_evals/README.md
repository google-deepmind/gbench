# Custom evaluations: examples

Two ways to evaluate your own data in gbench, from least to most code. Full reference:
[`docs/evals/plugins.md`](../../docs/evals/plugins.md).

## 1. No code: a JSONL of prompts (`custom_jsonl`)

When your questions have a checkable answer (substring, exact, numeric, or multiple-choice), write a
JSONL and point the built-in `custom_jsonl` runner at it. No Python required.

```bash
gbench --evals-only --eval-custom-jsonl examples/custom_evals/sample_benchmark.jsonl \
       --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer <model> \
       --max-output-tokens 16384
```

[`sample_benchmark.jsonl`](sample_benchmark.jsonl) shows the row format: a prompt
(`prompt` / `question` / `input` / `messages`), a gold (`gold` / `answer` / `gold_answer` /
`target`), an optional `eval_type` (`contains` [default], `exact`, `numeric`, `multiple_choice`, also
spelled `mc` or `letter`; any other value is an error), and an optional `category`. `contains`
matches the gold as a whole token, case-insensitively (the sample's `Au` row passes "Au" or "The
symbol is AU." but not "because"). A row without a prompt is an error, never skipped. See
[`docs/evals/custom_jsonl.md`](../../docs/evals/custom_jsonl.md) for every rule.

## 2. Code: a plugin (`--eval-plugins-dir`)

When you need custom sample loading (a DB/API), an LLM judge, or a metric the four `eval_type`s do
not cover, write a plugin. [`custom_qa_eval.py`](custom_qa_eval.py) is a complete, minimal one: it
exposes `run_custom_qa(...)` (the loader registers any `run_<name>` function), builds
`(messages, gold, extra)` samples, and delegates to the shared `run_eval_suite` helper with a scorer.

```bash
gbench --evals-only --eval-plugins-dir examples/custom_evals --evals custom_qa \
       --remote-endpoint http://127.0.0.1:8000/v1 --tokenizer <model> \
       --max-output-tokens 16384
```

Copy `custom_qa_eval.py` into your own plugins directory and adapt the loader + scorer. The built-in
[`custom_jsonl.py`](../../gbench/runners/eval_suites/custom_jsonl.py) is structured the same way, so
it doubles as a second reference implementation.
