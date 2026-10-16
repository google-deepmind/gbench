# custom_jsonl

Run your own evaluation from a JSONL file (no plugin, no code).

> **Note:** `custom_jsonl` is *not* part of `--evals all` (it needs a file argument). Pass
> it explicitly with `--eval-custom-jsonl`, or see it listed via `gbench --list evals`.

## Usage
```bash
gbench --evals-only \
       --eval-custom-jsonl /path/to/my_eval.jsonl \
       --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-26B-A4B-it \
       --max-output-tokens 16384
```

## File format
One JSON object per line. Either a chat-style `messages` list or a bare `prompt`:

```jsonl
{"prompt": "What is the capital of France?", "answer": "Paris", "category": "geography"}
{"messages": [{"role": "user", "content": "2+2?"}], "answer": "4"}
```

| Field | Required | Meaning |
|---|---|---|
| `prompt` *or* `messages` | yes | the request sent to the model (`question` or `input` is read as the prompt too); a line with none of these stops the run with an error naming the line (it would silently shrink the denominator) |
| `gold` / `answer` | no | expected answer (first present of `gold`, `answer`, `gold_answer`, `target`); without it the sample cannot be scored |
| `category` | no | groups rows in `category_accuracy` (`domain` is read when `category` is absent) |
| `eval_type` | no | how the row is scored (default `contains`); one of `contains`, `exact`, `numeric`, `multiple_choice` (also `mc`, `letter`): any other value stops the run with an error naming the line and the allowed values |

## Scoring
Deterministic, using the strategy the row declares in `eval_type`:

| `eval_type` | Passes when |
|---|---|
| `contains` *(default)* | the response equals the gold or contains it as a whole token (case-insensitive; no letter, digit or underscore right before or after it, so `Au` is not found in `because` and `4` not in `14`); failing that, a one-letter gold passes when the model's stated choice is that letter, and a numeric gold when some number in the response equals it within 1e-4 |
| `exact` | the whole response, surrounding whitespace stripped, equals the gold (case-insensitive), nothing else |
| `numeric` | some number in the response equals the gold within 1e-4. A number with commas is read both ways, as one number with comma-grouped thousands (`299,792,458`, `1,234.5`) and as a list (`85,90,100`, `(100,200)`), and passes when either reading holds the gold |
| `multiple_choice` (also `mc`, `letter`) | the model's *stated* choice (a boxed letter, else the letter after an answer anchor, else the last standalone letter A-J) is the gold letter |

An `eval_type` outside those values is an error, never scored as `contains`. Under `contains` a
one-letter gold still passes a response that uses that letter as a word (`B` in `plan B`), so give
such rows `multiple_choice`; give numeric golds `numeric`.

Rows with no gold cannot be scored and are reported as incorrect rather than silently
passed. Supply a gold for every row you want measured.

## Run knobs
`--eval-limit` caps the number of rows; `--eval-concurrency` (default `--batch-sizes[0]`) sets
the request concurrency; `--eval-thinking` enables the model's reasoning channel. With
`--eval-thinking`, set a `--max-output-tokens` large enough for the reasoning (e.g. 65536): a
reply cut off by that budget is scored on what arrived and counted in
`truncated_responses`.

The run logs its planned workload in one line at its start (for example `[<file stem>]
--eval-limit 5: 5 sample(s) x 1 attempt = 5 generations`) and records it in the result as
`eval_limit_plan` (`limit`, `samples`, `per_setting`, `attempts_per_sample`, `generations`).
