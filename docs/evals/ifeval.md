# ifeval

IFEval (Zhou et al., 2023, [arXiv:2311.07911](https://arxiv.org/abs/2311.07911)): 541 prompts
with verifiable instructions ("answer in at least 300 words", "wrap the entire output in JSON",
"do not use any commas", ...). Each prompt carries an `instruction_id_list` plus per-instruction
`kwargs`, and every instruction is checked programmatically. No LLM judge.

## Dataset
- `google/IFEval`, split `train`, pinned to revision
  `966cd89545d6b6acfd7638bc708b98261ca58e84`.
- One upstream correction is applied: google-research commit `26d8ccd` ("Fix an eval prompt",
  2024-06-11) changed key 2785 to ask for "at least 3 placeholders", which is what its kwargs
  check; the Hugging Face copy still says "at least one placeholder". With the fix, all 541
  prompts, instruction lists and kwargs equal upstream `data/input_data.jsonl`.
- The Hugging Face rows fill every kwarg name with `null`; those are dropped so each instruction
  receives only its own kwargs, as in the upstream jsonl.
- `--eval-limit N` scores a seeded random subset.
- `--shard I/N` selects shard I of the 541 prompts (round-robin in dataset order) before the
  `--eval-limit` draw; the result records `shard`.

## Protocol
- The prompt is sent as-is (one user message, no system prompt).
- **Scoring is upstream's own code**: `instructions.py`, `instructions_util.py`,
  `instructions_registry.py` and `evaluation_lib.py` from google-research
  `instruction_following_eval` at commit `e6890f85`, vendored in
  `gbench/third_party/eval_suites/_ifeval_vendor/` with the checkers unchanged (only imports
  differ; see its `__init__.py`). Each response goes through upstream's
  `test_instruction_following_strict` and `test_instruction_following_loose`.
- Two reproducibility choices that do not change what is measured:
  - `langdetect` is seeded (`DetectorFactory.seed = 0`); upstream leaves it unseeded.
  - Keys 1122 and 1129 ask for the frequency of the "letters" `#` and `!`, which upstream's
    `build_description` rejects and replaces with a random letter. gbench seeds that draw per
    prompt, so the same response always gets the same verdict.
- gbench trims leading and trailing whitespace from the reply before the per-sample check,
  whereas upstream checks the raw reply. Only `nth_paragraph_first_word` and the loose
  variants can see the difference (a reply that starts or ends with blank lines). The report
  re-checks every untrimmed reply and counts the prompts whose strict verdict would change.
- Verified: on upstream's GPT-4 reference responses
  (`input_response_data_gpt4_20231107_145030.jsonl`), this scorer and the upstream library
  give identical per-instruction verdicts on all prompts except the two random-letter ones.

## Headline metric
`accuracy` is **prompt-level strict accuracy**: a prompt counts when the response follows every
one of its instructions.

`ifeval_report` holds upstream `print_report`'s other numbers: `prompt_level_strict`,
`instruction_level_strict`, `prompt_level_loose`, `instruction_level_loose`, the counts
(`prompts`, `instructions`), `by_instruction_strict` (accuracy per instruction id) and
`prompts_changed_by_whitespace_trim`. `category_accuracy` groups prompts by the family of their
first instruction.

The headline and every number in `ifeval_report` are over the scored prompts: a request that got
no reply from the served endpoint (failed, timed out, or over the context window) is a harness
error, never the model's outcome. It is left out, counted (`failed_requests`, `scoring_excluded`,
`request_harness_errors`) and named in `status_reason` (status `completed_with_errors`); when no
prompt is scored, `accuracy` and the `ifeval_report` accuracies are null and
`accuracy_withheld_reason` says why. A reply that came back empty is the model's and follows no
instruction.

`leaderboard_comparable` is true for a full, greedy run in which every request was answered,
no reply was cut off at the max_tokens budget and trimming changed no verdict. Otherwise
`leaderboard_comparable_reason` names the cause: an `--eval-limit` subset, a sampled run (for
example `--eval-thinking` at temperature 1.0), requests that got no reply (`failed_requests`;
harness errors, left out of the score), replies that came back empty (they count as not
following their instructions), replies cut off at the budget (`truncated_responses`; checked on what arrived),
or the number of prompts whose verdict depends on the whitespace trim.

## Sampling
Temperature follows gbench's rule: 0.0 without `--eval-thinking` and 1.0 with it; `--temperature` or `GBENCH_IFEVAL_TEMPERATURE` overrides it.

## Requirements
- A running OpenAI-compatible `/v1` chat endpoint. No judge, no sandbox, no API key.
- Python packages used by the upstream checkers (both are in gbench's `evals` extra,
  `pip install -e ".[evals]"`):
  - `langdetect` (response language, all-capital and all-lowercase English):
    `pip install langdetect`;
  - `nltk` (sentence and word counting): `pip install nltk`.
- nltk's punkt tokenizer data: `python -m nltk.downloader punkt punkt_tab` (recent nltk
  versions load `punkt_tab`). Set `NLTK_DATA` if the data lives outside the default
  locations.
- If any of these is missing the suite stops with an error before generating anything; there
  is no heuristic substitute.
- The `datasets` package (base install) and network access to the Hugging Face Hub on the
  first run.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals ifeval --max-output-tokens 65536
```
Add `--eval-limit 20` for a quick check (not leaderboard-comparable).

The run logs its planned workload in one line at its start (for example `[ifeval] --eval-limit 5:
5 sample(s) x 1 attempt = 5 generations`) and records it in the result as `eval_limit_plan`
(`limit`, `samples`, `per_setting`, `attempts_per_sample`, `generations`).
