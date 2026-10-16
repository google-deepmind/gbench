# seal_tools

Seal-Tools (Wu et al., NLPCC 2024; [fairyshine/Seal-Tools](https://github.com/fairyshine/Seal-Tools)):
choose the right tools from a retrieved API list and fill their parameters, including multi-tool
and nested calls. gbench runs the repo's `LLM_eval.ToolLearning_eval` protocol.

## Dataset (pinned)
`casey-martin/Seal-Tools` `test` split at revision `d0fe2245740d01a22b8fdd22ec1f49e48fcb1fbf`:
1354 rows (700 in-domain + 654 out-domain). They are row-for-row identical to
`Seal-Tools_Dataset/dataset_for_finetune/test_in_domain.json` + `test_out_domain.json` in
`fairyshine/Seal-Tools` at commit `ce753ecd60ed08dd376984035de531ab8421f1c6`.

## Protocol
- **Prompt**: each row's first conversation turn (instruction, `api_list` and
  `task_instruction`), sent unchanged as a single user message, as `ToolLearning_eval` does.
- **Parsing** (`transform_output_format("ToolLearning", ...)`): every `'` in the reply becomes `"`
  and newlines are removed, then the first `[{"api"...]` list (bracket-matched) is taken; it must
  contain `api`, `parameters` and `responses` and parse as JSON, otherwise the reply has no
  valid format. Calls written any other way (prose, `name`/`arguments` objects) are not read.
  Inline reasoning tags are removed from the reply first; the per-sample verdict and the pooled
  counts score the same text.
- **Scoring** (`calculate_score_ToolLearning`): gold is the row's second turn. Every predicted
  call with an `api` key counts; it is correct if a gold call has the same name (first match);
  each of its parameters is correct only if the gold call has that parameter and
  `str(predicted) == str(gold)` (no case folding, no type coercion beyond `str`). Counts are
  pooled over the whole test set (the repo's `src/eval.py` merges in-domain and out-domain).
- The port was checked against the upstream functions on 1354 x 10 synthetic replies (gold,
  JSON, fenced prose, lowercased, stringified, missing `responses`, name/arguments, extra call,
  first call only, apostrophes): identical Format ACC, Tool and Parameter P/R/F1.

## Headline and how to read the result
- `accuracy` = `parameter_f1` (pooled Parameter F1).
- Also reported: `format_accuracy` (upstream `AMOUNT`: share of replies with a parseable call
  list), `tool_precision` / `tool_recall` / `tool_f1`, `parameter_precision` /
  `parameter_recall`, `per_domain` (the same metrics for in-domain and out-domain), and
  `exact_match_rate` (share of rows where every gold call and parameter was reproduced with
  nothing extra; a gbench summary, not an upstream metric).
- `leaderboard_comparable` is `false`: gbench always sends a temperature, which upstream does
  not. `leaderboard_comparable_reason` lists the temperature and anything else that differs (a
  subset or shard, decoding penalties, more than one attempt).
- `official_sampling`: upstream's settings (the repo's chat-model wrapper,
  `LLM_Evaluation/src/llm_tools/llms/ChatGPT.py`, sends only `model` and `messages`), the
  temperature this run sent, its source and whether it matched.

## Requirements
- A running OpenAI-compatible `/v1` endpoint.
- Network and the `datasets` client on the first run (HF Hub); cached afterwards. A dataset that
  cannot be loaded stops the suite with an `[INFRA REQUIRED]` error.
- No API key and no LLM judge.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals seal_tools --max-output-tokens 4096
```
Add `--eval-limit N` for a quick check (stratified by domain; flagged in the result).

The run logs its planned workload in one line at its start (for example `[seal_tools] --eval-limit
5: 5 sample(s) x 1 attempt = 5 generations`) and records it in the result as `eval_limit_plan`
(`limit`, `samples`, `per_setting`, `attempts_per_sample`, `generations`).

Sampling: the temperature is gbench's rule for every suite: 1.0 with `--thinking`, 0.0 without.
No other sampling field is sent, as upstream's `ChatGPT.py` (fairyshine/Seal-Tools at
`ce753ecd60ed08dd376984035de531ab8421f1c6`) sends none (nor a temperature); the temperature is
recorded under `official_sampling` and named in the comparability reason. `--temperature`, or
`GBENCH_SEAL_TOOLS_TEMPERATURE` for this suite alone (it takes precedence), sets another.
No `max_tokens` is sent either, with or without `--thinking`, as `ChatGPT.py` sends none: the
server grants the room left in the context window, `--max-output-tokens` is not applied
(`max_output_tokens_note` says so), and a reply cut off at the end of the context window is
counted in `truncated_responses`, as it would be cut off upstream.
