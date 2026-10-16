# longbench_v2_code

`longbench_v2_code` runs the **Code Repository Understanding** domain of
[LongBench-v2](https://github.com/THUDM/LongBench) (Bai et al., 2024): 50 four-choice questions
about multi-file code repositories whose context runs from about 100K characters to 16M
(roughly 38K to 5M tokens). It is not a separate upstream benchmark; the numbers it reports are
LongBench-v2 domain scores, in both of LongBench-v2's settings (non-CoT and w/ CoT).

Renamed from `culer` (gbench's earlier name for the same domain). `--evals culer` still runs
this suite, with a deprecation warning; every `GBENCH_CULER_*` variable below is still read as a
deprecated alias of its `GBENCH_LONGBENCH_V2_CODE_*` name, and `--eval-limit` draws the same
questions as before the rename.

## Dataset
- `zai-org/LongBench-v2` (formerly `THUDM/LongBench-v2`), split `train`, pinned to revision
  `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9`: 503 questions, 50 of them in the
  `Code Repository Understanding` domain (18 easy / 32 hard; 12 short, 9 medium, 29 long).
- `GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS=1` (legacy aliases `GBENCH_CULER_ALL_DOMAINS=1`,
  `CULER_ALL_DOMAINS=1`) scores all 503 questions,
  which makes the result an overall LongBench-v2 number rather than a code one.
- `--eval-limit N` runs N samples in total, a (setting, question) pair being one sample: the N
  are dealt over the two settings (non-CoT and w/ CoT), and each setting draws its share
  stratified by domain with a fixed seed, so the smaller share's questions are among the larger's
  (`--eval-limit 5`: 5 questions over the two settings, e.g. 3 non-CoT and 2 w/ CoT, not 5 in
  each). The headline setting (non-CoT) always gets a sample (`--eval-limit 1` runs one non-CoT
  question). A setting dealt no sample is not run. The run logs the plan at its start and records it as `eval_limit_plan`.
- `--shard I/N` selects shard I of the questions (round-robin in dataset order, after the domain
  filter) before the `--eval-limit` draw; the result records `shard`, and a shard of more than one
  is not leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`).
- Without `--eval-limit` both settings run the same questions (every question of the shard).

## Protocol
Ported from THUDM/LongBench `pred.py` at commit `2e00731f`
(`2e00731f8d0bff23dc4325161044d0ed8af94c1e`, the repository's current head; `pred.py` and the
three prompt files were last changed in `c5ea10bc`). LongBench-v2 reports two settings side by
side, and gbench runs both by default, on the same questions, each scored and reported on its
own:

- **non-CoT** (`python pred.py`, the headline): `prompts/0shot.txt` verbatim, filled with
  `pred.py`'s `.replace` chain (`$DOC$`, `$Q$`, `$C_A$` to `$C_D$`, each stripped), sent as one
  user message with `max_new_tokens=128`; that reply is scored.
- **w/ CoT** (`python pred.py --cot`): two queries per question.
  1. The CoT query: `prompts/0shot_cot.txt` verbatim (the same document and question, ending
     "Let's think step by step:"), filled with the same chain, `max_new_tokens=1024`.
  2. The answer query: `prompts/0shot_cot_ans.txt` verbatim ("The text is too long and omitted
     here.", the question and choices, "Let's think step by step: $COT$", then "Based on the
     above, what is the single, most likely answer choice? Format your response as follows:
     ..."), filled with the same chain and then `.replace('$COT$', response)`, where `response`
     is the CoT query's reply, stripped; `max_new_tokens=128`. This reply is scored.

  As in `pred.py` (`if output == '': continue`), a CoT query that fails or returns an empty
  reply gets no answer query. The chain of thought is the reply's content, without a reasoning
  channel (the `message.content` that `pred.py` reads).

A request of either setting (either query for w/ CoT) that gets no reply from the served
endpoint (failed, timed out, or over the context window) is a harness error, never the model's
answer: the question is left out of every number of its setting (`accuracy`,
`category_accuracy`, `by_difficulty_and_length`, `no_extractable_answer`, the @k block),
counted (`scoring_excluded`, `request_harness_errors`) and named in the setting's
`status_reason` and in the top-level `status_reason`, labelled by setting. When a setting has no
scored question, its `accuracy` is null with `accuracy_withheld_reason`.

**Empty and null replies are asked again, as a resumed `pred.py` run does.** `query_llm` returns
the reply's `message.content`; `get_pred` writes no prediction for a question whose output is
`''` (`if output == '': continue`) or None (`output.strip()` raises and ends the worker), and a
resumed `pred.py` run (it skips the questions already in its output file) asks it again. gbench's
requests (every query of both settings) go through a loopback relay that asks such a reply again,
the same request (`swebench_common.upstream_reask_relay`, `reply_content_is_null_or_empty`), up to
`NULL_REPLY_REASKS` = 5 re-asks. A reply with its answer inside the thinking channel has null
content, so it is asked again. Only a reply the model ended itself (`finish_reason` "stop") is asked
again: a reply cut off at the token budget (`finish_reason` "length", e.g. a thinking reply that ran
out of budget inside its reasoning) is the model's own outcome, never re-asked, scored as base
scores a truncated reply and named (`upstream_reask.cut_off_not_reasked`). The re-asks stay inside
base's per-request timeout (`EndpointRelay.reask_deadline`); a request whose re-asks run out, reach
that deadline or get no reply from the endpoint is scored on the last reply it got
(`reasks_exhausted`, of them `reasks_cut_short`), never a harness error. A reply still empty or null
after the 5 re-asks is scored wrong and
named in the reason (an answer-in-thinking one is counted in `answer_in_thinking`, both queries
added, `answer_in_thinking_rate` over every request, and does not make the status
`completed_with_errors`). The setting's `upstream_reask` records what was asked again and the
classes of the empty replies the relay saw (`answer_in_thinking_seen`,
`answer_in_thinking_rate_seen`, ...; per query
for w/ CoT: `upstream_reask.cot_query`, `upstream_reask.answer_query`). One difference remains for
w/ CoT: a resumed `pred.py --cot` run asks a question's CoT query again before a new answer query,
while gbench asks only the empty answer query again, with the CoT it has; when that happens the
setting names it in `leaderboard_comparable_reason`. See
[answer_in_thinking.md](answer_in_thinking.md).

Common to every query:
- **Truncation**: the filled prompt is tokenized with the served model's tokenizer; when it is
  longer than `max_len` tokens, the middle is dropped (`input_ids[:max_len//2] +
  input_ids[-max_len//2:]`) and the rest decoded back to text, so the instructions and the
  question at the end always survive. `pred.py`'s `query_llm` does this for every query, the
  CoT answer query included (whose prompt is short, so it is not cut in practice). Upstream sets
  `max_len` per model to the model's context window: its README says
  "first add your model path and its context window length to `config/`" and "Set
  `--max_model_len` to the context window length of the model", and `config/model2maxlen.json`
  gives 120,000 to every 128K-context model and 200,000 to the 200K-context
  `claude-3.5-sonnet-20241022` (the LongBench-v2 paper, section 4.1: "middle truncation ... for
  sequences exceeding the model's context window length"). gbench derives upstream's `max_len`
  (`upstream_max_prompt_tokens`) from the served window: **120,000 for a window up to 131,072
  tokens** (upstream's value for 128K-context models), and for a larger window **the window minus
  1,024 minus 64** tokens: the largest prompt the window holds with the larger official output cap
  of the two settings (the CoT query's 1,024; the non-CoT query's is 128) and 64 tokens of chat
  template, one `max_len` for every query of both settings, as `pred.py` uses one per model. The `max_len` sent is `min(upstream_max_prompt_tokens, window -
  output budget - 64)`, where the output budget is the `max_tokens` the query will carry: it is
  upstream's value whenever the window holds it plus the budget, and only a window too small for
  that (or a `--eval-thinking` budget larger than the official cap) shortens it, so that the
  answer budget is never squeezed. Examples without thinking: any window of 120,192 to 131,072
  tokens gives 120,000 for the non-CoT query (budget 128), and any window of 121,088 to 131,072
  tokens gives 120,000 for the CoT query (budget 1,024); a 262,144-token window gives 262,144 -
  1,024 - 64 = 261,056 for both settings. With `--eval-thinking --max-output-tokens 65536` a 131,072-token window gives
  131,072 - 65,536 - 64 = 65,472 (upstream's is 120,000), and a 262,144-token window gives
  262,144 - 65,536 - 64 = 196,544 (upstream's is 261,056 for both settings); prompts longer
  than that are counted as cut shorter than upstream.
- **The model's own context window**: gbench reads it from the model's config
  (`max_position_embeddings`, or its `text_config`'s) in the local Hugging Face cache, for
  `--tokenizer` (else the model id); `model_context_window` and `model_context_window_source`
  record it (null and "unknown" when no local config gives one). When it is larger than the
  served window, a prompt truncated on the served window keeps less than LongBench-v2's run at the
  model's full window would, and the setting names it in `leaderboard_comparable_reason`
  (LongBench-v2 serves a model at its context window: serve with `--max-model-len` equal to it).
- **Output budget**: each query's `query_llm(..., max_new_tokens=...)` is that query's official
  cap (`OfficialSampling.max_tokens`), applied by gbench's token-limit policy. A no-think
  request carries it exactly, whatever `--max-output-tokens` says: `max_tokens=128` for the
  non-CoT query, `max_tokens=1024` for the CoT query and `max_tokens=128` for the CoT answer
  query (`max_output_tokens_note` records a flag that was not applied). A `--eval-thinking`
  request carries the run's budget instead (`--max-output-tokens`, else the thinking floor), for
  all three queries, because the caps were written for non-reasoning replies and the reasoning
  shares the budget; `official_sampling` records the cap (`max_tokens: 128`, or 1024 for the CoT
  query), `max_tokens_applied: false` and the note "official cap not applied to reasoning".
- **Extraction**: `extract_answer` verbatim, on the scored reply. After removing `*`, the reply
  must contain `The correct answer is (X)` or `The correct answer is X` (X in A to D); otherwise
  there is no answer and the question is scored wrong.
- **Sampling**: gbench's rule, not `pred.py`'s 0.1 (`query_llm(..., temperature=0.1)` for every
  query): every query is sampled at **1.0 with `--eval-thinking`, 0.0 without**. The result
  records the 0.1 under `official_sampling`.
  `--temperature` or `GBENCH_LONGBENCH_V2_CODE_TEMPERATURE` (legacy alias
  `GBENCH_CULER_TEMPERATURE`) overrides it, for both settings and both modes.

## Headline metric
`accuracy` is the non-CoT setting's share of questions whose extracted letter equals the gold
letter (`headline_setting: non_cot`). Each setting is also reported on its own:
- `headline_metrics`: `non_cot_accuracy` and `cot_accuracy` (null for a setting that was not
  run or errored);
- `settings.non_cot` and `settings.cot`: each setting's full result without its traces
  (`accuracy`, `total_questions`, `correct_answers`, `category_accuracy`, `status`,
  `official_sampling`, `max_output_tokens_note`, `longbench_v2_code_report`,
  `leaderboard_comparable` and `leaderboard_comparable_reason`). `settings.cot` also has
  `unanswered_questions` and a per-query breakdown in `cot_query` and `answer_query` (requests,
  failed / empty / cut-off replies, the `max_tokens` sent, replies at the official cap, the
  note); its `official_sampling` holds one block per query (`cot_query`, `answer_query`);
- `sample_traces` holds both settings' traces, each marked with its `setting`, one per question
  (and per attempt with `--attempt-count`): `sample_idx` is unique over both settings, and
  `source_sample_idx` is the question's index, the same in both. A w/ CoT trace is the answer
  query's (`cot_step: answer`, with the CoT query's status, finish reason and token count in
  `extra_payload`), or the CoT query's when no answer query followed (`cot_step: cot`).

The other top-level fields (`official_sampling`, `max_output_tokens_note`, `failed_requests`
and the other counters, `category_accuracy`, `longbench_v2_code_report`) are the non-CoT
setting's; `status`, `leaderboard_comparable` and `duration_s` cover the whole run. `status` is
`success` only when both settings are.

`longbench_v2_code_report` (per setting): the dataset revision and scope, `window_tokens`,
`model_context_window` and `model_context_window_source`, `output_budget_tokens`,
`max_prompt_tokens`, `upstream_max_prompt_tokens` (upstream's `max_len` for the served window, as
above) and `upstream_max_len_rule` (how it was derived), the `tokenizer` used,
`prompts_truncated`, `prompts_cut_shorter_than_upstream` (prompts longer than
`max_prompt_tokens` when that is below `upstream_max_prompt_tokens`, i.e. cut shorter than
`pred.py` would cut them),
`no_extractable_answer` (scored replies without a "The correct answer is" statement, empty ones
included) and `by_difficulty_and_length` (easy / hard / short / medium / long, the
columns LongBench-v2's `result.py` prints). For w/ CoT, the budgets and truncation fields are the
CoT query's, and the report adds `answer_output_budget_tokens`, `answer_max_prompt_tokens`,
`answer_prompts_truncated`, `answer_prompts_cut_shorter_than_upstream` and
`questions_without_cot` (CoT queries that failed or came back empty). `category_accuracy` is per
domain.

`leaderboard_comparable` is computed per setting from what the run actually did: a setting's is
**true** when none of the differences below applies to it, and otherwise **false**, with its
`leaderboard_comparable_reason` listing every one that applies:
- the code-domain slice (unless `GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS=1`);
- a `--eval-thinking` run (LongBench-v2 has no native-reasoning setting), with the budget its
  requests carried (upstream sends `max_tokens=128`, or 1024 for the CoT query, which is not
  applied to reasoning);
- a no-think request sent a `max_tokens` other than its official cap (the context clamp shrank
  it);
- prompts cut shorter than upstream's `max_len` for the served window
  (`prompts_cut_shorter_than_upstream` > 0, which happens on a window of at most 131,072 tokens
  below 120,000 plus the query's output budget plus 64, or when a `--eval-thinking` budget is
  larger than the official cap; for w/ CoT also `answer_prompts_cut_shorter_than_upstream`);
- prompts truncated on a served window smaller than the model's own context window
  (`model_context_window` > `window_tokens`);
- a temperature other than 0.1 (gbench's 1.0 with `--eval-thinking` or 0.0 without, or an
  operator-set value);
- questions whose request got no reply from the served endpoint (failed, timed out, or did not
  fit the window: harness errors, left out of the score) or still came back empty or null after
  the 5 re-asks (scored wrong), for w/ CoT in either query (`pred.py` writes no prediction for
  either: `query_llm` returns `''` after 5 failed tries, and `get_pred` skips an empty output; a
  resumed run asks them again);
- replies cut off at the token budget with empty or null content
(`upstream_reask.cut_off_not_reasked`;
  scored wrong and not asked again, while a resumed `pred.py` run would ask them again);
- w/ CoT: answer queries asked again with the same CoT (a resumed `pred.py --cot` run asks the
  CoT query again too);
- an `--eval-limit` subset.

The top-level `leaderboard_comparable` is true only when both settings' are; its reason lists
the differences of both, a difference only one setting has followed by that setting's name
(`(non-CoT)` or `(w/ CoT)`). So a full, all-domain (`GBENCH_LONGBENCH_V2_CODE_ALL_DOMAINS=1`),
no-think run at temperature 0.1, served at the model's context window, on a window of at least
121,088 tokens, with every request answered, is comparable with LongBench-v2's published w/o CoT
and w/ CoT numbers; on a window of 120,192 to 121,087 tokens the non-CoT setting still is.

If the window leaves room for the non-CoT query but not for the CoT query (fewer than 4,096
prompt tokens after the CoT query's 1,024-token budget; only a window of 4,288 to 5,183 tokens),
the non-CoT setting runs and is scored, and `settings.cot` is an `infra_required` entry naming
the window (not run, not scored, `accuracy` null); `settings_infra_required` lists it and the
status is `completed_with_errors`.

## Requirements
- A serving endpoint with a **large context window**. The window is read from the server
  (`GET /v1/models`, which vLLM reports); if the server does not report one, set
  `GBENCH_LONGBENCH_V2_CODE_WINDOW` (legacy alias `GBENCH_CULER_WINDOW`) to the served window
  in tokens, otherwise the suite stops with an error. If the window leaves fewer than 4,096
  prompt tokens after the output budget, the suite also stops with an error. Serve the model at
  its context window, as LongBench-v2 does. Without thinking, a window of at least 121,088 tokens
  (120,000 prompt tokens + 1,024 + 64) truncates exactly as upstream does in both settings
  (120,192 is enough for the non-CoT setting alone); a `--eval-thinking` run keeps upstream's
  `max_len` only when its budget is no larger than the official cap, or, on a window of at most
  131,072 tokens, when the window holds 120,064 plus its budget.
- The served model's Hugging Face tokenizer, loadable with `transformers`
  (`AutoTokenizer.from_pretrained`). gbench uses `--tokenizer`, else the model id. If it cannot
  be loaded (no `transformers`, an id that is not a Hugging Face repo, no network on the first
  run), the suite stops with an error rather than truncating by characters.
- The `datasets` package and network access to the Hugging Face Hub on the first run (the
  download is about 450 MB).
- Tokenizing the long contexts takes a few minutes on CPU (the largest context alone takes
  about 15 seconds), once before each setting's first request, since each setting tokenizes its
  own prompt. The w/ CoT setting sends two queries per question, the first over the full
  document with up to 1,024 output tokens, so a default run takes more than twice as long as
  the non-CoT setting alone.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals longbench_v2_code --max-output-tokens 65536
```
`--max-output-tokens` applies to a `--eval-thinking` run only. Add `--eval-limit 10` for a quick
check. Both settings run; there is no switch to run one alone.
