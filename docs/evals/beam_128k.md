# beam_128k

BEAM ("Beyond a Million Tokens", Tavakoli et al., ICLR 2026) measures long-term conversational
memory: a model reads a long multi-session conversation between a user and an assistant, then
answers probing questions about it. This suite runs BEAM's **128K tier** (the paper's 100K/128K
conversations; the HF split is named `100K`): 20 conversations with 20 probing questions each,
two per memory ability, 400 questions in total. The ten abilities are abstention, contradiction
resolution, event ordering, information extraction, instruction following, knowledge update,
multi-session reasoning, preference following, summarization and temporal reasoning.

## Dataset
- `Mohammadta/BEAM`, file `data/100K-00000-of-00001.parquet`, pinned to revision
  `3205395e897e7318c7b094ef4e6047b9b82dbb03`.
- Each question carries a `rubric` (a list of criteria, 1 to 9 per question, 1,051 in total);
  the per-ability reference answer (`ideal_response`, `ideal_answer`, `answer`,
  `ideal_summary` or `expected_compliance`) is kept on the trace for reading only.
- `--eval-limit N` draws a subset stratified over the ten abilities.
  The run logs its planned workload at its start and records it as `eval_limit_plan`.

## Protocol
Ported from [mohammadtavakoli78/BEAM](https://github.com/mohammadtavakoli78/BEAM) at commit
`b2da22eac88bb0874c64665f13457eb99835774a` (the long-context evaluation path):
- **Prompt** (`src/answer_probing_questions/long_term_memory_methods.py`,
  `probing_question_evaluation`): the conversation goes in as real chat messages, every session's
  turns in order with their original `role` and `content` (metadata such as `time_anchor` is not
  sent), followed by upstream's final user turn, verbatim:
  `NOTE: Only provide the answer without any explanations.` / `Question: {question}` (with
  upstream's leading newline and indentation).
- **Earlier question turns** (as upstream's harness sends them). `probing_question_evaluation`
  reuses the saved history list by reference (`if saved_messages: messages = saved_messages`),
  appends the question turn to it and returns `messages[:-1]`, and `answer_generation.py` saves
  the list only once per conversation (after its first question). So from a conversation's third
  question on, upstream's prompt also carries the question turns of that conversation's questions
  2..k-1, as consecutive user turns before its own. gbench sends the same: the question at
  position p (0-based, in `answer_generation.py`'s order: `probing_questions`' keys, then each
  list's order, which the pinned HF data keeps) gets the question turns of positions 1..p-1, then
  its own. Each trace's `extra_payload` records `question_position` and
  `carried_question_turns`. The turns come from the conversation's full question list, so an
  `--eval-limit` subset sends each drawn question exactly the prompt it has in a full run.
- **Window pruning** (`prune_from_tail`): a conversation that does not fit the served window is
  cut from the head, keeping the most recent messages, always an even number of them so the
  kept history opens on a user turn. The pruned prompt leaves upstream's 2,000 tokens for the
  answer; with `--eval-thinking` it leaves the run's budget instead (`--max-output-tokens`, else
  the thinking floor), so pruning never cuts a thinking run below that budget (the token-limit
  policy). The conversations run 98k to 192k Gemma 4 tokens, so most are pruned at a
  131,072-token window and none at 262,144. Differences from upstream: upstream counts with
  `cl100k_base` (and when the history is roughly between window - 10,000 and window - 2,000
  tokens it returns an empty history, an upstream bug that is not reproduced); gbench counts the
  exact prompt (the server's `/tokenize`, else the run's tokenizer, else a conservative
  estimate). Each
  conversation is pruned once, sized for the drawn question whose turns after the history (its
  own plus the earlier question turns above) have the most tokens (not characters), so all its
  questions share one prefix and every one of them fits. Upstream prunes once too, for the
  conversation's first question, and adds the carried turns without pruning again.
- **Judge** (`src/evaluation/compute_metrics.py`): every rubric item is judged in its own call
  with `unified_llm_judge_base_prompt` (verbatim, `src/prompts.py`). The judge sees only the rubric
  item and the response, not the question or the reference answer, and returns JSON with a
  `score` of 1.0, 0.5 or 0.0; the question's score is the mean over its items. Replies are parsed
  as upstream parses them: `parse_json_response` (fenced block, whole reply, then the first
  `{...}`), and when that fails, `json.loads(json_repair.repair_json(reply))` with upstream's
  pinned `json_repair==0.44.1` (`requirements.txt` line 59). `repair_json` is heuristic and
  differs between releases, so gbench carries that exact release: the Python files of the PyPI
  wheel `json_repair-0.44.1-py3-none-any.whl` (sha256 `51d82532...`), byte-identical, under
  `gbench/third_party/_json_repair_vendor/` (MIT license kept there). The installed
  `json_repair`, if any, is not used. Before generation the suite checks the vendored files
  against their sha256 and raises if one differs. `json_repair_version` records `0.44.1`. A
  reply that still has no usable `score` (where upstream raises) is asked again, up to 3 calls in
  total, and then the question is marked `judge_grade: JUDGE_FAILED`, a harness error (below).
  The judge receives the reply as upstream's does (the model's `.content`), with only inline
  reasoning blocks (`<thought>...</thought>`, `<reasoning>...</reasoning>`) removed and no
  whitespace stripped. Every trace keeps each rubric item's judge replies with the Gemini model
  that gave each, and the score and `reason` read from the last (`beam_judge_calls`), as
  upstream's `llm_judge_responses` keeps each parsed judge JSON.
- **Event ordering** (`event_ordering_score`, `align_with_llm`): the response is split into lines
  as upstream splits it (`llm_response.split("\n")` on the unstripped reply, so a leading or
  trailing newline gives an empty line that stays unmatched);
  each line is matched to the first unused rubric event that upstream's `llm_equivalence` check
  (its system and user messages verbatim; "yes" in the reply means the same event) calls the same
  event; then Kendall tau-b (`scipy.stats.kendalltau`, variant b) between the reference order and
  the aligned response order, normalised to `tau_norm = (tau + 1) / 2`. A repeated
  (event, line) pair is asked once, since the judge runs at temperature 0. A NaN tau (constant
  ranks) counts as 0, like upstream's None case. Each alignment check is kept on the trace
  (`beam_alignment_calls`: the event, the line, the Gemini model, its reply and the decision).
- **Judge call parameters** (`src/llm.py`: `BuildLLm(model_name="gpt-4.1-mini", temperature=0)`,
  a `ChatOpenAI` called with `invoke(prompt)`): temperature 0, no `max_tokens` and no response
  format, so the rubric-item reply is read by upstream's parse and `json_repair` fallback. The
  result records these under `judge_sampling`.
- **Sampling**: upstream's answer generation runs at temperature 0 (`answer_generation.py`:
  the `--temperature` default and `model_config_default`); gbench sends the temperature of its
  rule, **1.0 with `--eval-thinking`, 0.0 without**.
  `--temperature` or `GBENCH_BEAM_128K_TEMPERATURE` override it in both modes. The
  result records `official_sampling` (the reference, the temperature used, its source and
  `temperature_matches_official`), and a mismatch is named in `leaderboard_comparable_reason`.
  Upstream's reader model is given no `max_tokens` (`src/llm.py` `BuildLLm`, or
  `init_chat_model`), so no request carries one: each answer may use what the window leaves
  after the pruned prompt, and `--max-output-tokens` (required by the CLI) is not sent
  (`max_output_tokens_note`; `official_sampling.send_max_tokens: false`). With
  `--eval-thinking` it sets the answer room pruning keeps.

## Headline metric and how to read it
- Per question: the mean rubric-item score, except event_ordering, which uses `tau_norm` (as
  upstream's `report_results.py` does). The rubric-item mean is also kept for event_ordering
  questions (`beam_llm_judge_score`), with precision, recall, F1 and `final_score` under
  `beam_event_ordering`.
- `beam_ability_scores`: the mean per ability (x100); the per-category table shows the same
  numbers (the 0.5 pass rate is kept as `pass_rate_at_0.5`).
- `accuracy`: the **unweighted mean of the ten ability scores**, the paper's Average row.
- Judge outages (`JUDGE_OUTAGE`) are excluded from the means and reported as `judge_outages`.
- **Harness errors** are never scored as the model's answer. A failed request (no response: a
  timeout, an over-context prompt, a server error; `judge_grade: HARNESS_ERROR`, with
  `harness_error` on the trace) and a question whose judge gave no usable score
  (`JUDGE_FAILED`) are left out of the ability means and of the pass rate (`scoring_excluded`),
  counted in `harness_errors` (`endpoint_failure`, `endpoint_timeout`, `over_context`,
  `judge_no_score`; also in `failed_requests` / `judge_failures`) and named in
  `leaderboard_comparable_reason`, and the run's `status` is `completed_with_errors`. Upstream's
  `answer_generation.py` catches a failed model call and records the text
  `Error in <ability> question type and <question> and for <file>` as the reply, which its judge
  then scores; gbench does not score a harness failure. A run with no question scored withholds
  its score (`accuracy: null`, `status: failed`, `accuracy_withheld_reason`). An empty response
  is judged like any other, as upstream judges it.
- The result records `beam_window_tokens`, `beam_output_reserve_tokens` (the answer room
  pruning keeps: 2,000, or the thinking budget), `beam_conversations`,
  `beam_pruned_conversations`, `harness_errors`, `judge`
  (`gbench-gemini-cascade`), `judge_models` (answered judge calls per model, alignment calls and
  retries included), `judge_cascade` (the configured cascade order) and `judge_sampling`.
  Each trace's `extra_payload` shows `history_messages` kept out of `history_messages_total`.

`leaderboard_comparable` is always **false**: upstream judges with gpt-4.1-mini at temperature
0, gbench with its Gemini judge cascade (named in `leaderboard_comparable_reason`). The reason
also states, on an `--eval-thinking` run, how many conversations were pruned with the
thinking-budget reserve instead of upstream's 2,000 tokens, any harness errors, and names an `--eval-limit` subset, a temperature other than 0 and replies cut off by the
run's token budget rather than at an official cap (`replies_cut_off_by_budget`;
`replies_cut_off_in_reasoning` counts those that ran out inside the reasoning channel with no
answer).

## Requirements
- **`GEMINI_API_KEY`** or **`GEMINI_API_KEYS`** (one key, or a comma/semicolon list;
  `GEMINI_API_KEYS` wins when both are set) for the judge. A missing or rejected key
  **hard-errors** (`infra_required`) before generation; there is no fallback scoring.
- `google-genai` and `scipy` (both in the `evals` extra: `pip install -e '.[evals]'`); without
  scipy the suite hard-errors (`infra_required`) before generation. `json_repair` 0.44.1 ships
  with gbench (vendored, see Judge above); nothing needs to be installed for it.
- The served context window: read from the server (`GET /v1/models`); set
  `GBENCH_BEAM_128K_WINDOW` (tokens) when the server does not report it, otherwise the suite
  hard-errors instead of sending unpruned conversations. A window too small to hold even the
  question plus the answer room (2,000 tokens, or the thinking budget) also hard-errors.
- For exact pruning, a vLLM endpoint (it has `/tokenize`) or `--tokenizer`.
- Network access to the Hugging Face Hub (or a warm HF cache) for the pinned revision.

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals beam_128k --max-output-tokens 16384
```
Add `--eval-limit 20` for a quick (non-comparable) subset. Loading counts prompt sizes to prune
the conversations (under a minute with a local tokenizer at a 128K window).

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered questions
(conversation by conversation) (round-robin: positions I, I+N, I+2N, ...), selected before the
`--eval-limit` draw, which is then taken within the shard (N shards run up to N x `--eval-limit`
questions). The result records `shard`, `shard_index` and `shard_count`, and a shard of more than
one is not leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`);
`gbench --merge` combines the shards' results as docs/distributed-gbench.md describes: it
re-counts the union of their traces, and refuses a suite whose headline is not a correct / scored
count over them unless the suite defines its own merge.
