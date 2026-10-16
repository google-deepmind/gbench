# wildbench

WildBench (allenai/WildBench): 1,024 real user tasks collected from chatbot conversations
(WildChat), many of them multi-turn, each judged against its own checklist. The suite runs the
whole `v2` test split, the set the WildBench leaderboard scores, and reports the leaderboard's
columns: **WB-Score** (`WB_score.task_macro`, the headline, and `WB_score`; individual scoring),
**WB-Reward** (pairwise comparison with three reference models, with the length penalty rule)
and **WB-Elo** (`WB-Elo (Raw)` and `WB-Elo (LC)`: an Elo simulation over the leaderboard's 63-model
pool plus the tested model), computed as the official pipeline computes them.

> **Renamed.** This suite was previously registered as `lmsys_noncoding_hard`. The tasks are
> WildBench's, not LMSYS's, so the old name did not say what runs. The old name still works as a
> deprecated alias (one warning per run), and `GBENCH_LMSYS_NONCODING_HARD_TEMPERATURE` is still
> read as a deprecated alias of `GBENCH_WILDBENCH_TEMPERATURE`. The old suite ran only the
> non-coding tasks of the `v2-hard` config (170 tasks); it now runs all 1,024 `v2` tasks,
> coding included, so its numbers are not comparable with runs from before the rename.

## Dataset
- `allenai/WildBench`, config `v2`, split `test`, pinned at revision
  `26c49eb39d7d5ce2099b0bbafed5a88dcce954ec` (the loading call on the dataset card): **1,024**
  tasks. By `primary_tag`: Information seeking 182, Coding & Debugging 170, Creative Writing 146,
  Reasoning 133, Planning 116, Math 87, Editing 82, Data Analysis 33, Role playing 30,
  Brainstorming 24, Advice seeking 21. The loader hard-errors if the split does not hold 1,024
  tasks or a task has no conversation.
- WB-Reward reference replies: `allenai/WildBench-V2-Model-Outputs` (the `HF_RESULTS_PATH` of
  `src/eval.py`), pinned at revision `d6755bc68220df853c0825a733430f73f5af2501`, configs
  `gpt-4-turbo-2024-04-09`, `claude-3-haiku-20240307` and `Llama-2-70b-chat-hf` (1,024 replies
  each; the reply is `output[0]`). Each `<config>/train-00000-of-00001.parquet` must match its
  sha256 (`0e94087c...`, `6bfda71a...`, `2234f379...`, the HF LFS oids); a download failure is a
  hard error (`infra_required`) and a checksum or size mismatch raises.
- WB-Elo pool replies: the same `allenai/WildBench-V2-Model-Outputs` revision, the 63 configs
  that have WildBench score results upstream (the files of
  `eval_results/v2.0625/score.v2/eval=gpt-4o-2024-05-13/` at commit `d6b8dcaf377d`, the scores
  `leaderboard/wb_elo.py` reads); each `<config>/train-00000-of-00001.parquet` must match the
  sha256 in `WB_ELO_POOL` (about 231 MB in all, fetched through the Hugging Face cache). A download
  failure is a hard error (`infra_required`), a checksum or size mismatch raises. Checked: for
  all 64,449 WB-Score results upstream stores for these models, the reply read back from the
  score prompt built from the pinned file equals upstream's stored `model_output`.
- The per-category table uses `primary_tag`; the task-macro values use the task groups below.

## Protocol (allenai/WildBench commit `d6b8dcaf377d`)
- The model gets the task's `conversation_input` turns as chat messages. WildBench's inference
  settings are temperature 0 and top_p 1.0 (`TEMP=0; TOP_P=1.0` in `scripts/_common_vllm.sh` and
  every per-model script, also the `configs` of the reference replies); gbench sends top_p 1.0 in
  both modes and the temperature of its rule: **1.0 with `--eval-thinking`, 0.0 without**. Without `--eval-thinking` every request carries the
  scripts' `MAX_TOKENS=4096` as its
  `max_tokens`, so `--max-output-tokens` is not applied (`max_output_tokens_note`), and a reply
  that stops at 4,096 tokens is judged as it arrived, as upstream judges it
  (`replies_at_official_max_tokens` counts them). The scripts define no reasoning-model settings,
  so with `--eval-thinking` that cap, written for non-reasoning generation, is not applied: each
  request is sent the run's `--max-output-tokens` (the token-limit policy; the result records
  `official_sampling.max_tokens: 4096`, `max_tokens_applied: false` and the budget sent).
- The leaderboard numbers come from the batch pipeline (`EVAL.md`: `src/eval.py --batch_mode`,
  then `src/openai_batch_eval/batch_results_format.py`, `leaderboard/data_dir/_create_tables.py`
  and `_merge_results.py`). gbench follows it: every prompt is built as `src/eval.py`
  `placeholder_generation()` builds it (checked byte-identical against upstream's own function
  on 4,096 prompts: 1,024 score and 3 x 1,024 pairwise), and every judge reply is read as
  `batch_results_format.py` reads it. The judge call follows `src/eval.py`: temperature 0 and JSON
  mode, as upstream's `response_format json_object`; the result records these under
  `judge_sampling`.

### WB-Score (`--mode score --max_words_to_eval -1`)
- The judge gets `evaluation/eval_template.score.v2.md` **verbatim**: the conversation history
  (every turn but the last, as `USER:` / `ASSISTANT:` lines) and the current user query in their
  own sections, the model's response, and the task checklist as `- item` lines. The template asks
  the judge to write its analysis against the checklist, summarize strengths and weaknesses, and
  rate 1-10 in a JSON object `{"strengths", "weaknesses", "score"}`, using its bands (1-2 very
  poor ... 9-10 perfect). Nothing is truncated (`--max_words_to_eval -1`).
- Every reply is judged, an empty one included: the batch pipeline submits it like any other
  (upstream's own results hold judged scores of 1, 2, 4, 7 and 8 for empty replies). A request
  that got no reply from the served endpoint (a server error, a timeout, an over-context
  rejection) is a **harness error**, never the model's outcome: it is not judged in either mode
  (`judge_grade: not_judged_failed_request`), carries `harness_error` and
  `scoring_excluded: true`, and is left out of WB_score, WB_score.task_macro, WB-Reward and
  WB-Elo. It is counted in `failed_requests` and `scoring_excluded`, named in `status_reason`,
  and makes the run `completed_with_errors`.
- The score is read as `batch_results_format.py` reads it: `json.loads` of the reply must give an
  object with a `"score"`, then `float()` as `_create_tables.py` applies it. A reply that does not
  is dropped upstream; gbench marks it a grading failure, `JUDGE_FAILED` with
  `scoring_excluded`, excluded from the mean and counted in `judge_failures` (and
  `scoring_excluded`). A cascade outage (every model failed) is `JUDGE_OUTAGE`, excluded the same
  way and counted in `judge_outages`. A judged trace goes from `status: PENDING_JUDGE` to
  `status: OK`; an empty reply keeps `status: EMPTY_RESPONSE` (it is still judged).

### WB-Reward (`--mode pairwise --max_words_to_eval 1000`)
- Each reply is compared with each reference reply. The judge gets
  `evaluation/eval_template.pairwise.v2.md` **verbatim** with the history, the current query and
  both replies cut to their first 1,000 space-separated words (upstream's `shorten()`, which
  appends `... (truncated)`), `[This model response is empty.]` in place of an empty reply, and
  the checklist. It writes an analysis of each response and picks one of `A++`, `A+`, `A=B`,
  `B+`, `B++`.
- Response A / B order: as `src/eval.py` draws it, `random.seed(42)` then one `random.random()`
  per task in `v2` test order; the tested model is Response A when the draw is below 0.5. The
  same order serves the three references (158,671 of 158,671 upstream comparisons checked). An
  `--eval-limit` subset keeps the full-set order.
- The choice is read as `batch_results_format.py` reads it: `json.loads` must give an object whose
  `"choice"` is one of the five strings; any other reply is dropped (counted per reference in
  `wb_reward_judge_failures`, and `JUDGE_FAILED` on the trace). Such a drop is a judge-side
  harness error: it is also counted in `judge_failures_dropped.wb_reward`, sets `status` to
  `completed_with_errors` (`wb_reward_judge_failures N` in `status_reason`) and is named in
  `leaderboard_comparable_reason`, with how many of the dropped replies stopped at the judge's
  `max_output_tokens`. A cascade outage is counted in
  `wb_reward_judge_outages`, sets `status` to `completed_with_errors` and is named in the reason.
  A request that got no reply from the served endpoint is not compared (no judge call; its
  `wb_pairwise.<ref>` records `judge_grade: not_judged_failed_request` with no outcome), so it is
  in no reference's `total`. A WB-Score grading failure or refusal on a task does not remove its
  comparisons, which are judged on their own, as upstream's pairwise tables do not read the score
  mode.
- Reward, per reference, as `_create_tables.py` pairwise mode: much better +1, slightly better
  +0.5, tie 0, slightly worse -0.5, much worse -1. A comparison in which either reply was
  truncated or empty is not counted but stays in the denominator (`total` = the comparisons whose
  reply was read). Length penalty with margin K >= 0: a slight win (or loss) becomes a tie when
  the winner's reply is more than K characters longer; K = -1 disables it. The per-group reward
  divides by the group's counted comparisons; the task-macro reward weights the groups present
  (0.5, 1.25, 1, 0.75, 1.25) over the sum of all five weights. Checked against upstream: 596
  reference tables (counts, totals, per-group results and rewards) reproduced exactly from
  upstream's own judge verdicts, and 5,336 leaderboard reward values within rounding.

### WB-Elo (`leaderboard/wb_elo.py`)
- The pool: the published replies of the 63 models above, score-graded exactly as the tested
  model's replies are (`eval_template.score.v2.md`, JSON mode, the same judge cascade), because
  WB-Elo compares scores given by one judge. A pool reply that is empty or ends with
  `... (truncated)` in the score prompt is not sent to the judge: `get_all_votes` skips such a
  reply before it looks at the score, so its score could not change WB-Elo. A pool judge reply
  without a JSON `"score"` is dropped (no vote), as the batch pipeline drops it; it is a
  judge-side harness error, counted in `wb_elo_detail.pool_judge_failures` and
  `judge_failures_dropped.wb_elo_pool` (a cached dropped verdict counts on every run that reuses
  it), sets `status` to `completed_with_errors` (`wb_elo_pool_judge_failures N` in
  `status_reason`) and is named in `leaderboard_comparable_reason`. Verdicts are
  cached as JSON lines under `$GBENCH_CACHE_DIR/wildbench/wb_elo_pool/<key>/<config>.jsonl`
  (default `~/.cache/gbench/...`); `<key>` hashes the judge cascade, the judge config the score
  calls send (temperature 0, JSON mode), the outputs revision and the score template, so another
  judge or judge setting never reuses them, and a verdict is reused only for the same prompt
  (sha256). An interrupted run resumes from the cache. A pool judge outage is not cached; it is
  counted in `wb_elo_detail.pool_judge_outages`, sets `status` to `completed_with_errors` and is
  named in the reason.
- Memory: before generation the pool files are read one model at a time, to check them and to
  keep, per pool reply that can vote, only its model, task, prompt sha256 and length; a model's
  replies are dropped before the next model's file is read. A pool reply's judge prompt is built
  when it is graded: the pool model's file is read again from the Hugging Face cache
  (sha256-checked) and its replies are dropped after the last one is graded, model by model. So
  the run holds at most one pool model's replies while it builds the items and at most two while
  it grades them, never the whole pool's prompts (about 564 million characters) or replies
  (about 191 million). The votes (about 2 million on a full run) are streamed into the Elo
  simulation, which keeps one compact tuple per vote.
- Votes, as `get_all_votes(margin=3, tie_margin=2)`: for every task and every pair of models
  with a score on it, no vote when either reply is empty or ends with `... (truncated)`; the
  higher score wins when it leads by at least 3; a tie when the scores differ by at most 2;
  otherwise no vote. Scores are `int()`-ed and replies stripped (`load_scores`). The tested
  model takes part under the name `gbench:model-under-test` with its WB-Score verdicts; a failed
  request, an outage or a dropped judge reply casts no vote.
- Elo, as `compute_single_round` / `compute_elo_based_on_votes` with the settings of the
  published columns (the `config` of the leaderboard Space's
  `data_dir/wb_elo_results.LM=-1.json` and `LM=1500.json` at revision `4b94d1d8`): K = 4,
  128 rounds, `dynamic=True`, `interval=100`. Every model starts at its LMSYS Arena Elo
  (`leaderboard/data_dir/all_stat.json` `Arena Elo (hard-en) - latest`, 43 models, the same file
  on the Space) or at 1000 without one (the tested model and 24 pool models). Each round
  shuffles the votes; a vote between two models that both have an Arena Elo is skipped; a model
  without one moves by K * (score - expected); a model with one moves too and is pulled halfway
  back to its Arena Elo on its 1st, 101st, ... vote. The value is the mean over the rounds.
  `WB_Elo` (`WB-Elo (Raw)`) has no length rule; `WB_Elo_LC` (`WB-Elo (LC)`) counts a vote as a
  tie when the two replies differ by more than 1,500 characters.
- Seed: upstream's `compute_elo_based_on_votes` runs its rounds in a forked
  `ProcessPoolExecutor` worker (even with `num_processes=1`; the published runs used 4), and
  Python reseeds `random` from OS entropy in a forked child, so the `random.seed(42)` of
  `compute_wb_elo` does not fix the shuffles and no two upstream runs give the same rounds.
  gbench runs the 128 rounds in one process after seeding with 42 (`compute_wb_elo`'s default
  seed; a gbench choice that makes a run deterministic), with the scores entered in a fixed
  order (pool models in sorted name order, then the tested model; tasks in split order), where
  upstream's order follows `os.listdir` and the batch-result files. Checked against upstream: fed upstream's own 63 GPT-4o score files
  (`eval_results/v2.0625/score.v2/eval=gpt-4o-2024-05-13/` at `d6b8dcaf377d`), this code gives
  all 67 models of the Space's published tables a value inside the published 95% interval,
  within 1.6 Elo of the published `WB-Elo (Raw)` (mean 0.5) and 1.5 of `WB-Elo (LC)` (mean
  0.4), rank correlation 0.9995 or more. The simulation takes about 10 minutes of CPU on a full
  run (about 2 million votes).

## Headline metric and how to read the result
- Per task: `wb_score_raw` (the judge's 1-10) and `wb_score` = (raw - 5) * 2, upstream's adjusted
  score (-8 .. +10); `wb_pairwise.<ref>` holds each comparison (`reference_model`,
  `target_side`, `judge_grade`, `judge_model`, `judge_reply` and, when read, the `outcome`:
  choice, winner, extent, reply lengths and truncated / empty flags).
- `accuracy` (headline) = `wb_score_task_macro` = **WB_score.task_macro**, the leaderboard's
  score column (`WB_score.task_macro`, shown as "Score" on the leaderboard Space), computed as
  `_create_tables.py` does: each task belongs to the task groups of its `primary_tag` and
  `secondary_tags` (`task_group_new`: Creative Tasks, Planning & Reasoning, Math & Data
  Analysis, Information/Advice seeking, Coding & Debugging; de-duplicated per task), each
  group's mean raw score is adjusted to (mean - 5) * 2, and the groups are averaged with the
  upstream weights (0.5, 1.25, 1, 0.75, 1.25 in that order) divided by the sum of all five
  weights, times 10. As upstream, a reply that is empty or ends with `... (truncated)` stays out
  of the group means; a request that got no reply from the served endpoint (a harness error), a
  judge outage, refusal or grading failure has no score.
  `wb_score_task_groups` holds each group's value (`WB_score.<group>`). On an `--eval-limit` run
  a group can be missing, and the macro still divides by all five weights, as upstream does.
  `accuracy` is null when no reply entered a group. The leaderboard Space sorts its main table
  by `WB-Elo (LC)` by default; that column is `wb_elo_lc` below.
- `wb_score` = **WB_score = mean(wb_score) * 10 = (mean raw - 5) * 20**, range -80 .. +100.
  This is upstream's `adjusted_score * 10` (`_create_tables.py` score mode, `_merge_results.py`),
  the `WB_score` column. Every scored reply counts, a truncated one included; a request that got
  no reply from the served endpoint (a harness error), a judge outage, refusal or grading failure
  is excluded. `scored_samples` is how many tasks entered this mean. When every task is a harness
  error, judge outage, refusal or grading failure, the run is `failed` and its score withheld
  (null with `accuracy_withheld_reason`, never 0).
- `wb_length` = the leaderboard's `Length` column: mean characters of the scored replies that
  are neither empty nor end with `... (truncated)`.
- **WB-Reward** at K = -1 (no length penalty, the leaderboard's default):
  `wb_reward_task_macro` = `task_macro_reward.K=-1` (the leaderboard's reward column, the mean of
  the three references' task-macro rewards), `wb_reward_mix` = `mixture_of_rewards.K=-1` (the
  mean of the three rewards), `wb_reward_by_reference` = `gpt4t_reward.K=-1`,
  `haiku_reward.K=-1`, `llama_reward.K=-1`, and `wb_reward_score_avg` = `RewardScore_Avg` (the
  leaderboard's hybrid column, (WB_score.task_macro + task_macro_reward) / 2). All rewards are
  x100 (range -100 .. +100). `wb_reward` holds, for every K of `_merge_results.py` (-1, 100,
  300, 500, 1000, 1500, 2000, 3000), the `columns` under upstream's names (`<ref>_reward.K=`,
  `<ref>_reward.<group>.K=`, `mixture_of_rewards.<group>.K=`, `<ref>_reward.task_macro.K=`,
  `mixture_of_rewards.K=`, `task_macro_reward.K=`, `RewardScore_Avg`) and each reference's
  `_create_tables.py` row (`references`: win_much, win, tie, lose, lose_much, total, avg_len,
  per-group results and rewards, reward, task_macro_reward). `wb_reward_references` names the
  pinned reference files.
- **WB-Elo**: `wb_elo` = `WB_Elo` (`WB-Elo (Raw)`) and `wb_elo_lc` = `WB_Elo_LC`
  (`WB-Elo (LC)`), the tested model's mean Elo over the 128 rounds (null when it cast no vote).
  `wb_elo_detail` holds, per column, the mean, std, median and 2.5/97.5 percentile interval over
  the rounds, the vote counts (`votes`, `non_tie_votes`, `target_votes`), the settings, the pool
  size, how many pool replies were graded and how many verdicts came from the cache, the pool
  judge failures, outages and answering models, and the cache directory. `wb_elo_tables` holds
  both columns for every model of the simulation (the 63 pool models graded by the gbench judge,
  the 4 Arena-only models and `gbench:model-under-test`), sorted as the leaderboard sorts them.
- `pass_rate_at_7` (secondary): the share of scored tasks rated 7 or more.
- The per-category table (one row per `primary_tag`) shows each category's **WB_score** (mean
  adjusted score x 10 over its tasks, the headline's scale, with WB_score's rules: a request that
  got no reply, a judge outage, refusal or grading failure left out); the category's pass@7
  share is kept in the cell as `pass_rate_at_7`. The leaderboard's own per-group values are the
  five `WB_score.<group>` in `wb_score_task_groups`.
- `dataset` names the pinned revision. `judge` is `gbench-gemini-cascade`, `judge_models` counts
  the answered WB-Score judge calls per Gemini model (one call per graded task),
  `wb_reward_judge_models` the answered pairwise calls, and `judge_cascade` lists the configured
  cascade order; each trace keeps `judge_reply` and `judge_model`.
- `official_sampling` records WildBench's settings (temperature 0, top_p 1.0, `max_tokens` 4096),
  the temperature the run used and where it came from, `temperature_matches_official` and
  `max_tokens_applied` (false with `--eval-thinking`, with `max_tokens_sent`, the budget sent).
  `max_output_tokens_note` says which budget each request carried and why;
  `official_max_tokens_sent` and `replies_at_official_max_tokens` count the requests sent the
  4,096 cap and the replies that stopped there.
- `judge_sampling` records the judge call: `src/eval.py`'s reference, temperature 0, JSON mode
  and `max_output_tokens` 9216: upstream's 1024 for the verdict plus 8192 for the Gemini judge's
  thinking (with `max_output_tokens_note`).
- **Judge-side failures are harness errors**, never the tested model's outcome. A judge outage
  (every model of the cascade failed, `JUDGE_OUTAGE`) and a judge refusal (the Gemini API blocked
  the prompt for every model, `JUDGE_REFUSED`, with the block reasons in `judge_block_reasons`;
  a property of the task's text, so a rerun would not recover it) are left out of WB_score,
  WB_score.task_macro, WB-Reward and WB-Elo. They are counted per mode: `judge_outages` (WB-Score),
  `wb_reward_judge_outages`, `wb_elo_detail.pool_judge_outages`, and `judge_refusals`
  (`wb_score`, `wb_reward`, `wb_elo_pool`, the refused `tasks` and the `block_reasons`), with
  `wb_reward_judge_refusals` and `wb_elo_detail.pool_judge_refusals` /
  `pool_judge_refused_sessions`. Any of them makes the run `completed_with_errors`
  (`status_reason` names them). Refusals are not cached with the pool verdicts. Telling a
  refusal from an outage needs the cascade to report the Gemini API's block reasons (`refused`
  and `block_reasons` in its call information); where it does not, a refusal is counted as an
  outage.
- `judge_replies_at_max_output_tokens` counts judge replies (WB-Score, WB-Reward and pool) that
  stopped at the 9216; `judge_outages_after_max_output_tokens` counts judge outages where every
  answer stopped there with no text (the thinking took the whole budget).
- `leaderboard_comparable` is **False**. `leaderboard_comparable_reason` says why: the judge is
  gbench's Gemini cascade rather than WildBench's `gpt-4o-2024-05-13` (WB-Score) and
  `gpt-4-turbo-2024-04-09` (WB-Reward) (a gbench convention), and WB-Elo's pool re-graded by the
  same cascade; it also names an `--eval-limit` subset or a `GBENCH_SHARD` shard (the pool is
  then graded on the same tasks), the task groups absent from the scored tasks
  (`wb_task_groups_absent`: upstream's `WB_score.task_macro` and `task_macro_reward` divide by all
  five group weights, so a subset without a group is capped at the present groups' share of the
  full scale), WB-Score replies not graded because of a judge outage or a
  judge refusal (the headline then covers only the tasks the judge graded), WB-Reward
  comparisons and pool replies lost to a judge outage or refusal, the task ids the judge refused,
  judge replies stopped at `max_output_tokens` 9216, a temperature other than 0, and replies cut
  off by the run's token budget rather than at an official cap (`replies_cut_off_by_budget`;
  `replies_cut_off_in_reasoning` counts those that ran out inside the reasoning channel with no
  answer).
- The judge call follows `src/eval.py`'s batch request (`temperature` 0, `response_format`
  JSON, `max_tokens` 1024, no run script overrides it) with the Gemini cascade as the model. The
  Gemini judge counts its thinking against `max_output_tokens` (gpt-4o and gpt-4-turbo do not
  think): at 1024 it stopped mid-verdict (on the 2026-10-08 smoke run 55 judge replies stopped
  there, across WB-Reward and the WB-Elo pool, and 54 pool verdicts were dropped). So **`max_output_tokens` 9216** is sent: upstream's 1024 for the verdict
  plus an 8192-token allowance for the thinking (`JUDGE_THINKING_ALLOWANCE`, which is
  `gemini_budget.GEMINI_THINKING_ALLOWANCE`, the one allowance every Gemini role written for a
  non-thinking model gets). A reply that still
  stops there is read as it arrived, dropped when it
  is not a readable verdict (as the batch pipeline drops an unreadable reply: a WB-Score grading
  failure, a WB-Reward `JUDGE_FAILED`, a pool reply without a score), and a call whose every
  answer stopped there with no text is a judge outage. The pool verdict cache is keyed by the
  judge config, so verdicts graded under another config are not reused.

## Requirements
- An OpenAI-compatible `/v1` chat endpoint serving the model under test.
- **`GEMINI_API_KEY`** or **`GEMINI_API_KEYS`** (one key, or a comma/semicolon list;
  `GEMINI_API_KEYS` wins when both are set) for the judge. The suite checks it with a live call
  before generating and **hard-errors** (`infra_required`) when it is unset or rejected; there is
  no keyword-overlap or other heuristic fallback. Network access to the Gemini API
  (`generativelanguage.googleapis.com`) is needed; `GEMINI_OPENAI_BASE_URL` moves only the key
  check. A full run makes 4,096 judge calls for the tested model (1,024 score + 3 x 1,024
  pairwise) plus one per non-empty, untruncated pool reply for WB-Elo (64,269 of the 63 x 1,024 =
  64,512 pool replies) the first time a judge cascade is used; later runs with the same cascade
  read the pool verdicts from the cache.
- Packages: the base install (`datasets`, `huggingface_hub`, `pyarrow`) plus `google-genai` (the
  `evals` extra: `pip install -e ".[evals]"`).
- Hugging Face Hub access to download `allenai/WildBench` (public, about 4 MB) and 63 files of
  `allenai/WildBench-V2-Model-Outputs` (public, about 231 MB, the three references included);
  `HF_TOKEN` optional.
- Disk: the pool verdict cache under `$GBENCH_CACHE_DIR` (default `~/.cache/gbench`), a few
  hundred MB for a full pool.
- Optional judge knobs (shared by all judged suites): `GBENCH_JUDGE_MODELS`,
  `GBENCH_JUDGE_MODEL`, `GBENCH_JUDGE_CASCADE_ROUNDS`, `GBENCH_JUDGE_BACKOFF`.
- Sampling: `GBENCH_WILDBENCH_TEMPERATURE` (or `--temperature`) overrides the temperature for
  this suite, in both modes; otherwise gbench's rule applies: 1.0 with `--eval-thinking`, 0.0
  without (`official_sampling` records WildBench's 0 and any mismatch). top_p 1.0 is sent in both modes.

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals wildbench --max-output-tokens 16384
```
Add `--eval-limit 20` for a quick subset stratified by `primary_tag` (flagged in the reason) and
`--eval-thinking` to measure the thinking mode (give it a larger `--max-output-tokens`, e.g.
32768). `--max-output-tokens` is required by the CLI; it is sent as every request's `max_tokens`
only with `--eval-thinking` (without it, each request carries WildBench's 4,096).

The run logs its planned workload in one line at its start (for example `[wildbench] --eval-limit
5: 5 sample(s) x 1 attempt = 5 generations`) and records it in the result as `eval_limit_plan`
(`limit`, `samples`, `per_setting`, `attempts_per_sample`, `generations`). The
suite is scored in a batch and runs 1 attempt per sample whatever `--attempt-count` asks, so the
plan says 1 attempt.
