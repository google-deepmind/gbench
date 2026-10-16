# arc_agi

ARC-AGI-1 (Francois Chollet's Abstraction and Reasoning Corpus): the public 400-task
**evaluation** set of 2D integer-grid puzzles. Each task shows a few input/output grid pairs;
the model must infer the transformation and produce the output grid for each test input.
This is not `allenai/ai2_arc` (a grade-school science multiple-choice set).

## Dataset
- The official ARC-AGI-1 data: `data/evaluation/*.json` of
  [fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI) at commit
  `399030444e0ab0cc8b4e199870fb20b863846f34` (the repository arc-agi-benchmarking's README
  clones into `data/arc-agi`): 400 tasks with 419 test pairs (19 tasks have two or three test
  inputs), read in task-id order. This includes upstream's 2024 corrections to nine tasks
  (for example the test outputs of `310f3251` and `79fb03f4`), which older mirrors lack.
- gbench downloads the commit's archive
  (`https://codeload.github.com/fchollet/ARC-AGI/tar.gz/<commit>`, about 300 KB) and checks the
  400 files against the commit's git tree id for `data/evaluation`,
  `4a19f8327d45dda5e1809ee47be08bf1484b493b`, so every file is byte-identical to upstream's. The
  verified files are cached in `$GBENCH_CACHE_DIR/arc_agi/<commit>/evaluation`
  (`~/.cache/gbench/...` by default) and re-checked against the same tree id on every run. A
  failed download or a mismatch stops the suite with an INFRA REQUIRED error; there is no other
  source. To run offline, put the 400 files there yourself.
- `--shard I/N` splits the set by whole tasks (shard I of N over the task-id order), before
  `--eval-limit`, so all test pairs of a task are scored in the same shard and each shard's
  per-task score is exact. The result records `shard`.
- `--eval-limit N` draws N whole tasks (all of a task's test pairs are kept), stratified with a
  fixed seed rather than taking the first N. A task is one sample; its test pairs and its 2
  attempts are not more samples. The run logs its plan on its first line and records it as `eval_limit_plan` (tasks, test pairs, attempts, generations, `retry_generations_at_most` for ARCTester's retries and `retry_generations`, the retries that ran).

## Protocol
- Prompt: `prompts/system_prompt.txt` from
  [arcprize/arc-agi-benchmarking](https://github.com/arcprize/arc-agi-benchmarking) (commit
  `28e67d54`), verbatim, filled exactly as `prompt_manager.convert_task_pairs_to_prompt` does
  (`--Example i--` blocks with JSON grids). One user message per test pair.
- Two attempts per test pair (`--attempt-count` below 2 is raised to 2; a higher value is
  honoured but makes the run non-comparable).
- Answer: read the way upstream builds its `Attempt` (`schemas.py`). The reply goes through
  upstream's own parser, `utils/parsing.py` (`extract_from_boxed`, `backscan_json_parser` and
  `parse_and_validate_json`) at the same commit, ported verbatim: the first `\boxed{...}` that
  holds a JSON list of lists, else the JSON list of lists that closes at the reply's last
  bracket. Thinking blocks are stripped first. The parse is then validated as
  `Union[str, List[List[int]]]` by pydantic in its default (lax) mode, as the `Attempt` model
  does, so integer-valued floats, bools and digit strings (`1.0`, `true`, `"1"`) become ints.
- Retries: when building the `Attempt` fails (the request got no reply, the parser found no
  list of lists, or pydantic rejected a cell such as `1.5` or `"a"`), ARCTester tries the
  attempt again, up to `--retry_attempts 2` tries in all (`main.py`); gbench sends such an
  attempt once more in the same way. An attempt still without an answer after its tries
  (nothing parseable, an empty reply, or cells pydantic rejects) is a null attempt and is scored
  wrong, as upstream scores it; an attempt whose last try got no reply from the served endpoint
  is a harness error, left out (see below). main.py's fallback to the provider's
  JSON extractor runs only when the answer is still a string, which the `Attempt` validator
  never leaves, so it never runs upstream and is not ported. An empty parse (`[]`, `[[]]`) is a
  valid, wrong answer in both harnesses.
- Scoring: exact grid match (`attempt.answer == test_pairs[pair_index].output`), as in
  upstream `main.py` and `score_task`.

## Headline metric
`accuracy` is **pass@2 averaged per task**, the official ARC-AGI score: a test pair is solved
when either attempt reproduces the gold grid, a task scores the fraction of its test pairs
solved (`score_task`: `task_score / num_pairs`), and the headline is the mean over tasks
(`score_submission`: `total_score / total_tasks`). So `total_questions` is the number of tasks
and `correct_answers` is the (possibly fractional) sum of task scores.

An attempt whose last try got no reply from the served endpoint (failed, timed out, or over the
context window) is a harness error, never a null attempt: it is left out (a pair is scored on
its other attempt, a pair or task with no scored attempt is not counted, and `effective_n` is
the number of tasks scored), counted in `scoring_excluded` and `request_harness_errors`, and
named in `status_reason`. When no task is scored, `accuracy` is null with
`accuracy_withheld_reason`. An empty last reply is the model's: a null attempt, scored wrong. A
last reply that stopped with its answer inside the thinking channel is a null attempt too, scored
wrong, but counted in `answer_in_thinking` (with `answer_in_thinking_of`, the attempts, and
`answer_in_thinking_rate`), not in `empty_responses`, and it does not change `status` (see
[answer_in_thinking.md](answer_in_thinking.md)).

Also reported:
- `arc_agi_report`: `tasks`, `test_pairs`, `pairs_solved`, `task_score_sum`, `score`,
  `unparsed_attempts` (null attempts: no valid answer after both tries), `retried_attempts`
  (tries repeated), `recovered_by_retry` (attempts whose second try produced an answer) and
  `tries_per_attempt` (2). A retried trace records `tries` and its `earlier_try`.
- `pass_at_k_per_pair` and `avg_at_k_accuracy`: pass@2 and the per-attempt mean over the 419
  pairs from the first tries, unweighted by task (for reference only).
- `attempts`: base's @k report (per pair, first tries).

`leaderboard_comparable` is true for a full run (all 400 tasks) with exactly 2 attempts, no
temperature sent (gbench always sends one, see Sampling, so its runs are not), and in which every
attempt got a reply from the server and none was cut off at the max_tokens budget.
Null attempts whose complete replies held no valid grid are part of the upstream protocol and
do not change the flag. Otherwise
`leaderboard_comparable_reason` names each cause: an `--eval-limit` subset, a different attempt
count, a temperature sent (every run: 0.0 without `--eval-thinking`, source `default`, 1.0
with it, source `thinking`, or an operator-set one), attempts whose last try got no reply (request failed, timed
out or did not fit the context window: harness errors, left out of the score), attempts whose
last try came back empty, or attempts whose last try was cut off at the max_tokens budget.

An attempt counts as cut off when the server stopped it at the budget (`finish_reason`
`length`), whatever label gbench's reply health gives it: `truncated`, `looping` (repeating
itself) or `non_convergent` (a budget hit far longer than the run's typical reply, relabelled
after the run). Cut-off attempts are scored on what arrived and counted in `attempts_cut_off`,
and the reason says how many had no answer text and how many were looping or non-convergent. A
larger `--max-output-tokens` lets a reply that was still converging finish; a looping or
non-convergent one may not finish at any budget. `status` is `completed_with_errors` when an
attempt got no reply or came back empty on its last try or more than 10% of the attempts were
cut off (`status_reason` names each), and `failed` when no task is scored.

## Sampling
arc-agi-benchmarking sends a model only the fields its entry in `models.yml` sets
(`schemas.ModelConfig` collects them into the API kwargs; `adapters/openai_base.py` passes them
on), and most entries set no temperature, so upstream's request carries none. gbench always
sends one, by its rule for every suite (`base.resolve_temperature`): **0.0** without
`--eval-thinking`, **1.0** with it; so its requests carry a field arc-agi-benchmarking does not
send and the run is not comparable. The result records upstream's setting under
`official_sampling`. `--temperature` or `GBENCH_ARC_AGI_TEMPERATURE` (which takes precedence)
overrides the temperature in either mode. `--max-output-tokens` must fit a full grid; with
`--eval-thinking` it also has to cover the reasoning.

## Requirements
- A running OpenAI-compatible `/v1` chat endpoint. No judge, no sandbox, no API key.
- `pydantic` 2 (a gbench base dependency) for the `Attempt` validation.
- Network access to `codeload.github.com` on the first run (the verified data is cached
  afterwards), or the 400 files of `data/evaluation` at the pinned commit placed in the cache
  directory above.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals arc_agi --eval-thinking \
       --max-output-tokens 65536
```
Add `--eval-limit 20` for a quick 20-task check (not leaderboard-comparable).
