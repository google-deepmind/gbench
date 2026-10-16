# amc_aime

A **gbench composite, not a canonical benchmark**: AMC 12 (2022-2023) and AIME (2022-2024)
competition problems from the AI-MO validation sets, prompted and scored by gbench and reported
both blended and per competition. There is no published leaderboard for the blend, and every
problem predates current training cutoffs, so treat the number as a **contaminated reference**,
not a reasoning measurement. For a clean AIME number use the [`aime`](aime.md) suite (MathArena's
AIME 2026, graded with MathArena's own protocol).

## Datasets

| category | dataset | pinned revision | N |
| --- | --- | --- | --- |
| `AMC` | `AI-MO/aimo-validation-amc` (train) | `69d78a4a2c840e82d69af6bc742bda09005f6316` | 83 (AMC 12A/12B 2022-2023; multiple choice converted to numeric answers) |
| `AIME` | `AI-MO/aimo-validation-aime` (train) | `13f9e12f613e720c2a2b2f345dd04b998a29494d` | 90 (AIME I/II 2022-2024) |

Both revisions are pinned in `amc_aime.py`. **Both datasets are required**: if either fails to
load, the suite raises instead of reporting a score over the other competition under the
`amc_aime` name.

## Protocol

- **Prompt (gbench's own):** `Problem: <problem>` followed by "Solve the math competition problem.
  Write your final answer as an integer at the end in the format: 'The final answer is X'."; with
  `--eval-thinking` the instruction says "step by step".
- **Extraction:** the shared `final_number` extractor: the last `\boxed{}` first, then an explicit
  answer anchor ("The final answer is", "Final Answer:"), then the last number in the reply.
- **Scoring:** exact integer match against the gold answer (pass@1; avg@k with `--attempt-count k`).
- `--eval-limit N` takes N problems in total, stratified across the two competitions (never N per
  competition). The run logs its plan on its first line and records it as `eval_limit_plan`.

## Reading the result

| field | meaning |
| --- | --- |
| `accuracy` | blended pass@1 over all AMC + AIME problems (a gbench composite) |
| `accuracy_amc`, `amc_n` | AMC 12 component |
| `accuracy_aime`, `aime_n` | AIME component |
| `dataset` | both dataset ids with their pinned revisions |
| `contamination_note` | the two caveats below, in words |
| `leaderboard_comparable` | always `false` |

`leaderboard_comparable_reason` states why: the AMC+AIME blend has no leaderboard, the prompt is
gbench's, and both sets are pre-cutoff with their AoPS worked solutions online (the rows' own
`url` field), so a high score cannot be told apart from recall. It also names, when they apply,
an `--eval-limit` subset (with the problems scored of the 173), replies cut off by the run's token
budget (`replies_cut_off_by_budget` / `replies_cut_off_in_reasoning`: `--max-output-tokens`, a
max_tokens the context window lowered), and requests that got no reply from the served endpoint
(harness errors, left out of the score). Quote `accuracy_amc` and `accuracy_aime` separately if
you quote anything.

## Requirements

- A running OpenAI-compatible `/v1` endpoint.
- Network access to the Hugging Face Hub on first run (both datasets are public; no token needed).
- No other packages, keys or containers.

## Run

```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals amc_aime --eval-thinking \
       --max-output-tokens 65536
```

`--max-output-tokens` is required on the command line and sets the per-request answer budget
(with `--eval-thinking`, leave room for the reasoning).

`GBENCH_AMC_AIME_TEMPERATURE` overrides the temperature for this suite (else 0.0 greedy, or 1.0
with `--eval-thinking`; `--temperature` sets it for the whole run).

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered problems (AMC
then AIME) (round-robin: positions I, I+N, I+2N, ...), selected before the `--eval-limit` draw,
which is then taken within the shard (N shards run up to N x `--eval-limit` problems). The result
records `shard`, `shard_index` and `shard_count`, and a shard of more than one is not
leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`). `gbench
--merge` combines the shards' results with the suite's own merge (`merge_shards` in
`amc_aime.py`): the generic re-count of the union of their traces (docs/distributed-gbench.md),
with `accuracy_amc` / `amc_n` / `accuracy_aime` / `aime_n` and the comparability reasons
recomputed over the union.
