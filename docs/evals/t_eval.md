# t_eval

Canonical T-Eval (Chen et al., ACL 2024; arXiv:2312.14033;
[open-compass/T-Eval](https://github.com/open-compass/T-Eval); HF `lovesnowbest/T-Eval`):
six separable tool-use sub-skills, scored deterministically (no LLM judge).

## Pinned sources

- Data: `lovesnowbest/T-Eval` at revision `af355ab2b62cdbae4262ac41c7529ffeae395012`
  (`data/<file>.json` and `data/<file>_zh.json`, the v2 files).
- Evaluators: `open-compass/T-Eval` at commit `58f22406404d7e2a4f36856a19c7f4dc28a0a5f0`; the
  vendored copies in `_teval_vendor/` differ from it only in import shims (no `mmengine`,
  relative imports, lazy `sentence_transformers` import) and two dropped debug prints.
- Sentence embedders (test.py's `bert_score_model`, chosen per setting: `if "_zh" in
  args.dataset_path: bert_score_model = "thenlper/gte-large-zh"`, else `"all-mpnet-base-v2"`):
  `sentence-transformers/all-mpnet-base-v2` at revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`
  for the English files and `thenlper/gte-large-zh` at revision
  `64c364e579de308104a9b2c170ca009502f4f545` for the Chinese files (both the repos' HEAD, pinned;
  `all-mpnet-base-v2` is the name sentence-transformers resolves to the first repo).

## What runs

Both settings T-Eval publishes, each on its own leaderboard (EN and ZH): the **8 English data
files** `instruct_v2`, `plan_str_v2`, `plan_json_v2`, `reason_str_v2`, `retrieve_str_v2`,
`understand_str_v2`, `reason_retrieve_understand_json_v2`, `review_str_v2` (the eight evaluations
`test_all_en.sh` runs, on the full files the README points to on Hugging Face; the script itself
defaults to the 1/5 `_subset` files distributed for faster inference), and the same 8 with a
`_zh` suffix, whose questions and answer formats are Chinese (`test_all_zh.sh`; the same
evaluators, with the `thenlper/gte-large-zh` embedder test.py uses for `_zh` files). Each sample is a pre-baked context (`origin_prompt`): every turn goes out as its own
chat message, `function` turns with the `user` role (the `api_role` upstream's meta templates give
them) and consecutive turns of one role kept apart, as upstream's templates render them. The
model emits one completion, which goes through test.py's `split_special_tokens` before scoring:
it is cut at the first leaked chat-template marker (`<eoa>`, `<|im_end|>`, `\nuser`,
`\nassistant`, `[INST]` and the like), and a leading `` ```json `` and surrounding backticks are
stripped.

## Scoring (canonical `convert_results`)

The upstream evaluators are vendored verbatim in
`gbench/third_party/_teval_vendor/` and driven per sample; the six dimensions combine
exactly as upstream:

- **Instruct** = mean of the json and string groups, each `(format + args_em)/2` (groups weighted equally)
- **Plan** = mean(plan_str F1, plan_json F1), bertscore graph match (Hungarian) + LIS
- **Reason** = mean(reason_str thought, rru_json thought), sentence-embedding cosine
- **Retrieve** = mean(retrieve_str name, rru_json name)
- **Understand** = mean(understand_str args, rru_json args)
- **Review** = review_str review_quality
- **Overall** = mean of the six, reported as `accuracy`; per-dimension scores in `dimension_scores`.

Plan and Reason embed with the setting's model (English `all-mpnet-base-v2`, Chinese
`thenlper/gte-large-zh`), one instance per setting shared by its scorers, as test.py passes
`bert_score_model` to `PlanningEvaluator` and `ReasonRetrieveUnderstandEvaluator`.

## Reading the result

`accuracy` is T-Eval's English Overall on the 0-100 scale (compare it with the EN leaderboard);
`dimension_scores` holds its six dimensions and `t_eval_file_metrics` each English file's raw
evaluator output. `overall_zh`, `dimension_scores_zh` and `t_eval_file_metrics_zh` are the same
for the Chinese setting (compare with the ZH leaderboard). `dataset_revision` records the data and
evaluator pins above, and `embedding_models` the embedder of each setting (`repo@revision`).

- The 16 files run as 16 generation passes, and the result covers all of them: `total_questions`,
  `failed_requests`, `empty_responses`, `answer_in_thinking` (replies with the answer inside the
  thinking channel, the model's outcome; its `answer_in_thinking_rate` is over every request, see
  [answer_in_thinking.md](answer_in_thinking.md)), `truncated_responses`, `over_context_prompts`,
  `timed_out_requests`, `discarded_outputs`, `scoring_excluded`, `request_harness_errors` and
  `response_health` are summed over the 16 files, `file_status` gives each file's own status
  (Chinese files under `<dimension>_zh`), and `status` applies gbench's usual rule to the summed
  counts (any failed request makes it `completed_with_errors`; nothing scored makes it
  `failed`). `status_reason` lists each file's own reason, by file.
- A request that got no reply from the served endpoint (failed, timed out, or over the context
  window) is a harness error, never the model's prediction: the evaluators never see it, so it
  is left out of its file's metrics, and it is counted (above) and named in `status_reason`. A
  file with no scored reply has no metrics (`null` in `t_eval_file_metrics`), and neither has
  its setting's Overall: `accuracy` (or `overall_zh`) is then null, with
  `accuracy_withheld_reason` (or `overall_zh_withheld_reason`) naming the files and why: a file
  with no sample in this shard or `--eval-limit`, or a file whose every request got no reply
  (harness errors).
- `category_accuracy` is empty: T-Eval's metrics are continuous scores, not right/wrong verdicts,
  so there is no per-category correct/total; the six dimensions are in `dimension_scores`.
  `correct_answers` is each setting's Overall times its scored sample count, summed and rounded,
  so pillar subtotals weigh the suite by its size.
- `leaderboard_comparable` is `true` for a full run at temperature 0.0 with one attempt, no
  decoding penalties and no reply cut off (`replies_cut_off`, see Budget); otherwise
  `leaderboard_comparable_reason` lists the `--eval-limit` subset, the shard, the temperature, the
  penalties, the attempts, the requests that got no reply (harness errors) or the cut replies. `official_sampling` records test.py's setting
  (greedy, max_new_tokens 512), the temperature this run used, whether it matched and whether
  the 512 cap was applied.
- Inline reasoning (`<thought>...</thought>` and the like) is removed from each reply before the
  evaluators read it, as every gbench suite scores replies. An empty reply is scored as an empty
  prediction by the evaluators; a request with no reply is a harness error and is not scored
  (above).

## Requirements

- An OpenAI-compatible `/v1` endpoint (single-turn generation).
- **`sentence-transformers`** + **`torch`** + **`networkx`**: required by the Plan/Reason/
  Retrieve/Understand scorers (BERTScore + the plan Hungarian matching; the vendored planning and
  reason/retrieve/understand evaluators import `networkx` when they load). `torch` comes with your
  vLLM serving stack (`pip install vllm`, or your validated CUDA build); install the other two into
  the same environment gbench runs in:
  ```bash
  pip install sentence-transformers networkx        # torch: comes with `pip install vllm`, or your CUDA build
  ```
  The vendored evaluators are imported only when `t_eval` runs, never when gbench loads its eval
  suites, so `gbench --list evals` and runs of other suites work without these packages. Before
  its first request `t_eval` checks both (`t_eval.missing_packages`) and hard-errors
  (`infra_required`) naming each one that is missing.
  The Plan/Reason scorers download two embedders from the Hugging Face Hub on first use, at the
  pinned revisions above, into the Hugging Face cache (`HF_HOME`, default `~/.cache/huggingface`):
  **`sentence-transformers/all-mpnet-base-v2`** (about 440 MB) for the English files and
  **`thenlper/gte-large-zh`** (about 650 MB) for the Chinese files. Each loads on the GPU when
  torch sees one (test.py's default device) and falls back to the CPU when the GPU load fails (a
  vLLM server on the same GPU owns its memory), or runs on the CPU when there is no GPU
  (`GBENCH_T_EVAL_DEVICE=cpu|cuda` forces a device). The device can change a score: the plan
  graph matching breaks ties between equal BERTScore similarities on float noise, which differs
  between CPU and GPU kernels (on the pinned data one English Plan reply scores 0.4444 on the GPU
  and 0.6667 on the CPU). The result records each setting's embedder and device
  (`t_eval_embedder`: `model`, `revision`, `device`, `fallback`, `forced`), and a setting whose
  embedder did not run on the GPU is named in `leaderboard_comparable_reason`. If
  `sentence-transformers` or `networkx` is missing, or installed but fails to import (a broken
  torch build, a missing shared library), the suite **hard-errors** (`infra_required`, naming the
  failed import), and so does an embedder that cannot be loaded (the error names it).
  It never skips and never scores on a partial matrix.
- Network and `huggingface_hub` on the first run (the 16 data files at the pinned revision); a
  file that cannot be downloaded stops the suite with an `[INFRA REQUIRED]` error.
- No API key, no LLM judge, no server/sandbox. Scoring is fully deterministic and offline.
- **Sampling:** temperature 0.0 without `--thinking`, 1.0 with it (gbench's rule for every
  suite); test.py generates greedily (`llm.chat(..., do_sample=False)`), recorded under
  `official_sampling`. `--temperature`, or `GBENCH_T_EVAL_TEMPERATURE` for this suite
  (it takes precedence), overrides it and is recorded under `official_sampling`. Scoring itself is
  deterministic regardless.
- **Budget:** test.py runs an open-weight model as a Hugging Face model (`--model_type hf`), built
  as `HFTransformerCasualLM(..., max_new_tokens=512)` (`test.py` line 115 at the pinned commit).
  Without `--thinking` every request carries exactly `max_tokens=512`, whatever
  `--max-output-tokens` says (`max_output_tokens_note` records that the flag was not applied); a
  reply that stops there is scored on what arrived, as upstream scores it, and is counted in
  `replies_at_official_max_tokens`, apart from truncations (it does not make the run
  `completed_with_errors`). With `--thinking` the cap, written for non-reasoning generation, is
  not applied: each request gets the run's `--max-output-tokens` (else the thinking floor), and
  `official_sampling` records `max_tokens: 512`, `max_tokens_applied: false` and the note
  "official cap not applied to reasoning" (`base.resolve_token_budget`). `replies_cut_off` counts
  the replies a budget stopped before they finished other than at the 512 (every one with
  `--thinking`; without it, one a served context window cut below the 512), an empty reply after
  reasoning included; any makes the run not leaderboard-comparable.
- The served chat template must accept two turns of one role in a row (gemma-4's does); a server
  whose template rejects them fails those requests: harness errors, counted in `failed_requests`
  and left out of the score.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals t_eval --max-output-tokens 2048
```
`--eval-limit N` is N rows for the whole suite, never N (or one) per file: the N are dealt over
the 16 files round-robin in a seeded order (one per file before any gets two, sampled within the
file). A file that gets 0 is not run (`file_status` `not_drawn`); a setting missing a dimension
then has no Overall (`accuracy_withheld_reason` / `overall_zh_withheld_reason` name the files with
no sample in the limit). `--eval-limit 16` or more covers every file.

The run logs its planned workload in one line at its start (for example `[t_eval] --eval-limit 5:
5 sample(s) x 1 attempt = 5 generations`, with the rows of each file) and records it in the result
as `eval_limit_plan` (`limit`, `samples`, `per_setting`, `attempts_per_sample`, `generations`).
