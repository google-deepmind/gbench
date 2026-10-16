# lab_bench

LAB-Bench (Future-House, [arXiv:2407.10362](https://arxiv.org/abs/2407.10362)), the Language
Agent Biology Benchmark: multiple-choice questions on capabilities biology research needs, in
8 categories: literature (`LitQA2`), databases (`DbQA`), supplementary material (`SuppQA`),
figures (`FigQA`), tables (`TableQA`), protocol troubleshooting (`ProtocolQA`), sequence
manipulation (`SeqQA`) and molecular cloning (`CloningScenarios`). gbench runs all 8 with the
repository's zero-shot baseline harness. **Scoring: LAB-Bench's own parse of the reply (an
`[ANSWER]` tag, or a reply made only of letters); accuracy, precision and coverage per
category, and the headline is the mean of the 8 category accuracies.**

## Dataset
- `futurehouse/lab-bench`, split `train` of each category config, pinned to revision
  `5c77cec648430f30611808808861eb86f81d5eaa`: the public split, **1,967 questions** (LitQA2 199,
  DbQA 520, SuppQA 82, FigQA 181 with one figure each, TableQA 244 with one or more table
  images each, ProtocolQA 108, SeqQA 600, CloningScenarios 33).
- This mirror was built by upstream's own `scripts/update_datasets_in_hub.py` from the
  evaluator's instances; its rows equal the repository's `*-public.jsonl` files at commit
  `998a8e0a` in content and order (checked on all 1,967 rows).
- LAB-Bench keeps about 20% of each category as a private test set, which is not public.
- `--eval-limit N` draws N questions stratified over the categories with a fixed seed.
- `--shard I/N` selects shard I of the 1,967 questions (round-robin over the question list,
  which runs category by category in upstream's order) before the `--eval-limit` draw; the
  result records `shard`.

## Protocol
Ported from Future-House/LAB-Bench at commit `998a8e0a` (`scripts/score_baseline.py`, which
runs `labbench.Evaluator` with the zero-shot agent and `use_cot=True`):
- **Question**: each category's `task.py`. ProtocolQA prepends the protocol text directly to
  the question; SuppQA prepends `Paper title: <title>\nDOI: <source>\n`; FigQA sends its figure
  and TableQA its table images after the text in the same user message, as
  `openai_utils.get_completion` does: each image is encoded as `labbench/utils.py` `encode_image`
  encodes it, the PIL image re-saved in its own format with PIL's defaults
  (`image.save(buf, format=image.format or "JPEG")`, labelled `image/<format>`), so a JPEG
  figure is re-encoded at quality 75 and its pixels differ from the dataset file's, as they do
  upstream; the other categories send the question as is.
- **Choices**: `utils.randomize_choices`. The ideal answer (for 5 LitQA2 questions it is the
  literal `null`, as upstream shows it), the refusal choice `Insufficient information to answer
  the question` and the distractors are shuffled and shown as `(A) ...`. Upstream shuffles with
  the unseeded global random generator; gbench seeds the shuffle per question, so a rerun asks
  the same questions.
- **Prompt**: `zero_shot.MCQ_INSTRUCT_TEMPLATE` verbatim, with chembench's `COT_PROMPT`
  (`Think step by step.`), after chembench's `post_process_prompts` (removes `\ce{}`, `$..$`,
  SMILES and `\pu{}` wrappers, keeping their contents). One user message, no system prompt.
- **Scoring**: exactly upstream's `zero_shot.run_task`, on the reply as the server returned it
  (not trimmed): chembench `prepare_mcq_answer(reply, MCQ_REGEX_TEMPLATE_1)` takes the sorted
  set of `[ANSWER]...[/ANSWER]` matches, or, with no tag, a reply made only of letters (`B`,
  `B, D`, `A B`); then `run_regex(create_multiple_choice_regex(offered letters), ...,
  return_first=True)` takes the first offered letter. Any other reply has no answer and is
  wrong, including `Final Answer: C` without a tag and a letter padded with spaces or
  newlines. Where upstream's `run_regex` would raise (a reply with no tag, no comma and no
  space, or an empty reply), gbench records no answer instead of stopping the run.
- A request that gets no reply from the served endpoint is a harness error, never the model's
  answer: one that fails after retries (upstream's tenacity gives up and the run stops), one
  that times out, and a prompt the server rejects as longer than its context window (upstream's
  `UnanswerableError` counts that one as unsure and wrong; the served window is a serving
  setting, so gbench does not score it). Each is left out of every metric below, counted and
  named in `status_reason` and in `leaderboard_comparable_reason`; when every request got none,
  `accuracy` is null with `accuracy_withheld_reason`. An empty reply has no answer and is wrong.

## Headline metric
`accuracy` is the **mean of the 8 category accuracies** (the categories the run covered: the
`metric` names how many were averaged and which were not run, as with an `--eval-limit` subset,
and the reason names them). A category the run asked but none of whose questions is in the score
(every request a harness error) withholds the mean (`accuracy` null with
`accuracy_withheld_reason` naming it; `lab_bench_report.categories_unscored`), rather than a mean
over the rest. LAB-Bench reports its categories separately; `lab_bench_report` has them all:
- `categories.<name>`: `accuracy`, `precision` (correct / answered, where choosing the refusal
  option is not answering), `coverage` (answered / total), `n_total`, `n_correct`, `n_sure`
(a rate with nothing to divide by, such as precision with no answered question, is `null`,
never 0),
  and the same per `subtasks` entry (`Evaluator.compute_metrics`, `metrics_all` and
  `metrics_<subset>`). A reply with no parseable letter counts as answered and wrong, as
  upstream's `sure = agent_output != unsure_choice` has it.
- `mean_category_accuracy` (the headline) and `all_questions` (the three metrics pooled over
  every question).
- `tagged_answers`, `letters_only_answers`, `no_letter` and `chose_refusal` counts.

`leaderboard_comparable` is always **false**: the public split is about 80% of each category,
while published LAB-Bench figures cover the full set, and the headline averages categories that
LAB-Bench reports separately. `leaderboard_comparable_reason` says so and also names requests that
got no reply or came back empty and an `--eval-limit` subset when they apply. Compare the
per-category values in `lab_bench_report` with other runs on the public split.

## Sampling
`scripts/score_baseline.py` builds each agent with `model_kwargs={"model": <name>}` and the
agent sends only those fields (`labbench/openai_utils.py`:
`chat.completions.create(messages=[msg], **self.model_kwargs)`; the Anyscale and Vertex agents
reuse it), so upstream's request carries **no temperature** and **no `max_tokens`**, and the
server's defaults apply (for a vLLM server, the model's `generation_config.json`, 1.0 for Gemma 4,
and the rest of the context window). gbench sends no `max_tokens` either, in both
modes, but always sends a temperature, by its rule: **1.0 with `--eval-thinking`, 0.0
without**. The result records this under `official_sampling` (`send_max_tokens: false`;
`temperature_matches_official: false`, which `leaderboard_comparable_reason` names). Under gbench's token-limit policy
`--max-output-tokens` is not applied (`max_output_tokens_note` says so); each prompt is still
counted, and one that cannot fit the context window is recorded as over-context.
`--temperature` or `GBENCH_LAB_BENCH_TEMPERATURE` replaces that temperature, in both modes,
which `leaderboard_comparable_reason` then names.

## Requirements
- A running OpenAI-compatible `/v1` chat endpoint that **processes images** in chat messages
  (`image_url` content parts): FigQA and TableQA send figures and tables. Before generating,
  the suite sends one of the dataset's images and checks that it adds prompt tokens (at least
  16) compared with the same text alone; if the request fails or the image is ignored, the
  suite stops with an `[INFRA REQUIRED]` error rather than scoring those questions without
  their figures. (With `--eval-limit` the check runs only when an image question is drawn.)
  No judge, no sandbox, no API key.
- The `datasets` package (base install) and network access to the Hugging Face Hub on the
  first run: about 335 MB (FigQA 227 MB and TableQA 106 MB of images). The dataset is public
  but its terms (CC BY-SA 4.0) apply.

## Run
```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals lab_bench --eval-thinking \
       --max-output-tokens 65536
```
`--max-output-tokens` is not applied (no request carries a max_tokens). Add `--eval-limit 40` for
a quick check across the categories.

The run logs its planned workload in one line at its start (for example `[lab_bench] --eval-limit
5: 5 sample(s) x 1 attempt = 5 generations`) and records it in the result as `eval_limit_plan`
(`limit`, `samples`, `per_setting`, `attempts_per_sample`, `generations`).
