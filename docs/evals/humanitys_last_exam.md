# humanitys_last_exam

Humanity's Last Exam (HLE, Center for AI Safety and Scale AI): 2,500 expert-written, closed-ended
questions across dozens of subjects (about 41% math), multiple-choice and short-answer, 342 of
them with an image.

## Dataset
- `cais/hle`, split `test` (2,500 rows), pinned at revision
  `5a81a4c7271a2a2a312b9a690f0c2fde837e4c29`. Its last data change is `021a3d7` (2025-05-20);
  the pinned commit only added `eval.yaml`. The dataset is **gated**: accept its terms on
  Hugging Face with the account whose `HF_TOKEN` gbench uses.
- `cais/hle-rolling` (a continually updated 1,000-question fork) and `cais/hle-diamond` are
  separate datasets; this suite runs the original HLE, which the official harness loads.
- The per-category table uses each row's `category`.

## Protocol (centerforaisafety/hle `hle_eval/`, commit `22ed3074b1e7`)
- **Prompt** (`run_model_predictions.py`): the official `SYSTEM_PROMPT`, verbatim, as a system
  turn (`Explanation: ... / Answer: ... / Confidence: ...`), and the question as the user turn.
  For the image subset the user turn is the question text followed by the image data-URI, in
  upstream's order.
- **Judge** (`run_judge_results.py`): every non-empty reply goes to the official `JUDGE_PROMPT`,
  verbatim, with the question text (no system prompt, no image), the reply and the gold answer;
  the question and the gold answer go in exactly as the dataset stores them (upstream passes
  `question["answer"]` unchanged; 27 of the 2,500 answers carry surrounding whitespace).
  Upstream requests structured output (`ExtractedAnswer`: extracted_final_answer, reasoning,
  correct yes/no, confidence, strict); gbench asks the Gemini judge for the same JSON schema, in
  the same field order. A reply laid out as the prompt's own `field: value` lines is read too. A
  missing confidence is 100, the prompt's own rule. A reply with no yes/no `correct` is a judge
  failure, not the model's answer (`JUDGE_FAILED`, `scoring_excluded: true`): left out of the
  accuracy and the calibration error, counted in `judge_failures` (the run is
  `completed_with_errors`) and named in `leaderboard_comparable_reason`. Upstream prints "Missing
  judge response ..., you should rerun the judge" for such a question and then divides by every
  question, so there it counts as wrong; gbench keeps a judge-side failure out of the model's
  score.
- **Judge call parameters**: those of `extract_answer` (`client.beta.chat.completions.parse(...,
  max_completion_tokens=4096, messages=[user prompt], response_format=ExtractedAnswer)`): the
  prompt as the only message, the schema as structured output, `max_output_tokens` 4096 and **no
  temperature** (upstream sends none). o3-mini counts its reasoning against
  `max_completion_tokens`, as Gemini counts its thinking against `max_output_tokens`, so the
  limit means the same thing. The result records these under `judge_sampling`.
- Correct = upstream's `"yes" in correct`. There is no deterministic pre-pass: the judge alone
  decides, as upstream.
- A request that got no reply from the served endpoint is a harness error: not judged, marked
  `not_judged_failed_request`, counted in `failed_requests`, left out of the score and named in
  `status_reason` (the run is `completed_with_errors`). An empty reply the server returned is the
  model's outcome: not judged (upstream has no prediction for it), marked
  `not_judged_empty_response`, counted in `empty_responses`, and it scores wrong.

## Headline metric and how to read the result
- `accuracy` (headline) = judge `correct: yes` / graded questions, in percent, with
  `accuracy_ci95_half_width` (upstream's Wald 95% half-width, `1.96 * sqrt(acc * (100 - acc) / n)`).
- `calibration_error` = upstream's `100 * round(calib_err(confidence, correct, p='2', beta=100), 2)`,
  ported verbatim, over the judged samples, with the judge-extracted confidence / 100.
  `calibration_samples` is how many fed it. The official estimator sorts by confidence, cuts bins
  of 100 (the last bin takes the remainder) and, as written upstream, never adds the last bin;
  kept so the number matches the official script. Below 200 judged samples there is no bin left
  to measure, so the value is `null` and `calibration_note` says why (upstream would report 0 or
  raise).
- A judge-cascade outage is excluded from accuracy and calibration and counted in
  `judge_outages`, like a judge failure above. Upstream divides by all 2,500 questions; gbench
  divides by the graded ones, so a judge-side failure cannot pose as a wrong answer. When nothing
  could be graded the run is `failed` and its score withheld.
- `judge` is `gbench-gemini-cascade`, `judge_models` counts the answered judge calls per Gemini
  model (one call per graded question), `judge_cascade` lists the configured cascade order and
  `judge_sampling` the judge call's settings (reference, no temperature, `max_output_tokens`
  4096, the response schema); each trace keeps `judge_response` (extracted answer, reasoning,
  correct, confidence), `hle_confidence`, `judge_model` (the Gemini model that answered) and
  `judge_reply` (its full reply). Gemini counts its thinking against the 4096, as o3-mini counts
  its reasoning against `max_completion_tokens`: a judge reply stopped before its JSON object
  closed is a judge failure (left out of the score, as above; upstream's parse raises and leaves
  the question unjudged), also counted in `judge_replies_incomplete_json` and marked on the trace
  with `judge_reply_complete_json: false`.
- `leaderboard_comparable` is **False**. `leaderboard_comparable_reason` says why: the judge is
  gbench's Gemini cascade, not the official `o3-mini-2025-01-31` (a gbench convention), and, on
  an `--eval-limit` run, that only a subset was graded, a temperature other than HLE's (a
  temperature other than 0 without `--eval-thinking`; with it, the 1.0 every thinking request is
  sampled at, or any other temperature, since HLE sends none), the judge
  failures left out of the score, and replies cut off by a `max_tokens` that was sent
  (`replies_cut_off_by_budget`; `replies_cut_off_in_reasoning` counts those that ran out inside
  the reasoning channel with no answer; HLE sends none, so this stays 0).
- A reply that runs to the end of the server's context window (a model that collapses into
  repetition, which the README reports: "we observed some model collapse") is HLE's protocol
  outcome, since HLE asks for "the maximum number of tokens available": it is judged as it
  arrived, counted in `replies_at_context_window` (truncated replies) or `looping_responses`,
  and does not make the run `completed_with_errors` or non-comparable. This holds when the server
  serves the model's native context window; a smaller `--max-model-len` shortens what "the
  maximum number of tokens available" means for the run.
- Sampling: gbench's rule, **temperature 1.0 with `--eval-thinking` and 0.0 without**,
  whatever HLE sends. HLE's own setting is recorded per mode: without `--eval-thinking`,
  temperature 0 (the HLE README states that its evaluation defaults to temperature 0, and
  `hle_eval/run_model_predictions.py` sent its `--temperature 0.0` default to non-reasoning
  models until commit `136e12e`); with `--eval-thinking`, no temperature (before `136e12e` the
  script sent none to reasoning models, `temperature=args.temperature if "o1" not in args.model
  else None`, and since then that line is commented out, so the script sends none to any model).
  For a thinking run `official_sampling` records HLE's setting with
  `temperature_matches_official: false`, and the temperature is named in
  `leaderboard_comparable_reason`.
  In both modes no `max_tokens` of its own: the README says "Our evaluation uses the maximum
  number of tokens available" and `run_model_predictions.py`'s `--max_completion_tokens`
  defaults to None, so each request asks for the rest of the context window (what the server
  grants when `max_tokens` is absent). The result records `official_sampling` (the reference,
  whether a temperature is sent, the temperature used, its source,
  `temperature_matches_official` and `send_max_tokens: false`).

## Requirements
- An OpenAI-compatible `/v1` chat endpoint serving the model under test. It must accept image
  inputs for the 342 image questions (a text-only server rejects those requests, and they count
  as failed).
- **`GEMINI_API_KEY`** or **`GEMINI_API_KEYS`** (one key, or a comma/semicolon list;
  `GEMINI_API_KEYS` wins when both are set) for the judge. The suite checks it with a live call
  before generating and **hard-errors** (`infra_required`) when it is unset or rejected; there is
  no fallback grader. Network access to the Gemini API (`generativelanguage.googleapis.com`) is
  needed; `GEMINI_OPENAI_BASE_URL` moves only the key check.
- **`HF_TOKEN`** for an account that has accepted the `cais/hle` terms (the first download is
  about 275 MB).
- Packages: the base install (`datasets`, `numpy`) plus `google-genai` (the `evals` extra:
  `pip install -e ".[evals]"`).
- Optional judge knobs (shared by all judged suites): `GBENCH_JUDGE_MODELS`,
  `GBENCH_JUDGE_MODEL`, `GBENCH_JUDGE_CASCADE_ROUNDS`, `GBENCH_JUDGE_BACKOFF`.
- Sampling: `GBENCH_HUMANITYS_LAST_EXAM_TEMPERATURE` (or `--temperature`) overrides the
  temperature for this suite, in both modes; otherwise gbench's rule applies: 1.0 with
  `--eval-thinking`, 0.0 without.

## Run
```bash
export GEMINI_API_KEY="..."
export HF_TOKEN="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals humanitys_last_exam --eval-thinking \
       --max-output-tokens 32768
```
`--eval-thinking` is recommended (frontier reasoning). `--max-output-tokens` is required by the
CLI but not applied here: HLE sends no completion-token cap, so each request asks for the rest of
the context window (`max_output_tokens_note` records the value given). Add `--eval-limit 20` for a
quick stratified subset (flagged in the reason; calibration needs at least 200 samples).
The run logs its planned workload at its start and records it as `eval_limit_plan`.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered questions
(round-robin: positions I, I+N, I+2N, ...), selected before the `--eval-limit` draw, which is then
taken within the shard (N shards run up to N x `--eval-limit` questions). The result records
`shard`, `shard_index` and `shard_count`, and a shard of more than one is not
leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`); `gbench
--merge` combines the shards' results as docs/distributed-gbench.md describes: it re-counts the
union of their traces, and refuses a suite whose headline is not a correct / scored count over
them unless the suite defines its own merge.
