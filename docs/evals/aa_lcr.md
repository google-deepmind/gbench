# aa_lcr

AA-LCR (Artificial Analysis Long Context Reasoning), version 1.1: 100 hard questions, each
answered over its own document set of about 100k tokens (company reports, industry reports,
government consultations, academia, legal, marketing and survey documents). Answers cannot be
looked up directly; they have to be reasoned from several documents. Open-form answers are
graded by an **LLM equality checker** against the official answer.

## Dataset
- `ArtificialAnalysis/AA-LCR`, pinned to revision `9a77ef56b717057ade24ceab4d273712a0b4f19e`
  (v1.1, 2026-09-04): `AA-LCR_Dataset.csv` (100 questions) plus
  `extracted_text/AA-LCR_extracted-text.zip` (the pre-extracted text of the ~230 documents).
  v1.1 corrected 16 answer keys and added the judge system prompt; the card says v1.0 scores
  are not comparable with v1.1 ones (v1.0 is revision `bdae010`, not used here).
- Every file listed in a question's `data_source_filenames` is loaded, in that order, from
  `lcr/<document_category>/<document_set_id>/<filename>` in the archive (as the card's
  `get_document_set` does). The archive stores non-ASCII names as UTF-8 without the zip UTF-8
  flag, and some names decomposed (NFD); gbench re-decodes and NFC-normalizes them so they
  match the CSV. A listed document that is still missing is a **hard error**, never a question
  with part of its evidence silently dropped.
- `--eval-limit N` draws a subset stratified over `document_category`.
  The run logs its planned workload at its start and records it as `eval_limit_plan`.

## Protocol
From the dataset card at the pinned revision, verbatim:
- **Prompt** ("Prompt Template"): `BEGIN INPUT DOCUMENTS`, each document as
  `BEGIN DOCUMENT {i}:` / text / `END DOCUMENT {i}` joined by blank lines, `END INPUT DOCUMENTS`,
  then `Answer the following question using the input documents provided above.` and the
  question between `START QUESTION` and `END QUESTION`. One user message, no system prompt.
- **Equality checker** ("Scoring Approach", v1.1): the card's system prompt (number
  equivalence, required formats, ordered lists, commit-to-one-answer) and user prompt, with the
  raw question, the official answer and the model's answer (reasoning-channel text excluded).
  The user prompt asks for JSON with a verdict of CORRECT or INCORRECT; gbench reads the
  `verdict` field of the JSON object in the reply (inside a code fence or not). A reply with no
  JSON object that still gives exactly one whole-word CORRECT or INCORRECT (a bare word, a JSON
  string such as `"INCORRECT"`, or `Verdict: INCORRECT`, markup stripped) is read too, not asked
  again; a reply naming both (a format echo such as "CORRECT or INCORRECT") is not. Those reads
  are marked `judge_verdict_relaxed_parse` on the attempt, counted in
  `judge_verdicts_read_by_relaxed_parse` and named in `leaderboard_comparable_reason`. The call sends the
  two prompts and nothing the card does not publish: no JSON mode and no temperature (the official
  checker, GPT-5.6 Luna (medium), is a reasoning model, which takes no temperature; its reasoning
  effort belongs to that model, and the Gemini cascade runs at its own default). The result
  records this under `judge_sampling`. Every judged attempt keeps the checker's replies with
  the Gemini model that gave each (`judge_replies`, up to 4,000 characters each) and the model of
  the last one (`judge_model`), so each verdict can be re-read. A reply with no usable verdict is
  asked again, up to 3 calls in total; if none gives one the attempt is marked
  `judge_grade: JUDGE_FAILED`, a harness error (below), never an incorrect answer. A judge outage
  (every model of the cascade failed) is `JUDGE_OUTAGE` and is excluded from the denominator.
  An empty response the server did return is the model's outcome: incorrect without a judge call,
  and it stays in the denominator.
- **Harness errors.** An attempt whose request produced no reply (base's `FAILED`, `TIMEOUT` or
  `OVER_CONTEXT`: an endpoint error, a timeout, or a prompt longer than the served context
  window) is `judge_grade: HARNESS_ERROR`, and one whose checker gave no verdict is
  `JUDGE_FAILED`. Neither is the model's outcome: both are left out of the accuracy (numerator and
  denominator, `scoring_excluded`), counted in `harness_errors` (`endpoint_failure`,
  `endpoint_timeout`, `over_context`, `judge_no_verdict`) and named in
  `leaderboard_comparable_reason`, and the run's `status` is `completed_with_errors`. When no
  attempt is scored (all harness errors or judge outages) the score is withheld (`accuracy:
  null`, `status: failed`, `accuracy_withheld_reason`). Artificial Analysis retries a failed API
  call up to 30 times and does not publish a result that persistent API failures affected; gbench
  does not score such attempts either.
- **Repeats**: Artificial Analysis runs 3 repeats and reports pass@1 aggregated across them, so
  the suite runs at least 3 attempts per question (a larger `--attempt-count` is honoured).
- **Sampling**: Artificial Analysis's General Testing Parameters (Intelligence Benchmarking
  methodology) set temperature 0 for non-reasoning models and, for reasoning models, 0.6 unless
  the model lab recommends another. gbench sends the temperature of its rule, **1.0 with
  `--eval-thinking`, 0.0 without**, and records the official value: 0 without
  `--eval-thinking`; with it, the lab's recommendation: the temperature the served model's
  shipped `generation_config.json` sets (read from `--tokenizer`, else the served model id: a
  local directory or a Hugging Face id through the HF cache), else AA's 0.6. A config that cannot
  be read stops the suite with `infra_required` before generation (the official value would
  otherwise be a guess). For Gemma 4, whose `generation_config.json` sets 1.0, that matches the official value, and for
  a model whose official value differs the mismatch is recorded. `--temperature` or
  `GBENCH_AA_LCR_TEMPERATURE` override it in both modes; the result
  records `official_sampling` (the reference, the temperature used, its source and
  `temperature_matches_official`), and a mismatch is named in `leaderboard_comparable_reason`.
- **Maximum output tokens** (the same General Testing Parameters): 16,384 tokens for
  non-reasoning models, adjusted downward where the context window is smaller; for reasoning
  models the maximum the model allows. Without `--eval-thinking` every request carries
  `max_tokens` 16,384, shrunk to what the window leaves after the prompt (`max_tokens_sent`
  shows it), and `--max-output-tokens` is not applied. A reply that stops at 16,384 is scored as
  it arrived (`replies_at_official_max_tokens` counts them); one that stops at a `max_tokens` the
  window lowered is scored as it arrived too, but counted in `replies_cut_off_by_budget` (a
  budget other than the official cap). With `--eval-thinking` no `max_tokens`
  is sent, so each request may use the rest of the context window, and `--max-output-tokens` is
  not applied either. `official_sampling` and `max_output_tokens_note` record which applied.

## Headline metric and how to read it
`accuracy` = correct attempts / judged attempts (harness errors and judge outages left out),
i.e. pass@1 averaged over the 3 (or more) repeats; `attempts` also reports pass@k, pass^k and how many questions changed verdict across
attempts. The result records `dataset` (with the revision), `judge` (`gbench-gemini-cascade`),
`judge_models` (answered judge calls per model, retries included), `judge_cascade` (the
configured cascade order) and `judge_sampling` (the checker call's settings, see above).

`leaderboard_comparable` is always **false**: Artificial Analysis grades with GPT-5.6 Luna
(medium), while gbench grades with its Gemini judge cascade (the models that answered are named
in `leaderboard_comparable_reason`). An `--eval-limit` subset and a temperature other than the
methodology's are also named there, and so are replies cut off by the run's token budget rather
than at an official cap (`replies_cut_off_by_budget`; `replies_cut_off_in_reasoning` counts
those that ran out inside the reasoning channel with no answer), and any harness errors.

The prompts average ~100k `cl100k_base` tokens; with the Gemma 4 tokenizer they are 89k to 123k
tokens (median 111k). Artificial Analysis requires a 128K context window. On a smaller served
window, prompts that do not fit are reported as `over_context_prompts`; they are harness
errors (`over_context`), left out of the accuracy, so serve at least a 128K window. On a
131,072-token window every prompt fits, but 37 of the 100 leave less room than 16,384 tokens,
and a no-think request then carries what fits (the methodology's own downward adjustment; the
`max_tokens_sent` field shows it, and a reply that stops there is counted in
`replies_cut_off_by_budget`); a thinking run gets the rest of the window and can reach its end. Serve the model at its native context length (256K for the larger Gemma 4 models) for full
headroom.

## Requirements
- **`GEMINI_API_KEY`** or **`GEMINI_API_KEYS`** (one key, or a comma/semicolon list;
  `GEMINI_API_KEYS` wins when both are set) for the equality checker. A missing or rejected key
  **hard-errors** (`infra_required`) before generation; there is no fallback scoring.
- The `google-genai` client (in the `evals` extra).
- Network access to the Hugging Face Hub (or a warm HF cache) for the pinned dataset revision.
- An OpenAI-compatible endpoint serving at least a 128K-token window.

## Run
```bash
export GEMINI_API_KEY="..."
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals aa_lcr --max-output-tokens 16384
```
`--max-output-tokens` is required by the CLI but not applied in either mode (see Maximum output
tokens above). Add `--eval-limit 20` for a quick (non-comparable) subset. The full run is 300 generations
(100 questions x 3 repeats) of ~100k-token prompts; prefix caching on the server helps.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered questions
(round-robin: positions I, I+N, I+2N, ...), selected before the `--eval-limit` draw, which is then
taken within the shard (N shards run up to N x `--eval-limit` questions). The result records
`shard`, `shard_index` and `shard_count`, and a shard of more than one is not
leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`); `gbench
--merge` combines the shards' results as docs/distributed-gbench.md describes: it re-counts the
union of their traces, and refuses a suite whose headline is not a correct / scored count over
them unless the suite defines its own merge.
