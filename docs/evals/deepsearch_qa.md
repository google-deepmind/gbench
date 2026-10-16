# deepsearch_qa

DeepSearchQA (Google DeepMind): 900 hand-written research tasks across 17 fields, each built
as a chain of dependent lookups on the open web and ending in a verifiable answer. 65% of the
tasks expect a *set* of answers (for example every country that meets several conditions), so
the benchmark measures exhaustive, precise multi-step search rather than single-fact
retrieval. The model gets a web search tool; an LLM rater compares its answer with the gold
answer set.

Sources: the dataset card, the
[technical report](https://storage.googleapis.com/deepmind-media/DeepSearchQA/DeepSearchQA_benchmark_paper.pdf)
and the official Kaggle starter notebook `andrewmingwang/deepsearchqa-starter-code` (version 1),
whose rater prompt and aggregation gbench copies.

## Dataset
- `google/deepsearchqa`, file `DSQA-full.csv` (split `eval`, Apache-2.0), pinned to revision
  `b2623f8653065c2672de6d941fc5434cd652376c`. Columns: `problem`, `problem_category`,
  `answer`, `answer_type` (`Single Answer` or `Set Answer`).
- The CSV is read with Python's csv module: four set-answer tasks have the literal answer
  `None` (no item qualifies), which pandas' default parsing would turn into NaN.
- `--eval-limit N` draws a subset of N tasks stratified by `problem_category`.

The run logs its planned workload in one line at its start (for example `[deepsearch_qa]
--eval-limit 5: 5 sample(s) x 1 attempt = 5 generations`) and records it in the result as
`eval_limit_plan` (`limit`, `samples`, `per_setting`, `attempts_per_sample`, `generations`).

## Protocol
- **Prompt.** The raw `problem` is the only message: no system prompt, header or footer, as in
  the notebook. `answer_type` is never shown to the model (the card forbids it).
- **Tools.** The request declares one tool, `web_search(query)`, backed by gbench's Gemini
  Google-Search grounding cascade. **The tool's results are written by Gemini, not taken from
  web pages.** Each call is one Gemini `generateContent` request with the `google_search` tool
  and the model's query as the prompt (`search_tool.gemini_search`); each result carries the
  title and URL of a source the grounding cited and, as its snippet, the part of the Gemini
  model's own answer that cites it (`groundingSupports[].segment.text`; the start of the answer
  when no segment cites it), and when no source is cited the tool returns the start of the
  answer as one result. No page is fetched. In the notebook the model under test runs Google
  Search itself (`generate_content` with `Tool(google_search=GoogleSearch())`) and no second
  model writes answers for it; here the model can hand the whole task to the tool and read back a
  Gemini model's grounded answer, so the score partly measures the Gemini model's research.
  DeepSearchQA defines no tool description, so the one the model sees is gbench's and says this
  (`gaia.GROUNDED_SEARCH_DESCRIPTION`, recorded as `search_tool_description`; the backend is
  named in `search_backend_kind`).
- **Tool transcripts.** Every trace keeps its `web_search` calls in `tool_transcript` (the
  arguments, the Gemini backend that answered, `gemini:<model>`, whether the search backend
  failed, and the reply the model read) and their number in `web_search_calls`.
- **Tool loop.** The reference runs one Gemini call with Google Search, so the search loop runs
  inside the model API with no harness step limit and no forced answer. gbench mirrors that:
  - no round cap and no tool-call cap: the loop ends when the model replies without a tool
    call, or when the transcript no longer fits the context window;
  - no injected turns: gbench's "answer now" and recovery prompts are off for this suite;
  - no turn carries a `max_tokens`, with or without `--thinking`, as the notebook's model call
    sets no `max_output_tokens` (see [Sampling](#sampling)): each turn may use the room the
    context window leaves; the global tool-loop turn ceiling
    (`GBENCH_TOOL_LOOP_MAX_OUTPUT_TOKENS`) does not apply;
  - the graded response is the last assistant message with content (when the loop ends on a
    tool call, the text the model wrote before it); when a model request fails mid loop, the
    sample is a harness error instead (below);
  - the assistant turns sent back to the model carry the model's own content and tool calls.
  The result records these settings in `tool_loop_protocol` and how each sample's loop ended
  (`tool_rounds.loop_ends`, per sample `tool_loop_end`).
- **Cost warning.** With no round or call cap, a model that keeps calling the tool is stopped
  only by the context window, and every turn may use the room the window leaves, so one task can
  cost many turns and many search calls. gbench knows the window when it launches the server or
  reads the served model's `max_model_len`; when it does not, nothing is refused ahead of time
  and the loop runs until the server rejects an over-long request. Watch
  `search_backend_calls` and `tool_rounds` on a first run.
- **Rater.** The notebook's `DEEPSEARCH_QA_PROMPT` + `GRADER_RATING_OUTPUT_EXAMPLE`, verbatim,
  filled with the problem, `answer_type`, the gold answer and the model's response. The rater
  returns, as JSON, a boolean per expected answer ("Correctness Details") and a list of
  answers not in the gold set ("Excessive Answers"); the reply is parsed with the notebook's
  rules (a ```json fence is unwrapped; malformed details or excessive answers make the rating
  invalid).
- **Scoring**, per rated task, exactly as the notebook's `aggregate_ratings` (technical report
  section 3.1): precision = found / (found + excessive), recall = found / expected, F1; and one
  category: Fully Correct (every expected answer, no excessive ones), Fully Incorrect (none of
  the expected answers), Correct with Extraneous Answers (every expected answer plus extra
  ones) or Partially Correct.
- **Edge cases.** An empty model response (the server answered with no text) is not sent to
  the rater and is scored Fully Incorrect with F1 0 (the report's definition: nothing submitted,
  nothing found). An invalid rater reply is a judge failure (`judge_grade: unparsed`) and is
  excluded from every metric, as in the notebook. A rater-cascade outage is `JUDGE_OUTAGE` and is
  excluded too. Every rated trace keeps the rater's model (`judge_model`), its reply (`judge_raw`,
  up to 4,000 characters) and its explanation (`judge_explanation`), as the notebook's
  `ItemRating` keeps `rating_response` and `answer_correctness_explanation`; a trace's `status`
  is `OK` (or `TRUNCATED` for a reply cut off at the end of the window) once it is graded.
- **Harness errors.** A sample that ended in a harness error is not sent to the rater and never
  scored as the model's outcome: `judge_grade: HARNESS_ERROR`, `harness_error` (kind and detail)
  and `scoring_excluded` on its trace, left out of every metric (`accuracy` and the
  `fully_correct` family, `f1_score`, the empty-response count), counted in `harness_errors`
  (`{kind: samples}`) and `num_harness_error`, and named in `leaderboard_comparable_reason`; the
  run's `status` is `completed_with_errors`. When no sample is rated the score is withheld
  (`accuracy: null`, `status: failed`, `accuracy_withheld_reason`). The kinds:
  - `endpoint_failure`, `endpoint_timeout`, `over_context`: a model request produced no reply (an
    endpoint error, a timeout, or a prompt that does not fit the served context window). Such a
    request is no longer counted as an empty model response;
  - `request_failed_mid_loop`: a model request of the tool loop failed after earlier turns (the
    text base would otherwise grade is a pre-search preamble, not the model's answer);
  - `search_backend_failure`: a `web_search` call no Gemini grounding model answered
    (`all-grounding-models-failed`, `deadline-passed`, or no key).

## Headline metric and how to read it
`accuracy` (also `fully_correct`) is the Fully Correct rate, the number the Kaggle leaderboard
quotes; `fully_correct_ci95` is the notebook's 95% interval half-width. The report names F1 its
ranking metric, so `f1_score` sits alongside, with `precision`, `recall`, `fully_incorrect`,
`correct_with_extraneous_answers`, `partially_correct` and `fully_correct_by_answer_type`.
`fully_correct_excluding_empty` / `f1_score_excluding_empty` apply the notebook's own rule of
dropping empty responses. Counts: `num_rated`, `num_empty_model_response`,
`num_invalid_auto_rater_response`, `num_judge_outage`, `num_harness_error`; `judge_models` records
which Gemini models graded (and each trace its own `judge_model`); `search_backend_calls` records
which grounding model served each lookup, and `harness_errors` the samples left out by kind.

`leaderboard_comparable` is always **false**. `leaderboard_comparable_reason` lists:
- the rater is gbench's Gemini judge cascade, not the `gemini-2.5-flash` the card prescribes
  (the card warns a different rater shifts results). It is called as the notebook calls its
  rater: the prompt alone with no generation config, so the API's default sampling applies.
  This item drops only when every grade of the run came from `gemini-2.5-flash`.
  `GBENCH_JUDGE_MODELS` is the one cascade for every Gemini role of the run, so a cascade set to
  that model also serves this suite's `web_search` grounding unless `GBENCH_SEARCH_MODELS` is set;
  and not every key is served that model (the Gemini API answers HTTP 404 for it on some, and the
  live check before the run then stops the suite);
- the temperature gbench sends (the notebook sends none);
- web access is gbench's Gemini-written `web_search` tool above (the cited sources' titles and
  URLs with segments of a Gemini model's grounded answer, no page text, no page fetch), where the
  notebook's model runs Google Search itself, so the score partly reflects the Gemini model's
  research;
- when it happens, how many empty responses were scored Fully Incorrect, any harness errors, and
  any `--eval-limit` subset.

## Requirements
- A running OpenAI-compatible `/v1` chat endpoint that accepts tool calls (vLLM:
  `--enable-auto-tool-choice --tool-call-parser <parser>`).
- **`GEMINI_API_KEYS` or `GEMINI_API_KEY`** (`GEMINI_API_KEYS` wins when both are set; either
  may be a comma or semicolon separated list): powers both the search tool and the rater.
  Required; without it, with a key the API rejects, or when a model of the Gemini cascade is one
  the API does not serve for the key, the suite hard-errors (`infra_required`) before any
  download.
- Python packages `huggingface_hub` and `aiohttp` (base install) and `google-genai` (the
  `evals` extra: `pip install -e ".[evals]"`). The dataset is public and ungated.
- Optional: `GBENCH_DEEPSEARCH_QA_TEMPERATURE` sets the temperature for this suite (default 1.0
  with `--thinking`, 0.0 without); the judge cascade (`GBENCH_JUDGE_MODELS`, ...) and web-search cascade
  (`GBENCH_SEARCH_MODELS`, `GBENCH_SEARCH_MAX_RPS`, ...) knobs are in [env-knobs](env-knobs.md).

## Sampling
The starter notebook calls the model with `GenerateContentConfig(tools=[google_search])` and
nothing else. gbench sends no other sampling field, but always sends a temperature, its rule for
every suite: 1.0 with `--thinking`, 0.0 without. `--temperature` or
`GBENCH_DEEPSEARCH_QA_TEMPERATURE` (which takes precedence) sets another. The result records it
under `official_sampling` and names it in `leaderboard_comparable_reason`. The
rater is called with no generation config, as the notebook's rater is (the Gemini API's default
sampling), and does not follow either setting.

No `max_tokens` is sent either, in either mode (the token-limit policy's "not sent" case,
`official_sampling.send_max_tokens: false`): the notebook's config sets no `max_output_tokens`, so
the server's default applies, the room left in the context window. `--max-output-tokens`, which
the CLI requires, is recorded as not applied in `max_output_tokens_note`. A turn cut off at the
end of the window is counted in `truncated_responses`, and an empty final response is scored
fully incorrect and named in `leaderboard_comparable_reason` (see above).

## Run
```bash
export GEMINI_API_KEY=...            # or GEMINI_API_KEYS=k1,k2
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals deepsearch_qa --max-output-tokens 16384
```
`--max-output-tokens` is required for every eval run and is recorded as not applied (no
`max_tokens` is sent). Add `--eval-limit 20` for a quick check.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered problems
(round-robin: positions I, I+N, I+2N, ...), selected before the `--eval-limit` draw, which is then
taken within the shard (N shards run up to N x `--eval-limit` problems). The result records
`shard`, `shard_index` and `shard_count`, and a shard of more than one is not
leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`); `gbench
--merge` combines the shards' results as docs/distributed-gbench.md describes: it re-counts the
union of their traces, and refuses a suite whose headline is not a correct / scored count over
them unless the suite defines its own merge.
