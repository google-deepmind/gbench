# livebench

Canonical LiveBench (White et al.; [livebench.ai](https://livebench.ai);
[github.com/LiveBench/LiveBench](https://github.com/LiveBench/LiveBench)), a monthly-refreshed,
contamination-limited benchmark. gbench runs its **6 core categories** (`coding`,
`data_analysis`, `instruction_following`, `language`, `math`, `reasoning`), each scored by
**LiveBench's own ground-truth judges** (no LLM judge), with **real code execution** for the
coding category.

The 7th category, `agentic_coding`, is **excluded** (LiveBench's default benchmark set already
excludes it; it needs a separate ~150GB Multi-SWE-Bench container harness).

Pinned by default: LiveBench commit `0baa4aa7` (`LIVEBENCH_COMMIT`), run from the local image
`gbench-livebench:0baa4aa7` (`LIVEBENCH_IMAGE_DEFAULT`), and LiveBench release **`2024-11-25`**
(`LIVEBENCH_RELEASE`), the release LiveBench's README tells users to pass to evaluate all
categories on its public questions (see "Releases" below). The image carries one recorded local
change to that checkout, a one-line fix for an upstream crash in the instruction_following
grading (see "Recorded patch" below).

## Why it's containerized

LiveBench's package (`latex2sympy2`, `spacy`, `litellm`, ...) and its coding-execution
environment (`code_runner/requirements_eval.txt`: `tensorflow`, `numba`, `opencv`,
`scikit-image`, ...) would churn a tightly-pinned torch/vLLM serving venv. So gbench does **not**
install LiveBench into your serving env; it runs LiveBench's own pipeline inside an isolated
Docker image, `gbench-livebench:0baa4aa7` by default, and only reads back the scores. gbench
hard-errors (never silently skips) if Docker or the image is missing.

## Setup (step by step)

### 1. Get the LiveBench checkout (needs git-lfs)

LiveBench LFS-tracks its `*.json` data, so cloning needs **git-lfs**:

```bash
# git-lfs - apt (system) OR pip:
sudo apt-get install -y git-lfs        # Debian/Ubuntu
#   or:  pip install git-lfs
git lfs install

git clone https://github.com/LiveBench/LiveBench
cd LiveBench
git checkout 0baa4aa7a78dc0276c0ccf28779dba1113580e18   # the commit gbench was reconciled against
git lfs pull                           # pull the LFS data
```

gbench's flags were reconciled against LiveBench commit `0baa4aa7` (main on 2026-09-24,
`LIVEBENCH_COMMIT` in `livebench.py`) and exercised end to end against an image built from
`bc5164c1` (2026-09-04); both carry the same `run_livebench.py` flags and model-config format.
(Upstream main has since moved one commit, to `85f874fc` on 2026-09-26, which changes only the
opt-in live concurrency resize of `gen_api_answer.py` (`--parallel-control-file`, a flag gbench does
not pass); what gbench runs is unchanged.)
Before each run gbench reads the image checkout's commit (`git rev-parse HEAD` in the image,
recorded as `livebench_image_commit`); an image built from another commit is reported as not
leaderboard-comparable, so build it from `0baa4aa7`. Keep the checkout's `.git` directory in the
build context (do not exclude it with a `.dockerignore`): an image whose commit cannot be read is
not verified as `0baa4aa7` and is reported as not comparable too.

### 2. Build the `gbench-livebench:0baa4aa7` image

Run this from the gbench repository root. The build context **must be your LiveBench checkout at
the pinned commit** (so its LFS-pulled data and its `.git` directory are included), not gbench's
`docker/` directory, and the tag is the first 8 characters of `LIVEBENCH_COMMIT`, which is the
image gbench runs by default (`BUILD_COMMAND` in `livebench.py`):

```bash
git -C /path/to/LiveBench checkout 0baa4aa7a78dc0276c0ccf28779dba1113580e18
git -C /path/to/LiveBench lfs pull
docker build -t gbench-livebench:0baa4aa7 \
    -f docker/livebench.Dockerfile /path/to/LiveBench
```

Right after copying the checkout, the Dockerfile applies gbench's recorded patch (see "Recorded
patch" below): it checks that `livebench/gen_ground_truth_judgment.py` has the pinned sha256
(`dd6bdce67dafd270030980fc864a37940f09a8aa83ce6f94d8657052afbdbd37`), replaces the one line, and
checks the patched sha256 (`19ed4fef82fd839acc0f558380adb4631fdbc5694948575e0c0d319f58ec3a85`);
the build stops if either check fails. It then installs LiveBench + its coding-execution
requirements (heavy, `tensorflow` etc.) against the constraints written inline in the Dockerfile
(`PIP_CONSTRAINT`: the installed set of the image built 2026-09-26), and fails if anything else is
installed; it records the installed Python and Debian packages in `/opt/gbench`. Then it bakes in the
data LiveBench's instruction_following grading would otherwise download at run time: the NLTK
packages `punkt`, `punkt_tab`, `averaged_perceptron_tagger` and `stopwords`, and spaCy's `en_core_web_sm`
(installed with `python -m spacy download`, the version that matches the image's spaCy). The NLTK
packages come from the nltk_data commit `98e14261eaec2244af1db4ec5b282112444495aa` (the files
`nltk.download` served on 2026-09-26), each checked against its pinned sha256.
A download that fails fails the build, as does one whose sha256 differs, so an image without that
data is never produced. (An image built from an
earlier version of the Dockerfile has only `punkt` and `punkt_tab`; LiveBench then downloads the
rest during grading, which needs network access. Rebuild it to bake them in.) It's a large image;
the first build is slow. If a package in
`requirements_eval.txt` needs an extra system lib, add it to the `apt-get` line in the
Dockerfile and rebuild. gbench never pulls the image and never falls back to an untagged
`gbench-livebench` (`:latest`), which may hold another commit; when the tagged image is missing
the suite stops with `infra_required` and these build steps. Override the image name with
`GBENCH_LIVEBENCH_IMAGE` if you tag it differently (its commit is still checked, below).

### 3. Run

```bash
# All 6 core categories at the pinned release 2024-11-25 (the default):
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-26B-A4B-it --evals livebench \
       --max-output-tokens 32768 \
       --suite-timeout 36000                 # LiveBench is long; raise the timeout
```
(`GBENCH_LIVEBENCH_RELEASE=<date>` runs another release; `GBENCH_LIVEBENCH_RELEASE=latest` passes
no release, so LiveBench takes its newest one, where `coding` has no public question; see
"Releases" below.)

Internally the suite runs LiveBench with `--mode single` (no tmux) and scores with
`--ignore-missing-judgments`; you don't set those. Quick smoke test: add `--eval-limit 12` (12
questions in total, drawn stratified by category, never 12 per category; see "Question selection"
below). The run logs its plan on its first line and records it as `eval_limit_plan` (questions per category).

What happens: gbench `docker run`s the image on docker's bridge network (an endpoint on this
host, such as `127.0.0.1:8000`, is reached there at `host.docker.internal:8000`; see
"Containers and network" below) and executes LiveBench's own `run_livebench.py`: it downloads the
questions, generates answers against your endpoint (via `--api-base` + `--use-litellm`), grades
them with LiveBench's per-category judges, then `show_livebench_result.py` writes
`all_groups.csv`. gbench reads that and reports **`accuracy` = the `average` over the categories
LiveBench scored** (the 6 core categories when each is scored), with each category in
`dimension_scores` and any category without a score in `categories_missing` (or, when gbench
selected the questions and gave a category none, in `categories_not_run_by_limit` /
`categories_inactive_at_release`; see "Coverage and status" below). A request that
failed at the endpoint is scored 0 inside LiveBench's averages, so while any did, the average is
withheld (`accuracy` is `null`, with `accuracy_withheld_reason`) and so is the score of each
category with one (`dimension_scores_withheld`); see `failed_requests` below.

## Generation settings forwarded to LiveBench

LiveBench, not gbench, sends the requests, so gbench hands it the run's settings the way
LiveBench's own model configs carry them. LiveBench keeps a config per model
(`livebench/model/model_configs/*.yml`), which `run_livebench.py --model` finds by display name or
alias (`get_model_config`, case-insensitive). Before the run, gbench asks the image's LiveBench for
the served model's config: `docker run --rm --network none <image> python -c ...` calls
`get_model_config` for the served model id (`GBENCH_LIVEBENCH_MODEL`, default the gbench model
name), then the run's `--tokenizer`, each as given and as its last path component, so
`google/gemma-4-31B-it` finds `gemma-4-31b-it`. The image not answering is `infra_required`; a
name LiveBench finds in several configs is an error. gbench then writes a one-off LiveBench model
config (display name `gbench-served-model`, provider `local`, API name = the served model id) to
the run's output directory; the container copies it into the installed package's `model_configs/`
directory, and `run_livebench.py --model gbench-served-model` selects it.

| setting | how it reaches the endpoint |
| --- | --- |
| served model with a LiveBench config (at `0baa4aa7`, for example `gemma-4-31b-it`: temperature 1.0, top_p 0.95, `max_tokens` 65536, reasoning enabled on its OpenRouter provider) | the kwargs that config sends on its leaderboard provider (`api_kwargs.default` updated with the entry of its `default_provider`, as `gen_api_answer.py` `setup_model` merges them) go into the one-off config's `api_kwargs.default`: the temperature, `top_p` and the other sampling fields, the token cap (see below), `stream` / `timeout` / `reasoning_effort`, and its `prompt_prefix` / `preserve_reasoning`. Its reasoning switch for that provider (`extra_body.reasoning.enabled`, or `extra_body.chat_template_kwargs.enable_thinking`) is the local endpoint's `chat_template_kwargs.enable_thinking`, which gbench sets from `--thinking`; a config whose switch differs from `--thinking` (a no-think run of `gemma-4-31b-it`, say) is not comparable. A field with no local counterpart (a vendor's `thinking_config`, `betas`, OpenRouter `provider` routing, ...) is not sent and makes the run not comparable. The result records the config, what was sent and what was not under `livebench_model_config`. |
| served model without one | LiveBench's defaults: `--max-tokens` (the temperature is gbench's, see below) |
| token budget | gbench's token-limit policy (`base.resolve_token_budget`). **No-think:** the official cap exactly, i.e. the model config's cap, else `gen_api_answer.py`'s `--max-tokens` default **4096**; `--max-output-tokens` (required on the command line) is not applied. **`--thinking`:** a model config's cap is that model's own setting and is sent (65536 for `gemma-4-31b-it`), or `--max-output-tokens` when that is larger (a thinking run is never capped below it); the 4096 default was written for non-reasoning generation and is not applied, so each request gets `--max-output-tokens`, or, when the suite is called programmatically without one, gbench's thinking floor 32768. **Fitted to the served window:** LiveBench sends one fixed `max_tokens` with every request and vLLM rejects a request whose prompt + `max_tokens` exceeds the window, so whatever its source (official cap, model config, `--max-output-tokens`, thinking floor or `GBENCH_LIVEBENCH_EXTRA_ARGS`), a budget that would leave the longest prompt less than 8192 tokens (half the window when that is smaller; the longest core prompt is about 8.1k characters) is lowered to fit the window the server reports, as gbench lowers each request of the other suites. The result records it (`max_output_tokens_source` ends in `fit-to-window`, the note says what the token-limit policy gives, then "lowered to N" and "each request was sent max_tokens=N", never naming the unlowered value as sent; `official_sampling.max_tokens_sent`), and an official cap lowered this way (the `gemma-4-31b-it` config's 65536 on a 32768-token window, say) is not comparable: the reason names the `--max-model-len` it needs (cap + 8192). A model config's cap goes into the one-off config (a config's cap replaces `--max-tokens` in LiveBench); otherwise gbench passes `--max-tokens N`. `max_output_tokens_note` states the mode's choice. |
| reasoning mode | `chat_template_kwargs: {enable_thinking: true/false}` in the one-off config's `api_kwargs.default.extra_body`. LiteLLM forwards `extra_body` in the request body, which is how vLLM's chat template receives it. |
| temperature | gbench's rule for every suite: **0.0** without `--thinking`, **1.0** with it, or the operator's `--temperature` / `GBENCH_LIVEBENCH_TEMPERATURE` (which takes precedence) in either mode. Always written into the one-off config's `api_kwargs.default.temperature` (a config's temperature replaces `get_answer`'s and `--force-temperature` on every request, a model config's own temperature included), never passed as `--force-temperature` (see Sampling below). |

Verified 2026-09-24 against a request-recording stub endpoint: every LiveBench request carried
`max_tokens` and `chat_template_kwargs.enable_thinking` as above. Verified 2026-09-25 in an image
built from `bc5164c1`: the in-image lookup finds `gemma-4-31b-it` for `google/gemma-4-31B-it`, and
with gbench's one-off config for it LiveBench's own `setup_model` and LiteLLM path sent
`temperature` 1.0 (over `get_answer`'s 0), `top_p` 0.95, `max_tokens` 65536 and
`chat_template_kwargs.enable_thinking: true` to the stub. A `--max-tokens` you put in
`GBENCH_LIVEBENCH_EXTRA_ARGS` wins over gbench's (it is written into the one-off config when the
model config carries the cap) and is what the result records; a value other than the policy's makes
the run not comparable.

## Configuration

| Env var | Purpose |
| --- | --- |
| `GBENCH_LIVEBENCH_IMAGE` | image tag (default `gbench-livebench:0baa4aa7`, i.e. `gbench-livebench:` + the first 8 characters of `LIVEBENCH_COMMIT`; an empty value is the default) |
| `GBENCH_LIVEBENCH_MODEL` | the model id sent to the endpoint (the model config's API name; default = the served model id gbench passes, which the eval runner matched against `--models`); also the first name LiveBench's config lookup tries. Another value is checked against the endpoint's `/v1/models` listing first (an id it does not serve is `infra_required`, never sent); recorded as `request_model_id` / `request_model_id_source` |
| `GBENCH_LIVEBENCH_RELEASE` | the LiveBench release date (default `2024-11-25`, `LIVEBENCH_RELEASE`); `latest` passes none, so LiveBench takes its newest release (unpinned, not comparable). See "Releases" below; this matters. |
| `GBENCH_LIVEBENCH_CATEGORIES` | restrict to a subset of the 6 core categories (space/comma list, any case; default all 6); any other name is an error before a container runs. A subset that names a multi-word category (`data_analysis`, `instruction_following`) runs by question id (see "Question selection"). |
| `GBENCH_LIVEBENCH_EXTRA_ARGS` | extra flags appended to `run_livebench.py`; a `--max-tokens` here overrides gbench's budget (not comparable when it differs from the token-limit policy's). Question selection here (`--question-id`, `--question-begin`, `--question-end`) is refused while gbench selects the questions (`--shard`, `--eval-limit`, a multi-word category subset) and otherwise makes the run not comparable. |

## Releases (important)

LiveBench is **monthly-refreshed**: each release adds questions and retires others, and a run is
defined by one **release date** (`--livebench-release-option`, which gbench passes to both
`run_livebench.py` and `show_livebench_result.py`). LiveBench's `load_questions`
(`livebench/common.py`) keeps a question when its `livebench_release_date` is on or before the
release and its `livebench_removal_date` is empty or after it.

At `0baa4aa7`, `LIVE_BENCH_RELEASES` (`livebench/common.py`) runs from 2024-06-24 to 2026-06-25,
and `gen_api_answer.py` defaults to the newest. But LiveBench withholds its newest questions; its
`README.md` says: "The current LiveBench release is 2025-04-25; however, not all questions for this
release are public on Huggingface. In order to evaluate all categories, you will need to pass
`--livebench-release-option 2024-11-25` to all scripts to use the most recent public questions."
The public datasets (`huggingface.co/datasets/livebench/<category>`, last modified 2025-04-07)
hold the questions released up to 2024-11-25. Active questions per release, counted with
LiveBench's filter on those datasets (2026-09-26):

| release | coding | data_analysis | instruction_following | language | math | reasoning | total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-11-25 | 128 | 150 | 200 | 140 | 232 | 150 | 1000 |
| each of 2025-04-02 ... 2026-06-25 | 0 | 150 | 200 | 50 | 182 | 100 | 682 |

1000 is the release size LiveBench's `changelog.md` keeps fixed ("all future updates add and
remove the same number, to stay at 1000 total questions"). From 2025-04-02 on, every public coding
question is removed (all 128 carry removal date 2025-04-02) and the questions that replaced the
retired ones are not public, so a later release scores only part of its questions and no coding.

So gbench pins **`2024-11-25`** (`LIVEBENCH_RELEASE`): the newest release whose questions are all
public, with all 6 core categories active. The result records `livebench_release` and
`livebench_release_source` (`default`, or `env:GBENCH_LIVEBENCH_RELEASE`).

- `GBENCH_LIVEBENCH_RELEASE=<date>` runs another release. LiveBench's `gen_api_answer.py` accepts
  only a date in `LIVE_BENCH_RELEASES` (any other value stops the run with no scores). At a
  release after 2024-11-25 `coding` has no active question. In a full run (no `--eval-limit`,
  no shard) gbench does not list the release's questions, so it cannot tell this from a
  category LiveBench failed to answer: `coding` is reported in `categories_missing`, the status
  is `completed_with_errors` with a `status_reason` saying so, and the run is not comparable.
  When gbench selects the questions (`--eval-limit`, `--shard`) it has the release's list, so
  `coding` goes to `categories_inactive_at_release` and is not an error.
- `GBENCH_LIVEBENCH_RELEASE=latest` passes no release, so LiveBench takes its own default (its
  newest release); `livebench_release` is `latest` and the run is not comparable (unpinned).

`leaderboard_comparable` needs the **full** question set (no `--eval-limit`) for a pinned release
with all default categories scored, plus the sampling and image conditions under "Reading the
result". It is **not** gated on greedy/`--thinking`: LiveBench's leaderboard includes reasoning
models, and the budget and reasoning mode are forwarded (above), so a `--thinking` run is a
faithful measurement of that mode.

## Question selection (`--eval-limit`, `--shard`)

With `--eval-limit N`, `--shard I/N`, or a `GBENCH_LIVEBENCH_CATEGORIES` subset that names a
multi-word category, gbench selects the questions by id:

1. It lists the release's questions in the image with LiveBench's own loader
   (`livebench.common.get_categories_tasks` + `load_questions`, with the release filter
   `gen_api_answer.py` applies). This probe container runs on the same bridge network as the run
   (it reads Hugging Face).
2. It keeps the categories asked for, orders their question ids, takes the shard first (id I, I+N,
   ... of that order; an id LiveBench uses in two tasks is one unit), then draws `--eval-limit` ids
   from it stratified by category with the seed `livebench` (`sampling.stratified_sample`): the same
   questions on every run, spread over the categories, never the first N of each task.
3. It passes the ids to both LiveBench scripts with `run_livebench.py --question-id` on the
   whole-suite bench name `live_bench`, which also avoids LiveBench's per-category loader (it
   truncates a multi-word category at the underscore, `data_analysis` -> `livebench/data`, and the
   run crashes).

Such a run is a subset, so it isn't `leaderboard_comparable`. The result records
`selected_question_ids`, the shard identity (`shard`, `shard_index`, `shard_count`,
`shard_applied`) and, for a partial shard, `sharded run (--shard I/N) ...` in
`leaderboard_comparable_reason`; an empty shard runs nothing and reports `accuracy: null` with
`accuracy_withheld_reason`, with status `success` and a `status_reason` (`empty shard I/N: ...`)
saying why; `gbench --merge` of shards that are all empty keeps those reasons. A subset of single-word categories with neither flag keeps LiveBench's
per-category bench names (`live_bench/math live_bench/reasoning`).

Per question, the container also copies LiveBench's judgment rows
(`data/live_bench/<category>/<task>/model_judgment/ground_truth_judgment.jsonl`): the result
carries one trace per judged question (`sample_traces`: `question_id`, `category`, `task`, `score`
in [0, 1], `sample_key`) and `questions_judged`. `total_questions` and `effective_n` count the
questions judged with a score (a `-1` row, which LiveBench leaves out, is not counted);
`correct_answers` is `null` with `correct_answers_note`: LiveBench scores each question in
[0, 1] and reports means, not a count of correct answers. The run's output directory, with
LiveBench's model answers (`model_answer/`), judgments (`model_judgment/`), `all_groups.csv` /
`all_tasks.csv` and gbench's model config, is kept under
`$GBENCH_CACHE_DIR/livebench/runs/<timestamp>-...` and recorded as `livebench_output_dir`; the
container (which runs as root) hands it to the host user running gbench when it is done.
`gbench --merge` recomputes LiveBench's aggregation over the union of the shards' traces, as
`show_livebench_result.py` does (per task the mean score x 100, per category the mean of its
tasks, the average the mean of the categories); a headline any shard withheld stays withheld, and
a question found in two shards is refused (MergeError: the shards overlap, so they are not the
partition of one ordered set), as the generic merge refuses it. LiveBench answers each question once, so
`--attempt-count` above 1 is not applied (`attempts.not_applied`).

### Coverage and status

The result records `livebench_release` and four category lists so the coverage is explicit:

- `categories_missing`: categories the run asked LiveBench to answer that have no score. With
  answer rows they were answered and not scored by LiveBench's grading (also in
  `categories_unscored`, below). Without answer rows: when gbench selected the questions, the
  category had selected questions LiveBench did not answer; in a full run, which lists no
  questions, it is most likely a category with no active question at the release (coding after
  2024-11-25), not checked. Each makes the status `completed_with_errors`, with a
  `status_reason` naming the categories and the cause.
- `categories_unscored`: of `categories_missing`, those LiveBench answered but did not score.
- `categories_not_run_by_limit`: categories with active questions at the release of which
  gbench's selection took none (the `--eval-limit` stratified draw gave the category 0, as
  `eval_limit_plan.per_setting` shows, or the shard holds none of its questions). They have no
  score by design and do **not** change the status; the run is a subset and is not comparable
  anyway (`categories with no question run ...` in `leaderboard_comparable_reason`).
- `categories_inactive_at_release`: categories LiveBench's loader lists no question of at the
  release (checked on the release's question list when gbench selects the questions; empty in a
  full run). Not an error either.

The status is `completed_with_errors` only for real failures: a category in
`categories_missing`, a request that failed at the endpoint (`failed_requests`), or requests
the server aborted (`server_aborted_requests`); `status_reason` lists which. 2026-10-11
validation run (pool-10-11-132525, `--eval-limit 5`): the stratified draw gave
instruction_following 0 of the 5 questions, the 5 were all judged (`accuracy` 80.0, no failed
request, no abort), and the earlier gate, which counted instruction_following as missing, marked
the run `completed_with_errors` with no `status_reason`; replaying that run's output through the
current gate (same 5 question ids from the release's 1000) gives `success` with
`categories_not_run_by_limit: ["instruction_following"]`. `gbench --merge` recomputes the lists
over the union of the shards (a category one shard ran no question of is not a gap when another
ran one) and keeps a shard's `completed_with_errors` with its `status_reason`. If **no** category
is scored, the suite hard-errors (nothing to report; a harness failure when LiveBench answered
some).

## Recorded patch: instruction_following grading at `0baa4aa7`

Every public `instruction_following` question (released 2024-06-24 or 2024-11-25) takes
LiveBench's "old instruction following" grading path: `incremental_judge.py` leaves it to the
batch pass ("old-format IF is graded per-task in batch only"), and in `gen_ground_truth_judgment.py`
(questions released before 2025-11-25) line 480 of that path,
`if_questions = list(set([m.question for m in old_instruction_following_matches]))`, raises
`TypeError: unhashable type: 'dict'` (the questions are dicts), so upstream never grades the
category. The line comes from LiveBench commit `366e385` ("group all questions together when
grading", 2025-12-12) and is unchanged at upstream main `85f874fc`.

gbench keeps the pinned commit and carries one minimal patch in its local image, applied by
`docker/livebench.Dockerfile` at build time (`LIVEBENCH_PATCH` in `livebench.py`):

```diff
-        if_questions = list(set([m.question for m in old_instruction_following_matches]))
+        if_questions = list({m.question['question_id']: m.question for m in old_instruction_following_matches}.values())
```

It de-duplicates the questions by `question_id` (a question appears once per model graded), which
is what the `set` was for. Nothing else in LiveBench changes. The image's `git rev-parse HEAD`
stays `0baa4aa7`; `git status` in it shows that one modified file.

- **Checked before each run:** gbench reads the sha256 of the image's
  `gen_ground_truth_judgment.py` (`docker run --rm --network none <image> python -c ...`). Anything
  other than the patched sha256 is `infra_required` with the build steps: an image built before
  this patch (the unpatched file, `dd6bdce6...`) must be rebuilt with the current Dockerfile.
- **Recorded on every result:** `livebench_patch` carries the file, the line, the upstream and
  patched text, the reason, both sha256 values, where it is applied and its side effect (below).
- **Verified** on 2026-09-26 in `gbench-livebench:0baa4aa7` rebuilt from a clone of
  github.com/LiveBench/LiveBench at `0baa4aa7`, with a model-free grading run: stub answers (no
  model) to all 200 instruction_following questions of release 2024-11-25, for two stub models
  (one answering each question with its own prompt, one answering "OK"), graded with LiveBench's
  own `gen_ground_truth_judgment.py` and `show_livebench_result.py` as gbench runs them
  (`--bench-name live_bench`, huggingface questions). The unpatched image raises the TypeError and
  writes no judgment; the rebuilt image exits 0, writes 400 judgments, and `all_groups.csv` gives
  instruction_following 28.9 and 9.5, the mean of the four per-task scores (28.875 and 9.4875)
  that LiveBench's own instruction_following scorer gives the same answers task by task. A gbench
  run of all 6 categories through `run_livebench` in the same image, against a model-free stub
  endpoint answering "OK" (`--eval-limit 18`, one question per task), scored all 6 categories,
  instruction_following included (`categories_unscored` empty, status `success`).
- **Side effect (upstream, not changed):** the same pooled grading call labels every
  instruction_following judgment row with one task name (`task_name = if_questions[0]['task']`),
  so LiveBench's `all_tasks.csv` shows a single instruction_following column (every row was
  labelled `story_generation` in the grading check above and `summarize` in the gbench run) and
  the category score in `all_groups.csv`, the one
  gbench reports, is the pooled mean of the rows. That equals the mean of the task means when the
  tasks contribute the same number of rows: 50 each (paraphrase, simplify, story_generation,
  summarize) at 2024-11-25. gbench counts the answer rows per instruction_following task; when
  they differ (an `--eval-limit` draw or a shard, say), the run is not comparable and the reason
  says why. When
  instruction_following is scored but none of its answer rows were copied out of the container,
  the equal counts cannot be verified, so the run is not comparable either and the reason says
  so (`instruction_following answer rows were not found in the run's output ...`).

If LiveBench's grading still writes no instruction_following score (or none for another
category it answered), gbench reports it as it is: the category is listed in
`categories_missing` and `categories_unscored`, the reason says it was answered but not scored,
LiveBench's last output is logged, the status is `completed_with_errors` and the run is not
comparable. It is neither scored 0 nor dropped silently.

## Reading the result

| field | meaning |
| --- | --- |
| `accuracy` | LiveBench's `average` over the categories it scored (equal weight; `categories_missing` lists any it did not). `null` when a request failed at the endpoint in a scored category, or the server aborted requests during the run (`failed_requests`, `server_aborted_requests`): LiveBench scored them 0 inside that average, so it would count harness failures as the model's answers. `accuracy_withheld_reason` then says how many and where |
| `dimension_scores` | per-category scores from `all_groups.csv`, for the categories where every request was answered |
| `dimension_scores_withheld` | the categories whose `all_groups.csv` score is withheld, each with the reason: the number of its requests that failed at the endpoint, or the server's aborted requests (which cannot be placed in a category, so every category is withheld then) |
| `livebench_release`, `livebench_release_source` | the release scored (`latest` when `GBENCH_LIVEBENCH_RELEASE=latest` passed none) and where it came from (`default`: `LIVEBENCH_RELEASE`, 2024-11-25; or `env:GBENCH_LIVEBENCH_RELEASE`) |
| `categories_missing`, `categories_unscored` | the categories the run asked for with no score in `all_groups.csv`; `categories_unscored` lists those of them LiveBench answered (answer rows copied out of the container) but its grading did not score, a harness failure; the others were selected but not answered (with gbench's selection) or, in a full run, most likely had no active question at the release. Either makes the status `completed_with_errors` (with `status_reason`) and the run not comparable |
| `categories_not_run_by_limit`, `categories_inactive_at_release` | categories gbench's question selection (`--eval-limit`, `--shard`) gave no question: active at the release but none selected, and none active at the release (LiveBench's loader lists none). Expected to have no score; they do not change the status. Both empty in a full run (see "Coverage and status") |
| `status`, `status_reason` | `success`, or `completed_with_errors` when a category in `categories_missing`, a failed request or a server abort exists; `status_reason` (present only then) names each |
| `thinking`, `max_output_tokens`, `max_output_tokens_source`, `max_output_tokens_note` | the reasoning mode and token budget LiveBench was given, where the budget came from (`official`, `cli:--max-output-tokens`, `default` or `env:GBENCH_LIVEBENCH_EXTRA_ARGS`, with `:fit-to-window` appended when it was lowered to fit the served window; `default:fit-to-window` for the thinking floor), and the mode's max_tokens choice. When the budget was lowered to fit the window or replaced by `GBENCH_LIVEBENCH_EXTRA_ARGS`, the note states what the token-limit policy gives, what changed it, and ends with `each request was sent max_tokens=N`; otherwise it is the policy's own note |
| `temperature`, `temperature_source`, `official_sampling` | no-think: 0.0 / `default`; `--thinking`: 1.0 / `thinking`; otherwise the operator's temperature and its source (`env:GBENCH_LIVEBENCH_TEMPERATURE` or `cli:--temperature`). `official_sampling` records the rule (LiveBench's, or the model config's), `temperature_matches_official`, the official `max_tokens` and whether it was applied (`reasoning_settings: true` when a model config's cap was used with `--thinking`) |
| `livebench_model_config` | `null` when LiveBench ships no config for the served model; else its `display_name`, the name that matched, the `provider` it runs on, what gbench `sent`, and what it could `not_reproduced` |
| `livebench_image_commit` | the LiveBench commit of the image's checkout (`null` when git cannot read it, which makes the run not comparable) |
| `livebench_patch` | gbench's one local change to that checkout (see "Recorded patch"): `file`, `line`, the `upstream` and `patched` text, the `reason`, `sha256_upstream` and `sha256_patched` (the latter checked in the image before the run), `applied_by` and its `side_effect` on instruction_following task labels |
| `replies_cut_off_by_budget`, `replies_cut_off_in_reasoning`, `replies_cut_off_uncounted`, `replies_cut_off_note` | replies cut off by a `max_tokens` other than the official cap (the run's budget, one lowered to fit the window, or `GBENCH_LIVEBENCH_EXTRA_ARGS`'s), with and without `--thinking`, read from LiveBench's answer rows (`data/<task>/model_answer/<display>.jsonl`, copied out of the container). A row LiveBench marks `eval_status: token_exhaustion` (no content at `max_tokens`) is cut off inside the reasoning channel. For the others gbench uses `total_output_tokens`, which LiveBench's litellm path (`completions.py` `chat_completion_litellm`) builds as the server's `completion_tokens` plus its `completion_tokens_details.reasoning_tokens`; vLLM's `completion_tokens` already include the reasoning and never exceed `max_tokens`, and vLLM reports `reasoning_tokens` whenever it runs a reasoning parser (checked in an image built from `0baa4aa7` against a stub streaming endpoint, 2026-09-26: `completion_tokens` 50 with `reasoning_tokens` 30 is recorded as `total_output_tokens` 80). So a total below `max_tokens` is not a cut-off; a total equal to it is one (a reply that finished with completion plus reasoning tokens landing exactly on `max_tokens` is not told apart); above it, the reply's completion tokens are (total + the answer's tokens) / 2, the answer (`choices[0].turns[0]`) counted with the run's tokenizer (`--tokenizer`, else the served model's id), and the reply is a cut-off when that is within 16 tokens of `max_tokens` (the tolerance covers the reasoning delimiters, counted as neither reasoning nor answer, and retokenization). The rule takes `reasoning_tokens` from the server's usage, as vLLM reports them; for a server that streams reasoning without them, litellm counts the reasoning with its own tokenizer (the in-image check above: `completion_tokens` 50 with 30 reasoning words and no details was recorded as 81), which the rule does not assume. `replies_cut_off_uncounted` counts rows gbench could not check: no output token count (LiveBench records -1 when the server returned no usage), several turns with a total at or above `max_tokens`, or a single turn above it when the run's tokenizer cannot be loaded; any makes the run not comparable. `replies_cut_off_note` states the rule. A reply at the official cap is the protocol's outcome and is not counted |
| `empty_replies`, `empty_replies_note` | LiveBench's `api_error` rows for replies the model finished with no answer text (after its reasoning, or none at all). A reply whose content is empty with a `finish_reason` other than `length` (empty at `length` is `token_exhaustion`, above) makes LiveBench's litellm path raise `No message returned from litellm (finish_reason=stop)`, and LiveBench records the row with `eval_status: api_error`, `error: Exception`, that `error_msg` and the answer `$ERROR$`, and scores it 0 (checked in an image built from `0baa4aa7` against a stub streaming endpoint, 2026-09-26: reasoning with empty content and `finish_reason` `stop`, and an immediate end of turn, both give that row). They are the model's outcome, scored 0 as LiveBench scores them, and change neither the status nor `leaderboard_comparable`. litellm (1.102.1 in the image) also reports a stream with no `finish_reason`, or with vLLM's `abort` (sent when the server aborts its in-flight requests, as at shutdown), as `stop`, so the row cannot tell such a request apart and it is counted here; a server that stops mid-run also fails the requests after it, which are counted in `failed_requests`. `empty_replies_note` (present when there are any) says this |
| `server_aborted_requests` | Requests the served vLLM server ended itself during the run (for example while draining at shutdown): the rise of vLLM's Prometheus counter `request_success_total` with `finished_reason` `abort` or `error` for the served model (every model when no line names it), read from the endpoint's `/metrics` before and after the run. LiveBench records such a request as an empty (or cut-short) reply, so up to that many answers are harness failures that cannot be told apart: any sets the status to `completed_with_errors`, makes the run not comparable and withholds `accuracy` and every category score (`dimension_scores_withheld`). The count is server-wide for that model, so other clients' aborted requests on a shared server are counted too. `null` when the endpoint exposes no vLLM `/metrics`, in which case an aborted request cannot be told apart from an empty reply |
| `failed_requests` | LiveBench's other `api_error` rows in the answer rows gbench found: requests that failed at the served endpoint (a connection error, an HTTP error status, a timeout, a stream that ends in an error; in the same in-image check an HTTP 500 and a stream ending in `finish_reason` `error` gave `InternalServerError` and `MidStreamFallbackError` rows) after the retries of LiveBench's litellm path (LiveBench scores those questions 0). They are harness failures, not the model's answers: any sets the status to `completed_with_errors`, makes the run not comparable and withholds `accuracy` and the score of each category with one (`dimension_scores_withheld`); `failed_requests_by_category` counts them per category (from the answer file's `data/live_bench/<category>/` path). gbench does not re-ask them: LiveBench's own `--retry-failures` re-asks every `$ERROR$` answer, the model's empty replies included (`common.filter_questions`), which would give the model a second try at answers that are its outcome. Run the suite again with the endpoint up for the whole run |
| `leaderboard_comparable`, `leaderboard_comparable_reason` | `true` only for a pinned release (the default or a date in `GBENCH_LIVEBENCH_RELEASE`, not `latest`), all 6 categories scored, the full question set, the temperature of LiveBench (or of the model config), the token-limit policy's budget (an official cap not lowered to fit the served window), no reply cut off by another budget (and, when the budget is not the official cap, LiveBench's answer rows found and every row checked), no request failed at the endpoint, a model config reproduced in full (reasoning switch included), an image verified as built from the pinned commit, and instruction_following tasks answered in equal numbers, verified from its answer rows (see "Recorded patch"); otherwise the reason lists which of those failed |

## Containers and network

Every container is started from the local LiveBench image:

| Container | What runs in it | Network |
| --- | --- | --- |
| patch check | the sha256 of the patched `gen_ground_truth_judgment.py` | `--network none` |
| model-config lookup | LiveBench's `get_model_config` for the served model | `--network none` |
| the run | `run_livebench.py` (question download from Hugging Face, inference against your endpoint, grading, which executes the model's code for `coding`) and `show_livebench_result.py` | docker's bridge network, relies on the host rule |

The run needs the endpoint and Hugging Face, so it cannot run without a network; it never uses the
host network. It gets `--add-host host.docker.internal:host-gateway`, and an endpoint on this host
(`localhost`, `127.0.0.0/8`, `0.0.0.0` or `[::1]` in the URL) is passed to LiveBench as
`--api-base http://host.docker.internal:<port>/...`, so it must listen on an address the docker
bridge reaches: vLLM as gbench starts it listens on every interface; a server bound to
`127.0.0.1` only does not. Any other endpoint URL is passed as given. On a cloud VM the run relies
on the host rule that keeps a bridge network away from the metadata server at `169.254.169.254`
(which hands out the VM's service-account token):

```bash
sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP
sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save   # keep it
```

Before the run starts, gbench's guard (`require_metadata_blocked`, see
[toolchains](toolchains.md#cloud-vms-the-metadata-server)) starts one short-lived container of the
image on the bridge network that opens and closes a TCP connection to `169.254.169.254:80` (no
request is sent); if it opens, the suite hard-errors (`infra_required`) naming the rule, and a
probe that gives no clear answer (neither `python3` nor `bash` in the image, a docker error or
timeout) is a hard error too. For an endpoint on this host, gbench also checks from a container of
the image on that network that a TCP connection to `host.docker.internal:<port>` opens, and
hard-errors (`infra_required`) otherwise. gbench never runs sudo or changes the host's firewall
itself. The vLLM `/metrics` reads (`server_aborted_requests`) are made from the host, with the URL
as given.

The run loads its questions from the Hugging Face Hub (`--question-source huggingface`, in both
`gen_api_answer.py` and `gen_ground_truth_judgment.py`; the NLTK and spaCy data grading would
download is baked into the image), so it needs working DNS on the bridge network: the same probe
container resolves `huggingface.co` and the endpoint's host when the URL names one (not an IP
address), and a name that does not resolve stops the suite before the run (`infra_required`),
naming the rule. On GCE the VPC DNS resolver is `169.254.169.254` itself, which is why the rule
spares port 53 (a plain `-j DROP` for the address cuts DNS in every container). A run that writes
no `all_groups.csv` and shows a failed host-name lookup in its output reports no score
(`infra_required`, naming the lookup), and so does a run in which every request failed at the
endpoint (every answer row an `api_error` other than an empty reply): LiveBench would score them
all 0, which measures the harness, not the model.

The run container has general internet access on the bridge network, as upstream's
`run_livebench.py` has on the machine it runs on; only the metadata server is blocked, so the
model's code that the `coding` grading executes can reach the internet as it can upstream.

## Requirements

- **Docker** + the **`gbench-livebench:0baa4aa7` image** (built above from a LiveBench checkout at
  the pinned commit, with gbench's recorded patch applied by the Dockerfile). Hard-errors
  (`infra_required`) with the build steps if either is missing, or if the image's
  `gen_ground_truth_judgment.py` is not the patched file (an image built before the patch).
- **git-lfs** (apt `git-lfs` or pip `git-lfs`) to check out LiveBench's LFS `*.json` data.
- A reachable OpenAI-compatible **endpoint** for the model under test (gbench's normal target;
  reached from the run's container on docker's bridge network, at `host.docker.internal` when
  it is on this host).
- **On a cloud VM, the host rule** `sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP`,
  made persistent (see "Containers and network").
- No API key, no LLM judge (LiveBench scores against ground truth).
- A loadable tokenizer for the served model whenever the run's `max_tokens` is not the official
  cap: gbench counts an answer's tokens with the run's tokenizer to tell a reply cut off after its
  answer began (see `replies_cut_off_by_budget`). That is `--tokenizer`, else the served model's
  id (loaded with `transformers`); pass `--tokenizer` when the served id is not a Hugging Face
  tokenizer. When no tokenizer can be loaded, rows whose `total_output_tokens` exceed
  `max_tokens` are uncounted and the run is not comparable.

## Notes / caveats

- **Runtime:** the full 6-category run generates + grades thousands of questions; it is slow.
  Raise `--suite-timeout`.
- **Endpoint routing:** LiveBench reaches the model through LiteLLM's OpenAI-compatible route
  (`--api-base` plus the `local` provider of gbench's model config), and sends
  `GBENCH_LIVEBENCH_MODEL` (default: the served model id gbench passes) as the request's
  `model`. Set it only to another name your server serves, without a provider prefix (it is
  checked against the endpoint's model list).
- `leaderboard_comparable` is set when all 6 default categories score on a pinned release over
  the full question set (no `--eval-limit`) at the temperature of LiveBench (or of the served
  model's LiveBench config), with the token-limit policy's budget (an official cap that fits the
  served window), no reply cut off by another budget, no request failed at the endpoint, the model
  config reproduced in full, an image verified as built from the pinned commit, and
  instruction_following tasks answered in equal numbers (verified from its answer rows). It is
  **not** gated on `--thinking`: LiveBench's leaderboard includes reasoning/thinking models, so a
  `--thinking` run is still a faithful measurement (a model config's own reasoning switch must
  match `--thinking`, though). When it is `false`, `leaderboard_comparable_reason` names the unmet
  conditions.
- **Sampling:** LiveBench's own (`OFFICIAL_SAMPLING` in
  `livebench.py`), or the served model's LiveBench config (see above), except the temperature.
  `livebench/gen_api_answer.py` `get_answer` (LiveBench@`0baa4aa7`) uses temperature 0, or the
  question's `required_temperature`, unless `--force-temperature` is given.
- **Temperature: 0.0 without `--thinking`, 1.0 with it**, gbench's rule for every suite
  (`base.resolve_temperature`). `GBENCH_LIVEBENCH_TEMPERATURE`, then `--temperature`, replaces it
  in either mode. The value is always written as the one-off config's temperature:
  `chat_completion_openai` / `chat_completion_litellm` start from `get_answer`'s temperature and
  then apply the config's `api_kwargs`, so it reaches every request, and no `--force-temperature`
  is passed, so no question with a `required_temperature` stops the run (LiveBench refuses, with
  an assertion, any such question when `--force-temperature` is given). A run is not
  leaderboard-comparable when the value differs from the served model's LiveBench config
  temperature (1.0 for `gemma-4-31b-it`) or, without a model config, from `get_answer`'s 0 (the
  question sets gbench has read carry no `required_temperature`); the reason names the
  temperature and its source.
