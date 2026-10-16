# ojbench setup

## What it measures

OJBench (`He-Ren/OJBench_testdata`): 232 competition problems from NOI and ICPC, each posed once
for **Python** and once for **C++** (464 prompts). The model writes a complete stdin/stdout program;
it is Accepted iff **every** testcase passes within the problem's time/memory limits. Judging uses
the official `ojbench` library over the DMOJ sandbox; this is the only correct scorer (a
substring/heuristic check cannot verify program behaviour).

## Dataset

Prompts: `prompts/full.jsonl` of `He-Ren/OJBench_testdata`, pinned to HF commit
`61cf9986f22c25d08e1657b03742124099c74353`. The testcases (`NOI/`, `ICPC/`) come from the snapshot
you mount (see Requirements); download it at the same revision.

## Protocol

- The dataset prompt already embeds the required response format, so it is sent **verbatim** as
  the single user turn; the response has its thinking tags stripped before judging.
- **Eight candidate solutions per problem** (the paper's protocol, arXiv 2506.16395 Section 3.1).
  Each candidate is judged on its own. `GBENCH_OJBENCH_SAMPLES` overrides the count;
  `--attempt-count` above 8 raises it.
- **Sampling**: the paper samples with "the hyper parameters officially recommended by each model,
  including temperature, top_k, and top_p" (Section 3.1). The suite reads them from the served
  model's own `generation_config.json` (`served_model_recommended_sampling` in `ojbench.py`): the
  run's `--tokenizer`, else the served model name, as a local directory holding the file or a
  Hugging Face model id read through the HF cache. It sends the `top_p` and `top_k` the file sets
  (a field it does not set is not sent) with and without `--thinking`; for Gemma 4 the file sets
  **temperature 1.0, top_p 0.95, top_k 64**. The temperature sent is gbench's rule for every suite
  whatever the file says (`base.resolve_temperature`): **1.0** with `--thinking` (for Gemma 4 the
  recommended value), **0.0** without. If the file cannot be read the suite
  hard-errors (`infra_required`) rather than send another model's settings. `--temperature`, or
  `GBENCH_OJBENCH_TEMPERATURE` for this suite alone (it takes precedence), overrides the
  temperature in either mode; the result records the official values and whether the run matched them under
  `official_sampling`, and the values read and their source under `recommended_sampling`.
- **max_tokens, per mode** (Section 3.1, applied by gbench's token-limit policy,
  `base.resolve_token_budget`):
  - `--thinking`: "For reasoning-oriented models ... we set the maximum number of tokens
    (max_tokens) to 64k", upstream's explicit reasoning-model setting: every request carries
    65536, or the run's `--max-output-tokens` when that is larger (a thinking run is never capped
    below it); a request whose prompt leaves less room in the served context window is sent less.
  - no-think: "For non-reasoning-oriented models, we set max_tokens to the default context length
    of the model": no request carries a `max_tokens`, so the server grants the rest of the context
    window, and `--max-output-tokens` is not applied.

  `max_output_tokens_note` states the mode's choice; `official_sampling` records it
  (`reasoning_settings: true`, `max_tokens` 65536 and `max_tokens_sent` with `--thinking`;
  `send_max_tokens: false` without).
- Metrics per language, as in the paper's Table 2 and the leaderboard:
  - **Pass@1** = mean over the candidates of "Accepted";
  - **Pass@8** = unbiased pass@k estimator with k = 8 (with exactly 8 candidates: the share of
    problems with at least one Accepted candidate);
  - **pass rate by difficulty** (easy / medium / hard), Pass@1.

## Reading the result

- `accuracy` (headline) = **C++ Pass@1**, the column the official leaderboard ranks by.
  `correct_answers` / `total_questions` count the C++ generations, so they match it.
- `per_language` = `{cpp, python}`, each with `pass_at_1`, `pass_at_8` (when every problem has at
  least 8 scored candidates), `pass_rate_by_difficulty`, `problems`, `generations`, `correct`.
  A generation left out of the score (no judge verdict, or no reply from the served endpoint) is
  counted in none of them.
- `scoring_excluded` = every generation left out of the score; `judge_no_verdict` = those the
  judge returned no verdict for; `request_harness_errors` = those whose request got no reply from
  the served endpoint after gbench's retries, by kind (`endpoint_failure`, `endpoint_timeout`,
  `over_context`: the prompt did not fit the served context window). Endpoint failures are harness
  errors, never the model's outcome: named in `status_reason`, status `completed_with_errors`. A
  run where no generation was scored has status `failed` and reports `accuracy` null with
  `accuracy_withheld_reason`; a run whose C++ generations were all left out of the score reports
  `accuracy` null with `accuracy_withheld_reason` too (the Python figures stay in `per_language`).
- `category_accuracy` = pooled Pass@1 per language; `pooled_pass_at_1` = both languages together
  (not a published number); `attempts` = gbench's avg@k / pass@k / pass^k over the C++
  generations when the headline is C++'s (`language: "cpp"`), with base's block over both
  languages kept as `attempts_pooled_both_languages`. Each judged trace carries the judge's
  detail as `execution`: the overall `verdict` (AC, WA, TLE, RTE, ...), the per-scale verdicts
  (`scale_verdicts`) and the first 20 entries of `detailed_results`.
- `temperature`, `temperature_source`, `official_sampling`: the temperature used, where it came
  from, and the recommended sampling. `recommended_sampling`: the fields read from the served
  model's `generation_config.json` and the file they came from.
- `leaderboard_comparable` is `True` only when none of these apply (each is named in
  `leaderboard_comparable_reason`):
  - `subset run (--eval-limit N)`;
  - fewer than 8 candidates per problem;
  - a temperature, top_p or top_k other than the served model's recommended ones (1.0 / 0.95 / 64
    for Gemma 4), or any other sampling field, decoding penalties
    (`GBENCH_REPETITION_PENALTY`, `GBENCH_FREQUENCY_PENALTY`, `GBENCH_PRESENCE_PENALTY`) included;
  - a `--thinking` run with a per-request `max_tokens` below 65536 (the paper gives reasoning
    models 64k; a prompt that left less room in the served window was sent less);
  - a no-think run that sent a `max_tokens` (the paper gives non-reasoning models their default
    context length, so none is sent);
  - replies that stopped (finish reason `length`) at a limit other than the paper's: without
    `--thinking`, the end of the served context window; with it, a `--max-output-tokens` above 64k
    or a `max_tokens` the window lowered (`replies_cut_off_by_budget`, and
    `replies_cut_off_in_reasoning` for those with no answer text). A `--thinking` reply that
    stopped at 65536 is the paper's own outcome;
  - generations the judge returned no verdict for (`judge_no_verdict`);
  - generation requests that got no reply from the served endpoint (`request_harness_errors`);
  - no C++ problem in the run (only possible with a tiny `--eval-limit`; the headline is then
    pooled), or no C++ generation scored (the headline is then withheld).

## Why the judge runs in a container

gbench does the model rollout itself (HTTP against the served model), then hands the generated
programs to a local image for judging. The judge (`ojbench` + the DMOJ `dmoj` judge-server + PyPy3
+ g++) is containerized because DMOJ's `cptbox` sandbox does **not** build on Python 3.12 (its
bundled Cython C uses `PyLongObject.ob_digit`, removed in 3.12) and the gbench serving env is 3.12.
On the image's Python **3.11** it compiles cleanly. Nothing DMOJ-versioned touches the main env.

> **DMOJ version is load-bearing.** OJBench's `judger.py` calls `problem.cases()`, an API present
> only on the DMOJ judge-server commit its README pins (`f098cd3a49a60186d1fadde5132329ec5f4f2213`).
> The PyPI `dmoj==4.1.0` that `pip install -e OJBench` would pull has **no** `Problem.cases()`, so
> every executable submission raises `AttributeError` inside the judge worker and NOTHING gets
> graded - a silent all-0. The Dockerfile therefore installs DMOJ from that exact commit and asserts
> `hasattr(Problem, "cases")` at build time. The runner also treats a submission the judge returned
> **no verdict** for as a harness failure (left out of the score, `judge_no_verdict`), and
> hard-errors if the judge graded
> nothing at all - a judge failure can never surface as a clean 0%.

## Container and network

The suite's only container is the judge of the local `gbench-ojbench:5e94480b` image, which compiles and
runs the model's programs against the mounted testdata. It runs with `--network none`: the
testdata is bind-mounted read-only, the records come in and the verdicts go out through a mounted
work directory, and the judge, PyPy3 and g++ are in the image, so it needs nothing from the
network. It therefore needs no host firewall rule and gbench's
metadata-server guard has nothing to probe; there is no host name to resolve, and the model's
programs have no internet access. The model's replies are generated by gbench itself, outside any
container.

## Requirements (else the suite hard-errors `infra_required`)
- **Docker** (a reachable daemon). The judge run needs `--cap-add SYS_PTRACE` and
  `--security-opt seccomp=unconfined` (gbench passes both) so the inner cptbox ptrace+seccomp
  sandbox works; Docker's default seccomp profile blocks installing the inner filter. It runs
  with `--network none` (see "Container and network").
- **The local image**, built once (gbench never pulls it):
  ```bash
  docker build -t gbench-ojbench:5e94480b -f docker/ojbench.Dockerfile docker
  ```
  Override the name with `GBENCH_OJBENCH_IMAGE` if you retag it. The image clones He-Ren/OJBench at
  commit `5e94480b1e135b98855cf5bc81213c256aff5b17` and installs `ojbench` + `dmoj` + PyPy3 + g++
  internally. Its pip installs (and the judge-server's build environment) resolve against
  `docker/ojbench.constraints.txt` (the installed set of the image built 2026-09-11, plus Cython
  3.3.0 for the judge-server build), and the build fails if anything else is installed
  (`docker/pinned_pip_check.py`); the image records its Python and Debian packages in
  `/opt/gbench`.
- **Testdata**, bind-mounted read-only: set `GBENCH_OJBENCH_TESTDATA` (legacy alias
  `OJBENCH_TESTDATA`) to a snapshot of `He-Ren/OJBench_testdata` containing `NOI/` and `ICPC/`
  (~7.85 GB). It is NOT baked into the image:
  ```bash
  hf download He-Ren/OJBench_testdata --repo-type dataset \
      --revision 61cf9986f22c25d08e1657b03742124099c74353 --local-dir <dir>
  ```
- `huggingface_hub` in the gbench env (base install) to read the prompts.
- **The served model's `generation_config.json`**: pass `--tokenizer` with the model's Hugging Face
  id (read through the HF cache; a gated model needs the usual HF token) or a local directory that
  holds the file. Without it the suite hard-errors (`infra_required`).

## Configuration

| Env var | Purpose |
| --- | --- |
| `GBENCH_OJBENCH_TESTDATA` | testdata snapshot (required; legacy alias `OJBENCH_TESTDATA`) |
| `GBENCH_OJBENCH_SAMPLES` | candidates per problem (positive integer, default 8) |
| `GBENCH_OJBENCH_TEMPERATURE` | per-suite temperature (overrides `--temperature`; default 0.0, or 1.0 with `--thinking`) |
| `GBENCH_OJBENCH_IMAGE` | judge image tag (default `gbench-ojbench:5e94480b`) |
| `GBENCH_OJBENCH_JUDGE_TIMEOUT_S` | outer timeout for the judge container |

## Run
```bash
export GBENCH_OJBENCH_TESTDATA=/path/to/OJBench_testdata
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals ojbench --thinking \
       --max-output-tokens 65536 --sandboxes 8
```
For a no-think run, drop `--thinking`: no `max_tokens` is sent, so each reply may use the rest of
the served context window (`--max-output-tokens` is required on the command line but not
applied), and the temperature is 0.0 (the recommended 1.0, which comparability needs, only with
`--temperature 1.0`); top_p and top_k are the recommended ones either way. With `--thinking`, every request
carries 65536, or `--max-output-tokens` when that is larger.
A full run is 464 prompts x 8 candidates = 3712 generations. `--sandboxes` bounds concurrent
generation and maps to the judge `num_workers` inside the container. Add `--eval-limit N` for a
smoke run (N problems for the whole suite, stratified across the two languages, never N per
language; reported as not leaderboard-comparable). The run logs its plan on its first line and records it as `eval_limit_plan`.

## Crash resilience

Upstream `ojbench.judger.judge_jsonl_data` did `result_queue.get()` with no timeout, so if a judge
worker crashed **hard** while executing a submission (a C-level cptbox death, no Python traceback),
its result was never queued and the whole judge **hung forever**. The image bakes a patch
(`docker/ojbench_patch_judger.py`): a bounded `get()` plus a break once **every** worker has exited
(a slow-but-alive worker keeps waiting; only all-dead stops the loop). A crashed-worker submission
comes back without a verdict; all others still get real verdicts; the judge never deadlocks. `OJBENCH_GET_TIMEOUT_S`
(seconds, a positive integer; default 120), set on the host, is forwarded into the judge container
and tunes the poll interval (any other value stops the suite before generation), and the host wraps
the whole container in
`GBENCH_OJBENCH_JUDGE_TIMEOUT_S` as an outer backstop. NOTE: a small number of submissions can crash
the cptbox sandbox during execution (root cause not yet isolated - the sandbox self-tests pass for
pypy3/C/C++, so it is submission-specific). gbench excludes a generation with no verdict from the
numerator and denominator (`judge_no_verdict`) and names the count in
`leaderboard_comparable_reason`, rather than scoring it as a wrong answer.

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered problems
(both languages) (round-robin: positions I, I+N, I+2N, ...), selected before the `--eval-limit`
draw, which is then taken within the shard (N shards run up to N x `--eval-limit` problems). The
result records `shard`, `shard_index` and `shard_count`, and a shard of more than one is not
leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`); `gbench
--merge` combines the shards' results as docs/distributed-gbench.md describes: it re-counts the
union of their traces, and refuses a suite whose headline is not a correct / scored count over
them unless the suite defines its own merge.
