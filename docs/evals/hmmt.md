# hmmt

HMMT February 2026 (Harvard-MIT Mathematics Tournament), the 33 final-answer problems MathArena
publishes as [`MathArena/hmmt_feb_2026`](https://huggingface.co/datasets/MathArena/hmmt_feb_2026)
and lists among its current final-answer competitions (README of
[eth-sri/matharena](https://github.com/eth-sri/matharena) at commit `b89f2f0a`). Answers are exact
competition values (integers, fractions, radicals, closed forms). gbench runs MathArena's whole
protocol, including its own grader.

## Dataset

| | |
| --- | --- |
| dataset | `MathArena/hmmt_feb_2026`, split `train` |
| revision | `02fba4f74d8e68e73e66a02d540fd979c05c274c` (pinned in `matharena_common.py`) |
| N | 33 |
| category | the row's `problem_type` labels joined with `/` (for example `Algebra/Number Theory`) |
| license | CC BY-NC-SA 4.0 |

## Protocol

Everything below is MathArena's, from eth-sri/matharena commit
`b89f2f0ad64ced464d2944f08c3c0aaeaa0df64b`.

**Prompt.** The instruction of `configs/competitions/hmmt/hmmt_feb_2026.yaml`, a blank line, then
the problem, as one user message with no system prompt (`runner.py` appends `\n\n{problem}` to an
instruction without a `{problem}` slot; `PureModelSolver.build_query` fills it). The prompt is the
same with and without `--eval-thinking`:

```
Put your final answer within \boxed{}.

<problem>
```

**Runs.** 4 runs per problem (MathArena's `--n` default), always: the suite runs
`max(4, --attempt-count)` attempts. The headline `accuracy` is the mean accuracy over all runs,
which is MathArena's score. The `attempts` block also carries pass@k and pass^k.

**Last chance.** After each reply, MathArena's runner checks the final assistant message with
`extract_answer(content, True)`. If it finds no parseable `\boxed{}` answer, the conversation is
sent back once (reasoning dropped, as MathArena's client does) followed by MathArena's last-chance
prompt, verbatim from `Runner._build_last_chance_prompt`:

```
Your last message does not provide a final answer in a way that follows the formatting instructions.
Please based on the conversation history, report the final answer again within \boxed{}.
If you did not find the answer, please use \boxed{None}.
Do not reason about the problem again or use tools, simply try to extract the final answer from the previous reasoning.
Boxed answers in thinking/reasoning stages will be ignored; only the final response message is considered.
```

The reply to it is what gets graded. It is sent with the run's own settings (sampling, thinking
flag, output budget), as MathArena reuses the solver's client arguments.

**Sampling.** MathArena runs each model with the settings of its
`configs/models/<provider>/<model>.yaml`, which carry the provider's recommended sampling and the
model's `max_tokens` (for example `qwen/qwen3.5_27b.yaml`: temperature 1.0, top_p 0.95,
presence_penalty 1.5, top_k 20, min_p 0.0, repetition_penalty 1.0, max_tokens 192000). gbench
applies the same rule to the served model before generation, on the first request and on the
last-chance request, with and without `--eval-thinking` (`hmmt.served_model_sampling`, shared by
hmmt and aime):

- **MathArena ships a config for the served model.** `hmmt.MATHARENA_MODEL_CONFIGS` holds the 59
  configs at `b89f2f0a` whose `other_params.huggingface_id` is set (the open-weight models a local
  server can serve), each with its git blob sha. The served model name and the run's `--tokenizer`
  are matched against `huggingface_id`, case-insensitively (the full id, else its last path
  component; two configs naming one id, as `moonshot/k25` and `k26` both name
  `moonshotai/Kimi-K2.5`, are told apart by their `model` field). Its fields are sent as
  MathArena's `APIClient` sends them: the sampling fields (top level or in `extra_body`),
  `reasoning_effort`, and `max_tokens` (a temperature, top_p or top_k of `null` is not sent, nor a
  temperature for a model name containing o1, o3 or o4; an `api: vllm` config sends only the
  fields MathArena's offline vLLM path passes to `SamplingParams`). `max_tokens` is the model's own setting, for reasoning too, applied
  by gbench's token-limit policy: exactly without `--eval-thinking`, and never below
  `--max-output-tokens` with it; a config with no `max_tokens` sends none. The config's reasoning
  switch (`extra_body.chat_template_kwargs.enable_thinking`, or GLM's `extra_body.thinking`) must
  match `--eval-thinking`; OpenRouter `provider` routing and a `harness` (these competitions set no
  `allow_harness`) do not apply to a local server; the fields MathArena's runner and `APIClient`
  consume themselves (`model`, `api`, streaming, retries, context limits, `batch_processing`, which
  picks a provider's batch API and never reaches the request) are not request fields; any other
  field is not reproduced and makes the run not comparable. The result records the config under
  `matharena_model_config`. MathArena's offline vLLM path (`api: vllm`) renders the model's
  default chat template (no `chat_template_kwargs`), while gbench's requests carry the run's
  thinking switch; `matharena_model_config.chat_template` says so for such a config (a template
  that does not read `enable_thinking` renders the same either way).
- **MathArena ships none** (no Gemma model has one): the served model's own `generation_config.json`
  (the run's `--tokenizer`, else the served model name, a local directory or a Hugging Face model
  id, as ojbench reads it) gives the `temperature`, `top_p` and `top_k` it sets (a field it does not
  set is not sent). Gemma 4's is **temperature 1.0, top_p 0.95, top_k 64**
  (`matharena_common.SAMPLING`). If the file cannot be read the suite hard-errors
  (`infra_required`) rather than send another model's settings. MathArena ships no config for the
  model, so it defines no `max_tokens` for it; gbench sends `--max-output-tokens` (its house
  budget, base's token-limit policy). This is gbench's choice for a model MathArena never ran, not
  an upstream setting: the token policy's "none when upstream sends none" applies to a config that
  sets no `max_tokens`, and a reply cut off at the house budget makes the run not comparable
  (below).

**Temperature: 1.0 with `--thinking`, 0.0 without**, whatever the config or
`generation_config.json` says: gbench's rule for every suite (`base.resolve_temperature`). For
Gemma 4 the thinking run's 1.0 is its recommended value; a run whose temperature is not the
config's or the file's (every no-think run of Gemma 4, or a thinking run of a model whose
MathArena config sets 0.6, say) is not comparable, and the reason names the temperature and its
source (`default` or `thinking`). The other sampling fields stay
the config's or the file's, on the last-chance request too.

`--temperature`, or `GBENCH_HMMT_TEMPERATURE` for this suite alone, overrides the temperature in either mode; the
result records the official values and whether the run matched them under `official_sampling`, and
the fields sent and their source under `recommended_sampling`.

**Grading.** MathArena's `grader.extract_and_grade` on the final assistant message, with the
competition config (`strict_parsing: false`: the last `\boxed{}` answer, else the last integer in
the message), comparing with `parser.check_answers` (sympy equivalence). The code is MathArena's own
`parser.py`, `parse_manual.py`, `grader.py`, `utils.py` and the two `tools/` modules `grader.py`
imports, vendored byte for byte in `gbench/third_party/matharena_src/` (with MathArena's
MIT `LICENSE`); `matharena_common.py` checks each file's git blob sha before every use.

It runs in the local `gbench-matharena:b89f2f0a` image because MathArena pins
`antlr4-python3-runtime==4.11`: its parser falls back to sympy's LaTeX parser, which refuses any
other antlr runtime, and the gbench environment carries 4.13. Without that runtime two of the 33
gold answers (`(3+\sqrt{6})^{-1/3}` and `50(1 - \frac{1}{2^{101} - 1})`) do not even match
themselves. Before generation the suite grades three known answers in the image (one of them needs
the LaTeX parser) and stops if any comes out wrong.

## Reading the result

- `accuracy`: mean accuracy over all runs (MathArena's score), leaving out the runs that are
  harness errors: a generation or last-chance request that got no reply from the served
  endpoint (failed, timed out, or over the context window) and a response MathArena's grader
  could not grade. Each is counted (`scoring_excluded`; the grading errors also under
  `judge_failures`) and named in `status_reason` (status `completed_with_errors`), never scored
  as the model's answer; when no run is scored, `accuracy` is null with
  `accuracy_withheld_reason`.
- `category_accuracy`: per `problem_type` label, over the same runs.
- `dataset`, `competition` (`HMMT February 2026`), and `matharena`: the MathArena commit, the
  competition config, its instruction, the grader image, the last-chance prompt and
  the sampling reference.
- `last_chance`: how many runs were reprompted, and how many of those reprompts got no reply
  (each such run carries its `harness_error`).
- `grading`: responses MathArena's grader could not grade (`errors`) and the grader's warning
  levels per run (`warnings`, 0 = none to 3 = major).
- Per run in `sample_traces`: `matharena_answer` (the parsed answer), `matharena_warning`, and,
  when reprompted, `matharena_last_chance` (the reply to the reprompt).
- `empty_responses` / `failed_requests` count first replies that came back empty, before the
  last chance; an empty first reply is reprompted and graded like any other.
- `temperature`, `temperature_source` and `official_sampling` (the official values, their
  source and `temperature_matches_official`); `recommended_sampling` (the fields sent and their
  source: MathArena's config of the served model, or its `generation_config.json`);
  `matharena_model_config` (the config, its blob sha, the name that matched, the fields that do not
  apply to a local server and those not reproduced; `null` when MathArena ships none).
- `leaderboard_comparable` is `true` only when none of these apply; otherwise
  `leaderboard_comparable_reason` names each one:
  - fewer than 4 runs per problem;
  - a temperature, top_p or top_k other than the served model's recommended ones (1.0 / 0.95 / 64
    for Gemma 4, which MathArena would use), or decoding penalties;
  - a sampling field MathArena does not send for the model, or a `reasoning_effort` other than its
    config's;
  - a field of MathArena's config of the served model that the run cannot reproduce (its reasoning
    switch set the other way, say);
  - the config's `max_tokens` not sent in full (a prompt left less room in the served context
    window; serve a larger `--max-model-len`);
  - a reply, first or last-chance (each tested against the `max_tokens` its own request carried),
    cut off by a limit other than the config's `max_tokens`
    (`replies_cut_off_by_budget`, and `replies_cut_off_in_reasoning` for those with no answer
    text): for a model MathArena ships no config for, as every Gemma model, any reply that reached
    `--max-output-tokens` (raise it);
  - an `--eval-limit` subset;
  - generation or last-chance requests that got no reply (harness errors, left out of the score;
    MathArena retries until it has one);
  - responses the grader could not grade (harness errors, left out of the score).

## Requirements

- A running OpenAI-compatible `/v1` endpoint.
- **Docker** and the locally built grader image (about 0.3 GB; gbench never pulls it). From the
  gbench repository root:

  ```bash
  docker build -t gbench-matharena:b89f2f0a -f docker/matharena.Dockerfile docker
  ```

  The build needs network access to PyPI; the image holds only pinned Python packages (sympy,
  antlr4-python3-runtime 4.11, regex, loguru, PyYAML, axiom-axle and their dependencies). gbench
  streams the vendored grader into it at run time and runs it with `--network none`. The suite
  **hard-errors** (`infra_required`) if Docker or the image is missing or fails the self-check.
- `GBENCH_MATHARENA_IMAGE` selects a differently tagged image (default `gbench-matharena:b89f2f0a`).
- **The served model's `generation_config.json`** (for its recommended sampling, when MathArena
  ships no config for it, as for every Gemma model): pass `--tokenizer` with the model's Hugging
  Face id (read through the HF cache and downloaded when not
  cached; a gated model needs the usual HF token) or a local directory that holds the file. The
  suite **hard-errors** (`infra_required`) when it cannot be read.
- `pyyaml` (installed with `datasets`) to read the vendored competition config.
- Network access to the Hugging Face Hub on first run (the dataset is public; no token needed).

## Run

```bash
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals hmmt --eval-thinking \
       --max-output-tokens 65536
```

- `--max-output-tokens` is required on the command line and, for a model with no MathArena config
  (every Gemma model), sets the per-request budget, for the
  first reply and for the last-chance reply. With `--eval-thinking` (recommended) give the
  reasoning room: MathArena's model configs allow 64k to 192k tokens.
- 4 runs per problem are always made (132 generations, plus one last-chance turn for each reply
  without a boxed answer). A larger `--attempt-count` is honoured.
- `--eval-limit N` runs a stratified subset of N problems (flagged in the result); MathArena's 4
  runs per problem are attempts, not more problems. The run logs its plan on its first line and records it as `eval_limit_plan`.
- The sampling is MathArena's for Gemma 4 (top_p 0.95, top_k 64) with and without
  `--eval-thinking`, at gbench's temperature (1.0 with `--eval-thinking`, 0.0 without); a model
  MathArena ships a config for gets that config's settings (at gbench's temperature whatever the
  config says), and its
  `max_tokens` replaces `--max-output-tokens` without `--eval-thinking` (with it, the larger of the
  two is sent). `GBENCH_HMMT_TEMPERATURE` overrides the temperature for this suite (it takes
  precedence over `--temperature`, which sets it for the whole run); either makes the run
  non-comparable unless it is the recommended one (1.0 for Gemma 4).

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered problems
(sorted by problem_idx) (round-robin: positions I, I+N, I+2N, ...), selected before the
`--eval-limit` draw, which is then taken within the shard (N shards run up to N x `--eval-limit`
problems). The result records `shard`, `shard_index` and `shard_count`, and a shard of more than
one is not leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`);
`gbench --merge` combines the shards' results as docs/distributed-gbench.md describes: it
re-counts the union of their traces, and refuses a suite whose headline is not a correct / scored
count over them unless the suite defines its own merge.
