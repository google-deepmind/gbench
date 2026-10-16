# nestful

NESTFUL (IBM, [IBM/NESTFUL](https://github.com/IBM/NESTFUL)): nested function calling, where the
output of one call feeds the arguments of the next. The model must write the whole sequence of
calls for a question, referencing earlier outputs as `$<label>.<field>$`. gbench runs IBM's own
pipeline (`src/eval.py` -> `src/instruct_data_prep.py` -> `src/scorer.py`) with its DeepSeek
prompt/parser pair, the path upstream uses for a chat model.

## Dataset (pinned)
`data_v2/nestful_data.jsonl` (1861 rows, the file `run.sh` evaluates) from an IBM/NESTFUL checkout
at commit `fc2c4123e73500a56185a5fb354f05d1c8b4890c`, the only data source, as upstream's `run.sh`
uses it (`--dataset data_v2/nestful_data.jsonl`, and `--executable_func_dir
data_v2/executable_functions` for the Win Rate). Before a run gbench checks the git blob ids of
`data_v2/nestful_data.jsonl`, `executable_functions/basic_functions.py` and
`executable_functions/func_file_map.json` against that commit: a data file that differs is a hard
error; function files that differ make the Win Rate `infra_required` (see Protocol). Without the
checkout's `data_v2/nestful_data.jsonl` the whole suite is `infra_required`, naming it; nothing is
generated. gbench reads no other copy of the data (IBM's Hugging Face release stores
`gold_answer` as `str(value)`, which loses the JSON type of 260 answers that the Win Rate
compares by value and type). `dataset_source` (`checkout`) and `dataset_revision` record it.

## Protocol
- **Prompt**: `deepseek_prompt_input`, byte for byte: the preamble, the row's tools, the strict
  output-format instruction and the first 3 examples of `src/icl_examples.json` in the system
  message; the row's `input` as the user message. As upstream, the tools are JSON-encoded twice
  (`src/eval.py` encodes them and `deepseek_prompt_input` encodes that string again), so the
  model sees them as a JSON string literal.
- **Parsing** (`parse_deepseek_output`): the whole reply (stripped) must be JSON. Prose around it or
  a Markdown code fence makes it a parsing error with no calls. Inline reasoning tags are removed
  first, for the per-sample verdict and the corpus metrics alike. Gold and prediction are grounded
  with `ground_seq_nested_repsonse`: each `$label.field` reference becomes the name of the call that
  produced the label and labels are dropped, so `$var_1` versus `$var1` spelling does not matter.
  No other rewriting is applied (an earlier gbench `$var_N -> $varN` rewrite is gone).
- **Scoring** (`calculate_scores`): per example, the calls are rendered as
  `name(sorted "arg = value")` strings, aligned by name (`post_process_api_with_args`) and compared
  with sklearn `accuracy_score` (Partial Match); a predicted `$ref` missing its closing `$` is
  repaired first. F1 Intent and F1 Slot are sklearn macro F1 with a `MultiLabelBinarizer` fit on
  the gold. The **Win Rate** executes the parsed reply (the model's own labels) against the
  checkout's reference functions (`calculate_ans`, 10 s alarm) and compares the final value with
  `gold_answer` (floats rounded to the gold's precision; a tuple result may equal a list gold).
  gbench runs these calls in a separate worker process so the alarm applies even when the suite
  runs off the main thread (as it does under `--suite-timeout`); a call stuck where the alarm
  cannot interrupt it (inside C code) is killed after 30 s and scores as a failure, and
  `win_rate_killed_calls` counts such calls.
- **The Win Rate worker's sandbox**: the Win Rate runs model-controlled code. The reply chooses
  which reference functions run and supplies their arguments. 18 of the functions offered in the
  data eval, exec or run their arguments as a command (a search of their source for `eval(`,
  `exec(`, `subprocess` and `os.system`), among them `evaluate_arithmetic`, `eval_condition`,
  `check_output`, `string_to_type` and `check_command`. So the worker always
  runs in a bubblewrap container of its own, whatever `GBENCH_SANDBOX` says. The container has
  new user, PID, IPC, UTS, network and cgroup namespaces, no capabilities and a new session, and
  it is killed with gbench. Its whole environment is `PATH` and `HOME=/tmp`. Its filesystem holds
  only `/usr` and the interpreter's own directories, the dynamic loader's cache, the reference
  functions directory (read-only, at its own path), a private minimal `/dev`, a private `/tmp`,
  which is the working directory, and (where bubblewrap offers `--ro-bind-data`, as 0.9 does) a
  read-only `/etc/hosts` that names `localhost` alone, so a function that names `localhost`
  reaches the container's own loopback as upstream's reaches its host's (the host's own
  `/etc/hosts` is not shown).
  There is no `/proc`, `/run`, `/home`, gbench tree, cache or credential file, and loopback is its
  only network interface. No key, token, Hugging Face token file, Docker socket or cloud metadata
  server is within its reach. The worker runs NESTFUL's scoring functions (a copy of gbench's
  port, sent as source) and imports nothing from gbench. When it starts, and again after every
  restart, it reports what it can see: its process id, the names of its environment variables,
  the entries of its root, whether `/proc` and `/run` exist, its network interfaces, and whether
  the functions are there and read-only. gbench refuses a worker whose report differs, and nothing
  is ever executed outside the sandbox. The worker's `sys.argv` is upstream's scoring command
  (`src/scorer.py --model_name ... --result_file_path ... --executable_func_dir ...`, as `run.sh`
  runs it), because three reference functions read it. `parse_command_line_options` parses it,
  meets scorer.py's options and exits, which upstream's bare `except:` scores as a failed call.
  The result records the sandbox under `win_rate_sandbox`: `sandbox`, `network`, the worker's
  report in `worker_reported`, and a `note`.
- **What the sandbox changes**: the reference functions have no network (loopback alone, no
  resolver), while upstream's scorer runs them with network. A call of a function that reaches
  the network (`get_url_content` and a few others) can therefore win upstream and fail here.
  The worker notices every failed use of the network: it wraps the `socket` module's name lookups
  (`getaddrinfo`, `gethostbyname`, `gethostbyname_ex`, `gethostbyaddr`, `getnameinfo`),
  `create_connection`, and a socket's `connect`, `connect_ex`, `sendto` and `bind` (a failed
  lookup counts wherever it happens, a failed connection or send for any reason but a
  non-blocking one still in progress, `bind` only when its address failed to resolve), and leaves
  what they return or raise unchanged. A refused connection to loopback counts too: the host's
  own local services are not in the container. On a row whose pinned gold sequence needs the
  network (one row, below), a call that did not win and failed such a use may have got its
  verdict from the sandbox rather than from the reply, and the worker cannot tell which (a host
  name that exists nowhere fails with network too), so it is treated as a harness error, never
  scored as a loss.
  - The row is marked `nestful_win_blocked_by_sandbox` (what the call tried, e.g.
    `getaddrinfo('www.example.com'): gaierror: [Errno -3] Temporary failure in name
    resolution`), counted in `win_rate_network_blocked_calls` (their sample ids in
    `win_rate_network_blocked_samples`) and left out of the Win Rate, which is computed over the
    rows whose call was scored. The other metrics execute nothing and keep the row.
  - `status_reason` and `leaderboard_comparable_reason` name them, the status is
    `completed_with_errors`, and the run is not leaderboard-comparable.
  - When the calls of every answered row are such calls, `win_rate` is `null` (never 0) and
    `win_rate_withheld_reason` says why.
  - A call that wins despite a failed use of the network keeps its win. A use of the network from
    a child process that a reference function starts (a command it runs) is not seen; none of the
    offered functions that import a network module starts one.
  - On every other row the gold sequence needs no network, so a losing call that failed a use of
    the network went off the gold path by the reply's own choice (a distractor such as
    `check_open_ports` offered next to a gold path of arithmetic, say). Such a call is scored as
    the loss it is, marked `nestful_win_network_blocked_offpath` (what it tried) and counted in
    `win_rate_network_blocked_offpath` (their sample ids in
    `win_rate_network_blocked_offpath_samples`); it is not a harness error and does not change
    the status or comparability. Of the 17 rows of the pinned data that offer a function that uses
    the network, 16 have gold sequences that use none.

  On the pinned data one gold sequence is such a call: sample
  `fa57c28a-e9d8-44fb-a358-a9404a557d5b` fetches `https://www.example.com`. This was checked by
  scoring every gold call sequence of the 1861 rows twice. Upstream's `src/scorer.py`
  `calculate_win_score` ran from the checkout, with network, under its own command line, and gave
  1857 wins. The sandboxed worker gave 1856 wins, 4 losses, and that one sequence reported as a
  failed use of the network (`getaddrinfo('www.example.com')`); nothing else differs. That row
  is the one whose gold needs the network: a losing reply on it that failed a use of the network
  is therefore a harness error here, not a loss. (The earlier
  unsandboxed worker gave 1858: it also won `c0147c17-efc0-44df-abbf-d32e2f988f91`, whose
  `parse_command_line_options` call fails under upstream's command line.)
- **Win Rate without its prerequisites**: only the Win Rate needs the checkout's
  `data_v2/executable_functions` and a working bubblewrap. In these cases, as long as the
  checkout's `data_v2/nestful_data.jsonl` is there, the other metrics are computed as usual and
  the Win Rate alone is reported `infra_required` in its own entry:
  - the functions directory is missing, lacks `basic_functions.py` or `func_file_map.json`, or
    those two files are not at the pinned revision;
  - bubblewrap is not installed, cannot create its namespaces, or its container is not isolated
    as described above (checked before any request is sent).

  This is how upstream's `calculate_scores(..., win_rate_flag=False)` (`src/scorer.py` at the
  pinned commit) reports the Win Rate: `"no"` instead of a number (below). If the worker has to
  be restarted after a kill and its sandbox does not come back, the Win Rate is withheld as a
  whole in the same way; it is never partly scored.
- The port was checked against `src/scorer.py` at the pinned commit on 1861 x 10 synthetic
  replies (exact, fenced, prose, relabelled, unlabelled, dropped/swapped/wrong calls, missing `$`):
  identical F1 Intent, F1 Slot, Partial, Full, Win Rate and parse-error counts. Two upstream
  crashes are scored instead of aborting the run: a reply that decodes to a bare
  number/bool/null counts as a parse failure, and a float gold without a decimal point skips the
  rounding.

## Headline and how to read the result
- `accuracy` = `full_match_accuracy` (scorer.py's "Full Match Accuracy": the share of examples
  whose Partial Match is exactly 1.0).
- Also reported: `partial_match_accuracy`, `f1_intent`, `f1_slot` (with `p_intent`, `r_intent`,
  `p_slot`, `r_slot`), `win_rate` (with `win_rate_network_blocked_calls` and
  `win_rate_network_blocked_offpath`, above), and
  `parsing_errors` (examples whose reply was not a clean JSON list; upstream prints this as
  "Parsing Errors"). A high `parsing_errors` usually means the model
  wrapped its answer in prose or a code fence, which NESTFUL counts as wrong.
- `official_sampling`: NESTFUL's settings (`src/eval.py`: `--temperature` default 0.0,
  `--max_tokens` default 1000; `run.sh` passes neither), the temperature this run used, its source
  and whether it matched, and whether the 1000 cap was applied.
- Budget: without `--thinking` every request carries exactly `max_tokens=1000`, whatever
  `--max-output-tokens` says (`max_output_tokens_note` records that the flag was not applied); a
  reply that stops there is scored on what arrived, as upstream scores it, and is counted in
  `replies_at_official_max_tokens`, apart from truncations. With `--thinking` the cap, written for
  non-reasoning generation, is not applied: each request gets the run's `--max-output-tokens`
  (else the thinking floor), and `official_sampling` records `max_tokens: 1000`,
  `max_tokens_applied: false` and the note "official cap not applied to reasoning"
  (`base.resolve_token_budget`). `replies_cut_off` counts the replies a budget stopped before they
  finished other than at the 1000 (every one with `--thinking`; without it, one a served context
  window cut below the 1000), an empty reply after reasoning included.
- **Requests that got no reply** (the endpoint failed, timed out or the prompt did not fit the
  context window) are harness errors, not the model's answers. base counts them (`failed_requests`,
  `request_harness_errors` by kind), names them in `status_reason` and sets the status to
  `completed_with_errors`. They are left out of every metric: Full and Partial Match, F1 Intent
  and F1 Slot, `parsing_errors` and the Win Rate are computed over the rows that were answered.
  Because upstream scores every row, `leaderboard_comparable_reason` then names how many requests
  were left out and their kinds (e.g. `1 of 1861 requests got no reply and are left out of every
  metric (harness errors: endpoint_failure 1); NESTFUL scores every row`).
- **Nothing answered**: when no request got a reply, nothing is scored. `accuracy`,
  `full_match_accuracy`, `partial_match_accuracy`, `f1_intent`, `f1_slot`, `p_*`, `r_*`,
  `parsing_errors` and `win_rate` are all `null`, never 0. `accuracy_withheld_reason` says why,
  the status is `failed`, and the run is not leaderboard-comparable.
- `leaderboard_comparable` is `true` only for a full run (all 1861 rows, no shard) at temperature
  0.0 where none of the following happened: thinking, decoding penalties, repeated attempts,
  requests left out as harness errors, Win Rate calls killed by the 30 s backstop (upstream's
  scorer would still be running those, not scoring them), Win Rate calls left out because they
  failed a use of the network the sandbox withholds on the row whose gold needs it
  (`win_rate_network_blocked_calls`, above),
  replies cut off (`replies_cut_off`), or a Win Rate reported `infra_required`. Otherwise
  `leaderboard_comparable_reason` lists what differs. `win_rate_sandbox.note` describes the
  sandbox's missing network on every run with a Win Rate; a run in which no call needed the
  network stays comparable.
- When a prerequisite of the Win Rate is missing, `win_rate` is `null` (never 0),
  `win_rate_killed_calls` is `null` and `metrics_infra_required` is `["win_rate"]`.
  `win_rate_metric` is its entry: `status: "infra_required"`, the `prerequisite` (the reference
  functions directory or the bubblewrap sandbox, whichever is missing), the reason in
  `infra_required`, an `[INFRA REQUIRED]` `error` text, `leaderboard_comparable: false` and
  `value: null`. The headline
  (`accuracy` = Full Match), Partial Match, F1 Intent / F1 Slot and `parsing_errors` are computed
  as with the functions; the suite's status is `completed_with_errors` (or worse, by the usual
  rules) and never `success`, and the run is not leaderboard-comparable.

## Requirements
1. A running OpenAI-compatible `/v1` endpoint.
2. An IBM/NESTFUL checkout at the pinned commit, the only data source, with
   `GBENCH_NESTFUL_FUNC_DIR` pointing at its `data_v2/executable_functions` directory (the dataset
   is read from `data_v2/nestful_data.jsonl` next to it):
   ```bash
   git clone https://github.com/IBM/NESTFUL.git
   git -C NESTFUL checkout fc2c4123e73500a56185a5fb354f05d1c8b4890c
   export GBENCH_NESTFUL_FUNC_DIR=$PWD/NESTFUL/data_v2/executable_functions
   ```
   With `GBENCH_PREREQS_DIR` set, `$GBENCH_PREREQS_DIR/NESTFUL/data_v2/executable_functions` (and
   `$GBENCH_PREREQS_DIR/NESTFUL/data_v2/nestful_data.jsonl` for the data) is used when the
   variable is unset.
   - No checkout (no `data_v2/nestful_data.jsonl` next to the directory `GBENCH_NESTFUL_FUNC_DIR`
     names, or, with the variable unset, under `$GBENCH_PREREQS_DIR/NESTFUL`): the whole suite is
     `infra_required`, naming the checkout; nothing is generated.
   - The data there but the reference functions missing (or at another revision): the Win Rate
     alone is reported `infra_required` and the other metrics run on the checkout's data.
   - A `nestful_data.jsonl` that is there but not at the pinned revision is a hard error
     (`infra_required`), never a skip or a partial score.
3. `scikit-learn` (the metrics are sklearn's); the suite hard-errors before generating if it is
   missing: `pip install scikit-learn`.
4. For the Win Rate: bubblewrap, allowed to create unprivileged user namespaces:
   ```bash
   sudo apt-get install bubblewrap
   # Ubuntu 23.10+ blocks unprivileged user namespaces by default:
   sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0
   ```
   Without it the Win Rate alone is `infra_required` and the other metrics run. `GBENCH_SANDBOX`
   does not apply: the Win Rate never runs unsandboxed.
5. No API key and no LLM judge.

> **Code execution.** The Win Rate executes the model's reply: the reply picks NESTFUL's reference
> functions by name and supplies their arguments, and some of those functions eval, exec or run
> their arguments. That is model-controlled code. It runs only in the worker's bubblewrap sandbox
> (no network, only `PATH` and `HOME=/tmp` in its environment, no host files beyond the
> interpreter and the reference functions, read-only). Each sequence is capped at 10 s by the same
> alarm upstream's scorer sets, with a 30 s hard stop for a call the alarm cannot interrupt. A
> call that loses after a failed use of the network is a harness error, left out of the Win Rate,
> on the one row whose gold needs the network, and a scored loss on any other row (see "What the
> sandbox changes").

## Run
```bash
export GBENCH_NESTFUL_FUNC_DIR=/path/to/NESTFUL/data_v2/executable_functions
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals nestful --max-output-tokens 4096
```
Add `--eval-limit N` for a quick check (flagged in the result). Without
`GBENCH_NESTFUL_FUNC_DIR` and without `$GBENCH_PREREQS_DIR/NESTFUL`, the same command stops with
`infra_required` naming the checkout.

The run logs its planned workload in one line at its start (for example `[nestful] --eval-limit 5:
5 sample(s) x 1 attempt = 5 generations`) and records it in the result as `eval_limit_plan`
(`limit`, `samples`, `per_setting`, `attempts_per_sample`, `generations`).

Sampling: temperature 0.0 without `--thinking`, 1.0 with it (gbench's rule for every suite);
upstream's `src/eval.py` generates at 0.0 (`SamplingParams(temperature=args.temperature, ...)`,
`--temperature` default 0.0), recorded under `official_sampling`. A temperature other than 0.0 is
named in the comparability reason.
`--temperature`, or `GBENCH_NESTFUL_TEMPERATURE` for this suite alone (it takes precedence),
overrides it; the result records it under `official_sampling` and is then not
leaderboard-comparable.
