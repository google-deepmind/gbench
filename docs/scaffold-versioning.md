# Scaffold versioning

## Why

A benchmark score is only interpretable if the model *and* the harness stayed
fixed. If the prompt template, the few-shot examples, the output schema, the
decode settings, the scoring script or the tool setup moved between two runs,
a score change no longer tells you whether the model improved or regressed. It
tells you that something moved, and you have no way to say what.

So gbench treats the scaffold as part of the experiment, roughly as important
as the weights. It is versioned, named, and printed beside every result.

> Pin the scaffold as tightly as the weights, otherwise your benchmark is
> measuring a moving target.

## The eleven fields

| Field | Axis |
| --- | --- |
| `model_checkpoint` | subject |
| `quantization` | subject |
| `prompt_version` | scaffold |
| `schema_version` | scaffold |
| `scoring_version` | scaffold |
| `tool_permissions` | scaffold |
| `attempts` | scaffold |
| `decode` | scaffold |
| `dataset_snapshot` | scaffold |
| `serving_environment` | scaffold |
| `load_shape` | scaffold |

`serving_environment` and `load_shape` are scaffold fields for the same
reason the prompt is. Serving the same weights under a different engine,
a different tensor-parallel width or a different KV budget is a different
question, and `batch_sizes=[1, 16, 50, 100]` is a property of the harness
rather than of the model.

## Two ids, not one

A run is the pair `(subject, scaffold)`, and the two are hashed separately.

- `subject_id`: what was tested. Model id and serving format.
- `scaffold_id`: how we asked. Everything else.

The separation is the entire design, and it exists to make one specific
comparison legal:

| | valid? | reading |
| --- | --- | --- |
| `subject_id` moved, `scaffold_id` held | yes | a like-for-like model comparison |
| `scaffold_id` moved, `subject_id` held | yes | a new experimental condition |
| both moved | no | uninterpretable, and no comparison should join them |

A quantization sweep is the first row: several subjects (checkpoints)
measured against one held scaffold. A single combined hash would make every
subject produce a different id and there would be nothing left to hold
constant, so the sweep could not be expressed at all.

A subject id is identity, not a pin: it hashes the model id and the serving
format, and no checkpoint revision or quantization level is resolved, so
`model_checkpoint` and `quantization` are reported unpinned on every subject.
A held subject id therefore says the same model id was asked, not that the
weights behind it held still.

## Reading it from the CLI

The console and `metadata.json` are two views of the same objects. Neither is
required for the other to work, because plenty of gbench runs are headless and
some consumers only ever use the CLI.

```
$ gbench --models google/gemma-4-E2B-it google/gemma-4-E4B-it --golden-only \
    --num-gpus 1 --tool-call-parser <parser>

================================================================================
SCAFFOLD CONTRACT (v2)
================================================================================

A result is the pair (subject, scaffold). Holding scaffold_id while
subject_id moves is a like-for-like model comparison. A scaffold_id that
moved means the harness moved, so scores either side are not comparable.

PILLAR  MODEL                  FORMAT  SUBJECT      SCAFFOLD
golden  google/gemma-4-E2B-it  hf      sb_37560700  sc_0a202bc4
golden  google/gemma-4-E4B-it  hf      sb_373e842c  sc_0a202bc4

sc_0a202bc4  pins 8/9: attempts, dataset_snapshot, decode, prompt_version,
             schema_version, scoring_version, serving_environment,
             tool_permissions
             dataset gd_661165ef over 16 case(s)
             unpinned: load_shape

Subject ids pin 0/2; unpinned: model_checkpoint, quantization. A subject id
hashes the model id and serving format; no checkpoint revision or quantization
level is resolved.
```

Two subjects, one scaffold. That is the comparison, and it is readable
without a service, an upload or a browser. (The ids in these examples are
illustrative: the serving environment includes the host's engine version and
GPU models, so the same command gives other ids on another host.)

A scaffold's coverage is counted over the nine scaffold fields and a
subject's over the two subject fields. Each block's `unpinned` names only the
fields of its own axis.

The same conditions are written to `metadata.json` under `conditions`, one
entry per `(pillar, model, format)` cell of the run's matrix.

## Coverage is disclosed, not assumed

`covers` and `unpinned` are not decoration. An id that silently claimed to pin
nine fields while actually pinning six would be worse than no id, because it
would licence a comparison that is not sound.

`unpinned` is derived from `covers` rather than hand-maintained, so the two
cannot drift, and `scaffold_id` hashes exactly the fields named in `covers`.
It is structurally impossible for the id to over-promise.

Each pinned field holds the value the runner resolves and uses, not the flag
it was derived from: a flag the runner never reads cannot move an id, and an
input the runner does read (a flag, an environment variable, a runner
constant) cannot be left out of it.

Note the direction of the guarantee. Disclosure makes the id *honest about its
scope*. It does not *detect* a change in an unpinned field. Only a content
hash detects.

### Why golden pins eight of nine

- `load_shape` is unpinned, and that is not an oversight. Golden issues its
  cases one at a time, so it does have a concurrency, it is one, and nothing
  in the harness records that as a decision. When it becomes explicit it
  should move into `covers`.

The other eight are pinned, three of them by subsumption: `prompt_version`,
`schema_version` and `tool_permissions` all live inside the case JSON, so a
content hash over `gbench/golden_dataset/*.json` covers all three at once.
Exactly three of the sixteen cases declare top-level `tools`/`tool_choice`.

Golden's serving environment is the native server's: the CLI starts one
server per model for the quality, Golden Set and eval pillars, at the model's
own window and with the operator's parsers.

A pillar whose scaffold is not yet modelled reports `scaffold_id: null` and
`0/9` rather than being omitted, so the gap is visible instead of looking
like the pillar did not run.

### What every pillar pins

| Pillar | pinned (as the runner resolves it) |
| --- | --- |
| all local runs (`serving_environment`) | GPU count, tensor-parallel and data-parallel sizes, memory utilization, scheduler settings, served window, the serve arguments appended to that server's command (`--vllm-serve-args` on every server; the `--tool-call-parser` / `--reasoning-parser` flags on the native server), the vLLM version of gbench's interpreter and a `GBENCH_VLLM_BIN` override, the product names of the selected GPUs. Against `--remote-endpoint`: the endpoint (hashed when not loopback) and nothing else. The port and the served model id are not pinned: the port changes nothing measured, and the served id names the subject (each result records it as `served_model_id`, with `served_model_id_source`; `metadata.json` records the `--served-model-id` override when one was given). |
| serving, serving_multimodal | the batch-size sweep, the first input and output length (the request shape serving uses), the request count per batch size (`BenchmarkConfig._serving_num_prompts`: auto-sized, or the `--personal` override; `--num-prompts` is not read), request rate, iterations and warmups, the dataset name and `--dataset-path` (text) or `generated` (multimodal: the runner builds its own images and text), images per request (multimodal), the measurement-method version as `dataset_snapshot.workload_version` (`methodology_record("serving")["methodology_version"]` from `gbench/analysis/perf_method.py`, built from the components serving's numbers depend on: payload, tpot_metric, cache_telemetry, serving_transport, reporting; a change to one bumps it and changes the id; offline throughput is not pinned, it uses none of that code), and the no-think protocol (`--thinking`, the no-think system prompt's sha256 and request fields) for multimodal serving and for text serving against a remote endpoint (a local text run drives `vllm bench serve` over `/v1/completions`, which carries none). |
| throughput, throughput_multimodal | every input and output length it sweeps, its prompt count (`num_prompts_throughput`), iterations, images per request (multimodal), and the dataset as `random` (generated for multimodal): the offline benchmark reads no batch size, request rate or `--dataset`, and runs no warmup. `--thinking` never reaches it. |
| stress_test, stress_test_multimodal | the workload the runner resolves (the pinned geometry and dataset with `--campaign` or explicit lengths, else random 128/128), reps, rate cap, requests per rate point, sample floor, images per request (multimodal), each resolved flag, else `GBENCH_STRESS_*`, else the runner default; the no-think protocol; `dataset_snapshot.workload_version` = `methodology_record("stress")["methodology_version"]`, built from all eight `METHODOLOGY_COMPONENTS` (payload, schedule, gate, itl_metric, tpot_metric, cache_telemetry, serving_transport, reporting); and as `scoring_version` the TTFT/ITL SLOs, `keep_up_frac`, `itl_gate` (the gate component's version), `certify` (the certification mode) and `certify_budget_min`, because they decide the max-sustainable-QPS verdict and whether it is certified. Every result also records `methodology_version` and `methodology_components`; across runs the changed id is the guard (gbench has no perf compare command). The client process count follows the host's cores and is not pinned. |
| quality | the native server's environment, the resolved GemmaClaw commit, the scenario subset. |
| golden | the dataset content hash, sampling, attempt policy, `SCORING_VERSION`, the native server's environment. |
| evals | see below. |

## The scorer

`scoring_version` means different things in the two pillars that pin it, and
both readings are deliberate.

Golden's scorer is code in `gbench/runners/golden.py`, not data, so it pins a
declared integer `SCORING_VERSION` that a human bumps when grading changes.
The alternative, hashing the source file, would break every series on an
unrelated edit to that module, and an id that churns on changes touching
nothing relevant stops carrying information. `SCORING_VERSION` is 4. Version 2 made
a case whose modality the model lacks `not_applicable` (it leaves the
denominator), `python_exec` without a code sandbox an error rather than a
failure, the endpoint the run's own server or `--remote-endpoint` (never
whatever answers on a default port), and an endpoint that does not serve the
model under test a harness error. Version 3 made the `chat_template` cases
forbid the served model's own chat-template tokens, read at run time from its
tokenizer (`forbid_chat_template_tokens`), instead of a fixed list of Gemma
delimiters. When that tokenizer cannot be loaded, or yields no token, those
cases are harness errors, never passes. Version 4: a --chat-template's
delimiter-shaped literals that the tokenizer has no token for are forbidden
too (`template_literal_delimiters`); the template and remote-code trust are
read from any spelling vLLM accepts, and `--tokenizer-trust-remote-code` sets
the trust. `golden.scoring_fingerprint()` /
`GOLDEN_SCORING_FP` is a reminder gate, not the id: a unit test fails when
the functions that decide a verdict change, until someone reconsiders
`SCORING_VERSION` and refreshes the fingerprint. The id stays driven by the declared version.

Quality's scorer is GemmaClaw, a separate repository, so it pins the commit.
`--gemmaclaw-commit` defaults to a sha rather than to a branch, because a
branch name is not a pin: it names a different commit next week while the
hash of the string `"main"` never moves. The default lives in
`DEFAULT_GEMMACLAW_COMMIT` in `gbench/core/config.py` and is currently
`d8bc6989...`, the `gemmaclaw-v2026.8.3` release.

```
$ gbench --models google/gemma-4-E2B-it --quality-only --dry-run --num-gpus 1 \
    --tool-call-parser <parser>

PILLAR   MODEL                  FORMAT  SUBJECT      SCAFFOLD
quality  google/gemma-4-E2B-it  hf      sb_37560700  sc_a932bccc
```

So the same command on the same host and engine gives the same `scaffold_id`
in six months. Nothing is looked up and the run does not need GitHub to be
reachable.

Anything that is not already a sha is resolved before it is hashed. Ask for
the development tip and the id moves to say the scorer moved:

```
$ gbench --models google/gemma-4-E2B-it --quality-only --dry-run --num-gpus 1 \
    --tool-call-parser <parser> --gemmaclaw-commit main
INFO - Resolved gemmaclaw ref 'main' to 08c0584d660591fa713928df9249e4ec37322ea5

PILLAR   MODEL                  FORMAT  SUBJECT      SCAFFOLD
quality  google/gemma-4-E2B-it  hf      sb_37560700  sc_b5a34722
```

(The ids in this example depend on the commit `main` resolves to that day.)

Resolving once rather than per condition is what keeps the sha in the id and
the sha in the checkout the same commit. Two lookups could straddle a push to
`main` and pin a scorer the run never used.

A ref that cannot be resolved is left as the literal string. That is honest
about the gap rather than a guess, and it predicts a failed run anyway,
because a ref the resolver cannot resolve is a ref the runner cannot check
out.

### Promoting the pinned scorer

By hand, and deliberately so. Take the newest `gemmaclaw-v*` tag, put its sha
in `DEFAULT_GEMMACLAW_COMMIT`, update the tag name and date in the comment
above it, and say so in the PR. There is no script for this.

Automating it would defeat the pin. A default that discovers the newest
release on its own is still a moving target, just a slower one: the day a new
tag lands, every unflagged run silently changes scorer. Promotion is supposed
to be a decision someone made and a reviewer saw.

## The dataset snapshot

`dataset_snapshot` hashes the canonical form of every case, plus every binary
asset by content. Canonical means sorted keys, no incidental whitespace, and
integers normalised to floats, so reformatting a case cannot move the id but
changing a prompt does.

All assets are hashed even on a subset run. Attributing an image to the cases
that reference it would mean parsing every message, and over-detecting is the
safe direction here: a spurious break costs a glance at a diff, a missed one
costs a wrong conclusion.

## The academic evals pillar

Every input the evals runner hands the suites is pinned, resolved the way the
run resolves it:

| Field | pinned |
| --- | --- |
| `serving_environment` | the native server's (above) |
| `load_shape` | the request concurrency (`--eval-concurrency`, else `--batch-sizes[0]`), the `--sandboxes` value every suite receives, and the containers a selected containerized suite runs at once (`--sandboxes`, else `--batch-sizes[0]`) |
| `dataset_snapshot` | the suites that run (`--evals` expanded as the runner expands it: comma lists, renamed suites, `all` and `plugins` with the discovered plugins, `custom_jsonl` when `--eval-custom-jsonl` is set), `--eval-categories`, `--eval-limit`, the shard (`--shard`, else `GBENCH_SHARD`, normalized; `1/1` is the whole set), the sha256 of the `--eval-custom-jsonl` file (content, not path), and the dataset-class run-shaping variables that are set |
| `decode` | `--temperature` (null = the rule's default for every suite: 1.0 with `--thinking`, else 0.0), `--max-output-tokens`, `--eval-max-soft-tokens`, and the decode-class variables that are set (penalties, soft-token overrides, the tool-loop budget, `GBENCH_<SUITE>_TEMPERATURE`) |
| `prompt_version` | the few-shot count, `--thinking`, and the prompt-class variables that are set (agentic or one-shot modes, prompt variants) |
| `attempts` | `--attempt-count`, the independent attempts per sample behind avg@k / pass@k, plus `suite_attempts_per_sample` for a selected suite that always runs more (MathArena's aime/hmmt: 4) |
| `scoring_version` | each suite's canonical-sync anchors and the anchors of every suite and shared module it imports (below), a discovered plugin's module sha256, the Gemini judge and search cascades as resolved (`GBENCH_JUDGE_MODELS` naming the default list is the same grading as leaving it unset) with the judge temperature, and the scoring-class variables that are set |

The run-shaping variables are the ones `base.run_shaping_env` declares
(`RUN_SHAPING_GLOBAL_ENV`, `RUN_SHAPING_SUITE_ENV` and every suite's
`GBENCH_<SUITE>_TEMPERATURE`); only those that are set are recorded, so an
unset variable leaves the id where it was. A deprecated bare-prefixed alias
(`RUN_SHAPING_ALIASES`, for example `TAU2_NUM_TRIALS`) is recorded under its
canonical name, so the legacy spelling moves the id as the canonical one does.
The same snapshot is the `env` part of each result's `run_shape`, which
`--skip-existing` compares before it reuses a result.

Every other `GBENCH_*` variable a suite reads is listed in
`base.NON_RUN_SHAPING_ENV` with the reason it changes nothing measured
(`infra`: where a pinned input lives, an image, a port, a concurrency, a
timeout whose expiry is a harness error; `credential`; `diagnostic`;
`internal`; `removed`). Tests fail when a suite module, its vendored code or
the code and entrypoints of its container (`gbench/docker/`, `docker/`) read a
`GBENCH_*` name that is in neither list, and they check that each declared
variable and alias moves the evals `scaffold_id`. A new knob therefore gets a
decision when it is added: run-shaping, or not and why.

### The scorer: canonical-sync anchors

The evals pillar pins its scorer version through the per-suite canonical-sync
registry (`gbench/runners/eval_suites/canonical_sync.py`):

- Each suite has an entry recording `synced` (the date it was last reconciled
  against upstream), `revision` (the pinned upstream commit, release or
  package==version; several pins separated by `, `), `dataset` (the dataset id
  and revision the loader reads) and `checksum` (sha256, git-blob or git-tree
  of the pinned content). These four are the entry's anchors. The prose fields
  (`upstream`, `method`) are not, so a documentation edit breaks no series.
  Every entry but `custom_jsonl` (pinned by content in `dataset_snapshot`)
  must lead its `revision` or `checksum` with a pin, and every pin a suite
  holds in code (`*_REVISION`, `*_COMMIT` constants) must appear in its
  anchors (`revision`, `dataset` or `checksum`, the fields the scaffold reads;
  the prose `upstream` field does not count); `tests/test_canonical_sync.py`
  enforces both.
- Modules several suites import (`base`, `sampling`, `swebench_common`,
  `vqa_common`, `tau_common`, ...) have entries of their own in `SHARED_SYNC`.
  A suite's `scoring_version` signal includes the anchors of every suite and
  shared module it imports, directly or through others (`DEPENDS_ON`), so a
  verdict-relevant change to `base.py`, recorded once by bumping
  `SHARED_SYNC["base"]["synced"]`, moves every evals series that runs through
  it.
- The dispatcher stamps the entry onto every result (`canonical_sync`, with
  the recorded code fingerprint as `code_fp`), and `gbench --eval-provenance`
  prints the table.

The id folds in declared anchors only, never the code fingerprint itself, for
the reason golden's `SCORING_VERSION` is declared: a content hash would break
every series on a refactor that cannot change a score. The gate below makes
sure the declaration is made.

### Updating a suite

When you change a suite's scorer, prompt, extraction, loader or harness (or a
shared module):

1. Make the code change.
2. Reconcile it against the upstream benchmark (do not invent a date or commit).
3. If the change can alter a verdict, update the entry (`CANONICAL_SYNC`, or
   `SHARED_SYNC` for a shared module): set `synced` to the reconcile date, and
   `revision` / `checksum` / `dataset` when the pin moved. This moves the evals
   `scaffold_id`. When the entry was already bumped the same day, append a
   serial to the date (`2026-09-28.2`, then `.3`): a date that did not change
   would move neither the anchor nor the id. If it cannot (a refactor), leave
   the entry and declare the change in `VERDICT_PRESERVING` with the new
   fingerprint and the reason.
4. Refresh the fingerprints:
   `python gbench/runners/eval_suites/canonical_sync.py --refresh-code-fp`.
   The refresh runs after the decision, never in place of it: it refuses, and
   writes nothing, while a recorded fingerprint moved and neither the entry's
   anchors moved since that fingerprint was recorded nor `VERDICT_PRESERVING`
   declares exactly the new fingerprint. It names each such suite or module.

### The gate

Each suite and shared module has a recorded code fingerprint (`CODE_FP`, a
generated block at the end of `canonical_sync.py`): the sha256 of its
docstring-stripped source AST (comments, formatting and docstrings do not
count), the `eval_suites` modules only it imports, the vendored data
directories it imports or names (byte for byte, a data `.txt` included,
whose `#` lines are content), and its Docker build files (files in
`gbench/docker/` and `docker/` named after it, or named in its code;
Dockerfiles, shell scripts and requirement, constraint and pin lists without
their comment and blank lines). The maintainer-owned plugin module
`swe_bench_pro_agentic` is never read. The fingerprint is the same on every
supported Python version.

- `tests/test_canonical_sync.py::test_code_fingerprints_current` (in the unit
  test workflow) fails when any fingerprint no longer matches the code, names
  each suite or module, and prints the steps above.
- Next to each fingerprint the generated block records a digest of the
  entry's anchors at the time (`FP_ANCHORS`). `--refresh-code-fp` judges a
  moved fingerprint against it (`refresh_violations`, the review rule below
  with the last refresh as the base), so running the refresh cannot turn the
  freshness test green over an undecided change. A moved fingerprint whose
  `FP_ANCHORS` digest is missing needs a decision too (a bump or a
  `VERDICT_PRESERVING` declaration): every recorded fingerprint has a digest,
  so a missing one is never evidence of a bump. An anchor that moved after
  the last refresh leaves the digest stale: `--check`, `--check-bump` and the
  unit tests fail until the refresh records it, so a later code change cannot
  count an older bump as its decision.
- `python gbench/runners/eval_suites/canonical_sync.py --check-bump --base
  <git ref>` is the review check: every fingerprint that moved since `<ref>`
  must come with a moved anchor or a `VERDICT_PRESERVING` declaration of
  exactly the new fingerprint. It needs nothing installed (stdlib only) and a
  checkout with the base commit (`fetch-depth: 0`). The Canonical Sync Gate
  workflow (`.github/workflows/canonical-sync-gate.yml`) runs it on every pull
  request (against the base commit) and push (against the previous commit) to
  main.
- `--check` runs the freshness checks alone (fingerprints and anchor digests).

## Changing a case

A snapshot id that moved tells a reader that the dataset changed. It does not
tell them *why*, and six months later "`gd_661165ef` became `gd_9c1e04b2`" is
most of the way to no information at all.

So a case whose content changed needs two things:

1. a bump to `meta.version`
2. an appended `meta.changelog` entry for the new version, with a non-empty
   `reason`

```json
"meta": {
  "version": "1.2",
  "changelog": [
    {
      "version": "1.1",
      "reason": "Baseline entry recorded when the scaffold versioning contract was adopted. Changes before this point are in git history rather than here."
    },
    {
      "version": "1.2",
      "reason": "Tightened answer_pattern so a longer number such as 4200 no longer satisfies a golden of 42."
    }
  ],
  "description": "..."
}
```

The changelog is append-only. Past entries may not be edited or removed, so
history cannot be quietly rewritten.

`meta.changelog` is excluded from the hash and from the gate's comparison.
Without that exclusion the rule is circular: appending the entry that explains
a bump is itself a content change, which would demand another bump, which
would demand another entry.

### The gate

```
$ python scripts/check_golden_changelog.py --base origin/main
```

Runs in CI as the `Golden Changelog` check. Reformatting alone is not a
content change and needs neither a bump nor an entry, because the gate reads
the same `canonical_json` the snapshot id reads. That equivalence is the point:
a change that would not move the id cannot trip the gate, and a gate that
nags on reindentation is one that gets switched off.

Deleting a case is reported rather than failed. There is no file left to carry
an entry, so the rule is unenforceable by construction.

## Where the code lives

| File | Role |
| --- | --- |
| `gbench/canonical.py` | `canonical_json`, `strip_changelog`. Stdlib only. |
| `gbench/core/scaffold.py` | Contract objects, id construction, console rendering. |
| `gbench/runners/eval_suites/canonical_sync.py` | The evals registry, shared-module entries, code fingerprints and their gate. Stdlib only. |
| `gbench/changelog_gate.py` | The changelog rules. |
| `scripts/check_golden_changelog.py` | Git plumbing for the gate. |
| `.github/workflows/golden-changelog.yml` | CI wiring. Needs `fetch-depth: 0`. |

`canonical.py` and `changelog_gate.py` sit outside `gbench.core` on purpose.
`gbench/core/__init__.py` imports the model registry (`models.py`), which
imports `huggingface_hub` (no network call at import). The gate reads JSON out
of a git diff and must import with no packages installed at all, so it lives
outside `gbench.core`. A policy gate
that goes red because a dependency is missing is a policy gate that gets marked
non-required.

## Contract version

`CONTRACT_VERSION` is bumped when the *meaning* of the blocks changes in a way
that makes older ids incomparable with newer ones. Adding a field to `covers`
does not need a bump, because `covers` already tells the reader what moved.

Version 2 made every pillar pin the inputs its runner resolves (the evals
knobs, run-shaping variables, judge cascade and shared-module anchors; the
stress load and SLOs; the effective serving counts, lengths and images; the
server's serve arguments, engine version, data-parallel size and GPU models),
dropped the unused GGUF file field from the subject id, and made each block's
`unpinned` name only its own axis. Every id moved, so no v1 id compares with a
v2 id.

The gbench commit is recorded beside the ids and never inside them. As an
input it would invalidate every scaffold on every commit to this repo, and an
id that churns on changes touching nothing relevant stops carrying
information.
