# Results: output layout, provenance and exit codes

Where gBench writes its results, how they record what was measured, and what its exit codes mean.

[Back to the README](../README.md)

## Output directory structure and result schemas

All run outputs are written to timestamped directories under `--results-dir`:

```
results/07-31-160229/
|-- metadata.json                                           # Run metadata: models, tags, preset, scaffold conditions, gbench commit
|-- summary.json                                            # Every result of the run (golden set included) and every failure; an eval result's sample_traces stay in its eval_*.json (`sample_traces_omitted` names the file and count)
|-- eval_summary.csv                                        # Eval results and failures as a table (written when evals ran)
|-- eval_gpqa_gemma-4-12B-it_hf.json                        # One eval suite's result, with its sample-level traces
|-- performance/
|   |-- serve_gemma-4-12B-it_hf_b32.json                    # Serving metrics per batch size (_mm suffix: multimodal; with --campaign, serve_<model>_<fmt>_<campaign>_b<N>.json)
|   |-- throughput_gemma-4-12B-it_hf_in1024_out128_b32.json # Offline throughput per geometry
|   `-- stress_gemma-4-12B-it_hf_text.json                  # Stress sweep results (text or mm; with --campaign, stress_<model>_<fmt>_<campaign>_<mode>.json)
|-- quality/
|   `-- quality_gemma-4-12B-it_hf.json                      # GemmaClaw agentic scenario scores
`-- logs/
    |-- server_gemma-4-12B-it_hf.log                        # Local vLLM server process stdout/stderr
    |-- throughput_gemma-4-12B-it_hf_in128_out512_b1.log    # Offline throughput benchmark output
    `-- quality_gemma-4-12B-it_hf.log                       # GemmaClaw QA suite console output
```

Each run claims its own `<timestamp>` directory (`-2`, `-3`, ... when another run took the name
in the same second). A `--results-dir` that already is a run directory (metadata.json or an
eval_*.json in it) is continued in place by one run at a time: the run writing there holds an
exclusive lock on its `.gbench-run.lock` (pid, short host name and start time inside). A run
that finds it held (swarm nodes given one shared directory), or a filesystem without file locks,
writes its own `<timestamp>` directory inside it instead, logs why and records
`results_dir_not_shared` in its metadata.json, so two runs that overlap in time never write the
same directory. A later run pointed at a finished run's directory continues it in place (its
metadata.json and summary.json replace the earlier run's), so give every node of a sharded run
its own `--results-dir`. `--skip-existing` still searches all of `--results-dir`. A dry run
creates nothing.

Every eval result carries, besides its scores and `sample_traces`: `served_model_id` and
`served_model_id_source` (the id the requests went to and how it was matched), `run_shape` (what
it was measured under: temperature, `--max-output-tokens`, thinking, `--eval-limit`, shard,
`--eval-n-shot`, soft tokens, `--attempt-count`, routed categories, the served id and window, the
judge and search cascades, the serve settings, the suite's run-shaping env knobs), `scaffold_id`,
`subject_id`, `gbench` (the commit), `finished_at`, `eval_limit`, `canonical_sync`, and
`reused_from` when `--skip-existing` reused it. A run-level flag a suite did not apply is said on
the result: `max_output_tokens_note`, `attempts.not_applied`, `temperature_applied` /
`temperature_note`, `thinking_applied` / `thinking_note`, and `shard_applied` /
`shard_requested`. metadata.json also records `model_metadata` (each model's identity, derived
values, `metadata_source` / `metadata_origin`, `revision` / `revision_source` and
`vision_tokens_source`) and the serve settings.

### Provenance and comparability

A result is the pair (subject, scaffold): `subject_id` hashes the model's observable identity and
`scaffold_id` hashes everything the run pinned about how it asked (for evals: the suites and their
reconciled provenance, the decoding, dataset and serving settings, the run-shaping knobs that are
set). Holding `scaffold_id` while `subject_id` moves is a like-for-like model comparison.
metadata.json records every planned condition with both ids and the gBench commit; the contract and
what each pillar pins are in [docs/scaffold-versioning.md](scaffold-versioning.md).
`gbench --eval-provenance` prints when each suite was last reconciled with its upstream and whether
it is pinned to a revision or checksum (`--list evals` shows the same per suite). Each eval result
also says whether it is leaderboard-comparable (`leaderboard_comparable`, with every reason in
`leaderboard_comparable_reason`: a subset, a partial shard, a self-judged run, a substituted
service, a deviation from the official sampling, ...).

### Exit codes

| Code | Meaning |
| :--- | :--- |
| `0` | Every planned pillar produced its result. |
| `1` | A model verdict failed: the Golden Set has failing cases. |
| `2` | A harness error, not a verdict on the model: a pillar that raised or whose server did not start, a performance configuration (text or multimodal) that failed, an eval suite without a score (`INFRA REQUIRED`, `ERROR`, `TIMEOUT`), a Golden Set error, an unreachable endpoint, a refused configuration (busy GPUs, an over-subscribed host, a model that cannot be served), or incomplete shards for `--merge`. A harness error outranks a Golden Set failure. |

Each pillar runs on its own: one that fails is recorded as a failed configuration (named in the
summary and in summary.json) and the model's next pillar still runs.
