# spider2 setup

Canonical Spider 2.0-lite (`xlang-ai/Spider2`, `spider2-lite`) enterprise text-to-SQL scored by
**execution accuracy** with the official `evaluation_suite/evaluate.py` code at the pinned commit
`cafb867313aab4e674652054198f383cf4018943` (`SPIDER2_COMMIT` in `spider2.py`). Its local (SQLite)
path is copied verbatim (function bodies unchanged but for local imports and the error `print`
going to the logger; `evaluate.py` itself cannot be imported: it imports `google.cloud.bigquery`
and replaces `sys.stdout` when imported), so each instance is scored as
`evaluate_single_sql_instance` scores it:

1. `extract_sql_query`: the first `` ```sql `` block of the reply, else the whole reply;
2. `get_sqlite_result`: the query runs against an in-memory copy of the instance's `.sqlite`, and the
   result is written to CSV in chunks of 500 rows and **read back with `pd.read_csv`**, so the
   prediction is compared with the types a CSV round trip gives it, as the gold CSVs are (a
   number stored as TEXT, such as `'2001'` or `'007'`, reads back as the integer). A query that
   fails to execute, or leaves no readable result file (upstream's `Evaluation Error`), scores 0,
   the model's outcome;
3. `resolve_gold_paths`: `{id}.csv` alone when it exists, else the variants `{id}_a.csv`, `{id}_b.csv`,
   ...;
4. `compare_pandas_table` (one `{id}.csv`) or `compare_multi_pandas_table` (variants), with the
   instance's `condition_cols` and `ignore_order` (default False) from `spider2lite_eval.jsonl`:
   every required gold column must be matched by *some* predicted column, numbers within
   `abs_tol=1e-2`, NA (`None`/`NaN`/`NaT`) as 0, rows sorted when `ignore_order`; a prediction
   passes if it matches **any** variant. `compare_multi_pandas_table` applies a flat
   `condition_cols` list to every variant only when there are two or more; with a single
   `{id}_a.csv` it passes the list's first entry, as upstream does.

A comparison that raises (upstream's `Python Script Error`, scored 0 there), or an instance that
reaches scoring without a gold file (upstream's `No matching gold file found`; the loader refuses
such a checkout first), is a harness error in gbench (see below).

**The prompt is gbench's own.** Spider2-lite prescribes no generation prompt: its baselines
(`spider2-lite/baselines/`, DAIL-SQL and DIN-SQL) each build their own from preprocessed schema
files (DAIL-SQL's `SQLPrompt`, with `ask_llm.py`'s temperature 0 and `max_tokens` 1000). gbench's
zero-shot prompt carries the **full database schema** (every table's `CREATE TABLE` DDL,
introspected from the `.sqlite`), the instance's **external-knowledge document** when it has one,
and the question, and asks for one SQLite query in a `` ```sql `` block. (A bare question with no
schema is not the Spider2-lite task; the model cannot know the table/column names.) The sampling is
the run's (see *Notes*); the result states both in `prompt`, and `leaderboard_comparable_reason`
names them.

`leaderboard_comparable_reason` also names an `--eval-limit` subset (how many of the local
instances ran, drawn stratified with a fixed seed) and the replies the run's token budget cut off
(`replies_cut_off_by_budget`, `replies_cut_off_in_reasoning`).

Only the **`local*` (SQLite) subset, 135 instances, runs offline.** The 180 BigQuery + 207
Snowflake + 25 BigQuery-GA instances of the 547-instance set need live cloud accounts/credentials
and are **excluded** (not scored 0). Because it is a subset, `leaderboard_comparable` is always
`false`. The reported `total_questions` is the scorable local subset; check the run log line
`Loaded N spider2 local(SQLite) samples`.

## Requirements

- `pandas` + `sqlite3`: `pandas` is a base gbench dependency; `sqlite3` ships with Python. No cloud
  SDKs are needed for the local subset. No network at run time (all data is local, see below).
- **The Spider2 repo checkout, at the pinned commit**: set **`GBENCH_SPIDER2_REPO_DIR`** to the
  `spider2-lite/` directory of an `xlang-ai/Spider2` clone checked out at
  `cafb867313aab4e674652054198f383cf4018943`. Question/db/external-knowledge **metadata is read
  from the repo** (`spider2-lite.jsonl`), NOT from the HF `xlangai/spider2-lite` snapshot, which is
  stale and drifts from the current gold (different questions/DB casing for some ids).
  External-knowledge documents are read from `<repo>/resource/documents/*.md`.
  ```bash
  git clone https://github.com/xlang-ai/Spider2
  git -C Spider2 checkout cafb867313aab4e674652054198f383cf4018943
  export GBENCH_SPIDER2_REPO_DIR=$PWD/Spider2/spider2-lite
  ```
  If `GBENCH_SPIDER2_REPO_DIR` is unset it is derived from `GBENCH_SPIDER2_GOLD_DIR` (two levels
  up), so a standard checkout only needs `GBENCH_SPIDER2_GOLD_DIR` set. The run reads the commit
  checked out in each directory (`git rev-parse HEAD`) and records it in `spider2_repo_revisions`
  (`metadata`, `gold`) beside `spider2_repo_pinned`; a checkout at another commit, or one whose
  commit cannot be read (not a git clone), is named in `leaderboard_comparable_reason`, since the
  questions, documents and gold can differ between commits.
- **Gold**: set `GBENCH_SPIDER2_GOLD_DIR` to `<repo>/spider2-lite/evaluation_suite/gold` (must
  contain `spider2lite_eval.jsonl` and `exec_result/`).
- **Local DBs**: set `GBENCH_SPIDER2_LOCALDB_DIR` to the directory of `*.sqlite` files. These are
  **not in the git repo**; download the `spider2-localdb` archive linked from
  `spider2-lite/evaluation_suite/README.md` (a Google Drive archive) and unzip it. Each instance's
  `db` maps 1:1 to `{db}.sqlite`. Concretely (Google Drive file id from the upstream README):
  ```bash
  pip install -U gdown
  gdown 1coEVsCZq-Xvj9p2TnhBFoFTsY-UoYGmG -O /tmp/spider2-localdb.zip
  mkdir -p "$GBENCH_SPIDER2_REPO_DIR/resource/databases/spider2-localdb"
  unzip -q /tmp/spider2-localdb.zip -d "$GBENCH_SPIDER2_REPO_DIR/resource/databases/spider2-localdb"
  # point at whichever dir directly contains the *.sqlite files (flatten one level if the zip nested them)
  export GBENCH_SPIDER2_LOCALDB_DIR="$GBENCH_SPIDER2_REPO_DIR/resource/databases/spider2-localdb"
  ```
  The bare `SPIDER2_REPO_DIR` / `SPIDER2_GOLD_DIR` / `SPIDER2_LOCALDB_DIR` names still work as
  deprecated aliases.

The suite **hard-errors** (`infra_required`, never skips, never a fabricated 0%) if pandas, the repo
metadata, the gold dir, or the local DB dir is missing, **and** if the checkout is incomplete (any
referenced `{db}.sqlite` or external-knowledge `.md` is absent, it lists them rather than silently
scoring those instances wrong). Each SQLite DB is copied into an in-memory connection before the
query runs, so the on-disk files are never mutated.

## Run
```bash
export GBENCH_SPIDER2_REPO_DIR=/data/Spider2/spider2-lite
export GBENCH_SPIDER2_GOLD_DIR=$GBENCH_SPIDER2_REPO_DIR/evaluation_suite/gold
export GBENCH_SPIDER2_LOCALDB_DIR=/data/spider2/localdb
gbench --evals-only --remote-endpoint http://127.0.0.1:8000/v1 \
       --tokenizer google/gemma-4-E4B-it --evals spider2 \
       --sandboxes 8 --eval-limit 20 --max-output-tokens 16384
```
The prompt asks for SQL in a `sql` fenced block; a reply without one is run as SQL whole, as
upstream's `extract_sql_query` does.

## Notes / caveats

- **Sampling:** `GBENCH_SPIDER2_TEMPERATURE` overrides the temperature for this suite (else gbench's
  rule for every suite: 0.0 without `--thinking`, 1.0 with it); it takes precedence over `--temperature`.
- **Concurrency:** `--sandboxes` bounds concurrent **model generation** (HTTP requests). SQL grading
  runs concurrently over all traces (each query capped at **120 s**, a gbench cap: upstream's
  local path sets no time limit); a query that times out counts as a wrong answer (the model's
  query did not finish). A grading-harness crash (for example
  reading a gold CSV) is a harness error, never the model's outcome: the trace is recorded as
  `SCORING_ERROR` with its `harness_error`, left out of `accuracy` (`scoring_excluded`), counted in
  `scoring_errors` and named in `status_reason` (status `completed_with_errors`).
- A request that got no reply from the served endpoint (failed, timed out, or over the context
  window) is a harness error too: it is not graded, it is left out of `accuracy`, counted
  (`scoring_excluded`, `request_harness_errors`) and named in `status_reason`. When nothing is
  scored, `accuracy` is null with `accuracy_withheld_reason`.
- **`leaderboard_comparable` is always `false`**: the SQLite-only subset is not the full published
  Spider2-lite set (which includes BigQuery + Snowflake), and the prompt and sampling are gbench's
  own; the reason also names grader crashes, requests with no reply, and a checkout that is not
  the pinned commit when there are any.
- **Reading the result:** `prompt` (what the prompt is and why it is gbench's own), `scoring`
  (the upstream code the scoring follows), `spider2_repo_pinned` and `spider2_repo_revisions`
  (the commit checked out where the metadata and the gold are read, `null` when unreadable),
  `scoring_errors` (grader crashes). Each graded trace carries upstream's `error_info` when the
  instance scored 0 (the SQLite error, or `Result Error`).

## Sharding

`--shard I/N` (or `GBENCH_SHARD=I/N`) runs shard I of N of the suite's full, ordered local SQLite
instances (round-robin: positions I, I+N, I+2N, ...), selected before the `--eval-limit` draw,
which is then taken within the shard (N shards run up to N x `--eval-limit` local SQLite
instances). The result records `shard`, `shard_index` and `shard_count`, and a shard of more than
one is not leaderboard-comparable (the shard reason is named in `leaderboard_comparable_reason`);
`gbench --merge` combines the shards' results as docs/distributed-gbench.md describes: it
re-counts the union of their traces, and refuses a suite whose headline is not a correct / scored
count over them unless the suite defines its own merge.
