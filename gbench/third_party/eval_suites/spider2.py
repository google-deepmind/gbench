# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Evaluation Suite: spider2
# Description: Spider 2.0-lite - enterprise text-to-SQL, execution accuracy (SQLite subset)

"""gbench native built-in runner for spider2 (Data & SQL Engineering).

Canonical Spider 2.0-lite (xlang-ai/Spider2) execution accuracy: run the model's SQL
against the target SQLite database, write the result to CSV and read it back, and compare it to
the gold (column-subset containment, numeric tolerance, NA-normalized, order-insensitive,
multi-gold) with the official evaluation_suite/evaluate.py code, copied verbatim (EVALUATE_PY).
The prompt is gbench's own (PROMPT_NOTE; Spider2-lite prescribes none): the full DB schema, any
external-knowledge document and the question. Only the 135 local* (SQLite)
instances of the 547-instance set run offline; the BigQuery + Snowflake instances need cloud
accounts and are excluded, so leaderboard_comparable is always False. Metadata is read from the
repo checkout (GBENCH_SPIDER2_REPO_DIR, pinned at SPIDER2_COMMIT; the run records the checkout's
commit), not the stale HF snapshot; gold + local DBs are not on HF -
point GBENCH_SPIDER2_GOLD_DIR / GBENCH_SPIDER2_LOCALDB_DIR at them (see docs; the bare SPIDER2_*
names still work as deprecated aliases). Hard-errors (infra_required,
never skips) if any prerequisite - or the referenced dbs/knowledge docs - are absent.

Sampling: gbench's rule for every suite: temperature 0.0 for no-think runs and 1.0 with
`--thinking` (base.resolve_temperature). Override it for a whole run with
`--temperature`, or for this suite alone with `GBENCH_SPIDER2_TEMPERATURE`,
which takes precedence over both. LLM-judge grading is pinned at 0.0 and is not
affected by either.
"""

import glob
import json
import logging
import math
import os
import re
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from .base import apply_shard, finalize_shard_fields, run_eval_suite, suite_env
from .metrics import record_budget_cutoffs
from .sampling import stratified_sample, presharded_for_base
from .swebench_common import infra_required, prereqs_path

logger = logging.getLogger(__name__)

PILLAR = "Data & SQL Engineering"
DOCS_URL = "docs/evals/spider2.md"

#: The xlang-ai/Spider2 commit the metadata, external-knowledge documents, gold config and gold
#: CSVs are read from (a checkout of it, GBENCH_SPIDER2_REPO_DIR / GBENCH_SPIDER2_GOLD_DIR), and
#: that the evaluation code below is copied from. The run records the checkout's HEAD
#: (repo_revisions) and names a checkout at another commit in leaderboard_comparable_reason.
SPIDER2_COMMIT = "cafb867313aab4e674652054198f383cf4018943"
EVALUATE_PY = (f"github.com/xlang-ai/Spider2@{SPIDER2_COMMIT[:8]} "
               "spider2-lite/evaluation_suite/evaluate.py")

#: The prompt is gbench's own: Spider2-lite prescribes no generation prompt (its baselines,
#: DAIL-SQL and DIN-SQL under spider2-lite/baselines/, each build their own from preprocessed
#: schema files), so it is stated with every result.
PROMPT_NOTE = (
    "gbench's own zero-shot prompt: every table's CREATE TABLE DDL from the .sqlite, the "
    "instance's external-knowledge document when it has one, and the question, asking for one "
    "SQLite query in a ```sql block. Spider2-lite prescribes no generation prompt (its baselines "
    "each build their own, e.g. DAIL-SQL's SQLPrompt from tables_preprocessed.json with "
    "temperature 0 and max_tokens 1000), so neither the prompt nor the sampling is an official one")


# --------------------------------------------------------------------------- #
# Upstream evaluate.py (EVALUATE_PY), the local (SQLite) path. The function bodies are copied
# verbatim, except that pandas and sqlite3 are imported inside them, the pandas type annotations
# are dropped and the error `print` goes to the logger (evaluate.py itself cannot be imported: it
# imports google.cloud.bigquery and replaces sys.stdout when imported).
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def load_gold_csv(file_path: str):
    """Cache gold CSV loads to avoid repeated disk reads during evaluation."""
    import pandas as pd
    return pd.read_csv(file_path)


def compare_multi_pandas_table(pred, multi_gold, multi_condition_cols=None, multi_ignore_order=False) -> int:
    if not multi_gold:
        return 0

    if multi_condition_cols in (None, [], [[]], [None]):
        multi_condition_cols = [[] for _ in range(len(multi_gold))]
    elif len(multi_gold) > 1 and not all(isinstance(sublist, list) for sublist in multi_condition_cols):
        multi_condition_cols = [multi_condition_cols for _ in range(len(multi_gold))]

    multi_ignore_order = [multi_ignore_order for _ in range(len(multi_gold))]

    for i, gold in enumerate(multi_gold):
        if compare_pandas_table(pred, gold, multi_condition_cols[i], multi_ignore_order[i]):
            return 1
    return 0


def compare_pandas_table(pred, gold, condition_cols=None, ignore_order: bool = False) -> int:
    import pandas as pd
    tolerance = 1e-2

    def normalize(value):
        if pd.isna(value):
            return 0
        return value

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        v1 = [normalize(x) for x in v1]
        v2 = [normalize(x) for x in v2]

        if ignore_order_:
            v1 = sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))
            v2 = sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))

        if len(v1) != len(v2):
            return False

        for a, b in zip(v1, v2):
            if pd.isna(a) and pd.isna(b):
                continue
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True

    if condition_cols:
        if not isinstance(condition_cols, (list, tuple)):
            condition_cols = [condition_cols]
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold

    pred_cols = pred
    t_gold_list = gold_cols.transpose().values.tolist()
    t_pred_list = pred_cols.transpose().values.tolist()
    score = 1

    for gold_vector in t_gold_list:
        if not any(vectors_match(gold_vector, pred_vector, ignore_order_=ignore_order) for pred_vector in t_pred_list):
            score = 0
            break

    return score


def get_sqlite_result(db_path: str, query: str, save_dir=None, file_name: str = "result.csv", chunksize: int = 500, instance_id: str = None):
    import sqlite3
    import pandas as pd
    prefix = f"[{instance_id}] " if instance_id else ""

    try:
        conn = sqlite3.connect(db_path)
        memory_conn = sqlite3.connect(":memory:")
        conn.backup(memory_conn)

        try:
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                for i, chunk in enumerate(pd.read_sql_query(query, memory_conn, chunksize=chunksize)):
                    mode = "a" if i > 0 else "w"
                    header = i == 0
                    chunk.to_csv(os.path.join(save_dir, file_name), mode=mode, header=header, index=False)
                return True, None

            df = pd.read_sql_query(query, memory_conn)
            return True, df
        finally:
            memory_conn.close()
            conn.close()
    except Exception as e:
        error_message = str(e)
        logger.debug(f"{prefix}An error occurred: {error_message}")
        return False, error_message


def extract_sql_query(pred_sql_query: str) -> str:
    pattern = r"```sql\n(.*?)\n```"
    match = re.search(pattern, pred_sql_query, re.DOTALL)

    if match:
        return match.group(1).strip()
    return pred_sql_query


def resolve_gold_paths(instance_id: str, gold_result_dir: str):
    base_path = Path(gold_result_dir) / f"{instance_id}.csv"
    if base_path.exists():
        return [base_path], True

    if "_" in instance_id:
        pattern = re.compile(rf"^{re.escape(instance_id)}(_[a-z])?\.csv$")
    else:
        pattern = re.compile(rf"^{re.escape(instance_id)}(_[a-z])?\.csv$")

    csv_files = sorted(
        file for file in os.listdir(gold_result_dir)
        if pattern.match(file)
    )
    return [Path(gold_result_dir) / file for file in csv_files], False


class SqlExecutionError(Exception):
    """The predicted query failed to execute (get_sqlite_result returned False) or left no
    readable result file: scores 0, the model's outcome, as upstream scores it."""


def _get_sqlite_result(db_path: str, sql: str, instance_id: Optional[str] = None):
    """The prediction as upstream compares it: get_sqlite_result writes the query's result to CSV
    in chunks, and it is read back with pd.read_csv. Raises SqlExecutionError (with the SQLite
    error, or upstream's 'Evaluation Error' when no readable result file was written) when the
    query fails; both score 0."""
    import pandas as pd
    result_file = f"{instance_id or 'result'}.csv"
    with tempfile.TemporaryDirectory(prefix="gbench_spider2_") as tmp:
        exe_flag, dbms_error_info = get_sqlite_result(db_path, sql, save_dir=tmp,
                                                      file_name=result_file,
                                                      instance_id=instance_id)
        if not exe_flag:
            raise SqlExecutionError(dbms_error_info)
        try:
            return pd.read_csv(os.path.join(tmp, result_file))
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            # The query ran but left no readable result file (e.g. no result set to write).
            # Upstream's pd.read_csv raises inside evaluate_single_sql_instance's try, which
            # scores the instance 0 with 'Evaluation Error: ...': the model's outcome.
            raise SqlExecutionError(f"Evaluation Error: {e}") from e


def _is_single_gold(gold_paths: List[Any], instance_id: Optional[str] = None) -> bool:
    """resolve_gold_paths' is_single for paths it returned: the one `{id}.csv`, not variants."""
    if len(gold_paths) != 1:
        return False
    stem = Path(str(gold_paths[0])).stem
    return stem == instance_id if instance_id else not re.search(r"_[a-z]$", stem)


def _score_against_golds(pred_df, gold_paths: List[Any], condition_cols, ignore_order,
                         is_single: Optional[bool] = None, instance_id: Optional[str] = None) -> int:
    """evaluate_single_sql_instance's comparison: 0 with no gold file, compare_pandas_table with
    the one `{id}.csv`, compare_multi_pandas_table with the variants. An exception here is
    upstream's 'Python Script Error'; the scorer makes it a harness error."""
    if not gold_paths:
        return 0
    if is_single is None:
        is_single = _is_single_gold(gold_paths, instance_id)
    if is_single:
        return compare_pandas_table(pred_df, load_gold_csv(str(gold_paths[0])), condition_cols,
                                    ignore_order)
    return compare_multi_pandas_table(pred_df, [load_gold_csv(str(p)) for p in gold_paths],
                                      condition_cols, ignore_order)


def score_local_instance(instance_id: str, pred_text: str, sqlite_path: str,
                         gold_paths: List[Any], standard: Dict[str, Any],
                         is_single: Optional[bool] = None) -> Tuple[int, Optional[str]]:
    """evaluate_single_sql_instance's local branch: extract the SQL, run it with the result written
    to CSV and read back, and compare it with the instance's gold file(s) (resolve_gold_paths),
    condition_cols and ignore_order (default False) from the gold config.

    Returns (score, error_info) as upstream records them. A query that fails to execute scores 0
    (the model's outcome). A comparison that raises is upstream's 'Python Script Error' (score 0
    there), and no gold file is upstream's 'No matching gold file found' (score 0 there, and the
    loader refuses a checkout without one); here both raise, and the scorer makes them harness
    errors."""
    try:
        pred_pd = _get_sqlite_result(sqlite_path, extract_sql_query(pred_text), instance_id)
    except SqlExecutionError as e:
        return 0, str(e)
    if not gold_paths:
        raise FileNotFoundError(f"No matching gold file found for {instance_id or 'the instance'}")
    score = _score_against_golds(pred_pd, gold_paths, standard.get("condition_cols"),
                                 standard.get("ignore_order", False), is_single, instance_id)
    return score, (None if score else "Result Error")


def _gold_dir() -> Optional[str]:
    return prereqs_path("Spider2/spider2-lite/evaluation_suite/gold",
                        suite_env("GBENCH_SPIDER2_GOLD_DIR", "SPIDER2_GOLD_DIR"))


def _localdb_dir() -> Optional[str]:
    return prereqs_path("Spider2/spider2-lite/resource/databases/spider2-localdb",
                        suite_env("GBENCH_SPIDER2_LOCALDB_DIR", "SPIDER2_LOCALDB_DIR"))


def _repo_dir() -> Optional[str]:
    """The `spider2-lite/` dir of an xlang-ai/Spider2 checkout.

    Canonical scoring (upstream evaluate.py) sources the question/db metadata from the SAME repo
    checkout as the gold - NOT from HF `xlangai/spider2-lite`, which is a stale snapshot whose
    questions/dbs drift from the current gold. Prefer GBENCH_SPIDER2_REPO_DIR; else derive it from
    GBENCH_SPIDER2_GOLD_DIR (which is <repo>/evaluation_suite/gold, so the repo is two levels up).
    """
    explicit = prereqs_path("Spider2/spider2-lite",
                            suite_env("GBENCH_SPIDER2_REPO_DIR", "SPIDER2_REPO_DIR"))
    if explicit:
        return explicit
    gd = _gold_dir()
    if gd:
        return os.path.dirname(os.path.dirname(os.path.abspath(gd)))  # gold -> evaluation_suite -> spider2-lite
    return None


def _metadata_path() -> Optional[str]:
    repo = _repo_dir()
    return os.path.join(repo, "spider2-lite.jsonl") if repo else None


def _documents_dir() -> Optional[str]:
    repo = _repo_dir()
    return os.path.join(repo, "resource", "documents") if repo else None


#: {db: "CREATE TABLE ..." joined} - introspected once per db from its .sqlite (self-contained;
#: matches the CREATE TABLE DDL the canonical DAIL-SQL baseline feeds, without the repo's
#: schema-dir casing quirks since the .sqlite filename is `{db}.sqlite` per evaluate.py).
_SCHEMA_CACHE: Dict[str, str] = {}


def _schema_for_db(localdb_dir: str, db: str) -> Optional[str]:
    """Full DDL (all CREATE TABLE statements) for a local SQLite db, or None if the file is absent."""
    if db in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[db]
    import sqlite3
    db_path = os.path.join(localdb_dir, f"{db}.sqlite")
    if not os.path.isfile(db_path):
        return None
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL "
            "ORDER BY name").fetchall()
    finally:
        con.close()
    schema = "\n\n".join(r[0].strip() for r in rows if r and r[0])
    _SCHEMA_CACHE[db] = schema
    return schema


def _resolve_gold_paths(gold_dir: str, iid: str) -> List[str]:
    """The gold CSV(s) upstream compares an instance with (resolve_gold_paths on
    `<gold>/exec_result`): `{iid}.csv` alone when it exists, else the single-letter variants
    `{iid}_a.csv` ... (a prediction passes if it matches ANY of them)."""
    exec_dir = os.path.join(gold_dir, "exec_result")
    if not os.path.isdir(exec_dir):
        return []
    return [str(p) for p in resolve_gold_paths(iid, exec_dir)[0]]


def _git_head(path: Optional[str]) -> Optional[str]:
    """The commit checked out in the git work tree holding `path`, or None when it cannot be
    read (not a git checkout, git missing, ...)."""
    if not path or not os.path.isdir(path):
        return None
    try:
        res = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=30)
    except Exception:                                                   # noqa: BLE001
        return None
    head = (res.stdout or "").strip()
    return head if res.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", head) else None


def repo_revisions() -> Dict[str, Optional[str]]:
    """The Spider2 commit checked out where the metadata (GBENCH_SPIDER2_REPO_DIR) and the gold
    (GBENCH_SPIDER2_GOLD_DIR) are read from."""
    return {"metadata": _git_head(_repo_dir()), "gold": _git_head(_gold_dir())}


def revision_gaps(revisions: Dict[str, Optional[str]]) -> List[str]:
    """Why the checkout is not the pinned SPIDER2_COMMIT, one entry per source; [] when both are."""
    gaps = []
    for source, head in sorted(revisions.items()):
        if head is None:
            gaps.append(f"the {source} checkout's commit could not be read (git rev-parse HEAD), "
                        f"so it is not verified as the pinned Spider2 {SPIDER2_COMMIT[:8]}")
        elif head != SPIDER2_COMMIT:
            gaps.append(f"the {source} checkout is Spider2 {head[:8]}, not the pinned "
                        f"{SPIDER2_COMMIT[:8]} (questions, documents and gold can differ)")
    return gaps


def check_spider2_prerequisites() -> Tuple[bool, str]:
    """pandas + repo metadata + local gold dir + local SQLite DB dir (SQLite subset runs offline)."""
    try:
        import pandas  # noqa: F401
    except ImportError:
        return False, "Python package 'pandas' is not installed."
    gd, ld = _gold_dir(), _localdb_dir()
    if not gd or not os.path.isfile(os.path.join(gd, "spider2lite_eval.jsonl")) \
            or not os.path.isdir(os.path.join(gd, "exec_result")):
        return False, ("Spider2 gold not found: set GBENCH_SPIDER2_GOLD_DIR to a checkout of "
                       "xlang-ai/Spider2/spider2-lite/evaluation_suite/gold "
                       "(needs spider2lite_eval.jsonl + exec_result/).")
    meta = _metadata_path()
    if not meta or not os.path.isfile(meta):
        return False, ("Spider2 metadata (spider2-lite.jsonl) not found: set GBENCH_SPIDER2_REPO_DIR to the "
                       "spider2-lite/ dir of an xlang-ai/Spider2 checkout (or leave it unset to "
                       "derive it from GBENCH_SPIDER2_GOLD_DIR). Canonical scoring reads questions/dbs from "
                       "the repo, NOT the stale HF snapshot.")
    if not ld or not glob.glob(os.path.join(ld, "*.sqlite")):
        return False, ("Spider2 local SQLite DBs not found: set GBENCH_SPIDER2_LOCALDB_DIR to the "
                       "unzipped spider2-localdb (*.sqlite) directory.")
    return True, ""


def _load_gold_config(gold_dir: str) -> Dict[str, Dict[str, Any]]:
    cfg = {}
    with open(os.path.join(gold_dir, "spider2lite_eval.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                cfg[row["instance_id"]] = row
    return cfg


def _load_spider2_samples(
    limit: Optional[int] = None,
) -> List[Tuple[List[Dict[str, Any]], Any, Dict[str, Any]]]:
    """Load Spider2-lite local* (SQLite) instances joined with local gold; raises on failure.

    Canonical: metadata (question/db/external_knowledge) comes from the REPO checkout (same source
    as the gold), the prompt carries the full DB schema + any external-knowledge document, and the
    scored set is the 135 local SQLite instances. A local db whose .sqlite (or a referenced
    external-knowledge doc) is missing is an INFRA gap - hard-error listing them, never silently
    score those instances wrong.
    """
    gold_dir = _gold_dir()
    localdb_dir = _localdb_dir()
    docs_dir = _documents_dir()
    meta_path = _metadata_path()
    try:
        rows = [json.loads(l) for l in open(meta_path, encoding="utf-8") if l.strip()]
    except Exception as e:
        logger.error(f"Failed to load spider2 metadata from {meta_path!r}: {e}")
        raise RuntimeError(f"Could not load spider2 metadata ({meta_path!r}): {e}") from e

    gold_cfg = _load_gold_config(gold_dir)
    samples = []
    missing_db: List[str] = []
    missing_doc: List[str] = []
    missing_gold: List[str] = []
    for item in rows:
        iid = item.get("instance_id")
        if not iid or not iid.startswith("local"):
            continue  # offline SQLite subset only
        g = gold_cfg.get(iid)
        if g is None:
            continue  # not in the official gold config -> not an officially-scored instance
        # An officially-scored instance whose exec_result CSV is absent is an INCOMPLETE checkout,
        # not a "skip this one": fold it into the hard-error below instead of silently dropping it.
        gold_paths = _resolve_gold_paths(gold_dir, iid)
        if not gold_paths:
            missing_gold.append(iid)
            continue
        db = item.get("db")
        question = item.get("question")
        if not db or not question:
            raise RuntimeError("spider2: unexpected schema (db/question); refusing to fabricate")

        # Full DB schema (canonical baselines feed every table's DDL). A missing .sqlite is infra.
        schema = _schema_for_db(localdb_dir, db)
        if schema is None:
            missing_db.append(f"{iid}({db}.sqlite)")
            continue

        # external_knowledge is a single .md filename (or null) resolved from resource/documents/.
        ek_name = item.get("external_knowledge")
        ek_text = ""
        if ek_name:
            ek_path = os.path.join(docs_dir, ek_name) if docs_dir else None
            if ek_path and os.path.isfile(ek_path):
                with open(ek_path, encoding="utf-8") as f:
                    ek_text = f.read().strip()
            else:
                missing_doc.append(f"{iid}({ek_name})")
                continue

        parts = ["/* Given the following database schema: */", schema]
        if ek_text:
            parts += ["", "/* External knowledge (read carefully): */", ek_text]
        parts += [
            "",
            f"Database: {db}",
            "",
            "Question:",
            question,
            "",
            "Write a single complete SQLite SQL query that answers the question. "
            "Return only the SQL inside a ```sql ... ``` code block.",
        ]
        # The gold config entry as upstream reads it (evaluate_single_sql_instance):
        # condition_cols as given, ignore_order defaulting to False.
        samples.append(([{"role": "user", "content": "\n".join(parts)}], iid, {
            "category": "sqlite", "db": db, "instance_id": iid,
            "condition_cols": g.get("condition_cols"),
            "ignore_order": g.get("ignore_order", False),
            "gold_paths": gold_paths,
        }))

    # No-partial: an incomplete checkout (missing local db, external-knowledge doc, or the gold CSV
    # of an officially-scored instance) must hard-error, not silently drop or mis-score instances
    # (the old code let a missing db be graded as a wrong answer and a missing gold CSV be dropped).
    if missing_db or missing_doc or missing_gold:
        bits = []
        if missing_db:
            bits.append(f"{len(missing_db)} local db file(s) absent from GBENCH_SPIDER2_LOCALDB_DIR: "
                        + ", ".join(missing_db[:10]) + (" ..." if len(missing_db) > 10 else ""))
        if missing_doc:
            bits.append(f"{len(missing_doc)} external-knowledge doc(s) absent from "
                        f"{docs_dir!r}: " + ", ".join(missing_doc[:10])
                        + (" ..." if len(missing_doc) > 10 else ""))
        if missing_gold:
            bits.append(f"{len(missing_gold)} gold exec_result CSV(s) absent for scored "
                        f"instance(s): " + ", ".join(missing_gold[:10])
                        + (" ..." if len(missing_gold) > 10 else ""))
        from .swebench_common import infra_required
        raise infra_required("spider2",
                             "the Spider2 checkout is incomplete - " + "; ".join(bits), DOCS_URL)

    if not samples:
        raise RuntimeError("spider2: no scorable local SQLite instances found")
    # --shard first, over the full ordered set, then the --eval-limit draw within the shard.
    samples, _ = apply_shard(samples, "spider2")
    # Stratified, not a contiguous head.
    samples = stratified_sample(samples, limit, None, seed="spider2")
    # Canonical local set: 135 SQLite instances (of the repo's 547; the 180 BigQuery + 207
    # Snowflake + 25 BigQuery-GA instances need cloud creds and are excluded).
    logger.info("Loaded %d spider2 local(SQLite) samples (of 135 local in the 547-instance "
                "spider2-lite set; BigQuery/Snowflake need cloud creds).", len(samples))
    return samples


def _make_scorer(localdb_dir: str, gold_dir: Optional[str] = None):
    async def _score(sample_traces: List[Dict[str, Any]]) -> None:
        import asyncio
        gold_result_dir = os.path.join(gold_dir or _gold_dir() or "", "exec_result")

        def _grade(tr: Dict[str, Any]) -> bool:
            # upstream evaluate_single_sql_instance's local branch (score_local_instance), with
            # the gold files the loader resolved (resolve_gold_paths)
            extra = tr.get("extra_payload") or {}
            db = extra.get("db")
            if not db:
                return False
            iid = extra.get("instance_id")
            db_path = os.path.join(localdb_dir, f"{db}.sqlite")
            if not os.path.isfile(db_path):
                # a harness error (sqlite3.connect would create an empty database there)
                raise FileNotFoundError(f"{db_path} is missing")
            gold_paths = extra.get("gold_paths")
            if gold_paths is None and iid:
                gold_paths = [str(p) for p in resolve_gold_paths(iid, gold_result_dir)[0]]
            score, error_info = score_local_instance(
                str(iid or ""), tr.get("response_text") or "", db_path, list(gold_paths or []),
                {k: extra[k] for k in ("condition_cols", "ignore_order") if k in extra})
            if error_info:
                tr["error_info"] = str(error_info)[:300]
            return score == 1

        async def _one(tr):
            if tr.get("scoring_excluded"):
                return      # no reply from the endpoint: a harness error base already marked
            try:
                tr["is_correct"] = await asyncio.wait_for(asyncio.to_thread(_grade, tr), timeout=120)
                tr["status"] = "OK"
            except asyncio.TimeoutError:
                # A query that will not finish in 120s fails execution accuracy (no result set) -
                # that IS a wrong answer, so is_correct=False, but note why.
                tr["is_correct"] = False
                tr["status"] = "OK"
                tr["scoring_note"] = "query timed out (>120s)"
            except Exception as e:
                # A grading-harness crash (e.g. reading a gold CSV) is a harness error, never the
                # model's outcome: left out of the score (scoring_excluded, which base names in
                # status_reason), marked SCORING_ERROR and counted in result['scoring_errors'].
                tr["is_correct"] = False
                tr["status"] = "SCORING_ERROR"
                tr["scoring_error"] = f"{type(e).__name__}: {e}"
                tr["scoring_excluded"] = True
                tr["harness_error"] = {"kind": "grader_error",
                                       "detail": tr["scoring_error"][:300]}

        await asyncio.gather(*[_one(t) for t in sample_traces])
    return _score


def run_spider2(
    model_name: str,
    base_url: str,
    concurrency: int,
    enable_thinking: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run Spider2-lite SQLite execution accuracy (hard-errors if gold/repo/DBs absent)."""
    ok, reason = check_spider2_prerequisites()
    if not ok:
        # No-skip policy: missing prereqs HARD-ERROR (never a fabricated 0%/skip row).
        raise infra_required("spider2", reason, DOCS_URL)
    samples = _load_spider2_samples(limit=kwargs.get("limit"))
    result = run_eval_suite(
        eval_name="spider2",
        model_name=model_name,
        base_url=base_url,
        concurrency=concurrency,
        samples=presharded_for_base(samples),  # this shard's instances, kept as is
        async_eval_fn=_make_scorer(_localdb_dir(), _gold_dir()),
        declared_scoring_mode="execution",  # SQL execution + result comparison, not an LLM judge
        thinking=enable_thinking,
        extra_payload=kwargs.get("extra_payload"),
        limit=kwargs.get("limit"),
        max_output_tokens=kwargs.get("max_output_tokens", 4096),
        temperature=kwargs.get("temperature"),
    )
    # gbench runs ONLY the offline local* (SQLite) subset of Spider2-lite (the BigQuery + Snowflake
    # instances need live cloud accounts and are excluded), so the number is NEVER comparable to the
    # published Spider2-lite leaderboard, which is over the full set. Record that unconditionally.
    result["leaderboard_comparable"] = False
    result["leaderboard_comparable_reason"] = (
        "SQLite-only subset of Spider2-lite (BigQuery + Snowflake instances excluded); not the "
        "full published leaderboard set; the prompt and sampling are gbench's own (Spider2-lite "
        "prescribes none)")
    result["prompt"] = PROMPT_NOTE
    result["scoring"] = (f"{EVALUATE_PY}, local path: extract_sql_query, get_sqlite_result with "
                         "the result written to CSV and read back, resolve_gold_paths, "
                         "compare_pandas_table / compare_multi_pandas_table (copied verbatim)")
    # The checkout the metadata and gold come from, against the pinned commit.
    revisions = repo_revisions()
    result["spider2_repo_pinned"] = SPIDER2_COMMIT
    result["spider2_repo_revisions"] = revisions
    for gap in revision_gaps(revisions):
        result["leaderboard_comparable_reason"] += f"; {gap}"
    limit = kwargs.get("limit")
    if limit:
        result["leaderboard_comparable_reason"] += (
            f"; subset run (--eval-limit {limit}: {len(result.get('sample_traces') or [])} of the "
            f"local instances, drawn stratified with a fixed seed)")
    # replies the run's token budget cut off (Spider2-lite sends none of its own)
    cut_reason = record_budget_cutoffs(result)
    if cut_reason:
        result["leaderboard_comparable_reason"] += f"; {cut_reason}"
    # Grading-harness crashes (SCORING_ERROR) are harness errors, left out of the score; the
    # requests that got no reply from the endpoint are too (base). Both are named here.
    traces = result.get("sample_traces", []) or []
    scoring_errors = sum(1 for t in traces if t.get("status") == "SCORING_ERROR")
    if scoring_errors:
        result["scoring_errors"] = scoring_errors
        result["leaderboard_comparable_reason"] += (
            f"; {scoring_errors} response(s) the grader crashed on (harness errors, left out of "
            "the score)")
    no_reply = sum(1 for t in traces if t.get("harness_error")
                   and t.get("status") != "SCORING_ERROR")
    if no_reply:
        result["leaderboard_comparable_reason"] += (
            f"; {no_reply} request(s) got no reply from the served endpoint (harness errors, left "
            "out of the score)")
    # One shard of a --shard I/N run is a partial set: the shard reason (finalize_shard_fields).
    return finalize_shard_fields(result)
