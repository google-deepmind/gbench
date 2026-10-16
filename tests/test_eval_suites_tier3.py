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

"""Tier-3 execution-harness eval suites: spider2, ojbench, swe_bench_pro, multi_swe_bench.

These four are gated behind heavy external harnesses (SQLite gold DBs / the DMOJ
sandbox / the SWE-bench-Pro & Multi-SWE-bench Docker harnesses) that are absent on
CI, so they SKIP cleanly here. These tests lock in: (1) registration + SANDBOX
concurrency wiring, (2) clean skip dicts with the right docs URL, (3) loaders that
raise (never fabricate) on a bad schema and slice canonically, (4) spider2's
result-set comparison semantics, which run fully offline against a synthetic DB.
"""

import asyncio
import csv
import json
import os
import sqlite3
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from gbench.runners.eval_suites import SUITES


# --------------------------------------------------------------------------- #
# registration + sandbox wiring
# --------------------------------------------------------------------------- #
def test_tier3_suites_registered():
    for name in ("spider2", "ojbench", "swe_bench_pro", "multi_swe_bench"):
        assert name in SUITES, f"{name} not registered in SUITES"


@pytest.mark.parametrize("name", ["spider2", "ojbench", "swe_bench_pro", "multi_swe_bench"])
def test_tier3_is_a_sandbox_eval(name):
    """--sandboxes must override concurrency for each Tier-3 (execution) suite."""
    from gbench.runners.evals import EvalsBenchmarkRunner
    from gbench.core.config import BenchmarkConfig

    config = BenchmarkConfig()
    config.sandboxes = 7
    runner = EvalsBenchmarkRunner(config)
    with patch.dict("gbench.runners.eval_suites.SUITES",
                    {name: MagicMock(return_value={"status": "success"})}) as mock_suites:
        runner._run_single_eval(name, "gemma-4-E4B-it", "http://localhost:8000/v1", num_threads=128)
        assert mock_suites[name].call_args[1]["concurrency"] == 7


# --------------------------------------------------------------------------- #
# clean skip dicts
# --------------------------------------------------------------------------- #
def test_spider2_hard_errors_on_missing_prereqs():
    # No-skip policy: an un-provisioned canonical harness must hard-error, not skip.
    from gbench.runners.eval_suites.spider2 import run_spider2
    with patch("gbench.runners.eval_suites.spider2.check_spider2_prerequisites",
               return_value=(False, "gold not found")):
        with pytest.raises(RuntimeError, match="docs/evals/spider2.md"):
            run_spider2("gemma-4-E4B-it", "http://localhost:8000/v1", 2)


def test_ojbench_hard_errors_on_missing_prereqs():
    # No-skip policy: an un-provisioned canonical harness must hard-error, not skip.
    from gbench.runners.eval_suites.ojbench import run_ojbench
    with patch("gbench.runners.eval_suites.ojbench.check_ojbench_prerequisites",
               return_value=(False, "ojbench not installed")):
        with pytest.raises(RuntimeError, match="docs/evals/ojbench.md"):
            run_ojbench("gemma-4-E4B-it", "http://localhost:8000/v1", 2)


def test_swe_bench_pro_hard_errors_on_missing_prereqs():
    # No-skip policy: missing harness/Docker/image must hard-error, not skip. The default route
    # checks the V2 prerequisites, the engine route the v1 ones; both are stubbed as missing so
    # the test never starts a real run on a provisioned machine.
    from gbench.runners.eval_suites.swe_bench_pro import run_swe_bench_pro
    with patch("gbench.runners.eval_suites.swe_bench_pro.check_swe_bench_pro_prerequisites",
               return_value=(False, "harness not found")), \
         patch("gbench.runners.eval_suites.swe_bench_pro.check_swe_bench_pro_v2_prerequisites",
               return_value=(False, "image not built")):
        with pytest.raises(RuntimeError, match="docs/evals/swe_bench_pro.md"):
            run_swe_bench_pro("gemma-4-E4B-it", "http://localhost:8000/v1", 2)


def test_swe_bench_pro_needs_no_opt_in():
    """With docker + harness + the raw-sample table present, the v1 check passes without any
    opt-in variable: cost is not a prerequisite."""
    from gbench.runners.eval_suites import swe_bench_pro as SP
    with tempfile.TemporaryDirectory() as hd:
        open(os.path.join(hd, "swe_bench_pro_eval.py"), "w").close()
        os.mkdir(os.path.join(hd, "run_scripts"))
        table = os.path.join(hd, SP.RAW_SAMPLE_NAME)
        env = {"SWE_BENCH_PRO_HARNESS_DIR": hd}  # note: SWE_BENCH_PRO_RUN unset
        with patch.dict(os.environ, env, clear=False), \
             patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)), \
             patch.dict("sys.modules", {"docker": MagicMock()}), \
             patch.object(SP, "_resolve_raw_sample", return_value=(table, "")):
            os.environ.pop("SWE_BENCH_PRO_RUN", None)
            os.environ.pop("GBENCH_SWE_BENCH_PRO_RUN", None)
            assert SP.check_swe_bench_pro_prerequisites() == (True, "")


def test_multi_swe_bench_hard_errors_on_missing_prereqs():
    # No-skip policy: an un-provisioned canonical harness must hard-error, not skip.
    from gbench.runners.eval_suites.multi_swe_bench import run_multi_swe_bench
    with patch("gbench.runners.eval_suites.multi_swe_bench.check_multi_swe_bench_prerequisites",
               return_value=(False, "multi_swe_bench not installed")):
        with pytest.raises(RuntimeError, match="docs/evals/multi_swe_bench.md"):
            run_multi_swe_bench("gemma-4-E4B-it", "http://localhost:8000/v1", 2)


def test_multi_swe_bench_prereq_survives_absent_parent_package():
    """find_spec RAISES (rather than returning None) when the parent package is absent.

    Hermetic: the old version asserted the real package was missing, so it silently
    became a no-op the day `multi-swe-bench` was installed on the box - and then failed
    outright. Mock the absence instead of depending on the environment.
    """
    from gbench.runners.eval_suites.multi_swe_bench import check_multi_swe_bench_prerequisites
    with patch("shutil.which", return_value="/usr/bin/docker"), \
         patch("subprocess.run", return_value=MagicMock(returncode=0)), \
         patch("importlib.util.find_spec", side_effect=ModuleNotFoundError("no parent")):
        ok, reason = check_multi_swe_bench_prerequisites()
    assert ok is False and "multi-swe-bench" in reason      # raised, and was caught


def test_multi_swe_bench_prereq_passes_when_the_package_is_present():
    from gbench.runners.eval_suites.multi_swe_bench import check_multi_swe_bench_prerequisites
    with patch("shutil.which", return_value="/usr/bin/docker"), \
         patch("subprocess.run", return_value=MagicMock(returncode=0)), \
         patch("importlib.util.find_spec", return_value=object()), \
         patch("gbench.runners.eval_suites.multi_swe_bench._installed_harness_version",
               return_value="1.1.2"):
        ok, reason = check_multi_swe_bench_prerequisites()
    assert ok is True and reason == ""


# --------------------------------------------------------------------------- #
# loaders: canonical slicing + no fabrication (raise on bad schema)
# --------------------------------------------------------------------------- #
def test_swe_bench_pro_loader_conditional_fields_and_raises():
    from gbench.runners.eval_suites import swe_bench_pro as SP
    diff = "diff --git a/x.py b/x.py\n@@\n-a\n+b\n"
    rows = [
        {"instance_id": "o__r-1", "repo": "o/r", "problem_statement": "fix",
         "requirements": "reqs", "interface": None, "patch": diff, "repo_language": "Python"},
        {"instance_id": "o__r-2", "repo": "o/r", "problem_statement": "fix2",
         "requirements": None, "interface": "def f(): ...", "patch": diff, "repo_language": "Go"},
    ]
    with patch.object(SP, "load_v2_rows", return_value=rows):
        samples = SP._load_swe_bench_pro_samples()
    assert [g for _m, g, _e in samples] == [diff, diff]
    assert samples[0][2]["instance_id"] == "o__r-1" and samples[0][2]["category"] == "Python"
    assert "Requirements:" in samples[0][0][0]["content"]
    assert "Interface:" not in samples[0][0][0]["content"]
    assert "Interface:" in samples[1][0][0]["content"]
    with patch("datasets.load_dataset", return_value=[{"repo": "x"}]):
        with pytest.raises(RuntimeError, match="expected 642"):
            SP._load_swe_bench_pro_samples()


def test_ojbench_loader_verbatim_prompt_and_raises():
    from gbench.runners.eval_suites import ojbench as OJ
    with tempfile.TemporaryDirectory() as d:
        good = os.path.join(d, "full.jsonl")
        with open(good, "w") as f:
            f.write(json.dumps({"id": 101, "prompt": "solve X", "language": "cpp",
                                "dataset": "NOI", "difficulty": "hard"}) + "\n")
        with patch("huggingface_hub.hf_hub_download", return_value=good):
            samples = OJ._load_ojbench_samples()
        assert samples[0][1] == 101
        assert samples[0][0][0]["content"] == "solve X"       # prompt VERBATIM
        assert samples[0][2]["category"] == "cpp"       # per-language columns (paper Table 2)
        assert samples[0][2]["row"] == {"id": 101, "dataset": "NOI",
                                        "language": "cpp", "difficulty": "hard"}
        bad = os.path.join(d, "bad.jsonl")
        with open(bad, "w") as f:
            f.write(json.dumps({"id": 1}) + "\n")
        with patch("huggingface_hub.hf_hub_download", return_value=bad):
            with pytest.raises(RuntimeError, match="refusing to fabricate"):
                OJ._load_ojbench_samples()


def test_ojbench_scorer_maps_by_id_and_language():
    from gbench.runners.eval_suites import ojbench as OJ
    captured = {}

    def fake_judge(records, num_workers):     # judging is now containerized (_judge_in_container)
        captured["records"] = records
        return [{"id": 101, "language": "cpp", "is_passed": True},
                {"id": 102, "language": "python", "is_passed": False}]

    with patch.object(OJ, "_judge_in_container", fake_judge):
        scorer = OJ._make_scorer(2)
        traces = [
            {"response_text": "<thought>x</thought>ans", "extra_payload": {"row": {"id": 101, "language": "cpp"}}},
            {"response_text": "ans2", "extra_payload": {"row": {"id": 102, "language": "python"}}},
            {"response_text": "ans3", "extra_payload": {"row": {"id": 999, "language": "cpp"}}},
        ]
        asyncio.run(scorer(traces))
    assert [t["is_correct"] for t in traces] == [True, False, False]
    # reasoning/thought tags are stripped before judging (now in the container-bound records)
    assert captured["records"][0]["content"] == "ans"


def test_multi_swe_bench_loader_and_src_cache():
    from gbench.runners.eval_suites import multi_swe_bench as MS
    diff = "diff --git a/x b/x\n@@\n"
    with tempfile.TemporaryDirectory() as d:
        jl = os.path.join(d, "python_dataset.jsonl")
        with open(jl, "w") as f:
            f.write(json.dumps({"org": "o", "repo": "r", "number": 5, "title": "t",
                                "body": "b", "base": {"sha": "abc"},
                                "fix_patch": diff, "instance_id": "o__r-5"}) + "\n")

        class _Api:
            def list_repo_files(self, *a, **k):
                # python/ is not one of the benchmark's seven languages, so the loader must
                # ignore it; go/ is.
                return ["python/python_dataset.jsonl", "go/o__r_dataset.jsonl", "README.md"]

        with patch("huggingface_hub.HfApi", _Api), \
             patch("huggingface_hub.hf_hub_download", return_value=jl):
            samples = MS._load_multi_swe_bench_samples()
    assert len(samples) == 1
    e = samples[0][2]
    assert e["instance_id"] == "o__r-5" and e["org"] == "o" and e["number"] == 5
    assert MS._SRC_FILES.get("o__r-5") == jl


# --------------------------------------------------------------------------- #
# spider2 result-set comparison (fully offline against a synthetic SQLite DB)
# --------------------------------------------------------------------------- #
@pytest.fixture
def spider2_env():
    with tempfile.TemporaryDirectory() as base:
        db = os.path.join(base, "shop.sqlite")
        con = sqlite3.connect(db)
        con.execute("CREATE TABLE sales(region TEXT, revenue REAL)")
        con.executemany("INSERT INTO sales VALUES (?,?)",
                        [("west", 100.0), ("east", 250.0), ("west", 50.0), ("north", 75.5)])
        con.commit()
        con.close()
        gold = os.path.join(base, "local001.csv")
        with open(gold, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["region", "total"])
            # deliberately different row order than a GROUP BY would emit
            w.writerow(["north", 75.5])
            w.writerow(["east", 250.0])
            w.writerow(["west", 150.0])
        yield db, [gold]


def test_spider2_execution_accuracy_semantics(spider2_env):
    from gbench.runners.eval_suites import spider2 as S
    db, golds = spider2_env

    # correct query matches order-insensitively
    good = S._get_sqlite_result(db, "SELECT region, SUM(revenue) AS total FROM sales GROUP BY region")
    assert S._score_against_golds(good, golds, [], True) == 1
    # wrong result set does not match
    bad = S._get_sqlite_result(db, "SELECT region, SUM(revenue) AS total FROM sales "
                                    "WHERE region!='north' GROUP BY region")
    assert S._score_against_golds(bad, golds, [], True) == 0
    # predicted superset of columns still matches (subset containment)
    sup = S._get_sqlite_result(db, "SELECT region, SUM(revenue) AS total, COUNT(*) AS n "
                                    "FROM sales GROUP BY region")
    assert S._score_against_golds(sup, golds, [], True) == 1


def test_spider2_numeric_tolerance_and_extract():
    import pandas as pd
    from gbench.runners.eval_suites import spider2 as S
    assert S.extract_sql_query("x\n```sql\nSELECT 1\n```\ny") == "SELECT 1"
    assert S.extract_sql_query("SELECT 2") == "SELECT 2"
    g = pd.DataFrame({"a": [150.0]})
    assert S.compare_pandas_table(pd.DataFrame({"a": [150.004]}), g, None, True) == 1
    assert S.compare_pandas_table(pd.DataFrame({"a": [150.5]}), g, None, True) == 0


def test_spider2_async_scorer_writes_is_correct(spider2_env):
    from gbench.runners.eval_suites import spider2 as S
    db, golds = spider2_env
    localdb_dir = os.path.dirname(db)
    scorer = S._make_scorer(localdb_dir)
    extra = {"category": "sqlite", "db": "shop", "condition_cols": [],
             "ignore_order": True, "gold_paths": golds}
    traces = [
        {"response_text": "```sql\nSELECT region, SUM(revenue) AS total FROM sales GROUP BY region\n```",
         "extra_payload": extra},
        {"response_text": "```sql\nSELECT region FROM sales\n```", "extra_payload": extra},
        {"response_text": "no sql here", "extra_payload": extra},
    ]
    asyncio.run(scorer(traces))
    assert [t["is_correct"] for t in traces] == [True, False, False]


# --------------------------------------------------------------------------- #
# swe_bench_pro raw-sample table
# --------------------------------------------------------------------------- #
def _fake_pro_frame():
    import pandas as pd
    from gbench.runners.eval_suites.swe_bench_pro import _RAW_SAMPLE_COLUMNS
    return pd.DataFrame([{c: f"v_{c}" for c in _RAW_SAMPLE_COLUMNS}])


def test_missing_csv_is_generated_from_the_canonical_dataset():
    """The Pro harness README names swe_bench_pro_full.csv but the clone never ships it."""
    from gbench.runners.eval_suites import swe_bench_pro as SP
    with tempfile.TemporaryDirectory() as hd:
        ds = _fake_pro_frame().to_dict("records")   # the pinned loader reads rows
        with patch.dict(os.environ, {"SWE_BENCH_PRO_HARNESS_DIR": hd}, clear=False), \
             patch("datasets.load_dataset", return_value=ds):
            os.environ.pop("SWE_BENCH_PRO_RAW_SAMPLE", None)
            path = SP.raw_sample_path()
        # named after the pinned dataset revision, not the bare swe_bench_pro_full.csv
        assert path == os.path.join(hd, SP.RAW_SAMPLE_NAME)
        assert os.path.isfile(path)
        header = open(path, encoding="utf-8").readline()
        for col in SP._RAW_SAMPLE_COLUMNS:
            assert col in header, f"the harness reads {col} off this table"


def test_existing_csv_is_never_overwritten():
    # a bare swe_bench_pro_full.csv is not tied to the pinned revision (it may hold
    # pre-pin or V2 rows), so it is no longer USED either; the pinned table is written beside it.
    from gbench.runners.eval_suites import swe_bench_pro as SP
    with tempfile.TemporaryDirectory() as hd:
        csv = os.path.join(hd, "swe_bench_pro_full.csv")
        with open(csv, "w", encoding="utf-8") as f:
            f.write("instance_id\nkeep-me\n")
        with patch.dict(os.environ, {"SWE_BENCH_PRO_HARNESS_DIR": hd}, clear=False), \
             patch("datasets.load_dataset", return_value=_fake_pro_frame().to_dict("records")):
            os.environ.pop("SWE_BENCH_PRO_RAW_SAMPLE", None)
            assert SP.raw_sample_path() == os.path.join(hd, SP.RAW_SAMPLE_NAME)
        assert "keep-me" in open(csv, encoding="utf-8").read()


def test_unreadable_dataset_is_a_skip_not_a_guess():
    from gbench.runners.eval_suites import swe_bench_pro as SP
    with tempfile.TemporaryDirectory() as hd:
        open(os.path.join(hd, "swe_bench_pro_eval.py"), "w").close()
        os.mkdir(os.path.join(hd, "run_scripts"))
        with patch.dict(os.environ,
                        {"SWE_BENCH_PRO_HARNESS_DIR": hd, "SWE_BENCH_PRO_RUN": "1"},
                        clear=False), \
             patch("shutil.which", return_value="/usr/bin/docker"), \
             patch("subprocess.run", return_value=MagicMock(returncode=0)), \
             patch.dict("sys.modules", {"docker": MagicMock()}), \
             patch("datasets.load_dataset", side_effect=OSError("offline")):
            os.environ.pop("SWE_BENCH_PRO_RAW_SAMPLE", None)
            ok, reason = SP.check_swe_bench_pro_prerequisites()
        assert ok is False
        assert "raw-sample table" in reason and "SWE_BENCH_PRO_RAW_SAMPLE" in reason


def test_bigcodebench_workdir_is_traversable_by_the_container_uid():
    """`mkdtemp` is 0700/host-uid; the image runs as uid 1000 and cannot traverse it.

    That made every relative path resolution inside the container raise PermissionError -
    including `datasets.load_dataset`'s `Path("bigcode/bigcodebench/state.json").exists()`
    probe - so the harness produced no results and the suite scored 0. Reproduced against
    the real image: 0700 raises, 0777 resolves and is writable.
    """
    import re
    src = open(os.path.join(os.path.dirname(__file__), "..", "gbench", "runners",
                            "eval_suites", "bigcodebench.py")).read()
    body = src[src.index("def _make_scorer"):]
    mk = body.index("mkdtemp")
    chmod = body.find("os.chmod(workdir, 0o777)")
    assert chmod != -1, "the bind-mounted workdir must be made traversable by the image uid"
    assert chmod > mk, "chmod must follow mkdtemp"
    # the samples file the container reads must not inherit a 077 umask either
    assert re.search(r"os\.chmod\(samples_path,\s*0o6\d\d\)", body), \
        "samples.jsonl must be readable by the container uid"


# --------------------------------------------------------------------------- #
# scicode reference outputs are fetched at eval time, like any other dataset
# --------------------------------------------------------------------------- #
def test_scicode_prefers_an_explicit_local_copy(tmp_path):
    from gbench.runners.eval_suites import scicode
    f = tmp_path / "test_data.h5"
    f.write_bytes(b"x")
    # The legacy bare env name still resolves the path (back-compat), but provenance now reports the
    # canonical knob name (env knobs were unified onto GBENCH_<SUITE>_ with the bare name as an alias).
    with patch.dict(os.environ, {"SCICODE_TEST_DATA": str(f)}, clear=False), \
         patch("huggingface_hub.hf_hub_download", side_effect=AssertionError("must not fetch")):
        path, prov = scicode.resolve_test_data()
    assert path == str(f) and prov == "GBENCH_SCICODE_TEST_DATA"


def test_scicode_fetches_the_mirror_when_no_local_copy(tmp_path):
    from gbench.runners.eval_suites import scicode
    f = tmp_path / "test_data.h5"
    f.write_bytes(b"x")
    env = {k: v for k, v in os.environ.items() if k != "SCICODE_TEST_DATA"}
    with patch.dict(os.environ, env, clear=True), \
         patch("huggingface_hub.hf_hub_download", return_value=str(f)):
        path, prov = scicode.resolve_test_data()
    assert path == str(f)
    assert prov.startswith("hf:"), "the mirror must be recorded as the provenance"


def test_scicode_hard_errors_when_the_data_cannot_be_obtained():
    """Every test binds `target` from this file; without it a perfect solution scores 0, so
    the no-skip policy requires a hard-error (not a silent skip)."""
    from gbench.runners.eval_suites.scicode import run_scicode
    env = {k: v for k, v in os.environ.items() if k != "SCICODE_TEST_DATA"}
    with patch.dict(os.environ, env, clear=True), \
         patch("huggingface_hub.hf_hub_download", side_effect=OSError("offline")):
        with pytest.raises(RuntimeError, match="structural 0"):
            run_scicode("m", "http://x", concurrency=1, limit=2)


def test_scicode_records_the_digest_of_what_decided_the_scores():
    import inspect
    from gbench.runners.eval_suites import scicode
    src = inspect.getsource(scicode.run_scicode)
    assert 'result["test_data"]' in src and "sha256" in src


def test_bigcodebench_writes_the_field_the_harness_requires():
    """`bigcodebench/data/utils.py::load_solutions` asserts each row has "completion" or
    "solution". gbench wrote "raw_solution", which passed the permission fix and then died
    with "No completion or solution found in sample!" -> "harness produced no parsed
    results" -> 0/N. Verified against the real image: a "solution" row is accepted."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "gbench", "runners",
                            "eval_suites", "bigcodebench.py")).read()
    body = src[src.index("def _make_scorer"):]
    write = body[body.index("samples.jsonl"):body.index("mount =")]
    code = "\n".join(l for l in write.splitlines() if not l.lstrip().startswith("#"))
    assert '"solution"' in code or '"completion"' in code
    assert '"raw_solution"' not in code


# --------------------------------------------------------------------------- #
# a harness that did not evaluate must not publish a 0% score
# --------------------------------------------------------------------------- #
def test_swebench_zero_completed_is_an_error_not_a_zero_score():
    """2026-08-17: swe_bench_multilingual published 0/20 as an ordinary score while the
    harness reported completed_instances=0 (9 errored, 11 empty patches)."""
    import inspect
    from gbench.runners.eval_suites import swebench_common as sc
    src = inspect.getsource(sc.execute_swebench)
    assert "completed_instances" in src, "the guard must inspect completed_instances"
    assert "harness/infrastructure failure" in src
    assert "partial_evaluation" in src, "a partly-evaluated run must say so"


def test_swe_bench_pro_all_empty_patches_is_an_error():
    """A looping model emits no diff; that is not a 0% resolved rate."""
    import inspect
    from gbench.runners.eval_suites import swe_bench_pro as sp
    src = inspect.getsource(sp)
    assert "empty_patch_instances" in src
    assert "submitted_with_patch" in src
    assert "measures the generation, not the resolved rate" in src


def test_multi_swe_bench_missing_report_is_an_error_not_a_zero_score():
    """2026-08-17: the harness died in 1 s with `ValueError: Invalid repo_dir: None`
    and the suite still published '20 questions, 0 correct, 0.00%' into the results
    table, indistinguishable from a model that resolved nothing."""
    import inspect
    from gbench.runners.eval_suites import multi_swe_bench as m
    # the guard lives in _report_reasons, shared by both modes
    src = inspect.getsource(m.run_multi_swe_bench) + inspect.getsource(m._report_reasons)
    assert "multi_swe_bench_report" in src, "the guard must inspect the harness report"
    assert 'result["status"] = "error"' in src
    assert "not a 0%" in src


def test_multi_swe_bench_surfaces_why_the_harness_failed():
    """`harness report not produced` alone does not say WHAT went wrong; the
    repo_dir crash was only findable by reading the raw sweep log."""
    import inspect
    from gbench.runners.eval_suites import multi_swe_bench as m
    src = inspect.getsource(m._make_scorer)
    assert "stderr_tail" in src


def test_bigcodebench_restricts_the_evaluator_to_the_sampled_tasks():
    """The evaluator asserts full-split coverage - "Expected 1140 problems, got 20" - so
    every --eval-limit run failed and reported 0/N. `--selective_evaluate` takes the task
    ids actually generated."""
    src = open(os.path.join(os.path.dirname(__file__), "..", "gbench", "runners",
                            "eval_suites", "bigcodebench.py")).read()
    body = src[src.index("def _make_scorer"):]
    assert "--selective_evaluate" in body
    assert "task_ids" in body and "sample_traces" in body


# --------------------------------------------------------------------------- #
# patch extraction: the harness rejects anything with a stray fence
# --------------------------------------------------------------------------- #
DIFF = ("diff --git a/f.py b/f.py\n--- a/f.py\n+++ b/f.py\n"
        "@@ -1,2 +1,2 @@\n-old\n+new\n")


def test_patch_is_taken_from_the_fenced_block_that_is_a_diff():
    """The old extractor looked only at the FIRST block; when the model showed current code
    first it fell through and swept up the closing fence -> "malformed patch at line 20"."""
    from gbench.runners.eval_suites.swebench_common import extract_patch
    resp = ("Here is the current code:\n```python\ndef f(): pass\n```\n"
            "And the fix:\n```diff\n" + DIFF + "```\n")
    got = extract_patch(resp)
    assert "```" not in got and got.startswith("diff --git")


def test_trailing_prose_after_the_diff_is_dropped():
    """Measured 2026-08-17: every non-empty patch in three suites carried a fence."""
    from gbench.runners.eval_suites.swebench_common import extract_patch
    resp = "```diff\n" + DIFF + "```\n\nActually, I should also mention that...\n"
    got = extract_patch(resp)
    assert "```" not in got and "Actually" not in got
    assert got.strip().endswith("+new")


def test_unfenced_diff_still_works():
    from gbench.runners.eval_suites.swebench_common import extract_patch
    assert extract_patch("Some preamble\n" + DIFF).startswith("diff --git")


def test_no_diff_yields_nothing_rather_than_prose():
    from gbench.runners.eval_suites.swebench_common import extract_patch
    assert extract_patch("I could not solve this task.") == ""


def test_every_swe_suite_uses_the_shared_extractor():
    """Three suites carried their own copy of the broken version."""
    import re as _re
    base = os.path.join(os.path.dirname(__file__), "..", "gbench", "runners", "eval_suites")
    # (swe_lancer extracts no patch: SWE-Lancer's own grader takes the task container's diff)
    for name in ("swe_bench_pro", "multi_swe_bench"):
        src = open(os.path.join(base, f"{name}.py")).read()
        body = src[src.index("def _extract_patch"):]
        body = body[:body.index("\ndef ")]
        code = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith("#"))
        assert "from .swebench_common import extract_patch" in code, name
        assert '"""' in body and "re.search" not in code, f"{name} still has its own regex"


HEADERLESS = ("--- a/pylint/utils.py\n+++ b/pylint/utils.py\n"
              "@@ -115,7 +115,7 @@\n     parser.add_argument(\n-        old\n+        new\n")


def test_a_unified_diff_without_the_git_header_is_still_a_patch():
    """The model emits `--- a/f` / `+++ b/f` / `@@` inside a ```diff block without the
    `diff --git` line. It applies fine with `git apply -p1`, but requiring the header threw
    it away: 40 of the 47 "no extractable patch" samples across the four SWE suites on
    2026-08-17 were this, i.e. most of the apparent model failure was extraction."""
    from gbench.runners.eval_suites.swebench_common import extract_patch
    got = extract_patch("Here is the fix:\n```diff\n" + HEADERLESS + "```\n")
    assert got.startswith("--- a/pylint/utils.py")
    assert "@@" in got and "```" not in got


def test_headerless_diff_works_unfenced_too():
    from gbench.runners.eval_suites.swebench_common import extract_patch
    assert extract_patch("Preamble\n" + HEADERLESS).startswith("--- a/")


def test_the_git_header_form_still_wins_when_both_are_present():
    from gbench.runners.eval_suites.swebench_common import extract_patch
    both = ("```diff\n" + HEADERLESS + "```\n```diff\ndiff --git a/x b/x\n--- a/x\n+++ b/x\n"
            "@@ -1 +1 @@\n-a\n+b\n```")
    # pass 1 scans every block for the full git header before falling back to headerless,
    # so the fully-formed patch wins regardless of order
    assert extract_patch(both).startswith("diff --git a/x b/x")


def test_prose_in_a_fence_is_not_mistaken_for_a_patch():
    """multi_swe_bench wraps rambling in fences ("Wait, I should provide a cleaner...")."""
    from gbench.runners.eval_suites.swebench_common import extract_patch
    assert extract_patch("```\nWait, I should provide a cleaner implementation.\n```") == ""
    assert extract_patch("```python\ndef f():\n    return 1\n```") == ""


def test_a_header_without_hunks_is_not_a_patch():
    from gbench.runners.eval_suites.swebench_common import extract_patch
    assert extract_patch("```diff\n--- a/f.py\n+++ b/f.py\n```") == ""


def test_multi_swe_bench_supplies_the_required_repo_dir():
    """`repo_dir` is the only harness field with no argparse default, and
    `_check_repo_dir` also requires the directory to exist. Omitting it killed every run
    in ~1s with `ValueError: Invalid repo_dir: None`, so the suite never scored."""
    import inspect
    from gbench.runners.eval_suites import multi_swe_bench as m
    src = inspect.getsource(m._make_scorer)
    assert '"repo_dir": repo_dir' in src, "config must carry repo_dir"
    assert "makedirs(repo_dir" in src, "the harness requires it to exist"
    assert "GBENCH_MULTI_SWE_REPO_DIR" in src, "operators need to relocate it"


def test_multi_swe_bench_repo_dir_is_not_inside_the_temp_workdir():
    """workdir is a fresh mkdtemp per run; cloning Java/Rust/Go/C++/TS repos into it
    would re-clone everything on every run."""
    import inspect
    from gbench.runners.eval_suites import multi_swe_bench as m
    src = inspect.getsource(m._make_scorer)
    assign = src[src.index("repo_dir = "):src.index("os.makedirs(repo_dir")]
    assert "workdir" not in assign, "repo_dir must live outside the per-run temp dir"
    assert "expanduser" in assign or "XDG_CACHE_HOME" in assign


def test_multi_swe_bench_samples_across_all_languages_not_just_the_first():
    """A contiguous head again. The loader used to `break` once it had `limit` rows while walking
    sorted(jsonl_files); `c/...` sorts first, so --eval-limit 20 returned 20 C instances
    (all facebook/zstd) and never opened java/go/rust/kotlin/cpp/js/ts. Measured on the
    2026-08-17 run: 1 distinct category in 20 samples, on a *multi*-language benchmark."""
    from unittest.mock import patch
    from gbench.runners.eval_suites import multi_swe_bench as m

    langs = ["c", "cpp", "go", "java", "js", "kotlin", "rust", "ts"]
    files = [f"{l}/{l}_dataset.jsonl" for l in langs]

    def fake_download(repo_id, filename, repo_type, revision=None):   # loader pins a revision
        lang = filename.split("/")[0]
        import json as _j, tempfile as _t, os as _o
        p = _o.path.join(_t.mkdtemp(), "d.jsonl")
        with open(p, "w") as f:
            for i in range(10):
                f.write(_j.dumps({"org": "o", "repo": lang, "number": i,
                                  "title": "t", "body": "b",
                                  "base": {"sha": "x"}, "fix_patch": "p"}) + "\n")
        return p

    class FakeApi:
        def list_repo_files(self, repo, repo_type, revision=None):
            return files

    with patch("huggingface_hub.HfApi", FakeApi), \
         patch("huggingface_hub.hf_hub_download", fake_download):
        samples = m._load_multi_swe_bench_samples(limit=16)

    got = {(s[2] or {}).get("category") for s in samples}
    assert len(samples) == 16
    assert len(got) >= 6, f"sampled only {sorted(got)} - expected a spread across languages"


def test_terminal_bench_gives_a_slow_local_model_more_agent_time(monkeypatch):
    """Measured 2026-08-18: every task in a 3-task --thinking run ended in
    AgentTimeoutError after 10-25 minutes, scoring a structural 0%. The TB2.1 leaderboard
    requires timeout_multiplier unset or 1.0, so the default is 1.0 in both modes and a
    slow endpoint gets more agent time only through the opt-in GBENCH_TB_TIMEOUT_MULTIPLIER
    (named in leaderboard_comparable_reason). Only the AGENT allowance is ever scaled - the
    verifier timeout is a correctness bound and must not move."""
    import inspect
    from gbench.runners.eval_suites import terminal_bench as T
    src = inspect.getsource(T)
    assert "--agent-timeout-multiplier" in src
    assert "--verifier-timeout-multiplier" not in src, "the verifier bound must stay fixed"
    assert "GBENCH_TB_AGENT_TIMEOUT_MULTIPLIER" not in src
    monkeypatch.delenv("GBENCH_TB_TIMEOUT_MULTIPLIER", raising=False)
    assert T._timeout_multiplier(True) == T._timeout_multiplier(False) == "1.0"
    monkeypatch.setenv("GBENCH_TB_TIMEOUT_MULTIPLIER", "4")
    assert T._timeout_multiplier(True) == "4.0"


def test_terminal_bench_preserves_the_agent_transcript():
    """`AgentTimeoutError` alone cannot distinguish a hard task from a looping model, and
    Harbor's jobs-dir is a TemporaryDirectory deleted on exit. After the 2026-08-18 run
    there was no way to check whether the two timed-out trials were stuck in a loop."""
    import inspect
    from gbench.runners.eval_suites import terminal_bench as T
    src = inspect.getsource(T)
    assert "_agent_transcript" in src
    assert "repetition_run" in src and "repetition_onset" in src, (
        "a timed-out trial must carry the same looping signal every other suite reports")


def test_terminal_bench_transcript_extraction_is_shape_tolerant():
    """Harbor's result.json schema varies by agent and version; missing data must yield an
    empty string, never an exception that loses the whole trial."""
    from gbench.runners.eval_suites.terminal_bench import _agent_transcript as f
    assert f({}, {}) == ""
    assert f({"messages": [{"content": "ls"}, {"text": "ls"}]}, {}) == "ls\nls"
    assert f({}, {"transcript": "cd /tmp"}) == "cd /tmp"
    assert f({"agent_output": "x" * 99999}, {}).__len__() <= 40000   # capped
