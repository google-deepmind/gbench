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

"""gbench predictions-scorer adapter for SWELancer (openai/SWELancer-Benchmark).

This module is no longer used: the swe_lancer suite now runs SWE-Lancer's own SimpleAgentSolver
and harness (gbench/docker/swe_lancer_harness.Dockerfile, swe_lancer_eval.py) and never imports
it. Kept until the maintainers remove it.

Upstream ships only `SimpleAgentSolver(model="gpt-4o")` (run_swelancer.py) - an agent that GENERATES
and scores in one loop, so there is no way to score patches produced by another model. This module
provides a `PredictionsSolver` that APPLIES a pre-generated unified-diff patch (from gbench's model)
into the task's container and then triggers SWELancer's OWN grading (the hidden Playwright pytest),
plus a `SWELancerPredictionsEval` that dumps per-issue results. gbench copies this file (and
run_swelancer_eval.py) into the SWELancer harness dir at run time and invokes it there.

Imports mirror swelancer_agent.py exactly (the nanoeval/alcatraz packages are vendored under the
harness's project/ and importable when run from the harness dir / uv env). It MUST be an importable
module, never __main__: nanoeval dill-serializes the whole EvalSpec (solver + eval) and asserts the
eval's module != "__main__".
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import chz
from typing_extensions import override

from nanoeval.solvers.computer_tasks.solver import PythonCodingSolver
from nanoeval.solvers.computer_tasks.steps import FinalResult, FinalResultSuccessful, Step
from nanoeval.solvers.computer_tasks.task import ComputerTask
from nanoeval_alcatraz.task_to_alcatraz_config import task_to_alcatraz_config
from nanoeval_alcatraz.alcatraz_computer_interface import AlcatrazComputerInterface
from alcatraz.clusters.local import LocalConfig

# RetryableSystemError signals an INFRA failure (nanoeval retries / records it as a system error)
# rather than a model wrong-answer - used when a patch cannot be applied. Import path varies by
# nanoeval version; fall back gracefully.
try:
    from nanoeval.solvers.computer_tasks.solver import RetryableSystemError
except Exception:  # pragma: no cover
    try:
        from nanoeval.eval import RetryableSystemError  # type: ignore
    except Exception:
        class RetryableSystemError(Exception):  # type: ignore
            pass

from swelancer import SWELancerEval  # upstream eval (harness dir is on sys.path at run time)


@chz.chz
class PredictionsSolver(PythonCodingSolver):
    """Apply gbench's unified-diff patch for the task, then let SWELancer grade the working tree.

    `predictions_path` -> a JSON file mapping {question_id: unified_diff_str}. (A path, not an inline
    dict, keeps the dill-pickled EvalSpec small.)
    """

    name: str = "PredictionsSolver"
    predictions_path: str = ""

    def shortname(self) -> str:
        return "predictions"

    @asynccontextmanager
    async def _start_computer(
        self, task: ComputerTask
    ) -> AsyncGenerator[AlcatrazComputerInterface, None]:
        # Same as SimpleAgentSolver._start_computer: build the alcatraz cluster (a local
        # `swelancer:latest` container) and expose it as a ComputerInterface.
        alcatraz_env = task_to_alcatraz_config(task, LocalConfig(pull_from_registry=False))
        async with alcatraz_env.build() as cluster:
            yield AlcatrazComputerInterface(cluster_value=cluster)

    @override
    async def run(self, task: ComputerTask) -> AsyncGenerator[Step | FinalResult, None]:
        preds = json.loads(Path(self.predictions_path).read_text(encoding="utf-8"))
        patch = preds.get(task.question_id)
        async with self._start_computer(task) as computer:
            # setup() creates the temp commit at the buggy baseline (so `git diff HEAD` == our patch)
            # and unpacks the hidden tests; mirrors upstream.
            await task.setup(computer)
            if patch:
                # Apply gbench's patch into the working tree (grade() runs tests against it; it does
                # NOT re-apply anything - it only records `git diff HEAD` into the log).
                await computer.upload(patch.encode("utf-8"), "/tmp/gbench.patch")
                res = await computer.send_shell_command(
                    "cd /app/expensify && git apply --whitespace=nowarn /tmp/gbench.patch")
                if res.exit_code != 0:
                    res = await computer.send_shell_command(
                        "cd /app/expensify && (git apply --3way /tmp/gbench.patch || "
                        "patch -p1 --fuzz=3 < /tmp/gbench.patch)")
                    if res.exit_code != 0:
                        # A patch that will not apply is an infra/harness condition (often a base
                        # mismatch), NOT a model 0 - raise so it is recorded as a system error.
                        raise RetryableSystemError(
                            "gbench patch failed to apply for %s: %s" % (
                                task.question_id,
                                res.output.decode("utf-8", "replace")[:500]))
            grade = await task.grade(computer)
            yield FinalResultSuccessful(grade=grade)


@chz.chz
class SWELancerPredictionsEval(SWELancerEval):
    """SWELancerEval that also writes per-issue results (resolved/score/price) to `output_dir`.

    `nanoeval.run` returns only a summary dict (and SWELancerEval.get_summary's earnings breakdown
    is orphaned under PythonCodingEval.get_full_summary), so we override get_full_summary - which IS
    called with the full (task, result) list in the driver process - to emit per_issue.json, then
    delegate to super().
    """

    output_dir: str = "swelancer_results"

    @override
    async def get_full_summary(self, results: list) -> dict:
        rows = []
        for task, res in results:
            qid = getattr(task, "question_id", None)
            price = getattr(task, "price", None)
            if isinstance(res, BaseException):
                rows.append({"question_id": qid, "resolved": None, "price": price,
                             "system_error": str(res)})
                continue
            grade = getattr(res, "grade", None)
            score = getattr(grade, "score", None)
            try:
                log = json.loads(getattr(grade, "grader_log", "") or "{}")
            except Exception:
                log = {}
            rows.append({
                "question_id": qid,
                "resolved": (bool(score) if score is not None else None),
                "score": score,
                "price": price,
                "earned": log.get("earned"),
                "available": log.get("available"),
                "variant": log.get("variant"),
            })
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "per_issue.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return await super().get_full_summary(results)
