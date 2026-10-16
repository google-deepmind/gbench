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
"""gbench's entry point for SWE-Lancer's own run, inside the gbench-swe-lancer-harness image.

It runs openai/frontier-evals project/swelancer exactly as `swelancer/run_swelancer.py` does:
the same `main` (setup_logging, then nanoeval.run(EvalSpec(eval=swelancer, runner=runner))),
the same chz command line (`swelancer.*=...`, `runner.*=...`), the same solver, tasks, images and
grading. The one addition is a report file: nanoeval.run returns only the summary, so the eval
is SWE-Lancer's SWELancerEval with a `report_path` field, whose get_full_summary calls
SWELancerEval.get_full_summary unchanged and then writes the summary plus one row per task
(question id, variant, price, whether it was solved, the dollars earned, the system error if the
rollout failed) to that path for gbench to read.

The class lives in an importable module (never __main__), as nanoeval requires of an eval.
Run as:

    python -c "import swe_lancer_eval; swe_lancer_eval.entry()" swelancer.split=diamond ...
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chz
from nanoeval.eval import RunnerArgs
from nanoeval.library_config import LibraryConfig
from nanoeval.setup import nanoeval_entrypoint
from nanoeval.solvers.computer_tasks.steps import FinalResult

from swelancer.eval import SWELancerEval
from swelancer.run_swelancer import main as swelancer_main
from swelancer.utils.custom_logging import swelancer_library_config


def task_row(task: Any, result: Any) -> dict[str, Any]:
    """One task's outcome as gbench reads it."""
    row: dict[str, Any] = {
        "question_id": task.question_id, "variant": task.variant, "price": task.price,
        "attempt_id": task.attempt_id, "run_id": task.run_id, "run_dir": task.run_dir,
        "docker_image": getattr(task, "docker_image", None),
    }
    if isinstance(result, FinalResult):
        try:
            log = json.loads(result.grade.grader_log or "{}")
        except (TypeError, ValueError):
            log = {}
        row.update({
            "correct": bool(result.correct), "score": result.grade.score,
            "earned": log.get("earned", 0.0), "available": log.get("available", task.price),
            "input_tokens": log.get("input_tokens", 0),
            "output_tokens": log.get("output_tokens", 0),
            "reasoning_tokens": log.get("reasoning_tokens", 0),
            "pytest_exit": log.get("pytest_exit"), "system_error": None,
        })
    else:
        row.update({"correct": False, "score": None, "earned": 0.0,
                    "available": task.price,
                    "system_error": f"{type(result).__name__}: {result}"})
    return row


@chz.chz
class GbenchSWELancerEval(SWELancerEval):
    report_path: str = chz.field(default="", doc="Where gbench reads the per-task report.")

    async def get_full_summary(self, results: list[tuple[Any, Any]]) -> dict[str, Any]:
        summary = await super().get_full_summary(results)
        if self.report_path:
            report = {"summary": summary,
                      "tasks": [task_row(task, result) for task, result in results]}
            Path(self.report_path).write_text(json.dumps(report, default=str, indent=1),
                                              encoding="utf-8")
        return summary


async def main(
    swelancer: GbenchSWELancerEval,
    runner: RunnerArgs,
    library_config: LibraryConfig = swelancer_library_config,
) -> None:
    await swelancer_main(swelancer, runner, library_config)


def entry() -> None:
    nanoeval_entrypoint(chz.entrypoint(main))
