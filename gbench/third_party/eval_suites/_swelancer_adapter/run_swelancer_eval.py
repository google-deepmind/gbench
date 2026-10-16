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

"""gbench SWELancer predictions-scorer entrypoint (the adapter upstream lacks).

This entrypoint is no longer used: the swe_lancer suite now runs SWE-Lancer's own
SimpleAgentSolver and harness (gbench/docker/swe_lancer_harness.Dockerfile, swe_lancer_eval.py)
and never runs it. Kept until the maintainers remove it.

Mirrors openai/SWELancer-Benchmark's run_swelancer.py exactly, but scores a PREDICTIONS FILE with
gbench's PredictionsSolver + SWELancerPredictionsEval instead of running the hardcoded gpt-4o agent.
gbench copies this + predictions_solver.py into the SWELancer harness dir and runs it there (uv env,
Docker socket available) as:

    python run_swelancer_eval.py --predictions preds.json --output_dir out --num_workers 4

`preds.json` = {question_id: unified_diff}. Per-issue results are written to <output_dir>/per_issue.json
and a summary to <output_dir>/summary.json.
"""
from __future__ import annotations

# Load environment before importing anything else (mirrors run_swelancer.py).
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import argparse
import json
from pathlib import Path

import nanoeval
from nanoeval.evaluation import EvalSpec, RunnerArgs
from nanoeval.recorder import dummy_recorder
from nanoeval.setup import nanoeval_entrypoint

from predictions_solver import PredictionsSolver, SWELancerPredictionsEval


def parse_args():
    p = argparse.ArgumentParser(description="Score gbench SWELancer predictions")
    p.add_argument("--predictions", required=True, help="JSON {question_id: unified_diff}")
    p.add_argument("--output_dir", default="swelancer_results")
    p.add_argument("--issue_ids", nargs="*", type=str, default=None,
                   help="restrict to these question_ids (default: all in --predictions)")
    p.add_argument("--num_workers", type=int, default=4)
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    taskset = args.issue_ids
    if not taskset:
        preds = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
        taskset = list(preds.keys())

    report = await nanoeval.run(
        EvalSpec(
            eval=SWELancerPredictionsEval(
                solver=PredictionsSolver(predictions_path=args.predictions),
                taskset=taskset,
                output_dir=args.output_dir,
            ),
            runner=RunnerArgs(
                concurrency=max(1, args.num_workers),
                experimental_use_multiprocessing=True,
                enable_slackbot=False,
                recorder=dummy_recorder(),
                max_retries=5,
            ),
        )
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(report, default=str, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    nanoeval_entrypoint(main())
