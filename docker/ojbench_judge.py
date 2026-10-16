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

"""Entrypoint for the gbench-ojbench image: judge pre-generated OJBench submissions.

Reads /work/records.jsonl ({id, dataset, language, difficulty, content} per line, written by
gbench after the model rollout), runs the official online-judge over the DMOJ sandbox, and
writes /work/results.jsonl ({id, language, is_passed} per line). The testdata (NOI/ + ICPC/)
is bind-mounted read-only at OJBENCH_TESTDATA (default /testdata)."""

import json
import os
from pathlib import Path

import ojbench


def main() -> None:
    td = os.environ.get("OJBENCH_TESTDATA", "/testdata")
    workers = max(1, int(os.environ.get("OJBENCH_WORKERS", "4")))
    records_path = os.environ.get("OJBENCH_RECORDS", "/work/records.jsonl")
    results_path = os.environ.get("OJBENCH_RESULTS", "/work/results.jsonl")

    ojbench.init(problem_dirs=[Path(td) / "NOI", Path(td) / "ICPC"])

    with open(records_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    results = ojbench.judge_jsonl_data(records, num_workers=workers) if records else []

    with open(results_path, "w", encoding="utf-8") as f:
        for r in results or []:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
