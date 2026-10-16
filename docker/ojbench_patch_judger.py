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

"""Baked into gbench-ojbench at build time: fix the deadlock in ojbench.judger.judge_jsonl_data.

Upstream does `message = result_queue.get()` with NO timeout inside `while not
ptracker.is_complete()`. A judge worker can crash hard (C-level, no traceback) while executing a
submission, so its result is never queued, `is_complete()` never becomes true, and the main
process blocks on `get()` forever (observed: worker zombie + main stuck in queues.py get).

Patch: use a bounded get; on timeout, if EVERY worker has exited, no further results are possible
(a worker crashed mid-submission) -> stop instead of deadlocking. Unjudged entries keep their
default not-passed verdict. A slow-but-alive worker is NOT aborted (only all-dead stops the loop).
"""

import os
import sys

PATH = "/opt/OJBench/ojbench/judger.py"
OLD = "        message = result_queue.get()\n"
NEW = (
    "        try:\n"
    "            message = result_queue.get(timeout=_GBENCH_GET_TIMEOUT_S)\n"
    "        except _gbench_queue.Empty:\n"
    "            # gbench patch: a worker can crash hard mid-submission (no result queued), which\n"
    "            # made the original unconditional get() hang forever. If every worker has exited,\n"
    "            # no further results are possible -> stop instead of deadlocking; unjudged entries\n"
    "            # keep their default (not-passed) verdict. Slow-but-alive workers keep waiting.\n"
    "            if not any(w.is_alive() for w in workers):\n"
    "                logger.error('gbench: all judge workers exited with tasks still pending; a "
    "worker crashed mid-submission - stopping to avoid a hang, remaining entries left unjudged "
    "(scored not-passed).')\n"
    "                break\n"
    "            continue\n"
)
ANCHOR = "def judge_jsonl_data("
INJECT = (
    "import queue as _gbench_queue\n"
    "import os as _gbench_os\n"
    "_GBENCH_GET_TIMEOUT_S = int(_gbench_os.environ.get('OJBENCH_GET_TIMEOUT_S', '120'))\n\n\n"
)


def main() -> None:
    src = open(PATH, encoding="utf-8").read()
    if "_GBENCH_GET_TIMEOUT_S" in src:
        print("ojbench_patch_judger: already patched; skipping")
        return
    if OLD not in src:
        sys.exit("ojbench_patch_judger: FAILED - could not find the result_queue.get() line to "
                 "patch (upstream judger.py changed). Update docker/ojbench_patch_judger.py.")
    if ANCHOR not in src:
        sys.exit("ojbench_patch_judger: FAILED - could not find judge_jsonl_data definition.")
    src = src.replace(ANCHOR, INJECT + ANCHOR, 1).replace(OLD, NEW, 1)
    open(PATH, "w", encoding="utf-8").write(src)
    print("ojbench_patch_judger: patched judger.py (bounded get + dead-worker break)")


if __name__ == "__main__":
    main()
