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
"""swebench error classification: a patch that would not apply is a MODEL failure (unresolved/0),
NOT an infrastructure failure. Only a true harness error (docker/build/timeout, or a missing log we
cannot inspect) is unscoreable. Regression for a run where all 3 SWE-bench-Live instances
failed to apply (bad model diffs) and the run was mislabeled `status: error` instead of 0% resolved."""

import os

from gbench.runners.eval_suites.swebench_common import _classify_swebench_errors


def _write_log(workdir, iid, body):
    d = os.path.join(workdir, "logs", "run_evaluation", "run1", "model1", iid)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "run_instance.log"), "w", encoding="utf-8") as f:
        f.write(body)


def test_patch_apply_failure_is_model_not_infra(tmp_path):
    wd = str(tmp_path)
    _write_log(wd, "a__patchfail", "Checked out repo\n>>>>> Patch Apply Failed:\ncan't find file to patch\n")
    _write_log(wd, "b__hunkfail", "patching file x.py\nHunk #2 FAILED at 7.\n1 out of 2 hunks FAILED\n")
    _write_log(wd, "c__docker", "docker: Error response from daemon: no space left on device\n")
    _write_log(wd, "d__timeout", "Container build timed out after 3600s\n")
    patch_fail, harness_err = _classify_swebench_errors(
        wd, ["a__patchfail", "b__hunkfail", "c__docker", "d__timeout"])
    assert patch_fail == {"a__patchfail", "b__hunkfail"}
    assert harness_err == {"c__docker", "d__timeout"}


def test_missing_log_is_treated_as_harness_error(tmp_path):
    # Conservative: if we cannot read the instance log, we cannot confirm a model failure -> infra.
    wd = str(tmp_path)
    _write_log(wd, "has_patch_fail", "malformed patch at line 7\n")
    patch_fail, harness_err = _classify_swebench_errors(wd, ["has_patch_fail", "no_log_at_all"])
    assert patch_fail == {"has_patch_fail"}
    assert harness_err == {"no_log_at_all"}


def test_empty_error_list(tmp_path):
    assert _classify_swebench_errors(str(tmp_path), []) == (set(), set())
