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

"""tau retrieval-sandbox leak prune (added 2026-08-20).

tau2's SandboxManager stages a ~700-file knowledge base into /tmp/agentic_search_<id>/ per
simulation and cleans up only on graceful __exit__; killed/timed-out sims leak the dir. On a
RAM-backed tmpfs these accumulate across runs until the INODE table fills and unrelated evals
crash with a misleading "No space left on device" (measured: 1,141 leaked dirs, ~800K inodes,
killed aider_polyglot). `_prune_stale_search_sandboxes` clears leftovers older than a
threshold at the start of each tau run.
"""
import os
import sys
import time

_S = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gbench")
if _S not in sys.path:
    sys.path.insert(0, _S)

from gbench.runners.eval_suites import tau_common as T  # noqa: E402


def _mk(tmp_path, name, age_h):
    d = tmp_path / name
    (d / "knowledge_base").mkdir(parents=True)
    (d / "srt-settings.json").write_text("{}")
    t = time.time() - age_h * 3600
    os.utime(d, (t, t))
    return d


def test_prunes_old_leftovers_keeps_fresh(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(T, "_TAU_SANDBOX_MAX_AGE_H", 2.0)
    old = _mk(tmp_path, "agentic_search_OLD", age_h=3)
    fresh = _mk(tmp_path, "agentic_search_FRESH", age_h=0)
    unrelated = _mk(tmp_path, "something_else", age_h=99)
    T._prune_stale_search_sandboxes()
    assert not old.exists(), "stale leftover must be pruned"
    assert fresh.exists(), "an in-use (fresh) sandbox must be kept"
    assert unrelated.exists(), "only agentic_search_* is touched"


def test_disabled_when_age_zero(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(T, "_TAU_SANDBOX_MAX_AGE_H", 0.0)
    old = _mk(tmp_path, "agentic_search_OLD", age_h=99)
    T._prune_stale_search_sandboxes()
    assert old.exists(), "age=0 disables pruning entirely"


def test_never_raises_on_missing_tmp(monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "gettempdir", lambda: "/nonexistent/xyz")
    monkeypatch.setattr(T, "_TAU_SANDBOX_MAX_AGE_H", 2.0)
    T._prune_stale_search_sandboxes()  # must not raise


# --- per-task trace saving: ON BY DEFAULT into the run's results dir (2026-08-21) -----------
def test_traces_default_to_the_run_results_dir(monkeypatch):
    """tau is a wrapped harness; without saving, its per-task data is lost at run end (unlike
    every other eval). Unset TAU2_SAVE_TRACES must default to <run dir>/tau_traces."""
    monkeypatch.delenv("TAU2_SAVE_TRACES", raising=False)
    monkeypatch.setenv("GBENCH_RESULTS_DIR", "/tmp/run-xyz")
    assert T._resolve_tau_trace_dir() == "/tmp/run-xyz/tau_traces"


def test_explicit_trace_path_overrides(monkeypatch):
    monkeypatch.setenv("TAU2_SAVE_TRACES", "/custom/traces")
    monkeypatch.setenv("GBENCH_RESULTS_DIR", "/tmp/run-xyz")
    assert T._resolve_tau_trace_dir() == "/custom/traces"


def test_empty_trace_env_disables_saving(monkeypatch):
    """`TAU2_SAVE_TRACES=` (explicit empty) is the opt-out."""
    monkeypatch.setenv("TAU2_SAVE_TRACES", "")
    monkeypatch.setenv("GBENCH_RESULTS_DIR", "/tmp/run-xyz")
    assert T._resolve_tau_trace_dir() is None


def test_no_run_dir_no_traces(monkeypatch):
    """No run dir known (e.g. a bare unit context) -> don't dump traces somewhere surprising."""
    monkeypatch.delenv("TAU2_SAVE_TRACES", raising=False)
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    assert T._resolve_tau_trace_dir() is None


def test_log_manager_exports_the_results_dir(tmp_path, monkeypatch):
    """LogManager must export GBENCH_RESULTS_DIR so tau (and others) can co-locate artifacts."""
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    from gbench.utils.log_manager import LogManager
    lm = LogManager(base_dir=str(tmp_path / "results"))
    assert os.environ.get("GBENCH_RESULTS_DIR") == str(lm.results_dir)
