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

"""scicode's sub-step test script is built as upstream's test_generated_code.py builds it (the
assembled code first, then the vendored h5 loader's import, the targets and the test cases), run
from a script file with `python <file>` (so a program over the kernel's 128 KiB argument limit
still runs) in a directory bound into the sandbox, a step that runs past its time is a failed
step, and a script that was never started (the sandbox did not start, or the script, the
interpreter or test_data.h5 is not there inside it) is no verdict (a harness error), never the
model's failed step.

The first tests stub the sandbox. The real-jail tests run the step through bubblewrap
(GBENCH_SANDBOX=required) wherever it works here, and skip where it does not; the structural test
checks the jail's command line binds the script's directory in either case."""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from gbench.runners.eval_suites import sandbox
from gbench.runners.eval_suites import scicode as S

_STARTED = S.STEP_STARTED + "\n"


class _Done:
    def __init__(self, rc=0, out=_STARTED, err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def _capture(monkeypatch, result=None, raises=None):
    seen = {}

    def run_sandboxed(argv, **kw):
        seen["argv"] = list(argv)
        seen["readable"] = list(kw.get("readable") or [])
        seen["writable"] = list(kw.get("writable") or [])
        with open(argv[5], encoding="utf-8") as f:
            seen["script"] = f.read()
        if raises is not None:
            raise raises
        return result or _Done()
    monkeypatch.setattr(S, "run_sandboxed", run_sandboxed)
    return seen


def test_the_script_puts_the_models_code_before_the_loader_import(monkeypatch):
    seen = _capture(monkeypatch)
    out = S._score_step("import numpy as np\ndef f():\n    return 1", "7.1",
                        ["assert f() == target"], "/data/test_data.h5", 30)
    assert out["correct"] is True and out["returncode"] == 0 and "harness_error" not in out
    lines = seen["script"].splitlines()
    assert lines[0] == "import numpy as np"
    loader = lines.index(S.H5_LOADER_IMPORT)
    assert loader > lines.index("    return 1")
    assert lines[loader + 1] == "targets = process_hdf5_to_tuple('7.1', 1, '/data/test_data.h5')"
    assert lines[loader + 2:] == ["target = targets[0]", "assert f() == target"]
    # the launcher execs `python <file>`, never `python -c`; the h5 is checked before it
    argv = seen["argv"]
    assert argv[:3] == ["sh", "-c", S._STEP_LAUNCH]
    assert argv[4] == sys.executable and argv[5].endswith("7.1.py")
    assert argv[6:] == ["/data/test_data.h5"]
    assert 'exec "$py" "$script"' in S._STEP_LAUNCH
    # the script's own directory is bound into the jail; the vendored package and h5 readable
    assert seen["writable"] == [os.path.dirname(argv[5])]
    assert S.SCICODE_VENDOR_DIR in seen["readable"] and "/data/test_data.h5" in seen["readable"]
    assert not os.path.exists(os.path.dirname(argv[5]))           # removed afterwards


def test_a_program_over_the_argument_limit_still_runs(monkeypatch):
    seen = _capture(monkeypatch)
    big = "x = 1\n" * 40000                                   # about 240 KB
    assert S._score_step(big, "1.1", ["assert True"], "/h5", 30)["correct"] is True
    assert len(seen["script"]) > 128 * 1024


def test_a_timeout_is_a_failed_step_and_a_script_that_cannot_run_is_no_verdict(monkeypatch):
    _capture(monkeypatch, raises=subprocess.TimeoutExpired(["sh"], 30, output=_STARTED.encode()))
    out = S._score_step("pass", "1.1", [], "/h5", 30)
    assert out["correct"] is False and out["timed_out"] is True and "harness_error" not in out
    _capture(monkeypatch, raises=OSError(7, "Argument list too long"))
    out = S._score_step("pass", "1.1", [], "/h5", 30)
    assert out["correct"] is None
    assert out["harness_error"]["kind"] == "scoring_failed"
    _capture(monkeypatch, result=_Done(1, _STARTED, "AssertionError"))
    out = S._score_step("pass", "1.1", ["assert False"], "/h5", 30)
    assert out["correct"] is False and "AssertionError" in out["stderr_tail"]
    assert "harness_error" not in out


def test_a_run_that_never_started_the_script_is_a_harness_error(monkeypatch):
    # bwrap failed to set the jail up: no start line, exit 1, like a failing test would
    _capture(monkeypatch, result=_Done(1, "", "bwrap: Creating new namespace failed: "
                                             "No space left on device\n"))
    out = S._score_step("pass", "1.1", ["assert True"], "/h5", 30)
    assert out["correct"] is None and out["returncode"] == 1
    assert out["harness_error"]["kind"] == "scoring_failed"
    assert "bwrap: Creating new namespace failed" in out["harness_error"]["detail"]
    # a file missing inside the jail: the launcher stops before the step's code
    _capture(monkeypatch, result=_Done(125, "", "not found where the step runs: /h5\n"))
    out = S._score_step("pass", "1.1", [], "/h5", 30)
    assert out["correct"] is None and "not found where the step runs: /h5" in \
        out["harness_error"]["detail"]
    # the jail did not start the script before the time ran out
    _capture(monkeypatch, raises=subprocess.TimeoutExpired(["sh"], 30, output=None))
    out = S._score_step("pass", "1.1", [], "/h5", 30)
    assert out["correct"] is None and "did not start" in out["harness_error"]["detail"]


def test_the_start_line_is_not_part_of_the_steps_output_and_the_model_cannot_fake_a_harness_error(
        monkeypatch):
    # a step that fails with the launcher's own words and exit status is still a failed step:
    # the start line came first, so the step's code ran
    _capture(monkeypatch, result=_Done(125, _STARTED + "partial\n",
                                       "not found where the step runs: /h5\n"))
    out = S._score_step("pass", "1.1", [], "/h5", 30)
    assert out["correct"] is False and out["returncode"] == 125 and "harness_error" not in out
    assert out["stdout_tail"] == "partial\n"


def test_the_jail_binds_the_script_directory_over_its_empty_tmp(monkeypatch):
    """The command line bubblewrap gets, whether or not bubblewrap works here: the script's
    directory is bound after the jail mounts its empty /tmp, so the script is there inside."""
    seen = {}

    def run_sandboxed(argv, writable=(), readable=(), network=False, **kw):
        seen["cmd"] = sandbox.wrap_argv(argv, writable=writable, readable=readable, force=True)
        seen["script"] = argv[5]
        return _Done()
    monkeypatch.setattr(S, "run_sandboxed", run_sandboxed)
    monkeypatch.setattr(tempfile, "tempdir", "/tmp")
    assert S._score_step("pass", "7.1", [], "/h5", 30)["correct"] is True
    cmd, script_dir = seen["cmd"], os.path.dirname(seen["script"])
    assert script_dir.startswith("/tmp/")
    tmpfs = next(i for i in range(len(cmd) - 1) if cmd[i:i + 2] == ["--tmpfs", "/tmp"])
    bind = next(i for i in range(len(cmd) - 2)
                if cmd[i:i + 3] == ["--bind", script_dir, script_dir])
    assert tmpfs < bind < cmd.index("sh")


def test_run_scicode_refuses_to_start_when_the_sandbox_cannot_run_a_step(monkeypatch):
    monkeypatch.setattr(S, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(S, "resolve_test_data", lambda: ("/tmp/x/test_data.h5", "env"))
    monkeypatch.setattr(S, "check_test_data_digest", lambda h5, prov: ("d" * 64, None))
    monkeypatch.setattr(S, "check_cmp_importable", lambda: None)
    monkeypatch.setattr(S, "check_step_sandbox", lambda h5: f"cannot open {h5} in the jail")

    def _no_generation(*a, **k):
        raise AssertionError("scicode loaded or generated before the sandbox was checked")
    monkeypatch.setattr(S, "_load_scicode_problems", _no_generation)
    with pytest.raises(RuntimeError) as err:
        S.run_scicode("m", "http://127.0.0.1:9/v1", concurrency=1)
    msg = str(err.value)
    assert "[INFRA REQUIRED] scicode" in msg and "cannot open /tmp/x/test_data.h5" in msg
    assert "GBENCH_SCICODE_TEST_DATA" in msg


def test_the_step_check_runs_a_trivial_step_only_where_there_is_a_jail(monkeypatch):
    monkeypatch.setenv("GBENCH_SANDBOX", "none")
    monkeypatch.setattr(S, "_score_step", lambda *a, **k: pytest.fail("ran without a jail"))
    assert S.check_step_sandbox("/nowhere.h5") is None
    monkeypatch.setenv("GBENCH_SANDBOX", "required")
    calls = []
    monkeypatch.setattr(S, "_score_step", lambda *a, **k: calls.append(a) or {
        "correct": None, "harness_error": {"kind": "scoring_failed", "detail": "no jail"}})
    assert "no jail" in S.check_step_sandbox("/data/t.h5")
    assert calls == [("pass", "sandbox_check", [], "/data/t.h5", 120)]
    monkeypatch.setattr(S, "_score_step", lambda *a, **k: {"correct": True})
    assert S.check_step_sandbox("/data/t.h5") is None


def test_the_vendored_loader_reads_targets_as_upstream(tmp_path):
    import h5py
    sys.path.insert(0, S.SCICODE_VENDOR_DIR)
    try:
        from scicode.parse.parse import process_hdf5_to_tuple
    finally:
        sys.path.remove(S.SCICODE_VENDOR_DIR)
    h5 = tmp_path / "test_data.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("7.1/test1/var1", data=np.array([1.0, 2.0]))
        f.create_dataset("7.1/test2/var1", data=3)
    targets = process_hdf5_to_tuple("7.1", 2, str(h5))
    assert np.array_equal(targets[0], np.array([1.0, 2.0])) and targets[1] == 3


# --------------------------------------------------------------------------- #
# Through the real bubblewrap jail                                             #
# --------------------------------------------------------------------------- #
def _clear_sandbox_caches():
    sandbox.sandbox_available.cache_clear()
    sandbox._probe_bwrap.cache_clear()


@pytest.fixture
def real_jail(monkeypatch, request):
    """GBENCH_SANDBOX=required (tests/conftest.py sets none) with fresh probes, the step scripts
    under /tmp (the jail's emptied directory); skips where bubblewrap does not work."""
    monkeypatch.setenv("GBENCH_SANDBOX", "required")
    monkeypatch.setattr(tempfile, "tempdir", "/tmp")
    _clear_sandbox_caches()
    request.addfinalizer(_clear_sandbox_caches)
    if not sandbox.sandbox_available():
        pytest.skip("bubblewrap is unavailable or blocked here")


def _write_h5(directory):
    import h5py
    path = os.path.join(directory, "test_data.h5")
    with h5py.File(path, "w") as f:
        f["7.1/test1/var1"] = np.array([1.0, 2.0])
    return path


@pytest.fixture
def visible_h5(request):
    """A reference file the jail shows: /var/tmp is neither emptied nor under the home."""
    try:
        d = tempfile.mkdtemp(prefix="gbench_scicode_test_", dir="/var/tmp")
    except OSError:
        pytest.skip("/var/tmp is not writable here")
    request.addfinalizer(lambda: shutil.rmtree(d, ignore_errors=True))
    return _write_h5(d)


@pytest.fixture
def hidden_h5(request):
    """A reference file under /tmp, which the jail replaces with an empty directory."""
    d = tempfile.mkdtemp(prefix="gbench_scicode_test_", dir="/tmp")
    request.addfinalizer(lambda: shutil.rmtree(d, ignore_errors=True))
    return _write_h5(d)


_TESTS = ["assert np.allclose(f(), target)"]
_RIGHT = "import numpy as np\ndef f():\n    return np.array([1.0, 2.0])\n"
_WRONG = "import numpy as np\ndef f():\n    return np.array([1.0, 3.0])\n"


def test_in_the_jail_a_passing_step_passes_and_a_failing_step_fails(real_jail, visible_h5):
    good = S._score_step(_RIGHT, "7.1", _TESTS, visible_h5, 120)
    assert good["correct"] is True and "harness_error" not in good, good
    bad = S._score_step(_WRONG, "7.1", _TESTS, visible_h5, 120)
    assert bad["correct"] is False and bad["returncode"] == 1 and "harness_error" not in bad
    assert "AssertionError" in bad["stderr_tail"]
    # a step that exits with the launcher's status and words is still the model's failed step
    fake = S._score_step("import sys\nsys.stderr.write('not found where the step runs: x\\n')\n"
                         "sys.exit(125)", "7.1", [], visible_h5, 120)
    assert fake["correct"] is False and fake["returncode"] == 125 and "harness_error" not in fake


def test_in_the_jail_a_step_past_its_time_is_a_failed_step(real_jail, visible_h5):
    out = S._score_step("import time\ntime.sleep(60)", "7.1", [], visible_h5, 3)
    assert out["correct"] is False and out["timed_out"] is True and "harness_error" not in out


def test_in_the_jail_setup_failures_are_harness_errors(real_jail, visible_h5, hidden_h5,
                                                       monkeypatch):
    # the reference file is not there inside the jail
    out = S._score_step(_RIGHT, "7.1", _TESTS, hidden_h5, 120)
    assert out["correct"] is None and hidden_h5 in out["harness_error"]["detail"]
    assert S.check_step_sandbox(visible_h5) is None
    assert hidden_h5 in S.check_step_sandbox(hidden_h5)
    # the script's directory not bound (the defect this replaced): no verdict, not a failed step
    real_run = S.run_sandboxed

    def unbound(argv, **kw):
        kw.pop("writable", None)
        return real_run(argv, **kw)
    monkeypatch.setattr(S, "run_sandboxed", unbound)
    out = S._score_step(_RIGHT, "7.1", _TESTS, visible_h5, 120)
    assert out["correct"] is None and "7.1.py" in out["harness_error"]["detail"]
    monkeypatch.setattr(S, "run_sandboxed", real_run)
    # bubblewrap itself fails to build the jail
    real_wrap = sandbox.wrap_argv

    def broken(argv, **kw):
        cmd = real_wrap(argv, **kw)
        return cmd[:1] + ["--ro-bind", "/gbench_no_such_source", "/mnt"] + cmd[1:]
    monkeypatch.setattr(sandbox, "wrap_argv", broken)
    out = S._score_step(_RIGHT, "7.1", _TESTS, visible_h5, 120)
    assert out["correct"] is None and "bwrap" in out["harness_error"]["detail"]


def test_in_the_jail_the_vendored_cmp_imports(real_jail):
    assert S.check_cmp_importable() is None
