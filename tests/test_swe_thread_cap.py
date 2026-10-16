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

"""SWE eval containers must not oversubscribe the host's cores.

Measured 2026-08-17 on `scikit-learn__scikit-learn-14710`: same image, same
unpatched source, same 4 CPUs, only the thread cap differing - `OMP_NUM_THREADS=4`
ran 78/78 tests in 6.28 s, uncapped ran 34/78 before being killed at 541.53 s.
In the real sweep that instance burned the harness's full 1800 s timeout and was
scored an error, which was 51% of the entire four-suite run.

Nothing here starts a container; the docker SDK is stubbed.
"""

import inspect
import os
import re
from unittest import mock

import pytest

from gbench.runners.eval_suites import swe_thread_cap as cap


# --------------------------------------------------------------------------- #
# how many threads
# --------------------------------------------------------------------------- #
def test_default_shares_the_box_between_concurrent_containers():
    """96 cores / 20 workers -> 4 each, which is the measured-good value."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(cap.ENV_VAR, None)
        with mock.patch.object(os, "cpu_count", return_value=96):
            assert cap.resolve_threads(20) == 4


def test_a_single_worker_is_still_capped():
    """The stalled instance had ~91 cores to ITSELF and still did not finish, so
    the cap cannot scale with the core count."""
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(cap.ENV_VAR, None)
        with mock.patch.object(os, "cpu_count", return_value=96):
            assert cap.resolve_threads(1) == cap.MAX_THREADS_PER_CONTAINER


def test_more_workers_than_cores_still_gets_one_thread():
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(cap.ENV_VAR, None)
        with mock.patch.object(os, "cpu_count", return_value=8):
            assert cap.resolve_threads(64) == 1


def test_suite_containers_share_the_box_over_the_whole_suite_in_its_context_only():
    """A part of a suite run at once with other parts (a swe_bench_live tab lane) passes its own
    share; with SUITE_CONTAINERS set to the suite's containers the cap is the whole suite's,
    and only inside that context. The env override still wins."""
    import contextvars
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop(cap.ENV_VAR, None)
        with mock.patch.object(os, "cpu_count", return_value=208):
            ctx = contextvars.copy_context()
            ctx.run(cap.SUITE_CONTAINERS.set, 100)
            assert ctx.run(cap.resolve_threads, 25) == 2      # 208 // 100, not 208 // 25
            assert cap.resolve_threads(25) == 4               # outside the context: unchanged
            ctx.run(cap.SUITE_CONTAINERS.set, 4)
            assert ctx.run(cap.resolve_threads, 25) == 4      # never fewer than its own share
    with mock.patch.dict(os.environ, {cap.ENV_VAR: "3"}):
        assert ctx.run(cap.resolve_threads, 25) == 3


@pytest.mark.parametrize("raw,expected", [("1", 1), ("12", 12), ("0", 0),
                                          ("off", 0), ("none", 0), ("false", 0)])
def test_env_override(raw, expected):
    with mock.patch.dict(os.environ, {cap.ENV_VAR: raw}):
        assert cap.resolve_threads(20) == expected


def test_garbage_env_falls_back_to_the_computed_default_not_to_uncapped():
    with mock.patch.dict(os.environ, {cap.ENV_VAR: "banana"}), \
         mock.patch.object(os, "cpu_count", return_value=96):
        assert cap.resolve_threads(20) == 4


# --------------------------------------------------------------------------- #
# the docker patch
# --------------------------------------------------------------------------- #
class _FakeCollection:
    """Stands in for docker.models.containers.ContainerCollection."""
    last_kwargs: dict = {}

    def create(self, image=None, command=None, **kwargs):
        type(self).last_kwargs = dict(kwargs, image=image, command=command)
        return "container"


@pytest.fixture
def collection():
    return type("C", (_FakeCollection,), {})


def test_install_injects_every_thread_variable(collection):
    assert cap.install(4, _collection=collection) is True
    collection().create(image="img", command="tail -f /dev/null")
    env = collection.last_kwargs["environment"]
    assert env == {v: "4" for v in cap.THREAD_VARS}
    # OMP is the one that actually mattered in the measurement
    assert env["OMP_NUM_THREADS"] == "4"


def test_install_preserves_the_arguments_swebench_passes(collection):
    """swebench passes name/user/detach/platform/cap_add; losing any of them
    would break container creation outright."""
    cap.install(2, _collection=collection)
    collection().create(image="img", command="tail -f /dev/null", name="n",
                        user="root", detach=True, platform="linux/x86_64", cap_add=["SYS_PTRACE"])
    kw = collection.last_kwargs
    assert kw["name"] == "n" and kw["user"] == "root" and kw["detach"] is True
    assert kw["platform"] == "linux/x86_64" and kw["cap_add"] == ["SYS_PTRACE"]
    assert kw["command"] == "tail -f /dev/null" and kw["image"] == "img"


def test_install_is_idempotent_and_does_not_stack_wrappers(collection):
    cap.install(4, _collection=collection)
    first = collection.create
    cap.install(4, _collection=collection)
    assert collection.create is first


def test_reinstalling_with_a_new_count_does_not_wrap_the_wrapper(collection):
    cap.install(4, _collection=collection)
    cap.install(1, _collection=collection)
    collection().create(image="img")
    assert collection.last_kwargs["environment"]["OMP_NUM_THREADS"] == "1"


def test_zero_threads_does_not_patch_anything(collection):
    before = collection.create
    assert cap.install(0, _collection=collection) is False
    assert collection.create is before


def test_a_caller_supplied_thread_count_wins(collection):
    """If something deliberately set OMP_NUM_THREADS, we must not overwrite it."""
    cap.install(4, _collection=collection)
    collection().create(image="img", environment={"OMP_NUM_THREADS": "16", "FOO": "bar"})
    env = collection.last_kwargs["environment"]
    assert env["OMP_NUM_THREADS"] == "16"     # caller's value
    assert env["MKL_NUM_THREADS"] == "4"      # ours, for the ones they left alone
    assert env["FOO"] == "bar"                # unrelated vars survive


def test_list_form_environment_is_supported(collection):
    """The docker SDK accepts NAME=value lists as well as dicts."""
    cap.install(4, _collection=collection)
    collection().create(image="img", environment=["FOO=bar", "OMP_NUM_THREADS=16"])
    env = collection.last_kwargs["environment"]
    assert "FOO=bar" in env and "OMP_NUM_THREADS=16" in env
    assert "OMP_NUM_THREADS=4" not in env     # not duplicated/overridden
    assert "MKL_NUM_THREADS=4" in env


# --------------------------------------------------------------------------- #
# the subprocess bootstrap
# --------------------------------------------------------------------------- #
_MODULE_CMD = ["/venv/bin/python", "-m", "swebench.harness.run_evaluation",
               "--dataset_name", "d", "--instance_ids", "a", "b"]
_SCRIPT_CMD = ["python", "/harness/swe_bench_pro_eval.py", "--patch_path=/p", "--num_workers=4"]


@pytest.mark.parametrize("cmd", [_MODULE_CMD, _SCRIPT_CMD])
def test_wrapping_preserves_the_interpreter_and_every_argument(cmd):
    out = cap.wrap_command(cmd, 4)
    assert out[0] == cmd[0]
    assert out[1] == "-c"
    # every original argument still present, in order, after the bootstrap
    tail = out[3:]
    original_args = cmd[3:] if cmd[1] == "-m" else cmd[2:]
    assert tail == original_args


@pytest.mark.parametrize("cmd", [_MODULE_CMD, _SCRIPT_CMD])
def test_the_bootstrap_is_valid_python(cmd):
    compile(cap.wrap_command(cmd, 4)[2], "<bootstrap>", "exec")


def test_the_bootstrap_names_the_right_target():
    mod = cap.wrap_command(_MODULE_CMD, 4)[2]
    assert "swebench.harness.run_evaluation" in mod and "run_module" in mod
    scr = cap.wrap_command(_SCRIPT_CMD, 4)[2]
    assert "/harness/swe_bench_pro_eval.py" in scr and "run_path" in scr


def test_the_bootstrap_is_fail_soft():
    """A broken shim must never be the reason a suite stops producing a number."""
    src = cap.wrap_command(_MODULE_CMD, 4)[2]
    assert "try:" in src and "except Exception" in src
    # the target runs OUTSIDE the try, so an import failure cannot skip it
    body = src.split("except Exception")[1]
    assert "run_module" in body


def test_the_bootstrap_actually_installs_the_requested_count():
    src = cap.wrap_command(_MODULE_CMD, 7)[2]
    assert re.search(r"install\(7\)", src)


def test_the_bootstrap_loads_the_shim_by_path_not_as_a_gbench_import():
    """Two reasons: importing the package would drag all 89 suites into the harness
    subprocess for a 40-line shim, and swe_bench_pro invokes plain `python` rather
    than sys.executable, where gbench may not be importable at all."""
    src = cap.wrap_command(_MODULE_CMD, 4)[2]
    assert "spec_from_file_location" in src
    assert "gbench.runners" not in src
    assert cap.__file__ in src


def test_the_shim_can_be_loaded_standalone():
    """The bootstrap execs this file outside the package, so it must not import
    anything from gbench."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_standalone_cap_probe", cap.__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # would raise if it needed the package
    assert mod.resolve_threads(20) >= 0


@pytest.mark.parametrize("cmd", [_MODULE_CMD, _SCRIPT_CMD])
def test_disabled_cap_returns_the_command_untouched(cmd):
    assert cap.wrap_command(cmd, 0) == cmd


def test_unrecognised_command_shape_is_left_alone():
    weird = ["python", "-X", "utf8", "something"]
    assert cap.wrap_command(weird, 4) == weird
    assert cap.wrap_command(["python"], 4) == ["python"]
    assert cap.wrap_command(["python", "-m"], 4) == ["python", "-m"]


def test_apply_reports_the_count_it_used():
    with mock.patch.dict(os.environ, {cap.ENV_VAR: "3"}):
        wrapped, threads = cap.apply(_MODULE_CMD, 20, "copilot_bench_swe")
    assert threads == 3 and wrapped[1] == "-c"


def test_apply_disabled_returns_the_original_command_and_zero():
    with mock.patch.dict(os.environ, {cap.ENV_VAR: "0"}):
        wrapped, threads = cap.apply(_MODULE_CMD, 20, "copilot_bench_swe")
    assert threads == 0 and wrapped == _MODULE_CMD


# --------------------------------------------------------------------------- #
# the suites are actually wired to it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module,scorer", [
    ("swebench_common", "make_swebench_scorer"),
    ("multi_swe_bench", "_make_scorer"),
    ("swe_bench_pro", "_make_scorer"),
])
def test_every_docker_swe_suite_caps_its_containers(module, scorer):
    import importlib
    mod = importlib.import_module(f"gbench.runners.eval_suites.{module}")
    src = inspect.getsource(getattr(mod, scorer))
    assert "swe_thread_cap.apply(" in src, f"{module} runs uncapped containers"
    assert 'metrics["docker_thread_cap"]' in src, (
        f"{module} must record the cap it scored under")
