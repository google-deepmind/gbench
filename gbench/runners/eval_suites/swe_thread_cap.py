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

"""Cap OpenMP/BLAS threads inside SWE-bench execution containers.

Why this exists
---------------
The SWE harnesses create one container per instance and never set a thread
limit, so every numerical library inside sees the *host* core count. On this
box that is 96, and the harness is handed ``max_workers`` = the LLM request
concurrency (20), so a single scoring pass can ask for 20 x 96 = 1920 OpenMP
threads on 96 cores. The datasets in these unit tests are tiny, so essentially
all of that time goes into spawning and spin-waiting on threads.

Measured 2026-08-17, ``scikit-learn__scikit-learn-14710``, same image, same
unpatched source, same 4 CPUs, only the thread cap differing:

======================  ====================  ==================
``OMP_NUM_THREADS``     tests completed       wall clock
======================  ====================  ==================
``4``                   78 / 78 passed        6.28 s
unset (library sees 96) 34 / 78               killed at 541.53 s
======================  ====================  ==================

In the real run that instance had ~91 cores to itself and still did not finish:
it burned the harness's full 1800 s per-instance timeout and was scored as an
error. That single instance was 51% of the whole four-suite sweep. The failure
is ours, not the model's - the patch under test only touched
``_check_early_stopping_scorer``, which the stalled test never calls.

How it works
------------
``swebench`` creates each container with ``command="tail -f /dev/null"`` and
then ``exec_run``s the eval script. Docker applies the container's create-time
environment to exec'd processes (verified directly, including through the
``bash -lc`` login shell the eval script runs under), so setting the variables
at create time is enough - there is no need to touch the generated eval script.

The harnesses run as *subprocesses*, so a monkeypatch applied here would not
reach them. :func:`wrap_command` therefore re-launches the same target through
a small ``python -c`` bootstrap that calls :func:`install` first. The bootstrap
is fail-soft: if the shim cannot be imported or applied, the harness still runs
exactly as it did before, uncapped.
"""

import contextvars
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

#: Env vars honoured by the threading runtimes these test suites pull in.
THREAD_VARS = (
    "OMP_NUM_THREADS",        # libgomp / libiomp - scikit-learn, scipy, numpy
    "OPENBLAS_NUM_THREADS",   # OpenBLAS
    "MKL_NUM_THREADS",        # Intel MKL
    "NUMEXPR_NUM_THREADS",    # numexpr (pandas.eval)
    "VECLIB_MAXIMUM_THREADS",  # Accelerate
    "RAYON_NUM_THREADS",      # Rust extensions (ruff, polars, tokenizers)
)

#: Upper bound per container. 4 is the measured-good value in the table above;
#: the pathology is per-parallel-region overhead on tiny arrays, not core
#: contention, so raising this with the core count does NOT help - the stalled
#: instance had ~91 cores to itself.
MAX_THREADS_PER_CONTAINER = 4

#: Set to 0 (or "off") to disable the cap entirely and restore the old behaviour.
ENV_VAR = "GBENCH_SWE_OMP_THREADS"

#: The containers the whole suite runs at one time, when its parts run at once and each part
#: passes only its own share as `max_workers` (swe_bench_live runs its tab lanes at once, each
#: with a share of --sandboxes): resolve_threads then shares the box over the whole suite, as a
#: run of the parts one after the other, each with the whole budget, did. Set for a part's
#: context (contextvars: it follows the part into asyncio.run and asyncio.to_thread); None
#: outside one.
SUITE_CONTAINERS: "contextvars.ContextVar[Optional[int]]" = contextvars.ContextVar(
    "gbench_swe_suite_containers", default=None)


def resolve_threads(max_workers: int) -> int:
    """Threads to allow per container. 0 means "do not cap". Within a context where
    SUITE_CONTAINERS is set, the box is shared over that many containers instead of
    `max_workers` (this part's share of them)."""
    raw = (os.environ.get(ENV_VAR) or "").strip()
    if raw:
        if raw.lower() in ("off", "none", "false"):
            return 0
        try:
            return max(0, int(raw))
        except ValueError:
            logger.warning("%s=%r is not an integer; using the computed default.", ENV_VAR, raw)
    cpus = os.cpu_count() or 8
    suite = SUITE_CONTAINERS.get()
    if suite:
        max_workers = max(int(max_workers or 0), int(suite))
    # Share the box between the concurrently-running containers, then clamp.
    return max(1, min(MAX_THREADS_PER_CONTAINER, cpus // max(1, max_workers)))


def thread_env(threads: int) -> Dict[str, str]:
    """The variables to set inside the container."""
    return {var: str(threads) for var in THREAD_VARS}


def install(threads: int, _collection: Optional[type] = None) -> bool:
    """Patch the docker SDK so every container is created with the thread cap.

    Runs *inside the harness subprocess* via the :func:`wrap_command` bootstrap.
    Returns True if the patch was applied (or was already in place).

    ``_collection`` is a testing seam so the unit tests do not have to mutate the
    real ``docker`` SDK class; production callers leave it None.
    """
    if threads <= 0:
        return False
    collection = _collection
    if collection is None:
        from docker.models.containers import ContainerCollection
        collection = ContainerCollection
    if getattr(collection, "_gbench_thread_cap", None) == threads:
        return True

    original = getattr(collection, "_gbench_thread_cap_original", None) or collection.create
    env = thread_env(threads)

    def create(self, image=None, command=None, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["environment"] = _merge_env(kwargs.get("environment"), env)
        return original(self, image=image, command=command, **kwargs)

    collection.create = create
    collection._gbench_thread_cap_original = original
    collection._gbench_thread_cap = threads
    return True


def _merge_env(existing: Union[None, Dict[str, Any], Sequence[str]],
               env: Dict[str, str]) -> Union[Dict[str, str], List[str]]:
    """Add `env` to whatever the caller already passed, without dropping it.

    The docker SDK accepts either a dict or a list of ``"NAME=value"`` strings.
    A caller-supplied value for one of our variables wins - if something
    deliberately asked for a different thread count, honour it.
    """
    if existing is None:
        return dict(env)
    if isinstance(existing, dict):
        merged = dict(env)
        merged.update({str(k): str(v) for k, v in existing.items()})
        return merged
    already = {str(item).split("=", 1)[0] for item in existing}
    return list(existing) + [f"{k}={v}" for k, v in env.items() if k not in already]


def _bootstrap_source(threads: int, module: Optional[str], script: Optional[str]) -> str:
    """Python for ``-c``: install the cap, then run the original target."""
    if module:
        target = f"runpy.run_module({module!r}, run_name='__main__', alter_sys=True)"
    else:
        target = f"sys.argv[0] = {script!r}; runpy.run_path({script!r}, run_name='__main__')"
    # Load this module by absolute path rather than as `gbench.runners.eval_suites.
    # swe_thread_cap`, for two reasons:
    #   1. importing the package would drag all 89 suites (and their dataset/ML
    #      dependencies) into the harness subprocess to run a 40-line shim;
    #   2. swe_bench_pro invokes plain `python`, not `sys.executable`, so gbench may
    #      not even be importable there - but any interpreter can exec a file path.
    # Loading by path also avoids putting the eval_suites directory, which has
    # generic module names like `base`, on the harness's sys.path. This module
    # deliberately imports nothing from gbench so it can load standalone.
    #
    # (Note: `python -m swebench.harness.run_evaluation` emits a runpy
    # RuntimeWarning about run_evaluation already being in sys.modules. That is
    # swebench's own doing - `swebench/harness/__init__.py` imports it - and the
    # stock command warns identically. It is not caused by this bootstrap.)
    #
    # Fail-soft: a broken shim must never stop the harness from running.
    return (
        "import importlib.util, runpy, sys\n"
        "try:\n"
        f"    _spec = importlib.util.spec_from_file_location('_gbench_swe_thread_cap', {__file__!r})\n"
        "    _shim = importlib.util.module_from_spec(_spec)\n"
        "    _spec.loader.exec_module(_shim)\n"
        f"    _shim.install({threads})\n"
        "except Exception as exc:\n"
        "    sys.stderr.write('gbench: docker thread cap not applied: %r\\n' % (exc,))\n"
        f"{target}\n"
    )


def wrap_command(cmd: Sequence[str], threads: int) -> List[str]:
    """Re-express `cmd` so the thread cap is installed before the harness starts.

    Accepts the two shapes the SWE suites use::

        [python, "-m", "swebench.harness.run_evaluation", *args]
        [python, "/path/to/swe_bench_pro_eval.py", *args]

    Returns `cmd` unchanged when the cap is disabled or the shape is unrecognised,
    so this can never be the reason a suite stops working.
    """
    cmd = list(cmd)
    if threads <= 0 or len(cmd) < 2:
        return cmd
    if cmd[1] == "-m":
        if len(cmd) < 3:
            return cmd
        source = _bootstrap_source(threads, module=cmd[2], script=None)
        rest = cmd[3:]
    elif cmd[1].endswith(".py"):
        source = _bootstrap_source(threads, module=None, script=cmd[1])
        rest = cmd[2:]
    else:
        logger.debug("swe_thread_cap: unrecognised command shape %r; leaving it alone.", cmd[:2])
        return cmd
    return [cmd[0], "-c", source, *rest]


def apply(cmd: Sequence[str], max_workers: int, eval_name: str) -> tuple:
    """Convenience wrapper: resolve the cap, wrap the command, log what happened.

    Returns ``(wrapped_cmd, threads)`` so the caller can record `threads` on the
    result - a run that was scored under a thread cap should say so.
    """
    threads = resolve_threads(max_workers)
    wrapped = wrap_command(cmd, threads)
    if threads <= 0:
        logger.info("[%s] docker thread cap disabled via %s; containers see all %s cores.",
                    eval_name, ENV_VAR, os.cpu_count())
    elif wrapped is not cmd and wrapped != list(cmd):
        logger.info("[%s] capping eval containers at %d thread(s) each "
                    "(%s workers on %s cores); set %s=0 to disable.",
                    eval_name, threads, max_workers, os.cpu_count(), ENV_VAR)
    return wrapped, threads
