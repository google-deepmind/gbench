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

"""The process groups of the vLLM servers and offline benchmarks gbench launches.

Each one is started in its own session (``start_new_session=True``), so its process group id is
its pid and one ``killpg`` reaches vLLM's API server, its EngineCore and every worker. A group is
tracked from launch until it is stopped; an atexit hook stops whatever is still tracked when the
interpreter exits. The CLI turns SIGTERM and SIGHUP into SystemExit while a run executes, so a
terminated run goes through the same path and leaves no server holding the GPUs.
"""

import atexit
import logging
import os
import signal
import subprocess
import threading
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_LIVE: Dict[int, subprocess.Popen] = {}
# Reentrant: the CLI's SIGTERM/SIGHUP handler (request_stop) can run on the main thread while
# that thread already holds the lock.
_LOCK = threading.RLock()
#: The signal that asked this run to stop, and when (request_stop): set by the CLI's
#: SIGTERM/SIGHUP handler while the run's main thread unwinds, and cleared when it has.
_STOP: Dict[str, float] = {}
#: thread ident -> (signal, time) for a worker thread the signal left running (a suite under
#: --suite-timeout runs in one): the main thread's SystemExit never reaches it, so it goes on
#: until it looks here (evals._run_with_timeout hands the signal over).
_THREAD_STOP: Dict[int, Tuple[int, float]] = {}


def request_stop(signum: int, thread: Optional[int] = None) -> None:
    """Record that the run was asked to stop by signal `signum`: for the whole run (the CLI's
    handler), or for the worker thread `thread` (a thread ident). The first signal is kept."""
    with _LOCK:
        if thread is None:
            if "signum" not in _STOP:
                _STOP.update(signum=float(signum), at=time.time())
        else:
            _THREAD_STOP.setdefault(int(thread), (int(signum), time.time()))


def stop_requested(thread: Optional[int] = None) -> Optional[Tuple[int, float]]:
    """(signal number, time it arrived) when the run, or the worker thread `thread` (default:
    the calling thread), was asked to stop, else None. Work that can outlive the main thread's
    SystemExit checks it before starting more (a suite's next setting, the scoring of rollouts
    a signal cut short)."""
    with _LOCK:
        if "signum" in _STOP:
            return int(_STOP["signum"]), float(_STOP["at"])
        return _THREAD_STOP.get(threading.get_ident() if thread is None else int(thread))


def clear_stop(thread: Optional[int] = None) -> None:
    """Forget the run's recorded stop, or the one handed to worker thread `thread`."""
    with _LOCK:
        if thread is None:
            _STOP.clear()
        else:
            _THREAD_STOP.pop(int(thread), None)


def track(proc: subprocess.Popen) -> None:
    """Remember a process started with start_new_session=True until stop_group() stops it."""
    with _LOCK:
        _LIVE[proc.pid] = proc


def untrack(proc: subprocess.Popen) -> None:
    with _LOCK:
        _LIVE.pop(proc.pid, None)


def tracked_pids() -> list:
    with _LOCK:
        return sorted(_LIVE)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_group(proc: subprocess.Popen, grace_s: float = 10.0) -> None:
    """SIGTERM the process group of ``proc`` (its pid: the process is a session leader), wait up to
    ``grace_s`` for the group to exit, then SIGKILL what is left. Untracks it. Idempotent: a group
    that is already gone is not an error."""
    untrack(proc)
    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + max(0.0, grace_s)
    try:
        proc.wait(timeout=max(0.1, grace_s))
    except subprocess.TimeoutExpired:
        pass
    # The leader may exit before the EngineCore and workers it spawned: wait for the whole group.
    while _group_alive(pgid) and time.monotonic() < deadline:
        time.sleep(0.2)
    if _group_alive(pgid):
        logger.warning(f"Process group {pgid} did not exit after SIGTERM; sending SIGKILL")
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def stop_all(grace_s: float = 5.0) -> None:
    """Stop every tracked process group (the atexit hook)."""
    with _LOCK:
        procs = list(_LIVE.values())
    for proc in procs:
        try:
            logger.info(f"Stopping process group {proc.pid} left running at exit")
            stop_group(proc, grace_s=grace_s)
        except Exception as e:                                             # noqa: BLE001
            logger.error(f"could not stop process group {proc.pid}: {e}")


atexit.register(stop_all)
