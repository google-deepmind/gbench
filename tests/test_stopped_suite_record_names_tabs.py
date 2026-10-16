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

"""A suite stopped by a signal keeps, in its saved record, what the stop halted.

swe_bench_live raises TabScheduleStopped with a record naming the tabs cut short, not started and
completed; evals' stopped record used to drop it, so only the log named them.
"""

import inspect
import threading

from gbench.runners import evals
from gbench.utils import process_groups


def test_evals_adds_the_suites_own_stop_record():
    src = inspect.getsource(evals)
    assert 'own = getattr(stop_exc, "record", None)' in src
    assert 'if key.startswith("tabs_") or key not in stopped:' in src


def test_the_stop_lock_is_reentrant():
    # the SIGTERM handler can call request_stop while the main thread holds the lock
    assert isinstance(process_groups._LOCK, type(threading.RLock()))
    with process_groups._LOCK:
        with process_groups._LOCK:
            pass
