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

"""A run started under `nohup` survives its terminal closing.

nohup sets SIGHUP to SIG_IGN; gbench's termination handlers replaced it, so on 2026-10-06 a
smoke run started with `nohup gbench ... &` stopped when the terminal closed. A signal the
process was started ignoring now stays ignored; SIGTERM, and SIGHUP without nohup, still stop
the run cleanly.
"""

import os
import signal
import subprocess
import sys
import textwrap

SCRIPT = textwrap.dedent("""
    import os, signal, sys, time
    from gbench import cli
    with cli._termination_handlers():
        print("ready", flush=True)
        try:
            os.kill(os.getpid(), signal.SIGHUP)
            time.sleep(0.5)
            print("survived", flush=True)
        except SystemExit as e:
            print("stopped", e.code, flush=True)
            sys.exit(0)
""")


def _run(ignore_hup):
    def pre():
        # set explicitly: a pytest started under nohup would otherwise hand SIG_IGN down
        signal.signal(signal.SIGHUP, signal.SIG_IGN if ignore_hup else signal.SIG_DFL)
    return subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True,
                          timeout=120, preexec_fn=pre, cwd=os.path.dirname(os.path.dirname(
                              os.path.abspath(__file__))))


def test_sighup_ignored_by_nohup_stays_ignored():
    out = _run(ignore_hup=True)
    assert "survived" in out.stdout, out.stdout + out.stderr


def test_sighup_without_nohup_still_stops_the_run_cleanly():
    out = _run(ignore_hup=False)
    assert f"stopped {128 + signal.SIGHUP}" in out.stdout, out.stdout + out.stderr
