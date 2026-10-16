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

"""nestful's Win Rate worker: upstream's 10 s alarm firing inside a socket call is the call's
timeout, not a failed use of the network; a partial line on the protocol stream never blocks the
parent past its deadline; an interpreter under /tmp stays visible in the sandbox. No network, no
model; files go to tmp only."""

import socket
import subprocess
import sys
import time

import pytest

from gbench.runners.eval_suites import nestful as N


@pytest.fixture(autouse=True)
def _restore_socket(monkeypatch):
    """_install_network_probe wraps the socket module's functions for its process: record them
    so they are restored after each test."""
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr",
                 "getnameinfo", "create_connection"):
        monkeypatch.setattr(socket, name, getattr(socket, name))
    for name in ("connect", "connect_ex", "sendto", "bind"):
        monkeypatch.setattr(socket.socket, name, getattr(socket.socket, name))


def test_the_alarm_inside_a_socket_call_is_not_a_failed_network_use(monkeypatch):
    def create_connection(*a, **k):
        raise TimeoutError("Time limit exceeded!")
    monkeypatch.setattr(socket, "create_connection", create_connection)
    refused = N._install_network_probe()
    with pytest.raises(TimeoutError):
        socket.create_connection(("example.com", 443))
    assert refused == []


def test_a_real_connection_failure_is_still_noted(monkeypatch):
    def create_connection(*a, **k):
        raise ConnectionRefusedError(111, "Connection refused")
    monkeypatch.setattr(socket, "create_connection", create_connection)
    refused = N._install_network_probe()
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("example.com", 443))
    assert len(refused) == 1 and refused[0].startswith("create_connection(")


def _worker_with(script):
    w = N._WinRateWorker.__new__(N._WinRateWorker)
    w._buf = b""
    w.proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE,
                              stdin=subprocess.PIPE)
    return w


def test_a_partial_line_does_not_block_past_the_deadline():
    w = _worker_with("import sys, time; sys.stdout.write('1'); sys.stdout.flush(); "
                     "time.sleep(30)")
    try:
        start = time.monotonic()
        assert w._readline(0.5) is None
        assert time.monotonic() - start < 5
    finally:
        w.proc.kill()
        w.proc.wait()


def test_lines_are_returned_one_at_a_time_and_the_end_of_the_stream_is_not_a_verdict():
    w = _worker_with("import sys; sys.stdout.write('1\\n0\\npartial'); sys.stdout.flush()")
    try:
        assert w._readline(5) == b"1\n"
        assert w._readline(5) == b"0\n"
        assert w._readline(5) == b"partial"            # no newline: never b"1\n" / b"0\n"
        assert w._readline(1) is None
    finally:
        w.proc.wait()


def test_an_interpreter_under_tmp_is_bound_after_the_private_tmp(monkeypatch, tmp_path):
    monkeypatch.setattr(N.shutil, "which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setattr(N, "_bwrap_options", lambda: frozenset())
    monkeypatch.setattr(N, "_interpreter_dirs", lambda: ["/opt/py", "/tmp/venv"])
    argv = N._sandbox_argv(str(tmp_path / "funcs"))
    tmpfs = argv.index("--tmpfs")
    venv = [i for i, a in enumerate(argv) if a == "/tmp/venv"]
    opt = [i for i, a in enumerate(argv) if a == "/opt/py"]
    assert venv and min(venv) > tmpfs, "the /tmp interpreter must come after the private /tmp"
    assert opt and max(opt) < tmpfs


def test_a_sandbox_that_does_not_start_names_the_interpreter(monkeypatch, tmp_path):
    monkeypatch.setattr(N, "_bwrap_options", lambda: frozenset())
    monkeypatch.setattr(N, "_sandbox_argv", lambda func_dir, hosts_fd=None: [
        sys.executable, "-c", "raise SystemExit(3)"])
    with pytest.raises(N.WinRateUnavailable) as err:
        N._WinRateWorker(str(tmp_path))
    assert f"could not start the interpreter {sys.executable}" in str(err.value)
