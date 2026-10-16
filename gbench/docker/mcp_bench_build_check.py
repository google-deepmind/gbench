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

"""Build-time check of the gbench-mcp-bench image: every server MCP-Bench's commands.json starts is
built and can start.

install.sh runs with its `set -e` turned into `set +e` and the post-install guards keep going past
a failed step (gbench/docker/mcp_bench.Dockerfile says why), so a server whose build failed would
otherwise surface only at run time, in the pre-run connection check. This check runs at the end of
the build and fails it, naming every server that is not ready.

For each commands.json entry, in its working directory (`cwd`, relative to a server directory, as
upstream writes it):
  - `uv run <rest>`: the server's own `.venv` must exist, and <rest> is checked with its python
    (or its console script);
  - `node <file> ...`: <file> must exist (the built JavaScript, e.g. dist/ or build/);
  - `python <file> ...`: <file> must exist and compile (checked in memory: nothing is written);
  - `python -m <module>`: <module> must be found, with the working directory first on sys.path as
    `python -m` puts it (a dotted module imports its package; no bytecode is written);
  - `<script> ...` under `uv run`: the console script must be in the server's `.venv/bin`.

Usage: python3 mcp_bench_build_check.py <mcp_servers directory>. Exit 0 when every server is ready,
1 otherwise (each failure listed), 2 when commands.json cannot be read.
"""

import json
import os
import shlex
import subprocess
import sys
from typing import Dict, List, Optional

_FIND_MODULE = ("import importlib.util, sys; sys.path.insert(0, ''); "
                "sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 3)")
_COMPILES = "import sys; compile(open(sys.argv[1], 'rb').read(), sys.argv[1], 'exec')"
_TIMEOUT_S = 300


def server_dir(servers: str, cwd: Optional[str]) -> str:
    """The server's working directory: commands.json writes it relative to a directory under
    mcp_servers/ (`../openapi-mcp-server`)."""
    return os.path.normpath(os.path.join(servers, "_", cwd or "."))


def _run(argv: List[str], cwd: str) -> Optional[str]:
    # PYTHONDONTWRITEBYTECODE: importing a server's package writes nothing into the checkout
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    try:
        done = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=_TIMEOUT_S,
                              env=env)
    except (OSError, subprocess.SubprocessError) as e:
        return f"{' '.join(argv[:2])}: {type(e).__name__}: {e}"
    if done.returncode:
        tail = (done.stderr or done.stdout or "").strip().splitlines()[-1:]
        return f"exit {done.returncode}" + (f": {tail[0][:200]}" if tail else "")
    return None


def check_server(servers: str, entry: Dict) -> Optional[str]:
    """None when the server commands.json describes by `entry` is ready, else what is missing."""
    cwd = server_dir(servers, entry.get("cwd"))
    if not os.path.isdir(cwd):
        return f"no directory {cwd}"
    try:
        tokens = shlex.split(str(entry.get("cmd") or ""))
    except ValueError as e:
        return f"cannot parse its command: {e}"
    python, bindir = "python3", None
    if tokens[:2] == ["uv", "run"]:
        bindir = os.path.join(cwd, ".venv", "bin")
        python = os.path.join(bindir, "python")
        if not os.path.isfile(python):
            return f"no .venv (expected {python})"
        tokens = tokens[2:]
    if not tokens:
        return "empty command"
    head = tokens[0]
    if head == "node":
        if len(tokens) < 2 or not os.path.isfile(os.path.join(cwd, tokens[1])):
            return f"no built entry point {os.path.join(cwd, tokens[1] if len(tokens) > 1 else '')}"
        return None
    if head == "python":
        if len(tokens) >= 3 and tokens[1] == "-m":
            why = _run([python, "-c", _FIND_MODULE, tokens[2]], cwd)
            return f"module {tokens[2]} not importable ({why})" if why else None
        if len(tokens) < 2:
            return "python without a script"
        script = os.path.join(cwd, tokens[1])
        if not os.path.isfile(script):
            return f"no script {script}"
        why = _run([python, "-c", _COMPILES, script], cwd)
        return f"script {script} does not compile ({why})" if why else None
    if bindir is not None:
        if not os.path.isfile(os.path.join(bindir, head)):
            return f"no console script {os.path.join(bindir, head)}"
        return None
    return f"unrecognized command {entry.get('cmd')!r}"


def main(argv: List[str]) -> int:
    servers = argv[1] if len(argv) > 1 else "/app/mcp-bench/mcp_servers"
    try:
        with open(os.path.join(servers, "commands.json"), encoding="utf-8") as f:
            commands = json.load(f)
    except (OSError, ValueError) as e:
        print(f"mcp_bench build check: cannot read commands.json: {e}", file=sys.stderr)
        return 2
    failed = {}
    for name in sorted(commands):
        why = check_server(servers, commands[name])
        if why:
            failed[name] = why
    ready = len(commands) - len(failed)
    print(f"mcp_bench build check: {ready}/{len(commands)} servers ready")
    for name, why in failed.items():
        print(f"  NOT READY {name}: {why}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
