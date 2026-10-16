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

"""The gbench-mcp-bench image's build check: a server commands.json starts whose build output is
missing fails the build, named; the Dockerfile runs the check and no longer tolerates a failed
install of the runner's packages. A fake mcp_servers tree in tmp; no Docker, no network."""

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "mcp_bench_build_check", REPO / "gbench" / "docker" / "mcp_bench_build_check.py")
C = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(C)


def _tree(tmp_path, commands):
    servers = tmp_path / "mcp_servers"
    servers.mkdir()
    (servers / "commands.json").write_text(json.dumps(commands))
    return servers


def _node(servers, name, entry):
    d = servers / name
    (d / Path(entry).parent).mkdir(parents=True, exist_ok=True)
    (d / entry).write_text("// built\n")


def test_every_ready_server_passes(tmp_path, capsys):
    servers = _tree(tmp_path, {
        "Node": {"cmd": "node build/index.js", "cwd": "../node-srv"},
        "Script": {"cmd": "python server.py", "cwd": "../py-srv"},
        "Module": {"cmd": "python -m json.tool", "cwd": "../mod-srv"},
    })
    _node(servers, "node-srv", "build/index.js")
    (servers / "py-srv").mkdir()
    (servers / "py-srv" / "server.py").write_text("print('ok')\n")
    (servers / "mod-srv").mkdir()
    assert C.main(["check", str(servers)]) == 0
    assert "3/3 servers ready" in capsys.readouterr().out
    # nothing written into the checkout
    assert not any(p.name == "__pycache__" for p in servers.rglob("*"))


def test_a_missing_build_fails_naming_each_server(tmp_path, capsys):
    servers = _tree(tmp_path, {
        "OKX Exchange": {"cmd": "node build/index.js", "cwd": "../okx-mcp"},
        "Broken Script": {"cmd": "python server.py", "cwd": "../broken"},
        "Wikipedia": {"cmd": "uv run python -m wikipedia_mcp", "cwd": "../wikipedia-mcp"},
        "Missing Module": {"cmd": "python -m gbench_no_such_module_xyz", "cwd": "../mm"},
        "BioMCP": {"cmd": "uv run biomcp run", "cwd": "../biomcp"},
    })
    (servers / "okx-mcp" / "src").mkdir(parents=True)          # built output missing
    (servers / "broken").mkdir()
    (servers / "broken" / "server.py").write_text("def broken(:\n")
    (servers / "wikipedia-mcp").mkdir()                         # no .venv
    (servers / "mm").mkdir()
    venv_bin = servers / "biomcp" / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(sys.executable)          # a .venv without the script
    assert C.main(["check", str(servers)]) == 1
    captured = capsys.readouterr()
    assert "0/5 servers ready" in captured.out
    for name, words in (("OKX Exchange", "no built entry point"),
                        ("Broken Script", "does not compile"),
                        ("Wikipedia", "no .venv"),
                        ("Missing Module", "not importable"),
                        ("BioMCP", "no console script")):
        [line] = [x for x in captured.err.splitlines() if f"NOT READY {name}: " in x]
        assert words in line, (name, line)


def test_an_unreadable_commands_file_is_its_own_failure(tmp_path):
    assert C.main(["check", str(tmp_path)]) == 2


def test_the_working_directory_is_resolved_as_upstream_writes_it(tmp_path):
    assert C.server_dir("/app/mcp-bench/mcp_servers", "../call-for-papers-mcp/main") == \
        "/app/mcp-bench/mcp_servers/call-for-papers-mcp/main"


def test_the_dockerfile_runs_the_check_and_no_longer_masks_runner_installs():
    df = (REPO / "gbench" / "docker" / "mcp_bench.Dockerfile").read_text()
    assert "COPY mcp_bench_build_check.py /opt/gbench/mcp_bench_build_check.py" in df
    assert "RUN python3 /opt/gbench/mcp_bench_build_check.py /app/mcp-bench/mcp_servers" in df
    runner = df[df.index("python3 -m pip install --no-cache-dir openai json_repair"):]
    runner = runner[:runner.index("\n\n")]
    assert "|| true" not in runner
    # the check runs after every install step
    assert df.index("mcp_bench_build_check.py /app/mcp-bench") > df.index('"mcp[cli]==1.16.0"')
