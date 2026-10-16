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

"""swe_bench_live: the task containers' runtime runs the model's heredocs (gbench's SWE-ReX fix).

SWE-agent-for-eval's DockerDeployment builds a runtime layer on every task image with the latest
swe-rex from PyPI (1.4.0, bashlex 0.18), whose BashSession split each command with bashlex and
joined the pieces with " ; ": `cat <<EOF > f` lost its body, a heredoc followed by another command
got that command on its terminator's line, and the shell waited until the fork's 90-minute
command limit (2026-10-08 smoke run). gbench/docker/sweagent_for_eval_swerex_heredoc.patch fixes
the vendored SWE-ReX: the runtime layer installs pinned distributions and the patched session
code (swerex/runtime/local.py), which runs a command holding a heredoc whole.

These tests apply the patch's local.py part to swe-rex 1.4.0 as installed here and drive the real
BashSession over a real bash (started without rc files, so the host's shell setup plays no part):
the stock session hangs on the model's command, the patched one runs it, and everything else
behaves as before. The image side (pins, sha256 chain, labels) is checked against the patch text.
"""

import asyncio
import os
import hashlib
import importlib.util
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gbench.runners.eval_suites import swe_bench_live as LV

REPO = Path(__file__).resolve().parents[1]
PATCH = REPO / LV.SWEREX_RUNTIME_PATCH
DOCKERFILE = REPO / "gbench" / "docker" / "sweagent_for_eval.Dockerfile"

swerex = pytest.importorskip("swerex", reason="swe-rex 1.4.0 comes with the evals extra")
pexpect = pytest.importorskip("pexpect")

#: The command Gemma sent in almost every instance of the 2026-10-08 smoke run.
MODEL_COMMAND = 'cat <<EOF > {d}/reproduce_issue.py\nprint("ok")\nEOF\npython3 {d}/reproduce_issue.py'


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def runtimes(tmp_path_factory):
    """(stock, patched): swe-rex 1.4.0's swerex/runtime/local.py as installed here, and the same
    file with the patch's local.py part applied (git apply, as the image build does)."""
    import swerex.runtime.local as installed
    src = Path(installed.__file__)
    assert hashlib.sha256(src.read_bytes()).hexdigest() == LV.SWEREX_RUNTIME_UPSTREAM_SHA256, (
        "the installed swe-rex is not 1.4.0 (its runtime/local.py differs)")
    root = tmp_path_factory.mktemp("swerex_patch")
    (root / "swerex" / "runtime").mkdir(parents=True)
    shutil.copy(src, root / "swerex" / "runtime" / "local.py")
    subprocess.run(["git", "apply", "--include=swerex/runtime/local.py", str(PATCH)], cwd=root,
                   check=True, capture_output=True)
    patched = root / "swerex" / "runtime" / "local.py"
    assert hashlib.sha256(patched.read_bytes()).hexdigest() == LV.SWEREX_RUNTIME_SHA256
    return (_load(src, "gbench_test_swerex_local_stock"),
            _load(patched, "gbench_test_swerex_local_patched"))


def _session(mod, monkeypatch, tmp_path):
    """A BashSession of `mod` over a bash that reads a minimal rc file instead of ~/.bashrc (the
    runtime spawns `/usr/bin/env bash`; the host's rc file may start another shell). Like a
    task image's root .bashrc, it sets its own prompt, which the session then replaces."""
    rc = tmp_path / "bashrc"
    rc.write_text("PS1='\\u@\\h:\\w\\$ '\n")
    real_spawn = mod.pexpect.spawn
    monkeypatch.setattr(mod.pexpect, "spawn", lambda cmd, **kw: real_spawn(
        f"/usr/bin/env bash --noprofile --rcfile {rc}" if cmd == "/usr/bin/env bash" else cmd,
        **kw))
    from swerex.runtime.abstract import CreateBashSessionRequest
    s = mod.BashSession(CreateBashSessionRequest(session="t"))
    asyncio.run(s.start())
    return s


def _run(session, command, timeout=10.0, check="silent"):
    """SWEEnv.communicate's call: check='ignore' there maps to 'silent' here."""
    from swerex.runtime.abstract import BashAction
    t0 = time.monotonic()
    obs = asyncio.run(session.run(BashAction(command=command, timeout=timeout, check=check)))
    return obs, time.monotonic() - t0


# ============================================================================================
# the defect, and the fix, in the session code the task containers run
# ============================================================================================
def test_the_stock_runtime_hangs_on_the_models_heredoc(runtimes, monkeypatch, tmp_path):
    stock, _ = runtimes
    from swerex.exceptions import CommandTimeoutError
    s = _session(stock, monkeypatch, tmp_path)
    with pytest.raises(CommandTimeoutError) as exc:
        _run(s, MODEL_COMMAND.format(d=tmp_path), timeout=3)
    assert "cat <<EOF > " in str(exc.value) and " ; python3 " in str(exc.value)
    assert not (tmp_path / "reproduce_issue.py").exists()


def test_the_patched_runtime_runs_the_models_heredoc(runtimes, monkeypatch, tmp_path):
    _, patched = runtimes
    files = tmp_path / "command_files"
    files.mkdir()
    monkeypatch.setattr(patched.tempfile, "tempdir", str(files))
    s = _session(patched, monkeypatch, tmp_path)
    obs, took = _run(s, MODEL_COMMAND.format(d=tmp_path))
    assert took < 5 and obs.exit_code == 0
    assert obs.output.strip() == "ok"
    assert (tmp_path / "reproduce_issue.py").read_text() == 'print("ok")\n'
    assert list(files.iterdir()) == []          # the command's file is gone
    obs, _ = _run(s, f"cat <<EOF >/dev/null\nx\nEOF\nls {files}")
    assert obs.output.strip() == ""             # gone before the command's first word ran


@pytest.mark.parametrize("command,want_output,want_code", [
    ("python3 - <<EOF 2>&1\nimport sys\nprint('out'); print('err', file=sys.stderr)\nEOF",
     "out\nerr", 0),
    ("cat <<EOF | sh\necho piped-ok\nEOF", "piped-ok", 0),
    ("cat <<A > {d}/a\nfirst\nA\ncat <<B > {d}/b\nsecond\nB\ncat {d}/a {d}/b", "first\nsecond", 0),
    ("cat <<'EOF'\n$HOME `echo x` $(echo y)\nEOF", "$HOME `echo x` $(echo y)", 0),
    ("cat <<-EOF\n\tindented\n\tEOF", "indented", 0),
    ("X=v\ncat <<EOF\n$X $(echo sub)\nEOF", "v sub", 0),
    ("cat > {d}/f <<EOF\nx\nEOF\ncat {d}/f", "x", 0),
    ("cat <<EOF > {d}/x\nhi\nEOF\nfalse", "", 1),
    ("python3 - <<EOF\nimport sys; sys.exit(7)\nEOF", "", 7),
    ("for i in 1 2; do cat <<EOF\nloop $i\nEOF\ndone", "loop 1\nloop 2", 0),
    ("echo $((1<<4))", "16", 0),
    ("cat <<< here-string", "here-string", 0),
    ("echo one\necho two", "one\ntwo", 0),
    ("true && echo and-ok", "and-ok", 0),
    ("echo a ; echo b", "a\nb", 0),
    ("ls /nonexistent-gbench-dir 2>/dev/null", "", 2),
])
def test_the_patched_runtime_runs_commands_as_bash_does(runtimes, monkeypatch, tmp_path,
                                                        command, want_output, want_code):
    _, patched = runtimes
    s = _session(patched, monkeypatch, tmp_path)
    obs, took = _run(s, command.replace("{d}", str(tmp_path)))
    assert took < 5
    assert (obs.output.strip(), obs.exit_code) == (want_output, want_code)


def test_session_state_persists_across_heredoc_commands(runtimes, monkeypatch, tmp_path):
    _, patched = runtimes
    s = _session(patched, monkeypatch, tmp_path)
    _run(s, f"cd {tmp_path}\nexport HD=persisted\ncat <<EOF > here.txt\n$HD\nEOF")
    obs, _ = _run(s, "pwd; cat here.txt; echo $HD")
    assert obs.output.strip().split("\n") == [str(tmp_path), "persisted", "persisted"]


def test_a_background_job_then_a_command(runtimes, monkeypatch, tmp_path):
    """`a &` then `b` on the next line: the stock join made it `a & ; b`, a syntax error."""
    stock, patched = runtimes
    command = f"(sleep 0.2; echo bg > {tmp_path}/bg.txt) &\necho started"
    obs, _ = _run(_session(stock, monkeypatch, tmp_path), command)
    assert "syntax error" in obs.output and obs.exit_code == 2
    assert patched._join_bash_commands(["sleep 1 &", "echo x"]) == "sleep 1 & echo x"
    assert patched._join_bash_commands(["a && b", "c"]) == "a && b ; c"
    assert patched._join_bash_commands(["echo \\&", "c"]) == "echo \\& ; c"
    s = _session(patched, monkeypatch, tmp_path)
    obs, _ = _run(s, command)
    assert obs.exit_code == 0 and obs.output.strip().endswith("started")
    obs, _ = _run(s, f"sleep 0.5; cat {tmp_path}/bg.txt")
    assert obs.output.strip().endswith("bg")


def test_an_unterminated_heredoc_ends_at_the_end_of_the_command(runtimes, monkeypatch,
                                                                tmp_path):
    """As in a script: bash warns and ends the body at the end of the text (the stock runtime
    waited for the terminator)."""
    _, patched = runtimes
    s = _session(patched, monkeypatch, tmp_path)
    obs, took = _run(s, f"cat <<EOF > {tmp_path}/u.txt\nno terminator")
    assert took < 5 and "delimited by end-of-file" in obs.output
    assert (tmp_path / "u.txt").read_text() == "no terminator\n"


def test_a_long_heredoc_command_still_times_out_and_the_session_recovers(runtimes, monkeypatch,
                                                                         tmp_path):
    """SWE-agent's command-cancelled path: CommandTimeoutError, then an interrupt; the session
    is usable after it."""
    _, patched = runtimes
    from swerex.exceptions import CommandTimeoutError
    from swerex.runtime.abstract import BashInterruptAction
    s = _session(patched, monkeypatch, tmp_path)
    with pytest.raises(CommandTimeoutError):
        _run(s, f"cat <<EOF > {tmp_path}/s.sh\nsleep 60\nEOF\nbash {tmp_path}/s.sh", timeout=1.5)
    asyncio.run(s.run(BashInterruptAction(session="t")))
    obs, _ = _run(s, "echo alive")
    assert obs.output.strip() == "alive" and obs.exit_code == 0


def test_a_large_heredoc_is_not_typed_into_the_terminal(runtimes, monkeypatch, tmp_path):
    """One line is sent whatever the command's size (its text goes in a file), so a long file
    written with a heredoc neither fills the terminal nor reaches a command's stdin."""
    _, patched = runtimes
    body = "".join(f"x{i} = {i}\n" for i in range(5000))
    s = _session(patched, monkeypatch, tmp_path)
    obs, took = _run(s, f"cat <<'EOF' > {tmp_path}/big.py\n{body}print(x4999)\nEOF\n"
                        f"python3 {tmp_path}/big.py")
    assert took < 10 and obs.output.strip() == "4999"
    obs, took = _run(s, "cat <<EOF >/dev/null\nx\nEOF\ntimeout 1 head -c 10; echo after")
    assert obs.output.strip() == "after"


def test_the_split_of_a_heredoc_free_command_is_unchanged(runtimes):
    stock, patched = runtimes
    for command in ("echo a\necho b", "a && b\nc", "x=$(ls)\necho $x", "echo 'a ; b'",
                    "for i in 1; do echo $i; done\necho z"):
        assert patched._split_bash_command(command) == stock._split_bash_command(command)
    heredoc = "cat <<EOF > f\nx\nEOF\npython3 f"
    assert patched._split_bash_command(heredoc) == [heredoc]
    line, path = patched._as_single_command(heredoc)
    try:
        assert Path(path).read_text() == heredoc
        q = re.escape(path)
        assert re.fullmatch(rf'builtin eval "\$\(command -p cat {q}; command -p rm -f {q}\)"',
                            line)
    finally:
        Path(path).unlink(missing_ok=True)


# ============================================================================================
# the image side: pins, sha256 chain, labels
# ============================================================================================
def _patch_text():
    return PATCH.read_text(encoding="utf-8")


def test_the_patch_carries_the_pins_and_the_session_code_hash():
    text = _patch_text()
    assert f'+GBENCH_RUNTIME_LOCAL_SHA256 = "{LV.SWEREX_RUNTIME_SHA256}"' in text
    assert f'+GBENCH_RUNTIME_LOCAL_UPSTREAM_SHA256 = "{LV.SWEREX_RUNTIME_UPSTREAM_SHA256}"' in text
    block = text[text.index("+GBENCH_RUNTIME_PINS = ("):text.index("+)\n", text.index(
        "+GBENCH_RUNTIME_PINS = ("))]
    pins = tuple(re.findall(r'"([^"]+==[^"]+)"', block))
    assert pins == LV.SWEREX_RUNTIME_PINS and pins[0] == "swe-rex==1.4.0"
    assert "bashlex==0.18" in pins and all(p.count("==") == 1 for p in pins)
    # upstream's unpinned install is replaced, and pip resolves nothing
    assert '-            f"RUN /root/python3.11/bin/pip3 install --no-cache-dir {PACKAGE_NAME}' in text
    assert "install --no-cache-dir --no-deps" in text and "pip3 check" in text


def test_the_dockerfile_applies_the_patch_and_labels_the_image():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert f"ARG SWEREX_RUNTIME_SHA256={LV.SWEREX_RUNTIME_SHA256}" in text
    assert f"ARG SWEREX_RUNTIME_UPSTREAM_SHA256={LV.SWEREX_RUNTIME_UPSTREAM_SHA256}" in text
    i_cp = text.index("cp -r swerex venv/lib/python3.12/site-packages/swerex")
    i_apply = text.index("git apply --unsafe-paths "
                         "--directory=/opt/SWE-agent-for-eval/venv/lib/python3.12/site-packages")
    assert i_cp < i_apply
    assert 'org.gbench.sweagent_for_eval_swerex_runtime="${SWEREX_RUNTIME_SHA256}"' in text
    assert f"docker build -t {LV.SWEAGENT_IMAGE_DEFAULT} " in text
    assert LV.SWEAGENT_IMAGE_DEFAULT == "gbench-sweagent-for-eval:82422422-rex3"


def test_the_patch_applies_to_the_vendored_docker_py_shape():
    """The docker.py hunk replaces exactly upstream's unpinned install line inside
    glibc_dockerfile and keeps the swerex-remote symlink and version check after it."""
    text = _patch_text()
    hunk = text[text.index("--- a/swerex/deployment/docker.py"):]
    assert "+            + _gbench_runtime_dockerfile_steps() +" in hunk
    assert ' f"RUN ln -s /root/python3.11/bin/{REMOTE_EXECUTABLE_NAME}' in hunk
    assert "assert hashlib.sha256(b).hexdigest()=='{GBENCH_RUNTIME_LOCAL_SHA256}'" in hunk


def test_the_image_carries_the_submit_review_fix():
    """SWE-agent reads the model's edited files at submit with errors="replace" (a byte that is not
    UTF-8 ended privacyidea__privacyidea-4251 with no patch on 2026-10-08)."""
    import hashlib
    from gbench.runners.eval_suites import swe_bench_live as LV
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docker = open(os.path.join(root, "gbench/docker/sweagent_for_eval.Dockerfile")).read()
    patch = open(os.path.join(root, LV.SWEAGENT_AGENTS_PATCH)).read()
    assert "git apply /opt/sweagent-for-eval-review-decode.patch" in docker
    assert f"SWEAGENT_AGENTS_SHA256={LV.SWEAGENT_AGENTS_SHA256}" in docker
    assert 'org.gbench.sweagent_for_eval_agents="${SWEAGENT_AGENTS_SHA256}"' in docker
    assert 'return self._env.read_file(full_path, errors="replace")' in patch
    assert "+                        read_method=read_edited_file," in patch


def _read_edited_file_from_patch():
    """The read_edited_file the review-decode patch adds to
    DefaultAgent._get_edited_files_with_context, compiled from the patch's own + lines."""
    import textwrap
    from gbench.runners.eval_suites import swe_bench_live as LV
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lines = open(os.path.join(root, LV.SWEAGENT_AGENTS_PATCH), encoding="utf-8").read().splitlines()
    start = lines.index("+        def read_edited_file(path: str) -> str:")
    body = []
    for line in lines[start:]:
        if not line.startswith("+"):
            break
        body.append(line[1:])
    return textwrap.dedent("\n".join(body))


def test_the_submit_read_skips_a_path_that_is_not_a_regular_file(tmp_path):
    """2026-10-09, duckdb__ducklake-1224: the model's patch moved the submodule duckdb's commit,
    PatchFormatter read /testbed/duckdb (a directory), IsADirectoryError ended the instance with
    no patch. The patched read asks the task container (here: this host, running the patch's own
    command list) whether the path exists and is not a regular file, and then does not read it
    (its record entry is ""); a regular file is read with errors="replace"; a missing path is
    still read (and fails), as upstream."""
    import logging
    import types
    from pathlib import PurePosixPath
    repo = tmp_path / "testbed"
    (repo / "duckdb").mkdir(parents=True)
    (repo / "src.cpp").write_bytes(b"int x;\n\x81\n")
    os.mkfifo(repo / "fifo")
    os.symlink("duckdb", repo / "link_to_dir")
    os.symlink("src.cpp", repo / "link_to_file")
    reads, commands = [], []

    class RexCommand:
        def __init__(self, command, timeout=None):
            self.command, self.timeout = command, timeout

    class Runtime:
        async def execute(self, cmd):
            commands.append(cmd.command)
            assert isinstance(cmd.command, list) and cmd.timeout
            rc = subprocess.run(cmd.command, timeout=cmd.timeout).returncode
            return types.SimpleNamespace(exit_code=rc)

    def read_file(path, errors=None):
        reads.append((path, errors))
        with open(path, encoding="utf-8", errors=errors) as f:   # blocks on a FIFO, raises on a dir
            return f.read()

    env = types.SimpleNamespace(repo=types.SimpleNamespace(repo_name=str(repo).lstrip("/")),
                                deployment=types.SimpleNamespace(runtime=Runtime()),
                                read_file=read_file)
    agent = types.SimpleNamespace(_env=env, logger=logging.getLogger("t"))
    scope = {"asyncio": asyncio, "PurePosixPath": PurePosixPath, "RexCommand": RexCommand,
             "self": agent}
    exec(_read_edited_file_from_patch(), scope)
    read = scope["read_edited_file"]
    assert read("duckdb") == "" and read("fifo") == "" and read("link_to_dir") == ""
    assert reads == []
    assert read("src.cpp") == "int x;\n\ufffd\n"
    assert read("link_to_file") == "int x;\n\ufffd\n"
    assert reads == [(str(repo / "src.cpp"), "replace"), (str(repo / "link_to_file"), "replace")]
    with pytest.raises(FileNotFoundError):
        read("gone.py")
    assert len(commands) == 6 and all(c[:2] == ["sh", "-c"] for c in commands)
