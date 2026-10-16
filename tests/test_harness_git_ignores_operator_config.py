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

"""A benchmark harness that runs git on the host does not read the operator's git config.

On 2026-10-08 the operator's `url.ssh://git@github.com/.insteadOf https://github.com/` turned
multi_swe_bench's public HTTPS clone into an SSH clone that failed, so no report was written.
"""

import os
import subprocess

from gbench.runners.eval_suites import multi_swe_bench as M
from gbench.runners.eval_suites import swebench_common as C


def test_the_harness_env_ignores_the_operators_git_config(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    (home / ".gitconfig").write_text(
        '[url "ssh://git@example.invalid/"]\n\tinsteadOf = https://example.org/\n')
    base = {"HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin")}

    def rewritten(env):
        out = subprocess.run(["git", "ls-remote", "--get-url", "https://example.org/x.git"],
                             capture_output=True, text=True, env=env, timeout=30)
        return out.stdout.strip()
    assert rewritten(base) == "ssh://git@example.invalid/x.git"        # the operator's rule
    assert rewritten(C.isolated_git_env(base)) == "https://example.org/x.git"
    env = C.isolated_git_env(base)
    assert env["GIT_CONFIG_NOSYSTEM"] == "1" and env["GIT_TERMINAL_PROMPT"] == "0"


def test_multi_swe_bench_runs_its_harness_with_the_isolated_env():
    src = open(M.__file__, encoding="utf-8").read()
    assert "env=swebench_common.isolated_git_env()" in src


def test_git_config_passed_in_the_environment_is_dropped_too():
    env = C.isolated_git_env({"PATH": os.environ.get("PATH", "/usr/bin"), "GIT_CONFIG_COUNT": "1",
                              "GIT_CONFIG_KEY_0": "url.ssh://x/.insteadOf",
                              "GIT_CONFIG_VALUE_0": "https://example.org/",
                              "GIT_CONFIG_PARAMETERS": "'core.hooksPath'='/x'"})
    assert not any(k.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")) for k in env)
    assert "GIT_CONFIG_COUNT" not in env and "GIT_CONFIG_PARAMETERS" not in env


def test_the_gemmaclaw_ref_lookup_uses_the_isolated_env(monkeypatch):
    from gbench.core import scaffold
    seen = {}

    def fake_run(argv, **kw):
        seen.update(kw)
        return subprocess.CompletedProcess(argv, 0, "a" * 40 + "\trefs/heads/main\n", "")
    monkeypatch.setattr(scaffold.subprocess, "run", fake_run)
    getattr(scaffold.resolve_gemmaclaw_sha, "cache_clear", lambda: None)()
    scaffold.resolve_gemmaclaw_sha("main-branch-for-test")
    assert seen["env"]["GIT_CONFIG_GLOBAL"] == os.devnull
