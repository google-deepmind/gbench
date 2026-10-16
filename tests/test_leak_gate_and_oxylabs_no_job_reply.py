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

"""The leak gate over what a commit would track, and mcp_atlas's oxylabs "Error: 'job'" reply.

* Leak gate (scripts/check_no_leaks.sh): no file a commit of this tree would track (the tracked
  files plus the untracked ones .gitignore does not exclude) matches the gate's patterns, so a
  fake key in a test must be built at run time; the gate keeps its key patterns at full strength
  and allowlists nothing under tests/; run in a scratch repository, it still fails a key-shaped
  literal and passes the split form the tests use.
* mcp_atlas: oxylabs-mcp 0.4.1 reads the job's id before the HTTP status, so an account refusal
  whose JSON body carries no job record reaches the model as exactly "Error: 'job'". That reply
  is a provider account error (a harness error, named apart), in a run and in the pre-flight.
* The credentialed-servers containers of mcp_atlas and mcp_bench hold real third-party keys while
  their servers execute the model's tool calls; the docs state that residual risk.

Git runs only on this checkout (read-only) and on repositories under tmp; Docker, Gemini, the
providers and the network are stubbed.
"""

import asyncio
import json
import os
import re
import shutil
import subprocess

import pytest

from gbench.runners.eval_suites import mcp_atlas as M

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GATE = os.path.join(_REPO, "scripts", "check_no_leaks.sh")


def _read(rel):
    with open(os.path.join(_REPO, rel), encoding="utf-8") as fh:
        return fh.read()


def _flat(text):
    return " ".join(text.split())


# =========================================================================== #
# the leak gate
# =========================================================================== #
def _gate_pattern():
    """The gate's generic pattern (its `generic='...'` line), plus the private denylist when
    GBENCH_LEAK_PATTERNS is set, OR-ed in the way the script does it."""
    m = re.search(r"^generic='([^']+)'$", _read("scripts/check_no_leaks.sh"), re.M)
    assert m, "scripts/check_no_leaks.sh has no generic='...' line"
    extra = os.environ.get("GBENCH_LEAK_PATTERNS", "")
    return m.group(1) + ("|" + extra if extra else "")


def _gate_allowed():
    """The gate's allowed public addresses (its `allowed='...'` line), removed from a line
    before the line is checked."""
    m = re.search(r"^allowed='([^']+)'$", _read("scripts/check_no_leaks.sh"), re.M)
    assert m, "scripts/check_no_leaks.sh has no allowed='...' line"
    return m.group(1)


def _gate_hit(pattern, line):
    return re.search(pattern, re.sub(_gate_allowed(), "", line)) is not None


def _gate_excludes():
    return re.findall(r"'(:\(exclude\)[^']+)'", _read("scripts/check_no_leaks.sh"))


def _git(*args, cwd=_REPO):
    return subprocess.run(["git", "-c", f"safe.directory={cwd}", *args], cwd=cwd,
                          capture_output=True, text=True)


def _in_git_checkout():
    if shutil.which("git") is None:
        return False
    return _git("rev-parse", "--is-inside-work-tree").stdout.strip() == "true"


#: What a commit of the tree holds besides the top-level files: the fallback walk without git reads
#: only these, so ignored trees (results/, data/, the egg-info, a virtualenv) never count.
_SHIPPED_DIRS = ("gbench", "tests", "scripts", "docs", "docker", "examples", "tools", ".github")


def _walk_hits(pattern, repo=_REPO):
    """Without git (a source tree that is not a checkout): every text file of the shipped paths
    (the top-level files and _SHIPPED_DIRS) except backups, caches and the gate's own excludes,
    matched with Python's re (the gate's patterns use only syntax POSIX ERE and Python share)."""
    rx = re.compile(pattern)
    skip = {p.split(")", 1)[1] for p in _gate_excludes()}
    hits = []
    for root, dirs, files in os.walk(repo):
        rel_root = os.path.relpath(root, repo)
        if rel_root == ".":
            dirs[:] = [d for d in dirs if d in _SHIPPED_DIRS]
        dirs[:] = [d for d in dirs if not d.startswith(".") or d == ".github"]
        dirs[:] = [d for d in dirs if d not in ("__pycache__", "node_modules")
                   and os.path.normpath(os.path.join(rel_root, d)) not in skip]
        for name in files:
            rel = os.path.normpath(os.path.join(rel_root, name))
            if rel in skip or ".bak" in name or name.endswith((".orig", ".pyc")):
                continue
            try:
                with open(os.path.join(root, name), "rb") as fh:
                    data = fh.read()
            except OSError:
                continue
            if b"\0" in data[:8000]:
                continue
            for n, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
                if rx.search(line) and _gate_hit(pattern, line):
                    hits.append(f"{rel}:{n}:{line.strip()[:120]}")
    return hits


def test_the_gate_keeps_its_key_patterns_and_allowlists_nothing_under_tests():
    generic = _gate_pattern().split("|")
    for needle in ("@google\\.com", "-----BEGIN [A-Z ]*PRIVATE KEY-----", "AKIA[0-9A-Z]{16}",
                   "ghp_[0-9A-Za-z]{36}", "sk-[A-Za-z0-9]{20,}", "AIza[0-9A-Za-z_-]{35}"):
        assert needle in generic, needle
    # only the script itself (it names the patterns) and the optional private plugins tree
    assert _gate_excludes() == [":(exclude)scripts/check_no_leaks.sh", ":(exclude)gbench/plugins"]
    # one public address (Google's Code of Conduct contact), stripped from a line before the check
    assert _gate_allowed() == "opensource@google\\.com"
    assert not _gate_hit(_gate_pattern(), "please email opensource@google.com.")
    # built at run time, so this line does not match the gate itself
    assert _gate_hit(_gate_pattern(), "opensource@google.com or someone@google" + ".com")
    workflow = _read(".github/workflows/leak-gate.yml")
    assert "run: bash scripts/check_no_leaks.sh\n" in workflow


def test_no_file_a_commit_of_this_tree_would_track_matches_the_leak_gate():
    """The gate scans tracked files, so a key-shaped fixture in a new, untracked test passes it
    until the commit that adds the file, and then fails the Leak Gate check. This scans what
    `git add -A` would track (git grep --untracked honours .gitignore, so backups are out)."""
    pattern = _gate_pattern()
    if _in_git_checkout():
        proc = _git("grep", "-nIE", "--untracked", "-e", pattern, "--", *_gate_excludes())
        assert proc.returncode in (0, 1), proc.stderr
        hits = [h for h in (proc.stdout.splitlines() if proc.returncode == 0 else [])
                if _gate_hit(pattern, h)]
    else:
        hits = _walk_hits(pattern)
    assert not hits, (
        "these lines match scripts/check_no_leaks.sh's patterns, so the Leak Gate fails once "
        "they are committed; build a fake key at run time (e.g. \"AIza\" + \"...\"), never as "
        "one literal:\n" + "\n".join(hits))


def test_the_walk_without_git_reads_only_the_shipped_paths(tmp_path):
    key = "AIza" + "k" * 35
    for rel in ("tests/test_x.py", "README.md", "results/run/log.txt", "data/cache.json",
                "gbench.egg-info/PKG-INFO", "venv/lib/site.py", "tests/test_x.py.bak_1"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"key = {key}\n")
    hits = _walk_hits(_gate_pattern(), repo=str(tmp_path))
    assert sorted(h.split(":", 1)[0] for h in hits) == ["README.md", os.path.join("tests",
                                                                                   "test_x.py")]


def _scratch_repo(tmp_path, files):
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "GBENCH_LEAK"))}
    env.update(HOME=str(tmp_path), GIT_CONFIG_NOSYSTEM="1",
               GIT_AUTHOR_NAME="gbench tests", GIT_AUTHOR_EMAIL="tests@example.invalid",
               GIT_COMMITTER_NAME="gbench tests", GIT_COMMITTER_EMAIL="tests@example.invalid")
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    for cmd in (["init", "-q"], ["add", "-A"],
                ["-c", "commit.gpgsign=false", "commit", "-q", "-m", "fixture"]):
        proc = subprocess.run(["git", *cmd], cwd=repo, env=env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
    return repo, env


def _run_gate(repo, env):
    return subprocess.run(["bash", _GATE], cwd=repo, env=env, capture_output=True, text=True)


@pytest.mark.parametrize("prefix,body", [("AIza", "Q" * 35), ("sk-", "Q" * 24)])
def test_the_gate_fails_a_key_shaped_literal_and_passes_the_split_form(tmp_path, prefix, body):
    assert shutil.which("git"), "the leak gate needs git, as its workflow's runner has it"
    literal = f'KEY = "{prefix}{body}"\n'
    repo, env = _scratch_repo(tmp_path, {"tests/test_fixture.py": literal,
                                         "README.md": "clean\n"})
    proc = _run_gate(repo, env)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "tests/test_fixture.py:1:" in proc.stderr
    # the same value, built at run time the way the tests build their fake keys
    (repo / "tests" / "test_fixture.py").write_text(f'KEY = "{prefix}" + "{body}"\n')
    proc = _run_gate(repo, env)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# =========================================================================== #
# mcp_atlas: oxylabs' error reply with no job record
# =========================================================================== #
_JOB_REASON = "no job record in the error reply (Error: 'job')"


def test_the_no_job_reply_is_exactly_what_the_pinned_server_returns():
    # oxylabs_mcp/utils.py (0.4.1): response_json['job']['id'] raises KeyError('job') before
    # raise_for_status(); oxylabs_client turns it into MCPServerError(f"Error: {str(e)}").
    assert M.OXYLABS_NO_JOB_REPLY == "Error: " + str(KeyError("job"))


@pytest.mark.parametrize("tool", ["oxylabs_universal_scraper", "oxylabs_amazon_product_scraper",
                                  "oxylabs_google_search_scraper"])
def test_oxylabs_no_job_reply_is_an_account_error(tool):
    assert M.provider_account_error(tool, "Error: 'job'") == _JOB_REASON
    assert not M.provider_unreachable(tool, "Error: 'job'")


@pytest.mark.parametrize("tool,text", [
    # a page that quotes the reply is the page
    ("oxylabs_universal_scraper", "# Troubleshooting\nThe client printed Error: 'job'"),
    ("oxylabs_universal_scraper", "Error: 'job' "),
    # another KeyError of the same code path names another key
    ("oxylabs_universal_scraper", "Error: 'id'"),
    # another server's reply is never an Oxylabs account error
    ("exa_web_fetch_exa", "Error: 'job'"),
    ("github_list_commits", "Error: 'job'"),
])
def test_only_the_exact_oxylabs_reply_counts(tool, text):
    assert M.provider_account_error(tool, text) == ""


def _sandbox():
    return M._Sandbox("http://127.0.0.1:9", cred_container="cred",
                      cred_servers=M.CREDENTIALED_SERVERS)


def test_a_run_counts_and_names_the_no_job_reply_as_a_harness_error(monkeypatch):
    reply = (200, json.dumps([{"type": "text", "text": "Error: 'job'"}]))

    async def exec_http(container, method, path, timeout, body=None):
        return reply

    monkeypatch.setattr(M, "_exec_http_async", exec_http)
    sb = _sandbox()
    token = M._ACTIVE_SANDBOX.set(sb)
    try:
        out = asyncio.run(M.execute_mcp_tool("oxylabs_universal_scraper",
                                             {"url": "https://a.example/"}, 7))
    finally:
        M._ACTIVE_SANDBOX.reset(token)
    # the model still gets the server's reply, as upstream
    assert out == "Error: 'job'"
    assert sb.harness_error_samples == {7: 1}
    assert sb.provider_account_errors == {f"oxylabs_universal_scraper: {_JOB_REASON}": 1}
    assert sb.provider_unreachable == {}


def test_the_preflight_fails_an_oxylabs_check_answered_with_no_job_record(monkeypatch):
    async def exec_http(container, method, path, timeout, body=None):
        if body["tool_name"].startswith("exa_"):
            return 200, '[{"type":"text","text":"Title: r"}]'
        return 200, json.dumps([{"type": "text", "text": "Error: 'job'"}])

    monkeypatch.setattr(M, "_exec_http_async", exec_http)
    monkeypatch.setattr(M, "gemini_search_reply",
                        lambda *a, **k: (200, '[{"type":"text","text":"r"}]', []))
    res = asyncio.run(M._run_health_calls("http://127.0.0.1:1", "cred", ("exa", "oxylabs")))
    # the release's own pass rule already fails a text reply that starts with "Error:"
    ok, detail = res["oxylabs"]
    assert not ok and "Error: 'job'" in detail
    assert res["exa"][0]
    reason = M.preflight_failure_reason(res, local_env=True, enabled=list(M.RELEASE_SERVERS))
    assert "oxylabs" in reason


def test_the_docs_describe_the_no_job_reply_as_an_account_error():
    doc = _flat(_read("docs/evals/mcp_atlas.md"))
    assert f"`<tool>: {_JOB_REASON}`" in doc
    assert "it counts as an account error too" in doc
    assert "reaches the model as `Error: 'job'`: it does not say its status" not in doc
    src = _flat(_read("gbench/runners/eval_suites/mcp_atlas.py"))
    assert "its error reply with no job record (\"Error: 'job'\")" in src


# =========================================================================== #
# the credentialed-servers containers: the documented residual risk
# =========================================================================== #
@pytest.mark.parametrize("doc,count", [("docs/evals/mcp_atlas.md", "Its 15 servers"),
                                       ("docs/evals/mcp_bench.md", "Its five servers")])
def test_the_docs_state_the_credentialed_containers_residual_risk(doc, count):
    text = _flat(_read(doc))
    assert text.isascii()
    assert "**The residual risk of the credentialed-servers container.**" in text
    i = text.index("**The residual risk of the credentialed-servers container.**")
    para = text[i:i + 1600]
    for needle in (count, "hold their real", "execute the model's tool calls",
                   "as upstream designed them", "No model-written code runs in that container",
                   "The residual risk: a flaw in one of", "which has outbound internet access",
                   "no Gemini key", "the cloud metadata server checked unreachable"):
        assert needle in para, (doc, needle)


@pytest.mark.parametrize("src", ["gbench/runners/eval_suites/mcp_atlas.py",
                                 "gbench/runners/eval_suites/mcp_bench.py"])
def test_the_suite_docstrings_name_the_residual_risk(src):
    text = _flat(_read(src))
    assert "The residual risk, accepted as upstream designed it:" in text
    assert "is readable by model-controlled code" not in text
