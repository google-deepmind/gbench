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

"""wildclawbench: the agent's web_search timeout is 60 s, and the proxy's search deadline is 57 s.

OpenClaw 2026.3.11 in wildclawbench-ubuntu:v1.3 aborts a web_search after
tools.web.search.timeoutSeconds, which the image does not set, so it defaults to 30 s. That fits
Brave (about a second per search). It does not fit Gemini Google Search grounding: 2026-10-13 A/B,
70 recorded agent queries x 3 repeats in search-engine mode, answered in time 65.2% at width 2
with the 27 s deadline (61.9% at width 3, 60.5% at width 4) and 94.8% (Wilson 95% 90.9-97.1%) at
width 4 with a 57 s deadline (answered: median 20.2 s, p90 38.4 s). Decision: the initial width
stays 2, the launcher sets tools.web.search.timeoutSeconds to 60 in every task container and the
proxy's deadline is 60 - 3 = 57 s, from one constant
(wildclawbench_cascade_judge.AGENT_WEB_SEARCH_TIMEOUT_S, imported by the launcher). The deviation
from upstream's 30 s is recorded on the result (gemini_substitutions, leaderboard_comparable_reason).

Deterministic: no Docker, no Gemini, no network beyond local stub servers.
"""

import json
import os
import re
import subprocess
import sys
import types

import pytest

from gbench.runners.eval_suites import wildclawbench as W

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DOCKER_DIR = os.path.join(_REPO, "gbench", "docker")
sys.path.insert(0, _DOCKER_DIR)
import wildclawbench_cascade_judge as J    # noqa: E402
import wildclawbench_run as RUN            # noqa: E402


@pytest.fixture
def wd(monkeypatch, tmp_path):
    monkeypatch.setenv("WILDCLAW_WORKDIR", str(tmp_path))
    J.reset_usage()
    J._SEARCH_TOOL["module"] = None
    yield tmp_path
    J._SEARCH_TOOL["module"] = None


def _lines(path):
    return [json.loads(x) for x in open(path).read().splitlines()] if os.path.exists(path) else []


# --------------------------------------------------------------------------- #
# One source of truth: the timeout and the deadline cannot drift.
# --------------------------------------------------------------------------- #
def test_the_timeout_is_60_s_and_the_deadline_57_s():
    assert J.AGENT_WEB_SEARCH_TIMEOUT_S == 60 and isinstance(J.AGENT_WEB_SEARCH_TIMEOUT_S, int)
    assert J.UPSTREAM_WEB_SEARCH_TIMEOUT_S == 30
    assert J.WEB_SEARCH_DEADLINE_S == J.AGENT_WEB_SEARCH_TIMEOUT_S - 3 == 57


def test_the_launcher_takes_the_timeout_from_the_proxy():
    assert RUN.AGENT_WEB_SEARCH_TIMEOUT_S is J.AGENT_WEB_SEARCH_TIMEOUT_S
    assert RUN.UPSTREAM_WEB_SEARCH_TIMEOUT_S is J.UPSTREAM_WEB_SEARCH_TIMEOUT_S
    src = open(os.path.join(_DOCKER_DIR, "wildclawbench_run.py"), encoding="utf-8").read()
    assert not re.search(r"^\s*\w*WEB_SEARCH_TIMEOUT_S\s*=", src, re.M)
    assert "OPENCLAW_WEB_SEARCH_TIMEOUT_S" not in src
    proxy = open(os.path.join(_DOCKER_DIR, "wildclawbench_cascade_judge.py"),
                 encoding="utf-8").read()
    assert "OPENCLAW_WEB_SEARCH_TIMEOUT_S" not in proxy
    assert re.search(r"^WEB_SEARCH_DEADLINE_S = AGENT_WEB_SEARCH_TIMEOUT_S - 3\.0$", proxy, re.M)


def test_the_runner_copy_matches_the_image_files():
    assert W.AGENT_WEB_SEARCH_TIMEOUT_S == J.AGENT_WEB_SEARCH_TIMEOUT_S
    assert W.UPSTREAM_WEB_SEARCH_TIMEOUT_S == J.UPSTREAM_WEB_SEARCH_TIMEOUT_S
    assert W.WEB_SEARCH_DEADLINE_S == J.WEB_SEARCH_DEADLINE_S


def test_the_image_label_and_default_tag_are_bumped():
    assert W.HARNESS_ERRORS_VERSION == "9"
    assert W._IMAGE_DEFAULT == "gbench-wildclawbench:316334cc-v9"
    df = open(os.path.join(_DOCKER_DIR, "wildclawbench.Dockerfile"), encoding="utf-8").read()
    assert f'LABEL {W.HARNESS_ERRORS_LABEL}="9"' in df
    assert f"docker build -t {W._IMAGE_DEFAULT} " in df


# --------------------------------------------------------------------------- #
# The launcher writes the timeout into the task container's OpenClaw config.
# --------------------------------------------------------------------------- #
def _fake_openclaw(tmp_path, search_extra=None):
    pkg = tmp_path / "pkg" / "openclaw"
    (pkg / "dist").mkdir(parents=True)
    (pkg / "dist" / "reply.js").write_text(
        'const BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search";\n')
    (pkg / "openclaw.mjs").write_text("// entry\n")
    os.chmod(pkg / "openclaw.mjs", 0o755)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    os.symlink(pkg / "openclaw.mjs", bindir / "openclaw")
    search = {"enabled": True, "provider": "brave", "apiKey": "${BRAVE_API_KEY}",
              **(search_extra or {})}
    cfg = tmp_path / "openclaw.json"
    cfg.write_text(json.dumps({"agents": {"defaults": {"workspace": str(tmp_path / "ws")}},
                               "tools": {"web": {"search": search}}}))
    return bindir, cfg


def _run_setup(bindir, cfg):
    env = {**os.environ, "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"}
    argv = RUN._setup_args("http://127.0.0.1:18795")
    argv[3] = str(cfg)                       # OPENCLAW_CONFIG -> the fake config
    return subprocess.run([sys.executable, "-c", RUN._SEARCH_SETUP_SCRIPT, *argv],
                          capture_output=True, text=True, env=env)


def test_setup_script_sets_timeout_seconds_to_60(tmp_path):
    bindir, cfg = _fake_openclaw(tmp_path)
    r = _run_setup(bindir, cfg)
    assert r.returncode == 0, r.stderr
    search = json.loads(cfg.read_text())["tools"]["web"]["search"]
    # OpenClaw's config schema takes a positive integer (z.number().int().positive())
    assert search["timeoutSeconds"] == 60 and isinstance(search["timeoutSeconds"], int)
    out = json.loads(r.stdout)
    assert out["timeout_seconds"] == 60 and out["image_timeout_seconds"] is None


def test_setup_script_overrides_and_reports_a_timeout_the_image_sets(tmp_path):
    bindir, cfg = _fake_openclaw(tmp_path, {"timeoutSeconds": 30})
    r = _run_setup(bindir, cfg)
    assert r.returncode == 0, r.stderr
    assert json.loads(cfg.read_text())["tools"]["web"]["search"]["timeoutSeconds"] == 60
    assert json.loads(r.stdout)["image_timeout_seconds"] == 30


def test_setup_command_and_preflight_pass_the_timeout(monkeypatch):
    monkeypatch.setenv("WILDCLAW_SEARCH_ORIGIN", "http://127.0.0.1:18795")
    cmd = RUN.search_setup_command("task_c")
    assert cmd[-2] == "60" and cmd[-1] == ",".join(RUN.SEARCH_SKILLS_REMOVED)
    seen = []

    def run(c, **k):
        seen.append(c)
        return subprocess.CompletedProcess(c, 0, stdout="{}", stderr="")
    assert RUN.preflight_search_setup(run=run)[0]
    assert seen[0][-2] == "60"


def test_launcher_summary_records_the_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(RUN, "WORKDIR", str(tmp_path))
    monkeypatch.setenv("WILDCLAW_SEARCH_ORIGIN", "http://127.0.0.1:18795")
    monkeypatch.setenv("WILDCLAW_MODEL_RELAY_URL", "http://127.0.0.1:18795/model/v1")
    setup = {"patched_files": 4, "provider": "brave", "enabled": True,
             "removed_search_skills": [], "timeout_seconds": 60, "image_timeout_seconds": None}
    monkeypatch.setattr(RUN, "preflight_search_setup", lambda: (True, json.dumps(setup)))
    first = {"t": 1.0, "dns": {"github.com": "ok"}, "tcp": {"github.com": "ok"}}
    monkeypatch.setattr(RUN, "start_network_monitor", lambda: ("netmon", None))
    monkeypatch.setattr(RUN, "wait_first_round", lambda name: (first, None))
    monkeypatch.setattr(RUN, "stop_network_monitor", lambda name: None)
    monkeypatch.setattr(RUN, "_run_batch", lambda cfg, model: str(tmp_path / "output"))
    monkeypatch.setattr(RUN, "_task_index", lambda: {})
    monkeypatch.setattr(RUN, "_summarize", lambda out, idx: {
        "n_tasks": 1, "n_scored": 1, "global_mean": 1.0, "judge_fallback_tasks": 0})
    assert RUN.main() == 0
    summary = json.loads((tmp_path / "wildclawbench_summary.json").read_text())
    assert summary["agent_web_search_timeout"] == {
        "timeout_s": 60, "upstream_timeout_s": 30, "applied_timeout_s": 60,
        "image_timeout_s": None}


# --------------------------------------------------------------------------- #
# The proxy: the deadline passed to search_with_fallback, and a 45 s answer delivered in time.
# --------------------------------------------------------------------------- #
def _stub(seen):
    def search_with_fallback(query, max_results=8, deadline=None):
        seen.append(deadline)
        return [{"title": "T", "url": "https://e.org", "snippet": ""}], "gemini:gemini-3.8-flash"
    return types.SimpleNamespace(search_with_fallback=search_with_fallback)


def test_the_lookup_gets_a_57_s_deadline(wd):
    seen = []
    J._SEARCH_TOOL["module"] = _stub(seen)
    code, _ = J.brave_search_reply("q=x", "172.17.0.4", started=1000.0)
    assert code == 200 and seen == [1057.0]


def test_an_answer_at_45_s_is_answered_and_one_after_60_s_is_a_harness_event(wd):
    J._SEARCH_TOOL["module"] = _stub([])
    code, _ = J.brave_search_reply("q=x", "c", started=0.0)
    # 45 s: past OpenClaw's old 30 s default, inside the 60 s timeout -> the agent got it
    J.settle_web_search("c", 1.0, 0.0, code, True, now=45.0)
    assert [a["kind"] for a in _lines(wd / J.GEMINI_ANSWERS)] == ["agent_web_search"]
    assert _lines(wd / J.HARNESS_EVENTS) == []
    code, _ = J.brave_search_reply("q=x", "c", started=0.0)
    J.settle_web_search("c", 1.0, 0.0, code, True, now=61.0)
    [event] = _lines(wd / J.HARNESS_EVENTS)
    assert event["kind"] == "agent_web_search" and "past OpenClaw's 60 s" in event["detail"]


def test_aborted_calls_count_from_the_60_s_timeout(tmp_path):
    def msg(ts, m):
        return json.dumps({"type": "message", "timestamp": ts, "message": m})
    call = {"role": "assistant", "content": [{"type": "toolCall", "id": "a",
                                              "name": "web_search", "arguments": {}}]}
    call_b = {"role": "assistant", "content": [{"type": "toolCall", "id": "b",
                                                "name": "web_search", "arguments": {}}]}

    def result(cid):
        return {"role": "toolResult", "toolCallId": cid, "toolName": "web_search",
                "content": [{"type": "text", "text": RUN.WEB_SEARCH_ABORTED}], "isError": True}
    chat = tmp_path / "chat.jsonl"
    chat.write_text("\n".join([
        msg("2026-10-13T10:00:00.000Z", call), msg("2026-10-13T10:01:00.050Z", result("a")),
        msg("2026-10-13T10:02:00.000Z", call_b), msg("2026-10-13T10:02:30.050Z", result("b")),
    ]) + "\n")
    assert RUN.count_web_search_aborts(str(chat)) == 1          # 60 s counted, 30 s not


# --------------------------------------------------------------------------- #
# The result records the deviation.
# --------------------------------------------------------------------------- #
def _web_search_entry(subs):
    return next(s for s in subs if s["component"] == "OpenClaw agent web_search tool")


def test_the_substitution_records_the_timeout():
    entry = _web_search_entry(W.gemini_substitutions())
    assert entry["agent_web_search_timeout_s"] == 60
    assert entry["upstream_timeout_s"] == 30
    assert entry["proxy_search_deadline_s"] == 57
    assert entry["config_key"] == "tools.web.search.timeoutSeconds"
    for needle in ("2026-10-13", "65.2%", "94.8%", "90.9-97.1%", "Brave",
                   "width 2 with a 57 s deadline (the shipped setting) answered 94.3% (Wilson 95% "
                   "90.3-96.7%) in time with no generated text (2026-10-13 A/B), and 21 of 21 "
                   "agent searches in the 2026-10-14 wildclawbench run"):
        assert needle in entry["reason"]
    assert "applied_timeout_s" not in entry
    applied = _web_search_entry(W.gemini_substitutions(
        {"agent_web_search_timeout": {"applied_timeout_s": 60, "image_timeout_s": None}}))
    assert applied["applied_timeout_s"] == 60 and applied["image_timeout_s"] is None


def test_leaderboard_reason_names_the_timeout():
    reason = "; ".join(W._SUBSTITUTION_REASONS)
    assert "web_search timeout is raised from its 30 s default to 60 s" in reason
    assert "deadline 57 s" in reason
    # each figure names the width it was measured at, the shipped width 2 at 57 s included
    assert ("94.8% of searches answered in time at width 4 with a 57 s deadline, 65.2% at width 2 "
            "with 27 s; width 2 with a 57 s deadline (the shipped setting) answered 94.3% (Wilson "
            "95% 90.3-96.7%) in time with no generated text (2026-10-13 A/B), and 21 of 21 agent "
            "searches in the 2026-10-14 wildclawbench run)") in reason


def test_the_run_result_records_the_applied_timeout(monkeypatch, tmp_path):
    import test_wildclawbench_search_single_path as SP
    from gbench.runners.eval_suites import container_guard
    # docker is faked: the cloud metadata guard answers "unreachable" without a probe
    monkeypatch.setattr(container_guard, "require_metadata_blocked",
                        lambda *a, **k: container_guard.UNREACHABLE)
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", SP.FAKE_KEY)
    monkeypatch.setenv("WILDCLAW_WORKDIR", str(tmp_path / "proxy_workdir"))
    os.makedirs(tmp_path / "proxy_workdir", exist_ok=True)
    result, _cap = SP._run_runner(monkeypatch, tmp_path, summary_extra={
        "agent_web_search_timeout": {"timeout_s": 60, "upstream_timeout_s": 30,
                                     "applied_timeout_s": 60, "image_timeout_s": None}})
    entry = _web_search_entry(result["gemini_substitutions"])
    assert entry["agent_web_search_timeout_s"] == 60 and entry["applied_timeout_s"] == 60
    assert "raised from its 30 s default to 60 s" in result["leaderboard_comparable_reason"]
    assert result["leaderboard_comparable"] is False


def test_the_guides_describe_the_timeout():
    doc = open(os.path.join(_REPO, "docs", "evals", "wildclawbench.md"), encoding="utf-8").read()
    for needle in ("`AGENT_WEB_SEARCH_TIMEOUT_S`", "OpenClaw waits at most 60 s",
                   "`tools.web.search.timeoutSeconds` to 60", "94.8%", "deadline of 57 s",
                   "`org.gbench.wildclawbench-harness-errors=9`",
                   "gbench-wildclawbench:316334cc-v9"):
        assert needle in doc, needle
    knobs = open(os.path.join(_REPO, "docs", "evals", "env-knobs.md"), encoding="utf-8").read()
    assert "`org.gbench.wildclawbench-harness-errors=9`" in knobs
    assert "The agent's web_search timeout is 60 s" in knobs
    assert "94.8% of the agent's searches answered in time at width 4" in knobs
    # the deviation is listed in every summary section of the guide, not only the search section
    sections = {h: doc.split(h, 1)[1].split("\n## ", 1)[0] for h in (
        "## Other changes (everything else is upstream, verbatim)",
        "## Known limitations (each makes the result `leaderboard_comparable=False`)",
        "## Leaderboard comparability")}
    for heading, body in sections.items():
        assert "30 s" in body and "60 s" in body, heading
