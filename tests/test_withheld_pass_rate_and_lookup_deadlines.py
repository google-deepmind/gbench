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

"""The shared files follow the suites' code.

* A withheld quality pass_rate (null) is never read as a number: the CLI prints 'withheld'.
* The cloud metadata server guard is documented where an operator looks (README, toolchains,
  env-knobs) with the exact host rule the code names.
* GEMINI_OPENAI_BASE_URL: the docs and base's docstring name the paths that actually check it.
* The removed judge-endpoint knobs, the wildclawbench proxy and harness errors, gaia2's knobs and
  tau2/tau3's agent-turn errors read as the code does, and the provenance records them.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _read(rel):
    return (REPO / rel).read_text(encoding="utf-8")


def _flat(text):
    """Whitespace collapsed, with Markdown blockquote markers at line starts removed."""
    return " ".join(re.sub(r"(?m)^\s*>\s?", "", text).split())


# The README was split: these docs hold sections moved verbatim out of the README, so a
# negative check that used to scan the whole README scans README.md plus all of them.
_README_SPLIT_DOCS = (
    "docs/installation.md", "docs/environment-variables.md", "docs/performance.md",
    "docs/eval-suites.md", "docs/golden-set.md", "docs/remote-endpoints.md",
    "docs/results.md", "docs/deployment.md", "docs/cli-reference.md",
)
_README_AND_SPLIT_DOCS = ("README.md", *_README_SPLIT_DOCS)


# --------------------------------------------------------------------------- #
# a withheld quality pass_rate
# --------------------------------------------------------------------------- #
def test_the_cli_prints_withheld_for_a_null_pass_rate(capsys):
    from gbench.cli import print_results_summary
    from gbench.core.config import BenchmarkConfig
    base = {"benchmark_type": "quality", "model_short": "m", "format": "remote",
            "gemmaclaw_commit": "d8bc6989133e", "passed_scenarios": 0, "total_scenarios": 0}
    print_results_summary([dict(base, pass_rate=None), dict(base, pass_rate=62.5)],
                          BenchmarkConfig())
    out = capsys.readouterr().out
    assert "withheld" in out and "62.5%" in out


# --------------------------------------------------------------------------- #
# the cloud metadata server guard, documented with the code's exact rule
# --------------------------------------------------------------------------- #
def test_the_guard_is_documented_with_the_exact_host_rule():
    from gbench.runners.eval_suites import container_guard as cg
    # the GCE VPC DNS resolver is the same address: the rule must spare port 53
    assert cg.FIREWALL_COMMAND == ("sudo iptables -I DOCKER-USER -d 169.254.169.254/32 "
                                   "-p tcp ! --dport 53 -j DROP")
    # the cloud-VM note moved verbatim from the README into docs/installation.md
    readme = _read("docs/installation.md")
    tools = _read("docs/evals/toolchains.md")
    knobs = _read("docs/evals/env-knobs.md")
    for doc in (readme, tools, knobs):
        flat = _flat(doc)
        assert cg.FIREWALL_COMMAND in flat
        assert cg.PERSIST_COMMAND in flat
        assert "`require_metadata_blocked" in flat
        assert "gBench never runs sudo or changes the host's firewall itself" in flat
        assert "`--network none` needs no probe" in flat
    assert "Note for cloud VMs (GCE and other hosts with a metadata server)" in readme
    assert "#cloud-vms-the-metadata-server" in readme
    assert "## Cloud VMs: the metadata server" in tools
    # no doc recommends dropping the address for the whole VM (it cuts the VM's DNS)
    for doc in (readme, tools, knobs):
        assert "OUTPUT -d 169.254.169.254" not in _flat(doc)
        assert "VPC DNS resolver" in _flat(doc)
    for rel in _README_AND_SPLIT_DOCS:   # README.md and every doc split out of it
        assert "OUTPUT -d 169.254.169.254" not in _flat(_read(rel)), rel
    guard = _flat(knobs[knobs.index("- Cloud metadata server guard"):
                        knobs.index("- `GBENCH_RESULTS_DIR`")])
    assert "no knob turns it off" in guard and "`python3`, else its `bash`" in guard
    assert f"{cg.METADATA_HOST}:{cg.METADATA_PORT}" in guard


def test_the_guard_docs_name_what_the_probe_program_is():
    from gbench.runners.eval_suites import container_guard as cg
    assert "python3" in cg.python_probe_script() or "socket" in cg.python_probe_script()
    tools = _flat(_read("docs/evals/toolchains.md"))
    assert "with the image's `python3`, else its `bash` (`/dev/tcp`) under coreutils `timeout`" \
        in tools
    assert "`--pull never`, `--read-only`, `--cap-drop ALL`" in tools


# --------------------------------------------------------------------------- #
# GEMINI_OPENAI_BASE_URL: the docs name the paths that check it
# --------------------------------------------------------------------------- #
def test_the_base_url_docs_name_the_paths_that_check_it(monkeypatch):
    from gbench.runners.eval_suites import base, tau_common
    knobs = _flat(_read("docs/evals/env-knobs.md"))
    assert ("a suite that hands the variable to a container or proxy hard-errors (infra_required) "
            "whatever a skip-key-validation knob says (gaia2, wildclawbench, tau2 and tau3)"
            ) in knobs
    assert "Every host-side judge or proxy path reads it through one helper" not in knobs
    assert ("tau2/tau3's image never gets it: gbench's Gemini relay on the host reads it through "
            "the same helper") in knobs
    doc = _flat(base.gemini_openai_base_url.__doc__)
    assert "The one reader of GEMINI_OPENAI_BASE_URL in os.environ" in doc
    assert "_credential_map" not in doc
    # what that sentence says of tau_common: the variable never enters the image
    for url in ("https://attacker.example/v1beta/openai/", "http://127.0.0.1:9/v1beta/openai/"):
        host = {"GEMINI_API_KEY": "AIzaW2nFakeKey000000000000000000000001",
                "GEMINI_OPENAI_BASE_URL": url}
        assert "GEMINI_OPENAI_BASE_URL" not in tau_common._launch_env(host)
        assert "GEMINI_OPENAI_BASE_URL" not in tau_common._forwarded_env(host)


def test_gaia2_and_tau_check_the_base_url_whatever_the_skip_knob_says():
    gaia2 = _read("gbench/runners/eval_suites/gaia2.py")
    run = gaia2[gaia2.index("def run_gaia2"):] if "def run_gaia2" in gaia2 else gaia2
    assert 'require_gemini_openai_base_url("gaia2", DOCS_URL)' in run
    tau = _read("gbench/runners/eval_suites/tau_common.py")
    assert "require_gemini_openai_base_url(eval_name, docs_url)" in tau


# --------------------------------------------------------------------------- #
# removed knobs, wildclawbench, gaia2, tau
# --------------------------------------------------------------------------- #
def test_env_knobs_follow_the_removed_knobs_and_the_suites():
    from gbench.runners.eval_suites import gaia2, wildclawbench
    knobs = _read("docs/evals/env-knobs.md")
    wc = _flat(knobs[knobs.index("- wildclawbench:"):knobs.index("- ui_control_osworld")])
    listed = wc[:wc.index("The Gemini keys")]
    for knob in wildclawbench._REMOVED_KNOBS:
        assert f"`{knob}`" not in listed
        assert f"Removed: `{knob}` (setting it logs a warning)" in wc
    assert "only its entrypoint and proxy hold them" in wc
    assert ("`GEMINI_OPENAI_BASE_URL` is passed by name, only as "
            "`base.require_gemini_openai_base_url()` accepts it") in wc
    assert (f"`{wildclawbench.HARNESS_ERRORS_LABEL}={wildclawbench.HARNESS_ERRORS_VERSION}`"
            in wc)
    assert wildclawbench._BRIDGE_GATEWAY_DEFAULT == "172.17.0.1" and "`172.17.0.1`" in wc
    g2 = _flat(knobs[knobs.index("- gaia2:"):knobs.index("- gdpval:")])
    listed = g2[:g2.index("Removed")]
    for knob in gaia2._REMOVED_KNOBS:
        assert f"`{knob}`" not in listed and f"`{knob}`" in g2
    assert "`GBENCH_GAIA2_KEEP_WORKDIR`" in listed
    assert "GBENCH_GAIA2_KEEP_WORKDIR" in _read("gbench/runners/eval_suites/gaia2.py")
    tau = _flat(knobs[knobs.index("- tau2 and tau3 (shared"):knobs.index("- wildclawbench:")])
    # the variable never enters the tau container: it is the Gemini relay's upstream
    assert "`GEMINI_OPENAI_BASE_URL` (accepted by `base.check_gemini_openai_base_url()`; " \
           "any other value hard-errors before the container starts) is the relay's " \
           "upstream and never goes in" in tau


def test_env_knobs_keep_to_100_columns_in_the_edited_sections():
    knobs = _read("docs/evals/env-knobs.md")
    start = knobs.index("\n", knobs.index("### Web search cascade"))  # the bullets, not the title
    section = knobs[start:knobs.index("## Per suite knobs")]
    assert all(len(line) <= 100 for line in section.splitlines())
    guard = knobs[knobs.index("- Cloud metadata server guard"):
                  knobs.index("- `GBENCH_RESULTS_DIR`")]
    assert all(len(line) <= 100 for line in guard.splitlines())


def test_resources_states_the_gaia2_judge_substitution():
    res = _read("docs/RESOURCES.md")
    assert ("hard-errors until provisioned; ARE's LLM judge runs on the gbench Gemini cascade in "
            "place of Llama-3.3-70B (recorded, not leaderboard-comparable)") in res


def test_readme_quality_lines_follow_the_container():
    # the GBENCH_SANDBOX row moved verbatim from the README into docs/environment-variables.md
    readme = _flat(_read("README.md") + "\n" + _read("docs/environment-variables.md"))
    for rel in _README_AND_SPLIT_DOCS:   # README.md and every doc split out of it
        assert "via a recorded edit of the checkout" not in _flat(_read(rel)), rel
    assert "via a recorded edit applied in the image's build context" in readme
    assert ("always runs in gBench's own Docker container (no network, read-only root, the suite "
            "run as a user without capabilities)") in readme
    quality = _read("gbench/runners/quality.py")
    assert '"--cap-drop", "ALL", "--cap-add", "SETUID", "--cap-add", "SETGID"' in quality
    assert "--inh-caps=-all" in _read("docker/quality_entrypoint.sh")


# --------------------------------------------------------------------------- #
# the provenance
# --------------------------------------------------------------------------- #
def test_provenance_records_the_lookup_deadlines_and_withheld_scores():
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as CS
    atlas = CS["mcp_atlas"]["method"]
    assert "each page lookup has its own 300 s deadline in place of the harness's 60 s tool-call " \
           "timeout" not in atlas
    assert "cache and 60 s timeout (search tools)" in atlas
    assert "a Gemini call still running when the task's wall clock stops it" in atlas
    assert "cache and 60 s timeout, recorded in search_substitution" not in atlas
    gaia2 = CS["gaia2"]["method"]
    assert "(GBENCH_GAIA2_JUDGE_ENDPOINT / GBENCH_GAIA2_JUDGE_MODEL removed)" in gaia2
    assert "is a harness error left out of every rate" in gaia2
    wc = CS["wildclawbench"]["method"]
    assert "OpenClaw ports 18789-18793 and 18800-18899 refused" in wc
    assert "the model relay never follows redirects" in wc
    assert "copy of base.check_gemini_openai_base_url" in CS["mcp_bench"]["method"]
    tau2 = CS["tau2"]["method"]
    assert "(agent_turn_errors), excluded from pass^k as upstream does" in tau2
    assert "a simulation that ends in an infrastructure error for any cause" in tau2
    tau3 = CS["tau3"]["method"]
    assert "raised on the agent's own turn it is listed apart (agent_turn_errors)" in tau3
    assert "a harness error as in tau2" in tau3
    cfb = CS["complexfuncbench"]["method"]
    assert "harness errors are never the model's outcome" in cfb
    assert "unknown_error); leaderboard_comparable=False; --shard is applied" in cfb
    assert cfb.endswith("an empty shard gives accuracy None; attempts not_applied")
    for suite in ("gaia2", "wildclawbench", "mcp_bench", "mcp_atlas", "tau2", "tau3",
                  "complexfuncbench"):
        assert CS[suite]["synced"] >= "2026-09-26", suite


def test_the_changed_shared_text_is_plain_ascii():
    for rel in ("docs/installation.md", "docs/evals/env-knobs.md", "docs/evals/toolchains.md",
                "docs/RESOURCES.md", "gbench/runners/eval_suites/container_guard.py"):
        text = _read(rel)
        bad = [i for i, c in enumerate(text) if ord(c) > 126 or (ord(c) < 32 and c not in "\n\t")]
        # the installation doc (moved verbatim from the README) carries a few pre-existing
        # non-ASCII characters; the new sections must not
        if rel == "docs/installation.md":
            start = text.index("> **Note for cloud VMs")
            section = text[start:text.index("### Conda Environment Installation")]
            assert all(ord(c) < 127 for c in section)
        else:
            assert not bad, (rel, bad[:5])


# --------------------------------------------------------------------------- #
# the grounding search deadline covers the last attempt and the rate limiter
# --------------------------------------------------------------------------- #
@pytest.fixture
def st(monkeypatch):
    from gbench.runners.eval_suites import search_tool
    monkeypatch.setattr(search_tool, "_RATE_LIMITER", search_tool._RateLimiter(0, 1))
    monkeypatch.setattr(search_tool, "_KEY_CYCLE", None)
    monkeypatch.setattr(search_tool, "_SEARCH_ATTEMPTS", 1)
    monkeypatch.setenv("GBENCH_SEARCH_CASCADE_ROUNDS", "1")
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "m1")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    # Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
    monkeypatch.setenv("GEMINI_API_KEY", "AIza" + "W2nSearchFakeKey00000000000000000001")
    search_tool.reset_backend_tally()
    yield search_tool
    search_tool.reset_backend_tally()


def _virtual_clock(st, monkeypatch, start=100.0):
    import types
    clock = [start]

    def sleep(seconds):
        clock[0] += seconds
    monkeypatch.setattr(st, "time", types.SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))
    return clock


def test_time_running_out_in_the_last_attempt_is_a_deadline_failure(st, monkeypatch):
    """One model, one round, one attempt: the request uses up its capped timeout and times out.
    The lookup is 'deadline-passed', not 'all-grounding-models-failed'."""
    import socket as _socket
    import urllib.request
    clock = _virtual_clock(st, monkeypatch)

    def slow(req, timeout=None):
        clock[0] += timeout
        raise _socket.timeout("timed out")
    monkeypatch.setattr(urllib.request, "urlopen", slow)
    results, backend = st.search_with_fallback("q", deadline=100.5)
    assert backend == "deadline-passed" and results[0]["title"] == "error"
    assert st.backend_tally() == {"deadline-passed": 1}


def test_a_failure_with_time_left_is_still_all_models_failed(st, monkeypatch):
    import urllib.error
    import urllib.request
    _virtual_clock(st, monkeypatch)

    def throttled(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 429, "quota", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", throttled)
    _results, backend = st.search_with_fallback("q", deadline=200.0)
    assert backend == "all-grounding-models-failed"
    _results, backend = st.search_with_fallback("q")
    assert backend == "all-grounding-models-failed"


def test_the_rate_limiter_wait_stops_at_the_deadline(st):
    clock = [0.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        clock[0] += seconds
    rl = st._RateLimiter(1.0, 1.0, clock=lambda: clock[0], sleep=sleep)
    assert rl.acquire() is True                    # the burst token
    assert rl.acquire(deadline=0.5) is False       # the next frees at 1.0: past the deadline
    assert slept == [] and clock[0] == 0.0
    assert rl.acquire(deadline=5.0) is True        # waits 1.0 s for it
    assert slept == [pytest.approx(1.0)]
    assert st._RateLimiter(0, 1).acquire(deadline=-1.0) is True   # disabled: never waits


def test_a_lookup_that_cannot_get_a_token_before_the_deadline_sends_nothing(st, monkeypatch):
    import urllib.request
    clock = _virtual_clock(st, monkeypatch)
    rl = st._RateLimiter(1.0, 1.0, clock=lambda: clock[0], sleep=lambda s: None)
    rl.tokens = 0.0
    monkeypatch.setattr(st, "_RATE_LIMITER", rl)

    def no_request(*a, **k):
        raise AssertionError("a request was sent without a token")
    monkeypatch.setattr(urllib.request, "urlopen", no_request)
    got = st.gemini_search("q", model="m1", deadline=100.5)
    assert got[0]["title"] == "error" and "deadline" in got[0]["snippet"]
    _results, backend = st.search_with_fallback("q", deadline=100.5)
    assert backend == "deadline-passed"
