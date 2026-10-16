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

"""The shared files follow the code.

* The quality relay counts its own 413 on a suite's model-endpoint call as a harness error.
* No unit test reaches the real cloud metadata server probe, and no test sees another test's
  cached verdict.
* README, toolchains and env-knobs name exactly the suites that call the metadata guard, each of
  those suites really calls it, and each suite's guide says so; the guard's DNS and IPv6 checks
  are documented with the code's exact rules.
* The Gemini key relay's documented interface is the code's.
* The quality overview, the SWE family's Windows tab and the knobs read as the code does, and
  the provenance records the network and guard changes.
"""

import inspect
import re
from pathlib import Path

import pytest

from gbench.runners import quality as Q
from gbench.runners.eval_suites import container_guard as cg

# The stub upstreams and the relay fixture of the quality isolation tests.
import test_quality_isolation as ISO
from test_quality_isolation import gemini, model_up, relay_to  # noqa: F401 - pytest fixtures

REPO = Path(__file__).resolve().parents[1]
SERVED = "google/gemma-4-E2B-it"


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
# the quality relay's own 413 on a suite call
# --------------------------------------------------------------------------- #
def test_an_oversized_suite_model_call_is_a_harness_error(relay_to, model_up, monkeypatch):
    monkeypatch.setattr(Q, "_MAX_REQUEST_BODY", 1024)
    relay = relay_to(model_upstream=model_up.url + "/v1", model_id=SERVED)
    status, data = ISO._relay_call(relay, "/model/chat/completions", body=b"x" * 2048)
    assert status == 413 and b"request body over 1024 bytes" in data
    assert model_up.requests == []
    rep = relay.report()
    assert rep["harness_errors"] == {"relay_body_too_large": 1}
    assert [kind for _t, kind in rep["harness_error_events"]] == ["relay_body_too_large"]
    assert rep["refused"] == []
    [row] = rep["model_endpoint_calls"]
    assert row["statuses"] == {"413 (relay)": 1}


def test_other_oversized_requests_stay_plain_refusals(relay_to, model_up, monkeypatch):
    monkeypatch.setattr(Q, "_MAX_REQUEST_BODY", 1024)
    relay = relay_to(model_upstream=model_up.url + "/v1", model_id=SERVED)
    big = b"x" * 2048
    assert ISO._relay_call(relay, "/model/metrics", body=big)[0] == 413         # not a suite call
    assert ISO._relay_call(relay, "/model/chat/completions", body=big,
                           token=False)[0] == 413                              # no token
    assert ISO._relay_call(relay, "/v1beta/models/gemini-3.1-pro-preview:generateContent",
                           body=big)[0] == 413                                  # Gemini route
    rep = relay.report()
    assert rep["harness_errors"] == {} and sum(r["requests"] for r in rep["refused"]) == 3
    no_model = relay_to()                               # no model endpoint configured
    assert ISO._relay_call(no_model, "/model/chat/completions", body=big)[0] == 413
    assert no_model.report()["harness_errors"] == {}


def test_the_413_rule_is_documented():
    doc = _flat(_read("docs/quality.md"))
    assert "it is the harness error `relay_body_too_large`" in doc
    assert "`relay_body_too_large` for a model-endpoint call the relay answered 413 itself" in doc


# --------------------------------------------------------------------------- #
# unit tests never reach the real probe, and never share a verdict
# --------------------------------------------------------------------------- #
def test_a_real_probe_from_a_unit_test_fails_at_once():
    with pytest.raises(pytest.fail.Exception, match="real cloud metadata server probe"):
        cg.require_metadata_blocked("x", "gbench-some-image:1", "bridge")
    assert cg.cached_verdicts() == {}


def test_the_task_network_gate_is_stubbed_outside_its_own_tests():
    from gbench.runners.eval_suites import swebench_common
    assert swebench_common.require_task_network_blocked("x", ["img"]) is None


def test_a_cached_verdict_is_left_for_the_next_test():
    from gbench.runners.eval_suites import tau_common
    cg._VERDICTS["bridge"] = cg.UNREACHABLE
    cg._RESOLVED.add(("bridge", "example.org"))
    tau_common._ENDPOINT_REACHABLE[("127.0.0.1", 1)] = True


def test_the_next_test_starts_without_it():
    from gbench.runners.eval_suites import tau_common
    assert cg.cached_verdicts() == {} and cg.cached_resolutions() == []
    assert tau_common._ENDPOINT_REACHABLE == {}


# --------------------------------------------------------------------------- #
# the suites that call the guard: named in the docs, calling it in code, saying so in their guide
# --------------------------------------------------------------------------- #
#: suite -> (module whose code calls the gate, a call it makes)
GUARDED = {
    "mcp_atlas": ("mcp_atlas", "container_guard.require_metadata_blocked("),
    "mcp_bench": ("mcp_bench", "container_guard.require_metadata_blocked("),
    "wildclawbench": ("wildclawbench", "container_guard.require_metadata_blocked("),
    "bfcl": ("bfcl", "require_endpoint_network("),
    "bfcl_v4_agentic": ("bfcl_v4_agentic", "require_endpoint_network("),
    "tau2": ("tau_common", "require_endpoint_network("),
    "tau3": ("tau_common", "require_endpoint_network("),
    "livebench": ("livebench", "require_endpoint_network("),
    "aider_polyglot": ("aider_polyglot", "require_endpoint_network("),
    "bigcodebench": ("bigcodebench", "require_metadata_blocked("),
    "copilot_bench_swe": ("swe_agentic", "require_rollout_network_blocked("),
    "swe_bench_multilingual": ("swe_agentic", "require_rollout_network_blocked("),
    "swe_bench_live": ("swe_bench_live", "require_task_network_blocked("),
    "multi_swe_bench": ("multi_swe_bench", "require_task_network_blocked("),
    "swe_bench_pro": ("swe_bench_pro", "require_task_network_blocked("),
    "terminal_bench": ("terminal_bench", "require_task_network_blocked("),
    "skillsbench": ("skillsbench", "require_task_network_blocked("),
    "swe_lancer": ("swe_lancer", "require_task_network_blocked("),
    "ui_control_osworld": ("ui_control_osworld", "require_task_network_blocked("),
}
#: suites whose model-code containers run with --network none
NO_NETWORK = {"ojbench": "ojbench", "multipl_e": "multipl_e", "putnam_formal": "putnam_formal"}


def _suite_src(module):
    return _read(f"gbench/runners/eval_suites/{module}.py")


@pytest.mark.parametrize("suite", sorted(GUARDED))
def test_every_suite_named_as_guarded_calls_the_guard(suite):
    module, call = GUARDED[suite]
    src = _suite_src(module)
    assert call in src, (suite, module, call)
    guide = _flat(_read(f"docs/evals/{suite}.md"))
    assert "metadata server" in guide, suite


def test_the_rollout_and_endpoint_gates_lead_to_the_guard():
    assert "swebench_common.require_task_network_blocked(" in _suite_src("swe_agentic")
    tau = _suite_src("tau_common")
    assert ("require_metadata_blocked(eval_name, image, ENDPOINT_NETWORK, docs_url, "
            "resolve=names)") in tau
    swe = _suite_src("swebench_common")
    assert "container_guard.require_metadata_blocked(eval_name, image, net, docs_url," in swe
    assert "require_metadata_blocked(\"quality\"" in _read("gbench/runners/quality.py")


@pytest.mark.parametrize("suite", sorted(NO_NETWORK))
def test_the_suites_named_network_less_run_with_network_none(suite):
    src = _suite_src(NO_NETWORK[suite])
    assert '"--network", "none"' in src, suite


def test_the_three_docs_name_the_guarded_suites():
    # the cloud-VM note moved verbatim from README.md into docs/installation.md
    install = _flat(_read("docs/installation.md"))
    note = install[install.index("Note for cloud VMs"):install.index("Conda Environment")]
    tools = _flat(_read("docs/evals/toolchains.md"))
    section = tools[tools.index("## Cloud VMs: the metadata server"):
                    tools.index("## Harnesses and gates")]
    knobs = _flat(_read("docs/evals/env-knobs.md"))
    guard = knobs[knobs.index("- Cloud metadata server guard"):
                  knobs.index("- `GBENCH_RESULTS_DIR`")]
    swe_family = {"copilot_bench_swe", "swe_bench_multilingual", "swe_bench_live",
                  "multi_swe_bench", "swe_bench_pro"}
    for suite in GUARDED:
        assert f"`{suite}`" in section, suite
        assert f"`{suite}`" in guard, suite
        if suite not in swe_family:
            assert f"`{suite}`" in note, suite
    assert "the SWE family" in note
    for suite in NO_NETWORK:
        assert f"`{suite}`" in section and f"`{suite}`" in guard, suite
    for doc in (note, section, guard):
        assert "each suite's guide says whether it does" not in doc
        assert "each suite's page says whether it does" not in doc
    assert "`quality`, whose container runs with `--network none`" in section


def test_the_dns_and_ipv6_checks_are_documented_with_the_code():
    sig = str(inspect.signature(cg.require_metadata_blocked))
    assert sig == ("(eval_name: 'str', image: 'str', network: 'Optional[str]', "
                   "docs_url: 'Optional[str]' = None, *, resolve: 'Iterable[str]' = ()) -> 'str'")
    tools = _flat(_read("docs/evals/toolchains.md"))
    knobs = _flat(_read("docs/evals/env-knobs.md"))
    # the README's install section (with its cloud-VM note) moved to docs/installation.md
    readme = _flat(_read("README.md") + "\n" + _read("docs/installation.md"))
    for doc in (tools, knobs):
        assert ("`require_metadata_blocked(eval_name, image, network, docs_url=None, *, "
                "resolve=())`") in doc
        assert cg.FIREWALL_COMMAND6 in doc
        assert "[fd20:ce::254]:80" in doc
        assert "cannot resolve host names" in doc
    assert "`[fd20:ce::254]:80`" in readme and "a name that does not resolve is a hard error" \
        in readme
    assert "`getent hosts` or `nslookup`" in tools
    assert cg.PLAIN_RULE_DELETE_COMMAND in tools
    # the message a DNS failure raises says what the docs say
    msg = cg.unresolved_message("bridge", {"example.org": "timeout"})
    assert "cannot resolve host names" in msg and cg.FIREWALL_COMMAND in msg
    assert "the VPC DNS resolver is 169.254.169.254" in msg
    for doc in (tools, knobs, readme):
        assert "OUTPUT -d 169.254.169.254" not in doc
    for rel in _README_AND_SPLIT_DOCS:   # README.md and every doc split out of it
        assert "OUTPUT -d 169.254.169.254" not in _flat(_read(rel)), rel


def test_toolchains_has_no_stray_punctuation():
    tools = _read("docs/evals/toolchains.md")
    assert ",;" not in tools
    assert "as upstream evals them; see below" in tools
    start = tools.index("## Cloud VMs: the metadata server")
    section = tools[start:tools.index("## Harnesses and gates")]
    assert max(len(line) for line in section.splitlines()) <= 100


# --------------------------------------------------------------------------- #
# the Gemini key relay's interface, as documented
# --------------------------------------------------------------------------- #
def test_the_relay_interface_is_documented_as_coded(monkeypatch):
    from gbench.runners.eval_suites import gemini_relay as GR
    from gbench.runners.eval_suites import search_tool
    doc = _flat(GR.__doc__)
    sig = inspect.signature(GR.GeminiRelay.__init__)
    assert list(sig.parameters)[1:3] == ["eval_name", "bind_host"]
    for name, param in list(sig.parameters.items())[3:]:
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert f"{name}=" in doc, name
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaW2oRelayDocKey00000000000000000001")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_OPENAI_BASE_URL", raising=False)
    env = GR.GeminiRelay("x", "127.0.0.1").client_env()
    for key in env:
        assert f'"{key}"' in GR.__doc__, key
    assert set(env) == {"GEMINI_API_KEY", "GEMINI_API_KEYS", "GEMINI_OPENAI_BASE_URL",
                        "GBENCH_GEMINI_RELAY_URL"}
    assert search_tool.RELAY_URL_ENV == GR.RELAY_URL_ENV == "GBENCH_GEMINI_RELAY_URL"
    assert search_tool.RELAY_PLACEHOLDER_PREFIX == GR.PLACEHOLDER_PREFIX
    knobs = _flat(_read("docs/evals/env-knobs.md"))
    assert "- `GBENCH_GEMINI_RELAY_URL`: not an operator setting." in knobs


# --------------------------------------------------------------------------- #
# README, env-knobs and the provenance read as the code does
# --------------------------------------------------------------------------- #
def test_the_readme_describes_the_current_quality_build_and_scoring():
    readme = _read("README.md")
    for rel in _README_AND_SPLIT_DOCS:   # README.md and every doc split out of it
        assert ".buildstamp" not in _read(rel), rel
    assert ("and plugin routing, run in gBench's local `gbench-quality` Docker image (built once "
            "per GemmaClaw commit).") in readme
    assert ("Harness errors (relay, endpoint, Gemini or container failures) are not scored; a "
            "model-endpoint 4xx other than 400 and 422 on the suite's own requests") in readme
    assert "withholds its pass rate (`null`, printed `withheld`), never 0." in readme
    assert '"withheld" if pass_rate is None' in _read("gbench/cli.py")


def test_the_readme_says_how_the_windows_tab_runs():
    from gbench.runners.eval_suites import swe_bench_live
    # the cloud-VM note moved to docs/installation.md, the suite catalog to docs/eval-suites.md
    readme = _flat("\n".join(_read(rel) for rel in
                              ("README.md", "docs/installation.md", "docs/eval-suites.md")))
    for rel in _README_AND_SPLIT_DOCS:   # README.md and every doc split out of it
        text = _flat(_read(rel))
        assert "cannot probe the Windows containers" not in text, rel
        assert "cannot probe Windows containers" not in text, rel
    assert ("Windows, which runs on a Windows Docker host only after its own cloud metadata server "
            "check (a PowerShell probe in one of its instance images) passes") in readme
    assert ("Windows (66; needs a Windows Docker host and a passing Windows metadata server check, "
            "reported `infra_required` in its own entry otherwise)") in readme
    assert not hasattr(swe_bench_live, "WINDOWS_METADATA_REASON")
    assert callable(swe_bench_live.check_windows_metadata_guard)
    assert swe_bench_live.WINDOWS_IMAGE_REFERENCE == "starryzhang/sweb.eval.win.*"
    for doc in ("docs/evals/toolchains.md", "docs/evals/env-knobs.md"):
        text = _flat(_read(doc))
        assert "Windows tab never runs" not in text, doc
        assert "check_windows_task_network" in text, doc
    tools = _flat(_read("docs/evals/toolchains.md"))
    assert "pulls the run's first task image" not in tools
    assert "nothing is pulled for it" in tools


def test_env_knobs_follow_the_suites():
    from gbench.runners.eval_suites import mcp_bench, skillsbench, wildclawbench
    knobs = _flat(_read("docs/evals/env-knobs.md"))
    assert "`GBENCH_MCP_BENCH_NETWORK` (the container's docker network, default `bridge`" in knobs
    assert mcp_bench._NETWORK_DEFAULT == "bridge"
    assert (f"`{wildclawbench.HARNESS_ERRORS_LABEL}={wildclawbench.HARNESS_ERRORS_VERSION}`"
            in knobs)
    assert "The task containers run on `GBENCH_WILDCLAWBENCH_TASK_NETWORK`, default `bridge`" \
        in knobs
    assert ("`GBENCH_SKILLSBENCH_API_KEY` (the served endpoint's key; unset, `dummy` or `EMPTY` "
            "means none) never enters a container or a command line") in knobs
    assert skillsbench.MODEL_KEY_PLACEHOLDER == "dummy"
    assert "a proxy variable on a loopback address is passed as `host.docker.internal`" in knobs


def test_the_provenance_records_the_network_and_guard_changes():
    from gbench.runners.eval_suites import wildclawbench
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as CS
    agentic = CS["bfcl_v4_agentic"]["method"]
    assert "host network, only the variables" not in agentic
    assert ("bfcl generate on docker's bridge network (never the host network), after the "
            "metadata-server guard") in agentic
    assert "the multi-turn part runs on docker's bridge network" in CS["bfcl"]["method"]
    for suite in ("tau2", "tau3"):
        assert "(the agent's api_base becomes host.docker.internal)" in CS[suite]["method"]
    assert "(--api-base becomes host.docker.internal)" in CS["livebench"]["method"]
    assert "(OPENAI_API_BASE becomes the relay URL" in CS["aider_polyglot"]["method"]
    assert "with the metadata-server guard checked before generation" in \
        CS["bigcodebench"]["method"]
    assert CS["ojbench"]["method"].endswith("; the judge container runs with --network none")
    assert "is a harness error left out of every rate" in CS["gaia2"]["method"]
    assert (f"org.gbench.wildclawbench-harness-errors={wildclawbench.HARNESS_ERRORS_VERSION}"
            in CS["wildclawbench"]["method"])
    assert "(GBENCH_MCP_BENCH_NETWORK; a model endpoint" in CS["mcp_bench"]["method"]
    assert "a run in which no model request got a reply publishes no score" in \
        CS["toolbench"]["method"]
    for suite in ("bfcl_v4_agentic", "bfcl", "tau2", "tau3", "livebench", "aider_polyglot",
                  "bigcodebench", "ojbench", "gaia2", "wildclawbench", "mcp_bench", "toolbench",
                  "mcp_atlas"):
        assert CS[suite]["synced"] >= "2026-09-26", suite


def test_the_changed_files_are_plain_ascii_without_internal_labels():
    rels = ("docs/evals/toolchains.md", "docs/evals/env-knobs.md", "docs/quality.md",
            "gbench/runners/eval_suites/container_guard.py",
            "gbench/runners/eval_suites/gemini_relay.py",
            "gbench/runners/eval_suites/search_tool.py", "tests/test_gemini_relay.py",
            "tests/test_container_guard.py", "tests/conftest.py")
    for rel in rels:
        text = _read(rel)
        bad = [i for i, c in enumerate(text) if ord(c) > 126 or (ord(c) < 32 and c not in "\n\t")]
        assert not bad, (rel, bad[:5])
        low = text.lower()
        # built from parts, so this file does not hold them
        for label in ("wave" + " 2", "wave" + "2", "the " + "reviewer", "integ" + "rator",
                      "fix" + "er"):
            assert label not in low, (rel, label)
    install = _read("docs/installation.md")
    note = install[install.index("> **Note for cloud VMs"):install.index("### Conda Environment")]
    assert all(ord(c) < 127 for c in note)
