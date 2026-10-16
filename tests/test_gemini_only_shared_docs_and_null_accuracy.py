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

"""The shared report, key-check, search and provenance code, and the shared docs, in step with
the Gemini-only rule and the per-suite changes they describe.

* The printed summary shows a null accuracy (a suite whose headline was not scored because a
  category or metric is infra_required) as '-' instead of crashing.
* The Gemini key's live check never sends the key to a host other than the Gemini API or a
  loopback stub.
* The grounding search takes GEMINI_API_KEYS as well as GEMINI_API_KEY.
* The shared docs name no SerpAPI, Brave, RapidAPI or OpenAI embedding requirement, and the
  provenance records every Gemini substitution, the LiveBench patch, ComplexFuncBench's dropped
  live comparison and NESTFUL's single data source.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# The README was split: these docs hold sections moved verbatim out of the README.
_README_SPLIT_DOCS = (
    "docs/installation.md", "docs/environment-variables.md", "docs/performance.md",
    "docs/eval-suites.md", "docs/golden-set.md", "docs/remote-endpoints.md",
    "docs/results.md", "docs/deployment.md", "docs/cli-reference.md",
)


def _readme_with(*docs: str) -> str:
    """README.md followed by the moved-out doc(s) that now hold the asserted text."""
    return "\n".join(_read(rel) for rel in ("README.md", *docs))


# --------------------------------------------------------------------------- #
# the printed summary with a null accuracy
# --------------------------------------------------------------------------- #
def test_pct_cell_formats_numbers_and_a_null_accuracy():
    from gbench import cli
    assert cli._pct_cell(12.3456) == "    12.35%"
    assert cli._pct_cell(0) == "     0.00%"
    assert cli._pct_cell(None) == "        - "
    assert len(cli._pct_cell(None)) == len(cli._pct_cell(99.0)) == 10


def _eval_result(name, accuracy, category_accuracy=None):
    r = {"benchmark_type": "eval", "eval_name": name, "model_short": "m", "format": "remote",
         "thinking": False, "total_questions": 10, "correct_answers": 4, "accuracy": accuracy,
         "status": "completed_with_errors"}
    if category_accuracy is not None:
        r["category_accuracy"] = category_accuracy
    return r


def test_the_printed_summary_shows_a_null_accuracy_as_a_dash(tmp_path, capsys):
    """acebench without a Gemini key: accuracy null and an agent_multi_turn row with a null
    accuracy; omnidocbench's documented null composite; a custom suite with a null accuracy."""
    from gbench import cli
    ace = _eval_result("acebench", None, {
        "normal_atom_bool": {"correct": 4, "total": 5, "accuracy": 80.0},
        "agent_multi_turn": {"status": "infra_required", "accuracy": None, "correct": 0,
                             "total": 0}})
    results = [ace, _eval_result("omnidocbench", None),
               _eval_result("my_custom_jsonl_suite", None,
                            {"a": {"correct": 1, "total": 2, "accuracy": None},
                             "b": {"correct": 1, "total": 2, "accuracy": 50.0}})]
    cli.print_results_summary(results, SimpleNamespace(results_dir=tmp_path))
    out = capsys.readouterr().out
    row = next(l for l in out.splitlines() if "ACEBENCH" in l)
    assert row.rstrip().endswith("-") and "None" not in row
    cat = next(l for l in out.splitlines() if "agent_multi_turn" in l)
    assert cat.rstrip().endswith("-")
    assert re.search(r"normal_atom_bool\s+5\s+4\s+80\.00%", out)
    assert "OMNIDOCBENCH" in out and "MY_CUSTOM_JSONL_SUITE" in out
    csv = cli._save_eval_summary_csv(tmp_path, results, [], [("5. Tool Use", ["acebench"])], {})
    assert csv is not None and "ACEBENCH" in csv.read_text()


# --------------------------------------------------------------------------- #
# the Gemini key's live check
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value,accepted", [
    (None, True),
    ("https://generativelanguage.googleapis.com/v1beta/openai/", True),
    ("http://127.0.0.1:5555/v1", True),
    ("http://localhost:5555/v1", True),
    ("http://[::1]:5555/v1", True),
    ("https://api.openai.com/v1", False),
    ("http://generativelanguage.googleapis.com/v1beta/openai/", False),
    ("https://generativelanguage.googleapis.com.evil.example/v1", False),
    ("https://user:pw@example.com/v1", False),
    ("not a url", False),
])
def test_the_live_check_endpoint_is_the_gemini_api_or_loopback(monkeypatch, value, accepted):
    from gbench.runners.eval_suites import base
    if value is None:
        monkeypatch.delenv("GEMINI_OPENAI_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", value)
    url, why = base.gemini_openai_base_url()
    if accepted:
        assert url and why == ""
        assert url == (value or base.GEMINI_OPENAI_BASE_URL).rstrip("/")
    else:
        assert url is None and "GEMINI_OPENAI_BASE_URL" in why
        assert "pw" not in why and "example" not in why


def test_a_foreign_live_check_endpoint_never_receives_the_key(monkeypatch):
    import urllib.request
    from gbench.runners.eval_suites import base

    def no_request(*a, **k):
        raise AssertionError("the key was sent")
    monkeypatch.setattr(urllib.request, "urlopen", no_request)
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", "https://api.openai.com/v1")
    ok, why = base.gemini_key_live_valid("AIzaFAKEKEY000000000001,AIzaFAKEKEY000000000002")
    assert ok is False and "not the Gemini API" in why and "AIzaFAKE" not in why
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaFAKEKEY000000000001")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    with pytest.raises(RuntimeError, match=r"\[INFRA REQUIRED\] simpleqa") as e:
        base.gemini_required_skip("simpleqa", "m")
    assert "GEMINI_OPENAI_BASE_URL" in str(e.value) and "AIzaFAKE" not in str(e.value)


def test_a_loopback_live_check_endpoint_is_pinged(monkeypatch):
    import io
    import urllib.request
    from gbench.runners.eval_suites import base
    seen = []

    class _Resp(io.BytesIO):
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=None):
        seen.append((req.full_url, req.get_header("Authorization")))
        return _Resp(b"{}")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", "http://127.0.0.1:5555/v1/")
    base.reset_gemini_model_checks()
    assert base.gemini_key_live_valid("AIzaFAKEKEY000000000001") == (True, "")
    # the key ping, then the model check of every cascade model, all to the loopback endpoint
    assert seen[0] == ("http://127.0.0.1:5555/v1/models", "Bearer AIzaFAKEKEY000000000001")
    assert seen[1:] == [(f"http://127.0.0.1:5555/v1/models/{m}", "Bearer AIzaFAKEKEY000000000001")
                        for m, _knob in base.gemini_cascade_models()]


def test_serpapi_is_no_longer_a_named_secret_but_is_still_masked():
    from gbench.runners.eval_suites import base
    assert "SERPAPI_API_KEY" not in base.SECRET_ENV_NAMES
    assert base.is_secret_env_name("SERPAPI_API_KEY")
    for name in ("GBENCH_TOOLBENCH_SIMULATOR_KEY", "TB_SIMULATOR_KEY"):
        assert name in base.SECRET_ENV_NAMES


def test_the_reasoning_settings_comment_cites_no_simple_evals_o_series():
    src = _read("gbench/runners/eval_suites/base.py")
    block = src[src.index("    #: The upstream's explicit reasoning-model settings"):]
    block = block[:block.index("reasoning: Optional")]
    assert "o-series" not in block and "OJBench" in block


# --------------------------------------------------------------------------- #
# the grounding search reads GEMINI_API_KEYS
# --------------------------------------------------------------------------- #
def test_the_search_takes_a_key_list_set_only_as_gemini_api_keys(monkeypatch):
    from gbench.runners.eval_suites import search_tool
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEYS", "k1;k2")
    monkeypatch.setattr(search_tool, "_KEY_CYCLE", None)
    assert search_tool.search_available()
    assert search_tool.search_backend_name().startswith("gemini-grounding-cascade[")
    assert {search_tool.get_gemini_api_key() for _ in range(4)} == {"k1", "k2"}
    monkeypatch.delenv("GEMINI_API_KEYS")
    assert not search_tool.search_available() and search_tool.search_backend_name() is None
    results, backend = search_tool.search_with_fallback("q")
    assert backend == "none" and results[0]["title"] == "error"


def test_the_search_docstring_describes_the_gemini_served_suites():
    from gbench.runners.eval_suites import search_tool
    doc = search_tool.__doc__
    assert "SerpAPI" not in doc
    assert "brave-search, ddg-search, exa, oxylabs" in doc and "bfcl_v4_agentic" in doc


# --------------------------------------------------------------------------- #
# the shared docs follow the code
# --------------------------------------------------------------------------- #
def test_no_shared_doc_asks_for_another_providers_key():
    for rel in ("README.md", *_README_SPLIT_DOCS, "docs/evals/env-knobs.md",
                "docs/evals/toolchains.md", "docs/RESOURCES.md", "docs/THIRD_PARTY_DATASETS.md"):
        text = _read(rel)
        # `RAPID_API_KEY` is no longer here: complexfuncbench's response_based tier is a
        # non-model tool backend that runs as upstream, with upstream's key (Gemini scope).
        # mcp_atlas' exa and oxylabs page tools run as upstream, with EXA_API_KEY and the
        # Oxylabs credentials (non-model tool backends), so those names are listed now.
        for gone in ("`SERPAPI_API_KEY`", "serpapi.com", "| `GBENCH_TAU2_OPENAI_API_KEY`",
                     "needs `GBENCH_TAU2_OPENAI_API_KEY`", "GBENCH_TOOLBENCH_SIMULATOR_BASE",
                     "GBENCH_TOOLBENCH_SIMULATOR_MODEL", "GBENCH_BFCL_SERPAPI_LOG",
                     "tau2-openai-gateway", "36-server"):
            assert gone not in text, (rel, gone)
    knobs = _read("docs/evals/env-knobs.md")
    std = knobs[knobs.index("- Standard third party variables"):
                knobs.index("- Credential values never reach")]
    assert "mcp_atlas reads no `BRAVE_API_KEY`" in std
    listed = std[:std.index("mcp_atlas reads no `BRAVE_API_KEY`")]
    for gone in ("BRAVE_API_KEY", "SERPAPI_API_KEY", "RAPID_API_KEY"):
        assert gone not in listed, gone
    for name in ("`EXA_API_KEY`", "`OXYLABS_USERNAME`", "`OXYLABS_PASSWORD`"):
        assert name in listed, name


def test_env_knobs_follow_the_suites():
    from gbench.runners.eval_suites import bfcl_v4_agentic, livebench, tau_common, wildclawbench
    knobs = _read("docs/evals/env-knobs.md")
    lb = knobs[knobs.index("- livebench:"):knobs.index("- cyberseceval:")]
    assert f"`{livebench.LIVEBENCH_IMAGE_DEFAULT}`" in lb
    assert f"`{livebench.LIVEBENCH_RELEASE}`" in lb
    assert f"`{bfcl_v4_agentic.SEARCH_LOG_ENV}`" in knobs
    assert f"<project root>/{bfcl_v4_agentic.SEARCH_LOG_NAME}`" in knobs
    for name in bfcl_v4_agentic.SEARCH_KNOB_NAMES:
        assert f"`{name}`" in knobs, name
    wc = knobs[knobs.index("- wildclawbench:"):knobs.index("- ui_control_osworld")]
    for name in wildclawbench._FORWARDED_KNOBS:
        assert f"`{name}`" in wc, name
    tau = knobs[knobs.index("- tau2 and tau3 (shared"):knobs.index("- wildclawbench:")]
    removed = tau[tau.index("Removed (setting one logs a warning"):]
    for name in tau_common._REMOVED_KNOBS:
        if name.startswith("GBENCH_") or "RETRIEVAL" in name:
            assert f"`{name}`" in removed, name
    assert "`TAU2_EMBED_*` aliases" in removed
    head = tau[:tau.index("The\n  assistant temperature")]
    assert "RETRIEVAL_CONFIG" not in head and "OPENAI" not in head
    assert "<gbench cache>/tau2`" in tau
    nestful = knobs[knobs.index("- nestful:"):knobs.index("- mcp_bench:")]
    assert "the only data source" in nestful and "Hugging Face" not in nestful
    assert "`NESTFUL/data_v2/nestful_data.jsonl` (nestful)" in knobs
    ace = knobs[knobs.index("- acebench:"):knobs.index("- api_bank:")]
    assert "agent_multi_turn alone is reported infra_required" in ace
    assert "hard-errors without a key" not in ace
    assert "120,000 up to a 131,072-token window" in " ".join(knobs.split())
    quality = knobs[knobs.index("- quality: needs"):]
    assert "`GBENCH_QUALITY_GIT_CONFIG_GLOBAL`" in quality[:600]


def test_env_knobs_state_the_simple_evals_chat_sampler_in_both_modes():
    knobs = _read("docs/evals/env-knobs.md")
    assert "no temperature and no max_tokens of their own with it" not in knobs
    assert "The\n  simple-evals suites use its reasoning-model samplers" not in knobs
    assert "a reply that reached\n  the end of the context window" not in knobs
    assert "simple_evals.py (lines 145-212 at 652c89d0" in knobs


def test_readme_follows_the_suites():
    assert "follow its reasoning samplers and send none" not in _readme_with(*_README_SPLIT_DOCS)
    readme = _readme_with("docs/cli-reference.md", "docs/eval-suites.md", "docs/installation.md")
    row = next(l for l in readme.splitlines() if l.startswith("| `--max-output-tokens` |"))
    assert "its w/ CoT setting sends 1024 for the CoT query and 128 for the answer query" in row
    tau3 = next(l for l in readme.splitlines() if "| `tau3` |" in l)
    assert "gemini-embedding-001" in tau3 and "`GEMINI_API_KEY`" in tau3
    atlas = next(l for l in readme.splitlines() if "| `mcp_atlas` |" in l)
    assert "35 of its 36 servers" in atlas and "not leaderboard-comparable" in atlas
    for name in ("bfcl", "bfcl_v4_agentic"):
        row = next(l for l in readme.splitlines() if f"| `{name}` |" in l)
        assert "Gemini Google Search grounding" in row and "`GEMINI_API_KEY`" in row
    assert "- **Gemini-only helper models**" in readme
    from gbench.runners import quality
    assert quality.GEMINI_ALT_MODEL in readme and quality.GEMINI_IMAGE_GENERATION_MODEL in readme
    major, minor = quality._MIN_NODE[:2]
    assert f"Node v{major}.{minor}-or-newer base image" in readme


def test_toolchains_and_resources_follow_the_suites():
    tools = _read("docs/evals/toolchains.md")
    row = next(l for l in tools.splitlines() if l.startswith("| `GEMINI_API_KEY`"))
    assert "`bfcl` and `bfcl_v4_agentic`" in row and "gemini-embedding-001" in row
    assert "without brave-search" in tools
    res = _read("docs/RESOURCES.md")
    assert "gemini-embedding-001" in res
    assert "brave-search, ddg-search, exa, oxylabs" in res
    assert "Gemini API only; gpt-4-turbo upstream" in res
    from gbench.runners.eval_suites import complexfuncbench, livebench, nestful
    assert complexfuncbench.UPSTREAM_COMMIT in res and nestful.NESTFUL_COMMIT in res
    assert livebench.LIVEBENCH_COMMIT in res
    assert f"at commit {complexfuncbench.UPSTREAM_COMMIT[:7]}." in _read("NOTICE")


# --------------------------------------------------------------------------- #
# the provenance records the substitutions and decisions
# --------------------------------------------------------------------------- #
def test_provenance_records_the_gemini_substitutions():
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as CS
    grounding = "Gemini Google Search grounding"
    assert grounding in CS["bfcl"]["method"] and grounding in CS["bfcl_v4_agentic"]["method"]
    for suite in ("bfcl", "bfcl_v4_agentic"):
        assert "SERPAPI_API_KEY" not in CS[suite]["method"]
        assert "serpapi.com" not in CS[suite]["method"]
    tau3 = CS["tau3"]["method"]
    assert "gemini-embedding-001" in tau3 and "GBENCH_TAU2_OPENAI" not in tau3
    # no Gemini key enters the tau image: gbench's Gemini relay adds it on the host
    assert "no key enters the image" in CS["tau2"]["method"]
    assert "GEMINI_API_KEY is the only provider credential" not in CS["tau2"]["method"]
    assert grounding in CS["wildclawbench"]["method"]
    atlas = CS["mcp_atlas"]["method"]
    assert grounding in atlas and "brave-search, ddg-search, exa and oxylabs" in atlas
    assert "leaderboard_comparable=False" in atlas
    assert "on the Gemini API only" in CS["toolbench"]["method"]
    for suite in ("bfcl", "bfcl_v4_agentic", "tau2", "tau3", "wildclawbench", "mcp_atlas",
                  "toolbench", "complexfuncbench", "nestful", "livebench", "acebench"):
        assert CS[suite]["synced"] >= "2026-09-26", suite


def test_provenance_records_the_maintainer_decisions():
    from gbench.runners.eval_suites import complexfuncbench, livebench, nestful
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as CS
    lb = CS["livebench"]["method"]
    patch = livebench.LIVEBENCH_PATCH
    assert "LIVEBENCH_PATCH" in lb and "livebench_patch" in lb
    assert patch["sha256_upstream"][:8] in lb and patch["sha256_patched"][:8] in lb
    assert f"release {livebench.LIVEBENCH_RELEASE} by default" in CS["livebench"]["dataset"]
    assert "(or the model config's) own temperature and the policy's budget" in lb
    assert "the image checks run with --network none" in lb
    cfb = CS["complexfuncbench"]
    assert cfb["revision"] == complexfuncbench.UPSTREAM_COMMIT
    assert "rule_based -> value_checker -> response_based -> llm_based" in cfb["method"]
    assert "RAPID_API_KEY read from the environment and never written" in cfb["method"]
    nf = CS["nestful"]
    assert nf["revision"] == nestful.NESTFUL_COMMIT
    assert "the only data source" in nf["dataset"] and "no Hugging Face copy" in nf["dataset"]
    assert "Win Rate alone is reported infra_required" in nf["method"]
    assert "ibm-research/nestful" not in str(nf)
    ace = CS["acebench"]["method"]
    assert "only agent_multi_turn is infra_required" in ace


def test_provenance_records_the_simple_evals_chat_sampler_and_longbench_cot():
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as CS
    for suite in ("gpqa", "gpqa_diamond", "multilingual_mmlu", "simpleqa", "healthbench"):
        method = CS[suite]["method"]
        assert CS[suite]["synced"] >= "2026-09-26", suite
        assert "not reasoning settings for a served model" in method, suite
        assert "reasoning-model samplers as the upstream's explicit" not in method, suite
    lb = CS["longbench_v2_code"]
    assert "without and with --cot" in lb["upstream"]
    assert "settings.cot" in lb["method"] and "121,088" in lb["method"]
