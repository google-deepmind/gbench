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

"""The shared Gemini endpoint helper, the grounding search's key handling, the printed summary
and the shared docs and provenance, in step with the suites they describe.

* GEMINI_OPENAI_BASE_URL accepts only the Gemini API over https or a loopback address, through
  one helper (base.gemini_openai_base_url); the judge client always calls the Gemini API.
* The grounding search sends the key in the x-goog-api-key header, never in a URL, masks it in
  error texts, reads the keys exactly as base.gemini_api_keys does and honours a deadline.
* The printed summary formats numpy accuracies as numbers.
* The shared docs and the provenance describe the quality sandbox and relay, mcp_atlas's URL
  context page tools, wildclawbench's removed search skills, the tau infrastructure-error rule,
  the pinned ComplexFuncBench file and the Gemini substitutions of gaia and gaia2.
"""

import ast
import io
import json
import re
import time
import tomllib
from pathlib import Path

import pytest

from gbench.runners.eval_suites import base

REPO = Path(__file__).resolve().parents[1]
# Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
KEY_A = "AIza" + "W2MfakeKEYaaaaaaaaaaaaaaaaaaaaaaaaaa1"
KEY_B = "AIza" + "W2MfakeKEYbbbbbbbbbbbbbbbbbbbbbbbbbb2"


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """`text` with every run of whitespace (line breaks and indentation) as one space."""
    return " ".join(text.split())


# --------------------------------------------------------------------------- #
# GEMINI_OPENAI_BASE_URL: one helper, two kinds of accepted value
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [
    None, "", "   ",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
    "https://generativelanguage.googleapis.com",
    "https://generativelanguage.googleapis.com:443/v1beta/openai",
    "http://127.0.0.1:5555/v1", "https://127.0.0.1:5555/v1", "http://127.8.9.10:1/v1",
    "http://localhost:5555/v1", "http://[::1]:5555/v1",
])
def test_the_gemini_api_over_https_and_loopback_addresses_are_accepted(value):
    url, why = base.check_gemini_openai_base_url(value)
    assert why == ""
    expected = (value or "").strip() or base.GEMINI_OPENAI_BASE_URL
    assert url == expected.rstrip("/")


@pytest.mark.parametrize("value", [
    "https://api.openai.com/v1",
    "http://generativelanguage.googleapis.com/v1beta/openai/",
    "https://generativelanguage.googleapis.com.evil.example/v1",
    "https://evil.example/generativelanguage.googleapis.com/v1",
    "https://user:SECRETPW@generativelanguage.googleapis.com/v1beta/openai/",
    "https://SECRETTOKEN@127.0.0.1:5555/v1",
    "https://generativelanguage.googleapis.com/v1\nX-Evil: 1",
    "https://generativelanguage.googleapis.com /v1",
    "http://127.0.0.1:99999/v1",
    "ftp://127.0.0.1/v1",
    "http://0.0.0.0:5555/v1",
    "http://169.254.169.254/computeMetadata/v1",
    "http://metadata.google.internal/computeMetadata/v1",
    "http://172.17.0.1:8100/v1",
    "http://localhost.:5555/v1",
    "not a url",
    "//generativelanguage.googleapis.com/v1",
])
def test_every_other_value_is_refused_without_repeating_it(value, monkeypatch):
    url, why = base.check_gemini_openai_base_url(value)
    assert url is None
    assert "GEMINI_OPENAI_BASE_URL" in why and "not the Gemini API" in why
    for secret in ("SECRETPW", "SECRETTOKEN", "evil", "169.254", "metadata"):
        assert secret not in why
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", value)
    assert base.gemini_openai_base_url() == (url, why)


def test_require_raises_infra_required_naming_the_variable_only(monkeypatch):
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", "https://relay.example.com/v1?token=SECRETQ")
    with pytest.raises(RuntimeError, match=r"\[INFRA REQUIRED\] gaia2") as e:
        base.require_gemini_openai_base_url("gaia2")
    assert "GEMINI_OPENAI_BASE_URL" in str(e.value) and "SECRETQ" not in str(e.value)
    assert "relay.example.com" not in str(e.value) and "docs/evals/gaia2.md" in str(e.value)
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", "http://127.0.0.1:7/v1/")
    assert base.require_gemini_openai_base_url("gaia2") == "http://127.0.0.1:7/v1"
    monkeypatch.delenv("GEMINI_OPENAI_BASE_URL")
    assert base.require_gemini_openai_base_url("gaia2") == base.GEMINI_OPENAI_BASE_URL.rstrip("/")


def test_a_refused_endpoint_is_named_once_whatever_the_number_of_keys(monkeypatch):
    import urllib.request

    def no_request(*a, **k):
        raise AssertionError("a key was sent")
    monkeypatch.setattr(urllib.request, "urlopen", no_request)
    monkeypatch.setenv("GEMINI_OPENAI_BASE_URL", "https://api.openai.com/v1")
    ok, why = base.gemini_key_live_valid(f"{KEY_A};{KEY_B},AIzaW2MthirdKEY000000000000000000003")
    assert ok is False and why.count("GEMINI_OPENAI_BASE_URL") == 1
    assert KEY_A not in why and KEY_B not in why
    assert base.gemini_key_live_valid("") == (False, "no GEMINI_API_KEY")


def test_the_judge_client_always_calls_the_gemini_api(monkeypatch):
    """google-genai reads GOOGLE_GEMINI_BASE_URL and GOOGLE_GENAI_USE_VERTEXAI on its own; the
    judge client is built with an explicit base URL and vertexai=False, so neither moves it."""
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "https://evil.example/")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GEMINI_API_KEYS", f"{KEY_A},{KEY_B}")
    monkeypatch.setattr(base, "_JUDGE_CLIENTS", None)
    monkeypatch.setattr(base, "_JUDGE_CLIENT_CYCLE", None)
    clients = base._judge_clients()
    assert len(clients) == 2
    for client in clients:
        api = client._api_client
        assert api.vertexai is False
        assert api._http_options.base_url == base.GEMINI_API_BASE_URL
        assert "evil" not in str(api._http_options.base_url)


def test_the_helper_is_documented_as_the_one_reader():
    knobs = _flat(_read("docs/evals/env-knobs.md"))
    assert "`base.gemini_openai_base_url()`" in knobs
    assert "`base.require_gemini_openai_base_url()`" in knobs
    assert ("Its only accepted values are the Gemini API over https (host "
            "`generativelanguage.googleapis.com`) and a loopback address") in knobs
    assert "must be on generativelanguage.googleapis.com" not in knobs
    assert "`GOOGLE_GEMINI_BASE_URL` and `GOOGLE_GENAI_USE_VERTEXAI` do not move it" in knobs
    doc = base.gemini_openai_base_url.__doc__
    assert "one reader of GEMINI_OPENAI_BASE_URL" in doc


def test_the_check_is_standard_library_only():
    """Container code that cannot import gbench mirrors check_gemini_openai_base_url verbatim."""
    import inspect
    src = inspect.getsource(base.check_gemini_openai_base_url)
    imports = re.findall(r"^\s+(?:from|import) (\S+)", src, re.M)
    assert imports and all(m.split(".")[0] in ("ipaddress", "urllib") for m in imports)


# --------------------------------------------------------------------------- #
# the grounding search: header, masking, key list, deadline
# --------------------------------------------------------------------------- #
@pytest.fixture
def st(monkeypatch):
    from gbench.runners.eval_suites import search_tool
    monkeypatch.setattr(search_tool, "_RATE_LIMITER", search_tool._RateLimiter(0, 1))
    monkeypatch.setattr(search_tool, "_KEY_CYCLE", None)
    monkeypatch.setattr(search_tool, "_SEARCH_ATTEMPTS", 1)
    monkeypatch.setenv("GBENCH_SEARCH_CASCADE_ROUNDS", "1")
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "gemini-3.7-flash")
    monkeypatch.delenv("GEMINI_API_KEYS", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    search_tool.reset_backend_tally()
    yield search_tool
    search_tool.reset_backend_tally()


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_the_key_travels_in_the_header_never_in_the_url(st, monkeypatch):
    import urllib.request
    seen = []

    def urlopen(req, timeout=None):
        seen.append((req.full_url, req.get_header("X-goog-api-key"), timeout))
        return _Resp(json.dumps({"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})
                     .encode())
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setenv("GEMINI_API_KEY", KEY_A)
    results, backend = st.search_with_fallback("q")
    assert backend == "gemini:gemini-3.7-flash" and results[0]["title"] == "grounded-answer"
    [(url, header, timeout)] = seen
    assert url == ("https://generativelanguage.googleapis.com/v1beta/models/"
                   "gemini-3.7-flash:generateContent")
    assert header == KEY_A and KEY_A not in url and "key=" not in url
    assert timeout == st._REQUEST_TIMEOUT_S


def test_a_failed_lookup_never_returns_a_key(st, monkeypatch):
    import urllib.parse
    import urllib.request

    def fail(req, timeout=None):
        key = req.get_header("X-goog-api-key")
        raise ValueError(f"Invalid header value {key!r} ({urllib.parse.quote(key)})")
    monkeypatch.setattr(urllib.request, "urlopen", fail)
    monkeypatch.setenv("GEMINI_API_KEYS", f"{KEY_A},k+/=w2mSECONDKEY9")
    results, backend = st.search_with_fallback("q")
    assert backend == "all-grounding-models-failed"
    text = json.dumps(results)
    assert KEY_A not in text and "k+/=w2mSECONDKEY9" not in text
    assert "k%2B%2F%3Dw2mSECONDKEY9" not in text and "[REDACTED]" in text


@pytest.mark.parametrize("keys_var,key_var,expected", [
    (";", None, []),
    (" , ; ", None, []),
    ("   ", KEY_A, [KEY_A]),
    (";", KEY_A, [KEY_A]),
    (f"{KEY_A};{KEY_B};{KEY_A}", KEY_B, [KEY_A, KEY_B]),
    (None, f" {KEY_B} , {KEY_A} ", [KEY_B, KEY_A]),
    (None, None, []),
])
def test_the_key_list_is_read_exactly_as_base_reads_it(st, monkeypatch, keys_var, key_var,
                                                        expected):
    for name, value in (("GEMINI_API_KEYS", keys_var), ("GEMINI_API_KEY", key_var)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    assert st._gemini_keys() == base.gemini_api_keys() == expected
    assert st.search_available() is bool(expected)
    got = {st.get_gemini_api_key() for _ in range(4)}
    assert got == (set(expected) or {""})
    if not expected:
        results, backend = st.search_with_fallback("q")        # no StopIteration
        assert backend == "none" and results[0]["title"] == "error"
        assert st.search_backend_name() is None


def test_the_key_rotation_follows_a_changed_key_list(st, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEYS", "k1,k2")
    assert {st.get_gemini_api_key() for _ in range(4)} == {"k1", "k2"}
    monkeypatch.setenv("GEMINI_API_KEYS", "k3,k4")
    assert {st.get_gemini_api_key() for _ in range(4)} == {"k3", "k4"}


def test_a_passed_deadline_stops_the_lookup_before_any_request(st, monkeypatch):
    import urllib.request

    def no_request(*a, **k):
        raise AssertionError("a request was sent after the deadline")
    monkeypatch.setattr(urllib.request, "urlopen", no_request)
    monkeypatch.setenv("GEMINI_API_KEY", KEY_A)
    results, backend = st.search_with_fallback("q", deadline=time.monotonic() - 1)
    assert backend == "deadline-passed" and results[0]["title"] == "error"
    assert st.backend_tally() == {"deadline-passed": 1}
    got = st.gemini_search("q", model="gemini-3.7-flash", deadline=time.monotonic() - 1)
    assert got[0]["title"] == "error" and "deadline" in got[0]["snippet"]


def test_the_deadline_caps_the_request_timeout_and_the_cascade(st, monkeypatch):
    """A virtual clock (search_tool's time.monotonic/time.sleep): each request advances it 0.1 s.
    The two models' timeouts are the time left, the 30 s backoff is cut at the deadline, and the
    next round never starts."""
    import types
    import urllib.error
    import urllib.request
    clock = [100.0]
    timeouts, slept = [], []

    def sleep(seconds):
        slept.append(seconds)
        clock[0] += seconds
    monkeypatch.setattr(st, "time", types.SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep))

    def throttled(req, timeout=None):
        timeouts.append(timeout)
        clock[0] += 0.1
        raise urllib.error.HTTPError(req.full_url, 429, "quota", {}, None)
    monkeypatch.setattr(urllib.request, "urlopen", throttled)
    monkeypatch.setattr(st, "_SEARCH_BACKOFF", 30.0)
    monkeypatch.setenv("GBENCH_SEARCH_CASCADE_ROUNDS", "5")
    monkeypatch.setenv("GBENCH_SEARCH_MODELS", "gemini-a,gemini-b")
    monkeypatch.setenv("GEMINI_API_KEY", KEY_A)
    results, backend = st.search_with_fallback("q", deadline=100.5)
    assert backend == "deadline-passed" and results[0]["title"] == "error"
    assert timeouts == pytest.approx([0.5, 0.4])
    assert slept == pytest.approx([0.3])
    assert clock[0] == pytest.approx(100.5)


def test_the_search_module_stays_standalone():
    """wildclawbench's proxy loads search_tool.py on its own: no gbench import."""
    tree = ast.parse(_read("gbench/runners/eval_suites/search_tool.py"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and not (node.module or "").startswith("gbench"), node.module
        elif isinstance(node, ast.Import):
            assert not any(a.name.startswith("gbench") for a in node.names)


def test_the_search_docstring_describes_the_url_context_exception():
    from gbench.runners.eval_suites import search_tool
    doc = _flat(search_tool.__doc__)
    assert "mcp_atlas is the one exception" not in doc
    assert "No suite fetches pages through Gemini" in doc
    for tool in ("ddg-search_fetch_content", "exa_web_fetch_exa", "oxylabs_universal_scraper",
                 "oxylabs_amazon_product_scraper"):
        assert tool in doc
    assert "x-goog-api-key" in doc and "?key=" not in _read(
        "gbench/runners/eval_suites/search_tool.py")


# --------------------------------------------------------------------------- #
# the printed summary with numpy accuracies
# --------------------------------------------------------------------------- #
def test_pct_cell_formats_numpy_scalars_as_numbers():
    np = pytest.importorskip("numpy")
    from gbench import cli
    assert cli._pct_cell(np.float32(12.5)) == "    12.50%"
    assert cli._pct_cell(np.int64(3)) == "     3.00%"
    assert cli._pct_cell(np.float64(0.0)) == "     0.00%"
    assert cli._pct_cell(True) == cli._pct_cell(None) == "        - "
    assert cli._pct_cell(np.bool_(True)) == "        - "


# --------------------------------------------------------------------------- #
# the shared docs follow the suites
# --------------------------------------------------------------------------- #
def test_readme_describes_the_quality_sandbox_relay_and_prerequisite():
    readme = _read("README.md")
    bullet = next(l for l in readme.splitlines() if l.startswith("- **Gemini-only helper models**"))
    from gbench.runners import quality
    for model in (quality.GEMINI_ALT_MODEL, quality.GEMINI_IMAGE_GENERATION_MODEL,
                  quality.GEMINI_EMBEDDING_MODEL, quality.UPSTREAM_EMBEDDING_MODEL):
        assert model.split("/", 1)[1] in bullet, model
    assert "`gemini_calls`" in bullet and "gBench's own Docker container" in bullet
    assert "a forwarder in the container" in bullet and "bubblewrap" not in bullet
    assert "Harness errors (relay, endpoint, Gemini or container failures) are not scored" in bullet
    install = _read("docs/installation.md")                # the note moved out of README.md
    flat = _flat(re.sub(r"(?m)^>", "", install))              # the note is a blockquote
    assert ("needs **Docker** (the CLI and a daemon this user can reach), git, a Gemini API key "
            "(`GEMINI_API_KEYS` or `GEMINI_API_KEY`)") in flat
    assert "its QA suite always runs in gBench's local `gbench-quality` container" in flat
    assert "No Node.js, pnpm or bubblewrap is needed on the host for it" in flat
    env_vars = _read("docs/environment-variables.md")     # the env-var table moved out of README.md
    row = next(l for l in env_vars.splitlines() if l.startswith("| `GBENCH_SANDBOX` |"))
    assert "quality runner's QA suite always runs in gBench's own Docker container" in row
    atlas = next(l for l in _read("docs/eval-suites.md").splitlines() if "| `mcp_atlas` |" in l)
    assert "their page tools run as upstream on the release's servers" in atlas
    gem = next(l for l in env_vars.splitlines() if l.startswith("| `GEMINI_API_KEY` |"))
    # URL fetching is not a model role (Gemini scope): mcp_atlas' page tools are listed with the
    # non-model backends that run as upstream.
    assert "URL fetching" not in gem and "URL context" not in gem and \
        "mcp_atlas' exa and oxylabs page tools included" in gem


def test_env_knobs_follow_the_quality_wildclawbench_mcp_atlas_and_tau_suites():
    from gbench.runners import quality
    from gbench.runners.eval_suites import wildclawbench
    knobs = _read("docs/evals/env-knobs.md")
    q = _flat(knobs[knobs.index("- quality: needs"):knobs.index("- `GBENCH_EVAL_PLUGINS_PATH`")])
    assert "git and Docker" in q and f"`{quality.GEMINI_RELAY_ENV}` is set by gbench" in q
    assert "bubblewrap" not in q and "forwards only the Gemini API calls the suite makes" in q
    assert "`GBENCH_SANDBOX` does not apply" in q and quality.GEMINI_EMBEDDING_MODEL in q
    wc = _flat(knobs[knobs.index("- wildclawbench:"):knobs.index("- ui_control_osworld")])
    assert f"port {wildclawbench._JUDGE_PORT_DEFAULT} by default" in wc
    assert "may not be 18789-18793 or 18800-18899" in wc
    assert wildclawbench._OPENCLAW_PORTS == frozenset({*range(18789, 18794), *range(18800, 18900)})
    assert "the placeholder `gbench-model-relay`" in wc
    for skill in ("`local-web-search`", "`web-search`", "`google-search`"):
        assert skill in wc
    search = _flat(knobs[knobs.index("### Web search cascade"):knobs.index("## Per suite knobs")])
    assert "mcp_atlas: its page tools," not in search
    assert "mcp_atlas's page tools (Gemini URL context lookups) use" not in search
    tau = _flat(knobs[knobs.index("- tau2 and tau3 (shared"):knobs.index("- wildclawbench:")])
    assert ("and so is a simulation that ends in an infrastructure error from a harness failure"
            in tau)
    assert "`agent_turn_errors`" in tau and "for any cause (a Gemini user simulator" not in tau
    nestful = _flat(knobs[knobs.index("- nestful:"):knobs.index("- mcp_bench:")])
    assert "a data file that differs from the pinned revision is a hard error" in nestful
    std = _flat(knobs[knobs.index("- Standard third party variables"):
                      knobs.index("- Credential values never reach")])
    assert ("`GBENCH_WILDCLAWBENCH_JUDGE_ENDPOINT`, `GBENCH_GAIA2_JUDGE_ENDPOINT` and "
            "`GBENCH_GAIA2_JUDGE_MODEL` are removed") in std


def test_toolchains_resources_and_dataset_inventory_follow_the_suites():
    from gbench.runners.eval_suites import complexfuncbench
    tools = _read("docs/evals/toolchains.md")
    bwrap = next(l for l in tools.splitlines() if l.startswith("| `bubblewrap` (`bwrap`) |"))
    assert "quality" not in bwrap
    qimg = next(l for l in tools.splitlines() if "docker/quality.Dockerfile" in l)
    assert "its QA suite always runs in a container of it" in qimg
    assert "run their page tools as upstream" in tools
    assert "the URL context tool for exa's and oxylabs' page tools" not in tools
    res = _read("docs/RESOURCES.md")
    assert f"ComplexFuncBench` at {complexfuncbench.HF_REVISION}, sha256-checked at load" in res
    assert ("their page tools (ddg-search fetch_content, exa web_fetch_exa, oxylabs "
            "universal_scraper and amazon_product_scraper) run as upstream") in res
    assert "DuckDuckGo and Google Custom Search skills are removed" in res
    assert "embedding models run on the Gemini API through a gbench key relay" in res
    inv = _read("docs/THIRD_PARTY_DATASETS.md")
    assert f"`THUDM/ComplexFuncBench` @{complexfuncbench.HF_REVISION}" in inv


def test_jsonschema_is_declared_for_mcp_atlas():
    with open(REPO / "pyproject.toml", "rb") as fh:
        evals = tomllib.load(fh)["project"]["optional-dependencies"]["evals"]
    assert "jsonschema" in evals
    with open(REPO / "uv.lock", "rb") as fh:
        lock = tomllib.load(fh)
    gb = next(p for p in lock["package"] if p["name"] == "gbench")
    assert {"name": "jsonschema"} in gb["optional-dependencies"]["evals"]
    assert {"name": "jsonschema", "marker": "extra == 'evals'"} in gb["metadata"]["requires-dist"]
    assert any(p["name"] == "jsonschema" for p in lock["package"])


# --------------------------------------------------------------------------- #
# the provenance records the substitutions and decisions
# --------------------------------------------------------------------------- #
def test_provenance_records_every_gemini_substitution_of_the_search_suites():
    from gbench.runners.eval_suites import complexfuncbench
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC as CS
    gaia = CS["gaia"]["method"]
    assert "web_search = Gemini Google Search grounding (search_tool cascade)" in gaia
    assert "snippet-only" not in gaia
    assert gaia.endswith("leaderboard_comparable=False")
    gaia2 = CS["gaia2"]["method"]
    assert "in place of ARE's default Llama-3.3-70B judge" in gaia2
    assert "gbench Gemini cascade" in gaia2 and "leaderboard_comparable=False" in gaia2
    atlas = CS["mcp_atlas"]["method"]
    assert "Gemini URL context tool" not in atlas and "HARNESS_ERROR" in atlas
    assert "jsonschema.validate" in atlas and "Gemini Google Search grounding" in atlas
    wc = CS["wildclawbench"]["method"]
    assert "deleted from every task container" in wc and "model relay" in wc
    assert "Gemini Google Search grounding" in wc
    cfb = CS["complexfuncbench"]
    assert complexfuncbench.HF_REVISION in cfb["dataset"]
    assert cfb["checksum"].startswith(f"sha256:{complexfuncbench.HF_SHA256} ")
    assert "use_fp16=True" in cfb["method"]
    assert "a simulation that ends in an infrastructure error for any cause" in CS["tau2"]["method"]
    assert "a harness error as in tau2" in CS["tau3"]["method"]
    assert "en_core_web_sm) baked in" in CS["livebench"]["method"]
    for suite in ("gaia", "gaia2", "mcp_atlas", "wildclawbench", "complexfuncbench", "tau2",
                  "tau3", "livebench"):
        assert CS[suite]["synced"] >= "2026-09-26", suite
