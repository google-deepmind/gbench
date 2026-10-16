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

"""Name resolution for the networked containers of tau2, tau3, livebench, aider_polyglot and
bigcodebench, and DNS or network failures during a run.

* Before a suite's bridge-network container starts, the metadata guard's probe container also
  resolves the host names that container looks up itself: the Gemini API (tau2, tau3), Hugging
  Face (livebench, bigcodebench), the toolchain hosts of the run's languages (aider_polyglot),
  and the model endpoint's host when it is a name (a proxy's instead when the container sends the
  request through one). A name that does not resolve is infra_required before any item runs.
* A DNS or network failure during a run that the output shows is a harness error, never the
  model's outcome: aider's exercises whose tests could not reach a toolchain host (or where aider
  gave up on the model endpoint) are left out of the score; a livebench or bigcodebench harness
  pass that produced nothing names the failed lookup and reports no score.
* ojbench and multipl_e run the model's code with --network none: nothing to resolve.

Docker and the probe are stubbed everywhere: no container starts and nothing is contacted.
"""

import asyncio
import json
import os
import re
import subprocess
import types
from pathlib import Path

import pytest

from gbench.runners.eval_suites import aider_polyglot as A
from gbench.runners.eval_suites import bigcodebench as BCB
from gbench.runners.eval_suites import container_guard as cg
from gbench.runners.eval_suites import livebench as LB
from gbench.runners.eval_suites import tau_common as TC

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "evals"
GEMINI = "generativelanguage.googleapis.com"
PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy")


def _flat(text):
    return " ".join(text.split())


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in PROXY_VARS + ("GBENCH_AIDER_LANGUAGES",):
        monkeypatch.delenv(name, raising=False)
    cg.reset_verdicts()
    TC._ENDPOINT_REACHABLE.clear()
    yield
    cg.reset_verdicts()
    TC._ENDPOINT_REACHABLE.clear()


def _guard_recorder(monkeypatch, fail=None):
    """Stands in for container_guard.require_metadata_blocked; records (args, resolve)."""
    calls = []

    def guard(eval_name, image, network, docs_url=None, *, resolve=()):
        calls.append(((eval_name, image, network, docs_url), list(resolve)))
        if fail is not None:
            raise fail
        return cg.UNREACHABLE
    monkeypatch.setattr(cg, "require_metadata_blocked", guard)
    monkeypatch.setattr(TC, "_require_endpoint_reachable", lambda *a, **k: None)
    return calls


class FakeDocker:
    """Answers each `docker run` from a list of (rc, stdout, stderr) and records its argv; any
    other docker command (info, image inspect, rm) succeeds."""

    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, argv, **kwargs):
        if argv[:2] != ["docker", "run"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        self.calls.append(list(argv))
        rc, out, err = self.answers.pop(0)
        return subprocess.CompletedProcess(argv, rc, out, err)


# --------------------------------------------------------------------------- #
# which names each suite's container resolves
# --------------------------------------------------------------------------- #
def test_each_suite_names_the_hosts_its_container_looks_up():
    assert TC.endpoint_container_hosts("tau2") == ([], True)
    assert TC.endpoint_container_hosts("tau3") == ([], True)
    assert TC.endpoint_container_hosts("livebench") == (["huggingface.co"], False)
    assert TC.endpoint_container_hosts("aider_polyglot") == (
        ["services.gradle.org", "repo.maven.apache.org", "index.crates.io", "static.crates.io"],
        False)
    assert TC.endpoint_container_hosts("bfcl") == ([], False)
    assert BCB.HARNESS_HOSTS == ("huggingface.co",)
    assert TC.GEMINI_API_HOST == GEMINI and TC._GEMINI_API_BASE == f"https://{GEMINI}/v1beta"


@pytest.mark.parametrize("langs, hosts", [
    ("python,go", []),
    ("java", ["services.gradle.org", "repo.maven.apache.org"]),
    ("rust cpp", ["index.crates.io", "static.crates.io"]),
])
def test_aider_resolves_only_the_toolchain_hosts_of_its_languages(monkeypatch, langs, hosts):
    monkeypatch.setenv("GBENCH_AIDER_LANGUAGES", langs)
    assert A.container_hosts() == hosts


@pytest.mark.parametrize("url, hosts, env, names", [
    ("http://127.0.0.1:8000/v1", [GEMINI], {}, [GEMINI]),
    ("http://localhost:8000/v1", [], {}, []),
    ("http://10.0.0.5:8000/v1", ["huggingface.co"], {}, ["huggingface.co"]),
    ("http://[fd00::5]:8000/v1", [], {}, []),
    ("https://models.example.com/v1", ["huggingface.co"], {},
     ["models.example.com", "huggingface.co"]),
    # a request the container sends through its proxy is resolved by the proxy
    ("https://models.example.com/v1", [GEMINI],
     {"HTTPS_PROXY": "http://proxy.corp:3128"}, ["proxy.corp"]),
    ("https://models.example.com/v1", [GEMINI],
     {"https_proxy": "http://proxy.corp:3128", "NO_PROXY": ".example.com"},
     ["models.example.com", "proxy.corp"]),
    ("http://127.0.0.1:8000/v1", [GEMINI],
     {"HTTPS_PROXY": f"http://{TC.HOST_ALIAS}:3128", "NO_PROXY": TC.HOST_ALIAS}, []),
    ("http://models.example.com:8000/v1", [GEMINI], {"HTTPS_PROXY": "http://10.1.1.1:3128"},
     ["models.example.com"]),
])
def test_container_lookups(url, hosts, env, names):
    assert TC.container_lookups(url, hosts, env) == names


def test_the_endpoint_gate_passes_the_suites_names_to_the_probe(monkeypatch):
    calls = _guard_recorder(monkeypatch)
    assert TC.require_endpoint_network("tau2", "img", "http://127.0.0.1:8000/v1", "d.md") \
        == f"http://{TC.HOST_ALIAS}:8000/v1"
    assert TC.require_endpoint_network("livebench", "img", "https://api.example.com/v1",
                                       "d.md") == "https://api.example.com/v1"
    monkeypatch.setenv("GBENCH_AIDER_LANGUAGES", "java")
    TC.require_endpoint_network("aider_polyglot", "img", "http://127.0.0.1:8000/v1", "d.md")
    TC.require_endpoint_network("bfcl", "img", "http://127.0.0.1:8000/v1", "d.md")
    TC.require_endpoint_network("bfcl", "img", "http://127.0.0.1:8000/v1", "d.md",
                                resolve=["example.org"])
    assert calls == [
        (("tau2", "img", "bridge", "d.md"), []),
        (("livebench", "img", "bridge", "d.md"), ["api.example.com", "huggingface.co"]),
        (("aider_polyglot", "img", "bridge", "d.md"),
         ["services.gradle.org", "repo.maven.apache.org"]),
        (("bfcl", "img", "bridge", "d.md"), []),
        (("bfcl", "img", "bridge", "d.md"), ["example.org"]),
    ]


def test_tau_resolves_its_proxy_instead_of_the_gemini_api_when_it_uses_one(monkeypatch):
    calls = _guard_recorder(monkeypatch)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.corp:3128")
    TC.require_endpoint_network("tau3", "img", "https://models.example.com/v1", "d.md")
    # livebench's container gets no proxy variables: it resolves Hugging Face itself
    TC.require_endpoint_network("livebench", "img", "http://127.0.0.1:8000/v1", "d.md")
    assert [names for _args, names in calls] == [["proxy.corp"], ["huggingface.co"]]


def test_a_name_that_does_not_resolve_stops_the_suite_before_the_endpoint_probe(monkeypatch):
    name = "models.example.com"
    fake = FakeDocker([(0, "GBENCH_METADATA_PROBE=unreachable timeout\n"
                           f"GBENCH_DNS_PROBE=fail {name} gaierror:-3\n", "")])
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(RuntimeError) as ei:
        TC.require_endpoint_network("tau2", "gbench-tau2:x", f"http://{name}:8000/v1",
                                    "docs/evals/tau2.md")
    msg = str(ei.value)
    assert "[INFRA REQUIRED] tau2" in msg and name in msg and "cannot resolve host names" in msg
    assert cg.FIREWALL_COMMAND in msg and cg.PLAIN_RULE_DELETE_COMMAND in msg
    (probe,) = [c for c in fake.calls if c[:2] == ["docker", "run"]]    # no endpoint probe
    assert "--pull" in probe and repr(name) in probe[-1]
    assert cg.cached_resolutions() == []


def test_a_resolved_name_is_not_probed_again(monkeypatch):
    fake = FakeDocker([(0, "GBENCH_METADATA_PROBE=unreachable timeout\n"
                           "GBENCH_DNS_PROBE=ok huggingface.co\n", "")])
    monkeypatch.setattr(subprocess, "run", fake)
    for _ in range(2):
        TC.require_endpoint_network("livebench", "img", "http://10.0.0.5:8000/v1", "d.md")
    assert len(fake.calls) == 1 and cg.cached_resolutions() == [("bridge", "huggingface.co")]


# --------------------------------------------------------------------------- #
# DNS and network failures in harness output
# --------------------------------------------------------------------------- #
GRADLE_WRAPPER = (
    "Downloading https://services.gradle.org/distributions/gradle-8.7-bin.zip\n"
    "\n"
    'Exception in thread "main" java.net.UnknownHostException: services.gradle.org\n'
    "\tat java.base/sun.nio.ch.NioSocketImpl.connect(NioSocketImpl.java:567)\n")
GRADLE_MAVEN = (
    "> Could not resolve all files for configuration ':testCompileClasspath'.\n"
    "   > Could not resolve org.junit:junit-bom:5.10.0.\n"
    "      > Could not get resource 'https://repo.maven.apache.org/maven2/org/junit/x.pom'.\n"
    "         > Could not GET 'https://repo.maven.apache.org/maven2/org/junit/x.pom'.\n"
    "            > repo.maven.apache.org: Temporary failure in name resolution\n")
CARGO_DNS = (
    "    Updating crates.io index\n"
    "error: failed to get `rand` as a dependency of package `simple-cipher v0.1.0`\n"
    "Caused by:\n  download of config.json failed\n"
    "Caused by:\n  failed to download from `https://index.crates.io/config.json`\n"
    "Caused by:\n  [6] Couldn't resolve host name (Could not resolve host: index.crates.io)\n")
CARGO_TIMEOUT = (
    "Caused by:\n  failed to download from `https://index.crates.io/config.json`\n"
    "\nCaused by:\n  [28] Timeout was reached (Connection timed out after 30000 milliseconds)\n")
JUNIT_FAIL = (
    "BowlingTest > twoRolls() FAILED\n"
    "    org.opentest4j.AssertionFailedError: expected: <10> but was: <9>\n"
    "5 tests completed, 1 failed\n")
MODEL_CODE_NET = (
    "FAILED test_fetch - requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.x.io',"
    " port=443): Max retries exceeded (Caused by NameResolutionError: Failed to resolve 'api.x.io')\n")


def test_network_failures_reads_the_toolchains_messages():
    hosts = A.TOOLCHAIN_FAILURE_HOSTS
    assert TC.network_failures(GRADLE_WRAPPER, hosts) == [
        'Exception in thread "main" java.net.UnknownHostException: services.gradle.org']
    assert TC.network_failures(GRADLE_MAVEN, hosts) == [
        "> repo.maven.apache.org: Temporary failure in name resolution"]
    assert TC.network_failures(CARGO_DNS, hosts) == [
        "[6] Couldn't resolve host name (Could not resolve host: index.crates.io)"]
    # a connection that timed out, with the URL on the line before
    assert TC.network_failures(CARGO_TIMEOUT, hosts) == [
        "[28] Timeout was reached (Connection timed out after 30000 milliseconds)"]
    assert TC.network_failures(CARGO_TIMEOUT, hosts, dns_only=True) == []
    # a test failure, or the model's own code failing to reach a host of its own, is not
    assert TC.network_failures(JUNIT_FAIL, hosts) == []
    assert TC.network_failures(MODEL_CODE_NET, hosts) == []
    # without hosts: any failed lookup
    assert TC.network_failures(MODEL_CODE_NET, dns_only=True)
    assert TC.network_failures("Connection refused", dns_only=True) == []
    # whole host names only
    assert TC.network_failures("UnknownHostException: notcrates.io", ["crates.io"]) == []
    assert TC.network_failures("UnknownHostException: static.crates.io.", ["crates.io"])


@pytest.mark.parametrize("output, kind", [
    (GRADLE_WRAPPER, "toolchain_network"), (GRADLE_MAVEN, "toolchain_network"),
    (CARGO_DNS, "toolchain_network"), (CARGO_TIMEOUT, "toolchain_network"),
])
def test_an_exercise_whose_tests_could_not_reach_a_toolchain_host_is_a_harness_error(output,
                                                                                      kind):
    # the test run's output block follows aider's own lines, as benchmark.py appends it
    history = ("#### fix the tests  \n\nsome reply\n\n> Applied edit to x  \n```\n" + output
               + "\n```")
    found = A.exercise_harness_error(history)
    assert found and found["kind"] == kind and found["detail"]


def test_aider_giving_up_on_the_model_endpoint_is_a_harness_error():
    retries = "".join(
        "> litellm.APIConnectionError: APIConnectionError: OpenAIException - Connection error.  \n"
        f"> Retrying in {d} seconds...  \n"
        for d in ("0.2", "0.5", "1.0", "2.0", "4.0", "8.0", "16.0", "32.0"))
    found = A.exercise_harness_error(retries + "```\nFAILED\n```")
    assert found["kind"] == "model_endpoint"
    assert found["detail"].startswith("litellm.APIConnectionError")
    # a request that failed once and then went through is not one
    assert A.exercise_harness_error("> litellm.Timeout: x  \n> Retrying in 0.2 seconds...  \n"
                                    "reply\n```\nok\n```") is None


@pytest.mark.parametrize("output", [JUNIT_FAIL, MODEL_CODE_NET, ""])
def test_an_ordinary_failure_is_the_models_outcome(output):
    assert A.exercise_harness_error("```\n" + output + "\n```") is None


def _exercise(root, lang, name, outcomes, history=None):
    d = root / lang / "exercises" / "practice" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / ".aider.results.json").write_text(json.dumps(
        {"language": lang, "testcase": name, "tests_outcomes": outcomes}))
    if history is not None:
        (d / ".aider.chat.history.md").write_text(history)


def test_parse_results_leaves_harness_error_exercises_out(tmp_path):
    _exercise(tmp_path, "python", "p1", [True], "ok")
    _exercise(tmp_path, "java", "bowling", [False, False], "```\n" + JUNIT_FAIL + "```")
    _exercise(tmp_path, "java", "bank", [False, False], "```\n" + GRADLE_WRAPPER + "```")
    _exercise(tmp_path, "rust", "grep", [False, True], "```\n" + CARGO_DNS + "```")
    agg = A._parse_results(str(tmp_path))
    assert agg["n"] == 2 and agg["pass1"] == 1 and agg["pass2"] == 1
    assert agg["per_lang"] == {"python": {"total": 1, "pass1": 1, "pass2": 1},
                               "java": {"total": 1, "pass1": 0, "pass2": 0}}
    assert [(h["exercise"], h["language"], h["kind"]) for h in agg["harness_errors"]] == [
        ("java/exercises/practice/bank", "java", "toolchain_network"),
        ("rust/exercises/practice/grep", "rust", "toolchain_network")]
    assert agg["columns"]["test_cases"] == 2


def _aider_run(monkeypatch, tmp_path, make):
    """run_aider_polyglot over a fake `docker run` that writes the exercises `make` gives."""
    monkeypatch.setenv("GBENCH_AIDER_BENCHMARK_DIR", str(tmp_path))
    monkeypatch.setenv("GBENCH_AIDER_MODEL", "openai/m")
    monkeypatch.setattr(A, "_check_prereqs", lambda image, bench: None)
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC.container_url(url))
    calls = {"n": 0}

    def fake_run(cmd, **k):
        calls["n"] += 1
        make(tmp_path / f"2026-run{calls['n']}--gbench")
        report = {"files_loaded": ["/gbench-aider/model-settings.yml"], "extra_params": {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
            "timeout": A.model_request_timeout_s(), "max_retries": 0,
            "temperature": 0.0}}  # a no-think run's temperature (base.resolve_temperature)
        return types.SimpleNamespace(returncode=0, stderr="",
                                     stdout=A._CHECK_MARKER + json.dumps(report) + "\n")
    monkeypatch.setattr(A.subprocess, "run", fake_run)


def test_a_full_aider_run_with_harness_errors_completes_with_errors(monkeypatch, tmp_path):
    langs = A._LANGUAGES

    def make(base):
        for i in range(A._EXPECTED_FULL):
            lang = langs[i % len(langs)]
            history = ("```\n" + GRADLE_WRAPPER + "```") if (lang == "java" and i < 20) else "ok"
            _exercise(base, lang, f"x{i}", [True] if history == "ok" else [False, False], history)
    _aider_run(monkeypatch, tmp_path, make)
    r = A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=1)
    java_bad = len([i for i in range(20) if langs[i % len(langs)] == "java"])
    assert r["status"] == "completed_with_errors" and r["harness_errors"] == java_bad
    assert r["harness_error_kinds"] == {"toolchain_network": java_bad}
    assert r["total_questions"] == A._EXPECTED_FULL - java_bad and r["accuracy"] == 100.0
    assert r["leaderboard_comparable"] is False
    assert f"{java_bad} exercise(s) with a harness error left out of the score" in \
        r["leaderboard_comparable_reason"]
    assert all(h["kind"] == "toolchain_network" for h in r["harness_error_exercises"])


def test_an_aider_run_whose_every_exercise_is_a_harness_error_reports_no_score(monkeypatch,
                                                                              tmp_path):
    def make(base):
        for i in range(3):
            _exercise(base, "java", f"x{i}", [False, False], "```\n" + GRADLE_MAVEN + "```")
    _aider_run(monkeypatch, tmp_path, make)
    for i in range(3):      # the exercises the limited run selects from
        (tmp_path / "polyglot-benchmark" / "java" / "exercises" / "practice" / f"x{i}").mkdir(
            parents=True, exist_ok=True)
    with pytest.raises(RuntimeError) as ei:
        A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=1, limit=3)
    msg = str(ei.value)
    assert "[INFRA REQUIRED] aider_polyglot" in msg and "no score is reported" in msg
    assert "3 toolchain_network" in msg and cg.FIREWALL_COMMAND in msg


def test_a_full_aider_run_still_needs_every_exercise_finished(monkeypatch, tmp_path):
    def make(base):
        for i in range(A._EXPECTED_FULL - 1):
            _exercise(base, "java" if i < 5 else "python", f"x{i}", [False],
                      ("```\n" + CARGO_DNS + "```") if i < 5 else "ok")
    _aider_run(monkeypatch, tmp_path, make)
    with pytest.raises(RuntimeError, match=r"partial run - finished 224/225 exercises "
                                           r"\(219 scored, 5 with a harness error\)"):
        A.run_aider_polyglot("m", "http://127.0.0.1:8000/v1", concurrency=1)


# --------------------------------------------------------------------------- #
# livebench
# --------------------------------------------------------------------------- #
def _livebench(monkeypatch, tmp_path, stdout="", stderr="", groups=True, rows=()):
    for name in ("GBENCH_LIVEBENCH_CATEGORIES", "GBENCH_LIVEBENCH_TEMPERATURE",
                 "GBENCH_LIVEBENCH_MODEL", "GBENCH_LIVEBENCH_EXTRA_ARGS",
                 "GBENCH_LIVEBENCH_RELEASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(LB, "_check_prereqs", lambda image: None)
    monkeypatch.setattr(LB, "upstream_model_config",
                        lambda image, names: {"name": None, "config": None, "commit": None})
    monkeypatch.setattr(LB, "server_aborted_requests", lambda url, ids: None)
    monkeypatch.setattr(LB.tempfile, "mkdtemp", lambda prefix="": str(tmp_path))
    monkeypatch.setattr(TC, "require_endpoint_network",
                        lambda eval_name, image, url, docs_url: TC.container_url(url))

    def fake_run(cmd, **_kw):
        cols = LB._CATEGORIES
        if groups:
            (tmp_path / "all_groups.csv").write_text(
                "model,average," + ",".join(cols) + "\nm,0.0," + ",".join("0.0" for _ in cols)
                + "\n", encoding="utf-8")
        if rows:
            d = tmp_path / LB._ANSWER_DIR / "data" / "live_bench" / "math" / "t" / "model_answer"
            d.mkdir(parents=True, exist_ok=True)
            (d / "m.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        return types.SimpleNamespace(returncode=0, stdout=stdout, stderr=stderr)
    monkeypatch.setattr(LB.subprocess, "run", fake_run)


def test_a_livebench_run_that_could_not_resolve_hugging_face_reports_no_score(monkeypatch,
                                                                             tmp_path):
    _livebench(monkeypatch, tmp_path, groups=False, stderr=(
        "requests.exceptions.ConnectionError: (MaxRetryError(\"HTTPSConnectionPool(host="
        "'huggingface.co', port=443): Max retries exceeded (Caused by NameResolutionError("
        "\"<urllib3.connection.HTTPSConnection>: Failed to resolve 'huggingface.co' ([Errno -3] "
        "Temporary failure in name resolution)\"))\"))\n"))
    with pytest.raises(RuntimeError) as ei:
        LB.run_livebench("m", "http://127.0.0.1:8000/v1", 4, max_output_tokens=4096)
    msg = str(ei.value)
    assert "[INFRA REQUIRED] livebench" in msg and "could not resolve a host name" in msg
    assert "huggingface.co" in msg and cg.FIREWALL_COMMAND in msg


def test_a_livebench_run_without_scores_and_no_dns_failure_is_still_a_harness_failure(
        monkeypatch, tmp_path):
    _livebench(monkeypatch, tmp_path, groups=False, stderr="Traceback: KeyError: 'x'\n")
    with pytest.raises(RuntimeError, match="harness/infra failure, not a 0%"):
        LB.run_livebench("m", "http://127.0.0.1:8000/v1", 4, max_output_tokens=4096)


def test_a_livebench_run_whose_every_request_failed_reports_no_score(monkeypatch, tmp_path):
    failed = {"eval_status": "api_error", "error": "APIConnectionError",
              "error_msg": "Connection error.", "choices": [{"turns": ["$ERROR$"]}]}
    _livebench(monkeypatch, tmp_path, rows=[failed, failed])
    with pytest.raises(RuntimeError) as ei:
        LB.run_livebench("m", "http://127.0.0.1:8000/v1", 4, max_output_tokens=4096)
    assert "[INFRA REQUIRED] livebench" in str(ei.value)
    assert "every one of the 2 requests" in str(ei.value)


def test_a_livebench_run_with_some_failed_requests_still_reports(monkeypatch, tmp_path):
    failed = {"eval_status": "api_error", "error": "APIConnectionError",
              "error_msg": "Connection error.", "choices": [{"turns": ["$ERROR$"]}]}
    ok = {"eval_status": "ok", "total_output_tokens": 10, "choices": [{"turns": ["42"]}]}
    _livebench(monkeypatch, tmp_path, rows=[failed, ok])
    r = LB.run_livebench("m", "http://127.0.0.1:8000/v1", 4, max_output_tokens=4096)
    assert r["status"] == "completed_with_errors" and r["failed_requests"] == 1


# --------------------------------------------------------------------------- #
# bigcodebench
# --------------------------------------------------------------------------- #
def test_bigcodebench_resolves_hugging_face_before_generation(monkeypatch):
    monkeypatch.setattr(BCB.shutil, "which", lambda _x: "/usr/bin/docker")
    fake = FakeDocker([(0, "GBENCH_METADATA_PROBE=unreachable timeout\n"
                           "GBENCH_DNS_PROBE=fail huggingface.co timeout\n", "")])
    monkeypatch.setattr(subprocess, "run", fake)       # the suite's docker calls and the probe
    with pytest.raises(RuntimeError) as ei:
        BCB.check_bigcodebench_prerequisites()
    msg = str(ei.value)
    assert "[INFRA REQUIRED] bigcodebench" in msg and "huggingface.co: timeout" in msg
    assert cg.FIREWALL_COMMAND in msg
    (probe,) = fake.calls
    assert probe[probe.index("--network") + 1] == "bridge" and "'huggingface.co'" in probe[-1]


# What the image prints with no DNS (captured from gbench-bigcodebench:09dd993f with --network none):
# datasets raises the ConnectionError `from None`, so no resolver text reaches the output.
HF_DNS = ("Traceback (most recent call last):\n"
          '  File "/usr/local/lib/python3.10/site-packages/datasets/load.py", line 1805, in '
          "dataset_module_factory\n"
          "    raise ConnectionError(f\"Couldn't reach '{path}' on the Hub "
          "({type(e).__name__})\")\n"
          "ConnectionError: Couldn't reach 'bigcode/bigcodebench' on the Hub (ConnectionError)\n")


def _bcb_score(monkeypatch, fail_splits):
    """Run the scorer over one instruct and one complete trace; the harness of `fail_splits`
    produces nothing and prints a failed Hugging Face lookup."""
    def fake_run(cmd, **k):
        mount = next(a for a in cmd if a.endswith(":/app:rw")).split(":", 1)[0]
        split = cmd[cmd.index("--split") + 1] if "--split" in cmd else None
        if "bigcodebench.sanitize" in cmd:
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if split in fail_splits:
            return types.SimpleNamespace(returncode=1, stdout="", stderr=HF_DNS)
        with open(os.path.join(mount, "samples_eval_results.json"), "w") as fh:
            json.dump({"eval": {"BigCodeBench/0": [{"status": "pass", "details": {}}]}}, fh)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(BCB.subprocess, "run", fake_run)
    traces = [{"extra_payload": {"task_id": "BigCodeBench/0", "split": s}, "response_text": "x"}
              for s in ("instruct", "complete")]
    metrics = {}
    asyncio.run(BCB._make_scorer("full", 1, metrics, "img")(traces))
    return traces, metrics


def test_a_bigcodebench_split_whose_harness_produced_nothing_is_not_scored(monkeypatch):
    traces, metrics = _bcb_score(monkeypatch, {"complete"})
    instruct, complete = traces
    assert instruct["is_correct"] is True and not instruct.get("scoring_excluded")
    assert complete["scoring_excluded"] is True and complete["status"] == "HARNESS_ERROR"
    assert metrics["bigcodebench_harness_errors"] == {"complete": {
        "kind": "hub_unreachable", "samples": 1,
        "detail": "ConnectionError: Couldn't reach 'bigcode/bigcodebench' on the Hub "
                  "(ConnectionError)"}}
    err = metrics["bigcodebench_report"]["error"]
    assert err.startswith("harness produced no parsed results (complete split): its containers "
                          "could not reach the Hugging Face Hub")
    splits = BCB._split_scores(traces, "full")
    assert set(splits) == {"instruct"}


def _bcb_run(monkeypatch, traces, metrics):
    monkeypatch.delenv("GBENCH_BIGCODEBENCH_SUBSET", raising=False)
    monkeypatch.setattr(BCB, "check_bigcodebench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(BCB, "_prefill_supported", lambda: True)
    monkeypatch.setattr(BCB, "_load_bigcodebench_samples", lambda limit=None, subset="full": [])

    def fake_make_scorer(subset, parallel, m, image):
        m.update(metrics)
        return None
    monkeypatch.setattr(BCB, "_make_scorer", fake_make_scorer)
    monkeypatch.setattr(BCB, "run_eval_suite", lambda **k: {
        "accuracy": 50.0, "temperature": 0.0, "status": "success", "sample_traces": traces,
        "max_tokens_sent": [BCB.UPSTREAM_MAX_NEW_TOKENS], "total_questions": 2,
        "correct_answers": 1})
    return BCB.run_bigcodebench("m", "http://h/v1", 1)


def test_bigcodebench_with_the_complete_split_unscored_completes_with_errors(monkeypatch):
    traces, metrics = _bcb_score(monkeypatch, {"complete"})
    r = _bcb_run(monkeypatch, traces, metrics)
    assert r["status"] == "completed_with_errors" and r["accuracy"] == 100.0
    assert r["leaderboard_comparable"] is False
    assert "harness error: harness produced no parsed results (complete split)" in \
        r["leaderboard_comparable_reason"]


def test_bigcodebench_without_the_instruct_split_withholds_the_headline(monkeypatch):
    traces, metrics = _bcb_score(monkeypatch, {"instruct"})
    r = _bcb_run(monkeypatch, traces, metrics)
    assert r["status"] == "completed_with_errors" and r["accuracy"] is None
    assert r["total_questions"] == 0 and r["correct_answers"] == 0
    assert r["accuracy_withheld_reason"].startswith("BigCodeBench-Instruct was not scored")
    assert set(r["splits"]) == {"complete"}


def test_bigcodebench_with_no_split_scored_is_an_error_without_a_score(monkeypatch):
    traces, metrics = _bcb_score(monkeypatch, {"instruct", "complete"})
    r = _bcb_run(monkeypatch, traces, metrics)
    assert r["status"] == "error" and r["accuracy"] is None and r["splits"] == {}
    assert r["leaderboard_comparable"] is False


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("doc, names", [
    ("tau2.md", [GEMINI]),
    ("tau3.md", [GEMINI]),
    ("livebench.md", ["huggingface.co"]),
    ("bigcodebench.md", ["huggingface.co"]),
    ("aider_polyglot.md", ["services.gradle.org", "repo.maven.apache.org", "index.crates.io",
                           "static.crates.io"]),
])
def test_the_bridge_guides_name_what_their_containers_resolve(doc, names):
    flat = _flat((DOCS / doc).read_text(encoding="utf-8"))
    for name in names:
        assert f"`{name}`" in flat, (doc, name)
    assert re.search(r"needs? working DNS", flat), doc
    assert "general internet access" in flat, doc
    assert "only the metadata server is blocked" in flat, doc


@pytest.mark.parametrize("doc", ["ojbench.md", "multipl_e.md"])
def test_the_no_network_guides_say_the_models_code_has_no_network(doc):
    flat = _flat((DOCS / doc).read_text(encoding="utf-8"))
    assert "no host name to resolve" in flat and "no internet access" in flat


def test_the_aider_guide_documents_the_harness_errors():
    flat = _flat((DOCS / "aider_polyglot.md").read_text(encoding="utf-8"))
    for key in ("`harness_errors`", "`harness_error_kinds`", "`harness_error_exercises`",
                "`toolchain_network`", "`model_endpoint`", "Retrying in 32.0 seconds..."):
        assert key in flat, key
