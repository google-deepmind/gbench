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

"""The SWE family's run-time hosts, harness errors and the Windows probe's IPv6 check.

Docker, Harbor, the harness and the network are faked everywhere: no container starts, nothing is
pulled and nothing contacts 169.254.169.254 or a package host.

* multi_swe_bench's two gates check the package hosts the instance containers' builds fetch from
  at test time, and an unresolved instance whose fix-patch run captured no test result because a
  package host did not resolve is a harness error once corroborated, left out of the score.
* swe_bench_pro's sandbox gate checks the hosts every sandbox installs the agent from.
* terminal_bench: a Harbor exception outside the model's control is a harness error, left out of
  the score; AgentTimeoutError and a verifier that ran without a reward stay the model's outcome.
* A DNS failure in one task container needs corroboration (another one in the run, or a fresh
  probe after the run) before it leaves the score.
* The Windows probe also connects to the metadata server's IPv6 address.
* require_task_network_blocked's docstring gives the probe-image order the code uses.
"""

import ast
import asyncio
import base64
import json
import os
import re
import subprocess
import tempfile

import pytest

from gbench.runners.eval_suites import container_guard as CG
from gbench.runners.eval_suites import multi_swe_bench as M
from gbench.runners.eval_suites import skillsbench as SB
from gbench.runners.eval_suites import swe_agentic as A
from gbench.runners.eval_suites import swe_bench_pro as P
from gbench.runners.eval_suites import swebench_common as C
from gbench.runners.eval_suites import terminal_bench as TB

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP = "[INFRA REQUIRED] gate says stop"
#: The real gate, taken at import (before the conftest fixture stubs it for this file).
_REAL_GATE = C.require_task_network_blocked

#: Real output of `mvn clean test` in mswebench/fasterxml_m_jackson-core:pr-1208 run with
#: `docker run --network none` (captured on the dev VM; ANSI colours removed).
MAVEN_OUT = (
    "[INFO] Scanning for projects...\n"
    "Downloading from sonatype-nexus-snapshots: https://oss.sonatype.org/content/repositories/"
    "snapshots/com/fasterxml/jackson/jackson-base/2.17.4-SNAPSHOT/maven-metadata.xml\n"
    "[ERROR]     Non-resolvable parent POM for com.fasterxml.jackson.core:jackson-core:"
    "2.17.4-SNAPSHOT: Could not transfer artifact com.fasterxml.jackson:jackson-base:pom:"
    "2.17.4-SNAPSHOT from/to sonatype-nexus-snapshots (https://oss.sonatype.org/content/"
    "repositories/snapshots): Transfer failed for https://oss.sonatype.org/content/repositories/"
    "snapshots/com/fasterxml/jackson/jackson-base/2.17.4-SNAPSHOT/jackson-base-2.17.4-SNAPSHOT"
    ".pom and 'parent.relativePath' points at wrong local POM @ line 8, column 11: Unknown host "
    "oss.sonatype.org -> [Help 2]\n")
APT_OUT = """Err:1 http://archive.ubuntu.com/ubuntu noble InRelease
  Temporary failure resolving 'archive.ubuntu.com'
"""
CURL_OUT = "curl: (6) Could not resolve host: astral.sh\n"
PYTEST_RAN = ("============================= test session starts ==============================\n"
              "FAILED ../tests/test_outputs.py::test_output - AssertionError\n")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    for var in ("GBENCH_SHARD", "GBENCH_SKILLSBENCH_API_KEY", "GBENCH_TB_KEEP_JOBS_DIR",
                "OPENAI_API_KEY", "GBENCH_MULTI_SWE_BENCH_AGENT_IMAGE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(C, "require_task_network_blocked", _REAL_GATE)
    CG.reset_verdicts()
    C.reset_task_network_verdicts()
    yield
    CG.reset_verdicts()
    C.reset_task_network_verdicts()


def _cp(argv, rc=0, out="", err=""):
    return subprocess.CompletedProcess(argv, rc, out, err)


class FakeDocker:
    """subprocess.run stand-in for the docker CLI on a Linux engine. `present`: local images;
    `unresolved`: names the probe container cannot resolve; `reach`: the metadata verdict."""

    def __init__(self, present=(), unresolved=(), reach="unreachable"):
        self.calls = []
        self.present = set(present)
        self.unresolved = set(unresolved)
        self.reach = reach
        self.probed = []                 # (network, [names]) per probe container

    def __call__(self, argv, **kw):
        argv = list(argv)
        self.calls.append(argv)
        if argv[:3] == ["docker", "image", "inspect"]:
            return _cp(argv, 0 if argv[-1] in self.present else 1, "sha256:x\n")
        if argv[:2] == ["docker", "images"]:
            ref = argv[argv.index("--filter") + 1].split("=", 1)[1]
            hits = [i for i in sorted(self.present) if re.fullmatch(
                ref.replace("*", ".*"), i.rsplit(":", 1)[0])]
            return _cp(argv, 0, "".join(f"{h}\n" for h in hits))
        if argv[:3] in (["docker", "network", "create"], ["docker", "network", "rm"]):
            return _cp(argv, 0)
        if argv[:2] == ["docker", "run"]:
            net = argv[argv.index("--network") + 1]
            found = re.search(r"names = (\[.*?\])\n", argv[-1])
            names = ast.literal_eval(found.group(1)) if found else []
            self.probed.append((net, names))
            out = f"GBENCH_METADATA_PROBE={self.reach}\n"
            for n in names:
                out += (f"GBENCH_DNS_PROBE=fail {n} gaierror:-3\n" if n in self.unresolved
                        else f"GBENCH_DNS_PROBE=ok {n}\n")
            return _cp(argv, 0, out)
        raise AssertionError(f"unexpected command {argv}")

    def of(self, *prefix):
        return [c for c in self.calls if c[:len(prefix)] == list(prefix)]


def _install(monkeypatch, fake):
    monkeypatch.setattr(C.subprocess, "run", fake)
    return fake


def _record(monkeypatch, stop=False):
    seen = []

    def _gate(name, images=(), **kw):
        seen.append((name, list(images), kw))
        if stop:
            raise RuntimeError(STOP)
    monkeypatch.setattr(C, "require_task_network_blocked", _gate)
    return seen


# ============================================================================================
# multi_swe_bench: the run-time package hosts and DNS failures in the harness pass
# ============================================================================================
def test_runtime_hosts_follow_the_languages_of_the_run():
    assert M.runtime_hosts({"java"}) == ("repo.maven.apache.org", "oss.sonatype.org",
                                         "plugins.gradle.org", "services.gradle.org",
                                         "github.com")
    assert M.runtime_hosts({"ts", "js"}) == ("registry.npmjs.org", "github.com")
    assert M.runtime_hosts({"rust", "go"}) == ("proxy.golang.org", "index.crates.io",
                                               "static.crates.io", "github.com")
    assert M.runtime_hosts({"c", "cpp"}) == ("github.com",)
    assert set(M.RUNTIME_HOSTS) == set(M._LANGUAGES)


def test_the_rollout_gate_checks_the_hosts_of_the_rollouts_languages(monkeypatch, tmp_path):
    seen = _record(monkeypatch, stop=True)
    monkeypatch.setattr(M.subprocess, "run", lambda *a, **k: pytest.fail("a process ran"))
    tasks = [A.AgentTask(instance_id="a__b-1", problem_statement="p", image="i:1",
                         category="java"),
             A.AgentTask(instance_id="c__d-2", problem_statement="p", image="i:2",
                         category="go")]
    with pytest.raises(RuntimeError, match="gate says stop"):
        M.run_mswe_agent(tasks, {}, "m", "http://127.0.0.1:1/v1", 1, str(tmp_path), 0.0, 0.95)
    assert tuple(seen[0][2]["resolve"]) == M.runtime_hosts({"java", "go"})


def test_the_harness_pass_gate_checks_the_hosts_of_the_submitted_languages(monkeypatch):
    seen = _record(monkeypatch, stop=True)
    monkeypatch.setattr(M.subprocess, "run", lambda *a, **k: pytest.fail("a process ran"))
    traces = [{"category": "js", "extra_payload": {"instance_id": "o__r-1", "org": "O",
                                                   "repo": "R", "number": 1},
               "agent_patch": "diff --git a b\n"}]
    with pytest.raises(RuntimeError, match="gate says stop"):
        asyncio.run(M._make_scorer("m", 1, {})(traces))
    assert tuple(seen[0][2]["resolve"]) == ("registry.npmjs.org", "github.com")


def _instance_dir(workdir, org, repo, number, counts=(0, 0, 0), log=MAVEN_OUT, report=True):
    inst = os.path.join(workdir, org, repo, "evals", f"pr-{number}")
    os.makedirs(inst, exist_ok=True)
    if report:
        passed, failed, skipped = counts
        with open(os.path.join(inst, "report.json"), "w") as f:
            json.dump({"org": org, "repo": repo, "number": number, "valid": False,
                       "fix_patch_result": {"passed_count": passed, "failed_count": failed,
                                            "skipped_count": skipped}}, f)
    with open(os.path.join(inst, "fix-patch-run.log"), "w") as f:
        f.write(log)
    return inst


def test_a_fix_patch_run_that_resolved_nothing_names_its_hosts(tmp_path):
    e = {"org": "fasterxml", "repo": "jackson-core", "number": 1208}
    _instance_dir(str(tmp_path), "fasterxml", "jackson-core", 1208)
    assert M.harness_dns_failure(str(tmp_path), e) == ["oss.sonatype.org"]


def test_a_fix_patch_run_that_captured_tests_keeps_its_verdict(tmp_path):
    e = {"org": "o", "repo": "r", "number": 3}
    _instance_dir(str(tmp_path), "o", "r", 3, counts=(10, 2, 0))
    assert M.harness_dns_failure(str(tmp_path), e) == []
    e2 = {"org": "o", "repo": "r", "number": 4}
    _instance_dir(str(tmp_path), "o", "r", 4, report=False)
    assert M.harness_dns_failure(str(tmp_path), e2) == []


def test_the_new_resolver_signatures_name_their_hosts():
    domains = M.RUNTIME_DOMAINS
    assert C.dns_failure_hosts(MAVEN_OUT, domains) == ["oss.sonatype.org"]
    assert C.dns_failure_hosts("Caused by: java.net.UnknownHostException: repo.maven.apache.org"
                               "\n", domains) == ["repo.maven.apache.org"]
    assert C.dns_failure_hosts(
        "npm ERR! request to https://registry.npmjs.org/x failed, reason: getaddrinfo EAI_AGAIN "
        "registry.npmjs.org\n", domains) == ["registry.npmjs.org"]
    assert C.dns_failure_hosts(
        'go: github.com/a/b@v1.0.0: Get "https://proxy.golang.org/github.com/a/b/@v/v1.0.0.mod": '
        "dial tcp: lookup proxy.golang.org on 127.0.0.11:53: server misbehaving\n",
        domains) == ["proxy.golang.org"]
    assert C.dns_failure_hosts(
        "error: failed to get `serde` as a dependency\n  [6] Couldn't resolve host name (Could "
        "not resolve host: index.crates.io)\n", domains) == ["index.crates.io"]
    # a host outside the package domains (a test's own service) is not a package host
    assert C.dns_failure_hosts("java.net.UnknownHostException: db.internal\n", domains) == []


class _Harness:
    """The multi-swe-bench harness stand-in: writes final_report.json and, for each instance, its
    evals directory (report.json and fix-patch-run.log) under the config's workdir."""

    def __init__(self, resolved, dns=(), tested=()):
        self.resolved, self.dns, self.tested = set(resolved), set(dns), set(tested)

    def __call__(self, argv, **kw):
        cfg = json.load(open(argv[argv.index("--config") + 1]))
        workdir = cfg["workdir"]
        rows = [json.loads(line) for line in open(cfg["patch_files"][0])]
        resolved, unresolved = [], []
        for r in rows:
            iid = f"{r['org']}/{r['repo']}:pr-{r['number']}"
            (resolved if r["number"] in self.resolved else unresolved).append(iid)
            if r["number"] in self.dns:
                _instance_dir(workdir, r["org"], r["repo"], r["number"])
            elif r["number"] in self.tested:
                _instance_dir(workdir, r["org"], r["repo"], r["number"], counts=(3, 1, 0),
                              log=MAVEN_OUT + "Tests run: 4, Failures: 1\n")
        report = {"resolved_ids": resolved, "unresolved_ids": unresolved,
                  "completed_ids": resolved + unresolved, "error_ids": [],
                  "incomplete_ids": [], "total_instances": len(rows)}
        os.makedirs(cfg["output_dir"], exist_ok=True)
        with open(os.path.join(cfg["output_dir"], "final_report.json"), "w") as f:
            json.dump(report, f)
        return _cp(argv, 0)


def _score(monkeypatch, numbers, harness, recheck=None):
    monkeypatch.setattr(C, "require_task_network_blocked", lambda *a, **k: None)
    monkeypatch.setattr(M.subprocess, "run", harness)
    rechecks = []

    def _recheck(eval_name, images, names, **kw):
        rechecks.append((eval_name, list(images), list(names), kw))
        return recheck(names) if recheck else list(names)
    monkeypatch.setattr(C, "unresolved_after_run", _recheck)
    metrics = {}
    traces = [{"sample_idx": i, "category": "java",
               "extra_payload": {"instance_id": f"fasterxml__jackson-core-{n}",
                                 "org": "fasterxml", "repo": "jackson-core", "number": n},
               "agent_patch": "diff --git a/x b/x\n"} for i, n in enumerate(numbers)]
    asyncio.run(M._make_scorer("m", 1, metrics)(traces))
    result = {"status": "success"}
    reasons = M._report_reasons(result, metrics)
    return traces, metrics, result, reasons, rechecks


def test_two_dns_failures_in_the_harness_pass_leave_the_score(monkeypatch):
    traces, metrics, result, reasons, rechecks = _score(
        monkeypatch, [1, 2, 3, 4], _Harness(resolved={1}, dns={2, 3}, tested={4}))
    by = {t["extra_payload"]["number"]: t for t in traces}
    assert by[1]["is_correct"] and not by[1].get("scoring_excluded")
    for n in (2, 3):
        assert by[n]["scoring_excluded"] is True and by[n]["harness_error"] == "dns"
        assert by[n]["dns_failure_hosts"] == ["oss.sonatype.org"]
        assert "2 task containers of this run" in by[n]["dns_failure_evidence"]
    # its tests ran: the verdict stands even though the log names a DNS failure
    assert not by[4].get("scoring_excluded") and "dns_failure_hosts" not in by[4]
    assert not rechecks                                     # two candidates corroborate
    rep = metrics["multi_swe_bench_report"]
    assert sorted(rep["harness_dns_failures"]) == ["fasterxml__jackson-core-2",
                                                   "fasterxml__jackson-core-3"]
    assert result["status"] == "completed_with_errors"
    assert result["harness_errors"] == {"dns": 2}
    assert result["partial_evaluation"]["harness_dns_failure_instances"] == 2
    assert any("lost to harness errors (DNS failures" in r for r in reasons)
    # the one-shot runner's accuracy: 1 resolved over the 2 scored instances
    from gbench.runners.eval_suites.base import tally_batch_scores
    correct, cats = tally_batch_scores(traces)
    assert correct == 1 and cats["java"]["total"] == 2


def test_a_lone_dns_failure_is_rechecked_after_the_run(monkeypatch):
    traces, metrics, result, reasons, rechecks = _score(
        monkeypatch, [1, 2], _Harness(resolved={1}, dns={2}), recheck=lambda names: [])
    by = {t["extra_payload"]["number"]: t for t in traces}
    assert not by[2].get("scoring_excluded") and "harness_error" not in by[2]
    assert "stays the model's outcome" in by[2]["dns_failure_note"]
    assert "the harness's test run could not resolve oss.sonatype.org" in by[2][
        "dns_failure_note"]
    [(name, images, names, kw)] = rechecks
    assert name == "multi_swe_bench" and names == ["oss.sonatype.org"]
    assert images[0] == M.mswe_agent_image() and kw["references"] == ["mswebench/*"]
    assert result["status"] == "success" and "harness_errors" not in result
    assert metrics["multi_swe_bench_report"]["dns_failures_kept"]

    traces, metrics, result, reasons, _ = _score(
        monkeypatch, [1, 2], _Harness(resolved={1}, dns={2}))   # still unresolvable
    by = {t["extra_payload"]["number"]: t for t in traces}
    assert by[2]["scoring_excluded"] is True
    assert "could not resolve oss.sonatype.org either" in by[2]["dns_failure_evidence"]
    assert result["status"] == "completed_with_errors"


def test_a_harness_pass_of_nothing_but_dns_failures_withholds_its_score(monkeypatch):
    _, _, result, reasons, _ = _score(monkeypatch, [2, 3], _Harness(resolved=(), dns={2, 3}))
    assert result["status"] == "error"
    assert "could not score any of 2 submitted instances" in result["error"]
    assert "2 whose test run could not resolve oss.sonatype.org" in result["error"]
    assert reasons == ["harness infrastructure failure (no scoreable instances)"]


def test_the_agentic_denominator_leaves_dns_harness_errors_out(monkeypatch):
    tasks = [A.AgentTask(instance_id=f"o__r-{n}", problem_statement="p",
                         image=f"mswebench/o_m_r:pr-{n}", category="java",
                         meta={"org": "o", "repo": "r", "number": n}) for n in (1, 2, 3, 4)]
    monkeypatch.setattr(M, "_load_multi_swe_bench_tasks", lambda: (tasks, {}))

    class _Relay:
        url = "http://127.0.0.1:9/v1"

        def __init__(self, *a, **k):
            pass

        def start(self):
            return self

        def stop(self):
            pass

        def record(self):
            return {"failures": {}, "requests": 0}
    monkeypatch.setattr(M, "EndpointRelay", _Relay)
    monkeypatch.setattr(M, "relay_result_reasons", lambda *a, **k: [])
    monkeypatch.setattr(M, "run_mswe_agent", lambda tasks, *a, **k: [
        M.mswe_agent_trace(i, t, {"instances": {t.instance_id: {
            "completed": True, "exit_status": "submitted", "patch": "diff --git a/x b/x\n"}}},
            "/tmp", 0.0) for i, t in enumerate(tasks)])

    def _scorer(model_name, workers, metrics):
        async def _score_traces(traces):
            for t in traces:
                n = t["extra_payload"]["number"]
                t["is_correct"] = n == 1
                if n in (3, 4):
                    t.update(scoring_excluded=True, harness_error="dns")
            metrics["multi_swe_bench_report"] = {
                "gbench_submitted_instances": 4, "harness_unscored_instances": 0,
                "harness_dns_failures": {"o__r-3": {"hosts": ["oss.sonatype.org"]},
                                         "o__r-4": {"hosts": ["oss.sonatype.org"]}}}
        return _score_traces
    monkeypatch.setattr(M, "_make_scorer", _scorer)
    res = M._run_agentic("m", "http://127.0.0.1:1/v1", 1, False, {})
    assert res["total_questions"] == 4 and res["scoring_excluded"] == 2
    assert res["correct_answers"] == 1 and res["accuracy"] == 50.0
    assert res["status"] == "completed_with_errors" and res["harness_errors"] == {"dns": 2}
    assert res["category_accuracy"]["java"] == {"correct": 1, "total": 2, "accuracy": 50.0}


# ============================================================================================
# swe_bench_pro: the hosts every sandbox installs the agent from
# ============================================================================================
def test_the_v2_gate_checks_the_sandbox_install_hosts(monkeypatch):
    seen = _record(monkeypatch)
    P.require_sandbox_network_blocked()
    assert tuple(seen[0][2]["resolve"]) == P.SANDBOX_INSTALL_HOSTS
    assert P.SANDBOX_INSTALL_HOSTS == ("deb.debian.org", "security.debian.org",
                                       "archive.debian.org", "archive.ubuntu.com", "astral.sh",
                                       "github.com", "pypi.org", "files.pythonhosted.org")
    P.require_v1_network_blocked()
    assert "resolve" not in seen[1][2]


# ============================================================================================
# terminal_bench: Harbor exceptions
# ============================================================================================
@pytest.mark.parametrize("exc,reward,verifier,harness", [
    (None, 0.0, True, False),
    ("AgentTimeoutError", 0.0, True, False),
    ("AgentTimeoutError", None, True, False),
    ("NonZeroAgentExitCodeError", None, True, False),
    ("ContextLengthExceededError", None, False, False),
    ("VerifierTimeoutError", None, True, False),
    ("RewardFileNotFoundError", None, True, False),
    ("RewardFileNotFoundError", None, False, True),
    ("EnvironmentStartTimeoutError", None, False, True),
    ("AgentSetupTimeoutError", None, False, True),
    ("HealthcheckError", None, False, True),
    ("APIConnectionError", None, False, True),
    ("ApiRateLimitError", None, False, True),
    ("DownloadVerifierDirError", None, True, True),
    ("CancelledError", None, False, True),
    ("RuntimeError", 1.0, True, False),            # the verifier finished: its reward stands
])
def test_harbor_exceptions_are_classified_like_harbors_retry_policy(exc, reward, verifier,
                                                                    harness):
    data = {"trial_name": "t"}
    if exc:
        data["exception_info"] = {"exception_type": exc, "exception_message": "boom  here"}
    if verifier:
        data["verifier"] = {"started_at": "t1"}
    why = TB.trial_exception_harness_error(data, reward)
    assert bool(why) is harness
    if harness:
        assert why == f"{exc}: boom here"


def _trial(root, name, reward, verifier="", agent="ls -la", exc=None, verifier_started=True):
    trial = os.path.join(root, "job", name)
    os.makedirs(os.path.join(trial, "verifier"))
    os.makedirs(os.path.join(trial, "agent"))
    with open(os.path.join(trial, "verifier", "test-stdout.txt"), "w") as f:
        f.write(verifier)
    with open(os.path.join(trial, "agent", "trajectory.json"), "w") as f:
        json.dump({"steps": [{"message": agent}]}, f)
    res = {"trial_name": name}
    if reward is not None:
        res["verifier_result"] = {"rewards": {"reward": reward}}
    if exc:
        res["exception_info"] = {"exception_type": exc, "exception_message": f"{exc} happened"}
    if verifier_started:
        res["verifier"] = {"started_at": "t1"}
    with open(os.path.join(trial, "result.json"), "w") as f:
        json.dump(res, f)


def _run_harbor(monkeypatch, trials, recheck=None):
    monkeypatch.setattr(TB, "check_terminal_bench_prerequisites", lambda: (True, ""))
    # A complete run of a dataset of exactly these tasks, one trial each: the run counts every
    # expected trial (FULL_TASK_COUNT x the trials per task) without a result as a harness error.
    monkeypatch.setattr(TB, "FULL_TASK_COUNT", len(trials))
    monkeypatch.setattr(TB, "TERMINAL_BENCH_TRIALS", 1)
    monkeypatch.setattr(TB, "require_task_metadata_blocked", lambda task_names=None: None)
    monkeypatch.setattr(TB, "prepare_task_images",
                        lambda task_names=None: ({"cached": 0}, {"present": 0}))
    rechecks = []

    def _recheck(task_names, hosts):
        rechecks.append(list(hosts))
        return recheck(hosts) if recheck else list(hosts)
    monkeypatch.setattr(TB, "dns_recheck", _recheck)

    class _Proc:
        returncode = 0
        stdout = iter(())

        def poll(self):
            return 0

        def wait(self):
            return 0

    def _popen(cmd, **kw):
        jobs = cmd[cmd.index("--jobs-dir") + 1]
        for args in trials:
            _trial(jobs, *args[:4], **(args[4] if len(args) > 4 else {}))
        return _Proc()
    monkeypatch.setattr(TB.subprocess, "Popen", _popen)
    return TB.run_terminal_bench("m", "http://h:1/v1", 1), rechecks


def test_infrastructure_exceptions_leave_the_score_and_model_ones_stay(monkeypatch):
    res, _ = _run_harbor(monkeypatch, [
        ("t-pass", 1.0, PYTEST_RAN.replace("FAILED", "PASSED")),
        ("t-timeout", 0.0, PYTEST_RAN, "ls", {"exc": "AgentTimeoutError"}),
        ("t-noreward", None, "", "ls", {"exc": "RewardFileNotFoundError"}),
        ("t-env", None, "", "ls", {"exc": "EnvironmentStartTimeoutError",
                                   "verifier_started": False}),
        ("t-api", None, "", "ls", {"exc": "APIConnectionError", "verifier_started": False}),
    ])
    assert res["status"] == "completed_with_errors"
    assert res["total_questions"] == 3 and res["correct_answers"] == 1
    assert res["accuracy"] == pytest.approx(33.33)
    assert res["harness_errors"] == {"exception": 2}
    assert res["harness_error_trials"] == ["t-api", "t-env"]
    assert res["failed_requests"] == 2 and res["partial"] is True
    assert res["trial_errors"] == 4
    assert res["trial_exceptions"] == {"APIConnectionError": 1, "AgentTimeoutError": 1,
                                       "EnvironmentStartTimeoutError": 1,
                                       "RewardFileNotFoundError": 1}
    by = {t["trial_name"]: t for t in res["sample_traces"]}
    assert by["t-env"]["harness_error"] == "exception"
    assert by["t-env"]["harness_error_reason"] == ("EnvironmentStartTimeoutError: "
                                                   "EnvironmentStartTimeoutError happened")
    assert "harness_error" not in by["t-timeout"] and "harness_error" not in by["t-noreward"]
    reason = res["leaderboard_comparable_reason"]
    assert "2 trial(s) lost to harness errors (exceptions outside the model's control)" in reason
    assert "raised harness exceptions" not in reason


def test_timeouts_alone_keep_the_run_a_clean_measurement(monkeypatch):
    res, _ = _run_harbor(monkeypatch, [
        ("t1", 0.0, PYTEST_RAN, "ls", {"exc": "AgentTimeoutError"}),
        ("t2", 1.0, PYTEST_RAN.replace("FAILED", "PASSED")),
    ])
    assert res["status"] == "success" and res["harness_errors"] == {}
    assert res["partial"] is False and res["total_questions"] == 2
    assert res["trial_errors"] == 1


def test_a_run_of_nothing_but_infrastructure_exceptions_withholds_its_score(monkeypatch):
    with pytest.raises(RuntimeError) as err:
        _run_harbor(monkeypatch, [
            ("t1", None, "", "ls", {"exc": "EnvironmentStartTimeoutError",
                                    "verifier_started": False}),
            ("t2", None, APT_OUT, "ls", {}),
            ("t3", None, APT_OUT, "ls", {}),
        ])
    text = str(err.value)
    assert text.startswith("[INFRA REQUIRED] terminal_bench")
    assert "every one of the 3 trial(s) failed outside the model's control" in text
    assert "1 raised EnvironmentStartTimeoutError" in text
    assert "2 lost their verifier to a DNS failure (archive.ubuntu.com)" in text


# ============================================================================================
# DNS failures need corroboration
# ============================================================================================
def test_two_candidates_corroborate_each_other():
    errors, kept = C.corroborated_dns_failures({"a": ["pypi.org"], "b": ["astral.sh"]},
                                               recheck=lambda h: pytest.fail("rechecked"))
    assert set(errors) == {"a", "b"} and not kept
    assert "2 task containers of this run" in errors["a"]


@pytest.mark.parametrize("answer,harness,phrase", [
    (["pypi.org"], True, "could not resolve pypi.org either"),
    ([], False, "a fresh probe container after the run resolved pypi.org"),
    (None, False, "no fresh probe after the run could check the hosts"),
    ("raise", False, "no fresh probe after the run could check the hosts"),
    (["other.org"], False, "resolved pypi.org"),
])
def test_a_lone_candidate_needs_the_recheck(answer, harness, phrase):
    def _recheck(hosts):
        assert hosts == ["pypi.org"]
        if answer == "raise":
            raise RuntimeError("docker gone")
        return answer
    errors, kept = C.corroborated_dns_failures({"a": ["pypi.org"]}, recheck=_recheck)
    assert ("a" in errors) is harness and ("a" in kept) is (not harness)
    text = errors.get("a") or kept.get("a")
    assert phrase in text
    if not harness:
        assert "stays the model's outcome" in text


def test_terminal_bench_keeps_an_uncorroborated_dns_failure_as_the_models(monkeypatch):
    res, rechecks = _run_harbor(monkeypatch, [
        ("t-pass", 1.0, PYTEST_RAN.replace("FAILED", "PASSED")),
        ("t-dns", 0.0, APT_OUT + CURL_OUT),
    ], recheck=lambda hosts: [])
    assert rechecks == [["archive.ubuntu.com", "astral.sh"]]
    assert res["status"] == "success" and res["harness_errors"] == {}
    assert res["total_questions"] == 2 and res["accuracy"] == 50.0
    by = {t["trial_name"]: t for t in res["sample_traces"]}
    assert "harness_error" not in by["t-dns"]
    assert "nothing outside this task container corroborates" in by["t-dns"]["dns_failure_note"]


def test_skillsbench_keeps_an_uncorroborated_dns_failure_as_the_models(monkeypatch):
    monkeypatch.setattr(SB, "check_skillsbench_prerequisites", lambda: (True, ""))
    _record(monkeypatch)
    monkeypatch.setattr(SB, "dns_recheck", lambda hosts: [])
    per_task = {"a": 1.0, "b": 0.0}

    def _run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            workdir = cmd[cmd.index("-v", cmd.index("-v") + 1) + 1].split(":")[0]
            summary = {"conditions": {"without-skills": {
                "mean_reward": 0.5, "n_scored": 2, "n_tasks": 2, "per_task": per_task}}}
            with open(os.path.join(workdir, "skillsbench_summary.json"), "w") as f:
                json.dump(summary, f)
            vdir = os.path.join(workdir, "jobs", "without-skills", "b", "job-1", "rollout-1",
                                "verifier")
            os.makedirs(vdir)
            with open(os.path.join(vdir, "test-stdout.txt"), "w") as f:
                f.write(CURL_OUT)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(SB.subprocess, "run", _run)
    res = SB.run_skillsbench("m", "http://127.0.0.1:8000/v1", 1)
    assert res["status"] == "success" and res["harness_errors"] == {}
    assert res["mean_reward"] == 0.5 and res["n_scored"] == 2
    kept = res["dns_failures_kept"]["without-skills"]["b"]
    assert kept["hosts"] == ["astral.sh"] and "stays the model's outcome" in kept["note"]


def test_the_recheck_probes_fresh_on_both_networks_without_the_cache(monkeypatch):
    fake = _install(monkeypatch, FakeDocker(present={"img"}, unresolved={"astral.sh"}))
    assert C.unresolved_after_run("suite", ["img"], ["astral.sh", "pypi.org"]) == ["astral.sh"]
    assert [n for _, n in fake.probed] == [["astral.sh", "pypi.org"]] * 2
    assert fake.probed[0][0] == "bridge"
    assert fake.probed[1][0].startswith("gbench-metadata-probe-")
    assert len(fake.of("docker", "network", "rm")) == 1
    fake.unresolved.clear()
    assert C.unresolved_after_run("suite", ["img"], ["astral.sh", "pypi.org"]) == []
    assert len(fake.probed) == 4                            # probed again: no cache
    assert not fake.of("docker", "pull")


def test_the_recheck_gives_no_answer_it_could_not_establish(monkeypatch):
    fake = _install(monkeypatch, FakeDocker())
    assert C.unresolved_after_run("suite", ["absent"], ["pypi.org"]) is None
    assert not fake.of("docker", "run")
    _install(monkeypatch, FakeDocker(present={"img"}, reach="reachable"))
    assert C.unresolved_after_run("suite", ["img"], ["pypi.org"]) is None

    def _broken(argv, **kw):
        if argv[:3] == ["docker", "image", "inspect"]:
            return _cp(argv, 0)
        return _cp(argv, 1, "", "Cannot connect to the Docker daemon")
    _install(monkeypatch, _broken)
    assert C.unresolved_after_run("suite", ["img"], ["pypi.org"]) is None
    assert C.unresolved_after_run("suite", ["img"], []) == []


# ============================================================================================
# the Windows probe: IPv4 and IPv6
# ============================================================================================
def test_the_windows_probe_also_connects_to_the_ipv6_address():
    script = C.windows_probe_script()
    assert "ConnectAsync('169.254.169.254', 80).Wait(3000)" in script
    assert "[System.Net.Sockets.AddressFamily]::InterNetworkV6" in script
    assert "ConnectAsync([System.Net.IPAddress]::Parse('fd20:ce::254'), 80).Wait(3000)" in script
    assert "'AddressFamilyNotSupported', 'AddressNotAvailable'" in script
    assert "$c.Close()" in script and "$c6.Close()" in script
    for word in ("Invoke-WebRequest", "Invoke-RestMethod", "WebClient", "HttpClient", "GET ",
                 "computeMetadata", "Metadata-Flavor", "GetStream", ".Send"):
        assert word not in script, word


class _WinDocker:
    def __init__(self, out):
        self.out, self.runs = out, []

    def __call__(self, argv, **kw):
        argv = list(argv)
        if argv[:2] == ["docker", "info"]:
            return _cp(argv, 0, "windows\n")
        if argv[:3] == ["docker", "image", "inspect"]:
            return _cp(argv, 0, "sha256:x\n")
        if argv[:2] == ["docker", "run"]:
            self.runs.append(argv)
            return _cp(argv, 0, self.out)
        if argv[:3] == ["docker", "rm", "-f"]:
            return _cp(argv, 0)
        raise AssertionError(f"unexpected command {argv}")


def test_an_ipv6_answer_fails_closed_naming_the_ipv6_address(monkeypatch):
    fake = _install(monkeypatch, _WinDocker("GBENCH_METADATA_PROBE=reachable ipv6\n"))
    ok, why = C.check_windows_task_network("swe_bench_live", ["win:1"])
    assert not ok and why == C.windows_reachable_message("ipv6")
    assert "[fd20:ce::254]:80, its IPv6 address" in why
    script = base64.b64decode(fake.runs[0][-1]).decode("utf-16-le")
    assert script == C.windows_probe_script()


def test_both_families_unreachable_pass(monkeypatch):
    _install(monkeypatch, _WinDocker(
        "GBENCH_METADATA_PROBE=unreachable NetworkUnreachable ipv6:AddressNotAvailable\n"))
    assert C.check_windows_task_network("swe_bench_live", ["win:1"]) == (True, "")


def test_an_ipv6_error_verdict_gives_no_verdict(monkeypatch):
    _install(monkeypatch, _WinDocker(
        "GBENCH_METADATA_PROBE=error InvalidArgument ipv6\n"))
    ok, why = C.check_windows_task_network("swe_bench_live", ["win:1"])
    assert not ok and "gave no verdict (error InvalidArgument ipv6)" in why


# ============================================================================================
# require_task_network_blocked: the probe-image order its docstring gives
# ============================================================================================
def test_the_probe_image_order_is_images_then_pull_then_references(monkeypatch):
    fake = _install(monkeypatch, FakeDocker(present={"org/task:1", "mswebench/x_m_y:pr-1"}))
    C.require_task_network_blocked("suite", ["absent"], pull="org/task:1",
                                   references=["mswebench/*"], networks=("bridge",))
    [run] = fake.of("docker", "run")
    assert "org/task:1" in run and "mswebench/x_m_y:pr-1" not in run
    doc = " ".join(C.require_task_network_blocked.__doc__.split())
    assert ("The probe runs in the first of `images` present on the host, else in `pull` if "
            "that is present, else in a local image matching one of `references`") in doc


# ============================================================================================
# the docs say what the code does
# ============================================================================================
def _doc(name):
    return " ".join(open(os.path.join(REPO, "docs", "evals", f"{name}.md"),
                         encoding="utf-8").read().split())


def test_the_docs_describe_the_new_checks():
    mswe = _doc("multi_swe_bench")
    for host in {h for hosts in M.RUNTIME_HOSTS.values() for h in hosts} | set(
            M.COMMON_RUNTIME_HOSTS):
        assert f"`{host}`" in mswe, host
    for field in ("harness_dns_failures", "dns_failures_kept", "scoring_excluded",
                  "dns_failure_note", "fix_patch_result"):
        assert f"`{field}`" in mswe, field
    pro = _doc("swe_bench_pro")
    for host in P.SANDBOX_INSTALL_HOSTS:
        assert f"`{host}`" in pro, host
    tb = _doc("terminal_bench")
    assert '`harness_error: "exception"`' in tb and "`trial_exceptions`" in tb
    for exc in TB.AGENT_OUTCOME_EXCEPTIONS + TB.VERIFIER_OUTCOME_EXCEPTIONS:
        assert f"`{exc}`" in tb, exc
    assert "corroborates the failure" in tb
    assert "`dns_failures_kept`" in _doc("skillsbench")
    live = _doc("swe_bench_live")
    assert "`[fd20:ce::254]:80`" in live and "`AddressNotAvailable`" in live
