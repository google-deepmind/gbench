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

"""The SWE family's metadata server gate: no pulls for the probe, name resolution, DNS failures
as harness errors, and the Windows tab's own check.

Docker is faked everywhere: no container or network starts, nothing is pulled and nothing
contacts 169.254.169.254.

* swebench_common.require_task_network_blocked never pulls: it probes with an image already on
  the host, and without one it is infra_required naming the `docker pull` of an image the run
  starts. With `resolve` the probe container also resolves the names the task containers need,
  on both task networks, and caches them.
* terminal_bench probes with a task image of the run's own tasks first and checks the verifiers'
  package hosts; a trial whose verifier could not resolve them before its tests ran is a harness
  error, left out of the score (a run of nothing but those withholds its score).
* skillsbench checks the same hosts and leaves such tasks, and tasks with no reward, out of its
  means.
* swe_bench_live's Windows tab runs only after a PowerShell probe in one of its own instance
  images, on a Windows-containers engine, found the metadata server unreachable; it fails closed
  otherwise, and on a Linux engine it stays infra_required with that reason.
"""

import ast
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
from gbench.runners.eval_suites import swe_bench_live as LV
from gbench.runners.eval_suites import swebench_common as C
from gbench.runners.eval_suites import terminal_bench as TB

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP = "[INFRA REQUIRED] gate says stop"
#: The real gate, taken at import (before the conftest fixture stubs it for this file).
_REAL_GATE = C.require_task_network_blocked

#: Real output of each installer in a container without network (captured on the dev VM with
#: `docker run --network none`): apt-get update, curl, pip and uv/uvx.
APT_OUT = """Err:1 http://archive.ubuntu.com/ubuntu noble InRelease
  Temporary failure resolving 'archive.ubuntu.com'
Err:2 http://security.ubuntu.com/ubuntu noble-security InRelease
  Temporary failure resolving 'security.ubuntu.com'
"""
CURL_OUT = "curl: (6) Could not resolve host: astral.sh\n"
PIP_OUT = ("WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, "
           "status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection."
           "HTTPSConnection object at 0x7023af210410>: Failed to establish a new connection: "
           "[Errno -3] Temporary failure in name resolution')': /simple/pytest/\n"
           "ERROR: Could not find a version that satisfies the requirement pytest==8.4.1 (from "
           "versions: none)\n")
UV_OUT = """error: Request failed after 3 retries in 6.3s
  Caused by: Failed to fetch: `https://pypi.org/simple/pytest/`
  Caused by: error sending request for url (https://pypi.org/simple/pytest/)
  Caused by: client error (Connect)
  Caused by: dns error
  Caused by: failed to lookup address information: Temporary failure in name resolution
"""
PYTEST_RAN = ("============================= test session starts ==============================\n"
              "FAILED ../tests/test_outputs.py::test_output - AssertionError\n")


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    for var in ("GBENCH_SHARD", "GBENCH_SKILLSBENCH_API_KEY", "GBENCH_TB_KEEP_JOBS_DIR",
                "OPENAI_API_KEY"):
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
    `unresolved`: names the probe container cannot resolve."""

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
            return _cp(argv, 0, "")
        if argv[:3] in (["docker", "network", "create"], ["docker", "network", "rm"]):
            return _cp(argv, 0)
        if argv[:2] == ["docker", "pull"]:
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
# the gate pulls nothing, and checks the names the task containers need
# ============================================================================================
def test_the_gate_never_pulls_and_names_the_pull_to_run(monkeypatch):
    fake = _install(monkeypatch, FakeDocker())
    with pytest.raises(RuntimeError) as err:
        C.require_task_network_blocked("suite", ["gbench-x"], pull="org/task:1",
                                       hint="Or build gbench-x.")
    text = str(err.value)
    assert text.startswith("[INFRA REQUIRED] suite")
    assert "the probe never pulls one" in text and "gbench-x" in text
    assert "Pull one on this host first: `docker pull org/task:1`." in text
    assert "Or build gbench-x." in text
    assert not fake.of("docker", "pull") and not fake.of("docker", "run")


def test_the_named_image_is_used_when_it_is_already_here(monkeypatch):
    fake = _install(monkeypatch, FakeDocker(present={"org/task:1"}))
    C.require_task_network_blocked("suite", ["gbench-x"], pull="org/task:1")
    runs = fake.of("docker", "run")
    assert len(runs) == 2 and all("org/task:1" in r and "never" in r for r in runs)
    assert not fake.of("docker", "pull")


def test_a_long_image_list_is_cut_to_five_in_the_message(monkeypatch):
    _install(monkeypatch, FakeDocker())
    images = [f"img:{i}" for i in range(20)]
    with pytest.raises(RuntimeError) as err:
        C.require_task_network_blocked("suite", images[1:], pull=images[0])
    text = str(err.value)
    assert "img:1, img:2, img:3, img:4, img:5, 15 more is present" in text
    assert "img:9" not in text and "`docker pull img:0`" in text


def test_names_are_resolved_on_both_task_networks_and_cached(monkeypatch):
    fake = _install(monkeypatch, FakeDocker(present={"img"}))
    C.require_task_network_blocked("suite", ["img"], resolve=("pypi.org", "github.com"))
    assert [n for _, n in fake.probed] == [["pypi.org", "github.com"]] * 2
    assert fake.probed[0][0] == "bridge"
    assert fake.probed[1][0].startswith("gbench-metadata-probe-")
    before = len(fake.calls)
    C.require_task_network_blocked("suite", ["img"], resolve=("github.com", "pypi.org"))
    assert len(fake.calls) == before                       # settled for both networks
    C.require_task_network_blocked("suite", ["img"], resolve=("pypi.org", "astral.sh"))
    assert [n for _, n in fake.probed[2:]] == [["astral.sh"], ["astral.sh"]]


def test_a_name_that_does_not_resolve_is_infra_required_and_not_cached(monkeypatch):
    fake = _install(monkeypatch, FakeDocker(present={"img"}, unresolved={"astral.sh"}))
    with pytest.raises(RuntimeError) as err:
        C.require_task_network_blocked("suite", ["img"], resolve=C.TASK_INSTALL_HOSTS)
    text = str(err.value)
    assert "cannot resolve host names (astral.sh: gaierror:-3)" in text
    assert CG.FIREWALL_COMMAND in text and "the VPC DNS resolver is 169.254.169.254" in text
    fake.unresolved.clear()
    C.require_task_network_blocked("suite", ["img"], resolve=C.TASK_INSTALL_HOSTS)
    # probed again: the failed name was not cached, the ones that resolved were
    assert fake.probed[-2] == ("bridge", ["astral.sh"])


def test_the_one_x_rollouts_name_the_pull_of_the_first_task_image(monkeypatch):
    fake = _install(monkeypatch, FakeDocker())
    task = A.AgentTask(instance_id="o__r-1", problem_statement="p", image="swebench/o:1",
                       category="o/r")
    with pytest.raises(RuntimeError, match=r"`docker pull swebench/o:1`"):
        A.require_rollout_network_blocked("copilot_bench_swe", [task])
    assert not fake.of("docker", "pull")


def test_the_host_harness_names_the_pull_of_the_first_instance_image(monkeypatch):
    fake = _install(monkeypatch, FakeDocker())
    image = A.swebench_image("a__b-1", "swebench")
    with pytest.raises(RuntimeError, match=re.escape(f"`docker pull {image}`")):
        C.require_harness_network_blocked("x", None, ["a__b-1", "a__b-2"], namespace="swebench")
    assert not fake.of("docker", "pull")


def test_the_multi_swe_bench_harness_probe_pulls_nothing(monkeypatch):
    seen = _record(monkeypatch, stop=True)
    monkeypatch.setattr(M.subprocess, "run", lambda *a, **k: pytest.fail("a process ran"))
    scorer = M._make_scorer("m", 1, {})
    traces = [{"extra_payload": {"instance_id": f"o__r-{n}", "org": "O", "repo": "R",
                                 "number": n}, "agent_patch": "diff --git a b\n"}
              for n in (1, 2)]
    import asyncio
    with pytest.raises(RuntimeError, match="gate says stop"):
        asyncio.run(scorer(traces))
    _, images, kw = seen[0]
    assert images == [M.mswe_agent_image(), "mswebench/o_m_r:pr-2"]
    assert kw["pull"] == "mswebench/o_m_r:pr-1" and kw["references"] == ["mswebench/*"]
    assert M.mswe_agent_image() in kw["hint"]


# ============================================================================================
# terminal_bench: the run's own task images, the verifiers' hosts, DNS failures in the run
# ============================================================================================
def _harbor_cache(tmp_path, monkeypatch, tasks, digest_layout=True):
    """Tasks in gbench's Harbor cache (terminal_bench.harbor_package_cache, the harness
    container's cache on this host; nothing imports Harbor on the host)."""
    root = tmp_path / "packages"
    for name, image in tasks.items():
        task = root / "terminal-bench" / name
        if digest_layout:
            task = task / ("0" * 64)
        task.mkdir(parents=True)
        (task / "task.toml").write_text(f'version = "1.0"\n[environment]\ndocker_image = '
                                        f'"{image}"\n')
    monkeypatch.setattr(TB, "harbor_package_cache", lambda: str(root))
    return root


@pytest.mark.parametrize("digest_layout", [True, False])
def test_harbor_task_images_reads_both_cache_layouts_and_the_runs_tasks(monkeypatch, tmp_path,
                                                                        digest_layout):
    _harbor_cache(tmp_path, monkeypatch, {"a-task": "alexgshaw/a:1", "b-task": "alexgshaw/b:1",
                                          "c-task": "alexgshaw/c:1"}, digest_layout)
    assert TB.harbor_task_images() == ["alexgshaw/a:1", "alexgshaw/b:1", "alexgshaw/c:1"]
    assert TB.harbor_task_images(task_names=["terminal-bench/c-task", "terminal-bench/a-task",
                                             "other-org/b-task"]) == ["alexgshaw/c:1",
                                                                      "alexgshaw/a:1"]


def test_terminal_bench_probes_with_the_runs_tasks_first_and_checks_the_hosts(monkeypatch,
                                                                              tmp_path):
    _harbor_cache(tmp_path, monkeypatch, {"a-task": "alexgshaw/a:1", "b-task": "alexgshaw/b:1",
                                          "c-task": "alexgshaw/c:1"})
    seen = _record(monkeypatch)
    TB.require_task_metadata_blocked(["terminal-bench/c-task"])
    _, images, kw = seen[0]
    assert kw.get("pull") is None
    assert images == ["alexgshaw/c:1", "alexgshaw/a:1", "alexgshaw/b:1", TB.harbor_image()]
    assert tuple(kw["resolve"]) == C.TASK_INSTALL_HOSTS
    assert C.TASK_INSTALL_HOSTS == ("archive.ubuntu.com", "deb.debian.org", "astral.sh",
                                    "github.com", "pypi.org", "files.pythonhosted.org")


def test_terminal_bench_without_a_local_task_image_names_the_pull(monkeypatch, tmp_path):
    # no task image and no harness image on the host: the check names every candidate and
    # pulls nothing
    _harbor_cache(tmp_path, monkeypatch, {"a-task": "alexgshaw/a:1", "b-task": "alexgshaw/b:1"})
    fake = _install(monkeypatch, FakeDocker())
    with pytest.raises(RuntimeError) as err:
        TB.require_task_metadata_blocked(["terminal-bench/b-task"])
    text = str(err.value)
    assert "alexgshaw/b:1" in text and TB.harbor_image() in text
    assert "Any task image of the dataset" in text
    assert not fake.of("docker", "pull") and not fake.of("docker", "run")


def test_a_shard_run_hands_its_own_tasks_to_the_gate(monkeypatch):
    monkeypatch.setenv("GBENCH_SHARD", "2/2")
    monkeypatch.setattr(TB, "check_terminal_bench_prerequisites", lambda: (True, ""))
    names = [f"terminal-bench/t{i}" for i in range(6)]
    monkeypatch.setattr(TB, "_dataset_task_names", lambda dataset: names)
    got = []

    def _gate(task_names=None):
        got.append(task_names)
        raise RuntimeError(STOP)
    monkeypatch.setattr(TB, "require_task_metadata_blocked", _gate)
    monkeypatch.setattr(TB.subprocess, "Popen", lambda *a, **k: pytest.fail("Harbor started"))
    with pytest.raises(RuntimeError, match="gate says stop"):
        TB.run_terminal_bench("m", "http://h:1/v1", 1)
    assert got == [names[1::2]]


def _trial(root, name, reward, verifier="", agent="ls -la"):
    trial = os.path.join(root, "job", name)
    os.makedirs(os.path.join(trial, "verifier"))
    os.makedirs(os.path.join(trial, "agent"))
    with open(os.path.join(trial, "verifier", "test-stdout.txt"), "w") as f:
        f.write(verifier)
    with open(os.path.join(trial, "agent", "trajectory.json"), "w") as f:
        json.dump({"steps": [{"message": agent}]}, f)
    with open(os.path.join(trial, "result.json"), "w") as f:
        json.dump({"trial_name": name, "verifier_result": {"rewards": {"reward": reward}}}, f)


def _run_harbor(monkeypatch, trials):
    monkeypatch.setattr(TB, "check_terminal_bench_prerequisites", lambda: (True, ""))
    # A complete run of a dataset of exactly these tasks, one trial each: the run counts every
    # expected trial (FULL_TASK_COUNT x the trials per task) without a result as a harness error.
    monkeypatch.setattr(TB, "FULL_TASK_COUNT", len(trials))
    monkeypatch.setattr(TB, "TERMINAL_BENCH_TRIALS", 1)
    monkeypatch.setattr(TB, "require_task_metadata_blocked", lambda task_names=None: None)
    monkeypatch.setattr(TB, "prepare_task_images",
                        lambda task_names=None: ({"cached": 0}, {"present": 0}))
    # A lone DNS failure needs corroboration: the fresh probe after the run still cannot resolve
    # the hosts here (test_swe_runtime_hosts_and_dns_corroboration covers the other outcomes).
    monkeypatch.setattr(TB, "dns_recheck", lambda task_names, hosts: list(hosts))

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
            _trial(jobs, *args)
        return _Proc()
    monkeypatch.setattr(TB.subprocess, "Popen", _popen)
    return TB.run_terminal_bench("m", "http://h:1/v1", 1)


def test_a_verifier_dns_failure_is_a_harness_error_left_out_of_the_score(monkeypatch):
    res = _run_harbor(monkeypatch, [
        ("t-pass", 1.0, PYTEST_RAN.replace("FAILED", "PASSED")),
        ("t-dns", 0.0, APT_OUT + "E: Unable to locate package curl\n" + CURL_OUT),
        ("t-model", 0.0, APT_OUT + PYTEST_RAN),
        ("t-resolver", 0.0, UV_OUT, "echo 'nameserver 1.1.1.1' > /etc/resolv.conf"),
    ])
    assert res["status"] == "completed_with_errors"
    assert res["total_questions"] == 3 and res["correct_answers"] == 1
    assert res["accuracy"] == pytest.approx(33.33)
    assert res["harness_errors"] == {"dns": 1} and res["harness_error_trials"] == ["t-dns"]
    assert res["failed_requests"] == 1 and res["partial"] is True
    assert "lost to harness errors (DNS failures" in res["leaderboard_comparable_reason"]
    by = {t["trial_name"]: t for t in res["sample_traces"]}
    assert by["t-dns"]["harness_error"] == "dns"
    assert by["t-dns"]["dns_failure_hosts"] == ["archive.ubuntu.com", "astral.sh",
                                                "security.ubuntu.com"]
    assert "harness_error" not in by["t-model"] and "dns_failure_hosts" not in by["t-model"]
    assert "harness_error" not in by["t-resolver"]
    assert by["t-resolver"]["dns_failure_hosts"] == ["pypi.org"]
    assert "stays the model's outcome" in by["t-resolver"]["dns_failure_note"]


def test_a_run_of_nothing_but_dns_harness_errors_withholds_its_score(monkeypatch):
    with pytest.raises(RuntimeError) as err:
        _run_harbor(monkeypatch, [("t1", 0.0, APT_OUT), ("t2", 0.0, PIP_OUT)])
    text = str(err.value)
    assert text.startswith("[INFRA REQUIRED] terminal_bench")
    assert "every one of the 2 trial(s) lost its verifier to a DNS failure" in text
    assert "archive.ubuntu.com" in text and "pypi.org" in text


def test_a_healthy_run_stays_success_with_no_harness_errors(monkeypatch):
    res = _run_harbor(monkeypatch, [("t1", 1.0, PYTEST_RAN), ("t2", 0.0, PYTEST_RAN)])
    assert res["status"] == "success" and res["harness_errors"] == {}
    assert res["total_questions"] == 2 and res["failed_requests"] == 0


def test_dns_failure_hosts_on_real_installer_output():
    assert C.dns_failure_hosts(APT_OUT) == ["archive.ubuntu.com", "security.ubuntu.com"]
    assert C.dns_failure_hosts(CURL_OUT) == ["astral.sh"]
    assert C.dns_failure_hosts(PIP_OUT) == ["pypi.org"]
    assert C.dns_failure_hosts(UV_OUT) == ["pypi.org"]
    # a name outside the install domains (the task's own service, say) is not an install host
    assert C.dns_failure_hosts("curl: (6) Could not resolve host: myservice\n") == []
    # once the tests ran, their verdict stands
    assert C.verifier_dns_failure(APT_OUT + PYTEST_RAN) == []
    assert C.verifier_dns_failure(APT_OUT) == ["archive.ubuntu.com", "security.ubuntu.com"]
    assert C.touched_resolver("cat /etc/hosts") and C.touched_resolver("vi /etc/resolv.conf")
    assert not C.touched_resolver("ls /etc && cat /etc/hostname")


# ============================================================================================
# skillsbench: the same hosts, and harness errors out of the means
# ============================================================================================
def _run_skillsbench(monkeypatch, per_task, verifier):
    monkeypatch.setattr(SB, "check_skillsbench_prerequisites", lambda: (True, ""))
    # A lone DNS failure needs corroboration: the fresh probe after the run still cannot resolve
    # the hosts here (test_swe_runtime_hosts_and_dns_corroboration covers the other outcomes).
    monkeypatch.setattr(SB, "dns_recheck", lambda hosts: list(hosts))
    seen = _record(monkeypatch)

    def _run(cmd, **kw):
        if cmd[:2] == ["docker", "run"]:
            workdir = cmd[cmd.index("-v", cmd.index("-v") + 1) + 1].split(":")[0]
            vals = [v for v in per_task.values() if isinstance(v, (int, float))]
            summary = {"conditions": {"without-skills": {
                "mean_reward": sum(vals) / len(vals) if vals else None,
                "n_scored": len(vals), "n_tasks": len(per_task), "per_task": per_task}}}
            with open(os.path.join(workdir, "skillsbench_summary.json"), "w") as f:
                json.dump(summary, f)
            for task, text in verifier.items():
                vdir = os.path.join(workdir, "jobs", "without-skills", task, "job-1", "rollout-1",
                                    "verifier")
                os.makedirs(vdir)
                with open(os.path.join(vdir, "test-stdout.txt"), "w") as f:
                    f.write(text)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(SB.subprocess, "run", _run)
    return SB.run_skillsbench("m", "http://127.0.0.1:8000/v1", 1), seen


def test_skillsbench_checks_the_install_hosts_and_leaves_harness_errors_out(monkeypatch):
    res, seen = _run_skillsbench(monkeypatch, {"a": 1.0, "b": 0.0, "c": None, "d": 0.0},
                                 {"b": CURL_OUT, "d": APT_OUT + PYTEST_RAN})
    assert tuple(seen[0][2]["resolve"]) == C.TASK_INSTALL_HOSTS
    assert res["status"] == "completed_with_errors"
    assert res["harness_errors"] == {"dns": 1, "no_reward": 1}
    assert res["harness_error_tasks"] == {"without-skills": {
        "b": {"kind": "dns", "hosts": ["astral.sh"]}, "c": {"kind": "no_reward", "stage": None,
                                                 "note": "the rollout wrote no result.json"}}}
    # a and d are scored (d's tests ran): mean 0.5, not 1/3 and not 0.25
    assert res["mean_reward"] == 0.5 and res["n_scored"] == 2 and res["accuracy"] == 50.0
    assert "lost to harness errors" in res["leaderboard_comparable_reason"]


def test_skillsbench_with_no_task_left_to_score_withholds_its_score(monkeypatch):
    with pytest.raises(RuntimeError) as err:
        _run_skillsbench(monkeypatch, {"a": 0.0, "b": None}, {"a": UV_OUT})
    text = str(err.value)
    assert "no task was scored" in text and "could not resolve pypi.org" in text


# ============================================================================================
# swe_bench_live's Windows tab: a PowerShell probe on the Windows-containers engine
# ============================================================================================
class FakeWinDocker:
    """subprocess.run stand-in for a docker CLI whose engine reports `os_type`. `answers`:
    program -> (docker exit code, stdout) of the probe."""

    def __init__(self, os_type="windows", info_rc=0, present=(), refs=None, answers=None,
                 timeout=False):
        self.calls = []
        self.os_type, self.info_rc = os_type, info_rc
        self.present, self.refs = set(present), dict(refs or {})
        self.answers = dict(answers or {})
        self.timeout = timeout

    def __call__(self, argv, **kw):
        argv = list(argv)
        self.calls.append((argv, kw))
        if argv[:2] == ["docker", "info"]:
            return _cp(argv, self.info_rc, self.os_type + "\n", "" if not self.info_rc else
                       "Cannot connect to the Docker daemon")
        if argv[:3] == ["docker", "image", "inspect"]:
            return _cp(argv, 0 if argv[-1] in self.present else 1)
        if argv[:2] == ["docker", "images"]:
            ref = argv[argv.index("--filter") + 1].split("=", 1)[1]
            return _cp(argv, 0, "".join(f"{i}\n" for i in self.refs.get(ref, [])))
        if argv[:3] == ["docker", "rm", "-f"]:
            return _cp(argv, 0)
        if argv[:2] == ["docker", "run"]:
            if self.timeout:
                raise subprocess.TimeoutExpired(argv, kw.get("timeout"))
            program = argv[argv.index("--entrypoint") + 1]
            rc, out = self.answers.get(program, (127, ""))
            return _cp(argv, rc, out, "" if out else f'exec: "{program}": executable file not '
                       "found")
        if argv[:2] == ["docker", "pull"]:
            raise AssertionError("the Windows probe pulled an image")
        raise AssertionError(f"unexpected command {argv}")

    def runs(self):
        return [a for a, _ in self.calls if a[:2] == ["docker", "run"]]


WIN_IMAGE = "starryzhang/sweb.eval.win.libsdl-org_1776_sdl-11761"


def test_on_a_linux_engine_the_windows_check_fails_with_that_reason(monkeypatch):
    fake = _install(monkeypatch, FakeWinDocker(os_type="linux", present={WIN_IMAGE}))
    ok, why = C.check_windows_task_network("swe_bench_live", [WIN_IMAGE])
    assert not ok and "runs linux containers (`docker info` OSType 'linux')" in why
    assert "Windows-containers engine" in why and not fake.runs()


def test_an_unreadable_engine_fails_closed(monkeypatch):
    fake = _install(monkeypatch, FakeWinDocker(info_rc=1))
    ok, why = C.check_windows_task_network("swe_bench_live", [WIN_IMAGE])
    assert not ok and "could not read the Docker engine's OSType" in why and not fake.runs()


def test_without_a_local_windows_image_it_names_the_pull(monkeypatch):
    fake = _install(monkeypatch, FakeWinDocker())
    ok, why = C.check_windows_task_network("swe_bench_live", [WIN_IMAGE, "other"],
                                           references=[LV.WINDOWS_IMAGE_REFERENCE])
    assert not ok and f"`docker pull {WIN_IMAGE}`" in why and "never pulls one" in why
    assert not fake.runs()


def test_an_unreachable_verdict_passes_and_is_cached(monkeypatch):
    fake = _install(monkeypatch, FakeWinDocker(present={WIN_IMAGE}, answers={
        "powershell": (0, "GBENCH_METADATA_PROBE=unreachable timeout\n")}))
    assert C.check_windows_task_network("swe_bench_live", ["absent", WIN_IMAGE]) == (True, "")
    [argv] = fake.runs()
    assert argv[argv.index("--pull") + 1] == "never" and "--network" not in argv
    assert argv[argv.index("--entrypoint") + 1] == "powershell"
    assert argv[argv.index("--label") + 1] == CG.PROBE_LABEL
    assert argv[-6:-1] == [WIN_IMAGE, "-NoLogo", "-NoProfile", "-NonInteractive",
                           "-EncodedCommand"]
    script = base64.b64decode(argv[-1]).decode("utf-16-le")
    assert script == C.windows_probe_script()
    assert "ConnectAsync('169.254.169.254', 80).Wait(3000)" in script
    before = len(fake.calls)
    assert C.check_windows_task_network("swe_bench_live", [WIN_IMAGE]) == (True, "")
    assert len(fake.calls) == before


def test_the_probe_sends_no_request():
    script = C.windows_probe_script()
    for word in ("Invoke-WebRequest", "Invoke-RestMethod", "WebClient", "HttpClient", "GET ",
                 "computeMetadata", "Metadata-Flavor", "GetStream", ".Send"):
        assert word not in script, word
    assert "$c.Close()" in script


def test_a_reachable_verdict_fails_closed_with_the_fix_message(monkeypatch):
    fake = _install(monkeypatch, FakeWinDocker(present={WIN_IMAGE}, answers={
        "powershell": (0, "GBENCH_METADATA_PROBE=reachable\n")}))
    ok, why = C.check_windows_task_network("swe_bench_live", [WIN_IMAGE])
    assert not ok and why == C.windows_reachable_message()
    assert "every port but 53" in why and "without cutting the VM's own access" in why
    assert "never changes the host's firewall" in why
    ok, _ = C.check_windows_task_network("swe_bench_live", [WIN_IMAGE])      # not cached
    assert not ok and len(fake.runs()) == 2


def test_without_powershell_it_tries_pwsh_and_fails_closed_without_either(monkeypatch):
    fake = _install(monkeypatch, FakeWinDocker(present={WIN_IMAGE}, answers={
        "pwsh": (0, "GBENCH_METADATA_PROBE=unreachable ConnectionRefused\n")}))
    assert C.check_windows_task_network("swe_bench_live", [WIN_IMAGE]) == (True, "")
    assert [a[a.index("--entrypoint") + 1] for a in fake.runs()] == ["powershell", "pwsh"]
    C.reset_task_network_verdicts()
    _install(monkeypatch, FakeWinDocker(present={WIN_IMAGE}))
    ok, why = C.check_windows_task_network("swe_bench_live", [WIN_IMAGE])
    assert not ok and "neither powershell nor pwsh answered" in why
    assert "powershell: docker exit 127" in why and "pwsh: docker exit 127" in why


def test_an_error_verdict_or_a_docker_timeout_fails_closed(monkeypatch):
    _install(monkeypatch, FakeWinDocker(present={WIN_IMAGE}, answers={
        "powershell": (0, "GBENCH_METADATA_PROBE=error AddressFamilyNotSupported\n")}))
    ok, why = C.check_windows_task_network("swe_bench_live", [WIN_IMAGE])
    assert not ok and "gave no verdict (error AddressFamilyNotSupported)" in why
    fake = _install(monkeypatch, FakeWinDocker(present={WIN_IMAGE}, timeout=True))
    ok, why = C.check_windows_task_network("swe_bench_live", [WIN_IMAGE])
    assert not ok and f"did not finish within {CG.PROBE_DOCKER_TIMEOUT_S} s" in why
    rm = [(a, kw) for a, kw in fake.calls if a[:3] == ["docker", "rm", "-f"]]
    assert rm and rm[0][1]["timeout"] == CG.PROBE_RM_TIMEOUT_S


def test_the_windows_tab_probes_with_its_own_instance_images(monkeypatch):
    rows = [{"instance_id": "i1", "docker_image": WIN_IMAGE},
            {"instance_id": "i2", "docker_image": "starryzhang/sweb.eval.win.x"},
            {"instance_id": "i3", "docker_image": WIN_IMAGE}]
    monkeypatch.setattr(LV, "load_setting_rows", lambda setting: rows)
    fake = _install(monkeypatch, FakeWinDocker(
        refs={LV.WINDOWS_IMAGE_REFERENCE: ["starryzhang/sweb.eval.win.z:latest"]},
        answers={"powershell": (0, "GBENCH_METADATA_PROBE=unreachable timeout\n")}))
    assert LV.check_windows_metadata_guard() == (True, "")
    inspected = [a[-1] for a, _ in fake.calls if a[:3] == ["docker", "image", "inspect"]]
    assert inspected == [WIN_IMAGE, "starryzhang/sweb.eval.win.x"]
    assert "starryzhang/sweb.eval.win.z:latest" in fake.runs()[0]


def test_the_windows_tab_runs_on_a_windows_engine_once_its_check_passes(monkeypatch):
    monkeypatch.setattr(LV, "check_windows_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(LV, "check_swe_bench_live_prerequisites", lambda settings: (True, ""))
    monkeypatch.setattr(LV, "load_setting_rows", lambda setting: [
        {"instance_id": "i1", "docker_image": WIN_IMAGE}])
    _install(monkeypatch, FakeWinDocker(present={WIN_IMAGE}, answers={
        "powershell": (0, "GBENCH_METADATA_PROBE=unreachable timeout\n")}))
    assert LV.missing_prerequisites(["windows"], A.AGENTIC) == {}
    ran = []
    monkeypatch.setattr(LV, "_run_setting", lambda setting, *a, **k: ran.append(setting) or {
        "status": "success", "accuracy": 10.0, "total_questions": 1, "correct_answers": 0,
        "leaderboard_comparable": False, "leaderboard_comparable_reason": "x",
        "tab": "Windows", "sample_traces": []})
    res = LV.run_swe_bench_live("m", "http://h:1/v1", 1, split="windows")
    assert ran == ["windows"] and res["status"] == "success"


def test_on_a_linux_engine_the_windows_tab_stays_infra_required(monkeypatch):
    monkeypatch.setattr(LV, "check_windows_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(LV, "check_swe_bench_live_prerequisites", lambda settings: (True, ""))
    monkeypatch.setattr(LV, "load_setting_rows", lambda setting: [
        {"instance_id": "i1", "docker_image": WIN_IMAGE}])
    _install(monkeypatch, FakeWinDocker(os_type="linux", present={WIN_IMAGE}))
    missing = LV.missing_prerequisites(["windows"], A.AGENTIC)
    assert "runs linux containers" in missing["windows"]
    monkeypatch.setattr(LV, "_run_setting", lambda *a, **k: pytest.fail("the tab started"))
    with pytest.raises(RuntimeError, match=r"INFRA REQUIRED.*runs linux containers"):
        LV.run_swe_bench_live("m", "http://h:1/v1", 1, split="windows")


def test_the_agentic_linux_tabs_check_the_swe_rex_build_hosts(monkeypatch, tmp_path):
    seen = _record(monkeypatch, stop=True)
    monkeypatch.setattr(LV.subprocess, "run", lambda *a, **k: pytest.fail("a process ran"))
    task = A.AgentTask(instance_id="o__r-1", problem_statement="p", image="i:1", category="c")
    with pytest.raises(RuntimeError, match="gate says stop"):
        LV.run_sweagent([task], [], "m", "http://127.0.0.1:1/v1", 1, str(tmp_path))
    assert tuple(seen[0][2]["resolve"]) == LV.SWEREX_BUILD_HOSTS
    monkeypatch.setattr(LV, "missing_prerequisites", lambda settings, mode: {})
    monkeypatch.setattr(LV, "_run_setting", lambda *a, **k: pytest.fail("a tab started"))
    with pytest.raises(RuntimeError, match="gate says stop"):
        LV.run_swe_bench_live("m", "http://h:1/v1", 1, eval_categories="lite")
    assert tuple(seen[1][2]["resolve"]) == LV.SWEREX_BUILD_HOSTS


# ============================================================================================
# the docs say what the code does
# ============================================================================================
def _doc(name):
    return " ".join(open(os.path.join(REPO, "docs", "evals", f"{name}.md"),
                         encoding="utf-8").read().split())


def test_the_docs_describe_the_checks_as_coded():
    live = _doc("swe_bench_live")
    assert "check_windows_metadata_guard" in live and "-EncodedCommand" in live
    assert f"docker pull {WIN_IMAGE}" in live and "`starryzhang/sweb.eval.win.*`" in live
    assert "cannot probe" not in live and "never runs" not in live.replace(
        "gbench never runs sudo", "")
    for host in LV.SWEREX_BUILD_HOSTS:
        assert f"`{host}`" in live, host
    tb = _doc("terminal_bench")
    assert "the run pulls it anyway" not in tb and "nothing is pulled for it" in tb
    assert "the harness image (the network verdict does not depend on the image)" in tb
    sb = _doc("skillsbench")
    for doc in (tb, sb):
        for host in C.TASK_INSTALL_HOSTS:
            assert f"`{host}`" in doc, host
        for domain in C.INSTALL_DOMAINS:
            assert f"`{domain}`" in doc, domain
    assert '`harness_error: "dns"`' in tb and "`harness_error_tasks`" in sb
    mswe = _doc("multi_swe_bench")
    assert "which gbench pulls on the host" not in mswe
    assert "Nothing is pulled for the probe" in mswe
