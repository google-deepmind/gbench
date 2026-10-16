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

"""The cloud metadata server gate of the SWE family and the agent suites, and skillsbench's key.

Docker is stubbed everywhere: no container or network starts and nothing contacts
169.254.169.254.

* swebench_common.require_task_network_blocked probes docker's default bridge and a user-defined
  bridge network (one gbench creates for the probe and removes), with an image the suite already
  has; each verdict is cached; a reachable network, no usable image or a network it cannot create
  is infra_required. Nothing is pulled for the probe: without a local image the gate names the
  `docker pull` of an image the run itself starts.
* Every suite calls it before the containers where the model's code runs start: when it raises,
  nothing starts.
* swe_bench_live's Windows tab runs only after its own Windows check passed
  (tests/test_guard_swe_dns_and_windows_probe.py covers that check over a fake docker).
* skillsbench: no key on any command line or in any container; with a key, a relay adds it.
"""

import asyncio
import json
import os
import subprocess
import sys

import pytest

from gbench.runners.eval_suites import container_guard as CG
from gbench.runners.eval_suites import multi_swe_bench as M
from gbench.runners.eval_suites import skillsbench as SB
from gbench.runners.eval_suites import swe_agentic as A
from gbench.runners.eval_suites import swe_bench_live as LV
from gbench.runners.eval_suites import swe_bench_pro as P
from gbench.runners.eval_suites import swe_lancer as SL
from gbench.runners.eval_suites import swebench_common as C
from gbench.runners.eval_suites import terminal_bench as TB
from gbench.runners.eval_suites import ui_control_osworld as OS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STOP = "[INFRA REQUIRED] gate says stop"
#: The real gate, taken at import (before any fixture could stub it), so these tests exercise it
#: even where a broader fixture stubs it for other test files.
_REAL_GATE = C.require_task_network_blocked
SECRET = "sk-skillsbench-endpoint-key-0123456789"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch, tmp_path):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("GBENCH_RESULTS_DIR", raising=False)
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    for var in ("OPENAI_API_KEY", "GBENCH_SHARD", "GBENCH_SKILLSBENCH_API_KEY",
                "GBENCH_SKILLSBENCH_TASK_HOST", "GBENCH_SKILLSBENCH_TASK_ENDPOINT",
                "GBENCH_SWEBENCH_PRO_AGENTIC", "GBENCH_SWE_BENCH_PRO_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(C, "require_task_network_blocked", _REAL_GATE)
    CG.reset_verdicts()
    C.reset_task_network_verdicts()
    yield
    CG.reset_verdicts()
    C.reset_task_network_verdicts()


# ============================================================================================
# the gate itself, over a fake docker
# ============================================================================================
class FakeDocker:
    """subprocess.run stand-in for the docker CLI. `present`: local images; `reach`: network
    name (or prefix) -> the probe's verdict word."""

    def __init__(self, present=(), refs=None, reach=None, create_rc=0, pull_rc=0):
        self.calls = []
        self.present = set(present)
        self.refs = dict(refs or {})
        self.reach = dict(reach or {})
        self.create_rc = create_rc
        self.pull_rc = pull_rc

    def _verdict(self, net):
        for key, word in self.reach.items():
            if net == key or net.startswith(key):
                return word
        return "unreachable"

    def __call__(self, argv, **kw):
        argv = list(argv)
        self.calls.append(argv)
        cp = lambda rc, out="": subprocess.CompletedProcess(argv, rc, out, "")  # noqa: E731
        if argv[:3] == ["docker", "image", "inspect"]:
            return cp(0 if argv[-1] in self.present else 1, "sha256:x\n")
        if argv[:2] == ["docker", "images"]:
            ref = argv[argv.index("--filter") + 1].split("=", 1)[1]
            return cp(0, "".join(f"{i}\n" for i in self.refs.get(ref, [])))
        if argv[:3] == ["docker", "network", "create"]:
            return cp(self.create_rc)
        if argv[:3] == ["docker", "network", "rm"]:
            return cp(0)
        if argv[:2] == ["docker", "pull"]:
            if self.pull_rc == 0:
                self.present.add(argv[-1])
            return cp(self.pull_rc)
        if argv[:2] == ["docker", "run"]:
            net = argv[argv.index("--network") + 1]
            return cp(0, f"GBENCH_METADATA_PROBE={self._verdict(net)}\n")
        raise AssertionError(f"unexpected command {argv}")

    def runs(self):
        return [c for c in self.calls if c[:2] == ["docker", "run"]]

    def of(self, *prefix):
        return [c for c in self.calls if c[:len(prefix)] == list(prefix)]


def _install(monkeypatch, fake):
    monkeypatch.setattr(C.subprocess, "run", fake)
    return fake


def test_both_task_networks_are_probed_once_and_then_cached(monkeypatch):
    fake = _install(monkeypatch, FakeDocker(present={"img"}))
    assert C.TASK_NETWORKS == ("bridge", C.USER_DEFINED_NETWORK)
    C.require_task_network_blocked("suite", ["missing", "img"])
    runs = fake.runs()
    assert [r[r.index("--network") + 1] for r in runs][0] == "bridge"
    user_net = runs[1][runs[1].index("--network") + 1]
    assert user_net.startswith("gbench-metadata-probe-") and len(runs) == 2
    # the probe ran in the first local image, never pulling
    assert all(r[r.index("--pull") + 1] == "never" and "img" in r for r in runs)
    create = fake.of("docker", "network", "create")[0]
    assert create[-1] == user_net and CG.PROBE_LABEL in create and "bridge" in create
    assert fake.of("docker", "network", "rm") == [["docker", "network", "rm", user_net]]
    assert not fake.of("docker", "pull")
    # settled: a later call starts nothing
    before = len(fake.calls)
    C.require_task_network_blocked("suite", ["img"])
    assert len(fake.calls) == before


def test_a_reachable_bridge_is_infra_required_naming_the_rule(monkeypatch):
    _install(monkeypatch, FakeDocker(present={"img"}, reach={"bridge": "reachable"}))
    with pytest.raises(RuntimeError) as err:
        C.require_task_network_blocked("suite", ["img"], docs_url="docs/evals/suite.md")
    text = str(err.value)
    assert text.startswith("[INFRA REQUIRED] suite")
    assert CG.FIREWALL_COMMAND in text and CG.PERSIST_COMMAND in text
    assert "docs/evals/suite.md" in text


def test_a_reachable_user_defined_network_is_infra_required_and_removed(monkeypatch):
    fake = _install(monkeypatch, FakeDocker(present={"img"},
                                            reach={"gbench-metadata-probe-": "reachable"}))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*DOCKER-USER"):
        C.require_task_network_blocked("suite", ["img"])
    assert len(fake.of("docker", "network", "rm")) == 1      # removed even when it failed
    with pytest.raises(RuntimeError):                         # and not cached as a pass
        C.require_task_network_blocked("suite", ["img"])


def test_a_network_it_cannot_create_is_infra_required(monkeypatch):
    _install(monkeypatch, FakeDocker(present={"img"}, create_rc=1))
    with pytest.raises(RuntimeError, match="could not create the user-defined docker network"):
        C.require_task_network_blocked("suite", ["img"])


def test_without_a_local_image_it_is_infra_required_and_nothing_is_pulled(monkeypatch):
    fake = _install(monkeypatch, FakeDocker())
    with pytest.raises(RuntimeError) as err:
        C.require_task_network_blocked("suite", ["gbench-x"], references=["repo/*"],
                                       hint="Build gbench-x.")
    text = str(err.value)
    assert "gbench-x" in text and "repo/* (any tag)" in text and "Build gbench-x." in text
    assert not fake.runs() and not fake.of("docker", "pull")


def test_a_reference_finds_a_local_image(monkeypatch):
    fake = _install(monkeypatch, FakeDocker(refs={"mswebench/*": ["mswebench/a_m_b:pr-1"]}))
    C.require_task_network_blocked("suite", ["gbench-absent"], references=["mswebench/*"])
    assert all("mswebench/a_m_b:pr-1" in r for r in fake.runs()) and fake.runs()


def test_the_runs_own_image_is_used_when_present_and_never_pulled(monkeypatch):
    fake = _install(monkeypatch, FakeDocker(present={"task:1"}))
    C.require_task_network_blocked("suite", ["absent:1"], pull="task:1")
    assert not fake.of("docker", "pull")
    assert fake.runs() and all("task:1" in r and "never" in r for r in fake.runs())
    CG.reset_verdicts()
    C.reset_task_network_verdicts()
    missing = _install(monkeypatch, FakeDocker())
    with pytest.raises(RuntimeError) as err:
        C.require_task_network_blocked("suite", ["absent:1"], pull="task:1", hint="Build it.")
    assert "Pull one on this host first: `docker pull task:1`." in str(err.value)
    assert "Build it." in str(err.value)
    assert not missing.runs() and not missing.of("docker", "pull")


def test_network_none_needs_nothing(monkeypatch):
    fake = _install(monkeypatch, FakeDocker())
    C.require_task_network_blocked("suite", [], networks=("none",))
    assert fake.calls == []


def test_the_host_network_named_by_a_task_is_probed_too(monkeypatch):
    assert A.run_args_networks(["--rm", "--network", "host"]) == ["host"]
    assert A.run_args_networks(["--net=my-net", "--rm"]) == ["my-net"]
    assert A.run_args_networks(["--rm"]) == [] and A.run_args_networks(None) == []
    fake = _install(monkeypatch, FakeDocker(present={"img:1"}, reach={"host": "reachable"}))
    task = A.AgentTask(instance_id="i", problem_statement="p", image="img:1", category="c",
                       run_args=["--rm", "--network", "host"])
    with pytest.raises(RuntimeError) as err:
        A.require_rollout_network_blocked("suite", [task])
    assert "does not cover it" in str(err.value) and CG.FIREWALL_COMMAND in str(err.value)
    assert "host" in [r[r.index("--network") + 1] for r in fake.runs()]


def test_the_harness_probe_uses_the_harness_image_else_a_pulled_instance_image(monkeypatch):
    seen = []
    monkeypatch.setattr(C, "require_task_network_blocked",
                        lambda name, images=(), **kw: seen.append((name, list(images), kw)))
    C.require_harness_network_blocked("copilot_bench_swe", "gbench-swebench-v5:5.0.2", ["a__b-1"])
    assert seen[-1][1] == ["gbench-swebench-v5:5.0.2"] and "pull" not in seen[-1][2]
    C.require_harness_network_blocked("x", None, ["a__b-1"], namespace="swebench")
    image = A.swebench_image("a__b-1", "swebench")
    assert seen[-1][1] == [image] and seen[-1][2]["pull"] == image


# ============================================================================================
# every suite: the gate runs before the containers where the model's code runs
# ============================================================================================
def _stop(monkeypatch, record=None):
    def _gate(name, images=(), **kw):
        if record is not None:
            record.append((name, list(images), kw))
        raise RuntimeError(STOP)
    monkeypatch.setattr(C, "require_task_network_blocked", _gate)


def _record(monkeypatch):
    seen = []
    monkeypatch.setattr(C, "require_task_network_blocked",
                        lambda name, images=(), **kw: seen.append((name, list(images), kw)))
    return seen


def _no_process(monkeypatch, *mods):
    def _fail(*a, **k):
        pytest.fail(f"a process started before the gate: {a[:1]}")
    for mod in mods:
        monkeypatch.setattr(mod.subprocess, "run", _fail)
        monkeypatch.setattr(mod.subprocess, "Popen", _fail)


def _task(iid="o__r-1", image="docker.io/swebench/sweb.eval.x86_64.o_1776_r-1:latest"):
    return A.AgentTask(instance_id=iid, problem_statement="fix it", image=image, category="o/r")


def test_agent_rollouts_wait_for_the_gate(monkeypatch):
    monkeypatch.setattr(A, "load_agent_config", lambda: ({}, "0" * 64, "1.17.4"))
    monkeypatch.setattr(A, "run_rollouts", lambda *a, **k: pytest.fail("a rollout started"))
    seen = []
    _stop(monkeypatch, seen)
    with pytest.raises(RuntimeError, match="gate says stop"):
        A.run_agentic("copilot_bench_swe", "m", "http://h:1/v1", 1, [_task()],
                      lambda traces: None, {})
    name, images, kw = seen[0]
    assert name == "copilot_bench_swe" and images == [_task().image]
    assert kw["pull"] == _task().image and set(C.TASK_NETWORKS) <= set(kw["networks"])


def test_agent_v2_rollouts_probe_with_the_agent_image(monkeypatch):
    monkeypatch.setattr(A, "describe_agent_v2", lambda v: {
        "config": {}, "config_sha256": A.AGENT_V2_CONFIG_SHA256["2.4.6"], "version": "2.4.6"})
    monkeypatch.setattr(A, "run_rollouts", lambda *a, **k: pytest.fail("a rollout started"))
    seen = []
    _stop(monkeypatch, seen)
    with pytest.raises(RuntimeError, match="gate says stop"):
        A.run_agentic("swe_bench_multilingual", "m", "http://h:1/v1", 1, [_task()],
                      lambda traces: None, {}, agent_v2="2.4.6")
    assert seen[0][1] == [A.agent_v2_image()] and "pull" not in seen[0][2]


def test_the_swebench_harness_waits_for_the_gate(monkeypatch):
    _no_process(monkeypatch, C)
    monkeypatch.setattr(C, "host_pull_images", lambda *a, **k: pytest.fail("pulled first"))
    seen = []
    _stop(monkeypatch, seen)
    scorer = C.make_swebench_scorer("swe_bench_live", "m", "SWE-bench-Live/SWE-bench-Live",
                                    "lite", "starryzhang", 1, {},
                                    harness_image="gbench-swe-bench-live:a145aa87",
                                    instance_image=LV.python_instance_image)
    traces = [{"extra_payload": {"instance_id": "repo__1"}, "agent_patch": "diff --git a b\n"}]
    with pytest.raises(RuntimeError, match="gate says stop"):
        asyncio.run(scorer(traces))
    assert seen[0][1] == ["gbench-swe-bench-live:a145aa87"]


def test_one_shot_swebench_checks_before_any_model_call(monkeypatch):
    monkeypatch.setattr(C, "run_eval_suite", lambda **k: pytest.fail("a model call ran"))
    _stop(monkeypatch)
    with pytest.raises(RuntimeError, match="gate says stop"):
        C.execute_swebench("copilot_bench_swe", "m", "http://h:1/v1", 1, "d", "test",
                           "swebench", harness_image="gbench-swebench-v5:5.0.2", harness_v5=True)


def test_sweagent_batches_wait_for_the_gate(monkeypatch, tmp_path):
    _no_process(monkeypatch, LV)
    monkeypatch.setattr(LV, "host_pull_images", lambda *a, **k: pytest.fail("pulled first"))
    seen = []
    _stop(monkeypatch, seen)
    with pytest.raises(RuntimeError, match="gate says stop"):
        LV.run_sweagent([_task()], [], "m", "http://127.0.0.1:1/v1", 1, str(tmp_path))
    assert seen[0][1] == [LV.sweagent_image()] and seen[0][2]["docs_url"] == LV.DOCS_URL


def test_the_multilang_harness_waits_for_the_gate(monkeypatch):
    _no_process(monkeypatch, LV)
    monkeypatch.setattr(LV, "host_pull_images", lambda *a, **k: pytest.fail("pulled first"))
    seen = []
    _stop(monkeypatch, seen)
    rows = [{"instance_id": "g-org__repo-0", "docker_image": "starryzhang/x", "_split": "go"}]
    scorer = LV.make_launch_scorer("go", rows, "linux", 1, {})
    with pytest.raises(RuntimeError, match="gate says stop"):
        asyncio.run(scorer([{"extra_payload": {"instance_id": "g-org__repo-0"},
                             "agent_patch": "diff --git a b\n"}]))
    assert seen[0][1] == [LV._image()]


def test_swe_bench_live_checks_before_the_first_linux_tab(monkeypatch):
    monkeypatch.setattr(LV, "missing_prerequisites", lambda settings, mode: {})
    monkeypatch.setattr(LV, "_run_setting", lambda *a, **k: pytest.fail("a tab started"))
    seen = []
    _stop(monkeypatch, seen)
    with pytest.raises(RuntimeError, match="gate says stop"):
        LV.run_swe_bench_live("m", "http://h:1/v1", 1, eval_categories="lite,go")
    assert seen[0][1] == [LV._image(), LV.sweagent_image()]


def test_the_windows_tab_waits_for_its_own_metadata_check(monkeypatch):
    why = "the cloud metadata server (169.254.169.254:80) is reachable from a Windows container"
    monkeypatch.setattr(LV, "check_windows_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(LV, "check_swe_bench_live_prerequisites", lambda settings: (True, ""))
    monkeypatch.setattr(LV, "check_sweagent_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(LV, "check_windows_metadata_guard", lambda: (False, why))
    missing = LV.missing_prerequisites(list(LV.SETTINGS), A.AGENTIC)
    assert missing == {"windows": why}
    monkeypatch.setattr(LV, "_run_setting", lambda *a, **k: pytest.fail("the tab started"))
    with pytest.raises(RuntimeError) as err:
        LV.run_swe_bench_live("m", "http://h:1/v1", 1, split="windows")
    assert why in str(err.value)
    # the runner itself refuses too
    monkeypatch.setattr(LV, "load_setting_rows", lambda s: pytest.fail("rows loaded"))
    with pytest.raises(RuntimeError, match="INFRA REQUIRED.*reachable from a Windows container"):
        LV._run_windows("windows", A.AGENTIC, "m", "http://h:1/v1", 1, {})


def test_mswe_agent_waits_for_the_gate(monkeypatch, tmp_path):
    _no_process(monkeypatch, M)
    monkeypatch.setattr(M, "host_pull_images", lambda *a, **k: pytest.fail("pulled first"))
    seen = []
    _stop(monkeypatch, seen)
    task = A.AgentTask(instance_id="o__r-1", problem_statement="p", image="mswebench/o_m_r:pr-1",
                       category="go")
    with pytest.raises(RuntimeError, match="gate says stop"):
        M.run_mswe_agent([task], {"o__r-1": {}}, "m", "http://127.0.0.1:1/v1", 1,
                         str(tmp_path), 0.0, 0.95)
    assert seen[0][1] == [M.mswe_agent_image()]


def test_the_multi_swe_bench_harness_waits_for_the_gate(monkeypatch):
    _no_process(monkeypatch, M)
    seen = []
    _stop(monkeypatch, seen)
    scorer = M._make_scorer("m", 1, {})
    traces = [{"extra_payload": {"instance_id": "o__r-1", "org": "O", "repo": "R", "number": 1},
               "agent_patch": "diff --git a b\n"}]
    with pytest.raises(RuntimeError, match="gate says stop"):
        asyncio.run(scorer(traces))
    name, images, kw = seen[0]
    assert images == [M.mswe_agent_image()] and kw["references"] == ["mswebench/*"]
    assert kw["pull"] == "mswebench/o_m_r:pr-1"


def test_swe_bench_pro_checks_before_any_harbor_job(monkeypatch):
    monkeypatch.setattr(P, "_run_v2", lambda *a, **k: pytest.fail("a Harbor job started"))
    monkeypatch.setattr(P, "KeyForwarder", lambda *a, **k: pytest.fail("forwarder started"))
    monkeypatch.setattr(P, "_load_swe_bench_pro_samples", lambda **k: pytest.fail("loaded"))
    seen = []
    _stop(monkeypatch, seen)
    with pytest.raises(RuntimeError, match="gate says stop"):
        P.run_swe_bench_pro_v2("m", "http://h:1/v1", 1)
    with pytest.raises(RuntimeError, match="gate says stop"):
        P._run_v2_oneshot("m", "http://h:1/v1", 1, False)
    assert [s[1] for s in seen] == [[P.v2_image()], [P.v2_image()]]


def test_swe_bench_pro_v1_route_and_scorer_check_first(monkeypatch):
    from gbench.runners.eval_suites import swe_bench_pro_agentic as PA
    monkeypatch.setenv("GBENCH_SWEBENCH_PRO_AGENTIC", "1")
    monkeypatch.setattr(P, "check_swe_bench_pro_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(PA, "execute_agentic", lambda *a, **k: pytest.fail("engine started"))
    seen = []
    _stop(monkeypatch, seen)
    with pytest.raises(RuntimeError, match="gate says stop"):
        P.run_swe_bench_pro("m", "http://h:1/v1", 1)
    assert seen[0][1] == [P.v2_image()] and seen[0][2]["references"] == ["jefzda/sweap-images"]
    _no_process(monkeypatch, P)
    monkeypatch.setattr(P, "_harness_dir", lambda: "/nonexistent")
    monkeypatch.setattr(P, "raw_sample_path", lambda: "/nonexistent/raw.csv")
    with pytest.raises(RuntimeError, match="gate says stop"):
        asyncio.run(P._make_scorer("m", 1, {})([{"extra_payload": {"instance_id": "i"},
                                                 "agent_patch": "diff --git a b\n"}]))


def test_swe_lancer_checks_before_its_first_task(monkeypatch):
    monkeypatch.setattr(SL, "check_swe_lancer_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(SL, "load_tasks", lambda: pytest.fail("the harness image ran"))
    monkeypatch.setattr(SL, "EndpointRelay", lambda *a, **k: pytest.fail("relay started"))
    seen = []
    _stop(monkeypatch, seen)
    with pytest.raises(RuntimeError, match="gate says stop"):
        SL.run_swe_lancer("m", "http://h:1/v1", 1)
    assert seen[0][1] == [SL.image()]


def _harbor_cache(tmp_path, monkeypatch, images):
    """Tasks in gbench's Harbor cache (terminal_bench.harbor_package_cache, the harness
    container's cache on this host; nothing imports Harbor on the host)."""
    root = tmp_path / "packages"
    for i, image in enumerate(images):
        task = root / "terminal-bench" / f"task-{i}"
        task.mkdir(parents=True)
        (task / "task.toml").write_text(f'version = "1.0"\n[environment]\ndocker_image = '
                                        f'"{image}"\ncpus = 1\n')
    (root / "terminal-bench" / "no-toml").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(TB, "harbor_package_cache", lambda: str(root))
    return root


def test_terminal_bench_reads_the_task_images_from_harbors_cache(monkeypatch, tmp_path):
    _harbor_cache(tmp_path, monkeypatch, ["alexgshaw/b:1", "alexgshaw/a:1", "alexgshaw/b:1"])
    assert TB.harbor_task_images() == ["alexgshaw/b:1", "alexgshaw/a:1"]
    seen = _record(monkeypatch)
    TB.require_task_metadata_blocked()
    # every cached task image is a candidate, then the harness image; nothing is pulled
    assert seen[0][1] == ["alexgshaw/b:1", "alexgshaw/a:1", TB.harbor_image()]
    assert seen[0][2].get("pull") is None
    assert "Harbor pulls each task's image itself" in seen[0][2]["hint"]


def test_terminal_bench_without_cached_tasks_asks_for_the_dataset(monkeypatch, tmp_path):
    """Without a cached task the probe runs in the harness image (the name is kept: the list in
    tests/test_cluster_scripts_and_ci_helpers.py names it); the hint says the cache holds no
    task of the dataset."""
    _harbor_cache(tmp_path, monkeypatch, [])
    seen = _record(monkeypatch)
    TB.require_task_metadata_blocked()
    assert seen[0][1] == [TB.harbor_image()] and seen[0][2].get("pull") is None
    assert "holds no task of the dataset" in seen[0][2]["hint"]


def test_terminal_bench_checks_before_harbor_starts(monkeypatch):
    monkeypatch.setattr(TB, "check_terminal_bench_prerequisites", lambda: (True, ""))
    monkeypatch.setattr(TB.subprocess, "Popen", lambda *a, **k: pytest.fail("Harbor started"))
    monkeypatch.setattr(TB, "harbor_task_images", lambda dataset=None: ["alexgshaw/a:1"])
    _stop(monkeypatch)
    with pytest.raises(RuntimeError, match="gate says stop"):
        TB.run_terminal_bench("m", "http://h:1/v1", 1)


def test_osworld_checks_before_the_orchestrator(monkeypatch):
    monkeypatch.setattr(OS, "check_ui_control_osworld_prerequisites", lambda: (True, ""))
    _no_process(monkeypatch, OS)
    seen = []
    _stop(monkeypatch, seen)
    with pytest.raises(RuntimeError, match="gate says stop"):
        OS.run_ui_control_osworld("m", "http://h:1/v1")
    assert seen[0][1] == [OS._image()]


# ============================================================================================
# skillsbench: the key stays in gbench
# ============================================================================================
class _FakeRelay:
    made = []

    def __init__(self, upstream, bind_host, key=None, name="", allowed_paths=None, **kw):
        self.upstream, self.bind_host, self._key = upstream, bind_host, key
        self.allowed_paths, self.stopped = tuple(allowed_paths or ()), False
        self.body_fields = kw.get("body_fields")
        _FakeRelay.made.append(self)

    def start(self):
        return self

    def stop(self):
        self.stopped = True

    @property
    def url(self):
        return f"http://{self.bind_host}:43210/v1"

    def record(self):
        return {"listens": self.url, "adds": "Authorization: Bearer <OPENAI_API_KEY>"}


def _run_skillsbench(monkeypatch, tmp_path):
    monkeypatch.setattr(SB, "check_skillsbench_prerequisites", lambda: (True, ""))
    _FakeRelay.made = []
    monkeypatch.setattr(SB, "EndpointRelay", _FakeRelay)
    calls = []

    def _run(cmd, **kw):
        calls.append({"cmd": list(cmd), "env": kw.get("env")})
        if cmd[:2] == ["docker", "run"]:
            workdir = cmd[cmd.index("-v", cmd.index("-v") + 1) + 1].split(":")[0]
            with open(os.path.join(workdir, "skillsbench_summary.json"), "w") as f:
                json.dump({"conditions": {"without-skills": {"mean_reward": 0.5, "n_tasks": 2,
                                                             "n_scored": 2}}}, f)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(SB.subprocess, "run", _run)
    _record(monkeypatch)
    res = SB.run_skillsbench("m", "http://127.0.0.1:8000/v1", 1)
    run = next(c for c in calls if c["cmd"][:2] == ["docker", "run"])
    return res, run


def test_skillsbench_without_a_key_passes_only_the_placeholder_by_name(monkeypatch, tmp_path):
    res, run = _run_skillsbench(monkeypatch, tmp_path)
    cmd, env = run["cmd"], run["env"]
    assert cmd[cmd.index("GBENCH_MODEL_API_KEY") - 1] == "-e"
    assert not any(a.startswith("GBENCH_MODEL_API_KEY=") for a in cmd)
    assert env["GBENCH_MODEL_API_KEY"] == SB.MODEL_KEY_PLACEHOLDER == "dummy"
    # every run goes through gbench's relay (it sets the run's temperature, 0.0 without
    # --thinking); without a key it adds none
    [relay] = _FakeRelay.made
    assert relay._key is None and relay.body_fields == {
        "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
    assert relay.upstream == "http://127.0.0.1:8000/v1" and relay.stopped
    assert f"GBENCH_MODEL_BASE_URL={relay.url}" in cmd and "model_relay" in res


def test_skillsbench_with_a_key_relays_and_the_key_goes_nowhere(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_SKILLSBENCH_API_KEY", SECRET)
    monkeypatch.setenv("GBENCH_SKILLSBENCH_TASK_ENDPOINT", "http://elsewhere:1/v1")
    res, run = _run_skillsbench(monkeypatch, tmp_path)
    cmd, env = run["cmd"], run["env"]
    relay = _FakeRelay.made[0]
    assert relay._key == SECRET and relay.bind_host == "172.17.0.1"
    assert relay.upstream == "http://127.0.0.1:8000/v1" and relay.stopped
    assert "/v1/responses" in relay.allowed_paths and "/v1/chat/completions" in relay.allowed_paths
    assert f"GBENCH_MODEL_BASE_URL={relay.url}" in cmd
    assert SECRET not in " ".join(cmd)
    assert SECRET not in json.dumps(env) and env["GBENCH_MODEL_API_KEY"] == "dummy"
    assert "GBENCH_SKILLSBENCH_API_KEY" not in env
    assert res["model_relay"]["adds"] == "Authorization: Bearer <GBENCH_SKILLSBENCH_API_KEY>"
    assert SECRET not in json.dumps(res, default=str)


@pytest.mark.parametrize("value", ["", "dummy", "EMPTY"])
def test_skillsbench_placeholder_keys_are_no_key(monkeypatch, value):
    monkeypatch.setenv("GBENCH_SKILLSBENCH_API_KEY", value)
    assert SB.endpoint_api_key() is None
    monkeypatch.setenv("GBENCH_SKILLSBENCH_TASK_HOST", "10.1.2.3")
    assert SB.relay_host() == "10.1.2.3"


def test_skillsbench_relay_that_cannot_listen_is_infra_required(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_SKILLSBENCH_API_KEY", SECRET)
    monkeypatch.setattr(SB, "check_skillsbench_prerequisites", lambda: (True, ""))
    _record(monkeypatch)

    class _Refuses(_FakeRelay):
        def start(self):
            raise OSError("Cannot assign requested address")
    monkeypatch.setattr(SB, "EndpointRelay", _Refuses)
    monkeypatch.setattr(SB.subprocess, "run", lambda cmd, **kw: (
        subprocess.CompletedProcess(cmd, 0, "", "") if cmd[:3] == ["docker", "rm", "-f"]
        else pytest.fail(f"started {cmd[:2]}")))
    with pytest.raises(RuntimeError) as err:
        SB.run_skillsbench("m", "http://127.0.0.1:8000/v1", 1)
    assert "GBENCH_SKILLSBENCH_TASK_HOST" in str(err.value) and SECRET not in str(err.value)


def test_skillsbench_checks_before_the_orchestrator(monkeypatch):
    monkeypatch.setattr(SB, "check_skillsbench_prerequisites", lambda: (True, ""))
    _no_process(monkeypatch, SB)
    seen = []
    _stop(monkeypatch, seen)
    with pytest.raises(RuntimeError, match="gate says stop"):
        SB.run_skillsbench("m", "http://h:1/v1", 1)
    assert seen[0][1] == [SB._image()]


def test_skillsbench_launcher_hands_the_agent_only_what_it_was_given():
    text = open(os.path.join(REPO, "gbench", "docker", "skillsbench_run.py"),
                encoding="utf-8").read()
    assert 'os.environ.get("GBENCH_MODEL_API_KEY", "dummy")' in text
    assert "never a real key" in text


# ============================================================================================
# the docs say what the code does
# ============================================================================================
def _doc(name):
    return " ".join(open(os.path.join(REPO, "docs", "evals", f"{name}.md"),
                         encoding="utf-8").read().split())


@pytest.mark.parametrize("name", ["copilot_bench_swe", "swe_bench_multilingual", "swe_bench_pro",
                                  "swe_bench_live", "multi_swe_bench", "swe_lancer",
                                  "terminal_bench", "skillsbench", "ui_control_osworld"])
def test_each_suite_doc_names_the_metadata_prerequisite(name):
    doc = _doc(name)
    assert CG.FIREWALL_COMMAND in doc and CG.PERSIST_COMMAND in doc
    assert "169.254.169.254:80" in doc and "gbench-metadata-probe-<id>" in doc
    assert "never runs sudo" in doc and "infra_required" in doc
    assert "toolchains.md#cloud-vms-the-metadata-server" in doc


def test_docs_name_the_windows_tab_and_the_skillsbench_relay():
    live = _doc("swe_bench_live")
    assert "check_windows_metadata_guard" in live and "-EncodedCommand" in live
    sb = _doc("skillsbench")
    assert "GBENCH_SKILLSBENCH_API_KEY" in sb and "`/v1/responses`" in sb
    assert "the command line carries only the name" in sb and "`model_relay`" in sb


def test_the_gate_is_stdlib_only_at_module_level_in_swebench_common():
    """The tau image loads swebench_common; the gate imports container_guard lazily."""
    import ast
    tree = ast.parse(open(C.__file__, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level:
            assert all(a.name != "container_guard" for a in node.names)
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] in sys.stdlib_module_names for a in node.names)
