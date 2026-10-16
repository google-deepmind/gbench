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

"""multi_swe_bench scoring (smoke run 2026-10-09): the Multi-SWE-bench harness failed to build
mswebench/simdjson_m_simdjson:base-cpp-7 (`FROM gcc:7`, Debian buster: `apt-get update` exits
100 because deb.debian.org no longer serves buster), so it wrote no final_report.json.
mswe_harness_build runs the harness with the archive.debian.org retry for such builds; and the
test runs of simdjson-949/954/958, whose build fetches simdjson's master (checkperf reference) at
test time, get that master pinned to the commit upstream's own instance images recorded; a run
whose pin could not be set up (before the patch is applied) is a harness error
(`checkperf_reference`), whatever the patch."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile

import pytest

from gbench.runners.eval_suites import mswe_harness_build as H
from gbench.runners.eval_suites import multi_swe_bench as M
from gbench.runners.eval_suites import swebench_common as C

#: apt's output of the smoke run's failing step (docker wraps the stderr in ANSI colour codes).
APT_LOG = """Step 4/5 : RUN apt-get update &&     apt-get install -y     build-essential
Ign:2 http://deb.debian.org/debian buster InRelease
Err:5 http://deb.debian.org/debian buster Release
  404  Not Found [IP: 151.101.2.132 80]
\x1b[91mE: The repository 'http://security.debian.org/debian-security buster/updates Release' does not have a Release file.
E: The repository 'http://deb.debian.org/debian buster Release' does not have a Release file.
\x1b[0m ---> Removed intermediate container 6a02076dbbe4"""
#: the harness's base-cpp-7 Dockerfile (multi_swe_bench 1.1.2, need_clone)
UPSTREAM = """FROM gcc:7



WORKDIR /home/

RUN git clone https://github.com/simdjson/simdjson.git /home/simdjson

RUN apt-get update &&     apt-get install -y     build-essential     pkg-config     wget     tar &&     wget https://cmake.org/files/v3.14/cmake-3.14.0-Linux-x86_64.tar.gz &&     tar -zxvf cmake-3.14.0-Linux-x86_64.tar.gz &&     mv cmake-3.14.0-Linux-x86_64 /opt/cmake &&     ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake &&     rm cmake-3.14.0-Linux-x86_64.tar.gz
RUN apt-get install -y cmake
"""


def test_the_repair_is_the_drivers_own():
    d = H.driver()
    assert os.path.samefile(H.DRIVER_PATH, os.path.join(
        os.path.dirname(M.__file__), "..", "..", "docker", "mswe_agent_run.py"))
    assert "archive.debian.org" in d.APT_ARCHIVE_REPAIR


def test_archived_suite_lines_are_apts_and_not_a_wget_404():
    lines = H.archived_suite_lines(APT_LOG)
    assert lines == [
        "E: The repository 'http://security.debian.org/debian-security buster/updates Release' "
        "does not have a Release file.",
        "E: The repository 'http://deb.debian.org/debian buster Release' does not have a "
        "Release file."]
    wget = ("--2026-10-10--  https://cmake.org/files/v3.14/x.tar.gz\n"
            "HTTP request sent, awaiting response... 404 Not Found\n")
    assert H.archived_suite_lines(wget) == []
    # old apt (jessie): the 404 on the debian.org URL line
    assert H.archived_suite_lines(
        "W: Failed to fetch http://deb.debian.org/debian/dists/jessie/main/binary-amd64/"
        "Packages  404  Not Found")


def test_with_archive_repair_adds_one_step_after_each_from_and_keeps_upstream():
    patched = H.with_archive_repair(UPSTREAM + "FROM alpine AS second\nRUN true\n")
    lines = patched.splitlines()
    froms = [i for i, ln in enumerate(lines) if ln.startswith("FROM ")]
    assert len(froms) == 2
    for i in froms:
        assert lines[i + 1].startswith("RUN [")
        assert json.loads(lines[i + 1][4:]) == ["/bin/sh", "-c", H.driver().APT_ARCHIVE_REPAIR]
        assert lines[i + 2] == f'LABEL {H.LABEL}="archive.debian.org"'
    # removing the added lines gives upstream's Dockerfile back, line for line
    kept = [ln for i, ln in enumerate(lines) if not any(i in (f + 1, f + 2) for f in froms)]
    assert kept == (UPSTREAM + "FROM alpine AS second\nRUN true\n").splitlines()


def test_repair_result_ignores_dockers_echo_of_the_step():
    step = H.with_archive_repair("FROM gcc:7\n").splitlines()[1]
    log = f"Step 2/7 : {step}\n ---> Running in 1\n\x1b[0mapt archive fallback: applied (buster)\n"
    assert "not applied (${ID:-unknown}" in log   # the echo carries the mark's text
    assert H.repair_result(log) == {"applied": True, "release": "buster"}
    assert H.repair_result(f"Step 2/7 : {step}\n") is None


class _Docker:
    """docker_util stand-in: each build call logs `logs[n]` and raises when `fails[n]`."""

    def __init__(self, logs, fails):
        self.logs, self.fails, self.calls = list(logs), list(fails), []
        self.build = self._build

    def _build(self, workdir, dockerfile_name, image_full_name, logger):
        n = len(self.calls)
        self.calls.append((dockerfile_name, open(os.path.join(workdir, dockerfile_name)).read()))
        for line in self.logs[n].splitlines():
            logger.info(line)
        if self.fails[n]:
            raise RuntimeError(f"Docker build failed: step returned a non-zero code: 100 ({n})")


class _Logger:
    def __init__(self):
        self.lines = []

    def info(self, msg, *a):
        self.lines.append(msg % a if a else msg)

    error = warning = debug = info


def _dir(tmp_path):
    d = tmp_path / "base-cpp-7"
    d.mkdir()
    (d / "Dockerfile").write_text(UPSTREAM)
    return d


def test_an_archived_debian_build_is_retried_with_the_repair_and_recorded(tmp_path):
    d, rec = _dir(tmp_path), tmp_path / "rec.jsonl"
    du = _Docker([APT_LOG, "Step 2/7 : RUN [...] apt archive fallback: not applied (x)\n"
                  "apt archive fallback: applied (buster)\nSuccessfully built 1"],
                 [True, False])
    assert H.install(str(rec), du)
    log = _Logger()
    du.build(d, "Dockerfile", "mswebench/simdjson_m_simdjson:base-cpp-7", log)
    assert [c[0] for c in du.calls] == ["Dockerfile", "Dockerfile" + H.PATCHED_SUFFIX]
    assert du.calls[0][1] == UPSTREAM                          # upstream's, untouched, first
    assert du.calls[1][1] == H.with_archive_repair(UPSTREAM)
    assert (d / "Dockerfile").read_text() == UPSTREAM
    [note] = H.read_records(str(rec))
    assert note["image"] == "mswebench/simdjson_m_simdjson:base-cpp-7"
    assert note["retry"] == "ok" and note["applied"] is True and note["release"] == "buster"
    assert len(note["apt_error"]) == 2 and "non-zero code: 100" in note["upstream_error"]
    # the harness's build log still gets every line, plus the retry's announcement
    assert any("Release file" in ln for ln in log.lines)
    assert any("apt archive fallback for mswebench/" in ln for ln in log.lines)


def test_a_build_that_works_or_fails_otherwise_is_upstreams(tmp_path):
    d, rec = _dir(tmp_path), tmp_path / "rec.jsonl"
    du = _Docker(["Successfully built 1"], [False])
    H.install(str(rec), du)
    du.build(d, "Dockerfile", "img", _Logger())
    assert len(du.calls) == 1 and not rec.exists()
    du = _Docker(["HTTP request sent, awaiting response... 404 Not Found"], [True])
    H.install(str(rec), du)
    with pytest.raises(RuntimeError, match=r"\(0\)"):
        du.build(d, "Dockerfile", "img", _Logger())
    assert len(du.calls) == 1 and not rec.exists()


def test_a_retry_that_fails_raises_and_is_recorded(tmp_path):
    d, rec = _dir(tmp_path), tmp_path / "rec.jsonl"
    du = _Docker([APT_LOG, "apt archive fallback: applied (buster)\nwget: 404"], [True, True])
    H.install(str(rec), du)
    with pytest.raises(RuntimeError, match=r"\(1\)"):
        du.build(d, "Dockerfile", "img", _Logger())
    [note] = H.read_records(str(rec))
    assert note["retry"].startswith("failed:")


def test_install_is_idempotent(tmp_path):
    du = _Docker([], [])
    H.install(None, du)
    first = du.build
    H.install(None, du)
    assert du.build is first and first._gbench_original == du._build


def test_main_strips_its_flag_installs_and_runs_the_harness_module(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(H, "install", lambda rec: seen.setdefault("record", rec))
    monkeypatch.setattr(H, "install_checkperf_reference",
                        lambda rec: seen.setdefault("checkperf_record", rec))

    def run_module(name, run_name, alter_sys):
        seen.update(name=name, run_name=run_name, argv=list(sys.argv))
    monkeypatch.setattr(H.runpy, "run_module", run_module)
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    H.main([H.RECORD_FLAG, str(tmp_path / "r.jsonl"), "--config", "c.json"])
    assert seen == {"record": str(tmp_path / "r.jsonl"),
                    "checkperf_record": str(tmp_path / "r.jsonl"), "name": H.HARNESS_MODULE,
                    "run_name": "__main__", "argv": [H.HARNESS_MODULE, "--config", "c.json"]}


def test_the_script_runs_standalone_without_gbench_on_the_path(tmp_path):
    """Loaded by path in the harness subprocess (as swe_thread_cap loads its shim): it imports
    nothing from gbench."""
    out = subprocess.run(
        [sys.executable, "-I", "-c",
         "import importlib.util,sys; s=importlib.util.spec_from_file_location('h', sys.argv[1]);"
         "m=importlib.util.module_from_spec(s); s.loader.exec_module(m);"
         "print('gbench' in sys.modules, 'archive.debian.org' in m.driver().APT_ARCHIVE_REPAIR)",
         H.__file__], capture_output=True, text=True, cwd=str(tmp_path))
    assert out.stdout.split() == ["False", "True"], out.stderr


# --------------------------------------------------------------------------------------------
# multi_swe_bench wiring
# --------------------------------------------------------------------------------------------
@pytest.fixture
def _scorer_env(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("GBENCH_SWE_OMP_THREADS", "0")
    monkeypatch.setattr(C, "require_task_network_blocked", lambda *a, **k: None)
    monkeypatch.setattr(C, "corroborated_dns_failures", lambda c, **k: ({}, {}))


def _traces(numbers):
    return [{"sample_idx": i, "category": "cpp",
             "extra_payload": {"instance_id": f"simdjson__simdjson-{n}", "org": "simdjson",
                               "repo": "simdjson", "number": n},
             "agent_patch": "diff --git a/x b/x\n"} for i, n in enumerate(numbers)]


PIN_OK = {"kind": "checkperf_reference", "pinned": True,
          "commit": H.CHECKPERF_REFERENCE["commit"], "detail": H.CHECKPERF_REFERENCE["commit"]}


class _Harness:
    """The harness stand-in: checks how it is launched, writes the record lines the script's
    wrappers write (build retries given; a pin record per checkperf instance run, `pins` overriding
    PIN_OK or None for none), each instance's report.json, and final_report.json."""

    def __init__(self, runs, builds=(), pins=None):
        self.runs, self.builds, self.pins, self.argv = runs, list(builds), pins or {}, None

    def __call__(self, argv, **kw):
        self.argv = list(argv)
        rec = argv[argv.index(H.RECORD_FLAG) + 1]
        cfg = json.load(open(argv[argv.index("--config") + 1]))
        rows = [json.loads(line) for line in open(cfg["patch_files"][0])]
        notes = list(self.builds)
        for r in rows:
            if r["number"] in H.CHECKPERF_REFERENCE["instances"]:
                pin = self.pins.get(r["number"], PIN_OK)
                if pin is not None:
                    notes.append({**pin, "number": r["number"],
                                  "image": f"mswebench/simdjson_m_simdjson:pr-{r['number']}"})
        with open(rec, "a") as f:
            for n in notes:
                f.write(json.dumps(n) + "\n")
        resolved, unresolved, errors = [], [], []
        for r in rows:
            iid = f"{r['org']}/{r['repo']}:pr-{r['number']}"
            verdict = self.runs[r["number"]]
            if verdict == "error":      # no test run at all (e.g. the image never built)
                errors.append(iid)
                continue
            d = os.path.join(cfg["workdir"], r["org"], r["repo"], "evals", f"pr-{r['number']}")
            os.makedirs(d, exist_ok=True)
            json.dump({"fix_patch_result": {"passed_count": 46, "failed_count": 0,
                                            "skipped_count": 0}},
                      open(os.path.join(d, "report.json"), "w"))
            (resolved if verdict == "resolved" else unresolved).append(iid)
        os.makedirs(cfg["output_dir"], exist_ok=True)
        json.dump({"resolved_ids": resolved, "unresolved_ids": unresolved,
                   "completed_ids": resolved + unresolved, "error_ids": errors,
                   "incomplete_ids": [], "total_instances": len(rows)},
                  open(os.path.join(cfg["output_dir"], "final_report.json"), "w"))
        return subprocess.CompletedProcess(argv, 0, "", "")


def test_the_scorer_runs_the_harness_through_the_build_wrapper(monkeypatch, _scorer_env):
    note = {"image": "mswebench/simdjson_m_simdjson:base-cpp-7", "retry": "ok",
            "applied": True, "release": "buster"}
    h = _Harness({949: "resolved"}, builds=[note])
    monkeypatch.setattr(M.subprocess, "run", h)
    metrics = {}
    traces = _traces([949])
    asyncio.run(M._make_scorer("m", 1, metrics)(traces))
    assert h.argv[:3] == [sys.executable, M.mswe_harness_build.__file__, H.RECORD_FLAG]
    assert "-m" not in h.argv and h.argv[-2] == "--config"
    assert metrics["harness_apt_archive_fallback"] == {
        "images": 1, "built": 1, "failed": [], "records": [note]}
    pins = metrics["harness_checkperf_reference"]
    assert pins["commit"] == H.CHECKPERF_REFERENCE["commit"] and pins["pinned"] == 1
    assert pins["failed"] == []
    assert traces[0]["is_correct"] and not traces[0].get("scoring_excluded")


# --------------------------------------------------------------------------------------------
# simdjson checkperf reference pin
# --------------------------------------------------------------------------------------------
def test_the_pin_is_scoped_to_the_three_checkperf_instances():
    assert H.CHECKPERF_REFERENCE["instances"] == (949, 954, 958)
    assert H.checkperf_instance("mswebench/simdjson_m_simdjson:pr-949") == 949
    assert H.checkperf_instance("mswebench/simdjson_m_simdjson:pr-958") == 958
    for other in ("mswebench/simdjson_m_simdjson:pr-1615", "mswebench/simdjson_m_simdjson:pr-544",
                  "mswebench/simdjson_m_simdjson:base-cpp-7", "mswebench/fmtlib_m_fmt:pr-949",
                  "mswebench/simdjson_m_simdjson:pr-9490", ""):
        assert H.checkperf_instance(other) is None, other


class _Runner:
    def __init__(self, output):
        self.output, self.calls = output, []

    def run(self, image, command, output_path=None, global_env=None, volumes=None):
        self.calls.append((image, command, output_path, global_env, volumes))
        return self.output


def test_the_run_wrapper_prefixes_the_pin_and_records_it(tmp_path):
    rec = tmp_path / "r.jsonl"
    du = _Runner("gbench-checkperf-reference: pinned " + H.CHECKPERF_REFERENCE["commit"]
                 + "\n1/1 Test #1: x ... Passed\n")
    original = du.run
    H.install_checkperf_reference(str(rec), du)
    out = du.run("mswebench/simdjson_m_simdjson:pr-949", "bash /home/fix-run.sh", "log", ["A=1"],
                 {"p": {"bind": "/home/fix.patch"}})
    assert out == original.__self__.output
    image, command, path, env, vols = original.__self__.calls[-1]
    assert command[:2] == ["/bin/bash", "-c"] and command[2].endswith("\nbash /home/fix-run.sh")
    assert command[2].startswith(H.checkperf_setup())
    assert (path, env, vols) == ("log", ["A=1"], {"p": {"bind": "/home/fix.patch"}})
    assert H.read_records(str(rec)) == [{
        "kind": "checkperf_reference", "image": "mswebench/simdjson_m_simdjson:pr-949",
        "number": 949, "commit": H.CHECKPERF_REFERENCE["commit"], "pinned": True,
        "detail": H.CHECKPERF_REFERENCE["commit"]}]
    # every other image runs exactly as upstream wrote it, unrecorded
    du.run("mswebench/simdjson_m_simdjson:pr-1615", "bash /home/fix-run.sh", "log2")
    assert original.__self__.calls[-1] == ("mswebench/simdjson_m_simdjson:pr-1615",
                                           "bash /home/fix-run.sh", "log2", None, None)
    assert len(H.read_records(str(rec))) == 1
    first = du.run
    H.install_checkperf_reference(str(rec), du)
    assert du.run is first


def test_a_run_without_the_pin_line_or_with_a_failure_records_not_pinned(tmp_path):
    rec = tmp_path / "r.jsonl"
    du = _Runner("gbench-checkperf-reference: failed fatal: no such commit\n")
    H.install_checkperf_reference(str(rec), du)
    du.run("mswebench/simdjson_m_simdjson:pr-954", "bash /home/fix-run.sh")
    assert H.read_records(str(rec))[-1]["pinned"] is False
    assert H.read_records(str(rec))[-1]["detail"] == "fatal: no such commit"
    assert H.checkperf_result("make: *** [all] Error 2")["pinned"] is False


def _git(*args, cwd=None, env=None):
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def _pin_repo(tmp_path, monkeypatch):
    """A repository standing in for the image's /home/simdjson (master two commits past the one
    to pin), an isolated HOME for git's global config, and CHECKPERF_REFERENCE pointed at them."""
    env = {**os.environ, "HOME": str(tmp_path / "home"), "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    env.pop("GIT_CONFIG_GLOBAL", None)
    os.makedirs(env["HOME"])
    repo = tmp_path / "simdjson"
    _git("init", "-q", "-b", "master", str(repo), env=env)
    shas = []
    for i in range(3):
        (repo / "f").write_text(str(i))
        _git("add", "f", cwd=repo, env=env)
        _git("commit", "-q", "-m", f"c{i}", cwd=repo, env=env)
        shas.append(_git("rev-parse", "HEAD", cwd=repo, env=env))
    monkeypatch.setitem(H.CHECKPERF_REFERENCE, "repo_dir", str(repo))
    monkeypatch.setitem(H.CHECKPERF_REFERENCE, "mirror", str(tmp_path / "mirror.git"))
    monkeypatch.setitem(H.CHECKPERF_REFERENCE, "url", "https://example.invalid/simdjson/simdjson")
    monkeypatch.setitem(H.CHECKPERF_REFERENCE, "commit", shas[0])
    return env, shas, tmp_path


def _upstream_sync(env, where):
    """benchmark/checkperf.cmake's sync_git_repository, as the instance's build runs it."""
    url = H.CHECKPERF_REFERENCE["url"]
    _git("init", "-q", str(where), env=env)
    _git("-C", str(where), "remote", "add", "origin", url, env=env)
    _git("remote", "set-url", "origin", url, cwd=where, env=env)
    _git("fetch", "--depth=1", "origin", "master", cwd=where, env=env)
    _git("reset", "--hard", "origin/master", cwd=where, env=env)
    return _git("rev-parse", "HEAD", cwd=where, env=env)


def test_the_setup_makes_upstreams_fetch_of_master_get_the_pinned_commit(_pin_repo):
    env, shas, tmp = _pin_repo
    out = subprocess.run(["bash", "-c", H.checkperf_setup()], env=env, capture_output=True,
                         text=True)
    assert H.checkperf_result(out.stdout) == {"pinned": True, "detail": shas[0]}, out
    # upstream's own commands, unchanged, now land on the pinned commit, not the newest master
    assert _upstream_sync(env, tmp / "checkperf-reference" / "master") == shas[0]
    # the image's repository is untouched
    assert _git("rev-parse", "master", cwd=H.CHECKPERF_REFERENCE["repo_dir"], env=env) == shas[2]


def test_a_setup_that_cannot_find_the_commit_fails_and_leaves_git_alone(_pin_repo, monkeypatch):
    env, shas, tmp = _pin_repo
    monkeypatch.setitem(H.CHECKPERF_REFERENCE, "commit", "0" * 40)
    out = subprocess.run(["bash", "-c", H.checkperf_setup()], env=env, capture_output=True,
                         text=True)
    got = H.checkperf_result(out.stdout)
    assert got["pinned"] is False and got["detail"], out
    assert subprocess.run(["git", "config", "--global", "--get-regexp", "insteadof"], env=env,
                          capture_output=True, text=True).stdout == ""


def test_pinned_checkperf_instances_are_scored_like_any_other(monkeypatch, _scorer_env):
    h = _Harness({949: "resolved", 954: "unresolved", 958: "resolved", 1615: "unresolved"})
    monkeypatch.setattr(M.subprocess, "run", h)
    metrics = {}
    traces = _traces([949, 954, 958, 1615])
    asyncio.run(M._make_scorer("m", 1, metrics)(traces))
    assert not any(t.get("scoring_excluded") for t in traces)
    assert [t["is_correct"] for t in traces] == [True, False, True, False]
    assert metrics["multi_swe_bench_report"]["harness_checkperf_reference_failures"] == {}
    assert metrics["harness_checkperf_reference"]["pinned"] == 3
    result = {"status": "success"}
    assert M._report_reasons(result, metrics) == [] and result["status"] == "success"


@pytest.mark.parametrize("verdict", ["resolved", "unresolved"])
def test_a_run_without_the_pin_is_a_harness_error_whatever_the_patch(monkeypatch, _scorer_env,
                                                                     verdict):
    failed = {"kind": "checkperf_reference", "pinned": False, "detail": "clone failed",
              "commit": H.CHECKPERF_REFERENCE["commit"]}
    h = _Harness({949: verdict, 954: verdict, 958: "error", 1615: "unresolved"},
                 pins={949: failed, 954: None})
    monkeypatch.setattr(M.subprocess, "run", h)
    metrics = {}
    traces = _traces([949, 954, 958, 1615])
    asyncio.run(M._make_scorer("m", 1, metrics)(traces))
    by = {t["extra_payload"]["number"]: t for t in traces}
    for n in (949, 954):
        assert by[n]["harness_error"] == M.CHECKPERF_REFERENCE and by[n]["scoring_excluded"]
        assert not by[n]["is_correct"]
    assert "clone failed" in by[949]["harness_error_reason"]
    assert "no record of the pin" in by[954]["harness_error_reason"]
    # no test run at all: already a no-verdict harness error, not counted twice
    assert by[958]["harness_error"] == M.NO_VERDICT
    assert not by[1615].get("scoring_excluded")
    rep = metrics["multi_swe_bench_report"]
    assert sorted(rep["harness_checkperf_reference_failures"]) == [
        "simdjson__simdjson-949", "simdjson__simdjson-954"]
    assert metrics["harness_checkperf_reference"]["failed"] == [
        "mswebench/simdjson_m_simdjson:pr-949"]
    result = {"status": "success"}
    reasons = M._report_reasons(result, metrics)
    assert result["status"] == "completed_with_errors"
    assert result["harness_errors"] == {M.CHECKPERF_REFERENCE: 2, M.NO_VERDICT: 1}
    assert result["partial_evaluation"]["harness_checkperf_reference_failure_instances"] == 2
    assert any("checkperf reference could not be set up" in r for r in reasons)
    head = C.scored_headline(traces)
    assert head["scored_instances"] == 1 and head["correct_answers"] == 0


def test_every_submission_without_the_pin_withholds_the_score(monkeypatch, _scorer_env):
    monkeypatch.setattr(M.subprocess, "run", _Harness({949: "unresolved"}, pins={949: None}))
    metrics = {}
    asyncio.run(M._make_scorer("m", 1, metrics)(_traces([949])))
    result = {"status": "success"}
    M._report_reasons(result, metrics)
    assert result["status"] == "error"
    assert "could not pin simdjson's checkperf reference" in result["error"]


def test_no_patch_dependent_test_env_exclusion_remains():
    for gone in ("TEST_ENV", "harness_test_env_failure", "CHECKPERF_REFERENCE_FAILURE_RE"):
        assert not hasattr(M, gone), gone
    src = open(M.__file__, encoding="utf-8").read()
    assert "harness_test_env_failures" not in src


def test_docs_describe_both_fixes():
    doc = open(os.path.join(os.path.dirname(M.__file__), "..", "..", "..", "docs", "evals",
                            "multi_swe_bench.md"), encoding="utf-8").read()
    for needle in ("mswe_harness_build.py", "harness_apt_archive_fallback",
                   "Dockerfile.gbench-apt-archive", "harness_checkperf_reference",
                   "harness_checkperf_reference_failures", "`checkperf_reference`",
                   H.CHECKPERF_REFERENCE["commit"], "simdjson-949, -954 and -958"):
        assert needle in doc, needle
    assert "test_env" not in doc
