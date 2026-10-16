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

"""Runs the Multi-SWE-bench harness (`multi_swe_bench.harness.run_evaluation`) with two additions:
the archive.debian.org fallback for the image builds of Debian releases the Debian mirrors no
longer serve, and the pinned reference of simdjson's checkperf build (CHECKPERF_REFERENCE).

The harness builds each instance's image chain from Dockerfiles it generates (the `base` image of
a repository, then the instance's `pr-<number>` image FROM it) unless an image of that name is
already present. The C++ base images of the gcc:7 instances (simdjson and the other buster-based
repositories: 48 of the 1,632 instances) are `FROM gcc:7`, Debian 10 (buster), and start with
`apt-get update`; deb.debian.org and security.debian.org now answer 404 for buster, so that step
exits 100 and the harness gives the instance no verdict. When upstream built those images
(2025-04) the mirrors still served buster.

The fallback is the one gbench/docker/mswe_agent_run.py applies to MSWE-agent's reset (its
APT_ARCHIVE_REPAIR and ARCHIVED_SUITE_RE, loaded from that file so the two cannot drift): the
upstream Dockerfile is built first, unchanged; only when that build fails and its log shows apt
reporting a debian.org suite gone from its mirror is it built once more, from a copy of the same
Dockerfile with one step added after each FROM line: the repair, which points the image's apt
sources at archive.debian.org (which serves those releases unchanged: buster's final point release
and its final security updates) with Acquire::Check-Valid-Until false, on an archived Debian
release only (a no-op anywhere else), and the label `org.gbench.apt-archive-fallback`. Every
other step, package and download is upstream's. An image whose build works is never touched.
Each retry is recorded (one JSON line per image, in the file named by `--gbench-build-record`):
the image, the apt errors that triggered it, what the repair did and whether the retry built.

It also pins the reference checkout simdjson's checkperf build fetches at test time
(CHECKPERF_REFERENCE): at the base commits of simdjson-949, -954 and -958, benchmark/checkperf.cmake
puts `checkperf-parse` in `all`, so the instance's fix-run.sh (`cmake --build .`) runs `git fetch
--depth=1 origin master` of https://github.com/simdjson/simdjson and builds that checkout with the
image's gcc 7 and CMake 3.14. Upstream's published instance images were built on 2025-03-18 (pushed
2025-03-27, all three together) from a clone whose master is 726c3eb611a1 (2025-03-17), the master
those runs fetched when upstream recorded the instances' test results; today's master no longer
configures there ("ondemand_odr_tests requires CXX20"), so the build stops before ctest whatever the
patch. For those three instances only, the harness's test container first makes a local bare clone
of the image's own /home/simdjson with `master` set to that commit (fetched from GitHub by its hash
only if the image's clone lacks it) and points git's url.<clone>.insteadOf at the repository URL:
the fetch, its depth, the reset and every build and test step are upstream's, only the commit
`master` names is fixed. The container's output starts with one line saying whether the pin was
applied, and each such run is recorded (kind "checkperf_reference") in the same record file.

Loaded and run by path (`python mswe_harness_build.py --gbench-build-record FILE --config
CONFIG`): it imports nothing from gbench, so the harness subprocess does not import the suites.
"""

import importlib.util
import json
import os
import re
import runpy
import sys
import threading
from typing import Any, Dict, List, Optional

#: gbench/docker/mswe_agent_run.py: the MSWE-agent driver, which defines the repair.
DRIVER_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "docker", "mswe_agent_run.py"))
RECORD_FLAG = "--gbench-build-record"
LABEL = "org.gbench.apt-archive-fallback"
#: The suffix of the patched copy of the upstream Dockerfile, next to it in the build directory.
PATCHED_SUFFIX = ".gbench-apt-archive"
HARNESS_MODULE = "multi_swe_bench.harness.run_evaluation"
_FROM_RE = re.compile(r"^\s*FROM\s", re.IGNORECASE)
#: ANSI colour codes docker's build stream wraps a step's stderr in.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_LOCK = threading.Lock()
_DRIVER: Any = None


def driver() -> Any:
    """The MSWE-agent driver module (stdlib imports only), loaded by path once."""
    global _DRIVER
    if _DRIVER is None:
        spec = importlib.util.spec_from_file_location("_gbench_mswe_agent_run", DRIVER_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _DRIVER = mod
    return _DRIVER


def archived_suite_lines(log_text: str) -> List[str]:
    """The lines of a build log in which apt reports a debian.org suite gone from its mirror
    (the driver's ARCHIVED_SUITE_RE on a line naming debian.org; a wget 404 elsewhere in a build
    is not one)."""
    rx = driver().ARCHIVED_SUITE_RE
    return [ln for ln in (_ANSI_RE.sub("", raw).strip() for raw in
                          str(log_text or "").splitlines())
            if rx.search(ln) and "debian.org" in ln]


def repair_result(log_text: str) -> Optional[Dict[str, Any]]:
    """What the repair step printed in a build log ({"applied", "release"}), else None. Only a
    line that starts with the mark counts: docker's "Step n/m : RUN [...]" echo of the step
    carries the mark's text too."""
    rx = driver()._APT_RESULT_RE
    for raw in str(log_text or "").splitlines():
        line = _ANSI_RE.sub("", raw).strip()
        found = rx.match(line)
        if found:
            return {"applied": found.group(1) == "applied", "release": found.group(2)}
    return None


def with_archive_repair(dockerfile: str) -> str:
    """The Dockerfile with the repair step (exec form, so the shell line reaches /bin/sh as it is)
    and the label added after each FROM line; nothing else changes."""
    step = "RUN " + json.dumps(["/bin/sh", "-c", driver().APT_ARCHIVE_REPAIR])
    label = f'LABEL {LABEL}="archive.debian.org"'
    out: List[str] = []
    for line in dockerfile.splitlines():
        out.append(line)
        if _FROM_RE.match(line):
            out.extend([step, label])
    return "\n".join(out) + "\n"


class _Tee:
    """Forwards to the harness's build logger and keeps every message docker_util.build logs."""

    def __init__(self, logger: Any):
        self._logger, self.lines = logger, []

    def _log(self, level: str, msg: Any, *args: Any, **kwargs: Any) -> None:
        try:
            self.lines.append(str(msg) % args if args else str(msg))
        except Exception:
            self.lines.append(str(msg))
        getattr(self._logger, level)(msg, *args, **kwargs)

    def debug(self, msg, *a, **k): self._log("debug", msg, *a, **k)      # noqa: E704
    def info(self, msg, *a, **k): self._log("info", msg, *a, **k)        # noqa: E704
    def warning(self, msg, *a, **k): self._log("warning", msg, *a, **k)  # noqa: E704
    def error(self, msg, *a, **k): self._log("error", msg, *a, **k)      # noqa: E704

    def __getattr__(self, name: str) -> Any:
        return getattr(self._logger, name)


def _record(path: Optional[str], note: Dict[str, Any]) -> None:
    if not path:
        return
    with _LOCK, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(note) + "\n")


def _tail(text: Any, limit: int = 400) -> str:
    return " ".join(str(text or "").split())[-limit:]


def install(record_path: Optional[str], docker_util: Any = None) -> bool:
    """Wrap `multi_swe_bench.utils.docker_util.build` (the harness calls it through the module,
    so the wrapper is what it runs). `docker_util` is a testing seam."""
    if docker_util is None:
        from multi_swe_bench.utils import docker_util
    if getattr(docker_util.build, "_gbench_apt_archive", False):
        return True
    original = docker_util.build

    def build(workdir, dockerfile_name, image_full_name, logger):
        tee = _Tee(logger)
        try:
            return original(workdir, dockerfile_name, image_full_name, tee)
        except Exception as first:
            apt_errors = archived_suite_lines("\n".join(tee.lines))
            if not apt_errors:
                raise
            first_error = first
        with open(os.path.join(str(workdir), dockerfile_name), encoding="utf-8") as f:
            upstream = f.read()
        patched_name = dockerfile_name + PATCHED_SUFFIX
        with open(os.path.join(str(workdir), patched_name), "w", encoding="utf-8",
                  newline="\n") as f:
            f.write(with_archive_repair(upstream))
        note: Dict[str, Any] = {"image": image_full_name, "apt_error": apt_errors[:6],
                                "upstream_error": _tail(first_error),
                                "dockerfile": patched_name}
        logger.info(f"gbench: apt archive fallback for {image_full_name}: the upstream build "
                    f"failed with apt reporting a suite gone from its mirror; building it again "
                    f"with the apt sources pointed at archive.debian.org ({patched_name})")
        retry = _Tee(logger)
        try:
            result = original(workdir, patched_name, image_full_name, retry)
        except Exception as e:
            note["retry"] = "failed: " + _tail(e)
            _record(record_path, note)
            raise
        found = repair_result("\n".join(retry.lines)) or {}
        note["applied"] = bool(found.get("applied"))
        note["release"] = found.get("release")
        note["retry"] = "ok"
        _record(record_path, note)
        return result

    build._gbench_apt_archive = True
    build._gbench_original = original
    docker_util.build = build
    return True


#: simdjson's checkperf reference (see the module docstring): the instances whose base commit's
#: benchmark/checkperf.cmake fetches `master` during `cmake --build .` (of the dataset's 20 simdjson
#: instances: 949, 954, 958; 11 others have a fetch in benchmark/dom/checkperf.cmake that is off by
#: default and that their fix-run.sh never turns on; 6 have none), the URL it fetches and the
#: commit pinned as that master: the master of the clone in upstream's published instance image
#: mswebench/simdjson_m_simdjson:pr-949 (created 2025-03-18).
CHECKPERF_REFERENCE: Dict[str, Any] = {
    "image_re": re.compile(r"(?:^|/)simdjson_m_simdjson:pr-(\d+)$"),
    "instances": (949, 954, 958),
    "repo_dir": "/home/simdjson",
    "mirror": "/tmp/gbench-checkperf-reference.git",
    "url": "https://github.com/simdjson/simdjson",
    "commit": "726c3eb611a15f7ad8690bfe800e5f5dbf95376d",
}
CHECKPERF_MARK = "gbench-checkperf-reference:"
_CHECKPERF_MARK_RE = re.compile(r"^" + re.escape(CHECKPERF_MARK) + r" (pinned|failed)\b(.*)$",
                                re.MULTILINE)


def checkperf_instance(image_full_name: str) -> Optional[int]:
    """The PR number when `image_full_name` is the harness image of an instance whose test run
    fetches simdjson's checkperf reference (CHECKPERF_REFERENCE), else None."""
    found = CHECKPERF_REFERENCE["image_re"].search(str(image_full_name or ""))
    if found and int(found.group(1)) in CHECKPERF_REFERENCE["instances"]:
        return int(found.group(1))
    return None


def checkperf_setup() -> str:
    """The shell lines run before the instance's own command: a bare clone of the image's repository
    (objects shared, so it is instant) whose `master` is the pinned commit, and git's insteadOf
    pointing the fetched URL at it; one line of output says whether it was applied. A failure
    leaves git untouched (the fetch then goes to GitHub, as upstream's does)."""
    c = CHECKPERF_REFERENCE
    ref = c["mirror"]
    return "\n".join([
        "gbench_checkperf_pin() {",
        f"  git clone -q --bare --shared {c['repo_dir']} {ref} || return 1",
        f"  git -C {ref} cat-file -e '{c['commit']}^{{commit}}' 2>/dev/null || "
        f"git -C {ref} fetch -q --depth=1 {c['url']} {c['commit']} || return 1",
        f"  git -C {ref} update-ref refs/heads/master {c['commit']} || return 1",
        f"  git config --global url.{ref}.insteadOf {c['url']}",
        "}",
        f"if gbench_checkperf_pin >{ref}.log 2>&1; then",
        f"  echo '{CHECKPERF_MARK} pinned {c['commit']}'",
        "else",
        f"  echo \"{CHECKPERF_MARK} failed $(tail -c 300 {ref}.log "
        "| tr '\\n' ' ')\"",
        "fi",
    ])


def checkperf_result(output: str) -> Dict[str, Any]:
    """What the setup line in a container's output says: {"pinned": bool, "detail": str}; a
    missing line is a failure."""
    found = _CHECKPERF_MARK_RE.search(str(output or ""))
    if not found:
        return {"pinned": False, "detail": "no pin line in the test run's output"}
    return {"pinned": found.group(1) == "pinned", "detail": found.group(2).strip()}


def install_checkperf_reference(record_path: Optional[str], docker_util: Any = None) -> bool:
    """Wrap `multi_swe_bench.utils.docker_util.run` (the harness's test run of each instance in
    human_mode, its default) so the CHECKPERF_REFERENCE instances run checkperf_setup() first.
    Every other image runs exactly as upstream wrote it. `docker_util` is a testing seam."""
    if docker_util is None:
        from multi_swe_bench.utils import docker_util
    if getattr(docker_util.run, "_gbench_checkperf", False):
        return True
    original = docker_util.run

    def run(image_full_name, run_command, output_path=None, global_env=None, volumes=None):
        number = checkperf_instance(image_full_name)
        if number is None:
            return original(image_full_name, run_command, output_path, global_env, volumes)
        command = ["/bin/bash", "-c", checkperf_setup() + "\n" + str(run_command)]
        output = original(image_full_name, command, output_path, global_env, volumes)
        _record(record_path, {"kind": "checkperf_reference", "image": image_full_name,
                              "number": number, "commit": CHECKPERF_REFERENCE["commit"],
                              **checkperf_result(output)})
        return output

    run._gbench_checkperf = True
    run._gbench_original = original
    docker_util.run = run
    return True


def read_records(path: str) -> List[Dict[str, Any]]:
    """The records `install`'s wrapper wrote (none when the file is absent)."""
    out: List[Dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError:
        pass
    return out


def main(argv: List[str]) -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    # run as a script, the suites' directory (generic module names) would lead sys.path
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here or not p]
    record = None
    if RECORD_FLAG in argv:
        at = argv.index(RECORD_FLAG)
        record = argv[at + 1]
        argv = argv[:at] + argv[at + 2:]
    try:
        install(record)
    except Exception as exc:  # the harness still runs as upstream wrote it
        sys.stderr.write(f"gbench: apt archive fallback not installed: {exc!r}\n")
    try:
        install_checkperf_reference(record)
    except Exception as exc:  # those instances' runs then record no pin (harness errors)
        sys.stderr.write(f"gbench: checkperf reference pin not installed: {exc!r}\n")
    sys.argv = [HARNESS_MODULE, *argv]
    runpy.run_module(HARNESS_MODULE, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main(sys.argv[1:])
