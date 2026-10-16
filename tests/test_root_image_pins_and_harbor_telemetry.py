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

"""Image pins of the repository-root Dockerfiles, Harbor's telemetry in terminal_bench's
harness container, and the OSWorld task image digest.

  * every docker/*.Dockerfile pins its base image by digest, and every header build command names
    its own file (a command split over two header lines included);
  * every header build command in docker/ and gbench/docker/ tags its image with the version-pinned
    tag its suite uses by default, and no code or doc refers to such an image without its tag;
  * the images whose Python dependencies resolve at build time (ojbench, aider_polyglot,
    bigcodebench, omnidocbench, livebench) install against the recorded set and check it; the
    downloads that floated (ImageMagick, the NLTK data) are pinned by release or commit and sha256;
  * docker/pinned_pip_check.py (the build-time check) accepts exactly the pinned set;
  * every terminal_bench harness container runs with HARBOR_TELEMETRY=0, whatever the host sets,
    the image sets it too, and the result records it;
  * ui_control_osworld requires the pinned happysixd/osworld-docker digest under the name OSWorld's
    docker provider starts, and says how to provision it.

Docker is stubbed throughout; nothing is pulled or built.
"""

import importlib.util
import os
import re
import subprocess
import types
from pathlib import Path

import pytest

from gbench.runners.eval_suites import terminal_bench as TB
from gbench.runners.eval_suites import ui_control_osworld as U

REPO = Path(__file__).resolve().parents[1]
ROOT_DOCKER = REPO / "docker"
_DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")


def _read(rel):
    return (REPO / rel).read_text(encoding="utf-8")


def _load_checker():
    spec = importlib.util.spec_from_file_location("pinned_pip_check",
                                                  ROOT_DOCKER / "pinned_pip_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =========================================================================== #
# Base images and header build commands of docker/*.Dockerfile
# =========================================================================== #
def _root_dockerfiles():
    return sorted(p for p in ROOT_DOCKER.glob("*.Dockerfile") if ".bak" not in p.name)


def test_every_root_dockerfile_pins_its_base_by_digest():
    assert _root_dockerfiles(), "no docker/*.Dockerfile found"
    for path in _root_dockerfiles():
        text = path.read_text(encoding="utf-8")
        args = dict(re.findall(r"^ARG\s+(\w+)=(\S+)", text, re.M))
        froms = re.findall(r"^FROM\s+(\S+)", text, re.M)
        assert froms, path.name
        for ref in froms:
            if ref.startswith("${"):
                ref = args[ref[2:-1]]
            assert _DIGEST.search(ref), (path.name, ref)


def _header_text(path):
    """The Dockerfile's text with its header's continuation lines joined: a build command written
    as `docker build -t NAME:TAG \\` on one comment line and `#     -f FILE CONTEXT` on the next
    reads as one line."""
    return re.sub(r"\s*\\\n#\s*", " ", path.read_text(encoding="utf-8"))


#: Root Dockerfiles whose header holds no `docker build ... -f` command of their own, and the
#: header text that says why.
_NO_HEADER_BUILD = {
    "quality.Dockerfile": "gbench builds this image itself",
    "swe_lancer.Dockerfile": "This file is no longer used",
}


def test_every_root_header_build_command_names_its_own_file():
    for path in _root_dockerfiles():
        cmds = [c for c in re.findall(r"docker build [^\n]*", _header_text(path)) if " -f " in c]
        for cmd in cmds:
            assert cmd.split(" -f ", 1)[1].split()[0] == f"docker/{path.name}", (path.name, cmd)
        if path.name in _NO_HEADER_BUILD:
            assert _NO_HEADER_BUILD[path.name] in path.read_text(encoding="utf-8"), path.name
        else:
            assert cmds, f"docker/{path.name}: no header build command with -f was found"
    assert "-f docker/swe_lancer.Dockerfile" in _read("docker/swe_lancer.Dockerfile")


def test_a_build_command_split_over_two_header_lines_is_read_whole(tmp_path):
    path = tmp_path / "x.Dockerfile"
    path.write_text("# Build:\n#   docker build -t gbench-x:0123abcd \\\n"
                    "#       -f docker/x.Dockerfile /path/to/checkout\nFROM scratch\n")
    assert re.findall(r"docker build [^\n]*", _header_text(path)) == [
        "docker build -t gbench-x:0123abcd -f docker/x.Dockerfile /path/to/checkout"]


# =========================================================================== #
# Header image tags are the suites' pinned defaults, and no reference drops the tag
# =========================================================================== #
#: Maintained separately (the maintainer's plugins); not scanned.
_PLUGIN_FILES = ("swe_bench_pro_agentic.py", "jetski_agent.py", "slashbin_agent.py",
                 "test_swe_bench_pro_agentic.py")


def _all_dockerfiles():
    return sorted(p for d in (ROOT_DOCKER, REPO / "gbench" / "docker")
                  for p in d.glob("*.Dockerfile") if ".bak" not in p.name)


def _header_builds():
    """(Dockerfile, NAME, TAG or None, -f path) for every header `docker build` command that names
    a file with -f, in docker/ and gbench/docker/, skipping a file its header marks unused."""
    out = []
    for path in _all_dockerfiles():
        text = _header_text(path)
        if "This file is no longer used" in text.split("FROM ", 1)[0]:
            continue
        for cmd in re.findall(r"docker build [^\n]*", text):
            m = re.search(r"-t\s+(\S+)\s+-f\s+(\S+)", cmd)
            if m:
                name, _, tag = m.group(1).partition(":")
                out.append((path, name, tag or None, m.group(2)))
    return out


def _code_string_constants():
    """Every module-level string of gbench/runners and its eval suites (plugins excluded)."""
    import importlib
    import pkgutil

    import gbench.runners as runners
    import gbench.runners.eval_suites as suites
    values = set()
    for pkg in (runners, suites):
        for info in pkgutil.iter_modules(pkg.__path__):
            if f"{info.name}.py" in _PLUGIN_FILES:
                continue
            mod = importlib.import_module(f"{pkg.__name__}.{info.name}")
            values |= {v for v in vars(mod).values() if isinstance(v, str)}
    return values


def test_every_header_image_tag_is_the_owning_suites_pinned_default():
    builds = _header_builds()
    assert len(builds) >= 25, builds
    defaults = _code_string_constants()
    for path, name, tag, dockerfile in builds:
        rel = path.relative_to(REPO).as_posix()
        assert dockerfile == rel, (rel, dockerfile)
        assert tag, f"{rel}: the header builds the untagged image {name}"
        image = f"{name}:{tag}"
        if image in defaults:
            continue
        # An intermediate image whose only use is to produce the suite's image in the next step
        # (`docker run --rm SRC | docker build -t FINAL -`, multipl_e): FINAL is the default.
        piped = re.search(r"docker run --rm " + re.escape(image) + r" \| docker build -t (\S+) -",
                          _header_text(path))
        assert piped and piped.group(1) in defaults, (
            f"{rel}: no suite module uses {image} as its image default")


def _shipped_text_files():
    """README.md, docs/, gbench/runners/ and tests/ as a commit of this tree would hold them."""
    roots = ["README.md", "docs", "gbench/runners", "tests"]
    try:
        listed = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", *roots],
                                cwd=REPO, capture_output=True, text=True, timeout=60)
        rels = listed.stdout.split() if listed.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        rels = None
    if rels is None:
        rels = [p.relative_to(REPO).as_posix() for r in roots for p in
                ([REPO / r] if (REPO / r).is_file() else (REPO / r).rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts]
    for rel in rels:
        name = rel.rsplit("/", 1)[-1]
        if ".bak" in name or name.endswith(".orig") or name in _PLUGIN_FILES \
                or rel.startswith("gbench/plugins/"):
            continue
        if not name.endswith((".py", ".md", ".sh", ".txt", ".toml", ".yml", ".yaml", ".json")):
            continue
        path = REPO / rel
        if path.is_file():
            yield rel, path.read_text(encoding="utf-8", errors="replace")


def test_no_code_or_doc_refers_to_a_local_image_without_its_tag():
    """A local image name used as an image reference carries its pinned tag: an untagged name is
    :latest, which may hold another build. Prose that names the image ("the gbench-tau2 image")
    is fine; a string that is exactly the name, a `-t NAME` build tag and `NAME:latest` are not."""
    names = sorted({name for _, name, _, _ in _header_builds()})
    assert len(names) >= 25, names
    bad = []
    for rel, text in _shipped_text_files():
        for name in names:
            n = re.escape(name)
            for rx in (r"([\"'])" + n + r"\1", r"-t\s+" + n + r"(?![\w:@/.-])",
                       r"(?<![\w./-])" + n + r":latest\b"):
                for m in re.finditer(rx, text):
                    bad.append(f"{rel}:{text.count(chr(10), 0, m.start()) + 1}: {m.group(0)}")
    assert not bad, "\n".join(bad)


# =========================================================================== #
# docker/pinned_pip_check.py
# =========================================================================== #
def _dists(monkeypatch, checker, rows):
    """Make importlib.metadata.distributions() yield (name, version, location) rows."""
    def dist(name, version, where):
        return types.SimpleNamespace(metadata={"Name": name}, version=version,
                                     locate_file=lambda _p, _w=where: _w)
    monkeypatch.setattr(checker.importlib.metadata, "distributions",
                        lambda: [dist(*r) for r in rows])


def test_the_check_accepts_exactly_the_pinned_set(monkeypatch, tmp_path):
    C = _load_checker()
    pins = tmp_path / "pins.txt"
    pins.write_text("# comment\nPyYAML==6.0.3\nfilelock==3.32.6\nCython==3.3.0\n")
    site = "/usr/local/lib/python3.11/site-packages"
    _dists(monkeypatch, C, [("pyyaml", "6.0.3", site), ("FileLock", "3.32.6", site),
                            ("dmoj", "4.1.0", "/opt/judge-server"), ("pip", "24.0", site)])
    freeze = tmp_path / "freeze.txt"
    assert C.main([str(pins), "--source", "dmoj", "--build-only", "Cython",
                   "--freeze", str(freeze)]) == 0
    assert "filelock==3.32.6" in freeze.read_text() and "pyyaml==6.0.3" in freeze.read_text()
    # a missing pin that is not build-only fails
    assert C.main([str(pins), "--source", "dmoj"]) == 1


def test_the_check_fails_on_drift_on_an_unpinned_package_and_on_a_source_not_named(
        monkeypatch, tmp_path, capsys):
    C = _load_checker()
    pins = tmp_path / "pins.txt"
    pins.write_text("pyyaml==6.0.3\n")
    site = "/usr/local/lib/python3.11/site-packages"
    _dists(monkeypatch, C, [("PyYAML", "6.0.2", site), ("extra", "1.0", site)])
    assert C.main([str(pins)]) == 1
    err = capsys.readouterr().err
    assert "pyyaml 6.0.2 is installed, the pin is 6.0.3" in err
    assert "extra 1.0 is installed but not pinned" in err


def test_the_check_takes_one_of_several_metadata_directories_and_ignores_debian_modules(
        monkeypatch, tmp_path):
    C = _load_checker()
    pins = tmp_path / "pins.txt"
    pins.write_text("numpy==1.21.2\n")
    site = "/usr/local/lib/python3.10/site-packages"
    # what `pip install -I numpy==1.21.2` over numpy 1.24.3 leaves behind
    _dists(monkeypatch, C, [("numpy", "1.24.3", site), ("numpy", "1.21.2", site),
                            ("python-apt", "2.4.0", "/usr/lib/python3/dist-packages")])
    assert C.main([str(pins), "--ignore-dir", "/usr/lib/python3/dist-packages"]) == 0
    assert C.main([str(pins)]) == 1


def test_enforce_installs_the_pinned_versions_and_removes_the_unpinned(monkeypatch, tmp_path):
    C = _load_checker()
    pins = tmp_path / "pins.txt"
    pins.write_text("a==1.0\nb==2.0\n")
    site = "/s"
    state = {"rows": [("a", "1.1", site), ("c", "3.0", site)]}
    calls = []

    def run(cmd, check):
        calls.append(cmd[3:])
        if cmd[3] == "install":
            state["rows"] = [r for r in state["rows"] if r[0] not in ("a", "b")] + [
                ("a", "1.0", site), ("b", "2.0", site)]
        else:
            state["rows"] = [r for r in state["rows"] if r[0] != "c"]
    monkeypatch.setattr(C.subprocess, "run", run)
    monkeypatch.setattr(C.importlib.metadata, "distributions", lambda: [
        types.SimpleNamespace(metadata={"Name": n}, version=v, locate_file=lambda _p, _w=w: _w)
        for n, v, w in state["rows"]])
    assert C.main([str(pins), "--enforce"]) == 0
    assert calls == [["install", "--no-cache-dir", "--no-deps", "a==1.0", "b==2.0"],
                     ["uninstall", "-y", "c"]]


def test_every_pins_file_is_name_equals_version():
    C = _load_checker()
    for name in ("ojbench.constraints.txt", "aider_polyglot.constraints.txt",
                 "bigcodebench.pins.txt", "toolbench_constraints.txt"):
        pins = C.read_pins(str(ROOT_DOCKER / name))
        assert len(pins) > 10, name
        assert all(v and " " not in v for _n, v in pins.values()), name


# =========================================================================== #
# The root images whose dependencies resolved at build time
# =========================================================================== #
def test_ojbench_installs_against_its_constraints_and_checks_them():
    text = _read("docker/ojbench.Dockerfile")
    assert "COPY ojbench.constraints.txt pinned_pip_check.py /opt/gbench/" in text
    assert text.count("PIP_CONSTRAINT=/opt/gbench/ojbench.constraints.txt") == 3
    assert "pinned_pip_check.py /opt/gbench/ojbench.constraints.txt" in text
    assert "--build-only Cython" in text and "Cython==3.3.0" in _read("docker/ojbench.constraints.txt")
    assert "ENV PIP_CONSTRAINT" not in text        # not left set in the judge's environment


def test_aider_installs_against_its_constraints_and_records_the_toolchains():
    text = _read("docker/aider_polyglot.Dockerfile")
    assert "pip3 install --no-cache-dir -c /opt/gbench/aider_polyglot.constraints.txt" in text
    assert "uv pip install --system --no-cache-dir -c /opt/gbench/aider_polyglot.constraints.txt" in text
    assert "--ignore-dir /usr/lib/python3/dist-packages" in text
    assert "/opt/gbench/toolchain-versions.txt" in text
    assert "with four differences" in text


def test_bigcodebench_enforces_its_recorded_package_set():
    text = _read("docker/bigcodebench.Dockerfile")
    assert "COPY bigcodebench.pins.txt pinned_pip_check.py /opt/gbench/" in text
    assert "--source bigcodebench --enforce" in text
    # enforced right after upstream's last pip step, before the dataset is pre-installed
    assert text.index("pip install datasets==2.17.0") < text.index("--enforce") < text.index(
        "get_bigcodebench(subset='full')")


def test_omnidocbench_pins_imagemagick_the_python_closure_and_the_nltk_data():
    text = _read("docker/omnidocbench.Dockerfile")
    assert "download.imagemagick.org/archive/binaries/magick -O" not in text
    assert "ARG IMAGEMAGICK_VERSION=7.1.2-31" in text
    assert re.search(r"ARG IMAGEMAGICK_SHA256=[0-9a-f]{64}", text)
    assert 'echo "${IMAGEMAGICK_SHA256}  /tmp/magick" | sha256sum -c -' in text
    assert "PIP_CONSTRAINT=/opt/gbench/omnidocbench.constraints.txt pip install --no-cache-dir -e ." in text
    assert "'numpy==1.24.4'" in text and "diff - /opt/gbench/pip-freeze.txt" in text
    assert "import nltk; nltk.download" not in text      # the unpinned download, errors ignored
    assert "|| true" not in text.split("# NLTK data")[-1]
    assert len(re.findall(r"'[0-9a-f]{64}  punkt(_tab)?\.zip'", text)) == 2


def test_livebench_pins_its_python_closure_and_the_nltk_data():
    text = _read("docker/livebench.Dockerfile")
    assert text.count("PIP_CONSTRAINT=/opt/gbench/livebench.constraints.txt") == 3
    assert "'en-core-web-sm==3.8.0'" in text and "'tensorflow==" in text
    assert "diff - /opt/gbench/pip-freeze.txt" in text
    assert "nltk.download(n" not in text                 # the unpinned downloads
    assert "ARG NLTK_DATA_COMMIT=" in text and len(re.findall(r"': '[0-9a-f]{64}'", text)) == 4


# =========================================================================== #
# terminal_bench: Harbor's telemetry off
# =========================================================================== #
def test_every_harness_container_runs_with_harbor_telemetry_off(monkeypatch, tmp_path):
    monkeypatch.setenv("GBENCH_CACHE_DIR", str(tmp_path / "cache"))
    for env_names in ([], ["HARBOR_TELEMETRY"], ["HARBOR_TELEMETRY=1", "OPENAI_BASE_URL"]):
        cmd = TB._container_command(["harbor", "--version"], env_names=env_names)
        at = cmd.index(TB.harbor_image())
        envs = [cmd[i + 1] for i, a in enumerate(cmd[:at]) if a == "-e"]
        telemetry = [e for e in envs if e.split("=", 1)[0] == "HARBOR_TELEMETRY"]
        assert telemetry == ["HARBOR_TELEMETRY=0"], (env_names, envs)
    env = {"HARBOR_TELEMETRY": "1", "HARBOR_REGISTRY_WEBSITE_URL": "https://h",
           "OPENAI_BASE_URL": "http://127.0.0.1:1/v1"}
    assert "HARBOR_TELEMETRY" not in TB.container_env_names(env)
    assert "HARBOR_REGISTRY_WEBSITE_URL" in TB.container_env_names(env)
    assert " -e HARBOR_TELEMETRY=0 " in TB.harbor_download_command()
    assert TB.HARNESS_TELEMETRY == {"harbor_telemetry": "disabled", "env": "HARBOR_TELEMETRY=0"}


def test_the_image_sets_harbor_telemetry_off_and_checks_it():
    text = _read("gbench/docker/terminal_bench.Dockerfile")
    assert re.search(r"^ENV HOME=/harbor-home \\\n\s+HARBOR_TELEMETRY=0$", text, re.M)
    assert "t._telemetry_disabled()" in text


def test_the_guide_documents_harbor_telemetry_off():
    doc = " ".join(_read("docs/evals/terminal_bench.md").split())
    assert "gets `HARBOR_TELEMETRY=0`" in doc and "`harness_telemetry`" in doc
    assert "gbench leaves Harbor's default alone" not in doc
    assert '-e HARBOR_TELEMETRY=0 gbench-terminal-bench:harbor-0.20.0' in doc


# =========================================================================== #
# ui_control_osworld: the task image digest
# =========================================================================== #
def _osworld_ready(monkeypatch, tmp_path, exists, ids):
    monkeypatch.setattr(U.shutil, "which", lambda _x: "/usr/bin/docker")
    monkeypatch.setattr(U, "_docker_image_exists", exists)
    real_exists = os.path.exists
    monkeypatch.setattr(U.os.path, "exists", lambda p: p == "/dev/kvm" or real_exists(p))
    (tmp_path / "docker_vm_data").mkdir()
    (tmp_path / "docker_vm_data" / "Ubuntu.qcow2").write_text("x")
    monkeypatch.setenv("GBENCH_OSWORLD_VM_DIR", str(tmp_path))
    monkeypatch.delenv("GBENCH_UI_CONTROL_OSWORLD_TASK_IMAGE", raising=False)

    def run(argv, **kw):
        if "{{.Id}}" in argv:
            return types.SimpleNamespace(returncode=0, stderr="", stdout=ids[argv[-1]])
        return types.SimpleNamespace(returncode=0, stderr="",
                                     stdout=U.LAUNCHER_VERSION if "--format" in argv else "")
    monkeypatch.setattr(U.subprocess, "run", run)


def test_the_osworld_task_image_is_pinned_by_digest():
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", U.TASK_IMAGE_DIGEST)
    assert U.TASK_IMAGE_PINNED == f"happysixd/osworld-docker@{U.TASK_IMAGE_DIGEST}"


def test_the_osworld_gate_requires_the_pinned_digest(monkeypatch, tmp_path):
    same = {U.TASK_IMAGE_PINNED: "sha256:aa", "happysixd/osworld-docker": "sha256:aa"}
    _osworld_ready(monkeypatch, tmp_path, lambda _i: True, same)
    assert U.check_ui_control_osworld_prerequisites() == (True, "")


def test_the_osworld_gate_refuses_another_image_under_the_name(monkeypatch, tmp_path):
    other = {U.TASK_IMAGE_PINNED: "sha256:aa", "happysixd/osworld-docker": "sha256:bb"}
    _osworld_ready(monkeypatch, tmp_path, lambda _i: True, other)
    ok, why = U.check_ui_control_osworld_prerequisites()
    assert not ok and f"not the pinned {U.TASK_IMAGE_PINNED}" in why
    assert f"`docker pull {U.TASK_IMAGE_PINNED}`" in why
    assert f"`docker tag {U.TASK_IMAGE_PINNED} happysixd/osworld-docker`" in why


def test_the_osworld_gate_refuses_when_the_pinned_digest_is_absent(monkeypatch, tmp_path):
    ids = {"happysixd/osworld-docker": "sha256:bb"}
    _osworld_ready(monkeypatch, tmp_path, lambda i: i != U.TASK_IMAGE_PINNED, ids)
    ok, why = U.check_ui_control_osworld_prerequisites()
    assert not ok and "not the pinned" in why
    _osworld_ready_missing = lambda i: i not in (U.TASK_IMAGE_PINNED, "happysixd/osworld-docker")
    monkeypatch.setattr(U, "_docker_image_exists", _osworld_ready_missing)
    ok, why = U.check_ui_control_osworld_prerequisites()
    assert not ok and "is not on the host daemon" in why and U.TASK_IMAGE_DIGEST in why


def test_the_osworld_guide_pins_the_digest():
    doc = " ".join(_read("docs/evals/ui_control_osworld.md").split())
    assert f"docker pull {U.TASK_IMAGE_PINNED}" in doc
    assert f"docker tag {U.TASK_IMAGE_PINNED} happysixd/osworld-docker" in doc
    assert "docker pull happysixd/osworld-docker " not in doc
    assert "`task_image`" in doc
