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

"""The dependency policy and the CI install.

* gbench never installs the serving stack (vllm, torch, transformers): it requires it at run time
  only for local serving, and the eval extras stay additive to the pinned core.
* Every third-party module gbench's host-side code imports is declared in pyproject.toml (the
  base dependencies or the `evals` / `omnidocbench` extras), or is listed below with the reason it
  is not (the serving stack, a harness that runs in its own local image, a checkout the user
  provides). Code that runs inside a local image (gbench/docker/) is not host-side.
* Host code never pulls a registry image of gbench's own tooling, never creates a virtualenv and
  never `uv run`s; no doc asks for a separate virtualenv.
* uv.lock records exactly what pyproject.toml declares, and one resolve holds every extra the
  Unit Tests workflow installs (there is no uv conflict to split it).
* The workflow installs the declared extras in one step and gates skips on the same extras; its
  Python legs match requires-python and the classifiers; every action is pinned to a commit.

Everything here reads files; nothing is installed, and no network, docker or model is used.
"""

import ast
import inspect
import pathlib
import re
import sys
import tomllib
from pathlib import Path

import pytest
import yaml


def _opt_deps():
    root = pathlib.Path(__file__).resolve().parents[1]
    with open(root / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["optional-dependencies"]


def _names(reqs):
    out = []
    for r in reqs:
        n = r.split(";")[0].split("[")[0]
        for sep in ("<", ">", "=", "!", "~", " "):
            n = n.split(sep)[0]
        out.append(n.strip().lower())
    return out


def test_local_extra_does_not_install_the_serving_stack():
    opt = _opt_deps()
    assert "local" in opt, "keep the [local] extra so `pip install gbench[local]` still resolves"
    for pkg in ("vllm", "torch", "transformers"):
        assert pkg not in _names(opt["local"]), (
            f"{pkg} must NOT be auto-installed by gbench[local] (provisioned per docs; "
            "gbench hard-errors at run time if it is missing)")


def test_eval_extras_do_not_pull_the_serving_stack():
    opt = _opt_deps()
    for extra in ("evals", "omnidocbench"):
        got = _names(opt.get(extra, []))
        for pkg in ("vllm", "torch", "transformers"):
            assert pkg not in got, f"the [{extra}] extra must not pull {pkg} (keeps it additive)"


def test_require_vllm_message_points_to_provisioning_not_the_local_extra():
    from gbench.utils.dependency_checks import require_vllm_engine
    src = inspect.getsource(require_vllm_engine)
    assert "gbench[local]" not in src, "must not tell users to pip install gbench[local] for vllm"
    assert "pip install vllm" in src
    assert "remote-endpoint" in src, "should note remote runs do not need vllm"


# --------------------------------------------------------------------------- #
# Dependency declarations and the CI install
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"

#: Plugin code is maintained separately and is not scanned.
_PLUGIN_FILES = {"swe_bench_pro_agentic.py", "jetski_agent.py", "slashbin_agent.py"}

#: Directories whose code does not run in gbench's own interpreter.
_NOT_HOST = {
    "gbench/docker": "each helper runs inside its suite's local image",
    "gbench/third_party/eval_suites/_swelancer_adapter": (
        "SWE-Lancer's harness adapter; gbench does not import it (swe_lancer is not registered)"),
}

#: Distributions whose import name is not their normalized project name.
_IMPORT_TO_DIST = {
    "PIL": "pillow",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "fitz": "pymupdf",
    "docx": "python-docx",
    "pptx": "python-pptx",
    "minisweagent": "mini-swe-agent",
    "google.genai": "google-genai",
    "dotenv": "python-dotenv",
}

#: Imported by host-side code but deliberately not declared, with the reason.
EXPLAINED = {
    "vllm": "the serving stack: provisioned by the user (README), required only for local serving",
    "httpx": ("a dependency of openai (declared); agent_dojo builds its client's httpx.Timeout "
              "(a short connect timeout) from it"),
    "torch": "the serving stack (see vllm); complexfuncbench's embedder and t_eval check for it",
    "transformers": "the serving stack's tokenizer library (see vllm)",
    "sentence_transformers": "pulls torch; t_eval and api_bank check for it at run time",
    "tau2": "tau2-bench runs in the local gbench-tau2 image (its own pins)",
    "rich": "imported by tau_common only inside the gbench-tau2 image",
    "rank_bm25": "imported by tau_common only inside the gbench-tau2 image (tau2's `knowledge`)",
    "harbor": "Harbor 0.20.0 runs in the local gbench-terminal-bench image (its own pins)",
    "bfcl_eval": ("bfcl-eval runs in the local gbench-bfcl image; the dev extra carries it only "
                  "for the unit tests"),
    "api_call_extraction": "a module of the API-Bank checkout (GBENCH_APIBANK_DIR)",
    "evaluator": "a module of the API-Bank checkout (GBENCH_APIBANK_DIR)",
    "tool_manager": "a module of the API-Bank checkout (GBENCH_APIBANK_DIR)",
    "matharena": "the vendored MathArena package itself (matharena_src)",
    "axle": ("MathArena's Lean client: the gbench-matharena:b89f2f0a image installs it; the final-answer "
             "grader never calls it, and host-side unit tests stub it"),
    "lean_explore": "MathArena's Lean search tool, imported lazily and never called by the grader",
}


def _pyproject():
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _req_name(requirement: str) -> str:
    from packaging.requirements import Requirement
    return _norm(Requirement(requirement).name)


def _declared(*extras: str):
    project = _pyproject()["project"]
    reqs = list(project["dependencies"])
    for extra in extras:
        reqs += project["optional-dependencies"][extra]
    return {_req_name(r) for r in reqs}


def _parse(path: Path) -> ast.Module:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)     # a vendored file's regex escapes
        return ast.parse(path.read_text(encoding="utf-8"))


def _host_files():
    for path in sorted((REPO / "gbench").rglob("*.py")):
        rel = path.relative_to(REPO).as_posix()
        if ("/plugins/" in rel or path.name in _PLUGIN_FILES or ".bak" in path.name
                or any(rel.startswith(d + "/") for d in _NOT_HOST)):
            continue
        yield rel, path


def _third_party_imports():
    """{module: [file:line, ...]} for every absolute import of a non-stdlib, non-gbench module
    (for `google`, the submodule: google.genai)."""
    found = {}
    for rel, path in _host_files():
        tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module + ("." + node.names[0].name if node.module == "google"
                                        else "")]
            else:
                continue
            for name in names:
                parts = name.split(".")
                top = ".".join(parts[:2]) if parts[0] == "google" else parts[0]
                if parts[0] in sys.stdlib_module_names or parts[0] in ("gbench", "__future__"):
                    continue
                found.setdefault(top, []).append(f"{rel}:{node.lineno}")
    return found


def test_every_host_side_import_is_declared_or_explained():
    declared = _declared("evals", "omnidocbench")
    missing = {}
    for module, where in _third_party_imports().items():
        dist = _norm(_IMPORT_TO_DIST.get(module, module))
        if dist in declared or module in EXPLAINED:
            continue
        missing[module] = where[:3]
    assert not missing, ("imported by host-side code but declared nowhere in pyproject.toml "
                         f"(declare it, or explain it in EXPLAINED): {missing}")


def test_the_explained_list_has_no_stale_entry():
    imported = _third_party_imports()
    declared = _declared("evals", "omnidocbench")
    for module, reason in EXPLAINED.items():
        assert reason.strip(), module
        assert module in imported, f"{module} is no longer imported: drop it from EXPLAINED"
        assert _norm(_IMPORT_TO_DIST.get(module, module)) not in declared, (
            f"{module} is declared now: drop it from EXPLAINED")


def test_the_harnesses_that_run_in_images_are_not_declared_for_the_host():
    project = _pyproject()["project"]
    extras = project["optional-dependencies"]
    runtime = _declared(*(e for e in extras if e != "dev"))
    for dist in ("harbor", "tau2", "bfcl-eval", "semgrep", "ojbench", "vllm", "torch",
                 "transformers"):
        assert dist not in runtime, f"{dist} must not be a runtime dependency (see pyproject)"
    # bfcl-eval is a test-only dependency, pinned to the release the harness image holds.
    from gbench.runners.eval_suites.bfcl_v4_agentic import BFCL_EVAL_RELEASE
    assert f"bfcl-eval=={BFCL_EVAL_RELEASE}" in extras["dev"]


# -- no registry pull, no virtualenv, no host `uv run` ------------------------------------------

def _code_strings(tree):
    """String constants of a module that are not docstrings."""
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings):
            yield node


def _argv_lists(tree):
    """Leading string elements of every list/tuple literal (argv-shaped)."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)):
            head = []
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    head.append(elt.value)
                else:
                    break
            if head:
                yield node.lineno, head


#: The only host-side `docker pull`: the per-task instance images an upstream SWE harness names
#: (part of the benchmark's definition, pulled only when absent), never gbench's own tooling.
_PULL_ALLOWED = {"gbench/runners/eval_suites/swebench_common.py"}


def test_host_code_pulls_no_registry_image_of_its_own_tooling():
    offenders = []
    for rel, path in _host_files():
        tree = _parse(path)
        for lineno, head in _argv_lists(tree):
            if head[:2] == ["docker", "pull"] and rel not in _PULL_ALLOWED:
                offenders.append(f"{rel}:{lineno}")
        for node in _code_strings(tree):
            if re.match(r"\s*docker pull\s", node.value) and rel not in _PULL_ALLOWED:
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, offenders


_VENV = re.compile(r"(^|\s)(uv (run|venv|sync|tool install)|python3? -m venv|virtualenv)(\s|$)")

#: argv literals that name `uv run` but run inside a local image, never on the host.
_UV_IN_IMAGE = {
    # mcp_atlas.CREDENTIALED_COMMAND: the credentialed-servers container's command (upstream's own
    # CMD in the gbench-mcp-atlas:f24ba3fb image), started with `docker exec`.
    ("gbench/third_party/eval_suites/mcp_atlas.py", ("uv", "run", "python", "-m", "uvicorn")),
}


def test_host_code_creates_no_virtualenv_and_never_uv_runs():
    offenders = []
    for rel, path in _host_files():
        tree = _parse(path)
        for lineno, head in _argv_lists(tree):
            if (head[:2] in (["uv", "run"], ["uv", "venv"], ["uv", "sync"])
                    or head[:1] == ["virtualenv"] or head[1:3] == ["-m", "venv"]):
                if (rel, tuple(head[:5])) not in _UV_IN_IMAGE:
                    offenders.append(f"{rel}:{lineno}")
        for node in _code_strings(tree):
            if _VENV.match(node.value.strip()):
                offenders.append(f"{rel}:{node.lineno}")
    assert not offenders, offenders


def test_no_doc_asks_for_a_separate_virtualenv():
    docs = [REPO / "README.md", REPO / "CONTRIBUTING.md", REPO / "tests" / "README.md",
            *sorted((REPO / "docs").rglob("*.md"))]
    # an instruction to make one ("a separate venv", "**separate venv**", "install it in a
    # separate ... environment"); a doc saying an image holds versions "in separate virtualenvs",
    # or that none is needed, is not one.
    pattern = re.compile(r"(\ba separate|\*\*separate) (virtual ?env|venv)"
                         r"|\binstall\w*\b[^.|]{0,40}\bseparate\b[^.|]{0,40}\b(environment|venv|virtualenv)",
                         re.IGNORECASE)
    hits = [f"{p.relative_to(REPO)}:{i}" for p in docs if ".bak" not in p.name
            for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
            if pattern.search(line)]
    assert not hits, hits


# -- uv.lock ---------------------------------------------------------------------------------

def _lock_gbench():
    with open(REPO / "uv.lock", "rb") as f:
        lock = tomllib.load(f)
    return lock, next(p for p in lock["package"] if p["name"] == "gbench")


def _spec(text: str) -> frozenset:
    from packaging.specifiers import SpecifierSet
    return frozenset(str(s) for s in SpecifierSet(text))


def test_uv_lock_records_exactly_what_pyproject_declares():
    from packaging.requirements import Requirement
    project = _pyproject()["project"]
    lock, pkg = _lock_gbench()
    want = set()
    for extra, reqs in [("", project["dependencies"]),
                        *project["optional-dependencies"].items()]:
        for req in reqs:
            r = Requirement(req)
            want.add((_norm(r.name), _spec(str(r.specifier)), extra))
    got = set()
    for d in pkg["metadata"]["requires-dist"]:
        m = re.search(r"extra == '([\w-]+)'", d.get("marker", ""))
        got.add((_norm(d["name"]), _spec(d.get("specifier", "")), m.group(1) if m else ""))
    assert got == want, ("uv.lock is stale: run `uv lock` after changing pyproject.toml",
                         sorted((n, sorted(s), e) for n, s, e in want ^ got))
    assert sorted(pkg["metadata"]["provides-extras"]) == sorted(project["optional-dependencies"])
    assert lock["requires-python"] == project["requires-python"]


def test_one_resolve_holds_every_extra_the_unit_tests_install():
    assert "conflicts" not in _pyproject().get("tool", {}).get("uv", {}), (
        "a declared uv conflict means the CI install cannot be one resolve")
    lock, _ = _lock_gbench()
    assert "conflicts" not in lock
    # no extra forks the resolution (a fork by Python version, from a dependency's own markers,
    # is not a split of the install)
    forked = sorted({p["name"] for p in lock["package"]
                     if any("extra ==" in m for m in p.get("resolution-markers") or [])})
    assert not forked, forked


# -- the Unit Tests workflow -----------------------------------------------------------------

def _workflow(name):
    return yaml.safe_load((REPO / ".github" / "workflows" / name).read_text(encoding="utf-8"))


def test_the_workflow_installs_the_extras_it_gates_on_in_one_step():
    steps = _workflow("test.yml")["jobs"]["pytest"]["steps"]
    runs = "\n".join(s.get("run", "") for s in steps)
    installs = re.findall(r'pip install -e "\.\[([\w,-]+)\]"', runs)
    assert installs == ["dev,evals,omnidocbench"], installs
    gate = re.search(r"--skip-gate-extras ([\w,-]+)", runs).group(1)
    assert sorted(gate.split(",")) == sorted(installs[0].split(","))
    extras = set(_pyproject()["project"]["optional-dependencies"])
    assert set(installs[0].split(",")) <= extras
    # and it checks that pip left no requirement of that one resolve unmet
    assert "python -m pip check" in runs


def test_the_python_legs_match_requires_python_and_the_classifiers():
    from packaging.specifiers import SpecifierSet
    project = _pyproject()["project"]
    legs = _workflow("test.yml")["jobs"]["pytest"]["strategy"]["matrix"]["python-version"]
    spec = SpecifierSet(project["requires-python"])
    classified = {c.rsplit(" :: ", 1)[1] for c in project["classifiers"]
                  if c.startswith("Programming Language :: Python :: 3.")}
    assert set(legs) == classified, (legs, classified)
    for leg in legs:
        assert f"{leg}.0" in spec, leg
    lowest = min(legs, key=lambda v: tuple(map(int, v.split("."))))
    major, minor = map(int, lowest.split("."))
    assert f"{major}.{minor - 1}.99" not in spec, "the lowest leg must be requires-python's floor"


def test_every_workflow_action_is_pinned_to_a_commit():
    unpinned = []
    for path in sorted((REPO / ".github" / "workflows").glob("*.yml")):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            m = re.search(r"uses:\s*([^\s#]+)", line)
            if m and not re.search(r"@[0-9a-f]{40}$", m.group(1)):
                unpinned.append(f"{path.name}:{i}: {m.group(1)}")
    assert not unpinned, unpinned


def test_every_workflow_scopes_its_token_to_reading_the_repository():
    for path in sorted((REPO / ".github" / "workflows").glob("*.yml")):
        assert _workflow(path.name).get("permissions") == {"contents": "read"}, path.name


@pytest.mark.parametrize("doc", ["tests/README.md", "CONTRIBUTING.md"])
def test_the_contributor_docs_give_the_workflows_install(doc):
    text = (REPO / doc).read_text(encoding="utf-8")
    runs = "\n".join(s.get("run", "") for s in _workflow("test.yml")["jobs"]["pytest"]["steps"])
    install = re.search(r'pip install -e "\.\[[\w,-]+\]" transformers', runs).group(0)
    assert install in text, (doc, install)
    assert "terminal-bench]" not in text
    assert "python -m nltk.downloader punkt punkt_tab" in text


# -- the bfcl-eval release the BFCL unit tests imported ---------------------------------------

class _Reporter:
    def __init__(self):
        self.lines = []

    def section(self, title):
        self.lines.append(f"== {title}")

    def write_line(self, line):
        self.lines.append(line)


def _tests_conftest(request):
    target = (REPO / "tests" / "conftest.py").resolve()
    return next(p for p in request.config.pluginmanager.get_plugins()
                if getattr(p, "__file__", None) and Path(p.__file__).resolve() == target)


@pytest.mark.parametrize("installed, noted", [("2025.12.17.1", True), (None, False),
                                              ("pinned", False)])
def test_the_session_summary_names_a_bfcl_eval_release_other_than_the_pin(
        request, monkeypatch, installed, noted):
    import importlib.metadata as md

    from gbench.runners.eval_suites.bfcl_v4_agentic import BFCL_EVAL_RELEASE
    release = BFCL_EVAL_RELEASE if installed == "pinned" else installed

    def fake_version(name):
        if name != "bfcl-eval" or release is None:
            raise md.PackageNotFoundError(name)
        return release

    monkeypatch.setattr(md, "version", fake_version)
    reporter = _Reporter()
    _tests_conftest(request).pytest_terminal_summary(reporter, 0, request.config)
    if noted:
        assert reporter.lines[0] == "== dependencies"
        assert f"bfcl-eval {release} is installed" in reporter.lines[1]
        assert BFCL_EVAL_RELEASE in reporter.lines[1]
    else:
        assert reporter.lines == []
