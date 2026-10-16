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

"""The shared docs follow the code, and the repository-wide gates hold:

* no doc tells a user to install the removed `terminal-bench` extra; the terminal_bench image
  build command the README and toolchains.md give is the one the Dockerfile header gives and
  produces the tag terminal_bench runs;
* toolchains.md's row for the evals extra names every distribution the extra declares;
* every Dockerfile (docker/ and gbench/docker/) pins each external base image by digest;
* scicode checks, before generation, that every module its selected problems import can be
  imported, and a missing one is infra_required, never a failed sub-step;
* the mcp_atlas and mcp_bench key rows of toolchains.md name every credential the code reads,
  and the mcp docs say the keys are checked live;
* no unit test reaches a live service-key check or inherits a real key from the shell;
* the shipping code carries the Apache license header (vendored code and the helpers copied into
  images excepted, each named);
* the provenance stamps describe the current harness (terminal_bench in its image);
* no test file name, test name, doc or comment carries a label of the process that produced it
  (work batches, review roles), and every tests/<file>.py a doc or config cites exists.

No test contacts a network, a model server, Gemini or Docker.
"""

import ast
import os
import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _docs():
    out = [REPO / "README.md", REPO / "CONTRIBUTING.md", REPO / "tests" / "README.md"]
    out += sorted((REPO / "docs").rglob("*.md"))
    return [p for p in out if p.is_file() and ".bak" not in p.name]


# --------------------------------------------------------------------------- #
# terminal_bench: Harbor in its local image, no host extra
# --------------------------------------------------------------------------- #
def test_no_doc_installs_the_removed_terminal_bench_extra():
    extras = tomllib.loads(_read("pyproject.toml"))["project"]["optional-dependencies"]
    assert "terminal-bench" not in extras
    hits = [str(p.relative_to(REPO)) for p in _docs()
            if "terminal-bench]" in p.read_text(encoding="utf-8")]
    assert hits == []


def test_the_terminal_bench_build_command_is_the_dockerfiles_and_makes_the_run_tag():
    from gbench.runners.eval_suites import terminal_bench as TB
    header = _read("gbench/docker/terminal_bench.Dockerfile")
    cmd = re.search(r"^#\s+(docker build -t (\S+) -f gbench/docker/terminal_bench\.Dockerfile "
                    r"gbench/docker)$", header, re.M)
    assert cmd, "the Dockerfile header gives no build command"
    assert cmd.group(2) == TB.HARBOR_IMAGE_DEFAULT == TB.harbor_image()
    for rel in ("docs/installation.md", "docs/evals/toolchains.md", "docs/evals/terminal_bench.md"):
        assert cmd.group(1) in _read(rel), rel
    # the Python row no longer gives Harbor as the reason for 3.12
    readme = _read("docs/installation.md")
    python_row = next(ln for ln in readme.splitlines() if ln.startswith("| **Python Environment**"))
    assert "Harbor" in python_row and "own local images" in python_row
    assert "needs 3.12 or later" not in python_row


def test_the_harness_container_env_in_the_docs_is_the_codes_allowlist():
    from gbench.runners.eval_suites import terminal_bench as TB
    env = {"OPENAI_API_KEY": "EMPTY", "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
           "HARBOR_TELEMETRY": "0", "HTTPS_PROXY": "http://proxy:1", "PATH": "/bin",
           "HOME": "/home/x", "SOME_OTHER": "1"}
    names = TB.container_env_names(env)
    assert names[:2] == ["OPENAI_API_KEY", "OPENAI_BASE_URL"]
    # HARBOR_TELEMETRY is set to 0 in every harness container, never passed from the host
    assert "HARBOR_TELEMETRY" not in names and "HTTPS_PROXY" in names
    assert "PATH" not in names and "HOME" not in names and "SOME_OTHER" not in names
    knobs = " ".join(_read("docs/evals/env-knobs.md").split())
    assert ("the harness container gets only OPENAI_API_KEY=EMPTY, OPENAI_BASE_URL and, when set, "
            "the `HARBOR_*` variables") in knobs
    tools = " ".join(_read("docs/evals/toolchains.md").split())
    assert "Terminus-2 runs in Harbor's process in its harness container on the host network" in tools
    assert "Harbor gets a credential-free environment" not in tools


def test_the_terminal_bench_stamp_describes_the_image():
    from gbench.runners.eval_suites.canonical_sync import CANONICAL_SYNC
    entry = CANONICAL_SYNC["terminal_bench"]
    method, revision = entry["method"], entry["revision"]
    assert "gbench-terminal-bench:harbor-0.20.0" in method
    assert "gbench-terminal-bench:harbor-0.20.0" in revision
    assert "'harbor==0.20.0'" in method
    assert "sys.executable" not in method and "pip install 'harbor" not in method
    assert "harness_image" in method


# --------------------------------------------------------------------------- #
# toolchains.md: the evals row
# --------------------------------------------------------------------------- #
def _dist(req: str) -> str:
    return re.split(r"[<>=!~;\[ ]", req, maxsplit=1)[0].strip().lower().replace("_", "-")


def test_the_evals_row_names_every_declared_distribution():
    extras = tomllib.loads(_read("pyproject.toml"))["project"]["optional-dependencies"]
    row = next(ln for ln in _read("docs/evals/toolchains.md").splitlines()
               if ln.startswith("| `pip install") and "[evals]" in ln.split("|")[1])
    named = {_dist(m) for m in re.findall(r"`([A-Za-z0-9_.\-]+)[^`]*`", row)}
    missing = sorted(_dist(r) for r in extras["evals"] if _dist(r) not in named)
    assert missing == [], f"toolchains.md's evals row omits {missing}"


def test_the_semgrep_row_warns_against_the_serving_environment():
    """semgrep runs only in cyberseceval's local image: toolchains.md gives its build command
    (the tag cyberseceval.py runs), says why it stays off the host, and asks for no host install."""
    from gbench.runners.eval_suites import cyberseceval as C
    lines = _read("docs/evals/toolchains.md").splitlines()
    row = next(ln for ln in lines if ln.startswith(f"| `{C.icd_image_build_command()}` |"))
    assert C.ICD_IMAGE in row and "--network none" in row
    assert "No semgrep and no PurpleLlama checkout on the host" in row
    assert "urllib3~=1.26" in row and "semgrep 1.51.0" in row
    assert "OpenTelemetry" not in row
    assert not [ln for ln in lines if "pip install semgrep" in ln]
    assert not [ln for ln in lines if "GBENCH_CYBERSECEVAL_ICD" in ln]


# --------------------------------------------------------------------------- #
# Dockerfiles: every external base pinned by digest
# --------------------------------------------------------------------------- #
_DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")


def _dockerfiles():
    out = []
    for d in (REPO / "docker", REPO / "gbench" / "docker"):
        out += [p for p in sorted(d.iterdir())
                if p.is_file() and "Dockerfile" in p.name and ".bak" not in p.name]
    return out


def test_every_dockerfile_pins_its_bases_by_digest():
    unpinned = []
    for path in _dockerfiles():
        text = path.read_text(encoding="utf-8")
        args = dict(re.findall(r"^ARG\s+(\w+)=(\S+)", text, re.M))
        stages = set()
        for m in re.finditer(r"^FROM\s+(?:--platform=\S+\s+)?(\S+)(?:\s+[Aa][Ss]\s+(\S+))?",
                             text, re.M):
            ref = re.sub(r"\$\{(\w+)\}|\$(\w+)",
                         lambda a: args.get(a.group(1) or a.group(2), a.group(0)), m.group(1))
            if ref not in stages and not _DIGEST.search(ref):
                unpinned.append(f"{path.relative_to(REPO)}: FROM {m.group(1)}")
            if m.group(2):
                stages.add(m.group(2))
    assert unpinned == []


def test_the_swe_lancer_header_builds_from_the_repository_path():
    text = _read("docker/swe_lancer.Dockerfile")
    assert "-f docker/swe_lancer.Dockerfile" in text
    assert "-f gbench/docker/swe_lancer.Dockerfile" not in text


# --------------------------------------------------------------------------- #
# scicode: every module the problems import is checked before generation
# --------------------------------------------------------------------------- #
def test_scicode_reads_the_modules_the_problems_import():
    from gbench.runners.eval_suites import scicode as S
    problems = [
        {"problem_id": "1", "required_dependencies":
            "import numpy as np\nfrom scipy.integrate import simpson\n"
            "from mpl_toolkits.mplot3d import Axes3D\nimport numpy.linalg as la"},
        {"problem_id": "2", "required_dependencies": "import itertools, math\n"},
        {"problem_id": "3", "required_dependencies": ""},
        {"problem_id": "4", "required_dependencies": "this is not python ("},
    ]
    assert S.dependency_modules(problems) == [
        "itertools", "math", "mpl_toolkits.mplot3d", "numpy", "numpy.linalg", "scipy.integrate"]
    assert S.missing_dependency_modules(["math", "gbench_no_such_module_x",
                                         "json.gbench_no_such_submodule"]) == [
        "gbench_no_such_module_x", "json.gbench_no_such_submodule"]


def test_scicode_refuses_a_missing_module_before_generation(monkeypatch):
    from gbench.runners.eval_suites import scicode as S
    monkeypatch.setattr(S, "sandbox_skip_reason", lambda: None)
    monkeypatch.setattr(S, "resolve_test_data", lambda: ("/nonexistent/test_data.h5", "local"))
    monkeypatch.setattr(S, "check_test_data_digest", lambda path, prov: ("0" * 64, None))
    monkeypatch.setattr(S, "check_cmp_importable", lambda: None)
    monkeypatch.setattr(S, "_load_scicode_problems", lambda limit=None: [{
        "problem_id": "7", "problem_name": "p", "global_idx": 0,
        "required_dependencies": "import numpy as np\nfrom gbench_no_such_module_x import thing",
        "sub_steps": [{"step_number": "7.1", "test_cases": []}]}])

    def _no_generation(*a, **k):
        raise AssertionError("scicode generated before its prerequisites were checked")
    monkeypatch.setattr(S, "_process_one_problem", _no_generation)
    with pytest.raises(RuntimeError) as err:
        S.run_scicode("m", "http://127.0.0.1:9/v1", concurrency=1, max_output_tokens=16)
    msg = str(err.value)
    assert "[INFRA REQUIRED] scicode" in msg and "gbench_no_such_module_x" in msg
    assert "pip install gbench_no_such_module_x" in msg


def test_matplotlib_is_declared_for_scicodes_mpl_toolkits_problem():
    from gbench.runners.eval_suites import scicode as S
    evals = {_dist(r) for r in tomllib.loads(_read("pyproject.toml"))["project"][
        "optional-dependencies"]["evals"]}
    assert S._DEPENDENCY_DISTRIBUTIONS["mpl_toolkits"] in evals
    assert "`matplotlib`" in _read("docs/evals/scicode.md")


# --------------------------------------------------------------------------- #
# mcp_atlas / mcp_bench key docs
# --------------------------------------------------------------------------- #
def test_the_keys_table_names_every_mcp_atlas_credential():
    from gbench.runners.eval_suites import mcp_atlas as MA
    row = next(ln for ln in _read("docs/evals/toolchains.md").splitlines()
               if ln.startswith("| `AIRTABLE_API_KEY`"))
    wanted = {v for s in MA.CREDENTIALED_SERVERS for v in MA.ENVIRONMENT_SERVERS[s]}
    assert wanted, "no credentialed server"
    missing = sorted(v for v in wanted if f"`{v}`" not in row)
    assert missing == []
    assert "`BRAVE_API_KEY`" not in row           # brave-search does not run


def test_the_mcp_docs_say_the_keys_are_checked_live():
    from gbench.runners.eval_suites import mcp_bench as MB
    tools = " ".join(_read("docs/evals/toolchains.md").split())
    knobs = " ".join(_read("docs/evals/env-knobs.md").split())
    assert "Whether a service accepts its key is not checked" not in tools
    assert "whether a service accepts its key is not checked" not in knobs
    assert f"after {MB.KEY_CHECK_ATTEMPTS} attempts" in tools
    assert f"after {MB.KEY_CHECK_ATTEMPTS} attempts" in knobs
    assert "provider_account_error_calls" in knobs
    assert "URL fetching" not in tools


# --------------------------------------------------------------------------- #
# hermetic tests: no live service-key check
# --------------------------------------------------------------------------- #
def test_no_test_reaches_a_live_key_check_or_inherits_a_key():
    from gbench.runners.eval_suites import complexfuncbench as CF
    from gbench.runners.eval_suites import mcp_bench as MB
    for key in ("NPS_API_KEY", "NASA_API_KEY", "HF_TOKEN", "NCI_API_KEY", "GOOGLE_MAPS_API_KEY",
                "RAPID_API_KEY", "GEMINI_API_KEY", "GEMINI_API_KEYS"):
        assert key not in os.environ
    with pytest.raises(pytest.fail.Exception):
        MB._key_check_get("https://example.invalid/", {}, 1.0)
    with pytest.raises(pytest.fail.Exception):
        CF._rapidapi_get("https://example.invalid/", {}, {}, 1.0)


# --------------------------------------------------------------------------- #
# license headers of the shipping code
# --------------------------------------------------------------------------- #
#: Shipping files without gbench's header, each for a reason.
_HEADER_EXEMPT: dict[str, str] = {}
_SHIP_HEADER = re.compile(r"\A(?:#![^\n]*\n)?(?:# -\*- coding: utf-8 -\*-\n)?"
                          r"# Copyright \d{4} Google LLC\n#\n"
                          r"# Licensed under the Apache License, Version 2\.0 \(the \"License\"\);\n")


#: The maintainer's plugin code, not checked here (excluded by name, never opened).
_PLUGIN_FILES = {"swe_bench_pro_agentic.py", "jetski_agent.py", "slashbin_agent.py"}


def _shipping_files():
    out = []
    for top in ("gbench", "scripts", "examples"):
        for p in sorted((REPO / top).rglob("*.py")):
            rel = p.relative_to(REPO).as_posix()
            if (".bak" in p.name or "__pycache__" in p.parts or rel.startswith("gbench/plugins/")
                    or p.name in _PLUGIN_FILES
                    or any(part.endswith("_vendor") or part == "matharena_src"
                           for part in p.parts)):
                continue
            out.append(rel)
    return out


def test_the_shipping_code_carries_the_license_header():
    missing = [rel for rel in _shipping_files() if rel not in _HEADER_EXEMPT
               and not _SHIP_HEADER.match((REPO / rel).read_text(encoding="utf-8"))]
    assert missing == [], "add the Apache license header to: " + ", ".join(missing)


def test_the_header_exemptions_are_current():
    files = set(_shipping_files())
    stale = sorted(rel for rel in _HEADER_EXEMPT if rel not in files
                   or _SHIP_HEADER.match((REPO / rel).read_text(encoding="utf-8")))
    assert stale == [], f"remove from _HEADER_EXEMPT: {stale}"


def test_a_header_leaves_the_code_unchanged():
    # the header is a comment block before the module docstring: the docstring is still the
    # module's, which the code fingerprint (docstring- and comment-free AST) confirms
    from gbench.runners.eval_suites import aa_lcr
    assert (aa_lcr.__doc__ or "").lstrip().startswith("gbench native built-in runner for aa_lcr")
    tree = ast.parse(_read("gbench/runners/eval_suites/aa_lcr.py"))
    assert isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant)


# --------------------------------------------------------------------------- #
# README / distributed docs: run directories and merges
# --------------------------------------------------------------------------- #
def test_the_results_dir_docs_follow_the_log_manager():
    from gbench.utils import log_manager
    # the results-dir text moved to docs/results.md, the --merge row to docs/cli-reference.md
    readme = " ".join((_read("docs/results.md") + "\n" + _read("docs/cli-reference.md")).split())
    assert f"`{log_manager.RUN_LOCK_NAME}`" in readme
    assert "`results_dir_not_shared`" in readme
    assert "results_dir_not_shared" in _read("gbench/cli.py")
    assert "a refusal writes nothing, and with `--dry-run`" in readme
    dist = " ".join(_read("docs/distributed-gbench.md").split())
    assert "de-duplicated by `sample_key`" not in dist
    assert "`merge_unreconciled_fields`" in dist
    for suite in ("mmlu", "mmlu_redux"):
        assert f"`{suite}`" in dist
        from importlib import import_module
        assert hasattr(import_module(f"gbench.runners.eval_suites.{suite}"), "merge_shards")


# --------------------------------------------------------------------------- #
# no internal labels; every cited test file exists
# --------------------------------------------------------------------------- #
#: Labels of the process that produced a change (work batches and rounds, review roles, audit
#: item numbers), never of what the code does. Built from parts, so this file does not hold them.
_CODE_LABEL_PARTS = [
    r"\b[Ww]" + r"ave ?\d", r"\b[Bb]" + r"atch [A-D]\d*\b", r"\buser" + r"chg\b",
    r"\bW" + r"S\d+\b", r"\bmaintainer " + r"rule \d", r"\baudit " + r"(?:RC|P\d|CC)",
    r"\bRC-" + r"\d\b", r"\bCC" + r"\d\b", r"\binteg" + r"rator\b", r"\bfix" + r"er\b",
    r"\bskep" + r"tic", r"\bthis " + r"round\b"]
_PROCESS_LABELS = re.compile("|".join(_CODE_LABEL_PARTS + [r"\bthe " + r"reviewer\b"]))
#: The labels checked in gbench/, whose comments may speak of the judge that reviews an answer
#: (putnam_formal's reviewer).
_BATCH_LABELS = re.compile("|".join(_CODE_LABEL_PARTS))
#: A test module or test function named after a work batch.
_BATCH_NAME = re.compile(r"^test_(?:w" + r"\d|w" + r"s\d|user" + r"chg|batch" + r"_[a-d]_|cc\d_|"
                         r"b(?:[1-9]|1[0-2])[ab]?_)")


def _tracked_text(roots, suffixes=(".py", ".md", ".sh", ".toml", ".yml", ".yaml", ".txt", "")):
    """(relative path, text) of the files a commit of this tree would hold under `roots`
    (git ls-files, else a walk), without backups and the maintainer's plugin code."""
    import subprocess
    try:
        proc = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", *roots], cwd=REPO,
                              capture_output=True, text=True, timeout=60)
        rels = proc.stdout.split() if proc.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        rels = None
    if rels is None:
        rels = [p.relative_to(REPO).as_posix() for r in roots
                for p in ([REPO / r] if (REPO / r).is_file() else (REPO / r).rglob("*"))
                if p.is_file() and "__pycache__" not in p.parts]
    for rel in sorted(set(rels)):
        name = rel.rsplit("/", 1)[-1]
        if (".bak" in name or name.endswith(".orig") or rel.startswith("gbench/plugins/")
                or name in _PLUGIN_FILES or name == "test_swe_bench_pro_agentic.py"):
            continue
        suffix = name[name.rfind("."):] if "." in name else ""
        if suffix not in suffixes or not (REPO / rel).is_file():
            continue
        yield rel, (REPO / rel).read_text(encoding="utf-8", errors="replace")


def _label_hits(roots, pattern):
    return [f"{rel}:{text.count(chr(10), 0, m.start()) + 1}: {m.group(0)}"
            for rel, text in _tracked_text(roots) for m in pattern.finditer(text)]


def test_no_test_file_or_test_is_named_after_a_work_batch():
    bad = [p.name for p in (REPO / "tests").glob("test_*.py") if _BATCH_NAME.match(p.name)]
    for path in (REPO / "tests").glob("test_*.py"):
        if path.name == "test_swe_bench_pro_agentic.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and _BATCH_NAME.match(node.name):
                bad.append(f"{path.name}::{node.name}")
    assert not bad, bad


def test_no_doc_test_or_config_carries_a_process_label():
    hits = _label_hits(["tests", "docs", "README.md", "CONTRIBUTING.md", "NOTICE",
                        "pyproject.toml", ".github", "scripts", "tools", "examples", "docker"],
                       _PROCESS_LABELS)
    assert not hits, "\n".join(hits)


def test_no_shipping_code_carries_a_batch_label():
    hits = _label_hits(["gbench"], _BATCH_LABELS)
    assert not hits, "\n".join(hits)


#: Where a doc, config or code comment cites a tests/<file>.py path.
_CITING_ROOTS = ("README.md", "CONTRIBUTING.md", "NOTICE", "pyproject.toml", "docs",
                 "tests/README.md", ".github", "scripts", "tools", "examples", "docker", "gbench")
#: Citations of a test file that is not there on purpose.
_CITED_ELSEWHERE = {
    ("tools/template_checks/README.md", "test_chat_template_gemma4.py"):
        "says where the template check used to sit",
    ("docs/evals_roadmap.md", "test_foo.py"): "the example of a new suite's test file",
}


def test_every_test_file_a_doc_or_config_cites_exists():
    missing = []
    for rel, text in _tracked_text(_CITING_ROOTS):
        for m in re.finditer(r"(?<![\w/])tests/(test_\w+\.py)", text):
            if (rel, m.group(1)) in _CITED_ELSEWHERE:
                continue
            if not (REPO / "tests" / m.group(1)).is_file():
                missing.append(f"{rel}:{text.count(chr(10), 0, m.start()) + 1}: {m.group(0)}")
    # tests/README.md also names test files without the directory
    for m in re.finditer(r"(?<![\w/])(test_\w+\.py)", _read("tests/README.md")):
        if not any((REPO / d / m.group(1)).is_file() for d in ("tests", "tools/template_checks")):
            missing.append(f"tests/README.md: {m.group(1)}")
    assert not missing, "\n".join(missing)
    for rel, name in _CITED_ELSEWHERE:
        assert f"tests/{name}" in _read(rel), (rel, name)


def test_the_label_patterns_catch_what_they_are_for():
    for text in ("W" + "ave 2b fixes", "(b" + "atch B11)", "W" + "S10 audit", "C" + "C7",
                 "audit " + "R" + "C-1", "the " + "reviewer's repro", "user" + "chg"):
        assert _PROCESS_LABELS.search(text), text
    for name in ("test_w" + "2o_mcp.py", "test_w" + "s5_concurrency.py", "test_c" + "c4_x",
                 "test_b" + "6_docs", "test_batch" + "_b_scoring.py", "test_user" + "chg_x.py"):
        assert _BATCH_NAME.match(name), name
    for fine in ("test_waveform.py", "test_bfcl_v4_agentic.py", "test_batch_sizes_default",
                 "test_b64_decoding", "test_builtin_scoring.py"):
        assert not _BATCH_NAME.match(fine), fine
