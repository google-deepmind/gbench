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

"""Build pins of the local images: what a rebuild reproduces, and what it records.

  * every Docker CLI package an image installs (docker-ce-cli and the compose and buildx
    plugins) is pinned to an exact version and checked in the build's dpkg record, and the
    Ubuntu images that install one do so from a dated Ubuntu archive snapshot;
  * NodeSource's Node.js, uv and get-pip.py are installed at pinned releases;
  * the images whose Python packages resolved at build time (the mini-swe-agent virtualenvs,
    Harbor, the WildClawBench orchestrator, ARE, OSWorld, MCP-Bench's system interpreter,
    StableToolBench) install against a constraints file taken from the checked image and fail
    the build when anything else is installed;
  * mcp_bench resolves its build's npm packages as of a fixed time and checks the managed CPython
    builds uv downloaded;
  * the MultiPL-E evaluator recipe gets its base image pinned by digest and a record step;
  * toolbench's per-step record counts the context-window rejections left to upstream apart.

Nothing is built or pulled; docker, Gemini, the network and the model endpoint are not used.
"""

import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ROOT_DOCKER = REPO / "docker"
PKG_DOCKER = REPO / "gbench" / "docker"
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _read(rel):
    return (REPO / rel).read_text(encoding="utf-8")


def _dockerfiles():
    """Every Dockerfile gbench builds (a file whose header says it is no longer used aside)."""
    out = []
    for d in (ROOT_DOCKER, PKG_DOCKER):
        for p in sorted(d.glob("*.Dockerfile")):
            if ".bak" in p.name:
                continue
            text = p.read_text(encoding="utf-8")
            if "This file is no longer used" in text.split("\nFROM ", 1)[0]:
                continue
            out.append((p.relative_to(REPO).as_posix(), text))
    assert len(out) >= 25, [r for r, _ in out]
    return out


def _code(text):
    """The Dockerfile without its comment lines."""
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


def _args(text):
    return dict(re.findall(r"^ARG\s+(\w+)=(\S+)", text, re.M))


def _checker():
    spec = importlib.util.spec_from_file_location("pinned_pip_check",
                                                  ROOT_DOCKER / "pinned_pip_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =========================================================================== #
# The Docker CLI packages
# =========================================================================== #
_DOCKER_PACKAGES = ("docker-ce-cli", "docker-compose-plugin", "docker-buildx-plugin")
_APT_VERSION = re.compile(r"^(\d+:)?\d+\.\d+\.\d+-\d+~(ubuntu|debian)\.[\d.]+~\w+$")

#: The versions the checked images carry (dpkg-query in each), pinned by the Dockerfiles that
#: were floating.
_CHECKED_DOCKER_VERSIONS = {
    "gbench/docker/mini_swe_agent_v2.Dockerfile": {
        "DOCKER_CLI_VERSION": "5:29.8.1-1~ubuntu.24.04~noble"},
    "gbench/docker/swe_bench_pro_v2.Dockerfile": {
        "DOCKER_CLI_VERSION": "5:29.8.1-1~ubuntu.24.04~noble",
        "DOCKER_COMPOSE_VERSION": "5.5.1-1~ubuntu.24.04~noble",
        "DOCKER_BUILDX_VERSION": "0.37.1-1~ubuntu.24.04~noble"},
    "gbench/docker/wildclawbench.Dockerfile": {
        "DOCKER_CLI_VERSION": "5:29.8.1-1~ubuntu.24.04~noble"},
    "gbench/docker/skillsbench.Dockerfile": {
        "DOCKER_CLI_VERSION": "5:29.8.1-1~ubuntu.24.04~noble",
        "DOCKER_COMPOSE_VERSION": "5.5.1-1~ubuntu.24.04~noble"},
}


def test_every_docker_cli_package_is_installed_at_a_pinned_version_and_checked():
    seen = 0
    for rel, text in _dockerfiles():
        code = _code(text)
        args = _args(text)
        for m in re.finditer(r"\b(" + "|".join(_DOCKER_PACKAGES) + r")\b(\S*)", code):
            pkg, rest = m.group(1), m.group(2)
            pin = re.match(r"=\$\{(\w+)\}", rest)
            assert pin, f"{rel}: {pkg} without a pinned version: {m.group(0)}"
            var = pin.group(1)
            assert _APT_VERSION.match(args.get(var, "")), (rel, var, args.get(var))
            assert re.search(r'grep -qx "' + re.escape(pkg) + r"=\$\{" + var + r'\}" ', code), (
                f"{rel}: {pkg} is not checked in the build's dpkg record")
            seen += 1
    assert seen >= 12, seen


def test_the_formerly_floating_docker_cli_images_pin_what_the_checked_images_have():
    for rel, want in _CHECKED_DOCKER_VERSIONS.items():
        args = _args(_read(rel))
        for var, version in want.items():
            assert args.get(var) == version, (rel, var, args.get(var))


def test_ubuntu_images_with_the_docker_cli_install_from_a_dated_archive_snapshot():
    checked = 0
    for rel, text in _dockerfiles():
        if not re.search(r"^FROM ubuntu:24\.04@sha256:[0-9a-f]{64}$", text, re.M):
            continue
        if "docker-ce-cli" not in _code(text):
            continue
        checked += 1
        assert re.search(r"^ARG UBUNTU_SNAPSHOT=\d{8}T\d{6}Z$", text, re.M), rel
        assert "https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" in text, rel
        assert ('test "$(grep -c "snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" '
                '/etc/apt/sources.list.d/ubuntu.sources)" = 2') in text, rel
        assert "--allow-downgrades" in text, rel
        assert re.search(r"dpkg-query -W -f '\$\{Package\}=\$\{Version\}\\n' > \S+", text), rel
    assert checked >= 7, checked


# =========================================================================== #
# Node.js, uv and pip installers
# =========================================================================== #
def test_nodesource_installs_are_pinned_to_one_build_and_checked():
    checked = 0
    for rel, text in _dockerfiles():
        code = _code(text)
        setup = re.search(r"deb\.nodesource\.com/setup_(\d+)\.x", code)
        if not setup:
            continue
        checked += 1
        version = _args(text).get("NODE_VERSION", "")
        assert re.fullmatch(setup.group(1) + r"\.\d+\.\d+-1nodesource1", version), (rel, version)
        pinned = ('"nodejs=${NODE_VERSION}"' in code
                  or "Package: nodejs\\nPin: version %s\\nPin-Priority: 1001" in code)
        assert pinned, f"{rel}: the nodejs install is not pinned to NODE_VERSION"
        assert """test "$(dpkg-query -W -f '${Version}' nodejs)" = "${NODE_VERSION}\"""" in code, rel
    assert checked == 3, checked


def test_aider_pins_node_and_resolves_its_npm_dependencies_as_of_a_fixed_time():
    text = _read("docker/aider_polyglot.Dockerfile")
    code = _code(text)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", _args(text)["NPM_CONFIG_BEFORE"])
    assert "ENV NPM_CONFIG_BEFORE" not in code
    assert code.index("ARG NPM_CONFIG_BEFORE=") < code.index("npm install \\")
    # the header still names its differences from upstream's benchmark/Dockerfile
    assert "with four differences" in text and "the JavaScript toolchain pinned" in text


def test_the_mcp_atlas_node_pin_keeps_the_release_step_verbatim():
    text = _read("docker/mcp_atlas.Dockerfile")
    setup = text.index("RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && apt-get update")
    pin = text.index("/etc/apt/preferences.d/gbench-nodejs")
    release = text.index("RUN apt-get install -y nodejs && rm -rf /var/lib/apt/lists/*")
    assert setup < pin < release
    assert "gbench:" in text[setup:release]
    assert "dpkg-query -W -f '${Package}=${Version}\\n' > /opt/gbench/dpkg-versions.txt" in text


def test_uv_and_get_pip_are_installed_at_pinned_releases():
    for rel, text in _dockerfiles():
        code = _code(text)
        assert "astral.sh/uv/install.sh" not in code, f"{rel}: the unversioned uv installer"
        if "astral.sh/uv/${UV_VERSION}/install.sh" in code:
            assert re.fullmatch(r"\d+\.\d+\.\d+", _args(text).get("UV_VERSION", "")), rel
            assert 'uv --version | grep -q "^uv ${UV_VERSION} "' in code, rel
        for m in re.finditer(r"python[\d.]*\s+\S*get-pip\.py[^\n]*", code):
            assert '"pip==${PIP_VERSION}"' in m.group(0), (rel, m.group(0))


# =========================================================================== #
# Python sets that resolved at build time: installed against the checked image's set
# =========================================================================== #
#: Dockerfile -> (constraints file in the build context, its path in the image, the check).
_CONSTRAINED = {
    "gbench/docker/mini_swe_agent_v2.Dockerfile": (
        "gbench/docker/mini_swe_agent_v2.constraints.txt",
        "/opt/gbench/mini_swe_agent_v2.constraints.txt", "| diff /tmp/pins.txt -"),
    "gbench/docker/swe_bench_pro_v2.Dockerfile": (
        "gbench/docker/swe_bench_pro_v2.constraints.txt",
        "/opt/gbench/swe_bench_pro_v2.constraints.txt", "| diff - /opt/harbor-pip-freeze.txt"),
    "gbench/docker/wildclawbench.Dockerfile": (
        "gbench/docker/wildclawbench.constraints.txt",
        "/opt/gbench/wildclawbench.constraints.txt", "| diff - /opt/gbench/pip-freeze.txt"),
    "gbench/docker/gaia2.Dockerfile": (
        "gbench/docker/gaia2.constraints.txt",
        "/opt/gbench/gaia2.constraints.txt", "| diff - /opt/gbench/pip-freeze.txt"),
    "gbench/docker/ui_control_osworld.Dockerfile": (
        "gbench/docker/ui_control_osworld.constraints.txt",
        "/opt/gbench/ui_control_osworld.constraints.txt", "| diff - /opt/gbench/pip-freeze.txt"),
    "gbench/docker/mcp_bench.Dockerfile": (
        "gbench/docker/mcp_bench_constraints.txt", "/app/constraints.txt",
        "| diff - /opt/gbench/pip-freeze.txt"),
    "docker/toolbench.Dockerfile": (
        "docker/toolbench_constraints.txt", "/tmp/toolbench_constraints.txt",
        "python /opt/gbench/pinned_pip_check.py /tmp/toolbench_constraints.txt"),
}


@pytest.mark.parametrize("dockerfile", sorted(_CONSTRAINED))
def test_the_resolved_python_set_is_installed_against_its_constraints_and_checked(dockerfile):
    constraints, dest, check = _CONSTRAINED[dockerfile]
    pins = _checker().read_pins(str(REPO / constraints))
    assert len(pins) >= 5, constraints
    assert all(v and " " not in v for _n, v in pins.values()), constraints
    text = _read(dockerfile)
    code = _code(text)
    name = constraints.rsplit("/", 1)[-1]
    assert re.search(r"^COPY " + re.escape(name) + r" " + re.escape(dest) + r"$", code, re.M), (
        dockerfile, name, dest)
    assert f"PIP_CONSTRAINT={dest}" in code, dockerfile
    assert check in code, dockerfile
    if "diff" in check:
        # the constraints file and the build's freeze are compared whole: every line of the
        # freeze is pinned, and nothing pinned is missing
        assert "freeze --all" in code, dockerfile
        assert f"grep -v -e '^#' -e '^$' {dest} | LC_ALL=C sort" in code, dockerfile


def test_each_constraints_file_pins_the_release_its_dockerfile_installs():
    pins = {rel: _checker().read_pins(str(REPO / c)) for rel, (c, _d, _k) in _CONSTRAINED.items()}
    args = {rel: _args(_read(rel)) for rel in _CONSTRAINED}
    # the virtualenvs: each release is installed by name, the rest of the set comes from the file
    assert "mini-swe-agent" not in pins["gbench/docker/mini_swe_agent_v2.Dockerfile"]
    assert pins["gbench/docker/mini_swe_agent_v2.Dockerfile"]["pip"][1] == "24.0"
    harbor = args["gbench/docker/swe_bench_pro_v2.Dockerfile"]["HARBOR_VERSION"]
    assert pins["gbench/docker/swe_bench_pro_v2.Dockerfile"]["harbor"][1] == harbor
    are = args["gbench/docker/gaia2.Dockerfile"]["ARE_VERSION"]
    assert pins["gbench/docker/gaia2.Dockerfile"]["meta-agents-research-environments"][1] == are
    assert pins["gbench/docker/ui_control_osworld.Dockerfile"]["docker"][1].startswith("7.")
    assert {"python-dotenv", "pyyaml"} <= set(pins["gbench/docker/wildclawbench.Dockerfile"])
    mcp = pins["gbench/docker/mcp_bench.Dockerfile"]
    assert mcp["mcp"][1] == "1.16.0"
    mcp_args = args["gbench/docker/mcp_bench.Dockerfile"]
    for pkg, var in (("pip", "PIP_VERSION"), ("setuptools", "SETUPTOOLS_VERSION"),
                     ("wheel", "WHEEL_VERSION")):
        assert mcp[pkg][1] == mcp_args[var], pkg
    tb = pins["docker/toolbench.Dockerfile"]
    assert {"setuptools", "wheel", "torch", "transformers"} <= set(tb)


def test_toolbench_records_its_python_and_debian_sets():
    text = _code(_read("docker/toolbench.Dockerfile"))
    assert "COPY pinned_pip_check.py /opt/gbench/pinned_pip_check.py" in text
    check = text.index("python /opt/gbench/pinned_pip_check.py /tmp/toolbench_constraints.txt")
    # after the last pip install and before the patches
    assert text.rindex("pip install --no-cache-dir") < check < text.index("toolbench_patch_llm.py")
    assert "--freeze /opt/gbench/pip-freeze.txt" in text
    assert "dpkg-query -W -f '${Package}=${Version}\\n' > /opt/gbench/dpkg-versions.txt" in text


def test_skillsbench_builds_its_environment_on_the_system_python_and_records_it():
    text = _code(_read("gbench/docker/skillsbench.Dockerfile"))
    assert "RUN uv sync --locked" in text
    assert "sys.base_prefix == '/usr'" in text
    assert "uv pip freeze --python .venv/bin/python > /opt/gbench/pip-freeze.txt" in text
    # the compose plugin comes from the same pinned apt step as the CLI
    assert text.count("apt-get install") == 3


# =========================================================================== #
# mcp_bench: toolchain, npm resolution, managed CPython builds
# =========================================================================== #
def test_mcp_bench_pins_node_pip_uv_and_the_build_time_npm_resolution():
    text = _read("gbench/docker/mcp_bench.Dockerfile")
    code = _code(text)
    args = _args(text)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", args["NPM_CONFIG_BEFORE"])
    # a build argument (the build's RUN steps only), declared before install.sh runs
    assert "ENV NPM_CONFIG_BEFORE" not in code
    assert code.index("ARG NPM_CONFIG_BEFORE=") < code.index("bash -c \"$(sed 's/^set -e.*/set +e/' install.sh)\"")
    assert '"nodejs=${NODE_VERSION}"' in code and args["NODE_VERSION"].startswith("22.")
    assert ('python3 /tmp/get-pip.py --no-cache-dir "pip==${PIP_VERSION}" \\\n'
            '        "setuptools==${SETUPTOOLS_VERSION}" "wheel==${WHEEL_VERSION}"') in code
    assert 'astral.sh/uv/${UV_VERSION}/install.sh' in code


def test_mcp_bench_checks_its_installed_set_editable_servers_and_managed_pythons():
    text = _read("gbench/docker/mcp_bench.Dockerfile")
    code = _code(text)
    check = code[code.index("ARG UV_MANAGED_PYTHONS="):]
    check = check[:check.index("\nLABEL ")]
    assert "python3 -m pip freeze --all --exclude-editable | LC_ALL=C sort > /opt/gbench/pip-freeze.txt" in check
    assert '= "bibliomantic_mcp_server mcp_reddit_server mcp_server_time unit_converter_mcp "' in check
    assert '= "${UV_MANAGED_PYTHONS} "' in check
    managed = _args(text)["UV_MANAGED_PYTHONS"].strip('"')
    assert re.fullmatch(r"(cpython-\d+\.\d+\.\d+-linux-x86_64-gnu ?)+", managed)
    assert "|| true" not in check and "; true" not in check
    # after the last install into the system interpreter
    assert code.index('python3 -m pip install --no-cache-dir "mcp[cli]==1.16.0"') < code.index(
        "ARG UV_MANAGED_PYTHONS=")


def test_the_mcp_bench_checks_parse_as_shell(tmp_path):
    """The check step's shell (the RUN after ARG UV_MANAGED_PYTHONS) is valid POSIX sh."""
    code = _code(_read("gbench/docker/mcp_bench.Dockerfile"))
    run = code[code.index("ARG UV_MANAGED_PYTHONS="):]
    run = run[run.index("RUN ") + 4:run.index("\nLABEL ")]
    script = tmp_path / "check.sh"
    script.write_text(run + "\n")
    assert subprocess.run(["sh", "-n", str(script)], capture_output=True).returncode == 0


# =========================================================================== #
# MultiPL-E: the evaluator recipe's base and record step
# =========================================================================== #
def _multipl_e_rewrite():
    """The source stage's commands that rewrite the evaluator recipe (the sed and the appended
    record step), as a shell script."""
    text = _read("docker/multipl_e.Dockerfile")
    start = text.index("    sed -i \\\n")
    end = text.index(">> Dockerfile; \\\n", start) + len(">> Dockerfile")
    return text[start:end].replace("Dockerfile; \\\n    printf", "Dockerfile\n    printf")


def test_the_multipl_e_recipe_base_is_pinned_to_the_source_stages_digest(tmp_path):
    text = _read("docker/multipl_e.Dockerfile")
    own = re.search(r"^FROM ubuntu:22\.04(@sha256:[0-9a-f]{64})$", text, re.M).group(1)
    recipe = tmp_path / "Dockerfile"
    recipe.write_text(
        "FROM ubuntu:22.04\nARG DEBIAN_FRONTEND=noninteractive\n"
        "RUN mkdir /usr/multiple && wget https://repo.mavenlibs.com/maven/org/javatuples/"
        "javatuples/1.2/javatuples-1.2.jar -O /usr/multiple/javatuples-1.2.jar\n"
        "COPY src /code\nWORKDIR /code\nENTRYPOINT [\"python3\", \"main.py\"]\n")
    script = tmp_path / "rewrite.sh"
    script.write_text("set -eu\n" + _multipl_e_rewrite() + "\n")
    proc = subprocess.run(["sh", str(script)], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    out = recipe.read_text().splitlines()
    assert out[0] == "FROM ubuntu:22.04" + own
    assert [ln for ln in out if ln.startswith("FROM ")] == [out[0]]
    assert "repo1.maven.org/maven2/org/javatuples/javatuples/1.2/javatuples-1.2.jar" in out[2]
    assert out[2].endswith("/usr/multiple/javatuples-1.2.jar\" | sha256sum -c -")
    record = out[-1]
    assert record.startswith("RUN mkdir -p /opt/gbench && dpkg-query -W -f '${Package}=${Version}\\n'")
    for name in ("dpkg-versions.txt", "pip-freeze.txt", "npm-global.txt", "clojure-tools.txt"):
        assert f"/opt/gbench/{name}" in record
    # the rewritten recipe is still checked against its own sha256
    assert re.search(r'echo "[0-9a-f]{64}  Dockerfile" \\\n\s+\| sha256sum --strict -c -', text)


# =========================================================================== #
# toolbench: context-window rejections left to upstream are counted apart
# =========================================================================== #
@pytest.fixture
def tb_client(monkeypatch, tmp_path):
    from test_delegated_harness_request_settings import _UPSTREAM_CHATGPT
    from test_gaia2_toolbench_capabilities_and_endpoint_failures import _CHATGPT_FUNCTION
    target = tmp_path / "chatgpt_function_model.py"
    target.write_text(_UPSTREAM_CHATGPT + _CHATGPT_FUNCTION)
    patch_src = (ROOT_DOCKER / "toolbench_patch_llm.py").read_text(encoding="utf-8").replace(
        'PATH = "/stb/toolbench/inference/LLM/chatgpt_function_model.py"', f"PATH = {str(target)!r}")
    exec(compile(patch_src, "toolbench_patch_llm.py", "exec"), {"__name__": "__main__"})
    monkeypatch.setitem(sys.modules, "termcolor", types.SimpleNamespace(colored=lambda *a, **k: ""))
    monkeypatch.syspath_prepend(str(ROOT_DOCKER))
    monkeypatch.syspath_prepend(str(tmp_path))
    log = tmp_path / "model_request_failures.jsonl"
    monkeypatch.setenv("TB_MODEL_FAILURE_LOG", str(log))
    monkeypatch.setenv("TB_GROUP", "G1_instruction")
    monkeypatch.setenv("TB_ENABLE_THINKING", "false")
    monkeypatch.setenv("TB_REQUEST_LOG", str(tmp_path / "requests.jsonl"))
    for k in ("TB_ENDPOINT", "TB_TEMPERATURE", "TB_MAX_TOKENS", "TB_REQUEST_TIMEOUT_S"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delitem(sys.modules, "chatgpt_function_model", raising=False)
    mod = importlib.import_module("chatgpt_function_model")
    plan = []

    class BadRequestError(Exception):
        status_code = 400

    class _Resp:
        choices = [types.SimpleNamespace(finish_reason="stop", message=types.SimpleNamespace(
            content="hi", tool_calls=None))]

        def dict(self):
            return {"choices": [{"message": {"role": "assistant", "content": "hi"}}],
                    "usage": {"total_tokens": 3}}

    errors = {
        "window": BadRequestError("This model's maximum context length is 8192 tokens. However, "
                                  "you requested 9100 tokens (8076 in the messages, 1024 in the "
                                  "completion)."),
        "content": BadRequestError("bad request: invalid tool"),
    }

    def create(**kw):
        step = plan.pop(0)
        if step in errors:
            raise errors[step]
        return _Resp()

    class _OpenAI:
        def __init__(self, base_url=None, api_key=None):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=create))

    monkeypatch.setattr(mod, "OpenAI", _OpenAI)
    llm = mod.ChatGPTFunction(base_url="http://127.0.0.1:1/v1")
    llm._gbench_query_id = 5
    llm.change_messages([{"role": "user", "content": "q"}])
    return types.SimpleNamespace(mod=mod, llm=llm, plan=plan, log=log)


def _records(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def test_a_context_window_rejection_left_to_upstream_is_counted_apart(tb_client):
    tb_client.plan[:] = ["window"] * 6
    msg, code, _ = tb_client.llm.parse([], 0)
    assert code == -1 and "maximum context length" in msg["content"]
    [rec] = _records(tb_client.log)
    assert rec["content_rejections"] == 6 and rec["over_context_rejections"] == 6
    assert rec["rejected"] is True and rec["unrecovered"] is False
    assert rec["endpoint_failures"] == 0
    policy = tb_client.mod._GBENCH_POLICY
    assert policy.counts["rejected_over_context"] == 6 and policy.failed_by_error == {}


def test_a_content_rejection_has_no_over_context_count(tb_client):
    tb_client.plan[:] = ["content", "window", "ok"]
    assert tb_client.llm.parse([], 0)[1] == 0
    [rec] = _records(tb_client.log)
    assert rec["content_rejections"] == 2 and rec["over_context_rejections"] == 1
    assert rec["rejected"] is False
    tb_client.plan[:] = ["content", "ok"]
    tb_client.llm.parse([], 0)
    last = _records(tb_client.log)[-1]
    assert last["content_rejections"] == 1 and "over_context_rejections" not in last
