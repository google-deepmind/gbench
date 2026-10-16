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

"""The cloud metadata server guard (container_guard.require_metadata_blocked).

Docker is stubbed in every test here: no container starts and nothing contacts 169.254.169.254.
The probe scripts themselves run against a socket on 127.0.0.1 only.

* `--network none` needs no probe; any other network mode is probed once, then cached.
* A connection that opens is infra_required naming the host firewall rule and its persistence;
  the host network gets its own message (DOCKER-USER does not see it).
* No verdict (docker missing, image missing, docker timeout, an unexpected socket error, no
  python3 and no bash in the image) is infra_required too: never a pass.
* The probe never pulls, sends nothing on the connection and asks for no token.
* It also tries the IPv6 metadata address; no IPv6 route is unreachable, a connection that opens
  is infra_required naming the ip6tables rule.
* With `resolve`, the same probe resolves the suite's host names: a name that does not resolve is
  infra_required naming the DNS side of the host rule; no clear answer is never a pass; resolved
  names are cached per (network, name), failed ones are not.
* The `docker rm -f` after a probe timeout is bounded, and the gate raises whatever it does.
"""

import socket
import subprocess
import sys
import threading
import time

import pytest

from gbench.runners.eval_suites import container_guard as cg


class FakeDocker:
    """Records every docker argv; answers `docker run` from a list of (rc, stdout, stderr)."""

    def __init__(self, answers=None, raise_on_run=None):
        self.calls = []
        self.answers = list(answers or [])
        self.raise_on_run = raise_on_run

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        if argv[:2] == ["docker", "rm"]:
            return subprocess.CompletedProcess(argv, 0, "", "")
        if self.raise_on_run is not None:
            raise self.raise_on_run
        rc, out, err = self.answers.pop(0)
        return subprocess.CompletedProcess(argv, rc, out, err)

    @property
    def runs(self):
        return [c for c in self.calls if c[:2] == ["docker", "run"]]


@pytest.fixture(autouse=True)
def _fresh_cache():
    cg.reset_verdicts()
    yield
    cg.reset_verdicts()


def _install(monkeypatch, fake):
    monkeypatch.setattr(cg.subprocess, "run", fake)
    return fake


REACH = (0, "GBENCH_METADATA_PROBE=reachable\n", "")
UNREACH = (0, "GBENCH_METADATA_PROBE=unreachable timeout\n", "")
NOT_FOUND = (127, "", 'docker: Error response from daemon: failed to create task: exec: '
                     '"python3": executable file not found in $PATH: unknown.\n')


def test_network_none_needs_no_probe(monkeypatch):
    fake = _install(monkeypatch, FakeDocker())
    assert cg.require_metadata_blocked("mcp_atlas", "img:1", "none") == cg.NO_NETWORK
    assert fake.calls == [] and cg.cached_verdicts() == {}


def test_unreachable_passes_and_is_cached_per_network(monkeypatch):
    fake = _install(monkeypatch, FakeDocker([UNREACH, UNREACH]))
    assert cg.require_metadata_blocked("mcp_atlas", "img:1", "bridge") == cg.UNREACHABLE
    assert cg.require_metadata_blocked("wildclawbench", "other:2", "bridge") == cg.UNREACHABLE
    assert len(fake.runs) == 1
    assert cg.require_metadata_blocked("mcp_atlas", "img:1", "run-net") == cg.UNREACHABLE
    assert len(fake.runs) == 2
    assert cg.cached_verdicts() == {"bridge": cg.UNREACHABLE, "run-net": cg.UNREACHABLE}


def test_default_network_is_the_bridge(monkeypatch):
    fake = _install(monkeypatch, FakeDocker([UNREACH]))
    for net in (None, "", "default", "bridge"):
        assert cg.require_metadata_blocked("x", "img:1", net) == cg.UNREACHABLE
    assert len(fake.runs) == 1
    argv = fake.runs[0]
    assert argv[argv.index("--network") + 1] == "bridge"


def test_reachable_is_infra_required_naming_the_rule_and_its_persistence(monkeypatch):
    fake = _install(monkeypatch, FakeDocker([REACH]))
    with pytest.raises(RuntimeError) as ei:
        cg.require_metadata_blocked("mcp_atlas", "gbench-mcp-atlas:f24ba3fb", "bridge",
                                    "docs/evals/mcp_atlas.md")
    msg = str(ei.value)
    assert "[INFRA REQUIRED] mcp_atlas" in msg
    assert "sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP" in msg
    assert cg.PERSIST_COMMAND in msg and "persistent" in msg
    assert "gbench never runs sudo" in msg and "'bridge'" in msg
    assert "docs/evals/mcp_atlas.md" in msg
    # cached: the next call raises again without a second probe
    with pytest.raises(RuntimeError):
        cg.require_metadata_blocked("mcp_atlas", "gbench-mcp-atlas:f24ba3fb", "bridge")
    assert len(fake.runs) == 1


def test_host_network_message_says_docker_user_does_not_cover_it(monkeypatch):
    _install(monkeypatch, FakeDocker([REACH]))
    with pytest.raises(RuntimeError) as ei:
        cg.require_metadata_blocked("wildclawbench", "wildclawbench-ubuntu:v1.3", "host")
    msg = str(ei.value)
    assert "DOCKER-USER chain does not filter the host network" in msg
    assert cg.FIREWALL_COMMAND in msg and "OUTPUT" not in msg
    assert "guest agent" in msg
    assert "bridge or user-defined docker network" in msg


def test_the_probe_argv_never_pulls_and_carries_nothing_of_the_host(monkeypatch):
    fake = _install(monkeypatch, FakeDocker([UNREACH]))
    cg.require_metadata_blocked("x", "local/img:tag", "run-net")
    argv = fake.runs[0]
    assert argv[:3] == ["docker", "run", "--rm"]
    assert argv[argv.index("--pull") + 1] == "never"
    assert argv[argv.index("--network") + 1] == "run-net"
    assert "--read-only" in argv and argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv[argv.index("--label") + 1] == cg.PROBE_LABEL
    assert argv[argv.index("--entrypoint") + 1] == "python3"
    assert argv[argv.index("--entrypoint") + 2] == "local/img:tag"
    assert not any(a in ("-e", "--env", "-v", "--volume", "--env-file", "--privileged")
                   for a in argv)
    script = argv[-1]
    assert "169.254.169.254" in script and "80" in script
    for needle in ("computeMetadata", "token", "Metadata-Flavor", "urllib", "http", "send"):
        assert needle not in script, needle


def test_the_bash_probe_asks_for_nothing_either():
    script = cg.bash_probe_script()
    assert "/dev/tcp/169.254.169.254/80" in script and "timeout 3" in script
    for needle in ("computeMetadata", "token", "Metadata-Flavor", "curl", "wget", "printf", ">&3"):
        assert needle not in script, needle


def test_without_python3_the_bash_probe_runs(monkeypatch):
    fake = _install(monkeypatch, FakeDocker([NOT_FOUND, UNREACH]))
    assert cg.require_metadata_blocked("x", "img:1", "bridge") == cg.UNREACHABLE
    assert [r[r.index("--entrypoint") + 1] for r in fake.runs] == ["python3", "bash"]
    assert fake.runs[1][-1] == cg.bash_probe_script()


def test_without_python3_and_bash_is_infra_required(monkeypatch):
    _install(monkeypatch, FakeDocker([NOT_FOUND, (127, "", "exec: \"bash\": not found\n")]))
    with pytest.raises(RuntimeError) as ei:
        cg.require_metadata_blocked("x", "alpine/socat:latest", "bridge")
    msg = str(ei.value)
    assert "needs python3, or bash with coreutils timeout" in msg
    assert "python3, bash not found" in msg and "alpine/socat:latest" in msg
    assert cg.cached_verdicts() == {}


@pytest.mark.parametrize("answer, needle", [
    ((125, "", "docker: Error response from daemon: No such image: img:1\n"), "No such image"),
    ((125, "", "docker: Error response from daemon: network run-net not found.\n"), "not found"),
    ((0, "GBENCH_METADATA_PROBE=error 22\n", ""), "error 22"),
    ((0, "GBENCH_METADATA_PROBE=error no-timeout-command\n", ""), "no-timeout-command"),
    ((1, "Traceback (most recent call last):\n", ""), "Traceback"),
    ((0, "", ""), "no output"),
])
def test_no_clear_verdict_is_never_a_pass(monkeypatch, answer, needle):
    _install(monkeypatch, FakeDocker([answer]))
    with pytest.raises(RuntimeError) as ei:
        cg.require_metadata_blocked("x", "img:1", "run-net")
    msg = str(ei.value)
    assert "gave no verdict" in msg and needle in msg
    assert cg.cached_verdicts() == {}


def test_an_unexpected_python_error_does_not_fall_back_to_bash(monkeypatch):
    fake = _install(monkeypatch, FakeDocker([(0, "GBENCH_METADATA_PROBE=error 97\n", "")]))
    with pytest.raises(RuntimeError):
        cg.require_metadata_blocked("x", "img:1", "bridge")
    assert len(fake.runs) == 1


def test_docker_missing_is_infra_required(monkeypatch):
    _install(monkeypatch, FakeDocker(raise_on_run=FileNotFoundError("docker")))
    with pytest.raises(RuntimeError) as ei:
        cg.require_metadata_blocked("x", "img:1", "bridge")
    assert "docker CLI is not on PATH" in str(ei.value)


def test_a_docker_timeout_removes_the_probe_and_is_infra_required(monkeypatch):
    fake = _install(monkeypatch, FakeDocker(
        raise_on_run=subprocess.TimeoutExpired(["docker", "run"], cg.PROBE_DOCKER_TIMEOUT_S)))
    with pytest.raises(RuntimeError) as ei:
        cg.require_metadata_blocked("x", "img:1", "bridge")
    assert f"did not finish within {cg.PROBE_DOCKER_TIMEOUT_S} s" in str(ei.value)
    name = fake.calls[0][fake.calls[0].index("--name") + 1]
    assert name.startswith("gbench-metadata-probe-")
    assert ["docker", "rm", "-f", name] in fake.calls
    assert cg.cached_verdicts() == {}


def test_docker_output_in_the_message_is_redacted(monkeypatch):
    # Split so that no source line matches the leak gate (scripts/check_no_leaks.sh).
    key = "AIza" + "ContainerGuardFakeKey000000000000001"
    monkeypatch.setenv("GEMINI_API_KEY", key)
    _install(monkeypatch, FakeDocker([(125, "", f"docker: bad reference {key}\n")]))
    with pytest.raises(RuntimeError) as ei:
        cg.require_metadata_blocked("x", "img:1", "bridge")
    assert key not in str(ei.value)


def test_concurrent_callers_share_one_probe(monkeypatch):
    fake = _install(monkeypatch, FakeDocker([UNREACH]))
    results, errors = [], []

    def call():
        try:
            results.append(cg.require_metadata_blocked("x", "img:1", "bridge"))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)
    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and results == [cg.UNREACHABLE] * 8 and len(fake.runs) == 1


def test_reset_forgets_the_verdicts(monkeypatch):
    fake = _install(monkeypatch, FakeDocker([REACH, UNREACH]))
    with pytest.raises(RuntimeError):
        cg.require_metadata_blocked("x", "img:1", "bridge")
    cg.reset_verdicts()
    assert cg.require_metadata_blocked("x", "img:1", "bridge") == cg.UNREACHABLE
    assert len(fake.runs) == 2


def test_the_guard_imports_only_the_standard_library():
    """Suites import it from container_guard; base does not import it, so the modules the tau
    image loads from this directory stay as they are."""
    import ast
    tree = ast.parse(open(cg.__file__, encoding="utf-8").read())
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {(a.name if isinstance(n, ast.Import) else n.module or "").split(".")[0]
             for n in top for a in n.names}
    assert names <= {"__future__", "logging", "subprocess", "threading", "uuid", "typing"}, names
    assert all(getattr(n, "level", 0) == 0 for n in top)
    from gbench.runners.eval_suites import base
    assert "container_guard" not in open(base.__file__, encoding="utf-8").read()


# --------------------------------------------------------------------------- #
# the probe scripts, run on this host against 127.0.0.1 (never the metadata server)
# --------------------------------------------------------------------------- #
@pytest.fixture
def listening_port():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    yield srv.getsockname()[1]
    srv.close()


@pytest.fixture
def closed_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _run(argv):
    return subprocess.run(argv, capture_output=True, text=True, timeout=30).stdout


def test_python_probe_reports_an_open_and_a_refused_connection(listening_port, closed_port):
    python = sys.executable
    assert cg._verdict(_run([python, "-c", cg.python_probe_script("127.0.0.1", listening_port)])) \
        == cg.REACHABLE
    assert cg._verdict(_run([python, "-c", cg.python_probe_script("127.0.0.1", closed_port)])) \
        == cg.UNREACHABLE


def test_bash_probe_reports_an_open_and_a_refused_connection(listening_port, closed_port):
    assert cg._verdict(_run(["bash", "-c", cg.bash_probe_script("127.0.0.1", listening_port)])) \
        == cg.REACHABLE
    assert cg._verdict(_run(["bash", "-c", cg.bash_probe_script("127.0.0.1", closed_port)])) \
        == cg.UNREACHABLE


def test_bash_probe_without_dev_tcp_gives_no_verdict(tmp_path):
    """A bash built without /dev/tcp says 'No such file or directory': not a pass."""
    script = cg.bash_probe_script("127.0.0.1", 1).replace(
        "exec 3<>/dev/tcp/127.0.0.1/1", f"exec 3<{tmp_path}/missing")
    word = cg._verdict(_run(["bash", "-c", script]))
    assert word == "error"


# --------------------------------------------------------------------------- #
# the name-resolution check (resolve=)
# --------------------------------------------------------------------------- #
def _dns(*lines):
    return (0, "GBENCH_METADATA_PROBE=unreachable timeout ipv6:101\n"
            + "".join(f"GBENCH_DNS_PROBE={line}\n" for line in lines), "")


NAMES = ("generativelanguage.googleapis.com", "huggingface.co")


def test_the_same_probe_resolves_the_names_the_suite_needs(monkeypatch):
    fake = _install(monkeypatch, FakeDocker([_dns(f"ok {NAMES[0]}", f"ok {NAMES[1]}")]))
    assert cg.require_metadata_blocked("tau2", "img:1", "bridge", resolve=NAMES) \
        == cg.UNREACHABLE
    [argv] = fake.runs
    script = argv[-1]
    assert script == cg.python_probe_script(resolve=list(NAMES))
    assert "getaddrinfo" in script and all(repr(n)[1:-1] in script for n in NAMES)
    assert argv[argv.index("--pull") + 1] == "never"
    assert cg.cached_resolutions() == [("bridge", NAMES[0]), ("bridge", NAMES[1])]


def test_resolved_names_are_cached_per_network_and_name(monkeypatch):
    fake = _install(monkeypatch, FakeDocker([
        _dns(f"ok {NAMES[0]}"), _dns(f"ok {NAMES[1]}"), _dns(f"ok {NAMES[0]}")]))
    cg.require_metadata_blocked("x", "img:1", "bridge", resolve=[NAMES[0]])
    cg.require_metadata_blocked("x", "img:1", "bridge", resolve=[NAMES[0]])
    cg.require_metadata_blocked("x", "img:1", "bridge")
    assert len(fake.runs) == 1
    # a new name on the same network: one more probe, for that name only
    cg.require_metadata_blocked("x", "img:1", "bridge", resolve=list(NAMES))
    assert len(fake.runs) == 2
    assert NAMES[0] not in fake.runs[1][-1] and NAMES[1] in fake.runs[1][-1]
    # another network: probed again
    cg.require_metadata_blocked("x", "img:1", "run-net", resolve=[NAMES[0]])
    assert len(fake.runs) == 3
    assert cg.cached_verdicts() == {"bridge": cg.UNREACHABLE, "run-net": cg.UNREACHABLE}


def test_a_name_that_does_not_resolve_is_infra_required_naming_the_dns_rule(monkeypatch):
    fake = _install(monkeypatch, FakeDocker([
        _dns(f"fail {NAMES[0]} timeout", f"ok {NAMES[1]}"),
        _dns(f"ok {NAMES[0]}")]))
    with pytest.raises(RuntimeError) as ei:
        cg.require_metadata_blocked("aider_polyglot", "img:1", "bridge", resolve=NAMES)
    msg = str(ei.value)
    assert "[INFRA REQUIRED] aider_polyglot" in msg
    assert "containers on the docker network 'bridge' cannot resolve host names" in msg
    assert f"{NAMES[0]}: timeout" in msg and NAMES[1] + ":" not in msg
    assert "the VPC DNS resolver is 169.254.169.254" in msg
    assert "drops all traffic to that address also drops DNS" in msg
    assert cg.FIREWALL_COMMAND in msg and cg.PLAIN_RULE_DELETE_COMMAND in msg
    assert "gbench never runs sudo" in msg and "OUTPUT" not in msg
    # the metadata verdict and the name that resolved are cached; the failed name is not
    assert cg.cached_verdicts() == {"bridge": cg.UNREACHABLE}
    assert cg.cached_resolutions() == [("bridge", NAMES[1])]
    assert cg.require_metadata_blocked("aider_polyglot", "img:1", "bridge", resolve=NAMES) \
        == cg.UNREACHABLE
    assert len(fake.runs) == 2 and NAMES[1] not in fake.runs[1][-1]


@pytest.mark.parametrize("dns_lines, needle", [
    ((), "no answer"),
    ((f"error {NAMES[0]} no-resolver-command",), "no-resolver-command"),
    ((f"error {NAMES[0]} UnicodeError",), "UnicodeError"),
])
def test_no_clear_name_verdict_is_never_a_pass(monkeypatch, dns_lines, needle):
    _install(monkeypatch, FakeDocker([_dns(*dns_lines)]))
    with pytest.raises(RuntimeError) as ei:
        cg.require_metadata_blocked("x", "img:1", "bridge", resolve=[NAMES[0]])
    msg = str(ei.value)
    assert "gave no name-resolution verdict" in msg and needle in msg
    assert cg.cached_resolutions() == []


def test_a_reachable_metadata_server_wins_over_the_name_check(monkeypatch):
    fake = _install(monkeypatch, FakeDocker([REACH]))
    with pytest.raises(RuntimeError) as ei:
        cg.require_metadata_blocked("x", "img:1", "bridge", resolve=[NAMES[0]])
    assert "is reachable from a container" in str(ei.value)
    with pytest.raises(RuntimeError):      # cached: no second probe, even for new names
        cg.require_metadata_blocked("x", "img:1", "bridge", resolve=[NAMES[1]])
    assert len(fake.runs) == 1


def test_network_none_checks_no_names(monkeypatch):
    fake = _install(monkeypatch, FakeDocker())
    assert cg.require_metadata_blocked("x", "img:1", "none", resolve=NAMES) == cg.NO_NETWORK
    assert fake.calls == []


def test_without_python3_the_bash_probe_resolves_the_names(monkeypatch):
    fake = _install(monkeypatch, FakeDocker([NOT_FOUND, _dns(f"ok {NAMES[0]}")]))
    assert cg.require_metadata_blocked("x", "img:1", "bridge", resolve=[NAMES[0]]) \
        == cg.UNREACHABLE
    script = fake.runs[1][-1]
    assert script == cg.bash_probe_script(resolve=[NAMES[0]])
    assert f"look {NAMES[0]} &" in script and "getent hosts" in script and "nslookup" in script


@pytest.mark.parametrize("bad", ["", "-rf", "a b", "x;reboot", "$(id)", "a/b", "..", "a" * 254,
                                 ".lead", "a..b", "ex'ample.com"])
def test_names_are_checked_before_they_reach_a_script(monkeypatch, bad):
    fake = _install(monkeypatch, FakeDocker())
    with pytest.raises(ValueError):
        cg.require_metadata_blocked("x", "img:1", "bridge", resolve=[bad])
    with pytest.raises(ValueError):
        cg.python_probe_script(resolve=[bad])
    with pytest.raises(ValueError):
        cg.bash_probe_script(resolve=[bad])
    assert fake.calls == []


def test_a_single_name_string_and_duplicates_are_accepted(monkeypatch):
    fake = _install(monkeypatch, FakeDocker([_dns(f"ok {NAMES[0]}")]))
    cg.require_metadata_blocked("x", "img:1", "bridge", resolve=NAMES[0])
    assert fake.runs[0][-1] == cg.python_probe_script(resolve=[NAMES[0]])
    assert cg._check_names([NAMES[0], NAMES[0] + ".", NAMES[0]]) == [NAMES[0]]


# --------------------------------------------------------------------------- #
# the IPv6 metadata address
# --------------------------------------------------------------------------- #
def test_the_probe_also_tries_the_ipv6_metadata_address():
    for script in (cg.python_probe_script(), cg.bash_probe_script()):
        assert "fd20:ce::254" in script and "169.254.169.254" in script
    # a probe of another host (the local-socket tests) leaves IPv6 out unless asked
    assert "fd20:ce::254" not in cg.python_probe_script("127.0.0.1", 1)
    assert "fd20:ce::254" not in cg.bash_probe_script("127.0.0.1", 1)


def test_metadata_reachable_over_ipv6_is_infra_required_naming_the_ip6tables_rule(monkeypatch):
    fake = _install(monkeypatch, FakeDocker([(0, "GBENCH_METADATA_PROBE=reachable ipv6\n", "")]))
    for _ in range(2):
        with pytest.raises(RuntimeError) as ei:
            cg.require_metadata_blocked("x", "img:1", "v6net")
        msg = str(ei.value)
        assert "[fd20:ce::254]:80" in msg and cg.FIREWALL_COMMAND6 in msg
        assert "does not cover IPv6" in msg and "OUTPUT" not in msg
    assert len(fake.runs) == 1
    assert cg.FIREWALL_COMMAND6 == ("sudo ip6tables -I DOCKER-USER -d fd20:ce::254/128 -p tcp "
                                    "! --dport 53 -j DROP")


def test_no_ipv6_route_is_unreachable_not_an_error(monkeypatch):
    _install(monkeypatch, FakeDocker([(0, "GBENCH_METADATA_PROBE=unreachable timeout "
                                          "ipv6:101\n", "")]))
    assert cg.require_metadata_blocked("x", "img:1", "bridge") == cg.UNREACHABLE


def test_an_unexpected_ipv6_error_is_no_verdict(monkeypatch):
    _install(monkeypatch, FakeDocker([(0, "GBENCH_METADATA_PROBE=error 22 ipv6\n", "")]))
    with pytest.raises(RuntimeError, match="gave no verdict"):
        cg.require_metadata_blocked("x", "img:1", "bridge")


# --------------------------------------------------------------------------- #
# a wedged daemon: the removal after a timeout is bounded too
# --------------------------------------------------------------------------- #
class _WedgedDocker(FakeDocker):
    def __init__(self, rm_raises):
        super().__init__(raise_on_run=subprocess.TimeoutExpired(["docker", "run"], 120))
        self.rm_raises = rm_raises
        self.rm_kwargs = []

    def __call__(self, argv, **kwargs):
        if argv[:2] == ["docker", "rm"]:
            self.calls.append(list(argv))
            self.rm_kwargs.append(kwargs)
            raise self.rm_raises
        return super().__call__(argv, **kwargs)


@pytest.mark.parametrize("rm_raises", [
    subprocess.TimeoutExpired(["docker", "rm"], 30), FileNotFoundError("docker"),
    PermissionError("docker")])
def test_a_hung_removal_after_a_probe_timeout_still_raises_infra_required(monkeypatch,
                                                                          rm_raises):
    fake = _install(monkeypatch, _WedgedDocker(rm_raises))
    with pytest.raises(RuntimeError) as ei:
        cg.require_metadata_blocked("x", "img:1", "bridge")
    assert f"did not finish within {cg.PROBE_DOCKER_TIMEOUT_S} s" in str(ei.value)
    assert fake.rm_kwargs and fake.rm_kwargs[0].get("timeout") == cg.PROBE_RM_TIMEOUT_S
    assert cg.cached_verdicts() == {}


def test_the_removal_after_a_timeout_has_a_timeout(monkeypatch):
    seen = []

    class Recording(FakeDocker):
        def __call__(self, argv, **kwargs):
            if argv[:2] == ["docker", "rm"]:
                seen.append(kwargs)
            return super().__call__(argv, **kwargs)
    _install(monkeypatch, Recording(raise_on_run=subprocess.TimeoutExpired(["docker"], 1)))
    with pytest.raises(RuntimeError):
        cg.require_metadata_blocked("x", "img:1", "bridge")
    assert seen and 0 < seen[0]["timeout"] <= 60


def test_the_guard_never_pulls(monkeypatch):
    fake = _install(monkeypatch, FakeDocker([NOT_FOUND, (125, "", "No such image: img:1\n")]))
    with pytest.raises(RuntimeError):
        cg.require_metadata_blocked("x", "img:1", "bridge", resolve=[NAMES[0]])
    assert fake.calls and all(c[:2] in (["docker", "run"], ["docker", "rm"]) for c in fake.calls)
    assert all(c[c.index("--pull") + 1] == "never" for c in fake.runs)


# --------------------------------------------------------------------------- #
# the probe scripts' new parts, run on this host (loopback and stubs only)
# --------------------------------------------------------------------------- #
@pytest.fixture
def v6_listener():
    try:
        srv = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        srv.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        srv.bind(("::1", 0))
    except OSError:
        pytest.skip("no IPv6 loopback on this host")
    srv.listen(4)
    port = srv.getsockname()[1]
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", port))      # the same port stays closed on IPv4
    except OSError:
        srv.close()
        pytest.skip("the port is taken on IPv4")
    probe.close()
    yield port
    srv.close()


def test_python_probe_reports_ipv6_reachable_and_the_family(v6_listener):
    out = _run([sys.executable, "-c",
                cg.python_probe_script("127.0.0.1", v6_listener, host6="::1")])
    assert cg._verdict_line(out) == (cg.REACHABLE, "ipv6")


def test_bash_probe_reports_ipv6_reachable_and_the_family(v6_listener):
    out = _run(["bash", "-c", cg.bash_probe_script("127.0.0.1", v6_listener, host6="::1")])
    assert cg._verdict_line(out) == (cg.REACHABLE, "ipv6")


def test_both_probes_report_refused_on_both_families_as_unreachable(closed_port):
    for argv in ([sys.executable, "-c", cg.python_probe_script("127.0.0.1", closed_port,
                                                               host6="::1")],
                 ["bash", "-c", cg.bash_probe_script("127.0.0.1", closed_port, host6="::1")]):
        word, detail = cg._verdict_line(_run(argv))
        assert word == cg.UNREACHABLE and "ipv6:" in detail


def test_python_probe_treats_no_ipv6_support_as_unreachable(closed_port):
    """socket.AF_INET6 missing (a Python or kernel without IPv6) is EAFNOSUPPORT: unreachable."""
    script = ("import socket\ndel socket.AF_INET6\n"
              + cg.python_probe_script("127.0.0.1", closed_port, host6="::1"))
    word, detail = cg._verdict_line(_run([sys.executable, "-c", script]))
    assert word == cg.UNREACHABLE and detail.endswith("ipv6:97")


def test_python_probe_resolves_names_and_reports_failures_and_timeouts(closed_port):
    ok = cg.python_probe_script("127.0.0.1", closed_port, resolve=["localhost"])
    assert cg._dns_results(_run([sys.executable, "-c", ok])) == {"localhost": ("ok", "")}
    failing = ("import socket\n"
               "def _gai(*a, **k):\n"
               "    raise socket.gaierror(-3, 'Temporary failure in name resolution')\n"
               "socket.getaddrinfo = _gai\n"
               + cg.python_probe_script("127.0.0.1", closed_port, resolve=["example.org"]))
    assert cg._dns_results(_run([sys.executable, "-c", failing])) == {
        "example.org": ("fail", "gaierror:-3")}
    slow = ("import socket, time\n"
            "socket.getaddrinfo = lambda *a, **k: time.sleep(30)\n"
            + cg.python_probe_script("127.0.0.1", closed_port, resolve=["example.org"],
                                     resolve_timeout_s=1))
    started = time.time()
    assert cg._dns_results(_run([sys.executable, "-c", slow])) == {
        "example.org": ("fail", "timeout")}
    assert time.time() - started < 20


def _fake_bin(tmp_path, name, body):
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    f = d / name
    f.write_text("#!/bin/bash\n" + body + "\n")
    f.chmod(0o755)
    return d


def test_bash_probe_resolves_with_getent(tmp_path, closed_port):
    d = _fake_bin(tmp_path, "getent",
                  'case "$2" in ok.example) exit 0 ;; slow.example) sleep 30 ;; *) exit 2 ;; '
                  'esac')
    script = cg.bash_probe_script("127.0.0.1", closed_port,
                                  resolve=["ok.example", "gone.example", "slow.example"],
                                  resolve_timeout_s=1)
    env = {"PATH": f"{d}:/usr/bin:/bin"}
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30,
                         env=env).stdout
    assert cg._dns_results(out) == {"ok.example": ("ok", ""),
                                    "gone.example": ("fail", "getent:not-found"),
                                    "slow.example": ("fail", "timeout")}


def test_bash_probe_without_getent_or_nslookup_gives_no_name_verdict(tmp_path, closed_port):
    d = tmp_path / "bin"
    d.mkdir()
    for prog in ("bash", "timeout"):
        (d / prog).symlink_to(subprocess.run(["bash", "-c", f"command -v {prog}"],
                                             capture_output=True, text=True).stdout.strip())
    script = cg.bash_probe_script("127.0.0.1", closed_port, resolve=["example.org"])
    out = subprocess.run([str(d / "bash"), "-c", script], capture_output=True, text=True,
                         timeout=30, env={"PATH": str(d)}).stdout
    assert cg._verdict(out) == cg.UNREACHABLE
    assert cg._dns_results(out) == {"example.org": ("error", "no-resolver-command")}
