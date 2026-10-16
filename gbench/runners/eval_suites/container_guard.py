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

"""Cloud metadata server guard for containers that run model-controlled code with network access.

On a cloud VM (a GCE VM, for example) the metadata server at 169.254.169.254 hands the VM's
service-account token to any process that can open a TCP connection to it on port 80. A container
on a docker bridge network or a user-defined docker network reaches it like any other address,
and a container on the host network shares the VM's own network stack. When the model under test
runs shell commands or code in such a container, it could read that token.

gbench never runs sudo and never changes the host's firewall. The operator blocks the metadata
server for docker networks once, on the host:

    sudo iptables -I DOCKER-USER -d 169.254.169.254/32 -p tcp ! --dport 53 -j DROP

and makes the rule persistent (for example with iptables-persistent:
`sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save`). On GCE the VPC
DNS resolver answers at the same address, so the rule drops TCP to every port but 53: the
metadata HTTP endpoints are blocked, DNS (TCP and UDP 53) keeps working. A plain `-j DROP` for
the address would also cut name resolution in every container. Docker sends the traffic of every
bridge and user-defined network through its DOCKER-USER chain, so the rule covers them all. It
does not cover the host network: DOCKER-USER never sees that traffic, and dropping the address for
the whole VM would also cut the VM's own DNS, its guest agent (SSH keys) and its credentials, so a
container on the host network that runs model-controlled code has to move to a bridge or
user-defined network. On a dual-stack VM whose docker network has IPv6, the metadata server also
answers at [fd20:ce::254]; the ip6tables rule for it is FIREWALL_COMMAND6.

require_metadata_blocked(eval_name, image, network, docs_url=None, *, resolve=()) is the gate. A
suite calls it before it starts a container that runs model-controlled code on `network` (bridge,
host, a user-defined network or `container:<name>`). It starts one short-lived probe container of
`image` (an image the suite already has locally: `--pull never`, so nothing is pulled) on that
network, which only tests whether a TCP connection to 169.254.169.254:80 opens, and closes it at
once: no request is sent, so no token or other metadata is read. It tests [fd20:ce::254]:80 the
same way; a container without an IPv6 route gets "unreachable" there, not an error. When
`resolve` names host names the suite's containers need, the same probe container also resolves
each of them (python3's socket.getaddrinfo; with bash, `getent hosts`, else `nslookup`), all at
once, each within PROBE_RESOLVE_TIMEOUT_S. The probe program is the image's python3; without
python3, the image's bash with its /dev/tcp redirection under coreutils `timeout`.

When the connection opens, the gate raises infra_required naming the host firewall command. When
a name does not resolve, it raises infra_required saying that containers on that network cannot
resolve host names, that on GCE the VPC DNS resolver is 169.254.169.254 so a rule dropping all
traffic to it also drops DNS, and naming FIREWALL_COMMAND. When the probe cannot run or its answer
is not clear (no python3 or bash in the image, no resolver program for a name check, docker
errors, an unexpected socket error, a timeout of docker itself), it raises infra_required too: the
gate never lets a container start on a verdict it could not establish. After a docker timeout the
probe container is removed with a bounded `docker rm -f`. `--network none` needs no probe. The
metadata verdict is cached per network mode and a resolved name per (network, name) for the rest
of the process, so a run probes each network once and each name once; a name that did not resolve
is not cached, so the next call probes it again.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import uuid
from typing import Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

METADATA_HOST = "169.254.169.254"
#: The metadata server's IPv6 address on GCE (dual-stack VMs).
METADATA_HOST6 = "fd20:ce::254"
METADATA_PORT = 80
#: The host firewall rule that blocks the metadata server for every docker bridge and
#: user-defined network. The operator adds it once; gbench never runs it. It spares port 53
#: because on GCE the VPC DNS resolver is the same address.
FIREWALL_COMMAND = f"sudo iptables -I DOCKER-USER -d {METADATA_HOST}/32 -p tcp ! --dport 53 -j DROP"
#: Its IPv6 counterpart, for a docker network with IPv6 on a dual-stack VM.
FIREWALL_COMMAND6 = (f"sudo ip6tables -I DOCKER-USER -d {METADATA_HOST6}/128 -p tcp ! --dport 53 "
                     "-j DROP")
#: One way to keep FIREWALL_COMMAND across reboots (Debian and Ubuntu).
PERSIST_COMMAND = "sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save"
#: Removes the earlier plain rule, which also dropped DNS to the same address.
PLAIN_RULE_DELETE_COMMAND = f"sudo iptables -D DOCKER-USER -d {METADATA_HOST}/32 -j DROP"

#: The connect timeout inside the probe container, in seconds.
PROBE_CONNECT_TIMEOUT_S = 3
#: The time the probe container gives all its name lookups together, in seconds.
PROBE_RESOLVE_TIMEOUT_S = 15
#: The limit for one `docker run` of the probe (container start plus the checks), in seconds.
PROBE_DOCKER_TIMEOUT_S = 120
#: The limit for the `docker rm -f` of a probe container that timed out, in seconds.
PROBE_RM_TIMEOUT_S = 30
#: The label on every probe container.
PROBE_LABEL = "org.gbench.metadata-probe=1"
_MARK = "GBENCH_METADATA_PROBE="
_DNS_MARK = "GBENCH_DNS_PROBE="

REACHABLE = "reachable"
UNREACHABLE = "unreachable"
NO_NETWORK = "no-network"

#: The characters of a host name the probe resolves (it is written into the probe script).
_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
_DEFAULT = object()

_VERDICTS: Dict[str, str] = {}
#: For a REACHABLE verdict, the address family that opened ("ipv4" or "ipv6").
_REACHED_BY: Dict[str, str] = {}
_RESOLVED: Set[Tuple[str, str]] = set()
_LOCK = threading.Lock()


def _host6_for(host: str, host6) -> Optional[str]:
    """The IPv6 address to probe: METADATA_HOST6 by default when probing the metadata server,
    none for another host (the scripts run against a local socket in tests)."""
    if host6 is _DEFAULT:
        return METADATA_HOST6 if host == METADATA_HOST else None
    return host6


def _check_names(resolve: Iterable[str]) -> List[str]:
    """The names in `resolve`, checked (letters, digits, '.', '-' and '_'; no leading '-' or
    '.'; at most 253 characters; a trailing '.' dropped) and de-duplicated in order."""
    if isinstance(resolve, str):
        resolve = [resolve]
    names: List[str] = []
    for raw in resolve or ():
        name = str(raw or "").strip()
        if name.endswith("."):
            name = name[:-1]
        if (not name or len(name) > 253 or name[0] in "-." or ".." in name
                or not set(name) <= _NAME_CHARS):
            raise ValueError(f"not a host name the metadata-server probe can resolve: {raw!r}")
        if name not in names:
            names.append(name)
    return names


def python_probe_script(host: str = METADATA_HOST, port: int = METADATA_PORT,
                        timeout_s: int = PROBE_CONNECT_TIMEOUT_S, *, host6=_DEFAULT,
                        resolve: Iterable[str] = (),
                        resolve_timeout_s: int = PROBE_RESOLVE_TIMEOUT_S) -> str:
    """The python3 probe: open a TCP connection to host:port (and [host6]:port), close it, print
    the verdict; then resolve each name in `resolve`.

    Only a connection that opens is `reachable`. A timeout, a refusal or an unreachable or
    forbidden route is `unreachable` (for IPv6 also no IPv6 route, address or support); any other
    error is `error <errno>`, which the gate treats as no verdict. Nothing is sent on the
    connection. Each name prints `ok <name>`, `fail <name> <why>` (no answer within
    `resolve_timeout_s`, or the resolver's error) or `error <name> <why>`."""
    h6 = _host6_for(host, host6)
    names = _check_names(resolve)
    t = int(timeout_s)
    lines = [
        "import errno, socket, sys\n",
        "NO_ROUTE = (errno.ECONNREFUSED, errno.EHOSTUNREACH, errno.ENETUNREACH,\n",
        "            errno.ETIMEDOUT, errno.EPERM, errno.EACCES)\n",
        "NO_V6 = (errno.EADDRNOTAVAIL, errno.EAFNOSUPPORT, errno.ENETUNREACH)\n",
        "def probe(family, addr, extra):\n",
        "    if family is None:\n",
        f"        return '{UNREACHABLE} ' + str(errno.EAFNOSUPPORT)\n",
        "    try:\n",
        "        s = socket.socket(family, socket.SOCK_STREAM)\n",
        "    except (OSError, IOError, socket.error) as e:\n",
        f"        return ('{UNREACHABLE} ' if e.errno in extra else 'error ') + str(e.errno)\n",
        f"    s.settimeout({t})\n",
        "    try:\n",
        "        s.connect(addr)\n",
        "    except socket.timeout:\n",
        f"        return '{UNREACHABLE} timeout'\n",
        "    except (OSError, IOError, socket.error) as e:\n",
        f"        return ('{UNREACHABLE} ' if e.errno in NO_ROUTE + extra else 'error ') + "
        "str(e.errno)\n",
        "    else:\n",
        f"        return '{REACHABLE}'\n",
        "    finally:\n",
        "        s.close()\n",
        f"v4 = probe(socket.AF_INET, ({host!r}, {int(port)}), ())\n",
    ]
    if h6:
        lines += [
            f"v6 = probe(getattr(socket, 'AF_INET6', None), ({h6!r}, {int(port)}, 0, 0), "
            "NO_V6)\n",
            f"if v4 == '{REACHABLE}':\n",
            "    verdict = v4 + ' ipv4'\n",
            f"elif v6 == '{REACHABLE}':\n",
            "    verdict = v6 + ' ipv6'\n",
            "elif v4.startswith('error'):\n",
            "    verdict = v4\n",
            "elif v6.startswith('error'):\n",
            "    verdict = v6 + ' ipv6'\n",
            "else:\n",
            "    verdict = v4 + ' ipv6:' + v6.split(' ', 1)[-1]\n",
        ]
    else:
        lines.append("verdict = v4\n")
    lines.append(f"print('{_MARK}' + verdict)\n")
    if names:
        lines += [
            "import threading, time\n",
            f"names = {names!r}\n",
            "found = {}\n",
            "def look(name):\n",
            "    try:\n",
            "        socket.getaddrinfo(name, 443, 0, socket.SOCK_STREAM)\n",
            "        found[name] = 'ok'\n",
            "    except socket.gaierror as e:\n",
            "        found[name] = 'fail gaierror:' + str(e.errno)\n",
            "    except Exception as e:\n",
            "        found[name] = 'error ' + type(e).__name__\n",
            "workers = [threading.Thread(target=look, args=(n,)) for n in names]\n",
            "for w in workers:\n",
            "    w.daemon = True\n",
            "    w.start()\n",
            f"end = time.time() + {int(resolve_timeout_s)}\n",
            "for w in workers:\n",
            "    w.join(max(0.0, end - time.time()))\n",
            "for n in names:\n",
            "    word, _, why = found.get(n, 'fail timeout').partition(' ')\n",
            f"    print('{_DNS_MARK}' + word + ' ' + n + (' ' + why if why else ''))\n",
        ]
    lines.append("sys.stdout.flush()\n")
    return "".join(lines)


def _bash_connect(var: str, host: str, port: int, t: int, extra_patterns: str = "") -> str:
    return (
        f"out=$(timeout {t} bash -c 'exec 3<>/dev/tcp/{host}/{int(port)}' 2>&1); rc=$?\n"
        'case "$rc:$out" in\n'
        f"  0:*) {var}='{REACHABLE}' ;;\n"
        f"  124:*) {var}='{UNREACHABLE} timeout' ;;\n"
        '  *"Connection refused"*|*"No route to host"*|*"Network is unreachable"*|'
        '*"Connection timed out"*|*"Operation not permitted"*|*"Permission denied"*'
        f"{extra_patterns})\n"
        f"    {var}='{UNREACHABLE}' ;;\n"
        f"  *) {var}=\"error rc=$rc\" ;;\n"
        "esac\n")


def bash_probe_script(host: str = METADATA_HOST, port: int = METADATA_PORT,
                      timeout_s: int = PROBE_CONNECT_TIMEOUT_S, *, host6=_DEFAULT,
                      resolve: Iterable[str] = (),
                      resolve_timeout_s: int = PROBE_RESOLVE_TIMEOUT_S) -> str:
    """The bash probe for an image without python3: bash's /dev/tcp redirection opens the
    connection under coreutils `timeout` and closes it on exit. Exit 0 is `reachable`; the
    timeout (124) or a refused, unreachable or forbidden connection is `unreachable` (for IPv6
    also no IPv6 address or support); anything else (no `timeout`, a bash built without /dev/tcp)
    is `error`, which the gate treats as no verdict. Nothing is sent on the connection. Each name
    in `resolve` is looked up with `getent hosts`, else `nslookup`, all at once under `timeout`:
    `ok <name>`, `fail <name> <why>`, or `error <name> <why>` (no such program, or getent's own
    error)."""
    h6 = _host6_for(host, host6)
    names = _check_names(resolve)
    t = int(timeout_s)
    script = (
        "command -v timeout >/dev/null 2>&1 || "
        f"{{ echo '{_MARK}error no-timeout-command'; exit 0; }}\n"
        + _bash_connect("v4", host, port, t))
    if h6:
        script += _bash_connect("v6", h6, port, t,
                                '|*"Cannot assign requested address"*'
                                '|*"Address family not supported"*')
        script += (
            'case "$v4:$v6" in\n'
            f"  {REACHABLE}:*) verdict='{REACHABLE} ipv4' ;;\n"
            f"  *:{REACHABLE}) verdict='{REACHABLE} ipv6' ;;\n"
            '  error*) verdict="$v4" ;;\n'
            '  *:error*) verdict="$v6 ipv6" ;;\n'
            '  *) verdict="$v4 ipv6:${v6#* }" ;;\n'
            "esac\n")
    else:
        script += 'verdict="$v4"\n'
    script += f'echo "{_MARK}$verdict"\n'
    if names:
        r = int(resolve_timeout_s)
        script += (
            "look() {\n"
            "  if command -v getent >/dev/null 2>&1; then\n"
            f"    timeout {r} getent hosts \"$1\" >/dev/null 2>&1; rc=$?\n"
            '    case "$rc" in\n'
            f"      0) echo \"{_DNS_MARK}ok $1\" ;;\n"
            f"      2) echo \"{_DNS_MARK}fail $1 getent:not-found\" ;;\n"
            f"      124) echo \"{_DNS_MARK}fail $1 timeout\" ;;\n"
            f"      *) echo \"{_DNS_MARK}error $1 getent:rc=$rc\" ;;\n"
            "    esac\n"
            "  elif command -v nslookup >/dev/null 2>&1; then\n"
            f"    out=$(timeout {r} nslookup \"$1\" 2>&1); rc=$?\n"
            '    case "$rc:$out" in\n'
            f"      0:*) echo \"{_DNS_MARK}ok $1\" ;;\n"
            f"      124:*) echo \"{_DNS_MARK}fail $1 timeout\" ;;\n"
            f"      *NXDOMAIN*) echo \"{_DNS_MARK}fail $1 nslookup:NXDOMAIN\" ;;\n"
            f"      *) echo \"{_DNS_MARK}fail $1 nslookup:rc=$rc\" ;;\n"
            "    esac\n"
            "  else\n"
            f"    echo \"{_DNS_MARK}error $1 no-resolver-command\"\n"
            "  fi\n"
            "}\n"
            + "".join(f"look {n} &\n" for n in names)
            + "wait\n")
    return script


def _normalize(network: Optional[str]) -> str:
    net = (network or "").strip()
    return "bridge" if net in ("", "default") else net


def _probe_argv(image: str, network: str, name: str, entrypoint: str, script: str) -> List[str]:
    return ["docker", "run", "--rm", "--pull", "never", "--name", name, "--label", PROBE_LABEL,
            "--network", network, "--read-only", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--entrypoint", entrypoint, image,
            "-c", script]


def _verdict_line(stdout: str) -> Optional[Tuple[str, str]]:
    """(verdict word, the rest of the line) of the probe's output, or None."""
    for line in (stdout or "").splitlines():
        line = line.strip()
        if line.startswith(_MARK):
            word, _, rest = line[len(_MARK):].partition(" ")
            return word, rest.strip()
    return None


def _verdict(stdout: str) -> Optional[str]:
    """The probe's verdict word from its output (`reachable`, `unreachable`, `error`), or None."""
    found = _verdict_line(stdout)
    return found[0] if found else None


def _dns_results(stdout: str) -> Dict[str, Tuple[str, str]]:
    """{name: (word, why)} from the probe's name-lookup lines (word: ok, fail or error)."""
    out: Dict[str, Tuple[str, str]] = {}
    for line in (stdout or "").splitlines():
        line = line.strip()
        if line.startswith(_DNS_MARK):
            parts = line[len(_DNS_MARK):].split(None, 2)
            if len(parts) >= 2:
                out[parts[1]] = (parts[0], parts[2] if len(parts) > 2 else "")
    return out


def _missing_program(proc: subprocess.CompletedProcess) -> bool:
    """True when docker could not start the probe's entrypoint (exit 127: not found in the image,
    126: not executable). The probe scripts themselves never exit with either code."""
    return proc.returncode in (126, 127) and _verdict(proc.stdout) is None


def _clip(text: str, limit: int = 300) -> str:
    text = " ".join(str(text or "").split())
    try:
        from .base import redact_text
        text = redact_text(text)
    except Exception:  # pragma: no cover - redaction is best effort in an error message
        pass
    return text if len(text) <= limit else text[:limit] + "..."


def _remove_probe(name: str) -> None:
    """`docker rm -f` of a probe container that timed out, itself bounded: a wedged daemon
    must not keep the gate from raising."""
    try:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True,
                       timeout=PROBE_RM_TIMEOUT_S)
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("could not remove the metadata-server probe container %s (%s)", name,
                       type(exc).__name__)


def _run_probe(eval_name: str, image: str, network: str, docs_url: str,
               names: List[str] = ()) -> Tuple[str, str, Dict[str, Tuple[str, str]]]:
    """(REACHABLE or UNREACHABLE, the verdict's detail, {name: (word, why)}) for `network`, from
    a probe container of `image`; raises infra_required when no verdict can be established,
    for the metadata server or for a name in `names`."""
    from .swebench_common import infra_required
    names = list(names or ())
    attempts = (("python3", python_probe_script(resolve=names)),
                ("bash", bash_probe_script(resolve=names)))
    missing: List[str] = []
    for entrypoint, script in attempts:
        name = f"gbench-metadata-probe-{uuid.uuid4().hex[:12]}"
        argv = _probe_argv(image, network, name, entrypoint, script)
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=PROBE_DOCKER_TIMEOUT_S)
        except FileNotFoundError as exc:
            raise infra_required(
                eval_name, "the docker CLI is not on PATH, so gbench cannot check that the cloud "
                "metadata server is blocked for the suite's containers.", docs_url) from exc
        except subprocess.TimeoutExpired as exc:
            _remove_probe(name)
            raise infra_required(
                eval_name, f"the metadata-server probe container ({image} on the docker network "
                f"{network!r}) did not finish within {PROBE_DOCKER_TIMEOUT_S} s, so gbench could "
                "not check that the cloud metadata server is blocked for the suite's containers.",
                docs_url) from exc
        found = _verdict_line(proc.stdout)
        word, detail = found if found else (None, "")
        if word in (REACHABLE, UNREACHABLE):
            dns = _dns_results(proc.stdout) if word == UNREACHABLE else {}
            unclear = [n for n in names if word == UNREACHABLE
                       and dns.get(n, ("", ""))[0] not in ("ok", "fail")]
            if unclear:
                shown = ", ".join(f"{n}: {' '.join(dns[n]).strip() if n in dns else 'no answer'}"
                                  for n in unclear)
                raise infra_required(
                    eval_name, f"the probe ({entrypoint} in {image} on the docker network "
                    f"{network!r}) gave no name-resolution verdict ({_clip(shown)}), so gbench "
                    "could not check that the suite's containers can resolve the host names they "
                    "need.", docs_url)
            return word, detail, dns
        if word is None and _missing_program(proc):
            missing.append(entrypoint)
            continue
        shown = _clip(proc.stdout if word is not None else (proc.stderr or proc.stdout))
        shown = shown or "no output"
        raise infra_required(
            eval_name, f"the metadata-server probe ({entrypoint} in {image} on the docker network "
            f"{network!r}) gave no verdict (docker exit {proc.returncode}: {shown}), "
            "so gbench could not check that the cloud metadata server (169.254.169.254) is "
            "blocked for the suite's containers.", docs_url)
    raise infra_required(
        eval_name, f"the metadata-server probe needs python3, or bash with coreutils timeout, in "
        f"the image {image}, which has neither ({', '.join(missing)} not found), so gbench could "
        "not check that the cloud metadata server (169.254.169.254) is blocked for the suite's "
        "containers.", docs_url)


def reachable_message(network: str, family: str = "ipv4") -> str:
    """The infra_required reason when the metadata server answers on `network` (over IPv6 when
    `family` is "ipv6")."""
    if family == "ipv6":
        return (f"the cloud metadata server ([{METADATA_HOST6}]:{METADATA_PORT}, its IPv6 address, "
                f"which hands out this VM's service-account token) is reachable from a container "
                f"on the docker network {network!r}, where the model under test runs code. Block "
                f"it once on the host with `{FIREWALL_COMMAND6}` (the IPv4 rule "
                f"`{FIREWALL_COMMAND}` does not cover IPv6) and make both rules persistent, for "
                f"example with `{PERSIST_COMMAND}`, or run these containers on a docker network "
                "without IPv6. gbench never runs sudo or changes the host's firewall itself.")
    head = (f"the cloud metadata server ({METADATA_HOST}:{METADATA_PORT}, which hands out this "
            f"VM's service-account token) is reachable from a container on the docker network "
            f"{network!r}, where the model under test runs code.")
    if network == "host":
        return (head + " Docker's DOCKER-USER chain does not filter the host network, so the rule "
                f"`{FIREWALL_COMMAND}` does not cover it, and blocking the address for the whole "
                "VM would also cut the VM's own DNS, guest agent and credentials. Run these "
                "containers on a bridge or user-defined docker network (see the suite's docs) "
                f"with that rule added once on the host and made persistent, for example with "
                f"`{PERSIST_COMMAND}`. gbench never runs sudo or changes the host's firewall "
                "itself.")
    return (head + f" Block it once on the host with `{FIREWALL_COMMAND}` and make the rule "
            f"persistent, for example with `{PERSIST_COMMAND}`. gbench never runs sudo or "
            "changes the host's firewall itself.")


#: Lookup failures in which the resolver ANSWERED that the name does not exist (getaddrinfo's
#: EAI_NONAME -2 and EAI_NODATA -5, nslookup's NXDOMAIN): a wrong host name, not a DNS outage.
NONEXISTENT_NAME_WHYS = frozenset({"gaierror:-2", "gaierror:-5", "nslookup:NXDOMAIN"})


def nonexistent_message(network: str, names: Dict[str, str]) -> str:
    """The infra_required reason when the resolver on `network` answered that the names in
    `names` ({name: why}) do not exist: a mistyped or wrong host name, not a firewall rule."""
    shown = ", ".join(f"{n}: {why}" for n, why in names.items())
    return (f"the host name(s) the suite's containers need do not exist: the resolver on the "
            f"docker network {network!r} answered that there is no such name ({shown}). Check "
            "the URL or host name (the model endpoint's URL, --remote-endpoint, a proxy "
            "variable, or the suite's settings); DNS itself answered, so no firewall rule is "
            "involved.")


def unresolved_message(network: str, failed: Dict[str, str]) -> str:
    """The infra_required reason when containers on `network` cannot resolve the names in
    `failed` ({name: why}). Names the resolver answered do not exist (NONEXISTENT_NAME_WHYS)
    get nonexistent_message instead; the firewall advice is for the others (a timeout,
    EAI_AGAIN, no answer)."""
    missing = {n: w for n, w in failed.items() if w in NONEXISTENT_NAME_WHYS}
    outage = {n: w for n, w in failed.items() if w not in NONEXISTENT_NAME_WHYS}
    if missing and not outage:
        return nonexistent_message(network, missing)
    if missing:
        return unresolved_message(network, outage) + " Also: " + nonexistent_message(network,
                                                                                     missing)
    shown = ", ".join(f"{n}: {why or 'no answer'}" for n, why in failed.items())
    return (f"containers on the docker network {network!r} cannot resolve host names ({shown}), "
            "which the suite's containers need, so its tasks would fail on the harness's side. "
            f"On GCE the VPC DNS resolver is {METADATA_HOST}, the metadata server's address, so a "
            "host firewall rule that drops all traffic to that address also drops DNS. Block the "
            f"metadata server with `{FIREWALL_COMMAND}` instead (it drops TCP to every port but "
            f"53); a plain rule for the address is removed with `{PLAIN_RULE_DELETE_COMMAND}`. "
            f"Make the rule persistent, for example with `{PERSIST_COMMAND}`. If no such rule is "
            "in place, check the docker daemon's DNS settings and this host's network. gbench "
            "never runs sudo or changes the host's firewall itself.")


def require_metadata_blocked(eval_name: str, image: str, network: Optional[str],
                             docs_url: Optional[str] = None, *,
                             resolve: Iterable[str] = ()) -> str:
    """Raise infra_required unless the cloud metadata server is unreachable from a container on
    `network` and, with `resolve`, each of those host names resolves there. Returns NO_NETWORK
    for `--network none` (no probe; `resolve` is not checked) and UNREACHABLE otherwise.

    Call it before starting a container that runs model-controlled code with network access.
    `image` must be an image the suite already has locally (the probe never pulls); `network` is
    the value the suite passes to `docker run --network` (None or "" is docker's default bridge);
    `resolve` names the hosts the suite's containers need (a model endpoint given by name, the
    Gemini API, a package index, ...). The metadata verdict is cached per network mode, and each
    resolved name per (network, name), for the rest of the process."""
    docs_url = docs_url or f"docs/evals/{eval_name}.md"
    net = _normalize(network)
    names = _check_names(resolve)
    if net == "none":
        return NO_NETWORK
    failed: Dict[str, str] = {}
    with _LOCK:
        verdict = _VERDICTS.get(net)
        pending = [n for n in names if (net, n) not in _RESOLVED]
        if verdict is None or (verdict == UNREACHABLE and pending):
            verdict, detail, dns = _run_probe(eval_name, image, net, docs_url, pending)
            _VERDICTS[net] = verdict
            if verdict == REACHABLE:
                _REACHED_BY[net] = "ipv6" if detail.startswith("ipv6") else "ipv4"
            for n in pending:
                word, why = dns.get(n, ("", ""))
                if word == "ok":
                    _RESOLVED.add((net, n))
                elif verdict == UNREACHABLE:
                    failed[n] = why
            logger.info("[%s] metadata-server probe on the docker network %r: %s%s", eval_name,
                        net, verdict, f" ({len(pending)} name(s) checked)" if pending else "")
        family = _REACHED_BY.get(net, "ipv4")
    if verdict == REACHABLE:
        from .swebench_common import infra_required
        raise infra_required(eval_name, reachable_message(net, family), docs_url)
    if failed:
        from .swebench_common import infra_required
        raise infra_required(eval_name, unresolved_message(net, failed), docs_url)
    return UNREACHABLE


def cached_verdicts() -> Dict[str, str]:
    """A copy of the metadata verdicts cached so far ({network mode: verdict})."""
    with _LOCK:
        return dict(_VERDICTS)


def cached_resolutions() -> List[Tuple[str, str]]:
    """The (network mode, host name) pairs found to resolve so far."""
    with _LOCK:
        return sorted(_RESOLVED)


def reset_verdicts() -> None:
    """Forget the cached verdicts and resolved names (tests, or after the operator changed the
    host firewall)."""
    with _LOCK:
        _VERDICTS.clear()
        _REACHED_BY.clear()
        _RESOLVED.clear()
