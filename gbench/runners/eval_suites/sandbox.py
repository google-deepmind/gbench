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

"""Process isolation for eval suites that execute model-written code.

`--sandboxes` only bounds how many of these run at once; it does not isolate anything.
Until now every code-executing suite ran `sys.executable -c <model output>` directly on the
host, with the user's full filesystem and network.

This wraps those calls in `bubblewrap`: read-only root, private `/tmp` and `/run` (so no
Docker socket), no network, new PID, IPC and UTS namespaces with a `/proc` that shows only the
sandbox's own processes, no capabilities, and an explicit writable bind for the scratch
directory the suite needs. The home directory ($HOME, and the account's home from the password
database when it differs) is replaced by an empty directory, into which only what a program
legitimately reads is bound back read-only: the interpreter's own directories and site-packages
when they live there, the Hugging Face hub cache (dataset files, e.g. SciCode's test_data.h5),
the data files an operator configures for sandboxed programs (SANDBOX_READABLE_ENV) and the
caller's `readable` paths. Credential files are then hidden wherever they are
(credential_paths: the huggingface_hub token files, $GOOGLE_APPLICATION_CREDENTIALS, the gcloud,
Docker, AWS, Azure, kube and gh configurations, ~/.netrc, ~/.ssh, ...), symlinked ones by their
target. run_sandboxed also passes the program an environment without the credential variables
(and without any variable whose value carries a credential), in every mode, `none` included;
with `none` nothing else is isolated, so the files are readable there.

    GBENCH_SANDBOX=required (default) require bwrap. If it is unavailable or blocked,
                                     code-executing suites SKIP (via sandbox_skip_reason)
                                     rather than run unsandboxed.
    GBENCH_SANDBOX=bwrap             require bwrap and FAIL LOUDLY (raise) if it is
                                     unavailable or blocked - a strict gate for CI.
    GBENCH_SANDBOX=none              run directly on the host (explicit unsandboxed opt-out)

Isolation is mandatory by default: executing model-written code without a sandbox is a
security risk, so there is no "degrade and run unsandboxed" policy - the only way to run
without isolation is to ask for it explicitly with `none`. Every mode self-probes once (a
bwrap that is installed but blocked - no user namespaces, a restrictive container - is
treated as unavailable) rather than assuming, so a broken sandbox surfaces as a clean skip
(or, under `bwrap`, a loud failure), not as a model that "cannot write code".
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = ["sandbox_mode", "sandbox_available", "sandbox_skip_reason",
           "wrap_argv", "run_sandboxed", "credential_free_env", "credential_paths",
           "masked_homes"]

_PROBE_TIMEOUT = 10


def sandbox_mode() -> str:
    mode = (os.environ.get("GBENCH_SANDBOX") or "required").strip().lower()
    return mode if mode in ("required", "bwrap", "none") else "required"


@functools.lru_cache(maxsize=1)
def _probe_bwrap() -> bool:
    """Does bwrap actually run here? Cached: this is asked once per code-executing sample."""
    if not shutil.which("bwrap"):
        return False
    try:
        proc = subprocess.run(
            wrap_argv([sys.executable, "-c", "print(1)"], force=True),
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT)
        return proc.returncode == 0 and proc.stdout.strip() == "1"
    except Exception as e:  # pragma: no cover - environment dependent
        logger.debug("bwrap probe failed: %s", e)
        return False


@functools.lru_cache(maxsize=1)
def sandbox_available() -> bool:
    """True when execution will actually be isolated (bubblewrap works here).

    Under the default `required` mode a False return means code-executing suites must
    SKIP (see `sandbox_skip_reason`) rather than run unsandboxed - so this stays silent
    there and lets the skip carry the message.
    """
    mode = sandbox_mode()
    if mode == "none":
        logger.warning("GBENCH_SANDBOX=none: model-written code runs directly on this host "
                       "with full filesystem and network access.")
        return False
    if _probe_bwrap():
        return True
    if mode == "bwrap":
        raise RuntimeError(
            "GBENCH_SANDBOX=bwrap was requested but bubblewrap is unavailable or blocked "
            "here. Install `bubblewrap` and ensure unprivileged user namespaces are "
            "permitted, set GBENCH_SANDBOX=none to run unsandboxed, or GBENCH_SANDBOX="
            "required to skip code-executing suites instead of failing.")
    # required: not isolated -> callers skip via sandbox_skip_reason(); never degrade.
    return False


def sandbox_skip_reason() -> Optional[str]:
    """Reason a code-executing suite must SKIP, or None if it may proceed.

    Default (`required`): model-written code must not run without isolation, so if
    bubblewrap is unavailable or blocked, return a message to skip with. `none` (explicit
    unsandboxed opt-out) and `bwrap` (fails loudly at the exec call instead of skipping)
    return None, so the suite proceeds.
    """
    mode = sandbox_mode()
    if mode in ("none", "bwrap"):
        return None
    if _probe_bwrap():
        return None
    if shutil.which("bwrap"):
        # Installed but the probe failed - almost always blocked unprivileged user
        # namespaces (Ubuntu 23.10+/24.04 ships apparmor_restrict_unprivileged_userns=1).
        return ("this suite executes model-written code and gbench requires bubblewrap "
                "isolation (GBENCH_SANDBOX=required, the default). bubblewrap IS installed "
                "but is BLOCKED from creating an unprivileged user namespace here. On "
                "Ubuntu 23.10+/24.04 this is the AppArmor restriction - allow it with "
                "`sudo sysctl kernel.apparmor_restrict_unprivileged_userns=0` (persist under "
                "/etc/sysctl.d/). Or set GBENCH_SANDBOX=none to run UNSANDBOXED (model code "
                "executes directly on this host)")
    return ("this suite executes model-written code and gbench requires bubblewrap for "
            "process isolation (GBENCH_SANDBOX=required, the default), but bubblewrap is not "
            "installed. Install it (`sudo apt-get install bubblewrap`) and allow unprivileged "
            "user namespaces, or set GBENCH_SANDBOX=none to run UNSANDBOXED (model code "
            "executes directly on this host)")


def wrap_argv(argv: Sequence[str], writable: Sequence[str] = (), network: bool = False,
              force: bool = False, readable: Sequence[str] = ()) -> List[str]:
    """Prefix `argv` with a bubblewrap jail, or return it unchanged when disabled.

    `writable` lists directories the command legitimately needs to write (a compiler's
    scratch dir, the temp file holding the program); `readable` lists files or directories
    under the home directory it legitimately reads. Everything else is read-only, the home
    directory is empty but for what _home_binds binds back, and credential files are hidden.
    """
    if not force and not sandbox_available():
        if sandbox_mode() in ("required", "bwrap"):
            # Defense in depth: a code-executing suite reached here without gating on
            # sandbox_skip_reason(). Fail loudly rather than silently run unsandboxed.
            raise RuntimeError(
                "Refusing to execute model-written code unsandboxed: bubblewrap is required "
                "(GBENCH_SANDBOX=required, the default) but is unavailable or blocked. Install "
                "`bubblewrap`, or set GBENCH_SANDBOX=none to run unsandboxed on purpose.")
        return list(argv)

    # /run (and /var/run when it is not a link to it) holds the Docker daemon's socket; the
    # /proc mounted after --unshare-pid shows the sandbox's own PID namespace, not the host's
    # processes and their environments.
    cmd = ["bwrap", "--ro-bind", "/", "/", "--tmpfs", "/run"]
    if os.path.isdir("/var/run") and not os.path.islink("/var/run"):
        cmd += ["--tmpfs", "/var/run"]
    cmd += ["--dev", "/dev", "--unshare-pid", "--proc", "/proc", "--unshare-ipc",
            "--unshare-uts", "--cap-drop", "ALL", "--tmpfs", "/tmp", "--die-with-parent",
            "--new-session"]
    if not network:
        cmd += ["--unshare-net"]
    homes = masked_homes()
    for home in homes:
        cmd += ["--tmpfs", home]
    binds = _home_binds(homes, readable)
    for src, dest in binds:
        cmd += ["--ro-bind", src, dest]
    shown = [p for p in writable if p and os.path.isdir(p)]
    for path in shown:
        cmd += ["--bind", path, path]
    masked: List[str] = []
    for path, is_dir in credential_paths(homes, [src for src, _ in binds]
                                         + [os.path.realpath(p) for p in shown]):
        # A credential under a directory bound at a second path (see _home_binds) is visible
        # there too, so it is masked at both places.
        aliases = [dest if path == src else os.path.join(dest, os.path.relpath(path, src))
                   for src, dest in binds if dest != src and _under(path, src)]
        for target in [path] + aliases:
            if target not in masked:
                masked.append(target)
                cmd += ["--tmpfs", target] if is_dir else ["--ro-bind", "/dev/null", target]
    return cmd + list(argv)


#: Variables naming data files an operator configures for a sandboxed program to read by path
#: (scicode's test_data.h5); bound back read-only when they lie in an emptied home directory.
SANDBOX_READABLE_ENV = ("GBENCH_SCICODE_TEST_DATA", "SCICODE_TEST_DATA")

#: Host directories the jail replaces with an empty tmpfs whatever the home directory is.
_EMPTIED = ("/tmp", "/run", "/var/run")

#: Homes masked_homes has already warned about (it runs for every sandboxed program).
_WARNED_HOMES: set = set()


def _under(path: str, root: str) -> bool:
    """`path` is `root` or lies inside it (both real paths)."""
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _prefixes() -> List[str]:
    return sorted({os.path.realpath(p) for p in (sys.prefix, sys.exec_prefix, sys.base_prefix,
                                                 sys.base_exec_prefix) if p})


def masked_homes() -> List[str]:
    """The home directories the jail replaces with an empty one: $HOME and the account's home
    from the password database, when they are real directories. A home that is one of the
    interpreter's prefixes, or lies inside one, cannot be emptied without breaking the
    interpreter; it stays (with a warning), and credential_paths still hides the credential
    files in it."""
    candidates = [os.environ.get("HOME")]
    try:
        import pwd
        candidates.append(pwd.getpwuid(os.getuid()).pw_dir)
    except (ImportError, KeyError):  # pragma: no cover - platform dependent
        pass
    homes: List[str] = []
    for cand in candidates:
        if not cand:
            continue
        home = os.path.realpath(os.path.expanduser(cand))
        if not os.path.isdir(home) or home == os.sep or home in homes:
            continue
        if any(_under(home, p) for p in _prefixes()):
            if home not in _WARNED_HOMES:
                _WARNED_HOMES.add(home)
                logger.warning("code sandbox: %s holds the interpreter, so it is not emptied; "
                               "only its credential files are hidden", home)
            continue
        homes.append(home)
    return homes


def _hf_hub_cache() -> str:
    """huggingface_hub's cache directory (constants.HF_HUB_CACHE), from the environment."""
    env = os.environ
    return (env.get("HF_HUB_CACHE") or env.get("HUGGINGFACE_HUB_CACHE")
            or os.path.join(env.get("HF_HOME") or os.path.join(
                env.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"),
                "huggingface"), "hub"))


def _wanted_paths(readable: Sequence[str] = ()) -> List[str]:
    """The paths a sandboxed program legitimately reads, as written (not resolved): the
    interpreter's directories (its prefixes, the directory of the executable, the sys.path
    entries inside them or named site-packages / dist-packages, the user site), the Hugging Face
    hub cache (dataset and model files; the token files sit outside it), the files named by
    SANDBOX_READABLE_ENV and the caller's `readable` paths."""
    import site
    prefixes = _prefixes()
    wanted = [sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix] + prefixes
    wanted += [os.path.dirname(os.path.abspath(sys.executable)),
               os.path.dirname(os.path.realpath(sys.executable))]
    try:
        wanted.append(site.getusersitepackages())
    except Exception:  # noqa: BLE001 - this interpreter has no user site
        pass
    for entry in sys.path:
        real = os.path.realpath(entry) if entry else ""
        if real and (any(_under(real, p) for p in prefixes)
                     or os.path.basename(real) in ("site-packages", "dist-packages")):
            wanted.append(entry)
    wanted.append(_hf_hub_cache())
    wanted += [os.environ.get(name) or "" for name in SANDBOX_READABLE_ENV]
    wanted += list(readable)
    return wanted


def _home_rebinds(homes: Sequence[str], readable: Sequence[str] = ()) -> List[str]:
    """What is bound back, read-only, into the emptied home directories at its real path: every
    path of _wanted_paths whose real path lies inside an emptied home, never a home itself or a
    directory containing one."""
    if not homes:
        return []
    out: List[str] = []
    for path in _wanted_paths(readable):
        if not path:
            continue
        real = os.path.realpath(os.path.expanduser(path))
        if (real in out or not os.path.exists(real)
                or not any(_under(real, h) for h in homes)
                or any(_under(h, real) for h in homes)):
            continue
        out.append(real)
    # parents before children, so a child bind lands inside its parent's bind
    return sorted(out, key=lambda p: (p.count(os.sep), p))


def _through_home_link(path: str, homes: Sequence[str]) -> Optional[str]:
    """Where `path` (absolute) lands in the jail when a symlink inside an emptied home lies on
    it: its components up to that symlink resolved (so the result lies in the emptied home even
    when the home itself is reached through a symlink, e.g. /home linking to another disk), the
    rest as written. None when no symlink inside an emptied home lies on the path."""
    parts = [p for p in path.split(os.sep) if p]
    cur = os.sep
    for i, comp in enumerate(parts):
        nxt = os.path.join(cur, comp)
        if os.path.islink(nxt) and any(_under(cur, h) for h in homes):
            return os.path.join(nxt, *parts[i + 1:])
        cur = os.path.realpath(nxt)
    return None


def _home_binds(homes: Sequence[str], readable: Sequence[str] = ()) -> List[Tuple[str, str]]:
    """(source, destination) read-only binds into the emptied home directories: each path of
    _home_rebinds at its real location, plus, for a wanted path the program reaches through a
    symlink inside an emptied home (for example ~/.cache/huggingface linking to another disk,
    or an interpreter reached as ~/venv/bin/python), its real target bound again where that
    path lands in the jail (_through_home_link: the path as written, with the part before the
    link resolved). The emptied home no longer holds the link, so without the second bind that
    path would not resolve in the jail."""
    if not homes:
        return []
    out = [(p, p) for p in _home_rebinds(homes, readable)]
    for path in _wanted_paths(readable):
        if not path:
            continue
        written = os.path.abspath(os.path.expanduser(path))
        real = os.path.realpath(written)
        dest = _through_home_link(written, homes) if os.path.exists(real) else None
        if (dest is None or dest == real or (real, dest) in out
                or any(_under(h, dest) for h in homes)):
            continue
        out.append((real, dest))
    # parents before children, so a child bind lands inside its parent's bind
    return sorted(out, key=lambda p: (p[1].count(os.sep), p[1]))


def credential_paths(homes: Sequence[str] = (), shown: Sequence[str] = ()
                     ) -> List[Tuple[str, bool]]:
    """Credential files and directories the jail hides, as (real path, is a directory): the
    huggingface_hub token files (the path base.secret_values reads, and stored_tokens next to
    it), $GOOGLE_APPLICATION_CREDENTIALS, the gcloud configuration ($CLOUDSDK_CONFIG and
    ~/.config/gcloud), the Docker client configuration ($DOCKER_CONFIG, else ~/.docker), the AWS,
    Azure, kube and gh configurations, ~/.netrc, ~/.git-credentials, ~/.pypirc, ~/.npmrc,
    ~/.boto, ~/.vault-token, ~/.ssh and ~/.gnupg. A symlink is hidden by its target. Only paths
    the jail would otherwise show are listed: those outside the emptied `homes` and _EMPTIED, or
    inside a directory `shown` in the jail (bound back or writable)."""
    env = os.environ
    home = os.path.expanduser("~")
    hf_home = env.get("HF_HOME") or os.path.join(
        env.get("XDG_CACHE_HOME") or os.path.join(home, ".cache"), "huggingface")
    config = env.get("XDG_CONFIG_HOME") or os.path.join(home, ".config")
    files = [env.get("HF_TOKEN_PATH") or os.path.join(hf_home, "token"),
             os.path.join(hf_home, "stored_tokens"),
             os.path.join(home, ".cache", "huggingface", "token"),
             os.path.join(home, ".huggingface", "token"),
             env.get("GOOGLE_APPLICATION_CREDENTIALS"), env.get("AWS_SHARED_CREDENTIALS_FILE"),
             env.get("AWS_CONFIG_FILE"), env.get("BOTO_CONFIG"), env.get("NETRC"),
             *(env.get("KUBECONFIG") or "").split(os.pathsep),
             *(os.path.join(home, n) for n in (".netrc", ".git-credentials", ".pypirc",
                                                ".npmrc", ".boto", ".vault-token"))]
    dirs = [env.get("CLOUDSDK_CONFIG"), os.path.join(home, ".config", "gcloud"),
            os.path.join(config, "gcloud"), env.get("DOCKER_CONFIG"),
            os.path.join(home, ".docker"), env.get("AZURE_CONFIG_DIR"), env.get("GH_CONFIG_DIR"),
            os.path.join(config, "gh"),
            *(os.path.join(home, n) for n in (".aws", ".azure", ".kube", ".ssh", ".gnupg"))]
    emptied = list(homes) + [os.path.realpath(d) for d in _EMPTIED]
    out: List[Tuple[str, bool]] = []
    for paths, is_dir in ((files, False), (dirs, True)):
        for path in paths:
            if not path:
                continue
            real = os.path.realpath(os.path.expanduser(path))
            if (real, is_dir) in out or not (os.path.isdir(real) if is_dir
                                              else os.path.isfile(real)):
                continue
            if any(_under(real, e) for e in emptied) and not any(_under(real, v)
                                                                 for v in shown):
                continue                    # the jail does not show it anyway
            out.append((real, is_dir))
    return out


def credential_free_env(env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """`env` (default: this process's environment) without the credential variables
    (base.is_secret_env_name) and without any variable whose value carries a credential value
    (a URL with a password, a copied key): what model-written code is given."""
    from .base import is_secret_env_name, secret_values
    env = dict(os.environ if env is None else env)
    secrets = secret_values()
    return {k: v for k, v in env.items()
            if not is_secret_env_name(k) and not any(s in (v or "") for s in secrets)}


def run_sandboxed(argv: Sequence[str], writable: Sequence[str] = (), network: bool = False,
                  readable: Sequence[str] = (), **kwargs: Any) -> "subprocess.CompletedProcess":
    """`subprocess.run` for model-written code, isolated when bubblewrap is available (the
    jail of wrap_argv: the home emptied, credential files hidden). The program gets `kwargs["env"]`
    (default: this process's environment) without its credentials (credential_free_env)."""
    kwargs["env"] = credential_free_env(kwargs.get("env"))
    return subprocess.run(wrap_argv(argv, writable=writable, network=network,
                                    readable=readable), **kwargs)
