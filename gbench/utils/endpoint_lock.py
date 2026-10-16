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

"""A host-wide reader/writer lock per model server, so performance measurements run
alone.

Every gbench process that sends traffic to a server holds the server's lock SHARED for
as long as it sends (eval pillars, the eval pool, the smoke run); a performance pillar
holds it EXCLUSIVE for the whole pillar. So a perf pillar never measures a server that
gbench is also evaluating, and an eval run never starts under a perf pillar.

The key is the upstream server, not the URL string: localhost, 127.0.0.1, 0.0.0.0 and
this host's own interface addresses all mean the same local server. The process that
knows the real upstream computes the key once and exports GBENCH_ENDPOINT_LOCK_KEY;
children (pool children, smoke taps, relays reached through another URL) inherit it.

Acquisition never waits silently: a perf pillar polls for up to GBENCH_PERF_LOCK_WAIT_S
(default 0) and then fails with the holders named; a shared holder fails at once if a
perf pillar holds the server. Holders (pid, start time, argv) are recorded next to the
lock file. POSIX only (fcntl.flock); elsewhere the lock is a no-op that says so.
"""

import json
import os
import socket
import sys
import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional
from urllib.parse import urlparse

KEY_ENV = "GBENCH_ENDPOINT_LOCK_KEY"
WAIT_ENV = "GBENCH_PERF_LOCK_WAIT_S"
POLL_S = 5.0
REPORT_EVERY_S = 60.0


class EndpointBusy(RuntimeError):
    """The server's lock is held in a conflicting mode by another gbench process."""


def _local_addresses() -> set:
    addrs = {"127.0.0.1", "::1", "0.0.0.0", "::", "localhost"}
    try:
        host = socket.gethostname()
        addrs.add(host)
        for info in socket.getaddrinfo(host, None):
            addrs.add(info[4][0])
    except OSError:
        pass
    try:                                          # every interface address (Linux)
        import fcntl                              # noqa: F401  (POSIX check)
        import subprocess
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2)
        addrs.update(out.stdout.split())
    except Exception:                             # noqa: BLE001
        pass
    return addrs


def endpoint_key(url: str) -> str:
    """Canonical lock key for a server URL: 'local:<port>' for any address of this host,
    else '<host>:<port>' (lower-case; default ports filled in)."""
    p = urlparse(url if "://" in url else "http://" + url)
    host = (p.hostname or "localhost").lower()
    port = p.port or (443 if p.scheme == "https" else 80)
    try:
        resolved = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except OSError:
        resolved = set()
    local = _local_addresses()
    if host in local or resolved & local or any(a.startswith("127.") for a in resolved):
        return f"local:{port}"
    # A remote host is keyed on its resolved addresses (two names for one server share a
    # lock); behind DNS round-robin or a load balancer, use one URL consistently. A
    # container IP and a published port of the same server are not unified.
    if resolved:
        return f"{'+'.join(sorted(resolved))}:{port}"
    return f"{host}:{port}"


def inherited_or(url: Optional[str]) -> Optional[str]:
    """The key a parent exported, else the key of ``url`` (None without either)."""
    k = os.environ.get(KEY_ENV)
    if k:
        return k
    return endpoint_key(url) if url else None


def _lock_dir() -> str:
    d = os.environ.get("GBENCH_LOCK_DIR")
    if not d:
        base = os.environ.get("GBENCH_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".cache", "gbench")
        d = os.path.join(base, "endpoint-locks")
    os.makedirs(d, exist_ok=True)
    return d


def _paths(key: str):
    """Lock and holders file names: a short readable prefix + a hash of the full key (a
    multi-IP key can exceed the filesystem's name limit)."""
    import hashlib
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in key)[:40]
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    d = _lock_dir()
    return os.path.join(d, f"{safe}-{digest}.lock"), os.path.join(d, f"{safe}-{digest}.holders")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _update_holders(path: str, add: Optional[dict] = None, remove_pid: Optional[int] = None) -> List[dict]:
    import fcntl
    with open(path, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        try:
            holders = json.loads(fh.read() or "[]")
        except ValueError:
            holders = []
        holders = [h for h in holders if _alive(int(h.get("pid", 0)))
                   and h.get("pid") != remove_pid]
        if add:
            holders = [h for h in holders if h.get("pid") != add["pid"]] + [add]
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps(holders))
        fh.flush()
        fcntl.flock(fh, fcntl.LOCK_UN)
    return holders


def holders(key: str) -> List[dict]:
    """The live processes recorded as holding ``key`` (empty when none or not POSIX)."""
    try:
        return _update_holders(_paths(key)[1])
    except ImportError:
        return []


def _describe(hs: List[dict]) -> str:
    if not hs:
        return "unknown holder"
    return "; ".join(f"pid {h['pid']} ({h.get('mode')}, since "
                     f"{time.strftime('%H:%M:%S', time.localtime(h.get('since', 0)))}: "
                     f"{' '.join(h.get('argv', [])[:6])})" for h in hs)


@contextmanager
def hold(key: Optional[str], exclusive: bool, wait_s: Optional[float] = None,
         log=None) -> Iterator[Optional[str]]:
    """Hold the server lock for ``key`` (shared, or exclusive for a perf pillar).

    Exclusive: poll every 5 s for up to ``wait_s`` (default GBENCH_PERF_LOCK_WAIT_S, 0),
    reporting the holders every 60 s, then raise EndpointBusy naming them. Shared: raise
    EndpointBusy at once if an exclusive holder exists. Yields the key (None: no key, or
    no flock on this platform, and nothing is locked)."""
    if not key:
        yield None
        return
    try:
        import fcntl
    except ImportError:                           # not POSIX: nothing to lock with
        if log:
            log(f"endpoint lock unavailable on this platform; {key} is not locked")
        yield None
        return
    try:
        lock_path, holders_path = _paths(key)
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    except OSError as e:                          # unwritable lock dir etc.: never fatal
        if log:
            log(f"endpoint lock unavailable ({e}); {key} is not locked")
        yield None
        return
    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    if wait_s is None:
        wait_s = float(os.environ.get(WAIT_ENV) or 0.0) if exclusive else 0.0
    deadline = time.time() + max(0.0, wait_s)
    last_report = 0.0
    try:
        while True:
            try:
                fcntl.flock(fd, mode | fcntl.LOCK_NB)
                break
            except OSError as e:
                if not isinstance(e, BlockingIOError):   # flock unsupported (e.g. some NFS)
                    if log:
                        log(f"endpoint lock unavailable ({e}); {key} is not locked")
                    os.close(fd)
                    fd = None
                    yield None
                    return
                hs = holders(key)
                now = time.time()
                if now >= deadline:
                    kind = "a performance pillar" if not exclusive else "other gbench runs"
                    raise EndpointBusy(f"endpoint {key} is busy (held by {kind}): {_describe(hs)}")
                if log and now - last_report >= REPORT_EVERY_S:
                    log(f"waiting for endpoint {key}: {_describe(hs)}")
                    last_report = now
                time.sleep(min(POLL_S, max(0.1, deadline - now)))
        _update_holders(holders_path, add={"pid": os.getpid(), "since": time.time(),
                                           "mode": "exclusive" if exclusive else "shared",
                                           "argv": sys.argv})
        try:
            yield key
        finally:
            _update_holders(holders_path, remove_pid=os.getpid())
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        if fd is not None:
            os.close(fd)
