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

"""Build-time check of a local image's Python packages against its recorded pins.

    python pinned_pip_check.py <pins file> [--source NAME ...] [--build-only NAME ...]
                               [--ignore-dir DIR ...] [--enforce] [--freeze OUT]

<pins file> holds `name==version` lines (comments and blank lines ignored): the installed set of
an earlier build of the image. The installed distributions must be exactly the pinned ones at the
pinned versions, except the source installs named with --source (pinned by commit in the
Dockerfile) and pip itself; a pin named with --build-only (a tool only the build uses, such as
Cython) need not be installed. Distributions under an --ignore-dir directory are not pip's (the
Debian-packaged Python modules of /usr/lib/python3/dist-packages, which float with the Debian or
Ubuntu archive like the image's other system packages). A distribution with several installed
metadata directories (what `pip install -I` over an installed version leaves behind) matches when
one of them is the pinned version, which is what `pip freeze` reported when the pins were taken.
The check exits 1, naming each difference, so a rebuild that resolved anything else fails instead
of producing a different image.

With --enforce (for Dockerfiles whose upstream steps reinstall packages at other versions on the
way, so a pip constraint cannot hold every step), the pins that are missing or installed at
another version are installed first with `pip install --no-deps name==version` and every
unpinned distribution is uninstalled; then the check runs. --freeze writes the installed set,
`name==version` per line (every installed version of a name), to OUT. Stdlib only.
"""

import argparse
import importlib
import importlib.metadata
import os
import re
import subprocess
import sys
from typing import Dict, List, Tuple

#: pip's own tooling may be absent from a pins file written before it was recorded.
TOOLING = frozenset({"pip"})


def canonical(name: str) -> str:
    """PEP 503 normalized project name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_pins(path: str) -> Dict[str, Tuple[str, str]]:
    """{canonical name: (name as written, version)} of the `name==version` lines of `path`."""
    pins: Dict[str, Tuple[str, str]] = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if "==" not in line:
                raise SystemExit(f"{path}: not a name==version pin: {raw.rstrip()}")
            name, version = (part.strip() for part in line.split("==", 1))
            pins[canonical(name)] = (name, version)
    return pins


def installed(ignore_dirs: List[str] = ()) -> Dict[str, List[str]]:
    """{canonical name: [version, ...]} of every distribution the interpreter sees outside
    `ignore_dirs` (a name with several metadata directories lists each version once)."""
    importlib.invalidate_caches()
    ignored = [os.path.realpath(d).rstrip("/") + "/" for d in ignore_dirs]
    out: Dict[str, List[str]] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        where = os.path.realpath(str(dist.locate_file(""))).rstrip("/") + "/"
        if any(where.startswith(d) for d in ignored):
            continue
        if name:
            versions = out.setdefault(canonical(name), [])
            if dist.version not in versions:
                versions.append(dist.version)
    return out


def problems(pins: Dict[str, Tuple[str, str]], have: Dict[str, List[str]],
             sources: List[str], build_only: List[str] = ()) -> List[str]:
    """Each installed distribution that is not pinned or none of whose versions is the pin, and
    each pin (other than a build-only one) that is not installed."""
    skip = {canonical(s) for s in sources} | TOOLING
    out = []
    for name, versions in sorted(have.items()):
        if name in skip:
            continue
        shown = ", ".join(versions)
        if name not in pins:
            out.append(f"{name} {shown} is installed but not pinned")
        elif pins[name][1] not in versions:
            out.append(f"{name} {shown} is installed, the pin is {pins[name][1]}")
    optional = skip | {canonical(b) for b in build_only}
    for key, (name, version) in sorted(pins.items()):
        if key not in have and key not in optional:
            out.append(f"{name}=={version} is pinned but not installed")
    return out


def enforce(pins: Dict[str, Tuple[str, str]], have: Dict[str, List[str]],
            sources: List[str], build_only: List[str] = ()) -> None:
    """Install the missing and drifted pins without dependencies, uninstall the unpinned."""
    skip = {canonical(s) for s in sources} | TOOLING
    optional = skip | {canonical(b) for b in build_only}
    fix = [f"{n}=={v}" for key, (n, v) in sorted(pins.items())
           if key not in optional and v not in have.get(key, [])]
    extra = sorted(n for n in have if n not in skip and n not in pins)
    if fix:
        print("installing the pinned versions of:", " ".join(fix), flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps",
                        *fix], check=True)
    if extra:
        print("uninstalling the unpinned:", " ".join(extra), flush=True)
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", *extra], check=True)


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("pins")
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--build-only", action="append", default=[])
    ap.add_argument("--ignore-dir", action="append", default=[])
    ap.add_argument("--enforce", action="store_true")
    ap.add_argument("--freeze")
    args = ap.parse_args(argv)
    pins = read_pins(args.pins)
    if args.enforce:
        enforce(pins, installed(args.ignore_dir), args.source, args.build_only)
    have = installed(args.ignore_dir)
    bad = problems(pins, have, args.source, args.build_only)
    if bad:
        print(f"the installed packages differ from {args.pins}:", file=sys.stderr)
        for line in bad:
            print("  " + line, file=sys.stderr)
        return 1
    if args.freeze:
        with open(args.freeze, "w", encoding="utf-8") as f:
            f.writelines(f"{n}=={v}\n" for n, vs in sorted(have.items()) for v in vs)
    print(f"{len(have)} installed distributions match {args.pins}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
