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

"""Build-time check of the terminal_bench harness image (terminal_bench.Dockerfile).

    python terminal_bench_verify_env.py <constraints file> <harbor version> <freeze output>

Fails the build (exit 1, naming each problem) unless the installed Harbor is exactly
<harbor version> and every installed distribution other than pip's own tooling is pinned in the
constraints file at the installed version, so the image holds the pinned set and nothing else.
Writes the installed set, `name==version` per line, to <freeze output>.
"""

import importlib.metadata
import re
import sys
from typing import Dict, List

#: The installer's own tooling, part of the base image rather than of Harbor's closure.
TOOLING = frozenset({"pip", "setuptools", "wheel"})


def canonical(name: str) -> str:
    """PEP 503 normalized project name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_constraints(path: str) -> Dict[str, str]:
    """{normalized name: version} of the `name==version` lines of `path` (comments skipped)."""
    pins: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            name, sep, version = line.partition("==")
            if not sep or not name.strip() or not version.strip():
                raise SystemExit(f"terminal_bench_verify_env: not a name==version pin: {raw!r}")
            pins[canonical(name.strip())] = version.strip()
    return pins


def installed() -> Dict[str, str]:
    """{normalized name: version} of every installed distribution."""
    return {canonical(d.metadata["Name"]): d.version for d in importlib.metadata.distributions()}


def problems(pins: Dict[str, str], found: Dict[str, str], harbor_version: str) -> List[str]:
    """What keeps the installed set from being exactly the pinned one."""
    out: List[str] = []
    if found.get("harbor") != harbor_version:
        out.append(f"harbor is {found.get('harbor')!r}, not {harbor_version}")
    for name, version in sorted(found.items()):
        if name in TOOLING:
            continue
        if name not in pins:
            out.append(f"{name}=={version} is installed but not pinned in the constraints")
        elif pins[name] != version:
            out.append(f"{name} is {version}, the constraints pin {pins[name]}")
    return out


def main(argv: List[str]) -> int:
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    constraints, harbor_version, freeze_path = argv[1:]
    found = installed()
    bad = problems(read_constraints(constraints), found, harbor_version)
    if bad:
        for line in bad:
            print(f"terminal_bench_verify_env: {line}", file=sys.stderr)
        return 1
    with open(freeze_path, "w", encoding="utf-8") as f:
        for name, version in sorted(found.items()):
            f.write(f"{name}=={version}\n")
    print(f"terminal_bench_verify_env: harbor {harbor_version} and {len(found)} distributions "
          f"match the constraints")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
