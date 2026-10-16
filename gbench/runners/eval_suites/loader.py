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

"""Dynamic plugin loader for custom and private evaluation suites in gbench."""

import importlib.util
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Registry mapping custom eval name -> custom capability pillar title
CUSTOM_PILLARS: Dict[str, str] = {}


def discover_and_register_plugins(
    plugin_paths: Optional[List[str]] = None,
) -> Dict[str, Callable[..., Dict[str, Any]]]:
    """Scan directories or files and dynamically register custom evaluation runner functions.

    Supports:
      - Paths passed via `--eval-plugins-dir` / `--eval-plugins-path`.
      - Colon-separated paths in `GBENCH_EVAL_PLUGINS_PATH` environment variable.
    """
    from . import SUITES

    all_paths: List[Path] = []

    # 1. From environment variable
    env_paths = os.environ.get("GBENCH_EVAL_PLUGINS_PATH", "").strip()
    if env_paths:
        for p in env_paths.split(os.pathsep):
            if p.strip():
                all_paths.append(Path(p.strip()))

    # 2. From explicit argument paths
    if plugin_paths:
        for p in plugin_paths:
            if p.strip():
                all_paths.append(Path(p.strip()))

    discovered: Dict[str, Callable[..., Dict[str, Any]]] = {}

    for path in all_paths:
        if not path.exists():
            logger.warning(f"Custom eval plugin path does not exist: {path}")
            continue

        python_files = [path] if path.is_file() and path.suffix == ".py" else list(path.rglob("*.py"))

        for py_file in python_files:
            if py_file.name.startswith(("_", ".")) or py_file.name.startswith("test_") or py_file.name in ("dataset_utils.py",):
                continue
            if any(part.startswith((".", "__")) for part in py_file.parts):
                continue

            try:
                if str(py_file.parent) not in sys.path:
                    sys.path.insert(0, str(py_file.parent))
                module_name = f"gbench_custom_{py_file.stem}"
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if not spec or not spec.loader:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                file_discovered: List[str] = []

                # Check for explicit SUITES dictionary
                if hasattr(module, "SUITES") and isinstance(module.SUITES, dict):
                    for name, fn in module.SUITES.items():
                        if callable(fn):
                            clean_name = name.lower().strip()
                            SUITES[clean_name] = fn
                            discovered[clean_name] = fn
                            file_discovered.append(clean_name)
                            logger.info(f"Registered custom eval plugin '{clean_name}' from {py_file.name}")

                # Check for run_* functions defined inside this module
                for attr_name in dir(module):
                    if attr_name.startswith("run_") and attr_name != "run_eval_suite":
                        fn = getattr(module, attr_name)
                        if callable(fn) and getattr(fn, "__module__", None) == module.__name__:
                            eval_name = attr_name[4:].lower().strip()
                            SUITES[eval_name] = fn
                            discovered[eval_name] = fn
                            file_discovered.append(eval_name)
                            logger.info(f"Registered custom eval runner '{eval_name}' from {py_file.name}")

                # Check for custom pillar category
                pillar_title = getattr(module, "PILLAR", None) or getattr(module, "PILLAR_TITLE", None)
                if pillar_title:
                    for name in file_discovered:
                        CUSTOM_PILLARS[name] = str(pillar_title)

            except Exception as e:
                logger.error(f"Failed to load custom eval plugin from {py_file}: {e}")

    return discovered
