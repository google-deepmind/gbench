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

"""Work around a Harbor terminus-2 bug in the truncated-response salvage path.

Harbor 0.20.0, `harbor/agents/terminus_2/terminus_2.py`::

    1104    if salvaged_response:
    1110        return salvaged_response            # <- a plain str
    ...
    1168    result = self._parser.parse_response(llm_response.content)

`_query_llm` is declared to return an `LLMResponse`, and every other path does. The
salvage branch returns the salvaged text itself, so the caller immediately raises::

    AttributeError: 'str' object has no attribute 'content'

and the whole trial dies.

Reachability: the branch runs only when a turn hits `max_tokens` AND the parser exposes
`salvage_truncated_response` AND salvage succeeds. Only `terminus_xml_plain_parser`
implements salvage, so with terminus-2's default `json` parser the branch is dead code and
the bug is invisible. gbench switched to the XML parser to stop truncated turns
livelocking the agent (they were re-asked forever, burning trials to their timeout), which
made this reachable: on 2026-08-19, 2 of 3 trials died here, each after a single truncated
turn.

Turn size is not a workaround. Those trials averaged 153-790 tokens per turn against an
8192 cap - it is one runaway turn that trips it, so raising the cap lowers the odds without
removing them.

This patch coerces a `str` return into `LLMResponse(content=...)`, which is what the
caller has always expected. Everything else is left alone.

Runs *inside the `harbor` subprocess* via :func:`wrap_command` - patching gbench's own
process would do nothing, because terminal_bench shells out to the Harbor CLI. Mirrors
`swe_thread_cap`, which installs its container thread cap the same way.
"""

import logging
import os
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Set to 0/off to run Harbor unpatched (restores the crash; useful for confirming it).
ENV_VAR = "GBENCH_TB_SALVAGE_PATCH"


def enabled() -> bool:
    return (os.environ.get(ENV_VAR) or "").strip().lower() not in ("0", "off", "false", "none")


def install(_agent_cls: Optional[type] = None, _response_cls: Optional[type] = None) -> bool:
    """Patch `TerminusAgent._query_llm` to always return an `LLMResponse`.

    Idempotent. Returns True when the patch is in place, False when it could not be
    applied - never raises, because failing to patch must not stop the eval running.

    `_agent_cls` / `_response_cls` are testing seams so the unit tests need not mutate the
    real Harbor classes; production callers leave them None.
    """
    agent_cls, response_cls = _agent_cls, _response_cls
    if agent_cls is None or response_cls is None:
        try:
            from harbor.agents.terminus_2.terminus_2 import Terminus2 as _A
            from harbor.llms.base import LLMResponse as _R
        except Exception as exc:                      # pragma: no cover - import shape varies
            try:
                import harbor.agents.terminus_2.terminus_2 as _m
                _A = next(v for k, v in vars(_m).items()
                          if isinstance(v, type) and k.lower().startswith("terminus"))
                from harbor.llms.base import LLMResponse as _R
            except Exception:
                logger.warning("terminus_salvage_patch: cannot import Harbor (%s); "
                               "leaving it unpatched.", exc)
                return False
        agent_cls = agent_cls or _A
        response_cls = response_cls or _R

    if getattr(agent_cls, "_gbench_salvage_patched", False):
        return True
    original = getattr(agent_cls, "_query_llm", None)
    if original is None:
        logger.warning("terminus_salvage_patch: %s has no _query_llm; not patching.",
                       agent_cls.__name__)
        return False

    async def _query_llm(self, *args: Any, **kwargs: Any):
        result = await original(self, *args, **kwargs)
        if isinstance(result, str):
            # The salvage branch. Wrap it in the type every caller expects.
            return response_cls(content=result)
        return result

    _query_llm.__name__ = "_query_llm"
    _query_llm.__doc__ = (original.__doc__ or "") + "\n\n[gbench] coerces a salvaged str "
    agent_cls._query_llm = _query_llm
    agent_cls._gbench_salvage_patched = True
    return True


def _bootstrap_source(module_path: str) -> str:
    """Python for ``-c``: install the patch, then run Harbor's CLI.

    Loads this module by absolute path rather than as
    `gbench.runners.eval_suites.terminus_salvage_patch`, so the subprocess does not import
    the whole gbench package (and its 89 suites) to run a small shim. This module
    deliberately imports nothing from gbench so it can load standalone.
    """
    return (
        "import importlib.util,sys\n"
        f"_s=importlib.util.spec_from_file_location('_gb_salvage',{module_path!r})\n"
        "_m=importlib.util.module_from_spec(_s);_s.loader.exec_module(_m)\n"
        "_m.install()\n"
        "from harbor.cli.main import app\n"
        "sys.argv[0]='harbor'\n"
        "app()\n"
    )


def wrap_command(cmd: Sequence[str], python: Optional[str] = None) -> List[str]:
    """Re-express `harbor run ...` so the patch is installed before Harbor starts.

    Returns `cmd` unchanged when disabled or when the shape is not the expected
    `["harbor", ...]`, so this can never be the reason terminal_bench stops working.
    """
    import sys
    cmd = list(cmd)
    if not enabled() or not cmd or os.path.basename(cmd[0]) != "harbor":
        return cmd
    return [python or sys.executable, "-c", _bootstrap_source(os.path.abspath(__file__)),
            *cmd[1:]]
