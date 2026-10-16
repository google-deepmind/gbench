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

"""Dependency validation and optional engine availability checks."""

import importlib.metadata
import importlib.util
import os
import shlex
import sys
from typing import Any, Dict, List, Optional

#: Replaces the command gbench runs vLLM's CLI with (shell-split), e.g. a vllm binary of another
#: installation. Unset: the interpreter gbench runs in (vllm_cli()).
VLLM_BIN_ENV = "GBENCH_VLLM_BIN"


def check_vllm_available() -> bool:
    """Check if the vllm package is installed and importable."""
    try:
        import vllm  # noqa: F401
        return True
    except ImportError:
        return False


def require_vllm_engine(feature_name: str = "Local vLLM engine execution") -> None:
    """Ensure vllm is installed.

    If missing, outputs a user-friendly error and exits gracefully.
    """
    if not check_vllm_available():
        sys.stderr.write(
            f"\n[ERROR] {feature_name} requires a local vLLM serving stack, which is not installed.\n\n"
            f"gbench does NOT install vllm/torch itself (they are GPU/CUDA-specific and often a\n"
            f"validated custom build; auto-installing would overwrite it or pull a mismatched CUDA).\n"
            f"Provision a serving stack for your hardware, then re-run:\n"
            f"    pip install vllm        # fresh machine: pulls a matching torch/CUDA\n"
            f"    # or use your existing validated torch + vLLM build\n\n"
            f"Only LOCAL serving needs this; runs against a served model with --remote-endpoint do not.\n\n"
        )
        sys.exit(1)


def vllm_cli() -> List[str]:
    """The command prefix that runs vLLM's CLI (``<prefix> serve ...``,
    ``<prefix> bench throughput ...``).

    Default: the interpreter gbench runs in with ``-m vllm.entrypoints.cli.main``, so the servers
    and offline benchmarks gbench launches use the vLLM that require_vllm_engine checked. A bare
    ``vllm`` would be looked up on PATH, whose first entry may belong to another environment.
    ``GBENCH_VLLM_BIN`` (shell-split) replaces it; vllm_engine_record() says which one ran."""
    raw = (os.environ.get(VLLM_BIN_ENV) or "").strip()
    if raw:
        return shlex.split(raw)
    return [sys.executable, "-m", "vllm.entrypoints.cli.main"]


def vllm_engine_record() -> Dict[str, Any]:
    """What the performance results record about the vLLM that ran: the CLI prefix
    (vllm_cli()), where it came from, and the version and location of the vllm package gbench's
    interpreter imports (None when it has none). With GBENCH_VLLM_BIN the version is that of the
    interpreter's package, which the override may not use; ``cli_source`` says so."""
    try:
        version: Optional[str] = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = None
    except Exception:                                                      # noqa: BLE001
        version = None
    try:
        spec = importlib.util.find_spec("vllm")
        path = spec.origin if spec is not None else None
    except (ImportError, ValueError):
        path = None
    override = bool((os.environ.get(VLLM_BIN_ENV) or "").strip())
    return {
        "cli": vllm_cli(),
        "cli_source": VLLM_BIN_ENV if override else "gbench's interpreter",
        "package_version": version,
        "package_path": path,
    }


def safe_get_tokenizer(tokenizer_path: str, custom_tokenizer: Optional[str] = None,
                       trust_remote_code: bool = False):
    """Safely load a tokenizer without crashing on Ollama tags or invalid HF repo IDs.

    Prioritizes custom_tokenizer if provided (e.g. via --tokenizer). ``trust_remote_code``:
    core.config.tokenizer_trust_remote_code (--tokenizer-trust-remote-code, else
    --trust-remote-code in --vllm-serve-args).
    """
    target_path = custom_tokenizer or tokenizer_path
    if not target_path or ":" in target_path or ("/" not in target_path and not os.path.exists(target_path)):
        return None
    try:
        from vllm.tokenizers import get_tokenizer
        return get_tokenizer(target_path, trust_remote_code=trust_remote_code)
    except Exception:
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(target_path, trust_remote_code=trust_remote_code)
        except Exception:
            return None
