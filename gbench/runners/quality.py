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

"""Quality benchmark runner using gemmaclaw QA suites.

Every external model the GemmaClaw QA suite calls besides the model under test runs on the Gemini
API through GemmaClaw's google plugin: the alternate model is passed as --alt-model
GEMINI_ALT_MODEL in place of the live-frontier lane's openai/gpt-5.4 default, the image-generation
model the media scenarios configure is GEMINI_IMAGE_GENERATION_MODEL in place of openai/gpt-image-1
(one recorded edit of the checkout), and the google plugin's other capabilities (memory
embeddings, image understanding) take the place of the OpenAI ones the lane would use.

The suite runs in a local Docker container (docker/quality.Dockerfile, built by gbench): the QA
lab, its gateways, and every tool the model under test runs, exec included, live only there, as
a user without capabilities, over a read-only copy of the checkout. The container has no network.
Its one way out is a loopback forwarder inside it that hands every request to a key relay in this
process over a unix socket; the relay alone holds the Gemini key (GEMINI_API_KEYS, else
GEMINI_API_KEY) and a remote endpoint's own key, forwards only the Gemini API calls the suite makes
and the model endpoint's chat calls, and records every call. The forwarder, running as another
user, adds the per-run relay token, so neither the key nor the token is readable by the model under
test. Failed relay, endpoint or container calls are harness errors: the scenarios they fall in are
reported apart and not scored. See docs/quality.md.
"""

import fcntl
import hashlib
import hmac
import http.client
import io
import json
import logging
import os
import re
import secrets
import shutil
import socketserver
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import zlib
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit

from ..core.config import BenchmarkConfig, normalize_endpoint
from ..core.models import ModelConfig, ModelFormat
from ..utils.log_manager import write_redacted_json
# The same constant the sha resolver in scaffold.py resolves against. A
# sha resolved against one remote and a checkout taken from another would
# pin a scorer this run never used.
from ..core.scaffold import GEMMACLAW_REMOTE
from .serving import ServingBenchmarkRunner

logger = logging.getLogger(__name__)

#: Minimum Node.js (major, minor) required by gemmaclaw: the quality image's build checks its node
#: against it (docker/quality.Dockerfile, MIN_NODE).
_MIN_NODE = (22, 19)

#: The live-frontier lane's defaults for the two models the QA suite calls besides the model under
#: test (extensions/qa-lab/src/providers/live-frontier/index.ts at the pinned commit): the
#: alternate model when --alt-model is not given, and the image-generation model the media
#: scenarios configure (ensureImageGenerationConfigured).
UPSTREAM_ALT_MODEL = "openai/gpt-5.4"
UPSTREAM_IMAGE_GENERATION_MODEL = "openai/gpt-image-1"
#: What gbench runs instead, on the Gemini API through GemmaClaw's google plugin: GemmaClaw's own
#: Google defaults at the pinned commit, GOOGLE_GEMINI_DEFAULT_MODEL (extensions/google/onboard.ts)
#: and DEFAULT_GOOGLE_IMAGE_MODEL (extensions/google/image-generation-provider.ts).
GEMINI_ALT_MODEL = "google/gemini-3.1-pro-preview"
GEMINI_IMAGE_GENERATION_MODEL = "google/gemini-3.1-flash-image-preview"
#: The google plugin's other capabilities the suite uses, next to the OpenAI ones the lane would
#: use with its openai plugin keyed: memory embeddings (the memorySearch provider is auto-selected
#: among the keyed ones, openai before gemini; DEFAULT_OPENAI_EMBEDDING_MODEL and
#: DEFAULT_GEMINI_EMBEDDING_MODEL), and image understanding for a model under test that declares no
#: image input (the image tool falls back to a keyed media provider, openai before google; each
#: plugin's default image model in openclaw.plugin.json).
UPSTREAM_EMBEDDING_MODEL = "openai/text-embedding-3-small"
GEMINI_EMBEDDING_MODEL = "google/gemini-embedding-001"
UPSTREAM_IMAGE_UNDERSTANDING_MODEL = "openai/gpt-5.4-mini"
GEMINI_IMAGE_UNDERSTANDING_MODEL = "google/gemini-3-flash-preview"

#: The alternate model as the google provider declares it (the pi-ai catalog entry of
#: gemini-3.1-pro-preview at the pinned commit), so the provider can carry the relay's base URL.
_GEMINI_ALT_MODEL_DEFINITION = {
    "id": "gemini-3.1-pro-preview", "name": "Gemini 3.1 Pro Preview",
    "api": "google-generative-ai", "reasoning": True, "input": ["text", "image"],
    "cost": {"input": 2, "output": 12, "cacheRead": 0.2, "cacheWrite": 0},
    "contextWindow": 1048576, "maxTokens": 65536,
}

#: The output cap GemmaClaw declares for a self-hosted (vLLM) model, SELF_HOSTED_DEFAULT_MAX_TOKENS
#: in src/agents/self-hosted-provider-defaults.ts at the pinned commit (DEFAULT_GEMMACLAW_COMMIT):
#: its vLLM provider setup declares every served model with it, and pi-ai sends max_tokens =
#: min(model maxTokens, 32000) on each request. gbench declares the model under test with the
#: value of the commit it runs (read from the checkout at setup; this one when it cannot be read).
GEMMACLAW_SELF_HOSTED_MAX_TOKENS = 8192
_SELF_HOSTED_DEFAULTS_FILE = "src/agents/self-hosted-provider-defaults.ts"
_SELF_HOSTED_MAX_TOKENS_RE = re.compile(r"SELF_HOSTED_DEFAULT_MAX_TOKENS\s*=\s*([0-9]+)")

#: The one upstream every Gemini call goes to. No variable changes it.
GEMINI_API_ORIGIN = "https://generativelanguage.googleapis.com"
#: The variable gbench's edit of extensions/google/api.ts reads: the relay's base URL.
GEMINI_RELAY_ENV = "GBENCH_GEMINI_RELAY_BASE_URL"
#: The API key gbench's own vLLM server is configured with (none): not a credential.
LOCAL_VLLM_KEY = "local-vllm"

#: The edits gbench makes to the GemmaClaw checkout, each (old, new) exactly once in its file.
#: 1. The QA lab's live-frontier provider hard-codes openai/gpt-image-1 as the image-generation
#: model (no CLI option or variable sets it), so its two image-generation fields name the google
#: plugin and the Gemini image model.
GEMINI_IMAGE_PATCH_FILE = "extensions/qa-lab/src/providers/live-frontier/index.ts"
_IMAGE_PATCH_OLD = (
    '  defaultImageGenerationProviderIds: ["openai"],\n'
    "  defaultImageGenerationModel: ({ modelProviderIds }) =>\n"
    '    modelProviderIds.includes("openai") ? "openai/gpt-image-1" : null,\n')
_IMAGE_PATCH_NEW = (
    "  // gbench: image generation on the Gemini API (google plugin), see gbench docs/quality.md.\n"
    '  defaultImageGenerationProviderIds: ["google"],\n'
    "  defaultImageGenerationModel: ({ modelProviderIds }) =>\n"
    f'    modelProviderIds.includes("google") ? "{GEMINI_IMAGE_GENERATION_MODEL}" : null,\n')
#: 2. The google plugin's HTTP helper (image generation, image understanding, speech) accepts
#: only https://generativelanguage.googleapis.com as its base URL and no private network, so it
#: also accepts exactly the relay's base URL (GEMINI_RELAY_ENV) and, for it, the loopback address.
GEMINI_RELAY_PATCH_FILE = "extensions/google/api.ts"
_RELAY_PATCH_OLD_1 = (
    "function resolveTrustedGoogleGenerativeAiBaseUrl(baseUrl?: string): string {\n"
    "  const normalized =\n"
    "    normalizeGoogleGenerativeAiBaseUrl(baseUrl ?? DEFAULT_GOOGLE_API_BASE_URL) ??\n"
    "    DEFAULT_GOOGLE_API_BASE_URL;\n")
_RELAY_PATCH_NEW_1 = (
    "// gbench: the Gemini API through gbench's local key relay, see gbench docs/quality.md.\n"
    "function isGbenchGeminiRelayBaseUrl(baseUrl?: string): boolean {\n"
    f"  const relay = process.env.{GEMINI_RELAY_ENV}?.trim();\n"
    "  return Boolean(\n"
    "    relay &&\n"
    "      baseUrl &&\n"
    "      normalizeGoogleGenerativeAiBaseUrl(baseUrl) ==="
    " normalizeGoogleGenerativeAiBaseUrl(relay),\n"
    "  );\n"
    "}\n"
    "\n"
    + _RELAY_PATCH_OLD_1 +
    "  if (isGbenchGeminiRelayBaseUrl(normalized)) {\n"
    "    return normalized;\n"
    "  }\n")
_RELAY_PATCH_OLD_2 = "    allowPrivateNetwork: false,\n"
_RELAY_PATCH_NEW_2 = "    allowPrivateNetwork: isGbenchGeminiRelayBaseUrl(params.baseUrl),\n"

_GEMMACLAW_PATCHES = (
    (GEMINI_IMAGE_PATCH_FILE, ((_IMAGE_PATCH_OLD, _IMAGE_PATCH_NEW),),
     f"image-generation model {UPSTREAM_IMAGE_GENERATION_MODEL} -> {GEMINI_IMAGE_GENERATION_MODEL} "
     "(defaultImageGenerationProviderIds ['openai'] -> ['google'])"),
    (GEMINI_RELAY_PATCH_FILE,
     ((_RELAY_PATCH_OLD_1, _RELAY_PATCH_NEW_1), (_RELAY_PATCH_OLD_2, _RELAY_PATCH_NEW_2)),
     f"the Gemini HTTP helper also accepts the gbench relay's base URL (${GEMINI_RELAY_ENV}) "
     "and, for it, the loopback address"),
)

#: Environment variables that carry a credential or setting of a model or service provider other
#: than the Gemini API, or any other credential: names ending in one of the suffixes, matching
#: _API_KEY_<n>, starting with one of the prefixes, or listed. None of gbench's environment reaches
#: the suite (its container gets a fixed set of variables); the names of these are recorded, and the
#: host-side git and docker subprocesses run without them and without the Gemini keys.
_KEPT_ENV = frozenset({"GEMINI_API_KEY", "GEMINI_API_KEYS", "VLLM_API_KEY"})
_PROVIDER_ENV_SUFFIXES = ("_API_KEY", "_API_KEYS", "_API_TOKEN", "_API_SECRET", "_OAUTH_TOKEN",
                          "_TOKEN", "_AUTHTOKEN", "_SECRET", "_PASSWORD", "_CREDENTIALS")
_PROVIDER_ENV_NUMBERED = re.compile(r"_API_KEY_[0-9]+$")
_PROVIDER_ENV_PREFIXES = ("OPENAI_", "ANTHROPIC_", "OPENROUTER_", "AZURE_OPENAI_", "AWS_",
                          "BEDROCK_", "CLAUDE_", "CODEX_", "OPENCLAW_LIVE_", "GOOGLE_", "VERTEX",
                          "MINIMAX_", "FAL_", "HUGGINGFACE_")
_PROVIDER_ENV_NAMES = frozenset({"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN",
                                 "COPILOT_GITHUB_TOKEN", "FAL_KEY"})


def _is_provider_env(name: str) -> bool:
    """Whether `name` is a variable of a provider other than the Gemini API (see above)."""
    up = name.upper()
    if up in _KEPT_ENV:
        return False
    return (up in _PROVIDER_ENV_NAMES or up.endswith(_PROVIDER_ENV_SUFFIXES)
            or bool(_PROVIDER_ENV_NUMBERED.search(up)) or up.startswith(_PROVIDER_ENV_PREFIXES))


def _gemini_keys(environ: Mapping[str, str]) -> List[str]:
    """The Gemini API keys of `environ`: GEMINI_API_KEYS when set, else GEMINI_API_KEY (each one
    key or a comma/semicolon list), de-duplicated, as base.gemini_api_keys reads them."""
    def _split(raw: Optional[str]) -> List[str]:
        return [k.strip() for k in re.split(r"[;,]", raw or "") if k.strip()]
    keys = _split(environ.get("GEMINI_API_KEYS")) or _split(environ.get("GEMINI_API_KEY"))
    return list(dict.fromkeys(keys))


def provider_free_env(environ: Mapping[str, str]) -> Tuple[Dict[str, str], List[str]]:
    """`environ` without any provider credential: no variable of a model or service provider
    other than the Gemini API, and no Gemini key either (the Gemini keys stay in this process,
    with the relay). Returns the environment and the sorted names of the non-Gemini variables
    removed (names only, never values)."""
    env: Dict[str, str] = {}
    removed: List[str] = []
    for name, value in environ.items():
        if name.upper() in ("GEMINI_API_KEY", "GEMINI_API_KEYS"):
            continue
        if _is_provider_env(name):
            removed.append(name)
        else:
            env[name] = value
    return env, sorted(removed)


#: Names of other keys (KAGGLE_KEY, SERPAPI_KEY, ...) the host-side commands do not get either.
_HOST_KEY_NAME = re.compile(r"_KEYS?$|_KEY_[0-9]+$")


def host_env(environ: Mapping[str, str]) -> Dict[str, str]:
    """The environment of gbench's host-side git and docker commands: provider_free_env(environ)
    without a variable named like a key (*_KEY, *_KEYS, *_KEY_<n>), and without any variable whose
    value holds a credential value, whatever its name: a Gemini key,
    the endpoint's key (base.endpoint_api_key: VLLM_API_KEY, else OPENAI_API_KEY), or any value
    base.secret_values finds (a credential variable's value, the password of a URL). Nothing of it
    reaches the suite's container."""
    from .eval_suites.base import endpoint_api_key, secret_values
    env, _removed = provider_free_env(environ)
    endpoint_key = endpoint_api_key(dict(environ)) or ""
    needles = [*_gemini_keys(environ),
               *([endpoint_key] if endpoint_key and endpoint_key != LOCAL_VLLM_KEY else []),
               *secret_values(dict(environ))]
    return {k: v for k, v in env.items()
            if not _HOST_KEY_NAME.search(k.upper()) and not any(n in (v or "") for n in needles)}


def sandbox_suite_env(environ: Mapping[str, str], *, home: str, npm_cache: str, config_path: str,
                      relay_base_url: str, model_api_key: str) -> Tuple[Dict[str, str], List[str]]:
    """The environment of the QA suite in its container: a fixed set of variables, none taken from
    `environ` (so no credential of gbench's environment, whatever its name or value, reaches it):
    HOME the container's fresh home, TMPDIR, LANG, npm's cache (`npm_cache`, used offline),
    OPENCLAW_CONFIG_PATH
    gbench's openclaw.json, GEMINI_RELAY_ENV the forwarder's Gemini base URL,
    OPENCLAW_ENABLE_PRIVATE_QA_CLI=1 and VLLM_API_KEY `model_api_key` (a placeholder: the relay
    adds the endpoint's own key). PATH is the image's. Returns it and the sorted names of the
    provider variables of `environ` (provider_free_env), recorded as left out."""
    env = {
        "HOME": home,
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "npm_config_cache": npm_cache,
        "npm_config_offline": "true",
        "npm_config_update_notifier": "false",
        "COREPACK_ENABLE_DOWNLOAD_PROMPT": "0",
        "OPENCLAW_CONFIG_PATH": config_path,
        GEMINI_RELAY_ENV: relay_base_url,
        "VLLM_API_KEY": model_api_key,
        "OPENCLAW_ENABLE_PRIVATE_QA_CLI": "1",
    }
    return env, provider_free_env(environ)[1]


def _patch_record(file: str, change: str, state: str, src: str) -> Dict[str, str]:
    return {"file": file, "change": change, "state": state,
            "sha256": hashlib.sha256(src.encode("utf-8")).hexdigest()}


def _apply_patch(repo: Path, file: str, edits: Sequence[Tuple[str, str]],
                 change: str) -> Dict[str, str]:
    """Apply `edits` to `file` in the checkout at `repo`, all or none (idempotent). Raises
    RuntimeError when an edit's block (upstream's, or gbench's) is not in the file exactly once,
    so a drifted checkout is never run half-patched. Returns the record stored on the result."""
    path = Path(repo) / file
    try:
        src = path.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"infra_required: cannot read {path} to make gbench's edit ({change}) "
                           f"({type(e).__name__})") from e
    out = src
    applied = False
    for old, new in edits:
        if out.count(new) == 1 and out.replace(new, "").count(old) == 0:
            continue
        if out.count(new) == 0 and out.count(old) == 1:
            out = out.replace(old, new, 1)
            applied = True
            continue
        raise RuntimeError(
            f"infra_required: {file} does not hold the block gbench replaces ({change}) exactly "
            "once; the checkout differs from the pinned GemmaClaw commit. See docs/quality.md")
    if applied:
        path.write_text(out, encoding="utf-8")
    return _patch_record(file, change, "applied" if applied else "already applied", out)


def _revert_patch(repo: Path, file: str, edits: Sequence[Tuple[str, str]]) -> None:
    """Undo _apply_patch (only gbench's own blocks are touched)."""
    path = Path(repo) / file
    try:
        src = path.read_text(encoding="utf-8")
    except OSError:
        return
    out = src
    for old, new in edits:
        if out.count(new) == 1:
            out = out.replace(new, old, 1)
    if out != src:
        path.write_text(out, encoding="utf-8")


def apply_gemini_image_patch(repo: Path) -> Dict[str, str]:
    """Make the QA lab's image-generation model GEMINI_IMAGE_GENERATION_MODEL in the checkout at
    `repo` (idempotent; raises RuntimeError on a drifted file). Returns the record stored on the
    result."""
    file, edits, change = _GEMMACLAW_PATCHES[0]
    return _apply_patch(repo, file, edits, change)


def revert_gemini_image_patch(repo: Path) -> None:
    """Undo apply_gemini_image_patch (only gbench's own block is touched)."""
    file, edits, _change = _GEMMACLAW_PATCHES[0]
    _revert_patch(repo, file, edits)


def apply_gemmaclaw_patches(repo: Path) -> List[Dict[str, str]]:
    """Make every gbench edit of the checkout at `repo` (_GEMMACLAW_PATCHES); one record each."""
    return [_apply_patch(repo, file, edits, change) for file, edits, change in _GEMMACLAW_PATCHES]


def revert_gemmaclaw_patches(repo: Path) -> None:
    """Undo apply_gemmaclaw_patches (only gbench's own blocks are touched)."""
    for file, edits, _change in _GEMMACLAW_PATCHES:
        _revert_patch(repo, file, edits)


# --------------------------------------------------------------------------------------------- #
# The container: where the QA suite runs, and the paths gbench and the relay use in it
# --------------------------------------------------------------------------------------------- #
#: The Dockerfile and its two helpers (repository root docker/), and the image name.
_DOCKER_DIR = Path(__file__).resolve().parents[2] / "docker"
QUALITY_DOCKERFILE = "quality.Dockerfile"
_IMAGE_HELPERS = ("quality_entrypoint.sh", "quality_relay_forwarder.py", "quality_npm_cache.mjs")
QUALITY_IMAGE_REPO = "gbench-quality"
#: Kept in step with docker/quality.Dockerfile, quality_entrypoint.sh and
#: quality_relay_forwarder.py.
SUITE_UID = 10001
FORWARDER_PORT = 18431
CONTAINER_REPO = "/opt/gemmaclaw"
CONTAINER_OUTPUT_DIRNAME = "gbench_qa_out"
CONTAINER_OUTPUT_DIR = f"{CONTAINER_REPO}/{CONTAINER_OUTPUT_DIRNAME}"
#: Where the suite stages each gateway's bundled-plugin tree (<repo>/.artifacts/qa-runtime/...,
#: extensions/qa-lab/src/bundled-plugin-staging.ts): writable, the rest of the checkout is not.
CONTAINER_ARTIFACTS_DIR = f"{CONTAINER_REPO}/.artifacts"
CONTAINER_STATUS_DIR = "/gbench-status"
CONTAINER_HOME = "/home/qa"
CONTAINER_CONFIG_PATH = "/etc/gbench/openclaw.json"
CONTAINER_RELAY_DIR = "/run/gbench-relay"
RELAY_SOCKET_NAME = "relay.sock"
CONTAINER_IMAGE_DATA_DIR = "/opt/gbench"
#: The npm cache holding every bundled plugin's runtime dependencies (docker/quality_npm_cache.mjs):
#: GemmaClaw installs them from it offline, into a per-container copy.
CONTAINER_NPM_CACHE = f"{CONTAINER_IMAGE_DATA_DIR}/npm-cache"
#: The base URLs the suite is given: the forwarder inside the container.
CONTAINER_GEMINI_BASE_URL = f"http://127.0.0.1:{FORWARDER_PORT}/v1beta"
CONTAINER_MODEL_BASE_URL = f"http://127.0.0.1:{FORWARDER_PORT}/model"
#: What the suite's configuration and VLLM_API_KEY carry where a key would go: not a credential
#: (the forwarder adds the relay token, the relay adds the real keys and drops this).
KEY_PLACEHOLDER = "gbench-relay"
#: The header the forwarder adds, carrying the per-run relay token.
RELAY_TOKEN_HEADER = "x-gbench-relay-token"
#: The flags of every `git` gbench runs on the host: no hook and no fsmonitor of the checkout runs.
_SAFE_GIT = ("-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false")
#: How long a suite that has printed its summary line may take to exit before gbench stops its
#: container (a suite whose isolated worker failed can leave servers open and never exit).
SUITE_EXIT_GRACE_S = 120.0
#: The line the suite prints once its summary is written (extensions/qa-lab/src/cli.runtime.ts).
_SUITE_SUMMARY_LINE = "QA suite summary:"
#: The step the suite reports for a scenario whose isolated worker failed before or around its
#: flow (the lab, the gateway, the transport): the harness failed, not the model under test.
_SUITE_WORKER_STEP = "isolated scenario worker"
#: The entrypoint's exit statuses for a container that failed before the suite ran.
_ENTRYPOINT_FAILURES = {94: "no /opt/gemmaclaw in the image", 95: "no command to run",
                        96: "the entrypoint did not start as root",
                        97: "no relay token on standard input",
                        98: "the relay forwarder did not start",
                        125: "docker run failed", 126: "the entrypoint could not be run",
                        127: "the entrypoint was not found"}


def quality_build_digest() -> str:
    """sha256 over the Dockerfile, its two helpers and gbench's edits of the checkout."""
    h = hashlib.sha256()
    for name in (QUALITY_DOCKERFILE, *_IMAGE_HELPERS):
        h.update(name.encode("utf-8") + b"\0" + (_DOCKER_DIR / name).read_bytes() + b"\0")
    h.update(json.dumps([[f, [list(e) for e in edits], change]
                         for f, edits, change in _GEMMACLAW_PATCHES]).encode("utf-8"))
    return h.hexdigest()


def quality_image_tag(commit: str) -> str:
    """gbench-quality:<commit[:12]>-<quality_build_digest()[:12]>."""
    return f"{QUALITY_IMAGE_REPO}:{commit[:12]}-{quality_build_digest()[:12]}"


def find_key_exposure(secret_values: Sequence[str], roots: Sequence[str],
                      files: Sequence[str] = ()) -> Optional[str]:
    """The first file under `roots` (walked, symlinks not followed) or in `files` that holds one of
    `secret_values`, or None. Unreadable files are skipped. Only the path is returned, never the
    value."""
    needles = [s.encode("utf-8") for s in secret_values if s]
    if not needles:
        return None

    def _holds(path: str) -> bool:
        try:
            st = os.lstat(path)
            if not stat.S_ISREG(st.st_mode) or st.st_size > (64 << 20):
                return False
            data = Path(path).read_bytes()
        except OSError:
            return False
        return any(n in data for n in needles)

    for path in files:
        if _holds(path):
            return path
    for root in roots:
        for dirpath, _dirs, names in os.walk(root, followlinks=False):
            for name in names:
                path = os.path.join(dirpath, name)
                if _holds(path):
                    return path
    return None


def assemble_build_context(repo: Path, commit: str, ctx: Path, *, git_env: Mapping[str, str],
                           secret_values: Sequence[str]) -> List[Dict[str, str]]:
    """Fill `ctx` with the quality image's build context: src/ is `git archive <commit>` of the
    checkout at `repo` (the commit's tree only: no working-tree change, no .git), with gbench's
    edits applied; gemmaclaw_patches.json records them; the Dockerfile and its helpers are copied.
    The checkout itself is only read. Raises RuntimeError (infra_required) when a file of the
    commit holds one of `secret_values` (the image would show it to the model under test) or an
    edit does not apply. Returns the edit records."""
    src = ctx / "src"
    src.mkdir(parents=True)
    archive = ctx / "src.tar"
    subprocess.run(["git", *_SAFE_GIT, "archive", "--format=tar", "-o", str(archive), commit],
                   cwd=str(repo), check=True, env=dict(git_env))
    with tarfile.open(archive) as tf:
        tf.extractall(src, filter="data")
    archive.unlink()
    exposed = find_key_exposure(secret_values, roots=[str(src)])
    if exposed:
        raise RuntimeError(
            f"infra_required: {os.path.relpath(exposed, src)} at GemmaClaw commit {commit[:12]} "
            "holds a Gemini key or the endpoint's key, and the quality image would show it to the "
            "model under test; remove it from that commit. See docs/quality.md")
    records = apply_gemmaclaw_patches(src)
    with open(ctx / "gemmaclaw_patches.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    for name in (QUALITY_DOCKERFILE, *_IMAGE_HELPERS):
        shutil.copy2(_DOCKER_DIR / name, ctx / name)
    return records


def _docker_image_commit(tag: str, env: Mapping[str, str]) -> Optional[str]:
    """The GemmaClaw commit the local image `tag` was built from (its label), or None."""
    res = subprocess.run(
        ["docker", "image", "inspect", "--format",
         '{{ index .Config.Labels "gbench.quality.gemmaclaw_commit" }}', tag],
        capture_output=True, text=True, env=dict(env))
    return res.stdout.strip() if res.returncode == 0 and res.stdout.strip() else None


def ensure_quality_image(repo: Path, commit: str, *, git_env: Mapping[str, str],
                         docker_env: Mapping[str, str], secret_values: Sequence[str]) -> str:
    """The local quality image for `commit`, built from the checkout at `repo` when there is none
    (docker build of a context from assemble_build_context; the build output goes to the
    console). Returns its tag. Raises RuntimeError (infra_required) when the build fails."""
    tag = quality_image_tag(commit)
    if _docker_image_commit(tag, docker_env) == commit:
        logger.info(f"Quality image {tag} is built")
        return tag
    logger.info(f"Building the quality image {tag} (GemmaClaw {commit[:12]}; the first build "
                "installs and builds GemmaClaw and takes several minutes)")
    with tempfile.TemporaryDirectory(prefix="gbench_quality_build_") as ctx:
        assemble_build_context(repo, commit, Path(ctx), git_env=git_env,
                               secret_values=secret_values)
        rc = subprocess.run(
            ["docker", "build", "-t", tag, "--build-arg", f"GEMMACLAW_COMMIT={commit}",
             "--build-arg", f"MIN_NODE={_MIN_NODE[0]}.{_MIN_NODE[1]}",
             "-f", str(Path(ctx) / QUALITY_DOCKERFILE), ctx],
            env=dict(docker_env)).returncode
    if rc != 0:
        raise RuntimeError(f"infra_required: building the quality image {tag} failed (docker "
                           f"build exit {rc}); see the build output above and docs/quality.md")
    if _docker_image_commit(tag, docker_env) != commit:
        raise RuntimeError(f"infra_required: the quality image {tag} does not carry GemmaClaw "
                           f"commit {commit}")
    return tag


def read_container_files(container: str, path: str, wanted: Callable[[str], bool], *,
                         env: Mapping[str, str],
                         max_file_bytes: int = 64 << 20) -> Dict[str, bytes]:
    """The regular files under `path` in the (stopped) container whose tar member name `wanted`
    accepts, read from the `docker cp` stream in memory: nothing of the container is written to
    the host, links and special files are skipped, and a file over `max_file_bytes` is left out.
    Raises RuntimeError when docker cp fails."""
    proc = subprocess.Popen(["docker", "cp", f"{container}:{path}", "-"], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, env=dict(env))
    out: Dict[str, bytes] = {}
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tf:
            for member in tf:
                name = member.name
                while name.startswith("./"):
                    name = name[2:]
                name = name.lstrip("/")
                if not member.isreg() or member.size > max_file_bytes or not wanted(name):
                    continue
                handle = tf.extractfile(member)
                if handle is not None:
                    out[name] = handle.read()
    except tarfile.TarError as e:
        raise RuntimeError(f"cannot read {path} from the quality container ({e})") from e
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr is not None else ""
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"cannot read {path} from the quality container (docker cp exit {rc}: "
                           f"{err.strip().splitlines()[-1] if err.strip() else 'no message'})")
    return out


#: The docker network of the quality container: none, only a loopback interface. The relay's unix
#: socket is its one way out, so the cloud metadata server guard needs no probe for it
#: (container_guard.require_metadata_blocked, called with the network of the run's command line).
QUALITY_CONTAINER_NETWORK = "none"


def argv_network(argv: Sequence[str]) -> Optional[str]:
    """The --network value of a `docker run` command line, or None when it sets none (docker's
    default bridge)."""
    for i, arg in enumerate(argv):
        if arg == "--network" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--network="):
            return arg.split("=", 1)[1]
    return None


def container_run_argv(image: str, name: str, *, relay_dir: str, config_file: str,
                       env: Mapping[str, str], command: Sequence[str]) -> List[str]:
    """The `docker run` command line of one quality run: no network (only loopback), a read-only
    root filesystem, tmpfs /tmp and home, anonymous volumes for the suite output, the checkout's
    .artifacts, gbench's status files and the npm cache (removed with the container), the relay's socket directory and
    gbench's openclaw.json read-only, every capability dropped but SETUID and SETGID (for the
    entrypoint to start the forwarder and the suite as their users), no-new-privileges, `env`
    and nothing else of the host environment. The relay token goes on standard input
    (--interactive), never on this line."""
    argv = ["docker", "run", "--name", name, "--interactive", "--init",
            "--network", QUALITY_CONTAINER_NETWORK, "--dns-search", ".",
            "--read-only",
            "--tmpfs", "/tmp:rw,nosuid,nodev,mode=1777",
            "--tmpfs",
            f"{CONTAINER_HOME}:rw,nosuid,nodev,uid={SUITE_UID},gid={SUITE_UID},mode=0700",
            "--shm-size", "1g",
            "--volume", CONTAINER_OUTPUT_DIR,
            "--volume", CONTAINER_ARTIFACTS_DIR,
            "--volume", CONTAINER_STATUS_DIR,
            "--volume", CONTAINER_NPM_CACHE,
            "--volume", f"{relay_dir}:{CONTAINER_RELAY_DIR}:ro",
            "--volume", f"{config_file}:{CONTAINER_CONFIG_PATH}:ro",
            "--cap-drop", "ALL", "--cap-add", "SETUID", "--cap-add", "SETGID",
            "--security-opt", "no-new-privileges",
            "--pids-limit", "8192",
            "--label", "gbench.quality.run=1"]
    for key, value in sorted(env.items()):
        argv += ["--env", f"{key}={value}"]
    return argv + [image, *command]


# --------------------------------------------------------------------------------------------- #
# The key relay
# --------------------------------------------------------------------------------------------- #
#: Request headers not forwarded (hop-by-hop, or recomputed) and the credential headers the relay
#: drops (the placeholder the suite sends, the relay token) before adding the real one.
_HOP_BY_HOP = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                         "proxy-connection", "te", "trailer", "trailers", "transfer-encoding",
                         "upgrade", "host", "content-length"})
_CREDENTIAL_HEADERS = frozenset({"authorization", "x-goog-api-key", RELAY_TOKEN_HEADER})
#: The Gemini API calls the QA suite makes, the only ones the relay forwards:
#: POST /v1beta/models/<model>:<method> with one of these models and methods, the query reduced to
#: `alt`, and only these request headers.
ALLOWED_GEMINI_MODELS = frozenset(m.split("/", 1)[1] for m in (
    GEMINI_ALT_MODEL, GEMINI_IMAGE_GENERATION_MODEL, GEMINI_EMBEDDING_MODEL,
    GEMINI_IMAGE_UNDERSTANDING_MODEL))
ALLOWED_GEMINI_METHODS = frozenset({"generateContent", "streamGenerateContent", "embedContent",
                                    "batchEmbedContents"})
_GEMINI_REQUEST_HEADERS = frozenset({"content-type", "accept", "user-agent", "x-goog-api-client"})
#: The model-under-test endpoint calls the relay forwards (the vllm provider's openai-completions
#: transport), below the endpoint's base URL.
ALLOWED_MODEL_CALLS = frozenset({("POST", "/chat/completions"), ("POST", "/completions"),
                                 ("GET", "/models")})
#: Statuses the Gemini route tries the next key on (every Gemini answer of 400 or more is a harness
#: error).
_ROTATE_STATUSES = (401, 403, 429)
#: Model-endpoint answers that are harness errors whatever the request: authentication, a timeout,
#: a rate limit (and every 5xx).
_MODEL_HARNESS_ERROR_STATUSES = (401, 403, 408, 429)
#: Model-endpoint answers to the request's content (an over-long or invalid request: 400 on vLLM,
#: 422 on servers that answer a validation error with it): the model's side, never harness errors.
_MODEL_CONTENT_REJECTION_STATUSES = (400, 422)


def model_status_is_harness_error(status: int, names_served_model: bool) -> bool:
    """Whether a model-endpoint answer is a harness error: a 5xx or one of
    _MODEL_HARNESS_ERROR_STATUSES, whatever the request; any other 4xx apart from
    _MODEL_CONTENT_REJECTION_STATUSES (404 from an endpoint that does not serve the configured
    model name or has no such path, 405, 413, 415, ...) when the request is one the suite makes
    (`names_served_model`: GET /models, or a body naming the model gbench configured). The same
    4xx on a request that names another model, or whose body is not such a request, is the
    model's side: only the model's own tools, through the forwarder, send one."""
    if status >= 500 or status in _MODEL_HARNESS_ERROR_STATUSES:
        return True
    return (400 <= status < 500 and status not in _MODEL_CONTENT_REJECTION_STATUSES
            and names_served_model)


#: Gemini finish reasons that mean the service gave no answer.
_NO_ANSWER_FINISH = frozenset({
    "SAFETY", "RECITATION", "OTHER", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "LANGUAGE",
    "IMAGE_SAFETY", "IMAGE_PROHIBITED_CONTENT", "IMAGE_OTHER", "IMAGE_RECITATION", "NO_IMAGE",
    "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS"})
#: Gemini replies above this size are streamed back without being inspected.
_INSPECT_CAP = 64 << 20
#: The largest request body the relay reads (the suite's largest, a chat turn with images, stays
#: far below it; GemmaClaw's own chat endpoint caps a body at 20 MB), and how many requests it
#: handles at once (more wait): what the container can make the relay hold in gbench's process.
_MAX_REQUEST_BODY = 64 << 20
_MAX_CONCURRENT_REQUESTS = 128


def gemini_reply_outcome(capability: str, method: str, data: bytes) -> str:
    """"answered", "non_answer" or "truncated" for a Gemini API reply of HTTP status 200: a reply
    that does not parse, carries a block reason, a no-answer finish reason, no answer part (text,
    function call, inline data; thoughts do not count; an image for image generation), or no
    embedding, is a non-answer; MAX_TOKENS, or a stream that ends without a finish reason, is
    truncated."""
    text = data.decode("utf-8", "replace").strip()
    objs: List[Any] = []
    try:
        if text.startswith("data:") or "\ndata:" in text:
            for line in text.splitlines():
                if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                    objs.append(json.loads(line[5:].strip()))
        elif text:
            parsed = json.loads(text)
            objs = parsed if isinstance(parsed, list) else [parsed]
    except ValueError:
        return "non_answer"
    objs = [o for o in objs if isinstance(o, dict)]
    if not objs:
        return "non_answer"
    if method == "embedContent":
        return "answered" if (objs[0].get("embedding") or {}).get("values") else "non_answer"
    if method == "batchEmbedContents":
        embs = objs[0].get("embeddings")
        return ("answered" if isinstance(embs, list) and embs and all(
            isinstance(e, dict) and e.get("values") for e in embs) else "non_answer")
    answered = blocked = False
    finish: Optional[str] = None
    for obj in objs:
        if (obj.get("promptFeedback") or {}).get("blockReason"):
            blocked = True
        for cand in obj.get("candidates") or []:
            if not isinstance(cand, dict):
                continue
            for part in (cand.get("content") or {}).get("parts") or []:
                if not isinstance(part, dict) or part.get("thought"):
                    continue
                if capability == "image generation":
                    answered = answered or bool(part.get("inlineData") or part.get("inline_data"))
                elif ((isinstance(part.get("text"), str) and part["text"].strip())
                        or part.get("functionCall") or part.get("inlineData")
                        or part.get("executableCode") or part.get("fileData")):
                    answered = True
            if cand.get("finishReason"):
                finish = str(cand["finishReason"])
    if blocked or finish in _NO_ANSWER_FINISH or not answered:
        return "non_answer"
    if finish is None or finish == "MAX_TOKENS":
        return "truncated"
    return "answered"


class _UpstreamBroke(Exception):
    """The upstream failed while its response was being streamed back."""


class _RelayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "gbench-relay"

    def log_message(self, format, *args):     # noqa: A002 - no request lines on stderr
        return

    def _dispatch(self):
        self.server.relay._handle(self)

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _dispatch


class _UnixRelayServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    request_queue_size = 128


def _path_shape(path: str) -> str:
    """A request path as recorded for a refused request: printable, at most 160 characters."""
    return re.sub(r"[^\x21-\x7e]", "?", path)[:160]


class KeyRelay:
    """The relay that holds every credential of a quality run, on a unix socket.

    The container's forwarder hands it every request with the per-run `token` in
    RELAY_TOKEN_HEADER; a request without it is answered 401 and counted in `rejected`. Requests
    under /model go to the model-under-test endpoint (`model_upstream`), with its key as a Bearer
    token when it has one; only ALLOWED_MODEL_CALLS are forwarded. Requests under /v1beta go to
    GEMINI_API_ORIGIN with the real Gemini key in x-goog-api-key, the keys taken in turn and the
    next one tried on 401/403/429; only the Gemini calls the suite makes (ALLOWED_GEMINI_MODELS x
    ALLOWED_GEMINI_METHODS, POST) are forwarded. Anything else is answered 403 and counted as
    refused (not a harness error). Credentials a request carries are dropped. Responses stream back
    as they arrive. Every forwarded call is counted by route, capability, model and method with its
    status (`report`). Harness errors, each with its time: an unreachable upstream, a response
    that breaks off, a Gemini answer of 400 or more, a Gemini reply of 200 that is a non-answer or
    truncated (gemini_reply_outcome), a model-endpoint call the suite makes whose body is over
    _MAX_REQUEST_BODY (answered 413 by the relay, relay_body_too_large; its body cannot be shown
    to name another model; any other oversized request is refused), and a model-endpoint answer
    that model_status_is_harness_error names: 401, 403, 408, 429 or 5xx, and any other 4xx but
    400 and 422 on a request that names the served model `model_id` (every request when it is
    not given) or is GET /models."""

    GEMINI_PREFIX = "/v1beta"
    MODEL_PREFIX = "/model"

    def __init__(self, gemini_keys: Sequence[str], *, model_upstream: Optional[str] = None,
                 model_key: Optional[str] = None, model_id: Optional[str] = None,
                 gemini_timeout_s: float = 600.0, model_timeout_s: float = 3600.0):
        if not gemini_keys:
            raise ValueError("the relay needs at least one Gemini API key")
        self.token = "gbench-relay-" + secrets.token_hex(24)
        self._keys = list(gemini_keys)
        self._gemini = urlsplit(GEMINI_API_ORIGIN)
        self._model = urlsplit(model_upstream.rstrip("/")) if model_upstream else None
        self._model_key = model_key
        self._model_id = model_id
        self._timeouts = {"gemini": gemini_timeout_s, "model": model_timeout_s}
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._slots = threading.BoundedSemaphore(_MAX_CONCURRENT_REQUESTS)
        self._inflight = 0
        self._next_key = 0
        self._calls: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        self._errors: Dict[str, int] = {}
        self._events: List[Tuple[float, str]] = []
        self._refused: Dict[Tuple[str, str, str], List[float]] = {}
        self._rejected = 0
        self._server: Optional[_UnixRelayServer] = None
        self._thread: Optional[threading.Thread] = None
        self.socket_path: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------------------------- #
    def start(self, socket_path: str) -> "KeyRelay":
        """Listen on the unix socket `socket_path` (mode 0666: the forwarder's user differs from
        gbench's; a request without the token is refused)."""
        server = _UnixRelayServer(socket_path, _RelayHandler)
        server.relay = self
        os.chmod(socket_path, 0o666)
        self._server = server
        self.socket_path = socket_path
        self._thread = threading.Thread(target=server.serve_forever, name="gbench-key-relay",
                                        daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass

    # -- accounting --------------------------------------------------------------------------- #
    def _count_error(self, kind: str, when: float) -> None:
        """Count a harness error; `when` is the time its request arrived."""
        with self._lock:
            self._errors[kind] = self._errors.get(kind, 0) + 1
            self._events.append((when, kind))

    def _record(self, label: Tuple[str, str, str, str], status: str,
                outcome: Optional[str] = None) -> None:
        with self._lock:
            entry = self._calls.setdefault(label, {"requests": 0, "statuses": {}})
            entry["requests"] += 1
            entry["statuses"][status] = entry["statuses"].get(status, 0) + 1
            if outcome in ("non_answer", "truncated"):
                key = "non_answers" if outcome == "non_answer" else "truncated"
                entry[key] = entry.get(key, 0) + 1

    def _refuse(self, h: BaseHTTPRequestHandler, route: str, path: str, why: str,
                when: float) -> None:
        with self._lock:
            self._refused.setdefault((route, h.command, _path_shape(path)), []).append(when)
        self._send_json(h, 403, f"gbench relay: {why}")

    def report(self, settle_s: float = 30.0) -> Dict[str, Any]:
        """{"gemini_calls", "model_endpoint_calls", "harness_errors" {kind: n},
        "harness_error_events" [(time, kind)], "refused" [{route, method, path, requests,
        times}], "rejected" n}, once no request is being handled (a call is recorded just after
        its response is sent; waits at most `settle_s`)."""
        with self._lock:
            self._idle.wait_for(lambda: self._inflight == 0, timeout=settle_s)
            rows = [{"route": route, "capability": cap, "model": model, "method": method,
                     "requests": v["requests"], "statuses": dict(sorted(v["statuses"].items())),
                     **{k: v[k] for k in ("non_answers", "truncated") if k in v}}
                    for (route, cap, model, method), v in sorted(self._calls.items())]
            errors = dict(sorted(self._errors.items()))
            events = list(self._events)
            refused = [{"route": route, "method": method, "path": path, "requests": len(times),
                        "times": list(times)}
                       for (route, method, path), times in sorted(self._refused.items())]
            rejected = self._rejected
        strip = lambda r: {k: v for k, v in r.items() if k != "route"}   # noqa: E731
        return {"gemini_calls": [strip(r) for r in rows if r["route"] == "gemini"],
                "model_endpoint_calls": [strip(r) for r in rows if r["route"] == "model"],
                "harness_errors": errors, "harness_error_events": events,
                "refused": refused, "rejected": rejected}

    @staticmethod
    def classify(path: str, body: bytes) -> Tuple[str, str, str]:
        """(capability, model, method) of a Gemini API request."""
        m = re.match(r"^/(?:upload/)?v1beta/(?:models|tunedModels)/([^:/]+):([A-Za-z]+)$", path)
        model, method = (m.group(1), m.group(2)) if m else ("", path)
        payload: Any = None
        if body[:1] in (b"{", b"["):
            try:
                payload = json.loads(body)
            except ValueError:
                payload = None
        text = json.dumps(payload) if payload is not None else ""
        if method in ("embedContent", "batchEmbedContents"):
            cap = "memory embeddings"
        elif f"google/{model}" == GEMINI_ALT_MODEL:
            cap = "alternate model"
        elif f"google/{model}" == GEMINI_IMAGE_GENERATION_MODEL:
            cap = "image generation"
        elif method == "predictLongRunning" or model.startswith("veo") or \
                path.startswith("/v1beta/operations"):
            cap = "video generation"
        elif model.startswith("lyria"):
            cap = "music generation"
        elif "tts" in model:
            cap = "speech"
        elif '"google_search"' in text or '"googleSearch"' in text:
            cap = "web search (Google Search grounding)"
        elif re.search(r'"mime_?[tT]ype": "(image|video|audio)/', text):
            cap = "media understanding"
        elif method in ("generateContent", "streamGenerateContent"):
            cap = "text generation"
        else:
            cap = "other"
        return cap, model, method

    # -- request handling --------------------------------------------------------------------- #
    def _names_served_model(self, method: str, sub: str, body: bytes) -> bool:
        """Whether a model-endpoint request is one the suite makes: GET /models, or a JSON body
        whose "model" is the served model `model_id` (every request when the relay was not given
        one). The suite's chat calls name it (GemmaClaw's openai-completions transport sends the
        configured model's id)."""
        if self._model_id is None or (method, sub) == ("GET", "/models"):
            return True
        payload = _json_or_none(body)
        return isinstance(payload, dict) and payload.get("model") == self._model_id

    def _rotation(self) -> List[str]:
        with self._lock:
            start = self._next_key % len(self._keys)
            self._next_key += 1
        return self._keys[start:] + self._keys[:start]

    @staticmethod
    def _read_body(h: BaseHTTPRequestHandler) -> Optional[bytes]:
        """The request body, or None when it is over _MAX_REQUEST_BODY (not read further)."""
        if "chunked" in (h.headers.get("Transfer-Encoding") or "").lower():
            chunks, total = [], 0
            while True:
                size = int(h.rfile.readline().split(b";", 1)[0].strip() or b"0", 16)
                if size == 0:
                    while h.rfile.readline() not in (b"\r\n", b"\n", b""):
                        pass
                    break
                total += size
                if total > _MAX_REQUEST_BODY:
                    return None
                chunks.append(h.rfile.read(size))
                h.rfile.readline()
            return b"".join(chunks)
        length = int(h.headers.get("Content-Length") or 0)
        if length > _MAX_REQUEST_BODY:
            return None
        return h.rfile.read(length) if length > 0 else b""

    @staticmethod
    def _send_json(h: BaseHTTPRequestHandler, status: int, message: str) -> None:
        data = json.dumps({"error": {"code": status, "message": message}}).encode("utf-8")
        h.send_response_only(status)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        if h.command != "HEAD":
            h.wfile.write(data)
        h.wfile.flush()

    def _handle(self, h: BaseHTTPRequestHandler) -> None:
        with self._lock:
            self._inflight += 1
        try:
            with self._slots:
                self._handle_request(h)
        finally:
            with self._lock:
                self._inflight -= 1
                self._idle.notify_all()

    def _handle_request(self, h: BaseHTTPRequestHandler) -> None:
        arrived = time.time()
        split = urlsplit(h.path)
        path = split.path or "/"
        query = [(k, v) for k, v in parse_qsl(split.query, keep_blank_values=True) if k != "key"]
        route = ("model" if path == self.MODEL_PREFIX or path.startswith(self.MODEL_PREFIX + "/")
                 else "gemini")
        try:
            body = self._read_body(h)
        except (OSError, ValueError):
            h.close_connection = True
            return
        if body is None:
            h.close_connection = True
            sub = path[len(self.MODEL_PREFIX):] or "/"
            presented = (h.headers.get(RELAY_TOKEN_HEADER) or "").strip()
            if (route == "model" and self._model is not None
                    and (h.command, sub) in ALLOWED_MODEL_CALLS
                    and hmac.compare_digest(presented.encode("utf-8"),
                                            self.token.encode("utf-8"))):
                # A model-endpoint call the suite makes, which the relay did not read in full: its
                # body cannot be shown to name another model, so the relay's own limit is a
                # harness error, as the endpoint's 413 on the same request would be.
                self._record((route, "model under test", "", sub), "413 (relay)")
                self._count_error("relay_body_too_large", arrived)
            else:
                with self._lock:
                    self._refused.setdefault((route, h.command, _path_shape(path)),
                                             []).append(arrived)
            self._send_json(h, 413, f"gbench relay: request body over {_MAX_REQUEST_BODY} bytes")
            return
        presented = (h.headers.get(RELAY_TOKEN_HEADER) or "").strip()
        if not hmac.compare_digest(presented.encode("utf-8"), self.token.encode("utf-8")):
            with self._lock:
                self._rejected += 1
            self._send_json(h, 401, "gbench relay: missing or wrong relay token")
            return
        if route == "model":
            sub = path[len(self.MODEL_PREFIX):] or "/"
            if self._model is None or (h.command, sub) not in ALLOWED_MODEL_CALLS:
                self._refuse(h, route, sub, f"{h.command} {_path_shape(sub)} is not a model "
                                            "endpoint call the QA suite makes", arrived)
                return
            upstream = self._model
            up_path = upstream.path.rstrip("/") + sub
            attempts = [{"Authorization": f"Bearer {self._model_key}"}] if self._model_key else [{}]
            label = (route, "model under test", "", sub)
            forwarded = {k: v for k, v in h.headers.items()
                         if k.lower() not in _HOP_BY_HOP and k.lower() not in _CREDENTIAL_HEADERS}
        else:
            up_path = re.sub(r"^/v1beta/v1beta(?=/|$)", "/v1beta", path)
            m = re.fullmatch(r"/v1beta/models/([^:/]+):([A-Za-z]+)", up_path)
            if (h.command != "POST" or not m or m.group(1) not in ALLOWED_GEMINI_MODELS
                    or m.group(2) not in ALLOWED_GEMINI_METHODS):
                self._refuse(h, route, up_path, f"{h.command} {_path_shape(up_path)} is not a "
                                                "Gemini API call the QA suite makes", arrived)
                return
            upstream = self._gemini
            attempts = [{"x-goog-api-key": key} for key in self._rotation()]
            label = (route, *self.classify(up_path, body))
            forwarded = {k: v for k, v in h.headers.items()
                         if k.lower() in _GEMINI_REQUEST_HEADERS}
            query = [(k, v) for k, v in query if k == "alt"]
        target = up_path + ("?" + urlencode(query) if query else "")
        conn = resp = None
        for i, extra in enumerate(attempts):
            cls = (http.client.HTTPSConnection if upstream.scheme == "https"
                   else http.client.HTTPConnection)
            conn = cls(upstream.hostname, upstream.port, timeout=self._timeouts[route])
            headers = dict(forwarded, **extra)
            if body or h.command in ("POST", "PUT", "PATCH"):
                headers["Content-Length"] = str(len(body))
            try:
                conn.request(h.command, target, body=body or None, headers=headers)
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException) as e:
                conn.close()
                self._record(label, "unreachable")
                self._count_error(f"{route}_unreachable", arrived)
                self._send_json(h, 502, f"gbench relay: {upstream.hostname} unreachable "
                                        f"({type(e).__name__})")
                return
            if route == "gemini" and resp.status in _ROTATE_STATUSES and i + 1 < len(attempts):
                resp.read()
                conn.close()
                continue
            break
        inspect = route == "gemini" and resp.status == 200
        # (the body is parsed for its model name only for a 4xx, the one case that needs it)
        status_error = (resp.status >= 400 if route == "gemini" else
                        model_status_is_harness_error(
                            resp.status, 400 <= resp.status < 500
                            and self._names_served_model(h.command, sub, body)))
        try:
            data = self._stream_back(h, resp, capture=inspect)
        except _UpstreamBroke:
            self._record(label, str(resp.status))
            self._count_error(f"{route}_stream_broken", arrived)
            h.close_connection = True
            conn.close()
            return
        except OSError:                          # the client went away
            self._record(label, str(resp.status))
            h.close_connection = True
            conn.close()
            return
        conn.close()
        outcome = None
        if inspect and data is not None:
            encoding = (resp.getheader("Content-Encoding") or "identity").lower()
            if encoding in ("gzip", "deflate"):
                try:
                    data = zlib.decompress(data, 47)
                except zlib.error:
                    data = None
            elif encoding != "identity":
                data = None
            if data is not None:
                outcome = gemini_reply_outcome(label[1], label[3], data)
        self._record(label, str(resp.status), outcome)
        if status_error:
            self._count_error(f"{route}_http_{resp.status}", arrived)
        elif outcome == "non_answer":
            self._count_error("gemini_non_answer", arrived)
        elif outcome == "truncated":
            self._count_error("gemini_truncated", arrived)

    @staticmethod
    def _stream_back(h: BaseHTTPRequestHandler, resp: http.client.HTTPResponse, *,
                     capture: bool = False) -> Optional[bytes]:
        """Stream `resp` back to the client. With `capture`, also return the body (None when it
        is over _INSPECT_CAP)."""
        kept: Optional[bytearray] = bytearray() if capture else None

        def _pull(read) -> bytes:
            nonlocal kept
            try:
                chunk = read(65536)
            except (OSError, http.client.HTTPException) as e:
                raise _UpstreamBroke(type(e).__name__) from e
            if kept is not None:
                if len(kept) + len(chunk) > _INSPECT_CAP:
                    kept = None
                else:
                    kept.extend(chunk)
            return chunk

        h.send_response_only(resp.status, resp.reason)
        for k, v in resp.getheaders():
            if k.lower() not in _HOP_BY_HOP and k.lower() not in ("date", "server"):
                h.send_header(k, v)
        length = resp.getheader("Content-Length")
        if h.command == "HEAD" or resp.status in (204, 304) or resp.status < 200:
            if length is not None:
                h.send_header("Content-Length", length)
            h.end_headers()
            h.wfile.flush()
            return bytes(kept) if kept is not None else None
        if length is not None:
            h.send_header("Content-Length", length)
            h.end_headers()
            remaining = int(length)
            while remaining > 0:
                chunk = _pull(resp.read)
                if not chunk:
                    raise _UpstreamBroke("short body")
                remaining -= len(chunk)
                h.wfile.write(chunk)
            h.wfile.flush()
            return bytes(kept) if kept is not None else None
        h.send_header("Transfer-Encoding", "chunked")
        h.end_headers()
        while True:
            chunk = _pull(resp.read1)
            if not chunk:
                break
            h.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            h.wfile.flush()
        h.wfile.write(b"0\r\n\r\n")
        h.wfile.flush()
        return bytes(kept) if kept is not None else None


# --------------------------------------------------------------------------------------------- #
# Harness errors and the scenarios they fall in
# --------------------------------------------------------------------------------------------- #
def _iso_seconds(value: Any) -> Optional[float]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def scenario_windows(summary: Mapping[str, Any],
                     scenario_summaries: Mapping[str, Mapping[str, Any]]
                     ) -> Dict[str, List[Tuple[float, float]]]:
    """When each scenario ran, by title: the run window (run.startedAt .. run.finishedAt) of each
    scenario's own summary (the suite runs every scenario in an isolated worker that writes
    scenarios/<id>/qa-suite-summary.json), or, for a run of one scenario, the run's window."""
    windows: Dict[str, List[Tuple[float, float]]] = {}
    for sid, summ in sorted(scenario_summaries.items()):
        run = summ.get("run") if isinstance(summ.get("run"), dict) else {}
        start, end = _iso_seconds(run.get("startedAt")), _iso_seconds(run.get("finishedAt"))
        names = [s.get("name") for s in summ.get("scenarios") or [] if isinstance(s, dict)]
        title = names[0] if len(names) == 1 and isinstance(names[0], str) else sid
        if start is not None and end is not None:
            windows.setdefault(title, []).append((start, end))
    scen = summary.get("scenarios")
    if not windows and isinstance(scen, list) and len(scen) == 1 and isinstance(scen[0], dict):
        run = summary.get("run") if isinstance(summary.get("run"), dict) else {}
        start, end = _iso_seconds(run.get("startedAt")), _iso_seconds(run.get("finishedAt"))
        if start is not None and end is not None and isinstance(scen[0].get("name"), str):
            windows[scen[0]["name"]] = [(start, end)]
    return windows


def attribute_events(events: Sequence[Tuple[Optional[float], str]],
                     windows: Mapping[str, Sequence[Tuple[float, float]]],
                     margin_s: float = 1.0,
                     unknown_spans: Sequence[Tuple[float, float]] = ()
                     ) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    """({scenario title: {kind: n}}, {kind: n}): each (time, kind) event counted against every
    scenario whose window (widened by `margin_s`) holds its time; an event no window holds,
    without a time, or held by one of `unknown_spans` (the window of a scenario that cannot be
    told, so it may have been any scenario's), is unattributed (once)."""
    by_title: Dict[str, Dict[str, int]] = {}
    unattributed: Dict[str, int] = {}
    for t, kind in events:
        hits = [title for title, spans in windows.items()
                if t is not None and any(s - margin_s <= t <= e + margin_s for s, e in spans)]
        unknown = t is not None and any(s - margin_s <= t <= e + margin_s
                                        for s, e in unknown_spans)
        if not hits or unknown:
            unattributed[kind] = unattributed.get(kind, 0) + 1
        for title in hits:
            by_title.setdefault(title, {})
            by_title[title][kind] = by_title[title].get(kind, 0) + 1
    return by_title, unattributed


# --------------------------------------------------------------------------------------------- #
# Scenarios that need a provider gbench does not run
# --------------------------------------------------------------------------------------------- #
#: Provider ids that are not "another provider": the model under test (vllm) and the Gemini API.
_OWN_PROVIDERS = frozenset({"vllm", "google"})


def _known_provider_ids(repo: Path) -> set:
    """Every provider and CLI-backend id a plugin of the checkout declares in its
    openclaw.plugin.json."""
    ids = set()
    for manifest in sorted((Path(repo) / "extensions").glob("*/openclaw.plugin.json")):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for key in ("providers", "cliBackends"):
            for pid in data.get(key) or []:
                if isinstance(pid, str) and pid.strip():
                    ids.add(pid.strip())
    return ids


def non_gemini_scenarios(repo: Path) -> Dict[str, Dict[str, Any]]:
    """The QA scenarios of the checkout at `repo` that need a model provider other than the model
    under test and the Gemini API, by id: {"title", "providers", "needs"}. A scenario needs one when
    its execution config requires the primary model's provider (requiredProvider /
    requiredLiveProvider, with requiredModel / requiredLiveModel) to be another one, or names a
    model of another provider ("<provider>/<model>" with a provider id a plugin declares)."""
    try:
        import yaml
    except ImportError:                                                     # pragma: no cover
        raise RuntimeError("infra_required: PyYAML (pyyaml) is needed to read the QA scenarios")
    known = _known_provider_ids(repo)
    found: Dict[str, Dict[str, Any]] = {}
    for md in sorted((Path(repo) / "qa" / "scenarios").rglob("*.md")):
        try:
            src = md.read_text(encoding="utf-8")
        except OSError:
            continue
        block = re.search(r"```yaml qa-scenario\n(.*?)\n```", src, re.S)
        if not block:
            continue
        try:
            meta = yaml.safe_load(block.group(1)) or {}
        except yaml.YAMLError:
            continue
        cfg = (meta.get("execution") or {}).get("config") or {}
        if not isinstance(cfg, dict) or not meta.get("id"):
            continue
        providers: List[str] = []
        needs: List[str] = []
        for pkey, mkey in (("requiredProvider", "requiredModel"),
                           ("requiredLiveProvider", "requiredLiveModel")):
            prov = cfg.get(pkey)
            if isinstance(prov, str) and prov.strip() and prov.strip() not in _OWN_PROVIDERS:
                prov = prov.strip()
                model = cfg.get(mkey)
                providers.append(prov)
                needs.append(f"primary model {prov}/{model}" if model else
                             f"primary model of provider {prov}")
        refs: List[str] = []

        def _walk(x: Any) -> None:
            if isinstance(x, dict):
                for v in x.values():
                    _walk(v)
            elif isinstance(x, list):
                for v in x:
                    _walk(v)
            elif isinstance(x, str):
                for m in re.finditer(r"(?<![\w./-])([a-z][a-z0-9-]*)/([A-Za-z0-9][\w.-]*)", x):
                    if m.group(1) in known and m.group(1) not in _OWN_PROVIDERS:
                        refs.append(m.group(0))
        _walk(cfg)
        for ref in dict.fromkeys(refs):
            providers.append(ref.split("/", 1)[0])
            needs.append(f"model {ref}")
        if providers:
            found[str(meta["id"])] = {"title": str(meta.get("title") or meta["id"]),
                                      "providers": sorted(set(providers)), "needs": needs}
    return found


def gemini_substitutions(image_input_declared: bool) -> List[Dict[str, str]]:
    """Every Gemini API capability the suite uses in place of an OpenAI one of the live-frontier
    lane: GemmaClaw's default (`upstream`), the Gemini model, the API and how it is set."""
    api = "Gemini API (GemmaClaw google plugin, through the gbench key relay)"
    subs = [
        {"role": "alternate model", "upstream": UPSTREAM_ALT_MODEL,
         "gemini_model": GEMINI_ALT_MODEL, "api": api, "how": "--alt-model"},
        {"role": "image-generation model", "upstream": UPSTREAM_IMAGE_GENERATION_MODEL,
         "gemini_model": GEMINI_IMAGE_GENERATION_MODEL, "api": api,
         "how": f"gbench edit of {GEMINI_IMAGE_PATCH_FILE}"},
        {"role": "memory embeddings", "upstream": UPSTREAM_EMBEDDING_MODEL,
         "gemini_model": GEMINI_EMBEDDING_MODEL, "api": api,
         "how": "memorySearch provider auto-selection (the google plugin is the keyed one)"},
    ]
    if not image_input_declared:
        subs.append({"role": "image understanding (image tool)",
                     "upstream": UPSTREAM_IMAGE_UNDERSTANDING_MODEL,
                     "gemini_model": GEMINI_IMAGE_UNDERSTANDING_MODEL, "api": api,
                     "how": "media-provider fallback for a model under test without image input"})
    return subs


def self_hosted_max_tokens(repo: Path, commit: str,
                           git_env: Mapping[str, str]) -> Tuple[int, str]:
    """(value, source) of SELF_HOSTED_DEFAULT_MAX_TOKENS at GemmaClaw `commit`, read on the host
    with `git show` (hooks and fsmonitor off; the checkout is only read). When the commit's file
    does not declare it, GEMMACLAW_SELF_HOSTED_MAX_TOKENS (the pinned commit's value), with a
    source that says so."""
    where = f"{_SELF_HOSTED_DEFAULTS_FILE} at GemmaClaw {commit[:12]}"
    try:
        res = subprocess.run(["git", *_SAFE_GIT, "show", f"{commit}:{_SELF_HOSTED_DEFAULTS_FILE}"],
                             cwd=str(repo), capture_output=True, text=True, env=dict(git_env),
                             timeout=60)
        match = _SELF_HOSTED_MAX_TOKENS_RE.search(res.stdout or "") if res.returncode == 0 \
            else None
    except (OSError, subprocess.SubprocessError):
        match = None
    if match:
        return int(match.group(1)), f"SELF_HOSTED_DEFAULT_MAX_TOKENS in {where}"
    return GEMMACLAW_SELF_HOSTED_MAX_TOKENS, (
        f"SELF_HOSTED_DEFAULT_MAX_TOKENS of the pinned GemmaClaw commit (not found in {where})")


def model_max_tokens(self_hosted: Optional[Tuple[int, str]], served_window: Optional[int],
                     max_output_tokens: Optional[int]) -> Dict[str, Any]:
    """The maxTokens gbench declares for the model under test in openclaw.json: GemmaClaw's own
    self-hosted default (`self_hosted`, else GEMMACLAW_SELF_HOSTED_MAX_TOKENS), which is what
    GemmaClaw's vLLM setup declares and so what the suite sends (pi-ai: min(maxTokens, 32000)),
    clamped only to the served window. The suite sends no thinking toggle, so --max-output-tokens
    (the eval pillar's flag) does not change it; the record says so when it was given.
    Returns {"model_max_tokens", "model_max_tokens_source", "max_output_tokens_note"}."""
    value, source = self_hosted or (
        GEMMACLAW_SELF_HOSTED_MAX_TOKENS,
        "SELF_HOSTED_DEFAULT_MAX_TOKENS of the pinned GemmaClaw commit")
    if served_window and served_window < value:
        value, source = served_window, f"{source}, clamped to the served window {served_window}"
    note = None
    if max_output_tokens is not None:
        note = (f"--max-output-tokens {max_output_tokens} not applied: the quality pillar declares "
                f"GemmaClaw's own self-hosted output cap ({value}) for the model under test")
    return {"model_max_tokens": value, "model_max_tokens_source": source,
            "max_output_tokens_note": note}


def _node_version(node_bin: Optional[str]) -> tuple:
    """Return the (major, minor) of a node binary, or (0, 0) if it is missing or
    unrunnable. ``None`` (e.g. shutil.which found nothing) -> (0, 0)."""
    if not node_bin:
        return (0, 0)
    try:
        out = subprocess.run([node_bin, "-v"], capture_output=True, text=True,
                             check=True).stdout.strip().lstrip("v")
        major, minor, *_ = (out.split(".") + ["0", "0"])
        return (int(major), int(minor))
    except (subprocess.SubprocessError, OSError, ValueError):
        return (0, 0)


class FileLock:
    """Simple Unix file locking context manager."""

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self.lock_file = None

    def __enter__(self):
        self.lock_file = open(self.lock_path, "w")
        fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.lock_file:
            fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
            self.lock_file.close()


def _json_or_none(data: Optional[bytes]) -> Any:
    if data is None:
        return None
    try:
        return json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


class QualityBenchmarkRunner(ServingBenchmarkRunner):
    """Runner for quality benchmarks using gemmaclaw QA suite."""

    def __init__(self, config: BenchmarkConfig):
        """Initialize the quality benchmark runner.

        Args:
            config: Benchmark configuration
        """
        super().__init__(config)
        self.gemmaclaw_repo_path = (
            Path(self.config.gemmaclaw_path)
            if self.config.gemmaclaw_path
            else Path(os.environ.get("GBENCH_CACHE_DIR") or "~/.cache/gbench").expanduser()
            / "gemmaclaw"
        )
        #: Why the last preflight check or setup failed (named in the run's error), the records of
        #: gbench's edits of the checkout (from the image, stored on the result) and the quality
        #: image _setup_gemmaclaw_repo made sure of.
        self._preflight_failure: Optional[str] = None
        self._setup_failure: Optional[str] = None
        self._gemmaclaw_patches: List[Dict[str, str]] = []
        self._image: Optional[str] = None
        #: (value, source) of the output cap GemmaClaw declares for a self-hosted model at the
        #: commit setup resolved (self_hosted_max_tokens); None until setup has read it.
        self._self_hosted_max_tokens: Optional[Tuple[int, str]] = None

    def _get_plugin_id_and_validity(self, plugin_dir: Path) -> tuple[Optional[str], bool]:
        """Resolve plugin ID and validity from its manifest or package.json.

        Returns:
            A tuple of (plugin_id, is_valid_plugin).
        """
        # Check openclaw.plugin.json first
        manifest_path = plugin_dir / "openclaw.plugin.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "id" in data:
                        return data["id"], True
            except Exception:
                pass
            return plugin_dir.name, True

        # Check package.json
        pkg_path = plugin_dir / "package.json"
        if pkg_path.exists():
            try:
                with open(pkg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "openclaw" in data:
                        pkg_name = data.get("name", "")
                        if pkg_name.startswith("@openclaw/"):
                            return pkg_name.split("/", 1)[1], True
                        return pkg_name or plugin_dir.name, True
            except Exception:
                pass

        return None, False

    @staticmethod
    def _docker_env() -> Dict[str, str]:
        """The environment of the host-side docker commands (host_env)."""
        return host_env(os.environ)

    def _preflight_check(self) -> bool:
        """Check the prerequisites: a Gemini API key, git, Docker (CLI and daemon) and gbench's
        docker/quality.Dockerfile.

        Returns:
            True if all checks pass, False otherwise.
        """
        logger.info("Performing quality runner preflight checks...")
        self._preflight_failure = None

        def _fail(why: str) -> bool:
            self._preflight_failure = why
            logger.error(why)
            return False

        # A Gemini API key: the alternate model and the image-generation model the QA suite calls
        # run on the Gemini API (GemmaClaw's google plugin); there is no other provider.
        if not _gemini_keys(os.environ):
            return _fail(
                "infra_required: neither GEMINI_API_KEYS nor GEMINI_API_KEY is set; the GemmaClaw "
                f"QA suite's alternate model ({GEMINI_ALT_MODEL}) and image-generation model "
                f"({GEMINI_IMAGE_GENERATION_MODEL}) run on the Gemini API")
        logger.info("Gemini API key check passed")

        # git: gbench clones or fetches GemmaClaw and archives the commit on the host.
        try:
            git_check = subprocess.run(["git", "--version"], capture_output=True, text=True,
                                       check=True)
            logger.info(f"git check passed: {git_check.stdout.strip()}")
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            return _fail(f"infra_required: git is not usable ({type(e).__name__}); gbench fetches "
                         "GemmaClaw with it. See docs/quality.md")

        # Docker: the suite runs in gbench's local quality container.
        why = None
        if not shutil.which("docker"):
            why = "Docker (the docker CLI) is not on PATH"
        else:
            try:
                probe = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                                       capture_output=True, text=True, timeout=60,
                                       env=self._docker_env())
                if probe.returncode != 0:
                    why = ("the Docker daemon is not reachable: "
                           + ((probe.stderr or "").strip().splitlines() or ["no message"])[-1])
            except (subprocess.SubprocessError, OSError) as e:
                why = f"the Docker daemon is not reachable ({type(e).__name__})"
        if why:
            return _fail(
                f"infra_required: {why}; the GemmaClaw QA suite runs in gbench's local Docker "
                "container so that no credential reaches the model under test (install Docker "
                "and give this user access to its daemon; see docs/quality.md)")
        if not (_DOCKER_DIR / QUALITY_DOCKERFILE).is_file():
            return _fail(
                f"infra_required: {_DOCKER_DIR / QUALITY_DOCKERFILE} is not in this gbench "
                "installation; the quality image is built from it (run gbench from its source "
                "checkout). See docs/quality.md")
        logger.info("Docker check passed")
        return True

    def _git_rewrite_safe_env(self) -> dict:
        """Env that neutralizes a host ~/.gitconfig ``insteadOf`` rule rewriting
        ``https://github.com`` -> ``ssh://git@github.com``. gemmaclaw is a PUBLIC
        repo; the rewrite turns public HTTPS fetches into SSH fetches needing a key,
        which hang on a prompt or fail ``Permission denied (publickey)``. Public
        repos need no auth, so we blank the global git config for these
        subprocesses and forbid interactive prompts (fail fast). Mirrors the
        swe_lancer eval fix; overridable via GBENCH_QUALITY_GIT_CONFIG_GLOBAL for
        a host that legitimately needs its rewrites (e.g. private deps).

        It also carries no provider credential (host_env): no variable of a model or service
        provider other than the Gemini API, and no variable holding a Gemini key or the
        endpoint's key, reaches the host-side git subprocesses (clone, fetch, rev-parse,
        archive)."""
        return self._with_git_settings(host_env(os.environ))

    @staticmethod
    def _with_git_settings(env: Dict[str, str]) -> Dict[str, str]:
        """`env` with the github-rewrite-safe git settings (see _git_rewrite_safe_env)."""
        env["GIT_CONFIG_GLOBAL"] = os.environ.get(
            "GBENCH_QUALITY_GIT_CONFIG_GLOBAL", os.devnull)
        env["GIT_TERMINAL_PROMPT"] = "0"
        env.setdefault(
            "GIT_SSH_COMMAND",
            "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10")
        return env

    def _secret_values(self) -> List[str]:
        """The credential values gbench knows before a run: the Gemini keys and the endpoint's
        key (base.endpoint_api_key)."""
        from .eval_suites.base import endpoint_api_key
        endpoint_key = endpoint_api_key() or ""
        return [*_gemini_keys(os.environ),
                *([endpoint_key] if endpoint_key and endpoint_key != LOCAL_VLLM_KEY else [])]

    @staticmethod
    def _resolve_commit(repo: Path, ref: str, git_env: Mapping[str, str],
                        remote_first: bool) -> str:
        """The full sha `ref` names in the checkout at `repo` (origin/<ref> first in gbench's own
        clone, so a branch name follows the fetched remote)."""
        refs = [ref] if re.fullmatch(r"[0-9a-f]{40}", ref) else (
            [f"origin/{ref}", ref] if remote_first else [ref, f"origin/{ref}"])
        for candidate in refs:
            res = subprocess.run(["git", *_SAFE_GIT, "rev-parse", "--verify", "--quiet",
                                  f"{candidate}^{{commit}}"],
                                 cwd=str(repo), capture_output=True, text=True, env=dict(git_env))
            sha = (res.stdout or "").strip()
            if res.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", sha):
                return sha
        raise RuntimeError(f"infra_required: GemmaClaw ref {ref!r} is not a commit of {repo}")

    def _setup_gemmaclaw_repo(self) -> Optional[str]:
        """Resolve the GemmaClaw commit and make sure its quality image exists.

        gbench's own checkout ($GBENCH_CACHE_DIR/gemmaclaw) is cloned (without a working tree)
        or fetched; --gemmaclaw-path is used as it is. git runs on the host with hooks and
        fsmonitor disabled, and only reads the checkout (clone, fetch, rev-parse, archive): no
        container ever sees it. The image is built from `git archive <commit>` with gbench's edits
        (ensure_quality_image).

        Returns:
            The full commit sha, or None on failure (the reason in self._setup_failure).
        """
        commit = self.config.gemmaclaw_commit
        repo = self.gemmaclaw_repo_path
        logger.info(f"Setting up gemmaclaw at {repo} (ref: {commit})")
        self._setup_failure = None
        git_env = self._git_rewrite_safe_env()
        lock_path = repo.parent / "gemmaclaw.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with FileLock(lock_path):
                if self.config.gemmaclaw_path:
                    if not repo.exists():
                        raise RuntimeError(f"infra_required: the --gemmaclaw-path checkout {repo} "
                                           "does not exist")
                    commit_hash = self._resolve_commit(repo, commit, git_env, False)
                else:
                    if not repo.exists():
                        repo.parent.mkdir(parents=True, exist_ok=True)
                        logger.info("Cloning gemmaclaw repo from GitHub...")
                        subprocess.run(["git", *_SAFE_GIT, "clone", "--no-checkout",
                                        GEMMACLAW_REMOTE, str(repo)], check=True, env=git_env)
                    else:
                        logger.info("gemmaclaw repo exists, fetching latest updates...")
                        subprocess.run(["git", *_SAFE_GIT, "fetch", "origin"], cwd=str(repo),
                                       check=True, env=git_env)
                    commit_hash = self._resolve_commit(repo, commit, git_env, True)
                logger.info(f"gemmaclaw is at commit: {commit_hash}")
                self._self_hosted_max_tokens = self_hosted_max_tokens(repo, commit_hash, git_env)
                self._image = ensure_quality_image(
                    repo, commit_hash, git_env=git_env, docker_env=self._docker_env(),
                    secret_values=self._secret_values())
                return commit_hash
        except Exception as e:
            self._setup_failure = str(e)
            logger.error(f"Failed to set up gemmaclaw repo: {e}")
            return None

    def _image_metadata(self, image: str, dest: Path) -> Dict[str, Any]:
        """Read what gbench needs of the image before a run, from a container created (never
        started) from it: the QA scenarios and plugin manifests (extracted under `dest`, data
        files only), gbench's edit records and the commit. Returns {"patches", "commit"}."""
        name = f"gbench-quality-meta-{secrets.token_hex(6)}"
        env = self._docker_env()
        subprocess.run(["docker", "create", "--name", name, "--network", "none", image],
                       check=True, capture_output=True, env=env)
        try:
            files = read_container_files(
                name, CONTAINER_IMAGE_DATA_DIR,
                lambda n: n in ("gbench/metadata.tar", "gbench/gemmaclaw_patches.json",
                                "gbench/gemmaclaw_commit"),
                env=env, max_file_bytes=256 << 20)
        finally:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, env=env)
        if "gbench/metadata.tar" not in files:
            raise RuntimeError(f"infra_required: the quality image {image} has no metadata bundle")
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(files["gbench/metadata.tar"])) as tf:
            members = [m for m in tf.getmembers()
                       if m.isreg() and m.name.endswith((".md", ".json"))]
            tf.extractall(dest, members=members, filter="data")
        patches = _json_or_none(files.get("gbench/gemmaclaw_patches.json"))
        return {"patches": patches if isinstance(patches, list) else [],
                "commit": files.get("gbench/gemmaclaw_commit", b"").decode("utf-8").strip()}

    def _collect_container_outputs(self, container: str) -> Dict[str, Any]:
        """What gbench reads from the stopped container: the suite summary, each scenario's own
        summary (by scenario id), the entrypoint's status and the forwarder's error records."""
        env = self._docker_env()
        scen_re = re.compile(rf"^{CONTAINER_OUTPUT_DIRNAME}/scenarios/([A-Za-z0-9._-]+)/"
                             r"qa-suite-summary\.json$")
        top_name = f"{CONTAINER_OUTPUT_DIRNAME}/qa-suite-summary.json"
        out = read_container_files(container, CONTAINER_OUTPUT_DIR,
                                   lambda n: n == top_name or bool(scen_re.match(n)), env=env)
        status = read_container_files(
            container, CONTAINER_STATUS_DIR,
            lambda n: n in ("gbench-status/entrypoint.json",
                            "gbench-status/forwarder-errors.jsonl"),
            env=env)
        scenario_summaries = {}
        for name, data in out.items():
            m = scen_re.match(name)
            parsed = _json_or_none(data)
            if m and isinstance(parsed, dict):
                scenario_summaries[m.group(1)] = parsed
        forwarder_errors = []
        for line in status.get("gbench-status/forwarder-errors.jsonl", b"").decode(
                "utf-8", "replace").splitlines():
            rec = _json_or_none(line.encode("utf-8"))
            if isinstance(rec, dict) and isinstance(rec.get("kind"), str):
                forwarder_errors.append(rec)
        return {"summary": _json_or_none(out.get(top_name)),
                "scenario_summaries": scenario_summaries,
                "entrypoint": _json_or_none(status.get("gbench-status/entrypoint.json")),
                "forwarder_errors": forwarder_errors}

    def _remove_container(self, container: str) -> None:
        subprocess.run(["docker", "rm", "-f", "-v", container], capture_output=True,
                       env=self._docker_env())

    def run(
        self,
        model: ModelConfig,
        format: ModelFormat,
    ) -> dict:
        """Run gemmaclaw quality benchmark suite against local vLLM.

        Args:
            model: Model configuration
            format: Model format (HF or GGUF)

        Returns:
            Dictionary with quality benchmark results

        Raises:
            RuntimeError: If benchmark execution fails
        """
        lm = self.config.log_manager
        output_file = lm.get_quality_result_path(model.short_name, format.value)
        log_file = lm.get_quality_log_path(model.short_name, format.value)

        # Preflight checks
        if not self.config.dry_run:
            if not self._preflight_check():
                raise RuntimeError("Quality runner preflight environment checks failed"
                                   + (f": {self._preflight_failure}"
                                      if self._preflight_failure else ""))

            commit_hash = self._setup_gemmaclaw_repo()
            if not commit_hash:
                raise RuntimeError("Failed to set up gemmaclaw repository"
                                   + (f": {self._setup_failure}" if self._setup_failure else ""))

        remote = self.config.remote_endpoint
        local = getattr(self.config, "local_endpoint", None)
        # The native server: the one the CLI started for this model's quality, Golden Set and eval
        # pillars (config.local_endpoint); a direct caller that configured no endpoint gets one
        # started (and stopped) here.
        if not self.config.dry_run and not remote and not local:
            if not self._start_server(model, format, context="native"):
                raise RuntimeError("Failed to start vLLM server")

        temp_dir = None
        extra_dirs: List[str] = []
        relay: Optional[KeyRelay] = None
        container: Optional[str] = None
        try:
            if self.config.dry_run:
                logger.info(f"[DRY RUN] Would run quality benchmarks for {model.short_name}")
                return {"dry_run": True}

            temp_dir = tempfile.mkdtemp(prefix="gbench_quality_")

            # Credential redaction for what this run writes and echoes.
            from ..utils.redaction import redact_file, redact_text, register_secret
            from .eval_suites.base import endpoint_api_key

            # The endpoint (one form for every pillar: ending in /v1) and its key. A remote
            # endpoint's: VLLM_API_KEY, else OPENAI_API_KEY (base.endpoint_api_key), else a GCP
            # identity token (opt-in). gbench's own server: VLLM_API_KEY only (the key vllm serve
            # reads from its environment).
            api_key = (endpoint_api_key() if remote else
                       endpoint_api_key({"VLLM_API_KEY": os.environ.get("VLLM_API_KEY") or ""}))
            # GCP identity-token auto-auth (metadata server + gcloud fallback) is OPT-IN: a public
            # user pointing at any *.googleapis.com / *.run.app endpoint should not trigger a
            # surprise metadata-server call or a `gcloud` subprocess. Enable with GBENCH_GCP_AUTH=1.
            _gcp_auth = os.environ.get("GBENCH_GCP_AUTH", "").strip().lower() in ("1", "true", "yes")
            if remote:
                endpoint_url = normalize_endpoint(remote)
                _looks_gcp = ("run.app" in endpoint_url or "googleapis.com" in endpoint_url)
                if not api_key and _looks_gcp and not _gcp_auth:
                    logger.info("Endpoint looks like a Google Cloud service but GBENCH_GCP_AUTH is not "
                                "set; not contacting the GCP metadata server or gcloud. Set VLLM_API_KEY "
                                "for auth, or GBENCH_GCP_AUTH=1 to fetch a GCP identity token automatically.")
                if not api_key and _looks_gcp and _gcp_auth:
                    # Try metadata server first (for Cloud Run)
                    try:
                        logger.info("Retrieving GCP identity token from Metadata Server...")
                        import urllib.request
                        # Use the service URL (without path) as audience
                        audience = endpoint_url.split("/v1")[0]
                        req = urllib.request.Request(
                            f"http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity?audience={audience}",
                            headers={"Metadata-Flavor": "Google"}
                        )
                        with urllib.request.urlopen(req, timeout=2) as resp:
                            api_key = resp.read().decode("utf-8").strip()
                        # Not in the environment, so name it a credential for redaction.
                        register_secret(api_key)
                        logger.info("Successfully retrieved token from Metadata Server.")
                    except Exception as e:
                        logger.debug(f"Metadata server token retrieval failed: {e}")

                    # Fallback to gcloud (for local dev)
                    if not api_key:
                        try:
                            logger.info("Retrieving GCP identity token via gcloud...")
                            api_key = subprocess.check_output(
                                ["gcloud", "auth", "print-identity-token"],
                                text=True
                            ).strip()
                            register_secret(api_key)
                        except Exception as e:
                            logger.warning(f"Could not retrieve GCP identity token: {e}. Defaulting to unauthenticated.")
            elif local:
                endpoint_url = normalize_endpoint(local)
            else:
                endpoint_url = f"http://127.0.0.1:{self.server_port}/v1"

            if not api_key:
                api_key = LOCAL_VLLM_KEY
            model_key = api_key if api_key != LOCAL_VLLM_KEY else None

            # The id the endpoint serves the model under test as (--served-model-id, else the
            # model's HF id, path or name matched against /v1/models; never the first listed id):
            # the id the relay expects, openclaw.json declares and the suite asks for.
            from .evals import (ServedModelError, ServedModelQuery, _SERVER_MAX_LEN,
                                _server_max_model_len, served_model_candidates,
                                served_model_for)
            _SERVER_MAX_LEN.pop(endpoint_url, None)   # same URL, a new model's window
            query = ServedModelQuery(served_model_candidates(model, format),
                                     getattr(self.config, "served_model_id", None),
                                     local=not remote)
            try:
                served = served_model_for(endpoint_url, query)
            except ServedModelError as e:
                raise RuntimeError(f"the model endpoint does not serve the model under test: {e}") \
                    from e
            model_path = served["id"]

            # The window the server actually serves (it may be capped by --max-model-len): if
            # GemmaClaw is told the model's full native window, its context management lets a
            # conversation grow past the real limit and the server rejects it with a 400.
            listed_window = _server_max_model_len(endpoint_url)
            explicit_window = (self.config.max_model_len_explicit
                               if (self.config.max_model_len_explicit or 0) > 0 else None)
            if listed_window:
                served_window = listed_window
                window_source = "the endpoint's /v1/models max_model_len"
            elif explicit_window:
                served_window, window_source = explicit_window, "--max-model-len"
            elif getattr(model, "metadata_source", None) == "defaults" \
                    and not getattr(model, "total_params_b", 0):
                # No config.json was read for the model (the CLI's "unread" rule), so
                # max_context_length is a placeholder; GemmaClaw would be told a window nobody
                # measured.
                raise RuntimeError(
                    "infra_required: the served context window is unknown: the endpoint lists no "
                    "max_model_len for the model and its config.json could not be read (pass "
                    "--tokenizer with the model's HF id for a remote endpoint, or check the model "
                    "id)")
            else:
                served_window, window_source = (model.max_context_length,
                                                "the model's configured context length")
            cap = (self._self_hosted_max_tokens or (GEMMACLAW_SELF_HOSTED_MAX_TOKENS,))[0]
            if served_window and served_window <= cap:
                # GemmaClaw declares its self-hosted output cap for the model and sends it with
                # every request: a window no larger than that cap cannot hold it plus a prompt,
                # so every request would be refused (a clamp to the window changes nothing).
                raise RuntimeError(
                    f"infra_required: the served context window ({served_window} tokens) cannot "
                    f"hold GemmaClaw's declared output cap ({cap} tokens) plus a prompt; serve "
                    f"a larger window (--max-model-len above {cap} plus the longest scenario "
                    "prompt)")
            max_tokens = model_max_tokens(self._self_hosted_max_tokens, served_window,
                                          getattr(self.config, "eval_max_output_tokens", None))
            model_under_test = {"served_model_id": model_path,
                                "served_model_id_source": served["how"],
                                "context_window": served_window,
                                "context_window_source": window_source, **max_tokens}

            gemini_keys = _gemini_keys(os.environ)
            if not gemini_keys:
                raise RuntimeError(
                    "infra_required: neither GEMINI_API_KEYS nor GEMINI_API_KEY is set; the key "
                    "relay of the GemmaClaw QA suite's Gemini API calls needs one")

            # What the run needs of the image: the scenarios and plugin manifests, and the record
            # of gbench's edits the image was built with.
            meta_dir = Path(temp_dir) / "metadata"
            meta = self._image_metadata(self._image, meta_dir)
            self._gemmaclaw_patches = list(meta["patches"])

            # The key relay: every credential stays in this process. The container reaches it
            # only through its forwarder, over this unix socket (the directory is mounted
            # read-only; the socket takes connections from the forwarder's user).
            relay_dir = os.path.join(temp_dir, "relay")
            if len(relay_dir) > 90:                  # a unix socket path holds 107 bytes
                extra_dirs.append(tempfile.mkdtemp(prefix="gbq_", dir="/tmp"))
                relay_dir = os.path.join(extra_dirs[-1], "relay")
            os.mkdir(relay_dir)
            os.chmod(relay_dir, 0o711)
            relay = KeyRelay(gemini_keys, model_upstream=endpoint_url, model_key=model_key,
                             model_id=model_path).start(os.path.join(relay_dir, RELAY_SOCKET_NAME))
            register_secret(relay.token)

            # openclaw.json: the model under test and the Gemini API, both through the forwarder.
            enabled_plugins = {"google", "qa-lab", "active-memory"}
            plugin_entries = {}
            extensions_dir = meta_dir / "extensions"
            if extensions_dir.exists():
                for item in sorted(extensions_dir.iterdir()):
                    if item.is_dir():
                        p_id, is_valid = self._get_plugin_id_and_validity(item)
                        if is_valid and p_id not in enabled_plugins:
                            plugin_entries[p_id] = {"enabled": False}
            for p in enabled_plugins:
                plugin_entries[p] = {"enabled": True}

            # A model under test that takes images is declared with image input, so image
            # attachments reach it and the image tool uses it, not a helper model.
            image_input = bool(getattr(model, "supports_multimodal", False))
            config_data = {
                "models": {
                    "providers": {
                        "vllm": {
                            "baseUrl": CONTAINER_MODEL_BASE_URL,
                            "apiKey": KEY_PLACEHOLDER,
                            "api": "openai-completions",
                            "models": [
                                {
                                    "id": model_path,
                                    "name": f"Local {model.short_name}",
                                    "input": ["text", "image"] if image_input else ["text"],
                                    "contextWindow": served_window,
                                    "maxTokens": max_tokens["model_max_tokens"]
                                }
                            ]
                        },
                        # The Gemini API through the forwarder and the relay: no key. GemmaClaw's
                        # model-request guard refuses the loopback address unless the provider
                        # allows a private network.
                        "google": {
                            "baseUrl": CONTAINER_GEMINI_BASE_URL,
                            "apiKey": KEY_PLACEHOLDER,
                            "api": "google-generative-ai",
                            "request": {"allowPrivateNetwork": True},
                            "models": [dict(_GEMINI_ALT_MODEL_DEFINITION)],
                        },
                    }
                },
                "plugins": {
                    "entries": plugin_entries
                }
            }
            config_path = os.path.join(temp_dir, "openclaw.json")
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)
            os.chmod(config_path, 0o644)

            logger.info(
                f"\n{'='*60}\n"
                f"  QUALITY BENCHMARK CONFIGURATION\n"
                f"{'='*60}\n"
                f"  Model:          {model.short_name} ({format.value})\n"
                f"  Served model:   {model_path}\n"
                f"  maxTokens:      {max_tokens['model_max_tokens']}\n"
                f"  Gemmaclaw Commit: {commit_hash[:10]}\n"
                f"  Image:          {self._image}\n"
                f"{'='*60}"
            )

            # The suite's environment: a fixed set of variables, nothing of gbench's.
            env, removed_env = sandbox_suite_env(
                os.environ, home=CONTAINER_HOME, npm_cache=CONTAINER_NPM_CACHE,
                config_path=CONTAINER_CONFIG_PATH, relay_base_url=CONTAINER_GEMINI_BASE_URL,
                model_api_key=KEY_PLACEHOLDER)
            if removed_env:
                logger.info("Not passed to the GemmaClaw suite (non-Gemini provider "
                            f"variables): {', '.join(removed_env)}")

            # The alternate model is explicit (the live-frontier default is openai/gpt-5.4):
            # GemmaClaw's google plugin on the Gemini API.
            cmd = [
                "pnpm", "openclaw", "qa", "suite",
                "--provider-mode", "live-frontier",
                "--model", f"vllm/{model_path}",
                "--alt-model", GEMINI_ALT_MODEL,
                "--output-dir", CONTAINER_OUTPUT_DIRNAME
            ]
            if getattr(self.config, "selected_scenarios", None):
                for scenario in self.config.selected_scenarios:
                    cmd.extend(["--scenario", scenario])

            container = f"gbench-quality-{secrets.token_hex(6)}"
            argv = container_run_argv(self._image, container, relay_dir=relay_dir,
                                      config_file=config_path, env=env, command=cmd)
            # The cloud metadata server guard, on the network of this very command line: for
            # --network none (the container's only way out is the relay's socket) it needs no
            # probe; any other network is probed and a reachable metadata server stops the run
            # with infra_required naming the host firewall rule.
            from .eval_suites.container_guard import require_metadata_blocked
            metadata_guard = require_metadata_blocked("quality", self._image, argv_network(argv),
                                                      docs_url="docs/quality.md")
            logger.info(f"Running gemmaclaw suite (container {container}): {' '.join(cmd)}")

            stopped_after_summary = False
            with open(log_file, "w") as log_fh:
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    env=self._docker_env(),
                )
                try:
                    # The relay token, on the container's standard input only.
                    process.stdin.write((relay.token + "\n").encode("utf-8"))
                    process.stdin.close()
                except (OSError, ValueError):
                    pass                         # docker exited early; its status says why
                summary_at: Optional[float] = None
                try:
                    # Read from log_file and write to sys.stdout in a deadlock-free way
                    with open(log_file, "r") as log_read:
                        while process.poll() is None:
                            line = log_read.readline()
                            if line:
                                if _SUITE_SUMMARY_LINE in line and summary_at is None:
                                    summary_at = time.monotonic()
                                sys.stdout.write(redact_text(line))
                                sys.stdout.flush()
                            elif (summary_at is not None and not stopped_after_summary
                                  and time.monotonic() - summary_at > SUITE_EXIT_GRACE_S):
                                logger.warning(
                                    f"The suite printed its summary {SUITE_EXIT_GRACE_S:.0f} s "
                                    "ago and has not exited; stopping its container")
                                subprocess.run(["docker", "stop", "--time", "10", container],
                                               capture_output=True, env=self._docker_env())
                                stopped_after_summary = True
                            else:
                                time.sleep(0.1)

                        # Read remaining lines
                        for line in log_read:
                            sys.stdout.write(redact_text(line))
                            sys.stdout.flush()

                    process.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    logger.error("gemmaclaw suite timed out after 3600 seconds. Killing process...")
                    subprocess.run(["docker", "kill", container], capture_output=True,
                                   env=self._docker_env())
                    process.kill()
                    process.wait()
                    raise RuntimeError("gemmaclaw suite timed out")
                finally:
                    # The suite wrote its own log under the results directory: no key stays in it.
                    redact_file(str(log_file))

            rc = process.returncode
            if rc in _ENTRYPOINT_FAILURES and not stopped_after_summary:
                raise RuntimeError(
                    f"the quality container failed before the suite ran: "
                    f"{_ENTRYPOINT_FAILURES[rc]} (exit {rc}). Check logs at: {log_file}")
            if rc != 0:
                logger.error(f"gemmaclaw suite exited with code {rc}")

            outputs = self._collect_container_outputs(container)
            qa_results = outputs.get("summary")
            if not isinstance(qa_results, dict):
                raise RuntimeError(
                    f"gemmaclaw failed to write suite summary. "
                    f"Check logs at: {log_file}"
                )
            relay_report = relay.report()
            outputs = {**outputs, "suite_exit_code": rc,
                       "suite_stopped_after_summary": stopped_after_summary}
            return self._score(model, format, commit_hash, qa_results, outputs, relay_report,
                               meta_dir, image_input, bool(model_key), removed_env, env,
                               output_file, metadata_guard=metadata_guard,
                               model_under_test=model_under_test)

        except Exception as e:
            logger.error(f"Quality benchmark failed: {e}")
            raise RuntimeError(f"Quality benchmark failed: {e}")

        finally:
            if relay is not None:
                relay.stop()
            if container is not None:
                self._remove_container(container)
            # Clean up temp folder
            for d in [temp_dir, *extra_dirs]:
                if d and os.path.exists(d):
                    shutil.rmtree(d, ignore_errors=True)
            # Cleanup server
            if not self.config.dry_run:
                self._cleanup_server()

    def _score(self, model: ModelConfig, format: ModelFormat, commit_hash: str,
               qa_results: Dict[str, Any], outputs: Mapping[str, Any],
               relay_report: Mapping[str, Any], meta_dir: Path, image_input: bool,
               endpoint_key_at_relay: bool, removed_env: List[str], env: Mapping[str, str],
               output_file: Path, *, metadata_guard: Optional[str] = None,
               model_under_test: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """The result of a run: scenarios that need another provider and scenarios harness errors
        fell in are listed apart and not scored; an error no scenario window holds, or no scenario
        left to score, withholds the pass rate (null, with pass_rate_withheld_reason), never
        0."""
        counts = qa_results.get("counts", {}) if isinstance(qa_results.get("counts"), dict) else {}
        blocked_scenarios = counts.get("blocked", 0)

        # Harness errors: the relay's, the forwarder's, and a forwarder that died.
        events: List[Tuple[Optional[float], str]] = list(relay_report["harness_error_events"])
        harness_errors: Dict[str, int] = dict(relay_report["harness_errors"])
        for rec in outputs.get("forwarder_errors") or []:
            kind = f"forwarder_{rec['kind']}"
            t = rec.get("t")
            events.append((float(t) if isinstance(t, (int, float)) else None, kind))
            harness_errors[kind] = harness_errors.get(kind, 0) + 1
        entry = outputs.get("entrypoint") if isinstance(outputs.get("entrypoint"), dict) else {}
        if entry.get("forwarder_alive_at_end") is False:
            events.append((None, "forwarder_exited"))
            harness_errors["forwarder_exited"] = harness_errors.get("forwarder_exited", 0) + 1
        harness_errors = dict(sorted(harness_errors.items()))

        scenario_summaries = outputs.get("scenario_summaries") or {}
        windows = scenario_windows(qa_results, scenario_summaries)
        id_by_title = {}
        for sid, summ in scenario_summaries.items():
            names = [s.get("name") for s in summ.get("scenarios") or [] if isinstance(s, dict)]
            if len(names) == 1 and isinstance(names[0], str):
                id_by_title[names[0]] = sid

        infra_required: List[Dict[str, Any]] = []
        harness_error_scenarios: List[Dict[str, Any]] = []
        scenarios = qa_results.get("scenarios")
        if isinstance(scenarios, list) and scenarios:
            # A window keyed by a title the summary does not list (a scenario's own summary that
            # names no scenario or several is keyed by its id) cannot be told apart: an error in
            # it may have been any scenario's, so it withholds the pass rate.
            titles = {s.get("name") for s in scenarios
                      if isinstance(s, dict) and isinstance(s.get("name"), str)}
            by_title_errors, unattributed = attribute_events(
                events, {t: w for t, w in windows.items() if t in titles},
                unknown_spans=[span for t, w in sorted(windows.items()) if t not in titles
                               for span in w])
            for s in scenarios:
                steps = s.get("steps") if isinstance(s, dict) else None
                if (isinstance(steps, list) and steps and all(
                        isinstance(st, dict) and st.get("name") == _SUITE_WORKER_STEP
                        for st in steps) and s.get("status") == "fail"
                        and isinstance(s.get("name"), str)):
                    errs = by_title_errors.setdefault(s["name"], {})
                    errs["suite_worker_failed"] = errs.get("suite_worker_failed", 0) + 1
                    harness_errors["suite_worker_failed"] = (
                        harness_errors.get("suite_worker_failed", 0) + 1)
            needs = non_gemini_scenarios(meta_dir)
            by_title = {v["title"]: (k, v) for k, v in needs.items()}
            scored = []
            for s in scenarios:
                title = s.get("name") if isinstance(s, dict) else None
                hit = by_title.get(title)
                if hit:
                    sid, info = hit
                    infra_required.append({"id": sid, "title": info["title"],
                                           "providers": info["providers"],
                                           "needs": info["needs"],
                                           "suite_status": s.get("status")})
                elif title in by_title_errors:
                    harness_error_scenarios.append({
                        "id": id_by_title.get(title), "title": title,
                        "suite_status": s.get("status"),
                        "harness_errors": dict(sorted(by_title_errors[title].items()))})
                else:
                    scored.append(s)
            reported = len(scenarios)
            total_scenarios = len(scored)
            passed_scenarios = sum(1 for s in scored
                                   if isinstance(s, dict) and s.get("status") == "pass")
            failed_scenarios = total_scenarios - passed_scenarios
        else:
            unattributed = {}
            for _t, kind in events:
                unattributed[kind] = unattributed.get(kind, 0) + 1
            reported = counts.get("total", 0)
            total_scenarios = counts.get("total", 0)
            passed_scenarios = counts.get("passed", 0)
            failed_scenarios = counts.get("failed", 0)
        unattributed = dict(sorted(unattributed.items()))
        harness_errors = dict(sorted(harness_errors.items()))

        # The pass rate is withheld (null, never 0) when a harness error may have hit any scenario
        # or when no scenario is left to score: a run whose every scenario a harness error fell
        # in (an endpoint down for the whole run, for example) has no score of the model's.
        why_withheld: List[str] = []
        if unattributed:
            why_withheld.append(
                "harness errors that no scenario's window holds (or the window of a scenario "
                "summary that does not name one scenario of the run), so any scenario may have "
                "met them: " + ", ".join(f"{k} x{v}" for k, v in unattributed.items()))
        if not total_scenarios > 0:
            left_out = [part for n, part in (
                (len(harness_error_scenarios),
                 f"{len(harness_error_scenarios)} had a harness error fall in them"),
                (len(infra_required),
                 f"{len(infra_required)} need a model provider gbench does not run")) if n]
            why_withheld.append(
                "no scenario could be scored: "
                + (f"of the {reported} scenario(s) the suite reported, " + " and ".join(left_out)
                   if left_out else "the suite reported none"))
        withheld = "; ".join(why_withheld) or None
        pass_rate: Optional[float] = (None if withheld else
                                      passed_scenarios / total_scenarios * 100)

        refused = []
        for r in relay_report.get("refused") or []:
            hits, _ = attribute_events([(t, "refused") for t in r["times"]], windows)
            refused.append({"route": r["route"], "method": r["method"], "path": r["path"],
                            "requests": r["requests"], "scenarios": sorted(hits)})

        substitutions = gemini_substitutions(image_input)
        reason = (
            "GemmaClaw's live-frontier QA lane calls OpenAI models besides the model under "
            "test; gbench runs every one of them on the Gemini API through GemmaClaw's google "
            "plugin instead: "
            + "; ".join(f"{s['role']} {s['upstream']} -> {s['gemini_model']}"
                        for s in substitutions)
            + f" (--alt-model, an edit of {GEMINI_IMAGE_PATCH_FILE}, and the google plugin's "
            "own capabilities), and the suite gets no key of another model or service provider; "
            "the model under test's tools can reach those Gemini API calls (the alternate model "
            "included) and the model endpoint through the container's relay forwarder, and such "
            "calls are counted with the suite's own")
        if infra_required:
            reason += ("; " + str(len(infra_required)) + " scenario(s) need a model provider "
                       "gbench does not run and are reported as infra_required, not scored: "
                       + ", ".join(f"{r['id']} ({', '.join(r['providers'])})"
                                   for r in infra_required))
        if harness_errors:
            reason += ("; harness errors: " + ", ".join(f"{k} x{v}" for k, v in
                                                        harness_errors.items())
                       + " (failed Gemini API, endpoint or relay calls, Gemini non-answers or "
                       "truncated replies, and scenario workers the suite could not run)")
            if harness_error_scenarios:
                reason += ("; " + str(len(harness_error_scenarios)) + " scenario(s) were "
                           "running when they happened and are not scored: "
                           + ", ".join(r["id"] or r["title"] for r in harness_error_scenarios))
        if withheld:
            reason += "; pass_rate withheld: " + withheld

        result = {
            "benchmark_type": "quality",
            "model_name": model.name,
            "model_short": model.short_name,
            "format": format.value,
            # The id the suite asked the endpoint for, how it was chosen, and the window and
            # output cap openclaw.json declared for the model under test.
            **dict(model_under_test or {}),
            "gemmaclaw_commit": commit_hash,
            "status": "completed_with_errors" if harness_errors else "completed",
            "total_scenarios": total_scenarios,
            "passed_scenarios": passed_scenarios,
            "failed_scenarios": failed_scenarios,
            "blocked_scenarios": blocked_scenarios,
            "pass_rate": pass_rate,
            "pass_rate_withheld_reason": withheld,
            "suite_counts": counts,
            "infra_required_scenarios": infra_required,
            "harness_error_scenarios": harness_error_scenarios,
            "unattributed_harness_errors": unattributed,
            "alternate_model": GEMINI_ALT_MODEL,
            "image_generation_model": GEMINI_IMAGE_GENERATION_MODEL,
            "image_input_declared": image_input,
            # Every Gemini capability the suite uses in place of an OpenAI one, and every
            # Gemini API call it made (counted at the relay).
            "gemini_substitutions": substitutions,
            "gemini_calls": relay_report["gemini_calls"],
            "model_endpoint_relayed": True,
            "model_endpoint_key_at_relay": endpoint_key_at_relay,
            "model_endpoint_calls": relay_report["model_endpoint_calls"],
            "harness_errors": harness_errors,
            "relay_refused_requests": refused,
            "suite_exit_code": outputs.get("suite_exit_code"),
            "suite_stopped_after_summary": bool(outputs.get("suite_stopped_after_summary")),
            "relay_rejected_requests": relay_report.get("rejected", 0),
            "gemmaclaw_patches": list(self._gemmaclaw_patches),
            "sandbox": {"tool": "docker",
                        "image": self._image,
                        "network": ("none: the relay's unix socket, through a forwarder inside "
                                    "the container, is the only way out"),
                        # the cloud metadata server guard's verdict for the container's network
                        # ("no-network": --network none needs no probe)
                        "metadata_guard": metadata_guard,
                        "root_filesystem": "read-only (the checkout included)",
                        "suite_user": f"qa (uid {SUITE_UID}), no capabilities, no-new-privileges",
                        "writable": ["/tmp (tmpfs)", f"{CONTAINER_HOME} (tmpfs)",
                                     *(f"{d} (a volume removed with the container)"
                                       for d in (CONTAINER_OUTPUT_DIR, CONTAINER_ARTIFACTS_DIR,
                                                 CONTAINER_NPM_CACHE))],
                        "gbench_only": [f"{CONTAINER_STATUS_DIR} (a volume removed with the "
                                        "container; the suite's user cannot read or write it)"],
                        "host_paths": [f"the relay's socket directory (as {CONTAINER_RELAY_DIR}, "
                                       "read-only)",
                                       f"openclaw.json (as {CONTAINER_CONFIG_PATH}, read-only)"]},
            # Names only: the variables the suite gets, and the provider variables of gbench's
            # environment it does not get.
            "suite_env": sorted(env),
            "non_gemini_env_removed": removed_env,
            "leaderboard_comparable": False,
            "leaderboard_comparable_reason": reason,
            "raw_results": qa_results
        }

        # The native server the CLI started (its bind address, why, whether it enforces a key).
        if getattr(self.config, "native_server_record", None) and not self.config.remote_endpoint:
            result["server"] = dict(self.config.native_server_record)
        # Save result to gbench logs (the shared redacting writer)
        write_redacted_json(output_file, result)

        logger.info(
            "Quality benchmark complete: "
            + (f"Pass rate={pass_rate:.1f}% " if pass_rate is not None else "Pass rate withheld ")
            + f"({passed_scenarios}/{total_scenarios} scenarios passed)"
        )
        return result
