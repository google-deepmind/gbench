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

"""Endpoint utilities: liveness, the served model id, and multimodal support.

Every request here carries the endpoint key (VLLM_API_KEY, else OPENAI_API_KEY:
base.endpoint_auth_headers) from gbench's own process.
"""

import json
import logging
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

from ..core.config import match_served_model, normalize_endpoint

logger = logging.getLogger(__name__)

TINY_1X1_PNG_BASE64 = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

#: HTTP statuses that mean the endpoint refused the credentials, not that it is down.
AUTH_REJECTION_STATUSES = (401, 403)
#: HTTP statuses with which an endpoint rejects an image payload for a model without vision
#: (vLLM answers 400 for a text-only model given an image; 415/422 are the other content
#: rejections). Anything else on the image probe is not a capability verdict.
IMAGE_REJECTION_STATUSES = (400, 415, 422)


class EndpointError(RuntimeError):
    """The endpoint could not be used as configured (unreachable, refused the key, or it does not
    serve the model the run names). A harness error, never a verdict on the model."""


def _auth_headers() -> Dict[str, str]:
    from ..runners.eval_suites.base import endpoint_auth_headers
    return endpoint_auth_headers()


def _root_url(base_url: str) -> str:
    """The server root of an OpenAI-compatible base URL (no trailing /v1)."""
    url = base_url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def verify_endpoint_functional(base_url: str, timeout: int = 10,
                               model_id: Optional[str] = None) -> Tuple[bool, str, Optional[int]]:
    """Check if an OpenAI/vLLM endpoint is functional and answering requests.

    GET <base>/v1/models with the endpoint key. A 401/403 is a hard failure (the key is missing
    or wrong: every request of the run would be refused), never passed by the unauthenticated
    /health fallback that is used only when /v1/models is unreachable.

    Args:
        base_url: Base URL of the endpoint (e.g. 'http://127.0.0.1:8000' or 'http://127.0.0.1:8000/v1')
        timeout: Timeout in seconds for the verification requests
        model_id: The served id the run uses; its entry's max_model_len is reported (else the
            first listed entry's).

    Returns:
        (is_functional, status_msg, max_model_len)
    """
    models_url = f"{normalize_endpoint(base_url)}/models"
    try:
        req = urllib.request.Request(models_url, headers=_auth_headers())
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m for m in (data.get("data") or []) if isinstance(m, dict)]
                entry = next((m for m in models if model_id and m.get("id") == model_id), None)
                entry = entry or (models[0] if models else None)
                max_len = None
                if entry and entry.get("max_model_len") is not None:
                    try:
                        max_len = int(entry["max_model_len"])
                    except (TypeError, ValueError):
                        max_len = None
                return True, "Endpoint is functional and answering /v1/models requests", max_len
    except urllib.error.HTTPError as e1:
        if e1.code in AUTH_REJECTION_STATUSES:
            return False, (f"{models_url} refused the request (HTTP {e1.code}): set VLLM_API_KEY "
                           "(or OPENAI_API_KEY) to the endpoint's key"), None
        return _health_fallback(base_url, timeout, e1)
    except Exception as e1:                                                # noqa: BLE001
        return _health_fallback(base_url, timeout, e1)
    return False, "Endpoint returned unexpected response format", None


def _health_fallback(base_url: str, timeout: int, first_error: Any
                     ) -> Tuple[bool, str, Optional[int]]:
    health_url = f"{_root_url(base_url)}/health"
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as resp:
            if resp.status == 200:
                return True, "Endpoint is functional (answered /health)", None
    except Exception as e2:                                                # noqa: BLE001
        return False, f"Unreachable (/v1/models: {first_error}; /health: {e2})", None
    return False, "Endpoint returned unexpected response format", None


def list_served_models(base_url: str, timeout: float = 5.0,
                       headers: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """The model entries (dicts with at least "id") an OpenAI-compatible endpoint lists at
    <base>/v1/models, with the endpoint key (or ``headers`` when given, e.g. a caller's own
    credential). Raises EndpointError when the listing cannot be read."""
    url = f"{normalize_endpoint(base_url)}/models"
    try:
        req = urllib.request.Request(url, headers=_auth_headers() if headers is None else headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise EndpointError(f"could not list the served models at {url} (HTTP {e.code})") from e
    except Exception as e:                                                 # noqa: BLE001
        raise EndpointError(f"could not list the served models at {url} ({e})") from e
    return [m for m in (data.get("data") or []) if isinstance(m, dict) and m.get("id")]


class ServedModel(NamedTuple):
    """The id a run sends as "model", how it was chosen, and that entry's context window."""
    id: str
    source: str
    max_model_len: Optional[int] = None


def resolve_served_model(base_url: str, candidates: Sequence[Optional[str]],
                         override: Optional[str] = None, *, local: bool = False,
                         timeout: float = 5.0,
                         headers: Optional[Dict[str, str]] = None) -> ServedModel:
    """The served id for a model, from the endpoint's /v1/models listing (match_served_model:
    --served-model-id first, then the candidates in order, e.g. the model's HF id, its path, its
    name, its short name).

    Raises EndpointError when the endpoint serves no matching id (a different model, or several
    and none of them this one: the message names --served-model-id), and when a LOCAL server
    (one gbench launched) cannot be listed. A remote endpoint whose listing fails yields the
    configured id (the override, else the first candidate) with source
    "configured (listing failed: ...)", so a proxy without /v1/models still runs and says so.
    ``headers`` replaces the endpoint key header for the listing (list_served_models)."""
    try:
        entries = list_served_models(base_url, timeout=timeout, headers=headers)
    except EndpointError as e:
        if local:
            raise
        configured = override or next((c for c in candidates if c), None)
        if not configured:
            raise
        logger.warning(f"{e}; using the configured model id '{configured}'")
        return ServedModel(str(configured), f"configured (listing failed: {e})", None)
    ids = [str(m["id"]) for m in entries]
    served_id, how = match_served_model(ids, list(candidates), override)
    if not served_id:
        raise EndpointError(f"{normalize_endpoint(base_url)}: {how}")
    entry = next((m for m in entries if str(m.get("id")) == served_id), {})
    max_len = entry.get("max_model_len")
    try:
        max_len = int(max_len) if max_len is not None else None
    except (TypeError, ValueError):
        max_len = None
    return ServedModel(served_id, how, max_len)


class MultimodalProbe(NamedTuple):
    """The image probe's verdict: "supported", "unsupported" (the endpoint rejected the image
    payload for the served model: a capability verdict) or "inconclusive" (the endpoint failed:
    5xx, an unknown model id, a refused key, a transport error), with the reason and the id
    probed."""
    verdict: str
    reason: str
    model_id: str


def probe_multimodal(base_url: str, model_id: str, timeout: int = 10) -> MultimodalProbe:
    """Probe whether an endpoint accepts an image for ``model_id``.

    Sends a 1x1 PNG chat request with max_tokens=1 to the served id: the listed id matching
    ``model_id`` (match_served_model), else ``model_id`` itself when the listing cannot be read.

    Capability is decided by HOW the endpoint responds, NOT by whether it responds within
    ``timeout``. A genuinely text-only model REJECTS an image payload fast with an HTTP 400/415/422
    (it validates the request before generating), whereas a multimodal model ACCEPTS the payload
    and only then does the slow work (a cold vision encoder, the image prefill). Decision table:
      - HTTP 200                                -> supported
      - HTTP 400/415/422 (payload rejected)     -> unsupported
      - timeout waiting for generation          -> supported (accepted but slow)
      - any other HTTP status (404 unknown model, 401/403 key, 429, 5xx), a model id the endpoint
        does not serve, or a connection error   -> inconclusive (a harness failure, not a verdict)
    """
    url = normalize_endpoint(base_url)
    chat_url = f"{url}/chat/completions"
    probe_id = model_id
    try:
        served = [str(m["id"]) for m in list_served_models(base_url, timeout=min(timeout, 10))]
    except EndpointError:
        served = None
    if served is not None:
        sid, how = match_served_model(served, [model_id])
        if not sid:
            return MultimodalProbe("inconclusive", f"the endpoint does not serve {model_id!r} ({how})",
                                   model_id)
        probe_id = sid

    payload = {
        "model": probe_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "test"},
                    {"type": "image_url", "image_url": {"url": TINY_1X1_PNG_BASE64}},
                ],
            }
        ],
        "max_tokens": 1,
    }
    req = urllib.request.Request(
        chat_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **_auth_headers()},
        method="POST",
    )
    slow = (f"image payload accepted, still generating after {timeout}s (slow cold vision load)")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return MultimodalProbe("supported", "HTTP 200", probe_id)
            return MultimodalProbe("inconclusive", f"HTTP {resp.status}", probe_id)
    except urllib.error.HTTPError as e:
        if e.code in IMAGE_REJECTION_STATUSES:
            return MultimodalProbe("unsupported", f"HTTP {e.code}: the image payload was rejected",
                                   probe_id)
        return MultimodalProbe("inconclusive", f"HTTP {e.code}", probe_id)
    except (socket.timeout, TimeoutError):
        return MultimodalProbe("supported", slow, probe_id)
    except urllib.error.URLError as e:
        if isinstance(e.reason, (socket.timeout, TimeoutError)):
            return MultimodalProbe("supported", slow, probe_id)
        return MultimodalProbe("inconclusive", f"transport error: {e.reason}", probe_id)
    except Exception as e:                                                 # noqa: BLE001
        return MultimodalProbe("inconclusive", f"{type(e).__name__}: {e}", probe_id)


def probe_multimodal_support(base_url: str, model_id: str, timeout: int = 10) -> bool:
    """Whether the multimodal pillars should run against this endpoint (probe_multimodal).

    False only for a definitive capability verdict (the endpoint rejected the image payload with
    400/415/422). True when the image was accepted, and also when the probe was inconclusive (an
    endpoint failure, an unknown model id): the multimodal pillars then run and their failure is
    recorded as a harness error (exit 2) instead of being skipped as if the model had no vision.
    """
    probe = probe_multimodal(base_url, model_id, timeout=timeout)
    where = f"{normalize_endpoint(base_url)}/chat/completions"
    if probe.verdict == "unsupported":
        logger.info(f"Multimodal probe for {probe.model_id} on {where}: {probe.reason} "
                    "-> unsupported")
        return False
    if probe.verdict == "supported":
        logger.info(f"Multimodal probe for {probe.model_id} on {where}: {probe.reason} "
                    "-> supported")
        return True
    logger.warning(f"Multimodal probe for {probe.model_id} on {where} was inconclusive "
                   f"({probe.reason}): the multimodal pillars run, and an endpoint failure is "
                   "recorded as a harness error rather than skipped as missing vision.")
    return True
