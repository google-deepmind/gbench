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

# Helper module to robustly load and parse gbench datasets from JSONL/CSV/Textproto formats.

import csv
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Allow large CSV fields (MRCR 128k text)
try:
    csv.field_size_limit(sys.maxsize)
except Exception:
    csv.field_size_limit(2147483647)


def extract_lossless_image_b64(img_val: Any, base_dir: Optional[str] = None) -> Optional[str]:
    """Extract a raw base64 string from image bytes, a PIL object, or a path/data-URI string.

    ``base_dir`` joins a bare relative filename (e.g. MedXpertQA's ``'MM-0-a.jpeg'``) against
    the directory its image archive was extracted to. For a ``data:...;base64,`` URI the
    base64 payload is returned (the mime is dropped; use :func:`to_data_uri` to keep it).
    """
    import base64
    import io
    if img_val is None:
        return None
    if isinstance(img_val, bytes):
        return base64.b64encode(img_val).decode("utf-8")
    if isinstance(img_val, dict) and "bytes" in img_val and img_val["bytes"]:
        return base64.b64encode(img_val["bytes"]).decode("utf-8")
    if isinstance(img_val, str):
        s = img_val.strip()
        if not s:
            return None
        if s.startswith("data:") and "," in s:          # data URI -> raw base64 payload
            return s.split(",", 1)[1]
        candidates = [s]
        if base_dir:
            candidates.insert(0, os.path.join(base_dir, s))
        for path in candidates:
            try:
                if os.path.isfile(path):
                    with open(path, "rb") as f:
                        return base64.b64encode(f.read()).decode("utf-8")
            except Exception:
                pass
        return None
    if hasattr(img_val, "filename") and img_val.filename:
        try:
            with open(img_val.filename, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            pass
    if hasattr(img_val, "save"):
        im = img_val
        if hasattr(im, "mode") and im.mode not in ("RGB", "L", "RGBA"):
            im = im.convert("RGB")
        buf = io.BytesIO()
        fmt = getattr(im, "format", None) or "PNG"
        try:
            im.save(buf, format=fmt)
        except Exception:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    return None


_IMAGE_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
}


def _guess_mime(name: Any, default: str = "image/png") -> str:
    ext = os.path.splitext(str(name).lower())[1]
    return _IMAGE_MIME.get(ext, default)


def to_data_uri(img_val: Any, base_dir: Optional[str] = None) -> Optional[str]:
    """A full ``data:image/<mime>;base64,<payload>`` URI for an image value, or None.

    Passes an existing data URI through unchanged (keeping its mime); for a path/filename
    sniffs the mime from the extension; for bytes/PIL uses the PIL ``format`` when known,
    else ``image/png`` (servers sniff the actual bytes, so a default is safe).
    """
    if img_val is None:
        return None
    if isinstance(img_val, str):
        s = img_val.strip()
        if s.startswith("data:") and "," in s:
            return s
        b64 = extract_lossless_image_b64(s, base_dir=base_dir)
        return f"data:{_guess_mime(s)};base64,{b64}" if b64 else None
    b64 = extract_lossless_image_b64(img_val)
    if not b64:
        return None
    fmt = getattr(img_val, "format", None)
    mime = f"image/{fmt.lower()}" if fmt else "image/png"
    return f"data:{mime};base64,{b64}"


def build_image_message(prompt_text: str, images: Any,
                        base_dir: Optional[str] = None) -> Tuple[List[Dict[str, Any]], bool]:
    """OpenAI chat messages attaching one or more images ahead of ``prompt_text``.

    ``images`` may be a single image value or a list (all resolvable ones are attached).
    Returns ``(messages, attached)``; ``attached`` is False when NO image resolved and the
    message fell back to text-only - callers should assert/log on it, because a silent
    text-only fallback on a multimodal suite scores wrong without erroring.
    """
    if images is None:
        imgs: List[Any] = []
    elif isinstance(images, (list, tuple)):
        imgs = list(images)
    else:
        imgs = [images]
    parts: List[Dict[str, Any]] = []
    for im in imgs:
        uri = to_data_uri(im, base_dir=base_dir)
        if uri:
            parts.append({"type": "image_url", "image_url": {"url": uri}})
    if not parts:
        return [{"role": "user", "content": prompt_text}], False
    parts.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": parts}], True
