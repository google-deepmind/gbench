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

"""Model registry: every model is resolved on demand from its --models value.

There is no built-in model table. A --models value is one of:

* a HuggingFace repo id (``org/name``);
* a local checkpoint directory (the value itself is an existing directory; nothing else is
  looked up on disk, so a HuggingFace id is never swapped for a same-named local folder);
* a ``gs://`` checkpoint prefix;
* with --remote-endpoint, the id that endpoint serves (for example an Ollama tag), which has no
  config of its own; the CLI can borrow metadata from --tokenizer (enrich_metadata_from).

The model's metadata (total parameters, MoE topology, vision and audio support, native context
length, and the per-image token count when the checkpoint declares a fixed one) is derived from
its own config.json: read from the directory for a local checkpoint, from the HuggingFace cache
or Hub for a repo id (at the commit it resolves to, recorded in ``revision``), and with
``gcloud storage cp`` for gs://. ``metadata_source`` records where the values came from:
"config", "tokenizer" (borrowed from another id) or "defaults" (nothing could be read).

Each model is registered under its exact id, and its short_name (used in result file names) is
unique within the process: a second id with the same basename gets its parent qualifier
(``someorg--my-model``), so two checkpoints that share a basename are never benchmarked as one or
written to the same files.
"""

import json
import logging
import os
import re
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from huggingface_hub import hf_hub_download, model_info

logger = logging.getLogger(__name__)


class ModelFormat(Enum):
    """How the model under test is reached: a checkpoint gbench serves with vLLM (HF), or an
    OpenAI-compatible endpoint someone else serves (REMOTE)."""

    HF = "hf"
    REMOTE = "remote-endpoint"


class ModelCategory(Enum):
    """Model categories, derived from the config: any vision or audio tower is MULTIMODAL."""

    TEXT = "text"
    MULTIMODAL = "multimodal"


# Where the metadata of a model came from (ModelConfig.metadata_source).
METADATA_FROM_CONFIG = "config"
METADATA_FROM_TOKENIZER = "tokenizer"
METADATA_DEFAULTS = "defaults"

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

#: Config keys that name a vision tower or an image token in layouts without a vision_config.
_VISION_HINT_KEYS = ("mm_vision_tower", "vision_tower", "img_processor", "image_token_index",
                     "image_token_id")
#: Config keys that carry the native context length, in the order they are read.
_CONTEXT_KEYS = ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length")


@dataclass
class ModelConfig:
    """One model under test.

    Core fields (short_name, hf_model_id) are required. Everything else is derived from the
    model's config.json at registration, or keeps the default shown here, which
    ``metadata_source == "defaults"`` then records.
    """

    # -- Required ----------------------------------------------------
    short_name: str
    hf_model_id: str

    # -- Derived from config.json (populated by _enrich_from_config) ---
    total_params_b: float = 0.0
    is_moe: bool = False
    num_experts: int = 0
    num_active_experts: int = 0
    supports_multimodal: bool = False
    supports_audio: bool = False
    # Per-image placeholder (soft) tokens when the checkpoint declares a fixed count (config.json
    # or its processor config); None when it declares none (dynamic-resolution processors), in
    # which case the caller must measure it or record it as unknown.
    vision_tokens_per_image: Optional[int] = None
    vision_tokens_source: Optional[str] = None     # "config" | "processor" | None
    category: ModelCategory = ModelCategory.TEXT
    max_context_length: int = 128_000

    # -- Identity and provenance -------------------------------------
    name: str = ""  # Human-readable name (defaults to short_name)
    # The HuggingFace commit the metadata was read at (a 40-hex sha), when the id is a repo id
    # and the commit could be resolved; None for local directories, gs:// and endpoint tags.
    revision: Optional[str] = None
    revision_source: Optional[str] = None          # "hub" | "cache" | None
    metadata_source: str = METADATA_DEFAULTS
    metadata_origin: Optional[str] = None          # the id or directory whose config was read

    def __post_init__(self):
        if not self.name:
            self.name = self.short_name

    def metadata_record(self) -> Dict[str, Any]:
        """What a run records about this model (metadata.json): its identity, the derived
        metadata, and where each came from (metadata_source / metadata_origin, revision /
        revision_source, vision_tokens_source)."""
        return {
            "name": self.name, "short_name": self.short_name, "id": self.hf_model_id,
            "revision": self.revision, "revision_source": self.revision_source,
            "metadata_source": self.metadata_source, "metadata_origin": self.metadata_origin,
            "total_params_b": self.total_params_b, "is_moe": self.is_moe,
            "num_experts": self.num_experts, "num_active_experts": self.num_active_experts,
            "supports_multimodal": self.supports_multimodal, "supports_audio": self.supports_audio,
            "vision_tokens_per_image": self.vision_tokens_per_image,
            "vision_tokens_source": self.vision_tokens_source,
            "category": self.category.value, "max_context_length": self.max_context_length,
            **({"vision_tokens_note": "declared by the model's config/processor; each "
                                      "multimodal result records the per-image count it used "
                                      "(per_image_tokens) and its source "
                                      "(per_image_tokens_source: measured, config or processor)"}
               if self.vision_tokens_per_image else {}),
        }

    def get_model_path(self, format: ModelFormat) -> str:
        """What `vllm serve` (or an endpoint client) is given for this model: the id itself. A
        local checkpoint's id is its absolute directory path."""
        if format in (ModelFormat.HF, ModelFormat.REMOTE):
            return self.hf_model_id
        raise ValueError(f"{format} is not served by gbench (only the HuggingFace checkpoint "
                         "or a remote endpoint)")


# -- Reading a model's config ------------------------------------

def _is_endpoint_tag(model_id: str) -> bool:
    """True for an id that names no checkpoint gbench can read: an endpoint tag such as an Ollama
    name ('m:4b') or a bare name without an org ('m')."""
    if os.path.isdir(model_id) or model_id.startswith("gs://"):
        return False
    return ":" in model_id or "/" not in model_id


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _fetch_file(model_id: str, filename: str, revision: Optional[str]
                ) -> Tuple[Optional[str], Optional[str]]:
    """(local path of `filename` of the model, error). A local directory is read in place; a repo
    id comes from the HuggingFace cache first (no network, no 401 noise for a gated model), else
    from the Hub, at `revision` when given."""
    if os.path.isdir(model_id):
        path = os.path.join(model_id, filename)
        return (path, None) if os.path.isfile(path) else (None, f"{path} does not exist")
    try:
        return hf_hub_download(model_id, filename, revision=revision,
                               local_files_only=True), None
    except Exception:                                                     # noqa: BLE001
        pass
    try:
        return hf_hub_download(model_id, filename, revision=revision), None
    except Exception as e:                                                # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _read_gcs_config(model_id: str) -> Dict[str, Any]:
    temp_dir = tempfile.mkdtemp(prefix="gbench_gcs_config_")
    config_path = os.path.join(temp_dir, "config.json")
    try:
        subprocess.run(
            ["gcloud", "storage", "cp", model_id.rstrip("/") + "/config.json", config_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        data = _read_json(config_path)
        if data is None:
            raise ValueError("config.json is not a JSON object")
        return data
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _snapshot_sha(path: Optional[str]) -> Optional[str]:
    """The commit of a file in the HuggingFace cache (.../snapshots/<sha>/<file>), else None."""
    if not path:
        return None
    parent = Path(path).parent.name
    return parent if _SHA_RE.match(parent) else None


def _local_safetensors_params(directory: str) -> Optional[float]:
    """Total parameters (in billions) of a local checkpoint, counted exactly from the tensor
    shapes in its *.safetensors headers (no weights are read); None when it has none.

    The files counted are those the checkpoint's model.safetensors.index.json names in its
    weight_map when it has one; else every *.safetensors, less a consolidated*.safetensors
    shipped beside model*.safetensors shards (the same weights in another layout)."""
    total = 0
    found = False
    root = Path(directory)
    paths = sorted(root.glob("*.safetensors"))
    try:
        index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
        named = sorted({str(v) for v in (index.get("weight_map") or {}).values()})
    except (OSError, ValueError, AttributeError):
        named = []
    if named:
        paths = [root / name for name in named]
    elif any(p.name.startswith("model") for p in paths):
        paths = [p for p in paths if not p.name.startswith("consolidated")]
    for path in paths:
        try:
            with open(path, "rb") as f:
                (length,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(length))
        except Exception:                                                 # noqa: BLE001
            return None
        for key, meta in header.items():
            if key == "__metadata__" or not isinstance(meta, dict):
                continue
            count = 1
            for dim in meta.get("shape") or []:
                count *= int(dim)
            total += count
            found = True
    return total / 1e9 if found else None


def _round_params_b(billions: float) -> float:
    """A parameter count in billions, to 0.1B; a model under 0.1B keeps 4 decimals so it never
    reads as 0 (which callers treat as 'unknown')."""
    return round(billions, 1) if billions >= 0.1 else round(billions, 4)


def _processor_image_tokens(model_id: str, revision: Optional[str]) -> Optional[int]:
    """The fixed per-image token count a model's processor declares (processor_config.json or
    preprocessor_config.json: image_seq_length, image_seq_len or num_image_tokens), else None."""
    if model_id.startswith("gs://"):
        return None
    for filename in ("processor_config.json", "preprocessor_config.json"):
        path, _err = _fetch_file(model_id, filename, revision)
        if not path:
            continue
        try:
            data = _read_json(path) or {}
        except Exception:                                                 # noqa: BLE001
            continue
        for key in ("image_seq_length", "image_seq_len", "num_image_tokens"):
            value = data.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None


def _enrich_from_config(model: ModelConfig) -> bool:
    """Populate the derived fields of `model` from its own config.json. Returns True when the
    config was read (metadata_source "config"); False otherwise, with a WARNING naming the id
    and the cause for anything but an endpoint tag (which has no config to read)."""
    model_id = model.hf_model_id
    if _is_endpoint_tag(model_id):
        return False
    info = None
    try:
        if model_id.startswith("gs://"):
            config = _read_gcs_config(model_id)
        else:
            if not os.path.isdir(model_id):
                # The Hub's view of the repo, when reachable: the commit the requested revision
                # resolves to, and the exact parameter count from its safetensors metadata.
                try:
                    info = model_info(model_id, revision=model.revision)
                except Exception:                                         # noqa: BLE001
                    info = None
                sha = getattr(info, "sha", None)
                if isinstance(sha, str) and _SHA_RE.match(sha):
                    model.revision, model.revision_source = sha, "hub"
            path, err = _fetch_file(model_id, "config.json", model.revision)
            if not path:
                raise OSError(err or "config.json not found")
            if model.revision is None:
                sha = _snapshot_sha(path)
                if sha:
                    model.revision, model.revision_source = sha, "cache"
            config = _read_json(path)
            if config is None:
                raise ValueError("config.json is not a JSON object")
    except Exception as e:                                                # noqa: BLE001
        logger.warning(
            f"Could not read config.json of {model_id} ({e}): its size, MoE topology, "
            "modalities and context length are unknown and keep their defaults. For a gated "
            "repo set HF_TOKEN; offline, a cached model works with HF_HUB_OFFLINE=1.")
        return False

    tc = config.get("text_config") or config

    # -- Architecture fields -----------------------------------------
    hidden = tc.get("hidden_size", 0) or 0
    layers = tc.get("num_hidden_layers", 0) or 0
    heads = tc.get("num_attention_heads", 0) or 0
    kv_heads = tc.get("num_key_value_heads", heads) or heads
    intermediate = tc.get("intermediate_size", 0) or 0
    vocab = tc.get("vocab_size", config.get("vocab_size", 0)) or 0

    # MoE detection (field names vary across vendors)
    num_experts = (tc.get("num_experts") or tc.get("num_local_experts")
                   or tc.get("n_routed_experts") or 0)
    num_active = tc.get("num_experts_per_tok") or tc.get("top_k_experts") or 0
    moe_intermediate = tc.get("moe_intermediate_size", 0) or 0
    n_shared = tc.get("n_shared_experts", 0) or 0
    is_moe = num_experts > 0

    model.is_moe = is_moe
    model.num_experts = num_experts
    model.num_active_experts = num_active

    # -- Multimodal / audio detection --------------------------------
    # A modality block that is present with a null value means the checkpoint has no tower for
    # it (a text-only variant of a multimodal family), so the value decides, not the key. A
    # config without a vision_config block is a vision model only when it names a vision tower
    # or an image token (older multimodal layouts); the architecture name is not evidence (a
    # "...ForConditionalGeneration" class is also a text-to-text or speech model).
    if "vision_config" in config:
        has_vision = config.get("vision_config") is not None
    else:
        has_vision = any(config.get(k) is not None for k in _VISION_HINT_KEYS)
    has_audio = config.get("audio_config") is not None
    model.supports_multimodal = has_vision
    model.supports_audio = has_audio
    # The fixed per-image token count, when the checkpoint declares one. These keys live at the
    # top level or under vision_config (never text_config); else the processor config.
    vcfg = config.get("vision_config") if isinstance(config.get("vision_config"), dict) else {}
    vtoks = (config.get("vision_soft_tokens_per_image") or vcfg.get("num_soft_tokens")
             or config.get("mm_tokens_per_image"))
    model.vision_tokens_per_image, model.vision_tokens_source = None, None
    if isinstance(vtoks, int) and not isinstance(vtoks, bool) and vtoks > 0:
        model.vision_tokens_per_image, model.vision_tokens_source = vtoks, "config"
    elif has_vision:
        ptoks = _processor_image_tokens(model_id, model.revision)
        if ptoks:
            model.vision_tokens_per_image, model.vision_tokens_source = ptoks, "processor"
    model.category = (ModelCategory.MULTIMODAL if (has_vision or has_audio)
                      else ModelCategory.TEXT)

    # -- Max context length ------------------------------------------
    for key in _CONTEXT_KEYS:
        max_pos = tc.get(key, config.get(key))
        if isinstance(max_pos, int) and not isinstance(max_pos, bool) and max_pos > 0:
            model.max_context_length = max_pos
            break

    # -- Total params ------------------------------------------------
    # Exact when known (the Hub's safetensors metadata, or a local checkpoint's safetensors
    # headers), else estimated from the architecture.
    exact = None
    if os.path.isdir(model_id):
        exact = _local_safetensors_params(model_id)
    else:
        st = getattr(info, "safetensors", None)
        st_total = getattr(st, "total", None)
        if isinstance(st_total, (int, float)) and not isinstance(st_total, bool) and st_total:
            exact = st_total / 1e9
    if exact:
        model.total_params_b = _round_params_b(exact)
    else:
        head_dim = tc.get("head_dim") or (hidden // heads if heads else 0)
        embed_params = vocab * hidden
        attn_per_layer = (hidden * heads * head_dim + hidden * kv_heads * head_dim * 2
                          + heads * head_dim * hidden)
        if is_moe:
            expert_inter = moe_intermediate if moe_intermediate else intermediate
            expert_ffn = 3 * hidden * expert_inter
            ffn_per_layer = num_experts * expert_ffn + n_shared * expert_ffn
        else:
            ffn_per_layer = 3 * hidden * intermediate
        model.total_params_b = _round_params_b(
            (embed_params + layers * (attn_per_layer + ffn_per_layer)) / 1e9)

    model.metadata_source = METADATA_FROM_CONFIG
    model.metadata_origin = model_id
    return True


def _hf_id_to_short_name(hf_model_id: str) -> str:
    """The basename of an id: 'google/gemma-4-E4B-it' -> 'gemma-4-E4B-it' (casing kept)."""
    return hf_model_id.rstrip("/").split("/")[-1]


#: The derived fields enrich_metadata_from copies (everything but identity and provenance).
_DERIVED_FIELDS = ("total_params_b", "is_moe", "num_experts", "num_active_experts",
                   "supports_multimodal", "supports_audio", "vision_tokens_per_image",
                   "vision_tokens_source", "category", "max_context_length")


def enrich_metadata_from(model: ModelConfig, source_id: str) -> bool:
    """Populate a model's DERIVED metadata (params, MoE, modalities, context, per-image tokens)
    from a DIFFERENT id - a HuggingFace repo id or a local checkpoint directory - without
    changing its identity (short_name, hf_model_id, revision).

    For a remote endpoint whose served id cannot be introspected (an endpoint tag such as an
    Ollama name), pass the user's --tokenizer so the report shows real params and modality
    instead of the defaults. Returns True, with metadata_source "tokenizer" and metadata_origin
    `source_id`, when that config was read and gave a parameter count."""
    if not source_id or _is_endpoint_tag(str(source_id)):
        return False
    source = str(source_id)
    if os.path.isdir(source):
        source = os.path.abspath(source)
    probe = ModelConfig(short_name=model.short_name, hf_model_id=source)
    try:
        read = _enrich_from_config(probe)
    except Exception:                                                     # noqa: BLE001
        return False
    if not read or not probe.total_params_b or probe.total_params_b <= 0:
        return False
    for field_name in _DERIVED_FIELDS:
        setattr(model, field_name, getattr(probe, field_name))
    model.metadata_source = METADATA_FROM_TOKENIZER
    model.metadata_origin = source
    return True


def _canonical_id(model_id: str) -> str:
    """The id a model is registered under: a local directory's absolute path, else the id with
    surrounding whitespace (and a gs:// prefix's trailing '/') removed."""
    mid = str(model_id).strip()
    if os.path.isdir(mid):
        return os.path.abspath(mid)
    return mid.rstrip("/") if mid.startswith("gs://") else mid


def _is_hub_id(model_id: str) -> bool:
    """A HuggingFace repo id ('org/name'): the Hub resolves these case-insensitively."""
    return (model_id.count("/") == 1 and ":" not in model_id
            and not model_id.startswith(("/", ".", "~")) and not os.path.isdir(model_id))


def _qualifier(model_id: str) -> str:
    """The component before an id's basename (the org of a repo id, the parent directory of a
    path, the bucket path of gs://), made safe for a file name; '' when there is none."""
    parts = [p for p in model_id.replace("gs://", "", 1).rstrip("/").split("/") if p]
    if len(parts) < 2:
        return ""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", parts[-2]).strip("._") or ""


class AmbiguousModelError(ValueError):
    """A lookup matched more than one registered model."""


class ModelRegistry:
    """The models of this process, registered on demand from their exact --models id.

    ``get(id)`` returns the model registered under that exact id (a HuggingFace repo id also
    matches case-insensitively, as the Hub resolves it), else registers it when the id names a
    readable checkpoint (a repo id, a local directory or gs://). Nothing is matched by basename:
    'someorg/m' and '/data/m' are distinct models from 'google/m', and each gets a unique
    short_name."""

    def __init__(self):
        self._models: Dict[str, ModelConfig] = {}

    def _find(self, model_id: str) -> Optional[ModelConfig]:
        key = _canonical_id(model_id)
        model = self._models.get(key)
        if model is not None:
            return model
        if _is_hub_id(key):
            hits = [m for k, m in self._models.items() if _is_hub_id(k) and k.lower() == key.lower()]
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                raise AmbiguousModelError(
                    f"{model_id!r} matches {[m.hf_model_id for m in hits]} case-insensitively; "
                    "pass the exact id")
        return None

    def _unique_short_name(self, model_id: str) -> str:
        base = _hf_id_to_short_name(model_id) or "model"
        taken = {m.short_name.lower() for m in self._models.values()}
        if base.lower() not in taken:
            return base
        qualifier = _qualifier(model_id)
        candidate = f"{qualifier}--{base}" if qualifier else base
        n = 2
        unique = candidate
        while unique.lower() in taken:
            unique = f"{candidate}-{n}"
            n += 1
        holder = next(m for m in self._models.values() if m.short_name.lower() == base.lower())
        logger.warning(
            f"{model_id} and {holder.hf_model_id} share the basename '{base}': {model_id} is "
            f"named '{unique}' in results so the two are never mixed up.")
        return unique

    def register_hf_model(self, hf_model_id: str) -> ModelConfig:
        """Register a model from its exact id (a repo id, a local checkpoint directory, gs://,
        or an endpoint tag) and derive its metadata from its config.json. An id registered
        before returns the existing entry."""
        existing = self._find(hf_model_id)
        if existing is not None:
            return existing
        key = _canonical_id(hf_model_id)
        short = self._unique_short_name(key)
        model = ModelConfig(short_name=short, hf_model_id=key)
        _enrich_from_config(model)
        self._models[key] = model
        logger.info(
            f"Registered {short} from {key} ({model.total_params_b:.1f}B, "
            f"{'MoE' if model.is_moe else 'dense'}, "
            f"{'multimodal' if model.supports_multimodal else 'text'}; "
            f"metadata: {model.metadata_source})")
        return model

    def get(self, name: str) -> Optional[ModelConfig]:
        """The model registered under exactly `name`; else, when `name` names a readable
        checkpoint (a repo id with '/', a local directory, gs://), that model newly registered;
        else None (an endpoint tag the caller registers explicitly)."""
        model = self._find(name)
        if model is not None:
            return model
        key = _canonical_id(name)
        if not _is_endpoint_tag(key):
            return self.register_hf_model(key)
        return None

    def get_by_category(self, category: ModelCategory) -> list:
        """The registered models of a category."""
        return [m for m in self._models.values() if m.category == category]

    def list_all(self) -> list:
        """Every registered model, in registration order."""
        return list(self._models.values())


# Global registry instance
registry = ModelRegistry()
