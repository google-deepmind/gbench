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

"""Unit tests for the model registry (gbench/core/models.py): on-demand registration from the
exact --models id, metadata enrichment from config.json, and the per-id identity rules.

The sizing helpers of gbench/core/config.py are tested in tests/test_config.py."""

import json
import logging
import struct
from unittest import mock

import pytest

from gbench.core import models as M
from gbench.core.models import ModelCategory, ModelConfig, ModelFormat, ModelRegistry


def _write_dir(path, config, processor=None, safetensors=None):
    """A local checkpoint directory: config.json, an optional processor_config.json and an
    optional model.safetensors whose header declares `safetensors` ({name: shape})."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config))
    if processor is not None:
        (path / "processor_config.json").write_text(json.dumps(processor))
    if safetensors is not None:
        header = {name: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]}
                  for name, shape in safetensors.items()}
        header["__metadata__"] = {"format": "pt"}
        blob = json.dumps(header).encode()
        (path / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob)
    return path


_DENSE = {"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
          "intermediate_size": 128, "vocab_size": 1000, "max_position_embeddings": 4096,
          "architectures": ["DenseForCausalLM"]}


# -- Registry ----------------------------------------------------

def test_the_registry_has_no_builtin_model_table():
    """No model table: a fresh registry is empty and the old table machinery is gone."""
    assert ModelRegistry().list_all() == []
    for gone in ("_MODEL_DEFS", "_build_models", "MODELS", "PROJECT_ROOT", "MODELS_DIR"):
        assert not hasattr(M, gone), gone


def test_a_hub_id_registers_and_enriches_from_its_config():
    """A --models repo id registers on demand and is enriched from its config.json (the session
    stub in the repo conftest): params estimated, dense text, metadata_source 'config'."""
    reg = ModelRegistry()
    m = reg.get("org/dense-enrich-test")
    assert m is not None and m.hf_model_id == "org/dense-enrich-test"
    assert m.short_name == "dense-enrich-test"
    assert m.total_params_b > 0
    assert m.category == ModelCategory.TEXT and not m.supports_multimodal
    assert m.metadata_source == M.METADATA_FROM_CONFIG
    assert m.metadata_origin == "org/dense-enrich-test"
    assert reg.get("org/dense-enrich-test") is m
    assert reg.get("ORG/Dense-Enrich-Test") is m        # the Hub resolves ids case-insensitively
    assert m in reg.get_by_category(ModelCategory.TEXT)
    assert reg.list_all() == [m]


def test_lookups_are_by_exact_id_never_by_basename():
    """A bare basename resolves to nothing (it is an endpoint tag the caller registers), so a
    served tag is never swapped for a registered repo that shares its basename."""
    reg = ModelRegistry()
    m = reg.get("org/some-model")
    assert reg.get("some-model") is None
    tag = reg.register_hf_model("some-model")
    assert tag is not m and tag.hf_model_id == "some-model"
    assert tag.short_name != m.short_name
    assert tag.metadata_source == M.METADATA_DEFAULTS    # a tag has no config to read


def test_ids_that_share_a_basename_are_distinct_models_with_distinct_short_names(tmp_path):
    """google/x, someorg/x and a local directory named x are three models: each keeps its own
    id and metadata, and result files (keyed by short_name) cannot collide."""
    local = _write_dir(tmp_path / "ft" / "gemma-x", dict(_DENSE, vision_config={"a": 1}),
                       safetensors={"w": [1000, 1000]})
    reg = ModelRegistry()
    a = reg.get("google/gemma-x")
    b = reg.get("someorg/gemma-x")
    c = reg.get(str(local))
    assert len({id(a), id(b), id(c)}) == 3
    assert (a.hf_model_id, b.hf_model_id, c.hf_model_id) == (
        "google/gemma-x", "someorg/gemma-x", str(local))
    assert a.short_name == "gemma-x"
    assert b.short_name == "someorg--gemma-x"
    assert c.short_name == "ft--gemma-x"
    assert len({a.short_name, b.short_name, c.short_name}) == 3
    # each carries its own metadata: only the local one has a vision tower
    assert c.supports_multimodal and not a.supports_multimodal and not b.supports_multimodal
    # and a later lookup of each id returns that id's model
    assert reg.get("someorg/gemma-x") is b and reg.get(str(local)) is c


def test_a_short_name_already_qualified_is_de_duplicated_further():
    reg = ModelRegistry()
    reg.register_hf_model("a/m")
    reg.register_hf_model("b/m")
    third = reg.register_hf_model("b-tag:b/m")   # an endpoint id whose qualifier is taken again
    names = [m.short_name for m in reg.list_all()]
    assert len(set(n.lower() for n in names)) == 3, names
    assert third.hf_model_id == "b-tag:b/m"


def test_a_local_checkpoint_is_read_from_its_own_directory(tmp_path):
    """A local directory is introspected from its config.json (never sent to the Hub): its
    modalities, context and exact parameter count (safetensors headers) are its own."""
    d = _write_dir(tmp_path / "my-vlm",
                   dict(_DENSE, vision_config={"hidden_size": 8}, audio_config={"x": 1},
                        max_position_embeddings=32768),
                   safetensors={"a": [1000, 1000], "b": [500, 1000]})
    reg = ModelRegistry()
    with mock.patch.object(M, "hf_hub_download", side_effect=AssertionError("no Hub")), \
            mock.patch.object(M, "model_info", side_effect=AssertionError("no Hub")):
        m = reg.get(str(d))
    assert m.hf_model_id == str(d)
    assert m.get_model_path(ModelFormat.HF) == str(d)
    assert m.supports_multimodal and m.supports_audio
    assert m.category == ModelCategory.MULTIMODAL
    assert m.max_context_length == 32768
    assert m.total_params_b == 0.0015
    assert m.revision is None and m.metadata_source == M.METADATA_FROM_CONFIG


def test_a_relative_local_path_registers_under_its_absolute_path(tmp_path, monkeypatch):
    _write_dir(tmp_path / "ckpt", _DENSE)
    monkeypatch.chdir(tmp_path)
    reg = ModelRegistry()
    m = reg.get("./ckpt")
    assert m.hf_model_id == str(tmp_path / "ckpt")
    assert reg.get(str(tmp_path / "ckpt")) is m


def test_no_local_weights_convention_shadows_a_hub_id(tmp_path, monkeypatch):
    """A folder named like the model under some models directory is never used for a Hub id:
    the id is what vllm serves."""
    _write_dir(tmp_path / "models" / "gbench" / "shadowed", _DENSE)
    _write_dir(tmp_path / "models" / "shadowed", _DENSE)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMMA_MODELS_DIR", str(tmp_path / "models"))
    m = ModelRegistry().get("org/shadowed")
    assert m.get_model_path(ModelFormat.HF) == "org/shadowed"
    assert not hasattr(m, "local_path")


def test_only_the_checkpoint_and_a_remote_endpoint_are_formats():
    """GGUF is not served (the CLI refuses --format gguf): no format, no path for it."""
    assert {f.value for f in ModelFormat} == {"hf", "remote-endpoint"}
    with pytest.raises(ValueError):
        ModelFormat("gguf")
    m = ModelConfig(short_name="m", hf_model_id="org/m")
    assert m.get_model_path(ModelFormat.REMOTE) == "org/m"
    with pytest.raises(ValueError):
        m.get_model_path("gguf")
    for gone in ("Priority", "gguf_file", "gguf_model_id", "priority", "local_path"):
        assert not hasattr(M, gone) and not hasattr(m, gone), gone


# -- Enrichment --------------------------------------------------

@pytest.mark.parametrize("experts_key", ["num_experts", "num_local_experts", "n_routed_experts"])
def test_moe_topology_is_read_from_any_vendor_key(tmp_path, experts_key):
    cfg = dict(_DENSE, moe_intermediate_size=32, num_experts_per_tok=2, **{experts_key: 8})
    m = ModelRegistry().get(str(_write_dir(tmp_path / experts_key, cfg)))
    assert m.is_moe and m.num_experts == 8 and m.num_active_experts == 2


def test_the_moe_estimate_counts_every_expert(tmp_path):
    dense = ModelRegistry().get(str(_write_dir(tmp_path / "dense", _DENSE)))
    moe = ModelRegistry().get(str(_write_dir(
        tmp_path / "moe", dict(_DENSE, num_experts=64, moe_intermediate_size=128,
                               hidden_size=2048, num_hidden_layers=24))))
    assert not dense.is_moe
    assert moe.total_params_b > dense.total_params_b


def test_a_null_modality_block_means_no_tower(tmp_path):
    """"vision_config": null / "audio_config": null (a text-only variant) is not a tower."""
    text = ModelRegistry().get(str(_write_dir(
        tmp_path / "t", dict(_DENSE, vision_config=None, audio_config=None))))
    assert not text.supports_multimodal and not text.supports_audio
    assert text.category == ModelCategory.TEXT
    vision_only = ModelRegistry().get(str(_write_dir(
        tmp_path / "v", dict(_DENSE, vision_config={"x": 1}, audio_config=None))))
    assert vision_only.supports_multimodal and not vision_only.supports_audio
    audio_only = ModelRegistry().get(str(_write_dir(
        tmp_path / "a", dict(_DENSE, audio_config={"x": 1}))))
    assert audio_only.supports_audio and not audio_only.supports_multimodal
    assert audio_only.category == ModelCategory.MULTIMODAL


@pytest.mark.parametrize("cfg_extra,expected", [
    ({"vision_soft_tokens_per_image": 111}, 111),
    ({"vision_config": {"num_soft_tokens": 222}}, 222),
    ({"mm_tokens_per_image": 333}, 333),
])
def test_per_image_tokens_come_from_the_config_keys(tmp_path, cfg_extra, expected):
    cfg = dict(_DENSE, vision_config={"x": 1})
    cfg.update(cfg_extra)
    m = ModelRegistry().get(str(_write_dir(tmp_path / f"k{expected}", cfg)))
    assert (m.vision_tokens_per_image, m.vision_tokens_source) == (expected, "config")


def test_per_image_tokens_fall_back_to_the_processor_config(tmp_path):
    m = ModelRegistry().get(str(_write_dir(
        tmp_path / "p", dict(_DENSE, vision_config={"x": 1}), processor={"image_seq_length": 64})))
    assert (m.vision_tokens_per_image, m.vision_tokens_source) == (64, "processor")


def test_per_image_tokens_are_unknown_when_nothing_declares_them(tmp_path):
    """A dynamic-resolution processor declares no fixed count: None, never a borrowed constant."""
    m = ModelRegistry().get(str(_write_dir(tmp_path / "dyn", dict(_DENSE, vision_config={"x": 1}),
                                           processor={"max_pixels": 12845056})))
    assert m.supports_multimodal
    assert (m.vision_tokens_per_image, m.vision_tokens_source) == (None, None)
    assert ModelConfig(short_name="m", hf_model_id="o/m").vision_tokens_per_image is None


def test_the_hub_safetensors_count_is_exact_and_the_commit_is_recorded(monkeypatch):
    sha = "a" * 40
    info = mock.MagicMock(sha=sha)
    info.safetensors.total = 7_300_000_000
    seen = []
    real = M.hf_hub_download       # the session stub: a dense config.json

    def _download(repo, filename, **kw):
        seen.append(kw.get("revision"))
        return real(repo, filename, **kw)
    monkeypatch.setattr(M, "model_info", lambda repo, revision=None: info)
    monkeypatch.setattr(M, "hf_hub_download", _download)
    m = ModelRegistry().get("org/exact")
    assert m.total_params_b == 7.3
    assert (m.revision, m.revision_source) == (sha, "hub")
    assert seen and set(seen) == {sha}        # config.json read at that same commit


def test_offline_the_commit_comes_from_the_cache_snapshot(tmp_path, monkeypatch):
    sha = "b" * 40
    snap = tmp_path / "snapshots" / sha
    _write_dir(snap, _DENSE)

    def _offline(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(M, "model_info", _offline)
    monkeypatch.setattr(M, "hf_hub_download", lambda repo, filename, **kw: str(snap / filename))
    m = ModelRegistry().get("org/cached")
    assert (m.revision, m.revision_source) == (sha, "cache")
    assert m.metadata_source == M.METADATA_FROM_CONFIG


def test_an_unreadable_config_warns_and_records_defaults(monkeypatch, caplog):
    def _fail(*a, **k):
        raise OSError("401 gated")
    monkeypatch.setattr(M, "hf_hub_download", _fail)
    monkeypatch.setattr(M, "model_info", _fail)
    with caplog.at_level(logging.WARNING, logger=M.__name__):
        m = ModelRegistry().get("org/gated")
    assert m.metadata_source == M.METADATA_DEFAULTS and m.total_params_b == 0
    assert any("org/gated" in r.getMessage() and r.levelno == logging.WARNING
               for r in caplog.records)


def test_an_endpoint_tag_is_not_looked_up_and_does_not_warn(monkeypatch, caplog):
    monkeypatch.setattr(M, "hf_hub_download", mock.Mock(side_effect=AssertionError("no")))
    with caplog.at_level(logging.WARNING, logger=M.__name__):
        m = ModelRegistry().register_hf_model("gemma-qat:4b")
    assert m.metadata_source == M.METADATA_DEFAULTS
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_enrich_metadata_from_copies_every_derived_field(tmp_path):
    """A tag's metadata borrowed from its --tokenizer: every derived field (audio and per-image
    tokens included) is copied, identity is kept, and the source is recorded."""
    tok = _write_dir(tmp_path / "tok", dict(_DENSE, vision_config={"num_soft_tokens": 99},
                                            audio_config={"x": 1}, num_experts=4,
                                            num_experts_per_tok=1, max_position_embeddings=8192))
    tag = ModelConfig(short_name="m-qat-4b", hf_model_id="m-qat:4b")
    assert M.enrich_metadata_from(tag, str(tok)) is True
    assert (tag.short_name, tag.hf_model_id, tag.revision) == ("m-qat-4b", "m-qat:4b", None)
    probe = ModelRegistry().get(str(tok))
    for field_name in M._DERIVED_FIELDS:
        assert getattr(tag, field_name) == getattr(probe, field_name), field_name
    assert tag.supports_audio and tag.supports_multimodal and tag.vision_tokens_per_image == 99
    assert (tag.metadata_source, tag.metadata_origin) == (M.METADATA_FROM_TOKENIZER, str(tok))


def test_enrich_metadata_from_refuses_tags_and_unreadable_sources(monkeypatch):
    tag = ModelConfig(short_name="t", hf_model_id="t:1")
    assert M.enrich_metadata_from(tag, "") is False
    assert M.enrich_metadata_from(tag, "other:tag") is False

    def _fail(*a, **k):
        raise OSError("offline")
    monkeypatch.setattr(M, "hf_hub_download", _fail)
    monkeypatch.setattr(M, "model_info", _fail)
    assert M.enrich_metadata_from(tag, "org/unreadable") is False
    assert tag.metadata_source == M.METADATA_DEFAULTS and tag.total_params_b == 0


def test_vision_is_read_from_the_config_not_the_architecture_name(tmp_path):
    """A '...ForConditionalGeneration' text-to-text model is not a vision model; a text-only
    variant whose vision_config is null is not either; an older layout that names its vision
    tower is."""
    t5 = ModelRegistry().get(str(_write_dir(
        tmp_path / "t5", dict(_DENSE, architectures=["T5ForConditionalGeneration"]))))
    assert not t5.supports_multimodal and t5.category == ModelCategory.TEXT
    null_vision = ModelRegistry().get(str(_write_dir(
        tmp_path / "nv", dict(_DENSE, architectures=["XForConditionalGeneration"],
                              vision_config=None))))
    assert not null_vision.supports_multimodal
    legacy = ModelRegistry().get(str(_write_dir(
        tmp_path / "legacy", dict(_DENSE, mm_vision_tower="openai/clip"))))
    assert legacy.supports_multimodal


@pytest.mark.parametrize("key", ["max_position_embeddings", "n_positions", "max_sequence_length",
                                 "seq_length"])
def test_the_native_context_is_read_from_any_standard_key(tmp_path, key):
    cfg = {k: v for k, v in _DENSE.items() if k != "max_position_embeddings"}
    cfg[key] = 12345
    m = ModelRegistry().get(str(_write_dir(tmp_path / key, cfg)))
    assert m.max_context_length == 12345


def test_the_metadata_record_carries_identity_values_and_their_sources(tmp_path):
    m = ModelRegistry().get(str(_write_dir(tmp_path / "rec", dict(
        _DENSE, vision_config={"num_soft_tokens": 7}))))
    rec = m.metadata_record()
    assert rec["id"] == str(tmp_path / "rec") and rec["short_name"] == "rec"
    assert rec["metadata_source"] == "config" and rec["metadata_origin"] == str(tmp_path / "rec")
    assert rec["vision_tokens_per_image"] == 7 and rec["vision_tokens_source"] == "config"
    assert rec["category"] == "multimodal" and rec["revision"] is None
    json.dumps(rec)                               # plain JSON for metadata.json
