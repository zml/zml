from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from export_layer0_prefix_reference import TOKEN_IDS, semantic_sha256
from export_layer_family_reference import load_prefix_source
from export_reference import MOONSHOT_REVISION, sha256_file, tensor_record


def make_prefix_fixture(tmp_path):
    path = tmp_path / "prefix.safetensors"
    tensors = {
        "prefix.layer0.out": torch.arange(24, dtype=torch.float32).reshape(1, 3, 8),
        "prefix.layer0.block_residual.out": torch.arange(24, dtype=torch.float32).reshape(3, 1, 8),
    }
    save_file(tensors, path)
    manifest = {
        "fixture": "synthetic-prefix",
        "moonshot_revision": MOONSHOT_REVISION,
        "token_ids": list(TOKEN_IDS),
        "layer_stop": 1,
        "cpu_inference_fallback": False,
        "tensor_file": path.name,
        "tensor_file_sha256": sha256_file(path),
        "tensor_semantic_sha256": semantic_sha256(tensors),
        "tensors": {name: tensor_record(value) for name, value in tensors.items()},
    }
    path.with_suffix(".json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    return path, manifest


def test_load_prefix_source_binds_actual_fixture(tmp_path):
    path, manifest = make_prefix_fixture(tmp_path)
    tensors, source = load_prefix_source(path)
    assert set(tensors) == set(manifest["tensors"])
    assert source["tensor_file_sha256"] == manifest["tensor_file_sha256"]
    assert source["tensor_semantic_sha256"] == manifest["tensor_semantic_sha256"]
    assert source["manifest_sha256"] == sha256_file(path.with_suffix(".json"))


def test_load_prefix_source_rejects_forged_semantic_hash(tmp_path):
    path, manifest = make_prefix_fixture(tmp_path)
    manifest["tensor_semantic_sha256"] = "0" * 64
    path.with_suffix(".json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    with pytest.raises(RuntimeError, match="aggregate semantic hash mismatch"):
        load_prefix_source(path)
