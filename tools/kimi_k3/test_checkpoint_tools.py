from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

import checkpoint_tools as tools


def make_checkpoint(root: Path, include_shard: bool = True) -> tuple[Path, np.ndarray]:
    root.mkdir()
    shard = "model-00001-of-000096.safetensors"
    name = "language_model.model.layers.0.input_layernorm.weight"
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    if include_shard:
        save_file({name: values}, root / shard)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {name: shard}}) + "\n"
    )
    (root / "config.json").write_text("{}\n")
    return root, values


def test_inspector_rejects_full_index_with_missing_shard(tmp_path):
    model, _ = make_checkpoint(tmp_path / "model", include_shard=False)
    result = tools.inspect(model)
    assert result["valid_for_loading"] is False
    assert result["missing_shards"] == ["model-00001-of-000096.safetensors"]


def test_s1_mini_index_is_self_contained_and_uses_symlink(tmp_path):
    model, _ = make_checkpoint(tmp_path / "model")
    output = tmp_path / "derived" / "S1"
    result = tools.make_mini(model, output, "S1")
    assert result["validation"]["valid_for_loading"] is True
    assert (output / "model-00001-of-000096.safetensors").is_symlink()
    assert result["manifest"]["source_files_modified"] is False


def test_mini_index_refuses_source_subdirectory(tmp_path):
    model, _ = make_checkpoint(tmp_path / "model")
    with pytest.raises(tools.CheckpointError, match="separate"):
        tools.make_mini(model, model / "derived", "S1")


def test_bounded_extraction_preserves_tensor(tmp_path):
    model, values = make_checkpoint(tmp_path / "model")
    output = tmp_path / "fixtures" / "one.safetensors"
    name = "language_model.model.layers.0.input_layernorm.weight"
    manifest = tools.extract(model, output, [name], max_bytes=values.nbytes)
    assert manifest["payload_bytes"] == values.nbytes
    np.testing.assert_array_equal(load_file(output)[name], values)
    reused = tools.extract(model, output, [name], max_bytes=values.nbytes)
    assert reused["reused"] is True


def test_extraction_enforces_byte_limit(tmp_path):
    model, values = make_checkpoint(tmp_path / "model")
    name = "language_model.model.layers.0.input_layernorm.weight"
    with pytest.raises(tools.CheckpointError, match="exceeds"):
        tools.extract(model, tmp_path / "out.safetensors", [name], max_bytes=values.nbytes - 1)


def test_extraction_refuses_missing_tensor(tmp_path):
    model, _ = make_checkpoint(tmp_path / "model")
    with pytest.raises(tools.CheckpointError, match="absent from index"):
        tools.extract(model, tmp_path / "out.safetensors", ["missing"], max_bytes=1)
