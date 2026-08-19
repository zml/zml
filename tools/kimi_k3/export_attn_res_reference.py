#!/usr/bin/env python3
"""Export synthetic and real-weight Kimi K3 Attention Residual fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

from export_reference import deterministic_setup, import_official
from reference_oracles import AttentionResidualResult, attention_residual_select


ROOT = Path("/ephemeral/kimi-k3")
CHECKPOINT = ROOT / "moonshot/kimi-k3"
OUTPUT = ROOT / "artifacts/fixtures/milestone-6"
SEED = 20260820
BLOCK_SIZE = 12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_sha256(tensors: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def record(value: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "elements": int(value.size),
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


def compare(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    delta = np.abs(actual - expected)
    close = np.isclose(actual, expected, atol=1e-5, rtol=1e-5)
    return {
        "passed": bool(close.all()),
        "atol": 1e-5,
        "rtol": 1e-5,
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean()) if delta.size else 0.0,
        "close_fraction": float(close.mean()) if close.size else 1.0,
    }


def official_select(
    modeling: Any,
    prefix: np.ndarray,
    blocks: np.ndarray,
    active: np.ndarray,
    norm_weight: np.ndarray,
    projection_weight: np.ndarray,
) -> np.ndarray:
    """Run the exact pinned `_apply_attn_res` on compact active CUDA sources."""
    hidden = prefix.shape[-1]
    norm = modeling.KimiRMSNorm(hidden, eps=1e-6).cuda().eval()
    projection = torch.nn.Linear(hidden, 1, bias=False).cuda().eval()
    with torch.inference_mode():
        norm.weight.copy_(torch.from_numpy(norm_weight).cuda())
        projection.weight.copy_(torch.from_numpy(projection_weight[None, :]).cuda())
        active_blocks = blocks[:, np.asarray(active, dtype=bool), :]
        output = modeling._apply_attn_res(
            torch.from_numpy(prefix).cuda(),
            torch.from_numpy(active_blocks).cuda(),
            projection,
            norm,
        )
    return output.float().cpu().numpy()


def add_case(
    tensors: dict[str, np.ndarray],
    checks: dict[str, Any],
    modeling: Any,
    name: str,
    prefix: np.ndarray,
    blocks: np.ndarray,
    active: np.ndarray,
    norm_weight: np.ndarray,
    projection_weight: np.ndarray,
) -> AttentionResidualResult:
    result = attention_residual_select(
        prefix, blocks, active, norm_weight, projection_weight, eps=1e-6
    )
    official = official_select(
        modeling, prefix, blocks, active, norm_weight, projection_weight
    )
    checks[name] = compare(official, result.output)
    if not checks[name]["passed"]:
        raise RuntimeError(f"official Attention Residual mismatch: {name}")
    values = {
        "prefix": prefix,
        "blocks": blocks,
        "active": np.asarray(active, dtype=np.bool_),
        "norm_weight": norm_weight,
        "projection_weight": projection_weight,
        "expected.output": result.output,
        "expected.candidates": result.candidates,
        "expected.scores": result.scores,
        "expected.masked_scores": result.masked_scores,
        "expected.probabilities": result.probabilities,
    }
    for suffix, value in values.items():
        tensors[f"{name}.{suffix}"] = np.ascontiguousarray(value)
    return result


def load_real_weights(checkpoint: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[str]]:
    weights: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    shards: list[str] = []
    for layer in range(4):
        shard_name = f"model-{layer + 1:05d}-of-000096.safetensors"
        shards.append(shard_name)
        prefix = f"language_model.model.layers.{layer}."
        with safe_open(checkpoint / shard_name, framework="pt", device="cuda:0") as values:
            for location in ("self_attention", "mlp"):
                norm = values.get_tensor(f"{prefix}{location}_res_norm.weight").float().cpu().numpy()
                projection = values.get_tensor(f"{prefix}{location}_res_proj.weight").float()[0].cpu().numpy()
                weights[f"layer{layer}.{location}"] = (norm, projection)
    shard_name = "model-00094-of-000096.safetensors"
    shards.append(shard_name)
    with safe_open(checkpoint / shard_name, framework="pt", device="cuda:0") as values:
        norm = values.get_tensor("language_model.model.output_attn_res_norm.weight").float().cpu().numpy()
        projection = values.get_tensor("language_model.model.output_attn_res_proj.weight").float()[0].cpu().numpy()
        weights["output"] = (norm, projection)
    return weights, shards


def build_fixture(checkpoint: Path, modeling: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    tensors: dict[str, np.ndarray] = {}
    checks: dict[str, Any] = {}

    prefix = rng.normal(size=(2, 8)).astype(np.float32)
    norm = rng.uniform(0.5, 1.5, size=(8,)).astype(np.float32)
    projection = rng.normal(size=(8,)).astype(np.float32)
    base_blocks = rng.normal(size=(2, 3, 8)).astype(np.float32)
    synthetic = {
        "synthetic.one_source": (np.zeros_like(base_blocks), [False, False, False]),
        "synthetic.multiple_sources": (base_blocks, [True, True, False]),
        "synthetic.all_sources": (base_blocks, [True, True, True]),
        "synthetic.inactive_stale": (
            np.concatenate((base_blocks[:, :1], np.full((2, 2, 8), 1e6, dtype=np.float32)), axis=1),
            [True, False, False],
        ),
    }
    for name, (blocks, active) in synthetic.items():
        add_case(tensors, checks, modeling, name, prefix, blocks, np.asarray(active), norm, projection)

    real_weights, shards = load_real_weights(checkpoint)
    hidden = 7168
    token_count = 2
    indices = np.arange(token_count * hidden, dtype=np.float32).reshape(token_count, hidden)
    prefix = (np.sin(indices * 0.00073) + 0.25 * np.cos(indices * 0.00019)).astype(np.float32)
    blocks = np.full((token_count, 2, hidden), 1e4, dtype=np.float32)
    active = np.asarray([False, False], dtype=np.bool_)

    locations: list[str] = []
    for layer in range(4):
        if layer > 0:
            name = f"real.layer{layer}.self_attention"
            norm_weight, proj_weight = real_weights[f"layer{layer}.self_attention"]
            add_case(tensors, checks, modeling, name, prefix, blocks, active, norm_weight, proj_weight)
            locations.append(name)

        if layer % BLOCK_SIZE == 0:
            slot = int(active.sum())
            blocks[:, slot, :] = prefix
            active[slot] = True
            prefix = np.zeros_like(prefix)

        attention_delta = (
            np.sin(indices * (0.00011 + layer * 0.00001) + layer) * 0.1
        ).astype(np.float32)
        prefix = prefix + attention_delta

        name = f"real.layer{layer}.mlp"
        norm_weight, proj_weight = real_weights[f"layer{layer}.mlp"]
        add_case(tensors, checks, modeling, name, prefix, blocks, active, norm_weight, proj_weight)
        locations.append(name)

        mlp_delta = (
            np.cos(indices * (0.00007 + layer * 0.00001) + layer) * 0.05
        ).astype(np.float32)
        prefix = prefix + mlp_delta

    norm_weight, proj_weight = real_weights["output"]
    add_case(tensors, checks, modeling, "real.output", prefix, blocks, active, norm_weight, proj_weight)
    locations.append("real.output")
    return tensors, {
        "official_vs_numpy": checks,
        "real_locations": locations,
        "source_shards": shards,
        "real_weight_tensors": 18,
        "block_size": BLOCK_SIZE,
        "fixed_block_capacity": 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    deterministic_setup()
    _, modeling = import_official(args.checkpoint)
    tensors, details = build_fixture(args.checkpoint, modeling)
    args.output.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output / "attn-res-reference.safetensors"
    save_file(tensors, tensor_path, metadata={"schema_version": "1", "milestone": "6"})
    manifest = {
        "schema_version": 1,
        "milestone": 6,
        "seed": SEED,
        "device": torch.cuda.get_device_name(0),
        "cpu_inference_fallback": False,
        "checkpoint_downloaded": False,
        "tensor_file": tensor_path.name,
        "tensor_file_sha256": sha256_file(tensor_path),
        "tensor_semantic_sha256": semantic_sha256(tensors),
        "tensors": {name: record(value) for name, value in sorted(tensors.items())},
        **details,
    }
    manifest_path = args.output / "attn-res-reference.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # KIMI_K3_TEMP_REMOVE_M20: verbose selector candidate/weight inventory is
    # retained for bring-up and removed at the cleanup milestone.
    if args.debug:
        for name, value in sorted(tensors.items()):
            if name.endswith(("expected.candidates", "expected.probabilities", "expected.output")):
                print(f"[kimi-k3-debug] {name} shape={value.shape} dtype={value.dtype}")
    print(json.dumps({"fixture": str(tensor_path), "locations": len(details["official_vs_numpy"])}))


if __name__ == "__main__":
    main()
