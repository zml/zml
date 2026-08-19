#!/usr/bin/env python3
"""Verify Milestone 13 latent-cache cases, hashes, and memory contracts."""

from __future__ import annotations

import hashlib
import json

import torch
from safetensors import safe_open

from export_layer0_prefix_reference import semantic_sha256
from export_mla_cache_cases import OUTPUT, SOURCE_SEMANTIC_SHA256
from export_reference import sha256_file, tensor_bytes


def main() -> None:
    manifest = json.loads((OUTPUT / "mla-latent-cache-cases.json").read_text())
    tensor_path = OUTPUT / manifest["tensor_file"]
    if sha256_file(tensor_path) != manifest["tensor_file_sha256"]:
        raise SystemExit("MLA cache fixture file hash mismatch")
    expected_contract = {
        "source_semantic_sha256": SOURCE_SEMANTIC_SHA256,
        "full_lengths": [1, 4, 8, 16],
        "split_length": 4,
        "split_points": [1, 2, 3],
        "repeated_decode_steps": 4,
        "case_count": 14,
        "cache_values_per_token": 576,
        "cache_dtype": "bfloat16",
    }
    if any(manifest.get(key) != value for key, value in expected_contract.items()):
        raise SystemExit("MLA cache fixture contract mismatch")
    if len(manifest["tensors"]) != 70:
        raise SystemExit("MLA cache tensor inventory mismatch")

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(tensor_path, framework="pt", device="cpu") as values:
        for name in values.keys():
            value = values.get_tensor(name).contiguous()
            tensors[name] = value
            expected = manifest["tensors"].get(name)
            dtype = str(value.dtype).removeprefix("torch.")
            if expected is None or list(value.shape) != expected["shape"] or dtype != expected["dtype"]:
                raise SystemExit(f"MLA cache tensor contract mismatch: {name}")
            if hashlib.sha256(tensor_bytes(value)).hexdigest() != expected["sha256"]:
                raise SystemExit(f"MLA cache tensor hash mismatch: {name}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise SystemExit(f"non-finite MLA cache tensor: {name}")
    if semantic_sha256(tensors) != manifest["tensor_semantic_sha256"]:
        raise SystemExit("MLA cache aggregate semantic hash mismatch")

    case_prefixes = sorted(name.removesuffix(".input") for name in tensors if name.endswith(".input"))
    if len(case_prefixes) != 14:
        raise SystemExit("MLA cache case prefix mismatch")
    for prefix in case_prefixes:
        input_value = tensors[f"{prefix}.input"]
        probabilities = tensors[f"{prefix}.expected.probabilities"].float()
        compressed = tensors[f"{prefix}.expected.cache.compressed"]
        extra = tensors[f"{prefix}.expected.cache.extra_key"]
        if compressed.shape[-1] != 512 or extra.shape[-1] != 64:
            raise SystemExit(f"MLA latent cache width mismatch: {prefix}")
        if compressed.shape[1] != probabilities.shape[-1] or extra.shape[1] != probabilities.shape[-1]:
            raise SystemExit(f"MLA latent cache length mismatch: {prefix}")
        if input_value.shape[1] != probabilities.shape[-2]:
            raise SystemExit(f"MLA query length mismatch: {prefix}")
        if not torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities.sum(-1)), atol=2e-3, rtol=2e-3):
            raise SystemExit(f"MLA cache probability normalization mismatch: {prefix}")

    latent_per_token_layer = (512 + 64) * 2
    expanded_per_token_layer = 96 * (192 + 128) * 2
    mla_1m = latent_per_token_layer * 24 * 1_000_000
    kda_all = (3 * 12288 * 4 * 2 + 96 * 128 * 128 * 4) * 69
    if (latent_per_token_layer, expanded_per_token_layer, mla_1m, kda_all) != (
        1152,
        61440,
        27_648_000_000,
        454_459_392,
    ):
        raise SystemExit("MLA/KDA cache memory formula mismatch")
    print(
        json.dumps(
            {
                "verified": str(tensor_path),
                "tensors": len(tensors),
                "cases": len(case_prefixes),
                "semantic_sha256": manifest["tensor_semantic_sha256"],
                "latent_bytes_per_token_layer": latent_per_token_layer,
                "expanded_bytes_per_token_layer": expanded_per_token_layer,
                "mla_1m_token_bytes": mla_1m,
                "all_kda_cache_bytes": kda_all,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
