#!/usr/bin/env python3
"""Verify the Milestone 11 selected-expert MoE fixture without inference."""

from __future__ import annotations

import hashlib
import json

import torch
from safetensors import safe_open

from export_layer0_prefix_reference import semantic_sha256
from export_moe_reference import OUTPUT
from export_reference import sha256_file, tensor_bytes


def main() -> None:
    manifest = json.loads((OUTPUT / "selected-moe-reference.json").read_text())
    tensor_path = OUTPUT / manifest["tensor_file"]
    if sha256_file(tensor_path) != manifest["tensor_file_sha256"]:
        raise SystemExit("MoE fixture file hash mismatch")
    if manifest["selected_expert_count"] != 61 or manifest["route_count"] != 64:
        raise SystemExit("MoE selected expert/route contract mismatch")
    if manifest["matrix_probe_count"] != 183 or len(manifest["per_matrix"]) != 183:
        raise SystemExit("MoE per-matrix inventory mismatch")
    if "test fixture" not in manifest["compact_map_scope"]:
        raise SystemExit("compact map scope is not test-only")

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(tensor_path, framework="pt", device="cpu") as values:
        for name in values.keys():
            value = values.get_tensor(name).contiguous()
            tensors[name] = value
            expected = manifest["tensors"].get(name)
            dtype = str(value.dtype).removeprefix("torch.")
            if expected is None or list(value.shape) != expected["shape"] or dtype != expected["dtype"]:
                raise SystemExit(f"MoE tensor contract mismatch: {name}")
            if hashlib.sha256(tensor_bytes(value)).hexdigest() != expected["sha256"]:
                raise SystemExit(f"MoE tensor hash mismatch: {name}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise SystemExit(f"non-finite MoE tensor: {name}")
    if set(tensors) != set(manifest["tensors"]):
        raise SystemExit("MoE tensor inventory mismatch")
    if semantic_sha256(tensors) != manifest["tensor_semantic_sha256"]:
        raise SystemExit("MoE aggregate semantic hash mismatch")

    selected = tensors["moe.selected_global_ids"].tolist()
    local = tensors["moe.local_route_ids"]
    global_ids = tensors["moe.global_route_ids"]
    reconstructed = torch.tensor(selected, dtype=torch.int64)[local]
    if not torch.equal(reconstructed, global_ids):
        raise SystemExit("global-to-local route reconstruction mismatch")
    if not torch.allclose(tensors["moe.route_weights"].sum(-1), torch.ones(1, 4), atol=1e-6, rtol=1e-6):
        raise SystemExit("MoE route weights are not normalized")
    print(
        json.dumps(
            {
                "verified": str(tensor_path),
                "tensors": len(tensors),
                "selected_experts": len(selected),
                "routes": int(global_ids.numel()),
                "matrix_probes": manifest["matrix_probe_count"],
                "semantic_sha256": manifest["tensor_semantic_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
