#!/usr/bin/env python3
"""Verify Milestone 14 sequential layer-family fixture without inference."""

from __future__ import annotations

import hashlib
import json

import torch
from safetensors import safe_open

from export_layer0_prefix_reference import semantic_sha256
from export_layer_family_reference import OUTPUT, PREFIX_FIXTURE, SHARDS, load_prefix_source
from export_reference import sha256_file, tensor_bytes


def main() -> None:
    manifest = json.loads((OUTPUT / "layer-family-reference.json").read_text())
    path = OUTPUT / manifest["tensor_file"]
    if sha256_file(path) != manifest["tensor_file_sha256"]:
        raise SystemExit("layer-family fixture file hash mismatch")
    _, prefix_source = load_prefix_source(PREFIX_FIXTURE)
    if manifest["prefix_fixture_semantic_sha256"] != prefix_source["tensor_semantic_sha256"]:
        raise SystemExit("layer-family prefix source mismatch")
    if manifest.get("prefix_fixture_source") != prefix_source:
        raise SystemExit("layer-family prefix provenance record mismatch")
    if manifest["checkpoint"] != {name: digest for name, digest in SHARDS.values()}:
        raise SystemExit("layer-family checkpoint contract mismatch")
    expected = {
        "1": ("kda", 61),
        "2": ("kda", 56),
        "3": ("mla", 53),
    }
    for layer, (attention, experts) in expected.items():
        details = manifest["layers"][layer]
        if details["attention"] != attention or details["selected_expert_count"] != experts:
            raise SystemExit(f"layer-family inventory mismatch: {layer}")
        if details["route_count"] != 64 or "isolated harness" not in details["compact_map_scope"]:
            raise SystemExit(f"layer-family route scope mismatch: {layer}")

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as values:
        for name in values.keys():
            value = values.get_tensor(name).contiguous()
            tensors[name] = value
            record = manifest["tensors"].get(name)
            dtype = str(value.dtype).removeprefix("torch.")
            if record is None or list(value.shape) != record["shape"] or dtype != record["dtype"]:
                raise SystemExit(f"layer-family tensor contract mismatch: {name}")
            if hashlib.sha256(tensor_bytes(value)).hexdigest() != record["sha256"]:
                raise SystemExit(f"layer-family tensor hash mismatch: {name}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise SystemExit(f"non-finite layer-family tensor: {name}")
    if len(tensors) != 163 or semantic_sha256(tensors) != manifest["tensor_semantic_sha256"]:
        raise SystemExit("layer-family aggregate semantic mismatch")
    for layer in (1, 2, 3):
        prefix = f"layer{layer}"
        if not torch.equal(tensors[f"{prefix}.route.global_ids"], torch.tensor(
            manifest["layers"][str(layer)]["selected_global_experts"], dtype=torch.int64
        )[tensors[f"{prefix}.route.local_ids"]]):
            raise SystemExit(f"layer-family global/local route mismatch: {layer}")
        expected_output = tensors[f"{prefix}.prefix_after_attention"] + tensors[f"{prefix}.moe.output"]
        if not torch.equal(expected_output, tensors[f"{prefix}.output"]):
            raise SystemExit(f"layer-family final residual mismatch: {layer}")
        if layer < 3 and not torch.equal(tensors[f"{prefix}.output"], tensors[f"layer{layer + 1}.input"]):
            raise SystemExit(f"layer-family sequential handoff mismatch: {layer}")
    print(json.dumps({
        "verified": str(path),
        "tensors": len(tensors),
        "semantic_sha256": manifest["tensor_semantic_sha256"],
        "selected_experts": {layer: data[1] for layer, data in expected.items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
