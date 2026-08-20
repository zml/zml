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
        decode = details.get("decode", {})
        if (
            decode.get("warm_tokens") != 3
            or decode.get("decode_tokens") != 1
            or decode.get("route_count") != 16
            or not isinstance(decode.get("route_comparison", {}).get("sets_match"), bool)
            or decode["route_comparison"].get("overlap_count", -1) < 0
            or decode["route_comparison"].get("union_count", -1) < 1
            or not decode.get("comparisons")
            or not all(report.get("passed") for report in decode["comparisons"].values())
        ):
            raise SystemExit(f"layer-family decode contract mismatch: {layer}")

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
    if len(tensors) != 240 or semantic_sha256(tensors) != manifest["tensor_semantic_sha256"]:
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
        decode = f"{prefix}.decode"
        if not torch.equal(
            tensors[f"{decode}.route.global_ids"],
            torch.tensor(
                manifest["layers"][str(layer)]["selected_global_experts"], dtype=torch.int64
            )[tensors[f"{decode}.route.local_ids"]],
        ):
            raise SystemExit(f"layer-family decode global/local route mismatch: {layer}")
        prefill_ids = set(tensors[f"{prefix}.route.global_ids"][:, -1:].flatten().tolist())
        decode_ids = set(tensors[f"{decode}.route.global_ids"].flatten().tolist())
        route_comparison = manifest["layers"][str(layer)]["decode"]["route_comparison"]
        if (
            route_comparison["sets_match"] != (prefill_ids == decode_ids)
            or route_comparison["overlap_count"] != len(prefill_ids & decode_ids)
            or route_comparison["union_count"] != len(prefill_ids | decode_ids)
            or route_comparison["prefill_only"] != sorted(prefill_ids - decode_ids)
            or route_comparison["decode_only"] != sorted(decode_ids - prefill_ids)
        ):
            raise SystemExit(f"layer-family decode route evidence mismatch: {layer}")
        decode_expected = (
            tensors[f"{decode}.prefix_after_attention"] + tensors[f"{decode}.moe.output"]
        )
        if not torch.equal(decode_expected, tensors[f"{decode}.output"]):
            raise SystemExit(f"layer-family decode final residual mismatch: {layer}")
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
