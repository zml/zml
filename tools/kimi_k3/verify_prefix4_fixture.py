#!/usr/bin/env python3
"""Verify the locked Milestone 15 sequential prefix fixture without inference."""

from __future__ import annotations

import hashlib
import json

import torch
from safetensors import safe_open

from export_layer0_prefix_reference import semantic_sha256
from export_prefix4_reference import (
    LAYER0_FIXTURE,
    LAYER_FAMILY_FILE_SHA256,
    LAYER_FAMILY_FIXTURE,
    LAYER_FAMILY_SEMANTIC_SHA256,
    OUTPUT,
)
from export_reference import sha256_file, tensor_bytes


def load_small(path, names):
    with safe_open(path, framework="pt", device="cpu") as source:
        return {name: source.get_tensor(name) for name in names}


def main() -> None:
    manifest_path = OUTPUT / "s4-prefix4-head-reference.json"
    manifest = json.loads(manifest_path.read_text())
    path = OUTPUT / manifest["tensor_file"]
    if sha256_file(path) != manifest["tensor_file_sha256"]:
        raise SystemExit("prefix4 fixture file hash mismatch")
    if manifest.get("layer_selection") != [0, 1, 2, 3] or manifest.get("layer_stop") != 4:
        raise SystemExit("prefix4 layer selection mismatch")
    if manifest.get("cpu_inference_fallback") is not False:
        raise SystemExit("prefix4 fixture permits CPU inference fallback")
    if manifest.get("prefill_greedy_token") != manifest.get("decode_greedy_token"):
        raise SystemExit("prefix4 prefill/decode greedy token mismatch")

    layer0_manifest = json.loads(LAYER0_FIXTURE.with_suffix(".json").read_text())
    family_manifest = json.loads(LAYER_FAMILY_FIXTURE.with_suffix(".json").read_text())
    sources = manifest.get("source_fixtures", {})
    if sources.get("layer0") != {
        "file_sha256": layer0_manifest["tensor_file_sha256"],
        "semantic_sha256": layer0_manifest["tensor_semantic_sha256"],
    }:
        raise SystemExit("prefix4 layer-0 source provenance mismatch")
    if sources.get("layer_families") != {
        "file_sha256": LAYER_FAMILY_FILE_SHA256,
        "semantic_sha256": LAYER_FAMILY_SEMANTIC_SHA256,
    }:
        raise SystemExit("prefix4 layer-family source provenance mismatch")
    if (
        family_manifest["tensor_file_sha256"] != LAYER_FAMILY_FILE_SHA256
        or family_manifest["tensor_semantic_sha256"] != LAYER_FAMILY_SEMANTIC_SHA256
    ):
        raise SystemExit("prefix4 layer-family source lock mismatch")

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as values:
        for name in values.keys():
            value = values.get_tensor(name).contiguous()
            tensors[name] = value
            record = manifest["tensors"].get(name)
            dtype = str(value.dtype).removeprefix("torch.")
            if record is None or list(value.shape) != record["shape"] or dtype != record["dtype"]:
                raise SystemExit(f"prefix4 tensor contract mismatch: {name}")
            if hashlib.sha256(tensor_bytes(value)).hexdigest() != record["sha256"]:
                raise SystemExit(f"prefix4 tensor hash mismatch: {name}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise SystemExit(f"non-finite prefix4 tensor: {name}")
    if len(tensors) != 104 or semantic_sha256(tensors) != manifest["tensor_semantic_sha256"]:
        raise SystemExit("prefix4 aggregate semantic mismatch")

    handoffs = (
        ("prefix.layer0.decode.output", "layer1.decode.input"),
        ("layer1.decode.output", "layer2.decode.input"),
        ("layer2.decode.output", "layer3.decode.input"),
    )
    for left, right in handoffs:
        if not torch.equal(tensors[left], tensors[right]):
            raise SystemExit(f"prefix4 sequential decode handoff mismatch: {left} -> {right}")

    sources_expected = load_small(
        LAYER_FAMILY_FIXTURE,
        {"layer1.output", "layer2.output", "layer3.output"},
    )
    layer0_expected = load_small(LAYER0_FIXTURE, {"prefix.layer0.out"})
    expected_layers = {0: layer0_expected["prefix.layer0.out"], **{
        layer: sources_expected[f"layer{layer}.output"] for layer in (1, 2, 3)
    }}
    for layer, expected in expected_layers.items():
        warm = tensors[f"prefix.layer0.warm.output"] if layer == 0 else tensors[f"layer{layer}.warm.output"]
        decode = tensors[f"prefix.layer0.decode.output"] if layer == 0 else tensors[f"layer{layer}.decode.output"]
        recomposed = torch.cat((warm, decode), dim=1)
        if not torch.allclose(recomposed.float(), expected.float(), atol=5e-2, rtol=2e-2):
            raise SystemExit(f"prefix4 warm/decode recomposition mismatch: layer {layer}")

    if not torch.equal(
        tensors["prefix.greedy_token"],
        tensors["prefix.logits"][:, -1].argmax(-1),
    ):
        raise SystemExit("prefix4 prefill greedy token is not logits argmax")
    if not torch.equal(
        tensors["decode.greedy_token"],
        tensors["decode.logits"][:, -1].argmax(-1),
    ):
        raise SystemExit("prefix4 decode greedy token is not logits argmax")
    if int(tensors["prefix.greedy_token"].item()) != manifest["prefill_greedy_token"]:
        raise SystemExit("prefix4 manifest greedy token mismatch")

    print(json.dumps({
        "verified": str(path),
        "tensors": len(tensors),
        "semantic_sha256": manifest["tensor_semantic_sha256"],
        "prefill_greedy_token": manifest["prefill_greedy_token"],
        "decode_greedy_token": manifest["decode_greedy_token"],
        "sequential_handoffs": len(handoffs),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
