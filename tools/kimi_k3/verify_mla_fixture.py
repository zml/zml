#!/usr/bin/env python3
"""Verify Milestone 12's expanded Gated NoPE MLA fixture without inference."""

from __future__ import annotations

import hashlib
import json

import torch
from safetensors import safe_open

from export_layer0_prefix_reference import semantic_sha256
from export_mla_reference import LENGTHS, OUTPUT
from export_reference import sha256_file, tensor_bytes


def main() -> None:
    manifest = json.loads((OUTPUT / "expanded-mla-reference.json").read_text())
    tensor_path = OUTPUT / manifest["tensor_file"]
    if sha256_file(tensor_path) != manifest["tensor_file_sha256"]:
        raise SystemExit("MLA fixture file hash mismatch")
    if manifest["lengths"] != list(LENGTHS) or manifest["decode_past_length"] != 4:
        raise SystemExit("MLA sequence/decode contract mismatch")
    if len(manifest["tensors"]) != 150 or manifest["boundary_count_per_case"] != 28:
        raise SystemExit("MLA tensor/boundary inventory mismatch")

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(tensor_path, framework="pt", device="cpu") as values:
        for name in values.keys():
            value = values.get_tensor(name).contiguous()
            tensors[name] = value
            expected = manifest["tensors"].get(name)
            dtype = str(value.dtype).removeprefix("torch.")
            if expected is None or list(value.shape) != expected["shape"] or dtype != expected["dtype"]:
                raise SystemExit(f"MLA tensor contract mismatch: {name}")
            if hashlib.sha256(tensor_bytes(value)).hexdigest() != expected["sha256"]:
                raise SystemExit(f"MLA tensor hash mismatch: {name}")
            allows_negative_infinity = name.endswith("causal_mask") or name.endswith("masked_scores")
            if value.is_floating_point() and not allows_negative_infinity and not torch.isfinite(value).all():
                raise SystemExit(f"non-finite MLA tensor: {name}")
    if set(tensors) != set(manifest["tensors"]):
        raise SystemExit("MLA tensor inventory mismatch")
    if semantic_sha256(tensors) != manifest["tensor_semantic_sha256"]:
        raise SystemExit("MLA aggregate semantic hash mismatch")

    for length in LENGTHS:
        prefix = f"len{length}"
        if not torch.equal(
            torch.cat((tensors[f"{prefix}.q_pass"], tensors[f"{prefix}.q_extra"]), -1),
            tensors[f"{prefix}.query"],
        ):
            raise SystemExit(f"MLA NoPE query join mismatch: {prefix}")
        if not torch.equal(tensors[f"{prefix}.cache_key"], tensors[f"{prefix}.key_new"]):
            raise SystemExit(f"MLA prefill key cache mismatch: {prefix}")
        if not torch.equal(tensors[f"{prefix}.cache_value"], tensors[f"{prefix}.value_new"]):
            raise SystemExit(f"MLA prefill value cache mismatch: {prefix}")
        probabilities = tensors[f"{prefix}.probabilities"].float()
        if not torch.allclose(probabilities.sum(-1), torch.ones_like(probabilities.sum(-1)), atol=2e-3, rtol=2e-3):
            raise SystemExit(f"MLA probability normalization mismatch: {prefix}")
        forbidden = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
        if torch.count_nonzero(probabilities[..., forbidden]):
            raise SystemExit(f"MLA causal probability mismatch: {prefix}")
        if not torch.allclose(
            tensors[f"{prefix}.output"].float(),
            tensors[f"{prefix}.official_output"].float(),
            atol=0.02,
            rtol=0.02,
        ):
            raise SystemExit(f"MLA pinned official output mismatch: {prefix}")

    if not torch.equal(tensors["decode.cache_key"][..., :4, :], tensors["decode.past_key"]):
        raise SystemExit("MLA decode key-cache prefix mismatch")
    if not torch.equal(tensors["decode.cache_value"][..., :4, :], tensors["decode.past_value"]):
        raise SystemExit("MLA decode value-cache prefix mismatch")
    if not torch.equal(tensors["decode.cache_key"][..., 4:, :], tensors["decode.key_new"]):
        raise SystemExit("MLA decode key-cache append mismatch")
    if not torch.equal(tensors["decode.cache_value"][..., 4:, :], tensors["decode.value_new"]):
        raise SystemExit("MLA decode value-cache append mismatch")
    if not torch.allclose(
        tensors["decode.output"].float(),
        tensors["decode.official_output"].float(),
        atol=0.02,
        rtol=0.02,
    ):
        raise SystemExit("MLA pinned official decode output mismatch")
    print(
        json.dumps(
            {
                "verified": str(tensor_path),
                "tensors": len(tensors),
                "prefill_lengths": list(LENGTHS),
                "decode_past_length": 4,
                "semantic_sha256": manifest["tensor_semantic_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
