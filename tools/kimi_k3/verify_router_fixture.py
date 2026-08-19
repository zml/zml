#!/usr/bin/env python3
"""Verify the Milestone 10 router fixture without running inference."""

from __future__ import annotations

import hashlib
import json

import torch
from safetensors import safe_open

from export_layer0_prefix_reference import semantic_sha256
from export_reference import sha256_file, tensor_bytes
from export_router_reference import OUTPUT


def main() -> None:
    manifest = json.loads((OUTPUT / "router-reference.json").read_text())
    tensor_path = OUTPUT / manifest["tensor_file"]
    if sha256_file(tensor_path) != manifest["tensor_file_sha256"]:
        raise SystemExit("router fixture file hash mismatch")
    if manifest["official_checked_rows"] != 8 or not manifest["bias_not_used_as_mixture_weight"]:
        raise SystemExit("router official/adversarial contract mismatch")
    if set(manifest["cases"]) != {"real", "tie", "bias", "grouped"}:
        raise SystemExit("router case inventory mismatch")

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(tensor_path, framework="pt", device="cpu") as values:
        for name in values.keys():
            value = values.get_tensor(name).contiguous()
            tensors[name] = value
            expected = manifest["tensors"].get(name)
            dtype = str(value.dtype).removeprefix("torch.")
            if expected is None or list(value.shape) != expected["shape"] or dtype != expected["dtype"]:
                raise SystemExit(f"router tensor contract mismatch: {name}")
            if hashlib.sha256(tensor_bytes(value)).hexdigest() != expected["sha256"]:
                raise SystemExit(f"router tensor hash mismatch: {name}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise SystemExit(f"non-finite router tensor: {name}")
    if set(tensors) != set(manifest["tensors"]):
        raise SystemExit("router tensor inventory mismatch")
    if semantic_sha256(tensors) != manifest["tensor_semantic_sha256"]:
        raise SystemExit("router aggregate semantic hash mismatch")
    if tensors["tie.topk_ids"].flatten().tolist() != list(range(16)):
        raise SystemExit("stable lower-index tie policy mismatch")
    raw_ids = torch.argsort(tensors["bias.raw_scores"], dim=-1, descending=True, stable=True)[..., :16]
    if torch.equal(raw_ids, tensors["bias.topk_ids"]):
        raise SystemExit("bias adversary does not prove selection-only correction")
    for case in ("real", "tie", "bias", "grouped"):
        weights = tensors[f"{case}.topk_weights"]
        scale = manifest["cases"][case]["scaling_factor"]
        if not torch.allclose(weights.sum(-1), torch.full_like(weights.sum(-1), scale), atol=1e-6, rtol=1e-6):
            raise SystemExit(f"normalized router weight sum mismatch: {case}")
    print(
        json.dumps(
            {
                "verified": str(tensor_path),
                "tensors": len(tensors),
                "cases": 4,
                "official_checked_rows": manifest["official_checked_rows"],
                "semantic_sha256": manifest["tensor_semantic_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
