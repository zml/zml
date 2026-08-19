#!/usr/bin/env python3
"""Verify the Milestone 9 one-layer S2 fixture without running inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from export_layer0_prefix_reference import DEFAULT_OUTPUT, TOKEN_IDS, semantic_sha256
from export_reference import sha256_file, tensor_bytes


def main() -> None:
    manifest_path = DEFAULT_OUTPUT / "s2-layer0-prefix-len4.json"
    manifest = json.loads(manifest_path.read_text())
    tensor_path = DEFAULT_OUTPUT / manifest["tensor_file"]
    if manifest["token_ids"] != list(TOKEN_IDS) or manifest["greedy_token"] != 4202:
        raise SystemExit("invalid S2 token/greedy contract")
    if manifest["cpu_inference_fallback"] or manifest["layer_stop"] != 1:
        raise SystemExit("invalid S2 execution contract")
    if sha256_file(tensor_path) != manifest["tensor_file_sha256"]:
        raise SystemExit("S2 prefix fixture file hash mismatch")

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(tensor_path, framework="pt", device="cpu") as values:
        for name in values.keys():
            value = values.get_tensor(name).contiguous()
            tensors[name] = value
            expected = manifest["tensors"].get(name)
            dtype = str(value.dtype).removeprefix("torch.")
            if expected is None or list(value.shape) != expected["shape"] or dtype != expected["dtype"]:
                raise SystemExit(f"S2 tensor contract mismatch: {name}")
            if hashlib.sha256(tensor_bytes(value)).hexdigest() != expected["sha256"]:
                raise SystemExit(f"S2 tensor hash mismatch: {name}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise SystemExit(f"non-finite S2 tensor: {name}")
    if set(tensors) != set(manifest["tensors"]):
        raise SystemExit("S2 tensor inventory mismatch")
    if semantic_sha256(tensors) != manifest["tensor_semantic_sha256"]:
        raise SystemExit("S2 aggregate semantic hash mismatch")
    print(
        json.dumps(
            {
                "verified": str(tensor_path),
                "tensors": len(tensors),
                "greedy_token": manifest["greedy_token"],
                "semantic_sha256": manifest["tensor_semantic_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
