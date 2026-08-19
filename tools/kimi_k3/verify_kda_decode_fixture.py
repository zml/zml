#!/usr/bin/env python3
"""Verify the Milestone 7 KDA decode fixture without running inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from export_kda_decode_reference import OUTPUT, STEPS, semantic_sha256, sha256_file


def main() -> None:
    manifest_path = OUTPUT / "kda-decode-reference.json"
    manifest = json.loads(manifest_path.read_text())
    tensor_path = OUTPUT / manifest["tensor_file"]
    if manifest["steps"] != STEPS or manifest["state_layout"] != "batch,head,value,key":
        raise SystemExit("invalid KDA decode fixture contract")
    if sha256_file(tensor_path) != manifest["tensor_file_sha256"]:
        raise SystemExit("KDA fixture file hash mismatch")
    tensors: dict[str, np.ndarray] = {}
    with safe_open(tensor_path, framework="np") as values:
        for name in values.keys():
            value = np.ascontiguousarray(values.get_tensor(name))
            tensors[name] = value
            expected = manifest["tensors"].get(name)
            if expected is None:
                raise SystemExit(f"unmanifested KDA tensor: {name}")
            if list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]:
                raise SystemExit(f"KDA tensor contract mismatch: {name}")
            if hashlib.sha256(value.tobytes()).hexdigest() != expected["sha256"]:
                raise SystemExit(f"KDA tensor semantic hash mismatch: {name}")
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise SystemExit(f"non-finite KDA tensor: {name}")
    if set(tensors) != set(manifest["tensors"]):
        raise SystemExit("KDA fixture tensor inventory mismatch")
    if semantic_sha256(tensors) != manifest["tensor_semantic_sha256"]:
        raise SystemExit("KDA fixture aggregate semantic hash mismatch")
    failed = [name for name, result in manifest["official_vs_readable"].items() if not result["passed"]]
    if failed:
        raise SystemExit(f"official/readable KDA comparisons failed: {failed}")
    numpy_failed = [name for name, result in manifest["numpy_vs_readable"].items() if not result["passed"]]
    if numpy_failed:
        raise SystemExit(f"NumPy/readable KDA comparisons failed: {numpy_failed}")
    print(
        json.dumps(
            {
                "verified": str(tensor_path),
                "tensors": len(tensors),
                "steps": STEPS,
                "official_checks": len(manifest["official_vs_readable"]),
                "numpy_checks": len(manifest["numpy_vs_readable"]),
                "semantic_sha256": manifest["tensor_semantic_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
