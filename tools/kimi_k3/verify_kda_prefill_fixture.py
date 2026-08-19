#!/usr/bin/env python3
"""Verify the Milestone 8 KDA prefill fixture without running inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from export_kda_prefill_reference import LENGTHS, OUTPUT
from export_kda_decode_reference import semantic_sha256, sha256_file


def main() -> None:
    manifest_path = OUTPUT / "kda-prefill-reference.json"
    manifest = json.loads(manifest_path.read_text())
    tensor_path = OUTPUT / manifest["tensor_file"]
    if manifest["lengths"] != list(LENGTHS) or manifest["split_points"] != 25:
        raise SystemExit("invalid KDA prefill length/split contract")
    if sha256_file(tensor_path) != manifest["tensor_file_sha256"]:
        raise SystemExit("KDA prefill fixture file hash mismatch")
    tensors: dict[str, np.ndarray] = {}
    with safe_open(tensor_path, framework="np") as values:
        for name in values.keys():
            value = np.ascontiguousarray(values.get_tensor(name))
            tensors[name] = value
            expected = manifest["tensors"].get(name)
            if expected is None or list(value.shape) != expected["shape"] or str(value.dtype) != expected["dtype"]:
                raise SystemExit(f"KDA prefill tensor contract mismatch: {name}")
            if hashlib.sha256(value.tobytes()).hexdigest() != expected["sha256"]:
                raise SystemExit(f"KDA prefill tensor hash mismatch: {name}")
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise SystemExit(f"non-finite KDA prefill tensor: {name}")
    if set(tensors) != set(manifest["tensors"]):
        raise SystemExit("KDA prefill tensor inventory mismatch")
    if semantic_sha256(tensors) != manifest["tensor_semantic_sha256"]:
        raise SystemExit("KDA prefill aggregate semantic hash mismatch")
    failed = [name for name, result in manifest["official_checks"].items() if not result["passed"]]
    if failed:
        raise SystemExit(f"official KDA prefill comparisons failed: {failed}")
    print(
        json.dumps(
            {
                "verified": str(tensor_path),
                "tensors": len(tensors),
                "official_checks": len(manifest["official_checks"]),
                "lengths": manifest["lengths"],
                "split_points": manifest["split_points"],
                "semantic_sha256": manifest["tensor_semantic_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
