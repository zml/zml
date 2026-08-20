#!/usr/bin/env python3
"""Verify the immutable Milestone-5 fixture and its independent checks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


ROOT = Path("/dev/shm/kimi-k3")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_sha256(tensors: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture-dir", type=Path, default=ROOT / "artifacts/fixtures/milestone-5"
    )
    parser.add_argument(
        "--lock", type=Path, default=ROOT / "zml/docs/kimi_k3/milestone-5-fixture-lock.json"
    )
    args = parser.parse_args()

    lock = json.loads(args.lock.read_text())
    manifest = json.loads((args.fixture_dir / "primitive-reference.json").read_text())
    tensor_path = args.fixture_dir / manifest["tensor_file"]
    actual_file_hash = sha256_file(tensor_path)
    if manifest["tensor_file_sha256"] != actual_file_hash:
        raise SystemExit("primitive manifest file hash is stale")
    if len(manifest["tensors"]) != lock["tensor_count"]:
        raise SystemExit("primitive tensor count differs from lock")
    if not all(check["passed"] for check in manifest["official_vs_numpy"].values()):
        raise SystemExit("official/NumPy comparison failed")

    semantic_tensors: dict[str, np.ndarray] = {}
    with safe_open(tensor_path, framework="np") as tensors:
        if set(tensors.keys()) != set(manifest["tensors"]):
            raise SystemExit("safetensors keys differ from manifest")
        for name, record in manifest["tensors"].items():
            value = np.ascontiguousarray(tensors.get_tensor(name))
            semantic_tensors[name] = value
            digest = hashlib.sha256(value.tobytes()).hexdigest()
            if digest != record["sha256"]:
                raise SystemExit(f"tensor hash mismatch: {name}")
    actual_semantic_hash = semantic_sha256(semantic_tensors)
    if actual_semantic_hash != lock["tensor_semantic_sha256"]:
        raise SystemExit(
            "primitive semantic hash mismatch: "
            f"{actual_semantic_hash} != {lock['tensor_semantic_sha256']}"
        )
    if manifest["tensor_semantic_sha256"] != actual_semantic_hash:
        raise SystemExit("primitive manifest semantic hash is stale")

    for name, expected in lock["selected_real_slice_sha256"].items():
        if manifest["tensors"][name]["sha256"] != expected:
            raise SystemExit(f"real checkpoint slice changed: {name}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "tensor_count": lock["tensor_count"],
                "tensor_file_sha256": actual_file_hash,
                "tensor_semantic_sha256": actual_semantic_hash,
                "official_checks": len(manifest["official_vs_numpy"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
