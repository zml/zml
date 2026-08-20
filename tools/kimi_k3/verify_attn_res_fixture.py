#!/usr/bin/env python3
"""Verify Milestone-6 Attention Residual fixture semantics and masks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


ROOT = Path("/dev/shm/kimi-k3")


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", type=Path, default=ROOT / "artifacts/fixtures/milestone-6")
    parser.add_argument("--lock", type=Path, default=ROOT / "zml/docs/kimi_k3/milestone-6-fixture-lock.json")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text())
    manifest = json.loads((args.fixture_dir / "attn-res-reference.json").read_text())
    path = args.fixture_dir / manifest["tensor_file"]
    if file_sha256(path) != manifest["tensor_file_sha256"]:
        raise SystemExit("Attention Residual fixture file hash differs from manifest")
    if len(manifest["tensors"]) != lock["tensor_count"]:
        raise SystemExit("Attention Residual tensor count differs from lock")
    if len(manifest["official_vs_numpy"]) != lock["official_comparison_count"]:
        raise SystemExit("official comparison count differs from lock")
    if not all(item["passed"] for item in manifest["official_vs_numpy"].values()):
        raise SystemExit("an official/NumPy Attention Residual comparison failed")

    loaded: dict[str, np.ndarray] = {}
    with safe_open(path, framework="np") as tensors:
        if set(tensors.keys()) != set(manifest["tensors"]):
            raise SystemExit("Attention Residual keys differ from manifest")
        for name, record in manifest["tensors"].items():
            value = np.ascontiguousarray(tensors.get_tensor(name))
            if hashlib.sha256(value.tobytes()).hexdigest() != record["sha256"]:
                raise SystemExit(f"Attention Residual tensor hash mismatch: {name}")
            loaded[name] = value
    semantic = semantic_sha256(loaded)
    if semantic != lock["tensor_semantic_sha256"] or semantic != manifest["tensor_semantic_sha256"]:
        raise SystemExit("Attention Residual semantic hash differs from lock")
    stale_probs = loaded["synthetic.inactive_stale.expected.probabilities"]
    if not np.array_equal(stale_probs[:, 1:3], np.zeros_like(stale_probs[:, 1:3])):
        raise SystemExit("inactive stale block slots received non-zero probability")
    one_probs = loaded["synthetic.one_source.expected.probabilities"]
    if not np.array_equal(one_probs[:, :3], np.zeros_like(one_probs[:, :3])):
        raise SystemExit("empty block slots received non-zero one-source probability")
    print(json.dumps({"status": "PASS", "tensors": len(loaded), "semantic_sha256": semantic, "official_checks": len(manifest["official_vs_numpy"])}, sort_keys=True))


if __name__ == "__main__":
    main()
