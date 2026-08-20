#!/usr/bin/env python3
"""Verify saved Milestone 3 fixtures without importing Moonshot or FLA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from safetensors import safe_open
import torch


EXPECTED = {
    "s0-operators",
    "s1-layer0-len1",
    "s1-layer0-len4",
    "s1-layer0-len8",
    "s1-layer0-len16",
    "s1-layer0-prefill4-decode1",
}


class FixtureError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: Any) -> str:
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def create_lock(root: Path) -> dict[str, Any]:
    aggregate = json.loads((root / "manifest.json").read_text())
    return {
        "schema_version": 1,
        "moonshot_revision": aggregate["moonshot_revision"],
        "policy": "Per-tensor hashes are stable; safetensors container hashes are not locked because header key order is non-canonical.",
        "fixtures": {
            row["name"]: {
                "tensors": {
                    name: {
                        "shape": record["shape"],
                        "dtype": record["dtype"],
                        "sha256": record["sha256"],
                    }
                    for name, record in sorted(
                        json.loads((root / row["manifest"]).read_text())["tensors"].items()
                    )
                }
            }
            for row in aggregate["fixtures"]
        },
    }


def verify(root: Path, lock_path: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    aggregate_path = root / "manifest.json"
    if not aggregate_path.is_file():
        raise FixtureError(f"missing aggregate manifest: {aggregate_path}")
    aggregate = json.loads(aggregate_path.read_text())
    fixture_rows = aggregate.get("fixtures", [])
    names = {row.get("name") for row in fixture_rows}
    if names != EXPECTED:
        raise FixtureError(f"fixture set mismatch: missing={sorted(EXPECTED - names)}, extra={sorted(names - EXPECTED)}")

    verified = []
    for row in fixture_rows:
        name = row["name"]
        manifest_path = root / row["manifest"]
        tensor_path = root / row["safetensors"]
        manifest = json.loads(manifest_path.read_text())
        file_hash = sha256_file(tensor_path)
        if file_hash != row["sha256"] or file_hash != manifest["tensor_file_sha256"]:
            raise FixtureError(f"file hash mismatch: {name}")
        if not manifest.get("numeric_hashes_stable") or manifest.get("repeat_runs") != 2:
            raise FixtureError(f"fixture lacks two-run stability proof: {name}")
        expected_tensors = manifest["tensors"]
        with safe_open(tensor_path, framework="pt", device="cpu") as tensors:
            actual_names = set(tensors.keys())
            if actual_names != set(expected_tensors):
                raise FixtureError(f"tensor-name mismatch: {name}")
            for tensor_name in actual_names:
                value = tensors.get_tensor(tensor_name)
                record = expected_tensors[tensor_name]
                if list(value.shape) != record["shape"] or str(value.dtype).removeprefix("torch.") != record["dtype"]:
                    raise FixtureError(f"shape/dtype mismatch: {name}:{tensor_name}")
                if tensor_sha256(value) != record["sha256"]:
                    raise FixtureError(f"tensor hash mismatch: {name}:{tensor_name}")
        if name == "s0-operators":
            failed = [
                check_name
                for check_name, result in manifest["comparisons"].items()
                if not result.get("passed")
            ]
            if failed:
                raise FixtureError(f"saved S0 comparison failures: {failed}")
        if name.endswith("prefill4-decode1") and not manifest.get("cache_handoff_exact"):
            raise FixtureError("saved continuation cache handoff is not exact")
        verified.append(
            {
                "fixture": name,
                "sha256": file_hash,
                "tensors": len(expected_tensors),
                "bytes": tensor_path.stat().st_size,
            }
        )
    if lock_path is not None:
        lock = json.loads(lock_path.read_text())
        current = create_lock(root)
        if current["moonshot_revision"] != lock.get("moonshot_revision"):
            raise FixtureError("fixture lock Moonshot revision mismatch")
        if current["fixtures"] != lock.get("fixtures"):
            raise FixtureError("fixture tensor records differ from the versioned lock")
    return {
        "schema_version": 1,
        "status": "PASS",
        "versioned_lock": str(lock_path.resolve()) if lock_path is not None else None,
        "fixtures": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fixture_dir",
        nargs="?",
        type=Path,
        default=Path("/dev/shm/kimi-k3/artifacts/fixtures/milestone-3"),
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("docs/kimi_k3/milestone-3-fixture-lock.json"),
    )
    parser.add_argument("--write-lock", action="store_true")
    args = parser.parse_args()
    if args.write_lock:
        lock = create_lock(args.fixture_dir.resolve())
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        args.lock.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(verify(args.fixture_dir, args.lock), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
