#!/usr/bin/env python3
"""Freeze authoritative Hugging Face shard metadata into an offline hash receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

SHARD_COUNT = 96
SHARD_RE = re.compile(r"^model-(\d{5})-of-000096\.safetensors$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ReceiptError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise ReceiptError(f"expected JSON object: {path}")
    return document


def expected_shards(index: dict[str, Any]) -> list[str]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ReceiptError("checkpoint index has no weight_map object")
    names = sorted(set(weight_map.values()))
    expected = [
        f"model-{number:05d}-of-000096.safetensors"
        for number in range(1, SHARD_COUNT + 1)
    ]
    if names != expected or any(not SHARD_RE.fullmatch(name) for name in names):
        raise ReceiptError("checkpoint index does not reference the exact 96-shard set")
    return names


def read_hub_metadata(path: Path) -> tuple[str, str, float]:
    lines = path.read_text().splitlines()
    if len(lines) != 3:
        raise ReceiptError(f"invalid Hugging Face download metadata: {path}")
    revision, etag, completed = lines
    revision = revision.lower()
    etag = etag.strip('"').lower()
    if not REVISION_RE.fullmatch(revision):
        raise ReceiptError(f"invalid revision in {path}")
    if not SHA256_RE.fullmatch(etag):
        raise ReceiptError(f"shard etag is not a SHA-256 digest: {path}")
    try:
        completed_at = float(completed)
    except ValueError as error:
        raise ReceiptError(f"invalid completion time in {path}") from error
    if not math.isfinite(completed_at) or completed_at <= 0:
        raise ReceiptError(f"invalid completion time in {path}")
    return revision, etag, completed_at


def build_receipt(model_dir: Path, baseline_manifest: Path | None = None) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    index_path = model_dir / "model.safetensors.index.json"
    config_path = model_dir / "config.json"
    if not index_path.is_file() or not config_path.is_file():
        raise ReceiptError("checkpoint requires config.json and model.safetensors.index.json")
    index = load_object(index_path)
    names = expected_shards(index)
    baseline_files: dict[str, Any] = {}
    if baseline_manifest is not None:
        baseline_files = load_object(baseline_manifest).get("files", {})
        if not isinstance(baseline_files, dict):
            raise ReceiptError("baseline manifest has no files object")

    metadata_dir = model_dir / ".cache" / "huggingface" / "download"
    files: dict[str, dict[str, Any]] = {}
    revisions: set[str] = set()
    completed_times: list[float] = []
    for name in names:
        shard = model_dir / name
        metadata = metadata_dir / f"{name}.metadata"
        if not shard.is_file():
            raise ReceiptError(f"missing checkpoint shard: {name}")
        if not metadata.is_file():
            raise ReceiptError(f"missing Hugging Face metadata: {metadata}")
        revision, expected_hash, completed_at = read_hub_metadata(metadata)
        revisions.add(revision)
        completed_times.append(completed_at)
        baseline = baseline_files.get(name)
        if baseline is not None:
            if not isinstance(baseline, dict) or baseline.get("sha256") != expected_hash:
                raise ReceiptError(f"download metadata disagrees with baseline hash: {name}")
            if baseline.get("bytes") is not None and baseline["bytes"] != shard.stat().st_size:
                raise ReceiptError(f"downloaded shard size disagrees with baseline: {name}")
        files[name] = {
            "bytes": shard.stat().st_size,
            "sha256": expected_hash,
        }

    if len(revisions) != 1:
        raise ReceiptError(f"checkpoint shards came from mixed revisions: {sorted(revisions)}")
    revision = next(iter(revisions))
    return {
        "schema_version": 1,
        "repository": "moonshotai/Kimi-K3",
        "source_revision": revision,
        "hash_source": "huggingface_hub_download_metadata_etag",
        "offline": True,
        "source_files_modified": False,
        "config_sha256": sha256(config_path),
        "index_sha256": sha256(index_path),
        "index_tensor_payload_bytes": int(index.get("metadata", {}).get("total_size", 0)),
        "physical_shard_bytes": sum(record["bytes"] for record in files.values()),
        "download_completed_unix_min": min(completed_times),
        "download_completed_unix_max": max(completed_times),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(args.model, args.baseline_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        f"KIMI_K3_CHECKPOINT_RECEIPT_PASS revision={receipt['source_revision']} "
        f"shards={len(receipt['files'])} bytes={receipt['physical_shard_bytes']}"
    )


if __name__ == "__main__":
    main()
