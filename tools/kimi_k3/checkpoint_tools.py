#!/usr/bin/env python3
"""Safe, bounded checkpoint inspection, mini-index, and extraction utilities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any, BinaryIO


TIERS = {
    "S1": {"model-00001-of-000096.safetensors"},
    "S2": {"model-00001-of-000096.safetensors", "model-00094-of-000096.safetensors"},
    "S5_TEXT": {f"model-{number:05d}-of-000096.safetensors" for number in range(1, 95)},
    "S4": {
        "model-00001-of-000096.safetensors",
        "model-00002-of-000096.safetensors",
        "model-00003-of-000096.safetensors",
        "model-00004-of-000096.safetensors",
        "model-00094-of-000096.safetensors",
    },
}
METADATA_FILES = [
    "LICENSE", "config.json", "configuration_kimi_k3.py", "encoding_k3.py",
    "generation_config.json", "media_utils.py", "modeling_kimi_k3.py",
    "modeling_kimi_linear.py", "tiktoken.model", "tokenization_kimi.py",
    "tokenizer_config.json",
]


class CheckpointError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_index(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "model.safetensors.index.json"
    document = json.loads(path.read_text())
    if not isinstance(document.get("weight_map"), dict):
        raise CheckpointError(f"invalid or missing weight_map: {path}")
    return document


def read_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise CheckpointError(f"truncated safetensors prefix: {path}")
        (length,) = struct.unpack("<Q", raw)
        if length > 256 * 1024 * 1024:
            raise CheckpointError(f"unsafe safetensors header length {length}: {path}")
        payload = stream.read(length)
        if len(payload) != length:
            raise CheckpointError(f"truncated safetensors header: {path}")
    return length, json.loads(payload)


def inspect(model_dir: Path) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    index = load_index(model_dir)
    shards = sorted(set(index["weight_map"].values()))
    missing = [name for name in shards if not (model_dir / name).is_file()]
    available = {}
    for name in shards:
        path = model_dir / name
        if path.is_file():
            _, header = read_header(path)
            available[name] = {
                "bytes": path.stat().st_size,
                "tensors": len(header) - ("__metadata__" in header),
            }
    return {
        "schema_version": 1,
        "model_dir": os.fspath(model_dir),
        "tensor_names": len(index["weight_map"]),
        "referenced_shards": len(shards),
        "available_shards": available,
        "missing_shards": missing,
        "valid_for_loading": not missing,
    }


def make_mini(model_dir: Path, output_dir: Path, tier: str) -> dict[str, Any]:
    model_dir, output_dir = model_dir.resolve(), output_dir.resolve()
    if output_dir == model_dir or model_dir in output_dir.parents:
        raise CheckpointError("derived output must be separate from the read-only source checkpoint")
    selected_shards = TIERS[tier]
    index = load_index(model_dir)
    weight_map = {name: shard for name, shard in index["weight_map"].items() if shard in selected_shards}
    referenced = set(weight_map.values())
    if referenced != selected_shards:
        raise CheckpointError(f"tier {tier} did not resolve every expected shard: {sorted(selected_shards - referenced)}")
    missing = [name for name in sorted(referenced) if not (model_dir / name).is_file()]
    if missing:
        raise CheckpointError(f"tier {tier} source shards are missing: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "model.safetensors.index.json"
    if index_path.resolve() == (model_dir / "model.safetensors.index.json").resolve():
        raise CheckpointError("refusing to overwrite source index")
    mini = {"metadata": index.get("metadata", {}), "weight_map": weight_map}
    index_path.write_text(json.dumps(mini, indent=2, sort_keys=True) + "\n")
    for name in sorted(referenced):
        target = output_dir / name
        source = model_dir / name
        if target.exists() or target.is_symlink():
            if target.resolve() != source:
                raise CheckpointError(f"existing derived shard link points elsewhere: {target}")
        else:
            target.symlink_to(source)
    for name in METADATA_FILES:
        source = model_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)

    manifest = {
        "schema_version": 1,
        "tier": tier,
        "source": os.fspath(model_dir),
        "source_index_sha256": sha256(model_dir / "model.safetensors.index.json"),
        "derived_index_sha256": sha256(index_path),
        "tensor_names": len(weight_map),
        "shards": {
            name: {"bytes": (model_dir / name).stat().st_size, "sha256": sha256(model_dir / name)}
            for name in sorted(referenced)
        },
        "source_files_modified": False,
    }
    (output_dir / "provenance.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validation = inspect(output_dir)
    if not validation["valid_for_loading"]:
        raise CheckpointError(f"derived tier failed validation: {validation['missing_shards']}")
    return {"manifest": manifest, "validation": validation}


def _copy_range(source: BinaryIO, output: BinaryIO, start: int, length: int) -> None:
    source.seek(start)
    remaining = length
    while remaining:
        chunk = source.read(min(16 * 1024 * 1024, remaining))
        if not chunk:
            raise CheckpointError("source tensor range ended early")
        output.write(chunk)
        remaining -= len(chunk)


def extract(model_dir: Path, output: Path, names: list[str], max_bytes: int) -> dict[str, Any]:
    model_dir, output = model_dir.resolve(), output.resolve()
    if not names or len(names) != len(set(names)):
        raise CheckpointError("tensor names must be non-empty and unique")
    if model_dir in output.parents:
        raise CheckpointError("extracted fixture must not be written inside the source checkpoint")
    provenance_path = output.with_suffix(output.suffix + ".provenance.json")
    if output.exists():
        if not provenance_path.is_file():
            raise CheckpointError(f"existing fixture has no provenance manifest: {output}")
        manifest = json.loads(provenance_path.read_text())
        if [item["name"] for item in manifest.get("tensors", [])] != names:
            raise CheckpointError(f"existing fixture tensor selection differs: {output}")
        if manifest.get("source") != os.fspath(model_dir) or manifest.get("sha256") != sha256(output):
            raise CheckpointError(f"existing fixture provenance validation failed: {output}")
        if manifest.get("payload_bytes", max_bytes + 1) > max_bytes:
            raise CheckpointError(f"existing fixture exceeds --max-bytes: {output}")
        return {**manifest, "reused": True}
    index = load_index(model_dir)
    selected: list[tuple[str, Path, int, int, dict[str, Any]]] = []
    total = 0
    header_cache: dict[Path, tuple[int, dict[str, Any]]] = {}
    for name in names:
        shard = index["weight_map"].get(name)
        if shard is None:
            raise CheckpointError(f"tensor is absent from index: {name}")
        shard_path = model_dir / shard
        if not shard_path.is_file():
            raise CheckpointError(f"tensor source shard is unavailable: {shard}")
        header_length, header = header_cache.setdefault(shard_path, read_header(shard_path))
        meta = header.get(name)
        if not isinstance(meta, dict) or "data_offsets" not in meta:
            raise CheckpointError(f"tensor is absent from shard header: {name}")
        begin, end = meta["data_offsets"]
        length = end - begin
        total += length
        if total > max_bytes:
            raise CheckpointError(f"selected payload {total} exceeds --max-bytes {max_bytes}")
        selected.append((name, shard_path, 8 + header_length + begin, length, meta))

    output.parent.mkdir(parents=True, exist_ok=True)
    output_header: dict[str, Any] = {"__metadata__": {"source": os.fspath(model_dir), "bounded_extraction": "true"}}
    offset = 0
    for name, _, _, length, meta in selected:
        output_header[name] = {"dtype": meta["dtype"], "shape": meta["shape"], "data_offsets": [offset, offset + length]}
        offset += length
    encoded = json.dumps(output_header, separators=(",", ":"), sort_keys=True).encode()
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with output.open("xb") as destination:
        destination.write(struct.pack("<Q", len(encoded)))
        destination.write(encoded)
        for _, shard_path, start, length, _ in selected:
            with shard_path.open("rb") as source:
                _copy_range(source, destination, start, length)
    manifest = {
        "schema_version": 1,
        "source": os.fspath(model_dir),
        "output": os.fspath(output),
        "payload_bytes": total,
        "file_bytes": output.stat().st_size,
        "sha256": sha256(output),
        "tensors": [{"name": name, "source_shard": path.name, "bytes": length, "dtype": meta["dtype"], "shape": meta["shape"]} for name, path, _, length, meta in selected],
    }
    provenance_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("model_dir", type=Path)
    mini_parser = sub.add_parser("make-mini")
    mini_parser.add_argument("model_dir", type=Path)
    mini_parser.add_argument("output_dir", type=Path)
    mini_parser.add_argument("--tier", choices=sorted(TIERS), required=True)
    extract_parser = sub.add_parser("extract")
    extract_parser.add_argument("model_dir", type=Path)
    extract_parser.add_argument("output", type=Path)
    extract_parser.add_argument("names", nargs="+")
    extract_parser.add_argument("--max-bytes", type=int, default=1 << 30)
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect(args.model_dir)
    elif args.command == "make-mini":
        result = make_mini(args.model_dir, args.output_dir, args.tier)
    else:
        result = extract(args.model_dir, args.output, args.names, args.max_bytes)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "inspect" and not result["valid_for_loading"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
