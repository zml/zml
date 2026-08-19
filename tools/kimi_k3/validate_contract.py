#!/usr/bin/env python3
"""Validate Kimi K3's static text contract and inventory local checkpoint data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


class ContractError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    text = config["text_config"]
    linear = text["linear_attn_config"]
    layers = text["num_hidden_layers"]
    full = linear["full_attn_layers"]
    kda = linear["kda_layers"]
    expected = set(range(1, layers + 1))

    checks = {
        "num_hidden_layers": (layers, 93),
        "hidden_size": (text["hidden_size"], 7168),
        "vocab_size": (text["vocab_size"], 163840),
        "num_attention_heads": (text["num_attention_heads"], 96),
        "num_experts": (text["num_experts"], 896),
        "num_experts_per_token": (text["num_experts_per_token"], 16),
        "first_k_dense_replace": (text["first_k_dense_replace"], 1),
        "routed_expert_hidden_size": (text["routed_expert_hidden_size"], 3584),
        "quant_group_size": (text["quantization_config"]["config_groups"]["group_0"]["weights"]["group_size"], 32),
    }
    for name, (actual, expected_value) in checks.items():
        if actual != expected_value:
            raise ContractError(f"{name}: {actual!r} != {expected_value!r}")
    if set(full) & set(kda) or set(full) | set(kda) != expected:
        raise ContractError("one-based MLA and KDA schedules must be a disjoint partition of 1..93")
    if len(full) != 24 or len(kda) != 69:
        raise ContractError(f"layer family counts must be 24 MLA/69 KDA, got {len(full)}/{len(kda)}")
    if full != sorted(full) or kda != sorted(kda):
        raise ContractError("layer schedules must be strictly ordered")
    expected_boundaries = {0: "kda_dense", 1: "kda_moe", 2: "kda_moe", 3: "mla_moe", 91: "mla_moe", 92: "mla_moe"}
    full_zero = {layer - 1 for layer in full}
    actual_boundaries = {
        layer: ("mla" if layer in full_zero else "kda") + ("_dense" if layer == 0 else "_moe")
        for layer in expected_boundaries
    }
    if actual_boundaries != expected_boundaries:
        raise ContractError(f"boundary schedule mismatch: {actual_boundaries}")
    return {
        "layers": layers,
        "mla_zero_based": sorted(full_zero),
        "kda_zero_based": sorted({layer - 1 for layer in kda}),
        "boundaries": actual_boundaries,
    }


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ContractError(f"truncated safetensors length: {path}")
        (header_length,) = struct.unpack("<Q", raw_length)
        if header_length > 256 * 1024 * 1024:
            raise ContractError(f"unreasonable safetensors header length {header_length}: {path}")
        header = stream.read(header_length)
        if len(header) != header_length:
            raise ContractError(f"truncated safetensors header: {path}")
    return json.loads(header)


def validate_source_map(path: Path) -> int:
    document = yaml.safe_load(path.read_text())
    if document.get("schema_version") != 1:
        raise ContractError("source-map schema_version must be 1")
    required = {"official", "zml", "checkpoint_prefix", "decision", "test"}
    entries = document.get("entries", [])
    if len(entries) < 10:
        raise ContractError("source-map must cover every planned operation family")
    for index, entry in enumerate(entries):
        missing = required - set(entry)
        if missing:
            raise ContractError(f"source-map entry {index} missing {sorted(missing)}")
    return len(entries)


def build_inventory(workspace: Path, output_dir: Path) -> dict[str, Any]:
    checkpoint = workspace / "moonshot" / "kimi-k3"
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]
    layer_pattern = re.compile(r"language_model\.model\.layers\.(\d+)\.")
    layer_ids = sorted({int(match.group(1)) for name in weight_map if (match := layer_pattern.search(name))})
    if layer_ids != list(range(93)):
        raise ContractError(f"checkpoint layer IDs are not exactly 0..92: {layer_ids[:3]}..{layer_ids[-3:]}")

    available_headers: dict[str, dict[str, Any]] = {}
    header_tensors: dict[str, dict[str, Any]] = {}
    for shard in sorted(set(weight_map.values())):
        path = checkpoint / shard
        if not path.is_file():
            continue
        header = read_safetensors_header(path)
        tensors = {name: meta for name, meta in header.items() if name != "__metadata__"}
        available_headers[shard] = {"bytes": path.stat().st_size, "tensors": len(tensors), "sha256": sha256(path)}
        for name, meta in tensors.items():
            header_tensors[name] = {"dtype": meta["dtype"], "shape": meta["shape"]}

    shard_counts = Counter(weight_map.values())
    output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = output_dir / "tensor-name-inventory.jsonl.gz"
    with inventory_path.open("wb") as raw_stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_stream, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as stream:
                for name in sorted(weight_map):
                    record = {"name": name, "shard": weight_map[name]}
                    record.update(header_tensors.get(name, {}))
                    stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")

    summary = {
        "schema_version": 1,
        "index_sha256": sha256(index_path),
        "tensor_names": len(weight_map),
        "shards_referenced": len(shard_counts),
        "layer_ids_zero_based": layer_ids,
        "available_shards": available_headers,
        "available_header_tensors": len(header_tensors),
        "inventory": {
            "path": inventory_path.relative_to(workspace).as_posix(),
            "bytes": inventory_path.stat().st_size,
            "sha256": sha256(inventory_path),
        },
    }
    (output_dir / "checkpoint-contract-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def memory_estimates() -> dict[str, Any]:
    bf16 = 2
    heads, key_dim, value_dim = 96, 128, 128
    kda_state = heads * key_dim * value_dim * bf16
    kda_conv = 3 * 12288 * (4 - 1) * bf16
    mla_per_token = (512 + 64) * bf16
    local_shards = 2341216112 + 16990911504 + 16990911504 + 16567501776 + 4697664072
    return {
        "schema_version": 1,
        "formulas": {
            "kda_recurrent_state_bytes_per_batch_layer": "96*128*128*2",
            "kda_conv_tail_upper_bound_bytes_per_batch_layer": "3*12288*(4-1)*2",
            "mla_cache_bytes_per_batch_layer_token": "(512+64)*2",
            "nominal_full_mxfp4_payload_bytes": "2.8e12*0.5 excluding scales/runtime buffers"
        },
        "bytes": {
            "kda_recurrent_state_per_batch_layer": kda_state,
            "kda_conv_tail_upper_bound_per_batch_layer": kda_conv,
            "mla_cache_per_batch_layer_token": mla_per_token,
            "five_local_shards": local_shards,
            "nominal_full_mxfp4_payload": 1_400_000_000_000
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path("/ephemeral/kimi-k3"))
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    config = json.loads((workspace / "moonshot" / "kimi-k3" / "config.json").read_text())
    schedule = validate_config(config)
    source_entries = validate_source_map(workspace / "zml" / "docs" / "kimi_k3" / "source-map.yaml")
    output_dir = workspace / "artifacts" / "contracts"
    summary = build_inventory(workspace, output_dir)
    estimates = memory_estimates()
    (output_dir / "memory-estimates.json").write_text(json.dumps(estimates, indent=2, sort_keys=True) + "\n")
    report = {"status": "PASS", "schedule": schedule, "source_map_entries": source_entries, "inventory": summary, "memory": estimates}
    (output_dir / "milestone-1-validation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
