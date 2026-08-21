#!/usr/bin/env python3
"""Offline Kimi K3 full-index, loading, and distributed-readiness preflight."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPERT_COUNT = 896
LAYER_COUNT = 93
KDA_COUNT = 69
MLA_COUNT = 24
MOE_LAYER_COUNT = 92
SHARD_COUNT = 96

HEAD_TENSORS = {
    "language_model.lm_head.weight",
    "language_model.model.embed_tokens.weight",
    "language_model.model.norm.weight",
    "language_model.model.output_attn_res_norm.weight",
    "language_model.model.output_attn_res_proj.weight",
}
COMMON_SUFFIXES = {
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attention_res_norm.weight",
    "self_attention_res_proj.weight",
    "mlp_res_norm.weight",
    "mlp_res_proj.weight",
}
KDA_SUFFIXES = {
    "self_attn.A_log",
    "self_attn.dt_bias",
    "self_attn.q_conv1d.weight",
    "self_attn.k_conv1d.weight",
    "self_attn.v_conv1d.weight",
    "self_attn.o_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.b_proj.weight",
    "self_attn.f_a_proj.weight",
    "self_attn.f_b_proj.weight",
    "self_attn.g_proj.weight",
    "self_attn.o_proj.weight",
}
MLA_SUFFIXES = {
    "self_attn.q_a_proj.weight",
    "self_attn.q_a_layernorm.weight",
    "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_b_proj.weight",
    "self_attn.g_proj.weight",
    "self_attn.o_proj.weight",
}
DENSE_SUFFIXES = {
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
}
MOE_SUFFIXES = {
    "block_sparse_moe.gate.weight",
    "block_sparse_moe.gate.e_score_correction_bias",
    "block_sparse_moe.routed_expert_down_proj.weight",
    "block_sparse_moe.routed_expert_norm.weight",
    "block_sparse_moe.routed_expert_up_proj.weight",
    "block_sparse_moe.shared_experts.gate_proj.weight",
    "block_sparse_moe.shared_experts.up_proj.weight",
    "block_sparse_moe.shared_experts.down_proj.weight",
}
EXPERT_COMPONENTS = {
    f"{projection}.{component}"
    for projection in ("w1", "w2", "w3")
    for component in ("weight_packed", "weight_scale")
}
LAYER_RE = re.compile(r"^language_model\.model\.layers\.(\d+)\.(.+)$")
EXPERT_RE = re.compile(
    r"^block_sparse_moe\.experts\.(\d+)\.(w[123])\.(weight_packed|weight_scale)$"
)
SHARD_RE = re.compile(r"^model-(\d{5})-of-000096\.safetensors$")
DTYPE_BYTES = {"U8": 1, "BF16": 2, "F32": 4}
TP_DIMS = (7168, 12288, 1536, 512, 128, 96)


class PreflightError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise PreflightError(f"expected JSON object: {path}")
    return document


def validate_config(config: dict[str, Any]) -> tuple[set[int], set[int]]:
    if config.get("model_type") != "kimi_k3":
        raise PreflightError("config model_type is not kimi_k3")
    text = config.get("text_config")
    if not isinstance(text, dict):
        raise PreflightError("config has no text_config object")
    expected = {
        "num_hidden_layers": LAYER_COUNT,
        "hidden_size": 7168,
        "first_k_dense_replace": 1,
        "num_experts": EXPERT_COUNT,
        "num_experts_per_token": 16,
        "attn_res_block_size": 12,
    }
    for field, value in expected.items():
        if text.get(field) != value:
            raise PreflightError(
                f"config {field}={text.get(field)!r}, expected {value!r}"
            )
    linear = text.get("linear_attn_config")
    if not isinstance(linear, dict):
        raise PreflightError("config has no linear_attn_config object")
    kda = {int(layer) - 1 for layer in linear.get("kda_layers", [])}
    mla = {int(layer) - 1 for layer in linear.get("full_attn_layers", [])}
    if len(kda) != KDA_COUNT or len(mla) != MLA_COUNT:
        raise PreflightError(
            f"invalid attention schedule counts: kda={len(kda)} mla={len(mla)}"
        )
    if kda & mla or kda | mla != set(range(LAYER_COUNT)):
        raise PreflightError("attention schedules are overlapping or incomplete")
    return kda, mla


def layer_kind(layer: int, kda_layers: set[int]) -> str:
    if layer == 0:
        return "kda_dense"
    return "kda_moe" if layer in kda_layers else "mla_moe"


def expected_nonexpert_suffixes(kind: str) -> set[str]:
    attention = KDA_SUFFIXES if kind.startswith("kda") else MLA_SUFFIXES
    feed_forward = DENSE_SUFFIXES if kind == "kda_dense" else MOE_SUFFIXES
    return COMMON_SUFFIXES | attention | feed_forward


def validate_layer_inventory(
    layer: int,
    kind: str,
    nonexpert_suffixes: set[str],
    expert_counts: dict[str, int],
) -> None:
    expected = expected_nonexpert_suffixes(kind)
    missing = sorted(expected - nonexpert_suffixes)
    unexpected = sorted(nonexpert_suffixes - expected)
    if missing or unexpected:
        raise PreflightError(
            f"layer {layer} {kind} non-expert mismatch: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    if kind == "kda_dense":
        if expert_counts:
            raise PreflightError(f"dense layer {layer} unexpectedly owns expert tensors")
        return
    if set(expert_counts) != EXPERT_COMPONENTS:
        raise PreflightError(
            f"layer {layer} expert components mismatch: "
            f"{sorted(set(expert_counts) ^ EXPERT_COMPONENTS)}"
        )
    bad = {
        component: count
        for component, count in expert_counts.items()
        if count != EXPERT_COUNT
    }
    if bad:
        raise PreflightError(f"layer {layer} incomplete expert ownership: {bad}")


def validate_index(
    index: dict[str, Any], kda_layers: set[int]
) -> dict[str, Any]:
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise PreflightError("index has no weight_map object")
    if len(weight_map) != 497_220:
        raise PreflightError(
            f"full index has {len(weight_map)} tensors, expected 497220"
        )

    layer_nonexpert: dict[int, set[str]] = defaultdict(set)
    layer_experts: dict[int, dict[str, set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )
    layer_shards: dict[int, set[str]] = defaultdict(set)
    head = set()
    vision = set()
    unexpected = []

    for name, shard in weight_map.items():
        if not isinstance(name, str) or not isinstance(shard, str):
            raise PreflightError("index weight_map entries must be string-to-string")
        match = LAYER_RE.fullmatch(name)
        if match:
            layer = int(match.group(1))
            suffix = match.group(2)
            if layer < 0 or layer >= LAYER_COUNT:
                raise PreflightError(f"out-of-range logical layer: {name}")
            layer_shards[layer].add(shard)
            expert = EXPERT_RE.fullmatch(suffix)
            if expert:
                expert_id = int(expert.group(1))
                if expert_id < 0 or expert_id >= EXPERT_COUNT:
                    raise PreflightError(f"out-of-range expert id: {name}")
                component = f"{expert.group(2)}.{expert.group(3)}"
                layer_experts[layer][component].add(expert_id)
            else:
                layer_nonexpert[layer].add(suffix)
        elif name in HEAD_TENSORS:
            head.add(name)
        elif name.startswith("vision_tower.") or name.startswith("mm_projector."):
            vision.add(name)
        else:
            unexpected.append(name)

    if head != HEAD_TENSORS:
        raise PreflightError(
            f"text head mismatch: missing={sorted(HEAD_TENSORS - head)} "
            f"unexpected={sorted(head - HEAD_TENSORS)}"
        )
    if unexpected:
        raise PreflightError(f"unexpected checkpoint namespaces: {unexpected[:8]}")
    if len(vision) != 168:
        raise PreflightError(f"vision exclusion inventory={len(vision)}, expected 168")

    family_counts = defaultdict(int)
    layer_records = []
    for layer in range(LAYER_COUNT):
        kind = layer_kind(layer, kda_layers)
        family_counts[kind] += 1
        expert_counts = {
            component: len(ids)
            for component, ids in layer_experts[layer].items()
        }
        validate_layer_inventory(
            layer, kind, layer_nonexpert[layer], expert_counts
        )
        layer_records.append(
            {
                "layer": layer,
                "family": kind,
                "tensor_count": len(layer_nonexpert[layer])
                + sum(expert_counts.values()),
                "shards": sorted(layer_shards[layer]),
            }
        )

    expected_families = {"kda_dense": 1, "kda_moe": 68, "mla_moe": 24}
    if dict(family_counts) != expected_families:
        raise PreflightError(
            f"layer family counts={dict(family_counts)}, expected={expected_families}"
        )

    shards = sorted(set(weight_map.values()))
    expected_shards = {
        f"model-{number:05d}-of-000096.safetensors"
        for number in range(1, SHARD_COUNT + 1)
    }
    if set(shards) != expected_shards:
        raise PreflightError(
            "full index shard set is incomplete or has unexpected filenames"
        )

    text_count = len(head) + sum(record["tensor_count"] for record in layer_records)
    if text_count != 497_052:
        raise PreflightError(f"text tensor count={text_count}, expected 497052")
    return {
        "tensor_names": len(weight_map),
        "text_tensor_names": text_count,
        "vision_ignored_tensor_names": len(vision),
        "referenced_shards": len(shards),
        "family_counts": dict(family_counts),
        "layer_records": layer_records,
        "shards": shards,
    }


def inventory_sizes(path: Path) -> dict[str, Any]:
    totals = {"text": 0, "vision": 0, "head": 0, "layer0": 0}
    layer_bytes = defaultdict(int)
    layer_ids = set()
    unknown_shape_records = 0
    records = 0
    opener = gzip.open if path.suffix == ".gz" else path.open
    with opener(path, "rt") as stream:
        for raw in stream:
            item = json.loads(raw)
            records += 1
            name = item["name"]
            match = LAYER_RE.fullmatch(name)
            if match:
                layer_ids.add(int(match.group(1)))
            elif name not in HEAD_TENSORS and not (
                name.startswith("vision_tower.") or name.startswith("mm_projector.")
            ):
                raise PreflightError(f"unexpected tensor in frozen inventory: {name}")
            if "dtype" not in item:
                unknown_shape_records += 1
                continue
            dtype = item["dtype"]
            if dtype not in DTYPE_BYTES:
                raise PreflightError(f"unsupported inventory dtype: {dtype}")
            size = DTYPE_BYTES[dtype] * math.prod(int(dim) for dim in item["shape"])
            if match:
                layer = int(match.group(1))
                layer_bytes[layer] += size
                totals["text"] += size
                if layer == 0:
                    totals["layer0"] += size
            elif name in HEAD_TENSORS:
                totals["text"] += size
                totals["head"] += size
            else:
                totals["vision"] += size
    if records != 497_220 or layer_ids != set(range(LAYER_COUNT)):
        raise PreflightError(
            f"inventory records/layers={records}/{len(layer_ids)}, expected 497220/93"
        )
    return {
        **totals,
        "records": records,
        "unknown_shape_records": unknown_shape_records,
        "max_staged_layer_bytes": max(layer_bytes.values()),
        "max_staged_layer": max(layer_bytes, key=layer_bytes.get),
        "resident_head_layer0_bytes": totals["head"] + totals["layer0"],
        "available_layer_bytes": {str(layer): size for layer, size in sorted(layer_bytes.items())},
    }


def choose_tensor_parallel(devices: int) -> int:
    for degree in range(min(devices, 32), 0, -1):
        if devices % degree == 0 and all(dim % degree == 0 for dim in TP_DIMS):
            return degree
    return 1


def partition_sizes(items: int, ranks: int) -> list[int]:
    base, remainder = divmod(items, ranks)
    return [base + int(rank < remainder) for rank in range(ranks)]


def detect_nvidia_devices() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=15
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    devices = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            continue
        devices.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "uuid": fields[2],
                "hbm_bytes": int(fields[3]) * 1024 * 1024,
            }
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        selected = []
        for token in (part.strip() for part in visible.split(",")):
            if not token:
                continue
            if token.isdigit():
                selected.extend(
                    device for device in devices if device["index"] == int(token)
                )
            else:
                selected.extend(
                    device for device in devices if device["uuid"] == token
                )
        devices = selected
    return devices


def load_hash_manifest(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    document = load_json(path)
    files = document.get("files")
    if not isinstance(files, dict):
        raise PreflightError(f"invalid checkpoint hash manifest: {path}")
    return files


def shard_status(
    model_dir: Path,
    shard_names: Iterable[str],
    known_hashes: dict[str, dict[str, Any]],
    verify_present_hashes: bool,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    records = []
    missing = []
    unverified = []
    for name in shard_names:
        path = model_dir / name
        known = known_hashes.get(name, {})
        present = path.is_file()
        record = {
            "name": name,
            "present": present,
            "bytes": path.stat().st_size if present else None,
            "expected_bytes": known.get("bytes"),
            "expected_sha256": known.get("sha256"),
            "actual_sha256": sha256(path) if present and verify_present_hashes else None,
        }
        if not present:
            missing.append(name)
        else:
            if record["expected_bytes"] is not None and record["bytes"] != record["expected_bytes"]:
                raise PreflightError(f"size mismatch for {name}")
            if verify_present_hashes and record["expected_sha256"] is not None:
                if record["actual_sha256"] != record["expected_sha256"]:
                    raise PreflightError(f"SHA-256 mismatch for {name}")
            if not verify_present_hashes:
                unverified.append(name)
        records.append(record)
    return records, missing, unverified


def distributed_scenarios(
    text_bytes: int,
    resident_bytes: int,
    max_layer_bytes: int,
    cache_bytes: int,
    device_counts: Iterable[int],
) -> list[dict[str, Any]]:
    scenarios = []
    for devices in sorted(set(int(value) for value in device_counts if int(value) > 0)):
        tensor_parallel = choose_tensor_parallel(devices)
        expert_parallel = devices // tensor_parallel
        expert_sizes = partition_sizes(EXPERT_COUNT, expert_parallel)
        scenarios.append(
            {
                "devices": devices,
                "tensor_parallel": tensor_parallel,
                "expert_parallel": expert_parallel,
                "experts_per_expert_rank_min": min(expert_sizes),
                "experts_per_expert_rank_max": max(expert_sizes),
                "full_text_storage_bytes_per_device": math.ceil(text_bytes / devices),
                "streaming_resident_bytes_per_device": math.ceil(resident_bytes / devices),
                "streaming_max_layer_bytes_per_device": math.ceil(max_layer_bytes / devices),
                "cache_1m_bytes_per_device": math.ceil(cache_bytes / tensor_parallel),
            }
        )
    return scenarios


def preflight(
    model_dir: Path,
    inventory: Path,
    hash_manifest: Path | None,
    scenario_devices: list[int],
    verify_present_hashes: bool,
) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise PreflightError("checkpoint requires config.json and model.safetensors.index.json")
    config = load_json(config_path)
    kda_layers, mla_layers = validate_config(config)
    index = load_json(index_path)
    contract = validate_index(index, kda_layers)
    sizes = inventory_sizes(inventory)
    total_checkpoint_bytes = int(index.get("metadata", {}).get("total_size", 0))
    if total_checkpoint_bytes <= 0:
        raise PreflightError("full index metadata has no positive total_size")
    known_hashes = load_hash_manifest(hash_manifest)
    shards, missing, unverified = shard_status(
        model_dir,
        contract["shards"],
        known_hashes,
        verify_present_hashes,
    )
    devices = detect_nvidia_devices()
    scenario_counts = [1, 8, 16, 24, 32, *scenario_devices]
    if devices:
        scenario_counts.append(len(devices))

    cache_1m = 28_102_459_392
    expert_component_peak = EXPERT_COUNT * 3584 * 1536
    expert_bank = (
        EXPERT_COUNT
        * (
            3072 * 1792
            + 3072 * 112
            + 3584 * 1536
            + 3584 * 96
            + 3072 * 1792
            + 3072 * 112
        )
    )
    ready_to_load = not missing
    hashes_complete = ready_to_load and not unverified
    full_hbm = sum(device["hbm_bytes"] for device in devices)
    minimum_streaming_hbm = (
        sizes["resident_head_layer0_bytes"]
        + sizes["max_staged_layer_bytes"]
        + cache_1m
    )
    hardware_ready = bool(devices) and full_hbm >= minimum_streaming_hbm

    return {
        "schema_version": 1,
        "status": (
            "READY_FOR_FULL_VALIDATION"
            if ready_to_load and hashes_complete and hardware_ready
            else "READY_FOR_WEIGHTS"
        ),
        "structural_gate_pass": True,
        "ready_to_load": ready_to_load,
        "ready_for_full_validation": ready_to_load and hashes_complete and hardware_ready,
        "downloads_attempted": False,
        "checkpoint": {
            "directory": str(model_dir),
            "config_sha256": sha256(config_path),
            "index_sha256": sha256(index_path),
            "referenced_shards": SHARD_COUNT,
            "present_shards": SHARD_COUNT - len(missing),
            "missing_shards": missing,
            "unverified_present_shards": unverified,
            "shards": shards,
        },
        "text_contract": {
            "logical_layers": LAYER_COUNT,
            "kda_layers": len(kda_layers),
            "mla_layers": len(mla_layers),
            "moe_layers": MOE_LAYER_COUNT,
            "experts_per_moe_layer": EXPERT_COUNT,
            "top_k": 16,
            "output_attention_residual": True,
            "vision_tensor_policy": "explicitly_ignored",
            **{key: contract[key] for key in (
                "tensor_names",
                "text_tensor_names",
                "vision_ignored_tensor_names",
                "family_counts",
            )},
        },
        "loading": {
            "resident": "text head plus layer 0",
            "staged_order": "logical layers 1..92, one complete layer at a time",
            "expert_read_granularity": "one packed/scale tensor per expert",
            "peak_host_expert_component_bytes": expert_component_peak,
            "expert_bank_device_bytes_before_sharding": expert_bank,
            "resident_head_layer0_bytes": sizes["resident_head_layer0_bytes"],
            "max_staged_layer": sizes["max_staged_layer"],
            "max_staged_layer_bytes": sizes["max_staged_layer_bytes"],
            "dequantized_full_model_host_copy": False,
        },
        "resources": {
            "full_checkpoint_bytes": total_checkpoint_bytes,
            "text_checkpoint_bytes_upper_bound": total_checkpoint_bytes,
            "available_inventory_text_bytes": sizes["text"],
            "vision_checkpoint_bytes_known": sizes["vision"],
            "inventory_unknown_shape_records": sizes["unknown_shape_records"],
            "kda_cache_bytes_batch1": 454_459_392,
            "mla_cache_1m_bytes_batch1": 27_648_000_000,
            "combined_cache_1m_bytes_batch1": cache_1m,
            "tensor_parallel_hidden_collective_bytes_per_token_lower_bound": 92
            * 7168
            * 2,
            "expert_dispatch_bytes_per_token_upper_bound": 92 * 16 * 7168 * 2,
            "scenarios": distributed_scenarios(
                total_checkpoint_bytes,
                sizes["resident_head_layer0_bytes"],
                sizes["max_staged_layer_bytes"],
                cache_1m,
                scenario_counts,
            ),
        },
        "hardware": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "nvidia_devices": devices,
            "aggregate_hbm_bytes": full_hbm,
            "minimum_streaming_hbm_bytes": minimum_streaming_hbm,
            "passes_streaming_memory_estimate": hardware_ready,
            "topology_validation": "deferred to ZML platform preflight",
        },
        "blocking_for_full_validation": [
            *([f"{len(missing)} checkpoint shards are missing"] if missing else []),
            *(
                [f"{len(unverified)} present shards lack an in-run verified hash"]
                if unverified
                else []
            ),
            *(["insufficient detected NVIDIA HBM for the conservative streaming estimate"] if not hardware_ready else []),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--hash-manifest", type=Path)
    parser.add_argument("--scenario-devices", type=int, action="append", default=[])
    parser.add_argument("--verify-present-hashes", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    report = preflight(
        args.model,
        args.inventory,
        args.hash_manifest,
        args.scenario_devices,
        args.verify_present_hashes,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    if not args.quiet:
        print(encoded, end="")
    if args.require_complete and not report["ready_for_full_validation"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
