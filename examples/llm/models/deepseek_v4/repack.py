#!/usr/bin/env python3
"""Merge DeepSeek V4 per-expert safetensors keys into stacked expert tensors.

The Hugging Face checkpoint stores routed expert tensors as separate keys:

    layers.0.ffn.experts.0.w1.scale
    layers.0.ffn.experts.1.w1.scale
    ...

This script rewrites them as stacked tensors and, by default, fuses w1/w3:

    layers.0.ffn.experts.w13.scale

with expert id as the leading dimension. w1 and w3 are interleaved along the
output/intermediate dimension as w1[0], w3[0], w1[1], w3[1], ... so fused
SwiGLU kernels can split paired columns locally. Non-expert tensors are copied
through unchanged, and a new `model.safetensors.index.json` is written for the
output.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open, save_file


EXPERT_RE = re.compile(
    r"^(?P<prefix>layers\.\d+\.ffn\.experts)\."
    r"(?P<expert>\d+)\."
    r"(?P<name>w[123]\.(?:weight|scale))$"
)
DEFAULT_WORKERS = min(2, os.cpu_count() or 1)


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def read_index(input_dir: Path) -> dict[str, Any]:
    index_path = input_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_shards(weight_map: dict[str, str]) -> dict[str, list[str]]:
    shards: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        shards[shard].append(name)
    return dict(sorted(shards.items()))


def collect_expert_groups(keys: list[str]) -> dict[tuple[str, str], dict[int, str]]:
    groups: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    for key in keys:
        match = EXPERT_RE.match(key)
        if match is None:
            continue
        group_key = (match.group("prefix"), match.group("name"))
        expert_id = int(match.group("expert"))
        if expert_id in groups[group_key]:
            raise ValueError(f"Duplicate expert {expert_id} for {group_key}")
        groups[group_key][expert_id] = key
    return groups


def validate_groups(
    groups: dict[tuple[str, str], dict[int, str]],
    expected_experts: int | None,
) -> int:
    if not groups:
        return expected_experts or 0

    counts = {len(experts) for experts in groups.values()}
    if len(counts) != 1:
        formatted = ", ".join(f"{key}: {len(value)}" for key, value in sorted(groups.items()))
        raise ValueError(f"Expert groups do not all have the same count: {formatted}")

    inferred_experts = counts.pop()
    if expected_experts is not None and inferred_experts != expected_experts:
        raise ValueError(
            f"Expected {expected_experts} experts, but found {inferred_experts} in this shard"
        )

    expected_ids = set(range(inferred_experts))
    for key, experts in groups.items():
        ids = set(experts)
        if ids != expected_ids:
            missing = sorted(expected_ids - ids)
            extra = sorted(ids - expected_ids)
            raise ValueError(f"Non-contiguous expert ids for {key}; missing={missing}, extra={extra}")

    return inferred_experts


def merge_group(
    reader: Any,
    expert_keys: dict[int, str],
    output_name: str,
    verbose: bool,
) -> torch.Tensor:
    first = reader.get_tensor(expert_keys[0])
    merged = torch.empty(
        (len(expert_keys), *first.shape),
        dtype=first.dtype,
        device=first.device,
    )
    merged[0].copy_(first)

    for expert_id in range(1, len(expert_keys)):
        tensor = reader.get_tensor(expert_keys[expert_id])
        if tensor.shape != first.shape:
            raise ValueError(
                f"Shape mismatch while building {output_name}: expert 0 has "
                f"{tuple(first.shape)}, expert {expert_id} has {tuple(tensor.shape)}"
            )
        if tensor.dtype != first.dtype:
            raise ValueError(
                f"Dtype mismatch while building {output_name}: expert 0 has "
                f"{first.dtype}, expert {expert_id} has {tensor.dtype}"
            )
        merged[expert_id].copy_(tensor)

    if verbose:
        print(f"  {output_name}: {tuple(merged.shape)} {merged.dtype}")
    return merged


def merge_fused_pair(
    reader: Any,
    left_keys: dict[int, str],
    right_keys: dict[int, str],
    output_name: str,
    verbose: bool,
) -> torch.Tensor:
    left_first = reader.get_tensor(left_keys[0])
    right_first = reader.get_tensor(right_keys[0])
    if left_first.shape != right_first.shape:
        raise ValueError(
            f"Shape mismatch while building {output_name}: left has "
            f"{tuple(left_first.shape)}, right has {tuple(right_first.shape)}"
        )
    if left_first.dtype != right_first.dtype:
        raise ValueError(
            f"Dtype mismatch while building {output_name}: left has "
            f"{left_first.dtype}, right has {right_first.dtype}"
        )

    fused_shape = (
        len(left_keys),
        left_first.shape[0] + right_first.shape[0],
        *left_first.shape[1:],
    )
    fused = torch.empty(fused_shape, dtype=left_first.dtype, device=left_first.device)
    fused[0, 0::2].copy_(left_first)
    fused[0, 1::2].copy_(right_first)

    for expert_id in range(1, len(left_keys)):
        left = reader.get_tensor(left_keys[expert_id])
        right = reader.get_tensor(right_keys[expert_id])
        if left.shape != left_first.shape or right.shape != right_first.shape:
            raise ValueError(
                f"Shape mismatch while building {output_name} expert {expert_id}: "
                f"left={tuple(left.shape)}, right={tuple(right.shape)}"
            )
        if left.dtype != left_first.dtype or right.dtype != right_first.dtype:
            raise ValueError(
                f"Dtype mismatch while building {output_name} expert {expert_id}: "
                f"left={left.dtype}, right={right.dtype}"
            )
        fused[expert_id, 0::2].copy_(left)
        fused[expert_id, 1::2].copy_(right)

    if verbose:
        print(f"  {output_name}: {tuple(fused.shape)} {fused.dtype}")
    return fused


def merged_output_names(
    groups: dict[tuple[str, str], dict[int, str]],
    fuse_w1_w3: bool,
    fused_name: str,
) -> list[str]:
    names: list[str] = []
    for prefix, name in sorted(groups):
        if fuse_w1_w3 and (name.startswith("w1.") or name.startswith("w3.")):
            continue
        names.append(f"{prefix}.{name}")

    if fuse_w1_w3:
        prefixes = sorted({prefix for prefix, _ in groups})
        for prefix in prefixes:
            for suffix in ("scale", "weight"):
                if (prefix, f"w1.{suffix}") in groups or (prefix, f"w3.{suffix}") in groups:
                    names.append(f"{prefix}.{fused_name}.{suffix}")
    return sorted(names)


def convert_shard(
    input_dir: Path,
    output_dir: Path,
    shard: str,
    keys: list[str],
    expected_experts: int | None,
    fuse_w1_w3: bool,
    fused_name: str,
    verbose: bool,
) -> tuple[dict[str, str], int, int]:
    groups = collect_expert_groups(keys)
    inferred_experts = validate_groups(groups, expected_experts)
    expert_input_keys = {key for experts in groups.values() for key in experts.values()}
    output_weight_map: dict[str, str] = {}
    output_tensors: dict[str, torch.Tensor] = {}

    input_path = input_dir / shard
    output_path = output_dir / shard
    with safe_open(input_path, framework="pt", device="cpu") as reader:
        metadata = reader.metadata()
        for key in sorted(keys):
            if key in expert_input_keys:
                continue
            output_tensors[key] = reader.get_tensor(key)
            output_weight_map[key] = shard

        fused_group_keys: set[tuple[str, str]] = set()
        if fuse_w1_w3:
            prefixes = sorted({prefix for prefix, _ in groups})
            for prefix in prefixes:
                for suffix in ("scale", "weight"):
                    left_key = (prefix, f"w1.{suffix}")
                    right_key = (prefix, f"w3.{suffix}")
                    if left_key not in groups and right_key not in groups:
                        continue
                    if left_key not in groups or right_key not in groups:
                        raise ValueError(
                            f"Cannot fuse {prefix} {suffix}: both w1 and w3 are required"
                        )
                    output_name = f"{prefix}.{fused_name}.{suffix}"
                    output_tensors[output_name] = merge_fused_pair(
                        reader, groups[left_key], groups[right_key], output_name, verbose
                    )
                    output_weight_map[output_name] = shard
                    fused_group_keys.update({left_key, right_key})

        for (prefix, name), expert_keys in sorted(groups.items()):
            if (prefix, name) in fused_group_keys:
                continue
            output_name = f"{prefix}.{name}"
            output_tensors[output_name] = merge_group(reader, expert_keys, output_name, verbose)
            output_weight_map[output_name] = shard

    bytes_written = sum(tensor_nbytes(tensor) for tensor in output_tensors.values())
    save_file(output_tensors, output_path, metadata=metadata)
    return output_weight_map, inferred_experts, bytes_written


def copy_shard(input_dir: Path, output_dir: Path, shard: str, keys: list[str]) -> dict[str, str]:
    shutil.copy2(input_dir / shard, output_dir / shard)
    return {key: shard for key in keys}


def process_shard(
    input_dir: Path,
    output_dir: Path,
    shard: str,
    keys: list[str],
    expected_experts: int | None,
    fuse_w1_w3: bool,
    fused_name: str,
    verbose: bool,
    torch_threads: int,
) -> tuple[str, dict[str, str], int, str]:
    torch.set_num_threads(torch_threads)
    groups = collect_expert_groups(keys)
    if not groups:
        return shard, copy_shard(input_dir, output_dir, shard, keys), 0, "copied"

    shard_weight_map, _, shard_size = convert_shard(
        input_dir=input_dir,
        output_dir=output_dir,
        shard=shard,
        keys=keys,
        expected_experts=expected_experts,
        fuse_w1_w3=fuse_w1_w3,
        fused_name=fused_name,
        verbose=verbose,
    )
    return shard, shard_weight_map, shard_size, "rewritten"


def copy_sidecar_files(input_dir: Path, output_dir: Path) -> None:
    skip_names = {"model.safetensors.index.json"}
    skip_suffixes = {".safetensors"}
    for path in input_dir.iterdir():
        if not path.is_file() or path.name in skip_names or path.suffix in skip_suffixes:
            continue
        shutil.copy2(path, output_dir / path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge DeepSeek V4 routed expert tensors inside safetensors shards."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing model-*.safetensors and model.safetensors.index.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the rewritten checkpoint.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Expected routed experts per layer. Defaults to config.json n_routed_experts, then inference.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    parser.add_argument(
        "--copy-sidecars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy root-level non-safetensors files like config/tokenizer files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print planned output names without writing files.",
    )
    parser.add_argument(
        "--fuse-w1-w3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fuse routed expert w1 and w3 into one interleaved tensor along the output dimension.",
    )
    parser.add_argument(
        "--fused-name",
        default="w13",
        help="Name to use for the fused w1/w3 tensors.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of safetensors shards to process in parallel. Increase if RAM and disk IO allow.",
    )
    parser.add_argument(
        "--parallel-backend",
        choices=("thread", "process"),
        default="thread",
        help="Parallel executor backend. Threads avoid macOS sandbox semaphore limits.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch CPU threads per worker process.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print each merged tensor shape.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if input_dir == output_dir and not args.dry_run:
        raise ValueError("Refusing to write in-place. Choose a different --output-dir.")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be >= 1")

    index = read_index(input_dir)
    weight_map = index["weight_map"]
    shards = collect_shards(weight_map)

    num_experts = args.num_experts
    config_path = input_dir / "config.json"
    if num_experts is None and config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            num_experts = json.load(f).get("n_routed_experts")

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite and not args.dry_run:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Pass --overwrite to reuse it."
        )

    total_expert_groups = 0
    for shard, keys in shards.items():
        total_expert_groups += len(collect_expert_groups(keys))
    expert_shards = sum(1 for keys in shards.values() if collect_expert_groups(keys))
    print(
        f"Found {len(shards)} shards and {total_expert_groups} expert tensor groups "
        f"({num_experts or 'inferred'} experts each)."
    )
    if not args.dry_run:
        print(
            f"Processing {expert_shards} rewrite shards with {args.workers} workers "
            f"and copying {len(shards) - expert_shards} shards unchanged."
        )

    if args.dry_run:
        for shard, keys in shards.items():
            groups = collect_expert_groups(keys)
            if not groups:
                continue
            validate_groups(groups, num_experts)
            print(shard)
            for name in merged_output_names(groups, args.fuse_w1_w3, args.fused_name):
                print(f"  {name}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    new_weight_map: dict[str, str] = {}
    total_size = 0
    max_workers = min(args.workers, len(shards))
    executor_cls: type[concurrent.futures.Executor]
    if args.parallel_backend == "process":
        executor_cls = concurrent.futures.ProcessPoolExecutor
    else:
        executor_cls = concurrent.futures.ThreadPoolExecutor

    with executor_cls(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_shard,
                input_dir,
                output_dir,
                shard,
                keys,
                num_experts,
                args.fuse_w1_w3,
                args.fused_name,
                args.verbose,
                args.torch_threads,
            )
            for shard, keys in shards.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            shard, shard_weight_map, shard_size, action = future.result()
            new_weight_map.update(shard_weight_map)
            total_size += shard_size
            print(f"{action.capitalize()} {shard}")

    new_index = dict(index)
    new_index["weight_map"] = dict(sorted(new_weight_map.items()))
    new_index["metadata"] = dict(new_index.get("metadata") or {})
    if "total_size" not in new_index["metadata"]:
        new_index["metadata"]["total_size"] = total_size
    with (output_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
        json.dump(new_index, f, indent=2, sort_keys=True)
        f.write("\n")

    if args.copy_sidecars:
        copy_sidecar_files(input_dir, output_dir)

    print(f"Done. Wrote merged checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
