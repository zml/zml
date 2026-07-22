#!/usr/bin/env python3
"""Convert DeepSeek V4 routed expert weights from packed FP4 to FP8.

Tensor names and shard assignments are preserved. The converter accepts both
the original per-expert tensors and the stacked tensors produced by repack.py.
Non-expert tensors and shards are copied unchanged.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open, save_file


EXPERT_WEIGHT_RE = re.compile(
    r"^layers\.\d+\.ffn\.experts(?:\.\d+)?\.w(?:1|2|3|13)\.weight$"
)
DEFAULT_WORKERS = min(2, os.cpu_count() or 1)
FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)


def cast_e2m1fn_to_e4m3fn(
    x: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast a 2D tensor from E2M1FN to E4M3FN losslessly."""
    assert x.dtype == torch.int8
    assert x.ndim == 2
    out_dim, in_dim = x.size()
    in_dim *= 2
    fp8_block_size = 128
    fp4_block_size = 32
    assert in_dim % fp8_block_size == 0 and out_dim % fp8_block_size == 0
    assert scale.size(0) == out_dim and scale.size(1) == in_dim // fp4_block_size

    x = x.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    x = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(2)

    # max_fp4 (6.0) * MAX_OFFSET must fit in e4m3fn (max 448)
    # 6.0 * 2^6 = 384 < 448; 6.0 * 2^7 = 768 > 448; so MAX_OFFSET_BITS = 6
    max_offset_bits = 6

    block_out = out_dim // fp8_block_size
    block_in = in_dim // fp8_block_size
    x = x.view(block_out, fp8_block_size, block_in, fp8_block_size).transpose(1, 2)
    scale = (
        scale.float()
        .view(block_out, fp8_block_size, block_in, -1)
        .transpose(1, 2)
        .flatten(2)
    )
    scale_max_offset_bits = scale.amax(dim=-1, keepdim=True) / (2**max_offset_bits)
    offset = scale / scale_max_offset_bits
    offset = offset.unflatten(-1, (fp8_block_size, -1)).repeat_interleave(
        fp4_block_size, dim=-1
    )
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return (
        x.to(torch.float8_e4m3fn),
        scale_max_offset_bits.squeeze(-1).to(torch.float8_e8m0fnu),
    )


def convert_expert_tensor(
    weight: torch.Tensor,
    scale: torch.Tensor,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim == 2:
        try:
            return cast_e2m1fn_to_e4m3fn(weight, scale)
        except AssertionError as error:
            raise ValueError(
                f"Invalid FP4 tensors for {name}: weight={tuple(weight.shape)} "
                f"{weight.dtype}, scale={tuple(scale.shape)} {scale.dtype}"
            ) from error

    if weight.ndim != 3 or scale.ndim != 3 or weight.shape[0] != scale.shape[0]:
        raise ValueError(
            f"Expected a 2D expert or matching stacked 3D experts for {name}; "
            f"got weight={tuple(weight.shape)}, scale={tuple(scale.shape)}"
        )

    converted_weight: torch.Tensor | None = None
    converted_scale: torch.Tensor | None = None
    for expert_id in range(weight.shape[0]):
        try:
            expert_weight, expert_scale = cast_e2m1fn_to_e4m3fn(
                weight[expert_id], scale[expert_id]
            )
        except AssertionError as error:
            raise ValueError(
                f"Invalid FP4 tensors for {name} expert {expert_id}: "
                f"weight={tuple(weight[expert_id].shape)} {weight.dtype}, "
                f"scale={tuple(scale[expert_id].shape)} {scale.dtype}"
            ) from error

        if converted_weight is None:
            converted_weight = torch.empty(
                (weight.shape[0], *expert_weight.shape), dtype=expert_weight.dtype
            )
            converted_scale = torch.empty(
                (scale.shape[0], *expert_scale.shape), dtype=expert_scale.dtype
            )
        converted_weight[expert_id].copy_(expert_weight)
        converted_scale[expert_id].copy_(expert_scale)

    assert converted_weight is not None and converted_scale is not None
    return converted_weight, converted_scale


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def read_index(input_dir: Path) -> dict[str, Any]:
    index_path = input_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    with index_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def collect_shards(weight_map: dict[str, str]) -> dict[str, list[str]]:
    shards: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        shards[shard].append(name)
    return dict(sorted(shards.items()))


def collect_conversion_pairs(
    weight_map: dict[str, str],
) -> dict[str, list[tuple[str, str]]]:
    pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for weight_name, shard in weight_map.items():
        if EXPERT_WEIGHT_RE.match(weight_name) is None:
            continue
        scale_name = weight_name.removesuffix("weight") + "scale"
        scale_shard = weight_map.get(scale_name)
        if scale_shard is None:
            raise ValueError(f"Missing scale tensor for {weight_name}: {scale_name}")
        if scale_shard != shard:
            raise ValueError(
                f"Weight and scale must be in the same shard: {weight_name} is in "
                f"{shard}, but {scale_name} is in {scale_shard}"
            )
        pairs[shard].append((weight_name, scale_name))
    return {shard: sorted(shard_pairs) for shard, shard_pairs in pairs.items()}


def save_tensors(
    output_path: Path,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, str] | None,
) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        save_file(tensors, tmp_path, metadata=metadata)
        os.replace(tmp_path, output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def shard_nbytes(path: Path, keys: list[str]) -> int:
    with safe_open(path, framework="pt", device="cpu") as reader:
        return sum(tensor_nbytes(reader.get_tensor(key)) for key in keys)


def convert_shard(
    input_path: Path,
    output_path: Path,
    keys: list[str],
    pairs: list[tuple[str, str]],
    verbose: bool,
) -> int:
    output_tensors: dict[str, torch.Tensor] = {}
    converted_keys = {name for pair in pairs for name in pair}
    with safe_open(input_path, framework="pt", device="cpu") as reader:
        metadata = reader.metadata()
        for weight_name, scale_name in pairs:
            weight, scale = convert_expert_tensor(
                reader.get_tensor(weight_name),
                reader.get_tensor(scale_name),
                weight_name,
            )
            output_tensors[weight_name] = weight
            output_tensors[scale_name] = scale
            if verbose:
                print(f"  {weight_name}: {tuple(weight.shape)} {weight.dtype}")
                print(f"  {scale_name}: {tuple(scale.shape)} {scale.dtype}")

        for key in sorted(keys):
            if key not in converted_keys:
                output_tensors[key] = reader.get_tensor(key)

    save_tensors(output_path, output_tensors, metadata)
    return sum(tensor_nbytes(tensor) for tensor in output_tensors.values())


def process_shard(
    input_dir: Path,
    output_dir: Path,
    shard: str,
    keys: list[str],
    pairs: list[tuple[str, str]],
    verbose: bool,
) -> tuple[str, int, str]:
    input_path = input_dir / shard
    output_path = output_dir / shard
    if not pairs:
        size = shard_nbytes(input_path, keys)
        if input_path == output_path:
            return shard, size, "kept"
        shutil.copy2(input_path, output_path)
        return shard, size, "copied"

    size = convert_shard(input_path, output_path, keys, pairs, verbose)
    return shard, size, "converted"


def copy_sidecar_files(input_dir: Path, output_dir: Path) -> None:
    if input_dir == output_dir:
        return
    for path in input_dir.iterdir():
        if (
            not path.is_file()
            or path.name == "model.safetensors.index.json"
            or path.suffix == ".safetensors"
        ):
            continue
        shutil.copy2(path, output_dir / path.name)


def write_index(index_path: Path, index: dict[str, Any]) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{index_path.name}.", suffix=".tmp", dir=index_path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(index, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(tmp_path, index_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DeepSeek V4 routed expert tensors from packed FP4 to FP8."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing safetensors shards and model.safetensors.index.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the converted checkpoint. Required unless --in-place is set.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the checkpoint directly under --input-dir.",
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
        help="Validate and print the expert tensors that would be converted.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of safetensors shards to process in parallel.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch CPU threads used by each conversion.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print every converted tensor shape."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if args.in_place:
        if args.output_dir is not None and args.output_dir.resolve() != input_dir:
            raise ValueError(
                "--in-place writes to --input-dir; omit --output-dir or use the same path."
            )
        output_dir = input_dir
    else:
        if args.output_dir is None:
            raise ValueError("--output-dir is required unless --in-place is set.")
        output_dir = args.output_dir.resolve()
        if input_dir == output_dir and not args.dry_run:
            raise ValueError("Refusing to write in-place. Pass --in-place to modify --input-dir.")

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    if args.torch_threads < 1:
        raise ValueError("--torch-threads must be >= 1")
    torch.set_num_threads(args.torch_threads)

    index = read_index(input_dir)
    weight_map = index["weight_map"]
    shards = collect_shards(weight_map)
    conversion_pairs = collect_conversion_pairs(weight_map)
    conversion_count = sum(len(pairs) for pairs in conversion_pairs.values())
    print(f"Found {conversion_count} FP4 expert weights across {len(conversion_pairs)} shards.")

    if args.dry_run:
        for shard, pairs in sorted(conversion_pairs.items()):
            print(shard)
            for weight_name, _ in pairs:
                print(f"  {weight_name}")
        return

    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not args.overwrite
        and not args.in_place
    ):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Pass --overwrite to reuse it."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    total_size = 0
    max_workers = min(args.workers, len(shards))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_shard,
                input_dir,
                output_dir,
                shard,
                keys,
                conversion_pairs.get(shard, []),
                args.verbose,
            )
            for shard, keys in shards.items()
        ]
        for future in concurrent.futures.as_completed(futures):
            shard, shard_size, action = future.result()
            total_size += shard_size
            print(f"{action.capitalize()} {shard}")

    new_index = dict(index)
    new_index["metadata"] = dict(new_index.get("metadata") or {})
    new_index["metadata"]["total_size"] = total_size
    write_index(output_dir / "model.safetensors.index.json", new_index)

    if args.copy_sidecars:
        copy_sidecar_files(input_dir, output_dir)

    if args.in_place:
        print(f"Done. Converted FP4 experts in place at {output_dir}")
    else:
        print(f"Done. Wrote FP8 experts to {output_dir}")


if __name__ == "__main__":
    main()
