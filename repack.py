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
output. Shards are rewritten directly from their safetensors byte ranges, so
peak memory is bounded by the configured streaming buffer rather than shard
size.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import stat
import struct
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


EXPERT_RE = re.compile(
    r"^(?P<prefix>layers\.\d+\.ffn\.experts)\."
    r"(?P<expert>\d+)\."
    r"(?P<name>w[123]\.(?:weight|scale))$"
)
DEFAULT_WORKERS = min(2, os.cpu_count() or 1)
DEFAULT_BUFFER_SIZE_MIB = 4
SAFETENSORS_HEADER_LENGTH = struct.Struct("<Q")


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


@dataclass(frozen=True)
class OutputTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    sources: tuple[TensorInfo, ...]
    interleaved_sources: tuple[TensorInfo, ...] = ()

    @property
    def nbytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.sources) + sum(
            tensor.nbytes for tensor in self.interleaved_sources
        )


def read_index(input_dir: Path) -> dict[str, Any]:
    index_path = input_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index file: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_safetensors_header(
    path: Path,
) -> tuple[dict[str, TensorInfo], dict[str, str] | None]:
    file_size = path.stat().st_size
    with path.open("rb") as f:
        header_length_bytes = f.read(SAFETENSORS_HEADER_LENGTH.size)
        if len(header_length_bytes) != SAFETENSORS_HEADER_LENGTH.size:
            raise ValueError(f"Invalid safetensors file (missing header length): {path}")
        (header_length,) = SAFETENSORS_HEADER_LENGTH.unpack(header_length_bytes)
        if header_length == 0 or header_length > file_size - SAFETENSORS_HEADER_LENGTH.size:
            raise ValueError(f"Invalid safetensors header length {header_length} in {path}")
        header_bytes = f.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"Truncated safetensors header in {path}")

    try:
        raw_header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid safetensors JSON header in {path}: {error}") from error
    if not isinstance(raw_header, dict):
        raise ValueError(f"Safetensors header must be an object in {path}")

    metadata = raw_header.pop("__metadata__", None)
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        )
    ):
        raise ValueError(
            f"Safetensors metadata must contain only string keys and values in {path}"
        )

    data_start = SAFETENSORS_HEADER_LENGTH.size + header_length
    data_size = file_size - data_start
    tensors: dict[str, TensorInfo] = {}
    for name, descriptor in raw_header.items():
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise ValueError(f"Invalid tensor descriptor for {name!r} in {path}")
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if not isinstance(dtype, str):
            raise ValueError(f"Invalid dtype for tensor {name!r} in {path}")
        if not isinstance(shape, list) or any(
            not isinstance(dimension, int) or dimension < 0 for dimension in shape
        ):
            raise ValueError(f"Invalid shape for tensor {name!r} in {path}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(offset, int) for offset in offsets)
        ):
            raise ValueError(f"Invalid data offsets for tensor {name!r} in {path}")
        start, end = offsets
        if start < 0 or end < start or end > data_size:
            raise ValueError(
                f"Out-of-range data offsets {offsets} for tensor {name!r} in {path}"
            )
        tensors[name] = TensorInfo(
            dtype=dtype,
            shape=tuple(shape),
            offset=data_start + start,
            nbytes=end - start,
        )
    return tensors, metadata


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


def plan_merged_group(
    tensors: dict[str, TensorInfo],
    expert_keys: dict[int, str],
    output_name: str,
    verbose: bool,
) -> OutputTensor:
    sources = tuple(
        tensors[expert_keys[expert_id]] for expert_id in range(len(expert_keys))
    )
    first = sources[0]
    for expert_id, tensor in enumerate(sources[1:], start=1):
        if tensor.shape != first.shape:
            raise ValueError(
                f"Shape mismatch while building {output_name}: expert 0 has "
                f"{first.shape}, expert {expert_id} has {tensor.shape}"
            )
        if tensor.dtype != first.dtype:
            raise ValueError(
                f"Dtype mismatch while building {output_name}: expert 0 has "
                f"{first.dtype}, expert {expert_id} has {tensor.dtype}"
            )
        if tensor.nbytes != first.nbytes:
            raise ValueError(
                f"Size mismatch while building {output_name}: expert 0 has "
                f"{first.nbytes} bytes, expert {expert_id} has {tensor.nbytes} bytes"
            )

    output = OutputTensor(
        name=output_name,
        dtype=first.dtype,
        shape=(len(sources), *first.shape),
        sources=sources,
    )
    if verbose:
        print(f"  {output_name}: {output.shape} {output.dtype}")
    return output


def plan_fused_pair(
    tensors: dict[str, TensorInfo],
    left_keys: dict[int, str],
    right_keys: dict[int, str],
    output_name: str,
    verbose: bool,
) -> OutputTensor:
    left_sources = tuple(
        tensors[left_keys[expert_id]] for expert_id in range(len(left_keys))
    )
    right_sources = tuple(
        tensors[right_keys[expert_id]] for expert_id in range(len(right_keys))
    )
    left_first = left_sources[0]
    right_first = right_sources[0]
    if left_first.shape != right_first.shape:
        raise ValueError(
            f"Shape mismatch while building {output_name}: left has "
            f"{left_first.shape}, right has {right_first.shape}"
        )
    if left_first.dtype != right_first.dtype:
        raise ValueError(
            f"Dtype mismatch while building {output_name}: left has "
            f"{left_first.dtype}, right has {right_first.dtype}"
        )
    if left_first.nbytes != right_first.nbytes:
        raise ValueError(
            f"Size mismatch while building {output_name}: left has "
            f"{left_first.nbytes} bytes, right has {right_first.nbytes} bytes"
        )
    if not left_first.shape:
        raise ValueError(f"Cannot interleave scalar tensors while building {output_name}")

    for expert_id, (left, right) in enumerate(
        zip(left_sources, right_sources, strict=True)
    ):
        if left.shape != left_first.shape or right.shape != right_first.shape:
            raise ValueError(
                f"Shape mismatch while building {output_name} expert {expert_id}: "
                f"left={left.shape}, right={right.shape}"
            )
        if left.dtype != left_first.dtype or right.dtype != right_first.dtype:
            raise ValueError(
                f"Dtype mismatch while building {output_name} expert {expert_id}: "
                f"left={left.dtype}, right={right.dtype}"
            )
        if left.nbytes != left_first.nbytes or right.nbytes != right_first.nbytes:
            raise ValueError(
                f"Size mismatch while building {output_name} expert {expert_id}: "
                f"left={left.nbytes} bytes, right={right.nbytes} bytes"
            )
        row_count = left.shape[0]
        if row_count == 0 and (left.nbytes or right.nbytes):
            raise ValueError(
                f"Empty leading dimension has non-empty data while building "
                f"{output_name} expert {expert_id}"
            )
        if row_count and (left.nbytes % row_count or right.nbytes % row_count):
            raise ValueError(
                f"Tensor byte size is not divisible by the leading dimension while "
                f"building {output_name} expert {expert_id}"
            )

    output = OutputTensor(
        name=output_name,
        dtype=left_first.dtype,
        shape=(
            len(left_sources),
            left_first.shape[0] + right_first.shape[0],
            *left_first.shape[1:],
        ),
        sources=left_sources,
        interleaved_sources=right_sources,
    )
    if verbose:
        print(f"  {output_name}: {output.shape} {output.dtype}")
    return output


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


def encode_safetensors_header(
    tensors: list[OutputTensor],
    metadata: dict[str, str] | None,
) -> bytes:
    header: dict[str, Any] = {}
    if metadata is not None:
        header["__metadata__"] = metadata

    offset = 0
    for tensor in tensors:
        header[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.nbytes],
        }
        offset += tensor.nbytes

    encoded = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return encoded + b" " * (-len(encoded) % 8)


def write_all(output: BinaryIO, data: memoryview) -> None:
    while data:
        written = output.write(data)
        if written is None or written == 0:
            raise OSError("Failed to make progress while writing output shard")
        data = data[written:]


def read_exact_at(source: BinaryIO, offset: int, output: memoryview) -> None:
    source.seek(offset)
    filled = 0
    while filled < len(output):
        count = source.readinto(output[filled:])
        if count is None or count == 0:
            raise EOFError(
                f"Unexpected end of input shard at byte {offset + filled}; "
                f"wanted {len(output) - filled} more bytes"
            )
        filled += count


def copy_range(
    source: BinaryIO,
    output: BinaryIO,
    offset: int,
    length: int,
    buffer: bytearray,
) -> None:
    source.seek(offset)
    buffer_view = memoryview(buffer)
    remaining = length
    while remaining:
        chunk_size = min(remaining, len(buffer))
        chunk = buffer_view[:chunk_size]
        count = source.readinto(chunk)
        if count is None or count == 0:
            raise EOFError(
                f"Unexpected end of input shard at byte {source.tell()}; "
                f"wanted {remaining} more bytes"
            )
        write_all(output, chunk[:count])
        remaining -= count


def write_interleaved(
    source: BinaryIO,
    output: BinaryIO,
    left: TensorInfo,
    right: TensorInfo,
    buffer: bytearray,
) -> None:
    row_count = left.shape[0]
    if row_count == 0:
        return
    left_row_bytes = left.nbytes // row_count
    right_row_bytes = right.nbytes // row_count
    if left_row_bytes != right_row_bytes:
        raise ValueError(
            f"Cannot interleave rows with different sizes: "
            f"{left_row_bytes} and {right_row_bytes} bytes"
        )
    if left_row_bytes == 0:
        return

    half_buffer = len(buffer) // 2
    if left_row_bytes > half_buffer:
        for row in range(row_count):
            copy_range(
                source,
                output,
                left.offset + row * left_row_bytes,
                left_row_bytes,
                buffer,
            )
            copy_range(
                source,
                output,
                right.offset + row * right_row_bytes,
                right_row_bytes,
                buffer,
            )
        return

    rows_per_batch = max(1, half_buffer // left_row_bytes)
    buffer_view = memoryview(buffer)
    for first_row in range(0, row_count, rows_per_batch):
        batch_rows = min(rows_per_batch, row_count - first_row)
        batch_bytes = batch_rows * left_row_bytes
        left_buffer = buffer_view[:batch_bytes]
        right_buffer = buffer_view[half_buffer : half_buffer + batch_bytes]
        read_exact_at(source, left.offset + first_row * left_row_bytes, left_buffer)
        read_exact_at(source, right.offset + first_row * right_row_bytes, right_buffer)
        for batch_row in range(batch_rows):
            row_start = batch_row * left_row_bytes
            row_end = row_start + left_row_bytes
            write_all(output, left_buffer[row_start:row_end])
            write_all(output, right_buffer[row_start:row_end])


def write_safetensors_streaming(
    input_path: Path,
    output_path: Path,
    tensors: list[OutputTensor],
    metadata: dict[str, str] | None,
    buffer_size: int,
) -> None:
    """Write a shard without materializing any complete tensor in memory."""
    if buffer_size < 1:
        raise ValueError("Streaming buffer size must be at least one byte")
    header = encode_safetensors_header(tensors, metadata)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        os.chmod(tmp_path, stat.S_IMODE(input_path.stat().st_mode))
        buffer = bytearray(buffer_size)
        with input_path.open("rb") as source, tmp_path.open("wb") as output:
            write_all(
                output,
                memoryview(SAFETENSORS_HEADER_LENGTH.pack(len(header))),
            )
            write_all(output, memoryview(header))
            for tensor in tensors:
                if tensor.interleaved_sources:
                    for left, right in zip(
                        tensor.sources, tensor.interleaved_sources, strict=True
                    ):
                        write_interleaved(source, output, left, right, buffer)
                else:
                    for source_tensor in tensor.sources:
                        copy_range(
                            source,
                            output,
                            source_tensor.offset,
                            source_tensor.nbytes,
                            buffer,
                        )
        os.replace(tmp_path, output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def convert_shard(
    input_dir: Path,
    output_dir: Path,
    shard: str,
    keys: list[str],
    expected_experts: int | None,
    fuse_w1_w3: bool,
    fused_name: str,
    verbose: bool,
    buffer_size: int,
) -> tuple[dict[str, str], int, int]:
    groups = collect_expert_groups(keys)
    inferred_experts = validate_groups(groups, expected_experts)
    expert_input_keys = {key for experts in groups.values() for key in experts.values()}
    output_weight_map: dict[str, str] = {}

    input_path = input_dir / shard
    output_path = output_dir / shard
    input_tensors, metadata = read_safetensors_header(input_path)
    missing_keys = sorted(set(keys) - input_tensors.keys())
    if missing_keys:
        raise KeyError(f"Keys from the index are missing in {input_path}: {missing_keys}")

    output_tensors: dict[str, OutputTensor] = {}

    def add_output(tensor: OutputTensor) -> None:
        if tensor.name in output_tensors:
            raise ValueError(
                f"Duplicate output tensor name while rewriting {shard}: {tensor.name}"
            )
        output_tensors[tensor.name] = tensor
        output_weight_map[tensor.name] = shard

    for key in sorted(keys):
        if key in expert_input_keys:
            continue
        tensor = input_tensors[key]
        add_output(
            OutputTensor(
                name=key,
                dtype=tensor.dtype,
                shape=tensor.shape,
                sources=(tensor,),
            )
        )

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
                add_output(
                    plan_fused_pair(
                        input_tensors,
                        groups[left_key],
                        groups[right_key],
                        output_name,
                        verbose,
                    )
                )
                fused_group_keys.update({left_key, right_key})

    for (prefix, name), expert_keys in sorted(groups.items()):
        if (prefix, name) in fused_group_keys:
            continue
        output_name = f"{prefix}.{name}"
        add_output(plan_merged_group(input_tensors, expert_keys, output_name, verbose))

    ordered_tensors = [output_tensors[name] for name in sorted(output_tensors)]
    bytes_written = sum(tensor.nbytes for tensor in ordered_tensors)
    write_safetensors_streaming(
        input_path,
        output_path,
        ordered_tensors,
        metadata,
        buffer_size,
    )
    return output_weight_map, inferred_experts, bytes_written


def copy_shard(
    input_dir: Path,
    output_dir: Path,
    shard: str,
    keys: list[str],
) -> dict[str, str]:
    shutil.copy2(input_dir / shard, output_dir / shard)
    return {key: shard for key in keys}


def shard_tensor_bytes(path: Path, keys: list[str]) -> int:
    tensors, _ = read_safetensors_header(path)
    missing_keys = sorted(set(keys) - tensors.keys())
    if missing_keys:
        raise KeyError(f"Keys from the index are missing in {path}: {missing_keys}")
    return sum(tensors[key].nbytes for key in keys)


def process_shard(
    input_dir: Path,
    output_dir: Path,
    shard: str,
    keys: list[str],
    expected_experts: int | None,
    fuse_w1_w3: bool,
    fused_name: str,
    verbose: bool,
    buffer_size: int,
) -> tuple[str, dict[str, str], int, str]:
    groups = collect_expert_groups(keys)
    if not groups:
        shard_size = shard_tensor_bytes(input_dir / shard, keys)
        if input_dir / shard == output_dir / shard:
            return shard, {key: shard for key in keys}, shard_size, "kept"
        return (
            shard,
            copy_shard(input_dir, output_dir, shard, keys),
            shard_size,
            "copied",
        )

    shard_weight_map, _, shard_size = convert_shard(
        input_dir=input_dir,
        output_dir=output_dir,
        shard=shard,
        keys=keys,
        expected_experts=expected_experts,
        fuse_w1_w3=fuse_w1_w3,
        fused_name=fused_name,
        verbose=verbose,
        buffer_size=buffer_size,
    )
    return shard, shard_weight_map, shard_size, "rewritten"


def copy_sidecar_files(input_dir: Path, output_dir: Path) -> None:
    if input_dir == output_dir:
        return
    skip_names = {"model.safetensors.index.json"}
    skip_suffixes = {".safetensors"}
    for path in input_dir.iterdir():
        if not path.is_file() or path.name in skip_names or path.suffix in skip_suffixes:
            continue
        shutil.copy2(path, output_dir / path.name)


def write_index(index_path: Path, index: dict[str, Any]) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{index_path.name}.",
        suffix=".tmp",
        dir=index_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, index_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


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
        default=None,
        help="Directory for the rewritten checkpoint. Required unless --in-place is set.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the checkpoint directly under --input-dir.",
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
        help="Number of safetensors shards to stream in parallel.",
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
        help="Deprecated compatibility option; the streaming writer does not use PyTorch.",
    )
    parser.add_argument(
        "--buffer-size-mib",
        type=int,
        default=DEFAULT_BUFFER_SIZE_MIB,
        help="Streaming copy buffer per worker in MiB.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print each merged tensor shape.")
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
    if args.buffer_size_mib < 1:
        raise ValueError("--buffer-size-mib must be >= 1")

    index = read_index(input_dir)
    weight_map = index["weight_map"]
    shards = collect_shards(weight_map)

    num_experts = args.num_experts
    config_path = input_dir / "config.json"
    if num_experts is None and config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            num_experts = json.load(f).get("n_routed_experts")

    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not args.overwrite
        and not args.dry_run
        and not args.in_place
    ):
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
        unchanged_action = "keeping" if args.in_place else "copying"
        print(
            f"Processing {expert_shards} rewrite shards with {args.workers} workers "
            f"({args.buffer_size_mib} MiB buffer each) and {unchanged_action} "
            f"{len(shards) - expert_shards} shards unchanged."
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
                args.buffer_size_mib * 1024 * 1024,
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
    write_index(output_dir / "model.safetensors.index.json", new_index)

    if args.copy_sidecars:
        copy_sidecar_files(input_dir, output_dir)

    if args.in_place:
        print(f"Done. Rewrote merged checkpoint in place at {output_dir}")
    else:
        print(f"Done. Wrote merged checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
