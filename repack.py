#!/usr/bin/env python3
"""Merge DeepSeek V4 per-expert safetensors keys with bounded memory.

The Hugging Face checkpoint stores routed expert tensors as separate keys:

    layers.0.ffn.experts.0.w1.scale
    layers.0.ffn.experts.1.w1.scale
    ...

This script rewrites them as stacked tensors and, by default, fuses w1/w3:

    layers.0.ffn.experts.w13.scale

with expert id as the leading dimension. w1 and w3 are interleaved along the
output/intermediate dimension as w1[0], w3[0], w1[1], w3[1], ... so fused
SwiGLU kernels can split paired columns locally. They can instead be concatenated
as all of w1 followed by all of w3.

Safetensors shards are streamed directly instead of being materialized in a
dictionary of Torch tensors. Peak working memory is therefore bounded by
--buffer-size-mb per worker rather than by the size of a shard. CPU streaming
does not import Torch. An optional CUDA or MPS device can perform the w1/w3
interleave in bounded chunks; unchanged and already-contiguous data still use
direct CPU streaming because that work is limited by storage bandwidth.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import json
import os
import re
import shutil
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
DEFAULT_WORKERS = 1
DEFAULT_BUFFER_SIZE_MB = 8
SAFETENSORS_HEADER_LENGTH = struct.Struct("<Q")
try:
    IOV_MAX = max(2, int(os.sysconf("SC_IOV_MAX")))
except (AttributeError, OSError, TypeError, ValueError):
    IOV_MAX = 1024


@dataclass(frozen=True)
class TensorInfo:
    name: str
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
    interleave: bool = False

    @property
    def nbytes(self) -> int:
        return sum(source.nbytes for source in self.sources)


def is_repacked_tensor(output: OutputTensor) -> bool:
    return any(EXPERT_RE.match(source.name) is not None for source in output.sources)


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
        formatted = ", ".join(
            f"{key}: {len(value)}" for key, value in sorted(groups.items())
        )
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
            raise ValueError(
                f"Non-contiguous expert ids for {key}; missing={missing}, extra={extra}"
            )

    return inferred_experts


def read_safetensors_header(
    path: Path,
) -> tuple[dict[str, str] | None, dict[str, TensorInfo]]:
    file_size = path.stat().st_size
    with path.open("rb") as file:
        encoded_length = file.read(SAFETENSORS_HEADER_LENGTH.size)
        if len(encoded_length) != SAFETENSORS_HEADER_LENGTH.size:
            raise ValueError(f"Invalid safetensors header in {path}")
        (header_length,) = SAFETENSORS_HEADER_LENGTH.unpack(encoded_length)
        if header_length > file_size - SAFETENSORS_HEADER_LENGTH.size:
            raise ValueError(f"Safetensors header exceeds file size in {path}")
        encoded_header = file.read(header_length)
        if len(encoded_header) != header_length:
            raise ValueError(f"Truncated safetensors header in {path}")

    try:
        header = json.loads(encoded_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid safetensors JSON header in {path}") from error
    if not isinstance(header, dict):
        raise ValueError(f"Safetensors header is not an object in {path}")

    metadata = header.pop("__metadata__", None)
    if metadata is not None and (
        not isinstance(metadata, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in metadata.items()
        )
    ):
        raise ValueError(f"Invalid safetensors metadata in {path}")

    data_start = SAFETENSORS_HEADER_LENGTH.size + header_length
    data_size = file_size - data_start
    tensors: dict[str, TensorInfo] = {}
    for name, descriptor in header.items():
        if not isinstance(name, str) or not isinstance(descriptor, dict):
            raise ValueError(f"Invalid tensor descriptor in {path}: {name!r}")
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        offsets = descriptor.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not all(isinstance(dimension, int) and dimension >= 0 for dimension in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(offset, int) for offset in offsets)
        ):
            raise ValueError(f"Invalid tensor descriptor for {name} in {path}")
        begin, end = offsets
        if begin < 0 or end < begin or end > data_size:
            raise ValueError(f"Invalid data offsets for {name} in {path}: {offsets}")
        tensors[name] = TensorInfo(
            name=name,
            dtype=dtype,
            shape=tuple(shape),
            offset=data_start + begin,
            nbytes=end - begin,
        )
    return metadata, tensors


def get_tensor_info(
    tensors: dict[str, TensorInfo],
    name: str,
    path: Path,
) -> TensorInfo:
    try:
        return tensors[name]
    except KeyError as error:
        raise ValueError(f"Index key {name} is missing from {path}") from error


def group_sources(
    tensors: dict[str, TensorInfo],
    expert_keys: dict[int, str],
    output_name: str,
    path: Path,
) -> tuple[TensorInfo, ...]:
    sources = tuple(
        get_tensor_info(tensors, expert_keys[expert_id], path)
        for expert_id in range(len(expert_keys))
    )
    first = sources[0]
    for expert_id, source in enumerate(sources[1:], start=1):
        if source.shape != first.shape:
            raise ValueError(
                f"Shape mismatch while building {output_name}: expert 0 has "
                f"{first.shape}, expert {expert_id} has {source.shape}"
            )
        if source.dtype != first.dtype:
            raise ValueError(
                f"Dtype mismatch while building {output_name}: expert 0 has "
                f"{first.dtype}, expert {expert_id} has {source.dtype}"
            )
        if source.nbytes != first.nbytes:
            raise ValueError(
                f"Byte-size mismatch while building {output_name}: expert 0 has "
                f"{first.nbytes} bytes, expert {expert_id} has {source.nbytes}"
            )
    return sources


def build_output_tensors(
    input_path: Path,
    keys: list[str],
    expected_experts: int | None,
    fuse_w1_w3: bool,
    fused_name: str,
    fuse_layout: str,
) -> tuple[dict[str, str] | None, list[OutputTensor], int]:
    metadata, tensors = read_safetensors_header(input_path)
    groups = collect_expert_groups(keys)
    inferred_experts = validate_groups(groups, expected_experts)
    expert_input_keys = {key for experts in groups.values() for key in experts.values()}
    outputs: dict[str, OutputTensor] = {}

    def add_output(output: OutputTensor) -> None:
        if output.name in outputs:
            raise ValueError(f"Output tensor name collision: {output.name}")
        outputs[output.name] = output

    for key in sorted(keys):
        if key in expert_input_keys:
            continue
        source = get_tensor_info(tensors, key, input_path)
        add_output(
            OutputTensor(
                name=key,
                dtype=source.dtype,
                shape=source.shape,
                sources=(source,),
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
                left_sources = group_sources(
                    tensors, groups[left_key], output_name, input_path
                )
                right_sources = group_sources(
                    tensors, groups[right_key], output_name, input_path
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
                        f"Byte-size mismatch while building {output_name}: left has "
                        f"{left_first.nbytes} bytes, right has {right_first.nbytes}"
                    )
                if not left_first.shape:
                    raise ValueError(f"Cannot fuse scalar tensors into {output_name}")

                sources = tuple(
                    source
                    for pair in zip(left_sources, right_sources, strict=True)
                    for source in pair
                )
                add_output(
                    OutputTensor(
                        name=output_name,
                        dtype=left_first.dtype,
                        shape=(
                            len(left_sources),
                            left_first.shape[0] + right_first.shape[0],
                            *left_first.shape[1:],
                        ),
                        sources=sources,
                        interleave=fuse_layout == "interleave",
                    )
                )
                fused_group_keys.update({left_key, right_key})

    for (prefix, name), expert_keys in sorted(groups.items()):
        if (prefix, name) in fused_group_keys:
            continue
        output_name = f"{prefix}.{name}"
        sources = group_sources(tensors, expert_keys, output_name, input_path)
        first = sources[0]
        add_output(
            OutputTensor(
                name=output_name,
                dtype=first.dtype,
                shape=(len(sources), *first.shape),
                sources=sources,
            )
        )

    return metadata, [outputs[name] for name in sorted(outputs)], inferred_experts


def encode_safetensors_header(
    outputs: list[OutputTensor],
    metadata: dict[str, str] | None,
) -> tuple[bytes, int]:
    header: dict[str, Any] = {}
    offset = 0
    for output in outputs:
        end = offset + output.nbytes
        header[output.name] = {
            "dtype": output.dtype,
            "shape": list(output.shape),
            "data_offsets": [offset, end],
        }
        offset = end
    if metadata is not None:
        header["__metadata__"] = metadata

    encoded = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    return encoded, offset


def read_exact_into(
    input_file: BinaryIO,
    offset: int,
    output: memoryview,
) -> None:
    input_file.seek(offset)
    position = 0
    while position < len(output):
        count = input_file.readinto(output[position:])
        if count is None or count <= 0:
            raise EOFError(f"Unexpected end of {input_file.name}")
        position += count


def write_all(output_file: BinaryIO, data: memoryview | bytes | bytearray) -> None:
    view = data if isinstance(data, memoryview) else memoryview(data)
    position = 0
    while position < len(view):
        count = output_file.write(view[position:])
        if count is None or count <= 0:
            raise OSError(f"Failed to write {output_file.name}")
        position += count


def writev_all(output_file: BinaryIO, buffers: list[memoryview]) -> None:
    if not hasattr(os, "writev"):
        for buffer in buffers:
            write_all(output_file, buffer)
        return

    buffer_index = 0
    buffer_offset = 0
    while buffer_index < len(buffers):
        pending = buffers[buffer_index:]
        if buffer_offset:
            pending = [pending[0][buffer_offset:], *pending[1:]]
        count = os.writev(output_file.fileno(), pending)
        if count <= 0:
            raise OSError(f"Failed to write {output_file.name}")

        while count and buffer_index < len(buffers):
            remaining = len(buffers[buffer_index]) - buffer_offset
            if count < remaining:
                buffer_offset += count
                count = 0
            else:
                count -= remaining
                buffer_index += 1
                buffer_offset = 0


def copy_source(
    input_file: BinaryIO,
    output_file: BinaryIO,
    source: TensorInfo,
    buffer: bytearray,
) -> None:
    view = memoryview(buffer)
    offset = source.offset
    remaining = source.nbytes
    while remaining:
        count = min(remaining, len(buffer))
        chunk = view[:count]
        read_exact_into(input_file, offset, chunk)
        write_all(output_file, chunk)
        offset += count
        remaining -= count


def interleave_rows_cpu(
    input_file: BinaryIO,
    output_file: BinaryIO,
    left: TensorInfo,
    right: TensorInfo,
    output_buffer: bytearray,
) -> None:
    rows = left.shape[0]
    if rows == 0:
        return
    if left.nbytes % rows:
        raise ValueError(f"Tensor byte size is not row-aligned: {left.name}")
    row_nbytes = left.nbytes // rows
    if row_nbytes == 0:
        return

    rows_per_chunk = max(
        1,
        min(
            len(output_buffer) // (2 * row_nbytes),
            IOV_MAX // 2,
        ),
    )
    buffer_view = memoryview(output_buffer)

    for row_start in range(0, rows, rows_per_chunk):
        row_count = min(rows_per_chunk, rows - row_start)
        source_nbytes = row_count * row_nbytes
        output_nbytes = 2 * source_nbytes
        if output_nbytes > len(output_buffer):
            # A single row pair is larger than the requested buffer. Stream each
            # complete row directly; row order itself is the interleave.
            for row in range(row_start, row_start + row_count):
                copy_source(
                    input_file,
                    output_file,
                    TensorInfo(
                        name=left.name,
                        dtype=left.dtype,
                        shape=left.shape[1:],
                        offset=left.offset + row * row_nbytes,
                        nbytes=row_nbytes,
                    ),
                    output_buffer,
                )
                copy_source(
                    input_file,
                    output_file,
                    TensorInfo(
                        name=right.name,
                        dtype=right.dtype,
                        shape=right.shape[1:],
                        offset=right.offset + row * row_nbytes,
                        nbytes=row_nbytes,
                    ),
                    output_buffer,
                )
            continue

        left_chunk = buffer_view[:source_nbytes]
        right_chunk = buffer_view[source_nbytes:output_nbytes]
        read_exact_into(
            input_file,
            left.offset + row_start * row_nbytes,
            left_chunk,
        )
        read_exact_into(
            input_file,
            right.offset + row_start * row_nbytes,
            right_chunk,
        )
        row_buffers: list[memoryview] = []
        for row in range(row_count):
            begin = row * row_nbytes
            end = begin + row_nbytes
            row_buffers.extend((left_chunk[begin:end], right_chunk[begin:end]))
        writev_all(output_file, row_buffers)


def flat_byte_view(tensor: Any) -> memoryview:
    view = memoryview(tensor.numpy())
    if view.ndim != 1:
        view = view.cast("B")
    return view


def interleave_rows_device(
    input_file: BinaryIO,
    output_file: BinaryIO,
    left: TensorInfo,
    right: TensorInfo,
    buffer_size: int,
    device: str,
) -> None:
    torch = importlib.import_module("torch")
    rows = left.shape[0]
    if rows == 0:
        return
    if left.nbytes % rows:
        raise ValueError(f"Tensor byte size is not row-aligned: {left.name}")
    row_nbytes = left.nbytes // rows
    if row_nbytes == 0:
        return
    if 2 * row_nbytes > buffer_size:
        copy_buffer = bytearray(buffer_size)
        for row in range(rows):
            for source in (left, right):
                copy_source(
                    input_file,
                    output_file,
                    TensorInfo(
                        name=source.name,
                        dtype=source.dtype,
                        shape=source.shape[1:],
                        offset=source.offset + row * row_nbytes,
                        nbytes=row_nbytes,
                    ),
                    copy_buffer,
                )
        return

    rows_per_chunk = max(1, buffer_size // (2 * row_nbytes))
    pin_memory = device.startswith("cuda")
    for row_start in range(0, rows, rows_per_chunk):
        row_count = min(rows_per_chunk, rows - row_start)
        source_nbytes = row_count * row_nbytes
        host = torch.empty(
            source_nbytes,
            dtype=torch.uint8,
            device="cpu",
            pin_memory=pin_memory,
        )
        host_view = flat_byte_view(host)

        read_exact_into(
            input_file,
            left.offset + row_start * row_nbytes,
            host_view,
        )
        left_device = host.to(device=device, non_blocking=False, copy=True).reshape(
            row_count, row_nbytes
        )

        read_exact_into(
            input_file,
            right.offset + row_start * row_nbytes,
            host_view,
        )
        right_device = host.to(device=device, non_blocking=False, copy=True).reshape(
            row_count, row_nbytes
        )

        interleaved = torch.stack((left_device, right_device), dim=1)
        host_output = interleaved.to(device="cpu", non_blocking=False).contiguous()
        write_all(output_file, flat_byte_view(host_output))


def write_safetensors(
    input_path: Path,
    output_path: Path,
    metadata: dict[str, str] | None,
    outputs: list[OutputTensor],
    buffer_size: int,
    device: str,
) -> int:
    encoded_header, total_size = encode_safetensors_header(outputs, metadata)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        with input_path.open("rb", buffering=0) as input_file, tmp_path.open(
            "wb", buffering=0
        ) as output_file:
            write_all(
                output_file,
                SAFETENSORS_HEADER_LENGTH.pack(len(encoded_header)),
            )
            write_all(output_file, encoded_header)

            direct_buffer: bytearray | None = None
            for output in outputs:
                if not output.interleave:
                    if direct_buffer is None:
                        direct_buffer = bytearray(buffer_size)
                    for source in output.sources:
                        copy_source(input_file, output_file, source, direct_buffer)
                    continue

                if len(output.sources) % 2:
                    raise ValueError(f"Interleave source count is odd for {output.name}")
                if device == "cpu":
                    if direct_buffer is None:
                        direct_buffer = bytearray(buffer_size)
                    for source_index in range(0, len(output.sources), 2):
                        interleave_rows_cpu(
                            input_file,
                            output_file,
                            output.sources[source_index],
                            output.sources[source_index + 1],
                            direct_buffer,
                        )
                else:
                    # Release the large CPU copy buffer before allocating pinned
                    # host/device staging buffers.
                    direct_buffer = None
                    for source_index in range(0, len(output.sources), 2):
                        interleave_rows_device(
                            input_file,
                            output_file,
                            output.sources[source_index],
                            output.sources[source_index + 1],
                            buffer_size,
                            device,
                        )

        os.replace(tmp_path, output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return total_size


def copy_shard(input_path: Path, output_path: Path) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(input_path, tmp_path)
        os.replace(tmp_path, output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def process_shard(
    input_dir: Path,
    output_dir: Path,
    shard: str,
    keys: list[str],
    expected_experts: int | None,
    fuse_w1_w3: bool,
    fused_name: str,
    fuse_layout: str,
    verbose: bool,
    buffer_size: int,
    device: str,
) -> tuple[str, dict[str, str], int, str]:
    input_path = input_dir / shard
    output_path = output_dir / shard
    groups = collect_expert_groups(keys)
    if not groups:
        _, tensors = read_safetensors_header(input_path)
        shard_size = sum(
            get_tensor_info(tensors, key, input_path).nbytes for key in keys
        )
        if input_path == output_path:
            return shard, {key: shard for key in keys}, shard_size, "kept"
        copy_shard(input_path, output_path)
        return shard, {key: shard for key in keys}, shard_size, "copied"

    metadata, outputs, _ = build_output_tensors(
        input_path=input_path,
        keys=keys,
        expected_experts=expected_experts,
        fuse_w1_w3=fuse_w1_w3,
        fused_name=fused_name,
        fuse_layout=fuse_layout,
    )
    if verbose:
        for output in outputs:
            if is_repacked_tensor(output):
                print(f"  {output.name}: {output.shape} {output.dtype}")
    shard_size = write_safetensors(
        input_path=input_path,
        output_path=output_path,
        metadata=metadata,
        outputs=outputs,
        buffer_size=buffer_size,
        device=device,
    )
    return (
        shard,
        {output.name: shard for output in outputs},
        shard_size,
        "rewritten",
    )


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
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(index, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(tmp_path, index_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"

    torch = importlib.import_module("torch")
    if requested in {"auto", "gpu"}:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if requested == "auto":
            return "cpu"
        raise ValueError("--device=gpu requested, but CUDA and MPS are unavailable")

    try:
        device = torch.device(requested)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"Invalid device: {requested}") from error
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"--device={requested} requested, but CUDA is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"--device={requested} requested, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available"
            )
    elif device.type == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError("--device=mps requested, but MPS is unavailable")
    else:
        raise ValueError(
            f"Unsupported device {requested!r}; use cpu, gpu, auto, cuda[:N], or mps"
        )
    return str(device)


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
        help=(
            "Expected routed experts per layer. Defaults to config.json "
            "n_routed_experts, then inference."
        ),
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
        help="Fuse routed expert w1 and w3 along the output dimension.",
    )
    parser.add_argument(
        "--fuse-layout",
        choices=("interleave", "concat"),
        default="interleave",
        help="Layout for fused w1/w3 tensors: alternate rows or concatenate w3 after w1.",
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
        help="Shards to process in parallel. Each worker uses its own bounded buffer.",
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
        help="Torch host threads used by GPU mode.",
    )
    parser.add_argument(
        "--buffer-size-mb",
        type=int,
        default=DEFAULT_BUFFER_SIZE_MB,
        help="Maximum streaming chunk size per worker in MiB (default: 8).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Interleave device: cpu, gpu, auto, cuda, cuda:N, or mps (default: cpu).",
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
    if args.buffer_size_mb < 1:
        raise ValueError("--buffer-size-mb must be >= 1")

    device = resolve_device(args.device)
    if device != "cpu":
        if args.parallel_backend == "process":
            raise ValueError("GPU mode requires --parallel-backend=thread")
        torch = importlib.import_module("torch")
        torch.set_num_threads(args.torch_threads)

    index = read_index(input_dir)
    weight_map = index["weight_map"]
    shards = collect_shards(weight_map)

    num_experts = args.num_experts
    config_path = input_dir / "config.json"
    if num_experts is None and config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            num_experts = json.load(file).get("n_routed_experts")

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

    total_expert_groups = sum(
        len(collect_expert_groups(keys)) for keys in shards.values()
    )
    expert_shards = sum(1 for keys in shards.values() if collect_expert_groups(keys))
    expert_count = (
        f"{num_experts} experts each"
        if num_experts is not None
        else "expert count inferred per shard"
    )
    print(
        f"Found {len(shards)} shards and {total_expert_groups} expert tensor groups "
        f"({expert_count})."
    )
    if not args.dry_run:
        unchanged_action = "keeping" if args.in_place else "copying"
        print(
            f"Processing {expert_shards} rewrite shards with {args.workers} workers, "
            f"{args.buffer_size_mb} MiB buffers, and device={device}; "
            f"{unchanged_action} {len(shards) - expert_shards} shards unchanged."
        )

    if args.dry_run:
        for shard, keys in shards.items():
            groups = collect_expert_groups(keys)
            if not groups:
                continue
            _, outputs, _ = build_output_tensors(
                input_path=input_dir / shard,
                keys=keys,
                expected_experts=num_experts,
                fuse_w1_w3=args.fuse_w1_w3,
                fused_name=args.fused_name,
                fuse_layout=args.fuse_layout,
            )
            print(shard)
            for output in outputs:
                if is_repacked_tensor(output):
                    print(f"  {output.name}: {output.shape} {output.dtype}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    new_weight_map: dict[str, str] = {}
    total_size = 0
    max_workers = min(args.workers, len(shards))

    buffer_size = args.buffer_size_mb * 1024 * 1024
    if max_workers <= 1:
        results = (
            process_shard(
                input_dir,
                output_dir,
                shard,
                keys,
                num_experts,
                args.fuse_w1_w3,
                args.fused_name,
                args.fuse_layout,
                args.verbose,
                buffer_size,
                device,
            )
            for shard, keys in shards.items()
        )
        for shard, shard_weight_map, shard_size, action in results:
            new_weight_map.update(shard_weight_map)
            total_size += shard_size
            print(f"{action.capitalize()} {shard}")
    else:
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
                    args.fuse_layout,
                    args.verbose,
                    buffer_size,
                    device,
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
