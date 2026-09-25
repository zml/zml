#!/usr/bin/env python3
"""Create an opt-in ZML Llama checkpoint with E4M3FN projection weights.

Requires PyTorch and NumPy. Embeddings, the vocabulary head, and normalization
weights remain unchanged. Each transformer projection gets an F32 power-of-two
scale per output channel. This is ZML's weight-storage format, not a checkpoint
that an unmodified Hugging Face loader can execute correctly.
"""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct

import numpy as np
import torch


def read_header(path):
    with path.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        return json.loads(source.read(header_size)), 8 + header_size


def quantized_projection(name):
    return name.startswith("model.layers.") and name.endswith(
        tuple(f".{projection}.weight" for projection in (
            "q_proj", "k_proj", "v_proj", "o_proj",
            "up_proj", "gate_proj", "down_proj",
        ))
    )


def quantize_rows(values):
    if not torch.isfinite(values).all():
        raise ValueError("Nonfinite checkpoint weight")
    maximum = values.abs().amax(dim=1)
    exponent = torch.ceil(torch.log2(torch.where(maximum > 0, maximum / 448, 1)))
    if not ((exponent >= -126) & (exponent <= 120)).all():
        raise ValueError("Weight scale is outside the supported normal F32 range")
    scale = torch.exp2(exponent)
    encoded = (values / scale[:, None]).to(torch.float8_e4m3fn)
    restored = encoded.float() * scale[:, None]
    if not torch.isfinite(restored).all():
        raise ValueError("FP8 conversion overflowed")
    return encoded, scale, restored


def convert_shard(source_path, destination_path, statistics):
    header, data_start = read_header(source_path)
    entries = sorted(
        ((name, meta) for name, meta in header.items() if name != "__metadata__"),
        key=lambda item: item[1]["data_offsets"][0],
    )
    out_header = {"__metadata__": header.get("__metadata__", {"format": "pt"})}
    offset = 0
    for name, meta in entries:
        size = meta["data_offsets"][1] - meta["data_offsets"][0]
        converted = quantized_projection(name)
        if converted:
            if name.removesuffix("weight") + "weight_scale" in header:
                raise ValueError(f"Checkpoint already contains a scale for {name}")
            if meta["dtype"] != "BF16" or len(meta["shape"]) != 2:
                raise ValueError(f"Expected a BF16 matrix for {name}")
            size //= 2
        out_header[name] = {
            "dtype": "F8_E4M3" if converted else meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [offset, offset + size],
        }
        offset += size
        if converted:
            rows = meta["shape"][0]
            out_header[name.removesuffix("weight") + "weight_scale"] = {
                "dtype": "F32", "shape": [rows],
                "data_offsets": [offset, offset + rows * 4],
            }
            offset += rows * 4
    encoded_header = json.dumps(out_header, separators=(",", ":")).encode()
    encoded_header += b" " * (-len(encoded_header) % 8)
    with source_path.open("rb") as source, destination_path.open("xb") as destination:
        destination.write(struct.pack("<Q", len(encoded_header)))
        destination.write(encoded_header)
        for name, meta in entries:
            begin, end = meta["data_offsets"]
            source.seek(data_start + begin)
            if not quantized_projection(name):
                remaining = end - begin
                while remaining:
                    chunk = source.read(min(remaining, 8 << 20))
                    if not chunk:
                        raise EOFError(name)
                    destination.write(chunk)
                    remaining -= len(chunk)
                continue
            rows, columns = meta["shape"]
            scales = []
            square_error = 0.0
            square_weight = 0.0
            maximum_error = 0.0
            for row in range(0, rows, 256):
                count = min(256, rows - row)
                raw = source.read(count * columns * 2)
                if len(raw) != count * columns * 2:
                    raise EOFError(name)
                bits = np.frombuffer(raw, dtype="<u2").copy().reshape(count, columns)
                values = torch.from_numpy(bits).view(torch.bfloat16).float()
                encoded, scale, restored = quantize_rows(values)
                destination.write(encoded.view(torch.uint8).numpy().tobytes())
                scales.append(scale.numpy().astype("<f4", copy=False).tobytes())
                error = restored - values
                square_error += error.double().square().sum().item()
                square_weight += values.double().square().sum().item()
                maximum_error = max(maximum_error, error.abs().max().item())
            for scale in scales:
                destination.write(scale)
            statistics[name] = {
                "relative_l2_error": math.sqrt(square_error / square_weight) if square_weight else 0.0,
                "maximum_absolute_error": maximum_error,
                "shape": meta["shape"],
            }
            print(name, statistics[name], flush=True)
        if destination.tell() != 8 + len(encoded_header) + offset:
            raise ValueError("Output size does not match the safetensors header")
    return {name: destination_path.name for name in out_header if name != "__metadata__"}, offset


def convert(source, destination):
    source = source.resolve()
    destination = destination.absolute()
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "llama":
        raise ValueError("This converter supports Llama checkpoints only")
    if destination.exists():
        raise FileExistsError(destination)
    partial = destination.with_name(destination.name + ".partial")
    partial.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    statistics = {}
    try:
        index = json.loads((source / "model.safetensors.index.json").read_text())
        weight_map = {}
        total_size = 0
        for shard in sorted(set(index["weight_map"].values())):
            if Path(shard).name != shard:
                raise ValueError("Shard names must be simple filenames")
            names, size = convert_shard(source / shard, partial / shard, statistics)
            if weight_map.keys() & names.keys():
                raise ValueError("Duplicate tensor name")
            weight_map.update(names)
            total_size += size
        if len(statistics) == 0:
            raise ValueError("No Llama transformer projections found")
        for path in source.iterdir():
            if path.is_file() and path.suffix != ".safetensors" and path.name not in (
                "model.safetensors.index.json", "config.json",
            ):
                shutil.copyfile(path, partial / path.name)
        config["zml_weight_storage"] = "fp8_e4m3fn_pow2_channel"
        (partial / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (partial / "model.safetensors.index.json").write_text(json.dumps({
            "metadata": {"total_size": total_size}, "weight_map": weight_map,
        }, indent=2) + "\n")
        (partial / "zml-fp8-report.json").write_text(json.dumps({
            "format": "fp8_e4m3fn_pow2_channel", "source": str(source),
            "converted_matrices": len(statistics), "total_tensor_bytes": total_size,
            "weights": statistics,
        }, indent=2) + "\n")
        os.rename(partial, destination)
    except BaseException:
        # Keep incomplete output for diagnosis; never overwrite or delete the source.
        print(f"Incomplete checkpoint retained at {partial}", flush=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)
