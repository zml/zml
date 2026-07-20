#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=2.0.0",
#   "safetensors>=0.7.0",
#   "torch>=2.5.0",
# ]
# ///
"""Rewrite Step 3.5 Flash MoE expert weights for ZML's fused MoE kernel.

The source checkpoint stores routed expert projections separately:

    model.layers.N.moe.gate_proj.weight
    model.layers.N.moe.up_proj.weight

ZML's Triton MoE kernel expects the two projections concatenated on the output
axis:

    model.layers.N.moe.gate_up_proj.weight

Default usage hardlinks the source model into a new directory and writes packed
sidecar shards, which keeps the source untouched and avoids loading full shards:

    uv run --script examples/llm/models/step3_5flash/rewrite_moe_weights.py \
      /var/models/step-3.5-flash /var/models/step-3.5-flash-wr

For upload-friendly output without unused gate/up tensors in the original MoE
shards, use compact mode:

    uv run --script examples/llm/models/step3_5flash/rewrite_moe_weights.py \
      --layout compact --copy-unchanged \
      /var/models/step-3.5-flash /var/models/step-3.5-flash-wr
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INDEX_NAME = "model.safetensors.index.json"
MOE_PROJ_RE = re.compile(r"^model\.layers\.(\d+)\.moe\.(gate_proj|up_proj)\.weight$")


@dataclass(frozen=True)
class PackSpec:
    layer: int
    gate_key: str
    up_key: str
    packed_key: str
    gate_file: str
    up_file: str
    packed_file: str


def load_index(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    if not isinstance(index.get("weight_map"), dict):
        raise ValueError(f"{path} does not contain a weight_map object")
    return index


def find_moe_packs(weight_map: dict[str, str], layout: str) -> list[PackSpec]:
    layers: dict[int, dict[str, tuple[str, str]]] = {}

    for key, filename in weight_map.items():
        match = MOE_PROJ_RE.match(key)
        if match is None:
            continue
        layer = int(match.group(1))
        kind = match.group(2)
        layers.setdefault(layer, {})[kind] = (key, filename)

    if not layers:
        raise ValueError("no routed MoE gate/up projection weights found in index")

    specs: list[PackSpec] = []
    missing: list[str] = []
    for layer in sorted(layers):
        entry = layers[layer]
        if "gate_proj" not in entry or "up_proj" not in entry:
            missing.append(f"layer {layer}")
            continue

        gate_key, gate_file = entry["gate_proj"]
        up_key, up_file = entry["up_proj"]
        packed_key = f"model.layers.{layer}.moe.gate_up_proj.weight"
        if packed_key in weight_map:
            raise ValueError(f"index already contains {packed_key}")

        if layout == "compact":
            if gate_file != up_file:
                raise ValueError(
                    f"compact layout requires gate/up in the same shard for layer {layer}: "
                    f"{gate_file} vs {up_file}"
                )
            packed_file = gate_file
        else:
            packed_file = f"model-layer-{layer:05d}.moe_gate_up.safetensors"

        specs.append(
            PackSpec(
                layer=layer,
                gate_key=gate_key,
                up_key=up_key,
                packed_key=packed_key,
                gate_file=gate_file,
                up_file=up_file,
                packed_file=packed_file,
            )
        )

    if missing:
        raise ValueError("missing gate/up projection pair for " + ", ".join(missing))
    return specs


def make_weight_map(weight_map: dict[str, str], specs: list[PackSpec]) -> dict[str, str]:
    by_removed_key: dict[str, PackSpec] = {}
    for spec in specs:
        by_removed_key[spec.gate_key] = spec
        by_removed_key[spec.up_key] = spec

    new_map: dict[str, str] = {}
    added: set[str] = set()
    for key, filename in weight_map.items():
        spec = by_removed_key.get(key)
        if spec is None:
            new_map[key] = filename
            continue
        if spec.packed_key not in added:
            new_map[spec.packed_key] = spec.packed_file
            added.add(spec.packed_key)

    for spec in specs:
        if spec.packed_key not in added:
            new_map[spec.packed_key] = spec.packed_file
    return new_map


def ensure_output_path(src_dir: Path, dst_dir: Path, force: bool) -> None:
    src_resolved = src_dir.resolve()
    dst_resolved = dst_dir.resolve()
    if src_resolved == dst_resolved:
        raise ValueError("source and output directories must be different")
    if src_resolved in dst_resolved.parents:
        raise ValueError("output directory must not be inside the source directory")
    if dst_dir.exists() and any(dst_dir.iterdir()) and not force:
        raise FileExistsError(f"{dst_dir} is not empty; pass --force to overwrite generated files")
    dst_dir.mkdir(parents=True, exist_ok=True)


def copy_or_link_file(src: Path, dst: Path, copy_file: bool, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not force:
            raise FileExistsError(f"{dst} already exists")
        dst.unlink()

    if src.is_symlink():
        target = os.readlink(src)
        os.symlink(target, dst)
        return

    if copy_file:
        shutil.copy2(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError as err:
            raise OSError(f"failed to hardlink {src} -> {dst}; retry with --copy-unchanged") from err


def prepare_output_file(path: Path, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if not force:
            raise FileExistsError(f"{path} already exists")
        path.unlink()


def copy_tree(
    src_dir: Path,
    dst_dir: Path,
    *,
    copy_unchanged: bool,
    skip_files: set[str],
    force: bool,
) -> None:
    for root, dirs, files in os.walk(src_dir):
        root_path = Path(root)
        rel_root = root_path.relative_to(src_dir)
        (dst_dir / rel_root).mkdir(parents=True, exist_ok=True)

        dirs[:] = [d for d in dirs if (rel_root / d).as_posix() not in skip_files]

        for filename in files:
            rel = (rel_root / filename).as_posix()
            if rel == INDEX_NAME or rel in skip_files:
                continue
            copy_or_link_file(src_dir / rel, dst_dir / rel, copy_unchanged, force)


def load_tensor(src_dir: Path, filename: str, key: str):
    from safetensors import safe_open

    with safe_open(str(src_dir / filename), framework="pt", device="cpu") as shard:
        if key not in shard.keys():
            raise KeyError(f"{key} not found in {filename}")
        return shard.get_tensor(key)


def load_shard(src_dir: Path, filename: str):
    from safetensors import safe_open

    with safe_open(str(src_dir / filename), framework="pt", device="cpu") as shard:
        metadata = shard.metadata()
        tensors = {key: shard.get_tensor(key) for key in shard.keys()}
    return tensors, metadata


def shard_metadata(src_dir: Path, filename: str):
    from safetensors import safe_open

    with safe_open(str(src_dir / filename), framework="pt", device="cpu") as shard:
        return shard.metadata()


def pack_gate_up(gate, up, spec: PackSpec):
    import torch

    if tuple(gate.shape) != tuple(up.shape):
        raise ValueError(
            f"layer {spec.layer}: gate/up shapes differ: {tuple(gate.shape)} vs {tuple(up.shape)}"
        )
    if gate.ndim != 3:
        raise ValueError(f"layer {spec.layer}: expected rank-3 expert weights, got {tuple(gate.shape)}")
    if gate.dtype != up.dtype:
        raise ValueError(f"layer {spec.layer}: dtype mismatch: {gate.dtype} vs {up.dtype}")
    return torch.cat((gate, up), dim=1).contiguous()


def write_sidecar_shards(src_dir: Path, dst_dir: Path, specs: list[PackSpec], force: bool) -> None:
    from safetensors.torch import save_file

    for spec in specs:
        output_path = dst_dir / spec.packed_file
        prepare_output_file(output_path, force)

        gate = load_tensor(src_dir, spec.gate_file, spec.gate_key)
        up = load_tensor(src_dir, spec.up_file, spec.up_key)
        packed = pack_gate_up(gate, up, spec)
        metadata = shard_metadata(src_dir, spec.gate_file)
        save_file({spec.packed_key: packed}, str(output_path), metadata=metadata)
        print(f"packed layer {spec.layer}: {spec.packed_key} -> {spec.packed_file}")


def write_compact_shards(src_dir: Path, dst_dir: Path, specs: list[PackSpec], force: bool) -> None:
    from collections import defaultdict

    from safetensors.torch import save_file

    by_file: dict[str, list[PackSpec]] = defaultdict(list)
    for spec in specs:
        by_file[spec.gate_file].append(spec)

    for filename, file_specs in sorted(by_file.items()):
        output_path = dst_dir / filename
        prepare_output_file(output_path, force)

        tensors, metadata = load_shard(src_dir, filename)
        for spec in file_specs:
            try:
                gate = tensors.pop(spec.gate_key)
                up = tensors.pop(spec.up_key)
            except KeyError as err:
                raise KeyError(f"{err.args[0]} not found in {filename}") from err
            tensors[spec.packed_key] = pack_gate_up(gate, up, spec)
            print(f"packed layer {spec.layer}: {spec.packed_key} -> {filename}")

        save_file(tensors, str(output_path), metadata=metadata)


def write_index(dst_dir: Path, index: dict[str, Any], new_weight_map: dict[str, str]) -> None:
    rewritten = dict(index)
    rewritten["weight_map"] = new_weight_map
    with (dst_dir / INDEX_NAME).open("w", encoding="utf-8") as f:
        json.dump(rewritten, f, indent=2)
        f.write("\n")


def rewrite(
    src_dir: Path,
    dst_dir: Path,
    *,
    layout: str,
    copy_unchanged: bool,
    force: bool,
    dry_run: bool,
) -> None:
    index = load_index(src_dir / INDEX_NAME)
    weight_map = index["weight_map"]
    specs = find_moe_packs(weight_map, layout)
    new_weight_map = make_weight_map(weight_map, specs)

    print(f"found {len(specs)} MoE layers to pack")
    if dry_run:
        for spec in specs:
            print(
                f"layer {spec.layer}: {spec.gate_key} + {spec.up_key} -> "
                f"{spec.packed_key} ({spec.packed_file})"
            )
        return

    ensure_output_path(src_dir, dst_dir, force)

    compact_files = {spec.gate_file for spec in specs} if layout == "compact" else set()
    copy_tree(
        src_dir,
        dst_dir,
        copy_unchanged=copy_unchanged,
        skip_files=compact_files,
        force=force,
    )

    if layout == "compact":
        write_compact_shards(src_dir, dst_dir, specs, force)
    else:
        write_sidecar_shards(src_dir, dst_dir, specs, force)

    write_index(dst_dir, index, new_weight_map)
    print(f"wrote rewritten index: {dst_dir / INDEX_NAME}")


def run_self_test() -> None:
    import torch
    from safetensors.torch import load_file, save_file

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "src"
        src.mkdir()

        gate = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        up = gate + 1000
        down = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        save_file(
            {
                "model.layers.3.moe.down_proj.weight": down,
                "model.layers.3.moe.gate_proj.weight": gate,
                "model.layers.3.moe.up_proj.weight": up,
                "model.layers.3.moe.gate.weight": torch.zeros(2, 4),
            },
            str(src / "model-00003.safetensors"),
        )
        with (src / INDEX_NAME).open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {"total_size": 0},
                    "weight_map": {
                        "model.layers.3.moe.down_proj.weight": "model-00003.safetensors",
                        "model.layers.3.moe.gate_proj.weight": "model-00003.safetensors",
                        "model.layers.3.moe.up_proj.weight": "model-00003.safetensors",
                        "model.layers.3.moe.gate.weight": "model-00003.safetensors",
                    },
                },
                f,
            )

        expected = torch.cat((gate, up), dim=1)

        sidecar = root / "sidecar"
        rewrite(src, sidecar, layout="sidecar", copy_unchanged=True, force=True, dry_run=False)
        sidecar_index = load_index(sidecar / INDEX_NAME)["weight_map"]
        assert "model.layers.3.moe.gate_proj.weight" not in sidecar_index
        assert "model.layers.3.moe.up_proj.weight" not in sidecar_index
        sidecar_tensors = load_file(str(sidecar / sidecar_index["model.layers.3.moe.gate_up_proj.weight"]))
        assert torch.equal(sidecar_tensors["model.layers.3.moe.gate_up_proj.weight"], expected)

        compact = root / "compact"
        rewrite(src, compact, layout="compact", copy_unchanged=True, force=True, dry_run=False)
        compact_index = load_index(compact / INDEX_NAME)["weight_map"]
        compact_tensors = load_file(str(compact / "model-00003.safetensors"))
        assert compact_index["model.layers.3.moe.gate_up_proj.weight"] == "model-00003.safetensors"
        assert "model.layers.3.moe.gate_proj.weight" not in compact_tensors
        assert "model.layers.3.moe.up_proj.weight" not in compact_tensors
        assert torch.equal(compact_tensors["model.layers.3.moe.gate_up_proj.weight"], expected)

    print("self-test passed")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, help="source model directory")
    parser.add_argument("output", nargs="?", type=Path, help="rewritten output model directory")
    parser.add_argument(
        "--layout",
        choices=("sidecar", "compact"),
        default="sidecar",
        help="sidecar writes packed-only shards; compact rewrites MoE shards without separate gate/up tensors",
    )
    parser.add_argument(
        "--copy-unchanged",
        action="store_true",
        help="copy unchanged files instead of hardlinking them",
    )
    parser.add_argument("--force", action="store_true", help="overwrite generated files in an existing output dir")
    parser.add_argument("--dry-run", action="store_true", help="print planned rewrites without writing files")
    parser.add_argument("--self-test", action="store_true", help="run a tiny synthetic rewrite test")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0

    if args.source is None or args.output is None:
        raise SystemExit("source and output are required unless --self-test is passed")

    rewrite(
        args.source,
        args.output,
        layout=args.layout,
        copy_unchanged=args.copy_unchanged,
        force=args.force,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
