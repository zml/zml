"""Dump moe.{up,gate,down}_proj.weight[0, 0, :8] for layers 41..44.

Usage:
    python3 dump_moe_first8.py <model_dir_or_path>

<model_dir_or_path> can be either:
  - a directory containing model.safetensors.index.json + the shards
  - a single .safetensors file (will look for the tensors inside it)
"""

import json
import sys
from pathlib import Path

from safetensors import safe_open


def open_index(model_dir: Path):
    index = model_dir / "model.safetensors.index.json"
    if not index.exists():
        raise FileNotFoundError(f"missing {index}")
    weight_map = json.loads(index.read_text())["weight_map"]
    return weight_map


def get_tensor(model_dir: Path, weight_map, key: str):
    shard = model_dir / weight_map[key]
    with safe_open(shard, framework="pt") as f:
        return f.get_tensor(key), shard.name


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)

    path = Path(sys.argv[1]).expanduser().resolve()

    if path.is_file() and path.suffix == ".safetensors":
        # Single-file mode
        with safe_open(path, framework="pt") as f:
            keys = set(f.keys())
            for li in (41, 42, 43, 44):
                print(f"=== layer {li} ===")
                for name in ("up_proj", "gate_proj", "down_proj"):
                    k = f"model.layers.{li}.moe.{name}.weight"
                    if k not in keys:
                        print(f"  {name}: MISSING")
                        continue
                    t = f.get_tensor(k)
                    print(f"  {name}: shape={tuple(t.shape)} dtype={t.dtype} "
                          f"[0,0,:8]={t[0, 0, :8].float().tolist()} "
                          f"norm={t.float().norm().item():.4f}")
        return

    if not path.is_dir():
        raise SystemExit(f"not a directory or .safetensors file: {path}")

    weight_map = open_index(path)
    for li in (41, 42, 43, 44):
        print(f"=== layer {li} ===")
        for name in ("up_proj", "gate_proj", "down_proj"):
            k = f"model.layers.{li}.moe.{name}.weight"
            if k not in weight_map:
                print(f"  {name}: MISSING from weight_map")
                continue
            t, shard = get_tensor(path, weight_map, k)
            print(f"  {name}: shard={shard} shape={tuple(t.shape)} "
                  f"dtype={t.dtype} "
                  f"[0,0,:8]={t[0, 0, :8].float().tolist()} "
                  f"norm={t.float().norm().item():.4f}")
        # router (HF flattens these directly under moe)
        for sub, slicer in (
            ("gate.weight", lambda t: t[0, :8]),
            ("router_bias", lambda t: t[:8]),
        ):
            k = f"model.layers.{li}.moe.{sub}"
            if k not in weight_map:
                print(f"  {sub}: MISSING from weight_map")
                continue
            t, shard = get_tensor(path, weight_map, k)
            print(f"  {sub}: shard={shard} shape={tuple(t.shape)} "
                  f"dtype={t.dtype} "
                  f"slice={slicer(t).float().tolist()} "
                  f"norm={t.float().norm().item():.4f}")


if __name__ == "__main__":
    main()
