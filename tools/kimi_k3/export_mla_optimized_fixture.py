#!/usr/bin/env python3
"""Export deterministic compact-cache MLA decode edge cases on NVIDIA CUDA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# cuBLAS requires this setting before CUDA initialization for reproducible
# reduction ordering in deterministic matrix multiplications.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from safetensors.torch import save_file


ROOT = Path("/dev/shm/kimi-k3")
OUTPUT = ROOT / "artifacts/fixtures/milestone-18"
SEED = 20260829
HEADS = 96
LATENT = 512
EXTRA = 64
CASES = (
    ("capacity1_valid1", 1, 1, False),
    ("capacity32_valid31", 32, 31, False),
    ("capacity32_valid32", 32, 32, False),
    ("capacity64_valid33", 64, 33, False),
    ("capacity64_valid63", 64, 63, False),
    ("capacity64_valid64", 64, 64, True),
    ("capacity128_valid65", 128, 65, False),
    ("capacity128_valid127", 128, 127, False),
    ("capacity128_valid128", 128, 128, False),
    ("capacity4096_valid4096", 4096, 4096, True),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic(shape: tuple[int, ...], phase: float, scale: float) -> torch.Tensor:
    elements = 1
    for dim in shape:
        elements *= dim
    index = torch.arange(elements, device="cuda", dtype=torch.float32).reshape(shape)
    return (torch.sin(index * 0.013 + phase) * scale + torch.cos(index * 0.0037 + phase * 0.3) * scale * 0.25).to(torch.bfloat16)


def add_case(tensors: dict[str, torch.Tensor], name: str, capacity: int, valid: int) -> None:
    q_absorbed = deterministic((1, HEADS, 1, LATENT), 0.2 + capacity * 0.001, 0.035)
    q_extra = deterministic((1, HEADS, 1, EXTRA), 0.7 + valid * 0.001, 0.05)
    compressed = deterministic((1, capacity, LATENT), 1.1 + capacity * 0.0001, 0.08)
    extra = deterministic((1, capacity, EXTRA), 1.7 + valid * 0.0001, 0.06)
    content = torch.einsum("bhqr,bkr->bhqk", q_absorbed, compressed)
    extra_score = torch.einsum("bhqe,bke->bhqk", q_extra, extra)
    score = (content + extra_score) * (192.0**-0.5)
    positions = torch.arange(capacity, device="cuda")
    score = score.masked_fill(positions >= valid, -torch.inf)
    probability = torch.softmax(score.float(), dim=-1).to(torch.bfloat16)
    expected = torch.einsum("bhqk,bkr->bhqr", probability, compressed)
    values = {
        "q_absorbed": q_absorbed,
        "q_extra": q_extra,
        "compressed": compressed,
        "extra": extra,
        "valid_tokens": torch.tensor([valid], device="cuda", dtype=torch.int32),
        "expected": expected,
    }
    for suffix, value in values.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise RuntimeError(f"nonfinite MLA fixture tensor: {name}.{suffix}")
        tensors[f"{name}.{suffix}"] = value.cpu().contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("NVIDIA CUDA is required; CPU fallback is forbidden")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tensors: dict[str, torch.Tensor] = {}
    for name, capacity, valid, _ in CASES:
        add_case(tensors, name, capacity, valid)
    args.output.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output / "mla-optimized-cases.safetensors"
    save_file(tensors, tensor_path, metadata={"kimi_k3": "milestone-18-schema-1"})
    manifest = {
        "schema_version": 1,
        "milestone": 18,
        "seed": SEED,
        "device": torch.cuda.get_device_name(0),
        "backend": "cuda",
        "cpu_inference_fallback": False,
        "checkpoint_downloaded": False,
        "cache_layout": {"compressed": "batch,token,latent=512", "extra_key": "batch,token,extra=64"},
        "expanded_kv_materialized": False,
        "cache_values_per_token": 576,
        "cases": [
            {"name": name, "capacity": capacity, "valid_tokens": valid, "benchmark": benchmark}
            for name, capacity, valid, benchmark in CASES
        ],
        "tensor_file": tensor_path.name,
        "tensor_file_sha256": sha256_file(tensor_path),
    }
    manifest_path = args.output / "mla-optimized-cases.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # KIMI_K3_TEMP_REMOVE_M20: verbose fixture inventory is bring-up debug output.
    if args.debug:
        print(f"[kimi-k3-debug] mla_optimized_cases={len(CASES)} tensors={len(tensors)}")
    print(json.dumps({"fixture": str(tensor_path), "cases": len(CASES), "sha256": manifest["tensor_file_sha256"]}))


if __name__ == "__main__":
    main()
