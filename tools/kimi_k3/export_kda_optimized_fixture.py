#!/usr/bin/env python3
"""Create deterministic KDA recurrence edge and production-shape fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


ROOT = Path("/dev/shm/kimi-k3")
OUTPUT = ROOT / "artifacts/fixtures/milestone-18"
SEED = 20260828
SMALL_LENGTHS = (1, 3, 4, 5, 31, 32, 33, 63, 64, 65, 257)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_sha256(tensors: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        array = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def values(shape: tuple[int, ...], phase: float, scale: float) -> np.ndarray:
    index = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return (np.sin(index * np.float32(0.017) + np.float32(phase)) * np.float32(scale)).astype(np.float32)


def case_inputs(sequence: int, heads: int, dim: int) -> tuple[np.ndarray, ...]:
    shape_k = (1, sequence, heads, dim)
    q = values(shape_k, 0.2, 0.11)
    k = values(shape_k, 0.7, 0.09)
    q /= np.sqrt(np.maximum(np.sum(q * q, axis=-1, keepdims=True), np.float32(1e-6)))
    k /= np.sqrt(np.maximum(np.sum(k * k, axis=-1, keepdims=True), np.float32(1e-6)))
    v = values((1, sequence, heads, dim), 1.1, 0.08)
    alpha_raw = values(shape_k, 0.4, 0.4)
    alpha = (np.float32(0.90) + np.float32(0.08) / (np.float32(1.0) + np.exp(-alpha_raw))).astype(np.float32)
    beta_raw = values((1, sequence, heads), 0.9, 0.7)
    beta = (np.float32(1.0) / (np.float32(1.0) + np.exp(-beta_raw))).astype(np.float32)
    state = values((1, heads, dim, dim), 1.7, 0.012)
    return q, k, v, alpha, beta, state


def recurrence(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    current = state.copy()
    output = np.empty_like(v)
    scale = np.float32(1.0 / np.sqrt(q.shape[-1]))
    for token in range(q.shape[1]):
        decayed = current * alpha[:, token, :, None, :]
        prediction = np.einsum("bhvk,bhk->bhv", decayed, k[:, token], optimize=False)
        error = (v[:, token] - prediction) * beta[:, token, :, None]
        current = decayed + error[..., None] * k[:, token, :, None, :]
        output[:, token] = np.einsum("bhvk,bhk->bhv", current, q[:, token], optimize=False) * scale
    return output.astype(np.float32), current.astype(np.float32)


def add_case(tensors: dict[str, np.ndarray], name: str, sequence: int, heads: int, dim: int) -> dict[str, int | str]:
    q, k, v, alpha, beta, state = case_inputs(sequence, heads, dim)
    output, final_state = recurrence(q, k, v, alpha, beta, state)
    for suffix, value in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("alpha", alpha),
        ("beta", beta),
        ("state", state),
        ("expected_output", output),
        ("expected_state", final_state),
    ):
        if not np.isfinite(value).all():
            raise RuntimeError(f"nonfinite fixture tensor: {name}.{suffix}")
        tensors[f"{name}.{suffix}"] = value
    return {"name": name, "sequence": sequence, "heads": heads, "key_dim": dim, "value_dim": dim}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    np.random.seed(SEED)
    tensors: dict[str, np.ndarray] = {}
    cases = [add_case(tensors, f"small_s{length}", length, 2, 16) for length in SMALL_LENGTHS]
    cases.append(add_case(tensors, "production_decode", 1, 96, 128))
    cases.append(add_case(tensors, "production_prefill64", 64, 96, 128))

    args.output.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output / "kda-optimized-cases.safetensors"
    save_file(tensors, tensor_path, metadata={"schema_version": "1", "milestone": "18"})
    manifest = {
        "schema_version": 1,
        "milestone": 18,
        "seed": SEED,
        "backend": "cuda",
        "cpu_inference_fallback": False,
        "checkpoint_downloaded": False,
        "equation": "channel-wise KDA state recurrence in FP32",
        "cases": cases,
        "tensor_file": tensor_path.name,
        "tensor_file_sha256": sha256_file(tensor_path),
        "tensor_semantic_sha256": semantic_sha256(tensors),
        "finite": True,
    }
    manifest_path = args.output / "kda-optimized-cases.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fixture": str(tensor_path), "cases": len(cases), "semantic_sha256": manifest["tensor_semantic_sha256"]}))


if __name__ == "__main__":
    main()
