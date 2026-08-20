#!/usr/bin/env python3
"""Create Milestone-5 primitive fixtures from NumPy and pinned Moonshot code.

The fixture is deliberately tiny.  Real checkpoint coverage reads only two
rows and one MXFP4 block from a local shard; it never opens or downloads the
complete model.  Torch operator checks are NVIDIA-only, while NumPy remains
the independent host-side oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
from safetensors import safe_open
from safetensors.numpy import save_file

from export_reference import deterministic_setup, import_official
from reference_oracles import (
    causal_conv_tail,
    causal_depthwise_conv1d,
    decode_e8m0,
    dequantize_mxfp4,
    expand_block32_scale,
    l2_norm,
    mla_nope_join,
    mla_scale,
    mxfp4_linear,
    rms_norm,
    sigmoid,
    situ_glu,
    softmax,
    topk_descending,
    unpack_e2m1,
)


ROOT = Path("/dev/shm/kimi-k3")
DEFAULT_CHECKPOINT = ROOT / "moonshot/kimi-k3"
DEFAULT_OUTPUT = ROOT / "artifacts/fixtures/milestone-5"
SEED = 20260819
REAL_NORM = "language_model.model.layers.0.input_layernorm.weight"
REAL_PACKED = "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed"
REAL_SCALE = "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def record(value: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        "elements": int(value.size),
    }


def semantic_sha256(tensors: dict[str, np.ndarray]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes independent of file ordering."""
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def comparison(actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    delta = np.abs(actual - expected)
    close = np.isclose(actual, expected, atol=atol, rtol=rtol)
    return {
        "atol": atol,
        "rtol": rtol,
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean()) if delta.size else 0.0,
        "close_fraction": float(close.mean()) if close.size else 1.0,
        "passed": bool(close.all()),
    }


def load_real_slices(checkpoint: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load selected local slices through CUDA, never a complete weight tensor."""
    shard1 = checkpoint / "model-00001-of-000096.safetensors"
    shard2 = checkpoint / "model-00002-of-000096.safetensors"
    if not shard1.is_file() or not shard2.is_file():
        raise FileNotFoundError("Milestone 5 requires only local checkpoint shards 1 and 2")
    with safe_open(shard1, framework="pt", device="cuda:0") as tensors:
        norm = tensors.get_slice(REAL_NORM)[:8].float().cpu().numpy()
    with safe_open(shard2, framework="pt", device="cuda:0") as tensors:
        packed = tensors.get_slice(REAL_PACKED)[:2, :16].cpu().numpy()
        scale = tensors.get_slice(REAL_SCALE)[:2, :1].cpu().numpy()
    return norm, packed, scale


def make_tensors(checkpoint: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED)
    tensors: dict[str, np.ndarray] = {}

    rms_input = rng.normal(size=(2, 8)).astype(np.float32)
    rms_weight = np.linspace(0.5, 1.5, 8, dtype=np.float32)
    real_norm, real_packed, real_scale = load_real_slices(checkpoint)
    real_rms_input = np.linspace(-2.0, 2.0, 8, dtype=np.float32).reshape(1, 8)
    tensors.update(
        {
            "rms.input": rms_input,
            "rms.weight": rms_weight,
            "rms.expected": rms_norm(rms_input, rms_weight),
            "rms_real.input": real_rms_input,
            "rms_real.weight": real_norm.astype(np.float32),
            "rms_real.expected": rms_norm(real_rms_input, real_norm),
        }
    )

    l2_input = rng.normal(size=(2, 8)).astype(np.float32)
    gate = np.asarray(
        [[-80.0, -8.0, -2.0, -0.1, 0.0, 0.5, 4.0, 80.0],
         [3.0, -3.0, 10.0, -10.0, 1.5, -1.5, 25.0, -25.0]],
        dtype=np.float32,
    )
    up = np.linspace(-40.0, 40.0, 16, dtype=np.float32).reshape(2, 8)
    tensors.update(
        {
            "l2.input": l2_input,
            "l2.expected": l2_norm(l2_input),
            "situ.gate": gate,
            "situ.up": up,
            "situ.expected": situ_glu(gate, up),
            "sigmoid.input": gate,
            "sigmoid.expected": sigmoid(gate),
            "softmax.input": gate,
            "softmax.expected": softmax(gate),
        }
    )

    topk_input = np.asarray(
        [[-2.0, 8.0, 0.25, 3.0, -7.0, 1.5, 4.0, 0.0],
         [9.0, -1.0, 2.0, 8.5, 0.5, 7.0, -3.0, 1.0]],
        dtype=np.float32,
    )
    topk_values, topk_ids = topk_descending(topk_input, 3)
    tensors.update(
        {
            "topk.input": topk_input,
            "topk.expected_values": topk_values,
            "topk.expected_ids": topk_ids,
        }
    )

    conv_input = rng.normal(size=(1, 6, 3)).astype(np.float32)
    conv_kernel = rng.normal(size=(3, 1, 4)).astype(np.float32)
    short_input = conv_input[:, :2, :].copy()
    tensors.update(
        {
            "conv.input": conv_input,
            "conv.kernel": conv_kernel,
            "conv.expected": causal_depthwise_conv1d(conv_input, conv_kernel),
            "conv.expected_tail": causal_conv_tail(conv_input, 3),
            "conv.short_input": short_input,
            "conv.short_expected_tail": causal_conv_tail(short_input, 3),
        }
    )

    mla_content = rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
    mla_extra = rng.normal(size=(1, 2, 3, 2)).astype(np.float32)
    mla_scores = rng.normal(size=(1, 2, 3, 5)).astype(np.float32)
    tensors.update(
        {
            "mla.content": mla_content,
            "mla.extra": mla_extra,
            "mla.expected_join": mla_nope_join(mla_content, mla_extra),
            "mla.scores": mla_scores,
            "mla.expected_scaled": mla_scale(mla_scores),
        }
    )

    packed = np.asarray(
        [[0x21, 0xF8, 0x43, 0x65, 0x87, 0xA9, 0xCB, 0xED] * 2,
         [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2],
        dtype=np.uint8,
    )
    scale = np.asarray([[126], [129]], dtype=np.uint8)
    linear_input = rng.normal(size=(3, 32)).astype(np.float32)
    real_linear_input = rng.normal(size=(3, 32)).astype(np.float32)
    tensors.update(
        {
            "mxfp4.packed": packed,
            "mxfp4.scale_e8m0": scale,
            "mxfp4.expected_unpacked": unpack_e2m1(packed),
            "mxfp4.expected_scale": decode_e8m0(scale),
            "mxfp4.expected_expanded": expand_block32_scale(scale),
            "mxfp4.expected_weight": dequantize_mxfp4(packed, scale),
            "mxfp4.linear_input": linear_input,
            "mxfp4.expected_linear": mxfp4_linear(linear_input, packed, scale),
            "mxfp4_real.packed": real_packed.astype(np.uint8),
            "mxfp4_real.scale_e8m0": real_scale.astype(np.uint8),
            "mxfp4_real.expected_weight": dequantize_mxfp4(real_packed, real_scale),
            "mxfp4_real.linear_input": real_linear_input,
            "mxfp4_real.expected_linear": mxfp4_linear(
                real_linear_input, real_packed, real_scale
            ),
        }
    )
    return {name: np.ascontiguousarray(value) for name, value in tensors.items()}


def verify_official(modeling: Any, tensors: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compare independent fixture math with actual pinned Moonshot CUDA ops."""
    checks: dict[str, Any] = {}
    with torch.inference_mode():
        rms = modeling.KimiRMSNorm(8).cuda().eval()
        rms.weight.copy_(torch.from_numpy(tensors["rms.weight"]).cuda())
        actual = rms(torch.from_numpy(tensors["rms.input"]).cuda()).float().cpu().numpy()
        checks["official_rms_vs_numpy"] = comparison(
            actual, tensors["rms.expected"], 1e-5, 1e-5
        )

        situ = modeling.SituAndMul(beta=4.0, linear_beta=25.0).cuda().eval()
        joined = np.concatenate((tensors["situ.gate"], tensors["situ.up"]), axis=-1)
        actual = situ(torch.from_numpy(joined).cuda()).float().cpu().numpy()
        checks["official_situ_vs_numpy"] = comparison(
            actual, tensors["situ.expected"], 1e-5, 1e-5
        )

        conv_input = torch.from_numpy(tensors["conv.input"]).cuda().transpose(1, 2)
        conv_kernel = torch.from_numpy(tensors["conv.kernel"]).cuda()
        actual = torch_functional.conv1d(
            torch_functional.pad(conv_input, (3, 0)), conv_kernel, groups=3
        ).transpose(1, 2)
        checks["torch_conv_vs_numpy"] = comparison(
            actual.cpu().numpy(), tensors["conv.expected"], 1e-5, 1e-5
        )

        input_tensor = torch.from_numpy(tensors["topk.input"]).cuda()
        actual_values, actual_ids = torch.topk(input_tensor, 3, dim=-1, sorted=True)
        checks["torch_topk_values_vs_numpy"] = comparison(
            actual_values.cpu().numpy(), tensors["topk.expected_values"], 0.0, 0.0
        )
        checks["torch_topk_ids_exact"] = {
            "passed": bool(
                np.array_equal(actual_ids.cpu().numpy(), tensors["topk.expected_ids"])
            )
        }
    failed = [name for name, result in checks.items() if not result["passed"]]
    if failed:
        raise RuntimeError(f"pinned official/Torch primitive comparisons failed: {failed}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    deterministic_setup()
    _, modeling = import_official(args.checkpoint)
    tensors = make_tensors(args.checkpoint)
    checks = verify_official(modeling, tensors)
    args.output.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output / "primitive-reference.safetensors"
    save_file(tensors, tensor_path, metadata={"schema_version": "1", "milestone": "5"})
    manifest = {
        "schema_version": 1,
        "milestone": 5,
        "seed": SEED,
        "device": torch.cuda.get_device_name(0),
        "cpu_inference_fallback": False,
        "checkpoint_access": {
            "downloaded": False,
            "shards_opened": [
                "model-00001-of-000096.safetensors",
                "model-00002-of-000096.safetensors",
            ],
            "real_tensor_slices": [REAL_NORM, REAL_PACKED, REAL_SCALE],
        },
        "tensor_file": tensor_path.name,
        "tensor_file_sha256": sha256_file(tensor_path),
        "tensor_semantic_sha256": semantic_sha256(tensors),
        "tensors": {name: record(value) for name, value in sorted(tensors.items())},
        "official_vs_numpy": checks,
        "tolerance_manifest": "zml/docs/kimi_k3/tolerances.json",
    }
    manifest_path = args.output / "primitive-reference.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # KIMI_K3_TEMP_REMOVE_M20: verbose activation inventory exists only for
    # bring-up and must be removed with temporary diagnostics at cleanup.
    if args.debug:
        for name, value in sorted(tensors.items()):
            print(f"[kimi-k3-debug] {name} shape={value.shape} dtype={value.dtype}")
    print(json.dumps({"fixture": str(tensor_path), "checks": checks}, sort_keys=True))


if __name__ == "__main__":
    main()
