#!/usr/bin/env python3
"""Export a four-step official/independent Kimi K3 KDA decode fixture.

The fixture uses tiny deterministic FP32 dimensions so every projection,
convolution, decay, recurrence, gate, and cache boundary remains inspectable.
The pinned Moonshot ``KimiDeltaAttention`` and FLA recurrent CUDA kernel are
the execution authority; an explicit Torch equation path supplies the
independent readable reference used by the ZML differential harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.numpy import save_file

from export_reference import deterministic_setup, import_official


ROOT = Path("/ephemeral/kimi-k3")
CHECKPOINT = ROOT / "moonshot/kimi-k3"
OUTPUT = ROOT / "artifacts/fixtures/milestone-7"
SEED = 20260821
STEPS = 4
HIDDEN = 10
HEADS = 2
HEAD_DIM = 3
CONV_SIZE = 4
EPS = 1e-5
LOWER_BOUND = -5.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_sha256(tensors: dict[str, np.ndarray]) -> str:
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


def record(value: np.ndarray) -> dict[str, Any]:
    value = np.ascontiguousarray(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "elements": int(value.size),
        "finite": int(np.isfinite(value).sum()) if np.issubdtype(value.dtype, np.floating) else None,
        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }


def compare(actual: torch.Tensor, expected: torch.Tensor, atol: float = 2e-4) -> dict[str, Any]:
    actual_f32 = actual.detach().float()
    expected_f32 = expected.detach().float()
    delta = (actual_f32 - expected_f32).abs()
    close = torch.isclose(actual_f32, expected_f32, atol=atol, rtol=2e-4)
    return {
        "passed": bool(close.all()),
        "atol": atol,
        "rtol": 2e-4,
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
        "close_fraction": float(close.float().mean()) if close.numel() else 1.0,
    }


def deterministic_fill(module: torch.nn.Module) -> None:
    """Fill parameters without relying on framework initializer versions."""
    with torch.no_grad():
        for index, (name, parameter) in enumerate(module.named_parameters()):
            values = torch.arange(parameter.numel(), device="cuda", dtype=torch.float32)
            values = (0.11 * torch.sin(values * (0.071 + index * 0.003) + index)).reshape(parameter.shape)
            if name == "A_log":
                values = torch.log(torch.linspace(1.25, 2.25, parameter.numel(), device="cuda")).reshape(parameter.shape)
            elif name == "dt_bias":
                values = torch.linspace(-0.7, 0.5, parameter.numel(), device="cuda").reshape(parameter.shape)
            elif name == "o_norm.weight":
                values = torch.linspace(0.8, 1.2, parameter.numel(), device="cuda").reshape(parameter.shape)
            parameter.copy_(values)


def conv_step(
    projected: torch.Tensor,
    cache: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    updated = torch.cat((cache[..., 1:], projected.unsqueeze(-1)), dim=-1)
    output = F.silu((updated * weight[:, 0, :].unsqueeze(0)).sum(dim=-1))
    return output, updated


def manual_step(
    layer: torch.nn.Module,
    hidden: torch.Tensor,
    caches: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    state: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    q_proj = F.linear(hidden, layer.q_proj.weight)
    k_proj = F.linear(hidden, layer.k_proj.weight)
    v_proj = F.linear(hidden, layer.v_proj.weight)
    q_conv, q_cache = conv_step(q_proj, caches[0], layer.q_conv1d.weight)
    k_conv, k_cache = conv_step(k_proj, caches[1], layer.k_conv1d.weight)
    v_conv, v_cache = conv_step(v_proj, caches[2], layer.v_conv1d.weight)

    q = q_conv.reshape(hidden.shape[0], HEADS, HEAD_DIM)
    k = k_conv.reshape(hidden.shape[0], HEADS, HEAD_DIM)
    v = v_conv.reshape(hidden.shape[0], HEADS, HEAD_DIM)
    q_norm = q.float() * torch.rsqrt(q.float().square().sum(dim=-1, keepdim=True) + 1e-6)
    k_norm = k.float() * torch.rsqrt(k.float().square().sum(dim=-1, keepdim=True) + 1e-6)

    decay_rank = F.linear(hidden, layer.f_a_proj.weight)
    raw_decay = F.linear(decay_rank, layer.f_b_proj.weight).reshape_as(q)
    decay_with_bias = raw_decay.float() + layer.dt_bias.float().reshape(HEADS, HEAD_DIM)
    log_alpha = LOWER_BOUND * torch.sigmoid(layer.A_log.float().exp()[None, :, None] * decay_with_bias)
    alpha = log_alpha.exp()
    raw_beta = F.linear(hidden, layer.b_proj.weight).float()
    beta = raw_beta.sigmoid()

    decayed_state = state.float() * alpha.unsqueeze(-2)
    prediction = torch.einsum("bhvk,bhk->bhv", decayed_state, k_norm)
    error = (v.float() - prediction) * beta.unsqueeze(-1)
    next_state = decayed_state + torch.einsum("bhv,bhk->bhvk", error, k_norm)
    recurrent_output = torch.einsum("bhvk,bhk->bhv", next_state, q_norm) / math.sqrt(HEAD_DIM)

    output_gate = F.linear(hidden, layer.g_proj.weight).reshape_as(recurrent_output)
    variance = recurrent_output.square().mean(dim=-1, keepdim=True)
    norm_gated = (
        recurrent_output * torch.rsqrt(variance + EPS)
        * layer.o_norm.weight.float()[None, None, :]
        * output_gate.sigmoid()
    )
    projection_output = F.linear(norm_gated.flatten(1), layer.o_proj.weight)
    values = {
        "hidden": hidden,
        "q_proj": q_proj,
        "k_proj": k_proj,
        "v_proj": v_proj,
        "q_conv": q_conv,
        "k_conv": k_conv,
        "v_conv": v_conv,
        "q": q,
        "k": k,
        "v": v,
        "q_norm": q_norm,
        "k_norm": k_norm,
        "raw_decay": raw_decay,
        "log_alpha": log_alpha,
        "alpha": alpha,
        "raw_beta": raw_beta,
        "beta": beta,
        "prediction": prediction,
        "error": error,
        "recurrent_state": next_state,
        "recurrent_output": recurrent_output,
        "output_gate": output_gate,
        "norm_gated": norm_gated,
        "projection_output": projection_output,
        "q_cache": q_cache,
        "k_cache": k_cache,
        "v_cache": v_cache,
    }
    return values, (q_cache, k_cache, v_cache), next_state


def install_hooks(layer: torch.nn.Module, captures: dict[str, list[Any]]) -> list[Any]:
    handles = []
    names = {
        "q_proj": layer.q_proj,
        "k_proj": layer.k_proj,
        "v_proj": layer.v_proj,
        "q_conv": layer.q_conv1d,
        "k_conv": layer.k_conv1d,
        "v_conv": layer.v_conv1d,
        "norm_gated": layer.o_norm,
        "projection_output": layer.o_proj,
    }
    for name, module in names.items():
        def hook(_module: Any, _inputs: Any, output: Any, *, key: str = name) -> None:
            value = output[0] if key.endswith("_conv") else output
            captures.setdefault(key, []).append(value.detach().clone())

        handles.append(module.register_forward_hook(hook))
    return handles


def numpy_replay(tensors: dict[str, np.ndarray]) -> dict[str, Any]:
    """Recompute every exported boundary with independent NumPy equations."""
    weights = {name.removeprefix("weights."): value for name, value in tensors.items() if name.startswith("weights.")}
    q_cache = tensors["inputs.initial_q_cache"].copy()
    k_cache = tensors["inputs.initial_k_cache"].copy()
    v_cache = tensors["inputs.initial_v_cache"].copy()
    state = tensors["inputs.initial_recurrent_state"].copy()
    checks: dict[str, Any] = {}

    def linear(value: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return np.asarray(value @ weight.T, dtype=np.float32)

    def sigmoid(value: np.ndarray) -> np.ndarray:
        return np.asarray(1.0 / (1.0 + np.exp(-value)), dtype=np.float32)

    def conv(value: np.ndarray, cache: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        updated = np.concatenate((cache[..., 1:], value[..., None]), axis=-1)
        raw = np.sum(updated * weight[None, :, :], axis=-1, dtype=np.float32)
        return np.asarray(raw * sigmoid(raw), dtype=np.float32), updated

    for step in range(STEPS):
        hidden = tensors[f"step.{step}.input.hidden"]
        q_proj = linear(hidden, weights["q_weight"])
        k_proj = linear(hidden, weights["k_weight"])
        v_proj = linear(hidden, weights["v_weight"])
        q_conv, q_cache = conv(q_proj, q_cache, weights["q_conv_weight"])
        k_conv, k_cache = conv(k_proj, k_cache, weights["k_conv_weight"])
        v_conv, v_cache = conv(v_proj, v_cache, weights["v_conv_weight"])
        q = q_conv.reshape(1, HEADS, HEAD_DIM)
        k = k_conv.reshape(1, HEADS, HEAD_DIM)
        v = v_conv.reshape(1, HEADS, HEAD_DIM)
        q_norm = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True, dtype=np.float32) + np.float32(1e-6))
        k_norm = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True, dtype=np.float32) + np.float32(1e-6))
        decay_rank = linear(hidden, weights["decay_a_weight"])
        raw_decay = linear(decay_rank, weights["decay_b_weight"]).reshape(1, HEADS, HEAD_DIM)
        log_alpha = LOWER_BOUND * sigmoid(
            np.exp(weights["a_log"], dtype=np.float32)[None, :, None]
            * (raw_decay + weights["dt_bias"])
        )
        alpha = np.exp(log_alpha, dtype=np.float32)
        raw_beta = linear(hidden, weights["beta_weight"])
        beta = sigmoid(raw_beta)
        decayed_state = state * alpha[:, :, None, :]
        prediction = np.einsum("bhvk,bhk->bhv", decayed_state, k_norm, dtype=np.float32)
        error = (v - prediction) * beta[..., None]
        state = decayed_state + np.einsum("bhv,bhk->bhvk", error, k_norm, dtype=np.float32)
        recurrent_output = np.einsum("bhvk,bhk->bhv", state, q_norm, dtype=np.float32) / np.float32(math.sqrt(HEAD_DIM))
        output_gate = linear(hidden, weights["gate_weight"]).reshape(1, HEADS, HEAD_DIM)
        variance = np.mean(recurrent_output * recurrent_output, axis=-1, keepdims=True, dtype=np.float32)
        norm_gated = (
            recurrent_output / np.sqrt(variance + np.float32(EPS))
            * weights["norm_weight"][None, None, :]
            * sigmoid(output_gate)
        )
        projection_output = linear(norm_gated.reshape(1, HEADS * HEAD_DIM), weights["output_weight"])
        actual = {
            "q_proj": q_proj,
            "k_proj": k_proj,
            "v_proj": v_proj,
            "q_conv": q_conv,
            "k_conv": k_conv,
            "v_conv": v_conv,
            "q": q,
            "k": k,
            "v": v,
            "q_norm": q_norm,
            "k_norm": k_norm,
            "raw_decay": raw_decay,
            "log_alpha": log_alpha,
            "alpha": alpha,
            "raw_beta": raw_beta,
            "beta": beta,
            "prediction": prediction,
            "error": error,
            "recurrent_state": state,
            "recurrent_output": recurrent_output,
            "output_gate": output_gate,
            "norm_gated": norm_gated,
            "projection_output": projection_output,
            "q_cache": q_cache,
            "k_cache": k_cache,
            "v_cache": v_cache,
        }
        for name, value in actual.items():
            expected = tensors[f"step.{step}.expected.{name}"]
            delta = np.abs(np.asarray(value, dtype=np.float32) - expected)
            close = np.isclose(value, expected, atol=2e-5, rtol=2e-5)
            result = {
                "passed": bool(close.all()),
                "atol": 2e-5,
                "rtol": 2e-5,
                "max_abs": float(delta.max(initial=0.0)),
                "mean_abs": float(delta.mean()) if delta.size else 0.0,
                "close_fraction": float(close.mean()) if close.size else 1.0,
            }
            checks[f"step.{step}.{name}"] = result
            if not result["passed"]:
                raise RuntimeError(f"NumPy KDA replay mismatch: step={step} boundary={name} {result}")
    return checks


def build_fixture(configuration: Any, modeling: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    config = configuration.KimiLinearConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        intermediate_size=16,
        rms_norm_eps=EPS,
        linear_attn_config={
            "short_conv_kernel_size": CONV_SIZE,
            "head_dim": HEAD_DIM,
            "num_heads": HEADS,
            "kda_layers": [1],
            "full_attn_layers": [],
            "use_full_rank_gate": True,
            "gate_lower_bound": LOWER_BOUND,
        },
    )
    layer = modeling.KimiDeltaAttention(config, 0).cuda().float().eval()
    deterministic_fill(layer)
    official_cache = modeling.KimiDynamicCache(config)
    captures: dict[str, list[Any]] = {}
    handles = install_hooks(layer, captures)

    positions = torch.arange(STEPS * HIDDEN, device="cuda", dtype=torch.float32).reshape(1, STEPS, HIDDEN)
    hidden_sequence = torch.sin(positions * 0.19) + 0.35 * torch.cos(positions * 0.07 + 0.4)
    manual_caches = tuple(
        torch.zeros(1, HEADS * HEAD_DIM, CONV_SIZE, device="cuda", dtype=torch.float32)
        for _ in range(3)
    )
    manual_state = torch.zeros(1, HEADS, HEAD_DIM, HEAD_DIM, device="cuda", dtype=torch.float32)
    step_values: list[dict[str, torch.Tensor]] = []
    official_outputs: list[torch.Tensor] = []
    checks: dict[str, Any] = {}

    with torch.inference_mode():
        for step in range(STEPS):
            hidden = hidden_sequence[:, step, :]
            manual, manual_caches, manual_state = manual_step(layer, hidden, manual_caches, manual_state)
            official = layer(hidden[:, None, :], cache_params=official_cache)[:, 0, :]
            official_outputs.append(official)
            step_values.append(manual)

            boundary_actual = {
                "q_proj": captures["q_proj"][step][:, 0, :],
                "k_proj": captures["k_proj"][step][:, 0, :],
                "v_proj": captures["v_proj"][step][:, 0, :],
                "q_conv": captures["q_conv"][step][:, 0, :],
                "k_conv": captures["k_conv"][step][:, 0, :],
                "v_conv": captures["v_conv"][step][:, 0, :],
                "norm_gated": captures["norm_gated"][step][:, 0, :, :],
                "projection_output": captures["projection_output"][step][:, 0, :],
                "q_cache": official_cache.conv_states[0][0],
                "k_cache": official_cache.conv_states[0][1],
                "v_cache": official_cache.conv_states[0][2],
                "recurrent_state": official_cache.recurrent_states[0],
                "output": official,
            }
            for name, actual in boundary_actual.items():
                expected_name = "projection_output" if name == "output" else name
                # The fused recurrence writes a rounded output which the gated
                # RMS normalization can amplify near zero. Cache/state checks
                # remain tighter than the two post-recurrence output stages.
                tolerance = 5e-3 if name in {"norm_gated", "projection_output", "output"} else 5e-4
                result = compare(actual, manual[expected_name], atol=tolerance)
                checks[f"step.{step}.{name}"] = result
                if not result["passed"]:
                    raise RuntimeError(f"official KDA differs from readable reference: step={step} boundary={name} {result}")

    for handle in handles:
        handle.remove()

    tensors: dict[str, np.ndarray] = {
        "inputs.hidden_sequence": hidden_sequence.cpu().numpy(),
        "inputs.initial_q_cache": np.zeros((1, HEADS * HEAD_DIM, CONV_SIZE), dtype=np.float32),
        "inputs.initial_k_cache": np.zeros((1, HEADS * HEAD_DIM, CONV_SIZE), dtype=np.float32),
        "inputs.initial_v_cache": np.zeros((1, HEADS * HEAD_DIM, CONV_SIZE), dtype=np.float32),
        "inputs.initial_recurrent_state": np.zeros((1, HEADS, HEAD_DIM, HEAD_DIM), dtype=np.float32),
    }
    weight_names = {
        "q_weight": layer.q_proj.weight,
        "k_weight": layer.k_proj.weight,
        "v_weight": layer.v_proj.weight,
        "q_conv_weight": layer.q_conv1d.weight[:, 0, :],
        "k_conv_weight": layer.k_conv1d.weight[:, 0, :],
        "v_conv_weight": layer.v_conv1d.weight[:, 0, :],
        "decay_a_weight": layer.f_a_proj.weight,
        "decay_b_weight": layer.f_b_proj.weight,
        "a_log": layer.A_log,
        "dt_bias": layer.dt_bias.reshape(HEADS, HEAD_DIM),
        "beta_weight": layer.b_proj.weight,
        "gate_weight": layer.g_proj.weight,
        "norm_weight": layer.o_norm.weight,
        "output_weight": layer.o_proj.weight,
    }
    for name, value in weight_names.items():
        tensors[f"weights.{name}"] = value.detach().cpu().numpy()
    for step, values in enumerate(step_values):
        tensors[f"step.{step}.input.hidden"] = values["hidden"].detach().cpu().numpy()
        for name, value in values.items():
            if name != "hidden":
                tensors[f"step.{step}.expected.{name}"] = value.detach().cpu().numpy()

    numpy_checks = numpy_replay(tensors)

    return tensors, {
        "official_symbol": "KimiDeltaAttention.forward",
        "official_kernel": "fla.ops.kda.fused_recurrent_kda",
        "steps": STEPS,
        "hidden_size": HIDDEN,
        "heads": HEADS,
        "head_dim": HEAD_DIM,
        "conv_size": CONV_SIZE,
        "rms_norm_eps": EPS,
        "gate_lower_bound": LOWER_BOUND,
        "state_layout": "batch,head,value,key",
        "conv_cache_layout": "batch,channel,kernel",
        "official_vs_readable": checks,
        "numpy_vs_readable": numpy_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    deterministic_setup()
    configuration, modeling = import_official(args.checkpoint)
    tensors, details = build_fixture(configuration, modeling)
    args.output.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output / "kda-decode-reference.safetensors"
    save_file(tensors, tensor_path, metadata={"schema_version": "1", "milestone": "7"})
    manifest = {
        "schema_version": 1,
        "milestone": 7,
        "seed": SEED,
        "device": torch.cuda.get_device_name(0),
        "cpu_inference_fallback": False,
        "checkpoint_downloaded": False,
        "tensor_file": tensor_path.name,
        "tensor_file_sha256": sha256_file(tensor_path),
        "tensor_semantic_sha256": semantic_sha256(tensors),
        "tensors": {name: record(value) for name, value in sorted(tensors.items())},
        **details,
    }
    manifest_path = args.output / "kda-decode-reference.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # KIMI_K3_TEMP_REMOVE_M20: named per-step activation statistics are retained
    # for KDA bring-up and removed during the cleanup milestone.
    if args.debug:
        for step in range(STEPS):
            for boundary in ("alpha", "prediction", "error", "recurrent_state", "projection_output"):
                value = tensors[f"step.{step}.expected.{boundary}"]
                print(
                    f"[kimi-k3-debug] step={step} boundary={boundary} shape={value.shape} "
                    f"min={value.min():.7g} max={value.max():.7g} rms={np.sqrt(np.mean(value * value)):.7g}"
                )
    print(
        json.dumps(
            {
                "fixture": str(tensor_path),
                "steps": STEPS,
                "official_checks": len(details["official_vs_readable"]),
                "numpy_checks": len(details["numpy_vs_readable"]),
            }
        )
    )


if __name__ == "__main__":
    main()
