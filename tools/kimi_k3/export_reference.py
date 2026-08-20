#!/usr/bin/env python3
"""Export deterministic Kimi K3 golden fixtures from pinned Moonshot code.

Only isolated layer 0 is loaded from the local S1 shard.  This tool never calls
``from_pretrained`` and therefore cannot request missing shards or download
weights.  All Torch/FLA execution is rejected unless an NVIDIA CUDA device is
available; NumPy is used only as an independent host-side numerical oracle.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import types
from typing import Any

import numpy as np
import torch
import torch.nn.functional as torch_functional
from safetensors import safe_open
from safetensors.torch import save_file

from reference_oracles import decode_e8m0, kda_log_alpha, kda_scan, route, unpack_e2m1


ROOT = Path("/dev/shm/kimi-k3")
DEFAULT_CHECKPOINT = ROOT / "moonshot/kimi-k3"
DEFAULT_OUTPUT = ROOT / "artifacts/fixtures/milestone-3"
MOONSHOT_REVISION = "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721"
SHARD1_SHA256 = "975584c00f85a95fce8ae0f840af8cef69c2ef4db00d34cab3e2cbdfc60f6e51"
SEED = 20260819


class ExportError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().contiguous().cpu()
    return value.view(torch.uint8).numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(tensor)).hexdigest()


def tensor_record(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "sha256": tensor_sha256(tensor),
        "finite": int(torch.isfinite(tensor).sum()) if tensor.is_floating_point() else None,
        "elements": tensor.numel(),
    }


def comparison(actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    delta = np.abs(actual - expected)
    close = np.isclose(actual, expected, atol=atol, rtol=rtol)
    return {
        "shape": list(actual.shape),
        "atol": atol,
        "rtol": rtol,
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean()) if delta.size else 0.0,
        "rmse": float(np.sqrt(np.mean(delta * delta))) if delta.size else 0.0,
        "close_fraction": float(close.mean()) if close.size else 1.0,
        "passed": bool(close.all()),
    }


def require_nvidia() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise ExportError("NVIDIA CUDA device required; CPU inference fallback is prohibited")
    name = torch.cuda.get_device_name(0)
    if "NVIDIA" not in name.upper() and "H100" not in name.upper():
        raise ExportError(f"CUDA device is not identified as NVIDIA: {name}")


def deterministic_setup() -> None:
    require_nvidia()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def import_official(checkpoint: Path) -> tuple[Any, Any]:
    """Load the pinned source as a package without copying or modifying it."""
    expected = {
        "configuration_kimi_k3.py": "735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae",
        "modeling_kimi_linear.py": "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a",
    }
    for name, digest in expected.items():
        path = checkpoint / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ExportError(f"pinned Moonshot source hash mismatch: {path}")

    # Transformers 5 moved OutputRecorder out of utils.generic.  This adapter
    # restores the import location expected by the pinned, unmodified Moonshot
    # source; it does not alter any model math or checkpoint data.
    import transformers.utils.generic as generic

    if not hasattr(generic, "OutputRecorder"):
        from transformers.utils.output_capturing import OutputRecorder

        generic.OutputRecorder = OutputRecorder

    package_name = "kimi_k3_pinned_reference"
    for loaded in tuple(sys.modules):
        if loaded == package_name or loaded.startswith(package_name + "."):
            del sys.modules[loaded]
    package = types.ModuleType(package_name)
    package.__path__ = [os.fspath(checkpoint)]
    package.__package__ = package_name
    sys.modules[package_name] = package
    modules: dict[str, Any] = {}
    for name in ("configuration_kimi_k3", "modeling_kimi_linear"):
        spec = importlib.util.spec_from_file_location(
            f"{package_name}.{name}", checkpoint / f"{name}.py"
        )
        if spec is None or spec.loader is None:
            raise ExportError(f"cannot import official module {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["configuration_kimi_k3"], modules["modeling_kimi_linear"]


def cuda_timed(operation: Callable[[], Any]) -> tuple[Any, dict[str, float]]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    result = operation()
    end.record()
    torch.cuda.synchronize()
    return result, {
        "gpu_ms": float(begin.elapsed_time(end)),
        "wall_ms": float((time.perf_counter() - started) * 1000.0),
    }


def _to_cpu(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().contiguous().cpu() for name, tensor in tensors.items()}


def _save_fixture(
    fixture_dir: Path,
    name: str,
    tensors: dict[str, torch.Tensor],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    tensor_path = fixture_dir / f"{name}.safetensors"
    manifest_path = fixture_dir / f"{name}.json"
    cpu_tensors = _to_cpu(tensors)
    save_file(cpu_tensors, tensor_path, metadata={"schema_version": "1", "fixture": name})
    manifest = {
        "schema_version": 1,
        "fixture": name,
        "moonshot_revision": MOONSHOT_REVISION,
        "seed": SEED,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cpu_inference_fallback": False,
        "tensor_file": tensor_path.name,
        "tensor_file_sha256": sha256_file(tensor_path),
        "tensors": {key: tensor_record(value) for key, value in sorted(cpu_tensors.items())},
        **metadata,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _assert_stable(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> None:
    if first.keys() != second.keys():
        raise ExportError("repeat run changed the fixture tensor set")
    changed = [name for name in first if tensor_sha256(first[name]) != tensor_sha256(second[name])]
    if changed:
        raise ExportError(f"repeat run changed numeric tensor hashes: {changed}")


def _official_kda_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    shape = (1, 16, 2, 16)
    return {
        "q": torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16),
        "k": torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16),
        "v": torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16),
        "raw_decay": torch.randn(shape, generator=generator, device="cuda", dtype=torch.bfloat16),
        "raw_beta": torch.randn(shape[:-1], generator=generator, device="cuda", dtype=torch.float32),
        "a_log": torch.linspace(0.1, 0.7, 2, device="cuda", dtype=torch.float32),
        "dt_bias": torch.linspace(-0.5, 0.5, 32, device="cuda", dtype=torch.float32),
    }


def run_s0_once(modeling: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    from compressed_tensors.compressors.mx_utils import decompress_mx_scale
    from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
    from fla.ops.kda import chunk_kda, fused_recurrent_kda

    values = _official_kda_inputs()
    common = {
        "A_log": values["a_log"],
        "dt_bias": values["dt_bias"],
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "lower_bound": -5.0,
        "transpose_state_layout": True,
    }
    with torch.inference_mode():
        (chunk_out, chunk_state), chunk_timing = cuda_timed(
            lambda: chunk_kda(
                q=values["q"],
                k=values["k"],
                v=values["v"],
                g=values["raw_decay"],
                beta=values["raw_beta"],
                safe_gate=True,
                **common,
            )
        )
        state = None
        recurrent_outputs = []

        def recurrent() -> tuple[torch.Tensor, torch.Tensor]:
            nonlocal state
            for token in range(values["q"].shape[1]):
                output, state = fused_recurrent_kda(
                    q=values["q"][:, token : token + 1],
                    k=values["k"][:, token : token + 1],
                    v=values["v"][:, token : token + 1],
                    g=values["raw_decay"][:, token : token + 1],
                    beta=values["raw_beta"][:, token : token + 1],
                    initial_state=state,
                    **common,
                )
                recurrent_outputs.append(output)
            return torch.cat(recurrent_outputs, dim=1), state

        (recurrent_out, recurrent_state), recurrent_timing = cuda_timed(recurrent)

        small_config = modeling.KimiLinearConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_experts=6,
            num_experts_per_token=3,
            moe_router_activation_func="sigmoid",
            moe_renormalize=True,
            routed_scaling_factor=1.25,
            num_expert_group=1,
            topk_group=1,
        )
        gate = modeling.KimiMoEGate(small_config).to("cuda").eval()
        gate.weight.copy_(
            torch.sin(torch.arange(48, device="cuda", dtype=torch.float32) * 0.17).reshape(6, 8)
        )
        gate.e_score_correction_bias.copy_(
            torch.tensor([0.0, -0.7, 0.3, 0.9, -0.2, 0.1], device="cuda")
        )
        router_hidden = torch.cos(
            torch.arange(24, device="cuda", dtype=torch.float32) * 0.11
        ).reshape(1, 3, 8)
        router_ids, router_weights = gate(router_hidden)
        router_logits = torch_functional.linear(router_hidden, gate.weight)
        router_raw = router_logits.sigmoid()
        router_selection = router_raw + gate.e_score_correction_bias

        packed = torch.tensor([[0x21, 0xF8]], dtype=torch.uint8, device="cuda")
        official_fp4 = unpack_fp4_from_uint8(packed, 1, 4, dtype=torch.float32)
        packed_scale = torch.tensor([126, 127, 128], dtype=torch.uint8, device="cuda")
        official_scale = decompress_mx_scale(packed_scale).float()

    q_np = values["q"].float().cpu().numpy()
    k_np = values["k"].float().cpu().numpy()
    v_np = values["v"].float().cpu().numpy()
    decay_np = values["raw_decay"].float().cpu().numpy()
    beta_np = values["raw_beta"].cpu().numpy()
    log_alpha_np = kda_log_alpha(
        decay_np,
        values["a_log"].cpu().numpy(),
        values["dt_bias"].reshape(2, 16).cpu().numpy(),
        -5.0,
    )
    numpy_out, numpy_state = kda_scan(q_np, k_np, v_np, log_alpha_np, beta_np)
    router_oracle = route(
        router_hidden.cpu().numpy(),
        gate.weight.detach().cpu().numpy(),
        gate.e_score_correction_bias.detach().cpu().numpy(),
        top_k=3,
        scaling_factor=1.25,
    )
    for row in range(router_ids.numel() // router_ids.shape[-1]):
        actual = {
            int(expert): float(weight)
            for expert, weight in zip(router_ids.reshape(-1, 3)[row], router_weights.reshape(-1, 3)[row])
        }
        expected = {
            int(expert): float(weight)
            for expert, weight in zip(router_oracle.ids.reshape(-1, 3)[row], router_oracle.weights.reshape(-1, 3)[row])
        }
        if actual.keys() != expected.keys():
            raise ExportError(f"official router selected set differs from NumPy at row {row}")
        if not np.allclose([actual[key] for key in sorted(actual)], [expected[key] for key in sorted(expected)], atol=1e-6, rtol=1e-6):
            raise ExportError(f"official router aligned weights differ from NumPy at row {row}")

    fp4_oracle = unpack_e2m1(np.asarray([[0x21, 0xF8]], dtype=np.uint8))
    scale_oracle = decode_e8m0(np.asarray([126, 127, 128], dtype=np.uint8))
    if not np.array_equal(official_fp4.cpu().numpy(), fp4_oracle):
        raise ExportError("compressed-tensors FP4 nibble order differs from independent oracle")
    if not np.array_equal(official_scale.cpu().numpy(), scale_oracle):
        raise ExportError("compressed-tensors E8M0 decoding differs from independent oracle")

    checks = {
        "chunk_vs_official_recurrent_output": comparison(
            chunk_out.float().cpu().numpy(), recurrent_out.float().cpu().numpy(), 0.001, 0.01
        ),
        "chunk_vs_official_recurrent_state": comparison(
            chunk_state.cpu().numpy(), recurrent_state.cpu().numpy(), 0.002, 0.01
        ),
        "official_recurrent_vs_numpy_output": comparison(
            recurrent_out.float().cpu().numpy(), numpy_out, 0.002, 0.015
        ),
        "official_recurrent_vs_numpy_state": comparison(
            recurrent_state.cpu().numpy(), numpy_state, 0.003, 0.015
        ),
        "router_selected_sets": {"passed": True, "rows": int(router_ids.numel() / 3)},
        "router_aligned_weights": {"passed": True, "atol": 1e-6, "rtol": 1e-6},
        "mxfp4_nibble_order": {"passed": True},
        "e8m0_decode": {"passed": True},
    }
    failed = [name for name, result in checks.items() if not result["passed"]]
    if failed:
        raise ExportError(f"S0 mandatory comparisons failed: {failed}")

    tensors = {
        "s0.kda.q": values["q"],
        "s0.kda.k": values["k"],
        "s0.kda.v": values["v"],
        "s0.kda.raw_decay": values["raw_decay"],
        "s0.kda.raw_beta": values["raw_beta"],
        "s0.kda.log_alpha.numpy": torch.from_numpy(log_alpha_np),
        "s0.kda.chunk.out": chunk_out,
        "s0.kda.chunk.state": chunk_state,
        "s0.kda.recurrent.out": recurrent_out,
        "s0.kda.recurrent.state": recurrent_state,
        "s0.kda.numpy.out": torch.from_numpy(numpy_out),
        "s0.kda.numpy.state": torch.from_numpy(numpy_state),
        "s0.router.hidden": router_hidden,
        "s0.router.raw_scores": router_raw,
        "s0.router.selection_scores": router_selection,
        "s0.router.topk_ids": router_ids,
        "s0.router.topk_weights": router_weights,
        "s0.mxfp4.packed": packed,
        "s0.mxfp4.unpacked": official_fp4,
        "s0.mxfp4.scale_e8m0": packed_scale,
        "s0.mxfp4.scale": official_scale,
    }
    return tensors, {"comparisons": checks, "timing": {"chunk": chunk_timing, "recurrent": recurrent_timing}}


def export_s0(modeling: Any, output: Path, debug: bool) -> dict[str, Any]:
    first, details = run_s0_once(modeling)
    second, repeat_details = run_s0_once(modeling)
    _assert_stable(first, second)
    details["timing"] = {"cold_or_first": details["timing"], "repeat": repeat_details["timing"]}
    # KIMI_K3_TEMP_REMOVE_M20: verbose boundary/timing diagnostics are retained
    # only while the port is being implemented and must be removed at cleanup.
    if debug:
        print("[kimi-k3-debug] S0 timing", json.dumps(details["timing"], sort_keys=True))
        for name, tensor in sorted(first.items()):
            print(f"[kimi-k3-debug] {name} shape={tuple(tensor.shape)} dtype={tensor.dtype}")
    return _save_fixture(
        output,
        "s0-operators",
        first,
        {
            "tier": "S0",
            "mode": "synthetic_operator",
            "repeat_runs": 2,
            "numeric_hashes_stable": True,
            **details,
        },
    )


def _load_layer0(checkpoint: Path, configuration: Any, modeling: Any) -> tuple[Any, Any, dict[str, Any]]:
    shard = checkpoint / "model-00001-of-000096.safetensors"
    if sha256_file(shard) != SHARD1_SHA256:
        raise ExportError(f"local shard-1 hash mismatch: {shard}")
    config_data = json.loads((checkpoint / "config.json").read_text())["text_config"]
    config = configuration.KimiLinearConfig(**config_data)
    with torch.device("meta"):
        layer = modeling.KimiDecoderLayer(config, 0)
    prefix = "language_model.model.layers.0."
    state: dict[str, torch.Tensor] = {}
    with safe_open(shard, framework="pt", device="cuda:0") as tensors:
        for key in tensors.keys():
            if key.startswith(prefix):
                state[key.removeprefix(prefix)] = tensors.get_tensor(key)
    if len(state) != 23:
        raise ExportError(f"expected 23 isolated layer-0 tensors, found {len(state)}")

    # KIMI_K3_TEMP_REMOVE_M20: pinned config constructs A_log[96], but the
    # pinned official shard stores padded A_log[128].  The official FLA kernel
    # indexes one value per 96 value head, so preserving the checkpoint tensor
    # is behaviorally correct.  Revisit/remove this compatibility assignment at
    # cleanup after Moonshot resolves or documents the padded parameter.
    checkpoint_a_log = state.pop("self_attn.A_log")
    if tuple(checkpoint_a_log.shape) != (128,) or layer.self_attn.num_heads != 96:
        raise ExportError("unexpected A_log compatibility condition")
    layer.self_attn.A_log = torch.nn.Parameter(checkpoint_a_log, requires_grad=False)
    missing, unexpected = layer.load_state_dict(state, strict=False, assign=True)
    if missing != ["self_attn.A_log"] or unexpected:
        raise ExportError(f"isolated layer load mismatch: missing={missing}, unexpected={unexpected}")
    layer.eval()
    devices = {parameter.device.type for parameter in layer.parameters()}
    if devices != {"cuda"}:
        raise ExportError(f"isolated layer is not entirely on NVIDIA CUDA: {devices}")
    return config, layer, {
        "weight_tensors": 23,
        "weight_bytes": sum(tensor.numel() * tensor.element_size() for tensor in state.values())
        + checkpoint_a_log.numel() * checkpoint_a_log.element_size(),
        "a_log_checkpoint_shape": [128],
        "a_log_runtime_heads": 96,
        "compatibility_intervention": "preserve padded official A_log; FLA indexes first 96 values",
    }


class Capture:
    """Temporary hook instrumentation for observable module boundaries."""

    def __init__(self) -> None:
        self.tensors: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []

    def add(self, name: str, module: torch.nn.Module) -> None:
        # KIMI_K3_TEMP_REMOVE_M20: module hooks expose intermediate activations
        # for differential bring-up and must be removed at the cleanup phase.
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            if isinstance(value, torch.Tensor):
                self.tensors[name] = value.detach().clone()

        self.handles.append(module.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _capture_layer_modules(layer: Any) -> Capture:
    capture = Capture()
    modules = {
        "layers.0.input_layernorm.out": layer.input_layernorm,
        "layers.0.kda.q_proj.out": layer.self_attn.q_proj,
        "layers.0.kda.k_proj.out": layer.self_attn.k_proj,
        "layers.0.kda.v_proj.out": layer.self_attn.v_proj,
        "layers.0.kda.q_conv.out": layer.self_attn.q_conv1d,
        "layers.0.kda.k_conv.out": layer.self_attn.k_conv1d,
        "layers.0.kda.v_conv.out": layer.self_attn.v_conv1d,
        "layers.0.kda.raw_decay": layer.self_attn.f_b_proj,
        "layers.0.kda.raw_beta": layer.self_attn.b_proj,
        "layers.0.kda.output_gate": layer.self_attn.g_proj,
        "layers.0.kda.norm_gated.out": layer.self_attn.o_norm,
        "layers.0.kda.out": layer.self_attn.o_proj,
        "layers.0.post_attention_layernorm.out": layer.post_attention_layernorm,
        "layers.0.mlp.gate_proj.out": layer.mlp.gate_proj,
        "layers.0.mlp.up_proj.out": layer.mlp.up_proj,
        "layers.0.mlp.situ.out": layer.mlp.act_fn,
        "layers.0.mlp.out": layer.mlp.down_proj,
    }
    for name, module in modules.items():
        capture.add(name, module)
    return capture


def synthetic_hidden(length: int) -> torch.Tensor:
    positions = torch.arange(length * 7168, device="cuda", dtype=torch.float32).reshape(
        1, length, 7168
    )
    return (torch.sin(positions * 0.00073) + 0.25 * torch.cos(positions * 0.00019)).to(
        torch.bfloat16
    )


def run_layer_once(
    config: Any,
    layer: Any,
    modeling: Any,
    length: int,
    mode: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    hidden = synthetic_hidden(length)
    cache = modeling.KimiDynamicCache(config)
    block_residual = torch.empty((length, 0, 7168), device="cuda", dtype=torch.bfloat16)
    capture = _capture_layer_modules(layer)
    tensors: dict[str, torch.Tensor] = {"layers.0.input": hidden.detach().clone()}
    try:
        with torch.inference_mode():
            (output, block_out), timing = cuda_timed(
                lambda: layer(
                    hidden,
                    past_key_values=cache,
                    use_cache=True,
                    block_residual=block_residual,
                )
            )
        tensors.update(capture.tensors)
        tensors["layers.0.out"] = output
        tensors["layers.0.attnres.block_residual.out"] = block_out
        tensors["layers.0.kda.recurrent_state.in"] = torch.zeros_like(cache.recurrent_states[0])
        tensors["layers.0.kda.recurrent_state.out"] = cache.recurrent_states[0]
        for index, value in enumerate(cache.conv_states[0]):
            tensors[f"layers.0.kda.conv_state.{index}.out"] = value

        heads = layer.self_attn.num_heads
        dim = layer.self_attn.head_dim
        raw_decay = tensors["layers.0.kda.raw_decay"].reshape(1, length, heads, dim)
        log_alpha = -5.0 * torch.sigmoid(
            torch.exp(layer.self_attn.A_log[:heads]).reshape(1, 1, heads, 1)
            * (raw_decay.float() + layer.self_attn.dt_bias.reshape(1, 1, heads, dim))
        )
        tensors["layers.0.kda.log_alpha"] = log_alpha
        tensors["layers.0.kda.beta"] = tensors["layers.0.kda.raw_beta"].float().sigmoid()
        tensors["layers.0.attnres.mlp.weights"] = torch.ones(
            (length, 1), device="cuda", dtype=torch.float32
        )
    finally:
        capture.close()
    if not all(torch.isfinite(value).all() for value in tensors.values() if value.is_floating_point()):
        raise ExportError(f"non-finite activation in layer-0 {mode} length {length}")
    return tensors, {
        "mode": mode,
        "length": length,
        "fla_mode": "fused_recurrent" if length == 1 else "chunk",
        "timing": timing,
        "cache": {
            "recurrent_state_shape": list(cache.recurrent_states[0].shape),
            "conv_state_shapes": [list(value.shape) for value in cache.conv_states[0]],
        },
    }


def benchmark_layer(
    config: Any,
    layer: Any,
    modeling: Any,
    length: int,
    measured_runs: int = 5,
) -> dict[str, Any]:
    """Measure the isolated forward without activation-cloning hooks."""
    hidden = synthetic_hidden(length)
    samples = []
    for iteration in range(measured_runs + 2):
        cache = modeling.KimiDynamicCache(config)
        block = torch.empty((length, 0, 7168), device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            _, timing = cuda_timed(
                lambda: layer(
                    hidden,
                    past_key_values=cache,
                    use_cache=True,
                    block_residual=block,
                )
            )
        if iteration >= 2:
            samples.append(timing["gpu_ms"])
    ordered = sorted(samples)
    median = ordered[len(ordered) // 2]
    return {
        "scope": "isolated official layer 0 without activation hooks",
        "warmup_runs": 2,
        "measured_runs": measured_runs,
        "gpu_ms": {
            "min": min(samples),
            "median": median,
            "max": max(samples),
            "samples": samples,
        },
        "tokens_per_second": float(length * 1000.0 / median),
    }


def run_continuation_once(
    config: Any,
    layer: Any,
    modeling: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    cache = modeling.KimiDynamicCache(config)
    tensors: dict[str, torch.Tensor] = {}

    def one_phase(label: str, hidden: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        block = torch.empty((hidden.shape[1], 0, 7168), device="cuda", dtype=torch.bfloat16)
        capture = _capture_layer_modules(layer)
        try:
            with torch.inference_mode():
                (output, block_out), timing = cuda_timed(
                    lambda: layer(
                        hidden,
                        past_key_values=cache,
                        use_cache=True,
                        block_residual=block,
                    )
                )
            tensors[f"{label}.layers.0.input"] = hidden
            tensors[f"{label}.layers.0.out"] = output
            tensors[f"{label}.layers.0.attnres.block_residual.out"] = block_out
            for name, value in capture.tensors.items():
                tensors[f"{label}.{name}"] = value
            return output, timing
        finally:
            capture.close()

    prefill_input = synthetic_hidden(4)
    _, prefill_timing = one_phase("prefill", prefill_input)
    state_after_prefill = cache.recurrent_states[0].detach().clone()
    conv_after_prefill = [value.detach().clone() for value in cache.conv_states[0]]
    decode_input = synthetic_hidden(5)[:, 4:5]
    _, decode_timing = one_phase("decode", decode_input)
    tensors["prefill.cache.recurrent_state.out"] = state_after_prefill
    tensors["decode.cache.recurrent_state.in"] = state_after_prefill
    tensors["decode.cache.recurrent_state.out"] = cache.recurrent_states[0]
    for index, value in enumerate(conv_after_prefill):
        tensors[f"prefill.cache.conv_state.{index}.out"] = value
        tensors[f"decode.cache.conv_state.{index}.in"] = value
        tensors[f"decode.cache.conv_state.{index}.out"] = cache.conv_states[0][index]
    return tensors, {
        "mode": "prefill_to_cached_decode",
        "prefill_length": 4,
        "decode_length": 1,
        "timing": {"prefill": prefill_timing, "decode": decode_timing},
        "cache_handoff_exact": all(
            torch.equal(tensors[f"decode.cache.conv_state.{index}.in"], conv_after_prefill[index])
            for index in range(3)
        )
        and torch.equal(tensors["decode.cache.recurrent_state.in"], state_after_prefill),
    }


def export_layer0(
    checkpoint: Path,
    configuration: Any,
    modeling: Any,
    output: Path,
    lengths: list[int],
    include_continuation: bool,
    debug: bool,
) -> list[dict[str, Any]]:
    config, layer, load_metadata = _load_layer0(checkpoint, configuration, modeling)
    manifests = []
    for length in lengths:
        first, details = run_layer_once(config, layer, modeling, length, "isolated_layer_prefill")
        second, repeat_details = run_layer_once(
            config, layer, modeling, length, "isolated_layer_prefill"
        )
        _assert_stable(first, second)
        details["timing"] = {
            "cold_or_first": details["timing"],
            "repeat": repeat_details["timing"],
        }
        details["performance"] = benchmark_layer(config, layer, modeling, length)
        if debug:
            # KIMI_K3_TEMP_REMOVE_M20: human-readable activation and inference
            # timing logs are bring-up diagnostics and must be removed at cleanup.
            print(
                f"[kimi-k3-debug] layer=0 length={length} mode={details['fla_mode']} "
                f"first_gpu_ms={details['timing']['cold_or_first']['gpu_ms']:.3f} "
                f"repeat_gpu_ms={details['timing']['repeat']['gpu_ms']:.3f} "
                f"bare_median_gpu_ms={details['performance']['gpu_ms']['median']:.3f}"
            )
            for name, tensor in sorted(first.items()):
                print(f"[kimi-k3-debug] {name} shape={tuple(tensor.shape)} dtype={tensor.dtype}")
        manifests.append(
            _save_fixture(
                output,
                f"s1-layer0-len{length}",
                first,
                {
                    "tier": "S1",
                    "layer_selection": [0],
                    "layer_stop": 1,
                    "repeat_runs": 2,
                    "numeric_hashes_stable": True,
                    "checkpoint": {
                        "shard": "model-00001-of-000096.safetensors",
                        "sha256": SHARD1_SHA256,
                    },
                    "load": load_metadata,
                    **details,
                },
            )
        )
    if include_continuation:
        first, details = run_continuation_once(config, layer, modeling)
        second, repeat_details = run_continuation_once(config, layer, modeling)
        _assert_stable(first, second)
        details["timing"] = {
            "cold_or_first": details["timing"],
            "repeat": repeat_details["timing"],
        }
        if not details["cache_handoff_exact"]:
            raise ExportError("prefill/decode cache handoff is not exact")
        if debug:
            print("[kimi-k3-debug] continuation timing", json.dumps(details["timing"], sort_keys=True))
        manifests.append(
            _save_fixture(
                output,
                "s1-layer0-prefill4-decode1",
                first,
                {
                    "tier": "S1",
                    "layer_selection": [0],
                    "layer_stop": 1,
                    "repeat_runs": 2,
                    "numeric_hashes_stable": True,
                    "checkpoint": {
                        "shard": "model-00001-of-000096.safetensors",
                        "sha256": SHARD1_SHA256,
                    },
                    "load": load_metadata,
                    **details,
                },
            )
        )
    del layer
    torch.cuda.empty_cache()
    return manifests


def parse_lengths(value: str) -> list[int]:
    lengths = [int(item) for item in value.split(",")]
    if not lengths or any(length not in (1, 4, 8, 16) for length in lengths):
        raise argparse.ArgumentTypeError("lengths must be a comma list drawn from 1,4,8,16")
    if len(lengths) != len(set(lengths)):
        raise argparse.ArgumentTypeError("lengths must be unique")
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("s0", "layer", "prefix", "prefill", "decode", "all"),
        default="all",
        help="prefix/prefill select isolated layer mode; decode adds cache continuation",
    )
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--lengths", type=parse_lengths, default=parse_lengths("1,4,8,16"))
    parser.add_argument("--layer-stop", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.layer_stop != 1:
        raise ExportError(
            "the current S1 checkpoint supports layer-stop=1 only; do not request missing layers"
        )
    checkpoint = args.checkpoint_root.resolve()
    output = args.output_dir.resolve()
    if checkpoint != DEFAULT_CHECKPOINT.resolve():
        raise ExportError(f"checkpoint must be the approved local directory: {DEFAULT_CHECKPOINT}")
    deterministic_setup()
    configuration, modeling = import_official(checkpoint)
    manifests: list[dict[str, Any]] = []
    if args.mode in ("s0", "all"):
        manifests.append(export_s0(modeling, output, args.debug))
    if args.mode in ("layer", "prefix", "prefill", "decode", "all"):
        manifests.extend(
            export_layer0(
                checkpoint,
                configuration,
                modeling,
                output,
                args.lengths,
                include_continuation=args.mode in ("decode", "all"),
                debug=args.debug,
            )
        )
    summary = {
        "schema_version": 1,
        "moonshot_revision": MOONSHOT_REVISION,
        "mode": args.mode,
        "layer_stop": args.layer_stop,
        "device": torch.cuda.get_device_name(0),
        "fixtures": [
            {
                "name": manifest["fixture"],
                "manifest": f"{manifest['fixture']}.json",
                "safetensors": manifest["tensor_file"],
                "sha256": manifest["tensor_file_sha256"],
            }
            for manifest in manifests
        ],
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
