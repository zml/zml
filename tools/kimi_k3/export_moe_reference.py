#!/usr/bin/env python3
"""Export a bounded real-weight Stable LatentMoE oracle for Gate B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
from typing import Any

import torch
import torch.nn.functional as torch_functional
from safetensors import safe_open

from export_layer0_prefix_reference import semantic_sha256
from export_reference import (
    DEFAULT_CHECKPOINT,
    MOONSHOT_REVISION,
    _assert_stable,
    _save_fixture,
    cuda_timed,
    deterministic_setup,
    sha256_file,
    tensor_bytes,
)


ROOT = Path("/dev/shm/kimi-k3")
OUTPUT = ROOT / "artifacts/fixtures/milestone-11"
ROUTER_FIXTURE = ROOT / "artifacts/fixtures/milestone-10/router-reference.safetensors"
ROUTER_SEMANTIC_SHA256 = "4eb2f2606d40a86f317253d4baba3329693871ce1742144646c8417a0f5da664"
SHARD = "model-00002-of-000096.safetensors"
SHARD_SHA256 = "26a3284e1d2cb567934ebef002e6a1813551d646739e8bcb1e9e3fe7f878e0f5"
LAYER = "language_model.model.layers.1.block_sparse_moe"
FP4 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def dequantize(packed: torch.Tensor, scale: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    packed = packed.to(device=device, dtype=torch.uint8)
    scale = scale.to(device=device, dtype=torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    nibble = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    table = FP4.to(device)
    sign = torch.where((nibble & 0x08) != 0, -1.0, 1.0)
    values = table[(nibble & 0x07).long()] * sign
    expanded = torch.exp2(scale.float() - 127.0).repeat_interleave(32, dim=-1)
    if values.shape != expanded.shape:
        raise RuntimeError(f"MXFP4 shape mismatch: {values.shape} != {expanded.shape}")
    return values * expanded


def situ(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    dtype = gate.dtype
    gate_f32 = gate.float()
    up_f32 = up.float()
    return (
        4.0 * torch.tanh(gate_f32 / 4.0) * torch.sigmoid(gate_f32)
        * (25.0 * torch.tanh(up_f32 / 25.0))
    ).to(dtype)


def rms_norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    dtype = value.dtype
    normalized = value.float() * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + 1e-5)
    return weight * normalized.to(dtype)


def load_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], torch.Tensor]:
    with safe_open(ROUTER_FIXTURE, framework="pt", device="cpu") as values:
        hidden = values.get_tensor("real.hidden").cuda()
        global_ids = values.get_tensor("real.topk_ids").to(torch.int64)
        route_weights = values.get_tensor("real.topk_weights").cuda()
    selected = sorted(set(global_ids.flatten().tolist()))
    global_to_local = {expert: index for index, expert in enumerate(selected)}
    local_ids = torch.tensor(
        [[global_to_local[int(expert)] for expert in row] for row in global_ids.reshape(-1, 16)],
        dtype=torch.int64,
    ).reshape_as(global_ids)
    return hidden, global_ids.cuda(), route_weights, selected, local_ids


def load_weights(checkpoint: Path, selected: list[int]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    shard = checkpoint / SHARD
    if sha256_file(shard) != SHARD_SHA256:
        raise RuntimeError(f"local shard-2 hash mismatch: {shard}")
    packed: dict[str, list[torch.Tensor]] = {
        "w1.packed": [], "w1.scale": [],
        "w2.packed": [], "w2.scale": [],
        "w3.packed": [], "w3.scale": [],
    }
    dense_names = {
        "routed_down": f"{LAYER}.routed_expert_down_proj.weight",
        "routed_norm": f"{LAYER}.routed_expert_norm.weight",
        "routed_up": f"{LAYER}.routed_expert_up_proj.weight",
        "shared_gate": f"{LAYER}.shared_experts.gate_proj.weight",
        "shared_up": f"{LAYER}.shared_experts.up_proj.weight",
        "shared_down": f"{LAYER}.shared_experts.down_proj.weight",
    }
    with safe_open(shard, framework="pt", device="cpu") as values:
        for expert in selected:
            for matrix in ("w1", "w2", "w3"):
                prefix = f"{LAYER}.experts.{expert}.{matrix}"
                packed[f"{matrix}.packed"].append(values.get_tensor(f"{prefix}.weight_packed"))
                packed[f"{matrix}.scale"].append(values.get_tensor(f"{prefix}.weight_scale"))
        dense = {name: values.get_tensor(key).cuda() for name, key in dense_names.items()}
    return {name: torch.stack(parts) for name, parts in packed.items()}, dense


def run_reference(
    hidden: torch.Tensor,
    route_weights: torch.Tensor,
    local_ids: torch.Tensor,
    selected_weights: dict[str, torch.Tensor],
    dense: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    token_hidden = hidden.reshape(-1, hidden.shape[-1])
    routed_input = torch_functional.linear(token_hidden, dense["routed_down"])
    route_outputs = torch.empty(
        (*local_ids.reshape(-1, 16).shape, routed_input.shape[-1]),
        device="cuda",
        dtype=torch.bfloat16,
    )
    probe_w13 = torch.sin(
        torch.arange(len(selected_weights["w1.packed"]) * 3584, device="cuda", dtype=torch.float32)
        .reshape(len(selected_weights["w1.packed"]), 3584)
        * 0.00031
    )
    probe_w2 = torch.cos(
        torch.arange(len(selected_weights["w2.packed"]) * 3072, device="cuda", dtype=torch.float32)
        .reshape(len(selected_weights["w2.packed"]), 3072)
        * 0.00029
    )
    probe_outputs: dict[str, list[torch.Tensor]] = {"w1": [], "w2": [], "w3": []}
    flat_local = local_ids.reshape(-1, 16)
    with torch.inference_mode():
        for local in range(len(selected_weights["w1.packed"])):
            matrices = {
                matrix: dequantize(
                    selected_weights[f"{matrix}.packed"][local],
                    selected_weights[f"{matrix}.scale"][local],
                )
                for matrix in ("w1", "w2", "w3")
            }
            probe_outputs["w1"].append(torch_functional.linear(probe_w13[local], matrices["w1"]))
            probe_outputs["w2"].append(torch_functional.linear(probe_w2[local], matrices["w2"]))
            probe_outputs["w3"].append(torch_functional.linear(probe_w13[local], matrices["w3"]))
            positions = torch.nonzero(flat_local == local, as_tuple=False)
            if positions.numel() == 0:
                raise RuntimeError(f"selected local expert {local} has no route")
            inputs = routed_input[positions[:, 0]]
            gate = torch_functional.linear(inputs.float(), matrices["w1"]).to(torch.bfloat16)
            up = torch_functional.linear(inputs.float(), matrices["w3"]).to(torch.bfloat16)
            activated = situ(gate, up)
            output = torch_functional.linear(activated.float(), matrices["w2"]).to(torch.bfloat16)
            route_outputs[positions[:, 0], positions[:, 1]] = output
            del matrices, gate, up, activated, output
        combined = (route_outputs.float() * route_weights.reshape(-1, 16, 1)).sum(1).to(torch.bfloat16)
        routed_norm = rms_norm(combined, dense["routed_norm"])
        routed_up = torch_functional.linear(routed_norm, dense["routed_up"])
        shared_gate = torch_functional.linear(token_hidden, dense["shared_gate"])
        shared_up = torch_functional.linear(token_hidden, dense["shared_up"])
        shared_situ = situ(shared_gate, shared_up)
        shared_output = torch_functional.linear(shared_situ, dense["shared_down"])
        final = routed_up.reshape_as(hidden) + shared_output.reshape_as(hidden)
    return {
        "moe.input": hidden,
        "moe.local_route_ids": local_ids,
        "moe.route_weights": route_weights,
        "moe.routed_down": routed_input,
        "moe.route_outputs": route_outputs,
        "moe.combined_latent": combined,
        "moe.routed_norm": routed_norm,
        "moe.routed_up": routed_up,
        "moe.shared_gate": shared_gate,
        "moe.shared_up": shared_up,
        "moe.shared_situ": shared_situ,
        "moe.shared_output": shared_output,
        "moe.final": final,
        "probe.w13.input": probe_w13,
        "probe.w2.input": probe_w2,
        "probe.w1.output": torch.stack(probe_outputs["w1"]),
        "probe.w2.output": torch.stack(probe_outputs["w2"]),
        "probe.w3.output": torch.stack(probe_outputs["w3"]),
    }


def per_matrix_hashes(selected: list[int], weights: dict[str, torch.Tensor]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for local, expert in enumerate(selected):
        for matrix in ("w1", "w2", "w3"):
            packed = weights[f"{matrix}.packed"][local].contiguous()
            scale = weights[f"{matrix}.scale"][local].contiguous()
            result[f"{expert}.{matrix}"] = {
                "global_expert": expert,
                "local_expert": local,
                "packed_sha256": hashlib.sha256(tensor_bytes(packed)).hexdigest(),
                "scale_sha256": hashlib.sha256(tensor_bytes(scale)).hexdigest(),
                "probe_output_key": f"probe.{matrix}.output",
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    checkpoint = args.checkpoint_root.resolve()
    if checkpoint != DEFAULT_CHECKPOINT.resolve():
        raise RuntimeError(f"checkpoint must be the approved local directory: {DEFAULT_CHECKPOINT}")
    deterministic_setup()
    hidden, global_ids, route_weights, selected, local_ids = load_inputs()
    selected_weights, dense = load_weights(checkpoint, selected)
    first, first_timing = cuda_timed(
        lambda: run_reference(hidden, route_weights, local_ids.cuda(), selected_weights, dense)
    )
    second, repeat_timing = cuda_timed(
        lambda: run_reference(hidden, route_weights, local_ids.cuda(), selected_weights, dense)
    )
    _assert_stable(first, second)
    tensors = {
        "moe.global_route_ids": global_ids.cpu(),
        "moe.selected_global_ids": torch.tensor(selected, dtype=torch.int64),
        **{f"selected.{name}": value for name, value in selected_weights.items()},
        **{f"dense.{name}": value.cpu() for name, value in dense.items()},
        **first,
    }
    peak_hbm = torch.cuda.max_memory_allocated()
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    manifest = _save_fixture(
        args.output_dir.resolve(),
        "selected-moe-reference",
        tensors,
        {
            "mode": "test_only_selected_expert_stable_latent_moe",
            "numeric_hashes_stable": True,
            "repeat_runs": 2,
            "tensor_semantic_sha256": semantic_sha256(tensors),
            "checkpoint": {SHARD: SHARD_SHA256},
            "router_fixture_semantic_sha256": ROUTER_SEMANTIC_SHA256,
            "selected_global_experts": selected,
            "selected_expert_count": len(selected),
            "route_count": int(global_ids.numel()),
            "global_to_local": {str(expert): local for local, expert in enumerate(selected)},
            "per_matrix": per_matrix_hashes(selected, selected_weights),
            "matrix_probe_count": len(selected) * 3,
            "timing": {"cold_or_first": first_timing, "repeat": repeat_timing},
            "peak_memory": {"cuda_allocated_bytes": peak_hbm, "host_max_rss_bytes": peak_rss},
            "compact_map_scope": "test fixture and harness only; not accepted by production MoE API",
        },
    )
    summary = {
        "schema_version": 1,
        "moonshot_revision": MOONSHOT_REVISION,
        "device": torch.cuda.get_device_name(0),
        "fixture": manifest["fixture"],
        "manifest": "selected-moe-reference.json",
        "safetensors": manifest["tensor_file"],
        "sha256": manifest["tensor_file_sha256"],
        "semantic_sha256": manifest["tensor_semantic_sha256"],
        "selected_experts": len(selected),
        "routes": int(global_ids.numel()),
    }
    (args.output_dir.resolve() / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    # KIMI_K3_TEMP_REMOVE_M20: selected-expert inventory, activation shapes,
    # and oracle timing are Gate B diagnostics removed during cleanup.
    if args.debug:
        print("[kimi-k3-debug] selected experts", selected)
        print("[kimi-k3-debug] peak memory", manifest["peak_memory"])
        print("[kimi-k3-debug] timing", json.dumps(manifest["timing"], sort_keys=True))
        for name, value in sorted(first.items()):
            print(f"[kimi-k3-debug] {name} shape={tuple(value.shape)} dtype={value.dtype}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
