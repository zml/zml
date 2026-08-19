#!/usr/bin/env python3
"""Export official and adversarial Kimi K3 router fixtures on NVIDIA CUDA."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
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
    import_official,
    sha256_file,
)


ROOT = Path("/ephemeral/kimi-k3")
OUTPUT = ROOT / "artifacts/fixtures/milestone-10"
LAYER1_SHARD = "model-00002-of-000096.safetensors"
LAYER1_SHA256 = "26a3284e1d2cb567934ebef002e6a1813551d646739e8bcb1e9e3fe7f878e0f5"
PREFIX_FIXTURE = ROOT / "artifacts/fixtures/milestone-9/s2-layer0-prefix-len4.safetensors"
PREFIX_SEMANTIC_SHA256 = "6f35e2906880085829d5e411cc4d7fcc1b598397055fe7460ab91865ab05b15d"


@dataclass(frozen=True)
class RouteConfig:
    top_k: int
    num_expert_group: int = 1
    topk_group: int = 1
    scaling_factor: float = 1.0


def canonical_route(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    config: RouteConfig,
) -> dict[str, torch.Tensor]:
    logits = torch_functional.linear(hidden.float(), weight.float())
    raw = logits.sigmoid()
    selection = raw + bias.float()
    choice = selection
    experts = selection.shape[-1]
    if config.num_expert_group > 1 and config.num_expert_group > config.topk_group:
        grouped = selection.reshape(*selection.shape[:-1], config.num_expert_group, -1)
        group_scores = grouped.topk(2, dim=-1).values.sum(-1)
        groups = torch.argsort(group_scores, dim=-1, descending=True, stable=True)[
            ..., : config.topk_group
        ]
        mask = torch.zeros_like(group_scores, dtype=torch.bool)
        mask.scatter_(-1, groups, True)
        choice = grouped.masked_fill(~mask.unsqueeze(-1), float("-inf")).reshape(
            *selection.shape[:-1], experts
        )
    ids = torch.argsort(choice, dim=-1, descending=True, stable=True)[..., : config.top_k]
    chosen = raw.gather(-1, ids)
    normalized = chosen / (chosen.sum(-1, keepdim=True) + 1e-20)
    return {
        "logits": logits,
        "raw_scores": raw,
        "selection_scores": selection,
        "topk_ids": ids.to(torch.int64),
        "topk_raw_weights": chosen,
        "topk_weights": normalized * config.scaling_factor,
    }


def check_official_gate(
    modeling: Any,
    hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    config: RouteConfig,
    expected: dict[str, torch.Tensor],
) -> None:
    gate_config = SimpleNamespace(
        num_experts_per_token=config.top_k,
        num_experts=weight.shape[0],
        routed_scaling_factor=config.scaling_factor,
        moe_router_activation_func="sigmoid",
        num_expert_group=config.num_expert_group,
        topk_group=config.topk_group,
        moe_renormalize=True,
        hidden_size=weight.shape[1],
    )
    with torch.device("meta"):
        gate = modeling.KimiMoEGate(gate_config)
    gate.weight = torch.nn.Parameter(weight, requires_grad=False)
    gate.e_score_correction_bias = torch.nn.Parameter(bias, requires_grad=False)
    gate.eval()
    with torch.inference_mode():
        ids, weights = gate(hidden)
    flat_official_ids = ids.reshape(-1, config.top_k)
    flat_expected_ids = expected["topk_ids"].reshape(-1, config.top_k)
    flat_official_weights = weights.reshape(-1, config.top_k)
    flat_expected_weights = expected["topk_weights"].reshape(-1, config.top_k)
    for row in range(flat_official_ids.shape[0]):
        official = {
            int(expert): float(value)
            for expert, value in zip(flat_official_ids[row], flat_official_weights[row])
        }
        canonical = {
            int(expert): float(value)
            for expert, value in zip(flat_expected_ids[row], flat_expected_weights[row])
        }
        if official.keys() != canonical.keys():
            raise RuntimeError(f"official selected expert set differs at row {row}")
        for expert in official:
            if abs(official[expert] - canonical[expert]) > 1e-6:
                raise RuntimeError(f"official aligned weight differs at row {row}, expert {expert}")


def load_layer1_prefix_hidden(
    checkpoint: Path,
    configuration: Any,
    modeling: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    config_data = json.loads((checkpoint / "config.json").read_text())["text_config"]
    config = configuration.KimiLinearConfig(**config_data)
    with torch.device("meta"):
        layer = modeling.KimiDecoderLayer(config, 1)
    prefix = "language_model.model.layers.1."
    state: dict[str, torch.Tensor] = {}
    shard = checkpoint / LAYER1_SHARD
    if sha256_file(shard) != LAYER1_SHA256:
        raise RuntimeError(f"local shard-2 hash mismatch: {shard}")
    with safe_open(shard, framework="pt", device="cuda:0") as tensors:
        for key in tensors.keys():
            if not key.startswith(prefix):
                continue
            local = key.removeprefix(prefix)
            if any(
                marker in local
                for marker in (
                    "experts.",
                    "shared_experts.",
                    "routed_expert_down_proj",
                    "routed_expert_norm",
                    "routed_expert_up_proj",
                )
            ):
                continue
            state[local] = tensors.get_tensor(key)
    checkpoint_a_log = state.pop("self_attn.A_log")
    if tuple(checkpoint_a_log.shape) != (128,) or layer.self_attn.num_heads != 96:
        raise RuntimeError("unexpected layer-1 A_log compatibility condition")
    layer.self_attn.A_log = torch.nn.Parameter(checkpoint_a_log, requires_grad=False)
    missing, unexpected = layer.load_state_dict(state, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"unexpected layer-1 checkpoint keys: {unexpected}")
    required_missing = [
        name
        for name in missing
        if name != "self_attn.A_log"
        and not any(
            marker in name
            for marker in (
                "experts.",
                "shared_experts.",
                "routed_expert_down_proj",
                "routed_expert_norm",
                "routed_expert_up_proj",
            )
        )
    ]
    if required_missing:
        raise RuntimeError(f"missing required layer-1 tensors: {required_missing}")
    layer.eval()

    with safe_open(PREFIX_FIXTURE, framework="pt", device="cpu") as fixture:
        prefix_sum = fixture.get_tensor("prefix.layer0.out").cuda()
        block_residual = fixture.get_tensor("prefix.layer0.block_residual.out").cuda()
    batch, sequence, hidden_size = prefix_sum.shape
    with torch.inference_mode():
        selected_input = modeling._apply_attn_res(
            prefix_sum.reshape(-1, hidden_size),
            block_residual,
            layer.self_attention_res_proj,
            layer.self_attention_res_norm,
        ).reshape(batch, sequence, hidden_size)
        normalized_input = layer.input_layernorm(selected_input)
        cache = modeling.KimiDynamicCache(config)
        attention = layer.self_attn(normalized_input, cache_params=cache)
        prefix_after_attention = prefix_sum + attention
        selected_mlp = modeling._apply_attn_res(
            prefix_after_attention.reshape(-1, hidden_size),
            block_residual,
            layer.mlp_res_proj,
            layer.mlp_res_norm,
        ).reshape(batch, sequence, hidden_size)
        router_hidden = layer.post_attention_layernorm(selected_mlp)
    return router_hidden, {
        "loaded_tensors": len(state) + 1,
        "loaded_bytes": sum(value.numel() * value.element_size() for value in state.values())
        + checkpoint_a_log.numel() * checkpoint_a_log.element_size(),
        "ignored_missing_parameters": len(missing) - 1,
    }


def build_fixture(
    checkpoint: Path,
    configuration: Any,
    modeling: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors: dict[str, torch.Tensor] = {}
    configs: dict[str, dict[str, Any]] = {}

    real_hidden, layer_load = load_layer1_prefix_hidden(checkpoint, configuration, modeling)
    shard = checkpoint / LAYER1_SHARD
    with safe_open(shard, framework="pt", device="cuda:0") as values:
        real_weight = values.get_tensor(
            "language_model.model.layers.1.block_sparse_moe.gate.weight"
        )
        real_bias = values.get_tensor(
            "language_model.model.layers.1.block_sparse_moe.gate.e_score_correction_bias"
        )
    cases: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, RouteConfig, bool]] = {
        "real": (real_hidden, real_weight, real_bias, RouteConfig(top_k=16), True),
    }

    tie_hidden = torch.zeros((1, 1, 4), device="cuda", dtype=torch.float32)
    tie_weight = torch.zeros((20, 4), device="cuda", dtype=torch.float32)
    tie_bias = torch.zeros(20, device="cuda", dtype=torch.float32)
    cases["tie"] = (tie_hidden, tie_weight, tie_bias, RouteConfig(top_k=16), False)

    positions = torch.arange(16, device="cuda", dtype=torch.float32).reshape(1, 2, 8)
    bias_hidden = torch.sin(positions * 0.37)
    expert_positions = torch.arange(24 * 8, device="cuda", dtype=torch.float32).reshape(24, 8)
    bias_weight = torch.cos(expert_positions * 0.071)
    bias = torch.linspace(-0.7, 0.9, 24, device="cuda", dtype=torch.float32)
    cases["bias"] = (bias_hidden, bias_weight, bias, RouteConfig(top_k=16), True)

    grouped_hidden = torch.cos(positions * 0.23)
    grouped_weight = torch.sin(expert_positions * 0.053)
    grouped_bias = torch.cos(torch.arange(24, device="cuda", dtype=torch.float32) * 0.41) * 0.13
    cases["grouped"] = (
        grouped_hidden,
        grouped_weight,
        grouped_bias,
        RouteConfig(top_k=4, num_expert_group=4, topk_group=2, scaling_factor=1.25),
        True,
    )

    official_checks = 0
    for name, (hidden, weight, bias_value, config, check_official) in cases.items():
        expected = canonical_route(hidden, weight, bias_value, config)
        if check_official:
            check_official_gate(modeling, hidden, weight, bias_value, config, expected)
            official_checks += hidden.numel() // hidden.shape[-1]
        tensors[f"{name}.hidden"] = hidden
        if name != "real":
            tensors[f"{name}.weight"] = weight
            tensors[f"{name}.correction_bias"] = bias_value
        for boundary, value in expected.items():
            tensors[f"{name}.{boundary}"] = value
        histogram = torch.bincount(expected["topk_ids"].reshape(-1), minlength=weight.shape[0])
        configs[name] = {
            **config.__dict__,
            "experts": weight.shape[0],
            "hidden_size": weight.shape[1],
            "tokens": hidden.numel() // hidden.shape[-1],
            "route_histogram": {
                str(index): int(count)
                for index, count in enumerate(histogram.tolist())
                if count
            },
            "official_gate_set_and_aligned_weight_check": check_official,
        }
    # The bias case must prove that selection bias changed at least one route
    # relative to ranking the raw mixture weights alone.
    raw_ids = torch.argsort(
        tensors["bias.raw_scores"], dim=-1, descending=True, stable=True
    )[..., :16]
    if torch.equal(raw_ids, tensors["bias.topk_ids"]):
        raise RuntimeError("adversarial bias fixture did not change route selection")
    return tensors, {
        "cases": configs,
        "official_checked_rows": official_checks,
        "layer1_load": layer_load,
        "tie_policy": "stable descending order; exact ties select lower expert index first",
        "bias_not_used_as_mixture_weight": True,
    }


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
    configuration, modeling = import_official(checkpoint)
    first, first_timing = cuda_timed(lambda: build_fixture(checkpoint, configuration, modeling))
    second, repeat_timing = cuda_timed(lambda: build_fixture(checkpoint, configuration, modeling))
    first_tensors, details = first
    second_tensors, second_details = second
    _assert_stable(first_tensors, second_tensors)
    if details != second_details:
        raise RuntimeError("router fixture metadata changed across repeat runs")
    manifest = _save_fixture(
        args.output_dir.resolve(),
        "router-reference",
        first_tensors,
        {
            "mode": "official_real_layer1_and_synthetic_router",
            "numeric_hashes_stable": True,
            "repeat_runs": 2,
            "tensor_semantic_sha256": semantic_sha256(first_tensors),
            "checkpoint": {LAYER1_SHARD: LAYER1_SHA256},
            "prefix_fixture_semantic_sha256": PREFIX_SEMANTIC_SHA256,
            "timing": {"cold_or_first": first_timing, "repeat": repeat_timing},
            **details,
        },
    )
    summary = {
        "schema_version": 1,
        "moonshot_revision": MOONSHOT_REVISION,
        "device": torch.cuda.get_device_name(0),
        "fixture": manifest["fixture"],
        "manifest": "router-reference.json",
        "safetensors": manifest["tensor_file"],
        "sha256": manifest["tensor_file_sha256"],
        "semantic_sha256": manifest["tensor_semantic_sha256"],
    }
    (args.output_dir.resolve() / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    # KIMI_K3_TEMP_REMOVE_M20: router activation inventories and reference
    # timings are differential diagnostics removed during cleanup.
    if args.debug:
        print("[kimi-k3-debug] router timing", json.dumps(manifest["timing"], sort_keys=True))
        for name, tensor in sorted(first_tensors.items()):
            print(f"[kimi-k3-debug] {name} shape={tuple(tensor.shape)} dtype={tensor.dtype}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
