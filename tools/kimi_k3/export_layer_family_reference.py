#!/usr/bin/env python3
"""Export sequential real-weight layer 1/2/3 KDA/MLA+MoE reference fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as torch_functional
from safetensors import safe_open

from export_layer0_prefix_reference import TOKEN_IDS, semantic_sha256
from export_mla_reference import causal_mask
from export_moe_reference import dequantize, rms_norm, situ
from export_reference import (
    DEFAULT_CHECKPOINT,
    MOONSHOT_REVISION,
    _assert_stable,
    _save_fixture,
    cuda_timed,
    deterministic_setup,
    import_official,
    sha256_file,
    tensor_bytes,
)
from export_router_reference import canonical_route, RouteConfig


ROOT = Path(os.environ.get("KIMI_K3_WORKSPACE", Path(__file__).resolve().parents[3]))
OUTPUT = ROOT / "artifacts/fixtures/milestone-14"
PREFIX_FIXTURE = ROOT / "artifacts/fixtures/milestone-9/s2-layer0-prefix-len4.safetensors"
SHARDS = {
    1: ("model-00002-of-000096.safetensors", "26a3284e1d2cb567934ebef002e6a1813551d646739e8bcb1e9e3fe7f878e0f5"),
    2: ("model-00003-of-000096.safetensors", "e54af9de4c554956082364010f732443bcd5097390f0121a33fb35e37280b5a9"),
    3: ("model-00004-of-000096.safetensors", "5955fd8feda89b1af8400c25e885e7177d47edff155f54b318beb8dd1cec5c05"),
}


def load_prefix_source(path: Path = PREFIX_FIXTURE) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Validate and load the exact Milestone 9 fixture consumed by this export."""
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("moonshot_revision") != MOONSHOT_REVISION
        or manifest.get("token_ids") != list(TOKEN_IDS)
        or manifest.get("layer_stop") != 1
        or manifest.get("cpu_inference_fallback")
    ):
        raise RuntimeError(f"prefix fixture execution contract mismatch: {manifest_path}")
    if manifest.get("tensor_file") != path.name:
        raise RuntimeError(f"prefix fixture manifest points to a different tensor file: {manifest_path}")
    file_sha256 = sha256_file(path)
    if file_sha256 != manifest.get("tensor_file_sha256"):
        raise RuntimeError(f"prefix fixture file hash mismatch: {path}")

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as values:
        for name in values.keys():
            value = values.get_tensor(name).contiguous()
            record = manifest.get("tensors", {}).get(name)
            dtype = str(value.dtype).removeprefix("torch.")
            if record is None or list(value.shape) != record.get("shape") or dtype != record.get("dtype"):
                raise RuntimeError(f"prefix fixture tensor contract mismatch: {name}")
            if hashlib.sha256(tensor_bytes(value)).hexdigest() != record.get("sha256"):
                raise RuntimeError(f"prefix fixture tensor hash mismatch: {name}")
            if value.is_floating_point() and not torch.isfinite(value).all():
                raise RuntimeError(f"non-finite prefix fixture tensor: {name}")
            tensors[name] = value
    if set(tensors) != set(manifest.get("tensors", {})):
        raise RuntimeError("prefix fixture tensor inventory mismatch")
    semantic = semantic_sha256(tensors)
    if semantic != manifest.get("tensor_semantic_sha256"):
        raise RuntimeError("prefix fixture aggregate semantic hash mismatch")
    required = {"prefix.layer0.out", "prefix.layer0.block_residual.out"}
    if not required.issubset(tensors):
        raise RuntimeError(f"prefix fixture is missing required tensors: {sorted(required - tensors.keys())}")
    return tensors, {
        "fixture": manifest.get("fixture"),
        "manifest": manifest_path.name,
        "manifest_sha256": sha256_file(manifest_path),
        "moonshot_revision": manifest["moonshot_revision"],
        "tensor_file": path.name,
        "tensor_file_sha256": file_sha256,
        "tensor_semantic_sha256": semantic,
    }


def load_layer(
    checkpoint: Path,
    config: Any,
    modeling: Any,
    layer_index: int,
) -> tuple[Any, dict[str, torch.Tensor]]:
    with torch.device("meta"):
        layer = modeling.KimiDecoderLayer(config, layer_index)
    shard_name, shard_hash = SHARDS[layer_index]
    shard = checkpoint / shard_name
    if sha256_file(shard) != shard_hash:
        raise RuntimeError(f"local layer-{layer_index} shard hash mismatch: {shard}")
    prefix = f"language_model.model.layers.{layer_index}."
    state: dict[str, torch.Tensor] = {}
    with safe_open(shard, framework="pt", device="cuda:0") as values:
        for key in values.keys():
            if not key.startswith(prefix):
                continue
            local = key.removeprefix(prefix)
            if any(
                marker in local
                for marker in (
                    "block_sparse_moe.experts.",
                    "block_sparse_moe.shared_experts.",
                    "block_sparse_moe.routed_expert_down_proj",
                    "block_sparse_moe.routed_expert_norm",
                    "block_sparse_moe.routed_expert_up_proj",
                )
            ):
                continue
            state[local] = values.get_tensor(key)
    checkpoint_a_log = state.pop("self_attn.A_log", None)
    if checkpoint_a_log is not None:
        if tuple(checkpoint_a_log.shape) != (128,) or layer.self_attn.num_heads != 96:
            raise RuntimeError(f"unexpected layer-{layer_index} A_log condition")
        layer.self_attn.A_log = torch.nn.Parameter(checkpoint_a_log, requires_grad=False)
    missing, unexpected = layer.load_state_dict(state, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"unexpected layer-{layer_index} keys: {unexpected}")
    allowed = (
        "block_sparse_moe.experts.",
        "block_sparse_moe.shared_experts.",
        "block_sparse_moe.routed_expert_down_proj",
        "block_sparse_moe.routed_expert_norm",
        "block_sparse_moe.routed_expert_up_proj",
    )
    required_missing = [name for name in missing if name != "self_attn.A_log" and not any(x in name for x in allowed)]
    if required_missing:
        raise RuntimeError(f"missing layer-{layer_index} non-expert keys: {required_missing}")
    layer.eval()
    exported = dict(state)
    if checkpoint_a_log is not None:
        exported["self_attn.A_log"] = checkpoint_a_log
    return layer, exported


def load_selected_moe(
    checkpoint: Path,
    layer_index: int,
    selected: list[int],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    shard_name, _ = SHARDS[layer_index]
    layer = f"language_model.model.layers.{layer_index}.block_sparse_moe"
    packed: dict[str, list[torch.Tensor]] = {
        "w1.packed": [], "w1.scale": [], "w2.packed": [],
        "w2.scale": [], "w3.packed": [], "w3.scale": [],
    }
    dense_names = {
        "routed_down": f"{layer}.routed_expert_down_proj.weight",
        "routed_norm": f"{layer}.routed_expert_norm.weight",
        "routed_up": f"{layer}.routed_expert_up_proj.weight",
        "shared_gate": f"{layer}.shared_experts.gate_proj.weight",
        "shared_up": f"{layer}.shared_experts.up_proj.weight",
        "shared_down": f"{layer}.shared_experts.down_proj.weight",
    }
    with safe_open(checkpoint / shard_name, framework="pt", device="cpu") as values:
        for expert in selected:
            for matrix in ("w1", "w2", "w3"):
                prefix = f"{layer}.experts.{expert}.{matrix}"
                packed[f"{matrix}.packed"].append(values.get_tensor(f"{prefix}.weight_packed"))
                packed[f"{matrix}.scale"].append(values.get_tensor(f"{prefix}.weight_scale"))
        dense = {name: values.get_tensor(key).cuda() for name, key in dense_names.items()}
    return {name: torch.stack(parts) for name, parts in packed.items()}, dense


def selected_moe(
    hidden: torch.Tensor,
    route_weights: torch.Tensor,
    local_ids: torch.Tensor,
    packed: dict[str, torch.Tensor],
    dense: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    token_hidden = hidden.reshape(-1, hidden.shape[-1])
    routed_down = torch_functional.linear(token_hidden, dense["routed_down"])
    route_outputs = torch.empty((token_hidden.shape[0], 16, 3584), device="cuda", dtype=torch.bfloat16)
    flat_local = local_ids.reshape(-1, 16)
    for local in range(packed["w1.packed"].shape[0]):
        matrices = {
            matrix: dequantize(packed[f"{matrix}.packed"][local], packed[f"{matrix}.scale"][local])
            for matrix in ("w1", "w2", "w3")
        }
        positions = torch.nonzero(flat_local == local, as_tuple=False)
        inputs = routed_down[positions[:, 0]]
        gate = torch_functional.linear(inputs.float(), matrices["w1"]).to(torch.bfloat16)
        up = torch_functional.linear(inputs.float(), matrices["w3"]).to(torch.bfloat16)
        route_outputs[positions[:, 0], positions[:, 1]] = torch_functional.linear(
            situ(gate, up).float(), matrices["w2"]
        ).to(torch.bfloat16)
    combined = (route_outputs.float() * route_weights.reshape(-1, 16, 1)).sum(1).to(torch.bfloat16)
    routed_norm = rms_norm(combined, dense["routed_norm"])
    routed_up = torch_functional.linear(routed_norm, dense["routed_up"])
    shared_gate = torch_functional.linear(token_hidden, dense["shared_gate"])
    shared_up = torch_functional.linear(token_hidden, dense["shared_up"])
    shared_output = torch_functional.linear(situ(shared_gate, shared_up), dense["shared_down"])
    final = routed_up.reshape_as(hidden) + shared_output.reshape_as(hidden)
    return {
        "routed_down": routed_down,
        "route_outputs": route_outputs,
        "combined_latent": combined,
        "routed_norm": routed_norm,
        "routed_up": routed_up,
        "shared_output": shared_output,
        "output": final,
    }


def attention_forward(
    layer: Any,
    hidden: torch.Tensor,
    cache: Any,
    past_length: int,
) -> torch.Tensor:
    if layer.is_linear_attn:
        return layer.self_attn(hidden, cache_params=cache)
    sequence = hidden.shape[1]
    return layer.self_attn(
        hidden,
        attention_mask=causal_mask(sequence, past_length + sequence, past_length),
        past_key_values=cache,
    )


def snapshot_cache(cache: Any, layer_index: int, linear: bool, prefix: str) -> dict[str, torch.Tensor]:
    if linear:
        tensors = {
            f"{prefix}.recurrent": cache.recurrent_states[layer_index].clone(),
        }
        for index, value in enumerate(cache.conv_states[layer_index]):
            tensors[f"{prefix}.conv{index}"] = value.clone()
        return tensors
    return {
        f"{prefix}.key": cache.key_cache[layer_index].clone(),
        f"{prefix}.value": cache.value_cache[layer_index].clone(),
    }


def close_report(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    delta = (actual_f32 - expected_f32).abs()
    close = torch.isclose(actual_f32, expected_f32, atol=atol, rtol=rtol)
    return {
        "atol": atol,
        "rtol": rtol,
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean().item()) if delta.numel() else 0.0,
        "close_fraction": float(close.float().mean().item()) if close.numel() else 1.0,
        "passed": bool(close.all().item()),
    }


def run_layer(
    checkpoint: Path,
    config: Any,
    modeling: Any,
    layer_index: int,
    hidden: torch.Tensor,
    block_residual: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
    layer, layer_weights = load_layer(checkpoint, config, modeling, layer_index)
    batch, sequence, width = hidden.shape
    if sequence != 4:
        raise RuntimeError(f"layer-family decode contract requires four tokens, got {sequence}")

    selected_input = modeling._apply_attn_res(
        hidden.reshape(-1, width), block_residual,
        layer.self_attention_res_proj, layer.self_attention_res_norm,
    ).reshape_as(hidden)
    input_norm = layer.input_layernorm(selected_input)
    cache = modeling.KimiDynamicCache(config)
    attention = attention_forward(layer, input_norm, cache, 0)
    prefix_after_attention = hidden + attention
    selected_mlp = modeling._apply_attn_res(
        prefix_after_attention.reshape(-1, width), block_residual,
        layer.mlp_res_proj, layer.mlp_res_norm,
    ).reshape_as(hidden)
    moe_input = layer.post_attention_layernorm(selected_mlp)
    route = canonical_route(
        moe_input,
        layer.block_sparse_moe.gate.weight,
        layer.block_sparse_moe.gate.e_score_correction_bias,
        RouteConfig(top_k=16),
    )
    selected = sorted(set(route["topk_ids"].flatten().tolist()))
    global_to_local = {expert: local for local, expert in enumerate(selected)}
    local_ids = torch.tensor(
        [[global_to_local[int(expert)] for expert in row] for row in route["topk_ids"].reshape(-1, 16)],
        dtype=torch.int64,
    ).reshape_as(route["topk_ids"]).cuda()
    packed, dense = load_selected_moe(checkpoint, layer_index, selected)
    moe = selected_moe(moe_input, route["topk_weights"], local_ids, packed, dense)
    output = prefix_after_attention + moe["output"]
    tensors: dict[str, torch.Tensor] = {
        "input": hidden,
        "block_residual": block_residual,
        "selected_input": selected_input,
        "input_norm": input_norm,
        "attention_output": attention,
        "prefix_after_attention": prefix_after_attention,
        "selected_mlp": selected_mlp,
        "moe_input": moe_input,
        "route.global_ids": route["topk_ids"],
        "route.local_ids": local_ids,
        "route.weights": route["topk_weights"],
        "output": output,
        **{f"moe.{name}": value for name, value in moe.items()},
        **{f"weights.layer.{name}": value for name, value in layer_weights.items()},
        **{f"weights.selected.{name}": value for name, value in packed.items()},
        **{f"weights.dense.{name}": value for name, value in dense.items()},
        **snapshot_cache(cache, layer_index, layer.is_linear_attn, "cache"),
    }

    warm_hidden = hidden[:, :-1]
    warm_blocks = block_residual[:-1]
    warm_selected = modeling._apply_attn_res(
        warm_hidden.reshape(-1, width), warm_blocks,
        layer.self_attention_res_proj, layer.self_attention_res_norm,
    ).reshape_as(warm_hidden)
    warm_norm = layer.input_layernorm(warm_selected)
    decode_cache = modeling.KimiDynamicCache(config)
    _ = attention_forward(layer, warm_norm, decode_cache, 0)
    decode_cache_in = snapshot_cache(
        decode_cache, layer_index, layer.is_linear_attn, "decode.cache_in"
    )

    decode_hidden = hidden[:, -1:]
    decode_blocks = block_residual[-1:]
    decode_selected_input = modeling._apply_attn_res(
        decode_hidden.reshape(-1, width), decode_blocks,
        layer.self_attention_res_proj, layer.self_attention_res_norm,
    ).reshape_as(decode_hidden)
    decode_input_norm = layer.input_layernorm(decode_selected_input)
    decode_attention = attention_forward(layer, decode_input_norm, decode_cache, sequence - 1)
    decode_prefix = decode_hidden + decode_attention
    decode_selected_mlp = modeling._apply_attn_res(
        decode_prefix.reshape(-1, width), decode_blocks,
        layer.mlp_res_proj, layer.mlp_res_norm,
    ).reshape_as(decode_hidden)
    decode_moe_input = layer.post_attention_layernorm(decode_selected_mlp)
    decode_route = canonical_route(
        decode_moe_input,
        layer.block_sparse_moe.gate.weight,
        layer.block_sparse_moe.gate.e_score_correction_bias,
        RouteConfig(top_k=16),
    )
    missing_decode_experts = sorted(set(decode_route["topk_ids"].flatten().tolist()) - set(selected))
    if missing_decode_experts:
        raise RuntimeError(
            f"layer-{layer_index} decode route escaped prefill compact map: {missing_decode_experts}"
        )
    decode_local_ids = torch.tensor(
        [
            [global_to_local[int(expert)] for expert in row]
            for row in decode_route["topk_ids"].reshape(-1, 16)
        ],
        dtype=torch.int64,
    ).reshape_as(decode_route["topk_ids"]).cuda()
    decode_moe = selected_moe(
        decode_moe_input,
        decode_route["topk_weights"],
        decode_local_ids,
        packed,
        dense,
    )
    decode_output = decode_prefix + decode_moe["output"]
    decode_cache_out = snapshot_cache(
        decode_cache, layer_index, layer.is_linear_attn, "decode.cache_out"
    )
    tensors.update({
        "decode.input": decode_hidden,
        "decode.block_residual": decode_blocks,
        "decode.selected_input": decode_selected_input,
        "decode.input_norm": decode_input_norm,
        "decode.attention_output": decode_attention,
        "decode.prefix_after_attention": decode_prefix,
        "decode.selected_mlp": decode_selected_mlp,
        "decode.moe_input": decode_moe_input,
        "decode.route.global_ids": decode_route["topk_ids"],
        "decode.route.local_ids": decode_local_ids,
        "decode.route.weights": decode_route["topk_weights"],
        "decode.output": decode_output,
        **{f"decode.moe.{name}": value for name, value in decode_moe.items()},
        **decode_cache_in,
        **decode_cache_out,
    })

    prefill_last_ids = route["topk_ids"][:, -1:]
    decode_ids = decode_route["topk_ids"]
    prefill_sorted, prefill_order = torch.sort(prefill_last_ids, dim=-1)
    decode_sorted, decode_order = torch.sort(decode_ids, dim=-1)
    route_sets_match = torch.equal(decode_sorted, prefill_sorted)
    prefill_set = set(prefill_last_ids.flatten().tolist())
    decode_set = set(decode_ids.flatten().tolist())
    route_comparison: dict[str, Any] = {
        "sets_match": route_sets_match,
        "overlap_count": len(prefill_set & decode_set),
        "union_count": len(prefill_set | decode_set),
        "prefill_only": sorted(prefill_set - decode_set),
        "decode_only": sorted(decode_set - prefill_set),
    }
    comparisons: dict[str, Any] = {
        "attention_output": close_report(
            decode_attention, attention[:, -1:], atol=5e-2, rtol=2e-2
        ),
        "output": close_report(decode_output, output[:, -1:], atol=5e-2, rtol=2e-2),
    }
    if route_sets_match:
        comparisons["route_weights_aligned"] = close_report(
            torch.gather(decode_route["topk_weights"], -1, decode_order),
            torch.gather(route["topk_weights"][:, -1:], -1, prefill_order),
            atol=2e-4,
            rtol=2e-4,
        )
    full_cache = snapshot_cache(cache, layer_index, layer.is_linear_attn, "full")
    for name, decode_value in decode_cache_out.items():
        suffix = name.removeprefix("decode.cache_out.")
        expected_value = full_cache[f"full.{suffix}"]
        atol = 5e-3 if suffix == "recurrent" else 5e-2
        comparisons[f"cache.{suffix}"] = close_report(
            decode_value, expected_value, atol=atol, rtol=2e-2
        )
    if not all(report["passed"] for report in comparisons.values()):
        raise RuntimeError(
            f"layer-{layer_index} prefill/decode reference mismatch: "
            f"routes={route_comparison} comparisons={comparisons}"
        )

    return output, tensors, {
        "layer": layer_index,
        "attention": "kda" if layer.is_linear_attn else "mla",
        "selected_global_experts": selected,
        "selected_expert_count": len(selected),
        "route_count": int(route["topk_ids"].numel()),
        "compact_map_scope": "fixture and isolated harness only",
        "decode": {
            "warm_tokens": sequence - 1,
            "decode_tokens": 1,
            "route_count": int(decode_route["topk_ids"].numel()),
            "route_comparison": route_comparison,
            "comparisons": comparisons,
        },
    }


@torch.inference_mode()
def build(
    checkpoint: Path,
    configuration: Any,
    modeling: Any,
    prefix_tensors: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    config_data = json.loads((checkpoint / "config.json").read_text())["text_config"]
    config = configuration.KimiLinearConfig(**config_data)
    config._attn_implementation = "eager"
    hidden = prefix_tensors["prefix.layer0.out"].cuda()
    block_residual = prefix_tensors["prefix.layer0.block_residual.out"].cuda()
    tensors: dict[str, torch.Tensor] = {}
    details: dict[str, Any] = {}
    for layer_index in (1, 2, 3):
        hidden, layer_tensors, layer_details = run_layer(
            checkpoint, config, modeling, layer_index, hidden, block_residual
        )
        tensors.update({
            f"layer{layer_index}.{name}": value.detach().contiguous().cpu()
            for name, value in layer_tensors.items()
        })
        details[str(layer_index)] = layer_details
        torch.cuda.empty_cache()
    return tensors, details


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    checkpoint = args.checkpoint_root.resolve()
    if checkpoint != DEFAULT_CHECKPOINT.resolve():
        raise RuntimeError(f"checkpoint must be the approved local directory: {DEFAULT_CHECKPOINT}")
    deterministic_setup()
    configuration, modeling = import_official(checkpoint)
    prefix_tensors, prefix_source = load_prefix_source(PREFIX_FIXTURE)
    first, first_timing = cuda_timed(lambda: build(checkpoint, configuration, modeling, prefix_tensors))
    second, repeat_timing = cuda_timed(lambda: build(checkpoint, configuration, modeling, prefix_tensors))
    first_tensors, details = first
    second_tensors, second_details = second
    _assert_stable(first_tensors, second_tensors)
    if details != second_details:
        raise RuntimeError("layer-family metadata changed across repeat runs")
    manifest = _save_fixture(
        args.output_dir.resolve(),
        "layer-family-reference",
        first_tensors,
        {
            "mode": "sequential_real_weight_layers_1_2_3_selected_experts",
            "layers": details,
            "prefix_fixture_semantic_sha256": prefix_source["tensor_semantic_sha256"],
            "prefix_fixture_source": prefix_source,
            "tensor_semantic_sha256": semantic_sha256(first_tensors),
            "timing": {"cold_or_first": first_timing, "repeat": repeat_timing},
            "checkpoint": {name: digest for name, digest in SHARDS.values()},
            "numeric_hashes_stable": True,
        },
    )
    summary = {
        "schema_version": 1,
        "moonshot_revision": MOONSHOT_REVISION,
        "device": torch.cuda.get_device_name(0),
        "fixture": manifest["fixture"],
        "manifest": "layer-family-reference.json",
        "safetensors": manifest["tensor_file"],
        "sha256": manifest["tensor_file_sha256"],
        "semantic_sha256": manifest["tensor_semantic_sha256"],
        "prefix_fixture_source": prefix_source,
        "layers": details,
    }
    (args.output_dir.resolve() / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
