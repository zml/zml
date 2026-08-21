#!/usr/bin/env python3
"""Export the four-layer Kimi K3 prefix/head oracle from locked local fixtures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as torch_functional
from safetensors import safe_open

from export_layer_family_reference import (
    attention_forward,
    load_layer,
    load_selected_moe,
    selected_moe,
    snapshot_cache,
)
from export_layer0_prefix_reference import TOKEN_IDS, semantic_sha256
from export_reference import (
    DEFAULT_CHECKPOINT,
    MOONSHOT_REVISION,
    SHARD1_SHA256,
    _assert_stable,
    _load_layer0,
    _save_fixture,
    cuda_timed,
    deterministic_setup,
    import_official,
    sha256_file,
)
from export_router_reference import RouteConfig, canonical_route


ROOT = Path(os.environ.get("KIMI_K3_WORKSPACE", "/dev/shm/kimi-k3"))
OUTPUT = ROOT / "artifacts/fixtures/milestone-15"
LAYER0_FIXTURE = ROOT / "artifacts/fixtures/milestone-9/s2-layer0-prefix-len4.safetensors"
LAYER_FAMILY_FIXTURE = ROOT / "artifacts/fixtures/milestone-14/layer-family-reference.safetensors"
SHARD94_NAME = "model-00094-of-000096.safetensors"
SHARD94_SHA256 = "ad66e1cb96b86963e63d6a0a466b6a407b13c9815cb480fe612480cc6bb3b6e1"
LAYER_FAMILY_FILE_SHA256 = "9e5687de705f823b3b6b765b2af66a4440e77113ef12366852aafd0a694a4e92"
LAYER_FAMILY_SEMANTIC_SHA256 = "847fd46368da4107ade6f6528eabfbb765d1de396008ca5ae82db0dce10f3aea"


def checked_manifest(path: Path, *, file_sha256: str | None = None) -> dict[str, Any]:
    manifest = json.loads(path.with_suffix(".json").read_text())
    actual = sha256_file(path)
    if actual != manifest["tensor_file_sha256"]:
        raise RuntimeError(f"fixture file hash mismatch: {path}")
    if file_sha256 is not None and actual != file_sha256:
        raise RuntimeError(f"fixture lock mismatch: {path}")
    return manifest


def diagnostic_head(
    hidden: torch.Tensor,
    block_residual: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    width = hidden.shape[-1]
    candidates = torch.cat(
        (block_residual, hidden.reshape(-1, width).unsqueeze(1)),
        dim=1,
    )
    candidates_float = candidates.float()
    normalized = candidates_float * torch.rsqrt(
        candidates_float.pow(2).mean(-1, keepdim=True) + 1e-5
    )
    score_weight = (
        weights["output_attn_res_norm"].float()
        * weights["output_attn_res_proj"].squeeze(0).float()
    )
    scores = (normalized * score_weight).sum(-1)
    probabilities = scores.softmax(-1)
    selected = torch.matmul(
        probabilities.unsqueeze(1), candidates_float
    ).squeeze(1).to(hidden.dtype).reshape_as(hidden)
    selected_float = selected.float()
    final_norm = weights["final_norm"] * (
        selected_float
        * torch.rsqrt(selected_float.pow(2).mean(-1, keepdim=True) + 1e-5)
    ).to(selected.dtype)
    logits = torch_functional.linear(final_norm, weights["lm_head"])
    return {
        "output_attn_res.candidates": candidates,
        "output_attn_res.scores": scores,
        "output_attn_res.weights": probabilities,
        "output_attn_res.out": selected,
        "final_norm.out": final_norm,
        "logits": logits,
        "greedy_token": logits[:, -1].argmax(-1),
    }


def cache_tensors(cache: Any, prefix: str) -> dict[str, torch.Tensor]:
    result = {f"{prefix}.cache.recurrent": cache.recurrent_states[0].clone()}
    for index, value in enumerate(cache.conv_states[0]):
        result[f"{prefix}.cache.conv{index}"] = value.clone()
    return result


def continuation_layer(
    checkpoint: Path,
    config: Any,
    modeling: Any,
    layer_index: int,
    warm_hidden: torch.Tensor,
    decode_hidden: torch.Tensor,
    warm_blocks: torch.Tensor,
    decode_blocks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    layer, _ = load_layer(checkpoint, config, modeling, layer_index)
    width = warm_hidden.shape[-1]
    cache = modeling.KimiDynamicCache(config)

    warm_selected_input = modeling._apply_attn_res(
        warm_hidden.reshape(-1, width),
        warm_blocks,
        layer.self_attention_res_proj,
        layer.self_attention_res_norm,
    ).reshape_as(warm_hidden)
    warm_input_norm = layer.input_layernorm(warm_selected_input)
    warm_attention = attention_forward(layer, warm_input_norm, cache, 0)
    warm_prefix = warm_hidden + warm_attention
    warm_selected_mlp = modeling._apply_attn_res(
        warm_prefix.reshape(-1, width),
        warm_blocks,
        layer.mlp_res_proj,
        layer.mlp_res_norm,
    ).reshape_as(warm_hidden)
    warm_moe_input = layer.post_attention_layernorm(warm_selected_mlp)
    warm_route = canonical_route(
        warm_moe_input,
        layer.block_sparse_moe.gate.weight,
        layer.block_sparse_moe.gate.e_score_correction_bias,
        RouteConfig(top_k=16),
    )
    cache_in = snapshot_cache(
        cache, layer_index, layer.is_linear_attn, f"layer{layer_index}.decode.cache_in"
    )

    decode_selected_input = modeling._apply_attn_res(
        decode_hidden.reshape(-1, width),
        decode_blocks,
        layer.self_attention_res_proj,
        layer.self_attention_res_norm,
    ).reshape_as(decode_hidden)
    decode_input_norm = layer.input_layernorm(decode_selected_input)
    decode_attention = attention_forward(layer, decode_input_norm, cache, len(TOKEN_IDS) - 1)
    decode_prefix = decode_hidden + decode_attention
    decode_selected_mlp = modeling._apply_attn_res(
        decode_prefix.reshape(-1, width),
        decode_blocks,
        layer.mlp_res_proj,
        layer.mlp_res_norm,
    ).reshape_as(decode_hidden)
    decode_moe_input = layer.post_attention_layernorm(decode_selected_mlp)
    decode_route = canonical_route(
        decode_moe_input,
        layer.block_sparse_moe.gate.weight,
        layer.block_sparse_moe.gate.e_score_correction_bias,
        RouteConfig(top_k=16),
    )

    selected = sorted(set(warm_route["topk_ids"].flatten().tolist()) | set(
        decode_route["topk_ids"].flatten().tolist()
    ))
    global_to_local = {expert: local for local, expert in enumerate(selected)}

    def local_ids(ids: torch.Tensor) -> torch.Tensor:
        return torch.tensor(
            [
                [global_to_local[int(expert)] for expert in row]
                for row in ids.reshape(-1, 16)
            ],
            dtype=torch.int64,
            device="cuda",
        ).reshape_as(ids)

    packed, dense = load_selected_moe(checkpoint, layer_index, selected)
    warm_local = local_ids(warm_route["topk_ids"])
    decode_local = local_ids(decode_route["topk_ids"])
    warm_moe = selected_moe(
        warm_moe_input, warm_route["topk_weights"], warm_local, packed, dense
    )
    decode_moe = selected_moe(
        decode_moe_input, decode_route["topk_weights"], decode_local, packed, dense
    )
    warm_output = warm_prefix + warm_moe["output"]
    decode_output = decode_prefix + decode_moe["output"]
    cache_out = snapshot_cache(
        cache, layer_index, layer.is_linear_attn, f"layer{layer_index}.decode.cache_out"
    )
    tensors = {
        f"layer{layer_index}.warm.output": warm_output,
        f"layer{layer_index}.decode.input": decode_hidden,
        f"layer{layer_index}.decode.selected_input": decode_selected_input,
        f"layer{layer_index}.decode.input_norm": decode_input_norm,
        f"layer{layer_index}.decode.attention_output": decode_attention,
        f"layer{layer_index}.decode.prefix_after_attention": decode_prefix,
        f"layer{layer_index}.decode.selected_mlp": decode_selected_mlp,
        f"layer{layer_index}.decode.moe_input": decode_moe_input,
        f"layer{layer_index}.decode.route.global_ids": decode_route["topk_ids"],
        f"layer{layer_index}.decode.route.local_ids": decode_local,
        f"layer{layer_index}.decode.route.weights": decode_route["topk_weights"],
        **{
            f"layer{layer_index}.decode.moe.{name}": value
            for name, value in decode_moe.items()
        },
        f"layer{layer_index}.decode.output": decode_output,
        **cache_in,
        **cache_out,
    }
    return warm_output, decode_output, tensors


@torch.inference_mode()
def build(
    checkpoint: Path,
    config: Any,
    layer0: Any,
    modeling: Any,
    embedding_weight: torch.Tensor,
    head_weights: dict[str, torch.Tensor],
    layer0_expected: dict[str, torch.Tensor],
    families: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    token_ids = torch.tensor([TOKEN_IDS], device="cuda", dtype=torch.int64)
    embedding = torch_functional.embedding(token_ids, embedding_weight)

    full_cache = modeling.KimiDynamicCache(config)
    full_blocks = embedding.new_zeros((len(TOKEN_IDS), 0, embedding.shape[-1]))
    full_output, full_blocks = layer0(
        embedding,
        past_key_values=full_cache,
        use_cache=True,
        block_residual=full_blocks,
    )
    if not torch.equal(full_output, layer0_expected["prefix.layer0.out"]):
        raise RuntimeError("layer-0 full output no longer matches locked prefix fixture")
    if not torch.equal(full_blocks, layer0_expected["prefix.layer0.block_residual.out"]):
        raise RuntimeError("layer-0 block source no longer matches locked prefix fixture")
    if not torch.equal(full_output, families["layer1.input"]):
        raise RuntimeError("layer-0 to layer-1 prefill handoff mismatch")

    warm_cache = modeling.KimiDynamicCache(config)
    warm_embedding = embedding[:, :-1]
    warm_blocks = warm_embedding.new_zeros((len(TOKEN_IDS) - 1, 0, embedding.shape[-1]))
    layer0_warm_output, warm_blocks = layer0(
        warm_embedding,
        past_key_values=warm_cache,
        use_cache=True,
        block_residual=warm_blocks,
    )
    warm_snapshot = cache_tensors(warm_cache, "decode.layer0.cache_in")

    decode_embedding = embedding[:, -1:]
    decode_blocks = decode_embedding.new_zeros((1, 0, embedding.shape[-1]))
    layer0_decode_output, decode_blocks = layer0(
        decode_embedding,
        past_key_values=warm_cache,
        use_cache=True,
        block_residual=decode_blocks,
    )
    continuation: dict[str, torch.Tensor] = {}
    warm_output = layer0_warm_output
    decode_output = layer0_decode_output
    for layer_index in (1, 2, 3):
        warm_output, decode_output, layer_tensors = continuation_layer(
            checkpoint,
            config,
            modeling,
            layer_index,
            warm_output,
            decode_output,
            warm_blocks,
            decode_blocks,
        )
        continuation.update(layer_tensors)
        recomposed = torch.cat((warm_output, decode_output), dim=1)
        expected = families[f"layer{layer_index}.output"]
        if not torch.allclose(
            recomposed.float(), expected.float(), atol=5e-2, rtol=2e-2
        ):
            delta = (recomposed.float() - expected.float()).abs().max().item()
            raise RuntimeError(
                f"layer-{layer_index} sequential continuation mismatch: max_abs={delta}"
            )

    prefill_head = diagnostic_head(families["layer3.output"], full_blocks, head_weights)
    decode_head = diagnostic_head(decode_output, decode_blocks, head_weights)
    tensors = {
        "prefix.token_ids": token_ids,
        "prefix.embedding.out": embedding,
        "prefix.layer0.warm.output": layer0_warm_output,
        "prefix.layer0.decode.output": layer0_decode_output,
        "prefix.layer0.decode.block_residual": decode_blocks,
        **warm_snapshot,
        **cache_tensors(warm_cache, "decode.layer0.cache_out"),
        **continuation,
        **{f"prefix.{name}": value for name, value in prefill_head.items()},
        **{f"decode.{name}": value for name, value in decode_head.items()},
    }
    return {name: value.detach().contiguous().cpu() for name, value in tensors.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    checkpoint = args.checkpoint_root.resolve()
    if checkpoint != DEFAULT_CHECKPOINT.resolve():
        raise RuntimeError(f"checkpoint must be the approved local directory: {DEFAULT_CHECKPOINT}")

    deterministic_setup()
    layer0_manifest = checked_manifest(LAYER0_FIXTURE)
    family_manifest = checked_manifest(
        LAYER_FAMILY_FIXTURE,
        file_sha256=LAYER_FAMILY_FILE_SHA256,
    )
    if family_manifest["tensor_semantic_sha256"] != LAYER_FAMILY_SEMANTIC_SHA256:
        raise RuntimeError("layer-family semantic lock mismatch")

    layer0_names = {
        "prefix.layer0.out",
        "prefix.layer0.block_residual.out",
    }
    with safe_open(LAYER0_FIXTURE, framework="pt", device="cuda:0") as source:
        layer0_expected = {name: source.get_tensor(name) for name in layer0_names}
    family_names = {
        "layer1.input",
        "layer1.output",
        "layer1.decode.input",
        "layer1.decode.output",
        "layer2.input",
        "layer2.output",
        "layer2.decode.input",
        "layer2.decode.output",
        "layer3.input",
        "layer3.output",
        "layer3.decode.input",
        "layer3.decode.output",
    }
    with safe_open(LAYER_FAMILY_FIXTURE, framework="pt", device="cuda:0") as source:
        families = {name: source.get_tensor(name) for name in family_names}

    configuration, modeling = import_official(checkpoint)
    config, layer0, layer0_load = _load_layer0(checkpoint, configuration, modeling)
    config._attn_implementation = "eager"
    shard94 = checkpoint / SHARD94_NAME
    if sha256_file(shard94) != SHARD94_SHA256:
        raise RuntimeError(f"local shard-94 hash mismatch: {shard94}")
    names = {
        "embedding": "language_model.model.embed_tokens.weight",
        "output_attn_res_norm": "language_model.model.output_attn_res_norm.weight",
        "output_attn_res_proj": "language_model.model.output_attn_res_proj.weight",
        "final_norm": "language_model.model.norm.weight",
        "lm_head": "language_model.lm_head.weight",
    }
    with safe_open(shard94, framework="pt", device="cuda:0") as tensors:
        loaded = {name: tensors.get_tensor(key) for name, key in names.items()}
    embedding_weight = loaded.pop("embedding")

    run = lambda: build(
        checkpoint,
        config,
        layer0,
        modeling,
        embedding_weight,
        loaded,
        layer0_expected,
        families,
    )
    first, first_timing = cuda_timed(run)
    second, repeat_timing = cuda_timed(run)
    _assert_stable(first, second)
    manifest = _save_fixture(
        args.output_dir.resolve(),
        "s4-prefix4-head-reference",
        first,
        {
            "tier": "S4",
            "mode": "four_layer_prefill_decode_diagnostic_head",
            "token_ids": list(TOKEN_IDS),
            "layer_selection": [0, 1, 2, 3],
            "layer_stop": 4,
            "repeat_runs": 2,
            "numeric_hashes_stable": True,
            "tensor_semantic_sha256": semantic_sha256(first),
            "prefill_greedy_token": int(first["prefix.greedy_token"].item()),
            "decode_greedy_token": int(first["decode.greedy_token"].item()),
            "source_fixtures": {
                "layer0": {
                    "file_sha256": layer0_manifest["tensor_file_sha256"],
                    "semantic_sha256": layer0_manifest["tensor_semantic_sha256"],
                },
                "layer_families": {
                    "file_sha256": family_manifest["tensor_file_sha256"],
                    "semantic_sha256": family_manifest["tensor_semantic_sha256"],
                },
            },
            "checkpoint": {
                "model-00001-of-000096.safetensors": SHARD1_SHA256,
                SHARD94_NAME: SHARD94_SHA256,
            },
            "load": layer0_load,
            "timing": {"cold_or_first": first_timing, "repeat": repeat_timing},
        },
    )
    summary = {
        "schema_version": 1,
        "moonshot_revision": MOONSHOT_REVISION,
        "device": torch.cuda.get_device_name(0),
        "fixture": manifest["fixture"],
        "manifest": f"{manifest['fixture']}.json",
        "safetensors": manifest["tensor_file"],
        "sha256": manifest["tensor_file_sha256"],
        "semantic_sha256": manifest["tensor_semantic_sha256"],
        "prefill_greedy_token": manifest["prefill_greedy_token"],
        "decode_greedy_token": manifest["decode_greedy_token"],
    }
    (args.output_dir.resolve() / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
