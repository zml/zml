#!/usr/bin/env python3
"""Export the one-layer Kimi K3 text-prefix oracle from local S2 weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as torch_functional
from safetensors import safe_open

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
    tensor_bytes,
)


ROOT = Path("/dev/shm/kimi-k3")
DEFAULT_OUTPUT = ROOT / "artifacts/fixtures/milestone-9"
SHARD94_NAME = "model-00094-of-000096.safetensors"
SHARD94_SHA256 = "ad66e1cb96b86963e63d6a0a466b6a407b13c9815cb480fe612480cc6bb3b6e1"
TOKEN_IDS = (1, 42, 32000, 160000)


def semantic_sha256(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(value.dtype).removeprefix("torch.").encode())
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(b"\0")
        digest.update(tensor_bytes(value))
    return digest.hexdigest()


def run_prefix_once(
    config: Any,
    layer: Any,
    modeling: Any,
    weights: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    token_ids = torch.tensor([TOKEN_IDS], device="cuda", dtype=torch.int64)

    def forward() -> dict[str, torch.Tensor]:
        embedding = torch_functional.embedding(token_ids, weights["embedding"])
        cache = modeling.KimiDynamicCache(config)
        block = embedding.new_zeros((len(TOKEN_IDS), 0, embedding.shape[-1]))
        layer_output, block_residual = layer(
            embedding,
            past_key_values=cache,
            use_cache=True,
            block_residual=block,
        )

        candidates = torch.cat(
            (block_residual, layer_output.reshape(-1, embedding.shape[-1]).unsqueeze(1)),
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
        selector_scores = (normalized * score_weight).sum(-1)
        selector_weights = selector_scores.softmax(-1)
        selected = torch.matmul(
            selector_weights.unsqueeze(1), candidates_float
        ).squeeze(1).to(embedding.dtype).reshape_as(embedding)

        final_float = selected.float()
        final_norm = weights["final_norm"] * (
            final_float
            * torch.rsqrt(final_float.pow(2).mean(-1, keepdim=True) + 1e-5)
        ).to(selected.dtype)
        logits = torch_functional.linear(final_norm, weights["lm_head"])
        greedy = logits[:, -1].argmax(-1)
        return {
            "prefix.token_ids": token_ids,
            "prefix.embedding.out": embedding,
            "prefix.layer0.out": layer_output,
            "prefix.layer0.block_residual.out": block_residual,
            "prefix.layer0.cache.conv_state.0.out": cache.conv_states[0][0],
            "prefix.layer0.cache.conv_state.1.out": cache.conv_states[0][1],
            "prefix.layer0.cache.conv_state.2.out": cache.conv_states[0][2],
            "prefix.layer0.cache.recurrent_state.out": cache.recurrent_states[0],
            "prefix.output_attn_res.candidates": candidates,
            "prefix.output_attn_res.scores": selector_scores,
            "prefix.output_attn_res.weights": selector_weights,
            "prefix.output_attn_res.out": selected,
            "prefix.final_norm.out": final_norm,
            "prefix.logits": logits,
            "prefix.greedy_token": greedy,
        }

    return cuda_timed(forward)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    checkpoint = args.checkpoint_root.resolve()
    if checkpoint != DEFAULT_CHECKPOINT.resolve():
        raise RuntimeError(f"checkpoint must be the approved local directory: {DEFAULT_CHECKPOINT}")

    deterministic_setup()
    configuration, modeling = import_official(checkpoint)
    config, layer, layer_load = _load_layer0(checkpoint, configuration, modeling)
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
        weights = {name: tensors.get_tensor(key) for name, key in names.items()}

    first, first_timing = run_prefix_once(config, layer, modeling, weights)
    second, repeat_timing = run_prefix_once(config, layer, modeling, weights)
    _assert_stable(first, second)
    manifest = _save_fixture(
        args.output_dir.resolve(),
        "s2-layer0-prefix-len4",
        first,
        {
            "tier": "S2",
            "mode": "embedding_to_one_layer_diagnostic_logits",
            "token_ids": list(TOKEN_IDS),
            "layer_selection": [0],
            "layer_stop": 1,
            "repeat_runs": 2,
            "numeric_hashes_stable": True,
            "tensor_semantic_sha256": semantic_sha256(first),
            "greedy_token": int(first["prefix.greedy_token"].item()),
            "checkpoint": {
                "model-00001-of-000096.safetensors": SHARD1_SHA256,
                SHARD94_NAME: SHARD94_SHA256,
            },
            "load": layer_load,
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
        "greedy_token": int(first["prefix.greedy_token"].item()),
    }
    (args.output_dir.resolve() / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
