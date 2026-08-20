#!/usr/bin/env python3
"""Export real-weight expanded Gated NoPE MLA fixtures for Milestone 12."""

from __future__ import annotations

import argparse
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
    import_official,
    sha256_file,
    synthetic_hidden,
)


ROOT = Path("/dev/shm/kimi-k3")
OUTPUT = ROOT / "artifacts/fixtures/milestone-12"
SHARD = "model-00004-of-000096.safetensors"
SHARD_SHA256 = "5955fd8feda89b1af8400c25e885e7177d47edff155f54b318beb8dd1cec5c05"
LAYER = 3
PREFIX = f"language_model.model.layers.{LAYER}.self_attn."
LENGTHS = (1, 4, 8, 16)


def rms_norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    dtype = value.dtype
    value_f32 = value.float()
    normalized = value_f32 * torch.rsqrt(value_f32.pow(2).mean(-1, keepdim=True) + 1e-5)
    return weight * normalized.to(dtype)


def causal_mask(query_length: int, key_length: int, past_length: int) -> torch.Tensor:
    query = torch.arange(query_length, device="cuda").reshape(query_length, 1) + past_length
    key = torch.arange(key_length, device="cuda").reshape(1, key_length)
    allowed = key <= query
    return torch.where(allowed, 0.0, -torch.inf).to(torch.bfloat16).reshape(
        1, 1, query_length, key_length
    )


def manual_forward(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
    past_key: torch.Tensor | None = None,
    past_value: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    batch, sequence, _ = hidden.shape
    past_length = 0 if past_key is None else past_key.shape[-2]
    q_a = torch_functional.linear(hidden, weights["q_a_proj"])
    q_norm = rms_norm(q_a, weights["q_a_layernorm"])
    q_b = torch_functional.linear(q_norm, weights["q_b_proj"])
    q_heads = q_b.view(batch, sequence, 96, 192).transpose(1, 2)
    q_pass, q_extra = torch.split(q_heads, [128, 64], dim=-1)

    kv_a = torch_functional.linear(hidden, weights["kv_a_proj_with_mqa"])
    compressed_kv, k_extra = torch.split(kv_a, [512, 64], dim=-1)
    kv_norm = rms_norm(compressed_kv, weights["kv_a_layernorm"])
    kv_b = torch_functional.linear(kv_norm, weights["kv_b_proj"])
    kv_heads = kv_b.view(batch, sequence, 96, 256).transpose(1, 2)
    k_pass, value_new = torch.split(kv_heads, [128, 128], dim=-1)
    k_extra_heads = k_extra.view(batch, 1, sequence, 64).expand(batch, 96, sequence, 64)
    query = torch.cat((q_pass, q_extra), dim=-1)
    key_new = torch.cat((k_pass, k_extra_heads), dim=-1)
    key = key_new if past_key is None else torch.cat((past_key, key_new), dim=-2)
    value = value_new if past_value is None else torch.cat((past_value, value_new), dim=-2)

    scores = torch.einsum("bhqd,bhkd->bhqk", query, key) * (192.0 ** -0.5)
    mask = causal_mask(sequence, key.shape[-2], past_length)
    masked_scores = scores + mask
    probabilities = torch_functional.softmax(masked_scores, dim=-1, dtype=torch.float32).to(query.dtype)
    aggregation = torch.einsum("bhqk,bhkd->bhqd", probabilities, value)
    flattened = aggregation.transpose(1, 2).contiguous().reshape(batch, sequence, 96 * 128)
    gate_logits = torch_functional.linear(hidden, weights["g_proj"])
    gate = gate_logits.sigmoid()
    gated = flattened * gate
    output = torch_functional.linear(gated, weights["o_proj"])
    return {
        "input": hidden,
        "q_a": q_a,
        "q_norm": q_norm,
        "q_b": q_b,
        "q_pass": q_pass,
        "q_extra": q_extra,
        "kv_a": kv_a,
        "compressed_kv": compressed_kv,
        "k_extra": k_extra,
        "kv_norm": kv_norm,
        "kv_b": kv_b,
        "k_pass": k_pass,
        "value_new": value_new,
        "query": query,
        "key_new": key_new,
        "cache_key": key,
        "cache_value": value,
        "scores": scores,
        "causal_mask": mask,
        "masked_scores": masked_scores,
        "probabilities": probabilities,
        "aggregation": aggregation,
        "flattened": flattened,
        "gate_logits": gate_logits,
        "gate": gate,
        "gated": gated,
        "output": output,
    }


def load_attention(checkpoint: Path) -> tuple[Any, Any, dict[str, torch.Tensor], Any]:
    configuration, modeling = import_official(checkpoint)
    config_data = json.loads((checkpoint / "config.json").read_text())["text_config"]
    config = configuration.KimiLinearConfig(**config_data)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        attention = modeling.KimiMLAAttention(config, LAYER)
    suffixes = (
        "q_a_proj", "q_a_layernorm", "q_b_proj", "kv_a_proj_with_mqa",
        "kv_a_layernorm", "kv_b_proj", "g_proj", "o_proj",
    )
    shard = checkpoint / SHARD
    if sha256_file(shard) != SHARD_SHA256:
        raise RuntimeError(f"local shard-4 hash mismatch: {shard}")
    state: dict[str, torch.Tensor] = {}
    weights: dict[str, torch.Tensor] = {}
    with safe_open(shard, framework="pt", device="cuda:0") as tensors:
        for suffix in suffixes:
            parameter = "weight"
            key = f"{PREFIX}{suffix}.{parameter}"
            value = tensors.get_tensor(key)
            state[f"{suffix}.{parameter}"] = value
            weights[suffix] = value
    missing, unexpected = attention.load_state_dict(state, strict=True, assign=True)
    if missing or unexpected:
        raise RuntimeError(f"isolated MLA load mismatch: missing={missing}, unexpected={unexpected}")
    attention.eval()
    if {parameter.device.type for parameter in attention.parameters()} != {"cuda"}:
        raise RuntimeError("isolated MLA attention is not entirely on NVIDIA CUDA")
    return config, attention, weights, modeling


def official_forward(
    config: Any,
    attention: Any,
    modeling: Any,
    hidden: torch.Tensor,
    cache: Any | None = None,
) -> torch.Tensor:
    past_length = 0 if cache is None else cache.get_seq_length(LAYER)
    mask = causal_mask(hidden.shape[1], past_length + hidden.shape[1], past_length)
    return attention(hidden, attention_mask=mask, past_key_values=cache)


def assert_official_close(label: str, manual: torch.Tensor, official: torch.Tensor) -> None:
    if not torch.allclose(manual.float(), official.float(), atol=2e-2, rtol=2e-2):
        delta = (manual.float() - official.float()).abs()
        raise RuntimeError(f"official/manual MLA mismatch {label}: max_abs={delta.max().item()}")


def run_all(config: Any, attention: Any, modeling: Any, weights: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    tensors: dict[str, torch.Tensor] = {}
    timings: dict[str, Any] = {}
    with torch.inference_mode():
        for length in LENGTHS:
            hidden = synthetic_hidden(length)
            result, timing = cuda_timed(lambda hidden=hidden: manual_forward(hidden, weights))
            official, official_timing = cuda_timed(
                lambda hidden=hidden: official_forward(config, attention, modeling, hidden)
            )
            assert_official_close(f"len{length}", result["output"], official)
            for name, value in result.items():
                tensors[f"len{length}.{name}"] = value
            tensors[f"len{length}.official_output"] = official
            timings[f"len{length}"] = {"manual": timing, "official": official_timing}

        cache = modeling.KimiDynamicCache(config)
        prefill_hidden = synthetic_hidden(4)
        prefill = manual_forward(prefill_hidden, weights)
        official_prefill = official_forward(config, attention, modeling, prefill_hidden, cache)
        assert_official_close("continuation.prefill", prefill["output"], official_prefill)
        decode_hidden = synthetic_hidden(5)[:, 4:5]
        decode, decode_timing = cuda_timed(
            lambda: manual_forward(
                decode_hidden,
                weights,
                prefill["cache_key"],
                prefill["cache_value"],
            )
        )
        official_decode, official_decode_timing = cuda_timed(
            lambda: official_forward(config, attention, modeling, decode_hidden, cache)
        )
        assert_official_close("continuation.decode", decode["output"], official_decode)
        for name, value in decode.items():
            tensors[f"decode.{name}"] = value
        tensors["decode.past_key"] = prefill["cache_key"]
        tensors["decode.past_value"] = prefill["cache_value"]
        tensors["decode.official_output"] = official_decode
        timings["decode_after_4"] = {"manual": decode_timing, "official": official_decode_timing}
    return tensors, timings


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
    config, attention, weights, modeling = load_attention(checkpoint)
    first, first_timing = cuda_timed(lambda: run_all(config, attention, modeling, weights))
    second, repeat_timing = cuda_timed(lambda: run_all(config, attention, modeling, weights))
    first_tensors, detailed_timing = first
    second_tensors, _ = second
    _assert_stable(first_tensors, second_tensors)
    tensors = {f"weights.{name}": value for name, value in weights.items()} | first_tensors
    manifest = _save_fixture(
        args.output_dir.resolve(),
        "expanded-mla-reference",
        tensors,
        {
            "mode": "isolated_layer3_expanded_gated_nope_mla",
            "layer": LAYER,
            "lengths": list(LENGTHS),
            "decode_past_length": 4,
            "boundary_count_per_case": 28,
            "tensor_semantic_sha256": semantic_sha256(tensors),
            "checkpoint": {SHARD: SHARD_SHA256},
            "timing": {
                "whole_first": first_timing,
                "whole_repeat": repeat_timing,
                "cases": detailed_timing,
            },
            "peak_memory": {
                "cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "host_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            },
            "official_final_tolerance": {"atol": 0.02, "rtol": 0.02},
            "cache_kind": "expanded per-head K/V correctness oracle",
        },
    )
    summary = {
        "schema_version": 1,
        "moonshot_revision": MOONSHOT_REVISION,
        "device": torch.cuda.get_device_name(0),
        "fixture": manifest["fixture"],
        "manifest": "expanded-mla-reference.json",
        "safetensors": manifest["tensor_file"],
        "sha256": manifest["tensor_file_sha256"],
        "semantic_sha256": manifest["tensor_semantic_sha256"],
        "lengths": list(LENGTHS),
        "decode_past_length": 4,
    }
    (args.output_dir.resolve() / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    # KIMI_K3_TEMP_REMOVE_M20: detailed activation inventory and synchronized
    # oracle timings are MLA bring-up diagnostics removed during cleanup.
    if args.debug:
        print("[kimi-k3-debug] MLA timing", json.dumps(manifest["timing"], sort_keys=True))
        print("[kimi-k3-debug] MLA peak memory", manifest["peak_memory"])
        for name, value in sorted(first_tensors.items()):
            print(f"[kimi-k3-debug] {name} shape={tuple(value.shape)} dtype={value.dtype}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
