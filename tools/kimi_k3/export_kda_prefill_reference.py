#!/usr/bin/env python3
"""Export sequential KDA prefill, split, decode, and continuation fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.numpy import save_file

from export_kda_decode_reference import (
    CHECKPOINT,
    CONV_SIZE,
    EPS,
    LOWER_BOUND,
    compare,
    deterministic_fill,
    manual_step,
    record,
    semantic_sha256,
    sha256_file,
)
from export_reference import deterministic_setup, import_official


ROOT = Path("/ephemeral/kimi-k3")
OUTPUT = ROOT / "artifacts/fixtures/milestone-8"
SEED = 20260822
LENGTHS = (1, 4, 8, 16)
HIDDEN = 40
HEADS = 2
HEAD_DIM = 16


def make_layer(configuration: Any, modeling: Any) -> tuple[Any, torch.nn.Module]:
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
    return config, layer


def hidden_values(length: int) -> torch.Tensor:
    positions = torch.arange(length * HIDDEN, device="cuda", dtype=torch.float32).reshape(1, length, HIDDEN)
    return torch.sin(positions * 0.137 + 0.2) + 0.3 * torch.cos(positions * 0.043 + 0.7)


def zero_state() -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    caches = tuple(
        torch.zeros(1, HEADS * HEAD_DIM, CONV_SIZE, device="cuda", dtype=torch.float32)
        for _ in range(3)
    )
    state = torch.zeros(1, HEADS, HEAD_DIM, HEAD_DIM, device="cuda", dtype=torch.float32)
    return caches, state


def manual_sequence(
    layer: torch.nn.Module,
    hidden: torch.Tensor,
    caches: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    state: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, list[dict[str, torch.Tensor]]]:
    outputs = []
    steps = []
    for index in range(hidden.shape[1]):
        values, caches, state = manual_step(layer, hidden[:, index, :], caches, state)
        outputs.append(values["projection_output"][:, None, :])
        steps.append(values)
    return torch.cat(outputs, dim=1), caches, state, steps


def official_sequence(
    modeling: Any,
    config: Any,
    layer: torch.nn.Module,
    hidden: torch.Tensor,
    chunks: list[tuple[int, int]],
    initial_caches: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    cache = modeling.KimiDynamicCache(config)
    if initial_caches is not None:
        cache.conv_states[0] = tuple(value.clone() for value in initial_caches)
        cache.recurrent_states[0] = initial_state.clone()
    outputs = []
    with torch.inference_mode():
        for start, end in chunks:
            outputs.append(layer(hidden[:, start:end, :], cache_params=cache))
    return (
        torch.cat(outputs, dim=1),
        tuple(value.clone() for value in cache.conv_states[0]),
        cache.recurrent_states[0].clone(),
    )


def add_check(checks: dict[str, Any], name: str, actual: torch.Tensor, expected: torch.Tensor, tolerance: float) -> None:
    result = compare(actual, expected, atol=tolerance)
    checks[name] = result
    if not result["passed"]:
        raise RuntimeError(f"KDA prefill reference mismatch: {name} {result}")


def add_cache_tensors(
    tensors: dict[str, np.ndarray],
    prefix: str,
    caches: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    state: torch.Tensor,
) -> None:
    for name, value in zip(("q_cache", "k_cache", "v_cache"), caches):
        tensors[f"{prefix}.{name}"] = value.detach().cpu().numpy()
    tensors[f"{prefix}.recurrent_state"] = state.detach().cpu().numpy()


def build_fixture(configuration: Any, modeling: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    config, layer = make_layer(configuration, modeling)
    tensors: dict[str, np.ndarray] = {}
    checks: dict[str, Any] = {}
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

    zero_caches, zero_recurrent = zero_state()
    add_cache_tensors(tensors, "inputs.initial", zero_caches, zero_recurrent)
    saved_prefix_cache = None
    saved_prefix_state = None

    for length in LENGTHS:
        hidden = hidden_values(length)
        manual_out, manual_caches, manual_state, steps = manual_sequence(
            layer,
            hidden,
            tuple(value.clone() for value in zero_caches),
            zero_recurrent.clone(),
        )
        full_out, full_caches, full_state = official_sequence(
            modeling, config, layer, hidden, [(0, length)]
        )
        decode_out, decode_caches, decode_state = official_sequence(
            modeling, config, layer, hidden, [(index, index + 1) for index in range(length)]
        )
        add_check(checks, f"len{length}.chunk_output", full_out, manual_out, 8e-3)
        add_check(checks, f"len{length}.decode_output", decode_out, manual_out, 8e-3)
        for name, actual, expected in zip(("q", "k", "v"), full_caches, manual_caches):
            add_check(checks, f"len{length}.chunk_{name}_cache", actual, expected, 5e-4)
        add_check(checks, f"len{length}.chunk_recurrent", full_state, manual_state, 8e-4)
        for name, actual, expected in zip(("q", "k", "v"), decode_caches, manual_caches):
            add_check(checks, f"len{length}.decode_{name}_cache", actual, expected, 5e-4)
        add_check(checks, f"len{length}.decode_recurrent", decode_state, manual_state, 8e-4)

        base = f"len{length}"
        tensors[f"{base}.input.hidden"] = hidden.cpu().numpy()
        tensors[f"{base}.expected.output"] = manual_out.cpu().numpy()
        add_cache_tensors(tensors, f"{base}.expected", manual_caches, manual_state)
        for token, values in enumerate(steps):
            tensors[f"{base}.token{token}.input.hidden"] = hidden[:, token, :].cpu().numpy()
            tensors[f"{base}.token{token}.expected.output"] = values["projection_output"].cpu().numpy()
            add_cache_tensors(
                tensors,
                f"{base}.token{token}.expected",
                (values["q_cache"], values["k_cache"], values["v_cache"]),
                values["recurrent_state"],
            )

        for split in range(1, length):
            split_out, split_caches, split_state = official_sequence(
                modeling, config, layer, hidden, [(0, split), (split, length)]
            )
            add_check(checks, f"len{length}.split{split}.output", split_out, full_out, 8e-3)
            for name, actual, expected in zip(("q", "k", "v"), split_caches, full_caches):
                add_check(checks, f"len{length}.split{split}.{name}_cache", actual, expected, 5e-4)
            add_check(checks, f"len{length}.split{split}.recurrent", split_state, full_state, 8e-4)
            tensors[f"{base}.split{split}.expected.first_output"] = manual_out[:, :split, :].cpu().numpy()
            tensors[f"{base}.split{split}.expected.second_output"] = manual_out[:, split:, :].cpu().numpy()
            tensors[f"{base}.split{split}.input.first"] = hidden[:, :split, :].cpu().numpy()
            tensors[f"{base}.split{split}.input.second"] = hidden[:, split:, :].cpu().numpy()
            prefix_values = steps[split - 1]
            add_cache_tensors(
                tensors,
                f"{base}.split{split}.expected.intermediate",
                (prefix_values["q_cache"], prefix_values["k_cache"], prefix_values["v_cache"]),
                prefix_values["recurrent_state"],
            )

        if length == 8:
            _, saved_prefix_cache, saved_prefix_state = official_sequence(
                modeling, config, layer, hidden, [(0, 4)]
            )

    assert saved_prefix_cache is not None and saved_prefix_state is not None
    continuation_hidden = hidden_values(8)[:, 4:, :]
    continuation_out, continuation_caches, continuation_state = official_sequence(
        modeling,
        config,
        layer,
        continuation_hidden,
        [(0, continuation_hidden.shape[1])],
        saved_prefix_cache,
        saved_prefix_state,
    )
    manual_cont_out, manual_cont_caches, manual_cont_state, _ = manual_sequence(
        layer,
        continuation_hidden,
        tuple(value.clone() for value in saved_prefix_cache),
        saved_prefix_state.clone(),
    )
    add_check(checks, "continuation.output", continuation_out, manual_cont_out, 8e-3)
    for name, actual, expected in zip(("q", "k", "v"), continuation_caches, manual_cont_caches):
        add_check(checks, f"continuation.{name}_cache", actual, expected, 5e-4)
    add_check(checks, "continuation.recurrent", continuation_state, manual_cont_state, 8e-4)
    tensors["continuation.input.hidden"] = continuation_hidden.cpu().numpy()
    add_cache_tensors(tensors, "continuation.initial", saved_prefix_cache, saved_prefix_state)
    tensors["continuation.expected.output"] = manual_cont_out.cpu().numpy()
    add_cache_tensors(tensors, "continuation.expected", manual_cont_caches, manual_cont_state)
    return tensors, {
        "official_symbol": "KimiDeltaAttention.forward",
        "official_chunk_kernel": "fla.ops.kda.chunk_kda",
        "official_decode_kernel": "fla.ops.kda.fused_recurrent_kda",
        "lengths": list(LENGTHS),
        "split_points": sum(length - 1 for length in LENGTHS),
        "official_checks": checks,
        "state_layout": "batch,head,value,key",
        "conv_cache_layout": "batch,channel,kernel",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    deterministic_setup()
    configuration, modeling = import_official(args.checkpoint)
    with torch.inference_mode():
        tensors, details = build_fixture(configuration, modeling)
    args.output.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output / "kda-prefill-reference.safetensors"
    save_file(tensors, tensor_path, metadata={"schema_version": "1", "milestone": "8"})
    manifest = {
        "schema_version": 1,
        "milestone": 8,
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
    manifest_path = args.output / "kda-prefill-reference.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    # KIMI_K3_TEMP_REMOVE_M20: prefill/split inventory is retained for bring-up
    # and removed during the cleanup milestone.
    if args.debug:
        print(
            f"[kimi-k3-debug] lengths={details['lengths']} split_points={details['split_points']} "
            f"official_checks={len(details['official_checks'])} tensors={len(tensors)}"
        )
    print(json.dumps({"fixture": str(tensor_path), "checks": len(details["official_checks"]), "tensors": len(tensors)}))


if __name__ == "__main__":
    main()
