#!/usr/bin/env python3
"""Derive compact latent-cache split/decode cases from the locked MLA oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from export_layer0_prefix_reference import semantic_sha256
from export_reference import MOONSHOT_REVISION, _save_fixture, sha256_file


ROOT = Path("/ephemeral/kimi-k3")
SOURCE = ROOT / "artifacts/fixtures/milestone-12"
OUTPUT = ROOT / "artifacts/fixtures/milestone-13"
SOURCE_SEMANTIC_SHA256 = "e642fb2ccf9fda74a7cfb013eb330a2c697c8511874d0aae0b0296d312c787a2"


def record_case(
    output: dict[str, torch.Tensor],
    prefix: str,
    source: dict[str, torch.Tensor],
    start: int,
    end: int,
    key_end: int,
) -> None:
    output[f"{prefix}.input"] = source["input"][:, start:end].clone().contiguous()
    output[f"{prefix}.expected.output"] = source["output"][:, start:end].clone().contiguous()
    output[f"{prefix}.expected.probabilities"] = source["probabilities"][..., start:end, :key_end].clone().contiguous()
    output[f"{prefix}.expected.cache.compressed"] = source["kv_norm"][:, :key_end].clone().contiguous()
    output[f"{prefix}.expected.cache.extra_key"] = source["k_extra"][:, :key_end].clone().contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    source_manifest = json.loads((args.source_dir / "expanded-mla-reference.json").read_text())
    if source_manifest["tensor_semantic_sha256"] != SOURCE_SEMANTIC_SHA256:
        raise RuntimeError("Milestone 12 semantic source lock mismatch")
    source_path = args.source_dir / source_manifest["tensor_file"]
    if sha256_file(source_path) != source_manifest["tensor_file_sha256"]:
        raise RuntimeError("Milestone 12 physical source hash mismatch")
    wanted = ("input", "output", "probabilities", "kv_norm", "k_extra")
    source: dict[int, dict[str, torch.Tensor]] = {}
    with safe_open(source_path, framework="pt", device="cpu") as values:
        for length in (1, 4, 8, 16):
            source[length] = {name: values.get_tensor(f"len{length}.{name}") for name in wanted}

    tensors: dict[str, torch.Tensor] = {}
    for length in (1, 4, 8, 16):
        record_case(tensors, f"full.len{length}", source[length], 0, length, length)
    for split in (1, 2, 3):
        record_case(tensors, f"split4.at{split}.first", source[4], 0, split, split)
        record_case(tensors, f"split4.at{split}.second", source[4], split, 4, 4)
    for token in range(4):
        record_case(tensors, f"decode4.token{token}", source[4], token, token + 1, token + 1)

    manifest = _save_fixture(
        args.output_dir.resolve(),
        "mla-latent-cache-cases",
        tensors,
        {
            "mode": "derived_expanded_to_latent_cache_cases",
            "source_semantic_sha256": SOURCE_SEMANTIC_SHA256,
            "tensor_semantic_sha256": semantic_sha256(tensors),
            "full_lengths": [1, 4, 8, 16],
            "split_length": 4,
            "split_points": [1, 2, 3],
            "repeated_decode_steps": 4,
            "case_count": 14,
            "cache_values_per_token": 576,
            "cache_dtype": "bfloat16",
            "numeric_hashes_stable": True,
            "derivation_only_no_inference": True,
        },
    )
    summary = {
        "schema_version": 1,
        "moonshot_revision": MOONSHOT_REVISION,
        "fixture": manifest["fixture"],
        "manifest": "mla-latent-cache-cases.json",
        "safetensors": manifest["tensor_file"],
        "sha256": manifest["tensor_file_sha256"],
        "semantic_sha256": manifest["tensor_semantic_sha256"],
        "cases": manifest["case_count"],
    }
    (args.output_dir.resolve() / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    # KIMI_K3_TEMP_REMOVE_M20: derived split/decode inventory is a cache
    # bring-up diagnostic removed when permanent session tests replace it.
    if args.debug:
        for name, value in sorted(tensors.items()):
            print(f"[kimi-k3-debug] {name} shape={tuple(value.shape)} dtype={value.dtype}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
