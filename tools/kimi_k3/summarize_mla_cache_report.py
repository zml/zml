#!/usr/bin/env python3
"""Create Milestone 13 latent-cache correctness and memory report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FIELDS = re.compile(r"(\w+)=([^ ]+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--metadata-log", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lines = args.log.read_text().splitlines()
    records = [dict(FIELDS.findall(line)) for line in lines if line.startswith("KIMI_K3_MLA_CACHE_PASS ")]
    if len(records) != 8 or not any(line.startswith("KIMI_K3_MLA_CACHE_ALL_PASS ") for line in lines):
        raise SystemExit("incomplete MLA latent-cache CUDA log")
    kinds = [record.get("kind") for record in records]
    if kinds.count("full") != 4 or kinds.count("split") != 3 or kinds.count("repeated_decode") != 1:
        raise SystemExit("MLA latent-cache CUDA case inventory mismatch")
    metadata = [
        dict(FIELDS.findall(line))
        for line in args.metadata_log.read_text().splitlines()
        if line.startswith("KIMI_K3_METADATA_PASS ")
    ]
    if len(metadata) != 1:
        raise SystemExit("missing packed-cache metadata result")
    expected_metadata = {
        "layers": "93",
        "kda_caches": "69",
        "mla_caches": "24",
        "mla_1m_bytes": "27648000000",
        "attnres_persisted": "false",
    }
    if any(metadata[0].get(key) != value for key, value in expected_metadata.items()):
        raise SystemExit("packed-cache metadata mismatch")
    manifest = json.loads(args.manifest.read_text())
    report = {
        "schema_version": 1,
        "milestone": 13,
        "status": "PASS",
        "backend": "cuda",
        "device": manifest["device"],
        "cpu_inference_fallback": False,
        "semantic_fixture_sha256": manifest["tensor_semantic_sha256"],
        "source_mla_semantic_sha256": manifest["source_semantic_sha256"],
        "cuda_cases": records,
        "packed_schedule": metadata[0],
        "memory": {
            "latent_values_per_token_layer": 576,
            "latent_bytes_per_token_layer_bf16": 1152,
            "expanded_bytes_per_token_layer_bf16": 61440,
            "compression_ratio": 61440 / 1152,
            "all_mla_layers_1m_token_bytes": 27_648_000_000,
            "all_kda_layers_persistent_bytes_batch1": 454_459_392,
            "combined_1m_token_bytes_batch1": 28_102_459_392,
        },
        "timing_scope": "correctness differential returning output, probabilities, latent cache, absorbed q, and latent readout",
        "timing_is_production_benchmark": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
