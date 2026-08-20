#!/usr/bin/env python3
"""Create the Milestone 14 composed-layer CUDA comparison report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


FIELDS = re.compile(r"(\w+)=([^ ]+)")
EXPECTED = {
    (1, "prefill"): ("kda", 61, 24),
    (1, "decode"): ("kda", 61, 24),
    (2, "prefill"): ("kda", 56, 24),
    (2, "decode"): ("kda", 56, 24),
    (3, "prefill"): ("mla", 53, 22),
    (3, "decode"): ("mla", 53, 22),
}


def summarize(log: str, manifest: dict[str, Any]) -> dict[str, Any]:
    lines = log.splitlines()
    raw = [
        dict(FIELDS.findall(line))
        for line in lines
        if line.startswith("KIMI_K3_LAYER_FAMILY_PASS ")
    ]
    all_pass = [
        dict(FIELDS.findall(line))
        for line in lines
        if line.startswith("KIMI_K3_LAYER_FAMILY_ALL_PASS ")
    ]
    if len(raw) != 6 or len(all_pass) != 1:
        raise ValueError("incomplete layer-family CUDA log")
    if all_pass[0] != {
        "layers": "1,2,3",
        "prefill": "3",
        "decode": "3",
        "backend": "cuda",
        "global_routes": "exact",
    }:
        raise ValueError("invalid layer-family completion marker")

    records: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for record in raw:
        key = (int(record["layer"]), record["mode"])
        if key in seen or key not in EXPECTED:
            raise ValueError(f"invalid layer-family case: {key}")
        seen.add(key)
        attention, experts, boundaries = EXPECTED[key]
        if (
            record.get("attention") != attention
            or int(record.get("experts", -1)) != experts
            or int(record.get("boundaries", -1)) != boundaries
        ):
            raise ValueError(f"layer-family inventory mismatch: {key}")
        numeric = {
            name: int(value)
            for name, value in record.items()
            if name.endswith("_us")
        }
        records.append({
            "layer": key[0],
            "mode": key[1],
            "attention": attention,
            "selected_expert_count": experts,
            "compared_boundaries": boundaries,
            **numeric,
        })
    if seen != set(EXPECTED):
        raise ValueError("missing layer-family CUDA case")

    return {
        "schema_version": 1,
        "milestone": 14,
        "status": "PASS",
        "backend": "cuda",
        "device": manifest["device"],
        "cpu_inference_fallback": False,
        "moonshot_revision": manifest["moonshot_revision"],
        "fixture_file_sha256": manifest["tensor_file_sha256"],
        "semantic_fixture_sha256": manifest["tensor_semantic_sha256"],
        "official_reference_timing": manifest["timing"],
        "cuda_cases": sorted(records, key=lambda item: (item["layer"], item["mode"])),
        "global_route_sets": "exact",
        "route_weight_tolerance": {
            "absolute": 0.002,
            "relative": 0.02,
            "minimum_close_fraction": 1.0,
        },
        "activation_tolerance": {
            "absolute": 0.05,
            "relative": 0.02,
            "minimum_close_fraction": 0.995,
        },
        "timing_scope": "synchronized diagnostic execution returning composed layer, route, expert, residual, and cache boundaries",
        "timing_is_production_benchmark": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.log.read_text(), json.loads(args.manifest.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
