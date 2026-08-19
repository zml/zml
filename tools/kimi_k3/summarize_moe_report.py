#!/usr/bin/env python3
"""Create the Milestone 11 Gate B correctness/timing/memory report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FIELDS = re.compile(r"(\w+)=([^ ]+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = [
        dict(FIELDS.findall(line))
        for line in args.log.read_text().splitlines()
        if line.startswith("KIMI_K3_MOE_PASS ")
    ]
    if len(records) != 1:
        raise SystemExit("incomplete MoE CUDA log")
    record = records[0]
    expected = {"experts": "61", "routes": "64", "matrices": "183", "boundaries": "13"}
    if any(record.get(key) != value for key, value in expected.items()):
        raise SystemExit("MoE CUDA inventory mismatch")
    report = {
        "schema_version": 1,
        "milestone": 11,
        "gate": "B",
        "status": "PASS",
        "backend": "cuda",
        "device": manifest["device"],
        "cpu_inference_fallback": False,
        "semantic_fixture_sha256": manifest["tensor_semantic_sha256"],
        "selected_global_experts": manifest["selected_global_experts"],
        "selected_expert_count": manifest["selected_expert_count"],
        "route_count": manifest["route_count"],
        "matrix_probe_count": manifest["matrix_probe_count"],
        "zml": record,
        "reference_timing": manifest["timing"],
        "peak_memory": manifest["peak_memory"],
        "compact_map_scope": manifest["compact_map_scope"],
        "timing_scope": "correctness-first dequantize-to-FP32/BF16 with 13 returned diagnostic boundaries",
        "timing_is_production_benchmark": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "gate": "B", "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
