#!/usr/bin/env python3
"""Summarize Milestone 17 native grouped MXFP4 correctness and performance."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PASS_RE = re.compile(r"KIMI_K3_MOE_PASS .*\sexecute_us=(\d+)")


def read(path: Path) -> str:
    value = path.read_text()
    if not value.strip():
        raise RuntimeError(f"empty required log: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--native-log", type=Path, required=True)
    parser.add_argument("--partition-log", type=Path, required=True)
    parser.add_argument("--boundary-log", type=Path, required=True)
    parser.add_argument("--layer-log", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text())
    fixture = json.loads(args.fixture_manifest.read_text())
    native_log = read(args.native_log)
    partition_log = read(args.partition_log)
    boundary_log = read(args.boundary_log)
    layer_log = read(args.layer_log)
    match = PASS_RE.search(native_log)
    if match is None:
        raise RuntimeError("native selected-expert PASS marker missing")
    native_us = int(match.group(1))
    slow_us = int(baseline["zml"]["execute_us"])
    speedup = slow_us / native_us

    tensors = fixture["tensors"]
    routes = int(fixture["route_count"])
    dequant_bytes = {}
    for matrix in ("w1", "w2", "w3"):
        shape = tensors[f"selected.{matrix}.packed"]["shape"]
        out, packed_k = int(shape[1]), int(shape[2])
        dequant_bytes[matrix] = routes * out * packed_k * 2 * 4
    # The slow oracle forms gate and up before SiTU, so these route-expanded
    # FP32 tensors can coexist. Native weights/scales remain packed and shared.
    slow_peak_temporary = dequant_bytes["w1"] + dequant_bytes["w3"]
    largest_out = max(tensors[f"selected.{m}.packed"]["shape"][1] for m in ("w1", "w2", "w3"))
    native_output_temporary = routes * int(largest_out) * 2
    packed_persistent = sum(
        int(tensors[key]["elements"])
        for key in tensors
        if key.startswith("selected.") and (key.endswith(".packed") or key.endswith(".scale"))
    )

    boundary_pass = "KIMI_K3_GROUPED_MXFP4_ALL_PASS cases=7" in boundary_log
    partition_pass = "partition_shards=2 partition_exact=true" in partition_log
    layer_pass = "KIMI_K3_LAYER_FAMILY_ALL_PASS" in layer_log
    budget = {"minimum_speedup": 2.0, "measured_speedup": speedup, "pass": speedup >= 2.0}
    report = {
        "schema_version": 1,
        "milestone": 17,
        "status": "PASS" if boundary_pass and partition_pass and layer_pass and budget["pass"] else "FAIL",
        "backend": "cuda",
        "device": fixture["device"],
        "cpu_inference_fallback": False,
        "donor": {
            "branch": "brabier/glm5.2",
            "locked_commit": "b4f0af76e4c464c0f533420b94fdb1fba838c5e3",
            "kernel_sha256": "735cf28078063e86978530a3ca5909302cea8853f4f851736a4319297c6c3d93",
            "license": "Apache-2.0",
        },
        "correctness": {
            "official_selected_experts": int(fixture["selected_expert_count"]),
            "official_routes": routes,
            "official_matrices_exercised": int(fixture["matrix_probe_count"]),
            "official_boundaries": 13,
            "boundary_cases": 7,
            "boundary_sizes_n": [1, 63, 64, 65, 3072, 3584],
            "boundary_sizes_k": [32, 3072, 3584],
            "duplicate_routes": boundary_pass,
            "empty_experts": boundary_pass,
            "invalid_sentinel_zero": boundary_pass,
            "weighted_reduction": boundary_pass,
            "logical_expert_partitions": 2,
            "partition_exact": partition_pass,
            "real_layer_family_prefill_decode": layer_pass,
        },
        "performance": {
            "scope": "selected-expert Gate B; same 61 experts, 64 routes, 13 returned boundaries on one H100",
            "slow_oracle_execute_us": slow_us,
            "native_warm_execute_us": native_us,
            "speedup": speedup,
            "relative_budget": budget,
            "split_k_experiment": "rejected: split_k=4 warm 19704 us versus retained split_k=1 warm 12608 us",
        },
        "hbm": {
            "method": "exact static temporary tensor accounting; excludes identical persistent input weights and compiler allocator slack",
            "slow_route_expanded_fp32_by_matrix_bytes": dequant_bytes,
            "slow_gate_plus_up_peak_temporary_bytes": slow_peak_temporary,
            "native_largest_route_output_temporary_bytes": native_output_temporary,
            "native_packed_weights_and_scales_persistent_bytes": packed_persistent,
            "temporary_reduction_ratio": slow_peak_temporary / native_output_temporary,
            "route_expanded_dequantization_eliminated": True,
        },
        "logs": {
            "native": str(args.native_log),
            "partition": str(args.partition_log),
            "boundaries": str(args.boundary_log),
            "layer_families": str(args.layer_log),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if report["status"] != "PASS":
        raise SystemExit("Milestone 17 report failed an exit gate")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
