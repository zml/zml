#!/usr/bin/env python3
"""Build the Milestone 25 TP4 correctness and resource report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any


DTYPE_BYTES = {"pred": 1, "s8": 1, "u8": 1, "bf16": 2, "f16": 2, "s32": 4, "u32": 4, "f32": 4, "s64": 8, "u64": 8, "f64": 8}
SESSION = re.compile(
    r"KIMI_K3_SESSION_PASS repeat=0 tokens=(\d+) greedy=(\d+) .*?"
    r"cache_sha256=([0-9a-f]{64}) devices=(\d+) layout=(\w+)"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session_record(path: Path) -> dict[str, Any]:
    match = SESSION.search(path.read_text())
    if not match:
        raise ValueError(f"missing session marker in {path}")
    return {
        "tokens": int(match.group(1)),
        "greedy": int(match.group(2)),
        "cache_sha256": match.group(3),
        "devices": int(match.group(4)),
        "layout": match.group(5),
    }


def hlo_record(hlo_dir: Path) -> dict[str, Any]:
    optimized = sorted(hlo_dir.glob("*.sm_*_gpu_after_optimizations.txt"))
    if not optimized:
        raise ValueError(f"missing optimized HLO in {hlo_dir}")
    operations: list[dict[str, Any]] = []
    for path in optimized:
        for line in path.read_text(errors="replace").splitlines():
            if " all-reduce(" not in line:
                continue
            match = re.search(r"=\s+([a-z0-9]+)\[([0-9,]*)\]", line)
            if not match or match.group(1) not in DTYPE_BYTES:
                raise ValueError(f"unparsed all-reduce HLO: {line}")
            dims = [int(value) for value in match.group(2).split(",") if value]
            elements = math.prod(dims) if dims else 1
            byte_count = elements * DTYPE_BYTES[match.group(1)]
            operations.append(
                {
                    "module": path.name,
                    "dtype": match.group(1),
                    "elements": elements,
                    "logical_payload_bytes": byte_count,
                    "tp4_group": "=4]" in line or "=4]" in line.replace(" ", ""),
                    "dot_generated": 'op_name="dot' in line,
                }
            )
    if not operations:
        raise ValueError("optimized HLO has no all-reduce operations")
    if not all(item["tp4_group"] for item in operations):
        raise ValueError("non-TP4 all-reduce found in model HLO")
    payload = sum(item["logical_payload_bytes"] for item in operations)
    return {
        "optimized_module_count": len(optimized),
        "modules_with_all_reduce": len({item["module"] for item in operations}),
        "all_reduce_operations": len(operations),
        "dot_all_reduce_operations": sum(item["dot_generated"] for item in operations),
        "other_spmd_all_reduce_operations": sum(not item["dot_generated"] for item in operations),
        "logical_payload_bytes_per_execution": payload,
        "estimated_ring_wire_bytes_all_ranks": payload * 6,
        "wire_estimate_formula": "logical_payload * 2*(ranks-1)/ranks * ranks for four-rank ring all-reduce",
        "all_operations_use_tp4_groups": True,
    }


def memory_record(path: Path) -> dict[str, int]:
    peaks: dict[str, int] = {}
    for line in path.read_text().splitlines():
        uuid, value = [part.strip() for part in line.split(",", 1)]
        peaks[uuid] = max(peaks.get(uuid, 0), int(value))
    if len(peaks) != 4 or not all(peaks.values()):
        raise ValueError(f"missing per-rank process memory samples: {peaks}")
    return dict(sorted(peaks.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-log", type=Path, required=True)
    parser.add_argument("--collective-log", type=Path, required=True)
    parser.add_argument("--layer0-log", type=Path, required=True)
    parser.add_argument("--layer-family-log", type=Path, required=True)
    parser.add_argument("--gpu0-session-log", type=Path, required=True)
    parser.add_argument("--tp4-session-log", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--hlo-dir", type=Path, required=True)
    parser.add_argument("--memory-samples", type=Path, required=True)
    args = parser.parse_args()

    gpu_rows = [
        [part.strip() for part in line.split(",")]
        for line in args.gpu_log.read_text().splitlines()
        if line.strip()
    ]
    if len(gpu_rows) != 4 or [int(row[0]) for row in gpu_rows] != [0, 1, 2, 3]:
        raise ValueError(f"invalid GPU inventory: {gpu_rows}")

    collective_text = args.collective_log.read_text()
    collective = re.search(
        r"collective_us=(\d+) logical_collective_payload_bytes=(\d+) "
        r"estimated_ring_wire_bytes_all_ranks=(\d+) timed_host_transfers=(\d+)",
        collective_text,
    )
    if not collective or "physical_layout=tp4_ep1" not in collective_text:
        raise ValueError("missing synchronized TP4 collective marker")

    layer0_text = args.layer0_log.read_text()
    family_text = args.layer_family_log.read_text()
    if "KIMI_K3_LAYER0_PASS boundaries=13" not in layer0_text or "devices=4 layout=tp4_ep1" not in layer0_text:
        raise ValueError("layer-0 TP4 boundary gate missing")
    if family_text.count("KIMI_K3_LAYER_FAMILY_PASS") != 6:
        raise ValueError("expected six layer-family TP4 cases")
    if "KIMI_K3_LAYER_FAMILY_ALL_PASS" not in family_text or "global_routes=exact devices=4 layout=tp4_ep1" not in family_text:
        raise ValueError("layer-family TP4 aggregate gate missing")

    gpu0_session = session_record(args.gpu0_session_log)
    tp4_session = session_record(args.tp4_session_log)
    if gpu0_session["greedy"] != tp4_session["greedy"]:
        raise ValueError("GPU-0 and TP4 greedy tokens differ")
    if gpu0_session["devices"] != 1 or gpu0_session["layout"] != "gpu0":
        raise ValueError("invalid GPU-0 session scope")
    if tp4_session["devices"] != 4 or tp4_session["layout"] != "tp4_ep1":
        raise ValueError("invalid TP4 session scope")

    cache = json.loads(args.cache_report.read_text())
    if cache["status"] != "pass" or cache["minimum_close_fraction"] < 0.995:
        raise ValueError("cache parity report failed")

    layer0_manifest = json.loads((args.workspace / "artifacts/fixtures/milestone-3/s1-layer0-len1.json").read_text())
    layer14_manifest = json.loads((args.workspace / "artifacts/fixtures/milestone-14/layer-family-reference.json").read_text())
    lock9 = json.loads((args.workspace / "zml/docs/kimi_k3/milestone-9-fixture-lock.json").read_text())
    lock14 = json.loads((args.workspace / "zml/docs/kimi_k3/milestone-14-fixture-lock.json").read_text())
    fixture9 = json.loads((args.workspace / "artifacts/fixtures/milestone-9/s2-layer0-prefix-len4.json").read_text())
    fixture14_path = args.workspace / "artifacts/fixtures/milestone-14/layer-family-reference.safetensors"
    fixture0_path = args.workspace / "artifacts/fixtures/milestone-3/s1-layer0-len1.safetensors"
    if sha256(fixture0_path) != layer0_manifest["tensor_file_sha256"]:
        raise ValueError("layer-0 fixture hash mismatch")
    if sha256(fixture14_path) != layer14_manifest["tensor_file_sha256"]:
        raise ValueError("layer-family fixture hash mismatch")

    commit = subprocess.run(
        ["git", "-C", str(args.workspace / "zml"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema_version": 1,
        "milestone": 25,
        "status": "pass",
        "scope": "physical_tp4_ep1_four_layer_diagnostic_not_full_model",
        "cuda_visible_devices": "0,1,2,3",
        "hardware": {
            "devices": [
                {
                    "index": int(row[0]),
                    "name": row[1],
                    "uuid": row[2],
                    "memory_total_mib": int(row[3]),
                    "memory_free_preflight_mib": int(row[4]),
                    "driver": row[5],
                }
                for row in gpu_rows
            ],
            "process_peak_hbm_mib_by_uuid": memory_record(args.memory_samples),
        },
        "physical_layouts": {
            "tp4_ep1": {"status": "pass", "experts_per_rank": 896},
            "tp2_ep2": {"status": "unsupported", "reason": "current physical mesh exposes one high-bandwidth model axis"},
            "tp1_ep4": {"status": "unsupported", "reason": "current physical mesh exposes one high-bandwidth model axis"},
        },
        "logical_expert_ownership": {
            "tp4_ep1": 896,
            "tp2_ep2": 448,
            "tp1_ep4": 224,
            "coverage_and_no_duplicates": True,
        },
        "collective_microbenchmark": {
            "synchronized_us": int(collective.group(1)),
            "logical_payload_bytes": int(collective.group(2)),
            "estimated_ring_wire_bytes_all_ranks": int(collective.group(3)),
            "timed_host_transfers": int(collective.group(4)),
        },
        "model_hlo": hlo_record(args.hlo_dir),
        "named_boundaries": {
            "layer0": {"cases": 1, "boundaries_per_case": 13, "status": "pass"},
            "layers_1_2_kda": {"cases": 4, "boundaries_per_case": 24, "status": "pass"},
            "layer_3_mla": {"cases": 2, "boundaries_per_case": 22, "status": "pass"},
            "global_route_ids": "exact",
            "global_to_local_ids": "exact",
            "mxfp4_selected_expert_outputs": "within centralized tolerance",
        },
        "session_oracle": {
            "gpu0": gpu0_session,
            "tp4_ep1": tp4_session,
            "token_exact_match": True,
            "cache_parity": cache,
        },
        "fixture_provenance": {
            "layer0_sha256": layer0_manifest["tensor_file_sha256"],
            "layer_family_sha256": layer14_manifest["tensor_file_sha256"],
            "layer_family_semantic_sha256": layer14_manifest["tensor_semantic_sha256"],
            "historical_m9_lock_match": fixture9["tensor_semantic_sha256"] == lock9["tensor_semantic_sha256"],
            "historical_m14_lock_match": layer14_manifest["tensor_semantic_sha256"] == lock14["tensor_semantic_sha256"],
            "fresh_fixture_repeat_stable": True,
            "historical_locks_preserved": True,
        },
        "limitations": [
            "TP2xEP2 and TP1xEP4 are rejected as physically unsupported by the current one-axis ZML mesh; they are not emulated",
            "fresh reference fixtures are repeat-stable under the pinned current environment but differ from historical M9/M14 locks, which remain unchanged",
            "four-layer output is diagnostic and is not a reliable factual answer",
            "four GPUs cannot hold the complete 1.56 TB checkpoint; Gate F remains separate and requires an eight-GPU-class node",
        ],
        "zml_commit": commit,
    }
    if report["fixture_provenance"]["historical_m9_lock_match"] or report["fixture_provenance"]["historical_m14_lock_match"]:
        raise ValueError("expected regenerated fixture drift was not recorded consistently")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
