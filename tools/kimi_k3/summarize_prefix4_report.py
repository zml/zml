#!/usr/bin/env python3
"""Summarize the Milestone 15 ZML full-prefix comparison log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LAYER = re.compile(
    r"KIMI_K3_PREFIX4_LAYER_PASS layer=(?P<layer>\d+) attention=(?P<attention>\w+)"
    r"(?: experts=(?P<experts>\d+))? load_us=(?P<load>\d+) compile_us=(?P<compile>\d+)"
    r" execute_us=(?P<execute>\d+)"
)
ROUTE = re.compile(
    r"KIMI_K3_PREFIX4_ROUTE_OVERLAP prefix=(?P<prefix>\S+) matched=(?P<matched>\d+)"
    r" total=(?P<total>\d+) fraction=(?P<fraction>[0-9.]+)"
)
GREEDY = re.compile(
    r"KIMI_K3_PREFIX4_GREEDY_TIE prefix=(?P<prefix>\S+) actual_token=(?P<actual>\d+)"
    r" official_token=(?P<official>\d+).* actual_max=(?P<actual_max>[-0-9.]+)"
    r" official_max=(?P<official_max>[-0-9.]+)"
)
HEAD = re.compile(
    r"KIMI_K3_PREFIX4_HEAD_PASS load_us=(?P<load>\d+) compile_us=(?P<compile>\d+) execute_us=(?P<execute>\d+)"
)


def summarize(log: str, manifest: dict, trace_path: Path) -> dict:
    if "KIMI_K3_PREFIX4_REFERENCE_HEAD_PASS inputs=official tolerance=strict" not in log:
        raise ValueError("missing strict reference-head pass marker")
    if "KIMI_K3_PREFIX4_ALL_PASS" not in log:
        raise ValueError("missing full-prefix pass marker")
    layers = {}
    for match in LAYER.finditer(log):
        row = match.groupdict()
        layers[row["layer"]] = {
            "attention": row["attention"],
            "experts": int(row["experts"]) if row["experts"] else None,
            "load_us": int(row["load"]),
            "compile_us": int(row["compile"]),
            "execute_us": int(row["execute"]),
        }
    if set(layers) != {"0", "1", "2", "3"}:
        raise ValueError(f"incomplete layer timings: {sorted(layers)}")
    routes = {}
    for match in ROUTE.finditer(log):
        row = match.groupdict()
        routes[row["prefix"]] = {
            "matched": int(row["matched"]),
            "total": int(row["total"]),
            "fraction": float(row["fraction"]),
        }
    if set(routes) != {"layer1", "layer1.decode", "layer2", "layer2.decode", "layer3", "layer3.decode"}:
        raise ValueError(f"incomplete route telemetry: {sorted(routes)}")
    if min(row["fraction"] for row in routes.values()) < 0.85:
        raise ValueError("chained route overlap below 85%")
    greedy = {}
    for match in GREEDY.finditer(log):
        row = match.groupdict()
        greedy[row["prefix"]] = {
            "actual_token": int(row["actual"]),
            "official_token": int(row["official"]),
            "actual_max": float(row["actual_max"]),
            "official_max": float(row["official_max"]),
            "acceptance": "maximum_tie_sets_intersect",
        }
    if set(greedy) != {"prefix", "decode"}:
        raise ValueError(f"incomplete greedy telemetry: {sorted(greedy)}")
    head_match = HEAD.search(log)
    if head_match is None:
        raise ValueError("missing chained head timing")
    if manifest.get("prefill_greedy_token") != manifest.get("decode_greedy_token"):
        raise ValueError("official fixture tokens disagree")
    if not trace_path.is_file() or trace_path.stat().st_size == 0:
        raise ValueError("missing Perfetto trace")
    return {
        "schema_version": 1,
        "milestone": 15,
        "backend": "cuda",
        "model_path": "/dev/shm/kimi-k3/moonshot/kimi-k3",
        "experts_per_moe": 896,
        "routing": {
            "mode": "normal_global_no_injection",
            "independent_exact_route_evidence": "milestone-14",
            "chained_overlap": routes,
            "minimum_required_fraction": 0.85,
        },
        "layers": layers,
        "head": {
            "official_input_parity": "strict_pass",
            "chained": {
                "load_us": int(head_match.group("load")),
                "compile_us": int(head_match.group("compile")),
                "execute_us": int(head_match.group("execute")),
            },
            "greedy": greedy,
        },
        "fixture": {
            "tensor_file_sha256": manifest["tensor_file_sha256"],
            "tensor_semantic_sha256": manifest["tensor_semantic_sha256"],
            "official_greedy_token": manifest["prefill_greedy_token"],
        },
        "trace": {
            "path": str(trace_path),
            "bytes": trace_path.stat().st_size,
            "format": "perfetto_trace_json",
        },
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.log.read_text(), json.loads(args.manifest.read_text()), args.trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
