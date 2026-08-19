#!/usr/bin/env python3
"""Convert the CUDA ZML primitive test log into a machine-readable report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PASS = re.compile(
    r"^KIMI_K3_PRIMITIVE_PASS name=(?P<name>\S+) "
    r"elapsed_us=(?P<elapsed>\d+) input_shape=(?P<shape>.+)$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixture = json.loads(args.fixture_manifest.read_text())
    comparisons = []
    all_pass = False
    for line in args.log.read_text().splitlines():
        match = PASS.match(line)
        if match:
            comparisons.append(
                {
                    "name": match.group("name"),
                    "passed": True,
                    "elapsed_us": int(match.group("elapsed")),
                    "input_shape": match.group("shape"),
                }
            )
        if line == "KIMI_K3_PRIMITIVES_ALL_PASS count=20 backend=cuda":
            all_pass = True
    if len(comparisons) != 20 or len({item["name"] for item in comparisons}) != 20:
        raise SystemExit(f"expected 20 unique primitive comparisons, found {len(comparisons)}")
    if not all_pass:
        raise SystemExit("CUDA test log lacks the all-pass sentinel")
    if not all(check["passed"] for check in fixture["official_vs_numpy"].values()):
        raise SystemExit("fixture manifest contains a failed official/NumPy comparison")

    report = {
        "schema_version": 1,
        "milestone": 5,
        "status": "PASS",
        "backend": "cuda",
        "device": fixture["device"],
        "cpu_inference_fallback": False,
        "fixture_file_sha256": fixture["tensor_file_sha256"],
        "fixture_semantic_sha256": fixture["tensor_semantic_sha256"],
        "tolerance_manifest": fixture["tolerance_manifest"],
        "official_vs_numpy": fixture["official_vs_numpy"],
        "zml_vs_numpy": comparisons,
        "timing_scope": "synchronized executable call plus device-to-host comparison",
        "timing_summary_us": {
            "min": min(item["elapsed_us"] for item in comparisons),
            "max": max(item["elapsed_us"] for item in comparisons),
            "mean": sum(item["elapsed_us"] for item in comparisons) / len(comparisons),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "comparisons": len(comparisons), "output": str(args.output)}))


if __name__ == "__main__":
    main()
