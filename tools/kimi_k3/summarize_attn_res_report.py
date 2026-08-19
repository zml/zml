#!/usr/bin/env python3
"""Create the saved Milestone-6 CUDA comparison/timing report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PASS = re.compile(r"^KIMI_K3_ATTN_RES_PASS name=(\S+) elapsed_us=(\d+) candidates=(\{.*\}) probabilities=(\{.*\})$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cases = []
    workspace_pass = False
    all_pass = False
    for line in args.log.read_text().splitlines():
        match = PASS.match(line)
        if match:
            cases.append({"name": match.group(1), "elapsed_us": int(match.group(2)), "candidates": match.group(3), "probabilities": match.group(4), "passed": True})
        workspace_pass |= line.startswith("KIMI_K3_ATTN_RES_WORKSPACE_PASS")
        all_pass |= line == "KIMI_K3_ATTN_RES_ALL_PASS count=12 backend=cuda"
    if len(cases) != 12 or not workspace_pass or not all_pass:
        raise SystemExit("incomplete Attention Residual CUDA log")
    report = {
        "schema_version": 1,
        "milestone": 6,
        "status": "PASS",
        "backend": "cuda",
        "device": manifest["device"],
        "cpu_inference_fallback": False,
        "workspace_reset_passed": True,
        "semantic_fixture_sha256": manifest["tensor_semantic_sha256"],
        "official_vs_numpy": manifest["official_vs_numpy"],
        "zml_vs_numpy": cases,
        "timing_scope": "synchronized selector execution plus four device-to-host comparisons",
        "timing_summary_us": {"min": min(x["elapsed_us"] for x in cases), "max": max(x["elapsed_us"] for x in cases), "mean": sum(x["elapsed_us"] for x in cases) / len(cases)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "cases": len(cases), "output": str(args.output)}))


if __name__ == "__main__":
    main()
