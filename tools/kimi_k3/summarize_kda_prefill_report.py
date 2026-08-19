#!/usr/bin/env python3
"""Create the Milestone 8 CUDA correctness/timing report."""

from __future__ import annotations
import argparse, json, re
from pathlib import Path

PASS = re.compile(r"^KIMI_K3_KDA_PREFILL_PASS kind=(\S+)(.*)$")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cases = []
    all_pass = False
    for line in args.log.read_text().splitlines():
        match = PASS.match(line)
        if match:
            fields = dict(re.findall(r"(\w+)=([^ ]+)", match.group(2)))
            cases.append({"kind": match.group(1), **fields, "passed": True})
        all_pass |= line == "KIMI_K3_KDA_PREFILL_ALL_PASS full=4 token_decode=4 splits=15 continuation=1 backend=cuda"
    if len(cases) != 24 or not all_pass:
        raise SystemExit("incomplete KDA prefill CUDA log")
    timings = [int(case["elapsed_us"]) for case in cases if "elapsed_us" in case]
    report = {
        "schema_version": 1, "milestone": 8, "status": "PASS", "backend": "cuda",
        "device": manifest["device"], "cpu_inference_fallback": False,
        "semantic_fixture_sha256": manifest["tensor_semantic_sha256"],
        "official_checks": manifest["official_checks"], "zml_cases": cases,
        "lengths": manifest["lengths"], "official_split_points": manifest["split_points"],
        "zml_split_points": 15, "token_decode_steps": 29,
        "timing_is_production_benchmark": False,
        "timing_scope": "synchronized tiny reference calls with device-to-host comparisons",
        "timing_summary_us": {"min": min(timings), "max": max(timings), "mean": sum(timings) / len(timings)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "cases": len(cases), "output": str(args.output)}, sort_keys=True))

if __name__ == "__main__":
    main()
