#!/usr/bin/env python3
"""Create the Milestone 12 Gate C correctness/timing/memory report."""

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
    lines = args.log.read_text().splitlines()
    records = [dict(FIELDS.findall(line)) for line in lines if line.startswith("KIMI_K3_MLA_PASS ")]
    if len(records) != 5 or not any(line.startswith("KIMI_K3_MLA_ALL_PASS ") for line in lines):
        raise SystemExit("incomplete MLA CUDA log")
    prefill = sorted(
        (record for record in records if record.get("kind") == "prefill"),
        key=lambda record: int(record["length"]),
    )
    decode = [record for record in records if record.get("kind") == "decode"]
    if [int(record["length"]) for record in prefill] != [1, 4, 8, 16] or len(decode) != 1:
        raise SystemExit("MLA CUDA case inventory mismatch")
    if any(record.get("boundaries") != "25" for record in records):
        raise SystemExit("MLA CUDA boundary inventory mismatch")
    report = {
        "schema_version": 1,
        "milestone": 12,
        "gate": "C",
        "status": "PASS",
        "backend": "cuda",
        "device": manifest["device"],
        "cpu_inference_fallback": False,
        "semantic_fixture_sha256": manifest["tensor_semantic_sha256"],
        "prefill": prefill,
        "decode": decode[0],
        "reference_timing": manifest["timing"],
        "peak_memory": manifest["peak_memory"],
        "cache_kind": manifest["cache_kind"],
        "timing_scope": "readable expanded-cache MLA with 25 returned comparison boundaries",
        "timing_is_production_benchmark": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "gate": "C", "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
