#!/usr/bin/env python3
"""Create the Milestone 10 router correctness/timing/histogram report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PASS = re.compile(r"^KIMI_K3_ROUTER_PASS (.*)$")
FIELDS = re.compile(r"(\w+)=([^ ]+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    cases: dict[str, dict[str, str]] = {}
    all_pass = False
    for line in args.log.read_text().splitlines():
        match = PASS.match(line)
        if match:
            fields = dict(FIELDS.findall(match.group(1)))
            cases[fields.pop("case")] = fields
        all_pass |= line == "KIMI_K3_ROUTER_ALL_PASS cases=4 real_tokens=4 exact_sets=true backend=cuda"
    if set(cases) != {"real", "tie", "bias", "grouped"} or not all_pass:
        raise SystemExit("incomplete router CUDA log")
    if any(case.get("boundaries") != "6" for case in cases.values()):
        raise SystemExit("incomplete router boundary inventory")
    report = {
        "schema_version": 1,
        "milestone": 10,
        "status": "PASS",
        "backend": "cuda",
        "device": manifest["device"],
        "cpu_inference_fallback": False,
        "semantic_fixture_sha256": manifest["tensor_semantic_sha256"],
        "fixture_file_sha256": manifest["tensor_file_sha256"],
        "official_checked_rows": manifest["official_checked_rows"],
        "strict_fp32_tolerance": {"absolute": 1e-5, "relative": 1e-5, "minimum_close_fraction": 1.0},
        "tie_policy": manifest["tie_policy"],
        "bias_not_used_as_mixture_weight": manifest["bias_not_used_as_mixture_weight"],
        "cases": {
            name: {
                **values,
                "route_histogram": manifest["cases"][name]["route_histogram"],
                "config": {
                    key: manifest["cases"][name][key]
                    for key in ("experts", "top_k", "num_expert_group", "topk_group", "scaling_factor")
                },
            }
            for name, values in cases.items()
        },
        "timing_scope": "synchronized diagnostic route-plan execution followed by device-to-host boundary comparisons",
        "timing_is_production_benchmark": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "cases": len(cases), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
