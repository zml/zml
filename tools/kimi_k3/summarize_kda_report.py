#!/usr/bin/env python3
"""Create the saved Milestone 7 KDA CUDA comparison/timing report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


STEP = re.compile(
    r"^KIMI_K3_KDA_STEP_PASS step=(\d+) boundaries=(\d+) elapsed_us=(\d+) "
    r"output=(\{.*?\}) recurrent_cache=(\{.*?\}) conv_cache=(\{.*\})$"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    steps = []
    all_pass = False
    for line in args.log.read_text().splitlines():
        match = STEP.match(line)
        if match:
            steps.append(
                {
                    "step": int(match.group(1)),
                    "boundaries": int(match.group(2)),
                    "elapsed_us": int(match.group(3)),
                    "output_shape": match.group(4),
                    "recurrent_cache_shape": match.group(5),
                    "conv_cache_shape": match.group(6),
                    "passed": True,
                }
            )
        all_pass |= line == "KIMI_K3_KDA_ALL_PASS steps=4 boundaries_per_step=26 backend=cuda"
    if [item["step"] for item in steps] != [0, 1, 2, 3] or not all_pass:
        raise SystemExit("incomplete KDA CUDA differential log")
    samples = [item["elapsed_us"] for item in steps]
    report = {
        "schema_version": 1,
        "milestone": 7,
        "status": "PASS",
        "backend": "cuda",
        "device": manifest["device"],
        "cpu_inference_fallback": False,
        "semantic_fixture_sha256": manifest["tensor_semantic_sha256"],
        "state_layout": manifest["state_layout"],
        "conv_cache_layout": manifest["conv_cache_layout"],
        "official_vs_readable": manifest["official_vs_readable"],
        "numpy_vs_readable": manifest["numpy_vs_readable"],
        "zml_vs_readable": steps,
        "zml_named_comparisons": sum(item["boundaries"] for item in steps),
        "timing_scope": "synchronized tiny reference decode plus 26 device-to-host boundary comparisons",
        "timing_is_production_benchmark": False,
        "timing_summary_us": {
            "first_call": samples[0],
            "steady_min": min(samples[1:]),
            "steady_max": max(samples[1:]),
            "steady_mean": sum(samples[1:]) / len(samples[1:]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "PASS",
                "steps": len(steps),
                "zml_named_comparisons": report["zml_named_comparisons"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
