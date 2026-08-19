#!/usr/bin/env python3
"""Create the Milestone 9 real-weight Gate A correctness/timing report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FIELDS = re.compile(r"(\w+)=([^ ]+)")


def parse_pass(log: Path, marker: str) -> dict[str, str]:
    lines = [line for line in log.read_text().splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise SystemExit(f"expected one {marker} record in {log}")
    return dict(FIELDS.findall(lines[0]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    lengths: dict[str, dict[str, str]] = {}
    for length in (1, 4, 8, 16):
        record = parse_pass(args.log_dir / f"zml-layer0-len{length}.stdout.log", "KIMI_K3_LAYER0_PASS")
        if record.get("boundaries") != "13":
            raise SystemExit(f"incomplete length-{length} boundary inventory")
        lengths[str(length)] = record
    cache = parse_pass(args.log_dir / "zml-cache-handoff.stdout.log", "KIMI_K3_LAYER0_CACHE_PASS")
    prefix = parse_pass(args.log_dir / "zml-prefix.stdout.log", "KIMI_K3_PREFIX_PASS")
    if cache.get("boundaries") != "30" or prefix.get("boundaries") != "13":
        raise SystemExit("incomplete cache or prefix boundary inventory")
    report = {
        "schema_version": 1,
        "milestone": 9,
        "status": "PASS",
        "gate": "A",
        "backend": "cuda",
        "device": manifest["device"],
        "cpu_inference_fallback": False,
        "weights": manifest["checkpoint"],
        "semantic_fixture_sha256": manifest["tensor_semantic_sha256"],
        "fixture_file_sha256": manifest["tensor_file_sha256"],
        "official_greedy_token": manifest["greedy_token"],
        "zml_layer0_lengths": lengths,
        "zml_cache_handoff": cache,
        "zml_prefix": prefix,
        "official_prefix_timing": manifest["timing"],
        "timing_scope": "debug-disabled synchronized execution; fixture comparison occurs after timing",
        "timing_is_production_benchmark": False,
        "selector_tolerance": {
            "absolute": 0.0002,
            "relative": 0.002,
            "minimum_close_fraction": 1.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "gate": "A", "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
