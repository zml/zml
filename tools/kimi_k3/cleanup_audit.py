#!/usr/bin/env python3
"""Audit the permanent Kimi K3 production and conformance boundary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SCAN_ROOTS = ("examples/llm", "tools/kimi_k3", "docs/kimi_k3")
TEMPORARY_PATTERNS = (
    re.compile("KIMI_K3_" + "TEMP_" + "REMOVE"),
    re.compile("TO" + "DO" + ".*KIMI_K3", re.IGNORECASE),
    re.compile("rou" + "te" + ".*" + "over" + "ride", re.IGNORECASE),
)


def _text_files(root: Path):
    for relative in SCAN_ROOTS:
        base = root / relative
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".zig", ".py", ".md", ".json", ".bazel"}:
                yield path


def audit(root: Path) -> dict[str, Any]:
    root = root.resolve()
    issues: list[str] = []
    scanned = 0
    for path in _text_files(root):
        scanned += 1
        text = path.read_text(errors="replace")
        for pattern in TEMPORARY_PATTERNS:
            if pattern.search(text):
                issues.append(f"temporary pattern {pattern.pattern!r}: {path.relative_to(root)}")

    production = {
        "main": (root / "examples/llm/main.zig").read_text(),
        "common": (root / "examples/llm/models/common.zig").read_text(),
        "model": (root / "examples/llm/models/kimi_k3/model.zig").read_text(),
        "inference": (root / "examples/llm/models/kimi_k3/inference.zig").read_text(),
        "session": (root / "examples/llm/models/kimi_k3/session.zig").read_text(),
        "runtime_weights": (root / "examples/llm/models/kimi_k3/runtime_weights.zig").read_text(),
        "build": (root / "examples/llm/BUILD.bazel").read_text(),
    }
    for name in ("main", "common", "model"):
        if "kimi_k3_layer_limit" in production[name]:
            issues.append(f"public partial-layer hook remains in {name}")
    fixed_layer_match = re.search(
        r"pub const example_resident_layer_count: usize = (\d+);",
        production["model"],
    )
    full_layer_match = re.search(
        r"pub const full_model_layer_count: usize = (\d+);",
        production["model"],
    )
    if fixed_layer_match is None or int(fixed_layer_match.group(1)) != 47:
        issues.append("four-rank Kimi example is not fixed to 47 resident layers")
    if full_layer_match is None or int(full_layer_match.group(1)) != 93:
        issues.append("eight-rank Kimi example is not fixed to 93 resident layers")
    for required_resident_example in (
        "fixed_example_prefix = true",
        "KimiK3NormalExampleRequiresFourOrEightCudaDevices",
        "return self.loadResidentBuffers(",
        "KIMI_K3_DIAGNOSTIC_WARNING layers={} full_model={} reliable_answer=false",
        "4 => example_resident_layer_count",
        "8 => full_model_layer_count",
    ):
        if required_resident_example not in production["model"]:
            issues.append(f"missing resident Kimi example invariant: {required_resident_example}")
    for required_expert_partition in (
        "shared_axis",
        "withPartitioning(.{ .expert = .experts })",
        "KimiK3SharedAxisExpertPartitionRequiresFourOrEightCudaDevices",
        "device_count == 4 or device_count == 8",
    ):
        if required_expert_partition not in production["runtime_weights"] and required_expert_partition not in production["model"]:
            issues.append(f"missing Kimi shared-axis expert invariant: {required_expert_partition}")

    # Model-local loading telemetry is required; inference/session hot paths
    # remain free of timing and logging.
    for name in ("session",):
        if "std.Io.Clock.now" in production[name] or "log.info(" in production[name]:
            issues.append(f"hot-path timing/logging remains in {name}")
    for token in (
        "layer.forwardLayer0,",
        "layer.diagnosticSessionHead,",
        ") layer.KdaMoeResult {",
        ") layer.MlaMoeResult {",
    ):
        if token in production["inference"]:
            issues.append(f"expanded diagnostic executable result remains: {token}")
    for required in (
        "layer.forwardLayer0Compact",
        "layer.forwardKdaMoeDecodeCompact",
        "layer.forwardMlaMoeSessionCompact",
        "layer.sessionHead",
    ):
        if required not in production["inference"]:
            issues.append(f"missing compact production executable: {required}")
    if "name = \"kimi_k3_diagnostic\"" in production["build"]:
        issues.append("obsolete kimi_k3_diagnostic Bazel target remains")
    for relative in (
        "examples/llm/kimi_k3_diagnostic.zig",
        "examples/llm/models/kimi_k3_diagnostic.zig",
    ):
        if (root / relative).exists():
            issues.append(f"obsolete diagnostic file remains: {relative}")

    return {
        "schema_version": 1,
        "status": "pass" if not issues else "fail",
        "scanned_files": scanned,
        "production_layer_limit": False,
        "public_configurable_layer_limit": False,
        "normal_example_fixed_resident_layers": 47,
        "normal_example_eight_gpu_resident_layers": 93,
        "normal_example_supported_device_counts": [4, 8],
        "production_hot_path_debug": False,
        "production_results": "compact",
        "diagnostic_target_removed": True,
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    if report["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
