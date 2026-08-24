#!/usr/bin/env python3
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cleanup_audit


class CleanupAuditTests(unittest.TestCase):
    def test_repository_passes(self) -> None:
        report = cleanup_audit.audit(Path(__file__).resolve().parents[2])
        self.assertEqual(report["status"], "pass", report["issues"])
        self.assertEqual(report["production_results"], "compact")
        self.assertFalse(report["public_configurable_layer_limit"])
        self.assertEqual(report["normal_example_fixed_resident_layers"], 47)

    def test_temporary_marker_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in cleanup_audit.SCAN_ROOTS:
                (root / relative).mkdir(parents=True)
            source_root = Path(__file__).resolve().parents[2]
            required = (
                "examples/llm/main.zig",
                "examples/llm/models/common.zig",
                "examples/llm/models/kimi_k3/model.zig",
                "examples/llm/models/kimi_k3/inference.zig",
                "examples/llm/models/kimi_k3/session.zig",
                "examples/llm/models/kimi_k3/runtime_weights.zig",
                "examples/llm/BUILD.bazel",
            )
            for relative in required:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text((source_root / relative).read_text())
            (root / "tools/kimi_k3/probe.py").write_text("# KIMI_K3_" + "TEMP_REMOVE_M20\n")
            report = cleanup_audit.audit(root)
            self.assertEqual(report["status"], "fail")
            self.assertTrue(any("temporary pattern" in issue for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
