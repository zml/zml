#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import checkpoint_receipt as receipt


class CheckpointReceiptTest(unittest.TestCase):
    def make_model(self, root: Path, *, revisions: dict[int, str] | None = None) -> Path:
        model = root / "model"
        metadata_dir = model / ".cache" / "huggingface" / "download"
        metadata_dir.mkdir(parents=True)
        (model / "config.json").write_text('{"model_type":"kimi_k3"}\n')
        weight_map = {}
        for number in range(1, receipt.SHARD_COUNT + 1):
            name = f"model-{number:05d}-of-000096.safetensors"
            (model / name).write_bytes(bytes([number % 256]))
            digest = f"{number:064x}"
            revision = (revisions or {}).get(number, "a" * 40)
            (metadata_dir / f"{name}.metadata").write_text(
                f"{revision}\n{digest}\n1700000000.{number}\n"
            )
            weight_map[f"tensor.{number}"] = name
        (model / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 96}, "weight_map": weight_map})
        )
        return model

    def test_builds_complete_offline_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self.make_model(root)
            name = "model-00001-of-000096.safetensors"
            baseline = root / "baseline.json"
            baseline.write_text(json.dumps({"files": {name: {"bytes": 1, "sha256": f"{1:064x}"}}}))
            result = receipt.build_receipt(model, baseline)
            self.assertEqual(result["source_revision"], "a" * 40)
            self.assertEqual(len(result["files"]), 96)
            self.assertEqual(result["physical_shard_bytes"], 96)
            self.assertTrue(result["offline"])

    def test_rejects_mixed_revisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = self.make_model(Path(directory), revisions={96: "b" * 40})
            with self.assertRaisesRegex(receipt.ReceiptError, "mixed revisions"):
                receipt.build_receipt(model)

    def test_rejects_baseline_hash_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self.make_model(root)
            baseline = root / "baseline.json"
            baseline.write_text(json.dumps({"files": {
                "model-00001-of-000096.safetensors": {"bytes": 1, "sha256": "f" * 64}
            }}))
            with self.assertRaisesRegex(receipt.ReceiptError, "baseline hash"):
                receipt.build_receipt(model, baseline)

    def test_rejects_missing_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = self.make_model(Path(directory))
            (model / ".cache/huggingface/download/model-00096-of-000096.safetensors.metadata").unlink()
            with self.assertRaisesRegex(receipt.ReceiptError, "missing Hugging Face metadata"):
                receipt.build_receipt(model)


if __name__ == "__main__":
    unittest.main()
