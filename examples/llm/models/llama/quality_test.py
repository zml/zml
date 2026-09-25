"""Check next-token alignment, normalization and logit-file validation."""

import hashlib
import math
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

import quality


class QualityTest(unittest.TestCase):
    def write_logits(self, path, values, tokens):
        values = np.asarray(values, dtype=np.float32)
        windows, seqlen, vocab = values.shape
        header = quality.MAGIC + struct.pack("<III", seqlen, vocab, windows)
        header += hashlib.sha256(tokens).digest()
        bf16 = (values.view(np.uint32) >> 16).astype("<u2")
        path.write_bytes(header + bf16.tobytes())

    def test_next_token_alignment_and_identical_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token_path, logits_path = root / "tokens", root / "logits"
            tokens = np.asarray([[0, 1, 2]], dtype="<u4").tobytes()
            token_path.write_bytes(tokens)
            self.write_logits(logits_path, [[[0, 1, 2], [2, 1, 0]]], tokens)
            result = quality.compare(token_path, logits_path, logits_path)
            expected_nll = math.log(1 + math.e + math.exp(2)) - 0.5
            self.assertAlmostEqual(result["reference_perplexity"], math.exp(expected_nll), places=12)
            self.assertEqual(result["scored_tokens"], 2)
            self.assertEqual(result["mean_kl_reference_to_candidate"], 0)
            self.assertEqual(result["perplexity_relative_change"], 0)
            self.assertEqual(result["top1_agreement"], 1)
            broken = root / "broken"
            broken.write_bytes(logits_path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "length"):
                quality.compare(token_path, logits_path, broken)
            token_path.write_bytes(np.asarray([[1, 1, 2]], dtype="<u4").tobytes())
            with self.assertRaisesRegex(ValueError, "hash"):
                quality.compare(token_path, logits_path, logits_path)

    def test_shift_invariance_and_prediction_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokens = np.asarray([[0, 0, 1]], dtype="<u4").tobytes()
            token_path = root / "tokens"
            token_path.write_bytes(tokens)
            reference, shifted, changed = (root / name for name in ("reference", "shifted", "changed"))
            values = np.asarray([[[2, 1, 0], [0, 2, 1]]], dtype=np.float32)
            self.write_logits(reference, values, tokens)
            self.write_logits(shifted, values + 100, tokens)
            self.write_logits(changed, values[..., ::-1].copy(), tokens)
            same = quality.compare(token_path, reference, shifted)
            self.assertEqual(same["mean_kl_reference_to_candidate"], 0)
            self.assertEqual(same["perplexity_relative_change"], 0)
            other = quality.compare(token_path, reference, changed)
            self.assertGreater(other["mean_kl_reference_to_candidate"], 0)
            self.assertGreater(other["perplexity_relative_change"], 0)
            self.assertLess(other["top1_agreement"], 1)
            self.write_logits(changed, [[[float("inf"), 0, 0], [0, 0, 0]]], tokens)
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                quality.compare(token_path, reference, changed)


if __name__ == "__main__":
    unittest.main()
