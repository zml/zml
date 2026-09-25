"""Serialization and numerical checks for the optional FP8 checkpoint utility."""

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import torch

import quantize_fp8


class QuantizeFp8Test(unittest.TestCase):
    def test_power_of_two_scaling_and_zero_rows(self):
        values = torch.tensor([[1.0, -2.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        encoded, scale, restored = quantize_fp8.quantize_rows(values)
        self.assertEqual(encoded.dtype, torch.float8_e4m3fn)
        self.assertTrue(torch.equal(scale, torch.tensor([1 / 128, 1.0])))
        self.assertTrue(torch.equal(restored, values))
        for invalid in (float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                quantize_fp8.quantize_rows(torch.tensor([[invalid]]))

    def test_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source", root / "converted"
            source.mkdir()
            values = torch.tensor([[1.0, -2.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.bfloat16)
            payload = values.view(torch.uint16).numpy().astype("<u2").tobytes()
            projection = "model.layers.0.self_attn.q_proj.weight"
            embedding = "model.embed_tokens.weight"
            header = {
                projection: {"dtype": "BF16", "shape": [2, 4], "data_offsets": [0, 16]},
                embedding: {"dtype": "BF16", "shape": [2, 4], "data_offsets": [16, 32]},
            }
            encoded_header = json.dumps(header).encode()
            shard = source / "model.safetensors"
            shard.write_bytes(struct.pack("<Q", len(encoded_header)) + encoded_header + payload * 2)
            source_hash = hashlib.sha256(shard.read_bytes()).digest()
            (source / "config.json").write_text('{"model_type":"llama"}')
            (source / "tokenizer.json").write_text("fixture tokenizer")
            (source / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {projection: shard.name, embedding: shard.name},
            }))
            quantize_fp8.convert(source, destination)
            self.assertEqual(hashlib.sha256(shard.read_bytes()).digest(), source_hash)
            out_header, start = quantize_fp8.read_header(destination / shard.name)
            data = (destination / shard.name).read_bytes()[start:]
            qmeta = out_header[projection]
            smeta = out_header[projection.removesuffix("weight") + "weight_scale"]
            emeta = out_header[embedding]
            self.assertEqual(qmeta["dtype"], "F8_E4M3")
            self.assertEqual(qmeta["shape"], [2, 4])
            self.assertEqual(data[slice(*emeta["data_offsets"])], payload)
            quantized = torch.from_numpy(np.frombuffer(data[slice(*qmeta["data_offsets"])], dtype=np.uint8).copy()).view(torch.float8_e4m3fn).reshape(2, 4)
            scale = torch.from_numpy(np.frombuffer(data[slice(*smeta["data_offsets"])], dtype="<f4").copy())
            self.assertTrue(torch.equal(quantized.float() * scale[:, None], values.float()))
            index = json.loads((destination / "model.safetensors.index.json").read_text())
            self.assertEqual(index["metadata"]["total_size"], len(data))
            self.assertEqual(set(index["weight_map"]), set(out_header) - {"__metadata__"})
            self.assertEqual((destination / "tokenizer.json").read_text(), "fixture tokenizer")
            self.assertEqual(json.loads((destination / "zml-fp8-report.json").read_text())["converted_matrices"], 1)
            with self.assertRaises(FileExistsError):
                quantize_fp8.convert(source, destination)


if __name__ == "__main__":
    unittest.main()
