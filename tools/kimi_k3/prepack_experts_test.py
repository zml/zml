import json
import os
from pathlib import Path
import tempfile
import unittest

import prepack_experts as prepack


class FakeSource:
    def __init__(self, expert_count: int):
        self.descriptors = {}
        payload = bytearray()
        for expert in sorted(range(expert_count), key=lambda value: str(value)):
            for component_index, (projection, component) in enumerate(
                prepack.COMPONENTS
            ):
                name = prepack.source_name(
                    1, expert, projection, component
                )
                offset = len(payload)
                payload.append((expert * 10 + component_index) & 0xFF)
                self.descriptors[name] = {
                    "path": Path("synthetic.safetensors"),
                    "shard": "synthetic.safetensors",
                    "offset": offset,
                    "nbytes": 1,
                    "shape": [1],
                    "dtype": "U8",
                }
        self.payload = bytes(payload)

    def tensor(self, name):
        return self.descriptors[name]

    def read_extent(self, first, size):
        return self.payload[first["offset"] : first["offset"] + size]


class PrepackExpertsTest(unittest.TestCase):
    def test_header_round_trip(self):
        tensors = {
            "x": {
                "dtype": "U8",
                "shape": [3],
                "data_offsets": [0, 3],
            }
        }
        encoded = prepack.encode_safetensors_header(tensors)
        self.assertEqual(len(encoded) % 8, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.safetensors"
            path.write_bytes(encoded + b"abc")
            decoded, data_start = prepack.read_safetensors_header(path)
            self.assertEqual(decoded, tensors)
            self.assertEqual(data_start, len(encoded))

    def test_physical_lexicographic_read_maps_to_global_ids(self):
        expert_count = 11
        source = FakeSource(expert_count)
        tensors = {}
        cursor = 0
        for projection, component in prepack.COMPONENTS:
            name = prepack.cache_name(1, 0, projection, component)
            tensors[name] = {
                "dtype": "U8",
                "shape": [expert_count, 1],
                "data_offsets": [cursor, cursor + expert_count],
            }
            cursor += expert_count
        layout = {
            "part": 0,
            "first_expert": 0,
            "end_expert": expert_count,
            "tensors": tensors,
            "data_start": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packed.bin"
            path.write_bytes(bytes(cursor))
            fd = os.open(path, os.O_RDWR)
            try:
                transferred = prepack.pack_unit(source, fd, layout, 1)
            finally:
                os.close(fd)
            self.assertEqual(transferred, len(source.payload))
            packed = path.read_bytes()
        for component_index in range(len(prepack.COMPONENTS)):
            begin = component_index * expert_count
            expected = bytes(
                (expert * 10 + component_index) & 0xFF
                for expert in range(expert_count)
            )
            self.assertEqual(packed[begin : begin + expert_count], expected)

    def test_partial_journal_resumes_and_rejects_conflict(self):
        fingerprint = {
            "source_index_sha256": "c" * 64,
            "source_config_sha256": "d" * 64,
        }
        layouts = [
            {
                "part": part,
                "filename": f"part-{part}.safetensors",
                "header": b"header",
                "data_start": 6,
                "payload_bytes": 2,
            }
            for part in range(prepack.PARTS)
        ]
        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "partial"
            journal = prepack.initialize_partial(
                partial,
                layouts,
                {},
                fingerprint,
            )
            journal["completed"] = ["1:0"]
            prepack.atomic_json(partial / "journal.json", journal)
            resumed = prepack.load_or_initialize(
                Path(directory),
                partial,
                layouts,
                {},
                fingerprint,
            )
            self.assertEqual(resumed["completed"], ["1:0"])
            conflicting = dict(fingerprint)
            conflicting["source_config_sha256"] = "e" * 64
            with self.assertRaisesRegex(
                ValueError,
                "partial cache fingerprint mismatch",
            ):
                prepack.load_or_initialize(
                    Path(directory),
                    partial,
                    layouts,
                    {},
                    conflicting,
                )

    def test_valid_cache_manifest_and_invalid_size(self):
        fingerprint = {
            "source_index_sha256": "a" * 64,
            "source_config_sha256": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            files = []
            for part in range(prepack.PARTS):
                name = f"experts-{part:05d}-of-{prepack.PARTS:05d}.safetensors"
                path = cache / name
                path.write_bytes(b"x")
                files.append(
                    {
                        "name": name,
                        "first_expert": part * prepack.EXPERTS_PER_PART,
                        "end_expert": (part + 1) * prepack.EXPERTS_PER_PART,
                        "payload_bytes": prepack.EXPECTED_PART_PAYLOAD,
                        "file_size": 1,
                        "sha256": prepack.sha256_file(path),
                    }
                )
            manifest = {
                "schema_version": prepack.SCHEMA_VERSION,
                "expert_count": prepack.EXPERTS,
                "canonical_parts": prepack.PARTS,
                "experts_per_part": prepack.EXPERTS_PER_PART,
                "first_layer": prepack.FIRST_LAYER,
                "end_layer": prepack.END_LAYER,
                "expert_payload_bytes": prepack.EXPECTED_PAYLOAD,
                **fingerprint,
                "tensor_shapes": prepack.tensor_shape_manifest(),
                "files": files,
            }
            prepack.atomic_json(cache / "manifest.json", manifest)
            self.assertTrue(
                prepack.validate_completed_cache(cache, fingerprint)
            )
            (cache / files[0]["name"]).write_bytes(b"y")
            self.assertFalse(
                prepack.validate_completed_cache(cache, fingerprint)
            )

    def test_manifest_records_canonical_tensor_shapes(self):
        shapes = prepack.tensor_shape_manifest()
        self.assertEqual(len(shapes), len(prepack.COMPONENTS))
        self.assertEqual(shapes[0]["shape"], [224, 3072, 1792])
        self.assertEqual(shapes[3]["shape"], [224, 3584, 96])
        self.assertTrue(all(entry["dtype"] == "U8" for entry in shapes))


if __name__ == "__main__":
    unittest.main()
