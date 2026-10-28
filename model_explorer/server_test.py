"""Smoke tests for the server and its Bazel-packaged converters."""

from html.parser import HTMLParser
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.parse

import server


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()

    def convert(self, path, adapter):
        response = self.client.post("/apipost/v1/send_command", json={
            "extensionId": adapter,
            "cmdId": "convert",
            "modelPath": str(path),
            "settings": {},
            "deleteAfterConversion": False,
        })
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertNotIn("error", result)
        self.assertTrue(path.exists())
        return result["graphCollections"]

    def test_health_ui_and_mlir_link(self):
        self.assertEqual(self.client.get("/check_health").data, b"model_explorer_ok")
        with self.client.get("/") as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn(b"<html", response.data)
        url = urllib.parse.urlsplit(server.path_to_url("/tmp/model.mlir"))
        data = json.loads(urllib.parse.parse_qs(url.query)["data"][0])
        self.assertEqual(data["models"][0]["adapterId"], "builtin_mlir")

    def test_file_menu_links(self):
        class Links(HTMLParser):
            def __init__(self):
                super().__init__()
                self.urls = []

            def handle_starttag(self, tag, attrs):
                if tag == "a":
                    self.urls.append(dict(attrs)["href"])

        with tempfile.TemporaryDirectory() as directory:
            hlo = Path(directory) / 'model "quoted" & #1.hlo.pb'
            mlir = Path(directory) / "module.mlir"
            hlo.touch()
            mlir.touch()
            (Path(directory) / "old.dot").touch()
            with patch.object(server, "ROOT_DIR", directory):
                response = self.client.get("/list")
            self.assertEqual(response.status_code, 200)
            links = Links()
            links.feed(response.get_data(as_text=True))
            models = []
            for url in links.urls:
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
                data = json.loads(query["data"][0])
                self.assertNotIn("uiState", data)
                models.extend(data["models"])
            self.assertCountEqual(models, [
                {"url": str(hlo), "adapterId": "model_explorer_hlo"},
                {"url": str(mlir), "adapterId": "builtin_mlir"},
            ])
            extensions = self.client.get("/api/v1/get_extensions").get_json()
            hlo_extension = next(ext for ext in extensions if ext["id"] == "model_explorer_hlo")
            self.assertIn("pb", hlo_extension["fileExts"])

    def test_hlo_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.hlo.pb"
            # HloProto { hlo_module { name: "smoke" } } in protobuf wire format.
            path.write_bytes(b"\x0a\x07\x0a\x05smoke")
            collections = self.convert(path, "model_explorer_hlo")
            self.assertEqual(len(collections), 1)
            self.assertEqual(collections[0]["label"], "smoke")
            self.assertEqual(collections[0]["graphs"], [])

    def test_optional_dump_context(self):
        def varint(value):
            data = bytearray()
            while value > 127:
                data.append((value & 127) | 128)
                value >>= 7
            data.append(value)
            return bytes(data)

        def field(number, value):
            if isinstance(value, int):
                return varint(number << 3) + varint(value)
            if isinstance(value, str):
                value = value.encode()
            return varint((number << 3) | 2) + varint(len(value)) + value

        # Minimal HLO with one scalar F32 all-reduce, channel 3.
        instruction = field(1, "reduce") + field(2, "all-reduce") + field(3, field(2, 11)) + field(26, 3)
        computation = field(1, "entry") + field(2, instruction)
        proto = field(1, field(1, "context") + field(3, computation))
        with tempfile.TemporaryDirectory() as directory:
            stem = Path(directory) / "module_0001.context"
            path = Path(str(stem) + ".after_spmd_partitioner.hlo.pb")
            path.write_bytes(proto)

            def attrs():
                graph = self.convert(path, "model_explorer_hlo")[0]["graphs"][0]
                return {a["key"]: a["value"] for a in graph["nodes"][0]["attrs"]}

            self.assertEqual(attrs()["device connected"], "? / ?")
            stem.mkdir()
            gpu = field(3, "test GPU") + field(4, 1) + field(5, 1) + field(6, 2)
            topology = field(9, field(1, "type.googleapis.com/xla.GpuTopologyProto") + field(2, gpu))
            (stem / "topology.pb").write_bytes(topology)
            final = Path(str(stem) + ".sm_8.9_gpu_after_optimizations.hlo.pb")
            final.write_bytes(proto)
            enriched = attrs()
            self.assertEqual(enriched["device connected"], "? / 2")
            self.assertNotIn("optimized collective", enriched)
            self.assertEqual(enriched["bytes/device"], "out: 4B, in: ?")
            (stem / "topology.pb").write_bytes(b"\xff")
            self.assertEqual(attrs()["device connected"], "? / ?")
            Path(str(stem) + ".sm_9.0_gpu_after_optimizations.hlo.pb").write_bytes(proto)
            self.assertEqual(attrs()["bytes/device"], "out: 4B, in: ?")

    def test_mlir_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.mlir"
            path.write_text(
                'module { sdy.mesh @mesh = <["x"=1]> '
                'func.func @main(%arg0: tensor<2xf32>) -> tensor<2xf32> '
                '{ return %arg0 : tensor<2xf32> } }'
            )
            self.assertTrue(self.convert(path, "builtin_mlir")[0]["graphs"])


if __name__ == "__main__":
    unittest.main()
