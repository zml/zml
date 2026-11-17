"""Model Explorer adapter backed by the Bazel-built HLO converter."""

import json
from pathlib import Path
import subprocess
import tempfile

from model_explorer.adapter import Adapter, AdapterMetadata
from model_explorer.graph_builder import GraphCollection
from python.runfiles import runfiles


class HloAdapter(Adapter):
    metadata = AdapterMetadata(
        id="model_explorer_hlo",
        name="XLA HLO",
        description="Convert an XLA HloProto to Model Explorer graphs.",
        fileExts=["pb"],
    )

    def convert(self, model_path, settings):
        binary = runfiles.Create().Rlocation("zml/model_explorer/hlo2json")
        if not binary or not Path(binary).is_file():
            raise FileNotFoundError("hlo2json executable is missing from Bazel runfiles")
        with tempfile.TemporaryDirectory(prefix="hlo2json-") as directory:
            output = Path(directory) / "graphs.json"
            result = subprocess.run(
                [binary, str(Path(model_path).resolve()), str(output)],
                capture_output=True,
                text=True,
            )
            if result.returncode:
                raise RuntimeError(f"HLO conversion failed: {result.stderr.strip()}")
            return {"graphCollections": [GraphCollection(**json.loads(output.read_text()))]}
