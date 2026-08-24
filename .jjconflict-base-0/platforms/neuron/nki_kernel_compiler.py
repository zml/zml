"""Compile inline ZML NKI kernels into Neuron custom-native backend configs.

The `nki_kernel.zig` helper writes one JSON request per inline kernel.
This script asks NKI to produce BIR and returns the backend_config expected by
`AwsNeuronCustomNativeKernel`.
"""

import argparse
import inspect
import json
import re
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

import ml_dtypes
import numpy as np


@dataclass(frozen=True)
class TensorSpec:
    dtype: str
    shape: tuple[int, ...]


def dtype_from_name(name: str):
    mapping = {
        "bool": np.bool_,
        "i8": np.int8,
        "i16": np.int16,
        "i32": np.int32,
        "i64": np.int64,
        "u8": np.uint8,
        "u16": np.uint16,
        "u32": np.uint32,
        "u64": np.uint64,
        "f16": np.float16,
        "f32": np.float32,
        "f64": np.float64,
    }

    if name == "bf16":
        return ml_dtypes.bfloat16

    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def parse_tensor_specs(raw_specs: list[dict]) -> tuple[TensorSpec, ...]:
    return tuple(
        TensorSpec(
            dtype=spec["dtype"],
            shape=tuple(spec.get("shape", spec["dims"])),
        )
        for spec in raw_specs
    )


def load_kernel(source_path: Path, source: str, entrypoint: str):
    module_name = re.sub(r"\W", "_", source_path.stem) or "kernel"
    if module_name[0].isdigit():
        module_name = f"kernel_{module_name}"

    module = types.ModuleType(module_name)
    module.__file__ = str(source_path)
    module.__dict__["__builtins__"] = __builtins__
    sys.modules[module_name] = module
    exec(compile(source, str(source_path), "exec"), module.__dict__)

    try:
        return module.__dict__[entrypoint]
    except KeyError as exc:
        raise KeyError(f"entrypoint {entrypoint!r} not found in {source_path}") from exc


def make_inputs(inputs: tuple[TensorSpec, ...]) -> list[np.ndarray]:
    """Create zero-filled specialization inputs matching the declared ZML signature."""
    arrays: list[np.ndarray] = []
    for input_spec in inputs:
        arrays.append(
            np.zeros(
                shape=input_spec.shape,
                dtype=dtype_from_name(input_spec.dtype),
            )
        )
    return arrays


def compile_kernel(
    source: str,
    entrypoint: str,
    target: str,
    tool_bin_dir: str,
    neuronx_cc_args: tuple[str, ...],
    inputs: tuple[TensorSpec, ...],
    outputs: tuple[TensorSpec, ...],
) -> str:
    """Run the full embedded-kernel pipeline for one custom-call."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="zml-nki-", dir="."))

    source_path = tmp_dir / "kernel.py"
    artifacts_dir = tmp_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=False)

    source_path.write_text(source, encoding="utf-8")

    kernel = load_kernel(source_path, source, entrypoint)

    input_names = [
        parameter.name
        for parameter in inspect.signature(kernel).parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]

    if len(input_names) != len(inputs):
        raise ValueError(
            f"{entrypoint} expects {len(input_names)} inputs, ZML declared {len(inputs)}"
        )

    from nki.compiler.driver import compile_to_bir
    from nki.compiler.frontend import ParserFrontend
    from nki.compiler.ncc_driver import CompilationMode, CompileOptions

    compile_opts = CompileOptions(
        target=target,
        artifacts_dir=str(artifacts_dir),
        output_path=str(artifacts_dir / "unused.neff"),
        neuronx_cc_path=str(Path(tool_bin_dir) / "neuronx-cc"),
        neuronx_cc_args=neuronx_cc_args,
        mode=CompilationMode.INTEGRATION,
    ).disable_backend_optimizations()

    nir = compile_to_bir(
        kernel,
        frontend=ParserFrontend(),
        inputs=dict(zip(input_names, make_inputs(inputs), strict=True)),
        compile_opts=compile_opts,
        output_names=[f"output{i}" for i in range(len(outputs))],
    )

    return nir.build_config().backend_config_b64.decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile a ZML NKI kernel to a custom-native backend config"
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    request_path = Path(args.request)
    result_path = Path(args.result)

    request = json.loads(request_path.read_text(encoding="utf-8"))

    backend_config_b64 = compile_kernel(
        source=request["source"],
        entrypoint=request["entrypoint"],
        target=request["target"],
        tool_bin_dir=request["tool_bin_dir"],
        neuronx_cc_args=tuple(request["neuronx_cc_args"]),
        inputs=parse_tensor_specs(request["inputs"]),
        outputs=parse_tensor_specs(request["outputs"]),
    )

    result_path.write_text(
        json.dumps({"backend_config_b64": backend_config_b64}),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
