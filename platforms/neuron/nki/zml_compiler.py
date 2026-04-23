"""Compile inline ZML NKI kernels into Neuron custom-native backend configs."""

import argparse
import base64
import json
import os
import re
import sys
import tempfile
import types
from dataclasses import dataclass, replace
from pathlib import Path

import ml_dtypes
import numpy as np


@dataclass(frozen=True)
class TensorSpec:
    dtype: str
    shape: tuple[int, ...]
    name: str | None = None


@dataclass(frozen=True)
class KernelSignature:
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]


@dataclass(frozen=True)
class CompileRequest:
    entrypoint: str
    source: str
    target: str
    tool_bin_dir: str
    signature: KernelSignature


@dataclass(frozen=True)
class CompileResult:
    backend_config_b64: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]


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


def rewrite_source_entrypoint(
    source: str, entrypoint: str, input_names: list[str]
) -> tuple[str, list[str]]:
    """Rewrite the kernel entrypoint so compiler-visible argument names are stable."""
    sanitized_names: list[str] = []
    used_names: set[str] = set()
    for i, name in enumerate(input_names):
        candidate = re.sub(r"\W", "_", name)
        if not candidate or candidate[0].isdigit():
            candidate = f"arg_{i}"
        while candidate in used_names:
            candidate = f"{candidate}_{i}"
        used_names.add(candidate)
        sanitized_names.append(candidate)

    original_name = f"__zml_orig_{entrypoint}"
    pattern = rf"(?m)^def\s+{re.escape(entrypoint)}\s*\("
    rewritten, count = re.subn(pattern, f"def {original_name}(", source, count=1)
    if count != 1:
        raise ValueError(f"failed to rewrite entrypoint {entrypoint!r} in source")

    wrapper_args = ", ".join(sanitized_names)
    wrapper = f"\n\ndef {entrypoint}({wrapper_args}):\n    return {original_name}({wrapper_args})\n"
    return rewritten + wrapper, sanitized_names


def default_input_names(count: int) -> list[str]:
    return [f"input{i}" for i in range(count)]


def default_output_names(count: int) -> list[str]:
    return [f"output{i}" for i in range(count)]


def parse_signature(raw_signature: dict) -> KernelSignature:
    def parse_tensor_spec(raw_spec: dict) -> TensorSpec:
        return TensorSpec(
            dtype=raw_spec["dtype"],
            shape=tuple(raw_spec.get("shape", raw_spec["dims"])),
            name=raw_spec.get("name"),
        )

    return KernelSignature(
        inputs=tuple(parse_tensor_spec(spec) for spec in raw_signature["inputs"]),
        outputs=tuple(parse_tensor_spec(spec) for spec in raw_signature["outputs"]),
    )


def parse_request(raw_request: dict) -> CompileRequest:
    return CompileRequest(
        entrypoint=raw_request["entrypoint"],
        source=raw_request["source"],
        target=raw_request["target"],
        tool_bin_dir=raw_request["tool_bin_dir"],
        signature=parse_signature(
            {
                "inputs": raw_request["inputs"],
                "outputs": raw_request["outputs"],
            }
        ),
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


def make_inputs(signature: KernelSignature) -> list[np.ndarray]:
    """Create zero-filled specialization inputs matching the declared ZML signature."""
    inputs: list[np.ndarray] = []
    for input_spec in signature.inputs:
        inputs.append(
            np.zeros(
                shape=input_spec.shape,
                dtype=dtype_from_name(input_spec.dtype),
            )
        )
    return inputs


def build_backend_config(
    target: str, bir, result: object
) -> tuple[str, list[str], list[str]]:
    from nki.compiler.target import target_to_nc_version  # type: ignore

    aliases = result.input_output_aliases or {}
    aliased_input_names = set(aliases.values())
    input_name_to_idx = {spec.name: idx for idx, spec in enumerate(result.input_specs)}

    input_names: list[str] = []
    for spec in result.input_specs:
        if spec.name in aliased_input_names:
            input_names.append(f"{spec.name}.must_alias_input")
        else:
            input_names.append(spec.name)

    alias_pairs: list[list[int]] = []
    for output_idx, input_name in aliases.items():
        input_idx = input_name_to_idx.get(input_name)
        if input_idx is not None:
            alias_pairs.append([input_idx, output_idx])

    output_names: list[str] = []
    for idx, spec in enumerate(result.output_specs):
        if idx in aliases:
            output_names.append(aliases[idx])
        else:
            output_names.append(spec.name)

    backend_config = {
        "func_name": result.function_name,
        "platform_target": target_to_nc_version(target),
        "kernel_format": "bir",
        "kernel_version": 1,
        "klir_binary": {
            "binary": bir.kernel_json_path,
            "input_names": input_names,
            "output_names": output_names,
            "version_identifier": "",
            "aliases": alias_pairs,
        },
        "grid": 1,
        "has_collectives": result.has_collectives,
        "mac_count": result.mac_count,
    }
    return (
        base64.b64encode(json.dumps(backend_config).encode("utf-8")).decode("ascii"),
        input_names,
        output_names,
    )


def configure_python_path(tool_bin_dir: str) -> None:
    path = os.environ.get("PATH")
    os.environ["PATH"] = tool_bin_dir if not path else f"{tool_bin_dir}:{path}"


def rewrite_source_for_compilation(
    source: str,
    entrypoint: str,
    signature: KernelSignature,
) -> tuple[str, list[str]]:
    declared_input_names = [
        input_spec.name or default_input_names(len(signature.inputs))[i]
        for i, input_spec in enumerate(signature.inputs)
    ]
    return rewrite_source_entrypoint(source, entrypoint, declared_input_names)


def compile_to_bir_with_signature(
    kernel,
    signature: KernelSignature,
    input_names: list[str],
    target: str,
    artifacts_dir: Path,
):
    compile_inputs = make_inputs(signature)
    input_map = dict(zip(input_names, compile_inputs, strict=True))
    output_names = default_output_names(len(signature.outputs))

    from nki.compiler.driver import compile_to_bir  # type: ignore
    from nki.compiler.frontend import ParserFrontend  # type: ignore
    from nki.compiler.ncc_driver import CompilationMode, CompileOptions  # type: ignore

    compile_opts = CompileOptions(
        target=target,
        artifacts_dir=str(artifacts_dir),
        output_path=str(artifacts_dir / "unused.neff"),
    )
    compile_opts = replace(compile_opts, mode=CompilationMode.INTEGRATION)
    return compile_to_bir(
        kernel,
        frontend=ParserFrontend(),
        inputs=input_map,
        compile_opts=compile_opts,
        output_names=output_names,
    )


def compile_request(request: CompileRequest) -> CompileResult:
    configure_python_path(request.tool_bin_dir)

    # Leave the scratch directory under the outer Neuron sandbox. The backend
    # config points at compiler artifacts produced inside this directory.
    tmp_dir = Path(tempfile.mkdtemp(prefix="zml-nki-", dir="."))
    source_path = tmp_dir / "kernel.py"
    artifacts_dir = tmp_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=False)
    rewritten_source, input_names = rewrite_source_for_compilation(
        request.source,
        request.entrypoint,
        request.signature,
    )
    # Persist only the rewritten module. That keeps compiler diagnostics aligned
    # with the callable surface that compile_to_bir actually sees.
    source_path.write_text(rewritten_source, encoding="utf-8")
    kernel = load_kernel(source_path, rewritten_source, request.entrypoint)
    bir, result = compile_to_bir_with_signature(
        kernel,
        request.signature,
        input_names,
        request.target,
        artifacts_dir,
    )

    backend_config_b64, compiled_input_names, compiled_output_names = build_backend_config(
        request.target,
        bir,
        result,
    )
    return CompileResult(
        backend_config_b64=backend_config_b64,
        input_names=tuple(compiled_input_names),
        output_names=tuple(compiled_output_names),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile a ZML NKI kernel to a custom-native backend config"
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()

    request_path = Path(args.request)
    result_path = Path(args.result)
    request = parse_request(json.loads(request_path.read_text(encoding="utf-8")))
    result = compile_request(request)
    result_path.write_text(
        json.dumps(
            {
                "backend_config_b64": result.backend_config_b64,
                "input_names": list(result.input_names),
                "output_names": list(result.output_names),
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
