import base64
import argparse
from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import re
import sys
import types

import numpy as np

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None

try:
    import nki  # type: ignore
except ImportError:
    from neuronxcc import nki  # type: ignore


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
        if ml_dtypes is None:
            raise ValueError("bf16 requires ml_dtypes")
        return ml_dtypes.bfloat16

    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def rewrite_source_entrypoint(source: str, entrypoint: str, input_names: list[str]) -> tuple[str, list[str]]:
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


def write_io_names(io_names_output: Path, input_names: list[str], output_names: list[str]) -> None:
    with io_names_output.open("w", encoding="utf-8") as f:
        for i, name in enumerate(input_names):
            f.write(f"input{i}={name}\n")
        for i, name in enumerate(output_names):
            f.write(f"output{i}={name}\n")


def make_inputs(signature: dict) -> list[np.ndarray]:
    inputs: list[np.ndarray] = []
    for input_spec in signature["inputs"]:
        inputs.append(
            np.zeros(
                shape=tuple(input_spec["shape"]),
                dtype=dtype_from_name(input_spec["dtype"]),
            )
        )
    return inputs


def build_backend_config(target: str, bir, result: object) -> tuple[bytes, list[str], list[str]]:
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
    return base64.b64encode(json.dumps(backend_config).encode("utf-8")), input_names, output_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile a ZML NKI kernel to a custom-native backend config")
    parser.add_argument("--source", required=True)
    parser.add_argument("--entrypoint", required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--backend-config-output", required=True)
    parser.add_argument("--io-names-output", required=True)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--tool-bin-dir", required=True)
    parser.add_argument("--target", default="")
    args = parser.parse_args()

    if args.target:
        os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = args.target

    path = os.environ.get("PATH")
    os.environ["PATH"] = args.tool_bin_dir if not path else f"{args.tool_bin_dir}:{path}"

    source_path = Path(args.source)
    signature_path = Path(args.signature)
    output_path = Path(args.backend_config_output)
    io_names_output = Path(args.io_names_output)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=False)

    signature = json.loads(signature_path.read_text(encoding="utf-8"))
    source = source_path.read_text(encoding="utf-8")

    declared_input_names = [
        input_spec.get("name") or default_input_names(len(signature["inputs"]))[i]
        for i, input_spec in enumerate(signature["inputs"])
    ]
    source, input_names = rewrite_source_entrypoint(source, args.entrypoint, declared_input_names)
    source_path.write_text(source, encoding="utf-8")

    kernel = load_kernel(source_path, source, args.entrypoint)
    inputs = make_inputs(signature)
    input_map = dict(zip(input_names, inputs, strict=True))
    output_names = default_output_names(len(signature["outputs"]))

    from nki.compiler.driver import compile_to_bir  # type: ignore
    from nki.compiler.frontend import ParserFrontend  # type: ignore
    from nki.compiler.ncc_driver import CompilationMode, CompileOptions  # type: ignore

    compile_opts = CompileOptions(
        target=args.target or "trn1",
        artifacts_dir=str(artifacts_dir),
        output_path=str(artifacts_dir / "unused.neff"),
    )
    compile_opts = replace(compile_opts, mode=CompilationMode.INTEGRATION)
    bir, result = compile_to_bir(
        kernel,
        frontend=ParserFrontend(),
        inputs=input_map,
        compile_opts=compile_opts,
        output_names=output_names,
    )
    backend_config, compiled_input_names, compiled_output_names = build_backend_config(
        args.target or "trn1",
        bir,
        result,
    )
    output_path.write_bytes(backend_config)
    write_io_names(io_names_output, compiled_input_names, compiled_output_names)


if __name__ == "__main__":
    main()
