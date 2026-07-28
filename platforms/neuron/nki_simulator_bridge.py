"""Embedded-Python bridge between XLA FFI buffers and ``nki.simulate``."""

import hashlib
import importlib
import os
import sys
import types

import ml_dtypes
import nki
import nki.language as nl
import numpy as np


_DTYPES = {
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
    "bf16": ml_dtypes.bfloat16,
}
_KERNELS = {}
_REDUCE_COMPAT_INSTALLED = False


def _install_reduce_compat():
    """Teach NKI 0.5's simulator the nl.max used by Trn3 nkilib."""
    global _REDUCE_COMPAT_INSTALLED
    if _REDUCE_COMPAT_INSTALLED:
        return
    language_ops = importlib.import_module("nki.language._ops")
    original_get_numpy_reduce_op = language_ops.get_numpy_reduce_op

    def get_numpy_reduce_op(op):
        if op is nl.max:
            return np.max
        return original_get_numpy_reduce_op(op)

    language_ops.get_numpy_reduce_op = get_numpy_reduce_op
    _REDUCE_COMPAT_INSTALLED = True


def _load_kernel(source, entrypoint, target):
    digest = hashlib.sha256(
        (target + "\0" + entrypoint + "\0" + source).encode("utf-8")
    ).hexdigest()
    cache_key = (digest, entrypoint, target)
    if cache_key in _KERNELS:
        return _KERNELS[cache_key]

    module_name = f"_zml_nki_simulator_{digest}"
    module = types.ModuleType(module_name)
    module.__file__ = f"<{module_name}.py>"
    module.__dict__["__builtins__"] = __builtins__
    sys.modules[module_name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    kernel = module.__dict__[entrypoint]
    _KERNELS[cache_key] = kernel
    return kernel


def _array(descriptor):
    buffer, dtype_name, shape = descriptor
    try:
        dtype = _DTYPES[dtype_name]
    except KeyError as exc:
        raise ValueError(f"unsupported NKI simulator dtype: {dtype_name}") from exc
    return np.frombuffer(buffer, dtype=dtype).reshape(shape)


def run(source, entrypoint, target, grid, input_descriptors, output_descriptors):
    """Run one NKI launch directly over CPU PJRT input/output buffers."""
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = target
    os.environ.setdefault("NKI_PRECISE_FP", "1")
    if target == "trn3":
        _install_reduce_compat()

    kernel = _load_kernel(source, entrypoint, target)
    launch = kernel[grid] if grid > 0 else kernel
    inputs = [_array(descriptor) for descriptor in input_descriptors]
    outputs = [_array(descriptor) for descriptor in output_descriptors]
    result = nki.simulate(launch)(*inputs)
    results = result if isinstance(result, tuple) else (result,)
    if len(results) != len(outputs):
        raise ValueError(
            f"{entrypoint} returned {len(results)} outputs; XLA expects {len(outputs)}"
        )
    for destination, value in zip(outputs, results, strict=True):
        np.copyto(destination, np.asarray(value, dtype=destination.dtype))
