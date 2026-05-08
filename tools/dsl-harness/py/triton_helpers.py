"""Shared helpers for kernel py-references: a `FakeTensor` shim that
satisfies `JITFunction.warmup`'s duck-typed contract, and a
`dtype_str(name, default)` mapping harness dtype tags to Triton's form."""

from __future__ import annotations

from typing import Any, Tuple


_DTYPE_MAP: dict[str, str] = {
    "bf16": "bf16",
    "f16": "fp16",
    "f32": "fp32",
    "f8e4m3fn": "fp8e4nv",
    "f8e5m2": "fp8e5",
    "i8": "i8",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
}


def dtype_str(name: Any, default: str) -> str:
    if isinstance(name, str):
        return _DTYPE_MAP.get(name, default)
    return default


class FakeTensor:
    def __init__(self, dtype: str, shape: Tuple[int, ...]) -> None:
        self.dtype = dtype
        self.shape = tuple(shape)
        # Contiguous strides, last dim fastest.
        stride = 1
        out = []
        for size in reversed(self.shape):
            out.append(stride)
            stride *= int(size)
        self._strides = tuple(reversed(out))

    def stride(self, dim: int) -> int:
        return self._strides[dim]

    @staticmethod
    def data_ptr() -> int:
        return 0


def fake(dtype: str, *shape: int) -> FakeTensor:
    return FakeTensor(dtype, shape)
