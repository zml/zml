import triton.language as tl

class FakeTensor:
    def __init__(self, dtype: str, shape, strides=None):
        self.dtype = "fp8e4nv" if dtype == "f8e4m3fn" else dtype
        self.shape = tuple(shape)
        self._strides = tuple(strides) if strides is not None else contiguous_strides(self.shape)

    def stride(self, dim: int) -> int:
        return self._strides[dim]

    @staticmethod
    def data_ptr() -> int:
        return 0


def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    out = []
    for size in reversed(shape):
        out.append(stride)
        stride *= int(size)
    return tuple(reversed(out))

def triton_dtype(dtype: str):
    mapping = {
        "bf16": tl.bfloat16,
        "f16": tl.float16,
        "f32": tl.float32,
        "i8": tl.int8,
        "i16": tl.int16,
        "i32": tl.int32,
        "i64": tl.int64,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported Triton compute dtype: {dtype}")
    return mapping[dtype]
