from triton.backends.compiler import GPUTarget
from triton.backends.driver import Benchmarker, DriverBase


_TYPE_MAP = {
    "i1": "int8_t",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u1": "uint8_t",
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
    "fp16": "double",
    "bf16": "double",
    "fp32": "double",
    "f32": "double",
    "fp64": "double",
}


class Driver(DriverBase):
    @classmethod
    def is_active(cls):
        return True

    def map_python_to_cpp_type(self, ty: str) -> str:
        if ty.startswith("*"):
            return "void*"
        return _TYPE_MAP[ty]

    def get_current_target(self):
        return GPUTarget("cpu", "generic", 32)

    def get_current_device(self):
        return "cpu"

    def get_current_stream(self, device):
        del device
        return None

    def get_active_torch_device(self):
        return "cpu"

    def get_benchmarker(self) -> Benchmarker:
        def _benchmarker(kernel_call, *, quantiles, **kwargs):
            del kernel_call, kwargs
            return [0.0 for _ in quantiles]

        return _benchmarker

    def __init__(self) -> None:
        pass
