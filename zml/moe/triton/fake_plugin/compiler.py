from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict

from triton.backends.compiler import BaseBackend, GPUTarget, Language


@dataclass(frozen=True)
class FakeOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 2
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    extern_libs: tuple = ()
    sanitize_overflow: bool = True
    debug: bool = False
    instrumentation_mode: str = ""
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: tuple = ("tf32", "tf32x3", "ieee", "bf16x3", "bf16x6")
    max_num_imprecise_acc_default: int = 0
    backend_name: str = "cpu"

    def hash(self) -> str:
        return "fake-options-v1"


class Backend(BaseBackend):
    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "fakebin"

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "cpu"

    def hash(self) -> str:
        return "fake-backend-v1"

    def parse_options(self, options: dict) -> object:
        args = {
            k: options[k] for k in FakeOptions.__dataclass_fields__.keys() if k in options and options[k] is not None
        }
        return FakeOptions(**args)

    def add_stages(self, stages: dict, options: object, language: Language = Language.TRITON) -> None:
        del options, language
        stages["ttir"] = lambda src, metadata: self._keep_ttir(src, metadata)
        stages[self.binary_ext] = lambda src, metadata: self._emit_fake_binary(src, metadata)

    def _keep_ttir(self, src, metadata):
        metadata.setdefault("shared", 0)
        metadata.setdefault("name", "fake_kernel")
        return src

    def _emit_fake_binary(self, src, metadata) -> bytes:
        del src
        metadata.setdefault("shared", 0)
        metadata.setdefault("name", "fake_kernel")
        return b"fake-binary"

    def pack_metadata(self, metadata):
        return (
            getattr(metadata, "num_warps", 1),
            getattr(metadata, "num_ctas", 1),
            getattr(metadata, "shared", 0),
        )

    def get_codegen_implementation(self, options: Any):
        del options
        return {"min_dot_size": lambda lhs_type, rhs_type: (1, 1, 1)}

    def load_dialects(self, context):
        del context

    def get_module_map(self) -> Dict[str, ModuleType]:
        return {}

    @staticmethod
    def parse_attr(desc):
        assert isinstance(desc, str)
        ret = []
        if "D" in desc:
            ret += [["tt.divisibility", 16]]
        return ret

    @staticmethod
    def get_int_specialization(arg, **kwargs):
        if arg % 16 == 0 and kwargs.get("align", False):
            return "D"
        return ""

    @staticmethod
    def get_tensor_specialization(arg, **kwargs):
        if arg.data_ptr() % 16 == 0 and kwargs.get("align", False):
            return "D"
        return ""
