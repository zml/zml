import argparse
import json

import triton
import triton.backends as triton_backends
import triton.language as tl
from triton.backends import Backend as BackendRegistration
from triton.runtime.driver import driver as runtime_driver

from fake_plugin.compiler import Backend as FakeCompilerBackend
from fake_plugin.driver import Driver as FakeDriver
from triton_kernels.moe import (
    fused_moe_kernel,
    moe_align_block_size_kernel,
    count_and_sort_expert_tokens_kernel,
)


class FakeTensor:
    def __init__(self, dtype: str, shape, strides=None):
        self.dtype = dtype
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


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


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


def register_fake_backend() -> None:
    triton_backends.backends["mybackend_runtime"] = BackendRegistration(
        compiler=FakeCompilerBackend,
        driver=FakeDriver,
    )
    runtime_driver.set_active(FakeDriver())


def compile_fused_moe_kernel(cfg: dict) -> str:
    num_tokens = cfg["num_tokens"]
    top_k = cfg["top_k"]
    num_experts = cfg["num_experts"]
    out_features = cfg["out_features"]
    in_features = cfg["in_features"]
    max_num_tokens_padded = cfg["max_num_tokens_padded"]
    num_valid_tokens = cfg["num_valid_tokens"]

    block_size_m = cfg["block_size_m"]
    block_size_n = cfg["block_size_n"]
    block_size_k = cfg["block_size_k"]
    group_size_m = cfg["group_size_m"]
    split_k = cfg["split_k"]
    group_n = cfg["group_n"]
    group_k = cfg["group_k"]
    naive_block_assignment = cfg["naive_block_assignment"]
    mul_routed_weight = cfg["mul_routed_weight"]
    compute_type = triton_dtype(cfg["compute_type"])
    use_fp8_w8a8 = cfg["use_fp8_w8a8"]
    use_int8_w8a8 = cfg["use_int8_w8a8"]
    use_int8_w8a16 = cfg["use_int8_w8a16"]
    per_channel_quant = cfg["per_channel_quant"]
    has_bias = cfg["has_bias"]
    num_warps = cfg["num_warps"]
    num_stages = cfg["num_stages"]

    a_shape = (num_tokens, in_features)
    b_shape = (num_experts, out_features, in_features)
    c_shape = (num_tokens, top_k, out_features)

    if group_k > 0:
        a_scale_shape = (num_tokens, ceil_div(in_features, group_k))
    else:
        a_scale_shape = (num_tokens,)

    if group_k > 0 and group_n > 0:
        b_scale_shape = (num_experts, ceil_div(in_features, group_k), ceil_div(out_features, group_n))
    else:
        b_scale_shape = (num_experts, out_features)

    kwargs = {
        "a_ptr": FakeTensor(cfg["a_dtype"], a_shape),
        "b_ptr": FakeTensor(cfg["b_dtype"], b_shape),
        "c_ptr": FakeTensor(cfg["c_dtype"], c_shape),
        "b_bias_ptr": FakeTensor(cfg.get("b_bias_dtype", "bf16"), (num_experts, out_features)),
        "a_scale_ptr": FakeTensor(cfg.get("a_scale_dtype", "fp32"), a_scale_shape),
        "b_scale_ptr": FakeTensor(cfg.get("b_scale_dtype", "fp32"), b_scale_shape),
        "topk_weights_ptr": FakeTensor(cfg.get("topk_weights_dtype", "fp32"), (num_tokens, top_k)),
        "sorted_token_ids_ptr": FakeTensor("i32", (max_num_tokens_padded,)),
        "expert_ids_ptr": FakeTensor("i32", (ceil_div(max_num_tokens_padded, block_size_m),)),
        "num_tokens_post_padded_ptr": FakeTensor("i32", (1,)),
        "N_ptr": FakeTensor("i64", (1,)),
        "K_ptr": FakeTensor("i64", (1,)),
        "EM_ptr": FakeTensor("i64", (1,)),
        "num_valid_tokens_ptr": FakeTensor("i64", (1,)),
        "stride_am_ptr": FakeTensor("i64", (1,)),
        "stride_ak": 1,
        "stride_be_ptr": FakeTensor("i64", (1,)),
        "stride_bk": 1,
        "stride_bn_ptr": FakeTensor("i64", (1,)),
        "stride_cm_ptr": FakeTensor("i64", (1,)),
        "stride_cn": 1,
        "stride_asm_ptr": FakeTensor("i64", (1,)),
        "stride_ask_ptr": FakeTensor("i64", (1,)),
        "stride_bse_ptr": FakeTensor("i64", (1,)),
        "stride_bsk_ptr": FakeTensor("i64", (1,)),
        "stride_bsn_ptr": FakeTensor("i64", (1,)),
        "stride_bbe_ptr": FakeTensor("i64", (1,)),
        "stride_bbn_ptr": FakeTensor("i64", (1,)),
        "group_n": group_n,
        "group_k": group_k,
        "naive_block_assignment": naive_block_assignment,
        "BLOCK_SIZE_M": block_size_m,
        "BLOCK_SIZE_N": block_size_n,
        "BLOCK_SIZE_K": block_size_k,
        "GROUP_SIZE_M": group_size_m,
        "SPLIT_K": split_k,
        "MUL_ROUTED_WEIGHT": mul_routed_weight,
        "top_k": top_k,
        "compute_type": compute_type,
        "use_fp8_w8a8": use_fp8_w8a8,
        "use_int8_w8a8": use_int8_w8a8,
        "use_int8_w8a16": use_int8_w8a16,
        "per_channel_quant": per_channel_quant,
        "HAS_BIAS": has_bias,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "out_ptr": FakeTensor(cfg["c_dtype"], c_shape),
    }

    em_effective = max_num_tokens_padded
    if num_tokens < block_size_m:
        em_effective = min(max_num_tokens_padded, num_valid_tokens * block_size_m)
    grid_x = ceil_div(em_effective, block_size_m) * ceil_div(out_features, block_size_n)

    kernel = fused_moe_kernel.warmup(grid=(grid_x,), **kwargs)
    return kernel.asm["ttir"]


def compile_align_block_size_kernel(cfg: dict) -> str:
    kernel_name = cfg["kernel_name"]
    numel = cfg["numel"]
    num_experts = cfg["num_experts"]
    padded_num_experts = cfg["padded_num_experts"]
    max_num_tokens_padded = cfg["max_num_tokens_padded"]
    max_num_m_blocks = cfg["max_num_m_blocks"]
    block_size_m = cfg["block_size_m"]
    experts_per_warp = cfg["experts_per_warp"]
    hist_block = cfg["hist_block"]
    sort_block_size = cfg["sort_block_size"]
    sort_grid_x = cfg["sort_grid_x"]

    topk_ids = FakeTensor("i32", (numel,))
    cumsums = FakeTensor("i32", (num_experts + 1,))
    expert_ids = FakeTensor("i32", (max_num_m_blocks,))
    sorted_ids = FakeTensor("i32", (max_num_tokens_padded,))
    num_tokens_post_pad = FakeTensor("i32", (1,))

    if kernel_name == "moe_align_block_size_kernel":
        kernel = moe_align_block_size_kernel.warmup(
            grid=(2,),
            topk_ids_ptr=topk_ids,
            sorted_token_ids_ptr=sorted_ids,
            expert_ids_ptr=expert_ids,
            num_tokens_post_pad_ptr=num_tokens_post_pad,
            cumsum_ptr=cumsums,
            BLOCK_SIZE_M=block_size_m,
            NUMEL=numel,
            NUM_EXPERTS=num_experts,
            PADDED_NUM_EXPERTS=padded_num_experts,
            MAX_NUM_TOKENS_PADDED=max_num_tokens_padded,
            MAX_NUM_M_BLOCKS=max_num_m_blocks,
            HIST_BLOCK=hist_block,
            num_warps=8,
            num_stages=1,
            out0_ptr=sorted_ids,
            out1_ptr=expert_ids,
            out2_ptr=num_tokens_post_pad,
            out3_ptr=cumsums,
        )
    elif kernel_name == "count_and_sort_expert_tokens_kernel":
        kernel = count_and_sort_expert_tokens_kernel.warmup(
            grid=(sort_grid_x,),
            topk_ids_ptr=topk_ids,
            sorted_token_ids_ptr=sorted_ids,
            cumsum_ptr=cumsums,
            BLOCK_SIZE=sort_block_size,
            NUMEL=numel,
            NUM_EXPERTS=num_experts,
            num_warps=4,
            num_stages=1,
            out0_ptr=sorted_ids,
            out1_ptr=cumsums,
        )
    else:
        raise ValueError(f"Unsupported align kernel name: {kernel_name}")

    return kernel.asm["ttir"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TTIR for wrapped MoE Triton kernels")
    parser.add_argument("--config", required=True, help="Raw JSON string")
    args = parser.parse_args()

    cfg = json.loads(args.config)

    register_fake_backend()
    if cfg.get("kernel_family") == "align_block_size":
        ttir = compile_align_block_size_kernel(cfg)
    else:
        ttir = compile_fused_moe_kernel(cfg)
    print(ttir)


if __name__ == "__main__":
    main()
